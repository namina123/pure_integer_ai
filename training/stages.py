"""training.stages — 五阶段编排（§十二最优训练路径 + G5/C6 harness 真接线）。

  STAGE1_SKELETON     结构骨架建图（冷启动·observe only·教师元定义补 PRIMARY）
  STAGE2_CAUSES_ABS   因果+抽象并行建图（observe only·CAUSES 只结构化源+断奶前 LLM 教师③）
  STAGE3_REWARD       reward 闭环开启（judge+反传·H2 小批量标定权重再开全量）
  STAGE4_PROMOTE_WEAN promote 收敛+断奶（三重 SHADOW→PRIMARY·双曲线趋势 D1·LLM 退场）
  STAGE5_MULTIMODAL   多模态接口预留（defer·非训练·机制骨架最小闭环随模态扩展）

  stage_gate_config(stage) -> StageGateConfig   每阶段 gate 配比（observe only / reward off / promote off）
  stage_metric_gate(stage, metrics) -> bool     度量门控（图规模/CAUSES 覆盖率/reward 导通率/promote 率达标才进下阶段）
  is_skippable(stage) -> bool                   E8 stage-skip（纯 observe 累积 skippable·分布标定 non-skippable）
  build_judge_fn(graph, weights, teacher) -> JudgeFn   G5/C6 harness 真接线（teacher.judge_ground_truth → self_proof_fn·weaning pre）

**G5/C6 harness 真接线（卷三 Stage 5 占位 → Stage 6 落地）**：
  judge self_proof_fn 在卷三是 None（pass=1 占位）·Stage 6 录放层落真接线——
  build_judge_fn 把 teacher.judge_ground_truth 绑成 self_proof_fn·断奶前（WEANING_PRE）注入 judge。
  断奶后（WEANING_POST）teacher 退场·self_proof_fn=None（Mode B minimal=结构门自洽+反馈环 achieved·G5 vacate pass=1·
    深化 C6 re-derivation correctness 自评 D 墙+VM 接线 defer·见 doc/重来_ModeB自洽设计补充.md）。
  TEACHER_MODE OFF → build_judge_fn 返 self_proof_fn=None（bit-identical 占位·不产 vacuous reward）。

**能力是地基涌现非单独训练**（§十二核心判断）：最优路径=建地基顺序（结构骨架→因果+抽象→reward→promote 断奶）·
  不是建能力的顺序。observe 先 reward 后破死锁（阶段1-2 observe only·阶段3 才开 reward）。

铁律：纯整数（度量×1000·阈值 oracle 标）/ 不写死（阶段配比涌现自§十二非硬编码语义规则·gate 二分 live-read）/
  外部只启发（教师经录放层只标定权重非定义判据·断奶后退场）/ 不走外挂 LLM（REPLAY 零 LLM·MODE_RECORD 离线）/ 几百G不重训
  （每阶段度量门控合格才进下·每正式 run 新 run_id）。
诚实边界：阶段度量门控是经验阈值非理论保证（oracle 标）/ 能力涌现非训练保证（地基建好≠能力必现·D 墙）/
  G5/C6 Mode A 教师 ground-truth 是断奶前对照·断奶后自评无外部标准（D 墙·Mode B defer）。
"""
from __future__ import annotations

import contextvars
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from pure_integer_ai.config import gates
from pure_integer_ai.cognition.shared.types import (
    JudgeWeights, WEANING_PRE, WEANING_POST,
    DOMAIN_CODE, DOMAIN_MATH,
)
from pure_integer_ai.cognition.result.judge import judge, SelfProofFn
from pure_integer_ai.teacher.source_independence import sources_disjoint

# ---- 五阶段（§十二最优训练路径） ----
STAGE1_SKELETON = 1      # 结构骨架建图（冷启动·observe only）
STAGE2_CAUSES_ABS = 2    # 因果+抽象并行（observe only）
STAGE3_REWARD = 3        # reward 闭环（judge+反传·H2 标定）
STAGE4_PROMOTE_WEAN = 4  # promote 收敛+断奶
STAGE5_MULTIMODAL = 5    # 多模态接口预留（defer·非训练）

STAGES: tuple[int, ...] = (STAGE1_SKELETON, STAGE2_CAUSES_ABS,
                           STAGE3_REWARD, STAGE4_PROMOTE_WEAN, STAGE5_MULTIMODAL)

# E8 skippable 阶段（纯 observe 累积·产物纯图累积·续训跳过不部分白训）
# 阶段3 reward 闭环须重标定权重 H2 → non-skippable
SKIPPABLE_STAGES: frozenset[int] = frozenset({STAGE1_SKELETON, STAGE2_CAUSES_ABS})


# ---- 度量门控阈值（oracle 标占位·§十二阶段门控·防缺防超喂） ----
# 图规模/CAUSES 覆盖率/reward 导通率/promote 率·×1000
FLOOR_GRAPH_SIZE_S1 = 100      # 阶段1→2：概念点 ≥ 100
FLOOR_CAUSES_COV_S2 = 50       # 阶段2→3：CAUSES 覆盖率 ≥ 5%（×1000→50）
FLOOR_CONDUCTION_S3 = 500      # 阶段3→4：reward 导通率 ≥ 50%
FLOOR_PROMOTE_S4 = 100         # 阶段4→5：promote 率 ≥ 10%

_SCALE = 1000
_FLOOR_NAMES = frozenset({
    "FLOOR_GRAPH_SIZE_S1",
    "FLOOR_CAUSES_COV_S2",
    "FLOOR_CONDUCTION_S3",
    "FLOOR_PROMOTE_S4",
})
_FLOOR_OVERRIDES: contextvars.ContextVar[dict[str, int] | None] = (
    contextvars.ContextVar("zero_ai_stage_floor_overrides", default=None)
)


def push_stage_floor_overrides(
        overrides: dict[str, int],
        ) -> contextvars.Token[dict[str, int] | None]:
    """叠加当前执行上下文的阶段阈值覆盖，并返回可精确复位的 token。"""
    current = _FLOOR_OVERRIDES.get()
    merged = {} if current is None else dict(current)
    for name, value in overrides.items():
        if name not in _FLOOR_NAMES:
            raise AttributeError(f"未知阶段阈值: {name}")
        if type(value) is not int or value < 0:
            raise TypeError(f"阶段阈值必须是非负 int: {name}")
        merged[name] = value
    return _FLOOR_OVERRIDES.set(merged)


def reset_stage_floor_overrides(
        token: contextvars.Token[dict[str, int] | None]) -> None:
    """使用 push 返回的 token 恢复调用前阶段阈值上下文。"""
    _FLOOR_OVERRIDES.reset(token)


@contextmanager
def stage_floor_overrides(overrides: dict[str, int]) -> Iterator[None]:
    """在当前嵌套或并发上下文内临时覆盖阶段阈值。"""
    token = push_stage_floor_overrides(overrides)
    try:
        yield
    finally:
        reset_stage_floor_overrides(token)


def _stage_floor(name: str) -> int:
    """读取当前上下文覆盖；未覆盖时回退到可由测试设置的进程基线。"""
    current = _FLOOR_OVERRIDES.get()
    if current is not None and name in current:
        return current[name]
    return globals()[name]


@dataclass
class StageGateConfig:
    """每阶段 gate 配比（§十二·observe only / reward off / promote off）。"""
    stage: int
    observe_active: bool = True       # observe 建图（阶段1-4 持续·阶段5 defer）
    reward_active: bool = False       # reward 反传（阶段3 才开·破死锁）
    promote_active: bool = False      # promote 三重（阶段4 才开）
    teacher_active: bool = False      # 教师（断奶前在位·阶段4 断奶后退场）
    weaning_phase: int = WEANING_PRE


@dataclass
class StageMetrics:
    """阶段度量（同源 D2 jsonl·门控用·×1000）。"""
    graph_size: int = 0            # 概念点数
    causes_coverage: int = 0       # CAUSES 覆盖率（有 CAUSES 出边节点占比）
    conduction_rate: int = 0       # reward 导通率（reward>0 占比）
    promote_rate: int = 0          # promote 率（SHADOW→PRIMARY 达率）
    realizes_rate: int = 0         # 达 sink ∧ reward>0 占比（不依赖 REALIZES 边·度量名 legacy）
    oov_promote_rate: int = 0      # OOV 晋升率


def stage_gate_config(stage: int) -> StageGateConfig:
    """每阶段 gate 配比（§十二·observe 先 reward 后破死锁）。"""
    if stage == STAGE1_SKELETON:
        return StageGateConfig(stage, observe_active=True, reward_active=False,
                               promote_active=False, teacher_active=True,
                               weaning_phase=WEANING_PRE)
    if stage == STAGE2_CAUSES_ABS:
        return StageGateConfig(stage, observe_active=True, reward_active=False,
                               promote_active=False, teacher_active=True,
                               weaning_phase=WEANING_PRE)
    if stage == STAGE3_REWARD:
        return StageGateConfig(stage, observe_active=True, reward_active=True,
                               promote_active=False, teacher_active=True,
                               weaning_phase=WEANING_PRE)
    if stage == STAGE4_PROMOTE_WEAN:
        return StageGateConfig(stage, observe_active=True, reward_active=True,
                               promote_active=True, teacher_active=True,
                               weaning_phase=WEANING_PRE)
    # STAGE5_MULTIMODAL：defer·非训练
    return StageGateConfig(stage, observe_active=False, reward_active=False,
                           promote_active=False, teacher_active=False,
                           weaning_phase=WEANING_POST)


def stage_metric_gate(stage: int, metrics: StageMetrics) -> bool:
    """度量门控（该阶段度量达标 → 允许进下一阶段·§十二防缺防超喂）。

    返 True = 该阶段已达标可收口进下一阶段。
    """
    if stage == STAGE1_SKELETON:
        return metrics.graph_size >= _stage_floor("FLOOR_GRAPH_SIZE_S1")
    if stage == STAGE2_CAUSES_ABS:
        return metrics.causes_coverage >= _stage_floor("FLOOR_CAUSES_COV_S2")
    if stage == STAGE3_REWARD:
        return metrics.conduction_rate >= _stage_floor("FLOOR_CONDUCTION_S3")
    if stage == STAGE4_PROMOTE_WEAN:
        return metrics.promote_rate >= _stage_floor("FLOOR_PROMOTE_S4")
    # STAGE5：defer·无门控
    return True


def is_skippable(stage: int) -> bool:
    """E8 stage-skip（纯 observe 累积 skippable·分布标定 non-skippable）。"""
    return stage in SKIPPABLE_STAGES


# ---- G5/C6 harness 真接线（teacher → self_proof_fn·weaning pre） ----

# JudgeFn 契约（episode_loop 注入）：(output, path, input, workmem) -> (reward, GMeta)
JudgeFn = Callable[..., tuple[int, Any]]


def build_judge_fn(graph: Any, weights: JudgeWeights,
                   teacher: Any = None, *,
                   weaning_phase: int = WEANING_PRE,
                   judge_source_id: int | None = None) -> JudgeFn:
    """G5/C6 harness 真接线（卷三 Stage 5 占位 → Stage 6 录放层落地）。

    把 teacher.judge_ground_truth 绑成 judge 的 self_proof_fn·断奶前注入。
    TEACHER_MODE OFF / weaning_post / teacher=None → self_proof_fn=None（pass=1 占位·bit-identical）。

    **D3 裁判源独立性（§十一 #4-bis line710）**：judge_source_id 默认=teacher.source_id（当前裁判=
    训练教师本尊·同源→sources_disjoint False→D3 硬前置挡→can_wean=False·诚实）。独立裁判实例分离后
    caller 传独立 judge_source_id·与 teacher.source_id 不相交→D3 通过。
    """
    self_proof_fn: SelfProofFn | None = None
    if (gates.TEACHER_MODE and teacher is not None
            and weaning_phase == WEANING_PRE):
        self_proof_fn = teacher.judge_ground_truth

    # D3·裁判源独立性判定（当前裁判=教师本尊→同源→False·诚实挡假断奶·独立裁判分离后自然通过）
    teacher_sid = getattr(teacher, "source_id", 0) if teacher is not None else 0
    j_sid = judge_source_id if judge_source_id is not None else teacher_sid
    judge_source_independent = sources_disjoint({j_sid}, {teacher_sid})

    def judge_fn(output: Any, dag_path: Any, input_payload: Any,
                 workmem: Any) -> tuple[int, Any]:
        return judge(output, dag_path, input_payload, graph, weights, workmem,
                     self_proof_fn=self_proof_fn)
    judge_fn.judge_source_independent = judge_source_independent  # type: ignore[attr-defined]
    return judge_fn


def stage_active_gates(cfg: StageGateConfig) -> dict[str, bool]:
    """阶段实际生效的 gate 状态（结合全局 gate 二分·TEACHER_MODE/TRAINING_MODE）。

    全局 gate OFF 时即使阶段配比要 teacher/reward·也不生效（bit-identical 默认）。
    """
    return {
        "observe": cfg.observe_active and gates.TRAINING_MODE,
        "reward": cfg.reward_active and gates.TRAINING_MODE,
        "promote": cfg.promote_active and gates.TRAINING_MODE,
        "teacher": cfg.teacher_active and gates.TEACHER_MODE,
    }
