"""正式训练放量前的隔离试跑和多维验收门。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from pure_integer_ai.cognition.shared.types import Episode
from pure_integer_ai.config import gates
from pure_integer_ai.experiments.collection import CollectedItem
from pure_integer_ai.experiments.language_generation_episode import (
    TypedLanguageEpisode,
)
from pure_integer_ai.experiments.round_runtime import (
    DefaultRoundRunner,
    RoundRunner,
    _run_runner_episodes,
)
from pure_integer_ai.experiments.train_context import TrainContext
from pure_integer_ai.experiments.train_diagnostics import (
    _anti_collapse_summary,
    _causes_coverage,
    _edge_count,
    _graph_size,
)
from pure_integer_ai.training.cursor import (
    CursorState,
    check_replay_coverage,
    cursor_resume,
    mark_completed,
)
from pure_integer_ai.training.oracle import validate_b1_b4
from pure_integer_ai.experiments.verification_orchestration import (
    APPLICABILITY_APPLICABLE,
    VERDICT_REFUTE,
    VERDICT_SUPPORT,
)
from pure_integer_ai.training.stages import (
    SKIPPABLE_STAGES,
    STAGE1_SKELETON,
    STAGE3_REWARD,
    STAGES,
)

PRE_FLIGHT_ROUNDS = 50000
PRE_FLIGHT_MEM_BUDGET_PER_ROUND = 4096

@dataclass
class PreFlightReport:
    """E7 pre-flight 验收报告（6 项全过才放量·守几百G不重训红线）。"""

    metrics_signal: bool = False       # ① 度量真有信号（图/CAUSES/导通率非全0非盲）
    mem_ok: bool = False               # ② 内存峰值<mem_hard_pct（轻量代理·真 mem 工程层 defer）
    reward_gate_ok: bool = False       # ③ reward gate 实际生效（judge 门否决/反传只 CAUSES）
    replay_coverage_ok: bool = False   # ④ replay 覆盖率≥阈值（E4 续训前置）
    cursor_resume_ok: bool = False     # ⑤ cursor resume 能跳已完成阶段（E8 续训机制）
    # ⑥ 防塌柱③ 探索压力（S12·9a·闭环证伪剩 D 墙前置）。柱③ 是唯一在"无显式失败"时 active 的柱
    # （① 结构 judge / ② 真负通路 在试跑 happy path 上 dormant·不算塌·anti_collapse.py:12）。
    # 故 collapse_ok 口径=柱③ 无失守（有 PR 的 episode 全柱③ OK·方差够 dormant OR 注入缓解）·
    # 非三柱全过（避 happy path 误报塌）。verified=0（全空 PR·dag_path 未跑）→ 退化放行 +
    # detail["collapse_degraded"]=True 诚实标（无 PR 可验·非趋平退化信号·由 ①/③ 门先拦）。
    collapse_ok: bool = False
    anti_theater_ok: bool = True       # ⑦ 反 theater 自我考核（#726 片2·层2锚点+层3反向回归·默认 True=未触发 passthrough·生产 caller 传 config+backend_factory 触发后真判·I-新闭环"旗标对自身失效残留"）
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return (self.metrics_signal and self.mem_ok and self.reward_gate_ok
                and self.replay_coverage_ok and self.cursor_resume_ok
                and self.collapse_ok and self.anti_theater_ok)


def pre_flight(ctx: TrainContext, corpus: list[CollectedItem], *,
               rounds: int = PRE_FLIGHT_ROUNDS,
               runner: RoundRunner | None = None,
               replay_needed: list[tuple[int, tuple]] | None = None,
               config: Any | None = None,
               backend_factory: "Callable[[], Any] | None" = None) -> PreFlightReport:
    """在独立评测沙箱运行放量门，只向调用方返回验收报告。"""
    from pure_integer_ai.experiments.evaluation_isolation import isolated_evaluation

    with isolated_evaluation(ctx, label="pre_flight") as eval_ctx:
        return _pre_flight_impl(
            eval_ctx,
            corpus,
            rounds=rounds,
            runner=runner,
            replay_needed=replay_needed,
            config=config,
            backend_factory=backend_factory,
        )


def _pre_flight_impl(ctx: TrainContext, corpus: list[CollectedItem], *,
                     rounds: int = PRE_FLIGHT_ROUNDS,
                     runner: RoundRunner | None = None,
                     replay_needed: list[tuple[int, tuple]] | None = None,
                     config: Any | None = None,
                     backend_factory: "Callable[[], Any] | None" = None) -> PreFlightReport:
    """E7 pre-flight 放量门（小规模试跑 → 5 验收项·全过才放量·§十二 line903）。

    rounds 经验初值 50000（oracle 可调）·试跑 min(rounds, len(corpus)) 轮。
    失败=禁放量（修配置重试小规模）非"继续跑看看"。
    """
    validate_b1_b4()   # B1-B4 占位校验前置（防漂移）
    r = runner or DefaultRoundRunner()
    trial = corpus[:min(rounds, len(corpus))] if corpus else []
    eps: list[Episode | TypedLanguageEpisode] = []
    # A1：生产试跑 observe 须自产 CAUSES（同 formal_train·CUE_EXTRACTOR_MODE ON·致命3 残留·断奶后语言域源）。
    # 翻在此非 run_round_full（cue 在 split+observe 被读·run_round_full 保 gate-respecting 可测单元）·详见 formal_train A1 块。
    preflight_gate_token = gates.push_gate_overrides({
        "CUE_EXTRACTOR_MODE": True,
    })
    try:
        for rid, item in enumerate(trial):
            eps.extend(_run_runner_episodes(
                ctx, r, item, STAGE3_REWARD, rid))
    finally:
        gates.reset_gate_overrides(preflight_gate_token)

    legacy_eps = [item for item in eps if isinstance(item, Episode)]
    typed_eps = [
        item for item in eps if isinstance(item, TypedLanguageEpisode)]
    typed_signals = tuple(
        signal for episode in typed_eps for signal in episode.signals)
    rep = PreFlightReport()
    # ① 度量真有信号
    gsize = _graph_size(ctx)
    ccov = _causes_coverage(ctx)
    cond = (
        sum(1 for e in legacy_eps if e.reward > 0) * 1000
        // max(len(legacy_eps), 1)
        if legacy_eps else 0
    )
    typed_generation_complete = sum(
        int(item.generation_complete) for item in typed_eps)
    typed_postcheck_complete = sum(
        int(item.postcheck_complete is True) for item in typed_eps)
    rep.metrics_signal = (
        gsize > 0
        or ccov > 0
        or cond > 0
        or typed_generation_complete > 0
        or bool(typed_signals)
    )
    rep.detail["graph_size"] = gsize
    rep.detail["causes_coverage"] = ccov
    rep.detail["conduction_rate"] = cond
    rep.detail["legacy_episode_count"] = len(legacy_eps)
    rep.detail["typed_episode_count"] = len(typed_eps)
    rep.detail["typed_generation_complete"] = typed_generation_complete
    rep.detail["typed_postcheck_complete"] = typed_postcheck_complete
    rep.detail["typed_signal_count"] = len(typed_signals)
    rep.detail["typed_signal_support"] = sum(
        1 for item in typed_signals
        if item.applicability == APPLICABILITY_APPLICABLE
        and item.verdict == VERDICT_SUPPORT)
    rep.detail["typed_signal_refute"] = sum(
        1 for item in typed_signals
        if item.applicability == APPLICABILITY_APPLICABLE
        and item.verdict == VERDICT_REFUTE)

    # ② 内存峰值（轻量代理：试跑后概念点+边总数 ≤ 每 round 预算×rounds·防超线性膨胀 OOM·
    #    stub ② 修：旧版硬编码 True·真 OS 级 mem_hard_pct 监控 defer 工程层·此为纯整 in-process 代理 falsifiable）
    peak_resource = _graph_size(ctx) + _edge_count(ctx)
    mem_budget = PRE_FLIGHT_MEM_BUDGET_PER_ROUND * max(len(trial), 1)
    rep.mem_ok = peak_resource <= mem_budget
    rep.detail["peak_resource"] = peak_resource
    rep.detail["mem_budget"] = mem_budget
    rep.detail["trial_rounds"] = len(trial)

    # ③ reward gate 实际生效（judge 产 veto 或 reward>0·非全 0 盲·stub ③ 修：删 cond>=0 恒真尾·须真产信号）
    has_veto = any(
        e.judge_veto_count > 0 or e.dead_end_count > 0
        for e in legacy_eps)
    has_pos = any(e.reward > 0 for e in legacy_eps)
    has_typed_postcheck = bool(typed_signals)
    rep.reward_gate_ok = has_veto or has_pos or has_typed_postcheck
    rep.detail["has_veto"] = has_veto
    rep.detail["has_pos_reward"] = has_pos
    rep.detail["has_typed_postcheck"] = has_typed_postcheck

    # ④ replay 覆盖率（E4·教师续训前置）
    if ctx.teacher is not None and replay_needed:
        rep.replay_coverage_ok = check_replay_coverage(ctx.teacher, replay_needed)
    else:
        rep.replay_coverage_ok = True   # 无教师/无 needed → 放行（非续训场景）
    rep.detail["replay_needed_count"] = len(replay_needed or [])

    # ⑤ cursor resume 能跳（E8·机制验·跳已完成 skippable）
    st = CursorState(base_run_id="preflight", run_id="preflight")
    mark_completed(st, STAGE1_SKELETON, skippable=True)
    todo = cursor_resume(st, list(STAGES), skippable=SKIPPABLE_STAGES)
    rep.cursor_resume_ok = STAGE1_SKELETON not in todo and STAGE3_REWARD in todo
    rep.detail["cursor_todo"] = todo

    # ⑥ 防塌柱③ 探索压力（S12·9a·闭环证伪剩 D 墙前置：collapse_ok 进 passed 阻塞门）。
    # _anti_collapse_summary 对非空 pr_vector 的 episode 跑 anti_collapse_verify·汇总柱①②③ 计数。
    # collapse_ok 口径=柱③ 无失守（有 PR 的 episode 全柱③ OK）·非三柱全过——柱③ 是唯一在"无显式
    # 失败"时 active 的柱（①② 在试跑 happy path 上 dormant 不算塌·anti_collapse.py:12）·故只用柱③
    # 做放量阻塞判据。reward 阶段 EXPLORATION_MODE ON（run_round_full :339）→ dag_path 内注入 →
    # 趋平时柱③ OK；柱③ 失守（注入失败/EXPLORATION_MODE 关且趋平）= 趋平退化信号 → 禁放量。
    # verified=0（全空 PR·dag_path 未跑·如代码域语料）→ 退化放行 + 诚实标（无 PR 可验·非趋平信号）。
    ac = _anti_collapse_summary(eps)
    rep.detail["anti_collapse"] = ac
    if ac["verified"] == 0:
        rep.collapse_ok = True
        rep.detail["collapse_degraded"] = True    # 退化放行（无 PR 可验·由 ①/③ 门先拦）
    else:
        rep.collapse_ok = (ac["pillar3_ok"] == ac["verified"])
        rep.detail["collapse_degraded"] = False

    # ⑦ 反 theater 自我考核（#726 片2·层2锚点+层3反向回归·I-新闭环"旗标对自身失效残留"）。
    # 机制在测试中真活（test_capability_anti_theater 验锚点真造 FAIL + mutation 验判据敏感·非死 theater）·
    # 但生产 caller 从未传 anti_theater=True -> 旗标对自身失效残留（D6 病更深层）。pre_flight 是放量门·
    # 天然反 theater 自检点：caller 传 config+backend_factory -> 跑锚点（corpus 层注入坏语料·独立 backend·
    # 验期望维度判 FAIL 非 dead theater）+ 反向回归（8 维判据可证伪+NE 守恒）-> anti_theater_ok 真判。
    # 默认 None（既有 9+ caller 不传）-> skip·anti_theater_ok=True passthrough·守 bit-identical（既有测零翻）。
    # 主入口 :2398 传 config+lambda:DictBackend() -> 生产 pre_flight 触发自检·fail 禁放量（passed 含 anti_theater_ok）。
    if config is not None and backend_factory is not None:
        from pure_integer_ai.experiments.capability_exam import (
            run_anti_theater_anchor, run_reverse_regression)
        _pf_runner = runner or DefaultRoundRunner()
        _anchors = run_anti_theater_anchor(config, backend_factory, runner=_pf_runner)
        _regressions = run_reverse_regression()
        rep.anti_theater_ok = (all(a.passed for a in _anchors)
                               and all(r.passed for r in _regressions))
        rep.detail["anti_theater_triggered"] = True
        rep.detail["anti_theater_anchor"] = [
            a.to_dict() for a in sorted(_anchors, key=lambda a: a.name)]
        rep.detail["anti_theater_regression"] = [
            r.to_dict() for r in sorted(_regressions, key=lambda r: r.dim)]
    else:
        rep.detail["anti_theater_triggered"] = False
        rep.detail["anti_theater_note"] = (
            "反 theater 自我考核未触发（caller 未传 config/backend_factory·#726 片2 生产 caller opt-in·"
            "机制在 test_capability_anti_theater 真活·此 caller 跳过自检）")
    return rep

__all__ = [
    "PRE_FLIGHT_MEM_BUDGET_PER_ROUND",
    "PRE_FLIGHT_ROUNDS",
    "PreFlightReport",
    "_pre_flight_impl",
    "pre_flight",
]
