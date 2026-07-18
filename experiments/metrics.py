"""experiments.metrics — 度量 jsonl（同源 D2·oracle导通率/source_dist/断奶曲线）。

MetricsCollector 收集每轮 episode 聚合 + 运行时图状态 → 计算 StageMetrics + 断奶四度量
→ 写 jsonl（同源 D2 = 与 stepper/episode 运行时数据同源·非另起度量管线）。

  conduction_rate  reward 导通率（reward>0 episode 占比·×1000·oracle 导通率）
  realizes_rate    达 sink ∧ reward>0 占比（×1000·度量名 legacy·不依赖 REALIZES 边·doc §8.7-P2 REALIZES 永死）
  judge_self_rate  judge 自评通过率（G5 非 veto 占比·断奶前教师对照·×1000）
  oov_promote_rate OOV 晋升率（SHADOW→PRIMARY 达率·×1000）
  promote_rate     promote 率（本轮 promote 计数 / 图规模·×1000）
  source_dist      edge source 分布（计数·五类收集来源审计）

**同源 D2**：度量从 Episode 聚合 + edge_store 行读出·非独立 instrumentation·
  与 a2_stepper 拓扑层序（layer 隐含步进时钟·dag_path:207 enumerate(topo_layers)）同源口径（运行时 episode 数据）。
  度量不另起管线·直接消费 episode_loop 产出的 Episode（F5 聚合层）。

**断奶曲线**：四度量按 round 序累积 → weaning_series() 喂 weaning_check（双曲线趋势 D1）。
  与收敛 N=1000 episode 窗口口径不同（B3）·断奶是 run 级趋势非 episode 级窗口·不混。

铁律：纯整数（率×1000·全整·assert_no_float）/ 确定性（round_id caller 传整序·无墙钟·
  jsonl 行序确定 bit-identical）/ append-only（metrics jsonl 只 append 不改）/
  外部只启发（度量是信号非判据·断奶判据在 weaning 双曲线非单阈值）。
诚实边界：度量是运行时信号非语义判定（导通率≠能力·接地墙）/ 断奶曲线趋势模型非精确拟合 /
  source_dist 审计非语义真伪 / stable≠correct（度量好≠模型对）。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Iterable

from pure_integer_ai.crosscut.guards.float_guard import assert_no_float
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.cognition.shared.types import Episode, TERMINAL_REACHED_SINK
from pure_integer_ai.training.stages import StageMetrics
from pure_integer_ai.teacher.weaning import WeaningMetrics

_SCALE = 1000


def _rate(num: int, den: int) -> int:
    """纯整数率 num/den ×1000（den=0 → 0·防除零·冷启动无 episode 不报假信号）。"""
    if den <= 0:
        return 0
    return (num * _SCALE) // den


@dataclass
class RoundMetrics:
    """单轮度量（jsonl 一行·同源 D2·纯整数）。"""

    round_id: int = 0
    stage: int = 0
    graph_size: int = 0
    causes_coverage: int = 0          # ×1000
    conduction_rate: int = 0          # ×1000
    promote_rate: int = 0             # ×1000
    realizes_rate: int = 0            # ×1000
    judge_self_rate: int = 0          # ×1000
    oov_promote_rate: int = 0         # ×1000
    episode_count: int = 0
    reward_pos: int = 0               # reward>0 episode 数
    dead_end_count: int = 0
    veto_count: int = 0
    source_dist: dict[int, int] = field(default_factory=dict)
    # §8.7-全 生成侧全环·task-driven L8 episode 度量（反 theater ③下游读者锚·OutputResult.parts→计数真读）。
    # generate 行独有（stage 行恒 0）·generate_verified=外真验过 episode 数·generate_total=task-driven episode 总数。
    generate_verified: int = 0
    generate_total: int = 0

    def to_json(self) -> dict[str, Any]:
        """jsonl 行（sort_keys 确定性·source_dist int 键转 str 因 JSON 键须 str）。"""
        return {
            "round_id": self.round_id,
            "stage": self.stage,
            "graph_size": self.graph_size,
            "causes_coverage": self.causes_coverage,
            "conduction_rate": self.conduction_rate,
            "promote_rate": self.promote_rate,
            "realizes_rate": self.realizes_rate,
            "judge_self_rate": self.judge_self_rate,
            "oov_promote_rate": self.oov_promote_rate,
            "episode_count": self.episode_count,
            "reward_pos": self.reward_pos,
            "dead_end_count": self.dead_end_count,
            "veto_count": self.veto_count,
            "generate_verified": self.generate_verified,
            "generate_total": self.generate_total,
            "source_dist": {str(k): v for k, v in sorted(self.source_dist.items())},
        }


class MetricsCollector:
    """度量收集器（同源 D2 jsonl + StageMetrics snapshot + 断奶 series）。

    用法：formal_train 每轮调 record_round → 写 jsonl + 更新 snapshot/series。
    snapshot() 喂 stage_metric_gate（阶段间门控）·weaning_series() 喂 weaning_check（断奶判据）。
    """

    def __init__(self, metrics_path: str) -> None:
        self.metrics_path = metrics_path
        os.makedirs(os.path.dirname(os.path.abspath(metrics_path)), exist_ok=True)
        self._fh = open(metrics_path, "a", encoding="utf-8")
        self._last: RoundMetrics = RoundMetrics()
        self._series: list[WeaningMetrics] = []

    # ---- 主入口 ----

    def record_round(self, round_id: int, stage: int,
                     episodes: Iterable[Episode], *,
                     graph_size: int, causes_coverage: int,
                     promote_count: int, oov_promote_count: int,
                     source_counts: dict[int, int] | None = None,
                     intervention_rate: int = 0,
                     holdout_retention: int = 0,
                     dependency: int = 0,
                     count_g5_self_assess: bool = False) -> RoundMetrics:
        """记录一轮度量（同源 D2·写 jsonl + 更新 snapshot/series）。

        round_id  caller 传整序（无墙钟·formal_train 轮次序）。
        episodes  本轮 episode_loop 产出的 Episode 列表（F5 聚合层）。
        intervention_rate/holdout_retention/dependency  D1 两规定曲线 + 依赖度（×1000·
          caller 从教师介入计数/探针评估/租借比例算·默认 0·断奶判据 D1 方向性用·§十一 #4-bis）。
        """
        assert_int(round_id, stage, graph_size, causes_coverage,
                   promote_count, oov_promote_count,
                   intervention_rate, holdout_retention, dependency,
                   int(count_g5_self_assess),
                   _where="metrics.record_round")
        ep_list = list(episodes)
        total = len(ep_list)
        reward_pos = sum(1 for e in ep_list if e.reward > 0)
        dead_end = sum(e.dead_end_count for e in ep_list)
        veto = sum(e.judge_veto_count for e in ep_list)
        realizes = sum(1 for e in ep_list
                       if e.terminal == TERMINAL_REACHED_SINK and e.reward > 0)
        # judge 自评通过率：REACHED_SINK 且 judge 产 reward>0（G5 非 veto·断奶前教师对照）
        judged = sum(1 for e in ep_list
                     if e.terminal == TERMINAL_REACHED_SINK)
        # W7 断点5：count_g5_self_assess=True 时算术域 verify round 不排除 G5_active（g5 是"verify 门 active"
        # 标志 formal_train verify round 非 judge 排除条件·度量口径错配→judge_self 恒 0→floors_met 永假）。
        # 默认 False（language/code 域 + bit-identical：G5_active 排除不变）。
        judge_self = sum(1 for e in ep_list
                         if e.terminal == TERMINAL_REACHED_SINK and e.reward > 0
                         and (count_g5_self_assess or not e.judge_G5_active))

        m = RoundMetrics(
            round_id=round_id,
            stage=stage,
            graph_size=graph_size,
            causes_coverage=causes_coverage,
            conduction_rate=_rate(reward_pos, total),
            promote_rate=_rate(promote_count, max(graph_size, 1)),
            realizes_rate=_rate(realizes, total),
            judge_self_rate=_rate(judge_self, max(judged, 1)),
            oov_promote_rate=_rate(oov_promote_count, max(graph_size, 1)),
            episode_count=total,
            reward_pos=reward_pos,
            dead_end_count=dead_end,
            veto_count=veto,
            source_dist=dict(source_counts or {}),
        )
        for v in (m.conduction_rate, m.promote_rate, m.realizes_rate,
                  m.judge_self_rate, m.oov_promote_rate):
            assert_no_float(v, _where="metrics.record_round.rate")
        self._write(m)
        self._last = m
        self._series.append(WeaningMetrics(
            rounds=round_id,
            conduction_rate=m.conduction_rate,
            realizes_rate=m.realizes_rate,
            judge_self_rate=m.judge_self_rate,
            oov_promote_rate=m.oov_promote_rate,
            intervention_rate=intervention_rate,
            holdout_retention=holdout_retention,
            dependency=dependency,
        ))
        return m

    def backfill_retention(self, retention: int) -> None:
        """W7 断点1：回填 holdout_retention 到全 series 点。

        _run_simulated_offline_eval 在 stage loop **后**跑（依赖 stage4 末状态）·此时 weaning_series 已被
        record_round baked 成全 0（record 时 ctx.holdout_retention 默认 0）。retention 是探针集属性
        （eval 一次采·trivial fixture 恒 1000·不随轮变）→ 回填全 series 点语义正确·解 D1 曲线②
        _retention_stable 读全 0 永不过 FLOOR_RETENTION（W6 memory 标的风险·W7 修）。
        仅 ctx.holdout_retention>0 时 caller 调（simulate_offline_eval=True·默认 OFF 不调·bit-identical）。
        """
        assert_int(retention, _where="metrics.backfill_retention")
        for m in self._series:
            m.holdout_retention = retention

    def _write(self, m: RoundMetrics) -> None:
        self._fh.write(json.dumps(m.to_json(), ensure_ascii=False, sort_keys=True))
        self._fh.write("\n")
        self._fh.flush()

    def record_generate_round(self, round_id: int,
                              episodes: Iterable[Episode]) -> RoundMetrics:
        """§8.7-全 生成侧全环度量（反 theater ③下游读者锚·task-driven episode→计数真读）。

        task-driven generate episodes → generate_verified/generate_total 计数·写独立 jsonl 行（stage=0·
        与阶段行 disambiguate）。**不入 weaning_series**（非断奶曲线·断奶判据在语言域四度量非算子生成）/
        **不更新 _last snapshot**（非阶段门控·stage_metric_gate 不读 generate）·纯观测信号行（度量是信号
        非判据·§十一）。generate_verified = **e.output.parts 非空** episode 数（真读 OutputResult·反 theater ③
        下游读者锚·parts 非空⟺verified·审计必修）·generate_total = generate episode 总数。
        conduction_rate 本行 = reward_pos/total（与 parts 一致·冗余双保险）。
        """
        assert_int(round_id, _where="metrics.record_generate_round")
        ep_list = list(episodes)
        total = len(ep_list)
        # **反 theater ③下游读者锚·审计必修**：generate_verified 须从 e.output.parts 计数（真读 OutputResult·
        # 非死写）·parts 非空 ⟺ verified（_run_task_driven_generate 守·未验→parts=[]）。读 reward 会落
        # §8.7-洗-证伪 candidate (Z) theater（parts 死写无消费者）·故必读 parts。
        # P0 #1041 构造② judge J4word 已落（reward truthiness 校准·modality-safe：非语言 output token_refs 空
        # → J4word=0 不扰 code/arith reward）。构造① metrics verified 真词感知**defer Phase2**：record_generate_round
        # 同时服务 code+language task-driven·code generate 无 token_refs·token-aware verified 误杀 code 计数
        # （test_task_driven_metrics_downstream_reader 验）·须 modality-aware（capability_exam 知域·Phase2 接）。
        verified = sum(1 for e in ep_list
                       if e.output is not None and e.output.parts)
        reward_pos = sum(1 for e in ep_list if e.reward > 0)   # conduction_rate 用（与 parts 一致·冗余双保险）
        m = RoundMetrics(
            round_id=round_id, stage=0,   # stage=0 = generate pass（真实 stage ∈ {1..5}·disambiguate）
            episode_count=total, reward_pos=reward_pos,
            conduction_rate=_rate(reward_pos, total),
            generate_verified=verified, generate_total=total,
        )
        assert_no_float(m.conduction_rate, _where="metrics.record_generate_round.rate")
        self._write(m)   # 只写 jsonl·不更新 _last / 不 append _series（观测信号非门控/断奶）
        return m

    # ---- 消费 ----

    def snapshot(self) -> StageMetrics:
        """当前 StageMetrics（喂 stage_metric_gate·阶段间门控）。"""
        l = self._last
        return StageMetrics(
            graph_size=l.graph_size,
            causes_coverage=l.causes_coverage,
            conduction_rate=l.conduction_rate,
            promote_rate=l.promote_rate,
            realizes_rate=l.realizes_rate,
            oov_promote_rate=l.oov_promote_rate,
        )

    def weaning_series(self) -> list[WeaningMetrics]:
        """断奶四度量 series（喂 weaning_check·双曲线趋势 D1·run 级非 episode 级）。"""
        return list(self._series)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None  # type: ignore[assignment]

    def __enter__(self) -> "MetricsCollector":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
