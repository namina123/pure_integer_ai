"""cognition.result.convergence — 模块5 收敛判据（含负通路活跃·假收敛识别·§十四子问题3）。

convergence_check(history: EpisodeHistory) -> ConvergenceReport
  收敛判据必须含负通路活跃否则假收敛：
    steady_state = 比率方差<阈 ∧ 导通率平台 ∧ promote率平台 ∧ 负通路活跃 ∧ 非塌信号
    反塌信号: sn/tn→1均匀 + PR bias方差→0 + 负通路failure计数=0（含judge veto+死路·M7同口径）= 塌信号非收敛
    非"done"是行为稳态非语义真理。

  B3 落盘：固定 N=1000 episode 窗口（纯整·bit-identical）·断奶评估点对齐 D1 window_rounds=4 runs
    （双曲线趋势·非布尔阈值·两窗口口径不同 episode vs runs 同源度量 jsonl·不混）。
  M7 落盘：负通路 failure 计数=0 用 failure_count（含 judge veto+死路·与 neg_pathway_active 同口径·
    消解"judge veto 活跃+死路不活跃误判塌信号"的口径分裂）·neg_reward_count 仅留诊断。

  M10 记忆启用过渡验收门控衔接（§十三M10）：训练后开记忆须重测此收敛判据（记忆参与后重测
    导通率/负通路活跃/趋平信号/收敛·达标才确认记忆启用稳态）·本模块是 M10 的验收工具。

铁律：纯整数（比率/方差/计数全纯整 ×1000）/ 确定性（固定窗口·bit-identical）。
诚实边界：steady-state 是行为稳态非语义真理（§十四·非"done"）/ 收敛是行为经验性非结构保证。
**B2 接+降级（2026-07-03）**：
  - **接（live·单点源）**：`neg_pathway_active_from(eps)` = D2 负通路活跃单点实现（failure_count=
    judge_veto+dead_end>0·M7 同口径·与 convergence_check.neg_pathway_active 同源）·formal_train D2
    调此**非内联**（消 design "不重复实现" gap·本模块不再整模块零 caller）。
  - **降级（defer·full 验收）**：`convergence_check` / `EpisodeHistory` / `StatRecord` = 收敛验收 ready
    原语·full steady_state/real_convergence 验收 defer——需①linkage_conduction_rate 链路导通率
    （reward→strength→PR→dag_path→generate 各环真传信号·伪代码未实现·当前仅 metrics.py 简单 reward>0 占比·非链路版·
    **反 theater 正解=4 per-link 布尔/计数非单 int 比率**·禁"算出精确导通率值"·weaning 六闸门已替代此验收载体）。
    ②sn_tn_ratio_variance 需 path_result 线程（dag_path/Episode 签名连锁改·接 formal_train 碰 bit-identical）。
    **原"realizes_rate 需 REALIZES 复活"已纠正（2026-07-11 孤儿审计）**：stale——doc §8.7-P2 REALIZES 永死不复用·
    metrics.py:136-137 realizes_rate = REACHED_SINK∧reward>0 根本不依赖 REALIZES 边·度量已活·非 defer 依赖。
    故 convergence_check 不接（partial StatRecord 致 steady_state/collapse_signal 伪信号·纸面闭合）·
    full 验收随收敛验收实施期/M10 gate 落（同 companion.py:93 ready 原语范式·保留 test_stage5 单测）。
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from pure_integer_ai.crosscut.guards.float_guard import assert_no_float
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.cognition.shared.types import ConvergenceReport, Episode


def neg_pathway_active_from(eps: list[Episode]) -> bool:
    """D2 负通路活跃单点源（B2 接·formal_train D2 调此非内联·消 design-code gap）。

    failure_count = judge_veto_count + dead_end_count > 0（M7 同口径·与 convergence_check.neg_pathway_active
    同源 failure_count_recent>0）。空 eps → False（无负通路证据·诚实）。

    full convergence_check steady_state 验收 defer（需 linkage_conduction_rate 链路导通率未实现·
    见模块 docstring B2 降级）·此 helper 只取 neg_pathway_active 信号（D2 断奶硬前置）。
    """
    return any(e.judge_veto_count > 0 or e.dead_end_count > 0 for e in eps)

# B3 落盘：固定 N=1000 episode 窗口（纯整·bit-identical·oracle 标）
CONVERGENCE_WINDOW = 1000

# oracle 标定阈值起点（§十四·实施用此占位·oracle 验后调）
THETA_UNIFORM = 1          # sn/tn→1 均匀（比率方差<此值=均匀趋同·塌信号条件1）
THETA_VARIANCE = 1         # PR bias 方差→0（<此值=趋平·塌信号条件2·与 anti_collapse 同阈）
THETA_RATIO_VAR = 100      # 比率方差<阈（steady-state·有方差但低·非→1均匀）
THETA_DELTA = 1            # 导通率/promote 平台（连续 delta<此值=平台）


@dataclass
class StatRecord:
    """单 episode 统计记录（EpisodeHistory 窗口元素·纯整）。

    failure_count        judge_veto + dead_end（M7 同口径·负通路 failure 计数）
    neg_reward_count     死路负 reward 计数（诊断用·不进 collapse_signal·M7 落盘）
    pr_variance          PR bias 方差（×1000·anti_collapse integer_variance）
    sn_tn_ratio_variance sn/tn 比率方差（×1000·越低越趋同→1）
    conduction_rate      链路导通率（×1000·reward→strength→PR→generate 各环真传信号）
    realizes_rate        REACHED_SINK∧reward>0 占比（×1000·metrics.py:136-137 同源·不依赖 REALIZES 边·
                         doc §8.7-P2 REALIZES 永死不复用·本字段 defer full convergence 接线前零 caller）
    promote_rate         伴随晋升率（×1000·伴随闸收敛信号）
    """

    failure_count: int = 0
    neg_reward_count: int = 0
    pr_variance: int = 0
    sn_tn_ratio_variance: int = 0
    conduction_rate: int = 0
    realizes_rate: int = 0
    promote_rate: int = 0


@dataclass
class _RateDelta:
    """率 + 连续 delta（平台判据用·🟡落盘·统一结构体对齐 conduction·消解 promote_rate.delta 矛盾）。"""

    rate: int = 0
    delta: int = 0


@dataclass
class EpisodeHistory:
    """多 episode 历史（固定 N=1000 窗口·FIFO·纯整）。

    append(stat) 追加·窗口满自动淘汰最旧。recent 聚合供 convergence_check 消费。
    """

    window: int = CONVERGENCE_WINDOW
    _records: deque = field(default_factory=lambda: deque(maxlen=CONVERGENCE_WINDOW))

    def append(self, stat: StatRecord) -> None:
        """追加单 episode 统计（窗口满 FIFO 淘汰最旧）。"""
        assert_no_float(stat.failure_count, stat.neg_reward_count,
                        stat.pr_variance, stat.sn_tn_ratio_variance,
                        stat.conduction_rate, stat.realizes_rate, stat.promote_rate,
                        _where="EpisodeHistory.append")
        self._records.append(stat)

    def __len__(self) -> int:
        return len(self._records)

    # ---- recent 聚合（窗口内） ----

    @property
    def failure_count_recent(self) -> int:
        return sum(s.failure_count for s in self._records)

    @property
    def neg_reward_count_recent(self) -> int:
        return sum(s.neg_reward_count for s in self._records)

    @property
    def pr_bias_variance(self) -> int:
        """PR bias 方差（最近 episode·趋平信号用）。空历史→0。"""
        return self._records[-1].pr_variance if self._records else 0

    @property
    def sn_tn_ratio_variance(self) -> int:
        return self._records[-1].sn_tn_ratio_variance if self._records else 0

    @property
    def realizes_hit_rate(self) -> int:
        return self._records[-1].realizes_rate if self._records else 0

    def _rate_delta(self, attr: str) -> _RateDelta:
        """率 + 连续 delta（最近两记录差·平台判据）。"""
        if not self._records:
            return _RateDelta()
        recs = list(self._records)
        cur = getattr(recs[-1], attr)
        prev = getattr(recs[-2], attr) if len(recs) >= 2 else cur
        return _RateDelta(rate=cur, delta=abs(cur - prev))

    @property
    def conduction(self) -> _RateDelta:
        return self._rate_delta("conduction_rate")

    @property
    def promote_stats(self) -> _RateDelta:
        """🟡落盘·结构体{rate,delta}（原 promote_rate 标量后取 .delta 矛盾·统一结构体对齐 conduction）。"""
        return self._rate_delta("promote_rate")


def convergence_check(history: EpisodeHistory) -> ConvergenceReport:
    """收敛判据（含负通路活跃·假收敛识别）。返 ConvergenceReport。"""
    report = ConvergenceReport()
    if len(history) == 0:
        return report   # 空历史·未稳态未塌（诚实·需观测）

    # —— 反塌信号（关键·假收敛识别·M7 同口径） ——
    sn_tn_uniform = history.sn_tn_ratio_variance < THETA_UNIFORM
    pr_var_zero = history.pr_bias_variance < THETA_VARIANCE
    neg_pathway_inactive = history.failure_count_recent == 0   # M7·含 judge veto+死路
    report.collapse_signal = (sn_tn_uniform and pr_var_zero
                              and neg_pathway_inactive)

    # —— 负通路活跃（真收敛必须·D2 断奶硬前置同源度量） ——
    neg_pathway_active = history.failure_count_recent > 0
    report.neg_pathway_active = neg_pathway_active

    # —— steady-state 判定 ——
    conduction = history.conduction
    promote = history.promote_stats
    plateau_conduction = conduction.delta < THETA_DELTA
    plateau_promote = promote.delta < THETA_DELTA
    ratio_var_low = history.sn_tn_ratio_variance < THETA_RATIO_VAR
    report.steady_state = (
        ratio_var_low and plateau_conduction and plateau_promote
        and neg_pathway_active
        and not report.collapse_signal
    )

    # —— 假收敛识别 ——
    if report.collapse_signal:
        report.real_convergence = False   # 塌信号=假收敛·趋平退化非稳态
    elif not neg_pathway_active:
        report.real_convergence = False   # 负通路不活跃=假收敛·防塌柱② dormant
    else:
        report.real_convergence = report.steady_state
    return report
