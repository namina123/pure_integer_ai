"""teacher.weaning_calibration — D5 Mode B 就绪预验台账 + calibration_set（§十一 #4-bis line712·#358 完整实现）。

D1 禁了 weaning**决策**的布尔 flip·但 C6 自证机 Mode 承重在 weaning 时仍有布尔切换副作用——Mode A
（教师 ground-truth·断奶前承重）→Mode B（self-consistency·断奶后承重·墙内弱）。D1/D4 趋势与探针在
教师在位（Mode A 承重）条件下测·post-weaning 在 Mode B 条件下运营——**分布漂移**：D1 趋势不预测
Mode B 性能。补硬前置：**weaning 前须 Mode B 就绪预验**——断奶评估点在独立探针集（D4）上额外跑
Mode B 模拟评估·Mode B 通过率趋势不回升∧window_rounds 确认才允许断奶。

**Mode B 最小形态（不依赖 A3 多路径·VM A1/A2 已图灵完备·self_proof 3 态已锁 #361）**=单路径 VM
re-derivation + 结构不变量。多路径交叉自证是深化·单路径预验现在可建。

台账 weaning_calibration（core=False·DISC_MUTABLE_MONOTONE）：
  (round_id, mode_a_pass, mode_b_pass, calibrated)  # 0/1 纯整
断奶前预验阶段在小批量 calibration_set 上并行跑 Mode A（vm 执行 vs 教师 ground-truth）与 Mode B
（self-consistency 交叉自证）·记台账·算 false_pass_rate/mode_b_agreement。

post-weaning false_pass_rate 滚动更新折扣率（G1 落盘 line712）：post-weaning 持续采 Mode B 信号
反馈到台账滚动更新折扣率（MUTABLE_MONOTONE·非冻结）。**Mode B 墙内弱不可完全解除**（断奶后无 GT·
stable≠correct）·D5 是诚实补强非消除。

铁律：纯整数（success 0/1·计数测度·交叉积有理对·禁浮点）/ 外部只启发（Mode A 断奶前教师·断奶后退场·
  Mode B 永不依赖外部）/ 不走外挂 LLM（断奶后·Mode B 墙内自证）/ MUTABLE_MONOTONE（预验台账）。
诚实边界：Mode B 自洽非真理（两路径一致≠对错·§十四 C6）/ 墙内弱不可完全解除 / stable≠correct。
"""
from __future__ import annotations

from typing import Any

from pure_integer_ai.storage.backend import StorageBackend, register_extension_table
from pure_integer_ai.storage import discipline as disc
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.teacher.weaning import WEANING_WINDOW_ROUNDS

CALIBRATION_TABLE = "weaning_calibration"

_CALIBRATION_COLUMNS = [
    ("round_id", "INT"),       # 训练轮次序（caller 传整序·无墙钟）
    ("mode_a_pass", "INT"),    # Mode A 通过（vm 执行 vs 教师 GT·0/1）
    ("mode_b_pass", "INT"),    # Mode B 通过（self-consistency 单路径·0/1）
    ("calibrated", "INT"),     # 已标定折扣率（0/1·post-weaning 滚动更新）
]
_CALIBRATION_INDEXES = [
    ("round_id",),
]


def register_weaning_calibration(backend: StorageBackend) -> None:
    """注册 D5 预验台账（启动调一次·teacher 扩展表·core=False·DISC_MUTABLE_MONOTONE）。"""
    register_extension_table(
        backend, CALIBRATION_TABLE, _CALIBRATION_COLUMNS,
        discipline=disc.DISC_MUTABLE_MONOTONE,
        indexes=_CALIBRATION_INDEXES,
    )


def record_calibration(backend: StorageBackend, *, round_id: int,
                       mode_a_pass: int, mode_b_pass: int) -> None:
    """记一轮 Mode A vs B 并行预验结果（0/1 纯整·MUTABLE_MONOTONE）。

    断奶前预验阶段在 calibration_set 上并行跑 Mode A / Mode B·记台账供标定量计算。
    """
    assert_int(round_id, mode_a_pass, mode_b_pass, _where="record_calibration")
    if mode_a_pass not in (0, 1) or mode_b_pass not in (0, 1):
        raise ValueError("mode_a_pass/mode_b_pass 须 0/1（纯整计数测度）")
    backend.insert(CALIBRATION_TABLE, {
        "round_id": round_id,
        "mode_a_pass": mode_a_pass,
        "mode_b_pass": mode_b_pass,
        "calibrated": 0,
    })


def _all_rows(backend: StorageBackend) -> list[dict[str, Any]]:
    return backend.select(CALIBRATION_TABLE)


def false_pass_rate(backend: StorageBackend) -> tuple[int, int]:
    """mode_b 误报率 = count(mode_a==fail ∧ mode_b==success) / count(all)·有理对 (p,q)·纯整禁浮点。

    p = count(mode_a_pass==0 and mode_b_pass==1) / q = count(all)。
    断奶后 Mode B 承重时 harness 据 false_pass_rate 给 Mode B success 加置信折扣（G1 落盘）。
    """
    rows = _all_rows(backend)
    q = len(rows)
    if q == 0:
        return (0, 1)
    p = sum(1 for r in rows if int(r["mode_a_pass"]) == 0 and int(r["mode_b_pass"]) == 1)
    return (p, q)


def mode_b_agreement(backend: StorageBackend) -> tuple[int, int]:
    """mode_b 一致率 = count(mode_a==mode_b==success) / count(mode_a==success)·有理对 (p,q)。

    p = count(mode_a_pass==1 and mode_b_pass==1) / q = count(mode_a_pass==1)。
    """
    rows = _all_rows(backend)
    a_pass = [r for r in rows if int(r["mode_a_pass"]) == 1]
    q = len(a_pass)
    if q == 0:
        return (0, 1)
    p = sum(1 for r in a_pass if int(r["mode_b_pass"]) == 1)
    return (p, q)


def mode_b_pass_series(backend: StorageBackend) -> list[tuple[int, int]]:
    """Mode B 通过率序列 (round_id, rate×1000)·按 round_id 升序·D5 趋势不回升判定用。

    每轮 rate = count(mode_b_pass==1) / count(that round) ×1000·纯整。
    """
    rows = _all_rows(backend)
    by_round: dict[int, list[int]] = {}
    for r in rows:
        rid = int(r["round_id"])
        by_round.setdefault(rid, []).append(int(r["mode_b_pass"]))
    series = []
    for rid in sorted(by_round):
        passes = by_round[rid]
        total = len(passes)
        rate = (sum(passes) * 1000) // total if total > 0 else 0
        series.append((rid, rate))
    return series


def mode_b_prevalidated(backend: StorageBackend, *,
                        window_rounds: int = WEANING_WINDOW_ROUNDS) -> bool:
    """D5·Mode B 预验通过：window 内 mode_b 通过率趋势不回升 ∧ 至少 window_rounds 轮有记录。

    预验非新增墙钟/浮点（Mode B 是 VM 执行路径纯整比对）·非新增教师依赖（Mode B 不依赖教师 GT）。
    '不回升'=window 内通过率后窗 ≤ 前窗（交叉积有理对·late_p·early_q ≤ early_p·late_q）∧ 最新达下限
    （Mode B 通过率 ≥ FLOOR_MODE_B·防低通过率平台假预验）。
    """
    series = [rate for _, rate in mode_b_pass_series(backend)]
    if len(series) < window_rounds:
        return False
    window = series[-window_rounds:]
    half = len(window) // 2
    early_p, early_q = (sum(window[:half]), max(len(window[:half]), 1))
    late_p, late_q = (sum(window[half:]), max(len(window[half:]), 1))
    not_rising = late_p * early_q <= early_p * late_q
    return not_rising and window[-1] >= FLOOR_MODE_B


# Mode B 通过率下限（oracle 标占位·×1000·防低通过率平台假预验）
FLOOR_MODE_B = 500   # Mode B 通过率 ≥ 50%（单路径 VM re-derivation·墙内弱·下限低于 Mode A）
