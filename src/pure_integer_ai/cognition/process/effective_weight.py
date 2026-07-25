"""cognition.process.effective_weight — 模块7 effective_weight = strength × rate（H4）。

衔接条件③ 精确化：PR 主权重读 effective_weight 非裸 strength。
  - PRECEDES：strength 恒 = 1·rate 恒 = 1（序边结构真值·不接 reward·§7.1）
  - CAUSES：effective_weight = strength × (sn×1000/(sn+tn))（rate 0..1000·率降权脏边）
  - REFERS_TO 性质B（OCCURRENCE·cross-space 结构锚·F3）：occurrence 时序衰减权重
        max(0, strength×DECAY_K − logical_age)（I5 floor 防负·记忆主导场景进 PR 邻接）
  - COOCCURS SHADOW 不进 PR 传播（卷一模块6 隔离·本函数不被 COOCCURS 调用）

H4 闭环：reward 调 sn/tn → rate 变 → effective_weight 变 → 下轮 PR 权重变（模块8→模块2）。
消解 strength 单调脏边放大（§十三D·脏边早期被错 reward 强化后 strength 不降·PR 持续放大
脏边·rate 降权缓解·降权靠 rate 非降 strength 守 MONOTONE）。

纯整数：rate = sn×1000//(sn+tn) 整数除法（0..1000）·occurrence 衰减 max(0,...) 纯整（I5）。
诚实边界：rate 是统计成功率非语义可信度（stable≠correct）。
"""
from __future__ import annotations

from typing import Any

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage.edge_types import EDGE_PRECEDES, EDGE_CAUSES, EDGE_REFERS_TO
from pure_integer_ai.storage.edge_store import SUBTYPE_OCCURRENCE

# ×1000 缩放（rate 域 0..1000）
RATE_SCALE = 1000

# occurrence 衰减系数（I5 floor 防负权重·oracle 标定起点）
DECAY_K = 1


def edge_rate(edge: dict[str, Any]) -> int:
    """CAUSES 边的成功率 rate ∈ [0, RATE_SCALE]（sn×1000/(sn+tn)·纯整整数除法）。

    sn+tn==0（无观测）→ 0（新边·待 reward·is_unobserved 区分未观测 vs 已失败）。
    rate>0 ⟺ sn>0。PRECEDES rate 恒 = RATE_SCALE（结构真值·effective_weight 走专支返 1 非此函数）。
    """
    sn = edge.get("sn", 0) or 0
    tn = edge.get("tn", 0) or 0
    total = sn + tn
    if total <= 0:
        return 0
    return (sn * RATE_SCALE) // total


def is_unobserved(edge: dict[str, Any]) -> bool:
    """CAUSES 边是否未观测（sn==0 且 tn==0·item3 缺漏2·2026-07-02）。

    区分冷启动两态：未观测（sn=tn=0·该给机会·stepper 放行·R5 兜底）vs
    已失败（sn=0 tn>0·真无因·stepper 挡·正确停滞·§十三D）。edge_rate 对两者都返 0·
    本谓词区分·解冷启动死锁（rate=0 边进不了 path.edges→R5 永不触发→sn 永 0）。
    """
    return (edge.get("sn", 0) or 0) == 0 and (edge.get("tn", 0) or 0) == 0


def effective_weight(edge: dict[str, Any], *, current_seq: int = 0) -> int:
    """H4：effective_weight = strength × rate（纯整·×1000 缩放）。

    current_seq：当前 timestamp_seq（OCCURRENCE 衰减 logical_age = current_seq −
      memory_time_attach 用·无墙钟·audit_event 自增序）。非 OCCURRENCE 边忽略。
    """
    et = edge["edge_type"]
    strength = edge.get("strength", 0) or 0
    if et == EDGE_PRECEDES:
        # 序边结构真值·strength 恒 = 1（§7.1·reward 永不调）
        assert strength == 1, f"PRECEDES strength 须恒=1·got {strength}"
        return 1
    if et == EDGE_CAUSES:
        total = (edge.get("sn", 0) or 0) + (edge.get("tn", 0) or 0)
        if total == 0:
            return 0   # 无观测·零权重（待 reward）
        return strength * edge_rate(edge)   # strength × rate（rate 0..1000·×1000 缩放）
    if et == EDGE_REFERS_TO and edge.get("subtype") == SUBTYPE_OCCURRENCE:
        # F3·cross-space 结构锚·occurrence 时序衰减·非学习对象不接 reward 无 rate 乘子
        attach = edge.get("memory_time_attach")
        logical_age = (current_seq - attach) if attach is not None else 0
        if logical_age < 0:
            logical_age = 0
        w = strength * DECAY_K - logical_age
        return w if w > 0 else 0   # I5 floor 防负权重
    # COOCCURS SHADOW 不进 PR 传播（卷一隔离·本函数不应被 COOCCURS 调用）
    assert et in (EDGE_PRECEDES, EDGE_CAUSES, EDGE_REFERS_TO), (
        f"effective_weight: PR 只读 {{PRECEDES,CAUSES,REFERS_TO}}·got et={et}"
    )
    return 0   # REFERS_TO 非 OCCURRENCE（PURE_ALIAS/METAPHOR）不进 PR 邻接
