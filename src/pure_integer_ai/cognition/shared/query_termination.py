"""综合终止谓词：结构/证据闭合、竞争收敛、最小深度、边际、冲突与预算。

终止不是固定层数；每个 release 声明最大节点/边/读取/深度预算，请求状态在
每轮扩展后重新计算终止原因。本模块只做纯整数判定，不读存储；候选证据、
绑定覆盖与冲突情况由调用方在 QueryState 中提供。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.query_state import (
    TERMINATION_ANSWER_CLOSED,
    TERMINATION_BUDGET_EXHAUSTED,
    TERMINATION_CLARIFY_CONFLICT,
    TERMINATION_CLARIFY_MISSING_BINDING,
    TERMINATION_CYCLE_GUARD,
    TERMINATION_MARGINAL_CONVERGED,
    TERMINATION_NO_FRONTIER,
    TERMINATION_OPEN,
    QueryState,
)


@dataclass(frozen=True, slots=True)
class TerminationPredicateInput:
    """扩展器每轮提供给终止判定的整数状态摘要。

    all fields are nonnegative integers or boolean flags encoded as ints;
    no host object, wall-clock value, or language string participates.
    """

    depth: int
    minimum_depth: int
    node_count: int
    edge_count: int
    read_count: int
    frontier_count: int
    required_slots_open: int = 0
    evidence_closed: int = 0  # 1=所有必要端点均有活动来源与可核验关系
    conflict_open: int = 0    # 1=存在未裁决冲突
    best_score: int = 0
    second_score: int = 0
    marginal_gain: int = 0    # 新增深度带来的支持/覆盖增量（整数）
    cycle_hit: int = 0        # 1=最近扩展命中等价 visited 状态
    generation_ready: int = 0  # 1=存在可填充生成结构且结果 token 可由图重建

    def __post_init__(self) -> None:
        for name in (
                "depth", "minimum_depth", "node_count", "edge_count",
                "read_count", "frontier_count", "required_slots_open",
                "evidence_closed", "conflict_open", "best_score",
                "second_score", "marginal_gain", "cycle_hit",
                "generation_ready"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"TerminationPredicateInput.{name} 非法")


def _structure_closed(state: TerminationPredicateInput) -> bool:
    """结构闭合：问题类型/实体/关系/角色/限定均有绑定。"""
    return state.required_slots_open == 0


def _evidence_closed(state: TerminationPredicateInput) -> bool:
    """证据闭合：候选必要端点来源可核验。"""
    return state.evidence_closed == 1


def _conflict_resolved(state: TerminationPredicateInput) -> bool:
    """冲突闭合：无未裁决冲突或已明确需要澄清。"""
    return state.conflict_open == 0


def _minimum_depth_satisfied(state: TerminationPredicateInput) -> bool:
    """强制最小思考深度已满足。"""
    return state.depth >= state.minimum_depth


def _competitive_margin(state: TerminationPredicateInput) -> bool:
    """最佳相对次佳达到策略门槛（整数边界）。"""
    if state.second_score <= 0:
        return True
    return (state.best_score - state.second_score) * 1000 >= (
        max(1, state.second_score) * 25)


def _marginal_converged(state: TerminationPredicateInput) -> bool:
    """新增深度边际收益为零或为负。"""
    return state.marginal_gain <= 0


def _generation_verifiable(state: TerminationPredicateInput) -> bool:
    """生成可验证：每个输出槽都有整数图依据。"""
    return state.generation_ready == 1


def _budget_exhausted(state: TerminationPredicateInput,
                      budgets: tuple[int, int, int, int]) -> bool:
    """预算/循环：节点、边、读取、最大安全深度任一越界。"""
    max_nodes, max_edges, max_reads, max_depth = budgets
    return (state.node_count > max_nodes
            or state.edge_count > max_edges
            or state.read_count > max_reads
            or state.depth > max_depth)


def evaluate_termination(
        state: TerminationPredicateInput,
        *,
        budgets: tuple[int, int, int, int],
        ) -> int:
    """按文档 11.3 顺序评估综合终止谓词，返回协议整数原因。

    优先级：循环/预算最先（安全边界），随后结构缺失/冲突需要澄清，然后
    证据/生成闭合与收敛裁决回答，最后无 frontier 与边际收敛收尾。
    """
    if state.cycle_hit == 1:
        return TERMINATION_CYCLE_GUARD
    if _budget_exhausted(state, budgets):
        return TERMINATION_BUDGET_EXHAUSTED
    if not _structure_closed(state):
        return TERMINATION_CLARIFY_MISSING_BINDING
    if not _conflict_resolved(state):
        return TERMINATION_CLARIFY_CONFLICT
    if not _evidence_closed(state):
        # 证据未闭合不是澄清：只要预算允许就继续扩展。
        return TERMINATION_OPEN
    if (state.depth < state.minimum_depth
            or not _generation_verifiable(state)):
        return TERMINATION_OPEN
    if _competitive_margin(state) and _generation_verifiable(state):
        return TERMINATION_ANSWER_CLOSED
    if state.frontier_count == 0:
        return TERMINATION_NO_FRONTIER
    if _marginal_converged(state) and _minimum_depth_satisfied(state):
        return TERMINATION_MARGINAL_CONVERGED
    return TERMINATION_OPEN


def from_query_state(state: QueryState) -> TerminationPredicateInput:
    """从统一 QueryState 派生终止判定输入（纯整数投影）。"""
    return TerminationPredicateInput(
        depth=state.depth,
        minimum_depth=state.minimum_depth,
        node_count=state.node_count,
        edge_count=state.edge_count,
        read_count=state.read_count,
        frontier_count=len(state.frontier),
        required_slots_open=0,
        evidence_closed=0,
        conflict_open=0,
        best_score=state.score,
        second_score=0,
        marginal_gain=0,
        cycle_hit=0,
        generation_ready=0,
    )


__all__ = [
    "TerminationPredicateInput",
    "evaluate_termination",
    "from_query_state",
]
