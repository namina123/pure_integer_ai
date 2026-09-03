"""统一深度查询驱动：seed -> expand -> bind/filter -> visited -> terminate。

驱动本身不读任何存储；扩展器（Core/Memory/Dialogue 各自实现）把当前
frontier 的最高优先级边解析为下一层候选并回写 QueryState。深度由数据和
综合终止谓词决定，禁止固定层数或首跳即出。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from pure_integer_ai.cognition.shared.query_state import (
    TERMINATION_ANSWER_CLOSED,
    TERMINATION_OPEN,
    FrontierEntry,
    QueryBudget,
    QueryState,
    VisitedKey,
)
from pure_integer_ai.cognition.shared.query_termination import (
    TerminationPredicateInput,
    evaluate_termination,
)


@dataclass(frozen=True, slots=True)
class ExpansionOutcome:
    """一次扩展后的新状态与扩展侧证据。"""

    state: QueryState
    expanded_edge: FrontierEntry | None = None
    reads: int = 0


class QueryExpander(Protocol):
    """把一个 frontier 边扩展为下一层状态。"""

    def expand(self, state: QueryState,
               edge: FrontierEntry) -> ExpansionOutcome:
        """返回扩展后的状态；无法扩展时必须把该边从 frontier 移除。"""
        ...


class QueryTracer(Protocol):
    """记录每一步节点/边/深度/终止原因的内部整数 trace。"""

    def record(self, state: QueryState, reason: int) -> None:
        """记录一次状态推进（含终止）。"""
        ...


def _dedupe_frontier(frontier: tuple[FrontierEntry, ...],
                     visited: tuple[VisitedKey, ...]) -> tuple[FrontierEntry, ...]:
    """去掉已访问等价状态对应的边；等价键由 VisitedKey 稳定序列化。"""
    seen = {item.stable_key() for item in visited}
    return tuple(
        entry for entry in frontier
        if (entry.owner_space, entry.edge_key, 0) not in seen
    )


def run_query(
        state: QueryState,
        *,
        expander: QueryExpander,
        tracer: QueryTracer | None = None,
        ) -> QueryState:
    """执行文档 11.2 深度扩展循环，返回带终止原因的最终状态。

    循环只依赖纯整数状态与协议终止谓词；没有可扩展边或预算耗尽时按协议
    返回 clarify/fail 原因，不猜测、不近邻、不随机。
    """
    budget = state.budget
    budgets = budget.stable_key()
    state = state.canonical()
    while True:
        predicate = TerminationPredicateInput(
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
        reason = evaluate_termination(predicate, budgets=budgets)
        if reason != TERMINATION_OPEN:
            final = state.with_(termination=reason)
            if tracer is not None:
                tracer.record(final, reason)
            return final
        if not state.frontier:
            final = state.with_(termination=4)  # NO_FRONTIER
            if tracer is not None:
                tracer.record(final, 4)
            return final
        frontier = _dedupe_frontier(state.frontier, state.visited)
        if not frontier:
            final = state.with_(termination=6)  # CYCLE_GUARD
            if tracer is not None:
                tracer.record(final, 6)
            return final
        edge = frontier[0]
        outcome = expander.expand(state, edge)
        if not isinstance(outcome, ExpansionOutcome):
            raise TypeError("expander 必须返回 ExpansionOutcome")
        next_state = outcome.state
        next_state = next_state.canonical()
        # 等价状态防护：若扩展没有改变状态摘要，则把该边丢弃并继续。
        if next_state.stable_key() == state.stable_key():
            remaining = tuple(item for item in frontier if item != edge)
            state = state.with_(frontier=remaining)
            continue
        state = next_state
        if tracer is not None:
            tracer.record(state, TERMINATION_OPEN)


__all__ = [
    "ExpansionOutcome",
    "QueryExpander",
    "QueryTracer",
    "run_query",
]
