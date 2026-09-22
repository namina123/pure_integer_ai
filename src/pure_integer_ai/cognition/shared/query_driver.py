"""统一深度查询驱动：seed -> expand -> bind/filter -> visited -> terminate。

驱动本身不读任何存储；扩展器（Core/Memory/Dialogue 各自实现）把当前
frontier 的最高优先级边解析为下一层候选并回写 QueryState。深度由数据和
综合终止谓词决定，禁止固定层数或首跳即出。
"""
from __future__ import annotations

from dataclasses import dataclass, replace
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
    evaluate_termination,
    from_query_state,
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


def unvisited_frontier(frontier: tuple[FrontierEntry, ...],
                       visited: tuple[VisitedKey, ...]) -> tuple[FrontierEntry, ...]:
    """派生可扩展边，供终止、生成及采用共用；不改写完整 frontier 证据。"""
    seen = {
        (item.space, item.node_key, item.direction)
        for item in visited
    }
    return tuple(
        entry for entry in frontier
        if (
            entry.owner_space,
            entry.target_key if entry.target_key else entry.edge_key,
            entry.direction,
        ) not in seen
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
        frontier = unvisited_frontier(state.frontier, state.visited)
        # 已访问边仍保留在 trace，但不能假装还有待求证的 frontier。
        # 有限图正常收敛后，缺槽/冲突必须交给综合终止器，不伪报执行循环。
        predicate = replace(from_query_state(state), frontier_count=len(frontier))
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
        if not frontier:
            # All remaining entries may simply be already-visited duplicate
            # edges after a finite graph converges.  Re-evaluate the terminal
            # predicates with an empty effective frontier before declaring a
            # cycle; otherwise a closed answer reached on the last edge is
            # incorrectly downgraded to CYCLE_GUARD.
            terminal_predicate = replace(
                from_query_state(state), frontier_count=0)
            terminal_reason = evaluate_termination(
                terminal_predicate, budgets=budgets)
            if terminal_reason != TERMINATION_OPEN:
                final = state.with_(termination=terminal_reason)
            else:
                final = state.with_(termination=6, cycle_hit=1)  # CYCLE_GUARD
            if tracer is not None:
                tracer.record(final, final.termination)
            return final
        edge = frontier[0]
        outcome = expander.expand(state, edge)
        if not isinstance(outcome, ExpansionOutcome):
            raise TypeError("expander 必须返回 ExpansionOutcome")
        next_state = outcome.state
        # 专用扩展器可以承诺已按 QueryState 规范顺序返回，避免每扩一条边
        # 再全量复制并排序 frontier/binding/evidence/visited。未声明的通用
        # 扩展器仍走原 canonical 边界，协议行为不放宽。
        if getattr(expander, "canonical_output", 0) != 1:
            next_state = next_state.canonical()
        # 等价状态防护：两侧已 canonical，直接使用不可变 dataclass 的
        # 逐字段结构比较，避免为巨型 Evidence 集合重复构造扁平稳定键。
        # stable_key 协议仍保留给跨语言 trace/持久化；这里只需要等价
        # 判定，不能让派生编码的临时分配改变查询可运行性。
        if next_state == state:
            remaining = tuple(item for item in frontier if item != edge)
            state = state.with_(frontier=remaining, cycle_hit=1)
            continue
        state = next_state
        if tracer is not None:
            tracer.record(state, TERMINATION_OPEN)


__all__ = [
    "ExpansionOutcome",
    "QueryExpander",
    "QueryTracer",
    "run_query",
    "unvisited_frontier",
]
