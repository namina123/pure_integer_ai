"""共同查询前沿的确定性根参与调度；不读图、不改状态或预算。"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.query_state import (
    ROOT_PENDING,
    FrontierEntry,
    QueryRoot,
)


def select_frontier_entry(
        frontier: tuple[FrontierEntry, ...],
        roots: tuple[QueryRoot, ...],
        ) -> FrontierEntry:
    """先扩展待参与根的现有种子，同层沿用原规范优先顺序。

    frontier须为调用方已经按原合同排序且排除visited的完整视图。
    只选择实际目标等于待参与根的原边；后代边不能代替根种子。
    原frontier、根状态、全部来源及计数均由调用方和实际扩展器保留。
    """
    if not frontier:
        raise ValueError("选择查询边需要非空未访问frontier")
    pending = {(root.owner_space, root.root_key)
               for root in roots if root.status == ROOT_PENDING}
    if pending:
        for entry in frontier:
            if ((entry.owner_space, entry.root_key) in pending
                    and (entry.target_key or entry.edge_key) == entry.root_key):
                return entry
    return frontier[0]
