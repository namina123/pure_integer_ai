"""阶段 B 纵切：把已恢复 Core active relation 接到 QueryState 的多跳查询桥。

消费 TrainedRelationGraphRuntime 的 active surface facts（命题/角色/表层）与
QueryState 协议：输入 token 先与已学习 binding 表层做连续匹配形成概念 anchor
（表层只作为 realization 匹配入口，语义沿概念边扩展），然后按 filler 反查相邻
命题形成多跳 frontier，直到综合终止谓词收敛。本模块不读取课程、QA 或私有标签。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pure_integer_ai.cognition.shared.query_driver import (
    ExpansionOutcome,
    QueryExpander,
    run_query,
)
from pure_integer_ai.cognition.shared.query_state import (
    SPACE_CORE,
    TERMINATION_ANSWER_CLOSED,
    TERMINATION_CLARIFY_MISSING_BINDING,
    TERMINATION_NO_FRONTIER,
    BindingEntry,
    FrontierEntry,
    QueryAnchor,
    QueryBudget,
    QueryState,
    VisitedKey,
    seed_query,
)
from pure_integer_ai.experiments.trained_relation_graph_runtime import (
    ActiveRelationSurface,
    TrainedRelationGraphRuntime,
)

_TRACE_FORMAT = "PURE_INTEGER_TRAINED_GRAPH_QUERY_TRACE_V1"


@dataclass(frozen=True, slots=True)
class CoreHop:
    """一次概念级多跳路径步骤（命题、角色方向、端点 filler 与表层）。"""

    proposition: tuple[int, ...]
    predicate: tuple[int, ...]
    from_filler: tuple[int, ...]
    to_filler: tuple[int, ...]
    surface: str
    depth: int


@dataclass(frozen=True, slots=True)
class ConceptAnchor:
    """输入表层匹配到的概念 filler 及其角色表层。"""

    filler: tuple[int, ...]
    surface: str
    kind: int


class TrainedCoreQueryRuntime:
    """只读恢复 active relation 并执行概念级多跳查询的桥接运行时。"""

    def __init__(self, database: str | Path) -> None:
        self.runtime = TrainedRelationGraphRuntime(database)
        self.facts = self.runtime.active_surface_facts()
        self.surface_to_filler: dict[str, tuple[tuple[int, ...], ...]] = {}
        self.filler_surface: dict[tuple[int, ...], str] = {}
        for fact in self.facts:
            for binding in fact.bindings:
                surface = binding.surface
                filler = binding.filler.stable_key()
                if not surface.strip():
                    continue
                current = self.surface_to_filler.get(surface, ())
                if filler not in current:
                    self.surface_to_filler[surface] = current + (filler,)
                self.filler_surface.setdefault(filler, surface)
        self.filler_edges: dict[tuple[int, ...], tuple[ActiveRelationSurface, ...]] = {}
        for fact in self.facts:
            seen: set[tuple[int, ...]] = set()
            for binding in fact.bindings:
                key = binding.filler.stable_key()
                if key in seen:
                    continue
                seen.add(key)
                self.filler_edges.setdefault(key, ())
                if fact not in self.filler_edges[key]:
                    self.filler_edges[key] = self.filler_edges[key] + (fact,)

    def close(self) -> None:
        """关闭只读桥接 owner。"""
        self.runtime.close()

    def __enter__(self) -> "TrainedCoreQueryRuntime":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    def anchor_concepts(self, surface: str) -> tuple[ConceptAnchor, ...]:
        """把输入表层与已学习 binding 表层匹配，产生概念 anchor 候选。

        只做连续包含匹配并保留全部候选；不按长度、字符形状或频次选句。
        同一表层对应多个 filler 时全部保留，交由绑定/证据裁决。
        """
        result: dict[tuple[int, ...], ConceptAnchor] = {}
        for learned, fillers in self.surface_to_filler.items():
            if not learned or learned not in surface:
                continue
            for filler in fillers:
                if filler not in result:
                    result[filler] = ConceptAnchor(filler, learned, 1)
        return tuple(sorted(
            result.values(), key=lambda item: item.filler))

    def query(self, surface: str, *, minimum_depth: int = 1,
              max_depth: int = 4) -> dict[str, object]:
        """执行一次概念级多跳查询并返回可重放 trace。"""
        anchors = self.anchor_concepts(surface)
        query_key = tuple(ord(value) for value in surface)
        if not anchors:
            state = seed_query(
                query_key, minimum_depth=minimum_depth,
                budget=QueryBudget(max_depth=max_depth),
            )
            return self._finalize(
                state, (), termination=TERMINATION_CLARIFY_MISSING_BINDING,
                reason="NO_CONCEPT_ANCHOR")
        initial = seed_query(
            query_key,
            anchors=tuple(QueryAnchor(
                SPACE_CORE, item.filler, kind=1, scope_key=()) for item in anchors),
            minimum_depth=minimum_depth,
            budget=QueryBudget(max_depth=max_depth),
        )
        frontier = tuple(
            FrontierEntry(
                (SPACE_CORE, *item.filler), SPACE_CORE,
                depth=0, required_slot_gain=100)
            for item in anchors)
        expander = _CoreGraphExpander(self)
        state = run_query(
            initial.with_(frontier=frontier), expander=expander)
        hops = tuple(sorted(
            expander.hops, key=lambda item: (item.depth, item.proposition)))
        return self._finalize(
            state, hops, termination=state.termination,
            reason=_TERMINATION_NAMES.get(state.termination, "OPEN"))

    @staticmethod
    def _finalize(state: QueryState, hops: tuple[CoreHop, ...], *,
                  termination: int, reason: str) -> dict[str, object]:
        """把最终状态、多跳路径与终止原因组装为可重放 trace。"""
        bindings = tuple(
            {"role": list(item.role_key), "filler": list(item.filler_key),
             "space": item.space, "conflict_kept": item.conflict_kept}
            for item in state.bindings)
        return {
            "format": _TRACE_FORMAT,
            "schema_version": 1,
            "termination": termination,
            "termination_reason": reason,
            "depth": state.depth,
            "minimum_depth": state.minimum_depth,
            "node_count": state.node_count,
            "edge_count": state.edge_count,
            "read_count": state.read_count,
            "score": state.score,
            "hops": [
                {
                    "proposition": list(item.proposition),
                    "predicate": list(item.predicate),
                    "from_filler": list(item.from_filler),
                    "to_filler": list(item.to_filler),
                    "surface": item.surface,
                    "depth": item.depth,
                }
                for item in hops
            ],
            "bindings": bindings,
            "query_key": list(state.query_key),
            "state_key": list(state.stable_key()),
        }


_TERMINATION_NAMES = {
    TERMINATION_ANSWER_CLOSED: "ANSWER_CLOSED",
    TERMINATION_CLARIFY_MISSING_BINDING: "CLARIFY_MISSING_BINDING",
    TERMINATION_NO_FRONTIER: "NO_FRONTIER",
    5: "BUDGET_EXHAUSTED",
    6: "CYCLE_GUARD",
    7: "MARGINAL_CONVERGED",
}


class _CoreGraphExpander(QueryExpander):
    """沿 active relation 的 filler 邻接关系扩展；visited 按节点防环。

    每次扩展只处理当前最高优先级边：新邻边并入剩余 frontier（不覆盖），
    保证同一查询能探索全部可达概念而不是只沿一条链。边权重用整数
    required_slot_gain，等权时按 edge_key 稳定决胜。
    """

    def __init__(self, owner: TrainedCoreQueryRuntime) -> None:
        self.owner = owner
        self.hops: list[CoreHop] = []

    def expand(self, state: QueryState,
               edge: FrontierEntry) -> ExpansionOutcome:
        remaining = tuple(item for item in state.frontier if item != edge)
        origin = tuple(edge.edge_key[1:])
        if origin in {item.node_key for item in state.visited}:
            return ExpansionOutcome(
                state.with_(frontier=remaining), edge, 0)
        facts = self.owner.filler_edges.get(origin, ())
        if not facts:
            return ExpansionOutcome(
                state.with_(frontier=remaining), edge, 0)
        depth = edge.depth + 1
        # visited 只记录已作为 origin 扩展过的节点；邻居入队但不在本轮
        # 标记为已访问，否则第二跳会被误判为环而无法继续。
        visited_set = {item.node_key for item in state.visited}
        visited_set.add(origin)
        next_edges: list[FrontierEntry] = []
        new_bindings = list(state.bindings)
        hops: list[CoreHop] = []
        discovered: set[tuple[int, ...]] = set()
        for fact in facts:
            proposition = fact.proposition.stable_key()
            predicate = fact.predicate.stable_key()
            for binding in fact.bindings:
                filler = binding.filler.stable_key()
                if filler == origin or filler in visited_set:
                    continue
                if filler in discovered:
                    continue
                discovered.add(filler)
                new_bindings.append(BindingEntry(
                    role_key=proposition,
                    filler_key=filler,
                    space=SPACE_CORE,
                ))
                hops.append(CoreHop(
                    proposition, predicate, origin, filler,
                    binding.surface, depth))
                next_edges.append(FrontierEntry(
                    (SPACE_CORE, *filler), SPACE_CORE,
                    depth=depth, required_slot_gain=50,
                    evidence_support=1,
                ))
        self.hops.extend(hops)
        visited = tuple(sorted(
            (VisitedKey(SPACE_CORE, key, direction=0)
             for key in visited_set),
            key=lambda item: item.stable_key()))
        return ExpansionOutcome(
            state.with_(
                depth=max(state.depth, depth),
                frontier=tuple(sorted(
                    (*remaining, *next_edges),
                    key=lambda entry: entry.priority_tuple(), reverse=True)),
                visited=visited,
                bindings=tuple(new_bindings),
                node_count=state.node_count + len(facts),
                edge_count=state.edge_count + len(hops),
                read_count=state.read_count + len(facts),
                score=state.score + len(hops),
            ),
            edge,
            len(facts),
        )


__all__ = [
    "ConceptAnchor",
    "CoreHop",
    "TrainedCoreQueryRuntime",
]
