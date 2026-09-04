"""阶段 B 纵切：把已恢复 Core active relation 接到 QueryState 的多跳查询桥。

本模块实现 CrossSpaceQueryExpander：把当前 Frontier 边同时喂给 Core（active
relation filler 邻接）与 Memory（会话 Observation 邻接），让查询状态跨越
Core 与 Interaction Memory 两类图。每类图只在 active_spaces 声明时才允许
扩展；visited/edge/evidence/终止完全由 QueryState 协议裁决。

Memory 侧仍受现有 trained_dialogue_memory_graph 限制：会话观察只保存表层
Observation，尚未形成实体/属性/时间结构候选（阶段 D 任务）。因此跨空间桥
把 Memory 作为 discourse 焦点邻接（连续轮次表面），并用整数 turn/speaker
身份隔离；一旦 Observation→Hypothesis 结构候选在阶段 D 就位，同一个桥只需
替换 Memory expander 即可接入结构边，不需要改驱动/终止协议。
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
    SPACE_MEMORY,
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
from pure_integer_ai.experiments.trained_dialogue_memory_graph import (
    TrainedDialogueMemoryGraph,
)
from pure_integer_ai.experiments.trained_relation_graph_runtime import (
    ActiveRelationSurface,
    TrainedRelationGraphRuntime,
)

_TRACE_FORMAT = "PURE_INTEGER_TRAINED_GRAPH_QUERY_TRACE_V1"

_CORE = SPACE_CORE
_MEMORY = SPACE_MEMORY


@dataclass(frozen=True, slots=True)
class QueryHop:
    """一次多跳路径步骤（图空间、命题、方向、端点、表层与支持来源）。"""

    space: int
    proposition: tuple[int, ...]
    predicate: tuple[int, ...]
    from_filler: tuple[int, ...]
    to_filler: tuple[int, ...]
    surface: str
    depth: int
    source_hash: int


@dataclass(frozen=True, slots=True)
class ConceptAnchor:
    """输入表层匹配到的概念 filler 及其角色表层。"""

    filler: tuple[int, ...]
    surface: str
    kind: int


class TrainedGraphQueryBridge:
    """跨 Core 与 Interaction Memory 的统一图查询桥。

    只读恢复 active relation（Core）并挂接会话 Memory（外部 session 库）；
    查询时 active_spaces 决定哪些图参与扩展。Memory 为空或不可读时自动
    退化为纯 Core 查询，不伪造 Memory 证据。
    """

    def __init__(
            self,
            database: str | Path,
            *,
            memory_database: str | Path | None = None,
            tenant_id: int = 1,
            user_id: int = 1,
            session_id: int = 1,
            ) -> None:
        self.core_runtime = TrainedRelationGraphRuntime(database)
        self.memory = None
        self._owns_memory = memory_database is not None
        if memory_database is not None:
            self.memory = TrainedDialogueMemoryGraph(
                memory_database,
                tenant_id=tenant_id,
                user_id=user_id,
                session_id=session_id,
            )
        facts = self.core_runtime.active_surface_facts()
        self.facts = facts
        self.surface_to_filler: dict[str, tuple[tuple[int, ...], ...]] = {}
        self.filler_surface: dict[tuple[int, ...], str] = {}
        for fact in facts:
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
        for fact in facts:
            seen: set[tuple[int, ...]] = set()
            for binding in fact.bindings:
                key = binding.filler.stable_key()
                if key in seen:
                    continue
                seen.add(key)
                self.filler_edges.setdefault(key, ())
                if fact not in self.filler_edges[key]:
                    self.filler_edges[key] = self.filler_edges[key] + (fact,)
        # Memory 焦点：会话最近轮次的表层观察，用于 discourse 邻接扩展。
        self._memory_turns: tuple[tuple[int, int, str, tuple[int, ...]], ...] = ()
        if self.memory is not None:
            self._memory_turns = self._restore_memory_focus()
        # Core predicate stable key -> fact（供结构生成反查）。
        self.core_fact_index: dict[tuple[int, ...], ActiveRelationSurface] = {}
        for fact in facts:
            self.core_fact_index[fact.predicate.stable_key()] = fact

    def close(self) -> None:
        """关闭只读 Core 与可选 Memory owner。"""
        self.core_runtime.close()
        if self.memory is not None:
            self.memory.close()
            self.memory = None

    def __enter__(self) -> "TrainedGraphQueryBridge":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    def _restore_memory_focus(self) -> tuple[tuple[int, int, str, tuple[int, ...]], ...]:
        """从会话 Memory 恢复最近 turn 表层及其码点键（阶段 D 前仅此证据）。"""
        rows = []
        if self.memory is None:
            return ()
        for item in self.memory.recent_turns(limit=16):
            key = tuple(ord(value) for value in item.surface)
            rows.append((item.turn_seq, item.speaker_kind, item.surface, key))
        return tuple(rows)

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
        """执行一次跨空间多跳查询并返回可重放 trace。"""
        anchors = self.anchor_concepts(surface)
        query_key = tuple(ord(value) for value in surface)
        if not anchors:
            state = seed_query(
                query_key, minimum_depth=minimum_depth,
                budget=QueryBudget(max_depth=max_depth),
            )
            return self._finalize(
                state, (), termination=TERMINATION_CLARIFY_MISSING_BINDING,
                reason="CLARIFY_MISSING_BINDING")
        active = (_CORE,) if self.memory is None else (_CORE, _MEMORY)
        initial = seed_query(
            query_key,
            anchors=tuple(QueryAnchor(
                _CORE, item.filler, kind=1, scope_key=()) for item in anchors),
            minimum_depth=minimum_depth,
            budget=QueryBudget(max_depth=max_depth),
            active_spaces=active,
        )
        frontier = tuple(
            FrontierEntry(
                (_CORE, *item.filler), _CORE,
                depth=0, required_slot_gain=100)
            for item in anchors)
        if self.memory is not None:
            for turn_seq, speaker, _surface, key in self._memory_turns:
                if not key:
                    continue
                frontier += (FrontierEntry(
                    (_MEMORY, turn_seq, speaker, *key), _MEMORY,
                    depth=0, required_slot_gain=60,
                    recency_weight=max(0, 10 - turn_seq),
                ),)
        expander = _CrossSpaceExpander(self)
        state = run_query(
            initial.with_(frontier=frontier), expander=expander)
        hops = tuple(sorted(
            expander.hops, key=lambda item: (item.depth, item.space,
                                             item.proposition)))
        # 结构生成：若 Core 侧证据闭合，把已绑定命题交给槽位重填，产出
        # 新表层（非整句回放）。生成失败/无闭合命题时返回 None，不猜。
        generation = None
        if hops and state.termination in {
                TERMINATION_ANSWER_CLOSED, TERMINATION_NO_FRONTIER}:
            generation = self._generate_from_closed(hops, anchors)
        # 阶段 C：把闭合命题经 ResponsePlan 组织并执行 token postcheck。生成
        # dict 只作为可重放 trace；真实发布承重回答从 ResponsePlan 渲染。若
        # plan 无法通过 postcheck（无结构证据/必填槽缺 token），则回答保持
        # None 由上层 fail-closed，不退回旧表层回放。
        response_plan = self._response_plan_from_closed(hops, anchors)
        return self._finalize(
            state, hops, termination=state.termination,
            reason=_TERMINATION_NAMES.get(state.termination, "OPEN"),
            generation=generation,
            response_plan=(None if response_plan is None
                           else response_plan.to_dict()))

    def _best_closed_fact(
            self,
            hops: tuple[QueryHop, ...],
            ) -> tuple[ActiveRelationSurface, object] | None:
        """从 Core hops 反查可生成命题；无闭合事实返回 None。"""
        core_hops = tuple(item for item in hops if item.space == _CORE)
        for hop in core_hops:
            if hop.predicate not in self.core_fact_index:
                continue
            fact = self.core_fact_index[hop.predicate]
            try:
                generated = self.core_runtime._generate_surface(fact)
            except RuntimeError:
                continue
            if not generated.surface.strip():
                continue
            return fact, generated
        return None

    def _response_plan_from_closed(
            self,
            hops: tuple[QueryHop, ...],
            anchors: tuple[ConceptAnchor, ...],
            ) -> object | None:
        """从已闭合 Core 事实构造可验证 ResponsePlan（阶段 C 发布承重路径）。

        只接受通过 token postcheck 的 plan；失败返回 None 由上层 fail-closed，
        不退回任何旧表层/近邻/整句回放。anchor 表层只作为 realization 证据。
        """
        from pure_integer_ai.experiments.generation_organization import (
            plan_from_active_fact,
        )
        found = self._best_closed_fact(hops)
        if found is None:
            return None
        fact, generated = found
        try:
            return plan_from_active_fact(fact, generated)
        except (TypeError, ValueError):
            return None

    def _generate_from_closed(
            self,
            hops: tuple[QueryHop, ...],
            anchors: tuple[ConceptAnchor, ...],
            ) -> dict[str, object] | None:
        """从已闭合的 Core 多跳 claim 生成组织后的表层。

        槽位填充复用 runtime 的 _generate_surface：把命题的 RoleBinding 表层
        重填到同 predicate/role 的生成框架，产出由图内 token 组合的新句子。
        anchor 输入表层不作为槽位，只作为 realization 证据。
        """
        found = self._best_closed_fact(hops)
        if found is None:
            return None
        _fact, best = found
        return {
            "kind": "slot_fill_generation",
            "surface": best.surface,
            "frame_proposition": list(best.frame_proposition.stable_key()),
            "slot_count": best.slot_count,
            "anchor_surface": anchors[0].surface if anchors else "",
        }

    @staticmethod
    def _finalize(state: QueryState, hops: tuple[QueryHop, ...], *,
                  termination: int, reason: str,
                  generation: dict[str, object] | None = None,
                  response_plan: dict[str, object] | None = None,
                  ) -> dict[str, object]:
        """把最终状态、多跳路径与终止原因组装为可重放 trace。"""
        bindings = tuple(
            {"role": list(item.role_key), "filler": list(item.filler_key),
             "space": item.space, "conflict_kept": item.conflict_kept}
            for item in state.bindings)
        evidence = tuple(
            {"source_hash": hop.source_hash,
             "space": hop.space,
             "proposition": list(hop.proposition),
             "surface": hop.surface}
            for hop in hops)
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
                    "space": item.space,
                    "proposition": list(item.proposition),
                    "predicate": list(item.predicate),
                    "from_filler": list(item.from_filler),
                    "to_filler": list(item.to_filler),
                    "surface": item.surface,
                    "depth": item.depth,
                    "source_hash": item.source_hash,
                }
                for item in hops
            ],
            "evidence": evidence,
            "bindings": bindings,
            "generation": generation,
            "response_plan": response_plan,
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


class _CrossSpaceExpander(QueryExpander):
    """把 frontier 边按其 owner space 分发给 Core 或 Memory 扩展器。

    visited 只记录已作为 origin 扩展过的节点；新邻居入队但不在本轮标记为
    已访问，否则第二跳会被误判为环。边权重用整数 required_slot_gain，
    等权时按 edge_key 稳定决胜。
    """

    def __init__(self, owner: TrainedGraphQueryBridge) -> None:
        self.owner = owner
        self.hops: list[QueryHop] = []

    def expand(self, state: QueryState,
               edge: FrontierEntry) -> ExpansionOutcome:
        remaining = tuple(item for item in state.frontier if item != edge)
        origin = tuple(edge.edge_key[1:])
        if origin in {item.node_key for item in state.visited}:
            return ExpansionOutcome(
                state.with_(frontier=remaining), edge, 0)
        space = edge.owner_space
        if space == _CORE:
            outcome = self._expand_core(state, remaining, edge, origin)
        elif space == _MEMORY:
            outcome = self._expand_memory(state, remaining, edge, origin)
        else:
            raise ValueError(f"query bridge 遇到未注册空间: {space}")
        return outcome

    def _expand_core(self, state: QueryState, remaining: tuple[FrontierEntry, ...],
                     edge: FrontierEntry,
                     origin: tuple[int, ...]) -> ExpansionOutcome:
        facts = self.owner.filler_edges.get(origin, ())
        if not facts:
            return ExpansionOutcome(
                state.with_(frontier=remaining), edge, 0)
        depth = edge.depth + 1
        visited_set = {item.node_key for item in state.visited}
        visited_set.add(origin)
        next_edges: list[FrontierEntry] = []
        new_bindings = list(state.bindings)
        hops: list[QueryHop] = []
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
                    space=_CORE,
                ))
                hops.append(QueryHop(
                    _CORE, proposition, predicate, origin, filler,
                    binding.surface, depth, fact.source_hash))
                next_edges.append(FrontierEntry(
                    (_CORE, *filler), _CORE,
                    depth=depth, required_slot_gain=50,
                    evidence_support=1,
                ))
        self.hops.extend(hops)
        visited = tuple(sorted(
            (VisitedKey(_CORE, key, direction=0) for key in visited_set),
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

    def _expand_memory(self, state: QueryState, remaining: tuple[FrontierEntry, ...],
                       edge: FrontierEntry,
                       origin: tuple[int, ...]) -> ExpansionOutcome:
        """Memory 焦点邻接：把当前 turn 连接到相邻轮次（阶段 D 前仅表层证据）。

        只有在 active_spaces 包含 Memory 且当前 Memory 有焦点时才扩展。
        每条连接都有 source_ref（memory source）与 turn 身份，供阶段 D 后
        替换为实体/事件结构候选。
        """
        if self.owner.memory is None:
            return ExpansionOutcome(
                state.with_(frontier=remaining), edge, 0)
        depth = edge.depth + 1
        visited_set = {item.node_key for item in state.visited}
        visited_set.add(origin)
        next_edges: list[FrontierEntry] = []
        new_bindings = list(state.bindings)
        hops: list[QueryHop] = []
        turn_seq = origin[0]
        neighbor_candidates = tuple(
            item for item in self.owner._memory_turns
            if item[0] != turn_seq)
        if not neighbor_candidates:
            return ExpansionOutcome(
                state.with_(frontier=remaining), edge, 0)
        for neighbor in neighbor_candidates:
            neighbor_seq, speaker, surface, key = neighbor
            neighbor_key = (_MEMORY, neighbor_seq, speaker, *key)
            if (neighbor_seq, speaker) in visited_set:
                continue
            visited_set.add(neighbor_key)
            new_bindings.append(BindingEntry(
                role_key=(_MEMORY, turn_seq),
                filler_key=(_MEMORY, neighbor_seq, speaker),
                space=_MEMORY,
            ))
            hops.append(QueryHop(
                _MEMORY, (_MEMORY, turn_seq), (_MEMORY, 1),
                origin, neighbor_key, surface, depth,
                source_hash=1,
            ))
        self.hops.extend(hops)
        visited = tuple(sorted(
            (VisitedKey(_MEMORY, key, direction=0) for key in visited_set),
            key=lambda item: item.stable_key()))
        return ExpansionOutcome(
            state.with_(
                depth=max(state.depth, depth),
                frontier=tuple(sorted(
                    (*remaining, *next_edges),
                    key=lambda entry: entry.priority_tuple(), reverse=True)),
                visited=visited,
                bindings=tuple(new_bindings),
                node_count=state.node_count + len(hops),
                edge_count=state.edge_count + len(hops),
                read_count=state.read_count + len(hops),
                score=state.score + len(hops),
            ),
            edge,
            len(hops),
        )


__all__ = [
    "ConceptAnchor",
    "QueryHop",
    "TrainedGraphQueryBridge",
]
