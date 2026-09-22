"""统一查询状态协议：QueryState/frontier/binding/evidence/visited/termination。

本模块只承载跨 Core、Memory、Dialogue/Companion 查询的纯整数状态与稳定
序列化；不绑定任何存储后端、宿主对象或语言表。所有可影响结果的集合都
在序列化前排序，frontier 优先级按文档 29.3 整数元组降序比较，禁止对象 id、
字典插入序、墙钟或随机数参与决策。
"""
from __future__ import annotations

from dataclasses import dataclass, replace


SPACE_CORE = 1
SPACE_MEMORY = 2
SPACE_DIALOGUE = 3
_VALID_SPACES = frozenset({SPACE_CORE, SPACE_MEMORY, SPACE_DIALOGUE})

ROOT_PENDING = 1
ROOT_EXPANDED = 2
ROOT_EMPTY = 3
_VALID_ROOT_STATUSES = frozenset({ROOT_PENDING, ROOT_EXPANDED, ROOT_EMPTY})

TERMINATION_OPEN = 0
TERMINATION_ANSWER_CLOSED = 1
TERMINATION_CLARIFY_CONFLICT = 2
TERMINATION_CLARIFY_MISSING_BINDING = 3
TERMINATION_NO_FRONTIER = 4
TERMINATION_BUDGET_EXHAUSTED = 5
TERMINATION_CYCLE_GUARD = 6
TERMINATION_MARGINAL_CONVERGED = 7
# A non-factual dialogue act can be fully generated from the shared graph
# state while the requested proposition remains unresolved.  Keep this
# terminal distinct from both factual ANSWER_CLOSED and missing-binding
# clarification so the trace describes the actual delivered act.
TERMINATION_DIALOGUE_UNKNOWN = 8
_VALID_TERMINATIONS = frozenset({
    TERMINATION_OPEN,
    TERMINATION_ANSWER_CLOSED,
    TERMINATION_CLARIFY_CONFLICT,
    TERMINATION_CLARIFY_MISSING_BINDING,
    TERMINATION_NO_FRONTIER,
    TERMINATION_BUDGET_EXHAUSTED,
    TERMINATION_CYCLE_GUARD,
    TERMINATION_MARGINAL_CONVERGED,
    TERMINATION_DIALOGUE_UNKNOWN,
})


def _strict_tuple(value, *, label):
    """校验非负整数 tuple；空 tuple 仅在显式允许时接受。"""
    if type(value) is not tuple:
        raise TypeError(f"{label} 必须是 tuple")
    if any(type(item) is not int or item < 0 for item in value):
        raise ValueError(f"{label} 只能包含非负整数")
    return value


@dataclass(frozen=True, slots=True)
class QueryBudget:
    """请求级确定性预算；超限只能澄清或失败，不参与语义。"""

    max_nodes: int = 4096
    max_edges: int = 8192
    max_reads: int = 16384
    max_depth: int = 12

    def __post_init__(self) -> None:
        for label in ("max_nodes", "max_edges", "max_reads", "max_depth"):
            value = getattr(self, label)
            if type(value) is not int or value <= 0:
                raise ValueError(f"QueryBudget.{label} 必须是正整数")

    def stable_key(self) -> tuple[int, ...]:
        """返回预算的稳定整数键。"""
        return (
            self.max_nodes, self.max_edges, self.max_reads, self.max_depth)


@dataclass(frozen=True, slots=True)
class QueryAnchor:
    """输入激活的一个节点/边/结构候选。"""

    space: int
    ref_key: tuple[int, ...]
    kind: int = 1  # 1=node 2=edge 3=structure
    support: int = 1
    scope_key: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.space not in _VALID_SPACES:
            raise ValueError("QueryAnchor.space 未注册")
        _strict_tuple(self.ref_key, label="QueryAnchor.ref_key")
        if not self.ref_key:
            raise ValueError("QueryAnchor.ref_key 不能为空")
        if type(self.kind) is not int or self.kind <= 0:
            raise ValueError("QueryAnchor.kind 必须是正整数")
        if type(self.support) is not int or self.support <= 0:
            raise ValueError("QueryAnchor.support 必须是正整数")
        _strict_tuple(self.scope_key, label="QueryAnchor.scope_key")

    def stable_key(self) -> tuple[int, ...]:
        """返回 anchor 的稳定整数键（空间+引用+种类）。"""
        return (self.space, self.kind, len(self.ref_key), *self.ref_key,
                len(self.scope_key), *self.scope_key, self.support)


@dataclass(frozen=True, slots=True)
class QueryRoot:
    """一个图库在当前请求中的根参与记录。

    ``ROOT_EMPTY`` 是可审计的未命中，不是禁用该图库；非空根在综合终止
    前必须由同一 frontier 实际扩展为 ``ROOT_EXPANDED``。priority_weight
    是请求级定点整数策略，只影响 frontier 排序，不改变回答资格。
    """

    owner_space: int
    root_key: tuple[int, ...]
    status: int
    priority_weight: int
    source_ref: tuple[int, ...] = ()
    scope_key: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.owner_space not in _VALID_SPACES:
            raise ValueError("QueryRoot.owner_space 未注册")
        _strict_tuple(self.root_key, label="QueryRoot.root_key")
        if not self.root_key:
            raise ValueError("QueryRoot.root_key 不能为空")
        if self.status not in _VALID_ROOT_STATUSES:
            raise ValueError("QueryRoot.status 未注册")
        if type(self.priority_weight) is not int or self.priority_weight <= 0:
            raise ValueError("QueryRoot.priority_weight 必须是正整数")
        _strict_tuple(self.source_ref, label="QueryRoot.source_ref")
        _strict_tuple(self.scope_key, label="QueryRoot.scope_key")

    def expanded(self) -> "QueryRoot":
        """返回已扩展形态；空根保持空状态。"""
        if self.status == ROOT_EMPTY:
            return self
        return replace(self, status=ROOT_EXPANDED)

    def stable_key(self) -> tuple[int, ...]:
        return (
            self.owner_space,
            self.status,
            self.priority_weight,
            len(self.root_key), *self.root_key,
            len(self.source_ref), *self.source_ref,
            len(self.scope_key), *self.scope_key,
        )


@dataclass(frozen=True, slots=True)
class FrontierEntry:
    """一条待扩展边：优先级元组 + 端点 + 稳定决胜 key。

    优先级字段（降序比较）：required_slot_gain、evidence_support、
    relation_fit、discourse_fit、source_trust、recency_weight、-depth、
    stable_edge_key。负号仅用于排序编码，实际存储仍为非负整数域。
    """

    edge_key: tuple[int, ...]
    owner_space: int
    target_key: tuple[int, ...] = ()
    root_key: tuple[int, ...] = ()
    scope_key: tuple[int, ...] = ()
    source_ref: tuple[int, ...] = ()
    direction: int = 0
    owner_weight: int = 1
    depth: int = 0
    required_slot_gain: int = 0
    evidence_support: int = 0
    relation_fit: int = 0
    discourse_fit: int = 0
    source_trust: int = 0
    recency_weight: int = 0

    def __post_init__(self) -> None:
        _strict_tuple(self.edge_key, label="FrontierEntry.edge_key")
        if not self.edge_key:
            raise ValueError("FrontierEntry.edge_key 不能为空")
        if self.owner_space not in _VALID_SPACES:
            raise ValueError("FrontierEntry.owner_space 未注册")
        for label in ("target_key", "root_key", "scope_key", "source_ref"):
            _strict_tuple(
                getattr(self, label), label=f"FrontierEntry.{label}")
        if self.direction not in {0, 1, 2}:
            raise ValueError("FrontierEntry.direction 未注册")
        if type(self.owner_weight) is not int or self.owner_weight <= 0:
            raise ValueError("FrontierEntry.owner_weight 必须是正整数")
        for label in ("depth", "required_slot_gain", "evidence_support",
                      "relation_fit", "discourse_fit", "source_trust",
                      "recency_weight"):
            value = getattr(self, label)
            if type(value) is not int or value < 0:
                raise ValueError(f"FrontierEntry.{label} 必须是非负整数")

    def priority_tuple(self) -> tuple:
        """返回按降序比较的完整优先级元组，不扁平复制长整数键。

        每个变长字段仍先比较长度再比较内容，与 ``stable_key()`` 的
        扁平长度前缀顺序完全一致；嵌套 tuple 只复用原键对象。
        """
        return (
            self.required_slot_gain,
            self.owner_weight,
            self.evidence_support,
            self.relation_fit,
            self.discourse_fit,
            self.source_trust,
            self.recency_weight,
            -self.depth,
            self.owner_space,
            self.direction,
            self.owner_weight,
            self.depth,
            self.required_slot_gain,
            self.evidence_support,
            self.relation_fit,
            self.discourse_fit,
            self.source_trust,
            self.recency_weight,
            len(self.edge_key), self.edge_key,
            len(self.target_key), self.target_key,
            len(self.root_key), self.root_key,
            len(self.scope_key), self.scope_key,
            len(self.source_ref), self.source_ref,
        )

    def stable_key(self) -> tuple[int, ...]:
        """返回含 owner/scope/source/direction/root 的完整稳定记录。"""
        return (
            self.owner_space,
            self.direction,
            self.owner_weight,
            self.depth,
            self.required_slot_gain,
            self.evidence_support,
            self.relation_fit,
            self.discourse_fit,
            self.source_trust,
            self.recency_weight,
            len(self.edge_key), *self.edge_key,
            len(self.target_key), *self.target_key,
            len(self.root_key), *self.root_key,
            len(self.scope_key), *self.scope_key,
            len(self.source_ref), *self.source_ref,
        )


def sort_frontier(frontier: tuple[FrontierEntry, ...]) -> tuple[FrontierEntry, ...]:
    """按优先级元组降序稳定排序；同优时按 edge_key 决胜。"""
    return tuple(sorted(
        frontier, key=lambda entry: entry.priority_tuple(), reverse=True))


@dataclass(frozen=True, slots=True)
class BindingEntry:
    """一个 role/slot 绑定候选；冲突显式保留，不覆盖。"""

    role_key: tuple[int, ...]
    filler_key: tuple[int, ...]
    space: int = SPACE_CORE
    scope_key: tuple[int, ...] = ()
    conflict_kept: int = 0  # 1=该绑定与另一候选冲突且双方证据均保留

    def __post_init__(self) -> None:
        for label in ("role_key", "filler_key", "scope_key"):
            _strict_tuple(getattr(self, label), label=f"BindingEntry.{label}")
        if not self.role_key or not self.filler_key:
            raise ValueError("BindingEntry role/filler 不能为空")
        if self.space not in _VALID_SPACES:
            raise ValueError("BindingEntry.space 未注册")
        if self.conflict_kept not in {0, 1}:
            raise ValueError("BindingEntry.conflict_kept 必须是 0 或 1")

    def stable_key(self) -> tuple[int, ...]:
        return (self.space, len(self.role_key), *self.role_key,
                len(self.filler_key), *self.filler_key,
                len(self.scope_key), *self.scope_key, self.conflict_kept)


@dataclass(frozen=True, slots=True)
class EvidenceEntry:
    """一条可回读、可校验的证据（支持/反对/未知）。"""

    source_ref: tuple[int, ...]
    hypothesis_key: tuple[int, ...]
    space: int = SPACE_CORE
    polarity: int = 1  # 1=support 2=refute 3=unknown
    trust: int = 1
    evidence_key: tuple[int, ...] = ()
    scope_key: tuple[int, ...] = ()
    payload_key: tuple[int, ...] = ()
    supersedes_key: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        _strict_tuple(self.source_ref, label="EvidenceEntry.source_ref")
        _strict_tuple(self.hypothesis_key, label="EvidenceEntry.hypothesis_key")
        if not self.source_ref or not self.hypothesis_key:
            raise ValueError("EvidenceEntry 引用不能为空")
        if self.space not in _VALID_SPACES:
            raise ValueError("EvidenceEntry.space 未注册")
        if self.polarity not in {1, 2, 3}:
            raise ValueError("EvidenceEntry.polarity 未注册")
        if type(self.trust) is not int or self.trust <= 0:
            raise ValueError("EvidenceEntry.trust 必须是正整数")
        for label in ("evidence_key", "scope_key", "payload_key", "supersedes_key"):
            _strict_tuple(getattr(self, label), label=f"EvidenceEntry.{label}")

    def stable_key(self) -> tuple[int, ...]:
        return (self.space, self.polarity, self.trust,
                len(self.source_ref), *self.source_ref,
                len(self.hypothesis_key), *self.hypothesis_key,
                len(self.evidence_key), *self.evidence_key,
                len(self.scope_key), *self.scope_key,
                len(self.payload_key), *self.payload_key,
                len(self.supersedes_key), *self.supersedes_key)


@dataclass(frozen=True, slots=True)
class VisitedKey:
    """visited 状态摘要；等价定义跨语言一致，禁止对象地址。"""

    space: int
    node_key: tuple[int, ...]
    direction: int = 0  # 0=无向 1=正向 2=反向
    binding_digest: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.space not in _VALID_SPACES:
            raise ValueError("VisitedKey.space 未注册")
        _strict_tuple(self.node_key, label="VisitedKey.node_key")
        if not self.node_key:
            raise ValueError("VisitedKey.node_key 不能为空")
        _strict_tuple(self.binding_digest, label="VisitedKey.binding_digest")
        if self.direction not in {0, 1, 2}:
            raise ValueError("VisitedKey.direction 未注册")

    def stable_key(self) -> tuple[int, ...]:
        return (self.space, self.direction, len(self.node_key), *self.node_key,
                len(self.binding_digest), *self.binding_digest)


def _packed(value: tuple[int, ...]) -> tuple[int, ...]:
    """为可变长键加长度边界，避免拼接歧义。"""
    return len(value), *value


def _binding_order_key(item: BindingEntry) -> tuple:
    """Sort a binding without flattening its complete stable key."""
    return (item.space, len(item.role_key), item.role_key,
            len(item.filler_key), item.filler_key,
            len(item.scope_key), item.scope_key, item.conflict_kept)


def _evidence_order_key(item: EvidenceEntry) -> tuple:
    """Sort evidence by the stable-key field order using nested tuples.

    Nested immutable tuples compare identically to their flattened length
    prefixed form, while avoiding a fresh large tuple for every candidate on
    every QueryState canonicalization.
    """
    return (item.space, item.polarity, item.trust,
            len(item.source_ref), item.source_ref,
            len(item.hypothesis_key), item.hypothesis_key,
            len(item.evidence_key), item.evidence_key,
            len(item.scope_key), item.scope_key,
            len(item.payload_key), item.payload_key,
            len(item.supersedes_key), item.supersedes_key)


def _visited_order_key(item: VisitedKey) -> tuple:
    """Sort visited entries without materializing stable-key copies."""
    return (item.space, item.direction, len(item.node_key), item.node_key,
            len(item.binding_digest), item.binding_digest)


@dataclass(frozen=True, slots=True)
class QueryState:
    """一次图查询的完整不可变状态。

    字段语义见 docs/自由对话图模型_深度查询与全面训练长任务_20260903.md
    第 11.1/29.1 节；任何 Python 对象身份、字典插入序和墙钟随机性都不得
    作为语义依据。
    """

    query_key: tuple[int, ...]
    roots: tuple[QueryRoot, ...] = ()
    anchors: tuple[QueryAnchor, ...] = ()
    frontier: tuple[FrontierEntry, ...] = ()
    bindings: tuple[BindingEntry, ...] = ()
    evidence: tuple[EvidenceEntry, ...] = ()
    visited: tuple[VisitedKey, ...] = ()
    depth: int = 0
    minimum_depth: int = 1
    node_count: int = 0
    edge_count: int = 0
    read_count: int = 0
    score: int = 0
    required_slots_open: int = 0
    evidence_closed: int = 0
    conflict_open: int = 0
    best_score: int = 0
    second_score: int = 0
    best_candidate_key: tuple[int, ...] = ()
    second_candidate_key: tuple[int, ...] = ()
    marginal_gain: int = 0
    cycle_hit: int = 0
    generation_ready: int = 0
    termination: int = TERMINATION_OPEN
    budget: QueryBudget = QueryBudget()
    active_spaces: tuple[int, ...] = (SPACE_CORE, SPACE_MEMORY, SPACE_DIALOGUE)

    def __post_init__(self) -> None:
        _strict_tuple(self.query_key, label="QueryState.query_key")
        if not self.query_key:
            raise ValueError("QueryState.query_key 不能为空")
        for label, expected in (
                ("roots", QueryRoot),
                ("anchors", QueryAnchor),
                ("frontier", FrontierEntry),
                ("bindings", BindingEntry),
                ("evidence", EvidenceEntry),
                ("visited", VisitedKey)):
            values = getattr(self, label)
            if type(values) is not tuple or any(
                    not isinstance(item, expected) for item in values):
                raise TypeError(f"QueryState.{label} 类型错误")
        for label in (
                "depth", "minimum_depth", "node_count", "edge_count",
                "read_count", "score", "required_slots_open",
                "evidence_closed", "conflict_open", "best_score",
                "second_score", "marginal_gain", "cycle_hit",
                "generation_ready"):
            value = getattr(self, label)
            if type(value) is not int or value < 0:
                raise ValueError(f"QueryState.{label} 必须是非负整数")
        for label in (
                "evidence_closed", "conflict_open", "cycle_hit",
                "generation_ready"):
            if getattr(self, label) not in {0, 1}:
                raise ValueError(f"QueryState.{label} 必须是 0 或 1")
        _strict_tuple(
            self.best_candidate_key, label="QueryState.best_candidate_key")
        _strict_tuple(
            self.second_candidate_key,
            label="QueryState.second_candidate_key")
        if self.termination not in _VALID_TERMINATIONS:
            raise ValueError("QueryState.termination 未注册")
        if not isinstance(self.budget, QueryBudget):
            raise TypeError("QueryState.budget 必须是 QueryBudget")
        if not self.active_spaces or any(
                space not in _VALID_SPACES for space in self.active_spaces):
            raise ValueError("QueryState.active_spaces 未注册")
        if len(set(self.active_spaces)) != len(self.active_spaces):
            raise ValueError("QueryState.active_spaces 不得重复")

    def with_(self, **changes) -> "QueryState":
        """返回替换字段后的新状态；budget 需整体传入。"""
        return replace(self, **changes)

    def canonical(self) -> "QueryState":
        """返回排序后的规范形态：anchors/frontier/bindings/evidence/visited。"""
        return replace(
            self,
            roots=tuple(sorted(
                self.roots, key=lambda item: item.stable_key())),
            anchors=tuple(sorted(
                self.anchors, key=lambda item: item.stable_key())),
            frontier=sort_frontier(self.frontier),
            bindings=tuple(sorted(
                self.bindings, key=_binding_order_key)),
            evidence=tuple(sorted(
                self.evidence, key=_evidence_order_key)),
            visited=tuple(sorted(
                self.visited, key=_visited_order_key)),
            active_spaces=tuple(sorted(self.active_spaces)),
        )

    def stable_key(self) -> tuple[int, ...]:
        """返回完整状态的可重建稳定键；必须先 canonical() 保证排序稳定。"""
        state = self.canonical()
        return (
            len(state.query_key), *state.query_key,
            state.depth, state.minimum_depth,
            state.node_count, state.edge_count, state.read_count, state.score,
            state.required_slots_open,
            state.evidence_closed,
            state.conflict_open,
            state.best_score,
            state.second_score,
            len(state.best_candidate_key), *state.best_candidate_key,
            len(state.second_candidate_key), *state.second_candidate_key,
            state.marginal_gain,
            state.cycle_hit,
            state.generation_ready,
            state.termination,
            *state.budget.stable_key(),
            len(state.active_spaces), *state.active_spaces,
            len(state.roots),
            *(item for root in state.roots
              for item in _packed(root.stable_key())),
            len(state.anchors),
            *(item for anchor in state.anchors
              for item in _packed(anchor.stable_key())),
            len(state.frontier),
            *(item for entry in state.frontier
              for item in _packed(entry.stable_key())),
            len(state.bindings),
            *(item for binding in state.bindings
              for item in _packed(binding.stable_key())),
            len(state.evidence),
            *(item for evidence in state.evidence
              for item in _packed(evidence.stable_key())),
            len(state.visited),
            *(item for visited in state.visited
              for item in _packed(visited.stable_key())),
        )


def seed_query(
        query_key: tuple[int, ...],
        *,
        roots: tuple[QueryRoot, ...] = (),
        anchors: tuple[QueryAnchor, ...] = (),
        minimum_depth: int = 1,
        budget: QueryBudget = QueryBudget(),
        active_spaces: tuple[int, ...] = (SPACE_CORE, SPACE_MEMORY, SPACE_DIALOGUE),
        ) -> QueryState:
    """建立查询起点；anchors 空时仍保留 key 供 clarify/fail 判定。"""
    return QueryState(
        query_key=query_key,
        roots=roots,
        anchors=anchors,
        minimum_depth=minimum_depth,
        budget=budget,
        active_spaces=active_spaces,
    )


__all__ = [
    "SPACE_CORE", "SPACE_MEMORY", "SPACE_DIALOGUE",
    "ROOT_PENDING", "ROOT_EXPANDED", "ROOT_EMPTY",
    "TERMINATION_OPEN", "TERMINATION_ANSWER_CLOSED",
    "TERMINATION_CLARIFY_CONFLICT", "TERMINATION_CLARIFY_MISSING_BINDING",
    "TERMINATION_NO_FRONTIER", "TERMINATION_BUDGET_EXHAUSTED",
    "TERMINATION_CYCLE_GUARD", "TERMINATION_MARGINAL_CONVERGED",
    "TERMINATION_DIALOGUE_UNKNOWN",
    "BindingEntry", "EvidenceEntry", "FrontierEntry", "QueryAnchor",
    "QueryRoot",
    "QueryBudget", "QueryState", "VisitedKey",
    "seed_query", "sort_frontier",
]
