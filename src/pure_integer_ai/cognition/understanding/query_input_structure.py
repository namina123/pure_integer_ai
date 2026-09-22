"""只读查询入口的纯整数 token/span/structure 候选。

本模块不解析自然语言或载体语法。调用方把训练后图恢复出的 span 投影注入
``TrainedInputStructure``；运行时执行整数序列定位、显式图关系传播、区间顺序/
包含闭合，并保留全部并列语义候选。可迁移表达必须由图关系授权，不能仅因某
来源表层引用相同 filler 就升级为全局别名。候选存在不等于命题为真，后续仍须
由同一 QueryState 的关系与 Evidence 闭合。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.understanding.query_open_roles import (
    OpenRelationCandidate,
    OpenRoleFrame,
    match_open_roles,
)

from pure_integer_ai.cognition.understanding.query_concept_links import (
    InputConceptRoute,
    TrainedConceptLink,
    TrainedConceptLinks,
)
from pure_integer_ai.crosscut.determinism.hasher import Hasher


INPUT_STRUCTURE_VERSION = 1
QUERY_INPUT_TRACE_VERSION = 3
QUERY_GRAPH_INPUT_VERSION = 91580

SPAN_SOURCE = 1
SPAN_TRAINED = 2

PROJECTION_FILLER = 1
PROJECTION_PREDICATE = 2

STRUCTURE_OPEN = 1
STRUCTURE_CLOSED = 2
STRUCTURE_ORDER_CONFLICT = 3

CARRIER_GRAPH_SOURCE = 1
CARRIER_SEMANTIC_GRAPH = 2

_SOURCE_HASHER = Hasher("query_input_structure.source.v1")
_VALID_SPAN_KINDS = frozenset({SPAN_SOURCE, SPAN_TRAINED})
_VALID_PROJECTION_KINDS = frozenset({
    PROJECTION_FILLER,
    PROJECTION_PREDICATE,
})
_VALID_STRUCTURE_STATES = frozenset({
    STRUCTURE_OPEN,
    STRUCTURE_CLOSED,
    STRUCTURE_ORDER_CONFLICT,
})


def _strict_key(value, *, where: str, allow_empty: bool = False) -> tuple[int, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{where} 必须是 tuple")
    if not allow_empty and not value:
        raise ValueError(f"{where} 不能为空")
    if any(type(item) is not int or item < 0 for item in value):
        raise ValueError(f"{where} 只能包含非负严格整数")
    return value


def _pack(value: tuple[int, ...]) -> tuple[int, ...]:
    return len(value), *value


@dataclass(frozen=True, slots=True)
class TrainedInputMember:
    """训练图中一个 endpoint 或 predicate cue 的整数 span 投影。"""

    values: tuple[int, ...]
    candidate_key: tuple[int, ...]
    object_kind: int
    projection_kind: int
    proposition_key: tuple[int, ...]
    predicate_key: tuple[int, ...]
    source_ref: tuple[int, ...]
    trained_start: int
    trained_end: int
    member_ordinal: int
    role_key: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        for name in (
                "values", "candidate_key", "proposition_key",
                "predicate_key", "source_ref"):
            _strict_key(getattr(self, name), where=f"TrainedInputMember.{name}")
        _strict_key(
            self.role_key,
            where="TrainedInputMember.role_key",
            allow_empty=True,
        )
        for name in (
                "object_kind", "trained_start", "trained_end",
                "member_ordinal"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(
                    f"TrainedInputMember.{name} 必须是非负严格整数")
        if self.object_kind <= 0:
            raise ValueError("TrainedInputMember.object_kind 必须为正整数")
        if self.projection_kind not in _VALID_PROJECTION_KINDS:
            raise ValueError("TrainedInputMember.projection_kind 未注册")
        if self.trained_end <= self.trained_start:
            raise ValueError("训练 span 必须为非空正向区间")
        if self.trained_end - self.trained_start != len(self.values):
            raise ValueError("训练 span 区间与 values 长度不一致")
        if self.candidate_key[0] != self.object_kind:
            raise ValueError("candidate_key 与 object_kind 不一致")
        if ((self.projection_kind == PROJECTION_FILLER) != bool(self.role_key)):
            raise ValueError("只有 filler 投影必须携带 role_key")

    def stable_key(self) -> tuple[int, ...]:
        return (
            INPUT_STRUCTURE_VERSION,
            self.projection_kind,
            self.object_kind,
            self.trained_start,
            self.trained_end,
            self.member_ordinal,
            *_pack(self.values),
            *_pack(self.candidate_key),
            *_pack(self.proposition_key),
            *_pack(self.predicate_key),
            *_pack(self.role_key),
            *_pack(self.source_ref),
        )


@dataclass(frozen=True, slots=True)
class TrainedInputStructure:
    """一个命题在训练图中的有序成员结构，不携带完整来源正文。"""

    proposition_key: tuple[int, ...]
    predicate_key: tuple[int, ...]
    source_ref: tuple[int, ...]
    carrier_key: tuple[int, ...]
    members: tuple[TrainedInputMember, ...]
    semantic_structure_key: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        for name in (
                "proposition_key", "predicate_key", "source_ref",
                "carrier_key"):
            _strict_key(
                getattr(self, name), where=f"TrainedInputStructure.{name}")
        _strict_key(
            self.semantic_structure_key,
            where="TrainedInputStructure.semantic_structure_key",
            allow_empty=True,
        )
        if (type(self.members) is not tuple or not self.members
                or any(not isinstance(item, TrainedInputMember)
                       for item in self.members)):
            raise TypeError("TrainedInputStructure.members 类型非法")
        expected = tuple(sorted(
            self.members,
            key=lambda item: (
                item.trained_start,
                item.trained_end,
                item.projection_kind,
                item.member_ordinal,
                item.stable_key(),
            ),
        ))
        if self.members != expected or len(set(self.members)) != len(self.members):
            raise ValueError("训练结构成员必须按来源位置排序去重")
        if any(
                item.proposition_key != self.proposition_key
                or item.predicate_key != self.predicate_key
                or item.source_ref != self.source_ref
                for item in self.members):
            raise ValueError("训练结构成员跨 proposition/predicate/source")
        if sum(
                item.projection_kind == PROJECTION_PREDICATE
                for item in self.members) != 1:
            raise ValueError("训练结构必须有且只有一个 predicate cue")
        if not any(
                item.projection_kind == PROJECTION_FILLER
                for item in self.members):
            raise ValueError("训练结构至少需要一个 filler")
        if tuple(item.member_ordinal for item in self.members) != tuple(
                range(len(self.members))):
            raise ValueError("训练结构 member_ordinal 必须连续")

    def topology_key(self) -> tuple[int, ...]:
        """返回不含表层 token 的已训练角色拓扑。"""
        result = [INPUT_STRUCTURE_VERSION, len(self.members)]
        for item in self.members:
            result.extend((
                item.member_ordinal,
                item.projection_kind,
                item.object_kind,
                *_pack(item.role_key),
            ))
        return tuple(result)

    def stable_key(self) -> tuple[int, ...]:
        result = [
            INPUT_STRUCTURE_VERSION,
            *_pack(self.proposition_key),
            *_pack(self.predicate_key),
            *_pack(self.source_ref),
            *_pack(self.carrier_key),
            *_pack(self.semantic_structure_key),
            len(self.members),
        ]
        for item in self.members:
            result.extend(_pack(item.stable_key()))
        return tuple(result)

    def relation_structure_key(self) -> tuple[int, ...]:
        """返回图内 StructureConcept；旧结构退回其整数拓扑摘要。"""
        return (
            self.semantic_structure_key
            if self.semantic_structure_key else self.topology_key())


@dataclass(frozen=True, slots=True)
class InputToken:
    """来源中的一个原始整数 token。"""

    ordinal: int
    value: int

    def __post_init__(self) -> None:
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ValueError("InputToken.ordinal 必须是非负严格整数")
        if type(self.value) is not int or self.value < 0:
            raise ValueError("InputToken.value 必须是非负严格整数")

    def stable_key(self) -> tuple[int, ...]:
        return self.ordinal, self.value


@dataclass(frozen=True, slots=True)
class InputSpan:
    """当前输入来源中的一个精确整数 token 区间。"""

    source_ref: tuple[int, ...]
    start: int
    end: int
    span_kind: int
    values: tuple[int, ...]

    def __post_init__(self) -> None:
        _strict_key(self.source_ref, where="InputSpan.source_ref")
        _strict_key(self.values, where="InputSpan.values")
        if (type(self.start) is not int or type(self.end) is not int
                or self.start < 0 or self.end <= self.start):
            raise ValueError("InputSpan 必须是非空正向严格整数区间")
        if self.span_kind not in _VALID_SPAN_KINDS:
            raise ValueError("InputSpan.span_kind 未注册")
        if self.end - self.start != len(self.values):
            raise ValueError("InputSpan 区间与 values 长度不一致")

    @property
    def ref_key(self) -> tuple[int, ...]:
        return (
            INPUT_STRUCTURE_VERSION,
            self.source_ref[1],
            self.start,
            self.end,
            self.span_kind,
        )

    def stable_key(self) -> tuple[int, ...]:
        return (
            *_pack(self.ref_key),
            *_pack(self.source_ref),
            *_pack(self.values),
        )


@dataclass(frozen=True, slots=True)
class InputSpanRelation:
    """两个输入 span 间的顺序或包含关系。"""

    left_ref: tuple[int, ...]
    right_ref: tuple[int, ...]

    def __post_init__(self) -> None:
        _strict_key(self.left_ref, where="InputSpanRelation.left_ref")
        _strict_key(self.right_ref, where="InputSpanRelation.right_ref")
        if self.left_ref == self.right_ref:
            raise ValueError("InputSpanRelation 不允许自环")

    def stable_key(self) -> tuple[int, ...]:
        return *_pack(self.left_ref), *_pack(self.right_ref)


@dataclass(frozen=True, slots=True)
class InputSemanticCandidate:
    """训练 span 投影产生的开放语义候选；不携带真值。"""

    candidate_key: tuple[int, ...]
    object_kind: int
    projection_kind: int
    span_ref: tuple[int, ...]
    proposition_key: tuple[int, ...]
    predicate_key: tuple[int, ...]
    role_key: tuple[int, ...]
    source_ref: tuple[int, ...]
    member_ordinal: int
    concept_route_key: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        for name in (
                "candidate_key", "span_ref", "proposition_key",
                "predicate_key", "source_ref"):
            _strict_key(
                getattr(self, name), where=f"InputSemanticCandidate.{name}")
        _strict_key(
            self.role_key,
            where="InputSemanticCandidate.role_key",
            allow_empty=True,
        )
        if type(self.object_kind) is not int or self.object_kind <= 0:
            raise ValueError("semantic candidate object_kind 非法")
        if self.candidate_key[0] != self.object_kind:
            raise ValueError("semantic candidate key/kind 不一致")
        if self.projection_kind not in _VALID_PROJECTION_KINDS:
            raise ValueError("semantic candidate projection_kind 未注册")
        if type(self.member_ordinal) is not int or self.member_ordinal < 0:
            raise ValueError("semantic candidate member_ordinal 非法")
        _strict_key(self.concept_route_key, where="semantic candidate.concept_route_key",
                    allow_empty=True)

    def stable_key(self) -> tuple[int, ...]:
        return (
            self.projection_kind,
            self.object_kind,
            self.member_ordinal,
            *_pack(self.candidate_key),
            *_pack(self.span_ref),
            *_pack(self.proposition_key),
            *_pack(self.predicate_key),
            *_pack(self.role_key),
            *_pack(self.source_ref),
            *_pack(self.concept_route_key),
        )


@dataclass(frozen=True, slots=True)
class InputRelationCandidate:
    """一个训练后命题拓扑在当前输入中的开放或闭合投影。"""

    proposition_key: tuple[int, ...]
    predicate_key: tuple[int, ...]
    structure_key: tuple[int, ...]
    carrier_key: tuple[int, ...]
    source_ref: tuple[int, ...]
    span_refs: tuple[tuple[int, ...], ...]
    required_count: int
    support_count: int
    role_coverage: int
    predicate_coverage: int
    required_open: int
    state: int
    conflict_count: int

    def __post_init__(self) -> None:
        for name in (
                "proposition_key", "predicate_key", "structure_key",
                "carrier_key", "source_ref"):
            _strict_key(
                getattr(self, name), where=f"InputRelationCandidate.{name}")
        if (type(self.span_refs) is not tuple or not self.span_refs
                or any(type(item) is not tuple for item in self.span_refs)):
            raise TypeError("relation candidate span_refs 类型非法")
        for item in self.span_refs:
            _strict_key(item, where="InputRelationCandidate.span_ref")
        if self.span_refs != tuple(sorted(set(self.span_refs))):
            raise ValueError("relation candidate span_refs 必须排序去重")
        for name in (
                "required_count", "support_count", "role_coverage",
                "predicate_coverage", "required_open", "conflict_count"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(
                    f"InputRelationCandidate.{name} 必须是非负严格整数")
        if not 0 < self.support_count <= self.required_count:
            raise ValueError("relation candidate support_count 非法")
        if self.required_open != self.required_count - self.support_count:
            raise ValueError("relation candidate required_open 不守恒")
        if self.predicate_coverage not in {0, 1}:
            raise ValueError("predicate_coverage 必须是 0 或 1")
        if self.state not in _VALID_STRUCTURE_STATES:
            raise ValueError("relation candidate state 未注册")
        if ((self.state == STRUCTURE_ORDER_CONFLICT)
                != (self.conflict_count > 0)):
            raise ValueError("relation candidate conflict/state 不闭合")
        if self.state == STRUCTURE_CLOSED and self.required_open:
            raise ValueError("闭合 relation candidate 不得缺成员")

    @property
    def score(self) -> int:
        return (
            self.role_coverage * 1000
            + self.predicate_coverage * 1000
            + int(self.state == STRUCTURE_CLOSED) * 1000
        )

    def stable_key(self) -> tuple[int, ...]:
        result = [
            self.state,
            self.required_count,
            self.support_count,
            self.role_coverage,
            self.predicate_coverage,
            self.required_open,
            self.conflict_count,
            self.score,
            *_pack(self.proposition_key),
            *_pack(self.predicate_key),
            *_pack(self.structure_key),
            *_pack(self.carrier_key),
            *_pack(self.source_ref),
            len(self.span_refs),
        ]
        for item in self.span_refs:
            result.extend(_pack(item))
        return tuple(result)


@dataclass(frozen=True, slots=True)
class InputCarrierCandidate:
    """由命中的训练结构提供的 carrier 候选，而非宿主格式猜测。"""

    carrier_key: tuple[int, ...]
    structure_key: tuple[int, ...]
    source_ref: tuple[int, ...]
    input_span_ref: tuple[int, ...]
    support_count: int

    def __post_init__(self) -> None:
        for name in (
                "carrier_key", "structure_key", "source_ref",
                "input_span_ref"):
            _strict_key(
                getattr(self, name), where=f"InputCarrierCandidate.{name}")
        if type(self.support_count) is not int or self.support_count <= 0:
            raise ValueError("carrier candidate support_count 必须为正严格整数")

    def stable_key(self) -> tuple[int, ...]:
        return (
            self.support_count,
            *_pack(self.carrier_key),
            *_pack(self.structure_key),
            *_pack(self.source_ref),
            *_pack(self.input_span_ref),
        )


@dataclass(frozen=True, slots=True)
class QueryInputStructure:
    """一次查询的纯整数来源、token、span、carrier 与开放语义候选。"""

    source_ref: tuple[int, ...]
    tokens: tuple[InputToken, ...]
    source_span_ref: tuple[int, ...]
    spans: tuple[InputSpan, ...]
    precedes: tuple[InputSpanRelation, ...]
    contains: tuple[InputSpanRelation, ...]
    carrier_candidates: tuple[InputCarrierCandidate, ...]
    semantic_candidates: tuple[InputSemanticCandidate, ...]
    relation_candidates: tuple[InputRelationCandidate, ...]
    concept_links: tuple[TrainedConceptLink, ...] = ()
    concept_routes: tuple[InputConceptRoute, ...] = ()
    open_relations: tuple[OpenRelationCandidate, ...] = ()
    graph_object_keys: tuple[tuple[int, ...], ...] = ()

    def __post_init__(self) -> None:
        _strict_key(self.source_ref, where="QueryInputStructure.source_ref")
        _strict_key(
            self.source_span_ref, where="QueryInputStructure.source_span_ref")
        specifications = (
            ("tokens", InputToken),
            ("spans", InputSpan),
            ("precedes", InputSpanRelation),
            ("contains", InputSpanRelation),
            ("carrier_candidates", InputCarrierCandidate),
            ("semantic_candidates", InputSemanticCandidate),
            ("relation_candidates", InputRelationCandidate),
            ("concept_links", TrainedConceptLink),
            ("concept_routes", InputConceptRoute),
            ("open_relations", OpenRelationCandidate),
        )
        for name, expected in specifications:
            values = getattr(self, name)
            if type(values) is not tuple or any(
                    not isinstance(item, expected) for item in values):
                raise TypeError(f"QueryInputStructure.{name} 类型非法")
            if values != tuple(sorted(
                    set(values), key=lambda item: item.stable_key())):
                raise ValueError(f"QueryInputStructure.{name} 必须规范排序去重")
        if (type(self.graph_object_keys) is not tuple
                or any(type(key) is not tuple or not key
                       or any(type(item) is not int or item < 0 for item in key)
                       for key in self.graph_object_keys)
                or self.graph_object_keys != tuple(sorted(set(self.graph_object_keys)))):
            raise ValueError("QueryInputStructure graph_object_keys 必须是规范纯整数键")
        graph_only = not self.tokens and not self.spans
        if graph_only:
            if (not self.graph_object_keys
                    or any((self.precedes, self.contains, self.carrier_candidates,
                            self.semantic_candidates, self.relation_candidates,
                            self.concept_links, self.concept_routes,
                            self.open_relations))
                    or self.source_span_ref != self.source_ref):
                raise ValueError("无 token/span 输入必须是完整 graph-only 结构")
        elif not self.tokens or not self.spans:
            raise ValueError("文字 QueryInputStructure token/span 必须同时存在")
        if tuple(item.ordinal for item in self.tokens) != tuple(
                range(len(self.tokens))):
            raise ValueError("QueryInputStructure token ordinal 不连续")
        if any(item.source_ref != self.source_ref or item.values != self.token_values
               for item in self.open_relations):
            raise ValueError("开放角色假设不得跨来源或丢弃输入")
        span_refs = {item.ref_key for item in self.spans}
        if not graph_only and self.source_span_ref not in span_refs:
            raise ValueError("QueryInputStructure 缺少 source span")
        if any(
                item.source_ref != self.source_ref
                for item in self.spans):
            raise ValueError("QueryInputStructure span 跨输入来源")
        if any(
                item.span_ref not in span_refs
                for item in self.semantic_candidates):
            raise ValueError("semantic candidate 引用了未知输入 span")
        routes = {item.stable_key(): item for item in self.concept_routes}
        if any(item.span_ref not in span_refs for item in self.concept_routes):
            raise ValueError("concept route 引用了未知输入 span")
        for item in self.semantic_candidates:
            if item.concept_route_key:
                route = routes.get(item.concept_route_key)
                if route is None or (route.span_ref, route.target_key) != (
                        item.span_ref, item.candidate_key):
                    raise ValueError("semantic candidate 与显式概念路径不一致")

    @property
    def token_values(self) -> tuple[int, ...]:
        return tuple(item.value for item in self.tokens)

    def stable_key(self) -> tuple[int, ...]:
        result = [
            QUERY_INPUT_TRACE_VERSION,
            *_pack(self.source_ref),
            *_pack(self.source_span_ref),
        ]
        for collection in (
                self.tokens, self.spans, self.precedes, self.contains,
                self.carrier_candidates, self.semantic_candidates,
                self.relation_candidates, self.concept_links, self.concept_routes,
                self.open_relations):
            result.append(len(collection))
            for item in collection:
                result.extend(_pack(item.stable_key()))
        if self.graph_object_keys:
            result.append(len(self.graph_object_keys))
            for key in self.graph_object_keys:
                result.extend(_pack(key))
        return tuple(result)

    def integer_trace(self) -> dict[str, object]:
        """投影为仅含整数值的 JSON 兼容 trace。"""
        return {
            "schema_version": QUERY_INPUT_TRACE_VERSION,
            "source_ref": list(self.source_ref),
            "token_values": list(self.token_values),
            "token_order": [item.ordinal for item in self.tokens],
            "source_span_ref": list(self.source_span_ref),
            "spans": [
                {
                    "ref": list(item.ref_key),
                    "start": item.start,
                    "end": item.end,
                    "kind": item.span_kind,
                    "values": list(item.values),
                }
                for item in self.spans
            ],
            "precedes": [list(item.stable_key()) for item in self.precedes],
            "contains": [list(item.stable_key()) for item in self.contains],
            "carrier_candidates": [
                {
                    "carrier": list(item.carrier_key),
                    "structure": list(item.structure_key),
                    "source_ref": list(item.source_ref),
                    "input_span": list(item.input_span_ref),
                    "support": item.support_count,
                }
                for item in self.carrier_candidates
            ],
            "semantic_candidates": [
                {
                    "candidate": list(item.candidate_key),
                    "object_kind": item.object_kind,
                    "projection_kind": item.projection_kind,
                    "span": list(item.span_ref),
                    "proposition": list(item.proposition_key),
                    "predicate": list(item.predicate_key),
                    "role": list(item.role_key),
                    "source_ref": list(item.source_ref),
                    "member_ordinal": item.member_ordinal,
                    "concept_route": list(item.concept_route_key),
                }
                for item in self.semantic_candidates
            ],
            "relation_candidates": [
                {
                    "proposition": list(item.proposition_key),
                    "predicate": list(item.predicate_key),
                    "structure": list(item.structure_key),
                    "carrier": list(item.carrier_key),
                    "source_ref": list(item.source_ref),
                    "span_refs": [list(value) for value in item.span_refs],
                    "required_count": item.required_count,
                    "support_count": item.support_count,
                    "role_coverage": item.role_coverage,
                    "predicate_coverage": item.predicate_coverage,
                    "required_open": item.required_open,
                    "state": item.state,
                    "conflict_count": item.conflict_count,
                    "score": item.score,
                }
                for item in self.relation_candidates
            ],
            "stable_key": list(self.stable_key()),
            "concept_links": [list(item.stable_key()) for item in self.concept_links],
            "concept_routes": [list(item.stable_key()) for item in self.concept_routes],
            "open_relations": [list(item.stable_key()) for item in self.open_relations],
            "graph_object_keys": [list(key) for key in self.graph_object_keys],
        }

    def concept_evidence_closed(
            self, proposition: tuple[int, ...],
            evidence_claims: frozenset[tuple[int, ...]],
            ) -> bool:
        """选中命题必须有按训练角色序排列、且全部关系证据已展开的 Span 链。"""
        routes = {item.stable_key(): item for item in self.concept_routes}
        graph = TrainedConceptLinks(self.concept_links)
        reached_by_origin: dict[tuple[int, ...], frozenset[tuple[int, ...]]] = {}
        spans = {item.ref_key: item for item in self.spans}
        members: dict[tuple[int, ...], dict[int, set[InputSpan]]] = {}
        for candidate in self.semantic_candidates:
            if candidate.proposition_key != proposition:
                continue
            choices = members.setdefault(candidate.source_ref, {}).setdefault(
                candidate.member_ordinal, set())
            closed = not candidate.concept_route_key
            if not closed:
                route = routes[candidate.concept_route_key]
                if route.origin_key not in reached_by_origin:
                    reached_by_origin[route.origin_key] = graph.reachable(
                        route.origin_key, evidence_claims=evidence_claims)[0]
                closed = (route.origin_proposition_key in evidence_claims
                          and route.target_key in reached_by_origin[route.origin_key])
            if closed:
                choices.add(spans[candidate.span_ref])
        for relation in self.relation_candidates:
            if relation.proposition_key != proposition or relation.state != STRUCTURE_CLOSED:
                continue
            by_ordinal = members.get(relation.source_ref, {})
            # Pair/inverse candidates intentionally omit the surface
            # predicate member. Their two filler spans are nevertheless a
            # complete graph-backed closure and must not be rejected for the
            # absent lexical cue.
            if relation.required_count == 2 and relation.predicate_coverage == 0:
                if len(by_ordinal) == 2:
                    return True
                continue
            if len(by_ordinal) != relation.required_count:
                continue
            cursor = 0
            for ordinal in sorted(by_ordinal):
                choices = tuple(item for item in by_ordinal[ordinal] if item.start >= cursor)
                if not choices:
                    break
                cursor = min(choices, key=lambda item: (item.end, item.start)).end
            else:
                return True
        return False


def graph_query_input_structure(
        graph_object_keys: tuple[tuple[int, ...], ...],
        ) -> QueryInputStructure:
    """Build a graph-native query input without inventing a text carrier."""
    if (type(graph_object_keys) is not tuple or not graph_object_keys
            or any(type(key) is not tuple or not key
                   or any(type(item) is not int or item < 0 for item in key)
                   for key in graph_object_keys)):
        raise ValueError("graph-only input 必须包含非空非负整数对象键")
    canonical = tuple(sorted(set(graph_object_keys)))
    if canonical != graph_object_keys:
        raise ValueError("graph-only input 对象键必须规范排序去重")
    source = [QUERY_GRAPH_INPUT_VERSION, len(canonical)]
    for key in canonical:
        source.extend(_pack(key))
    source_ref = tuple(source)
    return QueryInputStructure(
        source_ref,
        (),
        source_ref,
        (),
        (),
        (),
        (),
        (),
        (),
        graph_object_keys=canonical,
    )


class TrainedInputStructureProjector:
    """把训练后整数 span 拓扑投影到一次输入，不包含语言或 carrier 分支。"""

    def __init__(self, structures: tuple[TrainedInputStructure, ...], *,
                 concept_links: tuple[TrainedConceptLink, ...] = (),
                 open_role_frames: tuple[OpenRoleFrame, ...] = ()) -> None:
        if (type(structures) is not tuple
                or any(not isinstance(item, TrainedInputStructure)
                       for item in structures)):
            raise TypeError("structures 必须是 TrainedInputStructure tuple")
        self.structures = tuple(sorted(
            set(structures), key=lambda item: item.stable_key()))
        self.concept_graph = TrainedConceptLinks(concept_links)
        if type(open_role_frames) is not tuple or any(
                not isinstance(item, OpenRoleFrame) for item in open_role_frames):
            raise TypeError("open_role_frames 必须是 OpenRoleFrame tuple")
        self.open_role_frames = tuple(sorted(set(open_role_frames),
                                            key=lambda item: item.stable_key()))
        generic_frames = []
        for structure in self.structures:
            fillers = tuple(sorted(
                (item for item in structure.members
                 if item.projection_kind == PROJECTION_FILLER),
                key=lambda item: item.member_ordinal))
            if len(fillers) != 2:
                continue
            roles = tuple(item.role_key for item in fillers)
            if any(not role for role in roles) or len(set(roles)) != 2:
                continue
            generic_frames.append(OpenRoleFrame(
                structure.proposition_key, structure.predicate_key,
                structure.source_ref, roles, ((), (), ()), (0, 1)))
            break
        self._generic_open_role_frames = tuple(sorted(
            set(generic_frames), key=lambda item: item.stable_key()))
        postings: dict[int, list[TrainedInputMember]] = {}
        fillers: dict[tuple[int, ...], list[TrainedInputMember]] = {}
        structures_by_first: dict[int, list[TrainedInputStructure]] = {}
        for structure in self.structures:
            for first in {item.values[0] for item in structure.members}:
                structures_by_first.setdefault(first, []).append(structure)
            for member in structure.members:
                postings.setdefault(member.values[0], []).append(member)
                if member.projection_kind == PROJECTION_FILLER:
                    fillers.setdefault(member.candidate_key, []).append(member)
        self._fillers = {key: tuple(sorted(set(items), key=lambda item: item.stable_key()))
                         for key, items in fillers.items()}
        self._postings = {
            value: tuple(sorted(items, key=lambda item: item.stable_key()))
            for value, items in postings.items()
        }
        self._structures_by_first = {
            value: tuple(sorted(set(items), key=lambda item: item.stable_key()))
            for value, items in structures_by_first.items()
        }

    @staticmethod
    def _source_ref(values: tuple[int, ...]) -> tuple[int, ...]:
        digest = _SOURCE_HASHER.h63_tagged_int_tuple(
            INPUT_STRUCTURE_VERSION, values) + 1
        return INPUT_STRUCTURE_VERSION, digest, len(values)

    @staticmethod
    def _ordered_chain_exists(
            members: tuple[TrainedInputMember, ...],
            matches: dict[tuple[int, ...], tuple[InputSpan, ...]],
            ) -> bool:
        """贪心选择最早结束区间；对全序非重叠成员判定存在性是完备的。"""
        cursor = 0
        for member in members:
            choices = tuple(
                item for item in matches.get(member.stable_key(), ())
                if item.start >= cursor)
            if not choices:
                return False
            selected = min(
                choices, key=lambda item: (item.end, item.start, item.ref_key))
            cursor = selected.end
        return True

    def project(self, values: tuple[int, ...]) -> QueryInputStructure:
        """产生全部整数 span/结构候选；空白与语言类别不参与判断。"""
        _strict_key(values, where="query input values")
        source_ref = self._source_ref(values)
        tokens = tuple(InputToken(index, value) for index, value in enumerate(values))
        source_span = InputSpan(
            source_ref, 0, len(values), SPAN_SOURCE, values)
        span_by_interval: dict[tuple[int, int], InputSpan] = {}
        matches: dict[tuple[int, ...], list[InputSpan]] = {}
        semantic_candidates: list[InputSemanticCandidate] = []
        direct_members: dict[tuple[int, ...], TrainedInputMember] = {}
        for start, value in enumerate(values):
            for member in self._postings.get(value, ()):
                end = start + len(member.values)
                if end > len(values) or values[start:end] != member.values:
                    continue
                interval = (start, end)
                span = span_by_interval.get(interval)
                if span is None:
                    span = InputSpan(
                        source_ref, start, end, SPAN_TRAINED, member.values)
                    span_by_interval[interval] = span
                matches.setdefault(member.stable_key(), []).append(span)
                direct_members[member.stable_key()] = member
                semantic_candidates.append(InputSemanticCandidate(
                    member.candidate_key,
                    member.object_kind,
                    member.projection_kind,
                    span.ref_key,
                    member.proposition_key,
                    member.predicate_key,
                    member.role_key,
                    member.source_ref,
                    member.member_ordinal,
                ))
        direct_matches = {key: tuple(value) for key, value in matches.items()}
        concept_routes: set[InputConceptRoute] = set()
        concept_links: set[TrainedConceptLink] = set()
        closures = {}
        for key, member in sorted(direct_members.items()):
            if (member.projection_kind != PROJECTION_FILLER
                    or (member.proposition_key, member.candidate_key)
                    not in self.concept_graph.expression_origins):
                continue
            if member.candidate_key not in closures:
                closures[member.candidate_key] = self.concept_graph.reachable(member.candidate_key)
            reached, links = closures[member.candidate_key]
            concept_links.update(links)
            for target in sorted(reached - {member.candidate_key}):
                for target_member in self._fillers.get(target, ()):
                    for span in direct_matches[key]:
                        route = InputConceptRoute(
                            span.ref_key, member.candidate_key, target,
                            member.proposition_key, target_member.proposition_key,
                            key, target_member.stable_key())
                        concept_routes.add(route)
                        matches.setdefault(target_member.stable_key(), []).append(span)
                        semantic_candidates.append(InputSemanticCandidate(
                            target, target_member.object_kind, PROJECTION_FILLER,
                            span.ref_key, target_member.proposition_key,
                            target_member.predicate_key, target_member.role_key,
                            target_member.source_ref, target_member.member_ordinal,
                            route.stable_key()))
        normalized_matches = {
            key: tuple(sorted(set(items), key=lambda item: item.stable_key()))
            for key, items in matches.items()
        }
        matched_spans = tuple(sorted(
            span_by_interval.values(), key=lambda item: item.stable_key()))
        spans = tuple(sorted(
            {source_span, *matched_spans}, key=lambda item: item.stable_key()))

        precedes = []
        contains = []
        for left in spans:
            for right in spans:
                if left == right:
                    continue
                if left.end <= right.start:
                    precedes.append(InputSpanRelation(
                        left.ref_key, right.ref_key))
                if (left.start <= right.start and right.end <= left.end
                        and (left.start, left.end) != (right.start, right.end)):
                    contains.append(InputSpanRelation(
                        left.ref_key, right.ref_key))

        relation_candidates = []
        carrier_candidates = []
        candidate_structures = tuple(sorted({
            structure
            for value in set(values)
            for structure in self._structures_by_first.get(value, ())
        }, key=lambda item: item.stable_key()))
        for structure in candidate_structures:
            supported = tuple(
                member for member in structure.members
                if normalized_matches.get(member.stable_key()))
            if not supported:
                continue
            required_count = len(structure.members)
            support_count = len(supported)
            all_members = support_count == required_count
            order_closed = all_members and self._ordered_chain_exists(
                structure.members, normalized_matches)
            # Pair/inverse closure is a graph-backed route for binary
            # relations whose two trained fillers are both present while the
            # surface predicate is expressed differently.  It does not apply
            # to n-ary/property structures or to one-sided matches.
            filler_supported = tuple(
                item for item in supported
                if item.projection_kind == PROJECTION_FILLER)
            pair_closed = (
                len(structure.members) == 3
                and sum(item.projection_kind == PROJECTION_FILLER
                        for item in structure.members) == 2
                and len(filler_supported) == 2
                and not any(item.projection_kind == PROJECTION_PREDICATE
                            for item in supported))
            if pair_closed:
                required_count = 2
                support_count = 2
                order_closed = True
            if order_closed:
                state = STRUCTURE_CLOSED
                conflict_count = 0
            elif all_members:
                state = STRUCTURE_ORDER_CONFLICT
                conflict_count = 1
            else:
                state = STRUCTURE_OPEN
                conflict_count = 0
            span_refs = tuple(sorted({
                span.ref_key
                for member in supported
                for span in normalized_matches[member.stable_key()]
            }))
            role_coverage = sum(
                member.projection_kind == PROJECTION_FILLER
                for member in supported)
            predicate_coverage = int(any(
                member.projection_kind == PROJECTION_PREDICATE
                for member in supported))
            relation = InputRelationCandidate(
                structure.proposition_key,
                structure.predicate_key,
                structure.relation_structure_key(),
                structure.carrier_key,
                structure.source_ref,
                span_refs,
                required_count,
                support_count,
                role_coverage,
                predicate_coverage,
                required_count - support_count,
                state,
                conflict_count,
            )
            relation_candidates.append(relation)
            carrier_candidates.append(InputCarrierCandidate(
                structure.carrier_key,
                structure.relation_structure_key(),
                structure.source_ref,
                source_span.ref_key,
                support_count,
            ))

        open_relations = match_open_roles(self.open_role_frames, values, source_ref)
        if not open_relations:
            # A trained frame may include source-local outer carrier material
            # (for example a terminator) absent from the current input.  When
            # its exact predicate span is present, retain that delimiter and
            # the trained binary role order while relaxing only the outer
            # carrier.  Predicate-free text remains a generic Observation;
            # it does not gain a relation parse from arbitrary token splits.
            adaptive_frames = []
            for structure in candidate_structures:
                fillers = tuple(
                    item for item in structure.members
                    if item.projection_kind == PROJECTION_FILLER)
                predicates = tuple(
                    item for item in structure.members
                    if item.projection_kind == PROJECTION_PREDICATE
                    and normalized_matches.get(item.stable_key()))
                if len(fillers) != 2 or len(predicates) != 1:
                    continue
                ordered_fillers = tuple(sorted(
                    fillers, key=lambda item: item.member_ordinal))
                predicate = predicates[0]
                if predicate.member_ordinal < ordered_fillers[0].member_ordinal:
                    gaps = (predicate.values, (), ())
                elif predicate.member_ordinal > ordered_fillers[1].member_ordinal:
                    gaps = ((), (), predicate.values)
                else:
                    gaps = ((), predicate.values, ())
                adaptive_frames.append(OpenRoleFrame(
                    structure.proposition_key,
                    structure.predicate_key,
                    structure.source_ref,
                    tuple(item.role_key for item in ordered_fillers),
                    gaps,
                    (0, len(values)),
                ))
            open_relations = match_open_roles(
                tuple(adaptive_frames), values, source_ref)
        # Do not manufacture an open relation from arbitrary code-point
        # diversity.  A generic frame has no predicate/span evidence and would
        # turn every unknown sentence into a character-position split, which
        # is both a false graph claim and a hidden character-neighbour route.
        # Uncovered input remains a source-local generic Memory Observation;
        # real open relations still come from trained structure frames above.

        return QueryInputStructure(
            source_ref,
            tuple(sorted(tokens, key=lambda item: item.stable_key())),
            source_span.ref_key,
            spans,
            tuple(sorted(
                set(precedes), key=lambda item: item.stable_key())),
            tuple(sorted(
                set(contains), key=lambda item: item.stable_key())),
            tuple(sorted(
                set(carrier_candidates), key=lambda item: item.stable_key())),
            tuple(sorted(
                set(semantic_candidates), key=lambda item: item.stable_key())),
            tuple(sorted(
                set(relation_candidates), key=lambda item: item.stable_key())),
            tuple(sorted(concept_links, key=lambda item: item.stable_key())),
            tuple(sorted(concept_routes, key=lambda item: item.stable_key())),
            open_relations,
        )


__all__ = [
    "CARRIER_GRAPH_SOURCE",
    "CARRIER_SEMANTIC_GRAPH",
    "INPUT_STRUCTURE_VERSION",
    "PROJECTION_FILLER",
    "PROJECTION_PREDICATE",
    "SPAN_SOURCE",
    "SPAN_TRAINED",
    "STRUCTURE_CLOSED",
    "STRUCTURE_OPEN",
    "STRUCTURE_ORDER_CONFLICT",
    "InputCarrierCandidate",
    "InputRelationCandidate",
    "InputSemanticCandidate",
    "InputSpan",
    "InputSpanRelation",
    "InputToken",
    "QUERY_GRAPH_INPUT_VERSION",
    "QueryInputStructure",
    "TrainedInputMember",
    "TrainedInputStructure",
    "TrainedInputStructureProjector",
    "graph_query_input_structure",
]
