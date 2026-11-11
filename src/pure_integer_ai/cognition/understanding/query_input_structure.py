"""只读查询入口的纯整数 token/span/structure 候选。

本模块不解析自然语言或载体语法。调用方把训练后图恢复出的 span 投影注入
``TrainedInputStructure``；运行时执行整数序列定位、显式图关系传播、区间顺序/
包含闭合，并保留全部并列语义候选。可迁移表达必须由图关系授权，不能仅因某
来源表层引用相同 filler 就升级为全局别名。候选存在不等于命题为真，后续仍须
由同一 QueryState 的关系与 Evidence 闭合。
"""
from __future__ import annotations

from collections.abc import Iterator
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
from pure_integer_ai.cognition.understanding.query_operation_requests import InputOperationRequestBinding
from pure_integer_ai.cognition.understanding.query_assertion_scope import InputAssertionScopeBinding
from pure_integer_ai.crosscut.determinism.hasher import Hasher
from pure_integer_ai.cognition.understanding.query_provenance_identity import (
    semantic_candidate_identity_key as _semantic_candidate_identity_key,
)


INPUT_STRUCTURE_VERSION = 1
QUERY_INPUT_TRACE_VERSION = 4
QUERY_GRAPH_INPUT_VERSION = 91580
INPUT_DISCOURSE_OPERATION_VERSION = 1

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


def _ordered_span_chain_exists(
        spans_by_ordinal: dict[int, set["InputSpan"]],
        *,
        cursor: int = 0,
        ) -> bool:
    """Require one non-overlapping integer span per trained role ordinal."""
    for ordinal in sorted(spans_by_ordinal):
        choices = tuple(sorted(
            (span for span in spans_by_ordinal[ordinal]
             if span.start >= cursor),
            key=lambda item: (item.end, item.start, item.ref_key),
        ))
        if not choices:
            return False
        cursor = choices[0].end
    return True


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


def _relation_candidate_semantic_key(
        semantic_candidates: tuple[InputSemanticCandidate, ...] | list[
            InputSemanticCandidate],
        candidate: InputRelationCandidate,
        *,
        source_provenance_key: tuple[int, ...] | None = None,
        ) -> tuple[object, ...]:
    """返回已观察关系拓扑；缺失成员时保留完整命题和物理来源资格。

    全部成员无冲突闭合后才允许按当前完整投影聚合物理来源。开放投影
    没有证明未观察端点相等，必须留待 Core 的完整槽验证与等价仲裁。
    来源证明只核验来源，不能代替未观察的 Role/filler。
    """
    semantic_members = tuple(sorted({
        (
            item.member_ordinal,
            item.projection_kind,
            item.object_kind,
            # A verified source proof makes the authored document the
            # identity boundary.  Physical duplicate rows may allocate
            # different local declaration tails to the same role/span;
            # retaining those tails would reintroduce the duplicate tie.
            (_semantic_candidate_identity_key(item.candidate_key)
             if source_provenance_key is None else ()),
            item.role_key,
            item.span_ref,
        )
        for item in semantic_candidates
        if item.proposition_key == candidate.proposition_key
        and item.predicate_key == candidate.predicate_key
        and item.source_ref == candidate.source_ref
        and item.span_ref in candidate.span_refs
    }))
    result = (
        candidate.predicate_key,
        candidate.structure_key,
        candidate.carrier_key,
        candidate.span_refs,
        candidate.required_count,
        candidate.support_count,
        candidate.role_coverage,
        candidate.predicate_coverage,
        candidate.required_open,
        candidate.state,
        candidate.conflict_count,
        semantic_members,
        (() if (candidate.state == STRUCTURE_CLOSED
                and candidate.required_open == 0
                and candidate.support_count == candidate.required_count
                and candidate.conflict_count == 0)
         else (candidate.proposition_key, candidate.source_ref)),
    )
    # 不完整候选保留来源范围，开放框架也使用这个键，不能先删除答案尚未
    # 出现的竞争命题，再把幸存者误当唯一答案。原图对象与来源始终不变。
    return result


def relation_candidate_semantic_key(
        structure: "QueryInputStructure",
        candidate: InputRelationCandidate,
        ) -> tuple[object, ...]:
    """Return the input-bound semantic topology of one relation candidate."""
    # A query may carry a verified proof for the physical source row.  The
    # proof is the boundary at which duplicate authored declarations may be
    # collapsed; without it, the lower key keeps the normalized endpoint
    # identity and therefore remains conservative.  Do the lookup here so
    # every consumer (ranking, scope joins, and source aggregation) observes
    # the same proof-aware key.
    source_provenance = dict(structure.source_provenance_keys)
    source_provenance_key = source_provenance.get(candidate.source_ref)
    return _relation_candidate_semantic_key(
        structure.semantic_candidates, candidate,
        source_provenance_key=source_provenance_key,
    )


def relation_candidate_semantic_groups(
        structure: "QueryInputStructure",
        candidates: tuple[InputRelationCandidate, ...] | None = None,
        ) -> tuple[tuple[tuple[object, ...], tuple[InputRelationCandidate, ...]], ...]:
    """Group relation candidates by exact input-bound semantic topology.

    The returned members retain their original proposition/source identities;
    the grouping is only a consumer-side provenance aggregation boundary.
    """
    values = (structure.relation_candidates
              if candidates is None else tuple(candidates))
    groups: dict[tuple[object, ...], list[InputRelationCandidate]] = {}
    for candidate in values:
        key = relation_candidate_semantic_key(structure, candidate)
        groups.setdefault(key, []).append(candidate)
    return tuple(sorted(
        ((key, tuple(sorted(set(items), key=lambda item: item.stable_key())))
         for key, items in groups.items()),
        key=lambda item: item[0],
    ))


@dataclass(frozen=True, slots=True)
class TrainedInputDiscourseOperator:
    """An integer surface relation for a trained discourse operation."""

    operation_key: tuple[int, ...]
    surface_values: tuple[int, ...]

    def __post_init__(self) -> None:
        _strict_key(self.operation_key, where="discourse operator key")
        _strict_key(self.surface_values, where="discourse operator surface")

    def stable_key(self) -> tuple[int, ...]:
        return (
            INPUT_DISCOURSE_OPERATION_VERSION,
            *_pack(self.operation_key),
            *_pack(self.surface_values),
        )


@dataclass(frozen=True, slots=True)
class InputDiscourseComposition:
    """A trained operator joining two input scopes and their relation candidates."""

    operation_key: tuple[int, ...]
    operation_span_ref: tuple[int, ...]
    operand_scope_refs: tuple[tuple[int, ...], tuple[int, ...]]
    operand_relation_keys: tuple[
        tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]
    operand_states: tuple[int, int]
    state: int

    def __post_init__(self) -> None:
        _strict_key(self.operation_key, where="composition operation key")
        _strict_key(self.operation_span_ref, where="composition operator span")
        if (type(self.operand_scope_refs) is not tuple
                or len(self.operand_scope_refs) != 2
                or any(type(item) is not tuple for item in self.operand_scope_refs)):
            raise TypeError("composition requires two operand scopes")
        for item in self.operand_scope_refs:
            _strict_key(item, where="composition operand scope", allow_empty=True)
        if (type(self.operand_relation_keys) is not tuple
                or len(self.operand_relation_keys) != 2):
            raise TypeError("composition requires two relation-candidate groups")
        for group in self.operand_relation_keys:
            if (type(group) is not tuple
                    or group != tuple(sorted(set(group)))):
                raise ValueError("composition relation groups must be canonical")
            for item in group:
                _strict_key(item, where="composition relation candidate key")
        if (type(self.operand_states) is not tuple or len(self.operand_states) != 2
                or any(type(item) is not int or item not in {
                    STRUCTURE_OPEN, STRUCTURE_CLOSED, STRUCTURE_ORDER_CONFLICT}
                       for item in self.operand_states)):
            raise ValueError("composition operand state is invalid")
        if self.state not in {
                STRUCTURE_OPEN, STRUCTURE_CLOSED, STRUCTURE_ORDER_CONFLICT}:
            raise ValueError("composition state is invalid")

    def stable_key(self) -> tuple[int, ...]:
        result = [
            INPUT_DISCOURSE_OPERATION_VERSION,
            *_pack(self.operation_key),
            *_pack(self.operation_span_ref),
        ]
        for scope, relations, state in zip(
                self.operand_scope_refs, self.operand_relation_keys,
                self.operand_states):
            result.extend((*_pack(scope), state, len(relations)))
            for relation in relations:
                result.extend(_pack(relation))
        result.append(self.state)
        return tuple(result)


@dataclass(frozen=True, slots=True)
class InputDiscourseGap:
    """An integer-only unresolved boundary between complete clause scopes."""

    gap_span_ref: tuple[int, ...]
    left_relation_keys: tuple[tuple[int, ...], ...]
    right_relation_keys: tuple[tuple[int, ...], ...]
    state: int = STRUCTURE_OPEN

    def __post_init__(self) -> None:
        _strict_key(self.gap_span_ref, where="discourse gap span")
        for name in ("left_relation_keys", "right_relation_keys"):
            values = getattr(self, name)
            if (type(values) is not tuple
                    or values != tuple(sorted(set(values)))):
                raise ValueError("discourse gap relation keys must be canonical")
            for value in values:
                _strict_key(value, where="discourse gap relation key")
        if self.state not in {
                STRUCTURE_OPEN, STRUCTURE_CLOSED, STRUCTURE_ORDER_CONFLICT}:
            raise ValueError("discourse gap state is invalid")

    def stable_key(self) -> tuple[int, ...]:
        result = [INPUT_DISCOURSE_OPERATION_VERSION, *_pack(self.gap_span_ref)]
        for group in (self.left_relation_keys, self.right_relation_keys):
            result.append(len(group))
            for relation in group:
                result.extend(_pack(relation))
        result.append(self.state)
        return tuple(result)


@dataclass(frozen=True, slots=True)
class InputRelationQuestionBinding:
    """Current-input syntax slots joined to one independently closed relation."""

    frame_key: tuple[int, ...]
    operation_key: tuple[int, ...]
    relation_key: tuple[int, ...]
    slot_span_refs: tuple[tuple[int, ...], ...]
    answer_role_key: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        for name in ("frame_key", "operation_key", "relation_key"):
            _strict_key(getattr(self, name), where=f"question binding {name}")
        _strict_key(self.answer_role_key,
                    where="question binding answer_role_key", allow_empty=True)
        if (type(self.slot_span_refs) is not tuple
                or len(self.slot_span_refs) < (1 if self.answer_role_key else 2)
                or (self.answer_role_key and len(self.slot_span_refs) != 1)
                or len(set(self.slot_span_refs)) != len(self.slot_span_refs)):
            raise ValueError("question binding role slot spans are invalid")
        for ref in self.slot_span_refs:
            _strict_key(ref, where="question slot span")

    def stable_key(self) -> tuple[int, ...]:
        if self.answer_role_key:
            return (
                92032, *_pack(self.frame_key), *_pack(self.operation_key),
                *_pack(self.relation_key), len(self.slot_span_refs),
                *(value for ref in self.slot_span_refs for value in _pack(ref)),
                *_pack(self.answer_role_key),
            )
        return (
            92024, *_pack(self.frame_key), *_pack(self.operation_key),
            *_pack(self.relation_key),
            *(value for ref in self.slot_span_refs for value in _pack(ref)),
        )


@dataclass(frozen=True, slots=True)
class InputPendingRelationComposition:
    """Full question syntax bound to an unresolved algebra candidate, not a fact."""

    frame_key: tuple[int, ...]
    operation_key: tuple[int, ...]
    pending_key: tuple[int, ...]
    slot_span_refs: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        for key in (self.frame_key, self.operation_key, self.pending_key):
            _strict_key(key, where="pending composition query")
        if (type(self.slot_span_refs) is not tuple or len(self.slot_span_refs) != 2
                or len(set(self.slot_span_refs)) != 2):
            raise ValueError("pending composition requires two distinct exact role spans")
        for key in self.slot_span_refs:
            _strict_key(key, where="pending composition span")

    def stable_key(self) -> tuple[int, ...]:
        return (92111, *_pack(self.frame_key), *_pack(self.operation_key),
                *_pack(self.pending_key), *(_pack(self.slot_span_refs[0])),
                *(_pack(self.slot_span_refs[1])))


def scope_local_conjunction_candidate_keys(
        structure: "QueryInputStructure",
        authorized_operation_keys: frozenset[tuple[int, ...]],
        ) -> tuple[tuple[tuple[int, ...], tuple[tuple[int, ...], ...]], ...]:
    """Return exact closed relation candidates for each authorized scope join.

    The projector deliberately retains every competing relation candidate.
    This helper consumes that integer record only when each operand scope has
    exactly one locally closed candidate identity.  It never chooses by
    proposition key, score, surface order, or character similarity.
    """
    if not authorized_operation_keys or structure.discourse_gaps:
        return ()
    spans = {item.ref_key: item for item in structure.spans}
    candidates = {item.stable_key(): item for item in structure.relation_candidates}
    semantic = structure.semantic_candidates

    def local_closed(candidate: InputRelationCandidate,
                     scope_ref: tuple[int, ...]) -> bool:
        scope = spans.get(scope_ref)
        if scope is None:
            return False
        local_refs = {
            ref for ref in candidate.span_refs
            if (ref in spans and scope.start <= spans[ref].start
                and spans[ref].end <= scope.end)
        }
        ordinals: dict[int, set[tuple[int, ...]]] = {}
        for item in semantic:
            if (item.proposition_key == candidate.proposition_key
                    and item.source_ref == candidate.source_ref
                    and item.span_ref in local_refs):
                ordinals.setdefault(item.member_ordinal, set()).add(item.span_ref)
        if candidate.required_count == 2 and candidate.predicate_coverage == 0:
            ordered_spans = {
                ordinal: {spans[ref] for ref in refs}
                for ordinal, refs in ordinals.items()
            }
            return (len(ordinals) == 2
                    and _ordered_span_chain_exists(
                        ordered_spans, cursor=scope.start))
        if len(ordinals) < candidate.required_count:
            return False
        ordered_spans = {
            ordinal: {spans[ref] for ref in refs}
            for ordinal, refs in ordinals.items()
        }
        if not _ordered_span_chain_exists(
                ordered_spans, cursor=scope.start):
            return False
        return len(ordinals) == candidate.required_count

    bindings = []
    for composition in structure.discourse_compositions:
        if (composition.state != STRUCTURE_CLOSED
                or composition.operation_key not in authorized_operation_keys
                or any(state != STRUCTURE_CLOSED
                       for state in composition.operand_states)):
            continue
        selected = []
        for scope_ref, relation_keys in zip(
                composition.operand_scope_refs,
                composition.operand_relation_keys):
            local_candidates = tuple(
                candidates[key]
                for key in relation_keys
                if key in candidates
                and candidates[key].state == STRUCTURE_CLOSED
                and local_closed(candidates[key], scope_ref)
            )
            local_groups = relation_candidate_semantic_groups(
                structure, local_candidates)
            if len(local_groups) != 1:
                selected = []
                break
            # Keep one exact candidate identity for downstream graph joins;
            # other members remain available as provenance in the structure.
            selected.append(local_groups[0][1][0].stable_key())
        if (len(selected) == 2
                and candidates[selected[0]].proposition_key
                != candidates[selected[1]].proposition_key):
            bindings.append((composition.stable_key(), tuple(selected)))
    return tuple(sorted(set(bindings)))


def scope_local_conjunction_members(
        structure: "QueryInputStructure",
        authorized_operation_keys: frozenset[tuple[int, ...]],
        ) -> tuple[tuple[tuple[int, ...], tuple[tuple[int, ...], ...]], ...]:
    """Return graph-authorized, scope-local proposition members."""
    candidates = {
        item.stable_key(): item for item in structure.relation_candidates}
    return tuple(sorted({
        (composition_key, tuple(
            candidates[key].proposition_key for key in relation_keys))
        for composition_key, relation_keys
        in scope_local_conjunction_candidate_keys(
            structure, authorized_operation_keys)
    }))


def scope_local_fact_query_candidate_keys(
        structure: "QueryInputStructure",
        authorized_operation_keys: frozenset[tuple[int, ...]] | dict[
            tuple[int, ...], tuple[int, ...]],
        ) -> tuple[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]], ...]:
    """Return one closed relation scope for an authorized unary query.

    The legacy ``frozenset`` form authorizes the left-scope mode.  A mapping
    from operation key to integer mode can additionally authorize the right-
    scope mode.  The operation contract is supplied by a separate integer
    graph; this helper only validates exact spans, role ordinals and local
    closure.  It never selects by score, proposition order, or surface
    similarity.
    """
    if not authorized_operation_keys or structure.discourse_gaps:
        return ()
    mode_by_operation = (
        authorized_operation_keys
        if isinstance(authorized_operation_keys, dict)
        else {key: (91672, 1) for key in authorized_operation_keys}
    )
    # Count every authorized operation occurrence before inspecting scope
    # closure.  An additional occurrence may be open/partial and therefore
    # absent from ``result``; accepting the one closed occurrence would still
    # turn an ambiguous repeated fact-query marker into an answer contract.
    authorized_compositions = tuple(
        item for item in structure.discourse_compositions
        if item.operation_key in mode_by_operation)
    if len(authorized_compositions) != 1:
        return ()
    spans = {item.ref_key: item for item in structure.spans}
    candidates = {item.stable_key(): item for item in structure.relation_candidates}

    def local_closed(candidate: InputRelationCandidate,
                     scope_ref: tuple[int, ...]) -> bool:
        scope = spans.get(scope_ref)
        if scope is None or candidate.state != STRUCTURE_CLOSED:
            return False
        local_refs = {
            ref for ref in candidate.span_refs
            if ref in spans and scope.start <= spans[ref].start
            and spans[ref].end <= scope.end
        }
        ordinals: dict[int, set[InputSpan]] = {}
        for item in structure.semantic_candidates:
            if (item.proposition_key == candidate.proposition_key
                    and item.source_ref == candidate.source_ref
                    and item.span_ref in local_refs):
                ordinals.setdefault(item.member_ordinal, set()).add(
                    spans[item.span_ref])
        if candidate.required_count == 2 and candidate.predicate_coverage == 0:
            return (len(ordinals) == 2
                    and _ordered_span_chain_exists(ordinals,
                                                  cursor=scope.start))
        if len(ordinals) < candidate.required_count:
            return False
        if not _ordered_span_chain_exists(ordinals, cursor=scope.start):
            return False
        return len(ordinals) == candidate.required_count

    result = []
    for composition in structure.discourse_compositions:
        mode = mode_by_operation.get(composition.operation_key)
        if (mode is None or len(composition.operand_scope_refs) != 2
                or composition.state != STRUCTURE_OPEN):
            continue
        if mode == (91672, 1):
            scope_index = 0
            if (composition.operand_scope_refs[1]
                    or composition.operand_states != (
                        STRUCTURE_CLOSED, STRUCTURE_OPEN)):
                continue
        elif mode == (91672, 2):
            scope_index = 1
            if (composition.operand_scope_refs[0]
                    or composition.operand_states != (
                        STRUCTURE_OPEN, STRUCTURE_CLOSED)):
                continue
        else:
            continue
        local_candidates = tuple(
            candidates[key]
            for key in composition.operand_relation_keys[scope_index]
            if key in candidates and local_closed(
                candidates[key], composition.operand_scope_refs[scope_index]))
        local_groups = relation_candidate_semantic_groups(
            structure, local_candidates)
        if len(local_groups) == 1:
            result.append((composition.stable_key(), composition.operation_key,
                           local_groups[0][1][0].stable_key()))
    # Unary fact-query authority remains unique even when another operation
    # family is present in the same input.  Multiple fact-query matches stay
    # fail-closed; one fact-query composition may coexist with a conjunction
    # composition whose scopes are validated independently.
    unique = tuple(sorted(set(result)))
    operation_keys = {item[1] for item in unique}
    if len(operation_keys) != 1:
        return ()
    return unique if len(unique) == 1 else ()


def scope_local_general_operation_candidate_keys(
        structure: "QueryInputStructure",
        authorized_operation_keys: frozenset[tuple[int, ...]],
        ) -> tuple[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]], ...]:
    """Return one closed right scope for a general trained operation.

    General operations are surface-level discourse acts whose integer marker
    precedes one relation scope (for example a confirmation prefix).  The
    marker itself is not a fact-query mode and must not be allowed to select a
    relation by score or by proposition identity.  Only the exact shape
    ``(empty, closed)`` with one locally closed semantic group is consumable;
    open, conflicting, repeated, or competing scopes stay fail-closed.
    """
    if not authorized_operation_keys or structure.discourse_gaps:
        return ()
    authorized_compositions = tuple(
        item for item in structure.discourse_compositions
        if item.operation_key in authorized_operation_keys)
    if len(authorized_compositions) != 1:
        return ()
    if any(item.operation_key not in authorized_operation_keys
           for item in structure.discourse_compositions):
        return ()
    composition = authorized_compositions[0]
    if (composition.state != STRUCTURE_OPEN
            or composition.operand_scope_refs[0]
            or not composition.operand_scope_refs[1]
            or composition.operand_states != (
                STRUCTURE_OPEN, STRUCTURE_CLOSED)):
        return ()
    spans = {item.ref_key: item for item in structure.spans}
    scope = spans.get(composition.operand_scope_refs[1])
    if scope is None:
        return ()
    candidates = {item.stable_key(): item for item in structure.relation_candidates}

    def local_closed(candidate: InputRelationCandidate) -> bool:
        if candidate.state != STRUCTURE_CLOSED:
            return False
        local_refs = {
            ref for ref in candidate.span_refs
            if ref in spans and scope.start <= spans[ref].start
            and spans[ref].end <= scope.end
        }
        ordinals: dict[int, set[InputSpan]] = {}
        for item in structure.semantic_candidates:
            if (item.proposition_key == candidate.proposition_key
                    and item.source_ref == candidate.source_ref
                    and item.span_ref in local_refs):
                ordinals.setdefault(item.member_ordinal, set()).add(
                    spans[item.span_ref])
        if candidate.required_count == 2 and candidate.predicate_coverage == 0:
            return (len(ordinals) == 2
                    and _ordered_span_chain_exists(
                        ordinals, cursor=scope.start))
        if len(ordinals) < candidate.required_count:
            return False
        if not _ordered_span_chain_exists(ordinals, cursor=scope.start):
            return False
        return len(ordinals) == candidate.required_count

    local_candidates = tuple(
        candidates[key]
        for key in composition.operand_relation_keys[1]
        if key in candidates and local_closed(candidates[key]))
    local_groups = relation_candidate_semantic_groups(
        structure, local_candidates)
    if len(local_groups) != 1:
        return ()
    selected = local_groups[0][1][0].stable_key()
    return ((composition.stable_key(), composition.operation_key, selected),)


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
    discourse_compositions: tuple[InputDiscourseComposition, ...] = ()
    discourse_gaps: tuple[InputDiscourseGap, ...] = ()
    operation_keys: tuple[tuple[int, ...], ...] = ()
    relation_question_bindings: tuple[InputRelationQuestionBinding, ...] = ()
    pending_relation_composition_bindings: tuple[InputPendingRelationComposition, ...] = ()
    operation_request_bindings: tuple[InputOperationRequestBinding, ...] = ()
    assertion_scope_bindings: tuple[InputAssertionScopeBinding, ...] = ()
    # Query-bound proof of a training source.  Each entry is
    # ``(source_ref, proof_key)`` and contains only validated integers.  It is
    # deliberately kept out of ``stable_key``: provenance changes grouping at
    # the consumer boundary, while candidate/evidence identities remain the
    # exact graph identities restored from the model.
    source_provenance_keys: tuple[
        tuple[tuple[int, ...], tuple[int, ...]], ...] = ()

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
            ("discourse_compositions", InputDiscourseComposition),
            ("discourse_gaps", InputDiscourseGap),
            ("relation_question_bindings", InputRelationQuestionBinding),
            ("pending_relation_composition_bindings", InputPendingRelationComposition),
            ("operation_request_bindings", InputOperationRequestBinding),
            ("assertion_scope_bindings", InputAssertionScopeBinding),
        )
        for name, expected in specifications:
            values = getattr(self, name)
            if type(values) is not tuple or any(
                    not isinstance(item, expected) for item in values):
                raise TypeError(f"QueryInputStructure.{name} 类型非法")
            if values != tuple(sorted(
                    set(values), key=lambda item: item.stable_key())):
                raise ValueError(f"QueryInputStructure.{name} 必须规范排序去重")
        if (type(self.source_provenance_keys) is not tuple
                or any(type(item) is not tuple or len(item) != 2
                       or type(item[0]) is not tuple
                       or type(item[1]) is not tuple
                       for item in self.source_provenance_keys)
                or self.source_provenance_keys != tuple(sorted(
                    set(self.source_provenance_keys), key=lambda item: item[0]))):
            raise ValueError(
                "QueryInputStructure source_provenance_keys 必须规范排序去重")
        for source_ref, proof_key in self.source_provenance_keys:
            _strict_key(source_ref,
                        where="QueryInputStructure.source_provenance.source_ref")
            _strict_key(proof_key,
                        where="QueryInputStructure.source_provenance.proof_key")
        if (type(self.graph_object_keys) is not tuple
                or any(type(key) is not tuple or not key
                       or any(type(item) is not int or item < 0 for item in key)
                       for key in self.graph_object_keys)
                or self.graph_object_keys != tuple(sorted(set(self.graph_object_keys)))):
            raise ValueError("QueryInputStructure graph_object_keys 必须是规范纯整数键")
        if (type(self.operation_keys) is not tuple
                or any(type(key) is not tuple or not key
                       or any(type(item) is not int or item < 0 for item in key)
                       for key in self.operation_keys)
                or self.operation_keys != tuple(sorted(set(self.operation_keys)))):
            raise ValueError("QueryInputStructure operation_keys 必须是规范纯整数键")
        graph_only = not self.tokens and not self.spans
        if graph_only:
            if (not self.graph_object_keys
                    or any((self.precedes, self.contains, self.carrier_candidates,
                            self.semantic_candidates, self.relation_candidates,
                            self.concept_links, self.concept_routes,
                            self.open_relations, self.discourse_compositions,
                            self.discourse_gaps,
                            self.relation_question_bindings,
                            self.pending_relation_composition_bindings, self.operation_request_bindings,
                            self.assertion_scope_bindings))
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
        relation_keys = {item.stable_key() for item in self.relation_candidates}
        if self.assertion_scope_bindings:
            if (len(self.assertion_scope_bindings) != 1 or self.relation_question_bindings
                    or self.discourse_compositions or self.discourse_gaps
                    or self.pending_relation_composition_bindings or self.operation_request_bindings):
                raise ValueError("assertion scope 需要唯一完整且不混合的输入")
            binding = self.assertion_scope_bindings[0]
            if (binding.source_ref != self.source_ref or binding.relation_key not in relation_keys
                    or any(ref not in span_refs for ref in binding.slot_span_refs)):
                raise ValueError("assertion scope 丢失当前来源、关系或角色跨度")
            from pure_integer_ai.cognition.understanding.query_assertion_scope import (
                TrainedAssertionScopeFrame, match_assertion_scope_frames,
            )
            if match_assertion_scope_frames((TrainedAssertionScopeFrame.from_stable_key(
                    binding.frame_key),), self) != self.assertion_scope_bindings:
                raise ValueError("assertion scope 没有完整输入语法及精确角色证明")
        if any(
                item.relation_key not in relation_keys
                or item.operation_key not in self.operation_keys
                or any(ref not in span_refs for ref in item.slot_span_refs)
                for item in self.relation_question_bindings):
            raise ValueError("question binding is not closed in current input")
        if self.pending_relation_composition_bindings:
            if (self.relation_question_bindings or self.discourse_compositions or self.discourse_gaps
                    or len(self.pending_relation_composition_bindings) != 1):
                raise ValueError("pending composition query needs one complete unambiguous question")
            if any(binding.operation_key not in self.operation_keys
                   or any(ref not in span_refs for ref in binding.slot_span_refs)
                   for binding in self.pending_relation_composition_bindings):
                raise ValueError("pending composition query spans/operation are not in the current input")
        if self.operation_request_bindings:
            if (len(self.operation_request_bindings) != 1 or self.relation_question_bindings
                    or self.pending_relation_composition_bindings or self.discourse_compositions or self.discourse_gaps):
                raise ValueError("operation request needs one complete unambiguous input interface")
            members = {item.stable_key(): item for item in self.semantic_candidates}
            binding = self.operation_request_bindings[0]
            if binding.source_ref != self.source_ref or self.operation_keys != (binding.operation_key,):
                raise ValueError("operation request crosses the actual input source/operation")
            for _variable, target, span, projections in binding.readings:
                if span not in span_refs or any(key not in members or members[key].span_ref != span
                        or members[key].candidate_key != target for key in projections):
                    raise ValueError("operation request loses its actual span/alias projection")
        for composition in self.discourse_compositions:
            referenced_spans = {
                composition.operation_span_ref,
                *(item for item in composition.operand_scope_refs if item),
            }
            if not referenced_spans <= span_refs:
                raise ValueError("discourse composition 引用了未知输入 span")
            if any(key not in relation_keys
                   for group in composition.operand_relation_keys
                   for key in group):
                raise ValueError(
                    "discourse composition 引用了未知 relation candidate")
        for gap in self.discourse_gaps:
            if gap.gap_span_ref not in span_refs:
                raise ValueError("discourse gap 引用了未知输入 span")
            if any(key not in relation_keys
                   for key in (*gap.left_relation_keys,
                               *gap.right_relation_keys)):
                raise ValueError("discourse gap 引用了未知 relation candidate")
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
            (7 if self.assertion_scope_bindings else 6 if self.operation_request_bindings else 5 if self.pending_relation_composition_bindings else QUERY_INPUT_TRACE_VERSION
             if (self.discourse_compositions or self.discourse_gaps
                 or self.relation_question_bindings) else 3),
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
        if self.discourse_compositions:
            result.extend((INPUT_DISCOURSE_OPERATION_VERSION,
                           len(self.discourse_compositions)))
            for item in self.discourse_compositions:
                result.extend(_pack(item.stable_key()))
        if self.discourse_gaps:
            result.extend((INPUT_DISCOURSE_OPERATION_VERSION,
                           len(self.discourse_gaps)))
            for item in self.discourse_gaps:
                result.extend(_pack(item.stable_key()))
        if self.graph_object_keys:
            result.append(len(self.graph_object_keys))
            for key in self.graph_object_keys:
                result.extend(_pack(key))
        if self.operation_keys:
            result.append(len(self.operation_keys))
            for key in self.operation_keys:
                result.extend(_pack(key))
        if self.relation_question_bindings:
            result.append(len(self.relation_question_bindings))
            for binding in self.relation_question_bindings:
                result.extend(_pack(binding.stable_key()))
        if self.pending_relation_composition_bindings:
            result.append(len(self.pending_relation_composition_bindings))
            for binding in self.pending_relation_composition_bindings:
                result.extend(_pack(binding.stable_key()))
        if self.operation_request_bindings:
            result.append(len(self.operation_request_bindings))
            for binding in self.operation_request_bindings:
                result.extend(_pack(binding.stable_key()))
        if self.assertion_scope_bindings:
            result.append(len(self.assertion_scope_bindings))
            for binding in self.assertion_scope_bindings:
                result.extend(_pack(binding.stable_key()))
        return tuple(result)

    def integer_trace(self) -> dict[str, object]:
        """投影为仅含整数值的 JSON 兼容 trace。"""
        return {
            "schema_version": (
                7 if self.assertion_scope_bindings else 6 if self.operation_request_bindings else 5 if self.pending_relation_composition_bindings else QUERY_INPUT_TRACE_VERSION
                if (self.discourse_compositions or self.discourse_gaps
                    or self.relation_question_bindings) else 3),
            "source_ref": list(self.source_ref),
            "assertion_scope_bindings": [list(item.stable_key()) for item in self.assertion_scope_bindings],
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
            "discourse_compositions": [
                list(item.stable_key()) for item in self.discourse_compositions],
            "discourse_gaps": [
                list(item.stable_key()) for item in self.discourse_gaps],
            "graph_object_keys": [list(key) for key in self.graph_object_keys],
            "operation_keys": [list(key) for key in self.operation_keys],
            "relation_question_bindings": [
                list(item.stable_key()) for item in self.relation_question_bindings],
            "pending_relation_composition_bindings": [
                list(item.stable_key()) for item in self.pending_relation_composition_bindings],
            **({"operation_request_bindings": [list(item.stable_key()) for item in self.operation_request_bindings]}
               if self.operation_request_bindings else {}),
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
        closed_semantic_keys = set()
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
                closed_semantic_keys.add(candidate.stable_key())
        for relation in self.relation_candidates:
            if relation.proposition_key != proposition or relation.state != STRUCTURE_CLOSED:
                continue
            by_ordinal = members.get(relation.source_ref, {})
            # Pair/inverse candidates intentionally omit the surface
            # predicate member. Their two filler spans are nevertheless a
            # complete graph-backed closure and must not be rejected for the
            # absent lexical cue.
            if relation.required_count == 2 and relation.predicate_coverage == 0:
                if (len(by_ordinal) == 2
                        and _ordered_span_chain_exists(by_ordinal)):
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

        # 问式槽序与来源成员序分别保留；精确语法和已闭合成员交给专用消费者。
        from pure_integer_ai.cognition.understanding.query_question_evidence import (
            complete_question_evidence_closed,
        )
        return complete_question_evidence_closed(
            self, proposition, evidence_claims, frozenset(closed_semantic_keys))


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
                 open_role_frames: tuple[OpenRoleFrame, ...] = (),
                 envelope_role_frames: tuple[OpenRoleFrame, ...] = (),
                 discourse_operators: tuple[
                     TrainedInputDiscourseOperator, ...] = (),
                 slot_question_frames: tuple = (),
                 pair_closure_requires_predicate_keys: frozenset[
                     tuple[int, ...]] = frozenset()) -> None:
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
        if type(envelope_role_frames) is not tuple or any(
                not isinstance(item, OpenRoleFrame) for item in envelope_role_frames):
            raise TypeError("envelope_role_frames 必须是训练后 OpenRoleFrame tuple")
        self.envelope_role_frames = tuple(sorted(set(envelope_role_frames),
                                                key=lambda item: item.stable_key()))
        from pure_integer_ai.cognition.understanding.query_relation_questions import TrainedRelationQuestionFrame
        if (type(slot_question_frames) is not tuple or any(
                not isinstance(item, TrainedRelationQuestionFrame) or not item.is_slot_question
                for item in slot_question_frames)):
            raise TypeError("slot_question_frames 必须是已训练 WH 整数帧")
        self.slot_question_frames = tuple(sorted(set(slot_question_frames),
                                                 key=lambda item: item.stable_key()))
        if (type(discourse_operators) is not tuple
                or any(not isinstance(item, TrainedInputDiscourseOperator)
                       for item in discourse_operators)):
            raise TypeError("discourse_operators must be trained integer records")
        self.discourse_operators = tuple(sorted(
            set(discourse_operators), key=lambda item: item.stable_key()))
        if (not isinstance(pair_closure_requires_predicate_keys, frozenset)
                or any(type(item) is not tuple or not item
                       or any(type(value) is not int or value < 0
                              for value in item)
                       for item in pair_closure_requires_predicate_keys)):
            raise TypeError(
                "pair_closure_requires_predicate_keys 必须是整数 predicate key frozenset")
        self._pair_closure_requires_predicate_keys = frozenset(
            pair_closure_requires_predicate_keys)
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
        single_postings: dict[int, list[TrainedInputMember]] = {}
        prefix_postings: dict[tuple[int, int], list[TrainedInputMember]] = {}
        fillers: dict[tuple[int, ...], list[TrainedInputMember]] = {}
        structures_by_first: dict[int, list[TrainedInputStructure]] = {}
        for structure in self.structures:
            for first in {item.values[0] for item in structure.members}:
                structures_by_first.setdefault(first, []).append(structure)
            for member in structure.members:
                if len(member.values) == 1:
                    single_postings.setdefault(member.values[0], []).append(member)
                else:
                    prefix_postings.setdefault(
                        (member.values[0], member.values[1]), []).append(member)
                if member.projection_kind == PROJECTION_FILLER:
                    fillers.setdefault(member.candidate_key, []).append(member)
        self._fillers = {key: tuple(sorted(set(items), key=lambda item: item.stable_key()))
                         for key, items in fillers.items()}
        self._single_postings = {
            value: tuple(sorted(items, key=lambda item: item.stable_key()))
            for value, items in single_postings.items()
        }
        self._prefix_postings = {
            prefix: tuple(sorted(items, key=lambda item: item.stable_key()))
            for prefix, items in prefix_postings.items()
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

    def _matching_members(
            self, values: tuple[int, ...],
            ) -> Iterator[tuple[int, TrainedInputMember]]:
        """Yield all exact member spans using a two-codepoint prefix index."""
        for start, value in enumerate(values):
            for member in self._single_postings.get(value, ()):
                yield start, member
            if start + 1 >= len(values):
                continue
            prefix = (value, values[start + 1])
            for member in self._prefix_postings.get(prefix, ()):
                end = start + len(member.values)
                if end <= len(values) and values[start:end] == member.values:
                    yield start, member

    @staticmethod
    def _select_operation_matches(
            matches: list[tuple[int, int, TrainedInputDiscourseOperator]],
            ) -> list[tuple[int, int, TrainedInputDiscourseOperator]]:
        """Resolve nested surfaces for one abstract operation deterministically.

        A trained operation may have several surface realizations.  When one
        realization is a strict interval superset of another realization of
        the same operation (for example ``?`` inside ``吗？``), the shorter
        marker is only a suffix/prefix observation of the longer marker and
        must not create a second operation scope.  Distinct operation keys or
        partially-overlapping alternatives remain present and therefore still
        fail closed at the unique-operation contract.
        """
        by_operation: dict[tuple[int, ...], list[tuple[int, int,
                                                         TrainedInputDiscourseOperator]]] = {}
        for row in matches:
            by_operation.setdefault(row[2].operation_key, []).append(row)
        selected: list[tuple[int, int, TrainedInputDiscourseOperator]] = []
        for operation_key, rows in by_operation.items():
            ordered = sorted(rows, key=lambda item: (
                item[0], -(item[1] - item[0]), item[1], item[2].stable_key()))
            kept = []
            for candidate in ordered:
                contained_by = any(
                    prior[0] <= candidate[0]
                    and candidate[1] <= prior[1]
                    and (prior[0], prior[1]) != (candidate[0], candidate[1])
                    for prior in kept)
                if contained_by:
                    continue
                kept.append(candidate)
            selected.extend(kept)
        return sorted(selected, key=lambda item: (
            item[0], item[1], item[2].stable_key()))

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
        for start, member in self._matching_members(values):
            end = start + len(member.values)
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
        matched_members = dict(direct_members)
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
                        matched_members[target_member.stable_key()] = target_member
                        matches.setdefault(target_member.stable_key(), []).append(span)
                        semantic_candidates.append(InputSemanticCandidate(
                            target, target_member.object_kind, PROJECTION_FILLER,
                            span.ref_key, target_member.proposition_key,
                            target_member.predicate_key, target_member.role_key,
                            target_member.source_ref, target_member.member_ordinal,
                            route.stable_key()))
        if self.slot_question_frames:
            from pure_integer_ai.cognition.understanding.query_slot_question_scope import slot_question_projection_roles
            groups = {}
            for key, member in matched_members.items():
                if member.projection_kind == PROJECTION_FILLER:
                    groups.setdefault((member.predicate_key, member.proposition_key,
                                       member.source_ref), set()).update(
                        (member.role_key, span.start, span.end) for span in matches[key])
            role_span_groups = {}
            for (predicate, _proposition, _source), group in groups.items():
                role_span_groups.setdefault(predicate, []).append(frozenset(group))
            projection_roles = slot_question_projection_roles(
                self.slot_question_frames, values, source_ref,
                role_span_groups={key: tuple(items) for key, items in role_span_groups.items()})
            grounded_roles = {predicate: roles for predicate, roles in projection_roles.items()
                if any(roles.issubset(group) for group in role_span_groups.get(predicate, ()))}
            for key, member in matched_members.items():
                allowed_roles = projection_roles.get(member.predicate_key)
                if member.projection_kind == PROJECTION_FILLER and allowed_roles is not None:
                    # 保留所有字面观察和路径，只让完整语法实际观察的角色和
                    # 精确跨度计入覆盖。答案名称的内部子串不能填补答案角色。
                    direct_roles = grounded_roles.get(member.predicate_key)
                    matches[key] = [span for span in matches[key]
                        if (span in direct_matches.get(key, ()) and direct_roles is None)
                        or (member.role_key, span.start, span.end) in allowed_roles]
            matched_span_by_ref = {span.ref_key: span for span in span_by_interval.values()}
            semantic_candidates = [item for item in semantic_candidates
                if (not item.concept_route_key or item.predicate_key not in projection_roles
                    or (item.role_key, matched_span_by_ref[item.span_ref].start,
                        matched_span_by_ref[item.span_ref].end)
                    in projection_roles[item.predicate_key])]
        normalized_matches = {
            key: tuple(sorted(set(items), key=lambda item: item.stable_key()))
            for key, items in matches.items()
        }
        matched_spans = tuple(sorted(
            span_by_interval.values(), key=lambda item: item.stable_key()))
        operation_matches = []
        for operator in self.discourse_operators:
            width = len(operator.surface_values)
            for start in range(0, len(values) - width + 1):
                if values[start:start + width] == operator.surface_values:
                    operation_matches.append((start, start + width, operator))
        operation_matches = self._select_operation_matches(operation_matches)
        operation_spans: set[InputSpan] = set()
        operand_scopes: dict[tuple[int, ...], InputSpan] = {}
        operation_scope_rows = []
        for index, (start, end, operator) in enumerate(operation_matches):
            left_start = operation_matches[index - 1][1] if index else 0
            right_end = (operation_matches[index + 1][0]
                         if index + 1 < len(operation_matches) else len(values))
            operation_span = InputSpan(
                source_ref, start, end, SPAN_TRAINED,
                operator.surface_values)
            operation_spans.add(operation_span)
            scope_refs = []
            for scope_start, scope_end in ((left_start, start), (end, right_end)):
                if scope_start >= scope_end:
                    scope_refs.append(())
                    continue
                scope = InputSpan(
                    source_ref, scope_start, scope_end, SPAN_SOURCE,
                    values[scope_start:scope_end])
                operand_scopes[scope.ref_key] = scope
                scope_refs.append(scope.ref_key)
            operation_scope_rows.append(
                (operator, operation_span, tuple(scope_refs)))
        spans = tuple(sorted(
            {source_span, *matched_spans, *operation_spans,
             *operand_scopes.values()}, key=lambda item: item.stable_key()))

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
                and (structure.predicate_key
                     not in self._pair_closure_requires_predicate_keys)
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

        open_frames = self.open_role_frames
        if self.slot_question_frames:
            from pure_integer_ai.cognition.understanding.query_slot_question_composition import composed_slot_open_role_frames
            derived = composed_slot_open_role_frames(
                self.slot_question_frames, open_frames, values)
            open_frames = tuple(sorted(set((*open_frames, *derived)),
                                       key=lambda item: item.stable_key()))
        open_relations = match_open_roles(open_frames, values, source_ref)
        if not open_relations and self.envelope_role_frames:
            # 外围 carrier 不在当前输入时，先消费图中明确训练的命题包络。
            # 必须保留全部内部间隔；不能只留下谓词而把其余连接结构丢掉。
            open_relations = match_open_roles(
                self.envelope_role_frames, values, source_ref)
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
        # Dynamic role spans have an exact input interval from a trained
        # open-role parse even when no surface member projects an entity.
        # Keep those spans available for reference resolution without adding
        # semantic candidates or Core support for the unknown filler.
        known_intervals = {(item.start, item.end) for item in spans}
        open_role_spans = tuple(
            InputSpan(source_ref, binding.start, binding.end, SPAN_SOURCE,
                      binding.values)
            for candidate in open_relations
            for binding in candidate.bindings
            if (binding.start, binding.end) not in known_intervals)
        if open_role_spans:
            spans = tuple(sorted(
                {*spans, *open_role_spans},
                key=lambda item: item.stable_key()))
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
                            and (left.start, left.end)
                            != (right.start, right.end)):
                        contains.append(InputSpanRelation(
                            left.ref_key, right.ref_key))
        # Do not manufacture an open relation from arbitrary code-point
        # diversity.  A generic frame has no predicate/span evidence and would
        # turn every unknown sentence into a character-position split, which
        # is both a false graph claim and a hidden character-neighbour route.
        # Uncovered input remains a source-local generic Memory Observation;
        # real open relations still come from trained structure frames above.

        span_by_ref = {item.ref_key: item for item in spans}
        relation_rows = tuple(sorted(
            set(relation_candidates), key=lambda item: item.stable_key()))

        def local_relation_state(candidate, local_spans):
            """Recheck ordered member coverage inside one discourse scope."""
            local_refs = {item.ref_key for item in local_spans}
            members_by_ordinal: dict[int, set[InputSpan]] = {}
            for semantic in semantic_candidates:
                if (semantic.proposition_key == candidate.proposition_key
                        and semantic.source_ref == candidate.source_ref
                        and semantic.span_ref in local_refs):
                    members_by_ordinal.setdefault(
                        semantic.member_ordinal, set()).add(
                            span_by_ref[semantic.span_ref])
            if (candidate.required_count == 2
                    and candidate.predicate_coverage == 0):
                return (STRUCTURE_CLOSED
                        if (len(members_by_ordinal) == 2
                            and _ordered_span_chain_exists(
                                members_by_ordinal,
                                cursor=min(item.start for item in local_spans)))
                        else STRUCTURE_OPEN)
            if len(members_by_ordinal) < candidate.required_count:
                return STRUCTURE_OPEN
            ordinals = tuple(sorted(members_by_ordinal))
            if not _ordered_span_chain_exists(
                    members_by_ordinal,
                    cursor=min(item.start for item in local_spans)):
                return STRUCTURE_ORDER_CONFLICT
            return (STRUCTURE_CLOSED
                    if len(ordinals) == candidate.required_count
                    else STRUCTURE_OPEN)

        def operand_relations(scope_ref: tuple[int, ...]):
            if not scope_ref:
                return (), STRUCTURE_OPEN
            scope = span_by_ref[scope_ref]
            selected = []
            complete = []
            conflict = False
            for candidate in relation_rows:
                candidate_spans = tuple(
                    span_by_ref[ref] for ref in candidate.span_refs
                    if ref in span_by_ref)
                local = tuple(item for item in candidate_spans
                              if scope.start <= item.start
                              and item.end <= scope.end)
                if not local:
                    continue
                selected.append(candidate)
                local_state = local_relation_state(candidate, local)
                if local_state == STRUCTURE_ORDER_CONFLICT:
                    conflict = True
                if local_state == STRUCTURE_CLOSED:
                    complete.append(candidate)
            if conflict:
                state = STRUCTURE_ORDER_CONFLICT
            elif complete:
                # Partial projections from other trained propositions are
                # retained as UNKNOWN candidates, but must not reopen a scope
                # when one semantic topology has the sole complete role chain.
                # A second topology remains a real ambiguity.
                complete_keys = {
                    _relation_candidate_semantic_key(semantic_candidates, item)
                    for item in complete
                }
                competing_complete = any(
                    local_relation_state(
                        item,
                        tuple(span_by_ref[ref] for ref in item.span_refs
                              if ref in span_by_ref
                              and scope.start <= span_by_ref[ref].start
                              and span_by_ref[ref].end <= scope.end),
                    ) == STRUCTURE_CLOSED
                    and _relation_candidate_semantic_key(
                        semantic_candidates, item) not in complete_keys
                    for item in selected)
                state = (STRUCTURE_CLOSED
                         if len(complete_keys) == 1 and not competing_complete
                         else STRUCTURE_OPEN)
            else:
                state = STRUCTURE_OPEN
            return tuple(sorted(
                (item.stable_key() for item in selected))), state

        compositions = []
        for operator, operation_span, scope_refs in operation_scope_rows:
            operand_rows = tuple(operand_relations(item) for item in scope_refs)
            operand_keys = (operand_rows[0][0], operand_rows[1][0])
            operand_states = (operand_rows[0][1], operand_rows[1][1])
            if STRUCTURE_ORDER_CONFLICT in operand_states:
                state = STRUCTURE_ORDER_CONFLICT
            elif all(item == STRUCTURE_CLOSED for item in operand_states):
                state = STRUCTURE_CLOSED
            else:
                state = STRUCTURE_OPEN
            compositions.append(InputDiscourseComposition(
                operator.operation_key,
                operation_span.ref_key,
                scope_refs,
                operand_keys,
                operand_states,
                state,
            ))

        # A boundary that is not authorized by a trained operator is still
        # meaningful when graph evidence proves two separate clause scopes.
        # Materialize it as an unresolved integer gap so the bridge cannot
        # silently answer only one side.  This does not classify the boundary
        # by punctuation, character class, or a fixed vocabulary.
        operator_intervals = {
            (span.start, span.end) for _start, _end, _operator in operation_matches
            for span in (
                InputSpan(source_ref, _start, _end, SPAN_TRAINED,
                          values[_start:_end]),)
        }
        scoped_candidates = []
        for candidate in relation_rows:
            if candidate.state not in {
                    STRUCTURE_CLOSED, STRUCTURE_ORDER_CONFLICT}:
                continue
            candidate_spans = tuple(
                span_by_ref[ref] for ref in candidate.span_refs
                if ref in span_by_ref)
            if not candidate_spans:
                continue
            ordered = tuple(sorted(candidate_spans,
                                  key=lambda item: (item.start, item.end)))
            clusters: list[list[InputSpan]] = [[]]
            for span in ordered:
                if clusters[-1] and span.start > clusters[-1][-1].end:
                    clusters.append([])
                clusters[-1].append(span)
            for cluster in clusters:
                if len(cluster) < candidate.required_count:
                    continue
                scoped_candidates.append((
                    min(item.start for item in cluster),
                    max(item.end for item in cluster),
                    candidate,
                ))
        gaps_by_interval: dict[tuple[int, int], list[InputDiscourseGap]] = {}
        gap_spans: set[InputSpan] = set()
        for left_start, left_end, left in scoped_candidates:
            for right_start, right_end, right in scoped_candidates:
                if left_end >= right_start:
                    continue
                gap_interval = (left_end, right_start)
                if gap_interval in operator_intervals:
                    continue
                gap_span = InputSpan(
                    source_ref, gap_interval[0], gap_interval[1], SPAN_SOURCE,
                    values[gap_interval[0]:gap_interval[1]])
                gap_spans.add(gap_span)
                gap_state = (STRUCTURE_ORDER_CONFLICT
                             if STRUCTURE_ORDER_CONFLICT in {
                                 left.state, right.state}
                             else STRUCTURE_OPEN)
                gap = InputDiscourseGap(
                    gap_span.ref_key,
                    (left.stable_key(),),
                    (right.stable_key(),),
                    gap_state,
                )
                gaps_by_interval.setdefault(gap_interval, []).append(gap)
        gaps = []
        for gap_interval, candidates in gaps_by_interval.items():
            unique = tuple(sorted(set(candidates), key=lambda item: item.stable_key()))
            # Preserve competing witnesses in the integer key while exposing
            # one canonical unresolved gap record for QueryState gating.
            first = unique[0]
            if len(unique) > 1:
                first = InputDiscourseGap(
                    first.gap_span_ref,
                    tuple(sorted({key for item in unique
                                  for key in item.left_relation_keys})),
                    tuple(sorted({key for item in unique
                                  for key in item.right_relation_keys})),
                    (STRUCTURE_ORDER_CONFLICT
                     if any(item.state == STRUCTURE_ORDER_CONFLICT
                            for item in unique) else STRUCTURE_OPEN),
                )
            gaps.append(first)

        final_spans = tuple(sorted(
            {*spans, *gap_spans}, key=lambda item: item.stable_key()))
        # Operation identity is a result of matching a trained integer surface
        # operator in this input.  Callers may not manufacture it by passing a
        # course label alongside an otherwise unrelated surface.
        operation_keys = tuple(sorted({
            operator.operation_key for _start, _end, operator in operation_matches
        }))
        return QueryInputStructure(
            source_ref,
            tuple(sorted(tokens, key=lambda item: item.stable_key())),
            source_span.ref_key,
            final_spans,
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
            discourse_compositions=tuple(sorted(
                compositions, key=lambda item: item.stable_key())),
            discourse_gaps=tuple(sorted(
                gaps, key=lambda item: item.stable_key())),
            operation_keys=operation_keys,
        )


__all__ = [
    "CARRIER_GRAPH_SOURCE",
    "CARRIER_SEMANTIC_GRAPH",
    "INPUT_DISCOURSE_OPERATION_VERSION",
    "INPUT_STRUCTURE_VERSION",
    "PROJECTION_FILLER",
    "PROJECTION_PREDICATE",
    "SPAN_SOURCE",
    "SPAN_TRAINED",
    "STRUCTURE_CLOSED",
    "STRUCTURE_OPEN",
    "STRUCTURE_ORDER_CONFLICT",
    "InputCarrierCandidate",
    "InputDiscourseComposition",
    "InputDiscourseGap",
    "InputRelationQuestionBinding",
    "scope_local_conjunction_candidate_keys",
    "scope_local_conjunction_members",
    "scope_local_fact_query_candidate_keys",
    "scope_local_general_operation_candidate_keys",
    "InputRelationCandidate",
    "InputSemanticCandidate",
    "InputSpan",
    "InputSpanRelation",
    "InputToken",
    "QUERY_GRAPH_INPUT_VERSION",
    "QueryInputStructure",
    "relation_candidate_semantic_groups",
    "relation_candidate_semantic_key",
    "TrainedInputMember",
    "TrainedInputDiscourseOperator",
    "TrainedInputStructure",
    "TrainedInputStructureProjector",
    "graph_query_input_structure",
]
