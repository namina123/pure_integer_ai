"""D-02C 原创 typed relation seed 的共享纯合同。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONCEPT,
    OBJECT_ENTITY,
    OBJECT_EVENT,
    OBJECT_OCCURRENCE,
    OBJECT_PROPOSITION,
    OBJECT_SET_EXPR,
    object_contracts_by_kind,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    EXPECTED_STATES,
    CanonicalJsonObject,
)


SOURCE_KEY = "AUTHORED_CC0_V1"
LICENSE_ID = "CC0-1.0"
RELATION_REGISTRY = "AUTHORED_RELATION_V1"
SCHEMA_REGISTRY = "AUTHORED_RELATION_SCHEMA_V1"
ROLE_REGISTRY = "AUTHORED_RELATION_ROLE_V1"

DIRECTION_SYMMETRIC = 1
DIRECTION_FORWARD = 2
REQUEST_REFERENCE_RESOLUTION = 1
REQUEST_RELATION_EVALUATION = 2
REQUEST_PROPERTY_SELECTION = 3
REQUEST_MEREOLOGY_QUERY = 4
REQUEST_SYMMETRIC_PAIR_QUERY = 5
REQUEST_EVENT_TIME_VERIFICATION = 6
REQUEST_CAUSAL_VERIFICATION = 7

RELATION_PURE_ALIAS = 1
RELATION_REFERS = 2
RELATION_SUBSET = 3
RELATION_MEMBER = 4
RELATION_PROPERTY = 5
RELATION_PART_OF = 6
RELATION_HAS_PART = 7
RELATION_SIMILAR = 8
RELATION_ANTONYM = 9
RELATION_EVENT_BEFORE = 10
RELATION_EVENT_AFTER = 11
RELATION_EVENT_SAME = 12
RELATION_EVENT_UNKNOWN = 13
RELATION_CAUSES = 14
SCHEMA_PURE_ALIAS = 1
SCHEMA_REFERS = 2
SCHEMA_SUBSET = 3
SCHEMA_MEMBER = 4
SCHEMA_PROPERTY = 5
SCHEMA_PART_OF = 6
SCHEMA_HAS_PART = 7
SCHEMA_SIMILAR = 8
SCHEMA_ANTONYM = 9
SCHEMA_EVENT_BEFORE = 10
SCHEMA_EVENT_AFTER = 11
SCHEMA_EVENT_SAME = 12
SCHEMA_EVENT_UNKNOWN = 13
SCHEMA_CAUSES = 14
ROLE_ALIAS_LEFT = 1
ROLE_ALIAS_RIGHT = 2
ROLE_REFERS_FROM = 3
ROLE_REFERS_TO = 4
ROLE_SUBSET_CHILD = 5
ROLE_SUBSET_PARENT = 6
ROLE_MEMBER_ELEMENT = 7
ROLE_MEMBER_SET = 8
ROLE_PROPERTY_SUBJECT = 9
ROLE_PROPERTY_ATTRIBUTE = 10
ROLE_PROPERTY_VALUE = 11
ROLE_PROPERTY_POLARITY = 12
ROLE_PROPERTY_MODALITY = 13
ROLE_PROPERTY_INTENSITY = 14
ROLE_PART_OF_PART = 15
ROLE_PART_OF_WHOLE = 16
ROLE_HAS_PART_PART = 17
ROLE_HAS_PART_WHOLE = 18
ROLE_SIMILAR_LEFT = 19
ROLE_SIMILAR_RIGHT = 20
ROLE_ANTONYM_LEFT = 21
ROLE_ANTONYM_RIGHT = 22
ROLE_EVENT_BEFORE_SUBJECT = 23
ROLE_EVENT_BEFORE_OBJECT = 24
ROLE_EVENT_AFTER_SUBJECT = 25
ROLE_EVENT_AFTER_OBJECT = 26
ROLE_EVENT_SAME_SUBJECT = 27
ROLE_EVENT_SAME_OBJECT = 28
ROLE_EVENT_UNKNOWN_SUBJECT = 29
ROLE_EVENT_UNKNOWN_OBJECT = 30
ROLE_CAUSE = 31
ROLE_EFFECT = 32

SUPPORTED_ENDPOINT_KINDS = frozenset({
    OBJECT_CONCEPT,
    OBJECT_ENTITY,
    OBJECT_EVENT,
    OBJECT_OCCURRENCE,
    OBJECT_PROPOSITION,
    OBJECT_SET_EXPR,
})
ALLOWED_PERTURBATIONS = frozenset({
    "NONE",
    "CONTENT_REPLACEMENT",
    "DIRECTION_REVERSAL",
    "PSEUDO_RELATION",
    "CONFLICT_SOURCE",
    "PARSER_REVISION",
    "TYPE_MISMATCH",
    "INVERSE_RELATION",
    "ROLE_MISMATCH",
    "VALUE_REPLACEMENT",
    "INTENSITY_REPLACEMENT",
    "RELATION_CONFUSION",
    "PAIR_REVERSAL",
    "ALIAS_CONFUSION",
    "UNKNOWN_DIRECTION",
    "OCCURRENCE_ORDER_CONFUSION",
    "STRUCTURE_ORDER_CONFUSION",
    "TEMPORAL_ONLY",
    "CORRELATION_CONFUSION",
    "CONFOUNDING_CONFUSION",
    "COUNTERFACTUAL_OVERCLAIM",
})
REQUIRED_SAMPLE_ROLES = frozenset({
    "support", "refute", "conflict", "supersede"})

_SEED_FIELDS = frozenset({
    "anchor",
    "bindings",
    "consumer_request",
    "context_local_id",
    "directionality",
    "endpoints",
    "expected_payload",
    "expected_state",
    "family",
    "label_owner",
    "license_id",
    "logical_order",
    "perturbation_kind",
    "relation_family",
    "relation_kind",
    "relation_registry",
    "sample_role",
    "schema_kind",
    "schema_registry",
    "seed_id",
    "split",
    "supersedes_seed_id",
    "surface",
    "template_family",
})
_ANCHOR_FIELDS = frozenset({
    "end", "ordinal", "start", "surface_fragment"})
_ENDPOINT_FIELDS = frozenset({
    "end",
    "endpoint_id",
    "local_id",
    "object_kind",
    "ordinal",
    "start",
    "surface_fragment",
})
_BINDING_FIELDS = frozenset({
    "endpoint_id", "ordinal", "role_kind", "role_registry"})
_TYPED_BINDING_FIELDS = frozenset({
    *_BINDING_FIELDS, "allowed_object_kinds"})
_REQUEST_FIELDS = frozenset({
    "max_facts",
    "max_routes",
    "max_states",
    "origin_endpoint_id",
    "request_kind",
    "target_object_kinds",
})
_PROPERTY_REQUEST_FIELDS = frozenset({
    "attribute_endpoint_id",
    "max_direct_facts",
    "max_options",
    "request_kind",
    "subject_endpoint_id",
})
_MEREOLOGY_REQUEST_FIELDS = frozenset({
    "max_closure_statements",
    "max_direct_facts",
    "max_options",
    "max_rule_applications",
    "part_endpoint_id",
    "request_kind",
    "whole_endpoint_id",
})
_SYMMETRIC_REQUEST_FIELDS = frozenset({
    "counterpart_endpoint_id",
    "endpoint_id",
    "max_direct_facts",
    "max_options",
    "max_total_direct_facts",
    "request_kind",
})
_EVENT_TIME_REQUEST_FIELDS = frozenset({
    "max_evidence_requests",
    "max_relations",
    "object_endpoint_id",
    "request_kind",
    "subject_endpoint_id",
})
_CAUSAL_REQUEST_FIELDS = frozenset({
    "cause_endpoint_id",
    "effect_endpoint_id",
    "max_evidence_requests",
    "max_relations",
    "max_witness_inputs",
    "request_kind",
})


class AuthoredRelationCourseError(RuntimeError):
    """原创 relation seed 的 typed 结构、owner、split 或修正链非法。"""


def _text(value: Any, *, where: str, allow_empty: bool = False) -> str:
    """要求关系 seed 文本为无首尾空白字符串。"""
    if not isinstance(value, str) or value.strip() != value:
        raise AuthoredRelationCourseError(f"{where} 必须是无首尾空白字符串")
    if not allow_empty and not value:
        raise AuthoredRelationCourseError(f"{where} 不能为空")
    return value


def _positive_int(value: Any, *, where: str) -> int:
    """要求关系、schema、Role 和预算坐标为正严格整数。"""
    if type(value) is not int or value <= 0:
        raise AuthoredRelationCourseError(f"{where} 必须是正严格整数")
    return value


def _nonnegative_int(value: Any, *, where: str) -> int:
    """要求 span 与 ordinal 为非负严格整数。"""
    if type(value) is not int or value < 0:
        raise AuthoredRelationCourseError(f"{where} 必须是非负严格整数")
    return value


@dataclass(frozen=True)
class RelationAnchorSeed:
    """relation Proposition 的来源 occurrence anchor。"""

    surface_fragment: str
    start: int
    end: int
    ordinal: int

    def __post_init__(self) -> None:
        _text(self.surface_fragment, where="RelationAnchorSeed.surface_fragment")
        _nonnegative_int(self.start, where="RelationAnchorSeed.start")
        _nonnegative_int(self.end, where="RelationAnchorSeed.end")
        _nonnegative_int(self.ordinal, where="RelationAnchorSeed.ordinal")
        if self.end <= self.start:
            raise AuthoredRelationCourseError("relation anchor span 必须有正宽度")

    @classmethod
    def from_dict(cls, value: Any) -> "RelationAnchorSeed":
        """从严格字段集合恢复 relation anchor。"""
        if not isinstance(value, dict) or set(value) != _ANCHOR_FIELDS:
            raise AuthoredRelationCourseError("relation anchor 字段集合漂移")
        return cls(
            _text(value["surface_fragment"], where="anchor.surface_fragment"),
            value["start"],
            value["end"],
            value["ordinal"],
        )


@dataclass(frozen=True)
class RelationEndpointSeed:
    """relation Role 可引用的来源 span 和一等对象声明。"""

    endpoint_id: str
    surface_fragment: str
    start: int
    end: int
    ordinal: int
    object_kind: int
    local_id: int | None

    def __post_init__(self) -> None:
        _text(self.endpoint_id, where="RelationEndpointSeed.endpoint_id")
        _text(self.surface_fragment, where="RelationEndpointSeed.surface_fragment")
        _nonnegative_int(self.start, where="RelationEndpointSeed.start")
        _nonnegative_int(self.end, where="RelationEndpointSeed.end")
        _nonnegative_int(self.ordinal, where="RelationEndpointSeed.ordinal")
        if self.end <= self.start:
            raise AuthoredRelationCourseError("relation endpoint span 必须有正宽度")
        if type(self.object_kind) is not int or self.object_kind not in (
                SUPPORTED_ENDPOINT_KINDS):
            raise AuthoredRelationCourseError("relation endpoint object kind 未支持")
        if self.object_kind == OBJECT_OCCURRENCE:
            if self.local_id is not None:
                raise AuthoredRelationCourseError("Occurrence endpoint 不得伪造 local_id")
        else:
            _positive_int(self.local_id, where="RelationEndpointSeed.local_id")

    @classmethod
    def from_dict(cls, value: Any) -> "RelationEndpointSeed":
        """从严格字段集合恢复 relation endpoint。"""
        if not isinstance(value, dict) or set(value) != _ENDPOINT_FIELDS:
            raise AuthoredRelationCourseError("relation endpoint 字段集合漂移")
        return cls(
            _text(value["endpoint_id"], where="endpoint_id"),
            _text(value["surface_fragment"], where="endpoint.surface_fragment"),
            value["start"],
            value["end"],
            value["ordinal"],
            value["object_kind"],
            value["local_id"],
        )


@dataclass(frozen=True)
class RelationBindingSeed:
    """一个课程 Role 坐标到 endpoint 的原子绑定。"""

    role_registry: str
    role_kind: int
    endpoint_id: str
    ordinal: int
    allowed_object_kinds: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.role_registry != ROLE_REGISTRY:
            raise AuthoredRelationCourseError("relation Role registry 非冻结坐标")
        _positive_int(self.role_kind, where="RelationBindingSeed.role_kind")
        _text(self.endpoint_id, where="RelationBindingSeed.endpoint_id")
        _nonnegative_int(self.ordinal, where="RelationBindingSeed.ordinal")
        contracts = object_contracts_by_kind()
        if (not isinstance(self.allowed_object_kinds, tuple)
                or any(type(item) is not int or item not in contracts
                       or not contracts[item].authoritative_identity
                       for item in self.allowed_object_kinds)
                or len(set(self.allowed_object_kinds))
                != len(self.allowed_object_kinds)):
            raise AuthoredRelationCourseError(
                "relation Role allowed_object_kinds 非权威或重复")

    @classmethod
    def from_dict(cls, value: Any) -> "RelationBindingSeed":
        """从严格字段集合恢复 relation RoleBinding。"""
        if (not isinstance(value, dict)
                or set(value) not in {_BINDING_FIELDS, _TYPED_BINDING_FIELDS}):
            raise AuthoredRelationCourseError("relation binding 字段集合漂移")
        allowed = value.get("allowed_object_kinds", [])
        if not isinstance(allowed, list):
            raise AuthoredRelationCourseError(
                "relation binding allowed_object_kinds 必须是列表")
        return cls(
            _text(value["role_registry"], where="role_registry"),
            value["role_kind"],
            _text(value["endpoint_id"], where="binding.endpoint_id"),
            value["ordinal"],
            tuple(allowed),
        )


@dataclass(frozen=True)
class RelationConsumerRequestSeed:
    """ActiveAliasRouteFinder 风格的只读 reference consumer 请求。"""

    request_kind: int
    origin_endpoint_id: str
    target_object_kinds: tuple[int, ...]
    max_facts: int
    max_states: int | None
    max_routes: int | None
    attribute_endpoint_id: str = ""
    max_options: int | None = None
    whole_endpoint_id: str = ""
    max_closure_statements: int | None = None
    max_rule_applications: int | None = None
    counterpart_endpoint_id: str = ""
    max_total_direct_facts: int | None = None
    object_endpoint_id: str = ""
    max_evidence_requests: int | None = None
    max_relations: int | None = None
    effect_endpoint_id: str = ""
    max_witness_inputs: int | None = None

    def __post_init__(self) -> None:
        _positive_int(self.request_kind, where="RelationConsumerRequestSeed.request_kind")
        _text(
            self.origin_endpoint_id,
            where="RelationConsumerRequestSeed.origin_endpoint_id",
        )
        if not isinstance(self.target_object_kinds, tuple):
            raise AuthoredRelationCourseError("target_object_kinds 必须是 tuple")
        contracts = object_contracts_by_kind()
        if (any(type(item) is not int or item not in contracts
                or not contracts[item].authoritative_identity
                for item in self.target_object_kinds)
                or len(set(self.target_object_kinds))
                != len(self.target_object_kinds)):
            raise AuthoredRelationCourseError("target_object_kinds 非权威或重复")
        _positive_int(
            self.max_facts,
            where="RelationConsumerRequestSeed.max_facts",
        )
        if self.request_kind != REQUEST_CAUSAL_VERIFICATION and (
                self.effect_endpoint_id
                or self.max_witness_inputs is not None):
            raise AuthoredRelationCourseError(
                "非 causal consumer 不得携带 causal query 字段")
        if self.request_kind == REQUEST_CAUSAL_VERIFICATION:
            _text(
                self.effect_endpoint_id,
                where="RelationConsumerRequestSeed.effect_endpoint_id",
            )
            if (self.target_object_kinds or self.attribute_endpoint_id
                    or self.whole_endpoint_id or self.counterpart_endpoint_id
                    or self.object_endpoint_id
                    or self.max_states is not None or self.max_routes is not None
                    or self.max_options is not None
                    or self.max_closure_statements is not None
                    or self.max_rule_applications is not None
                    or self.max_total_direct_facts is not None):
                raise AuthoredRelationCourseError(
                    "causal consumer 不得携带其他 query 字段")
            for name, value in (
                    ("max_evidence_requests", self.max_evidence_requests),
                    ("max_relations", self.max_relations),
                    ("max_witness_inputs", self.max_witness_inputs)):
                _positive_int(
                    value,
                    where=f"RelationConsumerRequestSeed.{name}",
                )
            return
        if self.request_kind == REQUEST_MEREOLOGY_QUERY:
            _text(
                self.whole_endpoint_id,
                where="RelationConsumerRequestSeed.whole_endpoint_id",
            )
            if (self.target_object_kinds or self.attribute_endpoint_id
                    or self.max_states is not None
                    or self.max_routes is not None):
                raise AuthoredRelationCourseError(
                    "MEREOLOGY consumer 不得携带其他 query 字段")
            for name, value in (
                    ("max_options", self.max_options),
                    ("max_closure_statements", self.max_closure_statements),
                    ("max_rule_applications", self.max_rule_applications)):
                _positive_int(
                    value,
                    where=f"RelationConsumerRequestSeed.{name}",
                )
            if (self.counterpart_endpoint_id
                    or self.max_total_direct_facts is not None):
                raise AuthoredRelationCourseError(
                    "MEREOLOGY consumer 不得携带 symmetric query 字段")
            if (self.object_endpoint_id
                    or self.max_evidence_requests is not None
                    or self.max_relations is not None):
                raise AuthoredRelationCourseError(
                    "MEREOLOGY consumer 不得携带 event-time query 字段")
            return
        if self.request_kind == REQUEST_SYMMETRIC_PAIR_QUERY:
            _text(
                self.counterpart_endpoint_id,
                where="RelationConsumerRequestSeed.counterpart_endpoint_id",
            )
            if (self.target_object_kinds or self.attribute_endpoint_id
                    or self.whole_endpoint_id or self.max_states is not None
                    or self.max_routes is not None
                    or self.max_closure_statements is not None
                    or self.max_rule_applications is not None):
                raise AuthoredRelationCourseError(
                    "symmetric consumer 不得携带其他 query 字段")
            for name, value in (
                    ("max_options", self.max_options),
                    ("max_total_direct_facts", self.max_total_direct_facts)):
                _positive_int(
                    value,
                    where=f"RelationConsumerRequestSeed.{name}",
                )
            if (self.object_endpoint_id
                    or self.max_evidence_requests is not None
                    or self.max_relations is not None):
                raise AuthoredRelationCourseError(
                    "symmetric consumer 不得携带 event-time query 字段")
            return
        if self.request_kind == REQUEST_EVENT_TIME_VERIFICATION:
            _text(
                self.object_endpoint_id,
                where="RelationConsumerRequestSeed.object_endpoint_id",
            )
            if (self.target_object_kinds or self.attribute_endpoint_id
                    or self.whole_endpoint_id or self.counterpart_endpoint_id
                    or self.max_states is not None or self.max_routes is not None
                    or self.max_options is not None
                    or self.max_closure_statements is not None
                    or self.max_rule_applications is not None
                    or self.max_total_direct_facts is not None):
                raise AuthoredRelationCourseError(
                    "event-time consumer 不得携带其他 query 字段")
            for name, value in (
                    ("max_evidence_requests", self.max_evidence_requests),
                    ("max_relations", self.max_relations)):
                _positive_int(
                    value,
                    where=f"RelationConsumerRequestSeed.{name}",
                )
            return
        if self.request_kind == REQUEST_PROPERTY_SELECTION:
            _text(
                self.attribute_endpoint_id,
                where="RelationConsumerRequestSeed.attribute_endpoint_id",
            )
            if self.target_object_kinds:
                raise AuthoredRelationCourseError(
                    "PROPERTY consumer 不得携带 route target kinds")
            if self.max_states is not None or self.max_routes is not None:
                raise AuthoredRelationCourseError(
                    "PROPERTY consumer 不得携带 route budget")
            _positive_int(
                self.max_options,
                where="RelationConsumerRequestSeed.max_options",
            )
            if (self.whole_endpoint_id
                    or self.max_closure_statements is not None
                    or self.max_rule_applications is not None):
                raise AuthoredRelationCourseError(
                    "PROPERTY consumer 不得携带 mereology query 字段")
            if (self.counterpart_endpoint_id
                    or self.max_total_direct_facts is not None):
                raise AuthoredRelationCourseError(
                    "PROPERTY consumer 不得携带 symmetric query 字段")
            if (self.object_endpoint_id
                    or self.max_evidence_requests is not None
                    or self.max_relations is not None):
                raise AuthoredRelationCourseError(
                    "PROPERTY consumer 不得携带 event-time query 字段")
            return
        if (self.attribute_endpoint_id or self.max_options is not None
                or self.whole_endpoint_id
                or self.max_closure_statements is not None
                or self.max_rule_applications is not None):
            raise AuthoredRelationCourseError(
                "非 PROPERTY consumer 不得携带 property query 字段")
        if (self.counterpart_endpoint_id
                or self.max_total_direct_facts is not None):
            raise AuthoredRelationCourseError(
                "route consumer 不得携带 symmetric query 字段")
        if (self.object_endpoint_id
                or self.max_evidence_requests is not None
                or self.max_relations is not None):
            raise AuthoredRelationCourseError(
                "route consumer 不得携带 event-time query 字段")
        if not self.target_object_kinds:
            raise AuthoredRelationCourseError("target_object_kinds 不能为空")
        for name, value in (
                ("max_states", self.max_states),
                ("max_routes", self.max_routes)):
            _positive_int(value, where=f"RelationConsumerRequestSeed.{name}")

    @classmethod
    def from_dict(cls, value: Any) -> "RelationConsumerRequestSeed":
        """从严格字段集合恢复 consumer 请求。"""
        if not isinstance(value, dict):
            raise AuthoredRelationCourseError("consumer request 字段集合漂移")
        if set(value) == _PROPERTY_REQUEST_FIELDS:
            return cls(
                value["request_kind"],
                _text(
                    value["subject_endpoint_id"],
                    where="subject_endpoint_id",
                ),
                (),
                value["max_direct_facts"],
                None,
                None,
                _text(
                    value["attribute_endpoint_id"],
                    where="attribute_endpoint_id",
                ),
                value["max_options"],
            )
        if set(value) == _MEREOLOGY_REQUEST_FIELDS:
            return cls(
                value["request_kind"],
                _text(
                    value["part_endpoint_id"],
                    where="part_endpoint_id",
                ),
                (),
                value["max_direct_facts"],
                None,
                None,
                "",
                value["max_options"],
                _text(
                    value["whole_endpoint_id"],
                    where="whole_endpoint_id",
                ),
                value["max_closure_statements"],
                value["max_rule_applications"],
            )
        if set(value) == _SYMMETRIC_REQUEST_FIELDS:
            return cls(
                value["request_kind"],
                _text(value["endpoint_id"], where="endpoint_id"),
                (),
                value["max_direct_facts"],
                None,
                None,
                "",
                value["max_options"],
                "",
                None,
                None,
                _text(
                    value["counterpart_endpoint_id"],
                    where="counterpart_endpoint_id",
                ),
                value["max_total_direct_facts"],
            )
        if set(value) == _EVENT_TIME_REQUEST_FIELDS:
            return cls(
                value["request_kind"],
                _text(
                    value["subject_endpoint_id"],
                    where="subject_endpoint_id",
                ),
                (),
                value["max_evidence_requests"],
                None,
                None,
                "",
                None,
                "",
                None,
                None,
                "",
                None,
                _text(
                    value["object_endpoint_id"],
                    where="object_endpoint_id",
                ),
                value["max_evidence_requests"],
                value["max_relations"],
            )
        if set(value) == _CAUSAL_REQUEST_FIELDS:
            return cls(
                request_kind=value["request_kind"],
                origin_endpoint_id=_text(
                    value["cause_endpoint_id"],
                    where="cause_endpoint_id",
                ),
                target_object_kinds=(),
                max_facts=value["max_evidence_requests"],
                max_states=None,
                max_routes=None,
                max_evidence_requests=value["max_evidence_requests"],
                max_relations=value["max_relations"],
                effect_endpoint_id=_text(
                    value["effect_endpoint_id"],
                    where="effect_endpoint_id",
                ),
                max_witness_inputs=value["max_witness_inputs"],
            )
        if set(value) != _REQUEST_FIELDS:
            raise AuthoredRelationCourseError("consumer request 字段集合漂移")
        targets = value["target_object_kinds"]
        if not isinstance(targets, list):
            raise AuthoredRelationCourseError("target_object_kinds 必须是列表")
        return cls(
            value["request_kind"],
            _text(value["origin_endpoint_id"], where="origin_endpoint_id"),
            tuple(targets),
            value["max_facts"],
            value["max_states"],
            value["max_routes"],
        )


@dataclass(frozen=True)
class AuthoredRelationSeed:
    """一条可编译为 RelationSchema 和 AtomicPropositionDefinition 的候选。"""

    seed_id: str
    family: str
    template_family: str
    label_owner: str
    split: str
    sample_role: str
    relation_family: str
    relation_registry: str
    relation_kind: int
    schema_registry: str
    schema_kind: int
    directionality: int
    surface: str
    anchor: RelationAnchorSeed
    endpoints: tuple[RelationEndpointSeed, ...]
    bindings: tuple[RelationBindingSeed, ...]
    context_local_id: int
    consumer_request: RelationConsumerRequestSeed
    expected_state: str
    expected_payload: CanonicalJsonObject
    perturbation_kind: str
    supersedes_seed_id: str
    logical_order: int

    def __post_init__(self) -> None:
        for name, value in (
                ("seed_id", self.seed_id),
                ("family", self.family),
                ("template_family", self.template_family),
                ("relation_family", self.relation_family),
                ("surface", self.surface),
                ("perturbation_kind", self.perturbation_kind)):
            _text(value, where=f"AuthoredRelationSeed.{name}")
        _text(
            self.supersedes_seed_id,
            where="AuthoredRelationSeed.supersedes_seed_id",
            allow_empty=True,
        )
        if self.label_owner not in {"teacher", "evaluator"}:
            raise AuthoredRelationCourseError("label_owner 必须是 teacher/evaluator")
        expected_split = "train" if self.label_owner == "teacher" else "held_out"
        if self.split != expected_split:
            raise AuthoredRelationCourseError("label_owner 与 split 不一致")
        if self.sample_role not in REQUIRED_SAMPLE_ROLES:
            raise AuthoredRelationCourseError("sample_role 不属于 relation 课程")
        if self.sample_role == "supersede" and not self.supersedes_seed_id:
            raise AuthoredRelationCourseError("supersede seed 必须声明替代目标")
        if self.sample_role != "supersede" and self.supersedes_seed_id:
            raise AuthoredRelationCourseError("非 supersede seed 不得声明替代目标")
        if self.relation_registry != RELATION_REGISTRY:
            raise AuthoredRelationCourseError("relation registry 非冻结课程坐标")
        if self.schema_registry != SCHEMA_REGISTRY:
            raise AuthoredRelationCourseError("relation schema registry 非冻结课程坐标")
        _positive_int(self.relation_kind, where="AuthoredRelationSeed.relation_kind")
        _positive_int(self.schema_kind, where="AuthoredRelationSeed.schema_kind")
        if self.directionality not in {DIRECTION_SYMMETRIC, DIRECTION_FORWARD}:
            raise AuthoredRelationCourseError("relation directionality 未注册")
        _positive_int(
            self.context_local_id,
            where="AuthoredRelationSeed.context_local_id",
        )
        if self.expected_state not in EXPECTED_STATES:
            raise AuthoredRelationCourseError("expected_state 非四态")
        if self.perturbation_kind not in ALLOWED_PERTURBATIONS:
            raise AuthoredRelationCourseError("relation perturbation 未注册")
        _positive_int(self.logical_order, where="AuthoredRelationSeed.logical_order")
        if not self.endpoints or not self.bindings:
            raise AuthoredRelationCourseError("relation endpoints/bindings 不能为空")
        if self.anchor.end > len(self.surface) or self.surface[
                self.anchor.start:self.anchor.end] != self.anchor.surface_fragment:
            raise AuthoredRelationCourseError("relation anchor span 与 surface 不一致")

        endpoint_ids = [item.endpoint_id for item in self.endpoints]
        if len(set(endpoint_ids)) != len(endpoint_ids):
            raise AuthoredRelationCourseError("relation endpoint_id 重复")
        endpoint_spans = [
            (item.start, item.end, item.ordinal) for item in self.endpoints]
        if endpoint_spans != sorted(endpoint_spans):
            raise AuthoredRelationCourseError("relation endpoint 必须按来源 span 排序")
        previous_end = -1
        for endpoint in self.endpoints:
            if endpoint.end > len(self.surface) or self.surface[
                    endpoint.start:endpoint.end] != endpoint.surface_fragment:
                raise AuthoredRelationCourseError("relation endpoint span 与 surface 不一致")
            if endpoint.start < previous_end:
                raise AuthoredRelationCourseError("relation endpoint span 不得重叠")
            previous_end = endpoint.end
        endpoint_index = {item.endpoint_id: item for item in self.endpoints}
        slots = [(item.role_kind, item.ordinal) for item in self.bindings]
        if len(set(slots)) != len(slots):
            raise AuthoredRelationCourseError("relation Role/ordinal slot 重复")
        bound_ids = [item.endpoint_id for item in self.bindings]
        if any(item not in endpoint_index for item in bound_ids):
            raise AuthoredRelationCourseError("relation binding 引用未知 endpoint")
        if len(bound_ids) != len(endpoint_ids) or set(bound_ids) != set(endpoint_ids):
            raise AuthoredRelationCourseError("relation bindings 必须恰好覆盖全部 endpoint")
        if self.consumer_request.origin_endpoint_id not in endpoint_index:
            raise AuthoredRelationCourseError("consumer origin 引用未知 endpoint")
        attribute_endpoint_id = self.consumer_request.attribute_endpoint_id
        if (attribute_endpoint_id
                and attribute_endpoint_id not in endpoint_index):
            raise AuthoredRelationCourseError(
                "consumer attribute 引用未知 endpoint")
        if attribute_endpoint_id == self.consumer_request.origin_endpoint_id:
            raise AuthoredRelationCourseError(
                "consumer subject/attribute endpoint 不得相同")
        whole_endpoint_id = self.consumer_request.whole_endpoint_id
        if whole_endpoint_id and whole_endpoint_id not in endpoint_index:
            raise AuthoredRelationCourseError(
                "consumer whole 引用未知 endpoint")
        if whole_endpoint_id == self.consumer_request.origin_endpoint_id:
            raise AuthoredRelationCourseError(
                "consumer part/whole endpoint 不得相同")
        counterpart_endpoint_id = self.consumer_request.counterpart_endpoint_id
        if (counterpart_endpoint_id
                and counterpart_endpoint_id not in endpoint_index):
            raise AuthoredRelationCourseError(
                "consumer counterpart 引用未知 endpoint")
        if counterpart_endpoint_id == self.consumer_request.origin_endpoint_id:
            raise AuthoredRelationCourseError(
                "consumer pair endpoint 不得相同")
        object_endpoint_id = self.consumer_request.object_endpoint_id
        if object_endpoint_id and object_endpoint_id not in endpoint_index:
            raise AuthoredRelationCourseError(
                "consumer event-time object 引用未知 endpoint")
        if object_endpoint_id == self.consumer_request.origin_endpoint_id:
            raise AuthoredRelationCourseError(
                "consumer event-time endpoint 不得相同")
        effect_endpoint_id = self.consumer_request.effect_endpoint_id
        if effect_endpoint_id and effect_endpoint_id not in endpoint_index:
            raise AuthoredRelationCourseError(
                "consumer causal effect 引用未知 endpoint")
        if effect_endpoint_id == self.consumer_request.origin_endpoint_id:
            raise AuthoredRelationCourseError(
                "consumer causal endpoint 不得相同")
        for binding in self.bindings:
            allowed = binding.allowed_object_kinds
            endpoint = endpoint_index[binding.endpoint_id]
            if allowed and endpoint.object_kind not in allowed:
                raise AuthoredRelationCourseError(
                    "relation endpoint 不满足 Role allowed_object_kinds")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AuthoredRelationSeed":
        """从严格字段集合恢复一条共享 relation seed。"""
        if set(value) != _SEED_FIELDS:
            raise AuthoredRelationCourseError("relation seed 字段集合漂移")
        if value["license_id"] != LICENSE_ID:
            raise AuthoredRelationCourseError("relation seed 必须是 CC0-1.0")
        endpoints = value["endpoints"]
        bindings = value["bindings"]
        if not isinstance(endpoints, list) or not isinstance(bindings, list):
            raise AuthoredRelationCourseError("endpoints/bindings 必须是列表")
        return cls(
            _text(value["seed_id"], where="seed_id"),
            _text(value["family"], where="family"),
            _text(value["template_family"], where="template_family"),
            _text(value["label_owner"], where="label_owner"),
            _text(value["split"], where="split"),
            _text(value["sample_role"], where="sample_role"),
            _text(value["relation_family"], where="relation_family"),
            _text(value["relation_registry"], where="relation_registry"),
            value["relation_kind"],
            _text(value["schema_registry"], where="schema_registry"),
            value["schema_kind"],
            value["directionality"],
            _text(value["surface"], where="surface"),
            RelationAnchorSeed.from_dict(value["anchor"]),
            tuple(RelationEndpointSeed.from_dict(item) for item in endpoints),
            tuple(RelationBindingSeed.from_dict(item) for item in bindings),
            value["context_local_id"],
            RelationConsumerRequestSeed.from_dict(value["consumer_request"]),
            _text(value["expected_state"], where="expected_state"),
            CanonicalJsonObject.from_value(value["expected_payload"]),
            _text(value["perturbation_kind"], where="perturbation_kind"),
            _text(
                value["supersedes_seed_id"],
                where="supersedes_seed_id",
                allow_empty=True,
            ),
            value["logical_order"],
        )


__all__ = [
    "ALLOWED_PERTURBATIONS",
    "AuthoredRelationCourseError",
    "AuthoredRelationSeed",
    "DIRECTION_FORWARD",
    "DIRECTION_SYMMETRIC",
    "LICENSE_ID",
    "RELATION_PURE_ALIAS",
    "RELATION_REFERS",
    "RELATION_SUBSET",
    "RELATION_MEMBER",
    "RELATION_PROPERTY",
    "RELATION_PART_OF",
    "RELATION_HAS_PART",
    "RELATION_SIMILAR",
    "RELATION_ANTONYM",
    "RELATION_EVENT_BEFORE",
    "RELATION_EVENT_AFTER",
    "RELATION_EVENT_SAME",
    "RELATION_EVENT_UNKNOWN",
    "RELATION_CAUSES",
    "RELATION_REGISTRY",
    "REQUEST_REFERENCE_RESOLUTION",
    "REQUEST_RELATION_EVALUATION",
    "REQUEST_PROPERTY_SELECTION",
    "REQUEST_MEREOLOGY_QUERY",
    "REQUEST_SYMMETRIC_PAIR_QUERY",
    "REQUEST_EVENT_TIME_VERIFICATION",
    "REQUEST_CAUSAL_VERIFICATION",
    "REQUIRED_SAMPLE_ROLES",
    "ROLE_ALIAS_LEFT",
    "ROLE_ALIAS_RIGHT",
    "ROLE_REFERS_FROM",
    "ROLE_REFERS_TO",
    "ROLE_SUBSET_CHILD",
    "ROLE_SUBSET_PARENT",
    "ROLE_MEMBER_ELEMENT",
    "ROLE_MEMBER_SET",
    "ROLE_PROPERTY_SUBJECT",
    "ROLE_PROPERTY_ATTRIBUTE",
    "ROLE_PROPERTY_VALUE",
    "ROLE_PROPERTY_POLARITY",
    "ROLE_PROPERTY_MODALITY",
    "ROLE_PROPERTY_INTENSITY",
    "ROLE_PART_OF_PART",
    "ROLE_PART_OF_WHOLE",
    "ROLE_HAS_PART_PART",
    "ROLE_HAS_PART_WHOLE",
    "ROLE_SIMILAR_LEFT",
    "ROLE_SIMILAR_RIGHT",
    "ROLE_ANTONYM_LEFT",
    "ROLE_ANTONYM_RIGHT",
    "ROLE_EVENT_BEFORE_SUBJECT",
    "ROLE_EVENT_BEFORE_OBJECT",
    "ROLE_EVENT_AFTER_SUBJECT",
    "ROLE_EVENT_AFTER_OBJECT",
    "ROLE_EVENT_SAME_SUBJECT",
    "ROLE_EVENT_SAME_OBJECT",
    "ROLE_EVENT_UNKNOWN_SUBJECT",
    "ROLE_EVENT_UNKNOWN_OBJECT",
    "ROLE_CAUSE",
    "ROLE_EFFECT",
    "ROLE_REGISTRY",
    "SCHEMA_PURE_ALIAS",
    "SCHEMA_REFERS",
    "SCHEMA_SUBSET",
    "SCHEMA_MEMBER",
    "SCHEMA_PROPERTY",
    "SCHEMA_PART_OF",
    "SCHEMA_HAS_PART",
    "SCHEMA_SIMILAR",
    "SCHEMA_ANTONYM",
    "SCHEMA_EVENT_BEFORE",
    "SCHEMA_EVENT_AFTER",
    "SCHEMA_EVENT_SAME",
    "SCHEMA_EVENT_UNKNOWN",
    "SCHEMA_CAUSES",
    "SCHEMA_REGISTRY",
    "SOURCE_KEY",
    "SUPPORTED_ENDPOINT_KINDS",
    "RelationAnchorSeed",
    "RelationBindingSeed",
    "RelationConsumerRequestSeed",
    "RelationEndpointSeed",
]
