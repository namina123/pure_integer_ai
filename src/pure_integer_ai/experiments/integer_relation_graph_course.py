"""纯整数 relation graph-course 的编译、恢复和 W-06 接线。

课程交换文件只保存完整的一等身份、RoleBinding、schema、Span 区间和
Unicode codepoint。运行时不会回读 authored JSONL、正文或答案；恢复后仅把
这些整数记录适配到既有 W-06 typed owner，随后由正式 Core/Memory/生成链消费。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONCEPT,
    OBJECT_CONTEXT_SCOPE,
    OBJECT_OCCURRENCE,
    OBJECT_PROPOSITION,
    OBJECT_ROLE,
    OBJECT_ROLE_BINDING,
    OBJECT_STRUCTURE_CONCEPT,
    ObjectIdentity,
    SourceRef,
)
from pure_integer_ai.cognition.shared.semantic_object import (
    AtomicRoleBinding,
    semantic_source,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    ObservationRecord,
    parse_canonical_json_bytes,
    SourceRefRecord,
    StableRecordKey,
    TeacherEvidenceRecord,
)
from pure_integer_ai.experiments.ph2_w06_adapter import (
    W06TypedAdapterOutput,
    adapt_w06_training_payload,
)
from pure_integer_ai.experiments.ph2_w06_payload import W06TrainingPayload
from pure_integer_ai.experiments.ph2_w06_source_semantic import (
    W06_RELATION_PROFILES,
)
from pure_integer_ai.experiments.ph2_w06_registry import W06_RELATION_REGISTRY


INTEGER_RELATION_GRAPH_COURSE_FORMAT = (
    "PURE_INTEGER_RELATION_GRAPH_COURSE_V1")
LICENSE_ID = "CC0-1.0"
_SAMPLE_ROLE_NAMES = {1: "support", 2: "refute", 3: "supersede", 4: "conflict"}
_SAMPLE_ROLE_STATES = {1: "TRUE", 2: "FALSE", 3: "TRUE", 4: "CONFLICT"}
_STATE_CODES = {1: "TRUE", 2: "FALSE", 3: "UNKNOWN", 4: "CONFLICT"}
_STATE_BY_NAME = {value: key for key, value in _STATE_CODES.items()}
_TEXT_TAG = "__text_values__"
_PERTURBATIONS = (
    "NONE", "CONTENT_REPLACEMENT", "DIRECTION_REVERSAL",
    "PSEUDO_RELATION", "CONFLICT_SOURCE", "PARSER_REVISION",
    "TYPE_MISMATCH", "INVERSE_RELATION", "ROLE_MISMATCH",
    "VALUE_REPLACEMENT", "INTENSITY_REPLACEMENT", "RELATION_CONFUSION",
    "PAIR_REVERSAL", "ALIAS_CONFUSION", "UNKNOWN_DIRECTION",
    "OCCURRENCE_ORDER_CONFUSION", "STRUCTURE_ORDER_CONFUSION",
    "TEMPORAL_ONLY", "CORRELATION_CONFUSION", "CONFOUNDING_CONFUSION",
    "COUNTERFACTUAL_OVERCLAIM",
)
_PROFILE_BY_KIND = {
    profile.relation_kind: (family, profile)
    for family, profile in W06_RELATION_PROFILES.items()
}
_TEACHER_KIND_BY_SUBSTAGE = {
    entry.substage_key: entry.teacher_evidence_kinds[0]
    for entry in W06_RELATION_REGISTRY.values()
}
_ROLE_BINDING_FIELDS = frozenset(
    {"binding_key", "filler_key", "ordinal", "role_key"})
_SLOT_FIELDS = frozenset(
    {"allowed_object_kinds", "max_count", "min_count", "role_key"})
_CONSTRAINT_FIELDS = frozenset({"constraint_key", "role_keys"})
_ENDPOINT_FIELDS = frozenset(
    {"endpoint_key", "object_kind", "ordinal", "start", "end"})
_RECORD_FIELDS = frozenset({
    "source_ref_key", "dataset_key", "artifact_key", "observation_key",
    "semantic_key", "proposition_key", "predicate_key", "context_key",
    "source_anchor_key", "schema_key", "relation_key", "role_bindings",
    "slots", "relation_constraints", "endpoints", "consumer_request",
    "optional_payload", "surface_values", "anchor_span", "relation_kind",
    "directionality", "sample_role_code", "perturbation_code", "logical_order",
    "source_cluster_key", "dedup_cluster_key", "content_group_key",
    "template_group_key", "shape_group_key", "teacher_expected_state_code",
    "teacher_decision_values", "teacher_expected_payload",
    "teacher_seed_values", "supersedes_key", "prerequisite_keys",
})
_COURSE_FIELDS = frozenset({
    "format", "source_identity", "license_identity", "record_count",
    "records", "free_dialogue_complete",
})
_OPTIONAL_PAYLOAD_FIELDS = frozenset({
    "causal_protocol", "confidence_units", "event_time_protocol",
    "mereology_protocol", "polarity_key", "rational_role_values",
    "semantic_pair_protocol", "uncertainty_units",
})


class IntegerRelationGraphCourseError(ValueError):
    """整数 graph-course 记录缺字段、身份漂移或 Span 不闭合。"""


def _key(value, *, where: str, positive: bool = False) -> tuple[int, ...]:
    """恢复非空整数键；stable record key 另要求全部为正整数。"""
    if not isinstance(value, (list, tuple)) or not value:
        raise IntegerRelationGraphCourseError(f"{where} 必须是非空整数列表")
    if any(type(item) is not int or item < 0
           or (positive and item <= 0) for item in value):
        raise IntegerRelationGraphCourseError(f"{where} 必须是严格整数键")
    return tuple(value)


def _optional_key(value: Any, *, where: str) -> tuple[int, ...] | None:
    """恢复可选 StableRecordKey；缺失只表示没有 supersedes 边。"""
    if value is None:
        return None
    return _key(value, where=where, positive=True)


def _identity(value, *, where: str) -> ObjectIdentity:
    """从完整 stable key 恢复一等对象，不按局部 id 猜测。"""
    try:
        return ObjectIdentity.from_stable_key(_key(value, where=where))
    except (TypeError, ValueError) as error:
        raise IntegerRelationGraphCourseError(f"{where} 身份非法") from error


def _occurrence_source(identity: ObjectIdentity, *, where: str) -> SourceRef:
    """从完整 Occurrence 身份恢复其来源，不把 occurrence 当 semantic object。"""
    if identity.object_kind != OBJECT_OCCURRENCE or len(identity.components) < 4:
        raise IntegerRelationGraphCourseError(f"{where} occurrence 身份非法")
    try:
        return SourceRef.from_stable_key(identity.components[:-3])
    except (TypeError, ValueError) as error:
        raise IntegerRelationGraphCourseError(f"{where} occurrence 来源非法") from error


def _surface(values: tuple[int, ...]) -> str:
    """只把已核验 codepoint 转成训练期 Span 表面，不建立词表。"""
    if (not values or any(type(item) is not int or item < 0 or item > 0x10FFFF
                          for item in values)):
        raise IntegerRelationGraphCourseError("surface_values codepoint 非法")
    try:
        return "".join(chr(item) for item in values)
    except ValueError as error:
        raise IntegerRelationGraphCourseError("surface_values 无法编码") from error


def _integer_tree(value: Any, *, where: str) -> None:
    """拒绝 graph-course 嵌套载荷中的文本、浮点和布尔语义值。"""
    if value is None or type(value) is int:
        return
    if isinstance(value, (str, bool, float)):
        raise IntegerRelationGraphCourseError(
            f"{where} 只能包含纯整数/空值，不得包含 {type(value).__name__}")
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _integer_tree(item, where=f"{where}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise IntegerRelationGraphCourseError(
                    f"{where} object key 非法")
            _integer_tree(item, where=f"{where}.{key}")
        return
    raise IntegerRelationGraphCourseError(
        f"{where} 含不支持类型 {type(value).__name__}")


def _encode_integer_payload(value: Any, *, where: str) -> Any:
    """把训练期标注中的短文本转为 codepoint tag，保留无文本图载荷。"""
    if value is None or type(value) is int:
        return value
    if isinstance(value, str):
        return {_TEXT_TAG: list(map(ord, value))}
    if isinstance(value, (list, tuple)):
        return [_encode_integer_payload(item, where=f"{where}[{index}]")
                for index, item in enumerate(value)]
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise IntegerRelationGraphCourseError(f"{where} key 非法")
            result[key] = _encode_integer_payload(item, where=f"{where}.{key}")
        return result
    raise IntegerRelationGraphCourseError(
        f"{where} 含不支持类型 {type(value).__name__}")


def _decode_integer_payload(value: Any, *, where: str) -> Any:
    """恢复训练期 teacher payload 的 codepoint tag，不接受隐式字符串。"""
    if value is None or type(value) is int:
        return value
    if isinstance(value, list):
        return [_decode_integer_payload(item, where=f"{where}[{index}]")
                for index, item in enumerate(value)]
    if isinstance(value, dict):
        if set(value) == {_TEXT_TAG}:
            values = value[_TEXT_TAG]
            if (not isinstance(values, list)
                    or any(type(item) is not int or not 0 <= item <= 0x10FFFF
                           for item in values)):
                raise IntegerRelationGraphCourseError(f"{where} text tag 非法")
            return "".join(chr(item) for item in values)
        return {
            key: _decode_integer_payload(item, where=f"{where}.{key}")
            for key, item in value.items()
        }
    raise IntegerRelationGraphCourseError(
        f"{where} 含不支持类型 {type(value).__name__}")


def _graph_object(
        value: Any, *, fields: frozenset[str] | None, where: str,
        ) -> CanonicalJsonObject:
    """把一个整数 object 规范化为不可变 CanonicalJsonObject。"""
    raw = value.to_value() if isinstance(value, CanonicalJsonObject) else value
    if not isinstance(raw, dict):
        raise IntegerRelationGraphCourseError(f"{where} 必须是 object")
    if fields is not None and set(raw) != fields:
        raise IntegerRelationGraphCourseError(f"{where} 字段集合漂移")
    _integer_tree(raw, where=where)
    try:
        return CanonicalJsonObject.from_value(raw)
    except ValueError as error:
        raise IntegerRelationGraphCourseError(f"{where} 不是规范 object") from error


def _graph_objects(
        values: Any, *, fields: frozenset[str], where: str,
        ) -> tuple[CanonicalJsonObject, ...]:
    """规范化 object sequence，并拒绝字符串/正文偷渡。"""
    if not isinstance(values, (list, tuple)):
        raise IntegerRelationGraphCourseError(f"{where} 必须是 object 列表")
    return tuple(
        _graph_object(item, fields=fields, where=f"{where}[{index}]")
        for index, item in enumerate(values)
    )


def _object_value(value: CanonicalJsonObject) -> dict[str, Any]:
    """恢复一个独立的可变副本供 W-06 payload 构造使用。"""
    return value.to_value()


@dataclass(frozen=True, slots=True)
class IntegerRelationGraphRecord:
    """可直接进入 W-06/Core 的单条整数 relation 记录。"""

    source_ref_key: tuple[int, ...]
    dataset_key: tuple[int, ...]
    artifact_key: tuple[int, ...]
    observation_key: tuple[int, ...]
    semantic_key: tuple[int, ...]
    proposition_key: tuple[int, ...]
    predicate_key: tuple[int, ...]
    context_key: tuple[int, ...]
    source_anchor_key: tuple[int, ...]
    schema_key: tuple[int, ...]
    relation_key: tuple[int, ...]
    role_bindings: tuple[CanonicalJsonObject, ...]
    slots: tuple[CanonicalJsonObject, ...]
    endpoints: tuple[CanonicalJsonObject, ...]
    surface_values: tuple[int, ...]
    anchor_span: tuple[int, int, int]
    relation_kind: int
    directionality: int
    sample_role_code: int
    perturbation_code: int
    logical_order: int
    source_cluster_key: tuple[int, ...]
    dedup_cluster_key: tuple[int, ...]
    content_group_key: tuple[int, ...]
    template_group_key: tuple[int, ...]
    shape_group_key: tuple[int, ...]
    relation_constraints: tuple[CanonicalJsonObject, ...] = ()
    consumer_request: CanonicalJsonObject = field(
        default_factory=lambda: CanonicalJsonObject.from_value({}))
    optional_payload: CanonicalJsonObject = field(
        default_factory=lambda: CanonicalJsonObject.from_value({}))
    teacher_expected_state_code: int = 0
    teacher_decision_values: tuple[int, ...] = ()
    teacher_expected_payload: CanonicalJsonObject = field(
        default_factory=lambda: CanonicalJsonObject.from_value({}))
    teacher_seed_values: tuple[int, ...] = ()
    supersedes_key: tuple[int, ...] | None = None
    prerequisite_keys: tuple[tuple[int, ...], ...] = ()

    def __post_init__(self) -> None:
        for name in (
                "source_ref_key", "dataset_key", "artifact_key",
                "observation_key", "source_cluster_key", "dedup_cluster_key",
                "content_group_key", "template_group_key", "shape_group_key"):
            object.__setattr__(
                self, name, _key(getattr(self, name), where=name, positive=True))
        for name in (
                "semantic_key", "proposition_key", "predicate_key", "context_key",
                "source_anchor_key", "schema_key", "relation_key"):
            object.__setattr__(self, name, _key(getattr(self, name), where=name))
        object.__setattr__(self, "surface_values", _key(
            self.surface_values, where="surface_values"))
        object.__setattr__(self, "anchor_span", _key(
            self.anchor_span, where="anchor_span"))
        object.__setattr__(
            self, "role_bindings", _graph_objects(
                self.role_bindings, fields=_ROLE_BINDING_FIELDS,
                where="role_bindings"))
        object.__setattr__(
            self, "slots", _graph_objects(
                self.slots, fields=_SLOT_FIELDS, where="slots"))
        object.__setattr__(
            self, "relation_constraints", _graph_objects(
                self.relation_constraints, fields=_CONSTRAINT_FIELDS,
                where="relation_constraints"))
        object.__setattr__(
            self, "endpoints", _graph_objects(
                self.endpoints, fields=_ENDPOINT_FIELDS, where="endpoints"))
        object.__setattr__(self, "consumer_request", _graph_object(
            self.consumer_request, fields=None, where="consumer_request"))
        object.__setattr__(self, "optional_payload", _graph_object(
            self.optional_payload, fields=None, where="optional_payload"))
        proposition = _identity(self.proposition_key, where="proposition_key")
        predicate = _identity(self.predicate_key, where="predicate_key")
        context = _identity(self.context_key, where="context_key")
        anchor = _identity(self.source_anchor_key, where="source_anchor_key")
        schema = _identity(self.schema_key, where="schema_key")
        relation = _identity(self.relation_key, where="relation_key")
        if proposition.object_kind != OBJECT_PROPOSITION:
            raise IntegerRelationGraphCourseError("proposition_key 类型错误")
        if predicate.object_kind != OBJECT_CONCEPT:
            raise IntegerRelationGraphCourseError("predicate_key 类型错误")
        if context.object_kind != OBJECT_CONTEXT_SCOPE:
            raise IntegerRelationGraphCourseError("context_key 类型错误")
        if anchor.object_kind != OBJECT_OCCURRENCE:
            raise IntegerRelationGraphCourseError("source_anchor_key 类型错误")
        if schema.object_kind != OBJECT_STRUCTURE_CONCEPT:
            raise IntegerRelationGraphCourseError("schema_key 类型错误")
        if relation.object_kind != OBJECT_CONCEPT or relation != predicate:
            raise IntegerRelationGraphCourseError("relation/predicate 类型或身份漂移")
        try:
            if (semantic_source(proposition) != semantic_source(context)
                    or semantic_source(proposition) != _occurrence_source(
                        anchor, where="source_anchor_key")):
                raise IntegerRelationGraphCourseError("Proposition/Context/anchor 来源漂移")
        except (TypeError, ValueError) as error:
            raise IntegerRelationGraphCourseError("语义来源身份非法") from error
        profile_pair = _PROFILE_BY_KIND.get(self.relation_kind)
        if profile_pair is None or self.directionality != profile_pair[1].directionality:
            raise IntegerRelationGraphCourseError("relation registry/directionality 非法")
        if (type(self.sample_role_code) is not int
                or self.sample_role_code not in _SAMPLE_ROLE_NAMES):
            raise IntegerRelationGraphCourseError("sample role code 非法")
        if (type(self.perturbation_code) is not int
                or not 0 <= self.perturbation_code < len(_PERTURBATIONS)):
            raise IntegerRelationGraphCourseError("perturbation code 非法")
        if type(self.teacher_expected_state_code) is not int:
            raise IntegerRelationGraphCourseError("teacher state code 非法")
        state_code = self.teacher_expected_state_code
        if state_code == 0:
            state_code = _STATE_BY_NAME[_SAMPLE_ROLE_STATES[self.sample_role_code]]
        if state_code not in _STATE_CODES:
            raise IntegerRelationGraphCourseError("teacher state code 未注册")
        object.__setattr__(self, "teacher_expected_state_code", state_code)
        decision_values = tuple(self.teacher_decision_values)
        if not decision_values:
            fallback = (
                "reject_type_mismatch"
                if self.perturbation_code == _PERTURBATIONS.index("TYPE_MISMATCH")
                else _SAMPLE_ROLE_NAMES[self.sample_role_code]
            )
            decision_values = tuple(map(ord, fallback))
        object.__setattr__(self, "teacher_decision_values", _key(
            decision_values, where="teacher_decision_values"))
        _surface(self.teacher_decision_values)
        expected_payload = _graph_object(
            self.teacher_expected_payload, fields=None,
            where="teacher_expected_payload")
        object.__setattr__(self, "teacher_expected_payload", expected_payload)
        seed_values = tuple(self.teacher_seed_values)
        if not seed_values:
            seed_values = tuple(map(ord, "integer-graph-course-teacher"))
        object.__setattr__(self, "teacher_seed_values", _key(
            seed_values, where="teacher_seed_values"))
        _surface(self.teacher_seed_values)
        object.__setattr__(self, "supersedes_key", _optional_key(
            self.supersedes_key, where="supersedes_key"))
        prerequisites = tuple(
            _key(item, where="prerequisite_key", positive=True)
            for item in self.prerequisite_keys)
        if len(set(prerequisites)) != len(prerequisites):
            raise IntegerRelationGraphCourseError("prerequisite_keys 不得重复")
        object.__setattr__(self, "prerequisite_keys", prerequisites)
        if self.supersedes_key == self.observation_key:
            raise IntegerRelationGraphCourseError("supersedes_key 不得指向自身")
        if self.observation_key in prerequisites:
            raise IntegerRelationGraphCourseError(
                "prerequisite_keys 不得指向自身")
        if type(self.logical_order) is not int or self.logical_order <= 0:
            raise IntegerRelationGraphCourseError("logical_order 非法")
        if type(self.relation_kind) is not int or self.relation_kind <= 0:
            raise IntegerRelationGraphCourseError("relation_kind 非法")
        if type(self.directionality) is not int:
            raise IntegerRelationGraphCourseError("directionality 非法")
        values = self.surface_values
        text = _surface(values)
        if (type(self.anchor_span) is not tuple or len(self.anchor_span) != 3
                or any(type(item) is not int for item in self.anchor_span)
                or self.anchor_span[0] < 0
                or self.anchor_span[1] <= self.anchor_span[0]
                or self.anchor_span[1] > len(text)
                or self.anchor_span[2] < 0):
            raise IntegerRelationGraphCourseError("anchor Span 非法")
        if not self.role_bindings or not self.slots or not self.endpoints:
            raise IntegerRelationGraphCourseError("relation graph record 结构为空")
        if len(self.role_bindings) != len(self.endpoints):
            raise IntegerRelationGraphCourseError("RoleBinding/endpoint 数量不一致")
        endpoint_keys = set()
        slot_roles = set()
        binding_roles = set()
        binding_keys = set()
        for item in self.endpoints:
            value = _object_value(item)
            endpoint = _identity(value["endpoint_key"], where="endpoint_key")
            if (type(value["object_kind"]) is not int
                    or endpoint.object_kind != value["object_kind"]):
                raise IntegerRelationGraphCourseError("endpoint object kind 漂移")
            if (type(value["start"]) is not int
                    or type(value["end"]) is not int
                    or value["start"] < 0 or value["end"] <= value["start"]
                    or value["end"] > len(text)
                    or type(value["ordinal"]) is not int
                    or value["ordinal"] < 0):
                raise IntegerRelationGraphCourseError("endpoint Span 非法")
            if endpoint.stable_key() in endpoint_keys:
                raise IntegerRelationGraphCourseError("endpoint identity 重复")
            endpoint_keys.add(endpoint.stable_key())
        for item in self.slots:
            value = _object_value(item)
            role = _identity(value["role_key"], where="slot.role_key")
            if role.object_kind != OBJECT_ROLE:
                raise IntegerRelationGraphCourseError("schema slot 不是 Role")
            allowed = value["allowed_object_kinds"]
            if (not isinstance(allowed, list) or not allowed
                    or any(type(kind) is not int or kind <= 0 for kind in allowed)):
                raise IntegerRelationGraphCourseError("schema slot allowed kinds 非法")
            if type(value["min_count"]) is not int or value["min_count"] < 0:
                raise IntegerRelationGraphCourseError("schema slot min_count 非法")
            maximum = value["max_count"]
            if (maximum is not None
                    and (type(maximum) is not int or maximum < value["min_count"])):
                raise IntegerRelationGraphCourseError("schema slot max_count 非法")
            if role.stable_key() in slot_roles:
                raise IntegerRelationGraphCourseError("schema slot Role 重复")
            slot_roles.add(role.stable_key())
        for item in self.relation_constraints:
            value = _object_value(item)
            constraint = _identity(
                value["constraint_key"], where="constraint.constraint_key")
            if constraint.object_kind != OBJECT_STRUCTURE_CONCEPT:
                raise IntegerRelationGraphCourseError("constraint identity 类型错误")
            roles = value["role_keys"]
            if (not isinstance(roles, list) or len(roles) < 2
                    or any(_identity(role, where="constraint.role_key").object_kind
                           != OBJECT_ROLE for role in roles)
                    or len({tuple(role) for role in roles}) != len(roles)):
                raise IntegerRelationGraphCourseError("constraint Role 集合非法")
        for item in self.role_bindings:
            value = _object_value(item)
            role = _identity(value["role_key"], where="binding.role_key")
            filler = _identity(value["filler_key"], where="binding.filler_key")
            binding = _identity(value["binding_key"], where="binding_key")
            if role.object_kind != OBJECT_ROLE or binding.object_kind != OBJECT_ROLE_BINDING:
                raise IntegerRelationGraphCourseError("RoleBinding identity 类型错误")
            if filler.stable_key() not in endpoint_keys:
                raise IntegerRelationGraphCourseError("RoleBinding filler 缺少 endpoint")
            if type(value["ordinal"]) is not int or value["ordinal"] < 0:
                raise IntegerRelationGraphCourseError("RoleBinding ordinal 非法")
            expected = AtomicRoleBinding(role, filler, value["ordinal"]).identity_for(
                proposition)
            if expected.stable_key() != binding.stable_key():
                raise IntegerRelationGraphCourseError("RoleBinding identity 漂移")
            if role.stable_key() in binding_roles or binding.stable_key() in binding_keys:
                raise IntegerRelationGraphCourseError("RoleBinding 重复")
            binding_roles.add(role.stable_key())
            binding_keys.add(binding.stable_key())
        if binding_roles != slot_roles:
            raise IntegerRelationGraphCourseError("RoleBinding/schema Role 集合漂移")

    def stable_key(self) -> tuple[int, ...]:
        """返回 record 的规范整数身份。"""
        return (
            len(self.observation_key), *self.observation_key,
            len(self.proposition_key), *self.proposition_key,
        )


@dataclass(frozen=True, slots=True)
class IntegerRelationGraphCourse:
    """一个来源/许可绑定的整数 graph-course。"""

    source_identity: tuple[int, ...]
    license_identity: tuple[int, ...]
    records: tuple[IntegerRelationGraphRecord, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_identity", _key(
            self.source_identity, where="course.source_identity", positive=True))
        object.__setattr__(self, "license_identity", _key(
            self.license_identity, where="course.license_identity", positive=True))
        if not isinstance(self.records, (tuple, list)) or not self.records:
            raise IntegerRelationGraphCourseError("graph-course 不能为空")
        if any(not isinstance(item, IntegerRelationGraphRecord)
               for item in self.records):
            raise IntegerRelationGraphCourseError("graph-course record 类型错误")
        records = tuple(self.records)
        object.__setattr__(self, "records", records)
        ordered = tuple(sorted(records, key=lambda item: item.stable_key()))
        if ordered != records or len({item.stable_key() for item in records}) != len(records):
            raise IntegerRelationGraphCourseError("graph-course records 必须规范排序去重")


def _record_dict(record: IntegerRelationGraphRecord) -> dict[str, object]:
    """把整数 record 写成无正文、无答案表的规范 object。"""
    return {
        "source_ref_key": list(record.source_ref_key),
        "dataset_key": list(record.dataset_key),
        "artifact_key": list(record.artifact_key),
        "observation_key": list(record.observation_key),
        "semantic_key": list(record.semantic_key),
        "proposition_key": list(record.proposition_key),
        "predicate_key": list(record.predicate_key),
        "context_key": list(record.context_key),
        "source_anchor_key": list(record.source_anchor_key),
        "schema_key": list(record.schema_key),
        "relation_key": list(record.relation_key),
        "role_bindings": [_object_value(item) for item in record.role_bindings],
        "slots": [_object_value(item) for item in record.slots],
        "relation_constraints": [
            _object_value(item) for item in record.relation_constraints],
        "endpoints": [_object_value(item) for item in record.endpoints],
        "consumer_request": _object_value(record.consumer_request),
        "optional_payload": _object_value(record.optional_payload),
        "surface_values": list(record.surface_values),
        "anchor_span": list(record.anchor_span),
        "relation_kind": record.relation_kind,
        "directionality": record.directionality,
        "sample_role_code": record.sample_role_code,
        "perturbation_code": record.perturbation_code,
        "logical_order": record.logical_order,
        "source_cluster_key": list(record.source_cluster_key),
        "dedup_cluster_key": list(record.dedup_cluster_key),
        "content_group_key": list(record.content_group_key),
        "template_group_key": list(record.template_group_key),
        "shape_group_key": list(record.shape_group_key),
        "teacher_expected_state_code": record.teacher_expected_state_code,
        "teacher_decision_values": list(record.teacher_decision_values),
        "teacher_expected_payload": _object_value(record.teacher_expected_payload),
        "teacher_seed_values": list(record.teacher_seed_values),
        "supersedes_key": (
            None if record.supersedes_key is None
            else list(record.supersedes_key)),
        "prerequisite_keys": [list(item) for item in record.prerequisite_keys],
    }


def _record_from_dict(value: object) -> IntegerRelationGraphRecord:
    """严格恢复一个整数 graph-course record。"""
    if not isinstance(value, dict):
        raise IntegerRelationGraphCourseError("record 根必须是 object")
    if set(value) != _RECORD_FIELDS:
        raise IntegerRelationGraphCourseError("record 字段集合漂移")
    try:
        return IntegerRelationGraphRecord(
            _key(value["source_ref_key"], where="source_ref_key", positive=True),
            _key(value["dataset_key"], where="dataset_key", positive=True),
            _key(value["artifact_key"], where="artifact_key", positive=True),
            _key(value["observation_key"], where="observation_key", positive=True),
            _key(value["semantic_key"], where="semantic_key"),
            _key(value["proposition_key"], where="proposition_key"),
            _key(value["predicate_key"], where="predicate_key"),
            _key(value["context_key"], where="context_key"),
            _key(value["source_anchor_key"], where="source_anchor_key"),
            _key(value["schema_key"], where="schema_key"),
            _key(value["relation_key"], where="relation_key"),
            tuple(value["role_bindings"]), tuple(value["slots"]),
            tuple(value["endpoints"]), tuple(value["surface_values"]),
            tuple(value["anchor_span"]), value["relation_kind"],
            value["directionality"], value["sample_role_code"],
            value["perturbation_code"], value["logical_order"],
            _key(value["source_cluster_key"], where="source_cluster_key", positive=True),
            _key(value["dedup_cluster_key"], where="dedup_cluster_key", positive=True),
            _key(value["content_group_key"], where="content_group_key", positive=True),
            _key(value["template_group_key"], where="template_group_key", positive=True),
            _key(value["shape_group_key"], where="shape_group_key", positive=True),
            tuple(value["relation_constraints"]),
            value["consumer_request"], value["optional_payload"],
            value["teacher_expected_state_code"],
            tuple(value["teacher_decision_values"]),
            value["teacher_expected_payload"],
            tuple(value["teacher_seed_values"]),
            _optional_key(value["supersedes_key"], where="supersedes_key"),
            tuple(tuple(item) for item in value["prerequisite_keys"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, IntegerRelationGraphCourseError):
            raise
        raise IntegerRelationGraphCourseError("graph-course record 字段非法") from error


def write_integer_relation_graph_course(
        path: str | Path,
        course: IntegerRelationGraphCourse,
        ) -> str:
    """写入整数 graph-course，并返回文件 SHA-256。"""
    if not isinstance(course, IntegerRelationGraphCourse):
        raise TypeError("course 类型错误")
    target = Path(path).resolve()
    if target.exists():
        raise FileExistsError(target)
    payload = {
        "format": INTEGER_RELATION_GRAPH_COURSE_FORMAT,
        "source_identity": list(course.source_identity),
        "license_identity": list(course.license_identity),
        "record_count": len(course.records),
        "records": [_record_dict(item) for item in course.records],
        "free_dialogue_complete": 0,
    }
    raw = (json.dumps(payload, ensure_ascii=True, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("ascii")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def read_integer_relation_graph_course(path: str | Path) -> IntegerRelationGraphCourse:
    """只读恢复整数 graph-course；正文/答案字段不在合同内。"""
    raw = Path(path).resolve().read_bytes()
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise IntegerRelationGraphCourseError("graph-course 必须以单个换行结束")
    try:
        payload = parse_canonical_json_bytes(raw[:-1], require_object=True)
    except ValueError as error:
        raise IntegerRelationGraphCourseError("graph-course JSON 不规范") from error
    if set(payload) != _COURSE_FIELDS:
        raise IntegerRelationGraphCourseError("graph-course 顶层字段集合漂移")
    if payload.get("format") != INTEGER_RELATION_GRAPH_COURSE_FORMAT:
        raise IntegerRelationGraphCourseError("graph-course format 漂移")
    if not isinstance(payload.get("records"), list):
        raise IntegerRelationGraphCourseError("graph-course records 必须是列表")
    records = tuple(sorted(
        (_record_from_dict(item) for item in payload["records"]),
        key=lambda item: item.stable_key()))
    if (type(payload.get("record_count")) is not int
            or payload["record_count"] != len(records)
            or payload.get("free_dialogue_complete") != 0):
        raise IntegerRelationGraphCourseError("graph-course count/gate 漂移")
    return IntegerRelationGraphCourse(
        _key(payload.get("source_identity"), where="source_identity", positive=True),
        _key(payload.get("license_identity"), where="license_identity", positive=True),
        records,
    )


def integer_course_from_w06_adapter(
        adapter: W06TypedAdapterOutput,
        *,
        source_identity: tuple[int, ...],
        license_identity: tuple[int, ...],
        ) -> IntegerRelationGraphCourse:
    """在训练边界把现有 W-06 adapter 压缩为整数 graph-course。"""
    if not isinstance(adapter, W06TypedAdapterOutput):
        raise TypeError("adapter 类型错误")
    rows = []
    entries = [
        (item.observation, item.source.record)
        for item in adapter.observations
    ]
    entries.extend(
        (item.observation, item.source_record)
        for item in adapter.rejections
    )
    teacher_by_observation = {
        item.observation.stable_key: item.teacher_record
        for item in (*adapter.evidence, *adapter.rejection_evidence)
    }
    for observation, source_record in entries:
        if observation.split != "train":
            continue
        value = observation.typed_payload.to_value()
        profile = W06_RELATION_PROFILES.get(value.get("relation_family"))
        if profile is None:
            raise IntegerRelationGraphCourseError("adapter relation family 未注册")
        anchor = tuple(value["source_anchor_span"])
        teacher_record = teacher_by_observation.get(observation.stable_key)
        if teacher_record is None:
            raise IntegerRelationGraphCourseError(
                "adapter observation 缺少匹配 teacher Evidence")
        teacher_value = teacher_record.typed_evidence.to_value()
        expected_payload = teacher_value.get("expected_payload")
        if (not isinstance(expected_payload, dict)
                or not isinstance(expected_payload.get("decision"), str)):
            raise IntegerRelationGraphCourseError(
                "teacher decision 必須是有來源的课程标注")
        rows.append(IntegerRelationGraphRecord(
            tuple(source_record.stable_key.components),
            tuple(observation.dataset_key.components),
            tuple(observation.artifact_key.components),
            tuple(observation.stable_key.components),
            tuple(value["semantic_key"]),
            tuple(value["candidate_definition"]["proposition_key"]),
            tuple(value["candidate_definition"]["predicate_key"]),
            tuple(value["candidate_definition"]["context_key"]),
            tuple(value["candidate_definition"]["source_anchor_key"]),
            tuple(value["relation_schema"]["schema_key"]),
            tuple(value["relation_schema"]["relation_key"]),
            tuple(value["candidate_definition"]["role_bindings"]),
            tuple(value["relation_schema"]["slots"]),
            tuple({key: item[key] for key in _ENDPOINT_FIELDS}
                  for item in value["endpoints"]),
            tuple(map(ord, value["surface"])),
            anchor,
            profile.relation_kind,
            profile.directionality,
            {"support": 1, "refute": 2, "supersede": 3, "conflict": 4}[
                observation.sample_role],
            _PERTURBATIONS.index(observation.perturbation_kind),
            observation.logical_order,
            tuple(source_record.source_cluster_key.components),
            tuple(observation.dedup_cluster_key.components),
            tuple(observation.content_group_key.components),
            tuple(observation.template_group_key.components),
            tuple(observation.shape_group_key.components),
            tuple(value["relation_schema"]["constraints"]),
            value["consumer_request"],
            {key: value[key] for key in _OPTIONAL_PAYLOAD_FIELDS if key in value},
            _STATE_BY_NAME[teacher_value["expected_state"]],
            tuple(map(ord, expected_payload["decision"])),
            _encode_integer_payload(
                expected_payload, where="teacher_expected_payload"),
            tuple(map(ord, teacher_value.get(
                "seed_id", "integer-graph-course-teacher"))),
            (None if observation.supersedes_key is None else tuple(
                observation.supersedes_key.components)),
            tuple(tuple(item.components) for item in observation.prerequisite_keys),
        ))
    if not rows:
        raise IntegerRelationGraphCourseError("adapter 没有 train relation record")
    return IntegerRelationGraphCourse(
        tuple(source_identity), tuple(license_identity),
        tuple(sorted(rows, key=lambda item: item.stable_key())))


def _payload(record: IntegerRelationGraphRecord) -> dict[str, object]:
    """把整数 record 恢复成 W-06 adapter 所需的短期 typed payload。"""
    family, profile = _PROFILE_BY_KIND[record.relation_kind]
    surface = _surface(record.surface_values)
    payload: dict[str, object] = {
        "candidate_definition": {
            "context_key": list(record.context_key),
            "predicate_key": list(record.predicate_key),
            "proposition_key": list(record.proposition_key),
            "role_bindings": [
                _object_value(item) for item in record.role_bindings],
            "source_anchor_key": list(record.source_anchor_key),
        },
        "consumer_request": _object_value(record.consumer_request),
        "directionality": record.directionality,
        "endpoints": [
            {**_object_value(item), "surface_fragment": surface[
                _object_value(item)["start"]:_object_value(item)["end"]]}
            for item in record.endpoints
        ],
        "query_kind": "typed_relation_candidate",
        "relation_family": family,
        "relation_schema": {
            "constraints": [
                _object_value(item) for item in record.relation_constraints],
            "relation_key": list(record.relation_key),
            "schema_key": list(record.schema_key),
            "slots": [_object_value(item) for item in record.slots],
        },
        "semantic_key": list(record.semantic_key),
        "source_anchor_span": list(record.anchor_span),
        "surface": surface,
    }
    payload.update(_object_value(record.optional_payload))
    return payload


def _source_record(
        record: IntegerRelationGraphRecord, digest: str,
        course: IntegerRelationGraphCourse,
        ) -> SourceRefRecord:
    """建立只含整数 Span 摘要的训练期 SourceRefRecord。"""
    source_token = ".".join(str(item) for item in course.source_identity)
    license_token = ".".join(str(item) for item in course.license_identity)
    return SourceRefRecord(
        1, 1, 1,
        StableRecordKey(record.dataset_key),
        StableRecordKey(record.artifact_key),
        StableRecordKey(record.source_ref_key),
        "AUTHORED_CC0_V1",
        "integer-graph-course",
        "integer-graph-course-v1",
        f"urn:pure-integer:graph-course:{source_token}",
        f"pure_integer_ai source={source_token} license={license_token}",
        "sha256:" + digest,
        digest,
        LICENSE_ID,
        "PUBLIC",
        "pure_integer_ai integer graph course",
        1,
        CanonicalJsonObject.from_value({
            "surface_values": list(record.surface_values),
            "source_identity": list(course.source_identity),
            "license_identity": list(course.license_identity),
        }),
        record.logical_order,
        StableRecordKey(record.source_cluster_key),
    )


def integer_relation_graph_course_to_w06_adapter(
        course: IntegerRelationGraphCourse,
        *,
        require_full_curriculum: bool = True,
        ) -> W06TypedAdapterOutput:
    """从整数 course 恢复 W-06 typed adapter，不读取正文或旧答案。"""
    if not isinstance(course, IntegerRelationGraphCourse):
        raise TypeError("course 类型错误")
    sources, observations, teachers = [], [], []
    for record in course.records:
        digest = hashlib.sha256(
            json.dumps(_record_dict(record), sort_keys=True,
                       separators=(",", ":")).encode("ascii")).hexdigest()
        source = _source_record(record, digest, course)
        payload = _payload(record)
        profile = _PROFILE_BY_KIND[record.relation_kind][1]
        observation = ObservationRecord(
            1, 1, 1,
            StableRecordKey(record.dataset_key),
            StableRecordKey(record.artifact_key),
            StableRecordKey(record.observation_key),
            "W-06", profile.substage_key, "train", "zh",
            "integer-graph-relation", source.stable_key, LICENSE_ID,
            StableRecordKey(record.dedup_cluster_key),
            StableRecordKey(record.content_group_key),
            StableRecordKey(record.template_group_key),
            StableRecordKey(record.shape_group_key),
            "forming", _SAMPLE_ROLE_NAMES[record.sample_role_code],
            "TypedRelationQuery", CanonicalJsonObject.from_value(payload),
            _PERTURBATIONS[record.perturbation_code],
            (None if record.supersedes_key is None
             else StableRecordKey(record.supersedes_key)),
            tuple(StableRecordKey(item) for item in record.prerequisite_keys),
            record.logical_order,
        )
        teacher = TeacherEvidenceRecord(
            1, 1, 1,
            StableRecordKey(record.dataset_key),
            StableRecordKey(record.artifact_key),
            StableRecordKey((1, 99002, *record.observation_key)),
            observation.stable_key,
            _TEACHER_KIND_BY_SUBSTAGE[profile.substage_key],
            CanonicalJsonObject.from_value({
                "expected_payload": _decode_integer_payload(
                    _object_value(record.teacher_expected_payload),
                    where="teacher_expected_payload"),
                "expected_state": _STATE_CODES[
                    record.teacher_expected_state_code],
                "seed_id": _surface(record.teacher_seed_values),
            }),
            source.stable_key, "W-06", 0,
            StableRecordKey((1, 99001, *course.source_identity)),
        )
        sources.append(source)
        observations.append(observation)
        teachers.append(teacher)
    return adapt_w06_training_payload(W06TrainingPayload(
        tuple(sources), tuple(observations), tuple(teachers)),
        require_full_curriculum=require_full_curriculum)


__all__ = [
    "INTEGER_RELATION_GRAPH_COURSE_FORMAT",
    "IntegerRelationGraphCourse",
    "IntegerRelationGraphCourseError",
    "IntegerRelationGraphRecord",
    "integer_course_from_w06_adapter",
    "integer_relation_graph_course_to_w06_adapter",
    "read_integer_relation_graph_course",
    "write_integer_relation_graph_course",
]
