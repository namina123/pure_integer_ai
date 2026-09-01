"""把 W-06 typed relation 课程无损适配到现役 S-00/R-00/H-05 对象。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pure_integer_ai.cognition.shared.evidence_candidate import (
    CANDIDATE_AS_SUBJECT,
    CandidateBinding,
)
from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EVIDENCE_UNKNOWN,
)
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    OBJECT_CONTEXT_SCOPE,
    OBJECT_OCCURRENCE,
    OBJECT_PROPOSITION,
    OBJECT_ROLE,
    OBJECT_ROLE_BINDING,
    CorpusVersion,
    CurriculumVersion,
    ObjectIdentity,
    ParserVersion,
    PrimitiveVersion,
    SourceRef,
    VersionBundle,
    concept_identity,
)
from pure_integer_ai.cognition.shared.relation_closure import (
    RelationClosureCandidateSpec,
    RelationClosureProtocol,
)
from pure_integer_ai.cognition.shared.semantic_object import (
    AtomicPropositionDefinition,
    AtomicRoleBinding,
    semantic_source,
)
from pure_integer_ai.cognition.shared.typed_relation import (
    RelationSchema,
    RelationSlotSchema,
    SameKindConstraint,
)
from pure_integer_ai.experiments.ph2_authored_relation_compile import (
    authored_relation_identity,
    authored_relation_role_identity,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    ObservationRecord,
    SourceRefRecord,
    StableRecordKey,
    TeacherEvidenceRecord,
)
from pure_integer_ai.experiments.ph2_w06_contract import W06_ZERO_EXECUTION_STATE
from pure_integer_ai.experiments.ph2_w06_payload import W06TrainingPayload
from pure_integer_ai.experiments.ph2_w06_registry import (
    W06_RELATION_REGISTRY,
    W06RegistryError,
    audit_w06_registry_payload,
)
from pure_integer_ai.experiments.ph2_w06_source_semantic import (
    W06_RELATION_PROFILES,
    W06RelationProfile,
)


W06_NAMESPACE = 60606
W06_IDENTITY_VERSIONS = VersionBundle(
    CorpusVersion(1),
    ParserVersion(1),
    PrimitiveVersion(1),
    CurriculumVersion(1),
)
W06_SELECTION_UNSELECTED = "UNSELECTED"
W06_REJECTION_TYPE_MISMATCH = "TYPE_MISMATCH_REJECTED"

_PAYLOAD_REQUIRED_FIELDS = frozenset({
    "candidate_definition",
    "consumer_request",
    "directionality",
    "endpoints",
    "query_kind",
    "relation_family",
    "relation_schema",
    "surface",
})
_PAYLOAD_OPTIONAL_FIELDS = frozenset({
    "causal_protocol",
    "confidence_units",
    "event_time_protocol",
    "mereology_protocol",
    "polarity_key",
    "rational_role_values",
    "semantic_pair_protocol",
    "uncertainty_units",
})
_CANDIDATE_FIELDS = frozenset({
    "context_key",
    "predicate_key",
    "proposition_key",
    "role_bindings",
    "source_anchor_key",
})
_ROLE_BINDING_FIELDS = frozenset({
    "binding_key", "filler_key", "ordinal", "role_key"})
_SCHEMA_FIELDS = frozenset({
    "constraints", "relation_key", "schema_key", "slots"})
_SLOT_FIELDS = frozenset({
    "allowed_object_kinds", "max_count", "min_count", "role_key"})
_CONSTRAINT_FIELDS = frozenset({"constraint_key", "role_keys"})
_ENDPOINT_FIELDS = frozenset({
    "end", "endpoint_key", "object_kind", "ordinal", "start",
    "surface_fragment",
})
_TEACHER_FIELDS = frozenset({"expected_payload", "expected_state", "seed_id"})


class W06TypedAdapterError(ValueError):
    """W-06 typed relation、schema、来源或 Evidence 绑定不闭合。"""


def _pack(value: tuple[int, ...]) -> tuple[int, ...]:
    """给可变长稳定键增加长度前缀，避免字段拼接歧义。"""
    return len(value), *value


def _text_key(value: str) -> tuple[int, ...]:
    """把课程提供的控制文本编码为确定性整数键，不解释自然语言表面。"""
    if not isinstance(value, str) or not value or value.strip() != value:
        raise W06TypedAdapterError("W-06 adapter text key 非法")
    return len(value), *(ord(character) for character in value)


def _strict_nonnegative(value: Any, *, where: str) -> int:
    """校验 span、ordinal、置信量和基数字段为非负严格整数。"""
    if type(value) is not int or value < 0:
        raise W06TypedAdapterError(f"{where} 必须是非负严格整数")
    return value


def _object_key(value: Any, *, where: str) -> ObjectIdentity:
    """从规范 JSON 整数列表恢复完整 ObjectIdentity。"""
    if (not isinstance(value, list) or not value
            or any(type(item) is not int for item in value)):
        raise W06TypedAdapterError(f"{where} 必须是非空严格整数列表")
    try:
        return ObjectIdentity.from_stable_key(tuple(value))
    except (TypeError, ValueError) as error:
        raise W06TypedAdapterError(f"{where} 不是合法 ObjectIdentity") from error


def _optional_units(value: Any, *, where: str) -> int | None:
    """恢复可选的整数置信或不确定性量，缺省不伪造数值。"""
    if value is None:
        return None
    return _strict_nonnegative(value, where=where)


def _occurrence_source(identity: ObjectIdentity) -> SourceRef:
    """从 Occurrence 全键恢复 SourceRef，并拒绝截断布局。"""
    if identity.object_kind != OBJECT_OCCURRENCE or len(identity.components) < 4:
        raise W06TypedAdapterError("W-06 occurrence identity 类型或长度非法")
    try:
        return SourceRef.from_stable_key(identity.components[:-3])
    except (TypeError, ValueError) as error:
        raise W06TypedAdapterError("W-06 occurrence SourceRef 非法") from error


def _domain_predicate(kind: int) -> ObjectIdentity:
    """返回只描述结构字段的候选 binding predicate，不承载 relation 真值。"""
    return concept_identity(
        (W06_NAMESPACE, 400, kind),
        versions=W06_IDENTITY_VERSIONS,
    )


def _direction_identity(directionality: int) -> ObjectIdentity:
    """把冻结方向枚举提升为一等离散字段值。"""
    return concept_identity(
        (W06_NAMESPACE, 401, directionality),
        versions=W06_IDENTITY_VERSIONS,
    )


def w06_directionality_binding_predicate() -> ObjectIdentity:
    """返回训练候选中承载关系方向性的一等字段 predicate。"""
    return _domain_predicate(3)


def w06_directionality_value(directionality: int) -> ObjectIdentity:
    """返回训练图中一个冻结方向枚举对应的一等字段值。"""
    return _direction_identity(directionality)


@dataclass(frozen=True)
class W06SourceBinding:
    """连接 D-02 SourceRefRecord 与 payload 内一等语义 SourceRef。"""

    record: SourceRefRecord
    semantic_source: SourceRef

    def __post_init__(self) -> None:
        if not isinstance(self.record, SourceRefRecord):
            raise TypeError("W-06 source binding record 类型非法")
        if not isinstance(self.semantic_source, SourceRef):
            raise TypeError("W-06 source binding semantic source 类型非法")


@dataclass(frozen=True)
class W06RelationEndpoint:
    """保留 relation endpoint 的一等身份、对象类型和来源 span。"""

    identity: ObjectIdentity
    object_kind: int
    start: int
    end: int
    ordinal: int
    surface_fragment: str

    def __post_init__(self) -> None:
        if self.identity.object_kind != self.object_kind:
            raise W06TypedAdapterError("W-06 endpoint object kind 与身份不一致")
        if self.end <= self.start or not self.surface_fragment:
            raise W06TypedAdapterError("W-06 endpoint span 或 fragment 非法")


@dataclass(frozen=True)
class W06RoleTypeViolation:
    """记录一个 proposal Role 的权威允许类型、payload slot 与实际 filler。"""

    role: ObjectIdentity
    filler: ObjectIdentity
    allowed_object_kinds: tuple[int, ...]
    declared_object_kinds: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.role.object_kind != OBJECT_ROLE:
            raise W06TypedAdapterError("W-06 rejection violation Role 类型非法")
        if (not self.allowed_object_kinds or not self.declared_object_kinds
                or any(type(item) is not int for item in (
                    *self.allowed_object_kinds, *self.declared_object_kinds))):
            raise W06TypedAdapterError("W-06 rejection violation 类型集合非法")
        if (self.filler.object_kind in self.allowed_object_kinds
                and set(self.declared_object_kinds).issubset(
                    self.allowed_object_kinds)):
            raise W06TypedAdapterError("W-06 rejection violation 没有真实类型破坏")


@dataclass(frozen=True)
class W06SchemaRejection:
    """在 relation candidate 形成前被 W-06 schema firewall 拒绝的 proposal。"""

    observation: ObservationRecord
    source_record: SourceRefRecord
    source_ref: SourceRef
    substage_key: str
    relation_family: str
    directionality: int
    surface: str
    endpoints: tuple[W06RelationEndpoint, ...]
    schema_identity: ObjectIdentity
    relation: ObjectIdentity
    proposition: ObjectIdentity
    consumer_request: CanonicalJsonObject
    proposal_payload: CanonicalJsonObject
    violations: tuple[W06RoleTypeViolation, ...]
    rejection_state: str
    provenance: CanonicalJsonObject

    def __post_init__(self) -> None:
        if (self.rejection_state != W06_REJECTION_TYPE_MISMATCH
                or self.observation.sample_role != "refute"
                or self.observation.perturbation_kind != "TYPE_MISMATCH"
                or not self.violations):
            raise W06TypedAdapterError("W-06 schema rejection 合同不闭合")
        if (self.proposition.object_kind != OBJECT_PROPOSITION
                or self.source_ref != _occurrence_source(_object_key(
                    self.proposal_payload.to_value()["candidate_definition"][
                        "source_anchor_key"],
                    where="rejection.source_anchor_key",
                ))):
            raise W06TypedAdapterError("W-06 schema rejection 命题或来源漂移")


@dataclass(frozen=True)
class W06RelationCandidate:
    """一个未预选的 typed relation 候选及其来源、修订和 consumer 信封。"""

    observation: ObservationRecord
    source_record: SourceRefRecord
    source_ref: SourceRef
    substage_key: str
    relation_family: str
    directionality: int
    surface: str
    endpoints: tuple[W06RelationEndpoint, ...]
    schema: RelationSchema
    proposition: AtomicPropositionDefinition
    spec: RelationClosureCandidateSpec
    consumer_request: CanonicalJsonObject
    domain_protocols: CanonicalJsonObject
    rational_role_values: CanonicalJsonObject | None
    polarity: ObjectIdentity | None
    confidence_units: int | None
    uncertainty_units: int | None
    sample_role: str
    perturbation_kind: str
    supersedes_observation_key: StableRecordKey | None
    selection_state: str
    provenance: CanonicalJsonObject

    def __post_init__(self) -> None:
        if self.selection_state != W06_SELECTION_UNSELECTED:
            raise W06TypedAdapterError("W-06 adapter 不得预选 relation candidate")
        if (self.proposition.proposition != self.spec.proposition.proposition
                or self.schema != self.spec.schema
                or self.source_ref != self.proposition.source):
            raise W06TypedAdapterError("W-06 candidate 的命题、schema 或来源漂移")
        if self.polarity is not None and not isinstance(
                self.polarity, ObjectIdentity):
            raise TypeError("W-06 polarity 必须是一等对象或 None")


@dataclass(frozen=True)
class W06ObservationEnvelope:
    """一个 W-06 TypedRelationQuery 的来源和候选信封。"""

    observation: ObservationRecord
    source: W06SourceBinding
    candidate: W06RelationCandidate


@dataclass(frozen=True)
class W06EvidenceBinding:
    """train TeacherEvidence 到 H-05 stance 与 lifecycle 请求的薄映射。"""

    teacher_record: TeacherEvidenceRecord
    observation: ObservationRecord
    candidate: ObjectIdentity
    stances: tuple[int, ...]
    decision: str
    expected_payload: CanonicalJsonObject
    reason_key: tuple[int, ...]
    logical_order: int
    withdrawal_level: int
    supersedes_observation_key: StableRecordKey | None


@dataclass(frozen=True)
class W06RejectionEvidenceBinding:
    """把 FALSE teacher Evidence 归因到 schema rejection，而非 H-05 hypothesis。"""

    teacher_record: TeacherEvidenceRecord
    observation: ObservationRecord
    rejection_state: str
    expected_state: str
    decision: str
    expected_payload: CanonicalJsonObject
    reason_key: tuple[int, ...]
    logical_order: int
    withdrawal_level: int


@dataclass(frozen=True)
class W06TypedAdapterOutput:
    """W06-02 的纯确定性适配结果，不执行 transaction 或 learning write。"""

    source_bindings: tuple[W06SourceBinding, ...]
    observations: tuple[W06ObservationEnvelope, ...]
    candidates: tuple[W06RelationCandidate, ...]
    schemas: tuple[RelationSchema, ...]
    evidence: tuple[W06EvidenceBinding, ...]
    rejections: tuple[W06SchemaRejection, ...]
    rejection_evidence: tuple[W06RejectionEvidenceBinding, ...]
    execution_state: tuple[tuple[str, int], ...] = tuple(
        sorted(W06_ZERO_EXECUTION_STATE.items()))

    def candidates_for_substage(
            self, substage_key: str) -> tuple[W06RelationCandidate, ...]:
        """按冻结子阶段返回候选，不按 surface 或 cue 路由。"""
        if substage_key not in W06_RELATION_REGISTRY:
            raise W06TypedAdapterError("W-06 adapter substage 未注册")
        return tuple(
            item for item in self.candidates
            if item.substage_key == substage_key)

    def candidates_for_relation(
            self, relation_family: str) -> tuple[W06RelationCandidate, ...]:
        """按显式 relation family 返回候选，未知 family fail closed。"""
        if relation_family not in W06_RELATION_PROFILES:
            raise W06TypedAdapterError("W-06 adapter relation 未注册")
        return tuple(
            item for item in self.candidates
            if item.relation_family == relation_family)

    def rejections_for_relation(
            self, relation_family: str) -> tuple[W06SchemaRejection, ...]:
        """按显式 relation family 返回 schema rejection，未知 family fail closed。"""
        if relation_family not in W06_RELATION_PROFILES:
            raise W06TypedAdapterError("W-06 adapter relation 未注册")
        return tuple(
            item for item in self.rejections
            if item.relation_family == relation_family)


def _typed_payload(observation: ObservationRecord) -> dict[str, Any]:
    """读取并核验 TypedRelationQuery 顶层字段，不接受额外语义字段偷渡。"""
    value = observation.typed_payload.to_value()
    if not isinstance(value, dict):
        raise W06TypedAdapterError("W-06 relation payload 必须是 object")
    fields = set(value)
    if (not _PAYLOAD_REQUIRED_FIELDS.issubset(fields)
            or fields - _PAYLOAD_REQUIRED_FIELDS - _PAYLOAD_OPTIONAL_FIELDS):
        raise W06TypedAdapterError("W-06 relation payload 字段集合漂移")
    if value.get("query_kind") != "typed_relation_candidate":
        raise W06TypedAdapterError("W-06 relation query_kind 漂移")
    if not isinstance(value.get("surface"), str) or not value["surface"]:
        raise W06TypedAdapterError("W-06 relation surface 非法")
    if not isinstance(value.get("endpoints"), list) or not value["endpoints"]:
        raise W06TypedAdapterError("W-06 relation endpoints 非法")
    candidate = value.get("candidate_definition")
    schema = value.get("relation_schema")
    consumer = value.get("consumer_request")
    if not isinstance(candidate, dict) or set(candidate) != _CANDIDATE_FIELDS:
        raise W06TypedAdapterError("W-06 candidate_definition 字段漂移")
    if not isinstance(candidate.get("role_bindings"), list):
        raise W06TypedAdapterError("W-06 candidate role_bindings 非列表")
    if not isinstance(schema, dict) or set(schema) != _SCHEMA_FIELDS:
        raise W06TypedAdapterError("W-06 relation_schema 字段漂移")
    if not isinstance(consumer, dict):
        raise W06TypedAdapterError("W-06 consumer_request 非 object")
    return value


def _adapt_endpoints(
        values: list[Any], *, surface: str,
        ) -> tuple[W06RelationEndpoint, ...]:
    """恢复 endpoint 身份并逐 span 核对来源表面，不做 cue-to-label 映射。"""
    result = []
    for value in values:
        if not isinstance(value, dict) or set(value) != _ENDPOINT_FIELDS:
            raise W06TypedAdapterError("W-06 endpoint 字段集合漂移")
        start = _strict_nonnegative(value["start"], where="endpoint.start")
        end = _strict_nonnegative(value["end"], where="endpoint.end")
        ordinal = _strict_nonnegative(value["ordinal"], where="endpoint.ordinal")
        fragment = value["surface_fragment"]
        if (end > len(surface) or not isinstance(fragment, str)
                or surface[start:end] != fragment):
            raise W06TypedAdapterError("W-06 endpoint span 与 surface 不一致")
        kind = value["object_kind"]
        if type(kind) is not int:
            raise W06TypedAdapterError("W-06 endpoint object kind 非严格整数")
        result.append(W06RelationEndpoint(
            _object_key(value["endpoint_key"], where="endpoint.endpoint_key"),
            kind,
            start,
            end,
            ordinal,
            fragment,
        ))
    identities = [item.identity for item in result]
    if len(identities) != len(set(identities)):
        raise W06TypedAdapterError("W-06 endpoint identity 重复")
    return tuple(result)


def _adapt_schema(value: dict[str, Any]) -> RelationSchema:
    """恢复 relation/schema/Role slot 与 same-kind constraint 全结构。"""
    relation = _object_key(value["relation_key"], where="schema.relation_key")
    schema_key = _object_key(value["schema_key"], where="schema.schema_key")
    raw_slots = value["slots"]
    raw_constraints = value["constraints"]
    if not isinstance(raw_slots, list) or not raw_slots:
        raise W06TypedAdapterError("W-06 schema slots 为空或非列表")
    if not isinstance(raw_constraints, list):
        raise W06TypedAdapterError("W-06 schema constraints 非列表")
    slots = []
    for item in raw_slots:
        if not isinstance(item, dict) or set(item) != _SLOT_FIELDS:
            raise W06TypedAdapterError("W-06 schema slot 字段漂移")
        allowed = item["allowed_object_kinds"]
        if (not isinstance(allowed, list) or not allowed
                or any(type(kind) is not int for kind in allowed)):
            raise W06TypedAdapterError("W-06 slot allowed_object_kinds 非法")
        maximum = item["max_count"]
        if maximum is not None:
            maximum = _strict_nonnegative(maximum, where="slot.max_count")
        slots.append(RelationSlotSchema(
            _object_key(item["role_key"], where="slot.role_key"),
            frozenset(allowed),
            _strict_nonnegative(item["min_count"], where="slot.min_count"),
            maximum,
        ))
    constraints = []
    for item in raw_constraints:
        if not isinstance(item, dict) or set(item) != _CONSTRAINT_FIELDS:
            raise W06TypedAdapterError("W-06 same-kind constraint 字段漂移")
        roles = item["role_keys"]
        if not isinstance(roles, list):
            raise W06TypedAdapterError("W-06 constraint role_keys 非列表")
        constraints.append(SameKindConstraint(
            _object_key(item["constraint_key"], where="constraint.constraint_key"),
            tuple(_object_key(role, where="constraint.role_key")
                  for role in roles),
        ))
    try:
        return RelationSchema(
            schema_key, relation, tuple(slots), tuple(constraints))
    except (TypeError, ValueError) as error:
        raise W06TypedAdapterError("W-06 RelationSchema 非法") from error


def _adapt_proposition(
        value: dict[str, Any],
        schema: RelationSchema,
        endpoints: tuple[W06RelationEndpoint, ...],
        *,
        validate_schema: bool = True,
        ) -> AtomicPropositionDefinition:
    """恢复 Proposition/Context/RoleBinding，并与 schema 和 endpoint 交叉核验。"""
    proposition = _object_key(value["proposition_key"], where="proposition_key")
    predicate = _object_key(value["predicate_key"], where="predicate_key")
    anchor = _object_key(value["source_anchor_key"], where="source_anchor_key")
    context = _object_key(value["context_key"], where="context_key")
    if (proposition.object_kind != OBJECT_PROPOSITION
            or anchor.object_kind != OBJECT_OCCURRENCE
            or context.object_kind != OBJECT_CONTEXT_SCOPE):
        raise W06TypedAdapterError("W-06 Proposition/anchor/context 对象类型漂移")
    bindings = []
    binding_ids = set()
    endpoint_ids = {item.identity for item in endpoints}
    for item in value["role_bindings"]:
        if not isinstance(item, dict) or set(item) != _ROLE_BINDING_FIELDS:
            raise W06TypedAdapterError("W-06 RoleBinding 字段集合漂移")
        role = _object_key(item["role_key"], where="binding.role_key")
        filler = _object_key(item["filler_key"], where="binding.filler_key")
        binding_id = _object_key(item["binding_key"], where="binding.binding_key")
        ordinal = _strict_nonnegative(item["ordinal"], where="binding.ordinal")
        if role.object_kind != OBJECT_ROLE or filler not in endpoint_ids:
            raise W06TypedAdapterError("W-06 Role 或 filler 未匹配 endpoint")
        binding = AtomicRoleBinding(role, filler, ordinal)
        if (binding_id.object_kind != OBJECT_ROLE_BINDING
                or binding.identity_for(proposition) != binding_id
                or binding_id in binding_ids):
            raise W06TypedAdapterError("W-06 RoleBinding identity 漂移或重复")
        binding_ids.add(binding_id)
        bindings.append(binding)
    if len(bindings) != len(endpoints):
        raise W06TypedAdapterError("W-06 RoleBinding 未恰好覆盖 endpoint")
    try:
        definition = AtomicPropositionDefinition(
            proposition, predicate, anchor, context, tuple(bindings))
        if (_occurrence_source(anchor) != definition.source
                or semantic_source(context) != definition.source):
            raise W06TypedAdapterError("W-06 anchor/context 与 Proposition 来源漂移")
        if validate_schema:
            schema.validate_definition(definition)
        return definition
    except (TypeError, ValueError) as error:
        if isinstance(error, W06TypedAdapterError):
            raise
        raise W06TypedAdapterError("W-06 AtomicPropositionDefinition 非法") from error


def _profile_type_violations(
        profile: W06RelationProfile,
        schema: RelationSchema,
        proposition: AtomicPropositionDefinition,
        ) -> tuple[W06RoleTypeViolation, ...]:
    """用独立 W-06 relation profile 审计 payload slot 与实际 Role filler。"""
    expected = {
        authored_relation_role_identity(role_kind): allowed
        for role_kind, allowed in profile.role_object_kinds
    }
    slots = {slot.role: slot for slot in schema.slots}
    bindings = {binding.role: binding for binding in proposition.bindings}
    if (set(slots) != set(expected) or set(bindings) != set(expected)
            or len(slots) != len(schema.slots)
            or len(bindings) != len(proposition.bindings)):
        raise W06TypedAdapterError("W-06 schema/profile Role 集合漂移")
    violations = []
    for role in sorted(expected, key=ObjectIdentity.stable_key):
        slot = slots[role]
        binding = bindings[role]
        if slot.min_count != 1 or slot.max_count != 1:
            raise W06TypedAdapterError("W-06 schema Role 基数漂移")
        allowed = expected[role]
        declared = tuple(sorted(slot.allowed_object_kinds))
        if (binding.filler.object_kind not in allowed
                or not set(declared).issubset(allowed)):
            violations.append(W06RoleTypeViolation(
                role,
                binding.filler,
                tuple(sorted(allowed)),
                declared,
            ))
    return tuple(violations)


def _competition_root(
        observation: ObservationRecord,
        observations: dict[StableRecordKey, ObservationRecord],
        ) -> StableRecordKey:
    """沿 append-only supersedes 链返回 lifecycle competition 根并拒绝环。"""
    current = observation
    seen = set()
    while current.supersedes_key is not None:
        if current.stable_key in seen:
            raise W06TypedAdapterError("W-06 supersedes 链成环")
        seen.add(current.stable_key)
        prior = observations.get(current.supersedes_key)
        if (prior is None
                or prior.logical_order >= current.logical_order
                or prior.substage != current.substage):
            raise W06TypedAdapterError("W-06 supersedes target 缺失、过晚或跨子阶段")
        current = prior
    return current.stable_key


def _candidate_spec(
        observation: ObservationRecord,
        proposition: AtomicPropositionDefinition,
        schema: RelationSchema,
        directionality: int,
        competition_root: StableRecordKey,
        ) -> RelationClosureCandidateSpec:
    """把 typed relation 接入现役 R-00 候选定义并保留结构字段。"""
    domain = [
        CandidateBinding(
            _domain_predicate(1), proposition.source_anchor,
            0, CANDIDATE_AS_SUBJECT),
        CandidateBinding(
            _domain_predicate(2), proposition.context,
            0, CANDIDATE_AS_SUBJECT),
        CandidateBinding(
            _domain_predicate(3), _direction_identity(directionality),
            0, CANDIDATE_AS_SUBJECT),
    ]
    for ordinal, binding in enumerate(proposition.canonical_bindings()):
        domain.append(CandidateBinding(
            _domain_predicate(4),
            binding.identity_for(proposition.proposition),
            ordinal,
            CANDIDATE_AS_SUBJECT,
        ))
    return RelationClosureCandidateSpec(
        proposition,
        schema,
        (W06_NAMESPACE, 100, *_pack(competition_root.stable_key())),
        (proposition.source,),
        tuple(domain),
    )


def _adapt_candidate(
        observation: ObservationRecord,
        source_record: SourceRefRecord,
        observations: dict[StableRecordKey, ObservationRecord],
        ) -> W06RelationCandidate | W06SchemaRejection:
    """把合法 proposal 恢复为 candidate，并在形成前分流类型拒绝项。"""
    value = _typed_payload(observation)
    relation_family = value["relation_family"]
    if not isinstance(relation_family, str):
        raise W06TypedAdapterError("W-06 relation_family 非文本")
    profile = W06_RELATION_PROFILES.get(relation_family)
    if profile is None or profile.substage_key != observation.substage:
        raise W06TypedAdapterError("W-06 relation family 与 substage 漂移")
    directionality = value["directionality"]
    if type(directionality) is not int or directionality != profile.directionality:
        raise W06TypedAdapterError("W-06 directionality 与 registry 漂移")
    endpoints = _adapt_endpoints(value["endpoints"], surface=value["surface"])
    schema = _adapt_schema(value["relation_schema"])
    proposition = _adapt_proposition(
        value["candidate_definition"],
        schema,
        endpoints,
        validate_schema=observation.perturbation_kind != "TYPE_MISMATCH",
    )
    if (schema.relation != authored_relation_identity(profile.relation_kind)
            or proposition.predicate != schema.relation):
        raise W06TypedAdapterError("W-06 relation identity 未匹配冻结 relation kind")
    violations = _profile_type_violations(profile, schema, proposition)
    if violations:
        if (observation.sample_role != "refute"
                or observation.perturbation_kind != "TYPE_MISMATCH"):
            raise W06TypedAdapterError("W-06 普通 relation proposal 违反类型 profile")
        return W06SchemaRejection(
            observation,
            source_record,
            proposition.source,
            observation.substage,
            relation_family,
            directionality,
            value["surface"],
            endpoints,
            schema.schema,
            schema.relation,
            proposition.proposition,
            CanonicalJsonObject.from_value(value["consumer_request"]),
            CanonicalJsonObject.from_value(value),
            violations,
            W06_REJECTION_TYPE_MISMATCH,
            CanonicalJsonObject.from_value({
                "adapter": "ph2_w06_typed_relation_v1",
                "payload_kind": observation.payload_kind,
                "query_kind": value["query_kind"],
                "source_revision": {
                    "parser_version": source_record.parser_version,
                    "revision_id": source_record.revision_id,
                    "snapshot_id": source_record.snapshot_id,
                },
            }),
        )
    if observation.perturbation_kind == "TYPE_MISMATCH":
        raise W06TypedAdapterError("W-06 TYPE_MISMATCH proposal 没有真实类型破坏")
    root = _competition_root(observation, observations)
    spec = _candidate_spec(
        observation, proposition, schema, directionality, root)
    rational = value.get("rational_role_values")
    rational_value = (
        None if rational is None else CanonicalJsonObject.from_value({
            "values": rational,
        }))
    polarity_value = value.get("polarity_key")
    polarity = (
        None if polarity_value is None
        else _object_key(polarity_value, where="polarity_key"))
    return W06RelationCandidate(
        observation,
        source_record,
        proposition.source,
        observation.substage,
        relation_family,
        directionality,
        value["surface"],
        endpoints,
        schema,
        proposition,
        spec,
        CanonicalJsonObject.from_value(value["consumer_request"]),
        CanonicalJsonObject.from_value({
            key: value[key]
            for key in sorted(value)
            if key.endswith("_protocol")
        }),
        rational_value,
        polarity,
        _optional_units(value.get("confidence_units"), where="confidence_units"),
        _optional_units(value.get("uncertainty_units"), where="uncertainty_units"),
        observation.sample_role,
        observation.perturbation_kind,
        observation.supersedes_key,
        W06_SELECTION_UNSELECTED,
        CanonicalJsonObject.from_value({
            "adapter": "ph2_w06_typed_relation_v1",
            "payload_kind": observation.payload_kind,
            "query_kind": value["query_kind"],
            "source_revision": {
                "parser_version": source_record.parser_version,
                "revision_id": source_record.revision_id,
                "snapshot_id": source_record.snapshot_id,
            },
        }),
    )


def _teacher_stances(teacher: TeacherEvidenceRecord) -> tuple[int, ...]:
    """把 teacher 四态映射为 H-05 support/refute/unknown stance。"""
    value = teacher.typed_evidence.to_value()
    mapping = {
        "TRUE": (EVIDENCE_SUPPORT,),
        "FALSE": (EVIDENCE_REFUTE,),
        "UNKNOWN": (EVIDENCE_UNKNOWN,),
        "CONFLICT": (EVIDENCE_SUPPORT, EVIDENCE_REFUTE),
    }
    result = mapping.get(value.get("expected_state"))
    if result is None:
        raise W06TypedAdapterError("W-06 teacher Evidence expected_state 非法")
    return result


def _evidence_binding(
        envelope: W06ObservationEnvelope,
        teacher: TeacherEvidenceRecord,
        ) -> W06EvidenceBinding:
    """把独立 train TeacherEvidence 绑定到 relation candidate。"""
    observation = envelope.observation
    value = teacher.typed_evidence.to_value()
    if (teacher.observation_key != observation.stable_key
            or teacher.source_ref_key != observation.source_ref_key
            or teacher.visible_from_stage != "W-06"):
        raise W06TypedAdapterError("W-06 teacher Evidence 引用漂移")
    if not isinstance(value, dict) or set(value) != _TEACHER_FIELDS:
        raise W06TypedAdapterError("W-06 teacher Evidence 字段集合漂移")
    expected_payload = value["expected_payload"]
    if not isinstance(expected_payload, dict):
        raise W06TypedAdapterError("W-06 teacher expected_payload 非 object")
    decision = expected_payload.get("decision")
    if not isinstance(decision, str) or not decision:
        raise W06TypedAdapterError("W-06 teacher expected_payload 缺 decision")
    reason_key = (
        W06_NAMESPACE,
        1,
        *_pack(teacher.stable_key.stable_key()),
        *_pack(_text_key(teacher.evidence_kind)),
        *_pack(_text_key(observation.perturbation_kind)),
        *_pack(_text_key(decision)),
        teacher.withdrawal_level,
    )
    return W06EvidenceBinding(
        teacher,
        observation,
        envelope.candidate.proposition.proposition,
        _teacher_stances(teacher),
        decision,
        CanonicalJsonObject.from_value(expected_payload),
        reason_key,
        observation.logical_order,
        teacher.withdrawal_level,
        observation.supersedes_key,
    )


def _rejection_evidence_binding(
        rejection: W06SchemaRejection,
        teacher: TeacherEvidenceRecord,
        ) -> W06RejectionEvidenceBinding:
    """只接受 FALSE teacher Evidence，并把它绑定到 rejection audit。"""
    observation = rejection.observation
    value = teacher.typed_evidence.to_value()
    if (teacher.observation_key != observation.stable_key
            or teacher.source_ref_key != observation.source_ref_key
            or teacher.visible_from_stage != "W-06"):
        raise W06TypedAdapterError("W-06 rejection teacher Evidence 引用漂移")
    if not isinstance(value, dict) or set(value) != _TEACHER_FIELDS:
        raise W06TypedAdapterError("W-06 rejection teacher Evidence 字段集合漂移")
    expected_payload = value["expected_payload"]
    decision = (
        expected_payload.get("decision")
        if isinstance(expected_payload, dict) else None
    )
    if value["expected_state"] != "FALSE" or decision != "reject_type_mismatch":
        raise W06TypedAdapterError("W-06 schema rejection 必须由明确 FALSE 决策支持")
    reason_key = (
        W06_NAMESPACE,
        2,
        *_pack(teacher.stable_key.stable_key()),
        *_pack(_text_key(teacher.evidence_kind)),
        *_pack(_text_key(observation.perturbation_kind)),
        *_pack(_text_key(decision)),
        teacher.withdrawal_level,
    )
    return W06RejectionEvidenceBinding(
        teacher,
        observation,
        rejection.rejection_state,
        value["expected_state"],
        decision,
        CanonicalJsonObject.from_value(expected_payload),
        reason_key,
        observation.logical_order,
        teacher.withdrawal_level,
    )


def adapt_w06_training_payload(
        payload: W06TrainingPayload,
        ) -> W06TypedAdapterOutput:
    """只适配 W-06 TypedRelationQuery，保留其他阶段 train pack 不消费。"""
    if not isinstance(payload, W06TrainingPayload):
        raise TypeError("W-06 adapter 只接受 W06TrainingPayload")
    try:
        audit_w06_registry_payload(payload)
    except W06RegistryError as error:
        raise W06TypedAdapterError(
            f"W-06 registry payload 审计失败：{error}") from error
    sources = {item.stable_key: item for item in payload.source_refs}
    observations = {
        item.stable_key: item for item in payload.observations
        if item.w_stage == "W-06" and item.payload_kind == "TypedRelationQuery"
    }
    teachers: dict[StableRecordKey, TeacherEvidenceRecord] = {}
    for teacher in payload.teacher_evidence:
        if teacher.observation_key not in observations:
            continue
        if teacher.observation_key in teachers:
            raise W06TypedAdapterError("W-06 Observation 有多个 teacher Evidence")
        teachers[teacher.observation_key] = teacher
    if set(teachers) != set(observations):
        raise W06TypedAdapterError("W-06 Observation/TeacherEvidence 不闭合")

    envelopes = []
    candidates = []
    evidence = []
    rejections = []
    rejection_evidence = []
    source_bindings: dict[StableRecordKey, W06SourceBinding] = {}
    schemas: dict[ObjectIdentity, RelationSchema] = {}
    for observation in sorted(
            observations.values(), key=lambda item: item.logical_order):
        source_record = sources.get(observation.source_ref_key)
        if (source_record is None
                or source_record.license_id != observation.license_partition):
            raise W06TypedAdapterError("W-06 Observation 来源或许可漂移")
        adapted = _adapt_candidate(observation, source_record, observations)
        source_binding = W06SourceBinding(source_record, adapted.source_ref)
        prior_source = source_bindings.get(source_record.stable_key)
        if prior_source is not None and prior_source != source_binding:
            raise W06TypedAdapterError("W-06 SourceRefRecord 绑定多个语义来源")
        source_bindings[source_record.stable_key] = source_binding
        teacher = teachers[observation.stable_key]
        if isinstance(adapted, W06SchemaRejection):
            rejections.append(adapted)
            rejection_evidence.append(_rejection_evidence_binding(
                adapted, teacher))
            continue
        candidate = adapted
        prior_schema = schemas.get(candidate.schema.schema)
        if prior_schema is not None and prior_schema != candidate.schema:
            raise W06TypedAdapterError("W-06 同一 schema identity 绑定不同定义")
        schemas[candidate.schema.schema] = candidate.schema
        envelope = W06ObservationEnvelope(
            observation, source_binding, candidate)
        envelopes.append(envelope)
        candidates.append(candidate)
        evidence.append(_evidence_binding(
            envelope, teacher))
    if len({item.proposition.proposition for item in candidates}) != len(candidates):
        raise W06TypedAdapterError("W-06 relation candidate identity 重复")
    if (len(candidates) + len(rejections) != len(observations)
            or len(evidence) != len(candidates)
            or len(rejection_evidence) != len(rejections)):
        raise W06TypedAdapterError("W-06 accepted/rejected proposal 分账不闭合")
    return W06TypedAdapterOutput(
        tuple(sorted(
            source_bindings.values(),
            key=lambda item: item.record.stable_key.stable_key(),
        )),
        tuple(sorted(
            envelopes,
            key=lambda item: item.observation.stable_key.stable_key(),
        )),
        tuple(sorted(
            candidates,
            key=lambda item: item.proposition.proposition.stable_key(),
        )),
        tuple(schemas[key] for key in sorted(schemas, key=ObjectIdentity.stable_key)),
        tuple(sorted(
            evidence,
            key=lambda item: (
                item.logical_order,
                item.teacher_record.stable_key.stable_key(),
            ),
        )),
        tuple(sorted(
            rejections,
            key=lambda item: item.observation.stable_key.stable_key(),
        )),
        tuple(sorted(
            rejection_evidence,
            key=lambda item: (
                item.logical_order,
                item.teacher_record.stable_key.stable_key(),
            ),
        )),
    )


def w06_relation_protocol() -> RelationClosureProtocol:
    """返回 W-06 独占 relation/schema 字段协议，供 lifecycle runtime 复用。"""
    from pure_integer_ai.cognition.shared.relation_closure import (
        RelationClosureField,
    )
    return RelationClosureProtocol(
        RelationClosureField(_domain_predicate(10)),
        RelationClosureField(_domain_predicate(11)),
    )


__all__ = [
    "W06EvidenceBinding",
    "W06ObservationEnvelope",
    "W06RejectionEvidenceBinding",
    "W06RelationCandidate",
    "W06RelationEndpoint",
    "W06RoleTypeViolation",
    "W06SchemaRejection",
    "W06SourceBinding",
    "W06TypedAdapterError",
    "W06TypedAdapterOutput",
    "W06_IDENTITY_VERSIONS",
    "W06_NAMESPACE",
    "W06_REJECTION_TYPE_MISMATCH",
    "adapt_w06_training_payload",
    "w06_directionality_binding_predicate",
    "w06_directionality_value",
    "w06_relation_protocol",
]
