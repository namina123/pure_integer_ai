"""W-05 occurrence、角色与原子命题的 train-only typed 适配层。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pure_integer_ai.cognition.shared.evidence_candidate import (
    CANDIDATE_AS_SUBJECT,
    CandidateBinding,
    EvidenceCandidateDefinition,
)
from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EVIDENCE_UNKNOWN,
)
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    OBJECT_CONTEXT_SCOPE,
    OBJECT_ENTITY,
    OBJECT_EVENT,
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
    occurrence_identity,
)
from pure_integer_ai.cognition.shared.semantic_object import (
    AtomicPropositionDefinition,
    AtomicRoleBinding,
    semantic_source,
    validate_semantic_identity,
)
from pure_integer_ai.experiments.ph2_authored_atomic_schema import (
    ALLOWED_PERTURBATIONS,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    ObservationRecord,
    SourceRefRecord,
    StableRecordKey,
    TeacherEvidenceRecord,
)
from pure_integer_ai.experiments.ph2_w05_payload import W05TrainingPayload


W05_NAMESPACE = 50505
W05_SOURCE_REF_KIND = 50502
W05_SELECTION_UNSELECTED = "UNSELECTED"
W05_IDENTITY_VERSIONS = VersionBundle(
    CorpusVersion(5),
    ParserVersion(1),
    PrimitiveVersion(1),
    CurriculumVersion(5),
)
W05_ATOMIC_IDENTITY_VERSIONS = VersionBundle(
    CorpusVersion(1),
    ParserVersion(1),
    PrimitiveVersion(1),
    CurriculumVersion(1),
)
W05_ZERO_EXECUTION_STATE = (
    ("LANGUAGE_CAPABILITY_MASTERED", 0),
    ("LANGUAGE_READINESS", 0),
    ("W05_STARTED", 0),
    ("W06_STARTED", 0),
    ("formal_w05_training_runs", 0),
    ("learning_writes", 0),
    ("teacher_calls", 0),
)

_PAYLOAD_FIELDS = frozenset({
    "candidate_definition",
    "occurrence_order",
    "occurrences",
    "query_kind",
    "surface",
})
_CANDIDATE_FIELDS = frozenset({
    "context_key",
    "predicate_key",
    "proposition_key",
    "role_bindings",
    "source_anchor_key",
})
_OCCURRENCE_FIELDS = frozenset({
    "end",
    "identity_key",
    "ordinal",
    "semantic_key",
    "semantic_kind",
    "start",
    "surface_fragment",
})
_ROLE_BINDING_FIELDS = frozenset({
    "binding_key",
    "filler_key",
    "ordinal",
    "role_key",
})
_TEACHER_FIELDS = frozenset({
    "expected_payload",
    "expected_state",
    "seed_id",
})


class W05TypedAdapterError(ValueError):
    """W-05 typed occurrence、命题或 teacher 绑定不闭合。"""


def _pack(value: tuple[int, ...]) -> tuple[int, ...]:
    """给可变长稳定键增加长度前缀。"""
    return len(value), *value


def _text_key(value: str) -> tuple[int, ...]:
    """无解释地编码课程已提供的控制文本。"""
    if not isinstance(value, str) or not value or value.strip() != value:
        raise W05TypedAdapterError("adapter text key 非法")
    return len(value), *(ord(character) for character in value)


def _strict_nonnegative(value: Any, *, where: str) -> int:
    """校验 span、ordinal 和顺序字段。"""
    if type(value) is not int or value < 0:
        raise W05TypedAdapterError(f"{where} 必须是非负严格整数")
    return value


def _object_key(value: Any, *, where: str) -> ObjectIdentity:
    """从 JSON 整数列表恢复完整 ObjectIdentity。"""
    if (not isinstance(value, list) or not value
            or any(type(item) is not int for item in value)):
        raise W05TypedAdapterError(f"{where} 必须是非空严格整数列表")
    try:
        return ObjectIdentity.from_stable_key(tuple(value))
    except (TypeError, ValueError) as exc:
        raise W05TypedAdapterError(f"{where} 不是合法 ObjectIdentity") from exc


def _occurrence_source(identity: ObjectIdentity) -> SourceRef:
    """从 occurrence 全键恢复来源，并拒绝截断布局。"""
    if identity.object_kind != OBJECT_OCCURRENCE or len(identity.components) < 4:
        raise W05TypedAdapterError("occurrence identity 类型或长度非法")
    try:
        return SourceRef.from_stable_key(identity.components[:-3])
    except (TypeError, ValueError) as exc:
        raise W05TypedAdapterError("occurrence SourceRef 非法") from exc


def _binding_predicate(kind: int) -> ObjectIdentity:
    """返回 W-05 候选图中的结构边类型，不承载自然语言角色意义。"""
    return concept_identity(
        (W05_NAMESPACE, 400, kind),
        versions=W05_IDENTITY_VERSIONS,
    )


@dataclass(frozen=True)
class W05OccurrenceBinding:
    """一个 span occurrence 与其 Entity/Event 候选的无损关联。"""

    identity: ObjectIdentity
    semantic_object: ObjectIdentity
    start: int
    end: int
    ordinal: int
    surface_fragment: str
    source: SourceRef

    def __post_init__(self) -> None:
        if self.identity != occurrence_identity(
                self.source,
                start=self.start,
                end=self.end,
                ordinal=self.ordinal):
            raise W05TypedAdapterError("occurrence identity 与 source/span/ordinal 漂移")
        if self.semantic_object.object_kind not in {OBJECT_ENTITY, OBJECT_EVENT}:
            raise W05TypedAdapterError("occurrence semantic object 必须是 Entity/Event")
        if semantic_source(self.semantic_object) != self.source:
            raise W05TypedAdapterError("occurrence 与 semantic object 来源不一致")


@dataclass(frozen=True)
class W05SourceBinding:
    """D-02 SourceRefRecord 与 payload 内一等 SourceRef 的只读桥。"""

    record: SourceRefRecord
    source_ref: SourceRef

    def __post_init__(self) -> None:
        if not isinstance(self.record, SourceRefRecord):
            raise TypeError("source binding record 类型非法")
        if not isinstance(self.source_ref, SourceRef):
            raise TypeError("source binding source_ref 类型非法")


@dataclass(frozen=True)
class W05AtomicPropositionCandidate:
    """一个未预选的 occurrence/RoleBinding/Proposition 候选信封。"""

    observation: ObservationRecord
    source_record: SourceRefRecord
    source_ref: SourceRef
    surface: str
    occurrences: tuple[W05OccurrenceBinding, ...]
    occurrence_order: tuple[ObjectIdentity, ...]
    proposition_definition: AtomicPropositionDefinition
    candidate: ObjectIdentity
    competition_key: tuple[int, ...]
    definition: EvidenceCandidateDefinition
    sample_role: str
    perturbation_kind: str
    selection_state: str
    supersedes_observation_key: StableRecordKey | None
    provenance: CanonicalJsonObject

    def __post_init__(self) -> None:
        if self.selection_state != W05_SELECTION_UNSELECTED:
            raise W05TypedAdapterError("atomic candidate 不得在 adapter 中预选")
        if self.candidate != self.proposition_definition.proposition:
            raise W05TypedAdapterError("candidate 与 Proposition 身份漂移")
        if self.definition.candidate != self.candidate:
            raise W05TypedAdapterError("EvidenceCandidateDefinition 身份漂移")
        if self.occurrence_order != tuple(
                item.identity for item in self.occurrences):
            raise W05TypedAdapterError("occurrence_order 未与 occurrence 信封同序")

    def role_binding_identities(self) -> tuple[ObjectIdentity, ...]:
        """返回当前命题的 canonical 一等 RoleBinding 身份。"""
        return tuple(
            item.identity_for(self.proposition_definition.proposition)
            for item in self.proposition_definition.canonical_bindings()
        )

    def semantic_objects(self) -> tuple[ObjectIdentity, ...]:
        """按 occurrence 顺序返回 Entity/Event 候选，不按 surface 折叠。"""
        return tuple(item.semantic_object for item in self.occurrences)


@dataclass(frozen=True)
class W05ObservationEnvelope:
    """一个 W-05 AtomicPropositionQuery 的全部 typed 内容。"""

    observation: ObservationRecord
    source: W05SourceBinding
    candidates: tuple[W05AtomicPropositionCandidate, ...]
    parser_provenance: CanonicalJsonObject


@dataclass(frozen=True)
class W05EvidenceBinding:
    """train TeacherEvidence 到 H-05 stance 的薄映射。"""

    teacher_record: TeacherEvidenceRecord
    observation: ObservationRecord
    source_ref: SourceRef
    candidates: tuple[ObjectIdentity, ...]
    stances: tuple[int, ...]
    decision: str
    expected_payload: CanonicalJsonObject
    perturbation_kind: str
    reason_key: tuple[int, ...]
    logical_order: int
    withdrawal_level: int
    supersedes_observation_key: StableRecordKey | None


@dataclass(frozen=True)
class W05TypedAdapterOutput:
    """W05-02 的纯确定性输出；不执行 transaction 或 learning write。"""

    source_bindings: tuple[W05SourceBinding, ...]
    observations: tuple[W05ObservationEnvelope, ...]
    candidates: tuple[W05AtomicPropositionCandidate, ...]
    evidence: tuple[W05EvidenceBinding, ...]
    execution_state: tuple[tuple[str, int], ...] = W05_ZERO_EXECUTION_STATE

    def candidates_for_perturbation(
            self,
            perturbation_kind: str,
            ) -> tuple[W05AtomicPropositionCandidate, ...]:
        """按课程显式 perturbation 身份返回候选，不解释 surface cue。"""
        return tuple(
            item for item in self.candidates
            if item.perturbation_kind == perturbation_kind
        )

    def candidates_for_surface(
            self,
            surface: str,
            ) -> tuple[W05AtomicPropositionCandidate, ...]:
        """返回同 surface 的全部来源化候选，不折叠 occurrence 身份。"""
        return tuple(item for item in self.candidates if item.surface == surface)


def _typed_payload(observation: ObservationRecord) -> dict[str, Any]:
    """读取并核验 AtomicPropositionQuery 顶层字段。"""
    value = observation.typed_payload.to_value()
    if not isinstance(value, dict) or set(value) != _PAYLOAD_FIELDS:
        raise W05TypedAdapterError("atomic Observation payload 字段集合漂移")
    if value.get("query_kind") != "occurrence_role_atomic_proposition":
        raise W05TypedAdapterError("AtomicPropositionQuery query_kind 漂移")
    if not isinstance(value.get("surface"), str) or not value["surface"]:
        raise W05TypedAdapterError("atomic surface 非法")
    if not isinstance(value.get("occurrences"), list):
        raise W05TypedAdapterError("atomic occurrences 必须是列表")
    if not isinstance(value.get("occurrence_order"), list):
        raise W05TypedAdapterError("atomic occurrence_order 必须是列表")
    candidate = value.get("candidate_definition")
    if not isinstance(candidate, dict) or set(candidate) != _CANDIDATE_FIELDS:
        raise W05TypedAdapterError("atomic candidate_definition 字段集合漂移")
    if not isinstance(candidate.get("role_bindings"), list):
        raise W05TypedAdapterError("atomic role_bindings 必须是列表")
    return value


def _adapt_occurrence(value: Any, *, surface: str) -> W05OccurrenceBinding:
    """恢复 occurrence 与 Entity/Event，并重算 source/span/ordinal 身份。"""
    if not isinstance(value, dict) or set(value) != _OCCURRENCE_FIELDS:
        raise W05TypedAdapterError("atomic occurrence 字段集合漂移")
    start = _strict_nonnegative(value["start"], where="occurrence.start")
    end = _strict_nonnegative(value["end"], where="occurrence.end")
    ordinal = _strict_nonnegative(
        value["ordinal"], where="occurrence.ordinal")
    fragment = value["surface_fragment"]
    if (end <= start or end > len(surface)
            or not isinstance(fragment, str) or not fragment
            or surface[start:end] != fragment):
        raise W05TypedAdapterError("occurrence span 与 surface_fragment 不一致")
    identity = _object_key(value["identity_key"], where="occurrence.identity_key")
    semantic = _object_key(value["semantic_key"], where="occurrence.semantic_key")
    semantic_kind = value["semantic_kind"]
    if type(semantic_kind) is not int or semantic_kind not in {
            OBJECT_ENTITY, OBJECT_EVENT}:
        raise W05TypedAdapterError("occurrence semantic_kind 非 Entity/Event")
    if semantic.object_kind != semantic_kind:
        raise W05TypedAdapterError("occurrence semantic_kind 与 identity 不一致")
    try:
        validate_semantic_identity(semantic)
    except (TypeError, ValueError) as exc:
        raise W05TypedAdapterError("occurrence semantic identity 非法") from exc
    source = _occurrence_source(identity)
    return W05OccurrenceBinding(
        identity,
        semantic,
        start,
        end,
        ordinal,
        fragment,
        source,
    )


def _adapt_proposition(
        value: dict[str, Any],
        occurrences: tuple[W05OccurrenceBinding, ...],
        ) -> AtomicPropositionDefinition:
    """恢复并交叉核验 Proposition、ContextScope 和 RoleBinding 全键。"""
    proposition = _object_key(value["proposition_key"], where="proposition_key")
    predicate = _object_key(value["predicate_key"], where="predicate_key")
    source_anchor = _object_key(
        value["source_anchor_key"], where="source_anchor_key")
    context = _object_key(value["context_key"], where="context_key")
    if proposition.object_kind != OBJECT_PROPOSITION:
        raise W05TypedAdapterError("candidate proposition_key 不是 Proposition")
    if context.object_kind != OBJECT_CONTEXT_SCOPE:
        raise W05TypedAdapterError("candidate context_key 不是 ContextScope")
    occurrence_ids = {item.identity for item in occurrences}
    semantic_objects = {item.semantic_object for item in occurrences}
    if source_anchor not in occurrence_ids:
        raise W05TypedAdapterError("source anchor 不属于 occurrence 集")
    if len({item.source for item in occurrences}) != 1:
        raise W05TypedAdapterError("同一 atomic query 的 occurrence 来源不唯一")
    source = occurrences[0].source
    try:
        if semantic_source(proposition) != source:
            raise W05TypedAdapterError("Proposition 与 occurrence 来源不一致")
        if semantic_source(context) != source:
            raise W05TypedAdapterError("ContextScope 与 occurrence 来源不一致")
        validate_semantic_identity(predicate)
    except (TypeError, ValueError) as exc:
        raise W05TypedAdapterError("candidate semantic identity 非法") from exc

    bindings = []
    binding_ids: set[ObjectIdentity] = set()
    for raw in value["role_bindings"]:
        if not isinstance(raw, dict) or set(raw) != _ROLE_BINDING_FIELDS:
            raise W05TypedAdapterError("atomic role binding 字段集合漂移")
        role = _object_key(raw["role_key"], where="role_binding.role_key")
        filler = _object_key(raw["filler_key"], where="role_binding.filler_key")
        binding_identity = _object_key(
            raw["binding_key"], where="role_binding.binding_key")
        ordinal = _strict_nonnegative(
            raw["ordinal"], where="role_binding.ordinal")
        if role.object_kind != OBJECT_ROLE:
            raise W05TypedAdapterError("role binding role_key 不是 Role")
        if filler not in semantic_objects:
            raise W05TypedAdapterError("Role filler 不属于 occurrence semantic object 集")
        binding = AtomicRoleBinding(role, filler, ordinal)
        if (binding_identity.object_kind != OBJECT_ROLE_BINDING
                or binding.identity_for(proposition) != binding_identity):
            raise W05TypedAdapterError("RoleBinding identity 与 role/filler/ordinal 漂移")
        if binding_identity in binding_ids:
            raise W05TypedAdapterError("RoleBinding identity 重复")
        binding_ids.add(binding_identity)
        bindings.append(binding)
    if not bindings:
        raise W05TypedAdapterError("atomic proposition 缺 RoleBinding")
    try:
        return AtomicPropositionDefinition(
            proposition,
            predicate,
            source_anchor,
            context,
            tuple(bindings),
        )
    except (TypeError, ValueError) as exc:
        raise W05TypedAdapterError("AtomicPropositionDefinition 非法") from exc


def _candidate_definition(
        observation: ObservationRecord,
        proposition: AtomicPropositionDefinition,
        occurrences: tuple[W05OccurrenceBinding, ...],
        competition_root: StableRecordKey,
        ) -> EvidenceCandidateDefinition:
    """把一等命题结构接入现役 H-05 候选定义。"""
    values: list[CandidateBinding] = [
        CandidateBinding(
            _binding_predicate(1),
            proposition.predicate,
            0,
            CANDIDATE_AS_SUBJECT,
        ),
        CandidateBinding(
            _binding_predicate(2),
            proposition.source_anchor,
            0,
            CANDIDATE_AS_SUBJECT,
        ),
        CandidateBinding(
            _binding_predicate(3),
            proposition.context,
            0,
            CANDIDATE_AS_SUBJECT,
        ),
    ]
    for ordinal, binding in enumerate(proposition.canonical_bindings()):
        values.append(CandidateBinding(
            _binding_predicate(4),
            binding.identity_for(proposition.proposition),
            ordinal,
            CANDIDATE_AS_SUBJECT,
        ))
    for ordinal, occurrence in enumerate(occurrences):
        values.extend((
            CandidateBinding(
                _binding_predicate(5),
                occurrence.identity,
                ordinal,
                CANDIDATE_AS_SUBJECT,
            ),
            CandidateBinding(
                _binding_predicate(6),
                occurrence.semantic_object,
                ordinal,
                CANDIDATE_AS_SUBJECT,
            ),
        ))
    competition = (
        W05_NAMESPACE,
        100,
        *_pack(competition_root.stable_key()),
    )
    return EvidenceCandidateDefinition(
        proposition.proposition,
        competition,
        tuple(values),
        (proposition.source,),
    )


def _adapt_candidate(
        observation: ObservationRecord,
        source_record: SourceRefRecord,
        competition_root: StableRecordKey,
        ) -> W05AtomicPropositionCandidate:
    """把一个 train AtomicPropositionQuery 映射为未选择候选。"""
    value = _typed_payload(observation)
    surface = value["surface"]
    occurrences = tuple(
        _adapt_occurrence(item, surface=surface)
        for item in value["occurrences"]
    )
    if not occurrences:
        raise W05TypedAdapterError("AtomicPropositionQuery 缺 occurrence")
    occurrence_ids = tuple(item.identity for item in occurrences)
    if len(set(occurrence_ids)) != len(occurrence_ids):
        raise W05TypedAdapterError("occurrence identity 重复")
    order = tuple(
        _object_key(item, where="occurrence_order item")
        for item in value["occurrence_order"]
    )
    if (len(order) != len(occurrence_ids)
            or len(set(order)) != len(order)
            or set(order) != set(occurrence_ids)
            or order != occurrence_ids):
        raise W05TypedAdapterError("occurrence_order 必须唯一、完整且与 payload 同序")
    proposition = _adapt_proposition(value["candidate_definition"], occurrences)
    definition = _candidate_definition(
        observation,
        proposition,
        occurrences,
        competition_root,
    )
    return W05AtomicPropositionCandidate(
        observation,
        source_record,
        proposition.source,
        surface,
        occurrences,
        order,
        proposition,
        proposition.proposition,
        definition.competition_key,
        definition,
        observation.sample_role,
        observation.perturbation_kind,
        W05_SELECTION_UNSELECTED,
        observation.supersedes_key,
        CanonicalJsonObject.from_value({
            "adapter": "ph2_w05_occurrence_role_atomic_v1",
            "payload_kind": observation.payload_kind,
            "query_kind": value["query_kind"],
            "sample_role": observation.sample_role,
        }),
    )


def _teacher_stances(teacher: TeacherEvidenceRecord) -> tuple[int, ...]:
    """把 train teacher 四态映射到 H-05 stance。"""
    value = teacher.typed_evidence.to_value()
    state = value.get("expected_state")
    mapping = {
        "TRUE": (EVIDENCE_SUPPORT,),
        "FALSE": (EVIDENCE_REFUTE,),
        "UNKNOWN": (EVIDENCE_UNKNOWN,),
        "CONFLICT": (EVIDENCE_SUPPORT, EVIDENCE_REFUTE),
    }
    result = mapping.get(state)
    if result is None:
        raise W05TypedAdapterError("teacher Evidence 缺合法 expected_state")
    return result


def _evidence_binding(
        envelope: W05ObservationEnvelope,
        teacher: TeacherEvidenceRecord,
        ) -> W05EvidenceBinding:
    """把独立 train teacher Evidence 绑定到原子命题候选。"""
    observation = envelope.observation
    value = teacher.typed_evidence.to_value()
    if (teacher.observation_key != observation.stable_key
            or teacher.source_ref_key != observation.source_ref_key
            or teacher.visible_from_stage != "W-05"):
        raise W05TypedAdapterError("teacher Evidence 与 W-05 Observation 不一致")
    if not isinstance(value, dict) or set(value) != _TEACHER_FIELDS:
        raise W05TypedAdapterError("atomic teacher Evidence 字段集合漂移")
    expected_payload = value.get("expected_payload")
    if not isinstance(expected_payload, dict):
        raise W05TypedAdapterError("teacher expected_payload 非 object")
    decision = expected_payload.get("decision")
    if not isinstance(decision, str) or not decision:
        raise W05TypedAdapterError("teacher expected_payload 缺 decision")
    if observation.perturbation_kind not in ALLOWED_PERTURBATIONS:
        raise W05TypedAdapterError("atomic perturbation kind 未注册")
    candidate = envelope.candidates[0]
    reason_key = (
        W05_NAMESPACE,
        1,
        *_pack(teacher.stable_key.stable_key()),
        *_pack(_text_key(teacher.evidence_kind)),
        *_pack(_text_key(observation.perturbation_kind)),
        *_pack(_text_key(decision)),
        teacher.withdrawal_level,
    )
    return W05EvidenceBinding(
        teacher,
        observation,
        candidate.source_ref,
        (candidate.candidate,),
        _teacher_stances(teacher),
        decision,
        CanonicalJsonObject.from_value(expected_payload),
        observation.perturbation_kind,
        reason_key,
        observation.logical_order,
        teacher.withdrawal_level,
        observation.supersedes_key,
    )


def adapt_w05_training_payload(
        payload: W05TrainingPayload,
        ) -> W05TypedAdapterOutput:
    """只适配 W-05 AtomicPropositionQuery，保留其余 train pack 不消费。"""
    if not isinstance(payload, W05TrainingPayload):
        raise TypeError("W-05 adapter 只接受 W05TrainingPayload")
    source_records = tuple(sorted(
        payload.source_refs,
        key=lambda item: item.stable_key.stable_key(),
    ))
    observations = tuple(sorted(
        payload.observations,
        key=lambda item: item.stable_key.stable_key(),
    ))
    teachers = tuple(sorted(
        payload.teacher_evidence,
        key=lambda item: item.stable_key.stable_key(),
    ))
    if (len({item.stable_key for item in source_records}) != len(source_records)
            or len({item.stable_key for item in observations}) != len(observations)
            or len({item.stable_key for item in teachers}) != len(teachers)):
        raise W05TypedAdapterError("train records 存在重复 stable key")
    source_by_key = {item.stable_key: item for item in source_records}
    teacher_by_observation: dict[StableRecordKey, TeacherEvidenceRecord] = {}
    for teacher in teachers:
        if teacher.observation_key in teacher_by_observation:
            raise W05TypedAdapterError("同一 Observation 有多个 teacher Evidence")
        teacher_by_observation[teacher.observation_key] = teacher

    atomic_observations = {
        item.stable_key: item for item in observations
        if item.w_stage == "W-05"
        and item.payload_kind == "AtomicPropositionQuery"
    }

    def competition_root(observation: ObservationRecord) -> StableRecordKey:
        """沿 append-only supersedes 链解析 lifecycle competition 根。"""
        current = observation
        seen: set[StableRecordKey] = set()
        while current.supersedes_key is not None:
            if current.stable_key in seen:
                raise W05TypedAdapterError("atomic supersedes 链成环")
            seen.add(current.stable_key)
            prior = atomic_observations.get(current.supersedes_key)
            if prior is None or prior.logical_order >= current.logical_order:
                raise W05TypedAdapterError("atomic supersedes target 缺失或不早于 replacement")
            current = prior
        return current.stable_key

    envelopes: list[W05ObservationEnvelope] = []
    candidates: list[W05AtomicPropositionCandidate] = []
    evidence: list[W05EvidenceBinding] = []
    source_bindings: dict[StableRecordKey, W05SourceBinding] = {}
    for observation in observations:
        if (observation.w_stage != "W-05"
                or observation.payload_kind != "AtomicPropositionQuery"):
            continue
        source_record = source_by_key.get(observation.source_ref_key)
        if (source_record is None
                or source_record.license_id != observation.license_partition):
            raise W05TypedAdapterError("Atomic Observation 来源或许可漂移")
        teacher = teacher_by_observation.get(observation.stable_key)
        if teacher is None:
            raise W05TypedAdapterError("Atomic Observation 缺 train TeacherEvidence")
        candidate = _adapt_candidate(
            observation,
            source_record,
            competition_root(observation),
        )
        source_binding = W05SourceBinding(source_record, candidate.source_ref)
        prior = source_bindings.get(source_record.stable_key)
        if prior is not None and prior != source_binding:
            raise W05TypedAdapterError("SourceRefRecord 绑定多个 payload SourceRef")
        source_bindings[source_record.stable_key] = source_binding
        envelope = W05ObservationEnvelope(
            observation,
            source_binding,
            (candidate,),
            CanonicalJsonObject.from_value({
                "adapter": "ph2_w05_occurrence_role_atomic_v1",
                "payload_kind": observation.payload_kind,
            }),
        )
        envelopes.append(envelope)
        candidates.append(candidate)
        evidence.append(_evidence_binding(envelope, teacher))
    if not envelopes:
        raise W05TypedAdapterError("W-05 train payload 缺 AtomicPropositionQuery")
    candidate_ids = [item.candidate for item in candidates]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise W05TypedAdapterError("W-05 atomic candidate identity 重复")
    return W05TypedAdapterOutput(
        tuple(sorted(
            source_bindings.values(),
            key=lambda item: item.record.stable_key.stable_key(),
        )),
        tuple(sorted(
            envelopes,
            key=lambda item: item.observation.stable_key.stable_key(),
        )),
        tuple(sorted(candidates, key=lambda item: item.definition.stable_key())),
        tuple(sorted(evidence, key=lambda item: item.logical_order)),
    )


__all__ = [
    "W05AtomicPropositionCandidate",
    "W05EvidenceBinding",
    "W05ObservationEnvelope",
    "W05OccurrenceBinding",
    "W05SourceBinding",
    "W05TypedAdapterError",
    "W05TypedAdapterOutput",
    "W05_ATOMIC_IDENTITY_VERSIONS",
    "W05_IDENTITY_VERSIONS",
    "W05_NAMESPACE",
    "adapt_w05_training_payload",
]
