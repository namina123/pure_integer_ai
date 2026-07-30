"""PH2 W-03 train Observation 到 typed Sense 候选的纯适配层。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.evidence_candidate import (
    CANDIDATE_AS_OBJECT,
    CANDIDATE_AS_SUBJECT,
    EvidenceCandidateDefinition,
)
from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EVIDENCE_UNKNOWN,
)
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    OBJECT_CONCEPT,
    OBJECT_CONTEXT_SCOPE,
    OBJECT_LANGUAGE_ATOM,
    OBJECT_SENSE,
    CorpusVersion,
    CurriculumVersion,
    ObjectIdentity,
    ParserVersion,
    PrimitiveVersion,
    SourceRef,
    VersionBundle,
    concept_identity,
    language_atom_identity,
    language_branch_identity,
    representation_identity,
    sense_identity,
    span_identity,
)
from pure_integer_ai.cognition.shared.semantic_object import (
    context_scope_identity,
)
from pure_integer_ai.cognition.understanding.language_candidate import (
    CandidateFieldProtocol,
    SenseCandidateProtocol,
    SenseCandidateSpec,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    ObservationRecord,
    SourceRefRecord,
    StableRecordKey,
    TeacherEvidenceRecord,
)
from pure_integer_ai.experiments.ph2_w03_adapter_extractors import (
    W03ExtractedCandidate,
    W03ExtractedObservation,
    W03SurfaceAnchor,
    extract_w03_observations,
)
from pure_integer_ai.experiments.ph2_w03_payload import W03TrainingPayload


W03_SOURCE_REF_KIND = 30302
W03_REPRESENTATION_FAMILY = (30302, 1)
W03_SENSE_HYPOTHESIS_KIND = (30302, 2)
W03_SELECTION_UNSELECTED = "UNSELECTED"
W03_IDENTITY_VERSIONS = VersionBundle(
    CorpusVersion(3),
    ParserVersion(1),
    PrimitiveVersion(1),
    CurriculumVersion(3),
)
W03_CONTEXT_SOURCE = SourceRef(
    W03_SOURCE_REF_KIND,
    1,
    0,
    GLOBAL_OWNER_SCOPE,
    W03_IDENTITY_VERSIONS,
)
W03_ZERO_EXECUTION_STATE = (
    ("LANGUAGE_CAPABILITY_MASTERED", 0),
    ("LANGUAGE_READINESS", 0),
    ("W03_STARTED", 0),
    ("W04_STARTED", 0),
    ("formal_w03_training_runs", 0),
    ("learning_writes", 0),
    ("teacher_calls", 0),
)


class W03TypedAdapterError(ValueError):
    """W-03 record identity、teacher 绑定或 typed 映射非法。"""


def _packed(value: tuple[int, ...]) -> tuple[int, ...]:
    return len(value), *value


def _text_key(value: str) -> tuple[int, ...]:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise W03TypedAdapterError("adapter text key 非法")
    return len(value), *(ord(character) for character in value)


def _versions(source: SourceRefRecord) -> VersionBundle:
    return VersionBundle(
        CorpusVersion(source.course_version),
        ParserVersion(source.parser_version),
        PrimitiveVersion(source.schema_version),
        CurriculumVersion(source.course_version),
    )


def _source_ref(source: SourceRefRecord) -> SourceRef:
    return SourceRef(
        W03_SOURCE_REF_KIND,
        source.stable_key.components[-1],
        source.record_ordinal,
        GLOBAL_OWNER_SCOPE,
        _versions(source),
    )


def _sense_protocol() -> SenseCandidateProtocol:
    atom_predicate = concept_identity((30302, 10))
    concept_predicate = concept_identity((30302, 11))
    context_predicate = concept_identity((30302, 12))
    return SenseCandidateProtocol(
        (OBJECT_SENSE,),
        CandidateFieldProtocol(
            atom_predicate,
            CANDIDATE_AS_OBJECT,
            0,
            (OBJECT_LANGUAGE_ATOM,),
        ),
        CandidateFieldProtocol(
            concept_predicate,
            CANDIDATE_AS_SUBJECT,
            0,
            (OBJECT_CONCEPT,),
        ),
        CandidateFieldProtocol(
            context_predicate,
            CANDIDATE_AS_SUBJECT,
            0,
            (OBJECT_CONTEXT_SCOPE,),
        ),
    )


@dataclass(frozen=True)
class W03SourceBinding:
    """原 SourceRefRecord 与 cognition SourceRef 的无损桥。"""

    record: SourceRefRecord
    source_ref: SourceRef

    def __post_init__(self) -> None:
        if not isinstance(self.record, SourceRefRecord):
            raise TypeError("source binding record 类型非法")
        if not isinstance(self.source_ref, SourceRef):
            raise TypeError("source binding source_ref 类型非法")
        if self.source_ref != _source_ref(self.record):
            raise W03TypedAdapterError("SourceRef typed identity 漂移")


@dataclass(frozen=True)
class W03AdaptedAnchor:
    """表层、LanguageBranch、Representation 与来源 Span 的独立身份。"""

    extracted: W03SurfaceAnchor
    branch: ObjectIdentity
    atom: ObjectIdentity
    representation: ObjectIdentity
    span: ObjectIdentity


@dataclass(frozen=True)
class W03SenseCandidateEnvelope:
    """保留原 observation 且不携带 teacher expected/label 的候选信封。"""

    observation: ObservationRecord
    source_record: SourceRefRecord
    source_ref: SourceRef
    anchor: W03AdaptedAnchor
    sense: ObjectIdentity
    concept: ObjectIdentity
    context: ObjectIdentity
    competition_key: tuple[int, ...]
    spec: SenseCandidateSpec
    definition: EvidenceCandidateDefinition
    candidate_kind: str
    selection_state: str
    external_nondefinitive: bool
    lexicalized_multiword: bool
    supersedes_observation_key: StableRecordKey | None
    prerequisite_keys: tuple[StableRecordKey, ...]
    provenance: CanonicalJsonObject

    def __post_init__(self) -> None:
        if self.selection_state != W03_SELECTION_UNSELECTED:
            raise W03TypedAdapterError("Sense candidate 不得在 adapter 中预选")
        if self.definition != self.spec.definition(_sense_protocol()):
            raise W03TypedAdapterError("Sense candidate definition 漂移")


@dataclass(frozen=True)
class W03ObservationEnvelope:
    """一个 W-03 Observation 的全部 typed anchors 与候选集合。"""

    observation: ObservationRecord
    source: W03SourceBinding
    anchors: tuple[W03AdaptedAnchor, ...]
    candidates: tuple[W03SenseCandidateEnvelope, ...]
    parser_provenance: CanonicalJsonObject
    external_nondefinitive: bool


@dataclass(frozen=True)
class W03EvidenceBinding:
    """teacher Evidence 到现有 Evidence stance 的薄映射，不复制 label。"""

    teacher_record: TeacherEvidenceRecord
    observation: ObservationRecord
    source_ref: SourceRef
    candidates: tuple[ObjectIdentity, ...]
    stances: tuple[int, ...]
    reason_key: tuple[int, ...]
    logical_order: int
    withdrawal_level: int
    external_nondefinitive: bool
    supersedes_observation_key: StableRecordKey | None
    prerequisite_keys: tuple[StableRecordKey, ...]


@dataclass(frozen=True)
class W03TypedAdapterOutput:
    """W03-02 的确定性纯输出；没有 transaction 或 learning write。"""

    sense_protocol: SenseCandidateProtocol
    source_bindings: tuple[W03SourceBinding, ...]
    retained_w02_observations: tuple[ObservationRecord, ...]
    retained_w02_teacher_evidence: tuple[TeacherEvidenceRecord, ...]
    observations: tuple[W03ObservationEnvelope, ...]
    candidates: tuple[W03SenseCandidateEnvelope, ...]
    evidence: tuple[W03EvidenceBinding, ...]
    execution_state: tuple[tuple[str, int], ...] = W03_ZERO_EXECUTION_STATE

    def candidates_for_atom(
            self, atom: ObjectIdentity,
            ) -> tuple[W03SenseCandidateEnvelope, ...]:
        """返回同 atom 的完整未选择候选集，不按排序私选首项。"""
        return tuple(
            item for item in self.candidates if item.anchor.atom == atom)


def _adapt_anchor(
        extracted: W03SurfaceAnchor,
        source: SourceRef,
        ) -> W03AdaptedAnchor:
    versions = W03_IDENTITY_VERSIONS
    branch = language_branch_identity(
        (30302, *_text_key(extracted.branch_language)),
        versions=versions,
    )
    representation = representation_identity(
        (*W03_REPRESENTATION_FAMILY,
         *_text_key(extracted.branch_language)),
        tuple(ord(character) for character in extracted.surface),
        versions=versions,
    )
    atom = language_atom_identity(
        branch,
        (30302, *_packed(representation.stable_key())),
    )
    span = span_identity(
        source,
        members=extracted.members,
        ordinal=extracted.ordinal,
    )
    return W03AdaptedAnchor(
        extracted, branch, atom, representation, span)


def _competition_key(
        atom: ObjectIdentity,
        context: ObjectIdentity,
        extracted: W03ExtractedCandidate,
        ) -> tuple[int, ...]:
    return (
        1,
        *_packed(atom.stable_key()),
        *_packed(context.stable_key()),
        *_packed(extracted.competition_context_key),
    )


def _adapt_candidate(
        extracted_observation: W03ExtractedObservation,
        extracted: W03ExtractedCandidate,
        source_binding: W03SourceBinding,
        anchors: tuple[W03AdaptedAnchor, ...],
        protocol: SenseCandidateProtocol,
        ) -> W03SenseCandidateEnvelope:
    try:
        anchor = anchors[extracted.anchor_ordinal]
    except IndexError as exc:
        raise W03TypedAdapterError("candidate anchor ordinal 越界") from exc
    versions = W03_IDENTITY_VERSIONS
    sense = sense_identity(
        source_binding.source_ref,
        sense_key=extracted.sense_key,
    )
    concept = concept_identity(extracted.concept_key, versions=versions)
    context = context_scope_identity(
        W03_CONTEXT_SOURCE,
        extracted.context_key,
    )
    competition = _competition_key(anchor.atom, context, extracted)
    spec = SenseCandidateSpec(
        sense,
        competition,
        anchor.atom,
        concept,
        context,
        (source_binding.source_ref,),
    )
    definition = spec.definition(protocol)
    observation = extracted_observation.observation
    return W03SenseCandidateEnvelope(
        observation,
        source_binding.record,
        source_binding.source_ref,
        anchor,
        sense,
        concept,
        context,
        competition,
        spec,
        definition,
        extracted.candidate_kind,
        W03_SELECTION_UNSELECTED,
        extracted.external_nondefinitive,
        extracted.lexicalized_multiword,
        observation.supersedes_key,
        observation.prerequisite_keys,
        extracted.provenance,
    )


def _teacher_stances(
        teacher: TeacherEvidenceRecord,
        *,
        external_nondefinitive: bool,
        ) -> tuple[int, ...]:
    if external_nondefinitive:
        return (EVIDENCE_UNKNOWN,)
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
        raise W03TypedAdapterError("teacher Evidence 缺合法 expected_state")
    return result


def _evidence_binding(
        envelope: W03ObservationEnvelope,
        teacher: TeacherEvidenceRecord,
        ) -> W03EvidenceBinding:
    observation = envelope.observation
    if (teacher.observation_key != observation.stable_key
            or teacher.source_ref_key != observation.source_ref_key
            or teacher.visible_from_stage != "W-03"):
        raise W03TypedAdapterError("teacher Evidence 与 W-03 Observation 不一致")
    reason_key = (
        1,
        *teacher.stable_key.stable_key(),
        *_text_key(teacher.evidence_kind),
        teacher.withdrawal_level,
    )
    return W03EvidenceBinding(
        teacher,
        observation,
        envelope.source.source_ref,
        tuple(item.sense for item in envelope.candidates),
        _teacher_stances(
            teacher,
            external_nondefinitive=envelope.external_nondefinitive,
        ),
        reason_key,
        observation.logical_order,
        teacher.withdrawal_level,
        envelope.external_nondefinitive,
        observation.supersedes_key,
        observation.prerequisite_keys,
    )


def adapt_w03_training_payload(
        payload: W03TrainingPayload,
        ) -> W03TypedAdapterOutput:
    """把 firewall 的原子 train 交付纯映射为 W03 typed 候选和 Evidence。"""
    if not isinstance(payload, W03TrainingPayload):
        raise TypeError("W-03 adapter 只接受 W03TrainingPayload")
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
        raise W03TypedAdapterError("train records 存在重复 stable key")
    source_by_key = {item.stable_key: item for item in source_records}
    teacher_by_observation: dict[StableRecordKey, TeacherEvidenceRecord] = {}
    for teacher in teachers:
        if teacher.observation_key in teacher_by_observation:
            raise W03TypedAdapterError("同一 Observation 有多个 teacher Evidence")
        teacher_by_observation[teacher.observation_key] = teacher
    for observation in observations:
        source = source_by_key.get(observation.source_ref_key)
        if source is None:
            raise W03TypedAdapterError("Observation 引用未知 SourceRef")
        if source.license_id != observation.license_partition:
            raise W03TypedAdapterError("Observation license 与 SourceRef 漂移")
        if observation.stable_key not in teacher_by_observation:
            raise W03TypedAdapterError("Observation 缺 teacher Evidence")

    source_bindings = tuple(
        W03SourceBinding(item, _source_ref(item)) for item in source_records)
    source_binding_by_key = {
        item.record.stable_key: item for item in source_bindings}
    extracted = extract_w03_observations(observations)
    protocol = _sense_protocol()
    observation_envelopes: list[W03ObservationEnvelope] = []
    candidate_envelopes: list[W03SenseCandidateEnvelope] = []
    evidence_bindings: list[W03EvidenceBinding] = []
    for item in extracted:
        source_binding = source_binding_by_key[item.observation.source_ref_key]
        anchors = tuple(
            _adapt_anchor(anchor, source_binding.source_ref)
            for anchor in item.anchors
        )
        candidates = tuple(
            _adapt_candidate(item, candidate, source_binding, anchors, protocol)
            for candidate in item.candidates
        )
        envelope = W03ObservationEnvelope(
            item.observation,
            source_binding,
            anchors,
            candidates,
            item.parser_provenance,
            item.external_nondefinitive,
        )
        observation_envelopes.append(envelope)
        candidate_envelopes.extend(candidates)
        evidence_bindings.append(_evidence_binding(
            envelope,
            teacher_by_observation[item.observation.stable_key],
        ))

    retained_w02 = tuple(
        item for item in observations if item.w_stage == "W-02")
    retained_w02_keys = {item.stable_key for item in retained_w02}
    retained_w02_teacher = tuple(
        item for item in teachers if item.observation_key in retained_w02_keys)
    return W03TypedAdapterOutput(
        protocol,
        source_bindings,
        retained_w02,
        retained_w02_teacher,
        tuple(observation_envelopes),
        tuple(sorted(
            candidate_envelopes,
            key=lambda item: item.definition.stable_key(),
        )),
        tuple(sorted(
            evidence_bindings,
            key=lambda item: item.observation.stable_key.stable_key(),
        )),
    )


__all__ = [
    "W03AdaptedAnchor",
    "W03EvidenceBinding",
    "W03ObservationEnvelope",
    "W03SenseCandidateEnvelope",
    "W03SourceBinding",
    "W03TypedAdapterError",
    "W03TypedAdapterOutput",
    "adapt_w03_training_payload",
]
