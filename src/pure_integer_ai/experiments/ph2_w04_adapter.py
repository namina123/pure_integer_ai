"""PH2 W-04 train Observation 到 primitive/surface 候选的纯适配层。"""
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
    OBJECT_CONCEPT,
    OBJECT_CONTEXT_SCOPE,
    OBJECT_LANGUAGE_ATOM,
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
)
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.cognition.shared.semantic_object import context_scope_identity
from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    ObservationRecord,
    SourceRefRecord,
    StableRecordKey,
    TeacherEvidenceRecord,
)
from pure_integer_ai.experiments.ph2_w04_payload import W04TrainingPayload


W04_NAMESPACE = 40404
W04_SOURCE_REF_KIND = 40402
W04_REPRESENTATION_FAMILY = (40402, 1)
W04_PRIMITIVE_SURFACE_HYPOTHESIS_KIND = (40402, 2)
W04_SELECTION_UNSELECTED = "UNSELECTED"
W04_IDENTITY_VERSIONS = VersionBundle(
    CorpusVersion(4),
    ParserVersion(1),
    PrimitiveVersion(1),
    CurriculumVersion(4),
)
W04_CONTEXT_SOURCE = SourceRef(
    W04_SOURCE_REF_KIND,
    1,
    0,
    GLOBAL_OWNER_SCOPE,
    W04_IDENTITY_VERSIONS,
)
W04_ZERO_EXECUTION_STATE = (
    ("LANGUAGE_CAPABILITY_MASTERED", 0),
    ("LANGUAGE_READINESS", 0),
    ("W04_STARTED", 0),
    ("W05_STARTED", 0),
    ("formal_w04_training_runs", 0),
    ("learning_writes", 0),
    ("teacher_calls", 0),
)
PRIMITIVE_REGISTRY_CODES = {
    "relation": 1,
    "operator": 2,
    "symbol_type": 3,
}
PRIMITIVE_DECISION_CODES = {
    "bind": 1,
    "reject_mismatch": 2,
    "keep_ambiguous": 3,
    "insufficient_for_binding": 4,
}


class W04TypedAdapterError(ValueError):
    """W-04 record identity、teacher 绑定或 typed primitive 映射非法。"""


def _pack(value: tuple[int, ...]) -> tuple[int, ...]:
    """给稳定整数键增加长度前缀。"""
    return len(value), *value


def _text_key(value: str) -> tuple[int, ...]:
    """把已在数据 pack 中出现的文本转成身份键，不解释为 cue 规则。"""
    if not isinstance(value, str) or not value or value.strip() != value:
        raise W04TypedAdapterError("adapter text key 非法")
    return len(value), *(ord(character) for character in value)


def _versions(source: SourceRefRecord) -> VersionBundle:
    """按 SourceRefRecord 恢复 PH2 数据版本。"""
    return VersionBundle(
        CorpusVersion(source.course_version),
        ParserVersion(source.parser_version),
        PrimitiveVersion(source.schema_version),
        CurriculumVersion(source.course_version),
    )


def _source_ref(source: SourceRefRecord) -> SourceRef:
    """把 SourceRefRecord 投影为 cognition SourceRef。"""
    return SourceRef(
        W04_SOURCE_REF_KIND,
        source.stable_key.components[-1],
        source.record_ordinal,
        GLOBAL_OWNER_SCOPE,
        _versions(source),
    )


def _primitive_object(registry: str, kind: int) -> ObjectIdentity:
    """把冻结 primitive registry/kind 坐标映射为一等 Concept。"""
    try:
        registry_code = PRIMITIVE_REGISTRY_CODES[registry]
    except KeyError as exc:
        raise W04TypedAdapterError("primitive registry 未注册") from exc
    if type(kind) is not int or kind <= 0:
        raise W04TypedAdapterError("primitive kind 必须是正严格整数")
    return concept_identity(
        (W04_NAMESPACE, 100, registry_code, kind),
        versions=W04_IDENTITY_VERSIONS,
    )


def _candidate_object(
        observation: ObservationRecord,
        registry: str,
        kind: int,
        surface: str,
        context: str,
        ) -> ObjectIdentity:
    """从 observation 与 primitive/surface/context 生成候选对象身份。"""
    return concept_identity(
        (
            W04_NAMESPACE,
            200,
            *_pack(observation.stable_key.stable_key()),
            *_pack(_text_key(registry)),
            kind,
            *_pack(_text_key(surface)),
            *_pack(_text_key(context)),
        ),
        versions=W04_IDENTITY_VERSIONS,
    )


def _context_object(context: str) -> ObjectIdentity:
    """把数据中的 context 文本转成 W-04 查询 scope 身份。"""
    return context_scope_identity(W04_CONTEXT_SOURCE, (
        W04_NAMESPACE,
        300,
        *_text_key(context),
    ))


def _surface_atom(surface: str) -> ObjectIdentity:
    """把 surface 作为 LanguageAtom 身份，不绑定任何 primitive label。"""
    representation = representation_identity(
        (*W04_REPRESENTATION_FAMILY, *_text_key("zh")),
        tuple(ord(character) for character in surface),
        versions=W04_IDENTITY_VERSIONS,
    )
    branch = language_branch_identity(
        (W04_NAMESPACE, 301, *_text_key("zh")),
        versions=W04_IDENTITY_VERSIONS,
    )
    return language_atom_identity(
        branch,
        (W04_NAMESPACE, 302, *_pack(representation.stable_key())),
    )


def _competition_key(
        surface: ObjectIdentity,
        context: ObjectIdentity,
        primitive: ObjectIdentity,
        ) -> tuple[int, ...]:
    """同 surface/context 下 primitive 竞争，保留同原语多 surface 的独立候选。"""
    return (
        1,
        *_pack(surface.stable_key()),
        *_pack(context.stable_key()),
        *_pack(primitive.stable_key()),
    )


def _source_by_key(
        source_refs: tuple[SourceRefRecord, ...],
        ) -> dict[StableRecordKey, SourceRefRecord]:
    """建立 SourceRefRecord 唯一索引。"""
    result = {item.stable_key: item for item in source_refs}
    if len(result) != len(source_refs):
        raise W04TypedAdapterError("SourceRef stable_key 重复")
    return result


def _typed_payload(observation: ObservationRecord) -> dict[str, Any]:
    """读取 primitive-surface Observation payload 并核字段。"""
    value = observation.typed_payload.to_value()
    if not isinstance(value, dict):
        raise W04TypedAdapterError("primitive Observation payload 非 object")
    if value.get("query_kind") != "primitive_surface_mapping":
        raise W04TypedAdapterError("W-04 adapter 只处理 primitive surface query")
    primitive = value.get("candidate_primitive")
    if not isinstance(primitive, dict):
        raise W04TypedAdapterError("candidate_primitive 缺失")
    registry = primitive.get("registry")
    kind = primitive.get("kind")
    surface = value.get("surface_form")
    context = value.get("context")
    if (not isinstance(registry, str) or not isinstance(surface, str)
            or not isinstance(context, str) or type(kind) is not int):
        raise W04TypedAdapterError("primitive payload 字段类型非法")
    return value


@dataclass(frozen=True)
class W04SourceBinding:
    """原 SourceRefRecord 与 cognition SourceRef 的无损桥。"""

    record: SourceRefRecord
    source_ref: SourceRef

    def __post_init__(self) -> None:
        if not isinstance(self.record, SourceRefRecord):
            raise TypeError("source binding record 类型非法")
        if not isinstance(self.source_ref, SourceRef):
            raise TypeError("source binding source_ref 类型非法")
        if self.source_ref != _source_ref(self.record):
            raise W04TypedAdapterError("SourceRef typed identity 漂移")


@dataclass(frozen=True)
class W04PrimitiveSurfaceCandidate:
    """保留 observation、surface、context 和 primitive 坐标的候选信封。"""

    observation: ObservationRecord
    source_record: SourceRefRecord
    source_ref: SourceRef
    surface_form: str
    context_text: str
    surface_atom: ObjectIdentity
    context: ObjectIdentity
    primitive: ObjectIdentity
    primitive_registry: str
    primitive_kind: int
    candidate: ObjectIdentity
    competition_key: tuple[int, ...]
    definition: EvidenceCandidateDefinition
    sample_role: str
    perturbation_kind: str
    selection_state: str
    supersedes_observation_key: StableRecordKey | None
    provenance: CanonicalJsonObject

    def __post_init__(self) -> None:
        if self.selection_state != W04_SELECTION_UNSELECTED:
            raise W04TypedAdapterError("Primitive candidate 不得在 adapter 中预选")
        if self.definition.candidate != self.candidate:
            raise W04TypedAdapterError("Primitive candidate definition 漂移")

    def coordinate_key(self) -> tuple[str, int]:
        """返回外部可读的 primitive 坐标键。"""
        return self.primitive_registry, self.primitive_kind


@dataclass(frozen=True)
class W04ObservationEnvelope:
    """一个 W-04 Observation 的全部 typed primitive/surface 候选。"""

    observation: ObservationRecord
    source: W04SourceBinding
    candidates: tuple[W04PrimitiveSurfaceCandidate, ...]
    parser_provenance: CanonicalJsonObject


@dataclass(frozen=True)
class W04EvidenceBinding:
    """teacher Evidence 到 H-05 stance 的薄映射，不复制 evaluator label。"""

    teacher_record: TeacherEvidenceRecord
    observation: ObservationRecord
    source_ref: SourceRef
    candidates: tuple[ObjectIdentity, ...]
    stances: tuple[int, ...]
    decision: str
    reason_key: tuple[int, ...]
    logical_order: int
    withdrawal_level: int
    supersedes_observation_key: StableRecordKey | None


@dataclass(frozen=True)
class W04TypedAdapterOutput:
    """W04-02 的确定性纯输出；没有 transaction 或 learning write。"""

    source_bindings: tuple[W04SourceBinding, ...]
    observations: tuple[W04ObservationEnvelope, ...]
    candidates: tuple[W04PrimitiveSurfaceCandidate, ...]
    evidence: tuple[W04EvidenceBinding, ...]
    execution_state: tuple[tuple[str, int], ...] = W04_ZERO_EXECUTION_STATE

    def candidates_for_surface(
            self,
            surface_form: str,
            *,
            context_text: str | None = None,
            ) -> tuple[W04PrimitiveSurfaceCandidate, ...]:
        """按 surface/context 返回完整候选集，不按排序私选首项。"""
        return tuple(
            item for item in self.candidates
            if item.surface_form == surface_form
            and (context_text is None or item.context_text == context_text)
        )

    def candidates_for_primitive(
            self,
            primitive_registry: str,
            primitive_kind: int,
            ) -> tuple[W04PrimitiveSurfaceCandidate, ...]:
        """返回同一 primitive 坐标的全部 surface 候选。"""
        return tuple(
            item for item in self.candidates
            if item.primitive_registry == primitive_registry
            and item.primitive_kind == primitive_kind
        )


def _definition(
        candidate: ObjectIdentity,
        competition: tuple[int, ...],
        surface: ObjectIdentity,
        context: ObjectIdentity,
        primitive: ObjectIdentity,
        source_ref: SourceRef,
        ) -> EvidenceCandidateDefinition:
    """构造 H-05 候选定义，字段均为一等对象绑定。"""
    return EvidenceCandidateDefinition(
        candidate,
        competition,
        (
            CandidateBinding(
                concept_identity((W04_NAMESPACE, 400, 1)),
                surface,
                0,
                CANDIDATE_AS_SUBJECT,
            ),
            CandidateBinding(
                concept_identity((W04_NAMESPACE, 400, 2)),
                context,
                0,
                CANDIDATE_AS_SUBJECT,
            ),
            CandidateBinding(
                concept_identity((W04_NAMESPACE, 400, 3)),
                primitive,
                0,
                CANDIDATE_AS_SUBJECT,
            ),
        ),
        (source_ref,),
    )


def _adapt_candidate(
        observation: ObservationRecord,
        source_binding: W04SourceBinding,
        ) -> W04PrimitiveSurfaceCandidate:
    """把单条 primitive Observation 映射为未选择候选。"""
    value = _typed_payload(observation)
    primitive = value["candidate_primitive"]
    registry = str(primitive["registry"])
    kind = primitive["kind"]
    surface = str(value["surface_form"])
    context_text = str(value["context"])
    primitive_object = _primitive_object(registry, kind)
    surface_atom = _surface_atom(surface)
    context = _context_object(context_text)
    candidate = _candidate_object(
        observation, registry, kind, surface, context_text)
    competition = _competition_key(surface_atom, context, primitive_object)
    definition = _definition(
        candidate,
        competition,
        surface_atom,
        context,
        primitive_object,
        source_binding.source_ref,
    )
    return W04PrimitiveSurfaceCandidate(
        observation,
        source_binding.record,
        source_binding.source_ref,
        surface,
        context_text,
        surface_atom,
        context,
        primitive_object,
        registry,
        kind,
        candidate,
        competition,
        definition,
        observation.sample_role,
        observation.perturbation_kind,
        W04_SELECTION_UNSELECTED,
        observation.supersedes_key,
        CanonicalJsonObject.from_value({
            "payload_kind": observation.payload_kind,
            "query_kind": value["query_kind"],
            "sample_role": observation.sample_role,
        }),
    )


def _teacher_stances(teacher: TeacherEvidenceRecord) -> tuple[int, ...]:
    """从 teacher expected_state 映射为 H-05 三态 stance。"""
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
        raise W04TypedAdapterError("teacher Evidence 缺合法 expected_state")
    return result


def _teacher_decision(teacher: TeacherEvidenceRecord) -> str:
    """读取 data-only expected_payload 的决策类型，不建立 Python cue 表。"""
    value = teacher.typed_evidence.to_value()
    payload = value.get("expected_payload")
    if not isinstance(payload, dict):
        raise W04TypedAdapterError("teacher expected_payload 非 object")
    decision = payload.get("decision")
    if decision not in PRIMITIVE_DECISION_CODES:
        raise W04TypedAdapterError("teacher primitive decision 未注册")
    return str(decision)


def _evidence_binding(
        envelope: W04ObservationEnvelope,
        teacher: TeacherEvidenceRecord,
        ) -> W04EvidenceBinding:
    """把 teacher Evidence 绑定到 observation 的全部候选。"""
    observation = envelope.observation
    if (teacher.observation_key != observation.stable_key
            or teacher.source_ref_key != observation.source_ref_key
            or teacher.visible_from_stage != "W-04"):
        raise W04TypedAdapterError("teacher Evidence 与 W-04 Observation 不一致")
    decision = _teacher_decision(teacher)
    reason_key = (
        1,
        *teacher.stable_key.stable_key(),
        *_text_key(teacher.evidence_kind),
        PRIMITIVE_DECISION_CODES[decision],
        teacher.withdrawal_level,
    )
    return W04EvidenceBinding(
        teacher,
        observation,
        envelope.source.source_ref,
        tuple(item.candidate for item in envelope.candidates),
        _teacher_stances(teacher),
        decision,
        reason_key,
        observation.logical_order,
        teacher.withdrawal_level,
        observation.supersedes_key,
    )


def adapt_w04_training_payload(
        payload: W04TrainingPayload,
        ) -> W04TypedAdapterOutput:
    """把 firewall 的原子 train 交付纯映射为 W-04 primitive 候选和 Evidence。"""
    if not isinstance(payload, W04TrainingPayload):
        raise TypeError("W-04 adapter 只接受 W04TrainingPayload")
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
        raise W04TypedAdapterError("train records 存在重复 stable key")
    source_by_key = _source_by_key(source_records)
    teacher_by_observation: dict[StableRecordKey, TeacherEvidenceRecord] = {}
    for teacher in teachers:
        if teacher.observation_key in teacher_by_observation:
            raise W04TypedAdapterError("同一 Observation 有多个 teacher Evidence")
        teacher_by_observation[teacher.observation_key] = teacher

    source_bindings = tuple(
        W04SourceBinding(item, _source_ref(item)) for item in source_records)
    source_binding_by_key = {
        item.record.stable_key: item for item in source_bindings}
    observation_envelopes: list[W04ObservationEnvelope] = []
    candidate_envelopes: list[W04PrimitiveSurfaceCandidate] = []
    evidence_bindings: list[W04EvidenceBinding] = []
    for observation in observations:
        if observation.w_stage != "W-04":
            continue
        source = source_by_key.get(observation.source_ref_key)
        if source is None or source.license_id != observation.license_partition:
            raise W04TypedAdapterError("Observation 来源或许可漂移")
        if observation.stable_key not in teacher_by_observation:
            raise W04TypedAdapterError("Observation 缺 teacher Evidence")
        source_binding = source_binding_by_key[observation.source_ref_key]
        candidate = _adapt_candidate(observation, source_binding)
        envelope = W04ObservationEnvelope(
            observation,
            source_binding,
            (candidate,),
            CanonicalJsonObject.from_value({
                "adapter": "ph2_w04_primitive_surface_v1",
                "payload_kind": observation.payload_kind,
            }),
        )
        observation_envelopes.append(envelope)
        candidate_envelopes.append(candidate)
        evidence_bindings.append(_evidence_binding(
            envelope,
            teacher_by_observation[observation.stable_key],
        ))
    if not observation_envelopes:
        raise W04TypedAdapterError("W-04 train payload 缺 primitive Observation")
    return W04TypedAdapterOutput(
        source_bindings,
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
    "PRIMITIVE_DECISION_CODES",
    "PRIMITIVE_REGISTRY_CODES",
    "W04EvidenceBinding",
    "W04ObservationEnvelope",
    "W04PrimitiveSurfaceCandidate",
    "W04SourceBinding",
    "W04TypedAdapterError",
    "W04TypedAdapterOutput",
    "W04_NAMESPACE",
    "W04_ZERO_EXECUTION_STATE",
    "adapt_w04_training_payload",
]
