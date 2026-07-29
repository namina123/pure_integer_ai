"""W-02 train-only payload 到既有 Candidate/Evidence/Core 的薄适配与双向消费。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

from pure_integer_ai.cognition.shared.candidate_projection import (
    CandidateProjectionGraph,
    CandidateProjectionProtocol,
)
from pure_integer_ai.cognition.shared.candidate_runtime import (
    CandidateLearningRuntime,
    CandidateProjectionMetadata,
)
from pure_integer_ai.cognition.shared.candidate_verifier import (
    IndependentObjectVerifier,
    IndependentVerifierProtocol,
    RevealedObjectObservation,
)
from pure_integer_ai.cognition.shared.evidence_candidate import (
    CandidateBinding,
    EvidenceCandidateDefinition,
    EvidenceCandidateEngine,
    EvidenceCandidateProtocol,
)
from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EVIDENCE_UNKNOWN,
    HypothesisLedger,
    LIFECYCLE_ACTIVE,
    LIFECYCLE_ARCHIVED,
    LIFECYCLE_SUPERSEDED,
)
from pure_integer_ai.cognition.shared.hypothesis_resolution import (
    HypothesisResolver,
)
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    OBJECT_ARTIFACT,
    ObjectIdentity,
    SourceRef,
    VersionBundle,
    concept_identity,
    language_branch_identity,
    representation_identity,
)
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.cognition.shared.training_hypothesis import (
    TrainingHypothesisEventSink,
    TrainingHypothesisHistoryProtocol,
)
from pure_integer_ai.cognition.understanding.segmentation_candidates import (
    SegmentationCandidate,
    SegmentationPart,
    build_segmentation_candidates,
)
from pure_integer_ai.cognition.understanding.word_form_index import WordFormIndex
from pure_integer_ai.crosscut.determinism.hasher import Hasher
from pure_integer_ai.experiments.ph2_authored_morphology_course import (
    AuthoredMorphologyCourseError,
    validate_morphology_payload,
)
from pure_integer_ai.experiments.ph2_authored_text_fidelity_course import (
    AuthoredTextFidelityCourseError,
    validate_text_fidelity_payload,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    EXPECTED_STATES,
    ObservationRecord,
    SourceRefRecord,
    TeacherEvidenceRecord,
    canonical_json_bytes,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_w02_contract import W02TrainingPayload
from pure_integer_ai.experiments.ph2_w02_use import (
    DIRECTION_GENERATION,
    DIRECTION_UNDERSTANDING,
    OUTCOME_FAILURE,
    OUTCOME_SUCCESS,
    OUTCOME_UNKNOWN,
    W02AttributionReport,
    W02DirectionalAttribution,
    W02UseOutcomeStore,
)
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.storage import discipline as disc
from pure_integer_ai.storage.backend import (
    TYPE_INT,
    TYPE_TEXT,
    StorageBackend,
    register_extension_table,
)


GENERATION_GENERATED = "GENERATED"
GENERATION_UNKNOWN = "UNKNOWN"
GENERATION_CONFLICT = "CONFLICT"
SELECTION_ADOPTED = "ADOPTED"
SELECTION_UNKNOWN = "UNKNOWN"
SELECTION_CONFLICT = "CONFLICT"
_ALLOWED_MODES = ("fresh", "restart", "resume")
_TEXT_PAYLOAD_KIND = "TextFidelityCandidateV1"
_MORPH_PAYLOAD_KIND = "MorphologyCandidateV1"
_CANDIDATE_PREFIX = 920201
_ENVELOPE_VERSION = 1
_TEXT_KEY_VERSION = 1
_SOURCE_KIND = 920202
_TEACHER_SOURCE_KIND = 920203
_AGGREGATE_SOURCE_KIND = 920204
_UNICODE_FAMILY = (920205, 1)
_LANGUAGE_BRANCH_KEY = (920206, 1)
_INVENTORY_RELATION_KEY = (920207, 1)
_HYPOTHESIS_KIND = (920208, 1)
_FORMATION_REASON = (920209, 1)
_HISTORY_NAMESPACE = (920210, 1)
_FIELD_ENVELOPE = concept_identity((920211, 1))
_FIELD_RAW = concept_identity((920211, 2))
_FIELD_DERIVED = concept_identity((920211, 3))
_FIELD_KIND = concept_identity((920211, 4))
_FIELD_RELATION = concept_identity((920211, 5))
_FIELD_GENERATION = concept_identity((920211, 6))
_SOURCE_HASHER = Hasher("ph2.w02.source.v1")
_TEACHER_HASHER = Hasher("ph2.w02.teacher_source.v1")
_ENVELOPE_HASHER = Hasher("ph2.w02.envelope.v1")
_ENVELOPE_TABLE = "ph2_w02_envelope"
_ENVELOPE_OBSERVATION = 1
_ENVELOPE_TEACHER = 2
_ENVELOPE_SOURCE = 3


class W02LearningError(RuntimeError):
    """W-02 typed mapper、候选历史或双向 consumer 违反冻结合同。"""


def _text_key(value: str) -> tuple[int, ...]:
    """把开放文本无损编码为长度前缀 UTF-8 整数键。"""
    if not isinstance(value, str) or not value:
        raise W02LearningError("W-02 文本键必须非空")
    payload = value.encode("utf-8")
    return _TEXT_KEY_VERSION, len(payload), *payload


def _artifact_identity(value: dict[str, Any]) -> ObjectIdentity:
    """把规范对象映射为小型 size/SHA Artifact identity。"""
    payload = canonical_json_bytes(value)
    digest = hashlib.sha256(payload).digest()
    words = tuple(
        int.from_bytes(digest[start:start + 4], "big")
        for start in range(0, len(digest), 4)
    )
    return ObjectIdentity(
        OBJECT_ARTIFACT,
        (_ENVELOPE_VERSION, len(payload), *words),
    )


class _W02EnvelopeStore:
    """补足 Candidate identity 不适合承载长 JSON 的小型不可变领域 store。"""

    def __init__(self, backend: StorageBackend) -> None:
        self.backend = backend
        register_extension_table(
            backend,
            _ENVELOPE_TABLE,
            [
                ("envelope_id", TYPE_INT),
                ("owner_kind", TYPE_INT),
                ("size_bytes", TYPE_INT),
                ("sha256", TYPE_TEXT),
                ("payload_json", TYPE_TEXT),
            ],
            disc.DISC_APPEND_ONLY,
            indexes=[("envelope_id",), ("owner_kind",)],
            recovery_key=("envelope_id",),
        )

    def put(self, value: dict[str, Any], *, owner_kind: int) -> ObjectIdentity:
        """幂等写规范 JSON；相同短 id 异内容立即拒绝。"""
        if owner_kind not in {
                _ENVELOPE_OBSERVATION, _ENVELOPE_TEACHER, _ENVELOPE_SOURCE}:
            raise W02LearningError("W-02 envelope owner 非法")
        identity = _artifact_identity(value)
        payload = canonical_json_bytes(value)
        sha256 = hashlib.sha256(payload).hexdigest()
        envelope_id = _ENVELOPE_HASHER.h63(identity.stable_key()) or 1
        row = {
            "envelope_id": envelope_id,
            "owner_kind": owner_kind,
            "size_bytes": len(payload),
            "sha256": sha256,
            "payload_json": payload.decode("utf-8"),
        }
        existing = self.backend.select(
            _ENVELOPE_TABLE, {"envelope_id": envelope_id})
        if existing:
            if len(existing) != 1 or existing[0] != row:
                raise W02LearningError("W-02 envelope id 碰撞或异内容覆盖")
            return identity
        self.backend.insert(_ENVELOPE_TABLE, row)
        return identity

    def read(self, identity: ObjectIdentity) -> dict[str, Any]:
        """按完整 size/SHA identity 恢复规范 JSON。"""
        if (identity.object_kind != OBJECT_ARTIFACT
                or len(identity.components) != 10
                or identity.components[0] != _ENVELOPE_VERSION):
            raise W02LearningError("W-02 envelope Artifact identity 损坏")
        envelope_id = _ENVELOPE_HASHER.h63(identity.stable_key()) or 1
        rows = self.backend.select(
            _ENVELOPE_TABLE, {"envelope_id": envelope_id})
        if len(rows) != 1:
            raise W02LearningError("W-02 envelope 缺失或重复")
        row = rows[0]
        text = row.get("payload_json")
        if not isinstance(text, str):
            raise W02LearningError("W-02 envelope payload 类型损坏")
        payload = text.encode("utf-8")
        expected = _artifact_identity(
            parse_canonical_json_bytes(payload, require_object=True))
        if (expected != identity
                or row.get("size_bytes") != len(payload)
                or row.get("sha256") != hashlib.sha256(payload).hexdigest()):
            raise W02LearningError("W-02 envelope size/SHA/identity 漂移")
        value = parse_canonical_json_bytes(payload, require_object=True)
        assert isinstance(value, dict)
        return value

    def values(self, *, owner_kind: int) -> tuple[dict[str, Any], ...]:
        """逐行验完整 payload/size/SHA/id，并返回指定唯一 owner 的规范对象。"""
        if owner_kind not in {
                _ENVELOPE_OBSERVATION, _ENVELOPE_TEACHER, _ENVELOPE_SOURCE}:
            raise W02LearningError("W-02 envelope owner 非法")
        result = []
        rows = self.backend.select(_ENVELOPE_TABLE, {"owner_kind": owner_kind})
        for row in rows:
            text = row.get("payload_json")
            if not isinstance(text, str):
                raise W02LearningError("W-02 envelope payload 类型损坏")
            payload = text.encode("utf-8")
            try:
                value = parse_canonical_json_bytes(payload, require_object=True)
            except (TypeError, ValueError) as exc:
                raise W02LearningError("W-02 envelope JSON 损坏") from exc
            assert isinstance(value, dict)
            identity = _artifact_identity(value)
            expected_id = _ENVELOPE_HASHER.h63(identity.stable_key()) or 1
            if (row.get("envelope_id") != expected_id
                    or row.get("size_bytes") != len(payload)
                    or row.get("sha256") != hashlib.sha256(payload).hexdigest()):
                raise W02LearningError("W-02 envelope size/SHA/identity 漂移")
            result.append(value)
        return tuple(sorted(result, key=canonical_json_bytes))

    def state_key(self) -> tuple:
        """返回三类 envelope 的完整 owner 与内容身份，篡改时失败关闭。"""
        result = []
        for owner_kind in (
                _ENVELOPE_OBSERVATION, _ENVELOPE_TEACHER, _ENVELOPE_SOURCE):
            values = self.values(owner_kind=owner_kind)
            result.append((
                owner_kind,
                tuple(_artifact_identity(item).stable_key() for item in values),
            ))
        return tuple(result)


def _source(record: SourceRefRecord) -> SourceRef:
    """把完整 Dataset SourceRef key 映射到稳定训练来源。"""
    source_id = _SOURCE_HASHER.h63(record.stable_key.components) or 1
    return SourceRef(
        _SOURCE_KIND,
        source_id,
        record.record_ordinal,
        GLOBAL_OWNER_SCOPE,
        VersionBundle(),
    )


def _teacher_source(record: TeacherEvidenceRecord) -> SourceRef:
    """把录放式 teacher Evidence 自身作为独立 verifier 来源。"""
    source_id = _TEACHER_HASHER.h63(record.stable_key.components) or 1
    return SourceRef(
        _TEACHER_SOURCE_KIND,
        source_id,
        0,
        GLOBAL_OWNER_SCOPE,
        VersionBundle(),
    )


def _projection_protocol() -> CandidateProjectionProtocol:
    values = tuple(concept_identity((920220, item)) for item in range(1, 14))
    return CandidateProjectionProtocol(*values, (920221, 1))


def _aggregate_source() -> SourceRef:
    return SourceRef(
        _AGGREGATE_SOURCE_KIND,
        1,
        0,
        GLOBAL_OWNER_SCOPE,
        VersionBundle(),
    )


def _evidence_protocol() -> EvidenceCandidateProtocol:
    source = _aggregate_source()
    return EvidenceCandidateProtocol(
        _HYPOTHESIS_KIND,
        _FORMATION_REASON,
        source,
        document_scope(source),
        1,
    )


def _history_protocol() -> TrainingHypothesisHistoryProtocol:
    source = _aggregate_source()
    return TrainingHypothesisHistoryProtocol(
        _HISTORY_NAMESPACE,
        _HYPOTHESIS_KIND,
        source,
        document_scope(source),
    )


def _verifier() -> IndependentObjectVerifier:
    return IndependentObjectVerifier(IndependentVerifierProtocol(
        concept_identity((920230, 1)),
        (920230, 2),
        (920230, 3),
        (920230, 4),
        (920230, 5),
    ))


def _metadata() -> CandidateProjectionMetadata:
    return CandidateProjectionMetadata(920240, 0, 1)


@dataclass(frozen=True)
class W02LearnedCandidate:
    """从 Candidate definition/H-00/H-04 无损回读的一条 W-02 候选。"""

    candidate_id: str
    candidate_group: str
    payload_kind: str
    raw_text: str
    derived_text: str
    raw_sha256: str
    derived_sha256: str
    normalization_operations: tuple[str, ...]
    relation_kinds: tuple[str, ...]
    lifecycle: str
    epistemic_status: int
    active: bool
    observation_key: tuple[int, ...]
    payload: dict[str, Any]


@dataclass(frozen=True)
class W02LearningReport:
    """本批真实 H-00/H-04/Core 写与 owner 分账。"""

    raw_observation_count: int
    teacher_evidence_count: int
    candidate_count: int
    prediction_count: int
    evidence_count: int
    decision_count: int
    projection_event_count: int
    active_candidate_count: int
    active_lifecycle_count: int
    core_learning_writes: int
    memory_learning_writes: int
    word_form_writes: int
    replayed: bool = False


@dataclass(frozen=True)
class W02UnderstandingResult:
    """保留 raw、完整 lattice、active 边界候选和 OOV 计数的理解结果。"""

    raw_text: str
    candidates: tuple[SegmentationCandidate, ...]
    active_boundary_candidates: tuple[str, ...]
    known_word_form_count: int

    def stable_key(self) -> tuple[int, ...]:
        payload = self.raw_text.encode("utf-8")
        values: list[int] = [1, len(payload), *payload, len(self.candidates)]
        for candidate in self.candidates:
            key = candidate.stable_key()
            values.extend((len(key), *key))
            values.extend(int(part.known_word_form) for part in candidate.parts)
        values.append(len(self.active_boundary_candidates))
        for item in self.active_boundary_candidates:
            key = _text_key(item)
            values.extend((len(key), *key))
        values.append(self.known_word_form_count)
        return tuple(values)


@dataclass(frozen=True)
class W02MorphologyTarget:
    """不含 expected surface 的 typed morphology 表层实现目标。"""

    construction_key: str
    stem_surface: str
    component_surfaces: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (not isinstance(self.construction_key, str)
                or not self.construction_key
                or not isinstance(self.stem_surface, str)
                or not self.stem_surface):
            raise W02LearningError("morphology target 必须给 construction 与 stem")
        if (not isinstance(self.component_surfaces, tuple)
                or any(not isinstance(item, str) or not item
                       for item in self.component_surfaces)):
            raise W02LearningError("morphology target components 类型错误")

    def stable_key(self) -> tuple[int, ...]:
        """返回不含 expected surface 的 typed morphology 请求身份。"""
        construction = _text_key(self.construction_key)
        stem = _text_key(self.stem_surface)
        values = [
            1,
            len(construction), *construction,
            len(stem), *stem,
            len(self.component_surfaces),
        ]
        for component in self.component_surfaces:
            key = _text_key(component)
            values.extend((len(key), *key))
        return tuple(values)


@dataclass(frozen=True)
class W02GenerationResult:
    """由 active 形态关系产生的零或多个合法 surface。"""

    status: str
    surfaces: tuple[str, ...]
    used_candidate_keys: tuple[tuple[int, ...], ...]

    def stable_key(self) -> tuple[int, ...]:
        values: list[int] = [
            (GENERATION_UNKNOWN, GENERATION_GENERATED, GENERATION_CONFLICT).index(
                self.status),
            len(self.surfaces),
        ]
        for surface in self.surfaces:
            key = _text_key(surface)
            values.extend((len(key), *key))
        values.append(len(self.used_candidate_keys))
        for key in self.used_candidate_keys:
            values.extend((len(key), *key))
        return tuple(values)


@dataclass(frozen=True)
class W02UnderstandingSelection:
    """理解 consumer 对 active 多候选的采用、冲突或 unknown 决定。"""

    status: str
    candidate_id: str
    understanding: W02UnderstandingResult

    def __post_init__(self) -> None:
        """要求采用态精确指向 active Candidate，其他状态不伪造选择。"""
        if self.status not in {
                SELECTION_ADOPTED, SELECTION_UNKNOWN, SELECTION_CONFLICT}:
            raise W02LearningError("understanding selection status 未登记")
        if not isinstance(self.understanding, W02UnderstandingResult):
            raise TypeError("understanding selection result 类型错误")
        if self.status == SELECTION_ADOPTED:
            if self.candidate_id not in (
                    self.understanding.active_boundary_candidates):
                raise W02LearningError("adopted boundary 不属于 active Candidate")
        elif self.candidate_id:
            raise W02LearningError("非 adopted selection 不得携带 Candidate")

    def stable_key(self) -> tuple[int, ...]:
        """返回选择状态、Candidate 与完整理解结果身份。"""
        candidate = (() if not self.candidate_id
                     else _text_key(self.candidate_id))
        understanding = self.understanding.stable_key()
        return (
            (SELECTION_UNKNOWN, SELECTION_ADOPTED,
             SELECTION_CONFLICT).index(self.status),
            len(candidate), *candidate,
            len(understanding), *understanding,
        )


def _snapshot_count(backend: StorageBackend) -> tuple[int, int]:
    """返回 Core 与 Memory 表行数，供本批 owner 写计数。"""
    snapshot = backend.snapshot()
    schema = backend.schema_snapshot()
    total = sum(
        len(rows) for table, rows in snapshot.items()
        if schema[table]["core"])
    memory = sum(
        len(rows) for table, rows in snapshot.items()
        if table.startswith("memory_"))
    return total, memory


class W02LearningRuntime:
    """唯一 W-02 候选 owner 及理解/生成只读 consumer。"""

    def __init__(
            self,
            backend: StorageBackend,
            candidate_runtime: CandidateLearningRuntime,
            word_forms: WordFormIndex,
            branch,
            envelopes: _W02EnvelopeStore,
            use_outcomes: W02UseOutcomeStore,
            *,
            mode: str,
            ) -> None:
        self.backend = backend
        self.candidate_runtime = candidate_runtime
        self.word_forms = word_forms
        self.branch = branch
        self.envelopes = envelopes
        self.use_outcomes = use_outcomes
        self.mode = mode
        self._consumed = bool(candidate_runtime.engine.definitions())

    def _definition(self, observation: ObservationRecord,
                    source: SourceRef) -> EvidenceCandidateDefinition:
        value = observation.typed_payload.to_value()
        if observation.payload_kind == _TEXT_PAYLOAD_KIND:
            raw_text = value["raw_observation"]["text"]
            derived_text = value["derived_candidate"]["text"]
        elif observation.payload_kind == _MORPH_PAYLOAD_KIND:
            raw_text = value["observed_surface"]["text"]
            derived_text = raw_text
        else:
            raise W02LearningError("W-02 payload kind 未登记")
        bindings = [
            CandidateBinding(
                _FIELD_ENVELOPE,
                _artifact_identity(observation.to_dict()),
                0,
            ),
            CandidateBinding(
                _FIELD_RAW,
                representation_identity(_UNICODE_FAMILY, tuple(map(ord, raw_text))),
                0,
            ),
            CandidateBinding(
                _FIELD_DERIVED,
                representation_identity(
                    _UNICODE_FAMILY, tuple(map(ord, derived_text))),
                0,
            ),
            CandidateBinding(
                _FIELD_KIND,
                concept_identity(_text_key(value["candidate_kind"])),
                0,
            ),
        ]
        if observation.payload_kind == _MORPH_PAYLOAD_KIND:
            for index, relation in enumerate(value["morphology_relations"], 1):
                bindings.append(CandidateBinding(
                    _FIELD_RELATION, _artifact_identity(relation), index))
            bindings.append(CandidateBinding(
                _FIELD_GENERATION,
                _artifact_identity(value["generation_constraint"]),
                0,
            ))
        return EvidenceCandidateDefinition(
            concept_identity((
                _CANDIDATE_PREFIX,
                len(observation.stable_key.components),
                *observation.stable_key.components,
            )),
            _text_key(value["candidate_group"]),
            tuple(bindings),
            (source,),
        )

    def _preflight_payload(
            self,
            payload: W02TrainingPayload,
            ) -> tuple[
                dict,
                tuple[ObservationRecord, ...],
                tuple[TeacherEvidenceRecord, ...],
            ]:
        """在首个 envelope/Candidate 写前闭合 typed schema、owner、引用和 revision。"""
        if (not isinstance(payload.source_refs, tuple)
                or any(not isinstance(item, SourceRefRecord)
                       for item in payload.source_refs)
                or not isinstance(payload.observations, tuple)
                or any(not isinstance(item, ObservationRecord)
                       for item in payload.observations)
                or not isinstance(payload.teacher_evidence, tuple)
                or any(not isinstance(item, TeacherEvidenceRecord)
                       for item in payload.teacher_evidence)):
            raise W02LearningError("W-02 train payload record 类型非法")
        source_by_key = {item.stable_key: item for item in payload.source_refs}
        if len(source_by_key) != len(payload.source_refs):
            raise W02LearningError("W-02 SourceRef stable key 重复")
        observations = tuple(sorted(
            payload.observations,
            key=lambda item: (item.substage, item.logical_order, item.stable_key),
        ))
        if len({item.stable_key for item in observations}) != len(observations):
            raise W02LearningError("W-02 Observation stable key 重复")
        candidate_ids = []
        values_by_key = {}
        try:
            for item in observations:
                if item.payload_kind == _TEXT_PAYLOAD_KIND:
                    validate_text_fidelity_payload(item.typed_payload)
                elif item.payload_kind == _MORPH_PAYLOAD_KIND:
                    validate_morphology_payload(item.typed_payload)
                else:
                    raise W02LearningError("W-02 Observation payload kind 越界")
                source = source_by_key.get(item.source_ref_key)
                if (item.w_stage != "W-02" or item.split != "train"
                        or source is None
                        or source.dataset_key != item.dataset_key
                        or source.artifact_key != item.artifact_key
                        or source.course_version != item.course_version):
                    raise W02LearningError(
                        "W-02 Observation 不是闭合 train/SourceRef 记录")
                value = item.typed_payload.to_value()
                candidate_ids.append(value["candidate_id"])
                values_by_key[item.stable_key] = value
        except W02LearningError:
            raise
        except (AuthoredMorphologyCourseError,
                AuthoredTextFidelityCourseError,
                KeyError, TypeError, ValueError) as exc:
            raise W02LearningError("W-02 LC-01/LC-02 typed payload 损坏") from exc
        if len(set(candidate_ids)) != len(candidate_ids):
            raise W02LearningError("W-02 candidate_id 重复")

        teacher_records = tuple(sorted(
            payload.teacher_evidence,
            key=lambda item: item.stable_key,
        ))
        if len({item.stable_key for item in teacher_records}) != len(
                teacher_records):
            raise W02LearningError("W-02 teacher Evidence stable key 重复")
        by_observation = {item.stable_key: item for item in observations}
        teacher_by_observation = {}
        owner_by_artifact = {}
        artifact_by_owner = {}
        for teacher in teacher_records:
            observation = by_observation.get(teacher.observation_key)
            source = source_by_key.get(teacher.source_ref_key)
            if observation is None or source is None:
                raise W02LearningError(
                    "teacher Evidence 引用非 train Observation/SourceRef")
            if teacher.observation_key in teacher_by_observation:
                raise W02LearningError(
                    "同一 train Observation 绑定多条 teacher Evidence")
            if (teacher.visible_from_stage != "W-02"
                    or teacher.source_ref_key != observation.source_ref_key
                    or teacher.dataset_key != observation.dataset_key
                    or teacher.artifact_key != observation.artifact_key
                    or teacher.course_version != observation.course_version):
                raise W02LearningError("teacher Evidence owner/source 身份漂移")
            artifact_route = (teacher.dataset_key, teacher.artifact_key)
            prior_owner = owner_by_artifact.setdefault(
                artifact_route, teacher.owner_key)
            prior_artifact = artifact_by_owner.setdefault(
                teacher.owner_key, artifact_route)
            if prior_owner != teacher.owner_key or prior_artifact != artifact_route:
                raise W02LearningError("teacher Evidence 唯一 owner 漂移")
            evidence = teacher.typed_evidence.to_value()
            if (set(evidence) != {
                    "expected_payload", "expected_state", "seed_id"}
                    or evidence["expected_state"] not in EXPECTED_STATES
                    or not isinstance(evidence["expected_payload"], dict)
                    or evidence["seed_id"] != source.revision_id):
                raise W02LearningError("teacher typed Evidence 损坏")
            teacher_by_observation[teacher.observation_key] = teacher
        if set(teacher_by_observation) != set(by_observation):
            raise W02LearningError("train Observation 与 teacher Evidence 非一一闭合")
        for observation in observations:
            if observation.supersedes_key is None:
                continue
            prior = by_observation.get(observation.supersedes_key)
            if (prior is None
                    or prior.logical_order >= observation.logical_order
                    or values_by_key[prior.stable_key]["candidate_group"]
                    != values_by_key[observation.stable_key]["candidate_group"]):
                raise W02LearningError("supersede 引用或竞争组非法")
        return source_by_key, observations, teacher_records

    def _persist_payload(self, payload: W02TrainingPayload) -> None:
        """按 owner 写入完整规范记录，供重放和 fresh connection 校验。"""
        for record in payload.source_refs:
            self.envelopes.put(record.to_dict(), owner_kind=_ENVELOPE_SOURCE)
        for record in payload.observations:
            self.envelopes.put(record.to_dict(), owner_kind=_ENVELOPE_OBSERVATION)
        for record in payload.teacher_evidence:
            self.envelopes.put(record.to_dict(), owner_kind=_ENVELOPE_TEACHER)

    def _matches_persisted_payload(self, payload: W02TrainingPayload) -> bool:
        """逐字节比较三 owner 规范记录，拒绝同 owner 的异内容重放。"""
        actual = (
            self.envelopes.values(owner_kind=_ENVELOPE_SOURCE),
            self.envelopes.values(owner_kind=_ENVELOPE_OBSERVATION),
            self.envelopes.values(owner_kind=_ENVELOPE_TEACHER),
        )
        expected = tuple(tuple(sorted(
            (item.to_dict() for item in records), key=canonical_json_bytes))
            for records in (
                payload.source_refs,
                payload.observations,
                payload.teacher_evidence,
            ))
        return actual == expected

    def _replay_report(
            self,
            observations: tuple[ObservationRecord, ...],
            teacher_records: tuple[TeacherEvidenceRecord, ...],
            ) -> W02LearningReport:
        """从持久状态形成零新增写的幂等重放报告。"""
        report = self.candidate_runtime.report()
        return W02LearningReport(
            sum(item.payload_kind == _TEXT_PAYLOAD_KIND
                for item in observations),
            len(teacher_records),
            report.candidate_count,
            report.prediction_count,
            report.evidence_count,
            report.decision_count,
            report.projection_event_count,
            report.active_projection_count,
            sum(item.lifecycle == "ACTIVE" for item in self.candidates()),
            0,
            0,
            0,
            True,
        )

    def consume(
            self,
            payload: W02TrainingPayload,
            *,
            commit: bool = True,
            ) -> W02LearningReport:
        """验证后形成候选、揭示 replay Evidence、supersede 并建立词形索引。"""
        if not isinstance(payload, W02TrainingPayload):
            raise TypeError("W-02 consume 需要 W02TrainingPayload")
        if type(commit) is not bool:
            raise TypeError("W-02 consume commit 必须是 bool")
        source_by_key, observations, teacher_records = self._preflight_payload(
            payload)
        if self._consumed:
            if not self._matches_persisted_payload(payload):
                raise W02LearningError("同一 W-02 owner 重放了不同 train payload")
            return self._replay_report(observations, teacher_records)
        before_total, before_memory = _snapshot_count(self.backend)
        definitions = tuple(
            self._definition(item, _source(source_by_key[item.source_ref_key]))
            for item in observations)
        requests = tuple(
            (definition, index * 2 + 1)
            for index, definition in enumerate(definitions)
        )
        self.candidate_runtime.preflight_register_many(requests)
        self._persist_payload(payload)
        hypotheses = self.candidate_runtime.register_many(requests)
        by_observation = dict(zip(
            (item.stable_key for item in observations),
            zip(observations, definitions, hypotheses, strict=True),
            strict=True,
        ))
        superseded_observations = {
            item.supersedes_key for item in observations
            if item.supersedes_key is not None
        }
        teacher_records = tuple(sorted(
            teacher_records,
            key=lambda item: (
                by_observation[item.observation_key][0].substage,
                by_observation[item.observation_key][0].logical_order,
                item.stable_key,
            ),
        ))
        teacher_by_observation = {
            item.observation_key: item for item in teacher_records}
        for teacher in teacher_records:
            triple = by_observation.get(teacher.observation_key)
            if triple is None:
                raise W02LearningError("teacher Evidence 引用非 train Observation")
            observation, definition, hypothesis = triple
            expected_state = teacher.typed_evidence.to_value()["expected_state"]
            deferred_refute = (
                expected_state == "FALSE"
                and observation.stable_key in superseded_observations)
            if not deferred_refute:
                self._recognize_teacher(
                    hypothesis, definition, teacher,
                    expected_state=expected_state,
                    archive_refuted=(expected_state == "FALSE"),
                )
            if observation.supersedes_key is not None:
                prior = by_observation.get(observation.supersedes_key)
                if prior is None:
                    raise W02LearningError("supersede 引用非本批旧 Observation")
                _, prior_definition, prior_hypothesis = prior
                if (prior_definition.competition_key
                        != definition.competition_key):
                    raise W02LearningError("supersede 候选不在同一竞争组")
                prior_teacher = teacher_by_observation.get(
                    observation.supersedes_key)
                if prior_teacher is None:
                    raise W02LearningError("supersede 旧候选缺 teacher Evidence")
                self._recognize_teacher(
                    prior_hypothesis,
                    prior_definition,
                    teacher,
                    expected_state="FALSE",
                    replacement=hypothesis,
                    archive_refuted=False,
                    additional_trace_teacher=(
                        prior_teacher
                        if prior_teacher.typed_evidence.to_value()[
                            "expected_state"] == "FALSE"
                        else None),
                )
        word_form_writes = self._install_active_word_forms()
        after_total, after_memory = _snapshot_count(self.backend)
        active_candidate_count = sum(
            item.active for item in self.candidates())
        active_lifecycle_count = sum(
            item.lifecycle == "ACTIVE" for item in self.candidates())
        if commit:
            self.backend.commit()
        self._consumed = True
        report = self.candidate_runtime.report()
        return W02LearningReport(
            sum(item.payload_kind == _TEXT_PAYLOAD_KIND
                for item in observations),
            len(teacher_records),
            report.candidate_count,
            report.prediction_count,
            report.evidence_count,
            report.decision_count,
            report.projection_event_count,
            active_candidate_count,
            active_lifecycle_count,
            after_total - before_total,
            after_memory - before_memory,
            word_form_writes,
            False,
        )

    def _recognize_teacher(
            self,
            hypothesis,
            definition: EvidenceCandidateDefinition,
            teacher: TeacherEvidenceRecord,
            *,
            expected_state: str,
            replacement=None,
            archive_refuted: bool,
            additional_trace_teacher: TeacherEvidenceRecord | None = None,
            ) -> None:
        """先冻结预测，再把一条录放 Evidence 映射为三态核验和可选替代。"""
        verifier_source = _teacher_source(teacher)
        scope = document_scope(verifier_source)
        predicted = definition.candidate
        if expected_state in {"TRUE", "CONFLICT"}:
            supported = (predicted,)
            refuted = ()
        elif expected_state == "FALSE":
            supported = ()
            refuted = (predicted,)
        elif expected_state == "UNKNOWN":
            supported = refuted = ()
        else:
            raise W02LearningError("teacher expected_state 未登记")
        envelope = next(
            item.value for item in definition.bindings
            if item.predicate == _FIELD_ENVELOPE)
        teacher_identity = self.envelopes.put(
            teacher.to_dict(), owner_kind=_ENVELOPE_TEACHER)
        trace_keys = [teacher_identity.stable_key()]
        if additional_trace_teacher is not None:
            trace_keys.append(self.envelopes.put(
                additional_trace_teacher.to_dict(),
                owner_kind=_ENVELOPE_TEACHER,
            ).stable_key())
        trace: list[int] = [len(trace_keys)]
        for key in trace_keys:
            trace.extend((len(key), *key))
        timestamp, resolve_timestamp, projection_timestamp = (
            self.candidate_runtime.next_timestamps(3))
        event_key = (
            *teacher.stable_key.components,
            0 if replacement is None else 1,
        )
        self.candidate_runtime.recognize(
            hypothesis,
            observation=verifier_source,
            scope=scope,
            event_key=event_key,
            visible_inputs=(envelope,),
            predicted=predicted,
            revealed=RevealedObjectObservation(
                verifier_source,
                scope,
                event_key,
                verifier_source,
                supported,
                refuted,
                tuple(trace),
            ),
            timestamp_seq=timestamp,
            resolve_timestamp_seq=resolve_timestamp,
            projection_timestamp_seq=projection_timestamp,
            archive_refuted=archive_refuted,
            replacement=replacement,
        )

    def _install_active_word_forms(self) -> int:
        """只把 active 结构候选的组成单位写入权威 WordFormIndex。"""
        before = len(self.word_forms.forms(branch=self.branch))
        scope = document_scope(_aggregate_source())
        for candidate in self.candidates():
            if not candidate.active or candidate.payload_kind != _MORPH_PAYLOAD_KIND:
                continue
            value = candidate.payload
            if (value["baseline_kind"] == "DICTIONARY_REPLAY_ONLY"
                    or value["candidate_kind"] in {"UNKNOWN", "GENERATION"}):
                continue
            for unit in value["analysis_units"]:
                if unit["unit_kind"] == "CONSTRUCTION":
                    continue
                self.word_forms.ensure(
                    unit["surface"],
                    branch=self.branch,
                    scope=scope,
                    provenance_kind=920250,
                    content_version=1,
                )
        after = len(self.word_forms.forms(branch=self.branch))
        return after - before

    def candidates(self) -> tuple[W02LearnedCandidate, ...]:
        """从既有 Candidate definitions 与 H-00/H-04 状态恢复领域视图。"""
        result = []
        lifecycle_names = {
            LIFECYCLE_ACTIVE: "ACTIVE",
            LIFECYCLE_ARCHIVED: "ARCHIVED",
            LIFECYCLE_SUPERSEDED: "SUPERSEDED",
        }
        for definition in self.candidate_runtime.engine.definitions():
            envelope = next((
                item.value for item in definition.bindings
                if item.predicate == _FIELD_ENVELOPE and item.ordinal == 0
            ), None)
            if envelope is None:
                raise W02LearningError("W-02 Candidate 缺 Observation envelope")
            record = ObservationRecord.from_dict(self.envelopes.read(envelope))
            value = record.typed_payload.to_value()
            if record.payload_kind == _TEXT_PAYLOAD_KIND:
                raw = value["raw_observation"]
                derived = value["derived_candidate"]
                operations = tuple(
                    item["operation_kind"]
                    for item in value["normalization_receipt"]["operations"])
                relations: tuple[str, ...] = ()
            elif record.payload_kind == _MORPH_PAYLOAD_KIND:
                raw = derived = value["observed_surface"]
                operations = ()
                relations = tuple(
                    item["relation_kind"]
                    for item in value["morphology_relations"])
            else:
                raise W02LearningError("恢复 Candidate payload kind 越界")
            hypothesis = definition.hypothesis(_evidence_protocol())
            snapshot = self.candidate_runtime.engine.ledger.snapshot(hypothesis)
            result.append(W02LearnedCandidate(
                value["candidate_id"],
                value["candidate_group"],
                record.payload_kind,
                raw["text"],
                derived["text"],
                raw["sha256"],
                derived["sha256"],
                operations,
                relations,
                lifecycle_names[snapshot.lifecycle],
                snapshot.epistemic_status,
                self.candidate_runtime.engine.active(hypothesis) is not None,
                record.stable_key.components,
                value,
            ))
        return tuple(sorted(result, key=lambda item: item.candidate_id))

    def candidate(self, candidate_id: str) -> W02LearnedCandidate:
        """按课程候选 id 精确回读，重复或缺失均拒绝。"""
        matches = tuple(
            item for item in self.candidates()
            if item.candidate_id == candidate_id)
        if len(matches) != 1:
            raise KeyError("W-02 candidate_id 不唯一或不存在")
        return matches[0]

    def active_morphology_relation_kinds(self) -> tuple[str, ...]:
        """返回 active morphology Candidate 实际承载的关系种类。"""
        return tuple(sorted({
            relation
            for candidate in self.candidates()
            if candidate.active and candidate.payload_kind == _MORPH_PAYLOAD_KIND
            for relation in candidate.relation_kinds
        }))

    def understand(self, raw_text: str) -> W02UnderstandingResult:
        """由权威词形索引和 active Candidate 形成多分词及连续 OOV。"""
        if not isinstance(raw_text, str) or not raw_text:
            raise W02LearningError("understanding raw text 必须非空")
        lattice = self.word_forms.match_lattice(raw_text, branch=self.branch)
        candidates = {
            item.stable_key(): item
            for item in build_segmentation_candidates(
                raw_text, lattice, candidate_limit=32)
        }
        active_boundary: list[str] = []
        known_forms = self.word_forms.forms(branch=self.branch)
        for learned in self.candidates():
            if not learned.active:
                continue
            value = learned.payload
            segments: tuple[str, ...] = ()
            if (learned.payload_kind == _TEXT_PAYLOAD_KIND
                    and value["candidate_kind"] == "SEGMENTATION"
                    and value["raw_observation"]["text"] == raw_text):
                segments = tuple(value["derived_candidate"]["segments"])
            elif (learned.payload_kind == _MORPH_PAYLOAD_KIND
                  and value["candidate_kind"] == "SEGMENTATION"
                  and value["observed_surface"]["text"] == raw_text):
                segments = tuple(
                    unit["surface"] for unit in value["analysis_units"]
                    if unit["unit_kind"] != "CONSTRUCTION")
            if not segments or "".join(segments) != raw_text:
                continue
            cursor = 0
            parts = []
            for segment in segments:
                end = cursor + len(segment)
                parts.append(SegmentationPart(
                    cursor,
                    end,
                    segment,
                    tuple(map(ord, segment)) in known_forms,
                ))
                cursor = end
            candidate = SegmentationCandidate(tuple(parts))
            candidates[candidate.stable_key()] = candidate
            active_boundary.append(learned.candidate_id)
        ordered = tuple(sorted(
            candidates.values(), key=lambda item: item.stable_key()))
        return W02UnderstandingResult(
            raw_text,
            ordered,
            tuple(sorted(active_boundary)),
            len({surface for matches in lattice for surface in matches}),
        )

    def select_understanding(
            self,
            raw_text: str,
            *,
            outcome_assessment_enabled: bool = True,
            ) -> W02UnderstandingSelection:
        """按方向专属 assessment 选择 active 边界；禁用时保留真实冲突。"""
        if type(outcome_assessment_enabled) is not bool:
            raise TypeError("outcome_assessment_enabled 必须是 bool")
        understanding = self.understand(raw_text)
        candidates = understanding.active_boundary_candidates
        if not candidates:
            return W02UnderstandingSelection(
                SELECTION_UNKNOWN, "", understanding)
        if len(candidates) == 1:
            return W02UnderstandingSelection(
                SELECTION_ADOPTED, candidates[0], understanding)
        if not outcome_assessment_enabled:
            return W02UnderstandingSelection(
                SELECTION_CONFLICT, "", understanding)
        scored = tuple(
            (self.use_outcomes.score(
                DIRECTION_UNDERSTANDING,
                self.candidate(candidate_id).observation_key,
            ), candidate_id)
            for candidate_id in candidates
        )
        best = max(item[0] for item in scored)
        winners = tuple(item[1] for item in scored if item[0] == best)
        if best > 0 and len(winners) == 1:
            return W02UnderstandingSelection(
                SELECTION_ADOPTED, winners[0], understanding)
        return W02UnderstandingSelection(
            SELECTION_CONFLICT, "", understanding)

    def record_understanding_outcome(
            self,
            raw_text: str,
            candidate_id: str,
            *,
            outcome_kind: str,
            assessment_enabled: bool = True,
            commit: bool = True,
            ) -> W02DirectionalAttribution:
        """只为本次实际采用的 active 边界 Candidate 写理解 Use/outcome。"""
        understanding = self.understand(raw_text)
        if candidate_id not in understanding.active_boundary_candidates:
            raise W02LearningError("理解 Use 未采用 active boundary Candidate")
        candidate = self.candidate(candidate_id)
        request_key = _text_key(raw_text)
        use_key = (
            1,
            len(request_key), *request_key,
            len(candidate.observation_key), *candidate.observation_key,
        )
        return self.use_outcomes.record(
            direction=DIRECTION_UNDERSTANDING,
            use_key=use_key,
            request_key=request_key,
            candidate_key=candidate.observation_key,
            outcome_kind=outcome_kind,
            outcome_trace_key=understanding.stable_key(),
            assessment_enabled=assessment_enabled,
            commit=commit,
        )

    def generate(
            self,
            target: W02MorphologyTarget,
            *,
            morphology_consumer_enabled: bool = True,
            ) -> W02GenerationResult:
        """只从 active 关系模板组合 surface；缺词形、冲突或回放基线均失败关闭。"""
        if not isinstance(target, W02MorphologyTarget):
            raise TypeError("generate target 类型错误")
        if type(morphology_consumer_enabled) is not bool:
            raise TypeError("morphology_consumer_enabled 必须是 bool")
        if not morphology_consumer_enabled:
            return W02GenerationResult(GENERATION_UNKNOWN, (), ())
        if self.word_forms.lookup(
                target.stem_surface, branch=self.branch) is None:
            return W02GenerationResult(GENERATION_UNKNOWN, (), ())
        surfaces: dict[str, tuple[int, ...]] = {}
        for learned in self.candidates():
            if not learned.active or learned.payload_kind != _MORPH_PAYLOAD_KIND:
                continue
            value = learned.payload
            if (value["construction_key"] != target.construction_key
                    or value["baseline_kind"] == "DICTIONARY_REPLAY_ONLY"
                    or value["candidate_kind"] in {
                        "UNKNOWN", "SEGMENTATION", "GENERATION"}):
                continue
            generated = self._apply_rule(value, target)
            if generated is not None:
                surfaces[generated] = learned.observation_key
        ordered = tuple(sorted(surfaces))
        if not ordered:
            return W02GenerationResult(GENERATION_UNKNOWN, (), ())
        status = (
            GENERATION_GENERATED if len(ordered) == 1
            else GENERATION_CONFLICT)
        return W02GenerationResult(
            status,
            ordered,
            tuple(surfaces[item] for item in ordered),
        )

    def record_generation_outcome(
            self,
            target: W02MorphologyTarget,
            chosen_surface: str,
            *,
            outcome_kind: str,
            assessment_enabled: bool = True,
            commit: bool = True,
            ) -> W02DirectionalAttribution:
        """只为实际生成并采用的一个 surface 对应 Candidate 写生成 Use/outcome。"""
        if not isinstance(chosen_surface, str) or not chosen_surface:
            raise W02LearningError("generation Use chosen surface 必须非空")
        result = self.generate(target)
        try:
            index = result.surfaces.index(chosen_surface)
        except ValueError as exc:
            raise W02LearningError(
                "generation Use surface 不是本次 consumer 实际输出") from exc
        candidate_key = result.used_candidate_keys[index]
        request_key = target.stable_key()
        surface_key = _text_key(chosen_surface)
        use_key = (
            2,
            len(request_key), *request_key,
            len(surface_key), *surface_key,
            len(candidate_key), *candidate_key,
        )
        return self.use_outcomes.record(
            direction=DIRECTION_GENERATION,
            use_key=use_key,
            request_key=request_key,
            candidate_key=candidate_key,
            outcome_kind=outcome_kind,
            outcome_trace_key=result.stable_key(),
            assessment_enabled=assessment_enabled,
            commit=commit,
        )

    def attribution_report(self) -> W02AttributionReport:
        """回读理解/生成分方向 Use、outcome 与 assessment 计数。"""
        return self.use_outcomes.report()

    def _apply_rule(
            self, value: dict[str, Any], target: W02MorphologyTarget,
            ) -> str | None:
        """按显式 relation/span 结构应用 affix、redup、compound 或局部 exception。"""
        units = {item["unit_id"]: item for item in value["analysis_units"]}
        relations = value["morphology_relations"]
        kinds = {item["relation_kind"] for item in relations}
        if "REDUPLICATES" in kinds:
            return target.stem_surface + target.stem_surface
        if "ATTACHES_AFFIX" in kinds:
            relation = next(
                item for item in relations
                if item["relation_kind"] == "ATTACHES_AFFIX")
            affix = units[relation["source_unit_id"]]
            stem = units[relation["target_unit_id"]]
            if self.word_forms.lookup(
                    affix["surface"], branch=self.branch) is None:
                return None
            return (
                target.stem_surface + affix["surface"]
                if affix["start"] >= stem["end"]
                else affix["surface"] + target.stem_surface)
        if "COMPOUND_COMPONENT" in kinds:
            components = target.component_surfaces
            if not components:
                return None
            all_parts = (*components, target.stem_surface)
            if any(self.word_forms.lookup(item, branch=self.branch) is None
                   for item in all_parts):
                return None
            return "".join(all_parts)
        if "EXCEPTION_TO" in kinds:
            observed = value["observed_surface"]["text"]
            return observed if target.stem_surface == observed else None
        return None

    def state_key(self) -> tuple:
        """返回 Candidate/H-00/H-04/Core 词形的完整可比较逻辑状态。"""
        forms = tuple(sorted(
            (codepoints, ref.stable_key())
            for codepoints, ref in self.word_forms.forms(
                branch=self.branch).items()
        ))
        return (
            self.candidate_runtime.state_key(),
            forms,
            self.envelopes.state_key(),
            self.use_outcomes.state_key(),
        )


def open_w02_learning_runtime(
        backend: StorageBackend,
        *,
        mode: str,
        ) -> W02LearningRuntime:
    """装配或从 SQLite/Core 历史恢复唯一 W-02 学习 owner。"""
    if not isinstance(backend, StorageBackend):
        raise TypeError("W-02 learning backend 类型错误")
    if mode not in _ALLOWED_MODES:
        raise W02LearningError("W-02 learning mode 非法")
    ctx = make_train_context(backend)
    aggregate_scope = document_scope(_aggregate_source())
    ctx.scoped_identity_store.register_scope(aggregate_scope)
    graph = CandidateProjectionGraph(ctx.graph_ontology, _projection_protocol())
    history = ctx.training_candidate_history
    assert history is not None
    history_protocol = _history_protocol()
    sink = TrainingHypothesisEventSink(history, history_protocol)
    existing = sink.hypotheses()
    if mode == "fresh" and existing:
        raise W02LearningError("fresh W-02 learning 要求不存在候选历史")
    if mode == "resume" and not existing:
        raise W02LearningError("resume W-02 learning 缺少候选历史")
    if existing:
        candidate_runtime = CandidateLearningRuntime.restore_for_training_graph(
            _evidence_protocol(),
            graph,
            _verifier(),
            _metadata(),
            history,
            history_protocol,
        )
    else:
        ledger = HypothesisLedger(sink)
        resolver = HypothesisResolver(ledger, sink=sink)
        candidate_runtime = CandidateLearningRuntime(
            EvidenceCandidateEngine(
                _evidence_protocol(), ledger=ledger, resolver=resolver),
            graph,
            _verifier(),
            _metadata(),
        )
    branch = ctx.graph_ontology.materialize(
        language_branch_identity(_LANGUAGE_BRANCH_KEY))
    word_forms = WordFormIndex(
        backend,
        ctx.concept_index,
        ontology=ctx.graph_ontology,
        unicode_family_key=_UNICODE_FAMILY,
        inventory_relation_key=_INVENTORY_RELATION_KEY,
    )
    envelopes = _W02EnvelopeStore(backend)
    use_outcomes = W02UseOutcomeStore(backend)
    return W02LearningRuntime(
        backend, candidate_runtime, word_forms, branch, envelopes, use_outcomes,
        mode=mode)


__all__ = [
    "GENERATION_CONFLICT",
    "GENERATION_GENERATED",
    "GENERATION_UNKNOWN",
    "OUTCOME_FAILURE",
    "OUTCOME_SUCCESS",
    "OUTCOME_UNKNOWN",
    "SELECTION_ADOPTED",
    "SELECTION_CONFLICT",
    "SELECTION_UNKNOWN",
    "W02GenerationResult",
    "W02LearnedCandidate",
    "W02LearningError",
    "W02LearningReport",
    "W02LearningRuntime",
    "W02MorphologyTarget",
    "W02UnderstandingResult",
    "W02UnderstandingSelection",
    "open_w02_learning_runtime",
]
