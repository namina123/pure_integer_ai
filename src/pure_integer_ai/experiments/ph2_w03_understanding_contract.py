"""W03-03 理解闭环的协议、identity helper 与只读值对象。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.candidate_projection import (
    CandidateProjectionProtocol,
)
from pure_integer_ai.cognition.shared.candidate_runtime import (
    CandidateLearningOutcome,
)
from pure_integer_ai.cognition.shared.candidate_verifier import (
    IndependentObjectVerifier,
    IndependentVerifierProtocol,
)
from pure_integer_ai.cognition.shared.evidence_candidate import (
    EvidenceCandidateEngine,
    EvidenceCandidateProtocol,
)
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    CorpusVersion,
    CurriculumVersion,
    ObjectIdentity,
    ParserVersion,
    PrimitiveVersion,
    SourceRef,
    VersionBundle,
    concept_identity,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    ScopeIdentity,
    document_scope,
)
from pure_integer_ai.cognition.shared.training_hypothesis import (
    TrainingHypothesisHistoryProtocol,
)
from pure_integer_ai.cognition.understanding.language_candidate import (
    ActiveSenseCandidate,
)
from pure_integer_ai.crosscut.determinism.hasher import Hasher
from pure_integer_ai.experiments.ph2_dataset_contract import (
    TeacherEvidenceRecord,
)
from pure_integer_ai.experiments.ph2_w03_adapter import (
    W03EvidenceBinding,
    W03SenseCandidateEnvelope,
)


W03_UNDERSTANDING_UNIQUE = "UNIQUE"
W03_UNDERSTANDING_AMBIGUOUS = "AMBIGUOUS"
W03_UNDERSTANDING_CLARIFY = "CLARIFY"
W03_UNDERSTANDING_UNKNOWN = "UNKNOWN"
_W03_UNDERSTANDING_STATUSES = {
    W03_UNDERSTANDING_UNIQUE,
    W03_UNDERSTANDING_AMBIGUOUS,
    W03_UNDERSTANDING_CLARIFY,
    W03_UNDERSTANDING_UNKNOWN,
}
_W03_STATUS_CODES = {
    W03_UNDERSTANDING_UNIQUE: 1,
    W03_UNDERSTANDING_AMBIGUOUS: 2,
    W03_UNDERSTANDING_CLARIFY: 3,
    W03_UNDERSTANDING_UNKNOWN: 4,
}
_W03_NAMESPACE = 30303
_TEACHER_SOURCE_HASHER = Hasher((_W03_NAMESPACE, 1))
_EVENT_HASHER = Hasher((_W03_NAMESPACE, 2))


class W03UnderstandingError(RuntimeError):
    """W-03 understanding identity、Evidence 或重复消费非法。"""


def _pack(value: tuple[int, ...]) -> tuple[int, ...]:
    """给稳定整数键增加长度前缀。"""
    return len(value), *value


def _projection_protocol() -> CandidateProjectionProtocol:
    """建立 W-03 独占的候选 lifecycle 图协议。"""
    values = tuple(
        concept_identity((_W03_NAMESPACE, 100 + ordinal))
        for ordinal in range(13)
    )
    return CandidateProjectionProtocol(
        values[0],
        values[1],
        values[2],
        values[3],
        values[4],
        values[5],
        values[6],
        values[7],
        values[8],
        values[9],
        values[10],
        values[11],
        values[12],
        (_W03_NAMESPACE, 200),
    )


def _aggregate_source(versions: VersionBundle) -> SourceRef:
    """按 candidate version 建立只作 owner aggregate 的 W-03 来源。"""
    source_id = _TEACHER_SOURCE_HASHER.h63(versions.stable_key()) or 1
    return SourceRef(
        _W03_NAMESPACE,
        source_id,
        0,
        GLOBAL_OWNER_SCOPE,
        versions,
    )


def _candidate_engine(versions: VersionBundle) -> EvidenceCandidateEngine:
    """为一个严格 version bucket 建立 W-03 Evidence owner。"""
    aggregate = _aggregate_source(versions)
    return EvidenceCandidateEngine(EvidenceCandidateProtocol(
        (_W03_NAMESPACE, 201),
        (_W03_NAMESPACE, 202),
        aggregate,
        document_scope(aggregate),
        1,
    ))


def _history_protocol(
        versions: VersionBundle,
        ) -> TrainingHypothesisHistoryProtocol:
    """按 candidate version 隔离 W-03 Core H-00/H-04 持久历史。"""
    engine = _candidate_engine(versions)
    return TrainingHypothesisHistoryProtocol(
        (_W03_NAMESPACE, 208, *versions.stable_key()),
        engine.protocol.hypothesis_kind_key,
        engine.protocol.aggregate_source,
        engine.protocol.aggregate_scope,
    )


def _verifier() -> IndependentObjectVerifier:
    """建立只读取显式 teacher reveal 的独立三态 verifier。"""
    return IndependentObjectVerifier(IndependentVerifierProtocol(
        concept_identity((_W03_NAMESPACE, 203)),
        (_W03_NAMESPACE, 204),
        (_W03_NAMESPACE, 205),
        (_W03_NAMESPACE, 206),
        (_W03_NAMESPACE, 207),
    ))


def _teacher_source(binding: W03EvidenceBinding) -> SourceRef:
    """把完整 teacher record identity 投影为运行时 SourceRef。"""
    teacher = binding.teacher_record
    identity = (
        *_pack(teacher.stable_key.stable_key()),
        *_pack(teacher.owner_key.stable_key()),
        *_pack(teacher.source_ref_key.stable_key()),
    )
    source_id = _TEACHER_SOURCE_HASHER.h63(identity) or 1
    document_id = _TEACHER_SOURCE_HASHER.h63((
        *_pack(teacher.observation_key.stable_key()),
        binding.logical_order,
    )) or 1
    return SourceRef(
        _W03_NAMESPACE,
        source_id,
        document_id,
        GLOBAL_OWNER_SCOPE,
        VersionBundle(
            CorpusVersion(teacher.course_version),
            ParserVersion(teacher.schema_version),
            PrimitiveVersion(teacher.schema_version),
            CurriculumVersion(teacher.course_version),
        ),
    )


def _event_key(
        binding: W03EvidenceBinding,
        candidate: W03SenseCandidateEnvelope,
        *,
        stance: int,
        stance_ordinal: int,
        derived_supersede: bool,
        ) -> tuple[int, ...]:
    """从 teacher、候选、立场和用途构造唯一 recognition event key。"""
    role = 2 if derived_supersede else 1
    digest = _EVENT_HASHER.h63((
        *_pack(binding.teacher_record.stable_key.stable_key()),
        *_pack(candidate.sense.stable_key()),
        stance,
        stance_ordinal,
        role,
    )) or 1
    return _W03_NAMESPACE, role, digest, stance


@dataclass(frozen=True)
class W03SenseResolution:
    """一次 atom/context 查询的完整候选、active 集和严格结果。"""

    status: str
    atom: ObjectIdentity
    context: ObjectIdentity | None
    candidates: tuple[W03SenseCandidateEnvelope, ...]
    active: tuple[ActiveSenseCandidate, ...]
    selected: ActiveSenseCandidate | None
    clarify_required: bool
    reason_key: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.status not in _W03_UNDERSTANDING_STATUSES:
            raise ValueError("understanding status 非法")
        if not isinstance(self.atom, ObjectIdentity):
            raise TypeError("resolution atom 类型非法")
        if self.context is not None and not isinstance(
                self.context, ObjectIdentity):
            raise TypeError("resolution context 类型非法")
        if self.status == W03_UNDERSTANDING_UNIQUE:
            if len(self.active) != 1 or self.selected != self.active[0]:
                raise ValueError("UNIQUE 必须且只能采用一个 active Sense")
        elif self.selected is not None:
            raise ValueError("非 UNIQUE 结果不得私选 Sense")
        if self.clarify_required != (
                self.status in {
                    W03_UNDERSTANDING_AMBIGUOUS,
                    W03_UNDERSTANDING_CLARIFY,
                }):
            raise ValueError("clarify_required 与查询状态不一致")

    def stable_key(self) -> tuple[int, ...]:
        """返回不依赖输入排列的纯整数查询结果键。"""
        context_key = () if self.context is None else self.context.stable_key()
        values = [
            1,
            _W03_STATUS_CODES[self.status],
            *_pack(self.atom.stable_key()),
            *_pack(context_key),
            len(self.candidates),
        ]
        for item in self.candidates:
            values.extend(_pack(item.sense.stable_key()))
        values.append(len(self.active))
        for item in self.active:
            values.extend(_pack(item.sense.stable_key()))
        values.extend(_pack(self.reason_key))
        return tuple(values)


@dataclass(frozen=True)
class W03EvidenceAccount:
    """一条保留 teacher record、scope 和 runtime outcome 的 Evidence 分账。"""

    teacher_record: TeacherEvidenceRecord
    candidate: ObjectIdentity
    stance: int
    observation_source: SourceRef
    scope: ScopeIdentity
    event_key: tuple[int, ...]
    outcome: CandidateLearningOutcome
    derived_supersede: bool


@dataclass(frozen=True)
class W03EvidenceApplication:
    """一个 Observation Evidence 的候选结果与 revision 退出结果。"""

    binding: W03EvidenceBinding
    accounts: tuple[W03EvidenceAccount, ...]
    before_supersede: W03SenseResolution | None
    superseded_candidates: tuple[ObjectIdentity, ...]


@dataclass(frozen=True)
class W03UnderstandingReport:
    """W03-03 内存闭环计数，不把测试执行冒充正式训练。"""

    candidate_count: int
    applied_observation_evidence_count: int
    candidate_recognition_count: int
    unbound_evidence_count: int
    active_sense_count: int
    source_conflict_candidate_count: int
    execution_state: tuple[tuple[str, int], ...]


__all__ = [
    "W03_UNDERSTANDING_AMBIGUOUS",
    "W03_UNDERSTANDING_CLARIFY",
    "W03_UNDERSTANDING_UNIQUE",
    "W03_UNDERSTANDING_UNKNOWN",
    "W03EvidenceAccount",
    "W03EvidenceApplication",
    "W03SenseResolution",
    "W03UnderstandingError",
    "W03UnderstandingReport",
]
