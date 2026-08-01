"""W-04 primitive/surface Evidence、competition 与 lifecycle 编排。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.candidate_projection import (
    CandidateProjectionGraph,
    CandidateProjectionProtocol,
)
from pure_integer_ai.cognition.shared.candidate_runtime import (
    CandidateLearningOutcome,
    CandidateLearningRuntime,
    CandidateProjectionMetadata,
)
from pure_integer_ai.cognition.shared.candidate_verifier import (
    IndependentObjectVerifier,
    IndependentVerifierProtocol,
    RevealedObjectObservation,
)
from pure_integer_ai.cognition.shared.evidence_candidate import (
    EvidenceCandidateEngine,
    EvidenceCandidateProtocol,
)
from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EVIDENCE_UNKNOWN,
)
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    ObjectIdentity,
    SourceRef,
    concept_identity,
)
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.experiments.formal_train import make_train_context
from pure_integer_ai.experiments.ph2_w04_adapter import (
    W04EvidenceBinding,
    W04_IDENTITY_VERSIONS,
    W04PrimitiveSurfaceCandidate,
    W04TypedAdapterOutput,
    W04_NAMESPACE,
)
from pure_integer_ai.storage.edge_store import EPI_STRUCTURED, SOURCE_BARE_TEXT


class W04LearningError(RuntimeError):
    """W-04 候选生命周期、Evidence 或 supersede 处理不闭合。"""


def _pack(value: tuple[int, ...]) -> tuple[int, ...]:
    """给稳定整数键增加长度前缀。"""
    return len(value), *value


def _projection_protocol() -> CandidateProjectionProtocol:
    """建立 W-04 独占的候选 lifecycle 图协议。"""
    values = tuple(
        concept_identity((W04_NAMESPACE, 500 + ordinal))
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
        (W04_NAMESPACE, 600),
    )


def _aggregate_source() -> SourceRef:
    """建立只作 owner aggregate 的 W-04 来源。"""
    return SourceRef(
        W04_NAMESPACE,
        999,
        0,
        GLOBAL_OWNER_SCOPE,
        W04_IDENTITY_VERSIONS,
    )


def _candidate_engine() -> EvidenceCandidateEngine:
    """为 W-04 Evidence owner 建立 H-05 engine。"""
    aggregate = _aggregate_source()
    return EvidenceCandidateEngine(EvidenceCandidateProtocol(
        (W04_NAMESPACE, 701),
        (W04_NAMESPACE, 702),
        aggregate,
        document_scope(aggregate),
        1,
    ))


def _verifier() -> IndependentObjectVerifier:
    """建立只读取显式 teacher reveal 的独立三态 verifier。"""
    return IndependentObjectVerifier(IndependentVerifierProtocol(
        concept_identity((W04_NAMESPACE, 703)),
        (W04_NAMESPACE, 704),
        (W04_NAMESPACE, 705),
        (W04_NAMESPACE, 706),
        (W04_NAMESPACE, 707),
    ))


def _event_key(
        binding: W04EvidenceBinding,
        candidate: ObjectIdentity,
        *,
        stance: int,
        stance_ordinal: int,
        derived_supersede: bool,
        ) -> tuple[int, ...]:
    """从 teacher、候选、立场和用途构造 recognition event key。"""
    role = 2 if derived_supersede else 1
    return (
        W04_NAMESPACE,
        role,
        *_pack(binding.teacher_record.stable_key.stable_key()),
        *_pack(candidate.stable_key()),
        stance,
        stance_ordinal,
    )


def _teacher_source(binding: W04EvidenceBinding) -> SourceRef:
    """把 teacher record identity 投影为独立 recognition SourceRef。"""
    key = binding.teacher_record.stable_key.stable_key()
    return SourceRef(
        W04_NAMESPACE,
        key[-1],
        binding.logical_order,
        GLOBAL_OWNER_SCOPE,
        W04_IDENTITY_VERSIONS,
    )


@dataclass(frozen=True)
class W04EvidenceAccount:
    """一条保留 teacher record、scope 和 runtime outcome 的 Evidence 分账。"""

    teacher_record: object
    candidate: ObjectIdentity
    stance: int
    observation_source: SourceRef
    event_key: tuple[int, ...]
    outcome: CandidateLearningOutcome
    derived_supersede: bool


@dataclass(frozen=True)
class W04EvidenceApplication:
    """一个 Observation Evidence 的候选结果与 supersede 退出结果。"""

    binding: W04EvidenceBinding
    accounts: tuple[W04EvidenceAccount, ...]
    superseded_candidates: tuple[ObjectIdentity, ...]


@dataclass(frozen=True)
class W04LearningResult:
    """W04-02 runtime 内存闭环计数，不把测试执行冒充正式训练。"""

    candidate_count: int
    evidence_application_count: int
    account_count: int
    active_candidate_count: int
    superseded_candidate_count: int
    conflict_candidate_count: int
    unknown_candidate_count: int


class W04PrimitiveSurfaceLearningRuntime:
    """W-04 primitive/surface 候选登记、Evidence 应用和 active 查询。"""

    def __init__(self, backend) -> None:
        context = make_train_context(backend)
        self.protocol = _projection_protocol()
        self.graph = CandidateProjectionGraph(context.graph_ontology, self.protocol)
        self.learning = CandidateLearningRuntime(
            _candidate_engine(),
            self.graph,
            _verifier(),
            CandidateProjectionMetadata(SOURCE_BARE_TEXT, EPI_STRUCTURED),
        )
        self._hypotheses: dict[ObjectIdentity, object] = {}
        self._candidates: dict[ObjectIdentity, W04PrimitiveSurfaceCandidate] = {}
        self._superseded: set[ObjectIdentity] = set()
        self._applications: list[W04EvidenceApplication] = []

    def register_adapter_output(self, adapter: W04TypedAdapterOutput) -> None:
        """按 adapter 输出登记全部 W-04 候选定义。"""
        requests = []
        for ordinal, candidate in enumerate(adapter.candidates):
            self._candidates[candidate.candidate] = candidate
            requests.append((candidate.definition, ordinal * 10))
        hypotheses = self.learning.register_many(tuple(requests))
        self._hypotheses.update({
            candidate.candidate: hypothesis
            for candidate, hypothesis in zip(adapter.candidates, hypotheses, strict=True)
        })

    def apply_evidence(
            self,
            binding: W04EvidenceBinding,
            ) -> W04EvidenceApplication:
        """把一条 teacher Evidence 应用到候选，支持 supersede 和 conflict 双 stance。"""
        accounts: list[W04EvidenceAccount] = []
        superseded: list[ObjectIdentity] = []
        replacement = None
        if binding.supersedes_observation_key is not None:
            for candidate in self._candidates.values():
                if candidate.observation.stable_key == binding.supersedes_observation_key:
                    self._superseded.add(candidate.candidate)
                    superseded.append(candidate.candidate)
            if binding.candidates:
                replacement = self._hypotheses.get(binding.candidates[0])
        for candidate_id in binding.candidates:
            hypothesis = self._hypotheses.get(candidate_id)
            candidate = self._candidates.get(candidate_id)
            if hypothesis is None or candidate is None:
                raise W04LearningError("Evidence 引用未登记候选")
            teacher_source = _teacher_source(binding)
            teacher_scope = document_scope(teacher_source)
            for ordinal, stance in enumerate(binding.stances):
                event_key = _event_key(
                    binding,
                    candidate_id,
                    stance=stance,
                    stance_ordinal=ordinal,
                    derived_supersede=False,
                )
                revealed = RevealedObjectObservation(
                    teacher_source,
                    teacher_scope,
                    event_key,
                    teacher_source,
                    supported_targets=(
                        (candidate.primitive,) if stance == EVIDENCE_SUPPORT else ()
                    ),
                    refuted_targets=(
                        (candidate.primitive,) if stance == EVIDENCE_REFUTE else ()
                    ),
                    trace=binding.reason_key,
                )
                timestamps = self.learning.next_timestamps(3)
                outcome = self.learning.recognize(
                    hypothesis,
                    observation=teacher_source,
                    scope=teacher_scope,
                    event_key=event_key,
                    visible_inputs=(
                        candidate.surface_atom,
                        candidate.context,
                        candidate.primitive,
                    ),
                    predicted=candidate.primitive,
                    revealed=revealed,
                    timestamp_seq=timestamps[0],
                    resolve_timestamp_seq=timestamps[1],
                    projection_timestamp_seq=timestamps[2],
                    archive_refuted=False,
                    replacement=replacement if candidate_id in superseded else None,
                )
                accounts.append(W04EvidenceAccount(
                    binding.teacher_record,
                    candidate_id,
                    stance,
                    teacher_source,
                    event_key,
                    outcome,
                    False,
                ))
        application = W04EvidenceApplication(
            binding,
            tuple(accounts),
            tuple(sorted(set(superseded), key=ObjectIdentity.stable_key)),
        )
        self._applications.append(application)
        return application

    def apply_all(
            self,
            adapter: W04TypedAdapterOutput,
            ) -> tuple[W04EvidenceApplication, ...]:
        """按 logical_order 顺序应用全部 train Evidence。"""
        if not self._hypotheses:
            self.register_adapter_output(adapter)
        applications = []
        for binding in sorted(adapter.evidence, key=lambda item: item.logical_order):
            applications.append(self.apply_evidence(binding))
        return tuple(applications)

    def active_candidates(self) -> tuple[W04PrimitiveSurfaceCandidate, ...]:
        """返回 active supported 且未被 supersede 的候选。"""
        active = []
        for candidate_id, hypothesis in self._hypotheses.items():
            if candidate_id in self._superseded:
                continue
            if self.learning.engine.active(hypothesis) is not None:
                active.append(self._candidates[candidate_id])
        return tuple(sorted(active, key=lambda item: item.candidate.stable_key()))

    def registered_candidates(self) -> tuple[W04PrimitiveSurfaceCandidate, ...]:
        """返回全部已登记候选，不折叠同 surface 或同 primitive 竞争。"""
        return tuple(sorted(
            self._candidates.values(),
            key=lambda item: item.candidate.stable_key(),
        ))

    def superseded_candidates(self) -> tuple[W04PrimitiveSurfaceCandidate, ...]:
        """返回被 replacement Evidence 明确 supersede 的候选。"""
        return tuple(sorted(
            (self._candidates[item] for item in self._superseded),
            key=lambda item: item.candidate.stable_key(),
        ))

    def report(self) -> W04LearningResult:
        """返回候选生命周期计数。"""
        active = self.active_candidates()
        conflicts = {
            account.candidate for app in self._applications
            for account in app.accounts if account.stance == EVIDENCE_REFUTE
        }
        unknowns = {
            account.candidate for app in self._applications
            for account in app.accounts if account.stance == EVIDENCE_UNKNOWN
        }
        return W04LearningResult(
            len(self._candidates),
            len(self._applications),
            sum(len(item.accounts) for item in self._applications),
            len(active),
            len(self._superseded),
            len(conflicts),
            len(unknowns),
        )


def build_w04_learning_runtime(
        backend,
        adapter: W04TypedAdapterOutput,
        ) -> W04PrimitiveSurfaceLearningRuntime:
    """构建 runtime 并应用 adapter 中的 train Evidence。"""
    runtime = W04PrimitiveSurfaceLearningRuntime(backend)
    runtime.apply_all(adapter)
    return runtime


__all__ = [
    "W04EvidenceAccount",
    "W04EvidenceApplication",
    "W04LearningError",
    "W04LearningResult",
    "W04PrimitiveSurfaceLearningRuntime",
    "build_w04_learning_runtime",
]
