"""W-05 原子命题 Evidence、candidate graph 与 supersede 生命周期。"""
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
from pure_integer_ai.experiments.ph2_dataset_contract import TeacherEvidenceRecord
from pure_integer_ai.experiments.ph2_w05_adapter import (
    W05AtomicPropositionCandidate,
    W05EvidenceBinding,
    W05TypedAdapterOutput,
    W05_ATOMIC_IDENTITY_VERSIONS,
    W05_IDENTITY_VERSIONS,
    W05_NAMESPACE,
)
from pure_integer_ai.storage.edge_store import EPI_STRUCTURED, SOURCE_BARE_TEXT


class W05LearningError(RuntimeError):
    """W-05 原子命题 Evidence 或 supersede 生命周期不闭合。"""


def _pack(value: tuple[int, ...]) -> tuple[int, ...]:
    """给完整稳定键增加长度前缀。"""
    return len(value), *value


def _projection_protocol() -> CandidateProjectionProtocol:
    """建立 W-05 独立候选 lifecycle 图协议。"""
    values = tuple(
        concept_identity((W05_NAMESPACE, 500 + ordinal))
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
        (W05_NAMESPACE, 600),
    )


def _aggregate_source() -> SourceRef:
    """建立只作 W-05 candidate aggregate 的来源。"""
    return SourceRef(
        W05_NAMESPACE,
        999,
        0,
        GLOBAL_OWNER_SCOPE,
        W05_ATOMIC_IDENTITY_VERSIONS,
    )


def _candidate_engine() -> EvidenceCandidateEngine:
    """为 W-05 Evidence owner 建立现役 H-05 engine。"""
    aggregate = _aggregate_source()
    return EvidenceCandidateEngine(EvidenceCandidateProtocol(
        (W05_NAMESPACE, 701),
        (W05_NAMESPACE, 702),
        aggregate,
        document_scope(aggregate),
        1,
    ))


def _verifier() -> IndependentObjectVerifier:
    """建立只读取 train TeacherEvidence reveal 的独立三态 verifier。"""
    return IndependentObjectVerifier(IndependentVerifierProtocol(
        concept_identity((W05_NAMESPACE, 703)),
        (W05_NAMESPACE, 704),
        (W05_NAMESPACE, 705),
        (W05_NAMESPACE, 706),
        (W05_NAMESPACE, 707),
    ))


def _teacher_source(binding: W05EvidenceBinding) -> SourceRef:
    """把 teacher record identity 投影为独立 recognition SourceRef。"""
    key = binding.teacher_record.stable_key.stable_key()
    return SourceRef(
        W05_NAMESPACE,
        key[-1],
        binding.logical_order,
        GLOBAL_OWNER_SCOPE,
        W05_IDENTITY_VERSIONS,
    )


def _event_key(
        binding: W05EvidenceBinding,
        candidate: ObjectIdentity,
        *,
        stance: int,
        stance_ordinal: int,
        derived_supersede: bool,
        ) -> tuple[int, ...]:
    """从 teacher、候选、stance 和用途构造唯一 recognition event。"""
    return (
        W05_NAMESPACE,
        2 if derived_supersede else 1,
        *_pack(binding.teacher_record.stable_key.stable_key()),
        *_pack(candidate.stable_key()),
        stance,
        stance_ordinal,
    )


def _visible_inputs(
        candidate: W05AtomicPropositionCandidate,
        ) -> tuple[ObjectIdentity, ...]:
    """返回 prediction 前可见的一等结构输入，不读取 teacher expected payload。"""
    definition = candidate.proposition_definition
    values = [
        definition.predicate,
        definition.source_anchor,
        definition.context,
        *(item.identity for item in candidate.occurrences),
        *(item.semantic_object for item in candidate.occurrences),
        *(item.identity_for(definition.proposition)
          for item in definition.canonical_bindings()),
    ]
    unique = {item.stable_key(): item for item in values}
    return tuple(unique[key] for key in sorted(unique))


@dataclass(frozen=True)
class W05EvidenceAccount:
    """一条 teacher/derived Evidence 与 runtime outcome 的独立分账。"""

    teacher_record: TeacherEvidenceRecord
    candidate: ObjectIdentity
    stance: int
    observation_source: SourceRef
    event_key: tuple[int, ...]
    outcome: CandidateLearningOutcome
    derived_supersede: bool


@dataclass(frozen=True)
class W05EvidenceApplication:
    """一个 train Observation 的普通 Evidence 与派生 supersede 退出结果。"""

    binding: W05EvidenceBinding
    accounts: tuple[W05EvidenceAccount, ...]
    superseded_candidates: tuple[ObjectIdentity, ...]


@dataclass(frozen=True)
class W05LearningResult:
    """W05-02 内存闭环计数，不把专项测试冒充正式训练。"""

    candidate_count: int
    evidence_application_count: int
    account_count: int
    active_candidate_count: int
    superseded_candidate_count: int
    conflict_candidate_count: int
    unknown_candidate_count: int
    occurrence_count: int
    role_binding_count: int


class W05AtomicPropositionLearningRuntime:
    """W-05 Proposition 候选登记、Evidence 应用和 active 查询。"""

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
        self._candidates: dict[
            ObjectIdentity, W05AtomicPropositionCandidate] = {}
        self._superseded: set[ObjectIdentity] = set()
        self._applications: list[W05EvidenceApplication] = []

    def register_adapter_output(self, adapter: W05TypedAdapterOutput) -> None:
        """按 adapter 输出登记全部 W-05 原子命题候选。"""
        if self._hypotheses:
            raise W05LearningError("W-05 adapter output 不得重复登记")
        requests = []
        for ordinal, candidate in enumerate(adapter.candidates):
            if candidate.candidate in self._candidates:
                raise W05LearningError("W-05 candidate identity 重复")
            self._candidates[candidate.candidate] = candidate
            requests.append((candidate.definition, ordinal * 10))
        hypotheses = self.learning.register_many(tuple(requests))
        self._hypotheses.update({
            candidate.candidate: hypothesis
            for candidate, hypothesis in zip(
                adapter.candidates, hypotheses, strict=True)
        })

    def _recognize(
            self,
            binding: W05EvidenceBinding,
            candidate_id: ObjectIdentity,
            *,
            stance: int,
            stance_ordinal: int,
            derived_supersede: bool,
            replacement=None,
            ) -> W05EvidenceAccount:
        """执行一次 prediction→independent reveal→H-04→graph 同步。"""
        hypothesis = self._hypotheses.get(candidate_id)
        candidate = self._candidates.get(candidate_id)
        if hypothesis is None or candidate is None:
            raise W05LearningError("Evidence 引用未登记 Proposition 候选")
        teacher_source = _teacher_source(binding)
        teacher_scope = document_scope(teacher_source)
        event_key = _event_key(
            binding,
            candidate_id,
            stance=stance,
            stance_ordinal=stance_ordinal,
            derived_supersede=derived_supersede,
        )
        revealed = RevealedObjectObservation(
            teacher_source,
            teacher_scope,
            event_key,
            teacher_source,
            supported_targets=(
                (candidate_id,) if stance == EVIDENCE_SUPPORT else ()
            ),
            refuted_targets=(
                (candidate_id,) if stance == EVIDENCE_REFUTE else ()
            ),
            trace=binding.reason_key,
        )
        timestamps = self.learning.next_timestamps(3)
        outcome = self.learning.recognize(
            hypothesis,
            observation=teacher_source,
            scope=teacher_scope,
            event_key=event_key,
            visible_inputs=_visible_inputs(candidate),
            predicted=candidate_id,
            revealed=revealed,
            timestamp_seq=timestamps[0],
            resolve_timestamp_seq=timestamps[1],
            projection_timestamp_seq=timestamps[2],
            archive_refuted=False,
            replacement=replacement,
        )
        return W05EvidenceAccount(
            binding.teacher_record,
            candidate_id,
            stance,
            teacher_source,
            event_key,
            outcome,
            derived_supersede,
        )

    def apply_evidence(
            self,
            binding: W05EvidenceBinding,
            ) -> W05EvidenceApplication:
        """应用 teacher Evidence，并把 omission→restore 记为真实 supersede。"""
        if not self._hypotheses:
            raise W05LearningError("应用 Evidence 前必须登记 W-05 candidates")
        accounts: list[W05EvidenceAccount] = []
        superseded: list[ObjectIdentity] = []
        replacement = (
            self._hypotheses.get(binding.candidates[0])
            if binding.candidates else None
        )
        if binding.supersedes_observation_key is not None:
            targets = tuple(sorted(
                (
                    candidate.candidate
                    for candidate in self._candidates.values()
                    if candidate.observation.stable_key
                    == binding.supersedes_observation_key
                ),
                key=ObjectIdentity.stable_key,
            ))
            if not targets or replacement is None:
                raise W05LearningError("supersede target 或 replacement candidate 缺失")
            for ordinal, target in enumerate(targets):
                accounts.append(self._recognize(
                    binding,
                    target,
                    stance=EVIDENCE_REFUTE,
                    stance_ordinal=ordinal,
                    derived_supersede=True,
                    replacement=replacement,
                ))
                self._superseded.add(target)
                superseded.append(target)
        for candidate_id in binding.candidates:
            for ordinal, stance in enumerate(binding.stances):
                accounts.append(self._recognize(
                    binding,
                    candidate_id,
                    stance=stance,
                    stance_ordinal=ordinal,
                    derived_supersede=False,
                ))
        application = W05EvidenceApplication(
            binding,
            tuple(accounts),
            tuple(sorted(set(superseded), key=ObjectIdentity.stable_key)),
        )
        self._applications.append(application)
        return application

    def apply_all(
            self,
            adapter: W05TypedAdapterOutput,
            ) -> tuple[W05EvidenceApplication, ...]:
        """按 logical_order 应用全部 train Evidence。"""
        if not self._hypotheses:
            self.register_adapter_output(adapter)
        applications = []
        for binding in sorted(adapter.evidence, key=lambda item: item.logical_order):
            applications.append(self.apply_evidence(binding))
        return tuple(applications)

    def active_candidates(self) -> tuple[W05AtomicPropositionCandidate, ...]:
        """返回 active supported 且未被 supersede 的 Proposition 候选。"""
        active = []
        for candidate_id, hypothesis in self._hypotheses.items():
            if candidate_id in self._superseded:
                continue
            if self.learning.engine.active(hypothesis) is not None:
                active.append(self._candidates[candidate_id])
        return tuple(sorted(active, key=lambda item: item.candidate.stable_key()))

    def registered_candidates(self) -> tuple[W05AtomicPropositionCandidate, ...]:
        """返回全部已登记候选，不折叠同 surface 的 occurrence。"""
        return tuple(sorted(
            self._candidates.values(),
            key=lambda item: item.candidate.stable_key(),
        ))

    def superseded_candidates(self) -> tuple[W05AtomicPropositionCandidate, ...]:
        """返回被 restore Evidence 明确 supersede 的 omission 候选。"""
        return tuple(sorted(
            (self._candidates[item] for item in self._superseded),
            key=lambda item: item.candidate.stable_key(),
        ))

    def applications(self) -> tuple[W05EvidenceApplication, ...]:
        """返回全部已应用 Evidence，供独立 consumer 只读投影。"""
        return tuple(self._applications)

    def hypothesis_for(self, candidate: ObjectIdentity):
        """返回候选的 H-05 Hypothesis，未知候选 fail closed。"""
        try:
            return self._hypotheses[candidate]
        except KeyError as exc:
            raise W05LearningError("candidate 未登记") from exc

    def report(self) -> W05LearningResult:
        """返回 occurrence、RoleBinding 和 Evidence 生命周期计数。"""
        stance_by_candidate: dict[ObjectIdentity, set[int]] = {}
        for application in self._applications:
            for account in application.accounts:
                if account.derived_supersede:
                    continue
                stance_by_candidate.setdefault(account.candidate, set()).add(
                    account.stance)
        conflicts = sum(
            1 for values in stance_by_candidate.values()
            if EVIDENCE_SUPPORT in values and EVIDENCE_REFUTE in values
        )
        unknowns = sum(
            1 for values in stance_by_candidate.values()
            if EVIDENCE_UNKNOWN in values
        )
        return W05LearningResult(
            len(self._candidates),
            len(self._applications),
            sum(len(item.accounts) for item in self._applications),
            len(self.active_candidates()),
            len(self._superseded),
            conflicts,
            unknowns,
            sum(len(item.occurrences) for item in self._candidates.values()),
            sum(
                len(item.proposition_definition.bindings)
                for item in self._candidates.values()
            ),
        )


def build_w05_learning_runtime(
        backend,
        adapter: W05TypedAdapterOutput,
        ) -> W05AtomicPropositionLearningRuntime:
    """构建 runtime 并应用 adapter 中唯一的 train TeacherEvidence 集。"""
    runtime = W05AtomicPropositionLearningRuntime(backend)
    runtime.apply_all(adapter)
    return runtime


__all__ = [
    "W05AtomicPropositionLearningRuntime",
    "W05EvidenceAccount",
    "W05EvidenceApplication",
    "W05LearningError",
    "W05LearningResult",
    "build_w05_learning_runtime",
]
