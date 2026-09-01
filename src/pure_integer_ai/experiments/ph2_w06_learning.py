"""W-06 typed relation 的 SemanticGraph、H-05、R-00 与 withdrawal 闭环。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.candidate_projection import (
    CandidateProjectionGraph,
    CandidateProjectionProtocol,
)
from pure_integer_ai.cognition.shared.candidate_runtime import (
    CandidateEvidenceRevisionOutcome,
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
from pure_integer_ai.cognition.shared.graph_ontology import (
    relation_concept_identity,
)
from pure_integer_ai.cognition.shared.hypothesis import (
    EPISTEMIC_CONFLICTED,
    EPISTEMIC_REFUTED,
    EPISTEMIC_UNKNOWN,
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EVIDENCE_UNKNOWN,
    EvidenceRecord,
    HypothesisLedger,
    LIFECYCLE_ARCHIVED,
    LIFECYCLE_SUPERSEDED,
)
from pure_integer_ai.cognition.shared.hypothesis_resolution import (
    HypothesisResolver,
)
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    ObjectIdentity,
    SourceRef,
    concept_identity,
    occurrence_identity,
)
from pure_integer_ai.cognition.shared.relation_closure import (
    ActiveRelationClosureConsumer,
)
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.cognition.shared.semantic_graph import (
    AtomicPropositionPredicates,
    SemanticGraph,
)
from pure_integer_ai.cognition.shared.training_hypothesis import (
    TrainingHypothesisEventSink,
    TrainingHypothesisHistoryProtocol,
)
from pure_integer_ai.crosscut.determinism.hasher import Hasher
from pure_integer_ai.experiments.evaluation_protocol import ProtocolKey
from pure_integer_ai.experiments.formal_train import make_train_context
from pure_integer_ai.experiments.train_context import TrainContext
from pure_integer_ai.experiments.ph2_w06_adapter import (
    W06EvidenceBinding,
    W06RelationCandidate,
    W06TypedAdapterOutput,
    W06_IDENTITY_VERSIONS,
    W06_NAMESPACE,
    w06_relation_protocol,
)
from pure_integer_ai.experiments.relation_closure_runtime import (
    RelationClosureRecognitionInput,
    RelationClosureRecognitionTrace,
    RelationClosureRuntime,
)
from pure_integer_ai.storage.edge_store import EPI_STRUCTURED, SOURCE_BARE_TEXT


_WITHDRAWAL_HASHER = Hasher("ph2.w06.withdrawal.evidence.v1")


class W06LearningError(RuntimeError):
    """W-06 relation forming、Evidence、withdrawal 或 lifecycle 不闭合。"""


def _pack(value: tuple[int, ...]) -> tuple[int, ...]:
    """给可变长稳定键增加长度前缀。"""
    return len(value), *value


def _semantic_graph(ontology) -> SemanticGraph:
    """在宿主 ontology 上安装 W-06 独立 S-00 原子命题协议。"""
    identities = tuple(
        relation_concept_identity(
            (W06_NAMESPACE, 800, ordinal),
            versions=W06_IDENTITY_VERSIONS,
        )
        for ordinal in range(1, 7)
    )
    refs = tuple(ontology.materialize(item) for item in identities)
    return SemanticGraph(ontology, AtomicPropositionPredicates(*refs))


def _projection_protocol() -> CandidateProjectionProtocol:
    """建立 W-06 独占的 H-05 lifecycle 图字段协议。"""
    values = tuple(
        concept_identity(
            (W06_NAMESPACE, 500 + ordinal),
            versions=W06_IDENTITY_VERSIONS,
        )
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
        (W06_NAMESPACE, 600),
    )


def _aggregate_source() -> SourceRef:
    """建立只作为 W-06 H-05 aggregate owner 的非 forming 来源。"""
    return SourceRef(
        W06_NAMESPACE,
        999,
        0,
        GLOBAL_OWNER_SCOPE,
        W06_IDENTITY_VERSIONS,
    )


def _candidate_engine() -> EvidenceCandidateEngine:
    """建立 minimum forming source 为一的现役 H-05 engine。"""
    aggregate = _aggregate_source()
    return EvidenceCandidateEngine(EvidenceCandidateProtocol(
        (W06_NAMESPACE, 701),
        (W06_NAMESPACE, 702),
        aggregate,
        document_scope(aggregate),
        1,
    ))


def _history_protocol() -> TrainingHypothesisHistoryProtocol:
    """声明 W-06 H-00/H-04 在正式训练 Core 中的独立历史边界。"""
    aggregate = _aggregate_source()
    return TrainingHypothesisHistoryProtocol(
        (W06_NAMESPACE, 708),
        (W06_NAMESPACE, 701),
        aggregate,
        document_scope(aggregate),
    )


def _verifier() -> IndependentObjectVerifier:
    """建立只消费显式 train reveal 的 W-06 独立三态 verifier。"""
    return IndependentObjectVerifier(IndependentVerifierProtocol(
        concept_identity(
            (W06_NAMESPACE, 703), versions=W06_IDENTITY_VERSIONS),
        (W06_NAMESPACE, 704),
        (W06_NAMESPACE, 705),
        (W06_NAMESPACE, 706),
        (W06_NAMESPACE, 707),
    ))


def _teacher_source(binding: W06EvidenceBinding) -> SourceRef:
    """把 teacher record 全键映射为独立 recognition SourceRef。"""
    key = binding.teacher_record.stable_key.stable_key()
    return SourceRef(
        W06_NAMESPACE,
        key[-1],
        binding.logical_order,
        GLOBAL_OWNER_SCOPE,
        W06_IDENTITY_VERSIONS,
    )


def _event_key(
        binding: W06EvidenceBinding,
        candidate: ObjectIdentity,
        *,
        stance: int,
        stance_ordinal: int,
        derived_supersede: bool,
        ) -> tuple[int, ...]:
    """从 teacher、candidate、stance 和 lifecycle 用途构造幂等事件键。"""
    return (
        W06_NAMESPACE,
        2 if derived_supersede else 1,
        *_pack(binding.teacher_record.stable_key.stable_key()),
        *_pack(candidate.stable_key()),
        stance,
        stance_ordinal,
    )


def _visible_inputs(
        candidate: W06RelationCandidate,
        observation_anchor: ObjectIdentity,
        ) -> tuple[ObjectIdentity, ...]:
    """返回 prediction 前可见的 typed relation 结构，不读取 expected payload。"""
    proposition = candidate.proposition
    values = [
        observation_anchor,
        proposition.predicate,
        candidate.schema.schema,
        proposition.source_anchor,
        proposition.context,
        *(item.identity for item in candidate.endpoints),
        *(item.role for item in proposition.canonical_bindings()),
        *(item.identity_for(proposition.proposition)
          for item in proposition.canonical_bindings()),
    ]
    unique = {item.stable_key(): item for item in values}
    return tuple(unique[key] for key in sorted(unique))


@dataclass(frozen=True)
class W06EvidenceAccount:
    """一条 teacher 或 derived supersede Evidence 的 R-00 完整结果。"""

    binding: W06EvidenceBinding
    candidate: ObjectIdentity
    stance: int
    observation_source: SourceRef
    event_key: tuple[int, ...]
    trace: RelationClosureRecognitionTrace
    derived_supersede: bool


@dataclass(frozen=True)
class W06EvidenceApplication:
    """一个 train Observation 的普通 Evidence 与 reparse supersede 分账。"""

    binding: W06EvidenceBinding
    accounts: tuple[W06EvidenceAccount, ...]
    superseded_candidates: tuple[ObjectIdentity, ...]
    reparse: bool


@dataclass(frozen=True)
class W06WithdrawalAccount:
    """一次显式 withdrawal 对旧 Evidence 的 append-only revision 记录。"""

    prior: W06EvidenceAccount
    withdrawal_level: int
    evidence: EvidenceRecord
    outcome: CandidateEvidenceRevisionOutcome


@dataclass(frozen=True)
class W06LearningResult:
    """W06-02 内存闭环计数，不表示正式训练或 relation bearing PASS。"""

    candidate_count: int
    schema_rejection_count: int
    relation_family_count: int
    evidence_application_count: int
    evidence_account_count: int
    active_candidate_count: int
    archived_candidate_count: int
    superseded_candidate_count: int
    conflict_candidate_count: int
    unknown_candidate_count: int
    reparse_count: int
    withdrawal_count: int


class W06RelationLearningRuntime:
    """把 W-06 adapter 输出接入 SemanticGraph、H-05、H-04 和 R-00。"""

    def __init__(self, backend, *, context: TrainContext | None = None) -> None:
        """建立 W-06 owner；正式训练可注入现有 TrainContext。\n\n        注入 context 时所有 SemanticGraph、候选投影和 H-05 状态都绑定到
        正式训练正在使用的同一 SQLite 图，避免形成 evaluator-local 的旁路库。
        未注入时保留既有 public bounded runtime 行为。
        """
        if context is None:
            context = make_train_context(backend)
        elif not isinstance(context, TrainContext):
            raise TypeError("W-06 context 必须是 TrainContext")
        if context.backend is not backend:
            raise ValueError("W-06 context 必须绑定传入 backend")
        self.semantic_graph = _semantic_graph(context.graph_ontology)
        self.projection_protocol = _projection_protocol()
        self.candidate_graph = CandidateProjectionGraph(
            context.graph_ontology, self.projection_protocol)
        history = context.training_candidate_history
        if history is None:
            raise W06LearningError("W-06 正式 owner 缺少 Core 训练候选历史")
        history_protocol = _history_protocol()
        sink = TrainingHypothesisEventSink(history, history_protocol)
        verifier = _verifier()
        metadata = CandidateProjectionMetadata(
            SOURCE_BARE_TEXT, EPI_STRUCTURED)
        if sink.hypotheses():
            self.learning = CandidateLearningRuntime.restore_for_training_graph(
                _candidate_engine().protocol,
                self.candidate_graph,
                verifier,
                metadata,
                history,
                history_protocol,
            )
        else:
            ledger = HypothesisLedger(sink)
            resolver = HypothesisResolver(ledger, sink=sink)
            engine = EvidenceCandidateEngine(
                _candidate_engine().protocol,
                ledger=ledger,
                resolver=resolver,
            )
            self.learning = CandidateLearningRuntime(
                engine,
                self.candidate_graph,
                verifier,
                metadata,
            )
        self.relation_protocol = w06_relation_protocol()
        self.consumer: ActiveRelationClosureConsumer | None = None
        self.closure: RelationClosureRuntime | None = None
        self._adapter: W06TypedAdapterOutput | None = None
        self._candidates: dict[ObjectIdentity, W06RelationCandidate] = {}
        self._applications: dict[tuple[int, ...], W06EvidenceApplication] = {}
        self._withdrawals: dict[
            tuple[int, int], W06WithdrawalAccount] = {}

    def register_adapter_output(self, adapter: W06TypedAdapterOutput) -> None:
        """预检后物化 50 个合法命题并幂等登记 relation candidates。"""
        if not isinstance(adapter, W06TypedAdapterOutput):
            raise TypeError("W-06 learning 只接受 W06TypedAdapterOutput")
        if self._adapter is not None:
            if self._adapter != adapter:
                raise W06LearningError("W-06 runtime 不得重复登记漂移 adapter")
            return
        if ({item.proposition.proposition for item in adapter.candidates}
                & {item.proposition for item in adapter.rejections}):
            raise W06LearningError("schema rejection 不得进入合法 candidate 集")

        self.consumer = ActiveRelationClosureConsumer(
            self.semantic_graph,
            self.candidate_graph,
            self.relation_protocol,
            adapter.schemas,
            engine=self.learning.engine,
        )
        self.closure = RelationClosureRuntime(
            self.learning,
            self.semantic_graph,
            self.consumer,
            self.relation_protocol,
        )
        requests = tuple(
            (candidate.spec, ordinal)
            for ordinal, candidate in enumerate(adapter.candidates)
        )
        self.closure.preflight_many(requests, ())
        for candidate in adapter.candidates:
            self.semantic_graph.define_atomic(
                candidate.proposition,
                scope=document_scope(candidate.source_ref),
                provenance_kind=SOURCE_BARE_TEXT,
                epistemic_origin=EPI_STRUCTURED,
            )
            self._candidates[candidate.proposition.proposition] = candidate
        self.closure.form_many(requests)
        self._adapter = adapter

    def _recognize(
            self,
            binding: W06EvidenceBinding,
            candidate_id: ObjectIdentity,
            *,
            stance: int,
            stance_ordinal: int,
            derived_supersede: bool,
            replacement: ObjectIdentity | None = None,
            ) -> W06EvidenceAccount:
        """执行一次来源化 prediction、独立 reveal、H-04 和 relation projection。"""
        if self.closure is None:
            raise W06LearningError("relation recognition 前必须登记 adapter")
        candidate = self._candidates.get(candidate_id)
        if candidate is None:
            raise W06LearningError("Evidence 引用了未登记 relation candidate")
        teacher_source = _teacher_source(binding)
        scope = document_scope(teacher_source)
        event_key = _event_key(
            binding,
            candidate_id,
            stance=stance,
            stance_ordinal=stance_ordinal,
            derived_supersede=derived_supersede,
        )
        anchor = occurrence_identity(
            teacher_source, start=0, end=1, ordinal=0)
        revealed = RevealedObjectObservation(
            teacher_source,
            scope,
            event_key,
            teacher_source,
            supported_targets=(
                (candidate_id,) if stance == EVIDENCE_SUPPORT else ()),
            refuted_targets=(
                (candidate_id,) if stance == EVIDENCE_REFUTE else ()),
            trace=binding.reason_key,
        )
        input_value = RelationClosureRecognitionInput(
            candidate_id,
            teacher_source,
            scope,
            ProtocolKey((
                W06_NAMESPACE,
                900,
                *_pack(binding.teacher_record.stable_key.stable_key()),
                stance,
                stance_ordinal,
            )),
            event_key,
            anchor,
            _visible_inputs(candidate, anchor),
            revealed,
            archive_refuted=False,
            replacement=replacement,
        )
        trace = self.closure.recognize(input_value)
        return W06EvidenceAccount(
            binding,
            candidate_id,
            stance,
            teacher_source,
            event_key,
            trace,
            derived_supersede,
        )

    def apply_evidence(
            self, binding: W06EvidenceBinding) -> W06EvidenceApplication:
        """应用一条 teacher Evidence，并把 parser revision 写成真实 supersede。"""
        if self._adapter is None:
            raise W06LearningError("应用 Evidence 前必须登记 adapter")
        route = binding.teacher_record.stable_key.stable_key()
        existing = self._applications.get(route)
        if existing is not None:
            if existing.binding != binding:
                raise W06LearningError("同一 teacher route 绑定不同 Evidence")
            return existing

        accounts = []
        superseded = []
        for ordinal, stance in enumerate(binding.stances):
            accounts.append(self._recognize(
                binding,
                binding.candidate,
                stance=stance,
                stance_ordinal=ordinal,
                derived_supersede=False,
            ))
        if binding.supersedes_observation_key is not None:
            targets = tuple(sorted(
                (
                    item.proposition.proposition
                    for item in self._candidates.values()
                    if item.observation.stable_key
                    == binding.supersedes_observation_key
                ),
                key=ObjectIdentity.stable_key,
            ))
            if not targets or binding.candidate not in self._candidates:
                raise W06LearningError("reparse supersede target 或 replacement 缺失")
            for ordinal, target in enumerate(targets):
                accounts.append(self._recognize(
                    binding,
                    target,
                    stance=EVIDENCE_REFUTE,
                    stance_ordinal=ordinal,
                    derived_supersede=True,
                    replacement=binding.candidate,
                ))
                superseded.append(target)
        reparse = binding.supersedes_observation_key is not None
        if reparse != (binding.observation.perturbation_kind == "PARSER_REVISION"):
            raise W06LearningError("W-06 reparse 标记与 supersede 链不一致")
        application = W06EvidenceApplication(
            binding,
            tuple(accounts),
            tuple(sorted(set(superseded), key=ObjectIdentity.stable_key)),
            reparse,
        )
        self._applications[route] = application
        return application

    def apply_all(
            self, adapter: W06TypedAdapterOutput,
            ) -> tuple[W06EvidenceApplication, ...]:
        """按全局稳定顺序应用全部 accepted train Evidence，不消费 rejection。"""
        self.register_adapter_output(adapter)
        if self._adapter != adapter:
            raise W06LearningError("W-06 apply_all adapter identity 漂移")
        for binding in sorted(
                adapter.evidence,
                key=lambda item: (
                    item.logical_order,
                    item.teacher_record.stable_key.stable_key(),
                )):
            self.apply_evidence(binding)
        return self.applications()

    def withdraw_evidence(
            self,
            account: W06EvidenceAccount,
            *,
            withdrawal_level: int,
            ) -> W06WithdrawalAccount:
        """以 UNKNOWN superseding Evidence 撤回当前 active support，并重算投影。"""
        if not isinstance(account, W06EvidenceAccount):
            raise TypeError("withdrawal account 类型非法")
        if type(withdrawal_level) is not int or withdrawal_level not in {1, 2, 3}:
            raise ValueError("withdrawal_level 必须为 1..3 严格整数")
        if account.stance != EVIDENCE_SUPPORT or account.derived_supersede:
            raise W06LearningError("withdrawal 只能撤回普通 support Evidence")
        prior = account.trace.outcome.evidence
        existing_levels = {
            level for evidence_id, level in self._withdrawals
            if evidence_id == prior.evidence_id
        }
        if existing_levels:
            if withdrawal_level not in existing_levels:
                raise W06LearningError("同一 Evidence 不得以不同等级重复 withdrawal")
            return self._withdrawals[(prior.evidence_id, withdrawal_level)]
        if self.learning.engine.active(prior.hypothesis) is None:
            raise W06LearningError("withdrawal 前 candidate 必须是 active supported")

        source = SourceRef(
            W06_NAMESPACE,
            account.binding.teacher_record.stable_key.stable_key()[-1],
            withdrawal_level,
            GLOBAL_OWNER_SCOPE,
            W06_IDENTITY_VERSIONS,
        )
        timestamps = self.learning.next_timestamps(3)
        reason_key = (
            W06_NAMESPACE,
            3,
            *_pack(account.binding.teacher_record.stable_key.stable_key()),
            prior.evidence_id,
            withdrawal_level,
        )
        evidence_id = _WITHDRAWAL_HASHER.h63((
            prior.stable_key(),
            source.stable_key(),
            withdrawal_level,
        )) or 1
        evidence = EvidenceRecord(
            evidence_id,
            prior.hypothesis,
            EVIDENCE_UNKNOWN,
            reason_key,
            source,
            timestamps[0],
            payload=(W06_NAMESPACE, withdrawal_level),
            supersedes_evidence_id=prior.evidence_id,
        )
        outcome = self.learning.revise_evidence(
            evidence,
            resolve_timestamp_seq=timestamps[1],
            projection_timestamp_seq=timestamps[2],
        )
        if self.consumer is None or self.consumer.lookup_proposition(
                account.candidate):
            raise W06LearningError("withdrawal 后 active projection 未消失")
        result = W06WithdrawalAccount(
            account,
            withdrawal_level,
            evidence,
            outcome,
        )
        self._withdrawals[(prior.evidence_id, withdrawal_level)] = result
        return result

    def applications(self) -> tuple[W06EvidenceApplication, ...]:
        """按 teacher 全键返回全部幂等 Evidence application。"""
        return tuple(
            self._applications[key]
            for key in sorted(self._applications)
        )

    def registered_candidates(self) -> tuple[W06RelationCandidate, ...]:
        """返回全部已登记合法候选，不含 schema rejection。"""
        return tuple(
            self._candidates[key]
            for key in sorted(self._candidates, key=ObjectIdentity.stable_key)
        )

    def active_candidates(self) -> tuple[W06RelationCandidate, ...]:
        """返回当前 active supported 且有 R-00 图投影的 relation candidates。"""
        if self.consumer is None:
            return ()
        return tuple(sorted(
            (
                candidate
                for candidate in self._candidates.values()
                if self.consumer.lookup_proposition(
                    candidate.proposition.proposition)
            ),
            key=lambda item: item.proposition.proposition.stable_key(),
        ))

    def snapshot_for(self, candidate: ObjectIdentity):
        """返回一个已登记 candidate 的 R-00 当前认识论快照。"""
        if self.closure is None or candidate not in self._candidates:
            raise W06LearningError("relation candidate 未登记")
        return self.closure.snapshot_for_proposition(candidate)

    def report(self) -> W06LearningResult:
        """按 H-00 lifecycle 与 epistemic 四态派生 W06-02 有界计数。"""
        if self._adapter is None:
            raise W06LearningError("W-06 report 前必须登记 adapter")
        snapshots = tuple(
            self.snapshot_for(item) for item in sorted(
                self._candidates, key=ObjectIdentity.stable_key)
        )
        return W06LearningResult(
            len(self._candidates),
            len(self._adapter.rejections),
            len({item.relation_family for item in self._candidates.values()}),
            len(self._applications),
            sum(len(item.accounts) for item in self._applications.values()),
            len(self.active_candidates()),
            sum(item.snapshot.lifecycle == LIFECYCLE_ARCHIVED for item in snapshots),
            sum(item.snapshot.lifecycle == LIFECYCLE_SUPERSEDED for item in snapshots),
            sum(
                item.snapshot.epistemic_status == EPISTEMIC_CONFLICTED
                for item in snapshots),
            sum(
                item.snapshot.epistemic_status == EPISTEMIC_UNKNOWN
                for item in snapshots),
            sum(item.reparse for item in self._applications.values()),
            len(self._withdrawals),
        )


def build_w06_learning_runtime(
        backend,
        adapter: W06TypedAdapterOutput,
        *,
        context: TrainContext | None = None,
        ) -> W06RelationLearningRuntime:
    """构建 public bounded runtime 并应用 accepted train Evidence。"""
    runtime = W06RelationLearningRuntime(backend, context=context)
    runtime.apply_all(adapter)
    return runtime


__all__ = [
    "W06EvidenceAccount",
    "W06EvidenceApplication",
    "W06LearningError",
    "W06LearningResult",
    "W06RelationLearningRuntime",
    "W06WithdrawalAccount",
    "build_w06_learning_runtime",
]
