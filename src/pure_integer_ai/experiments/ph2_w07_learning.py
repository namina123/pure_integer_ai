"""W-07 typed operator 的共享 H-04/H-05 lifecycle 薄编排。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.candidate_projection import (
    CandidateProjectionGraph,
    CandidateProjectionProtocol,
)
from pure_integer_ai.cognition.shared.candidate_runtime import (
    CandidateEvidenceRevisionOutcome,
    CandidateLearningOutcome,
    CandidateLearningRuntime,
    CandidateProjectionMetadata,
    CandidateRecognitionRequest,
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
    EPISTEMIC_CONFLICTED,
    EPISTEMIC_REFUTED,
    EPISTEMIC_UNKNOWN,
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EVIDENCE_UNKNOWN,
    EvidenceRecord,
    LIFECYCLE_ACTIVE,
    LIFECYCLE_ARCHIVED,
    LIFECYCLE_SUPERSEDED,
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
from pure_integer_ai.cognition.shared.logic_candidate import (
    LogicOperatorCandidateSpec,
)
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.cognition.shared.typed_binding import BoundProposition
from pure_integer_ai.crosscut.determinism.hasher import Hasher
from pure_integer_ai.experiments.formal_train import make_train_context
from pure_integer_ai.experiments.logic_closure_runtime import LogicClosureRuntime
from pure_integer_ai.experiments.ph2_w05_contract import digest_value
from pure_integer_ai.experiments.ph2_w07_adapter import (
    W07EvidenceBinding,
    W07LogicProposal,
    W07TypedAdapterOutput,
    W07_IDENTITY_VERSIONS,
    W07_NAMESPACE,
    w07_logic_candidate_protocol,
)
from pure_integer_ai.experiments.ph2_w07_contract import W07_SUBSTAGE_ORDER
from pure_integer_ai.storage.edge_store import EPI_STRUCTURED, SOURCE_BARE_TEXT


_WITHDRAWAL_HASHER = Hasher("ph2.w07.withdrawal.evidence.v1")
_ARCHIVE_HASHER = Hasher("ph2.w07.archive.evidence.v1")
_COURSE_CANDIDATE_VERSIONS = VersionBundle(
    CorpusVersion(1),
    ParserVersion(1),
    PrimitiveVersion(1),
    CurriculumVersion(1),
)


class W07LearningError(RuntimeError):
    """W-07 forming、operator Evidence 或 lifecycle 无法闭合。"""


def _pack(value: tuple[int, ...]) -> tuple[int, ...]:
    return len(value), *value


def _projection_protocol() -> CandidateProjectionProtocol:
    values = tuple(
        concept_identity(
            (W07_NAMESPACE, 600 + ordinal),
            versions=W07_IDENTITY_VERSIONS,
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
        (W07_NAMESPACE, 700),
    )


def _aggregate_source() -> SourceRef:
    return SourceRef(
        W07_NAMESPACE,
        999,
        0,
        GLOBAL_OWNER_SCOPE,
        _COURSE_CANDIDATE_VERSIONS,
    )


def _candidate_engine() -> EvidenceCandidateEngine:
    aggregate = _aggregate_source()
    return EvidenceCandidateEngine(EvidenceCandidateProtocol(
        (W07_NAMESPACE, 701),
        (W07_NAMESPACE, 702),
        aggregate,
        document_scope(aggregate),
        1,
    ))


def _verifier() -> IndependentObjectVerifier:
    return IndependentObjectVerifier(IndependentVerifierProtocol(
        concept_identity(
            (W07_NAMESPACE, 703), versions=W07_IDENTITY_VERSIONS),
        (W07_NAMESPACE, 704),
        (W07_NAMESPACE, 705),
        (W07_NAMESPACE, 706),
        (W07_NAMESPACE, 707),
    ))


def _teacher_source(binding: W07EvidenceBinding, *, purpose: int = 1) -> SourceRef:
    key = binding.teacher_record.stable_key.stable_key()
    return SourceRef(
        W07_NAMESPACE,
        key[-1],
        binding.proposal.observation.logical_order * 10 + purpose,
        GLOBAL_OWNER_SCOPE,
        W07_IDENTITY_VERSIONS,
    )


def _event_key(
        binding: W07EvidenceBinding,
        spec: LogicOperatorCandidateSpec,
        *,
        stance: int,
        stance_ordinal: int,
        derived_supersede: bool,
        ) -> tuple[int, ...]:
    return (
        W07_NAMESPACE,
        2 if derived_supersede else 1,
        *_pack(binding.teacher_record.stable_key.stable_key()),
        *_pack(spec.candidate.stable_key()),
        stance,
        stance_ordinal,
    )


def _find_bound(
        root: BoundProposition,
        candidate: ObjectIdentity,
        ) -> BoundProposition:
    if root.template == candidate:
        return root
    for binding in root.bindings:
        if isinstance(binding.filler, BoundProposition):
            try:
                return _find_bound(binding.filler, candidate)
            except KeyError:
                pass
    raise KeyError("W-07 candidate 不在 proposal bound tree")


def _visible_inputs(
        proposal: W07LogicProposal,
        spec: LogicOperatorCandidateSpec,
        ) -> tuple[ObjectIdentity, ...]:
    """只投影 prediction 前可见的 typed 结构，不读取 expected content。"""
    root = _find_bound(proposal.bound_root, spec.candidate)
    values = [
        root.instruction,
        root.predicate,
        root.structure,
        root.source_anchor,
        root.context,
        spec.definition.instruction,
        spec.definition.structure,
        *(item.role for item in spec.definition.slots),
    ]

    def visit(item: BoundProposition) -> None:
        values.extend((
            item.instruction,
            item.predicate,
            item.structure,
            item.source_anchor,
            item.context,
            *item.introduced_binders,
            *item.applied_variables,
        ))
        for binding in item.bindings:
            values.append(binding.role)
            if isinstance(binding.filler, BoundProposition):
                visit(binding.filler)
            else:
                values.append(binding.filler)

    visit(root)
    unique = {item.stable_key(): item for item in values}
    return tuple(unique[key] for key in sorted(unique))


def _binding_key(binding: W07EvidenceBinding) -> tuple:
    protocol = w07_logic_candidate_protocol()
    return (
        digest_value(binding.teacher_record.to_dict()),
        digest_value(binding.proposal.observation.to_dict()),
        binding.proposal.source_binding.source_ref.stable_key(),
        binding.proposal.source_protocol.stable_key(),
        binding.proposal.bound_root.stable_key(),
        binding.proposal.request_scope.stable_key(),
        binding.stances,
        binding.content_stances,
        binding.expected_state,
        digest_value(binding.expected_payload.to_value()),
        binding.reason_key,
        (None if binding.supersedes_observation_key is None else
         binding.supersedes_observation_key.stable_key()),
        tuple(item.stable_key(protocol) for item in binding.proposal.specs),
    )


@dataclass(frozen=True)
class W07OperatorEvidenceAccount:
    """一条只裁决 operator adoption 的 H-04/H-05 Evidence。"""

    binding: W07EvidenceBinding
    spec: LogicOperatorCandidateSpec
    stance: int
    observation_source: SourceRef
    event_key: tuple[int, ...]
    outcome: CandidateLearningOutcome
    derived_supersede: bool


@dataclass(frozen=True)
class W07EvidenceApplication:
    """一个 train label 的 operator Evidence 与 reparse 退出分账。"""

    binding: W07EvidenceBinding
    accounts: tuple[W07OperatorEvidenceAccount, ...]
    superseded_candidates: tuple[ObjectIdentity, ...]
    reparse: bool


@dataclass(frozen=True)
class W07WithdrawalAccount:
    """UNKNOWN superseding Evidence 对当前 support 的 append-only 撤回。"""

    prior: W07OperatorEvidenceAccount
    withdrawal_level: int
    evidence: EvidenceRecord
    outcome: CandidateEvidenceRevisionOutcome


@dataclass(frozen=True)
class W07ArchiveAccount:
    """显式 refute revision 触发的 archive，旧 Evidence 不删除。"""

    prior: W07OperatorEvidenceAccount
    evidence: EvidenceRecord
    outcome: CandidateLearningOutcome | CandidateEvidenceRevisionOutcome
    automatic: bool


@dataclass(frozen=True)
class W07LearningResult:
    """W07-02 public bounded lifecycle 计数，不表示正式能力 PASS。"""

    candidate_count: int
    schema_rejection_count: int
    evidence_application_count: int
    operator_evidence_account_count: int
    active_operator_count: int
    refuted_candidate_count: int
    conflict_candidate_count: int
    unknown_candidate_count: int
    archived_candidate_count: int
    superseded_candidate_count: int
    reparse_count: int
    withdrawal_count: int


@dataclass(frozen=True)
class _W07RecognitionSeed:
    binding: W07EvidenceBinding
    spec: LogicOperatorCandidateSpec
    stance: int
    stance_ordinal: int
    derived_supersede: bool
    replacement: LogicOperatorCandidateSpec | None = None


@dataclass(frozen=True)
class _W07RecognitionPlan:
    seed: _W07RecognitionSeed
    observation_source: SourceRef
    event_key: tuple[int, ...]
    request: CandidateRecognitionRequest


class W07LogicLearningRuntime:
    """以单一 CandidateLearningRuntime/LogicClosureRuntime 管理七类候选。"""

    def __init__(self, backend) -> None:
        context = make_train_context(backend)
        self.projection_protocol = _projection_protocol()
        self.candidate_graph = CandidateProjectionGraph(
            context.graph_ontology, self.projection_protocol)
        self.learning = CandidateLearningRuntime(
            _candidate_engine(),
            self.candidate_graph,
            _verifier(),
            CandidateProjectionMetadata(SOURCE_BARE_TEXT, EPI_STRUCTURED),
        )
        self.logic = LogicClosureRuntime(
            self.learning, w07_logic_candidate_protocol())
        self._adapter: W07TypedAdapterOutput | None = None
        self._adapter_key: tuple[int, ...] | None = None
        self._proposals_by_observation = {}
        self._proposal_by_candidate = {}
        self._allowed_bindings = {}
        self._applications: dict[tuple[int, ...], W07EvidenceApplication] = {}
        self._withdrawals: dict[tuple[int, int], W07WithdrawalAccount] = {}
        self._archives: dict[int, W07ArchiveAccount] = {}

    def register_adapter_output(self, adapter: W07TypedAdapterOutput) -> None:
        """批量 form 全部合法 spec；schema rejection 永不进入 owner。"""
        if not isinstance(adapter, W07TypedAdapterOutput):
            raise TypeError("W-07 learning 只接受 W07TypedAdapterOutput")
        adapter_key = adapter.stable_key()
        if self._adapter is not None:
            if self._adapter_key != adapter_key:
                raise W07LearningError("W-07 runtime 不得重复登记漂移 adapter")
            return
        if adapter.protocol != self.logic.protocol:
            raise W07LearningError("W-07 adapter/runtime candidate protocol 漂移")
        if any(
                item.candidate.versions != _COURSE_CANDIDATE_VERSIONS
                for item in adapter.specs):
            raise W07LearningError("W-07 course candidate version 漂移")
        accepted = {
            item.observation.stable_key for item in adapter.proposals}
        rejected = {
            item.observation.stable_key for item in adapter.rejections}
        if accepted & rejected:
            raise W07LearningError("schema rejection 混入合法 proposal")
        requests = tuple(
            (spec, ordinal)
            for ordinal, spec in enumerate(adapter.specs)
        )
        self.logic.form_many(requests)
        self._proposals_by_observation = {
            item.observation.stable_key: item for item in adapter.proposals}
        proposal_by_candidate = {}
        for proposal in adapter.proposals:
            for spec in proposal.specs:
                prior = proposal_by_candidate.get(spec.candidate)
                if prior is not None and prior is not proposal:
                    raise W07LearningError(
                        "同一 W-07 candidate 属于多个 proposal")
                proposal_by_candidate[spec.candidate] = proposal
        self._proposal_by_candidate = proposal_by_candidate
        self._allowed_bindings = {
            item.teacher_record.stable_key.stable_key(): _binding_key(item)
            for item in adapter.evidence
        }
        self._adapter = adapter
        self._adapter_key = adapter_key

    def _plan_recognition(
            self,
            seed: _W07RecognitionSeed,
            timestamps: tuple[int, int, int],
            ) -> _W07RecognitionPlan:
        if self._adapter is None:
            raise W07LearningError("operator recognition 前必须登记 adapter")
        if seed.spec.candidate not in {
                item.candidate for item in self.logic.specs()}:
            raise W07LearningError("Evidence 引用了未登记 logic candidate")
        observation_source = _teacher_source(
            seed.binding, purpose=2 if seed.derived_supersede else 1)
        scope = document_scope(observation_source)
        event_key = _event_key(
            seed.binding,
            seed.spec,
            stance=seed.stance,
            stance_ordinal=seed.stance_ordinal,
            derived_supersede=seed.derived_supersede,
        )
        hypothesis = self.learning.hypothesis_for_candidate(
            seed.spec.candidate)
        replacement_hypothesis = (
            None if seed.replacement is None
            else self.learning.hypothesis_for_candidate(
                seed.replacement.candidate)
        )
        proposal = self._proposal_by_candidate.get(seed.spec.candidate)
        if proposal is None:
            raise W07LearningError("operator candidate 缺原始 proposal")
        revealed = RevealedObjectObservation(
            observation_source,
            scope,
            event_key,
            observation_source,
            supported_targets=(
                (seed.spec.candidate,)
                if seed.stance == EVIDENCE_SUPPORT else ()),
            refuted_targets=(
                (seed.spec.candidate,)
                if seed.stance == EVIDENCE_REFUTE else ()),
            trace=seed.binding.reason_key,
        )
        request = CandidateRecognitionRequest(
            hypothesis,
            observation_source,
            scope,
            event_key,
            _visible_inputs(proposal, seed.spec),
            seed.spec.candidate,
            revealed,
            *timestamps,
            replacement=replacement_hypothesis,
        )
        return _W07RecognitionPlan(
            seed, observation_source, event_key, request)

    def _commit_seeds(
            self,
            seeds: tuple[_W07RecognitionSeed, ...],
            ) -> tuple[W07OperatorEvidenceAccount, ...]:
        if not seeds:
            return ()
        timestamps = self.learning.next_timestamps(len(seeds) * 3)
        plans = tuple(
            self._plan_recognition(
                seed,
                timestamps[index * 3:index * 3 + 3],
            )
            for index, seed in enumerate(seeds)
        )
        outcomes = self.logic.recognize_many(tuple(
            item.request for item in plans))
        accounts = []
        for plan, outcome in zip(plans, outcomes, strict=True):
            if outcome.evidence.stance != plan.seed.stance:
                raise W07LearningError(
                    "independent verifier 返回了错误 operator stance")
            accounts.append(W07OperatorEvidenceAccount(
                plan.seed.binding,
                plan.seed.spec,
                plan.seed.stance,
                plan.observation_source,
                plan.event_key,
                outcome,
                plan.seed.derived_supersede,
            ))
        return tuple(accounts)

    @staticmethod
    def _replacement_for(
            target: LogicOperatorCandidateSpec,
            replacements: tuple[LogicOperatorCandidateSpec, ...],
            ) -> LogicOperatorCandidateSpec:
        matches = tuple(
            item for item in replacements
            if item.competition_key == target.competition_key
        )
        if len(matches) != 1:
            raise W07LearningError(
                "W-07 reparse target 缺同竞争组唯一 replacement")
        return matches[0]

    def apply_evidence(
            self, binding: W07EvidenceBinding) -> W07EvidenceApplication:
        """应用 operator Evidence；content expected 四态不写入 H-05。"""
        if self._adapter is None:
            raise W07LearningError("应用 Evidence 前必须登记 adapter")
        if not isinstance(binding, W07EvidenceBinding):
            raise TypeError("W-07 apply_evidence 需要 W07EvidenceBinding")
        route = binding.teacher_record.stable_key.stable_key()
        binding_key = _binding_key(binding)
        if self._allowed_bindings.get(route) != binding_key:
            raise W07LearningError("W-07 Evidence 不属于已登记 adapter")
        existing = self._applications.get(route)
        if existing is not None:
            if _binding_key(existing.binding) != binding_key:
                raise W07LearningError("同一 teacher route 绑定不同 Evidence")
            return existing

        seeds = []
        for spec_index, spec in enumerate(binding.proposal.specs):
            for stance_index, stance in enumerate(binding.stances):
                seeds.append(_W07RecognitionSeed(
                    binding,
                    spec,
                    stance,
                    spec_index * 10 + stance_index,
                    False,
                ))

        superseded = []
        target_key = binding.supersedes_observation_key
        if target_key is not None:
            target = self._proposals_by_observation.get(target_key)
            if target is None:
                raise W07LearningError("W-07 reparse supersede target 缺失")
            for target_index, target_spec in enumerate(target.specs):
                replacement = self._replacement_for(
                    target_spec, binding.proposal.specs)
                seeds.append(_W07RecognitionSeed(
                    binding,
                    target_spec,
                    EVIDENCE_REFUTE,
                    100 + target_index,
                    True,
                    replacement,
                ))
                superseded.append(target_spec.candidate)
        reparse = target_key is not None
        if reparse != (
                binding.proposal.observation.perturbation_kind
                == "PARSER_REVISION"):
            raise W07LearningError("W-07 reparse 标记与 supersede 链不一致")
        application = W07EvidenceApplication(
            binding,
            self._commit_seeds(tuple(seeds)),
            tuple(sorted(set(superseded), key=ObjectIdentity.stable_key)),
            reparse,
        )
        self._applications[route] = application
        return application

    def apply_all(
            self,
            adapter: W07TypedAdapterOutput,
            ) -> tuple[W07EvidenceApplication, ...]:
        """按冻结子阶段和逻辑序应用 accepted Evidence，不消费 rejection。"""
        self.register_adapter_output(adapter)
        if self._adapter_key != adapter.stable_key():
            raise W07LearningError("W-07 apply_all adapter identity 漂移")
        ordered = tuple(sorted(
                adapter.evidence,
                key=lambda item: (
                    W07_SUBSTAGE_ORDER.index(
                        item.proposal.observation.substage),
                    item.proposal.observation.logical_order,
                    item.teacher_record.stable_key.stable_key(),
                )))
        if not self._applications:
            seeds = []
            groups = []
            for binding in ordered:
                route = binding.teacher_record.stable_key.stable_key()
                binding_key = _binding_key(binding)
                if self._allowed_bindings.get(route) != binding_key:
                    raise W07LearningError(
                        "W-07 Evidence 不属于已登记 adapter")
                start = len(seeds)
                local_seeds = []
                for spec_index, spec in enumerate(binding.proposal.specs):
                    for stance_index, stance in enumerate(binding.stances):
                        local_seeds.append(_W07RecognitionSeed(
                            binding,
                            spec,
                            stance,
                            spec_index * 10 + stance_index,
                            False,
                        ))
                superseded = []
                target_key = binding.supersedes_observation_key
                if target_key is not None:
                    target = self._proposals_by_observation.get(target_key)
                    if target is None:
                        raise W07LearningError(
                            "W-07 reparse supersede target 缺失")
                    for target_index, target_spec in enumerate(target.specs):
                        replacement = self._replacement_for(
                            target_spec, binding.proposal.specs)
                        local_seeds.append(_W07RecognitionSeed(
                            binding,
                            target_spec,
                            EVIDENCE_REFUTE,
                            100 + target_index,
                            True,
                            replacement,
                        ))
                        superseded.append(target_spec.candidate)
                reparse = target_key is not None
                if reparse != (
                        binding.proposal.observation.perturbation_kind
                        == "PARSER_REVISION"):
                    raise W07LearningError(
                        "W-07 reparse 标记与 supersede 链不一致")
                seeds.extend(local_seeds)
                groups.append((
                    binding,
                    route,
                    start,
                    len(seeds),
                    tuple(sorted(
                        set(superseded), key=ObjectIdentity.stable_key)),
                    reparse,
                ))
            accounts = self._commit_seeds(tuple(seeds))
            for binding, route, start, end, superseded, reparse in groups:
                self._applications[route] = W07EvidenceApplication(
                    binding,
                    accounts[start:end],
                    superseded,
                    reparse,
                )
        else:
            for binding in ordered:
                self.apply_evidence(binding)
        return self.applications()

    def withdraw_evidence(
            self,
            account: W07OperatorEvidenceAccount,
            *,
            withdrawal_level: int,
            ) -> W07WithdrawalAccount:
        """用 UNKNOWN revision 撤回当前 support，旧 Evidence 留在历史。"""
        if not isinstance(account, W07OperatorEvidenceAccount):
            raise TypeError("W-07 withdrawal account 类型非法")
        if type(withdrawal_level) is not int or withdrawal_level not in {1, 2, 3}:
            raise ValueError("withdrawal_level 必须为 1..3 严格整数")
        if account.stance != EVIDENCE_SUPPORT or account.derived_supersede:
            raise W07LearningError("withdrawal 只能撤回普通 support Evidence")
        prior = account.outcome.evidence
        existing_levels = {
            level for evidence_id, level in self._withdrawals
            if evidence_id == prior.evidence_id
        }
        if existing_levels:
            if withdrawal_level not in existing_levels:
                raise W07LearningError("同一 Evidence 不得以不同等级重复 withdrawal")
            return self._withdrawals[(prior.evidence_id, withdrawal_level)]
        if self.logic.adoption(account.spec) is None:
            raise W07LearningError("withdrawal 前 operator 必须为 active supported")
        source = _teacher_source(account.binding, purpose=3 + withdrawal_level)
        timestamps = self.learning.next_timestamps(3)
        reason_key = (
            W07_NAMESPACE,
            3,
            *_pack(account.binding.teacher_record.stable_key.stable_key()),
            prior.evidence_id,
            withdrawal_level,
        )
        evidence_id = _WITHDRAWAL_HASHER.h63((
            prior.stable_key(), source.stable_key(), withdrawal_level,
        )) or 1
        evidence = EvidenceRecord(
            evidence_id,
            prior.hypothesis,
            EVIDENCE_UNKNOWN,
            reason_key,
            source,
            timestamps[0],
            payload=(W07_NAMESPACE, withdrawal_level),
            supersedes_evidence_id=prior.evidence_id,
        )
        outcome = self.learning.revise_evidence(
            evidence,
            resolve_timestamp_seq=timestamps[1],
            projection_timestamp_seq=timestamps[2],
        )
        if self.logic.adoption(account.spec) is not None:
            raise W07LearningError("withdrawal 后 operator active projection 未退出")
        result = W07WithdrawalAccount(
            account, withdrawal_level, evidence, outcome)
        self._withdrawals[(prior.evidence_id, withdrawal_level)] = result
        return result

    def archive_refuted(
            self,
            account: W07OperatorEvidenceAccount,
            ) -> W07ArchiveAccount:
        """以 append-only refute revision 归档纯反驳候选。"""
        if not isinstance(account, W07OperatorEvidenceAccount):
            raise TypeError("W-07 archive account 类型非法")
        if account.stance != EVIDENCE_REFUTE or account.derived_supersede:
            raise W07LearningError("archive 只接受普通 refute Evidence")
        prior = account.outcome.evidence
        existing = self._archives.get(prior.evidence_id)
        if existing is not None:
            return existing
        snapshot = self.learning.engine.ledger.snapshot(prior.hypothesis)
        if snapshot.lifecycle == LIFECYCLE_ARCHIVED:
            result = W07ArchiveAccount(
                account, prior, account.outcome, True)
            self._archives[prior.evidence_id] = result
            return result
        if (snapshot.lifecycle != LIFECYCLE_ACTIVE
                or snapshot.epistemic_status != EPISTEMIC_REFUTED):
            raise W07LearningError("archive 前 candidate 必须 active+refuted")
        source = _teacher_source(account.binding, purpose=8)
        timestamps = self.learning.next_timestamps(3)
        reason_key = (
            W07_NAMESPACE,
            4,
            *_pack(account.binding.teacher_record.stable_key.stable_key()),
            prior.evidence_id,
        )
        evidence_id = _ARCHIVE_HASHER.h63((
            prior.stable_key(), source.stable_key(),
        )) or 1
        evidence = EvidenceRecord(
            evidence_id,
            prior.hypothesis,
            EVIDENCE_REFUTE,
            reason_key,
            source,
            timestamps[0],
            payload=(W07_NAMESPACE, 4),
            supersedes_evidence_id=prior.evidence_id,
        )
        outcome = self.learning.revise_evidence(
            evidence,
            resolve_timestamp_seq=timestamps[1],
            projection_timestamp_seq=timestamps[2],
            archive_refuted=True,
        )
        archived = self.learning.engine.ledger.snapshot(prior.hypothesis)
        if archived.lifecycle != LIFECYCLE_ARCHIVED:
            raise W07LearningError("refuted operator 未进入 archived lifecycle")
        result = W07ArchiveAccount(account, evidence, outcome, False)
        self._archives[prior.evidence_id] = result
        return result

    def applications(self) -> tuple[W07EvidenceApplication, ...]:
        return tuple(
            self._applications[key] for key in sorted(self._applications))

    def proposals(self) -> tuple[W07LogicProposal, ...]:
        return () if self._adapter is None else self._adapter.proposals

    def registered_specs(self) -> tuple[LogicOperatorCandidateSpec, ...]:
        return self.logic.specs()

    def active_specs(self) -> tuple[LogicOperatorCandidateSpec, ...]:
        return tuple(
            item for item in self.logic.specs()
            if self.logic.adoption(item) is not None
        )

    def snapshot_for(self, candidate: ObjectIdentity):
        try:
            hypothesis = self.learning.hypothesis_for_candidate(candidate)
        except KeyError as error:
            raise W07LearningError("logic candidate 未登记") from error
        return self.learning.engine.ledger.snapshot(hypothesis)

    def report(self) -> W07LearningResult:
        if self._adapter is None:
            raise W07LearningError("W-07 report 前必须登记 adapter")
        snapshots = tuple(
            self.snapshot_for(item.candidate) for item in self.logic.specs())
        return W07LearningResult(
            len(self.logic.specs()),
            len(self._adapter.rejections),
            len(self._applications),
            sum(len(item.accounts) for item in self._applications.values()),
            len(self.active_specs()),
            sum(item.epistemic_status == EPISTEMIC_REFUTED
                for item in snapshots),
            sum(item.epistemic_status == EPISTEMIC_CONFLICTED
                for item in snapshots),
            sum(item.epistemic_status == EPISTEMIC_UNKNOWN
                for item in snapshots),
            sum(item.lifecycle == LIFECYCLE_ARCHIVED for item in snapshots),
            sum(item.lifecycle == LIFECYCLE_SUPERSEDED for item in snapshots),
            sum(item.reparse for item in self._applications.values()),
            len(self._withdrawals),
        )


def build_w07_learning_runtime(
        backend,
        adapter: W07TypedAdapterOutput,
        ) -> W07LogicLearningRuntime:
    """构建 public bounded shared owner 并应用 accepted operator Evidence。"""
    runtime = W07LogicLearningRuntime(backend)
    runtime.apply_all(adapter)
    return runtime


__all__ = [
    "W07ArchiveAccount",
    "W07EvidenceApplication",
    "W07LearningError",
    "W07LearningResult",
    "W07LogicLearningRuntime",
    "W07OperatorEvidenceAccount",
    "W07WithdrawalAccount",
    "build_w07_learning_runtime",
]
