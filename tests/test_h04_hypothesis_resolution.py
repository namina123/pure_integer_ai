"""H-04 resolver 两轴状态、保守比较、退出和审计链对抗测试。"""
from __future__ import annotations

import pytest

from pure_integer_ai.cognition.shared.description_length import (
    DescriptionCandidate,
    DescriptionEncoding,
    DescriptionFragment,
    DescriptionLengthEngine,
    DescriptionLengthProblem,
    DescriptionModel,
    DescriptionObservation,
    DescriptionTerm,
)
from pure_integer_ai.cognition.shared.description_resolution import (
    DescriptionLengthResolverScorer,
)
from pure_integer_ai.cognition.shared.hypothesis import (
    EPISTEMIC_CONFLICTED,
    EPISTEMIC_REFUTED,
    EPISTEMIC_SUPPORTED,
    EPISTEMIC_UNKNOWN,
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EvidenceRecord,
    HypothesisKey,
    HypothesisLedger,
    LIFECYCLE_ACTIVE,
    LIFECYCLE_ARCHIVED,
    LIFECYCLE_SUPERSEDED,
)
from pure_integer_ai.cognition.shared.hypothesis_resolution import (
    ArchiveDirective,
    HypothesisResolver,
    PREFERENCE_EQUIVALENT,
    PREFERENCE_INCOMPARABLE,
    PREFERENCE_LEFT_BETTER,
    PREFERENCE_RIGHT_BETTER,
    RESOLUTION_ADOPTED,
    RESOLUTION_EXITED,
    RESOLUTION_RETAINED,
    ReplacementDirective,
    ResolverDecision,
    ResolverPreference,
)
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    OBJECT_REPRESENTATION,
    OBJECT_STRUCTURE_CONCEPT,
    ObjectIdentity,
    SourceRef,
    VersionBundle,
)
from pure_integer_ai.cognition.shared.perturbation import (
    ASSESSMENT_CHANGED,
    PerturbationAssessment,
    PerturbationEngine,
    PerturbationProtocol,
    build_permutation_trace,
)
from pure_integer_ai.cognition.shared.scope_identity import document_scope


def _source(document_id: int = 1, *, source_id: int = 1500) -> SourceRef:
    """构造带完整 owner/version 的测试来源。"""
    return SourceRef(
        1,
        source_id,
        document_id,
        GLOBAL_OWNER_SCOPE,
        VersionBundle(),
    )


def _hypothesis(source: SourceRef, marker: int) -> HypothesisKey:
    """构造同一来源、scope 和竞争组中的候选。"""
    return HypothesisKey(
        (15000, 1),
        (15001, marker),
        (15002, 1),
        document_scope(source),
        source,
    )


def _evidence(
        evidence_id: int, hypothesis: HypothesisKey, stance: int, *,
        source: SourceRef | None = None, timestamp_seq: int = 1,
        ) -> EvidenceRecord:
    """构造理由和 payload 均保持开放整数协议的证据。"""
    return EvidenceRecord(
        evidence_id,
        hypothesis,
        stance,
        (15100, stance),
        hypothesis.observation if source is None else source,
        timestamp_seq,
        payload=(15101, evidence_id),
    )


def _ledger(*hypotheses: HypothesisKey) -> HypothesisLedger:
    """建立拥有给定竞争候选的真实 H-00 ledger。"""
    ledger = HypothesisLedger()
    for hypothesis in hypotheses:
        ledger.register(hypothesis)
    return ledger


class _StaticScorer:
    """测试用 typed scorer；关系语义由实例参数注入。"""

    def __init__(self, scorer_key: tuple[int, ...], preference: int) -> None:
        self.scorer_key = scorer_key
        self.preference = preference

    def preferences(
            self, hypotheses: tuple[HypothesisKey, ...],
            ) -> tuple[ResolverPreference, ...]:
        """为 resolver 请求的每个规范候选对返回同一注入关系。"""
        return tuple(
            ResolverPreference(
                self.scorer_key,
                hypotheses[left],
                hypotheses[right],
                self.preference,
                (15200, left, right),
            )
            for left in range(len(hypotheses))
            for right in range(left + 1, len(hypotheses))
        )


class _IncompleteScorer:
    """故意漏掉候选对，用于证明通用层会 fail closed。"""

    scorer_key = (15210, 1)

    def preferences(
            self, hypotheses: tuple[HypothesisKey, ...],
            ) -> tuple[ResolverPreference, ...]:
        """返回空关系，候选数大于一时必须被 resolver 拒绝。"""
        return ()


def test_two_axes_source_accounts_and_frequency_do_not_hide_conflict():
    """生命周期与证据四态正交，重复支持不能投票吞掉定向反驳。"""
    source = _source(1)
    other_source = _source(2, source_id=1501)
    supported = _hypothesis(source, 1)
    conflicted = _hypothesis(source, 2)
    unknown = _hypothesis(source, 3)
    ledger = _ledger(supported, conflicted, unknown)
    ledger.append_evidence(_evidence(1, supported, EVIDENCE_SUPPORT))
    for evidence_id in range(2, 12):
        ledger.append_evidence(_evidence(
            evidence_id,
            conflicted,
            EVIDENCE_SUPPORT,
            source=source,
            timestamp_seq=evidence_id,
        ))
    ledger.append_evidence(_evidence(
        12,
        conflicted,
        EVIDENCE_REFUTE,
        source=other_source,
        timestamp_seq=12,
    ))

    decision = HypothesisResolver(ledger).resolve(
        supported, timestamp_seq=13)

    assert decision.adopted_hypotheses == (supported,)
    conflict_trace = decision.candidate(conflicted)
    assert conflict_trace.before.lifecycle == LIFECYCLE_ACTIVE
    assert conflict_trace.after.lifecycle == LIFECYCLE_ACTIVE
    assert conflict_trace.after.epistemic_status == EPISTEMIC_CONFLICTED
    assert conflict_trace.role == RESOLUTION_RETAINED
    assert tuple(account.source for account in conflict_trace.source_accounts) == (
        source,
        other_source,
    )
    assert conflict_trace.source_accounts[0].epistemic_status == (
        EPISTEMIC_SUPPORTED)
    assert conflict_trace.source_accounts[1].epistemic_status == (
        EPISTEMIC_REFUTED)
    assert decision.candidate(unknown).after.epistemic_status == (
        EPISTEMIC_UNKNOWN)


def test_refuted_candidates_archive_or_explicitly_supersede_and_replay_is_idempotent():
    """纯反驳默认归档，只有显式同组 replacement 才 supersede，重放不增事件。"""
    source = _source(3)
    rejected = _hypothesis(source, 1)
    archived = _hypothesis(source, 2)
    replacement = _hypothesis(source, 3)
    ledger = _ledger(rejected, archived, replacement)
    ledger.append_evidence(_evidence(
        21, rejected, EVIDENCE_REFUTE, timestamp_seq=3))
    ledger.append_evidence(_evidence(
        22, archived, EVIDENCE_REFUTE, timestamp_seq=4))
    ledger.append_evidence(_evidence(
        23, replacement, EVIDENCE_SUPPORT, timestamp_seq=5))
    resolver = HypothesisResolver(ledger)
    directive = ReplacementDirective(rejected, replacement, 21)

    first = resolver.resolve(
        rejected,
        timestamp_seq=6,
        replacements=(directive,),
    )
    state = resolver.state_key(), ledger.state_key()
    replay = resolver.resolve(
        rejected,
        timestamp_seq=6,
        replacements=(directive,),
    )

    assert replay == first
    assert (resolver.state_key(), ledger.state_key()) == state
    assert first.candidate(rejected).after.lifecycle == LIFECYCLE_SUPERSEDED
    assert first.candidate(archived).after.lifecycle == LIFECYCLE_ARCHIVED
    assert first.candidate(replacement).role == RESOLUTION_ADOPTED
    assert len(ledger.evidence_history(rejected)) == 1
    rejected_transition = ledger.transition_history(rejected)[0]
    assert rejected_transition.reason_evidence_id == 21
    assert rejected_transition.replacement == replacement
    assert first.candidate(rejected).transition_event_id == (
        rejected_transition.event_id)


def test_explicit_archive_exits_conflicted_candidate_without_fake_replacement():
    """显式归档可退出 conflicted 宽候选，并保持重放幂等和空 replacement。"""
    source = _source(31)
    broad = _hypothesis(source, 1)
    retained = _hypothesis(source, 2)
    ledger = _ledger(broad, retained)
    ledger.append_evidence(_evidence(31, broad, EVIDENCE_SUPPORT))
    ledger.append_evidence(_evidence(
        32, broad, EVIDENCE_REFUTE, timestamp_seq=2))
    ledger.append_evidence(_evidence(33, retained, EVIDENCE_SUPPORT))
    resolver = HypothesisResolver(ledger)
    directive = ArchiveDirective(broad, 32)

    first = resolver.resolve(
        broad, timestamp_seq=3, archives=(directive,))
    replay = resolver.resolve(
        broad, timestamp_seq=3, archives=(directive,))

    assert replay == first
    assert first.candidate(broad).before.epistemic_status == (
        EPISTEMIC_CONFLICTED)
    assert first.candidate(broad).after.lifecycle == LIFECYCLE_ARCHIVED
    transition = ledger.transition_history(broad)[0]
    assert transition.reason_evidence_id == 32
    assert transition.replacement is None


def test_archive_and_replacement_are_mutually_exclusive_before_any_write():
    """同一候选的归档与替代指令冲突时，ledger 和决策历史必须零写。"""
    source = _source(32)
    rejected = _hypothesis(source, 1)
    replacement = _hypothesis(source, 2)
    ledger = _ledger(rejected, replacement)
    ledger.append_evidence(_evidence(
        34, rejected, EVIDENCE_REFUTE, timestamp_seq=2))
    ledger.append_evidence(_evidence(35, replacement, EVIDENCE_SUPPORT))
    resolver = HypothesisResolver(ledger)
    before = ledger.state_key(), resolver.state_key()

    with pytest.raises(ValueError, match="同时 archive"):
        resolver.resolve(
            rejected,
            timestamp_seq=3,
            replacements=(ReplacementDirective(
                rejected, replacement, 34),),
            archives=(ArchiveDirective(rejected, 34),),
        )

    assert (ledger.state_key(), resolver.state_key()) == before


def test_exit_batch_validates_all_timestamps_before_first_transition():
    """竞争组内后续退出逻辑序非法时，先前合法候选也不得部分归档。"""
    source = _source(33)
    first = _hypothesis(source, 1)
    second = _hypothesis(source, 2)
    ledger = _ledger(first, second)
    ledger.append_evidence(_evidence(
        36, first, EVIDENCE_REFUTE, timestamp_seq=1))
    ledger.append_evidence(_evidence(
        37, second, EVIDENCE_REFUTE, timestamp_seq=5))
    resolver = HypothesisResolver(ledger)
    before = ledger.state_key(), resolver.state_key()

    with pytest.raises(ValueError, match="不得早于退出理由"):
        resolver.resolve(first, timestamp_seq=3)

    assert (ledger.state_key(), resolver.state_key()) == before
    assert ledger.snapshot(first).lifecycle == LIFECYCLE_ACTIVE
    assert ledger.snapshot(second).lifecycle == LIFECYCLE_ACTIVE


def test_scorer_conflict_incomparable_and_equivalence_preserve_multiple_candidates():
    """反向、不可比和同分都不允许稳定键偷偷强选单 winner。"""
    source = _source(4)
    left = _hypothesis(source, 1)
    right = _hypothesis(source, 2)
    ledger = _ledger(left, right)
    ledger.append_evidence(_evidence(31, left, EVIDENCE_SUPPORT))
    ledger.append_evidence(_evidence(32, right, EVIDENCE_SUPPORT))
    resolver = HypothesisResolver(ledger)

    conflicted = resolver.resolve(
        left,
        timestamp_seq=2,
        scorers=(
            _StaticScorer((15300, 1), PREFERENCE_LEFT_BETTER),
            _StaticScorer((15300, 2), PREFERENCE_RIGHT_BETTER),
        ),
    )
    incomparable = resolver.resolve(
        left,
        timestamp_seq=3,
        scorers=(_StaticScorer(
            (15300, 3), PREFERENCE_INCOMPARABLE),),
    )
    equivalent = resolver.resolve(
        left,
        timestamp_seq=4,
        scorers=(_StaticScorer(
            (15300, 4), PREFERENCE_EQUIVALENT),),
    )

    assert conflicted.adopted_hypotheses == (left, right)
    assert incomparable.adopted_hypotheses == (left, right)
    assert equivalent.adopted_hypotheses == (left, right)


def test_all_applicable_scorers_must_agree_before_one_candidate_dominates():
    """同向加等价可形成支配，但 scorer 漏候选对必须 fail closed。"""
    source = _source(5)
    left = _hypothesis(source, 1)
    right = _hypothesis(source, 2)
    ledger = _ledger(left, right)
    ledger.append_evidence(_evidence(41, left, EVIDENCE_SUPPORT))
    ledger.append_evidence(_evidence(42, right, EVIDENCE_SUPPORT))
    resolver = HypothesisResolver(ledger)

    decision = resolver.resolve(
        left,
        timestamp_seq=2,
        scorers=(
            _StaticScorer((15400, 1), PREFERENCE_LEFT_BETTER),
            _StaticScorer((15400, 2), PREFERENCE_EQUIVALENT),
        ),
    )
    assert decision.adopted_hypotheses == (left,)
    assert decision.candidate(right).dominated_by == (left,)
    with pytest.raises(ValueError, match="完整覆盖"):
        resolver.resolve(
            left,
            timestamp_seq=3,
            scorers=(_IncompleteScorer(),),
        )


def test_invalid_scorer_fails_before_any_lifecycle_or_decision_write():
    """scorer 漏关系时即使存在纯反驳候选，也不得留下部分 transition。"""
    source = _source(51)
    rejected = _hypothesis(source, 1)
    first = _hypothesis(source, 2)
    second = _hypothesis(source, 3)
    ledger = _ledger(rejected, first, second)
    ledger.append_evidence(_evidence(
        45, rejected, EVIDENCE_REFUTE, timestamp_seq=1))
    ledger.append_evidence(_evidence(46, first, EVIDENCE_SUPPORT))
    ledger.append_evidence(_evidence(47, second, EVIDENCE_SUPPORT))
    resolver = HypothesisResolver(ledger)
    before = ledger.state_key(), resolver.state_key()

    with pytest.raises(ValueError, match="完整覆盖"):
        resolver.resolve(
            rejected,
            timestamp_seq=2,
            scorers=(_IncompleteScorer(),),
        )

    assert (ledger.state_key(), resolver.state_key()) == before
    assert ledger.snapshot(rejected).lifecycle == LIFECYCLE_ACTIVE


def test_targeted_counterexample_records_degradation_and_old_decision_chain():
    """加入反例后旧 adopted 候选退出，先前决策和 prior role 仍可审计。"""
    source = _source(6)
    old = _hypothesis(source, 1)
    replacement = _hypothesis(source, 2)
    ledger = _ledger(old, replacement)
    ledger.append_evidence(_evidence(51, old, EVIDENCE_SUPPORT))
    ledger.append_evidence(_evidence(52, replacement, EVIDENCE_SUPPORT))
    resolver = HypothesisResolver(ledger)
    initial = resolver.resolve(
        old,
        timestamp_seq=2,
        scorers=(_StaticScorer(
            (15500, 1), PREFERENCE_LEFT_BETTER),),
    )
    ledger.append_evidence(_evidence(
        53,
        old,
        EVIDENCE_REFUTE,
        timestamp_seq=3,
    ))

    conflicted = resolver.resolve(old, timestamp_seq=4)
    ledger.append_evidence(EvidenceRecord(
        54,
        old,
        EVIDENCE_REFUTE,
        (15510, 1),
        source,
        5,
        supersedes_evidence_id=51,
    ))
    exited = resolver.resolve(
        old,
        timestamp_seq=6,
        replacements=(ReplacementDirective(old, replacement, 54),),
    )

    assert initial.adopted_hypotheses == (old,)
    assert conflicted.candidate(old).after.epistemic_status == (
        EPISTEMIC_CONFLICTED)
    assert conflicted.candidate(old).after.lifecycle == LIFECYCLE_ACTIVE
    assert exited.candidate(old).role == RESOLUTION_EXITED
    assert exited.candidate(old).prior_role == RESOLUTION_RETAINED
    assert exited.adopted_hypotheses == (replacement,)
    assert exited.previous_decision_id == conflicted.decision_id
    assert tuple(
        item.decision_id for item in resolver.decision_history(old)
    ) == (
        initial.decision_id,
        conflicted.decision_id,
        exited.decision_id,
    )


def test_resolver_decision_roundtrip_requires_complete_chain_and_content():
    """H-04 决策必须完整往返，缺前驱或任一整数位漂移都拒绝恢复。"""
    source = _source(62)
    old = _hypothesis(source, 1)
    replacement = _hypothesis(source, 2)
    ledger = _ledger(old, replacement)
    ledger.append_evidence(_evidence(61, old, EVIDENCE_SUPPORT))
    ledger.append_evidence(_evidence(62, replacement, EVIDENCE_SUPPORT))
    resolver = HypothesisResolver(ledger)
    first = resolver.resolve(old, timestamp_seq=3)
    ledger.append_evidence(_evidence(
        63, old, EVIDENCE_REFUTE, timestamp_seq=4))
    second = resolver.resolve(old, timestamp_seq=5)

    assert ResolverPreference.from_stable_key(
        ResolverPreference(
            (15520, 1), old, replacement, PREFERENCE_EQUIVALENT,
        ).stable_key()
    )
    assert HypothesisResolver.from_history(
        ledger,
        (first, second),
    ).decision_history(old) == (first, second)

    tampered = list(second.stable_key())
    tampered[-1] += 1
    with pytest.raises(ValueError, match="不一致|非法|尾随"):
        ResolverDecision.from_stable_key(tuple(tampered))
    with pytest.raises(ValueError, match="缺前驱|根"):
        HypothesisResolver.from_history(ledger, (second,))


def _unit(value: int) -> ObjectIdentity:
    """构造 H-03 问题中的一等被编码对象。"""
    return ObjectIdentity(OBJECT_REPRESENTATION, (15600, value))


def _fragment(value: int) -> ObjectIdentity:
    """构造 H-03 候选定义的一等 fragment。"""
    return ObjectIdentity(OBJECT_STRUCTURE_CONCEPT, (15610, value))


def test_h02a_targeted_evidence_flows_into_same_h04_owner_ledger():
    """H-02A 定向反驳经同一 ledger 使错误候选退出，不复制 Evidence。"""
    source = _source(61)
    rejected = _hypothesis(source, 1)
    retained = _hypothesis(source, 2)
    ledger = _ledger(rejected, retained)
    perturbation = PerturbationEngine(
        PerturbationProtocol(
            (15611, 1),
            (15611, 2),
            (15611, 3),
        ),
        ledger=ledger,
    )
    trace = build_permutation_trace(
        (_unit(1), _unit(2)),
        output_order=(1, 0),
        transform_key=(15612, 1),
        source=source,
        scope=document_scope(source),
    )
    evidence = perturbation.evaluate(
        trace,
        candidates=(rejected, retained),
        verifier=lambda _trace, _candidates: PerturbationAssessment(
            ASSESSMENT_CHANGED,
            (15613, 1),
            (rejected,),
        ),
        evidence_source=source,
        timestamp_seq=1,
    )

    decision = HypothesisResolver(ledger).resolve(
        rejected, timestamp_seq=2)

    assert len(evidence.evidence_ids) == 1
    assert ledger.evidence_history(rejected)[0].evidence_id == (
        evidence.evidence_ids[0])
    assert decision.candidate(rejected).after.lifecycle == LIFECYCLE_ARCHIVED
    assert decision.adopted_hypotheses == (retained,)


def _description_problem(source: SourceRef) -> DescriptionLengthProblem:
    """构造含一条四对象观察的完整描述长度问题。"""
    return DescriptionLengthProblem((DescriptionObservation(
        source,
        document_scope(source),
        (15620, 1),
        (_unit(1), _unit(2), _unit(3), _unit(4)),
    ),))


def _literal_candidate(
        hypothesis: HypothesisKey,
        problem: DescriptionLengthProblem,
        ) -> DescriptionCandidate:
    """构造逐对象 literal 的无损 H-03 候选。"""
    observation = problem.observations[0]
    return DescriptionCandidate(
        DescriptionModel(hypothesis),
        (DescriptionEncoding(
            observation.source,
            observation.scope,
            observation.event_key,
            tuple(DescriptionTerm.literal(unit) for unit in observation.units),
        ),),
    )


def _memorized_candidate(
        hypothesis: HypothesisKey,
        problem: DescriptionLengthProblem,
        ) -> DescriptionCandidate:
    """构造只用一次 fragment 的过窄记忆候选。"""
    observation = problem.observations[0]
    fragment = DescriptionFragment(
        _fragment(1),
        tuple(DescriptionTerm.literal(unit) for unit in observation.units),
    )
    return DescriptionCandidate(
        DescriptionModel(hypothesis, (fragment,)),
        (DescriptionEncoding(
            observation.source,
            observation.scope,
            observation.event_key,
            (DescriptionTerm.fragment(fragment.fragment),),
        ),),
    )


def test_description_length_adapter_prefers_lower_cost_but_preserves_exact_tie():
    """H-03 adapter 以 bit 成本低者更优，同成本不按 Hypothesis 键决胜。"""
    source = _source(7)
    literal_hypothesis = _hypothesis(source, 2)
    memorized_hypothesis = _hypothesis(source, 1)
    tied_hypothesis = _hypothesis(source, 3)
    ledger = _ledger(
        literal_hypothesis,
        memorized_hypothesis,
        tied_hypothesis,
    )
    for evidence_id, hypothesis in enumerate(
            (literal_hypothesis, memorized_hypothesis, tied_hypothesis),
            start=61):
        ledger.append_evidence(_evidence(
            evidence_id, hypothesis, EVIDENCE_SUPPORT))
    problem = _description_problem(source)
    literal = _literal_candidate(literal_hypothesis, problem)
    memorized = _memorized_candidate(memorized_hypothesis, problem)
    tied = _literal_candidate(tied_hypothesis, problem)
    scorer = DescriptionLengthResolverScorer(
        (15700, 1),
        engine=DescriptionLengthEngine(ledger),
        problem=problem,
        candidates=(literal, memorized, tied),
    )

    decision = HypothesisResolver(ledger).resolve(
        literal_hypothesis,
        timestamp_seq=2,
        scorers=(scorer,),
    )

    assert decision.adopted_hypotheses == (
        literal_hypothesis,
        tied_hypothesis,
    )
    assert decision.candidate(memorized_hypothesis).dominated_by == (
        literal_hypothesis,
        tied_hypothesis,
    )
    assert all(preference.payload for preference in decision.preferences)


def test_preview_and_clone_keep_host_ledger_and_decision_history_isolated():
    """预览和克隆可产生完整决策，但宿主 Evidence、转换和历史位级不变。"""
    source = _source(8)
    rejected = _hypothesis(source, 1)
    replacement = _hypothesis(source, 2)
    ledger = _ledger(rejected, replacement)
    ledger.append_evidence(_evidence(
        71, rejected, EVIDENCE_REFUTE, timestamp_seq=2))
    ledger.append_evidence(_evidence(
        72, replacement, EVIDENCE_SUPPORT, timestamp_seq=2))
    resolver = HypothesisResolver(ledger)
    host_state = ledger.state_key(), resolver.state_key()

    preview = resolver.resolve(
        rejected,
        timestamp_seq=3,
        replacements=(ReplacementDirective(rejected, replacement, 71),),
        commit=False,
    )
    cloned_ledger = ledger.clone()
    cloned = resolver.clone(ledger=cloned_ledger)
    cloned_decision = cloned.resolve(
        rejected,
        timestamp_seq=3,
        replacements=(ReplacementDirective(rejected, replacement, 71),),
    )

    assert preview.candidate(rejected).after.lifecycle == LIFECYCLE_SUPERSEDED
    assert cloned_decision.candidate(rejected).after.lifecycle == (
        LIFECYCLE_SUPERSEDED)
    assert (ledger.state_key(), resolver.state_key()) == host_state
    assert cloned_ledger.state_key() != ledger.state_key()
    assert cloned.state_key() != resolver.state_key()


def test_recovery_accepts_append_supersede_and_new_candidate_visibility():
    """旧 decision 后 Evidence 前进和新候选可见不应阻断决策链恢复。"""
    source = _source(81)
    old = _hypothesis(source, 1)
    added = _hypothesis(source, 2)
    ledger = _ledger(old)
    ledger.append_evidence(_evidence(
        81, old, EVIDENCE_SUPPORT, timestamp_seq=1))
    original = HypothesisResolver(ledger).resolve(old, timestamp_seq=2)

    ledger.append_evidence(_evidence(
        82, old, EVIDENCE_SUPPORT, timestamp_seq=3))
    ledger.append_evidence(EvidenceRecord(
        83,
        old,
        EVIDENCE_REFUTE,
        (15100, EVIDENCE_REFUTE),
        source,
        4,
        payload=(15101, 83),
        supersedes_evidence_id=81,
    ))
    ledger.register(added)
    ledger.append_evidence(_evidence(
        84, added, EVIDENCE_SUPPORT, timestamp_seq=5))

    recovered = HypothesisResolver.from_history(ledger, (original,))
    assert recovered.decision_history(old) == (original,)
    advanced = recovered.resolve(old, timestamp_seq=6)
    assert advanced.previous_decision_id == original.decision_id
    assert {item.hypothesis for item in advanced.candidates} == {old, added}
    assert advanced.adopted_hypotheses == (added,)


def test_recovery_still_rejects_lifecycle_drift_after_latest_decision():
    """Evidence 可前进不等于外部可以绕过 H-04 改写 lifecycle。"""
    source = _source(82)
    hypothesis = _hypothesis(source, 1)
    ledger = _ledger(hypothesis)
    ledger.append_evidence(_evidence(
        91, hypothesis, EVIDENCE_SUPPORT, timestamp_seq=1))
    original = HypothesisResolver(ledger).resolve(
        hypothesis, timestamp_seq=2)
    ledger.append_evidence(_evidence(
        92, hypothesis, EVIDENCE_REFUTE, timestamp_seq=3))
    HypothesisResolver(ledger).resolve(
        hypothesis,
        timestamp_seq=4,
        archives=(ArchiveDirective(hypothesis, 92),),
    )

    with pytest.raises(ValueError, match="当前生命周期不一致"):
        HypothesisResolver.from_history(ledger, (original,))
