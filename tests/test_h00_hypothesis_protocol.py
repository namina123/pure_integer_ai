from __future__ import annotations

import pytest

from pure_integer_ai.cognition.shared.hypothesis import (
    EPISTEMIC_CONFLICTED,
    EPISTEMIC_REFUTED,
    EPISTEMIC_SUPPORTED,
    EPISTEMIC_UNKNOWN,
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EVIDENCE_UNKNOWN,
    EvidenceRecord,
    HypothesisKey,
    HypothesisLedger,
    HypothesisTransition,
    LIFECYCLE_ACTIVE,
    LIFECYCLE_ARCHIVED,
    LIFECYCLE_SUPERSEDED,
)
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    OBJECT_HYPOTHESIS,
    SourceRef,
    VersionBundle,
)
from pure_integer_ai.cognition.shared.scope_identity import document_scope


def _source(document_id: int = 1, source_id: int = 101) -> SourceRef:
    """构造 H-00 测试使用的稳定观察或证据来源。"""
    return SourceRef(
        1,
        source_id,
        document_id,
        GLOBAL_OWNER_SCOPE,
        VersionBundle(),
    )


def _hypothesis(candidate: int, *, observation=None) -> HypothesisKey:
    """构造同一竞争组内的候选键。"""
    source = observation or _source()
    return HypothesisKey(
        (11,),
        (candidate,),
        (9001,),
        document_scope(source),
        source,
    )


def _evidence(evidence_id: int, hypothesis: HypothesisKey, stance: int, *,
              source_id: int = 201, supersedes: int = 0) -> EvidenceRecord:
    """构造带独立来源和逻辑序的不可变 Evidence。"""
    return EvidenceRecord(
        evidence_id,
        hypothesis,
        stance,
        (31,),
        _source(document_id=evidence_id, source_id=source_id),
        evidence_id,
        payload=(7, evidence_id),
        supersedes_evidence_id=supersedes,
    )


def test_same_observation_can_support_mutually_exclusive_candidates():
    """同一观察可同时保留多个互斥候选，不由登记顺序强选 winner。"""
    ledger = HypothesisLedger()
    first = ledger.register(_hypothesis(1))
    second = ledger.register(_hypothesis(2))
    ledger.append_evidence(_evidence(1, first, EVIDENCE_SUPPORT))
    ledger.append_evidence(_evidence(2, second, EVIDENCE_SUPPORT))

    competition = ledger.competition(first)
    assert tuple(item.hypothesis for item in competition) == (first, second)
    assert all(item.lifecycle == LIFECYCLE_ACTIVE for item in competition)
    assert all(
        item.epistemic_status == EPISTEMIC_SUPPORTED
        for item in competition)


def test_epistemic_status_is_derived_without_overloading_lifecycle():
    """支持、反驳、未知和冲突只由活动 Evidence 派生。"""
    ledger = HypothesisLedger()
    supported = ledger.register(_hypothesis(1))
    refuted = ledger.register(_hypothesis(2))
    unknown = ledger.register(_hypothesis(3))
    conflicted = ledger.register(_hypothesis(4))
    ledger.append_evidence(_evidence(1, supported, EVIDENCE_SUPPORT))
    ledger.append_evidence(_evidence(2, refuted, EVIDENCE_REFUTE))
    ledger.append_evidence(_evidence(3, unknown, EVIDENCE_UNKNOWN))
    ledger.append_evidence(_evidence(4, conflicted, EVIDENCE_SUPPORT))
    ledger.append_evidence(_evidence(5, conflicted, EVIDENCE_REFUTE))

    assert ledger.snapshot(supported).epistemic_status == EPISTEMIC_SUPPORTED
    assert ledger.snapshot(refuted).epistemic_status == EPISTEMIC_REFUTED
    assert ledger.snapshot(unknown).epistemic_status == EPISTEMIC_UNKNOWN
    assert ledger.snapshot(conflicted).epistemic_status == EPISTEMIC_CONFLICTED
    assert ledger.snapshot(conflicted).lifecycle == LIFECYCLE_ACTIVE


def test_evidence_correction_is_append_only_and_recomputes_snapshot():
    """反例可追加替代旧支持，但历史 Evidence 不被删除。"""
    ledger = HypothesisLedger()
    hypothesis = ledger.register(_hypothesis(1))
    ledger.append_evidence(_evidence(1, hypothesis, EVIDENCE_SUPPORT))
    ledger.append_evidence(_evidence(
        2, hypothesis, EVIDENCE_REFUTE, supersedes=1))

    snapshot = ledger.snapshot(hypothesis)
    assert snapshot.epistemic_status == EPISTEMIC_REFUTED
    assert snapshot.support_evidence_ids == ()
    assert snapshot.refute_evidence_ids == (2,)
    assert tuple(
        item.evidence_id for item in ledger.evidence_history(hypothesis)
    ) == (1, 2)


def test_lifecycle_transition_requires_evidence_reason_and_replacement():
    """superseded/archived 只能经有理由的单向 append-only 事件进入。"""
    ledger = HypothesisLedger()
    old = ledger.register(_hypothesis(1))
    replacement = ledger.register(_hypothesis(2))
    ledger.append_evidence(_evidence(1, old, EVIDENCE_REFUTE))
    ledger.append_transition(HypothesisTransition(
        10,
        old,
        LIFECYCLE_ACTIVE,
        LIFECYCLE_SUPERSEDED,
        1,
        (41,),
        2,
        replacement,
    ))
    assert ledger.snapshot(old).lifecycle == LIFECYCLE_SUPERSEDED

    ledger.append_transition(HypothesisTransition(
        11,
        old,
        LIFECYCLE_SUPERSEDED,
        LIFECYCLE_ARCHIVED,
        1,
        (42,),
        3,
    ))
    assert ledger.snapshot(old).lifecycle == LIFECYCLE_ARCHIVED
    assert tuple(
        event.event_id for event in ledger.transition_history(old)
    ) == (10, 11)


def test_invalid_cross_candidate_evidence_and_transition_fail_closed():
    """Evidence 和 replacement 不得跨候选或竞争组静默串接。"""
    ledger = HypothesisLedger()
    first = ledger.register(_hypothesis(1))
    second = ledger.register(_hypothesis(2))
    ledger.append_evidence(_evidence(1, first, EVIDENCE_SUPPORT))
    with pytest.raises(ValueError, match="其他候选"):
        ledger.append_evidence(_evidence(
            2, second, EVIDENCE_REFUTE, supersedes=1))

    other_group = HypothesisKey(
        (11,),
        (3,),
        (9002,),
        first.scope,
        first.observation,
    )
    ledger.register(other_group)
    with pytest.raises(ValueError, match="同一竞争组"):
        ledger.append_transition(HypothesisTransition(
            10,
            first,
            LIFECYCLE_ACTIVE,
            LIFECYCLE_SUPERSEDED,
            1,
            (41,),
            2,
            other_group,
        ))

    other_observation = _source(document_id=99, source_id=102)
    same_bare_group = ledger.register(HypothesisKey(
        first.hypothesis_kind,
        (4,),
        first.competition_key,
        document_scope(other_observation),
        other_observation,
    ))
    with pytest.raises(ValueError, match="同一竞争组"):
        ledger.append_transition(HypothesisTransition(
            11,
            first,
            LIFECYCLE_ACTIVE,
            LIFECYCLE_SUPERSEDED,
            1,
            (41,),
            2,
            same_bare_group,
        ))


def test_hypothesis_identity_and_clone_are_stable_and_isolated():
    """候选可映射为一等 Hypothesis 身份，评测试算 clone 不回写原 ledger。"""
    ledger = HypothesisLedger()
    hypothesis = ledger.register(_hypothesis(1))
    ledger.append_evidence(_evidence(1, hypothesis, EVIDENCE_SUPPORT))
    identity = hypothesis.object_identity()
    assert identity.object_kind == OBJECT_HYPOTHESIS
    assert identity.components == hypothesis.stable_key()

    cloned = ledger.clone()
    cloned.append_evidence(_evidence(2, hypothesis, EVIDENCE_REFUTE))
    assert ledger.snapshot(hypothesis).epistemic_status == EPISTEMIC_SUPPORTED
    assert cloned.snapshot(hypothesis).epistemic_status == EPISTEMIC_CONFLICTED


def test_candidate_index_returns_all_competitions_and_isolates_full_scope():
    """对象恢复索引保留全部竞争组，并精确隔离 kind、来源和 scope。"""
    ledger = HypothesisLedger()
    source = _source()
    scope = document_scope(source)
    first = ledger.register(HypothesisKey(
        (11,), (7,), (9001,), scope, source))
    second = ledger.register(HypothesisKey(
        (11,), (7,), (9002,), scope, source))
    other_kind = ledger.register(HypothesisKey(
        (12,), (7,), (9003,), scope, source))
    other_source = _source(document_id=2, source_id=102)
    other_observation = ledger.register(HypothesisKey(
        (11,), (7,), (9004,), document_scope(other_source), other_source))
    for evidence_id, hypothesis in enumerate(
            (first, second, other_kind, other_observation), start=1):
        ledger.append_evidence(_evidence(
            evidence_id, hypothesis, EVIDENCE_SUPPORT))

    expected = (ledger.snapshot(first), ledger.snapshot(second))
    arguments = {
        "observation": source,
        "scope": scope,
        "hypothesis_kind": (11,),
    }
    assert ledger.candidate_snapshots((7,), **arguments) == expected
    assert ledger.candidate_snapshots(
        (7,), observation=source, scope=scope,
        hypothesis_kind=(12,),
    ) == (ledger.snapshot(other_kind),)
    assert ledger.candidate_snapshots(
        (7,), observation=other_source, scope=document_scope(other_source),
        hypothesis_kind=(11,),
    ) == (ledger.snapshot(other_observation),)

    cloned = ledger.clone()
    assert cloned.candidate_snapshots((7,), **arguments) == expected
    third = cloned.register(HypothesisKey(
        (11,), (7,), (9005,), scope, source))
    assert tuple(
        item.hypothesis
        for item in cloned.candidate_snapshots((7,), **arguments)
    ) == (first, second, third)
    assert ledger.candidate_snapshots((7,), **arguments) == expected


def test_hypothesis_key_round_trip_rejects_truncation_and_trailing_data():
    """图恢复所需的候选反序列化必须完整且 fail closed。"""
    hypothesis = _hypothesis(7)
    stable = hypothesis.stable_key()
    assert HypothesisKey.from_stable_key(stable) == hypothesis
    with pytest.raises(ValueError):
        HypothesisKey.from_stable_key(stable[:-1])
    with pytest.raises(ValueError, match="尾随"):
        HypothesisKey.from_stable_key((*stable, 1))


def test_duplicate_ids_are_idempotent_only_for_identical_events():
    """重复重放相同事件幂等，复用 id 写不同内容必须拒绝。"""
    ledger = HypothesisLedger()
    hypothesis = ledger.register(_hypothesis(1))
    evidence = _evidence(1, hypothesis, EVIDENCE_SUPPORT)
    assert ledger.append_evidence(evidence) == evidence
    assert ledger.append_evidence(evidence) == evidence
    with pytest.raises(ValueError, match="不同 Evidence"):
        ledger.append_evidence(_evidence(
            1, hypothesis, EVIDENCE_REFUTE, source_id=202))


def test_correction_and_transition_cannot_precede_their_reason():
    """supersede 和生命周期事件不得在逻辑序上倒置因果。"""
    ledger = HypothesisLedger()
    hypothesis = ledger.register(_hypothesis(1))
    replacement = ledger.register(_hypothesis(2))
    old = EvidenceRecord(
        1,
        hypothesis,
        EVIDENCE_SUPPORT,
        (31,),
        _source(document_id=1, source_id=201),
        10,
    )
    ledger.append_evidence(old)
    with pytest.raises(ValueError, match="不得早于旧事件"):
        ledger.append_evidence(EvidenceRecord(
            2,
            hypothesis,
            EVIDENCE_REFUTE,
            (31,),
            _source(document_id=2, source_id=202),
            9,
            supersedes_evidence_id=1,
        ))
    with pytest.raises(ValueError, match="不得早于理由"):
        ledger.append_transition(HypothesisTransition(
            10,
            hypothesis,
            LIFECYCLE_ACTIVE,
            LIFECYCLE_SUPERSEDED,
            1,
            (41,),
            9,
            replacement,
        ))
