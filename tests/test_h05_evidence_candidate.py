"""H-05 独立样本候选、负证据和 H-04 消费投影专项。"""
from __future__ import annotations

import pytest

from pure_integer_ai.cognition.shared.evidence_candidate import (
    CandidateBinding,
    CandidateVerification,
    EvidenceCandidateDefinition,
    EvidenceCandidateEngine,
    EvidenceCandidateError,
    EvidenceCandidateProtocol,
)
from pure_integer_ai.cognition.shared.hypothesis import (
    EPISTEMIC_CONFLICTED,
    EPISTEMIC_UNKNOWN,
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EVIDENCE_UNKNOWN,
    LIFECYCLE_ARCHIVED,
)
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    SourceRef,
    VersionBundle,
    concept_identity,
    sense_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.scope_identity import document_scope


def _source(source_id: int, document_id: int = 0) -> SourceRef:
    """构造同一 owner/version 下的测试来源。"""
    return SourceRef(
        7,
        source_id,
        document_id,
        GLOBAL_OWNER_SCOPE,
        VersionBundle(),
    )


def _engine(*, minimum: int = 2) -> EvidenceCandidateEngine:
    """构造使用真实 aggregate manifest 的候选 owner。"""
    aggregate = _source(900)
    return EvidenceCandidateEngine(EvidenceCandidateProtocol(
        hypothesis_kind_key=(11, 1),
        formation_reason_key=(11, 2),
        aggregate_source=aggregate,
        aggregate_scope=document_scope(aggregate),
        minimum_forming_sources=minimum,
    ))


def _definition(
        local_key: int, *, forming=(1, 2), competition=(21, 1),
        sense: bool = False) -> EvidenceCandidateDefinition:
    """构造结构或 Sense 候选及其开放图绑定。"""
    source = _source(700 + local_key)
    candidate = (
        sense_identity(source, sense_key=(local_key, 1))
        if sense else structure_concept_identity((31, local_key))
    )
    return EvidenceCandidateDefinition(
        candidate=candidate,
        competition_key=competition,
        bindings=(
            CandidateBinding(
                concept_identity((41, 1)),
                concept_identity((51, local_key)),
            ),
            CandidateBinding(
                concept_identity((41, 2)),
                concept_identity((61, 1)),
            ),
        ),
        forming_sources=tuple(_source(item) for item in forming),
    )


def _verification(stance: int, *, reason: int = 1) -> CandidateVerification:
    """构造与候选 aggregate 分离的 verifier 揭示。"""
    return CandidateVerification(
        stance=stance,
        reason_key=(71, reason),
        source=_source(800 + reason),
        authority=concept_identity((81, 1)),
        authority_version=(91, 1),
        trace=(101, reason),
    )


def _predict(engine: EvidenceCandidateEngine, hypothesis, *, source_id: int):
    """为一个独立来源冻结 cue/structure 预测。"""
    observation = _source(source_id)
    return engine.predict(
        hypothesis,
        observation=observation,
        scope=document_scope(observation),
        event_key=(111, source_id),
        visible_inputs=(concept_identity((121, source_id)),),
        predicted=concept_identity((131, 1)),
    )


def test_forming_only_registers_unknown_and_cannot_be_consumed():
    """形成 K 只允许提出候选，不得直接产生 support 或 active 消费。"""
    engine = _engine()
    hypothesis = engine.register(_definition(1), timestamp_base=10)

    snapshot = engine.ledger.snapshot(hypothesis)
    assert snapshot.epistemic_status == EPISTEMIC_UNKNOWN
    assert len(snapshot.unknown_evidence_ids) == 2
    assert engine.active(hypothesis) is None

    decision = engine.resolve(hypothesis, timestamp_seq=20)
    # H-04 adopted 在此表示保留未决候选；H-05 supported 闸仍禁止业务消费。
    assert hypothesis in decision.adopted_hypotheses
    assert engine.active(hypothesis) is None


def test_prediction_rejects_forming_source_and_must_precede_reveal():
    """forming root 不得混入 recognition，未冻结 prediction 也不得直接揭示。"""
    engine = _engine()
    hypothesis = engine.register(_definition(1))

    with pytest.raises(EvidenceCandidateError, match="forming observation"):
        _predict(engine, hypothesis, source_id=1)

    other = _engine()
    other_hypothesis = other.register(_definition(1))
    prediction = _predict(engine, hypothesis, source_id=3)
    forged = prediction.__class__(
        other_hypothesis,
        prediction.observation,
        prediction.scope,
        prediction.event_key,
        prediction.visible_inputs,
        prediction.predicted,
    )
    with pytest.raises(EvidenceCandidateError, match="先行冻结"):
        other.reveal(forged, _verification(EVIDENCE_SUPPORT), timestamp_seq=30)


def test_independent_support_requires_h04_before_active_projection():
    """独立样本 support 仍须经已提交 H-04 decision 才可供消费者采用。"""
    engine = _engine()
    hypothesis = engine.register(_definition(1))
    prediction = _predict(engine, hypothesis, source_id=3)
    evidence = engine.reveal(
        prediction,
        _verification(EVIDENCE_SUPPORT),
        timestamp_seq=30,
    )

    assert evidence.stance == EVIDENCE_SUPPORT
    assert engine.active(hypothesis) is None
    decision = engine.resolve(hypothesis, timestamp_seq=31)
    active = engine.active(hypothesis)
    assert active is not None
    assert active.decision == decision
    assert active.definition == _definition(1)


def test_same_observation_does_not_gain_duplicate_support():
    """同一 observation/event 精确重放幂等，改写揭示则因事件冲突失败。"""
    engine = _engine()
    hypothesis = engine.register(_definition(1))
    prediction = _predict(engine, hypothesis, source_id=3)
    first = engine.reveal(
        prediction,
        _verification(EVIDENCE_SUPPORT),
        timestamp_seq=30,
    )
    replay = engine.reveal(
        prediction,
        _verification(EVIDENCE_SUPPORT),
        timestamp_seq=30,
    )
    assert replay == first
    assert engine.ledger.snapshot(hypothesis).support_evidence_ids == (
        first.evidence_id,)

    with pytest.raises(ValueError, match="evidence_id"):
        engine.reveal(
            prediction,
            _verification(EVIDENCE_REFUTE, reason=2),
            timestamp_seq=31,
        )


def test_unknown_verifier_result_never_becomes_false_or_active():
    """独立来源缺真值只能记 unknown，不能把图缺边解释成 refute。"""
    engine = _engine()
    hypothesis = engine.register(_definition(1))
    prediction = _predict(engine, hypothesis, source_id=3)
    evidence = engine.reveal(
        prediction,
        _verification(EVIDENCE_UNKNOWN),
        timestamp_seq=30,
    )
    decision = engine.resolve(hypothesis, timestamp_seq=31)

    assert evidence.stance == EVIDENCE_UNKNOWN
    assert not engine.ledger.snapshot(hypothesis).refute_evidence_ids
    assert hypothesis in decision.adopted_hypotheses
    assert engine.active(hypothesis) is None


def test_wrong_cue_can_be_archived_after_conflicting_negative():
    """已有支持的错误 cue 收到定向反例后保持 conflicted，显式归档才退出。"""
    engine = _engine()
    hypothesis = engine.register(_definition(1))
    positive = _predict(engine, hypothesis, source_id=3)
    engine.reveal(
        positive,
        _verification(EVIDENCE_SUPPORT),
        timestamp_seq=30,
    )
    engine.resolve(hypothesis, timestamp_seq=31)
    assert engine.active(hypothesis) is not None

    negative = _predict(engine, hypothesis, source_id=4)
    engine.reveal(
        negative,
        _verification(EVIDENCE_REFUTE, reason=2),
        timestamp_seq=40,
    )
    assert engine.ledger.snapshot(
        hypothesis).epistemic_status == EPISTEMIC_CONFLICTED
    engine.resolve(
        hypothesis,
        timestamp_seq=41,
        archive_refuted=True,
    )

    assert engine.ledger.snapshot(hypothesis).lifecycle == LIFECYCLE_ARCHIVED
    assert engine.active(hypothesis) is None


def test_multiple_supported_senses_remain_visible_but_strict_consumer_fails():
    """观察等价的多个 Sense 可并存，严格语义消费者不得按稳定序私选。"""
    engine = _engine()
    first = engine.register(_definition(1, sense=True))
    second = engine.register(_definition(2, sense=True))
    for ordinal, hypothesis in enumerate((first, second), start=3):
        prediction = _predict(engine, hypothesis, source_id=ordinal)
        engine.reveal(
            prediction,
            _verification(EVIDENCE_SUPPORT, reason=ordinal),
            timestamp_seq=20 + ordinal,
        )
    decision = engine.resolve(first, timestamp_seq=30)

    assert set(decision.adopted_hypotheses) == {first, second}
    assert len(engine.active_competition(first)) == 2
    with pytest.raises(LookupError, match="唯一"):
        engine.require_unique(first)


def test_clone_keeps_held_out_evidence_out_of_training_owner():
    """held-out clone 可执行完整 reveal/resolve，但不得回写训练候选 owner。"""
    engine = _engine()
    hypothesis = engine.register(_definition(1))
    baseline = engine.state_key()
    cloned = engine.clone()
    prediction = _predict(cloned, hypothesis, source_id=3)
    cloned.reveal(
        prediction,
        _verification(EVIDENCE_SUPPORT),
        timestamp_seq=30,
    )
    cloned.resolve(hypothesis, timestamp_seq=31)

    assert cloned.active(hypothesis) is not None
    assert engine.active(hypothesis) is None
    assert engine.state_key() == baseline


def test_candidate_formation_condition_is_injected_not_fixed_in_engine():
    """不同课程可注入不同形成条件，核心不持有固定 K。"""
    engine = _engine(minimum=3)
    with pytest.raises(EvidenceCandidateError, match="形成"):
        engine.register(_definition(1, forming=(1, 2)))

    hypothesis = engine.register(_definition(1, forming=(1, 2, 3)))
    assert len(engine.ledger.snapshot(hypothesis).unknown_evidence_ids) == 3
