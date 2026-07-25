"""H-06 顺序模式累计、context split 和 L-06 typed adapter 对抗测试。"""
from __future__ import annotations

import pytest

from pure_integer_ai.cognition.shared.hypothesis import (
    EPISTEMIC_CONFLICTED,
    EPISTEMIC_REFUTED,
    EPISTEMIC_SUPPORTED,
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    LIFECYCLE_ACTIVE,
    LIFECYCLE_ARCHIVED,
)
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    ParserVersion,
    SourceRef,
    VersionBundle,
    concept_identity,
    language_branch_identity,
    occurrence_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.order_facts import OrderFactIndex
from pure_integer_ai.cognition.shared.order_hypothesis import (
    OrderAssessment,
    OrderContextSplitAssessment,
    OrderHypothesisEngine,
    OrderLearningProtocol,
    OrderObservation,
    OrderPattern,
)
from pure_integer_ai.cognition.shared.order_perturbation import (
    OrderPerturbationAdapter,
)
from pure_integer_ai.cognition.shared.perturbation import (
    build_permutation_trace,
)
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.cognition.understanding.occurrence_index import (
    OccurrenceIndex,
    OccurrenceProtocol,
)
from pure_integer_ai.cognition.understanding.occurrence_order import (
    OccurrenceOrderProtocol,
    OccurrenceOrderReader,
    OccurrenceOrderWriter,
)
from pure_integer_ai.cognition.understanding.order_hypothesis_adapter import (
    OccurrenceOrderHypothesisAdapter,
    TypedOrderProjection,
)
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.experiments.formal_train import make_train_context


def _source(document_id: int, *, parser: int = 1,
            source_id: int = 16100) -> SourceRef:
    """构造真实观察来源，并允许同文档切换 parser version。"""
    return SourceRef(
        71,
        source_id,
        document_id,
        GLOBAL_OWNER_SCOPE,
        VersionBundle(parser=ParserVersion(parser)),
    )


def _protocol() -> OrderLearningProtocol:
    """构造与任何真实观察分离的版本化 aggregate manifest。"""
    manifest = SourceRef(
        72,
        16200,
        0,
        GLOBAL_OWNER_SCOPE,
        VersionBundle(),
    )
    return OrderLearningProtocol(
        (16210, 1),
        (16211, 1),
        (16211, 2),
        (16211, 3),
        (16212, 1),
        manifest,
        document_scope(manifest),
    )


def _pattern(*, context: int = 1, constraint: int = 1,
             condition: int = 0) -> OrderPattern:
    """构造共享结构和规范 slot、但可替换 context/constraint 的模式。"""
    conditions = (
        () if condition == 0
        else (concept_identity((16308, condition)),)
    )
    return OrderPattern(
        language_branch_identity((16300, 1)),
        concept_identity((16301, 1)),
        structure_concept_identity((16302, 1)),
        structure_concept_identity((16303, 1)),
        structure_concept_identity((16304, 1)),
        structure_concept_identity((16304, 2)),
        concept_identity((16305, constraint)),
        concept_identity((16306, context)),
        conditions,
    )


def _observation(
        pattern: OrderPattern, source: SourceRef, event: int, *,
        reverse_positions: bool = False) -> OrderObservation:
    """用不同 occurrence surface span 构造同一 typed slot 模式的观察。"""
    first_position, second_position = (
        (1, 0) if reverse_positions else (0, 1))
    return OrderObservation(
        source,
        document_scope(source),
        (16400, event),
        pattern.language_branch,
        pattern.structure_family,
        pattern.structure_candidate,
        pattern.first_slot,
        pattern.second_slot,
        pattern.context,
        pattern.conditions,
        occurrence_identity(source, start=event * 2, end=event * 2 + 1,
                            ordinal=0),
        occurrence_identity(source, start=event * 2 + 1, end=event * 2 + 2,
                            ordinal=0),
        first_position,
        second_position,
        (first_position, second_position),
    )


def _stance(value: int, detail: int):
    """返回只由测试参数决定的注入式三态 verifier。"""
    return lambda _pattern, _observation: OrderAssessment(
        value, (16410, detail))


def test_replaced_vocabulary_shares_pattern_and_frequency_does_not_hide_refute():
    """不同词面 occurrence 共享 typed 模式，高频支持不能投票吞掉反例。"""
    engine = OrderHypothesisEngine(_protocol())
    pattern = _pattern()
    results = []
    for document_id in range(1, 6):
        results.append(engine.accumulate(
            pattern,
            _observation(pattern, _source(document_id), document_id),
            _stance(EVIDENCE_SUPPORT, document_id),
            timestamp_seq=document_id,
        ))
    refute = engine.accumulate(
        pattern,
        _observation(pattern, _source(6), 6, reverse_positions=True),
        _stance(EVIDENCE_REFUTE, 6),
        timestamp_seq=6,
    )

    hypothesis = engine.hypothesis_for(pattern)
    snapshot = engine.ledger.snapshot(hypothesis)
    assert all(item.hypothesis == hypothesis for item in results)
    assert refute.hypothesis == hypothesis
    assert snapshot.epistemic_status == EPISTEMIC_CONFLICTED
    assert snapshot.lifecycle == LIFECYCLE_ACTIVE
    assert len(snapshot.support_evidence_ids) == 5
    assert len(snapshot.refute_evidence_ids) == 1
    assert engine.resolver.state_key() == ((), ())


def test_opposite_orders_in_different_contexts_do_not_compete():
    """A-B 与 B-A 在不同 context 中形成不同竞争组并同时得到支持。"""
    engine = OrderHypothesisEngine(_protocol())
    forward = _pattern(context=2, constraint=1)
    reverse = _pattern(context=3, constraint=2)
    forward_result = engine.accumulate(
        forward,
        _observation(forward, _source(11), 1),
        _stance(EVIDENCE_SUPPORT, 1),
        timestamp_seq=1,
    )
    reverse_result = engine.accumulate(
        reverse,
        _observation(reverse, _source(12), 1, reverse_positions=True),
        _stance(EVIDENCE_SUPPORT, 2),
        timestamp_seq=2,
    )

    assert forward_result.hypothesis.competition_key != (
        reverse_result.hypothesis.competition_key)
    assert engine.ledger.snapshot(
        forward_result.hypothesis).epistemic_status == EPISTEMIC_SUPPORTED
    assert engine.ledger.snapshot(
        reverse_result.hypothesis).epistemic_status == EPISTEMIC_SUPPORTED


def test_parser_version_evidence_supersede_replaces_active_stance_only():
    """新 parser 来源可显式替代旧 Evidence，旧事件仍保留完整历史。"""
    engine = OrderHypothesisEngine(_protocol())
    pattern = _pattern()
    old = engine.accumulate(
        pattern,
        _observation(pattern, _source(21, parser=1), 1),
        _stance(EVIDENCE_SUPPORT, 1),
        timestamp_seq=1,
    )
    new = engine.accumulate(
        pattern,
        _observation(pattern, _source(21, parser=2), 1),
        _stance(EVIDENCE_REFUTE, 2),
        timestamp_seq=2,
        supersedes_evidence_id=old.evidence.evidence_id,
    )

    snapshot = engine.ledger.snapshot(old.hypothesis)
    history = engine.ledger.evidence_history(old.hypothesis)
    assert snapshot.epistemic_status == EPISTEMIC_REFUTED
    assert snapshot.support_evidence_ids == ()
    assert snapshot.refute_evidence_ids == (new.evidence.evidence_id,)
    assert len(history) == 2
    assert history[1].supersedes_evidence_id == old.evidence.evidence_id


def test_context_split_archives_broad_parent_and_preserves_multiple_children():
    """宽模式引用 active refute 归档，一对多 child 和旧决策均 append-only 保留。"""
    engine = OrderHypothesisEngine(_protocol())
    parent = _pattern(context=1, constraint=1)
    first_child = _pattern(context=2, constraint=1, condition=1)
    second_child = _pattern(context=3, constraint=2, condition=2)
    parent_support = engine.accumulate(
        parent,
        _observation(parent, _source(31), 1),
        _stance(EVIDENCE_SUPPORT, 1),
        timestamp_seq=1,
    )
    parent_refute = engine.accumulate(
        parent,
        _observation(parent, _source(32), 1, reverse_positions=True),
        _stance(EVIDENCE_REFUTE, 2),
        timestamp_seq=2,
    )
    for timestamp, child in enumerate(
            (first_child, second_child), start=3):
        engine.accumulate(
            child,
            _observation(child, _source(30 + timestamp), 1,
                         reverse_positions=child is second_child),
            _stance(EVIDENCE_SUPPORT, timestamp),
            timestamp_seq=timestamp,
        )

    event = engine.split_context(
        parent,
        (second_child, first_child),
        verifier=lambda _parent, _children: OrderContextSplitAssessment(
            True, (16500, 1)),
        reason_evidence_id=parent_refute.evidence.evidence_id,
        timestamp_seq=5,
    )
    state = engine.state_key()
    replay = engine.split_context(
        parent,
        (first_child, second_child),
        verifier=lambda _parent, _children: OrderContextSplitAssessment(
            True, (16500, 1)),
        reason_evidence_id=parent_refute.evidence.evidence_id,
        timestamp_seq=5,
    )

    assert replay == event
    assert engine.state_key() == state
    assert engine.ledger.snapshot(
        parent_support.hypothesis).lifecycle == LIFECYCLE_ARCHIVED
    assert len(event.children) == 2
    assert all(
        engine.ledger.snapshot(child).lifecycle == LIFECYCLE_ACTIVE
        for child in event.children
    )
    assert len(engine.resolver.decision_history(parent_support.hypothesis)) == 1
    assert engine.split_history(parent) == (event,)


def test_invalid_context_split_has_zero_partial_writes():
    """未登记 child 使 split 失败时，不得先归档 parent 或写 resolver 决策。"""
    engine = OrderHypothesisEngine(_protocol())
    parent = _pattern(context=1)
    registered = _pattern(context=2, condition=1)
    missing = _pattern(context=3, constraint=2, condition=2)
    engine.accumulate(
        parent,
        _observation(parent, _source(41), 1),
        _stance(EVIDENCE_SUPPORT, 1),
        timestamp_seq=1,
    )
    refute = engine.accumulate(
        parent,
        _observation(parent, _source(42), 1),
        _stance(EVIDENCE_REFUTE, 2),
        timestamp_seq=2,
    )
    engine.register_pattern(registered)
    before = engine.state_key()

    with pytest.raises(ValueError, match="先完整登记"):
        engine.split_context(
            parent,
            (registered, missing),
            verifier=lambda _parent, _children: OrderContextSplitAssessment(
                True, (16510, 1)),
            reason_evidence_id=refute.evidence.evidence_id,
            timestamp_seq=3,
        )

    assert engine.state_key() == before
    assert engine.ledger.snapshot(
        engine.hypothesis_for(parent)).lifecycle == LIFECYCLE_ACTIVE

    engine.register_pattern(missing)
    before_rejected = engine.state_key()
    with pytest.raises(ValueError, match="领域 verifier"):
        engine.split_context(
            parent,
            (registered, missing),
            verifier=lambda _parent, _children: OrderContextSplitAssessment(
                False, (16510, 2)),
            reason_evidence_id=refute.evidence.evidence_id,
            timestamp_seq=3,
        )
    assert engine.state_key() == before_rejected


def test_h02a_refute_shares_ledger_and_derived_indexes_rebuild():
    """H-02A 乱序反例进入同一 ledger，H-06 重建时不误解析外部 payload。"""
    engine = OrderHypothesisEngine(_protocol())
    pattern = _pattern()
    source = _source(51)
    support = engine.accumulate(
        pattern,
        _observation(pattern, source, 1),
        _stance(EVIDENCE_SUPPORT, 1),
        timestamp_seq=1,
    )
    observation = _observation(pattern, source, 2)
    trace = build_permutation_trace(
        (observation.first_occurrence, observation.second_occurrence),
        output_order=(1, 0),
        transform_key=(16600, 1),
        source=source,
        scope=document_scope(source),
    )
    OrderPerturbationAdapter(engine).evaluate(
        pattern,
        observation,
        trace,
        lambda _pattern, _observation, _trace: OrderAssessment(
            EVIDENCE_REFUTE, (16602, 1)),
        timestamp_seq=2,
    )
    before = engine.state_key()

    engine.clear_derived_indexes()
    engine.rebuild_derived_indexes()

    assert engine.state_key() == before
    assert engine.ledger.snapshot(
        support.hypothesis).epistemic_status == EPISTEMIC_CONFLICTED


def test_clone_accumulates_without_mutating_host():
    """评测试算 clone 可追加新来源 Evidence，宿主完整状态保持位级不变。"""
    engine = OrderHypothesisEngine(_protocol())
    pattern = _pattern()
    engine.accumulate(
        pattern,
        _observation(pattern, _source(61), 1),
        _stance(EVIDENCE_SUPPORT, 1),
        timestamp_seq=1,
    )
    host = engine.state_key()
    cloned = engine.clone()

    cloned.accumulate(
        pattern,
        _observation(pattern, _source(62), 1),
        _stance(EVIDENCE_REFUTE, 2),
        timestamp_seq=2,
    )

    assert engine.state_key() == host
    assert cloned.state_key() != host


def test_l06_adapter_requires_typed_mapping_and_preserves_real_assertion():
    """adapter 只接受 mapper 显式槽位映射，并完整复制 L-06 反向端点和断言。"""
    backend = DictBackend()
    try:
        context = make_train_context(backend)
        ontology = context.graph_ontology
        scoped = context.scoped_identity_store
        occurrences = OccurrenceIndex(
            ontology, scoped, OccurrenceProtocol((16700, 1)))
        facts = OrderFactIndex(ontology, scoped)
        order_protocol = OccurrenceOrderProtocol((16700, 2))
        writer = OccurrenceOrderWriter(facts, order_protocol)
        reader = OccurrenceOrderReader(facts, occurrences, order_protocol)
        source = _source(71, parser=4)
        scope = document_scope(source)
        previous = occurrences.record(
            source=source, raw_text="甲乙", scope=scope,
            start=0, end=1, ordinal=0,
            segment_index=0, local_index=0, document_index=0,
        )
        current = occurrences.record(
            source=source, raw_text="甲乙", scope=scope,
            start=1, end=2, ordinal=0,
            segment_index=0, local_index=1, document_index=1,
        )
        fact = writer.record_adjacent(
            previous.occurrence,
            current.occurrence,
            source=source,
            scope=scope,
            previous_position=0,
            current_position=1,
        )
        pattern = _pattern(context=4, constraint=2)
        adapter = OccurrenceOrderHypothesisAdapter(reader)

        mapped = adapter.project(
            scope,
            lambda _step: (TypedOrderProjection(pattern, (1, 0)),),
        )

        assert len(mapped) == 1
        observation = mapped[0].observation
        assert observation.first_occurrence == ontology.identity_of(
            current.occurrence)
        assert observation.second_occurrence == ontology.identity_of(
            previous.occurrence)
        assert (observation.first_position, observation.second_position) == (
            1, 0)
        assert observation.qualifiers == (0, 1)
        assert observation.event_key == fact.statement.assertion.stable_key()

        with pytest.raises(ValueError, match="完整映射"):
            adapter.project(
                scope,
                lambda _step: (TypedOrderProjection(pattern, (0, 0)),),
            )
    finally:
        backend.close()
