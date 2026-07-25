"""H-02A 完整扰动 trace、三态 Evidence 和分词边界 adapter 对抗测试。"""
from __future__ import annotations

import pytest

from pure_integer_ai.cognition.shared.hypothesis import (
    EPISTEMIC_REFUTED,
    EPISTEMIC_UNKNOWN,
    LIFECYCLE_ACTIVE,
    HypothesisKey,
    HypothesisLedger,
    HypothesisSnapshot,
)
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    OBJECT_REPRESENTATION,
    ObjectIdentity,
    SourceRef,
    VersionBundle,
)
from pure_integer_ai.cognition.shared.perturbation import (
    ASSESSMENT_CHANGED,
    ASSESSMENT_EQUIVALENT,
    ASSESSMENT_UNKNOWN,
    PerturbationAssessment,
    PerturbationEngine,
    PerturbationProtocol,
    PerturbationTrace,
    SourceDuplicateLedger,
    build_permutation_trace,
    build_replacement_trace,
)
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.cognition.understanding.segmentation_candidates import (
    SegmentationCandidate,
    SegmentationPart,
)
from pure_integer_ai.cognition.understanding.segmentation_hypothesis import (
    SegmentationHypothesisCandidate,
    SegmentationResult,
)
from pure_integer_ai.cognition.understanding.segmentation_perturbation import (
    SegmentationPerturbationAdapter,
    SegmentationPerturbationProtocol,
)
from pure_integer_ai.cognition.understanding.segmentation_span import (
    SegmentationSpanMaterializer,
    SegmentationSpanProtocol,
)
from pure_integer_ai.cognition.understanding.span_index import (
    SpanIndex,
    SpanProtocol,
)
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.storage.backend import DictBackend


def _source(document_id: int) -> SourceRef:
    """构造互相隔离且可稳定恢复的测试来源。"""
    return SourceRef(
        1,
        1200,
        document_id,
        GLOBAL_OWNER_SCOPE,
        VersionBundle(),
    )


def _unit(value: int) -> ObjectIdentity:
    """构造不绑定语言或 Unicode 的一等表示对象。"""
    return ObjectIdentity(OBJECT_REPRESENTATION, (12000, value))


def _hypothesis(
        source: SourceRef, marker: int, *,
        competition_key: tuple[int, ...] = (12100, 1),
        ) -> HypothesisKey:
    """构造同来源下可选择共享或隔离竞争边界的 H-00 候选。"""
    return HypothesisKey(
        (12101, 1),
        (12102, marker),
        competition_key,
        document_scope(source),
        source,
    )


def _protocol() -> PerturbationProtocol:
    """注入测试使用的反驳、unknown 和重复诊断键。"""
    return PerturbationProtocol(
        refute_reason_key=(12200, 1),
        unknown_reason_key=(12200, 2),
        duplicate_transform_key=(12200, 3),
    )


def _engine(
        source: SourceRef, *, candidate_count: int = 2,
        ) -> tuple[PerturbationEngine, tuple[HypothesisKey, ...]]:
    """建立拥有给定候选的真实 ledger，并返回绑定后的扰动 engine。"""
    ledger = HypothesisLedger()
    candidates = tuple(
        _hypothesis(source, index + 1)
        for index in range(candidate_count)
    )
    for candidate in candidates:
        ledger.register(candidate)
    return PerturbationEngine(_protocol(), ledger=ledger), candidates


def _permutation(source: SourceRef):
    """构造三对象循环移位 trace，供三态和隔离测试复用。"""
    return build_permutation_trace(
        (_unit(1), _unit(2), _unit(3)),
        output_order=(2, 0, 1),
        transform_key=(12300, 1),
        source=source,
        scope=document_scope(source),
        metadata_keys=((12300, 2),),
    )


def test_permutation_trace_is_complete_stable_and_rejects_invalid_orders():
    """乱序 trace 保存双侧变化位置，且拒绝丢项、重复、恒等和表面未变化。"""
    source = _source(1)
    trace = _permutation(source)

    assert trace.original == (_unit(1), _unit(2), _unit(3))
    assert trace.transformed == (_unit(3), _unit(1), _unit(2))
    assert trace.output_to_input == (2, 0, 1)
    assert trace.affected_input_positions == (0, 1, 2)
    assert trace.affected_output_positions == (0, 1, 2)
    assert trace.stable_key() == _permutation(source).stable_key()

    for invalid in ((0, 1), (0, 0, 2), (0, 1, 2)):
        with pytest.raises(ValueError):
            build_permutation_trace(
                trace.original,
                output_order=invalid,
                transform_key=(12300, 1),
                source=source,
                scope=document_scope(source),
            )
    with pytest.raises(ValueError, match="真实改变"):
        build_permutation_trace(
            (_unit(1), _unit(1)),
            output_order=(1, 0),
            transform_key=(12300, 1),
            source=source,
            scope=document_scope(source),
        )


def test_replacement_trace_recovers_insert_delete_and_rejects_false_mapping():
    """替换 trace 从映射恢复删除、插入和位移，非 -1 映射不得偷换身份。"""
    source = _source(2)
    trace = build_replacement_trace(
        (_unit(1), _unit(2)),
        (_unit(2), _unit(3)),
        output_to_input=(1, -1),
        transform_key=(12400, 1),
        source=source,
        scope=document_scope(source),
    )

    assert trace.affected_input_positions == (0, 1)
    assert trace.affected_output_positions == (0, 1)
    packed = trace.stable_key()
    assert all(value in packed for value in _unit(1).stable_key())
    assert all(value in packed for value in _unit(3).stable_key())

    with pytest.raises(ValueError, match="身份必须完全相同"):
        build_replacement_trace(
            (_unit(1),),
            (_unit(2),),
            output_to_input=(0,),
            transform_key=(12400, 1),
            source=source,
            scope=document_scope(source),
        )


def test_changed_refutes_only_verifier_target_and_rejects_scope_or_authority_leak():
    """CHANGED 只污染 verifier 指定候选，跨来源和候选外反驳均 fail closed。"""
    source = _source(3)
    engine, candidates = _engine(source)
    trace = _permutation(source)
    result = engine.evaluate(
        trace,
        candidates=candidates,
        verifier=lambda _trace, _candidates: PerturbationAssessment(
            ASSESSMENT_CHANGED,
            (12500, 1),
            (candidates[0],),
        ),
        evidence_source=source,
        timestamp_seq=1,
    )

    assert len(result.evidence_ids) == 1
    assert result.snapshots[0].epistemic_status == EPISTEMIC_REFUTED
    assert engine.ledger.evidence_history(candidates[1]) == ()
    assert engine.ledger.snapshot(candidates[1]).epistemic_status == (
        EPISTEMIC_UNKNOWN)

    outside = _hypothesis(source, 3)
    engine.ledger.register(outside)
    with pytest.raises(ValueError, match="candidates 外"):
        engine.evaluate(
            trace,
            candidates=candidates,
            verifier=lambda _trace, _candidates: PerturbationAssessment(
                ASSESSMENT_CHANGED,
                (12500, 2),
                (outside,),
            ),
            evidence_source=source,
            timestamp_seq=2,
        )

    other_source = _source(4)
    foreign = _hypothesis(other_source, 1)
    engine.ledger.register(foreign)
    with pytest.raises(ValueError, match="同一来源和 scope"):
        engine.evaluate(
            trace,
            candidates=(foreign,),
            verifier=lambda _trace, _candidates: PerturbationAssessment(
                ASSESSMENT_UNKNOWN,
                (12500, 3),
            ),
            evidence_source=source,
            timestamp_seq=3,
        )


def test_equivalent_writes_nothing_and_unknown_keeps_lifecycle_active():
    """EQUIVALENT 保持 ledger 位级不变，UNKNOWN 只追加 unknown 而不退出候选。"""
    source = _source(5)
    engine, candidates = _engine(source)
    trace = _permutation(source)
    before = engine.ledger.state_key()

    equivalent = engine.evaluate(
        trace,
        candidates=candidates,
        verifier=lambda _trace, _candidates: PerturbationAssessment(
            ASSESSMENT_EQUIVALENT,
            (12600, 1),
        ),
        evidence_source=source,
        timestamp_seq=1,
    )
    assert equivalent.evidence_ids == ()
    assert engine.ledger.state_key() == before

    unknown = engine.evaluate(
        trace,
        candidates=candidates,
        verifier=lambda _trace, _candidates: PerturbationAssessment(
            ASSESSMENT_UNKNOWN,
            (12600, 2),
        ),
        evidence_source=source,
        timestamp_seq=2,
    )
    assert len(unknown.evidence_ids) == len(candidates)
    for candidate in candidates:
        snapshot = engine.ledger.snapshot(candidate)
        assert snapshot.lifecycle == LIFECYCLE_ACTIVE
        assert snapshot.epistemic_status == EPISTEMIC_UNKNOWN
        assert snapshot.refute_evidence_ids == ()
        assert len(snapshot.unknown_evidence_ids) == 1


def test_source_duplicate_is_exact_diagnostic_and_rejects_content_drift():
    """同源同槽仅精确重放产诊断，内容漂移和 CHANGED 裁决均被拒绝。"""
    source = _source(6)
    scope = document_scope(source)
    duplicate_ledger = SourceDuplicateLedger(
        _protocol().duplicate_transform_key)
    units = (_unit(1), _unit(2))
    assert duplicate_ledger.register(
        units, source=source, scope=scope, event_key=(12700, 1)) is None
    trace = duplicate_ledger.register(
        units, source=source, scope=scope, event_key=(12700, 1))
    assert trace is not None
    assert trace.is_duplicate_diagnostic
    assert trace.output_to_input == (0, 1)
    assert trace.affected_input_positions == ()
    assert trace.affected_output_positions == ()

    with pytest.raises(ValueError, match="身份位置映射"):
        PerturbationTrace(
            _protocol().duplicate_transform_key,
            (_unit(1), _unit(1)),
            (_unit(1), _unit(1)),
            (1, 0),
            (0, 1),
            (0, 1),
            source,
            scope,
            duplicate_of=source,
        )

    with pytest.raises(ValueError, match="内容或 scope 漂移"):
        duplicate_ledger.register(
            (_unit(1), _unit(3)),
            source=source,
            scope=scope,
            event_key=(12700, 1),
        )

    engine, candidates = _engine(source, candidate_count=1)
    with pytest.raises(ValueError, match="语义变化"):
        engine.evaluate(
            trace,
            candidates=candidates,
            verifier=lambda _trace, _candidates: PerturbationAssessment(
                ASSESSMENT_CHANGED,
                (12700, 2),
                candidates,
            ),
            evidence_source=source,
            timestamp_seq=1,
        )


def test_clone_states_isolate_sandbox_evidence_and_duplicate_events():
    """H-00 与重复登记表克隆后的评测写入不修改宿主完整状态。"""
    source = _source(7)
    host_engine, candidates = _engine(source, candidate_count=1)
    host_duplicate = SourceDuplicateLedger(_protocol().duplicate_transform_key)
    host_duplicate.register(
        (_unit(1),),
        source=source,
        scope=document_scope(source),
        event_key=(12800, 1),
    )
    host_hypothesis_state = host_engine.ledger.state_key()
    host_duplicate_state = host_duplicate.state_key()

    sandbox_engine = PerturbationEngine(
        _protocol(), ledger=host_engine.ledger.clone())
    sandbox_engine.evaluate(
        _permutation(source),
        candidates=candidates,
        verifier=lambda _trace, _candidates: PerturbationAssessment(
            ASSESSMENT_UNKNOWN,
            (12800, 3),
        ),
        evidence_source=source,
        timestamp_seq=1,
    )
    sandbox_duplicate = host_duplicate.clone()
    sandbox_duplicate.register(
        (_unit(2),),
        source=source,
        scope=document_scope(source),
        event_key=(12800, 2),
    )

    assert host_engine.ledger.state_key() == host_hypothesis_state
    assert host_duplicate.state_key() == host_duplicate_state
    assert sandbox_engine.ledger.state_key() != host_hypothesis_state
    assert sandbox_duplicate.state_key() != host_duplicate_state


def _span_protocol() -> SegmentationSpanProtocol:
    """注入测试分词 Span 所需的关系和结构概念键。"""
    return SegmentationSpanProtocol(
        span_protocol=SpanProtocol(
            structure_relation_key=(12900, 1),
            constituent_relation_key=(12900, 2),
            occurrence_relation_key=(12900, 3),
            candidate_relation_key=(12900, 4),
        ),
        document_structure_key=(12901, 1),
        part_structure_key=(12901, 2),
        candidate_shape_namespace_key=(12901, 3),
    )


def _span_result(source: SourceRef) -> SegmentationResult:
    """构造同竞争组的细粒度和粗粒度合法分词歧义。"""
    fine = _hypothesis(source, 1, competition_key=(12910, 1))
    coarse = _hypothesis(source, 2, competition_key=(12910, 1))
    snapshots = tuple(
        HypothesisSnapshot(
            hypothesis,
            LIFECYCLE_ACTIVE,
            EPISTEMIC_UNKNOWN,
            (),
            (),
            (),
        )
        for hypothesis in (fine, coarse)
    )
    fine_segmentation = SegmentationCandidate((
        SegmentationPart(0, 1, "甲", False),
        SegmentationPart(1, 2, "乙", False),
    ))
    coarse_segmentation = SegmentationCandidate((
        SegmentationPart(0, 2, "甲乙", False),
    ))
    return SegmentationResult(
        "甲乙",
        (
            SegmentationHypothesisCandidate(
                fine_segmentation, fine, snapshots[0]),
            SegmentationHypothesisCandidate(
                coarse_segmentation, coarse, snapshots[1]),
        ),
        fine,
    )


def test_span_adapter_requires_materialized_same_competition_and_preserves_ambiguity():
    """边界 adapter 只接已物化同组候选，EQUIVALENT 不把合法歧义强制标负。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        protocol = _span_protocol()
        spans = SpanIndex(
            ctx.graph_ontology,
            ctx.scoped_identity_store,
            protocol.span_protocol,
        )
        materializer = SegmentationSpanMaterializer(spans, protocol)
        result = _span_result(_source(8))
        materializer.materialize(result)
        adapter = SegmentationPerturbationAdapter(
            materializer,
            SegmentationPerturbationProtocol((12920, 1)),
        )
        fine, coarse = tuple(
            candidate.hypothesis for candidate in result.candidates)
        trace = adapter.build_boundary_replacement(fine, coarse)

        assert trace.original == (fine.object_identity(),)
        assert trace.transformed == (coarse.object_identity(),)
        assert trace.output_to_input == (-1,)
        assert trace.affected_input_positions == (0,)
        assert trace.affected_output_positions == (0,)
        assert trace.metadata_keys == (
            (2, 1, 0, 1, 1, 1, 2),
            (1, 1, 0, 2),
        )

        ledger = HypothesisLedger()
        ledger.register(fine)
        ledger.register(coarse)
        engine = PerturbationEngine(_protocol(), ledger=ledger)
        before = ledger.state_key()
        result_evidence = engine.evaluate(
            trace,
            candidates=(fine, coarse),
            verifier=lambda _trace, _candidates: PerturbationAssessment(
                ASSESSMENT_EQUIVALENT,
                (12920, 2),
            ),
            evidence_source=fine.observation,
            timestamp_seq=1,
        )
        assert result_evidence.evidence_ids == ()
        assert ledger.state_key() == before

        unmaterialized = _hypothesis(
            fine.observation,
            3,
            competition_key=fine.competition_key,
        )
        with pytest.raises(LookupError, match="先物化"):
            adapter.build_boundary_replacement(fine, unmaterialized)
        cross_competition = _hypothesis(
            fine.observation,
            4,
            competition_key=(12910, 2),
        )
        with pytest.raises(ValueError, match="同一竞争组"):
            adapter.build_boundary_replacement(fine, cross_competition)

        same_boundary = _hypothesis(
            fine.observation,
            5,
            competition_key=fine.competition_key,
        )
        fine_segmentation = result.candidates[0].segmentation
        same_boundary_result = SegmentationResult(
            result.text,
            (
                result.candidates[0],
                SegmentationHypothesisCandidate(
                    fine_segmentation,
                    same_boundary,
                    HypothesisSnapshot(
                        same_boundary,
                        LIFECYCLE_ACTIVE,
                        EPISTEMIC_UNKNOWN,
                        (),
                        (),
                        (),
                    ),
                ),
            ),
            fine,
        )
        materializer.materialize(same_boundary_result)
        with pytest.raises(ValueError, match="真实改变"):
            adapter.build_boundary_replacement(fine, same_boundary)
    finally:
        backend.close()
