"""H-01 遮蔽预测、条件计数、语言 occurrence 接线和隔离回归。"""
from __future__ import annotations

import pytest

from pure_integer_ai.cognition.shared.hypothesis import (
    EPISTEMIC_REFUTED,
    LIFECYCLE_ACTIVE,
)
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    OBJECT_REPRESENTATION,
    ObjectIdentity,
    SourceRef,
    VersionBundle,
)
from pure_integer_ai.cognition.shared.prediction import (
    ConditionalPredictionModel,
    PredictionEngine,
    PredictionProtocol,
    build_masked_example,
)
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.cognition.shared.types import (
    LANG_ZH,
    MODALITY_LANGUAGE,
    ObserveResult,
)
from pure_integer_ai.cognition.understanding.language_prediction import (
    LanguagePredictionProtocol,
    LanguagePredictionRuntime,
)
from pure_integer_ai.cognition.understanding.occurrence_index import (
    OccurrenceIndex,
    OccurrenceProtocol,
)
from pure_integer_ai.experiments.collection import CollectedItem
from pure_integer_ai.experiments.evaluation_isolation import isolated_evaluation
from pure_integer_ai.experiments.language_observation import (
    _run_item_predictions,
)
from pure_integer_ai.experiments.language_protocol_runtime import (
    install_language_graph_protocols,
)
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.storage.backend import DictBackend


def _source(source_id: int) -> SourceRef:
    """构造不同文档的稳定裸文本来源。"""
    return SourceRef(
        1,
        source_id,
        source_id,
        GLOBAL_OWNER_SCOPE,
        VersionBundle(),
    )


def _unit(value: int) -> ObjectIdentity:
    """构造与具体 Unicode 或语言无关的表示概念身份。"""
    return ObjectIdentity(OBJECT_REPRESENTATION, (88000, value))


def _prediction_protocol() -> PredictionProtocol:
    """返回测试注入的预测 kind、理由和整数分数单位。"""
    return PredictionProtocol(
        hypothesis_kind_key=(88010, 1),
        support_reason_key=(88010, 2),
        refute_reason_key=(88010, 3),
        unknown_reason_key=(88010, 4),
        score_scale=1000,
    )


def _language_protocol() -> LanguagePredictionProtocol:
    """返回同时覆盖单单元和多单元 masked/next 的语言协议。"""
    return LanguagePredictionProtocol(
        prediction=_prediction_protocol(),
        masked_objective_key=(88020, 1),
        next_objective_key=(88020, 2),
        condition_namespace_key=(88020, 3),
        unit_object_kinds=(OBJECT_REPRESENTATION,),
        masked_widths=(1, 2),
        next_widths=(1, 2),
        candidate_limit=8,
        clock_kind=88021,
    )


def _example(
        source: SourceRef, units: tuple[ObjectIdentity, ...], *,
        target_start: int = 1,
        condition: tuple[int, ...] = (88030, 1),
        ):
    """构造共享结构条件但完整可见内容可变化的单目标样本。"""
    return build_masked_example(
        units,
        target_start=target_start,
        target_width=1,
        objective_key=(88031, 1),
        source=source,
        reveal_suffix=True,
        condition_keys=(condition,),
    )


def _record_document(
        ctx, occurrences: OccurrenceIndex, source: SourceRef,
        unit_values: tuple[int, ...],
        ) -> tuple:
    """把表示概念物化为同一来源的 occurrence 序列。"""
    raw_text = "x" * len(unit_values)
    scope = document_scope(source)
    refs = []
    for index, value in enumerate(unit_values):
        representation = ctx.graph_ontology.materialize(_unit(value))
        record = occurrences.record(
            source=source,
            raw_text=raw_text,
            scope=scope,
            start=index,
            end=index + 1,
            ordinal=0,
            segment_index=0,
            local_index=index,
            document_index=index,
            typed_candidates=(representation,),
        )
        refs.append(record.occurrence)
    return tuple(refs)


def test_masked_context_physically_excludes_target_and_next_hides_suffix():
    """masked/next builder 不得把目标或 next 后文留在 predictor 输入对象中。"""
    source = _source(1)
    units = (_unit(1), _unit(2), _unit(3), _unit(4))
    masked = build_masked_example(
        units,
        target_start=1,
        target_width=2,
        objective_key=(88100, 1),
        source=source,
        reveal_suffix=True,
    )
    assert masked.context.visible_prefix == (_unit(1),)
    assert masked.context.visible_suffix == (_unit(4),)
    assert masked.target.units == (_unit(2), _unit(3))

    next_example = build_masked_example(
        units,
        target_start=1,
        target_width=1,
        objective_key=(88100, 2),
        source=source,
        reveal_suffix=False,
    )
    assert next_example.context.visible_prefix == (_unit(1),)
    assert next_example.context.visible_suffix == ()
    assert next_example.target.units == (_unit(2),)


def test_conditional_model_excludes_current_source_and_uses_injected_backoff():
    """同源观察不可自答，内容替换后仍可沿调用方结构条件比较候选。"""
    first_source = _source(2)
    second_source = _source(3)
    first = _example(first_source, (_unit(1), _unit(2), _unit(3)))
    changed = _example(second_source, (_unit(4), _unit(5), _unit(6)))
    model = ConditionalPredictionModel(1000)
    assert model.observe(first) == first
    assert model.observe(first) == first
    assert model.observation_count() == 1

    assert model.known_targets(
        first.context,
        candidate_limit=4,
        excluded_sources=(first_source,),
    ) == ()
    candidates = model.known_targets(
        changed.context,
        candidate_limit=4,
        excluded_sources=(second_source,),
    )
    assert candidates == (first.target,)
    score = model.score(
        changed.context,
        first.target,
        excluded_sources=(second_source,),
    )
    assert score.condition_level == 1
    assert (score.support_count, score.total_count, score.scaled_score) == (
        1, 1, 1000)

    conflicting = build_masked_example(
        (_unit(1), _unit(7), _unit(3)),
        target_start=1,
        target_width=1,
        objective_key=first.context.objective_key,
        source=first_source,
        reveal_suffix=True,
        condition_keys=first.context.condition_keys,
        event_key=first.event_key,
    )
    with pytest.raises(ValueError, match="事件槽"):
        model.observe(conflicting)


def test_prediction_failure_refutes_only_formed_candidate_hypothesis():
    """揭示目标发生在 predict 之后，失败只向本次候选追加反驳 Evidence。"""
    training_source = _source(4)
    evaluation_source = _source(5)
    trained = _example(training_source, (_unit(1), _unit(2), _unit(3)))
    query = _example(evaluation_source, (_unit(4), _unit(5), _unit(6)))
    engine = PredictionEngine(_prediction_protocol())
    engine.observe(trained)

    predicted = engine.predict(
        query.context,
        observation=evaluation_source,
        scope=document_scope(evaluation_source),
        candidate_limit=4,
    )
    assert len(predicted.candidates) == 1
    assert predicted.candidates[0].target == trained.target
    assert query.target != trained.target
    evaluated = engine.evaluate(
        predicted,
        expected=query.target,
        evidence_source=evaluation_source,
        timestamp_seq=1,
    )
    assert evaluated.candidates[0].snapshot.epistemic_status == EPISTEMIC_REFUTED
    assert evaluated.candidates[0].snapshot.lifecycle != LIFECYCLE_ACTIVE
    assert predicted.selection_decision_id > 0
    assert evaluated.evaluation_decision_id > 0
    assert engine.ledger.evidence_history(
        evaluated.candidates[0].hypothesis)[0].source == evaluation_source


def test_language_runtime_predicts_before_learning_entire_document():
    """第二文档的新目标不得在同文档后序预测中成为候选，重放也不得重复计数。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        occurrences = OccurrenceIndex(
            ctx.graph_ontology,
            ctx.scoped_identity_store,
            OccurrenceProtocol((88200, 1)),
        )
        runtime = LanguagePredictionRuntime(
            ctx.graph_ontology,
            occurrences,
            _language_protocol(),
        )
        first_refs = _record_document(ctx, occurrences, _source(6), (1, 2, 3))
        first = runtime.observe_document(first_refs)
        assert first.prediction_count == 8
        assert first.candidate_count == 0
        assert runtime.engine.model.observation_count() == 8
        assert runtime.observe_document(first_refs) is first
        assert runtime.engine.model.observation_count() == 8

        second_refs = _record_document(ctx, occurrences, _source(7), (4, 4, 4))
        second = runtime.observe_document(second_refs)
        assert second.prediction_count == 8
        assert second.candidate_count > 0
        assert all(
            candidate.target.units != (_unit(4),)
            for evaluation in second.evaluations
            for candidate in evaluation.result.candidates
        )
        assert runtime.engine.model.observation_count() == 16
    finally:
        backend.close()


def test_protocol_installer_round_boundary_and_v06_clone_are_isolated():
    """正式薄接线暴露结果，V-06 评测学习不得修改宿主预测模型或报告。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        occurrence_protocol = OccurrenceProtocol((88300, 1))
        install_language_graph_protocols(
            ctx,
            occurrence_protocol=occurrence_protocol,
            prediction_protocol=_language_protocol(),
        )
        host_refs = _record_document(
            ctx, ctx.occurrence_index, _source(8), (1, 2, 3))
        observed = ObserveResult(occurrence_refs=list(host_refs))
        item = CollectedItem(
            tokens=["x", "x", "x"],
            modality=MODALITY_LANGUAGE,
            lang=LANG_ZH,
            source=1,
            source_ref=_source(8),
            raw_text="xxx",
        )
        _run_item_predictions(ctx, item, observed)
        assert len(observed.prediction_results) == 8
        assert len(ctx.language_prediction_reports) == 1
        host_state = ctx.language_prediction_runtime.state_key()

        with isolated_evaluation(ctx, label="h01") as eval_ctx:
            eval_refs = _record_document(
                eval_ctx,
                eval_ctx.occurrence_index,
                _source(9),
                (1, 2, 3),
            )
            report = eval_ctx.language_prediction_runtime.observe_document(
                eval_refs)
            assert report.candidate_count > 0
            assert eval_ctx.language_prediction_runtime.state_key() != host_state

        assert ctx.language_prediction_runtime.state_key() == host_state
        assert len(ctx.language_prediction_reports) == 1
    finally:
        backend.close()


def test_prediction_protocol_requires_occurrence_and_unique_typed_unit():
    """缺 L-03 或 occurrence 多义时必须 fail closed/跳过，不能按 ordinal 偷选。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        with pytest.raises(ValueError, match="L-03 occurrence"):
            install_language_graph_protocols(
                ctx,
                prediction_protocol=_language_protocol(),
            )

        occurrences = OccurrenceIndex(
            ctx.graph_ontology,
            ctx.scoped_identity_store,
            OccurrenceProtocol((88400, 1)),
        )
        runtime = LanguagePredictionRuntime(
            ctx.graph_ontology,
            occurrences,
            _language_protocol(),
        )
        source = _source(10)
        first = ctx.graph_ontology.materialize(_unit(1))
        second = ctx.graph_ontology.materialize(_unit(2))
        ambiguous = occurrences.record(
            source=source,
            raw_text="x",
            scope=document_scope(source),
            start=0,
            end=1,
            ordinal=0,
            segment_index=0,
            local_index=0,
            document_index=0,
            typed_candidates=(first, second),
        )
        report = runtime.observe_document((ambiguous.occurrence,))
        assert report.unresolved_occurrences == (ambiguous.occurrence,)
        assert report.prediction_count == 0
        assert runtime.engine.model.observation_count() == 0
    finally:
        backend.close()
