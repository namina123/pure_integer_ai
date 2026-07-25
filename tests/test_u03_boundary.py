"""U-03 句界候选、Span 选择、对抗语境和恢复测试。"""
from __future__ import annotations

import pytest

from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EVIDENCE_UNKNOWN,
)
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    ParserVersion,
    SourceRef,
    VersionBundle,
)
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.cognition.understanding.boundary_hypothesis import (
    BoundaryCandidate,
    BoundaryEvidenceProfile,
    BoundaryEvidenceSpec,
    BoundaryHypothesisEngine,
    BoundaryHypothesisProtocol,
)
from pure_integer_ai.cognition.understanding.boundary_span import (
    BoundarySpanMaterializer,
    BoundarySpanProtocol,
)
from pure_integer_ai.cognition.understanding.segmentation_span import (
    SegmentationSpanProtocol,
)
from pure_integer_ai.cognition.understanding.occurrence_index import (
    OccurrenceIndex,
    OccurrenceProtocol,
)
from pure_integer_ai.cognition.understanding.span_index import (
    SpanIndex,
    SpanProtocol,
)
from pure_integer_ai.experiments.formal_train import make_train_context
from pure_integer_ai.experiments.collection import CollectedItem
from pure_integer_ai.experiments.formal_train import (
    DefaultRoundRunner,
    FormalTrainConfig,
    _item_sentence_bounds,
    _split_item_to_segments,
    formal_train,
)
from pure_integer_ai.experiments.evaluation_isolation import isolated_evaluation
from pure_integer_ai.experiments.text_segments import sentence_bounds
from pure_integer_ai.storage.backend import DictBackend, SQLiteBackend
from pure_integer_ai.storage.edge_store import SOURCE_BARE_TEXT
from pure_integer_ai.training.cursor import dump_run, load_run
from pure_integer_ai.training.stages import STAGE1_SKELETON


def _backend(kind: str):
    """为 Dict/SQLite 两类后端创建独立测试实例。"""
    if kind == "dict":
        return DictBackend()
    if kind == "sqlite":
        return SQLiteBackend(":memory:")
    raise ValueError(kind)


def _source(document_id: int, *, parser: int = 7) -> SourceRef:
    """构造带 parser version 的稳定裸文本来源。"""
    return SourceRef(
        SOURCE_BARE_TEXT,
        73001,
        document_id,
        GLOBAL_OWNER_SCOPE,
        VersionBundle(parser=ParserVersion(parser)),
    )


def _span_protocol() -> SpanProtocol:
    """注入 U-03 与 L-04 共用的基础 Span 关系。"""
    return SpanProtocol(
        structure_relation_key=(99300, 1),
        constituent_relation_key=(99300, 2),
        occurrence_relation_key=(99300, 3),
        candidate_relation_key=(99300, 4),
    )


def _boundary_protocol() -> BoundarySpanProtocol:
    """注入不含语言字面量或码点作用的句界协议。"""
    return BoundarySpanProtocol(
        hypothesis_protocol=BoundaryHypothesisProtocol((99310, 1)),
        span_protocol=_span_protocol(),
        document_structure_key=(99310, 2),
        candidate_structure_key=(99310, 3),
        anchor_structure_key=(99310, 4),
        candidate_shape_namespace_key=(99310, 5),
        selection_relation_key=(99310, 6),
        withdrawal_relation_key=(99310, 7),
        selection_clock_kind=993107,
    )


def _segmentation_span_protocol() -> SegmentationSpanProtocol:
    """为 formal_train 装配同一个 L-04 Span 基础协议。"""
    return SegmentationSpanProtocol(
        span_protocol=_span_protocol(),
        document_structure_key=(99311, 1),
        part_structure_key=(99311, 2),
        candidate_shape_namespace_key=(99311, 3),
        atomic_structure_key=None,
    )


def _environment(backend):
    """装配 Hypothesis、occurrence、Span 和句界物化器。"""
    ctx = make_train_context(backend)
    occurrences = OccurrenceIndex(
        ctx.graph_ontology,
        ctx.scoped_identity_store,
        OccurrenceProtocol((99300, 5)),
    )
    spans = SpanIndex(
        ctx.graph_ontology,
        ctx.scoped_identity_store,
        _span_protocol(),
        occurrences,
    )
    protocol = _boundary_protocol()
    engine = BoundaryHypothesisEngine(protocol.hypothesis_protocol)
    materializer = BoundarySpanMaterializer(spans, protocol)
    ctx.occurrence_index = occurrences
    ctx.span_index = spans
    ctx.boundary_hypothesis_engine = engine
    ctx.boundary_span_materializer = materializer
    return ctx, occurrences, spans, engine, materializer


def _profile(*items: tuple[tuple[int, ...], int, tuple[int, ...]]
             ) -> BoundaryEvidenceProfile:
    """把测试中的锚点、立场和理由组装为显式证据 profile。"""
    return BoundaryEvidenceProfile(tuple(
        BoundaryEvidenceSpec(
            BoundaryCandidate(anchors),
            stance,
            reason,
        )
        for anchors, stance, reason in items
    ))


def _char_spans(text: str) -> tuple[tuple[int, int, int], ...]:
    """构造逐码点 winner token span，便于只测试边界映射。"""
    return tuple((index, index + 1, 0) for index in range(len(text)))


def test_without_evidence_keeps_whole_input_and_writes_no_boundary_span():
    """无 Evidence 时不得按字符猜句界，也不应制造空候选图。"""
    backend = DictBackend()
    try:
        _, _, spans, engine, materializer = _environment(backend)
        source = _source(1)
        result = engine.resolve(
            "3.14",
            observation=source,
            scope=document_scope(source),
            language_key=(1,),
        )
        materialized = materializer.materialize(
            result,
            token_spans=_char_spans(result.text),
        )

        assert result.selected_hypothesis is None
        assert materialized.decision.token_cuts(
            _char_spans(result.text)) == ()
        assert sentence_bounds(len(result.text)) == [(0, len(result.text))]
        assert materialized.document is None
        assert spans.span_count() == 0
    finally:
        backend.close()


def test_without_boundary_evidence_does_not_block_language_occurrences():
    """整段保留时 occurrence 仍属语言观察，不要求伪造句界候选。"""
    backend = DictBackend()
    try:
        _, occurrences, spans, engine, materializer = _environment(backend)
        text = "甲乙"
        source = _source(9)
        scope = document_scope(source)
        token_spans = _char_spans(text)
        occurrence_refs = tuple(
            occurrences.record(
                source=source,
                raw_text=text,
                scope=scope,
                start=start,
                end=end,
                ordinal=ordinal,
                segment_index=0,
                local_index=index,
                document_index=index,
            ).occurrence
            for index, (start, end, ordinal) in enumerate(token_spans)
        )
        result = engine.resolve(
            text,
            observation=source,
            scope=scope,
            language_key=(2,),
        )
        materialized = materializer.materialize(
            result,
            token_spans=token_spans,
            occurrence_refs=occurrence_refs,
        )

        assert materialized.decision.anchors == ()
        assert spans.span_count() == 0
        assert occurrences.occurrence_count() == 2
    finally:
        backend.close()


@pytest.mark.parametrize("kind", ["dict", "sqlite"])
def test_explicit_profile_materializes_boundary_spans_and_occurrences(kind):
    """唯一支持候选必须形成根 Span、零宽锚点和 winner occurrence 成员。"""
    backend = _backend(kind)
    try:
        _, occurrences, spans, engine, materializer = _environment(backend)
        text = "甲。乙"
        source = _source(2)
        scope = document_scope(source)
        result = engine.resolve(
            text,
            observation=source,
            scope=scope,
            language_key=(11, 2),
            profile=_profile(((2,), EVIDENCE_SUPPORT, (99320, 1))),
        )
        token_spans = _char_spans(text)
        occurrence_refs = tuple(
            occurrences.record(
                source=source,
                raw_text=text,
                scope=scope,
                start=start,
                end=end,
                ordinal=ordinal,
                segment_index=0 if index < 2 else 1,
                local_index=index if index < 2 else index - 2,
                document_index=index,
            ).occurrence
            for index, (start, end, ordinal) in enumerate(token_spans)
        )
        materialized = materializer.materialize(
            result,
            token_spans=token_spans,
            occurrence_refs=occurrence_refs,
        )

        assert materialized.decision.anchors == (2,)
        assert materialized.decision.token_cuts(token_spans) == (2,)
        assert sentence_bounds(3, cut_after=(2,)) == [(0, 2), (2, 3)]
        assert len(materialized.candidates) == 1
        candidate = materialized.candidates[0]
        assert spans.read(candidate.anchors[0]).members == ((2, 2),)
        assert len({
            spans.read(materialized.document).ordinal,
            spans.read(candidate.root).ordinal,
            spans.read(candidate.anchors[0]).ordinal,
        }) == 3
        assert tuple(
            statement.object
            for statement in spans.read(candidate.root).occurrences
        ) == occurrence_refs
    finally:
        backend.close()


@pytest.mark.parametrize(
    "text,anchors,expected",
    [
        ("3.14", (), ()),
        ("e.g.", (), ()),
        ("obj.field", (), ()),
        ("句.后", (2,), (2,)),
    ],
)
def test_same_dot_representation_has_context_specific_boundary_decisions(
        text, anchors, expected):
    """小数、缩写、成员访问和句末点只由注入证据区分。"""
    backend = DictBackend()
    try:
        _, _, _, engine, _ = _environment(backend)
        source = _source(100 + len(text))
        profile = (
            BoundaryEvidenceProfile()
            if not anchors else _profile(
                (anchors, EVIDENCE_SUPPORT, (99330, len(text))))
        )
        result = engine.resolve(
            text,
            observation=source,
            scope=document_scope(source),
            language_key=(21,),
            profile=profile,
        )
        assert result.decision().token_cuts(_char_spans(text)) == expected
    finally:
        backend.close()


def test_consecutive_marks_and_quote_use_one_injected_anchor_without_empty_segments():
    """连续表示和引号不应自行重复切分，显式单锚点只产生两个非空段。"""
    backend = DictBackend()
    try:
        _, _, _, engine, _ = _environment(backend)
        text = "甲！？』乙"
        source = _source(3)
        result = engine.resolve(
            text,
            observation=source,
            scope=document_scope(source),
            language_key=(31,),
            profile=_profile(((4,), EVIDENCE_SUPPORT, (99340, 1))),
        )
        cuts = result.decision().token_cuts(_char_spans(text))
        assert cuts == (4,)
        assert sentence_bounds(len(text), cut_after=cuts) == [(0, 4), (4, 5)]
    finally:
        backend.close()


def test_same_unicode_sequence_does_not_share_boundary_role_across_languages():
    """同一表示在两个语言竞争键中不得自动共享句界决定。"""
    backend = DictBackend()
    try:
        _, _, _, engine, _ = _environment(backend)
        text = "甲.乙"
        source_a = _source(4)
        source_b = _source(5)
        selected = engine.resolve(
            text,
            observation=source_a,
            scope=document_scope(source_a),
            language_key=(41,),
            profile=_profile(((2,), EVIDENCE_SUPPORT, (99350, 1))),
        )
        undecided = engine.resolve(
            text,
            observation=source_b,
            scope=document_scope(source_b),
            language_key=(42,),
        )

        assert selected.decision().anchors == (2,)
        assert undecided.decision().anchors == ()
    finally:
        backend.close()


def test_feedback_supersedes_wrong_selection_and_dump_resume_restores_it(
        tmp_path):
    """错误边界须同时替代 H-00 生命周期、候选 link 和选择断言。"""
    backend = DictBackend()
    try:
        ctx, _, spans, engine, materializer = _environment(backend)
        text = "甲乙丙"
        source = _source(6)
        scope = document_scope(source)
        profile = _profile(
            ((1,), EVIDENCE_SUPPORT, (99360, 1)),
            ((2,), EVIDENCE_UNKNOWN, (99360, 2)),
        )
        first = engine.resolve(
            text,
            observation=source,
            scope=scope,
            language_key=(51,),
            profile=profile,
        )
        first_materialized = materializer.materialize(
            first,
            token_spans=_char_spans(text),
        )
        old = first.selected_hypothesis
        new = next(
            item.hypothesis for item in first.candidates
            if item.hypothesis != old
        )
        assert old is not None

        engine.record_feedback(
            old,
            stance=EVIDENCE_REFUTE,
            source=source,
            reason_key=(99360, 3),
            timestamp_seq=10,
            replacement=new,
        )
        engine.record_feedback(
            new,
            stance=EVIDENCE_SUPPORT,
            source=source,
            reason_key=(99360, 4),
            timestamp_seq=11,
        )
        corrected = engine.resolve(
            text,
            observation=source,
            scope=scope,
            language_key=(51,),
            profile=profile,
        )
        corrected_materialized = materializer.supersede_selected(
            corrected,
            token_spans=_char_spans(text),
        )

        assert first_materialized.decision.anchors == (1,)
        assert corrected_materialized.decision.anchors == (2,)
        active_hypotheses = {
            spans.ontology.identity_of(statement.subject).components
            for statement in spans.candidate_statements()
        }
        assert old.stable_key() not in active_hypotheses
        assert new.stable_key() in active_hypotheses

        dump_run(
            backend,
            str(tmp_path),
            "u03_resume",
            spaces=[ctx.space_id],
        )
    finally:
        backend.close()

    restored_backend = DictBackend()
    try:
        restored_ctx, _, restored_spans, restored_engine, restored_materializer = (
            _environment(restored_backend))
        load_run(restored_backend, str(tmp_path), "u03_resume")
        empty_result = restored_engine.resolve(
            text,
            observation=source,
            scope=scope,
            language_key=(51,),
        )
        restored = restored_materializer.materialize(
            empty_result,
            token_spans=_char_spans(text),
        )

        assert restored.decision.anchors == (2,)
        assert restored.decision.selected_hypothesis == new
        assert len(restored_spans.candidate_statements()) == 1
        assert restored_ctx.graph_ontology.identity_of(
            restored.selected_statement.object).object_kind > 0
    finally:
        restored_backend.close()


def test_current_undecided_result_withdraws_stale_selection_but_keeps_candidate():
    """新反例使边界未决时撤销旧选择，但不抹掉仍 active+conflicted 的候选。"""
    backend = DictBackend()
    try:
        ctx, _, spans, engine, materializer = _environment(backend)
        text = "甲乙丙"
        source = _source(16)
        scope = document_scope(source)
        profile = _profile(
            ((1,), EVIDENCE_SUPPORT, (99361, 1)),
        )
        selected = engine.resolve(
            text,
            observation=source,
            scope=scope,
            language_key=(52,),
            profile=profile,
        )
        first = materializer.materialize(
            selected,
            token_spans=_char_spans(text),
        )
        hypothesis = selected.selected_hypothesis
        assert hypothesis is not None
        assert first.selected_statement is not None

        engine.record_feedback(
            hypothesis,
            stance=EVIDENCE_REFUTE,
            source=_source(17),
            reason_key=(99361, 2),
            timestamp_seq=10,
        )
        undecided = engine.resolve(
            text,
            observation=source,
            scope=scope,
            language_key=(52,),
            profile=profile,
        )
        withdrawn = materializer.materialize(
            undecided,
            token_spans=_char_spans(text),
        )

        assert undecided.selected_hypothesis is None
        assert len(undecided.candidates) == 1
        assert withdrawn.decision.anchors == ()
        assert withdrawn.selected_statement is None
        assert ctx.scoped_identity_store.assertion_is_superseded(
            first.selected_statement.assertion_hash)
        assert len(spans.candidate_statements()) == 1
        assert len(withdrawn.statement_hashes) > 0
    finally:
        backend.close()


def test_multiple_active_boundary_candidates_withdraw_stale_unique_selection():
    """第二个 active support 造成多解时须撤销旧选择，并保留两个候选 link。"""
    backend = DictBackend()
    try:
        ctx, _, spans, engine, materializer = _environment(backend)
        text = "甲乙丙"
        source = _source(18)
        scope = document_scope(source)
        first_result = engine.resolve(
            text,
            observation=source,
            scope=scope,
            language_key=(53,),
            profile=_profile(((1,), EVIDENCE_SUPPORT, (99362, 1))),
        )
        first = materializer.materialize(
            first_result,
            token_spans=_char_spans(text),
        )
        assert first.selected_statement is not None

        ambiguous = engine.resolve(
            text,
            observation=source,
            scope=scope,
            language_key=(53,),
            profile=_profile(
                ((1,), EVIDENCE_SUPPORT, (99362, 1)),
                ((2,), EVIDENCE_SUPPORT, (99362, 2)),
            ),
        )
        withdrawn = materializer.materialize(
            ambiguous,
            token_spans=_char_spans(text),
        )

        assert ambiguous.selected_hypothesis is None
        assert len(ambiguous.candidates) == 2
        assert withdrawn.decision.anchors == ()
        assert withdrawn.selected_statement is None
        assert ctx.scoped_identity_store.assertion_is_superseded(
            first.selected_statement.assertion_hash)
        assert len(spans.candidate_statements()) == 2
    finally:
        backend.close()


def test_boundary_anchor_must_align_to_current_winner_token_end():
    """边界落入 token 内部时必须拒绝，不能近似映射到最近切点。"""
    backend = DictBackend()
    try:
        _, _, _, engine, _ = _environment(backend)
        source = _source(7)
        result = engine.resolve(
            "甲乙丙",
            observation=source,
            scope=document_scope(source),
            language_key=(61,),
            profile=_profile(((1,), EVIDENCE_SUPPORT, (99370, 1))),
        )
        with pytest.raises(ValueError, match="无法唯一对齐"):
            result.decision().token_cuts(((0, 2, 0), (2, 3, 0)))
    finally:
        backend.close()


def test_evaluation_clone_keeps_boundary_evidence_and_spans_off_host():
    """V-06 沙箱中的句界 Evidence、Span 和选择不得回写正式上下文。"""
    backend = DictBackend()
    try:
        ctx, _, spans, engine, _ = _environment(backend)
        host_span_count = spans.span_count()
        host_ledger_state = engine.ledger.state_key()
        with isolated_evaluation(ctx, label="u03") as eval_ctx:
            text = "甲。乙"
            source = _source(10)
            result = eval_ctx.boundary_hypothesis_engine.resolve(
                text,
                observation=source,
                scope=document_scope(source),
                language_key=(71,),
                profile=_profile(
                    ((2,), EVIDENCE_SUPPORT, (99375, 1))),
            )
            eval_ctx.boundary_span_materializer.materialize(
                result,
                token_spans=_char_spans(text),
            )
            assert eval_ctx.span_index.span_count() > host_span_count

        assert spans.span_count() == host_span_count
        assert engine.ledger.state_key() == host_ledger_state
    finally:
        backend.close()


def test_formal_train_consumes_same_boundary_decision_everywhere(
        tmp_path, monkeypatch):
    """正式入口、observe 分段和共享 helper 必须消费同一 active 图决定。"""
    from pure_integer_ai.training import stages as training_stages

    monkeypatch.setattr(training_stages, "FLOOR_GRAPH_SIZE_S1", 0)
    text = "甲。乙"
    source = _source(8)
    item = CollectedItem(
        tokens=list(text),
        raw_text=text,
        boundary_profile=_profile(
            ((2,), EVIDENCE_SUPPORT, (99380, 1))),
        source=SOURCE_BARE_TEXT,
        source_ref=source,
    )
    config = FormalTrainConfig(
        run_dir=str(tmp_path),
        run_id="u03_formal",
        rounds_per_stage=1,
        active_training_stages=(STAGE1_SKELETON,),
        persist_graph_dump=False,
        language_occurrence_protocol=OccurrenceProtocol((99380, 2)),
        language_span_protocol=_segmentation_span_protocol(),
        language_boundary_protocol=_boundary_protocol(),
    )

    result = formal_train(
        config,
        [item],
        backend=DictBackend(),
        runner=DefaultRoundRunner(),
    )

    assert _item_sentence_bounds(item) == [(0, 2), (2, 3)]
    assert len(_split_item_to_segments(item)) == 2
    assert item.boundary_decision.anchors == (2,)
    assert result.occurrence_count == 3
    assert result.span_count == 3
    assert result.span_candidate_fact_count == 1


def test_active_formal_train_code_contains_no_sentence_end_character_set():
    """U-03 完成后 active 编排器不得保留或改名句末字符白名单。"""
    import inspect
    from pure_integer_ai.experiments import formal_train as formal_train_module

    source = inspect.getsource(formal_train_module)
    assert "_SENT_END_CHARS" not in source
    assert "frozenset(\"。.!?" not in source
