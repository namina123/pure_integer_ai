"""L-04 递归 Span、完整成员和候选替代测试。"""
from __future__ import annotations

import pytest

from pure_integer_ai.cognition.shared.hypothesis import (
    EPISTEMIC_UNKNOWN,
    EVIDENCE_REFUTE,
    LIFECYCLE_ACTIVE,
    HypothesisKey,
    HypothesisSnapshot,
)
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    ParserVersion,
    SourceRef,
    VersionBundle,
    language_branch_identity,
    normalize_span_members,
    span_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    CLOCK_OBSERVATION,
    LogicalClock,
    LogicalClockIdentity,
    document_scope,
)
from pure_integer_ai.cognition.understanding.occurrence_index import (
    OccurrenceIndex,
    OccurrenceProtocol,
)
from pure_integer_ai.cognition.understanding.segmentation_hypothesis import (
    SegmentationHypothesisCandidate,
    SegmentationHypothesisEngine,
    SegmentationProtocol,
    SegmentationResult,
)
from pure_integer_ai.cognition.understanding.segmentation_candidates import (
    SegmentationCandidate,
    SegmentationPart,
)
from pure_integer_ai.cognition.understanding.segmentation_span import (
    SegmentationSpanMaterializer,
    SegmentationSpanProtocol,
)
from pure_integer_ai.cognition.understanding.span_index import (
    SpanIndex,
    SpanProtocol,
)
from pure_integer_ai.cognition.understanding.word_form_provider import (
    WordFormProvider,
)
from pure_integer_ai.experiments.collection import CollectedItem
from pure_integer_ai.experiments.evaluation_isolation import isolated_evaluation
from pure_integer_ai.experiments.formal_train import (
    DefaultRoundRunner,
    FormalTrainConfig,
    formal_train,
    make_train_context,
)
from pure_integer_ai.storage.backend import DictBackend, SQLiteBackend
from pure_integer_ai.storage.edge_store import SOURCE_BARE_TEXT
from pure_integer_ai.storage.span import (
    SPAN_MEMBER_TABLE,
    SPAN_TABLE,
    SpanStorageRecord,
    SpanStorageIntegrityError,
    SpanStore,
)
from pure_integer_ai.storage.telemetry import (
    BackendTelemetryCollector,
    collect_backend_telemetry,
)
from pure_integer_ai.training.cursor import load_run
from pure_integer_ai.training.stages import STAGE1_SKELETON


def _backend(kind: str):
    """为两类后端建立独立测试存储。"""
    if kind == "dict":
        return DictBackend()
    if kind == "sqlite":
        return SQLiteBackend(":memory:")
    raise ValueError(kind)


def _source(source_id: int = 1) -> SourceRef:
    """构造带非零 parser version 的稳定来源。"""
    return SourceRef(
        SOURCE_BARE_TEXT,
        401,
        source_id,
        GLOBAL_OWNER_SCOPE,
        VersionBundle(parser=ParserVersion(3)),
    )


def _protocol() -> SpanProtocol:
    """注入四种互不混型的 Span 图关系。"""
    return SpanProtocol(
        structure_relation_key=(99100, 1),
        constituent_relation_key=(99100, 2),
        occurrence_relation_key=(99100, 3),
        candidate_relation_key=(99100, 4),
    )


def _indexes(ctx):
    """组装 L-03 occurrence 与 L-04 Span 索引。"""
    occurrences = OccurrenceIndex(
        ctx.graph_ontology,
        ctx.scoped_identity_store,
        OccurrenceProtocol((99100, 5)),
    )
    spans = SpanIndex(
        ctx.graph_ontology,
        ctx.scoped_identity_store,
        _protocol(),
        occurrences,
    )
    return occurrences, spans


def _segmentation_result(ctx, source: SourceRef, text: str):
    """生成含长词与字符边界竞争的确定性 L-02 结果。"""
    branch = ctx.graph_ontology.materialize(
        language_branch_identity((99130, 1)))
    engine = SegmentationHypothesisEngine(SegmentationProtocol(
        hypothesis_kind_key=(99130, 2),
        lexical_match_reason_key=(99130, 3),
        oov_reason_key=(99130, 4),
        candidate_limit=3,
    ))
    lattice = (
        (text, text[0]),
        (text[1],),
    )
    return engine.parse(
        text,
        lattice=lattice,
        branch=branch,
        observation=source,
        scope=document_scope(source),
        visible_form=lambda surface: None,
    )


def _segmentation_span_protocol() -> SegmentationSpanProtocol:
    """注入分词层级和跨来源共享形状键。"""
    return SegmentationSpanProtocol(
        span_protocol=_protocol(),
        document_structure_key=(99140, 1),
        part_structure_key=(99140, 2),
        candidate_shape_namespace_key=(99140, 3),
        atomic_structure_key=(99140, 4),
    )


def _manual_segmentation_result(
        source: SourceRef, text: str = "甲乙") -> SegmentationResult:
    """构造无需课程目录的两种边界候选，供顶层接线和恢复测试。"""
    scope = document_scope(source)
    competition_key = (99150, len(text), *(ord(char) for char in text))
    fine_key = HypothesisKey(
        (99150, 1),
        (*competition_key, 1),
        competition_key,
        scope,
        source,
    )
    coarse_key = HypothesisKey(
        (99150, 1),
        (*competition_key, 2),
        competition_key,
        scope,
        source,
    )
    fine = SegmentationCandidate((
        SegmentationPart(0, 1, text[0], False),
        SegmentationPart(1, 2, text[1], False),
    ))
    coarse = SegmentationCandidate((
        SegmentationPart(0, 2, text, False),
    ))
    fine_snapshot = HypothesisSnapshot(
        fine_key,
        LIFECYCLE_ACTIVE,
        EPISTEMIC_UNKNOWN,
        (),
        (),
        (),
    )
    coarse_snapshot = HypothesisSnapshot(
        coarse_key,
        LIFECYCLE_ACTIVE,
        EPISTEMIC_UNKNOWN,
        (),
        (),
        (),
    )
    return SegmentationResult(
        text,
        (
            SegmentationHypothesisCandidate(fine, fine_key, fine_snapshot),
            SegmentationHypothesisCandidate(
                coarse,
                coarse_key,
                coarse_snapshot,
            ),
        ),
        fine_key,
    )


def _shared_part_segmentation_result(source: SourceRef) -> SegmentationResult:
    """构造两个共享首个 part、但根结构不同的三码点候选。"""
    text = "甲乙丙"
    scope = document_scope(source)
    competition_key = (99151, len(text), *(ord(char) for char in text))
    first_key = HypothesisKey(
        (99151, 1),
        (*competition_key, 1),
        competition_key,
        scope,
        source,
    )
    second_key = HypothesisKey(
        (99151, 1),
        (*competition_key, 2),
        competition_key,
        scope,
        source,
    )
    first = SegmentationCandidate((
        SegmentationPart(0, 1, text[0], False),
        SegmentationPart(1, 3, text[1:3], False),
    ))
    second = SegmentationCandidate((
        SegmentationPart(0, 1, text[0], False),
        SegmentationPart(1, 2, text[1], False),
        SegmentationPart(2, 3, text[2], False),
    ))
    snapshots = tuple(
        HypothesisSnapshot(
            key,
            LIFECYCLE_ACTIVE,
            EPISTEMIC_UNKNOWN,
            (),
            (),
            (),
        )
        for key in (first_key, second_key)
    )
    return SegmentationResult(
        text,
        (
            SegmentationHypothesisCandidate(first, first_key, snapshots[0]),
            SegmentationHypothesisCandidate(second, second_key, snapshots[1]),
        ),
        first_key,
    )


def test_span_identity_normalizes_members_and_keeps_empty_ordinal():
    """身份保留完整非连续范围与同位 ordinal，不再混入宿主 span kind。"""
    source = _source()
    assert normalize_span_members(((2, 3), (0, 1))) == ((0, 1), (2, 3))
    assert normalize_span_members(((0, 1), (1, 2))) == ((0, 2),)
    first = span_identity(source, members=((0, 1), (2, 3)))
    reordered = span_identity(source, members=((2, 3), (0, 1)))
    repeated = span_identity(
        source,
        members=((0, 1), (2, 3)),
        ordinal=1,
    )
    empty = span_identity(source, members=((1, 1),))
    assert first == reordered
    assert first != repeated
    assert empty != first
    with pytest.raises(ValueError, match="零宽 span"):
        span_identity(source, members=((1, 1), (2, 3)))


def test_span_verified_cache_is_zero_io_and_explicitly_invalidated():
    """同一写事务复用已核验 Span，清缓存后仍能发现外部重复行。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        _, spans = _indexes(ctx)
        source = _source(19)
        scope = document_scope(source)
        structure = structure_concept_identity((99119, 1))
        first = spans.ensure_ref(
            source=source,
            raw_text="甲乙",
            scope=scope,
            members=((0, 2),),
            structures=(structure,),
        )

        collector = BackendTelemetryCollector()
        with collect_backend_telemetry(collector):
            assert spans.ensure_ref(
                source=source,
                raw_text="甲乙",
                scope=scope,
                members=((0, 2),),
                structures=(structure,),
            ) == first
        assert collector.operation_snapshot() == {}

        row = backend.select(SPAN_TABLE, where={
            "space_id": first.space_id,
            "local_id": first.local_id,
        })[0]
        backend.insert(SPAN_TABLE, row)
        spans.clear_runtime_caches()
        with pytest.raises(SpanStorageIntegrityError, match="唯一"):
            spans.ensure_ref(
                source=source,
                raw_text="甲乙",
                scope=scope,
                members=((0, 2),),
                structures=(structure,),
            )
    finally:
        backend.close()


def test_span_fresh_add_does_not_read_back_successful_writes():
    """新 Span 只做存在性预检，成功插入后不重复读取刚写入的同一内容。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        store = SpanStore(backend)
        record = SpanStorageRecord(
            space_id=ctx.space_id,
            local_id=900001,
            source_hash=101,
            scope_hash=102,
            member_count=1,
            ordinal=0,
            parser_version=1,
            envelope_start=2,
            envelope_end=4,
        )
        collector = BackendTelemetryCollector()
        with collect_backend_telemetry(collector):
            assert store.add(record, ((2, 4),)) == record
        operations = collector.operation_snapshot()
        assert operations == {
            ("insert", SPAN_MEMBER_TABLE): (1, 1, 0),
            ("insert", SPAN_TABLE): (1, 1, 0),
            ("select", SPAN_TABLE): (1, 0, 0),
        }
        store.clear_runtime_caches()
        assert store.read(ctx.space_id, record.local_id) == (
            record,
            ((2, 4),),
        )
    finally:
        backend.close()


@pytest.mark.parametrize("kind", ["dict", "sqlite"])
def test_recursive_span_round_trip_preserves_gaps_repetition_and_occurrence(
        kind: str):
    """非连续、零宽、重复同范围和有序 constituent 可完整回读。"""
    backend = _backend(kind)
    try:
        ctx = make_train_context(backend)
        occurrences, spans = _indexes(ctx)
        source = _source()
        scope = document_scope(source)
        raw_text = "甲X乙"
        occurrence = occurrences.record(
            source=source,
            raw_text=raw_text,
            scope=scope,
            start=2,
            end=3,
            ordinal=0,
            segment_index=0,
            local_index=0,
            document_index=0,
        )
        structure = structure_concept_identity((99110, 1))
        parent = spans.ensure(
            source=source,
            raw_text=raw_text,
            scope=scope,
            members=((0, 1), (2, 3)),
            structures=(structure,),
        )
        first = spans.ensure(
            source=source,
            raw_text=raw_text,
            scope=scope,
            members=((0, 1),),
        )
        empty = spans.ensure(
            source=source,
            raw_text=raw_text,
            scope=scope,
            members=((1, 1),),
        )
        repeated = spans.ensure(
            source=source,
            raw_text=raw_text,
            scope=scope,
            members=((0, 1),),
            ordinal=1,
        )
        spans.add_constituent(parent.span, first.span, member_ordinal=0)
        spans.add_constituent(parent.span, empty.span, member_ordinal=1)
        spans.add_constituent(parent.span, repeated.span, member_ordinal=2)
        spans.add_occurrence(
            parent.span,
            occurrence.occurrence,
            member_ordinal=3,
        )

        restored = spans.read(parent.span)
        assert restored.members == ((0, 1), (2, 3))
        assert restored.structures[0].object_kind == structure.object_kind
        assert tuple(
            statement.assertion.qualifiers[0]
            for statement in restored.constituents
        ) == (0, 1, 2)
        assert restored.occurrences[0].object == occurrence.occurrence
        assert restored.occurrences[0].assertion.qualifiers == (3,)
        assert spans.span_count() == 4

        crossing_gap = spans.ensure(
            source=source,
            raw_text=raw_text,
            scope=scope,
            members=((0, 3),),
            ordinal=2,
        )
        with pytest.raises(ValueError, match="完整落在 parent"):
            spans.add_constituent(
                parent.span,
                crossing_gap.span,
                member_ordinal=4,
            )
    finally:
        backend.close()


def test_constituent_cycle_and_read_side_effects_fail_closed():
    """同范围分层可以嵌套，但反向成环拒绝，纯读取不得新建 predicate。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        _, spans = _indexes(ctx)
        source = _source()
        scope = document_scope(source)
        outer = spans.ensure(
            source=source,
            raw_text="甲",
            scope=scope,
            members=((0, 1),),
        )
        inner = spans.ensure(
            source=source,
            raw_text="甲",
            scope=scope,
            members=((0, 1),),
            ordinal=1,
        )
        graph_rows_before = backend.select("graph_object", where=None)
        spans.read(outer.span)
        assert backend.select("graph_object", where=None) == graph_rows_before
        spans.add_constituent(outer.span, inner.span, member_ordinal=0)
        with pytest.raises(ValueError, match="不得形成环"):
            spans.add_constituent(inner.span, outer.span, member_ordinal=0)
    finally:
        backend.close()


@pytest.mark.parametrize("kind", ["dict", "sqlite"])
def test_span_candidate_supersede_keeps_history_and_filters_active(kind: str):
    """错误边界只替代候选 assertion，旧 Span、statement 和 Evidence 身份仍保留。"""
    backend = _backend(kind)
    try:
        ctx = make_train_context(backend)
        _, spans = _indexes(ctx)
        source = _source()
        scope = document_scope(source)
        old_span = spans.ensure(
            source=source,
            raw_text="甲乙",
            scope=scope,
            members=((0, 1), (1, 2)),
        )
        new_span = spans.ensure(
            source=source,
            raw_text="甲乙",
            scope=scope,
            members=((0, 2),),
            ordinal=1,
        )
        competition_key = (99120, 1)
        old = HypothesisKey(
            (99120, 2),
            (99120, 3),
            competition_key,
            scope,
            source,
        )
        new = HypothesisKey(
            (99120, 2),
            (99120, 4),
            competition_key,
            scope,
            source,
        )
        old_statement = spans.add_candidate(old, old_span.span)
        new_statement = spans.add_candidate(new, new_span.span)
        clock = LogicalClock(LogicalClockIdentity(
            scope,
            CLOCK_OBSERVATION,
        ))

        event_hash = spans.supersede_candidate(
            old,
            new,
            clock.advance(),
        )

        assert event_hash > 0
        assert spans.candidate_statements(active_only=False) == (
            old_statement,
            new_statement,
        )
        assert spans.candidate_statements() == (new_statement,)
        assert spans.span_count() == 2
    finally:
        backend.close()


def test_segmentation_lattice_materializes_recursive_spans_and_shared_shape():
    """完整 lattice 保留独立根，共享 part/atomic，并隔离跨来源对象。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        occurrences, spans = _indexes(ctx)
        materializer = SegmentationSpanMaterializer(
            spans,
            _segmentation_span_protocol(),
        )
        source = _source(1)
        result = _segmentation_result(ctx, source, "甲乙")
        selected = result.selected
        assert selected is not None
        occurrence_refs = []
        for document_index, part in enumerate(selected.segmentation.parts):
            occurrence_refs.append(occurrences.record(
                source=source,
                raw_text=result.text,
                scope=document_scope(source),
                start=part.start,
                end=part.end,
                ordinal=0,
                segment_index=0,
                local_index=document_index,
                document_index=document_index,
            ).occurrence)

        first = materializer.materialize(
            result,
            occurrence_refs=tuple(occurrence_refs),
        )
        storage_after_first = backend.snapshot()
        assert materializer.materialize(
            result,
            occurrence_refs=tuple(occurrence_refs),
        ) == first
        assert backend.snapshot() == storage_after_first

        assert len(first.candidates) == len(result.candidates) == 2
        assert len(first.atomic_spans) == 2
        assert spans.span_count() == 5
        assert len({item.root for item in first.candidates}) == 2
        for item in first.candidates:
            root_record = spans.read(item.root)
            assert root_record.ordinal == spans.ensure_role_ordinal(
                item.hypothesis.object_identity())
            assert root_record.ordinal > 0
            assert tuple(
                statement.object for statement in root_record.constituents
            ) == item.parts
            assert tuple(
                statement.assertion.qualifiers
                for statement in root_record.constituents
            ) == tuple((index,) for index in range(len(item.parts)))
        selected_span = materializer.candidate(result.selected_hypothesis)
        assert selected_span is not None
        assert len(selected_span.parts) == len(occurrence_refs)
        for part_ref, occurrence_ref in zip(
                selected_span.parts, occurrence_refs):
            part_record = spans.read(part_ref)
            assert part_record.occurrences[0].object == occurrence_ref
        alternative = next(
            item for item in first.candidates
            if item.hypothesis != result.selected_hypothesis)
        assert all(not spans.read(part).occurrences for part in alternative.parts)
        two_part = next(item for item in first.candidates if len(item.parts) == 2)
        assert two_part.parts == first.atomic_spans
        assert all(not spans.read(part).constituents for part in two_part.parts)

        second_source = _source(2)
        second_result = _segmentation_result(ctx, second_source, "丙丁")
        second = materializer.materialize(second_result)
        first_two_part = next(
            item for item in first.candidates if len(item.parts) == 2)
        second_two_part = next(
            item for item in second.candidates if len(item.parts) == 2)
        first_structures = {
            ctx.graph_ontology.identity_of(ref)
            for ref in spans.read(first_two_part.root).structures
        }
        second_structures = {
            ctx.graph_ontology.identity_of(ref)
            for ref in spans.read(second_two_part.root).structures
        }
        assert first_structures == second_structures
        assert first_two_part.root != second_two_part.root
        assert set(occurrence_refs).isdisjoint(second.span_refs)

        old = alternative.hypothesis
        new = result.selected_hypothesis
        clock = LogicalClock(LogicalClockIdentity(
            document_scope(source),
            CLOCK_OBSERVATION,
        ))
        materializer.supersede_candidate(old, new, clock.advance())
        active = spans.candidate_statements()
        assert alternative.candidate_statement not in active
        assert selected_span.candidate_statement in active
        assert len(spans.candidate_statements(active_only=False)) == 4
    finally:
        backend.close()


def test_span_role_ordinal_is_global_recoverable_and_read_only():
    """共享 registry 区分同几何角色，重复登记稳定，缺失只读解析不写后端。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        _, spans = _indexes(ctx)
        first_role = structure_concept_identity((99141, 1))
        second_role = structure_concept_identity((99141, 2))
        before = backend.snapshot()

        assert spans.resolve_role_ordinal(first_role) is None
        assert backend.snapshot() == before
        first_ordinal = spans.ensure_role_ordinal(first_role)
        second_ordinal = spans.ensure_role_ordinal(second_role)

        assert first_ordinal > 0
        assert first_ordinal != second_ordinal
        assert spans.ensure_role_ordinal(first_role) == first_ordinal
        assert spans.resolve_role_ordinal(first_role) == first_ordinal
        source = _source(7)
        scope = document_scope(source)
        first = spans.ensure(
            source=source,
            raw_text="甲",
            scope=scope,
            members=((0, 1),),
            ordinal=first_ordinal,
            structures=(first_role,),
        )
        second = spans.ensure(
            source=source,
            raw_text="甲",
            scope=scope,
            members=((0, 1),),
            ordinal=second_ordinal,
            structures=(second_role,),
        )
        assert first.span != second.span
    finally:
        backend.close()


def test_candidates_share_identical_parts_without_losing_graph_paths():
    """候选共享同几何 part，但根、成员序和非自指 atomic 路径保持完整。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        _, spans = _indexes(ctx)
        materializer = SegmentationSpanMaterializer(
            spans,
            _segmentation_span_protocol(),
        )
        result = _shared_part_segmentation_result(_source(8))

        materialized = materializer.materialize(result)
        first = materializer.candidate(result.candidates[0].hypothesis)
        second = materializer.candidate(result.candidates[1].hypothesis)

        assert first is not None and second is not None
        assert first.root != second.root
        assert first.parts[0] == second.parts[0]
        assert first.parts[0] == materialized.atomic_spans[0]
        assert spans.span_count() == 6
        for item in (first, second):
            root_record = spans.read(item.root)
            assert tuple(
                statement.object for statement in root_record.constituents
            ) == item.parts
            assert tuple(
                statement.assertion.qualifiers
                for statement in root_record.constituents
            ) == tuple((index,) for index in range(len(item.parts)))
        shared_record = spans.read(first.parts[0])
        shared_structures = {
            ctx.graph_ontology.identity_of(ref)
            for ref in shared_record.structures
        }
        assert shared_structures == {
            structure_concept_identity((99140, 2)),
            structure_concept_identity((99140, 4)),
        }
        assert shared_record.constituents == ()
        assert tuple(
            statement.object for statement in spans.read(first.parts[1]).constituents
        ) == materialized.atomic_spans[1:]
    finally:
        backend.close()


def test_formal_train_materializes_full_lattice_and_restores_span_tables(
        tmp_path, monkeypatch):
    """正式入口必须接入完整候选树、报告计数，并从 dump 恢复 Span 本体。"""
    from pure_integer_ai.training import stages as training_stages

    monkeypatch.setattr(training_stages, "FLOOR_GRAPH_SIZE_S1", 0)
    source = _source(9)
    parse = _manual_segmentation_result(source)
    item = CollectedItem(
        tokens=list(parse.tokens),
        raw_text=parse.text,
        word_form_parse=parse,
        source=SOURCE_BARE_TEXT,
        source_ref=source,
    )
    config = FormalTrainConfig(
        run_dir=str(tmp_path),
        run_id="l04_formal",
        rounds_per_stage=1,
        active_training_stages=(STAGE1_SKELETON,),
        language_occurrence_protocol=OccurrenceProtocol((99160, 1)),
        language_span_protocol=_segmentation_span_protocol(),
    )

    training_backend = DictBackend()
    result = formal_train(
        config,
        [item],
        backend=training_backend,
        runner=DefaultRoundRunner(),
    )

    assert result.occurrence_count == 2
    assert result.span_count == 5
    assert result.span_candidate_fact_count == 2
    trained_snapshot = training_backend.snapshot()

    restored_backend = DictBackend()
    try:
        restored_ctx = make_train_context(restored_backend)
        load_run(restored_backend, str(tmp_path), "l04_formal")
        restored_occurrences = OccurrenceIndex(
            restored_ctx.graph_ontology,
            restored_ctx.scoped_identity_store,
            OccurrenceProtocol((99160, 1)),
        )
        restored_spans = SpanIndex(
            restored_ctx.graph_ontology,
            restored_ctx.scoped_identity_store,
            _protocol(),
            restored_occurrences,
        )
        assert restored_backend.snapshot() == trained_snapshot
        assert restored_spans.span_count() == 5
        assert len(restored_spans.candidate_statements()) == 2
    finally:
        restored_backend.close()


def test_evaluation_clone_materializes_spans_without_host_writes():
    """V-06 clone 的 Span、candidate 和缓存必须全部留在评测后端。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        ctx.occurrence_index, ctx.span_index = _indexes(ctx)
        ctx.segmentation_span_materializer = SegmentationSpanMaterializer(
            ctx.span_index,
            _segmentation_span_protocol(),
        )
        source = _source(11)
        parse = _manual_segmentation_result(source)
        item = CollectedItem(
            tokens=list(parse.tokens),
            raw_text=parse.text,
            word_form_parse=parse,
            source=SOURCE_BARE_TEXT,
            source_ref=source,
        )
        runner = DefaultRoundRunner()
        runner.run_round(ctx, item, STAGE1_SKELETON, 0)
        host_span_count = ctx.span_index.span_count()
        host_candidate_count = len(ctx.span_index.candidate_statements())

        with isolated_evaluation(ctx, label="l04") as eval_ctx:
            eval_source = _source(12)
            eval_parse = _manual_segmentation_result(eval_source)
            eval_item = CollectedItem(
                tokens=list(eval_parse.tokens),
                raw_text=eval_parse.text,
                word_form_parse=eval_parse,
                source=SOURCE_BARE_TEXT,
                source_ref=eval_source,
            )
            runner.run_round(
                eval_ctx,
                eval_item,
                STAGE1_SKELETON,
                0,
            )
            assert eval_ctx.span_index.span_count() > host_span_count

        assert ctx.span_index.span_count() == host_span_count
        assert len(ctx.span_index.candidate_statements()) == host_candidate_count
    finally:
        backend.close()


def test_provider_feedback_synchronizes_hypothesis_and_span_supersede():
    """正式边界反馈必须同时替代 H-00 生命周期和图内 Span candidate link。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        occurrences, spans = _indexes(ctx)
        materializer = SegmentationSpanMaterializer(
            spans,
            _segmentation_span_protocol(),
        )
        branch = ctx.graph_ontology.materialize(
            language_branch_identity((99170, 1)))
        provider = WordFormProvider(
            backend=backend,
            concept_index=ctx.concept_index,
            ontology=ctx.graph_ontology,
            branch=branch,
            runtime_language=1,
            unicode_family_key=(99170, 2),
            inventory_relation_key=(99170, 3),
            catalog_identity=(99170, 4),
            catalog={},
            segmentation_protocol=SegmentationProtocol(
                hypothesis_kind_key=(99170, 5),
                lexical_match_reason_key=(99170, 6),
                oov_reason_key=(99170, 7),
                candidate_limit=3,
            ),
        )
        provider.install_segmentation_span_materializer(materializer)
        source = _source(13)
        result = provider.parse_text(
            "甲乙",
            observation=source,
            scope=document_scope(source),
        )
        assert result is not None and len(result.candidates) >= 2
        materialized = materializer.materialize(result)
        old = next(
            item.hypothesis for item in materialized.candidates
            if item.hypothesis != result.selected_hypothesis)
        new = result.selected_hypothesis

        snapshot = provider.record_segmentation_feedback(
            old,
            stance=EVIDENCE_REFUTE,
            source=source,
            reason_key=(99170, 8),
            timestamp_seq=7,
            replacement=new,
        )

        assert snapshot.lifecycle != LIFECYCLE_ACTIVE
        old_statement = materializer.candidate(old).candidate_statement
        new_statement = materializer.candidate(new).candidate_statement
        assert old_statement not in spans.candidate_statements()
        assert new_statement in spans.candidate_statements()
    finally:
        backend.close()
