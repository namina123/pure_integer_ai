"""L-03 一等 occurrence、来源回读和评测隔离测试。"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from pure_integer_ai.cognition.shared.graph_ontology import (
    relation_concept_identity,
)
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    ParserVersion,
    SourceRef,
    VersionBundle,
    concept_identity,
    occurrence_identity,
    representation_identity,
    sense_identity,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    SCOPE_DOCUMENT,
    document_scope,
)
from pure_integer_ai.cognition.understanding.occurrence_index import (
    OccurrenceIndex,
    OccurrenceProtocol,
)
from pure_integer_ai.cognition.understanding.segmentation_candidates import (
    SegmentationCandidate,
    SegmentationPart,
)
from pure_integer_ai.experiments.collection import (
    COLLECT_PRECEDES,
    CollectedItem,
)
from pure_integer_ai.experiments.corpus_identity import assign_corpus_source_refs
from pure_integer_ai.experiments.evaluation_isolation import isolated_evaluation
from pure_integer_ai.experiments.formal_train import (
    DefaultRoundRunner,
    FormalTrainConfig,
    formal_train,
    make_train_context,
)
from pure_integer_ai.storage.backend import DictBackend, SQLiteBackend
from pure_integer_ai.storage.edge_store import SOURCE_BARE_TEXT
from pure_integer_ai.training.cursor import dump_run, load_run
from pure_integer_ai.training.stages import STAGE1_SKELETON


_PROTOCOL = OccurrenceProtocol(
    candidate_relation_key=(93001, 1),
    speaker_relation_key=(93001, 2),
)


def _backend(kind: str):
    """为两类后端建立独立存储。"""
    if kind == "dict":
        return DictBackend()
    if kind == "sqlite":
        return SQLiteBackend(":memory:")
    raise ValueError(kind)


def _source(source_id: int, document_id: int, *, parser: int = 7) -> SourceRef:
    """构造带显式 parser version 的稳定测试来源。"""
    return SourceRef(
        SOURCE_BARE_TEXT,
        source_id,
        document_id,
        GLOBAL_OWNER_SCOPE,
        VersionBundle(parser=ParserVersion(parser)),
    )


def _index(backend):
    """建立包含图本体、scope registry 和 occurrence facade 的上下文。"""
    ctx = make_train_context(backend)
    index = OccurrenceIndex(
        ctx.graph_ontology,
        ctx.scoped_identity_store,
        _PROTOCOL,
    )
    ctx.occurrence_index = index
    return ctx, index


class _FixedRepresentationProvider:
    """为组合测试返回预先物化的 Representation，不承担分词。"""

    def __init__(self, representations):
        self._representations = representations

    def observe_surface(self, surface: str, *, runtime_language: int,
                        space_id: int):
        """按 surface 返回已注入表示；语言和空间只保持正式调用签名。"""
        return self._representations.get(surface)


@pytest.mark.parametrize("kind", ["dict", "sqlite"])
def test_occurrences_keep_position_speaker_and_candidate_types_separate(kind: str):
    """重复词、跨句、零宽和多 speaker 不合并，typed/legacy 候选不混型。"""
    backend = _backend(kind)
    try:
        ctx, index = _index(backend)
        source_a = _source(11, 1)
        source_b = _source(11, 2)
        scope_a = document_scope(source_a)
        scope_b = document_scope(source_b)
        raw_text = "甲。甲"
        speaker_a = concept_identity((94001, 1))
        speaker_b = concept_identity((94001, 2))
        concept = ctx.graph_ontology.materialize(concept_identity((95001,)))
        sense = ctx.graph_ontology.materialize(
            sense_identity(source_a, sense_key=(95002,)))
        representation = ctx.graph_ontology.materialize(
            representation_identity((95003,), (0x7532,)))
        legacy = ctx.concept_index.ensure(
            "兼容候选", space_id=ctx.space_id)

        first = index.record(
            source=source_a,
            raw_text=raw_text,
            scope=scope_a,
            start=0,
            end=1,
            ordinal=0,
            segment_index=0,
            local_index=0,
            document_index=0,
            speaker=speaker_a,
            typed_candidates=(concept, sense, representation),
            legacy_candidates=(legacy,),
        )
        repeated = index.record(
            source=source_a,
            raw_text=raw_text,
            scope=scope_a,
            start=0,
            end=1,
            ordinal=0,
            segment_index=0,
            local_index=0,
            document_index=0,
            speaker=speaker_a,
            typed_candidates=(concept, sense, representation),
            legacy_candidates=(legacy,),
        )
        second = index.record(
            source=source_a,
            raw_text=raw_text,
            scope=scope_a,
            start=2,
            end=3,
            ordinal=0,
            segment_index=1,
            local_index=0,
            document_index=1,
            speaker=speaker_a,
            typed_candidates=(concept,),
        )
        zero_a = index.record(
            source=source_a,
            raw_text=raw_text,
            scope=scope_a,
            start=1,
            end=1,
            ordinal=0,
            segment_index=0,
            local_index=1,
            document_index=1,
        )
        zero_b = index.record(
            source=source_a,
            raw_text=raw_text,
            scope=scope_a,
            start=1,
            end=1,
            ordinal=1,
            segment_index=0,
            local_index=1,
            document_index=1,
        )
        other_source = index.record(
            source=source_b,
            raw_text=raw_text,
            scope=scope_b,
            start=0,
            end=1,
            ordinal=0,
            segment_index=0,
            local_index=0,
            document_index=0,
            speaker=speaker_b,
            typed_candidates=(concept,),
        )

        assert repeated == first
        assert len({
            first.occurrence,
            second.occurrence,
            zero_a.occurrence,
            zero_b.occurrence,
            other_source.occurrence,
        }) == 5
        assert first.raw_text == raw_text
        assert first.surface == "甲"
        assert first.parser_version == 7
        assert first.scope == scope_a
        assert first.scope.scope_kind == SCOPE_DOCUMENT
        assert first.segment_index == 0
        assert first.local_index == 0
        assert first.document_index == 0
        assert first.speaker != other_source.speaker
        assert zero_a.surface == zero_b.surface == ""
        assert [candidate.typed_ref for candidate in first.candidates[:3]] == [
            concept,
            sense,
            representation,
        ]
        assert first.candidates[3].legacy_ref == legacy
        assert index.occurrence_count() == 5
        assert index.source_count() == 2

        candidate_predicate = ctx.graph_ontology.resolve(
            relation_concept_identity(_PROTOCOL.candidate_relation_key))
        speaker_predicate = ctx.graph_ontology.resolve(
            relation_concept_identity(_PROTOCOL.speaker_relation_key))
        assert candidate_predicate is not None
        assert speaker_predicate is not None
        assert len(ctx.graph_ontology.statements(
            predicate=candidate_predicate,
            subject=first.occurrence,
        )) == 3
        assert len(ctx.graph_ontology.statements(
            predicate=speaker_predicate,
            subject=first.occurrence,
        )) == 1
    finally:
        backend.close()


@pytest.mark.parametrize("kind", ["dict", "sqlite"])
def test_occurrence_round_trip_survives_dump_and_resume(kind: str, tmp_path):
    """dump/load 后 occurrence 身份、原文、scope、speaker 和候选保持一致。"""
    backend = _backend(kind)
    restored_backend = _backend(kind)
    try:
        ctx, index = _index(backend)
        source = _source(21, 3, parser=9)
        scope = document_scope(source)
        representation = ctx.graph_ontology.materialize(
            representation_identity((96001,), (0x7532, 0x4E59)))
        legacy = ctx.concept_index.ensure("旧端点", space_id=ctx.space_id)
        record = index.record(
            source=source,
            raw_text="甲乙",
            scope=scope,
            start=0,
            end=2,
            ordinal=0,
            segment_index=0,
            local_index=0,
            document_index=0,
            speaker=concept_identity((96002,)),
            typed_candidates=(representation,),
            legacy_candidates=(legacy,),
        )
        dump_run(
            backend,
            str(tmp_path),
            f"l03_{kind}",
            spaces=[ctx.space_id],
        )

        restored_ctx, restored_index = _index(restored_backend)
        load_run(restored_backend, str(tmp_path), f"l03_{kind}")
        recovered = restored_index.read(record.occurrence)

        assert recovered == record
        assert restored_index.occurrence_count() == 1
        assert restored_index.source_count() == 1
        assert restored_ctx.graph_ontology.identity_of(
            recovered.occurrence) == occurrence_identity(
                source, start=0, end=2, ordinal=0)
    finally:
        backend.close()
        restored_backend.close()


def test_formal_observe_reuses_source_occurrences_across_rounds():
    """训练 round 的 episode scope 改变时，不得复制或冲突来源 occurrence。"""
    backend = DictBackend()
    try:
        ctx, index = _index(backend)
        item = CollectedItem(
            tokens=["甲", "甲"],
            raw_text="甲甲",
            role_seq=[1, 1],
            collect_type=COLLECT_PRECEDES,
            source=SOURCE_BARE_TEXT,
        )
        assign_corpus_source_refs([item])
        runner = DefaultRoundRunner()

        runner.run_round(ctx, item, 1, 0)
        runner.run_round(ctx, item, 1, 1)

        assert item.source_ref is not None
        occurrence = ctx.graph_ontology.resolve(occurrence_identity(
            item.source_ref, start=0, end=1, ordinal=0))
        assert occurrence is not None
        assert index.read(occurrence).scope == document_scope(item.source_ref)
        assert index.occurrence_count() == 2
        assert index.source_count() == 1
    finally:
        backend.close()


def test_l02_winner_spans_and_observed_representation_reach_occurrence():
    """L-02 winner 的精确 span 和真实 observe 后表示候选必须同时进入 L-03。"""
    backend = DictBackend()
    try:
        ctx, index = _index(backend)
        source = _source(31, 1, parser=12)
        representation = ctx.graph_ontology.materialize(
            representation_identity((97001,), (0x7532, 0x4E59)))
        ctx.word_form_providers = _FixedRepresentationProvider({
            "甲乙": representation,
        })
        item = CollectedItem(
            tokens=["甲乙", "甲"],
            raw_text="甲乙甲",
            word_form_parse=SimpleNamespace(
                selected=SimpleNamespace(
                    segmentation=SegmentationCandidate((
                        SegmentationPart(0, 2, "甲乙", True),
                        SegmentationPart(2, 3, "甲", False),
                    )),
                ),
            ),
            role_seq=[1, 1],
            collect_type=COLLECT_PRECEDES,
            source=SOURCE_BARE_TEXT,
            source_ref=source,
        )

        DefaultRoundRunner().run_round(ctx, item, 1, 0)

        first_ref = ctx.graph_ontology.resolve(occurrence_identity(
            source, start=0, end=2, ordinal=0))
        second_ref = ctx.graph_ontology.resolve(occurrence_identity(
            source, start=2, end=3, ordinal=0))
        assert first_ref is not None
        assert second_ref is not None
        first = index.read(first_ref)
        second = index.read(second_ref)
        assert first.surface == "甲乙"
        assert first.parser_version == 12
        assert any(
            candidate.typed_ref == representation
            for candidate in first.candidates
        )
        assert second.surface == "甲"
        assert index.occurrence_count() == 2
    finally:
        backend.close()


def test_formal_train_assembles_occurrence_index_and_reports_counts(
        tmp_path, monkeypatch):
    """顶层训练入口必须装配 L-03，并在多轮结束后报告唯一来源计数。"""
    from pure_integer_ai.training import stages as training_stages

    monkeypatch.setattr(training_stages, "FLOOR_GRAPH_SIZE_S1", 0)
    backend = DictBackend()
    try:
        item = CollectedItem(
            tokens=["甲", "甲"],
            raw_text="甲甲",
            role_seq=[1, 1],
            collect_type=COLLECT_PRECEDES,
            source=SOURCE_BARE_TEXT,
        )
        config = FormalTrainConfig(
            run_dir=str(tmp_path),
            run_id="l03_formal",
            rounds_per_stage=2,
            active_training_stages=(STAGE1_SKELETON,),
            language_occurrence_protocol=_PROTOCOL,
        )

        result = formal_train(
            config,
            [item],
            backend=backend,
            runner=DefaultRoundRunner(),
        )

        assert result.occurrence_count == 2
        assert result.source_record_count == 1
        assert result.stages_completed == [STAGE1_SKELETON]
    finally:
        backend.close()


def test_evaluation_occurrences_are_written_only_to_the_clone():
    """评测可在 clone 中构造可回源 occurrence，但不得回写正式 backend。"""
    backend = DictBackend()
    try:
        ctx, index = _index(backend)
        item = CollectedItem(
            tokens=["甲", "乙"],
            raw_text="甲乙",
            role_seq=[1, 1],
            collect_type=COLLECT_PRECEDES,
            source=SOURCE_BARE_TEXT,
        )
        assign_corpus_source_refs([item])

        with isolated_evaluation(ctx, label="l03") as eval_ctx:
            DefaultRoundRunner().run_round(eval_ctx, item, 1, 0)
            assert eval_ctx.occurrence_index is not None
            assert eval_ctx.occurrence_index.occurrence_count() == 2
            assert eval_ctx.occurrence_index.source_count() == 1

        assert index.occurrence_count() == 0
        assert index.source_count() == 0
    finally:
        backend.close()
