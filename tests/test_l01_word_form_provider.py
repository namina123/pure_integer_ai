from __future__ import annotations

from pathlib import Path

import pytest

import pure_integer_ai.experiments.formal_train as formal_train_module
import pure_integer_ai.experiments.language_course_intake as language_intake_module

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_LANGUAGE_ATOM,
    OBJECT_REPRESENTATION,
    OBJECT_SENSE,
)
from pure_integer_ai.cognition.shared.language_object_index import (
    LanguageObjectIndex,
)
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.cognition.shared.types import LANG_ZH
from pure_integer_ai.cognition.shared.relation_primitives import (
    REL_CAUSES,
    ensure_relation_primitives,
)
from pure_integer_ai.cognition.understanding.cue_words import (
    CAUSES_CUE_FORWARD,
    cue_type_of,
)
from pure_integer_ai.cognition.understanding.emergent_relation_signal import (
    record_emergent_relation_signal_shadow,
)
from pure_integer_ai.cognition.understanding.segmentation_hypothesis import (
    SegmentationProtocol,
)
from pure_integer_ai.cognition.understanding.word_form_index import (
    WordFormIndex,
)
from pure_integer_ai.experiments.chinese_semantic_kb_adapter import (
    PARSER_DECIMAL_TAB,
    PARSER_DOCUMENT,
    PARSER_RELATION_MARKER,
    PARSER_SURFACE_LINE,
    PARSER_SYMMETRIC_AT,
    PROFILES,
    build_manifest,
)
from pure_integer_ai.experiments.chinese_semantic_kb_curriculum import (
    SPLIT_HELD_OUT,
    SPLIT_TRAIN,
    ChineseSemanticKBCurriculum,
    CourseSplitPolicy,
    build_curriculum_artifacts,
)
from pure_integer_ai.experiments.collection import CollectedItem
from pure_integer_ai.experiments.data_manifest import write_manifest
from pure_integer_ai.experiments.evaluation_isolation import isolated_evaluation
from pure_integer_ai.experiments.formal_train import (
    DefaultRoundRunner,
    FormalTrainConfig,
    _apply_word_form_providers,
    formal_train,
    make_train_context,
)
from pure_integer_ai.experiments.language_course_intake import (
    build_word_form_providers,
)
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_types import EDGE_RELATION_SIGNAL
from pure_integer_ai.storage.graph_object import GRAPH_OBJECT_TABLE
from pure_integer_ai.storage.node_store import NODE_CONCEPT, TIER_PRIMARY
from pure_integer_ai.storage.word_form_index import (
    WORD_FORM_LEGACY_BRIDGE_TABLE,
    load_legacy_word_form_bridges,
)
from pure_integer_ai.training.stages import STAGE1_SKELETON


def _profile_line(profile) -> str:
    """为每种 D-00 parser 生成一条最小合法记录。"""
    if profile.parser_kind == PARSER_DOCUMENT:
        return "# 测试来源"
    if profile.parser_kind == PARSER_RELATION_MARKER:
        return f"甲,{profile.relation_marker},乙"
    if profile.parser_kind == PARSER_SYMMETRIC_AT:
        return "上@下"
    if profile.parser_kind == PARSER_DECIMAL_TAB:
        return "很\t2.5"
    if profile.parser_kind == PARSER_SURFACE_LINE:
        return "基础项"
    raise AssertionError(profile.parser_kind)


def _build_course(tmp_path: Path):
    """生成同时含 train 和 held-out 词形的最小版本化课程。"""
    raw_root = tmp_path / "raw"
    for profile in PROFILES:
        path = raw_root / profile.relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_profile_line(profile) + "\n", encoding="utf-8")
    generated = "".join(
        f"课程专用词{index:04d}\n" for index in range(256))
    (raw_root / "dict/停用词.txt").write_text(
        generated, encoding="utf-8")

    source_manifest, _ = build_manifest(
        raw_root,
        dataset_version="l01-fixture-v1",
        unicode_sequence_family=(81001,),
    )
    source_path = tmp_path / "manifest" / "source.json"
    write_manifest(source_manifest, source_path, raw_root=raw_root)
    course_root = tmp_path / "course"
    course_manifest = build_curriculum_artifacts(
        source_manifest,
        raw_root,
        course_root,
        split_policy=CourseSplitPolicy(1, 700, 100, 200),
    )
    curriculum = ChineseSemanticKBCurriculum(course_manifest, course_root)
    generated_forms = tuple(
        item for item in curriculum.iter_word_forms(category="stoplist")
        if item.surface.startswith("课程专用词"))
    train_form = next(
        item for item in generated_forms if item.split == SPLIT_TRAIN)
    held_out_form = next(
        item for item in generated_forms if item.split == SPLIT_HELD_OUT)
    return source_path, course_root, train_form, held_out_form


def _providers(ctx, source_path: Path, course_root: Path, *, splits):
    """在给定上下文装配测试课程 provider。"""
    return build_word_form_providers(
        backend=ctx.backend,
        concept_index=ctx.concept_index,
        ontology=ctx.graph_ontology,
        course_root=course_root,
        source_manifest_path=source_path,
        runtime_language=LANG_ZH,
        visible_splits=splits,
    )


def _config(tmp_path: Path, run_id: str, source_path: Path,
            course_root: Path, **kwargs) -> FormalTrainConfig:
    """构造只跑 L-01 所需 observe 切片的正式训练配置。"""
    return FormalTrainConfig(
        run_dir=str(tmp_path / "runs"),
        run_id=run_id,
        rounds_per_stage=1,
        active_training_stages=(STAGE1_SKELETON,),
        language_course_root=str(course_root),
        language_source_manifest_path=str(source_path),
        language_course_runtime_language=LANG_ZH,
        language_course_visible_splits=(SPLIT_TRAIN,),
        **kwargs,
    )


def test_course_visibility_is_read_only_and_observe_materializes_lazily(
        tmp_path, monkeypatch):
    """课程只提供 FMM；真实观察后才产生 Representation 和 legacy 桥。"""
    source_path, course_root, train_form, held_out_form = _build_course(
        tmp_path)
    backend = DictBackend()
    ctx = make_train_context(backend)
    registry, report = _providers(
        ctx, source_path, course_root, splits=(SPLIT_TRAIN,))
    provider = registry.provider(LANG_ZH)
    assert provider is not None

    visible = provider.visible_form(train_form.surface)
    assert visible is not None
    assert visible.course_split == SPLIT_TRAIN
    assert visible.source_ref == train_form.source_ref
    assert provider.visible_form(held_out_form.surface) is None
    assert provider.segment_text(train_form.surface) == [train_form.surface]
    assert provider.segment_text(held_out_form.surface) != [held_out_form.surface]
    assert provider.index.lookup(
        train_form.surface, branch=provider.branch) is None
    assert provider.materialized_count == 0
    assert report.selected_splits == (SPLIT_TRAIN,)

    legacy = ctx.concept_index.ensure(
        train_form.surface,
        space_id=ctx.space_id,
        node_type=NODE_CONCEPT,
    )
    representation = registry.observe_surface(
        train_form.surface,
        runtime_language=LANG_ZH,
        space_id=ctx.space_id,
    )
    assert representation is not None
    assert representation.object_kind == OBJECT_REPRESENTATION
    assert provider.index.lookup(
        train_form.surface, branch=provider.branch) == representation
    bridges = load_legacy_word_form_bridges(backend, legacy_ref=legacy)
    assert bridges == ((
        NODE_CONCEPT,
        OBJECT_REPRESENTATION,
        representation.space_id,
        representation.local_id,
    ),)
    relation = ensure_relation_primitives(
        ctx.concept_index, backend, space_id=ctx.space_id)[REL_CAUSES]
    record_emergent_relation_signal_shadow(
        ctx.edge_store, legacy, relation, space_id=ctx.space_id)
    ctx.edge_store.set_tier(
        space_id_from=legacy[0],
        local_id_from=legacy[1],
        space_id_to=relation[0],
        local_id_to=relation[1],
        edge_type=EDGE_RELATION_SIGNAL,
        new_tier=TIER_PRIMARY,
    )
    monkeypatch.setattr(
        formal_train_module.gates,
        "EMERGENT_RELATION_CUE_READBACK_MODE",
        True,
    )
    assert cue_type_of(
        train_form.surface,
        LANG_ZH,
        backend=backend,
        edge_store=ctx.edge_store,
        space_id=ctx.space_id,
        concept_index=ctx.concept_index,
    ) == CAUSES_CUE_FORWARD
    statements = ctx.graph_ontology.statements(
        subject=provider.branch,
        object_ref=representation,
    )
    assert len(statements) == 1
    assert statements[0].assertion.scope == document_scope(
        train_form.source_ref)

    object_kinds = {
        row["object_kind"]
        for row in backend.select(GRAPH_OBJECT_TABLE, where=None)
    }
    assert OBJECT_LANGUAGE_ATOM not in object_kinds
    assert OBJECT_SENSE not in object_kinds
    objects = LanguageObjectIndex(ctx.graph_ontology)
    assert objects.lookup_concept(tuple(ord(ch) for ch in train_form.surface)) is None


def test_formal_train_segments_before_probe_and_disabled_provider_degrades(
        tmp_path, monkeypatch):
    """正式入口统一重分词后再切 probe，关闭 provider 时保留旧整段 token。"""
    source_path, course_root, train_form, _held_out_form = _build_course(
        tmp_path)
    train_text = train_form.surface + "龘"
    probe_text = train_form.surface + "靐"
    corpus = [
        CollectedItem(tokens=[train_text], raw_text=train_text),
        CollectedItem(tokens=[probe_text], raw_text=probe_text),
    ]
    monkeypatch.setattr(
        formal_train_module, "stage_metric_gate", lambda _stage, _snap: True)
    built_registries = []
    original_builder = language_intake_module.build_word_form_providers

    def _capture_builder(**kwargs):
        """保留正式入口实际装配的 provider，核验 held-out ledger 写隔离。"""
        built = original_builder(**kwargs)
        built_registries.append(built[0])
        return built

    monkeypatch.setattr(
        language_intake_module,
        "build_word_form_providers",
        _capture_builder,
    )
    result = formal_train(
        _config(
            tmp_path,
            "l01-probe",
            source_path,
            course_root,
            probe_holdout=1,
            persist_graph_dump=False,
            language_segmentation_protocol=SegmentationProtocol(
                (83001,), (83002,), (83003,), 8),
        ),
        corpus,
        backend=DictBackend(),
        runner=DefaultRoundRunner(),
    )

    assert corpus[0].tokens[0] == train_form.surface
    assert corpus[1].tokens[0] == train_form.surface
    assert corpus[0].word_form_parse is not None
    assert corpus[1].word_form_parse is not None
    assert len(corpus[0].word_form_parse.candidates) >= 2
    assert result.probe_set is not None
    assert result.execution.training_items == 1
    assert result.execution.probe_items == 1
    assert result.word_form_course_report is not None
    assert result.word_form_course_report.selected_splits == (SPLIT_TRAIN,)
    assert result.word_form_course_report.retokenized_items == 2
    provider = built_registries[0].provider(LANG_ZH)
    assert provider is not None
    for candidate in corpus[0].word_form_parse.candidates:
        assert provider.segmentation_snapshot(candidate.hypothesis)
    for candidate in corpus[1].word_form_parse.candidates:
        with pytest.raises(KeyError, match="尚未登记"):
            provider.segmentation_snapshot(candidate.hypothesis)

    single_corpus = [CollectedItem(tokens=[train_text], raw_text=train_text)]
    single = formal_train(
        _config(
            tmp_path,
            "l02-no-effective-probe",
            source_path,
            course_root,
            probe_holdout=1,
            persist_graph_dump=False,
            language_segmentation_protocol=SegmentationProtocol(
                (83001,), (83002,), (83003,), 8),
        ),
        single_corpus,
        backend=DictBackend(),
        runner=DefaultRoundRunner(),
    )
    single_provider = built_registries[1].provider(LANG_ZH)
    assert single_provider is not None
    assert single.execution.training_items == 1
    assert single.execution.probe_items == 0
    for candidate in single_corpus[0].word_form_parse.candidates:
        assert single_provider.segmentation_snapshot(candidate.hypothesis)

    disabled = CollectedItem(tokens=[probe_text], raw_text=probe_text)
    assert _apply_word_form_providers([disabled], None) == 0
    assert disabled.tokens == [probe_text]


def test_formal_train_rejects_partial_course_and_index_bound_retokenization(
        tmp_path):
    """课程配置和带索引标注的重分词都必须 fail closed。"""
    partial = FormalTrainConfig(
        run_dir=str(tmp_path / "runs"),
        run_id="partial",
        language_course_root=str(tmp_path / "missing"),
    )
    with pytest.raises(ValueError, match="必须同时配置"):
        formal_train(partial, [], backend=DictBackend())

    source_path, course_root, train_form, _held_out_form = _build_course(
        tmp_path / "indexed")
    ctx = make_train_context(DictBackend())
    registry, _report = _providers(
        ctx, source_path, course_root, splits=(SPLIT_TRAIN,))
    indexed = CollectedItem(
        tokens=[train_form.surface],
        raw_text=train_form.surface,
        role_seq=[1],
    )
    with pytest.raises(ValueError, match="token 索引标注"):
        _apply_word_form_providers([indexed], registry)


def test_same_manifest_resume_keeps_segmentation_and_object_identity(
        tmp_path, monkeypatch):
    """dump/load 后从同一 manifest 重建 FMM，分词和 Representation 身份不漂移。"""
    source_path, course_root, train_form, _held_out_form = _build_course(
        tmp_path)
    text = train_form.surface + "龘"
    monkeypatch.setattr(
        formal_train_module, "stage_metric_gate", lambda _stage, _snap: True)

    first_corpus = [CollectedItem(tokens=[text], raw_text=text)]
    first_backend = DictBackend()
    first = formal_train(
        _config(tmp_path, "base", source_path, course_root),
        first_corpus,
        backend=first_backend,
        runner=DefaultRoundRunner(),
    )
    first_bridges = first_backend.select(
        WORD_FORM_LEGACY_BRIDGE_TABLE, where=None)
    assert first_corpus[0].tokens[0] == train_form.surface
    assert len(first_bridges) == 1
    assert first.dump_spaces

    resumed_corpus = [CollectedItem(tokens=[text], raw_text=text)]
    resumed_backend = DictBackend()
    resumed = formal_train(
        _config(
            tmp_path,
            "resumed",
            source_path,
            course_root,
            resume=True,
            base_run_id="base",
            persist_graph_dump=False,
        ),
        resumed_corpus,
        backend=resumed_backend,
        runner=DefaultRoundRunner(),
    )
    assert resumed_corpus[0].tokens == first_corpus[0].tokens
    assert resumed_backend.select(
        WORD_FORM_LEGACY_BRIDGE_TABLE, where=None) == first_bridges
    assert STAGE1_SKELETON in resumed.stages_skipped
    assert resumed.word_form_course_report == first.word_form_course_report


def test_evaluation_clone_rebuilds_provider_without_host_writes(tmp_path):
    """V-06 克隆可消费课程词形，但物化和 legacy 桥只留在评测后端。"""
    source_path, course_root, train_form, _held_out_form = _build_course(
        tmp_path)
    host = make_train_context(DictBackend())
    registry, report = _providers(
        host, source_path, course_root, splits=(SPLIT_TRAIN,))
    host.word_form_providers = registry
    host.word_form_course_report = report
    host_rows_before = tuple(
        host.backend.select(GRAPH_OBJECT_TABLE, where=None))

    with isolated_evaluation(host, label="l01-provider") as cloned:
        provider = cloned.word_form_providers.provider(LANG_ZH)
        assert provider is not None
        assert provider.segment_text(train_form.surface) == [train_form.surface]
        legacy = cloned.concept_index.ensure(
            train_form.surface,
            space_id=cloned.space_id,
            node_type=NODE_CONCEPT,
        )
        representation = cloned.word_form_providers.observe_surface(
            train_form.surface,
            runtime_language=LANG_ZH,
            space_id=cloned.space_id,
        )
        assert representation is not None
        assert load_legacy_word_form_bridges(
            cloned.backend, legacy_ref=legacy)

    assert tuple(host.backend.select(
        GRAPH_OBJECT_TABLE, where=None)) == host_rows_before
    assert host.backend.select(WORD_FORM_LEGACY_BRIDGE_TABLE, where=None) == []


def test_word_form_segment_cache_avoids_reloading_catalog(monkeypatch):
    """FMM 编译缓存命中后不再复制或扫描权威词形目录。"""
    index = WordFormIndex(DictBackend())
    calls = 0

    def _forms(**_kwargs):
        nonlocal calls
        calls += 1
        return {(30002, 30002): (1, 1)}

    monkeypatch.setattr(index, "forms", _forms)
    assert index.segment("甲甲", language=1, space_id=1) == ["甲甲"]
    assert index.segment("甲甲", language=1, space_id=1) == ["甲甲"]
    assert calls == 1
