"""FT17 indexed dispatch over every learned FT11-FT15 question route."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_authored_primitive_atomic_bridge_course import (
    compile_authored_primitive_atomic_bridge_course,
)
from pure_integer_ai.experiments.ph2_authored_semantic_primitive_bridge_course import (
    compile_authored_semantic_primitive_bridge_course,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_question_feature_catalog import (
    raw_question_feature_catalog,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_question_feature_composition import (
    build_three_role_question_feature_composition,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_question_feature_index import (
    QUESTION_FEATURE_INDEX_EXPRESSION_BOUNDARY,
    QUESTION_FEATURE_INDEX_SHA256,
    build_raw_question_feature_index,
    lookup_indexed_question_feature_entries,
    run_indexed_raw_question_feature_registry_answer,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_question_feature_registry import (
    RawQuestionFeatureRegistryEntry,
    build_raw_question_feature_registry,
    run_raw_question_feature_registry_answer,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question_alias import (
    build_learned_predicate_alias_bridge,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question_contract import (
    RawQuestionRequest,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question_generalization import (
    build_raw_question_generalization,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question_implicit import (
    build_implicit_question_bundle,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question_three_role import (
    build_three_role_question_bundle,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_vertical_generalization import (
    build_w03_w04_w05_vertical_generalization_overlay,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_vertical_three_role import (
    build_w03_w04_w05_three_role_vertical_overlay,
)


REPOSITORY = Path(__file__).resolve().parents[1]
DATA = REPOSITORY / "data/ph2"


def _compile_overlay(root, semantic, primitive_map, atomic, builder):
    base = compile_authored_semantic_primitive_bridge_course(
        DATA / semantic,
        root / "base",
    )
    donor = compile_authored_primitive_atomic_bridge_course(
        DATA / primitive_map,
        DATA / atomic,
        root / "donor",
    )
    return builder(base, donor)


@pytest.fixture(scope="module")
def indexed_registry(tmp_path_factory):
    two_root = tmp_path_factory.mktemp("ft17_two_role")
    two_overlay = _compile_overlay(
        two_root,
        "authored_semantic_primitive_bridge_generalization_v1.jsonl.sample",
        "authored_primitive_atomic_bridge_map_generalization_v1.jsonl.sample",
        "authored_primitive_atomic_bridge_seed_generalization_v1.jsonl.sample",
        build_w03_w04_w05_vertical_generalization_overlay,
    )
    two_explicit = build_raw_question_generalization(
        two_overlay,
        DATA / "authored_vertical_question_cause_generalization_v1.jsonl.sample",
        DATA / "authored_vertical_question_effect_generalization_v1.jsonl.sample",
    )
    two_catalog = raw_question_feature_catalog(two_explicit)
    two_entry = RawQuestionFeatureRegistryEntry(
        two_catalog,
        build_learned_predicate_alias_bridge(two_catalog),
        build_implicit_question_bundle(
            two_catalog,
            DATA / "authored_vertical_question_implicit_reason_v1.jsonl.sample",
            DATA / "authored_vertical_question_implicit_result_v1.jsonl.sample",
        ),
    )

    three_root = tmp_path_factory.mktemp("ft17_three_role")
    three_overlay = _compile_overlay(
        three_root,
        "authored_semantic_primitive_bridge_three_role_v1.jsonl.sample",
        "authored_primitive_atomic_bridge_map_three_role_v1.jsonl.sample",
        "authored_primitive_atomic_bridge_seed_three_role_v1.jsonl.sample",
        build_w03_w04_w05_three_role_vertical_overlay,
    )
    three_explicit = build_three_role_question_bundle(
        three_overlay,
        DATA / "authored_vertical_question_three_role_actor_v1.jsonl.sample",
        DATA / "authored_vertical_question_three_role_location_v1.jsonl.sample",
    )
    three_composition = build_three_role_question_feature_composition(
        three_explicit,
        DATA / (
            "authored_vertical_question_three_role_implicit_actor_v1.jsonl.sample"),
        DATA / (
            "authored_vertical_question_three_role_implicit_location_v1.jsonl.sample"),
    )
    three_entry = RawQuestionFeatureRegistryEntry(
        three_composition.feature_catalog,
        three_composition.alias_bridge,
        three_composition.implicit_bundle,
    )
    registry = build_raw_question_feature_registry((two_entry, three_entry))
    return (
        build_raw_question_feature_index(registry),
        two_entry,
        three_entry,
    )


def _learned_alias_surface(entry, construction, alias_surface):
    predicate_ordinals = tuple(
        ordinal for ordinal, segment in enumerate(construction.segments)
        if segment.kind == "PREDICATE"
    )
    assert len(predicate_ordinals) == 1
    return "".join(
        alias_surface if ordinal == predicate_ordinals[0]
        else segment.surface
        for ordinal, segment in enumerate(construction.segments)
    )


def _aligned_alias_surface(entry, construction):
    proposition_key = construction.vertical_result.link.proposition_key
    alias = next(
        route.alias_surface for route in entry.alias_bridge.routes
        if route.proposition_key == proposition_key
    )
    return _learned_alias_surface(entry, construction, alias)


def _assert_scan_equal(index, request):
    scan = run_raw_question_feature_registry_answer(index.registry, request)
    indexed = run_indexed_raw_question_feature_registry_answer(index, request)
    assert indexed.to_dict() == scan.to_dict()
    assert indexed.sha256() == scan.sha256()
    return indexed


def test_index_is_derived_and_frozen(indexed_registry) -> None:
    index, _, _ = indexed_registry
    assert QUESTION_FEATURE_INDEX_EXPRESSION_BOUNDARY == (
        ("index_source", "LEARNED_CONSTRUCTIONS_AND_ALIAS_ROUTES"),
        ("candidate_effect", "NARROW_ONLY"),
        ("global_priority", "EXACT_THEN_ALIAS_THEN_IMPLICIT"),
        ("result_projection", "BYTE_IDENTICAL_TO_FT16_SCAN_RUNTIME"),
        ("source_binding", "OPTIONAL_SOURCE_REF_RETAINS_CATALOG_OWNERSHIP"),
        ("handwritten_dispatch", "FORBIDDEN"),
    )
    assert index.identity_sha256 == QUESTION_FEATURE_INDEX_SHA256
    assert index.sha256() == QUESTION_FEATURE_INDEX_SHA256
    assert index.exact_rows and index.alias_rows and index.implicit_rows
    assert all(row.postings for rows in (
        index.exact_rows, index.alias_rows, index.implicit_rows)
               for row in rows)


def test_every_learned_question_is_present_in_its_derived_index(
        indexed_registry) -> None:
    index, two_entry, three_entry = indexed_registry
    for entry in (two_entry, three_entry):
        entry_sha = entry.sha256()
        for construction in entry.feature_catalog.catalog:
            exact = RawQuestionRequest(
                construction.question_surface,
                construction.source_record_key,
            )
            alias = RawQuestionRequest(
                _aligned_alias_surface(entry, construction),
                construction.source_record_key,
            )
            assert entry_sha in lookup_indexed_question_feature_entries(
                index, "EXACT", exact)
            assert entry_sha in lookup_indexed_question_feature_entries(
                index, "ALIAS", alias)
        for construction in entry.implicit_bundle.catalog:
            request = RawQuestionRequest(
                construction.question_surface,
                construction.source_record_key,
            )
            assert entry_sha in lookup_indexed_question_feature_entries(
                index, "IMPLICIT", request)


def test_both_catalogs_and_all_phases_match_ft16_byte_for_byte(
        indexed_registry) -> None:
    index, two_entry, three_entry = indexed_registry
    for entry in (two_entry, three_entry):
        exact = entry.feature_catalog.catalog[0]
        implicit = entry.implicit_bundle.catalog[0]
        requests = (
            RawQuestionRequest(exact.question_surface),
            RawQuestionRequest(_aligned_alias_surface(entry, exact)),
            RawQuestionRequest(implicit.question_surface),
        )
        for request in requests:
            _assert_scan_equal(index, request)


def test_wrong_sources_missing_structures_and_repeated_calls_are_identical(
        indexed_registry) -> None:
    index, two_entry, three_entry = indexed_registry
    two = two_entry.feature_catalog.catalog[0]
    three_sources = {
        item.source_record_key for item in three_entry.feature_catalog.catalog}
    wrong_source = next(
        item for item in three_sources if item != two.source_record_key)
    wrong = RawQuestionRequest(two.question_surface, wrong_source)
    missing = RawQuestionRequest("未学习的索引结构？")
    for request in (wrong, missing):
        result = _assert_scan_equal(index, request)
        assert result.status == "UNKNOWN"
        assert all(trace.implicit_result is not None for trace in result.traces)
    first = run_indexed_raw_question_feature_registry_answer(index, missing)
    second = run_indexed_raw_question_feature_registry_answer(index, missing)
    assert first.sha256() == second.sha256()


@pytest.mark.parametrize("phase", ("EXACT", "ALIAS", "IMPLICIT"))
def test_lookup_retains_optional_source_binding(
        indexed_registry, phase) -> None:
    index, two_entry, _ = indexed_registry
    if phase == "EXACT":
        construction = two_entry.feature_catalog.catalog[0]
        surface = construction.question_surface
    elif phase == "ALIAS":
        construction = two_entry.feature_catalog.catalog[0]
        surface = _aligned_alias_surface(two_entry, construction)
    else:
        construction = two_entry.implicit_bundle.catalog[0]
        surface = construction.question_surface
    unscoped = lookup_indexed_question_feature_entries(
        index, phase, RawQuestionRequest(surface))
    scoped = lookup_indexed_question_feature_entries(
        index,
        phase,
        RawQuestionRequest(surface, construction.source_record_key),
    )
    assert scoped
    assert set(scoped).issubset(unscoped)


def test_only_indexed_catalog_candidates_execute(
        indexed_registry, monkeypatch) -> None:
    from pure_integer_ai.experiments import (
        ph2_w03_w04_w05_question_feature_index as runtime,
    )

    index, two_entry, _ = indexed_registry
    construction = two_entry.feature_catalog.catalog[0]
    request = RawQuestionRequest(
        construction.question_surface,
        construction.source_record_key,
    )
    expected = set(lookup_indexed_question_feature_entries(
        index, "EXACT", request))
    calls = []
    exact = runtime.run_raw_question_feature_answer

    def exact_probe(catalog, candidate_request):
        calls.append(catalog.sha256())
        return exact(catalog, candidate_request)

    monkeypatch.setattr(
        runtime,
        "run_raw_question_feature_answer",
        exact_probe,
    )
    result = runtime.run_indexed_raw_question_feature_registry_answer(
        index, request)
    assert result.status == "ANSWER"
    assert set(calls) == {
        entry.feature_catalog.sha256()
        for entry in index.registry.entries
        if entry.sha256() in expected
    }
    assert len(calls) < len(index.registry.entries)


def test_alias_and_implicit_continuations_only_run_indexed_candidates(
        indexed_registry, monkeypatch) -> None:
    from pure_integer_ai.experiments import (
        ph2_w03_w04_w05_question_feature_index as runtime,
    )

    index, two_entry, _ = indexed_registry
    exact = two_entry.feature_catalog.catalog[0]
    implicit = two_entry.implicit_bundle.catalog[0]
    requests = (
        (
            "ALIAS",
            RawQuestionRequest(_aligned_alias_surface(two_entry, exact)),
            "continue_question_feature_predicate_alias_answer",
        ),
        (
            "IMPLICIT",
            RawQuestionRequest(implicit.question_surface),
            "continue_implicit_predicate_question_answer",
        ),
    )
    for phase, request, attribute in requests:
        calls = []
        original = getattr(runtime, attribute)

        def probe(*args, _original=original, **kwargs):
            calls.append(args[0])
            return _original(*args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(runtime, attribute, probe)
            result = runtime.run_indexed_raw_question_feature_registry_answer(
                index, request)
        expected = lookup_indexed_question_feature_entries(
            index, phase, request)
        assert result.status == "ANSWER"
        assert len(calls) == len(expected)
        assert len(calls) < len(index.registry.entries)


def test_real_overlapping_learned_construction_converges_end_to_end(
        indexed_registry) -> None:
    _, two_entry, _ = indexed_registry
    shared = two_entry.feature_catalog.catalog[0]
    projected_catalog = replace(
        two_entry.feature_catalog,
        catalog=(shared,),
    )
    projected_implicit = replace(
        two_entry.implicit_bundle,
        explicit_catalog=projected_catalog,
    )
    projected_entry = RawQuestionFeatureRegistryEntry(
        projected_catalog,
        two_entry.alias_bridge,
        projected_implicit,
    )
    registry = build_raw_question_feature_registry(
        (two_entry, projected_entry))
    index = build_raw_question_feature_index(registry)
    request = RawQuestionRequest(
        shared.question_surface,
        shared.source_record_key,
    )
    result = _assert_scan_equal(index, request)
    assert result.status == "ANSWER"
    assert len(result.interpretations) == 1
    assert sum(
        trace.exact_result.status == "ANSWER" for trace in result.traces
    ) == 2


def test_index_runtime_contains_no_question_or_bundle_dispatch_table() -> None:
    source = (REPOSITORY / (
        "src/pure_integer_ai/experiments/"
        "ph2_w03_w04_w05_question_feature_index.py"
    )).read_text(encoding="utf-8")
    for value in ("暴雨", "寒潮", "山区", "桥面", "什么", "哪里"):
        assert value not in source
    assert "bundle_name" not in source
