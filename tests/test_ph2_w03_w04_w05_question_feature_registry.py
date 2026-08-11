"""FT16 global registry dispatch over two-Role and three-Role catalogs."""
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
    THREE_ROLE_FEATURE_CATALOG_SHA256,
    THREE_ROLE_IMPLICIT_QUESTION_BUNDLE_SHA256,
    THREE_ROLE_PREDICATE_ALIAS_BRIDGE_SHA256,
    THREE_ROLE_QUESTION_FEATURE_COMPOSITION_SHA256,
    build_three_role_question_feature_composition,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_question_feature_registry import (
    QUESTION_FEATURE_DISPATCH_PHASES,
    QUESTION_FEATURE_REGISTRY_EXPRESSION_BOUNDARY,
    QUESTION_FEATURE_REGISTRY_SHA256,
    RawQuestionFeatureRegistryEntry,
    build_raw_question_feature_registry,
    resolve_question_feature_interpretations,
    run_raw_question_feature_registry_answer,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question_alias import (
    PREDICATE_ALIAS_BRIDGE_SHA256,
    build_learned_predicate_alias_bridge,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question_contract import (
    RawQuestionRequest,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question_generalization import (
    RAW_QUESTION_GENERALIZATION_BUNDLE_SHA256,
    build_raw_question_generalization,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question_implicit import (
    IMPLICIT_QUESTION_BUNDLE_SHA256,
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
def feature_registry(tmp_path_factory):
    two_root = tmp_path_factory.mktemp("ft16_two_role")
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
    two_alias = build_learned_predicate_alias_bridge(two_catalog)
    two_implicit = build_implicit_question_bundle(
        two_catalog,
        DATA / "authored_vertical_question_implicit_reason_v1.jsonl.sample",
        DATA / "authored_vertical_question_implicit_result_v1.jsonl.sample",
    )
    two_entry = RawQuestionFeatureRegistryEntry(
        two_catalog,
        two_alias,
        two_implicit,
    )

    three_root = tmp_path_factory.mktemp("ft16_three_role")
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
    registry = build_raw_question_feature_registry(
        (two_entry, three_entry),
        expected_identity_sha256=QUESTION_FEATURE_REGISTRY_SHA256,
    )
    return registry, two_entry, three_entry, three_composition


def _alias_surface(entry, construction) -> str:
    route = next(
        item for item in entry.alias_bridge.routes
        if item.proposition_key
        == construction.vertical_result.link.proposition_key
    )
    return "".join(
        route.alias_surface if item.kind == "PREDICATE" else item.surface
        for item in construction.segments
    )


def _phase_question(entry, phase):
    if phase == "EXACT":
        return entry.feature_catalog.catalog[0].question_surface
    if phase == "ALIAS":
        return _alias_surface(entry, entry.feature_catalog.catalog[0])
    return entry.implicit_bundle.catalog[0].question_surface


def test_registry_freezes_both_catalogs_without_merging_public_batches(
        feature_registry) -> None:
    registry, two_entry, three_entry, three_composition = feature_registry
    assert QUESTION_FEATURE_DISPATCH_PHASES == (
        "EXACT", "ALIAS", "IMPLICIT")
    assert QUESTION_FEATURE_REGISTRY_EXPRESSION_BOUNDARY == (
        ("caller_input", "RAW_QUESTION_AND_OPTIONAL_SOURCE_REF"),
        ("catalog_selection", "READ_ONLY_REGISTRY"),
        ("global_priority", "EXACT_THEN_ALIAS_THEN_IMPLICIT"),
        ("equivalent_provenance", "CONVERGES_BY_TYPED_INTERPRETATION"),
        ("non_equivalent_interpretations", "CLARIFY"),
        ("missing_structure", "UNKNOWN"),
        ("source_binding", "NEVER_MERGED_ACROSS_PUBLIC_BATCHES"),
    )
    assert len(registry.entries) == 2
    assert registry.identity_sha256 == QUESTION_FEATURE_REGISTRY_SHA256
    assert {item.feature_catalog.bundle_identity_sha256
            for item in registry.entries} == {
        RAW_QUESTION_GENERALIZATION_BUNDLE_SHA256,
        three_composition.feature_catalog.bundle_identity_sha256,
    }
    assert two_entry.alias_bridge.identity_sha256 == (
        PREDICATE_ALIAS_BRIDGE_SHA256)
    assert two_entry.implicit_bundle.identity_sha256 == (
        IMPLICIT_QUESTION_BUNDLE_SHA256)
    assert three_entry.feature_catalog.sha256() == (
        THREE_ROLE_FEATURE_CATALOG_SHA256)
    assert three_entry.alias_bridge.identity_sha256 == (
        THREE_ROLE_PREDICATE_ALIAS_BRIDGE_SHA256)
    assert three_entry.implicit_bundle.identity_sha256 == (
        THREE_ROLE_IMPLICIT_QUESTION_BUNDLE_SHA256)
    assert three_composition.identity_sha256 == (
        THREE_ROLE_QUESTION_FEATURE_COMPOSITION_SHA256)
    assert two_entry.feature_catalog.w03_batch is not (
        three_entry.feature_catalog.w03_batch)
    assert two_entry.feature_catalog.w04_batch is not (
        three_entry.feature_catalog.w04_batch)
    assert two_entry.feature_catalog.w05_batch is not (
        three_entry.feature_catalog.w05_batch)


@pytest.mark.parametrize("phase", QUESTION_FEATURE_DISPATCH_PHASES)
def test_two_role_registry_dispatch_reaches_each_global_phase(
        feature_registry, phase) -> None:
    registry, two_entry, _, _ = feature_registry
    result = run_raw_question_feature_registry_answer(
        registry,
        RawQuestionRequest(_phase_question(two_entry, phase)),
    )
    assert result.status == "ANSWER"
    assert result.answer_surface is not None
    assert result.decisive_phase == phase
    assert len(result.interpretations) == 1
    assert result.selected_entry_sha256 == two_entry.sha256()


@pytest.mark.parametrize("phase", QUESTION_FEATURE_DISPATCH_PHASES)
def test_three_role_registry_dispatch_reaches_each_global_phase(
        feature_registry, phase) -> None:
    registry, _, three_entry, _ = feature_registry
    result = run_raw_question_feature_registry_answer(
        registry,
        RawQuestionRequest(_phase_question(three_entry, phase)),
    )
    assert result.status == "ANSWER"
    assert result.answer_surface is not None
    assert result.decisive_phase == phase
    assert len(result.interpretations) == 1
    assert result.selected_entry_sha256 == three_entry.sha256()


def test_wrong_source_and_missing_structure_remain_globally_unknown(
        feature_registry) -> None:
    registry, two_entry, _, _ = feature_registry
    construction = two_entry.feature_catalog.catalog[0]
    wrong_source = next(
        item.source_record_key for item in two_entry.feature_catalog.catalog
        if item.source_record_key != construction.source_record_key)
    wrong = run_raw_question_feature_registry_answer(
        registry,
        RawQuestionRequest(construction.question_surface, wrong_source),
    )
    missing = run_raw_question_feature_registry_answer(
        registry,
        RawQuestionRequest("未学习的结构？"),
    )
    for result in (wrong, missing):
        assert result.status == "UNKNOWN"
        assert result.answer_surface is None
        assert result.decisive_phase is None
        assert result.interpretations == ()
        assert result.selected_entry_sha256 is None
        assert all(item.implicit_result is not None for item in result.traces)


def test_equivalent_provenance_converges_and_cross_catalog_meanings_clarify(
        feature_registry) -> None:
    registry, two_entry, three_entry, _ = feature_registry
    two = run_raw_question_feature_registry_answer(
        registry,
        RawQuestionRequest(_phase_question(two_entry, "EXACT")),
    ).interpretations[0]
    three = run_raw_question_feature_registry_answer(
        registry,
        RawQuestionRequest(_phase_question(three_entry, "EXACT")),
    ).interpretations[0]
    assert resolve_question_feature_interpretations(()) == "MISSING"
    assert resolve_question_feature_interpretations((two, two)) == "SELECTED"
    assert resolve_question_feature_interpretations((two, three)) == (
        "AMBIGUOUS")
    for conflict in (
            replace(two, primitive_kind=two.primitive_kind + 1),
            replace(two, proposition_key=(9, 1)),
            replace(two, target_role_key=(9, 2)),
            replace(two, answer_filler_key=(9, 3)),
            replace(two, answer_surface=two.answer_surface + "冲突")):
        assert resolve_question_feature_interpretations((two, conflict)) == (
            "AMBIGUOUS")


def test_dispatch_completes_all_exact_catalogs_before_alias_and_stops_before_implicit(
        feature_registry, monkeypatch) -> None:
    from pure_integer_ai.experiments import (
        ph2_w03_w04_w05_question_feature_registry as runtime,
    )

    registry, two_entry, _, _ = feature_registry
    events = []
    exact = runtime.run_raw_question_feature_answer
    alias = runtime.run_question_feature_predicate_alias_answer

    def exact_probe(catalog, request):
        events.append(("EXACT", catalog.sha256()))
        return exact(catalog, request)

    def alias_probe(bridge, catalog, request):
        events.append(("ALIAS", catalog.sha256()))
        return alias(bridge, catalog, request)

    def implicit_forbidden(*args, **kwargs):
        raise AssertionError("implicit phase ran before global alias resolution")

    monkeypatch.setattr(runtime, "run_raw_question_feature_answer", exact_probe)
    monkeypatch.setattr(
        runtime,
        "run_question_feature_predicate_alias_answer",
        alias_probe,
    )
    monkeypatch.setattr(
        runtime,
        "run_implicit_predicate_question_answer",
        implicit_forbidden,
    )
    result = runtime.run_raw_question_feature_registry_answer(
        registry,
        RawQuestionRequest(_phase_question(two_entry, "ALIAS")),
    )
    assert result.status == "ANSWER"
    assert result.decisive_phase == "ALIAS"
    assert tuple(item[0] for item in events[:len(registry.entries)]) == (
        "EXACT", "EXACT")
    assert {item[1] for item in events[:len(registry.entries)]} == {
        item.feature_catalog.sha256() for item in registry.entries}


def test_registry_runtime_contains_no_question_or_bundle_dispatch_table() -> None:
    source = (REPOSITORY / (
        "src/pure_integer_ai/experiments/"
        "ph2_w03_w04_w05_question_feature_registry.py"
    )).read_text(encoding="utf-8")
    for value in ("暴雨", "寒潮", "山区", "桥面", "什么", "哪里"):
        assert value not in source
    assert "bundle_name" not in source
