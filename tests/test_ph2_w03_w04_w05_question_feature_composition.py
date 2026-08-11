"""FT15 三 Role 事实上的显式、alias 与隐式问题组合。"""
from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_authored_primitive_atomic_bridge_course import (
    compile_authored_primitive_atomic_bridge_course,
)
from pure_integer_ai.experiments.ph2_authored_semantic_primitive_bridge_course import (
    compile_authored_semantic_primitive_bridge_course,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_question_feature_composition import (
    THREE_ROLE_FEATURE_CATALOG_SHA256,
    THREE_ROLE_IMPLICIT_ACTOR_SAMPLE_SHA256,
    THREE_ROLE_IMPLICIT_LOCATION_SAMPLE_SHA256,
    THREE_ROLE_IMPLICIT_QUESTION_BUNDLE_SHA256,
    THREE_ROLE_PREDICATE_ALIAS_BRIDGE_SHA256,
    THREE_ROLE_QUESTION_FEATURE_ANSWER_SHA256S,
    THREE_ROLE_QUESTION_FEATURE_COMPOSITION_SHA256,
    THREE_ROLE_QUESTION_FEATURE_EXPRESSION_BOUNDARY,
    build_three_role_question_feature_composition,
    run_three_role_question_feature_answer,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question_contract import (
    RawQuestionRequest,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question_implicit import (
    ImplicitQuestionInterpretationKey,
    resolve_implicit_question_interpretations,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question_three_role import (
    build_three_role_question_bundle,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_vertical_three_role import (
    build_w03_w04_w05_three_role_vertical_overlay,
)


REPOSITORY = Path(__file__).resolve().parents[1]
SEMANTIC_SAMPLE = REPOSITORY / (
    "data/ph2/authored_semantic_primitive_bridge_three_role_v1.jsonl.sample")
PRIMITIVE_MAP_SAMPLE = REPOSITORY / (
    "data/ph2/authored_primitive_atomic_bridge_map_three_role_v1.jsonl.sample")
ATOMIC_SAMPLE = REPOSITORY / (
    "data/ph2/authored_primitive_atomic_bridge_seed_three_role_v1.jsonl.sample")
ACTOR_SAMPLE = REPOSITORY / (
    "data/ph2/authored_vertical_question_three_role_actor_v1.jsonl.sample")
LOCATION_SAMPLE = REPOSITORY / (
    "data/ph2/authored_vertical_question_three_role_location_v1.jsonl.sample")
IMPLICIT_ACTOR_SAMPLE = REPOSITORY / (
    "data/ph2/authored_vertical_question_three_role_implicit_actor_v1.jsonl.sample")
IMPLICIT_LOCATION_SAMPLE = REPOSITORY / (
    "data/ph2/authored_vertical_question_three_role_implicit_location_v1.jsonl.sample")


@pytest.fixture(scope="module")
def composition(tmp_path_factory):
    root = tmp_path_factory.mktemp("ft15_question_feature_composition")
    base = compile_authored_semantic_primitive_bridge_course(
        SEMANTIC_SAMPLE,
        root / "base",
    )
    donor = compile_authored_primitive_atomic_bridge_course(
        PRIMITIVE_MAP_SAMPLE,
        ATOMIC_SAMPLE,
        root / "donor",
    )
    overlay = build_w03_w04_w05_three_role_vertical_overlay(base, donor)
    explicit = build_three_role_question_bundle(
        overlay,
        ACTOR_SAMPLE,
        LOCATION_SAMPLE,
    )
    return build_three_role_question_feature_composition(
        explicit,
        IMPLICIT_ACTOR_SAMPLE,
        IMPLICIT_LOCATION_SAMPLE,
    )


def _alias_surface(construction, bridge) -> str:
    route = next(
        item for item in bridge.routes
        if item.proposition_key
        == construction.vertical_result.link.proposition_key
    )
    return "".join(
        route.alias_surface if item.kind == "PREDICATE" else item.surface
        for item in construction.segments
    )


def _run(composition, surface: str, *, source=None):
    return run_three_role_question_feature_answer(
        composition,
        RawQuestionRequest(surface, source),
    )


@pytest.mark.parametrize(
    ("path", "expected_sha"),
    (
        (IMPLICIT_ACTOR_SAMPLE, THREE_ROLE_IMPLICIT_ACTOR_SAMPLE_SHA256),
        (IMPLICIT_LOCATION_SAMPLE, THREE_ROLE_IMPLICIT_LOCATION_SAMPLE_SHA256),
    ),
)
def test_implicit_samples_are_public_canonical_and_answer_free(
        path, expected_sha) -> None:
    payload = path.read_bytes()
    value = parse_canonical_json_bytes(payload[:-1], require_object=True)
    assert hashlib.sha256(payload).hexdigest() == expected_sha
    assert value["license_id"] == "CC0-1.0"
    assert value["redistribution_policy"] == "PUBLIC"
    assert all("answer" not in key.lower() for key in value)
    assert all(item["kind"] != "PREDICATE" for item in value["segments"])


def test_composition_uses_one_shared_catalog_and_frozen_public_routes(
        composition) -> None:
    assert composition.feature_catalog.sha256() == (
        THREE_ROLE_FEATURE_CATALOG_SHA256)
    assert composition.alias_bridge.identity_sha256 == (
        THREE_ROLE_PREDICATE_ALIAS_BRIDGE_SHA256)
    assert composition.implicit_bundle.identity_sha256 == (
        THREE_ROLE_IMPLICIT_QUESTION_BUNDLE_SHA256)
    assert composition.identity_sha256 == (
        THREE_ROLE_QUESTION_FEATURE_COMPOSITION_SHA256)
    assert composition.implicit_bundle.explicit_catalog is (
        composition.feature_catalog)
    assert len(composition.feature_catalog.catalog) == 4
    assert len(composition.alias_bridge.routes) == 2
    assert len(composition.implicit_bundle.catalog) == 4
    assert THREE_ROLE_QUESTION_FEATURE_EXPRESSION_BOUNDARY == (
        ("explicit_predicate", "THREE_ROLE_TWO_TARGETS_TWO_CONTENTS"),
        ("predicate_alias", "PUBLIC_SUPERSEDE_ROUTES"),
        ("implicit_predicate", "TWO_ANSWER_FREE_LEARNED_CONSTRUCTIONS"),
        ("missing_learned_feature", "UNKNOWN"),
        ("non_equivalent_interpretations", "CLARIFY"),
        ("role_inventory", "PROVEN_FOR_THREE_ROLE_PROPOSITIONS"),
    )


def test_explicit_alias_and_implicit_cross_products_answer_both_roles(
        composition) -> None:
    explicit = [
        _run(composition, item.question_surface)
        for item in composition.feature_catalog.catalog
    ]
    aliases = [
        _run(composition, _alias_surface(item, composition.alias_bridge))
        for item in composition.feature_catalog.catalog
    ]
    implicit = [
        _run(composition, item.question_surface)
        for item in composition.implicit_bundle.catalog
    ]
    assert all(item.status == "ANSWER" for item in explicit + aliases + implicit)
    assert all(item.implicit_result is None for item in explicit + aliases)
    assert all(item.predicate_result.exact_result.status == "ANSWER"
               for item in explicit)
    assert all(item.predicate_result.exact_result.status == "UNKNOWN"
               and item.predicate_result.selected_match is not None
               and item.predicate_result.normalized_result is not None
               for item in aliases)
    assert all(item.predicate_result.status == "UNKNOWN"
               and item.implicit_result is not None
               and item.implicit_result.status == "ANSWER"
               for item in implicit)
    assert {item.answer_surface for item in explicit} == {
        "暴雨", "寒潮", "山区", "桥面"}
    assert {item.answer_surface for item in aliases} == {
        "暴雨", "寒潮", "山区", "桥面"}
    assert {item.answer_surface for item in implicit} == {
        "暴雨", "寒潮", "山区", "桥面"}
    assert tuple(item.sha256() for item in explicit + aliases + implicit) == (
        THREE_ROLE_QUESTION_FEATURE_ANSWER_SHA256S)


def test_missing_feature_wrong_source_and_broken_vertical_remain_unknown(
        composition) -> None:
    first = composition.feature_catalog.catalog[0]
    other_source = next(
        item.source_record_key for item in composition.feature_catalog.catalog
        if item.source_record_key != first.source_record_key
    )
    assert _run(composition, "未学习的问题？").status == "UNKNOWN"
    assert _run(
        composition,
        first.question_surface,
        source=other_source,
    ).status == "UNKNOWN"
    broken = replace(
        composition.feature_catalog,
        overlay_validation_sha256="0" * 64,
    )
    with pytest.raises(ValueError):
        replace(composition, feature_catalog=broken)


def test_non_equivalent_implicit_interpretations_remain_clarify(
        composition) -> None:
    first, second = composition.implicit_bundle.catalog[:2]
    one = ImplicitQuestionInterpretationKey(
        first.vertical_result.link.primitive_registry,
        first.vertical_result.link.primitive_kind,
        first.vertical_result.link.proposition_key,
        first.target_role_key,
    )
    other = ImplicitQuestionInterpretationKey(
        second.vertical_result.link.primitive_registry,
        second.vertical_result.link.primitive_kind,
        second.vertical_result.link.proposition_key,
        second.target_role_key,
    )
    assert resolve_implicit_question_interpretations((one,)) == "SELECTED"
    assert resolve_implicit_question_interpretations((one, other)) == (
        "AMBIGUOUS")


def test_repeated_composed_answer_is_identity_stable(composition) -> None:
    construction = composition.implicit_bundle.catalog[0]
    first = _run(
        composition,
        construction.question_surface,
        source=construction.source_record_key,
    )
    second = _run(
        composition,
        construction.question_surface,
        source=construction.source_record_key,
    )
    assert first == second
    assert first.sha256() == second.sha256()
    assert first.implicit_result is not None
    assert first.implicit_result.typed_result is not None
    assert first.implicit_result.typed_result.state_before_sha256 == (
        first.implicit_result.typed_result.state_after_sha256)


def test_shared_runtime_contains_no_question_or_answer_string_tables() -> None:
    source = (REPOSITORY / (
        "src/pure_integer_ai/experiments/"
        "ph2_w03_w04_w05_question_feature_composition.py"
    )).read_text(encoding="utf-8")
    for value in ("暴雨", "寒潮", "山区", "桥面", "什么", "哪里"):
        assert value not in source
