"""FT12 learned predicate aliases over two constructions and two contents."""
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
from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question_alias import (
    PREDICATE_ALIAS_ANSWER_SHA256S,
    PREDICATE_ALIAS_BRIDGE_SHA256,
    PREDICATE_ALIAS_EXPRESSION_BOUNDARY,
    build_learned_predicate_alias_bridge,
    resolve_learned_predicate_alias,
    run_raw_question_predicate_alias_answer,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question_alias_contract import (
    LearnedPredicateAliasBridge,
    LearnedPredicateAliasRoute,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question_contract import (
    RawQuestionRequest,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question_generalization import (
    build_raw_question_generalization,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_vertical_generalization import (
    build_w03_w04_w05_vertical_generalization_overlay,
)


REPOSITORY = Path(__file__).resolve().parents[1]
SEMANTIC_SAMPLE = (
    REPOSITORY
    / "data/ph2/authored_semantic_primitive_bridge_generalization_v1.jsonl.sample"
)
PRIMITIVE_MAP_SAMPLE = (
    REPOSITORY
    / "data/ph2/authored_primitive_atomic_bridge_map_generalization_v1.jsonl.sample"
)
ATOMIC_SAMPLE = (
    REPOSITORY
    / "data/ph2/authored_primitive_atomic_bridge_seed_generalization_v1.jsonl.sample"
)
CAUSE_QUESTION_SAMPLE = (
    REPOSITORY
    / "data/ph2/authored_vertical_question_cause_generalization_v1.jsonl.sample"
)
EFFECT_QUESTION_SAMPLE = (
    REPOSITORY
    / "data/ph2/authored_vertical_question_effect_generalization_v1.jsonl.sample"
)


@pytest.fixture(scope="module")
def overlay(tmp_path_factory):
    root = tmp_path_factory.mktemp("ft12_predicate_alias")
    base = compile_authored_semantic_primitive_bridge_course(
        SEMANTIC_SAMPLE,
        root / "base",
    )
    donor = compile_authored_primitive_atomic_bridge_course(
        PRIMITIVE_MAP_SAMPLE,
        ATOMIC_SAMPLE,
        root / "donor",
    )
    return build_w03_w04_w05_vertical_generalization_overlay(base, donor)


@pytest.fixture(scope="module")
def bundle(overlay):
    return build_raw_question_generalization(
        overlay,
        CAUSE_QUESTION_SAMPLE,
        EFFECT_QUESTION_SAMPLE,
    )


@pytest.fixture(scope="module")
def bridge(overlay, bundle):
    return build_learned_predicate_alias_bridge(overlay, bundle)


def _run(overlay, bundle, bridge, question, *, source=None):
    return run_raw_question_predicate_alias_answer(
        bridge,
        bundle.catalog,
        overlay.w03_batch,
        overlay.w04_batch,
        overlay.w05_batch,
        RawQuestionRequest(question, source),
        overlay_validation_sha256=overlay.validation_sha256,
    )


def _bridge_with_routes(bridge, routes):
    ordered = tuple(sorted(routes, key=LearnedPredicateAliasRoute.sha256))
    payload = {
        "overlay_validation_sha256": bridge.overlay_validation_sha256,
        "raw_question_bundle_sha256": bridge.raw_question_bundle_sha256,
        "routes": [item.to_dict() for item in ordered],
        "w03_source_binding_sha256": bridge.w03_source_binding_sha256,
        "w04_source_binding_sha256": bridge.w04_source_binding_sha256,
    }
    identity = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return LearnedPredicateAliasBridge(
        bridge.overlay_validation_sha256,
        bridge.raw_question_bundle_sha256,
        bridge.w03_source_binding_sha256,
        bridge.w04_source_binding_sha256,
        ordered,
        identity,
    )


def test_bridge_uses_two_public_source_bound_replacement_routes(
        overlay, bundle, bridge) -> None:
    assert bridge.identity_sha256 == PREDICATE_ALIAS_BRIDGE_SHA256
    assert bridge.raw_question_bundle_sha256 == bundle.identity_sha256
    assert bridge.overlay_validation_sha256 == overlay.validation_sha256
    assert len(bridge.routes) == 2
    assert len({item.proposition_key for item in bridge.routes}) == 2
    assert len({item.predicate_source_ref_key for item in bridge.routes}) == 2
    assert len({item.alias_source_ref_key for item in bridge.routes}) == 2
    assert {len(item.evidence_keys) for item in bridge.routes} == {4}
    assert {item.primitive_registry for item in bridge.routes} == {"relation"}
    assert {item.primitive_kind for item in bridge.routes} == {4}
    assert PREDICATE_ALIAS_EXPRESSION_BOUNDARY == (
        ("explicit_predicate_alias", "SUPPORTED_BY_TWO_SOURCE_BOUND_ROUTES"),
        ("missing_alias_route", "UNKNOWN"),
        ("non_equivalent_alias_routes", "CLARIFY"),
        ("implicit_predicate", "UNKNOWN_UNTIL_A_LEARNED_STRUCTURE_EXISTS"),
        ("role_inventory", "CURRENTLY_PROVEN_FOR_TWO_ROLE_PROPOSITIONS"),
    )


def test_alias_resolution_groups_equivalent_provenance_without_private_winner(
        bridge) -> None:
    resolution = resolve_learned_predicate_alias(
        bridge, bridge.routes[0].alias_surface)

    assert resolution.status == "SELECTED"
    assert resolution.selected is not None
    assert len(resolution.selected.routes) == 2
    assert {
        item.predicate_source_ref_key
        for item in resolution.selected.routes
    } == {item.predicate_source_ref_key for item in bridge.routes}


def test_alias_replacement_answers_two_constructions_by_two_contents(
        overlay, bundle, bridge) -> None:
    questions = {
        "什么导致河水上涨？": "暴雨",
        "暴雨导致什么？": "河水上涨",
        "什么导致路面结冰？": "寒潮",
        "寒潮导致什么？": "路面结冰",
    }
    results = {
        question: _run(overlay, bundle, bridge, question)
        for question in questions
    }

    assert {
        question: (result.status, result.answer_surface)
        for question, result in results.items()
    } == {
        question: ("ANSWER", answer)
        for question, answer in questions.items()
    }
    assert all(
        result.exact_result.status == "UNKNOWN"
        and result.selected_match is not None
        and result.selected_match.selected
        and result.normalized_result is not None
        and result.normalized_result.typed_result is not None
        and result.normalized_result.typed_result.proof is not None
        and len(result.selected_match.aligned_route_sha256s) == 1
        for result in results.values()
    )
    assert tuple(sorted(
        item.sha256() for item in results.values()
    )) == PREDICATE_ALIAS_ANSWER_SHA256S


def test_ft11_exact_question_remains_an_exact_passthrough(
        overlay, bundle, bridge) -> None:
    result = _run(overlay, bundle, bridge, "什么使得河水上涨？")

    assert result.status == "ANSWER"
    assert result.answer_surface == "暴雨"
    assert result.exact_result.status == "ANSWER"
    assert result.matches == ()
    assert result.selected_match is None
    assert result.normalized_result is None


@pytest.mark.parametrize(
    "question",
    (
        "什么引发河水上涨？",
        "为何路面结冰？",
        "寒潮导致路面与桥面怎样？",
    ),
)
def test_missing_alias_implicit_and_unlearned_role_structure_remain_unknown(
        overlay, bundle, bridge, question) -> None:
    result = _run(overlay, bundle, bridge, question)

    assert result.status == "UNKNOWN"
    assert result.answer_surface is None
    assert result.selected_match is None
    assert result.normalized_result is None
    assert all(
        item.resolution.status == "MISSING" for item in result.matches)


def test_two_non_equivalent_complete_route_groups_require_clarification(
        overlay, bundle, bridge) -> None:
    conflicts = tuple(
        replace(
            item,
            sense_label=item.sense_label + "-conflict",
            primitive_kind=item.primitive_kind + 1,
        )
        for item in bridge.routes
    )
    ambiguous = _bridge_with_routes(bridge, (*bridge.routes, *conflicts))

    result = _run(
        overlay,
        bundle,
        ambiguous,
        "什么导致河水上涨？",
    )

    assert result.status == "CLARIFY"
    assert result.answer_surface is None
    assert result.selected_match is None
    assert result.normalized_result is None
    assert len(result.matches) == 1
    assert result.matches[0].resolution.status == "AMBIGUOUS"
    assert len(result.matches[0].resolution.options) == 2


def test_source_binding_repeatability_and_state_are_preserved(
        overlay, bundle, bridge) -> None:
    construction = next(
        item for item in bundle.catalog
        if item.question_surface == "什么使得路面结冰？")
    wrong_source = next(
        item.source_record_key for item in bundle.catalog
        if item.question_surface == "什么使得河水上涨？")
    before = tuple(
        batch.record_commitment
        for batch in (
            overlay.w03_batch,
            overlay.w04_batch,
            overlay.w05_batch,
        )
    )

    unknown = _run(
        overlay,
        bundle,
        bridge,
        "什么导致路面结冰？",
        source=wrong_source,
    )
    first = _run(
        overlay,
        bundle,
        bridge,
        "什么导致路面结冰？",
        source=construction.source_record_key,
    )
    second = _run(
        overlay,
        bundle,
        bridge,
        "什么导致路面结冰？",
        source=construction.source_record_key,
    )

    assert unknown.status == "UNKNOWN"
    assert first.status == second.status == "ANSWER"
    assert first.answer_surface == second.answer_surface == "寒潮"
    assert first.sha256() == second.sha256()
    assert before == tuple(
        batch.record_commitment
        for batch in (
            overlay.w03_batch,
            overlay.w04_batch,
            overlay.w05_batch,
        )
    )


def test_runtime_contains_no_predicate_alias_string_table() -> None:
    source = (
        REPOSITORY
        / "src/pure_integer_ai/experiments/"
        "ph2_w03_w04_w05_raw_question_alias.py"
    ).read_text(encoding="utf-8")

    assert "导致" not in source
    assert "使得" not in source
    assert "PREDICATE_ALIAS_TABLE" not in source
