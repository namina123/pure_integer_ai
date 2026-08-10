"""FT11 双问题构造、双来源命题与严格未知边界。"""
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
    canonical_json_line,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question import (
    run_raw_question_answer,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_question_answer import (
    project_w03_w04_w05_question_answer,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question_contract import (
    RawQuestionRequest,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question_generalization import (
    RAW_QUESTION_CAUSE_GENERALIZATION_SAMPLE_SHA256,
    RAW_QUESTION_EFFECT_GENERALIZATION_SAMPLE_SHA256,
    RAW_QUESTION_GENERALIZATION_BUNDLE_SHA256,
    RAW_QUESTION_GENERALIZATION_EXPRESSION_BOUNDARY,
    RAW_QUESTION_GENERALIZATION_VERTICAL_SHA256S,
    build_raw_question_generalization,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_vertical_generalization import (
    VERTICAL_GENERALIZATION_OVERLAY_VALIDATION_SHA256,
    VERTICAL_GENERALIZATION_TARGETS,
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
    root = tmp_path_factory.mktemp("ft11_question_generalization")
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


def _run(overlay, bundle, question, *, source=None):
    return run_raw_question_answer(
        bundle.catalog,
        overlay.w03_batch,
        overlay.w04_batch,
        overlay.w05_batch,
        RawQuestionRequest(question, source),
        overlay_validation_sha256=overlay.validation_sha256,
    )


@pytest.fixture(scope="module")
def answers(overlay, bundle):
    values = {}
    for construction in bundle.catalog:
        values[construction.question_surface] = _run(
            overlay,
            bundle,
            construction.question_surface,
            source=construction.source_record_key,
        )
    return values


@pytest.mark.parametrize(
    ("path", "expected_sha", "forbidden_answer"),
    (
        (
            CAUSE_QUESTION_SAMPLE,
            RAW_QUESTION_CAUSE_GENERALIZATION_SAMPLE_SHA256,
            "暴雨",
        ),
        (
            EFFECT_QUESTION_SAMPLE,
            RAW_QUESTION_EFFECT_GENERALIZATION_SAMPLE_SHA256,
            "路面结冰",
        ),
    ),
)
def test_question_samples_are_canonical_public_and_answer_free(
        path, expected_sha, forbidden_answer) -> None:
    payload = path.read_bytes()
    value = parse_canonical_json_bytes(payload[:-1], require_object=True)

    assert hashlib.sha256(payload).hexdigest() == expected_sha
    assert canonical_json_line(value) == payload
    assert value["license_id"] == "CC0-1.0"
    assert value["redistribution_policy"] == "PUBLIC"
    assert forbidden_answer.encode("utf-8") not in payload
    keys = set(value)
    for segment in value["segments"]:
        keys.update(segment)
    assert not any(
        "answer" in key or "expected" in key or "label" in key
        for key in keys)


def test_overlay_closes_two_distinct_source_bound_vertical_chains(
        overlay, bundle) -> None:
    assert overlay.validation_sha256 == (
        VERTICAL_GENERALIZATION_OVERLAY_VALIDATION_SHA256)
    assert len(overlay.targets) == 2
    assert len({
        item.base_source.stable_key for item in overlay.targets
    }) == 2
    assert len({
        item.overlay_w05_observation.stable_key for item in overlay.targets
    }) == 2
    assert (
        len(overlay.w03_batch.pairs),
        len(overlay.w04_batch.pairs),
        len(overlay.w05_batch.pairs),
    ) == (6, 6, 8)
    for item in overlay.targets:
        assert item.base_w04_observation.prerequisite_keys == (
            item.base_w03_observation.stable_key,)
        assert item.overlay_w05_observation.prerequisite_keys == (
            item.base_w04_observation.stable_key,)
        assert item.overlay_w05_observation.source_ref_key == (
            item.base_source.stable_key)
    assert tuple(
        item.sha256() for item in bundle.vertical_results
    ) == RAW_QUESTION_GENERALIZATION_VERTICAL_SHA256S
    assert all(item.status == "BRIDGED" for item in bundle.vertical_results)


def test_catalog_is_two_constructions_by_two_contents(bundle) -> None:
    assert bundle.identity_sha256 == RAW_QUESTION_GENERALIZATION_BUNDLE_SHA256
    assert len(bundle.patterns) == 2
    assert len(bundle.catalog) == 4
    assert len({
        item.pattern.sha256() for item in bundle.catalog
    }) == 2
    assert len({
        item.source_record_key for item in bundle.catalog
    }) == 2
    assert len({
        (item.pattern.sha256(), item.source_record_key)
        for item in bundle.catalog
    }) == 4
    assert RAW_QUESTION_GENERALIZATION_EXPRESSION_BOUNDARY == (
        ("different_segment_order", "SUPPORTED_BY_LEARNED_CONSTRUCTION"),
        ("explicit_predicate", "SUPPORTED_BY_SOURCE_OCCURRENCE"),
        ("implicit_predicate", "UNKNOWN_UNTIL_A_LEARNED_STRUCTURE_EXISTS"),
        ("predicate_alias", "UNKNOWN_UNTIL_A_LEARNED_LEXICAL_LINK_EXISTS"),
        ("role_inventory", "CURRENTLY_PROVEN_FOR_TWO_ROLE_PROPOSITIONS"),
    )


def test_construction_and_content_replacement_both_answer(
        answers) -> None:
    assert {
        question: (result.status, result.answer_surface)
        for question, result in answers.items()
    } == {
        "什么使得河水上涨？": ("ANSWER", "暴雨"),
        "暴雨使得什么？": ("ANSWER", "河水上涨"),
        "什么使得路面结冰？": ("ANSWER", "寒潮"),
        "寒潮使得什么？": ("ANSWER", "路面结冰"),
    }
    assert all(
        result.typed_result is not None
        and result.typed_result.proof is not None
        and result.typed_result.state_before_sha256
        == result.typed_result.state_after_sha256
        for result in answers.values()
    )


def test_equivalent_generation_evidence_does_not_become_false_ambiguity(
        answers) -> None:
    for result in answers.values():
        typed = result.typed_result
        assert typed is not None and typed.proof is not None
        vertical = typed.vertical_result
        matching = tuple(
            item
            for item in vertical.w04_w05.w05_result.generation_options
            if item.target_proposition_key == vertical.link.proposition_key
        )
        assert len(matching) == 2
        assert len({
            item.construction_source_ref_key for item in matching
        }) == 2
        assert {item.surface for item in matching} == {
            typed.proof.generated_proposition_surface}


def test_functionally_different_generation_evidence_still_clarifies(
        answers) -> None:
    typed = answers["什么使得河水上涨？"].typed_result
    assert typed is not None
    vertical = typed.vertical_result
    w05 = vertical.w04_w05.w05_result
    changed = replace(w05.generation_options[1], surface="冲突输出。")
    changed_w05 = replace(
        w05,
        generation_options=(w05.generation_options[0], changed),
    )
    changed_bridge = replace(vertical.w04_w05, w05_result=changed_w05)
    changed_vertical = replace(vertical, w04_w05=changed_bridge)

    result = project_w03_w04_w05_question_answer(
        typed.request,
        changed_vertical,
        state_sha256=typed.state_before_sha256,
    )

    assert result.status == "CLARIFY"
    assert result.answer_surface is None
    assert result.proof is None


@pytest.mark.parametrize(
    "question",
    (
        "何者造成路面结冰？",
        "为何路面结冰？",
        "寒潮使得路面与桥面怎样？",
    ),
)
def test_unlearned_alias_implicit_and_role_structure_are_unknown(
        overlay, bundle, question) -> None:
    result = _run(overlay, bundle, question)

    assert result.status == "UNKNOWN"
    assert result.answer_surface is None
    assert result.selected_construction is None
    assert result.typed_result is None


def test_source_binding_and_repeatability_remain_exact(
        overlay, bundle) -> None:
    construction = next(
        item for item in bundle.catalog
        if item.question_surface == "什么使得路面结冰？")
    wrong_source = next(
        item.base_source.stable_key.components
        for item in overlay.targets
        if item.spec == VERTICAL_GENERALIZATION_TARGETS[0])
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
        construction.question_surface,
        source=wrong_source,
    )
    first = _run(
        overlay,
        bundle,
        construction.question_surface,
        source=construction.source_record_key,
    )
    second = _run(
        overlay,
        bundle,
        construction.question_surface,
        source=construction.source_record_key,
    )

    assert unknown.status == "UNKNOWN"
    assert first.status == second.status == "ANSWER"
    assert first.sha256() == second.sha256()
    assert before == tuple(
        batch.record_commitment
        for batch in (
            overlay.w03_batch,
            overlay.w04_batch,
            overlay.w05_batch,
        )
    )
