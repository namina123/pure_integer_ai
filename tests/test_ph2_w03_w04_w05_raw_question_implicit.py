"""FT13 双内容上的已学隐式谓词问题构造。"""
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
    IMPLICIT_QUESTION_ANSWER_SHA256S,
    IMPLICIT_QUESTION_BUNDLE_SHA256,
    IMPLICIT_QUESTION_EXPRESSION_BOUNDARY,
    IMPLICIT_QUESTION_REASON_SAMPLE_SHA256,
    IMPLICIT_QUESTION_RESULT_SAMPLE_SHA256,
    ImplicitQuestionInterpretationKey,
    build_implicit_question_bundle,
    resolve_implicit_question_interpretations,
    run_implicit_predicate_question_answer,
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
REASON_QUESTION_SAMPLE = (
    REPOSITORY
    / "data/ph2/authored_vertical_question_implicit_reason_v1.jsonl.sample"
)
RESULT_QUESTION_SAMPLE = (
    REPOSITORY
    / "data/ph2/authored_vertical_question_implicit_result_v1.jsonl.sample"
)


@pytest.fixture(scope="module")
def overlay(tmp_path_factory):
    root = tmp_path_factory.mktemp("ft13_implicit_question")
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
def explicit_bundle(overlay):
    return build_raw_question_generalization(
        overlay,
        CAUSE_QUESTION_SAMPLE,
        EFFECT_QUESTION_SAMPLE,
    )


@pytest.fixture(scope="module")
def alias_bridge(overlay, explicit_bundle):
    return build_learned_predicate_alias_bridge(overlay, explicit_bundle)


@pytest.fixture(scope="module")
def implicit_bundle(explicit_bundle):
    return build_implicit_question_bundle(
        explicit_bundle,
        REASON_QUESTION_SAMPLE,
        RESULT_QUESTION_SAMPLE,
    )


def _run(overlay, alias_bridge, implicit_bundle, question, *, source=None):
    return run_implicit_predicate_question_answer(
        alias_bridge,
        implicit_bundle,
        overlay.w03_batch,
        overlay.w04_batch,
        overlay.w05_batch,
        RawQuestionRequest(question, source),
        overlay_validation_sha256=overlay.validation_sha256,
    )


@pytest.mark.parametrize(
    ("path", "expected_sha", "forbidden_answer"),
    (
        (
            REASON_QUESTION_SAMPLE,
            IMPLICIT_QUESTION_REASON_SAMPLE_SHA256,
            "暴雨",
        ),
        (
            RESULT_QUESTION_SAMPLE,
            IMPLICIT_QUESTION_RESULT_SAMPLE_SHA256,
            "路面结冰",
        ),
    ),
)
def test_implicit_samples_are_canonical_public_and_answer_free(
        path, expected_sha, forbidden_answer) -> None:
    payload = path.read_bytes()
    value = parse_canonical_json_bytes(payload[:-1], require_object=True)

    assert hashlib.sha256(payload).hexdigest() == expected_sha
    assert canonical_json_line(value) == payload
    assert value["license_id"] == "CC0-1.0"
    assert value["redistribution_policy"] == "PUBLIC"
    assert forbidden_answer.encode("utf-8") not in payload
    assert "PREDICATE" not in {
        segment["kind"] for segment in value["segments"]}
    keys = set(value)
    for segment in value["segments"]:
        keys.update(segment)
    assert not any(
        "answer" in key or "expected" in key or "label" in key
        for key in keys)


def test_bundle_is_two_implicit_constructions_by_two_contents(
        explicit_bundle, alias_bridge, implicit_bundle) -> None:
    assert explicit_bundle.identity_sha256 == (
        RAW_QUESTION_GENERALIZATION_BUNDLE_SHA256)
    assert alias_bridge.identity_sha256 == PREDICATE_ALIAS_BRIDGE_SHA256
    assert implicit_bundle.identity_sha256 == IMPLICIT_QUESTION_BUNDLE_SHA256
    assert len(implicit_bundle.patterns) == 2
    assert len(implicit_bundle.catalog) == 4
    assert len({
        item.pattern.sha256() for item in implicit_bundle.catalog
    }) == 2
    assert len({
        item.source_record_key for item in implicit_bundle.catalog
    }) == 2
    assert all(
        sum(segment.kind == "PREDICATE" for segment in pattern.segments) == 0
        and sum(segment.kind == "BOUNDARY" for segment in pattern.segments)
        == 2
        for pattern in implicit_bundle.patterns
    )
    assert IMPLICIT_QUESTION_EXPRESSION_BOUNDARY == (
        ("explicit_or_alias_predicate", "PRESERVED_FROM_FT12"),
        ("implicit_predicate", "SUPPORTED_BY_LEARNED_CONSTRUCTION"),
        ("construction_replacement", "TWO_INDEPENDENT_PATTERNS"),
        ("content_replacement", "TWO_SOURCE_BOUND_PROPOSITIONS"),
        ("missing_structure", "UNKNOWN"),
        ("non_equivalent_interpretations", "CLARIFY"),
        ("role_inventory", "CURRENTLY_PROVEN_FOR_TWO_ROLE_PROPOSITIONS"),
    )


def test_implicit_construction_and_content_replacement_answer(
        overlay, alias_bridge, implicit_bundle) -> None:
    questions = {
        "河水上涨的原因是什么？": "暴雨",
        "路面结冰的原因是什么？": "寒潮",
        "暴雨的结果是什么？": "河水上涨",
        "寒潮的结果是什么？": "路面结冰",
    }
    results = {
        question: _run(overlay, alias_bridge, implicit_bundle, question)
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
        result.predicate_result.status == "UNKNOWN"
        and result.implicit_result is not None
        and result.implicit_result.typed_result is not None
        and result.implicit_result.typed_result.proof is not None
        and result.implicit_result.typed_result.state_before_sha256
        == result.implicit_result.typed_result.state_after_sha256
        for result in results.values()
    )
    assert tuple(sorted(
        result.sha256() for result in results.values()
    )) == IMPLICIT_QUESTION_ANSWER_SHA256S


@pytest.mark.parametrize(
    ("question", "answer"),
    (
        ("什么使得河水上涨？", "暴雨"),
        ("什么导致河水上涨？", "暴雨"),
    ),
)
def test_ft11_and_ft12_answers_remain_passthrough(
        overlay, alias_bridge, implicit_bundle, question, answer) -> None:
    result = _run(overlay, alias_bridge, implicit_bundle, question)

    assert result.status == "ANSWER"
    assert result.answer_surface == answer
    assert result.predicate_result.status == "ANSWER"
    assert result.implicit_result is None


def test_missing_structure_and_wrong_source_remain_unknown(
        overlay, alias_bridge, implicit_bundle) -> None:
    construction = next(
        item for item in implicit_bundle.catalog
        if item.question_surface == "路面结冰的原因是什么？")
    wrong_source = next(
        item.source_record_key for item in implicit_bundle.catalog
        if item.source_record_key != construction.source_record_key)
    questions = (
        ("河水上涨是怎么回事？", None),
        ("寒潮将怎样？", None),
        (construction.question_surface, wrong_source),
    )

    for question, source in questions:
        result = _run(
            overlay,
            alias_bridge,
            implicit_bundle,
            question,
            source=source,
        )
        assert result.status == "UNKNOWN"
        assert result.answer_surface is None
        assert result.predicate_result.status == "UNKNOWN"
        assert result.implicit_result is not None
        assert result.implicit_result.status == "UNKNOWN"


def test_non_equivalent_primitive_role_interpretations_clarify() -> None:
    base = ImplicitQuestionInterpretationKey(
        "relation", 4, (1, 2), (3, 4))

    assert resolve_implicit_question_interpretations(()) == "MISSING"
    assert resolve_implicit_question_interpretations(
        (base, base)) == "SELECTED"
    assert resolve_implicit_question_interpretations((
        base,
        replace(base, primitive_kind=5),
    )) == "AMBIGUOUS"
    assert resolve_implicit_question_interpretations((
        base,
        replace(base, target_role_key=(9, 9)),
    )) == "AMBIGUOUS"


def test_source_binding_repeatability_and_public_state_are_preserved(
        overlay, alias_bridge, implicit_bundle) -> None:
    construction = next(
        item for item in implicit_bundle.catalog
        if item.question_surface == "寒潮的结果是什么？")
    before = tuple(
        batch.record_commitment
        for batch in (
            overlay.w03_batch,
            overlay.w04_batch,
            overlay.w05_batch,
        )
    )

    first = _run(
        overlay,
        alias_bridge,
        implicit_bundle,
        construction.question_surface,
        source=construction.source_record_key,
    )
    second = _run(
        overlay,
        alias_bridge,
        implicit_bundle,
        construction.question_surface,
        source=construction.source_record_key,
    )

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


def test_production_runtime_contains_no_question_word_dispatch() -> None:
    source = (
        REPOSITORY
        / "src/pure_integer_ai/experiments/"
        "ph2_w03_w04_w05_raw_question_implicit.py"
    ).read_text(encoding="utf-8")

    assert "河水上涨的原因是什么？" not in source
    assert "寒潮的结果是什么？" not in source
    assert "为何" not in source
    assert "为什么" not in source
