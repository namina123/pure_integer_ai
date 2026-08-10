"""FT10 从无答案问题构造恢复 target Role 并进入 FT09。"""
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
    RAW_QUESTION_ANSWER_RESULT_SHA256,
    RAW_QUESTION_CONSTRUCTION_SHA256,
    RAW_QUESTION_PATTERN_SHA256,
    RAW_QUESTION_REQUEST_SHA256,
    RAW_QUESTION_SAMPLE_SHA256,
    build_raw_question_catalog,
    compile_raw_question_pattern,
    run_raw_question_answer,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question_contract import (
    RawQuestionRequest,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_vertical import (
    run_w03_w04_w05_vertical_query,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_vertical_contract import (
    W03W04W05VerticalQuery,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_vertical_overlay import (
    VERTICAL_CONTEXT,
    VERTICAL_SURFACE,
    build_w03_w04_w05_vertical_overlay,
)
from pure_integer_ai.experiments.ph2_w04_payload import W04TrainingPayload
from pure_integer_ai.experiments.ph2_w04_v2_public_source import (
    build_w04_v2_public_evaluation_batch,
)
from pure_integer_ai.experiments.ph2_w05_payload import W05TrainingPayload
from pure_integer_ai.experiments.ph2_w05_v2_public_source import (
    build_w05_v2_public_evaluation_batch,
)


REPOSITORY = Path(__file__).resolve().parents[1]
SEMANTIC_PRIMITIVE_SAMPLE = (
    REPOSITORY /
    "data/ph2/authored_semantic_primitive_bridge_seed_v1.jsonl.sample")
PRIMITIVE_MAP_SAMPLE = (
    REPOSITORY /
    "data/ph2/authored_primitive_atomic_bridge_map_v1.jsonl.sample")
ATOMIC_SAMPLE = (
    REPOSITORY /
    "data/ph2/authored_primitive_atomic_bridge_seed_v1.jsonl.sample")
QUESTION_SAMPLE = (
    REPOSITORY /
    "data/ph2/authored_vertical_question_construction_v1.jsonl.sample")


@pytest.fixture(scope="module")
def overlay(tmp_path_factory):
    root = tmp_path_factory.mktemp("ft10_raw_question")
    base = compile_authored_semantic_primitive_bridge_course(
        SEMANTIC_PRIMITIVE_SAMPLE, root / "base")
    donor = compile_authored_primitive_atomic_bridge_course(
        PRIMITIVE_MAP_SAMPLE, ATOMIC_SAMPLE, root / "donor")
    return build_w03_w04_w05_vertical_overlay(base, donor)


@pytest.fixture(scope="module")
def vertical(overlay):
    return run_w03_w04_w05_vertical_query(
        overlay.w03_batch,
        overlay.w04_batch,
        overlay.w05_batch,
        W03W04W05VerticalQuery(
            VERTICAL_SURFACE,
            VERTICAL_CONTEXT,
            VERTICAL_CONTEXT,
        ),
        overlay_validation_sha256=overlay.validation_sha256,
    )


@pytest.fixture(scope="module")
def pattern(vertical):
    return compile_raw_question_pattern(QUESTION_SAMPLE, vertical)


@pytest.fixture(scope="module")
def catalog(pattern, vertical):
    return build_raw_question_catalog((pattern,), (vertical,))


def _run(overlay, catalog, question, *, source=None, w04=None, w05=None):
    return run_raw_question_answer(
        catalog,
        overlay.w03_batch,
        overlay.w04_batch if w04 is None else w04,
        overlay.w05_batch if w05 is None else w05,
        RawQuestionRequest(question, source),
        overlay_validation_sha256=overlay.validation_sha256,
    )


def _variant_sample(tmp_path, *, construction_id, question_surface):
    payload = QUESTION_SAMPLE.read_bytes()
    value = parse_canonical_json_bytes(payload[:-1], require_object=True)
    value["construction_id"] = construction_id
    value["question_surface"] = question_surface
    path = tmp_path / f"{construction_id}.jsonl"
    body = canonical_json_line(value)
    path.write_bytes(body)
    return path, hashlib.sha256(body).hexdigest()


def test_question_sample_is_canonical_public_and_contains_no_answer_label(
        vertical) -> None:
    payload = QUESTION_SAMPLE.read_bytes()
    value = parse_canonical_json_bytes(payload[:-1], require_object=True)

    assert hashlib.sha256(payload).hexdigest() == RAW_QUESTION_SAMPLE_SHA256
    assert canonical_json_line(value) == payload
    assert value["license_id"] == "CC0-1.0"
    assert value["redistribution_policy"] == "PUBLIC"
    assert value["exemplar_vertical_sha256"] == vertical.sha256()
    assert "暴雨".encode("utf-8") not in payload
    assert not {
        "vertical_context", "vertical_proposition_surface", "vertical_surface",
    }.intersection(value)
    flattened_keys = set(value)
    for segment in value["segments"]:
        flattened_keys.update(segment)
    assert not any(
        "answer" in key or "expected" in key or "label" in key
        for key in flattened_keys)
    compiled = compile_raw_question_pattern(QUESTION_SAMPLE, vertical)
    assert compiled.sample_sha256 == RAW_QUESTION_SAMPLE_SHA256
    assert compiled.sha256() == RAW_QUESTION_PATTERN_SHA256


def test_raw_question_recovers_role_and_answers_only_through_ft09(
        overlay, catalog) -> None:
    request = RawQuestionRequest("什么使得河水上涨？")

    result = _run(overlay, catalog, request.question_surface)

    assert request.to_dict() == {
        "question_surface": "什么使得河水上涨？",
        "source_record_key": None,
    }
    assert request.sha256() == RAW_QUESTION_REQUEST_SHA256
    assert result.status == "ANSWER"
    assert result.answer_surface == "暴雨"
    assert result.selected_construction is not None
    assert result.selected_construction.sha256() == (
        RAW_QUESTION_CONSTRUCTION_SHA256)
    assert result.typed_result is not None
    assert result.typed_result.status == "ANSWER"
    assert result.typed_result.request.target_role_keys == (
        result.selected_construction.target_role_key,)
    assert result.typed_result.state_before_sha256 == (
        result.typed_result.state_after_sha256)
    assert result.sha256() == RAW_QUESTION_ANSWER_RESULT_SHA256


def test_variable_surface_is_learned_from_data_not_hardcoded(
        tmp_path, overlay, vertical) -> None:
    path, sample_sha = _variant_sample(
        tmp_path,
        construction_id="causal-alternative-variable-v1",
        question_surface="何者使得河水上涨？",
    )
    alternative = compile_raw_question_pattern(
        path,
        vertical,
        expected_sample_sha256=sample_sha,
    )
    alternative_catalog = build_raw_question_catalog(
        (alternative,), (vertical,))

    answered = _run(overlay, alternative_catalog, "何者使得河水上涨？")
    unknown = _run(overlay, alternative_catalog, "什么使得河水上涨？")

    assert answered.status == "ANSWER"
    assert answered.answer_surface == "暴雨"
    assert unknown.status == "UNKNOWN"


def test_unlearned_raw_question_is_unknown(overlay, catalog) -> None:
    result = _run(overlay, catalog, "为何河水上涨？")

    assert result.status == "UNKNOWN"
    assert result.answer_surface is None
    assert result.selected_construction is None
    assert result.typed_result is None


def test_multiple_learned_constructions_require_clarification(
        tmp_path, overlay, vertical, pattern) -> None:
    path, sample_sha = _variant_sample(
        tmp_path,
        construction_id="causal-duplicate-evidence-v1",
        question_surface="什么使得河水上涨？",
    )
    duplicate = compile_raw_question_pattern(
        path,
        vertical,
        expected_sample_sha256=sample_sha,
    )
    ambiguous_catalog = build_raw_question_catalog(
        tuple(sorted((pattern, duplicate), key=lambda item: item.sha256())),
        (vertical,),
    )

    result = _run(overlay, ambiguous_catalog, "什么使得河水上涨？")

    assert result.status == "CLARIFY"
    assert len(result.matched_construction_sha256s) == 2
    assert result.selected_construction is None
    assert result.typed_result is None


def test_source_domain_mismatch_is_unknown(overlay, catalog) -> None:
    result = _run(
        overlay,
        catalog,
        "什么使得河水上涨？",
        source=(1, 8_888_888),
    )

    assert result.status == "UNKNOWN"
    assert result.matched_construction_sha256s == ()


@pytest.mark.parametrize("edge", ["W03_W04", "W04_W05"])
def test_current_vertical_break_prevents_raw_answer(
        edge, overlay, catalog) -> None:
    w04 = None
    w05 = None
    if edge == "W03_W04":
        observations = tuple(
            replace(item.observation, prerequisite_keys=())
            if item.observation.stable_key
            == overlay.base_w04_observation.stable_key
            else item.observation
            for item in overlay.w04_batch.pairs
        )
        w04 = build_w04_v2_public_evaluation_batch(W04TrainingPayload(
            tuple(item.record for item in overlay.w04_batch.source_records),
            observations,
            tuple(item.evidence for item in overlay.w04_batch.pairs),
        ))
    else:
        observations = tuple(
            replace(item.observation, prerequisite_keys=())
            if item.observation.stable_key
            == overlay.overlay_w05_observation.stable_key
            else item.observation
            for item in overlay.w05_batch.pairs
        )
        w05 = build_w05_v2_public_evaluation_batch(W05TrainingPayload(
            tuple(item.record for item in overlay.w05_batch.source_records),
            observations,
            tuple(item.evidence for item in overlay.w05_batch.pairs),
        ))

    result = _run(
        overlay,
        catalog,
        "什么使得河水上涨？",
        w04=w04,
        w05=w05,
    )

    assert result.status == "UNKNOWN"
    assert result.selected_construction is not None
    assert result.typed_result is not None
    assert result.typed_result.vertical_result.status == "UNKNOWN"


def test_raw_question_is_repeatable_and_learning_records_are_read_only(
        overlay, catalog) -> None:
    before = tuple(
        batch.record_commitment
        for batch in (overlay.w03_batch, overlay.w04_batch, overlay.w05_batch))

    first = _run(overlay, catalog, "什么使得河水上涨？")
    second = _run(overlay, catalog, "什么使得河水上涨？")

    assert first.sha256() == second.sha256()
    assert before == tuple(
        batch.record_commitment
        for batch in (overlay.w03_batch, overlay.w04_batch, overlay.w05_batch))
