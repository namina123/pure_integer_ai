"""FT14 已学三 Role 问题构造与严格未知边界。"""
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
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question_contract import (
    RawQuestionRequest,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question_three_role import (
    THREE_ROLE_ACTOR_QUESTION_SAMPLE_SHA256,
    THREE_ROLE_LOCATION_QUESTION_SAMPLE_SHA256,
    THREE_ROLE_QUESTION_BUNDLE_SHA256,
    THREE_ROLE_QUESTION_EXPRESSION_BOUNDARY,
    THREE_ROLE_QUESTION_VERTICAL_SHA256S,
    build_three_role_question_bundle,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_vertical import (
    run_w03_w04_w05_vertical_query,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_vertical_contract import (
    W03W04W05VerticalQuery,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_vertical_three_role import (
    THREE_ROLE_BASE_MANIFEST_SHA256,
    THREE_ROLE_BASE_SAMPLE_SHA256,
    THREE_ROLE_DONOR_ATOMIC_SHA256,
    THREE_ROLE_DONOR_MANIFEST_SHA256,
    THREE_ROLE_DONOR_MAP_SHA256,
    THREE_ROLE_VERTICAL_TARGETS,
    VERTICAL_THREE_ROLE_OVERLAY_VALIDATION_SHA256,
    build_w03_w04_w05_three_role_vertical_overlay,
)
from pure_integer_ai.experiments.ph2_w05_payload import W05TrainingPayload
from pure_integer_ai.experiments.ph2_w05_v2_public_source import (
    build_w05_v2_public_evaluation_batch,
)


REPOSITORY = Path(__file__).resolve().parents[1]
SEMANTIC_SAMPLE = (
    REPOSITORY
    / "data/ph2/authored_semantic_primitive_bridge_three_role_v1.jsonl.sample"
)
PRIMITIVE_MAP_SAMPLE = (
    REPOSITORY
    / "data/ph2/authored_primitive_atomic_bridge_map_three_role_v1.jsonl.sample"
)
ATOMIC_SAMPLE = (
    REPOSITORY
    / "data/ph2/authored_primitive_atomic_bridge_seed_three_role_v1.jsonl.sample"
)
ACTOR_QUESTION_SAMPLE = (
    REPOSITORY
    / "data/ph2/authored_vertical_question_three_role_actor_v1.jsonl.sample"
)
LOCATION_QUESTION_SAMPLE = (
    REPOSITORY
    / "data/ph2/authored_vertical_question_three_role_location_v1.jsonl.sample"
)


@pytest.fixture(scope="module")
def overlay(tmp_path_factory):
    root = tmp_path_factory.mktemp("ft14_three_role")
    base = compile_authored_semantic_primitive_bridge_course(
        SEMANTIC_SAMPLE,
        root / "base",
    )
    donor = compile_authored_primitive_atomic_bridge_course(
        PRIMITIVE_MAP_SAMPLE,
        ATOMIC_SAMPLE,
        root / "donor",
    )
    return build_w03_w04_w05_three_role_vertical_overlay(base, donor)


@pytest.fixture(scope="module")
def bundle(overlay):
    return build_three_role_question_bundle(
        overlay,
        ACTOR_QUESTION_SAMPLE,
        LOCATION_QUESTION_SAMPLE,
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


@pytest.mark.parametrize(
    ("path", "expected_sha", "forbidden_answer"),
    (
        (
            ACTOR_QUESTION_SAMPLE,
            THREE_ROLE_ACTOR_QUESTION_SAMPLE_SHA256,
            "暴雨",
        ),
        (
            LOCATION_QUESTION_SAMPLE,
            THREE_ROLE_LOCATION_QUESTION_SAMPLE_SHA256,
            "山区",
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
        for key in keys
    )


def test_overlay_freezes_complete_three_role_vertical_dependencies(
        overlay) -> None:
    assert overlay.validation_sha256 == (
        VERTICAL_THREE_ROLE_OVERLAY_VALIDATION_SHA256)
    assert (
        overlay.projection.base_sample_sha256,
        overlay.projection.base_manifest_sha256,
        overlay.projection.donor_map_sha256,
        overlay.projection.donor_atomic_sha256,
        overlay.projection.donor_manifest_sha256,
    ) == (
        THREE_ROLE_BASE_SAMPLE_SHA256,
        THREE_ROLE_BASE_MANIFEST_SHA256,
        THREE_ROLE_DONOR_MAP_SHA256,
        THREE_ROLE_DONOR_ATOMIC_SHA256,
        THREE_ROLE_DONOR_MANIFEST_SHA256,
    )
    assert (
        len(overlay.w03_batch.pairs),
        len(overlay.w04_batch.pairs),
        len(overlay.w05_batch.pairs),
    ) == (10, 10, 12)
    assert len(overlay.targets) == 2
    for item in overlay.targets:
        payload = item.overlay_w05_observation.typed_payload.to_value()
        assert item.base_w04_observation.prerequisite_keys == (
            item.base_w03_observation.stable_key,)
        assert item.overlay_w05_observation.prerequisite_keys == (
            item.base_w04_observation.stable_key,)
        assert len(payload["candidate_definition"]["role_bindings"]) == 3
        assert len(payload["occurrences"]) == 4


def test_vertical_results_retain_generation_and_all_role_bindings(
        overlay, bundle) -> None:
    assert tuple(
        item.sha256() for item in bundle.vertical_results
    ) == THREE_ROLE_QUESTION_VERTICAL_SHA256S
    for result in bundle.vertical_results:
        w05 = result.w04_w05.w05_result
        assert result.status == "BRIDGED"
        assert w05.generation_status == "READY"
        assert len(w05.generation_options) == 2
        candidate = next(
            item for item in w05.candidates
            if item.proposition_key == result.link.proposition_key
        )
        assert len(candidate.role_bindings) == 3
        assert len(candidate.occurrences) == 4
        assert all(
            option.role_binding_keys
            == tuple(item.identity_key for item in candidate.role_bindings)
            for option in w05.generation_options
        )


def test_bundle_is_two_constructions_by_two_contents_and_roles(bundle) -> None:
    assert bundle.identity_sha256 == THREE_ROLE_QUESTION_BUNDLE_SHA256
    assert len(bundle.patterns) == 2
    assert len(bundle.catalog) == 4
    assert len({item.pattern.sha256() for item in bundle.catalog}) == 2
    assert len({item.source_record_key for item in bundle.catalog}) == 2
    assert len({item.target_role_key for item in bundle.catalog}) == 2
    assert len({
        (item.pattern.sha256(), item.source_record_key)
        for item in bundle.catalog
    }) == 4
    assert THREE_ROLE_QUESTION_EXPRESSION_BOUNDARY[-1] == (
        "four_role_or_more",
        "UNKNOWN_UNTIL_PUBLIC_VERTICAL_DEPENDENCIES_EXIST",
    )


def test_construction_and_content_replacement_answer_two_target_roles(
        overlay, bundle) -> None:
    answers = {
        construction.question_surface: _run(
            overlay,
            bundle,
            construction.question_surface,
            source=construction.source_record_key,
        )
        for construction in bundle.catalog
    }

    assert {
        question: (result.status, result.answer_surface)
        for question, result in answers.items()
    } == {
        "什么在山区使得河水上涨？": ("ANSWER", "暴雨"),
        "暴雨在哪里使得河水上涨？": ("ANSWER", "山区"),
        "什么在桥面使得路面结冰？": ("ANSWER", "寒潮"),
        "寒潮在哪里使得路面结冰？": ("ANSWER", "桥面"),
    }
    assert all(
        result.typed_result is not None
        and result.typed_result.proof is not None
        and result.typed_result.state_before_sha256
        == result.typed_result.state_after_sha256
        for result in answers.values()
    )


def test_wrong_source_and_unlearned_four_role_surface_remain_unknown(
        overlay, bundle) -> None:
    construction = bundle.catalog[0]
    wrong_source = next(
        item.source_record_key for item in bundle.catalog
        if item.source_record_key != construction.source_record_key
    )

    wrong = _run(
        overlay,
        bundle,
        construction.question_surface,
        source=wrong_source,
    )
    unlearned = _run(
        overlay,
        bundle,
        "谁在何时于何地使得什么发生？",
    )

    assert wrong.status == "UNKNOWN"
    assert unlearned.status == "UNKNOWN"
    assert wrong.typed_result is None
    assert unlearned.typed_result is None


def test_missing_w04_to_w05_prerequisite_keeps_vertical_unknown(
        overlay) -> None:
    spec = THREE_ROLE_VERTICAL_TARGETS[0]
    target = overlay.target(spec)
    observations = tuple(
        replace(item.observation, prerequisite_keys=())
        if item.observation.stable_key
        == target.overlay_w05_observation.stable_key
        else item.observation
        for item in overlay.w05_batch.pairs
    )
    broken = build_w05_v2_public_evaluation_batch(W05TrainingPayload(
        tuple(item.record for item in overlay.w05_batch.source_records),
        observations,
        tuple(item.evidence for item in overlay.w05_batch.pairs),
    ))
    result = run_w03_w04_w05_vertical_query(
        overlay.w03_batch,
        overlay.w04_batch,
        broken,
        W03W04W05VerticalQuery(
            spec.surface,
            spec.context,
            spec.proposition_surface,
        ),
        overlay_validation_sha256=overlay.validation_sha256,
    )

    assert result.status == "UNKNOWN"
    assert result.link is None
    assert result.w03_w04.status == "BRIDGED"
    assert result.w04_w05.status == "UNKNOWN"


def test_repeated_answer_is_identity_stable(overlay, bundle) -> None:
    construction = bundle.catalog[0]
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

    assert first.sha256() == second.sha256()
    assert first.typed_result.state_before_sha256 == (
        second.typed_result.state_after_sha256)
