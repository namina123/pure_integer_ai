"""Frozen W-03/W-04 base plus strict W-05 vertical overlay tests."""
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
from pure_integer_ai.experiments.ph2_w03_w04_w05_vertical import (
    run_w03_w04_w05_vertical_query,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_vertical_contract import (
    W03W04W05VerticalError,
    W03W04W05VerticalQuery,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_vertical_overlay import (
    VERTICAL_CONTEXT,
    VERTICAL_OVERLAY_VALIDATION_SHA256,
    VERTICAL_PROPOSITION_SURFACE,
    VERTICAL_SURFACE,
    W03_W04_W05_VERTICAL_RESULT_SHA256,
    W03_W04_BASE_BRIDGE_SHA256,
    W03_W04_BASE_MANIFEST_SHA256,
    W03_W04_BASE_SAMPLE_SHA256,
    W04_W05_DONOR_ATOMIC_SHA256,
    W04_W05_DONOR_MANIFEST_SHA256,
    W04_W05_DONOR_MAP_SHA256,
    W04_W05_OVERLAY_BRIDGE_SHA256,
    build_w03_w04_w05_vertical_overlay,
)
from pure_integer_ai.experiments.ph2_w04_payload import W04TrainingPayload
from pure_integer_ai.experiments.ph2_w04_v2_public_source import (
    build_w04_v2_public_evaluation_batch,
)
from pure_integer_ai.experiments.ph2_w05_payload import W05TrainingPayload
from pure_integer_ai.experiments.ph2_w05_v2_public_plugin import (
    build_w05_v2_public_capability_plugin,
)
from pure_integer_ai.experiments.ph2_w05_v2_public_preflight import (
    build_w05_v2_public_preflight,
)
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


def _overlay(tmp_path):
    base = compile_authored_semantic_primitive_bridge_course(
        SEMANTIC_PRIMITIVE_SAMPLE, tmp_path / "base")
    donor = compile_authored_primitive_atomic_bridge_course(
        PRIMITIVE_MAP_SAMPLE, ATOMIC_SAMPLE, tmp_path / "donor")
    return build_w03_w04_w05_vertical_overlay(base, donor)


def _query() -> W03W04W05VerticalQuery:
    return W03W04W05VerticalQuery(
        VERTICAL_SURFACE,
        VERTICAL_CONTEXT,
        VERTICAL_PROPOSITION_SURFACE,
    )


def test_overlay_freezes_both_inputs_and_preserves_six_w05_structures(
        tmp_path) -> None:
    overlay = _overlay(tmp_path)

    assert overlay.base_sample_sha256 == W03_W04_BASE_SAMPLE_SHA256
    assert overlay.base_manifest_sha256 == W03_W04_BASE_MANIFEST_SHA256
    assert overlay.donor_map_sha256 == W04_W05_DONOR_MAP_SHA256
    assert overlay.donor_atomic_sha256 == W04_W05_DONOR_ATOMIC_SHA256
    assert overlay.donor_manifest_sha256 == W04_W05_DONOR_MANIFEST_SHA256
    assert overlay.validation_sha256 == VERTICAL_OVERLAY_VALIDATION_SHA256
    assert len(overlay.w03_batch.pairs) == 4
    assert len(overlay.w04_batch.pairs) == 4
    assert len(overlay.w05_batch.pairs) == 6
    assert len(overlay.w05_batch.source_records) == 6
    assert {item.observation.perturbation_kind for item in overlay.w05_batch.pairs} == {
        "NONE",
        "ROLE_SWAP",
        "ORDER_REVERSAL",
        "SCOPE_SHIFT",
        "OCCURRENCE_OMISSION",
        "OCCURRENCE_RESTORE",
    }


def test_overlay_target_uses_old_source_and_two_exact_prerequisites(
        tmp_path) -> None:
    overlay = _overlay(tmp_path)
    w03 = overlay.base_w03_observation
    w04 = overlay.base_w04_observation
    w05 = overlay.overlay_w05_observation

    assert w04.prerequisite_keys == (w03.stable_key,)
    assert w05.prerequisite_keys == (w04.stable_key,)
    assert w03.source_ref_key == w04.source_ref_key == w05.source_ref_key
    assert w03.logical_order < w04.logical_order < w05.logical_order
    assert w05.typed_payload.to_value()["surface"] == VERTICAL_CONTEXT
    assert len(overlay.dependency_w04_observations) == 6
    assert all("expected_state" not in item.observation.typed_payload.to_value()
               for item in overlay.w05_batch.pairs)
    assert all("expected_payload" not in item.observation.typed_payload.to_value()
               for item in overlay.w05_batch.pairs)


def test_overlay_retains_w05_public_p0_p2_and_builds_one_vertical_link(
        tmp_path) -> None:
    overlay = _overlay(tmp_path)
    plugin = build_w05_v2_public_capability_plugin(REPOSITORY)
    preflight = build_w05_v2_public_preflight(
        REPOSITORY, overlay.w05_batch, plugin)

    assert (preflight.p0.status, preflight.p1.status,
            preflight.p2.status) == ("PASS", "PASS", "PASS")
    assert preflight.outcome.result_set.status == "PASS"
    result = run_w03_w04_w05_vertical_query(
        overlay.w03_batch,
        overlay.w04_batch,
        overlay.w05_batch,
        _query(),
        overlay_validation_sha256=overlay.validation_sha256,
    )

    assert result.status == "BRIDGED"
    assert result.link is not None
    assert result.w03_w04.status == result.w04_w05.status == "BRIDGED"
    assert result.link.source_ref_key == (
        overlay.base_w03_observation.source_ref_key.stable_key())
    assert result.link.w04_observation_key == (
        overlay.base_w04_observation.stable_key.stable_key())
    assert result.link.w05_observation_key == (
        overlay.overlay_w05_observation.stable_key.stable_key())
    assert (result.link.primitive_registry,
            result.link.primitive_kind) == ("relation", 4)
    assert result.w03_w04.sha256() == W03_W04_BASE_BRIDGE_SHA256
    assert result.w04_w05.sha256() == W04_W05_OVERLAY_BRIDGE_SHA256
    assert result.sha256() == W03_W04_W05_VERTICAL_RESULT_SHA256


def test_removing_w03_to_w04_prerequisite_fails_total_chain_closed(
        tmp_path) -> None:
    overlay = _overlay(tmp_path)
    broken_observations = tuple(
        replace(item.observation, prerequisite_keys=())
        if item.observation.stable_key == overlay.base_w04_observation.stable_key
        else item.observation
        for item in overlay.w04_batch.pairs
    )
    broken_w04 = build_w04_v2_public_evaluation_batch(W04TrainingPayload(
        tuple(item.record for item in overlay.w04_batch.source_records),
        broken_observations,
        tuple(item.evidence for item in overlay.w04_batch.pairs),
    ))

    result = run_w03_w04_w05_vertical_query(
        overlay.w03_batch,
        broken_w04,
        overlay.w05_batch,
        _query(),
        overlay_validation_sha256=overlay.validation_sha256,
    )

    assert result.status == "UNKNOWN"
    assert result.link is None
    assert result.w03_w04.status == "UNKNOWN"
    assert result.w04_w05.status == "BRIDGED"
    assert result.w03_w04.w03_result.status == "UNIQUE"
    assert result.w03_w04.w04_result.status == "UNIQUE"


def test_removing_w04_to_w05_prerequisite_fails_total_chain_closed(
        tmp_path) -> None:
    overlay = _overlay(tmp_path)
    broken_observations = tuple(
        replace(item.observation, prerequisite_keys=())
        if item.observation.stable_key == overlay.overlay_w05_observation.stable_key
        else item.observation
        for item in overlay.w05_batch.pairs
    )
    broken_w05 = build_w05_v2_public_evaluation_batch(W05TrainingPayload(
        tuple(item.record for item in overlay.w05_batch.source_records),
        broken_observations,
        tuple(item.evidence for item in overlay.w05_batch.pairs),
    ))

    result = run_w03_w04_w05_vertical_query(
        overlay.w03_batch,
        overlay.w04_batch,
        broken_w05,
        _query(),
        overlay_validation_sha256=overlay.validation_sha256,
    )

    assert result.status == "UNKNOWN"
    assert result.link is None
    assert result.w03_w04.status == "BRIDGED"
    assert result.w04_w05.status == "UNKNOWN"
    assert result.w04_w05.w04_result.status == "UNIQUE"
    assert result.w04_w05.w05_result.status == "UNIQUE"


def test_vertical_result_rejects_unfrozen_overlay_commitment(tmp_path) -> None:
    overlay = _overlay(tmp_path)

    with pytest.raises(
            W03W04W05VerticalError,
            match="not bound to the frozen overlay"):
        run_w03_w04_w05_vertical_query(
            overlay.w03_batch,
            overlay.w04_batch,
            overlay.w05_batch,
            _query(),
            overlay_validation_sha256="0" * 64,
        )
