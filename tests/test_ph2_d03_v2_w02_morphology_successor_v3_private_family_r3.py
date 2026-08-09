"""Public tests for the successor V3 R3 private family and guard."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    read_canonical_object,
    write_immutable_json,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_private_evaluator_r3 import (
    W02_MORPH_V3_PRIVATE_R3_EVALUATOR_VERSION,
    W02_MORPH_V3_PRIVATE_SUPPORT_KEYS,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_private_family_r3 import (
    W02_MORPH_V3_PRIVATE_R3_FAMILY_CODE_PATHS,
    W02_MORPH_V3_PRIVATE_R3_FORMAL_FAMILY_NAME,
    W02_MORPH_V3_PRIVATE_R3_GUARD_AVAILABLE,
    W02MorphologySuccessorV3PrivateR3FamilyError,
    build_w02_morphology_successor_v3_private_r3_family_freeze,
    consume_w02_morphology_successor_v3_private_r3_guard,
    verify_w02_morphology_successor_v3_private_r3_consumed_guard,
    w02_morphology_successor_v3_private_r3_guard_value,
    w02_morphology_successor_v3_private_r3_registration_from_freeze,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_private_publication_r3 import (
    W02_MORPH_V3_PRIVATE_R3_ZERO_CALL_WINDOWS,
    W02_MORPH_V3_PRIVATE_R3_ZERO_WRITE_KEYS,
    _validate_formal_pass_report,
)
from pure_integer_ai.experiments.run_ph2_d03_v2_w02_morphology_successor_v3_private_evaluation_r3 import (
    _failure_seal,
)


def _repository() -> Path:
    return Path(__file__).resolve().parents[1]


def _result(key: str, denominator: int) -> dict[str, object]:
    return {
        "denominator": denominator,
        "dimension_key": key,
        "evidence_sha256": hashlib.sha256(key.encode("utf-8")).hexdigest(),
        "failed": 0,
        "ne": 0,
        "numerator": denominator,
        "status": "PASS",
    }


def _formal_pass_report(
        freeze: dict[str, object], family_freeze_sha256: str,
        ) -> dict[str, object]:
    dimensions = [
        _result(key, denominator)
        for key, denominator in freeze["dimension_denominator_counts"].items()
    ]
    support = [_result(key, 1) for key in W02_MORPH_V3_PRIVATE_SUPPORT_KEYS]
    chain = freeze["artifact_chain"]
    transport_bytes = sum(
        row["transport_size_bytes"] for row in freeze["owner_input_files"])
    pair_count = freeze["owner_pair_count"]
    return {
        "artifact_kind": (
            "PH2_D03_V2_W02_MORPHOLOGY_SUCCESSOR_V3_PRIVATE_R3_"
            "EVALUATION_REPORT"),
        "artifact_version": W02_MORPH_V3_PRIVATE_R3_EVALUATOR_VERSION,
        "candidate_artifact_manifest_sha256":
            chain["candidate_artifact_manifest_sha256"],
        "candidate_semantic_sha256": chain["candidate_semantic_sha256"],
        "dimension_results": dimensions,
        "evaluation_count": pair_count,
        "family_commitment": freeze["registration"]["family_commitment"],
        "family_counts": {
            "AUTHORED_OOV": 0,
            "UD_ANNOTATION": pair_count,
            "UNICODE_ANNOTATION": 0,
        },
        "family_freeze_sha256": family_freeze_sha256,
        "formal_dev_calibration_runs": 1,
        "formal_private_evaluation_runs": 1,
        "formal_shadow_audit_runs": 1,
        "formal_successor_transform_runs": 1,
        "formal_successor_v2_transform_runs": 1,
        "formal_successor_v3_route_dev_runs": 1,
        "formal_successor_v3_route_shadow_runs": 1,
        "formal_training_runs": 1,
        "hard_conjunct_results": [*dimensions, *support],
        "input_pair_count": pair_count,
        "label_record_reads": pair_count,
        "label_sanitization_count": 0,
        "language_capability_mastered": 0,
        "language_readiness": 0,
        "logic_operations": 1,
        "max_v2_edge_candidates_per_requested_span": 8,
        "next_action": "W03_COMPILE_FREEZE",
        "observation_reads": pair_count,
        "private_content_stream_reads": 7,
        "private_family_registered": 1,
        "private_payload_gets": freeze["owner_source_count"] + pair_count * 2,
        "private_payload_reads": 28,
        "private_post_content_transport_reads": 7,
        "private_record_reads": freeze["owner_source_count"] + pair_count * 2,
        "private_transport_validation_reads": 14,
        "release_key": "PH2-D03-V2",
        "route_authorized_count": pair_count,
        "route_capability_sha256s": ["d" * 64],
        "route_semantic_sha256": "e" * 64,
        "routed_index_semantic_sha256": "f" * 64,
        "run_id": 1,
        "run_scope": "FORMAL_BLIND_PRIVATE_EVALUATION",
        "source_count": freeze["owner_source_count"],
        "source_identity_sha256": "a" * 64,
        "stage_key": "W-02",
        "status": "PASS",
        "support_results": support,
        "teacher_calls": 0,
        "transport_bytes_read": transport_bytes * 4,
        "unregistered_family_count": 0,
        "unregistered_label_result": {
            "denominator": 0,
            "dimension_key": "W-02-V2-UNREGISTERED-LABEL",
            "evidence_sha256": hashlib.sha256(b"[]").hexdigest(),
            "failed": 0,
            "ne": 0,
            "numerator": 0,
            "status": "PASS",
        },
        "unknown_dimension_key_count": 0,
        "unknown_expected_family_count": 0,
        "unknown_expected_state_count": 0,
        "v1_overlay_artifact_manifest_sha256":
            chain["v1_overlay_artifact_manifest_sha256"],
        "v1_overlay_semantic_sha256": chain["v1_overlay_semantic_sha256"],
        "v2_overlay_artifact_manifest_sha256":
            chain["v2_overlay_artifact_manifest_sha256"],
        "v2_overlay_semantic_sha256": chain["v2_overlay_semantic_sha256"],
        "validated_layout_count": 7,
        "zero_call_windows": [
            {"api_calls": 0, "llm_calls": 0, "teacher_calls": 0,
             "window_key": key}
            for key in W02_MORPH_V3_PRIVATE_R3_ZERO_CALL_WINDOWS
        ],
        "zero_write_audit": {
            key: 0 for key in W02_MORPH_V3_PRIVATE_R3_ZERO_WRITE_KEYS
        },
    }


def test_r3_family_freeze_is_metadata_only_and_binds_v4_owner() -> None:
    value = build_w02_morphology_successor_v3_private_r3_family_freeze(
        _repository())
    assert value["status"] == (
        "W02_SUCCESSOR_V3_R3_BLIND_PRIVATE_FAMILY_FROZEN")
    assert value["formal_private_evaluation_runs"] == 0
    assert value["private_payload_reads"] == 0
    assert value["private_family_registered"] == 1
    assert value["previous_consumed_r2_reuse_authorized"] == 0
    assert value["owner_pair_count"] == 500
    assert value["owner_source_count"] == 500
    assert len(value["owner_input_files"]) == 7
    assert all(row["license_ids"] == ["CC-BY-SA-3.0"]
               for row in value["owner_input_files"])
    assert tuple(row["repository_file"] for row in value["code_files"]) == (
        W02_MORPH_V3_PRIVATE_R3_FAMILY_CODE_PATHS)
    assert w02_morphology_successor_v3_private_r3_registration_from_freeze(
        value).to_dict() == value["registration"]


def test_r3_guard_commits_intent_and_is_single_use(tmp_path: Path) -> None:
    freeze = build_w02_morphology_successor_v3_private_r3_family_freeze(
        _repository())
    root = tmp_path / W02_MORPH_V3_PRIVATE_R3_FORMAL_FAMILY_NAME
    root.mkdir()
    guard = w02_morphology_successor_v3_private_r3_guard_value(
        family_commitment=freeze["registration"]["family_commitment"],
        artifact_chain_sha256=freeze["artifact_chain_sha256"],
        code_freeze_sha256=freeze["code_freeze_sha256"],
    )
    write_immutable_json(
        guard,
        root / Path(*W02_MORPH_V3_PRIVATE_R3_GUARD_AVAILABLE.split("/")))
    run_identity = "b" * 64
    consume_w02_morphology_successor_v3_private_r3_guard(
        root,
        expected_guard_sha256=freeze["first_run_guard_sha256"],
        run_identity_sha256=run_identity,
    )
    verify_w02_morphology_successor_v3_private_r3_consumed_guard(
        root,
        expected_guard_sha256=freeze["first_run_guard_sha256"],
        run_identity_sha256=run_identity,
    )
    with pytest.raises(
            W02MorphologySuccessorV3PrivateR3FamilyError,
            match="already consumed"):
        consume_w02_morphology_successor_v3_private_r3_guard(
            root,
            expected_guard_sha256=freeze["first_run_guard_sha256"],
            run_identity_sha256=run_identity,
        )


def test_r3_pass_projection_rejects_ne_or_label_sanitization() -> None:
    freeze = build_w02_morphology_successor_v3_private_r3_family_freeze(
        _repository())
    report = _formal_pass_report(freeze, "c" * 64)
    _validate_formal_pass_report(report, freeze, "c" * 64)

    report["label_sanitization_count"] = 1
    report["unknown_dimension_key_count"] = 1
    with pytest.raises(
            W02MorphologySuccessorV3PrivateR3FamilyError,
            match="identity"):
        _validate_formal_pass_report(report, freeze, "c" * 64)


def test_r3_runner_failure_seal_is_append_only(tmp_path: Path) -> None:
    publication = tmp_path / "publication"
    publication.mkdir()
    error = RuntimeError("synthetic-r3-private-runner-failure")
    _failure_seal(publication, error, phase="PRE_GUARD")
    target = publication / "run-000001.failure.json"
    before = hashlib.sha256(target.read_bytes()).hexdigest()
    value = read_canonical_object(target)
    assert value["formal_private_evaluation_runs"] == 0
    assert value["private_payload_reads_minimum"] == 0
    assert "synthetic-r3-private-runner-failure" not in target.read_text("utf-8")
    _failure_seal(publication, RuntimeError("different"), phase="PRE_GUARD")
    assert hashlib.sha256(target.read_bytes()).hexdigest() == before
