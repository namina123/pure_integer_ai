"""R3 revision of the successor V3 blind private evaluator."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Iterable

from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    EvaluatorLabelRecord,
    ObservationRecord,
    SourceRefRecord,
)
from pure_integer_ai.experiments.ph2_d03_v2_blind_private_source_extension_v4 import (
    blind_private_source_specs_v4,
)
from pure_integer_ai.experiments.ph2_d03_v2_evaluator_contract import (
    V2EvaluatorBoundaryContract,
    V2EvaluatorResourceBudget,
    V2PrivateFamilyRegistration,
    validate_v2_safe_report,
)
from pure_integer_ai.experiments.ph2_d03_v2_evaluator_firewall import (
    V2PhysicalRoots,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_compiler import _dimension_key
from pure_integer_ai.experiments.ph2_d03_v2_w02_dev_calibration import (
    W02_DEV_DIMENSIONS,
    W02DevCalibrationError,
    _expected_family,
    _hash_value,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_dev_calibration import (
    W02MorphologySuccessorDevCalibrationError,
    _requested_spans,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_private_evaluator import (
    W02_MORPH_V3_PRIVATE_SUPPORT_KEYS,
    evaluate_w02_morphology_successor_v3_private_pair_stream,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_private_io import (
    authorize_w02_morphology_successor_v3_private_files,
    iter_w02_morphology_successor_v3_private_pairs,
    iter_w02_morphology_successor_v3_private_records,
    v3_private_file_by_layout,
    v3_private_split_layout,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_private_owner_contract import (
    W02_MORPH_V3_PRIVATE_SPLITS,
    W02MorphologySuccessorV3PrivateFileIdentity,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_route import (
    W02MorphologySourceCapability,
    w02_ud_morphology_source_capability,
)


W02_MORPH_V3_PRIVATE_R3_EVALUATOR_VERSION = (
    "PH2-D03-V2-W02-MORPHOLOGY-SUCCESSOR-V3-PRIVATE-EVALUATOR-R3-V1"
)
_FALLBACK_EXPECTED_PAYLOAD = CanonicalJsonObject.from_value({
    "oov_boundaries": [],
    "oov_units": [],
})


# object-model: exception
class W02MorphologySuccessorV3PrivateR3EvaluationError(RuntimeError):
    """The R3 source route, aggregate, or resource contract drifted."""


def _prepare_label_stream(
        pairs: Iterable[tuple[ObservationRecord, EvaluatorLabelRecord]],
        ) -> tuple[
            tuple[tuple[ObservationRecord, EvaluatorLabelRecord], ...],
            dict[str, object],
        ]:
    """Sanitize only unregistered labels and permanently force their NE."""
    dimension_by_key = {
        _dimension_key(name).components: name for name in W02_DEV_DIMENSIONS
    }
    first_dimension_key = _dimension_key(W02_DEV_DIMENSIONS[0])
    prepared: list[tuple[ObservationRecord, EvaluatorLabelRecord]] = []
    original_counts = {name: 0 for name in W02_DEV_DIMENSIONS}
    valid_family_counts = {name: 0 for name in (
        "AUTHORED_OOV", "UD_ANNOTATION", "UNICODE_ANNOTATION")}
    affected_dimensions: set[str] = set()
    evidence: list[str] = []
    unknown_dimension = 0
    unknown_state = 0
    unknown_family = 0
    sanitization_count = 0
    for observation, evaluation in pairs:
        if (not isinstance(observation, ObservationRecord)
                or not isinstance(evaluation, EvaluatorLabelRecord)):
            raise W02MorphologySuccessorV3PrivateR3EvaluationError(
                "R3 private pair type drifted")
        dimension = dimension_by_key.get(evaluation.dimension_key.components)
        if dimension is not None:
            original_counts[dimension] += 1
        state_ok = evaluation.expected_state == "TRUE"
        family = None
        payload_ok = True
        try:
            family = _expected_family(evaluation.expected_payload.to_value())
            _requested_spans(evaluation)
        except (W02DevCalibrationError,
                W02MorphologySuccessorDevCalibrationError,
                KeyError, TypeError, ValueError):
            payload_ok = False
        registered = dimension is not None and state_ok and payload_ok
        if registered:
            assert family is not None
            valid_family_counts[family] += 1
            prepared.append((observation, evaluation))
            continue

        sanitization_count += 1
        unknown_dimension += int(dimension is None)
        unknown_state += int(not state_ok)
        unknown_family += int(not payload_ok)
        affected = dimension or W02_DEV_DIMENSIONS[0]
        affected_dimensions.add(affected)
        evidence.append(_hash_value(evaluation.to_dict()))
        prepared.append((observation, replace(
            evaluation,
            dimension_key=(
                evaluation.dimension_key if dimension is not None
                else first_dimension_key),
            expected_state="TRUE",
            expected_payload=(
                evaluation.expected_payload if payload_ok
                else _FALLBACK_EXPECTED_PAYLOAD),
        )))
    return tuple(prepared), {
        "affected_dimensions": tuple(sorted(affected_dimensions)),
        "evidence": tuple(evidence),
        "original_dimension_counts": original_counts,
        "sanitization_count": sanitization_count,
        "unknown_dimension_key_count": unknown_dimension,
        "unknown_expected_family_count": unknown_family,
        "unknown_expected_state_count": unknown_state,
        "valid_family_counts": valid_family_counts,
    }


def _apply_label_fail_closed(
        core: dict[str, object], audit: dict[str, object],
        ) -> dict[str, object]:
    sanitization_count = int(audit["sanitization_count"])
    evidence = list(audit["evidence"])
    unregistered = {
        "denominator": sanitization_count,
        "dimension_key": "W-02-V2-UNREGISTERED-LABEL",
        "evidence_sha256": _hash_value(evidence),
        "failed": 0,
        "ne": sanitization_count,
        "numerator": 0,
        "status": "NE" if sanitization_count else "PASS",
    }
    result = dict(core)
    result.update({
        "label_sanitization_count": sanitization_count,
        "unknown_dimension_key_count": audit["unknown_dimension_key_count"],
        "unknown_expected_family_count": audit["unknown_expected_family_count"],
        "unknown_expected_state_count": audit["unknown_expected_state_count"],
        "unregistered_label_result": unregistered,
        "unregistered_family_count": sanitization_count,
    })
    if not sanitization_count:
        return result

    affected = set(audit["affected_dimensions"])
    counts = audit["original_dimension_counts"]
    dimensions = []
    for row in core["dimension_results"]:
        current = dict(row)
        name = str(current["dimension_key"])
        if name in affected:
            denominator = int(counts[name])
            current.update({
                "denominator": denominator,
                "evidence_sha256": _hash_value({
                    "dimension": name,
                    "unregistered_label_evidence": evidence,
                }),
                "failed": 0,
                "ne": denominator,
                "numerator": 0,
                "status": "NE",
            })
        dimensions.append(current)
    support = list(core["support_results"])
    hard = [*dimensions, *support]
    result.update({
        "dimension_results": dimensions,
        "family_counts": dict(audit["valid_family_counts"]),
        "hard_conjunct_results": hard,
        "status": (
            "FAIL" if any(row["status"] == "FAIL" for row in hard)
            else "NE"),
    })
    return result


def evaluate_w02_morphology_successor_v3_private_r3_pair_stream(
        candidate_artifact_root: str | Path,
        v1_overlay_artifact_root: str | Path,
        v2_overlay_artifact_root: str | Path,
        sources: Iterable[SourceRefRecord],
        capabilities: Iterable[W02MorphologySourceCapability],
        pairs: Iterable[tuple[ObservationRecord, EvaluatorLabelRecord]],
        budget: V2EvaluatorResourceBudget,
        ) -> dict[str, object]:
    """Evaluate all pairs; malformed label registration becomes aggregate NE."""
    prepared, audit = _prepare_label_stream(pairs)
    core = evaluate_w02_morphology_successor_v3_private_pair_stream(
        candidate_artifact_root,
        v1_overlay_artifact_root,
        v2_overlay_artifact_root,
        sources,
        capabilities,
        prepared,
        budget,
    )
    return _apply_label_fail_closed(core, audit)


def run_w02_morphology_successor_v3_private_r3_evaluation(
        boundary: V2EvaluatorBoundaryContract,
        roots: V2PhysicalRoots,
        registration: V2PrivateFamilyRegistration,
        files: tuple[W02MorphologySuccessorV3PrivateFileIdentity, ...],
        candidate_artifact_root: str | Path,
        v1_overlay_artifact_root: str | Path,
        v2_overlay_artifact_root: str | Path,
        *,
        family_freeze_sha256: str,
        run_id: int = 1,
        ) -> dict[str, object]:
    """Consume an already-guarded R3 family in one full formal stream."""
    if run_id != 1:
        raise W02MorphologySuccessorV3PrivateR3EvaluationError(
            "formal R3 private run_id must be one")
    if (not isinstance(registration, V2PrivateFamilyRegistration)
            or registration.stage_key != "W-02"
            or registration.formal_run_count != 0
            or registration.private_payload_reads != 0):
        raise W02MorphologySuccessorV3PrivateR3EvaluationError(
            "R3 private registration state drifted")
    permits = authorize_w02_morphology_successor_v3_private_files(
        boundary, roots, registration, files)
    sources = iter_w02_morphology_successor_v3_private_records(
        v3_private_file_by_layout(files, "PRIVATE_SOURCE"),
        permits["PRIVATE_SOURCE"])
    pairs = (
        pair for split in W02_MORPH_V3_PRIVATE_SPLITS
        for pair in iter_w02_morphology_successor_v3_private_pairs(
            files, permits, split)
    )
    capabilities = tuple(
        w02_ud_morphology_source_capability(spec)
        for spec in blind_private_source_specs_v4())
    core = evaluate_w02_morphology_successor_v3_private_r3_pair_stream(
        candidate_artifact_root,
        v1_overlay_artifact_root,
        v2_overlay_artifact_root,
        sources,
        capabilities,
        pairs,
        registration.resource_budget,
    )
    authorize_w02_morphology_successor_v3_private_files(
        boundary, roots, registration, files)
    expected = sum(
        v3_private_file_by_layout(
            files, v3_private_split_layout(split, "observation")).record_count
        for split in W02_MORPH_V3_PRIVATE_SPLITS)
    if core["input_pair_count"] != expected:
        raise W02MorphologySuccessorV3PrivateR3EvaluationError(
            "R3 private input was not fully consumed")
    transport_bytes = sum(row.transport_size_bytes for row in files)
    private_record_reads = core["source_count"] + core["input_pair_count"] * 2
    report = {
        "artifact_kind": (
            "PH2_D03_V2_W02_MORPHOLOGY_SUCCESSOR_V3_PRIVATE_R3_"
            "EVALUATION_REPORT"),
        "artifact_version": W02_MORPH_V3_PRIVATE_R3_EVALUATOR_VERSION,
        **core,
        "family_commitment": registration.family_commitment,
        "family_freeze_sha256": family_freeze_sha256,
        "formal_dev_calibration_runs": 1,
        "formal_private_evaluation_runs": 1,
        "formal_shadow_audit_runs": 1,
        "formal_successor_transform_runs": 1,
        "formal_successor_v2_transform_runs": 1,
        "formal_successor_v3_route_dev_runs": 1,
        "formal_successor_v3_route_shadow_runs": 1,
        "formal_training_runs": 1,
        "label_record_reads": core["input_pair_count"],
        "language_capability_mastered": 0,
        "language_readiness": 0,
        "next_action": (
            "W03_COMPILE_FREEZE" if core["status"] == "PASS"
            else "W02_PRIVATE_FAILED_OR_NE_STOP"),
        "observation_reads": core["input_pair_count"],
        "private_content_stream_reads": len(files),
        "private_family_registered": 1,
        "private_payload_gets": private_record_reads,
        "private_payload_reads": len(files) * 4,
        "private_post_content_transport_reads": len(files),
        "private_record_reads": private_record_reads,
        "private_transport_validation_reads": len(files) * 2,
        "release_key": "PH2-D03-V2",
        "run_id": 1,
        "run_scope": "FORMAL_BLIND_PRIVATE_EVALUATION",
        "stage_key": "W-02",
        "teacher_calls": 0,
        "transport_bytes_read": transport_bytes * 4,
        "validated_layout_count": len(files),
        "zero_write_audit": {
            "assessment_writes": 0,
            "candidate_writes": 0,
            "clock_writes": 0,
            "companion_writes": 0,
            "core_writes": 0,
            "evaluator_label_writes": 0,
            "evidence_writes": 0,
            "host_writes": 0,
            "memory_writes": 0,
            "use_writes": 0,
            "v1_overlay_writes": 0,
            "v2_overlay_writes": 0,
            "v3_route_writes": 0,
        },
    }
    if (report["private_record_reads"]
            > registration.resource_budget.max_records
            or report["transport_bytes_read"]
            > registration.resource_budget.max_payload_bytes
            or report["private_payload_gets"]
            > registration.resource_budget.max_payload_gets):
        raise W02MorphologySuccessorV3PrivateR3EvaluationError(
            "R3 private non-logic resource budget exceeded")
    validate_v2_safe_report(report)
    return report


__all__ = [
    "W02_MORPH_V3_PRIVATE_R3_EVALUATOR_VERSION",
    "W02_MORPH_V3_PRIVATE_SUPPORT_KEYS",
    "W02MorphologySuccessorV3PrivateR3EvaluationError",
    "evaluate_w02_morphology_successor_v3_private_r3_pair_stream",
    "run_w02_morphology_successor_v3_private_r3_evaluation",
]
