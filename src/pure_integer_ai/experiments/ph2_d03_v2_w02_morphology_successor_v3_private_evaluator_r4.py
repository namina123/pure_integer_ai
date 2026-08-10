"""R4 revision of the successor V3 blind private evaluator."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from pure_integer_ai.experiments.ph2_dataset_contract import (
    EvaluatorLabelRecord,
    ObservationRecord,
    SourceRefRecord,
)
from pure_integer_ai.experiments.ph2_d03_v2_blind_private_source_extension_v5 import (
    blind_private_source_specs_v5,
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
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_private_evaluator import (
    W02_MORPH_V3_PRIVATE_SUPPORT_KEYS,
    evaluate_w02_morphology_successor_v3_private_pair_stream,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_private_evaluator_r3 import (
    _apply_label_fail_closed,
    _prepare_label_stream,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_private_io_r4 import (
    authorize_w02_morphology_successor_v3_private_r4_files,
    iter_w02_morphology_successor_v3_private_r4_pairs,
    r4_private_file_by_layout,
    r4_private_split_layout,
    read_and_close_w02_morphology_successor_v3_private_r4_sources,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_private_owner_r4_contract import (
    W02_MORPH_V3_PRIVATE_SPLITS,
    W02MorphologySuccessorV3PrivateR4FileIdentity,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_route import (
    W02MorphologySourceCapability,
    w02_ud_morphology_source_capability,
)


W02_MORPH_V3_PRIVATE_R4_EVALUATOR_VERSION = (
    "PH2-D03-V2-W02-MORPHOLOGY-SUCCESSOR-V3-PRIVATE-EVALUATOR-R4-V1"
)


# object-model: exception
class W02MorphologySuccessorV3PrivateR4EvaluationError(RuntimeError):
    """The R4 V5-first route, aggregate, or resource contract drifted."""


def evaluate_w02_morphology_successor_v3_private_r4_pair_stream(
        candidate_artifact_root: str | Path,
        v1_overlay_artifact_root: str | Path,
        v2_overlay_artifact_root: str | Path,
        sources: Iterable[SourceRefRecord],
        capabilities: Iterable[W02MorphologySourceCapability],
        pairs: Iterable[tuple[ObservationRecord, EvaluatorLabelRecord]],
        budget: V2EvaluatorResourceBudget,
        ) -> dict[str, object]:
    """Retain R3 label fail-closed behavior after V5 source closure."""
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


def run_w02_morphology_successor_v3_private_r4_evaluation(
        boundary: V2EvaluatorBoundaryContract,
        roots: V2PhysicalRoots,
        registration: V2PrivateFamilyRegistration,
        files: tuple[W02MorphologySuccessorV3PrivateR4FileIdentity, ...],
        candidate_artifact_root: str | Path,
        v1_overlay_artifact_root: str | Path,
        v2_overlay_artifact_root: str | Path,
        *, family_freeze_sha256: str, run_id: int = 1,
        ) -> dict[str, object]:
    """Consume an already-guarded R4 family in one V5-first formal stream."""
    if run_id != 1:
        raise W02MorphologySuccessorV3PrivateR4EvaluationError(
            "formal R4 private run_id must be one")
    if (not isinstance(registration, V2PrivateFamilyRegistration)
            or registration.stage_key != "W-02"
            or registration.formal_run_count != 0
            or registration.private_payload_reads != 0):
        raise W02MorphologySuccessorV3PrivateR4EvaluationError(
            "R4 private registration state drifted")
    permits = authorize_w02_morphology_successor_v3_private_r4_files(
        boundary, roots, registration, files)

    # This materialization is deliberate: every SourceRef is validated by V5,
    # and its full content/transport identity is closed before pair I/O exists.
    sources = read_and_close_w02_morphology_successor_v3_private_r4_sources(
        files, permits)
    pairs = (
        pair for split in W02_MORPH_V3_PRIVATE_SPLITS
        for pair in iter_w02_morphology_successor_v3_private_r4_pairs(
            files, permits, split)
    )
    capabilities = tuple(
        w02_ud_morphology_source_capability(spec)
        for spec in blind_private_source_specs_v5())
    core = evaluate_w02_morphology_successor_v3_private_r4_pair_stream(
        candidate_artifact_root,
        v1_overlay_artifact_root,
        v2_overlay_artifact_root,
        sources,
        capabilities,
        pairs,
        registration.resource_budget,
    )
    authorize_w02_morphology_successor_v3_private_r4_files(
        boundary, roots, registration, files)
    expected = sum(
        r4_private_file_by_layout(
            files, r4_private_split_layout(split, "observation")).record_count
        for split in W02_MORPH_V3_PRIVATE_SPLITS)
    if core["source_count"] != len(sources) or core["input_pair_count"] != expected:
        raise W02MorphologySuccessorV3PrivateR4EvaluationError(
            "R4 private input was not fully consumed")
    transport_bytes = sum(row.transport_size_bytes for row in files)
    private_record_reads = core["source_count"] + core["input_pair_count"] * 2
    report = {
        "artifact_kind": (
            "PH2_D03_V2_W02_MORPHOLOGY_SUCCESSOR_V3_PRIVATE_R4_"
            "EVALUATION_REPORT"),
        "artifact_version": W02_MORPH_V3_PRIVATE_R4_EVALUATOR_VERSION,
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
        "source_ref_closure_before_pair_stream": 1,
        "source_ref_records_closed": len(sources),
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
    if (report["private_record_reads"] > registration.resource_budget.max_records
            or report["transport_bytes_read"]
            > registration.resource_budget.max_payload_bytes
            or report["private_payload_gets"]
            > registration.resource_budget.max_payload_gets):
        raise W02MorphologySuccessorV3PrivateR4EvaluationError(
            "R4 private non-logic resource budget exceeded")
    validate_v2_safe_report(report)
    return report


__all__ = [
    "W02_MORPH_V3_PRIVATE_R4_EVALUATOR_VERSION",
    "W02_MORPH_V3_PRIVATE_SUPPORT_KEYS",
    "W02MorphologySuccessorV3PrivateR4EvaluationError",
    "evaluate_w02_morphology_successor_v3_private_r4_pair_stream",
    "run_w02_morphology_successor_v3_private_r4_evaluation",
]
