"""PASS-only public receipt for the successor V3 blind private run."""
from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
    read_canonical_object,
    write_immutable_json,
)
from pure_integer_ai.experiments.ph2_d03_v2_evaluator_contract import (
    validate_v2_safe_report,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_dev_calibration import (
    _sha256_file,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_private_evaluator import (
    W02_MORPH_V3_PRIVATE_EVALUATOR_VERSION,
    W02_MORPH_V3_PRIVATE_SUPPORT_KEYS,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_private_family import (
    W02_MORPH_V3_PRIVATE_FAMILY_FREEZE_PATH,
    W02_MORPH_V3_PRIVATE_FORMAL_FAMILY_NAME,
    W02MorphologySuccessorV3PrivateFamilyError,
    read_w02_morphology_successor_v3_private_family_freeze,
    verify_w02_morphology_successor_v3_private_consumed_guard,
    w02_morphology_successor_v3_private_run_identity,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_private_owner import (
    W02_MORPH_V3_PRIVATE_LAYOUTS,
    W02_MORPH_V3_PRIVATE_PAIR_COUNT,
    W02_MORPH_V3_PRIVATE_SOURCE_COUNT,
)


W02_MORPH_V3_PRIVATE_RECEIPT_VERSION = (
    "PH2-D03-V2-W02-MORPHOLOGY-SUCCESSOR-V3-PRIVATE-RECEIPT-V1"
)
W02_MORPH_V3_PRIVATE_RECEIPT_PATH = (
    "data/ph2/manifests/d03_v2/stages/"
    "ph2_d03_v2_w02_morphology_successor_v3_private_evaluation_receipt_v1.json"
)
W02_MORPH_V3_PRIVATE_REPORT = "publication/run-000001.report.json"
W02_MORPH_V3_PRIVATE_ZERO_WRITE_KEYS = frozenset({
    "assessment_writes", "candidate_writes", "clock_writes",
    "companion_writes", "core_writes", "evaluator_label_writes",
    "evidence_writes", "host_writes", "memory_writes", "use_writes",
    "v1_overlay_writes", "v2_overlay_writes", "v3_route_writes",
})
W02_MORPH_V3_PRIVATE_ZERO_CALL_WINDOWS = (
    "BEFORE_PRIVATE_READ",
    "DURING_PRIVATE_EVALUATION",
    "AFTER_PRIVATE_EVALUATION",
)


def _sha256(value: object, *, where: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise W02MorphologySuccessorV3PrivateFamilyError(
            f"{where} is not lowercase SHA-256")
    return value


def _repository_file(repository: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    target = (repository / Path(*pure.parts)).resolve()
    if (not relative or "\\" in relative or pure.is_absolute()
            or pure.as_posix() != relative or ".." in pure.parts
            or target.is_symlink() or not target.is_relative_to(repository)
            or not target.is_file()):
        raise W02MorphologySuccessorV3PrivateFamilyError(
            "V3 private publication repository file is invalid")
    return target


def _family_root(value: str | Path) -> Path:
    root = Path(value).resolve()
    if (root.name != W02_MORPH_V3_PRIVATE_FORMAL_FAMILY_NAME
            or root.is_symlink() or not root.is_dir()):
        raise W02MorphologySuccessorV3PrivateFamilyError(
            "V3 private publication family root is invalid")
    return root


def _pass_dimension_map(
        rows: object, expected: dict[str, int]) -> dict[str, dict[str, object]]:
    if not isinstance(rows, list):
        raise W02MorphologySuccessorV3PrivateFamilyError(
            "V3 private dimension results are not a list")
    by_key: dict[str, dict[str, object]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("dimension_key"), str):
            raise W02MorphologySuccessorV3PrivateFamilyError(
                "V3 private dimension result is invalid")
        by_key[str(row["dimension_key"])] = row
    if set(by_key) != set(expected):
        raise W02MorphologySuccessorV3PrivateFamilyError(
            "V3 private dimension inventory drifted")
    for key, denominator in expected.items():
        row = by_key[key]
        if (row.get("denominator") != denominator
                or row.get("numerator") != denominator
                or row.get("failed") != 0 or row.get("ne") != 0
                or row.get("status") != "PASS"):
            raise W02MorphologySuccessorV3PrivateFamilyError(
                "V3 private dimension did not pass its frozen denominator")
        _sha256(row.get("evidence_sha256"), where="V3 private dimension evidence")
    return by_key


def _validate_formal_pass_report(
        report: dict[str, Any], freeze: dict[str, object],
        family_freeze_sha256: str) -> None:
    validate_v2_safe_report(report)
    chain = freeze["artifact_chain"]
    assert isinstance(chain, dict)
    dimensions = freeze["dimension_denominator_counts"]
    assert isinstance(dimensions, dict)
    _pass_dimension_map(report.get("dimension_results"), dimensions)
    support = _pass_dimension_map(
        report.get("support_results"),
        {key: 1 for key in W02_MORPH_V3_PRIVATE_SUPPORT_KEYS})
    hard = report.get("hard_conjunct_results")
    if (not isinstance(hard, list)
            or hard != [*report["dimension_results"], *report["support_results"]]
            or len(hard) != 9):
        raise W02MorphologySuccessorV3PrivateFamilyError(
            "V3 private hard-conjunct projection drifted")
    zero_write = report.get("zero_write_audit")
    zero_call_windows = report.get("zero_call_windows")
    family_counts = report.get("family_counts")
    route_capabilities = report.get("route_capability_sha256s")
    transport_bytes = sum(
        int(row["transport_size_bytes"])
        for row in freeze["owner_input_files"])
    if (report.get("artifact_kind")
            != "PH2_D03_V2_W02_MORPHOLOGY_SUCCESSOR_V3_PRIVATE_EVALUATION_REPORT"
            or report.get("artifact_version")
            != W02_MORPH_V3_PRIVATE_EVALUATOR_VERSION
            or report.get("status") != "PASS"
            or report.get("run_scope") != "FORMAL_BLIND_PRIVATE_EVALUATION"
            or report.get("run_id") != 1
            or report.get("family_freeze_sha256") != family_freeze_sha256
            or report.get("family_commitment")
            != freeze["registration"]["family_commitment"]
            or report.get("candidate_artifact_manifest_sha256")
            != chain["candidate_artifact_manifest_sha256"]
            or report.get("candidate_semantic_sha256")
            != chain["candidate_semantic_sha256"]
            or report.get("v1_overlay_artifact_manifest_sha256")
            != chain["v1_overlay_artifact_manifest_sha256"]
            or report.get("v1_overlay_semantic_sha256")
            != chain["v1_overlay_semantic_sha256"]
            or report.get("v2_overlay_artifact_manifest_sha256")
            != chain["v2_overlay_artifact_manifest_sha256"]
            or report.get("v2_overlay_semantic_sha256")
            != chain["v2_overlay_semantic_sha256"]
            or report.get("source_count") != freeze["owner_source_count"]
            or report.get("input_pair_count") != freeze["owner_pair_count"]
            or report.get("evaluation_count") != freeze["owner_pair_count"]
            or report.get("route_authorized_count") != freeze["owner_pair_count"]
            or report.get("observation_reads") != freeze["owner_pair_count"]
            or report.get("label_record_reads") != freeze["owner_pair_count"]
            or report.get("validated_layout_count")
            != len(W02_MORPH_V3_PRIVATE_LAYOUTS)
            or report.get("private_content_stream_reads") != 7
            or report.get("private_post_content_transport_reads") != 7
            or report.get("private_transport_validation_reads") != 14
            or report.get("private_payload_reads") != 28
            or report.get("private_record_reads")
            != W02_MORPH_V3_PRIVATE_SOURCE_COUNT
            + W02_MORPH_V3_PRIVATE_PAIR_COUNT * 2
            or report.get("private_payload_gets")
            != report.get("private_record_reads")
            or report.get("transport_bytes_read") != transport_bytes * 4
            or report.get("max_v2_edge_candidates_per_requested_span", 9) > 8
            or report.get("formal_dev_calibration_runs") != 1
            or report.get("formal_private_evaluation_runs") != 1
            or report.get("formal_shadow_audit_runs") != 1
            or report.get("formal_successor_transform_runs") != 1
            or report.get("formal_successor_v2_transform_runs") != 1
            or report.get("formal_successor_v3_route_dev_runs") != 1
            or report.get("formal_successor_v3_route_shadow_runs") != 1
            or report.get("formal_training_runs") != 1
            or report.get("private_family_registered") != 1
            or report.get("teacher_calls") != 0
            or report.get("language_capability_mastered") != 0
            or report.get("language_readiness") != 0
            or report.get("release_key") != "PH2-D03-V2"
            or report.get("stage_key") != "W-02"
            or report.get("next_action") != "W03_COMPILE_FREEZE"
            or not isinstance(family_counts, dict)
            or family_counts != {
                "AUTHORED_OOV": 0,
                "UD_ANNOTATION": W02_MORPH_V3_PRIVATE_PAIR_COUNT,
                "UNICODE_ANNOTATION": 0,
            }
            or not isinstance(route_capabilities, list)
            or len(route_capabilities) != 1
            or any(_sha256(value, where="V3 route capability") != value
                   for value in route_capabilities)
            or not isinstance(zero_write, dict)
            or set(zero_write) != W02_MORPH_V3_PRIVATE_ZERO_WRITE_KEYS
            or any(value != 0 for value in zero_write.values())
            or zero_call_windows != [
                {"api_calls": 0, "llm_calls": 0, "teacher_calls": 0,
                 "window_key": key}
                for key in W02_MORPH_V3_PRIVATE_ZERO_CALL_WINDOWS
            ]
            or type(report.get("logic_operations")) is not int
            or report["logic_operations"] <= 0
            or report["logic_operations"]
            > freeze["resource_budget"]["max_logic_operations"]):
        raise W02MorphologySuccessorV3PrivateFamilyError(
            "V3 private formal PASS report identity or resource state drifted")
    for name in (
            "source_identity_sha256", "route_semantic_sha256",
            "routed_index_semantic_sha256"):
        _sha256(report.get(name), where=f"V3 private {name}")
    if set(support) != set(W02_MORPH_V3_PRIVATE_SUPPORT_KEYS):
        raise W02MorphologySuccessorV3PrivateFamilyError(
            "V3 private support gate inventory drifted")


def build_w02_morphology_successor_v3_private_receipt(
        repository_root: str | Path, family_root: str | Path) -> dict[str, object]:
    """Project a PASS-only, aggregate-only receipt from the sealed family."""
    repository = Path(repository_root).resolve()
    root = _family_root(family_root)
    freeze = read_w02_morphology_successor_v3_private_family_freeze(repository)
    freeze_path = _repository_file(
        repository, W02_MORPH_V3_PRIVATE_FAMILY_FREEZE_PATH)
    freeze_size, freeze_sha = _sha256_file(freeze_path)
    run_identity = w02_morphology_successor_v3_private_run_identity(
        freeze, freeze_sha)
    verify_w02_morphology_successor_v3_private_consumed_guard(
        root,
        expected_guard_sha256=str(freeze["first_run_guard_sha256"]),
        run_identity_sha256=run_identity,
    )
    report_path = root / Path(*PurePosixPath(W02_MORPH_V3_PRIVATE_REPORT).parts)
    if report_path.is_symlink() or not report_path.is_file():
        raise W02MorphologySuccessorV3PrivateFamilyError(
            "V3 private formal report is missing")
    report = read_canonical_object(report_path)
    if (not isinstance(report, dict)
            or report_path.read_bytes() != canonical_json_bytes(report) + b"\n"):
        raise W02MorphologySuccessorV3PrivateFamilyError(
            "V3 private formal report is not canonical")
    _validate_formal_pass_report(report, freeze, freeze_sha)
    report_size, report_sha = _sha256_file(report_path)
    return {
        "artifact_chain_sha256": freeze["artifact_chain_sha256"],
        "artifact_kind": (
            "PH2_D03_V2_W02_MORPHOLOGY_SUCCESSOR_V3_PRIVATE_RECEIPT"),
        "artifact_version": W02_MORPH_V3_PRIVATE_RECEIPT_VERSION,
        "code_freeze_sha256": freeze["code_freeze_sha256"],
        "dimension_results": report["dimension_results"],
        "family_commitment": report["family_commitment"],
        "family_counts": report["family_counts"],
        "family_freeze_file_sha256": freeze_sha,
        "family_freeze_size_bytes": freeze_size,
        "formal_dev_calibration_runs": 1,
        "formal_private_evaluation_runs": 1,
        "formal_shadow_audit_runs": 1,
        "formal_successor_transform_runs": 1,
        "formal_successor_v2_transform_runs": 1,
        "formal_successor_v3_route_dev_runs": 1,
        "formal_successor_v3_route_shadow_runs": 1,
        "formal_training_runs": 1,
        "language_capability_mastered": 0,
        "language_readiness": 0,
        "logic_operations": report["logic_operations"],
        "max_v2_edge_candidates_per_requested_span":
            report["max_v2_edge_candidates_per_requested_span"],
        "next_action": "W03_COMPILE_FREEZE",
        "owner_family_key": freeze["owner_family_key"],
        "owner_metadata_sha256": freeze["owner_metadata_sha256"],
        "private_family_registered": 1,
        "private_payload_gets": report["private_payload_gets"],
        "private_payload_reads": report["private_payload_reads"],
        "private_record_reads": report["private_record_reads"],
        "release_key": "PH2-D03-V2",
        "report_file_sha256": report_sha,
        "report_size_bytes": report_size,
        "route_authorized_count": report["route_authorized_count"],
        "route_semantic_sha256": report["route_semantic_sha256"],
        "run_id": 1,
        "stage_key": "W-02",
        "status": "W02_SUCCESSOR_V3_RUNTIME_EVIDENCED",
        "support_results": report["support_results"],
        "teacher_calls": 0,
        "transport_bytes_read": report["transport_bytes_read"],
        "zero_write_audit": report["zero_write_audit"],
    }


def publish_w02_morphology_successor_v3_private_receipt(
        repository_root: str | Path, family_root: str | Path) -> Path:
    repository = Path(repository_root).resolve()
    value = build_w02_morphology_successor_v3_private_receipt(
        repository, family_root)
    validate_v2_safe_report(value)
    target = repository / Path(*PurePosixPath(
        W02_MORPH_V3_PRIVATE_RECEIPT_PATH).parts)
    write_immutable_json(value, target)
    if target.read_bytes() != canonical_json_bytes(value) + b"\n":
        raise W02MorphologySuccessorV3PrivateFamilyError(
            "V3 private public receipt publication drifted")
    return target


__all__ = [
    "W02_MORPH_V3_PRIVATE_RECEIPT_PATH",
    "W02_MORPH_V3_PRIVATE_RECEIPT_VERSION",
    "W02_MORPH_V3_PRIVATE_REPORT",
    "build_w02_morphology_successor_v3_private_receipt",
    "publish_w02_morphology_successor_v3_private_receipt",
]
