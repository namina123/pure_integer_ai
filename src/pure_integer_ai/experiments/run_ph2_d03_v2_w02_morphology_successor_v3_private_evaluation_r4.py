"""Run the only successor V3 R4 blind private evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    read_canonical_object,
    write_immutable_json,
)
from pure_integer_ai.experiments.ph2_d03_v2_evaluator_contract import (
    read_v2_evaluator_boundary_contract,
)
from pure_integer_ai.experiments.ph2_d03_v2_evaluator_firewall import (
    V2PhysicalRoots,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_candidate_store import (
    read_w02_candidate_artifact,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_dev_calibration import (
    _sha256_file,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_overlay import (
    read_w02_morphology_overlay_artifact,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v2_overlay import (
    read_w02_morphology_successor_v2_overlay_artifact,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_private_evaluator_r4 import (
    run_w02_morphology_successor_v3_private_r4_evaluation,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_private_family_r4 import (
    W02_MORPH_V3_PRIVATE_R4_EXPOSURE_LEDGER,
    W02_MORPH_V3_PRIVATE_R4_FAMILY_DOCUMENT,
    W02_MORPH_V3_PRIVATE_R4_FAMILY_FREEZE_PATH,
    W02_MORPH_V3_PRIVATE_R4_FORMAL_FAMILY_NAME,
    W02_MORPH_V3_PRIVATE_R4_REGISTRATION_DOCUMENT,
    consume_w02_morphology_successor_v3_private_r4_guard,
    read_w02_morphology_successor_v3_private_r4_family_freeze,
    verify_w02_morphology_successor_v3_private_r4_consumed_guard,
    w02_morphology_successor_v3_private_r4_registration_from_freeze,
    w02_morphology_successor_v3_private_r4_run_identity,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_private_owner_r4 import (
    W02_MORPH_V3_PRIVATE_PATHS,
    read_w02_morphology_successor_v3_private_owner_r4_receipt,
)


def _failure_seal(
        publication: Path, error: BaseException, *, phase: str,
        report: dict[str, object] | None = None) -> None:
    target = publication / "run-000001.failure.json"
    if target.exists():
        return
    write_immutable_json({
        "artifact_kind": (
            "PH2_D03_V2_W02_MORPHOLOGY_SUCCESSOR_V3_PRIVATE_R4_FAILURE_SEAL"),
        "blind_receipt_published": 0,
        "error_evidence_sha256": hashlib.sha256(
            (type(error).__name__ + ":" + str(error)).encode("utf-8")
        ).hexdigest(),
        "error_type": type(error).__name__,
        "failure_phase": phase,
        "formal_private_evaluation_runs": int(phase != "PRE_GUARD"),
        "language_capability_mastered": 0,
        "language_readiness": 0,
        "private_payload_read_count_known": int(report is not None),
        "private_payload_reads_minimum": (
            int(report.get("private_payload_reads", 0)) if report else 0),
        "run_id": 1,
        "stage_key": "W-02",
        "status": "NE_NO_RECEIPT",
    }, target)


def _private_owner_root(value: str | Path, repository: Path) -> Path:
    root = Path(value).resolve()
    if (root.is_symlink() or not root.is_dir()
            or root.is_relative_to(repository)):
        raise RuntimeError("R4 private owner root is invalid")
    return root


def _validate_pre_guard_inputs(
        compile_root: Path,
        owner_root: Path,
        files: tuple[object, ...],
        freeze: dict[str, object],
        candidate_artifact_root: str | Path,
        v1_overlay_artifact_root: str | Path,
        v2_overlay_artifact_root: str | Path,
        ) -> None:
    required_roots = tuple(compile_root / name for name in (
        "candidate-train", "teacher-train", "dev-calibration", "shadow-audit"))
    if (compile_root.is_symlink() or not compile_root.is_dir()
            or any(root.is_symlink() or not root.is_dir()
                   for root in required_roots)):
        raise RuntimeError("compile root layout is invalid")
    for identity in files:
        layout_key = getattr(identity, "layout_key", "")
        relative = W02_MORPH_V3_PRIVATE_PATHS.get(layout_key)
        target = owner_root / relative if relative else owner_root
        if (not relative or target.is_symlink() or not target.is_file()
                or target.stat().st_size
                != getattr(identity, "transport_size_bytes", -1)):
            raise RuntimeError("R4 private owner transport metadata drifted")
    candidate = read_w02_candidate_artifact(candidate_artifact_root)
    v1 = read_w02_morphology_overlay_artifact(v1_overlay_artifact_root)
    v2 = read_w02_morphology_successor_v2_overlay_artifact(
        v2_overlay_artifact_root)
    chain = freeze["artifact_chain"]
    assert isinstance(chain, dict)
    if (candidate.artifact_manifest_sha256
            != chain["candidate_artifact_manifest_sha256"]
            or candidate.candidate_semantic_sha256
            != chain["candidate_semantic_sha256"]
            or v1.artifact_manifest_sha256
            != chain["v1_overlay_artifact_manifest_sha256"]
            or v1.overlay_semantic_sha256
            != chain["v1_overlay_semantic_sha256"]
            or v1.parent_candidate_semantic_sha256
            != candidate.candidate_semantic_sha256
            or v2.artifact_manifest_sha256
            != chain["v2_overlay_artifact_manifest_sha256"]
            or v2.semantic_sha256 != chain["v2_overlay_semantic_sha256"]):
        raise RuntimeError("formal artifact roots do not match the R4 freeze")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--compile-root", required=True)
    parser.add_argument("--private-owner-root", required=True)
    parser.add_argument("--candidate-artifact-root", required=True)
    parser.add_argument("--v1-overlay-artifact-root", required=True)
    parser.add_argument("--v2-overlay-artifact-root", required=True)
    parser.add_argument("--family-root", required=True)
    arguments = parser.parse_args()

    repository = Path(arguments.repository_root).resolve()
    compile_root = Path(arguments.compile_root).resolve()
    family_root = Path(arguments.family_root).resolve()
    if family_root.name != W02_MORPH_V3_PRIVATE_R4_FORMAL_FAMILY_NAME:
        raise RuntimeError("R4 private formal family name drifted")
    publication = family_root / "publication"
    publication.mkdir(exist_ok=False)
    report_path = publication / "run-000001.report.json"
    failure_path = publication / "run-000001.failure.json"
    if report_path.exists() or failure_path.exists():
        raise RuntimeError("R4 private formal family has already been consumed")

    phase = "PRE_GUARD"
    report: dict[str, object] | None = None
    try:
        freeze = read_w02_morphology_successor_v3_private_r4_family_freeze(
            repository)
        freeze_path = repository / Path(
            *W02_MORPH_V3_PRIVATE_R4_FAMILY_FREEZE_PATH.split("/"))
        family_freeze_sha = _sha256_file(freeze_path)[1]
        if (read_canonical_object(
                family_root / W02_MORPH_V3_PRIVATE_R4_FAMILY_DOCUMENT) != freeze
                or read_canonical_object(
                    family_root /
                    W02_MORPH_V3_PRIVATE_R4_REGISTRATION_DOCUMENT)
                != freeze["registration"]):
            raise RuntimeError("R4 private formal family documents drifted")
        registration = (
            w02_morphology_successor_v3_private_r4_registration_from_freeze(
                freeze))
        _, files = read_w02_morphology_successor_v3_private_owner_r4_receipt(
            repository)
        if [row.to_dict() for row in files] != freeze["owner_input_files"]:
            raise RuntimeError("R4 private owner inventory drifted before guard")
        owner_root = _private_owner_root(arguments.private_owner_root, repository)
        _validate_pre_guard_inputs(
            compile_root,
            owner_root,
            files,
            freeze,
            arguments.candidate_artifact_root,
            arguments.v1_overlay_artifact_root,
            arguments.v2_overlay_artifact_root,
        )
        run_identity = w02_morphology_successor_v3_private_r4_run_identity(
            freeze, family_freeze_sha)
        consume_w02_morphology_successor_v3_private_r4_guard(
            family_root,
            expected_guard_sha256=str(freeze["first_run_guard_sha256"]),
            run_identity_sha256=run_identity,
        )
        phase = "PRIVATE_AUTHORIZATION_OR_EVALUATION"
        roots = V2PhysicalRoots.from_paths(
            compile_root / "candidate-train",
            compile_root / "teacher-train",
            compile_root / "dev-calibration",
            compile_root / "shadow-audit",
            owner_root,
            family_root / W02_MORPH_V3_PRIVATE_R4_EXPOSURE_LEDGER,
        )
        report = run_w02_morphology_successor_v3_private_r4_evaluation(
            read_v2_evaluator_boundary_contract(repository),
            roots,
            registration,
            files,
            arguments.candidate_artifact_root,
            arguments.v1_overlay_artifact_root,
            arguments.v2_overlay_artifact_root,
            family_freeze_sha256=family_freeze_sha,
            run_id=1,
        )
        verify_w02_morphology_successor_v3_private_r4_consumed_guard(
            family_root,
            expected_guard_sha256=str(freeze["first_run_guard_sha256"]),
            run_identity_sha256=run_identity,
        )
        write_immutable_json(report, report_path)
        phase = "REPORT_PUBLISHED"
        if report["status"] != "PASS":
            error = RuntimeError(
                "formal R4 private result was FAIL or NE; receipt blocked")
            _failure_seal(
                publication, error, phase="FORMAL_RESULT_FAIL_OR_NE",
                report=report)
            print(json.dumps({
                "formal_private_evaluation_runs": 1,
                "report_sha256": hashlib.sha256(
                    report_path.read_bytes()).hexdigest(),
                "status": report["status"],
            }, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            return 2
    except BaseException as error:
        _failure_seal(publication, error, phase=phase, report=report)
        raise

    assert report is not None
    print(json.dumps({
        "formal_private_evaluation_runs": 1,
        "hard_conjunct_results": report["hard_conjunct_results"],
        "logic_operations": report["logic_operations"],
        "private_payload_reads": report["private_payload_reads"],
        "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "route_authorized_count": report["route_authorized_count"],
        "status": report["status"],
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
