"""运行唯一 PH2-D03-V2 W-02 morphology successor blind private evaluation。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
    read_canonical_object,
    write_immutable_json,
)
from pure_integer_ai.experiments.ph2_d03_v2_evaluator_contract import (
    read_v2_evaluator_boundary_contract,
)
from pure_integer_ai.experiments.ph2_d03_v2_evaluator_firewall import (
    V2PhysicalRoots,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_contract import (
    read_w02_compile_freeze,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_dev_calibration import (
    _sha256_file,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_private_evaluator import (
    run_w02_morphology_successor_private_evaluation,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_private_family import (
    W02_PRIVATE_EXPOSURE_LEDGER,
    W02_PRIVATE_FAMILY_DOCUMENT,
    W02_PRIVATE_FAMILY_FREEZE_PATH,
    W02_PRIVATE_REGISTRATION_DOCUMENT,
    consume_w02_morphology_successor_private_guard,
    read_w02_morphology_successor_private_family_freeze,
    verify_w02_morphology_successor_private_consumed_guard,
    w02_private_registration_from_freeze,
)


def _failure_seal(
        publication: Path, error: BaseException, *, phase: str) -> None:
    write_immutable_json({
        "artifact_kind": (
            "PH2_D03_V2_W02_MORPHOLOGY_SUCCESSOR_PRIVATE_FAILURE_SEAL"),
        "blind_receipt_published": 0,
        "error_evidence_sha256": hashlib.sha256(
            (type(error).__name__ + ":" + str(error)).encode("utf-8")
        ).hexdigest(),
        "error_type": type(error).__name__,
        "failure_phase": phase,
        "formal_private_evaluation_runs": int(phase != "PRE_GUARD"),
        "language_capability_mastered": 0,
        "language_readiness": 0,
        "private_payload_read_count_known": int(phase == "PRE_GUARD"),
        "private_payload_reads_minimum": 0,
        "run_id": 1,
        "stage_key": "W-02",
        "status": "NE_NO_RECEIPT",
    }, publication / "run-000001.failure.json")


def _run_identity(freeze: dict[str, object], family_freeze_sha256: str) -> str:
    registration = freeze["registration"]
    assert isinstance(registration, dict)
    return hashlib.sha256(canonical_json_bytes({
        "candidate_binding_sha256": freeze["candidate_binding_sha256"],
        "family_commitment": registration["family_commitment"],
        "family_freeze_sha256": family_freeze_sha256,
        "run_id": 1,
        "run_scope": "FORMAL_BLIND_PRIVATE_EVALUATION",
    })).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--compile-root", required=True)
    parser.add_argument("--candidate-artifact-root", required=True)
    parser.add_argument("--overlay-artifact-root", required=True)
    parser.add_argument("--family-root", required=True)
    arguments = parser.parse_args()
    repository = Path(arguments.repository_root).resolve()
    compile_root = Path(arguments.compile_root).resolve()
    family_root = Path(arguments.family_root).resolve()
    publication = family_root / "publication"
    publication.mkdir(exist_ok=False)
    report_path = publication / "run-000001.report.json"
    failure_path = publication / "run-000001.failure.json"
    if report_path.exists() or failure_path.exists():
        raise RuntimeError("W-02 private formal family 已消费，不得重跑")
    phase = "PRE_GUARD"
    try:
        freeze = read_w02_morphology_successor_private_family_freeze(repository)
        public_freeze_path = repository / Path(
            *W02_PRIVATE_FAMILY_FREEZE_PATH.split("/"))
        family_freeze_sha = _sha256_file(public_freeze_path)[1]
        if (read_canonical_object(family_root / W02_PRIVATE_FAMILY_DOCUMENT)
                != freeze
                or read_canonical_object(
                    family_root / W02_PRIVATE_REGISTRATION_DOCUMENT)
                != freeze["registration"]):
            raise RuntimeError("W-02 private family root document 漂移")
        registration = w02_private_registration_from_freeze(freeze)
        run_identity = _run_identity(freeze, family_freeze_sha)
        consume_w02_morphology_successor_private_guard(
            family_root,
            expected_guard_sha256=str(freeze["first_run_guard_sha256"]),
            run_identity_sha256=run_identity)
        phase = "PRIVATE_AUTHORIZATION_OR_EVALUATION"
        roots = V2PhysicalRoots.from_paths(
            compile_root / "candidate-train",
            compile_root / "teacher-train",
            compile_root / "dev-calibration",
            compile_root / "shadow-audit",
            compile_root / "private-evaluator",
            family_root / W02_PRIVATE_EXPOSURE_LEDGER,
        )
        report = run_w02_morphology_successor_private_evaluation(
            read_v2_evaluator_boundary_contract(repository), roots,
            registration, read_w02_compile_freeze(repository),
            arguments.candidate_artifact_root,
            arguments.overlay_artifact_root,
            family_freeze_sha256=family_freeze_sha,
            run_id=1)
        verify_w02_morphology_successor_private_consumed_guard(
            family_root,
            expected_guard_sha256=str(freeze["first_run_guard_sha256"]),
            run_identity_sha256=run_identity)
        write_immutable_json(report, report_path)
        phase = "REPORT_PUBLISHED"
    except BaseException as error:
        _failure_seal(publication, error, phase=phase)
        raise
    print(json.dumps({
        "formal_private_evaluation_runs": 1,
        "hard_conjunct_results": report["hard_conjunct_results"],
        "logic_operations": report["logic_operations"],
        "private_payload_reads": report["private_payload_reads"],
        "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "status": report["status"],
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
