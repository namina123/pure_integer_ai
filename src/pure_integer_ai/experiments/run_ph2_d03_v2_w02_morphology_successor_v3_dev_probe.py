"""Run the single formal W-02 V3 source-route dev probe."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_d03_contract_core import write_immutable_json
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v3_dev_probe import (
    W02_MORPH_V3_DEV_FREEZE_PATH,
    assert_w02_morphology_successor_v3_dev_preflight,
    read_w02_morphology_successor_v3_dev_freeze,
    run_w02_morphology_successor_v3_dev_preflight,
)


def _failure_seal(output: Path, error: BaseException) -> None:
    write_immutable_json({
        "artifact_kind": (
            "PH2_D03_V2_W02_MORPHOLOGY_SUCCESSOR_V3_ROUTE_DEV_FAILURE_SEAL"),
        "error_evidence_sha256": hashlib.sha256(
            (type(error).__name__ + ":" + str(error)).encode("utf-8")
        ).hexdigest(),
        "error_type": type(error).__name__,
        "formal_dev_calibration_runs": 1,
        "formal_private_evaluation_runs": 0,
        "private_payload_reads": 0,
        "run_id": 1,
        "stage_key": "W-02",
        "status": "FAILED_OR_NE_NO_SHADOW_ADVANCE",
    }, output / "run-000001.failure.json")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--dev-root", required=True)
    parser.add_argument("--candidate-artifact-root", required=True)
    parser.add_argument("--v1-overlay-artifact-root", required=True)
    parser.add_argument("--v2-overlay-artifact-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", default=1, type=int)
    arguments = parser.parse_args()
    if arguments.run_id != 1:
        raise RuntimeError("successor V3 route dev formal run_id must be 1")
    output = Path(arguments.output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "run-000001.report.json"
    failure_path = output / "run-000001.failure.json"
    if report_path.exists() or failure_path.exists():
        raise RuntimeError("successor V3 route dev formal family is consumed")
    try:
        freeze = read_w02_morphology_successor_v3_dev_freeze(
            arguments.repository_root)
        preflight = run_w02_morphology_successor_v3_dev_preflight(
            arguments.repository_root,
            arguments.dev_root,
            arguments.candidate_artifact_root,
            arguments.v1_overlay_artifact_root,
            arguments.v2_overlay_artifact_root,
        )
        assert_w02_morphology_successor_v3_dev_preflight(preflight, freeze)
        report = {
            **preflight,
            "code_freeze_sha256": freeze["code_freeze_sha256"],
            "formal_dev_calibration_runs": 1,
            "freeze_file_sha256": hashlib.sha256(
                (Path(arguments.repository_root).resolve()
                 / W02_MORPH_V3_DEV_FREEZE_PATH).read_bytes()).hexdigest(),
            "next_action": "W02_SUCCESSOR_V3_ROUTE_SHADOW_FREEZE",
            "run_id": 1,
            "run_scope": "FORMAL",
        }
        write_immutable_json(report, report_path)
    except BaseException as error:
        _failure_seal(output, error)
        raise
    print(json.dumps({
        "dimension_results": report["dimension_results"],
        "formal_dev_calibration_runs": 1,
        "formal_private_evaluation_runs": 0,
        "logic_operations": report["logic_operations"],
        "private_payload_reads": 0,
        "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "status": report["status"],
        "teacher_calls": 0,
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
