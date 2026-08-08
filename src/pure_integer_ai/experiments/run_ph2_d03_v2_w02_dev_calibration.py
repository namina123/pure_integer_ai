"""运行唯一 PH2-D03-V2 W-02 正式 dev calibration。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_d03_contract_core import write_immutable_json
from pure_integer_ai.experiments.ph2_d03_v2_w02_dev_calibration import (
    run_w02_dev_calibration,
)


def _failure_seal(output_root: Path, error: BaseException) -> None:
    target = output_root / "run-000001.failure.json"
    write_immutable_json({
        "artifact_kind": "PH2_D03_V2_W02_DEV_CALIBRATION_FAILURE_SEAL",
        "error_evidence_sha256": hashlib.sha256(
            (type(error).__name__ + ":" + str(error)).encode("utf-8")).hexdigest(),
        "error_type": type(error).__name__,
        "formal_dev_calibration_runs": 1,
        "formal_private_evaluation_runs": 0,
        "private_payload_reads": 0,
        "run_id": 1,
        "stage_key": "W-02",
        "status": "FAILED_OR_NE_NO_PRIVATE_ADVANCE",
    }, target)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--dev-root", required=True)
    parser.add_argument("--candidate-artifact-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", default=1, type=int)
    arguments = parser.parse_args()
    output = Path(arguments.output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    target = output / "run-000001.report.json"
    if target.exists():
        raise RuntimeError("W-02 dev formal report 已存在，不得重跑")
    try:
        report = run_w02_dev_calibration(
            arguments.repository_root,
            arguments.dev_root,
            arguments.candidate_artifact_root,
            run_id=arguments.run_id,
        )
        write_immutable_json(report, target)
    except BaseException as error:
        _failure_seal(output, error)
        raise
    print(json.dumps({
        "dimension_results": report["dimension_results"],
        "formal_dev_calibration_runs": 1,
        "formal_private_evaluation_runs": 0,
        "logic_operations": report["logic_operations"],
        "private_payload_reads": 0,
        "report_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "status": report["status"],
        "teacher_calls": 0,
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
