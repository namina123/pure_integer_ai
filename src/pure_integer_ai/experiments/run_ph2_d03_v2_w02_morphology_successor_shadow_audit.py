"""运行唯一 PH2-D03-V2 W-02 morphology successor shadow audit。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_d03_contract_core import write_immutable_json
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_shadow_audit import (
    run_w02_morphology_successor_shadow_audit,
)


def _failure_seal(output_root: Path, error: BaseException) -> None:
    write_immutable_json({
        "artifact_kind": (
            "PH2_D03_V2_W02_MORPHOLOGY_SUCCESSOR_SHADOW_AUDIT_FAILURE_SEAL"),
        "error_evidence_sha256": hashlib.sha256(
            (type(error).__name__ + ":" + str(error)).encode("utf-8")
        ).hexdigest(),
        "error_type": type(error).__name__,
        "formal_private_evaluation_runs": 0,
        "formal_shadow_audit_runs": 1,
        "private_payload_reads": 0,
        "run_id": 1,
        "stage_key": "W-02",
        "status": "FAILED_OR_NE_NO_PRIVATE_REGISTRATION",
    }, output_root / "run-000001.failure.json")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--shadow-root", required=True)
    parser.add_argument("--candidate-artifact-root", required=True)
    parser.add_argument("--overlay-artifact-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", default=1, type=int)
    arguments = parser.parse_args()
    output = Path(arguments.output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    target = output / "run-000001.report.json"
    failure = output / "run-000001.failure.json"
    if target.exists() or failure.exists():
        raise RuntimeError("shadow formal family 已消费，不得重跑")
    try:
        report = run_w02_morphology_successor_shadow_audit(
            arguments.repository_root, arguments.shadow_root,
            arguments.candidate_artifact_root, arguments.overlay_artifact_root,
            run_id=arguments.run_id)
        write_immutable_json(report, target)
    except BaseException as error:
        _failure_seal(output, error)
        raise
    print(json.dumps({
        "audit_results": report["audit_results"],
        "formal_private_evaluation_runs": 0,
        "formal_shadow_audit_runs": 1,
        "logic_operations": report["logic_operations"],
        "private_payload_reads": 0,
        "report_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "status": report["status"],
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
