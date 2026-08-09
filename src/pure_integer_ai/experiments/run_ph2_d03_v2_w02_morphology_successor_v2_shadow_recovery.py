"""执行唯一 W-02 successor V2 shadow recovery B。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_d03_contract_core import write_immutable_json
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v2_shadow_audit import (
    run_w02_morphology_successor_v2_shadow_audit,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v2_shadow_recovery import (
    W02_MORPH_V2_SHADOW_RECOVERY_REPORT,
    consume_w02_morphology_successor_v2_shadow_recovery_guard,
    recovery_report_fields,
    write_w02_morphology_successor_v2_shadow_recovery_failure,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--shadow-root", required=True)
    parser.add_argument("--candidate-artifact-root", required=True)
    parser.add_argument("--v1-overlay-artifact-root", required=True)
    parser.add_argument("--v2-overlay-artifact-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", default=1, type=int)
    arguments = parser.parse_args()
    if arguments.run_id != 1:
        raise RuntimeError("shadow recovery run_id 固定为 1")
    output = Path(arguments.output_root).resolve()
    consume_w02_morphology_successor_v2_shadow_recovery_guard(
        arguments.repository_root, output)
    target = output / W02_MORPH_V2_SHADOW_RECOVERY_REPORT
    try:
        report = run_w02_morphology_successor_v2_shadow_audit(
            arguments.repository_root,
            arguments.shadow_root,
            arguments.candidate_artifact_root,
            arguments.v1_overlay_artifact_root,
            arguments.v2_overlay_artifact_root,
            run_id=1,
        )
        report.update(recovery_report_fields(
            arguments.repository_root, output,
            passed=report.get("status") == "PASS"))
        write_immutable_json(report, target)
    except BaseException as error:
        write_w02_morphology_successor_v2_shadow_recovery_failure(
            arguments.repository_root, output, error)
        raise
    print(json.dumps({
        "audit_results": report["audit_results"],
        "formal_shadow_audit_attempts": 2,
        "formal_shadow_audit_passes": report["formal_shadow_audit_passes"],
        "formal_shadow_recovery_runs": 1,
        "logic_operations": report["logic_operations"],
        "report_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "status": report["status"],
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
