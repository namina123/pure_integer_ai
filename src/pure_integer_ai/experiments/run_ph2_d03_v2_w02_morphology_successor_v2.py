"""运行唯一 W-02 formal morphology successor V2 transform。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_d03_contract_core import write_immutable_json
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v2_contract import (
    W02_MORPH_V2_GUARD_CONSUMED,
    read_w02_morphology_successor_v2_runtime_freeze,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v2_overlay import (
    run_w02_morphology_successor_v2_overlay_formal,
)


def _failure_seal(successor_root: Path, run_id: int, error: BaseException) -> None:
    """guard 消费后封存失败；消费前的只读拒绝不计正式运行。"""
    consumed = successor_root / Path(*W02_MORPH_V2_GUARD_CONSUMED.split("/"))
    if not consumed.is_file():
        return
    target = (successor_root / "morphology-v2-overlay-store" / "failures"
              / f"run-{run_id:06d}.json")
    write_immutable_json({
        "artifact_kind": "PH2_D03_V2_W02_MORPHOLOGY_SUCCESSOR_V2_FAILURE_SEAL",
        "error_evidence_sha256": hashlib.sha256(
            (type(error).__name__ + ":" + str(error)).encode("utf-8")
        ).hexdigest(),
        "error_type": type(error).__name__,
        "formal_successor_v2_transform_runs": 1,
        "formal_training_runs": 0,
        "private_payload_reads": 0,
        "run_id": run_id,
        "stage_key": "W-02",
        "status": "FAILED_OR_NE_NO_RERUN",
        "teacher_calls": 0,
    }, target)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--successor-root", required=True)
    parser.add_argument("--candidate-artifact-root")
    parser.add_argument("--v1-overlay-artifact-root")
    parser.add_argument("--workers", required=True, type=int, choices=(1, 2, 4))
    parser.add_argument("--mode", default="fresh", choices=("fresh", "resume"))
    parser.add_argument("--run-id", default=1, type=int)
    arguments = parser.parse_args()
    repository = Path(arguments.repository_root).resolve()
    successor = Path(arguments.successor_root).resolve()
    freeze = read_w02_morphology_successor_v2_runtime_freeze(repository)
    try:
        result = run_w02_morphology_successor_v2_overlay_formal(
            successor_root=successor,
            candidate_artifact_root=arguments.candidate_artifact_root,
            v1_overlay_artifact_root=arguments.v1_overlay_artifact_root,
            runtime_freeze_sha256=freeze.sha256(),
            expected_guard_sha256=freeze.first_run_guard_sha256,
            expected_parent_candidate_manifest_sha256=(
                freeze.parent_candidate_manifest_sha256),
            expected_parent_candidate_semantic_sha256=(
                freeze.parent_candidate_semantic_sha256),
            expected_parent_v1_manifest_sha256=freeze.parent_v1_manifest_sha256,
            expected_parent_v1_semantic_sha256=freeze.parent_v1_semantic_sha256,
            run_id=arguments.run_id, requested_workers=arguments.workers,
            mode=arguments.mode, budget=freeze.resource_budget)
    except BaseException as error:
        _failure_seal(successor, arguments.run_id, error)
        raise
    print(json.dumps({
        "artifact_manifest_sha256": result.artifact_manifest_sha256,
        "formal_private_evaluation_runs": 0,
        "formal_successor_v2_transform_runs": 1,
        "formal_training_runs": 0,
        "logic_operations": result.logic_operations,
        "private_payload_reads": 0,
        "rule_row_count": result.rule_row_count,
        "run_identity_sha256": result.run_identity_sha256,
        "semantic_sha256": result.semantic_sha256,
        "status": "MORPHOLOGY_V2_OVERLAY_SEALED",
        "teacher_calls": 0,
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
