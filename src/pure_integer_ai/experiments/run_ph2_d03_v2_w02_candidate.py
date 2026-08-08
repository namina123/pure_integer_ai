"""运行唯一 PH2-D03-V2 W-02 正式 Candidate first run。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_d03_contract_core import write_immutable_json
from pure_integer_ai.experiments.ph2_d03_v2_w02_candidate_store import (
    run_w02_candidate,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_contract import (
    W02_FIRST_RUN_GUARD_CONSUMED,
    read_w02_compile_freeze,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_runtime_contract import (
    read_w02_candidate_runtime_freeze,
)


def _failure_seal(candidate_root: Path, run_id: int, error: BaseException) -> None:
    consumed = candidate_root / Path(*W02_FIRST_RUN_GUARD_CONSUMED.split("/"))
    if not consumed.is_file():
        return
    target = candidate_root / "candidate-store" / "failures" / f"run-{run_id:06d}.json"
    write_immutable_json({
        "artifact_kind": "PH2_D03_V2_W02_CANDIDATE_FAILURE_SEAL",
        "error_evidence_sha256": hashlib.sha256(
            (type(error).__name__ + ":" + str(error)).encode("utf-8")).hexdigest(),
        "error_type": type(error).__name__,
        "formal_training_runs": 1,
        "run_id": run_id,
        "stage_key": "W-02",
        "status": "FAILED_OR_NE_NO_RERUN",
    }, target)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--formal-root", required=True)
    parser.add_argument("--workers", required=True, type=int, choices=(1, 2, 4))
    parser.add_argument("--mode", default="fresh", choices=("fresh", "resume"))
    parser.add_argument("--run-id", default=1, type=int)
    arguments = parser.parse_args()
    repository = Path(arguments.repository_root).resolve()
    formal = Path(arguments.formal_root).resolve()
    parent = read_w02_compile_freeze(repository)
    runtime = read_w02_candidate_runtime_freeze(repository)
    candidate = formal / "candidate-train"
    teacher = formal / "teacher-train"
    try:
        result = run_w02_candidate(
            repository_root=repository,
            candidate_root=candidate,
            teacher_root=teacher,
            compile_freeze_sha256=parent.sha256(),
            runtime_freeze_sha256=runtime.sha256(),
            pack_commitment=parent.pack_commitment,
            expected_guard_sha256=parent.first_run_guard_sha256,
            run_id=arguments.run_id,
            requested_workers=arguments.workers,
            mode=arguments.mode,
            budget=runtime.resource_budget,
        )
    except BaseException as error:
        _failure_seal(candidate, arguments.run_id, error)
        raise
    print(json.dumps({
        "artifact_manifest_sha256": result.artifact_manifest_sha256,
        "candidate_semantic_sha256": result.candidate_semantic_sha256,
        "formal_private_evaluation_runs": 0,
        "formal_training_runs": 1,
        "generated_probe_sha256": result.generated_probe_sha256,
        "logic_operations": result.logic_operations,
        "pair_count": result.pair_count,
        "private_payload_reads": 0,
        "run_identity_sha256": result.run_identity_sha256,
        "source_count": result.source_count,
        "status": "CANDIDATE_ARTIFACT_SEALED",
        "teacher_calls": 0,
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
