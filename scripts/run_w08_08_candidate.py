"""在全新 Git 外 root 执行唯一 W08 Candidate 正式运行。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

from pure_integer_ai.experiments.ph2_w08_candidate import execute_w08_candidate_once
from pure_integer_ai.experiments.ph2_w08_candidate_contract import (
    W08_CANDIDATE_FORMAL_MODE,
    W08_CANDIDATE_FORMAL_WORKER_COUNT,
    build_w08_candidate_contract,
    publish_w08_candidate_contract_freeze,
)
from pure_integer_ai.experiments.ph2_w08_runtime_contract import W08RuntimeConfig


ROOT = Path(__file__).resolve().parents[1]


def _git(*args: str) -> str:
    command = ["git"]
    proxy = os.environ.get("W08_GIT_PROXY")
    if proxy:
        command.extend((
            "-c", f"http.proxy={proxy}",
            "-c", f"https.proxy={proxy}",
        ))
    return subprocess.run(
        [*command, *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
        env=os.environ.copy(),
    ).stdout.strip()


def _preflight_public_head() -> str:
    status = _git("status", "--porcelain=v1")
    if status:
        raise RuntimeError("W08 Candidate 要求 public worktree clean")
    head = _git("rev-parse", "HEAD")
    tracking = _git("rev-parse", "origin/master")
    live = _git("ls-remote", "origin", "refs/heads/master").split()
    if len(live) != 2 or head != tracking or head != live[0]:
        raise RuntimeError("W08 Candidate local/origin/live HEAD 不一致")
    return head


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", type=Path, required=True)
    args = parser.parse_args()
    candidate_root = args.candidate_root.resolve()
    if candidate_root.exists():
        raise RuntimeError("W08 Candidate root 已存在，禁止复用")
    head = _preflight_public_head()
    contract = build_w08_candidate_contract(
        ROOT,
        current_public_head_commit_sha1=head,
    )
    freeze_path, freeze_sha = publish_w08_candidate_contract_freeze(
        ROOT,
        candidate_root,
        contract,
    )
    run_root = candidate_root / "host"
    config = W08RuntimeConfig(
        ROOT,
        run_root,
        run_root / "coordinator.sqlite",
        worker_count=W08_CANDIDATE_FORMAL_WORKER_COUNT,
        mode=W08_CANDIDATE_FORMAL_MODE,
    )
    outcome, readback, host_path, host_sha, guard_path, guard_sha, seal_path, seal_sha = (
        execute_w08_candidate_once(
            ROOT,
            candidate_root,
            config=config,
            contract=contract,
            candidate_contract_sha256=freeze_sha,
        )
    )
    print(json.dumps({
        "candidate_contract_sha256": freeze_sha,
        "candidate_first_run_guard_sha256": guard_sha,
        "candidate_host_freeze_sha256": host_sha,
        "candidate_terminal_seal_sha256": seal_sha,
        "compiled_artifact_count": outcome.compiled_artifact_count,
        "dump_readback_payload_gets": readback.payload_gets_this_call,
        "formal_run_count": 1,
        "future_payload_reads": outcome.future_payload_reads,
        "open_generation_state": outcome.open_generation_state,
        "paths_exist": all(path.is_file() for path in (
            freeze_path, guard_path, host_path, seal_path
        )),
        "public_head_commit_sha1": head,
        "status": "PASS",
        "teacher_calls": outcome.teacher_calls,
        "use_count": len(outcome.uses),
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
