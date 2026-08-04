"""冻结并执行唯一 W08-09 private evaluator family。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from pure_integer_ai.experiments.ph2_w08_candidate import (
    W08_CANDIDATE_HOST_FREEZE_NAME,
    W08_CANDIDATE_TERMINAL_SEAL_NAME,
)
from pure_integer_ai.experiments.ph2_w08_candidate_contract import (
    W08_CANDIDATE_CONTRACT_FREEZE_NAME,
    W08_CANDIDATE_FIRST_RUN_GUARD_NAME,
)
from pure_integer_ai.experiments.ph2_w08_evaluator_family import (
    build_w08_private_family_documents,
    publish_w08_private_family,
)
from pure_integer_ai.experiments.ph2_w08_evaluator_runtime import (
    W08PrivateEvaluatorRuntimeConfig,
    run_w08_private_evaluation_once,
)


ROOT = Path(__file__).resolve().parents[1]


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
        env=os.environ.copy(),
    ).stdout.strip()


def _preflight_public_head() -> str:
    if _git("status", "--porcelain=v1"):
        raise RuntimeError("W08 private evaluator 要求 public worktree clean")
    head = _git("rev-parse", "HEAD")
    tracking = _git("rev-parse", "origin/master")
    live = _git("ls-remote", "origin", "refs/heads/master").split()
    if len(live) != 2 or head != tracking or head != live[0]:
        raise RuntimeError("W08 private evaluator local/origin/live HEAD 不一致")
    return head


def _file_sha(root: Path, name: str) -> str:
    path = root / name
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("W08 Candidate artifact 缺失或为链接")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--private-root", type=Path, required=True)
    args = parser.parse_args()
    candidate = args.candidate_root.resolve()
    private = args.private_root.resolve()
    if private.exists():
        raise RuntimeError("W08 private root 已存在，禁止复用")
    head = _preflight_public_head()
    documents = build_w08_private_family_documents(
        ROOT,
        candidate_contract_sha256=_file_sha(
            candidate, W08_CANDIDATE_CONTRACT_FREEZE_NAME
        ),
        candidate_guard_sha256=_file_sha(
            candidate, W08_CANDIDATE_FIRST_RUN_GUARD_NAME
        ),
        candidate_host_sha256=_file_sha(candidate, W08_CANDIDATE_HOST_FREEZE_NAME),
        candidate_seal_sha256=_file_sha(
            candidate, W08_CANDIDATE_TERMINAL_SEAL_NAME
        ),
        evaluator_public_head_commit_sha1=head,
    )
    _, freeze_sha = publish_w08_private_family(
        private,
        documents,
        forbidden_roots=(ROOT, candidate),
    )
    result = run_w08_private_evaluation_once(
        W08PrivateEvaluatorRuntimeConfig(
            ROOT,
            candidate,
            private,
            private / "execution",
        ),
        family_freeze_sha256=freeze_sha,
    )
    print(json.dumps({
        "aggregate_sha256": result.aggregate_sha256,
        "candidate_host_freeze_sha256": documents.candidate_host_sha256,
        "family_freeze_sha256": result.family_freeze_sha256,
        "first_run_guard_sha256": result.first_run_guard_sha256,
        "formal_run_count": 1,
        "private_dump_sha256": result.dump_sha256,
        "recommendation_sha256": result.recommendation_sha256,
        "status": result.status,
    }, ensure_ascii=False, sort_keys=True))
    return 0 if result.status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
