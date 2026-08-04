"""冻结并执行唯一 W08-09 private evaluator family。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from pure_integer_ai.experiments.ph2_w08_authority import W08_DIMENSION_KEYS
from pure_integer_ai.experiments.ph2_w08_candidate import (
    W08_CANDIDATE_HOST_FREEZE_NAME,
    W08_CANDIDATE_TERMINAL_SEAL_NAME,
)
from pure_integer_ai.experiments.ph2_w08_candidate_contract import (
    W08_CANDIDATE_CONTRACT_FREEZE_NAME,
    W08_CANDIDATE_FORMAL_MODE,
    W08_CANDIDATE_FORMAL_WORKER_COUNT,
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
from pure_integer_ai.experiments.ph2_w08_inference_contract import (
    W08_CANDIDATE_INFERENCE_INTERFACE_VERSION,
)
from pure_integer_ai.experiments.ph2_w08_runtime import (
    load_w08_candidate_inference_state,
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


def _candidate_document(root: Path, name: str) -> dict[str, object]:
    path = root / name
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("W08 Candidate artifact 缺失或为链接")
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("W08 Candidate artifact 不是 canonical JSON") from error
    if not isinstance(value, dict) or canonical_json_bytes(value) != payload:
        raise RuntimeError("W08 Candidate artifact canonical identity 漂移")
    return value


def _preflight_candidate(candidate: Path, *, public_head: str) -> dict[str, str]:
    """在创建 private root 前证明 Candidate PASS 与 V2 state 可执行。"""
    names = {
        "contract": W08_CANDIDATE_CONTRACT_FREEZE_NAME,
        "guard": W08_CANDIDATE_FIRST_RUN_GUARD_NAME,
        "host": W08_CANDIDATE_HOST_FREEZE_NAME,
        "seal": W08_CANDIDATE_TERMINAL_SEAL_NAME,
    }
    values = {key: _candidate_document(candidate, name) for key, name in names.items()}
    digests = {key: _file_sha(candidate, name) for key, name in names.items()}
    host = values["host"]
    seal = values["seal"]
    interface = host.get("private_inference_interface")
    host_evidence = host.get("host_evidence")
    if (
        values["guard"].get("formal_run_count_after") != 1
        or host.get("formal_run_count") != 1
        or host.get("candidate_sealed") != 1
        or host.get("terminal_state") != "PASS"
        or host.get("public_head_commit_sha1") != public_head
        or seal.get("terminal_state") != "PASS"
        or seal.get("candidate_host_freeze_sha256") != digests["host"]
        or seal.get("candidate_contract_sha256") != digests["contract"]
        or seal.get("candidate_first_run_guard_sha256") != digests["guard"]
        or host.get("candidate_contract_sha256") != digests["contract"]
        or host.get("candidate_first_run_guard_sha256") != digests["guard"]
        or not isinstance(interface, dict)
        or not isinstance(host_evidence, dict)
        or interface != host_evidence.get("private_inference_interface")
        or interface.get("version") != W08_CANDIDATE_INFERENCE_INTERFACE_VERSION
        or interface.get("executable") != 1
        or interface.get("evaluator_label_inputs") != 0
        or interface.get("per_case_invocation_required") != 1
        or tuple(interface.get("component_keys", ())) != W08_DIMENSION_KEYS
        or seal.get("candidate_inference_state_sha256")
        != interface.get("state_commitment")
        or seal.get("candidate_inference_state_key") != interface.get("state_key")
    ):
        raise RuntimeError("W08 private family 创建前 Candidate PASS/V2 preflight 失败")
    state = load_w08_candidate_inference_state(W08RuntimeConfig(
        ROOT,
        candidate / "host",
        candidate / "host" / "coordinator.sqlite",
        worker_count=W08_CANDIDATE_FORMAL_WORKER_COUNT,
        mode=W08_CANDIDATE_FORMAL_MODE,
    ))
    if (
        state.sha256() != interface.get("state_commitment")
        or list(state.state_key) != interface.get("state_key")
        or len(state.rules) != interface.get("rule_count")
    ):
        raise RuntimeError("W08 private family 创建前 Candidate inference state 漂移")
    return digests


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
    candidate_digests = _preflight_candidate(candidate, public_head=head)
    documents = build_w08_private_family_documents(
        ROOT,
        candidate_contract_sha256=candidate_digests["contract"],
        candidate_guard_sha256=candidate_digests["guard"],
        candidate_host_sha256=candidate_digests["host"],
        candidate_seal_sha256=candidate_digests["seal"],
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
