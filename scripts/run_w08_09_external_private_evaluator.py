"""分离冻结并唯一执行 W08 external private evaluator family。"""
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
    W08_CANDIDATE_FIRST_RUN_GUARD_NAME,
    W08_CANDIDATE_FORMAL_MODE,
    W08_CANDIDATE_FORMAL_WORKER_COUNT,
)
from pure_integer_ai.experiments.ph2_w08_evaluator_contract import (
    W08_PRIVATE_AGGREGATE_NAME,
    W08_PRIVATE_FAMILY_FREEZE_NAME,
    W08_PRIVATE_FIRST_RUN_GUARD_NAME,
)
from pure_integer_ai.experiments.ph2_w08_evaluator_family import (
    build_w08_private_family_documents,
    publish_w08_private_family,
)
from pure_integer_ai.experiments.ph2_w08_evaluator_runtime import (
    W08PrivateEvaluatorRuntimeConfig,
    run_w08_private_evaluation_once,
)
from pure_integer_ai.experiments.ph2_w08_external_package import (
    read_w08_external_private_manifest,
    validate_w08_external_private_package_metadata,
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
        raise RuntimeError("W08 external evaluator 要求 public worktree clean")
    head = _git("rev-parse", "HEAD")
    tracking = _git("rev-parse", "origin/master")
    live = _git("ls-remote", "origin", "refs/heads/master").split()
    if len(live) != 2 or head != tracking or head != live[0]:
        raise RuntimeError("W08 external evaluator local/origin/live HEAD 不一致")
    return head


def _document(root: Path, name: str) -> tuple[dict[str, object], str]:
    path = root / name
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("W08 external evaluator artifact 缺失或为链接")
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError("W08 external evaluator artifact 不是 canonical JSON") from error
    if not isinstance(value, dict) or canonical_json_bytes(value) != payload:
        raise RuntimeError("W08 external evaluator artifact canonical identity 漂移")
    return value, hashlib.sha256(payload).hexdigest()


def _preflight_candidate(candidate: Path, *, public_head: str) -> dict[str, str]:
    names = {
        "contract": W08_CANDIDATE_CONTRACT_FREEZE_NAME,
        "guard": W08_CANDIDATE_FIRST_RUN_GUARD_NAME,
        "host": W08_CANDIDATE_HOST_FREEZE_NAME,
        "seal": W08_CANDIDATE_TERMINAL_SEAL_NAME,
    }
    loaded = {key: _document(candidate, name) for key, name in names.items()}
    values = {key: item[0] for key, item in loaded.items()}
    digests = {key: item[1] for key, item in loaded.items()}
    host = values["host"]
    seal = values["seal"]
    interface = host.get("private_inference_interface")
    evidence = host.get("host_evidence")
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
        or not isinstance(evidence, dict)
        or interface != evidence.get("private_inference_interface")
        or interface.get("version") != W08_CANDIDATE_INFERENCE_INTERFACE_VERSION
        or interface.get("executable") != 1
        or interface.get("evaluator_label_inputs") != 0
        or interface.get("per_case_invocation_required") != 1
        or tuple(interface.get("component_keys", ())) != W08_DIMENSION_KEYS
        or seal.get("candidate_inference_state_sha256")
        != interface.get("state_commitment")
        or seal.get("candidate_inference_state_key") != interface.get("state_key")
    ):
        raise RuntimeError("W08 external family Candidate PASS/V2 preflight 失败")
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
        raise RuntimeError("W08 external family Candidate inference state 漂移")
    return digests


def _validate_roots(candidate: Path, payload: Path, family: Path) -> None:
    roots = (ROOT.resolve(), candidate.resolve(), payload.resolve(), family.resolve())
    if any(
        left == right or left.is_relative_to(right) or right.is_relative_to(left)
        for index, left in enumerate(roots)
        for right in roots[index + 1:]
    ):
        raise RuntimeError("W08 external public/Candidate/payload/family root 未隔离")


def _freeze(args: argparse.Namespace) -> int:
    candidate = args.candidate_root.resolve()
    payload = args.private_payload_root.resolve()
    family = args.family_root.resolve()
    if family.exists():
        raise RuntimeError("W08 external family root 已存在，禁止复用")
    _validate_roots(candidate, payload, family)
    head = _preflight_public_head()
    candidate_digests = _preflight_candidate(candidate, public_head=head)
    manifest = read_w08_external_private_manifest(payload, args.package_manifest)
    validate_w08_external_private_package_metadata(payload, manifest)
    documents = build_w08_private_family_documents(
        ROOT,
        candidate_contract_sha256=candidate_digests["contract"],
        candidate_guard_sha256=candidate_digests["guard"],
        candidate_host_sha256=candidate_digests["host"],
        candidate_seal_sha256=candidate_digests["seal"],
        evaluator_public_head_commit_sha1=head,
        nonce=(8, 53, 89, 131),
        external_package_manifest=manifest,
    )
    _, freeze_sha = publish_w08_private_family(
        family,
        documents,
        forbidden_roots=(ROOT, candidate, payload),
    )
    print(json.dumps({
        "family_freeze_sha256": freeze_sha,
        "formal_run_count": 0,
        "package_commitment": manifest.package_commitment,
        "package_manifest_sha256": manifest.sha256(),
        "private_payload_reads": 0,
        "state": "FROZEN",
    }, ensure_ascii=False, sort_keys=True))
    return 0


def _run(args: argparse.Namespace) -> int:
    candidate = args.candidate_root.resolve()
    payload = args.private_payload_root.resolve()
    family = args.family_root.resolve()
    _validate_roots(candidate, payload, family)
    head = _preflight_public_head()
    _preflight_candidate(candidate, public_head=head)
    freeze, freeze_sha = _document(family, W08_PRIVATE_FAMILY_FREEZE_NAME)
    if (
        freeze_sha != args.family_freeze_sha256
        or freeze.get("formal_run_count") != 0
        or freeze.get("private_payload_reads") != 0
        or (family / W08_PRIVATE_FIRST_RUN_GUARD_NAME).exists()
        or (family / "publication" / W08_PRIVATE_AGGREGATE_NAME).exists()
    ):
        raise RuntimeError("W08 external family formal run count/guard preflight 失败")
    result = run_w08_private_evaluation_once(
        W08PrivateEvaluatorRuntimeConfig(
            ROOT,
            candidate,
            family,
            family / "execution",
            private_payload_root=payload,
            package_manifest=args.package_manifest,
        ),
        family_freeze_sha256=freeze_sha,
    )
    print(json.dumps({
        "aggregate_sha256": result.aggregate_sha256,
        "family_freeze_sha256": result.family_freeze_sha256,
        "first_run_guard_sha256": result.first_run_guard_sha256,
        "formal_run_count": 1,
        "private_dump_sha256": result.dump_sha256,
        "recommendation_sha256": result.recommendation_sha256,
        "status": result.status,
    }, ensure_ascii=False, sort_keys=True))
    return 0 if result.status == "PASS" else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("freeze", "run"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--candidate-root", type=Path, required=True)
        subparser.add_argument("--private-payload-root", type=Path, required=True)
        subparser.add_argument("--package-manifest", type=Path, required=True)
        subparser.add_argument("--family-root", type=Path, required=True)
        if command == "run":
            subparser.add_argument("--family-freeze-sha256", required=True)
    args = parser.parse_args()
    return _freeze(args) if args.command == "freeze" else _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
