"""创建、冻结并唯一执行 W09-10 rotation private evaluator。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from pure_integer_ai.experiments.ph2_w09_contract import (
    make_w09_request,
    open_w09_frozen_contract,
)
from pure_integer_ai.experiments.ph2_w09_evaluator_contract import (
    W09_PRIVATE_AGGREGATE_NAME,
    W09_PRIVATE_FAMILY_FREEZE_NAME,
    W09_PRIVATE_FIRST_RUN_GUARD_NAME,
)
from pure_integer_ai.experiments.ph2_w09_evaluator_family import (
    build_w09_private_family_documents,
    publish_w09_private_family,
)
from pure_integer_ai.experiments.ph2_w09_evaluator_runtime import (
    W09PrivateEvaluatorRuntimeConfig,
    run_w09_private_evaluation_once,
)
from pure_integer_ai.experiments.ph2_w09_firewall import W09PayloadFirewall
from pure_integer_ai.experiments.ph2_w09_rotation import (
    build_w09_rotation_records,
    read_w09_rotation_manifest,
    validate_w09_rotation_metadata,
    write_w09_rotation_package,
)

ROOT = Path(__file__).resolve().parents[1]
_CANDIDATE_NAMES = {
    "contract": "candidate_contract_freeze.json",
    "guard": "formal_first_run_guard.json",
    "host": "candidate_host_freeze.json",
    "seal": "candidate_terminal_seal.json",
}


def _git(*args: str) -> str:
    """执行只读 Git 命令；可用进程级 W09_GIT_PROXY。"""
    command = ["git"]
    proxy = os.environ.get("W09_GIT_PROXY")
    if proxy:
        command.extend(("-c", f"http.proxy={proxy}", "-c", f"https.proxy={proxy}"))
    return subprocess.run(
        [*command, *args], cwd=ROOT, check=True, capture_output=True,
        text=True, timeout=30, env=os.environ.copy(),
    ).stdout.strip()


def _preflight_public_head() -> str:
    """要求 local/origin/live 三方一致且 worktree clean。"""
    if _git("status", "--porcelain=v1"):
        raise RuntimeError("W09 private evaluator 要求 public worktree clean")
    head = _git("rev-parse", "HEAD")
    tracking = _git("rev-parse", "origin/master")
    live = _git("ls-remote", "origin", "refs/heads/master").split()
    if len(live) != 2 or head != tracking or head != live[0]:
        raise RuntimeError("W09 private evaluator local/origin/live HEAD 不一致")
    return head


def _document(root: Path, name: str) -> tuple[dict[str, object], str]:
    """回读一个 canonical Candidate/family 文档。"""
    path = root / name
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("W09 private evaluator artifact 缺失或为链接")
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError("W09 private evaluator artifact JSON 非法") from error
    if not isinstance(value, dict) or canonical_json_bytes(value) != payload:
        raise RuntimeError("W09 private evaluator artifact 非 canonical")
    return value, hashlib.sha256(payload).hexdigest()


def _preflight_candidate(candidate: Path, *, public_head: str) -> dict[str, str]:
    """只读验证 W09 Candidate 四文档和正式 PASS 状态。"""
    loaded = {key: _document(candidate, name) for key, name in _CANDIDATE_NAMES.items()}
    values = {key: item[0] for key, item in loaded.items()}
    digests = {key: item[1] for key, item in loaded.items()}
    host = values["host"]
    seal = values["seal"]
    state = host.get("execution_state", {})
    if (
        values["contract"].get("formal_run_count") != 0
        or values["guard"].get("formal_run_count_after") != 1
        or host.get("formal_run_count") != 1
        or host.get("candidate_sealed") != 1
        or host.get("public_head_commit_sha1") != public_head
        or seal.get("terminal_state") != "PASS"
        or seal.get("candidate_host_freeze_sha256") != digests["host"]
        or state.get("W09_RUNTIME_EVIDENCED") != 1
        or state.get("formal_w09_training_runs") != 1
        or state.get("teacher_calls") != 0
        or state.get("LANGUAGE_CAPABILITY_MASTERED") != 0
        or state.get("LANGUAGE_READINESS") != 0
    ):
        raise RuntimeError("W09 Candidate PASS preflight 失败")
    return digests


def _validate_roots(candidate: Path, rotation: Path, family: Path | None = None) -> None:
    """拒绝 public/Candidate/rotation/family root 嵌套或重合。"""
    roots = [ROOT.resolve(), candidate.resolve(), rotation.resolve()]
    if family is not None:
        roots.append(family.resolve())
    if any(left == right or left.is_relative_to(right) or right.is_relative_to(left) for index, left in enumerate(roots) for right in roots[index + 1:]):
        raise RuntimeError("W09 public/Candidate/rotation/family root 未隔离")


def _create_rotation(args: argparse.Namespace) -> int:
    """从 public train-only payload 创建全新 Git 外 rotation package。"""
    candidate = args.candidate_root.resolve()
    rotation = args.rotation_root.resolve()
    if rotation.exists():
        raise RuntimeError("W09 rotation root 已存在，禁止复用")
    _validate_roots(candidate, rotation)
    head = _preflight_public_head()
    _preflight_candidate(candidate, public_head=head)
    context = open_w09_frozen_contract(ROOT)
    payload = W09PayloadFirewall.open(ROOT, context, make_w09_request(context)).read_training_payload()
    records = build_w09_rotation_records(payload)
    path, manifest_sha, manifest = write_w09_rotation_package(rotation, records)
    del path
    validate_w09_rotation_metadata(rotation, manifest)
    print(json.dumps({
        "manifest_sha256": manifest_sha,
        "observation_count": manifest.observation_identity.record_count,
        "package_commitment": manifest.package_commitment,
        "state": "ROTATION_FROZEN",
    }, ensure_ascii=False, sort_keys=True))
    return 0


def _freeze(args: argparse.Namespace) -> int:
    """在首次 private payload read 前冻结全新 family metadata。"""
    candidate = args.candidate_root.resolve()
    rotation = args.rotation_root.resolve()
    family = args.family_root.resolve()
    if family.exists():
        raise RuntimeError("W09 family root 已存在，禁止复用")
    _validate_roots(candidate, rotation, family)
    head = _preflight_public_head()
    digests = _preflight_candidate(candidate, public_head=head)
    manifest = read_w09_rotation_manifest(rotation, expected_sha256=args.rotation_manifest_sha256)
    validate_w09_rotation_metadata(rotation, manifest)
    documents = build_w09_private_family_documents(
        ROOT,
        candidate_contract_sha256=digests["contract"],
        candidate_guard_sha256=digests["guard"],
        candidate_host_sha256=digests["host"],
        candidate_seal_sha256=digests["seal"],
        evaluator_public_head_commit_sha1=head,
        rotation_manifest=manifest,
    )
    _, freeze_sha = publish_w09_private_family(
        family, documents, forbidden_roots=(ROOT, candidate, rotation),
    )
    print(json.dumps({
        "family_freeze_sha256": freeze_sha,
        "formal_run_count": 0,
        "private_payload_reads": 0,
        "rotation_package_commitment": manifest.package_commitment,
        "state": "FROZEN",
    }, ensure_ascii=False, sort_keys=True))
    return 0


def _run(args: argparse.Namespace) -> int:
    """核验 guard 未消费后唯一执行 formal evaluator。"""
    candidate = args.candidate_root.resolve()
    rotation = args.rotation_root.resolve()
    family = args.family_root.resolve()
    _validate_roots(candidate, rotation, family)
    head = _preflight_public_head()
    _preflight_candidate(candidate, public_head=head)
    freeze, freeze_sha = _document(family, W09_PRIVATE_FAMILY_FREEZE_NAME)
    if (
        freeze_sha != args.family_freeze_sha256
        or freeze.get("formal_run_count") != 0
        or freeze.get("private_payload_reads") != 0
        or (family / W09_PRIVATE_FIRST_RUN_GUARD_NAME).exists()
        or (family / "publication" / W09_PRIVATE_AGGREGATE_NAME).exists()
    ):
        raise RuntimeError("W09 family guard/formal count preflight 失败")
    result = run_w09_private_evaluation_once(
        W09PrivateEvaluatorRuntimeConfig(
            ROOT, candidate, family, family / "execution", rotation,
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
        "terminal_seal_sha256": result.terminal_seal_sha256,
    }, ensure_ascii=False, sort_keys=True))
    return 0 if result.status == "PASS" else 2


def main() -> int:
    """解析 create-rotation/freeze/run 三个严格顺序命令。"""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create-rotation")
    create.add_argument("--candidate-root", type=Path, required=True)
    create.add_argument("--rotation-root", type=Path, required=True)
    freeze = subparsers.add_parser("freeze")
    run = subparsers.add_parser("run")
    for subparser in (freeze, run):
        subparser.add_argument("--candidate-root", type=Path, required=True)
        subparser.add_argument("--rotation-root", type=Path, required=True)
        subparser.add_argument("--family-root", type=Path, required=True)
        subparser.add_argument("--family-freeze-sha256" if subparser is run else "--rotation-manifest-sha256", required=True)
    args = parser.parse_args()
    if args.command == "create-rotation":
        return _create_rotation(args)
    return _freeze(args) if args.command == "freeze" else _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
