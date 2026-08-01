"""W-04 主 owner 的公开 runtime receipt 排他发布合同。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from pure_integer_ai.experiments.ph2_w04_candidate import (
    W04_FORMAL_EXECUTION_STATE,
)
from pure_integer_ai.experiments.ph2_w04_contract import (
    W04_AGGREGATION_POLICY,
    W04_EVALUATION_ORDER,
    W04_GENERATION_HARD_CONJUNCT,
    W04_OPEN_GENERATION_STATE,
)
from pure_integer_ai.experiments.ph2_w04_evaluator_contract import (
    W04_PRIVATE_ABLATION_KEYS,
)


W04_PUBLIC_RUNTIME_RECEIPT_RELATIVE_PATH = (
    "data/ph2/manifests/d03_v1/w04_runtime_evidence_receipt_v1.json"
)
W04_PUBLIC_RUNTIME_RECEIPT_NAME = "w04_runtime_evidence_receipt_v1.json"
W04_REQUIRED_VERIFICATION_JOBS = (
    "W04 bounded local specialization",
    "W04 adjacent retention",
    "W04 compile/source guards",
    "W04 identity/secret gate",
)


class W04ReleaseError(RuntimeError):
    """W-04 receipt 输入、安全 hard conjunct 或 append-only 发布错误。"""


def _strict_sha256(value: object, *, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise W04ReleaseError(f"{label} 不是规范 SHA-256")
    return value


def _strict_sha1(value: object, *, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 40
            or any(char not in "0123456789abcdef" for char in value)):
        raise W04ReleaseError(f"{label} 不是规范 SHA-1")
    return value


def _read_canonical(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    if not path.is_file() or path.is_symlink():
        raise W04ReleaseError(f"{label} 缺失或为链接")
    payload = path.read_bytes()
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise W04ReleaseError(f"{label} JSON 非法") from exc
    if not isinstance(value, dict) or canonical_json_bytes(value) != payload:
        raise W04ReleaseError(f"{label} 非 canonical object")
    return value, payload


def _check_aggregate(path: Path, expected_sha: str) -> dict[str, Any]:
    expected = _strict_sha256(expected_sha, label="aggregate")
    aggregate, payload = _read_canonical(path, label="aggregate")
    if hashlib.sha256(payload).hexdigest() != expected:
        raise W04ReleaseError("aggregate SHA 漂移")
    required_fields = {
        "ablation_results", "artifact_kind", "case_commitment",
        "cluster_commitment", "dimension_results", "fail_count",
        "failure_phase", "family_commitment", "formal_run_count",
        "format_version", "generation_ablation_statuses", "host_writes",
        "infrastructure", "label_commitment", "label_writes", "ne_count",
        "pass_count", "payload_commitment", "status",
    }
    if (set(aggregate) != required_fields
            or aggregate["artifact_kind"]
            != "PH2_W04_PRIVATE_EVALUATION_AGGREGATE"
            or aggregate["format_version"] != 1
            or aggregate["failure_phase"] != "NONE"
            or aggregate["formal_run_count"] != 1
            or aggregate["status"] != "PASS"
            or aggregate["pass_count"] != len(W04_EVALUATION_ORDER)
            or aggregate["fail_count"] != 0
            or aggregate["ne_count"] != 0
            or aggregate["host_writes"] != 0
            or aggregate["label_writes"] != 0):
        raise W04ReleaseError("aggregate 未满足 W-04 PASS hard conjunct")
    dimensions = aggregate["dimension_results"]
    if (not isinstance(dimensions, list)
            or tuple(item.get("dimension_key") for item in dimensions)
            != W04_EVALUATION_ORDER):
        raise W04ReleaseError("aggregate 五维顺序漂移")
    for item in dimensions:
        if (set(item) != {
                "dimension_key", "evidence_commitment", "fail_count",
                "ne_count", "passed", "required", "status"}
                or item["status"] != "PASS"
                or item["passed"] != 1
                or item["required"] != 1
                or item["fail_count"] != 0
                or item["ne_count"] != 0):
            raise W04ReleaseError("aggregate dimension 非 1/1 PASS")
        _strict_sha256(item["evidence_commitment"], label="dimension evidence")
    ablations = aggregate["ablation_results"]
    if (not isinstance(ablations, list) or len(ablations) != 5
            or tuple(item.get("ablation_key") for item in ablations)
            != W04_PRIVATE_ABLATION_KEYS):
        raise W04ReleaseError("aggregate 五项 ablation 顺序漂移")
    for ordinal, item in enumerate(ablations):
        expected_statuses = [
            "FAIL" if index == ordinal else "PASS"
            for index in range(len(W04_EVALUATION_ORDER))]
        if (set(item) != {"ablation_key", "dimension_statuses"}
                or item["dimension_statuses"] != expected_statuses):
            raise W04ReleaseError("aggregate ablation 未正交击穿")
    if aggregate["generation_ablation_statuses"] != [
            "PASS", "PASS", "PASS", "PASS", "FAIL"]:
        raise W04ReleaseError("generation ablation 未击穿")
    infrastructure = aggregate["infrastructure"]
    expected_infrastructure = {
        "candidate_inventory_match": 1,
        "clone_dump_readback": 1,
        "clone_host_copy_match": 1,
        "evaluator_label_writes": 0,
        "host_copy_unchanged": 1,
        "public_repo_writes": 0,
    }
    if infrastructure != expected_infrastructure:
        raise W04ReleaseError("aggregate clone/readback/owner isolation 未闭合")
    for key in (
            "family_commitment", "payload_commitment", "case_commitment",
            "label_commitment", "cluster_commitment"):
        _strict_sha256(aggregate[key], label=key)
    forbidden = (b"surface", b"expected", b"private_path", b"message")
    if any(token in payload for token in forbidden):
        raise W04ReleaseError("aggregate 含 private 字段")
    return aggregate


def _check_recommendation(
        path: Path,
        expected_sha: str,
        aggregate: dict[str, Any],
        aggregate_sha: str,
        ) -> dict[str, Any]:
    expected = _strict_sha256(expected_sha, label="recommendation")
    recommendation, payload = _read_canonical(path, label="recommendation")
    if hashlib.sha256(payload).hexdigest() != expected:
        raise W04ReleaseError("recommendation SHA 漂移")
    if (set(recommendation) != {
            "aggregate_sha256", "artifact_kind",
            "candidate_host_freeze_sha256", "family_commitment",
            "formal_run_count", "format_version", "recommend_runtime_receipt"}
            or recommendation["artifact_kind"]
            != "PH2_W04_RUNTIME_RECEIPT_RECOMMENDATION"
            or recommendation["aggregate_sha256"] != aggregate_sha
            or recommendation["family_commitment"]
            != aggregate["family_commitment"]
            or recommendation["formal_run_count"] != 1
            or recommendation["format_version"] != 1
            or recommendation["recommend_runtime_receipt"] != 1):
        raise W04ReleaseError("recommendation binding 漂移")
    _strict_sha256(
        recommendation["candidate_host_freeze_sha256"],
        label="recommendation candidate host")
    return recommendation


def publish_w04_runtime_receipt(
        repository_root: str | Path,
        *,
        aggregate_path: str | Path,
        aggregate_sha256: str,
        recommendation_path: str | Path,
        recommendation_sha256: str,
        candidate_host_freeze_path: str | Path,
        candidate_contract_sha256: str,
        candidate_host_freeze_sha256: str,
        w03_receipt_sha256: str,
        d03_receipt_sha256: str,
        d03_global_manifest_sha256: str,
        d03_stage_manifest_sha256: str,
        pre_w04_gate_sha256: str,
        publication_commit_sha1: str,
        verification_run_id: int,
        verification_jobs: tuple[tuple[str, str], ...],
        ) -> tuple[Path, str]:
    """主 owner 只读安全摘要并 append-only 发布 W-04 receipt。"""
    root = Path(repository_root).resolve()
    aggregate = _check_aggregate(
        Path(aggregate_path).resolve(), aggregate_sha256)
    recommendation = _check_recommendation(
        Path(recommendation_path).resolve(),
        recommendation_sha256,
        aggregate,
        aggregate_sha256,
    )
    contract_sha = _strict_sha256(
        candidate_contract_sha256, label="candidate contract")
    host_sha = _strict_sha256(
        candidate_host_freeze_sha256, label="candidate host")
    host, host_bytes = _read_canonical(
        Path(candidate_host_freeze_path).resolve(),
        label="candidate host freeze")
    if (hashlib.sha256(host_bytes).hexdigest() != host_sha
            or recommendation["candidate_host_freeze_sha256"] != host_sha
            or host.get("candidate_contract_sha256") != contract_sha
            or host.get("formal_run_count") != 1
            or host.get("execution_state") != W04_FORMAL_EXECUTION_STATE
            or host.get("open_generation_state") != W04_OPEN_GENERATION_STATE
            or host.get("self_excluded") != 1):
        raise W04ReleaseError("candidate host freeze 状态漂移")
    parents = {}
    for label, value in (
            ("w03_receipt_sha256", w03_receipt_sha256),
            ("d03_receipt_sha256", d03_receipt_sha256),
            ("d03_global_manifest_sha256", d03_global_manifest_sha256),
            ("d03_stage_manifest_sha256", d03_stage_manifest_sha256),
            ("pre_w04_gate_sha256", pre_w04_gate_sha256)):
        parents[label] = _strict_sha256(value, label=label)
    commit = _strict_sha1(publication_commit_sha1, label="publication commit")
    if (type(verification_run_id) is not int or verification_run_id <= 0
            or not isinstance(verification_jobs, tuple)
            or tuple(item[0] for item in verification_jobs)
            != W04_REQUIRED_VERIFICATION_JOBS):
        raise W04ReleaseError("verification identity 非法")
    jobs = []
    for name, status in verification_jobs:
        if status != "PASS":
            raise W04ReleaseError("verification job 未 PASS")
        jobs.append({"job": name, "status": status})
    receipt = {
        "ablation_results": aggregate["ablation_results"],
        "aggregate_sha256": _strict_sha256(
            aggregate_sha256, label="aggregate"),
        "artifact_kind": "PH2_W04_RUNTIME_EVIDENCE_RECEIPT",
        "candidate_contract_sha256": contract_sha,
        "candidate_host_freeze_sha256": host_sha,
        "case_commitment": aggregate["case_commitment"],
        "cluster_commitment": aggregate["cluster_commitment"],
        "dimension_results": aggregate["dimension_results"],
        "execution_state": {
            "LANGUAGE_CAPABILITY_MASTERED": 0,
            "LANGUAGE_READINESS": 0,
            "W04_BLOCKED_FAILED": 0,
            "W04_RUNTIME_EVIDENCED": 1,
            "W04_STARTED": 1,
            "W05_STARTED": 0,
            "formal_w04_training_runs": 1,
            "teacher_calls": 0,
        },
        "family_commitment": aggregate["family_commitment"],
        "format_version": 1,
        "generation_ablation_statuses": aggregate[
            "generation_ablation_statuses"],
        "generation_hard_conjunct": W04_GENERATION_HARD_CONJUNCT,
        "hard_conjunct_policy": W04_AGGREGATION_POLICY,
        "infrastructure": aggregate["infrastructure"],
        "label_commitment": aggregate["label_commitment"],
        "open_generation_state": W04_OPEN_GENERATION_STATE,
        "parent_identities": parents,
        "payload_commitment": aggregate["payload_commitment"],
        "publication": {
            "commit_sha1": commit,
            "verification_jobs": jobs,
            "verification_run_id": verification_run_id,
        },
        "receipt_relative_path": W04_PUBLIC_RUNTIME_RECEIPT_RELATIVE_PATH,
        "receipt_self_excluded": 1,
        "recommendation_sha256": _strict_sha256(
            recommendation_sha256, label="recommendation"),
        "required": 1,
        "stage_key": "W-04",
        "status": "RUNTIME_EVIDENCED",
    }
    encoded = canonical_json_bytes(receipt)
    target = root / Path(*W04_PUBLIC_RUNTIME_RECEIPT_RELATIVE_PATH.split("/"))
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(encoded)
    except FileExistsError as exc:
        raise W04ReleaseError("W-04 runtime receipt 不可覆盖") from exc
    return target, hashlib.sha256(encoded).hexdigest()


__all__ = [
    "W04_PUBLIC_RUNTIME_RECEIPT_NAME",
    "W04_PUBLIC_RUNTIME_RECEIPT_RELATIVE_PATH",
    "W04_REQUIRED_VERIFICATION_JOBS",
    "W04ReleaseError",
    "publish_w04_runtime_receipt",
]
