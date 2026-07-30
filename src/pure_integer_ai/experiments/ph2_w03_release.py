"""W-03 主 owner 的公开安全 runtime receipt 排他发布合同。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from pure_integer_ai.experiments.ph2_w03_candidate import (
    W03_FORMAL_EXECUTION_STATE,
)
from pure_integer_ai.experiments.ph2_w03_contract import (
    W03_ABLATION_KEYS,
    W03_AGGREGATION_POLICY,
    W03_EVALUATION_ORDER,
)


W03_PUBLIC_RUNTIME_RECEIPT_RELATIVE_PATH = (
    "data/ph2/manifests/d03_v1/w03_runtime_evidence_receipt_v1.json"
)
W03_PUBLIC_RUNTIME_RECEIPT_NAME = "w03_runtime_evidence_receipt_v1.json"
W03_REQUIRED_CI_JOBS = (
    "Python 3.11 on ubuntu-latest",
    "Python 3.14 on ubuntu-latest",
    "Python 3.14 on windows-latest",
    "Secret scan",
)


class W03ReleaseError(RuntimeError):
    """W-03 安全 receipt 的输入或发布边界错误。"""


def _strict_sha256(value: object, *, label: str) -> str:
    """验证规范小写 SHA-256。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(item not in "0123456789abcdef" for item in value)):
        raise W03ReleaseError(f"{label} 不是规范 SHA-256")
    return value


def _strict_sha1(value: object, *, label: str) -> str:
    """验证规范小写 Git SHA-1。"""
    if (not isinstance(value, str) or len(value) != 40
            or any(item not in "0123456789abcdef" for item in value)):
        raise W03ReleaseError(f"{label} 不是规范 SHA-1")
    return value


def _read_canonical(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    """读取无链接 canonical JSON object。"""
    if not path.is_file() or path.is_symlink():
        raise W03ReleaseError(f"{label} 文件缺失或为链接")
    payload = path.read_bytes()
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise W03ReleaseError(f"{label} JSON 无法解析") from exc
    if not isinstance(value, dict) or canonical_json_bytes(value) != payload:
        raise W03ReleaseError(f"{label} JSON 非 canonical object")
    return value, payload


def _check_aggregate(
        aggregate_path: Path,
        aggregate_sha256: str,
        ) -> dict[str, Any]:
    """只接受已通过五维 hard conjunct 的安全 aggregate。"""
    expected_sha = _strict_sha256(aggregate_sha256, label="aggregate")
    aggregate, payload = _read_canonical(aggregate_path, label="aggregate")
    if hashlib.sha256(payload).hexdigest() != expected_sha:
        raise W03ReleaseError("aggregate SHA 漂移")
    expected_fields = {
        "ablation_results", "artifact_kind", "case_commitment",
        "cluster_commitment", "dimension_results", "fail_count",
        "failure_phase", "family_commitment", "formal_run_count",
        "format_version", "generation_ablation_statuses", "host_writes",
        "infrastructure", "label_commitment", "label_writes", "ne_count",
        "pass_count", "payload_commitment", "status",
    }
    if (set(aggregate) != expected_fields
            or aggregate["artifact_kind"]
            != "PH2_W03_PRIVATE_EVALUATION_AGGREGATE"
            or aggregate["format_version"] != 1
            or aggregate["failure_phase"] != "NONE"
            or aggregate["formal_run_count"] != 1
            or aggregate["status"] != "PASS"
            or aggregate["pass_count"] != len(W03_EVALUATION_ORDER)
            or aggregate["fail_count"] != 0
            or aggregate["ne_count"] != 0
            or aggregate["host_writes"] != 0
            or aggregate["label_writes"] != 0):
        raise W03ReleaseError("aggregate 未满足 W-03 PASS hard conjunct")
    for key in ("family_commitment", "payload_commitment", "case_commitment",
                "label_commitment", "cluster_commitment"):
        _strict_sha256(aggregate[key], label=key)
    dimensions = aggregate["dimension_results"]
    if (not isinstance(dimensions, list)
            or tuple(item.get("dimension_key") for item in dimensions)
            != W03_EVALUATION_ORDER):
        raise W03ReleaseError("aggregate 五维顺序漂移")
    for item in dimensions:
        if (set(item) != {
                "dimension_key", "evidence_commitment", "fail_count",
                "ne_count", "passed", "required", "status"}
                or item["status"] != "PASS"
                or item["passed"] != 1
                or item["required"] != 1
                or item["fail_count"] != 0
                or item["ne_count"] != 0):
            raise W03ReleaseError("aggregate dimension 非 1/1 PASS")
        _strict_sha256(item["evidence_commitment"], label="evidence")
    ablations = aggregate["ablation_results"]
    if (not isinstance(ablations, list) or len(ablations) != 4
            or tuple(item.get("ablation_key") for item in ablations)
            != W03_ABLATION_KEYS):
        raise W03ReleaseError("aggregate bearing ablation 顺序漂移")
    for ordinal, item in enumerate(ablations):
        expected = [
            "FAIL" if index == ordinal else "PASS"
            for index in range(len(W03_EVALUATION_ORDER))]
        if (set(item) != {"ablation_key", "dimension_statuses"}
                or item["dimension_statuses"] != expected):
            raise W03ReleaseError("aggregate bearing ablation 未正交击穿")
    if aggregate["generation_ablation_statuses"] != [
            "PASS", "PASS", "PASS", "PASS", "FAIL"]:
        raise W03ReleaseError("aggregate generation ablation 未击穿")
    infrastructure = aggregate["infrastructure"]
    if (set(infrastructure) != {
            "candidate_inventory_match", "clone_dump_readback",
            "clone_host_copy_match", "host_copy_unchanged", "label_writes",
            "restore_learning_writes"}
            or infrastructure != {
                "candidate_inventory_match": 1,
                "clone_dump_readback": 1,
                "clone_host_copy_match": 1,
                "host_copy_unchanged": 1,
                "label_writes": 0,
                "restore_learning_writes": 0,
            }):
        raise W03ReleaseError("aggregate infrastructure 零写或 readback 未通过")
    forbidden = (b"surface", b"expected", b"private_path", b"exception", b"message")
    if any(item in payload for item in forbidden):
        raise W03ReleaseError("aggregate 含不安全文本")
    return aggregate


def _check_recommendation(
        recommendation_path: Path,
        recommendation_sha256: str,
        aggregate: dict[str, Any],
        aggregate_sha256: str,
        ) -> dict[str, Any]:
    """验证 evaluator recommendation 只绑定安全 aggregate commitment。"""
    expected_sha = _strict_sha256(recommendation_sha256, label="recommendation")
    recommendation, payload = _read_canonical(
        recommendation_path, label="recommendation")
    if hashlib.sha256(payload).hexdigest() != expected_sha:
        raise W03ReleaseError("recommendation SHA 漂移")
    if (set(recommendation) != {
            "aggregate_sha256", "artifact_kind", "candidate_host_freeze_sha256",
            "family_commitment", "formal_run_count", "format_version",
            "recommend_runtime_receipt"}
            or recommendation["artifact_kind"]
            != "PH2_W03_RUNTIME_RECEIPT_RECOMMENDATION"
            or recommendation["aggregate_sha256"] != aggregate_sha256
            or recommendation["family_commitment"]
            != aggregate["family_commitment"]
            or recommendation["formal_run_count"] != 1
            or recommendation["format_version"] != 1
            or recommendation["recommend_runtime_receipt"] != 1):
        raise W03ReleaseError("recommendation binding 漂移")
    _strict_sha256(
        recommendation["candidate_host_freeze_sha256"],
        label="recommendation candidate host")
    return recommendation


def publish_w03_runtime_receipt(
        repository_root: str | Path,
        *,
        aggregate_path: str | Path,
        aggregate_sha256: str,
        recommendation_path: str | Path,
        recommendation_sha256: str,
        candidate_host_freeze_path: str | Path,
        candidate_contract_sha256: str,
        candidate_host_freeze_sha256: str,
        w02_receipt_sha256: str,
        d03_receipt_sha256: str,
        d03_global_manifest_sha256: str,
        d03_stage_manifest_sha256: str,
        publication_commit_sha1: str,
        publication_ci_run_id: int,
        publication_ci_jobs: tuple[tuple[str, str], ...],
        ) -> tuple[Path, str]:
    """主 owner 读取安全摘要并以 xb 发布不可覆盖 W-03 receipt。"""
    root = Path(repository_root).resolve()
    aggregate_file = Path(aggregate_path).resolve()
    recommendation_file = Path(recommendation_path).resolve()
    host_file = Path(candidate_host_freeze_path).resolve()
    aggregate = _check_aggregate(aggregate_file, aggregate_sha256)
    candidate_host_sha = _strict_sha256(
        candidate_host_freeze_sha256, label="candidate host freeze")
    recommendation = _check_recommendation(
        recommendation_file,
        recommendation_sha256,
        aggregate,
        aggregate_sha256,
    )
    if recommendation["candidate_host_freeze_sha256"] != candidate_host_sha:
        raise W03ReleaseError("recommendation candidate host binding 漂移")
    candidate_contract_sha = _strict_sha256(
        candidate_contract_sha256, label="candidate contract")
    if (not host_file.is_file() or host_file.is_symlink()
            or hashlib.sha256(host_file.read_bytes()).hexdigest() != candidate_host_sha):
        raise W03ReleaseError("candidate host freeze identity 漂移")
    host, _ = _read_canonical(host_file, label="candidate host freeze")
    if (host.get("candidate_contract_sha256") != candidate_contract_sha
            or host.get("formal_run_count") != 1
            or host.get("execution_state") != W03_FORMAL_EXECUTION_STATE
            or host.get("self_excluded") != 1):
        raise W03ReleaseError("candidate host freeze 状态漂移")
    for label, value in (
            ("W-02 receipt", w02_receipt_sha256),
            ("D-03 receipt", d03_receipt_sha256),
            ("D-03 global manifest", d03_global_manifest_sha256),
            ("D-03 stage manifest", d03_stage_manifest_sha256)):
        _strict_sha256(value, label=label)
    commit = _strict_sha1(publication_commit_sha1, label="publication commit")
    if (type(publication_ci_run_id) is not int or publication_ci_run_id <= 0
            or not isinstance(publication_ci_jobs, tuple)
            or tuple(item[0] for item in publication_ci_jobs)
            != W03_REQUIRED_CI_JOBS):
        raise W03ReleaseError("publication CI identity 非法")
    ci_jobs = []
    for name, status in publication_ci_jobs:
        if (not isinstance(name, str) or not name
                or status != "success"):
            raise W03ReleaseError("publication CI 未四项成功")
        ci_jobs.append({"job": name, "conclusion": status})
    receipt = {
        "ablation_results": aggregate["ablation_results"],
        "aggregate_sha256": _strict_sha256(aggregate_sha256, label="aggregate"),
        "artifact_kind": "PH2_W03_RUNTIME_EVIDENCE_RECEIPT",
        "candidate_contract_sha256": candidate_contract_sha,
        "candidate_host_freeze_sha256": candidate_host_sha,
        "case_commitment": aggregate["case_commitment"],
        "cluster_commitment": aggregate["cluster_commitment"],
        "dimension_results": aggregate["dimension_results"],
        "d03_global_manifest_sha256": d03_global_manifest_sha256,
        "d03_receipt_sha256": d03_receipt_sha256,
        "d03_stage_manifest_sha256": d03_stage_manifest_sha256,
        "execution_state": {
            "LANGUAGE_CAPABILITY_MASTERED": 0,
            "LANGUAGE_READINESS": 0,
            "W03_BLOCKED_FAILED": 0,
            "W03_RUNTIME_EVIDENCED": 1,
            "W03_STARTED": 1,
            "W04_STARTED": 0,
            "formal_w03_training_runs": 1,
            "teacher_calls": 0,
        },
        "family_commitment": aggregate["family_commitment"],
        "format_version": 1,
        "generation_ablation_statuses": aggregate[
            "generation_ablation_statuses"],
        "hard_conjunct_policy": W03_AGGREGATION_POLICY,
        "infrastructure": aggregate["infrastructure"],
        "label_commitment": aggregate["label_commitment"],
        "payload_commitment": aggregate["payload_commitment"],
        "publication": {
            "ci_jobs": ci_jobs,
            "ci_run_id": publication_ci_run_id,
            "commit_sha1": commit,
        },
        "receipt_relative_path": W03_PUBLIC_RUNTIME_RECEIPT_RELATIVE_PATH,
        "receipt_self_excluded": 1,
        "recommendation_sha256": _strict_sha256(
            recommendation_sha256, label="recommendation"),
        "required": 1,
        "status": "RUNTIME_EVIDENCED",
        "stage_key": "W-03",
        "w02_receipt_sha256": w02_receipt_sha256,
    }
    encoded = canonical_json_bytes(receipt)
    target = root / Path(*W03_PUBLIC_RUNTIME_RECEIPT_RELATIVE_PATH.split("/"))
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(encoded)
    except FileExistsError as exc:
        raise W03ReleaseError("W-03 runtime receipt 不可覆盖") from exc
    return target, hashlib.sha256(encoded).hexdigest()


__all__ = [
    "W03_PUBLIC_RUNTIME_RECEIPT_NAME",
    "W03_PUBLIC_RUNTIME_RECEIPT_RELATIVE_PATH",
    "W03_REQUIRED_CI_JOBS",
    "W03ReleaseError",
    "publish_w03_runtime_receipt",
]
