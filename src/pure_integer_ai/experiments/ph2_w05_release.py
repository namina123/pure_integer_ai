"""W-05 主 owner 的公开 runtime receipt 排他发布合同。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from pure_integer_ai.experiments.ph2_w05_candidate import (
    W05_FORMAL_EXECUTION_STATE,
)
from pure_integer_ai.experiments.ph2_w05_contract import (
    W05_AGGREGATION_POLICY,
    W05_ATOMIC_PACK_SHA256,
    W05_D03_GLOBAL_MANIFEST_SHA256,
    W05_EVALUATION_ORDER,
    W05_GENERATION_HARD_CONJUNCT,
    W05_INVALIDATION_GRAPH_SHA256,
    W05_LC16_DIRECTIONAL_SHA256,
    W05_LC16_MAPPER_SHA256,
    W05_LC16_OVERLAY_SHA256,
    W05_LC16_PROJECTION_SHA256,
    W05_OPEN_GENERATION_STATE,
    W05_PRE_W04_GATE_SHA256,
    W05_PRIVATE_ABLATION_KEYS,
    W05_RESOURCE_BUDGET,
    W05_STAGE_MANIFEST_SHA256,
    W05_W04_RECEIPT_SHA256,
)


W05_PUBLIC_RUNTIME_RECEIPT_RELATIVE_PATH = (
    "data/ph2/manifests/d03_v1/w05_runtime_evidence_receipt_v1.json"
)
W05_PUBLIC_RUNTIME_RECEIPT_NAME = "w05_runtime_evidence_receipt_v1.json"
W05_REQUIRED_VERIFICATION_JOBS = (
    "W05 bounded local specialization",
    "W05 adjacent retention",
    "W05 compile/source guards",
    "W05 identity/secret gate",
)
W05_D03_RECEIPT_SHA256 = (
    "8efd5f8c559bb22f0d2587fea4d38ee94d2dc10cf13ca0f787f3489f45847aef"
)
W05_EXPECTED_PARENT_IDENTITIES = {
    "atomic_pack_sha256": W05_ATOMIC_PACK_SHA256,
    "d03_global_manifest_sha256": W05_D03_GLOBAL_MANIFEST_SHA256,
    "d03_receipt_sha256": W05_D03_RECEIPT_SHA256,
    "d03_stage_manifest_sha256": W05_STAGE_MANIFEST_SHA256,
    "invalidation_graph_sha256": W05_INVALIDATION_GRAPH_SHA256,
    "lc16_directional_sha256": W05_LC16_DIRECTIONAL_SHA256,
    "lc16_mapper_sha256": W05_LC16_MAPPER_SHA256,
    "lc16_overlay_sha256": W05_LC16_OVERLAY_SHA256,
    "lc16_projection_sha256": W05_LC16_PROJECTION_SHA256,
    "pre_w04_gate_sha256": W05_PRE_W04_GATE_SHA256,
    "w04_receipt_sha256": W05_W04_RECEIPT_SHA256,
}
W05_RECEIPT_EXECUTION_STATE = {
    "LANGUAGE_CAPABILITY_MASTERED": 0,
    "LANGUAGE_READINESS": 0,
    "W05_BLOCKED_FAILED": 0,
    "W05_RUNTIME_EVIDENCED": 1,
    "W05_STARTED": 1,
    "W06_STARTED": 0,
    "formal_w05_training_runs": 1,
    "teacher_calls": 0,
}


class W05ReleaseError(RuntimeError):
    """W-05 receipt 输入、安全 hard conjunct 或 append-only 发布错误。"""


def _strict_sha256(value: object, *, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise W05ReleaseError(f"{label} 不是规范 SHA-256")
    return value


def _strict_sha1(value: object, *, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 40
            or any(char not in "0123456789abcdef" for char in value)):
        raise W05ReleaseError(f"{label} 不是规范 SHA-1")
    return value


def _read_canonical(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    if not path.is_file() or path.is_symlink():
        raise W05ReleaseError(f"{label} 缺失或为链接")
    payload = path.read_bytes()
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise W05ReleaseError(f"{label} JSON 非法") from exc
    if not isinstance(value, dict) or canonical_json_bytes(value) != payload:
        raise W05ReleaseError(f"{label} 非 canonical object")
    return value, payload


def _check_aggregate(path: Path, expected_sha: str) -> dict[str, Any]:
    expected = _strict_sha256(expected_sha, label="aggregate")
    aggregate, payload = _read_canonical(path, label="aggregate")
    if hashlib.sha256(payload).hexdigest() != expected:
        raise W05ReleaseError("aggregate SHA 漂移")
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
            != "PH2_W05_PRIVATE_EVALUATION_AGGREGATE"
            or aggregate["format_version"] != 1
            or aggregate["failure_phase"] != "NONE"
            or aggregate["formal_run_count"] != 1
            or aggregate["status"] != "PASS"
            or aggregate["pass_count"] != len(W05_EVALUATION_ORDER)
            or aggregate["fail_count"] != 0
            or aggregate["ne_count"] != 0
            or aggregate["host_writes"] != 0
            or aggregate["label_writes"] != 0):
        raise W05ReleaseError("aggregate 未满足 W-05 PASS hard conjunct")
    dimensions = aggregate["dimension_results"]
    if (not isinstance(dimensions, list)
            or tuple(item.get("dimension_key") for item in dimensions)
            != W05_EVALUATION_ORDER):
        raise W05ReleaseError("aggregate 五维顺序漂移")
    for item in dimensions:
        if (set(item) != {
                "dimension_key", "evidence_commitment", "fail_count",
                "ne_count", "passed", "required", "status"}
                or item["status"] != "PASS"
                or item["passed"] != 1
                or item["required"] != 1
                or item["fail_count"] != 0
                or item["ne_count"] != 0):
            raise W05ReleaseError("aggregate dimension 非 1/1 PASS")
        _strict_sha256(item["evidence_commitment"], label="dimension evidence")
    ablations = aggregate["ablation_results"]
    if (not isinstance(ablations, list) or len(ablations) != 5
            or tuple(item.get("ablation_key") for item in ablations)
            != W05_PRIVATE_ABLATION_KEYS):
        raise W05ReleaseError("aggregate 五项 ablation 顺序漂移")
    for ordinal, item in enumerate(ablations):
        expected_statuses = [
            "FAIL" if index == ordinal else "PASS"
            for index in range(len(W05_EVALUATION_ORDER))]
        if (set(item) != {"ablation_key", "dimension_statuses"}
                or item["dimension_statuses"] != expected_statuses):
            raise W05ReleaseError("aggregate ablation 未正交击穿")
    if aggregate["generation_ablation_statuses"] != [
            "PASS", "PASS", "PASS", "PASS", "FAIL"]:
        raise W05ReleaseError("generation ablation 未击穿")
    infrastructure = aggregate["infrastructure"]
    expected_infrastructure = {
        "candidate_inventory_match": 1,
        "carrier_projection_count": 9,
        "carrier_scope_digest_match": 1,
        "clone_dump_readback": 1,
        "clone_host_copy_match": 1,
        "evaluator_label_writes": 0,
        "host_copy_unchanged": 1,
        "public_repo_writes": 0,
        "role_proposition_scope_cell_count": 27,
    }
    if infrastructure != expected_infrastructure:
        raise W05ReleaseError("aggregate clone/readback/owner isolation 未闭合")
    for key in (
            "family_commitment", "payload_commitment", "case_commitment",
            "label_commitment", "cluster_commitment"):
        _strict_sha256(aggregate[key], label=key)
    forbidden = (b"surface", b"expected", b"private_path", b"message")
    if any(token in payload for token in forbidden):
        raise W05ReleaseError("aggregate 含 private 字段")
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
        raise W05ReleaseError("recommendation SHA 漂移")
    if (set(recommendation) != {
            "aggregate_sha256", "artifact_kind",
            "candidate_host_freeze_sha256", "family_commitment",
            "formal_run_count", "format_version", "recommend_runtime_receipt"}
            or recommendation["artifact_kind"]
            != "PH2_W05_RUNTIME_RECEIPT_RECOMMENDATION"
            or recommendation["aggregate_sha256"] != aggregate_sha
            or recommendation["family_commitment"]
            != aggregate["family_commitment"]
            or recommendation["formal_run_count"] != 1
            or recommendation["format_version"] != 1
            or recommendation["recommend_runtime_receipt"] != 1):
        raise W05ReleaseError("recommendation binding 漂移")
    _strict_sha256(
        recommendation["candidate_host_freeze_sha256"],
        label="recommendation candidate host")
    return recommendation


def publish_w05_runtime_receipt(
        repository_root: str | Path,
        *,
        aggregate_path: str | Path,
        aggregate_sha256: str,
        recommendation_path: str | Path,
        recommendation_sha256: str,
        candidate_host_freeze_path: str | Path,
        candidate_contract_sha256: str,
        candidate_host_freeze_sha256: str,
        w04_receipt_sha256: str,
        d03_receipt_sha256: str,
        d03_global_manifest_sha256: str,
        d03_stage_manifest_sha256: str,
        invalidation_graph_sha256: str,
        atomic_pack_sha256: str,
        pre_w04_gate_sha256: str,
        lc16_overlay_sha256: str,
        lc16_mapper_sha256: str,
        lc16_projection_sha256: str,
        lc16_directional_sha256: str,
        publication_commit_sha1: str,
        verification_run_id: int,
        verification_jobs: tuple[tuple[str, str], ...],
        ) -> tuple[Path, str]:
    """主 owner 只读安全摘要并 append-only 发布 W-05 receipt。"""
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
    host_evidence = host.get("host_evidence")
    readback_evidence = host.get("dump_readback_evidence")
    owner_writes = host.get("owner_write_counts")
    expected_artifact_counts = {
        "CANDIDATE": 6,
        "CARRIER_PROJECTION": 9,
        "EVIDENCE_ACCOUNT": 8,
        "EVIDENCE_APPLICATION": 6,
        "GENERATION_CHOICE": 1,
        "GENERATION_DECISION": 1,
        "GENERATION_OUTCOME": 1,
        "GENERATION_USE": 1,
        "LOGICAL_SHARD": 16,
        "OCCURRENCE": 19,
        "REASONING_OUTCOME": 1,
        "REASONING_USE": 1,
        "ROLE_BINDING": 11,
        "ROLE_PROPOSITION_SCOPE_CELL": 27,
        "UNDERSTANDING_OUTCOME": 1,
        "UNDERSTANDING_USE": 1,
    }
    if (hashlib.sha256(host_bytes).hexdigest() != host_sha
            or recommendation["candidate_host_freeze_sha256"] != host_sha
            or host.get("candidate_contract_sha256") != contract_sha
            or host.get("formal_run_count") != 1
            or host.get("execution_state") != W05_FORMAL_EXECUTION_STATE
            or host.get("open_generation_state") != W05_OPEN_GENERATION_STATE
            or host.get("self_excluded") != 1
            or not isinstance(host_evidence, dict)
            or not isinstance(readback_evidence, dict)
            or not isinstance(owner_writes, dict)
            or dict(host_evidence.get("artifact_counts", ()))
            != expected_artifact_counts
            or dict(readback_evidence.get("artifact_counts", ()))
            != expected_artifact_counts
            or host_evidence.get("execution_state") != W05_FORMAL_EXECUTION_STATE
            or readback_evidence.get("execution_state")
            != W05_FORMAL_EXECUTION_STATE
            or host_evidence.get("transaction_event_count") != 5
            or readback_evidence.get("transaction_event_count") != 5
            or host_evidence.get("learning_attempt_count") != 1
            or readback_evidence.get("learning_attempt_count") != 1
            or readback_evidence.get("dump_readback") != 1
            or readback_evidence.get("payload_gets_this_call") != 0
            or readback_evidence.get("payload_bytes_this_call") != 0
            or readback_evidence.get("new_learning_write_count") != 0
            or host_evidence.get("teacher_calls") != 0
            or readback_evidence.get("teacher_calls") != 0
            or owner_writes.get("evaluator_label_writes") != 0
            or owner_writes.get("readback_learning_writes") != 0
            or owner_writes.get("formal_training_runs") != 1
            or owner_writes.get("teacher_calls") != 0):
        raise W05ReleaseError("candidate host freeze 状态漂移")
    resource_report = host_evidence.get("resource_report")
    if not isinstance(resource_report, dict):
        raise W05ReleaseError("candidate resource report 缺失")
    budget_pairs = (
        ("actual_checkpoint_count", "max_checkpoint_count"),
        ("actual_logic_operations", "max_logic_operations"),
        ("actual_payload_bytes", "max_payload_bytes"),
        ("actual_payload_gets", "max_payload_gets"),
        ("actual_recompute_objects", "max_recompute_objects"),
        ("actual_records", "max_records"),
        ("actual_segments", "max_segments"),
        ("actual_workers", "max_workers"),
    )
    if (resource_report.get("actual_workers") != 4
            or resource_report.get("teacher_calls") != 0
            or any(type(resource_report.get(actual)) is not int
                   or resource_report[actual] < 0
                   or resource_report[actual] > W05_RESOURCE_BUDGET[maximum]
                   for actual, maximum in budget_pairs)):
        raise W05ReleaseError("candidate resource budget 未闭合")
    host_digests = host_evidence.get("host_digests")
    readback_digests = readback_evidence.get("host_digests")
    if (not isinstance(host_digests, dict)
            or host_digests != readback_digests
            or set(host_digests) != {
                "candidate", "carrier_scope", "generation", "logical",
                "reasoning", "transaction", "understanding",
            }):
        raise W05ReleaseError("candidate host/readback digest 未闭合")
    for key, value in host_digests.items():
        _strict_sha256(value, label=f"candidate {key} digest")
    parents = {}
    for label, value in (
            ("w04_receipt_sha256", w04_receipt_sha256),
            ("d03_receipt_sha256", d03_receipt_sha256),
            ("d03_global_manifest_sha256", d03_global_manifest_sha256),
            ("d03_stage_manifest_sha256", d03_stage_manifest_sha256),
            ("invalidation_graph_sha256", invalidation_graph_sha256),
            ("atomic_pack_sha256", atomic_pack_sha256),
            ("pre_w04_gate_sha256", pre_w04_gate_sha256),
            ("lc16_overlay_sha256", lc16_overlay_sha256),
            ("lc16_mapper_sha256", lc16_mapper_sha256),
            ("lc16_projection_sha256", lc16_projection_sha256),
            ("lc16_directional_sha256", lc16_directional_sha256)):
        parents[label] = _strict_sha256(value, label=label)
    if parents != W05_EXPECTED_PARENT_IDENTITIES:
        raise W05ReleaseError("W-05 receipt parent identity 漂移")
    commit = _strict_sha1(publication_commit_sha1, label="publication commit")
    if (type(verification_run_id) is not int or verification_run_id <= 0
            or not isinstance(verification_jobs, tuple)
            or tuple(item[0] for item in verification_jobs)
            != W05_REQUIRED_VERIFICATION_JOBS):
        raise W05ReleaseError("verification identity 非法")
    jobs = []
    for name, status in verification_jobs:
        if status != "PASS":
            raise W05ReleaseError("verification job 未 PASS")
        jobs.append({"job": name, "status": status})
    receipt = {
        "ablation_results": aggregate["ablation_results"],
        "aggregate_sha256": _strict_sha256(
            aggregate_sha256, label="aggregate"),
        "artifact_kind": "PH2_W05_RUNTIME_EVIDENCE_RECEIPT",
        "candidate_contract_sha256": contract_sha,
        "candidate_host_freeze_sha256": host_sha,
        "case_commitment": aggregate["case_commitment"],
        "candidate_evidence": {
            "artifact_counts": host_evidence["artifact_counts"],
            "dump_manifest_sha256": host_evidence["dump_manifest_sha256"],
            "dump_readback": readback_evidence["dump_readback"],
            "host_digests": host_digests,
            "learning_attempt_count": host_evidence["learning_attempt_count"],
            "resource_report": resource_report,
            "transaction_event_count": host_evidence[
                "transaction_event_count"],
        },
        "cluster_commitment": aggregate["cluster_commitment"],
        "dimension_results": aggregate["dimension_results"],
        "execution_state": dict(W05_RECEIPT_EXECUTION_STATE),
        "family_commitment": aggregate["family_commitment"],
        "format_version": 1,
        "generation_ablation_statuses": aggregate[
            "generation_ablation_statuses"],
        "generation_hard_conjunct": W05_GENERATION_HARD_CONJUNCT,
        "hard_conjunct_policy": W05_AGGREGATION_POLICY,
        "infrastructure": aggregate["infrastructure"],
        "label_commitment": aggregate["label_commitment"],
        "open_generation_state": W05_OPEN_GENERATION_STATE,
        "parent_identities": parents,
        "payload_commitment": aggregate["payload_commitment"],
        "publication": {
            "commit_sha1": commit,
            "verification_jobs": jobs,
            "verification_run_id": verification_run_id,
        },
        "receipt_relative_path": W05_PUBLIC_RUNTIME_RECEIPT_RELATIVE_PATH,
        "receipt_self_excluded": 1,
        "recommendation_sha256": _strict_sha256(
            recommendation_sha256, label="recommendation"),
        "required": 1,
        "stage_key": "W-05",
        "status": "RUNTIME_EVIDENCED",
    }
    encoded = canonical_json_bytes(receipt)
    target = root / Path(*W05_PUBLIC_RUNTIME_RECEIPT_RELATIVE_PATH.split("/"))
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(encoded)
    except FileExistsError as exc:
        raise W05ReleaseError("W-05 runtime receipt 不可覆盖") from exc
    return target, hashlib.sha256(encoded).hexdigest()


def read_w05_runtime_receipt(repository_root: str | Path) -> dict[str, Any]:
    """规范回读已发布 receipt，并重验状态、parent 与安全字段。"""
    root = Path(repository_root).resolve()
    path = root / Path(*W05_PUBLIC_RUNTIME_RECEIPT_RELATIVE_PATH.split("/"))
    receipt, payload = _read_canonical(path, label="W-05 runtime receipt")
    if (receipt.get("artifact_kind") != "PH2_W05_RUNTIME_EVIDENCE_RECEIPT"
            or receipt.get("format_version") != 1
            or receipt.get("required") != 1
            or receipt.get("stage_key") != "W-05"
            or receipt.get("status") != "RUNTIME_EVIDENCED"
            or receipt.get("receipt_relative_path")
            != W05_PUBLIC_RUNTIME_RECEIPT_RELATIVE_PATH
            or receipt.get("receipt_self_excluded") != 1
            or receipt.get("execution_state") != W05_RECEIPT_EXECUTION_STATE
            or receipt.get("open_generation_state")
            != W05_OPEN_GENERATION_STATE
            or receipt.get("parent_identities")
            != W05_EXPECTED_PARENT_IDENTITIES
            or receipt.get("generation_hard_conjunct")
            != W05_GENERATION_HARD_CONJUNCT
            or receipt.get("hard_conjunct_policy")
            != W05_AGGREGATION_POLICY):
        raise W05ReleaseError("W-05 runtime receipt public contract 漂移")
    dimensions = receipt.get("dimension_results")
    ablations = receipt.get("ablation_results")
    publication = receipt.get("publication")
    if (not isinstance(dimensions, list)
            or tuple(item.get("dimension_key") for item in dimensions)
            != W05_EVALUATION_ORDER
            or any(item.get("status") != "PASS" for item in dimensions)
            or not isinstance(ablations, list)
            or tuple(item.get("ablation_key") for item in ablations)
            != W05_PRIVATE_ABLATION_KEYS
            or not isinstance(publication, dict)
            or tuple(item.get("job") for item in publication.get(
                "verification_jobs", ())) != W05_REQUIRED_VERIFICATION_JOBS
            or any(item.get("status") != "PASS"
                   for item in publication.get("verification_jobs", ()))):
        raise W05ReleaseError("W-05 runtime receipt PASS evidence 漂移")
    for key in (
            "aggregate_sha256", "candidate_contract_sha256",
            "candidate_host_freeze_sha256", "case_commitment",
            "cluster_commitment", "family_commitment", "label_commitment",
            "payload_commitment", "recommendation_sha256"):
        _strict_sha256(receipt.get(key), label=f"receipt {key}")
    forbidden = (b"surface", b"expected", b"private_path", b"message")
    if any(token in payload for token in forbidden):
        raise W05ReleaseError("W-05 runtime receipt 含 private 字段")
    return receipt


__all__ = [
    "W05_D03_RECEIPT_SHA256",
    "W05_EXPECTED_PARENT_IDENTITIES",
    "W05_PUBLIC_RUNTIME_RECEIPT_NAME",
    "W05_PUBLIC_RUNTIME_RECEIPT_RELATIVE_PATH",
    "W05_REQUIRED_VERIFICATION_JOBS",
    "W05_RECEIPT_EXECUTION_STATE",
    "W05ReleaseError",
    "publish_w05_runtime_receipt",
    "read_w05_runtime_receipt",
]
