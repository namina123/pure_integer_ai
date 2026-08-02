"""W-06 主 owner 的公开 runtime receipt 排他发布合同。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from pure_integer_ai.experiments.ph2_w06_candidate import (
    W06_CANDIDATE_HOST_FREEZE_KIND,
    W06_FORMAL_EXECUTION_STATE,
)
from pure_integer_ai.experiments.ph2_w06_contract import (
    W06_AGGREGATION_POLICY,
    W06_EVALUATION_ORDER,
    W06_EXPECTED_SHA256,
    W06_GLOBAL_MANIFEST_PATH,
    W06_INVALIDATION_GRAPH_PATH,
    W06_OPEN_GENERATION_STATE,
    W06_PRIVATE_ABLATION_KEYS,
    W06_RESOURCE_BUDGET,
    W06_STAGE_MANIFEST_PATH,
    W06_W05_RECEIPT_PATH,
)
from pure_integer_ai.experiments.ph2_w06_source_semantic import (
    W06_GENERATION_HARD_CONJUNCT,
)
from pure_integer_ai.experiments.ph2_w06_source_semantic_overlay import (
    W06_SOURCE_OVERLAY_PATH,
)


W06_PUBLIC_RUNTIME_RECEIPT_RELATIVE_PATH = (
    "data/ph2/manifests/d03_v1/w06_runtime_evidence_receipt_v1.json"
)
W06_PUBLIC_RUNTIME_RECEIPT_NAME = "w06_runtime_evidence_receipt_v1.json"
W06_REQUIRED_VERIFICATION_JOBS = (
    "W06 bounded local specialization",
    "W06 adjacent retention",
    "W06 compile/source guards",
    "W06 identity/secret gate",
)
W06_EXPECTED_PARENT_IDENTITIES = {
    "d03_global_manifest_sha256": W06_EXPECTED_SHA256[
        W06_GLOBAL_MANIFEST_PATH],
    "d03_stage_manifest_sha256": W06_EXPECTED_SHA256[
        W06_STAGE_MANIFEST_PATH],
    "invalidation_graph_sha256": W06_EXPECTED_SHA256[
        W06_INVALIDATION_GRAPH_PATH],
    "source_overlay_sha256": W06_EXPECTED_SHA256[W06_SOURCE_OVERLAY_PATH],
    "w05_receipt_sha256": W06_EXPECTED_SHA256[W06_W05_RECEIPT_PATH],
}
W06_EXPECTED_RETENTION_IDENTITIES = (
    (
        "data/ph2/manifests/d03_v1/w03_runtime_evidence_receipt_v1.json",
        "ef64636ab287eacbacae4040f59da74bb4105374cba31d756e1ddefaf86043f6",
    ),
    (
        "data/ph2/manifests/d03_v1/w04_runtime_evidence_receipt_v1.json",
        "153db3d7f3c0fca04642f4198df16e3c1adb0f5c78e4d6c7c59d35122989727b",
    ),
    (
        "data/ph2/manifests/d03_v1/w05_runtime_evidence_receipt_v1.json",
        "64c2fff496e766df880d2db1b184e2b8a009abd3b37b1a1b1331900458ccff78",
    ),
    (
        "data/ph2/manifests/w02_lc16_supplemental_runtime_receipt_v1.json",
        "141a6c2341671d4d92d9974a355b8081fd12dff17315f5d1f60913a45c31c8f1",
    ),
)
W06_EXPECTED_ARTIFACT_COUNTS = {
    "ACTIVE_RELATION": 17,
    "CANDIDATE": 50,
    "CARRIER_PROJECTION": 9,
    "EVIDENCE_ACCOUNT": 64,
    "EVIDENCE_APPLICATION": 50,
    "LOGICAL_SHARD": 16,
    "RELATION_FAMILY": 14,
    "RELATION_SCOPE_CELL": 27,
    "RELATION_USE": 3,
    "SCHEMA_REJECTION": 1,
    "SUBSTAGE": 7,
}
W06_RECEIPT_EXECUTION_STATE = {
    "LANGUAGE_CAPABILITY_MASTERED": 0,
    "LANGUAGE_READINESS": 0,
    "W06_BLOCKED_FAILED": 0,
    "W06_RUNTIME_EVIDENCED": 1,
    "W06_STARTED": 1,
    "W07_STARTED": 0,
    "formal_w06_training_runs": 1,
    "teacher_calls": 0,
}
_FORBIDDEN_SAFE_OUTPUT_TOKENS = (
    b"surface", b"expected", b"private_path", b"message")


class W06ReleaseError(RuntimeError):
    """W-06 receipt 输入、安全 hard conjunct 或排他发布错误。"""


def _strict_sha256(value: object, *, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise W06ReleaseError(f"{label} 不是规范 SHA-256")
    return value


def _strict_sha1(value: object, *, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 40
            or any(char not in "0123456789abcdef" for char in value)):
        raise W06ReleaseError(f"{label} 不是规范 SHA-1")
    return value


def _read_canonical(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    if not path.is_file() or path.is_symlink():
        raise W06ReleaseError(f"{label} 缺失或为链接")
    payload = path.read_bytes()
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise W06ReleaseError(f"{label} JSON 非法") from exc
    if not isinstance(value, dict) or canonical_json_bytes(value) != payload:
        raise W06ReleaseError(f"{label} 非 canonical object")
    return value, payload


def _check_aggregate(path: Path, expected_sha: str) -> dict[str, Any]:
    expected = _strict_sha256(expected_sha, label="aggregate")
    aggregate, payload = _read_canonical(path, label="aggregate")
    if hashlib.sha256(payload).hexdigest() != expected:
        raise W06ReleaseError("aggregate SHA 漂移")
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
            != "PH2_W06_PRIVATE_EVALUATION_AGGREGATE"
            or aggregate["format_version"] != 1
            or aggregate["failure_phase"] != "NONE"
            or aggregate["formal_run_count"] != 1
            or aggregate["status"] != "PASS"
            or aggregate["pass_count"] != len(W06_EVALUATION_ORDER)
            or aggregate["fail_count"] != 0
            or aggregate["ne_count"] != 0
            or aggregate["host_writes"] != 0
            or aggregate["label_writes"] != 0):
        raise W06ReleaseError("aggregate 未满足 W-06 PASS hard conjunct")
    dimensions = aggregate["dimension_results"]
    if (not isinstance(dimensions, list)
            or tuple(item.get("dimension_key") for item in dimensions)
            != W06_EVALUATION_ORDER):
        raise W06ReleaseError("aggregate 八维顺序漂移")
    for item in dimensions:
        if (set(item) != {
                "dimension_key", "evidence_commitment", "fail_count",
                "ne_count", "passed", "required", "status"}
                or item["status"] != "PASS"
                or item["passed"] != 1
                or item["required"] != 1
                or item["fail_count"] != 0
                or item["ne_count"] != 0):
            raise W06ReleaseError("aggregate dimension 非 1/1 PASS")
        _strict_sha256(item["evidence_commitment"], label="dimension evidence")
    ablations = aggregate["ablation_results"]
    if (not isinstance(ablations, list)
            or tuple(item.get("ablation_key") for item in ablations)
            != W06_PRIVATE_ABLATION_KEYS):
        raise W06ReleaseError("aggregate 八项 ablation 顺序漂移")
    for ordinal, item in enumerate(ablations):
        expected_statuses = [
            "FAIL" if index == ordinal else "PASS"
            for index in range(len(W06_EVALUATION_ORDER))]
        if (set(item) != {"ablation_key", "dimension_statuses"}
                or item["dimension_statuses"] != expected_statuses):
            raise W06ReleaseError("aggregate ablation 未正交击穿")
    expected_generation = [
        *("PASS" for _ in range(len(W06_EVALUATION_ORDER) - 1)),
        "FAIL",
    ]
    if aggregate["generation_ablation_statuses"] != expected_generation:
        raise W06ReleaseError("generation ablation 未击穿")
    expected_infrastructure = {
        "candidate_inventory_match": 1,
        "carrier_projection_count": 9,
        "carrier_scope_digest_match": 1,
        "clone_dump_readback": 1,
        "clone_host_copy_match": 1,
        "evaluator_label_writes": 0,
        "host_copy_unchanged": 1,
        "public_repo_writes": 0,
        "relation_scope_cell_count": 27,
    }
    if aggregate["infrastructure"] != expected_infrastructure:
        raise W06ReleaseError("aggregate clone/readback/owner isolation 未闭合")
    for key in (
            "family_commitment", "payload_commitment", "case_commitment",
            "label_commitment", "cluster_commitment"):
        _strict_sha256(aggregate[key], label=key)
    if any(token in payload for token in _FORBIDDEN_SAFE_OUTPUT_TOKENS):
        raise W06ReleaseError("aggregate 含 private 字段")
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
        raise W06ReleaseError("recommendation SHA 漂移")
    if (set(recommendation) != {
            "aggregate_sha256", "artifact_kind",
            "candidate_host_freeze_sha256", "family_commitment",
            "formal_run_count", "format_version", "recommend_runtime_receipt"}
            or recommendation["artifact_kind"]
            != "PH2_W06_RUNTIME_RECEIPT_RECOMMENDATION"
            or recommendation["aggregate_sha256"] != aggregate_sha
            or recommendation["family_commitment"]
            != aggregate["family_commitment"]
            or recommendation["formal_run_count"] != 1
            or recommendation["format_version"] != 1
            or recommendation["recommend_runtime_receipt"] != 1):
        raise W06ReleaseError("recommendation binding 漂移")
    _strict_sha256(
        recommendation["candidate_host_freeze_sha256"],
        label="recommendation candidate host")
    if any(token in payload for token in _FORBIDDEN_SAFE_OUTPUT_TOKENS):
        raise W06ReleaseError("recommendation 含 private 字段")
    return recommendation


def _check_candidate_host(
        path: Path,
        *,
        host_sha: str,
        contract_sha: str,
        publication_commit: str,
        recommendation: dict[str, Any],
        ) -> tuple[dict[str, Any], dict[str, Any]]:
    host, host_bytes = _read_canonical(path, label="candidate host freeze")
    host_evidence = host.get("host_evidence")
    readback = host.get("dump_readback_evidence")
    writes = host.get("owner_write_counts")
    if (hashlib.sha256(host_bytes).hexdigest() != host_sha
            or recommendation["candidate_host_freeze_sha256"] != host_sha
            or host.get("artifact_kind") != W06_CANDIDATE_HOST_FREEZE_KIND
            or host.get("candidate_contract_sha256") != contract_sha
            or host.get("remote_commit_sha1") != publication_commit
            or host.get("formal_run_count") != 1
            or host.get("execution_state") != W06_FORMAL_EXECUTION_STATE
            or host.get("open_generation_state") != W06_OPEN_GENERATION_STATE
            or host.get("self_excluded") != 1
            or not isinstance(host_evidence, dict)
            or not isinstance(readback, dict)
            or not isinstance(writes, dict)):
        raise W06ReleaseError("candidate host freeze identity 漂移")
    if (dict(host_evidence.get("artifact_counts", ()))
            != W06_EXPECTED_ARTIFACT_COUNTS
            or dict(readback.get("artifact_counts", ()))
            != W06_EXPECTED_ARTIFACT_COUNTS
            or host_evidence.get("execution_state")
            != W06_FORMAL_EXECUTION_STATE
            or readback.get("execution_state") != W06_FORMAL_EXECUTION_STATE
            or host_evidence.get("transaction_event_count") != 5
            or readback.get("transaction_event_count") != 5
            or host_evidence.get("learning_attempt_count") != 1
            or readback.get("learning_attempt_count") != 1
            or host_evidence.get("dump_readback") != 0
            or readback.get("dump_readback") != 1
            or host_evidence.get("new_learning_write_count") != 126
            or host_evidence.get("payload_gets_this_call") != 54
            or host_evidence.get("payload_bytes_this_call") != 199296
            or readback.get("new_learning_write_count") != 0
            or readback.get("payload_gets_this_call") != 0
            or readback.get("payload_bytes_this_call") != 0
            or host_evidence.get("teacher_calls") != 0
            or readback.get("teacher_calls") != 0
            or host_evidence.get("owned_tables")
            != ["graph_object", "ph2_w06_transaction_event"]
            or readback.get("owned_tables")
            != ["graph_object", "ph2_w06_transaction_event"]
            or tuple(tuple(item) for item in host_evidence.get(
                "retention_sha256", ())) != W06_EXPECTED_RETENTION_IDENTITIES
            or host_evidence.get("retention_sha256")
            != readback.get("retention_sha256")
            or writes != {
                "artifact_writes": 258,
                "evaluator_label_writes": 0,
                "formal_training_runs": 1,
                "readback_learning_writes": 0,
                "teacher_calls": 0,
            }):
        raise W06ReleaseError("candidate host freeze 状态漂移")
    resource = host_evidence.get("resource_report")
    if (not isinstance(resource, dict)
            or resource != readback.get("resource_report")):
        raise W06ReleaseError("candidate resource report 漂移")
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
    if (resource.get("actual_workers") != 4
            or resource.get("teacher_calls") != 0
            or any(type(resource.get(actual)) is not int
                   or resource[actual] < 0
                   or resource[actual] > W06_RESOURCE_BUDGET[maximum]
                   for actual, maximum in budget_pairs)):
        raise W06ReleaseError("candidate resource budget 未闭合")
    host_digests = host_evidence.get("host_digests")
    if (not isinstance(host_digests, dict)
            or host_digests != readback.get("host_digests")
            or set(host_digests) != {
                "active_projection", "candidate", "carrier_scope", "logical",
                "relation", "source_evidence", "transaction"}):
        raise W06ReleaseError("candidate host/readback digest 未闭合")
    for key, value in host_digests.items():
        _strict_sha256(value, label=f"candidate {key} digest")
    for label in ("dump_manifest_sha256",):
        value = _strict_sha256(host_evidence.get(label), label=f"host {label}")
        if value != readback.get(label):
            raise W06ReleaseError("candidate dump identity 漂移")
    return host_evidence, readback


def publish_w06_runtime_receipt(
        repository_root: str | Path,
        *,
        aggregate_path: str | Path,
        aggregate_sha256: str,
        recommendation_path: str | Path,
        recommendation_sha256: str,
        candidate_host_freeze_path: str | Path,
        candidate_contract_sha256: str,
        candidate_first_run_guard_sha256: str,
        candidate_host_freeze_sha256: str,
        private_family_freeze_sha256: str,
        private_first_run_guard_sha256: str,
        w05_receipt_sha256: str,
        d03_global_manifest_sha256: str,
        d03_stage_manifest_sha256: str,
        invalidation_graph_sha256: str,
        source_overlay_sha256: str,
        publication_commit_sha1: str,
        verification_run_id: int,
        verification_jobs: tuple[tuple[str, str], ...],
        ) -> tuple[Path, str]:
    """只读安全摘要并 append-only 发布 W-06 runtime receipt。"""
    root = Path(repository_root).resolve()
    aggregate = _check_aggregate(
        Path(aggregate_path).resolve(), aggregate_sha256)
    recommendation = _check_recommendation(
        Path(recommendation_path).resolve(), recommendation_sha256,
        aggregate, aggregate_sha256)
    contract_sha = _strict_sha256(
        candidate_contract_sha256, label="candidate contract")
    candidate_guard_sha = _strict_sha256(
        candidate_first_run_guard_sha256, label="candidate guard")
    host_sha = _strict_sha256(
        candidate_host_freeze_sha256, label="candidate host")
    family_freeze_sha = _strict_sha256(
        private_family_freeze_sha256, label="private family freeze")
    private_guard_sha = _strict_sha256(
        private_first_run_guard_sha256, label="private guard")
    commit = _strict_sha1(publication_commit_sha1, label="publication commit")
    host_evidence, readback = _check_candidate_host(
        Path(candidate_host_freeze_path).resolve(),
        host_sha=host_sha,
        contract_sha=contract_sha,
        publication_commit=commit,
        recommendation=recommendation,
    )
    parents = {
        "d03_global_manifest_sha256": _strict_sha256(
            d03_global_manifest_sha256, label="d03 global manifest"),
        "d03_stage_manifest_sha256": _strict_sha256(
            d03_stage_manifest_sha256, label="d03 stage manifest"),
        "invalidation_graph_sha256": _strict_sha256(
            invalidation_graph_sha256, label="invalidation graph"),
        "source_overlay_sha256": _strict_sha256(
            source_overlay_sha256, label="source overlay"),
        "w05_receipt_sha256": _strict_sha256(
            w05_receipt_sha256, label="W-05 receipt"),
    }
    if parents != W06_EXPECTED_PARENT_IDENTITIES:
        raise W06ReleaseError("W-06 receipt parent identity 漂移")
    if (type(verification_run_id) is not int or verification_run_id <= 0
            or not isinstance(verification_jobs, tuple)
            or tuple(item[0] for item in verification_jobs)
            != W06_REQUIRED_VERIFICATION_JOBS):
        raise W06ReleaseError("verification identity 非法")
    jobs = []
    for name, status in verification_jobs:
        if status != "PASS":
            raise W06ReleaseError("verification job 未 PASS")
        jobs.append({"job": name, "status": status})
    receipt = {
        "ablation_results": aggregate["ablation_results"],
        "aggregate_sha256": _strict_sha256(
            aggregate_sha256, label="aggregate"),
        "artifact_kind": "PH2_W06_RUNTIME_EVIDENCE_RECEIPT",
        "candidate_contract_sha256": contract_sha,
        "candidate_evidence": {
            "artifact_counts": host_evidence["artifact_counts"],
            "dump_manifest_sha256": host_evidence["dump_manifest_sha256"],
            "dump_readback": readback["dump_readback"],
            "host_digests": host_evidence["host_digests"],
            "learning_attempt_count": host_evidence[
                "learning_attempt_count"],
            "new_learning_write_count": host_evidence[
                "new_learning_write_count"],
            "owned_tables": host_evidence["owned_tables"],
            "payload_bytes": host_evidence["payload_bytes_this_call"],
            "payload_gets": host_evidence["payload_gets_this_call"],
            "resource_report": host_evidence["resource_report"],
            "retention_sha256": host_evidence["retention_sha256"],
            "transaction_event_count": host_evidence[
                "transaction_event_count"],
        },
        "candidate_first_run_guard_sha256": candidate_guard_sha,
        "candidate_host_freeze_sha256": host_sha,
        "case_commitment": aggregate["case_commitment"],
        "cluster_commitment": aggregate["cluster_commitment"],
        "dimension_results": aggregate["dimension_results"],
        "execution_state": dict(W06_RECEIPT_EXECUTION_STATE),
        "family_commitment": aggregate["family_commitment"],
        "format_version": 1,
        "generation_ablation_statuses": aggregate[
            "generation_ablation_statuses"],
        "generation_hard_conjunct": W06_GENERATION_HARD_CONJUNCT,
        "hard_conjunct_policy": W06_AGGREGATION_POLICY,
        "infrastructure": aggregate["infrastructure"],
        "label_commitment": aggregate["label_commitment"],
        "open_generation_state": W06_OPEN_GENERATION_STATE,
        "parent_identities": parents,
        "payload_commitment": aggregate["payload_commitment"],
        "private_family_freeze_sha256": family_freeze_sha,
        "private_first_run_guard_sha256": private_guard_sha,
        "publication": {
            "commit_sha1": commit,
            "verification_jobs": jobs,
            "verification_run_id": verification_run_id,
        },
        "receipt_relative_path": W06_PUBLIC_RUNTIME_RECEIPT_RELATIVE_PATH,
        "receipt_self_excluded": 1,
        "recommendation_sha256": _strict_sha256(
            recommendation_sha256, label="recommendation"),
        "required": 1,
        "stage_key": "W-06",
        "status": "RUNTIME_EVIDENCED",
    }
    encoded = canonical_json_bytes(receipt)
    if any(token in encoded for token in _FORBIDDEN_SAFE_OUTPUT_TOKENS):
        raise W06ReleaseError("W-06 runtime receipt 含 private 字段")
    target = root / Path(*W06_PUBLIC_RUNTIME_RECEIPT_RELATIVE_PATH.split("/"))
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(encoded)
    except FileExistsError as exc:
        raise W06ReleaseError("W-06 runtime receipt 不可覆盖") from exc
    return target, hashlib.sha256(encoded).hexdigest()


def read_w06_runtime_receipt(repository_root: str | Path) -> dict[str, Any]:
    """规范回读已发布 receipt，并重验状态、parent 与安全字段。"""
    root = Path(repository_root).resolve()
    path = root / Path(*W06_PUBLIC_RUNTIME_RECEIPT_RELATIVE_PATH.split("/"))
    receipt, payload = _read_canonical(path, label="W-06 runtime receipt")
    if (receipt.get("artifact_kind") != "PH2_W06_RUNTIME_EVIDENCE_RECEIPT"
            or receipt.get("format_version") != 1
            or receipt.get("required") != 1
            or receipt.get("stage_key") != "W-06"
            or receipt.get("status") != "RUNTIME_EVIDENCED"
            or receipt.get("receipt_relative_path")
            != W06_PUBLIC_RUNTIME_RECEIPT_RELATIVE_PATH
            or receipt.get("receipt_self_excluded") != 1
            or receipt.get("execution_state") != W06_RECEIPT_EXECUTION_STATE
            or receipt.get("open_generation_state")
            != W06_OPEN_GENERATION_STATE
            or receipt.get("parent_identities")
            != W06_EXPECTED_PARENT_IDENTITIES
            or receipt.get("generation_hard_conjunct")
            != W06_GENERATION_HARD_CONJUNCT
            or receipt.get("hard_conjunct_policy")
            != W06_AGGREGATION_POLICY):
        raise W06ReleaseError("W-06 runtime receipt public contract 漂移")
    dimensions = receipt.get("dimension_results")
    ablations = receipt.get("ablation_results")
    publication = receipt.get("publication")
    if (not isinstance(dimensions, list)
            or tuple(item.get("dimension_key") for item in dimensions)
            != W06_EVALUATION_ORDER
            or any(item.get("status") != "PASS" for item in dimensions)
            or not isinstance(ablations, list)
            or tuple(item.get("ablation_key") for item in ablations)
            != W06_PRIVATE_ABLATION_KEYS
            or not isinstance(publication, dict)
            or tuple(item.get("job") for item in publication.get(
                "verification_jobs", ())) != W06_REQUIRED_VERIFICATION_JOBS
            or any(item.get("status") != "PASS"
                   for item in publication.get("verification_jobs", ()) )):
        raise W06ReleaseError("W-06 runtime receipt PASS evidence 漂移")
    for key in (
            "aggregate_sha256", "candidate_contract_sha256",
            "candidate_first_run_guard_sha256",
            "candidate_host_freeze_sha256", "case_commitment",
            "cluster_commitment", "family_commitment", "label_commitment",
            "payload_commitment", "private_family_freeze_sha256",
            "private_first_run_guard_sha256", "recommendation_sha256"):
        _strict_sha256(receipt.get(key), label=f"receipt {key}")
    if any(token in payload for token in _FORBIDDEN_SAFE_OUTPUT_TOKENS):
        raise W06ReleaseError("W-06 runtime receipt 含 private 字段")
    return receipt


__all__ = [
    "W06_EXPECTED_PARENT_IDENTITIES",
    "W06_EXPECTED_RETENTION_IDENTITIES",
    "W06_PUBLIC_RUNTIME_RECEIPT_NAME",
    "W06_PUBLIC_RUNTIME_RECEIPT_RELATIVE_PATH",
    "W06_RECEIPT_EXECUTION_STATE",
    "W06_REQUIRED_VERIFICATION_JOBS",
    "W06ReleaseError",
    "publish_w06_runtime_receipt",
    "read_w06_runtime_receipt",
]
