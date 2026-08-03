"""W-07 主 owner 的公开 runtime receipt 排他发布合同。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from pure_integer_ai.experiments.ph2_w07_candidate import (
    W07_CANDIDATE_CONTRACT_KIND,
    W07_CANDIDATE_FIRST_RUN_GUARD_KIND,
    W07_CANDIDATE_HOST_FREEZE_KIND,
    W07_FORMAL_EXECUTION_STATE,
    verify_w07_candidate_contract_freeze,
)
from pure_integer_ai.experiments.ph2_w07_contract import (
    W07_AGGREGATION_POLICY,
    W07_EXPECTED_SHA256,
    W07_GENERATION_CHOICE_PATH,
    W07_GENERATION_HARD_CONJUNCT,
    W07_GENERATION_OUTCOME_PATH,
    W07_GLOBAL_MANIFEST_PATH,
    W07_INVALIDATION_GRAPH_PATH,
    W07_LC13_DIRECTIONAL_PATH,
    W07_LC16_DIRECTIONAL_PATH,
    W07_LC16_OVERLAY_PATH,
    W07_OPEN_GENERATION_STATE,
    W07_PUBLIC_ABLATION_KEYS,
    W07_PUBLIC_DIMENSION_KEYS,
    W07_RESOURCE_BUDGET,
    W07_STAGE_MANIFEST_PATH,
    W07_SUBSTAGE_ORDER,
    W07_W06_RECEIPT_PATH,
)


W07_PUBLIC_RUNTIME_RECEIPT_RELATIVE_PATH = (
    "data/ph2/manifests/d03_v1/w07_runtime_evidence_receipt_v1.json"
)
W07_PUBLIC_RUNTIME_RECEIPT_NAME = "w07_runtime_evidence_receipt_v1.json"
W07_REQUIRED_VERIFICATION_JOBS = (
    "W07 bounded logic/evaluator",
    "W07 candidate/retention",
    "W07 release/canonical safety",
    "W07 identity/secret gate",
)
W07_EXPECTED_PARENT_IDENTITIES = {
    "d03_global_manifest_sha256": W07_EXPECTED_SHA256[
        W07_GLOBAL_MANIFEST_PATH],
    "d03_stage_manifest_sha256": W07_EXPECTED_SHA256[
        W07_STAGE_MANIFEST_PATH],
    "generation_choice_sha256": W07_EXPECTED_SHA256[
        W07_GENERATION_CHOICE_PATH],
    "generation_outcome_sha256": W07_EXPECTED_SHA256[
        W07_GENERATION_OUTCOME_PATH],
    "invalidation_graph_sha256": W07_EXPECTED_SHA256[
        W07_INVALIDATION_GRAPH_PATH],
    "lc13_directional_sha256": W07_EXPECTED_SHA256[
        W07_LC13_DIRECTIONAL_PATH],
    "lc16_directional_sha256": W07_EXPECTED_SHA256[
        W07_LC16_DIRECTIONAL_PATH],
    "lc16_overlay_sha256": W07_EXPECTED_SHA256[W07_LC16_OVERLAY_PATH],
    "w06_receipt_sha256": W07_EXPECTED_SHA256[W07_W06_RECEIPT_PATH],
}
W07_EXPECTED_RETENTION_IDENTITIES = (
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
        "data/ph2/manifests/d03_v1/w06_runtime_evidence_receipt_v1.json",
        "aaf35a8346446e80d71f057ae391d9a734a864ced317fa06f2ea01f99efbc0e7",
    ),
    (
        "data/ph2/manifests/w02_lc16_supplemental_runtime_receipt_v1.json",
        "141a6c2341671d4d92d9974a355b8081fd12dff17315f5d1f60913a45c31c8f1",
    ),
)
W07_EXPECTED_ARTIFACT_COUNTS = {
    "ACTIVE_OPERATOR": 36,
    "CANDIDATE": 71,
    "CARRIER_PROJECTION": 9,
    "EVIDENCE_ACCOUNT": 94,
    "EVIDENCE_APPLICATION": 63,
    "LOGICAL_SHARD": 16,
    "LOGIC_SCOPE_CELL": 189,
    "LOGIC_USE": 21,
    "OPERATOR_PROFILE": 7,
    "SCHEMA_REJECTION": 3,
    "SUBSTAGE": 7,
}
W07_RECEIPT_EXECUTION_STATE = {
    "LANGUAGE_CAPABILITY_MASTERED": 0,
    "LANGUAGE_READINESS": 0,
    "W07_BLOCKED_FAILED": 0,
    "W07_RUNTIME_EVIDENCED": 1,
    "W07_STARTED": 1,
    "W08_STARTED": 0,
    "formal_w07_training_runs": 1,
    "teacher_calls": 0,
}
_FORBIDDEN_SAFE_OUTPUT_TOKENS = (
    b"case_key", b"label_key", b"surface", b"expected",
    b"private_path", b"message", b":\\\\",
)
_RESULT_FIELDS = {
    "dimension_key", "evidence_commitment", "fail_count", "ne_count",
    "passed", "required", "status",
}


class W07ReleaseError(RuntimeError):
    """W-07 receipt 输入、安全 hard conjunct 或排他发布错误。"""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _strict_sha256(value: object, *, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise W07ReleaseError(f"{label} 不是规范 SHA-256")
    return value


def _strict_sha1(value: object, *, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 40
            or any(char not in "0123456789abcdef" for char in value)):
        raise W07ReleaseError(f"{label} 不是规范 SHA-1")
    return value


def _read_canonical(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    if not path.is_file() or path.is_symlink():
        raise W07ReleaseError(f"{label} 缺失或为链接")
    payload = path.read_bytes()
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise W07ReleaseError(f"{label} JSON 非法") from error
    if not isinstance(value, dict) or canonical_json_bytes(value) != payload:
        raise W07ReleaseError(f"{label} 非 canonical object")
    return value, payload


def _check_result(
        item: object,
        *,
        dimension: str,
        expected_status: str,
        ) -> None:
    if not isinstance(item, dict) or set(item) != _RESULT_FIELDS:
        raise W07ReleaseError("dimension result schema 漂移")
    expected_values = (
        (1, 0, 0) if expected_status == "PASS" else (0, 1, 0))
    if (item["dimension_key"] != dimension
            or item["status"] != expected_status
            or item["required"] != 1
            or (item["passed"], item["fail_count"], item["ne_count"])
            != expected_values):
        raise W07ReleaseError("dimension result 未满足 1/1 hard conjunct")
    _strict_sha256(item["evidence_commitment"], label="dimension evidence")


def _check_infrastructure(
        infrastructure: object,
        *,
        evaluator_head: str,
        ) -> None:
    expected_keys = {
        "baseline_consumer_audit", "candidate_inventory_match",
        "carrier_projection_count", "carrier_scope_digest_match",
        "clone_dump_readback", "clone_host_copy_match",
        "evaluator_public_head_commit_sha1", "host_copy_unchanged",
        "ledger_audit", "ledger_audit_commitment", "ledger_count",
        "ledgers_closed", "ledgers_committed", "logic_scope_cell_count",
        "logic_use_count", "owner_audit_complete",
    }
    if not isinstance(infrastructure, dict) or set(infrastructure) != expected_keys:
        raise W07ReleaseError("aggregate infrastructure schema 漂移")
    expected_scalars = {
        "candidate_inventory_match": 1,
        "carrier_projection_count": 9,
        "carrier_scope_digest_match": 1,
        "clone_dump_readback": 1,
        "clone_host_copy_match": 1,
        "evaluator_public_head_commit_sha1": evaluator_head,
        "host_copy_unchanged": 1,
        "ledger_count": 7,
        "ledgers_closed": 1,
        "ledgers_committed": 1,
        "logic_scope_cell_count": 189,
        "logic_use_count": 21,
        "owner_audit_complete": 1,
    }
    if any(infrastructure[key] != value
           for key, value in expected_scalars.items()):
        raise W07ReleaseError("aggregate clone/ledger/owner isolation 未闭合")
    expected_audit = {
        "generation_choices": 15,
        "generation_outcomes": 15,
        "generation_uses": 15,
        "nested_generation_layer_uses": 2,
        "nested_reasoning_layer_uses": 2,
        "nested_understanding_layer_uses": 2,
        "reasoning_outcomes": 8,
        "reasoning_uses": 8,
        "understanding_outcomes": 8,
        "understanding_uses": 8,
    }
    if infrastructure["baseline_consumer_audit"] != expected_audit:
        raise W07ReleaseError("aggregate real facade audit 漂移")
    ledgers = infrastructure["ledger_audit"]
    if (not isinstance(ledgers, list)
            or tuple(item.get("substage") for item in ledgers)
            != W07_SUBSTAGE_ORDER):
        raise W07ReleaseError("aggregate ledger order 漂移")
    for item in ledgers:
        if (set(item) != {
                "nonempty_table_count", "row_count", "substage", "table_count"}
                or item["table_count"] != 53
                or item["nonempty_table_count"] != 7
                or type(item["row_count"]) is not int
                or item["row_count"] <= 0):
            raise W07ReleaseError("aggregate committed ledger audit 漂移")
    commitment = _sha256(canonical_json_bytes(ledgers))
    if infrastructure["ledger_audit_commitment"] != commitment:
        raise W07ReleaseError("aggregate ledger commitment 漂移")


def _check_aggregate(
        path: Path,
        expected_sha: str,
        evaluator_head: str,
        ) -> dict[str, Any]:
    expected = _strict_sha256(expected_sha, label="aggregate")
    aggregate, payload = _read_canonical(path, label="aggregate")
    required_fields = {
        "ablation_results", "artifact_kind", "baseline_results",
        "case_commitment", "cluster_commitment", "diagnostic_cursor",
        "evaluator_version", "fail_count", "failure_kind",
        "family_commitment", "formal_run_count", "format_version",
        "generation_ablation_statuses", "host_writes", "infrastructure",
        "label_commitment", "label_writes", "ne_count", "pass_count",
        "payload_commitment", "public_repo_writes", "status",
    }
    if (_sha256(payload) != expected
            or set(aggregate) != required_fields
            or aggregate["artifact_kind"]
            != "PH2_W07_PRIVATE_EVALUATION_V2_AGGREGATE"
            or aggregate["format_version"] != 2
            or aggregate["evaluator_version"] != 2
            or aggregate["formal_run_count"] != 1
            or aggregate["failure_kind"] != "NONE"
            or aggregate["status"] != "PASS"
            or aggregate["pass_count"] != len(W07_PUBLIC_DIMENSION_KEYS)
            or aggregate["fail_count"] != 0
            or aggregate["ne_count"] != 0
            or any(aggregate[key] != 0 for key in (
                "host_writes", "label_writes", "public_repo_writes"))
            or aggregate["diagnostic_cursor"] != {
                "ablation_key": "NONE",
                "dimension_key": "NONE",
                "operation": "PUBLISH_REPORT",
                "phase": "REPORT_SAFETY",
            }):
        raise W07ReleaseError("aggregate 未满足 W-07 PASS hard conjunct")
    dimensions = aggregate["baseline_results"]
    if (not isinstance(dimensions, list)
            or len(dimensions) != len(W07_PUBLIC_DIMENSION_KEYS)):
        raise W07ReleaseError("aggregate baseline 数量漂移")
    for dimension, item in zip(W07_PUBLIC_DIMENSION_KEYS, dimensions, strict=True):
        _check_result(item, dimension=dimension, expected_status="PASS")
    ablations = aggregate["ablation_results"]
    if (not isinstance(ablations, list)
            or tuple(item.get("ablation_key") for item in ablations)
            != W07_PUBLIC_ABLATION_KEYS):
        raise W07ReleaseError("aggregate 八项 ablation 顺序漂移")
    for ordinal, ablation in enumerate(ablations):
        if set(ablation) != {"ablation_key", "dimension_results"}:
            raise W07ReleaseError("aggregate ablation schema 漂移")
        results = ablation["dimension_results"]
        if not isinstance(results, list) or len(results) != len(dimensions):
            raise W07ReleaseError("aggregate ablation 结果数量漂移")
        for index, (dimension, item) in enumerate(zip(
                W07_PUBLIC_DIMENSION_KEYS, results, strict=True)):
            _check_result(
                item,
                dimension=dimension,
                expected_status="FAIL" if index == ordinal else "PASS",
            )
    expected_generation = [
        * ("PASS" for _ in range(len(W07_PUBLIC_DIMENSION_KEYS) - 1)),
        "FAIL",
    ]
    if aggregate["generation_ablation_statuses"] != expected_generation:
        raise W07ReleaseError("generation ablation 未击穿")
    _check_infrastructure(
        aggregate["infrastructure"], evaluator_head=evaluator_head)
    for key in (
            "family_commitment", "payload_commitment", "case_commitment",
            "label_commitment", "cluster_commitment"):
        _strict_sha256(aggregate[key], label=key)
    if any(token in payload for token in _FORBIDDEN_SAFE_OUTPUT_TOKENS):
        raise W07ReleaseError("aggregate 含 private 字段或绝对路径")
    return aggregate


def _check_recommendation(
        path: Path,
        expected_sha: str,
        aggregate: dict[str, Any],
        aggregate_sha: str,
        *,
        contract_sha: str,
        host_sha: str,
        evaluator_head: str,
        ) -> dict[str, Any]:
    expected = _strict_sha256(expected_sha, label="recommendation")
    recommendation, payload = _read_canonical(path, label="recommendation")
    expected_fields = {
        "aggregate_sha256", "artifact_kind", "candidate_contract_sha256",
        "candidate_host_freeze_sha256", "evaluator_public_head_commit_sha1",
        "evaluator_version", "family_commitment", "formal_run_count",
        "format_version", "recommend_runtime_receipt",
    }
    if (_sha256(payload) != expected
            or set(recommendation) != expected_fields
            or recommendation["artifact_kind"]
            != "PH2_W07_RUNTIME_RECEIPT_V2_RECOMMENDATION"
            or recommendation["aggregate_sha256"] != aggregate_sha
            or recommendation["candidate_contract_sha256"] != contract_sha
            or recommendation["candidate_host_freeze_sha256"] != host_sha
            or recommendation["evaluator_public_head_commit_sha1"]
            != evaluator_head
            or recommendation["evaluator_version"] != 2
            or recommendation["family_commitment"]
            != aggregate["family_commitment"]
            or recommendation["formal_run_count"] != 1
            or recommendation["format_version"] != 2
            or recommendation["recommend_runtime_receipt"] != 1):
        raise W07ReleaseError("recommendation binding 漂移")
    if any(token in payload for token in _FORBIDDEN_SAFE_OUTPUT_TOKENS):
        raise W07ReleaseError("recommendation 含 private 字段或绝对路径")
    return recommendation


def _check_candidate_artifacts(
        repository: Path,
        *,
        contract_path: Path,
        contract_sha: str,
        guard_path: Path,
        guard_sha: str,
        host_path: Path,
        host_sha: str,
        dump_path: Path,
        candidate_head: str,
        ) -> tuple[dict[str, Any], dict[str, Any]]:
    contract, contract_bytes = _read_canonical(
        contract_path, label="candidate contract freeze")
    if (_sha256(contract_bytes) != contract_sha
            or contract.get("artifact_kind") != W07_CANDIDATE_CONTRACT_KIND
            or contract.get("public_head_commit_sha1") != candidate_head
            or verify_w07_candidate_contract_freeze(
                repository, contract_path.parent,
                candidate_contract_sha256=contract_sha) != contract):
        raise W07ReleaseError("candidate contract identity 漂移")
    guard, guard_bytes = _read_canonical(
        guard_path, label="candidate first-run guard")
    if (_sha256(guard_bytes) != guard_sha
            or guard.get("artifact_kind")
            != W07_CANDIDATE_FIRST_RUN_GUARD_KIND
            or guard.get("candidate_contract_sha256") != contract_sha
            or guard.get("public_head_commit_sha1") != candidate_head
            or guard.get("formal_run_count_before") != 0
            or guard.get("formal_run_count_after") != 1):
        raise W07ReleaseError("candidate guard identity 漂移")
    host, host_bytes = _read_canonical(host_path, label="candidate host freeze")
    host_evidence = host.get("host_evidence")
    readback = host.get("dump_readback_evidence")
    if (_sha256(host_bytes) != host_sha
            or set(host) != {
                "artifact_kind", "candidate_contract_sha256",
                "candidate_first_run_guard_sha256", "dump_readback_evidence",
                "execution_state", "formal_run_count", "format_version",
                "host_evidence", "open_generation_state",
                "owner_write_counts", "public_head_commit_sha1", "self_excluded"}
            or host["artifact_kind"] != W07_CANDIDATE_HOST_FREEZE_KIND
            or host["candidate_contract_sha256"] != contract_sha
            or host["candidate_first_run_guard_sha256"] != guard_sha
            or host["public_head_commit_sha1"] != candidate_head
            or host["formal_run_count"] != 1
            or host["format_version"] != 1
            or host["execution_state"] != W07_FORMAL_EXECUTION_STATE
            or host["open_generation_state"] != W07_OPEN_GENERATION_STATE
            or host["self_excluded"] != 1
            or not isinstance(host_evidence, dict)
            or not isinstance(readback, dict)):
        raise W07ReleaseError("candidate host freeze identity 漂移")
    if (dict(host_evidence.get("artifact_counts", ()))
            != W07_EXPECTED_ARTIFACT_COUNTS
            or host_evidence.get("artifact_counts")
            != readback.get("artifact_counts")
            or host_evidence.get("execution_state") != W07_FORMAL_EXECUTION_STATE
            or readback.get("execution_state") != W07_FORMAL_EXECUTION_STATE
            or host_evidence.get("transaction_event_count") != 5
            or readback.get("transaction_event_count") != 5
            or host_evidence.get("learning_attempt_count") != 1
            or readback.get("learning_attempt_count") != 1
            or host_evidence.get("dump_readback") != 0
            or readback.get("dump_readback") != 1
            or host_evidence.get("new_learning_write_count") != 174
            or host_evidence.get("payload_gets_this_call") != 21
            or host_evidence.get("payload_bytes_this_call") != 36741
            or readback.get("new_learning_write_count") != 0
            or readback.get("payload_gets_this_call") != 0
            or readback.get("payload_bytes_this_call") != 0
            or host_evidence.get("teacher_calls") != 0
            or readback.get("teacher_calls") != 0
            or host_evidence.get("owned_tables")
            != ["graph_object", "ph2_w07_transaction_event"]
            or readback.get("owned_tables")
            != ["graph_object", "ph2_w07_transaction_event"]
            or tuple(tuple(item) for item in host_evidence.get(
                "retention_sha256", ())) != W07_EXPECTED_RETENTION_IDENTITIES
            or host_evidence.get("retention_sha256")
            != readback.get("retention_sha256")
            or host.get("owner_write_counts") != {
                "artifact_writes": 174,
                "evaluator_label_writes": 0,
                "formal_training_runs": 1,
                "readback_learning_writes": 0,
                "teacher_calls": 0,
            }):
        raise W07ReleaseError("candidate host freeze 状态漂移")
    resource = host_evidence.get("resource_report")
    if not isinstance(resource, dict) or resource != readback.get("resource_report"):
        raise W07ReleaseError("candidate resource report 漂移")
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
                   or resource[actual] > W07_RESOURCE_BUDGET[maximum]
                   for actual, maximum in budget_pairs)):
        raise W07ReleaseError("candidate resource budget 未闭合")
    host_digests = host_evidence.get("host_digests")
    if (not isinstance(host_digests, dict)
            or host_digests != readback.get("host_digests")
            or set(host_digests) != {
                "active_projection", "candidate", "carrier_scope", "logic",
                "logical", "source_evidence", "transaction"}):
        raise W07ReleaseError("candidate host/readback digest 未闭合")
    for key, value in host_digests.items():
        _strict_sha256(value, label=f"candidate {key} digest")
    dump_sha = _strict_sha256(
        host_evidence.get("dump_manifest_sha256"), label="dump manifest")
    if (dump_sha != readback.get("dump_manifest_sha256")
            or not dump_path.is_file() or dump_path.is_symlink()
            or _sha256(dump_path.read_bytes()) != dump_sha):
        raise W07ReleaseError("candidate dump identity 漂移")
    return host_evidence, readback


def _check_private_metadata(
        *,
        freeze_path: Path,
        freeze_sha: str,
        guard_path: Path,
        guard_sha: str,
        aggregate: dict[str, Any],
        contract_sha: str,
        host_sha: str,
        evaluator_head: str,
        ) -> None:
    freeze, freeze_bytes = _read_canonical(freeze_path, label="private family freeze")
    if (_sha256(freeze_bytes) != freeze_sha
            or freeze.get("artifact_kind")
            != "PH2_W07_PRIVATE_FAMILY_V2_FREEZE"
            or freeze.get("format_version") != 2
            or freeze.get("evaluator_version") != 2
            or freeze.get("formal_run_count") != 0
            or freeze.get("self_excluded") != 1
            or freeze.get("candidate_contract_sha256") != contract_sha
            or freeze.get("candidate_host_freeze_sha256") != host_sha
            or freeze.get("evaluator_public_head_commit_sha1")
            != evaluator_head
            or freeze.get("family_key") != aggregate["family_commitment"]
            or any(freeze.get(key) != aggregate[key] for key in (
                "payload_commitment", "case_commitment", "label_commitment",
                "cluster_commitment"))):
        raise W07ReleaseError("private v2 family metadata 漂移")
    guard, guard_bytes = _read_canonical(guard_path, label="private first-run guard")
    if (_sha256(guard_bytes) != guard_sha
            or guard != {
                "artifact_kind": "PH2_W07_PRIVATE_V2_FIRST_RUN_GUARD",
                "evaluator_version": 2,
                "family_freeze_sha256": freeze_sha,
                "formal_run_count_after": 1,
                "formal_run_count_before": 0,
                "format_version": 2,
            }):
        raise W07ReleaseError("private v2 guard metadata 漂移")


def _check_public_parents(root: Path) -> dict[str, str]:
    paths = {
        "d03_global_manifest_sha256": W07_GLOBAL_MANIFEST_PATH,
        "d03_stage_manifest_sha256": W07_STAGE_MANIFEST_PATH,
        "generation_choice_sha256": W07_GENERATION_CHOICE_PATH,
        "generation_outcome_sha256": W07_GENERATION_OUTCOME_PATH,
        "invalidation_graph_sha256": W07_INVALIDATION_GRAPH_PATH,
        "lc13_directional_sha256": W07_LC13_DIRECTIONAL_PATH,
        "lc16_directional_sha256": W07_LC16_DIRECTIONAL_PATH,
        "lc16_overlay_sha256": W07_LC16_OVERLAY_PATH,
        "w06_receipt_sha256": W07_W06_RECEIPT_PATH,
    }
    observed = {}
    for key, relative in paths.items():
        path = root / Path(*relative.split("/"))
        if not path.is_file() or path.is_symlink():
            raise W07ReleaseError("W-07 public parent 缺失或为链接")
        observed[key] = _sha256(path.read_bytes())
    if observed != W07_EXPECTED_PARENT_IDENTITIES:
        raise W07ReleaseError("W-07 receipt parent identity 漂移")
    return observed


def publish_w07_runtime_receipt(
        repository_root: str | Path,
        *,
        aggregate_path: str | Path,
        aggregate_sha256: str,
        recommendation_path: str | Path,
        recommendation_sha256: str,
        candidate_contract_freeze_path: str | Path,
        candidate_contract_sha256: str,
        candidate_first_run_guard_path: str | Path,
        candidate_first_run_guard_sha256: str,
        candidate_host_freeze_path: str | Path,
        candidate_host_freeze_sha256: str,
        candidate_dump_manifest_path: str | Path,
        private_family_freeze_path: str | Path,
        private_family_freeze_sha256: str,
        private_first_run_guard_path: str | Path,
        private_first_run_guard_sha256: str,
        candidate_public_head_commit_sha1: str,
        evaluator_public_head_commit_sha1: str,
        verification_run_id: int,
        verification_jobs: tuple[tuple[str, str], ...],
        ) -> tuple[Path, str]:
    """只读安全摘要并 append-only 发布 W-07 runtime receipt。"""
    root = Path(repository_root).resolve()
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
    candidate_head = _strict_sha1(
        candidate_public_head_commit_sha1, label="candidate public HEAD")
    evaluator_head = _strict_sha1(
        evaluator_public_head_commit_sha1, label="evaluator public HEAD")
    aggregate = _check_aggregate(
        Path(aggregate_path).resolve(), aggregate_sha256, evaluator_head)
    _check_recommendation(
        Path(recommendation_path).resolve(), recommendation_sha256,
        aggregate, aggregate_sha256,
        contract_sha=contract_sha,
        host_sha=host_sha,
        evaluator_head=evaluator_head,
    )
    host_evidence, readback = _check_candidate_artifacts(
        root,
        contract_path=Path(candidate_contract_freeze_path).resolve(),
        contract_sha=contract_sha,
        guard_path=Path(candidate_first_run_guard_path).resolve(),
        guard_sha=candidate_guard_sha,
        host_path=Path(candidate_host_freeze_path).resolve(),
        host_sha=host_sha,
        dump_path=Path(candidate_dump_manifest_path).resolve(),
        candidate_head=candidate_head,
    )
    _check_private_metadata(
        freeze_path=Path(private_family_freeze_path).resolve(),
        freeze_sha=family_freeze_sha,
        guard_path=Path(private_first_run_guard_path).resolve(),
        guard_sha=private_guard_sha,
        aggregate=aggregate,
        contract_sha=contract_sha,
        host_sha=host_sha,
        evaluator_head=evaluator_head,
    )
    parents = _check_public_parents(root)
    if (type(verification_run_id) is not int or verification_run_id <= 0
            or not isinstance(verification_jobs, tuple)
            or tuple(item[0] for item in verification_jobs)
            != W07_REQUIRED_VERIFICATION_JOBS):
        raise W07ReleaseError("verification identity 非法")
    jobs = []
    for name, status in verification_jobs:
        if status != "PASS":
            raise W07ReleaseError("verification job 未 PASS")
        jobs.append({"job": name, "status": status})
    receipt = {
        "ablation_results": [
            {
                "ablation_key": item["ablation_key"],
                "dimension_statuses": [
                    result["status"] for result in item["dimension_results"]],
            }
            for item in aggregate["ablation_results"]
        ],
        "aggregate_sha256": _strict_sha256(
            aggregate_sha256, label="aggregate"),
        "artifact_kind": "PH2_W07_RUNTIME_EVIDENCE_RECEIPT",
        "candidate_contract_sha256": contract_sha,
        "candidate_evidence": {
            "artifact_counts": host_evidence["artifact_counts"],
            "dump_manifest_sha256": host_evidence["dump_manifest_sha256"],
            "dump_readback": readback["dump_readback"],
            "host_digests": host_evidence["host_digests"],
            "learning_attempt_count": host_evidence["learning_attempt_count"],
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
        "candidate_public_head_commit_sha1": candidate_head,
        "case_commitment": aggregate["case_commitment"],
        "cluster_commitment": aggregate["cluster_commitment"],
        "dimension_results": aggregate["baseline_results"],
        "evaluator_public_head_commit_sha1": evaluator_head,
        "evaluator_version": 2,
        "execution_state": dict(W07_RECEIPT_EXECUTION_STATE),
        "family_commitment": aggregate["family_commitment"],
        "format_version": 1,
        "generation_ablation_statuses": aggregate[
            "generation_ablation_statuses"],
        "generation_hard_conjunct": W07_GENERATION_HARD_CONJUNCT,
        "hard_conjunct_policy": W07_AGGREGATION_POLICY,
        "infrastructure": aggregate["infrastructure"],
        "label_commitment": aggregate["label_commitment"],
        "open_generation_state": W07_OPEN_GENERATION_STATE,
        "parent_identities": parents,
        "payload_commitment": aggregate["payload_commitment"],
        "private_family_freeze_sha256": family_freeze_sha,
        "private_first_run_guard_sha256": private_guard_sha,
        "publication": {
            "candidate_public_head_commit_sha1": candidate_head,
            "evaluator_public_head_commit_sha1": evaluator_head,
            "verification_jobs": jobs,
            "verification_run_id": verification_run_id,
        },
        "receipt_relative_path": W07_PUBLIC_RUNTIME_RECEIPT_RELATIVE_PATH,
        "receipt_self_excluded": 1,
        "recommendation_sha256": _strict_sha256(
            recommendation_sha256, label="recommendation"),
        "required": 1,
        "stage_key": "W-07",
        "status": "RUNTIME_EVIDENCED",
    }
    encoded = canonical_json_bytes(receipt)
    if any(token in encoded for token in _FORBIDDEN_SAFE_OUTPUT_TOKENS):
        raise W07ReleaseError("W-07 runtime receipt 含 private 字段或绝对路径")
    target = root / Path(*W07_PUBLIC_RUNTIME_RECEIPT_RELATIVE_PATH.split("/"))
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(encoded)
    except FileExistsError as error:
        raise W07ReleaseError("W-07 runtime receipt 不可覆盖") from error
    return target, _sha256(encoded)


def read_w07_runtime_receipt(repository_root: str | Path) -> dict[str, Any]:
    """规范回读已发布 receipt，并重验状态、parent 与安全字段。"""
    root = Path(repository_root).resolve()
    path = root / Path(*W07_PUBLIC_RUNTIME_RECEIPT_RELATIVE_PATH.split("/"))
    receipt, payload = _read_canonical(path, label="W-07 runtime receipt")
    required_fields = {
        "ablation_results", "aggregate_sha256", "artifact_kind",
        "candidate_contract_sha256", "candidate_evidence",
        "candidate_first_run_guard_sha256", "candidate_host_freeze_sha256",
        "candidate_public_head_commit_sha1", "case_commitment",
        "cluster_commitment", "dimension_results",
        "evaluator_public_head_commit_sha1", "evaluator_version",
        "execution_state", "family_commitment", "format_version",
        "generation_ablation_statuses", "generation_hard_conjunct",
        "hard_conjunct_policy", "infrastructure", "label_commitment",
        "open_generation_state", "parent_identities", "payload_commitment",
        "private_family_freeze_sha256", "private_first_run_guard_sha256",
        "publication", "receipt_relative_path", "receipt_self_excluded",
        "recommendation_sha256", "required", "stage_key", "status",
    }
    if (set(receipt) != required_fields
            or receipt.get("artifact_kind")
            != "PH2_W07_RUNTIME_EVIDENCE_RECEIPT"
            or receipt.get("format_version") != 1
            or receipt.get("required") != 1
            or receipt.get("stage_key") != "W-07"
            or receipt.get("status") != "RUNTIME_EVIDENCED"
            or receipt.get("receipt_relative_path")
            != W07_PUBLIC_RUNTIME_RECEIPT_RELATIVE_PATH
            or receipt.get("receipt_self_excluded") != 1
            or receipt.get("evaluator_version") != 2
            or receipt.get("execution_state") != W07_RECEIPT_EXECUTION_STATE
            or receipt.get("open_generation_state")
            != W07_OPEN_GENERATION_STATE
            or receipt.get("parent_identities") != _check_public_parents(root)
            or receipt.get("generation_hard_conjunct")
            != W07_GENERATION_HARD_CONJUNCT
            or receipt.get("hard_conjunct_policy")
            != W07_AGGREGATION_POLICY):
        raise W07ReleaseError("W-07 runtime receipt public contract 漂移")
    dimensions = receipt.get("dimension_results")
    ablations = receipt.get("ablation_results")
    publication = receipt.get("publication")
    if (not isinstance(dimensions, list)
            or tuple(item.get("dimension_key") for item in dimensions)
            != W07_PUBLIC_DIMENSION_KEYS
            or any(item.get("status") != "PASS" for item in dimensions)
            or not isinstance(ablations, list)
            or tuple(item.get("ablation_key") for item in ablations)
            != W07_PUBLIC_ABLATION_KEYS
            or any(item.get("dimension_statuses") != [
                "FAIL" if index == ordinal else "PASS"
                for index in range(len(W07_PUBLIC_DIMENSION_KEYS))]
                for ordinal, item in enumerate(ablations))
            or not isinstance(publication, dict)
            or set(publication) != {
                "candidate_public_head_commit_sha1",
                "evaluator_public_head_commit_sha1", "verification_jobs",
                "verification_run_id"}
            or publication.get("candidate_public_head_commit_sha1")
            != receipt.get("candidate_public_head_commit_sha1")
            or publication.get("evaluator_public_head_commit_sha1")
            != receipt.get("evaluator_public_head_commit_sha1")
            or type(publication.get("verification_run_id")) is not int
            or publication["verification_run_id"] <= 0
            or tuple(item.get("job") for item in publication.get(
                "verification_jobs", ())) != W07_REQUIRED_VERIFICATION_JOBS
            or any(item.get("status") != "PASS"
                   for item in publication.get("verification_jobs", ()) )):
        raise W07ReleaseError("W-07 runtime receipt PASS evidence 漂移")
    for dimension, item in zip(
            W07_PUBLIC_DIMENSION_KEYS, dimensions, strict=True):
        _check_result(item, dimension=dimension, expected_status="PASS")
    evaluator_head = _strict_sha1(
        receipt.get("evaluator_public_head_commit_sha1"),
        label="receipt evaluator HEAD")
    _check_infrastructure(
        receipt.get("infrastructure"), evaluator_head=evaluator_head)
    evidence = receipt.get("candidate_evidence")
    if (not isinstance(evidence, dict)
            or set(evidence) != {
                "artifact_counts", "dump_manifest_sha256", "dump_readback",
                "host_digests", "learning_attempt_count",
                "new_learning_write_count", "owned_tables", "payload_bytes",
                "payload_gets", "resource_report", "retention_sha256",
                "transaction_event_count"}
            or dict(evidence["artifact_counts"])
            != W07_EXPECTED_ARTIFACT_COUNTS
            or evidence["dump_readback"] != 1
            or evidence["learning_attempt_count"] != 1
            or evidence["new_learning_write_count"] != 174
            or evidence["owned_tables"]
            != ["graph_object", "ph2_w07_transaction_event"]
            or evidence["payload_bytes"] != 36741
            or evidence["payload_gets"] != 21
            or tuple(tuple(item) for item in evidence["retention_sha256"])
            != W07_EXPECTED_RETENTION_IDENTITIES
            or evidence["transaction_event_count"] != 5):
        raise W07ReleaseError("W-07 runtime receipt candidate evidence 漂移")
    resource = evidence["resource_report"]
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
    if (not isinstance(resource, dict)
            or resource.get("actual_workers") != 4
            or resource.get("teacher_calls") != 0
            or any(type(resource.get(actual)) is not int
                   or resource[actual] < 0
                   or resource[actual] > W07_RESOURCE_BUDGET[maximum]
                   for actual, maximum in budget_pairs)):
        raise W07ReleaseError("W-07 runtime receipt resource 漂移")
    host_digests = evidence["host_digests"]
    if (not isinstance(host_digests, dict)
            or set(host_digests) != {
                "active_projection", "candidate", "carrier_scope", "logic",
                "logical", "source_evidence", "transaction"}):
        raise W07ReleaseError("W-07 runtime receipt host digest 漂移")
    for key, value in host_digests.items():
        _strict_sha256(value, label=f"receipt {key} digest")
    _strict_sha256(
        evidence.get("dump_manifest_sha256"), label="receipt dump manifest")
    for key in (
            "aggregate_sha256", "candidate_contract_sha256",
            "candidate_first_run_guard_sha256",
            "candidate_host_freeze_sha256", "case_commitment",
            "cluster_commitment", "family_commitment", "label_commitment",
            "payload_commitment", "private_family_freeze_sha256",
            "private_first_run_guard_sha256", "recommendation_sha256"):
        _strict_sha256(receipt.get(key), label=f"receipt {key}")
    _strict_sha1(
        receipt.get("candidate_public_head_commit_sha1"),
        label="receipt candidate HEAD")
    if any(token in payload for token in _FORBIDDEN_SAFE_OUTPUT_TOKENS):
        raise W07ReleaseError("W-07 runtime receipt 含 private 字段或绝对路径")
    return receipt


__all__ = [
    "W07_EXPECTED_PARENT_IDENTITIES",
    "W07_EXPECTED_RETENTION_IDENTITIES",
    "W07_PUBLIC_RUNTIME_RECEIPT_NAME",
    "W07_PUBLIC_RUNTIME_RECEIPT_RELATIVE_PATH",
    "W07_RECEIPT_EXECUTION_STATE",
    "W07_REQUIRED_VERIFICATION_JOBS",
    "W07ReleaseError",
    "publish_w07_runtime_receipt",
    "read_w07_runtime_receipt",
]
