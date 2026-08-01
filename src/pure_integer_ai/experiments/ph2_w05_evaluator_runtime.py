"""W-05 private evaluator 的一次性运行、隔离和安全发布。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from pure_integer_ai.experiments.ph2_w05_adapter import adapt_w05_training_payload
from pure_integer_ai.experiments.ph2_w05_contract import (
    W05_FORMAL_RUN_ID,
    W05_PRIVATE_ABLATION_KEYS,
    W05_RUNNER_KEY,
    W05_W04_BASE_RUN_ID,
    W05RunRequest,
    digest_value,
    open_w05_frozen_context,
)
from pure_integer_ai.experiments.ph2_w05_evaluator import (
    W05EvaluatorAblation,
    evaluate_w05_learning_runtime,
)
from pure_integer_ai.experiments.ph2_w05_evaluator_contract import (
    W05_EVALUATOR_PHASES,
    W05_GENERATION_ABLATION_KEY,
    W05_PRIVATE_AGGREGATE_NAME,
    W05_PRIVATE_CASE_NAME,
    W05_PRIVATE_CLUSTER_NAME,
    W05_PRIVATE_FAMILY_FREEZE_NAME,
    W05_PRIVATE_FIRST_RUN_GUARD_NAME,
    W05_PRIVATE_LABEL_NAME,
    W05_PRIVATE_RECOMMENDATION_NAME,
    W05_PRIVATE_SCHEMA_NAME,
    W05_PRIVATE_SOURCE_NAME,
    W05PrivateEvaluationError,
    decode_w05_private_documents,
    public_safe_w05_aggregate,
)
from pure_integer_ai.experiments.ph2_w05_evaluator_family import (
    consume_w05_private_first_run_guard,
)
from pure_integer_ai.experiments.ph2_w05_firewall import W05PayloadFirewall
from pure_integer_ai.experiments.ph2_w05_learning import (
    build_w05_learning_runtime,
)
from pure_integer_ai.experiments.ph2_w05_runtime import (
    W05RuntimeConfig,
    load_w05_candidate_dump,
)
from pure_integer_ai.storage.backend import SQLiteBackend


class W05EvaluatorInfrastructureError(RuntimeError):
    """private evaluator 的 owner/root/phase/integrity 错误。"""


class W05EvaluatorInjectedFault(W05EvaluatorInfrastructureError):
    """冻结 phase registry 中的受控故障。"""


@dataclass(frozen=True)
class W05PrivateEvaluatorRuntimeConfig:
    repository_root: str | Path
    global_manifest_path: str
    candidate_root: str | Path
    family_root: str | Path
    execution_root: str | Path
    current_remote_commit_sha1: str
    fault_phase: str | None = None
    dependency_root: str | Path | None = None


@dataclass(frozen=True)
class W05PrivateEvaluatorRunResult:
    status: str
    aggregate_path: Path
    aggregate_sha256: str
    recommendation_path: Path | None
    recommendation_sha256: str | None
    family_freeze_sha256: str
    guard_sha256: str


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_canonical(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    if not path.is_file() or path.is_symlink():
        raise W05EvaluatorInfrastructureError(f"{label} 缺失或为链接")
    payload = path.read_bytes()
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise W05EvaluatorInfrastructureError(f"{label} JSON 非法") from exc
    if not isinstance(value, dict) or canonical_json_bytes(value) != payload:
        raise W05EvaluatorInfrastructureError(f"{label} 非 canonical object")
    return value, payload


def _tree_digest(root: Path) -> str:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if any(part in {".pytest_cache", "__pycache__"} for part in Path(relative).parts):
            continue
        rows.append((relative, _sha256(path.read_bytes())))
    return _sha256(canonical_json_bytes(rows))


def _validate_roots(
        repository: Path,
        candidate: Path,
        family: Path,
        execution: Path,
        ) -> None:
    owners = (repository, candidate, family)
    for index, left in enumerate(owners):
        for right in owners[index + 1:]:
            if (left == right or left.is_relative_to(right)
                    or right.is_relative_to(left)):
                raise W05EvaluatorInfrastructureError("W-05 evaluator owner root 未隔离")
    if execution == family or not execution.is_relative_to(family):
        raise W05EvaluatorInfrastructureError("W-05 evaluator execution root 未位于 family")


def _candidate_documents(
        candidate_root: Path,
        ) -> tuple[dict[str, Any], bytes, dict[str, Any], bytes]:
    contract, contract_bytes = _read_canonical(
        candidate_root / "candidate_contract_freeze.json",
        label="candidate contract")
    host, host_bytes = _read_canonical(
        candidate_root / "candidate_host_freeze.json",
        label="candidate host freeze")
    contract_sha = _sha256(contract_bytes)
    if (host.get("candidate_contract_sha256") != contract_sha
            or host.get("formal_run_count") != 1
            or host.get("self_excluded") != 1
            or host.get("execution_state", {}).get("W05_STARTED") != 1):
        raise W05EvaluatorInfrastructureError("candidate freeze 状态或绑定漂移")
    return contract, contract_bytes, host, host_bytes


def _family_documents(
        family_root: Path,
        family_freeze_sha256: str,
        ) -> tuple[dict[str, Any], tuple[bytes, ...]]:
    freeze, freeze_bytes = _read_canonical(
        family_root / W05_PRIVATE_FAMILY_FREEZE_NAME,
        label="private family freeze")
    if _sha256(freeze_bytes) != family_freeze_sha256:
        raise W05EvaluatorInfrastructureError("private family freeze SHA 漂移")
    if (freeze.get("formal_run_count") != 0
            or freeze.get("self_excluded") != 1
            or freeze.get("ablation_order")
            != list(W05_PRIVATE_ABLATION_KEYS)):
        raise W05EvaluatorInfrastructureError("private family 已运行或字段漂移")
    names = (
        W05_PRIVATE_SOURCE_NAME,
        W05_PRIVATE_SCHEMA_NAME,
        W05_PRIVATE_CASE_NAME,
        W05_PRIVATE_LABEL_NAME,
        W05_PRIVATE_CLUSTER_NAME,
    )
    payloads = []
    inventory = freeze.get("file_inventory")
    if not isinstance(inventory, list):
        raise W05EvaluatorInfrastructureError("private family inventory 缺失")
    by_name = {item.get("path"): item for item in inventory
               if isinstance(item, dict)}
    for name in names:
        path = family_root / name
        payload = path.read_bytes()
        row = by_name.get(name)
        if (not isinstance(row, dict)
                or row.get("sha256") != _sha256(payload)
                or row.get("size_bytes") != len(payload)):
            raise W05EvaluatorInfrastructureError("private family file identity 漂移")
        payloads.append(payload)
    return freeze, tuple(payloads)


def _dump_root(candidate_root: Path) -> Path:
    manifests = tuple(candidate_root.rglob("w05_dump_manifest.json"))
    if len(manifests) != 1:
        raise W05EvaluatorInfrastructureError("candidate dump 数量不是 1")
    return manifests[0].parent.parent


def _candidate_config(
        config: W05PrivateEvaluatorRuntimeConfig,
        contract: dict[str, Any],
        *,
        run_root: Path,
        sqlite_path: Path,
        ) -> W05RuntimeConfig:
    request = contract["candidate_request"]
    return W05RuntimeConfig(
        repository_root=config.repository_root,
        global_manifest_path=config.global_manifest_path,
        run_root=run_root,
        sqlite_path=sqlite_path,
        run_id=W05_FORMAL_RUN_ID,
        parent_run_id=W05_W04_BASE_RUN_ID,
        base_run_id=W05_W04_BASE_RUN_ID,
        base_fence_key=tuple(request["base_fence_key"]),
        worker_count=int(request["worker_count"]),
        mode="fresh",
        current_remote_commit_sha1=config.current_remote_commit_sha1,
        dependency_root=config.dependency_root,
    )


def _build_learning(
        config: W05PrivateEvaluatorRuntimeConfig,
        contract: dict[str, Any],
        execution_root: Path,
        ) -> tuple[SQLiteBackend, Any, Any]:
    backend = SQLiteBackend(str(execution_root / "evaluation.sqlite"))
    profile = backend.storage_capabilities().stable_key()
    context = open_w05_frozen_context(
        Path(config.repository_root),
        config.global_manifest_path,
        current_remote_commit_sha1=config.current_remote_commit_sha1,
        backend_profile_key=profile,
        dependency_root=config.dependency_root,
    )
    request_data = contract["candidate_request"]
    request = W05RunRequest(
        run_id=W05_FORMAL_RUN_ID,
        parent_run_id=W05_W04_BASE_RUN_ID,
        base_run_id=W05_W04_BASE_RUN_ID,
        stage_key=context.stage_key,
        owner_key=context.owner_key,
        runner_key=W05_RUNNER_KEY,
        current_remote_commit_sha1=context.current_remote_commit_sha1,
        pre_w04_gate_key=context.pre_w04_gate_key,
        w04_receipt_key=digest_value(context.w04_receipt_identity.to_dict()),
        d03_context_key=context.stable_key(),
        backend_profile_key=profile,
        base_fence_key=tuple(request_data["base_fence_key"]),
        worker_count=int(request_data["worker_count"]),
        mode="fresh",
        resource_budget=tuple(sorted(context.resource_budget.items())),
        candidate_payload_paths=tuple(
            item.relative_path for item in context.candidate_payload_bindings),
        teacher_evidence_paths=tuple(
            item.relative_path for item in context.teacher_evidence_bindings),
    )
    payload = W05PayloadFirewall.open(
        Path(config.repository_root),
        context,
        request,
        dependency_root=config.dependency_root,
    ).read_training_payload()
    adapter = adapt_w05_training_payload(payload)
    return backend, build_w05_learning_runtime(backend, adapter), payload


def _enter_phase(config: W05PrivateEvaluatorRuntimeConfig, phase: str) -> None:
    if config.fault_phase not in {None, *W05_EVALUATOR_PHASES}:
        raise ValueError("未知 W-05 evaluator phase")
    if config.fault_phase == phase:
        raise W05EvaluatorInjectedFault(phase)


def _write_exclusive(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as exc:
        raise W05EvaluatorInfrastructureError("private output 不可覆盖") from exc
    return _sha256(payload)


def _publish_failure(
        family_root: Path,
        *,
        family: dict[str, Any],
        phase: str,
        family_freeze_sha256: str,
        guard_sha256: str,
        ) -> W05PrivateEvaluatorRunResult:
    aggregate = public_safe_w05_aggregate(
        (),
        family_commitment=family["family_key"],
        payload_commitment=family["payload_commitment"],
        case_commitment=family["case_commitment"],
        label_commitment=family["label_commitment"],
        cluster_commitment=family["cluster_commitment"],
        failure_phase=phase,
        formal_run_count=1,
        host_writes=0,
        label_writes=0,
    )
    aggregate["infrastructure"] = {
        "candidate_inventory_match": 0,
        "clone_dump_readback": 0,
        "clone_host_copy_match": 0,
        "host_copy_unchanged": 0,
        "evaluator_label_writes": 0,
        "public_repo_writes": 0,
    }
    encoded = canonical_json_bytes(aggregate)
    path = family_root / "publication" / W05_PRIVATE_AGGREGATE_NAME
    aggregate_sha = _write_exclusive(path, encoded)
    return W05PrivateEvaluatorRunResult(
        "NE", path, aggregate_sha, None, None,
        family_freeze_sha256, guard_sha256)


def run_w05_private_evaluation_once(
        config: W05PrivateEvaluatorRuntimeConfig,
        *,
        family_freeze_sha256: str,
        ) -> W05PrivateEvaluatorRunResult:
    """消费 private guard，完成一次 clone/readback 与五项正交评测。"""
    if not isinstance(config, W05PrivateEvaluatorRuntimeConfig):
        raise TypeError("W-05 evaluator config 类型非法")
    repository = Path(config.repository_root).resolve()
    candidate = Path(config.candidate_root).resolve()
    family_root = Path(config.family_root).resolve()
    execution = Path(config.execution_root).resolve()
    _validate_roots(repository, candidate, family_root, execution)
    contract, _, host, _ = _candidate_documents(candidate)
    family, documents = _family_documents(family_root, family_freeze_sha256)
    if (family.get("candidate_contract_sha256")
            != _sha256((candidate / "candidate_contract_freeze.json").read_bytes())
            or family.get("candidate_host_freeze_sha256")
            != _sha256((candidate / "candidate_host_freeze.json").read_bytes())):
        raise W05EvaluatorInfrastructureError("private family candidate binding 漂移")
    candidate_before = _tree_digest(candidate)
    label_path = family_root / W05_PRIVATE_LABEL_NAME
    label_before = _sha256(label_path.read_bytes())
    repository_before = _tree_digest(repository)
    _guard_path, guard_sha = consume_w05_private_first_run_guard(
        family_root, family_freeze_sha256=family_freeze_sha256)
    current_phase = "PAYLOAD_DECODE"
    try:
        _enter_phase(config, current_phase)
        private_payload = decode_w05_private_documents(*documents)
        execution.mkdir(parents=True, exist_ok=False)
        current_phase = "CLONE_LOAD"
        _enter_phase(config, current_phase)
        dump_root = _dump_root(candidate)
        clone_config = _candidate_config(
            config, contract, run_root=dump_root,
            sqlite_path=execution / "clone.sqlite",
        )
        clone_outcome = load_w05_candidate_dump(clone_config)
        current_phase = "HOST_COPY"
        _enter_phase(config, current_phase)
        dump_path = dump_root / f"w05_run_{W05_FORMAL_RUN_ID:020d}" / (
            "w05_dump_manifest.json")
        host_copy = execution / "host_copy.dump"
        shutil.copyfile(dump_path, host_copy)
        clone_copy_sha = _sha256(dump_path.read_bytes())
        host_copy_sha = _sha256(host_copy.read_bytes())
        current_phase = "CLONE_COMPARE"
        _enter_phase(config, current_phase)
        host_digest = host["host_evidence"]["host_digests"]
        clone_digest_match = all((
            clone_outcome.logical_state_digest == host_digest["logical"],
            clone_outcome.candidate_digest == host_digest["candidate"],
            clone_outcome.understanding_digest == host_digest["understanding"],
            clone_outcome.reasoning_digest == host_digest["reasoning"],
            clone_outcome.generation_digest == host_digest["generation"],
            clone_outcome.carrier_scope_digest == host_digest["carrier_scope"],
            clone_outcome.transaction_digest == host_digest["transaction"],
        ))
        artifact_counts = dict(clone_outcome.artifact_counts)
        carrier_scope_ok = (
            clone_outcome.carrier_scope_digest == host_digest["carrier_scope"]
            and artifact_counts.get("CARRIER_PROJECTION") == 9
            and artifact_counts.get("ROLE_PROPOSITION_SCOPE_CELL") == 27
        )
        if not clone_digest_match or not carrier_scope_ok:
            raise W05EvaluatorInfrastructureError("candidate dump 与 host freeze 漂移")
        clone_ok = int(clone_copy_sha == host_copy_sha)
        current_phase = "BASELINE"
        _enter_phase(config, current_phase)
        backend, learning, _ = _build_learning(config, contract, execution)
        try:
            baseline = evaluate_w05_learning_runtime(
                learning, private_payload.cases)
            ablation_results = []
            gate_passes = []
            for ordinal, key in enumerate(W05_PRIVATE_ABLATION_KEYS):
                phase = W05_EVALUATOR_PHASES[5 + ordinal]
                current_phase = phase
                _enter_phase(config, phase)
                values = evaluate_w05_learning_runtime(
                    learning,
                    private_payload.cases,
                    ablation=W05EvaluatorAblation(key),
                )
                target = values[ordinal].dimension_key
                expected = tuple(
                    "FAIL" if item.dimension_key == target else "PASS"
                    for item in values
                )
                statuses = tuple(item.status for item in values)
                gate_passes.append(statuses == expected
                                  and not any(item.ne_count for item in values))
                ablation_results.append({
                    "ablation_key": key,
                    "dimension_statuses": list(statuses),
                })
            generation_statuses = list(
                next(item for item in ablation_results
                     if item["ablation_key"] == W05_GENERATION_ABLATION_KEY)
                ["dimension_statuses"])
        finally:
            backend.close()
        current_phase = "INTEGRITY"
        _enter_phase(config, current_phase)
        candidate_after = _tree_digest(candidate)
        label_after = _sha256(label_path.read_bytes())
        repository_after = _tree_digest(repository)
        host_writes = int(candidate_after != candidate_before)
        label_writes = int(label_after != label_before)
        public_writes = int(repository_after != repository_before)
        if host_writes or label_writes or public_writes:
            raise W05EvaluatorInfrastructureError("private evaluator owner isolation 失败")
        aggregate = public_safe_w05_aggregate(
            baseline,
            family_commitment=family["family_key"],
            payload_commitment=family["payload_commitment"],
            case_commitment=family["case_commitment"],
            label_commitment=family["label_commitment"],
            cluster_commitment=family["cluster_commitment"],
            failure_phase="NONE",
            formal_run_count=1,
            host_writes=host_writes,
            label_writes=label_writes,
        )
        aggregate["ablation_results"] = ablation_results
        aggregate["generation_ablation_statuses"] = generation_statuses
        aggregate["infrastructure"] = {
            "candidate_inventory_match": int(clone_digest_match),
            "carrier_projection_count": artifact_counts["CARRIER_PROJECTION"],
            "carrier_scope_digest_match": int(carrier_scope_ok),
            "clone_dump_readback": int(clone_outcome.dump_readback),
            "clone_host_copy_match": clone_ok,
            "evaluator_label_writes": label_writes,
            "host_copy_unchanged": 1,
            "public_repo_writes": public_writes,
            "role_proposition_scope_cell_count": artifact_counts[
                "ROLE_PROPOSITION_SCOPE_CELL"],
        }
        if not all(gate_passes):
            aggregate["status"] = "FAIL"
            aggregate["fail_count"] = max(int(aggregate["fail_count"]), 1)
        current_phase = "REPORT_SAFETY"
        _enter_phase(config, current_phase)
        encoded = canonical_json_bytes(aggregate)
        forbidden = (
            *(item.case_key.encode("utf-8") for item in private_payload.cases),
            *(item.label_key.encode("utf-8") for item in private_payload.labels),
            b"surface", b"expected", b"private_path", b"message",
        )
        if any(token in encoded for token in forbidden):
            raise W05EvaluatorInfrastructureError("安全 aggregate 泄漏 private 字段")
        aggregate_path = family_root / "publication" / W05_PRIVATE_AGGREGATE_NAME
        aggregate_sha = _write_exclusive(aggregate_path, encoded)
        recommendation_path = None
        recommendation_sha = None
        if aggregate["status"] == "PASS" and all(gate_passes):
            recommendation = canonical_json_bytes({
                "aggregate_sha256": aggregate_sha,
                "artifact_kind": "PH2_W05_RUNTIME_RECEIPT_RECOMMENDATION",
                "candidate_host_freeze_sha256": family[
                    "candidate_host_freeze_sha256"],
                "family_commitment": family["family_key"],
                "formal_run_count": 1,
                "format_version": 1,
                "recommend_runtime_receipt": 1,
            })
            recommendation_path = family_root / "publication" / (
                W05_PRIVATE_RECOMMENDATION_NAME)
            recommendation_sha = _write_exclusive(
                recommendation_path, recommendation)
        return W05PrivateEvaluatorRunResult(
            str(aggregate["status"]),
            aggregate_path,
            aggregate_sha,
            recommendation_path,
            recommendation_sha,
            family_freeze_sha256,
            guard_sha,
        )
    except Exception:
        return _publish_failure(
            family_root,
            family=family,
            phase=current_phase,
            family_freeze_sha256=family_freeze_sha256,
            guard_sha256=guard_sha,
        )


__all__ = [
    "W05EvaluatorInfrastructureError",
    "W05EvaluatorInjectedFault",
    "W05PrivateEvaluatorRunResult",
    "W05PrivateEvaluatorRuntimeConfig",
    "run_w05_private_evaluation_once",
]
