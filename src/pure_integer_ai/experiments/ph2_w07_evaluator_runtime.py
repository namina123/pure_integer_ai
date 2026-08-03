"""W-07 private evaluator 的一次性运行、隔离与安全发布。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from pure_integer_ai.experiments.ph2_w07_adapter import adapt_w07_training_payload
from pure_integer_ai.experiments.ph2_w07_candidate import (
    W07_CANDIDATE_CONTRACT_FREEZE_NAME,
    W07_CANDIDATE_HOST_FREEZE_NAME,
    verify_w07_candidate_contract_freeze,
)
from pure_integer_ai.experiments.ph2_w07_contract import (
    W07_BASELINE_COMMIT_SHA1,
    W07_FORMAL_RUN_ID,
    W07_GENERATION_ABLATION_KEY,
    W07_PUBLIC_ABLATION_KEYS,
    W07_RESOURCE_BUDGET,
    W07_RUNNER_KEY,
    W07_STAGE_KEY,
    W07_W06_BASE_RUN_ID,
    W07RunRequest,
    open_w07_frozen_context,
    validate_w07_request,
)
from pure_integer_ai.experiments.ph2_w07_evaluator import (
    W07EvaluatorAblation,
    evaluate_w07_learning_runtime,
)
from pure_integer_ai.experiments.ph2_w07_evaluator_consumers import (
    build_w07_evaluator_consumer_suite,
)
from pure_integer_ai.experiments.ph2_w07_evaluator_contract import (
    W07_EVALUATOR_PHASES,
    W07_PRIVATE_AGGREGATE_NAME,
    W07_PRIVATE_CASE_NAME,
    W07_PRIVATE_CLUSTER_NAME,
    W07_PRIVATE_FAMILY_FREEZE_NAME,
    W07_PRIVATE_HARD_REQUIREMENTS,
    W07_PRIVATE_LABEL_NAME,
    W07_PRIVATE_RECOMMENDATION_NAME,
    W07_PRIVATE_SCHEMA_NAME,
    W07_PRIVATE_SOURCE_NAME,
    W07PrivateEvaluationError,
    decode_w07_private_documents,
    public_safe_w07_aggregate,
)
from pure_integer_ai.experiments.ph2_w07_evaluator_family import (
    consume_w07_private_first_run_guard,
)
from pure_integer_ai.experiments.ph2_w07_firewall import W07PayloadFirewall
from pure_integer_ai.experiments.ph2_w07_runtime import (
    W07RuntimeConfig,
    load_w07_public_dump,
)
from pure_integer_ai.storage.backend import SQLiteBackend


class W07EvaluatorInfrastructureError(RuntimeError):
    """evaluator root、phase、owner 或 candidate 完整性失败。"""


class W07EvaluatorInjectedFault(W07EvaluatorInfrastructureError):
    """在预注册 evaluator phase 注入的受控故障。"""


@dataclass(frozen=True)
class W07PrivateEvaluatorRuntimeConfig:
    repository_root: str | Path
    candidate_root: str | Path
    family_root: str | Path
    execution_root: str | Path
    evaluator_public_head_commit_sha1: str
    fault_phase: str | None = None


@dataclass(frozen=True)
class W07PrivateEvaluatorRunResult:
    status: str
    aggregate_path: Path
    aggregate_sha256: str
    recommendation_path: Path | None
    recommendation_sha256: str | None
    family_freeze_sha256: str
    guard_sha256: str


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _strict_sha1(value: object, *, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 40
            or any(char not in "0123456789abcdef" for char in value)):
        raise W07EvaluatorInfrastructureError(
            f"{label} is not canonical SHA-1")
    return value


def _read_canonical(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    if not path.is_file() or path.is_symlink():
        raise W07EvaluatorInfrastructureError(f"{label} missing or linked")
    payload = path.read_bytes()
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise W07EvaluatorInfrastructureError(f"{label} JSON invalid") from error
    if not isinstance(value, dict) or canonical_json_bytes(value) != payload:
        raise W07EvaluatorInfrastructureError(f"{label} is not canonical")
    return value, payload


def _tree_digest(root: Path) -> str:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if any(part in {
                ".git", ".pytest_cache", ".pytest_tmp_safe", "__pycache__"}
               for part in Path(relative).parts):
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
                raise W07EvaluatorInfrastructureError(
                    "W-07 evaluator owner roots are not isolated")
    if execution == family or not execution.is_relative_to(family):
        raise W07EvaluatorInfrastructureError(
            "W-07 execution root is outside the private family")


def _candidate_documents(
        repository_root: Path,
        candidate_root: Path,
        ) -> tuple[dict[str, Any], bytes, dict[str, Any], bytes]:
    contract, contract_bytes = _read_canonical(
        candidate_root / W07_CANDIDATE_CONTRACT_FREEZE_NAME,
        label="candidate contract")
    contract_sha = _sha256(contract_bytes)
    verified = verify_w07_candidate_contract_freeze(
        repository_root,
        candidate_root,
        candidate_contract_sha256=contract_sha,
    )
    if verified != contract:
        raise W07EvaluatorInfrastructureError("candidate contract readback drift")
    host, host_bytes = _read_canonical(
        candidate_root / W07_CANDIDATE_HOST_FREEZE_NAME,
        label="candidate host freeze")
    state = host.get("execution_state", {})
    if (host.get("candidate_contract_sha256") != contract_sha
            or host.get("formal_run_count") != 1
            or host.get("self_excluded") != 1
            or state.get("W07_STARTED") != 1
            or state.get("formal_w07_training_runs") != 1
            or state.get("teacher_calls") != 0
            or state.get("W08_STARTED") != 0):
        raise W07EvaluatorInfrastructureError(
            "candidate freeze state or binding drift")
    return contract, contract_bytes, host, host_bytes


def _family_documents(
        family_root: Path,
        family_freeze_sha256: str,
        evaluator_head: str,
        ) -> tuple[dict[str, Any], tuple[bytes, ...]]:
    freeze, freeze_bytes = _read_canonical(
        family_root / W07_PRIVATE_FAMILY_FREEZE_NAME,
        label="private family freeze")
    if _sha256(freeze_bytes) != family_freeze_sha256:
        raise W07EvaluatorInfrastructureError("private family freeze SHA drift")
    if (freeze.get("formal_run_count") != 0
            or freeze.get("self_excluded") != 1
            or freeze.get("ablation_order") != list(W07_PUBLIC_ABLATION_KEYS)
            or freeze.get("hard_requirements")
            != list(W07_PRIVATE_HARD_REQUIREMENTS)
            or freeze.get("evaluator_public_head_commit_sha1")
            != evaluator_head):
        raise W07EvaluatorInfrastructureError(
            "private family already ran or fields drifted")
    names = (
        W07_PRIVATE_SOURCE_NAME,
        W07_PRIVATE_SCHEMA_NAME,
        W07_PRIVATE_CASE_NAME,
        W07_PRIVATE_LABEL_NAME,
        W07_PRIVATE_CLUSTER_NAME,
    )
    inventory = freeze.get("file_inventory")
    if not isinstance(inventory, list):
        raise W07EvaluatorInfrastructureError("private family inventory missing")
    by_name = {
        item.get("path"): item for item in inventory if isinstance(item, dict)
    }
    payloads = []
    for name in names:
        path = family_root / name
        if not path.is_file() or path.is_symlink():
            raise W07EvaluatorInfrastructureError("private family file missing")
        payload = path.read_bytes()
        row = by_name.get(name)
        if (not isinstance(row, dict)
                or row.get("sha256") != _sha256(payload)
                or row.get("size_bytes") != len(payload)):
            raise W07EvaluatorInfrastructureError(
                "private family file identity drift")
        payloads.append(payload)
    return freeze, tuple(payloads)


def _dump_root(candidate_root: Path) -> Path:
    manifests = tuple(candidate_root.rglob("w07_dump_manifest.json"))
    if len(manifests) != 1:
        raise W07EvaluatorInfrastructureError("candidate dump count is not one")
    return manifests[0].parent.parent


def _candidate_config(
        config: W07PrivateEvaluatorRuntimeConfig,
        contract: dict[str, Any],
        *,
        run_root: Path,
        sqlite_path: Path,
        ) -> W07RuntimeConfig:
    request = contract["candidate_request"]
    return W07RuntimeConfig(
        repository_root=config.repository_root,
        run_root=run_root,
        sqlite_path=sqlite_path,
        run_id=W07_FORMAL_RUN_ID,
        parent_run_id=W07_W06_BASE_RUN_ID,
        base_run_id=W07_W06_BASE_RUN_ID,
        base_fence_key=tuple(request["base_fence_key"]),
        worker_count=int(request["worker_count"]),
        mode="fresh",
        baseline_commit_sha1=W07_BASELINE_COMMIT_SHA1,
    )


def _build_consumer_suite(
        config: W07PrivateEvaluatorRuntimeConfig,
        contract: dict[str, Any],
        execution_root: Path,
        ):
    profile_backend = SQLiteBackend(
        str(execution_root / "evaluation_profile.sqlite"))
    try:
        profile = profile_backend.storage_capabilities().stable_key()
        context = open_w07_frozen_context(
            Path(config.repository_root),
            baseline_commit_sha1=W07_BASELINE_COMMIT_SHA1,
            backend_profile_key=profile,
        )
        request_data = contract["candidate_request"]
        request = validate_w07_request(context, W07RunRequest(
            run_id=W07_FORMAL_RUN_ID,
            parent_run_id=W07_W06_BASE_RUN_ID,
            base_run_id=W07_W06_BASE_RUN_ID,
            stage_key=W07_STAGE_KEY,
            owner_key=context.owner_key,
            runner_key=W07_RUNNER_KEY,
            baseline_commit_sha1=context.baseline_commit_sha1,
            context_key=context.stable_key(),
            backend_profile_key=profile,
            base_fence_key=tuple(request_data["base_fence_key"]),
            worker_count=int(request_data["worker_count"]),
            mode="fresh",
            resource_budget=tuple(sorted(W07_RESOURCE_BUDGET.items())),
            candidate_payload_paths=tuple(
                item.relative_path for item in context.candidate_payload_bindings),
            teacher_evidence_paths=tuple(
                item.relative_path for item in context.teacher_evidence_bindings),
        ))
        payload = W07PayloadFirewall.open(
            Path(config.repository_root), context, request,
        ).read_training_payload()
    finally:
        profile_backend.close()
    adapter = adapt_w07_training_payload(payload)
    suite = build_w07_evaluator_consumer_suite(
        config.repository_root,
        adapter,
        backend_factory=lambda substage: SQLiteBackend(str(
            execution_root / f"logic_{substage.casefold()}.sqlite")),
    )
    return suite, payload


def _enter_phase(config: W07PrivateEvaluatorRuntimeConfig, phase: str) -> None:
    if config.fault_phase not in {None, *W07_EVALUATOR_PHASES}:
        raise ValueError("unknown W-07 evaluator phase")
    if config.fault_phase == phase:
        raise W07EvaluatorInjectedFault(phase)


def _write_exclusive(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as error:
        raise W07EvaluatorInfrastructureError(
            "private output is immutable") from error
    return _sha256(payload)


def _publish_failure(
        family_root: Path,
        *,
        family: dict[str, Any],
        phase: str,
        family_freeze_sha256: str,
        guard_sha256: str,
        ) -> W07PrivateEvaluatorRunResult:
    aggregate = public_safe_w07_aggregate(
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
        "evaluator_label_writes": 0,
        "host_copy_unchanged": 0,
        "public_repo_writes": 0,
    }
    encoded = canonical_json_bytes(aggregate)
    path = family_root / "publication" / W07_PRIVATE_AGGREGATE_NAME
    aggregate_sha = _write_exclusive(path, encoded)
    return W07PrivateEvaluatorRunResult(
        "NE", path, aggregate_sha, None, None,
        family_freeze_sha256, guard_sha256)


def run_w07_private_evaluation_once(
        config: W07PrivateEvaluatorRuntimeConfig,
        *,
        family_freeze_sha256: str,
        ) -> W07PrivateEvaluatorRunResult:
    """消费唯一 guard，并封存 PASS、FAIL 或枚举 NE。"""
    if not isinstance(config, W07PrivateEvaluatorRuntimeConfig):
        raise TypeError("W-07 evaluator config type drift")
    evaluator_head = _strict_sha1(
        config.evaluator_public_head_commit_sha1, label="evaluator HEAD")
    repository = Path(config.repository_root).resolve()
    candidate = Path(config.candidate_root).resolve()
    family_root = Path(config.family_root).resolve()
    execution = Path(config.execution_root).resolve()
    _validate_roots(repository, candidate, family_root, execution)
    contract, contract_bytes, host, host_bytes = _candidate_documents(
        repository, candidate)
    family, documents = _family_documents(
        family_root, family_freeze_sha256, evaluator_head)
    if (family.get("candidate_contract_sha256") != _sha256(contract_bytes)
            or family.get("candidate_host_freeze_sha256")
            != _sha256(host_bytes)):
        raise W07EvaluatorInfrastructureError(
            "private family candidate binding drift")
    candidate_before = _tree_digest(candidate)
    label_path = family_root / W07_PRIVATE_LABEL_NAME
    label_before = _sha256(label_path.read_bytes())
    repository_before = _tree_digest(repository)
    _guard_path, guard_sha = consume_w07_private_first_run_guard(
        family_root, family_freeze_sha256=family_freeze_sha256)
    current_phase = "PAYLOAD_DECODE"
    try:
        _enter_phase(config, current_phase)
        private_payload = decode_w07_private_documents(*documents)
        if private_payload.evaluator_public_head_commit_sha1 != evaluator_head:
            raise W07EvaluatorInfrastructureError(
                "private source evaluator HEAD drift")
        execution.mkdir(parents=True, exist_ok=False)
        current_phase = "CLONE_LOAD"
        _enter_phase(config, current_phase)
        dump_root = _dump_root(candidate)
        clone_config = _candidate_config(
            config,
            contract,
            run_root=dump_root,
            sqlite_path=execution / "clone.sqlite",
        )
        clone_outcome = load_w07_public_dump(clone_config)
        current_phase = "HOST_COPY"
        _enter_phase(config, current_phase)
        dump_path = dump_root / f"w07_run_{W07_FORMAL_RUN_ID:020d}" / (
            "w07_dump_manifest.json")
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
            clone_outcome.logic_digest == host_digest["logic"],
            clone_outcome.source_evidence_digest == host_digest["source_evidence"],
            clone_outcome.active_projection_digest == host_digest["active_projection"],
            clone_outcome.carrier_scope_digest == host_digest["carrier_scope"],
            clone_outcome.transaction_digest == host_digest["transaction"],
        ))
        artifact_counts = dict(clone_outcome.artifact_counts)
        carrier_scope_ok = all((
            clone_outcome.carrier_scope_digest == host_digest["carrier_scope"],
            artifact_counts.get("CARRIER_PROJECTION") == 9,
            artifact_counts.get("LOGIC_SCOPE_CELL") == 189,
            artifact_counts.get("LOGIC_USE") == 21,
        ))
        if not clone_digest_match or not carrier_scope_ok:
            raise W07EvaluatorInfrastructureError(
                "candidate dump and host freeze drift")
        clone_ok = int(clone_copy_sha == host_copy_sha)
        current_phase = "BASELINE"
        _enter_phase(config, current_phase)
        suite, _ = _build_consumer_suite(config, contract, execution)
        try:
            baseline = evaluate_w07_learning_runtime(
                suite, private_payload.cases, evaluation_ordinal=0)
            baseline_audit = suite.audit()
            ablation_results = []
            gate_passes = []
            for ordinal, key in enumerate(W07_PUBLIC_ABLATION_KEYS):
                phase = W07_EVALUATOR_PHASES[5 + ordinal]
                current_phase = phase
                _enter_phase(config, phase)
                values = evaluate_w07_learning_runtime(
                    suite,
                    private_payload.cases,
                    ablation=W07EvaluatorAblation(key),
                    evaluation_ordinal=ordinal + 1,
                )
                expected = tuple(
                    "FAIL" if index == ordinal else "PASS"
                    for index in range(len(W07_PUBLIC_DIMENSION_KEYS)))
                statuses = tuple(item.status for item in values)
                gate_passes.append(
                    statuses == expected
                    and not any(item.ne_count for item in values))
                ablation_results.append({
                    "ablation_key": key,
                    "dimension_statuses": list(statuses),
                })
            generation_statuses = list(next(
                item for item in ablation_results
                if item["ablation_key"] == W07_GENERATION_ABLATION_KEY
            )["dimension_statuses"])
        finally:
            suite.close()
        current_phase = "INTEGRITY"
        _enter_phase(config, current_phase)
        candidate_after = _tree_digest(candidate)
        label_after = _sha256(label_path.read_bytes())
        repository_after = _tree_digest(repository)
        host_writes = int(candidate_after != candidate_before)
        label_writes = int(label_after != label_before)
        public_writes = int(repository_after != repository_before)
        if host_writes or label_writes or public_writes:
            raise W07EvaluatorInfrastructureError(
                "private evaluator owner isolation failed")
        aggregate = public_safe_w07_aggregate(
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
            "baseline_consumer_audit": baseline_audit,
            "candidate_inventory_match": int(clone_digest_match),
            "carrier_projection_count": artifact_counts["CARRIER_PROJECTION"],
            "carrier_scope_digest_match": int(carrier_scope_ok),
            "clone_dump_readback": int(clone_outcome.dump_readback),
            "clone_host_copy_match": clone_ok,
            "evaluator_label_writes": label_writes,
            "evaluator_public_head_commit_sha1": evaluator_head,
            "host_copy_unchanged": 1,
            "logic_scope_cell_count": artifact_counts["LOGIC_SCOPE_CELL"],
            "logic_use_count": artifact_counts["LOGIC_USE"],
            "public_repo_writes": public_writes,
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
            b"surface", b"expected", b"private" + b"_path", b"message",
        )
        if any(token in encoded for token in forbidden):
            raise W07EvaluatorInfrastructureError(
                "safe aggregate leaked private fields")
        aggregate_path = family_root / "publication" / W07_PRIVATE_AGGREGATE_NAME
        aggregate_sha = _write_exclusive(aggregate_path, encoded)
        recommendation_path = None
        recommendation_sha = None
        if aggregate["status"] == "PASS" and all(gate_passes):
            recommendation = canonical_json_bytes({
                "aggregate_sha256": aggregate_sha,
                "artifact_kind": "PH2_W07_RUNTIME_RECEIPT_RECOMMENDATION",
                "candidate_contract_sha256": family[
                    "candidate_contract_sha256"],
                "candidate_host_freeze_sha256": family[
                    "candidate_host_freeze_sha256"],
                "evaluator_public_head_commit_sha1": evaluator_head,
                "family_commitment": family["family_key"],
                "formal_run_count": 1,
                "format_version": 1,
                "recommend_runtime_receipt": 1,
            })
            recommendation_path = family_root / "publication" / (
                W07_PRIVATE_RECOMMENDATION_NAME)
            recommendation_sha = _write_exclusive(
                recommendation_path, recommendation)
        return W07PrivateEvaluatorRunResult(
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
    "W07EvaluatorInfrastructureError",
    "W07EvaluatorInjectedFault",
    "W07PrivateEvaluatorRunResult",
    "W07PrivateEvaluatorRuntimeConfig",
    "run_w07_private_evaluation_once",
]
