"""W-03 private evaluator 的 fresh clone、host-copy、五维执行与安全发布。"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from pure_integer_ai.experiments.ph2_d03_release_catalog import (
    FORMAL_GLOBAL_MANIFEST_PATH,
)
from pure_integer_ai.experiments.ph2_w03_artifacts import (
    ARTIFACT_GENERATION_OUTCOME,
    W03ArtifactStore,
    restore_training_payload,
)
from pure_integer_ai.experiments.ph2_w03_candidate import (
    W03_CANDIDATE_CONTRACT_FREEZE_NAME,
    W03_CANDIDATE_HOST_FREEZE_NAME,
    W03_FORMAL_EXECUTION_STATE,
    formalize_w03_candidate_outcome,
)
from pure_integer_ai.experiments.ph2_w03_context import open_w03_frozen_context
from pure_integer_ai.experiments.ph2_w03_continuity import (
    formal_w03_publication_baseline,
    verify_formal_w02_continuity,
)
from pure_integer_ai.experiments.ph2_w03_contract import (
    W03_ABLATION_KEYS,
    W03_DIMENSION_KEYS,
)
from pure_integer_ai.experiments.ph2_w03_evaluator import (
    W03EvaluatorAblation,
    evaluate_w03_learning_runtime,
)
from pure_integer_ai.experiments.ph2_w03_evaluator_contract import (
    W03_EVALUATOR_FAILURE_PHASES,
    W03_EVALUATOR_PHASES,
    W03_PRIVATE_AGGREGATE_NAME,
    W03_PRIVATE_FAMILY_FREEZE_NAME,
    W03_PRIVATE_RECOMMENDATION_NAME,
    W03PrivateDimensionResult,
    W03PrivatePayload,
    decode_w03_private_documents,
    evidence_commitment,
    public_safe_w03_aggregate,
)
from pure_integer_ai.experiments.ph2_w03_evaluator_family import (
    consume_w03_private_first_run_guard,
)
from pure_integer_ai.experiments.ph2_w03_learning import run_w03_learning
from pure_integer_ai.experiments.ph2_w03_runtime import (
    W03RunOutcome,
    W03RuntimeConfig,
    load_w03_candidate_dump,
)
from pure_integer_ai.storage.backend import SQLiteBackend


_PRIVATE_DOCUMENT_NAMES = (
    "private_source.json",
    "private_schema.json",
    "private_cases.json",
    "private_labels.json",
    "private_clusters.json",
)


class W03EvaluatorInfrastructureError(RuntimeError):
    """clone、host-copy、identity、zero-write 或 publication 基础设施错误。"""


class W03EvaluatorInjectedFault(W03EvaluatorInfrastructureError):
    """仅供 formal family 创建前的 synthetic phase/fault 测试。"""

    def __init__(self, phase: str) -> None:
        super().__init__("W-03 evaluator synthetic phase fault")
        self.failure_phase = phase


class _W03EvaluatorPhaseError(W03EvaluatorInfrastructureError):
    """把内部异常收敛为冻结 phase，不让动态 message 进入报告。"""

    def __init__(self, phase: str) -> None:
        super().__init__("W-03 evaluator phase infrastructure failure")
        self.failure_phase = phase


@dataclass(frozen=True)
class W03PrivateEvaluatorRuntimeConfig:
    """正式 candidate、private family 与临时 clone owner 的物理边界。"""

    repository_root: str | Path
    w02_artifacts_root: str | Path
    candidate_root: str | Path
    family_root: str | Path
    execution_root: str | Path
    global_manifest_path: str = FORMAL_GLOBAL_MANIFEST_PATH
    fault_phase: str | None = None


@dataclass(frozen=True)
class W03PrivateEvaluatorRunResult:
    """唯一正式 evaluator 的安全 aggregate、推荐件与零写证据。"""

    aggregate_path: Path
    aggregate_sha256: str
    recommendation_path: Path | None
    recommendation_sha256: str | None
    aggregate: dict[str, object]
    phase_counts: tuple[tuple[str, int], ...]
    clone_outcome: W03RunOutcome
    host_copy_sha256: str
    candidate_host_writes: int
    label_writes: int


def _sha256(path: Path) -> str:
    """读取普通文件 SHA，并拒绝缺失、目录或 symlink。"""
    if not path.is_file() or path.is_symlink():
        raise W03EvaluatorInfrastructureError("evaluator identity 文件缺失或为链接")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_canonical(path: Path, *, label: str) -> dict[str, Any]:
    """严格读取 canonical JSON object。"""
    if not path.is_file() or path.is_symlink():
        raise W03EvaluatorInfrastructureError(f"{label} 文件缺失或为链接")
    payload = path.read_bytes()
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise W03EvaluatorInfrastructureError(f"{label} JSON 无法解析") from exc
    if not isinstance(value, dict) or canonical_json_bytes(value) != payload:
        raise W03EvaluatorInfrastructureError(f"{label} JSON 非 canonical object")
    return value


def _resolve(root: Path, relative: str) -> Path:
    """在 owner root 内解析规范 POSIX 相对路径。"""
    if (not isinstance(relative, str) or not relative
            or PurePosixPath(relative).as_posix() != relative
            or PurePosixPath(relative).is_absolute()
            or ".." in PurePosixPath(relative).parts):
        raise W03EvaluatorInfrastructureError("evaluator artifact path 非规范")
    target = (root / Path(*PurePosixPath(relative).parts)).resolve()
    if not target.is_relative_to(root):
        raise W03EvaluatorInfrastructureError("evaluator artifact path 逃逸")
    return target


def _inventory_state(root: Path, inventory: list[dict[str, Any]]) -> tuple:
    """逐项回验 freeze inventory 并返回完整 identity tuple。"""
    result = []
    if not isinstance(inventory, list) or not inventory:
        raise W03EvaluatorInfrastructureError("evaluator inventory 为空")
    for row in inventory:
        if (not isinstance(row, dict)
                or set(row) != {"path", "sha256", "size_bytes"}):
            raise W03EvaluatorInfrastructureError("evaluator inventory 行非法")
        path = _resolve(root, row["path"])
        payload = path.read_bytes() if path.is_file() and not path.is_symlink() else None
        if (payload is None or len(payload) != row["size_bytes"]
                or hashlib.sha256(payload).hexdigest() != row["sha256"]):
            raise W03EvaluatorInfrastructureError("evaluator inventory identity 漂移")
        result.append((row["path"], row["size_bytes"], row["sha256"]))
    return tuple(result)


def _verify_candidate(
        candidate_root: Path,
        *,
        contract_sha256: str,
        host_sha256: str,
        ) -> tuple[dict[str, Any], dict[str, Any], tuple]:
    """回验 candidate 三件 freeze 与 self-excluded host inventory。"""
    contract_path = candidate_root / W03_CANDIDATE_CONTRACT_FREEZE_NAME
    host_path = candidate_root / W03_CANDIDATE_HOST_FREEZE_NAME
    if _sha256(contract_path) != contract_sha256 or _sha256(host_path) != host_sha256:
        raise W03EvaluatorInfrastructureError("candidate contract/host freeze SHA 漂移")
    contract = _read_canonical(contract_path, label="candidate contract")
    host = _read_canonical(host_path, label="candidate host freeze")
    if (host.get("candidate_contract_sha256") != contract_sha256
            or host.get("formal_run_count") != 1
            or host.get("execution_state") != W03_FORMAL_EXECUTION_STATE
            or host.get("self_excluded") != 1):
        raise W03EvaluatorInfrastructureError("candidate host freeze 状态漂移")
    inventory = _inventory_state(candidate_root, host.get("artifact_inventory"))
    return contract, host, inventory


def _verify_family(
        family_root: Path,
        family_freeze_sha256: str,
        ) -> tuple[dict[str, Any], tuple[bytes, ...]]:
    """在 private decode 前回验 family freeze、文档 identity 与 commitment。"""
    freeze_path = family_root / W03_PRIVATE_FAMILY_FREEZE_NAME
    if _sha256(freeze_path) != family_freeze_sha256:
        raise W03EvaluatorInfrastructureError("private family freeze SHA 漂移")
    freeze = _read_canonical(freeze_path, label="private family freeze")
    required = {
        "artifact_kind", "candidate_contract_sha256",
        "candidate_host_freeze_sha256", "case_commitment",
        "cluster_commitment", "document_inventory", "failure_phase_registry",
        "family_commitment", "fault_registry", "formal_run_count",
        "format_version", "label_commitment", "payload_commitment",
        "self_excluded",
    }
    if (set(freeze) != required
            or freeze["artifact_kind"] != "PH2_W03_PRIVATE_FAMILY_FREEZE"
            or freeze["format_version"] != 1
            or freeze["formal_run_count"] != 0
            or freeze["self_excluded"] != 1
            or freeze["fault_registry"] != list(W03_EVALUATOR_PHASES)
            or freeze["failure_phase_registry"]
            != list(W03_EVALUATOR_FAILURE_PHASES)):
        raise W03EvaluatorInfrastructureError("private family freeze 字段/状态漂移")
    identities = _inventory_state(family_root, freeze["document_inventory"])
    if tuple(item[0] for item in identities) != _PRIVATE_DOCUMENT_NAMES:
        raise W03EvaluatorInfrastructureError("private family document 顺序漂移")
    documents = tuple((family_root / name).read_bytes()
                      for name in _PRIVATE_DOCUMENT_NAMES)
    if (hashlib.sha256(documents[2]).hexdigest() != freeze["case_commitment"]
            or hashlib.sha256(documents[3]).hexdigest() != freeze["label_commitment"]
            or hashlib.sha256(documents[4]).hexdigest() != freeze["cluster_commitment"]):
        raise W03EvaluatorInfrastructureError("private case/label/cluster commitment 漂移")
    payload_digest = hashlib.sha256()
    for document in documents:
        payload_digest.update(len(document).to_bytes(8, "big"))
        payload_digest.update(document)
    if payload_digest.hexdigest() != freeze["payload_commitment"]:
        raise W03EvaluatorInfrastructureError("private payload commitment 漂移")
    return freeze, documents


def _candidate_config(
        config: W03PrivateEvaluatorRuntimeConfig,
        contract: dict[str, Any],
        ) -> W03RuntimeConfig:
    """只从 frozen candidate request 恢复 dump clone 所需 runtime config。"""
    request = contract.get("candidate_request")
    if not isinstance(request, dict):
        raise W03EvaluatorInfrastructureError("candidate request freeze 缺失")
    candidate_root = Path(config.candidate_root).resolve()
    return W03RuntimeConfig(
        repository_root=Path(config.repository_root).resolve(),
        global_manifest_path=config.global_manifest_path,
        w02_artifacts_root=Path(config.w02_artifacts_root).resolve(),
        run_root=candidate_root / "runs",
        sqlite_path=candidate_root / "candidate.sqlite3",
        run_id=request["run_id"],
        parent_run_id=request["parent_run_id"],
        base_run_id=request["base_run_id"],
        base_fence_key=tuple(request["base_fence_key"]),
        worker_count=request["worker_count"],
        mode="resume",
        current_remote_commit_sha1=contract["remote_commit_sha1"],
    )


def _outcome_matches_host(
        outcome: W03RunOutcome,
        host: dict[str, Any],
        ) -> bool:
    """比较 freeze 中的八摘要、artifact count、dump 与 W-02 retention。"""
    evidence = host.get("host_evidence")
    if not isinstance(evidence, dict):
        return False
    expected_digests = {
        "artifact": outcome.artifact_digest,
        "candidate_history": outcome.candidate_history_digest,
        "cursor": outcome.cursor_digest,
        "generation": outcome.generation_digest,
        "logical": outcome.logical_state_digest,
        "projection": outcome.projection_digest,
        "retention": outcome.retention_digest,
    }
    return (
        evidence.get("host_digests") == expected_digests
        and evidence.get("artifact_counts")
        == [list(item) for item in outcome.artifact_counts]
        and evidence.get("w02_retention_passed") == 1
        and outcome.w02_retention_passed
        and outcome.w02_host_write_count == 0
        and outcome.new_learning_write_count == 0
        and outcome.dump_readback
    )


def _hit(config: W03PrivateEvaluatorRuntimeConfig, phase: str) -> None:
    """在 synthetic preflight 选择的唯一 phase 注入显式基础设施错误。"""
    if config.fault_phase is None:
        return
    if config.fault_phase not in W03_EVALUATOR_PHASES:
        raise W03EvaluatorInfrastructureError("未知 evaluator fault phase")
    if config.fault_phase == phase:
        raise W03EvaluatorInjectedFault(phase)


def _enter_phase(
        config: W03PrivateEvaluatorRuntimeConfig,
        phase_counts: dict[str, int],
        phase: str,
        ) -> None:
    """按冻结 registry 记录一次 phase 进入并执行 synthetic fault hook。"""
    phase_counts[phase] += 1
    _hit(config, phase)


def _write_exclusive(path: Path, payload: bytes, *, label: str) -> str:
    """排他写 publication 文件并返回 SHA。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as exc:
        raise W03EvaluatorInfrastructureError(f"{label} 不可覆盖") from exc
    return hashlib.sha256(payload).hexdigest()


def _publish_failure(
        family_root: Path,
        freeze: dict[str, Any],
        phase: str,
        *,
        host_writes: int,
        label_writes: int,
        ) -> tuple[Path, str, dict[str, object]]:
    """基础设施异常只发布枚举 phase 和 commitment，不投影异常 message。"""
    aggregate = public_safe_w03_aggregate(
        (),
        family_commitment=freeze["family_commitment"],
        payload_commitment=freeze["payload_commitment"],
        case_commitment=freeze["case_commitment"],
        label_commitment=freeze["label_commitment"],
        cluster_commitment=freeze["cluster_commitment"],
        failure_phase=phase,
        formal_run_count=1,
        host_writes=host_writes,
        label_writes=label_writes,
    )
    path = family_root / "publication" / W03_PRIVATE_AGGREGATE_NAME
    encoded = canonical_json_bytes(aggregate)
    return path, _write_exclusive(path, encoded, label="private NE aggregate"), aggregate


def _restore_and_evaluate(
        config: W03PrivateEvaluatorRuntimeConfig,
        candidate_config: W03RuntimeConfig,
        host_copy: Path,
        private_payload: W03PrivatePayload,
        phase_counts: dict[str, int],
        ) -> tuple[
            tuple[W03PrivateDimensionResult, ...],
            tuple[tuple[str, tuple[str, ...]], ...],
            tuple[str, ...],
            tuple[bool, ...],
            int,
            ]:
    """在 host-copy 上零新写恢复 runtime，并执行 baseline、四消融和 generation 消融。"""
    before = _sha256(host_copy)
    backend = None
    current_phase = "BASELINE"
    try:
        backend = SQLiteBackend(str(host_copy))
        repository = Path(config.repository_root).resolve()
        continuity = verify_formal_w02_continuity(
            repository, Path(config.w02_artifacts_root).resolve())
        context = open_w03_frozen_context(
            repository,
            config.global_manifest_path,
            current_remote_commit_sha1=candidate_config.current_remote_commit_sha1,
            w02_continuity=continuity,
            publication_baseline=formal_w03_publication_baseline(),
            backend_profile_key=backend.storage_capabilities().stable_key(),
        )
        store = W03ArtifactStore(backend)
        training_payload = restore_training_payload(store)
        learning = run_w03_learning(
            backend, training_payload, context, restore=True)
        if learning.new_learning_write_count != 0:
            raise W03EvaluatorInfrastructureError("evaluator host-copy restore 产生学习写")
        persisted_outcomes = tuple(store.payloads(ARTIFACT_GENERATION_OUTCOME))
        _enter_phase(config, phase_counts, current_phase)
        baseline = evaluate_w03_learning_runtime(
            learning.understanding,
            private_payload.cases,
            persisted_generation_outcomes=persisted_outcomes,
        )
        ablations = []
        gate_passes = []
        for phase, ablation_key, dimension_key in zip(
                W03_EVALUATOR_PHASES[5:9],
                W03_ABLATION_KEYS,
                W03_DIMENSION_KEYS,
                strict=True):
            current_phase = phase
            _enter_phase(config, phase_counts, current_phase)
            values = evaluate_w03_learning_runtime(
                learning.understanding,
                private_payload.cases,
                persisted_generation_outcomes=persisted_outcomes,
                ablation=W03EvaluatorAblation(ablation_key),
            )
            statuses = tuple(item.status for item in values)
            expected = tuple(
                "FAIL" if item.dimension_key == dimension_key else "PASS"
                for item in values)
            gate_passes.append(
                statuses == expected and not any(item.ne_count for item in values))
            ablations.append((ablation_key, statuses))
        current_phase = "GENERATION"
        _enter_phase(config, phase_counts, current_phase)
        generation_disabled = evaluate_w03_learning_runtime(
            learning.understanding,
            private_payload.cases,
            persisted_generation_outcomes=persisted_outcomes,
            sense_consumer_connected=False,
        )
        generation_statuses = tuple(item.status for item in generation_disabled)
        gate_passes.append(
            generation_statuses[:4] == ("PASS",) * 4
            and generation_statuses[4] == "FAIL"
            and not any(item.ne_count for item in generation_disabled))
        return (
            baseline,
            tuple(ablations),
            generation_statuses,
            tuple(gate_passes),
            learning.new_learning_write_count,
        )
    except (W03EvaluatorInjectedFault, _W03EvaluatorPhaseError):
        raise
    except Exception as exc:
        raise _W03EvaluatorPhaseError(current_phase) from exc
    finally:
        if backend is not None:
            try:
                backend.close()
            except Exception as exc:
                raise _W03EvaluatorPhaseError("INTEGRITY") from exc
        try:
            unchanged = _sha256(host_copy) == before
        except Exception as exc:
            raise _W03EvaluatorPhaseError("INTEGRITY") from exc
        if not unchanged:
            raise _W03EvaluatorPhaseError("INTEGRITY")


def _apply_capability_gates(
        baseline: tuple[W03PrivateDimensionResult, ...],
        gate_passes: tuple[bool, ...],
        ) -> tuple[W03PrivateDimensionResult, ...]:
    """把各维 baseline 与对应消融合取，能力失败保持 FAIL 而非 NE。"""
    if len(baseline) != len(gate_passes):
        raise W03EvaluatorInfrastructureError("evaluator capability gate 数量漂移")
    results = []
    for item, gate_passed in zip(baseline, gate_passes, strict=True):
        if type(gate_passed) is not bool:
            raise W03EvaluatorInfrastructureError("evaluator capability gate 非 bool")
        if item.status == "PASS" and not gate_passed:
            item = replace(
                item,
                status="FAIL",
                passed=0,
                fail_count=1,
                evidence_commitment=evidence_commitment({
                    "ablation_gate_passed": 0,
                    "baseline_evidence_commitment": item.evidence_commitment,
                    "dimension_key": item.dimension_key,
                }),
            )
        results.append(item)
    return tuple(results)


def _validate_owner_roots(
        repository: Path,
        w02_root: Path,
        candidate_root: Path,
        family_root: Path,
        execution_root: Path,
        ) -> None:
    """要求公开、W-02、candidate、family 隔离，execution 仅位于 family 下。"""
    owners = (repository, w02_root, candidate_root, family_root)
    for index, left in enumerate(owners):
        for right in owners[index + 1:]:
            if (left == right or left.is_relative_to(right)
                    or right.is_relative_to(left)):
                raise W03EvaluatorInfrastructureError("evaluator owner 物理 root 未隔离")
    if execution_root == family_root or not execution_root.is_relative_to(family_root):
        raise W03EvaluatorInfrastructureError("evaluator execution root 未隔离")


def run_w03_private_evaluation_once(
        config: W03PrivateEvaluatorRuntimeConfig,
        *,
        family_freeze_sha256: str,
        ) -> W03PrivateEvaluatorRunResult:
    """消费唯一 private guard，完成 fresh clone/host-copy/五维评测和安全发布。"""
    if not isinstance(config, W03PrivateEvaluatorRuntimeConfig):
        raise TypeError("W-03 evaluator config 类型非法")
    repository = Path(config.repository_root).resolve()
    w02_root = Path(config.w02_artifacts_root).resolve()
    candidate_root = Path(config.candidate_root).resolve()
    family_root = Path(config.family_root).resolve()
    execution_root = Path(config.execution_root).resolve()
    _validate_owner_roots(
        repository, w02_root, candidate_root, family_root, execution_root)
    freeze, documents = _verify_family(
        family_root, family_freeze_sha256)
    contract, host, candidate_before = _verify_candidate(
        candidate_root,
        contract_sha256=freeze["candidate_contract_sha256"],
        host_sha256=freeze["candidate_host_freeze_sha256"],
    )
    label_before = hashlib.sha256(documents[3]).hexdigest()
    consume_w03_private_first_run_guard(
        family_root, family_freeze_sha256=family_freeze_sha256)
    phase_counts = {phase: 0 for phase in W03_EVALUATOR_PHASES}
    current_phase = "PAYLOAD_DECODE"
    clone_outcome = None
    host_copy_sha = "0" * 64
    try:
        _enter_phase(config, phase_counts, current_phase)
        private_payload = decode_w03_private_documents(*documents)
        if (private_payload.candidate_contract_sha256
                != freeze["candidate_contract_sha256"]
                or private_payload.candidate_host_freeze_sha256
                != freeze["candidate_host_freeze_sha256"]):
            raise W03EvaluatorInfrastructureError(
                "private family candidate binding 漂移")
        current_phase = "CLONE_LOAD"
        _enter_phase(config, phase_counts, current_phase)
        execution_root.mkdir(parents=True, exist_ok=False)
        clone_path = execution_root / "fresh_clone.sqlite3"
        candidate_config = _candidate_config(config, contract)
        clone_outcome = formalize_w03_candidate_outcome(
            load_w03_candidate_dump(
                candidate_config,
                target_sqlite_path=clone_path,
            ))
        current_phase = "HOST_COPY"
        _enter_phase(config, phase_counts, current_phase)
        host_copy = execution_root / "host_copy.sqlite3"
        shutil.copyfile(clone_path, host_copy)
        clone_sha = _sha256(clone_path)
        host_copy_sha = _sha256(host_copy)
        if clone_sha != host_copy_sha:
            raise W03EvaluatorInfrastructureError("fresh clone/host-copy bytes 漂移")
        current_phase = "CLONE_COMPARE"
        _enter_phase(config, phase_counts, current_phase)
        if not _outcome_matches_host(clone_outcome, host):
            raise W03EvaluatorInfrastructureError("fresh clone 与 candidate freeze 漂移")
        current_phase = "BASELINE"
        baseline, ablations, generation_statuses, gate_passes, restore_writes = (
            _restore_and_evaluate(
                config,
                candidate_config,
                host_copy,
                private_payload,
                phase_counts,
            )
        )
        capability_results = _apply_capability_gates(baseline, gate_passes)
        current_phase = "INTEGRITY"
        _enter_phase(config, phase_counts, current_phase)
        _, _, candidate_after = _verify_candidate(
            candidate_root,
            contract_sha256=freeze["candidate_contract_sha256"],
            host_sha256=freeze["candidate_host_freeze_sha256"],
        )
        label_after = _sha256(family_root / "private_labels.json")
        host_writes = int(candidate_after != candidate_before)
        label_writes = int(label_after != label_before)
        if host_writes or label_writes or restore_writes != 0:
            raise W03EvaluatorInfrastructureError("evaluator host/label/restore 零写失败")
        aggregate = public_safe_w03_aggregate(
            capability_results,
            family_commitment=freeze["family_commitment"],
            payload_commitment=freeze["payload_commitment"],
            case_commitment=freeze["case_commitment"],
            label_commitment=freeze["label_commitment"],
            cluster_commitment=freeze["cluster_commitment"],
            failure_phase="NONE",
            formal_run_count=1,
            host_writes=host_writes,
            label_writes=label_writes,
        )
        aggregate["ablation_results"] = [
            {"ablation_key": key, "dimension_statuses": list(statuses)}
            for key, statuses in ablations]
        aggregate["generation_ablation_statuses"] = list(generation_statuses)
        aggregate["infrastructure"] = {
            "candidate_inventory_match": 1,
            "clone_dump_readback": int(clone_outcome.dump_readback),
            "clone_host_copy_match": 1,
            "host_copy_unchanged": 1,
            "label_writes": label_writes,
            "restore_learning_writes": restore_writes,
        }
        current_phase = "REPORT_SAFETY"
        _enter_phase(config, phase_counts, current_phase)
        encoded = canonical_json_bytes(aggregate)
        forbidden = (
            *(item.case_key.encode("utf-8") for item in private_payload.cases),
            *(item.label_key.encode("utf-8") for item in private_payload.labels),
            b"surface", b"expected", b"private_path", b"exception", b"message",
        )
        if any(item in encoded for item in forbidden):
            raise W03EvaluatorInfrastructureError("安全 aggregate 泄漏 private 字段")
        aggregate_path = family_root / "publication" / W03_PRIVATE_AGGREGATE_NAME
        aggregate_sha = _write_exclusive(
            aggregate_path, encoded, label="private aggregate")
        recommendation_path = None
        recommendation_sha = None
        if aggregate["status"] == "PASS":
            recommendation = canonical_json_bytes({
                "aggregate_sha256": aggregate_sha,
                "artifact_kind": "PH2_W03_RUNTIME_RECEIPT_RECOMMENDATION",
                "candidate_host_freeze_sha256": freeze[
                    "candidate_host_freeze_sha256"],
                "family_commitment": freeze["family_commitment"],
                "formal_run_count": 1,
                "format_version": 1,
                "recommend_runtime_receipt": 1,
            })
            recommendation_path = (
                family_root / "publication" / W03_PRIVATE_RECOMMENDATION_NAME)
            recommendation_sha = _write_exclusive(
                recommendation_path,
                recommendation,
                label="runtime receipt recommendation",
            )
        return W03PrivateEvaluatorRunResult(
            aggregate_path,
            aggregate_sha,
            recommendation_path,
            recommendation_sha,
            aggregate,
            tuple(phase_counts.items()),
            clone_outcome,
            host_copy_sha,
            host_writes,
            label_writes,
        )
    except Exception as exc:
        if not (family_root / "publication" / W03_PRIVATE_AGGREGATE_NAME).exists():
            try:
                _, _, candidate_after = _verify_candidate(
                    candidate_root,
                    contract_sha256=freeze["candidate_contract_sha256"],
                    host_sha256=freeze["candidate_host_freeze_sha256"],
                )
                host_writes = int(candidate_after != candidate_before)
            except Exception:
                host_writes = 1
            try:
                label_writes = int(
                    _sha256(family_root / "private_labels.json") != label_before)
            except Exception:
                label_writes = 1
            failure_phase = getattr(exc, "failure_phase", current_phase)
            _publish_failure(
                family_root,
                freeze,
                failure_phase,
                host_writes=host_writes,
                label_writes=label_writes,
            )
        raise


__all__ = [
    "W03EvaluatorInfrastructureError",
    "W03EvaluatorInjectedFault",
    "W03PrivateEvaluatorRunResult",
    "W03PrivateEvaluatorRuntimeConfig",
    "run_w03_private_evaluation_once",
]
