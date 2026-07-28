"""D-02 外部来源 pack 的 Dict/SQLite 批次、恢复和精确隔离运行器。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from pure_integer_ai.experiments.evaluation_isolation import clone_backend
from pure_integer_ai.experiments.ph2_dataset_contract import (
    JSONL_RECORD_KINDS,
    CanonicalJsonObject,
    ObservationRecord,
    canonical_json_bytes,
    canonical_json_line,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_source_pack_compiler import (
    SourcePackBuild,
    SourcePackCompilerError,
    compile_or_resume_source_pack,
    read_source_pack_view,
)
from pure_integer_ai.experiments.ph2_source_pack_contract import (
    SourceObservationSeed,
    SourcePackSpec,
)
from pure_integer_ai.storage.backend import (
    TYPE_INT,
    TYPE_TEXT,
    StorageBackend,
    register_extension_table,
)
from pure_integer_ai.storage.telemetry import collect_backend_telemetry


SOURCE_PACK_STATE_TABLE = "ph2_source_pack_state"
SOURCE_PACK_RESULT_TABLE = "ph2_source_pack_result"
SOURCE_PACK_CLONE_AUDIT_TABLE = "ph2_source_pack_clone_audit"
SOURCE_PACK_TABLES = frozenset({
    SOURCE_PACK_STATE_TABLE,
    SOURCE_PACK_RESULT_TABLE,
    SOURCE_PACK_CLONE_AUDIT_TABLE,
})
SOURCE_PACK_RUN_ID = 1
SOURCE_PACK_WORKER_COUNTS = (1, 2, 4)
SOURCE_PACK_FAULT_CODES = (
    "BAD_COMBINATION",
    "BAD_LICENSE",
    "BAD_RECORD",
    "BAD_SOURCE",
)


class SourcePackRuntimeError(RuntimeError):
    """来源 pack 批次合同、cursor、恢复或零写审计失败。"""


@dataclass(frozen=True)
class SourcePackTask:
    """一个稳定 pack id 对应的 spec 和已核准 seed。"""

    pack_id: int
    spec: SourcePackSpec
    seeds: tuple[SourceObservationSeed, ...]

    def __post_init__(self) -> None:
        if type(self.pack_id) is not int or self.pack_id <= 0:
            raise SourcePackRuntimeError("source pack_id 必须是正严格整数")
        if not isinstance(self.spec, SourcePackSpec):
            raise SourcePackRuntimeError("source pack task spec 类型错误")
        if not isinstance(self.seeds, tuple) or not self.seeds:
            raise SourcePackRuntimeError("source pack task seeds 不能为空")
        if any(not isinstance(item, SourceObservationSeed) for item in self.seeds):
            raise SourcePackRuntimeError("source pack task seed 类型错误")

    def to_contract_dict(self) -> dict[str, Any]:
        """导出不含 raw 正文、backend、worker 和路径根的任务合同。"""
        return {
            "pack_id": self.pack_id,
            "seeds": [item.to_contract_dict() for item in self.seeds],
            "spec": self.spec.to_contract_dict(),
        }


@dataclass(frozen=True)
class SourcePackBatchRunResult:
    """返回规范报告及本次执行维度，不把它们混入规范 hash。"""

    report: CanonicalJsonObject
    normative_sha256: str
    contract_sha256: str
    backend_kind: str
    worker_count: int
    resumed_pack_count: int
    published_pack_count: int


def _sha256_value(value: Any) -> str:
    """返回规范 JSON 值 SHA-256。"""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _register_tables(backend: StorageBackend) -> None:
    """幂等注册三个非 Core、非训练的来源 pack 扩展表。"""
    register_extension_table(
        backend,
        SOURCE_PACK_STATE_TABLE,
        [
            ("run_id", TYPE_INT),
            ("contract_sha256", TYPE_TEXT),
            ("release_binding_sha256", TYPE_TEXT),
            ("completed_count", TYPE_INT),
            ("complete", TYPE_INT),
            ("report_json", TYPE_TEXT),
        ],
        indexes=[("run_id",)],
        recovery_key=("run_id",),
    )
    register_extension_table(
        backend,
        SOURCE_PACK_RESULT_TABLE,
        [
            ("run_id", TYPE_INT),
            ("pack_id", TYPE_INT),
            ("payload_json", TYPE_TEXT),
        ],
        indexes=[("run_id", "pack_id")],
        recovery_key=("run_id", "pack_id"),
    )
    register_extension_table(
        backend,
        SOURCE_PACK_CLONE_AUDIT_TABLE,
        [
            ("run_id", TYPE_INT),
            ("audit_kind", TYPE_TEXT),
            ("record_count", TYPE_INT),
        ],
        indexes=[("run_id", "audit_kind")],
        recovery_key=("run_id", "audit_kind"),
    )


def _state(backend: StorageBackend) -> dict[str, Any] | None:
    """读取唯一批次 state 并核对 cursor/report 形状。"""
    rows = backend.select(
        SOURCE_PACK_STATE_TABLE, {"run_id": SOURCE_PACK_RUN_ID})
    if not rows:
        return None
    if len(rows) != 1:
        raise SourcePackRuntimeError("source pack state 重复")
    row = rows[0]
    completed = row.get("completed_count")
    complete = row.get("complete")
    report_json = row.get("report_json")
    if type(completed) is not int or completed < 0:
        raise SourcePackRuntimeError("source pack cursor 非法")
    if complete not in {0, 1} or not isinstance(report_json, str):
        raise SourcePackRuntimeError("source pack state 状态非法")
    if bool(complete) != bool(report_json):
        raise SourcePackRuntimeError("source pack report 与完成位不一致")
    return row


def _initialize(
        backend: StorageBackend,
        *,
        contract_sha256: str,
        release_binding_sha256: str) -> None:
    """初始化或核对不可漂移的合同和 release 绑定。"""
    row = _state(backend)
    if row is None:
        backend.insert(SOURCE_PACK_STATE_TABLE, {
            "run_id": SOURCE_PACK_RUN_ID,
            "contract_sha256": contract_sha256,
            "release_binding_sha256": release_binding_sha256,
            "completed_count": 0,
            "complete": 0,
            "report_json": "",
        })
        backend.commit()
        return
    if (row.get("contract_sha256") != contract_sha256
            or row.get("release_binding_sha256") != release_binding_sha256):
        raise SourcePackRuntimeError("source pack resume 合同或 release_root 漂移")


def _decode(text: Any, *, where: str) -> dict[str, Any]:
    """恢复 backend TEXT 中的规范 JSON object。"""
    if not isinstance(text, str) or not text:
        raise SourcePackRuntimeError(f"{where} 不能为空")
    try:
        value = parse_canonical_json_bytes(
            text.encode("utf-8"), require_object=True)
    except Exception as error:
        raise SourcePackRuntimeError(f"{where} 非规范 JSON") from error
    assert isinstance(value, dict)
    return value


def _load_results(backend: StorageBackend) -> dict[int, dict[str, Any]]:
    """恢复已提交结果并要求 cursor 与结果数一致。"""
    rows = backend.select(
        SOURCE_PACK_RESULT_TABLE,
        {"run_id": SOURCE_PACK_RUN_ID},
        order_by="pack_id",
    )
    results: dict[int, dict[str, Any]] = {}
    for row in rows:
        pack_id = row.get("pack_id")
        if type(pack_id) is not int or pack_id <= 0 or pack_id in results:
            raise SourcePackRuntimeError("source pack result id 非法或重复")
        results[pack_id] = _decode(
            row.get("payload_json"), where=f"source pack result[{pack_id}]")
    state = _state(backend)
    if state is None or state.get("completed_count") != len(results):
        raise SourcePackRuntimeError("source pack cursor 与结果数不一致")
    return results


def _store_result(
        backend: StorageBackend,
        pack_id: int,
        payload: dict[str, Any]) -> None:
    """不可覆盖地提交单 pack 结果并推进 cursor。"""
    if backend.count(SOURCE_PACK_RESULT_TABLE, {
            "run_id": SOURCE_PACK_RUN_ID, "pack_id": pack_id}):
        raise SourcePackRuntimeError("source pack result 禁止覆盖")
    backend.insert(SOURCE_PACK_RESULT_TABLE, {
        "run_id": SOURCE_PACK_RUN_ID,
        "pack_id": pack_id,
        "payload_json": canonical_json_bytes(payload).decode("utf-8"),
    })
    state = _state(backend)
    if state is None or state.get("complete") != 0:
        raise SourcePackRuntimeError("source pack state 不可推进")
    updated = backend.update(
        SOURCE_PACK_STATE_TABLE,
        {"run_id": SOURCE_PACK_RUN_ID},
        {"completed_count": state["completed_count"] + 1},
    )
    if updated != 1:
        raise SourcePackRuntimeError("source pack cursor 推进失败")
    backend.commit()


def _finish(backend: StorageBackend, report: dict[str, Any], count: int) -> None:
    """全部结果提交后冻结规范 report；重复完成必须逐字节相等。"""
    state = _state(backend)
    if state is None or state.get("completed_count") != count:
        raise SourcePackRuntimeError("source pack 尚未全部提交")
    if state.get("complete") == 1:
        if _decode(state["report_json"], where="stored source pack report") != report:
            raise SourcePackRuntimeError("source pack 已完成 report 漂移")
        return
    updated = backend.update(
        SOURCE_PACK_STATE_TABLE,
        {"run_id": SOURCE_PACK_RUN_ID},
        {
            "complete": 1,
            "report_json": canonical_json_bytes(report).decode("utf-8"),
        },
    )
    if updated != 1:
        raise SourcePackRuntimeError("source pack report 发布失败")
    backend.commit()


def _normalize_tasks(
        tasks: tuple[SourcePackTask, ...]) -> tuple[SourcePackTask, ...]:
    """要求 task id、pack 名和 source/license 身份唯一且排序稳定。"""
    if not isinstance(tasks, tuple) or not tasks:
        raise SourcePackRuntimeError("source pack tasks 不能为空")
    if any(not isinstance(item, SourcePackTask) for item in tasks):
        raise SourcePackRuntimeError("source pack task 类型错误")
    ordered = tuple(sorted(tasks, key=lambda item: item.pack_id))
    ids = [item.pack_id for item in ordered]
    names = [item.spec.pack_name for item in ordered]
    if len(ids) != len(set(ids)) or len(names) != len(set(names)):
        raise SourcePackRuntimeError("source pack task id/name 重复")
    return ordered


def _normalize_faults(
        faults: Mapping[int, str] | None,
        tasks: tuple[SourcePackTask, ...]) -> dict[int, str]:
    """只允许一次精确来源/许可/记录/组合失败注入。"""
    result = dict(faults or {})
    if len(result) > 1:
        raise SourcePackRuntimeError("source pack 一次只允许一个 fault")
    ids = {item.pack_id for item in tasks}
    for pack_id, code in result.items():
        if type(pack_id) is not int or pack_id not in ids:
            raise SourcePackRuntimeError("source pack fault id 不在 tasks")
        if code not in SOURCE_PACK_FAULT_CODES:
            raise SourcePackRuntimeError("source pack fault code 未注册")
    return result


def _execution_order(
        tasks: tuple[SourcePackTask, ...], worker_count: int,
        ) -> tuple[SourcePackTask, ...]:
    """worker 数改变 shard 调度，但不改变最终规范排序。"""
    shards = tuple(
        tuple(item for item in tasks
              if (item.pack_id - 1) % worker_count == shard)
        for shard in range(worker_count)
    )
    return tuple(item for shard in shards for item in shard)


def _faulted_task(task: SourcePackTask, code: str) -> SourcePackTask:
    """构造单点坏许可、来源、私有字段或组合跨 split 输入。"""
    if code == "BAD_LICENSE":
        return replace(task, spec=replace(task.spec, license_id="UNKNOWN"))
    first = task.seeds[0]
    if code == "BAD_SOURCE":
        seeds = (replace(first, source_identity=""),) + task.seeds[1:]
    elif code == "BAD_RECORD":
        raw = first.raw_observation.to_value()
        raw["expected"] = "private"
        seeds = (replace(
            first,
            raw_observation=CanonicalJsonObject.from_value(raw)),
            ) + task.seeds[1:]
    elif code == "BAD_COMBINATION":
        alternative = next((
            item for item in task.seeds[1:] if item.split != first.split), None)
        if alternative is None:
            raise SourcePackRuntimeError(
                "BAD_COMBINATION 需要至少两个不同 split")
        seeds = tuple(
            replace(item, combination_parts=first.combination_parts)
            if item.seed_id == alternative.seed_id else item
            for item in task.seeds
        )
    else:
        raise SourcePackRuntimeError("source pack fault code 非法")
    return replace(task, seeds=seeds)


def _pass_result(task: SourcePackTask, build: SourcePackBuild) -> dict[str, Any]:
    """形成不含执行顺序、backend、worker 和宿主路径的 pack 证据。"""
    bundle = build.bundle
    student_digests: list[str] = []
    for split in bundle.manifest.splits:
        student = read_source_pack_view(
            build.pack_root, reader_kind="student", split=split)
        if any(not isinstance(item, ObservationRecord) for item in student):
            raise SourcePackRuntimeError("student view 读到非 Observation")
        student_digests.append(_sha256_value(
            [item.to_dict() for item in student]))
        read_source_pack_view(
            build.pack_root, reader_kind="teacher", split=split)
        read_source_pack_view(
            build.pack_root, reader_kind="evaluator", split=split)
    read_source_pack_view(build.pack_root, reader_kind="source_audit")
    combination = bundle.combination_audit.to_value()
    return {
        "anomaly_code": None,
        "combination_cluster_count": combination["combination_cluster_count"],
        "contract_sha256": build.contract_sha256,
        "files": [item.to_dict() for item in bundle.manifest.files],
        "license_partition": bundle.manifest.license_partition,
        "manifest_sha256": bundle.manifest.sha256(),
        "owner_read_isolation": 1,
        "pack_id": task.pack_id,
        "pack_name": task.spec.pack_name,
        "record_counts": {
            "evaluator_label": len(bundle.evaluators),
            "observation": len(bundle.observations),
            "source_ref": len(bundle.sources),
            "teacher_evidence": len(bundle.teachers),
        },
        "source_cluster_count": bundle.validation.source_cluster_count,
        "source_key": bundle.manifest.source_key,
        "splits": list(bundle.manifest.splits),
        "status": "PASS",
        "student_view_sha256": _sha256_value(student_digests),
    }


def _anomaly_result(
        task: SourcePackTask,
        code: str,
        error: Exception) -> dict[str, Any]:
    """只记录稳定失败类型和 code，不泄露绝对路径或正文。"""
    error_type = type(error).__name__
    return {
        "anomaly_code": code,
        "error_identity_sha256": _sha256_value({
            "anomaly_code": code,
            "error_type": error_type,
            "pack_id": task.pack_id,
        }),
        "error_type": error_type,
        "pack_id": task.pack_id,
        "pack_name": task.spec.pack_name,
        "status": "ANOMALY",
    }


def _execute(
        task: SourcePackTask,
        release_root: Path,
        fault_code: str | None) -> tuple[dict[str, Any], bool]:
    """执行/恢复单 pack；正常错误 fail-closed，注入错误精确隔离。"""
    if fault_code is None:
        build = compile_or_resume_source_pack(
            task.spec, task.seeds, release_root)
        return _pass_result(task, build), build.published
    try:
        faulted = _faulted_task(task, fault_code)
        build = compile_or_resume_source_pack(
            faulted.spec, faulted.seeds, release_root)
    except Exception as error:
        return _anomaly_result(task, fault_code, error), False
    raise SourcePackRuntimeError(
        f"fault {fault_code} 被 source pack compiler 静默接受: "
        f"{build.pack_root.name}")


def _clone_audit(
        backend: StorageBackend,
        results: tuple[dict[str, Any], ...]) -> None:
    """证明来源 pack 只读审计在 clone 写入时不污染宿主。"""
    baseline = backend.recovery_state_snapshot()
    cloned = clone_backend(backend)
    try:
        cloned.insert(SOURCE_PACK_CLONE_AUDIT_TABLE, {
            "run_id": SOURCE_PACK_RUN_ID,
            "audit_kind": "V06_SOURCE_PACK_READ_ONLY",
            "record_count": len(results),
        })
        cloned.commit()
        if cloned.count(SOURCE_PACK_CLONE_AUDIT_TABLE, {
                "run_id": SOURCE_PACK_RUN_ID}) != 1:
            raise SourcePackRuntimeError("source pack clone audit 未写入 clone")
    finally:
        cloned.close()
    if backend.recovery_state_snapshot() != baseline:
        raise SourcePackRuntimeError("source pack clone 写回宿主")
    if backend.count(SOURCE_PACK_CLONE_AUDIT_TABLE, {
            "run_id": SOURCE_PACK_RUN_ID}) != 0:
        raise SourcePackRuntimeError("source pack clone 污染宿主")


def run_source_pack_batch(
        tasks: tuple[SourcePackTask, ...],
        release_root: str | Path,
        backend: StorageBackend,
        *,
        worker_count: int = 1,
        faults: Mapping[int, str] | None = None,
        ) -> SourcePackBatchRunResult:
    """运行或恢复统一来源 pack；不调用 teacher、训练或学习状态写入。"""
    ordered = _normalize_tasks(tasks)
    if worker_count not in SOURCE_PACK_WORKER_COUNTS:
        raise SourcePackRuntimeError("source pack worker_count 只支持 1/2/4")
    normalized_faults = _normalize_faults(faults, ordered)
    release = Path(release_root).resolve()
    release.mkdir(parents=True, exist_ok=True)
    contract_payload = {
        "faults": [
            {"code": normalized_faults[key], "pack_id": key}
            for key in sorted(normalized_faults)
        ],
        "source_pack_runtime_version": 1,
        "tasks": [item.to_contract_dict() for item in ordered],
    }
    contract_sha256 = _sha256_value(contract_payload)
    release_binding_sha256 = _sha256_value({
        "contract_sha256": contract_sha256,
        "release_root": str(release),
    })
    published_count = 0
    with collect_backend_telemetry() as telemetry:
        _register_tables(backend)
        _initialize(
            backend,
            contract_sha256=contract_sha256,
            release_binding_sha256=release_binding_sha256,
        )
        initial = _load_results(backend)
        expected_ids = {item.pack_id for item in ordered}
        if set(initial) - expected_ids:
            raise SourcePackRuntimeError("source pack backend 含任务外结果")
        resumed_count = len(initial)
        for task in _execution_order(ordered, worker_count):
            if task.pack_id in initial:
                continue
            result, published = _execute(
                task, release, normalized_faults.get(task.pack_id))
            _store_result(backend, task.pack_id, result)
            initial[task.pack_id] = result
            published_count += 1 if published else 0
        results = _load_results(backend)
        if set(results) != expected_ids:
            raise SourcePackRuntimeError("source pack 未形成完整 result 集")
        result_rows = tuple(results[item.pack_id] for item in ordered)
        _clone_audit(backend, result_rows)
        operations = telemetry.operation_snapshot()
        non_source_writes = sum(
            rows
            for (operation, table), (_, rows, _) in operations.items()
            if operation in {"insert", "update", "delete"}
            and table not in SOURCE_PACK_TABLES
        )
        if non_source_writes != 0:
            raise SourcePackRuntimeError("source pack 写入了非自身 backend 表")
        pass_rows = [item for item in result_rows if item["status"] == "PASS"]
        anomalies = [item for item in result_rows if item["status"] == "ANOMALY"]
        if not normalized_faults and len(pass_rows) != len(ordered):
            raise SourcePackRuntimeError("正常 source pack 批次存在异常")
        if normalized_faults and len(anomalies) != 1:
            raise SourcePackRuntimeError("source pack fault 未精确隔离为一个 anomaly")
        report = {
            "anomalies": anomalies,
            "anomaly_count": len(anomalies),
            "artifact_kind": "PH2_D02_SOURCE_PACK_BATCH",
            "combination_cluster_count": sum(
                item["combination_cluster_count"] for item in pass_rows),
            "companion_enabled": 0,
            "contract_sha256": contract_sha256,
            "execution_dimensions_excluded_from_normative_hash": [
                "backend_kind", "release_root", "resume_history", "worker_count"],
            "formal_training_started": 0,
            "mastered": 0,
            "memory_enabled": 0,
            "non_source_backend_write_count": non_source_writes,
            "owner_read_isolation": 1,
            "packs": list(result_rows),
            "source_cluster_count": sum(
                item["source_cluster_count"] for item in pass_rows),
            "source_pack_runtime_version": 1,
            "successful_pack_count": len(pass_rows),
            "supported_backends": ["DictBackend", "SQLiteBackend"],
            "supported_worker_counts": list(SOURCE_PACK_WORKER_COUNTS),
            "teacher_call_count": 0,
            "training_state_write_count": 0,
            "v06_clone_host_write_count": 0,
            "v06_clone_training_state_write_count": 0,
        }
        _finish(backend, report, len(ordered))
    payload = canonical_json_line(report)
    return SourcePackBatchRunResult(
        CanonicalJsonObject.from_value(report),
        hashlib.sha256(payload).hexdigest(),
        contract_sha256,
        type(backend).__name__,
        worker_count,
        resumed_count,
        published_count,
    )


__all__ = [
    "SOURCE_PACK_CLONE_AUDIT_TABLE",
    "SOURCE_PACK_FAULT_CODES",
    "SOURCE_PACK_RESULT_TABLE",
    "SOURCE_PACK_STATE_TABLE",
    "SOURCE_PACK_TABLES",
    "SOURCE_PACK_WORKER_COUNTS",
    "SourcePackBatchRunResult",
    "SourcePackRuntimeError",
    "SourcePackTask",
    "run_source_pack_batch",
]
