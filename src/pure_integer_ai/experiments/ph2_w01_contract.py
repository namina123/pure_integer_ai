"""正式中文 PH2 W-01 阶段 0 的冻结输入、入口和状态语义合同。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from pure_integer_ai.experiments.ph2_d03_contract_core import D03ContractError
from pure_integer_ai.experiments.ph2_d03_publication import (
    D03PublicationReceipt,
    read_d03_publication_receipt,
)
from pure_integer_ai.experiments.ph2_d03_release_catalog import (
    FORMAL_GLOBAL_MANIFEST_PATH,
    FORMAL_RECEIPT_PATH,
)
from pure_integer_ai.experiments.ph2_d03_release_reader import D03ReleaseReader
from pure_integer_ai.experiments.ph2_dataset_core import canonical_json_bytes


D03_GLOBAL_MANIFEST_PATH = FORMAL_GLOBAL_MANIFEST_PATH
W01_FORMAT_VERSION = 1
W01_STAGE_KEY = "W-01"
W02_STAGE_KEY = "W-02"
W01_OWNER_KEY = "PH2_W01_TRANSACTION_OWNER"
W01_RUNNER_KEY = "PH2_LANGUAGE_STAGE0"
W01_PROTOCOL_STATUS = "W01_PROTOCOL_VERIFIED"
W01_CURSOR_VERSION = "PH2-W01-CURSOR-V1"
W01_PROTOCOL_INPUT_COUNT = 16
W01_ALLOWED_MODES = ("fresh", "restart", "resume")


class W01ContractError(RuntimeError):
    """W-01 发布依赖、入口身份、可见性或状态语义不满足。"""


def _digest_bytes(payload: bytes) -> tuple[int, ...]:
    """把规范字节映射为开放纯整数身份键。"""
    return tuple(hashlib.sha256(payload).digest())


def _digest_value(value: Any) -> tuple[int, ...]:
    """把无浮点结构对象映射为确定整数身份键。"""
    return _digest_bytes(canonical_json_bytes(value))


def _strict_key(value: object, *, label: str) -> tuple[int, ...]:
    """核验开放身份使用非空严格整数 tuple。"""
    if (not isinstance(value, tuple) or not value
            or any(type(item) is not int for item in value)):
        raise W01ContractError(f"{label} 必须是非空严格整数 tuple")
    return value


def _overlay_path(primary: Path, dependency: Path, relative: str) -> Path:
    """在候选/依赖根内解析只读文件并拒绝路径逃逸。"""
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts:
        raise W01ContractError("D-03 overlay 路径不规范")
    for root in (primary, dependency):
        target = (root / Path(*pure.parts)).resolve()
        if target.is_relative_to(root) and target.is_file():
            return target
    raise W01ContractError(f"D-03 文件缺失: {relative}")


@dataclass(frozen=True)
class W01ProtocolInput:
    """一个只由 D-03 协议元数据派生的冻结逻辑输入。"""

    input_name: str
    identity_key: tuple[int, ...]

    def __post_init__(self) -> None:
        """要求输入名唯一可读且身份为严格整数键。"""
        if not isinstance(self.input_name, str) or not self.input_name:
            raise W01ContractError("W-01 protocol input name 不能为空")
        _strict_key(self.identity_key, label="W-01 protocol input identity")

    def to_dict(self) -> dict[str, Any]:
        """导出供 K-03 冻结 manifest 消费的规范投影。"""
        return {
            "identity_key": list(self.identity_key),
            "input_name": self.input_name,
        }


@dataclass(frozen=True)
class W01FrozenContext:
    """由已发布 D-03 唯一派生的 W-01 只读协议上下文。"""

    d03_release_key: str
    d03_published: int
    d03_content_commit_sha1: str
    d03_receipt_sha256: str
    global_manifest_path: str
    global_manifest_sha256: str
    stage_manifest_path: str
    stage_manifest_sha256: str
    stage_key: str
    stage_ordinal: int
    next_stage_key: str
    train_pack_keys: tuple[str, ...]
    candidate_allowed_paths: tuple[str, ...]
    future_pack_count: int
    held_out_visible_count: int
    evaluator_visible_count: int
    payload_reads: int
    payload_bytes: int
    startup_protocol_file_count: int
    logical_shard_count: int
    allowed_worker_counts: tuple[int, ...]
    failure_point_keys: tuple[str, ...]
    merge_barrier_key: str
    cursor_version: str
    resource_budget: dict[str, int]
    version_keys: tuple[tuple[str, str], ...]
    protocol_inputs: tuple[W01ProtocolInput, ...]

    def __post_init__(self) -> None:
        """冻结 W-01 首阶段、零读取和 D-03 规定的 shard/worker 边界。"""
        if self.d03_published != 1:
            raise W01ContractError("D-03 receipt d03_published 不为 1")
        if (self.stage_key != W01_STAGE_KEY or self.stage_ordinal != 1
                or self.next_stage_key != W02_STAGE_KEY):
            raise W01ContractError("D-03 首阶段不是唯一 W-01 -> W-02 顺序")
        if self.train_pack_keys or self.candidate_allowed_paths:
            raise W01ContractError("W-01 不得具有训练 payload 白名单")
        if self.payload_reads != 0 or self.payload_bytes != 0:
            raise W01ContractError("W-01 装配阶段发生 payload 读取")
        if self.logical_shard_count != W01_PROTOCOL_INPUT_COUNT:
            raise W01ContractError("W-01 logical shard count 必须为 16")
        if self.allowed_worker_counts != (1, 2, 4):
            raise W01ContractError("W-01 worker count 必须冻结为 1/2/4")
        if (len(self.protocol_inputs) != W01_PROTOCOL_INPUT_COUNT
                or len({item.input_name for item in self.protocol_inputs})
                != W01_PROTOCOL_INPUT_COUNT):
            raise W01ContractError("W-01 protocol input 必须恰好 16 项且唯一")

    def stable_key(self) -> tuple[int, ...]:
        """返回绑定 receipt/global/stage/version/恢复协议的紧凑身份。"""
        return _digest_value({
            "allowed_worker_counts": list(self.allowed_worker_counts),
            "cursor_version": self.cursor_version,
            "d03_content_commit_sha1": self.d03_content_commit_sha1,
            "d03_receipt_sha256": self.d03_receipt_sha256,
            "d03_release_key": self.d03_release_key,
            "failure_point_keys": list(self.failure_point_keys),
            "global_manifest_path": self.global_manifest_path,
            "global_manifest_sha256": self.global_manifest_sha256,
            "logical_shard_count": self.logical_shard_count,
            "merge_barrier_key": self.merge_barrier_key,
            "protocol_inputs": [item.to_dict() for item in self.protocol_inputs],
            "resource_budget": dict(self.resource_budget),
            "stage_key": self.stage_key,
            "stage_manifest_path": self.stage_manifest_path,
            "stage_manifest_sha256": self.stage_manifest_sha256,
            "version_keys": [list(item) for item in self.version_keys],
        })

    def verified_zero_learning_state(
            self,
            *,
            protocol_execution_runs: int,
            ) -> dict[str, int]:
        """形成协议成功但语言能力、readiness 和学习写全零的独立状态。"""
        if type(protocol_execution_runs) is not int or protocol_execution_runs != 1:
            raise W01ContractError("W-01 成功必须且只能记一次 protocol execution")
        return {
            "LANGUAGE_CAPABILITY_MASTERED": 0,
            "LANGUAGE_READINESS": 0,
            "W01_PROTOCOL_VERIFIED": 1,
            "W02_STARTED": 0,
            "assessment_updates": 0,
            "companion_writes": 0,
            "core_learning_writes": 0,
            "evaluator_label_writes": 0,
            "formal_training_runs": 0,
            "mastered_claims": 0,
            "memory_learning_writes": 0,
            "protocol_execution_runs": protocol_execution_runs,
            "readiness_claims": 0,
            "teacher_calls": 0,
            "use_learning_writes": 0,
            "w02_semantic_writes": 0,
        }


@dataclass(frozen=True)
class W01RunRequest:
    """独立语言阶段 0 入口接受的完整 typed 请求。"""

    run_id: int
    parent_run_id: int
    base_run_id: int
    stage_key: str
    owner_key: str
    runner_key: str
    d03_context_key: tuple[int, ...]
    backend_profile_key: tuple[int, ...]
    base_fence_key: tuple[int, ...]
    worker_count: int
    mode: str
    requested_payload_paths: tuple[str, ...]

    def execution_identity_key(self) -> tuple[int, ...]:
        """返回不含 worker/mode 的执行身份，允许同 run 改调度恢复。"""
        return _digest_value({
            "backend_profile_key": list(self.backend_profile_key),
            "base_fence_key": list(self.base_fence_key),
            "base_run_id": self.base_run_id,
            "d03_context_key": list(self.d03_context_key),
            "owner_key": self.owner_key,
            "parent_run_id": self.parent_run_id,
            "run_id": self.run_id,
            "runner_key": self.runner_key,
            "stage_key": self.stage_key,
        })

    def scheduling_key(self) -> tuple[int, ...]:
        """返回只描述本次恢复模式和物理 worker 数的调度键。"""
        return _digest_value({"mode": self.mode, "worker_count": self.worker_count})


def _protocol_inputs(
        receipt: D03PublicationReceipt,
        reader: D03ReleaseReader,
        stage_index: int,
        candidate_paths: tuple[str, ...],
        ) -> tuple[W01ProtocolInput, ...]:
    """从 receipt/global/W-01 合同元数据派生固定十六个逻辑输入。"""
    stage = reader.stages[stage_index]
    visibility = stage.data_visibility
    recovery = stage.recovery_binding
    budget = stage.resource_budget
    identity = reader.global_manifest.release_identity
    values = (
        ("D03_RECEIPT_IDENTITY", receipt.to_dict()),
        ("D03_GLOBAL_IDENTITY", receipt.global_manifest_identity.to_dict()),
        ("D03_RELEASE_IDENTITY", identity.to_dict()),
        ("W01_STAGE_IDENTITY", stage.stage_identity.to_dict()),
        ("W01_STAGE_VISIBILITY", visibility.to_dict()),
        ("W01_RESOURCE_BUDGET", budget.to_dict()),
        ("W01_RECOVERY_BINDING", recovery.to_dict()),
        ("W01_EVALUATION_BINDING", stage.evaluation_binding.to_dict()),
        ("W01_EMPTY_TRAIN_INPUT", {"allowed_paths": list(candidate_paths)}),
        ("W01_EMPTY_LEARNING_STATE", verified_zero_learning_state_template()),
        ("W01_CURSOR_VERSION", {"cursor_version": recovery.cursor_version}),
        ("W01_MERGE_BARRIER", {"merge_barrier_key": recovery.merge_barrier_key}),
        ("W01_FAILURE_POINTS", {"keys": list(recovery.failure_point_keys)}),
        ("W01_VERSION_KEYS", {"keys": [list(item) for item in identity.version_keys]}),
        ("W01_OWNER", {"owner_key": W01_OWNER_KEY}),
        ("W01_NEXT_STAGE", {"next_stage_key": W02_STAGE_KEY, "started": 0}),
    )
    return tuple(W01ProtocolInput(name, _digest_value(value))
                 for name, value in values)


def verified_zero_learning_state_template() -> dict[str, int]:
    """返回尚未执行时用于协议输入的全零学习状态模板。"""
    return {
        "LANGUAGE_CAPABILITY_MASTERED": 0,
        "LANGUAGE_READINESS": 0,
        "W01_PROTOCOL_VERIFIED": 0,
        "W02_STARTED": 0,
        "companion_writes": 0,
        "core_learning_writes": 0,
        "formal_training_runs": 0,
        "memory_learning_writes": 0,
        "teacher_calls": 0,
        "use_learning_writes": 0,
        "w02_semantic_writes": 0,
    }


def open_w01_frozen_context(
        repository_root: str | Path,
        global_manifest_path: str = D03_GLOBAL_MANIFEST_PATH,
        *,
        dependency_root: str | Path | None = None,
        ) -> W01FrozenContext:
    """严格回读发布 receipt 与 D-03 reader，并只投影 W-01 协议元数据。"""
    primary = Path(repository_root).resolve()
    dependency = (Path(dependency_root).resolve()
                  if dependency_root is not None else primary)
    try:
        receipt_path = _overlay_path(primary, dependency, FORMAL_RECEIPT_PATH)
        receipt = read_d03_publication_receipt(receipt_path)
        if (receipt.execution_state.get("d03_published") != 1
                or receipt.publication_state.d03_published != 1):
            raise W01ContractError("D-03 receipt d03_published 不为 1")
        reader = D03ReleaseReader.open(
            primary,
            global_manifest_path,
            dependency_root=dependency,
            require_publication=True,
        )
    except W01ContractError:
        raise
    except (D03ContractError, OSError, ValueError, TypeError) as exc:
        raise W01ContractError(
            f"D-03 receipt d03_published/发布身份无效: {exc}") from exc
    if receipt.global_manifest_identity.relative_path != global_manifest_path:
        raise W01ContractError("D-03 receipt 与请求 global manifest 漂移")
    stages = reader.stages
    if not stages or stages[0].stage_identity.stage_key != W01_STAGE_KEY:
        raise W01ContractError("D-03 计划首阶段不是 W-01")
    stage = stages[0]
    stage_reference = reader.global_manifest.stage_manifests[0]
    if stage_reference.artifact_key != W01_STAGE_KEY:
        raise W01ContractError("D-03 W-01 stage reference 漂移")
    candidate = reader.visibility(W01_STAGE_KEY, "candidate")
    if candidate.allowed_paths:
        raise W01ContractError("W-01 candidate payload 白名单必须为空")
    identity = reader.global_manifest.release_identity
    protocols = _protocol_inputs(receipt, reader, 0, candidate.allowed_paths)
    startup_paths = {
        FORMAL_RECEIPT_PATH,
        global_manifest_path,
        identity.parent_gate_path,
        identity.capability_baseline_path,
        identity.source_coverage_path,
        reader.global_manifest.historical_hold_receipt.relative_path,
        reader.global_manifest.invalidation_graph.file_identity.relative_path,
        *(item.file_identity.relative_path
          for item in reader.global_manifest.stage_manifests),
        *(item.relative_path for item in reader.global_manifest.paper_files),
        *(item.evidence_identity.relative_path
          for item in reader.global_manifest.excluded_sources),
    }
    return W01FrozenContext(
        d03_release_key=receipt.release_key,
        d03_published=1,
        d03_content_commit_sha1=receipt.content_commit_sha1,
        d03_receipt_sha256=receipt.sha256(),
        global_manifest_path=receipt.global_manifest_identity.relative_path,
        global_manifest_sha256=receipt.global_manifest_identity.sha256,
        stage_manifest_path=stage_reference.file_identity.relative_path,
        stage_manifest_sha256=stage_reference.file_identity.sha256,
        stage_key=stage.stage_identity.stage_key,
        stage_ordinal=stage.stage_identity.ordinal,
        next_stage_key=W02_STAGE_KEY,
        train_pack_keys=stage.data_visibility.train_pack_keys,
        candidate_allowed_paths=candidate.allowed_paths,
        future_pack_count=len(stage.data_visibility.future_pack_keys),
        held_out_visible_count=len(stage.data_visibility.held_out_pack_keys),
        evaluator_visible_count=len(stage.data_visibility.evaluator_pack_keys),
        payload_reads=candidate.payload_reads,
        payload_bytes=candidate.payload_bytes,
        startup_protocol_file_count=len(startup_paths),
        logical_shard_count=stage.recovery_binding.logical_shard_count,
        allowed_worker_counts=stage.recovery_binding.allowed_worker_counts,
        failure_point_keys=stage.recovery_binding.failure_point_keys,
        merge_barrier_key=stage.recovery_binding.merge_barrier_key,
        cursor_version=stage.recovery_binding.cursor_version,
        resource_budget=stage.resource_budget.to_dict(),
        version_keys=identity.version_keys,
        protocol_inputs=protocols,
    )


def validate_w01_request(
        context: W01FrozenContext,
        request: W01RunRequest,
        ) -> W01RunRequest:
    """在任何执行/读取 payload 前核验阶段、owner、版本、worker 和恢复模式。"""
    if not isinstance(context, W01FrozenContext):
        raise W01ContractError("W-01 context 类型错误")
    if not isinstance(request, W01RunRequest):
        raise W01ContractError("W-01 request 类型错误")
    if (type(request.run_id) is not int or request.run_id <= 0
            or type(request.parent_run_id) is not int or request.parent_run_id < 0
            or type(request.base_run_id) is not int or request.base_run_id < 0):
        raise W01ContractError("run/parent/base id 必须是严格非负整数且 run 为正")
    if request.stage_key != context.stage_key or request.stage_key != W01_STAGE_KEY:
        raise W01ContractError("W-01 是唯一允许的首阶段")
    if request.owner_key != W01_OWNER_KEY:
        raise W01ContractError("W-01 transaction owner 未授权")
    if request.runner_key != W01_RUNNER_KEY:
        raise W01ContractError("必须使用独立语言 PH2 stage-0 runner")
    if request.d03_context_key != context.stable_key():
        raise W01ContractError("W-01 请求 D-03 identity 漂移")
    _strict_key(request.backend_profile_key, label="backend profile key")
    _strict_key(request.base_fence_key, label="base fence key")
    if request.worker_count not in context.allowed_worker_counts:
        raise W01ContractError("worker count 不在 D-03 允许集合")
    if request.mode not in W01_ALLOWED_MODES:
        raise W01ContractError("mode 必须是 fresh/restart/resume")
    if (not isinstance(request.requested_payload_paths, tuple)
            or request.requested_payload_paths):
        raise W01ContractError("W-01 不接受任何 corpus/private payload path")
    return request


__all__ = [
    "D03_GLOBAL_MANIFEST_PATH",
    "W01_ALLOWED_MODES",
    "W01_CURSOR_VERSION",
    "W01_OWNER_KEY",
    "W01_PROTOCOL_STATUS",
    "W01_RUNNER_KEY",
    "W01_STAGE_KEY",
    "W01ContractError",
    "W01FrozenContext",
    "W01ProtocolInput",
    "W01RunRequest",
    "open_w01_frozen_context",
    "validate_w01_request",
    "verified_zero_learning_state_template",
]
