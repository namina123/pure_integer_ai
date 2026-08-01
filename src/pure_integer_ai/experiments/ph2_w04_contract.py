"""PH2 W-04 冻结原语与表层对应的公开合同。

本模块只做 payload 前身份、owner、split、资源、恢复和 hard-conjunct
绑定；不读取任何训练 payload，不创建候选，不写学习状态。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    D03FileIdentity,
    sha1_text,
)
from pure_integer_ai.experiments.ph2_d03_release_catalog import (
    FORMAL_GLOBAL_MANIFEST_PATH,
    FORMAL_RECEIPT_PATH,
)
from pure_integer_ai.experiments.ph2_d03_release_reader import (
    D03ReleaseReader,
    VisibleArtifactFile,
)
from pure_integer_ai.experiments.ph2_dataset_contract import ArtifactFileIdentity
from pure_integer_ai.experiments.ph2_dataset_core import canonical_json_bytes
from pure_integer_ai.experiments.ph2_dataset_io import read_artifact_manifest
from pure_integer_ai.experiments.ph2_d03_stage_contract import EvaluationThreshold
from pure_integer_ai.experiments.ph2_j_lc_pre_w04_catalog import (
    build_j_lc_pre_w04_gate,
)
from pure_integer_ai.experiments.ph2_j_lc_pre_w04_contract import (
    MANIFEST_PATH as PRE_W04_GATE_PATH,
    PUBLISHED_STATE as PRE_W04_PUBLISHED_STATE,
    read_j_lc_pre_w04_gate,
    verify_j_lc_pre_w04_files,
)


W04_FORMAT_VERSION = 1
W04_STAGE_KEY = "W-04"
W04_PREREQUISITE_STAGE_KEY = "W-03"
W04_OWNER_KEY = "PH2_W04_TRANSACTION_OWNER"
W04_RUNNER_KEY = "PH2_LANGUAGE_STAGE4_PRIMITIVE_SURFACE"
W04_ALLOWED_MODES = ("fresh", "restart", "resume")
W04_ALLOWED_WORKER_COUNTS = (1, 2, 4)
W04_FORMAL_RUN_ID = 5
W04_W03_BASE_RUN_ID = 4
W04_TRAIN_PACK_KEYS = (
    "AUTHORED_CC0_V1--CC0-1.0--lc01-text-fidelity-v1",
    "AUTHORED_CC0_V1--CC0-1.0--lc02-morphology-v1",
    "AUTHORED_CC0_V1--CC0-1.0--lc03-construction-v1",
    "AUTHORED_CC0_V1--CC0-1.0--primitive-v1",
    "AUTHORED_CC0_V1--CC0-1.0--sense-v1",
    "WIKIDATA_REVISION_V1--CC0-1.0--source-pack-v1",
    "ZHWIKTIONARY_20260701--CC-BY-SA-4.0--source-pack-v1",
)
W04_DIMENSION_KEYS = (
    "W-04-CONTENT_REPLACEMENT",
    "W-04-CUE_REPLACEMENT",
    "W-04-EVIDENCE_ABLATION",
    "W-04-SEED_ABLATION",
)
W04_ABLATION_KEYS = tuple(f"{key}-ABLATION" for key in W04_DIMENSION_KEYS)
W04_GENERATION_HARD_CONJUNCT = (
    "W-04-GENERATION-PRIMITIVE-SURFACE-HARD-CONJUNCT"
)
W04_EVALUATION_ORDER = (*W04_DIMENSION_KEYS, W04_GENERATION_HARD_CONJUNCT)
W04_AGGREGATION_POLICY = (
    "ALL_4_W04_BEARINGS_AND_GENERATION_MUST_PASS"
)
W04_LOGICAL_CLOCK_VERSION = "PH2-W04-LOGICAL-CLOCK-V1"
W04_ALLOWED_WRITE_OWNERS = (W04_OWNER_KEY,)
W04_FORBIDDEN_WRITE_OWNERS = (
    "PH2_W01_TRANSACTION_OWNER",
    "PH2_W02_TRANSACTION_OWNER",
    "PH2_W03_TRANSACTION_OWNER",
    "PH2_PRIVATE_EVALUATOR",
    "MEMORY",
    "COMPANION",
)
W04_ZERO_EXECUTION_STATE = {
    "LANGUAGE_CAPABILITY_MASTERED": 0,
    "LANGUAGE_READINESS": 0,
    "W04_STARTED": 0,
    "W05_STARTED": 0,
    "formal_w04_training_runs": 0,
    "teacher_calls": 0,
}
W04_OPEN_GENERATION_STATE = "NE_NOT_YET_EVALUABLE"
W04_RESOURCE_BUDGET = {
    "max_checkpoint_count": 1024,
    "max_logic_operations": 4_000_000,
    "max_payload_bytes": 268_435_456,
    "max_payload_gets": 262_144,
    "max_recompute_objects": 400_000,
    "max_records": 400_000,
    "max_segments": 16_384,
    "max_workers": 4,
}
W04_D03_GLOBAL_MANIFEST_SHA256 = (
    "384329cf651ea4c5e4bc9d0b5dc4da7b22a71bc008bfabe468c86278dd9d40b6"
)
W04_STAGE_MANIFEST_SHA256 = (
    "a9b10ab0c65d14db89a962f2d0c055231484052996d047ec1bc0480fda2d9e84"
)
W04_PRE_GATE_SHA256 = (
    "c37bab6f02bd3adab2c546b5f79f070e3d232c481c70ef751727ea1edeff8c82"
)


class W04ContractError(RuntimeError):
    """W-04 冻结身份、owner/split 或 payload 前授权失败。"""


def digest_value(value: Any) -> tuple[int, ...]:
    """为无浮点值返回确定性整数 tuple 摘要。"""
    return tuple(hashlib.sha256(canonical_json_bytes(value)).digest())


def strict_key(value: object, *, label: str) -> tuple[int, ...]:
    """要求键是只含严格整数的非空 tuple。"""
    if (not isinstance(value, tuple) or not value
            or any(type(item) is not int for item in value)):
        raise W04ContractError(f"{label} must be a non-empty strict integer tuple")
    return value


def safe_relative_path(value: object, *, label: str) -> str:
    """要求路径是无逃逸的规范 POSIX 相对路径。"""
    if not isinstance(value, str) or not value:
        raise W04ContractError(f"{label} must be a non-empty path")
    path = PurePosixPath(value)
    if (path.is_absolute() or ".." in path.parts or "\\" in value
            or ":" in value
            or path.as_posix() != value or "//" in value):
        raise W04ContractError(f"{label} is not a canonical safe path")
    return value


def _strict_sha256(value: object, *, label: str) -> str:
    """验证规范小写 SHA-256。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(item not in "0123456789abcdef" for item in value)):
        raise W04ContractError(f"{label} is not canonical SHA-256")
    return value


@dataclass(frozen=True, order=True)
class W04PayloadBinding:
    """把一个授权路径绑定到 pack、owner/split 和完整文件身份。"""

    relative_path: str
    pack_key: str
    owner_kind: str
    split: str | None
    record_count: int
    transport_size_bytes: int
    transport_sha256: str
    content_size_bytes: int
    content_sha256: str
    file_identity: ArtifactFileIdentity

    def __post_init__(self) -> None:
        object.__setattr__(self, "relative_path", safe_relative_path(
            self.relative_path, label="payload path"))
        if not isinstance(self.pack_key, str) or not self.pack_key:
            raise W04ContractError("payload pack key is empty")
        if self.owner_kind not in {"source", "observation", "teacher", "evaluator"}:
            raise W04ContractError("payload owner kind is invalid")
        if self.split not in {None, "train", "dev", "held_out"}:
            raise W04ContractError("payload split is invalid")
        for name in ("record_count", "transport_size_bytes", "content_size_bytes"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise W04ContractError(
                    f"payload {name} must be a positive strict integer")
        _strict_sha256(self.transport_sha256, label="payload transport digest")
        _strict_sha256(self.content_sha256, label="payload content digest")
        if not isinstance(self.file_identity, ArtifactFileIdentity):
            raise W04ContractError("payload is missing ArtifactFileIdentity")
        identity = self.file_identity
        if (not self.relative_path.endswith("/" + identity.relative_path)
                or identity.owner_kind != self.owner_kind
                or identity.split != self.split
                or identity.record_count != self.record_count
                or identity.transport_size_bytes != self.transport_size_bytes
                or identity.transport_sha256 != self.transport_sha256
                or identity.content_size_bytes != self.content_size_bytes
                or identity.content_sha256 != self.content_sha256):
            raise W04ContractError("payload identity does not match its D-03 path")

    def to_dict(self) -> dict[str, object]:
        """导出可审计 payload binding。"""
        return {
            "content_sha256": self.content_sha256,
            "content_size_bytes": self.content_size_bytes,
            "file_identity": self.file_identity.to_dict(),
            "owner_kind": self.owner_kind,
            "pack_key": self.pack_key,
            "record_count": self.record_count,
            "relative_path": self.relative_path,
            "split": self.split,
            "transport_sha256": self.transport_sha256,
            "transport_size_bytes": self.transport_size_bytes,
        }


@dataclass(frozen=True, order=True)
class W04PackBinding:
    """冻结 W-04 可见的七个 train pack manifest。"""

    pack_key: str
    source_key: str
    license_id: str
    earliest_stage: str
    manifest_identity: D03FileIdentity
    total_record_count: int
    source_cluster_count: int

    def __post_init__(self) -> None:
        if self.pack_key not in W04_TRAIN_PACK_KEYS:
            raise W04ContractError("W-04 pack is outside the frozen train set")
        if not self.source_key or not self.license_id:
            raise W04ContractError("W-04 pack source/license is empty")
        if self.earliest_stage not in {"W-02", "W-03", "W-04"}:
            raise W04ContractError("W-04 train pack earliest stage drifted")
        if not isinstance(self.manifest_identity, D03FileIdentity):
            raise W04ContractError("W-04 pack manifest identity is missing")
        if (type(self.total_record_count) is not int or self.total_record_count <= 0
                or type(self.source_cluster_count) is not int
                or self.source_cluster_count <= 0):
            raise W04ContractError("W-04 pack counts must be positive strict integers")

    def to_dict(self) -> dict[str, object]:
        """导出 pack binding。"""
        return {
            "earliest_stage": self.earliest_stage,
            "license_id": self.license_id,
            "manifest_identity": self.manifest_identity.to_dict(),
            "pack_key": self.pack_key,
            "source_cluster_count": self.source_cluster_count,
            "source_key": self.source_key,
            "total_record_count": self.total_record_count,
        }


@dataclass
class W04PayloadAudit:
    """分开记录 transport 工作和完整 consumer 交付。"""

    transport_attempts: int = 0
    transport_bytes: int = 0
    payload_gets: int = 0
    payload_bytes: int = 0
    source_ref_reads: int = 0
    observation_reads: int = 0
    teacher_evidence_reads: int = 0
    teacher_calls: int = 0
    learning_writes: int = 0


@dataclass(frozen=True)
class W04FrozenContext:
    """汇总打开 W-04 payload 前所需的全部不可变身份。"""

    current_remote_commit_sha1: str
    d03_receipt_identity: D03FileIdentity
    d03_global_manifest_identity: D03FileIdentity
    stage_manifest_identity: D03FileIdentity
    pre_w04_gate_sha256: str
    pre_w04_gate_key: tuple[int, ...]
    stage_key: str
    stage_ordinal: int
    prerequisite_stage_keys: tuple[str, ...]
    train_pack_keys: tuple[str, ...]
    pack_bindings: tuple[W04PackBinding, ...]
    candidate_payload_bindings: tuple[W04PayloadBinding, ...]
    teacher_evidence_bindings: tuple[W04PayloadBinding, ...]
    evaluator_visible_bindings: tuple[W04PayloadBinding, ...]
    d03_thresholds: tuple[EvaluationThreshold, ...]
    d03_ablation_keys: tuple[str, ...]
    dimension_keys: tuple[str, ...]
    ablation_keys: tuple[str, ...]
    generation_hard_conjunct: str
    evaluation_order: tuple[str, ...]
    aggregation_policy: str
    allowed_worker_counts: tuple[int, ...]
    failure_point_keys: tuple[str, ...]
    logical_shard_count: int
    merge_barrier_key: str
    cursor_version: str
    logical_clock_version: str
    resource_budget: dict[str, int]
    version_keys: tuple[tuple[str, str], ...]
    run_id: int
    parent_run_id: int
    base_run_id: int
    backend_profile_key: tuple[int, ...]
    base_fence_key: tuple[int, ...]
    owner_key: str
    allowed_write_owners: tuple[str, ...]
    forbidden_write_owners: tuple[str, ...]
    execution_state: dict[str, int]
    open_generation_state: str
    payload_gets: int = 0
    payload_bytes: int = 0
    learning_writes: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "current_remote_commit_sha1", sha1_text(
            self.current_remote_commit_sha1, where="W-04 current remote commit"))
        if self.stage_key != W04_STAGE_KEY or self.stage_ordinal != 4:
            raise W04ContractError("W-04 stage identity drifted")
        if self.prerequisite_stage_keys != (W04_PREREQUISITE_STAGE_KEY,):
            raise W04ContractError("W-04 prerequisite stage drifted")
        if self.train_pack_keys != W04_TRAIN_PACK_KEYS:
            raise W04ContractError("W-04 train pack order or identity drifted")
        if tuple(item.pack_key for item in self.pack_bindings) != W04_TRAIN_PACK_KEYS:
            raise W04ContractError("W-04 seven-pack bindings are incomplete")
        _strict_sha256(self.pre_w04_gate_sha256, label="pre-W04 gate")
        if self.pre_w04_gate_sha256 != W04_PRE_GATE_SHA256:
            raise W04ContractError("pre-W04 gate SHA drifted")
        strict_key(self.pre_w04_gate_key, label="pre-W04 gate key")
        candidate = tuple(item.relative_path for item in self.candidate_payload_bindings)
        teacher = tuple(item.relative_path for item in self.teacher_evidence_bindings)
        evaluator = tuple(item.relative_path for item in self.evaluator_visible_bindings)
        if any(len(paths) != len(set(paths)) for paths in (candidate, teacher, evaluator)):
            raise W04ContractError("W-04 visible path set contains duplicates")
        if set(candidate) & set(teacher):
            raise W04ContractError("candidate and teacher-only bindings overlap")
        if any((item.owner_kind, item.split) not in {
                ("source", None), ("observation", "train")}
               for item in self.candidate_payload_bindings):
            raise W04ContractError("candidate binding contains non-train/private data")
        if any((item.owner_kind, item.split) != ("teacher", "train")
               for item in self.teacher_evidence_bindings):
            raise W04ContractError("teacher binding contains non-train Evidence")
        if any((item.owner_kind, item.split) not in {
                ("source", None), ("observation", "dev"),
                ("observation", "held_out"), ("evaluator", "dev"),
                ("evaluator", "held_out")}
               for item in self.evaluator_visible_bindings):
            raise W04ContractError("evaluator binding owner/split drifted")
        if (tuple(item.dimension_key for item in self.d03_thresholds)
                != W04_DIMENSION_KEYS
                or self.d03_ablation_keys != W04_ABLATION_KEYS
                or self.dimension_keys != W04_DIMENSION_KEYS
                or self.ablation_keys != W04_ABLATION_KEYS):
            raise W04ContractError("W-04 frozen bearing or ablation identity drifted")
        if any(
                item.bearing != 1
                or item.min_pass_numerator != 1
                or item.min_pass_denominator != 1
                or item.max_fail_count != 0
                or item.ne_policy != "BLOCK"
                or item.preregistered != 1
                for item in self.d03_thresholds):
            raise W04ContractError("W-04 D-03 thresholds were relaxed")
        if (self.generation_hard_conjunct != W04_GENERATION_HARD_CONJUNCT
                or self.evaluation_order != W04_EVALUATION_ORDER
                or self.aggregation_policy != W04_AGGREGATION_POLICY):
            raise W04ContractError("W-04 generation or total hard conjunct drifted")
        if (self.allowed_worker_counts != W04_ALLOWED_WORKER_COUNTS
                or self.logical_shard_count != 16
                or len(self.failure_point_keys) != 6
                or self.resource_budget != W04_RESOURCE_BUDGET):
            raise W04ContractError("W-04 recovery or resource contract drifted")
        if (self.run_id != W04_FORMAL_RUN_ID
                or self.parent_run_id != W04_W03_BASE_RUN_ID
                or self.base_run_id != W04_W03_BASE_RUN_ID):
            raise W04ContractError("W-04 run/parent/base identity drifted")
        strict_key(self.backend_profile_key, label="backend profile key")
        strict_key(self.base_fence_key, label="base fence key")
        if (self.owner_key != W04_OWNER_KEY
                or self.allowed_write_owners != W04_ALLOWED_WRITE_OWNERS
                or self.forbidden_write_owners != W04_FORBIDDEN_WRITE_OWNERS
                or set(self.allowed_write_owners) & set(self.forbidden_write_owners)):
            raise W04ContractError("W-04 transaction/write owner boundary drifted")
        if (self.logical_clock_version != W04_LOGICAL_CLOCK_VERSION
                or self.execution_state != W04_ZERO_EXECUTION_STATE
                or self.open_generation_state != W04_OPEN_GENERATION_STATE):
            raise W04ContractError("W-04 pre-training state is not frozen")
        if self.payload_gets != 0 or self.payload_bytes != 0 or self.learning_writes != 0:
            raise W04ContractError("W-04 context construction touched payload or learning")

    def stable_key(self) -> tuple[int, ...]:
        """绑定 pre-gate、D-03、payload、恢复和状态。"""
        return digest_value({
            "ablation_keys": list(self.ablation_keys),
            "aggregation_policy": self.aggregation_policy,
            "allowed_worker_counts": list(self.allowed_worker_counts),
            "allowed_write_owners": list(self.allowed_write_owners),
            "backend_profile_key": list(self.backend_profile_key),
            "base_fence_key": list(self.base_fence_key),
            "base_run_id": self.base_run_id,
            "candidate_payload_bindings": [
                item.to_dict() for item in self.candidate_payload_bindings],
            "current_remote_commit_sha1": self.current_remote_commit_sha1,
            "cursor_version": self.cursor_version,
            "d03_global_manifest_identity": self.d03_global_manifest_identity.to_dict(),
            "d03_receipt_identity": self.d03_receipt_identity.to_dict(),
            "d03_thresholds": [item.to_dict() for item in self.d03_thresholds],
            "evaluation_order": list(self.evaluation_order),
            "evaluator_visible_bindings": [
                item.to_dict() for item in self.evaluator_visible_bindings],
            "execution_state": dict(self.execution_state),
            "failure_point_keys": list(self.failure_point_keys),
            "forbidden_write_owners": list(self.forbidden_write_owners),
            "logical_clock_version": self.logical_clock_version,
            "logical_shard_count": self.logical_shard_count,
            "merge_barrier_key": self.merge_barrier_key,
            "open_generation_state": self.open_generation_state,
            "owner_key": self.owner_key,
            "pack_bindings": [item.to_dict() for item in self.pack_bindings],
            "parent_run_id": self.parent_run_id,
            "pre_w04_gate_key": list(self.pre_w04_gate_key),
            "pre_w04_gate_sha256": self.pre_w04_gate_sha256,
            "resource_budget": dict(self.resource_budget),
            "run_id": self.run_id,
            "stage_manifest_identity": self.stage_manifest_identity.to_dict(),
            "teacher_evidence_bindings": [
                item.to_dict() for item in self.teacher_evidence_bindings],
            "version_keys": [list(item) for item in self.version_keys],
        })


@dataclass(frozen=True)
class W04RunRequest:
    """不含 evaluator、future 或 private 字段的完整 candidate 请求。"""

    run_id: int
    parent_run_id: int
    base_run_id: int
    stage_key: str
    owner_key: str
    runner_key: str
    current_remote_commit_sha1: str
    pre_w04_gate_key: tuple[int, ...]
    d03_context_key: tuple[int, ...]
    backend_profile_key: tuple[int, ...]
    base_fence_key: tuple[int, ...]
    worker_count: int
    mode: str
    resource_budget: tuple[tuple[str, int], ...]
    candidate_payload_paths: tuple[str, ...]
    teacher_evidence_paths: tuple[str, ...]

    def execution_identity_key(self) -> tuple[int, ...]:
        """返回排除调度 mode/worker 的正式执行身份。"""
        return digest_value({
            "backend_profile_key": list(self.backend_profile_key),
            "base_fence_key": list(self.base_fence_key),
            "base_run_id": self.base_run_id,
            "candidate_payload_paths": list(self.candidate_payload_paths),
            "current_remote_commit_sha1": self.current_remote_commit_sha1,
            "d03_context_key": list(self.d03_context_key),
            "owner_key": self.owner_key,
            "parent_run_id": self.parent_run_id,
            "pre_w04_gate_key": list(self.pre_w04_gate_key),
            "resource_budget": dict(self.resource_budget),
            "run_id": self.run_id,
            "runner_key": self.runner_key,
            "stage_key": self.stage_key,
            "teacher_evidence_paths": list(self.teacher_evidence_paths),
        })

    def scheduling_key(self) -> tuple[int, ...]:
        """返回仅表示执行调度形态的稳定键。"""
        return digest_value({"mode": self.mode, "worker_count": self.worker_count})


def validate_w04_request(
        context: W04FrozenContext,
        request: W04RunRequest,
        ) -> W04RunRequest:
    """在任何 payload transport 前拒绝全部身份和路径漂移。"""
    if not isinstance(context, W04FrozenContext):
        raise W04ContractError("W-04 context type is invalid")
    if not isinstance(request, W04RunRequest):
        raise W04ContractError("W-04 request type is invalid")
    if (type(request.run_id) is not int
            or request.run_id != context.run_id
            or type(request.parent_run_id) is not int
            or request.parent_run_id != context.parent_run_id
            or type(request.base_run_id) is not int
            or request.base_run_id != context.base_run_id):
        raise W04ContractError("W-04 run/parent/base id did not continue W-03 run 4")
    if request.stage_key != W04_STAGE_KEY or request.stage_key != context.stage_key:
        raise W04ContractError("W-04 is the only authorized stage")
    if request.owner_key != context.owner_key or request.owner_key != W04_OWNER_KEY:
        raise W04ContractError("W-04 transaction owner is unauthorized")
    if request.runner_key != W04_RUNNER_KEY:
        raise W04ContractError("W-04 requires the independent stage-4 runner")
    if request.current_remote_commit_sha1 != context.current_remote_commit_sha1:
        raise W04ContractError("W-04 current remote commit drifted")
    if request.pre_w04_gate_key != context.pre_w04_gate_key:
        raise W04ContractError("pre-W04 gate identity drifted")
    if request.d03_context_key != context.stable_key():
        raise W04ContractError("D-03/W-04 frozen context identity drifted")
    if request.backend_profile_key != context.backend_profile_key:
        raise W04ContractError("backend profile identity drifted")
    if request.base_fence_key != context.base_fence_key:
        raise W04ContractError("W-03 base fence identity drifted")
    if request.worker_count not in context.allowed_worker_counts:
        raise W04ContractError("worker count is outside 1/2/4")
    if request.mode not in W04_ALLOWED_MODES:
        raise W04ContractError("mode must be fresh/restart/resume")
    expected_budget = tuple(sorted(context.resource_budget.items()))
    if request.resource_budget != expected_budget:
        raise W04ContractError("W-04 resource budget identity drifted")
    expected_candidate = tuple(
        item.relative_path for item in context.candidate_payload_bindings)
    expected_teacher = tuple(
        item.relative_path for item in context.teacher_evidence_bindings)
    for path in (*request.candidate_payload_paths, *request.teacher_evidence_paths):
        safe_relative_path(path, label="request payload path")
    if request.candidate_payload_paths != expected_candidate:
        raise W04ContractError("candidate payload paths are not the exact train whitelist")
    if request.teacher_evidence_paths != expected_teacher:
        raise W04ContractError("teacher Evidence paths are not the exact train whitelist")
    evaluator_private = {
        item.relative_path for item in context.evaluator_visible_bindings
        if item.owner_kind != "source"
    }
    requested = set(request.candidate_payload_paths) | set(request.teacher_evidence_paths)
    if evaluator_private & requested:
        raise W04ContractError("candidate request contains evaluator/private data")
    return request


def _payload_binding(item: VisibleArtifactFile) -> W04PayloadBinding:
    """把 D-03 reader 可见文件转为 W-04 payload binding。"""
    identity = item.file_identity
    return W04PayloadBinding(
        relative_path=item.relative_path,
        pack_key=item.pack_key,
        owner_kind=identity.owner_kind,
        split=identity.split,
        record_count=identity.record_count,
        transport_size_bytes=identity.transport_size_bytes,
        transport_sha256=identity.transport_sha256,
        content_size_bytes=identity.content_size_bytes,
        content_sha256=identity.content_sha256,
        file_identity=identity,
    )


def _overlay_path(primary: Path, dependency: Path, relative: str) -> Path:
    """在候选根和依赖根中只读解析一个公开 metadata 文件。"""
    parts = Path(*PurePosixPath(relative).parts)
    for root in (primary, dependency):
        target = (root / parts).resolve()
        if target.is_relative_to(root) and target.is_file():
            return target
    raise W04ContractError(f"frozen W-04 metadata file is missing: {relative}")


def _file_identity(primary: Path, dependency: Path, relative: str) -> D03FileIdentity:
    """返回公开 metadata 的现场文件身份。"""
    path = _overlay_path(primary, dependency, relative)
    payload = path.read_bytes()
    return D03FileIdentity(relative, len(payload), hashlib.sha256(payload).hexdigest())


def _verify_pack_manifest(primary: Path, dependency: Path, pack) -> None:
    """逐字段闭合 pack manifest 的来源、许可、路径、owner 和 split。"""
    path = _overlay_path(primary, dependency, pack.manifest_identity.relative_path)
    payload = path.read_bytes()
    if (len(payload) != pack.manifest_identity.size_bytes
            or hashlib.sha256(payload).hexdigest() != pack.manifest_identity.sha256):
        raise W04ContractError("W-04 pack manifest identity drifted")
    manifest = read_artifact_manifest(path)
    if (manifest.source_key != pack.source_key
            or manifest.license_partition != pack.license_id
            or manifest.record_count != pack.total_record_count
            or len(manifest.source_cluster_keys) != pack.source_cluster_count
            or pack.earliest_stage not in manifest.w_stages):
        raise W04ContractError("W-04 pack source/license/count/stage drifted")
    prefix = PurePosixPath(pack.manifest_identity.relative_path).parent
    files = {
        PurePosixPath(prefix, item.relative_path).as_posix(): item
        for item in manifest.files
    }
    if set(files) != set(pack.payload_paths):
        raise W04ContractError("W-04 pack manifest does not cover its exact paths")


def open_w04_frozen_context(
        repository_root: str | Path,
        global_manifest_path: str = FORMAL_GLOBAL_MANIFEST_PATH,
        *,
        current_remote_commit_sha1: str,
        backend_profile_key: tuple[int, ...],
        dependency_root: str | Path | None = None,
        ) -> W04FrozenContext:
    """只读 pre-gate、D-03 和 W-04 manifest，并保持 payload 计数为零。"""
    strict_key(backend_profile_key, label="backend profile key")
    primary = Path(repository_root).resolve()
    dependency = (Path(dependency_root).resolve()
                  if dependency_root is not None else primary)
    gate_path = _overlay_path(primary, dependency, PRE_W04_GATE_PATH)
    gate = read_j_lc_pre_w04_gate(gate_path)
    if (gate.canonical_bytes() != build_j_lc_pre_w04_gate(primary).canonical_bytes()
            or gate.sha256() != W04_PRE_GATE_SHA256
            or gate.published_state != PRE_W04_PUBLISHED_STATE
            or gate.published_state.get("W04_ALLOWED") != 1
            or gate.published_state.get("W04_STARTED") != 0):
        raise W04ContractError("pre-W04 gate is not the frozen PASS/allowed state")
    verify_j_lc_pre_w04_files(gate, repository_root=primary)
    reader = D03ReleaseReader.open(
        primary,
        global_manifest_path,
        dependency_root=dependency,
        require_publication=True,
    )
    d03_receipt_identity = _file_identity(primary, dependency, FORMAL_RECEIPT_PATH)
    global_identity = _file_identity(primary, dependency, global_manifest_path)
    if global_identity.sha256 != W04_D03_GLOBAL_MANIFEST_SHA256:
        raise W04ContractError("D-03 global manifest identity drifted")
    stage_index = 3
    stage = reader.stages[stage_index]
    stage_reference = reader.global_manifest.stage_manifests[stage_index]
    if stage_reference.artifact_key != W04_STAGE_KEY:
        raise W04ContractError("D-03 W-04 stage reference drifted")
    stage_identity = stage_reference.file_identity
    if stage_identity.sha256 != W04_STAGE_MANIFEST_SHA256:
        raise W04ContractError("W-04 stage manifest identity drifted")
    candidate_view = reader.visibility(W04_STAGE_KEY, "candidate")
    teacher_view = reader.visibility(W04_STAGE_KEY, "teacher")
    evaluator_view = reader.visibility(W04_STAGE_KEY, "evaluator")
    if any(view.payload_reads != 0 or view.payload_bytes != 0
           for view in (candidate_view, teacher_view, evaluator_view)):
        raise W04ContractError("W-04 visibility reader touched payload")
    candidate_traces = reader.visible_file_identities(W04_STAGE_KEY, "candidate")
    teacher_traces = reader.visible_file_identities(W04_STAGE_KEY, "teacher")
    evaluator_traces = reader.visible_file_identities(W04_STAGE_KEY, "evaluator")
    candidate_paths = {item.relative_path for item in candidate_traces}
    teacher_only_traces = tuple(
        item for item in teacher_traces if item.relative_path not in candidate_paths)

    pack_catalog = {
        item.pack_key: item for item in reader.global_manifest.pack_bindings}
    packs: list[W04PackBinding] = []
    for key in W04_TRAIN_PACK_KEYS:
        pack = pack_catalog.get(key)
        if pack is None:
            raise W04ContractError("W-04 frozen train pack is missing")
        _verify_pack_manifest(primary, dependency, pack)
        packs.append(W04PackBinding(
            pack_key=pack.pack_key,
            source_key=pack.source_key,
            license_id=pack.license_id,
            earliest_stage=pack.earliest_stage,
            manifest_identity=pack.manifest_identity,
            total_record_count=pack.total_record_count,
            source_cluster_count=pack.source_cluster_count,
        ))
    evaluation = stage.evaluation_binding
    recovery = stage.recovery_binding
    release = reader.global_manifest.release_identity
    pre_gate_key = digest_value(gate.to_dict())
    base_fence = digest_value({
        "gate": gate.sha256(),
        "stage": stage_identity.sha256,
        "global": global_identity.sha256,
    })
    return W04FrozenContext(
        current_remote_commit_sha1=current_remote_commit_sha1,
        d03_receipt_identity=d03_receipt_identity,
        d03_global_manifest_identity=global_identity,
        stage_manifest_identity=stage_identity,
        pre_w04_gate_sha256=gate.sha256(),
        pre_w04_gate_key=pre_gate_key,
        stage_key=stage.stage_identity.stage_key,
        stage_ordinal=stage.stage_identity.ordinal,
        prerequisite_stage_keys=stage.stage_identity.prerequisite_stage_keys,
        train_pack_keys=stage.data_visibility.train_pack_keys,
        pack_bindings=tuple(packs),
        candidate_payload_bindings=tuple(_payload_binding(item)
                                         for item in candidate_traces),
        teacher_evidence_bindings=tuple(_payload_binding(item)
                                        for item in teacher_only_traces),
        evaluator_visible_bindings=tuple(_payload_binding(item)
                                         for item in evaluator_traces),
        d03_thresholds=evaluation.thresholds,
        d03_ablation_keys=evaluation.ablation_keys,
        dimension_keys=W04_DIMENSION_KEYS,
        ablation_keys=W04_ABLATION_KEYS,
        generation_hard_conjunct=W04_GENERATION_HARD_CONJUNCT,
        evaluation_order=W04_EVALUATION_ORDER,
        aggregation_policy=W04_AGGREGATION_POLICY,
        allowed_worker_counts=recovery.allowed_worker_counts,
        failure_point_keys=recovery.failure_point_keys,
        logical_shard_count=recovery.logical_shard_count,
        merge_barrier_key=recovery.merge_barrier_key,
        cursor_version=recovery.cursor_version,
        logical_clock_version=W04_LOGICAL_CLOCK_VERSION,
        resource_budget=stage.resource_budget.to_dict(),
        version_keys=release.version_keys,
        run_id=W04_FORMAL_RUN_ID,
        parent_run_id=W04_W03_BASE_RUN_ID,
        base_run_id=W04_W03_BASE_RUN_ID,
        backend_profile_key=backend_profile_key,
        base_fence_key=base_fence,
        owner_key=W04_OWNER_KEY,
        allowed_write_owners=W04_ALLOWED_WRITE_OWNERS,
        forbidden_write_owners=W04_FORBIDDEN_WRITE_OWNERS,
        execution_state=dict(W04_ZERO_EXECUTION_STATE),
        open_generation_state=W04_OPEN_GENERATION_STATE,
    )


__all__ = [
    "W04_ABLATION_KEYS",
    "W04_AGGREGATION_POLICY",
    "W04_ALLOWED_MODES",
    "W04_ALLOWED_WORKER_COUNTS",
    "W04_DIMENSION_KEYS",
    "W04_EVALUATION_ORDER",
    "W04_FORMAL_RUN_ID",
    "W04_GENERATION_HARD_CONJUNCT",
    "W04_OPEN_GENERATION_STATE",
    "W04_OWNER_KEY",
    "W04_RESOURCE_BUDGET",
    "W04_RUNNER_KEY",
    "W04_STAGE_KEY",
    "W04_TRAIN_PACK_KEYS",
    "W04_W03_BASE_RUN_ID",
    "W04ContractError",
    "W04FrozenContext",
    "W04PackBinding",
    "W04PayloadAudit",
    "W04PayloadBinding",
    "W04RunRequest",
    "digest_value",
    "open_w04_frozen_context",
    "safe_relative_path",
    "strict_key",
    "validate_w04_request",
]
