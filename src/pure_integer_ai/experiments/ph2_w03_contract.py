"""PH2 W-03 冻结身份、请求和 payload 前合同值。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    D03FileIdentity,
    sha1_text,
)
from pure_integer_ai.experiments.ph2_dataset_contract import ArtifactFileIdentity
from pure_integer_ai.experiments.ph2_dataset_core import canonical_json_bytes
from pure_integer_ai.experiments.ph2_d03_stage_contract import EvaluationThreshold


W03_FORMAT_VERSION = 1
W03_STAGE_KEY = "W-03"
W03_PREREQUISITE_STAGE_KEY = "W-02"
W03_OWNER_KEY = "PH2_W03_TRANSACTION_OWNER"
W03_RUNNER_KEY = "PH2_LANGUAGE_STAGE2"
W03_ALLOWED_MODES = ("fresh", "restart", "resume")
W03_ALLOWED_WORKER_COUNTS = (1, 2, 4)
W03_FORMAL_RUN_ID = 4
W03_W02_BASE_RUN_ID = 3
W03_TRAIN_PACK_KEYS = (
    "AUTHORED_CC0_V1--CC0-1.0--lc01-text-fidelity-v1",
    "AUTHORED_CC0_V1--CC0-1.0--lc02-morphology-v1",
    "AUTHORED_CC0_V1--CC0-1.0--lc03-construction-v1",
    "AUTHORED_CC0_V1--CC0-1.0--sense-v1",
    "WIKIDATA_REVISION_V1--CC0-1.0--source-pack-v1",
    "ZHWIKTIONARY_20260701--CC-BY-SA-4.0--source-pack-v1",
)
W03_DIMENSION_KEYS = (
    "W-03-CONCEPT-SPLIT",
    "W-03-POLYSEMY-COMPETITION",
    "W-03-SOURCE-CONFLICT",
    "W-03-SUPERSEDE",
)
W03_D03_DIMENSION_KEYS = (
    "W-03-CONCEPT_SPLIT",
    "W-03-POLYSEMY_COMPETITION",
    "W-03-SOURCE_CONFLICT",
    "W-03-SUPERSEDE",
)
W03_DIMENSION_KEY_MAP = tuple(zip(
    W03_D03_DIMENSION_KEYS,
    W03_DIMENSION_KEYS,
    strict=True,
))
W03_ABLATION_KEYS = tuple(f"{key}-ABLATION" for key in W03_DIMENSION_KEYS)
W03_D03_ABLATION_KEYS = tuple(
    f"{key}-ABLATION" for key in W03_D03_DIMENSION_KEYS)
W03_GENERATION_HARD_CONJUNCT = (
    "W-03-GENERATION-SENSE-SELECTION-HARD-CONJUNCT"
)
W03_EVALUATION_ORDER = (*W03_DIMENSION_KEYS, W03_GENERATION_HARD_CONJUNCT)
W03_AGGREGATION_POLICY = "ALL_W03_BEARINGS_AND_GENERATION_MUST_PASS"
W03_LOGICAL_CLOCK_VERSION = "PH2-W03-LOGICAL-CLOCK-V1"
W03_ALLOWED_WRITE_OWNERS = (W03_OWNER_KEY,)
W03_FORBIDDEN_WRITE_OWNERS = (
    "PH2_W01_TRANSACTION_OWNER",
    "PH2_W02_TRANSACTION_OWNER",
    "PH2_PRIVATE_EVALUATOR",
    "MEMORY",
    "COMPANION",
)
W03_ZERO_EXECUTION_STATE = {
    "LANGUAGE_CAPABILITY_MASTERED": 0,
    "LANGUAGE_READINESS": 0,
    "W03_STARTED": 0,
    "W04_STARTED": 0,
    "formal_w03_training_runs": 0,
    "teacher_calls": 0,
}
W03_RESOURCE_BUDGET = {
    "max_checkpoint_count": 768,
    "max_logic_operations": 3_000_000,
    "max_payload_bytes": 201_326_592,
    "max_payload_gets": 196_608,
    "max_recompute_objects": 300_000,
    "max_records": 300_000,
    "max_segments": 12_288,
    "max_workers": 4,
}


class W03ContractError(RuntimeError):
    """W-03 冻结身份或 payload 前授权失败。"""


def digest_value(value: Any) -> tuple[int, ...]:
    """为无浮点值返回确定性整数 tuple 摘要。"""
    return tuple(hashlib.sha256(canonical_json_bytes(value)).digest())


def strict_key(value: object, *, label: str) -> tuple[int, ...]:
    """要求键是只含严格整数的非空 tuple。"""
    if (not isinstance(value, tuple) or not value
            or any(type(item) is not int for item in value)):
        raise W03ContractError(f"{label} must be a non-empty strict integer tuple")
    return value


def safe_relative_path(value: object, *, label: str) -> str:
    """要求路径是无逃逸的规范 POSIX 相对路径。"""
    if not isinstance(value, str) or not value:
        raise W03ContractError(f"{label} must be a non-empty path")
    path = PurePosixPath(value)
    if (path.is_absolute() or ".." in path.parts or "\\" in value
            or ":" in value
            or path.as_posix() != value or "//" in value):
        raise W03ContractError(f"{label} is not a canonical safe path")
    return value


def _strict_sha256(value: object, *, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(item not in "0123456789abcdef" for item in value)):
        raise W03ContractError(f"{label} is not canonical SHA-256")
    return value


@dataclass(frozen=True, order=True)
class W03PayloadBinding:
    """把一个授权路径绑定到 pack 和完整文件身份。"""

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
            raise W03ContractError("payload pack key is empty")
        if self.owner_kind not in {"source", "observation", "teacher", "evaluator"}:
            raise W03ContractError("payload owner kind is invalid")
        if self.split not in {None, "train", "dev", "held_out"}:
            raise W03ContractError("payload split is invalid")
        for name in ("record_count", "transport_size_bytes", "content_size_bytes"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise W03ContractError(f"payload {name} must be a positive strict integer")
        _strict_sha256(self.transport_sha256, label="payload transport digest")
        _strict_sha256(self.content_sha256, label="payload content digest")
        if not isinstance(self.file_identity, ArtifactFileIdentity):
            raise W03ContractError("payload is missing ArtifactFileIdentity")
        identity = self.file_identity
        if (not self.relative_path.endswith("/" + identity.relative_path)
                or identity.owner_kind != self.owner_kind
                or identity.split != self.split
                or identity.record_count != self.record_count
                or identity.transport_size_bytes != self.transport_size_bytes
                or identity.transport_sha256 != self.transport_sha256
                or identity.content_size_bytes != self.content_size_bytes
                or identity.content_sha256 != self.content_sha256):
            raise W03ContractError("payload identity does not match its D-03 path")

    def to_dict(self) -> dict[str, object]:
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
class W03PackBinding:
    """冻结六个 W-03 train pack manifest 之一。"""

    pack_key: str
    source_key: str
    license_id: str
    earliest_stage: str
    manifest_identity: D03FileIdentity
    total_record_count: int
    source_cluster_count: int

    def __post_init__(self) -> None:
        if self.pack_key not in W03_TRAIN_PACK_KEYS:
            raise W03ContractError("W-03 pack is outside the frozen train set")
        if not self.source_key or not self.license_id:
            raise W03ContractError("W-03 pack source/license is empty")
        if self.earliest_stage not in {"W-02", "W-03"}:
            raise W03ContractError("W-03 train pack earliest stage drifted")
        if not isinstance(self.manifest_identity, D03FileIdentity):
            raise W03ContractError("W-03 pack manifest identity is missing")
        if (type(self.total_record_count) is not int or self.total_record_count <= 0
                or type(self.source_cluster_count) is not int
                or self.source_cluster_count <= 0):
            raise W03ContractError("W-03 pack counts must be positive strict integers")

    def to_dict(self) -> dict[str, object]:
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
class W03PayloadAudit:
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
class W03FrozenContext:
    """汇总打开 W-03 payload 前所需的全部不可变身份。"""

    current_remote_commit_sha1: str
    publication_baseline: Any
    w02_continuity: Any
    d03_receipt_identity: D03FileIdentity
    d03_global_manifest_identity: D03FileIdentity
    stage_manifest_identity: D03FileIdentity
    stage_key: str
    stage_ordinal: int
    prerequisite_stage_keys: tuple[str, ...]
    train_pack_keys: tuple[str, ...]
    pack_bindings: tuple[W03PackBinding, ...]
    candidate_payload_bindings: tuple[W03PayloadBinding, ...]
    teacher_evidence_bindings: tuple[W03PayloadBinding, ...]
    evaluator_visible_bindings: tuple[W03PayloadBinding, ...]
    d03_thresholds: tuple[EvaluationThreshold, ...]
    d03_ablation_keys: tuple[str, ...]
    dimension_key_map: tuple[tuple[str, str], ...]
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
    payload_gets: int = 0
    payload_bytes: int = 0
    learning_writes: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "current_remote_commit_sha1", sha1_text(
            self.current_remote_commit_sha1, where="W-03 current remote commit"))
        if self.stage_key != W03_STAGE_KEY or self.stage_ordinal != 3:
            raise W03ContractError("W-03 stage identity drifted")
        if self.prerequisite_stage_keys != (W03_PREREQUISITE_STAGE_KEY,):
            raise W03ContractError("W-03 prerequisite stage drifted")
        if self.train_pack_keys != W03_TRAIN_PACK_KEYS:
            raise W03ContractError("W-03 train pack order or identity drifted")
        if tuple(item.pack_key for item in self.pack_bindings) != W03_TRAIN_PACK_KEYS:
            raise W03ContractError("W-03 six-pack bindings are incomplete")
        candidate = tuple(item.relative_path for item in self.candidate_payload_bindings)
        teacher = tuple(item.relative_path for item in self.teacher_evidence_bindings)
        evaluator = tuple(item.relative_path for item in self.evaluator_visible_bindings)
        if (len(candidate), len(teacher), len(evaluator)) != (12, 6, 29):
            raise W03ContractError("W-03 12/6/29 file binding counts drifted")
        if any(len(paths) != len(set(paths)) for paths in (candidate, teacher, evaluator)):
            raise W03ContractError("W-03 visible path set contains duplicates")
        if set(candidate) & set(teacher):
            raise W03ContractError("candidate and teacher-only bindings overlap")
        if any((item.owner_kind, item.split) not in {
                ("source", None), ("observation", "train")}
               for item in self.candidate_payload_bindings):
            raise W03ContractError("candidate binding contains non-train/private data")
        if any((item.owner_kind, item.split) != ("teacher", "train")
               for item in self.teacher_evidence_bindings):
            raise W03ContractError("teacher binding contains non-train Evidence")
        if any((item.owner_kind, item.split) not in {
                ("source", None), ("observation", "dev"),
                ("observation", "held_out"), ("evaluator", "dev"),
                ("evaluator", "held_out")}
               for item in self.evaluator_visible_bindings):
            raise W03ContractError("evaluator binding owner/split drifted")
        if (tuple(item.dimension_key for item in self.d03_thresholds)
                != W03_D03_DIMENSION_KEYS
                or self.d03_ablation_keys != W03_D03_ABLATION_KEYS
                or self.dimension_key_map != W03_DIMENSION_KEY_MAP
                or self.dimension_keys != W03_DIMENSION_KEYS
                or self.ablation_keys != W03_ABLATION_KEYS):
            raise W03ContractError("W-03 frozen bearing or ablation identity drifted")
        if any(
                item.bearing != 1
                or item.min_pass_numerator != 1
                or item.min_pass_denominator != 1
                or item.max_fail_count != 0
                or item.ne_policy != "BLOCK"
                or item.preregistered != 1
                for item in self.d03_thresholds):
            raise W03ContractError("W-03 D-03 thresholds were relaxed")
        if (self.generation_hard_conjunct != W03_GENERATION_HARD_CONJUNCT
                or self.evaluation_order != W03_EVALUATION_ORDER
                or self.aggregation_policy != W03_AGGREGATION_POLICY):
            raise W03ContractError("W-03 generation or total hard conjunct drifted")
        if (self.allowed_worker_counts != W03_ALLOWED_WORKER_COUNTS
                or self.logical_shard_count != 16
                or len(self.failure_point_keys) != 6
                or self.resource_budget != W03_RESOURCE_BUDGET):
            raise W03ContractError("W-03 recovery or resource contract drifted")
        if (self.run_id != W03_FORMAL_RUN_ID
                or self.parent_run_id != W03_W02_BASE_RUN_ID
                or self.base_run_id != W03_W02_BASE_RUN_ID):
            raise W03ContractError("W-03 run/parent/base identity drifted")
        strict_key(self.backend_profile_key, label="backend profile key")
        strict_key(self.base_fence_key, label="base fence key")
        if (self.owner_key != W03_OWNER_KEY
                or self.allowed_write_owners != W03_ALLOWED_WRITE_OWNERS
                or self.forbidden_write_owners != W03_FORBIDDEN_WRITE_OWNERS
                or set(self.allowed_write_owners) & set(self.forbidden_write_owners)):
            raise W03ContractError("W-03 transaction/write owner boundary drifted")
        if (self.logical_clock_version != W03_LOGICAL_CLOCK_VERSION
                or self.execution_state != W03_ZERO_EXECUTION_STATE):
            raise W03ContractError("W-03 pre-training state is not all zero")
        if self.payload_gets != 0 or self.payload_bytes != 0 or self.learning_writes != 0:
            raise W03ContractError("W-03 context construction touched payload or learning")

    def stable_key(self) -> tuple[int, ...]:
        """绑定 publication、continuity、D-03、payload、恢复和状态。"""
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
            "dimension_key_map": [list(item) for item in self.dimension_key_map],
            "evaluation_order": list(self.evaluation_order),
            "evaluator_visible_bindings": [
                item.to_dict() for item in self.evaluator_visible_bindings],
            "execution_state": dict(self.execution_state),
            "failure_point_keys": list(self.failure_point_keys),
            "forbidden_write_owners": list(self.forbidden_write_owners),
            "logical_clock_version": self.logical_clock_version,
            "logical_shard_count": self.logical_shard_count,
            "merge_barrier_key": self.merge_barrier_key,
            "owner_key": self.owner_key,
            "pack_bindings": [item.to_dict() for item in self.pack_bindings],
            "parent_run_id": self.parent_run_id,
            "publication_baseline_key": list(self.publication_baseline.stable_key()),
            "resource_budget": dict(self.resource_budget),
            "run_id": self.run_id,
            "stage_manifest_identity": self.stage_manifest_identity.to_dict(),
            "teacher_evidence_bindings": [
                item.to_dict() for item in self.teacher_evidence_bindings],
            "version_keys": [list(item) for item in self.version_keys],
            "w02_continuity_key": list(self.w02_continuity.stable_key()),
        })


@dataclass(frozen=True)
class W03RunRequest:
    """不含 evaluator 或未来阶段字段的完整 candidate 请求。"""

    run_id: int
    parent_run_id: int
    base_run_id: int
    stage_key: str
    owner_key: str
    runner_key: str
    publication_baseline_key: tuple[int, ...]
    current_remote_commit_sha1: str
    w02_continuity_key: tuple[int, ...]
    d03_context_key: tuple[int, ...]
    backend_profile_key: tuple[int, ...]
    base_fence_key: tuple[int, ...]
    worker_count: int
    mode: str
    resource_budget: tuple[tuple[str, int], ...]
    candidate_payload_paths: tuple[str, ...]
    teacher_evidence_paths: tuple[str, ...]

    def execution_identity_key(self) -> tuple[int, ...]:
        return digest_value({
            "backend_profile_key": list(self.backend_profile_key),
            "base_fence_key": list(self.base_fence_key),
            "base_run_id": self.base_run_id,
            "candidate_payload_paths": list(self.candidate_payload_paths),
            "current_remote_commit_sha1": self.current_remote_commit_sha1,
            "d03_context_key": list(self.d03_context_key),
            "owner_key": self.owner_key,
            "parent_run_id": self.parent_run_id,
            "publication_baseline_key": list(self.publication_baseline_key),
            "resource_budget": dict(self.resource_budget),
            "run_id": self.run_id,
            "runner_key": self.runner_key,
            "stage_key": self.stage_key,
            "teacher_evidence_paths": list(self.teacher_evidence_paths),
            "w02_continuity_key": list(self.w02_continuity_key),
        })

    def scheduling_key(self) -> tuple[int, ...]:
        return digest_value({"mode": self.mode, "worker_count": self.worker_count})


def validate_w03_request(
        context: W03FrozenContext,
        request: W03RunRequest,
        ) -> W03RunRequest:
    """在任何 payload transport 前拒绝全部身份和路径漂移。"""
    if not isinstance(context, W03FrozenContext):
        raise W03ContractError("W-03 context type is invalid")
    if not isinstance(request, W03RunRequest):
        raise W03ContractError("W-03 request type is invalid")
    if (type(request.run_id) is not int
            or request.run_id != context.run_id
            or type(request.parent_run_id) is not int
            or request.parent_run_id != context.parent_run_id
            or type(request.base_run_id) is not int
            or request.base_run_id != context.base_run_id):
        raise W03ContractError("W-03 run/parent/base id did not continue W-02 run 3")
    if request.stage_key != W03_STAGE_KEY or request.stage_key != context.stage_key:
        raise W03ContractError("W-03 is the only authorized stage")
    if request.owner_key != context.owner_key or request.owner_key != W03_OWNER_KEY:
        raise W03ContractError("W-03 transaction owner is unauthorized")
    if request.runner_key != W03_RUNNER_KEY:
        raise W03ContractError("W-03 requires the independent stage-2 runner")
    if request.publication_baseline_key != context.publication_baseline.stable_key():
        raise W03ContractError("W-03 publication baseline identity drifted")
    if request.current_remote_commit_sha1 != context.current_remote_commit_sha1:
        raise W03ContractError("W-03 current remote commit drifted")
    if request.w02_continuity_key != context.w02_continuity.stable_key():
        raise W03ContractError("W-02 continuity or retention identity drifted")
    if request.d03_context_key != context.stable_key():
        raise W03ContractError("D-03/W-03 frozen context identity drifted")
    if request.backend_profile_key != context.backend_profile_key:
        raise W03ContractError("backend profile identity drifted")
    if request.base_fence_key != context.base_fence_key:
        raise W03ContractError("W-02 base fence identity drifted")
    if request.worker_count not in context.allowed_worker_counts:
        raise W03ContractError("worker count is outside 1/2/4")
    if request.mode not in W03_ALLOWED_MODES:
        raise W03ContractError("mode must be fresh/restart/resume")
    expected_budget = tuple(sorted(context.resource_budget.items()))
    if request.resource_budget != expected_budget:
        raise W03ContractError("W-03 resource budget identity drifted")
    expected_candidate = tuple(
        item.relative_path for item in context.candidate_payload_bindings)
    expected_teacher = tuple(
        item.relative_path for item in context.teacher_evidence_bindings)
    for path in (*request.candidate_payload_paths, *request.teacher_evidence_paths):
        safe_relative_path(path, label="request payload path")
    if request.candidate_payload_paths != expected_candidate:
        raise W03ContractError("candidate payload paths are not the exact train whitelist")
    if request.teacher_evidence_paths != expected_teacher:
        raise W03ContractError("teacher Evidence paths are not the exact train whitelist")
    evaluator_private = {
        item.relative_path for item in context.evaluator_visible_bindings
        if item.owner_kind != "source"
    }
    requested = set(request.candidate_payload_paths) | set(request.teacher_evidence_paths)
    if evaluator_private & requested:
        raise W03ContractError("candidate request contains evaluator/private data")
    return request


__all__ = [
    "W03_ABLATION_KEYS",
    "W03_AGGREGATION_POLICY",
    "W03_ALLOWED_MODES",
    "W03_ALLOWED_WORKER_COUNTS",
    "W03_DIMENSION_KEYS",
    "W03_EVALUATION_ORDER",
    "W03_FORMAL_RUN_ID",
    "W03_GENERATION_HARD_CONJUNCT",
    "W03_OWNER_KEY",
    "W03_RESOURCE_BUDGET",
    "W03_RUNNER_KEY",
    "W03_STAGE_KEY",
    "W03_TRAIN_PACK_KEYS",
    "W03_W02_BASE_RUN_ID",
    "W03ContractError",
    "W03FrozenContext",
    "W03PackBinding",
    "W03PayloadAudit",
    "W03PayloadBinding",
    "W03RunRequest",
    "digest_value",
    "safe_relative_path",
    "strict_key",
    "validate_w03_request",
]
