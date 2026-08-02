"""W-06 public contract、有效 pack 替换、运行身份与零训练状态。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path, PurePosixPath
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import (
    ArtifactFileIdentity,
    DatasetContractError,
    canonical_json_bytes,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_dataset_io import read_artifact_manifest
from pure_integer_ai.experiments.ph2_w05_contract import digest_value
from pure_integer_ai.experiments.ph2_w06_source_semantic import (
    W06_GENERATION_HARD_CONJUNCT,
    W06_RELATION_SUBSTAGE_ORDER,
)
from pure_integer_ai.experiments.ph2_w06_source_semantic_overlay import (
    W06_SOURCE_OVERLAY_PATH,
    canonical_w06_source_semantic_overlay_bytes,
    read_w06_source_semantic_overlay,
)


W06_FORMAT_VERSION = 1
W06_STAGE_KEY = "W-06"
W06_PREREQUISITE_STAGE_KEY = "W-05"
W06_OWNER_KEY = "PH2_W06_TRANSACTION_OWNER"
W06_RUNNER_KEY = "PH2_LANGUAGE_STAGE6_TYPED_RELATIONS"
W06_FORMAL_RUN_ID = 7
W06_W05_BASE_RUN_ID = 6
W06_ALLOWED_MODES = ("fresh", "restart", "resume")
W06_ALLOWED_WORKER_COUNTS = (1, 2, 4)
W06_LOGICAL_CLOCK_VERSION = "PH2-W06-LOGICAL-CLOCK-V1"
W06_CURSOR_VERSION = "PH2-D03-CURSOR-V1"
W06_MERGE_BARRIER_KEY = "PH2-D03-STABLE-MERGE-BARRIER-V1"
W06_FAILURE_POINT_KEYS = (
    "BEFORE_FIRST_SHARD",
    "AFTER_PARTIAL_SHARD",
    "BEFORE_MERGE_PREVIEW",
    "AFTER_MERGE_BEFORE_COMMIT",
    "AFTER_COMMIT_BEFORE_CURSOR",
    "AFTER_MANIFEST_PUBLISH",
)
W06_OPEN_GENERATION_STATE = "NE_NOT_YET_EVALUABLE"
W06_CANDIDATE_OWNER = "PH2_TRAIN_CANDIDATE"
W06_TEACHER_OWNER = "PH2_TRAINING_EVIDENCE"
W06_EVALUATOR_OWNER = "PH2_PRIVATE_EVALUATOR"
W06_ALLOWED_WRITE_OWNERS = (W06_OWNER_KEY,)
W06_FORBIDDEN_WRITE_OWNERS = (
    "PH2_W01_TRANSACTION_OWNER",
    "PH2_W02_TRANSACTION_OWNER",
    "PH2_W03_TRANSACTION_OWNER",
    "PH2_W04_TRANSACTION_OWNER",
    "PH2_W05_TRANSACTION_OWNER",
    W06_EVALUATOR_OWNER,
    "MEMORY",
    "COMPANION",
)
W06_STAGE_MANIFEST_PATH = (
    "data/ph2/manifests/d03_v1/stages/w06_stage_manifest_v1.json"
)
W06_GLOBAL_MANIFEST_PATH = (
    "data/ph2/manifests/d03_v1/ph2_global_course_manifest_v1.json"
)
W06_INVALIDATION_GRAPH_PATH = (
    "data/ph2/manifests/d03_v1/stage_invalidation_graph_v1.json"
)
W06_W05_RECEIPT_PATH = (
    "data/ph2/manifests/d03_v1/w05_runtime_evidence_receipt_v1.json"
)
W06_V2_PACK_KEY = "AUTHORED_CC0_V1--CC0-1.0--alias-refers-w06-v2"
W06_V1_PACK_KEY = "AUTHORED_CC0_V1--CC0-1.0--alias-refers-v1"
W06_V2_PACK_PREFIX = (
    "ph2_w06_dataset_artifacts/source_semantic_overlay_v1/packs/"
    + W06_V2_PACK_KEY
)
W06_EXPECTED_SHA256 = {
    W06_STAGE_MANIFEST_PATH: "a9beda13955e4708b5f2bb7f4d2b106be1bdf709c82acaefcfa95ca7d276e00a",
    W06_GLOBAL_MANIFEST_PATH: "384329cf651ea4c5e4bc9d0b5dc4da7b22a71bc008bfabe468c86278dd9d40b6",
    W06_INVALIDATION_GRAPH_PATH: "21cf4d3cd65afeb0f93054773b97fa4194ee5f14dc463ff40af5813fdb0facce",
    W06_W05_RECEIPT_PATH: "64c2fff496e766df880d2db1b184e2b8a009abd3b37b1a1b1331900458ccff78",
    W06_SOURCE_OVERLAY_PATH: "f5cae297254191dffb5bcacdafbdc461dcd1cf3a1340de27d9a8c98c598bfbbc",
}
W06_RESOURCE_BUDGET = {
    "max_checkpoint_count": 1536,
    "max_logic_operations": 6000000,
    "max_payload_bytes": 402653184,
    "max_payload_gets": 393216,
    "max_recompute_objects": 600000,
    "max_records": 600000,
    "max_segments": 24576,
    "max_workers": 4,
}
W06_DIMENSION_KEYS = (
    "W-06-CAUSES",
    "W-06-MEREOLOGY",
    "W-06-PRECEDES",
    "W-06-PROPERTY",
    "W-06-PURE_ALIAS_REFERS",
    "W-06-SIMILAR_ANTONYM",
    "W-06-SUBSET_MEMBER",
)
W06_ABLATION_KEYS = tuple(f"{item}-ABLATION" for item in W06_DIMENSION_KEYS)
W06_GENERATION_ABLATION_KEY = f"{W06_GENERATION_HARD_CONJUNCT}-ABLATION"
W06_PRIVATE_ABLATION_KEYS = (*W06_ABLATION_KEYS, W06_GENERATION_ABLATION_KEY)
W06_EVALUATION_ORDER = (*W06_DIMENSION_KEYS, W06_GENERATION_HARD_CONJUNCT)
W06_AGGREGATION_POLICY = "ALL_7_W06_BEARINGS_AND_GENERATION_MUST_PASS"
W06_ZERO_EXECUTION_STATE = {
    "LANGUAGE_CAPABILITY_MASTERED": 0,
    "LANGUAGE_READINESS": 0,
    "W06_STARTED": 0,
    "W07_STARTED": 0,
    "formal_w06_training_runs": 0,
    "teacher_calls": 0,
}


class W06ContractError(RuntimeError):
    """W-06 parent、可见性、payload、运行或资源身份发生漂移。"""


def _sha256(path: Path) -> str:
    """流式计算文件 SHA-256。"""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while True:
                block = handle.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
    except OSError as error:
        raise W06ContractError(f"无法读取 W-06 依赖：{path}") from error
    return digest.hexdigest()


def _canonical_object(path: Path) -> dict[str, Any]:
    """读取可带单个末尾换行的规范 JSON object。"""
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise W06ContractError(f"无法读取 W-06 JSON：{path}") from error
    body = payload[:-1] if payload.endswith(b"\n") else payload
    try:
        value = parse_canonical_json_bytes(body, require_object=True)
    except DatasetContractError as error:
        raise W06ContractError(f"W-06 JSON 非规范：{path}") from error
    assert isinstance(value, dict)
    return value


def _safe_relative(value: str) -> str:
    """拒绝绝对路径、反斜杠、点段和空路径。"""
    if not isinstance(value, str) or not value or "\\" in value:
        raise W06ContractError("W-06 payload path 非规范")
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(item in {"", ".", ".."} for item in pure.parts):
        raise W06ContractError("W-06 payload path 越界")
    return pure.as_posix()


@dataclass(frozen=True, order=True)
class W06PayloadBinding:
    """绑定一个有效 pack 文件的路径、owner/split 和完整内容身份。"""

    relative_path: str
    pack_key: str
    owner_kind: str
    split: str | None
    file_identity: ArtifactFileIdentity

    def __post_init__(self) -> None:
        object.__setattr__(self, "relative_path", _safe_relative(self.relative_path))
        if not self.pack_key:
            raise W06ContractError("W-06 payload pack key 为空")
        if self.owner_kind not in {"source", "observation", "teacher", "evaluator"}:
            raise W06ContractError("W-06 payload owner 非法")
        if self.split not in {None, "train", "dev", "held_out"}:
            raise W06ContractError("W-06 payload split 非法")
        if not isinstance(self.file_identity, ArtifactFileIdentity):
            raise W06ContractError("W-06 payload 缺少 ArtifactFileIdentity")
        identity = self.file_identity
        if (not self.relative_path.endswith("/" + identity.relative_path)
                or identity.owner_kind != self.owner_kind
                or identity.split != self.split):
            raise W06ContractError("W-06 payload path 与文件身份不一致")

    def to_dict(self) -> dict[str, Any]:
        """导出稳定 payload binding。"""
        return {
            "file_identity": self.file_identity.to_dict(),
            "owner_kind": self.owner_kind,
            "pack_key": self.pack_key,
            "relative_path": self.relative_path,
            "split": self.split,
        }


@dataclass(frozen=True, order=True)
class W06PackBinding:
    """记录 stage pack 与 overlay 有效 pack 的 manifest 身份。"""

    stage_pack_key: str
    effective_pack_key: str
    source_key: str
    license_id: str
    earliest_stage: str
    manifest_relative_path: str
    manifest_sha256: str
    total_record_count: int
    source_cluster_count: int
    overlay_replacement: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "manifest_relative_path", _safe_relative(
            self.manifest_relative_path))
        if self.overlay_replacement not in {0, 1}:
            raise W06ContractError("W-06 overlay replacement 必须是 0/1")
        if self.overlay_replacement == 1 and (
                self.stage_pack_key != W06_V1_PACK_KEY
                or self.effective_pack_key != W06_V2_PACK_KEY):
            raise W06ContractError("W-06 alias/refers overlay 替换坐标漂移")
        if type(self.total_record_count) is not int or self.total_record_count <= 0:
            raise W06ContractError("W-06 pack record count 非法")
        if type(self.source_cluster_count) is not int or self.source_cluster_count <= 0:
            raise W06ContractError("W-06 pack cluster count 非法")
        if len(self.manifest_sha256) != 64:
            raise W06ContractError("W-06 pack manifest SHA 非法")

    def to_dict(self) -> dict[str, Any]:
        """导出稳定 pack binding。"""
        return {
            "earliest_stage": self.earliest_stage,
            "effective_pack_key": self.effective_pack_key,
            "license_id": self.license_id,
            "manifest_relative_path": self.manifest_relative_path,
            "manifest_sha256": self.manifest_sha256,
            "overlay_replacement": self.overlay_replacement,
            "source_cluster_count": self.source_cluster_count,
            "source_key": self.source_key,
            "stage_pack_key": self.stage_pack_key,
            "total_record_count": self.total_record_count,
        }


@dataclass
class W06PayloadAudit:
    """记录 payload transport、交付和零 teacher/learning 写。"""

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
class W06FrozenContext:
    """汇总任何 W-06 payload 读取前必须闭合的公开身份。"""

    current_remote_commit_sha1: str
    parent_sha256: tuple[tuple[str, str], ...]
    source_overlay_sha256: str
    stage_train_pack_keys: tuple[str, ...]
    effective_train_pack_keys: tuple[str, ...]
    pack_bindings: tuple[W06PackBinding, ...]
    candidate_payload_bindings: tuple[W06PayloadBinding, ...]
    teacher_evidence_bindings: tuple[W06PayloadBinding, ...]
    evaluator_visible_bindings: tuple[W06PayloadBinding, ...]
    relation_substage_order: tuple[str, ...]
    dimension_keys: tuple[str, ...]
    ablation_keys: tuple[str, ...]
    private_ablation_keys: tuple[str, ...]
    generation_hard_conjunct: str
    evaluation_order: tuple[str, ...]
    aggregation_policy: str
    allowed_worker_counts: tuple[int, ...]
    failure_point_keys: tuple[str, ...]
    logical_shard_count: int
    merge_barrier_key: str
    cursor_version: str
    logical_clock_version: str
    resource_budget: tuple[tuple[str, int], ...]
    run_id: int
    parent_run_id: int
    base_run_id: int
    backend_profile_key: tuple[int, ...]
    base_fence_key: tuple[int, ...]
    owner_key: str
    candidate_owner: str
    teacher_owner: str
    evaluator_owner: str
    allowed_write_owners: tuple[str, ...]
    forbidden_write_owners: tuple[str, ...]
    runner_key: str
    execution_state: tuple[tuple[str, int], ...]
    open_generation_state: str
    payload_gets: int = 0
    payload_bytes: int = 0
    learning_writes: int = 0

    def __post_init__(self) -> None:
        if (len(self.current_remote_commit_sha1) != 40
                or any(item not in "0123456789abcdef"
                       for item in self.current_remote_commit_sha1)):
            raise W06ContractError("W-06 remote commit SHA1 非法")
        if self.source_overlay_sha256 != W06_EXPECTED_SHA256[W06_SOURCE_OVERLAY_PATH]:
            raise W06ContractError("W-06 source overlay SHA 漂移")
        if len(self.stage_train_pack_keys) != 18:
            raise W06ContractError("W-06 stage train pack 数量漂移")
        if len(self.effective_train_pack_keys) != 18:
            raise W06ContractError("W-06 effective train pack 数量漂移")
        if (self.stage_train_pack_keys[0] != W06_V1_PACK_KEY
                or W06_V1_PACK_KEY in self.effective_train_pack_keys
                or sum(item.overlay_replacement for item in self.pack_bindings) != 1):
            raise W06ContractError("W-06 v1 历史坐标或 v2 单次替换漂移")
        if tuple(item.stage_pack_key for item in self.pack_bindings) != (
                self.stage_train_pack_keys):
            raise W06ContractError("W-06 pack binding 顺序漂移")
        if tuple(item.effective_pack_key for item in self.pack_bindings) != (
                self.effective_train_pack_keys):
            raise W06ContractError("W-06 effective pack binding 不完整")
        if self.effective_train_pack_keys[0] != W06_V2_PACK_KEY:
            raise W06ContractError("W-06 stable alias/refers v2 未替换首包")
        candidate_paths = tuple(
            item.relative_path for item in self.candidate_payload_bindings)
        teacher_paths = tuple(
            item.relative_path for item in self.teacher_evidence_bindings)
        evaluator_paths = tuple(
            item.relative_path for item in self.evaluator_visible_bindings)
        if any(len(paths) != len(set(paths))
               for paths in (candidate_paths, teacher_paths, evaluator_paths)):
            raise W06ContractError("W-06 可见 payload path 重复")
        if set(candidate_paths) & set(teacher_paths):
            raise W06ContractError("W-06 candidate 与 teacher payload 重叠")
        if W06_V1_PACK_KEY in {
                item.pack_key for item in (
                    *self.candidate_payload_bindings,
                    *self.teacher_evidence_bindings,
                    *self.evaluator_visible_bindings,
                )
        }:
            raise W06ContractError("W-06 candidate 仍可见旧 occurrence REFERS pack")
        if any((item.owner_kind, item.split) not in {
                ("source", None), ("observation", "train")}
               for item in self.candidate_payload_bindings):
            raise W06ContractError("W-06 candidate whitelist 含非 train payload")
        if any((item.owner_kind, item.split) != ("teacher", "train")
               for item in self.teacher_evidence_bindings):
            raise W06ContractError("W-06 teacher whitelist 含非 train Evidence")
        if any((item.owner_kind, item.split) not in {
                ("observation", "dev"), ("observation", "held_out"),
                ("evaluator", "dev"), ("evaluator", "held_out")}
               for item in self.evaluator_visible_bindings):
            raise W06ContractError("W-06 evaluator 可见性含非法 owner/split")
        if (self.relation_substage_order != W06_RELATION_SUBSTAGE_ORDER
                or self.dimension_keys != W06_DIMENSION_KEYS
                or self.ablation_keys != W06_ABLATION_KEYS
                or self.private_ablation_keys != W06_PRIVATE_ABLATION_KEYS
                or self.generation_hard_conjunct != W06_GENERATION_HARD_CONJUNCT
                or self.evaluation_order != W06_EVALUATION_ORDER
                or self.aggregation_policy != W06_AGGREGATION_POLICY):
            raise W06ContractError("W-06 relation/evaluation 合同漂移")
        if (self.allowed_worker_counts != W06_ALLOWED_WORKER_COUNTS
                or self.logical_shard_count != 16
                or self.failure_point_keys != W06_FAILURE_POINT_KEYS
                or self.merge_barrier_key != W06_MERGE_BARRIER_KEY
                or self.cursor_version != W06_CURSOR_VERSION
                or self.logical_clock_version != W06_LOGICAL_CLOCK_VERSION
                or dict(self.resource_budget) != W06_RESOURCE_BUDGET):
            raise W06ContractError("W-06 recovery/resource 合同漂移")
        if (self.run_id != W06_FORMAL_RUN_ID
                or self.parent_run_id != W06_W05_BASE_RUN_ID
                or self.base_run_id != W06_W05_BASE_RUN_ID):
            raise W06ContractError("W-06 run/base 身份漂移")
        if (not self.backend_profile_key
                or not self.base_fence_key
                or any(type(item) is not int for item in (
                    *self.backend_profile_key, *self.base_fence_key))):
            raise W06ContractError("W-06 backend/base fence key 非法")
        if (self.owner_key != W06_OWNER_KEY
                or self.candidate_owner != W06_CANDIDATE_OWNER
                or self.teacher_owner != W06_TEACHER_OWNER
                or self.evaluator_owner != W06_EVALUATOR_OWNER
                or self.allowed_write_owners != W06_ALLOWED_WRITE_OWNERS
                or self.forbidden_write_owners != W06_FORBIDDEN_WRITE_OWNERS
                or set(self.allowed_write_owners) & set(self.forbidden_write_owners)
                or self.runner_key != W06_RUNNER_KEY
                or dict(self.execution_state) != W06_ZERO_EXECUTION_STATE
                or self.open_generation_state != W06_OPEN_GENERATION_STATE):
            raise W06ContractError("W-06 pre-training 状态漂移")
        if self.payload_gets or self.payload_bytes or self.learning_writes:
            raise W06ContractError("W-06 context 构建触碰了 payload/learning")

    def stable_key(self) -> tuple[int, ...]:
        """返回绑定 parent、pack、payload、评测、恢复和状态的完整键。"""
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
            "candidate_owner": self.candidate_owner,
            "current_remote_commit_sha1": self.current_remote_commit_sha1,
            "cursor_version": self.cursor_version,
            "dimension_keys": list(self.dimension_keys),
            "effective_train_pack_keys": list(self.effective_train_pack_keys),
            "evaluation_order": list(self.evaluation_order),
            "execution_state": dict(self.execution_state),
            "failure_point_keys": list(self.failure_point_keys),
            "forbidden_write_owners": list(self.forbidden_write_owners),
            "generation_hard_conjunct": self.generation_hard_conjunct,
            "logical_clock_version": self.logical_clock_version,
            "logical_shard_count": self.logical_shard_count,
            "merge_barrier_key": self.merge_barrier_key,
            "open_generation_state": self.open_generation_state,
            "owner_key": self.owner_key,
            "pack_bindings": [item.to_dict() for item in self.pack_bindings],
            "parent_run_id": self.parent_run_id,
            "parent_sha256": [list(item) for item in self.parent_sha256],
            "private_ablation_keys": list(self.private_ablation_keys),
            "relation_substage_order": list(self.relation_substage_order),
            "resource_budget": dict(self.resource_budget),
            "run_id": self.run_id,
            "runner_key": self.runner_key,
            "source_overlay_sha256": self.source_overlay_sha256,
            "stage_train_pack_keys": list(self.stage_train_pack_keys),
            "teacher_owner": self.teacher_owner,
            "evaluator_owner": self.evaluator_owner,
            "teacher_evidence_bindings": [
                item.to_dict() for item in self.teacher_evidence_bindings],
        })


@dataclass(frozen=True)
class W06RunRequest:
    """声明不含 evaluator/private 字段的唯一 W-06 candidate 请求。"""

    run_id: int
    parent_run_id: int
    base_run_id: int
    stage_key: str
    owner_key: str
    runner_key: str
    current_remote_commit_sha1: str
    source_overlay_sha256: str
    context_key: tuple[int, ...]
    backend_profile_key: tuple[int, ...]
    base_fence_key: tuple[int, ...]
    worker_count: int
    mode: str
    resource_budget: tuple[tuple[str, int], ...]
    candidate_payload_paths: tuple[str, ...]
    teacher_evidence_paths: tuple[str, ...]

    def execution_identity_key(self) -> tuple[int, ...]:
        """返回排除 worker/mode 的正式执行身份。"""
        return digest_value({
            "base_fence_key": list(self.base_fence_key),
            "base_run_id": self.base_run_id,
            "candidate_payload_paths": list(self.candidate_payload_paths),
            "context_key": list(self.context_key),
            "current_remote_commit_sha1": self.current_remote_commit_sha1,
            "owner_key": self.owner_key,
            "parent_run_id": self.parent_run_id,
            "resource_budget": dict(self.resource_budget),
            "run_id": self.run_id,
            "runner_key": self.runner_key,
            "source_overlay_sha256": self.source_overlay_sha256,
            "stage_key": self.stage_key,
            "teacher_evidence_paths": list(self.teacher_evidence_paths),
        })

    def scheduling_key(self) -> tuple[int, ...]:
        """返回只包含 worker 与 fresh/restart/resume 的调度身份。"""
        return digest_value({"mode": self.mode, "worker_count": self.worker_count})


def validate_w06_request(
        context: W06FrozenContext,
        request: W06RunRequest,
        ) -> W06RunRequest:
    """在任何 transport 前拒绝 stage、owner、资源和路径漂移。"""
    if not isinstance(context, W06FrozenContext) or not isinstance(request, W06RunRequest):
        raise W06ContractError("W-06 context/request 类型非法")
    if (request.run_id, request.parent_run_id, request.base_run_id) != (
            context.run_id, context.parent_run_id, context.base_run_id):
        raise W06ContractError("W-06 run/parent/base id 漂移")
    if (request.stage_key != W06_STAGE_KEY
            or request.owner_key != context.owner_key
            or request.runner_key != context.runner_key):
        raise W06ContractError("W-06 stage/owner/runner 未授权")
    if request.current_remote_commit_sha1 != context.current_remote_commit_sha1:
        raise W06ContractError("W-06 remote commit 漂移")
    if request.source_overlay_sha256 != context.source_overlay_sha256:
        raise W06ContractError("W-06 source overlay 请求漂移")
    if (request.context_key != context.stable_key()
            or request.backend_profile_key != context.backend_profile_key
            or request.base_fence_key != context.base_fence_key):
        raise W06ContractError("W-06 context/backend/base fence 漂移")
    if request.worker_count not in context.allowed_worker_counts:
        raise W06ContractError("W-06 worker count 非法")
    if request.mode not in W06_ALLOWED_MODES:
        raise W06ContractError("W-06 mode 非法")
    if request.resource_budget != tuple(sorted(W06_RESOURCE_BUDGET.items())):
        raise W06ContractError("W-06 resource budget 漂移")
    candidate = tuple(item.relative_path for item in context.candidate_payload_bindings)
    teacher = tuple(item.relative_path for item in context.teacher_evidence_bindings)
    if request.candidate_payload_paths != candidate:
        raise W06ContractError("W-06 request 非精确 train whitelist")
    if request.teacher_evidence_paths != teacher:
        raise W06ContractError("W-06 request 非精确 teacher whitelist")
    return request


def _binding(prefix: str, pack_key: str, identity: ArtifactFileIdentity) -> W06PayloadBinding:
    """把 pack 内文件身份提升为 repository-relative binding。"""
    return W06PayloadBinding(
        f"{prefix}/{identity.relative_path}",
        pack_key,
        identity.owner_kind,
        identity.split,
        identity,
    )


def open_w06_frozen_context(
        repository_root: str | Path,
        *,
        current_remote_commit_sha1: str,
        backend_profile_key: tuple[int, ...],
        ) -> W06FrozenContext:
    """只读打开 stage/global/overlay 和 18 个有效 train pack 身份。"""
    root = Path(repository_root).resolve()
    parent_sha = []
    for relative, expected in W06_EXPECTED_SHA256.items():
        actual = _sha256(root / relative)
        if actual != expected:
            raise W06ContractError(f"W-06 parent SHA 漂移：{relative}")
        parent_sha.append((relative, actual))
    overlay_path = root / W06_SOURCE_OVERLAY_PATH
    if overlay_path.read_bytes() != canonical_w06_source_semantic_overlay_bytes(root):
        raise W06ContractError("W-06 source overlay stored/rebuild 不一致")
    overlay = read_w06_source_semantic_overlay(overlay_path)
    stage = _canonical_object(root / W06_STAGE_MANIFEST_PATH)
    global_manifest = _canonical_object(root / W06_GLOBAL_MANIFEST_PATH)
    identity = stage.get("stage_identity", {})
    visibility = stage.get("data_visibility", {})
    recovery = stage.get("recovery_binding", {})
    evaluation = stage.get("evaluation_binding", {})
    stage_keys = tuple(visibility.get("train_pack_keys", ()))
    if (identity.get("stage_key") != W06_STAGE_KEY
            or identity.get("ordinal") != 6
            or tuple(identity.get("prerequisite_stage_keys", ()))
            != (W06_PREREQUISITE_STAGE_KEY,)
            or tuple(identity.get("substage_keys", ()))
            != W06_RELATION_SUBSTAGE_ORDER):
        raise W06ContractError("W-06 stage identity 漂移")
    thresholds = tuple(evaluation.get("thresholds", ()))
    if (tuple(item.get("dimension_key") for item in thresholds) != W06_DIMENSION_KEYS
            or tuple(evaluation.get("ablation_keys", ())) != W06_ABLATION_KEYS
            or any(
                item.get("bearing") != 1
                or item.get("min_pass_numerator") != 1
                or item.get("min_pass_denominator") != 1
                or item.get("max_fail_count") != 0
                or item.get("ne_policy") != "BLOCK"
                or item.get("preregistered") != 1
                for item in thresholds
            )):
        raise W06ContractError("W-06 bearing/threshold 被放宽")
    if (tuple(recovery.get("allowed_worker_counts", ())) != W06_ALLOWED_WORKER_COUNTS
            or recovery.get("logical_shard_count") != 16
            or tuple(recovery.get("failure_point_keys", ())) != W06_FAILURE_POINT_KEYS
            or recovery.get("merge_barrier_key") != W06_MERGE_BARRIER_KEY
            or recovery.get("cursor_version") != W06_CURSOR_VERSION
            or stage.get("resource_budget") != W06_RESOURCE_BUDGET):
        raise W06ContractError("W-06 recovery/resource manifest 漂移")
    if (visibility.get("candidate_owner") != W06_CANDIDATE_OWNER
            or visibility.get("teacher_owner") != W06_TEACHER_OWNER
            or visibility.get("evaluator_owner") != W06_EVALUATOR_OWNER
            or tuple(visibility.get("candidate_allowed_splits", ())) != ("train",)
            or tuple(visibility.get("candidate_forbidden_splits", ()))
            != ("dev", "held_out", "adversarial", "wall")):
        raise W06ContractError("W-06 manifest owner/split 防火墙漂移")

    global_by_key = {
        item["pack_key"]: item for item in global_manifest.get("pack_bindings", ())
    }
    pack_bindings = []
    candidate = []
    teacher = []
    evaluator = []
    for stage_key in stage_keys:
        replacement = stage_key == W06_V1_PACK_KEY
        effective_key = W06_V2_PACK_KEY if replacement else stage_key
        if replacement:
            prefix = W06_V2_PACK_PREFIX
            manifest_path = root / prefix / "manifest.json"
            manifest = read_artifact_manifest(manifest_path)
            canonical_sha = hashlib.sha256(
                canonical_json_bytes(manifest.to_dict())).hexdigest()
            course = overlay["stable_v2_course"]
            if (canonical_sha != course["pack_manifest_sha256"]
                    or manifest.source_key != course["source_key"]
                    or manifest.record_count != 27
                    or len(manifest.source_cluster_keys) != 2):
                raise W06ContractError("W-06 v2 pack 未匹配 source overlay")
            earliest = "W-06"
            manifest_sha = _sha256(manifest_path)
            source_key = manifest.source_key
            license_id = manifest.license_partition
            total = manifest.record_count
            clusters = len(manifest.source_cluster_keys)
        else:
            frozen = global_by_key.get(stage_key)
            if frozen is None:
                raise W06ContractError("W-06 stage train pack 未进入 D-03 global")
            prefix = str(Path(frozen["manifest_identity"]["relative_path"]).parent).replace(
                "\\", "/")
            manifest_path = root / frozen["manifest_identity"]["relative_path"]
            manifest_sha = _sha256(manifest_path)
            if manifest_sha != frozen["manifest_identity"]["sha256"]:
                raise W06ContractError("W-06 D-03 pack manifest SHA 漂移")
            manifest = read_artifact_manifest(manifest_path)
            earliest = frozen["earliest_stage"]
            source_key = frozen["source_key"]
            license_id = frozen["license_id"]
            total = frozen["total_record_count"]
            clusters = frozen["source_cluster_count"]
            if (manifest.source_key != source_key
                    or manifest.license_partition != license_id
                    or manifest.record_count != total
                    or len(manifest.source_cluster_keys) != clusters):
                raise W06ContractError("W-06 D-03 pack 内容与 global binding 漂移")
        pack_bindings.append(W06PackBinding(
            stage_key,
            effective_key,
            source_key,
            license_id,
            earliest,
            f"{prefix}/manifest.json",
            manifest_sha,
            total,
            clusters,
            1 if replacement else 0,
        ))
        for file_identity in manifest.files:
            item = _binding(prefix, effective_key, file_identity)
            if (item.owner_kind, item.split) in {
                    ("source", None), ("observation", "train")}:
                candidate.append(item)
            if (item.owner_kind, item.split) == ("teacher", "train"):
                teacher.append(item)
            if (item.owner_kind, item.split) in {
                    ("observation", "held_out"), ("observation", "dev"),
                    ("evaluator", "held_out"), ("evaluator", "dev")}:
                evaluator.append(item)

    effective_keys = tuple(item.effective_pack_key for item in pack_bindings)
    return W06FrozenContext(
        current_remote_commit_sha1,
        tuple(parent_sha),
        W06_EXPECTED_SHA256[W06_SOURCE_OVERLAY_PATH],
        stage_keys,
        effective_keys,
        tuple(pack_bindings),
        tuple(candidate),
        tuple(teacher),
        tuple(evaluator),
        W06_RELATION_SUBSTAGE_ORDER,
        W06_DIMENSION_KEYS,
        W06_ABLATION_KEYS,
        W06_PRIVATE_ABLATION_KEYS,
        W06_GENERATION_HARD_CONJUNCT,
        W06_EVALUATION_ORDER,
        W06_AGGREGATION_POLICY,
        W06_ALLOWED_WORKER_COUNTS,
        tuple(recovery["failure_point_keys"]),
        recovery["logical_shard_count"],
        recovery["merge_barrier_key"],
        recovery["cursor_version"],
        W06_LOGICAL_CLOCK_VERSION,
        tuple(sorted(W06_RESOURCE_BUDGET.items())),
        W06_FORMAL_RUN_ID,
        W06_W05_BASE_RUN_ID,
        W06_W05_BASE_RUN_ID,
        backend_profile_key,
        digest_value({
            "source_overlay_sha256": W06_EXPECTED_SHA256[W06_SOURCE_OVERLAY_PATH],
            "w05_receipt_sha256": W06_EXPECTED_SHA256[W06_W05_RECEIPT_PATH],
        }),
        W06_OWNER_KEY,
        W06_CANDIDATE_OWNER,
        W06_TEACHER_OWNER,
        W06_EVALUATOR_OWNER,
        W06_ALLOWED_WRITE_OWNERS,
        W06_FORBIDDEN_WRITE_OWNERS,
        W06_RUNNER_KEY,
        tuple(sorted(W06_ZERO_EXECUTION_STATE.items())),
        W06_OPEN_GENERATION_STATE,
    )


__all__ = [
    "W06_ABLATION_KEYS",
    "W06_AGGREGATION_POLICY",
    "W06_ALLOWED_MODES",
    "W06_ALLOWED_WORKER_COUNTS",
    "W06_ALLOWED_WRITE_OWNERS",
    "W06_CANDIDATE_OWNER",
    "W06_CURSOR_VERSION",
    "W06_DIMENSION_KEYS",
    "W06_EVALUATOR_OWNER",
    "W06_EVALUATION_ORDER",
    "W06_EXPECTED_SHA256",
    "W06_FORMAL_RUN_ID",
    "W06_FORBIDDEN_WRITE_OWNERS",
    "W06_FAILURE_POINT_KEYS",
    "W06_GENERATION_ABLATION_KEY",
    "W06_LOGICAL_CLOCK_VERSION",
    "W06_MERGE_BARRIER_KEY",
    "W06_OPEN_GENERATION_STATE",
    "W06_PRIVATE_ABLATION_KEYS",
    "W06_RESOURCE_BUDGET",
    "W06_RUNNER_KEY",
    "W06_STAGE_KEY",
    "W06_TEACHER_OWNER",
    "W06_V1_PACK_KEY",
    "W06_V2_PACK_KEY",
    "W06_W05_BASE_RUN_ID",
    "W06ContractError",
    "W06FrozenContext",
    "W06PackBinding",
    "W06PayloadAudit",
    "W06PayloadBinding",
    "W06RunRequest",
    "open_w06_frozen_context",
    "validate_w06_request",
]
