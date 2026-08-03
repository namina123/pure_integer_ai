"""W-07 public parent、课程白名单、恢复资源与零执行状态合同。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path, PurePosixPath
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import (
    ArtifactFileIdentity,
    DatasetContractError,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_dataset_io import read_artifact_manifest
from pure_integer_ai.experiments.ph2_w05_contract import digest_value


W07_FORMAT_VERSION = 1
W07_STAGE_KEY = "W-07"
W07_PREREQUISITE_STAGE_KEY = "W-06"
W07_OWNER_KEY = "PH2_W07_TRANSACTION_OWNER"
W07_RUNNER_KEY = "PH2_LANGUAGE_STAGE7_TYPED_LOGIC"
W07_FORMAL_RUN_ID = 8
W07_W06_BASE_RUN_ID = 7
W07_BASELINE_COMMIT_SHA1 = "312283acdbcafdbaa04f3c59bb03f94e6cc46a6d"
W07_W06_RELEASE_COMMIT_SHA1 = "9c6c41d90d27dd02935483391bdd320d78be741f"
W07_ALLOWED_MODES = ("fresh", "restart", "resume")
W07_ALLOWED_WORKER_COUNTS = (1, 2, 4)
W07_LOGICAL_CLOCK_VERSION = "PH2-W07-LOGICAL-CLOCK-V1"
W07_CURSOR_VERSION = "PH2-D03-CURSOR-V1"
W07_MERGE_BARRIER_KEY = "PH2-D03-STABLE-MERGE-BARRIER-V1"
W07_FAILURE_POINT_KEYS = (
    "BEFORE_FIRST_SHARD",
    "AFTER_PARTIAL_SHARD",
    "BEFORE_MERGE_PREVIEW",
    "AFTER_MERGE_BEFORE_COMMIT",
    "AFTER_COMMIT_BEFORE_CURSOR",
    "AFTER_MANIFEST_PUBLISH",
)
W07_OPEN_GENERATION_STATE = "NE_NOT_YET_EVALUABLE"
W07_CANDIDATE_OWNER = "PH2_TRAIN_CANDIDATE"
W07_TEACHER_OWNER = "PH2_TRAINING_EVIDENCE"
W07_EVALUATOR_OWNER = "PH2_PRIVATE_EVALUATOR"
W07_ALLOWED_WRITE_OWNERS = (W07_OWNER_KEY,)
W07_FORBIDDEN_WRITE_OWNERS = (
    "PH2_W01_TRANSACTION_OWNER",
    "PH2_W02_TRANSACTION_OWNER",
    "PH2_W03_TRANSACTION_OWNER",
    "PH2_W04_TRANSACTION_OWNER",
    "PH2_W05_TRANSACTION_OWNER",
    "PH2_W06_TRANSACTION_OWNER",
    W07_EVALUATOR_OWNER,
    "MEMORY",
    "COMPANION",
)

W07_STAGE_MANIFEST_PATH = (
    "data/ph2/manifests/d03_v1/stages/w07_stage_manifest_v1.json")
W07_GLOBAL_MANIFEST_PATH = (
    "data/ph2/manifests/d03_v1/ph2_global_course_manifest_v1.json")
W07_INVALIDATION_GRAPH_PATH = (
    "data/ph2/manifests/d03_v1/stage_invalidation_graph_v1.json")
W07_W06_RECEIPT_PATH = (
    "data/ph2/manifests/d03_v1/w06_runtime_evidence_receipt_v1.json")
W07_LC16_OVERLAY_PATH = "data/ph2/manifests/d03_lc16_successor_overlay_v1.json"
W07_LC16_DIRECTIONAL_PATH = (
    "data/ph2/manifests/lc16_carrier_directional_runtime_v1.json")
W07_LC13_DIRECTIONAL_PATH = (
    "data/ph2/manifests/lc13_directional_consumer_manifest_v2.json")
W07_GENERATION_CHOICE_PATH = (
    "data/ph2/manifests/gg01_generation_choice_contract_v2.json")
W07_GENERATION_OUTCOME_PATH = (
    "data/ph2/manifests/gg02_generation_choice_outcome_bridge_v1.json")

W07_EXPECTED_SHA256 = {
    W07_STAGE_MANIFEST_PATH:
        "8f9833809db3b0529bf904a64a44469ed02795a52a52ee3d5a4fd4e481c3f6e8",
    W07_GLOBAL_MANIFEST_PATH:
        "384329cf651ea4c5e4bc9d0b5dc4da7b22a71bc008bfabe468c86278dd9d40b6",
    W07_INVALIDATION_GRAPH_PATH:
        "21cf4d3cd65afeb0f93054773b97fa4194ee5f14dc463ff40af5813fdb0facce",
    W07_W06_RECEIPT_PATH:
        "aaf35a8346446e80d71f057ae391d9a734a864ced317fa06f2ea01f99efbc0e7",
    W07_LC16_OVERLAY_PATH:
        "6cb9ab991ff41ecd87905f446ed5d75b2ad83e9d6f43124e2a69e15e7135083d",
    W07_LC16_DIRECTIONAL_PATH:
        "c7119639340c9baa5d80c8b582df8131376c3c8dd182f9414717f553a942985e",
    W07_LC13_DIRECTIONAL_PATH:
        "e1b47097d9d863e746be00e3d66936ae775f79cdb42e4377e0c82b56fea9b2b3",
    W07_GENERATION_CHOICE_PATH:
        "9a2a5b1b989d602deb41b6e4fde63747282a690192f2209a48b66bdac91efcd6",
    W07_GENERATION_OUTCOME_PATH:
        "664f65725b02a1f57d45a948a6efea009e37003252b9ee230d4d90788546adb2",
}

W07_SUBSTAGE_ORDER = (
    "NOT", "AND_OR", "CONDITION", "EXISTS", "FORALL", "MODAL",
    "NESTED_SCOPE",
)
W07_FORMING_PACK_KEYS = (
    "AUTHORED_CC0_V1--CC0-1.0--not-v1",
    "AUTHORED_CC0_V1--CC0-1.0--and-or-v1",
    "AUTHORED_CC0_V1--CC0-1.0--condition-v1",
    "AUTHORED_CC0_V1--CC0-1.0--exists-v1",
    "AUTHORED_CC0_V1--CC0-1.0--forall-v1",
    "AUTHORED_CC0_V1--CC0-1.0--modal-v1",
    "AUTHORED_CC0_V1--CC0-1.0--nested-scope-v1",
)
W07_HISTORICAL_DIMENSION_KEYS = (
    "W-07-AND_OR",
    "W-07-CONDITION",
    "W-07-EXISTS",
    "W-07-FORALL",
    "W-07-MODAL",
    "W-07-NESTED_SCOPE",
    "W-07-NOT",
)
W07_HISTORICAL_ABLATION_KEYS = tuple(
    f"{item}-ABLATION" for item in W07_HISTORICAL_DIMENSION_KEYS)
W07_GENERATION_HARD_CONJUNCT = (
    "W-07-GENERATION-LOGIC-SCOPE-HARD-CONJUNCT")
W07_GENERATION_ABLATION_KEY = f"{W07_GENERATION_HARD_CONJUNCT}-ABLATION"
W07_PUBLIC_DIMENSION_KEYS = (
    *W07_HISTORICAL_DIMENSION_KEYS, W07_GENERATION_HARD_CONJUNCT)
W07_PUBLIC_ABLATION_KEYS = (
    *W07_HISTORICAL_ABLATION_KEYS, W07_GENERATION_ABLATION_KEY)
W07_AGGREGATION_POLICY = "ALL_7_W07_BEARINGS_AND_GENERATION_MUST_PASS"
W07_RESOURCE_BUDGET = {
    "max_checkpoint_count": 1792,
    "max_logic_operations": 7000000,
    "max_payload_bytes": 469762048,
    "max_payload_gets": 458752,
    "max_recompute_objects": 700000,
    "max_records": 700000,
    "max_segments": 28672,
    "max_workers": 4,
}
W07_ZERO_EXECUTION_STATE = {
    "LANGUAGE_CAPABILITY_MASTERED": 0,
    "LANGUAGE_READINESS": 0,
    "W06_RUNTIME_EVIDENCED": 1,
    "W06_STARTED": 1,
    "W07_STARTED": 0,
    "W08_STARTED": 0,
    "formal_w07_training_runs": 0,
    "teacher_calls": 0,
}


class W07ContractError(RuntimeError):
    """W-07 parent、可见性、payload、运行或资源身份发生漂移。"""


def _sha256(path: Path) -> str:
    """流式计算文件 SHA-256。"""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
    except OSError as error:
        raise W07ContractError(f"无法读取 W-07 依赖：{path}") from error
    return digest.hexdigest()


def _canonical_object(path: Path) -> dict[str, Any]:
    """读取可带单个末尾换行的规范 JSON object。"""
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise W07ContractError(f"无法读取 W-07 JSON：{path}") from error
    body = payload[:-1] if payload.endswith(b"\n") else payload
    try:
        value = parse_canonical_json_bytes(body, require_object=True)
    except DatasetContractError as error:
        raise W07ContractError(f"W-07 JSON 非规范：{path}") from error
    assert isinstance(value, dict)
    return value


def _safe_relative(value: str) -> str:
    """拒绝绝对路径、反斜杠、点段和空路径。"""
    if not isinstance(value, str) or not value or "\\" in value:
        raise W07ContractError("W-07 payload path 非规范")
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(item in {"", ".", ".."} for item in pure.parts):
        raise W07ContractError("W-07 payload path 越界")
    return pure.as_posix()


@dataclass(frozen=True, order=True)
class W07PayloadBinding:
    """绑定有效 W07 pack 文件的路径、owner/split 与完整身份。"""

    relative_path: str
    pack_key: str
    owner_kind: str
    split: str | None
    file_identity: ArtifactFileIdentity

    def __post_init__(self) -> None:
        object.__setattr__(self, "relative_path", _safe_relative(self.relative_path))
        if self.pack_key not in W07_FORMING_PACK_KEYS:
            raise W07ContractError("W-07 payload pack 不具 forming 资格")
        if self.owner_kind not in {"source", "observation", "teacher", "evaluator"}:
            raise W07ContractError("W-07 payload owner 非法")
        if self.split not in {None, "train", "dev", "held_out"}:
            raise W07ContractError("W-07 payload split 非法")
        if not isinstance(self.file_identity, ArtifactFileIdentity):
            raise W07ContractError("W-07 payload 缺 ArtifactFileIdentity")
        identity = self.file_identity
        if (not self.relative_path.endswith("/" + identity.relative_path)
                or identity.owner_kind != self.owner_kind
                or identity.split != self.split):
            raise W07ContractError("W-07 payload path 与文件身份不一致")

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_identity": self.file_identity.to_dict(),
            "owner_kind": self.owner_kind,
            "pack_key": self.pack_key,
            "relative_path": self.relative_path,
            "split": self.split,
        }


@dataclass(frozen=True, order=True)
class W07PackBinding:
    """记录一个 W07 forming pack 的 D03 manifest 身份。"""

    pack_key: str
    source_key: str
    license_id: str
    earliest_stage: str
    manifest_relative_path: str
    manifest_sha256: str
    total_record_count: int
    source_cluster_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "manifest_relative_path", _safe_relative(
            self.manifest_relative_path))
        if (self.pack_key not in W07_FORMING_PACK_KEYS
                or self.source_key != "AUTHORED_CC0_V1"
                or self.license_id != "CC0-1.0"
                or self.earliest_stage != W07_STAGE_KEY):
            raise W07ContractError("W-07 forming pack 身份漂移")
        if (type(self.total_record_count) is not int
                or self.total_record_count <= 0
                or type(self.source_cluster_count) is not int
                or self.source_cluster_count <= 0
                or len(self.manifest_sha256) != 64):
            raise W07ContractError("W-07 forming pack 计数或 SHA 非法")

    def to_dict(self) -> dict[str, Any]:
        return {
            "earliest_stage": self.earliest_stage,
            "license_id": self.license_id,
            "manifest_relative_path": self.manifest_relative_path,
            "manifest_sha256": self.manifest_sha256,
            "pack_key": self.pack_key,
            "source_cluster_count": self.source_cluster_count,
            "source_key": self.source_key,
            "total_record_count": self.total_record_count,
        }


@dataclass
class W07PayloadAudit:
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
class W07FrozenContext:
    """汇总任何 W-07 payload 读取前必须闭合的公开身份。"""

    baseline_commit_sha1: str
    parent_sha256: tuple[tuple[str, str], ...]
    stage_train_pack_keys: tuple[str, ...]
    forming_pack_keys: tuple[str, ...]
    nonforming_train_pack_keys: tuple[str, ...]
    pack_bindings: tuple[W07PackBinding, ...]
    candidate_payload_bindings: tuple[W07PayloadBinding, ...]
    teacher_evidence_bindings: tuple[W07PayloadBinding, ...]
    forbidden_payload_bindings: tuple[W07PayloadBinding, ...]
    substage_order: tuple[str, ...]
    historical_dimension_keys: tuple[str, ...]
    historical_ablation_keys: tuple[str, ...]
    public_dimension_keys: tuple[str, ...]
    public_ablation_keys: tuple[str, ...]
    generation_hard_conjunct: str
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
        if self.baseline_commit_sha1 != W07_BASELINE_COMMIT_SHA1:
            raise W07ContractError("W-07 baseline commit 漂移")
        if (len(self.stage_train_pack_keys) != 25
                or self.forming_pack_keys != W07_FORMING_PACK_KEYS
                or len(self.nonforming_train_pack_keys) != 18
                or set(self.forming_pack_keys) & set(self.nonforming_train_pack_keys)
                or set(self.stage_train_pack_keys)
                != set((*self.forming_pack_keys, *self.nonforming_train_pack_keys))):
            raise W07ContractError("W-07 train/forming pack 边界漂移")
        if tuple(item.pack_key for item in self.pack_bindings) != self.forming_pack_keys:
            raise W07ContractError("W-07 forming pack binding 顺序漂移")
        candidate_paths = tuple(
            item.relative_path for item in self.candidate_payload_bindings)
        teacher_paths = tuple(
            item.relative_path for item in self.teacher_evidence_bindings)
        forbidden_paths = tuple(
            item.relative_path for item in self.forbidden_payload_bindings)
        if any(len(items) != len(set(items)) for items in (
                candidate_paths, teacher_paths, forbidden_paths)):
            raise W07ContractError("W-07 payload path 重复")
        if ((set(candidate_paths) | set(teacher_paths)) & set(forbidden_paths)
                or set(candidate_paths) & set(teacher_paths)):
            raise W07ContractError("W-07 train 与 forbidden payload 重叠")
        if any((item.owner_kind, item.split) not in {
                ("source", None), ("observation", "train")}
               for item in self.candidate_payload_bindings):
            raise W07ContractError("W-07 candidate whitelist 含非 train payload")
        if any((item.owner_kind, item.split) != ("teacher", "train")
               for item in self.teacher_evidence_bindings):
            raise W07ContractError("W-07 teacher whitelist 含非 train Evidence")
        if any((item.owner_kind, item.split) not in {
                ("observation", "dev"), ("observation", "held_out"),
                ("evaluator", "dev"), ("evaluator", "held_out")}
               for item in self.forbidden_payload_bindings):
            raise W07ContractError("W-07 forbidden owner/split 边界漂移")
        if (self.substage_order != W07_SUBSTAGE_ORDER
                or self.historical_dimension_keys != W07_HISTORICAL_DIMENSION_KEYS
                or self.historical_ablation_keys != W07_HISTORICAL_ABLATION_KEYS
                or self.public_dimension_keys != W07_PUBLIC_DIMENSION_KEYS
                or self.public_ablation_keys != W07_PUBLIC_ABLATION_KEYS
                or self.generation_hard_conjunct != W07_GENERATION_HARD_CONJUNCT
                or self.aggregation_policy != W07_AGGREGATION_POLICY):
            raise W07ContractError("W-07 evaluation/W07-G 合同漂移")
        if (self.allowed_worker_counts != W07_ALLOWED_WORKER_COUNTS
                or self.logical_shard_count != 16
                or self.failure_point_keys != W07_FAILURE_POINT_KEYS
                or self.merge_barrier_key != W07_MERGE_BARRIER_KEY
                or self.cursor_version != W07_CURSOR_VERSION
                or self.logical_clock_version != W07_LOGICAL_CLOCK_VERSION
                or dict(self.resource_budget) != W07_RESOURCE_BUDGET):
            raise W07ContractError("W-07 recovery/resource 合同漂移")
        if (self.run_id, self.parent_run_id, self.base_run_id) != (
                W07_FORMAL_RUN_ID, W07_W06_BASE_RUN_ID, W07_W06_BASE_RUN_ID):
            raise W07ContractError("W-07 run/base 身份漂移")
        if (not self.backend_profile_key or not self.base_fence_key
                or any(type(item) is not int for item in (
                    *self.backend_profile_key, *self.base_fence_key))):
            raise W07ContractError("W-07 backend/base fence key 非法")
        if (self.owner_key != W07_OWNER_KEY
                or self.candidate_owner != W07_CANDIDATE_OWNER
                or self.teacher_owner != W07_TEACHER_OWNER
                or self.evaluator_owner != W07_EVALUATOR_OWNER
                or self.allowed_write_owners != W07_ALLOWED_WRITE_OWNERS
                or self.forbidden_write_owners != W07_FORBIDDEN_WRITE_OWNERS
                or set(self.allowed_write_owners) & set(self.forbidden_write_owners)
                or self.runner_key != W07_RUNNER_KEY
                or dict(self.execution_state) != W07_ZERO_EXECUTION_STATE
                or self.open_generation_state != W07_OPEN_GENERATION_STATE):
            raise W07ContractError("W-07 pre-training 状态漂移")
        if self.payload_gets or self.payload_bytes or self.learning_writes:
            raise W07ContractError("W-07 context 构建触碰了 payload/learning")

    def stable_key(self) -> tuple[int, ...]:
        return digest_value({
            "aggregation_policy": self.aggregation_policy,
            "allowed_worker_counts": list(self.allowed_worker_counts),
            "allowed_write_owners": list(self.allowed_write_owners),
            "backend_profile_key": list(self.backend_profile_key),
            "base_fence_key": list(self.base_fence_key),
            "base_run_id": self.base_run_id,
            "baseline_commit_sha1": self.baseline_commit_sha1,
            "candidate_payload_bindings": [
                item.to_dict() for item in self.candidate_payload_bindings],
            "candidate_owner": self.candidate_owner,
            "cursor_version": self.cursor_version,
            "execution_state": dict(self.execution_state),
            "failure_point_keys": list(self.failure_point_keys),
            "forbidden_write_owners": list(self.forbidden_write_owners),
            "forming_pack_keys": list(self.forming_pack_keys),
            "generation_hard_conjunct": self.generation_hard_conjunct,
            "historical_ablation_keys": list(self.historical_ablation_keys),
            "historical_dimension_keys": list(self.historical_dimension_keys),
            "logical_clock_version": self.logical_clock_version,
            "logical_shard_count": self.logical_shard_count,
            "merge_barrier_key": self.merge_barrier_key,
            "nonforming_train_pack_keys": list(self.nonforming_train_pack_keys),
            "open_generation_state": self.open_generation_state,
            "owner_key": self.owner_key,
            "pack_bindings": [item.to_dict() for item in self.pack_bindings],
            "parent_run_id": self.parent_run_id,
            "parent_sha256": [list(item) for item in self.parent_sha256],
            "public_ablation_keys": list(self.public_ablation_keys),
            "public_dimension_keys": list(self.public_dimension_keys),
            "resource_budget": dict(self.resource_budget),
            "run_id": self.run_id,
            "runner_key": self.runner_key,
            "stage_train_pack_keys": list(self.stage_train_pack_keys),
            "substage_order": list(self.substage_order),
            "teacher_evidence_bindings": [
                item.to_dict() for item in self.teacher_evidence_bindings],
            "teacher_owner": self.teacher_owner,
            "evaluator_owner": self.evaluator_owner,
        })


@dataclass(frozen=True)
class W07RunRequest:
    """声明不含 evaluator/future 字段的 W-07 candidate 请求。"""

    run_id: int
    parent_run_id: int
    base_run_id: int
    stage_key: str
    owner_key: str
    runner_key: str
    baseline_commit_sha1: str
    context_key: tuple[int, ...]
    backend_profile_key: tuple[int, ...]
    base_fence_key: tuple[int, ...]
    worker_count: int
    mode: str
    resource_budget: tuple[tuple[str, int], ...]
    candidate_payload_paths: tuple[str, ...]
    teacher_evidence_paths: tuple[str, ...]

    def execution_identity_key(self) -> tuple[int, ...]:
        return digest_value({
            "base_fence_key": list(self.base_fence_key),
            "base_run_id": self.base_run_id,
            "baseline_commit_sha1": self.baseline_commit_sha1,
            "candidate_payload_paths": list(self.candidate_payload_paths),
            "context_key": list(self.context_key),
            "owner_key": self.owner_key,
            "parent_run_id": self.parent_run_id,
            "resource_budget": dict(self.resource_budget),
            "run_id": self.run_id,
            "runner_key": self.runner_key,
            "stage_key": self.stage_key,
            "teacher_evidence_paths": list(self.teacher_evidence_paths),
        })

    def scheduling_key(self) -> tuple[int, ...]:
        return digest_value({"mode": self.mode, "worker_count": self.worker_count})


def validate_w07_request(
        context: W07FrozenContext,
        request: W07RunRequest,
        ) -> W07RunRequest:
    """在任何 transport 前拒绝 stage、owner、资源和路径漂移。"""
    if not isinstance(context, W07FrozenContext) or not isinstance(
            request, W07RunRequest):
        raise W07ContractError("W-07 context/request 类型非法")
    if (request.run_id, request.parent_run_id, request.base_run_id) != (
            context.run_id, context.parent_run_id, context.base_run_id):
        raise W07ContractError("W-07 run/parent/base id 漂移")
    if (request.stage_key != W07_STAGE_KEY
            or request.owner_key != context.owner_key
            or request.runner_key != context.runner_key):
        raise W07ContractError("W-07 stage/owner/runner 未授权")
    if request.baseline_commit_sha1 != context.baseline_commit_sha1:
        raise W07ContractError("W-07 baseline commit 漂移")
    if (request.context_key != context.stable_key()
            or request.backend_profile_key != context.backend_profile_key
            or request.base_fence_key != context.base_fence_key):
        raise W07ContractError("W-07 context/backend/base fence 漂移")
    if request.worker_count not in context.allowed_worker_counts:
        raise W07ContractError("W-07 worker count 非法")
    if request.mode not in W07_ALLOWED_MODES:
        raise W07ContractError("W-07 mode 非法")
    if request.resource_budget != tuple(sorted(W07_RESOURCE_BUDGET.items())):
        raise W07ContractError("W-07 resource budget 漂移")
    candidate = tuple(
        item.relative_path for item in context.candidate_payload_bindings)
    teacher = tuple(
        item.relative_path for item in context.teacher_evidence_bindings)
    if request.candidate_payload_paths != candidate:
        raise W07ContractError("W-07 request 非精确 train whitelist")
    if request.teacher_evidence_paths != teacher:
        raise W07ContractError("W-07 request 非精确 teacher whitelist")
    return request


def _binding(
        prefix: str,
        pack_key: str,
        identity: ArtifactFileIdentity,
        ) -> W07PayloadBinding:
    return W07PayloadBinding(
        f"{prefix}/{identity.relative_path}",
        pack_key,
        identity.owner_kind,
        identity.split,
        identity,
    )


def _validate_w06_receipt(receipt: dict[str, Any]) -> None:
    """只读确认 W07 prerequisite receipt 的八维与八消融闭合。"""
    state = receipt.get("execution_state", {})
    dimensions = receipt.get("dimension_results", ())
    ablations = receipt.get("ablation_results", ())
    if (receipt.get("status") != "RUNTIME_EVIDENCED"
            or receipt.get("stage_key") != "W-06"
            or len(dimensions) != 8
            or any(item.get("status") != "PASS" for item in dimensions)
            or len(ablations) != 8
            or any(tuple(item.get("dimension_statuses", ())).count("FAIL") != 1
                   for item in ablations)
            or state.get("W06_STARTED") != 1
            or state.get("W06_RUNTIME_EVIDENCED") != 1
            or state.get("W06_BLOCKED_FAILED") != 0
            or state.get("formal_w06_training_runs") != 1
            or state.get("teacher_calls") != 0
            or state.get("W07_STARTED") != 0):
        raise W07ContractError("W-06 prerequisite receipt 未闭合")


def open_w07_frozen_context(
        repository_root: str | Path,
        *,
        baseline_commit_sha1: str,
        backend_profile_key: tuple[int, ...],
        ) -> W07FrozenContext:
    """只读打开 public parent 与七个 forming train pack 身份。"""
    root = Path(repository_root).resolve()
    if baseline_commit_sha1 != W07_BASELINE_COMMIT_SHA1:
        raise W07ContractError("W-07 baseline commit 不是 W07-00 检查点")
    parent_sha = []
    for relative, expected in W07_EXPECTED_SHA256.items():
        actual = _sha256(root / relative)
        if actual != expected:
            raise W07ContractError(f"W-07 parent SHA 漂移：{relative}")
        parent_sha.append((relative, actual))

    stage = _canonical_object(root / W07_STAGE_MANIFEST_PATH)
    global_manifest = _canonical_object(root / W07_GLOBAL_MANIFEST_PATH)
    _validate_w06_receipt(_canonical_object(root / W07_W06_RECEIPT_PATH))
    identity = stage.get("stage_identity", {})
    visibility = stage.get("data_visibility", {})
    recovery = stage.get("recovery_binding", {})
    evaluation = stage.get("evaluation_binding", {})
    stage_keys = tuple(visibility.get("train_pack_keys", ()))
    if (identity.get("stage_key") != W07_STAGE_KEY
            or identity.get("ordinal") != 7
            or tuple(identity.get("prerequisite_stage_keys", ()))
            != (W07_PREREQUISITE_STAGE_KEY,)
            or tuple(identity.get("substage_keys", ())) != W07_SUBSTAGE_ORDER):
        raise W07ContractError("W-07 stage identity 漂移")
    if (len(stage_keys) != 25
            or not set(W07_FORMING_PACK_KEYS).issubset(stage_keys)
            or len(tuple(visibility.get("dev_pack_keys", ()))) != 1
            or len(tuple(visibility.get("held_out_pack_keys", ()))) != 28
            or len(tuple(visibility.get("evaluator_pack_keys", ()))) != 28
            or len(tuple(visibility.get("future_pack_keys", ()))) != 9):
        raise W07ContractError("W-07 stage pack inventory 漂移")
    thresholds = tuple(evaluation.get("thresholds", ()))
    if (tuple(item.get("dimension_key") for item in thresholds)
            != W07_HISTORICAL_DIMENSION_KEYS
            or tuple(evaluation.get("ablation_keys", ()))
            != W07_HISTORICAL_ABLATION_KEYS
            or any(
                item.get("bearing") != 1
                or item.get("min_pass_numerator") != 1
                or item.get("min_pass_denominator") != 1
                or item.get("max_fail_count") != 0
                or item.get("ne_policy") != "BLOCK"
                or item.get("preregistered") != 1
                for item in thresholds)):
        raise W07ContractError("W-07 bearing/threshold 被放宽")
    if (tuple(recovery.get("allowed_worker_counts", ()))
            != W07_ALLOWED_WORKER_COUNTS
            or recovery.get("logical_shard_count") != 16
            or tuple(recovery.get("failure_point_keys", ()))
            != W07_FAILURE_POINT_KEYS
            or recovery.get("merge_barrier_key") != W07_MERGE_BARRIER_KEY
            or recovery.get("cursor_version") != W07_CURSOR_VERSION
            or stage.get("resource_budget") != W07_RESOURCE_BUDGET):
        raise W07ContractError("W-07 recovery/resource manifest 漂移")
    if (visibility.get("candidate_owner") != W07_CANDIDATE_OWNER
            or visibility.get("teacher_owner") != W07_TEACHER_OWNER
            or visibility.get("evaluator_owner") != W07_EVALUATOR_OWNER
            or tuple(visibility.get("candidate_allowed_splits", ())) != ("train",)
            or tuple(visibility.get("candidate_forbidden_splits", ()))
            != ("dev", "held_out", "adversarial", "wall")):
        raise W07ContractError("W-07 manifest owner/split 防火墙漂移")

    global_by_key = {
        item["pack_key"]: item for item in global_manifest.get("pack_bindings", ())}
    pack_bindings = []
    candidate = []
    teacher = []
    forbidden = []
    for pack_key in W07_FORMING_PACK_KEYS:
        frozen = global_by_key.get(pack_key)
        if frozen is None:
            raise W07ContractError("W-07 forming pack 未进入 D-03 global")
        manifest_relative = frozen["manifest_identity"]["relative_path"]
        manifest_path = root / manifest_relative
        manifest_sha = _sha256(manifest_path)
        if manifest_sha != frozen["manifest_identity"]["sha256"]:
            raise W07ContractError("W-07 pack manifest SHA 漂移")
        manifest = read_artifact_manifest(manifest_path)
        if (frozen["earliest_stage"] != W07_STAGE_KEY
                or frozen["source_key"] != "AUTHORED_CC0_V1"
                or frozen["license_id"] != "CC0-1.0"
                or manifest.source_key != frozen["source_key"]
                or manifest.license_partition != frozen["license_id"]
                or manifest.record_count != frozen["total_record_count"]
                or len(manifest.source_cluster_keys)
                != frozen["source_cluster_count"]):
            raise W07ContractError("W-07 pack 内容与 global binding 漂移")
        prefix = PurePosixPath(manifest_relative).parent.as_posix()
        pack_bindings.append(W07PackBinding(
            pack_key,
            frozen["source_key"],
            frozen["license_id"],
            frozen["earliest_stage"],
            manifest_relative,
            manifest_sha,
            frozen["total_record_count"],
            frozen["source_cluster_count"],
        ))
        for file_identity in manifest.files:
            item = _binding(prefix, pack_key, file_identity)
            if (item.owner_kind, item.split) in {
                    ("source", None), ("observation", "train")}:
                candidate.append(item)
            elif (item.owner_kind, item.split) == ("teacher", "train"):
                teacher.append(item)
            elif (item.owner_kind, item.split) in {
                    ("observation", "dev"), ("observation", "held_out"),
                    ("evaluator", "dev"), ("evaluator", "held_out")}:
                forbidden.append(item)
            else:
                raise W07ContractError("W-07 pack 出现未注册 owner/split")

    nonforming = tuple(item for item in stage_keys
                       if item not in set(W07_FORMING_PACK_KEYS))
    return W07FrozenContext(
        baseline_commit_sha1,
        tuple(parent_sha),
        stage_keys,
        W07_FORMING_PACK_KEYS,
        nonforming,
        tuple(pack_bindings),
        tuple(candidate),
        tuple(teacher),
        tuple(forbidden),
        W07_SUBSTAGE_ORDER,
        W07_HISTORICAL_DIMENSION_KEYS,
        W07_HISTORICAL_ABLATION_KEYS,
        W07_PUBLIC_DIMENSION_KEYS,
        W07_PUBLIC_ABLATION_KEYS,
        W07_GENERATION_HARD_CONJUNCT,
        W07_AGGREGATION_POLICY,
        W07_ALLOWED_WORKER_COUNTS,
        W07_FAILURE_POINT_KEYS,
        recovery["logical_shard_count"],
        recovery["merge_barrier_key"],
        recovery["cursor_version"],
        W07_LOGICAL_CLOCK_VERSION,
        tuple(sorted(W07_RESOURCE_BUDGET.items())),
        W07_FORMAL_RUN_ID,
        W07_W06_BASE_RUN_ID,
        W07_W06_BASE_RUN_ID,
        backend_profile_key,
        digest_value({
            "baseline_commit_sha1": baseline_commit_sha1,
            "w06_receipt_sha256": W07_EXPECTED_SHA256[W07_W06_RECEIPT_PATH],
        }),
        W07_OWNER_KEY,
        W07_CANDIDATE_OWNER,
        W07_TEACHER_OWNER,
        W07_EVALUATOR_OWNER,
        W07_ALLOWED_WRITE_OWNERS,
        W07_FORBIDDEN_WRITE_OWNERS,
        W07_RUNNER_KEY,
        tuple(sorted(W07_ZERO_EXECUTION_STATE.items())),
        W07_OPEN_GENERATION_STATE,
    )


__all__ = [name for name in globals() if name.startswith("W07_")] + [
    "W07ContractError",
    "W07FrozenContext",
    "W07PackBinding",
    "W07PayloadAudit",
    "W07PayloadBinding",
    "W07RunRequest",
    "open_w07_frozen_context",
    "validate_w07_request",
]
