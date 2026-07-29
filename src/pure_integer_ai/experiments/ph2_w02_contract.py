"""正式中文 PH2 W-02 的冻结输入、入口和 payload 前授权合同。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    D03ContractError,
    sha1_text,
)
from pure_integer_ai.experiments.ph2_d03_publication import (
    read_d03_publication_receipt,
)
from pure_integer_ai.experiments.ph2_d03_release_catalog import (
    FORMAL_GLOBAL_MANIFEST_PATH,
    FORMAL_RECEIPT_PATH,
)
from pure_integer_ai.experiments.ph2_d03_release_reader import D03ReleaseReader
from pure_integer_ai.experiments.ph2_dataset_core import (
    DatasetContractError,
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    ArtifactFileIdentity,
    ObservationRecord,
    SourceRefRecord,
    TeacherEvidenceRecord,
)
from pure_integer_ai.experiments.ph2_dataset_io import (
    DatasetArtifactIOError,
    read_artifact_manifest,
    read_record_artifact,
)
from pure_integer_ai.experiments.ph2_d03_stage_contract import (
    EvaluationThreshold,
)
from pure_integer_ai.experiments.ph2_w01_receipt import (
    W01_FORMAL_RECEIPT_PATH,
    W01ReceiptError,
    read_w01_formal_receipt,
)


D03_GLOBAL_MANIFEST_PATH = FORMAL_GLOBAL_MANIFEST_PATH
W02_FORMAT_VERSION = 1
W02_STAGE_KEY = "W-02"
W01_STAGE_KEY = "W-01"
W02_OWNER_KEY = "PH2_W02_TRANSACTION_OWNER"
W02_RUNNER_KEY = "PH2_LANGUAGE_STAGE1"
W02_ALLOWED_MODES = ("fresh", "restart", "resume")
W02_TRAIN_PACK_KEYS = (
    "AUTHORED_CC0_V1--CC0-1.0--lc01-text-fidelity-v1",
    "AUTHORED_CC0_V1--CC0-1.0--lc02-morphology-v1",
)
W02_DIMENSION_KEYS = (
    "W-02-BOUNDARY_WITHDRAWAL",
    "W-02-MULTI_CANDIDATE",
    "W-02-NEW_CONTENT_MORPHOLOGY",
    "W-02-OOV",
)
W02_ABLATION_KEYS = tuple(
    f"{item}-ABLATION" for item in W02_DIMENSION_KEYS)


class W02ContractError(RuntimeError):
    """W-02 发布依赖、入口身份或 payload 前可见性不满足。"""


def _digest_value(value: Any) -> tuple[int, ...]:
    """把无浮点结构对象映射为稳定 SHA-256 整数键。"""
    return tuple(hashlib.sha256(canonical_json_bytes(value)).digest())


def _strict_key(value: object, *, label: str) -> tuple[int, ...]:
    """要求开放身份是非空严格整数 tuple。"""
    if (not isinstance(value, tuple) or not value
            or any(type(item) is not int for item in value)):
        raise W02ContractError(f"{label} 必须是非空严格整数 tuple")
    return value


def _relative_path(value: object, *, label: str) -> str:
    """要求路径是无逃逸、无反斜杠的规范 POSIX 相对路径。"""
    if not isinstance(value, str) or not value:
        raise W02ContractError(f"{label} 必须是非空路径")
    pure = PurePosixPath(value)
    if (pure.is_absolute() or ".." in pure.parts or "\\" in value
            or pure.as_posix() != value):
        raise W02ContractError(f"{label} 不是规范安全路径")
    return value


def _overlay_file(primary: Path, dependency: Path, relative: str) -> Path:
    """在主根或依赖根内解析文件，并拒绝符号链接或父路径逃逸。"""
    normalized = _relative_path(relative, label="overlay path")
    parts = Path(*PurePosixPath(normalized).parts)
    for root in (primary, dependency):
        target = (root / parts).resolve()
        if target.is_relative_to(root) and target.is_file():
            return target
    raise W02ContractError(f"冻结文件缺失: {relative}")


def _verify_file(path: Path, *, size_bytes: int, sha256: str) -> None:
    """只回验 manifest/receipt 元数据文件的字节身份。"""
    payload = path.read_bytes()
    if (len(payload) != size_bytes
            or hashlib.sha256(payload).hexdigest() != sha256):
        raise W02ContractError("冻结 manifest 文件身份漂移")


@dataclass(frozen=True, order=True)
class W02PayloadBinding:
    """一个候选可见 payload 的 D-03 transport 与内容身份。"""

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
        """核验 payload 路径、owner、split、计数和双摘要。"""
        object.__setattr__(self, "relative_path", _relative_path(
            self.relative_path, label="payload path"))
        if (not isinstance(self.pack_key, str) or not self.pack_key
                or self.owner_kind not in {"source", "observation", "teacher"}
                or self.split not in {None, "train"}):
            raise W02ContractError("W-02 payload pack/owner/split 非法")
        for name in ("record_count", "transport_size_bytes", "content_size_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise W02ContractError(f"payload {name} 必须为正严格整数")
        for name in ("transport_sha256", "content_sha256"):
            value = getattr(self, name)
            if (not isinstance(value, str) or len(value) != 64
                    or any(item not in "0123456789abcdef" for item in value)):
                raise W02ContractError(f"payload {name} 不是规范 SHA-256")
        if not isinstance(self.file_identity, ArtifactFileIdentity):
            raise W02ContractError("payload 缺少完整 ArtifactFileIdentity")
        identity = self.file_identity
        if (not self.relative_path.endswith("/" + identity.relative_path)
                or identity.owner_kind != self.owner_kind
                or identity.split != self.split
                or identity.record_count != self.record_count
                or identity.transport_size_bytes != self.transport_size_bytes
                or identity.transport_sha256 != self.transport_sha256
                or identity.content_size_bytes != self.content_size_bytes
                or identity.content_sha256 != self.content_sha256):
            raise W02ContractError("payload 完整文件身份与 D-03 路径绑定漂移")
        if self.owner_kind == "teacher":
            if "/owners/teacher/" not in self.relative_path or self.split != "train":
                raise W02ContractError("teacher Evidence 不在 train 专属 owner")
        elif "/owners/" in self.relative_path:
            raise W02ContractError("candidate payload 不得位于私有 owner")

    def to_dict(self) -> dict[str, object]:
        """导出供执行身份和后续 firewall 使用的规范绑定。"""
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
class W02PackBinding:
    """W-02 训练 pack 的发布 manifest、许可、来源和计数身份。"""

    pack_key: str
    source_key: str
    license_id: str
    manifest_path: str
    manifest_size_bytes: int
    manifest_sha256: str
    record_count: int
    source_cluster_count: int

    def __post_init__(self) -> None:
        """要求 pack 属于冻结二元集合且具有正计数和安全 manifest。"""
        if self.pack_key not in W02_TRAIN_PACK_KEYS:
            raise W02ContractError("W-02 pack 不在冻结白名单")
        if self.source_key != "AUTHORED_CC0_V1" or self.license_id != "CC0-1.0":
            raise W02ContractError("W-02 pack 来源或许可漂移")
        object.__setattr__(self, "manifest_path", _relative_path(
            self.manifest_path, label="pack manifest path"))
        if (type(self.manifest_size_bytes) is not int
                or self.manifest_size_bytes <= 0
                or type(self.record_count) is not int or self.record_count <= 0
                or type(self.source_cluster_count) is not int
                or self.source_cluster_count <= 0):
            raise W02ContractError("W-02 pack 计数必须为正严格整数")
        if (len(self.manifest_sha256) != 64
                or any(item not in "0123456789abcdef"
                       for item in self.manifest_sha256)):
            raise W02ContractError("W-02 pack manifest SHA-256 非法")

    def to_dict(self) -> dict[str, object]:
        """导出 pack 的稳定执行身份。"""
        return {
            "license_id": self.license_id,
            "manifest_path": self.manifest_path,
            "manifest_sha256": self.manifest_sha256,
            "manifest_size_bytes": self.manifest_size_bytes,
            "pack_key": self.pack_key,
            "record_count": self.record_count,
            "source_cluster_count": self.source_cluster_count,
            "source_key": self.source_key,
        }


@dataclass
class W02PayloadAudit:
    """区分底层 transport 尝试与成功交付给训练 consumer 的 payload。"""

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
class W02TrainingPayload:
    """经防火墙完整核验后一次性交付的 train-only typed 记录。"""

    source_refs: tuple[SourceRefRecord, ...]
    observations: tuple[ObservationRecord, ...]
    teacher_evidence: tuple[TeacherEvidenceRecord, ...]


class W02PayloadFirewall:
    """在 W-02 candidate 与 D-03 文件系统之间实施精确的单次读取边界。"""

    def __init__(
            self,
            primary: Path,
            dependency: Path,
            context: "W02FrozenContext",
            request: "W02RunRequest",
            audit: W02PayloadAudit,
            ) -> None:
        self._primary = primary
        self._dependency = dependency
        self._context = context
        self._request = request
        self.audit = audit
        self._consumed = False

    @classmethod
    def open(
            cls,
            repository_root: str | Path,
            context: "W02FrozenContext",
            request: "W02RunRequest",
            *,
            dependency_root: str | Path | None = None,
            audit: W02PayloadAudit | None = None,
            ) -> "W02PayloadFirewall":
        """先核完整 request；失败时不进行任何 transport 访问。"""
        if audit is not None and not isinstance(audit, W02PayloadAudit):
            raise W02ContractError("payload audit 类型错误")
        actual_audit = audit if audit is not None else W02PayloadAudit()
        validate_w02_request(context, request)
        primary = Path(repository_root).resolve()
        dependency = (
            Path(dependency_root).resolve()
            if dependency_root is not None else primary)
        return cls(primary, dependency, context, request, actual_audit)

    def _read_binding(self, binding: W02PayloadBinding):
        """按完整 transport/content identity 读取一个已授权文件。"""
        target = _overlay_file(
            self._primary, self._dependency, binding.relative_path)
        local_parts = Path(binding.file_identity.relative_path).parts
        artifact_root = target.parents[len(local_parts) - 1]
        self.audit.transport_attempts += 1
        try:
            self.audit.transport_bytes += target.stat().st_size
            records = read_record_artifact(
                artifact_root, binding.file_identity)
        except (DatasetArtifactIOError, OSError) as exc:
            raise W02ContractError(
                f"payload transport/gzip/SHA-256 核验失败: {binding.relative_path}"
            ) from exc
        self.audit.payload_gets += 1
        self.audit.payload_bytes += binding.transport_size_bytes
        return records

    def read_training_payload(self) -> W02TrainingPayload:
        """只读一次精确白名单，并在交付前闭合 owner 与引用完整性。"""
        if self._consumed:
            raise W02ContractError("同一 payload firewall 不得重复读取")
        self._consumed = True
        source_refs: list[SourceRefRecord] = []
        observations: list[ObservationRecord] = []
        teacher: list[TeacherEvidenceRecord] = []
        try:
            for binding in self._context.candidate_payload_bindings:
                records = self._read_binding(binding)
                if binding.owner_kind == "source":
                    if any(not isinstance(item, SourceRefRecord)
                           for item in records):
                        raise W02ContractError("candidate SourceRef 文件混入其他记录")
                    source_refs.extend(records)
                elif binding.owner_kind == "observation":
                    if any(not isinstance(item, ObservationRecord)
                           for item in records):
                        raise W02ContractError("candidate Observation 文件混入其他记录")
                    observations.extend(records)
                else:
                    raise W02ContractError("candidate 白名单含非法 owner")
            for binding in self._context.teacher_evidence_bindings:
                records = self._read_binding(binding)
                if any(not isinstance(item, TeacherEvidenceRecord)
                       for item in records):
                    raise W02ContractError("teacher Evidence 文件混入其他记录")
                teacher.extend(records)
            self._validate_records(source_refs, observations, teacher)
        except W02ContractError:
            raise
        except (TypeError, ValueError, KeyError) as exc:
            raise W02ContractError("W-02 train payload 引用或 owner 无效") from exc
        self.audit.source_ref_reads += len(source_refs)
        self.audit.observation_reads += len(observations)
        self.audit.teacher_evidence_reads += len(teacher)
        return W02TrainingPayload(
            tuple(source_refs), tuple(observations), tuple(teacher))

    @staticmethod
    def _validate_records(
            source_refs: list[SourceRefRecord],
            observations: list[ObservationRecord],
            teacher: list[TeacherEvidenceRecord],
            ) -> None:
        """要求所有交付记录都属于 W-02 train，并闭合来源/Observation 引用。"""
        source_keys = {item.stable_key for item in source_refs}
        observation_keys = {item.stable_key for item in observations}
        if (len(source_keys) != len(source_refs)
                or len(observation_keys) != len(observations)):
            raise W02ContractError("W-02 train SourceRef/Observation key 重复")
        if any(
                item.w_stage != W02_STAGE_KEY
                or item.split != "train"
                or item.source_ref_key not in source_keys
                for item in observations):
            raise W02ContractError("Observation 不是闭合的 W-02 train 记录")
        if any(
                item.visible_from_stage != W02_STAGE_KEY
                or item.observation_key not in observation_keys
                or item.source_ref_key not in source_keys
                for item in teacher):
            raise W02ContractError("teacher Evidence 越级或引用非 train 记录")


@dataclass(frozen=True)
class W02FrozenContext:
    """绑定远端基线、D-03/W-01 和 W-02 三视图的只读上下文。"""

    current_remote_commit_sha1: str
    d03_release_key: str
    d03_published: int
    d03_content_commit_sha1: str
    d03_receipt_sha256: str
    global_manifest_path: str
    global_manifest_sha256: str
    stage_manifest_path: str
    stage_manifest_sha256: str
    w01_receipt_path: str
    w01_receipt_sha256: str
    w01_status: str
    w01_run_id: int
    w01_cursor_next_stage: str
    stage_key: str
    stage_ordinal: int
    prerequisite_stage_keys: tuple[str, ...]
    train_pack_keys: tuple[str, ...]
    pack_bindings: tuple[W02PackBinding, ...]
    candidate_payload_bindings: tuple[W02PayloadBinding, ...]
    teacher_evidence_bindings: tuple[W02PayloadBinding, ...]
    evaluator_visible_paths: tuple[str, ...]
    evaluator_private_paths: tuple[str, ...]
    thresholds: tuple[EvaluationThreshold, ...]
    dimension_keys: tuple[str, ...]
    ablation_keys: tuple[str, ...]
    aggregation_policy: str
    allowed_worker_counts: tuple[int, ...]
    failure_point_keys: tuple[str, ...]
    logical_shard_count: int
    merge_barrier_key: str
    cursor_version: str
    resource_budget: dict[str, int]
    version_keys: tuple[tuple[str, str], ...]
    payload_gets: int = 0
    payload_bytes: int = 0
    learning_writes: int = 0

    def __post_init__(self) -> None:
        """核验所有前置、白名单、私有隔离和预注册合取未被放宽。"""
        object.__setattr__(self, "current_remote_commit_sha1", sha1_text(
            self.current_remote_commit_sha1, where="current remote commit"))
        if (self.d03_published != 1 or self.d03_release_key != "PH2-D03-V1"
                or self.w01_status != "W01_PROTOCOL_VERIFIED"
                or self.w01_run_id != 1
                or self.w01_cursor_next_stage != W02_STAGE_KEY):
            raise W02ContractError("D-03/W-01 正式前置未闭合")
        if (self.stage_key != W02_STAGE_KEY or self.stage_ordinal != 2
                or self.prerequisite_stage_keys != (W01_STAGE_KEY,)):
            raise W02ContractError("W-02 阶段身份或前置顺序漂移")
        if self.train_pack_keys != W02_TRAIN_PACK_KEYS:
            raise W02ContractError("W-02 train pack 集漂移")
        if tuple(item.pack_key for item in self.pack_bindings) != W02_TRAIN_PACK_KEYS:
            raise W02ContractError("W-02 pack binding 不完整")
        candidate_paths = tuple(
            item.relative_path for item in self.candidate_payload_bindings)
        teacher_paths = tuple(
            item.relative_path for item in self.teacher_evidence_bindings)
        if (len(candidate_paths) != 4 or len(set(candidate_paths)) != 4
                or len(teacher_paths) != 2 or len(set(teacher_paths)) != 2):
            raise W02ContractError("W-02 candidate/teacher 白名单数量漂移")
        if (not isinstance(self.evaluator_visible_paths, tuple)
                or len(self.evaluator_visible_paths) != 9
                or len(set(self.evaluator_visible_paths)) != 9):
            raise W02ContractError("W-02 evaluator 可见路径集合漂移")
        if (not isinstance(self.evaluator_private_paths, tuple)
                or len(self.evaluator_private_paths) != 7
                or len(set(self.evaluator_private_paths)) != 7):
            raise W02ContractError("W-02 evaluator 私有路径集合漂移")
        if set(self.evaluator_private_paths) != (
                set(self.evaluator_visible_paths) - set(candidate_paths)):
            raise W02ContractError("W-02 evaluator 私有路径推导漂移")
        private = set(self.evaluator_private_paths)
        if private & (set(candidate_paths) | set(teacher_paths)):
            raise W02ContractError("candidate/teacher 与 evaluator 路径交叉")
        for path in self.evaluator_private_paths:
            _relative_path(path, label="evaluator private path")
        if (self.dimension_keys != W02_DIMENSION_KEYS
                or self.ablation_keys != W02_ABLATION_KEYS
                or self.aggregation_policy != "ALL_BEARING_DIMENSIONS_MUST_PASS"):
            raise W02ContractError("W-02 evaluator 合取或消融漂移")
        if tuple(item.dimension_key for item in self.thresholds) != (
                W02_DIMENSION_KEYS):
            raise W02ContractError("W-02 evaluator threshold 维度漂移")
        if any(
                item.bearing != 1
                or item.min_pass_numerator != 1
                or item.min_pass_denominator != 1
                or item.max_fail_count != 0
                or item.ne_policy != "BLOCK"
                or item.preregistered != 1
                for item in self.thresholds):
            raise W02ContractError("W-02 evaluator threshold 被放宽")
        if (self.allowed_worker_counts != (1, 2, 4)
                or len(self.failure_point_keys) != 6
                or self.logical_shard_count != 16):
            raise W02ContractError("W-02 worker/shard/fault 合同漂移")
        if (self.payload_gets != 0 or self.payload_bytes != 0
                or self.learning_writes != 0):
            raise W02ContractError("W-02 context 构造前发生 payload 或学习写")

    def stable_key(self) -> tuple[int, ...]:
        """返回绑定远端、发布物、输入、评测、恢复和预算的身份。"""
        return _digest_value({
            "ablation_keys": list(self.ablation_keys),
            "aggregation_policy": self.aggregation_policy,
            "allowed_worker_counts": list(self.allowed_worker_counts),
            "candidate_payload_bindings": [
                item.to_dict() for item in self.candidate_payload_bindings],
            "current_remote_commit_sha1": self.current_remote_commit_sha1,
            "cursor_version": self.cursor_version,
            "d03_content_commit_sha1": self.d03_content_commit_sha1,
            "d03_receipt_sha256": self.d03_receipt_sha256,
            "d03_release_key": self.d03_release_key,
            "dimension_keys": list(self.dimension_keys),
            "evaluator_visible_paths": list(self.evaluator_visible_paths),
            "evaluator_private_paths": list(self.evaluator_private_paths),
            "failure_point_keys": list(self.failure_point_keys),
            "global_manifest_path": self.global_manifest_path,
            "global_manifest_sha256": self.global_manifest_sha256,
            "logical_shard_count": self.logical_shard_count,
            "merge_barrier_key": self.merge_barrier_key,
            "pack_bindings": [item.to_dict() for item in self.pack_bindings],
            "resource_budget": dict(self.resource_budget),
            "stage_key": self.stage_key,
            "stage_manifest_path": self.stage_manifest_path,
            "stage_manifest_sha256": self.stage_manifest_sha256,
            "teacher_evidence_bindings": [
                item.to_dict() for item in self.teacher_evidence_bindings],
            "thresholds": [item.to_dict() for item in self.thresholds],
            "version_keys": [list(item) for item in self.version_keys],
            "w01_receipt_path": self.w01_receipt_path,
            "w01_receipt_sha256": self.w01_receipt_sha256,
            "w01_run_id": self.w01_run_id,
            "w01_status": self.w01_status,
        })


@dataclass(frozen=True)
class W02RunRequest:
    """candidate runner 可见的完整 typed 请求，不含 evaluator 私有路径字段。"""

    run_id: int
    parent_run_id: int
    base_run_id: int
    stage_key: str
    owner_key: str
    runner_key: str
    current_remote_commit_sha1: str
    d03_context_key: tuple[int, ...]
    w01_receipt_sha256: str
    backend_profile_key: tuple[int, ...]
    base_fence_key: tuple[int, ...]
    worker_count: int
    mode: str
    candidate_payload_paths: tuple[str, ...]
    teacher_evidence_paths: tuple[str, ...]

    def execution_identity_key(self) -> tuple[int, ...]:
        """返回不含 worker/mode 的执行身份，允许同一 run 改调度恢复。"""
        return _digest_value({
            "backend_profile_key": list(self.backend_profile_key),
            "base_fence_key": list(self.base_fence_key),
            "base_run_id": self.base_run_id,
            "candidate_payload_paths": list(self.candidate_payload_paths),
            "current_remote_commit_sha1": self.current_remote_commit_sha1,
            "d03_context_key": list(self.d03_context_key),
            "owner_key": self.owner_key,
            "parent_run_id": self.parent_run_id,
            "run_id": self.run_id,
            "runner_key": self.runner_key,
            "stage_key": self.stage_key,
            "teacher_evidence_paths": list(self.teacher_evidence_paths),
            "w01_receipt_sha256": self.w01_receipt_sha256,
        })

    def scheduling_key(self) -> tuple[int, ...]:
        """返回仅含恢复 mode 和物理 worker 数的调度身份。"""
        return _digest_value({"mode": self.mode, "worker_count": self.worker_count})


def _manifest_bindings(
        primary: Path,
        dependency: Path,
        reader: D03ReleaseReader,
        ) -> tuple[
            tuple[W02PackBinding, ...],
            tuple[W02PayloadBinding, ...],
            tuple[W02PayloadBinding, ...],
        ]:
    """只读两个 pack manifest，把 D-03 路径映射为可读前核验的文件身份。"""
    packs: list[W02PackBinding] = []
    candidate: list[W02PayloadBinding] = []
    teacher: list[W02PayloadBinding] = []
    catalog = {
        item.pack_key: item for item in reader.global_manifest.pack_bindings}
    for pack_key in W02_TRAIN_PACK_KEYS:
        pack = catalog.get(pack_key)
        if pack is None or pack.earliest_stage != W02_STAGE_KEY:
            raise W02ContractError("W-02 pack catalog 缺失或阶段漂移")
        manifest_path = _overlay_file(
            primary, dependency, pack.manifest_identity.relative_path)
        _verify_file(
            manifest_path,
            size_bytes=pack.manifest_identity.size_bytes,
            sha256=pack.manifest_identity.sha256,
        )
        manifest = read_artifact_manifest(manifest_path)
        if (manifest.record_count != pack.total_record_count
                or manifest.source_key != pack.source_key
                or manifest.license_partition != pack.license_id
                or len(manifest.source_cluster_keys) != pack.source_cluster_count
                or manifest.splits != ("train", "held_out")
                or W02_STAGE_KEY not in manifest.w_stages):
            raise W02ContractError("W-02 pack manifest 与 D-03 catalog 漂移")
        prefix = PurePosixPath(pack.manifest_identity.relative_path).parent
        files = {
            PurePosixPath(prefix, item.relative_path).as_posix(): item
            for item in manifest.files
        }
        expected_paths = set(pack.payload_paths)
        if set(files) != expected_paths:
            raise W02ContractError("W-02 pack manifest 未覆盖全部物理路径")
        packs.append(W02PackBinding(
            pack.pack_key,
            pack.source_key,
            pack.license_id,
            pack.manifest_identity.relative_path,
            pack.manifest_identity.size_bytes,
            pack.manifest_identity.sha256,
            pack.total_record_count,
            pack.source_cluster_count,
        ))
        for relative in (*pack.source_ref_paths, *pack.train_observation_paths):
            item = files[relative]
            candidate.append(W02PayloadBinding(
                relative,
                pack.pack_key,
                item.owner_kind,
                item.split,
                item.record_count,
                item.transport_size_bytes,
                item.transport_sha256,
                item.content_size_bytes,
                item.content_sha256,
                item,
            ))
        for relative in pack.teacher_evidence_paths:
            item = files[relative]
            if item.owner_kind != "teacher" or item.split != "train":
                raise W02ContractError("W-02 teacher Evidence owner/split 漂移")
            teacher.append(W02PayloadBinding(
                relative,
                pack.pack_key,
                item.owner_kind,
                item.split,
                item.record_count,
                item.transport_size_bytes,
                item.transport_sha256,
                item.content_size_bytes,
                item.content_sha256,
                item,
            ))
    return tuple(sorted(packs)), tuple(sorted(candidate)), tuple(sorted(teacher))


def open_w02_frozen_context(
        repository_root: str | Path,
        global_manifest_path: str = D03_GLOBAL_MANIFEST_PATH,
        *,
        current_remote_commit_sha1: str,
        dependency_root: str | Path | None = None,
        ) -> W02FrozenContext:
    """回读 D-03/W-01 和 pack manifest，但在返回前绝不打开 payload gzip。"""
    primary = Path(repository_root).resolve()
    dependency = (
        Path(dependency_root).resolve()
        if dependency_root is not None else primary)
    commit_sha1 = sha1_text(
        current_remote_commit_sha1, where="current remote commit")
    try:
        d03_path = _overlay_file(primary, dependency, FORMAL_RECEIPT_PATH)
        d03_receipt = read_d03_publication_receipt(d03_path)
        if (d03_receipt.execution_state.get("d03_published") != 1
                or d03_receipt.publication_state.d03_published != 1):
            raise W02ContractError("D-03 publication receipt 未正式发布")
        reader = D03ReleaseReader.open(
            primary,
            global_manifest_path,
            dependency_root=dependency,
            require_publication=True,
        )
        w01_root = primary if (primary / W01_FORMAL_RECEIPT_PATH).is_file() else dependency
        w01 = read_w01_formal_receipt(w01_root)
        packs, candidate_bindings, teacher_bindings = _manifest_bindings(
            primary, dependency, reader)
    except W02ContractError:
        raise
    except (
            D03ContractError,
            DatasetContractError,
            W01ReceiptError,
            OSError,
            TypeError,
            ValueError,
            KeyError,
            ) as exc:
        raise W02ContractError(f"D-03/W-01/W-02 冻结输入无效: {exc}") from exc
    if d03_receipt.global_manifest_identity.relative_path != global_manifest_path:
        raise W02ContractError("D-03 receipt 与请求 global manifest 漂移")
    if (w01.execution_state.get("W01_PROTOCOL_VERIFIED") != 1
            or w01.execution_state.get("W02_STARTED") != 0
            or w01.execution_state.get("formal_training_runs") != 0
            or w01.execution_state.get("teacher_calls") != 0):
        raise W02ContractError("W-01 状态不允许启动 W-02")
    stage_index = 1
    stage = reader.stages[stage_index]
    stage_reference = reader.global_manifest.stage_manifests[stage_index]
    if stage_reference.artifact_key != W02_STAGE_KEY:
        raise W02ContractError("D-03 W-02 stage reference 漂移")
    candidate_view = reader.visibility(W02_STAGE_KEY, "candidate")
    teacher_view = reader.visibility(W02_STAGE_KEY, "teacher")
    evaluator_view = reader.visibility(W02_STAGE_KEY, "evaluator")
    if any(
            view.payload_reads != 0 or view.payload_bytes != 0
            for view in (candidate_view, teacher_view, evaluator_view)):
        raise W02ContractError("W-02 visibility 构造提前读取 payload")
    candidate_paths = tuple(
        item.relative_path for item in candidate_bindings)
    teacher_paths = tuple(
        item.relative_path for item in teacher_bindings)
    if candidate_paths != candidate_view.allowed_paths:
        raise W02ContractError("W-02 candidate manifest binding 与 reader 白名单漂移")
    teacher_only = tuple(sorted(
        set(teacher_view.allowed_paths) - set(candidate_view.allowed_paths)))
    if teacher_paths != teacher_only:
        raise W02ContractError("W-02 teacher manifest binding 与 reader 白名单漂移")
    evaluation = stage.evaluation_binding
    recovery = stage.recovery_binding
    identity = reader.global_manifest.release_identity
    return W02FrozenContext(
        current_remote_commit_sha1=commit_sha1,
        d03_release_key=d03_receipt.release_key,
        d03_published=1,
        d03_content_commit_sha1=d03_receipt.content_commit_sha1,
        d03_receipt_sha256=d03_receipt.sha256(),
        global_manifest_path=d03_receipt.global_manifest_identity.relative_path,
        global_manifest_sha256=d03_receipt.global_manifest_identity.sha256,
        stage_manifest_path=stage_reference.file_identity.relative_path,
        stage_manifest_sha256=stage_reference.file_identity.sha256,
        w01_receipt_path=W01_FORMAL_RECEIPT_PATH,
        w01_receipt_sha256=w01.sha256(),
        w01_status="W01_PROTOCOL_VERIFIED",
        w01_run_id=int(w01.formal_run["run_id"]),
        w01_cursor_next_stage=W02_STAGE_KEY,
        stage_key=stage.stage_identity.stage_key,
        stage_ordinal=stage.stage_identity.ordinal,
        prerequisite_stage_keys=stage.stage_identity.prerequisite_stage_keys,
        train_pack_keys=stage.data_visibility.train_pack_keys,
        pack_bindings=packs,
        candidate_payload_bindings=candidate_bindings,
        teacher_evidence_bindings=teacher_bindings,
        evaluator_visible_paths=evaluator_view.allowed_paths,
        evaluator_private_paths=tuple(sorted(
            set(evaluator_view.allowed_paths) - set(candidate_view.allowed_paths))),
        thresholds=evaluation.thresholds,
        dimension_keys=tuple(item.dimension_key for item in evaluation.thresholds),
        ablation_keys=evaluation.ablation_keys,
        aggregation_policy=evaluation.aggregation_policy,
        allowed_worker_counts=recovery.allowed_worker_counts,
        failure_point_keys=recovery.failure_point_keys,
        logical_shard_count=recovery.logical_shard_count,
        merge_barrier_key=recovery.merge_barrier_key,
        cursor_version=recovery.cursor_version,
        resource_budget=stage.resource_budget.to_dict(),
        version_keys=identity.version_keys,
    )


def validate_w02_request(
        context: W02FrozenContext,
        request: W02RunRequest,
        ) -> W02RunRequest:
    """在任一 payload get 或学习写之前校验 candidate 的完整入口身份。"""
    if not isinstance(context, W02FrozenContext):
        raise W02ContractError("W-02 context 类型错误")
    if not isinstance(request, W02RunRequest):
        raise W02ContractError("W-02 request 类型错误")
    if (type(request.run_id) is not int or request.run_id <= 0
            or type(request.parent_run_id) is not int
            or type(request.base_run_id) is not int
            or request.parent_run_id != context.w01_run_id
            or request.base_run_id != context.w01_run_id):
        raise W02ContractError("run/parent/base id 未从正式 W-01 续接")
    if request.stage_key != context.stage_key or request.stage_key != W02_STAGE_KEY:
        raise W02ContractError("W-02 是唯一允许的当前阶段")
    if request.owner_key != W02_OWNER_KEY:
        raise W02ContractError("W-02 transaction owner 未授权")
    if request.runner_key != W02_RUNNER_KEY:
        raise W02ContractError("必须使用独立 W-02 language stage runner")
    if request.current_remote_commit_sha1 != context.current_remote_commit_sha1:
        raise W02ContractError("current remote commit identity 漂移")
    if request.d03_context_key != context.stable_key():
        raise W02ContractError("W-02 请求 D-03 identity 漂移")
    if request.w01_receipt_sha256 != context.w01_receipt_sha256:
        raise W02ContractError("W-02 请求 W-01 receipt identity 漂移")
    _strict_key(request.backend_profile_key, label="backend profile key")
    _strict_key(request.base_fence_key, label="base fence key")
    if request.worker_count not in context.allowed_worker_counts:
        raise W02ContractError("worker count 不在 D-03 允许集合")
    if request.mode not in W02_ALLOWED_MODES:
        raise W02ContractError("mode 必须是 fresh/restart/resume")
    expected_candidate = tuple(
        item.relative_path for item in context.candidate_payload_bindings)
    expected_teacher = tuple(
        item.relative_path for item in context.teacher_evidence_bindings)
    for path in (*request.candidate_payload_paths, *request.teacher_evidence_paths):
        _relative_path(path, label="request payload path")
    if request.candidate_payload_paths != expected_candidate:
        raise W02ContractError("candidate payload path 不等于精确白名单")
    if request.teacher_evidence_paths != expected_teacher:
        raise W02ContractError("teacher Evidence path 不等于精确白名单")
    private = set(context.evaluator_private_paths)
    if private & set(request.candidate_payload_paths):
        raise W02ContractError("candidate 请求包含 evaluator 私有 path")
    if private & set(request.teacher_evidence_paths):
        raise W02ContractError("teacher 请求包含 evaluator 私有 path")
    return request


__all__ = [
    "D03_GLOBAL_MANIFEST_PATH",
    "W02_ABLATION_KEYS",
    "W02_ALLOWED_MODES",
    "W02_DIMENSION_KEYS",
    "W02_OWNER_KEY",
    "W02_RUNNER_KEY",
    "W02_STAGE_KEY",
    "W02ContractError",
    "W02FrozenContext",
    "W02PackBinding",
    "W02PayloadAudit",
    "W02PayloadBinding",
    "W02PayloadFirewall",
    "W02RunRequest",
    "W02TrainingPayload",
    "open_w02_frozen_context",
    "validate_w02_request",
]
