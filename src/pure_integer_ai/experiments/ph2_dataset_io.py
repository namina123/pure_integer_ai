"""PH2 统一资料的原子规范 JSONL/gzip 和 manifest 读写边界。"""
from __future__ import annotations

import gzip
import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable

from pure_integer_ai.experiments.ph2_dataset_contract import (
    ALLOWED_LICENSE_IDS,
    JSONL_RECORD_KINDS,
    OWNER_KINDS,
    RECORD_EVALUATOR_LABEL,
    RECORD_OBSERVATION,
    RECORD_SOURCE_REF,
    RECORD_TEACHER_EVIDENCE,
    SPLITS,
    ArtifactFileIdentity,
    ArtifactManifest,
    DatasetContractError,
    DatasetRecord,
    EvaluatorLabelRecord,
    ObservationRecord,
    SourceRefRecord,
    StableRecordKey,
    TeacherEvidenceRecord,
    canonical_json_line,
    parse_canonical_json_bytes,
    record_from_dict,
    record_kind,
)


JsonlRecord = (
    SourceRefRecord
    | ObservationRecord
    | TeacherEvidenceRecord
    | EvaluatorLabelRecord
)


class DatasetArtifactIOError(RuntimeError):
    """资料 artifact 路径、规范编码、双 hash 或记录范围不一致。"""


def _sha256_file(path: Path) -> str:
    """以固定块大小流式计算文件 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _safe_target(root: Path, relative_path: str) -> Path:
    """把安全 POSIX 相对路径解析到 root 内并拒绝路径逃逸。"""
    pure = PurePosixPath(relative_path)
    if (pure.is_absolute() or ".." in pure.parts or "\\" in relative_path
            or pure.as_posix() != relative_path):
        raise DatasetArtifactIOError("artifact 路径必须是安全 POSIX 相对路径")
    resolved_root = root.resolve()
    target = (resolved_root / Path(*pure.parts)).resolve()
    if not target.is_relative_to(resolved_root):
        raise DatasetArtifactIOError("artifact 路径逃逸 root")
    return target


def _validate_owner_path(
        record_kind_value: str,
        owner_kind: str,
        relative_path: str,
        split: str | None) -> None:
    """冻结 Observation、teacher 和 evaluator 的物理 owner 目录边界。"""
    if record_kind_value == RECORD_SOURCE_REF:
        if owner_kind != "source" or relative_path != "source_refs.jsonl.gz" or split is not None:
            raise DatasetArtifactIOError("SourceRef 必须独占 source_refs.jsonl.gz")
        return
    if record_kind_value == RECORD_OBSERVATION:
        if owner_kind == "observation":
            if split is None or relative_path != f"observations/{split}.jsonl.gz":
                raise DatasetArtifactIOError("Observation 路径与 split 不一致")
            return
        if owner_kind == "anomaly":
            if not relative_path.startswith("anomalies/"):
                raise DatasetArtifactIOError("anomaly Observation 必须位于 anomalies/")
            return
        raise DatasetArtifactIOError("Observation owner_kind 非法")
    if record_kind_value == RECORD_TEACHER_EVIDENCE:
        if owner_kind != "teacher" or not relative_path.startswith("owners/teacher/"):
            raise DatasetArtifactIOError("TeacherEvidence 必须位于 owners/teacher/")
        return
    if record_kind_value == RECORD_EVALUATOR_LABEL:
        if owner_kind != "evaluator" or not relative_path.startswith("owners/evaluator/"):
            raise DatasetArtifactIOError("EvaluatorLabel 必须位于 owners/evaluator/")
        return
    raise DatasetArtifactIOError("未知 JSONL record kind")


@dataclass(frozen=True)
class ArtifactWriteSpec:
    """声明一个待写 JSONL/gzip 文件的物理路径、owner、split 和许可。"""

    record_kind: str
    owner_kind: str
    relative_path: str
    split: str | None
    license_partition: str
    source_cluster_keys: tuple[StableRecordKey, ...]

    def __post_init__(self) -> None:
        if self.record_kind not in JSONL_RECORD_KINDS:
            raise DatasetArtifactIOError("ArtifactWriteSpec.record_kind 非法")
        if self.owner_kind not in OWNER_KINDS:
            raise DatasetArtifactIOError("ArtifactWriteSpec.owner_kind 非法")
        if self.split is not None and self.split not in SPLITS:
            raise DatasetArtifactIOError("ArtifactWriteSpec.split 非法")
        if self.license_partition not in ALLOWED_LICENSE_IDS:
            raise DatasetArtifactIOError("ArtifactWriteSpec.license_partition 非法")
        if not isinstance(self.source_cluster_keys, tuple) or not self.source_cluster_keys:
            raise DatasetArtifactIOError("ArtifactWriteSpec.source_cluster_keys 不能为空")
        if any(not isinstance(key, StableRecordKey) for key in self.source_cluster_keys):
            raise DatasetArtifactIOError("ArtifactWriteSpec.source_cluster_keys 类型错误")
        if len(set(self.source_cluster_keys)) != len(self.source_cluster_keys):
            raise DatasetArtifactIOError("ArtifactWriteSpec.source_cluster_keys 不得重复")
        _validate_owner_path(
            self.record_kind,
            self.owner_kind,
            self.relative_path,
            self.split,
        )


def write_record_artifact(
        records: Iterable[JsonlRecord],
        root: str | Path,
        spec: ArtifactWriteSpec) -> ArtifactFileIdentity:
    """按 stable key 排序并原子写 deterministic gzip，同时返回双 hash 身份。"""
    record_tuple = tuple(records)
    for item in record_tuple:
        if record_kind(item) != spec.record_kind:
            raise DatasetArtifactIOError("artifact 混入其他 record kind")
        if (spec.owner_kind == "anomaly"
                and (not isinstance(item, ObservationRecord)
                     or item.sample_role != "anomaly")):
            raise DatasetArtifactIOError("anomalies/ 只能写 sample_role=anomaly")
        if (isinstance(item, ObservationRecord) and spec.split is not None
                and item.split != spec.split):
            raise DatasetArtifactIOError("Observation split 与文件路径不一致")
    ordered = tuple(sorted(record_tuple, key=lambda item: item.stable_key))
    keys = [item.stable_key for item in ordered]
    if len(keys) != len(set(keys)):
        raise DatasetArtifactIOError("artifact stable key 重复")

    output_root = Path(root).resolve()
    target = _safe_target(output_root, spec.relative_path)
    if target.exists():
        raise DatasetArtifactIOError("artifact 目标已存在，禁止覆盖")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_handle = tempfile.NamedTemporaryFile(
        prefix=f".{target.name}.building-",
        dir=target.parent,
        delete=False,
    )
    temporary = Path(temporary_handle.name)
    temporary_handle.close()
    content_digest = hashlib.sha256()
    content_size = 0
    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(
                    filename="", mode="wb", fileobj=raw, mtime=0) as stream:
                for item in ordered:
                    payload = canonical_json_line(item.to_dict())
                    content_digest.update(payload)
                    content_size += len(payload)
                    stream.write(payload)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return ArtifactFileIdentity(
        spec.record_kind,
        spec.owner_kind,
        spec.relative_path,
        spec.split,
        spec.license_partition,
        len(ordered),
        content_digest.hexdigest(),
        _sha256_file(target),
        content_size,
        target.stat().st_size,
        keys[0] if keys else None,
        keys[-1] if keys else None,
        tuple(sorted(set(spec.source_cluster_keys))),
    )


def read_record_artifact(
        root: str | Path,
        identity: ArtifactFileIdentity) -> tuple[JsonlRecord, ...]:
    """核验 transport 后解压规范 JSONL，并复核 content hash、计数和键范围。"""
    _validate_owner_path(
        identity.record_kind,
        identity.owner_kind,
        identity.relative_path,
        identity.split,
    )
    path = _safe_target(Path(root), identity.relative_path)
    if not path.is_file() or path.stat().st_size != identity.transport_size_bytes:
        raise DatasetArtifactIOError("artifact 缺失或 transport size 变化")
    if _sha256_file(path) != identity.transport_sha256:
        raise DatasetArtifactIOError("artifact transport SHA-256 变化")
    content_digest = hashlib.sha256()
    content_size = 0
    records: list[JsonlRecord] = []
    try:
        with path.open("rb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="rb") as stream:
                for line_number, line in enumerate(stream, start=1):
                    if not line.endswith(b"\n") or line.endswith(b"\n\n"):
                        raise DatasetArtifactIOError(
                            f"artifact 第 {line_number} 行换行非法")
                    content_digest.update(line)
                    content_size += len(line)
                    value = parse_canonical_json_bytes(
                        line[:-1], require_object=True)
                    assert isinstance(value, dict)
                    record = record_from_dict(value)
                    if record_kind(record) != identity.record_kind:
                        raise DatasetArtifactIOError("artifact record kind 漂移")
                    if not isinstance(record, (
                            SourceRefRecord,
                            ObservationRecord,
                            TeacherEvidenceRecord,
                            EvaluatorLabelRecord)):
                        raise DatasetArtifactIOError("artifact 含 manifest record")
                    records.append(record)
    except (OSError, EOFError, DatasetContractError) as error:
        if isinstance(error, DatasetArtifactIOError):
            raise
        raise DatasetArtifactIOError("artifact gzip/JSONL 内容损坏") from error
    keys = [item.stable_key for item in records]
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        raise DatasetArtifactIOError("artifact 键序或唯一性变化")
    if len(records) != identity.record_count:
        raise DatasetArtifactIOError("artifact record_count 变化")
    if content_size != identity.content_size_bytes:
        raise DatasetArtifactIOError("artifact content size 变化")
    if content_digest.hexdigest() != identity.content_sha256:
        raise DatasetArtifactIOError("artifact content SHA-256 变化")
    first = keys[0] if keys else None
    last = keys[-1] if keys else None
    if first != identity.first_record_key or last != identity.last_record_key:
        raise DatasetArtifactIOError("artifact key range 变化")
    return tuple(records)


def write_artifact_manifest(
        manifest: ArtifactManifest,
        root: str | Path,
        *,
        relative_path: str = "manifest.json") -> Path:
    """原子独占写规范 manifest，既有路径必须换 artifact 版本。"""
    output_root = Path(root).resolve()
    target = _safe_target(output_root, relative_path)
    if target.exists():
        raise DatasetArtifactIOError("manifest 已存在，禁止覆盖")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_handle = tempfile.NamedTemporaryFile(
        prefix=f".{target.name}.building-",
        dir=target.parent,
        delete=False,
    )
    temporary = Path(temporary_handle.name)
    try:
        temporary_handle.write(manifest.canonical_bytes())
        temporary_handle.close()
        os.replace(temporary, target)
    finally:
        if not temporary_handle.closed:
            temporary_handle.close()
        if temporary.exists():
            temporary.unlink()
    return target


def read_artifact_manifest(path: str | Path) -> ArtifactManifest:
    """严格读取规范 manifest 并拒绝非规范编码或总计数漂移。"""
    manifest_path = Path(path)
    try:
        payload = manifest_path.read_bytes()
    except OSError as error:
        raise DatasetArtifactIOError("manifest 无法读取") from error
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise DatasetArtifactIOError("manifest 必须以单个换行结束")
    try:
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
        assert isinstance(value, dict)
        manifest = ArtifactManifest.from_dict(value)
    except DatasetContractError as error:
        raise DatasetArtifactIOError("manifest 合同或规范编码损坏") from error
    if manifest.canonical_bytes() != payload:
        raise DatasetArtifactIOError("manifest 规范字节不一致")
    return manifest


__all__ = [
    "ArtifactWriteSpec",
    "DatasetArtifactIOError",
    "JsonlRecord",
    "read_artifact_manifest",
    "read_record_artifact",
    "write_artifact_manifest",
    "write_record_artifact",
]
