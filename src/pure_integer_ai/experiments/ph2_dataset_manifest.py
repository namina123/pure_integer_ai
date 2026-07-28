"""PH2 单来源单许可 pack 的文件身份与 artifact manifest 合同。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, ClassVar

from pure_integer_ai.experiments.ph2_dataset_core import (
    JSONL_RECORD_KINDS,
    LOCAL_ONLY_LICENSE_IDS,
    OWNER_KINDS,
    RECORD_ARTIFACT_MANIFEST,
    REDISTRIBUTION_POLICIES,
    SPLITS,
    W_STAGES,
    DatasetContractError,
    StableRecordKey,
    _enum_text,
    _license_id,
    _nonempty_text,
    _nonnegative_int,
    _positive_int,
    _record_key_tuple,
    _sha256,
    canonical_json_line,
)


@dataclass(frozen=True, order=True)
class ArtifactFileIdentity:
    """一个规范 JSONL/gzip 文件的物理 owner、双 hash、计数和键范围。"""

    record_kind: str
    owner_kind: str
    relative_path: str
    split: str | None
    license_partition: str
    record_count: int
    content_sha256: str
    transport_sha256: str
    content_size_bytes: int
    transport_size_bytes: int
    first_record_key: StableRecordKey | None
    last_record_key: StableRecordKey | None
    source_cluster_keys: tuple[StableRecordKey, ...]

    def __post_init__(self) -> None:
        _enum_text(self.record_kind, JSONL_RECORD_KINDS, where="ArtifactFileIdentity.record_kind")
        _enum_text(self.owner_kind, OWNER_KINDS, where="ArtifactFileIdentity.owner_kind")
        path_text = _nonempty_text(
            self.relative_path, where="ArtifactFileIdentity.relative_path")
        path = PurePosixPath(path_text)
        if (path.is_absolute() or ".." in path.parts or "\\" in path_text
                or path_text != path.as_posix()):
            raise DatasetContractError("ArtifactFileIdentity.relative_path 必须是安全 POSIX 相对路径")
        if self.split is not None:
            _enum_text(self.split, SPLITS, where="ArtifactFileIdentity.split")
        object.__setattr__(self, "license_partition", _license_id(
            self.license_partition, where="ArtifactFileIdentity.license_partition"))
        _nonnegative_int(self.record_count, where="ArtifactFileIdentity.record_count")
        object.__setattr__(self, "content_sha256", _sha256(
            self.content_sha256, where="ArtifactFileIdentity.content_sha256"))
        object.__setattr__(self, "transport_sha256", _sha256(
            self.transport_sha256, where="ArtifactFileIdentity.transport_sha256"))
        _nonnegative_int(
            self.content_size_bytes,
            where="ArtifactFileIdentity.content_size_bytes",
        )
        _nonnegative_int(
            self.transport_size_bytes,
            where="ArtifactFileIdentity.transport_size_bytes",
        )
        if self.record_count == 0:
            if self.first_record_key is not None or self.last_record_key is not None:
                raise DatasetContractError("空 artifact 不得声明键范围")
        else:
            if (not isinstance(self.first_record_key, StableRecordKey)
                    or not isinstance(self.last_record_key, StableRecordKey)):
                raise DatasetContractError("非空 artifact 必须声明完整键范围")
            if self.first_record_key > self.last_record_key:
                raise DatasetContractError("artifact 首键不得大于末键")
        _record_key_tuple(
            self.source_cluster_keys,
            where="ArtifactFileIdentity.source_cluster_keys",
        )
        object.__setattr__(
            self, "source_cluster_keys", tuple(sorted(self.source_cluster_keys)))

    def to_dict(self) -> dict[str, Any]:
        """导出 ArtifactFileIdentity 的规范 JSON object。"""
        return {
            "content_sha256": self.content_sha256,
            "content_size_bytes": self.content_size_bytes,
            "first_record_key": (
                self.first_record_key.to_list()
                if self.first_record_key is not None else None
            ),
            "last_record_key": (
                self.last_record_key.to_list()
                if self.last_record_key is not None else None
            ),
            "license_partition": self.license_partition,
            "owner_kind": self.owner_kind,
            "record_count": self.record_count,
            "record_kind": self.record_kind,
            "relative_path": self.relative_path,
            "source_cluster_keys": [key.to_list() for key in self.source_cluster_keys],
            "split": self.split,
            "transport_sha256": self.transport_sha256,
            "transport_size_bytes": self.transport_size_bytes,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ArtifactFileIdentity":
        """从 manifest JSON object 恢复文件身份。"""
        raw_first = value["first_record_key"]
        raw_last = value["last_record_key"]
        return cls(
            str(value["record_kind"]),
            str(value["owner_kind"]),
            str(value["relative_path"]),
            str(value["split"]) if value["split"] is not None else None,
            str(value["license_partition"]),
            value["record_count"],
            str(value["content_sha256"]),
            str(value["transport_sha256"]),
            value["content_size_bytes"],
            value["transport_size_bytes"],
            (StableRecordKey.from_value(raw_first, where="file.first_record_key")
             if raw_first is not None else None),
            (StableRecordKey.from_value(raw_last, where="file.last_record_key")
             if raw_last is not None else None),
            tuple(StableRecordKey.from_value(item, where="file.source_cluster_key")
                  for item in value["source_cluster_keys"]),
        )


@dataclass(frozen=True)
class ArtifactManifest:
    """冻结一个单来源单许可 pack 的文件、版本、split、来源和失效边界。"""

    RECORD_KIND: ClassVar[str] = RECORD_ARTIFACT_MANIFEST

    format_version: int
    schema_version: int
    course_version: int
    artifact_version: int
    dataset_key: StableRecordKey
    stable_key: StableRecordKey
    source_key: str
    license_partition: str
    redistribution_policy: str
    adapter_version: int
    generator_version: int
    parser_version: int
    files: tuple[ArtifactFileIdentity, ...]
    splits: tuple[str, ...]
    w_stages: tuple[str, ...]
    source_cluster_keys: tuple[StableRecordKey, ...]
    prerequisite_manifest_keys: tuple[StableRecordKey, ...]
    earliest_invalidated_stage: str

    def __post_init__(self) -> None:
        for name, value in (
                ("format_version", self.format_version),
                ("schema_version", self.schema_version),
                ("course_version", self.course_version),
                ("artifact_version", self.artifact_version),
                ("adapter_version", self.adapter_version),
                ("generator_version", self.generator_version),
                ("parser_version", self.parser_version)):
            _positive_int(value, where=f"ArtifactManifest.{name}")
        if (not isinstance(self.dataset_key, StableRecordKey)
                or not isinstance(self.stable_key, StableRecordKey)):
            raise DatasetContractError("ArtifactManifest 身份键类型错误")
        _nonempty_text(self.source_key, where="ArtifactManifest.source_key")
        object.__setattr__(self, "license_partition", _license_id(
            self.license_partition, where="ArtifactManifest.license_partition"))
        _enum_text(
            self.redistribution_policy,
            REDISTRIBUTION_POLICIES,
            where="ArtifactManifest.redistribution_policy",
        )
        if (self.license_partition in LOCAL_ONLY_LICENSE_IDS
                and self.redistribution_policy != "LOCAL_ONLY"):
            raise DatasetContractError("许可不清 manifest 只能标 LOCAL_ONLY")
        if not isinstance(self.files, tuple) or not self.files:
            raise DatasetContractError("ArtifactManifest.files 不能为空")
        if any(not isinstance(item, ArtifactFileIdentity) for item in self.files):
            raise DatasetContractError("ArtifactManifest.files 类型错误")
        paths = [item.relative_path for item in self.files]
        if len(paths) != len(set(paths)):
            raise DatasetContractError("ArtifactManifest 文件路径重复")
        if any(item.license_partition != self.license_partition
               for item in self.files):
            raise DatasetContractError("ArtifactManifest 含混许可文件")
        object.__setattr__(
            self, "files", tuple(sorted(
                self.files, key=lambda item: item.relative_path)))
        if (not isinstance(self.splits, tuple) or not self.splits
                or len(set(self.splits)) != len(self.splits)):
            raise DatasetContractError("ArtifactManifest.splits 不能为空或重复")
        for split in self.splits:
            _enum_text(split, SPLITS, where="ArtifactManifest.split")
        object.__setattr__(
            self, "splits", tuple(sorted(self.splits, key=SPLITS.index)))
        if (not isinstance(self.w_stages, tuple) or not self.w_stages
                or len(set(self.w_stages)) != len(self.w_stages)):
            raise DatasetContractError("ArtifactManifest.w_stages 不能为空或重复")
        for stage in self.w_stages:
            _enum_text(stage, W_STAGES, where="ArtifactManifest.w_stage")
        object.__setattr__(
            self, "w_stages", tuple(sorted(self.w_stages, key=W_STAGES.index)))
        _record_key_tuple(
            self.source_cluster_keys,
            where="ArtifactManifest.source_cluster_keys",
        )
        object.__setattr__(
            self, "source_cluster_keys", tuple(sorted(self.source_cluster_keys)))
        _record_key_tuple(
            self.prerequisite_manifest_keys,
            where="ArtifactManifest.prerequisite_manifest_keys",
            allow_empty=True,
        )
        object.__setattr__(
            self,
            "prerequisite_manifest_keys",
            tuple(sorted(self.prerequisite_manifest_keys)),
        )
        _enum_text(
            self.earliest_invalidated_stage,
            W_STAGES,
            where="ArtifactManifest.earliest_invalidated_stage",
        )

    @property
    def record_count(self) -> int:
        """返回全部 JSONL 文件声明记录数之和。"""
        return sum(item.record_count for item in self.files)

    def to_dict(self) -> dict[str, Any]:
        """导出 ArtifactManifest 的规范 JSON object。"""
        return {
            "adapter_version": self.adapter_version,
            "artifact_version": self.artifact_version,
            "course_version": self.course_version,
            "dataset_key": self.dataset_key.to_list(),
            "earliest_invalidated_stage": self.earliest_invalidated_stage,
            "files": [item.to_dict() for item in self.files],
            "format_version": self.format_version,
            "generator_version": self.generator_version,
            "license_partition": self.license_partition,
            "parser_version": self.parser_version,
            "prerequisite_manifest_keys": [
                key.to_list() for key in self.prerequisite_manifest_keys
            ],
            "record_count": self.record_count,
            "record_kind": self.RECORD_KIND,
            "redistribution_policy": self.redistribution_policy,
            "schema_version": self.schema_version,
            "source_cluster_keys": [key.to_list() for key in self.source_cluster_keys],
            "source_key": self.source_key,
            "splits": list(self.splits),
            "stable_key": self.stable_key.to_list(),
            "w_stages": list(self.w_stages),
        }

    def canonical_bytes(self) -> bytes:
        """返回以单个换行结束的 manifest 规范 UTF-8 字节。"""
        return canonical_json_line(self.to_dict())

    def sha256(self) -> str:
        """返回完整 manifest 规范字节 SHA-256。"""
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ArtifactManifest":
        """从已解析 JSON object 恢复 manifest 并核对总记录数。"""
        manifest = cls(
            value["format_version"],
            value["schema_version"],
            value["course_version"],
            value["artifact_version"],
            StableRecordKey.from_value(value["dataset_key"], where="manifest.dataset_key"),
            StableRecordKey.from_value(value["stable_key"], where="manifest.stable_key"),
            str(value["source_key"]),
            str(value["license_partition"]),
            str(value["redistribution_policy"]),
            value["adapter_version"],
            value["generator_version"],
            value["parser_version"],
            tuple(ArtifactFileIdentity.from_dict(item) for item in value["files"]),
            tuple(str(item) for item in value["splits"]),
            tuple(str(item) for item in value["w_stages"]),
            tuple(StableRecordKey.from_value(item, where="manifest.source_cluster_key")
                  for item in value["source_cluster_keys"]),
            tuple(StableRecordKey.from_value(
                item, where="manifest.prerequisite_manifest_key")
                  for item in value["prerequisite_manifest_keys"]),
            str(value["earliest_invalidated_stage"]),
        )
        if value.get("record_count") != manifest.record_count:
            raise DatasetContractError("ArtifactManifest.record_count 与文件求和不一致")
        return manifest


__all__ = ["ArtifactFileIdentity", "ArtifactManifest"]
