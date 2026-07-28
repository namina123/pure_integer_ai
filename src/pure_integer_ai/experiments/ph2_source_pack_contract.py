"""D-02 外部来源统一 pack 的纯输入、覆盖和冻结状态合同。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import (
    ALLOWED_LICENSE_IDS,
    REDISTRIBUTION_POLICIES,
    SAMPLE_ROLES,
    SPLITS,
    W_STAGES,
    CanonicalJsonObject,
    StableRecordKey,
    canonical_json_bytes,
    canonical_json_line,
)


SOURCE_PACK_CONTRACT_VERSION = 1
SOURCE_PACK_STATUSES = (
    "PACK_FROZEN",
    "PILOT_EVIDENCED",
    "BLOCKED",
)


class SourcePackContractError(ValueError):
    """外部来源 pack 的输入、组合轴或覆盖状态不满足冻结合同。"""


def _text(value: Any, *, where: str, allow_empty: bool = False) -> str:
    """要求文本类型正确，非空值不得含首尾空白。"""
    if not isinstance(value, str):
        raise SourcePackContractError(f"{where} 必须是字符串")
    if value and value.strip() != value:
        raise SourcePackContractError(f"{where} 不得含首尾空白")
    if not allow_empty and not value:
        raise SourcePackContractError(f"{where} 不能为空")
    return value


def _positive(value: Any, *, where: str) -> int:
    """要求版本和序号为正严格整数。"""
    if type(value) is not int or value <= 0:
        raise SourcePackContractError(f"{where} 必须是正严格整数")
    return value


def _sha256(value: Any, *, where: str) -> str:
    """要求小写 SHA-256。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(item not in "0123456789abcdef" for item in value)):
        raise SourcePackContractError(f"{where} 必须是小写 SHA-256")
    return value


def _checksum(value: Any, *, where: str) -> str:
    """要求带算法前缀的上游 checksum。"""
    text = _text(value, where=where)
    if ":" not in text:
        raise SourcePackContractError(f"{where} 缺少算法前缀")
    algorithm, digest = text.split(":", 1)
    if algorithm not in {"md5", "sha1", "sha256"}:
        raise SourcePackContractError(f"{where} 算法未注册")
    expected = {"md5": 32, "sha1": 40, "sha256": 64}[algorithm]
    if len(digest) != expected or any(
            item not in "0123456789abcdef" for item in digest):
        raise SourcePackContractError(f"{where} digest 非法")
    return text


def _relative_path(value: Any, *, where: str) -> str:
    """要求可公开携带的安全 POSIX 相对路径。"""
    text = _text(value, where=where)
    path = PurePosixPath(text)
    if (path.is_absolute() or ".." in path.parts or "\\" in text
            or path.as_posix() != text):
        raise SourcePackContractError(f"{where} 必须是安全 POSIX 相对路径")
    return text


def _parts(value: Any, *, where: str) -> tuple[Any, ...]:
    """要求 cluster 输入为非空规范值元组。"""
    if not isinstance(value, tuple) or not value:
        raise SourcePackContractError(f"{where} 不能为空")
    canonical_json_bytes(list(value))
    return value


def _exact_keys(value: Any, expected: set[str], *, where: str) -> dict[str, Any]:
    """要求恢复对象字段集合精确，拒绝环境字段和静默扩展。"""
    if not isinstance(value, dict) or set(value) != expected:
        raise SourcePackContractError(f"{where} 字段不精确")
    return value


def stable_source_pack_key(namespace: str, *parts: Any) -> StableRecordKey:
    """从完整规范身份生成版本化正整数 stable key。"""
    payload = canonical_json_bytes({
        "namespace": _text(namespace, where="stable key namespace"),
        "parts": list(parts),
        "version": SOURCE_PACK_CONTRACT_VERSION,
    })
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    value &= (1 << 63) - 1
    return StableRecordKey((1, value if value > 0 else 1))


@dataclass(frozen=True)
class SourcePackSpec:
    """冻结一个外部来源单许可 pack 的版本、来源和失效边界。"""

    source_key: str
    license_id: str
    redistribution_policy: str
    snapshot_id: str
    official_url: str
    attribution: str
    raw_snapshot_manifest_relative_path: str
    raw_snapshot_manifest_sha256: str
    course_version: int
    artifact_version: int
    adapter_version: int
    generator_version: int
    parser_version: int
    pack_name: str
    stage: str
    substage: str
    earliest_invalidated_stage: str

    def __post_init__(self) -> None:
        for name in (
                "source_key", "snapshot_id", "official_url", "attribution",
                "pack_name", "stage", "substage", "earliest_invalidated_stage"):
            _text(getattr(self, name), where=f"SourcePackSpec.{name}")
        if self.license_id not in ALLOWED_LICENSE_IDS:
            raise SourcePackContractError("SourcePackSpec.license_id 未注册")
        if self.redistribution_policy not in REDISTRIBUTION_POLICIES:
            raise SourcePackContractError(
                "SourcePackSpec.redistribution_policy 未注册")
        if (self.license_id.startswith("NOASSERTION")
                and self.redistribution_policy != "LOCAL_ONLY"):
            raise SourcePackContractError("许可不清来源不得公开发布")
        if not self.official_url.startswith(("https://", "urn:", "local:")):
            raise SourcePackContractError("SourcePackSpec.official_url scheme 非法")
        _relative_path(
            self.raw_snapshot_manifest_relative_path,
            where="SourcePackSpec.raw_snapshot_manifest_relative_path",
        )
        _sha256(
            self.raw_snapshot_manifest_sha256,
            where="SourcePackSpec.raw_snapshot_manifest_sha256",
        )
        for name in (
                "course_version", "artifact_version", "adapter_version",
                "generator_version", "parser_version"):
            _positive(getattr(self, name), where=f"SourcePackSpec.{name}")
        if self.stage not in W_STAGES:
            raise SourcePackContractError("SourcePackSpec.stage 未注册")
        if self.earliest_invalidated_stage not in W_STAGES:
            raise SourcePackContractError(
                "SourcePackSpec.earliest_invalidated_stage 未注册")

    def to_contract_dict(self) -> dict[str, Any]:
        """导出不含宿主路径或网络环境的规范输入合同。"""
        return {
            "adapter_version": self.adapter_version,
            "artifact_version": self.artifact_version,
            "attribution": self.attribution,
            "course_version": self.course_version,
            "earliest_invalidated_stage": self.earliest_invalidated_stage,
            "generator_version": self.generator_version,
            "license_id": self.license_id,
            "official_url": self.official_url,
            "pack_name": self.pack_name,
            "parser_version": self.parser_version,
            "raw_snapshot_manifest_relative_path": (
                self.raw_snapshot_manifest_relative_path),
            "raw_snapshot_manifest_sha256": self.raw_snapshot_manifest_sha256,
            "redistribution_policy": self.redistribution_policy,
            "snapshot_id": self.snapshot_id,
            "source_key": self.source_key,
            "stage": self.stage,
            "substage": self.substage,
        }


@dataclass(frozen=True)
class SourceObservationSeed:
    """由现有 adapter 核准、保留 raw Observation 的统一 pack 输入。"""

    seed_id: str
    split: str
    language: str
    representation: str
    source_identity: str
    upstream_checksum: str
    local_sha256: str
    source_span: CanonicalJsonObject
    raw_observation: CanonicalJsonObject
    combination_axes: CanonicalJsonObject
    source_cluster_parts: tuple[Any, ...]
    dedup_parts: tuple[Any, ...]
    content_parts: tuple[Any, ...]
    template_parts: tuple[Any, ...]
    shape_parts: tuple[Any, ...]
    combination_parts: tuple[Any, ...]
    sample_role: str
    perturbation_kind: str
    logical_order: int

    def __post_init__(self) -> None:
        for name in (
                "seed_id", "language", "representation", "source_identity",
                "perturbation_kind"):
            _text(getattr(self, name), where=f"SourceObservationSeed.{name}")
        if self.split not in SPLITS:
            raise SourcePackContractError("SourceObservationSeed.split 未注册")
        if self.sample_role not in SAMPLE_ROLES:
            raise SourcePackContractError("SourceObservationSeed.sample_role 未注册")
        _checksum(
            self.upstream_checksum,
            where="SourceObservationSeed.upstream_checksum",
        )
        _sha256(self.local_sha256, where="SourceObservationSeed.local_sha256")
        if not isinstance(self.source_span, CanonicalJsonObject):
            raise SourcePackContractError("SourceObservationSeed.source_span 类型错误")
        if not isinstance(self.raw_observation, CanonicalJsonObject):
            raise SourcePackContractError(
                "SourceObservationSeed.raw_observation 类型错误")
        if not isinstance(self.combination_axes, CanonicalJsonObject):
            raise SourcePackContractError(
                "SourceObservationSeed.combination_axes 类型错误")
        for name in (
                "source_cluster_parts", "dedup_parts", "content_parts",
                "template_parts", "shape_parts", "combination_parts"):
            _parts(getattr(self, name), where=f"SourceObservationSeed.{name}")
        _positive(self.logical_order, where="SourceObservationSeed.logical_order")

    @property
    def raw_observation_sha256(self) -> str:
        """返回完整 raw Observation 规范值 SHA-256。"""
        return hashlib.sha256(canonical_json_bytes(
            self.raw_observation.to_value())).hexdigest()

    def to_contract_dict(self) -> dict[str, Any]:
        """导出 batch 合同所需的输入身份，不复制 raw 正文。"""
        return {
            "combination_axes_sha256": hashlib.sha256(canonical_json_bytes(
                self.combination_axes.to_value())).hexdigest(),
            "combination_parts": list(self.combination_parts),
            "content_parts": list(self.content_parts),
            "dedup_parts": list(self.dedup_parts),
            "language": self.language,
            "local_sha256": self.local_sha256,
            "logical_order": self.logical_order,
            "perturbation_kind": self.perturbation_kind,
            "raw_observation_sha256": self.raw_observation_sha256,
            "representation": self.representation,
            "sample_role": self.sample_role,
            "seed_id": self.seed_id,
            "shape_parts": list(self.shape_parts),
            "source_cluster_parts": list(self.source_cluster_parts),
            "source_identity": self.source_identity,
            "source_span_sha256": hashlib.sha256(canonical_json_bytes(
                self.source_span.to_value())).hexdigest(),
            "split": self.split,
            "template_parts": list(self.template_parts),
            "upstream_checksum": self.upstream_checksum,
        }


@dataclass(frozen=True)
class SourcePackCoverageEntry:
    """登记一个正式许可分区的 pack、pilot 或明确 blocker。"""

    source_key: str
    license_partition: str
    status: str
    raw_snapshot_manifest_relative_path: str
    raw_snapshot_manifest_sha256: str
    pack_manifest_relative_path: str
    pack_manifest_sha256: str
    pack_record_count: int
    splits: tuple[str, ...]
    source_cluster_count: int
    combination_cluster_count: int
    blocker_code: str
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        _text(self.source_key, where="SourcePackCoverageEntry.source_key")
        if self.license_partition not in ALLOWED_LICENSE_IDS and (
                self.license_partition != "UNRESOLVED/BLOCKED"):
            raise SourcePackContractError("coverage license partition 未注册")
        if self.status not in SOURCE_PACK_STATUSES:
            raise SourcePackContractError("coverage status 未注册")
        _relative_path(
            self.raw_snapshot_manifest_relative_path,
            where="coverage raw snapshot path",
        )
        _sha256(
            self.raw_snapshot_manifest_sha256,
            where="coverage raw snapshot sha256",
        )
        if self.status == "PACK_FROZEN":
            _relative_path(
                self.pack_manifest_relative_path,
                where="coverage pack manifest path",
            )
            _sha256(
                self.pack_manifest_sha256,
                where="coverage pack manifest sha256",
            )
            _positive(self.pack_record_count, where="coverage pack record count")
            _positive(
                self.source_cluster_count,
                where="coverage source cluster count",
            )
            _positive(
                self.combination_cluster_count,
                where="coverage combination cluster count",
            )
            if self.blocker_code:
                raise SourcePackContractError("PACK_FROZEN 不得带 blocker")
        elif self.status == "BLOCKED":
            _text(self.blocker_code, where="coverage blocker code")
            if (self.pack_manifest_relative_path or self.pack_manifest_sha256
                    or self.pack_record_count or self.source_cluster_count
                    or self.combination_cluster_count):
                raise SourcePackContractError("BLOCKED 不得伪造 pack identity")
        else:
            if self.blocker_code:
                raise SourcePackContractError("PILOT_EVIDENCED 不得带 blocker")
        if not isinstance(self.splits, tuple) or any(
                item not in SPLITS for item in self.splits):
            raise SourcePackContractError("coverage splits 非法")
        if self.status == "PACK_FROZEN" and not self.splits:
            raise SourcePackContractError("PACK_FROZEN splits 不能为空")
        if not isinstance(self.evidence_refs, tuple) or not self.evidence_refs:
            raise SourcePackContractError("coverage evidence refs 不能为空")
        for item in self.evidence_refs:
            _relative_path(item, where="coverage evidence ref")

    def to_dict(self) -> dict[str, Any]:
        """导出一个来源许可分区的规范覆盖记录。"""
        return {
            "blocker_code": self.blocker_code,
            "combination_cluster_count": self.combination_cluster_count,
            "evidence_refs": list(self.evidence_refs),
            "license_partition": self.license_partition,
            "pack_manifest_relative_path": self.pack_manifest_relative_path,
            "pack_manifest_sha256": self.pack_manifest_sha256,
            "pack_record_count": self.pack_record_count,
            "raw_snapshot_manifest_relative_path": (
                self.raw_snapshot_manifest_relative_path),
            "raw_snapshot_manifest_sha256": self.raw_snapshot_manifest_sha256,
            "source_cluster_count": self.source_cluster_count,
            "source_key": self.source_key,
            "splits": list(self.splits),
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SourcePackCoverageEntry":
        """从精确 JSON object 恢复来源覆盖记录。"""
        raw = _exact_keys(value, {
            "blocker_code", "combination_cluster_count", "evidence_refs",
            "license_partition", "pack_manifest_relative_path",
            "pack_manifest_sha256", "pack_record_count",
            "raw_snapshot_manifest_relative_path",
            "raw_snapshot_manifest_sha256", "source_cluster_count",
            "source_key", "splits", "status",
        }, where="SourcePackCoverageEntry")
        return cls(
            str(raw["source_key"]),
            str(raw["license_partition"]),
            str(raw["status"]),
            str(raw["raw_snapshot_manifest_relative_path"]),
            str(raw["raw_snapshot_manifest_sha256"]),
            str(raw["pack_manifest_relative_path"]),
            str(raw["pack_manifest_sha256"]),
            raw["pack_record_count"],
            tuple(str(item) for item in raw["splits"]),
            raw["source_cluster_count"],
            raw["combination_cluster_count"],
            str(raw["blocker_code"]),
            tuple(str(item) for item in raw["evidence_refs"]),
        )


@dataclass(frozen=True)
class SourcePackCoverageManifest:
    """冻结 D-02 来源到统一 pack/blocker 的完整覆盖账。"""

    format_version: int
    artifact_version: str
    entries: tuple[SourcePackCoverageEntry, ...]
    d03_published: int = 0
    w01_started: int = 0
    formal_training_runs: int = 0
    teacher_calls: int = 0
    learning_state_writes: int = 0
    mastered_claims: int = 0
    readiness_claims: int = 0

    def __post_init__(self) -> None:
        _positive(self.format_version, where="coverage manifest format version")
        _text(self.artifact_version, where="coverage manifest artifact version")
        if not isinstance(self.entries, tuple) or not self.entries:
            raise SourcePackContractError("coverage manifest entries 不能为空")
        if any(not isinstance(item, SourcePackCoverageEntry)
               for item in self.entries):
            raise SourcePackContractError("coverage manifest entry 类型错误")
        identities = [
            (item.source_key, item.license_partition) for item in self.entries]
        if identities != sorted(identities) or len(identities) != len(set(identities)):
            raise SourcePackContractError("coverage entries 必须排序且身份唯一")
        for name in (
                "d03_published", "w01_started", "formal_training_runs",
                "teacher_calls", "learning_state_writes", "mastered_claims",
                "readiness_claims"):
            if getattr(self, name) != 0:
                raise SourcePackContractError(f"coverage {name} 必须为 0")

    def to_dict(self) -> dict[str, Any]:
        """导出规范覆盖 manifest。"""
        return {
            "artifact_kind": "PH2_D02_SOURCE_PACK_COVERAGE",
            "artifact_version": self.artifact_version,
            "d03_published": self.d03_published,
            "entries": [item.to_dict() for item in self.entries],
            "formal_training_runs": self.formal_training_runs,
            "format_version": self.format_version,
            "learning_state_writes": self.learning_state_writes,
            "mastered_claims": self.mastered_claims,
            "readiness_claims": self.readiness_claims,
            "teacher_calls": self.teacher_calls,
            "w01_started": self.w01_started,
        }

    def canonical_bytes(self) -> bytes:
        """返回单换行结尾的规范 artifact 字节。"""
        return canonical_json_line(self.to_dict())

    def sha256(self) -> str:
        """返回规范覆盖 manifest SHA-256。"""
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SourcePackCoverageManifest":
        """从精确 JSON object 恢复覆盖 manifest。"""
        raw = _exact_keys(value, {
            "artifact_kind", "artifact_version", "d03_published", "entries",
            "formal_training_runs", "format_version", "learning_state_writes",
            "mastered_claims", "readiness_claims", "teacher_calls",
            "w01_started",
        }, where="SourcePackCoverageManifest")
        if raw["artifact_kind"] != "PH2_D02_SOURCE_PACK_COVERAGE":
            raise SourcePackContractError("coverage artifact_kind 非法")
        return cls(
            raw["format_version"],
            str(raw["artifact_version"]),
            tuple(SourcePackCoverageEntry.from_dict(item)
                  for item in raw["entries"]),
            raw["d03_published"],
            raw["w01_started"],
            raw["formal_training_runs"],
            raw["teacher_calls"],
            raw["learning_state_writes"],
            raw["mastered_claims"],
            raw["readiness_claims"],
        )


__all__ = [
    "SOURCE_PACK_CONTRACT_VERSION",
    "SOURCE_PACK_STATUSES",
    "SourceObservationSeed",
    "SourcePackContractError",
    "SourcePackCoverageEntry",
    "SourcePackCoverageManifest",
    "SourcePackSpec",
    "stable_source_pack_key",
]
