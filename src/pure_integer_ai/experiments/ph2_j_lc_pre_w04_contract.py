"""J-LC-PRE-W04 的严格、append-only 公开合取 gate 合同。"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_typed_carrier_pack_contract import (
    IN_SCOPE_CARRIER_KEYS,
)


FORMAT_VERSION = 1
ARTIFACT_KIND = "PH2_J_LC_PRE_W04_GATE"
ARTIFACT_VERSION = "J-LC-PRE-W04-20260801-A"
ARTIFACT_STATUS = "PASS"
MANIFEST_PATH = "data/ph2/manifests/j_lc_pre_w04_v1.json"
PARENT_HEAD_SHA1 = "20f13f783d7379485760094a57e28f952485a482"
ORIGINAL_W02_RECEIPT_SHA256 = (
    "6b1344bfb226ea2488760987a838b4a7d4016f14831d6ed58c78b9ff0e45a2eb")

DEPENDENCY_ROLES = (
    "LC_COVERAGE_V2",
    "TYPED_CARRIER_PARENT",
    "CARRIER_NEUTRAL_MAPPER",
    "SHARED_PROJECTION_RUNTIME",
    "DIRECTIONAL_RUNTIME",
    "D03_LC16_OVERLAY",
    "ORIGINAL_W03_RECEIPT_WITH_W02_COMMITMENT",
    "W02_LC16_SUPPLEMENTAL_PASS_RECEIPT",
    "W03_LC16_SUPPLEMENTAL_PASS_RECEIPT",
)
EVIDENCE_ROLES = ("CATALOG", "CONTRACT", "TEST")
W04_BLOCKING_FAILURE_KEYS = (
    "PARENT_DIRECTIONAL_IDENTITY_DRIFT",
    "TYPED_COURSE_OR_ADAPTER_IDENTITY_DRIFT",
    "W02_SUPPLEMENTAL_FAIL_OR_NE",
    "W03_SUPPLEMENTAL_FAIL_OR_NE",
)
PUBLISHED_STATE = {
    "J_LC_PRE_W04": "PASS",
    "LANGUAGE_CAPABILITY_MASTERED": 0,
    "LANGUAGE_READINESS": 0,
    "W04_ALLOWED": 1,
    "W04_STARTED": 0,
}
SUPPLEMENTAL_RECEIPT_STATUSES = {
    "W02_LC16_SUPPLEMENTAL": "PASS",
    "W03_LC16_SUPPLEMENTAL": "PASS",
}
OPEN_GENERATION_SUFFIX = ("W-08", "W-09", "J-LC-W09", "J-F2")


class JLcPreW04Error(RuntimeError):
    """pre-W04 gate identity、语义或 append-only 条件不闭合。"""


def _exact(value: Any, keys: set[str], *, where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise JLcPreW04Error(f"{where} 字段不精确")
    return value


def _sha256(value: Any, *, where: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(item not in "0123456789abcdef" for item in value)):
        raise JLcPreW04Error(f"{where} 必须是小写 SHA-256")
    return value


def _sha1(value: Any, *, where: str) -> str:
    if (not isinstance(value, str) or len(value) != 40
            or any(item not in "0123456789abcdef" for item in value)):
        raise JLcPreW04Error(f"{where} 必须是小写 SHA-1")
    return value


def _relative(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise JLcPreW04Error(f"{where} 相对路径非法")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        raise JLcPreW04Error(f"{where} 相对路径越界")
    return value


def _positive(value: Any, *, where: str) -> int:
    if type(value) is not int or value <= 0:
        raise JLcPreW04Error(f"{where} 必须是正整数")
    return value


@dataclass(frozen=True, order=True)
class GateFileIdentity:
    """一个必须逐字节相等的公开文件身份。"""

    role: str
    relative_path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.role, str) or not self.role:
            raise JLcPreW04Error("file identity role 非法")
        _relative(self.relative_path, where=self.role)
        _positive(self.size_bytes, where=f"{self.role}.size_bytes")
        _sha256(self.sha256, where=f"{self.role}.sha256")

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "role": self.role,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "GateFileIdentity":
        raw = _exact(value, {
            "relative_path", "role", "sha256", "size_bytes",
        }, where="GateFileIdentity")
        return cls(
            str(raw["role"]), str(raw["relative_path"]),
            raw["size_bytes"], str(raw["sha256"]),
        )


@dataclass(frozen=True, order=True)
class GateCarrierBinding:
    """一个 carrier 的当前 adapter manifest 与原创 sample 身份。"""

    carrier_key: str
    manifest_identity: GateFileIdentity
    sample_identity: GateFileIdentity

    def __post_init__(self) -> None:
        if self.carrier_key not in IN_SCOPE_CARRIER_KEYS:
            raise JLcPreW04Error("carrier binding key 非法")
        if (self.manifest_identity.role != "CARRIER_MANIFEST"
                or self.sample_identity.role != "CARRIER_SAMPLE"):
            raise JLcPreW04Error("carrier binding role 漂移")

    def to_dict(self) -> dict[str, Any]:
        return {
            "carrier_key": self.carrier_key,
            "manifest_identity": self.manifest_identity.to_dict(),
            "sample_identity": self.sample_identity.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "GateCarrierBinding":
        raw = _exact(value, {
            "carrier_key", "manifest_identity", "sample_identity",
        }, where="GateCarrierBinding")
        return cls(
            str(raw["carrier_key"]),
            GateFileIdentity.from_dict(raw["manifest_identity"]),
            GateFileIdentity.from_dict(raw["sample_identity"]),
        )


@dataclass(frozen=True, order=True)
class OriginalW02ReceiptCommitment:
    """原 W-02 receipt 在原 W-03 receipt 字节中的 canonical commitment。"""

    source_relative_path: str
    source_sha256: str
    json_field: str
    commitment_sha256: str

    def __post_init__(self) -> None:
        _relative(self.source_relative_path, where="original W-02 source")
        _sha256(self.source_sha256, where="original W-02 source SHA")
        if self.json_field != "w02_receipt_sha256":
            raise JLcPreW04Error("原 W-02 receipt commitment 字段漂移")
        if self.commitment_sha256 != ORIGINAL_W02_RECEIPT_SHA256:
            raise JLcPreW04Error("原 W-02 receipt commitment 漂移")

    def to_dict(self) -> dict[str, Any]:
        return {
            "commitment_sha256": self.commitment_sha256,
            "json_field": self.json_field,
            "source_relative_path": self.source_relative_path,
            "source_sha256": self.source_sha256,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "OriginalW02ReceiptCommitment":
        raw = _exact(value, {
            "commitment_sha256", "json_field", "source_relative_path",
            "source_sha256",
        }, where="OriginalW02ReceiptCommitment")
        return cls(
            str(raw["source_relative_path"]), str(raw["source_sha256"]),
            str(raw["json_field"]), str(raw["commitment_sha256"]),
        )


@dataclass(frozen=True, order=True)
class OpenGenerationBoundary:
    """开放生成 NE 只传播 W-08+，不得被 replay 覆盖。"""

    current_status: str
    failure_suffix: tuple[str, ...]
    runtime_evidenced: int
    included_in_current_directional_evidence: int
    aggregate_with_source_replay: int
    blocks_w04: int

    def __post_init__(self) -> None:
        if (self.current_status != "NE_NOT_YET_EVALUABLE"
                or self.failure_suffix != OPEN_GENERATION_SUFFIX
                or self.runtime_evidenced != 0
                or self.included_in_current_directional_evidence != 0
                or self.aggregate_with_source_replay != 0
                or self.blocks_w04 != 0
                or "W-04" in self.failure_suffix):
            raise JLcPreW04Error("open generation 边界漂移")

    def to_dict(self) -> dict[str, Any]:
        return {
            "aggregate_with_source_replay": self.aggregate_with_source_replay,
            "blocks_w04": self.blocks_w04,
            "current_status": self.current_status,
            "failure_suffix": list(self.failure_suffix),
            "included_in_current_directional_evidence": (
                self.included_in_current_directional_evidence),
            "runtime_evidenced": self.runtime_evidenced,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "OpenGenerationBoundary":
        raw = _exact(value, {
            "aggregate_with_source_replay", "blocks_w04", "current_status",
            "failure_suffix", "included_in_current_directional_evidence",
            "runtime_evidenced",
        }, where="OpenGenerationBoundary")
        return cls(
            str(raw["current_status"]),
            tuple(str(item) for item in raw["failure_suffix"]),
            raw["runtime_evidenced"],
            raw["included_in_current_directional_evidence"],
            raw["aggregate_with_source_replay"], raw["blocks_w04"],
        )


@dataclass(frozen=True)
class JLcPreW04Gate:
    """只允许发布 W-04 许可，不启动 W-04 或声明能力成熟。"""

    format_version: int
    artifact_kind: str
    artifact_version: str
    artifact_status: str
    parent_head_sha1: str
    dependencies: tuple[GateFileIdentity, ...]
    carrier_bindings: tuple[GateCarrierBinding, ...]
    evidence_files: tuple[GateFileIdentity, ...]
    original_w02_receipt: OriginalW02ReceiptCommitment
    supplemental_receipt_statuses: dict[str, str]
    w04_blocking_failure_keys: tuple[str, ...]
    resolved_w04_blocking_failure_keys: tuple[str, ...]
    unresolved_w04_blocking_failure_keys: tuple[str, ...]
    open_generation: OpenGenerationBoundary
    published_state: dict[str, Any]

    def __post_init__(self) -> None:
        if (self.format_version != FORMAT_VERSION
                or self.artifact_kind != ARTIFACT_KIND
                or self.artifact_version != ARTIFACT_VERSION
                or self.artifact_status != ARTIFACT_STATUS):
            raise JLcPreW04Error("pre-W04 gate artifact identity 漂移")
        if _sha1(self.parent_head_sha1, where="parent head") != PARENT_HEAD_SHA1:
            raise JLcPreW04Error("pre-W04 parent head 漂移")
        if tuple(item.role for item in self.dependencies) != DEPENDENCY_ROLES:
            raise JLcPreW04Error("pre-W04 dependency role 顺序漂移")
        if tuple(item.carrier_key for item in self.carrier_bindings) \
                != IN_SCOPE_CARRIER_KEYS:
            raise JLcPreW04Error("pre-W04 九 carrier binding 未闭合")
        if tuple(item.role for item in self.evidence_files) != EVIDENCE_ROLES:
            raise JLcPreW04Error("pre-W04 evidence role 顺序漂移")
        original_w03 = self.dependencies[DEPENDENCY_ROLES.index(
            "ORIGINAL_W03_RECEIPT_WITH_W02_COMMITMENT")]
        if (self.original_w02_receipt.source_relative_path
                != original_w03.relative_path
                or self.original_w02_receipt.source_sha256
                != original_w03.sha256):
            raise JLcPreW04Error("原 W-02 commitment 未绑定原 W-03 receipt bytes")
        if self.supplemental_receipt_statuses != SUPPLEMENTAL_RECEIPT_STATUSES:
            raise JLcPreW04Error("两道 supplemental receipt 非全 PASS")
        if (self.w04_blocking_failure_keys != W04_BLOCKING_FAILURE_KEYS
                or self.resolved_w04_blocking_failure_keys
                != W04_BLOCKING_FAILURE_KEYS
                or self.unresolved_w04_blocking_failure_keys != ()):
            raise JLcPreW04Error("存在影响 W-04 的未处理 failure")
        if self.published_state != PUBLISHED_STATE:
            raise JLcPreW04Error("pre-W04 只能发布冻结的五项状态")

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": self.artifact_kind,
            "artifact_status": self.artifact_status,
            "artifact_version": self.artifact_version,
            "carrier_bindings": [item.to_dict() for item in self.carrier_bindings],
            "dependencies": [item.to_dict() for item in self.dependencies],
            "evidence_files": [item.to_dict() for item in self.evidence_files],
            "format_version": self.format_version,
            "open_generation": self.open_generation.to_dict(),
            "original_w02_receipt": self.original_w02_receipt.to_dict(),
            "parent_head_sha1": self.parent_head_sha1,
            "published_state": dict(sorted(self.published_state.items())),
            "resolved_w04_blocking_failure_keys": list(
                self.resolved_w04_blocking_failure_keys),
            "supplemental_receipt_statuses": dict(sorted(
                self.supplemental_receipt_statuses.items())),
            "unresolved_w04_blocking_failure_keys": list(
                self.unresolved_w04_blocking_failure_keys),
            "w04_blocking_failure_keys": list(self.w04_blocking_failure_keys),
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict()) + b"\n"

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, value: Any) -> "JLcPreW04Gate":
        raw = _exact(value, {
            "artifact_kind", "artifact_status", "artifact_version",
            "carrier_bindings", "dependencies", "evidence_files",
            "format_version", "open_generation", "original_w02_receipt",
            "parent_head_sha1", "published_state",
            "resolved_w04_blocking_failure_keys",
            "supplemental_receipt_statuses",
            "unresolved_w04_blocking_failure_keys",
            "w04_blocking_failure_keys",
        }, where="JLcPreW04Gate")
        return cls(
            raw["format_version"], str(raw["artifact_kind"]),
            str(raw["artifact_version"]), str(raw["artifact_status"]),
            str(raw["parent_head_sha1"]),
            tuple(GateFileIdentity.from_dict(item)
                  for item in raw["dependencies"]),
            tuple(GateCarrierBinding.from_dict(item)
                  for item in raw["carrier_bindings"]),
            tuple(GateFileIdentity.from_dict(item)
                  for item in raw["evidence_files"]),
            OriginalW02ReceiptCommitment.from_dict(
                raw["original_w02_receipt"]),
            {str(key): str(item) for key, item
             in raw["supplemental_receipt_statuses"].items()},
            tuple(str(item) for item in raw["w04_blocking_failure_keys"]),
            tuple(str(item) for item
                  in raw["resolved_w04_blocking_failure_keys"]),
            tuple(str(item) for item
                  in raw["unresolved_w04_blocking_failure_keys"]),
            OpenGenerationBoundary.from_dict(raw["open_generation"]),
            dict(raw["published_state"]),
        )


def read_j_lc_pre_w04_gate(path: str | Path) -> JLcPreW04Gate:
    """严格回读 canonical gate。"""
    target = Path(path)
    try:
        payload = target.read_bytes()
        if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
            raise JLcPreW04Error("pre-W04 gate newline 非法")
        gate = JLcPreW04Gate.from_dict(
            parse_canonical_json_bytes(payload[:-1], require_object=True))
    except JLcPreW04Error:
        raise
    except Exception as error:
        raise JLcPreW04Error("pre-W04 gate 无法回读") from error
    if gate.canonical_bytes() != payload:
        raise JLcPreW04Error("pre-W04 gate 非 canonical bytes")
    return gate


def write_j_lc_pre_w04_gate(
        gate: JLcPreW04Gate,
        path: str | Path,
        ) -> Path:
    """append-only 创建 gate；既存目标即使同 bytes 也拒绝。"""
    if not isinstance(gate, JLcPreW04Gate):
        raise JLcPreW04Error("pre-W04 gate 类型非法")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(gate.canonical_bytes())
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise JLcPreW04Error("pre-W04 gate 已存在，禁止覆盖或重发") from error
    return target


def verify_j_lc_pre_w04_files(
        gate: JLcPreW04Gate,
        *,
        repository_root: str | Path,
        ) -> None:
    """逐文件回验全部 parent、九载体与 gate evidence identity。"""
    root = Path(repository_root).resolve()
    identities = list(gate.dependencies) + list(gate.evidence_files)
    for binding in gate.carrier_bindings:
        identities.extend((binding.manifest_identity, binding.sample_identity))
    for identity in identities:
        target = (root / Path(*identity.relative_path.split("/"))).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise JLcPreW04Error("pre-W04 dependency 缺失或路径越界")
        payload = target.read_bytes()
        if (len(payload) != identity.size_bytes
                or hashlib.sha256(payload).hexdigest() != identity.sha256):
            raise JLcPreW04Error(
                f"pre-W04 dependency identity 漂移: {identity.role}")


__all__ = [
    "ARTIFACT_KIND", "ARTIFACT_STATUS", "ARTIFACT_VERSION",
    "DEPENDENCY_ROLES", "EVIDENCE_ROLES", "FORMAT_VERSION",
    "GateCarrierBinding", "GateFileIdentity", "JLcPreW04Error",
    "JLcPreW04Gate", "MANIFEST_PATH", "OPEN_GENERATION_SUFFIX",
    "ORIGINAL_W02_RECEIPT_SHA256", "OpenGenerationBoundary",
    "OriginalW02ReceiptCommitment", "PARENT_HEAD_SHA1", "PUBLISHED_STATE",
    "SUPPLEMENTAL_RECEIPT_STATUSES", "W04_BLOCKING_FAILURE_KEYS",
    "read_j_lc_pre_w04_gate", "verify_j_lc_pre_w04_files",
    "write_j_lc_pre_w04_gate",
]
