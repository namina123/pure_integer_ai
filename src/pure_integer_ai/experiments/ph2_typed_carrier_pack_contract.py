"""LC-16 typed carrier pack 的公开输入、预算和 fail-closed 合同。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from pure_integer_ai.cognition.shared.artifact_envelope import (
    ANCHOR_DOCUMENT_REGION,
    ANCHOR_GRID_RECT,
    ANCHOR_REFERENCE_SLOT,
    ANCHOR_TEXT_RANGE,
    ANCHOR_TRANSCRIPT_ALIGNMENT,
    ANCHOR_TREE_PATH,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    StableRecordKey,
    canonical_json_bytes,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_language_coverage_v2_contract import (
    DIRECTIONS,
    IN_SCOPE_CARRIER_KEYS,
)


FORMAT_VERSION = 1
ARTIFACT_KIND = "PH2_LC16_TYPED_CARRIER_PACK"
ARTIFACT_VERSION = "LC16-TYPED-CARRIER-PACK-20260731-A"
ARTIFACT_STATUS = "PACK_FROZEN_CONTRACT_ONLY"
SAMPLE_KINDS = (
    "POSITIVE",
    "NEGATIVE",
    "AMBIGUOUS",
    "UNKNOWN",
    "REVISION",
    "GENERATION",
    "RETENTION",
)
SAMPLE_SPLITS = {
    "POSITIVE": "train",
    "NEGATIVE": "train",
    "AMBIGUOUS": "dev",
    "UNKNOWN": "held_out",
    "REVISION": "held_out",
    "GENERATION": "adversarial",
    "RETENTION": "adversarial",
}
PAYLOAD_STATES = ("UNMATERIALIZED",)
RUNTIME_STATES = ("NOT_RUN",)
EVIDENCE_ROLES = ("CONTRACT", "COVERAGE", "CATALOG", "TEST")
PACK_EXECUTION_STATE = {
    "LANGUAGE_CAPABILITY_MASTERED": 0,
    "LANGUAGE_READINESS": 0,
    "W04_STARTED": 0,
    "carrier_runtime": 0,
    "formal_training_runs": 0,
    "llm_calls": 0,
    "memory_learning_writes": 0,
    "teacher_calls": 0,
}
PACK_ANCHOR_KINDS = {
    "DOCUMENT_CONTAINER": (
        ANCHOR_TEXT_RANGE,
        ANCHOR_TREE_PATH,
        ANCHOR_DOCUMENT_REGION,
    ),
    "HTML": (
        ANCHOR_TEXT_RANGE,
        ANCHOR_TREE_PATH,
        ANCHOR_DOCUMENT_REGION,
    ),
    "MARKDOWN": (
        ANCHOR_TEXT_RANGE,
        ANCHOR_TREE_PATH,
        ANCHOR_DOCUMENT_REGION,
    ),
    "MATH_NOTATION": (ANCHOR_TEXT_RANGE, ANCHOR_TREE_PATH),
    "PLAIN_TEXT": (ANCHOR_TEXT_RANGE,),
    "REFERENCE_LINK_EMBED": (
        ANCHOR_TEXT_RANGE,
        ANCHOR_DOCUMENT_REGION,
        ANCHOR_REFERENCE_SLOT,
    ),
    "SOURCE_CODE": (
        ANCHOR_TEXT_RANGE,
        ANCHOR_TREE_PATH,
        ANCHOR_DOCUMENT_REGION,
    ),
    "TABLE_GRID": (
        ANCHOR_TEXT_RANGE,
        ANCHOR_GRID_RECT,
        ANCHOR_DOCUMENT_REGION,
    ),
    "TRANSCRIBED_OCR_ASR": (
        ANCHOR_TEXT_RANGE,
        ANCHOR_DOCUMENT_REGION,
        ANCHOR_TRANSCRIPT_ALIGNMENT,
    ),
}


class TypedCarrierPackError(RuntimeError):
    """typed carrier pack 的身份、覆盖或零执行边界不闭合。"""


def _exact(value: Any, expected: set[str], *, where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise TypedCarrierPackError(f"{where} 字段不精确")
    return value


def _text(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise TypedCarrierPackError(f"{where} 必须是非空规范文本")
    return value


def _enum(value: Any, allowed: tuple[str, ...], *, where: str) -> str:
    text = _text(value, where=where)
    if text not in allowed:
        raise TypedCarrierPackError(f"{where} 未登记")
    return text


def _positive(value: Any, *, where: str) -> int:
    if type(value) is not int or value <= 0:
        raise TypedCarrierPackError(f"{where} 必须是正严格整数")
    return value


def _nonnegative(value: Any, *, where: str) -> int:
    if type(value) is not int or value < 0:
        raise TypedCarrierPackError(f"{where} 必须是非负严格整数")
    return value


def _sha256(value: Any, *, where: str) -> str:
    text = _text(value, where=where).lower()
    if len(text) != 64 or any(item not in "0123456789abcdef" for item in text):
        raise TypedCarrierPackError(f"{where} 必须是小写 SHA-256")
    return text


def _relative_path(value: Any, *, where: str) -> str:
    text = _text(value, where=where)
    path = PurePosixPath(text)
    if (path.is_absolute() or ".." in path.parts or "\\" in text
            or path.as_posix() != text or ":" in path.parts[0]):
        raise TypedCarrierPackError(f"{where} 必须是安全 POSIX 相对路径")
    return text


def _key(value: Any, *, where: str) -> StableRecordKey:
    if not isinstance(value, StableRecordKey):
        raise TypedCarrierPackError(f"{where} 必须是 StableRecordKey")
    return value


def _keys(
        value: tuple[StableRecordKey, ...],
        *,
        where: str,
        allow_empty: bool = False,
        ) -> tuple[StableRecordKey, ...]:
    if (not isinstance(value, tuple)
            or (not allow_empty and not value)
            or any(not isinstance(item, StableRecordKey) for item in value)):
        raise TypedCarrierPackError(f"{where} 必须是 StableRecordKey tuple")
    if value != tuple(sorted(set(value))):
        raise TypedCarrierPackError(f"{where} 必须排序去重")
    return value


def _keys_from(value: Any, *, where: str) -> tuple[StableRecordKey, ...]:
    if not isinstance(value, list):
        raise TypedCarrierPackError(f"{where} 必须是列表")
    try:
        return tuple(StableRecordKey.from_value(item, where=where)
                     for item in value)
    except Exception as error:
        raise TypedCarrierPackError(f"{where} 稳定键非法") from error


@dataclass(frozen=True, order=True)
class TypedCarrierEvidenceFile:
    """pack 合同、coverage、catalog 和测试的文件级身份。"""

    relative_path: str
    role: str
    byte_count: int
    sha256: str

    def __post_init__(self) -> None:
        _relative_path(self.relative_path, where="evidence relative_path")
        _enum(self.role, EVIDENCE_ROLES, where="evidence role")
        _positive(self.byte_count, where="evidence byte_count")
        _sha256(self.sha256, where="evidence sha256")

    def to_dict(self) -> dict[str, Any]:
        return {
            "byte_count": self.byte_count,
            "relative_path": self.relative_path,
            "role": self.role,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TypedCarrierEvidenceFile":
        raw = _exact(value, {
            "byte_count", "relative_path", "role", "sha256",
        }, where="TypedCarrierEvidenceFile")
        return cls(
            str(raw["relative_path"]),
            str(raw["role"]),
            raw["byte_count"],
            str(raw["sha256"]),
        )


@dataclass(frozen=True, order=True)
class TypedCarrierBudget:
    """单一 carrier 的 raw、结构、边和引用硬预算。"""

    carrier_key: str
    max_cases: int
    max_raw_units: int
    max_structure_nodes: int
    max_edges: int
    max_depth: int
    max_references: int

    def __post_init__(self) -> None:
        _enum(self.carrier_key, IN_SCOPE_CARRIER_KEYS,
              where="TypedCarrierBudget.carrier_key")
        for name in (
                "max_cases", "max_raw_units", "max_structure_nodes",
                "max_edges", "max_depth", "max_references"):
            _positive(getattr(self, name), where=f"budget {name}")
        if self.max_cases != len(SAMPLE_KINDS):
            raise TypedCarrierPackError("budget max_cases 必须覆盖全部样本类型")

    def to_dict(self) -> dict[str, Any]:
        return {
            "carrier_key": self.carrier_key,
            "max_cases": self.max_cases,
            "max_depth": self.max_depth,
            "max_edges": self.max_edges,
            "max_raw_units": self.max_raw_units,
            "max_references": self.max_references,
            "max_structure_nodes": self.max_structure_nodes,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TypedCarrierBudget":
        raw = _exact(value, {
            "carrier_key", "max_cases", "max_depth", "max_edges",
            "max_raw_units", "max_references", "max_structure_nodes",
        }, where="TypedCarrierBudget")
        return cls(
            str(raw["carrier_key"]),
            raw["max_cases"],
            raw["max_raw_units"],
            raw["max_structure_nodes"],
            raw["max_edges"],
            raw["max_depth"],
            raw["max_references"],
        )


@dataclass(frozen=True, order=True)
class TypedCarrierCase:
    """不含 raw payload 的一个 typed carrier 学习/验证样本声明。"""

    case_key: StableRecordKey
    carrier_key: str
    sample_kind: str
    owner_key: StableRecordKey
    split: str
    content_cluster: StableRecordKey
    template_cluster: StableRecordKey
    shape_cluster: StableRecordKey
    combination_cluster: StableRecordKey
    anchor_kinds: tuple[int, ...]
    directions: tuple[str, ...]
    payload_state: str = "UNMATERIALIZED"
    runtime_state: str = "NOT_RUN"

    def __post_init__(self) -> None:
        _key(self.case_key, where="case_key")
        _enum(self.carrier_key, IN_SCOPE_CARRIER_KEYS,
              where="TypedCarrierCase.carrier_key")
        _enum(self.sample_kind, SAMPLE_KINDS,
              where="TypedCarrierCase.sample_kind")
        _key(self.owner_key, where="owner_key")
        _enum(self.split, ("train", "dev", "held_out", "adversarial"),
              where="case split")
        for name in (
                "content_cluster", "template_cluster", "shape_cluster",
                "combination_cluster"):
            _key(getattr(self, name), where=name)
        if (not isinstance(self.anchor_kinds, tuple)
                or not self.anchor_kinds
                or any(type(item) is not int or item <= 0
                       for item in self.anchor_kinds)
                or self.anchor_kinds != tuple(sorted(set(self.anchor_kinds)))
                or self.anchor_kinds != PACK_ANCHOR_KINDS[self.carrier_key]):
            raise TypedCarrierPackError("case anchor_kinds 与 carrier schema 不一致")
        if self.directions != DIRECTIONS:
            raise TypedCarrierPackError("case directions 必须覆盖三向")
        _enum(self.payload_state, PAYLOAD_STATES, where="case payload_state")
        _enum(self.runtime_state, RUNTIME_STATES, where="case runtime_state")
        if self.split != SAMPLE_SPLITS[self.sample_kind]:
            raise TypedCarrierPackError("sample kind 与 split 不一致")

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor_kinds": list(self.anchor_kinds),
            "case_key": self.case_key.to_list(),
            "carrier_key": self.carrier_key,
            "combination_cluster": self.combination_cluster.to_list(),
            "content_cluster": self.content_cluster.to_list(),
            "directions": list(self.directions),
            "owner_key": self.owner_key.to_list(),
            "payload_state": self.payload_state,
            "runtime_state": self.runtime_state,
            "sample_kind": self.sample_kind,
            "shape_cluster": self.shape_cluster.to_list(),
            "split": self.split,
            "template_cluster": self.template_cluster.to_list(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TypedCarrierCase":
        raw = _exact(value, {
            "anchor_kinds", "case_key", "carrier_key", "combination_cluster",
            "content_cluster", "directions", "owner_key", "payload_state",
            "runtime_state", "sample_kind", "shape_cluster", "split",
            "template_cluster",
        }, where="TypedCarrierCase")
        return cls(
            StableRecordKey.from_value(raw["case_key"], where="case_key"),
            str(raw["carrier_key"]),
            str(raw["sample_kind"]),
            StableRecordKey.from_value(raw["owner_key"], where="owner_key"),
            str(raw["split"]),
            StableRecordKey.from_value(
                raw["content_cluster"], where="content_cluster"),
            StableRecordKey.from_value(
                raw["template_cluster"], where="template_cluster"),
            StableRecordKey.from_value(raw["shape_cluster"], where="shape_cluster"),
            StableRecordKey.from_value(
                raw["combination_cluster"], where="combination_cluster"),
            tuple(raw["anchor_kinds"]),
            tuple(str(item) for item in raw["directions"]),
            str(raw["payload_state"]),
            str(raw["runtime_state"]),
        )


def _validate_cases(cases: tuple[TypedCarrierCase, ...]) -> None:
    if not isinstance(cases, tuple) or not cases:
        raise TypedCarrierPackError("typed carrier cases 不能为空")
    if any(not isinstance(item, TypedCarrierCase) for item in cases):
        raise TypedCarrierPackError("typed carrier case 类型错误")
    if cases != tuple(sorted(cases, key=lambda item: item.case_key)):
        raise TypedCarrierPackError("typed carrier cases 必须按 case_key 排序")
    if len({item.case_key for item in cases}) != len(cases):
        raise TypedCarrierPackError("case_key 不得重复")
    expected = {
        carrier: set(SAMPLE_KINDS) for carrier in IN_SCOPE_CARRIER_KEYS
    }
    for item in cases:
        expected[item.carrier_key].discard(item.sample_kind)
    if any(expected.values()):
        raise TypedCarrierPackError("每个 carrier 必须完整覆盖七类样本")
    if len(cases) != len(IN_SCOPE_CARRIER_KEYS) * len(SAMPLE_KINDS):
        raise TypedCarrierPackError("typed carrier case 数量漂移")
    if len({item.owner_key for item in cases}) != len(cases):
        raise TypedCarrierPackError("owner_key 必须逐 case 隔离")
    for axis_name in (
            "content_cluster", "template_cluster", "shape_cluster",
            "combination_cluster"):
        values_by_split: dict[str, set[StableRecordKey]] = {}
        for item in cases:
            values_by_split.setdefault(item.split, set()).add(
                getattr(item, axis_name))
        splits = tuple(values_by_split)
        for index, first in enumerate(splits):
            for second in splits[index + 1:]:
                if values_by_split[first] & values_by_split[second]:
                    raise TypedCarrierPackError(
                        f"{axis_name} 跨 split 重叠")


@dataclass(frozen=True, order=True)
class TypedCarrierPackManifest:
    """冻结 typed carrier pack 的覆盖、预算、样本轴和零执行边界。"""

    format_version: int
    artifact_version: str
    artifact_status: str
    coverage_v2_sha256: str
    budgets: tuple[TypedCarrierBudget, ...]
    cases: tuple[TypedCarrierCase, ...]
    execution_state: CanonicalJsonObject
    evidence_files: tuple[TypedCarrierEvidenceFile, ...]

    def __post_init__(self) -> None:
        if self.format_version != FORMAT_VERSION:
            raise TypedCarrierPackError("format_version 非法")
        if self.artifact_version != ARTIFACT_VERSION:
            raise TypedCarrierPackError("artifact_version 非法")
        if self.artifact_status != ARTIFACT_STATUS:
            raise TypedCarrierPackError("artifact_status 非法")
        _sha256(self.coverage_v2_sha256, where="coverage_v2_sha256")
        if (not isinstance(self.budgets, tuple)
                or any(not isinstance(item, TypedCarrierBudget)
                       for item in self.budgets)
                or tuple(item.carrier_key for item in self.budgets)
                != IN_SCOPE_CARRIER_KEYS):
            raise TypedCarrierPackError("budgets 必须精确列出九类 carrier")
        if len({item.carrier_key for item in self.budgets}) != len(self.budgets):
            raise TypedCarrierPackError("budget carrier 不得重复")
        _validate_cases(self.cases)
        if (not isinstance(self.execution_state, CanonicalJsonObject)
                or self.execution_state.to_value() != PACK_EXECUTION_STATE):
            raise TypedCarrierPackError("pack execution_state 必须全零")
        if (not isinstance(self.evidence_files, tuple)
                or not self.evidence_files
                or any(not isinstance(item, TypedCarrierEvidenceFile)
                       for item in self.evidence_files)):
            raise TypedCarrierPackError("pack evidence_files 非法")
        evidence = tuple(sorted(
            self.evidence_files,
            key=lambda item: (item.relative_path, item.role),
        ))
        if evidence != self.evidence_files:
            raise TypedCarrierPackError("pack evidence_files 必须排序")
        if {item.role for item in evidence} != set(EVIDENCE_ROLES):
            raise TypedCarrierPackError("pack evidence role 未闭合")
        if len({item.relative_path for item in evidence}) != len(evidence):
            raise TypedCarrierPackError("pack evidence path 不得重复")

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": ARTIFACT_KIND,
            "artifact_status": self.artifact_status,
            "artifact_version": self.artifact_version,
            "budgets": [item.to_dict() for item in self.budgets],
            "cases": [item.to_dict() for item in self.cases],
            "coverage_v2_sha256": self.coverage_v2_sha256,
            "evidence_files": [item.to_dict() for item in self.evidence_files],
            "execution_state": self.execution_state.to_value(),
            "format_version": self.format_version,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict()) + b"\n"

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TypedCarrierPackManifest":
        raw = _exact(value, {
            "artifact_kind", "artifact_status", "artifact_version", "budgets",
            "cases", "coverage_v2_sha256", "evidence_files",
            "execution_state", "format_version",
        }, where="TypedCarrierPackManifest")
        if raw["artifact_kind"] != ARTIFACT_KIND:
            raise TypedCarrierPackError("artifact_kind 非法")
        return cls(
            raw["format_version"],
            str(raw["artifact_version"]),
            str(raw["artifact_status"]),
            str(raw["coverage_v2_sha256"]),
            tuple(TypedCarrierBudget.from_dict(item) for item in raw["budgets"]),
            tuple(TypedCarrierCase.from_dict(item) for item in raw["cases"]),
            CanonicalJsonObject.from_value(raw["execution_state"]),
            tuple(TypedCarrierEvidenceFile.from_dict(item)
                  for item in raw["evidence_files"]),
        )


def read_typed_carrier_pack_manifest(
        path: str | Path,
        ) -> TypedCarrierPackManifest:
    """严格回读 canonical typed carrier pack manifest。"""
    try:
        payload = Path(path).read_bytes()
        if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
            raise TypedCarrierPackError("pack manifest newline 非法")
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
        assert isinstance(value, dict)
        manifest = TypedCarrierPackManifest.from_dict(value)
    except TypedCarrierPackError:
        raise
    except Exception as error:
        raise TypedCarrierPackError("pack manifest 损坏") from error
    if manifest.canonical_bytes() != payload:
        raise TypedCarrierPackError("pack manifest 非规范字节")
    return manifest


def write_typed_carrier_pack_manifest(
        manifest: TypedCarrierPackManifest,
        path: str | Path,
        ) -> Path:
    """独占或幂等写 pack manifest，不允许异内容覆盖。"""
    if not isinstance(manifest, TypedCarrierPackManifest):
        raise TypedCarrierPackError("pack manifest 类型非法")
    target = Path(path)
    payload = manifest.canonical_bytes()
    if target.exists():
        if not target.is_file() or target.read_bytes() != payload:
            raise TypedCarrierPackError("pack manifest 已存在且内容不同")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(payload)
    except OSError as error:
        raise TypedCarrierPackError("pack manifest 无法写入") from error
    return target


def verify_typed_carrier_pack_files(
        manifest: TypedCarrierPackManifest,
        *,
        repository_root: str | Path,
        ) -> None:
    """逐字节回验 pack 的合同、coverage、catalog 和测试证据。"""
    root = Path(repository_root).resolve()
    for item in manifest.evidence_files:
        path = (root / Path(*item.relative_path.split("/"))).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise TypedCarrierPackError("pack evidence 路径逃逸") from error
        if not path.is_file():
            raise TypedCarrierPackError("pack evidence 文件缺失")
        payload = path.read_bytes()
        if (len(payload) != item.byte_count
                or hashlib.sha256(payload).hexdigest() != item.sha256):
            raise TypedCarrierPackError("pack evidence 文件身份漂移")


__all__ = [
    "ANCHOR_DOCUMENT_REGION",
    "ANCHOR_GRID_RECT",
    "ANCHOR_REFERENCE_SLOT",
    "ANCHOR_TEXT_RANGE",
    "ANCHOR_TRANSCRIPT_ALIGNMENT",
    "ANCHOR_TREE_PATH",
    "ARTIFACT_KIND",
    "ARTIFACT_STATUS",
    "ARTIFACT_VERSION",
    "DIRECTIONS",
    "EVIDENCE_ROLES",
    "FORMAT_VERSION",
    "IN_SCOPE_CARRIER_KEYS",
    "PACK_ANCHOR_KINDS",
    "PACK_EXECUTION_STATE",
    "PAYLOAD_STATES",
    "RUNTIME_STATES",
    "SAMPLE_KINDS",
    "SAMPLE_SPLITS",
    "TypedCarrierBudget",
    "TypedCarrierCase",
    "TypedCarrierEvidenceFile",
    "TypedCarrierPackError",
    "TypedCarrierPackManifest",
    "read_typed_carrier_pack_manifest",
    "verify_typed_carrier_pack_files",
    "write_typed_carrier_pack_manifest",
]
