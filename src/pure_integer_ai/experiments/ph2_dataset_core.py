"""PH2 资料合同共用的冻结枚举、严格整数键和规范 JSON 基元。"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


FORMAT_VERSION = 1
SCHEMA_VERSION = 1

W_STAGES = (
    "W-01", "W-02", "W-03", "W-04", "W-05",
    "W-06", "W-07", "W-08", "W-09",
)
SPLITS = ("train", "dev", "held_out", "adversarial", "wall")
EPISTEMIC_ROLES = ("forming", "reveal", "teacher", "evaluator")
SAMPLE_ROLES = (
    "support", "refute", "conflict", "supersede", "anomaly",
    "read_only_probe",
)
EXPECTED_STATES = ("TRUE", "FALSE", "UNKNOWN", "CONFLICT")
REDISTRIBUTION_POLICIES = ("PUBLIC", "LOCAL_ONLY")
OWNER_MODES = ("read_only",)

PUBLIC_LICENSE_IDS = frozenset({
    "CC0-1.0",
    "CC-BY-4.0",
    "CC-BY-SA-3.0",
    "CC-BY-SA-4.0",
})
LOCAL_ONLY_LICENSE_IDS = frozenset({
    "NOASSERTION-README-SHARING-NOT-RECOMMENDED",
})
ALLOWED_LICENSE_IDS = PUBLIC_LICENSE_IDS | LOCAL_ONLY_LICENSE_IDS

RECORD_SOURCE_REF = "source_ref"
RECORD_OBSERVATION = "observation"
RECORD_TEACHER_EVIDENCE = "teacher_evidence"
RECORD_EVALUATOR_LABEL = "evaluator_label"
RECORD_ARTIFACT_MANIFEST = "artifact_manifest"
JSONL_RECORD_KINDS = (
    RECORD_SOURCE_REF,
    RECORD_OBSERVATION,
    RECORD_TEACHER_EVIDENCE,
    RECORD_EVALUATOR_LABEL,
)
OWNER_KINDS = ("source", "observation", "teacher", "evaluator", "anomaly")


class DatasetContractError(ValueError):
    """资料记录、版本、许可或规范 JSON 不满足统一合同。"""


def _positive_int(value: Any, *, where: str) -> int:
    """要求值为正严格整数并返回原值。"""
    if type(value) is not int or value <= 0:
        raise DatasetContractError(f"{where} 必须是正严格整数")
    return value


def _nonnegative_int(value: Any, *, where: str) -> int:
    """要求值为非负严格整数并返回原值。"""
    if type(value) is not int or value < 0:
        raise DatasetContractError(f"{where} 必须是非负严格整数")
    return value


def _nonempty_text(value: Any, *, where: str) -> str:
    """要求值为去除首尾空白后仍非空的字符串。"""
    if not isinstance(value, str) or not value or value.strip() != value:
        raise DatasetContractError(f"{where} 必须是无首尾空白的非空字符串")
    return value


def _enum_text(value: Any, allowed: tuple[str, ...], *, where: str) -> str:
    """要求字符串属于调用方给定的冻结枚举。"""
    text = _nonempty_text(value, where=where)
    if text not in allowed:
        raise DatasetContractError(f"{where} 不在允许集合中: {text}")
    return text


def _sha256(value: Any, *, where: str) -> str:
    """要求值为可归一成小写的 SHA-256 十六进制串。"""
    text = _nonempty_text(value, where=where).lower()
    if len(text) != 64 or any(character not in "0123456789abcdef"
                              for character in text):
        raise DatasetContractError(f"{where} 必须是 64 位 SHA-256")
    return text


def _upstream_checksum(value: Any, *, where: str) -> str:
    """校验带算法前缀的上游 SHA-1/SHA-256 摘要。"""
    text = _nonempty_text(value, where=where).lower()
    if ":" not in text:
        raise DatasetContractError(f"{where} 必须带 sha1: 或 sha256: 前缀")
    algorithm, digest = text.split(":", 1)
    expected_length = {"sha1": 40, "sha256": 64}.get(algorithm)
    if (expected_length is None or len(digest) != expected_length
            or any(character not in "0123456789abcdef"
                   for character in digest)):
        raise DatasetContractError(f"{where} 上游摘要非法")
    return text


def _license_id(value: Any, *, where: str) -> str:
    """要求许可标识属于 PH2 已冻结的许可分区。"""
    text = _nonempty_text(value, where=where)
    if text not in ALLOWED_LICENSE_IDS:
        raise DatasetContractError(f"{where} 不是 PH2 允许许可")
    return text


def _record_key_tuple(
        values: Any, *, where: str,
        allow_empty: bool = False) -> tuple["StableRecordKey", ...]:
    """校验 StableRecordKey tuple 并拒绝重复键。"""
    if not isinstance(values, tuple):
        raise DatasetContractError(f"{where} 必须是 StableRecordKey tuple")
    if not values and not allow_empty:
        raise DatasetContractError(f"{where} 不能为空")
    if any(not isinstance(value, StableRecordKey) for value in values):
        raise DatasetContractError(f"{where} 含非 StableRecordKey")
    if len(set(values)) != len(values):
        raise DatasetContractError(f"{where} 不得重复")
    return values


def _reject_json_number(text: str) -> None:
    """拒绝 JSON 浮点和非有限数值文本。"""
    raise DatasetContractError(f"规范 JSON 禁止非整数数值: {text}")


def _validate_json_value(value: Any, *, where: str) -> None:
    """递归校验 JSON 值只含 null/bool/int/str/list/object。"""
    if value is None or isinstance(value, (bool, str)) or type(value) is int:
        return
    if isinstance(value, float):
        raise DatasetContractError(f"{where} 禁止浮点")
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_json_value(item, where=f"{where}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise DatasetContractError(f"{where} 的 object key 必须是非空字符串")
            _validate_json_value(item, where=f"{where}.{key}")
        return
    raise DatasetContractError(f"{where} 含 JSON 不支持类型 {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """把无浮点 JSON 值编码为排序键、紧凑分隔符的 UTF-8 字节。"""
    _validate_json_value(value, where="canonical_json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_json_line(value: Any) -> bytes:
    """返回以单个换行结束的规范 JSONL 记录。"""
    return canonical_json_bytes(value) + b"\n"


def parse_canonical_json_bytes(payload: bytes, *, require_object: bool) -> Any:
    """严格解析 UTF-8 规范 JSON，并拒绝浮点、常量和非规范编码。"""
    if not isinstance(payload, bytes) or not payload:
        raise DatasetContractError("规范 JSON payload 必须是非空 bytes")
    try:
        value = json.loads(
            payload.decode("utf-8"),
            parse_float=_reject_json_number,
            parse_constant=_reject_json_number,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise DatasetContractError("规范 JSON payload 损坏") from error
    _validate_json_value(value, where="parsed_json")
    if require_object and not isinstance(value, dict):
        raise DatasetContractError("规范 JSON 根必须是 object")
    if canonical_json_bytes(value) != payload:
        raise DatasetContractError("JSON 编码不是规范形式")
    return value


@dataclass(frozen=True, order=True)
class StableRecordKey:
    """资料记录、cluster、owner 和 manifest 共用的完整整数键。"""

    components: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.components, tuple) or not self.components:
            raise DatasetContractError("StableRecordKey.components 不能为空")
        if any(type(value) is not int or value <= 0
               for value in self.components):
            raise DatasetContractError("StableRecordKey.components 必须是正严格整数")

    def to_list(self) -> list[int]:
        """把稳定键编码为 JSON 整数列表。"""
        return list(self.components)

    def stable_key(self) -> tuple[int, ...]:
        """返回完整整数键，不以摘要替代身份。"""
        return self.components

    @classmethod
    def from_value(cls, value: Any, *, where: str) -> "StableRecordKey":
        """从 JSON 整数列表恢复稳定键。"""
        if not isinstance(value, list) or not value:
            raise DatasetContractError(f"{where} 必须是非空整数列表")
        if any(type(item) is not int or item <= 0 for item in value):
            raise DatasetContractError(f"{where} 必须只含正严格整数")
        return cls(tuple(value))


@dataclass(frozen=True)
class CanonicalJsonObject:
    """以规范字节保存不可变 typed payload，避免冻结对象内藏可变 dict。"""

    payload: bytes

    def __post_init__(self) -> None:
        parse_canonical_json_bytes(self.payload, require_object=True)

    @classmethod
    def from_value(cls, value: dict[str, Any]) -> "CanonicalJsonObject":
        """从无浮点 JSON object 建立不可变规范载荷。"""
        if not isinstance(value, dict):
            raise DatasetContractError("CanonicalJsonObject 根必须是 dict")
        return cls(canonical_json_bytes(value))

    def to_value(self) -> dict[str, Any]:
        """恢复独立 JSON object，调用方修改不会改变已冻结载荷。"""
        value = parse_canonical_json_bytes(self.payload, require_object=True)
        assert isinstance(value, dict)
        return value

    def sha256(self) -> str:
        """返回 payload 的 SHA-256，用于审计而不替代完整载荷。"""
        return hashlib.sha256(self.payload).hexdigest()


__all__ = [
    "ALLOWED_LICENSE_IDS",
    "CanonicalJsonObject",
    "DatasetContractError",
    "EPISTEMIC_ROLES",
    "EXPECTED_STATES",
    "FORMAT_VERSION",
    "JSONL_RECORD_KINDS",
    "LOCAL_ONLY_LICENSE_IDS",
    "OWNER_KINDS",
    "OWNER_MODES",
    "PUBLIC_LICENSE_IDS",
    "RECORD_ARTIFACT_MANIFEST",
    "RECORD_EVALUATOR_LABEL",
    "RECORD_OBSERVATION",
    "RECORD_SOURCE_REF",
    "RECORD_TEACHER_EVIDENCE",
    "REDISTRIBUTION_POLICIES",
    "SAMPLE_ROLES",
    "SCHEMA_VERSION",
    "SPLITS",
    "StableRecordKey",
    "W_STAGES",
    "canonical_json_bytes",
    "canonical_json_line",
    "parse_canonical_json_bytes",
]
