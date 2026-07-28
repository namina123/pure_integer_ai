"""LC-09 跨轴迁移、组合 held-out 与范围收缩的纯合同。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    canonical_json_bytes,
    parse_canonical_json_bytes,
)


FORMAT_VERSION = 1
ARTIFACT_STATUS = "CONTRACT_FROZEN"
RUNTIME_STATUS = "NOT_STARTED"
TRANSFER_AXIS_KEYS = (
    "CODE_SWITCH",
    "DIALECT",
    "DOMAIN",
    "ERA",
    "GENRE",
    "LANGUAGE",
    "LENGTH",
    "REGISTER",
    "SCRIPT",
    "SOURCE",
)
AXIS_DECLARATION_STATES = (
    "BASELINE_ONLY",
    "DECLARED_NE",
    "UNDECLARED",
    "VARIATION_OBSERVED",
)
PACK_KINDS = ("AUTHORED_COURSE", "SOURCE_PACK")
COMBINATION_SPLIT_STATES = (
    "FROZEN",
    "NE_AXIS_UNDECLARED",
    "NE_COMBINATION_OVERLAP",
    "NE_SINGLE_SPLIT",
)
SPLIT_PROBE_KINDS = (
    "DOUBLE_AXIS",
    "FULL_COMBINATION",
    "SINGLE_AXIS",
)
SPLIT_PROBE_VERDICTS = ("PASS", "REJECT")
VERIFIER_DIMENSIONS = (
    "ALL_FORMAL_PACKS_INVENTORIED",
    "ALL_TRANSFER_AXES_EXPLICIT_OR_NE",
    "DOUBLE_AXIS_HELD_OUT",
    "FULL_COMBINATION_HELD_OUT",
    "NO_DOMAIN_TO_GLOBAL_EXTRAPOLATION",
    "SCOPE_CONTRACTION_REPLAYABLE",
    "SINGLE_AXIS_HELD_OUT",
    "ZERO_HOST_LEARNING_WRITE",
)
VERIFIER_NE_CONDITIONS = (
    "AXIS_VALUE_UNDECLARED",
    "FORMAL_RUNTIME_NOT_EXECUTED",
    "PACK_HAS_SINGLE_SPLIT",
    "STANDARD_COMBINATION_NOT_ISOLATED",
    "TRANSFER_RESULT_NOT_OBSERVED",
)
SCOPE_CONTRACTION_PROTOCOLS = (
    "OBJECT_SCOPE_IS_EXPLICIT_AXIS_PRODUCT",
    "ONE_AXIS_PASS_DOES_NOT_AUTHORIZE_OTHER_AXES",
    "PRIOR_SCOPE_REPLAY_PRECEDES_EXPANSION",
    "SCOPE_CONTRACTION_APPEND_ONLY_RECEIPT",
    "UNDECLARED_AXIS_FORCES_NE",
)
EXECUTION_STATE = {
    "companion_writes": 0,
    "core_learning_writes": 0,
    "d03_published": 0,
    "formal_training_runs": 0,
    "mastered_claims": 0,
    "memory_learning_writes": 0,
    "readiness_claims": 0,
    "teacher_calls": 0,
    "use_learning_writes": 0,
    "w01_started": 0,
}


class TransferAxisContractError(RuntimeError):
    """LC-09 轴声明、split probe 或 manifest 不满足冻结边界。"""


def _text(value: Any, *, where: str, allow_empty: bool = False) -> str:
    """要求文本无首尾空白，并按字段控制空值。"""
    if not isinstance(value, str) or value.strip() != value:
        raise TransferAxisContractError(f"{where} 必须是规范文本")
    if not allow_empty and not value:
        raise TransferAxisContractError(f"{where} 不能为空")
    return value


def _flag(value: Any, *, where: str) -> int:
    """要求协议开关为严格 0/1。"""
    if type(value) is not int or value not in {0, 1}:
        raise TransferAxisContractError(f"{where} 必须是 0/1")
    return value


def _nonnegative(value: Any, *, where: str) -> int:
    """要求计数为非负严格整数。"""
    if type(value) is not int or value < 0:
        raise TransferAxisContractError(f"{where} 必须是非负严格整数")
    return value


def _sha256(value: Any, *, where: str) -> str:
    """要求字段为小写 SHA-256。"""
    text = _text(value, where=where)
    if len(text) != 64 or any(item not in "0123456789abcdef" for item in text):
        raise TransferAxisContractError(f"{where} 必须是 SHA-256")
    return text


def _relative_path(value: Any, *, where: str) -> str:
    """要求 artifact 引用为可迁移的安全 POSIX 相对路径。"""
    text = _text(value, where=where)
    path = PurePosixPath(text)
    if (path.is_absolute() or ".." in path.parts or "\\" in text
            or path.as_posix() != text or ":" in path.parts[0]):
        raise TransferAxisContractError(f"{where} 必须是安全相对路径")
    return text


def _strict_text_tuple(
        value: Any,
        *,
        where: str,
        allow_empty: bool = False,
        ) -> tuple[str, ...]:
    """要求字符串元组已排序去重，避免序列化时静默改序。"""
    if not isinstance(value, tuple):
        raise TransferAxisContractError(f"{where} 必须是 tuple")
    if not allow_empty and not value:
        raise TransferAxisContractError(f"{where} 不能为空")
    for item in value:
        _text(item, where=where)
    if tuple(sorted(set(value))) != value:
        raise TransferAxisContractError(f"{where} 必须排序去重")
    return value


def _axis_value_mapping(
        value: CanonicalJsonObject,
        *,
        where: str,
        ) -> dict[str, tuple[str, ...]]:
    """恢复十轴到排序值列表的精确映射。"""
    if not isinstance(value, CanonicalJsonObject):
        raise TransferAxisContractError(f"{where} 类型非法")
    raw = value.to_value()
    if not isinstance(raw, dict) or tuple(sorted(raw)) != TRANSFER_AXIS_KEYS:
        raise TransferAxisContractError(f"{where} 未列全十轴")
    result: dict[str, tuple[str, ...]] = {}
    for axis in TRANSFER_AXIS_KEYS:
        items = raw[axis]
        if not isinstance(items, list):
            raise TransferAxisContractError(f"{where}.{axis} 必须是列表")
        values = tuple(items)
        result[axis] = _strict_text_tuple(
            values, where=f"{where}.{axis}", allow_empty=True)
    return result


def _axis_state_mapping(
        value: CanonicalJsonObject,
        *,
        where: str,
        ) -> dict[str, str]:
    """恢复十轴到声明状态的精确映射。"""
    if not isinstance(value, CanonicalJsonObject):
        raise TransferAxisContractError(f"{where} 类型非法")
    raw = value.to_value()
    if not isinstance(raw, dict) or tuple(sorted(raw)) != TRANSFER_AXIS_KEYS:
        raise TransferAxisContractError(f"{where} 未列全十轴")
    result: dict[str, str] = {}
    for axis in TRANSFER_AXIS_KEYS:
        state = _text(raw[axis], where=f"{where}.{axis}")
        if state not in AXIS_DECLARATION_STATES:
            raise TransferAxisContractError(f"{where}.{axis} 状态非法")
        result[axis] = state
    return result


def _combination_mapping(
        value: CanonicalJsonObject,
        *,
        where: str,
        ) -> dict[str, str]:
    """恢复一个列全十轴的组合。"""
    if not isinstance(value, CanonicalJsonObject):
        raise TransferAxisContractError(f"{where} 类型非法")
    raw = value.to_value()
    if not isinstance(raw, dict) or tuple(sorted(raw)) != TRANSFER_AXIS_KEYS:
        raise TransferAxisContractError(f"{where} 必须列全十轴")
    return {
        axis: _text(raw[axis], where=f"{where}.{axis}")
        for axis in TRANSFER_AXIS_KEYS
    }


def _combination_tuple(value: dict[str, str]) -> tuple[str, ...]:
    """按冻结轴序把组合映射投影为稳定元组。"""
    return tuple(value[axis] for axis in TRANSFER_AXIS_KEYS)


def evaluate_transfer_split_probe(
        probe_kind: str,
        isolated_axes: tuple[str, ...],
        train_combinations: tuple[CanonicalJsonObject, ...],
        held_out_combinations: tuple[CanonicalJsonObject, ...],
        ) -> tuple[str, str]:
    """按单轴、双轴或完整组合规则判定 held-out 是否真实隔离。"""
    if probe_kind not in SPLIT_PROBE_KINDS:
        raise TransferAxisContractError("split probe kind 未注册")
    _strict_text_tuple(isolated_axes, where="isolated_axes")
    if any(axis not in TRANSFER_AXIS_KEYS for axis in isolated_axes):
        raise TransferAxisContractError("isolated_axes 含未登记轴")
    expected_count = {
        "SINGLE_AXIS": 1,
        "DOUBLE_AXIS": 2,
        "FULL_COMBINATION": len(TRANSFER_AXIS_KEYS),
    }[probe_kind]
    if len(isolated_axes) != expected_count:
        raise TransferAxisContractError("split probe 隔离轴数量错误")
    if not train_combinations or not held_out_combinations:
        raise TransferAxisContractError("split probe train/held_out 不能为空")
    train = tuple(_combination_mapping(
        item, where="train combination") for item in train_combinations)
    held = tuple(_combination_mapping(
        item, where="held_out combination") for item in held_out_combinations)
    train_full = {_combination_tuple(item) for item in train}
    held_full = {_combination_tuple(item) for item in held}
    if train_full & held_full:
        return "REJECT", "COMPLETE_COMBINATION_LEAK"
    train_values = {
        axis: {item[axis] for item in train} for axis in TRANSFER_AXIS_KEYS
    }
    held_values = {
        axis: {item[axis] for item in held} for axis in TRANSFER_AXIS_KEYS
    }
    isolated = set(isolated_axes)
    if probe_kind == "SINGLE_AXIS":
        axis = isolated_axes[0]
        if not held_values[axis].isdisjoint(train_values[axis]):
            return "REJECT", "SINGLE_AXIS_VALUE_NOT_HELD_OUT"
        for key in TRANSFER_AXIS_KEYS:
            if key != axis and not held_values[key] <= train_values[key]:
                return "REJECT", "NON_ISOLATED_AXIS_VALUE_UNSEEN"
        return "PASS", "NONE"
    for key in TRANSFER_AXIS_KEYS:
        if not held_values[key] <= train_values[key]:
            return "REJECT", "COMPONENT_VALUE_UNSEEN"
    if probe_kind == "DOUBLE_AXIS":
        pair = tuple(isolated_axes)
        train_pairs = {(item[pair[0]], item[pair[1]]) for item in train}
        held_pairs = {(item[pair[0]], item[pair[1]]) for item in held}
        if train_pairs & held_pairs:
            return "REJECT", "DOUBLE_AXIS_PAIR_LEAK"
    return "PASS", "NONE"


@dataclass(frozen=True)
class TransferPackAudit:
    """一个正式 pack 的十轴声明、split 事实和 NE 边界。"""

    pack_manifest_relative_path: str
    pack_manifest_sha256: str
    pack_kind: str
    source_keys: tuple[str, ...]
    license_ids: tuple[str, ...]
    splits: tuple[str, ...]
    axis_values: CanonicalJsonObject
    held_out_axis_values: CanonicalJsonObject
    axis_states: CanonicalJsonObject
    combination_split_state: str
    train_combination_count: int
    held_out_combination_count: int
    transfer_claim_state: str
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        _relative_path(
            self.pack_manifest_relative_path,
            where="pack_manifest_relative_path")
        _sha256(self.pack_manifest_sha256, where="pack_manifest_sha256")
        if self.pack_kind not in PACK_KINDS:
            raise TransferAxisContractError("pack_kind 未注册")
        _strict_text_tuple(self.source_keys, where="source_keys")
        _strict_text_tuple(self.license_ids, where="license_ids")
        _strict_text_tuple(self.splits, where="splits")
        values = _axis_value_mapping(self.axis_values, where="axis_values")
        held_values = _axis_value_mapping(
            self.held_out_axis_values, where="held_out_axis_values")
        states = _axis_state_mapping(self.axis_states, where="axis_states")
        for axis in TRANSFER_AXIS_KEYS:
            if not set(held_values[axis]) <= set(values[axis]):
                raise TransferAxisContractError("held_out 轴值不在 pack 轴值内")
            if states[axis] == "UNDECLARED" and values[axis]:
                raise TransferAxisContractError("UNDECLARED 轴不得带值")
            if states[axis] != "UNDECLARED" and not values[axis]:
                raise TransferAxisContractError("已声明轴必须带值")
            if (states[axis] == "BASELINE_ONLY"
                    and len(values[axis]) != 1):
                raise TransferAxisContractError("BASELINE_ONLY 必须只有一个值")
            if (states[axis] == "VARIATION_OBSERVED"
                    and len(values[axis]) < 2):
                raise TransferAxisContractError("VARIATION_OBSERVED 至少两个值")
        if self.combination_split_state not in COMBINATION_SPLIT_STATES:
            raise TransferAxisContractError("combination_split_state 未注册")
        _nonnegative(
            self.train_combination_count, where="train_combination_count")
        _nonnegative(
            self.held_out_combination_count,
            where="held_out_combination_count")
        if self.transfer_claim_state != "NE":
            raise TransferAxisContractError("D-03 前 pack 不得发 transfer PASS")
        _strict_text_tuple(self.evidence_refs, where="evidence_refs")
        for item in self.evidence_refs:
            _relative_path(item, where="evidence_ref")
        has_undeclared = any(
            state == "UNDECLARED" for state in states.values())
        if (has_undeclared
                and self.combination_split_state != "NE_AXIS_UNDECLARED"):
            raise TransferAxisContractError("缺轴 pack 必须显式 NE")
        if (self.combination_split_state == "FROZEN"
                and (self.train_combination_count == 0
                     or self.held_out_combination_count == 0)):
            raise TransferAxisContractError("FROZEN split 必须双边有组合")

    def to_dict(self) -> dict[str, Any]:
        """投影为规范 JSON 对象。"""
        return {
            "axis_states": self.axis_states.to_value(),
            "axis_values": self.axis_values.to_value(),
            "combination_split_state": self.combination_split_state,
            "evidence_refs": list(self.evidence_refs),
            "held_out_axis_values": self.held_out_axis_values.to_value(),
            "held_out_combination_count": self.held_out_combination_count,
            "license_ids": list(self.license_ids),
            "pack_kind": self.pack_kind,
            "pack_manifest_relative_path": self.pack_manifest_relative_path,
            "pack_manifest_sha256": self.pack_manifest_sha256,
            "source_keys": list(self.source_keys),
            "splits": list(self.splits),
            "train_combination_count": self.train_combination_count,
            "transfer_claim_state": self.transfer_claim_state,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TransferPackAudit":
        """从精确字段对象恢复 pack 轴审计。"""
        expected = {
            "axis_states", "axis_values", "combination_split_state",
            "evidence_refs", "held_out_axis_values",
            "held_out_combination_count", "license_ids", "pack_kind",
            "pack_manifest_relative_path", "pack_manifest_sha256",
            "source_keys", "splits", "train_combination_count",
            "transfer_claim_state",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise TransferAxisContractError("TransferPackAudit 字段不精确")
        return cls(
            str(value["pack_manifest_relative_path"]),
            str(value["pack_manifest_sha256"]),
            str(value["pack_kind"]),
            tuple(str(item) for item in value["source_keys"]),
            tuple(str(item) for item in value["license_ids"]),
            tuple(str(item) for item in value["splits"]),
            CanonicalJsonObject.from_value(value["axis_values"]),
            CanonicalJsonObject.from_value(value["held_out_axis_values"]),
            CanonicalJsonObject.from_value(value["axis_states"]),
            str(value["combination_split_state"]),
            value["train_combination_count"],
            value["held_out_combination_count"],
            str(value["transfer_claim_state"]),
            tuple(str(item) for item in value["evidence_refs"]),
        )


@dataclass(frozen=True)
class BlockedTransferSource:
    """没有合法 pack 的来源及其迁移轴阻断证据。"""

    source_key: str
    blocker_code: str
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        _text(self.source_key, where="blocked source_key")
        _text(self.blocker_code, where="blocked blocker_code")
        _strict_text_tuple(self.evidence_refs, where="blocked evidence_refs")
        for item in self.evidence_refs:
            _relative_path(item, where="blocked evidence_ref")

    def to_dict(self) -> dict[str, Any]:
        """投影为规范 JSON 对象。"""
        return {
            "blocker_code": self.blocker_code,
            "evidence_refs": list(self.evidence_refs),
            "source_key": self.source_key,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BlockedTransferSource":
        """从精确字段对象恢复阻断来源。"""
        if not isinstance(value, dict) or set(value) != {
                "blocker_code", "evidence_refs", "source_key"}:
            raise TransferAxisContractError("BlockedTransferSource 字段不精确")
        return cls(
            str(value["source_key"]),
            str(value["blocker_code"]),
            tuple(str(item) for item in value["evidence_refs"]),
        )


@dataclass(frozen=True)
class TransferSplitProbe:
    """一个单轴、双轴或完整组合 held-out 的可执行冻结 fixture。"""

    probe_key: str
    probe_kind: str
    isolated_axes: tuple[str, ...]
    train_combinations: tuple[CanonicalJsonObject, ...]
    held_out_combinations: tuple[CanonicalJsonObject, ...]
    verdict: str
    failure_code: str
    host_learning_writes: int

    def __post_init__(self) -> None:
        _text(self.probe_key, where="probe_key")
        if self.probe_kind not in SPLIT_PROBE_KINDS:
            raise TransferAxisContractError("probe_kind 未注册")
        if self.verdict not in SPLIT_PROBE_VERDICTS:
            raise TransferAxisContractError("probe verdict 非法")
        _text(self.failure_code, where="probe failure_code")
        _flag(self.host_learning_writes, where="host_learning_writes")
        if self.host_learning_writes != 0:
            raise TransferAxisContractError("split probe 不得写学习宿主")
        actual = evaluate_transfer_split_probe(
            self.probe_kind,
            self.isolated_axes,
            self.train_combinations,
            self.held_out_combinations,
        )
        if actual != (self.verdict, self.failure_code):
            raise TransferAxisContractError("split probe verdict 与 fixture 不一致")

    def to_dict(self) -> dict[str, Any]:
        """投影为规范 JSON 对象。"""
        return {
            "failure_code": self.failure_code,
            "held_out_combinations": [
                item.to_value() for item in self.held_out_combinations],
            "host_learning_writes": self.host_learning_writes,
            "isolated_axes": list(self.isolated_axes),
            "probe_key": self.probe_key,
            "probe_kind": self.probe_kind,
            "train_combinations": [
                item.to_value() for item in self.train_combinations],
            "verdict": self.verdict,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TransferSplitProbe":
        """从精确字段对象恢复 split probe。"""
        expected = {
            "failure_code", "held_out_combinations", "host_learning_writes",
            "isolated_axes", "probe_key", "probe_kind",
            "train_combinations", "verdict",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise TransferAxisContractError("TransferSplitProbe 字段不精确")
        return cls(
            str(value["probe_key"]),
            str(value["probe_kind"]),
            tuple(str(item) for item in value["isolated_axes"]),
            tuple(CanonicalJsonObject.from_value(item)
                  for item in value["train_combinations"]),
            tuple(CanonicalJsonObject.from_value(item)
                  for item in value["held_out_combinations"]),
            str(value["verdict"]),
            str(value["failure_code"]),
            value["host_learning_writes"],
        )


@dataclass(frozen=True)
class LanguageTransferAxisManifest:
    """LC-09 的正式 pack 轴账、held-out fixture 和零运行状态。"""

    format_version: int
    artifact_version: str
    artifact_status: str
    runtime_status: str
    task_key: str
    axis_keys: tuple[str, ...]
    pack_inventory_count: int
    pack_audits: tuple[TransferPackAudit, ...]
    blocked_sources: tuple[BlockedTransferSource, ...]
    split_probes: tuple[TransferSplitProbe, ...]
    verifier_dimensions: tuple[str, ...]
    verifier_ne_conditions: tuple[str, ...]
    scope_contraction_protocols: tuple[str, ...]
    runtime_transfer_pass_authority: int
    execution_state: CanonicalJsonObject

    def __post_init__(self) -> None:
        if self.format_version != FORMAT_VERSION:
            raise TransferAxisContractError("LC-09 format_version 漂移")
        _text(self.artifact_version, where="artifact_version")
        if self.artifact_status != ARTIFACT_STATUS:
            raise TransferAxisContractError("artifact_status 非 CONTRACT_FROZEN")
        if self.runtime_status != RUNTIME_STATUS:
            raise TransferAxisContractError("runtime_status 非 NOT_STARTED")
        if self.task_key != "LC-09":
            raise TransferAxisContractError("task_key 非 LC-09")
        if self.axis_keys != TRANSFER_AXIS_KEYS:
            raise TransferAxisContractError("axis_keys 未冻结")
        _nonnegative(self.pack_inventory_count, where="pack_inventory_count")
        if (not isinstance(self.pack_audits, tuple)
                or not self.pack_audits
                or not all(isinstance(item, TransferPackAudit)
                           for item in self.pack_audits)):
            raise TransferAxisContractError("pack_audits 类型非法")
        audits = tuple(sorted(
            self.pack_audits,
            key=lambda item: item.pack_manifest_relative_path))
        object.__setattr__(self, "pack_audits", audits)
        paths = tuple(item.pack_manifest_relative_path for item in audits)
        if len(set(paths)) != len(paths) or len(audits) != self.pack_inventory_count:
            raise TransferAxisContractError("pack inventory 重复或计数漂移")
        if (not isinstance(self.blocked_sources, tuple)
                or not self.blocked_sources
                or not all(isinstance(item, BlockedTransferSource)
                           for item in self.blocked_sources)):
            raise TransferAxisContractError("blocked_sources 类型非法")
        blocked = tuple(sorted(
            self.blocked_sources, key=lambda item: item.source_key))
        object.__setattr__(self, "blocked_sources", blocked)
        if len({item.source_key for item in blocked}) != len(blocked):
            raise TransferAxisContractError("blocked source 重复")
        if (not isinstance(self.split_probes, tuple)
                or not all(isinstance(item, TransferSplitProbe)
                           for item in self.split_probes)):
            raise TransferAxisContractError("split_probes 类型非法")
        probes = tuple(sorted(self.split_probes, key=lambda item: item.probe_kind))
        object.__setattr__(self, "split_probes", probes)
        if tuple(item.probe_kind for item in probes) != SPLIT_PROBE_KINDS:
            raise TransferAxisContractError("三类 split probe 未列全")
        if any(item.verdict != "PASS" for item in probes):
            raise TransferAxisContractError("LC-09 冻结 fixture 必须全部 PASS")
        if self.verifier_dimensions != VERIFIER_DIMENSIONS:
            raise TransferAxisContractError("verifier_dimensions 漂移")
        if self.verifier_ne_conditions != VERIFIER_NE_CONDITIONS:
            raise TransferAxisContractError("verifier_ne_conditions 漂移")
        if self.scope_contraction_protocols != SCOPE_CONTRACTION_PROTOCOLS:
            raise TransferAxisContractError("scope contraction 协议漂移")
        _flag(
            self.runtime_transfer_pass_authority,
            where="runtime_transfer_pass_authority")
        if self.runtime_transfer_pass_authority != 0:
            raise TransferAxisContractError("LC-09 不得发 runtime transfer PASS")
        if (not isinstance(self.execution_state, CanonicalJsonObject)
                or self.execution_state.to_value() != EXECUTION_STATE):
            raise TransferAxisContractError("LC-09 execution_state 非全零")

    def to_dict(self) -> dict[str, Any]:
        """投影为规范 JSON 对象。"""
        return {
            "artifact_kind": "PH2_LC09_TRANSFER_AXIS_MANIFEST",
            "artifact_status": self.artifact_status,
            "artifact_version": self.artifact_version,
            "axis_keys": list(self.axis_keys),
            "blocked_sources": [item.to_dict() for item in self.blocked_sources],
            "execution_state": self.execution_state.to_value(),
            "format_version": self.format_version,
            "pack_audits": [item.to_dict() for item in self.pack_audits],
            "pack_inventory_count": self.pack_inventory_count,
            "runtime_status": self.runtime_status,
            "runtime_transfer_pass_authority": (
                self.runtime_transfer_pass_authority),
            "scope_contraction_protocols": list(
                self.scope_contraction_protocols),
            "split_probes": [item.to_dict() for item in self.split_probes],
            "task_key": self.task_key,
            "verifier_dimensions": list(self.verifier_dimensions),
            "verifier_ne_conditions": list(self.verifier_ne_conditions),
        }

    def canonical_bytes(self) -> bytes:
        """返回带单一结尾换行的规范 manifest 字节。"""
        return canonical_json_bytes(self.to_dict()) + b"\n"

    def sha256(self) -> str:
        """返回规范 manifest SHA-256。"""
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "LanguageTransferAxisManifest":
        """从精确字段对象恢复 LC-09 manifest。"""
        expected = {
            "artifact_kind", "artifact_status", "artifact_version",
            "axis_keys", "blocked_sources", "execution_state",
            "format_version", "pack_audits", "pack_inventory_count",
            "runtime_status", "runtime_transfer_pass_authority",
            "scope_contraction_protocols", "split_probes", "task_key",
            "verifier_dimensions", "verifier_ne_conditions",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise TransferAxisContractError("LC-09 manifest 字段不精确")
        if value["artifact_kind"] != "PH2_LC09_TRANSFER_AXIS_MANIFEST":
            raise TransferAxisContractError("LC-09 artifact_kind 非法")
        return cls(
            value["format_version"],
            str(value["artifact_version"]),
            str(value["artifact_status"]),
            str(value["runtime_status"]),
            str(value["task_key"]),
            tuple(str(item) for item in value["axis_keys"]),
            value["pack_inventory_count"],
            tuple(TransferPackAudit.from_dict(item)
                  for item in value["pack_audits"]),
            tuple(BlockedTransferSource.from_dict(item)
                  for item in value["blocked_sources"]),
            tuple(TransferSplitProbe.from_dict(item)
                  for item in value["split_probes"]),
            tuple(str(item) for item in value["verifier_dimensions"]),
            tuple(str(item) for item in value["verifier_ne_conditions"]),
            tuple(str(item) for item in value["scope_contraction_protocols"]),
            value["runtime_transfer_pass_authority"],
            CanonicalJsonObject.from_value(value["execution_state"]),
        )


def write_transfer_axis_manifest(
        manifest: LanguageTransferAxisManifest,
        path: str | Path,
        ) -> Path:
    """独占或幂等发布 LC-09 manifest，拒绝覆盖不同内容。"""
    if not isinstance(manifest, LanguageTransferAxisManifest):
        raise TransferAxisContractError("manifest 类型非法")
    target = Path(path)
    payload = manifest.canonical_bytes()
    if target.exists():
        if not target.is_file() or target.read_bytes() != payload:
            raise TransferAxisContractError("LC-09 manifest 已存在且内容不同")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(payload)
    except OSError as error:
        raise TransferAxisContractError("LC-09 manifest 无法发布") from error
    return target


def read_transfer_axis_manifest(
        path: str | Path,
        ) -> LanguageTransferAxisManifest:
    """严格回读规范 LC-09 manifest。"""
    try:
        payload = Path(path).read_bytes()
        if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
            raise TransferAxisContractError("LC-09 manifest newline 非法")
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
        assert isinstance(value, dict)
        manifest = LanguageTransferAxisManifest.from_dict(value)
    except TransferAxisContractError:
        raise
    except (OSError, UnicodeError, ValueError, AssertionError) as error:
        raise TransferAxisContractError("LC-09 manifest 损坏") from error
    if manifest.canonical_bytes() != payload:
        raise TransferAxisContractError("LC-09 manifest 非规范 JSON")
    return manifest


__all__ = [
    "ARTIFACT_STATUS",
    "AXIS_DECLARATION_STATES",
    "BlockedTransferSource",
    "COMBINATION_SPLIT_STATES",
    "EXECUTION_STATE",
    "FORMAT_VERSION",
    "LanguageTransferAxisManifest",
    "PACK_KINDS",
    "RUNTIME_STATUS",
    "SCOPE_CONTRACTION_PROTOCOLS",
    "SPLIT_PROBE_KINDS",
    "TRANSFER_AXIS_KEYS",
    "TransferAxisContractError",
    "TransferPackAudit",
    "TransferSplitProbe",
    "VERIFIER_DIMENSIONS",
    "VERIFIER_NE_CONDITIONS",
    "evaluate_transfer_split_probe",
    "read_transfer_axis_manifest",
    "write_transfer_axis_manifest",
]
