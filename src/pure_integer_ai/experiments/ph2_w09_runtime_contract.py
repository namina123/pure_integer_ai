"""W09-08 transaction freeze 的 metadata、资源和规范结果合同。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pure_integer_ai.experiments.ph2_w05_contract import digest_value
from pure_integer_ai.experiments.ph2_w09_authority import (
    W09_ALLOWED_WORKER_COUNTS,
    W09_DIMENSION_KEYS,
    W09_FAILURE_POINT_KEYS,
    W09_RESOURCE_BUDGET,
    W09_ZERO_EXECUTION_STATE,
)
from pure_integer_ai.experiments.ph2_w09_contract import W09_OWNER_KEY


W09_RUNTIME_RUN_ID = 10
W09_RUNTIME_PARENT_RUN_ID = 9
W09_RUNTIME_DUMP_NAME = "w09_runtime_evidence_dump.json"
W09_RUNTIME_OWNED_TABLES = ("ph2_w09_transaction_event",)
W09_RUNTIME_STATUS = "PUBLIC_BOUNDED_PASS"
W09_RUNTIME_J_LC_STATUS = "PUBLIC_BOUNDED_NOT_FORMAL"
W09_RUNTIME_COMPONENT_KEYS = W09_DIMENSION_KEYS
W09_RUNTIME_ABLATION_KEYS = tuple(
    f"{component}-ABLATION" for component in W09_DIMENSION_KEYS
)
W09_RUNTIME_AUXILIARY_KEYS = (
    "WINDOW-1",
    "WINDOW-2",
    "WINDOW-3",
    "J-LC-W09",
    "V-06-CLONE",
    "ROLLBACK-AUDIT",
)


class W09RuntimeError(RuntimeError):
    """W09-08 运行、恢复、规范 dump 或独立 owner 合同发生漂移。"""


def _key(value: object, *, where: str, allow_empty: bool = False) -> tuple[int, ...]:
    """校验只含 Python int 的稳定 identity。"""
    if (
        not isinstance(value, tuple)
        or (not allow_empty and not value)
        or any(type(item) is not int or item < 0 for item in value)
    ):
        raise W09RuntimeError(f"{where} 不是严格整数 key")
    return value


def _keys(value: object, *, where: str, allow_empty: bool = False) -> tuple[tuple[int, ...], ...]:
    """校验无重复、排序的 key inventory。"""
    if not isinstance(value, tuple):
        raise W09RuntimeError(f"{where} 不是 key tuple")
    result = tuple(_key(item, where=where) for item in value)
    if not allow_empty and not result:
        raise W09RuntimeError(f"{where} 为空")
    if len(result) != len(set(result)) or result != tuple(sorted(result)):
        raise W09RuntimeError(f"{where} 不是 canonical inventory")
    return result


def _sha(value: object, *, where: str) -> str:
    """校验小写 SHA-256 字符串。"""
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(item not in "0123456789abcdef" for item in value)
    ):
        raise W09RuntimeError(f"{where} SHA-256 非法")
    return value


def _sha_key(value: tuple[int, ...]) -> str:
    """把稳定整数身份转成公开 receipt 可用的 SHA。"""
    return bytes(digest_value({"state": list(value)})).hex()


@dataclass(frozen=True)
class W09RuntimeConfig:
    """W09-08 public runtime 的隔离运行根、调度和故障配置。"""

    repository_root: str | Path
    run_root: str | Path
    sqlite_path: str | Path
    run_id: int = W09_RUNTIME_RUN_ID
    parent_run_id: int = W09_RUNTIME_PARENT_RUN_ID
    base_run_id: int = W09_RUNTIME_PARENT_RUN_ID
    worker_count: int = 1
    mode: str = "fresh"
    fault_point: str | None = None


@dataclass(frozen=True)
class W09LogicalShard:
    """不受物理 worker 影响的十六个逻辑分片。"""

    shard_index: int
    event_keys: tuple[tuple[int, ...], ...]
    shard_key: tuple[int, ...]

    def __post_init__(self) -> None:
        """核验分片范围、事件 inventory 和确定性 shard identity。"""
        if type(self.shard_index) is not int or not 0 <= self.shard_index < 16:
            raise W09RuntimeError("W-09 logical shard index 非法")
        _keys(self.event_keys, where="logical shard events", allow_empty=True)
        _key(self.shard_key, where="logical shard key")

    def to_dict(self) -> dict[str, Any]:
        """转换为 metadata-only JSON 对象。"""
        return {
            "event_keys": [list(item) for item in self.event_keys],
            "shard_index": self.shard_index,
            "shard_key": list(self.shard_key),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "W09LogicalShard":
        """从规范 JSON 对象恢复逻辑分片。"""
        if not isinstance(value, dict):
            raise W09RuntimeError("W-09 logical shard object 非法")
        return cls(
            int(value["shard_index"]),
            tuple(tuple(int(item) for item in key) for key in value["event_keys"]),
            tuple(int(item) for item in value["shard_key"]),
        )


@dataclass(frozen=True)
class W09RuntimeComponentReceipt:
    """一个维度或辅助结果的 owner、result 和 receipt 三重身份。"""

    component_key: str
    owner_key: str
    result_key: tuple[int, ...]
    receipt_key: tuple[int, ...]
    status: str

    def __post_init__(self) -> None:
        """要求 component 顺序外的身份仍互相独立且只发布 bounded 状态。"""
        if not isinstance(self.component_key, str) or not self.component_key:
            raise W09RuntimeError("W-09 component key 非法")
        if self.owner_key != W09_OWNER_KEY:
            raise W09RuntimeError("W-09 component owner 混写")
        _key(self.result_key, where=f"{self.component_key} result")
        _key(self.receipt_key, where=f"{self.component_key} receipt")
        if self.status != W09_RUNTIME_STATUS:
            raise W09RuntimeError("W-09 component 尚未 bounded PASS")
        if self.result_key == self.receipt_key:
            raise W09RuntimeError("W-09 component result/receipt identity 重合")

    def to_dict(self) -> dict[str, Any]:
        """转换为不含 payload surface 的 receipt metadata。"""
        return {
            "component_key": self.component_key,
            "owner_key": self.owner_key,
            "receipt_key": list(self.receipt_key),
            "result_key": list(self.result_key),
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "W09RuntimeComponentReceipt":
        """从规范 JSON 对象恢复 component receipt。"""
        if not isinstance(value, dict):
            raise W09RuntimeError("W-09 component receipt object 非法")
        return cls(
            str(value["component_key"]),
            str(value["owner_key"]),
            tuple(int(item) for item in value["result_key"]),
            tuple(int(item) for item in value["receipt_key"]),
            str(value["status"]),
        )

    def stable_key(self) -> tuple[int, ...]:
        """返回同时包含 result/receipt 的独立 component identity。"""
        return digest_value({
            "component": self.component_key,
            "owner": self.owner_key,
            "receipt": list(self.receipt_key),
            "result": list(self.result_key),
            "status": self.status,
        })


@dataclass(frozen=True)
class W09ResourceNormalization:
    """绑定 manifest 上限且排除物理 worker 的资源归一账。"""

    counts: tuple[tuple[str, int], ...]
    canonical_result_key: tuple[int, ...]

    def __post_init__(self) -> None:
        """核验完整排序计数与 worker-independent canonical key。"""
        if tuple(key for key, _ in self.counts) != tuple(sorted(W09_RESOURCE_BUDGET)):
            raise W09RuntimeError("W-09 normalized resource fields 漂移")
        if len(self.counts) != len(W09_RESOURCE_BUDGET):
            raise W09RuntimeError("W-09 normalized resource fields 不完整")
        if any(
            type(value) is not int
            or value < 0
            or value > W09_RESOURCE_BUDGET[key]
            for key, value in self.counts
        ):
            raise W09RuntimeError("W-09 normalized resource count 超限")
        _key(self.canonical_result_key, where="W-09 normalized resource result")

    def to_dict(self) -> dict[str, Any]:
        """转换为 canonical resource metadata。"""
        return {
            "canonical_result_key": list(self.canonical_result_key),
            "counts": {key: value for key, value in self.counts},
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "W09ResourceNormalization":
        """从规范 JSON 对象恢复资源归一账。"""
        if not isinstance(value, dict):
            raise W09RuntimeError("W-09 normalized resource object 非法")
        return cls(
            tuple((key, int(value["counts"][key])) for key in sorted(W09_RESOURCE_BUDGET)),
            tuple(int(item) for item in value["canonical_result_key"]),
        )

    def stable_key(self) -> tuple[int, ...]:
        """返回不含 worker scheduling 的资源 identity。"""
        return digest_value(self.to_dict())


@dataclass(frozen=True)
class W09RuntimeEvidence:
    """五维、三窗口、J-LC、clone、rollback 共 owner 的现场证据。"""

    host_state_key: tuple[int, ...]
    dimension_receipts: tuple[W09RuntimeComponentReceipt, ...]
    ablation_receipts: tuple[W09RuntimeComponentReceipt, ...]
    window_receipts: tuple[W09RuntimeComponentReceipt, ...]
    j_lc_receipt: W09RuntimeComponentReceipt
    clone_receipt: W09RuntimeComponentReceipt
    rollback_receipt: W09RuntimeComponentReceipt
    resource_normalization: W09ResourceNormalization
    logical_shards: tuple[W09LogicalShard, ...]
    learning_event_keys: tuple[tuple[int, ...], ...]
    payload_gets: int
    payload_bytes: int
    teacher_calls: int
    api_calls: int
    llm_calls: int
    host_write_count: int

    def __post_init__(self) -> None:
        """以硬合取校验覆盖、身份独立、十六分片和零调用隔离。"""
        _key(self.host_state_key, where="W-09 host state")
        if tuple(item.component_key for item in self.dimension_receipts) != W09_RUNTIME_COMPONENT_KEYS:
            raise W09RuntimeError("W-09 five dimension receipt coverage 漂移")
        if tuple(item.component_key for item in self.ablation_receipts) != W09_RUNTIME_ABLATION_KEYS:
            raise W09RuntimeError("W-09 five bearing ablation receipt coverage 漂移")
        if tuple(item.component_key for item in self.window_receipts) != W09_RUNTIME_AUXILIARY_KEYS[:3]:
            raise W09RuntimeError("W-09 three window receipt coverage 漂移")
        auxiliary = (
            self.j_lc_receipt,
            self.clone_receipt,
            self.rollback_receipt,
        )
        if tuple(item.component_key for item in auxiliary) != W09_RUNTIME_AUXILIARY_KEYS[3:]:
            raise W09RuntimeError("W-09 auxiliary receipt coverage 漂移")
        receipts = (
            *self.dimension_receipts,
            *self.ablation_receipts,
            *self.window_receipts,
            *auxiliary,
        )
        if any(not isinstance(item, W09RuntimeComponentReceipt) for item in receipts):
            raise W09RuntimeError("W-09 receipt type 不完整")
        if len({item.result_key for item in receipts}) != len(receipts):
            raise W09RuntimeError("W-09 component result identity 被复用")
        if len({item.receipt_key for item in receipts}) != len(receipts):
            raise W09RuntimeError("W-09 component receipt identity 被复用")
        if set(item.result_key for item in receipts).intersection(item.receipt_key for item in receipts):
            raise W09RuntimeError("W-09 result 与 receipt identity 混用")
        if len(self.logical_shards) != 16 or tuple(item.shard_index for item in self.logical_shards) != tuple(range(16)):
            raise W09RuntimeError("W-09 logical shard inventory 不完整")
        _keys(self.learning_event_keys, where="W-09 learning events")
        if any(
            type(value) is not int or value < 0
            for value in (
                self.payload_gets,
                self.payload_bytes,
                self.teacher_calls,
                self.api_calls,
                self.llm_calls,
                self.host_write_count,
            )
        ):
            raise W09RuntimeError("W-09 runtime audit count 非法")
        if any((self.teacher_calls, self.api_calls, self.llm_calls, self.host_write_count)):
            raise W09RuntimeError("W-09 runtime crossed zero-call or host fence")

    @property
    def all_receipts(self) -> tuple[W09RuntimeComponentReceipt, ...]:
        """按固定顺序返回全部独立 receipt。"""
        return (
            *self.dimension_receipts,
            *self.ablation_receipts,
            *self.window_receipts,
            self.j_lc_receipt,
            self.clone_receipt,
            self.rollback_receipt,
        )

    def semantic_key(self) -> tuple[int, ...]:
        """返回排除 worker、mode、run 和恢复历史的规范语义身份。"""
        return digest_value({
            "dimensions": [list(item.stable_key()) for item in self.dimension_receipts],
            "ablations": [list(item.stable_key()) for item in self.ablation_receipts],
            "events": [list(item) for item in self.learning_event_keys],
            "host": list(self.host_state_key),
            "j_lc": list(self.j_lc_receipt.stable_key()),
            "logical_shards": [item.to_dict() for item in self.logical_shards],
            "resource": list(self.resource_normalization.stable_key()),
            "rollback": list(self.rollback_receipt.stable_key()),
            "v06": list(self.clone_receipt.stable_key()),
            "windows": [list(item.stable_key()) for item in self.window_receipts],
        })

    def to_dict(self) -> dict[str, Any]:
        """转换为不含原文、surface、expected 或 label 的 metadata dump。"""
        return {
            "ablation_receipts": [item.to_dict() for item in self.ablation_receipts],
            "auxiliary_receipts": [
                self.j_lc_receipt.to_dict(),
                self.clone_receipt.to_dict(),
                self.rollback_receipt.to_dict(),
            ],
            "dimension_receipts": [item.to_dict() for item in self.dimension_receipts],
            "host_state_key": list(self.host_state_key),
            "learning_event_keys": [list(item) for item in self.learning_event_keys],
            "logical_shards": [item.to_dict() for item in self.logical_shards],
            "payload_audit": {
                "api_calls": self.api_calls,
                "host_write_count": self.host_write_count,
                "llm_calls": self.llm_calls,
                "payload_bytes": self.payload_bytes,
                "payload_gets": self.payload_gets,
                "teacher_calls": self.teacher_calls,
            },
            "resource_normalization": self.resource_normalization.to_dict(),
            "window_receipts": [item.to_dict() for item in self.window_receipts],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "W09RuntimeEvidence":
        """从 canonical metadata dump 恢复并重新执行所有不变量。"""
        if not isinstance(value, dict):
            raise W09RuntimeError("W-09 runtime evidence object 非法")
        audit = value["payload_audit"]
        auxiliary = tuple(
            W09RuntimeComponentReceipt.from_dict(item)
            for item in value["auxiliary_receipts"]
        )
        return cls(
            tuple(int(item) for item in value["host_state_key"]),
            tuple(W09RuntimeComponentReceipt.from_dict(item) for item in value["dimension_receipts"]),
            tuple(W09RuntimeComponentReceipt.from_dict(item) for item in value["ablation_receipts"]),
            tuple(W09RuntimeComponentReceipt.from_dict(item) for item in value["window_receipts"]),
            auxiliary[0],
            auxiliary[1],
            auxiliary[2],
            W09ResourceNormalization.from_dict(value["resource_normalization"]),
            tuple(W09LogicalShard.from_dict(item) for item in value["logical_shards"]),
            tuple(tuple(int(part) for part in item) for item in value["learning_event_keys"]),
            int(audit["payload_gets"]),
            int(audit["payload_bytes"]),
            int(audit["teacher_calls"]),
            int(audit["api_calls"]),
            int(audit["llm_calls"]),
            int(audit["host_write_count"]),
        )


@dataclass(frozen=True)
class W09RunOutcome:
    """一次 W09-08 执行或零 payload dump readback 的公开结果。"""

    semantic_state_key: tuple[int, ...]
    transaction_commitment_key: tuple[int, ...]
    scheduling_key: tuple[int, ...]
    dump_manifest_sha256: str
    evidence: W09RuntimeEvidence
    transaction_event_count: int
    execution_state: tuple[tuple[str, int], ...]
    formal_evidenced: int
    language_capability_mastered: int
    language_readiness: int
    payload_gets_this_call: int
    payload_bytes_this_call: int
    dump_readback: bool = False

    def __post_init__(self) -> None:
        """核验最终状态仍是 public bounded、零 formal 写入且语义可回读。"""
        _key(self.semantic_state_key, where="W-09 outcome semantic")
        _key(self.transaction_commitment_key, where="W-09 outcome transaction")
        _key(self.scheduling_key, where="W-09 outcome scheduling")
        _sha(self.dump_manifest_sha256, where="W-09 runtime dump")
        if not isinstance(self.evidence, W09RuntimeEvidence):
            raise W09RuntimeError("W-09 outcome evidence type 非法")
        if self.semantic_state_key != self.evidence.semantic_key():
            raise W09RuntimeError("W-09 outcome semantic key 漂移")
        if self.transaction_event_count != 5:
            raise W09RuntimeError("W-09 transaction 未形成五事件")
        if tuple(self.execution_state) != tuple(sorted(W09_ZERO_EXECUTION_STATE.items())):
            raise W09RuntimeError("W-09 execution state 漂移")
        if any(value != 0 for value in (
            self.formal_evidenced,
            self.language_capability_mastered,
            self.language_readiness,
        )):
            raise W09RuntimeError("W09-08 提前发布 formal/mastered/readiness")
        if any(
            type(value) is not int or value < 0
            for value in (self.payload_gets_this_call, self.payload_bytes_this_call)
        ):
            raise W09RuntimeError("W-09 outcome payload audit count 非法")
        if self.dump_readback and (
            self.payload_gets_this_call or self.payload_bytes_this_call
        ):
            raise W09RuntimeError("W-09 dump readback 二次读取 train payload")

    def canonical_key(self) -> tuple[int, ...]:
        """返回 worker/mode/fresh-resume 无关的语义 key。"""
        return self.semantic_state_key


def build_w09_resource_normalization(
    counts: tuple[tuple[str, int], ...],
    canonical_result_key: tuple[int, ...],
) -> W09ResourceNormalization:
    """用当前窗口实际资源计数创建规范归一账。"""
    return W09ResourceNormalization(counts, canonical_result_key)


__all__ = [
    "W09LogicalShard",
    "W09ResourceNormalization",
    "W09RunOutcome",
    "W09RuntimeComponentReceipt",
    "W09RuntimeConfig",
    "W09RuntimeError",
    "W09RuntimeEvidence",
    "W09_RUNTIME_AUXILIARY_KEYS",
    "W09_RUNTIME_ABLATION_KEYS",
    "W09_RUNTIME_COMPONENT_KEYS",
    "W09_RUNTIME_DUMP_NAME",
    "W09_RUNTIME_J_LC_STATUS",
    "W09_RUNTIME_OWNED_TABLES",
    "W09_RUNTIME_PARENT_RUN_ID",
    "W09_RUNTIME_RUN_ID",
    "W09_RUNTIME_STATUS",
    "build_w09_resource_normalization",
]
