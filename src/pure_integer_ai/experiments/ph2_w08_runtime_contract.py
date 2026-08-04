"""W08-07 train artifact、逐向 Use、资源与 dump outcome 合同。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pure_integer_ai.experiments.ph2_w05_contract import digest_value
from pure_integer_ai.experiments.ph2_w08_authority import W08_DIMENSION_KEYS
from pure_integer_ai.experiments.ph2_w08_contract import (
    W08_CONSUMER_KEYS,
    W08_RESOURCE_BUDGET,
    W08_ZERO_EXECUTION_STATE,
)
from pure_integer_ai.experiments.ph2_w08_inference_contract import (
    W08_CANDIDATE_INFERENCE_INTERFACE_VERSION,
)


W08_FORMAL_RUN_ID = 9
W08_W07_BASE_RUN_ID = 8
W08_OPEN_GENERATION_PREFORMAL_STATE = "NE_NOT_YET_EVALUABLE"
W08_RUNTIME_HARD_CONJUNCT_KEYS = (
    "OPEN_GENERATION",
    "LC16_DISCOURSE_REFERENCE_GENERATION",
)
W08_RUNTIME_OWNED_TABLES = ("ph2_w08_transaction_event",)
W08_FORMAL_EXECUTION_STATE = {
    **W08_ZERO_EXECUTION_STATE,
    "W08_STARTED": 1,
    "formal_w08_training_runs": 1,
}


class W08RuntimeError(RuntimeError):
    """W08-07 artifact、Use、资源、恢复或 dump 合同发生漂移。"""


def _key(value: object, *, where: str) -> tuple[int, ...]:
    if not isinstance(value, tuple) or not value or any(type(item) is not int for item in value):
        raise W08RuntimeError(f"{where} 不是严格整数 key")
    return value


def _keys(
    values: object,
    *,
    where: str,
    allow_empty: bool = False,
) -> tuple[tuple[int, ...], ...]:
    if not isinstance(values, tuple) or any(
        not isinstance(item, tuple)
        or not item
        or any(type(value) is not int for value in item)
        for item in values
    ):
        raise W08RuntimeError(f"{where} 不是整数 key tuple")
    if not allow_empty and not values:
        raise W08RuntimeError(f"{where} 为空")
    if len(values) != len(set(values)) or values != tuple(sorted(values)):
        raise W08RuntimeError(f"{where} 不是 canonical inventory")
    return values


def _sha256(value: object, *, where: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise W08RuntimeError(f"{where} SHA-256 非法")
    return value


@dataclass(frozen=True)
class W08RuntimeConfig:
    """W08-07 public host 的隔离运行根、调度与故障配置。"""

    repository_root: str | Path
    run_root: str | Path
    sqlite_path: str | Path
    run_id: int = W08_FORMAL_RUN_ID
    parent_run_id: int = W08_W07_BASE_RUN_ID
    base_run_id: int = W08_W07_BASE_RUN_ID
    worker_count: int = 1
    mode: str = "fresh"
    fault_point: str | None = None


@dataclass(frozen=True, order=True)
class W08TrainArtifact:
    """一个由 train payload 实际编译的承重维 artifact。"""

    dimension_key: str
    artifact_kind: str
    artifact_key: tuple[int, ...]
    evidence_keys: tuple[tuple[int, ...], ...]
    record_count: int

    def __post_init__(self) -> None:
        if self.dimension_key not in W08_DIMENSION_KEYS or not self.artifact_kind:
            raise W08RuntimeError("W08 train artifact 身份非法")
        _key(self.artifact_key, where="train artifact")
        _keys(self.evidence_keys, where="train artifact Evidence")
        if type(self.record_count) is not int or self.record_count <= 0:
            raise W08RuntimeError("W08 train artifact record count 非法")

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_key": list(self.artifact_key),
            "artifact_kind": self.artifact_kind,
            "dimension_key": self.dimension_key,
            "evidence_keys": [list(item) for item in self.evidence_keys],
            "record_count": self.record_count,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "W08TrainArtifact":
        return cls(
            str(value["dimension_key"]),
            str(value["artifact_kind"]),
            tuple(value["artifact_key"]),
            tuple(tuple(item) for item in value["evidence_keys"]),
            int(value["record_count"]),
        )


@dataclass(frozen=True, order=True)
class W08RuntimeUse:
    """一个维度在一个方向上的 exact Use 与 outcome。"""

    dimension_key: str
    consumer_key: str
    request_key: tuple[int, ...]
    selected_artifact_key: tuple[int, ...]
    evidence_keys: tuple[tuple[int, ...], ...]
    directional_choice_key: tuple[int, ...]
    use_key: tuple[int, ...]
    outcome_state: str
    outcome_key: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.dimension_key not in W08_DIMENSION_KEYS:
            raise W08RuntimeError("W08 runtime Use 维度非法")
        if self.consumer_key not in W08_CONSUMER_KEYS:
            raise W08RuntimeError("W08 runtime Use consumer 非法")
        for name in (
            "request_key",
            "selected_artifact_key",
            "directional_choice_key",
            "use_key",
            "outcome_key",
        ):
            _key(getattr(self, name), where=f"runtime Use {name}")
        _keys(self.evidence_keys, where="runtime Use Evidence")
        if self.outcome_state != "RESOLVED":
            raise W08RuntimeError("W08 runtime Use outcome 未闭合")

    def to_dict(self) -> dict[str, object]:
        return {
            "consumer_key": self.consumer_key,
            "dimension_key": self.dimension_key,
            "directional_choice_key": list(self.directional_choice_key),
            "evidence_keys": [list(item) for item in self.evidence_keys],
            "outcome_key": list(self.outcome_key),
            "outcome_state": self.outcome_state,
            "request_key": list(self.request_key),
            "selected_artifact_key": list(self.selected_artifact_key),
            "use_key": list(self.use_key),
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "W08RuntimeUse":
        return cls(
            str(value["dimension_key"]),
            str(value["consumer_key"]),
            tuple(value["request_key"]),
            tuple(value["selected_artifact_key"]),
            tuple(tuple(item) for item in value["evidence_keys"]),
            tuple(value["directional_choice_key"]),
            tuple(value["use_key"]),
            str(value["outcome_state"]),
            tuple(value["outcome_key"]),
        )


@dataclass(frozen=True, order=True)
class W08HardConjunctEvidence:
    """开放生成或 LC-16 的 public bounded 证据绑定。"""

    conjunct_key: str
    state: str
    evidence_sha256: str
    evidence_key: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.conjunct_key not in W08_RUNTIME_HARD_CONJUNCT_KEYS:
            raise W08RuntimeError("W08 hard conjunct 未注册")
        if self.state != "PUBLIC_BOUNDED_PASS":
            raise W08RuntimeError("W08 hard conjunct public evidence 未闭合")
        _sha256(self.evidence_sha256, where="hard conjunct evidence")
        _key(self.evidence_key, where="hard conjunct evidence key")

    def to_dict(self) -> dict[str, object]:
        return {
            "conjunct_key": self.conjunct_key,
            "evidence_key": list(self.evidence_key),
            "evidence_sha256": self.evidence_sha256,
            "state": self.state,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "W08HardConjunctEvidence":
        return cls(
            str(value["conjunct_key"]),
            str(value["state"]),
            str(value["evidence_sha256"]),
            tuple(value["evidence_key"]),
        )


@dataclass(frozen=True, order=True)
class W08LogicalShard:
    """与物理 worker 无关的一个 canonical 逻辑 shard。"""

    shard_index: int
    record_keys: tuple[tuple[int, ...], ...]
    shard_key: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.shard_index) is not int or not 0 <= self.shard_index < 16:
            raise W08RuntimeError("W08 logical shard index 非法")
        _keys(self.record_keys, where="logical shard records", allow_empty=True)
        _key(self.shard_key, where="logical shard key")

    def to_dict(self) -> dict[str, object]:
        return {
            "record_keys": [list(item) for item in self.record_keys],
            "shard_index": self.shard_index,
            "shard_key": list(self.shard_key),
        }


@dataclass(frozen=True)
class W08RuntimeResourceReceipt:
    """W08-07 实际资源计数及 frozen 上限核验。"""

    actual_records: int
    actual_payload_bytes: int
    actual_payload_gets: int
    actual_logic_operations: int
    actual_recompute_objects: int
    actual_segments: int
    actual_checkpoint_count: int
    actual_workers: int

    def __post_init__(self) -> None:
        values = tuple(getattr(self, name) for name in self.__dataclass_fields__)
        if any(type(value) is not int or value < 0 for value in values):
            raise W08RuntimeError("W08 runtime resource count 非法")
        mapping = {
            "actual_records": "max_records",
            "actual_payload_bytes": "max_payload_bytes",
            "actual_payload_gets": "max_payload_gets",
            "actual_logic_operations": "max_logic_operations",
            "actual_recompute_objects": "max_recompute_objects",
            "actual_segments": "max_segments",
            "actual_checkpoint_count": "max_checkpoint_count",
            "actual_workers": "max_workers",
        }
        if any(getattr(self, actual) > W08_RESOURCE_BUDGET[maximum] for actual, maximum in mapping.items()):
            raise W08RuntimeError("W08 runtime resource budget 超限")

    def to_dict(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "W08RuntimeResourceReceipt":
        return cls(*(int(value[name]) for name in cls.__dataclass_fields__))


@dataclass(frozen=True)
class W08RunOutcome:
    """一次 W08-07 执行或零 payload dump readback 的公开证据。"""

    semantic_state_key: tuple[int, ...]
    artifact_commitment_key: tuple[int, ...]
    use_commitment_key: tuple[int, ...]
    hard_conjunct_commitment_key: tuple[int, ...]
    retention_commitment_key: tuple[int, ...]
    transaction_commitment_key: tuple[int, ...]
    scheduling_key: tuple[int, ...]
    dump_manifest_sha256: str
    inference_state_key: tuple[int, ...]
    inference_state_sha256: str
    inference_interface_version: str
    inference_rule_count: int
    artifacts: tuple[W08TrainArtifact, ...]
    uses: tuple[W08RuntimeUse, ...]
    hard_conjuncts: tuple[W08HardConjunctEvidence, ...]
    retention_sha256: tuple[tuple[str, str], ...]
    resource_report: W08RuntimeResourceReceipt
    owned_tables: tuple[str, ...]
    execution_state: tuple[tuple[str, int], ...]
    open_generation_state: str
    transaction_event_count: int
    compiled_artifact_count: int
    payload_gets_this_call: int
    payload_bytes_this_call: int
    teacher_calls: int
    evaluator_label_reads: int
    future_payload_reads: int
    host_learning_writes: int
    memory_learning_writes: int
    dump_readback: bool = False

    def __post_init__(self) -> None:
        for name in (
            "semantic_state_key",
            "artifact_commitment_key",
            "use_commitment_key",
            "hard_conjunct_commitment_key",
            "retention_commitment_key",
            "transaction_commitment_key",
            "scheduling_key",
            "inference_state_key",
        ):
            _key(getattr(self, name), where=f"runtime outcome {name}")
        _sha256(self.dump_manifest_sha256, where="runtime dump")
        _sha256(self.inference_state_sha256, where="runtime inference state")
        if self.inference_interface_version != W08_CANDIDATE_INFERENCE_INTERFACE_VERSION:
            raise W08RuntimeError("W08 runtime inference interface version 漂移")
        if type(self.inference_rule_count) is not int or self.inference_rule_count <= 0:
            raise W08RuntimeError("W08 runtime inference rule count 非法")
        if tuple(item.dimension_key for item in self.artifacts) != W08_DIMENSION_KEYS:
            raise W08RuntimeError("W08 runtime artifact order 漂移")
        expected_uses = tuple(
            (dimension, consumer)
            for dimension in W08_DIMENSION_KEYS
            for consumer in W08_CONSUMER_KEYS
        )
        if tuple((item.dimension_key, item.consumer_key) for item in self.uses) != expected_uses:
            raise W08RuntimeError("W08 runtime 15 条 U/R/G Use 不闭合")
        if tuple(item.conjunct_key for item in self.hard_conjuncts) != W08_RUNTIME_HARD_CONJUNCT_KEYS:
            raise W08RuntimeError("W08 runtime hard conjunct inventory 漂移")
        if self.owned_tables != W08_RUNTIME_OWNED_TABLES:
            raise W08RuntimeError("W08 runtime owned table inventory 漂移")
        if dict(self.execution_state) not in (
            W08_ZERO_EXECUTION_STATE,
            W08_FORMAL_EXECUTION_STATE,
        ):
            raise W08RuntimeError("W08 runtime formal 状态非法")
        if self.open_generation_state != W08_OPEN_GENERATION_PREFORMAL_STATE:
            raise W08RuntimeError("W08 runtime 提前改变 OPEN_GENERATION")
        if self.transaction_event_count != 5 or self.compiled_artifact_count != len(self.artifacts):
            raise W08RuntimeError("W08 runtime transaction/artifact count 漂移")
        counts = (
            self.payload_gets_this_call,
            self.payload_bytes_this_call,
            self.teacher_calls,
            self.evaluator_label_reads,
            self.future_payload_reads,
            self.host_learning_writes,
            self.memory_learning_writes,
        )
        if any(type(value) is not int or value < 0 for value in counts):
            raise W08RuntimeError("W08 runtime audit count 非法")
        if any(counts[2:]):
            raise W08RuntimeError("W08 runtime 越过 teacher/private/future/host 边界")
        if self.dump_readback and (self.payload_gets_this_call or self.payload_bytes_this_call):
            raise W08RuntimeError("W08 dump readback 二次读取了 train payload")

    def canonical_key(self) -> tuple[int, ...]:
        """排除物理调度与恢复历史，返回 worker/mode 无关的语义身份。"""
        return self.semantic_state_key


def build_semantic_state_key(
    artifacts: tuple[W08TrainArtifact, ...],
    uses: tuple[W08RuntimeUse, ...],
    hard_conjuncts: tuple[W08HardConjunctEvidence, ...],
    retention: tuple[tuple[str, str], ...],
    inference_state_key: tuple[int, ...],
) -> tuple[int, ...]:
    return digest_value(
        {
            "artifacts": [item.to_dict() for item in artifacts],
            "hard_conjuncts": [item.to_dict() for item in hard_conjuncts],
            "inference_state_key": list(inference_state_key),
            "retention": [list(item) for item in retention],
            "uses": [item.to_dict() for item in uses],
        }
    )


__all__ = [
    "W08_FORMAL_RUN_ID",
    "W08_FORMAL_EXECUTION_STATE",
    "W08_RUNTIME_HARD_CONJUNCT_KEYS",
    "W08_RUNTIME_OWNED_TABLES",
    "W08_OPEN_GENERATION_PREFORMAL_STATE",
    "W08_W07_BASE_RUN_ID",
    "W08HardConjunctEvidence",
    "W08LogicalShard",
    "W08RunOutcome",
    "W08RuntimeConfig",
    "W08RuntimeError",
    "W08RuntimeResourceReceipt",
    "W08RuntimeUse",
    "W08TrainArtifact",
    "build_semantic_state_key",
]
