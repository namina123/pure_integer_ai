"""PH2-D03-V2 successor 的身份、owner、split、预算和失效政策。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    D03ContractError,
    STAGE_KEYS,
    canonical_json_bytes,
    exact_dict,
    flag,
    positive,
    sha256_text,
    text,
)


V2_AUTHORITY_FORMAT_VERSION = 1
V2_RELEASE_KEY = "PH2-D03-V2"
V2_RELEASE_VERSION = "PH2-D03-formal-successor-v2"
V2_SCHEMA_VERSION = 2
V2_COURSE_VERSION = 2
V2_ADAPTER_VERSION = 2
V2_GENERATOR_VERSION = 2
V2_PARSER_VERSION = 2
V2_CARRIER_SCHEMA_VERSION = 2
V2_EXECUTION_STAGES = ("FT00", *STAGE_KEYS[1:], "PW")
V2_OWNER_KEYS = (
    "PH2_V2_CANDIDATE",
    "PH2_V2_TEACHER",
    "PH2_V2_DEV_CALIBRATOR",
    "PH2_V2_SHADOW_AUDITOR",
    "PH2_V2_PRIVATE_EVALUATOR",
)
V2_OWNER_MODES = ("candidate", "teacher", "dev", "shadow", "private_evaluator")
V2_SPLITS = ("train", "dev", "held_out", "adversarial", "wall")
V2_SCALE_KEYS = ("P0", "P1", "P2", "HARD_CEILING")
V2_RUN_SCALE_KEYS = ("P0", "P1", "P2")
V2_SCALE_RECORD_LIMITS = {
    "P0": 3_200,
    "P1": 12_800,
    "P2": 51_200,
    "HARD_CEILING": 900_000,
}
V2_DEFERRED_P3_MIN_RECORDS = 100_000
V2_DEFERRED_P3_MAX_RECORDS = 300_000
V2_P3_ACTIVATION_POLICY = "FREEZE_ONLY_AFTER_P2_SLOPE_PASS"
V2_ALLOWED_WORKERS = (1, 2, 4)
V2_LOGICAL_SHARD_COUNT = 128
V2_CHECKPOINT_FORMAT_VERSION = 1
V2_RUN_ID_POLICY = "NEW_POSITIVE_INTEGER_REQUIRED"
V2_MERGE_BARRIER_KEY = "PH2-D03-V2-CANONICAL-MERGE-BARRIER-V1"
V2_RUN_IDENTITY_FIELDS = (
    "release_key", "stage_key", "scale_key", "run_id",
    "logical_shard_count", "input_manifest_sha256", "parent_run_sha256",
)
V2_CHECKPOINT_IDENTITY_FIELDS = (
    "release_key", "run_identity_sha256", "owner_key", "pack_key",
    "source_state_sha256", "logical_shard_index", "cursor_record_key",
    "input_manifest_sha256",
)
V2_INVALIDATION_FORMAT_VERSION = 1
V2_INVALIDATION_VERSION = "PH2-D03-V2-invalidation-graph-v1"
V2_CONTRACT_KIND = "PH2_D03_V2_SUCCESSOR_CONTRACT"
V2_CONTRACT_VERSION = "PH2-D03-V2-successor-contract-v1"
V2_CONTRACT_PATH = (
    "data/ph2/manifests/d03_v2/"
    "ph2_d03_v2_successor_contract_v1.json"
)
V1_PUBLIC_RECEIPT_PATH = (
    "data/ph2/manifests/d03_v1/"
    "ph2_d03_post_publication_receipt_v1.json"
)
V1_PUBLIC_RECEIPT_SIZE_BYTES = 49_842
V1_PUBLIC_RECEIPT_SHA256 = (
    "8efd5f8c559bb22f0d2587fea4d38ee94d2dc10cf13ca0f787f3489f45847aef"
)
V2_INITIAL_EXECUTION_STATE = {
    "PH2_D03_V2_PUBLISHED": 0,
    "FT00_COMPLETE": 0,
    "FORMAL_TRAINING_RUNS": 0,
    "LANGUAGE_CAPABILITY_MASTERED": 0,
    "LANGUAGE_READINESS": 0,
    "TEACHER_EXIT_EVIDENCED": 0,
}


def _ordered_unique(values: tuple[str, ...], *, where: str) -> tuple[str, ...]:
    """要求文本序列无重复并保持调用方冻结顺序。"""
    if not isinstance(values, tuple) or not values:
        raise D03ContractError(f"{where} 必须是非空 tuple")
    if any(not isinstance(value, str) or not value for value in values):
        raise D03ContractError(f"{where} 含非法文本")
    if len(values) != len(set(values)):
        raise D03ContractError(f"{where} 不得重复")
    return values


@dataclass(frozen=True)
class V2OwnerPolicy:
    """冻结一个 owner namespace 的读取、写入和可见 split。"""

    owner_key: str
    mode: str
    allowed_splits: tuple[str, ...]
    writable_targets: tuple[str, ...]
    readable_private: int

    def __post_init__(self) -> None:
        text(self.owner_key, where="v2 owner_key")
        if self.owner_key not in V2_OWNER_KEYS:
            raise D03ContractError("v2 owner_key 未注册")
        if self.mode not in V2_OWNER_MODES:
            raise D03ContractError("v2 owner mode 未注册")
        if self.allowed_splits != tuple(
                split for split in V2_SPLITS if split in self.allowed_splits):
            raise D03ContractError("v2 owner split 必须按冻结顺序")
        if any(split not in V2_SPLITS for split in self.allowed_splits):
            raise D03ContractError("v2 owner 含未知 split")
        if any(not isinstance(target, str) or not target for target in self.writable_targets):
            raise D03ContractError("v2 owner writable target 非法")
        flag(self.readable_private, where="v2 owner readable_private")
        if self.mode in {"candidate", "teacher", "dev", "shadow"} and self.readable_private:
            raise D03ContractError("非 evaluator owner 不得读取 private")
        if self.mode == "private_evaluator" and self.readable_private != 1:
            raise D03ContractError("private evaluator 必须显式标 private read")

    def to_dict(self) -> dict[str, Any]:
        """导出 owner policy。"""
        return {
            "allowed_splits": list(self.allowed_splits),
            "mode": self.mode,
            "owner_key": self.owner_key,
            "readable_private": self.readable_private,
            "writable_targets": list(self.writable_targets),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "V2OwnerPolicy":
        """从严格 object 恢复 owner policy。"""
        raw = exact_dict(value, {
            "allowed_splits", "mode", "owner_key", "readable_private",
            "writable_targets",
        }, where="V2OwnerPolicy")
        if not isinstance(raw["allowed_splits"], list) or not isinstance(
                raw["writable_targets"], list):
            raise D03ContractError("v2 owner arrays 类型非法")
        return cls(
            str(raw["owner_key"]), str(raw["mode"]),
            tuple(str(item) for item in raw["allowed_splits"]),
            tuple(str(item) for item in raw["writable_targets"]),
            raw["readable_private"],
        )


OWNER_POLICIES = (
    V2OwnerPolicy("PH2_V2_CANDIDATE", "candidate", ("train",), ("candidate",), 0),
    V2OwnerPolicy("PH2_V2_TEACHER", "teacher", ("train",), ("teacher",), 0),
    V2OwnerPolicy("PH2_V2_DEV_CALIBRATOR", "dev", ("dev",), (), 0),
    V2OwnerPolicy(
        "PH2_V2_SHADOW_AUDITOR", "shadow", ("train", "dev"), ("shadow",), 0),
    V2OwnerPolicy(
        "PH2_V2_PRIVATE_EVALUATOR", "private_evaluator",
        ("held_out", "adversarial", "wall"), ("private_evaluator",), 1,
    ),
)


@dataclass(frozen=True)
class V2SplitPolicy:
    """冻结按来源/文档/内容/模板/结构簇切分的整数策略。"""

    split_keys: tuple[str, ...]
    cluster_dimensions: tuple[str, ...]
    train_numerator: int
    dev_numerator: int
    held_out_numerator: int
    adversarial_numerator: int
    denominator: int
    wall_independent: int

    def __post_init__(self) -> None:
        if self.split_keys != V2_SPLITS:
            raise D03ContractError("v2 split 集合或顺序漂移")
        _ordered_unique(self.cluster_dimensions, where="v2 split cluster dimensions")
        values = (
            self.train_numerator, self.dev_numerator,
            self.held_out_numerator, self.adversarial_numerator,
        )
        if type(self.denominator) is not int or self.denominator <= 0:
            raise D03ContractError("v2 split denominator 非法")
        if any(type(value) is not int or value < 0 for value in values):
            raise D03ContractError("v2 split numerator 非法")
        if sum(values) != self.denominator:
            raise D03ContractError("v2 split numerators 不闭合")
        flag(self.wall_independent, where="v2 wall_independent")
        if self.wall_independent != 1:
            raise D03ContractError("wall 必须是独立集合")

    def to_dict(self) -> dict[str, Any]:
        """导出 split policy。"""
        return {
            "adversarial_numerator": self.adversarial_numerator,
            "cluster_dimensions": list(self.cluster_dimensions),
            "denominator": self.denominator,
            "dev_numerator": self.dev_numerator,
            "held_out_numerator": self.held_out_numerator,
            "split_keys": list(self.split_keys),
            "train_numerator": self.train_numerator,
            "wall_independent": self.wall_independent,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "V2SplitPolicy":
        """从严格 object 恢复 split policy。"""
        raw = exact_dict(value, {
            "adversarial_numerator", "cluster_dimensions", "denominator",
            "dev_numerator", "held_out_numerator", "split_keys",
            "train_numerator", "wall_independent",
        }, where="V2SplitPolicy")
        if not all(isinstance(raw[key], list) for key in (
                "cluster_dimensions", "split_keys")):
            raise D03ContractError("v2 split arrays 类型非法")
        return cls(
            tuple(str(item) for item in raw["split_keys"]),
            tuple(str(item) for item in raw["cluster_dimensions"]),
            raw["train_numerator"], raw["dev_numerator"],
            raw["held_out_numerator"], raw["adversarial_numerator"],
            raw["denominator"], raw["wall_independent"],
        )


V2_SPLIT_POLICY = V2SplitPolicy(
    V2_SPLITS,
    ("source_cluster", "document_cluster", "entity_graph_cluster",
     "content_cluster", "template_cluster", "shape_cluster"),
    70, 10, 15, 5, 100, 1,
)


@dataclass(frozen=True)
class V2ScaleBudget:
    """冻结一个规模档位的记录硬上限和运行资源上限。"""

    scale_key: str
    max_records: int
    max_workers: int
    max_logical_shards: int
    max_checkpoint_count: int

    def __post_init__(self) -> None:
        if self.scale_key not in V2_SCALE_KEYS:
            raise D03ContractError("v2 scale key 未注册")
        if self.max_records != V2_SCALE_RECORD_LIMITS[self.scale_key]:
            raise D03ContractError("v2 scale record ceiling 漂移")
        positive(self.max_workers, where="v2 scale max_workers")
        if self.max_workers != 4:
            raise D03ContractError("v2 scale 必须允许最多 4 worker")
        positive(self.max_logical_shards, where="v2 scale logical shards")
        positive(self.max_checkpoint_count, where="v2 scale checkpoints")

    def to_dict(self) -> dict[str, Any]:
        """导出规模预算。"""
        return {
            "max_checkpoint_count": self.max_checkpoint_count,
            "max_logical_shards": self.max_logical_shards,
            "max_records": self.max_records,
            "max_workers": self.max_workers,
            "scale_key": self.scale_key,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "V2ScaleBudget":
        """从严格 object 恢复规模预算。"""
        raw = exact_dict(value, {
            "max_checkpoint_count", "max_logical_shards", "max_records",
            "max_workers", "scale_key",
        }, where="V2ScaleBudget")
        return cls(
            str(raw["scale_key"]), raw["max_records"], raw["max_workers"],
            raw["max_logical_shards"], raw["max_checkpoint_count"],
        )


V2_SCALE_BUDGETS = tuple(
    V2ScaleBudget(key, limit, 4, V2_LOGICAL_SHARD_COUNT, 256)
    for key, limit in V2_SCALE_RECORD_LIMITS.items()
)


@dataclass(frozen=True, order=True)
class V2RunIdentity:
    """冻结与 worker 数无关的语义运行身份和输入摘要。"""

    release_key: str
    stage_key: str
    scale_key: str
    run_id: int
    logical_shard_count: int
    input_manifest_sha256: str
    parent_run_sha256: str

    def __post_init__(self) -> None:
        if self.release_key != V2_RELEASE_KEY:
            raise D03ContractError("v2 run release identity 漂移")
        if self.stage_key not in V2_EXECUTION_STAGES:
            raise D03ContractError("v2 run stage 未注册")
        if self.scale_key not in V2_RUN_SCALE_KEYS:
            raise D03ContractError("v2 run scale 未注册")
        positive(self.run_id, where="v2 run id")
        if self.logical_shard_count != V2_LOGICAL_SHARD_COUNT:
            raise D03ContractError("v2 logical shard count 漂移")
        sha256_text(self.input_manifest_sha256, where="v2 input manifest sha256")
        if self.parent_run_sha256:
            sha256_text(self.parent_run_sha256, where="v2 parent run sha256")

    def to_dict(self) -> dict[str, Any]:
        """导出规范运行身份。"""
        return {
            "input_manifest_sha256": self.input_manifest_sha256,
            "logical_shard_count": self.logical_shard_count,
            "parent_run_sha256": self.parent_run_sha256,
            "release_key": self.release_key,
            "run_id": self.run_id,
            "scale_key": self.scale_key,
            "stage_key": self.stage_key,
        }

    def sha256(self) -> str:
        """返回运行身份规范摘要，供 checkpoint 绑定。"""
        return hashlib.sha256(canonical_json_bytes(self.to_dict())).hexdigest()

    @classmethod
    def from_dict(cls, value: Any) -> "V2RunIdentity":
        """从严格 object 恢复运行身份。"""
        raw = exact_dict(value, {
            "input_manifest_sha256", "logical_shard_count", "parent_run_sha256",
            "release_key", "run_id", "scale_key", "stage_key",
        }, where="V2RunIdentity")
        return cls(
            str(raw["release_key"]), str(raw["stage_key"]),
            str(raw["scale_key"]), raw["run_id"], raw["logical_shard_count"],
            str(raw["input_manifest_sha256"]),
            str(raw["parent_run_sha256"]),
        )


V2_CHANGE_KINDS = (
    "CODE_VERSION", "SCHEMA_VERSION", "OWNER_NAMESPACE", "SPLIT_POLICY",
    "RESOURCE_POLICY", "RUNNER_VERSION", "COURSE_VERSION", "PARSER_VERSION",
    "CARRIER_SCHEMA", "PACK_CONTENT", "SOURCE_SET", "LICENSE",
    "EVALUATOR_VERSION",
)
V2_PACK_INVALIDATION_KINDS = ("PACK_CONTENT", "SOURCE_SET", "LICENSE")
V2_PACK_EARLIEST_STAGES = V2_EXECUTION_STAGES[1:-1]
V2_UNKNOWN_INVALIDATION_POLICY = "FAIL_CLOSED"


@dataclass(frozen=True, order=True)
class V2InvalidationRule:
    """冻结一个变化主题的最早失效阶段和完整后缀。"""

    change_kind: str
    subject_key: str
    earliest_stage: str
    suffix: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.change_kind not in V2_CHANGE_KINDS:
            raise D03ContractError("v2 invalidation change kind 未注册")
        text(self.subject_key, where="v2 invalidation subject")
        if self.earliest_stage not in V2_EXECUTION_STAGES:
            raise D03ContractError("v2 invalidation earliest stage 非法")
        expected = V2_EXECUTION_STAGES[V2_EXECUTION_STAGES.index(self.earliest_stage):]
        if self.suffix != expected:
            raise D03ContractError("v2 invalidation suffix 不完整或乱序")

    def to_dict(self) -> dict[str, Any]:
        """导出失效规则。"""
        return {
            "change_kind": self.change_kind,
            "earliest_stage": self.earliest_stage,
            "subject_key": self.subject_key,
            "suffix": list(self.suffix),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "V2InvalidationRule":
        """从严格 object 恢复失效规则。"""
        raw = exact_dict(value, {
            "change_kind", "earliest_stage", "subject_key", "suffix",
        }, where="V2InvalidationRule")
        if not isinstance(raw["suffix"], list):
            raise D03ContractError("v2 invalidation suffix 必须是数组")
        return cls(
            str(raw["change_kind"]), str(raw["subject_key"]),
            str(raw["earliest_stage"]), tuple(str(item) for item in raw["suffix"]),
        )


def build_v2_invalidation_rules(
        pack_earliest_stages: tuple[tuple[str, str], ...] = (),
        ) -> tuple[V2InvalidationRule, ...]:
    """按固定全局政策和 pack earliest stage 形成完整失效规则。"""
    rules: list[V2InvalidationRule] = []
    for change_kind in (
            "CODE_VERSION", "SCHEMA_VERSION", "OWNER_NAMESPACE", "SPLIT_POLICY",
            "RESOURCE_POLICY", "RUNNER_VERSION"):
        rules.append(V2InvalidationRule(
            change_kind, "GLOBAL", "FT00", V2_EXECUTION_STAGES))
    for change_kind in ("COURSE_VERSION", "PARSER_VERSION", "CARRIER_SCHEMA"):
        rules.append(V2InvalidationRule(
            change_kind, "GLOBAL", "W-02",
            V2_EXECUTION_STAGES[V2_EXECUTION_STAGES.index("W-02"):]))
    for stage in V2_EXECUTION_STAGES[1:]:
        rules.append(V2InvalidationRule(
            "EVALUATOR_VERSION", stage, stage,
            V2_EXECUTION_STAGES[V2_EXECUTION_STAGES.index(stage):]))
    for subject, earliest in pack_earliest_stages:
        if earliest not in V2_EXECUTION_STAGES[1:-1]:
            raise D03ContractError("pack earliest stage 必须是 W-02..W-09")
        suffix = V2_EXECUTION_STAGES[V2_EXECUTION_STAGES.index(earliest):]
        for change_kind in V2_PACK_INVALIDATION_KINDS:
            rules.append(V2InvalidationRule(change_kind, subject, earliest, suffix))
    result = tuple(sorted(rules))
    if len({(item.change_kind, item.subject_key) for item in result}) != len(result):
        raise D03ContractError("v2 invalidation rule 重复")
    return result


def invalidation_suffix(
        rules: tuple[V2InvalidationRule, ...],
        change_kind: str,
        subject_key: str,
        ) -> tuple[str, ...]:
    """解析一个变化的严格失效后缀，未知项 fail closed。"""
    matches = tuple(item for item in rules
                    if item.change_kind == change_kind and item.subject_key == subject_key)
    if len(matches) != 1:
        raise D03ContractError("v2 invalidation subject 未唯一注册")
    return matches[0].suffix


def validate_v2_initial_state(value: Any) -> dict[str, int]:
    """校验 successor 初始状态，禁止继承 v1 mastery 或训练计数。"""
    raw = exact_dict(value, set(V2_INITIAL_EXECUTION_STATE), where="v2 initial state")
    result = {key: flag(raw[key], where=f"v2 initial state.{key}")
              for key in V2_INITIAL_EXECUTION_STATE}
    if result != V2_INITIAL_EXECUTION_STATE:
        raise D03ContractError("v2 initial state 必须全部为零")
    return result


__all__ = [
    "OWNER_POLICIES", "V2_ADAPTER_VERSION", "V2_ALLOWED_WORKERS",
    "V2_AUTHORITY_FORMAT_VERSION", "V2_CARRIER_SCHEMA_VERSION",
    "V2_CHANGE_KINDS", "V2_CHECKPOINT_FORMAT_VERSION",
    "V2_CHECKPOINT_IDENTITY_FIELDS", "V2_CONTRACT_KIND",
    "V2_CONTRACT_PATH", "V2_CONTRACT_VERSION", "V2_COURSE_VERSION",
    "V2_DEFERRED_P3_MAX_RECORDS", "V2_DEFERRED_P3_MIN_RECORDS",
    "V2_EXECUTION_STAGES", "V2_GENERATOR_VERSION",
    "V2_INITIAL_EXECUTION_STATE", "V2_INVALIDATION_FORMAT_VERSION",
    "V2_INVALIDATION_VERSION", "V2_LOGICAL_SHARD_COUNT", "V2_MERGE_BARRIER_KEY",
    "V2_OWNER_KEYS", "V2_OWNER_MODES", "V2_P3_ACTIVATION_POLICY",
    "V2_PACK_EARLIEST_STAGES", "V2_PACK_INVALIDATION_KINDS",
    "V2_PARSER_VERSION", "V2_RELEASE_KEY", "V2_RELEASE_VERSION",
    "V2_RUN_ID_POLICY", "V2_SCALE_BUDGETS", "V2_RUN_IDENTITY_FIELDS",
    "V2_RUN_SCALE_KEYS",
    "V2_SCALE_KEYS", "V2_SCALE_RECORD_LIMITS", "V2_SCHEMA_VERSION", "V2_SPLITS",
    "V2_SPLIT_POLICY", "V1_PUBLIC_RECEIPT_PATH", "V1_PUBLIC_RECEIPT_SHA256",
    "V1_PUBLIC_RECEIPT_SIZE_BYTES", "V2_UNKNOWN_INVALIDATION_POLICY",
    "V2InvalidationRule", "V2OwnerPolicy",
    "V2RunIdentity", "V2ScaleBudget", "V2SplitPolicy", "build_v2_invalidation_rules",
    "invalidation_suffix", "validate_v2_initial_state",
]
