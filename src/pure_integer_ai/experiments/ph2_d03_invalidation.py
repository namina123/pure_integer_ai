"""D-03 阶段依赖图和变化到最早失效后缀的纯解析。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    D03ContractError,
    FORMAT_VERSION,
    STAGE_KEYS,
    exact_dict,
    string_tuple,
    text,
)


INVALIDATION_ARTIFACT_KIND = "PH2_D03_STAGE_INVALIDATION_GRAPH"
CHANGE_KINDS = (
    "BACKEND_VERSION",
    "CODE_VERSION",
    "COURSE_VERSION",
    "DATA_VERSION",
    "EVALUATOR_VERSION",
    "LICENSE",
    "LOCATION_VERSION",
    "PACK_CONTENT",
    "PARSER_VERSION",
    "PRIMITIVE_VERSION",
    "SCHEMA_VERSION",
    "SEGMENT_VERSION",
    "SOURCE_SET",
)


@dataclass(frozen=True, order=True)
class StageDependencyEdge:
    """表示一个阶段对更早阶段的严格依赖。"""

    consumer_stage: str
    prerequisite_stage: str

    def __post_init__(self) -> None:
        if self.consumer_stage not in STAGE_KEYS or self.prerequisite_stage not in STAGE_KEYS:
            raise D03ContractError("stage dependency 引用未知阶段")
        if STAGE_KEYS.index(self.prerequisite_stage) >= STAGE_KEYS.index(self.consumer_stage):
            raise D03ContractError("stage dependency 顺序非法或形成环")

    def to_dict(self) -> dict[str, str]:
        """导出阶段依赖边。"""
        return {
            "consumer_stage": self.consumer_stage,
            "prerequisite_stage": self.prerequisite_stage,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "StageDependencyEdge":
        """从严格 object 恢复阶段依赖边。"""
        raw = exact_dict(value, {
            "consumer_stage", "prerequisite_stage",
        }, where="StageDependencyEdge")
        return cls(str(raw["consumer_stage"]), str(raw["prerequisite_stage"]))


@dataclass(frozen=True, order=True)
class StageInvalidationRule:
    """冻结一个变化对象的最早受影响阶段和全部后缀。"""

    change_kind: str
    subject_key: str
    earliest_stage: str
    invalidated_stage_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.change_kind not in CHANGE_KINDS:
            raise D03ContractError("未知 invalidation change kind")
        text(self.subject_key, where="invalidation subject")
        if self.earliest_stage not in STAGE_KEYS:
            raise D03ContractError("invalidation earliest stage 非法")
        expected = STAGE_KEYS[STAGE_KEYS.index(self.earliest_stage):]
        if self.invalidated_stage_keys != expected:
            raise D03ContractError("invalidation 必须包含最早阶段及完整后缀")

    def to_dict(self) -> dict[str, Any]:
        """导出失效规则。"""
        return {
            "change_kind": self.change_kind,
            "earliest_stage": self.earliest_stage,
            "invalidated_stage_keys": list(self.invalidated_stage_keys),
            "subject_key": self.subject_key,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "StageInvalidationRule":
        """从严格 object 恢复失效规则。"""
        raw = exact_dict(value, {
            "change_kind", "earliest_stage", "invalidated_stage_keys",
            "subject_key",
        }, where="StageInvalidationRule")
        return cls(
            str(raw["change_kind"]), str(raw["subject_key"]),
            str(raw["earliest_stage"]),
            string_tuple(raw["invalidated_stage_keys"], where="invalidated stages"),
        )


@dataclass(frozen=True)
class InvalidationResult:
    """返回 reader 已解析的最早阶段和完整失效后缀。"""

    earliest_stage: str
    invalidated_stage_keys: tuple[str, ...]


@dataclass(frozen=True)
class StageInvalidationGraph:
    """合取九阶段链和按变化对象精确寻址的失效规则。"""

    format_version: int
    artifact_kind: str
    artifact_version: str
    release_key: str
    stage_keys: tuple[str, ...]
    stage_edges: tuple[StageDependencyEdge, ...]
    rules: tuple[StageInvalidationRule, ...]

    def __post_init__(self) -> None:
        if self.format_version != FORMAT_VERSION:
            raise D03ContractError("invalidation format_version 非法")
        if self.artifact_kind != INVALIDATION_ARTIFACT_KIND:
            raise D03ContractError("invalidation artifact_kind 非法")
        text(self.artifact_version, where="invalidation artifact version")
        text(self.release_key, where="invalidation release key")
        if self.stage_keys != STAGE_KEYS:
            raise D03ContractError("invalidation stage 顺序漂移")
        expected_edges = tuple(
            StageDependencyEdge(STAGE_KEYS[index], STAGE_KEYS[index - 1])
            for index in range(1, len(STAGE_KEYS))
        )
        if self.stage_edges != expected_edges:
            raise D03ContractError("invalidation stage chain 顺序非法或形成环")
        if (not isinstance(self.rules, tuple) or not self.rules
                or any(not isinstance(item, StageInvalidationRule)
                       for item in self.rules)):
            raise D03ContractError("invalidation rules 不能为空")
        rules = tuple(sorted(self.rules))
        if len({(item.change_kind, item.subject_key) for item in rules}) != len(rules):
            raise D03ContractError("invalidation rule key 重复")
        object.__setattr__(self, "rules", rules)

    def invalidate(self, change_kind: str, subject_key: str) -> InvalidationResult:
        """按变化种类和对象返回唯一最早阶段及完整后缀。"""
        for rule in self.rules:
            if rule.change_kind == change_kind and rule.subject_key == subject_key:
                return InvalidationResult(
                    rule.earliest_stage, rule.invalidated_stage_keys)
        raise D03ContractError("未知 change/subject，不得猜测失效范围")

    def to_dict(self) -> dict[str, Any]:
        """导出规范失效图。"""
        return {
            "artifact_kind": self.artifact_kind,
            "artifact_version": self.artifact_version,
            "format_version": self.format_version,
            "release_key": self.release_key,
            "rules": [item.to_dict() for item in self.rules],
            "stage_edges": [item.to_dict() for item in self.stage_edges],
            "stage_keys": list(self.stage_keys),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "StageInvalidationGraph":
        """从严格 object 恢复阶段失效图。"""
        raw = exact_dict(value, {
            "artifact_kind", "artifact_version", "format_version",
            "release_key", "rules", "stage_edges", "stage_keys",
        }, where="StageInvalidationGraph")
        if not isinstance(raw["stage_edges"], list) or not isinstance(raw["rules"], list):
            raise D03ContractError("invalidation edges/rules 必须是数组")
        return cls(
            raw["format_version"], str(raw["artifact_kind"]),
            str(raw["artifact_version"]), str(raw["release_key"]),
            string_tuple(raw["stage_keys"], where="invalidation stage keys"),
            tuple(StageDependencyEdge.from_dict(item) for item in raw["stage_edges"]),
            tuple(StageInvalidationRule.from_dict(item) for item in raw["rules"]),
        )


__all__ = [
    "CHANGE_KINDS",
    "INVALIDATION_ARTIFACT_KIND",
    "InvalidationResult",
    "StageDependencyEdge",
    "StageInvalidationGraph",
    "StageInvalidationRule",
]
