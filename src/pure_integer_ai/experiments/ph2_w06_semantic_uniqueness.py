"""W-06 训练前语义命题唯一化审计。

审计键只由 payload 中声明的整数语义字段组成。相同语义键允许出现多条
Observation；它们必须汇聚到一个 Proposition，来源和 Evidence 仍逐条保留。
同键结构不一致则在训练启动前 fail closed。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    ObservationRecord,
)
from pure_integer_ai.experiments.ph2_w06_payload import W06TrainingPayload


class W06SemanticUniquenessError(RuntimeError):
    """语义命题键缺失、重复结构冲突或 payload 不是整数合同。"""


def _ints(value: Any, *, where: str) -> tuple[int, ...]:
    if (not isinstance(value, list) or not value
            or any(type(item) is not int for item in value)):
        raise W06SemanticUniquenessError(
            f"{where} 必须是非空严格整数列表")
    return tuple(value)


def _payload(observation: ObservationRecord) -> dict[str, Any]:
    if not isinstance(observation, ObservationRecord):
        raise TypeError("语义唯一化审计需要 ObservationRecord")
    value = observation.typed_payload.to_value()
    if (not isinstance(value, dict)
            or value.get("query_kind") != "typed_relation_candidate"):
        raise W06SemanticUniquenessError("W-06 semantic payload 类型非法")
    return value


def _shape(value: dict[str, Any]) -> tuple[int, ...]:
    """抽取不含 source/span/surface 的结构整数签名。"""
    semantic = _ints(value.get("semantic_key"), where="semantic_key")
    directionality = value.get("directionality")
    if type(directionality) is not int:
        raise W06SemanticUniquenessError("directionality 必须是严格整数")
    schema = value.get("relation_schema")
    if not isinstance(schema, dict):
        raise W06SemanticUniquenessError("relation_schema 缺失")
    schema_key = _ints(schema.get("schema_key"), where="schema_key")
    relation_key = _ints(schema.get("relation_key"), where="relation_key")
    slots = schema.get("slots")
    if not isinstance(slots, list):
        raise W06SemanticUniquenessError("schema.slots 缺失")
    slot_values: list[int] = []
    for slot in slots:
        if not isinstance(slot, dict):
            raise W06SemanticUniquenessError("schema slot 非 object")
        role = _ints(slot.get("role_key"), where="slot.role_key")
        allowed = slot.get("allowed_object_kinds")
        if (not isinstance(allowed, list)
                or any(type(item) is not int for item in allowed)):
            raise W06SemanticUniquenessError("slot allowed kinds 非整数列表")
        slot_values.extend((len(role), *role, len(allowed), *sorted(allowed),
                            slot.get("min_count", -1),
                            slot.get("max_count", -1) or -1))
    if any(type(item) is not int for item in slot_values):
        raise W06SemanticUniquenessError("schema slot cardinality 非整数")
    return (
        len(semantic), *semantic,
        directionality,
        len(relation_key), *relation_key,
        len(schema_key), *schema_key,
        len(slot_values), *slot_values,
    )


@dataclass(frozen=True)
class W06SemanticUniquenessReport:
    """训练启动前语义命题分组和冲突摘要。"""

    observation_count: int
    semantic_group_count: int
    duplicate_observation_count: int
    conflict_group_count: int
    groups: tuple[tuple[tuple[int, ...], int, tuple[int, ...]], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "observation_count": self.observation_count,
            "semantic_group_count": self.semantic_group_count,
            "duplicate_observation_count": self.duplicate_observation_count,
            "conflict_group_count": self.conflict_group_count,
            "groups": [
                {
                    "semantic_key": list(key),
                    "observation_count": count,
                    "shape_key": list(shape),
                }
                for key, count, shape in self.groups
            ],
        }


def audit_w06_semantic_uniqueness(
        payload: W06TrainingPayload,
        ) -> W06SemanticUniquenessReport:
    """按声明 semantic_key 分组，并拒绝同键结构竞争。"""
    if not isinstance(payload, W06TrainingPayload):
        raise TypeError("语义唯一化审计需要 W06TrainingPayload")
    grouped: dict[tuple[int, ...], list[tuple[int, ...]]] = {}
    for observation in payload.observations:
        # The W-06 adapter intentionally consumes only its typed relation
        # slice.  Earlier public course observations retain their historical
        # payload contract and are not semantic-identity inputs here.
        if (observation.w_stage != "W-06"
                or observation.payload_kind != "TypedRelationQuery"):
            continue
        value = _payload(observation)
        key = _ints(value.get("semantic_key"), where="semantic_key")
        grouped.setdefault(key, []).append(_shape(value))
    rows = []
    conflicts = 0
    duplicate_count = 0
    for key in sorted(grouped):
        shapes = grouped[key]
        unique_shapes = tuple(sorted(set(shapes)))
        if len(shapes) > 1:
            duplicate_count += len(shapes) - 1
        if len(unique_shapes) > 1:
            conflicts += 1
        rows.append((key, len(shapes), unique_shapes[0]))
    report = W06SemanticUniquenessReport(
        len(payload.observations),
        len(rows),
        duplicate_count,
        conflicts,
        tuple(rows),
    )
    if conflicts:
        raise W06SemanticUniquenessError(
            "W-06 semantic proposition key 结构冲突，训练启动被阻断")
    return report


__all__ = [
    "W06SemanticUniquenessError",
    "W06SemanticUniquenessReport",
    "audit_w06_semantic_uniqueness",
]
