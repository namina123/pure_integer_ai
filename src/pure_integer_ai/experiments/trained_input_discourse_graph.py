"""训练后输入话语算子的纯整数图物化与冷恢复。

课程文件只存在于训练交换边界；运行时只读取本模块写入的整数记录，
不读取课程表面、旧 trace 或答案。operator 的表层值、操作身份、来源
身份和许可身份均以有序整数 part 保存，删除索引后仍可恢复完整记录。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from pure_integer_ai.cognition.understanding.query_input_structure import (
    TrainedInputDiscourseOperator,
)
from pure_integer_ai.storage import discipline as disc
from pure_integer_ai.storage.backend import TYPE_INT, SQLiteBackend, register_extension_table


GRAPH_FORMAT = "PURE_INTEGER_INPUT_DISCOURSE_OPERATOR_GRAPH_V1"
GRAPH_KIND = 91593
RECORD_KIND = 91594
OPERATOR_TABLE = "input_discourse_operator"
OPERATOR_PART_TABLE = "input_discourse_operator_part"
GRAPH_META_TABLE = "input_discourse_operator_graph"
PART_OPERATION = 1
PART_SURFACE = 2
PART_SOURCE = 3
PART_LICENSE = 4


class InputDiscourseGraphError(RuntimeError):
    """输入话语图缺失、碰撞或来源身份不完整。"""


def _strict_key(value: Iterable[int], *, where: str) -> tuple[int, ...]:
    result = tuple(value)
    if (not result or any(type(item) is not int or item < 0 for item in result)):
        raise ValueError(f"{where} 必须是非空非负整数键")
    return result


def _key_hash(*keys: tuple[int, ...]) -> int:
    payload = bytearray()
    for key in keys:
        payload.extend(len(key).to_bytes(4, "big", signed=False))
        for value in key:
            payload.extend(value.to_bytes(8, "big", signed=True))
    digest = hashlib.sha256(bytes(payload)).digest()
    value = int.from_bytes(digest[:8], "big", signed=False) & ((1 << 63) - 1)
    return value if value > 0 else 1


@dataclass(frozen=True, slots=True)
class InputDiscourseOperatorGraphRecord:
    """一条可从模型图恢复的 operator 记录。"""

    operator: TrainedInputDiscourseOperator
    source_identity: tuple[int, ...]
    license_identity: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.operator, TrainedInputDiscourseOperator):
            raise TypeError("operator 必须是训练后整数记录")
        _strict_key(self.source_identity, where="source_identity")
        _strict_key(self.license_identity, where="license_identity")

    def stable_key(self) -> tuple[int, ...]:
        return (
            RECORD_KIND,
            *_key_parts(self.operator.operation_key),
            *_key_parts(self.operator.surface_values),
            *_key_parts(self.source_identity),
            *_key_parts(self.license_identity),
        )


def _key_parts(key: tuple[int, ...]) -> tuple[int, ...]:
    return (len(key), *key)


def _register_tables(backend) -> None:
    register_extension_table(
        backend, OPERATOR_TABLE,
        [("operator_id", TYPE_INT), ("operation_hash", TYPE_INT),
         ("surface_hash", TYPE_INT), ("source_hash", TYPE_INT),
         ("license_hash", TYPE_INT), ("surface_count", TYPE_INT)],
        discipline=disc.DISC_APPEND_ONLY,
        indexes=[("operator_id",), ("operation_hash",)],
        recovery_key=("operator_id",),
    )
    register_extension_table(
        backend, OPERATOR_PART_TABLE,
        [("operator_id", TYPE_INT), ("field_code", TYPE_INT),
         ("part_ordinal", TYPE_INT), ("part_value", TYPE_INT)],
        discipline=disc.DISC_APPEND_ONLY,
        indexes=[("operator_id", "field_code", "part_ordinal")],
        recovery_key=("operator_id", "field_code", "part_ordinal"),
    )
    register_extension_table(
        backend, GRAPH_META_TABLE,
        [("graph_id", TYPE_INT), ("graph_kind", TYPE_INT),
         ("format_version", TYPE_INT), ("source_hash", TYPE_INT),
         ("license_hash", TYPE_INT), ("operator_count", TYPE_INT)],
        discipline=disc.DISC_APPEND_ONLY,
        indexes=[("graph_id",)],
        recovery_key=("graph_id",),
    )


def _part_rows(operator_id: int, field_code: int, key: tuple[int, ...]):
    return [{"operator_id": operator_id, "field_code": field_code,
             "part_ordinal": ordinal, "part_value": value}
            for ordinal, value in enumerate(key)]


def materialize_input_discourse_operator_graph(
        operators: tuple[TrainedInputDiscourseOperator, ...],
        database: str | Path,
        *,
        source_identity: tuple[int, ...],
        license_identity: tuple[int, ...],
        ) -> dict[str, int | str]:
    """把训练课程消费成独立模型图；不打开或复制 canonical。"""
    if (type(operators) is not tuple
            or any(not isinstance(item, TrainedInputDiscourseOperator)
                   for item in operators)):
        raise TypeError("operators 必须是训练后整数 tuple")
    operators = tuple(sorted(set(operators), key=lambda item: item.stable_key()))
    source_identity = _strict_key(source_identity, where="source_identity")
    license_identity = _strict_key(license_identity, where="license_identity")
    target = Path(database).resolve()
    if target.exists():
        raise ValueError(f"operator graph 必须写入新路径: {target}")
    backend = SQLiteBackend(str(target), performance_mode="bulk")
    graph_id = _key_hash((GRAPH_KIND,), source_identity, license_identity)
    try:
        _register_tables(backend)
        backend.insert(GRAPH_META_TABLE, {
            "graph_id": graph_id, "graph_kind": GRAPH_KIND,
            "format_version": 1, "source_hash": _key_hash(source_identity),
            "license_hash": _key_hash(license_identity),
            "operator_count": len(operators),
        })
        for operator in operators:
            operator_id = _key_hash(
                operator.operation_key, operator.surface_values,
                source_identity, license_identity)
            existing = backend.select(OPERATOR_TABLE,
                                      where={"operator_id": operator_id})
            row = {
                "operator_id": operator_id,
                "operation_hash": _key_hash(operator.operation_key),
                "surface_hash": _key_hash(operator.surface_values),
                "source_hash": _key_hash(source_identity),
                "license_hash": _key_hash(license_identity),
                "surface_count": len(operator.surface_values),
            }
            if existing and existing[0] != row:
                raise InputDiscourseGraphError("operator identity hash collision")
            if existing:
                continue
            backend.insert(OPERATOR_TABLE, row)
            parts = []
            for field, key in (
                    (PART_OPERATION, operator.operation_key),
                    (PART_SURFACE, operator.surface_values),
                    (PART_SOURCE, source_identity),
                    (PART_LICENSE, license_identity)):
                parts.extend(_part_rows(operator_id, field, key))
            backend.insert_many(OPERATOR_PART_TABLE, parts)
        backend.commit()
    finally:
        backend.close()
    return {
        "format": GRAPH_FORMAT,
        "graph_id": graph_id,
        "operator_count": len(operators),
        "source_identity": list(source_identity),
        "license_identity": list(license_identity),
    }


def _read_parts(backend, operator_id: int, field_code: int) -> tuple[int, ...]:
    rows = backend.select(OPERATOR_PART_TABLE, where={
        "operator_id": operator_id, "field_code": field_code,
    }, order_by="part_ordinal")
    values = tuple(int(row["part_value"]) for row in rows)
    if not values or tuple(row["part_ordinal"] for row in rows) != tuple(range(len(rows))):
        raise InputDiscourseGraphError("operator part 不连续或为空")
    return values


def read_input_discourse_operator_graph(
        database: str | Path,
        ) -> tuple[tuple[TrainedInputDiscourseOperator, ...], dict[str, tuple[int, ...] | int | str]]:
    """只读恢复完整 operator 与来源/许可身份。"""
    backend = SQLiteBackend(str(Path(database).resolve()), read_only=True)
    try:
        _register_tables(backend)
        metas = backend.select(GRAPH_META_TABLE, order_by="graph_id")
        if len(metas) != 1 or metas[0]["graph_kind"] != GRAPH_KIND:
            raise InputDiscourseGraphError("operator graph metadata 不唯一")
        meta = metas[0]
        rows = backend.select(OPERATOR_TABLE, order_by="operator_id")
        if len(rows) != meta["operator_count"]:
            raise InputDiscourseGraphError("operator graph count mismatch")
        records = []
        for row in rows:
            operation = _read_parts(backend, row["operator_id"], PART_OPERATION)
            surface = _read_parts(backend, row["operator_id"], PART_SURFACE)
            source = _read_parts(backend, row["operator_id"], PART_SOURCE)
            license_key = _read_parts(backend, row["operator_id"], PART_LICENSE)
            if (row["source_hash"] != _key_hash(source)
                    or row["license_hash"] != _key_hash(license_key)
                    or row["surface_count"] != len(surface)):
                raise InputDiscourseGraphError("operator graph identity mismatch")
            records.append(InputDiscourseOperatorGraphRecord(
                TrainedInputDiscourseOperator(operation, surface),
                source, license_key))
        records.sort(key=lambda item: item.stable_key())
        if len({item.stable_key() for item in records}) != len(records):
            raise InputDiscourseGraphError("operator graph identity duplicate")
        source = records[0].source_identity if records else ()
        license_key = records[0].license_identity if records else ()
        if (meta["source_hash"] != _key_hash(source)
                or meta["license_hash"] != _key_hash(license_key)):
            raise InputDiscourseGraphError("operator graph metadata identity mismatch")
        return (tuple(item.operator for item in records), {
            "format": GRAPH_FORMAT,
            "graph_id": int(meta["graph_id"]),
            "operator_count": int(meta["operator_count"]),
            "source_identity": source,
            "license_identity": license_key,
        })
    finally:
        backend.close()


__all__ = [
    "GRAPH_FORMAT", "GRAPH_KIND", "InputDiscourseGraphError",
    "InputDiscourseOperatorGraphRecord",
    "materialize_input_discourse_operator_graph",
    "read_input_discourse_operator_graph",
]
