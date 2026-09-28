"""Integer graph for trained discourse-conjunction contracts.

The graph authorizes an operator to bind two scope-local proposition members
as one composition.  It contains no proposition answers, source text, or
surface sentence; those are supplied by the current input projection and
validated again before entering QueryState.
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


GRAPH_FORMAT = "PURE_INTEGER_INPUT_DISCOURSE_CONJUNCTION_GRAPH_V1"
GRAPH_KIND = 91597
RECORD_KIND = 91598
CONJUNCTION_RELATION_KIND = 91599
CONJUNCTION_TABLE = "input_discourse_conjunction"
CONJUNCTION_PART_TABLE = "input_discourse_conjunction_part"
GRAPH_META_TABLE = "input_discourse_conjunction_graph"
PART_OPERATION = 1
PART_RELATION = 2
PART_SOURCE = 3
PART_LICENSE = 4


def _strict_key(value: Iterable[int], *, where: str) -> tuple[int, ...]:
    result = tuple(value)
    if not result or any(type(item) is not int or item < 0 for item in result):
        raise ValueError(f"{where} must be a non-empty non-negative integer key")
    return result


def _key_hash(*keys: tuple[int, ...]) -> int:
    payload = bytearray()
    for key in keys:
        payload.extend(len(key).to_bytes(4, "big", signed=False))
        for value in key:
            payload.extend(value.to_bytes(8, "big", signed=True))
    value = int.from_bytes(hashlib.sha256(bytes(payload)).digest()[:8], "big")
    value &= (1 << 63) - 1
    return value or 1


@dataclass(frozen=True, slots=True)
class TrainedInputDiscourseConjunction:
    """A learned two-operand scope membership relation."""

    operation_key: tuple[int, ...]
    relation_key: tuple[int, ...] = (CONJUNCTION_RELATION_KIND, 1)

    def __post_init__(self) -> None:
        _strict_key(self.operation_key, where="operation_key")
        _strict_key(self.relation_key, where="relation_key")

    @property
    def operand_roles(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return ((CONJUNCTION_RELATION_KIND, 2),
                (CONJUNCTION_RELATION_KIND, 3))

    def stable_key(self) -> tuple[int, ...]:
        return (RECORD_KIND, len(self.operation_key), *self.operation_key,
                len(self.relation_key), *self.relation_key,
                *(value for role in self.operand_roles
                  for value in (len(role), *role)))


class InputDiscourseConjunctionGraphError(RuntimeError):
    pass


def _register_tables(backend) -> None:
    register_extension_table(
        backend, CONJUNCTION_TABLE,
        [("conjunction_id", TYPE_INT), ("operation_hash", TYPE_INT),
         ("relation_hash", TYPE_INT), ("source_hash", TYPE_INT),
         ("license_hash", TYPE_INT)],
        discipline=disc.DISC_APPEND_ONLY,
        indexes=[("conjunction_id",), ("operation_hash",)],
        recovery_key=("conjunction_id",),
    )
    register_extension_table(
        backend, CONJUNCTION_PART_TABLE,
        [("conjunction_id", TYPE_INT), ("field_code", TYPE_INT),
         ("part_ordinal", TYPE_INT), ("part_value", TYPE_INT)],
        discipline=disc.DISC_APPEND_ONLY,
        indexes=[("conjunction_id", "field_code", "part_ordinal")],
        recovery_key=("conjunction_id", "field_code", "part_ordinal"),
    )
    register_extension_table(
        backend, GRAPH_META_TABLE,
        [("graph_id", TYPE_INT), ("graph_kind", TYPE_INT),
         ("format_version", TYPE_INT), ("source_hash", TYPE_INT),
         ("license_hash", TYPE_INT), ("conjunction_count", TYPE_INT)],
        discipline=disc.DISC_APPEND_ONLY,
        indexes=[("graph_id",)], recovery_key=("graph_id",),
    )


def _parts(conjunction_id: int, field_code: int,
           key: tuple[int, ...]) -> list[dict[str, int]]:
    return [{"conjunction_id": conjunction_id, "field_code": field_code,
             "part_ordinal": ordinal, "part_value": value}
            for ordinal, value in enumerate(key)]


def materialize_input_discourse_conjunction_graph(
        conjunctions: tuple[TrainedInputDiscourseConjunction, ...],
        database: str | Path, *, source_identity: tuple[int, ...],
        license_identity: tuple[int, ...]) -> dict[str, object]:
    if type(conjunctions) is not tuple or any(
            not isinstance(item, TrainedInputDiscourseConjunction)
            for item in conjunctions):
        raise TypeError("conjunctions must be a tuple of integer records")
    conjunctions = tuple(sorted(set(conjunctions), key=lambda item: item.stable_key()))
    source_identity = _strict_key(source_identity, where="source_identity")
    license_identity = _strict_key(license_identity, where="license_identity")
    target = Path(database).resolve()
    if target.exists():
        raise ValueError(f"conjunction graph must use a new path: {target}")
    backend = SQLiteBackend(str(target), performance_mode="bulk")
    graph_id = _key_hash((GRAPH_KIND,), source_identity, license_identity)
    try:
        _register_tables(backend)
        backend.insert(GRAPH_META_TABLE, {
            "graph_id": graph_id, "graph_kind": GRAPH_KIND,
            "format_version": 1, "source_hash": _key_hash(source_identity),
            "license_hash": _key_hash(license_identity),
            "conjunction_count": len(conjunctions),
        })
        for item in conjunctions:
            conjunction_id = _key_hash(item.operation_key, item.relation_key,
                                       source_identity, license_identity)
            backend.insert(CONJUNCTION_TABLE, {
                "conjunction_id": conjunction_id,
                "operation_hash": _key_hash(item.operation_key),
                "relation_hash": _key_hash(item.relation_key),
                "source_hash": _key_hash(source_identity),
                "license_hash": _key_hash(license_identity),
            })
            rows = []
            for field, key in ((PART_OPERATION, item.operation_key),
                               (PART_RELATION, item.relation_key),
                               (PART_SOURCE, source_identity),
                               (PART_LICENSE, license_identity)):
                rows.extend(_parts(conjunction_id, field, key))
            backend.insert_many(CONJUNCTION_PART_TABLE, rows)
        backend.commit()
    finally:
        backend.close()
    return {"format": GRAPH_FORMAT, "graph_id": graph_id,
            "conjunction_count": len(conjunctions),
            "source_identity": list(source_identity),
            "license_identity": list(license_identity)}


def _read_part(backend, conjunction_id: int, field_code: int) -> tuple[int, ...]:
    rows = backend.select(CONJUNCTION_PART_TABLE, where={
        "conjunction_id": conjunction_id, "field_code": field_code,
    }, order_by="part_ordinal")
    if not rows or tuple(row["part_ordinal"] for row in rows) != tuple(range(len(rows))):
        raise InputDiscourseConjunctionGraphError("conjunction integer parts are incomplete")
    return tuple(int(row["part_value"]) for row in rows)


def read_input_discourse_conjunction_graph(
        database: str | Path,
        ) -> tuple[tuple[TrainedInputDiscourseConjunction, ...], dict[str, object]]:
    backend = SQLiteBackend(str(Path(database).resolve()), read_only=True)
    try:
        _register_tables(backend)
        metas = backend.select(GRAPH_META_TABLE, order_by="graph_id")
        if len(metas) != 1 or metas[0]["graph_kind"] != GRAPH_KIND:
            raise InputDiscourseConjunctionGraphError("conjunction graph metadata is not unique")
        meta = metas[0]
        rows = backend.select(CONJUNCTION_TABLE, order_by="conjunction_id")
        if len(rows) != meta["conjunction_count"]:
            raise InputDiscourseConjunctionGraphError("conjunction count mismatch")
        records = []
        for row in rows:
            operation = _read_part(backend, row["conjunction_id"], PART_OPERATION)
            relation = _read_part(backend, row["conjunction_id"], PART_RELATION)
            source = _read_part(backend, row["conjunction_id"], PART_SOURCE)
            license_key = _read_part(backend, row["conjunction_id"], PART_LICENSE)
            if (row["operation_hash"] != _key_hash(operation)
                    or row["relation_hash"] != _key_hash(relation)
                    or row["source_hash"] != _key_hash(source)
                    or row["license_hash"] != _key_hash(license_key)):
                raise InputDiscourseConjunctionGraphError("conjunction identity mismatch")
            records.append((TrainedInputDiscourseConjunction(operation, relation),
                            source, license_key))
        records.sort(key=lambda item: item[0].stable_key())
        if len({item[0].stable_key() for item in records}) != len(records):
            raise InputDiscourseConjunctionGraphError("duplicate conjunction identity")
        source = records[0][1] if records else ()
        license_key = records[0][2] if records else ()
        if (meta["source_hash"] != _key_hash(source)
                or meta["license_hash"] != _key_hash(license_key)):
            raise InputDiscourseConjunctionGraphError("conjunction metadata identity mismatch")
        return tuple(item[0] for item in records), {
            "format": GRAPH_FORMAT, "graph_id": int(meta["graph_id"]),
            "conjunction_count": int(meta["conjunction_count"]),
            "source_identity": source, "license_identity": license_key,
        }
    finally:
        backend.close()


__all__ = [
    "CONJUNCTION_RELATION_KIND", "GRAPH_FORMAT", "GRAPH_KIND",
    "InputDiscourseConjunctionGraphError", "TrainedInputDiscourseConjunction",
    "materialize_input_discourse_conjunction_graph",
    "read_input_discourse_conjunction_graph",
]
