"""Read-only integer graph for non-conjunction input operation contracts.

The existing operator graph stores only the exact surface marker.  This sidecar
stores the trained operation's query contract separately so a marker can be
consumed by the production bridge without embedding a language-specific rule.
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


GRAPH_FORMAT = "PURE_INTEGER_INPUT_DISCOURSE_OPERATION_CONTRACT_GRAPH_V1"
GRAPH_KIND = 91670
RECORD_KIND = 91671
OPERATION_CONTRACT_TABLE = "input_discourse_operation_contract"
OPERATION_CONTRACT_PART_TABLE = "input_discourse_operation_contract_part"
GRAPH_META_TABLE = "input_discourse_operation_contract_graph"
PART_OPERATION = 1
PART_MODE = 2
PART_SOURCE = 3
PART_LICENSE = 4

MODE_FACT_QUERY_LEFT = (91672, 1)
MODE_FACT_QUERY_RIGHT = (91672, 2)
# 一般 operation 由显式整数图输入绑定；它不改变 relation scope，
# 但必须在同一 QueryState 留下 contract binding/evidence。
MODE_OPERATION_GENERAL = (91672, 3)


def _strict_key(value: Iterable[int], *, where: str) -> tuple[int, ...]:
    result = tuple(value)
    if (not result
            or any(type(item) is not int or item < 0 for item in result)):
        raise ValueError(f"{where} must be a non-empty non-negative integer key")
    return result


def _key_hash(*keys: tuple[int, ...]) -> int:
    payload = bytearray()
    for key in keys:
        payload.extend(len(key).to_bytes(4, "big", signed=False))
        for value in key:
            payload.extend(value.to_bytes(8, "big", signed=True))
    result = int.from_bytes(hashlib.sha256(bytes(payload)).digest()[:8], "big")
    result &= (1 << 63) - 1
    return result or 1


@dataclass(frozen=True, slots=True)
class TrainedInputDiscourseOperationContract:
    """A non-conjunction operation and its integer operand contract."""

    operation_key: tuple[int, ...]
    mode_key: tuple[int, ...] = MODE_FACT_QUERY_LEFT

    def __post_init__(self) -> None:
        _strict_key(self.operation_key, where="operation_key")
        _strict_key(self.mode_key, where="mode_key")

    def stable_key(self) -> tuple[int, ...]:
        return (RECORD_KIND, *_pack(self.operation_key), *_pack(self.mode_key))


def _pack(key: tuple[int, ...]) -> tuple[int, ...]:
    return len(key), *key


def _register_tables(backend) -> None:
    register_extension_table(
        backend, OPERATION_CONTRACT_TABLE,
        [("contract_id", TYPE_INT), ("operation_hash", TYPE_INT),
         ("mode_hash", TYPE_INT), ("source_hash", TYPE_INT),
         ("license_hash", TYPE_INT)],
        discipline=disc.DISC_APPEND_ONLY,
        indexes=[("contract_id",), ("operation_hash",)],
        recovery_key=("contract_id",),
    )
    register_extension_table(
        backend, OPERATION_CONTRACT_PART_TABLE,
        [("contract_id", TYPE_INT), ("field_code", TYPE_INT),
         ("part_ordinal", TYPE_INT), ("part_value", TYPE_INT)],
        discipline=disc.DISC_APPEND_ONLY,
        indexes=[("contract_id", "field_code", "part_ordinal")],
        recovery_key=("contract_id", "field_code", "part_ordinal"),
    )
    register_extension_table(
        backend, GRAPH_META_TABLE,
        [("graph_id", TYPE_INT), ("graph_kind", TYPE_INT),
         ("format_version", TYPE_INT), ("source_hash", TYPE_INT),
         ("license_hash", TYPE_INT), ("contract_count", TYPE_INT)],
        discipline=disc.DISC_APPEND_ONLY,
        indexes=[("graph_id",)], recovery_key=("graph_id",),
    )


def _part_rows(contract_id: int, field_code: int, key: tuple[int, ...]):
    return [{"contract_id": contract_id, "field_code": field_code,
             "part_ordinal": ordinal, "part_value": value}
            for ordinal, value in enumerate(key)]


def materialize_input_discourse_operation_graph(
        contracts: tuple[TrainedInputDiscourseOperationContract, ...],
        database: str | Path, *,
        source_identity: tuple[int, ...],
        license_identity: tuple[int, ...],
        operators: tuple[TrainedInputDiscourseOperator, ...] = (),
        ) -> dict[str, int | str]:
    """Materialize new integer contracts without opening canonical for writes."""
    if (type(contracts) is not tuple
            or any(not isinstance(item, TrainedInputDiscourseOperationContract)
                   for item in contracts)):
        raise TypeError("contracts must be a tuple of integer contracts")
    contracts = tuple(sorted(set(contracts), key=lambda item: item.stable_key()))
    if not contracts:
        raise ValueError("operation contract graph requires at least one contract")
    if type(operators) is not tuple or any(
            not isinstance(item, TrainedInputDiscourseOperator) for item in operators):
        raise TypeError("operators must be a tuple of integer operators")
    operator_keys = {item.operation_key for item in operators}
    if not operator_keys:
        raise ValueError("operation contract graph requires matching operators")
    if any(item.operation_key not in operator_keys for item in contracts):
        raise ValueError("operation contract references an unknown operator")
    source_identity = _strict_key(source_identity, where="source_identity")
    license_identity = _strict_key(license_identity, where="license_identity")
    target = Path(database).resolve()
    if target.exists():
        raise ValueError(f"operation contract graph must use a new path: {target}")
    backend = SQLiteBackend(str(target), performance_mode="bulk")
    graph_id = _key_hash((GRAPH_KIND,), source_identity, license_identity)
    try:
        _register_tables(backend)
        backend.insert(GRAPH_META_TABLE, {
            "graph_id": graph_id, "graph_kind": GRAPH_KIND,
            "format_version": 1, "source_hash": _key_hash(source_identity),
            "license_hash": _key_hash(license_identity),
            "contract_count": len(contracts),
        })
        for contract in contracts:
            contract_id = _key_hash(
                contract.operation_key, contract.mode_key,
                source_identity, license_identity)
            backend.insert(OPERATION_CONTRACT_TABLE, {
                "contract_id": contract_id,
                "operation_hash": _key_hash(contract.operation_key),
                "mode_hash": _key_hash(contract.mode_key),
                "source_hash": _key_hash(source_identity),
                "license_hash": _key_hash(license_identity),
            })
            parts = []
            for field, key in (
                    (PART_OPERATION, contract.operation_key),
                    (PART_MODE, contract.mode_key),
                    (PART_SOURCE, source_identity),
                    (PART_LICENSE, license_identity)):
                parts.extend(_part_rows(contract_id, field, key))
            backend.insert_many(OPERATION_CONTRACT_PART_TABLE, parts)
        backend.commit()
    finally:
        backend.close()
    return {
        "format": GRAPH_FORMAT, "graph_id": graph_id,
        "contract_count": len(contracts),
        "source_identity": list(source_identity),
        "license_identity": list(license_identity),
    }


def _read_part(backend, contract_id: int, field_code: int) -> tuple[int, ...]:
    rows = backend.select(OPERATION_CONTRACT_PART_TABLE, where={
        "contract_id": contract_id, "field_code": field_code,
    }, order_by="part_ordinal")
    if not rows or tuple(row["part_ordinal"] for row in rows) != tuple(range(len(rows))):
        raise RuntimeError("operation contract integer parts are incomplete")
    return tuple(int(row["part_value"]) for row in rows)


def read_input_discourse_operation_graph(
        database: str | Path,
        ) -> tuple[tuple[TrainedInputDiscourseOperationContract, ...], dict[str, object]]:
    """Read and validate contracts from a model graph in read-only mode."""
    backend = SQLiteBackend(str(Path(database).resolve()), read_only=True)
    try:
        _register_tables(backend)
        metas = backend.select(GRAPH_META_TABLE, order_by="graph_id")
        if len(metas) != 1 or metas[0]["graph_kind"] != GRAPH_KIND:
            raise RuntimeError("operation contract graph metadata is not unique")
        meta = metas[0]
        rows = backend.select(OPERATION_CONTRACT_TABLE, order_by="contract_id")
        if len(rows) != meta["contract_count"]:
            raise RuntimeError("operation contract count mismatch")
        records = []
        for row in rows:
            operation = _read_part(backend, row["contract_id"], PART_OPERATION)
            mode = _read_part(backend, row["contract_id"], PART_MODE)
            source = _read_part(backend, row["contract_id"], PART_SOURCE)
            license_key = _read_part(backend, row["contract_id"], PART_LICENSE)
            if (row["operation_hash"] != _key_hash(operation)
                    or row["mode_hash"] != _key_hash(mode)
                    or row["source_hash"] != _key_hash(source)
                    or row["license_hash"] != _key_hash(license_key)):
                raise RuntimeError("operation contract identity mismatch")
            records.append((TrainedInputDiscourseOperationContract(operation, mode),
                            source, license_key))
        records.sort(key=lambda item: item[0].stable_key())
        if len({item[0].stable_key() for item in records}) != len(records):
            raise RuntimeError("duplicate operation contract identity")
        if any(item[1] != records[0][1] or item[2] != records[0][2]
               for item in records[1:]):
            raise RuntimeError("operation contract provenance is not uniform")
        source = records[0][1] if records else ()
        license_key = records[0][2] if records else ()
        if (meta["source_hash"] != _key_hash(source)
                or meta["license_hash"] != _key_hash(license_key)):
            raise RuntimeError("operation contract metadata identity mismatch")
        return tuple(item[0] for item in records), {
            "format": GRAPH_FORMAT, "graph_id": int(meta["graph_id"]),
            "contract_count": int(meta["contract_count"]),
            "source_identity": source, "license_identity": license_key,
        }
    finally:
        backend.close()


__all__ = [
    "GRAPH_FORMAT", "GRAPH_KIND", "MODE_FACT_QUERY_LEFT",
    "MODE_FACT_QUERY_RIGHT", "MODE_OPERATION_GENERAL",
    "RECORD_KIND", "TrainedInputDiscourseOperationContract",
    "materialize_input_discourse_operation_graph",
    "read_input_discourse_operation_graph",
]
