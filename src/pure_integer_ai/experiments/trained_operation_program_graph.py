"""Append-only integer graph of licensed semantic operation programs."""
from __future__ import annotations

from pathlib import Path

from pure_integer_ai.cognition.shared.integer_operation_program import OperationProgramWitness
from pure_integer_ai.experiments.trained_relation_question_graph import _hash
from pure_integer_ai.storage import discipline as disc
from pure_integer_ai.storage.backend import TYPE_INT, SQLiteBackend, register_extension_table


GRAPH_KIND = 92220
TABLE = "integer_operation_program"
PART_TABLE = "integer_operation_program_part"
META_TABLE = "integer_operation_program_graph"


def _register(backend):
    register_extension_table(backend, TABLE,
        [("program_id", TYPE_INT), ("program_hash", TYPE_INT)],
        discipline=disc.DISC_APPEND_ONLY, indexes=[("program_id",)], recovery_key=("program_id",))
    register_extension_table(backend, PART_TABLE,
        [("program_id", TYPE_INT), ("part_ordinal", TYPE_INT), ("part_value", TYPE_INT)],
        discipline=disc.DISC_APPEND_ONLY, indexes=[("program_id", "part_ordinal")],
        recovery_key=("program_id", "part_ordinal"))
    register_extension_table(backend, META_TABLE,
        [("graph_kind", TYPE_INT), ("format_version", TYPE_INT), ("program_count", TYPE_INT)],
        discipline=disc.DISC_APPEND_ONLY, indexes=[("graph_kind",)], recovery_key=("graph_kind",))


def materialize_operation_program_graph(witnesses, database):
    if (type(witnesses) is not tuple
            or any(not isinstance(witness, OperationProgramWitness) for witness in witnesses)
            or len({witness.program.operation_key for witness in witnesses}) != len(witnesses)):
        raise ValueError("operation graph requires unique licensed train programs")
    target = Path(database).resolve()
    if target.exists():
        raise FileExistsError("operation graph requires a new path")
    backend = SQLiteBackend(str(target), performance_mode="bulk")
    try:
        _register(backend)
        backend.insert(META_TABLE, {"graph_kind": GRAPH_KIND, "format_version": 1,
                                    "program_count": len(witnesses)})
        for witness in sorted(witnesses, key=lambda item: item.stable_key()):
            key, identity = witness.stable_key(), _hash(witness.stable_key())
            if backend.select(TABLE, where={"program_id": identity}):
                raise ValueError("operation program hash collision")
            backend.insert(TABLE, {"program_id": identity, "program_hash": identity})
            backend.insert_many(PART_TABLE, [
                {"program_id": identity, "part_ordinal": index, "part_value": value}
                for index, value in enumerate(key)])
        backend.commit()
    finally:
        backend.close()


def read_operation_program_graph(database):
    backend = SQLiteBackend(str(Path(database).resolve(strict=True)), read_only=True)
    try:
        _register(backend)
        metas = backend.select(META_TABLE, order_by="graph_kind")
        rows = backend.select(TABLE, order_by="program_id")
        if (len(metas) != 1 or metas[0]["graph_kind"] != GRAPH_KIND
                or metas[0]["format_version"] != 1 or metas[0]["program_count"] != len(rows)
                or len({row["program_id"] for row in rows}) != len(rows)):
            raise ValueError("operation graph metadata is incomplete")
        parts = backend.select(PART_TABLE, order_by="program_id, part_ordinal")
        ids = {row["program_id"] for row in rows}
        if any(row["program_id"] not in ids for row in parts):
            raise ValueError("operation graph has orphan integer parts")
        result = []
        for row in rows:
            values = tuple(part for part in parts if part["program_id"] == row["program_id"])
            if not values or tuple(part["part_ordinal"] for part in values) != tuple(range(len(values))):
                raise ValueError("operation graph integer parts are incomplete")
            key = tuple(part["part_value"] for part in values)
            if _hash(key) != row["program_hash"] or row["program_id"] != row["program_hash"]:
                raise ValueError("operation graph full witness identity drifted")
            result.append(OperationProgramWitness.from_stable_key(key))
        if len({item.program.operation_key for item in result}) != len(result):
            raise ValueError("operation graph has competing instructions")
        return tuple(sorted(result, key=lambda item: item.stable_key()))
    finally:
        backend.close()
