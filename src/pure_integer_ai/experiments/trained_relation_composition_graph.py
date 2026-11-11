"""Cold, integer-only storage of explicit source-qualified join instructions."""
from __future__ import annotations

from pathlib import Path

from pure_integer_ai.cognition.shared.relation_composition import RelationCompositionWitness
from pure_integer_ai.experiments.trained_relation_question_graph import _hash
from pure_integer_ai.storage import discipline as disc
from pure_integer_ai.storage.backend import TYPE_INT, SQLiteBackend, register_extension_table


GRAPH_KIND = 92108
TABLE = "relation_composition_witness"
PART_TABLE = "relation_composition_witness_part"
META_TABLE = "relation_composition_graph"


def _register(backend):
    register_extension_table(backend, TABLE,
        [("witness_id", TYPE_INT), ("witness_hash", TYPE_INT)],
        discipline=disc.DISC_APPEND_ONLY, indexes=[("witness_id",)], recovery_key=("witness_id",))
    register_extension_table(backend, PART_TABLE,
        [("witness_id", TYPE_INT), ("part_ordinal", TYPE_INT), ("part_value", TYPE_INT)],
        discipline=disc.DISC_APPEND_ONLY, indexes=[("witness_id", "part_ordinal")],
        recovery_key=("witness_id", "part_ordinal"))
    register_extension_table(backend, META_TABLE,
        [("graph_kind", TYPE_INT), ("format_version", TYPE_INT), ("witness_count", TYPE_INT)],
        discipline=disc.DISC_APPEND_ONLY, indexes=[("graph_kind",)], recovery_key=("graph_kind",))


def materialize_relation_composition_graph(
        witnesses: tuple[RelationCompositionWitness, ...], database: str | Path) -> None:
    if (type(witnesses) is not tuple or not witnesses
            or any(not isinstance(item, RelationCompositionWitness) for item in witnesses)
            or len({item.rule.rule for item in witnesses}) != len(witnesses)):
        raise ValueError("composition graph needs unique licensed train instructions")
    target = Path(database).resolve()
    if target.exists():
        raise FileExistsError("composition graph requires a new path")
    backend = SQLiteBackend(str(target), performance_mode="bulk")
    try:
        _register(backend)
        backend.insert(META_TABLE, {"graph_kind": GRAPH_KIND, "format_version": 2,
                                    "witness_count": len(witnesses)})
        for witness in sorted(witnesses, key=lambda item: item.stable_key()):
            key = witness.stable_key()
            identity = _hash(key)
            if backend.select(TABLE, where={"witness_id": identity}):
                raise ValueError("composition witness hash collision")
            backend.insert(TABLE, {"witness_id": identity, "witness_hash": identity})
            backend.insert_many(PART_TABLE, [
                {"witness_id": identity, "part_ordinal": index, "part_value": value}
                for index, value in enumerate(key)])
        backend.commit()
    finally:
        backend.close()


def read_relation_composition_graph(database: str | Path) -> tuple[RelationCompositionWitness, ...]:
    backend = SQLiteBackend(str(Path(database).resolve(strict=True)), read_only=True)
    try:
        _register(backend)
        metas = backend.select(META_TABLE, order_by="graph_kind")
        rows = backend.select(TABLE, order_by="witness_id")
        if (len(metas) != 1 or metas[0]["graph_kind"] != GRAPH_KIND
                or metas[0]["format_version"] != 2
                or metas[0]["witness_count"] != len(rows)
                or not rows or len({row["witness_id"] for row in rows}) != len(rows)):
            raise RuntimeError("composition graph metadata is incomplete")
        all_parts = backend.select(PART_TABLE, order_by="witness_id, part_ordinal")
        if any(part["witness_id"] not in {row["witness_id"] for row in rows} for part in all_parts):
            raise RuntimeError("composition graph has orphan integer parts")
        result = []
        for row in rows:
            parts = tuple(part for part in all_parts if part["witness_id"] == row["witness_id"])
            if not parts or tuple(part["part_ordinal"] for part in parts) != tuple(range(len(parts))):
                raise RuntimeError("composition witness integer parts are incomplete")
            key = tuple(part["part_value"] for part in parts)
            if _hash(key) != row["witness_hash"] or row["witness_id"] != row["witness_hash"]:
                raise RuntimeError("composition witness identity mismatch")
            result.append(RelationCompositionWitness.from_stable_key(key))
        if len({item.rule.rule for item in result}) != len(result):
            raise RuntimeError("composition graph has competing instructions")
        return tuple(sorted(result, key=lambda item: item.stable_key()))
    finally:
        backend.close()
