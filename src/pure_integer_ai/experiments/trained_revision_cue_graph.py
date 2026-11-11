"""Append-only source-qualified integer graph for natural revision cues."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from pure_integer_ai.storage import discipline as disc
from pure_integer_ai.storage.backend import (
    TYPE_INT,
    SQLiteBackend,
    register_extension_table,
)


GRAPH_FORMAT = "PURE_INTEGER_TRAINED_REVISION_CUE_GRAPH_V1"
GRAPH_KIND = 91690
RECORD_KIND = 91691
REVISION_CUE_TABLE = "trained_revision_cue"
REVISION_CUE_PART_TABLE = "trained_revision_cue_part"
REVISION_CUE_META_TABLE = "trained_revision_cue_graph"
PART_CUE_IDENTITY = 1
PART_CUE_VALUES = 2
PART_ACTION = 3
PART_SOURCE = 4
PART_LICENSE = 5


def _key(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    if (type(value) is not tuple or not value
            or any(type(item) is not int or item < 0 for item in value)):
        raise ValueError(f"{label} must be a nonempty nonnegative integer tuple")
    return value


def _hash(*keys: tuple[int, ...]) -> int:
    digest = hashlib.sha256()
    for key in keys:
        digest.update(len(key).to_bytes(4, "big", signed=False))
        for value in key:
            digest.update(value.to_bytes(8, "big", signed=True))
    result = int.from_bytes(digest.digest()[:8], "big") & ((1 << 63) - 1)
    return result or 1


def _pack(key: tuple[int, ...]) -> tuple[int, ...]:
    return (len(key), *key)


def _unpack(values: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    if not values or values[0] <= 0 or len(values) != values[0] + 1:
        raise ValueError(f"{label} is truncated or empty")
    return values[1:]


@dataclass(frozen=True, slots=True)
class TrainedRevisionCue:
    """One trained cue surface and the integer revision action it signals."""

    cue_identity: tuple[int, ...]
    surface_values: tuple[int, ...]
    action_key: tuple[int, ...]

    def __post_init__(self) -> None:
        _key(self.cue_identity, label="cue_identity")
        _key(self.surface_values, label="surface_values")
        _key(self.action_key, label="action_key")

    def stable_key(self) -> tuple[int, ...]:
        return (
            RECORD_KIND,
            *_pack(self.cue_identity),
            *_pack(self.surface_values),
            *_pack(self.action_key),
        )


@dataclass(frozen=True, slots=True)
class RevisionCueGraphRecord:
    cue: TrainedRevisionCue
    source_identity: tuple[int, ...]
    license_identity: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.cue, TrainedRevisionCue):
            raise TypeError("cue must be a TrainedRevisionCue")
        _key(self.source_identity, label="source_identity")
        _key(self.license_identity, label="license_identity")

    def stable_key(self) -> tuple[int, ...]:
        return (
            GRAPH_KIND,
            *_pack(self.cue.stable_key()),
            *_pack(self.source_identity),
            *_pack(self.license_identity),
        )


def _register(backend: SQLiteBackend) -> None:
    register_extension_table(
        backend, REVISION_CUE_TABLE,
        [("cue_id", TYPE_INT), ("cue_hash", TYPE_INT),
         ("source_hash", TYPE_INT), ("license_hash", TYPE_INT),
         ("surface_count", TYPE_INT)],
        discipline=disc.DISC_APPEND_ONLY,
        indexes=[("cue_id",)], recovery_key=("cue_id",),
    )
    register_extension_table(
        backend, REVISION_CUE_PART_TABLE,
        [("cue_id", TYPE_INT), ("field_code", TYPE_INT),
         ("part_ordinal", TYPE_INT), ("part_value", TYPE_INT)],
        discipline=disc.DISC_APPEND_ONLY,
        indexes=[("cue_id", "field_code", "part_ordinal")],
        recovery_key=("cue_id", "field_code", "part_ordinal"),
    )
    register_extension_table(
        backend, REVISION_CUE_META_TABLE,
        [("graph_id", TYPE_INT), ("graph_kind", TYPE_INT),
         ("source_hash", TYPE_INT), ("license_hash", TYPE_INT),
         ("cue_count", TYPE_INT)],
        discipline=disc.DISC_APPEND_ONLY,
        indexes=[("graph_id",)], recovery_key=("graph_id",),
    )


def _part_rows(cue_id: int, field: int, values: tuple[int, ...]):
    return [{"cue_id": cue_id, "field_code": field,
             "part_ordinal": ordinal, "part_value": value}
            for ordinal, value in enumerate(values)]


def materialize_revision_cue_graph(
        cues: tuple[TrainedRevisionCue, ...], database: str | Path, *,
        source_identity: tuple[int, ...],
        license_identity: tuple[int, ...],
        ) -> dict[str, object]:
    """Materialize a new train-only cue graph; never overwrite an artifact."""
    if (type(cues) is not tuple or not cues
            or any(not isinstance(item, TrainedRevisionCue) for item in cues)):
        raise ValueError("revision cue graph requires trained cue records")
    cues = tuple(sorted(set(cues), key=lambda item: item.stable_key()))
    if len(cues) != len({item.cue_identity for item in cues}):
        raise ValueError("revision cue identities must be unique")
    source_identity = _key(source_identity, label="source_identity")
    license_identity = _key(license_identity, label="license_identity")
    target = Path(database).resolve()
    if target.exists():
        raise FileExistsError("revision cue graph requires a new database path")
    backend = SQLiteBackend(str(target), performance_mode="bulk")
    graph_id = _hash((GRAPH_KIND,), source_identity, license_identity)
    try:
        _register(backend)
        backend.insert(REVISION_CUE_META_TABLE, {
            "graph_id": graph_id,
            "graph_kind": GRAPH_KIND,
            "source_hash": _hash(source_identity),
            "license_hash": _hash(license_identity),
            "cue_count": len(cues),
        })
        for cue in cues:
            cue_id = _hash(cue.stable_key(), source_identity, license_identity)
            backend.insert(REVISION_CUE_TABLE, {
                "cue_id": cue_id,
                "cue_hash": _hash(cue.stable_key()),
                "source_hash": _hash(source_identity),
                "license_hash": _hash(license_identity),
                "surface_count": len(cue.surface_values),
            })
            backend.insert_many(REVISION_CUE_PART_TABLE, [
                *_part_rows(cue_id, PART_CUE_IDENTITY, cue.cue_identity),
                *_part_rows(cue_id, PART_CUE_VALUES, cue.surface_values),
                *_part_rows(cue_id, PART_ACTION, cue.action_key),
                *_part_rows(cue_id, PART_SOURCE, source_identity),
                *_part_rows(cue_id, PART_LICENSE, license_identity),
            ])
        backend.commit()
    finally:
        backend.close()
    return {
        "format": GRAPH_FORMAT,
        "graph_id": graph_id,
        "cue_count": len(cues),
        "source_identity": list(source_identity),
        "license_identity": list(license_identity),
    }


def _read_part(backend: SQLiteBackend, cue_id: int,
               field: int) -> tuple[int, ...]:
    rows = backend.select(
        REVISION_CUE_PART_TABLE,
        where={"cue_id": cue_id, "field_code": field},
        order_by="part_ordinal",
    )
    values = tuple(int(row["part_value"]) for row in rows)
    if (not values
            or tuple(row["part_ordinal"] for row in rows)
            != tuple(range(len(rows)))):
        raise ValueError("revision cue graph parts are incomplete")
    return values


def read_revision_cue_graph(
        database: str | Path,
        ) -> tuple[tuple[RevisionCueGraphRecord, ...], dict[str, object]]:
    """Cold restore cue and provenance identities from the integer graph."""
    backend = SQLiteBackend(str(Path(database).resolve()), read_only=True)
    try:
        _register(backend)
        metas = backend.select(REVISION_CUE_META_TABLE, order_by="graph_id")
        if len(metas) != 1 or metas[0]["graph_kind"] != GRAPH_KIND:
            raise ValueError("revision cue graph metadata is invalid")
        meta = metas[0]
        rows = backend.select(REVISION_CUE_TABLE, order_by="cue_id")
        if len(rows) != meta["cue_count"] or not rows:
            raise ValueError("revision cue graph count drifted")
        records = []
        for row in rows:
            cue_id = int(row["cue_id"])
            cue = TrainedRevisionCue(
                _read_part(backend, cue_id, PART_CUE_IDENTITY),
                _read_part(backend, cue_id, PART_CUE_VALUES),
                _read_part(backend, cue_id, PART_ACTION),
            )
            source = _read_part(backend, cue_id, PART_SOURCE)
            license_key = _read_part(backend, cue_id, PART_LICENSE)
            if (row["cue_hash"] != _hash(cue.stable_key())
                    or cue_id != _hash(cue.stable_key(), source, license_key)
                    or row["source_hash"] != _hash(source)
                    or row["license_hash"] != _hash(license_key)
                    or row["surface_count"] != len(cue.surface_values)):
                raise ValueError("revision cue graph identity drifted")
            records.append(RevisionCueGraphRecord(cue, source, license_key))
        records.sort(key=lambda item: item.stable_key())
        source = records[0].source_identity
        license_key = records[0].license_identity
        if (any(item.source_identity != source
                or item.license_identity != license_key for item in records)
                or meta["source_hash"] != _hash(source)
                or meta["license_hash"] != _hash(license_key)
                or meta["graph_id"] != _hash((GRAPH_KIND,), source, license_key)):
            raise ValueError("revision cue graph provenance drifted")
        if len({item.cue.cue_identity for item in records}) != len(records):
            raise ValueError("revision cue graph contains duplicate identities")
        return tuple(records), {
            "format": GRAPH_FORMAT,
            "graph_id": int(meta["graph_id"]),
            "cue_count": int(meta["cue_count"]),
            "source_identity": source,
            "license_identity": license_key,
        }
    finally:
        backend.close()


__all__ = [
    "GRAPH_FORMAT",
    "GRAPH_KIND",
    "RevisionCueGraphRecord",
    "TrainedRevisionCue",
    "materialize_revision_cue_graph",
    "read_revision_cue_graph",
]
