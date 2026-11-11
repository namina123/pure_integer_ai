"""Append-only integer graph for source-qualified variable question frames."""
from __future__ import annotations

import hashlib
from pathlib import Path

from pure_integer_ai.cognition.understanding.query_relation_questions import (
    TrainedRelationQuestionFrame,
)
from pure_integer_ai.storage import discipline as disc
from pure_integer_ai.storage.backend import TYPE_INT, SQLiteBackend, register_extension_table


GRAPH_FORMAT = "PURE_INTEGER_RELATION_QUESTION_GRAPH_V1"
GRAPH_KIND = 92025
FRAME_TABLE = "relation_question_frame"
PART_TABLE = "relation_question_frame_part"
META_TABLE = "relation_question_graph"
PART_FRAME = 1
PART_SOURCE = 2
PART_LICENSE = 3
MODE_RELATION_PAIR_QUERY = (91672, 4)


def _key(value: tuple[int, ...]) -> tuple[int, ...]:
    if (type(value) is not tuple or not value
            or any(type(item) is not int or item < 0 for item in value)):
        raise ValueError("question graph requires a nonempty integer key")
    return value


def _hash(*keys: tuple[int, ...]) -> int:
    digest = hashlib.sha256()
    for key in keys:
        digest.update(len(key).to_bytes(4, "big"))
        for value in key:
            digest.update(value.to_bytes(8, "big", signed=True))
    return (int.from_bytes(digest.digest()[:8], "big") & ((1 << 63) - 1)) or 1


def _register(backend: SQLiteBackend) -> None:
    register_extension_table(
        backend, FRAME_TABLE,
        [("frame_id", TYPE_INT), ("frame_hash", TYPE_INT),
         ("source_hash", TYPE_INT), ("license_hash", TYPE_INT)],
        discipline=disc.DISC_APPEND_ONLY, indexes=[("frame_id",)],
        recovery_key=("frame_id",),
    )
    register_extension_table(
        backend, PART_TABLE,
        [("frame_id", TYPE_INT), ("field_code", TYPE_INT),
         ("part_ordinal", TYPE_INT), ("part_value", TYPE_INT)],
        discipline=disc.DISC_APPEND_ONLY,
        indexes=[("frame_id", "field_code", "part_ordinal")],
        recovery_key=("frame_id", "field_code", "part_ordinal"),
    )
    register_extension_table(
        backend, META_TABLE,
        [("graph_id", TYPE_INT), ("graph_kind", TYPE_INT),
         ("source_hash", TYPE_INT), ("license_hash", TYPE_INT),
         ("frame_count", TYPE_INT)],
        discipline=disc.DISC_APPEND_ONLY, indexes=[("graph_id",)],
        recovery_key=("graph_id",),
    )


def _parts(frame_id: int, field: int, key: tuple[int, ...]) -> list[dict[str, int]]:
    return [
        {"frame_id": frame_id, "field_code": field,
         "part_ordinal": index, "part_value": value}
        for index, value in enumerate(key)
    ]


def materialize_relation_question_graph(
        frames: tuple[TrainedRelationQuestionFrame, ...], database: str | Path, *,
        source_identity: tuple[int, ...], license_identity: tuple[int, ...],
        ) -> dict[str, object]:
    if (type(frames) is not tuple or not frames
            or any(not isinstance(item, TrainedRelationQuestionFrame) for item in frames)
            or len(set(frames)) != len(frames)):
        raise ValueError("question graph requires unique trained frames")
    source_identity = _key(source_identity)
    license_identity = _key(license_identity)
    target = Path(database).resolve()
    if target.exists():
        raise FileExistsError("question graph requires a new database path")
    backend = SQLiteBackend(str(target), performance_mode="bulk")
    graph_id = _hash((GRAPH_KIND,), source_identity, license_identity)
    try:
        _register(backend)
        backend.insert(META_TABLE, {
            "graph_id": graph_id, "graph_kind": GRAPH_KIND,
            "source_hash": _hash(source_identity),
            "license_hash": _hash(license_identity), "frame_count": len(frames),
        })
        for frame in sorted(frames, key=lambda item: item.stable_key()):
            record = frame.stable_key()
            frame_id = _hash(record, source_identity, license_identity)
            if backend.select(FRAME_TABLE, where={"frame_id": frame_id}):
                raise ValueError("question frame identity collision")
            backend.insert(FRAME_TABLE, {
                "frame_id": frame_id, "frame_hash": _hash(record),
                "source_hash": _hash(source_identity),
                "license_hash": _hash(license_identity),
            })
            backend.insert_many(PART_TABLE, [
                *_parts(frame_id, PART_FRAME, record),
                *_parts(frame_id, PART_SOURCE, source_identity),
                *_parts(frame_id, PART_LICENSE, license_identity),
            ])
        backend.commit()
    finally:
        backend.close()
    return {
        "format": GRAPH_FORMAT, "graph_id": graph_id,
        "frame_count": len(frames),
        "source_identity": list(source_identity),
        "license_identity": list(license_identity),
    }


def _read_part(backend: SQLiteBackend, frame_id: int, field: int) -> tuple[int, ...]:
    rows = backend.select(PART_TABLE,
                          where={"frame_id": frame_id, "field_code": field},
                          order_by="part_ordinal")
    if not rows or tuple(row["part_ordinal"] for row in rows) != tuple(range(len(rows))):
        raise RuntimeError("question graph integer parts are incomplete")
    return tuple(row["part_value"] for row in rows)


def read_relation_question_graph(
        database: str | Path,
        ) -> tuple[tuple[TrainedRelationQuestionFrame, ...], dict[str, object]]:
    backend = SQLiteBackend(str(Path(database).resolve()), read_only=True)
    try:
        _register(backend)
        metas = backend.select(META_TABLE, order_by="graph_id")
        if len(metas) != 1 or metas[0]["graph_kind"] != GRAPH_KIND:
            raise RuntimeError("question graph metadata is invalid")
        meta = metas[0]
        rows = backend.select(FRAME_TABLE, order_by="frame_id")
        if not rows or len(rows) != meta["frame_count"]:
            raise RuntimeError("question graph frame count drifted")
        restored = []
        for row in rows:
            record = _read_part(backend, row["frame_id"], PART_FRAME)
            source = _read_part(backend, row["frame_id"], PART_SOURCE)
            license_key = _read_part(backend, row["frame_id"], PART_LICENSE)
            if (row["frame_hash"] != _hash(record)
                    or row["frame_id"] != _hash(record, source, license_key)
                    or row["source_hash"] != _hash(source)
                    or row["license_hash"] != _hash(license_key)):
                raise RuntimeError("question graph frame identity drifted")
            restored.append((TrainedRelationQuestionFrame.from_stable_key(record),
                             source, license_key))
        if (len({item[0].stable_key() for item in restored}) != len(restored)
                or any(item[1] != restored[0][1] or item[2] != restored[0][2]
                       for item in restored[1:])):
            raise RuntimeError("question graph provenance is not uniform")
        source = restored[0][1]
        license_key = restored[0][2]
        if (meta["source_hash"] != _hash(source)
                or meta["license_hash"] != _hash(license_key)
                or meta["graph_id"] != _hash((GRAPH_KIND,), source, license_key)):
            raise RuntimeError("question graph metadata identity drifted")
        return tuple(sorted((item[0] for item in restored),
                            key=lambda item: item.stable_key())), {
            "format": GRAPH_FORMAT, "graph_id": meta["graph_id"],
            "frame_count": len(restored),
            "source_identity": source, "license_identity": license_key,
        }
    finally:
        backend.close()


__all__ = [
    "GRAPH_FORMAT", "MODE_RELATION_PAIR_QUERY",
    "materialize_relation_question_graph", "read_relation_question_graph",
]
