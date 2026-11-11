"""Materialize and query a source-bound integer dialogue-act grammar."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from pure_integer_ai.storage import discipline as disc
from pure_integer_ai.storage.backend import TYPE_INT, SQLiteBackend, register_extension_table


GRAPH_FORMAT = "PURE_INTEGER_INPUT_DIALOGUE_ACT_GRAPH_V1"
GRAPH_KIND = 91697
RECORD_KIND = 91698
GRAPH_META_TABLE = "input_dialogue_act_graph"
GRAPH_PART_TABLE = "input_dialogue_act_graph_part"
LEXEME_TABLE = "input_dialogue_act_lexeme"
LEXEME_PART_TABLE = "input_dialogue_act_lexeme_part"
FRAME_TABLE = "input_dialogue_act_frame"
FRAME_PART_TABLE = "input_dialogue_act_frame_part"

GRAPH_SOURCE = 1
GRAPH_LICENSE = 2
GRAPH_COURSE_SHA256 = 3
LEXEME_SYMBOL = 1
LEXEME_SURFACE = 2
FRAME_ACT = 1
FRAME_SYMBOL_SEQUENCE = 2


class InputDialogueActGraphError(RuntimeError):
    """The dialogue-act graph is malformed, ambiguous, or provenance-invalid."""


def _strict_key(value: Iterable[int], *, where: str) -> tuple[int, ...]:
    result = tuple(value)
    if (not result or any(type(item) is not int or item < 0 for item in result)):
        raise ValueError(f"{where} must be a non-empty non-negative integer key")
    return result


def _unicode_values(value: Iterable[int], *, where: str) -> tuple[int, ...]:
    result = _strict_key(value, where=where)
    if any(item > 0x10FFFF or 0xD800 <= item <= 0xDFFF for item in result):
        raise ValueError(f"{where} must contain Unicode scalar values")
    return result


def _pack(key: tuple[int, ...]) -> tuple[int, ...]:
    return (len(key), *key)


def _key_hash(*keys: tuple[int, ...]) -> int:
    payload = bytearray()
    for key in keys:
        payload.extend(len(key).to_bytes(4, "big", signed=False))
        for value in key:
            payload.extend(value.to_bytes(8, "big", signed=True))
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    value &= (1 << 63) - 1
    return value or 1


def _symbol_sequence_key(symbols: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    result = [len(symbols)]
    for symbol in symbols:
        result.extend(_pack(symbol))
    return tuple(result)


def _unpack_symbol_sequence(values: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    if not values:
        raise InputDialogueActGraphError("frame symbol sequence is empty")
    count = values[0]
    cursor = 1
    symbols = []
    for _ in range(count):
        if cursor >= len(values):
            raise InputDialogueActGraphError("frame symbol sequence is truncated")
        width = values[cursor]
        cursor += 1
        if width <= 0 or cursor + width > len(values):
            raise InputDialogueActGraphError("frame symbol key is truncated")
        symbols.append(values[cursor:cursor + width])
        cursor += width
    if cursor != len(values):
        raise InputDialogueActGraphError("frame symbol sequence has trailing values")
    return tuple(symbols)


@dataclass(frozen=True, slots=True)
class TrainedInputDialogueActLexeme:
    """A language surface linked to one trained semantic symbol."""

    symbol_key: tuple[int, ...]
    surface_values: tuple[int, ...]

    def __post_init__(self) -> None:
        _strict_key(self.symbol_key, where="dialogue-act symbol key")
        _unicode_values(self.surface_values, where="dialogue-act surface")

    def stable_key(self) -> tuple[int, ...]:
        return (RECORD_KIND, 1, *_pack(self.symbol_key),
                *_pack(self.surface_values))


@dataclass(frozen=True, slots=True)
class TrainedInputDialogueActFrame:
    """An ordered semantic-symbol pattern that expresses one dialogue act."""

    act_key: tuple[int, ...]
    response_feature_mask: int
    symbol_sequence: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        _strict_key(self.act_key, where="dialogue-act key")
        if (type(self.response_feature_mask) is not int
                or self.response_feature_mask <= 0):
            raise ValueError("dialogue-act response feature mask must be positive")
        if (type(self.symbol_sequence) is not tuple or not self.symbol_sequence
                or any(type(item) is not tuple or not item
                       for item in self.symbol_sequence)):
            raise ValueError("dialogue-act frame must contain ordered symbols")
        for symbol in self.symbol_sequence:
            _strict_key(symbol, where="dialogue-act frame symbol")

    def stable_key(self) -> tuple[int, ...]:
        return (RECORD_KIND, 2, *_pack(self.act_key),
                self.response_feature_mask,
                *_pack(_symbol_sequence_key(self.symbol_sequence)))


@dataclass(frozen=True, slots=True)
class InputDialogueActMatch:
    """A unique act whose trained lexical path covers the complete input."""

    act_key: tuple[int, ...]
    response_feature_mask: int
    frame_keys: tuple[tuple[int, ...], ...]

    def stable_key(self) -> tuple[int, ...]:
        result = [RECORD_KIND, 3, *_pack(self.act_key),
                  self.response_feature_mask, len(self.frame_keys)]
        for key in self.frame_keys:
            result.extend(_pack(key))
        return tuple(result)


@dataclass(frozen=True, slots=True)
class TrainedInputDialogueActGraph:
    lexemes: tuple[TrainedInputDialogueActLexeme, ...]
    frames: tuple[TrainedInputDialogueActFrame, ...]
    source_identity: tuple[int, ...]
    license_identity: tuple[int, ...]
    course_sha256: tuple[int, ...]
    graph_id: int

    def match(self, surface_values: tuple[int, ...]) -> InputDialogueActMatch | None:
        """Match only complete, unambiguous integer paths through this graph."""
        if (type(surface_values) is not tuple
                or any(type(item) is not int for item in surface_values)):
            raise TypeError("dialogue-act input must be an integer tuple")
        if not surface_values or len(surface_values) > 4096:
            return None
        by_symbol: dict[tuple[int, ...], tuple[tuple[int, ...], ...]] = {}
        for lexeme in self.lexemes:
            by_symbol.setdefault(lexeme.symbol_key, ())
            if lexeme.surface_values not in by_symbol[lexeme.symbol_key]:
                by_symbol[lexeme.symbol_key] += (lexeme.surface_values,)
        matched = {}
        for frame in self.frames:
            memo: dict[
                tuple[int, int],
                tuple[tuple[tuple[int, ...], ...], ...],
            ] = {}

            def visit(symbol_ordinal: int, cursor: int):
                """Return at most two distinct lexical paths for ambiguity checks."""
                state = (symbol_ordinal, cursor)
                cached = memo.get(state)
                if cached is not None:
                    return cached
                if symbol_ordinal == len(frame.symbol_sequence):
                    result = ((),) if cursor == len(surface_values) else ()
                else:
                    paths = []
                    seen_paths = set()
                    symbol = frame.symbol_sequence[symbol_ordinal]
                    for lexical_surface in sorted(by_symbol.get(symbol, ())):
                        end = cursor + len(lexical_surface)
                        if (end > len(surface_values)
                                or surface_values[cursor:end] != lexical_surface):
                            continue
                        for suffix in visit(symbol_ordinal + 1, end):
                            path = (lexical_surface, *suffix)
                            if path not in seen_paths:
                                seen_paths.add(path)
                                paths.append(path)
                                if len(paths) == 2:
                                    break
                        if len(paths) == 2:
                            break
                    result = tuple(paths)
                memo[state] = result
                return result

            for lexical_path in visit(0, 0):
                key = (frame.stable_key(), lexical_path)
                matched[key] = frame
                if len(matched) > 1:
                    return None
        if len(matched) != 1:
            return None
        frame = next(iter(matched.values()))
        return InputDialogueActMatch(
            frame.act_key, frame.response_feature_mask,
            (frame.stable_key(),))


def _register_tables(backend) -> None:
    register_extension_table(
        backend, GRAPH_META_TABLE,
        [("graph_id", TYPE_INT), ("graph_kind", TYPE_INT),
         ("format_version", TYPE_INT), ("lexeme_count", TYPE_INT),
         ("frame_count", TYPE_INT), ("source_hash", TYPE_INT),
         ("license_hash", TYPE_INT), ("course_hash", TYPE_INT)],
        discipline=disc.DISC_APPEND_ONLY,
        indexes=[("graph_id",)], recovery_key=("graph_id",))
    register_extension_table(
        backend, GRAPH_PART_TABLE,
        [("graph_id", TYPE_INT), ("field_code", TYPE_INT),
         ("part_ordinal", TYPE_INT), ("part_value", TYPE_INT)],
        discipline=disc.DISC_APPEND_ONLY,
        indexes=[("graph_id", "field_code", "part_ordinal")],
        recovery_key=("graph_id", "field_code", "part_ordinal"))
    register_extension_table(
        backend, LEXEME_TABLE,
        [("graph_id", TYPE_INT), ("lexeme_id", TYPE_INT),
         ("symbol_hash", TYPE_INT), ("surface_hash", TYPE_INT),
         ("symbol_count", TYPE_INT), ("surface_count", TYPE_INT),
         ("source_hash", TYPE_INT), ("license_hash", TYPE_INT)],
        discipline=disc.DISC_APPEND_ONLY,
        indexes=[("graph_id",), ("lexeme_id",), ("symbol_hash",)],
        recovery_key=("graph_id", "lexeme_id"))
    register_extension_table(
        backend, LEXEME_PART_TABLE,
        [("lexeme_id", TYPE_INT), ("field_code", TYPE_INT),
         ("part_ordinal", TYPE_INT), ("part_value", TYPE_INT)],
        discipline=disc.DISC_APPEND_ONLY,
        indexes=[("lexeme_id", "field_code", "part_ordinal")],
        recovery_key=("lexeme_id", "field_code", "part_ordinal"))
    register_extension_table(
        backend, FRAME_TABLE,
        [("graph_id", TYPE_INT), ("frame_id", TYPE_INT),
         ("act_hash", TYPE_INT), ("symbol_count", TYPE_INT),
         ("response_feature_mask", TYPE_INT),
         ("source_hash", TYPE_INT), ("license_hash", TYPE_INT)],
        discipline=disc.DISC_APPEND_ONLY,
        indexes=[("graph_id",), ("frame_id",), ("act_hash",)],
        recovery_key=("graph_id", "frame_id"))
    register_extension_table(
        backend, FRAME_PART_TABLE,
        [("frame_id", TYPE_INT), ("field_code", TYPE_INT),
         ("part_ordinal", TYPE_INT), ("part_value", TYPE_INT)],
        discipline=disc.DISC_APPEND_ONLY,
        indexes=[("frame_id", "field_code", "part_ordinal")],
        recovery_key=("frame_id", "field_code", "part_ordinal"))


def _part_rows(owner_column: str, owner: int, field: int,
               values: tuple[int, ...]):
    return [{owner_column: owner, "field_code": field,
             "part_ordinal": ordinal, "part_value": value}
            for ordinal, value in enumerate(values)]


def _read_parts(backend, table: str, owner_column: str,
                owner: int, field: int) -> tuple[int, ...]:
    rows = backend.select(table, where={owner_column: owner, "field_code": field},
                          order_by="part_ordinal")
    values = tuple(int(row["part_value"]) for row in rows)
    if (not values or tuple(int(row["part_ordinal"]) for row in rows)
            != tuple(range(len(rows)))):
        raise InputDialogueActGraphError("dialogue-act graph parts are incomplete")
    return values


def materialize_input_dialogue_act_graph(
        lexemes: tuple[TrainedInputDialogueActLexeme, ...],
        frames: tuple[TrainedInputDialogueActFrame, ...],
        database: str | Path,
        *, source_identity: tuple[int, ...],
        license_identity: tuple[int, ...],
        course_sha256: tuple[int, ...],
        ) -> dict[str, int | str | list[int]]:
    """Write a new append-only SQLite graph containing integers only."""
    if (type(lexemes) is not tuple or not lexemes
            or any(not isinstance(item, TrainedInputDialogueActLexeme)
                   for item in lexemes)
            or type(frames) is not tuple or not frames
            or any(not isinstance(item, TrainedInputDialogueActFrame)
                   for item in frames)):
        raise TypeError("dialogue-act graph rows must be non-empty typed tuples")
    lexemes = tuple(sorted(set(lexemes), key=lambda item: item.stable_key()))
    frames = tuple(sorted(set(frames), key=lambda item: item.stable_key()))
    source_identity = _strict_key(source_identity, where="source identity")
    license_identity = _strict_key(license_identity, where="license identity")
    if (type(course_sha256) is not tuple or len(course_sha256) != 32
            or any(type(item) is not int or not 0 <= item <= 255
                   for item in course_sha256)):
        raise ValueError("course SHA-256 must be a 32-byte integer tuple")
    known_symbols = {item.symbol_key for item in lexemes}
    if any(symbol not in known_symbols
           for frame in frames for symbol in frame.symbol_sequence):
        raise ValueError("dialogue-act frame references an absent lexeme symbol")
    target = Path(database).resolve()
    if target.exists():
        raise ValueError(f"dialogue-act graph requires a new path: {target}")
    graph_id = _key_hash((GRAPH_KIND,), source_identity, license_identity,
                         course_sha256)
    source_hash = _key_hash(source_identity)
    license_hash = _key_hash(license_identity)
    backend = SQLiteBackend(str(target), performance_mode="bulk")
    try:
        _register_tables(backend)
        backend.insert(GRAPH_META_TABLE, {
            "graph_id": graph_id, "graph_kind": GRAPH_KIND,
            "format_version": 1, "lexeme_count": len(lexemes),
            "frame_count": len(frames), "source_hash": source_hash,
            "license_hash": license_hash,
            "course_hash": _key_hash(course_sha256),
        })
        graph_parts = []
        for field, key in ((GRAPH_SOURCE, source_identity),
                           (GRAPH_LICENSE, license_identity),
                           (GRAPH_COURSE_SHA256, course_sha256)):
            graph_parts.extend(_part_rows(
                "graph_id", graph_id, field, key))
        backend.insert_many(GRAPH_PART_TABLE, graph_parts)
        for item in lexemes:
            lexeme_id = _key_hash((graph_id,), item.symbol_key,
                                  item.surface_values)
            backend.insert(LEXEME_TABLE, {
                "graph_id": graph_id, "lexeme_id": lexeme_id,
                "symbol_hash": _key_hash(item.symbol_key),
                "surface_hash": _key_hash(item.surface_values),
                "symbol_count": len(item.symbol_key),
                "surface_count": len(item.surface_values),
                "source_hash": source_hash, "license_hash": license_hash,
            })
            rows = []
            for field, key in ((LEXEME_SYMBOL, item.symbol_key),
                               (LEXEME_SURFACE, item.surface_values)):
                rows.extend(_part_rows(
                    "lexeme_id", lexeme_id, field, key))
            backend.insert_many(LEXEME_PART_TABLE, rows)
        for item in frames:
            frame_id = _key_hash((graph_id,), item.act_key,
                                 (item.response_feature_mask,),
                                 _symbol_sequence_key(item.symbol_sequence))
            backend.insert(FRAME_TABLE, {
                "graph_id": graph_id, "frame_id": frame_id,
                "act_hash": _key_hash(item.act_key),
                "symbol_count": len(item.symbol_sequence),
                "response_feature_mask": item.response_feature_mask,
                "source_hash": source_hash, "license_hash": license_hash,
            })
            rows = []
            for field, key in (
                    (FRAME_ACT, item.act_key),
                    (FRAME_SYMBOL_SEQUENCE,
                     _symbol_sequence_key(item.symbol_sequence))):
                rows.extend(_part_rows(
                    "frame_id", frame_id, field, key))
            backend.insert_many(FRAME_PART_TABLE, rows)
        backend.commit()
    finally:
        backend.close()
    return {
        "format": GRAPH_FORMAT, "graph_id": graph_id,
        "lexeme_count": len(lexemes), "frame_count": len(frames),
        "source_identity": list(source_identity),
        "license_identity": list(license_identity),
    }


def read_input_dialogue_act_graph(
        database: str | Path,
        ) -> TrainedInputDialogueActGraph:
    """Cold-restore and verify every graph edge and provenance commitment."""
    backend = SQLiteBackend(str(Path(database).resolve()), read_only=True)
    try:
        _register_tables(backend)
        metas = backend.select(GRAPH_META_TABLE, order_by="graph_id")
        if len(metas) != 1 or int(metas[0]["graph_kind"]) != GRAPH_KIND:
            raise InputDialogueActGraphError("dialogue-act graph metadata is not unique")
        meta = metas[0]
        graph_id = int(meta["graph_id"])
        source = _read_parts(backend, GRAPH_PART_TABLE, "graph_id",
                             graph_id, GRAPH_SOURCE)
        license_key = _read_parts(backend, GRAPH_PART_TABLE, "graph_id",
                                  graph_id, GRAPH_LICENSE)
        course_digest = _read_parts(backend, GRAPH_PART_TABLE, "graph_id",
                                    graph_id, GRAPH_COURSE_SHA256)
        if (len(course_digest) != 32
                or int(meta["source_hash"]) != _key_hash(source)
                or int(meta["license_hash"]) != _key_hash(license_key)
                or int(meta["course_hash"]) != _key_hash(course_digest)
                or graph_id != _key_hash((GRAPH_KIND,), source, license_key,
                                         course_digest)):
            raise InputDialogueActGraphError("dialogue-act graph provenance mismatch")

        lexeme_rows = backend.select(LEXEME_TABLE, where={"graph_id": graph_id},
                                     order_by="lexeme_id")
        lexemes = []
        for row in lexeme_rows:
            lexeme_id = int(row["lexeme_id"])
            symbol = _read_parts(backend, LEXEME_PART_TABLE, "lexeme_id",
                                 lexeme_id, LEXEME_SYMBOL)
            surface = _read_parts(backend, LEXEME_PART_TABLE, "lexeme_id",
                                  lexeme_id, LEXEME_SURFACE)
            if (int(row["source_hash"]) != _key_hash(source)
                    or int(row["license_hash"]) != _key_hash(license_key)
                    or int(row["symbol_hash"]) != _key_hash(symbol)
                    or int(row["surface_hash"]) != _key_hash(surface)
                    or int(row["symbol_count"]) != len(symbol)
                    or int(row["surface_count"]) != len(surface)
                    or lexeme_id != _key_hash((graph_id,), symbol, surface)):
                raise InputDialogueActGraphError("dialogue-act lexeme identity mismatch")
            lexemes.append(TrainedInputDialogueActLexeme(symbol, surface))
        frame_rows = backend.select(FRAME_TABLE, where={"graph_id": graph_id},
                                    order_by="frame_id")
        frames = []
        for row in frame_rows:
            frame_id = int(row["frame_id"])
            act = _read_parts(backend, FRAME_PART_TABLE, "frame_id",
                              frame_id, FRAME_ACT)
            sequence = _unpack_symbol_sequence(_read_parts(
                backend, FRAME_PART_TABLE, "frame_id", frame_id,
                FRAME_SYMBOL_SEQUENCE))
            feature_mask = int(row["response_feature_mask"])
            if (int(row["source_hash"]) != _key_hash(source)
                    or int(row["license_hash"]) != _key_hash(license_key)
                    or int(row["act_hash"]) != _key_hash(act)
                    or int(row["symbol_count"]) != len(sequence)
                    or feature_mask <= 0
                    or frame_id != _key_hash(
                        (graph_id,), act, (feature_mask,),
                        _symbol_sequence_key(sequence))):
                raise InputDialogueActGraphError("dialogue-act frame identity mismatch")
            frames.append(TrainedInputDialogueActFrame(
                act, feature_mask, sequence))
        lexemes = tuple(sorted(lexemes, key=lambda item: item.stable_key()))
        frames = tuple(sorted(frames, key=lambda item: item.stable_key()))
        if (len(lexemes) != int(meta["lexeme_count"])
                or len(frames) != int(meta["frame_count"])
                or len(set(item.stable_key() for item in lexemes)) != len(lexemes)
                or len(set(item.stable_key() for item in frames)) != len(frames)):
            raise InputDialogueActGraphError("dialogue-act graph row counts differ")
        known_symbols = {item.symbol_key for item in lexemes}
        if any(symbol not in known_symbols
               for frame in frames for symbol in frame.symbol_sequence):
            raise InputDialogueActGraphError("dialogue-act frame has no lexical support")
        return TrainedInputDialogueActGraph(
            lexemes, frames, source, license_key, course_digest, graph_id)
    finally:
        backend.close()


__all__ = [
    "FRAME_ACT", "FRAME_SYMBOL_SEQUENCE", "GRAPH_FORMAT", "GRAPH_KIND",
    "GRAPH_LICENSE", "GRAPH_META_TABLE", "GRAPH_PART_TABLE",
    "GRAPH_SOURCE", "InputDialogueActGraphError", "InputDialogueActMatch",
    "LEXEME_SURFACE", "LEXEME_SYMBOL", "RECORD_KIND",
    "TrainedInputDialogueActFrame", "TrainedInputDialogueActGraph",
    "TrainedInputDialogueActLexeme", "materialize_input_dialogue_act_graph",
    "read_input_dialogue_act_graph",
]
