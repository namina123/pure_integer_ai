"""Integer-only materialization of trained relation surfaces and frames.

The training Core keeps source records for construction-time Span validation.
Published runtimes can instead consume this artifact, whose payload is made
only of canonical integer tuples.  No source text, JSONL course, or source
path is part of the artifact.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import sqlite3
import tempfile
from typing import Iterable

from pure_integer_ai.cognition.shared.identity import ObjectIdentity
from pure_integer_ai.storage.integer_codec import (
    decode_integer_tuple,
    encode_integer_tuple,
)
from pure_integer_ai.experiments.trained_relation_graph_runtime import (
    ActiveRelationSurface,
    RelationSurfaceBinding,
    RelationSurfaceFrame,
    TrainedRelationGraphRuntime,
)


ARTIFACT_FORMAT = "PURE_INTEGER_RELATION_SURFACE_ARTIFACT_V1"
ARTIFACT_SCHEMA_VERSION = 1
_FACT_ROW = 1
_FRAME_ROW = 2
_LANGUAGE_FACT_ROW = 3
LANGUAGE_ARTIFACT_SCHEMA_VERSION = 2
CARRIER_LANGUAGE_ARTIFACT_SCHEMA_VERSION = 3
_CARRIER_LANGUAGE_FRAME_ROW = 4
_CARRIER_LANGUAGE_FACT_ROW = 5
_SUPPORTED_SCHEMAS = (ARTIFACT_SCHEMA_VERSION, LANGUAGE_ARTIFACT_SCHEMA_VERSION,
                      CARRIER_LANGUAGE_ARTIFACT_SCHEMA_VERSION)


class RelationSurfaceArtifactError(ValueError):
    """The integer surface artifact is malformed or not Core-bound."""


def sha256_path(path: str | Path) -> str:
    """Return the deterministic SHA-256 of one regular file."""
    target = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_key(value: object, *, label: str) -> tuple[int, ...]:
    if not isinstance(value, tuple) or not value:
        raise RelationSurfaceArtifactError(f"{label} must be a non-empty integer tuple")
    if any(type(item) is not int for item in value):
        raise RelationSurfaceArtifactError(f"{label} must contain strict integers")
    return value


def _pack_key(values: tuple[int, ...]) -> list[int]:
    values = _strict_key(values, label="identity key")
    return [len(values), *values]


def _read_key(values: tuple[int, ...], cursor: int, *, label: str
              ) -> tuple[tuple[int, ...], int]:
    if cursor >= len(values):
        raise RelationSurfaceArtifactError(f"{label} is truncated")
    size = values[cursor]
    if type(size) is not int or size <= 0 or cursor + 1 + size > len(values):
        raise RelationSurfaceArtifactError(f"{label} has an invalid size")
    key = values[cursor + 1:cursor + 1 + size]
    if any(type(item) is not int for item in key):
        raise RelationSurfaceArtifactError(f"{label} is not integer encoded")
    return key, cursor + 1 + size


def _pack_codepoints(
        surface: str, *, label: str, allow_empty: bool = False,
        ) -> list[int]:
    if type(surface) is not str or (not allow_empty and not surface):
        raise RelationSurfaceArtifactError(f"{label} must be non-empty text")
    return [len(surface), *(ord(value) for value in surface)]


def _read_codepoints(
        values: tuple[int, ...], cursor: int, *, label: str,
        allow_empty: bool = False,
                     ) -> tuple[str, int]:
    if cursor >= len(values):
        raise RelationSurfaceArtifactError(f"{label} is truncated")
    size = values[cursor]
    if (type(size) is not int or size < (0 if allow_empty else 1)
            or cursor + 1 + size > len(values)):
        raise RelationSurfaceArtifactError(f"{label} has an invalid size")
    codepoints = values[cursor + 1:cursor + 1 + size]
    if any(type(item) is not int or not 0 <= item <= 0x10FFFF
           for item in codepoints):
        raise RelationSurfaceArtifactError(f"{label} has invalid codepoints")
    try:
        surface = "".join(map(chr, codepoints))
    except (TypeError, ValueError) as error:
        raise RelationSurfaceArtifactError(f"{label} cannot be restored") from error
    return surface, cursor + 1 + size


def _pack_fact(fact: ActiveRelationSurface, *, schema_version: int = ARTIFACT_SCHEMA_VERSION) -> tuple[int, ...]:
    """按冻结版本编码；V2不含完整来源句，不能用空正文槽冒充移除。"""
    if type(schema_version) is not int or schema_version not in _SUPPORTED_SCHEMAS:
        raise RelationSurfaceArtifactError("surface artifact schema is unsupported")
    values: list[int] = [_FACT_ROW if schema_version == ARTIFACT_SCHEMA_VERSION else
                         _CARRIER_LANGUAGE_FACT_ROW if schema_version == CARRIER_LANGUAGE_ARTIFACT_SCHEMA_VERSION else
                         _LANGUAGE_FACT_ROW]
    values.extend(_pack_key(fact.proposition.stable_key()))
    values.extend(_pack_key(fact.predicate.stable_key()))
    values.extend(_pack_codepoints(fact.cue, label="relation cue"))
    values.extend((fact.source_hash, fact.cue_start, fact.cue_end))
    if schema_version == ARTIFACT_SCHEMA_VERSION:
        values.extend(_pack_codepoints(fact.evidence_surface, label="evidence surface"))
    values.append(len(fact.bindings))
    for binding in fact.bindings:
        values.extend(_pack_key(binding.role.stable_key()))
        values.extend(_pack_key(binding.filler.stable_key()))
        values.extend(_pack_codepoints(binding.surface, label="binding surface"))
        values.extend((binding.source_hash, binding.start, binding.end))
    return tuple(values)


def _unpack_fact(values: tuple[int, ...], *, schema_version: int = ARTIFACT_SCHEMA_VERSION) -> ActiveRelationSurface:
    """严格按版本恢复；V2永不读取或合成来源句。"""
    tag = (_FACT_ROW if schema_version == ARTIFACT_SCHEMA_VERSION else
           _CARRIER_LANGUAGE_FACT_ROW if schema_version == CARRIER_LANGUAGE_ARTIFACT_SCHEMA_VERSION else
           _LANGUAGE_FACT_ROW)
    if type(schema_version) is not int or schema_version not in _SUPPORTED_SCHEMAS or not values or values[0] != tag:
        raise RelationSurfaceArtifactError("relation fact row kind is invalid")
    cursor = 1
    proposition_key, cursor = _read_key(values, cursor, label="fact proposition")
    predicate_key, cursor = _read_key(values, cursor, label="fact predicate")
    cue, cursor = _read_codepoints(values, cursor, label="fact cue")
    if cursor + 3 > len(values):
        raise RelationSurfaceArtifactError("fact source envelope is truncated")
    source_hash, cue_start, cue_end = values[cursor:cursor + 3]
    cursor += 3
    evidence_surface = ""
    if schema_version == ARTIFACT_SCHEMA_VERSION:
        evidence_surface, cursor = _read_codepoints(
            values, cursor, label="fact evidence surface")
    if cursor >= len(values):
        raise RelationSurfaceArtifactError("fact binding count is missing")
    binding_count = values[cursor]
    cursor += 1
    if type(binding_count) is not int or binding_count <= 0:
        raise RelationSurfaceArtifactError("fact binding count is invalid")
    bindings = []
    for _ in range(binding_count):
        role_key, cursor = _read_key(values, cursor, label="binding role")
        filler_key, cursor = _read_key(values, cursor, label="binding filler")
        surface, cursor = _read_codepoints(values, cursor, label="binding surface")
        if cursor + 3 > len(values):
            raise RelationSurfaceArtifactError("binding span is truncated")
        binding_source_hash, start, end = values[cursor:cursor + 3]
        cursor += 3
        try:
            role = ObjectIdentity.from_stable_key(role_key)
            filler = ObjectIdentity.from_stable_key(filler_key)
        except (TypeError, ValueError) as error:
            raise RelationSurfaceArtifactError("binding identity is invalid") from error
        bindings.append(RelationSurfaceBinding(
            role, filler, surface, binding_source_hash, start, end))
    if cursor != len(values):
        raise RelationSurfaceArtifactError("fact row has trailing integers")
    try:
        proposition = ObjectIdentity.from_stable_key(proposition_key)
        predicate = ObjectIdentity.from_stable_key(predicate_key)
    except (TypeError, ValueError) as error:
        raise RelationSurfaceArtifactError("fact identity is invalid") from error
    return ActiveRelationSurface(
        proposition, predicate, cue, tuple(bindings), evidence_surface,
        source_hash, cue_start, cue_end)


def _pack_frame(frame: RelationSurfaceFrame, *, schema_version: int = ARTIFACT_SCHEMA_VERSION) -> tuple[int, ...]:
    """版本3显式保存载体边缘；旧版本不得静默丢弃新成员。"""
    if type(schema_version) is not int or schema_version not in _SUPPORTED_SCHEMAS:
        raise RelationSurfaceArtifactError("frame schema is unsupported")
    carrier = schema_version == CARRIER_LANGUAGE_ARTIFACT_SCHEMA_VERSION
    if not carrier and (frame.carrier_prefix is not None or frame.carrier_suffix is not None):
        raise RelationSurfaceArtifactError("legacy frame cannot discard carrier members")
    values: list[int] = [_CARRIER_LANGUAGE_FRAME_ROW if carrier else _FRAME_ROW]
    values.extend(_pack_key(frame.proposition.stable_key()))
    values.extend(_pack_key(frame.predicate.stable_key()))
    values.append(len(frame.roles))
    for role in frame.roles:
        values.extend(_pack_key(role.stable_key()))
    values.append(len(frame.gaps))
    for gap in frame.gaps:
        values.extend(_pack_codepoints(gap, label="frame gap", allow_empty=True))
    values.extend((frame.source_hash, frame.envelope_start, frame.envelope_end))
    if carrier:
        values.extend(_pack_codepoints(frame.carrier_prefix, label="carrier prefix", allow_empty=True))
        values.extend(_pack_codepoints(frame.carrier_suffix, label="carrier suffix", allow_empty=True))
    return tuple(values)


def _unpack_frame(values: tuple[int, ...], *, schema_version: int = ARTIFACT_SCHEMA_VERSION) -> RelationSurfaceFrame:
    """严格恢复完整版本，不以空串补齐旧模型缺失的载体语言。"""
    carrier = schema_version == CARRIER_LANGUAGE_ARTIFACT_SCHEMA_VERSION
    if (type(schema_version) is not int or schema_version not in _SUPPORTED_SCHEMAS
            or not values or values[0] != (_CARRIER_LANGUAGE_FRAME_ROW if carrier else _FRAME_ROW)):
        raise RelationSurfaceArtifactError("relation frame row kind is invalid")
    cursor = 1
    proposition_key, cursor = _read_key(values, cursor, label="frame proposition")
    predicate_key, cursor = _read_key(values, cursor, label="frame predicate")
    if cursor >= len(values):
        raise RelationSurfaceArtifactError("frame role count is missing")
    role_count = values[cursor]
    cursor += 1
    if type(role_count) is not int or role_count <= 0:
        raise RelationSurfaceArtifactError("frame role count is invalid")
    roles = []
    for _ in range(role_count):
        role_key, cursor = _read_key(values, cursor, label="frame role")
        try:
            roles.append(ObjectIdentity.from_stable_key(role_key))
        except (TypeError, ValueError) as error:
            raise RelationSurfaceArtifactError("frame role identity is invalid") from error
    if cursor >= len(values):
        raise RelationSurfaceArtifactError("frame gap count is missing")
    gap_count = values[cursor]
    cursor += 1
    if type(gap_count) is not int or gap_count != role_count + 1:
        raise RelationSurfaceArtifactError("frame gap count does not match roles")
    gaps = []
    for _ in range(gap_count):
        gap, cursor = _read_codepoints(
            values, cursor, label="frame gap", allow_empty=True)
        gaps.append(gap)
    if cursor + 3 > len(values):
        raise RelationSurfaceArtifactError("frame envelope is missing or has trailing integers")
    source_hash, envelope_start, envelope_end = values[cursor:cursor + 3]
    cursor += 3
    prefix = suffix = None
    if carrier:
        prefix, cursor = _read_codepoints(values, cursor, label="carrier prefix", allow_empty=True)
        suffix, cursor = _read_codepoints(values, cursor, label="carrier suffix", allow_empty=True)
    if cursor != len(values):
        raise RelationSurfaceArtifactError("frame has trailing integers")
    try:
        proposition = ObjectIdentity.from_stable_key(proposition_key)
        predicate = ObjectIdentity.from_stable_key(predicate_key)
    except (TypeError, ValueError) as error:
        raise RelationSurfaceArtifactError("frame identity is invalid") from error
    return RelationSurfaceFrame(
        proposition, predicate, tuple(roles), tuple(gaps), source_hash,
        envelope_start, envelope_end, prefix, suffix)


def _header(core_sha256: str, fact_count: int, frame_count: int, *, schema_version: int = ARTIFACT_SCHEMA_VERSION) -> tuple[int, ...]:
    """完整版本属于语言组件身份，重绑定不能回退为包含来源句的旧协议。"""
    if type(schema_version) is not int or schema_version not in _SUPPORTED_SCHEMAS:
        raise RelationSurfaceArtifactError("surface artifact schema is unsupported")
    if type(core_sha256) is not str or len(core_sha256) != 64:
        raise RelationSurfaceArtifactError("Core SHA-256 is invalid")
    try:
        digest = bytes.fromhex(core_sha256)
    except ValueError as error:
        raise RelationSurfaceArtifactError("Core SHA-256 is not hexadecimal") from error
    return (schema_version, len(digest), *digest, fact_count, frame_count)


def _parse_header(values: tuple[int, ...]) -> tuple[str, int, int]:
    if len(values) < 4 or values[0] not in _SUPPORTED_SCHEMAS:
        raise RelationSurfaceArtifactError("surface artifact schema is unsupported")
    digest_size = values[1]
    if digest_size != 32 or len(values) != 2 + digest_size + 2:
        raise RelationSurfaceArtifactError("surface artifact header is malformed")
    digest = values[2:2 + digest_size]
    if any(type(item) is not int or not 0 <= item <= 255 for item in digest):
        raise RelationSurfaceArtifactError("surface artifact digest is malformed")
    fact_count, frame_count = values[-2:]
    if (type(fact_count) is not int or fact_count <= 0
            or type(frame_count) is not int or frame_count <= 0):
        raise RelationSurfaceArtifactError("surface artifact counts are invalid")
    return bytes(digest).hex(), fact_count, frame_count


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        "CREATE TABLE artifact_meta ("
        "key_id INTEGER PRIMARY KEY, payload BLOB NOT NULL);"
        "CREATE TABLE relation_surface_row ("
        "row_kind INTEGER NOT NULL, ordinal INTEGER NOT NULL, "
        "payload BLOB NOT NULL, PRIMARY KEY(row_kind, ordinal));"
    )


def materialize_relation_surface_artifact(
        core_database: str | Path,
        output_database: str | Path,
        *,
        binding_database: str | Path | None = None,
        ) -> dict[str, object]:
    """Build one fresh integer artifact from a source retaining raw text.

    ``binding_database`` is used by a release builder that first creates a
    text-free physical Core copy.  Facts are read from ``core_database``;
    the published artifact is bound to that final copy.
    """
    core = Path(core_database).resolve(strict=True)
    output = Path(output_database).resolve()
    if output.exists():
        raise FileExistsError(f"surface artifact exists; refusing overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    binding = (core if binding_database is None
               else Path(binding_database).resolve(strict=True))
    core_sha256 = sha256_path(binding)
    with TrainedRelationGraphRuntime(core) as runtime:
        facts = runtime.active_surface_facts()
        frames = runtime.active_surface_frames()
    if not facts or not frames:
        raise RelationSurfaceArtifactError("Core has no active relation surfaces/frames")
    connection = sqlite3.connect(output)
    try:
        _create_schema(connection)
        connection.execute(
            "INSERT INTO artifact_meta(key_id, payload) VALUES (?, ?)",
            (1, encode_integer_tuple(_header(core_sha256, len(facts), len(frames)))),
        )
        for ordinal, fact in enumerate(facts):
            connection.execute(
                "INSERT INTO relation_surface_row(row_kind, ordinal, payload) VALUES (?, ?, ?)",
                (_FACT_ROW, ordinal, encode_integer_tuple(_pack_fact(fact))),
            )
        for ordinal, frame in enumerate(frames):
            connection.execute(
                "INSERT INTO relation_surface_row(row_kind, ordinal, payload) VALUES (?, ?, ?)",
                (_FRAME_ROW, ordinal, encode_integer_tuple(_pack_frame(frame))),
            )
        connection.commit()
    finally:
        connection.close()
    read_relation_surface_artifact(output, expected_core_sha256=core_sha256)
    return {
        "format": ARTIFACT_FORMAT,
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "core_sha256": core_sha256,
        "fact_count": len(facts),
        "frame_count": len(frames),
        "database": str(output),
        "database_bytes": output.stat().st_size,
        "database_sha256": sha256_path(output),
    }


def read_relation_surface_artifact(
        database: str | Path,
        *,
        expected_core_sha256: str | None = None,
        ) -> tuple[tuple[ActiveRelationSurface, ...],
                   tuple[RelationSurfaceFrame, ...], dict[str, object]]:
    """Read and fully validate the immutable integer surface artifact."""
    path = Path(database).resolve(strict=True)
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        if integrity != ("ok",):
            raise RelationSurfaceArtifactError("surface artifact SQLite integrity failed")
        meta_rows = connection.execute(
            "SELECT key_id, payload FROM artifact_meta ORDER BY key_id").fetchall()
        if len(meta_rows) != 1 or meta_rows[0][0] != 1:
            raise RelationSurfaceArtifactError("surface artifact metadata is not unique")
        try:
            header = decode_integer_tuple(bytes(meta_rows[0][1]))
        except (TypeError, ValueError) as error:
            raise RelationSurfaceArtifactError("surface artifact metadata is not canonical") from error
        core_sha256, fact_count, frame_count = _parse_header(header)
        if expected_core_sha256 is not None and core_sha256 != expected_core_sha256:
            raise RelationSurfaceArtifactError("surface artifact is bound to another Core")
        rows = connection.execute(
            "SELECT row_kind, ordinal, payload FROM relation_surface_row "
            "ORDER BY row_kind, ordinal").fetchall()
    except sqlite3.DatabaseError as error:
        raise RelationSurfaceArtifactError("surface artifact SQLite schema is invalid") from error
    finally:
        connection.close()
    expected = [(_FACT_ROW, ordinal) for ordinal in range(fact_count)]
    expected.extend((_FRAME_ROW, ordinal) for ordinal in range(frame_count))
    actual = [(row[0], row[1]) for row in rows]
    if actual != expected:
        raise RelationSurfaceArtifactError("surface artifact row ordinals are not closed")
    facts = []
    frames = []
    for row_kind, _ordinal, payload in rows:
        try:
            values = decode_integer_tuple(bytes(payload))
        except (TypeError, ValueError) as error:
            raise RelationSurfaceArtifactError("surface artifact row is not canonical") from error
        if row_kind == _FACT_ROW:
            facts.append(_unpack_fact(values, schema_version=header[0]))
        elif row_kind == _FRAME_ROW:
            frames.append(_unpack_frame(values, schema_version=header[0]))
        else:
            raise RelationSurfaceArtifactError("surface artifact row kind is invalid")
    fact_keys = {item.proposition.stable_key() for item in facts}
    frame_keys = {item.proposition.stable_key() for item in frames}
    if len(fact_keys) != len(facts) or frame_keys != fact_keys:
        raise RelationSurfaceArtifactError("surface artifact fact/frame identities differ")
    if header[0] != ARTIFACT_SCHEMA_VERSION:
        validate_language_members(tuple(facts), tuple(frames),
            require_carrier=header[0] == CARRIER_LANGUAGE_ARTIFACT_SCHEMA_VERSION)
    return (tuple(facts), tuple(frames), {
        "format": ARTIFACT_FORMAT,
        "schema_version": header[0],
        "core_sha256": core_sha256,
        "fact_count": fact_count,
        "frame_count": frame_count,
        "database_sha256": sha256_path(path),
    })


def validate_language_members(facts, frames, *, require_carrier=False):
    """无正文成员须保留完整来源角色；V3另核验载体与每段位置闭合。"""
    by_key = {frame.proposition.stable_key(): frame for frame in frames}
    if len(by_key) != len(frames) or len(facts) != len(frames):
        raise RelationSurfaceArtifactError("language frame membership differs")
    for fact in facts:
        frame = by_key.get(fact.proposition.stable_key())
        if (fact.evidence_surface or frame is None or frame.predicate != fact.predicate
                or frame.source_hash != fact.source_hash
                or frame.roles != tuple(binding.role for binding in fact.bindings)
                or len(frame.gaps) != len(frame.roles) + 1
                or any(binding.source_hash != fact.source_hash for binding in fact.bindings)
                or any(binding.surface and binding.surface in gap
                       for binding in fact.bindings for gap in frame.gaps)):
            raise RelationSurfaceArtifactError("language members contain a source body or exemplar literal")
        if require_carrier:
            from pure_integer_ai.experiments.relation_language_carrier import validate_carrier_frame
            validate_carrier_frame(fact, frame)


def _artifact_row_payload_sha256(database: str | Path) -> str:
    """Hash canonical row coordinates and exact integer payload bytes."""
    path = Path(database).resolve(strict=True)
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    digest = hashlib.sha256()
    try:
        rows = connection.execute(
            "SELECT row_kind, ordinal, payload FROM relation_surface_row "
            "ORDER BY row_kind, ordinal")
        for row_kind, ordinal, payload in rows:
            payload = bytes(payload)
            digest.update(encode_integer_tuple((row_kind, ordinal, len(payload))))
            digest.update(payload)
    except sqlite3.DatabaseError as error:
        raise RelationSurfaceArtifactError(
            "surface artifact rows could not be hashed") from error
    finally:
        connection.close()
    return digest.hexdigest()


def rebind_relation_surface_artifact(
        source_core_database: str | Path,
        source_artifact_database: str | Path,
        target_core_database: str | Path,
        output_database: str | Path,
        ) -> dict[str, object]:
    """Copy an integer artifact to a new Core identity without changing rows.

    Both Core files are opened through ``TrainedRelationGraphRuntime`` so the
    source artifact and the rebound artifact must each match the corresponding
    active proposition set.  The output is committed only after both checks.
    """
    source_core = Path(source_core_database).resolve(strict=True)
    source_artifact = Path(source_artifact_database).resolve(strict=True)
    target_core = Path(target_core_database).resolve(strict=True)
    output = Path(output_database).resolve()
    if output.exists():
        raise FileExistsError(f"surface artifact exists; refusing overwrite: {output}")
    source_sha256 = sha256_path(source_core)
    target_sha256 = sha256_path(target_core)
    if source_sha256 == target_sha256:
        raise RelationSurfaceArtifactError(
            "surface artifact rebind requires a different target Core identity")

    facts, frames, source_metadata = read_relation_surface_artifact(
        source_artifact, expected_core_sha256=source_sha256)
    source_keys = {item.proposition.stable_key() for item in facts}
    source_frame_keys = {item.proposition.stable_key() for item in frames}
    with TrainedRelationGraphRuntime(
            source_core,
            surface_artifact_database=source_artifact) as runtime:
        source_active_keys = {
            item.definition.proposition.stable_key()
            for item in runtime.active_propositions()
        }
    if source_active_keys != source_keys or source_frame_keys != source_keys:
        raise RelationSurfaceArtifactError(
            "source artifact does not cover the source active Core")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
            prefix=f".{output.name}.", suffix=".tmp", dir=output.parent,
            delete=False) as temporary_file:
        temporary = Path(temporary_file.name)
    source_rows_sha256 = _artifact_row_payload_sha256(source_artifact)
    try:
        connection = sqlite3.connect(temporary)
        try:
            _create_schema(connection)
            connection.execute(
                "INSERT INTO artifact_meta(key_id, payload) VALUES (?, ?)",
                (1, encode_integer_tuple(
                    _header(target_sha256, len(facts), len(frames),
                        schema_version=source_metadata["schema_version"]))),
            )
            for ordinal, fact in enumerate(facts):
                connection.execute(
                    "INSERT INTO relation_surface_row "
                    "(row_kind, ordinal, payload) VALUES (?, ?, ?)",
                    (_FACT_ROW, ordinal, encode_integer_tuple(_pack_fact(fact,
                        schema_version=source_metadata["schema_version"]))),
                )
            for ordinal, frame in enumerate(frames):
                connection.execute(
                    "INSERT INTO relation_surface_row "
                    "(row_kind, ordinal, payload) VALUES (?, ?, ?)",
                    (_FRAME_ROW, ordinal, encode_integer_tuple(_pack_frame(frame,
                        schema_version=source_metadata["schema_version"]))),
                )
            connection.commit()
        finally:
            connection.close()

        copied_facts, copied_frames, target_metadata = read_relation_surface_artifact(
            temporary, expected_core_sha256=target_sha256)
        target_rows_sha256 = _artifact_row_payload_sha256(temporary)
        if (copied_facts != facts or copied_frames != frames
                or target_rows_sha256 != source_rows_sha256):
            raise RelationSurfaceArtifactError(
                "surface artifact row payloads changed during rebind")

        with TrainedRelationGraphRuntime(
                target_core,
                surface_artifact_database=temporary) as runtime:
            target_active_keys = {
                item.definition.proposition.stable_key()
                for item in runtime.active_propositions()
            }
        if target_active_keys != source_keys:
            raise RelationSurfaceArtifactError(
                "rebound artifact does not cover the target active Core")

        temporary.rename(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    return {
        "format": ARTIFACT_FORMAT,
        "source_core_sha256": source_sha256,
        "source_artifact_sha256": source_metadata["database_sha256"],
        "target_core_sha256": target_sha256,
        "target_artifact_sha256": sha256_path(output),
        "record_payload_sha256": source_rows_sha256,
        "fact_count": len(facts),
        "frame_count": len(frames),
        "active_proposition_count": len(target_active_keys),
        "database": str(output),
        "database_bytes": output.stat().st_size,
        "source_metadata": source_metadata,
        "target_metadata": target_metadata,
    }


__all__ = [
    "ARTIFACT_FORMAT",
    "ARTIFACT_SCHEMA_VERSION",
    "RelationSurfaceArtifactError",
    "materialize_relation_surface_artifact",
    "rebind_relation_surface_artifact",
    "read_relation_surface_artifact",
    "sha256_path",
]
