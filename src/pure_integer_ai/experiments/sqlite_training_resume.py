"""Bound SQLite page-copy resume path for large incremental training runs.

The portable JSON recovery package remains the cross-language authority.  This
module adds an explicitly published SQLite checkpoint for the current Python
host so a completed database can be copied by pages instead of reparsing and
reinserting billions of JSON fields.  Every use rebinds the source database,
summary, both cursors, portable recovery manifest, schema, and table counts.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
from typing import Any

from pure_integer_ai.storage.backend import StorageBackend
from pure_integer_ai.storage.k_run_boundary import (
    open_existing_run_root,
    open_plain_binary,
    write_exclusive_bytes,
)


SQLITE_RESUME_ARTIFACT_KIND = "SQLITE_TRAINING_RESUME_V1"
SQLITE_RESUME_MANIFEST = "sqlite_resume_manifest.json"


# object-model: exception; interop=sqlite-training-resume-v1
class SQLiteTrainingResumeError(ValueError):
    """SQLite resume publication, copy, or binding validation failed."""


# object-model: value; representation=struct; interop=sqlite-training-resume-v1
@dataclass(frozen=True, slots=True)
class SQLiteTrainingResumeBinding:
    """Validated immutable identity of one completed SQLite checkpoint."""

    run_id: str
    manifest_sha256: str
    database_sha256: str
    database_bytes: int
    page_count: int
    page_size: int
    schema_sha256: str
    table_counts_sha256: str
    table_counts: tuple[tuple[str, int], ...]
    cursor_payload: dict[str, Any]


def _canonical_json_line(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SQLiteTrainingResumeError(f"{label} cannot be read") from error
    if not isinstance(value, dict):
        raise SQLiteTrainingResumeError(f"{label} must be an object")
    return value


def _quoted(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _database_fingerprint(
        path: Path,
        ) -> tuple[int, int, str, str, tuple[tuple[str, int], ...]]:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        quick = connection.execute("PRAGMA quick_check").fetchone()
        if quick != ("ok",):
            raise SQLiteTrainingResumeError("SQLite quick_check failed")
        page_count = int(connection.execute(
            "PRAGMA page_count").fetchone()[0])
        page_size = int(connection.execute(
            "PRAGMA page_size").fetchone()[0])
        schema_rows = tuple(
            (str(row[0]), str(row[1]), "" if row[2] is None else str(row[2]))
            for row in connection.execute(
                "SELECT type,name,sql FROM sqlite_master "
                "WHERE name NOT LIKE 'sqlite_%' ORDER BY type,name"))
        table_names = tuple(row[1] for row in schema_rows if row[0] == "table")
        table_counts = tuple(
            (name, int(connection.execute(
                f"SELECT COUNT(*) FROM {_quoted(name)}").fetchone()[0]))
            for name in table_names)
    finally:
        connection.close()
    schema_payload = _canonical_json_line(schema_rows)
    counts_payload = _canonical_json_line(table_counts)
    return (
        page_count,
        page_size,
        hashlib.sha256(schema_payload).hexdigest(),
        hashlib.sha256(counts_payload).hexdigest(),
        table_counts,
    )


def _run_files(root: Path) -> dict[str, Path]:
    result = {
        "cursor": root / "cursor.json",
        "database": root / "training.sqlite3",
        "portable_manifest": root / "run.manifest.json",
        "portable_seal": root / "run.manifest.sha256",
        "summary": root / "training_summary.json",
        "training_cursor": root / "training_cursor.int",
    }
    if any(not path.is_file() for path in result.values()):
        raise SQLiteTrainingResumeError(
            "completed training run is missing SQLite or recovery files")
    return result


def _validate_summary(root: Path, files: dict[str, Path]) -> tuple[str, dict[str, Any]]:
    summary = _read_json(files["summary"], label="training summary")
    run_id = summary.get("run_id")
    stages = summary.get("stages_completed")
    if (not isinstance(run_id, str) or run_id != root.name
            or not isinstance(stages, list) or not stages
            or summary.get("database") is None):
        raise SQLiteTrainingResumeError("training summary is not completed")
    database = Path(str(summary["database"]))
    if not database.is_absolute():
        database = root / database
    if database.resolve() != files["database"].resolve():
        raise SQLiteTrainingResumeError("training summary database drifted")
    return run_id, summary


def _portable_seal(files: dict[str, Path]) -> str:
    try:
        line = files["portable_seal"].read_text(encoding="ascii").strip()
    except (OSError, UnicodeDecodeError) as error:
        raise SQLiteTrainingResumeError("portable recovery seal is unreadable") from error
    actual = _sha256_file(files["portable_manifest"])
    if line != actual:
        raise SQLiteTrainingResumeError("portable recovery seal drifted")
    return actual


def publish_sqlite_training_resume(
        run_root: str | Path, *, require_k_drive: bool = True,
        ) -> SQLiteTrainingResumeBinding:
    """Publish one non-overwritable binding after a run fully completes."""
    capability = open_existing_run_root(
        run_root, require_k_drive=require_k_drive,
        label="SQLite training resume source")
    root = capability.path
    existing = root / SQLITE_RESUME_MANIFEST
    if existing.is_file():
        return load_sqlite_training_resume(
            root, require_k_drive=require_k_drive)
    files = _run_files(root)
    run_id, summary = _validate_summary(root, files)
    portable_manifest_sha = _portable_seal(files)
    cursor_payload = _read_json(files["cursor"], label="portable cursor")
    page_count, page_size, schema_sha, counts_sha, counts = (
        _database_fingerprint(files["database"]))
    manifest = {
        "artifact_kind": SQLITE_RESUME_ARTIFACT_KIND,
        "database_bytes": files["database"].stat().st_size,
        "database_sha256": _sha256_file(files["database"]),
        "file_sha256": {
            "cursor.json": _sha256_file(files["cursor"]),
            "run.manifest.json": portable_manifest_sha,
            "run.manifest.sha256": _sha256_file(files["portable_seal"]),
            "training_cursor.int": _sha256_file(files["training_cursor"]),
            "training_summary.json": _sha256_file(files["summary"]),
        },
        "pack_sha256": summary.get("pack_sha256"),
        "page_count": page_count,
        "page_size": page_size,
        "run_id": run_id,
        "schema_sha256": schema_sha,
        "schema_version": 1,
        "status": "PASS",
        "table_counts": [list(item) for item in counts],
        "table_counts_sha256": counts_sha,
    }
    payload = _canonical_json_line(manifest)
    write_exclusive_bytes(
        capability, SQLITE_RESUME_MANIFEST, payload,
        label="SQLite training resume manifest")
    return load_sqlite_training_resume(
        root, expected_manifest_sha256=hashlib.sha256(payload).hexdigest(),
        require_k_drive=require_k_drive)


def load_sqlite_training_resume(
        run_root: str | Path, *, expected_manifest_sha256: str | None = None,
        require_k_drive: bool = True,
        ) -> SQLiteTrainingResumeBinding:
    """Validate source files and database against the closed binding."""
    capability = open_existing_run_root(
        run_root, require_k_drive=require_k_drive,
        label="SQLite training resume source")
    root = capability.path
    with open_plain_binary(
            capability, SQLITE_RESUME_MANIFEST,
            label="SQLite training resume manifest") as stream:
        payload = stream.read()
    manifest_sha = hashlib.sha256(payload).hexdigest()
    if (expected_manifest_sha256 is not None
            and manifest_sha != expected_manifest_sha256):
        raise SQLiteTrainingResumeError("SQLite resume manifest SHA drifted")
    try:
        manifest = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SQLiteTrainingResumeError("SQLite resume manifest is invalid") from error
    if (not isinstance(manifest, dict)
            or _canonical_json_line(manifest) != payload
            or manifest.get("artifact_kind") != SQLITE_RESUME_ARTIFACT_KIND
            or manifest.get("schema_version") != 1
            or manifest.get("status") != "PASS"):
        raise SQLiteTrainingResumeError("SQLite resume manifest drifted")
    files = _run_files(root)
    run_id, _summary = _validate_summary(root, files)
    if manifest.get("run_id") != run_id:
        raise SQLiteTrainingResumeError("SQLite resume run_id drifted")
    expected_files = manifest.get("file_sha256")
    if not isinstance(expected_files, dict):
        raise SQLiteTrainingResumeError("SQLite resume file inventory drifted")
    file_map = {
        "cursor.json": files["cursor"],
        "run.manifest.json": files["portable_manifest"],
        "run.manifest.sha256": files["portable_seal"],
        "training_cursor.int": files["training_cursor"],
        "training_summary.json": files["summary"],
    }
    if set(expected_files) != set(file_map):
        raise SQLiteTrainingResumeError("SQLite resume file inventory drifted")
    for name, path in file_map.items():
        if expected_files[name] != _sha256_file(path):
            raise SQLiteTrainingResumeError(
                f"SQLite resume source {name} drifted")
    _portable_seal(files)
    database_sha = _sha256_file(files["database"])
    if (manifest.get("database_sha256") != database_sha
            or manifest.get("database_bytes") != files["database"].stat().st_size):
        raise SQLiteTrainingResumeError("SQLite resume database drifted")
    page_count, page_size, schema_sha, counts_sha, counts = (
        _database_fingerprint(files["database"]))
    if (manifest.get("page_count") != page_count
            or manifest.get("page_size") != page_size
            or manifest.get("schema_sha256") != schema_sha
            or manifest.get("table_counts_sha256") != counts_sha
            or manifest.get("table_counts") != [list(item) for item in counts]):
        raise SQLiteTrainingResumeError("SQLite resume logical fingerprint drifted")
    return SQLiteTrainingResumeBinding(
        run_id, manifest_sha, database_sha,
        int(manifest["database_bytes"]), page_count, page_size,
        schema_sha, counts_sha, counts,
        _read_json(files["cursor"], label="portable cursor"),
    )


def prepare_sqlite_page_resume(
        source_run_root: str | Path, destination_database: str | Path, *,
        require_k_drive: bool = True,
        ) -> SQLiteTrainingResumeBinding:
    """Validate a source checkpoint and page-copy it to a new database."""
    source_root = Path(source_run_root).resolve()
    destination = Path(destination_database).resolve()
    if destination.exists() or not destination.parent.is_dir():
        raise SQLiteTrainingResumeError(
            "SQLite resume destination must be new under an existing run")
    if (require_k_drive and (source_root.drive.upper() != "K:"
                            or destination.drive.upper() != "K:")):
        raise SQLiteTrainingResumeError("SQLite page resume must stay on K drive")
    binding = load_sqlite_training_resume(
        source_root, require_k_drive=require_k_drive)
    source_database = source_root / "training.sqlite3"
    # The source binding is immutable and was already fingerprinted above.
    # Use a byte-for-byte staged copy instead of sqlite3.Connection.backup:
    # on multi-gigabyte databases the latter can be interrupted after writing
    # a partial file, leaving a destination that opens but reports malformed
    # pages.  Atomic replace keeps a failed copy from becoming a checkpoint.
    fd, partial_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".partial",
        dir=str(destination.parent),
    )
    os.close(fd)
    partial = Path(partial_name)
    try:
        shutil.copyfile(source_database, partial)
        os.replace(partial, destination)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    page_count, page_size, schema_sha, counts_sha, counts = (
        _database_fingerprint(destination))
    if (page_count != binding.page_count or page_size != binding.page_size
            or schema_sha != binding.schema_sha256
            or counts_sha != binding.table_counts_sha256
            or counts != binding.table_counts):
        raise SQLiteTrainingResumeError(
            "SQLite page resume destination fingerprint drifted")
    return binding


def validate_preloaded_sqlite_resume(
        backend: StorageBackend, source_run_root: str | Path, *,
        expected_manifest_sha256: str,
        require_k_drive: bool = True,
        ) -> dict[str, Any] | None:
    """Verify registered target tables/counts before formal resume consumes cursor."""
    binding = load_sqlite_training_resume(
        source_run_root,
        expected_manifest_sha256=expected_manifest_sha256,
        require_k_drive=require_k_drive,
    )
    schema = backend.schema_snapshot()
    expected_tables = dict(binding.table_counts)
    if set(schema) != set(expected_tables):
        raise SQLiteTrainingResumeError(
            "preloaded SQLite registered schema differs from checkpoint")
    actual_counts = tuple(
        (table, int(backend.count(table, where=None)))
        for table in sorted(schema))
    if actual_counts != binding.table_counts:
        raise SQLiteTrainingResumeError(
            "preloaded SQLite table counts differ from checkpoint")
    # id_pool is deliberately runtime-only and therefore is not copied by
    # SQLite. Rebase it from bounded MAX queries before any new object can be
    # materialized; this mirrors portable recovery's floor advancement without
    # expanding large tables into Python rows.
    spaces = ()
    if "space" in schema:
        spaces = tuple(sorted(
            int(row["space_id"])
            for row in backend.select("space", where=None)
            if type(row.get("space_id")) is int and row["space_id"] > 0))
    floors: dict[int, int] = {}
    for table, meta in sorted(schema.items()):
        columns = tuple(meta.get("columns", ()))
        for local_column in columns:
            if "local_id" not in local_column:
                continue
            space_column = local_column.replace("local_id", "space_id")
            if space_column not in columns:
                continue
            for space_id in spaces:
                rows = backend.select(
                    table,
                    where={space_column: space_id},
                    order_by=local_column,
                    descending=True,
                    limit=1,
                )
                if not rows:
                    continue
                local_id = rows[0].get(local_column)
                if type(local_id) is int and local_id > floors.get(space_id, 0):
                    floors[space_id] = local_id
    for space_id, floor in sorted(floors.items()):
        backend.advance_id_pool(space_id, floor)
    return binding.cursor_payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="publish SQLite training resume binding")
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args(argv)
    binding = publish_sqlite_training_resume(args.run_root)
    print(json.dumps({
        "database_bytes": binding.database_bytes,
        "database_sha256": binding.database_sha256,
        "manifest_sha256": binding.manifest_sha256,
        "run_id": binding.run_id,
        "status": "PASS",
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SQLITE_RESUME_MANIFEST", "SQLiteTrainingResumeBinding",
    "SQLiteTrainingResumeError", "load_sqlite_training_resume",
    "prepare_sqlite_page_resume", "publish_sqlite_training_resume",
    "validate_preloaded_sqlite_resume",
]
