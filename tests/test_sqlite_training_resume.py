"""SQLite page-copy training resume binding regression."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.sqlite_training_resume import (
    load_sqlite_training_resume,
    prepare_sqlite_page_resume,
    publish_sqlite_training_resume,
    validate_preloaded_sqlite_resume,
)
from pure_integer_ai.storage.backend import SQLiteBackend, TYPE_INT
from pure_integer_ai.storage.discipline import DISC_NONE


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(
        value, sort_keys=True, separators=(",", ":")), encoding="utf-8")


def _completed_run(root: Path) -> None:
    root.mkdir()
    database = root / "training.sqlite3"
    backend = SQLiteBackend(str(database), performance_mode="bulk")
    backend.register_table(
        "sample", [("id", TYPE_INT), ("value", TYPE_INT)], DISC_NONE)
    backend.insert("sample", {"id": 1, "value": 7})
    backend.insert("sample", {"id": 2, "value": 9})
    backend.commit()
    backend.close()
    _json(root / "cursor.json", {
        "base_run_id": "base", "completed": [1, 2],
        "non_skippable": [], "run_id": root.name,
    })
    _json(root / "training_summary.json", {
        "database": str(database), "pack_sha256": "12" * 32,
        "run_id": root.name, "stages_completed": [1, 2],
    })
    (root / "training_cursor.int").write_bytes(b"training-cursor")
    manifest_payload = b'{"portable":1}\n'
    (root / "run.manifest.json").write_bytes(manifest_payload)
    (root / "run.manifest.sha256").write_text(
        hashlib.sha256(manifest_payload).hexdigest() + "\n",
        encoding="ascii")


def test_publish_copy_and_validate_sqlite_resume(tmp_path: Path) -> None:
    source = tmp_path / "base-run"
    _completed_run(source)
    binding = publish_sqlite_training_resume(
        source, require_k_drive=False)
    assert binding.table_counts == (("sample", 2),)
    assert load_sqlite_training_resume(
        source, expected_manifest_sha256=binding.manifest_sha256,
        require_k_drive=False) == binding

    destination_root = tmp_path / "next-run"
    destination_root.mkdir()
    destination = destination_root / "training.sqlite3"
    copied = prepare_sqlite_page_resume(
        source, destination, require_k_drive=False)
    assert copied.manifest_sha256 == binding.manifest_sha256

    backend = SQLiteBackend(str(destination), performance_mode="bulk")
    backend.register_table(
        "sample", [("id", TYPE_INT), ("value", TYPE_INT)], DISC_NONE)
    cursor = validate_preloaded_sqlite_resume(
        backend, source,
        expected_manifest_sha256=binding.manifest_sha256,
        require_k_drive=False,
    )
    assert cursor == {
        "base_run_id": "base", "completed": [1, 2],
        "non_skippable": [], "run_id": "base-run",
    }
    assert backend.select("sample", where=None) == [
        {"id": 1, "value": 7}, {"id": 2, "value": 9}]
    backend.close()
