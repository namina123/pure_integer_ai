"""Public release closure and nested-manifest boundary checks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments.public_model_release import (
    PublicModelReleaseError,
    load_public_model_release,
)


def _canonical(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf-8")


def _write_fixture(root: Path) -> None:
    files = {
        "knowledge/broad_qa.sqlite3": b"qa",
        "model/training.sqlite3": b"training",
        "model/training_cursor.int": b"1\n",
        "model/training_summary.json": {
            "database": "training.sqlite3",
            "training_cursor": "training_cursor.int",
        },
        "model/dialogue_pack_manifest.json": {
            "source_files": [["data/course.jsonl", "00", 1]],
            "extra_course_paths": ["data/course.jsonl"],
            "surface_evidence_files": [["data/evidence.jsonl", "00"]],
            "private_formal_data": "FORBIDDEN",
        },
        "model/dialogue_protocol.json": {
            "format": "PURE_INTEGER_AI_DIALOGUE_PROTOCOL_CONFIG",
            "schema_version": 1,
            "transport": "jsonl",
            "encoding": "utf-8",
            "operations": ["turn", "quit", "exit"],
        },
        "data/course.jsonl": b"{}\n",
        "data/evidence.jsonl": b"{}\n",
        "data/ph2/sparse.json": {"boundary_flags": []},
    }
    for relative, value in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value if isinstance(value, bytes) else _canonical(value))
    source_manifest = {
        "format": "PUBLIC_SOURCE_MANIFEST_V1",
        "schema_version": 1,
        "qa_index": {"path": "knowledge/broad_qa.sqlite3"},
        "training": {
            "pack_manifest": "model/dialogue_pack_manifest.json",
            "course_files": ["data/course.jsonl"],
            "course_sidecars": [],
            "surface_courses": ["data/evidence.jsonl"],
        },
        "sparse_runtime": {
            "snapshot": "data/ph2/sparse.json",
            "source_files": [],
        },
    }
    (root / "source_manifest.json").write_bytes(_canonical(source_manifest))
    payload = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in {"public_model_release.json", "public_model_release.sha256"}:
            continue
        payload.append({
            "path": relative,
            "role": "training_state",
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    manifest = {
        "format": "PURE_INTEGER_AI_PUBLIC_MODEL_RELEASE",
        "schema_version": 1,
        "release_id": "fixture",
        "source_manifest": "source_manifest.json",
        "entry": {
            "qa_database": "knowledge/broad_qa.sqlite3",
            "training_root": "model",
            "sparse_snapshot": "data/ph2/sparse.json",
            "protocol_config": "model/dialogue_protocol.json",
            "protocol": "jsonl",
        },
        "files": payload,
    }
    manifest_path = root / "public_model_release.json"
    manifest_path.write_bytes(_canonical(manifest))
    (root / "public_model_release.sha256").write_text(
        hashlib.sha256(manifest_path.read_bytes()).hexdigest() + "\n",
        encoding="ascii")


def _refresh_manifest(root: Path) -> None:
    path = root / "public_model_release.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    entries = {item["path"]: item for item in value["files"]}
    for relative, item in entries.items():
        payload = root / relative
        item["size_bytes"] = payload.stat().st_size
        item["sha256"] = hashlib.sha256(payload.read_bytes()).hexdigest()
    path.write_bytes(_canonical(value))
    (root / "public_model_release.sha256").write_text(
        hashlib.sha256(path.read_bytes()).hexdigest() + "\n", encoding="ascii")


def test_release_manifest_is_closed_and_allows_forbidden_boundary(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    loaded = load_public_model_release(tmp_path, require_k_drive=False)
    assert loaded.release_id == "fixture"

    (tmp_path / "unexpected.bin").write_bytes(b"not listed")
    with pytest.raises(PublicModelReleaseError, match="文件集合不闭合"):
        load_public_model_release(tmp_path, require_k_drive=False)


def test_release_rejects_absolute_nested_manifest_path(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    nested = tmp_path / "model/dialogue_pack_manifest.json"
    value = json.loads(nested.read_text(encoding="utf-8"))
    value["extra_course_paths"] = ["C:/outside/course.jsonl"]
    nested.write_bytes(_canonical(value))
    _refresh_manifest(tmp_path)
    with pytest.raises(PublicModelReleaseError, match="越出 release root"):
        load_public_model_release(tmp_path, require_k_drive=False)


def test_release_rejects_absolute_source_identity(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    nested = tmp_path / "model/dialogue_pack_manifest.json"
    value = json.loads(nested.read_text(encoding="utf-8"))
    value["source_identities"] = [[
        "data/course.jsonl", "D:/host/course.jsonl"]]
    nested.write_bytes(_canonical(value))
    _refresh_manifest(tmp_path)
    with pytest.raises(PublicModelReleaseError, match="本机绝对路径"):
        load_public_model_release(tmp_path, require_k_drive=False)


def test_release_rejects_private_declaration_payload(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    nested = tmp_path / "model/dialogue_pack_manifest.json"
    value = json.loads(nested.read_text(encoding="utf-8"))
    value["private_eval"] = {"label": "secret"}
    nested.write_bytes(_canonical(value))
    _refresh_manifest(tmp_path)
    with pytest.raises(PublicModelReleaseError, match="private evaluator 数据"):
        load_public_model_release(tmp_path, require_k_drive=False)


def test_release_fast_validation_is_explicit_and_keeps_closed_manifest(
        tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    payload = tmp_path / "knowledge/broad_qa.sqlite3"
    payload.write_bytes(b"QX")  # same size: only the payload digest changes
    loaded = load_public_model_release(
        tmp_path, require_k_drive=False, verify_payload_hashes=False)
    assert loaded.release_id == "fixture"
    with pytest.raises(PublicModelReleaseError, match="release file hash 漂移"):
        load_public_model_release(tmp_path, require_k_drive=False)


def test_release_hash_validation_flag_requires_strict_bool(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    with pytest.raises(TypeError, match="严格 bool"):
        load_public_model_release(
            tmp_path, require_k_drive=False, verify_payload_hashes=1)
