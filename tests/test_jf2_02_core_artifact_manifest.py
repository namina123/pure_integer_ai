"""JF2-02 Core/artifact manifest 的 canonical、身份和发布边界测试。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments.j_f2_core_artifact_manifest import (
    ARTIFACT_KIND,
    MANIFEST_PATH,
    CoreArtifactManifestError,
    build_core_artifact_manifest,
    publish_core_artifact_manifest,
    read_core_artifact_manifest,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes


ROOT = Path(__file__).resolve().parents[1]


def test_build_has_required_public_closure() -> None:
    """真实公开树应形成完整角色覆盖且排除 J-F2 施工文件。"""
    manifest = build_core_artifact_manifest(ROOT)
    assert manifest.artifact_kind == ARTIFACT_KIND
    assert len(manifest.file_bindings) > 1000
    roles = {role for item in manifest.file_bindings for role in item.roles}
    assert {
        "BACKEND_CAPABILITY", "CORE_IMPLEMENTATION", "D03_PUBLICATION",
        "LC16", "PRIMITIVE_IMPLEMENTATION", "REQUIRED_REPORT",
        "SCHEMA_COURSE", "SEGMENT_LOCATION_RECOVERY", "W_RECEIPTS",
    } <= roles
    assert all("ph2_j_f2_contract.py" not in item.relative_path
               for item in manifest.file_bindings)


def test_publish_readback_and_append_only(tmp_path: Path) -> None:
    """临时发布路径必须逐字节回读，第二次发布必须拒绝覆盖。"""
    target = tmp_path / "j_f2_core_artifact_manifest_v1.json"
    published = publish_core_artifact_manifest(ROOT, target=target)
    restored = read_core_artifact_manifest(ROOT, target)
    assert restored == published
    with pytest.raises(CoreArtifactManifestError):
        publish_core_artifact_manifest(ROOT, target=target)


def test_reader_rejects_file_identity_drift(tmp_path: Path) -> None:
    """manifest 中的文件摘要被改写时，读回必须 fail closed。"""
    target = tmp_path / "manifest.json"
    manifest = publish_core_artifact_manifest(ROOT, target=target)
    value = json.loads(target.read_text(encoding="utf-8"))
    value["file_bindings"][0]["sha256"] = "0" * 64
    target.write_bytes(canonical_json_bytes(value) + b"\n")
    with pytest.raises(CoreArtifactManifestError):
        read_core_artifact_manifest(ROOT, target)
    value = manifest.to_dict()
    value["file_bindings"] = []
    target.write_bytes(canonical_json_bytes(value) + b"\n")
    with pytest.raises(CoreArtifactManifestError):
        read_core_artifact_manifest(ROOT, target)


def test_reader_rejects_path_escape(tmp_path: Path) -> None:
    """相对路径越界不能借 manifest 进入仓库外文件。"""
    target = tmp_path / "manifest.json"
    publish_core_artifact_manifest(ROOT, target=target)
    value = json.loads(target.read_text(encoding="utf-8"))
    value["file_bindings"][0]["relative_path"] = "../outside.json"
    target.write_bytes(canonical_json_bytes(value) + b"\n")
    with pytest.raises(CoreArtifactManifestError):
        read_core_artifact_manifest(ROOT, target)
