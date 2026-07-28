"""从三进程十万级报告构建 R-02 存储生产吸收 artifact。"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_storage_absorption_contract import (
    ARTIFACT_STATUS,
    ARTIFACT_VERSION,
    COMPATIBILITY,
    EXECUTION_STATE,
    OBJECT_KIND_REGISTRY,
    StorageAbsorptionManifest,
    StorageEvidenceFile,
)


MANIFEST_PATH = Path("data/ph2/manifests/r02_storage_absorption_v1.json")
PROFILE_ROOT = "r02_storage_profile_artifacts/run_v1"
PROFILE_PATHS = {
    "build": f"{PROFILE_ROOT}/build.json",
    "query": f"{PROFILE_ROOT}/query.json",
    "audit": f"{PROFILE_ROOT}/audit.json",
    "state": f"{PROFILE_ROOT}/profile.sqlite3",
}
SOURCE_PATHS = (
    "src/pure_integer_ai/experiments/ph2_storage_absorption_catalog.py",
    "src/pure_integer_ai/experiments/ph2_storage_absorption_contract.py",
    "src/pure_integer_ai/storage/segment_repository.py",
    "src/pure_integer_ai/storage/segment_write_intent.py",
    "src/pure_integer_ai/storage/tiered_segment_store.py",
)
TEST_PATHS = (
    "tests/r02_storage_profile_worker.py",
    "tests/test_k02_sealed_segment_paging.py",
    "tests/test_k04_memory_hot_set.py",
    "tests/test_m10_memory_batch_recovery.py",
    "tests/test_r02_storage_absorption.py",
    "tests/test_v03_recovery_probe.py",
)


class StorageAbsorptionCatalogError(RuntimeError):
    """R-02 三阶段报告或文件 inventory 无法闭合。"""


def _path(root: Path, relative_path: str) -> Path:
    path = (root / Path(*relative_path.split("/"))).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise StorageAbsorptionCatalogError("R-02 catalog 路径逃逸") from error
    return path


def _read_report(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise StorageAbsorptionCatalogError("profile report newline 非法")
    try:
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
    except ValueError as error:
        raise StorageAbsorptionCatalogError("profile report JSON 损坏") from error
    if not isinstance(value, dict):
        raise StorageAbsorptionCatalogError("profile report 非 object")
    return value


def _identity(
        root_key: str,
        root: Path,
        relative_path: str,
        role: str,
        ) -> StorageEvidenceFile:
    path = _path(root, relative_path)
    if not path.is_file():
        raise StorageAbsorptionCatalogError(
            f"R-02 evidence 缺失: {relative_path}")
    payload = path.read_bytes()
    return StorageEvidenceFile(
        root_key, relative_path, role, len(payload),
        hashlib.sha256(payload).hexdigest())


def build_storage_absorption_manifest(
        repository_root: str | Path,
        workspace_root: str | Path,
        ) -> StorageAbsorptionManifest:
    """核三进程硬门并冻结实现、测试、报告和 SQLite 身份。"""
    repository = Path(repository_root).resolve()
    workspace = Path(workspace_root).resolve()
    if repository.parent != workspace:
        raise StorageAbsorptionCatalogError("repository/workspace 边界非法")
    build = _read_report(_path(workspace, PROFILE_PATHS["build"]))
    query = _read_report(_path(workspace, PROFILE_PATHS["query"]))
    audit = _read_report(_path(workspace, PROFILE_PATHS["audit"]))
    queries = query.get("queries")
    if (build.get("mode") != "BUILD" or query.get("mode") != "QUERY"
            or audit.get("mode") != "AUDIT"
            or not isinstance(queries, list) or len(queries) != 10
            or any(not isinstance(item, dict) for item in queries)):
        raise StorageAbsorptionCatalogError("profile 阶段或 query inventory 漂移")
    if any(
            item.get("segment_payload_gets") != 1
            or item.get("record_count") != 1
            or item.get("segment_payload_bytes", 0) > build.get(
                "max_segment_bytes", 0)
            for item in queries):
        raise StorageAbsorptionCatalogError("exact query 资源门失败")
    metrics = {
        "active_readers_after": max(
            query.get("active_readers_after", -1),
            audit.get("active_readers_after", -1)),
        "active_write_intents_after": max(
            build.get("active_write_intents", -1),
            query.get("active_write_intents", -1),
            audit.get("active_write_intents", -1)),
        "audit_content_matches_build": int(
            audit.get("content_sha256") == build.get("content_sha256")),
        "audit_max_page_records": audit.get("max_page_records"),
        "audit_record_count": audit.get("record_count"),
        "audit_segment_payload_bytes": audit.get(
            "audit_segment_payload_bytes"),
        "audit_segment_payload_gets": audit.get(
            "audit_segment_payload_gets"),
        "build_content_sha256": build.get("content_sha256"),
        "exact_query_count": len(queries),
        "exact_query_record_count": sum(
            item["record_count"] for item in queries),
        "exact_query_segment_payload_bytes": query.get(
            "query_segment_payload_bytes"),
        "exact_query_segment_payload_gets": query.get(
            "query_segment_payload_gets"),
        "manifest_entry_count": query.get("manifest_entry_count"),
        "max_segment_bytes": build.get("max_segment_bytes"),
        "record_count": build.get("record_count"),
        "records_per_segment": build.get("records_per_segment"),
        "segment_count": build.get("segment_count"),
        "startup_segment_payload_bytes": max(
            query.get("startup_segment_payload_bytes", -1),
            audit.get("startup_segment_payload_bytes", -1)),
        "startup_segment_payload_gets": max(
            query.get("startup_segment_payload_gets", -1),
            audit.get("startup_segment_payload_gets", -1)),
    }
    evidence = [
        *(_identity("REPOSITORY", repository, path, "SOURCE")
          for path in SOURCE_PATHS),
        *(_identity("REPOSITORY", repository, path, "TEST")
          for path in TEST_PATHS),
        *(_identity(
            "WORKSPACE", workspace, PROFILE_PATHS[key], "PROFILE_REPORT")
          for key in ("audit", "build", "query")),
        _identity(
            "WORKSPACE", workspace, PROFILE_PATHS["state"], "PROFILE_STATE"),
    ]
    return StorageAbsorptionManifest(
        1,
        ARTIFACT_VERSION,
        ARTIFACT_STATUS,
        CanonicalJsonObject.from_value(OBJECT_KIND_REGISTRY),
        CanonicalJsonObject.from_value(COMPATIBILITY),
        CanonicalJsonObject.from_value(metrics),
        tuple(evidence),
        CanonicalJsonObject.from_value(EXECUTION_STATE),
    )


__all__ = [
    "MANIFEST_PATH",
    "PROFILE_PATHS",
    "PROFILE_ROOT",
    "SOURCE_PATHS",
    "TEST_PATHS",
    "StorageAbsorptionCatalogError",
    "build_storage_absorption_manifest",
]
