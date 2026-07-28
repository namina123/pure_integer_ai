"""构建 P3-Ia 对四个 LC 账和能力基线的文件级修订。"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from pure_integer_ai.experiments.ph2_capability_course_contract import (
    CapabilityCourseManifest,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_p3ia_ledger_revision_contract import (
    ARTIFACT_KINDS,
    ARTIFACT_STATUSES,
    ARTIFACT_VERSIONS,
    EXECUTION_STATE,
    INVARIANTS,
    LEDGER_FACTS,
    LedgerEvidenceFile,
    P3IaLedgerRevision,
    P3IaLedgerRevisionError,
)


REVISION_PATHS = {
    "LC-00": Path("data/ph2/manifests/language_capability_baseline_v40.json"),
    "LC-07": Path("data/ph2/manifests/lc07_discourse_information_course_v2.json"),
    "LC-09": Path("data/ph2/manifests/lc09_transfer_axis_manifest_v2.json"),
    "LC-13": Path("data/ph2/manifests/lc13_directional_consumer_manifest_v2.json"),
    "LC-15": Path("data/ph2/manifests/lc15_final_learning_objectives_v2.json"),
}
SUPERSEDES_PATHS = {
    "LC-00": "data/ph2/manifests/language_capability_baseline_v39.json",
    "LC-07": "data/ph2/manifests/lc07_discourse_information_course_v1.json",
    "LC-09": "data/ph2/manifests/lc09_transfer_axis_manifest_v1.json",
    "LC-13": "data/ph2/manifests/lc13_directional_consumer_manifest_v1.json",
    "LC-15": "data/ph2/manifests/lc15_final_learning_objectives_v1.json",
}
P3IA_COURSE_PATH = (
    "data/ph2/manifests/p3ia_free_text_hierarchy_recall_course_v2.json")
P3IA_SAMPLE_PATH = "data/ph2/authored_free_text_hierarchy_recall_course_seed_v1.jsonl.sample"
P3IA_PACK_PATH = (
    "ph2_p3ia_dataset_artifacts/d02_language_courses_v1/packs/"
    "AUTHORED_CC0_V1--CC0-1.0--free-text-hierarchy-recall-v1/manifest.json")
P3IA_RUNTIME_PATHS = (
    "src/pure_integer_ai/experiments/free_text_hierarchy_runtime.py",
    "src/pure_integer_ai/experiments/free_text_recall_runtime.py",
    "src/pure_integer_ai/experiments/free_text_revision_runtime.py",
    "src/pure_integer_ai/experiments/ph2_authored_free_text_hierarchy_recall_course.py",
    "src/pure_integer_ai/experiments/ph2_free_text_hierarchy_recall_catalog.py",
    "src/pure_integer_ai/experiments/ph2_free_text_hierarchy_recall_contract.py",
)
P3IA_EVALUATOR_PATH = (
    "src/pure_integer_ai/experiments/free_text_hierarchy_recall_evaluator.py")
P3IA_TEST_PATH = "tests/test_d02_p3ia_free_text_hierarchy_recall_runtime.py"


class P3IaLedgerRevisionCatalogError(RuntimeError):
    """P3-Ia 课程、旧账或当前 inventory 无法形成修订。"""


def _path(root: Path, relative_path: str) -> Path:
    result = (root / Path(*relative_path.split("/"))).resolve()
    try:
        result.relative_to(root)
    except ValueError as error:
        raise P3IaLedgerRevisionCatalogError("catalog 路径逃逸") from error
    return result


def _identity(
        root_key: str,
        root: Path,
        relative_path: str,
        role: str,
        ) -> LedgerEvidenceFile:
    path = _path(root, relative_path)
    if not path.is_file():
        raise P3IaLedgerRevisionCatalogError(
            f"P3-Ia evidence 文件缺失: {relative_path}")
    payload = path.read_bytes()
    return LedgerEvidenceFile(
        root_key, relative_path, role, len(payload),
        hashlib.sha256(payload).hexdigest())


def _virtual_identity(
        relative_path: str,
        revision: P3IaLedgerRevision,
        ) -> LedgerEvidenceFile:
    payload = revision.canonical_bytes()
    return LedgerEvidenceFile(
        "REPOSITORY", relative_path, "UPSTREAM_REVISION", len(payload),
        hashlib.sha256(payload).hexdigest())


def _canonical_object(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise P3IaLedgerRevisionCatalogError("上游 manifest newline 非法")
    try:
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
    except ValueError as error:
        raise P3IaLedgerRevisionCatalogError("上游 manifest JSON 损坏") from error
    if not isinstance(value, dict):
        raise P3IaLedgerRevisionCatalogError("上游 manifest 非 object")
    return value


def _audit_inputs(repository: Path, workspace: Path) -> None:
    course_path = _path(repository, P3IA_COURSE_PATH)
    course_value = _canonical_object(course_path)
    if course_value.get("artifact_kind") != "PH2_P3IA_CAPABILITY_COURSE_FREEZE":
        raise P3IaLedgerRevisionCatalogError("P3-Ia course kind 漂移")
    course_value["artifact_kind"] = "PH2_CAPABILITY_COURSE_FREEZE"
    try:
        course = CapabilityCourseManifest.from_dict(course_value)
    except Exception as error:
        raise P3IaLedgerRevisionCatalogError("P3-Ia course 无法回读") from error
    if (course.course_status != "COURSE_FROZEN"
            or course.runtime_status != "NOT_STARTED"
            or course.pack_manifest_relative_path != P3IA_PACK_PATH
            or course.sample_relative_path != P3IA_SAMPLE_PATH
            or course.execution_state.to_value() != EXECUTION_STATE
            or not {
                "DISCOURSE_INFORMATION_STRUCTURE",
                "MEMORY_DYNAMICS",
                "REFERENCE_DISCOURSE_REVISION",
                "TYPED_LEARNING_OBJECTIVES",
            }.issubset(course.capability_keys)):
        raise P3IaLedgerRevisionCatalogError("P3-Ia course 合同漂移")
    pack_path = _path(workspace, P3IA_PACK_PATH)
    if hashlib.sha256(pack_path.read_bytes()).hexdigest() != course.pack_manifest_sha256:
        raise P3IaLedgerRevisionCatalogError("P3-Ia pack hash 漂移")
    sample_path = _path(repository, P3IA_SAMPLE_PATH)
    if hashlib.sha256(sample_path.read_bytes()).hexdigest() != course.sample_sha256:
        raise P3IaLedgerRevisionCatalogError("P3-Ia sample hash 漂移")
    historical_pack_count = sum(
        path.is_file()
        for path in (workspace / "ph2_dataset_artifacts").rglob("manifest.json"))
    p3ia_pack_count = sum(
        path.is_file()
        for path in (workspace / "ph2_p3ia_dataset_artifacts").rglob(
            "manifest.json"))
    if (historical_pack_count != LEDGER_FACTS["LC-09"][
            "base_pack_inventory_count"]
            or p3ia_pack_count != 1
            or historical_pack_count + p3ia_pack_count
            != LEDGER_FACTS["LC-09"]["current_pack_inventory_count"]):
        raise P3IaLedgerRevisionCatalogError("当前正式 pack inventory 漂移")
    checks = {
        "LC-09": ("pack_inventory_count", 16),
        "LC-13": ("route_count", 60),
        "LC-15": ("course_source_count", 9),
    }
    for ledger_key, (field, expected) in checks.items():
        value = _canonical_object(_path(repository, SUPERSEDES_PATHS[ledger_key]))
        if value.get(field) != expected:
            raise P3IaLedgerRevisionCatalogError(
                f"{ledger_key} 历史基线字段漂移")


def _common_evidence(
        repository: Path,
        workspace: Path,
        ) -> tuple[LedgerEvidenceFile, ...]:
    evidence = [
        _identity("REPOSITORY", repository, P3IA_COURSE_PATH, "COURSE"),
        _identity("REPOSITORY", repository, P3IA_SAMPLE_PATH, "SAMPLE"),
        _identity("WORKSPACE", workspace, P3IA_PACK_PATH, "PACK"),
        _identity(
            "REPOSITORY", repository, P3IA_EVALUATOR_PATH, "EVALUATOR"),
        _identity("REPOSITORY", repository, P3IA_TEST_PATH, "TEST"),
    ]
    evidence.extend(
        _identity("REPOSITORY", repository, path, "RUNTIME")
        for path in P3IA_RUNTIME_PATHS)
    return tuple(evidence)


def _revision(
        ledger_key: str,
        repository: Path,
        common: tuple[LedgerEvidenceFile, ...],
        *,
        upstream: tuple[P3IaLedgerRevision, ...] = (),
        ) -> P3IaLedgerRevision:
    supersedes = _identity(
        "REPOSITORY", repository, SUPERSEDES_PATHS[ledger_key],
        "SUPERSEDED_LEDGER")
    evidence = [*common, supersedes]
    evidence.extend(
        _virtual_identity(REVISION_PATHS[item.ledger_key].as_posix(), item)
        for item in upstream)
    try:
        return P3IaLedgerRevision(
            1,
            ledger_key,
            ARTIFACT_KINDS[ledger_key],
            ARTIFACT_VERSIONS[ledger_key],
            ARTIFACT_STATUSES[ledger_key],
            SUPERSEDES_PATHS[ledger_key],
            supersedes.sha256,
            "COURSE_FROZEN",
            "CONTRACT_READY",
            "NE",
            "PH3",
            ("zh",),
            "NE",
            0,
            "NOT_STARTED",
            "PASS",
            CanonicalJsonObject.from_value(LEDGER_FACTS[ledger_key]),
            tuple(evidence),
            CanonicalJsonObject.from_value(INVARIANTS),
            CanonicalJsonObject.from_value(EXECUTION_STATE),
        )
    except P3IaLedgerRevisionError as error:
        raise P3IaLedgerRevisionCatalogError(
            f"{ledger_key} P3-Ia revision 构建失败") from error


def build_p3ia_ledger_revisions(
        repository_root: str | Path,
        workspace_root: str | Path,
        ) -> tuple[P3IaLedgerRevision, ...]:
    """构建四个 LC 修订，再以其规范身份闭合 v40 能力基线。"""
    repository = Path(repository_root).resolve()
    workspace = Path(workspace_root).resolve()
    if repository.parent != workspace:
        raise P3IaLedgerRevisionCatalogError("repository/workspace 边界非法")
    _audit_inputs(repository, workspace)
    common = _common_evidence(repository, workspace)
    lc_revisions = tuple(
        _revision(key, repository, common)
        for key in ("LC-07", "LC-09", "LC-13", "LC-15"))
    baseline = _revision(
        "LC-00", repository, common, upstream=lc_revisions)
    return (*lc_revisions, baseline)


__all__ = [
    "P3IA_COURSE_PATH",
    "P3IA_EVALUATOR_PATH",
    "P3IA_PACK_PATH",
    "P3IA_RUNTIME_PATHS",
    "P3IA_SAMPLE_PATH",
    "P3IA_TEST_PATH",
    "REVISION_PATHS",
    "SUPERSEDES_PATHS",
    "P3IaLedgerRevisionCatalogError",
    "build_p3ia_ledger_revisions",
]
