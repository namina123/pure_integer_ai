"""构建并发布 W-06 来源独立性与稳定 REFERS 的 append-only overlay。"""
from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
from typing import Any

from pure_integer_ai.experiments.ph2_authored_alias_refers_course import (
    read_authored_alias_refers_seeds,
)
from pure_integer_ai.experiments.ph2_authored_alias_refers_w06_course import (
    PACK_NAME,
    compile_authored_alias_refers_w06_course,
    read_authored_alias_refers_w06_seeds,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    DatasetContractError,
    canonical_json_bytes,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_w06_source_semantic import (
    W06_GENERATION_HARD_CONJUNCT,
    W06_RELATION_PROFILES,
    W06_RELATION_SUBSTAGE_ORDER,
    audit_w06_authored_source_isolation,
)


W06_SOURCE_OVERLAY_PATH = (
    "data/ph2/manifests/w06_source_semantic_overlay_v1.json"
)
W06_STAGE_PATH = "data/ph2/manifests/d03_v1/stages/w06_stage_manifest_v1.json"
W06_GLOBAL_PATH = (
    "data/ph2/manifests/d03_v1/ph2_global_course_manifest_v1.json"
)
W06_INVALIDATION_PATH = (
    "data/ph2/manifests/d03_v1/stage_invalidation_graph_v1.json"
)
W06_W05_RECEIPT_PATH = (
    "data/ph2/manifests/d03_v1/w05_runtime_evidence_receipt_v1.json"
)
W06_V1_SAMPLE_PATH = (
    "data/ph2/authored_relation_alias_refers_seed_v1.jsonl.sample"
)
W06_V2_SAMPLE_PATH = (
    "data/ph2/authored_relation_alias_refers_w06_seed_v2.jsonl.sample"
)
W06_V2_COURSE_PATH = (
    "src/pure_integer_ai/experiments/ph2_authored_alias_refers_w06_course.py"
)
W06_SEMANTIC_PATH = (
    "src/pure_integer_ai/experiments/ph2_w06_source_semantic.py"
)
W06_EXPECTED_PARENT_SHA256 = {
    W06_STAGE_PATH: "a9beda13955e4708b5f2bb7f4d2b106be1bdf709c82acaefcfa95ca7d276e00a",
    W06_GLOBAL_PATH: "384329cf651ea4c5e4bc9d0b5dc4da7b22a71bc008bfabe468c86278dd9d40b6",
    W06_INVALIDATION_PATH: "21cf4d3cd65afeb0f93054773b97fa4194ee5f14dc463ff40af5813fdb0facce",
    W06_W05_RECEIPT_PATH: "64c2fff496e766df880d2db1b184e2b8a009abd3b37b1a1b1331900458ccff78",
}


class W06SourceOverlayError(RuntimeError):
    """W-06 overlay 的 parent、课程或 canonical 字节发生漂移。"""


def _sha256(path: Path) -> str:
    """流式计算文件 SHA-256，不以路径或修改时间替代内容身份。"""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while True:
                block = handle.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
    except OSError as error:
        raise W06SourceOverlayError(f"无法读取 overlay 依赖：{path}") from error
    return digest.hexdigest()


def _overlay_temp_parent(root: Path) -> Path:
    """返回 repo 内可写 scratch 目录，避免系统 Temp 沙箱权限污染语义校验。"""
    target = root / ".tmp_w06_overlay"
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise W06SourceOverlayError("无法创建 W-06 overlay 临时目录") from error
    return target


def _make_overlay_temp_dir(parent: Path) -> Path:
    """用普通目录创建可写临时根，避开 Windows tempfile ACL 收窄。"""
    for ordinal in range(1, 4096):
        target = parent / f"ph2-w06-source-overlay-{ordinal:04d}"
        try:
            target.mkdir()
        except FileExistsError:
            continue
        except OSError as error:
            raise W06SourceOverlayError("无法创建 W-06 overlay 临时子目录") from error
        return target
    raise W06SourceOverlayError("W-06 overlay 临时目录槽位耗尽")


def _profile_value() -> list[dict[str, Any]]:
    """输出无 surface 的冻结 relation profile 注册表。"""
    result = []
    for family in sorted(W06_RELATION_PROFILES):
        profile = W06_RELATION_PROFILES[family]
        result.append({
            "closure_policy": profile.closure_policy,
            "directionality": profile.directionality,
            "relation_family": family,
            "relation_kind": profile.relation_kind,
            "role_object_kinds": [
                {
                    "allowed_object_kinds": sorted(allowed),
                    "role_kind": role,
                }
                for role, allowed in profile.role_object_kinds
            ],
            "substage_key": profile.substage_key,
        })
    return result


def build_w06_source_semantic_overlay(repo_root: str | Path) -> dict[str, Any]:
    """重编 v2 临时 pack，并绑定 parent、语义 firewall 和四重来源隔离。"""
    root = Path(repo_root)
    parents = {}
    for relative, expected in W06_EXPECTED_PARENT_SHA256.items():
        actual = _sha256(root / relative)
        if actual != expected:
            raise W06SourceOverlayError(f"W-06 overlay parent 漂移：{relative}")
        parents[relative] = actual

    v1_path = root / W06_V1_SAMPLE_PATH
    v2_path = root / W06_V2_SAMPLE_PATH
    legacy = read_authored_alias_refers_seeds(v1_path)
    stable = read_authored_alias_refers_w06_seeds(v2_path)
    legacy_occurrence_refers = sum(
        1 for seed in legacy
        if seed.relation_family == "REFERS"
        and any(item.object_kind == 5 for item in seed.endpoints)
    )
    if legacy_occurrence_refers <= 0:
        raise W06SourceOverlayError("旧 alias/refers 没有可证明的篇章指代拒绝样本")

    temp_parent = _overlay_temp_parent(root)
    temp = _make_overlay_temp_dir(temp_parent)
    try:
        build = compile_authored_alias_refers_w06_course(v2_path, temp)
        report = audit_w06_authored_source_isolation(build.pack_root, stable)
        pack_manifest_sha = hashlib.sha256(
            canonical_json_bytes(build.manifest.to_dict())
        ).hexdigest()
    finally:
        shutil.rmtree(temp, ignore_errors=True)
        try:
            temp_parent.rmdir()
        except OSError:
            pass

    return {
        "artifact_kind": "PH2_W06_SOURCE_SEMANTIC_OVERLAY",
        "artifact_version": "PH2-W06-SOURCE-SEMANTIC-OVERLAY-V1",
        "execution_state": {
            "LANGUAGE_CAPABILITY_MASTERED": 0,
            "LANGUAGE_READINESS": 0,
            "OPEN_GENERATION": "NE_NOT_YET_EVALUABLE",
            "W06_STARTED": 0,
            "W07_STARTED": 0,
            "teacher_calls": 0,
        },
        "format_version": 1,
        "generation_supplement": {
            "hard_conjunct": W06_GENERATION_HARD_CONJUNCT,
            "historical_stage_manifest_modified": 0,
            "status": "CONTRACT_REQUIRED_BEFORE_FORMAL_RUN",
        },
        "legacy_v1_boundary": {
            "occurrence_refers_rejected_count": legacy_occurrence_refers,
            "sample_sha256": _sha256(v1_path),
            "status": "RETAINED_HISTORY_REJECTED_BY_W06_FIREWALL",
        },
        "parent_identities": parents,
        "relation_profiles": _profile_value(),
        "relation_substage_order": list(W06_RELATION_SUBSTAGE_ORDER),
        "stable_v2_course": {
            "evaluator_family_count": len(report.evaluator_families),
            "evaluator_label_count": report.evaluator_label_count,
            "evaluator_owner_count": len(report.evaluator_owner_keys),
            "evaluator_template_count": len(report.evaluator_templates),
            "held_out_cluster_count": len(report.held_out_cluster_keys),
            "held_out_observation_count": report.held_out_observation_count,
            "pack_key": PACK_NAME,
            "pack_manifest_sha256": pack_manifest_sha,
            "sample_sha256": _sha256(v2_path),
            "source_independence_policy": (
                "DISTINCT_CLUSTER_OWNER_SEED_FAMILY_TEMPLATE"
            ),
            "source_key": report.source_key,
            "teacher_evidence_count": report.teacher_evidence_count,
            "teacher_family_count": len(report.teacher_families),
            "teacher_owner_count": len(report.teacher_owner_keys),
            "teacher_template_count": len(report.teacher_templates),
            "train_cluster_count": len(report.train_cluster_keys),
            "train_observation_count": report.train_observation_count,
        },
        "source_files": {
            W06_V2_COURSE_PATH: _sha256(root / W06_V2_COURSE_PATH),
            W06_SEMANTIC_PATH: _sha256(root / W06_SEMANTIC_PATH),
            W06_V2_SAMPLE_PATH: _sha256(v2_path),
        },
        "status": "W06_SOURCE_AND_SEMANTIC_PREREQUISITE_PASS",
    }


def canonical_w06_source_semantic_overlay_bytes(
        repo_root: str | Path,
        ) -> bytes:
    """返回 W-06 overlay 的规范 JSON 单行字节。"""
    return canonical_json_bytes(build_w06_source_semantic_overlay(repo_root)) + b"\n"


def read_w06_source_semantic_overlay(path: str | Path) -> dict[str, Any]:
    """严格回读已发布 overlay，拒绝非规范 JSON 或缺换行。"""
    target = Path(path)
    try:
        payload = target.read_bytes()
    except OSError as error:
        raise W06SourceOverlayError("无法读取 W-06 source overlay") from error
    if not payload.endswith(b"\n") or payload == b"\n":
        raise W06SourceOverlayError("W-06 source overlay 必须是单行规范 JSON")
    try:
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
    except DatasetContractError as error:
        raise W06SourceOverlayError(
            "W-06 source overlay 不是规范 JSON") from error
    if canonical_json_bytes(value) + b"\n" != payload:
        raise W06SourceOverlayError("W-06 source overlay canonical 字节漂移")
    assert isinstance(value, dict)
    if (value.get("artifact_kind") != "PH2_W06_SOURCE_SEMANTIC_OVERLAY"
            or value.get("status")
            != "W06_SOURCE_AND_SEMANTIC_PREREQUISITE_PASS"):
        raise W06SourceOverlayError("W-06 source overlay 合同字段漂移")
    return value


def publish_w06_source_semantic_overlay(
        repo_root: str | Path,
        output_path: str | Path | None = None,
        ) -> Path:
    """以排他创建发布 overlay，已有路径一律拒绝覆盖。"""
    root = Path(repo_root)
    target = root / (output_path or W06_SOURCE_OVERLAY_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_w06_source_semantic_overlay_bytes(root)
    try:
        with target.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as error:
        raise W06SourceOverlayError("W-06 source overlay 已存在，禁止覆盖") from error
    return target


__all__ = [
    "W06_EXPECTED_PARENT_SHA256",
    "W06_SOURCE_OVERLAY_PATH",
    "W06SourceOverlayError",
    "build_w06_source_semantic_overlay",
    "canonical_w06_source_semantic_overlay_bytes",
    "publish_w06_source_semantic_overlay",
    "read_w06_source_semantic_overlay",
]
