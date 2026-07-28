"""从当前 R-01..R-06 证据构建 v41 能力基线。"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from pure_integer_ai.experiments.ph2_authorized_center_generation_contract import (
    read_authorized_center_generation_manifest,
    verify_authorized_center_generation_files,
)
from pure_integer_ai.experiments.ph2_capability_baseline_v41_contract import (
    ARTIFACT_KIND,
    ARTIFACT_STATUS,
    ARTIFACT_VERSION,
    CAPABILITY_FACTS,
    EXECUTION_STATE,
    REQUIRED_EVIDENCE_ROLES,
    SUPERSEDES_PATH,
    VERSION_EVIDENCE,
    VERSION_KEYS,
    BaselineV41EvidenceFile,
    CapabilityBaselineV41Error,
    CapabilityBaselineV41Manifest,
)
from pure_integer_ai.experiments.ph2_correction_recovery_contract import (
    read_correction_recovery_manifest,
    verify_correction_recovery_files,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_long_context_absorption_contract import (
    read_long_context_absorption_manifest,
    verify_long_context_absorption_files,
)
from pure_integer_ai.experiments.ph2_p3ia_ledger_revision_contract import (
    read_p3ia_ledger_revision,
    verify_p3ia_ledger_revision_files,
)
from pure_integer_ai.experiments.ph2_source_pack_compiler import (
    read_source_pack_coverage_manifest,
)
from pure_integer_ai.experiments.ph2_storage_absorption_contract import (
    read_storage_absorption_manifest,
    verify_storage_absorption_files,
)
from pure_integer_ai.experiments.ph2_typed_proof_family_contract import (
    read_typed_proof_family_manifest,
    verify_typed_proof_family_files,
)


_R02_PATH = "data/ph2/manifests/r02_storage_absorption_v1.json"
_R03_PATH = "data/ph2/manifests/r03_correction_recovery_absorption_v1.json"
_R04_PATH = (
    "data/ph2/manifests/r04_authorized_center_generation_absorption_v1.json")
_R05_PATH = "data/ph2/manifests/r05_typed_proof_family_absorption_v1.json"
_R06_PATH = "data/ph2/manifests/r06_long_context_absorption_v1.json"
_P3IA_PATH = (
    "data/ph2/manifests/p3ia_free_text_hierarchy_recall_course_v2.json")
_D02_PATH = "data/ph2/manifests/d02_source_pack_coverage_v1.json"
_GG_PATHS = (
    "data/ph2/manifests/gg01_generation_choice_contract_v2.json",
    "data/ph2/manifests/gg02_generation_choice_outcome_bridge_v1.json",
    "data/ph2/manifests/gg03_generation_generalization_course_v1.json",
)
_MD_PATHS = (
    "data/ph2/manifests/md01_memory_dynamics_contract_v1.json",
    "data/ph2/manifests/md02_situation_state_adapter_v1.json",
    "data/ph2/manifests/md03_directional_center_adapter_v1.json",
    "data/ph2/manifests/md04_center_diffusion_probe_plan_v1.json",
    "data/ph2/manifests/md04_center_diffusion_probe_runs_v1.json",
    "data/ph2/manifests/md05_center_diffusion_decision_v1.json",
)


class CapabilityBaselineV41CatalogError(RuntimeError):
    """v41 上游 artifact、文件身份或生产裁决漂移。"""


def _path(root: Path, relative_path: str) -> Path:
    result = (root / Path(*relative_path.split("/"))).resolve()
    try:
        result.relative_to(root)
    except ValueError as error:
        raise CapabilityBaselineV41CatalogError("v41 catalog 路径逃逸") from error
    return result


def _json(root: Path, relative_path: str) -> dict[str, Any]:
    try:
        payload = _path(root, relative_path).read_bytes()
        if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
            raise CapabilityBaselineV41CatalogError("上游 JSON newline 非法")
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
        assert isinstance(value, dict)
        return value
    except CapabilityBaselineV41CatalogError:
        raise
    except Exception as error:
        raise CapabilityBaselineV41CatalogError("上游 JSON 损坏") from error


def _zero_execution(value: dict[str, Any], *, where: str) -> None:
    execution = value.get("execution_state")
    if isinstance(execution, dict):
        if any(item != 0 for item in execution.values()):
            raise CapabilityBaselineV41CatalogError(f"{where} 执行状态非零")
        return
    direct_keys = (
        "d03_published", "formal_training_runs", "learning_state_writes",
        "mastered_claims", "readiness_claims", "teacher_calls", "w01_started",
    )
    if any(value.get(key) != 0 for key in direct_keys):
        raise CapabilityBaselineV41CatalogError(f"{where} 执行状态非零")


def _identity(
        repository: Path,
        relative_path: str,
        role: str,
        ) -> BaselineV41EvidenceFile:
    path = _path(repository, relative_path)
    if not path.is_file():
        raise CapabilityBaselineV41CatalogError(
            f"v41 evidence 缺失: {relative_path}")
    payload = path.read_bytes()
    return BaselineV41EvidenceFile(
        relative_path, role, len(payload), hashlib.sha256(payload).hexdigest())


def _audit_absorption_artifacts(repository: Path, workspace: Path) -> None:
    r02 = read_storage_absorption_manifest(_path(repository, _R02_PATH))
    verify_storage_absorption_files(
        r02, repository_root=repository, workspace_root=workspace)
    r03 = read_correction_recovery_manifest(_path(repository, _R03_PATH))
    verify_correction_recovery_files(r03, repository_root=repository)
    r04 = read_authorized_center_generation_manifest(
        _path(repository, _R04_PATH))
    verify_authorized_center_generation_files(r04, repository_root=repository)
    r05 = read_typed_proof_family_manifest(_path(repository, _R05_PATH))
    verify_typed_proof_family_files(r05, repository_root=repository)
    r06 = read_long_context_absorption_manifest(_path(repository, _R06_PATH))
    verify_long_context_absorption_files(r06, repository_root=repository)
    artifacts = (r02, r03, r04, r05, r06)
    if any(item.artifact_status != "PRODUCTION_EVIDENCED" for item in artifacts):
        raise CapabilityBaselineV41CatalogError("R-02..R-06 状态漂移")
    if any(any(value != 0 for value in item.execution_state.to_value().values())
           for item in artifacts):
        raise CapabilityBaselineV41CatalogError("R-02..R-06 执行状态非零")


def _audit_language_and_crosscut(
        repository: Path, workspace: Path,
        ) -> None:
    v40 = read_p3ia_ledger_revision(_path(repository, SUPERSEDES_PATH))
    verify_p3ia_ledger_revision_files(
        v40, repository_root=repository, workspace_root=workspace)
    if not (
            v40.ledger_key == "LC-00"
            and v40.artifact_status == "BASELINE_FROZEN"
            and v40.p3ia_course_status == "COURSE_FROZEN"
            and v40.p3ia_production_contract_status == "CONTRACT_READY"
            and v40.p3ib_status == "NE"
            and v40.p3ib_phase == "PH3"
            and v40.code_switch_status == "NE"
            and v40.cross_language_pass_authority == 0):
        raise CapabilityBaselineV41CatalogError("v40 P3-Ia/P3-Ib 状态漂移")

    p3ia = _json(repository, _P3IA_PATH)
    if not (
            p3ia.get("artifact_kind") == "PH2_P3IA_CAPABILITY_COURSE_FREEZE"
            and p3ia.get("course_status") == "COURSE_FROZEN"
            and p3ia.get("runtime_status") == "NOT_STARTED"):
        raise CapabilityBaselineV41CatalogError("P3-Ia course 状态漂移")
    _zero_execution(p3ia, where="P3-Ia")

    coverage = read_source_pack_coverage_manifest(_path(repository, _D02_PATH))
    frozen = tuple(item for item in coverage.entries if item.status == "PACK_FROZEN")
    blocked = tuple(item for item in coverage.entries if item.status == "BLOCKED")
    if (len(coverage.entries), len(frozen), len(blocked)) != (7, 6, 1):
        raise CapabilityBaselineV41CatalogError("D-02 source coverage 漂移")
    if not (
            blocked[0].source_key == "CC_CEDICT_20260725"
            and blocked[0].blocker_code
            == "OFFICIAL_LICENSE_EVIDENCE_DIVERGENCE"):
        raise CapabilityBaselineV41CatalogError("CC-CEDICT blocker 漂移")
    _zero_execution(_json(repository, _D02_PATH), where="D-02")

    gg = tuple(_json(repository, path) for path in _GG_PATHS)
    if not (
            gg[0].get("artifact_status") == "CONTRACT_FROZEN"
            and gg[1].get("artifact_status") == "BRIDGE_FROZEN"
            and gg[2].get("course_status") == "COURSE_FROZEN"
            and gg[2].get("runtime_status") == "NOT_STARTED"):
        raise CapabilityBaselineV41CatalogError("GG 状态漂移")
    md = tuple(_json(repository, path) for path in _MD_PATHS)
    if not (
            md[0].get("artifact_status") == "CONTRACT_FROZEN"
            and md[1].get("artifact_status") == "ADAPTER_FROZEN"
            and md[2].get("artifact_status") == "ADAPTER_FROZEN"
            and md[3].get("results_observed") == 0
            and md[4].get("results_observed") == 1
            and md[5].get("results_observed") == 1
            and md[5].get("verdict") == "PASS"):
        raise CapabilityBaselineV41CatalogError("MD 状态漂移")
    for index, value in enumerate((*gg, *md)):
        _zero_execution(value, where=f"MD/GG-{index}")


def build_capability_baseline_v41(
        repository_root: str | Path,
        workspace_root: str | Path,
        ) -> CapabilityBaselineV41Manifest:
    """回验当前依赖并构建不可覆盖 v41 能力基线。"""
    repository = Path(repository_root).resolve()
    workspace = Path(workspace_root).resolve()
    if repository.parent != workspace:
        raise CapabilityBaselineV41CatalogError("repository/workspace 边界非法")
    _audit_absorption_artifacts(repository, workspace)
    _audit_language_and_crosscut(repository, workspace)
    supersedes = _path(repository, SUPERSEDES_PATH).read_bytes()
    evidence = tuple(
        _identity(repository, path, role)
        for path, role in REQUIRED_EVIDENCE_ROLES.items())
    try:
        return CapabilityBaselineV41Manifest(
            1,
            ARTIFACT_KIND,
            ARTIFACT_VERSION,
            ARTIFACT_STATUS,
            SUPERSEDES_PATH,
            hashlib.sha256(supersedes).hexdigest(),
            CanonicalJsonObject.from_value(VERSION_KEYS),
            CanonicalJsonObject.from_value(VERSION_EVIDENCE),
            CanonicalJsonObject.from_value(CAPABILITY_FACTS),
            evidence,
            CanonicalJsonObject.from_value(EXECUTION_STATE),
        )
    except CapabilityBaselineV41Error as error:
        raise CapabilityBaselineV41CatalogError("v41 构建失败") from error


__all__ = [
    "CapabilityBaselineV41CatalogError",
    "build_capability_baseline_v41",
]
