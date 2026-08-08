"""J-F2 最终联合闸的只读公开 preflight 合同。

该模块只消费公开 receipt 和未来的公开封存 manifest。它不能创建断奶
状态，也不能读取 Candidate/private evaluator payload；缺少 J-F1 或 Core
封存身份时必须保持 BLOCKED。
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path, PurePosixPath
from typing import Any

from pure_integer_ai.experiments.artifact_verification_mode import (
    CURRENT_HEAD_COMPATIBILITY_VERIFY,
    require_artifact_verification_mode,
)
from pure_integer_ai.experiments.j_f1_facility_receipt import (
    read_j_f1_facility_receipt,
)
from pure_integer_ai.experiments.j_f2_core_artifact_manifest import (
    CoreArtifactManifestError,
    read_core_artifact_manifest,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_w09_authority import (
    W09_ABLATION_KEYS,
    W09_DIMENSION_KEYS,
    W09_WALL_DIMENSION_KEYS,
)


FORMAT_VERSION = 1
ARTIFACT_KIND = "PH2_J_F2_FINAL_JOINT_PREFLIGHT"
ARTIFACT_VERSION = "J-F2-PREFLIGHT-20260806-A"
PUBLIC_RECEIPT_STATUS = "RUNTIME_EVIDENCED"
W01_RECEIPT_STATUS = "W01_PROTOCOL_VERIFIED"
W02_RECEIPT_SHA256 = (
    "6b1344bfb226ea2488760987a838b4a7d4016f14831d6ed58c78b9ff0e45a2eb"
)

D03_RECEIPT_PATH = "data/ph2/manifests/d03_v1/ph2_d03_post_publication_receipt_v1.json"
W01_RECEIPT_PATH = "data/ph2/manifests/w01_v1/ph2_w01_stage0_receipt_v2.json"
W02_SUPPLEMENTAL_PATH = "data/ph2/manifests/w02_lc16_supplemental_runtime_receipt_v1.json"
W03_SUPPLEMENTAL_PATH = "data/ph2/manifests/w03_lc16_supplemental_runtime_receipt_v1.json"
W03_RECEIPT_PATH = "data/ph2/manifests/d03_v1/w03_runtime_evidence_receipt_v1.json"
W04_RECEIPT_PATH = "data/ph2/manifests/d03_v1/w04_runtime_evidence_receipt_v1.json"
W05_RECEIPT_PATH = "data/ph2/manifests/d03_v1/w05_runtime_evidence_receipt_v1.json"
W06_RECEIPT_PATH = "data/ph2/manifests/d03_v1/w06_runtime_evidence_receipt_v1.json"
W07_RECEIPT_PATH = "data/ph2/manifests/d03_v1/w07_runtime_evidence_receipt_v1.json"
W08_RECEIPT_PATH = "data/ph2/manifests/d03_v1/w08_runtime_evidence_receipt_v1.json"
W09_RECEIPT_PATH = "data/ph2/manifests/d03_v1/w09_runtime_evidence_receipt_v1.json"
J_F1_RECEIPT_PATH = "data/ph2/manifests/j_f1_facility_receipt_v1.json"
CORE_ARTIFACT_PATH = "data/ph2/manifests/j_f2_core_artifact_manifest_v1.json"

RECEIPT_BINDINGS = (
    ("D03_POST_PUBLICATION", D03_RECEIPT_PATH),
    ("W01_PROTOCOL", W01_RECEIPT_PATH),
    ("W02_FORMAL_COMMITMENT", W03_RECEIPT_PATH),
    ("W02_LC16_SUPPLEMENTAL", W02_SUPPLEMENTAL_PATH),
    ("W03_LC16_SUPPLEMENTAL", W03_SUPPLEMENTAL_PATH),
    ("W03_RUNTIME", W03_RECEIPT_PATH),
    ("W04_RUNTIME", W04_RECEIPT_PATH),
    ("W05_RUNTIME", W05_RECEIPT_PATH),
    ("W06_RUNTIME", W06_RECEIPT_PATH),
    ("W07_RUNTIME", W07_RECEIPT_PATH),
    ("W08_RUNTIME", W08_RECEIPT_PATH),
    ("W09_RUNTIME", W09_RECEIPT_PATH),
)


class JF2PreflightError(RuntimeError):
    """J-F2 公开依赖、canonical bytes 或状态发生漂移。"""


def _exact(value: Any, keys: set[str], *, where: str) -> dict[str, Any]:
    """校验 typed object 的字段集合不被静默扩展或删减。"""
    if not isinstance(value, dict) or set(value) != keys:
        raise JF2PreflightError(f"{where} 字段不精确")
    return value


def _relative(value: str, *, where: str) -> str:
    """校验公开相对路径不能越界或混入 Windows 分隔符。"""
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise JF2PreflightError(f"{where} 相对路径非法")
    return value


def _sha256(value: str, *, where: str) -> str:
    """校验身份摘要为小写 SHA-256。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(item not in "0123456789abcdef" for item in value)):
        raise JF2PreflightError(f"{where} SHA-256 非法")
    return value


def _read_public_json(path: Path) -> dict[str, Any]:
    """只读回 canonical 公开 JSON，并拒绝额外换行或字段重排。"""
    try:
        payload = path.read_bytes()
        if payload.endswith(b"\n\n"):
            raise JF2PreflightError("公开 receipt 含多余尾换行")
        body = payload[:-1] if payload.endswith(b"\n") else payload
        value = parse_canonical_json_bytes(body, require_object=True)
        if not isinstance(value, dict):
            raise JF2PreflightError("公开 receipt 根必须是 object")
        if canonical_json_bytes(value) not in {payload, body}:
            raise JF2PreflightError("公开 receipt bytes 非 canonical")
        return value
    except JF2PreflightError:
        raise
    except Exception as error:
        raise JF2PreflightError(f"公开 receipt 无法读取: {path.name}") from error


def _identity(root: Path, relative_path: str) -> tuple[int, str] | None:
    """返回公开文件 size/SHA；缺失时返回 None，不猜测未来 artifact。"""
    target = (root / Path(*relative_path.split("/"))).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        return None
    payload = target.read_bytes()
    return len(payload), hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, order=True)
class JF2Dependency:
    """一个公开 J-F2 依赖的逐字节身份和当前状态。"""

    role: str
    relative_path: str
    status: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        """确保依赖状态和文件身份可稳定排序。"""
        _relative(self.relative_path, where=self.role)
        if self.status not in {"PASS", "MISSING", "NE", "FAIL"}:
            raise JF2PreflightError("J-F2 dependency status 非法")
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise JF2PreflightError("J-F2 dependency size 非法")
        if self.sha256 != "0" * 64:
            _sha256(self.sha256, where=self.role)
        elif self.status != "MISSING":
            raise JF2PreflightError("非缺失依赖不得使用空 SHA")

    def to_dict(self) -> dict[str, Any]:
        """返回公开且不含 private payload 的依赖身份。"""
        return {
            "relative_path": self.relative_path,
            "role": self.role,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "status": self.status,
        }


@dataclass(frozen=True)
class JF2PreflightReport:
    """汇总 J-F2 公开依赖；报告本身不授予 readiness。"""

    format_version: int
    artifact_kind: str
    artifact_version: str
    status: str
    dependencies: tuple[JF2Dependency, ...]
    blockers: tuple[str, ...]
    language_capability_mastered: int
    language_readiness: int

    def __post_init__(self) -> None:
        """保持阻断状态与 readiness=0 的硬合取边界。"""
        if (self.format_version != FORMAT_VERSION
                or self.artifact_kind != ARTIFACT_KIND
                or self.artifact_version != ARTIFACT_VERSION):
            raise JF2PreflightError("J-F2 preflight artifact identity 漂移")
        if self.status not in {"BLOCKED", "READY_FOR_FORMAL_SEAL"}:
            raise JF2PreflightError("J-F2 preflight status 非法")
        if not self.dependencies:
            raise JF2PreflightError("J-F2 缺少依赖清单")
        if tuple(item.role for item in self.dependencies) != tuple(
                role for role, _ in RECEIPT_BINDINGS) + (
                    "J_F1_FACILITY", "CORE_ARTIFACT"):
            raise JF2PreflightError("J-F2 dependency 顺序漂移")
        if tuple(sorted(set(self.blockers))) != self.blockers:
            raise JF2PreflightError("J-F2 blocker 必须唯一稳定排序")
        if self.status == "BLOCKED" and not self.blockers:
            raise JF2PreflightError("BLOCKED preflight 缺少 blocker")
        if self.status == "READY_FOR_FORMAL_SEAL" and self.blockers:
            raise JF2PreflightError("ready preflight 仍含 blocker")
        if self.language_capability_mastered != 1 or self.language_readiness != 0:
            raise JF2PreflightError("preflight 状态必须是 mastered=1/readiness=0")

    def to_dict(self) -> dict[str, Any]:
        """返回 canonical、公开且不会宣称正式断奶的报告。"""
        return {
            "artifact_kind": self.artifact_kind,
            "artifact_version": self.artifact_version,
            "blockers": list(self.blockers),
            "dependencies": [item.to_dict() for item in self.dependencies],
            "format_version": self.format_version,
            "language_capability_mastered": self.language_capability_mastered,
            "language_readiness": self.language_readiness,
            "status": self.status,
        }

    def canonical_bytes(self) -> bytes:
        """返回带单尾换行的 canonical JSON。"""
        return canonical_json_bytes(self.to_dict()) + b"\n"

    def sha256(self) -> str:
        """返回 preflight 报告的稳定摘要。"""
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


def _receipt_status(role: str, value: dict[str, Any]) -> tuple[str, list[str]]:
    """按阶段合同判定公开 receipt，不读取任何 expected/private 字段。"""
    if role == "W02_FORMAL_COMMITMENT":
        return ("PASS", [] ) if value.get("w02_receipt_sha256") == W02_RECEIPT_SHA256 else (
            "FAIL", ["W02_COMMITMENT_DRIFT"])
    if role in {"W02_LC16_SUPPLEMENTAL", "W03_LC16_SUPPLEMENTAL"}:
        return ("PASS", []) if value.get("status") == "PASS" else (
            "FAIL", [f"{role}_NOT_PASS"])
    state = value.get("execution_state")
    if not isinstance(state, dict):
        return "FAIL", [f"{role}_EXECUTION_STATE_MISSING"]
    blockers: list[str] = []
    if role == "D03_POST_PUBLICATION":
        if value.get("status") != "POST_PUBLISH_VERIFIED" or state.get("d03_published") != 1:
            blockers.append("D03_NOT_POST_PUBLISH_VERIFIED")
    elif role == "W01_PROTOCOL":
        if value.get("status") != W01_RECEIPT_STATUS or state.get("W01_PROTOCOL_VERIFIED") != 1:
            blockers.append("W01_PROTOCOL_NOT_VERIFIED")
    else:
        stage = role.split("_", 1)[0]
        if value.get("status") != PUBLIC_RECEIPT_STATUS:
            blockers.append(f"{stage}_NOT_RUNTIME_EVIDENCED")
        if state.get(f"{stage}_RUNTIME_EVIDENCED") != 1:
            blockers.append(f"{stage}_RUNTIME_FLAG_MISSING")
        if state.get(f"{stage}_BLOCKED_FAILED") != 0:
            blockers.append(f"{stage}_BLOCKED_OR_FAILED")
    if state.get("teacher_calls") != 0:
        blockers.append(f"{role}_TEACHER_CALL_NONZERO")
    return ("PASS" if not blockers else "FAIL"), blockers


def _w09_blockers(value: dict[str, Any]) -> list[str]:
    """核对 W09 与 J-LC-W09 的公开硬合取。"""
    blockers: list[str] = []
    state = value.get("execution_state", {})
    if state.get("PRE_WEAN_LANGUAGE_LEARNING_CAPABILITY_EVIDENCED") != 1:
        blockers.append("W09_PRE_WEAN_CAPABILITY_MISSING")
    if state.get("LANGUAGE_CAPABILITY_MASTERED") != 1:
        blockers.append("LANGUAGE_CAPABILITY_NOT_MASTERED")
    if state.get("LANGUAGE_READINESS") != 0:
        blockers.append("LANGUAGE_READINESS_PRESET")
    j_lc = value.get("j_lc", {})
    if (not isinstance(j_lc, dict) or j_lc.get("status") != "PASS"
            or j_lc.get("bearing_cell_count") != 216
            or j_lc.get("lc_task_count") != 16
            or j_lc.get("retention_continual_learning_cell_count") != 27):
        blockers.append("J_LC_W09_NOT_PASS")
    expected_walls = [[key, "NE"] for key in W09_WALL_DIMENSION_KEYS]
    if j_lc.get("wall_dimension_states") != expected_walls:
        blockers.append("W09_WALL_DIMENSION_DRIFT")
    generation = value.get("open_generation", {})
    expected_layers = [[key, "PASS"] for key in (
        "CONTENT_SEMANTICS", "STRUCTURE", "DISCOURSE_SCOPE",
        "MORPHOLOGY_SURFACE", "TASK_USE")]
    if (not isinstance(generation, dict) or generation.get("status") != "PASS"
            or generation.get("layer_states") != expected_layers
            or generation.get("output_invocation_count", 0) <= 0
            or any(generation.get(key) != 0 for key in (
                "complete_template_replay_count", "exact_surface_read_count",
                "source_replay_count"))):
        blockers.append("OPEN_GENERATION_NOT_PASS")
    dimensions = value.get("dimension_results", ())
    if (not isinstance(dimensions, list)
            or tuple(item.get("dimension_key") for item in dimensions
                     if isinstance(item, dict)) != W09_DIMENSION_KEYS
            or any(not isinstance(item, dict)
                   or item.get("status") != "PASS"
                   or item.get("required_count", 0) <= 0
                   or item.get("passed_count") != item.get("required_count")
                   or item.get("fail_count") != 0
                   or item.get("ne_count") != 0
                   for item in dimensions)):
        blockers.append("W09_BEARING_DIMENSION_NOT_PASS")
    ablations = value.get("ablation_results", ())
    if (not isinstance(ablations, list)
            or tuple(item.get("ablation_key") for item in ablations
                     if isinstance(item, dict)) != W09_ABLATION_KEYS
            or any(not isinstance(item, dict)
                   or item.get("target_dimension_key")
                   != W09_ABLATION_KEYS[index].removesuffix("-ABLATION")
                   or item.get("real_component_disabled") != 1
                   or item.get("status") != (
                       "PASS" if index < len(W09_DIMENSION_KEYS) else "NE")
                   or (index < len(W09_DIMENSION_KEYS)
                       and item.get("invocation_count", 0) <= 0)
                   or (index >= len(W09_DIMENSION_KEYS)
                       and item.get("invocation_count") != 0)
                   for index, item in enumerate(ablations))):
        blockers.append("W09_ABLATION_OR_WALL_DRIFT")
    v06 = value.get("v06", {})
    if (not isinstance(v06, dict) or v06.get("status") != "PASS"
            or v06.get("core_bit_identical") != 1
            or v06.get("host_write_count") != 0
            or v06.get("improved_probe_count", 0) <= 0
            or v06.get("improved_probe_count") != v06.get("independent_probe_count")
            or v06.get("isolated_learning_write_count", 0) <= 0):
        blockers.append("W09_V06_NOT_PASS")
    rollback = value.get("rollback", {})
    if (not isinstance(rollback, dict) or rollback.get("status") != "PASS"
            or rollback.get("invalidated_count", 0) <= 0
            or rollback.get("preserved_count", 0) <= 0
            or rollback.get("leaked_write_count") != 0):
        blockers.append("W09_ROLLBACK_NOT_PASS")
    resource = value.get("resource", {})
    if (not isinstance(resource, dict) or resource.get("status") != "PASS"
            or resource.get("fresh_resume_equivalent") != 1
            or resource.get("worker_1_2_4_invariant") != 1):
        blockers.append("W09_RESOURCE_NOT_PASS")
    windows = value.get("windows", ())
    if (not isinstance(windows, list) or len(windows) != 3
            or any(not isinstance(item, dict)
                   or item.get("window_ordinal") != index
                   or item.get("status") != "PASS"
                   or item.get("teacher_calls") != 0
                   for index, item in enumerate(windows, start=1))):
        blockers.append("W09_TEACHER_ZERO_WINDOW_NOT_PASS")
    write_counts = value.get("write_counts", {})
    expected_write_keys = {
        "assessment_writes", "candidate_writes", "clock_writes", "core_writes",
        "evidence_writes", "host_writes", "label_writes", "memory_writes",
        "public_writes", "use_writes",
    }
    if (not isinstance(write_counts, dict) or set(write_counts) != expected_write_keys
            or any(type(item) is not int or item != 0
                   for item in write_counts.values())):
        blockers.append("W09_PROTECTED_WRITE_NONZERO")
    owner_writes = value.get("candidate_evidence", {}).get("owner_write_counts", {})
    if (not isinstance(owner_writes, dict)
            or any(owner_writes.get(key) != 0 for key in (
                "companion_writes", "evaluator_label_writes", "host_learning_writes",
                "memory_learning_writes", "readback_payload_gets", "teacher_calls"))
            or owner_writes.get("formal_training_runs") != 1):
        blockers.append("W09_OWNER_BOUNDARY_DRIFT")
    if value.get("private_evidence", {}).get("terminal_state") != "PASS":
        blockers.append("W09_PRIVATE_TERMINAL_NOT_PASS")
    return blockers


def build_jf2_preflight(
        repository_root: str | Path,
        *,
        verification_mode: str = CURRENT_HEAD_COMPATIBILITY_VERIFY,
        ) -> JF2PreflightReport:
    """按显式历史或当前语义只读审计 J-F2 公开依赖。"""
    try:
        mode = require_artifact_verification_mode(verification_mode)
    except ValueError as error:
        raise JF2PreflightError(str(error)) from error
    root = Path(repository_root).resolve()
    dependencies: list[JF2Dependency] = []
    blockers: set[str] = set()
    loaded: dict[str, dict[str, Any]] = {}
    for role, relative_path in RECEIPT_BINDINGS:
        identity = _identity(root, relative_path)
        if identity is None:
            dependencies.append(JF2Dependency(role, relative_path, "MISSING", 0, "0" * 64))
            blockers.add(f"{role}_MISSING")
            continue
        size_bytes, sha256 = identity
        try:
            value = _read_public_json(root / Path(*relative_path.split("/")))
            loaded[role] = value
            status, issues = _receipt_status(role, value)
            blockers.update(issues)
        except JF2PreflightError:
            status = "FAIL"
            blockers.add(f"{role}_CANONICAL_DRIFT")
        dependencies.append(JF2Dependency(role, relative_path, status, size_bytes, sha256))

    w03 = loaded.get("W02_FORMAL_COMMITMENT")
    if w03 is not None:
        blockers.update(_w09_blockers(loaded.get("W09_RUNTIME", {})))

    j_f1_identity = _identity(root, J_F1_RECEIPT_PATH)
    if j_f1_identity is None:
        dependencies.append(JF2Dependency(
            "J_F1_FACILITY", J_F1_RECEIPT_PATH,
            "MISSING", 0, "0" * 64))
        blockers.add("J_F1_FACILITY_MISSING")
    else:
        size_bytes, sha256 = j_f1_identity
        try:
            read_j_f1_facility_receipt(
                root,
                verify_runtime=(mode == CURRENT_HEAD_COMPATIBILITY_VERIFY),
                verification_mode=mode,
            )
            j_f1_status = "PASS"
        except (OSError, RuntimeError, TypeError, ValueError):
            j_f1_status = "FAIL"
            blockers.add("J_F1_FACILITY_INVALID")
        dependencies.append(JF2Dependency(
            "J_F1_FACILITY", J_F1_RECEIPT_PATH,
            j_f1_status, size_bytes, sha256))

    core_identity = _identity(root, CORE_ARTIFACT_PATH)
    if core_identity is None:
        dependencies.append(JF2Dependency(
            "CORE_ARTIFACT", CORE_ARTIFACT_PATH,
            "MISSING", 0, "0" * 64))
        blockers.add("CORE_ARTIFACT_MISSING")
    else:
        size_bytes, sha256 = core_identity
        try:
            read_core_artifact_manifest(
                root,
                CORE_ARTIFACT_PATH,
                verify_files=(mode == CURRENT_HEAD_COMPATIBILITY_VERIFY),
            )
            core_status = "PASS"
        except (OSError, RuntimeError, TypeError, ValueError, CoreArtifactManifestError):
            core_status = "FAIL"
            blockers.add("CORE_ARTIFACT_INVALID")
        dependencies.append(JF2Dependency(
            "CORE_ARTIFACT", CORE_ARTIFACT_PATH,
            core_status, size_bytes, sha256))

    ordered_blockers = tuple(sorted(blockers))
    return JF2PreflightReport(
        FORMAT_VERSION,
        ARTIFACT_KIND,
        ARTIFACT_VERSION,
        "BLOCKED" if ordered_blockers else "READY_FOR_FORMAL_SEAL",
        tuple(dependencies),
        ordered_blockers,
        1,
        0,
    )


def read_jf2_preflight(path: str | Path) -> JF2PreflightReport:
    """严格回读 canonical J-F2 preflight 报告。"""
    target = Path(path)
    payload = target.read_bytes()
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise JF2PreflightError("J-F2 preflight newline 非法")
    value = parse_canonical_json_bytes(payload[:-1], require_object=True)
    raw = _exact(value, {
        "artifact_kind", "artifact_version", "blockers", "dependencies",
        "format_version", "language_capability_mastered", "language_readiness",
        "status",
    }, where="JF2PreflightReport")
    report = JF2PreflightReport(
        raw["format_version"], str(raw["artifact_kind"]),
        str(raw["artifact_version"]), str(raw["status"]),
        tuple(JF2Dependency(
            str(item["role"]), str(item["relative_path"]), str(item["status"]),
            item["size_bytes"], str(item["sha256"]),
        ) for item in raw["dependencies"]),
        tuple(str(item) for item in raw["blockers"]),
        raw["language_capability_mastered"], raw["language_readiness"],
    )
    if report.canonical_bytes() != payload:
        raise JF2PreflightError("J-F2 preflight 非 canonical bytes")
    return report


__all__ = [
    "ARTIFACT_KIND", "ARTIFACT_VERSION", "CORE_ARTIFACT_PATH", "D03_RECEIPT_PATH",
    "FORMAT_VERSION", "JF2Dependency", "JF2PreflightError", "JF2PreflightReport",
    "J_F1_RECEIPT_PATH", "RECEIPT_BINDINGS", "W02_RECEIPT_SHA256",
    "build_jf2_preflight", "read_jf2_preflight",
]
