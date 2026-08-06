"""J-F2 最终联合封存的公开合同与一次性发布器。

本模块只消费公开 preflight 与公开 W09 receipt。它把依赖身份和承重证据
写入自排除 seal；不修改 readiness 所属对象，也不读取 Git 外封存材料。
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
from pathlib import Path, PurePosixPath
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_j_f2_contract import (
    ARTIFACT_KIND as PREFLIGHT_ARTIFACT_KIND,
    ARTIFACT_VERSION as PREFLIGHT_ARTIFACT_VERSION,
    RECEIPT_BINDINGS,
    W09_RECEIPT_PATH,
    JF2Dependency,
    JF2PreflightError,
    build_jf2_preflight,
)
from pure_integer_ai.experiments.ph2_w09_authority import (
    W09_ABLATION_KEYS,
    W09_DIMENSION_KEYS,
    W09_WALL_DIMENSION_KEYS,
)


FORMAT_VERSION = 1
ARTIFACT_KIND = "PH2_J_F2_FINAL_JOINT_SEAL"
ARTIFACT_VERSION = "J-F2-FINAL-JOINT-SEAL-20260806-A"
SEAL_PATH = "data/ph2/manifests/j_f2_final_joint_seal_v1.json"

_DEPENDENCY_ROLES = tuple(role for role, _ in RECEIPT_BINDINGS) + (
    "J_F1_FACILITY", "CORE_ARTIFACT")
_HARD_CONJUNCT_KEYS = _DEPENDENCY_ROLES + (
    "PRE_WEAN_LANGUAGE_LEARNING_CAPABILITY",
    "LANGUAGE_CAPABILITY_MASTERED",
    "W09_BEARING_DIMENSIONS",
    "W09_REAL_ABLATIONS",
    "W09_J_LC",
    "W09_OPEN_GENERATION",
    "W09_V06_CONTINUAL",
    "W09_ROLLBACK",
    "W09_RESOURCE_RESUME_WORKER",
    "W09_TEACHER_ZERO_WINDOWS",
    "W09_PROTECTED_WRITES",
    "W09_OWNER_BOUNDARY",
    "W09_PRIVATE_TERMINAL_COMMITMENT",
)
_OPEN_GENERATION_LAYERS = (
    "CONTENT_SEMANTICS", "STRUCTURE", "DISCOURSE_SCOPE",
    "MORPHOLOGY_SURFACE", "TASK_USE",
)
_WRITE_COUNT_KEYS = frozenset({
    "assessment_writes", "candidate_writes", "clock_writes", "core_writes",
    "evidence_writes", "host_writes", "label_writes", "memory_writes",
    "public_writes", "use_writes",
})


class FinalJointSealError(RuntimeError):
    """最终联合封存缺失、漂移、重复发布或硬合取不成立。"""


def _exact(value: Any, keys: set[str], *, where: str) -> dict[str, Any]:
    """要求对象字段集合精确，拒绝静默扩展或删减。"""
    if not isinstance(value, dict) or set(value) != keys:
        raise FinalJointSealError(f"{where} 字段不精确")
    return value


def _relative(value: str, *, where: str) -> str:
    """要求公开路径是仓库内 POSIX 相对路径。"""
    path = PurePosixPath(value)
    if (not isinstance(value, str) or not value or path.is_absolute()
            or ".." in path.parts or "\\" in value):
        raise FinalJointSealError(f"{where} 相对路径非法")
    return value


def _sha256(value: str, *, where: str) -> str:
    """要求摘要为小写 SHA-256。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise FinalJointSealError(f"{where} SHA-256 非法")
    return value


def _integer(value: Any, *, where: str, minimum: int = 0) -> int:
    """要求状态与计数使用有下界的离散整数。"""
    if type(value) is not int or value < minimum:
        raise FinalJointSealError(f"{where} 整数非法")
    return value


def _read_public_json(root: Path, relative_path: str) -> dict[str, Any]:
    """严格读取仓库内公开 canonical JSON，不接受路径越界。"""
    _relative(relative_path, where="公开 JSON")
    target = (root / Path(*relative_path.split("/"))).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise FinalJointSealError(f"公开 JSON 缺失: {relative_path}")
    payload = target.read_bytes()
    if payload.endswith(b"\n\n"):
        raise FinalJointSealError(f"公开 JSON newline 非法: {relative_path}")
    body = payload[:-1] if payload.endswith(b"\n") else payload
    try:
        value = parse_canonical_json_bytes(body, require_object=True)
    except Exception as error:
        raise FinalJointSealError(f"公开 JSON 非 canonical: {relative_path}") from error
    if not isinstance(value, dict) or canonical_json_bytes(value) != body:
        raise FinalJointSealError(f"公开 JSON bytes 漂移: {relative_path}")
    return value


def _resolve_target(root: Path, value: str | Path) -> Path:
    """解析发布目标；相对目标必须留在公开仓库内。"""
    target = Path(value)
    if target.is_absolute():
        return target
    relative_path = str(value).replace("\\", "/")
    _relative(relative_path, where="seal target")
    resolved = (root / Path(*relative_path.split("/"))).resolve()
    if not resolved.is_relative_to(root):
        raise FinalJointSealError("seal target 越界")
    return resolved


@dataclass(frozen=True, order=True)
class SealConjunct:
    """一个必须为 PASS 的最终联合硬合取。"""

    conjunct_key: str
    status: str

    def __post_init__(self) -> None:
        """拒绝空键和任何非 PASS 承重状态。"""
        if not self.conjunct_key or self.status != "PASS":
            raise FinalJointSealError("final seal 硬合取未通过")

    def to_dict(self) -> dict[str, str]:
        """返回硬合取的 canonical 表示。"""
        return {"conjunct_key": self.conjunct_key, "status": self.status}


def _validate_dimensions(value: Any) -> None:
    """核对五个 W09 承重维度的完整通过计数与证据摘要。"""
    if not isinstance(value, list) or len(value) != len(W09_DIMENSION_KEYS):
        raise FinalJointSealError("W09 bearing dimensions 数量漂移")
    for expected_key, item in zip(W09_DIMENSION_KEYS, value, strict=True):
        entry = _exact(item, {
            "dimension_key", "evidence_sha256", "passed_count",
            "required_count", "status",
        }, where="W09 bearing dimension")
        if (entry["dimension_key"] != expected_key or entry["status"] != "PASS"
                or _integer(entry["required_count"], where=expected_key, minimum=1)
                != _integer(entry["passed_count"], where=expected_key, minimum=1)):
            raise FinalJointSealError("W09 bearing dimension 未完整通过")
        _sha256(entry["evidence_sha256"], where=expected_key)


def _validate_ablations(value: Any) -> None:
    """核对五个真实消融为 PASS，并保留两个墙维为 NE。"""
    if not isinstance(value, list) or len(value) != len(W09_ABLATION_KEYS):
        raise FinalJointSealError("W09 ablation 数量漂移")
    for index, (expected_key, item) in enumerate(
            zip(W09_ABLATION_KEYS, value, strict=True)):
        entry = _exact(item, {
            "ablation_key", "invocation_count", "real_component_disabled",
            "status", "target_dimension_key",
        }, where="W09 ablation")
        expected_status = "PASS" if index < len(W09_DIMENSION_KEYS) else "NE"
        invocation = _integer(entry["invocation_count"], where=expected_key)
        if (entry["ablation_key"] != expected_key
                or entry["target_dimension_key"]
                != expected_key.removesuffix("-ABLATION")
                or entry["real_component_disabled"] != 1
                or entry["status"] != expected_status
                or (expected_status == "PASS" and invocation <= 0)
                or (expected_status == "NE" and invocation != 0)):
            raise FinalJointSealError("W09 ablation 状态漂移")


def _validate_w09_evidence(value: Any) -> None:
    """核对 seal 中的公开 W09 承重摘要，不接触原始评测 payload。"""
    evidence = _exact(value, {
        "ablations", "bearing_dimensions", "execution_state", "j_lc",
        "open_generation", "owner_boundary", "private_terminal_state",
        "resource", "rollback", "v06", "windows", "write_counts",
    }, where="W09 evidence")
    if evidence["execution_state"] != {
            "LANGUAGE_CAPABILITY_MASTERED": 1,
            "LANGUAGE_READINESS": 0,
            "PRE_WEAN_LANGUAGE_LEARNING_CAPABILITY_EVIDENCED": 1,
            "W09_BLOCKED_FAILED": 0,
            "W09_RUNTIME_EVIDENCED": 1,
            "formal_w09_training_runs": 1,
            "teacher_calls": 0,
    }:
        raise FinalJointSealError("W09 execution state 漂移")
    _validate_dimensions(evidence["bearing_dimensions"])
    _validate_ablations(evidence["ablations"])

    j_lc = _exact(evidence["j_lc"], {
        "bearing_cell_count", "lc_task_count",
        "retention_continual_learning_cell_count", "status",
        "wall_dimension_states",
    }, where="W09 J-LC")
    if (j_lc["status"] != "PASS" or j_lc["bearing_cell_count"] != 216
            or j_lc["lc_task_count"] != 16
            or j_lc["retention_continual_learning_cell_count"] != 27
            or j_lc["wall_dimension_states"]
            != [[key, "NE"] for key in W09_WALL_DIMENSION_KEYS]):
        raise FinalJointSealError("W09 J-LC 或墙维状态漂移")

    generation = _exact(evidence["open_generation"], {
        "complete_template_replay_count", "exact_surface_read_count",
        "layer_states", "output_invocation_count", "source_replay_count",
        "status",
    }, where="W09 open generation")
    if (generation["status"] != "PASS"
            or generation["layer_states"]
            != [[key, "PASS"] for key in _OPEN_GENERATION_LAYERS]
            or _integer(generation["output_invocation_count"],
                        where="open generation", minimum=1) <= 0
            or any(generation[key] != 0 for key in (
                "complete_template_replay_count", "exact_surface_read_count",
                "source_replay_count"))):
        raise FinalJointSealError("W09 open generation 漂移")

    if evidence["v06"] != {
            "core_bit_identical": 1, "host_write_count": 0,
            "improved_probe_count": 309, "independent_probe_count": 309,
            "isolated_learning_write_count": 27, "status": "PASS",
    }:
        raise FinalJointSealError("W09 V-06 漂移")
    if evidence["rollback"] != {
            "invalidated_count": 3, "leaked_write_count": 0,
            "preserved_count": 27, "status": "PASS",
    }:
        raise FinalJointSealError("W09 rollback 漂移")
    if evidence["resource"] != {
            "fresh_resume_equivalent": 1, "status": "PASS",
            "worker_1_2_4_invariant": 1,
    }:
        raise FinalJointSealError("W09 resource 漂移")
    if evidence["windows"] != [
            {"status": "PASS", "teacher_calls": 0, "window_ordinal": index}
            for index in range(1, 4)]:
        raise FinalJointSealError("W09 teacher-zero windows 漂移")
    writes = evidence["write_counts"]
    if (not isinstance(writes, dict) or set(writes) != _WRITE_COUNT_KEYS
            or any(type(item) is not int or item != 0 for item in writes.values())):
        raise FinalJointSealError("W09 protected write 非零")
    if evidence["owner_boundary"] != {
            "companion_writes": 0, "evaluator_label_writes": 0,
            "formal_training_runs": 1, "host_learning_writes": 0,
            "memory_learning_writes": 0, "readback_payload_gets": 0,
            "teacher_calls": 0,
    }:
        raise FinalJointSealError("W09 owner boundary 漂移")
    if evidence["private_terminal_state"] != "PASS":
        raise FinalJointSealError("W09 public terminal commitment 未通过")


@dataclass(frozen=True)
class FinalJointSeal:
    """唯一公开 J-F2 final seal 的完整、可回验内容。"""

    format_version: int
    artifact_kind: str
    artifact_version: str
    seal_relative_path: str
    seal_self_excluded: int
    status: str
    preflight_identity: dict[str, Any]
    dependency_bindings: tuple[JF2Dependency, ...]
    hard_conjuncts: tuple[SealConjunct, ...]
    w09_evidence: dict[str, Any]
    publication_policy: dict[str, Any]
    readiness_transition: dict[str, Any]

    def __post_init__(self) -> None:
        """强制自排除、全 PASS、墙维 NE 和唯一受控 readiness 转换。"""
        if (self.format_version != FORMAT_VERSION
                or self.artifact_kind != ARTIFACT_KIND
                or self.artifact_version != ARTIFACT_VERSION
                or self.status != "SEALED"):
            raise FinalJointSealError("final seal artifact identity 漂移")
        if (self.seal_relative_path != SEAL_PATH
                or self.seal_self_excluded != 1):
            raise FinalJointSealError("final seal 自排除边界漂移")
        preflight = _exact(self.preflight_identity, {
            "artifact_kind", "artifact_version", "blocker_count",
            "report_sha256", "status",
        }, where="preflight identity")
        if (preflight["artifact_kind"] != PREFLIGHT_ARTIFACT_KIND
                or preflight["artifact_version"] != PREFLIGHT_ARTIFACT_VERSION
                or preflight["status"] != "READY_FOR_FORMAL_SEAL"
                or preflight["blocker_count"] != 0):
            raise FinalJointSealError("preflight 未准备正式封存")
        _sha256(preflight["report_sha256"], where="preflight report")
        if (tuple(item.role for item in self.dependency_bindings)
                != _DEPENDENCY_ROLES
                or any(item.status != "PASS" for item in self.dependency_bindings)):
            raise FinalJointSealError("final seal dependency 未全部通过")
        if (tuple(item.conjunct_key for item in self.hard_conjuncts)
                != _HARD_CONJUNCT_KEYS
                or any(item.status != "PASS" for item in self.hard_conjuncts)):
            raise FinalJointSealError("final seal 硬合取漂移")
        _validate_w09_evidence(self.w09_evidence)
        if self.publication_policy != {
                "append_only": 1,
                "dependency_reads_only": 1,
                "exclusive_create": 1,
                "seal_self_excluded": 1,
        }:
            raise FinalJointSealError("final seal 发布策略漂移")
        if self.readiness_transition != {
                "LANGUAGE_CAPABILITY_MASTERED": 1,
                "LANGUAGE_READINESS_AFTER_EXCLUSIVE_PUBLICATION": 1,
                "LANGUAGE_READINESS_BEFORE_PUBLICATION": 0,
                "PW00A_STARTED": 0,
                "can_ween_language_modified": 0,
        }:
            raise FinalJointSealError("final seal readiness 转换非法")

    def to_dict(self) -> dict[str, Any]:
        """返回公开、不含原始评测 payload 的 canonical seal。"""
        return {
            "artifact_kind": self.artifact_kind,
            "artifact_version": self.artifact_version,
            "dependency_bindings": [item.to_dict() for item in self.dependency_bindings],
            "format_version": self.format_version,
            "hard_conjuncts": [item.to_dict() for item in self.hard_conjuncts],
            "preflight_identity": deepcopy(self.preflight_identity),
            "publication_policy": deepcopy(self.publication_policy),
            "readiness_transition": deepcopy(self.readiness_transition),
            "seal_relative_path": self.seal_relative_path,
            "seal_self_excluded": self.seal_self_excluded,
            "status": self.status,
            "w09_evidence": deepcopy(self.w09_evidence),
        }

    def canonical_bytes(self) -> bytes:
        """返回带单尾换行的 canonical JSON。"""
        return canonical_json_bytes(self.to_dict()) + b"\n"

    def sha256(self) -> str:
        """返回 final seal 的稳定 SHA-256。"""
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


def _build_w09_evidence(value: dict[str, Any]) -> dict[str, Any]:
    """从公开 receipt 提取固定承重字段，不复制原始 case 或 payload。"""
    execution = value.get("execution_state", {})
    dimensions = value.get("dimension_results", [])
    ablations = value.get("ablation_results", [])
    candidate = value.get("candidate_evidence", {})
    return {
        "ablations": [{
            key: item[key] for key in (
                "ablation_key", "invocation_count", "real_component_disabled",
                "status", "target_dimension_key")
        } for item in ablations],
        "bearing_dimensions": [{
            key: item[key] for key in (
                "dimension_key", "evidence_sha256", "passed_count",
                "required_count", "status")
        } for item in dimensions],
        "execution_state": {key: execution.get(key) for key in (
            "LANGUAGE_CAPABILITY_MASTERED", "LANGUAGE_READINESS",
            "PRE_WEAN_LANGUAGE_LEARNING_CAPABILITY_EVIDENCED",
            "W09_BLOCKED_FAILED", "W09_RUNTIME_EVIDENCED",
            "formal_w09_training_runs", "teacher_calls")},
        "j_lc": {key: value.get("j_lc", {}).get(key) for key in (
            "bearing_cell_count", "lc_task_count",
            "retention_continual_learning_cell_count", "status",
            "wall_dimension_states")},
        "open_generation": {key: value.get("open_generation", {}).get(key)
                            for key in (
            "complete_template_replay_count", "exact_surface_read_count",
            "layer_states", "output_invocation_count", "source_replay_count",
            "status")},
        "owner_boundary": {key: candidate.get("owner_write_counts", {}).get(key)
                           for key in (
            "companion_writes", "evaluator_label_writes", "formal_training_runs",
            "host_learning_writes", "memory_learning_writes",
            "readback_payload_gets", "teacher_calls")},
        "private_terminal_state": value.get("private_evidence", {}).get(
            "terminal_state"),
        "resource": {key: value.get("resource", {}).get(key) for key in (
            "fresh_resume_equivalent", "status", "worker_1_2_4_invariant")},
        "rollback": dict(value.get("rollback", {})),
        "v06": dict(value.get("v06", {})),
        "windows": [dict(item) for item in value.get("windows", [])],
        "write_counts": dict(value.get("write_counts", {})),
    }


def build_final_joint_seal(repository_root: str | Path) -> FinalJointSeal:
    """执行只读最终合取，并构建尚未产生全局状态副作用的 seal 内容。"""
    root = Path(repository_root).resolve()
    preflight = build_jf2_preflight(root)
    if (preflight.status != "READY_FOR_FORMAL_SEAL" or preflight.blockers
            or preflight.language_capability_mastered != 1
            or preflight.language_readiness != 0
            or any(item.status != "PASS" for item in preflight.dependencies)):
        raise FinalJointSealError("J-F2 preflight 未满足正式封存条件")
    w09 = _read_public_json(root, W09_RECEIPT_PATH)
    return FinalJointSeal(
        FORMAT_VERSION, ARTIFACT_KIND, ARTIFACT_VERSION, SEAL_PATH, 1, "SEALED",
        {
            "artifact_kind": preflight.artifact_kind,
            "artifact_version": preflight.artifact_version,
            "blocker_count": len(preflight.blockers),
            "report_sha256": preflight.sha256(),
            "status": preflight.status,
        },
        preflight.dependencies,
        tuple(SealConjunct(key, "PASS") for key in _HARD_CONJUNCT_KEYS),
        _build_w09_evidence(w09),
        {
            "append_only": 1,
            "dependency_reads_only": 1,
            "exclusive_create": 1,
            "seal_self_excluded": 1,
        },
        {
            "LANGUAGE_CAPABILITY_MASTERED": 1,
            "LANGUAGE_READINESS_AFTER_EXCLUSIVE_PUBLICATION": 1,
            "LANGUAGE_READINESS_BEFORE_PUBLICATION": 0,
            "PW00A_STARTED": 0,
            "can_ween_language_modified": 0,
        },
    )


def read_final_joint_seal(
        repository_root: str | Path,
        path: str | Path = SEAL_PATH,
        *, verify_dependencies: bool = True) -> FinalJointSeal:
    """严格回读 canonical seal，并可重新执行全部公开依赖合取。"""
    root = Path(repository_root).resolve()
    target = _resolve_target(root, path)
    payload = target.read_bytes()
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise FinalJointSealError("final seal newline 非法")
    try:
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
    except Exception as error:
        raise FinalJointSealError("final seal JSON 非 canonical") from error
    raw = _exact(value, {
        "artifact_kind", "artifact_version", "dependency_bindings",
        "format_version", "hard_conjuncts", "preflight_identity",
        "publication_policy", "readiness_transition", "seal_relative_path",
        "seal_self_excluded", "status", "w09_evidence",
    }, where="FinalJointSeal")
    dependencies = []
    for item in raw["dependency_bindings"]:
        entry = _exact(item, {
            "relative_path", "role", "sha256", "size_bytes", "status",
        }, where="JF2Dependency")
        try:
            dependencies.append(JF2Dependency(
                str(entry["role"]), str(entry["relative_path"]),
                str(entry["status"]), entry["size_bytes"],
                str(entry["sha256"])))
        except JF2PreflightError as error:
            raise FinalJointSealError("final seal dependency binding 非法") from error
    conjuncts = []
    for item in raw["hard_conjuncts"]:
        entry = _exact(item, {"conjunct_key", "status"}, where="SealConjunct")
        conjuncts.append(SealConjunct(
            str(entry["conjunct_key"]), str(entry["status"])))
    seal = FinalJointSeal(
        raw["format_version"], str(raw["artifact_kind"]),
        str(raw["artifact_version"]), str(raw["seal_relative_path"]),
        raw["seal_self_excluded"], str(raw["status"]),
        dict(raw["preflight_identity"]), tuple(dependencies), tuple(conjuncts),
        dict(raw["w09_evidence"]), dict(raw["publication_policy"]),
        dict(raw["readiness_transition"]),
    )
    if seal.canonical_bytes() != payload:
        raise FinalJointSealError("final seal canonical bytes 漂移")
    if verify_dependencies and seal != build_final_joint_seal(root):
        raise FinalJointSealError("final seal 公开依赖身份漂移")
    return seal


def publish_final_joint_seal(
        repository_root: str | Path,
        *, target: str | Path = SEAL_PATH) -> FinalJointSeal:
    """以排他创建首次发布 final seal，目标已存在时不重跑 preflight。"""
    root = Path(repository_root).resolve()
    destination = _resolve_target(root, target)
    if destination.exists():
        raise FinalJointSealError("final seal 已发布，禁止覆盖")
    seal = build_final_joint_seal(root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as handle:
            handle.write(seal.canonical_bytes())
    except FileExistsError as error:
        raise FinalJointSealError("final seal 已发布，禁止覆盖") from error
    restored = read_final_joint_seal(root, destination, verify_dependencies=False)
    if restored != seal:
        raise FinalJointSealError("final seal 发布回读不一致")
    return restored


__all__ = [
    "ARTIFACT_KIND", "ARTIFACT_VERSION", "FORMAT_VERSION", "SEAL_PATH",
    "FinalJointSeal", "FinalJointSealError", "SealConjunct",
    "build_final_joint_seal", "publish_final_joint_seal",
    "read_final_joint_seal",
]
