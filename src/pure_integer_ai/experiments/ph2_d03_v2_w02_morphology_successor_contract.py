"""W-02 morphology successor 的 runtime freeze 与首次 transform guard。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path, PurePosixPath
from typing import Any

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
    exact_dict,
    read_canonical_object,
    write_immutable_json,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_candidate_publication import (
    W02_CANDIDATE_RECEIPT_PATH,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_overlay import (
    W02_MORPH_OVERLAY_OPERATION,
    W02MorphologyOverlayBudget,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor import (
    W02_MORPH_SUCCESSOR_VERSION,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_runtime_contract import (
    read_w02_candidate_runtime_freeze,
)


W02_MORPH_SUCCESSOR_FREEZE_VERSION = (
    "PH2-D03-V2-W02-MORPHOLOGY-SUCCESSOR-RUNTIME-FREEZE-V1")
W02_MORPH_SUCCESSOR_FREEZE_PATH = (
    "data/ph2/manifests/d03_v2/stages/"
    "ph2_d03_v2_w02_morphology_successor_runtime_freeze_v1.json"
)
W02_MORPH_SUCCESSOR_DEV_FAILURE_PATH = (
    "data/ph2/manifests/d03_v2/stages/"
    "ph2_d03_v2_w02_dev_calibration_report_v1.json"
)
W02_MORPH_SUCCESSOR_GUARD_VERSION = (
    "PH2-D03-V2-W02-MORPHOLOGY-SUCCESSOR-FIRST-RUN-GUARD-V1")
W02_MORPH_SUCCESSOR_GUARD_AVAILABLE = "run-guard/available.guard.json"
W02_MORPH_SUCCESSOR_GUARD_CONSUMED = "run-guard/consumed.guard.json"
W02_MORPH_SUCCESSOR_RUN_INTENT = "run-guard/run-intent.json"
W02_MORPH_SUCCESSOR_EXPECTED_SEMANTIC = (
    "a72c0ccb0d054537c04ac3e683a5730a7e81b767339957e2957e1cc172de0676")
W02_MORPH_SUCCESSOR_PREFLIGHT_MANIFEST = (
    "01a08e1e2e40bd413cf949e60aed7002bcfc658dd35b1031101799d7cfd993a5")
W02_MORPH_SUCCESSOR_CODE_PATHS = (
    "src/pure_integer_ai/experiments/ph2_d03_v2_w02_morphology_successor.py",
    "src/pure_integer_ai/experiments/ph2_d03_v2_w02_morphology_overlay.py",
    "src/pure_integer_ai/experiments/"
    "ph2_d03_v2_w02_morphology_successor_contract.py",
    "src/pure_integer_ai/experiments/"
    "ph2_d03_v2_w02_morphology_successor_publication.py",
    "src/pure_integer_ai/experiments/"
    "run_ph2_d03_v2_w02_morphology_successor.py",
    "tests/test_ph2_d03_v2_w02_morphology_successor.py",
    "tests/test_ph2_d03_v2_w02_morphology_successor_contract.py",
)


# object-model: exception
class W02MorphologySuccessorContractError(RuntimeError):
    """successor freeze、live code 或 guard 发生漂移。"""


def _sha256(value: object, *, where: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise W02MorphologySuccessorContractError(f"{where} 不是小写 SHA-256")
    return value


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
            size += len(block)
    return size, digest.hexdigest()


def _repository_file(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if (not relative or "\\" in relative or pure.is_absolute()
            or pure.as_posix() != relative or ".." in pure.parts):
        raise W02MorphologySuccessorContractError("successor code path 非法")
    current = root
    for part in pure.parts:
        current /= part
        if current.is_symlink():
            raise W02MorphologySuccessorContractError(
                "successor code path 经过 symlink")
    target = (root / Path(*pure.parts)).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise W02MorphologySuccessorContractError("successor code file 缺失")
    return target


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W02MorphologySuccessorCodeFile:
    """一个 successor runtime 承重文件的公开字节身份。"""

    repository_path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if self.repository_path not in W02_MORPH_SUCCESSOR_CODE_PATHS:
            raise W02MorphologySuccessorContractError("successor code file 未注册")
        if type(self.size_bytes) is not int or self.size_bytes <= 0:
            raise W02MorphologySuccessorContractError("successor code size 非法")
        _sha256(self.sha256, where="successor code SHA")

    def to_dict(self) -> dict[str, object]:
        return {
            "repository_path": self.repository_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "W02MorphologySuccessorCodeFile":
        raw = exact_dict(
            value, {"repository_path", "sha256", "size_bytes"},
            where="W02MorphologySuccessorCodeFile")
        return cls(
            str(raw["repository_path"]), raw["size_bytes"], str(raw["sha256"]))


def _code_freeze(
        repository: Path,
        ) -> tuple[tuple[W02MorphologySuccessorCodeFile, ...], str]:
    files = tuple(
        W02MorphologySuccessorCodeFile(relative, *_sha256_file(
            _repository_file(repository, relative)))
        for relative in W02_MORPH_SUCCESSOR_CODE_PATHS
    )
    commitment = hashlib.sha256(canonical_json_bytes(
        [item.to_dict() for item in files])).hexdigest()
    return files, commitment


def w02_morphology_successor_guard_value(
        *,
        runtime_code_freeze_sha256: str,
        parent_candidate_manifest_sha256: str,
        expected_overlay_semantic_sha256: str,
        ) -> dict[str, object]:
    """构造一次性 transform guard 的规范初始值。"""
    return {
        "artifact_kind": "PH2_D03_V2_W02_MORPHOLOGY_SUCCESSOR_FIRST_RUN_GUARD",
        "expected_overlay_semantic_sha256": _sha256(
            expected_overlay_semantic_sha256,
            where="successor guard expected semantic"),
        "formal_successor_transform_runs": 0,
        "formal_training_runs": 0,
        "format_version": 1,
        "guard_consumed": 0,
        "guard_version": W02_MORPH_SUCCESSOR_GUARD_VERSION,
        "operation_kind": W02_MORPH_OVERLAY_OPERATION,
        "parent_candidate_manifest_sha256": _sha256(
            parent_candidate_manifest_sha256,
            where="successor guard parent manifest"),
        "release_key": "PH2-D03-V2",
        "run_id_policy": "NEW_POSITIVE_INTEGER_REQUIRED",
        "runtime_code_freeze_sha256": _sha256(
            runtime_code_freeze_sha256,
            where="successor guard code freeze"),
        "stage_key": "W-02",
        "status": "AVAILABLE",
    }


def _guard_sha(value: dict[str, object]) -> str:
    return hashlib.sha256(canonical_json_bytes(value) + b"\n").hexdigest()


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W02MorphologySuccessorRuntimeFreeze:
    """唯一 formal successor transform 之前的公开冻结。"""

    parent_candidate_receipt_file_sha256: str
    parent_candidate_runtime_freeze_sha256: str
    parent_candidate_manifest_sha256: str
    parent_candidate_semantic_sha256: str
    parent_dev_failure_report_file_sha256: str
    preflight_manifest_sha256: str
    expected_overlay_semantic_sha256: str
    code_files: tuple[W02MorphologySuccessorCodeFile, ...]
    runtime_code_freeze_sha256: str
    first_run_guard_sha256: str
    resource_budget: W02MorphologyOverlayBudget

    def __post_init__(self) -> None:
        for name in (
                "parent_candidate_receipt_file_sha256",
                "parent_candidate_runtime_freeze_sha256",
                "parent_candidate_manifest_sha256",
                "parent_candidate_semantic_sha256",
                "parent_dev_failure_report_file_sha256",
                "preflight_manifest_sha256",
                "expected_overlay_semantic_sha256",
                "runtime_code_freeze_sha256", "first_run_guard_sha256"):
            _sha256(getattr(self, name), where=f"successor freeze {name}")
        if tuple(item.repository_path for item in self.code_files) != (
                W02_MORPH_SUCCESSOR_CODE_PATHS):
            raise W02MorphologySuccessorContractError(
                "successor code inventory 漂移")
        expected_code = hashlib.sha256(canonical_json_bytes(
            [item.to_dict() for item in self.code_files])).hexdigest()
        if expected_code != self.runtime_code_freeze_sha256:
            raise W02MorphologySuccessorContractError(
                "successor code commitment 漂移")
        expected_guard = _guard_sha(w02_morphology_successor_guard_value(
            runtime_code_freeze_sha256=self.runtime_code_freeze_sha256,
            parent_candidate_manifest_sha256=(
                self.parent_candidate_manifest_sha256),
            expected_overlay_semantic_sha256=(
                self.expected_overlay_semantic_sha256),
        ))
        if expected_guard != self.first_run_guard_sha256:
            raise W02MorphologySuccessorContractError("successor guard SHA 漂移")
        if not isinstance(self.resource_budget, W02MorphologyOverlayBudget):
            raise W02MorphologySuccessorContractError("successor budget 类型错误")

    def to_dict(self) -> dict[str, object]:
        return {
            "allowed_workers": [1, 2, 4],
            "artifact_kind": "PH2_D03_V2_W02_MORPHOLOGY_SUCCESSOR_RUNTIME_FREEZE",
            "artifact_version": W02_MORPH_SUCCESSOR_FREEZE_VERSION,
            "candidate_writes": 0,
            "code_files": [item.to_dict() for item in self.code_files],
            "expected_counts": {
                "logic_operations": 1_574_251,
                "max_form_length": 17,
                "morphology_observation_count": 3_997,
                "morphology_token_count": 97_959,
                "rule_row_count": 47_975,
                "training_pair_count": 51_200,
            },
            "expected_overlay_semantic_sha256": (
                self.expected_overlay_semantic_sha256),
            "first_run_guard_sha256": self.first_run_guard_sha256,
            "formal_private_evaluation_runs": 0,
            "formal_successor_transform_runs": 0,
            "formal_training_runs": 0,
            "language_capability_mastered": 0,
            "language_readiness": 0,
            "logical_shard_count": 128,
            "manifest_last_required": 1,
            "next_action": "W02_FORMAL_MORPHOLOGY_SUCCESSOR_TRANSFORM",
            "operation_kind": W02_MORPH_OVERLAY_OPERATION,
            "parent_candidate_manifest_sha256": (
                self.parent_candidate_manifest_sha256),
            "parent_candidate_receipt_file_sha256": (
                self.parent_candidate_receipt_file_sha256),
            "parent_candidate_runtime_freeze_sha256": (
                self.parent_candidate_runtime_freeze_sha256),
            "parent_candidate_semantic_sha256": (
                self.parent_candidate_semantic_sha256),
            "parent_dev_failure_report_file_sha256": (
                self.parent_dev_failure_report_file_sha256),
            "parent_dev_status": "FAIL",
            "parent_formal_training_runs": 1,
            "preflight_manifest_sha256": self.preflight_manifest_sha256,
            "private_family_registered": 0,
            "private_payload_reads": 0,
            "release_key": "PH2-D03-V2",
            "resource_budget": {
                "max_checkpoint_count": self.resource_budget.max_checkpoint_count,
                "max_input_rows": self.resource_budget.max_input_rows,
                "max_logic_operations": self.resource_budget.max_logic_operations,
                "max_payload_bytes": self.resource_budget.max_payload_bytes,
                "max_rule_rows": self.resource_budget.max_rule_rows,
                "max_shard_delta_bytes": (
                    self.resource_budget.max_shard_delta_bytes),
            },
            "runtime_code_freeze_sha256": self.runtime_code_freeze_sha256,
            "shadow_started": 0,
            "stage_key": "W-02",
            "status": "W02_MORPHOLOGY_SUCCESSOR_RUNTIME_FREEZE_COMPLETE",
            "successor_version": W02_MORPH_SUCCESSOR_VERSION,
            "teacher_calls": 0,
            "worker_canonical_equivalence_required": 1,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict()) + b"\n"

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


def build_w02_morphology_successor_runtime_freeze(
        repository_root: str | Path,
        ) -> W02MorphologySuccessorRuntimeFreeze:
    """回验旧 Candidate/FAIL 和 live code 后构造 successor freeze。"""
    repository = Path(repository_root).resolve()
    parent_runtime = read_w02_candidate_runtime_freeze(repository)
    receipt_path = _repository_file(repository, W02_CANDIDATE_RECEIPT_PATH)
    report_path = _repository_file(repository, W02_MORPH_SUCCESSOR_DEV_FAILURE_PATH)
    receipt = read_canonical_object(receipt_path)
    report = read_canonical_object(report_path)
    if (receipt.get("status") != "W02_CANDIDATE_ARTIFACT_FROZEN"
            or receipt.get("formal_training_runs") != 1
            or receipt.get("formal_private_evaluation_runs") != 0
            or receipt.get("private_payload_reads") != 0
            or receipt.get("teacher_calls") != 0
            or receipt.get("runtime_freeze_sha256") != parent_runtime.sha256()):
        raise W02MorphologySuccessorContractError(
            "successor parent Candidate receipt 非法")
    dimensions = report.get("dimension_results")
    if (not isinstance(dimensions, list)
            or any(not isinstance(item, dict) for item in dimensions)):
        raise W02MorphologySuccessorContractError(
            "successor parent dev dimensions 类型非法")
    morphology = tuple(
        item for item in dimensions
        if item.get("dimension_key") == "W-02-V2-NEW-CONTENT-MORPHOLOGY")
    if (report.get("status") != "FAIL" or len(morphology) != 1
            or morphology[0].get("failed") != 90
            or morphology[0].get("status") != "FAIL"
            or report.get("formal_private_evaluation_runs") != 0
            or report.get("private_payload_reads") != 0
            or report.get("candidate_artifact_manifest_sha256")
            != receipt.get("candidate_artifact_manifest_sha256")
            or report.get("candidate_semantic_sha256")
            != receipt.get("candidate_semantic_sha256")):
        raise W02MorphologySuccessorContractError(
            "successor parent dev FAIL evidence 漂移")
    files, code_sha = _code_freeze(repository)
    guard = w02_morphology_successor_guard_value(
        runtime_code_freeze_sha256=code_sha,
        parent_candidate_manifest_sha256=str(
            receipt["candidate_artifact_manifest_sha256"]),
        expected_overlay_semantic_sha256=W02_MORPH_SUCCESSOR_EXPECTED_SEMANTIC,
    )
    return W02MorphologySuccessorRuntimeFreeze(
        _sha256_file(receipt_path)[1], parent_runtime.sha256(),
        str(receipt["candidate_artifact_manifest_sha256"]),
        str(receipt["candidate_semantic_sha256"]),
        _sha256_file(report_path)[1], W02_MORPH_SUCCESSOR_PREFLIGHT_MANIFEST,
        W02_MORPH_SUCCESSOR_EXPECTED_SEMANTIC, files, code_sha,
        _guard_sha(guard), W02MorphologyOverlayBudget(),
    )


def publish_w02_morphology_successor_runtime_freeze(
        repository_root: str | Path,
        ) -> Path:
    """不可覆盖发布 successor runtime freeze。"""
    repository = Path(repository_root).resolve()
    freeze = build_w02_morphology_successor_runtime_freeze(repository)
    target = repository / Path(*PurePosixPath(
        W02_MORPH_SUCCESSOR_FREEZE_PATH).parts)
    write_immutable_json(freeze.to_dict(), target)
    if target.read_bytes() != freeze.canonical_bytes():
        raise W02MorphologySuccessorContractError("successor freeze 发布字节漂移")
    return target


def read_w02_morphology_successor_runtime_freeze(
        repository_root: str | Path,
        ) -> W02MorphologySuccessorRuntimeFreeze:
    """严格回读 freeze，并重算旧证据与当前 live code。"""
    repository = Path(repository_root).resolve()
    target = repository / Path(*PurePosixPath(
        W02_MORPH_SUCCESSOR_FREEZE_PATH).parts)
    raw = exact_dict(read_canonical_object(target), {
        "allowed_workers", "artifact_kind", "artifact_version",
        "candidate_writes", "code_files", "expected_counts",
        "expected_overlay_semantic_sha256", "first_run_guard_sha256",
        "formal_private_evaluation_runs", "formal_successor_transform_runs",
        "formal_training_runs", "language_capability_mastered",
        "language_readiness", "logical_shard_count", "manifest_last_required",
        "next_action", "operation_kind", "parent_candidate_manifest_sha256",
        "parent_candidate_receipt_file_sha256",
        "parent_candidate_runtime_freeze_sha256",
        "parent_candidate_semantic_sha256",
        "parent_dev_failure_report_file_sha256", "parent_dev_status",
        "parent_formal_training_runs", "preflight_manifest_sha256",
        "private_family_registered", "private_payload_reads", "release_key",
        "resource_budget", "runtime_code_freeze_sha256", "shadow_started",
        "stage_key", "status", "successor_version", "teacher_calls",
        "worker_canonical_equivalence_required",
    }, where="W02MorphologySuccessorRuntimeFreeze")
    if (raw["artifact_kind"]
            != "PH2_D03_V2_W02_MORPHOLOGY_SUCCESSOR_RUNTIME_FREEZE"
            or raw["artifact_version"] != W02_MORPH_SUCCESSOR_FREEZE_VERSION
            or raw["release_key"] != "PH2-D03-V2" or raw["stage_key"] != "W-02"
            or raw["operation_kind"] != W02_MORPH_OVERLAY_OPERATION
            or raw["successor_version"] != W02_MORPH_SUCCESSOR_VERSION
            or raw["status"]
            != "W02_MORPHOLOGY_SUCCESSOR_RUNTIME_FREEZE_COMPLETE"
            or raw["next_action"]
            != "W02_FORMAL_MORPHOLOGY_SUCCESSOR_TRANSFORM"
            or raw["allowed_workers"] != [1, 2, 4]
            or raw["logical_shard_count"] != 128
            or raw["parent_dev_status"] != "FAIL"
            or raw["parent_formal_training_runs"] != 1):
        raise W02MorphologySuccessorContractError("successor freeze 顶层身份漂移")
    zeros = (
        "candidate_writes", "formal_private_evaluation_runs",
        "formal_successor_transform_runs", "formal_training_runs",
        "language_capability_mastered", "language_readiness",
        "private_family_registered", "private_payload_reads",
        "shadow_started", "teacher_calls",
    )
    if (any(raw[key] != 0 for key in zeros)
            or raw["manifest_last_required"] != 1
            or raw["worker_canonical_equivalence_required"] != 1
            or raw["expected_counts"] != {
                "logic_operations": 1_574_251,
                "max_form_length": 17,
                "morphology_observation_count": 3_997,
                "morphology_token_count": 97_959,
                "rule_row_count": 47_975,
                "training_pair_count": 51_200,
            }):
        raise W02MorphologySuccessorContractError("successor freeze 状态/计数漂移")
    files_raw = raw["code_files"]
    if not isinstance(files_raw, list):
        raise W02MorphologySuccessorContractError("successor code files 非 list")
    budget_raw = exact_dict(raw["resource_budget"], {
        "max_checkpoint_count", "max_input_rows", "max_logic_operations",
        "max_payload_bytes", "max_rule_rows", "max_shard_delta_bytes",
    }, where="W02MorphologyOverlayBudget")
    freeze = W02MorphologySuccessorRuntimeFreeze(
        str(raw["parent_candidate_receipt_file_sha256"]),
        str(raw["parent_candidate_runtime_freeze_sha256"]),
        str(raw["parent_candidate_manifest_sha256"]),
        str(raw["parent_candidate_semantic_sha256"]),
        str(raw["parent_dev_failure_report_file_sha256"]),
        str(raw["preflight_manifest_sha256"]),
        str(raw["expected_overlay_semantic_sha256"]),
        tuple(W02MorphologySuccessorCodeFile.from_dict(item) for item in files_raw),
        str(raw["runtime_code_freeze_sha256"]),
        str(raw["first_run_guard_sha256"]),
        W02MorphologyOverlayBudget(**budget_raw),
    )
    current = build_w02_morphology_successor_runtime_freeze(repository)
    if freeze != current or target.read_bytes() != freeze.canonical_bytes():
        raise W02MorphologySuccessorContractError(
            "successor freeze 与 live code/parent evidence 漂移")
    return freeze


def publish_w02_morphology_successor_guard(
        successor_root: str | Path,
        freeze: W02MorphologySuccessorRuntimeFreeze,
        ) -> str:
    """在全新正式 root 独占发布 successor first-run guard。"""
    if not isinstance(freeze, W02MorphologySuccessorRuntimeFreeze):
        raise W02MorphologySuccessorContractError("successor guard freeze 类型错误")
    root = Path(successor_root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    target = root / Path(*PurePosixPath(
        W02_MORPH_SUCCESSOR_GUARD_AVAILABLE).parts)
    value = w02_morphology_successor_guard_value(
        runtime_code_freeze_sha256=freeze.runtime_code_freeze_sha256,
        parent_candidate_manifest_sha256=freeze.parent_candidate_manifest_sha256,
        expected_overlay_semantic_sha256=freeze.expected_overlay_semantic_sha256,
    )
    write_immutable_json(value, target)
    payload = target.read_bytes()
    if hashlib.sha256(payload).hexdigest() != freeze.first_run_guard_sha256:
        raise W02MorphologySuccessorContractError("successor guard 发布 SHA 漂移")
    return freeze.first_run_guard_sha256


def consume_w02_morphology_successor_guard(
        successor_root: str | Path,
        *,
        expected_guard_sha256: str,
        run_id: int,
        run_identity_sha256: str,
        ) -> None:
    """在任何 formal overlay store 写入前原子消费 guard。"""
    _sha256(expected_guard_sha256, where="successor expected guard")
    _sha256(run_identity_sha256, where="successor run identity")
    if type(run_id) is not int or run_id <= 0:
        raise W02MorphologySuccessorContractError("successor run_id 非法")
    root = Path(successor_root).resolve()
    available = root / Path(*PurePosixPath(
        W02_MORPH_SUCCESSOR_GUARD_AVAILABLE).parts)
    consumed = root / Path(*PurePosixPath(
        W02_MORPH_SUCCESSOR_GUARD_CONSUMED).parts)
    intent = root / Path(*PurePosixPath(W02_MORPH_SUCCESSOR_RUN_INTENT).parts)
    if consumed.exists() or intent.exists() or not available.is_file():
        raise W02MorphologySuccessorContractError(
            "successor guard 不可用或已消费")
    if _sha256_file(available)[1] != expected_guard_sha256:
        raise W02MorphologySuccessorContractError("successor guard 字节漂移")
    os.replace(available, consumed)
    write_immutable_json({
        "artifact_kind": "PH2_D03_V2_W02_MORPHOLOGY_SUCCESSOR_RUN_INTENT",
        "formal_successor_transform_runs": 1,
        "formal_training_runs": 0,
        "format_version": 1,
        "guard_sha256": expected_guard_sha256,
        "operation_kind": W02_MORPH_OVERLAY_OPERATION,
        "run_id": run_id,
        "run_identity_sha256": run_identity_sha256,
        "stage_key": "W-02",
        "status": "GUARD_CONSUMED_BEFORE_SUCCESSOR_TRANSFORM_WRITE",
    }, intent)


def verify_w02_morphology_successor_consumed_guard(
        successor_root: str | Path,
        *,
        expected_guard_sha256: str,
        run_id: int,
        run_identity_sha256: str,
        ) -> None:
    """严格回读 consumed guard 和 run intent，不产生任何写入。"""
    _sha256(expected_guard_sha256, where="successor consumed guard")
    _sha256(run_identity_sha256, where="successor consumed run identity")
    if type(run_id) is not int or run_id <= 0:
        raise W02MorphologySuccessorContractError("successor consumed run_id 非法")
    root = Path(successor_root).resolve()
    available = root / Path(*PurePosixPath(
        W02_MORPH_SUCCESSOR_GUARD_AVAILABLE).parts)
    consumed = root / Path(*PurePosixPath(
        W02_MORPH_SUCCESSOR_GUARD_CONSUMED).parts)
    intent = root / Path(*PurePosixPath(W02_MORPH_SUCCESSOR_RUN_INTENT).parts)
    if available.exists() or not consumed.is_file() or not intent.is_file():
        raise W02MorphologySuccessorContractError(
            "successor consumed guard 状态不闭合")
    if _sha256_file(consumed)[1] != expected_guard_sha256:
        raise W02MorphologySuccessorContractError(
            "successor consumed guard SHA 漂移")
    value = read_canonical_object(intent)
    if value != {
            "artifact_kind": "PH2_D03_V2_W02_MORPHOLOGY_SUCCESSOR_RUN_INTENT",
            "formal_successor_transform_runs": 1,
            "formal_training_runs": 0,
            "format_version": 1,
            "guard_sha256": expected_guard_sha256,
            "operation_kind": W02_MORPH_OVERLAY_OPERATION,
            "run_id": run_id,
            "run_identity_sha256": run_identity_sha256,
            "stage_key": "W-02",
            "status": "GUARD_CONSUMED_BEFORE_SUCCESSOR_TRANSFORM_WRITE",
            }:
        raise W02MorphologySuccessorContractError(
            "successor run intent 字段或身份漂移")


__all__ = [
    "W02_MORPH_SUCCESSOR_CODE_PATHS",
    "W02_MORPH_SUCCESSOR_FREEZE_PATH",
    "W02_MORPH_SUCCESSOR_GUARD_AVAILABLE",
    "W02_MORPH_SUCCESSOR_GUARD_CONSUMED",
    "W02_MORPH_SUCCESSOR_RUN_INTENT",
    "W02MorphologySuccessorCodeFile",
    "W02MorphologySuccessorContractError",
    "W02MorphologySuccessorRuntimeFreeze",
    "build_w02_morphology_successor_runtime_freeze",
    "consume_w02_morphology_successor_guard",
    "publish_w02_morphology_successor_guard",
    "publish_w02_morphology_successor_runtime_freeze",
    "read_w02_morphology_successor_runtime_freeze",
    "verify_w02_morphology_successor_consumed_guard",
    "w02_morphology_successor_guard_value",
]
