"""W-02 morphology successor V2 overlay 的正式冻结与首次运行 guard。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path, PurePosixPath

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
    read_canonical_object,
    write_immutable_json,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_candidate_publication import (
    W02_CANDIDATE_RECEIPT_PATH,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_publication import (
    W02_MORPH_SUCCESSOR_RECEIPT_PATH,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v2 import (
    W02_MORPH_SUCCESSOR_V2_VERSION,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_morphology_successor_v2_overlay import (
    W02MorphologySuccessorV2OverlayBudget,
)


W02_MORPH_V2_FREEZE_VERSION = (
    "PH2-D03-V2-W02-MORPHOLOGY-SUCCESSOR-V2-OVERLAY-RUNTIME-FREEZE-V1")
W02_MORPH_V2_FREEZE_PATH = (
    "data/ph2/manifests/d03_v2/stages/"
    "ph2_d03_v2_w02_morphology_successor_v2_overlay_runtime_freeze_v1.json")
W02_MORPH_V2_GUARD_VERSION = (
    "PH2-D03-V2-W02-MORPHOLOGY-SUCCESSOR-V2-FIRST-RUN-GUARD-V1")
W02_MORPH_V2_GUARD_AVAILABLE = "run-guard/available.guard.json"
W02_MORPH_V2_GUARD_CONSUMED = "run-guard/consumed.guard.json"
W02_MORPH_V2_RUN_INTENT = "run-guard/run-intent.json"
W02_MORPH_V2_OPERATION = "CANDIDATE_DERIVED_EDGE_LEMMA_TRANSFORM"
W02_MORPH_V2_EXPECTED_SEMANTIC = (
    "d2da281eca1a1fda5e5d4260320f08535c96d015c8838225c63dfabd8250df70")
W02_MORPH_V2_PREFLIGHT_MANIFEST = (
    "a841c2b046ff62bdab3a37bf362dc602217c761bd58097b1a93b037474fa3661")
W02_MORPH_V2_PREFLIGHT_TREE = (
    "1211157900a44d5cd4b8261e94dde95c89a665a95e49d3396138a6dd94fc77ea")
W02_MORPH_V2_CODE_PATHS = (
    "src/pure_integer_ai/experiments/"
    "ph2_d03_v2_w02_morphology_successor_v2.py",
    "src/pure_integer_ai/experiments/"
    "ph2_d03_v2_w02_morphology_successor_v2_overlay.py",
    "src/pure_integer_ai/experiments/"
    "ph2_d03_v2_w02_morphology_successor_v2_contract.py",
    "src/pure_integer_ai/experiments/"
    "ph2_d03_v2_w02_morphology_successor_v2_publication.py",
    "src/pure_integer_ai/experiments/"
    "run_ph2_d03_v2_w02_morphology_successor_v2.py",
    "tests/test_ph2_d03_v2_w02_morphology_successor_v2.py",
    "tests/test_ph2_d03_v2_w02_morphology_successor_v2_contract.py",
)


# object-model: exception
class W02MorphologySuccessorV2ContractError(RuntimeError):
    """V2 overlay freeze、live code 或 guard 发生漂移。"""


def _strict_sha(value: object, *, where: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise W02MorphologySuccessorV2ContractError(f"{where} 不是小写 SHA-256")
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
        raise W02MorphologySuccessorV2ContractError("V2 code path 非法")
    current = root
    for part in pure.parts:
        current /= part
        if current.is_symlink():
            raise W02MorphologySuccessorV2ContractError(
                "V2 code path 经过 symlink")
    target = (root / Path(*pure.parts)).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise W02MorphologySuccessorV2ContractError("V2 code/evidence 文件缺失")
    return target


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W02MorphologySuccessorV2CodeFile:
    """一个 V2 正式 transform 承重文件的公开身份。"""

    repository_path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if self.repository_path not in W02_MORPH_V2_CODE_PATHS:
            raise W02MorphologySuccessorV2ContractError("V2 code file 未注册")
        if type(self.size_bytes) is not int or self.size_bytes <= 0:
            raise W02MorphologySuccessorV2ContractError("V2 code size 非法")
        _strict_sha(self.sha256, where="V2 code SHA")

    def to_dict(self) -> dict[str, object]:
        return {
            "repository_path": self.repository_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


def _code_freeze(
        repository: Path,
        ) -> tuple[tuple[W02MorphologySuccessorV2CodeFile, ...], str]:
    files = tuple(
        W02MorphologySuccessorV2CodeFile(
            relative, *_sha256_file(_repository_file(repository, relative)))
        for relative in W02_MORPH_V2_CODE_PATHS)
    commitment = hashlib.sha256(canonical_json_bytes(
        [item.to_dict() for item in files])).hexdigest()
    return files, commitment


def w02_morphology_successor_v2_guard_value(
        *, runtime_code_freeze_sha256: str,
        parent_candidate_manifest_sha256: str,
        parent_v1_manifest_sha256: str,
        expected_semantic_sha256: str,
        ) -> dict[str, object]:
    """构造 V2 transform guard 的规范初始值。"""
    return {
        "artifact_kind": "PH2_D03_V2_W02_MORPHOLOGY_SUCCESSOR_V2_FIRST_RUN_GUARD",
        "expected_semantic_sha256": _strict_sha(
            expected_semantic_sha256, where="V2 expected semantic"),
        "formal_successor_v2_transform_runs": 0,
        "formal_training_runs": 0,
        "format_version": 1,
        "guard_consumed": 0,
        "guard_version": W02_MORPH_V2_GUARD_VERSION,
        "operation_kind": W02_MORPH_V2_OPERATION,
        "parent_candidate_manifest_sha256": _strict_sha(
            parent_candidate_manifest_sha256, where="V2 parent Candidate"),
        "parent_v1_overlay_manifest_sha256": _strict_sha(
            parent_v1_manifest_sha256, where="V2 parent V1 overlay"),
        "release_key": "PH2-D03-V2",
        "run_id_policy": "NEW_POSITIVE_INTEGER_REQUIRED",
        "runtime_code_freeze_sha256": _strict_sha(
            runtime_code_freeze_sha256, where="V2 code freeze"),
        "stage_key": "W-02",
        "status": "AVAILABLE",
    }


def _guard_sha(value: dict[str, object]) -> str:
    return hashlib.sha256(canonical_json_bytes(value) + b"\n").hexdigest()


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W02MorphologySuccessorV2RuntimeFreeze:
    """唯一 V2 formal transform 之前的公开冻结。"""

    candidate_receipt_file_sha256: str
    v1_receipt_file_sha256: str
    parent_candidate_manifest_sha256: str
    parent_candidate_semantic_sha256: str
    parent_v1_manifest_sha256: str
    parent_v1_semantic_sha256: str
    parent_v1_runtime_freeze_sha256: str
    preflight_manifest_sha256: str
    preflight_tree_sha256: str
    expected_semantic_sha256: str
    code_files: tuple[W02MorphologySuccessorV2CodeFile, ...]
    runtime_code_freeze_sha256: str
    first_run_guard_sha256: str
    resource_budget: W02MorphologySuccessorV2OverlayBudget

    def __post_init__(self) -> None:
        for name in (
                "candidate_receipt_file_sha256", "v1_receipt_file_sha256",
                "parent_candidate_manifest_sha256",
                "parent_candidate_semantic_sha256",
                "parent_v1_manifest_sha256", "parent_v1_semantic_sha256",
                "parent_v1_runtime_freeze_sha256",
                "preflight_manifest_sha256", "preflight_tree_sha256",
                "expected_semantic_sha256", "runtime_code_freeze_sha256",
                "first_run_guard_sha256"):
            _strict_sha(getattr(self, name), where=f"V2 freeze {name}")
        if tuple(item.repository_path for item in self.code_files) != (
                W02_MORPH_V2_CODE_PATHS):
            raise W02MorphologySuccessorV2ContractError(
                "V2 code inventory 漂移")
        expected_code = hashlib.sha256(canonical_json_bytes(
            [item.to_dict() for item in self.code_files])).hexdigest()
        if expected_code != self.runtime_code_freeze_sha256:
            raise W02MorphologySuccessorV2ContractError(
                "V2 code commitment 漂移")
        expected_guard = _guard_sha(w02_morphology_successor_v2_guard_value(
            runtime_code_freeze_sha256=self.runtime_code_freeze_sha256,
            parent_candidate_manifest_sha256=(
                self.parent_candidate_manifest_sha256),
            parent_v1_manifest_sha256=self.parent_v1_manifest_sha256,
            expected_semantic_sha256=self.expected_semantic_sha256))
        if expected_guard != self.first_run_guard_sha256:
            raise W02MorphologySuccessorV2ContractError("V2 guard SHA 漂移")
        if not isinstance(
                self.resource_budget, W02MorphologySuccessorV2OverlayBudget):
            raise W02MorphologySuccessorV2ContractError("V2 budget 类型错误")

    def to_dict(self) -> dict[str, object]:
        return {
            "allowed_workers": [1, 2, 4],
            "artifact_kind": "PH2_D03_V2_W02_MORPHOLOGY_SUCCESSOR_V2_RUNTIME_FREEZE",
            "artifact_version": W02_MORPH_V2_FREEZE_VERSION,
            "candidate_writes": 0,
            "code_files": [item.to_dict() for item in self.code_files],
            "expected_counts": {
                "accepted_lexeme_rows": 112,
                "accepted_support_count": 651,
                "logic_operations": 4_611,
                "rule_row_count": 383,
                "unsupported_lexeme_rows": 4,
                "unsupported_support_count": 4,
            },
            "expected_semantic_sha256": self.expected_semantic_sha256,
            "first_run_guard_sha256": self.first_run_guard_sha256,
            "formal_private_evaluation_runs": 0,
            "formal_successor_v2_transform_runs": 0,
            "formal_training_runs": 0,
            "language_capability_mastered": 0,
            "language_readiness": 0,
            "logical_shard_count": 128,
            "manifest_last_required": 1,
            "next_action": "W02_FORMAL_MORPHOLOGY_SUCCESSOR_V2_TRANSFORM",
            "operation_kind": W02_MORPH_V2_OPERATION,
            "parent_candidate_manifest_sha256": (
                self.parent_candidate_manifest_sha256),
            "parent_candidate_receipt_file_sha256": (
                self.candidate_receipt_file_sha256),
            "parent_candidate_semantic_sha256": (
                self.parent_candidate_semantic_sha256),
            "parent_formal_successor_transform_runs": 1,
            "parent_formal_training_runs": 1,
            "parent_v1_overlay_manifest_sha256": self.parent_v1_manifest_sha256,
            "parent_v1_overlay_receipt_file_sha256": self.v1_receipt_file_sha256,
            "parent_v1_overlay_semantic_sha256": self.parent_v1_semantic_sha256,
            "parent_v1_runtime_freeze_sha256": (
                self.parent_v1_runtime_freeze_sha256),
            "preflight_manifest_sha256": self.preflight_manifest_sha256,
            "preflight_tree_sha256": self.preflight_tree_sha256,
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
            "status": "W02_MORPHOLOGY_SUCCESSOR_V2_RUNTIME_FREEZE_COMPLETE",
            "successor_version": W02_MORPH_SUCCESSOR_V2_VERSION,
            "teacher_calls": 0,
            "v1_overlay_writes": 0,
            "v2_overlay_writes": 0,
            "worker_canonical_equivalence_required": 1,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict()) + b"\n"

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


def build_w02_morphology_successor_v2_runtime_freeze(
        repository_root: str | Path,
        ) -> W02MorphologySuccessorV2RuntimeFreeze:
    """回验双父 receipt 和 live code，构造 V2 runtime freeze。"""
    repository = Path(repository_root).resolve()
    candidate_path = _repository_file(repository, W02_CANDIDATE_RECEIPT_PATH)
    v1_path = _repository_file(repository, W02_MORPH_SUCCESSOR_RECEIPT_PATH)
    candidate = read_canonical_object(candidate_path)
    v1 = read_canonical_object(v1_path)
    if (candidate.get("status") != "W02_CANDIDATE_ARTIFACT_FROZEN"
            or candidate.get("formal_training_runs") != 1
            or candidate.get("formal_private_evaluation_runs") != 0
            or candidate.get("private_payload_reads") != 0
            or candidate.get("teacher_calls") != 0):
        raise W02MorphologySuccessorV2ContractError(
            "V2 parent Candidate receipt 漂移")
    if (v1.get("status") != "W02_MORPHOLOGY_SUCCESSOR_ARTIFACT_FROZEN"
            or v1.get("formal_successor_transform_runs") != 1
            or v1.get("formal_training_runs") != 0
            or v1.get("formal_private_evaluation_runs") != 0
            or v1.get("private_payload_reads") != 0
            or v1.get("teacher_calls") != 0
            or v1.get("candidate_writes") != 0
            or v1.get("overlay_writes") != 1
            or v1.get("parent_candidate_manifest_sha256")
            != candidate.get("candidate_artifact_manifest_sha256")
            or v1.get("parent_candidate_semantic_sha256")
            != candidate.get("candidate_semantic_sha256")):
        raise W02MorphologySuccessorV2ContractError(
            "V2 parent V1 overlay receipt 漂移")
    files, code_sha = _code_freeze(repository)
    guard = w02_morphology_successor_v2_guard_value(
        runtime_code_freeze_sha256=code_sha,
        parent_candidate_manifest_sha256=str(
            candidate["candidate_artifact_manifest_sha256"]),
        parent_v1_manifest_sha256=str(
            v1["overlay_artifact_manifest_sha256"]),
        expected_semantic_sha256=W02_MORPH_V2_EXPECTED_SEMANTIC)
    return W02MorphologySuccessorV2RuntimeFreeze(
        _sha256_file(candidate_path)[1], _sha256_file(v1_path)[1],
        str(candidate["candidate_artifact_manifest_sha256"]),
        str(candidate["candidate_semantic_sha256"]),
        str(v1["overlay_artifact_manifest_sha256"]),
        str(v1["overlay_semantic_sha256"]), str(v1["runtime_freeze_sha256"]),
        W02_MORPH_V2_PREFLIGHT_MANIFEST, W02_MORPH_V2_PREFLIGHT_TREE,
        W02_MORPH_V2_EXPECTED_SEMANTIC, files, code_sha, _guard_sha(guard),
        W02MorphologySuccessorV2OverlayBudget())


def publish_w02_morphology_successor_v2_runtime_freeze(
        repository_root: str | Path,
        ) -> Path:
    """不可覆盖发布 V2 runtime freeze。"""
    repository = Path(repository_root).resolve()
    freeze = build_w02_morphology_successor_v2_runtime_freeze(repository)
    target = repository / Path(*PurePosixPath(W02_MORPH_V2_FREEZE_PATH).parts)
    write_immutable_json(freeze.to_dict(), target)
    if target.read_bytes() != freeze.canonical_bytes():
        raise W02MorphologySuccessorV2ContractError("V2 freeze 发布字节漂移")
    return target


def read_w02_morphology_successor_v2_runtime_freeze(
        repository_root: str | Path,
        ) -> W02MorphologySuccessorV2RuntimeFreeze:
    """严格回读 freeze，并重算双父证据与 live code。"""
    repository = Path(repository_root).resolve()
    target = repository / Path(*PurePosixPath(W02_MORPH_V2_FREEZE_PATH).parts)
    freeze = build_w02_morphology_successor_v2_runtime_freeze(repository)
    value = read_canonical_object(target)
    if value != freeze.to_dict() or target.read_bytes() != freeze.canonical_bytes():
        raise W02MorphologySuccessorV2ContractError(
            "V2 freeze 与 live code/parent evidence 漂移")
    return freeze


def publish_w02_morphology_successor_v2_guard(
        successor_root: str | Path,
        freeze: W02MorphologySuccessorV2RuntimeFreeze,
        ) -> str:
    """在全新 formal root 独占发布 V2 first-run guard。"""
    if not isinstance(freeze, W02MorphologySuccessorV2RuntimeFreeze):
        raise W02MorphologySuccessorV2ContractError("V2 guard freeze 类型错误")
    root = Path(successor_root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    target = root / Path(*PurePosixPath(W02_MORPH_V2_GUARD_AVAILABLE).parts)
    value = w02_morphology_successor_v2_guard_value(
        runtime_code_freeze_sha256=freeze.runtime_code_freeze_sha256,
        parent_candidate_manifest_sha256=(
            freeze.parent_candidate_manifest_sha256),
        parent_v1_manifest_sha256=freeze.parent_v1_manifest_sha256,
        expected_semantic_sha256=freeze.expected_semantic_sha256)
    write_immutable_json(value, target)
    if _sha256_file(target)[1] != freeze.first_run_guard_sha256:
        raise W02MorphologySuccessorV2ContractError("V2 guard 发布 SHA 漂移")
    return freeze.first_run_guard_sha256


def consume_w02_morphology_successor_v2_guard(
        successor_root: str | Path, *, expected_guard_sha256: str,
        run_id: int, run_identity_sha256: str,
        ) -> None:
    """在 formal V2 store 首次写入前原子消费 guard。"""
    _strict_sha(expected_guard_sha256, where="V2 expected guard")
    _strict_sha(run_identity_sha256, where="V2 run identity")
    if type(run_id) is not int or run_id <= 0:
        raise W02MorphologySuccessorV2ContractError("V2 formal run_id 非法")
    root = Path(successor_root).resolve()
    available = root / Path(*PurePosixPath(W02_MORPH_V2_GUARD_AVAILABLE).parts)
    consumed = root / Path(*PurePosixPath(W02_MORPH_V2_GUARD_CONSUMED).parts)
    intent = root / Path(*PurePosixPath(W02_MORPH_V2_RUN_INTENT).parts)
    if consumed.exists() or intent.exists() or not available.is_file():
        raise W02MorphologySuccessorV2ContractError(
            "V2 guard 不可用或已消费")
    if _sha256_file(available)[1] != expected_guard_sha256:
        raise W02MorphologySuccessorV2ContractError("V2 guard 字节漂移")
    os.replace(available, consumed)
    write_immutable_json({
        "artifact_kind": "PH2_D03_V2_W02_MORPHOLOGY_SUCCESSOR_V2_RUN_INTENT",
        "formal_successor_v2_transform_runs": 1,
        "formal_training_runs": 0,
        "format_version": 1,
        "guard_sha256": expected_guard_sha256,
        "operation_kind": W02_MORPH_V2_OPERATION,
        "run_id": run_id,
        "run_identity_sha256": run_identity_sha256,
        "stage_key": "W-02",
        "status": "GUARD_CONSUMED_BEFORE_SUCCESSOR_V2_TRANSFORM_WRITE",
    }, intent)


def verify_w02_morphology_successor_v2_consumed_guard(
        successor_root: str | Path, *, expected_guard_sha256: str,
        run_id: int, run_identity_sha256: str,
        ) -> None:
    """严格回读 consumed guard 与 run intent，保持零写。"""
    _strict_sha(expected_guard_sha256, where="V2 consumed guard")
    _strict_sha(run_identity_sha256, where="V2 consumed identity")
    root = Path(successor_root).resolve()
    available = root / Path(*PurePosixPath(W02_MORPH_V2_GUARD_AVAILABLE).parts)
    consumed = root / Path(*PurePosixPath(W02_MORPH_V2_GUARD_CONSUMED).parts)
    intent = root / Path(*PurePosixPath(W02_MORPH_V2_RUN_INTENT).parts)
    if available.exists() or not consumed.is_file() or not intent.is_file():
        raise W02MorphologySuccessorV2ContractError(
            "V2 consumed guard 状态不闭合")
    if _sha256_file(consumed)[1] != expected_guard_sha256:
        raise W02MorphologySuccessorV2ContractError(
            "V2 consumed guard SHA 漂移")
    expected = {
        "artifact_kind": "PH2_D03_V2_W02_MORPHOLOGY_SUCCESSOR_V2_RUN_INTENT",
        "formal_successor_v2_transform_runs": 1,
        "formal_training_runs": 0,
        "format_version": 1,
        "guard_sha256": expected_guard_sha256,
        "operation_kind": W02_MORPH_V2_OPERATION,
        "run_id": run_id,
        "run_identity_sha256": run_identity_sha256,
        "stage_key": "W-02",
        "status": "GUARD_CONSUMED_BEFORE_SUCCESSOR_V2_TRANSFORM_WRITE",
    }
    if read_canonical_object(intent) != expected:
        raise W02MorphologySuccessorV2ContractError(
            "V2 run intent 字段或身份漂移")


__all__ = [
    "W02_MORPH_V2_CODE_PATHS", "W02_MORPH_V2_FREEZE_PATH",
    "W02_MORPH_V2_GUARD_AVAILABLE", "W02_MORPH_V2_GUARD_CONSUMED",
    "W02_MORPH_V2_RUN_INTENT", "W02MorphologySuccessorV2CodeFile",
    "W02MorphologySuccessorV2ContractError",
    "W02MorphologySuccessorV2RuntimeFreeze",
    "build_w02_morphology_successor_v2_runtime_freeze",
    "consume_w02_morphology_successor_v2_guard",
    "publish_w02_morphology_successor_v2_guard",
    "publish_w02_morphology_successor_v2_runtime_freeze",
    "read_w02_morphology_successor_v2_runtime_freeze",
    "verify_w02_morphology_successor_v2_consumed_guard",
    "w02_morphology_successor_v2_guard_value",
]
