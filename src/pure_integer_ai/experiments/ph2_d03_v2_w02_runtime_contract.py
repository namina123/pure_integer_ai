"""PH2-D03-V2 W-02 Candidate runtime 的追加式代码冻结合同。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path, PurePosixPath
from typing import Any

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
    exact_dict,
    read_canonical_object,
    write_immutable_json,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_candidate_store import (
    W02CandidateRuntimeBudget,
)
from pure_integer_ai.experiments.ph2_d03_v2_w02_contract import (
    W02CompileFreeze,
    read_w02_compile_freeze,
)


W02_RUNTIME_FREEZE_VERSION = "PH2-D03-V2-W02-CANDIDATE-RUNTIME-FREEZE-V1"
W02_RUNTIME_FREEZE_PATH = (
    "data/ph2/manifests/d03_v2/stages/"
    "ph2_d03_v2_w02_candidate_runtime_freeze_v1.json"
)
W02_RUNTIME_CODE_PATHS = (
    "src/pure_integer_ai/experiments/ph2_dataset_core.py",
    "src/pure_integer_ai/experiments/ph2_dataset_contract.py",
    "src/pure_integer_ai/experiments/ph2_dataset_records.py",
    "src/pure_integer_ai/experiments/ph2_dataset_owner_records.py",
    "src/pure_integer_ai/experiments/ph2_d03_contract_core.py",
    "src/pure_integer_ai/experiments/ph2_d03_v2_schema.py",
    "src/pure_integer_ai/experiments/ph2_d03_v2_streaming.py",
    "src/pure_integer_ai/experiments/ph2_d03_v2_w02_contract.py",
    "src/pure_integer_ai/experiments/ph2_d03_v2_w02_candidate_model.py",
    "src/pure_integer_ai/experiments/ph2_d03_v2_w02_candidate_io.py",
    "src/pure_integer_ai/experiments/ph2_d03_v2_w02_candidate_store.py",
    "src/pure_integer_ai/experiments/ph2_d03_v2_w02_runtime_contract.py",
    "src/pure_integer_ai/experiments/ph2_d03_v2_w02_candidate_publication.py",
    "src/pure_integer_ai/experiments/run_ph2_d03_v2_w02_candidate.py",
    "tests/test_ph2_d03_v2_w02_candidate_runtime.py",
)


# object-model: exception
class W02RuntimeFreezeError(RuntimeError):
    """Candidate runtime parent、代码或预算冻结发生漂移。"""


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
            size += len(block)
    return size, digest.hexdigest()


def _safe_repository_file(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if (not relative or "\\" in relative or pure.is_absolute()
            or pure.as_posix() != relative or ".." in pure.parts):
        raise W02RuntimeFreezeError("Candidate runtime code path 非法")
    current = root
    for part in pure.parts:
        current /= part
        if current.is_symlink():
            raise W02RuntimeFreezeError("Candidate runtime code path 经过 symlink")
    target = (root / Path(*pure.parts)).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise W02RuntimeFreezeError("Candidate runtime code file 缺失")
    return target


def _strict_sha(value: object, *, where: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise W02RuntimeFreezeError(f"{where} 必须是小写 SHA-256")
    return value


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W02RuntimeCodeFile:
    """一个 runtime 承重源码文件的公开字节身份。"""

    repository_path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if self.repository_path not in W02_RUNTIME_CODE_PATHS:
            raise W02RuntimeFreezeError("Candidate runtime code file 未注册")
        if type(self.size_bytes) is not int or self.size_bytes <= 0:
            raise W02RuntimeFreezeError("Candidate runtime code size 非法")
        _strict_sha(self.sha256, where="Candidate runtime code SHA")

    def to_dict(self) -> dict[str, object]:
        return {
            "repository_path": self.repository_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "W02RuntimeCodeFile":
        raw = exact_dict(value, {"repository_path", "sha256", "size_bytes"},
                         where="W02RuntimeCodeFile")
        return cls(str(raw["repository_path"]), raw["size_bytes"], str(raw["sha256"]))


def _runtime_code_freeze(repository: Path) -> tuple[tuple[W02RuntimeCodeFile, ...], str]:
    rows = []
    for relative in W02_RUNTIME_CODE_PATHS:
        size, digest = _sha256_file(_safe_repository_file(repository, relative))
        rows.append(W02RuntimeCodeFile(relative, size, digest))
    frozen = tuple(rows)
    commitment = hashlib.sha256(canonical_json_bytes(
        [item.to_dict() for item in frozen])).hexdigest()
    return frozen, commitment


def _verify_parent_live_code(repository: Path, freeze: W02CompileFreeze) -> None:
    """逐文件验证 parent code freeze，避免只回读 manifest 内部自洽。"""
    for item in freeze.code_files:
        size, digest = _sha256_file(
            _safe_repository_file(repository, item.repository_path))
        if size != item.size_bytes or digest != item.sha256:
            raise W02RuntimeFreezeError("Candidate parent live code 与 compile freeze 漂移")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W02CandidateRuntimeFreeze:
    """正式 Candidate 第一次写入前的追加式 runtime/code freeze。"""

    parent_compile_freeze_sha256: str
    parent_code_freeze_sha256: str
    pack_commitment: str
    candidate_contract_sha256: str
    first_run_guard_sha256: str
    code_files: tuple[W02RuntimeCodeFile, ...]
    runtime_code_freeze_sha256: str
    resource_budget: W02CandidateRuntimeBudget

    def __post_init__(self) -> None:
        for name in (
                "parent_compile_freeze_sha256", "parent_code_freeze_sha256",
                "pack_commitment", "candidate_contract_sha256",
                "first_run_guard_sha256", "runtime_code_freeze_sha256"):
            _strict_sha(getattr(self, name), where=f"Candidate runtime {name}")
        if tuple(item.repository_path for item in self.code_files) != W02_RUNTIME_CODE_PATHS:
            raise W02RuntimeFreezeError("Candidate runtime code inventory 漂移")
        expected = hashlib.sha256(canonical_json_bytes(
            [item.to_dict() for item in self.code_files])).hexdigest()
        if self.runtime_code_freeze_sha256 != expected:
            raise W02RuntimeFreezeError("Candidate runtime code commitment 漂移")
        if not isinstance(self.resource_budget, W02CandidateRuntimeBudget):
            raise W02RuntimeFreezeError("Candidate runtime budget 类型错误")

    def to_dict(self) -> dict[str, object]:
        return {
            "allowed_workers": [1, 2, 4],
            "artifact_kind": "PH2_D03_V2_W02_CANDIDATE_RUNTIME_FREEZE",
            "artifact_version": W02_RUNTIME_FREEZE_VERSION,
            "candidate_contract_sha256": self.candidate_contract_sha256,
            "candidate_writes": 0,
            "code_files": [item.to_dict() for item in self.code_files],
            "first_run_guard_sha256": self.first_run_guard_sha256,
            "formal_private_evaluation_runs": 0,
            "formal_training_runs": 0,
            "fresh_restart_resume_required": 1,
            "language_capability_mastered": 0,
            "language_readiness": 0,
            "logical_shard_count": 128,
            "manifest_last_required": 1,
            "next_action": "W02_FORMAL_CANDIDATE_FIRST_RUN",
            "pack_commitment": self.pack_commitment,
            "parent_code_freeze_sha256": self.parent_code_freeze_sha256,
            "parent_compile_freeze_sha256": self.parent_compile_freeze_sha256,
            "private_payload_reads": 0,
            "release_key": "PH2-D03-V2",
            "resource_budget": self.resource_budget.to_dict(),
            "runtime_code_freeze_sha256": self.runtime_code_freeze_sha256,
            "stage_key": "W-02",
            "status": "W02_CANDIDATE_RUNTIME_FREEZE_COMPLETE",
            "teacher_calls": 0,
            "worker_canonical_equivalence_required": 1,
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict()) + b"\n"

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


def build_w02_candidate_runtime_freeze(
        repository_root: str | Path,
        ) -> W02CandidateRuntimeFreeze:
    """回验 parent 和 live code 后构造尚未发布的 runtime freeze。"""
    repository = Path(repository_root).resolve()
    parent = read_w02_compile_freeze(repository)
    _verify_parent_live_code(repository, parent)
    files, code_sha = _runtime_code_freeze(repository)
    return W02CandidateRuntimeFreeze(
        parent.sha256(), parent.code_freeze_sha256, parent.pack_commitment,
        parent.candidate_contract_sha256, parent.first_run_guard_sha256,
        files, code_sha, W02CandidateRuntimeBudget())


def publish_w02_candidate_runtime_freeze(
        repository_root: str | Path,
        ) -> Path:
    """不可覆盖发布 Candidate runtime freeze。"""
    repository = Path(repository_root).resolve()
    freeze = build_w02_candidate_runtime_freeze(repository)
    target = repository / Path(*PurePosixPath(W02_RUNTIME_FREEZE_PATH).parts)
    write_immutable_json(freeze.to_dict(), target)
    if target.read_bytes() != freeze.canonical_bytes():
        raise W02RuntimeFreezeError("Candidate runtime freeze 发布字节漂移")
    return target


def read_w02_candidate_runtime_freeze(
        repository_root: str | Path,
        ) -> W02CandidateRuntimeFreeze:
    """严格回读 runtime freeze 并重算当前 live code。"""
    repository = Path(repository_root).resolve()
    target = repository / Path(*PurePosixPath(W02_RUNTIME_FREEZE_PATH).parts)
    raw = exact_dict(read_canonical_object(target), {
        "allowed_workers", "artifact_kind", "artifact_version",
        "candidate_contract_sha256", "candidate_writes", "code_files",
        "first_run_guard_sha256", "formal_private_evaluation_runs",
        "formal_training_runs", "fresh_restart_resume_required",
        "language_capability_mastered", "language_readiness",
        "logical_shard_count", "manifest_last_required", "next_action",
        "pack_commitment", "parent_code_freeze_sha256",
        "parent_compile_freeze_sha256", "private_payload_reads", "release_key",
        "resource_budget", "runtime_code_freeze_sha256", "stage_key", "status",
        "teacher_calls", "worker_canonical_equivalence_required",
    }, where="W02CandidateRuntimeFreeze")
    if (raw["artifact_kind"] != "PH2_D03_V2_W02_CANDIDATE_RUNTIME_FREEZE"
            or raw["artifact_version"] != W02_RUNTIME_FREEZE_VERSION
            or raw["release_key"] != "PH2-D03-V2" or raw["stage_key"] != "W-02"
            or raw["status"] != "W02_CANDIDATE_RUNTIME_FREEZE_COMPLETE"
            or raw["next_action"] != "W02_FORMAL_CANDIDATE_FIRST_RUN"
            or raw["allowed_workers"] != [1, 2, 4]
            or raw["logical_shard_count"] != 128):
        raise W02RuntimeFreezeError("Candidate runtime freeze 顶层身份漂移")
    zero_fields = (
        "candidate_writes", "formal_private_evaluation_runs", "formal_training_runs",
        "language_capability_mastered", "language_readiness", "private_payload_reads",
        "teacher_calls",
    )
    one_fields = (
        "fresh_restart_resume_required", "manifest_last_required",
        "worker_canonical_equivalence_required",
    )
    if (any(raw[name] != 0 for name in zero_fields)
            or any(raw[name] != 1 for name in one_fields)):
        raise W02RuntimeFreezeError("Candidate runtime freeze 初始状态漂移")
    files_raw = raw["code_files"]
    if not isinstance(files_raw, list):
        raise W02RuntimeFreezeError("Candidate runtime code inventory 类型错误")
    budget_raw = exact_dict(raw["resource_budget"], {
        "max_checkpoint_count", "max_logic_operations", "max_pairs",
        "max_payload_bytes", "max_shard_delta_bytes",
    }, where="W02CandidateRuntimeBudget")
    freeze = W02CandidateRuntimeFreeze(
        str(raw["parent_compile_freeze_sha256"]),
        str(raw["parent_code_freeze_sha256"]), str(raw["pack_commitment"]),
        str(raw["candidate_contract_sha256"]), str(raw["first_run_guard_sha256"]),
        tuple(W02RuntimeCodeFile.from_dict(item) for item in files_raw),
        str(raw["runtime_code_freeze_sha256"]),
        W02CandidateRuntimeBudget(**budget_raw),
    )
    current = build_w02_candidate_runtime_freeze(repository)
    if (freeze != current or target.read_bytes() != freeze.canonical_bytes()):
        raise W02RuntimeFreezeError("Candidate runtime freeze 与 live code 漂移")
    return freeze


__all__ = [
    "W02_RUNTIME_CODE_PATHS",
    "W02_RUNTIME_FREEZE_PATH",
    "W02CandidateRuntimeFreeze",
    "W02RuntimeCodeFile",
    "W02RuntimeFreezeError",
    "build_w02_candidate_runtime_freeze",
    "publish_w02_candidate_runtime_freeze",
    "read_w02_candidate_runtime_freeze",
]
