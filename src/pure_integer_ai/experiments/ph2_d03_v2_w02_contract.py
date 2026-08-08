"""PH2-D03-V2 W-02 正式 pack、代码冻结和首次运行合同。"""
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
from pure_integer_ai.experiments.ph2_d03_v2_authority import V2_RELEASE_KEY
from pure_integer_ai.experiments.ph2_d03_v2_evaluator_contract import (
    V2_STAGE_EVALUATION_POLICIES,
    V2EvaluatorResourceBudget,
)


W02_STAGE_KEY = "W-02"
W02_COMPILE_FREEZE_KIND = "PH2_D03_V2_W02_COMPILE_FREEZE"
W02_COMPILE_FREEZE_VERSION = "PH2-D03-V2-W02-compile-freeze-v1"
W02_COMPILE_FREEZE_PATH = (
    "data/ph2/manifests/d03_v2/stages/"
    "ph2_d03_v2_w02_compile_freeze_v1.json"
)
W02_FIRST_RUN_GUARD_VERSION = "PH2-D03-V2-W02-first-run-guard-v1"
W02_FIRST_RUN_GUARD_AVAILABLE = "run-guard/available.guard.json"
W02_FIRST_RUN_GUARD_CONSUMED = "run-guard/consumed.guard.json"
W02_FIRST_RUN_INTENT = "run-guard/run-intent.json"
W02_ALLOWED_WORKERS = (1, 2, 4)
W02_LOGICAL_SHARD_COUNT = 128
W02_TARGET_TRAIN_OBSERVATIONS = 51_200
W02_FORMAL_PRIVATE_RUNS = 0
W02_FORMAL_TRAINING_RUNS = 0

W02_SPLITS = ("train", "dev", "held_out", "adversarial", "wall")
W02_SOURCES = (
    "AUTHORED_CC0",
    "UD_ZH_GSDSIMP_R2_18",
    "ZHWIKTIONARY_20260701",
)
W02_LAYOUTS = {
    "CANDIDATE_SOURCE": (
        "CANDIDATE_TRAIN_ROOT", "source_ref", "", "source/source_refs.jsonl.gz"),
    "CANDIDATE_TRAIN_OBSERVATION": (
        "CANDIDATE_TRAIN_ROOT", "observation", "train", "observations/train.jsonl.gz"),
    "TEACHER_SOURCE": (
        "TEACHER_TRAIN_ROOT", "source_ref", "", "source/source_refs.jsonl.gz"),
    "TEACHER_TRAIN_EVIDENCE": (
        "TEACHER_TRAIN_ROOT", "teacher_evidence", "train", "teacher/train.evidence.jsonl.gz"),
    "DEV_SOURCE": (
        "DEV_CALIBRATION_ROOT", "source_ref", "", "source/source_refs.jsonl.gz"),
    "DEV_OBSERVATION": (
        "DEV_CALIBRATION_ROOT", "observation", "dev", "observations/dev.jsonl.gz"),
    "DEV_LABEL": (
        "DEV_CALIBRATION_ROOT", "evaluator_label", "dev", "evaluator/dev.labels.jsonl.gz"),
    "SHADOW_SOURCE": (
        "SHADOW_AUDIT_ROOT", "source_ref", "", "source/source_refs.jsonl.gz"),
    "SHADOW_TRAIN_OBSERVATION": (
        "SHADOW_AUDIT_ROOT", "observation", "train", "observations/train.jsonl.gz"),
    "SHADOW_DEV_OBSERVATION": (
        "SHADOW_AUDIT_ROOT", "observation", "dev", "observations/dev.jsonl.gz"),
    "PRIVATE_SOURCE": (
        "PRIVATE_EVALUATOR_ROOT", "source_ref", "", "source/source_refs.jsonl.gz"),
    "PRIVATE_HELD_OUT_OBSERVATION": (
        "PRIVATE_EVALUATOR_ROOT", "observation", "held_out", "observations/held_out.jsonl.gz"),
    "PRIVATE_ADVERSARIAL_OBSERVATION": (
        "PRIVATE_EVALUATOR_ROOT", "observation", "adversarial", "observations/adversarial.jsonl.gz"),
    "PRIVATE_WALL_OBSERVATION": (
        "PRIVATE_EVALUATOR_ROOT", "observation", "wall", "observations/wall.jsonl.gz"),
    "PRIVATE_HELD_OUT_LABEL": (
        "PRIVATE_EVALUATOR_ROOT", "evaluator_label", "held_out", "evaluator/held_out.labels.jsonl.gz"),
    "PRIVATE_ADVERSARIAL_LABEL": (
        "PRIVATE_EVALUATOR_ROOT", "evaluator_label", "adversarial", "evaluator/adversarial.labels.jsonl.gz"),
    "PRIVATE_WALL_LABEL": (
        "PRIVATE_EVALUATOR_ROOT", "evaluator_label", "wall", "evaluator/wall.labels.jsonl.gz"),
}

W02_CODE_FREEZE_PATHS = (
    "src/pure_integer_ai/experiments/ph2_d03_v2_authority.py",
    "src/pure_integer_ai/experiments/ph2_d03_v2_evaluator_contract.py",
    "src/pure_integer_ai/experiments/ph2_d03_v2_evaluator_firewall.py",
    "src/pure_integer_ai/experiments/ph2_d03_v2_registry.py",
    "src/pure_integer_ai/experiments/ph2_d03_v2_schema.py",
    "src/pure_integer_ai/experiments/ph2_d03_v2_streaming.py",
    "src/pure_integer_ai/experiments/ph2_d03_v2_w02_compiler.py",
    "src/pure_integer_ai/experiments/ph2_d03_v2_w02_contract.py",
    "src/pure_integer_ai/experiments/ph2_ud_gsdsimp_adapter.py",
)


class W02CompileFreezeError(RuntimeError):
    """W-02 数据、阈值、代码或首次运行边界发生漂移。"""


def _sha256(value: object, *, where: str) -> str:
    """校验小写 SHA-256 文本。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise W02CompileFreezeError(f"{where} 必须是小写 SHA-256")
    return value


def _positive(value: object, *, where: str, allow_zero: bool = False) -> int:
    """校验严格整数计数。"""
    if type(value) is not int or value < (0 if allow_zero else 1):
        raise W02CompileFreezeError(f"{where} 整数范围非法")
    return value


def _safe_relative(value: object, *, where: str) -> str:
    """拒绝绝对路径、反斜线和路径逃逸。"""
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise W02CompileFreezeError(f"{where} 不是安全相对路径")
    path = PurePosixPath(value)
    if (path.is_absolute() or path.as_posix() != value
            or any(part in {"", ".", ".."} for part in path.parts)):
        raise W02CompileFreezeError(f"{where} 不是安全相对路径")
    return value


def _hash_value(value: object) -> str:
    """返回规范 JSON object 的 SHA-256。"""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _hash_file(path: Path) -> tuple[int, str]:
    """流式返回文件长度和 SHA-256。"""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
            size += len(block)
    return size, digest.hexdigest()


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W02SourceSplitCount:
    """一个来源在五个隔离 split 上的 Observation 定额。"""

    source_key: str
    train: int
    dev: int
    held_out: int
    adversarial: int
    wall: int

    def __post_init__(self) -> None:
        if self.source_key not in W02_SOURCES:
            raise W02CompileFreezeError("W-02 来源未注册")
        for split in W02_SPLITS:
            _positive(getattr(self, split), where=f"{self.source_key}.{split}", allow_zero=True)
        if self.train <= 0:
            raise W02CompileFreezeError("W-02 每个来源必须有 forming 记录")

    def count(self, split: str) -> int:
        """返回一个已注册 split 的定额。"""
        if split not in W02_SPLITS:
            raise W02CompileFreezeError("W-02 split 未注册")
        return getattr(self, split)

    def to_dict(self) -> dict[str, object]:
        return {
            "adversarial": self.adversarial,
            "dev": self.dev,
            "held_out": self.held_out,
            "source_key": self.source_key,
            "train": self.train,
            "wall": self.wall,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "W02SourceSplitCount":
        """从严格 object 恢复来源定额。"""
        raw = exact_dict(value, {
            "adversarial", "dev", "held_out", "source_key", "train", "wall",
        }, where="W02SourceSplitCount")
        return cls(
            str(raw["source_key"]), raw["train"], raw["dev"],
            raw["held_out"], raw["adversarial"], raw["wall"],
        )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W02CompilePlan:
    """冻结 W-02 正式 Observation 规模，不混入其他 record kind。"""

    source_counts: tuple[W02SourceSplitCount, ...]
    authored_carrier_count: int = 9
    logical_shard_count: int = W02_LOGICAL_SHARD_COUNT
    allowed_workers: tuple[int, ...] = W02_ALLOWED_WORKERS

    def __post_init__(self) -> None:
        if (not isinstance(self.source_counts, tuple)
                or tuple(item.source_key for item in self.source_counts) != W02_SOURCES):
            raise W02CompileFreezeError("W-02 来源定额顺序漂移")
        if self.authored_carrier_count != 9:
            raise W02CompileFreezeError("W-02 authored family 必须覆盖九载体")
        authored = self.source_counts[0]
        if any(authored.count(split) % self.authored_carrier_count
               for split in W02_SPLITS):
            raise W02CompileFreezeError("W-02 authored 定额必须按九载体整族闭合")
        if (self.logical_shard_count != W02_LOGICAL_SHARD_COUNT
                or self.allowed_workers != W02_ALLOWED_WORKERS):
            raise W02CompileFreezeError("W-02 shard/worker 合同漂移")
        if self.split_total("train") != W02_TARGET_TRAIN_OBSERVATIONS:
            raise W02CompileFreezeError("W-02 train Observation 不等于 P2 规模")

    def split_total(self, split: str) -> int:
        """汇总一个 split 的 Observation 数。"""
        return sum(item.count(split) for item in self.source_counts)

    def total_observations(self) -> int:
        """返回五个 split 的 Observation 总数。"""
        return sum(self.split_total(split) for split in W02_SPLITS)

    def to_dict(self) -> dict[str, object]:
        return {
            "allowed_workers": list(self.allowed_workers),
            "authored_carrier_count": self.authored_carrier_count,
            "logical_shard_count": self.logical_shard_count,
            "source_counts": [item.to_dict() for item in self.source_counts],
            "split_totals": {
                split: self.split_total(split) for split in W02_SPLITS
            },
            "total_observations": self.total_observations(),
            "train_observation_target": W02_TARGET_TRAIN_OBSERVATIONS,
        }


def formal_w02_compile_plan() -> W02CompilePlan:
    """返回唯一正式定额；P2 指 Candidate 可见的 train Observation。"""
    return W02CompilePlan((
        W02SourceSplitCount("AUTHORED_CC0", 8_001, 1_206, 1_800, 603, 405),
        W02SourceSplitCount("UD_ZH_GSDSIMP_R2_18", 3_997, 500, 500, 0, 0),
        W02SourceSplitCount("ZHWIKTIONARY_20260701", 39_202, 5_600, 8_400, 2_800, 0),
    ))


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W02CodeFileFreeze:
    """一个 Candidate 承重源码文件的公开字节身份。"""

    repository_path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if self.repository_path not in W02_CODE_FREEZE_PATHS:
            raise W02CompileFreezeError("W-02 code freeze 文件未注册")
        _positive(self.size_bytes, where="W-02 code size")
        _sha256(self.sha256, where="W-02 code sha")

    def to_dict(self) -> dict[str, object]:
        return {
            "repository_path": self.repository_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "W02CodeFileFreeze":
        """从严格 object 恢复源码身份。"""
        raw = exact_dict(value, {
            "repository_path", "sha256", "size_bytes",
        }, where="W02CodeFileFreeze")
        return cls(str(raw["repository_path"]), raw["size_bytes"], str(raw["sha256"]))


def build_w02_code_freeze(repository_root: str | Path) -> tuple[tuple[W02CodeFileFreeze, ...], str]:
    """现场读取预注册生产文件并形成不含 commit 自引用的代码冻结。"""
    root = Path(repository_root).resolve()
    rows: list[W02CodeFileFreeze] = []
    for relative in W02_CODE_FREEZE_PATHS:
        target = (root / Path(*PurePosixPath(relative).parts)).resolve()
        if not target.is_relative_to(root) or not target.is_file() or target.is_symlink():
            raise W02CompileFreezeError("W-02 code freeze 文件缺失或路径非法")
        size, digest = _hash_file(target)
        rows.append(W02CodeFileFreeze(relative, size, digest))
    frozen = tuple(rows)
    return frozen, _hash_value([item.to_dict() for item in frozen])


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W02FileFreeze:
    """一个 owner 物理文件的安全公开身份，不包含真实根路径。"""

    layout_key: str
    root_key: str
    record_kind: str
    split: str
    record_count: int
    content_size_bytes: int
    content_sha256: str
    transport_size_bytes: int
    transport_sha256: str
    first_record_key: tuple[int, ...]
    last_record_key: tuple[int, ...]
    license_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        expected = W02_LAYOUTS.get(self.layout_key)
        if expected is None or expected[:3] != (
                self.root_key, self.record_kind, self.split):
            raise W02CompileFreezeError("W-02 owner file layout 漂移")
        for name in ("record_count", "content_size_bytes", "transport_size_bytes"):
            _positive(getattr(self, name), where=f"W-02 file {name}")
        _sha256(self.content_sha256, where="W-02 content sha")
        _sha256(self.transport_sha256, where="W-02 transport sha")
        for name in ("first_record_key", "last_record_key"):
            value = getattr(self, name)
            if (not isinstance(value, tuple) or not value
                    or any(type(item) is not int or item <= 0 for item in value)):
                raise W02CompileFreezeError(f"W-02 {name} 非法")
        if self.first_record_key > self.last_record_key:
            raise W02CompileFreezeError("W-02 file key range 逆序")
        allowed = ("CC0-1.0", "CC-BY-SA-4.0")
        if (not self.license_ids or self.license_ids != tuple(
                item for item in allowed if item in self.license_ids)):
            raise W02CompileFreezeError("W-02 file license 顺序非法")

    @property
    def storage_relative_path(self) -> str:
        """返回冻结 layout 的物理相对路径，仅供 runtime 内部解析。"""
        return W02_LAYOUTS[self.layout_key][3]

    def to_dict(self) -> dict[str, object]:
        return {
            "content_sha256": self.content_sha256,
            "content_size_bytes": self.content_size_bytes,
            "first_record_key": list(self.first_record_key),
            "last_record_key": list(self.last_record_key),
            "layout_key": self.layout_key,
            "license_ids": list(self.license_ids),
            "record_count": self.record_count,
            "record_kind": self.record_kind,
            "root_key": self.root_key,
            "split": self.split,
            "transport_sha256": self.transport_sha256,
            "transport_size_bytes": self.transport_size_bytes,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "W02FileFreeze":
        """从严格公开 object 恢复 owner 文件身份。"""
        raw = exact_dict(value, {
            "content_sha256", "content_size_bytes", "first_record_key",
            "last_record_key", "layout_key", "license_ids", "record_count",
            "record_kind", "root_key", "split", "transport_sha256",
            "transport_size_bytes",
        }, where="W02FileFreeze")
        if (not isinstance(raw["first_record_key"], list)
                or not isinstance(raw["last_record_key"], list)
                or not isinstance(raw["license_ids"], list)):
            raise W02CompileFreezeError("W-02 file 数组字段非法")
        return cls(
            str(raw["layout_key"]), str(raw["root_key"]),
            str(raw["record_kind"]), str(raw["split"]), raw["record_count"],
            raw["content_size_bytes"], str(raw["content_sha256"]),
            raw["transport_size_bytes"], str(raw["transport_sha256"]),
            tuple(raw["first_record_key"]), tuple(raw["last_record_key"]),
            tuple(str(item) for item in raw["license_ids"]),
        )


def w02_file_freeze_commitment(files: tuple[W02FileFreeze, ...]) -> str:
    """对 canonical layout 顺序的文件身份计算承诺。"""
    if tuple(item.layout_key for item in files) != tuple(W02_LAYOUTS):
        raise W02CompileFreezeError("W-02 file freeze inventory 不完整")
    return _hash_value([item.to_dict() for item in files])


def _source_ref_split_total(plan: W02CompilePlan, split: str) -> int:
    """按 authored 九载体共享一个 SourceRef 的规则汇总来源记录。"""
    authored, ud, wiktionary = plan.source_counts
    return (
        authored.count(split) // plan.authored_carrier_count
        + ud.count(split)
        + wiktionary.count(split)
    )


def _validate_w02_file_counts(
        plan: W02CompilePlan,
        files: tuple[W02FileFreeze, ...],
        ) -> None:
    """逐 layout 核对 SourceRef、Observation、Evidence 和 label 定额。"""
    by_layout = {item.layout_key: item.record_count for item in files}
    train_sources = _source_ref_split_total(plan, "train")
    dev_sources = _source_ref_split_total(plan, "dev")
    private_sources = sum(
        _source_ref_split_total(plan, split)
        for split in ("held_out", "adversarial", "wall")
    )
    expected = {
        "CANDIDATE_SOURCE": train_sources,
        "CANDIDATE_TRAIN_OBSERVATION": plan.split_total("train"),
        "TEACHER_SOURCE": train_sources,
        "TEACHER_TRAIN_EVIDENCE": plan.split_total("train"),
        "DEV_SOURCE": dev_sources,
        "DEV_OBSERVATION": plan.split_total("dev"),
        "DEV_LABEL": plan.split_total("dev"),
        "SHADOW_SOURCE": train_sources + dev_sources,
        "SHADOW_TRAIN_OBSERVATION": plan.split_total("train"),
        "SHADOW_DEV_OBSERVATION": plan.split_total("dev"),
        "PRIVATE_SOURCE": private_sources,
        "PRIVATE_HELD_OUT_OBSERVATION": plan.split_total("held_out"),
        "PRIVATE_ADVERSARIAL_OBSERVATION": plan.split_total("adversarial"),
        "PRIVATE_WALL_OBSERVATION": plan.split_total("wall"),
        "PRIVATE_HELD_OUT_LABEL": plan.split_total("held_out"),
        "PRIVATE_ADVERSARIAL_LABEL": plan.split_total("adversarial"),
        "PRIVATE_WALL_LABEL": plan.split_total("wall"),
    }
    if by_layout != expected:
        raise W02CompileFreezeError("W-02 file record count 与正式定额不一致")


def w02_candidate_contract_value(
        plan: W02CompilePlan,
        files: tuple[W02FileFreeze, ...],
        *,
        code_freeze_sha256: str,
        ) -> dict[str, object]:
    """构造正式 Candidate 运行前固定且尚未消费的输入合同。"""
    _sha256(code_freeze_sha256, where="W-02 code freeze")
    by_layout = {item.layout_key: item for item in files}
    if set(by_layout) != set(W02_LAYOUTS):
        raise W02CompileFreezeError("W-02 Candidate contract 文件不完整")
    return {
        "allowed_workers": list(plan.allowed_workers),
        "candidate_input_commitment": _hash_value([
            by_layout[key].to_dict() for key in (
                "CANDIDATE_SOURCE", "CANDIDATE_TRAIN_OBSERVATION",
                "TEACHER_SOURCE", "TEACHER_TRAIN_EVIDENCE",
            )
        ]),
        "candidate_owner_key": "PH2_V2_CANDIDATE",
        "candidate_write_target": "ISOLATED_W02_CANDIDATE_STORE",
        "code_freeze_sha256": code_freeze_sha256,
        "formal_training_runs": 0,
        "logical_shard_count": plan.logical_shard_count,
        "private_payload_reads": 0,
        "release_key": V2_RELEASE_KEY,
        "stage_key": W02_STAGE_KEY,
        "teacher_calls": 0,
        "train_observation_count": plan.split_total("train"),
        "visible_splits": ["train"],
    }


def w02_first_run_guard_value(
        *,
        candidate_contract_sha256: str,
        code_freeze_sha256: str,
        pack_commitment: str,
        ) -> dict[str, object]:
    """构造可被原子消费一次的 Candidate guard 初始字节。"""
    return {
        "artifact_kind": "PH2_D03_V2_W02_FIRST_RUN_GUARD",
        "candidate_contract_sha256": _sha256(
            candidate_contract_sha256, where="W-02 Candidate contract"),
        "code_freeze_sha256": _sha256(code_freeze_sha256, where="W-02 code freeze"),
        "formal_training_runs": 0,
        "format_version": 1,
        "guard_consumed": 0,
        "guard_version": W02_FIRST_RUN_GUARD_VERSION,
        "pack_commitment": _sha256(pack_commitment, where="W-02 pack commitment"),
        "release_key": V2_RELEASE_KEY,
        "run_id_policy": "NEW_POSITIVE_INTEGER_REQUIRED",
        "stage_key": W02_STAGE_KEY,
        "status": "AVAILABLE",
    }


def publish_w02_first_run_guard(candidate_root: str | Path, value: dict[str, object]) -> str:
    """独占发布首次运行 guard，并返回其规范字节 SHA。"""
    root = Path(candidate_root).resolve()
    target = root / Path(*PurePosixPath(W02_FIRST_RUN_GUARD_AVAILABLE).parts)
    write_immutable_json(value, target)
    payload = target.read_bytes()
    expected = canonical_json_bytes(value) + b"\n"
    if payload != expected:
        raise W02CompileFreezeError("W-02 first-run guard 规范字节漂移")
    return hashlib.sha256(payload).hexdigest()


def consume_w02_first_run_guard(
        candidate_root: str | Path,
        *,
        expected_guard_sha256: str,
        run_id: int,
        run_identity_sha256: str,
        ) -> None:
    """在任何 Candidate 写入前原子消费 guard；失败后不得自动复原。"""
    _positive(run_id, where="W-02 run id")
    _sha256(expected_guard_sha256, where="W-02 guard sha")
    _sha256(run_identity_sha256, where="W-02 run identity")
    root = Path(candidate_root).resolve()
    available = root / Path(*PurePosixPath(W02_FIRST_RUN_GUARD_AVAILABLE).parts)
    consumed = root / Path(*PurePosixPath(W02_FIRST_RUN_GUARD_CONSUMED).parts)
    intent = root / Path(*PurePosixPath(W02_FIRST_RUN_INTENT).parts)
    if consumed.exists() or intent.exists() or not available.is_file():
        raise W02CompileFreezeError("W-02 first-run guard 不可用或已经消费")
    _, actual = _hash_file(available)
    if actual != expected_guard_sha256:
        raise W02CompileFreezeError("W-02 first-run guard 字节漂移")
    consumed.parent.mkdir(parents=True, exist_ok=True)
    os.replace(available, consumed)
    write_immutable_json({
        "artifact_kind": "PH2_D03_V2_W02_RUN_INTENT",
        "format_version": 1,
        "guard_sha256": expected_guard_sha256,
        "run_id": run_id,
        "run_identity_sha256": run_identity_sha256,
        "stage_key": W02_STAGE_KEY,
        "status": "GUARD_CONSUMED_BEFORE_CANDIDATE_WRITE",
    }, intent)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W02CompileFreeze:
    """W-02 正式运行前唯一公开 freeze；不含 private 表层、标签或路径。"""

    parent_ft00_report_sha256: str
    source_snapshot_commitments: tuple[tuple[str, str], ...]
    plan: W02CompilePlan
    files: tuple[W02FileFreeze, ...]
    code_files: tuple[W02CodeFileFreeze, ...]
    code_freeze_sha256: str
    pack_commitment: str
    candidate_contract_sha256: str
    first_run_guard_sha256: str
    private_payload_commitment: str
    private_case_commitment: str
    private_label_commitment: str
    private_cluster_commitment: str
    resource_budget: V2EvaluatorResourceBudget

    def __post_init__(self) -> None:
        _sha256(self.parent_ft00_report_sha256, where="W-02 FT00 parent")
        if (not isinstance(self.source_snapshot_commitments, tuple)
                or tuple(key for key, _ in self.source_snapshot_commitments) != W02_SOURCES):
            raise W02CompileFreezeError("W-02 snapshot commitment 顺序漂移")
        for _, digest in self.source_snapshot_commitments:
            _sha256(digest, where="W-02 snapshot commitment")
        if self.plan.to_dict() != formal_w02_compile_plan().to_dict():
            raise W02CompileFreezeError("W-02 正式 plan 漂移")
        if tuple(item.repository_path for item in self.code_files) != W02_CODE_FREEZE_PATHS:
            raise W02CompileFreezeError("W-02 code file inventory 漂移")
        if self.code_freeze_sha256 != _hash_value(
                [item.to_dict() for item in self.code_files]):
            raise W02CompileFreezeError("W-02 code freeze 摘要漂移")
        if self.pack_commitment != w02_file_freeze_commitment(self.files):
            raise W02CompileFreezeError("W-02 pack commitment 漂移")
        _validate_w02_file_counts(self.plan, self.files)
        candidate = w02_candidate_contract_value(
            self.plan, self.files, code_freeze_sha256=self.code_freeze_sha256)
        if self.candidate_contract_sha256 != _hash_value(candidate):
            raise W02CompileFreezeError("W-02 Candidate contract 摘要漂移")
        for name in (
            "first_run_guard_sha256", "private_payload_commitment",
            "private_case_commitment", "private_label_commitment",
            "private_cluster_commitment",
        ):
            _sha256(getattr(self, name), where=f"W-02 {name}")
        expected_private_payload = _hash_value([
            item.to_dict() for item in self.files
            if item.root_key == "PRIVATE_EVALUATOR_ROOT"
        ])
        if self.private_payload_commitment != expected_private_payload:
            raise W02CompileFreezeError("W-02 private payload commitment 漂移")
        if not isinstance(self.resource_budget, V2EvaluatorResourceBudget):
            raise W02CompileFreezeError("W-02 evaluator resource 类型错误")

    def to_dict(self) -> dict[str, object]:
        policy = next(
            item for item in V2_STAGE_EVALUATION_POLICIES
            if item.stage_key == W02_STAGE_KEY)
        candidate = w02_candidate_contract_value(
            self.plan, self.files, code_freeze_sha256=self.code_freeze_sha256)
        return {
            "artifact_kind": W02_COMPILE_FREEZE_KIND,
            "artifact_version": W02_COMPILE_FREEZE_VERSION,
            "candidate_contract": candidate,
            "candidate_contract_sha256": self.candidate_contract_sha256,
            "code_files": [item.to_dict() for item in self.code_files],
            "code_freeze_sha256": self.code_freeze_sha256,
            "evaluator_policy": policy.to_dict(),
            "files": [item.to_dict() for item in self.files],
            "first_run_guard_sha256": self.first_run_guard_sha256,
            "formal_private_evaluation_runs": W02_FORMAL_PRIVATE_RUNS,
            "formal_training_runs": W02_FORMAL_TRAINING_RUNS,
            "format_version": 1,
            "ft00_complete": 1,
            "language_capability_mastered": 0,
            "language_readiness": 0,
            "next_action": "W02_FORMAL_CANDIDATE_FIRST_RUN",
            "pack_commitment": self.pack_commitment,
            "parent_ft00_report_sha256": self.parent_ft00_report_sha256,
            "plan": self.plan.to_dict(),
            "private_commitments": {
                "case_commitment": self.private_case_commitment,
                "cluster_commitment": self.private_cluster_commitment,
                "label_commitment": self.private_label_commitment,
                "payload_commitment": self.private_payload_commitment,
            },
            "private_payload_reads": 0,
            "release_key": V2_RELEASE_KEY,
            "resource_budget": self.resource_budget.to_dict(),
            "source_snapshot_commitments": {
                key: value for key, value in self.source_snapshot_commitments
            },
            "stage_key": W02_STAGE_KEY,
            "status": "W02_COMPILE_FREEZE_COMPLETE",
            "teacher_calls": 0,
        }

    def canonical_bytes(self) -> bytes:
        """返回带单尾换行的规范公开字节。"""
        return canonical_json_bytes(self.to_dict()) + b"\n"

    def sha256(self) -> str:
        """返回公开 freeze 文件内容摘要。"""
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


def publish_w02_compile_freeze(
        freeze: W02CompileFreeze,
        repository_root: str | Path,
        ) -> Path:
    """独占发布 W-02 compile freeze，已有不同字节时拒绝覆盖。"""
    if not isinstance(freeze, W02CompileFreeze):
        raise W02CompileFreezeError("W-02 freeze 类型错误")
    root = Path(repository_root).resolve()
    target = root / Path(*PurePosixPath(W02_COMPILE_FREEZE_PATH).parts)
    write_immutable_json(freeze.to_dict(), target)
    if target.read_bytes() != freeze.canonical_bytes():
        raise W02CompileFreezeError("W-02 freeze 发布字节漂移")
    return target


def read_w02_compile_freeze(repository_root: str | Path) -> W02CompileFreeze:
    """严格回读公开 freeze，并复算 plan、文件、代码和 Candidate 合同。"""
    root = Path(repository_root).resolve()
    target = root / Path(*PurePosixPath(W02_COMPILE_FREEZE_PATH).parts)
    value = read_canonical_object(target)
    raw = exact_dict(value, {
        "artifact_kind", "artifact_version", "candidate_contract",
        "candidate_contract_sha256", "code_files", "code_freeze_sha256",
        "evaluator_policy", "files", "first_run_guard_sha256",
        "formal_private_evaluation_runs", "formal_training_runs", "format_version",
        "ft00_complete", "language_capability_mastered", "language_readiness",
        "next_action", "pack_commitment", "parent_ft00_report_sha256", "plan",
        "private_commitments", "private_payload_reads", "release_key",
        "resource_budget", "source_snapshot_commitments", "stage_key", "status",
        "teacher_calls",
    }, where="W02CompileFreeze")
    if (raw["artifact_kind"] != W02_COMPILE_FREEZE_KIND
            or raw["artifact_version"] != W02_COMPILE_FREEZE_VERSION
            or raw["format_version"] != 1 or raw["release_key"] != V2_RELEASE_KEY
            or raw["stage_key"] != W02_STAGE_KEY
            or raw["status"] != "W02_COMPILE_FREEZE_COMPLETE"
            or raw["next_action"] != "W02_FORMAL_CANDIDATE_FIRST_RUN"):
        raise W02CompileFreezeError("W-02 freeze 顶层身份漂移")
    zero_fields = (
        "formal_private_evaluation_runs", "formal_training_runs",
        "language_capability_mastered", "language_readiness",
        "private_payload_reads", "teacher_calls",
    )
    if raw["ft00_complete"] != 1 or any(raw[name] != 0 for name in zero_fields):
        raise W02CompileFreezeError("W-02 freeze 初始状态漂移")
    if not isinstance(raw["source_snapshot_commitments"], dict):
        raise W02CompileFreezeError("W-02 snapshot commitments 类型错误")
    snapshots = tuple(
        (key, str(raw["source_snapshot_commitments"].get(key, "")))
        for key in W02_SOURCES
    )
    if set(raw["source_snapshot_commitments"]) != set(W02_SOURCES):
        raise W02CompileFreezeError("W-02 snapshot commitments 集合漂移")
    if not isinstance(raw["plan"], dict):
        raise W02CompileFreezeError("W-02 plan 类型错误")
    plan_raw = raw["plan"]
    exact_dict(plan_raw, {
        "allowed_workers", "authored_carrier_count", "logical_shard_count",
        "source_counts", "split_totals", "total_observations",
        "train_observation_target",
    }, where="W02CompilePlan")
    if not isinstance(plan_raw["source_counts"], list):
        raise W02CompileFreezeError("W-02 source_counts 类型错误")
    plan = W02CompilePlan(tuple(
        W02SourceSplitCount.from_dict(item) for item in plan_raw["source_counts"]
    ), plan_raw["authored_carrier_count"], plan_raw["logical_shard_count"],
       tuple(plan_raw["allowed_workers"]))
    if plan.to_dict() != plan_raw:
        raise W02CompileFreezeError("W-02 plan canonical 投影漂移")
    if not isinstance(raw["files"], list) or not isinstance(raw["code_files"], list):
        raise W02CompileFreezeError("W-02 freeze inventory 类型错误")
    files = tuple(W02FileFreeze.from_dict(item) for item in raw["files"])
    code_files = tuple(W02CodeFileFreeze.from_dict(item) for item in raw["code_files"])
    private = exact_dict(raw["private_commitments"], {
        "case_commitment", "cluster_commitment", "label_commitment",
        "payload_commitment",
    }, where="W02 private commitments")
    freeze = W02CompileFreeze(
        str(raw["parent_ft00_report_sha256"]), snapshots, plan, files, code_files,
        str(raw["code_freeze_sha256"]), str(raw["pack_commitment"]),
        str(raw["candidate_contract_sha256"]), str(raw["first_run_guard_sha256"]),
        str(private["payload_commitment"]), str(private["case_commitment"]),
        str(private["label_commitment"]), str(private["cluster_commitment"]),
        V2EvaluatorResourceBudget.from_dict(raw["resource_budget"]),
    )
    policy = next(
        item for item in V2_STAGE_EVALUATION_POLICIES
        if item.stage_key == W02_STAGE_KEY)
    if (raw["candidate_contract"] != w02_candidate_contract_value(
            plan, files, code_freeze_sha256=freeze.code_freeze_sha256)
            or raw["evaluator_policy"] != policy.to_dict()
            or freeze.canonical_bytes() != target.read_bytes()):
        raise W02CompileFreezeError("W-02 freeze canonical 回读漂移")
    return freeze


__all__ = [
    "W02_ALLOWED_WORKERS",
    "W02_CODE_FREEZE_PATHS",
    "W02_COMPILE_FREEZE_PATH",
    "W02_FIRST_RUN_GUARD_AVAILABLE",
    "W02_FIRST_RUN_GUARD_CONSUMED",
    "W02_LAYOUTS",
    "W02_LOGICAL_SHARD_COUNT",
    "W02_SPLITS",
    "W02_STAGE_KEY",
    "W02_TARGET_TRAIN_OBSERVATIONS",
    "W02CodeFileFreeze",
    "W02CompileFreeze",
    "W02CompileFreezeError",
    "W02CompilePlan",
    "W02FileFreeze",
    "W02SourceSplitCount",
    "build_w02_code_freeze",
    "consume_w02_first_run_guard",
    "formal_w02_compile_plan",
    "publish_w02_compile_freeze",
    "publish_w02_first_run_guard",
    "read_w02_compile_freeze",
    "w02_candidate_contract_value",
    "w02_file_freeze_commitment",
    "w02_first_run_guard_value",
]
