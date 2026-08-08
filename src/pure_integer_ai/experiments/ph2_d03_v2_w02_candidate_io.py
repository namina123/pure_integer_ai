"""PH2-D03-V2 W-02 正式 Candidate/teacher 输入的只读边界。"""
from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
from pathlib import Path, PurePosixPath
from typing import Iterator

from pure_integer_ai.experiments.ph2_d03_v2_schema import validate_v2_record
from pure_integer_ai.experiments.ph2_d03_v2_w02_contract import (
    W02_LAYOUTS,
    W02CompileFreeze,
    W02FileFreeze,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    ObservationRecord,
    SourceRefRecord,
    TeacherEvidenceRecord,
    canonical_json_bytes,
    parse_canonical_json_bytes,
)


W02_FORMAL_LAYOUT_PATHS = {
    "CANDIDATE_SOURCE": "source/source_refs.jsonl.gz",
    "CANDIDATE_TRAIN_OBSERVATION": "observations/train.jsonl.gz",
    "TEACHER_SOURCE": "source/source_refs.jsonl.gz",
    "TEACHER_TRAIN_EVIDENCE": "teacher/train.evidence.jsonl.gz",
}


# object-model: exception
class W02CandidateInputError(RuntimeError):
    """正式 Candidate 输入根、文件身份或记录配对不满足冻结合同。"""


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
            size += len(block)
    return size, digest.hexdigest()


def _safe_file(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if (not relative or "\\" in relative or pure.is_absolute()
            or pure.as_posix() != relative or any(part in {"", ".", ".."}
                                                  for part in pure.parts)):
        raise W02CandidateInputError("W-02 Candidate 输入路径非法")
    current = root
    for part in pure.parts:
        current /= part
        if current.is_symlink():
            raise W02CandidateInputError("W-02 Candidate 输入不得经过 symlink")
    target = (root / Path(*pure.parts)).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise W02CandidateInputError("W-02 Candidate 输入文件缺失或逃逸")
    return target


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W02FormalInputRoots:
    """只包含 Candidate 和 teacher 的两个物理输入根。"""

    candidate_root: Path
    teacher_root: Path

    def __post_init__(self) -> None:
        candidate = Path(self.candidate_root).resolve()
        teacher = Path(self.teacher_root).resolve()
        object.__setattr__(self, "candidate_root", candidate)
        object.__setattr__(self, "teacher_root", teacher)
        if (not candidate.is_dir() or not teacher.is_dir() or candidate == teacher
                or candidate.is_relative_to(teacher) or teacher.is_relative_to(candidate)):
            raise W02CandidateInputError("W-02 Candidate/teacher roots 必须存在且隔离")
        forbidden = {"private-evaluator", "dev-calibration", "shadow-audit"}
        if candidate.name in forbidden or teacher.name in forbidden:
            raise W02CandidateInputError("W-02 正式训练 root 指向非训练 owner")

    def root_for_layout(self, layout_key: str) -> Path:
        if layout_key.startswith("CANDIDATE_"):
            return self.candidate_root
        if layout_key.startswith("TEACHER_"):
            return self.teacher_root
        raise W02CandidateInputError("W-02 Candidate reader 拒绝非训练 layout")


def _freeze_for_layout(freeze: W02CompileFreeze, layout_key: str) -> W02FileFreeze:
    if not isinstance(freeze, W02CompileFreeze):
        raise TypeError("W-02 compile freeze 类型错误")
    if layout_key not in W02_FORMAL_LAYOUT_PATHS:
        raise W02CandidateInputError("W-02 Candidate reader layout 未注册")
    matches = tuple(item for item in freeze.files if item.layout_key == layout_key)
    if len(matches) != 1:
        raise W02CandidateInputError("W-02 Candidate layout freeze 不唯一")
    item = matches[0]
    root_key, record_kind, split, relative = W02_LAYOUTS[layout_key]
    if (item.root_key != root_key or item.record_kind != record_kind
            or item.split != split or relative != W02_FORMAL_LAYOUT_PATHS[layout_key]):
        raise W02CandidateInputError("W-02 Candidate layout 与 parent freeze 漂移")
    return item


def verify_w02_visible_transport(
        freeze: W02CompileFreeze,
        roots: W02FormalInputRoots,
        ) -> dict[str, tuple[int, str]]:
    """只校验四个 train-visible 物理文件，不打开任何 private/dev payload。"""
    if not isinstance(roots, W02FormalInputRoots):
        raise TypeError("W-02 formal roots 类型错误")
    report = {}
    for layout_key, relative in W02_FORMAL_LAYOUT_PATHS.items():
        identity = _freeze_for_layout(freeze, layout_key)
        target = _safe_file(roots.root_for_layout(layout_key), relative)
        size, digest = _sha256_file(target)
        if (size != identity.transport_size_bytes
                or digest != identity.transport_sha256):
            raise W02CandidateInputError("W-02 visible transport identity 漂移")
        report[layout_key] = (size, digest)
    return report


def iter_w02_frozen_records(
        freeze: W02CompileFreeze,
        roots: W02FormalInputRoots,
        layout_key: str,
        ) -> Iterator[object]:
    """完整消费一个 train-visible gzip JSONL，并在 EOF 闭合内容身份。"""
    identity = _freeze_for_layout(freeze, layout_key)
    target = _safe_file(
        roots.root_for_layout(layout_key), W02_FORMAL_LAYOUT_PATHS[layout_key])
    size, digest = _sha256_file(target)
    if size != identity.transport_size_bytes or digest != identity.transport_sha256:
        raise W02CandidateInputError("W-02 visible transport identity 漂移")
    content_digest = hashlib.sha256()
    content_size = 0
    count = 0
    first_key: tuple[int, ...] | None = None
    last_key: tuple[int, ...] | None = None
    previous_key: tuple[int, ...] | None = None
    try:
        with target.open("rb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="rb") as stream:
                for line_number, line in enumerate(stream, start=1):
                    if not line.endswith(b"\n") or line.endswith(b"\n\n"):
                        raise W02CandidateInputError(
                            f"W-02 visible JSONL 第 {line_number} 行换行非法")
                    content_digest.update(line)
                    content_size += len(line)
                    value = parse_canonical_json_bytes(line[:-1], require_object=True)
                    assert isinstance(value, dict)
                    record = validate_v2_record(value)
                    if getattr(record, "RECORD_KIND", None) != identity.record_kind:
                        raise W02CandidateInputError("W-02 visible record kind 漂移")
                    if (identity.split
                            and getattr(record, "split", identity.split) != identity.split):
                        raise W02CandidateInputError("W-02 visible record split 漂移")
                    key = record.stable_key.components
                    if previous_key is not None and key <= previous_key:
                        raise W02CandidateInputError("W-02 visible stable key 未严格排序")
                    previous_key = key
                    first_key = key if first_key is None else first_key
                    last_key = key
                    count += 1
                    yield record
    except (OSError, EOFError, ValueError) as error:
        if isinstance(error, W02CandidateInputError):
            raise
        raise W02CandidateInputError("W-02 visible gzip/JSONL 读取失败") from error
    if (count != identity.record_count or content_size != identity.content_size_bytes
            or content_digest.hexdigest() != identity.content_sha256
            or first_key != identity.first_record_key
            or last_key != identity.last_record_key):
        raise W02CandidateInputError("W-02 visible content identity 漂移")


def scan_w02_train_sources(
        freeze: W02CompileFreeze,
        roots: W02FormalInputRoots,
        ) -> tuple[int, str]:
    """回读 Candidate source refs，并证明 teacher source transport 与其相同。"""
    candidate = _freeze_for_layout(freeze, "CANDIDATE_SOURCE")
    teacher = _freeze_for_layout(freeze, "TEACHER_SOURCE")
    candidate_payload = candidate.to_dict()
    teacher_payload = teacher.to_dict()
    for field in ("layout_key", "root_key"):
        candidate_payload.pop(field)
        teacher_payload.pop(field)
    if candidate_payload != teacher_payload:
        raise W02CandidateInputError("W-02 Candidate/teacher source freeze 内容不一致")
    count = 0
    digest = hashlib.sha256()
    for record in iter_w02_frozen_records(
            freeze, roots, "CANDIDATE_SOURCE"):
        if not isinstance(record, SourceRefRecord):
            raise W02CandidateInputError("W-02 Candidate source 类型错误")
        digest.update(canonical_json_bytes(record.stable_key.to_list()))
        count += 1
    return count, digest.hexdigest()


def iter_w02_training_pairs(
        freeze: W02CompileFreeze,
        roots: W02FormalInputRoots,
        ) -> Iterator[tuple[ObservationRecord, TeacherEvidenceRecord]]:
    """单遍配对 51,200 个 Observation/Evidence，不物化全量 payload。"""
    observations = iter_w02_frozen_records(
        freeze, roots, "CANDIDATE_TRAIN_OBSERVATION")
    evidence = iter_w02_frozen_records(
        freeze, roots, "TEACHER_TRAIN_EVIDENCE")
    count = 0
    while True:
        try:
            observation = next(observations)
        except StopIteration:
            observation = None
        try:
            teacher = next(evidence)
        except StopIteration:
            teacher = None
        if observation is None or teacher is None:
            if observation is not None or teacher is not None:
                raise W02CandidateInputError("W-02 Observation/Evidence 数量不一致")
            break
        if not isinstance(observation, ObservationRecord):
            raise W02CandidateInputError("W-02 Candidate Observation 类型错误")
        if not isinstance(teacher, TeacherEvidenceRecord):
            raise W02CandidateInputError("W-02 Teacher Evidence 类型错误")
        if teacher.observation_key != observation.stable_key:
            raise W02CandidateInputError("W-02 Observation/Evidence 顺序或绑定漂移")
        count += 1
        yield observation, teacher
    if count != freeze.plan.split_total("train"):
        raise W02CandidateInputError("W-02 training pair 数量与正式 plan 漂移")


__all__ = [
    "W02CandidateInputError",
    "W02FormalInputRoots",
    "iter_w02_frozen_records",
    "iter_w02_training_pairs",
    "scan_w02_train_sources",
    "verify_w02_visible_transport",
]
