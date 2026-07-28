"""LC 第一阶段课程冻结、学习目标和零执行状态的纯合同。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    canonical_json_line,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_language_coverage_contract import (
    CAPABILITY_KEYS,
    SAMPLE_FAMILIES,
)


LANGUAGE_OBJECTIVE_KEYS = (
    "CONTROLLED_PERTURBATION",
    "CROSS_CONTEXT_CONSISTENCY",
    "GENERATION_ADOPTION",
    "GENERATION_FAILURE",
    "INTEGER_DESCRIPTION_LENGTH",
    "MASKED_SPAN",
    "MASKED_TOKEN",
    "NEXT_DISCOURSE_UNIT",
    "NEXT_SPAN",
    "NEXT_TOKEN",
    "ORDER_RECOVERY",
)
TEXT_FIDELITY_EVALUATOR_DIMENSIONS = (
    "CANDIDATE_LATTICE",
    "GENERATION_SURFACE_FIDELITY",
    "IRREVERSIBLE_LOSS_DISCLOSURE",
    "LEARNING_OBJECTIVE_BINDING",
    "NORMALIZATION_RECEIPT",
    "RAW_OBSERVATION_PRESERVATION",
    "RETENTION_REVERIFY",
)
TEXT_FIDELITY_PAYLOAD_KEYS = (
    "candidate_group",
    "candidate_id",
    "candidate_kind",
    "derived_candidate",
    "description_length",
    "information_loss",
    "loss_kind",
    "normalization_receipt",
    "objective_keys",
    "raw_observation",
    "retention_anchor_id",
    "sample_family",
    "selection_state",
)
COURSE_TASK_KEYS = ("LC-01", "LC-15-INITIAL")
COURSE_CAPABILITY_KEYS = ("RAW_TEXT_NOISE", "TYPED_LEARNING_OBJECTIVES")
COURSE_SPLIT_AXES = (
    "content_group",
    "family",
    "shape_group",
    "source_cluster",
    "template_group",
)
COURSE_RETENTION_PROTOCOLS = (
    "A_TO_B_REVERIFY_A",
    "DUMP_RESUME",
    "ROLLBACK_SCOPE_CONTRACTION",
    "SOURCE_WITHDRAWAL_LOCAL_INVALIDATION",
)
COURSE_VERIFIER_NE_CONDITIONS = (
    "CAPABILITY_LEARNED_REQUESTED",
    "NO_EVALUATOR_LABEL",
    "RUNTIME_GENERALIZATION_REQUESTED",
    "SEMANTIC_CORRECTNESS_REQUESTED",
)
COURSE_INVARIANTS = {
    "derived_candidate_only": 1,
    "evaluator_owner_read_only": 1,
    "irreversible_auto_adoption_allowed": 0,
    "raw_observation_append_only": 1,
    "student_visible_expected": 0,
    "teacher_call_required": 0,
}


class LanguageCourseContractError(RuntimeError):
    """课程冻结、目标分型或 manifest 字段不完整。"""


def _exact_keys(
        value: Any,
        keys: set[str],
        *,
        where: str) -> dict[str, Any]:
    """要求 JSON object 字段集合精确相等。"""
    if not isinstance(value, dict) or set(value) != keys:
        raise LanguageCourseContractError(f"{where} 字段不精确")
    return value


def _text(value: Any, *, where: str) -> str:
    """要求非空、无首尾空白文本。"""
    if (not isinstance(value, str) or not value
            or value.strip() != value):
        raise LanguageCourseContractError(f"{where} 必须是非空无首尾空白文本")
    return value


def _positive(value: Any, *, where: str) -> int:
    """要求正严格整数。"""
    if type(value) is not int or value <= 0:
        raise LanguageCourseContractError(f"{where} 必须是正严格整数")
    return value


def _zero(value: Any, *, where: str) -> int:
    """要求冻结 artifact 的执行状态为严格整数零。"""
    if type(value) is not int or value != 0:
        raise LanguageCourseContractError(f"{where} 必须为 0")
    return value


def _flag(value: Any, *, where: str) -> int:
    """要求严格整数布尔标志。"""
    if type(value) is not int or value not in (0, 1):
        raise LanguageCourseContractError(f"{where} 必须为 0/1")
    return value


def _sha256(value: Any, *, where: str) -> str:
    """要求小写 SHA-256。"""
    text = _text(value, where=where)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise LanguageCourseContractError(f"{where} 必须是小写 SHA-256")
    return text


def _relative_path(value: Any, *, where: str) -> str:
    """要求可迁移、安全 POSIX 相对路径。"""
    text = _text(value, where=where)
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or "\\" in text:
        raise LanguageCourseContractError(f"{where} 必须是安全 POSIX 相对路径")
    return text


def _tuple(
        value: Any,
        *,
        where: str,
        allow_empty: bool = False,
        ordered: bool = True) -> tuple[str, ...]:
    """要求文本 tuple 去重，并按声明决定是否排序。"""
    if not isinstance(value, tuple) or (not value and not allow_empty):
        raise LanguageCourseContractError(f"{where} 必须是文本 tuple")
    if any(not isinstance(item, str) or not item for item in value):
        raise LanguageCourseContractError(f"{where} 含非法文本")
    if len(value) != len(set(value)):
        raise LanguageCourseContractError(f"{where} 含重复项")
    if ordered and value != tuple(sorted(value)):
        raise LanguageCourseContractError(f"{where} 必须排序")
    return value


@dataclass(frozen=True)
class LearningObjectiveSpec:
    """LC-15 初版中一个候选淘汰目标及 Evidence owner。"""

    objective_key: str
    capability_keys: tuple[str, ...]
    elimination_signal: str
    training_evidence_owner: str
    evaluation_evidence_owner: str
    verifier_dimension: str
    elimination_required: int
    external_signal_allowed: int
    runtime_pass_authority: int

    def __post_init__(self) -> None:
        if self.objective_key not in LANGUAGE_OBJECTIVE_KEYS:
            raise LanguageCourseContractError("学习目标 key 未登记")
        _tuple(self.capability_keys, where="objective capability_keys")
        if any(key not in CAPABILITY_KEYS for key in self.capability_keys):
            raise LanguageCourseContractError("学习目标 capability 未登记")
        _text(self.elimination_signal, where="objective elimination_signal")
        if self.training_evidence_owner != "TEACHER_EVIDENCE":
            raise LanguageCourseContractError("训练目标 owner 必须是 TEACHER_EVIDENCE")
        if self.evaluation_evidence_owner != "EVALUATOR_LABEL":
            raise LanguageCourseContractError("评测目标 owner 必须是 EVALUATOR_LABEL")
        if self.verifier_dimension not in TEXT_FIDELITY_EVALUATOR_DIMENSIONS:
            raise LanguageCourseContractError("学习目标 verifier dimension 未登记")
        if _flag(self.elimination_required, where="elimination_required") != 1:
            raise LanguageCourseContractError("学习目标必须声明独立淘汰信号")
        if _flag(self.external_signal_allowed, where="external_signal_allowed") != 0:
            raise LanguageCourseContractError("外部观察不得混入纯目标信号")
        if _flag(self.runtime_pass_authority, where="runtime_pass_authority") != 0:
            raise LanguageCourseContractError("课程目标不得签发 runtime PASS")

    def to_dict(self) -> dict[str, Any]:
        """导出规范目标记录。"""
        return {
            "capability_keys": list(self.capability_keys),
            "elimination_required": self.elimination_required,
            "elimination_signal": self.elimination_signal,
            "evaluation_evidence_owner": self.evaluation_evidence_owner,
            "external_signal_allowed": self.external_signal_allowed,
            "objective_key": self.objective_key,
            "runtime_pass_authority": self.runtime_pass_authority,
            "training_evidence_owner": self.training_evidence_owner,
            "verifier_dimension": self.verifier_dimension,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "LearningObjectiveSpec":
        """从精确 JSON object 恢复目标记录。"""
        raw = _exact_keys(value, {
            "capability_keys", "elimination_required", "elimination_signal",
            "evaluation_evidence_owner", "external_signal_allowed",
            "objective_key", "runtime_pass_authority",
            "training_evidence_owner", "verifier_dimension",
        }, where="LearningObjectiveSpec")
        return cls(
            str(raw["objective_key"]),
            tuple(str(item) for item in raw["capability_keys"]),
            str(raw["elimination_signal"]),
            str(raw["training_evidence_owner"]),
            str(raw["evaluation_evidence_owner"]),
            str(raw["verifier_dimension"]),
            raw["elimination_required"],
            raw["external_signal_allowed"],
            raw["runtime_pass_authority"],
        )


@dataclass(frozen=True)
class LanguageCourseManifest:
    """冻结 LC-01 课程与 LC-15 初版目标，不声明 runtime 已学会。"""

    format_version: int
    artifact_version: str
    course_status: str
    objective_taxonomy_status: str
    task_keys: tuple[str, ...]
    capability_exit_states: CanonicalJsonObject
    stage: str
    substage: str
    source_key: str
    license_id: str
    sample_relative_path: str
    sample_sha256: str
    seed_count: int
    sample_families: tuple[str, ...]
    split_axes: tuple[str, ...]
    teacher_family_keys: tuple[str, ...]
    evaluator_family_keys: tuple[str, ...]
    teacher_template_keys: tuple[str, ...]
    evaluator_template_keys: tuple[str, ...]
    payload_kind: str
    payload_contract_keys: tuple[str, ...]
    objective_taxonomy_version: str
    objectives: tuple[LearningObjectiveSpec, ...]
    evaluator_dimensions: tuple[str, ...]
    verifier_ne_conditions: tuple[str, ...]
    retention_protocols: tuple[str, ...]
    pack_manifest_relative_path: str
    pack_manifest_sha256: str
    pack_record_count: int
    pack_splits: tuple[str, ...]
    invariants: CanonicalJsonObject
    d03_published: int = 0
    w01_started: int = 0
    formal_training_runs: int = 0
    teacher_calls: int = 0
    learning_state_writes: int = 0
    mastered_claims: int = 0
    readiness_claims: int = 0

    def __post_init__(self) -> None:
        if self.format_version != 1:
            raise LanguageCourseContractError("course format_version 非法")
        _text(self.artifact_version, where="course artifact_version")
        if self.course_status != "COURSE_FROZEN":
            raise LanguageCourseContractError("LC-01 course_status 必须冻结")
        if self.objective_taxonomy_status != "INITIAL_FROZEN":
            raise LanguageCourseContractError("LC-15 初版状态非法")
        if self.task_keys != COURSE_TASK_KEYS:
            raise LanguageCourseContractError("course task_keys 不完整")
        states = self.capability_exit_states.to_value()
        if states != {
                "RAW_TEXT_NOISE": "COURSE_FROZEN",
                "TYPED_LEARNING_OBJECTIVES": "PARTIAL_COURSE"}:
            raise LanguageCourseContractError("course capability 退出状态不诚实")
        if self.stage != "W-02" or self.substage != "LC01_TEXT_FIDELITY":
            raise LanguageCourseContractError("course 阶段位置非法")
        if self.source_key != "AUTHORED_CC0_V1" or self.license_id != "CC0-1.0":
            raise LanguageCourseContractError("course 来源或许可分区非法")
        _relative_path(self.sample_relative_path, where="sample_relative_path")
        _sha256(self.sample_sha256, where="sample_sha256")
        _positive(self.seed_count, where="seed_count")
        if self.sample_families != SAMPLE_FAMILIES:
            raise LanguageCourseContractError("course 七类 sample family 未列全")
        if self.split_axes != COURSE_SPLIT_AXES:
            raise LanguageCourseContractError("course split axes 未冻结")
        for name, value in (
                ("teacher_family_keys", self.teacher_family_keys),
                ("evaluator_family_keys", self.evaluator_family_keys),
                ("teacher_template_keys", self.teacher_template_keys),
                ("evaluator_template_keys", self.evaluator_template_keys)):
            _tuple(value, where=name)
        if set(self.teacher_family_keys) & set(self.evaluator_family_keys):
            raise LanguageCourseContractError("teacher/evaluator family 泄漏")
        if set(self.teacher_template_keys) & set(self.evaluator_template_keys):
            raise LanguageCourseContractError("teacher/evaluator template 泄漏")
        if self.payload_kind != "TextFidelityCandidateV1":
            raise LanguageCourseContractError("course payload_kind 非法")
        if self.payload_contract_keys != TEXT_FIDELITY_PAYLOAD_KEYS:
            raise LanguageCourseContractError("course payload contract keys 漂移")
        _text(self.objective_taxonomy_version, where="objective_taxonomy_version")
        if (not isinstance(self.objectives, tuple)
                or any(not isinstance(item, LearningObjectiveSpec)
                       for item in self.objectives)):
            raise LanguageCourseContractError("course objectives 类型非法")
        if tuple(item.objective_key for item in self.objectives) != (
                LANGUAGE_OBJECTIVE_KEYS):
            raise LanguageCourseContractError("course objectives 未列全或未排序")
        if self.evaluator_dimensions != TEXT_FIDELITY_EVALUATOR_DIMENSIONS:
            raise LanguageCourseContractError("course evaluator dimensions 未列全")
        if self.verifier_ne_conditions != COURSE_VERIFIER_NE_CONDITIONS:
            raise LanguageCourseContractError("course verifier NE 条件未冻结")
        if self.retention_protocols != COURSE_RETENTION_PROTOCOLS:
            raise LanguageCourseContractError("course retention 协议未冻结")
        _relative_path(
            self.pack_manifest_relative_path,
            where="pack_manifest_relative_path",
        )
        _sha256(self.pack_manifest_sha256, where="pack_manifest_sha256")
        _positive(self.pack_record_count, where="pack_record_count")
        if self.pack_splits != ("train", "held_out"):
            raise LanguageCourseContractError("course pack_splits 非法")
        if self.invariants.to_value() != COURSE_INVARIANTS:
            raise LanguageCourseContractError("course invariants 不完整")
        for name in (
                "d03_published", "w01_started", "formal_training_runs",
                "teacher_calls", "learning_state_writes", "mastered_claims",
                "readiness_claims"):
            _zero(getattr(self, name), where=f"course {name}")

    def to_dict(self) -> dict[str, Any]:
        """导出规范课程冻结 manifest。"""
        return {
            "artifact_kind": "PH2_LANGUAGE_COURSE_FREEZE",
            "artifact_version": self.artifact_version,
            "capability_exit_states": self.capability_exit_states.to_value(),
            "course_status": self.course_status,
            "d03_published": self.d03_published,
            "evaluator_dimensions": list(self.evaluator_dimensions),
            "evaluator_family_keys": list(self.evaluator_family_keys),
            "evaluator_template_keys": list(self.evaluator_template_keys),
            "formal_training_runs": self.formal_training_runs,
            "format_version": self.format_version,
            "invariants": self.invariants.to_value(),
            "learning_state_writes": self.learning_state_writes,
            "license_id": self.license_id,
            "mastered_claims": self.mastered_claims,
            "objective_taxonomy_status": self.objective_taxonomy_status,
            "objective_taxonomy_version": self.objective_taxonomy_version,
            "objectives": [item.to_dict() for item in self.objectives],
            "pack_manifest_relative_path": self.pack_manifest_relative_path,
            "pack_manifest_sha256": self.pack_manifest_sha256,
            "pack_record_count": self.pack_record_count,
            "pack_splits": list(self.pack_splits),
            "payload_contract_keys": list(self.payload_contract_keys),
            "payload_kind": self.payload_kind,
            "readiness_claims": self.readiness_claims,
            "retention_protocols": list(self.retention_protocols),
            "sample_families": list(self.sample_families),
            "sample_relative_path": self.sample_relative_path,
            "sample_sha256": self.sample_sha256,
            "seed_count": self.seed_count,
            "source_key": self.source_key,
            "split_axes": list(self.split_axes),
            "stage": self.stage,
            "substage": self.substage,
            "task_keys": list(self.task_keys),
            "teacher_calls": self.teacher_calls,
            "teacher_family_keys": list(self.teacher_family_keys),
            "teacher_template_keys": list(self.teacher_template_keys),
            "verifier_ne_conditions": list(self.verifier_ne_conditions),
            "w01_started": self.w01_started,
        }

    def canonical_bytes(self) -> bytes:
        """返回单换行结尾的规范 artifact 字节。"""
        return canonical_json_line(self.to_dict())

    def sha256(self) -> str:
        """返回规范课程 manifest SHA-256。"""
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "LanguageCourseManifest":
        """从精确 JSON object 恢复课程冻结 manifest。"""
        raw = _exact_keys(value, {
            "artifact_kind", "artifact_version", "capability_exit_states",
            "course_status", "d03_published", "evaluator_dimensions",
            "evaluator_family_keys", "evaluator_template_keys",
            "formal_training_runs", "format_version", "invariants",
            "learning_state_writes", "license_id", "mastered_claims",
            "objective_taxonomy_status", "objective_taxonomy_version",
            "objectives", "pack_manifest_relative_path",
            "pack_manifest_sha256", "pack_record_count", "pack_splits",
            "payload_contract_keys", "payload_kind", "readiness_claims",
            "retention_protocols", "sample_families",
            "sample_relative_path", "sample_sha256", "seed_count",
            "source_key", "split_axes", "stage", "substage", "task_keys",
            "teacher_calls", "teacher_family_keys", "teacher_template_keys",
            "verifier_ne_conditions", "w01_started",
        }, where="LanguageCourseManifest")
        if raw["artifact_kind"] != "PH2_LANGUAGE_COURSE_FREEZE":
            raise LanguageCourseContractError("course artifact_kind 非法")
        return cls(
            raw["format_version"],
            str(raw["artifact_version"]),
            str(raw["course_status"]),
            str(raw["objective_taxonomy_status"]),
            tuple(str(item) for item in raw["task_keys"]),
            CanonicalJsonObject.from_value(raw["capability_exit_states"]),
            str(raw["stage"]),
            str(raw["substage"]),
            str(raw["source_key"]),
            str(raw["license_id"]),
            str(raw["sample_relative_path"]),
            str(raw["sample_sha256"]),
            raw["seed_count"],
            tuple(str(item) for item in raw["sample_families"]),
            tuple(str(item) for item in raw["split_axes"]),
            tuple(str(item) for item in raw["teacher_family_keys"]),
            tuple(str(item) for item in raw["evaluator_family_keys"]),
            tuple(str(item) for item in raw["teacher_template_keys"]),
            tuple(str(item) for item in raw["evaluator_template_keys"]),
            str(raw["payload_kind"]),
            tuple(str(item) for item in raw["payload_contract_keys"]),
            str(raw["objective_taxonomy_version"]),
            tuple(LearningObjectiveSpec.from_dict(item)
                  for item in raw["objectives"]),
            tuple(str(item) for item in raw["evaluator_dimensions"]),
            tuple(str(item) for item in raw["verifier_ne_conditions"]),
            tuple(str(item) for item in raw["retention_protocols"]),
            str(raw["pack_manifest_relative_path"]),
            str(raw["pack_manifest_sha256"]),
            raw["pack_record_count"],
            tuple(str(item) for item in raw["pack_splits"]),
            CanonicalJsonObject.from_value(raw["invariants"]),
            raw["d03_published"],
            raw["w01_started"],
            raw["formal_training_runs"],
            raw["teacher_calls"],
            raw["learning_state_writes"],
            raw["mastered_claims"],
            raw["readiness_claims"],
        )


def write_language_course_manifest(
        manifest: LanguageCourseManifest,
        path: str | Path) -> Path:
    """独占或幂等发布课程冻结 manifest，禁止原地改版。"""
    if not isinstance(manifest, LanguageCourseManifest):
        raise LanguageCourseContractError("course manifest 类型错误")
    target = Path(path)
    payload = manifest.canonical_bytes()
    if target.exists():
        if not target.is_file() or target.read_bytes() != payload:
            raise LanguageCourseContractError("course manifest 已存在且内容不同")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(payload)
    except OSError as error:
        raise LanguageCourseContractError("course manifest 无法发布") from error
    return target


def read_language_course_manifest(path: str | Path) -> LanguageCourseManifest:
    """严格回读规范课程冻结 manifest。"""
    try:
        payload = Path(path).read_bytes()
        if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
            raise LanguageCourseContractError("course manifest 换行非法")
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
        assert isinstance(value, dict)
        manifest = LanguageCourseManifest.from_dict(value)
    except LanguageCourseContractError:
        raise
    except Exception as error:
        raise LanguageCourseContractError("course manifest 损坏") from error
    if manifest.canonical_bytes() != payload:
        raise LanguageCourseContractError("course manifest 非规范字节")
    return manifest


__all__ = [
    "COURSE_CAPABILITY_KEYS",
    "COURSE_INVARIANTS",
    "COURSE_RETENTION_PROTOCOLS",
    "COURSE_SPLIT_AXES",
    "COURSE_TASK_KEYS",
    "COURSE_VERIFIER_NE_CONDITIONS",
    "LANGUAGE_OBJECTIVE_KEYS",
    "LanguageCourseContractError",
    "LanguageCourseManifest",
    "LearningObjectiveSpec",
    "TEXT_FIDELITY_EVALUATOR_DIMENSIONS",
    "TEXT_FIDELITY_PAYLOAD_KEYS",
    "read_language_course_manifest",
    "write_language_course_manifest",
]
