"""LC-15 最终分型学习目标、能力课程绑定与消融预注册。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from pure_integer_ai.experiments.ph2_capability_course_contract import (
    CapabilityCourseManifest,
    read_capability_course_manifest,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    canonical_json_line,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_dataset_io import read_artifact_manifest
from pure_integer_ai.experiments.ph2_language_course_contract import (
    LANGUAGE_OBJECTIVE_KEYS,
    LanguageCourseManifest,
    read_language_course_manifest,
)
from pure_integer_ai.experiments.ph2_language_coverage_contract import (
    CAPABILITY_KEYS,
    SAMPLE_FAMILIES,
)


FINAL_OBJECTIVE_MANIFEST_PATH = Path(
    "data/ph2/manifests/lc15_final_learning_objectives_v1.json")
ARTIFACT_VERSION = "LC-15-final-learning-objectives-v1"
TASK_KEYS = ("LC-15-FINAL",)
CORE_CAPABILITY_KEYS = (
    "ATTRIBUTION_QUOTATION_PERSPECTIVE",
    "COMPARISON_QUANTITY_MEASURE",
    "DISCOURSE_INFORMATION_STRUCTURE",
    "EVENT_TIME_ASPECT",
    "MORPHOLOGY_WORD_FORM",
    "MULTIWORD_CONSTRUCTION",
    "OPEN_SET_CONTINUAL_LEARNING",
    "PRAGMATIC_CLARIFICATION_REPAIR",
    "RAW_TEXT_NOISE",
    "RECURSIVE_PARSE",
    "REFERENCE_DISCOURSE_REVISION",
    "SOURCE_UNCERTAINTY_REALITY",
)
BASELINE_ABLATION_KEYS = (
    "BOOT_ONLY",
    "FREQUENCY_ONLY",
    "NO_STRUCTURE",
    "SHUFFLED_COUNT",
)
VERIFIER_DIMENSIONS = (
    "BASELINE_ABLATION_PRE_REGISTRATION",
    "CANDIDATE_LIFECYCLE_BINDING",
    "CAPABILITY_OBJECTIVE_BINDING",
    "COURSE_MANIFEST_HASH_BINDING",
    "EVIDENCE_OWNER_ISOLATION",
    "SAMPLE_FAMILY_COVERAGE",
    "ZERO_RUNTIME_PASS_AUTHORITY",
)
VERIFIER_NE_CONDITIONS = (
    "CANDIDATE_ELIMINATION_NOT_EXECUTED",
    "CAPABILITY_LEARNED_REQUESTED",
    "NO_EVALUATOR_LABEL",
    "RUNTIME_ABLATION_NOT_EXECUTED",
    "RUNTIME_GENERALIZATION_REQUESTED",
)
CANDIDATE_LIFECYCLE_OUTCOMES = (
    "ARCHIVED",
    "CONSUMER_EXIT",
    "REFUTED",
    "SUPERSEDED",
)
EXECUTION_STATE = {
    "ablation_results_observed": 0,
    "candidate_eliminations_executed": 0,
    "companion_writes": 0,
    "core_learning_writes": 0,
    "d03_published": 0,
    "formal_training_runs": 0,
    "mastered_claims": 0,
    "memory_learning_writes": 0,
    "readiness_claims": 0,
    "teacher_calls": 0,
    "use_learning_writes": 0,
    "w01_started": 0,
}
INVARIANTS = {
    "compiler_or_record_count_is_not_elimination_evidence": 1,
    "course_frozen_is_not_capability_learned": 1,
    "count_increase_is_not_candidate_elimination": 1,
    "evaluator_signal_can_train": 0,
    "external_signal_allowed_in_pure_objective": 0,
    "incorrect_candidate_must_degrade_or_exit_at_runtime": 1,
    "runtime_pass_authority": 0,
    "student_reads_observation_only": 1,
    "teacher_call_forbidden": 1,
}

_OBJECTIVE_SIGNALS = {
    "CONTROLLED_PERTURBATION": "PERTURBATION_DISCRIMINATION_ERROR",
    "CROSS_CONTEXT_CONSISTENCY": "CROSS_CONTEXT_CONTRADICTION",
    "GENERATION_ADOPTION": "GENERATION_POSTCHECK_REJECTION",
    "GENERATION_FAILURE": "GENERATION_CONSUMPTION_FAILURE",
    "INTEGER_DESCRIPTION_LENGTH": "INTEGER_COMPRESSION_DELTA",
    "MASKED_SPAN": "MASKED_SPAN_RECONSTRUCTION_ERROR",
    "MASKED_TOKEN": "MASKED_TOKEN_RECONSTRUCTION_ERROR",
    "NEXT_DISCOURSE_UNIT": "NEXT_DISCOURSE_UNIT_PREDICTION_ERROR",
    "NEXT_SPAN": "NEXT_SPAN_PREDICTION_ERROR",
    "NEXT_TOKEN": "NEXT_TOKEN_PREDICTION_ERROR",
    "ORDER_RECOVERY": "ORDER_RECOVERY_ERROR",
}
_COURSE_SOURCES = (
    (
        "data/ph2/manifests/lc01_lc15_initial_course_v1.json",
        "tests/test_d02_lc01_text_fidelity_course.py",
    ),
    (
        "data/ph2/manifests/lc02_morphology_course_v1.json",
        "tests/test_d02_lc02_morphology_course.py",
    ),
    (
        "data/ph2/manifests/lc03_construction_course_v1.json",
        "tests/test_d02_lc03_construction_course.py",
    ),
    (
        "data/ph2/manifests/lc04_recursive_parse_course_v1.json",
        "tests/test_d02_lc04_recursive_parse_course.py",
    ),
    (
        "data/ph2/manifests/lc05_event_time_aspect_course_v1.json",
        "tests/test_d02_lc05_event_time_aspect_course.py",
    ),
    (
        "data/ph2/manifests/lc06_comparison_quantity_course_v1.json",
        "tests/test_d02_lc06_comparison_quantity_course.py",
    ),
    (
        "data/ph2/manifests/lc07_discourse_information_course_v1.json",
        "tests/test_d02_lc07_discourse_information_course.py",
    ),
    (
        "data/ph2/manifests/lc08_open_set_clarification_course_v1.json",
        "tests/test_d02_lc08_open_set_clarification_course.py",
    ),
    (
        "data/ph2/manifests/lc14_attribution_quotation_course_v1.json",
        "tests/test_d02_lc14_attribution_quotation_course.py",
    ),
)
_INITIAL_BASELINES = (
    "IRREVERSIBLE_LOSS_AUTO_ADOPTION_REJECT",
    "PRESELECTED_LATTICE_REJECT",
)
_INITIAL_ABLATIONS = (
    "DROP_NORMALIZATION_RECEIPT",
    "DROP_RAW_OBSERVATION",
    "REMOVE_CANDIDATE_LATTICE",
)


class LearningObjectiveCoverageError(RuntimeError):
    """LC-15 最终目标账、来源 hash 或 owner 隔离不完整。"""


def _exact(value: Any, keys: set[str], *, where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise LearningObjectiveCoverageError(f"{where} 字段集合非法")
    return value


def _text(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise LearningObjectiveCoverageError(f"{where} 必须是非空规范文本")
    return value


def _positive(value: Any, *, where: str) -> int:
    if type(value) is not int or value <= 0:
        raise LearningObjectiveCoverageError(f"{where} 必须是正严格整数")
    return value


def _flag(value: Any, *, where: str) -> int:
    if type(value) is not int or value not in {0, 1}:
        raise LearningObjectiveCoverageError(f"{where} 必须是 0/1")
    return value


def _sha256(value: Any, *, where: str) -> str:
    text = _text(value, where=where)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise LearningObjectiveCoverageError(f"{where} 必须是 SHA-256")
    return text


def _relative(value: Any, *, where: str) -> str:
    text = _text(value, where=where)
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or "\\" in text:
        raise LearningObjectiveCoverageError(f"{where} 必须是安全 POSIX 相对路径")
    return text


def _tuple(
        value: Any,
        *,
        where: str,
        allow_empty: bool = False,
        ) -> tuple[str, ...]:
    if not isinstance(value, tuple) or (not value and not allow_empty):
        raise LearningObjectiveCoverageError(f"{where} 必须是文本 tuple")
    if any(not isinstance(item, str) or not item or item.strip() != item
           for item in value):
        raise LearningObjectiveCoverageError(f"{where} 含非法文本")
    if value != tuple(sorted(set(value))):
        raise LearningObjectiveCoverageError(f"{where} 必须排序去重")
    return value


def _canonical_state(value: Any, expected: dict[str, Any], *, where: str) -> None:
    if not isinstance(value, CanonicalJsonObject) or value.to_value() != expected:
        raise LearningObjectiveCoverageError(f"{where} 不完整")


@dataclass(frozen=True)
class FinalLearningObjectiveSpec:
    """一个目标的独立淘汰信号和 Evidence owner。"""

    objective_key: str
    elimination_signal: str
    candidate_lifecycle_outcomes: tuple[str, ...]
    training_evidence_owner: str
    evaluation_evidence_owner: str
    external_signal_allowed: int
    evaluator_signal_can_train: int
    runtime_pass_authority: int

    def __post_init__(self) -> None:
        if self.objective_key not in LANGUAGE_OBJECTIVE_KEYS:
            raise LearningObjectiveCoverageError("objective_key 未登记")
        _text(self.elimination_signal, where="elimination_signal")
        if self.candidate_lifecycle_outcomes != CANDIDATE_LIFECYCLE_OUTCOMES:
            raise LearningObjectiveCoverageError("候选生命周期淘汰出口不完整")
        if self.training_evidence_owner != "TEACHER_EVIDENCE":
            raise LearningObjectiveCoverageError("训练信号 owner 非法")
        if self.evaluation_evidence_owner != "EVALUATOR_LABEL":
            raise LearningObjectiveCoverageError("评测信号 owner 非法")
        if _flag(self.external_signal_allowed, where="external signal") != 0:
            raise LearningObjectiveCoverageError("EXTERNAL 不得混入纯目标")
        if _flag(self.evaluator_signal_can_train, where="evaluator train") != 0:
            raise LearningObjectiveCoverageError("evaluator 不得训练")
        if _flag(self.runtime_pass_authority, where="runtime pass") != 0:
            raise LearningObjectiveCoverageError("目标账不得签发 runtime PASS")

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_lifecycle_outcomes": list(self.candidate_lifecycle_outcomes),
            "elimination_signal": self.elimination_signal,
            "evaluation_evidence_owner": self.evaluation_evidence_owner,
            "evaluator_signal_can_train": self.evaluator_signal_can_train,
            "external_signal_allowed": self.external_signal_allowed,
            "objective_key": self.objective_key,
            "runtime_pass_authority": self.runtime_pass_authority,
            "training_evidence_owner": self.training_evidence_owner,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FinalLearningObjectiveSpec":
        raw = _exact(value, {
            "candidate_lifecycle_outcomes", "elimination_signal",
            "evaluation_evidence_owner", "evaluator_signal_can_train",
            "external_signal_allowed", "objective_key",
            "runtime_pass_authority", "training_evidence_owner",
        }, where="FinalLearningObjectiveSpec")
        return cls(
            str(raw["objective_key"]), str(raw["elimination_signal"]),
            tuple(str(item) for item in raw["candidate_lifecycle_outcomes"]),
            str(raw["training_evidence_owner"]),
            str(raw["evaluation_evidence_owner"]),
            raw["external_signal_allowed"], raw["evaluator_signal_can_train"],
            raw["runtime_pass_authority"])


@dataclass(frozen=True)
class CapabilityObjectiveBinding:
    """一个能力对既有课程、目标、evaluator 和淘汰边界的绑定。"""

    capability_key: str
    task_key: str
    course_manifest_relative_path: str
    course_manifest_sha256: str
    pack_manifest_relative_path: str
    pack_manifest_sha256: str
    sample_family_states: CanonicalJsonObject
    objective_keys: tuple[str, ...]
    elimination_signals: tuple[str, ...]
    evaluator_dimensions: tuple[str, ...]
    baseline_kinds: tuple[str, ...]
    course_ablation_keys: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    training_evidence_owner: str
    evaluation_evidence_owner: str
    external_signal_allowed: int
    evaluator_signal_can_train: int
    runtime_elimination_status: str
    runtime_pass_authority: int

    def __post_init__(self) -> None:
        if self.capability_key not in CORE_CAPABILITY_KEYS:
            raise LearningObjectiveCoverageError("核心能力 binding 未登记")
        _text(self.task_key, where="task_key")
        _relative(
            self.course_manifest_relative_path,
            where="course_manifest_relative_path")
        _sha256(self.course_manifest_sha256, where="course manifest sha256")
        _relative(
            self.pack_manifest_relative_path,
            where="pack_manifest_relative_path")
        _sha256(self.pack_manifest_sha256, where="pack manifest sha256")
        _canonical_state(
            self.sample_family_states,
            {key: "FROZEN" for key in SAMPLE_FAMILIES},
            where="七类 sample family")
        objectives = _tuple(self.objective_keys, where="objective_keys")
        if any(item not in LANGUAGE_OBJECTIVE_KEYS for item in objectives):
            raise LearningObjectiveCoverageError("能力 objective 未登记")
        signals = _tuple(self.elimination_signals, where="elimination_signals")
        expected_signals = tuple(sorted(_OBJECTIVE_SIGNALS[key] for key in objectives))
        if signals != expected_signals:
            raise LearningObjectiveCoverageError("objective 与淘汰信号未一一绑定")
        _tuple(self.evaluator_dimensions, where="evaluator_dimensions")
        _tuple(self.baseline_kinds, where="baseline_kinds")
        _tuple(self.course_ablation_keys, where="course_ablation_keys")
        _tuple(self.evidence_refs, where="evidence_refs")
        if self.training_evidence_owner != "TEACHER_EVIDENCE":
            raise LearningObjectiveCoverageError("能力训练 owner 非法")
        if self.evaluation_evidence_owner != "EVALUATOR_LABEL":
            raise LearningObjectiveCoverageError("能力 evaluator owner 非法")
        if _flag(self.external_signal_allowed, where="external signal") != 0:
            raise LearningObjectiveCoverageError("能力 EXTERNAL 信号非法")
        if _flag(self.evaluator_signal_can_train, where="evaluator train") != 0:
            raise LearningObjectiveCoverageError("能力 evaluator 不得训练")
        if self.runtime_elimination_status != "NOT_STARTED":
            raise LearningObjectiveCoverageError("候选淘汰不得冒充已执行")
        if _flag(self.runtime_pass_authority, where="runtime pass") != 0:
            raise LearningObjectiveCoverageError("能力 binding 不得签发 PASS")

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_kinds": list(self.baseline_kinds),
            "capability_key": self.capability_key,
            "course_ablation_keys": list(self.course_ablation_keys),
            "course_manifest_relative_path": self.course_manifest_relative_path,
            "course_manifest_sha256": self.course_manifest_sha256,
            "elimination_signals": list(self.elimination_signals),
            "evaluation_evidence_owner": self.evaluation_evidence_owner,
            "evaluator_dimensions": list(self.evaluator_dimensions),
            "evaluator_signal_can_train": self.evaluator_signal_can_train,
            "evidence_refs": list(self.evidence_refs),
            "external_signal_allowed": self.external_signal_allowed,
            "objective_keys": list(self.objective_keys),
            "pack_manifest_relative_path": self.pack_manifest_relative_path,
            "pack_manifest_sha256": self.pack_manifest_sha256,
            "runtime_elimination_status": self.runtime_elimination_status,
            "runtime_pass_authority": self.runtime_pass_authority,
            "sample_family_states": self.sample_family_states.to_value(),
            "task_key": self.task_key,
            "training_evidence_owner": self.training_evidence_owner,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CapabilityObjectiveBinding":
        raw = _exact(value, {
            "baseline_kinds", "capability_key", "course_ablation_keys",
            "course_manifest_relative_path", "course_manifest_sha256",
            "elimination_signals", "evaluation_evidence_owner",
            "evaluator_dimensions", "evaluator_signal_can_train",
            "evidence_refs", "external_signal_allowed", "objective_keys",
            "pack_manifest_relative_path", "pack_manifest_sha256",
            "runtime_elimination_status", "runtime_pass_authority",
            "sample_family_states", "task_key", "training_evidence_owner",
        }, where="CapabilityObjectiveBinding")
        return cls(
            str(raw["capability_key"]), str(raw["task_key"]),
            str(raw["course_manifest_relative_path"]),
            str(raw["course_manifest_sha256"]),
            str(raw["pack_manifest_relative_path"]),
            str(raw["pack_manifest_sha256"]),
            CanonicalJsonObject.from_value(raw["sample_family_states"]),
            tuple(str(item) for item in raw["objective_keys"]),
            tuple(str(item) for item in raw["elimination_signals"]),
            tuple(str(item) for item in raw["evaluator_dimensions"]),
            tuple(str(item) for item in raw["baseline_kinds"]),
            tuple(str(item) for item in raw["course_ablation_keys"]),
            tuple(str(item) for item in raw["evidence_refs"]),
            str(raw["training_evidence_owner"]),
            str(raw["evaluation_evidence_owner"]),
            raw["external_signal_allowed"], raw["evaluator_signal_can_train"],
            str(raw["runtime_elimination_status"]),
            raw["runtime_pass_authority"])


@dataclass(frozen=True)
class BaselineAblationPreRegistration:
    """只频率/boot/无结构/打乱计数基线的结果盲预注册。"""

    ablation_key: str
    expected_effect: str
    runtime_status: str
    results_observed: int
    runtime_pass_authority: int

    def __post_init__(self) -> None:
        if self.ablation_key not in BASELINE_ABLATION_KEYS:
            raise LearningObjectiveCoverageError("baseline ablation 未登记")
        if self.expected_effect != "DEGRADE_AT_LEAST_ONE_PRE_REGISTERED_DIMENSION":
            raise LearningObjectiveCoverageError("baseline ablation 退化判据非法")
        if self.runtime_status != "NOT_STARTED":
            raise LearningObjectiveCoverageError("baseline ablation 不得冒充已执行")
        if _flag(self.results_observed, where="ablation result") != 0:
            raise LearningObjectiveCoverageError("预注册不得先看结果")
        if _flag(self.runtime_pass_authority, where="ablation pass") != 0:
            raise LearningObjectiveCoverageError("ablation 不得签发 PASS")

    def to_dict(self) -> dict[str, Any]:
        return {
            "ablation_key": self.ablation_key,
            "expected_effect": self.expected_effect,
            "results_observed": self.results_observed,
            "runtime_pass_authority": self.runtime_pass_authority,
            "runtime_status": self.runtime_status,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BaselineAblationPreRegistration":
        raw = _exact(value, {
            "ablation_key", "expected_effect", "results_observed",
            "runtime_pass_authority", "runtime_status",
        }, where="BaselineAblationPreRegistration")
        return cls(
            str(raw["ablation_key"]), str(raw["expected_effect"]),
            str(raw["runtime_status"]), raw["results_observed"],
            raw["runtime_pass_authority"])


@dataclass(frozen=True)
class FinalLearningObjectiveManifest:
    """LC-15 最终目标账；冻结课程，不声明候选已真实淘汰。"""

    format_version: int
    artifact_version: str
    course_status: str
    objective_taxonomy_status: str
    runtime_status: str
    task_keys: tuple[str, ...]
    capability_exit_states: CanonicalJsonObject
    course_source_count: int
    capability_bindings: tuple[CapabilityObjectiveBinding, ...]
    objectives: tuple[FinalLearningObjectiveSpec, ...]
    baseline_ablations: tuple[BaselineAblationPreRegistration, ...]
    verifier_dimensions: tuple[str, ...]
    verifier_ne_conditions: tuple[str, ...]
    invariants: CanonicalJsonObject
    execution_state: CanonicalJsonObject

    def __post_init__(self) -> None:
        if self.format_version != 1:
            raise LearningObjectiveCoverageError("format_version 非法")
        if self.artifact_version != ARTIFACT_VERSION:
            raise LearningObjectiveCoverageError("artifact_version 非法")
        if self.course_status != "COURSE_FROZEN":
            raise LearningObjectiveCoverageError("LC-15 course 未冻结")
        if self.objective_taxonomy_status != "FINAL_FROZEN":
            raise LearningObjectiveCoverageError("LC-15 taxonomy 未最终冻结")
        if self.runtime_status != "NOT_STARTED":
            raise LearningObjectiveCoverageError("LC-15 runtime 不得冒充启动")
        if self.task_keys != TASK_KEYS:
            raise LearningObjectiveCoverageError("LC-15 task_keys 非法")
        expected_states = {
            key: "COURSE_FROZEN"
            for key in (*CORE_CAPABILITY_KEYS, "TYPED_LEARNING_OBJECTIVES")
        }
        _canonical_state(
            self.capability_exit_states, expected_states,
            where="LC-15 capability exit states")
        if self.course_source_count != len(_COURSE_SOURCES):
            raise LearningObjectiveCoverageError("上游课程数非法")
        if (not isinstance(self.capability_bindings, tuple)
                or tuple(item.capability_key for item in self.capability_bindings)
                != CORE_CAPABILITY_KEYS):
            raise LearningObjectiveCoverageError("核心能力 binding 未列全或未排序")
        if (not isinstance(self.objectives, tuple)
                or tuple(item.objective_key for item in self.objectives)
                != LANGUAGE_OBJECTIVE_KEYS):
            raise LearningObjectiveCoverageError("十一类目标未列全")
        covered = {
            key for binding in self.capability_bindings
            for key in binding.objective_keys}
        if covered != set(LANGUAGE_OBJECTIVE_KEYS):
            raise LearningObjectiveCoverageError("课程未覆盖完整目标 taxonomy")
        if (not isinstance(self.baseline_ablations, tuple)
                or tuple(item.ablation_key for item in self.baseline_ablations)
                != BASELINE_ABLATION_KEYS):
            raise LearningObjectiveCoverageError("四基线消融未列全")
        if self.verifier_dimensions != VERIFIER_DIMENSIONS:
            raise LearningObjectiveCoverageError("verifier dimensions 漂移")
        if self.verifier_ne_conditions != VERIFIER_NE_CONDITIONS:
            raise LearningObjectiveCoverageError("verifier NE 漂移")
        _canonical_state(self.invariants, INVARIANTS, where="LC-15 invariants")
        _canonical_state(
            self.execution_state, EXECUTION_STATE, where="LC-15 execution state")

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "PH2_LC15_FINAL_LEARNING_OBJECTIVES",
            "artifact_version": self.artifact_version,
            "baseline_ablations": [item.to_dict() for item in self.baseline_ablations],
            "capability_bindings": [item.to_dict() for item in self.capability_bindings],
            "capability_exit_states": self.capability_exit_states.to_value(),
            "course_source_count": self.course_source_count,
            "course_status": self.course_status,
            "execution_state": self.execution_state.to_value(),
            "format_version": self.format_version,
            "invariants": self.invariants.to_value(),
            "objective_taxonomy_status": self.objective_taxonomy_status,
            "objectives": [item.to_dict() for item in self.objectives],
            "runtime_status": self.runtime_status,
            "task_keys": list(self.task_keys),
            "verifier_dimensions": list(self.verifier_dimensions),
            "verifier_ne_conditions": list(self.verifier_ne_conditions),
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_line(self.to_dict())

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FinalLearningObjectiveManifest":
        raw = _exact(value, {
            "artifact_kind", "artifact_version", "baseline_ablations",
            "capability_bindings", "capability_exit_states",
            "course_source_count", "course_status", "execution_state",
            "format_version", "invariants", "objective_taxonomy_status",
            "objectives", "runtime_status", "task_keys",
            "verifier_dimensions", "verifier_ne_conditions",
        }, where="FinalLearningObjectiveManifest")
        if raw["artifact_kind"] != "PH2_LC15_FINAL_LEARNING_OBJECTIVES":
            raise LearningObjectiveCoverageError("artifact_kind 非法")
        return cls(
            raw["format_version"], str(raw["artifact_version"]),
            str(raw["course_status"]), str(raw["objective_taxonomy_status"]),
            str(raw["runtime_status"]),
            tuple(str(item) for item in raw["task_keys"]),
            CanonicalJsonObject.from_value(raw["capability_exit_states"]),
            raw["course_source_count"],
            tuple(CapabilityObjectiveBinding.from_dict(item)
                  for item in raw["capability_bindings"]),
            tuple(FinalLearningObjectiveSpec.from_dict(item)
                  for item in raw["objectives"]),
            tuple(BaselineAblationPreRegistration.from_dict(item)
                  for item in raw["baseline_ablations"]),
            tuple(str(item) for item in raw["verifier_dimensions"]),
            tuple(str(item) for item in raw["verifier_ne_conditions"]),
            CanonicalJsonObject.from_value(raw["invariants"]),
            CanonicalJsonObject.from_value(raw["execution_state"]))


def _manifest_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _binding(
        capability_key: str,
        task_key: str,
        manifest_path: str,
        manifest_sha256: str,
        pack_path: str,
        pack_sha256: str,
        objective_keys: tuple[str, ...],
        evaluator_dimensions: tuple[str, ...],
        baseline_kinds: tuple[str, ...],
        ablation_keys: tuple[str, ...],
        test_path: str,
        ) -> CapabilityObjectiveBinding:
    return CapabilityObjectiveBinding(
        capability_key, task_key, manifest_path, manifest_sha256,
        pack_path, pack_sha256,
        CanonicalJsonObject.from_value({
            key: "FROZEN" for key in SAMPLE_FAMILIES}),
        tuple(sorted(objective_keys)),
        tuple(sorted(_OBJECTIVE_SIGNALS[key] for key in objective_keys)),
        tuple(sorted(evaluator_dimensions)), tuple(sorted(baseline_kinds)),
        tuple(sorted(ablation_keys)), tuple(sorted((manifest_path, test_path))),
        "TEACHER_EVIDENCE", "EVALUATOR_LABEL", 0, 0, "NOT_STARTED", 0)


def build_final_learning_objective_manifest(
        repository_root: str | Path,
        ) -> FinalLearningObjectiveManifest:
    """逐 hash 编译九个既有课程为 LC-15 最终目标账。"""
    root = Path(repository_root).resolve()
    bindings: list[CapabilityObjectiveBinding] = []
    for manifest_relative_path, test_path in _COURSE_SOURCES:
        manifest_path = root / Path(*manifest_relative_path.split("/"))
        if manifest_relative_path.endswith("lc01_lc15_initial_course_v1.json"):
            initial = read_language_course_manifest(manifest_path)
            bindings.append(_binding(
                "RAW_TEXT_NOISE", "LC-01", manifest_relative_path,
                _manifest_sha256(manifest_path),
                initial.pack_manifest_relative_path,
                initial.pack_manifest_sha256,
                tuple(item.objective_key for item in initial.objectives),
                initial.evaluator_dimensions, _INITIAL_BASELINES,
                _INITIAL_ABLATIONS, test_path))
            continue
        course = read_capability_course_manifest(manifest_path)
        for capability_key in course.capability_keys:
            if capability_key not in CORE_CAPABILITY_KEYS:
                continue
            bindings.append(_binding(
                capability_key, course.task_keys[0], manifest_relative_path,
                _manifest_sha256(manifest_path),
                course.pack_manifest_relative_path,
                course.pack_manifest_sha256,
                course.objective_keys, course.evaluator_dimensions,
                course.baseline_kinds, course.ablation_keys, test_path))
    bindings.sort(key=lambda item: item.capability_key)
    objectives = tuple(FinalLearningObjectiveSpec(
        key, _OBJECTIVE_SIGNALS[key], CANDIDATE_LIFECYCLE_OUTCOMES,
        "TEACHER_EVIDENCE", "EVALUATOR_LABEL", 0, 0, 0)
        for key in LANGUAGE_OBJECTIVE_KEYS)
    ablations = tuple(BaselineAblationPreRegistration(
        key, "DEGRADE_AT_LEAST_ONE_PRE_REGISTERED_DIMENSION",
        "NOT_STARTED", 0, 0) for key in BASELINE_ABLATION_KEYS)
    return FinalLearningObjectiveManifest(
        1, ARTIFACT_VERSION, "COURSE_FROZEN", "FINAL_FROZEN",
        "NOT_STARTED", TASK_KEYS,
        CanonicalJsonObject.from_value({
            key: "COURSE_FROZEN"
            for key in (*CORE_CAPABILITY_KEYS, "TYPED_LEARNING_OBJECTIVES")
        }),
        len(_COURSE_SOURCES), tuple(bindings), objectives, ablations,
        VERIFIER_DIMENSIONS, VERIFIER_NE_CONDITIONS,
        CanonicalJsonObject.from_value(INVARIANTS),
        CanonicalJsonObject.from_value(EXECUTION_STATE))


def verify_final_learning_objective_sources(
        manifest: FinalLearningObjectiveManifest,
        *,
        repository_root: str | Path,
        workspace_root: str | Path,
        ) -> None:
    """逐字节回验上游 course/pack，不把账目存在当淘汰执行。"""
    if not isinstance(manifest, FinalLearningObjectiveManifest):
        raise LearningObjectiveCoverageError("LC-15 manifest 类型非法")
    root = Path(repository_root).resolve()
    workspace = Path(workspace_root).resolve()
    for binding in manifest.capability_bindings:
        course_path = root / Path(*binding.course_manifest_relative_path.split("/"))
        try:
            if _manifest_sha256(course_path) != binding.course_manifest_sha256:
                raise LearningObjectiveCoverageError("上游 course manifest hash 漂移")
            if binding.course_manifest_relative_path.endswith(
                    "lc01_lc15_initial_course_v1.json"):
                course: LanguageCourseManifest | CapabilityCourseManifest = (
                    read_language_course_manifest(course_path))
                capabilities = set(course.capability_exit_states.to_value())
                objectives = {item.objective_key for item in course.objectives}
                baselines = set(_INITIAL_BASELINES)
                ablations = set(_INITIAL_ABLATIONS)
            else:
                course = read_capability_course_manifest(course_path)
                capabilities = set(course.capability_keys)
                objectives = set(course.objective_keys)
                baselines = set(course.baseline_kinds)
                ablations = set(course.ablation_keys)
            if binding.capability_key not in capabilities:
                raise LearningObjectiveCoverageError("上游 course capability 漂移")
            if set(binding.objective_keys) != objectives:
                raise LearningObjectiveCoverageError("上游 objective binding 漂移")
            if set(binding.baseline_kinds) != baselines:
                raise LearningObjectiveCoverageError("上游 baseline binding 漂移")
            if set(binding.course_ablation_keys) != ablations:
                raise LearningObjectiveCoverageError("上游 ablation binding 漂移")
            if (binding.pack_manifest_relative_path
                    != course.pack_manifest_relative_path
                    or binding.pack_manifest_sha256
                    != course.pack_manifest_sha256):
                raise LearningObjectiveCoverageError("上游 pack binding 漂移")
            pack_path = workspace / Path(
                *binding.pack_manifest_relative_path.split("/"))
            pack = read_artifact_manifest(pack_path)
            if pack.sha256() != binding.pack_manifest_sha256:
                raise LearningObjectiveCoverageError("上游 pack manifest hash 漂移")
        except LearningObjectiveCoverageError:
            raise
        except Exception as error:
            raise LearningObjectiveCoverageError("LC-15 上游证据无法回读") from error


def write_final_learning_objective_manifest(
        manifest: FinalLearningObjectiveManifest,
        path: str | Path,
        ) -> Path:
    """独占或幂等写 LC-15 最终 manifest，禁止同版本覆盖。"""
    if not isinstance(manifest, FinalLearningObjectiveManifest):
        raise LearningObjectiveCoverageError("LC-15 manifest 类型非法")
    target = Path(path)
    payload = manifest.canonical_bytes()
    if target.exists():
        if not target.is_file() or target.read_bytes() != payload:
            raise LearningObjectiveCoverageError("LC-15 manifest 已存在且内容不同")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(payload)
    except OSError as error:
        raise LearningObjectiveCoverageError("LC-15 manifest 无法发布") from error
    return target


def read_final_learning_objective_manifest(
        path: str | Path,
        ) -> FinalLearningObjectiveManifest:
    """严格回读规范 LC-15 最终 manifest。"""
    try:
        payload = Path(path).read_bytes()
        if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
            raise LearningObjectiveCoverageError("LC-15 manifest 换行非法")
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
        assert isinstance(value, dict)
        manifest = FinalLearningObjectiveManifest.from_dict(value)
    except LearningObjectiveCoverageError:
        raise
    except Exception as error:
        raise LearningObjectiveCoverageError("LC-15 manifest 损坏") from error
    if manifest.canonical_bytes() != payload:
        raise LearningObjectiveCoverageError("LC-15 manifest 非规范字节")
    return manifest


__all__ = [
    "ARTIFACT_VERSION",
    "BASELINE_ABLATION_KEYS",
    "CORE_CAPABILITY_KEYS",
    "FINAL_OBJECTIVE_MANIFEST_PATH",
    "BaselineAblationPreRegistration",
    "CapabilityObjectiveBinding",
    "FinalLearningObjectiveManifest",
    "FinalLearningObjectiveSpec",
    "LearningObjectiveCoverageError",
    "build_final_learning_objective_manifest",
    "read_final_learning_objective_manifest",
    "verify_final_learning_objective_sources",
    "write_final_learning_objective_manifest",
]
