"""LC-10 retention、回滚与范围收缩的纯合同。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    canonical_json_bytes,
    parse_canonical_json_bytes,
)


FORMAT_VERSION = 1
ARTIFACT_STATUS = "COURSE_FROZEN"
RUNTIME_STATUS = "NOT_STARTED"
RETENTION_PHASE_KEYS = (
    "BASELINE_A",
    "APPLY_B",
    "REVERIFY_A_AFTER_B",
    "DUMP_AFTER_B",
    "RESUME_AFTER_B",
    "WITHDRAW_B",
    "ROLLBACK_B",
    "REVERIFY_A_AFTER_ROLLBACK",
    "CONTRACT_SCOPE",
    "REVERIFY_CONTRACTED_SCOPE",
)
OUTCOME_CLASSES = (
    "FORGETTING",
    "INTERFERENCE",
    "NO_CHANGE",
    "POSITIVE_TRANSFER",
    "SCOPE_CONTRACTION",
)
CAPABILITY_VERDICTS = ("FAIL", "NE", "PASS")
FIXTURE_VERDICTS = ("PASS", "REJECT")
FIXTURE_KEYS = (
    "LC10_FORGETTING_REJECT_V1",
    "LC10_NO_CHANGE_ACCEPT_V1",
    "LC10_SCOPE_CONTRACTION_ACCEPT_V1",
)
RUNTIME_BINDING_STATES = (
    "AVAILABLE_NOT_EXECUTED",
    "FUTURE_REQUIRED",
    "PROTOCOL_ONLY_NE",
)
VERIFIER_DIMENSIONS = (
    "A_B_REVERIFY_A",
    "CORE_BIT_IDENTITY",
    "DUMP_RESUME_BIT_IDENTITY",
    "NO_MEAN_MASKING",
    "ROLLBACK_LOCALITY",
    "SCOPE_CONTRACTION_REPLAYABLE",
    "SOURCE_WITHDRAWAL_LOCALITY",
    "UNAFFECTED_CAPABILITY_BIT_IDENTITY",
    "ZERO_HOST_LEARNING_WRITE",
)
VERIFIER_NE_CONDITIONS = (
    "LC13_DIRECTIONAL_CONSUMER_MAP_NOT_FROZEN",
    "RETENTION_EPISODE_NOT_EXECUTED",
    "SOURCE_DEPENDENCY_MAP_MISSING",
    "V06_CLONE_NOT_EXECUTED",
    "W01_STATE_PROTOCOL_NOT_STARTED",
    "W09_NOT_STARTED",
)
FUTURE_PREREQUISITE_STATES = {
    "LC13_DIRECTIONAL_CONSUMER_MAP": "NOT_STARTED",
    "V06_RETENTION_CLONE_EXECUTION": "NOT_STARTED",
    "W01_STATE_PROTOCOL": "NOT_STARTED",
    "W09_RETENTION_GATE": "NOT_STARTED",
}
EXECUTION_STATE = {
    "companion_writes": 0,
    "core_learning_writes": 0,
    "d03_published": 0,
    "formal_training_runs": 0,
    "mastered_claims": 0,
    "memory_learning_writes": 0,
    "readiness_claims": 0,
    "retention_runtime_runs": 0,
    "teacher_calls": 0,
    "use_learning_writes": 0,
    "v06_clone_runs": 0,
    "w01_started": 0,
}


class RetentionRollbackContractError(RuntimeError):
    """LC-10 顺序、fixture、证据或零运行边界不满足冻结合同。"""


def _text(value: Any, *, where: str) -> str:
    """要求字段是无首尾空白的非空文本。"""
    if (not isinstance(value, str) or not value
            or value.strip() != value):
        raise RetentionRollbackContractError(f"{where} 必须是规范文本")
    return value


def _flag(value: Any, *, where: str) -> int:
    """要求协议开关为严格 0/1。"""
    if type(value) is not int or value not in {0, 1}:
        raise RetentionRollbackContractError(f"{where} 必须是 0/1")
    return value


def _strict_int(value: Any, *, where: str) -> int:
    """要求值为严格整数，不接受 bool 或 float。"""
    if type(value) is not int:
        raise RetentionRollbackContractError(f"{where} 必须是严格整数")
    return value


def _nonnegative(value: Any, *, where: str) -> int:
    """要求计数为非负严格整数。"""
    result = _strict_int(value, where=where)
    if result < 0:
        raise RetentionRollbackContractError(f"{where} 不得为负")
    return result


def _sha256(value: Any, *, where: str) -> str:
    """要求字段是小写 SHA-256。"""
    text = _text(value, where=where)
    if len(text) != 64 or any(item not in "0123456789abcdef" for item in text):
        raise RetentionRollbackContractError(f"{where} 必须是 SHA-256")
    return text


def _optional_sha256(value: Any, *, where: str) -> str:
    """要求字段是 SHA-256 或协议空值 NONE。"""
    if value == "NONE":
        return value
    return _sha256(value, where=where)


def _relative_path(value: Any, *, where: str) -> str:
    """要求证据引用是安全、可迁移的 POSIX 相对路径。"""
    text = _text(value, where=where)
    path = PurePosixPath(text)
    if (path.is_absolute() or ".." in path.parts or "\\" in text
            or path.as_posix() != text or ":" in path.parts[0]):
        raise RetentionRollbackContractError(f"{where} 必须是安全相对路径")
    return text


def _strict_text_tuple(
        value: Any,
        *,
        where: str,
        allow_empty: bool = False,
        ) -> tuple[str, ...]:
    """要求字符串元组已排序去重。"""
    if not isinstance(value, tuple):
        raise RetentionRollbackContractError(f"{where} 必须是 tuple")
    if not allow_empty and not value:
        raise RetentionRollbackContractError(f"{where} 不能为空")
    for item in value:
        _text(item, where=where)
    if tuple(sorted(set(value))) != value:
        raise RetentionRollbackContractError(f"{where} 必须排序去重")
    return value


def _ordered_text_tuple(value: Any, *, where: str) -> tuple[str, ...]:
    """要求协议顺序元组非空且无重复，但保留声明顺序。"""
    if not isinstance(value, tuple) or not value:
        raise RetentionRollbackContractError(f"{where} 必须是非空 tuple")
    for item in value:
        _text(item, where=where)
    if len(value) != len(set(value)):
        raise RetentionRollbackContractError(f"{where} 不得重复")
    return value


@dataclass(frozen=True)
class RetentionDimensionResult:
    """一个旧能力维在 A→B→重验 A 中的逐维结果。"""

    dimension_key: str
    before_verdict: str
    after_verdict: str
    quality_delta: int
    in_scope_before: int
    in_scope_after: int
    dependency_on_b: int
    source_withdrawal_affected: int
    classification: str

    def __post_init__(self) -> None:
        _text(self.dimension_key, where="dimension_key")
        if self.before_verdict not in CAPABILITY_VERDICTS:
            raise RetentionRollbackContractError("before_verdict 非法")
        if self.after_verdict not in CAPABILITY_VERDICTS:
            raise RetentionRollbackContractError("after_verdict 非法")
        _strict_int(self.quality_delta, where="quality_delta")
        for name in (
                "in_scope_before", "in_scope_after", "dependency_on_b",
                "source_withdrawal_affected"):
            _flag(getattr(self, name), where=name)
        if self.before_verdict != "PASS" or self.in_scope_before != 1:
            raise RetentionRollbackContractError(
                "LC-10 只重验此前已通过且在 scope 内的旧维")
        if self.classification not in OUTCOME_CLASSES:
            raise RetentionRollbackContractError("classification 未登记")
        expected = {
            "POSITIVE_TRANSFER": (
                self.after_verdict == "PASS" and self.quality_delta > 0
                and self.in_scope_after == 1),
            "NO_CHANGE": (
                self.after_verdict == "PASS" and self.quality_delta == 0
                and self.in_scope_after == 1),
            "INTERFERENCE": (
                self.after_verdict == "PASS" and self.quality_delta < 0
                and self.in_scope_after == 1),
            "FORGETTING": (
                self.after_verdict == "FAIL" and self.in_scope_after == 1),
            "SCOPE_CONTRACTION": (
                self.after_verdict == "NE" and self.quality_delta == 0
                and self.in_scope_after == 0
                and self.source_withdrawal_affected == 1),
        }[self.classification]
        if not expected:
            raise RetentionRollbackContractError(
                "逐维 verdict/delta/scope 与 classification 不一致")
        if (self.classification != "SCOPE_CONTRACTION"
                and self.in_scope_after != 1):
            raise RetentionRollbackContractError("非收缩结果不得移出 scope")

    def to_dict(self) -> dict[str, Any]:
        """投影为规范 JSON 对象。"""
        return {
            "after_verdict": self.after_verdict,
            "before_verdict": self.before_verdict,
            "classification": self.classification,
            "dependency_on_b": self.dependency_on_b,
            "dimension_key": self.dimension_key,
            "in_scope_after": self.in_scope_after,
            "in_scope_before": self.in_scope_before,
            "quality_delta": self.quality_delta,
            "source_withdrawal_affected": self.source_withdrawal_affected,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RetentionDimensionResult":
        """从精确字段对象恢复逐维结果。"""
        expected = {
            "after_verdict", "before_verdict", "classification",
            "dependency_on_b", "dimension_key", "in_scope_after",
            "in_scope_before", "quality_delta", "source_withdrawal_affected",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise RetentionRollbackContractError(
                "RetentionDimensionResult 字段不精确")
        return cls(
            str(value["dimension_key"]),
            str(value["before_verdict"]),
            str(value["after_verdict"]),
            value["quality_delta"],
            value["in_scope_before"],
            value["in_scope_after"],
            value["dependency_on_b"],
            value["source_withdrawal_affected"],
            str(value["classification"]),
        )


@dataclass(frozen=True)
class RetentionCheckpoint:
    """LC-10 顺序中的一个只读状态摘要和 append-only 计数。"""

    phase_key: str
    core_state_sha256: str
    unaffected_state_sha256: str
    dependent_state_sha256: str
    capability_matrix_sha256: str
    declared_scope_sha256: str
    dump_artifact_sha256: str
    source_b_visible: int
    append_only_event_count: int
    host_learning_writes: int

    def __post_init__(self) -> None:
        if self.phase_key not in RETENTION_PHASE_KEYS:
            raise RetentionRollbackContractError("checkpoint phase 未登记")
        for name in (
                "core_state_sha256", "unaffected_state_sha256",
                "dependent_state_sha256", "capability_matrix_sha256",
                "declared_scope_sha256"):
            _sha256(getattr(self, name), where=name)
        _optional_sha256(
            self.dump_artifact_sha256, where="dump_artifact_sha256")
        _flag(self.source_b_visible, where="source_b_visible")
        _nonnegative(
            self.append_only_event_count, where="append_only_event_count")
        _nonnegative(self.host_learning_writes, where="host_learning_writes")

    def to_dict(self) -> dict[str, Any]:
        """投影为规范 JSON 对象。"""
        return {
            "append_only_event_count": self.append_only_event_count,
            "capability_matrix_sha256": self.capability_matrix_sha256,
            "core_state_sha256": self.core_state_sha256,
            "declared_scope_sha256": self.declared_scope_sha256,
            "dependent_state_sha256": self.dependent_state_sha256,
            "dump_artifact_sha256": self.dump_artifact_sha256,
            "host_learning_writes": self.host_learning_writes,
            "phase_key": self.phase_key,
            "source_b_visible": self.source_b_visible,
            "unaffected_state_sha256": self.unaffected_state_sha256,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RetentionCheckpoint":
        """从精确字段对象恢复 checkpoint。"""
        expected = {
            "append_only_event_count", "capability_matrix_sha256",
            "core_state_sha256", "declared_scope_sha256",
            "dependent_state_sha256", "dump_artifact_sha256",
            "host_learning_writes", "phase_key", "source_b_visible",
            "unaffected_state_sha256",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise RetentionRollbackContractError(
                "RetentionCheckpoint 字段不精确")
        return cls(
            str(value["phase_key"]),
            str(value["core_state_sha256"]),
            str(value["unaffected_state_sha256"]),
            str(value["dependent_state_sha256"]),
            str(value["capability_matrix_sha256"]),
            str(value["declared_scope_sha256"]),
            str(value["dump_artifact_sha256"]),
            value["source_b_visible"],
            value["append_only_event_count"],
            value["host_learning_writes"],
        )


def evaluate_retention_fixture(
        dimension_results: tuple[RetentionDimensionResult, ...],
        checkpoints: tuple[RetentionCheckpoint, ...],
        ) -> tuple[str, str]:
    """逐项核验 LC-10 顺序、bit identity、局部性和旧维最坏结果。"""
    if (not isinstance(dimension_results, tuple) or not dimension_results
            or not all(isinstance(item, RetentionDimensionResult)
                       for item in dimension_results)):
        raise RetentionRollbackContractError("dimension_results 类型非法")
    if (not isinstance(checkpoints, tuple)
            or not all(isinstance(item, RetentionCheckpoint)
                       for item in checkpoints)):
        raise RetentionRollbackContractError("checkpoints 类型非法")
    if tuple(item.phase_key for item in checkpoints) != RETENTION_PHASE_KEYS:
        raise RetentionRollbackContractError("LC-10 phase 顺序未冻结")
    if any(item.host_learning_writes != 0 for item in checkpoints):
        return "REJECT", "HOST_LEARNING_WRITE_NONZERO"
    baseline = checkpoints[0]
    if any(item.core_state_sha256 != baseline.core_state_sha256
           for item in checkpoints):
        return "REJECT", "CORE_STATE_DRIFT"
    if any(item.unaffected_state_sha256 != baseline.unaffected_state_sha256
           for item in checkpoints):
        return "REJECT", "UNAFFECTED_CAPABILITY_DRIFT"
    expected_visibility = (0, 1, 1, 1, 1, 0, 0, 0, 0, 0)
    if tuple(item.source_b_visible for item in checkpoints) != expected_visibility:
        return "REJECT", "SOURCE_WITHDRAWAL_VISIBILITY_DRIFT"
    counts = tuple(item.append_only_event_count for item in checkpoints)
    if any(later < earlier for earlier, later in zip(counts, counts[1:])):
        return "REJECT", "EVENT_COUNT_REWOUND"
    if not (counts[0] < counts[1] <= counts[2]
            and counts[2] == counts[3] == counts[4]
            and counts[4] < counts[5] < counts[6] <= counts[7]
            and counts[7] < counts[8] <= counts[9]):
        return "REJECT", "APPEND_ONLY_SEQUENCE_DRIFT"
    apply_b, reverify_b, dump_b, resume_b = checkpoints[1:5]
    if not (
            apply_b.dependent_state_sha256
            == reverify_b.dependent_state_sha256
            == dump_b.dependent_state_sha256
            == resume_b.dependent_state_sha256
            and apply_b.capability_matrix_sha256
            == reverify_b.capability_matrix_sha256
            == dump_b.capability_matrix_sha256
            == resume_b.capability_matrix_sha256
            and apply_b.dependent_state_sha256
            != baseline.dependent_state_sha256):
        return "REJECT", "A_B_STATE_SEQUENCE_DRIFT"
    if (dump_b.dump_artifact_sha256 == "NONE"
            or dump_b.dump_artifact_sha256
            != resume_b.dump_artifact_sha256):
        return "REJECT", "DUMP_RESUME_DIGEST_DRIFT"
    if any(item.dump_artifact_sha256 != "NONE"
           for item in checkpoints[:3] + checkpoints[5:]):
        return "REJECT", "DUMP_DIGEST_PHASE_LEAK"
    withdrawn = checkpoints[5]
    if withdrawn.dependent_state_sha256 == apply_b.dependent_state_sha256:
        return "REJECT", "SOURCE_WITHDRAWAL_NOT_LOCALIZED"
    rollback, reverify_rollback = checkpoints[6:8]
    for item in (rollback, reverify_rollback):
        if (item.dependent_state_sha256 != baseline.dependent_state_sha256
                or item.capability_matrix_sha256
                != baseline.capability_matrix_sha256):
            return "REJECT", "ROLLBACK_NOT_BASELINE_EQUIVALENT"
    if any(item.declared_scope_sha256 != baseline.declared_scope_sha256
           for item in checkpoints[:8]):
        return "REJECT", "SCOPE_CHANGED_BEFORE_CONTRACTION"
    contracted, reverified_scope = checkpoints[8:10]
    if (contracted.declared_scope_sha256
            != reverified_scope.declared_scope_sha256):
        return "REJECT", "SCOPE_CONTRACTION_NOT_REPLAYABLE"
    has_contraction = any(
        item.classification == "SCOPE_CONTRACTION"
        for item in dimension_results)
    scope_changed = (
        contracted.declared_scope_sha256 != baseline.declared_scope_sha256)
    if has_contraction != scope_changed:
        return "REJECT", "SCOPE_CONTRACTION_RECEIPT_MISMATCH"
    if any(item.classification == "FORGETTING"
           for item in dimension_results):
        return "REJECT", "OLD_CAPABILITY_FORGOTTEN"
    return "PASS", "NONE"


@dataclass(frozen=True)
class RetentionProtocolFixture:
    """一组可直接运行判定的 LC-10 正例或负例 fixture。"""

    fixture_key: str
    dimension_results: tuple[RetentionDimensionResult, ...]
    checkpoints: tuple[RetentionCheckpoint, ...]
    expected_verdict: str
    expected_failure_code: str

    def __post_init__(self) -> None:
        if self.fixture_key not in FIXTURE_KEYS:
            raise RetentionRollbackContractError("fixture_key 未登记")
        if self.expected_verdict not in FIXTURE_VERDICTS:
            raise RetentionRollbackContractError("expected_verdict 非法")
        _text(self.expected_failure_code, where="expected_failure_code")
        if tuple(sorted(
                self.dimension_results,
                key=lambda item: item.dimension_key)) != self.dimension_results:
            raise RetentionRollbackContractError(
                "dimension_results 必须按 dimension_key 排序")
        if len({item.dimension_key for item in self.dimension_results}) != len(
                self.dimension_results):
            raise RetentionRollbackContractError("dimension_key 不得重复")
        actual = evaluate_retention_fixture(
            self.dimension_results, self.checkpoints)
        if actual != (self.expected_verdict, self.expected_failure_code):
            raise RetentionRollbackContractError(
                "fixture 预期与直接 evaluator 结果不一致")

    def to_dict(self) -> dict[str, Any]:
        """投影为规范 JSON 对象。"""
        return {
            "checkpoints": [item.to_dict() for item in self.checkpoints],
            "dimension_results": [
                item.to_dict() for item in self.dimension_results],
            "expected_failure_code": self.expected_failure_code,
            "expected_verdict": self.expected_verdict,
            "fixture_key": self.fixture_key,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RetentionProtocolFixture":
        """从精确字段对象恢复 fixture。"""
        expected = {
            "checkpoints", "dimension_results", "expected_failure_code",
            "expected_verdict", "fixture_key",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise RetentionRollbackContractError(
                "RetentionProtocolFixture 字段不精确")
        return cls(
            str(value["fixture_key"]),
            tuple(RetentionDimensionResult.from_dict(item)
                  for item in value["dimension_results"]),
            tuple(RetentionCheckpoint.from_dict(item)
                  for item in value["checkpoints"]),
            str(value["expected_verdict"]),
            str(value["expected_failure_code"]),
        )


@dataclass(frozen=True)
class RetentionRuntimeBinding:
    """现有设施对 LC-10 职责的可用状态和 NE 边界。"""

    binding_key: str
    binding_state: str
    responsibilities: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    ne_conditions: tuple[str, ...]

    def __post_init__(self) -> None:
        _text(self.binding_key, where="binding_key")
        if self.binding_state not in RUNTIME_BINDING_STATES:
            raise RetentionRollbackContractError("binding_state 未登记")
        _strict_text_tuple(
            self.responsibilities, where="binding responsibilities")
        _strict_text_tuple(self.evidence_refs, where="binding evidence_refs")
        _strict_text_tuple(self.ne_conditions, where="binding ne_conditions")
        for item in self.evidence_refs:
            _relative_path(item, where="binding evidence_ref")

    def to_dict(self) -> dict[str, Any]:
        """投影为规范 JSON 对象。"""
        return {
            "binding_key": self.binding_key,
            "binding_state": self.binding_state,
            "evidence_refs": list(self.evidence_refs),
            "ne_conditions": list(self.ne_conditions),
            "responsibilities": list(self.responsibilities),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RetentionRuntimeBinding":
        """从精确字段对象恢复 runtime binding。"""
        expected = {
            "binding_key", "binding_state", "evidence_refs",
            "ne_conditions", "responsibilities",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise RetentionRollbackContractError(
                "RetentionRuntimeBinding 字段不精确")
        return cls(
            str(value["binding_key"]),
            str(value["binding_state"]),
            tuple(str(item) for item in value["responsibilities"]),
            tuple(str(item) for item in value["evidence_refs"]),
            tuple(str(item) for item in value["ne_conditions"]),
        )


@dataclass(frozen=True)
class RetentionEvidenceFile:
    """一个直接支撑 runtime binding 的文件级身份。"""

    relative_path: str
    byte_count: int
    sha256: str

    def __post_init__(self) -> None:
        _relative_path(self.relative_path, where="evidence relative_path")
        _nonnegative(self.byte_count, where="evidence byte_count")
        if self.byte_count == 0:
            raise RetentionRollbackContractError("evidence 文件不得为空")
        _sha256(self.sha256, where="evidence sha256")

    def to_dict(self) -> dict[str, Any]:
        """投影为规范 JSON 对象。"""
        return {
            "byte_count": self.byte_count,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RetentionEvidenceFile":
        """从精确字段对象恢复文件身份。"""
        if not isinstance(value, dict) or set(value) != {
                "byte_count", "relative_path", "sha256"}:
            raise RetentionRollbackContractError(
                "RetentionEvidenceFile 字段不精确")
        return cls(
            str(value["relative_path"]),
            value["byte_count"],
            str(value["sha256"]),
        )


@dataclass(frozen=True)
class RetentionRollbackManifest:
    """LC-10 的顺序、fixture、现有设施边界和零执行正式 artifact。"""

    format_version: int
    artifact_version: str
    artifact_status: str
    runtime_status: str
    task_key: str
    retention_sequence: tuple[str, ...]
    outcome_classes: tuple[str, ...]
    fixtures: tuple[RetentionProtocolFixture, ...]
    runtime_bindings: tuple[RetentionRuntimeBinding, ...]
    evidence_files: tuple[RetentionEvidenceFile, ...]
    verifier_dimensions: tuple[str, ...]
    verifier_ne_conditions: tuple[str, ...]
    future_prerequisite_states: CanonicalJsonObject
    actual_retention_evidenced: int
    runtime_pass_authority: int
    execution_state: CanonicalJsonObject

    def __post_init__(self) -> None:
        if self.format_version != FORMAT_VERSION:
            raise RetentionRollbackContractError("format_version 漂移")
        _text(self.artifact_version, where="artifact_version")
        if self.artifact_status != ARTIFACT_STATUS:
            raise RetentionRollbackContractError("artifact_status 非 COURSE_FROZEN")
        if self.runtime_status != RUNTIME_STATUS:
            raise RetentionRollbackContractError("runtime_status 非 NOT_STARTED")
        if self.task_key != "LC-10":
            raise RetentionRollbackContractError("task_key 非 LC-10")
        if _ordered_text_tuple(
                self.retention_sequence, where="retention_sequence") != (
                RETENTION_PHASE_KEYS):
            raise RetentionRollbackContractError("retention_sequence 漂移")
        if self.outcome_classes != OUTCOME_CLASSES:
            raise RetentionRollbackContractError("outcome_classes 漂移")
        if (not isinstance(self.fixtures, tuple)
                or not all(isinstance(item, RetentionProtocolFixture)
                           for item in self.fixtures)):
            raise RetentionRollbackContractError("fixtures 类型非法")
        fixtures = tuple(sorted(self.fixtures, key=lambda item: item.fixture_key))
        object.__setattr__(self, "fixtures", fixtures)
        if tuple(item.fixture_key for item in fixtures) != FIXTURE_KEYS:
            raise RetentionRollbackContractError("LC-10 fixture 未列全")
        if tuple(item.expected_verdict for item in fixtures) != (
                "REJECT", "PASS", "PASS"):
            raise RetentionRollbackContractError("正负 fixture verdict 漂移")
        if (not isinstance(self.runtime_bindings, tuple)
                or not self.runtime_bindings
                or not all(isinstance(item, RetentionRuntimeBinding)
                           for item in self.runtime_bindings)):
            raise RetentionRollbackContractError("runtime_bindings 类型非法")
        bindings = tuple(sorted(
            self.runtime_bindings, key=lambda item: item.binding_key))
        object.__setattr__(self, "runtime_bindings", bindings)
        if len({item.binding_key for item in bindings}) != len(bindings):
            raise RetentionRollbackContractError("runtime binding 重复")
        if (not isinstance(self.evidence_files, tuple)
                or not self.evidence_files
                or not all(isinstance(item, RetentionEvidenceFile)
                           for item in self.evidence_files)):
            raise RetentionRollbackContractError("evidence_files 类型非法")
        evidence = tuple(sorted(
            self.evidence_files, key=lambda item: item.relative_path))
        object.__setattr__(self, "evidence_files", evidence)
        if len({item.relative_path for item in evidence}) != len(evidence):
            raise RetentionRollbackContractError("evidence file 重复")
        referenced = {
            path for binding in bindings for path in binding.evidence_refs
        }
        inventoried = {item.relative_path for item in evidence}
        if referenced != inventoried:
            raise RetentionRollbackContractError(
                "runtime binding evidence 与文件 inventory 未闭合")
        if self.verifier_dimensions != VERIFIER_DIMENSIONS:
            raise RetentionRollbackContractError("verifier_dimensions 漂移")
        if self.verifier_ne_conditions != VERIFIER_NE_CONDITIONS:
            raise RetentionRollbackContractError("verifier_ne_conditions 漂移")
        if (not isinstance(self.future_prerequisite_states, CanonicalJsonObject)
                or self.future_prerequisite_states.to_value()
                != FUTURE_PREREQUISITE_STATES):
            raise RetentionRollbackContractError("future prerequisite 状态漂移")
        _flag(
            self.actual_retention_evidenced,
            where="actual_retention_evidenced")
        _flag(self.runtime_pass_authority, where="runtime_pass_authority")
        if self.actual_retention_evidenced != 0:
            raise RetentionRollbackContractError(
                "D-03 前不得写 RETENTION_EVIDENCED")
        if self.runtime_pass_authority != 0:
            raise RetentionRollbackContractError(
                "LC-10 不得签发 runtime PASS")
        if (not isinstance(self.execution_state, CanonicalJsonObject)
                or self.execution_state.to_value() != EXECUTION_STATE):
            raise RetentionRollbackContractError("LC-10 execution_state 非全零")

    def to_dict(self) -> dict[str, Any]:
        """投影为规范 JSON 对象。"""
        return {
            "actual_retention_evidenced": self.actual_retention_evidenced,
            "artifact_kind": "PH2_LC10_RETENTION_ROLLBACK_MANIFEST",
            "artifact_status": self.artifact_status,
            "artifact_version": self.artifact_version,
            "evidence_files": [item.to_dict() for item in self.evidence_files],
            "execution_state": self.execution_state.to_value(),
            "fixtures": [item.to_dict() for item in self.fixtures],
            "format_version": self.format_version,
            "future_prerequisite_states": (
                self.future_prerequisite_states.to_value()),
            "outcome_classes": list(self.outcome_classes),
            "retention_sequence": list(self.retention_sequence),
            "runtime_bindings": [
                item.to_dict() for item in self.runtime_bindings],
            "runtime_pass_authority": self.runtime_pass_authority,
            "runtime_status": self.runtime_status,
            "task_key": self.task_key,
            "verifier_dimensions": list(self.verifier_dimensions),
            "verifier_ne_conditions": list(self.verifier_ne_conditions),
        }

    def canonical_bytes(self) -> bytes:
        """返回带单一结尾换行的规范 manifest 字节。"""
        return canonical_json_bytes(self.to_dict()) + b"\n"

    def sha256(self) -> str:
        """返回规范 manifest SHA-256。"""
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RetentionRollbackManifest":
        """从精确字段对象恢复 LC-10 manifest。"""
        expected = {
            "actual_retention_evidenced", "artifact_kind", "artifact_status",
            "artifact_version", "evidence_files", "execution_state",
            "fixtures", "format_version", "future_prerequisite_states",
            "outcome_classes", "retention_sequence", "runtime_bindings",
            "runtime_pass_authority", "runtime_status", "task_key",
            "verifier_dimensions", "verifier_ne_conditions",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise RetentionRollbackContractError(
                "RetentionRollbackManifest 字段不精确")
        if value["artifact_kind"] != "PH2_LC10_RETENTION_ROLLBACK_MANIFEST":
            raise RetentionRollbackContractError("artifact_kind 非法")
        return cls(
            value["format_version"],
            str(value["artifact_version"]),
            str(value["artifact_status"]),
            str(value["runtime_status"]),
            str(value["task_key"]),
            tuple(str(item) for item in value["retention_sequence"]),
            tuple(str(item) for item in value["outcome_classes"]),
            tuple(RetentionProtocolFixture.from_dict(item)
                  for item in value["fixtures"]),
            tuple(RetentionRuntimeBinding.from_dict(item)
                  for item in value["runtime_bindings"]),
            tuple(RetentionEvidenceFile.from_dict(item)
                  for item in value["evidence_files"]),
            tuple(str(item) for item in value["verifier_dimensions"]),
            tuple(str(item) for item in value["verifier_ne_conditions"]),
            CanonicalJsonObject.from_value(value["future_prerequisite_states"]),
            value["actual_retention_evidenced"],
            value["runtime_pass_authority"],
            CanonicalJsonObject.from_value(value["execution_state"]),
        )


def write_retention_rollback_manifest(
        manifest: RetentionRollbackManifest,
        path: str | Path,
        ) -> Path:
    """独占或幂等发布 LC-10 manifest，拒绝覆盖不同内容。"""
    if not isinstance(manifest, RetentionRollbackManifest):
        raise RetentionRollbackContractError("manifest 类型非法")
    target = Path(path)
    payload = manifest.canonical_bytes()
    if target.exists():
        if not target.is_file() or target.read_bytes() != payload:
            raise RetentionRollbackContractError(
                "LC-10 manifest 已存在且内容不同")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(payload)
    except OSError as error:
        raise RetentionRollbackContractError("LC-10 manifest 无法发布") from error
    return target


def read_retention_rollback_manifest(
        path: str | Path,
        ) -> RetentionRollbackManifest:
    """严格回读规范 LC-10 manifest。"""
    try:
        payload = Path(path).read_bytes()
        if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
            raise RetentionRollbackContractError("LC-10 manifest newline 非法")
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
        assert isinstance(value, dict)
        manifest = RetentionRollbackManifest.from_dict(value)
    except RetentionRollbackContractError:
        raise
    except (OSError, UnicodeError, ValueError, AssertionError) as error:
        raise RetentionRollbackContractError("LC-10 manifest 损坏") from error
    if manifest.canonical_bytes() != payload:
        raise RetentionRollbackContractError("LC-10 manifest 非规范 JSON")
    return manifest


__all__ = [
    "ARTIFACT_STATUS",
    "CAPABILITY_VERDICTS",
    "EXECUTION_STATE",
    "FIXTURE_KEYS",
    "FORMAT_VERSION",
    "FUTURE_PREREQUISITE_STATES",
    "OUTCOME_CLASSES",
    "RETENTION_PHASE_KEYS",
    "RUNTIME_BINDING_STATES",
    "RUNTIME_STATUS",
    "RetentionCheckpoint",
    "RetentionDimensionResult",
    "RetentionEvidenceFile",
    "RetentionProtocolFixture",
    "RetentionRollbackContractError",
    "RetentionRollbackManifest",
    "RetentionRuntimeBinding",
    "VERIFIER_DIMENSIONS",
    "VERIFIER_NE_CONDITIONS",
    "evaluate_retention_fixture",
    "read_retention_rollback_manifest",
    "write_retention_rollback_manifest",
]
