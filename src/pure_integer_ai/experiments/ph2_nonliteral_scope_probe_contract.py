"""NL-00 非字面与文化依赖语言的分层 bounded scope probe 合同。"""
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
ARTIFACT_STATUS = "SCOPE_DECIDED"
RUNTIME_STATUS = "NOT_CONNECTED"
LAYER_KEYS = (
    "CONVENTIONAL_IMPLICATURE",
    "CULTURAL_ALLUSION",
    "IRONY_HUMOR",
    "LEXICALIZED_IDIOM",
    "PRODUCTIVE_METAPHOR_METONYMY",
)
PROBE_VERDICTS = ("NE", "PASS", "REJECT")
REPRESENTATION_STATES = (
    "ABSENT",
    "AVAILABLE_NOT_EXECUTED",
    "PARTIAL_SCAFFOLD",
)
EVALUATOR_STATES = (
    "INDEPENDENT_EVALUATOR_ABSENT",
    "STRUCTURAL_ONLY",
)
GROUNDING_STATES = (
    "EXTERNAL_GROUNDING_NE",
    "LANGUAGE_INTERNAL",
    "SOURCE_CONDITIONAL",
)
REQUIRED_INVARIANTS = {
    "CONVENTIONAL_IMPLICATURE": (
        "CANCELLABILITY_DISTINCT_FROM_TRUTH",
        "DIRECT_CONTENT_PRESERVED",
        "SOURCE_SCOPE_PRESERVED",
    ),
    "CULTURAL_ALLUSION": (
        "ALLUSION_SOURCE_IDENTITY",
        "CULTURAL_GROUNDING_AUTHORIZED",
        "UNKNOWN_NOT_GUESSED",
    ),
    "IRONY_HUMOR": (
        "LITERAL_CONTENT_PRESERVED",
        "MIND_READING_FORBIDDEN",
        "STANCE_EXPECTATION_CONTRAST",
    ),
    "LEXICALIZED_IDIOM": (
        "ANTI_LITERAL_BASELINE",
        "LEXICALIZATION_IDENTITY",
        "REGISTER_SCOPE",
    ),
    "PRODUCTIVE_METAPHOR_METONYMY": (
        "GROUNDING_SCOPE_EXPLICIT",
        "LITERAL_FIGURATIVE_COMPETITION",
        "SOURCE_TARGET_MAPPING_TYPED",
    ),
}
EXPECTED_LAYER_VERDICTS = {
    "CONVENTIONAL_IMPLICATURE": "REJECT",
    "CULTURAL_ALLUSION": "NE",
    "IRONY_HUMOR": "REJECT",
    "LEXICALIZED_IDIOM": "PASS",
    "PRODUCTIVE_METAPHOR_METONYMY": "REJECT",
}
VERIFIER_DIMENSIONS = (
    "CANDIDATE_SOURCE_EXPLICIT",
    "COUNTEREXAMPLE_FAMILY_PRESENT",
    "GROUNDING_BOUNDARY_EXPLICIT",
    "LAYER_SPECIFIC_REPRESENTABILITY",
    "NO_DEFINITIVE_MIND_READING",
    "NO_SCOPE_AGGREGATION",
    "SOURCE_SCOPE_PRESERVATION",
    "VERIFIER_AUTHORITY_BOUNDED",
    "ZERO_HOST_LEARNING_WRITE",
)
VERIFIER_NE_CONDITIONS = (
    "CONVENTIONAL_IMPLICATURE_RUNTIME_ABSENT",
    "CULTURAL_GROUNDING_EVALUATOR_UNAUTHORIZED",
    "DISC08_DEPTH_UNDECIDED",
    "DISC12_EVALUATOR_SIGNAL_UNDECIDED",
    "IRONY_HUMOR_RUNTIME_ABSENT",
    "PRODUCTIVE_FIGURATIVE_MAPPING_RUNTIME_ABSENT",
    "W1_EXTERNAL_GROUNDING_NOT_AVAILABLE",
)
UNRESOLVED_DECISION_KEYS = ("DISC-08", "DISC-12")
EXECUTION_STATE = {
    "assessment_updates": 0,
    "companion_writes": 0,
    "core_learning_writes": 0,
    "d03_published": 0,
    "evaluator_label_writes": 0,
    "formal_training_runs": 0,
    "mastered_claims": 0,
    "memory_learning_writes": 0,
    "readiness_claims": 0,
    "teacher_calls": 0,
    "use_learning_writes": 0,
    "w01_started": 0,
}


class NonliteralScopeProbeContractError(RuntimeError):
    """NL-00 分层、证据、墙边界或零写事实违反合同。"""


def _text(value: Any, *, where: str) -> str:
    if (not isinstance(value, str) or not value
            or value.strip() != value):
        raise NonliteralScopeProbeContractError(f"{where} 必须是规范文本")
    return value


def _nonnegative(value: Any, *, where: str) -> int:
    if type(value) is not int or value < 0:
        raise NonliteralScopeProbeContractError(f"{where} 必须是非负严格整数")
    return value


def _flag(value: Any, *, where: str) -> int:
    if type(value) is not int or value not in {0, 1}:
        raise NonliteralScopeProbeContractError(f"{where} 必须是 0/1")
    return value


def _sha256(value: Any, *, where: str) -> str:
    text = _text(value, where=where)
    if len(text) != 64 or any(item not in "0123456789abcdef" for item in text):
        raise NonliteralScopeProbeContractError(f"{where} 必须是 SHA-256")
    return text


def _relative_path(value: Any, *, where: str) -> str:
    text = _text(value, where=where)
    path = PurePosixPath(text)
    if (path.is_absolute() or ".." in path.parts or "\\" in text
            or path.as_posix() != text or ":" in path.parts[0]):
        raise NonliteralScopeProbeContractError(f"{where} 必须是安全相对路径")
    return text


def _strict_text_tuple(
        value: Any,
        *,
        where: str,
        allow_empty: bool = False,
        ) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise NonliteralScopeProbeContractError(f"{where} 必须是 tuple")
    if not allow_empty and not value:
        raise NonliteralScopeProbeContractError(f"{where} 不能为空")
    for item in value:
        _text(item, where=where)
    if tuple(sorted(set(value))) != value:
        raise NonliteralScopeProbeContractError(f"{where} 必须排序去重")
    return value


def evaluate_nonliteral_scope_probe(
        invariant_results: CanonicalJsonObject,
        *,
        host_learning_writes: int,
        ) -> str:
    """逐层取最坏 invariant；REJECT/NE 不得被表示存在或其他层掩盖。"""
    if not isinstance(invariant_results, CanonicalJsonObject):
        raise NonliteralScopeProbeContractError("invariant_results 类型非法")
    values = invariant_results.to_value()
    if not isinstance(values, dict) or not values:
        raise NonliteralScopeProbeContractError("invariant_results 不能为空")
    if any(value not in PROBE_VERDICTS for value in values.values()):
        raise NonliteralScopeProbeContractError("invariant verdict 未登记")
    writes = _nonnegative(host_learning_writes, where="host_learning_writes")
    if writes or "REJECT" in values.values():
        return "REJECT"
    if "NE" in values.values():
        return "NE"
    return "PASS"


@dataclass(frozen=True)
class NonliteralLayerDecision:
    """一个非字面层的表示、候选、反例、verifier、接地和范围裁决。"""

    layer_key: str
    representation_state: str
    typed_layer_available: int
    representation_contracts: tuple[str, ...]
    candidate_source_contracts: tuple[str, ...]
    counterexample_contracts: tuple[str, ...]
    invariant_results: CanonicalJsonObject
    evaluator_state: str
    grounding_state: str
    host_learning_writes: int
    verdict: str
    scope_decision: str
    wall_decision: str
    evidence_refs: tuple[str, ...]
    ne_conditions: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.layer_key not in LAYER_KEYS:
            raise NonliteralScopeProbeContractError("layer_key 未登记")
        if self.representation_state not in REPRESENTATION_STATES:
            raise NonliteralScopeProbeContractError("representation_state 未登记")
        _flag(self.typed_layer_available, where="typed_layer_available")
        _strict_text_tuple(
            self.representation_contracts, where="representation_contracts")
        _strict_text_tuple(
            self.candidate_source_contracts,
            where="candidate_source_contracts")
        _strict_text_tuple(
            self.counterexample_contracts, where="counterexample_contracts")
        if not isinstance(self.invariant_results, CanonicalJsonObject):
            raise NonliteralScopeProbeContractError("invariant_results 类型非法")
        values = self.invariant_results.to_value()
        if tuple(sorted(values)) != REQUIRED_INVARIANTS[self.layer_key]:
            raise NonliteralScopeProbeContractError("layer invariant 未列全")
        if self.evaluator_state not in EVALUATOR_STATES:
            raise NonliteralScopeProbeContractError("evaluator_state 未登记")
        if self.grounding_state not in GROUNDING_STATES:
            raise NonliteralScopeProbeContractError("grounding_state 未登记")
        actual = evaluate_nonliteral_scope_probe(
            self.invariant_results,
            host_learning_writes=self.host_learning_writes,
        )
        if self.verdict != actual or self.verdict not in PROBE_VERDICTS:
            raise NonliteralScopeProbeContractError("layer verdict 与直接结果漂移")
        if self.verdict != EXPECTED_LAYER_VERDICTS[self.layer_key]:
            raise NonliteralScopeProbeContractError("NL-00 冻结 verdict 漂移")
        _text(self.scope_decision, where="scope_decision")
        _text(self.wall_decision, where="wall_decision")
        _strict_text_tuple(self.evidence_refs, where="evidence_refs")
        for item in self.evidence_refs:
            _relative_path(item, where="evidence_ref")
        _strict_text_tuple(
            self.ne_conditions, where="ne_conditions", allow_empty=True)
        if self.host_learning_writes != 0:
            raise NonliteralScopeProbeContractError("NL-00 禁止宿主学习写")
        if self.layer_key == "LEXICALIZED_IDIOM":
            if not (
                    self.typed_layer_available == 1
                    and self.representation_state == "AVAILABLE_NOT_EXECUTED"
                    and self.evaluator_state == "STRUCTURAL_ONLY"
                    and self.grounding_state == "LANGUAGE_INTERNAL"
                    and self.verdict == "PASS"
                    and not self.ne_conditions):
                raise NonliteralScopeProbeContractError("词汇化习语有界边界漂移")
        elif self.typed_layer_available != 0 or not self.ne_conditions:
            raise NonliteralScopeProbeContractError("高阶层必须保留缺口和 NE 条件")
        if (self.layer_key == "CULTURAL_ALLUSION"
                and (self.verdict != "NE"
                     or self.grounding_state != "EXTERNAL_GROUNDING_NE")):
            raise NonliteralScopeProbeContractError("文化典故必须保持接地 NE")
        if (self.layer_key == "IRONY_HUMOR"
                and values["MIND_READING_FORBIDDEN"] != "PASS"):
            raise NonliteralScopeProbeContractError("反讽/幽默不得冒充心理真值")

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_source_contracts": list(self.candidate_source_contracts),
            "counterexample_contracts": list(self.counterexample_contracts),
            "evaluator_state": self.evaluator_state,
            "evidence_refs": list(self.evidence_refs),
            "grounding_state": self.grounding_state,
            "host_learning_writes": self.host_learning_writes,
            "invariant_results": self.invariant_results.to_value(),
            "layer_key": self.layer_key,
            "ne_conditions": list(self.ne_conditions),
            "representation_contracts": list(self.representation_contracts),
            "representation_state": self.representation_state,
            "scope_decision": self.scope_decision,
            "typed_layer_available": self.typed_layer_available,
            "verdict": self.verdict,
            "wall_decision": self.wall_decision,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "NonliteralLayerDecision":
        expected = {
            "candidate_source_contracts", "counterexample_contracts",
            "evaluator_state", "evidence_refs", "grounding_state",
            "host_learning_writes", "invariant_results", "layer_key",
            "ne_conditions", "representation_contracts",
            "representation_state", "scope_decision",
            "typed_layer_available", "verdict", "wall_decision",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise NonliteralScopeProbeContractError(
                "NonliteralLayerDecision 字段不精确")
        return cls(
            str(value["layer_key"]),
            str(value["representation_state"]),
            value["typed_layer_available"],
            tuple(str(item) for item in value["representation_contracts"]),
            tuple(str(item) for item in value["candidate_source_contracts"]),
            tuple(str(item) for item in value["counterexample_contracts"]),
            CanonicalJsonObject.from_value(value["invariant_results"]),
            str(value["evaluator_state"]),
            str(value["grounding_state"]),
            value["host_learning_writes"],
            str(value["verdict"]),
            str(value["scope_decision"]),
            str(value["wall_decision"]),
            tuple(str(item) for item in value["evidence_refs"]),
            tuple(str(item) for item in value["ne_conditions"]),
        )


@dataclass(frozen=True)
class NonliteralEvidenceFile:
    """NL-00 所复用现有 typed facility、课程或测试的文件身份。"""

    relative_path: str
    byte_count: int
    sha256: str

    def __post_init__(self) -> None:
        _relative_path(self.relative_path, where="evidence relative_path")
        if _nonnegative(self.byte_count, where="evidence byte_count") == 0:
            raise NonliteralScopeProbeContractError("evidence 文件不得为空")
        _sha256(self.sha256, where="evidence sha256")

    def to_dict(self) -> dict[str, Any]:
        return {
            "byte_count": self.byte_count,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "NonliteralEvidenceFile":
        if not isinstance(value, dict) or set(value) != {
                "byte_count", "relative_path", "sha256"}:
            raise NonliteralScopeProbeContractError(
                "NonliteralEvidenceFile 字段不精确")
        return cls(
            str(value["relative_path"]), value["byte_count"],
            str(value["sha256"]),
        )


@dataclass(frozen=True)
class NonliteralScopeProbeManifest:
    """NL-00 五层可行性、墙边界和零运行的正式 artifact。"""

    format_version: int
    artifact_version: str
    artifact_status: str
    runtime_status: str
    task_key: str
    baseline_manifest_relative_path: str
    baseline_manifest_sha256: str
    decisions: tuple[NonliteralLayerDecision, ...]
    evidence_files: tuple[NonliteralEvidenceFile, ...]
    pass_count: int
    reject_count: int
    ne_count: int
    scope_claim_only: int
    capability_learned_claims: int
    runtime_pass_authority: int
    unresolved_decision_keys: tuple[str, ...]
    verifier_dimensions: tuple[str, ...]
    verifier_ne_conditions: tuple[str, ...]
    execution_state: CanonicalJsonObject

    def __post_init__(self) -> None:
        if self.format_version != FORMAT_VERSION:
            raise NonliteralScopeProbeContractError("format_version 漂移")
        _text(self.artifact_version, where="artifact_version")
        if self.artifact_status != ARTIFACT_STATUS:
            raise NonliteralScopeProbeContractError("artifact_status 非 SCOPE_DECIDED")
        if self.runtime_status != RUNTIME_STATUS:
            raise NonliteralScopeProbeContractError("runtime_status 非 NOT_CONNECTED")
        if self.task_key != "NL-00":
            raise NonliteralScopeProbeContractError("task_key 非 NL-00")
        _relative_path(
            self.baseline_manifest_relative_path,
            where="baseline_manifest_relative_path")
        _sha256(self.baseline_manifest_sha256, where="baseline_manifest_sha256")
        if (not isinstance(self.decisions, tuple)
                or not all(isinstance(item, NonliteralLayerDecision)
                           for item in self.decisions)):
            raise NonliteralScopeProbeContractError("decisions 类型非法")
        decisions = tuple(sorted(self.decisions, key=lambda item: item.layer_key))
        object.__setattr__(self, "decisions", decisions)
        if tuple(item.layer_key for item in decisions) != LAYER_KEYS:
            raise NonliteralScopeProbeContractError("五类非字面层未闭合")
        if (not isinstance(self.evidence_files, tuple)
                or not all(isinstance(item, NonliteralEvidenceFile)
                           for item in self.evidence_files)):
            raise NonliteralScopeProbeContractError("evidence_files 类型非法")
        evidence = tuple(sorted(
            self.evidence_files, key=lambda item: item.relative_path))
        object.__setattr__(self, "evidence_files", evidence)
        if len({item.relative_path for item in evidence}) != len(evidence):
            raise NonliteralScopeProbeContractError("evidence file 重复")
        referenced = {path for item in decisions for path in item.evidence_refs}
        inventoried = {item.relative_path for item in evidence}
        if referenced != inventoried:
            raise NonliteralScopeProbeContractError("evidence inventory 未闭合")
        counts = {
            "pass_count": sum(item.verdict == "PASS" for item in decisions),
            "reject_count": sum(item.verdict == "REJECT" for item in decisions),
            "ne_count": sum(item.verdict == "NE" for item in decisions),
        }
        for name, actual in counts.items():
            if _nonnegative(getattr(self, name), where=name) != actual:
                raise NonliteralScopeProbeContractError(f"{name} 漂移")
        _flag(self.scope_claim_only, where="scope_claim_only")
        _nonnegative(
            self.capability_learned_claims, where="capability_learned_claims")
        _flag(self.runtime_pass_authority, where="runtime_pass_authority")
        if (self.scope_claim_only != 1 or self.capability_learned_claims != 0
                or self.runtime_pass_authority != 0):
            raise NonliteralScopeProbeContractError("scope probe 越权声明能力 PASS")
        if self.unresolved_decision_keys != UNRESOLVED_DECISION_KEYS:
            raise NonliteralScopeProbeContractError("待裁决问题未原样保留")
        if self.verifier_dimensions != VERIFIER_DIMENSIONS:
            raise NonliteralScopeProbeContractError("verifier_dimensions 漂移")
        if self.verifier_ne_conditions != VERIFIER_NE_CONDITIONS:
            raise NonliteralScopeProbeContractError("verifier_ne_conditions 漂移")
        if (not isinstance(self.execution_state, CanonicalJsonObject)
                or self.execution_state.to_value() != EXECUTION_STATE):
            raise NonliteralScopeProbeContractError("execution_state 非全零")

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_kind": "PH2_NL00_NONLITERAL_SCOPE_PROBE_MANIFEST",
            "artifact_status": self.artifact_status,
            "artifact_version": self.artifact_version,
            "baseline_manifest_relative_path": self.baseline_manifest_relative_path,
            "baseline_manifest_sha256": self.baseline_manifest_sha256,
            "capability_learned_claims": self.capability_learned_claims,
            "decisions": [item.to_dict() for item in self.decisions],
            "evidence_files": [item.to_dict() for item in self.evidence_files],
            "execution_state": self.execution_state.to_value(),
            "format_version": self.format_version,
            "ne_count": self.ne_count,
            "pass_count": self.pass_count,
            "reject_count": self.reject_count,
            "runtime_pass_authority": self.runtime_pass_authority,
            "runtime_status": self.runtime_status,
            "scope_claim_only": self.scope_claim_only,
            "task_key": self.task_key,
            "unresolved_decision_keys": list(self.unresolved_decision_keys),
            "verifier_dimensions": list(self.verifier_dimensions),
            "verifier_ne_conditions": list(self.verifier_ne_conditions),
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict()) + b"\n"

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "NonliteralScopeProbeManifest":
        expected = {
            "artifact_kind", "artifact_status", "artifact_version",
            "baseline_manifest_relative_path", "baseline_manifest_sha256",
            "capability_learned_claims", "decisions", "evidence_files",
            "execution_state", "format_version", "ne_count", "pass_count",
            "reject_count", "runtime_pass_authority", "runtime_status",
            "scope_claim_only", "task_key", "unresolved_decision_keys",
            "verifier_dimensions", "verifier_ne_conditions",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise NonliteralScopeProbeContractError(
                "NonliteralScopeProbeManifest 字段不精确")
        if value["artifact_kind"] != "PH2_NL00_NONLITERAL_SCOPE_PROBE_MANIFEST":
            raise NonliteralScopeProbeContractError("artifact_kind 非法")
        return cls(
            value["format_version"], str(value["artifact_version"]),
            str(value["artifact_status"]), str(value["runtime_status"]),
            str(value["task_key"]),
            str(value["baseline_manifest_relative_path"]),
            str(value["baseline_manifest_sha256"]),
            tuple(NonliteralLayerDecision.from_dict(item)
                  for item in value["decisions"]),
            tuple(NonliteralEvidenceFile.from_dict(item)
                  for item in value["evidence_files"]),
            value["pass_count"], value["reject_count"], value["ne_count"],
            value["scope_claim_only"], value["capability_learned_claims"],
            value["runtime_pass_authority"],
            tuple(str(item) for item in value["unresolved_decision_keys"]),
            tuple(str(item) for item in value["verifier_dimensions"]),
            tuple(str(item) for item in value["verifier_ne_conditions"]),
            CanonicalJsonObject.from_value(value["execution_state"]),
        )


def write_nonliteral_scope_probe_manifest(
        manifest: NonliteralScopeProbeManifest,
        path: str | Path,
        ) -> Path:
    """独占或幂等发布 NL-00 manifest，拒绝覆盖不同内容。"""
    if not isinstance(manifest, NonliteralScopeProbeManifest):
        raise NonliteralScopeProbeContractError("manifest 类型非法")
    target = Path(path)
    payload = manifest.canonical_bytes()
    if target.exists():
        if not target.is_file() or target.read_bytes() != payload:
            raise NonliteralScopeProbeContractError("NL-00 manifest 已存在且内容不同")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(payload)
    except OSError as error:
        raise NonliteralScopeProbeContractError("NL-00 manifest 无法发布") from error
    return target


def read_nonliteral_scope_probe_manifest(
        path: str | Path,
        ) -> NonliteralScopeProbeManifest:
    """严格回读规范 NL-00 manifest。"""
    try:
        payload = Path(path).read_bytes()
        if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
            raise NonliteralScopeProbeContractError("NL-00 manifest newline 非法")
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
        assert isinstance(value, dict)
        manifest = NonliteralScopeProbeManifest.from_dict(value)
    except NonliteralScopeProbeContractError:
        raise
    except (OSError, UnicodeError, ValueError, AssertionError) as error:
        raise NonliteralScopeProbeContractError("NL-00 manifest 损坏") from error
    if manifest.canonical_bytes() != payload:
        raise NonliteralScopeProbeContractError("NL-00 manifest 非规范 JSON")
    return manifest


def verify_nonliteral_scope_probe_files(
        manifest: NonliteralScopeProbeManifest,
        *,
        repository_root: str | Path,
        ) -> None:
    """逐字节回验 baseline 和所有现有 facility，不接受文件身份漂移。"""
    if not isinstance(manifest, NonliteralScopeProbeManifest):
        raise NonliteralScopeProbeContractError("manifest 类型非法")
    root = Path(repository_root).resolve()
    baseline = root / Path(*manifest.baseline_manifest_relative_path.split("/"))
    if (not baseline.is_file()
            or hashlib.sha256(baseline.read_bytes()).hexdigest()
            != manifest.baseline_manifest_sha256):
        raise NonliteralScopeProbeContractError("NL-00 baseline 文件身份漂移")
    for item in manifest.evidence_files:
        path = root / Path(*item.relative_path.split("/"))
        if (not path.is_file() or path.stat().st_size != item.byte_count
                or hashlib.sha256(path.read_bytes()).hexdigest() != item.sha256):
            raise NonliteralScopeProbeContractError("NL-00 evidence 文件身份漂移")


__all__ = [
    "ARTIFACT_STATUS",
    "EVALUATOR_STATES",
    "EXECUTION_STATE",
    "EXPECTED_LAYER_VERDICTS",
    "FORMAT_VERSION",
    "GROUNDING_STATES",
    "LAYER_KEYS",
    "NonliteralEvidenceFile",
    "NonliteralLayerDecision",
    "NonliteralScopeProbeContractError",
    "NonliteralScopeProbeManifest",
    "PROBE_VERDICTS",
    "REPRESENTATION_STATES",
    "REQUIRED_INVARIANTS",
    "RUNTIME_STATUS",
    "UNRESOLVED_DECISION_KEYS",
    "VERIFIER_DIMENSIONS",
    "VERIFIER_NE_CONDITIONS",
    "evaluate_nonliteral_scope_probe",
    "read_nonliteral_scope_probe_manifest",
    "verify_nonliteral_scope_probe_files",
    "write_nonliteral_scope_probe_manifest",
]
