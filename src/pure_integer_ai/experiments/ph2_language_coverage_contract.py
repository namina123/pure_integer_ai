"""PH2 语言能力前沿、verifier 范围和课程停止账的纯合同。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
)


FORMAT_VERSION = 1

CAPABILITY_KEYS = (
    "ATTRIBUTION_QUOTATION_PERSPECTIVE",
    "COMPARISON_QUANTITY_MEASURE",
    "DISCOURSE_INFORMATION_STRUCTURE",
    "EVALUATOR_RETENTION_RESOURCE",
    "EVENT_TIME_ASPECT",
    "LAYERED_GENERATION",
    "MEMORY_DYNAMICS",
    "MORPHOLOGY_WORD_FORM",
    "MULTIWORD_CONSTRUCTION",
    "NONLITERAL_CULTURAL",
    "NON_TEXT_MEDIA",
    "OPEN_SET_CONTINUAL_LEARNING",
    "PRAGMATIC_CLARIFICATION_REPAIR",
    "RAW_TEXT_NOISE",
    "RECURSIVE_PARSE",
    "REFERENCE_DISCOURSE_REVISION",
    "RELATION_LOGIC_FOUR_STATE",
    "SOURCE_UNCERTAINTY_REALITY",
    "TRANSFER_AXES",
    "TYPED_LEARNING_OBJECTIVES",
)

COVERAGE_STATES = (
    "ABSENT",
    "COURSE_FROZEN",
    "DESIGNED",
    "OUT_OF_SCOPE",
    "RETENTION_EVIDENCED",
    "RUNTIME_CONNECTED",
    "RUNTIME_EVIDENCED",
    "WALL_BLOCKED",
)

IMPLEMENTATION_STATES = (
    "ABSENT",
    "ACTIVE_RUNTIME",
    "DESIGN_ONLY",
    "SCAFFOLD_ONLY",
)

SCOPE_AXES = (
    "code_switch",
    "dialect",
    "domain",
    "era",
    "genre",
    "language",
    "length",
    "medium",
    "noise",
    "register",
    "script_orthography",
)

FACT_DIMENSIONS = (
    "DATA_ISOLATION",
    "DIRECTIONAL_CONSUMPTION",
    "LEARNING_LOOP",
    "OBSERVATION_FIDELITY",
    "REPRESENTATION",
    "RESOURCE",
    "RETENTION",
    "SCOPE",
    "VERIFIER_CAPABILITY",
)

DIRECTIONS = ("GENERATION", "REASONING", "UNDERSTANDING")
DIRECTION_APPLICABILITY = ("ABSENT", "N_A", "REQUIRED")

SAMPLE_FAMILIES = (
    "AMBIGUOUS",
    "GENERATION",
    "NEGATIVE",
    "POSITIVE",
    "RETENTION",
    "REVISION",
    "UNKNOWN",
)

SAMPLE_COVERAGE_STATES = (
    "FROZEN",
    "MISSING",
    "NE",
    "OUT_OF_SCOPE",
    "WALL_BLOCKED",
)

COURSE_EXIT_STATES = (
    "BASELINE_ONLY",
    "COURSE_FROZEN",
    "OUT_OF_SCOPE",
    "PARTIAL_COURSE",
    "WALL_BLOCKED",
)


class LanguageCoverageContractError(RuntimeError):
    """能力前沿、verifier 或课程账不满足冻结合同。"""


def _text(value: Any, *, where: str) -> str:
    if (not isinstance(value, str) or not value
            or value.strip() != value):
        raise LanguageCoverageContractError(
            f"{where} 必须是非空无首尾空白文本")
    return value


def _positive(value: Any, *, where: str) -> int:
    if type(value) is not int or value <= 0:
        raise LanguageCoverageContractError(f"{where} 必须是正严格整数")
    return value


def _flag(value: Any, *, where: str) -> int:
    if type(value) is not int or value not in {0, 1}:
        raise LanguageCoverageContractError(f"{where} 必须为 0/1")
    return value


def _strict_tuple(
        value: Any,
        *,
        where: str,
        allow_empty: bool = False,
        ) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise LanguageCoverageContractError(f"{where} 必须是 tuple")
    result = tuple(_text(item, where=where) for item in value)
    if not allow_empty and not result:
        raise LanguageCoverageContractError(f"{where} 不得为空")
    if tuple(sorted(set(result))) != result:
        raise LanguageCoverageContractError(f"{where} 必须排序且去重")
    return result


def _ordered_tuple(value: Any, *, where: str) -> tuple[str, ...]:
    if not isinstance(value, tuple) or not value:
        raise LanguageCoverageContractError(f"{where} 必须是非空 tuple")
    result = tuple(_text(item, where=where) for item in value)
    if len(result) != len(set(result)):
        raise LanguageCoverageContractError(f"{where} 不得重复")
    return result


def _require_keys(
        value: Any,
        expected: set[str] | tuple[str, ...],
        *,
        where: str,
        ) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(expected):
        raise LanguageCoverageContractError(f"{where} 字段不精确")
    return value


def _canonical_mapping(
        value: CanonicalJsonObject,
        *,
        where: str,
        keys: tuple[str, ...],
        ) -> dict[str, Any]:
    if not isinstance(value, CanonicalJsonObject):
        raise LanguageCoverageContractError(f"{where} 类型非法")
    restored = value.to_value()
    return _require_keys(restored, keys, where=where)


def _state(value: Any, *, where: str) -> str:
    text = _text(value, where=where)
    if text not in COVERAGE_STATES:
        raise LanguageCoverageContractError(f"{where} 状态非法")
    return text


@dataclass(frozen=True)
class LanguageCapabilityCoverageEntry:
    """一个能力族的十类承重字段和逐维事实状态。"""

    capability_key: str
    task_keys: tuple[str, ...]
    phenomenon_scope: str
    implementation_state: str
    scope_axes: CanonicalJsonObject
    observation_contracts: tuple[str, ...]
    representation_contracts: tuple[str, ...]
    candidate_evidence_lifecycle: tuple[str, ...]
    directional_consumption: CanonicalJsonObject
    dataset_isolation_contracts: tuple[str, ...]
    verifier_keys: tuple[str, ...]
    retention_contracts: tuple[str, ...]
    resource_contracts: tuple[str, ...]
    fact_states: CanonicalJsonObject
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.capability_key not in CAPABILITY_KEYS:
            raise LanguageCoverageContractError("capability_key 未登记")
        object.__setattr__(self, "task_keys", _strict_tuple(
            self.task_keys, where="capability task_keys"))
        _text(self.phenomenon_scope, where="capability phenomenon_scope")
        if self.implementation_state not in IMPLEMENTATION_STATES:
            raise LanguageCoverageContractError("implementation_state 非法")
        axes = _canonical_mapping(
            self.scope_axes, where="capability scope_axes", keys=SCOPE_AXES)
        for axis, values in axes.items():
            if (not isinstance(values, list) or not values
                    or tuple(sorted(set(values))) != tuple(values)):
                raise LanguageCoverageContractError(
                    f"scope axis {axis} 必须是排序去重的非空 list")
            for item in values:
                _text(item, where=f"scope axis {axis}")
        for name in (
                "observation_contracts", "representation_contracts",
                "candidate_evidence_lifecycle",
                "dataset_isolation_contracts", "verifier_keys",
                "retention_contracts", "resource_contracts",
                "evidence_refs"):
            object.__setattr__(self, name, _strict_tuple(
                getattr(self, name), where=f"capability {name}"))
        directions = _canonical_mapping(
            self.directional_consumption,
            where="capability directional_consumption",
            keys=DIRECTIONS,
        )
        for direction, raw in directions.items():
            item = _require_keys(raw, {
                "applicability", "consumer_refs", "fact_state",
                "write_permissions",
            }, where=f"direction {direction}")
            applicability = _text(
                item["applicability"], where=f"direction {direction}")
            if applicability not in DIRECTION_APPLICABILITY:
                raise LanguageCoverageContractError(
                    f"direction {direction} applicability 非法")
            fact_state = _state(
                item["fact_state"], where=f"direction {direction}")
            for list_name in ("consumer_refs", "write_permissions"):
                values = item[list_name]
                if (not isinstance(values, list)
                        or tuple(sorted(set(values))) != tuple(values)):
                    raise LanguageCoverageContractError(
                        f"direction {direction} {list_name} 非法")
                for value in values:
                    _text(value, where=f"direction {direction} {list_name}")
            if applicability == "N_A" and (
                    item["consumer_refs"] or item["write_permissions"]
                    or fact_state != "OUT_OF_SCOPE"):
                raise LanguageCoverageContractError(
                    f"direction {direction} N_A 不得伪装 consumer")
            if applicability == "ABSENT" and fact_state != "ABSENT":
                raise LanguageCoverageContractError(
                    f"direction {direction} ABSENT 与事实状态不一致")
        states = _canonical_mapping(
            self.fact_states, where="capability fact_states",
            keys=FACT_DIMENSIONS)
        for dimension, state in states.items():
            _state(state, where=f"fact state {dimension}")
        runtime_states = {
            "RUNTIME_CONNECTED", "RUNTIME_EVIDENCED", "RETENTION_EVIDENCED",
        }
        if (self.implementation_state != "ACTIVE_RUNTIME"
                and any(state in runtime_states for state in states.values())):
            raise LanguageCoverageContractError(
                "非 ACTIVE_RUNTIME 能力不得写 runtime/retention evidence")

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_evidence_lifecycle": list(
                self.candidate_evidence_lifecycle),
            "capability_key": self.capability_key,
            "dataset_isolation_contracts": list(
                self.dataset_isolation_contracts),
            "directional_consumption": self.directional_consumption.to_value(),
            "evidence_refs": list(self.evidence_refs),
            "fact_states": self.fact_states.to_value(),
            "implementation_state": self.implementation_state,
            "observation_contracts": list(self.observation_contracts),
            "phenomenon_scope": self.phenomenon_scope,
            "representation_contracts": list(self.representation_contracts),
            "resource_contracts": list(self.resource_contracts),
            "retention_contracts": list(self.retention_contracts),
            "scope_axes": self.scope_axes.to_value(),
            "task_keys": list(self.task_keys),
            "verifier_keys": list(self.verifier_keys),
        }

    @classmethod
    def from_dict(
            cls, value: dict[str, Any]) -> "LanguageCapabilityCoverageEntry":
        _require_keys(value, {
            "candidate_evidence_lifecycle", "capability_key",
            "dataset_isolation_contracts", "directional_consumption",
            "evidence_refs", "fact_states", "implementation_state",
            "observation_contracts", "phenomenon_scope",
            "representation_contracts", "resource_contracts",
            "retention_contracts", "scope_axes", "task_keys",
            "verifier_keys",
        }, where="LanguageCapabilityCoverageEntry")
        return cls(
            str(value["capability_key"]),
            tuple(str(item) for item in value["task_keys"]),
            str(value["phenomenon_scope"]),
            str(value["implementation_state"]),
            CanonicalJsonObject.from_value(dict(value["scope_axes"])),
            tuple(str(item) for item in value["observation_contracts"]),
            tuple(str(item) for item in value["representation_contracts"]),
            tuple(str(item) for item in value["candidate_evidence_lifecycle"]),
            CanonicalJsonObject.from_value(
                dict(value["directional_consumption"])),
            tuple(str(item) for item in value["dataset_isolation_contracts"]),
            tuple(str(item) for item in value["verifier_keys"]),
            tuple(str(item) for item in value["retention_contracts"]),
            tuple(str(item) for item in value["resource_contracts"]),
            CanonicalJsonObject.from_value(dict(value["fact_states"])),
            tuple(str(item) for item in value["evidence_refs"]),
        )


@dataclass(frozen=True)
class LanguageCapabilityCoverageLedger:
    """LC-00 的版本化能力前沿账，不生成总分。"""

    format_version: int
    ledger_version: str
    scope_statement: str
    entries: tuple[LanguageCapabilityCoverageEntry, ...]

    def __post_init__(self) -> None:
        if self.format_version != FORMAT_VERSION:
            raise LanguageCoverageContractError("coverage format_version 非法")
        _text(self.ledger_version, where="coverage ledger_version")
        _text(self.scope_statement, where="coverage scope_statement")
        if (not isinstance(self.entries, tuple)
                or not all(isinstance(item, LanguageCapabilityCoverageEntry)
                           for item in self.entries)):
            raise LanguageCoverageContractError("coverage entries 类型非法")
        object.__setattr__(self, "entries", tuple(sorted(
            self.entries, key=lambda item: item.capability_key)))
        keys = tuple(item.capability_key for item in self.entries)
        if keys != CAPABILITY_KEYS:
            raise LanguageCoverageContractError(
                "coverage ledger 必须逐项列全能力族")

    def to_dict(self) -> dict[str, Any]:
        return {
            "entries": [item.to_dict() for item in self.entries],
            "format_version": self.format_version,
            "ledger_version": self.ledger_version,
            "scope_statement": self.scope_statement,
        }

    @classmethod
    def from_dict(
            cls, value: dict[str, Any]) -> "LanguageCapabilityCoverageLedger":
        _require_keys(value, {
            "entries", "format_version", "ledger_version", "scope_statement",
        }, where="LanguageCapabilityCoverageLedger")
        return cls(
            value["format_version"], str(value["ledger_version"]),
            str(value["scope_statement"]),
            tuple(LanguageCapabilityCoverageEntry.from_dict(item)
                  for item in value["entries"]),
        )


@dataclass(frozen=True)
class VerifierCapabilityRecord:
    """一个 verifier 可判维度、盲区、owner 与 NE 条件。"""

    verifier_key: str
    verifier_version: str
    registry_state: str
    capability_keys: tuple[str, ...]
    decidable_dimensions: tuple[str, ...]
    input_prerequisites: tuple[str, ...]
    blind_spots: tuple[str, ...]
    owner_key: str
    evidence_sources: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    ne_conditions: tuple[str, ...]
    writer_permissions: tuple[str, ...]
    can_issue_runtime_pass: int

    def __post_init__(self) -> None:
        _text(self.verifier_key, where="verifier key")
        _text(self.verifier_version, where="verifier version")
        _state(self.registry_state, where="verifier registry_state")
        for name in (
                "capability_keys", "decidable_dimensions",
                "input_prerequisites", "blind_spots", "evidence_sources",
                "evidence_refs", "ne_conditions", "writer_permissions"):
            object.__setattr__(self, name, _strict_tuple(
                getattr(self, name), where=f"verifier {name}",
                allow_empty=name == "writer_permissions"))
        if any(key not in CAPABILITY_KEYS for key in self.capability_keys):
            raise LanguageCoverageContractError("verifier capability 未登记")
        _text(self.owner_key, where="verifier owner_key")
        _flag(self.can_issue_runtime_pass, where="can_issue_runtime_pass")
        if self.owner_key == "TEACHER":
            raise LanguageCoverageContractError("teacher 不得成为最终 verifier owner")
        if (self.can_issue_runtime_pass
                and self.registry_state != "RUNTIME_EVIDENCED"):
            raise LanguageCoverageContractError(
                "未获 runtime evidence 的 verifier 不得发 PASS")

    def to_dict(self) -> dict[str, Any]:
        return {
            "blind_spots": list(self.blind_spots),
            "can_issue_runtime_pass": self.can_issue_runtime_pass,
            "capability_keys": list(self.capability_keys),
            "decidable_dimensions": list(self.decidable_dimensions),
            "evidence_refs": list(self.evidence_refs),
            "evidence_sources": list(self.evidence_sources),
            "input_prerequisites": list(self.input_prerequisites),
            "ne_conditions": list(self.ne_conditions),
            "owner_key": self.owner_key,
            "registry_state": self.registry_state,
            "verifier_key": self.verifier_key,
            "verifier_version": self.verifier_version,
            "writer_permissions": list(self.writer_permissions),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "VerifierCapabilityRecord":
        _require_keys(value, {
            "blind_spots", "can_issue_runtime_pass", "capability_keys",
            "decidable_dimensions", "evidence_refs", "evidence_sources",
            "input_prerequisites", "ne_conditions", "owner_key",
            "registry_state", "verifier_key", "verifier_version",
            "writer_permissions",
        }, where="VerifierCapabilityRecord")
        return cls(
            str(value["verifier_key"]), str(value["verifier_version"]),
            str(value["registry_state"]),
            tuple(str(item) for item in value["capability_keys"]),
            tuple(str(item) for item in value["decidable_dimensions"]),
            tuple(str(item) for item in value["input_prerequisites"]),
            tuple(str(item) for item in value["blind_spots"]),
            str(value["owner_key"]),
            tuple(str(item) for item in value["evidence_sources"]),
            tuple(str(item) for item in value["evidence_refs"]),
            tuple(str(item) for item in value["ne_conditions"]),
            tuple(str(item) for item in value["writer_permissions"]),
            value["can_issue_runtime_pass"],
        )


@dataclass(frozen=True)
class VerifierCapabilityRegistry:
    """LC-11 的 verifier 范围注册表。"""

    format_version: int
    registry_version: str
    records: tuple[VerifierCapabilityRecord, ...]

    def __post_init__(self) -> None:
        if self.format_version != FORMAT_VERSION:
            raise LanguageCoverageContractError("verifier format_version 非法")
        _text(self.registry_version, where="verifier registry_version")
        if (not isinstance(self.records, tuple) or not self.records
                or not all(isinstance(item, VerifierCapabilityRecord)
                           for item in self.records)):
            raise LanguageCoverageContractError("verifier records 类型非法")
        object.__setattr__(self, "records", tuple(sorted(
            self.records, key=lambda item: item.verifier_key)))
        keys = tuple(item.verifier_key for item in self.records)
        if len(keys) != len(set(keys)):
            raise LanguageCoverageContractError("verifier key 重复")

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "records": [item.to_dict() for item in self.records],
            "registry_version": self.registry_version,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "VerifierCapabilityRegistry":
        _require_keys(value, {
            "format_version", "records", "registry_version",
        }, where="VerifierCapabilityRegistry")
        return cls(
            value["format_version"], str(value["registry_version"]),
            tuple(VerifierCapabilityRecord.from_dict(item)
                  for item in value["records"]),
        )


@dataclass(frozen=True)
class CapabilityCourseCoverage:
    """一个能力族的前置 DAG、七类课程覆盖和最早失效事实。"""

    capability_key: str
    prerequisite_capability_keys: tuple[str, ...]
    external_prerequisites: tuple[str, ...]
    sample_family_states: CanonicalJsonObject
    earliest_failure_stage: str
    failure_suffix: tuple[str, ...]
    exit_state: str
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.capability_key not in CAPABILITY_KEYS:
            raise LanguageCoverageContractError("course capability 未登记")
        object.__setattr__(self, "prerequisite_capability_keys", _strict_tuple(
            self.prerequisite_capability_keys,
            where="course prerequisite_capability_keys", allow_empty=True))
        if any(key not in CAPABILITY_KEYS
               for key in self.prerequisite_capability_keys):
            raise LanguageCoverageContractError("course prerequisite 未登记")
        if self.capability_key in self.prerequisite_capability_keys:
            raise LanguageCoverageContractError("course 不得自依赖")
        for name in ("external_prerequisites", "evidence_refs"):
            object.__setattr__(self, name, _strict_tuple(
                getattr(self, name), where=f"course {name}"))
        object.__setattr__(self, "failure_suffix", _ordered_tuple(
            self.failure_suffix, where="course failure_suffix"))
        families = _canonical_mapping(
            self.sample_family_states, where="course sample_family_states",
            keys=SAMPLE_FAMILIES)
        for family, state in families.items():
            if state not in SAMPLE_COVERAGE_STATES:
                raise LanguageCoverageContractError(
                    f"course sample family {family} 状态非法")
        _text(self.earliest_failure_stage, where="earliest_failure_stage")
        if self.exit_state not in COURSE_EXIT_STATES:
            raise LanguageCoverageContractError("course exit_state 非法")
        if (self.exit_state == "COURSE_FROZEN"
                and any(state == "MISSING" for state in families.values())):
            raise LanguageCoverageContractError(
                "COURSE_FROZEN 不得隐藏缺失课程族")

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability_key": self.capability_key,
            "earliest_failure_stage": self.earliest_failure_stage,
            "evidence_refs": list(self.evidence_refs),
            "exit_state": self.exit_state,
            "external_prerequisites": list(self.external_prerequisites),
            "failure_suffix": list(self.failure_suffix),
            "prerequisite_capability_keys": list(
                self.prerequisite_capability_keys),
            "sample_family_states": self.sample_family_states.to_value(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CapabilityCourseCoverage":
        _require_keys(value, {
            "capability_key", "earliest_failure_stage", "evidence_refs",
            "exit_state", "external_prerequisites", "failure_suffix",
            "prerequisite_capability_keys", "sample_family_states",
        }, where="CapabilityCourseCoverage")
        return cls(
            str(value["capability_key"]),
            tuple(str(item) for item in value["prerequisite_capability_keys"]),
            tuple(str(item) for item in value["external_prerequisites"]),
            CanonicalJsonObject.from_value(dict(value["sample_family_states"])),
            str(value["earliest_failure_stage"]),
            tuple(str(item) for item in value["failure_suffix"]),
            str(value["exit_state"]),
            tuple(str(item) for item in value["evidence_refs"]),
        )


def _assert_acyclic(records: tuple[CapabilityCourseCoverage, ...]) -> None:
    graph = {
        item.capability_key: item.prerequisite_capability_keys
        for item in records
    }
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(key: str) -> None:
        if key in visiting:
            raise LanguageCoverageContractError("course prerequisite DAG 有环")
        if key in visited:
            return
        visiting.add(key)
        for dependency in graph[key]:
            visit(dependency)
        visiting.remove(key)
        visited.add(key)

    for key in sorted(graph):
        visit(key)


@dataclass(frozen=True)
class CapabilityCourseCoverageLedger:
    """LC-12 初版课程覆盖、缺口和停止账。"""

    format_version: int
    ledger_version: str
    records: tuple[CapabilityCourseCoverage, ...]

    def __post_init__(self) -> None:
        if self.format_version != FORMAT_VERSION:
            raise LanguageCoverageContractError("course format_version 非法")
        _text(self.ledger_version, where="course ledger_version")
        if (not isinstance(self.records, tuple)
                or not all(isinstance(item, CapabilityCourseCoverage)
                           for item in self.records)):
            raise LanguageCoverageContractError("course records 类型非法")
        object.__setattr__(self, "records", tuple(sorted(
            self.records, key=lambda item: item.capability_key)))
        keys = tuple(item.capability_key for item in self.records)
        if keys != CAPABILITY_KEYS:
            raise LanguageCoverageContractError(
                "course ledger 必须逐项列全能力族")
        _assert_acyclic(self.records)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "ledger_version": self.ledger_version,
            "records": [item.to_dict() for item in self.records],
        }

    @classmethod
    def from_dict(
            cls, value: dict[str, Any]) -> "CapabilityCourseCoverageLedger":
        _require_keys(value, {
            "format_version", "ledger_version", "records",
        }, where="CapabilityCourseCoverageLedger")
        return cls(
            value["format_version"], str(value["ledger_version"]),
            tuple(CapabilityCourseCoverage.from_dict(item)
                  for item in value["records"]),
        )


__all__ = [
    "CAPABILITY_KEYS",
    "COURSE_EXIT_STATES",
    "COVERAGE_STATES",
    "CapabilityCourseCoverage",
    "CapabilityCourseCoverageLedger",
    "DIRECTIONS",
    "FACT_DIMENSIONS",
    "FORMAT_VERSION",
    "IMPLEMENTATION_STATES",
    "LanguageCapabilityCoverageEntry",
    "LanguageCapabilityCoverageLedger",
    "LanguageCoverageContractError",
    "SAMPLE_COVERAGE_STATES",
    "SAMPLE_FAMILIES",
    "SCOPE_AXES",
    "VerifierCapabilityRecord",
    "VerifierCapabilityRegistry",
]
