"""MD-04/05 中心扩散 probe 的冻结输入、原始运行和评测合同。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    StableRecordKey,
    canonical_json_line,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_language_baseline_manifest import (
    MD_BASELINE_KEYS,
    MD_HARD_INVARIANT_KEYS,
    MD_SAMPLE_GROUP_KEYS,
)
from pure_integer_ai.experiments.ph2_memory_dynamics_contract import (
    MemoryDynamicsRunReport,
    MemoryRingReceipt,
    MemoryDynamicsStopDecision,
    zero_execution_state,
)


FORMAT_VERSION = 1
MD04_PREREGISTRATION_VERSION = "MD-00-center-expansion-preregistration-v1"
MD04_PLAN_VERSION = "MD-04-center-diffusion-probe-plan-v1"
MD04_RUN_VERSION = "MD-04-center-diffusion-probe-runs-v1"
MD05_DECISION_VERSION = "MD-05-center-diffusion-decision-v1"
MD04_SCALE_FACTORS = (1, 10, 100)
MD04_ABLATION_KEYS = (
    "DEPENDENCY_INVALIDATION",
    "LAYERED_ATTRIBUTION",
    "STOP_DECISION",
    "TYPED_CENTER",
    "TYPED_CHANNEL_SELECTION",
)
PROBE_STANCES = ("NEUTRAL", "REFUTE", "SUPPORT")
PROBE_PLACEMENTS = ("COLD", "HOT", "INDEX")
PROBE_CHANNELS = (
    "L0_ORIGIN",
    "L1_WORK_MEMORY",
    "L2_EPISODE_DOCUMENT",
    "L3_MEMORY_OVERLAY",
    "L4_SEALED_PAGE",
    "SPECIAL_TYPED_INDEX",
)
QUERY_METRIC_KEYS = (
    "cache_hits",
    "clean_evictions",
    "cold_read_bytes",
    "dirty_flushes",
    "omitted_fault_reports",
    "page_faults",
    "page_in_records",
    "peak_hot_bytes",
    "peak_hot_objects",
    "prefetched_pages",
    "released_pins",
    "segment_reads",
)
OUTCOME_AUDIT_KEYS = (
    "full_store_rewrite_count",
    "held_out_train_overlap_count",
    "host_learning_write_count",
    "old_evidence_preserved",
    "reader_epoch_leak_count",
    "teacher_call_count",
    "unaffected_projection_bit_identical",
    "unrelated_revision_change_count",
)


class MD04ProbeContractError(RuntimeError):
    """probe 的冻结输入、运行事实或独立决断不闭合。"""


def _exact(value: Any, expected: set[str], *, where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise MD04ProbeContractError(f"{where} 字段不精确")
    return value


def _text(value: Any, *, where: str) -> str:
    if (not isinstance(value, str) or not value
            or value.strip() != value):
        raise MD04ProbeContractError(f"{where} 必须是非空无首尾空白文本")
    return value


def _nonnegative(value: Any, *, where: str) -> int:
    if type(value) is not int or value < 0:
        raise MD04ProbeContractError(f"{where} 必须是非负严格整数")
    return value


def _positive(value: Any, *, where: str) -> int:
    if type(value) is not int or value <= 0:
        raise MD04ProbeContractError(f"{where} 必须是正严格整数")
    return value


def _flag(value: Any, *, where: str) -> int:
    if type(value) is not int or value not in {0, 1}:
        raise MD04ProbeContractError(f"{where} 必须为 0/1")
    return value


def _key(value: Any, *, where: str) -> StableRecordKey:
    if not isinstance(value, StableRecordKey):
        raise MD04ProbeContractError(f"{where} 必须是 StableRecordKey")
    return value


def _keys(
        value: tuple[StableRecordKey, ...],
        *,
        where: str,
        allow_empty: bool = False,
        ) -> tuple[StableRecordKey, ...]:
    if (not isinstance(value, tuple)
            or (not allow_empty and not value)
            or any(not isinstance(item, StableRecordKey) for item in value)):
        raise MD04ProbeContractError(f"{where} 必须是稳定键 tuple")
    if value != tuple(sorted(set(value))):
        raise MD04ProbeContractError(f"{where} 必须排序去重")
    return value


def _texts(
        value: tuple[str, ...],
        *,
        where: str,
        allow_empty: bool = False,
        exact: tuple[str, ...] | None = None,
        ) -> tuple[str, ...]:
    if (not isinstance(value, tuple)
            or (not allow_empty and not value)
            or any(not isinstance(item, str) or not item
                   or item.strip() != item for item in value)):
        raise MD04ProbeContractError(f"{where} 必须是文本 tuple")
    if value != tuple(sorted(set(value))):
        raise MD04ProbeContractError(f"{where} 必须排序去重")
    if exact is not None and value != exact:
        raise MD04ProbeContractError(f"{where} 未精确列全")
    return value


def _enum(value: Any, allowed: tuple[str, ...], *, where: str) -> str:
    text = _text(value, where=where)
    if text not in allowed:
        raise MD04ProbeContractError(f"{where} 非法")
    return text


def _sha256(value: Any, *, where: str) -> str:
    text = _text(value, where=where)
    if (len(text) != 64
            or any(char not in "0123456789abcdef" for char in text)):
        raise MD04ProbeContractError(f"{where} 必须是小写 SHA-256")
    return text


def _relative_path(value: Any, *, where: str) -> str:
    text = _text(value, where=where)
    path = PurePosixPath(text)
    if (path.is_absolute() or ".." in path.parts or "\\" in text
            or path.as_posix() != text):
        raise MD04ProbeContractError(f"{where} 必须是安全 POSIX 相对路径")
    return text


def _key_values(values: tuple[StableRecordKey, ...]) -> list[list[int]]:
    return [item.to_list() for item in values]


def _keys_from(value: Any, *, where: str) -> tuple[StableRecordKey, ...]:
    if not isinstance(value, list):
        raise MD04ProbeContractError(f"{where} 必须是列表")
    try:
        return tuple(StableRecordKey.from_value(item, where=where)
                     for item in value)
    except Exception as error:
        raise MD04ProbeContractError(f"{where} 稳定键损坏") from error


@dataclass(frozen=True, order=True)
class ProbeCenterRef:
    """冻结 MD-03 center 的完整运行引用，不复制其可变宿主。"""

    envelope_key: StableRecordKey
    center_key: StableRecordKey
    direction: str
    target_key: StableRecordKey
    owner_key: StableRecordKey
    scope_key: StableRecordKey
    source_key: StableRecordKey
    version_key: StableRecordKey
    expansion_profile_key: StableRecordKey
    adoption_condition_keys: tuple[StableRecordKey, ...]

    def __post_init__(self) -> None:
        for name in (
                "envelope_key", "center_key", "target_key", "owner_key",
                "scope_key", "source_key", "version_key",
                "expansion_profile_key"):
            _key(getattr(self, name), where=f"center ref {name}")
        _enum(
            self.direction,
            ("GENERATION", "REASONING", "UNDERSTANDING"),
            where="center ref direction",
        )
        _keys(
            self.adoption_condition_keys,
            where="center ref adoption conditions",
            allow_empty=True,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "adoption_condition_keys": _key_values(
                self.adoption_condition_keys),
            "center_key": self.center_key.to_list(),
            "direction": self.direction,
            "envelope_key": self.envelope_key.to_list(),
            "expansion_profile_key": self.expansion_profile_key.to_list(),
            "owner_key": self.owner_key.to_list(),
            "scope_key": self.scope_key.to_list(),
            "source_key": self.source_key.to_list(),
            "target_key": self.target_key.to_list(),
            "version_key": self.version_key.to_list(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ProbeCenterRef":
        raw = _exact(value, {
            "adoption_condition_keys", "center_key", "direction",
            "envelope_key", "expansion_profile_key", "owner_key",
            "scope_key", "source_key", "target_key", "version_key",
        }, where="ProbeCenterRef")
        key = lambda name: StableRecordKey.from_value(raw[name], where=name)
        return cls(
            key("envelope_key"), key("center_key"), str(raw["direction"]),
            key("target_key"), key("owner_key"), key("scope_key"),
            key("source_key"), key("version_key"),
            key("expansion_profile_key"),
            _keys_from(
                raw["adoption_condition_keys"],
                where="adoption_condition_keys"),
        )


@dataclass(frozen=True, order=True)
class ProbeMemoryCandidate:
    """runtime 可见的 typed 候选；不含 evaluator 正确答案。"""

    candidate_key: StableRecordKey
    target_key: StableRecordKey
    relation_key: StableRecordKey
    evidence_key: StableRecordKey
    source_key: StableRecordKey
    structure_key: StableRecordKey
    dependency_keys: tuple[StableRecordKey, ...]
    placement: str
    channel_key: str
    distance: int
    recency_rank: int
    activation: int
    stance: str
    authorized: int
    access_allowed: int
    grounded: int

    def __post_init__(self) -> None:
        for name in (
                "candidate_key", "target_key", "relation_key",
                "evidence_key", "source_key", "structure_key"):
            _key(getattr(self, name), where=f"candidate {name}")
        _keys(self.dependency_keys, where="candidate dependencies")
        _enum(self.placement, PROBE_PLACEMENTS, where="candidate placement")
        _enum(self.channel_key, PROBE_CHANNELS, where="candidate channel")
        _nonnegative(self.distance, where="candidate distance")
        _nonnegative(self.recency_rank, where="candidate recency")
        _nonnegative(self.activation, where="candidate activation")
        _enum(self.stance, PROBE_STANCES, where="candidate stance")
        _flag(self.authorized, where="candidate authorized")
        _flag(self.access_allowed, where="candidate access")
        _flag(self.grounded, where="candidate grounded")
        expected_channel = {
            "COLD": "L4_SEALED_PAGE",
            "HOT": "L1_WORK_MEMORY",
            "INDEX": "SPECIAL_TYPED_INDEX",
        }[self.placement]
        if self.channel_key != expected_channel:
            raise MD04ProbeContractError("candidate placement/channel 漂移")

    def to_dict(self) -> dict[str, Any]:
        return {
            "access_allowed": self.access_allowed,
            "activation": self.activation,
            "authorized": self.authorized,
            "candidate_key": self.candidate_key.to_list(),
            "channel_key": self.channel_key,
            "dependency_keys": _key_values(self.dependency_keys),
            "distance": self.distance,
            "evidence_key": self.evidence_key.to_list(),
            "grounded": self.grounded,
            "placement": self.placement,
            "recency_rank": self.recency_rank,
            "relation_key": self.relation_key.to_list(),
            "source_key": self.source_key.to_list(),
            "stance": self.stance,
            "structure_key": self.structure_key.to_list(),
            "target_key": self.target_key.to_list(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ProbeMemoryCandidate":
        raw = _exact(value, {
            "access_allowed", "activation", "authorized", "candidate_key",
            "channel_key", "dependency_keys", "distance", "evidence_key",
            "grounded", "placement", "recency_rank", "relation_key",
            "source_key", "stance", "structure_key", "target_key",
        }, where="ProbeMemoryCandidate")
        key = lambda name: StableRecordKey.from_value(raw[name], where=name)
        return cls(
            key("candidate_key"), key("target_key"), key("relation_key"),
            key("evidence_key"), key("source_key"), key("structure_key"),
            _keys_from(raw["dependency_keys"], where="dependency_keys"),
            str(raw["placement"]), str(raw["channel_key"]),
            raw["distance"], raw["recency_rank"], raw["activation"],
            str(raw["stance"]), raw["authorized"],
            raw["access_allowed"], raw["grounded"],
        )


@dataclass(frozen=True)
class ProbeCaseDefinition:
    """一个结果盲 fixture：只冻结 obligation、可见候选和访问边界。"""

    case_key: StableRecordKey
    sample_group_key: str
    center_refs: tuple[ProbeCenterRef, ...]
    allowed_relation_keys: tuple[StableRecordKey, ...]
    hot_candidates: tuple[ProbeMemoryCandidate, ...]
    cold_candidates: tuple[ProbeMemoryCandidate, ...]
    cold_range_lower_key: StableRecordKey
    cold_range_upper_key: StableRecordKey
    cold_channel_admitted: int
    conflict_scan_required: int
    local_revision_dependency_key: StableRecordKey | None
    resource_ceiling: CanonicalJsonObject

    def __post_init__(self) -> None:
        _key(self.case_key, where="probe case key")
        _enum(
            self.sample_group_key,
            MD_SAMPLE_GROUP_KEYS,
            where="probe sample group",
        )
        if (not isinstance(self.center_refs, tuple) or not self.center_refs
                or any(not isinstance(item, ProbeCenterRef)
                       for item in self.center_refs)):
            raise MD04ProbeContractError("probe center refs 非法")
        if self.center_refs != tuple(sorted(self.center_refs)):
            raise MD04ProbeContractError("probe center refs 必须排序")
        if len({item.envelope_key for item in self.center_refs}) != len(
                self.center_refs):
            raise MD04ProbeContractError("probe center refs 重复")
        _keys(self.allowed_relation_keys, where="probe allowed relations")
        for name, placement in (
                ("hot_candidates", {"HOT", "INDEX"}),
                ("cold_candidates", {"COLD"})):
            values = getattr(self, name)
            if (not isinstance(values, tuple)
                    or any(not isinstance(item, ProbeMemoryCandidate)
                           or item.placement not in placement
                           for item in values)):
                raise MD04ProbeContractError(f"probe {name} 类型或层级非法")
            if values != tuple(sorted(values)):
                raise MD04ProbeContractError(f"probe {name} 必须排序")
        candidate_keys = tuple(
            item.candidate_key
            for item in (*self.hot_candidates, *self.cold_candidates))
        if len(set(candidate_keys)) != len(candidate_keys):
            raise MD04ProbeContractError("probe candidate key 重复")
        _key(self.cold_range_lower_key, where="probe cold lower")
        _key(self.cold_range_upper_key, where="probe cold upper")
        if self.cold_range_lower_key > self.cold_range_upper_key:
            raise MD04ProbeContractError("probe cold range 反向")
        if any(
                item.candidate_key < self.cold_range_lower_key
                or item.candidate_key > self.cold_range_upper_key
                for item in self.cold_candidates):
            raise MD04ProbeContractError("cold candidate 超出冻结 typed range")
        _flag(self.cold_channel_admitted, where="probe cold admitted")
        _flag(self.conflict_scan_required, where="probe conflict scan")
        if self.local_revision_dependency_key is not None:
            _key(
                self.local_revision_dependency_key,
                where="probe revision dependency")
        ceilings = self.resource_ceiling.to_value()
        expected = {
            "max_agenda_entries", "max_candidates", "max_cold_bytes",
            "max_consumptions", "max_logic_steps", "max_page_reads",
            "max_recomputes", "max_scanned_objects",
        }
        if set(ceilings) != expected:
            raise MD04ProbeContractError("probe resource ceiling 字段不精确")
        for key, value in ceilings.items():
            _nonnegative(value, where=f"probe ceiling {key}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed_relation_keys": _key_values(self.allowed_relation_keys),
            "case_key": self.case_key.to_list(),
            "center_refs": [item.to_dict() for item in self.center_refs],
            "cold_candidates": [item.to_dict() for item in self.cold_candidates],
            "cold_channel_admitted": self.cold_channel_admitted,
            "cold_range_lower_key": self.cold_range_lower_key.to_list(),
            "cold_range_upper_key": self.cold_range_upper_key.to_list(),
            "conflict_scan_required": self.conflict_scan_required,
            "hot_candidates": [item.to_dict() for item in self.hot_candidates],
            "local_revision_dependency_key": (
                None if self.local_revision_dependency_key is None
                else self.local_revision_dependency_key.to_list()),
            "resource_ceiling": self.resource_ceiling.to_value(),
            "sample_group_key": self.sample_group_key,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ProbeCaseDefinition":
        raw = _exact(value, {
            "allowed_relation_keys", "case_key", "center_refs",
            "cold_candidates", "cold_channel_admitted",
            "cold_range_lower_key", "cold_range_upper_key",
            "conflict_scan_required", "hot_candidates",
            "local_revision_dependency_key", "resource_ceiling",
            "sample_group_key",
        }, where="ProbeCaseDefinition")
        revision = raw["local_revision_dependency_key"]
        return cls(
            StableRecordKey.from_value(raw["case_key"], where="case_key"),
            str(raw["sample_group_key"]),
            tuple(ProbeCenterRef.from_dict(item)
                  for item in raw["center_refs"]),
            _keys_from(raw["allowed_relation_keys"], where="allowed relations"),
            tuple(ProbeMemoryCandidate.from_dict(item)
                  for item in raw["hot_candidates"]),
            tuple(ProbeMemoryCandidate.from_dict(item)
                  for item in raw["cold_candidates"]),
            StableRecordKey.from_value(
                raw["cold_range_lower_key"], where="cold lower"),
            StableRecordKey.from_value(
                raw["cold_range_upper_key"], where="cold upper"),
            raw["cold_channel_admitted"], raw["conflict_scan_required"],
            (None if revision is None else StableRecordKey.from_value(
                revision, where="revision dependency")),
            CanonicalJsonObject.from_value(raw["resource_ceiling"]),
        )


@dataclass(frozen=True)
class MD04ProbePlan:
    """首次运行前不可覆盖的四基线、fixture、预算和阈值。"""

    format_version: int
    plan_version: str
    preregistration_version: str
    md03_manifest_relative_path: str
    md03_manifest_sha256: str
    baseline_manifest_relative_path: str
    baseline_manifest_sha256: str
    strategy_keys: tuple[str, ...]
    sample_group_keys: tuple[str, ...]
    hard_invariant_keys: tuple[str, ...]
    ablation_keys: tuple[str, ...]
    scale_factors: tuple[int, ...]
    cases: tuple[ProbeCaseDefinition, ...]
    threshold_policy: CanonicalJsonObject
    results_observed: int
    execution_state: CanonicalJsonObject

    def __post_init__(self) -> None:
        if self.format_version != FORMAT_VERSION:
            raise MD04ProbeContractError("MD-04 plan format 非法")
        if self.plan_version != MD04_PLAN_VERSION:
            raise MD04ProbeContractError("MD-04 plan version 非法")
        if self.preregistration_version != MD04_PREREGISTRATION_VERSION:
            raise MD04ProbeContractError("MD-04 prereg version 漂移")
        _relative_path(self.md03_manifest_relative_path, where="MD-03 path")
        _sha256(self.md03_manifest_sha256, where="MD-03 hash")
        _relative_path(self.baseline_manifest_relative_path, where="baseline path")
        _sha256(self.baseline_manifest_sha256, where="baseline hash")
        _texts(
            self.strategy_keys, where="MD-04 strategies",
            exact=MD_BASELINE_KEYS)
        _texts(
            self.sample_group_keys, where="MD-04 sample groups",
            exact=MD_SAMPLE_GROUP_KEYS)
        _texts(
            self.hard_invariant_keys, where="MD-04 hard invariants",
            exact=MD_HARD_INVARIANT_KEYS)
        _texts(
            self.ablation_keys, where="MD-04 ablations",
            exact=MD04_ABLATION_KEYS)
        if self.scale_factors != MD04_SCALE_FACTORS:
            raise MD04ProbeContractError("MD-04 scale factors 未冻结为 1/10/100")
        if (not isinstance(self.cases, tuple) or not self.cases
                or any(not isinstance(item, ProbeCaseDefinition)
                       for item in self.cases)):
            raise MD04ProbeContractError("MD-04 cases 非法")
        case_keys = tuple(item.case_key for item in self.cases)
        if case_keys != tuple(sorted(set(case_keys))):
            raise MD04ProbeContractError("MD-04 cases 必须按 key 排序去重")
        if {item.sample_group_key for item in self.cases} != set(
                MD_SAMPLE_GROUP_KEYS):
            raise MD04ProbeContractError("MD-04 cases 未覆盖全部样本组")
        thresholds = self.threshold_policy.to_value()
        if set(thresholds) != {
                "ablation_min_degraded_dimensions",
                "challenge_min_improvements", "freeze_before_run",
                "hard_zero_required", "max_peak_hot_objects",
                "quality_regression_allowed",
                "unrelated_growth_allowed",
                "unrelated_query_cold_read_max"}:
            raise MD04ProbeContractError("MD-04 threshold 字段不精确")
        if thresholds != {
                "ablation_min_degraded_dimensions": 1,
                "challenge_min_improvements": 1,
                "freeze_before_run": 1,
                "hard_zero_required": 1,
                "max_peak_hot_objects": 4,
                "quality_regression_allowed": 0,
                "unrelated_growth_allowed": 0,
                "unrelated_query_cold_read_max": 0}:
            raise MD04ProbeContractError("MD-04 threshold 值漂移")
        if self.results_observed != 0:
            raise MD04ProbeContractError("MD-04 plan 不得携带结果")
        if self.execution_state.to_value() != zero_execution_state().to_value():
            raise MD04ProbeContractError("MD-04 plan execution state 必须全零")

    def to_dict(self) -> dict[str, Any]:
        return {
            "ablation_keys": list(self.ablation_keys),
            "artifact_kind": "PH2_MD04_PROBE_PLAN",
            "baseline_manifest_relative_path": self.baseline_manifest_relative_path,
            "baseline_manifest_sha256": self.baseline_manifest_sha256,
            "cases": [item.to_dict() for item in self.cases],
            "execution_state": self.execution_state.to_value(),
            "format_version": self.format_version,
            "hard_invariant_keys": list(self.hard_invariant_keys),
            "md03_manifest_relative_path": self.md03_manifest_relative_path,
            "md03_manifest_sha256": self.md03_manifest_sha256,
            "plan_version": self.plan_version,
            "preregistration_version": self.preregistration_version,
            "results_observed": self.results_observed,
            "sample_group_keys": list(self.sample_group_keys),
            "scale_factors": list(self.scale_factors),
            "strategy_keys": list(self.strategy_keys),
            "threshold_policy": self.threshold_policy.to_value(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MD04ProbePlan":
        raw = _exact(value, {
            "ablation_keys", "artifact_kind",
            "baseline_manifest_relative_path", "baseline_manifest_sha256",
            "cases", "execution_state", "format_version",
            "hard_invariant_keys", "md03_manifest_relative_path",
            "md03_manifest_sha256", "plan_version",
            "preregistration_version", "results_observed",
            "sample_group_keys", "scale_factors", "strategy_keys",
            "threshold_policy",
        }, where="MD04ProbePlan")
        if raw["artifact_kind"] != "PH2_MD04_PROBE_PLAN":
            raise MD04ProbeContractError("MD-04 plan artifact kind 非法")
        return cls(
            raw["format_version"], str(raw["plan_version"]),
            str(raw["preregistration_version"]),
            str(raw["md03_manifest_relative_path"]),
            str(raw["md03_manifest_sha256"]),
            str(raw["baseline_manifest_relative_path"]),
            str(raw["baseline_manifest_sha256"]),
            tuple(str(item) for item in raw["strategy_keys"]),
            tuple(str(item) for item in raw["sample_group_keys"]),
            tuple(str(item) for item in raw["hard_invariant_keys"]),
            tuple(str(item) for item in raw["ablation_keys"]),
            tuple(raw["scale_factors"]),
            tuple(ProbeCaseDefinition.from_dict(item) for item in raw["cases"]),
            CanonicalJsonObject.from_value(raw["threshold_policy"]),
            raw["results_observed"],
            CanonicalJsonObject.from_value(raw["execution_state"]),
        )

    def canonical_bytes(self) -> bytes:
        return canonical_json_line(self.to_dict())


@dataclass(frozen=True)
class ProbeCaseOutcome:
    """orchestrator 的结果盲原始事实；质量判定留给 MD-05。"""

    case_key: StableRecordKey
    sample_group_key: str
    strategy_key: str
    scale_factor: int
    receipt_records: tuple[MemoryRingReceipt, ...]
    stop_decisions: tuple[MemoryDynamicsStopDecision, ...]
    adopted_candidate_keys: tuple[StableRecordKey, ...]
    generated_candidate_keys: tuple[StableRecordKey, ...]
    query_metrics: CanonicalJsonObject
    audit_values: CanonicalJsonObject

    def __post_init__(self) -> None:
        _key(self.case_key, where="outcome case key")
        _enum(self.sample_group_key, MD_SAMPLE_GROUP_KEYS, where="outcome group")
        _enum(self.strategy_key, MD_BASELINE_KEYS, where="outcome strategy")
        if self.scale_factor not in MD04_SCALE_FACTORS:
            raise MD04ProbeContractError("outcome scale 非法")
        if (not isinstance(self.receipt_records, tuple)
                or not self.receipt_records
                or any(not isinstance(item, MemoryRingReceipt)
                       for item in self.receipt_records)):
            raise MD04ProbeContractError("outcome receipts 非法")
        receipt_keys = tuple(item.receipt_key for item in self.receipt_records)
        if receipt_keys != tuple(sorted(set(receipt_keys))):
            raise MD04ProbeContractError("outcome receipt key 未排序去重")
        if (not isinstance(self.stop_decisions, tuple)
                or not self.stop_decisions
                or any(not isinstance(item, MemoryDynamicsStopDecision)
                       for item in self.stop_decisions)):
            raise MD04ProbeContractError("outcome stop decisions 非法")
        decision_keys = tuple(item.decision_key for item in self.stop_decisions)
        if decision_keys != tuple(sorted(set(decision_keys))):
            raise MD04ProbeContractError("outcome decisions 未排序去重")
        if {item.center_key for item in self.receipt_records} != {
                item.center_key for item in self.stop_decisions}:
            raise MD04ProbeContractError("outcome receipt/decision center 不闭合")
        _keys(
            self.adopted_candidate_keys,
            where="outcome adopted",
            allow_empty=True)
        _keys(
            self.generated_candidate_keys,
            where="outcome generated",
            allow_empty=True)
        metrics = self.query_metrics.to_value()
        if tuple(metrics) != QUERY_METRIC_KEYS:
            raise MD04ProbeContractError("outcome query metrics 字段不精确")
        for key, value in metrics.items():
            _nonnegative(value, where=f"outcome metric {key}")
        audit = self.audit_values.to_value()
        if tuple(audit) != OUTCOME_AUDIT_KEYS:
            raise MD04ProbeContractError("outcome audit 字段不精确")
        for key, value in audit.items():
            if key in {
                    "old_evidence_preserved",
                    "unaffected_projection_bit_identical"}:
                _flag(value, where=f"outcome audit {key}")
            else:
                _nonnegative(value, where=f"outcome audit {key}")
        if audit["host_learning_write_count"] != 0:
            raise MD04ProbeContractError("outcome 出现宿主学习写")
        if audit["teacher_call_count"] != 0:
            raise MD04ProbeContractError("outcome 出现 teacher call")

    def to_dict(self) -> dict[str, Any]:
        return {
            "adopted_candidate_keys": _key_values(self.adopted_candidate_keys),
            "audit_values": self.audit_values.to_value(),
            "case_key": self.case_key.to_list(),
            "generated_candidate_keys": _key_values(
                self.generated_candidate_keys),
            "query_metrics": self.query_metrics.to_value(),
            "receipt_records": [item.to_dict() for item in self.receipt_records],
            "sample_group_key": self.sample_group_key,
            "scale_factor": self.scale_factor,
            "stop_decisions": [item.to_dict() for item in self.stop_decisions],
            "strategy_key": self.strategy_key,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ProbeCaseOutcome":
        raw = _exact(value, {
            "adopted_candidate_keys", "audit_values", "case_key",
            "generated_candidate_keys", "query_metrics", "receipt_records",
            "sample_group_key", "scale_factor", "stop_decisions",
            "strategy_key",
        }, where="ProbeCaseOutcome")
        return cls(
            StableRecordKey.from_value(raw["case_key"], where="case_key"),
            str(raw["sample_group_key"]), str(raw["strategy_key"]),
            raw["scale_factor"],
            tuple(MemoryRingReceipt.from_dict(item)
                  for item in raw["receipt_records"]),
            tuple(MemoryDynamicsStopDecision.from_dict(item)
                  for item in raw["stop_decisions"]),
            _keys_from(raw["adopted_candidate_keys"], where="adopted"),
            _keys_from(raw["generated_candidate_keys"], where="generated"),
            CanonicalJsonObject.from_value(raw["query_metrics"]),
            CanonicalJsonObject.from_value(raw["audit_values"]),
        )


@dataclass(frozen=True)
class ProbeAblationOutcome:
    """同 fixture 的单部件关闭结果，只保存原始 case outcome。"""

    ablation_key: str
    outcomes: tuple[ProbeCaseOutcome, ...]

    def __post_init__(self) -> None:
        _enum(self.ablation_key, MD04_ABLATION_KEYS, where="ablation key")
        if (not isinstance(self.outcomes, tuple) or not self.outcomes
                or any(not isinstance(item, ProbeCaseOutcome)
                       for item in self.outcomes)):
            raise MD04ProbeContractError("ablation outcomes 非法")
        keys = tuple(item.case_key for item in self.outcomes)
        if keys != tuple(sorted(set(keys))):
            raise MD04ProbeContractError("ablation case 未排序去重")
        if any(item.strategy_key != "OBLIGATION_CONDITIONED_MULTICHANNEL_STOP"
               or item.scale_factor != 1 for item in self.outcomes):
            raise MD04ProbeContractError("ablation 必须基于主策略 1x")

    def to_dict(self) -> dict[str, Any]:
        return {
            "ablation_key": self.ablation_key,
            "outcomes": [item.to_dict() for item in self.outcomes],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ProbeAblationOutcome":
        raw = _exact(value, {"ablation_key", "outcomes"},
                     where="ProbeAblationOutcome")
        return cls(
            str(raw["ablation_key"]),
            tuple(ProbeCaseOutcome.from_dict(item)
                  for item in raw["outcomes"]),
        )


@dataclass(frozen=True)
class MD04ProbeRunArtifact:
    """四基线、主策略规模曲线和五项消融的原始可回放事实。"""

    format_version: int
    run_version: str
    plan_relative_path: str
    plan_sha256: str
    strategy_outcomes: tuple[ProbeCaseOutcome, ...]
    scale_outcomes: tuple[ProbeCaseOutcome, ...]
    ablation_outcomes: tuple[ProbeAblationOutcome, ...]
    results_observed: int
    host_learning_write_count: int
    execution_state: CanonicalJsonObject

    def __post_init__(self) -> None:
        if self.format_version != FORMAT_VERSION or self.run_version != MD04_RUN_VERSION:
            raise MD04ProbeContractError("MD-04 run 版本非法")
        _relative_path(self.plan_relative_path, where="MD-04 plan path")
        _sha256(self.plan_sha256, where="MD-04 plan hash")
        if (not isinstance(self.strategy_outcomes, tuple)
                or not self.strategy_outcomes):
            raise MD04ProbeContractError("MD-04 strategy outcomes 为空")
        strategy_pairs = tuple(
            (item.strategy_key, item.case_key)
            for item in self.strategy_outcomes)
        if strategy_pairs != tuple(sorted(set(strategy_pairs))):
            raise MD04ProbeContractError("MD-04 strategy outcomes 未排序去重")
        if {item.strategy_key for item in self.strategy_outcomes} != set(
                MD_BASELINE_KEYS):
            raise MD04ProbeContractError("MD-04 四基线未列全")
        if any(item.scale_factor != 1 for item in self.strategy_outcomes):
            raise MD04ProbeContractError("MD-04 四基线必须使用相同 1x fixture")
        cases_by_strategy = {
            strategy: {item.case_key for item in self.strategy_outcomes
                       if item.strategy_key == strategy}
            for strategy in MD_BASELINE_KEYS
        }
        if len({tuple(sorted(value)) for value in cases_by_strategy.values()}) != 1:
            raise MD04ProbeContractError("MD-04 四基线 fixture 不一致")
        if (not isinstance(self.scale_outcomes, tuple)
                or not self.scale_outcomes):
            raise MD04ProbeContractError("MD-04 scale outcomes 为空")
        scale_pairs = tuple(
            (item.scale_factor, item.case_key) for item in self.scale_outcomes)
        if scale_pairs != tuple(sorted(set(scale_pairs))):
            raise MD04ProbeContractError("MD-04 scale outcomes 未排序去重")
        if {item.scale_factor for item in self.scale_outcomes} != set(
                MD04_SCALE_FACTORS):
            raise MD04ProbeContractError("MD-04 规模曲线未列全 1/10/100")
        if any(item.strategy_key != "OBLIGATION_CONDITIONED_MULTICHANNEL_STOP"
               for item in self.scale_outcomes):
            raise MD04ProbeContractError("MD-04 规模曲线必须来自主策略")
        if (not isinstance(self.ablation_outcomes, tuple)
                or tuple(item.ablation_key for item in self.ablation_outcomes)
                != MD04_ABLATION_KEYS):
            raise MD04ProbeContractError("MD-04 消融未精确列全")
        if self.results_observed != 1:
            raise MD04ProbeContractError("MD-04 run 必须标记已观察结果")
        if self.host_learning_write_count != 0:
            raise MD04ProbeContractError("MD-04 run 出现宿主学习写")
        if self.execution_state.to_value() != zero_execution_state().to_value():
            raise MD04ProbeContractError("MD-04 run execution state 必须全零")

    def to_dict(self) -> dict[str, Any]:
        return {
            "ablation_outcomes": [item.to_dict()
                                  for item in self.ablation_outcomes],
            "artifact_kind": "PH2_MD04_PROBE_RUNS",
            "execution_state": self.execution_state.to_value(),
            "format_version": self.format_version,
            "host_learning_write_count": self.host_learning_write_count,
            "plan_relative_path": self.plan_relative_path,
            "plan_sha256": self.plan_sha256,
            "results_observed": self.results_observed,
            "run_version": self.run_version,
            "scale_outcomes": [item.to_dict() for item in self.scale_outcomes],
            "strategy_outcomes": [item.to_dict()
                                  for item in self.strategy_outcomes],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MD04ProbeRunArtifact":
        raw = _exact(value, {
            "ablation_outcomes", "artifact_kind", "execution_state",
            "format_version", "host_learning_write_count",
            "plan_relative_path", "plan_sha256", "results_observed",
            "run_version", "scale_outcomes", "strategy_outcomes",
        }, where="MD04ProbeRunArtifact")
        if raw["artifact_kind"] != "PH2_MD04_PROBE_RUNS":
            raise MD04ProbeContractError("MD-04 run artifact kind 非法")
        return cls(
            raw["format_version"], str(raw["run_version"]),
            str(raw["plan_relative_path"]), str(raw["plan_sha256"]),
            tuple(ProbeCaseOutcome.from_dict(item)
                  for item in raw["strategy_outcomes"]),
            tuple(ProbeCaseOutcome.from_dict(item)
                  for item in raw["scale_outcomes"]),
            tuple(ProbeAblationOutcome.from_dict(item)
                  for item in raw["ablation_outcomes"]),
            raw["results_observed"], raw["host_learning_write_count"],
            CanonicalJsonObject.from_value(raw["execution_state"]),
        )

    def canonical_bytes(self) -> bytes:
        return canonical_json_line(self.to_dict())


@dataclass(frozen=True, order=True)
class ProbeEvaluatorLabel:
    """只归 evaluator owner 的 held-out 结果标签，runtime 不可导入。"""

    label_key: StableRecordKey
    evaluator_owner_key: StableRecordKey
    case_key: StableRecordKey
    center_key: StableRecordKey
    expected_status: str
    correct_candidate_keys: tuple[StableRecordKey, ...]
    forbidden_generation_keys: tuple[StableRecordKey, ...]
    combination_cluster_key: StableRecordKey
    structure_held_out: int
    distance_held_out: int
    split: str

    def __post_init__(self) -> None:
        for name in (
                "label_key", "evaluator_owner_key", "case_key",
                "center_key", "combination_cluster_key"):
            _key(getattr(self, name), where=f"evaluator label {name}")
        _enum(
            self.expected_status,
            ("ACCESS_BLOCKED", "BUDGET_EXHAUSTED", "CLARIFY",
             "GROUNDING_BLOCKED", "RESOLVED", "SUPERSEDED", "UNKNOWN"),
            where="evaluator expected status",
        )
        _keys(
            self.correct_candidate_keys,
            where="evaluator correct candidates",
            allow_empty=True)
        _keys(
            self.forbidden_generation_keys,
            where="evaluator forbidden generation",
            allow_empty=True)
        _flag(self.structure_held_out, where="evaluator structure heldout")
        _flag(self.distance_held_out, where="evaluator distance heldout")
        if self.split != "held_out":
            raise MD04ProbeContractError("evaluator label 必须物理属于 held_out")

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_key": self.case_key.to_list(),
            "center_key": self.center_key.to_list(),
            "combination_cluster_key": self.combination_cluster_key.to_list(),
            "correct_candidate_keys": _key_values(self.correct_candidate_keys),
            "distance_held_out": self.distance_held_out,
            "evaluator_owner_key": self.evaluator_owner_key.to_list(),
            "expected_status": self.expected_status,
            "forbidden_generation_keys": _key_values(
                self.forbidden_generation_keys),
            "label_key": self.label_key.to_list(),
            "split": self.split,
            "structure_held_out": self.structure_held_out,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ProbeEvaluatorLabel":
        raw = _exact(value, {
            "case_key", "center_key", "combination_cluster_key",
            "correct_candidate_keys", "distance_held_out",
            "evaluator_owner_key", "expected_status",
            "forbidden_generation_keys", "label_key", "split",
            "structure_held_out",
        }, where="ProbeEvaluatorLabel")
        key = lambda name: StableRecordKey.from_value(raw[name], where=name)
        return cls(
            key("label_key"), key("evaluator_owner_key"), key("case_key"),
            key("center_key"), str(raw["expected_status"]),
            _keys_from(raw["correct_candidate_keys"], where="correct candidates"),
            _keys_from(
                raw["forbidden_generation_keys"],
                where="forbidden generation"),
            key("combination_cluster_key"), raw["structure_held_out"],
            raw["distance_held_out"], str(raw["split"]),
        )


@dataclass(frozen=True)
class MD05DecisionArtifact:
    """独立 evaluator 对四基线、规模和消融形成的最终 PASS/REJECT。"""

    format_version: int
    decision_version: str
    plan_relative_path: str
    plan_sha256: str
    run_relative_path: str
    run_sha256: str
    labels: tuple[ProbeEvaluatorLabel, ...]
    strategy_reports: tuple[MemoryDynamicsRunReport, ...]
    comparison_evidence: CanonicalJsonObject
    ablation_evidence: CanonicalJsonObject
    hard_invariant_failures: tuple[str, ...]
    verdict: str
    results_observed: int
    evaluator_host_write_count: int
    execution_state: CanonicalJsonObject

    def __post_init__(self) -> None:
        if (self.format_version != FORMAT_VERSION
                or self.decision_version != MD05_DECISION_VERSION):
            raise MD04ProbeContractError("MD-05 decision 版本非法")
        _relative_path(self.plan_relative_path, where="MD-05 plan path")
        _sha256(self.plan_sha256, where="MD-05 plan hash")
        _relative_path(self.run_relative_path, where="MD-05 run path")
        _sha256(self.run_sha256, where="MD-05 run hash")
        if (not isinstance(self.labels, tuple) or not self.labels
                or any(not isinstance(item, ProbeEvaluatorLabel)
                       for item in self.labels)
                or self.labels != tuple(sorted(self.labels))):
            raise MD04ProbeContractError("MD-05 labels 非法或未排序")
        if (not isinstance(self.strategy_reports, tuple)
                or len(self.strategy_reports) != len(MD_BASELINE_KEYS)
                or any(not isinstance(item, MemoryDynamicsRunReport)
                       for item in self.strategy_reports)):
            raise MD04ProbeContractError("MD-05 strategy reports 非法")
        if tuple(item.strategy_key for item in self.strategy_reports) != (
                MD_BASELINE_KEYS):
            raise MD04ProbeContractError("MD-05 strategy reports 顺序漂移")
        comparison = self.comparison_evidence.to_value()
        if set(comparison) != {
                "challenge_improvement_count", "far_source_chain_recovered",
                "held_out_combination_overlap_count",
                "irrelevant_query_cold_read_bytes",
                "no_quality_regression", "resource_growth_violation_count",
                "time_advance_full_store_rewrites"}:
            raise MD04ProbeContractError("MD-05 comparison 字段不精确")
        for key, value in comparison.items():
            if key in {"far_source_chain_recovered", "no_quality_regression"}:
                _flag(value, where=f"MD-05 comparison {key}")
            else:
                _nonnegative(value, where=f"MD-05 comparison {key}")
        ablations = self.ablation_evidence.to_value()
        if tuple(ablations) != MD04_ABLATION_KEYS:
            raise MD04ProbeContractError("MD-05 ablation evidence 未列全")
        for key, value in ablations.items():
            if (not isinstance(value, list) or not value
                    or tuple(value) != tuple(sorted(set(value)))
                    or any(not isinstance(item, str) or not item
                           for item in value)):
                raise MD04ProbeContractError(
                    f"MD-05 ablation {key} 未产生独立退化维度")
        _texts(
            self.hard_invariant_failures,
            where="MD-05 hard failures",
            allow_empty=True)
        _enum(self.verdict, ("PASS", "REJECT"), where="MD-05 verdict")
        if self.results_observed != 1:
            raise MD04ProbeContractError("MD-05 必须观察结果")
        if self.evaluator_host_write_count != 0:
            raise MD04ProbeContractError("MD-05 evaluator 写入宿主")
        if self.execution_state.to_value() != zero_execution_state().to_value():
            raise MD04ProbeContractError("MD-05 execution state 必须全零")
        candidate = next(
            item for item in self.strategy_reports
            if item.strategy_key
            == "OBLIGATION_CONDITIONED_MULTICHANNEL_STOP")
        pass_conditions = (
            not self.hard_invariant_failures
            and candidate.probe_decision == "PASS"
            and comparison["no_quality_regression"] == 1
            and comparison["challenge_improvement_count"] >= 1
            and comparison["resource_growth_violation_count"] == 0
            and comparison["irrelevant_query_cold_read_bytes"] == 0
            and comparison["time_advance_full_store_rewrites"] == 0
            and comparison["held_out_combination_overlap_count"] == 0
            and comparison["far_source_chain_recovered"] == 1
        )
        if (self.verdict == "PASS") != pass_conditions:
            raise MD04ProbeContractError("MD-05 verdict 与合取证据不一致")

    def to_dict(self) -> dict[str, Any]:
        return {
            "ablation_evidence": self.ablation_evidence.to_value(),
            "artifact_kind": "PH2_MD05_PROBE_DECISION",
            "comparison_evidence": self.comparison_evidence.to_value(),
            "decision_version": self.decision_version,
            "evaluator_host_write_count": self.evaluator_host_write_count,
            "execution_state": self.execution_state.to_value(),
            "format_version": self.format_version,
            "hard_invariant_failures": list(self.hard_invariant_failures),
            "labels": [item.to_dict() for item in self.labels],
            "plan_relative_path": self.plan_relative_path,
            "plan_sha256": self.plan_sha256,
            "results_observed": self.results_observed,
            "run_relative_path": self.run_relative_path,
            "run_sha256": self.run_sha256,
            "strategy_reports": [item.to_dict()
                                 for item in self.strategy_reports],
            "verdict": self.verdict,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MD05DecisionArtifact":
        raw = _exact(value, {
            "ablation_evidence", "artifact_kind", "comparison_evidence",
            "decision_version", "evaluator_host_write_count",
            "execution_state", "format_version", "hard_invariant_failures",
            "labels", "plan_relative_path", "plan_sha256",
            "results_observed", "run_relative_path", "run_sha256",
            "strategy_reports", "verdict",
        }, where="MD05DecisionArtifact")
        if raw["artifact_kind"] != "PH2_MD05_PROBE_DECISION":
            raise MD04ProbeContractError("MD-05 artifact kind 非法")
        return cls(
            raw["format_version"], str(raw["decision_version"]),
            str(raw["plan_relative_path"]), str(raw["plan_sha256"]),
            str(raw["run_relative_path"]), str(raw["run_sha256"]),
            tuple(ProbeEvaluatorLabel.from_dict(item) for item in raw["labels"]),
            tuple(MemoryDynamicsRunReport.from_dict(item)
                  for item in raw["strategy_reports"]),
            CanonicalJsonObject.from_value(raw["comparison_evidence"]),
            CanonicalJsonObject.from_value(raw["ablation_evidence"]),
            tuple(str(item) for item in raw["hard_invariant_failures"]),
            str(raw["verdict"]), raw["results_observed"],
            raw["evaluator_host_write_count"],
            CanonicalJsonObject.from_value(raw["execution_state"]),
        )

    def canonical_bytes(self) -> bytes:
        return canonical_json_line(self.to_dict())


def write_immutable_artifact(
        artifact: MD04ProbePlan | MD04ProbeRunArtifact | MD05DecisionArtifact,
        path: str | Path,
        ) -> Path:
    """独占或逐字节幂等发布 MD-04/05 artifact。"""
    if not isinstance(
            artifact, (MD04ProbePlan, MD04ProbeRunArtifact,
                       MD05DecisionArtifact)):
        raise MD04ProbeContractError("MD-04/05 artifact 类型错误")
    target = Path(path)
    payload = artifact.canonical_bytes()
    if target.exists():
        if not target.is_file() or target.read_bytes() != payload:
            raise MD04ProbeContractError("MD-04/05 artifact 已存在且内容不同")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(payload)
    except OSError as error:
        raise MD04ProbeContractError("MD-04/05 artifact 无法发布") from error
    return target


def _read(path: str | Path, cls):
    try:
        payload = Path(path).read_bytes()
        if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
            raise MD04ProbeContractError("MD-04/05 artifact 换行非法")
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
        assert isinstance(value, dict)
        artifact = cls.from_dict(value)
    except MD04ProbeContractError:
        raise
    except Exception as error:
        raise MD04ProbeContractError("MD-04/05 artifact 损坏") from error
    if artifact.canonical_bytes() != payload:
        raise MD04ProbeContractError("MD-04/05 artifact 非规范字节")
    return artifact


def read_md04_probe_plan(path: str | Path) -> MD04ProbePlan:
    return _read(path, MD04ProbePlan)


def read_md04_probe_runs(path: str | Path) -> MD04ProbeRunArtifact:
    return _read(path, MD04ProbeRunArtifact)


def read_md05_decision(path: str | Path) -> MD05DecisionArtifact:
    return _read(path, MD05DecisionArtifact)


def artifact_sha256(artifact: Any) -> str:
    if not hasattr(artifact, "canonical_bytes"):
        raise MD04ProbeContractError("artifact 缺少 canonical_bytes")
    return hashlib.sha256(artifact.canonical_bytes()).hexdigest()


__all__ = [
    "FORMAT_VERSION",
    "MD04_ABLATION_KEYS",
    "MD04_PLAN_VERSION",
    "MD04_PREREGISTRATION_VERSION",
    "MD04_RUN_VERSION",
    "MD04_SCALE_FACTORS",
    "MD05_DECISION_VERSION",
    "MD04ProbeContractError",
    "MD04ProbePlan",
    "MD04ProbeRunArtifact",
    "MD05DecisionArtifact",
    "ProbeAblationOutcome",
    "ProbeCaseDefinition",
    "ProbeCaseOutcome",
    "ProbeCenterRef",
    "ProbeEvaluatorLabel",
    "ProbeMemoryCandidate",
    "artifact_sha256",
    "read_md04_probe_plan",
    "read_md04_probe_runs",
    "read_md05_decision",
    "write_immutable_artifact",
]
