"""MD-01 中心、扩域、receipt、停止和 probe 报告的纯合同。"""
from __future__ import annotations

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
)


FORMAT_VERSION = 1

DIRECTIONS = ("GENERATION", "REASONING", "UNDERSTANDING")
CENTER_STRENGTHS = ("CONDITIONAL", "MANDATORY", "SPECULATIVE")
CENTER_STATES = ("ACTIVE", "CLOSED", "PAUSED", "SUPERSEDED")
ORIGIN_KINDS = (
    "GOAL",
    "OCCURRENCE",
    "OPEN_QUESTION",
    "PROPOSITION",
    "QUERY",
    "REVISION",
    "SPAN",
    "STRUCTURE_GAP",
)
EXPANSION_CHANNELS = (
    "L0_ORIGIN",
    "L1_WORK_MEMORY",
    "L2_EPISODE_DOCUMENT",
    "L3_MEMORY_OVERLAY",
    "L4_SEALED_PAGE",
    "SPECIAL_TYPED_INDEX",
)
RING_ACTIONS = ("CONTINUE", "PAUSE", "STOP")
STOP_STATES = (
    "ACCESS_BLOCKED",
    "BUDGET_EXHAUSTED",
    "CLARIFY",
    "GROUNDING_BLOCKED",
    "RESOLVED",
    "SUPERSEDED",
    "UNKNOWN",
)
RUN_STATUSES = ("COMPLETE", "NOT_STARTED")
PROBE_DECISIONS = ("NOT_EVALUATED", "PASS", "REJECT")

MD_METRIC_KEYS = (
    "ADOPTED_CORRECT",
    "AGENDA_ENTRIES",
    "CLARIFY_UNKNOWN_BLOCKED_CLASSIFICATION",
    "CONSUMED_OBJECTS",
    "GENERATION_ADDRESSEE_RECOVERABILITY",
    "GENERATION_SEMANTIC_PRESERVATION",
    "LOGIC_STEPS",
    "MISSED_REFUTE",
    "OLD_OBSERVATION_EVIDENCE_PRESERVED",
    "OPENED_PAGE_SEGMENT",
    "OWNER_SCOPE_VERSION_VIOLATION",
    "RECEIPT_COMPLETENESS",
    "RECOMPUTED_OBJECTS",
    "SCANNED_OBJECTS",
    "TEACHER_HELD_OUT_LEAKAGE",
    "UNAFFECTED_PROJECTION_BIT_IDENTITY",
    "WRONG_ADOPTION",
)
MD01_CONTRACT_KEYS = (
    "MEMORY_ATTENTION_CENTER",
    "MEMORY_DYNAMICS_RUN_REPORT",
    "MEMORY_DYNAMICS_STOP_DECISION",
    "MEMORY_EXPANSION_PROFILE",
    "MEMORY_RING_RECEIPT",
)
MD01_INVARIANT_KEYS = (
    "ACTIVATION_NEVER_AUTHORIZES_ADOPTION",
    "BUDGET_EXHAUSTED_DISTINCT_FROM_UNKNOWN",
    "EVALUATOR_HELD_OUT_ZERO_HOST_WRITE",
    "HARD_VETO_NOT_SCORE_OVERRIDABLE",
    "OWNER_SCOPE_VERSION_FAIL_CLOSED",
    "PHYSICAL_READ_SHARING_NEVER_MERGES_CENTER_IDENTITY",
    "RECEIPT_REQUIRED_FOR_EVERY_CHANNEL_READ",
    "RESOLVED_REQUIRES_CONFLICT_AND_AUTHORIZATION_EVIDENCE",
    "TEACHER_CALL_ZERO",
)
MD01_VERIFIER_DIMENSIONS = (
    "CANONICAL_ROUND_TRIP",
    "CENTER_IDENTITY_AND_BOUNDARY",
    "CHANNEL_BUDGET_AND_ADMISSION",
    "MULTI_CENTER_IDENTITY_PRESERVATION",
    "OWNER_SCOPE_VERSION_ISOLATION",
    "RECEIPT_COUNT_AND_REASON_COMPLETENESS",
    "STOP_STATE_SUFFICIENCY",
    "ZERO_LEARNING_WRITE",
)
MD01_NE_CONDITIONS = (
    "CENTER_DIFFUSION_QUALITY_REQUESTED",
    "MD02_ADAPTER_NOT_EXECUTED",
    "MD03_DIRECTIONAL_ADAPTER_NOT_EXECUTED",
    "MD04_PROBE_NOT_EXECUTED",
    "RUNTIME_GENERALIZATION_REQUESTED",
)
EXECUTION_STATE_KEYS = (
    "companion_writes",
    "core_learning_writes",
    "d03_published",
    "formal_training_runs",
    "mastered_claims",
    "memory_learning_writes",
    "readiness_claims",
    "teacher_calls",
    "use_learning_writes",
    "w01_started",
)
BUDGET_FIELDS = (
    "max_scanned_objects",
    "max_candidates",
    "max_agenda_entries",
    "max_consumptions",
    "max_page_reads",
    "max_recomputes",
    "max_logic_steps",
    "max_cold_bytes",
)


class MemoryDynamicsContractError(RuntimeError):
    """MD-01 合同字段、边界、预算或停止状态不完整。"""


def _exact_keys(value: Any, expected: set[str], *, where: str) -> dict[str, Any]:
    """要求 JSON object 的字段集合精确相等。"""
    if not isinstance(value, dict) or set(value) != expected:
        raise MemoryDynamicsContractError(f"{where} 字段不精确")
    return value


def _text(value: Any, *, where: str) -> str:
    """要求无首尾空白的非空文本。"""
    if (not isinstance(value, str) or not value
            or value.strip() != value):
        raise MemoryDynamicsContractError(f"{where} 必须是非空无首尾空白文本")
    return value


def _enum(value: Any, allowed: tuple[str, ...], *, where: str) -> str:
    """要求文本属于冻结枚举。"""
    text = _text(value, where=where)
    if text not in allowed:
        raise MemoryDynamicsContractError(f"{where} 状态非法")
    return text


def _positive(value: Any, *, where: str) -> int:
    """要求正严格整数。"""
    if type(value) is not int or value <= 0:
        raise MemoryDynamicsContractError(f"{where} 必须是正严格整数")
    return value


def _nonnegative(value: Any, *, where: str) -> int:
    """要求非负严格整数。"""
    if type(value) is not int or value < 0:
        raise MemoryDynamicsContractError(f"{where} 必须是非负严格整数")
    return value


def _flag(value: Any, *, where: str) -> int:
    """要求严格整数布尔标志。"""
    if type(value) is not int or value not in (0, 1):
        raise MemoryDynamicsContractError(f"{where} 必须为 0/1")
    return value


def _zero(value: Any, *, where: str) -> int:
    """要求未执行状态使用严格整数零。"""
    if type(value) is not int or value != 0:
        raise MemoryDynamicsContractError(f"{where} 必须为 0")
    return value


def _sha256(value: Any, *, where: str) -> str:
    """要求小写 SHA-256 文本。"""
    text = _text(value, where=where)
    if (len(text) != 64
            or any(char not in "0123456789abcdef" for char in text)):
        raise MemoryDynamicsContractError(f"{where} 必须是小写 SHA-256")
    return text


def _relative_path(value: Any, *, where: str) -> str:
    """要求可迁移的安全 POSIX 相对路径。"""
    text = _text(value, where=where)
    path = PurePosixPath(text)
    if (path.is_absolute() or ".." in path.parts or "\\" in text
            or path.as_posix() != text):
        raise MemoryDynamicsContractError(f"{where} 必须是安全 POSIX 相对路径")
    return text


def _text_tuple(
        value: Any,
        *,
        where: str,
        allow_empty: bool = False,
        ordered: bool = False,
        ) -> tuple[str, ...]:
    """要求文本 tuple 去重，并按声明保持顺序或规范排序。"""
    if not isinstance(value, tuple) or (not value and not allow_empty):
        raise MemoryDynamicsContractError(f"{where} 必须是文本 tuple")
    result = tuple(_text(item, where=where) for item in value)
    if len(result) != len(set(result)):
        raise MemoryDynamicsContractError(f"{where} 不得重复")
    if not ordered and result != tuple(sorted(result)):
        raise MemoryDynamicsContractError(f"{where} 必须排序")
    return result


def _key(value: Any, *, where: str) -> StableRecordKey:
    """要求一等完整整数键。"""
    if not isinstance(value, StableRecordKey):
        raise MemoryDynamicsContractError(f"{where} 必须是 StableRecordKey")
    return value


def _key_tuple(
        value: Any,
        *,
        where: str,
        allow_empty: bool = False,
        ) -> tuple[StableRecordKey, ...]:
    """要求稳定键 tuple 规范排序且不重复。"""
    if not isinstance(value, tuple) or (not value and not allow_empty):
        raise MemoryDynamicsContractError(f"{where} 必须是 StableRecordKey tuple")
    if any(not isinstance(item, StableRecordKey) for item in value):
        raise MemoryDynamicsContractError(f"{where} 含非法稳定键")
    if value != tuple(sorted(set(value))):
        raise MemoryDynamicsContractError(f"{where} 必须排序且去重")
    return value


def _keys_to_value(values: tuple[StableRecordKey, ...]) -> list[list[int]]:
    """把稳定键 tuple 导出为规范 JSON 列表。"""
    return [item.to_list() for item in values]


def _keys_from_value(value: Any, *, where: str) -> tuple[StableRecordKey, ...]:
    """从 JSON 列表恢复稳定键 tuple。"""
    if not isinstance(value, list):
        raise MemoryDynamicsContractError(f"{where} 必须是列表")
    return tuple(StableRecordKey.from_value(item, where=where) for item in value)


@dataclass(frozen=True)
class MemoryDynamicsBoundary:
    """一个 center、receipt 或决断共用的 owner/scope/source/version 边界。"""

    owner_key: StableRecordKey
    scope_key: StableRecordKey
    source_key: StableRecordKey
    version_key: StableRecordKey

    def __post_init__(self) -> None:
        for name in ("owner_key", "scope_key", "source_key", "version_key"):
            _key(getattr(self, name), where=f"boundary {name}")

    def to_dict(self) -> dict[str, Any]:
        """导出规范边界对象。"""
        return {
            "owner_key": self.owner_key.to_list(),
            "scope_key": self.scope_key.to_list(),
            "source_key": self.source_key.to_list(),
            "version_key": self.version_key.to_list(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MemoryDynamicsBoundary":
        """从精确 JSON object 恢复边界。"""
        raw = _exact_keys(value, {
            "owner_key", "scope_key", "source_key", "version_key",
        }, where="MemoryDynamicsBoundary")
        return cls(*(StableRecordKey.from_value(raw[name], where=name)
                     for name in (
                         "owner_key", "scope_key", "source_key", "version_key")))


@dataclass(frozen=True, order=True)
class MemoryCenterOrigin:
    """一个 center 的一等触发来源和完整依赖。"""

    origin_kind: str
    origin_key: StableRecordKey
    dependency_keys: tuple[StableRecordKey, ...]

    def __post_init__(self) -> None:
        _enum(self.origin_kind, ORIGIN_KINDS, where="origin kind")
        _key(self.origin_key, where="origin key")
        _key_tuple(self.dependency_keys, where="origin dependencies")

    def to_dict(self) -> dict[str, Any]:
        """导出规范 origin 对象。"""
        return {
            "dependency_keys": _keys_to_value(self.dependency_keys),
            "origin_key": self.origin_key.to_list(),
            "origin_kind": self.origin_kind,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MemoryCenterOrigin":
        """从精确 JSON object 恢复 origin。"""
        raw = _exact_keys(value, {
            "dependency_keys", "origin_key", "origin_kind",
        }, where="MemoryCenterOrigin")
        return cls(
            str(raw["origin_kind"]),
            StableRecordKey.from_value(raw["origin_key"], where="origin_key"),
            _keys_from_value(raw["dependency_keys"], where="dependency_keys"),
        )


@dataclass(frozen=True)
class MemoryAttentionCenter:
    """由未解决 typed obligation 形成的中心，不由显著性直接授权事实。"""

    center_key: StableRecordKey
    direction: str
    strength: str
    origins: tuple[MemoryCenterOrigin, ...]
    obligation_kind_key: StableRecordKey
    target_key: StableRecordKey
    boundary: MemoryDynamicsBoundary
    relation_profile_key: StableRecordKey
    expansion_profile_key: StableRecordKey
    dependency_keys: tuple[StableRecordKey, ...]
    state: str
    activation_only: int

    def __post_init__(self) -> None:
        _key(self.center_key, where="center key")
        _enum(self.direction, DIRECTIONS, where="center direction")
        _enum(self.strength, CENTER_STRENGTHS, where="center strength")
        if (not isinstance(self.origins, tuple) or not self.origins
                or any(not isinstance(item, MemoryCenterOrigin)
                       for item in self.origins)):
            raise MemoryDynamicsContractError("center origins 类型非法或为空")
        origin_keys = tuple((item.origin_kind, item.origin_key)
                            for item in self.origins)
        if origin_keys != tuple(sorted(set(origin_keys))):
            raise MemoryDynamicsContractError("center origins 必须排序且去重")
        _key(self.obligation_kind_key, where="center obligation kind")
        _key(self.target_key, where="center target")
        if not isinstance(self.boundary, MemoryDynamicsBoundary):
            raise MemoryDynamicsContractError("center boundary 类型非法")
        _key(self.relation_profile_key, where="center relation profile")
        _key(self.expansion_profile_key, where="center expansion profile")
        _key_tuple(self.dependency_keys, where="center dependencies")
        _enum(self.state, CENTER_STATES, where="center state")
        if _flag(self.activation_only, where="center activation_only") != 1:
            raise MemoryDynamicsContractError("center 召回只能形成 activation")

    def to_dict(self) -> dict[str, Any]:
        """导出完整中心身份。"""
        return {
            "activation_only": self.activation_only,
            "boundary": self.boundary.to_dict(),
            "center_key": self.center_key.to_list(),
            "dependency_keys": _keys_to_value(self.dependency_keys),
            "direction": self.direction,
            "expansion_profile_key": self.expansion_profile_key.to_list(),
            "obligation_kind_key": self.obligation_kind_key.to_list(),
            "origins": [item.to_dict() for item in self.origins],
            "relation_profile_key": self.relation_profile_key.to_list(),
            "state": self.state,
            "strength": self.strength,
            "target_key": self.target_key.to_list(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MemoryAttentionCenter":
        """从精确 JSON object 恢复中心。"""
        raw = _exact_keys(value, {
            "activation_only", "boundary", "center_key", "dependency_keys",
            "direction", "expansion_profile_key", "obligation_kind_key",
            "origins", "relation_profile_key", "state", "strength",
            "target_key",
        }, where="MemoryAttentionCenter")
        return cls(
            StableRecordKey.from_value(raw["center_key"], where="center_key"),
            str(raw["direction"]),
            str(raw["strength"]),
            tuple(MemoryCenterOrigin.from_dict(item) for item in raw["origins"]),
            StableRecordKey.from_value(
                raw["obligation_kind_key"], where="obligation_kind_key"),
            StableRecordKey.from_value(raw["target_key"], where="target_key"),
            MemoryDynamicsBoundary.from_dict(raw["boundary"]),
            StableRecordKey.from_value(
                raw["relation_profile_key"], where="relation_profile_key"),
            StableRecordKey.from_value(
                raw["expansion_profile_key"], where="expansion_profile_key"),
            _keys_from_value(raw["dependency_keys"], where="dependency_keys"),
            str(raw["state"]),
            raw["activation_only"],
        )


@dataclass(frozen=True)
class MemoryExpansionChannelBudget:
    """一个扩域通道的准入位和八类严格整数资源上限。"""

    channel_key: str
    admission_enabled: int
    max_scanned_objects: int
    max_candidates: int
    max_agenda_entries: int
    max_consumptions: int
    max_page_reads: int
    max_recomputes: int
    max_logic_steps: int
    max_cold_bytes: int

    def __post_init__(self) -> None:
        _enum(self.channel_key, EXPANSION_CHANNELS, where="channel key")
        _flag(self.admission_enabled, where="channel admission")
        for name in BUDGET_FIELDS:
            _nonnegative(getattr(self, name), where=f"channel {name}")
        if self.admission_enabled == 0 and any(
                getattr(self, name) != 0 for name in BUDGET_FIELDS):
            raise MemoryDynamicsContractError("关闭通道的预算必须全零")
        if self.admission_enabled == 1 and self.max_scanned_objects == 0:
            raise MemoryDynamicsContractError("开启通道必须允许扫描对象")

    def to_dict(self) -> dict[str, Any]:
        """导出通道预算。"""
        return {
            "admission_enabled": self.admission_enabled,
            "channel_key": self.channel_key,
            **{name: getattr(self, name) for name in BUDGET_FIELDS},
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MemoryExpansionChannelBudget":
        """从精确 JSON object 恢复通道预算。"""
        raw = _exact_keys(value, {
            "admission_enabled", "channel_key", *BUDGET_FIELDS,
        }, where="MemoryExpansionChannelBudget")
        return cls(
            str(raw["channel_key"]), raw["admission_enabled"],
            *(raw[name] for name in BUDGET_FIELDS),
        )


@dataclass(frozen=True)
class MemoryExpansionProfile:
    """按 obligation 冻结通道拓扑、距离、竞争、授权、veto 和预算。"""

    profile_key: StableRecordKey
    obligation_kind_keys: tuple[StableRecordKey, ...]
    channel_order: tuple[str, ...]
    channel_budgets: tuple[MemoryExpansionChannelBudget, ...]
    allowed_relation_keys: tuple[StableRecordKey, ...]
    distance_dimensions: tuple[str, ...]
    candidate_competition_fields: tuple[str, ...]
    adoption_authorization_fields: tuple[str, ...]
    hard_veto_keys: tuple[str, ...]
    parallel_channels_allowed: int
    physical_read_sharing_allowed: int
    semantic_center_merge_allowed: int
    global_score_can_override_veto: int

    def __post_init__(self) -> None:
        _key(self.profile_key, where="profile key")
        _key_tuple(self.obligation_kind_keys, where="profile obligation kinds")
        _text_tuple(
            self.channel_order, where="profile channel order", ordered=True)
        if set(self.channel_order) != set(EXPANSION_CHANNELS):
            raise MemoryDynamicsContractError("profile 必须显式列出全部扩域通道")
        if (not isinstance(self.channel_budgets, tuple)
                or any(not isinstance(item, MemoryExpansionChannelBudget)
                       for item in self.channel_budgets)):
            raise MemoryDynamicsContractError("profile channel budgets 类型非法")
        budget_order = tuple(item.channel_key for item in self.channel_budgets)
        if budget_order != self.channel_order:
            raise MemoryDynamicsContractError("profile 通道预算必须跟随通道顺序")
        _key_tuple(self.allowed_relation_keys, where="profile allowed relations")
        for name in (
                "distance_dimensions", "candidate_competition_fields",
                "adoption_authorization_fields", "hard_veto_keys"):
            _text_tuple(getattr(self, name), where=f"profile {name}")
        _flag(self.parallel_channels_allowed, where="profile parallel")
        if _flag(
                self.physical_read_sharing_allowed,
                where="profile read sharing") != 1:
            raise MemoryDynamicsContractError("profile 必须允许共享合法物理读取")
        if _flag(
                self.semantic_center_merge_allowed,
                where="profile semantic merge") != 0:
            raise MemoryDynamicsContractError("profile 不得合并不同中心语义身份")
        if _flag(
                self.global_score_can_override_veto,
                where="profile score veto") != 0:
            raise MemoryDynamicsContractError("总分不得抵消硬 veto")

    def to_dict(self) -> dict[str, Any]:
        """导出完整扩域 profile。"""
        return {
            "adoption_authorization_fields": list(
                self.adoption_authorization_fields),
            "allowed_relation_keys": _keys_to_value(self.allowed_relation_keys),
            "candidate_competition_fields": list(
                self.candidate_competition_fields),
            "channel_budgets": [item.to_dict() for item in self.channel_budgets],
            "channel_order": list(self.channel_order),
            "distance_dimensions": list(self.distance_dimensions),
            "global_score_can_override_veto": self.global_score_can_override_veto,
            "hard_veto_keys": list(self.hard_veto_keys),
            "obligation_kind_keys": _keys_to_value(self.obligation_kind_keys),
            "parallel_channels_allowed": self.parallel_channels_allowed,
            "physical_read_sharing_allowed": self.physical_read_sharing_allowed,
            "profile_key": self.profile_key.to_list(),
            "semantic_center_merge_allowed": self.semantic_center_merge_allowed,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MemoryExpansionProfile":
        """从精确 JSON object 恢复扩域 profile。"""
        raw = _exact_keys(value, {
            "adoption_authorization_fields", "allowed_relation_keys",
            "candidate_competition_fields", "channel_budgets",
            "channel_order", "distance_dimensions",
            "global_score_can_override_veto", "hard_veto_keys",
            "obligation_kind_keys", "parallel_channels_allowed",
            "physical_read_sharing_allowed", "profile_key",
            "semantic_center_merge_allowed",
        }, where="MemoryExpansionProfile")
        return cls(
            StableRecordKey.from_value(raw["profile_key"], where="profile_key"),
            _keys_from_value(
                raw["obligation_kind_keys"], where="obligation_kind_keys"),
            tuple(str(item) for item in raw["channel_order"]),
            tuple(MemoryExpansionChannelBudget.from_dict(item)
                  for item in raw["channel_budgets"]),
            _keys_from_value(
                raw["allowed_relation_keys"], where="allowed_relation_keys"),
            tuple(str(item) for item in raw["distance_dimensions"]),
            tuple(str(item) for item in raw["candidate_competition_fields"]),
            tuple(str(item) for item in raw["adoption_authorization_fields"]),
            tuple(str(item) for item in raw["hard_veto_keys"]),
            raw["parallel_channels_allowed"],
            raw["physical_read_sharing_allowed"],
            raw["semantic_center_merge_allowed"],
            raw["global_score_can_override_veto"],
        )


@dataclass(frozen=True)
class MemoryRingReceipt:
    """一次 center 进入一个通道的可回放读取、过滤、消费和去向事实。"""

    receipt_key: StableRecordKey
    center_key: StableRecordKey
    boundary: MemoryDynamicsBoundary
    channel_key: str
    physical_read_key: StableRecordKey
    start_logical_seq: int
    end_logical_seq: int
    query_anchor_keys: tuple[StableRecordKey, ...]
    allowed_relation_keys: tuple[StableRecordKey, ...]
    read_range_keys: tuple[StableRecordKey, ...]
    scanned_object_count: int
    candidate_count: int
    filtered_count: int
    filtered_reason_counts: CanonicalJsonObject
    agenda_candidate_keys: tuple[StableRecordKey, ...]
    score_reason_keys: tuple[str, ...]
    consumed_candidate_keys: tuple[StableRecordKey, ...]
    evidence_keys: tuple[StableRecordKey, ...]
    dependency_keys: tuple[StableRecordKey, ...]
    page_read_count: int
    recompute_count: int
    cold_read_bytes: int
    action: str
    stop_decision_key: StableRecordKey | None
    host_learning_write_count: int

    def __post_init__(self) -> None:
        for name in ("receipt_key", "center_key", "physical_read_key"):
            _key(getattr(self, name), where=f"receipt {name}")
        if not isinstance(self.boundary, MemoryDynamicsBoundary):
            raise MemoryDynamicsContractError("receipt boundary 类型非法")
        _enum(self.channel_key, EXPANSION_CHANNELS, where="receipt channel")
        _nonnegative(self.start_logical_seq, where="receipt start seq")
        _nonnegative(self.end_logical_seq, where="receipt end seq")
        if self.end_logical_seq < self.start_logical_seq:
            raise MemoryDynamicsContractError("receipt 结束序不得早于开始序")
        for name in (
                "query_anchor_keys", "allowed_relation_keys", "read_range_keys",
                "agenda_candidate_keys", "consumed_candidate_keys",
                "evidence_keys", "dependency_keys"):
            _key_tuple(
                getattr(self, name), where=f"receipt {name}",
                allow_empty=name in {"agenda_candidate_keys",
                                     "consumed_candidate_keys", "evidence_keys"})
        for name in (
                "scanned_object_count", "candidate_count", "filtered_count",
                "page_read_count", "recompute_count", "cold_read_bytes"):
            _nonnegative(getattr(self, name), where=f"receipt {name}")
        reasons = self.filtered_reason_counts.to_value()
        if not reasons and self.filtered_count != 0:
            raise MemoryDynamicsContractError("receipt 过滤原因不得缺失")
        if any(not isinstance(key, str) or not key for key in reasons):
            raise MemoryDynamicsContractError("receipt 过滤原因 key 非法")
        if list(reasons) != sorted(reasons):
            raise MemoryDynamicsContractError("receipt 过滤原因必须排序")
        for key, count in reasons.items():
            _nonnegative(count, where=f"receipt filter {key}")
        if sum(reasons.values()) != self.filtered_count:
            raise MemoryDynamicsContractError("receipt 过滤原因计数不闭合")
        if len(self.agenda_candidate_keys) > self.candidate_count:
            raise MemoryDynamicsContractError("receipt agenda 数超过候选数")
        if len(self.consumed_candidate_keys) > len(self.agenda_candidate_keys):
            raise MemoryDynamicsContractError("receipt consumption 数超过 agenda")
        if not set(self.consumed_candidate_keys).issubset(
                set(self.agenda_candidate_keys)):
            raise MemoryDynamicsContractError("receipt consumer 未消费 agenda 候选")
        _text_tuple(
            self.score_reason_keys, where="receipt score reasons",
            allow_empty=not self.agenda_candidate_keys)
        _enum(self.action, RING_ACTIONS, where="receipt action")
        if self.action == "STOP":
            _key(self.stop_decision_key, where="receipt 停止决断")
        elif self.stop_decision_key is not None:
            raise MemoryDynamicsContractError("非 STOP receipt 不得绑定停止决断")
        _zero(self.host_learning_write_count, where="receipt host writes")

    def to_dict(self) -> dict[str, Any]:
        """导出完整 ring receipt。"""
        return {
            "action": self.action,
            "agenda_candidate_keys": _keys_to_value(self.agenda_candidate_keys),
            "allowed_relation_keys": _keys_to_value(self.allowed_relation_keys),
            "boundary": self.boundary.to_dict(),
            "candidate_count": self.candidate_count,
            "center_key": self.center_key.to_list(),
            "channel_key": self.channel_key,
            "cold_read_bytes": self.cold_read_bytes,
            "consumed_candidate_keys": _keys_to_value(
                self.consumed_candidate_keys),
            "dependency_keys": _keys_to_value(self.dependency_keys),
            "end_logical_seq": self.end_logical_seq,
            "evidence_keys": _keys_to_value(self.evidence_keys),
            "filtered_count": self.filtered_count,
            "filtered_reason_counts": self.filtered_reason_counts.to_value(),
            "host_learning_write_count": self.host_learning_write_count,
            "page_read_count": self.page_read_count,
            "physical_read_key": self.physical_read_key.to_list(),
            "query_anchor_keys": _keys_to_value(self.query_anchor_keys),
            "read_range_keys": _keys_to_value(self.read_range_keys),
            "receipt_key": self.receipt_key.to_list(),
            "recompute_count": self.recompute_count,
            "scanned_object_count": self.scanned_object_count,
            "score_reason_keys": list(self.score_reason_keys),
            "start_logical_seq": self.start_logical_seq,
            "stop_decision_key": (
                None if self.stop_decision_key is None
                else self.stop_decision_key.to_list()),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MemoryRingReceipt":
        """从精确 JSON object 恢复 ring receipt。"""
        raw = _exact_keys(value, {
            "action", "agenda_candidate_keys", "allowed_relation_keys",
            "boundary", "candidate_count", "center_key", "channel_key",
            "cold_read_bytes", "consumed_candidate_keys", "dependency_keys",
            "end_logical_seq", "evidence_keys", "filtered_count",
            "filtered_reason_counts", "host_learning_write_count",
            "page_read_count", "physical_read_key", "query_anchor_keys",
            "read_range_keys", "receipt_key", "recompute_count",
            "scanned_object_count", "score_reason_keys", "start_logical_seq",
            "stop_decision_key",
        }, where="MemoryRingReceipt")
        stop_key = raw["stop_decision_key"]
        return cls(
            StableRecordKey.from_value(raw["receipt_key"], where="receipt_key"),
            StableRecordKey.from_value(raw["center_key"], where="center_key"),
            MemoryDynamicsBoundary.from_dict(raw["boundary"]),
            str(raw["channel_key"]),
            StableRecordKey.from_value(
                raw["physical_read_key"], where="physical_read_key"),
            raw["start_logical_seq"], raw["end_logical_seq"],
            _keys_from_value(raw["query_anchor_keys"], where="query_anchor_keys"),
            _keys_from_value(
                raw["allowed_relation_keys"], where="allowed_relation_keys"),
            _keys_from_value(raw["read_range_keys"], where="read_range_keys"),
            raw["scanned_object_count"], raw["candidate_count"],
            raw["filtered_count"],
            CanonicalJsonObject.from_value(raw["filtered_reason_counts"]),
            _keys_from_value(
                raw["agenda_candidate_keys"], where="agenda_candidate_keys"),
            tuple(str(item) for item in raw["score_reason_keys"]),
            _keys_from_value(
                raw["consumed_candidate_keys"],
                where="consumed_candidate_keys"),
            _keys_from_value(raw["evidence_keys"], where="evidence_keys"),
            _keys_from_value(raw["dependency_keys"], where="dependency_keys"),
            raw["page_read_count"], raw["recompute_count"],
            raw["cold_read_bytes"], str(raw["action"]),
            (None if stop_key is None else StableRecordKey.from_value(
                stop_key, where="stop_decision_key")),
            raw["host_learning_write_count"],
        )


@dataclass(frozen=True)
class MemoryDynamicsStopDecision:
    """区分解决、澄清、未知、接地/访问阻断、预算耗尽和替代。"""

    decision_key: StableRecordKey
    center_key: StableRecordKey
    boundary: MemoryDynamicsBoundary
    status: str
    satisfied_obligation_keys: tuple[StableRecordKey, ...]
    unresolved_obligation_keys: tuple[StableRecordKey, ...]
    conflict_keys: tuple[StableRecordKey, ...]
    hard_conflict_check_keys: tuple[StableRecordKey, ...]
    authorization_evidence_keys: tuple[StableRecordKey, ...]
    blocking_keys: tuple[StableRecordKey, ...]
    remaining_channel_keys: tuple[str, ...]
    replacement_center_key: StableRecordKey | None
    budget_exhausted: int
    reason_keys: tuple[str, ...]
    host_learning_write_count: int

    def __post_init__(self) -> None:
        _key(self.decision_key, where="stop decision key")
        _key(self.center_key, where="stop center key")
        if not isinstance(self.boundary, MemoryDynamicsBoundary):
            raise MemoryDynamicsContractError("stop boundary 类型非法")
        _enum(self.status, STOP_STATES, where="stop status")
        for name in (
                "satisfied_obligation_keys", "unresolved_obligation_keys",
                "conflict_keys", "hard_conflict_check_keys",
                "authorization_evidence_keys", "blocking_keys"):
            _key_tuple(
                getattr(self, name), where=f"stop {name}", allow_empty=True)
        _text_tuple(
            self.remaining_channel_keys, where="stop remaining channels",
            allow_empty=True, ordered=True)
        if any(item not in EXPANSION_CHANNELS
               for item in self.remaining_channel_keys):
            raise MemoryDynamicsContractError("stop remaining channel 未登记")
        if self.replacement_center_key is not None:
            _key(self.replacement_center_key, where="stop replacement center")
        _flag(self.budget_exhausted, where="stop budget exhausted")
        _text_tuple(self.reason_keys, where="stop reasons")
        _zero(self.host_learning_write_count, where="stop host writes")
        if (self.status != "BUDGET_EXHAUSTED"
                and self.budget_exhausted != 0):
            raise MemoryDynamicsContractError("只有预算耗尽状态可置预算耗尽位")
        if self.status == "RESOLVED":
            if (not self.satisfied_obligation_keys
                    or self.unresolved_obligation_keys
                    or not self.hard_conflict_check_keys
                    or not self.authorization_evidence_keys
                    or self.blocking_keys or self.remaining_channel_keys
                    or self.replacement_center_key is not None
                    or self.budget_exhausted):
                raise MemoryDynamicsContractError("RESOLVED 充分性证据不完整")
        else:
            if self.satisfied_obligation_keys:
                raise MemoryDynamicsContractError("非 RESOLVED 不得宣称义务已满足")
        if self.status == "CLARIFY" and (
                len(self.conflict_keys) < 2
                or not self.unresolved_obligation_keys):
            raise MemoryDynamicsContractError("CLARIFY 必须保留竞争与未决义务")
        if self.status == "UNKNOWN" and (
                not self.unresolved_obligation_keys
                or self.remaining_channel_keys or self.blocking_keys
                or self.budget_exhausted):
            raise MemoryDynamicsContractError("UNKNOWN 不得冒充阻断或预算耗尽")
        if self.status == "BUDGET_EXHAUSTED" and (
                not self.unresolved_obligation_keys
                or not self.remaining_channel_keys
                or self.budget_exhausted != 1):
            raise MemoryDynamicsContractError(
                "BUDGET_EXHAUSTED 预算耗尽事实不完整")
        if self.status in {"ACCESS_BLOCKED", "GROUNDING_BLOCKED"} and (
                not self.unresolved_obligation_keys or not self.blocking_keys
                or self.budget_exhausted):
            raise MemoryDynamicsContractError("阻断决断必须保留阻断依据")
        if self.status == "SUPERSEDED" and self.replacement_center_key is None:
            raise MemoryDynamicsContractError("SUPERSEDED 必须指向替代中心")
        if self.status != "SUPERSEDED" and self.replacement_center_key is not None:
            raise MemoryDynamicsContractError("只有 SUPERSEDED 可绑定替代中心")

    def to_dict(self) -> dict[str, Any]:
        """导出停止决断。"""
        return {
            "authorization_evidence_keys": _keys_to_value(
                self.authorization_evidence_keys),
            "blocking_keys": _keys_to_value(self.blocking_keys),
            "boundary": self.boundary.to_dict(),
            "budget_exhausted": self.budget_exhausted,
            "center_key": self.center_key.to_list(),
            "conflict_keys": _keys_to_value(self.conflict_keys),
            "decision_key": self.decision_key.to_list(),
            "hard_conflict_check_keys": _keys_to_value(
                self.hard_conflict_check_keys),
            "host_learning_write_count": self.host_learning_write_count,
            "reason_keys": list(self.reason_keys),
            "remaining_channel_keys": list(self.remaining_channel_keys),
            "replacement_center_key": (
                None if self.replacement_center_key is None
                else self.replacement_center_key.to_list()),
            "satisfied_obligation_keys": _keys_to_value(
                self.satisfied_obligation_keys),
            "status": self.status,
            "unresolved_obligation_keys": _keys_to_value(
                self.unresolved_obligation_keys),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MemoryDynamicsStopDecision":
        """从精确 JSON object 恢复停止决断。"""
        raw = _exact_keys(value, {
            "authorization_evidence_keys", "blocking_keys", "boundary",
            "budget_exhausted", "center_key", "conflict_keys",
            "decision_key", "hard_conflict_check_keys",
            "host_learning_write_count", "reason_keys",
            "remaining_channel_keys", "replacement_center_key",
            "satisfied_obligation_keys", "status",
            "unresolved_obligation_keys",
        }, where="MemoryDynamicsStopDecision")
        replacement = raw["replacement_center_key"]
        return cls(
            StableRecordKey.from_value(raw["decision_key"], where="decision_key"),
            StableRecordKey.from_value(raw["center_key"], where="center_key"),
            MemoryDynamicsBoundary.from_dict(raw["boundary"]),
            str(raw["status"]),
            _keys_from_value(
                raw["satisfied_obligation_keys"],
                where="satisfied_obligation_keys"),
            _keys_from_value(
                raw["unresolved_obligation_keys"],
                where="unresolved_obligation_keys"),
            _keys_from_value(raw["conflict_keys"], where="conflict_keys"),
            _keys_from_value(
                raw["hard_conflict_check_keys"],
                where="hard_conflict_check_keys"),
            _keys_from_value(
                raw["authorization_evidence_keys"],
                where="authorization_evidence_keys"),
            _keys_from_value(raw["blocking_keys"], where="blocking_keys"),
            tuple(str(item) for item in raw["remaining_channel_keys"]),
            (None if replacement is None else StableRecordKey.from_value(
                replacement, where="replacement_center_key")),
            raw["budget_exhausted"],
            tuple(str(item) for item in raw["reason_keys"]),
            raw["host_learning_write_count"],
        )


@dataclass(frozen=True)
class MemoryDynamicsRunReport:
    """一个 MD strategy 运行的引用、整数指标、硬不变量和诚实决断。"""

    format_version: int
    report_version: str
    run_key: StableRecordKey
    preregistration_version: str
    strategy_key: str
    run_status: str
    center_keys: tuple[StableRecordKey, ...]
    profile_keys: tuple[StableRecordKey, ...]
    receipt_keys: tuple[StableRecordKey, ...]
    stop_decision_keys: tuple[StableRecordKey, ...]
    metric_values: CanonicalJsonObject
    hard_invariant_failures: tuple[str, ...]
    results_observed: int
    probe_decision: str
    execution_state: CanonicalJsonObject

    def __post_init__(self) -> None:
        if self.format_version != FORMAT_VERSION:
            raise MemoryDynamicsContractError("run report format_version 非法")
        _text(self.report_version, where="run report version")
        _key(self.run_key, where="run key")
        _text(self.preregistration_version, where="run preregistration version")
        _enum(self.strategy_key, MD_BASELINE_KEYS, where="run strategy")
        _enum(self.run_status, RUN_STATUSES, where="run status")
        for name in (
                "center_keys", "profile_keys", "receipt_keys",
                "stop_decision_keys"):
            _key_tuple(
                getattr(self, name), where=f"run {name}", allow_empty=True)
        metrics = self.metric_values.to_value()
        if tuple(metrics) != MD_METRIC_KEYS:
            raise MemoryDynamicsContractError("run metrics 未精确列出预注册维度")
        for key, value in metrics.items():
            _nonnegative(value, where=f"run metric {key}")
        _text_tuple(
            self.hard_invariant_failures, where="run hard invariant failures",
            allow_empty=True)
        _flag(self.results_observed, where="run results observed")
        _enum(self.probe_decision, PROBE_DECISIONS, where="run probe decision")
        state = self.execution_state.to_value()
        if tuple(state) != EXECUTION_STATE_KEYS:
            raise MemoryDynamicsContractError("run execution state 字段不精确")
        for key, value in state.items():
            _zero(value, where=f"run execution {key}")
        if self.run_status == "NOT_STARTED":
            if (self.results_observed or self.probe_decision != "NOT_EVALUATED"
                    or any((self.center_keys, self.profile_keys,
                            self.receipt_keys, self.stop_decision_keys))
                    or any(metrics.values()) or self.hard_invariant_failures):
                raise MemoryDynamicsContractError("未运行报告不得携带结果")
        else:
            if (self.results_observed != 1
                    or not self.center_keys or not self.profile_keys
                    or not self.receipt_keys or not self.stop_decision_keys
                    or self.probe_decision == "NOT_EVALUATED"):
                raise MemoryDynamicsContractError("完成报告缺运行引用或决断")
        if self.probe_decision == "PASS" and self.hard_invariant_failures:
            raise MemoryDynamicsContractError("硬不变量失败时不得 PASS")

    def to_dict(self) -> dict[str, Any]:
        """导出规范 run report。"""
        return {
            "artifact_kind": "PH2_MEMORY_DYNAMICS_RUN_REPORT",
            "center_keys": _keys_to_value(self.center_keys),
            "execution_state": self.execution_state.to_value(),
            "format_version": self.format_version,
            "hard_invariant_failures": list(self.hard_invariant_failures),
            "metric_values": self.metric_values.to_value(),
            "preregistration_version": self.preregistration_version,
            "probe_decision": self.probe_decision,
            "profile_keys": _keys_to_value(self.profile_keys),
            "receipt_keys": _keys_to_value(self.receipt_keys),
            "report_version": self.report_version,
            "results_observed": self.results_observed,
            "run_key": self.run_key.to_list(),
            "run_status": self.run_status,
            "stop_decision_keys": _keys_to_value(self.stop_decision_keys),
            "strategy_key": self.strategy_key,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MemoryDynamicsRunReport":
        """从精确 JSON object 恢复 run report。"""
        raw = _exact_keys(value, {
            "artifact_kind", "center_keys", "execution_state",
            "format_version", "hard_invariant_failures", "metric_values",
            "preregistration_version", "probe_decision", "profile_keys",
            "receipt_keys", "report_version", "results_observed", "run_key",
            "run_status", "stop_decision_keys", "strategy_key",
        }, where="MemoryDynamicsRunReport")
        if raw["artifact_kind"] != "PH2_MEMORY_DYNAMICS_RUN_REPORT":
            raise MemoryDynamicsContractError("run report artifact_kind 非法")
        return cls(
            raw["format_version"], str(raw["report_version"]),
            StableRecordKey.from_value(raw["run_key"], where="run_key"),
            str(raw["preregistration_version"]), str(raw["strategy_key"]),
            str(raw["run_status"]),
            _keys_from_value(raw["center_keys"], where="center_keys"),
            _keys_from_value(raw["profile_keys"], where="profile_keys"),
            _keys_from_value(raw["receipt_keys"], where="receipt_keys"),
            _keys_from_value(
                raw["stop_decision_keys"], where="stop_decision_keys"),
            CanonicalJsonObject.from_value(raw["metric_values"]),
            tuple(str(item) for item in raw["hard_invariant_failures"]),
            raw["results_observed"], str(raw["probe_decision"]),
            CanonicalJsonObject.from_value(raw["execution_state"]),
        )

    def canonical_bytes(self) -> bytes:
        """返回规范单行 JSON 字节。"""
        return canonical_json_line(self.to_dict())


@dataclass(frozen=True)
class MD01ContractManifest:
    """冻结 MD-01 合同集合、复用边界、verifier 盲区和零执行事实。"""

    format_version: int
    artifact_version: str
    artifact_status: str
    task_keys: tuple[str, ...]
    md00_preregistration_version: str
    prerequisite_manifest_relative_path: str
    prerequisite_manifest_sha256: str
    contract_type_keys: tuple[str, ...]
    direction_keys: tuple[str, ...]
    strength_keys: tuple[str, ...]
    channel_keys: tuple[str, ...]
    stop_state_keys: tuple[str, ...]
    invariant_keys: tuple[str, ...]
    reused_component_refs: tuple[str, ...]
    verifier_dimensions: tuple[str, ...]
    verifier_ne_conditions: tuple[str, ...]
    runtime_status: str
    results_observed: int
    execution_state: CanonicalJsonObject

    def __post_init__(self) -> None:
        if self.format_version != FORMAT_VERSION:
            raise MemoryDynamicsContractError("MD-01 format_version 非法")
        _text(self.artifact_version, where="MD-01 artifact version")
        if self.artifact_status != "CONTRACT_FROZEN":
            raise MemoryDynamicsContractError("MD-01 artifact status 非法")
        if self.task_keys != ("MD-01",):
            raise MemoryDynamicsContractError("MD-01 task keys 非法")
        _text(self.md00_preregistration_version, where="MD-00 version")
        _relative_path(
            self.prerequisite_manifest_relative_path,
            where="MD-01 prerequisite path")
        _sha256(
            self.prerequisite_manifest_sha256,
            where="MD-01 prerequisite hash")
        expected_pairs = (
            (self.contract_type_keys, MD01_CONTRACT_KEYS, "contract types"),
            (self.direction_keys, DIRECTIONS, "directions"),
            (self.strength_keys, CENTER_STRENGTHS, "strengths"),
            (self.channel_keys, EXPANSION_CHANNELS, "channels"),
            (self.stop_state_keys, STOP_STATES, "stop states"),
            (self.invariant_keys, MD01_INVARIANT_KEYS, "invariants"),
            (self.verifier_dimensions, MD01_VERIFIER_DIMENSIONS,
             "verifier dimensions"),
            (self.verifier_ne_conditions, MD01_NE_CONDITIONS,
             "verifier NE conditions"),
        )
        for actual, expected, label in expected_pairs:
            if actual != expected:
                raise MemoryDynamicsContractError(f"MD-01 {label} 未列全")
        _text_tuple(
            self.reused_component_refs, where="MD-01 reused refs")
        if self.runtime_status != "NOT_STARTED":
            raise MemoryDynamicsContractError("MD-01 不得冒充 runtime 已接通")
        _zero(self.results_observed, where="MD-01 results observed")
        state = self.execution_state.to_value()
        if tuple(state) != EXECUTION_STATE_KEYS:
            raise MemoryDynamicsContractError("MD-01 execution state 字段不精确")
        for key, value in state.items():
            _zero(value, where=f"MD-01 execution {key}")

    def to_dict(self) -> dict[str, Any]:
        """导出规范 MD-01 manifest。"""
        return {
            "artifact_kind": "PH2_MD01_CONTRACT_FREEZE",
            "artifact_status": self.artifact_status,
            "artifact_version": self.artifact_version,
            "channel_keys": list(self.channel_keys),
            "contract_type_keys": list(self.contract_type_keys),
            "direction_keys": list(self.direction_keys),
            "execution_state": self.execution_state.to_value(),
            "format_version": self.format_version,
            "invariant_keys": list(self.invariant_keys),
            "md00_preregistration_version": self.md00_preregistration_version,
            "prerequisite_manifest_relative_path": (
                self.prerequisite_manifest_relative_path),
            "prerequisite_manifest_sha256": self.prerequisite_manifest_sha256,
            "results_observed": self.results_observed,
            "reused_component_refs": list(self.reused_component_refs),
            "runtime_status": self.runtime_status,
            "stop_state_keys": list(self.stop_state_keys),
            "strength_keys": list(self.strength_keys),
            "task_keys": list(self.task_keys),
            "verifier_dimensions": list(self.verifier_dimensions),
            "verifier_ne_conditions": list(self.verifier_ne_conditions),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MD01ContractManifest":
        """从精确 JSON object 恢复 MD-01 manifest。"""
        raw = _exact_keys(value, {
            "artifact_kind", "artifact_status", "artifact_version",
            "channel_keys", "contract_type_keys", "direction_keys",
            "execution_state", "format_version", "invariant_keys",
            "md00_preregistration_version",
            "prerequisite_manifest_relative_path",
            "prerequisite_manifest_sha256", "results_observed",
            "reused_component_refs", "runtime_status", "stop_state_keys",
            "strength_keys", "task_keys", "verifier_dimensions",
            "verifier_ne_conditions",
        }, where="MD01ContractManifest")
        if raw["artifact_kind"] != "PH2_MD01_CONTRACT_FREEZE":
            raise MemoryDynamicsContractError("MD-01 artifact_kind 非法")
        return cls(
            raw["format_version"], str(raw["artifact_version"]),
            str(raw["artifact_status"]),
            tuple(str(item) for item in raw["task_keys"]),
            str(raw["md00_preregistration_version"]),
            str(raw["prerequisite_manifest_relative_path"]),
            str(raw["prerequisite_manifest_sha256"]),
            tuple(str(item) for item in raw["contract_type_keys"]),
            tuple(str(item) for item in raw["direction_keys"]),
            tuple(str(item) for item in raw["strength_keys"]),
            tuple(str(item) for item in raw["channel_keys"]),
            tuple(str(item) for item in raw["stop_state_keys"]),
            tuple(str(item) for item in raw["invariant_keys"]),
            tuple(str(item) for item in raw["reused_component_refs"]),
            tuple(str(item) for item in raw["verifier_dimensions"]),
            tuple(str(item) for item in raw["verifier_ne_conditions"]),
            str(raw["runtime_status"]), raw["results_observed"],
            CanonicalJsonObject.from_value(raw["execution_state"]),
        )

    def canonical_bytes(self) -> bytes:
        """返回规范单行 JSON 字节。"""
        return canonical_json_line(self.to_dict())


def zero_execution_state() -> CanonicalJsonObject:
    """建立 D-03/W-01/training/teacher/学习宿主均为零的冻结状态。"""
    return CanonicalJsonObject.from_value({key: 0 for key in EXECUTION_STATE_KEYS})


def build_md01_contract_manifest(
        *,
        prerequisite_manifest_relative_path: str,
        prerequisite_manifest_sha256: str,
        ) -> MD01ContractManifest:
    """绑定 MD-00 预注册并冻结 MD-01 的五类纯合同。"""
    return MD01ContractManifest(
        FORMAT_VERSION,
        "MD-01-memory-dynamics-contract-v1",
        "CONTRACT_FROZEN",
        ("MD-01",),
        "MD-00-center-expansion-preregistration-v1",
        prerequisite_manifest_relative_path,
        prerequisite_manifest_sha256,
        MD01_CONTRACT_KEYS,
        DIRECTIONS,
        CENTER_STRENGTHS,
        EXPANSION_CHANNELS,
        STOP_STATES,
        MD01_INVARIANT_KEYS,
        tuple(sorted((
            "src/pure_integer_ai/cognition/shared/attractor_state.py",
            "src/pure_integer_ai/cognition/shared/memory_query.py",
            "src/pure_integer_ai/cognition/shared/reasoning_planner.py",
            "src/pure_integer_ai/cognition/shared/work_memory.py",
            "src/pure_integer_ai/experiments/attractor_runtime.py",
            "src/pure_integer_ai/experiments/memory_hot_set_runtime.py",
            "src/pure_integer_ai/experiments/memory_query_runtime.py",
        ))),
        MD01_VERIFIER_DIMENSIONS,
        MD01_NE_CONDITIONS,
        "NOT_STARTED",
        0,
        zero_execution_state(),
    )


def write_md01_contract_manifest(
        manifest: MD01ContractManifest,
        path: str | Path,
        ) -> Path:
    """独占或逐字节幂等发布 MD-01 manifest，禁止原地改版。"""
    if not isinstance(manifest, MD01ContractManifest):
        raise MemoryDynamicsContractError("MD-01 manifest 类型错误")
    target = Path(path)
    payload = manifest.canonical_bytes()
    if target.exists():
        if not target.is_file() or target.read_bytes() != payload:
            raise MemoryDynamicsContractError("MD-01 manifest 已存在且内容不同")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(payload)
    except OSError as error:
        raise MemoryDynamicsContractError("MD-01 manifest 无法发布") from error
    return target


def read_md01_contract_manifest(path: str | Path) -> MD01ContractManifest:
    """严格回读规范 MD-01 manifest。"""
    try:
        payload = Path(path).read_bytes()
        if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
            raise MemoryDynamicsContractError("MD-01 manifest 换行非法")
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
        assert isinstance(value, dict)
        manifest = MD01ContractManifest.from_dict(value)
    except MemoryDynamicsContractError:
        raise
    except Exception as error:
        raise MemoryDynamicsContractError("MD-01 manifest 损坏") from error
    if manifest.canonical_bytes() != payload:
        raise MemoryDynamicsContractError("MD-01 manifest 非规范字节")
    return manifest


__all__ = [
    "CENTER_STATES",
    "CENTER_STRENGTHS",
    "DIRECTIONS",
    "EXPANSION_CHANNELS",
    "MD01ContractManifest",
    "MD01_CONTRACT_KEYS",
    "MD01_INVARIANT_KEYS",
    "MD01_NE_CONDITIONS",
    "MD01_VERIFIER_DIMENSIONS",
    "MD_METRIC_KEYS",
    "MemoryAttentionCenter",
    "MemoryCenterOrigin",
    "MemoryDynamicsBoundary",
    "MemoryDynamicsContractError",
    "MemoryDynamicsRunReport",
    "MemoryDynamicsStopDecision",
    "MemoryExpansionChannelBudget",
    "MemoryExpansionProfile",
    "MemoryRingReceipt",
    "ORIGIN_KINDS",
    "STOP_STATES",
    "build_md01_contract_manifest",
    "read_md01_contract_manifest",
    "write_md01_contract_manifest",
    "zero_execution_state",
]
