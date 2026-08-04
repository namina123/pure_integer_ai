"""W08-05 有界长上下文请求、轨迹、资源与消融合同。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.experiments.ph2_dataset_contract import StableRecordKey
from pure_integer_ai.experiments.ph2_w05_contract import digest_value
from pure_integer_ai.experiments.ph2_w08_authority import W08_DIMENSION_KEYS
from pure_integer_ai.experiments.ph2_w08_contract import (
    W08_ALLOWED_MODES,
    W08_ALLOWED_WORKER_COUNTS,
    W08_CONSUMER_KEYS,
    W08_RESOURCE_BUDGET,
    W08_STOP_STATES,
)


W08_LONG_CONTEXT_COMPONENT_KEYS = (
    "HIERARCHY",
    "PERSISTENT_AGENDA",
    "COLD_PAGE_IN",
    "GENERATION_CHECKPOINT",
)
W08_LONG_CONTEXT_OWNER_KEYS = (
    "R-06-LONG-INPUT-HIERARCHY",
    "R-06-PERSISTENT-CONVERSATION-AGENDA",
    "R-04-AUTHORIZED-CENTER-K04-PAGE-IN",
    "R-06-LONG-GENERATION-CHECKPOINT",
    "PH2-W08-TYPED-CONSUMERS",
)
W08_LONG_CONTEXT_CASE_KEYS = (
    "MULTI_CENTER",
    "NEAR_DISTRACTOR",
    "DISTANT_COLD_RELEVANT",
    "SOURCE_CONFLICT",
    "UNKNOWN",
    "CLARIFY",
    "BUDGET_EXHAUSTED",
)


class W08LongContextError(ValueError):
    """W08-05 长上下文身份、资源或 owner 边界发生漂移。"""


def _key(value: object, *, where: str) -> tuple[int, ...]:
    if (
        not isinstance(value, tuple)
        or not value
        or any(type(item) is not int or item < 0 for item in value)
    ):
        raise W08LongContextError(f"{where} must be a non-empty integer tuple")
    return value


def _stable_keys(
    value: object, *, where: str, empty: bool = False
) -> tuple[StableRecordKey, ...]:
    if not isinstance(value, tuple) or (not value and not empty):
        raise W08LongContextError(f"{where} must be a tuple")
    if any(not isinstance(item, StableRecordKey) for item in value):
        raise TypeError(f"{where} contains a non-StableRecordKey")
    if value != tuple(sorted(set(value))):
        raise W08LongContextError(f"{where} must be canonical")
    return value


@dataclass(frozen=True)
class W08LongContextRequest:
    """不持有 payload 或宿主写权限的 W08 可见长上下文运行请求。"""

    request_key: tuple[int, ...]
    training_material_key: tuple[int, ...]
    worker_count: int
    mode: str
    resource_budget: tuple[tuple[str, int], ...]
    required_center_keys: tuple[StableRecordKey, ...]
    distractor_center_keys: tuple[StableRecordKey, ...] = ()
    conflict_center_keys: tuple[StableRecordKey, ...] = ()
    resolved_conflict_keys: tuple[StableRecordKey, ...] = ()
    clarification_candidate_keys: tuple[StableRecordKey, ...] = ()
    clarification_resolution_key: StableRecordKey | None = None
    component_flags: tuple[tuple[str, int], ...] = tuple(
        (key, 1) for key in W08_LONG_CONTEXT_COMPONENT_KEYS
    )
    page_limit: int = 2048
    logical_seq: int = 1

    def __post_init__(self) -> None:
        _key(self.request_key, where="long-context request")
        _key(self.training_material_key, where="long-context training material")
        if self.worker_count not in W08_ALLOWED_WORKER_COUNTS:
            raise W08LongContextError("long-context worker count is not registered")
        if self.mode not in W08_ALLOWED_MODES:
            raise W08LongContextError("long-context mode is not registered")
        if self.resource_budget != tuple(sorted(W08_RESOURCE_BUDGET.items())):
            raise W08LongContextError("long-context resource budget drifted")
        required = _stable_keys(
            self.required_center_keys,
            where="required centers",
            empty=True,
        )
        distractors = _stable_keys(
            self.distractor_center_keys,
            where="distractor centers",
            empty=True,
        )
        conflicts = _stable_keys(
            self.conflict_center_keys,
            where="conflict centers",
            empty=True,
        )
        resolved = _stable_keys(
            self.resolved_conflict_keys,
            where="resolved conflicts",
            empty=True,
        )
        clarifications = _stable_keys(
            self.clarification_candidate_keys,
            where="clarification candidates",
            empty=True,
        )
        if set(required) & set(distractors):
            raise W08LongContextError("required and distractor centers overlap")
        if not set(conflicts).issubset(required):
            raise W08LongContextError("conflicts must name required centers")
        if not set(resolved).issubset(conflicts):
            raise W08LongContextError("resolved conflicts are not registered conflicts")
        if self.clarification_resolution_key is not None:
            if not isinstance(self.clarification_resolution_key, StableRecordKey):
                raise TypeError("clarification resolution key type is invalid")
            if self.clarification_resolution_key not in clarifications:
                raise W08LongContextError("clarification selected outside candidates")
        if tuple(key for key, _ in self.component_flags) != W08_LONG_CONTEXT_COMPONENT_KEYS:
            raise W08LongContextError("long-context component inventory drifted")
        if any(value not in (0, 1) for _, value in self.component_flags):
            raise W08LongContextError("long-context component flag is not a bit")
        if type(self.page_limit) is not int or self.page_limit <= 0:
            raise W08LongContextError("long-context page limit must be positive")
        if self.page_limit >= W08_RESOURCE_BUDGET["max_checkpoint_count"]:
            raise W08LongContextError("long-context page limit exceeds checkpoint budget")
        if type(self.logical_seq) is not int or self.logical_seq <= 0:
            raise W08LongContextError("long-context logical sequence must be positive")

    def component_enabled(self, key: str) -> bool:
        try:
            return bool(dict(self.component_flags)[key])
        except KeyError as error:
            raise W08LongContextError("unknown long-context component") from error

    def scheduling_key(self) -> tuple[int, ...]:
        return digest_value({
            "mode": self.mode,
            "page_limit": self.page_limit,
            "request_key": list(self.request_key),
            "worker_count": self.worker_count,
        })


@dataclass(frozen=True, order=True)
class W08LongContextUse:
    """同一授权长上下文结果的一次 U/R/G 消费记录。"""

    consumer_key: str
    request_key: tuple[int, ...]
    selected_center_keys: tuple[StableRecordKey, ...]
    evidence_keys: tuple[StableRecordKey, ...]
    directional_choice_key: tuple[int, ...]
    use_key: tuple[int, ...]
    outcome_state: str

    def __post_init__(self) -> None:
        if self.consumer_key not in W08_CONSUMER_KEYS:
            raise W08LongContextError("long-context consumer is not registered")
        _key(self.request_key, where="long-context Use request")
        _stable_keys(self.selected_center_keys, where="long-context selected centers")
        _stable_keys(self.evidence_keys, where="long-context Use evidence")
        _key(self.directional_choice_key, where="long-context directional choice")
        _key(self.use_key, where="long-context Use")
        if self.outcome_state not in W08_STOP_STATES:
            raise W08LongContextError("long-context Use outcome is invalid")


@dataclass(frozen=True)
class W08LongContextResourceReceipt:
    """一次有界运行中实测的物理与逻辑工作收据。"""

    opened_segments: int
    opened_pages: int
    page_in_records: int
    payload_gets: int
    payload_bytes: int
    agenda_entries: int
    real_consumers: int
    recompute_objects: int
    logic_operations: int
    checkpoint_count: int
    stop_reason: str

    def __post_init__(self) -> None:
        values = tuple(
            getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "stop_reason"
        )
        if any(type(value) is not int or value < 0 for value in values):
            raise W08LongContextError("long-context resource count is invalid")
        if self.stop_reason not in W08_STOP_STATES:
            raise W08LongContextError("long-context stop reason is invalid")
        limits = W08_RESOURCE_BUDGET
        if (
            self.opened_segments > limits["max_segments"]
            or self.opened_pages > limits["max_payload_gets"]
            or self.payload_gets > limits["max_payload_gets"]
            or self.payload_bytes > limits["max_payload_bytes"]
            or self.page_in_records > limits["max_records"]
            or self.agenda_entries > limits["max_records"]
            or self.real_consumers > len(W08_CONSUMER_KEYS)
            or self.recompute_objects > limits["max_recompute_objects"]
            or self.logic_operations > limits["max_logic_operations"]
            or self.checkpoint_count > limits["max_checkpoint_count"]
        ):
            raise W08LongContextError("long-context resource budget was exceeded")


@dataclass(frozen=True)
class W08LongContextTrace:
    """层级、agenda、page-in、引用与 checkpoint 的规范证据。"""

    hierarchy_key: tuple[int, ...]
    prefix_content_digests: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    center_keys: tuple[StableRecordKey, ...]
    authorization_receipt_keys: tuple[StableRecordKey, ...]
    citation_record_keys: tuple[StableRecordKey, ...]
    agenda_digest: tuple[int, ...]
    checkpoint_digest: tuple[int, ...]
    checkpoint_prefix_digest: tuple[int, ...]
    checkpoint_cursor: int

    def __post_init__(self) -> None:
        _key(self.hierarchy_key, where="long-context hierarchy")
        if not self.prefix_content_digests:
            raise W08LongContextError("long-context hierarchy digest inventory is empty")
        for prefix, content in self.prefix_content_digests:
            if len(prefix) != 32 or len(content) != 32:
                raise W08LongContextError("long-context content digest is invalid")
        _stable_keys(self.center_keys, where="long-context trace centers")
        _stable_keys(
            self.authorization_receipt_keys,
            where="long-context authorization receipts",
        )
        _stable_keys(
            self.citation_record_keys,
            where="long-context citations",
        )
        for value, where in (
            (self.agenda_digest, "agenda digest"),
            (self.checkpoint_digest, "checkpoint digest"),
            (self.checkpoint_prefix_digest, "checkpoint prefix digest"),
        ):
            if len(value) != 32 or any(type(item) is not int for item in value):
                raise W08LongContextError(f"{where} is invalid")
        if type(self.checkpoint_cursor) is not int or self.checkpoint_cursor < 0:
            raise W08LongContextError("checkpoint cursor is invalid")

    def canonical_key(self) -> tuple[int, ...]:
        return digest_value({
            "agenda_digest": list(self.agenda_digest),
            "authorization_receipts": [
                list(item.components) for item in self.authorization_receipt_keys
            ],
            "center_keys": [list(item.components) for item in self.center_keys],
            "checkpoint_cursor": self.checkpoint_cursor,
            "checkpoint_digest": list(self.checkpoint_digest),
            "checkpoint_prefix_digest": list(self.checkpoint_prefix_digest),
            "citations": [list(item.components) for item in self.citation_record_keys],
            "hierarchy_key": list(self.hierarchy_key),
            "prefix_content_digests": [
                [list(prefix), list(content)]
                for prefix, content in self.prefix_content_digests
            ],
        })

    def outcome_key(self) -> tuple[int, ...]:
        """返回不含重启路径相关 metadata revision 的结果身份。"""
        return digest_value({
            "authorization_receipts": [
                list(item.components) for item in self.authorization_receipt_keys
            ],
            "center_keys": [list(item.components) for item in self.center_keys],
            "checkpoint_cursor": self.checkpoint_cursor,
            "checkpoint_prefix_digest": list(self.checkpoint_prefix_digest),
            "citations": [list(item.components) for item in self.citation_record_keys],
            "hierarchy_key": list(self.hierarchy_key),
            "prefix_content_digests": [
                [list(prefix), list(content)]
                for prefix, content in self.prefix_content_digests
            ],
        })


@dataclass(frozen=True)
class W08LongContextAuditReceipt:
    request_key: tuple[int, ...]
    state: str
    trace: W08LongContextTrace | None
    resources: W08LongContextResourceReceipt
    uses: tuple[W08LongContextUse, ...]
    owner_calls: tuple[str, ...]
    blocked_component: str = ""
    host_write_count: int = 0
    private_label_read_count: int = 0
    memory_learning_write_count: int = 0

    def __post_init__(self) -> None:
        _key(self.request_key, where="long-context audit request")
        if self.state not in W08_STOP_STATES or self.resources.stop_reason != self.state:
            raise W08LongContextError("long-context audit state drifted")
        if self.owner_calls != tuple(
            key for key in W08_LONG_CONTEXT_OWNER_KEYS if key in self.owner_calls
        ):
            raise W08LongContextError("long-context owner call order drifted")
        if any((
            self.host_write_count,
            self.private_label_read_count,
            self.memory_learning_write_count,
        )):
            raise W08LongContextError("long-context run crossed a write/private boundary")
        if self.state == "RESOLVED":
            if not isinstance(self.trace, W08LongContextTrace):
                raise TypeError("resolved long-context audit lacks a trace")
            if tuple(item.consumer_key for item in self.uses) != W08_CONSUMER_KEYS:
                raise W08LongContextError("resolved long-context audit lacks U/R/G")
            if self.resources.real_consumers != len(W08_CONSUMER_KEYS):
                raise W08LongContextError("real consumer count drifted")
            if self.blocked_component:
                raise W08LongContextError("resolved audit names a blocked component")
        elif self.trace is not None and self.blocked_component:
            raise W08LongContextError("component ablation exposed a full trace")

    def canonical_key(self) -> tuple[int, ...]:
        return digest_value({
            "request_key": list(self.request_key),
            "state": self.state,
            "trace": None if self.trace is None else list(self.trace.outcome_key()),
            "uses": [
                {
                    "consumer": item.consumer_key,
                    "direction": list(item.directional_choice_key),
                    "evidence": [list(key.components) for key in item.evidence_keys],
                    "outcome": item.outcome_state,
                    "selected": [
                        list(key.components) for key in item.selected_center_keys
                    ],
                    "use": list(item.use_key),
                }
                for item in self.uses
            ],
        })


@dataclass(frozen=True)
class W08LongContextAblationReport:
    internal_component: str
    affected_dimensions: tuple[str, ...]
    unaffected_dimensions: tuple[str, ...]


def assess_w08_long_context_ablation(
    *,
    internal_component: str,
    full_dimension_outcomes: dict[str, str],
    ablated_dimension_outcomes: dict[str, str],
) -> W08LongContextAblationReport:
    if internal_component not in W08_LONG_CONTEXT_COMPONENT_KEYS:
        raise W08LongContextError("long-context internal ablation is not registered")
    expected = set(W08_DIMENSION_KEYS)
    if set(full_dimension_outcomes) != expected or set(ablated_dimension_outcomes) != expected:
        raise W08LongContextError("long-context ablation inventory drifted")
    target = "W-08-LONG_CONTEXT"
    changed = tuple(
        key
        for key in W08_DIMENSION_KEYS
        if full_dimension_outcomes[key] != ablated_dimension_outcomes[key]
    )
    if (
        full_dimension_outcomes[target] != "PASS"
        or changed != (target,)
        or ablated_dimension_outcomes[target] == "PASS"
    ):
        raise W08LongContextError("long-context ablation is not orthogonal")
    return W08LongContextAblationReport(
        internal_component,
        changed,
        tuple(key for key in W08_DIMENSION_KEYS if key != target),
    )


__all__ = [
    "W08LongContextAblationReport",
    "W08LongContextAuditReceipt",
    "W08LongContextError",
    "W08LongContextRequest",
    "W08LongContextResourceReceipt",
    "W08LongContextTrace",
    "W08LongContextUse",
    "W08_LONG_CONTEXT_CASE_KEYS",
    "W08_LONG_CONTEXT_COMPONENT_KEYS",
    "W08_LONG_CONTEXT_OWNER_KEYS",
    "assess_w08_long_context_ablation",
]
