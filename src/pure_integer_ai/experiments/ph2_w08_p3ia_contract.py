"""公开 W08 P3-Ia 整合合同与消融记账。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.experiments.ph2_dataset_contract import StableRecordKey
from pure_integer_ai.experiments.ph2_w05_contract import digest_value
from pure_integer_ai.experiments.ph2_w08_authority import W08_DIMENSION_KEYS
from pure_integer_ai.experiments.ph2_w08_contract import (
    W08_CONSUMER_KEYS,
    W08_RESOURCE_BUDGET,
    W08_STOP_STATES,
)


W08_P3IA_COMPONENT_KEYS = (
    "HIERARCHY_FORMER",
    "CENTER_FORMER",
    "PARAPHRASE_EVIDENCE",
    "COLD_PAGE_IN",
)
W08_P3IA_OWNER_KEYS = (
    "P3IA_MECHANICAL_HIERARCHY_OWNER",
    "P3IA_LEARNED_CENTER_EXACT_RECALL_OWNER",
    "P3IA_RECALLED_QA_OWNER",
    "P3IA_TYPED_GENERATION_OWNER",
    "P3IA_LOCAL_REVISION_OWNER",
)


class W08P3IaError(ValueError):
    """P3-Ia 整合身份、runtime 或消融合同发生漂移。"""


def _key(value: object, *, where: str) -> tuple[int, ...]:
    if not isinstance(value, tuple) or not value or any(
        type(item) is not int for item in value
    ):
        raise W08P3IaError(f"{where} is not a strict integer key")
    return value


def _stable_keys(
    values: object, *, where: str, allow_empty: bool = False
) -> tuple[StableRecordKey, ...]:
    if not isinstance(values, tuple) or any(
        not isinstance(item, StableRecordKey) for item in values
    ):
        raise TypeError(f"{where} is not a StableRecordKey tuple")
    if not allow_empty and not values:
        raise W08P3IaError(f"{where} is empty")
    if values != tuple(sorted(set(values))):
        raise W08P3IaError(f"{where} is not canonical")
    return values


@dataclass(frozen=True)
class W08P3IaRequest:
    request_key: tuple[int, ...]
    case_key: StableRecordKey
    enabled_components: tuple[str, ...]
    reader_key: tuple[int, ...]
    logical_seq: int

    def __post_init__(self) -> None:
        _key(self.request_key, where="P3-Ia request")
        if not isinstance(self.case_key, StableRecordKey):
            raise TypeError("P3-Ia case key type is invalid")
        if (
            not isinstance(self.enabled_components, tuple)
            or any(item not in W08_P3IA_COMPONENT_KEYS for item in self.enabled_components)
            or self.enabled_components
            != tuple(
                item
                for item in W08_P3IA_COMPONENT_KEYS
                if item in self.enabled_components
            )
        ):
            raise W08P3IaError("P3-Ia component inventory is invalid")
        _key(self.reader_key, where="P3-Ia reader")
        if type(self.logical_seq) is not int or self.logical_seq < 0:
            raise W08P3IaError("P3-Ia logical sequence is invalid")

    def component_enabled(self, key: str) -> bool:
        if key not in W08_P3IA_COMPONENT_KEYS:
            raise W08P3IaError("P3-Ia component is not registered")
        return key in self.enabled_components

    def stable_key(self) -> tuple[int, ...]:
        return digest_value(
            {
                "request": list(self.request_key),
                "case": list(self.case_key.components),
                "components": list(self.enabled_components),
                "reader": list(self.reader_key),
                "logical_seq": self.logical_seq,
            }
        )


@dataclass(frozen=True, order=True)
class W08P3IaUse:
    consumer_key: str
    request_key: tuple[int, ...]
    selected_candidate_key: tuple[int, ...]
    evidence_keys: tuple[StableRecordKey, ...]
    directional_choice_key: tuple[int, ...]
    use_key: tuple[int, ...]
    outcome_state: str
    outcome_key: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.consumer_key not in W08_CONSUMER_KEYS:
            raise W08P3IaError("P3-Ia consumer is not registered")
        for name in (
            "request_key",
            "selected_candidate_key",
            "directional_choice_key",
            "use_key",
            "outcome_key",
        ):
            _key(getattr(self, name), where=f"P3-Ia Use {name}")
        _stable_keys(self.evidence_keys, where="P3-Ia Use Evidence")
        if self.outcome_state not in W08_STOP_STATES:
            raise W08P3IaError("P3-Ia Use outcome is invalid")


@dataclass(frozen=True)
class W08P3IaResourceReceipt:
    hierarchy_candidates: int
    matched_features: int
    formed_centers: int
    opened_segments: int
    payload_gets: int
    payload_bytes: int
    recalled_records: int
    real_consumers: int
    recompute_objects: int
    logic_operations: int

    def __post_init__(self) -> None:
        values = tuple(getattr(self, name) for name in self.__dataclass_fields__)
        if any(type(value) is not int or value < 0 for value in values):
            raise W08P3IaError("P3-Ia resource count is invalid")
        budget = W08_RESOURCE_BUDGET
        if (
            self.hierarchy_candidates > budget["max_records"]
            or self.matched_features > budget["max_records"]
            or self.formed_centers > budget["max_records"]
            or self.opened_segments > budget["max_segments"]
            or self.payload_gets > budget["max_payload_gets"]
            or self.payload_bytes > budget["max_payload_bytes"]
            or self.recalled_records > budget["max_records"]
            or self.real_consumers > len(W08_CONSUMER_KEYS)
            or self.recompute_objects > budget["max_recompute_objects"]
            or self.logic_operations > budget["max_logic_operations"]
        ):
            raise W08P3IaError("P3-Ia resource budget was exceeded")


@dataclass(frozen=True)
class W08P3IaTrace:
    hierarchy_key: tuple[int, ...]
    hierarchy_candidate_keys: tuple[StableRecordKey, ...]
    paraphrase_evidence_keys: tuple[StableRecordKey, ...]
    formed_center_keys: tuple[StableRecordKey, ...]
    authorization_receipt_key: tuple[int, ...]
    citation_record_key: StableRecordKey
    citation_start: int
    citation_end: int
    qa_result_key: tuple[int, ...]
    generation_result_key: tuple[int, ...]
    revision_result_key: tuple[int, ...]
    acl_checked_before_payload: int = 1
    citation_exact: int = 1
    full_document_reparse_count: int = 0

    def __post_init__(self) -> None:
        _key(self.hierarchy_key, where="P3-Ia hierarchy")
        _stable_keys(
            self.hierarchy_candidate_keys,
            where="P3-Ia hierarchy candidates",
        )
        _stable_keys(
            self.paraphrase_evidence_keys,
            where="P3-Ia paraphrase Evidence",
        )
        _stable_keys(self.formed_center_keys, where="P3-Ia centers")
        _key(self.authorization_receipt_key, where="P3-Ia authorization receipt")
        if not isinstance(self.citation_record_key, StableRecordKey):
            raise TypeError("P3-Ia citation record key type is invalid")
        if (
            type(self.citation_start) is not int
            or type(self.citation_end) is not int
            or self.citation_start < 0
            or self.citation_start >= self.citation_end
        ):
            raise W08P3IaError("P3-Ia citation span is invalid")
        for name in (
            "qa_result_key",
            "generation_result_key",
            "revision_result_key",
        ):
            _key(getattr(self, name), where=f"P3-Ia {name}")
        if (
            self.acl_checked_before_payload,
            self.citation_exact,
            self.full_document_reparse_count,
        ) != (1, 1, 0):
            raise W08P3IaError("P3-Ia ACL/citation/revision evidence drifted")

    def stable_key(self) -> tuple[int, ...]:
        return digest_value(
            {
                "hierarchy": list(self.hierarchy_key),
                "candidates": [
                    list(item.components) for item in self.hierarchy_candidate_keys
                ],
                "paraphrase_evidence": [
                    list(item.components) for item in self.paraphrase_evidence_keys
                ],
                "centers": [list(item.components) for item in self.formed_center_keys],
                "authorization": list(self.authorization_receipt_key),
                "citation_record": list(self.citation_record_key.components),
                "citation_span": [self.citation_start, self.citation_end],
                "qa": list(self.qa_result_key),
                "generation": list(self.generation_result_key),
                "revision": list(self.revision_result_key),
            }
        )


@dataclass(frozen=True)
class W08P3IaAuditReceipt:
    request_key: tuple[int, ...]
    state: str
    trace: W08P3IaTrace | None
    resources: W08P3IaResourceReceipt
    uses: tuple[W08P3IaUse, ...]
    owner_calls: tuple[str, ...]
    blocked_component: str = ""
    authored_answer_read_count: int = 0
    future_pack_read_count: int = 0
    evaluator_label_read_count: int = 0
    host_learning_write_count: int = 0
    memory_learning_write_count: int = 0

    def __post_init__(self) -> None:
        _key(self.request_key, where="P3-Ia audit request")
        if self.state not in W08_STOP_STATES:
            raise W08P3IaError("P3-Ia audit state is invalid")
        if self.owner_calls != tuple(
            item for item in W08_P3IA_OWNER_KEYS if item in self.owner_calls
        ):
            raise W08P3IaError("P3-Ia owner order drifted")
        if any(
            (
                self.authored_answer_read_count,
                self.future_pack_read_count,
                self.evaluator_label_read_count,
                self.host_learning_write_count,
                self.memory_learning_write_count,
            )
        ):
            raise W08P3IaError("P3-Ia audit crossed a forbidden boundary")
        if self.state == "RESOLVED":
            if not isinstance(self.trace, W08P3IaTrace):
                raise TypeError("resolved P3-Ia audit lacks a trace")
            if tuple(item.consumer_key for item in self.uses) != W08_CONSUMER_KEYS:
                raise W08P3IaError("resolved P3-Ia audit lacks exact U/R/G Use")
            if self.resources.real_consumers != len(W08_CONSUMER_KEYS):
                raise W08P3IaError("P3-Ia real consumer count drifted")
            if self.blocked_component:
                raise W08P3IaError("resolved P3-Ia audit names a blocked component")
        else:
            if self.trace is not None or self.uses:
                raise W08P3IaError("stopped P3-Ia audit leaked a trace or Use")
            if self.blocked_component and self.blocked_component not in W08_P3IA_COMPONENT_KEYS:
                raise W08P3IaError("P3-Ia blocked component is not registered")

    def canonical_key(self) -> tuple[int, ...]:
        return digest_value(
            {
                "request": list(self.request_key),
                "state": self.state,
                "trace": None if self.trace is None else list(self.trace.stable_key()),
                "uses": [
                    {
                        "consumer": item.consumer_key,
                        "selected": list(item.selected_candidate_key),
                        "use": list(item.use_key),
                        "outcome": list(item.outcome_key),
                    }
                    for item in self.uses
                ],
                "blocked": self.blocked_component,
            }
        )


@dataclass(frozen=True)
class W08P3IaSupportingAblationReport:
    component_key: str
    full_state: str
    ablated_state: str
    zero_publication: int

    def __post_init__(self) -> None:
        if self.component_key not in W08_P3IA_COMPONENT_KEYS:
            raise W08P3IaError("P3-Ia supporting ablation is not registered")
        if (
            self.full_state != "RESOLVED"
            or self.ablated_state == "RESOLVED"
            or self.zero_publication != 1
        ):
            raise W08P3IaError("P3-Ia supporting ablation did not bear")


@dataclass(frozen=True)
class W08P3IaStageAblationReport:
    affected_dimensions: tuple[str, ...]
    unaffected_dimensions: tuple[str, ...]


def assess_w08_p3ia_supporting_ablation(
    full: W08P3IaAuditReceipt,
    ablated: W08P3IaAuditReceipt,
    *,
    component_key: str,
) -> W08P3IaSupportingAblationReport:
    if not isinstance(full, W08P3IaAuditReceipt) or not isinstance(
        ablated, W08P3IaAuditReceipt
    ):
        raise TypeError("P3-Ia supporting ablation receipt type is invalid")
    if (
        full.request_key != ablated.request_key
        or ablated.blocked_component != component_key
    ):
        raise W08P3IaError("P3-Ia supporting ablation changed the case identity")
    return W08P3IaSupportingAblationReport(
        component_key,
        full.state,
        ablated.state,
        int(ablated.trace is None and not ablated.uses),
    )


def assess_w08_p3ia_stage_ablation(
    *,
    full_dimension_outcomes: dict[str, str],
    ablated_dimension_outcomes: dict[str, str],
) -> W08P3IaStageAblationReport:
    expected = set(W08_DIMENSION_KEYS)
    if set(full_dimension_outcomes) != expected or set(
        ablated_dimension_outcomes
    ) != expected:
        raise W08P3IaError("P3-Ia stage ablation dimension inventory drifted")
    target = "W-08-P3IA"
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
        raise W08P3IaError("P3-Ia stage ablation is not orthogonal")
    return W08P3IaStageAblationReport(
        changed,
        tuple(key for key in W08_DIMENSION_KEYS if key != target),
    )


__all__ = [
    "W08P3IaAuditReceipt",
    "W08P3IaError",
    "W08P3IaRequest",
    "W08P3IaResourceReceipt",
    "W08P3IaStageAblationReport",
    "W08P3IaSupportingAblationReport",
    "W08P3IaTrace",
    "W08P3IaUse",
    "W08_P3IA_COMPONENT_KEYS",
    "W08_P3IA_OWNER_KEYS",
    "assess_w08_p3ia_stage_ablation",
    "assess_w08_p3ia_supporting_ablation",
]
