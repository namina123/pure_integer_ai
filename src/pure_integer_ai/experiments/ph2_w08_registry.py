"""W-08 五维 schema、U/R/G consumer 与 train payload registry。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pure_integer_ai.experiments.ph2_w08_authority import (
    W08_ABLATION_KEYS,
    W08_DIMENSION_KEYS,
    W08_SUBTASK_ORDER,
)
from pure_integer_ai.experiments.ph2_w08_contract import (
    W08_CONSUMER_KEYS,
    W08_LEARNING_PACK_KEYS,
    W08_RESOURCE_BUDGET,
    W08_STOP_STATES,
)
from pure_integer_ai.experiments.ph2_w08_payload import W08TrainingPayload


class W08RegistryError(RuntimeError):
    """W-08 五维 schema 或 train Evidence 覆盖不闭合。"""


@dataclass(frozen=True)
class W08DimensionSchema:
    """声明一个 bearing 的请求、结果、trace、资源和消费边界。"""

    subtask_key: str
    dimension_key: str
    ablation_key: str
    request_kind: str
    result_kind: str
    trace_fields: tuple[str, ...]
    resource_fields: tuple[str, ...]
    consumer_keys: tuple[str, ...]
    use_fields: tuple[str, ...]
    outcome_states: tuple[str, ...]
    allowed_write_owners: tuple[str, ...]

    def __post_init__(self) -> None:
        ordinal = W08_SUBTASK_ORDER.index(self.subtask_key)
        if (
            self.dimension_key != W08_DIMENSION_KEYS[ordinal]
            or self.ablation_key != W08_ABLATION_KEYS[ordinal]
            or not self.request_kind
            or not self.result_kind
            or self.consumer_keys != W08_CONSUMER_KEYS
            or self.outcome_states != W08_STOP_STATES
            or self.allowed_write_owners != ("PH2_W08_TRANSACTION_OWNER",)
        ):
            raise W08RegistryError("W-08 dimension schema drifted")
        required_trace = {
            "source_key", "scope_key", "version_key", "owner_key",
            "logical_clock", "evidence_key", "use_key", "outcome_key",
        }
        if not required_trace.issubset(self.trace_fields):
            raise W08RegistryError("W-08 trace schema is incomplete")
        if set(self.resource_fields) != set(W08_RESOURCE_BUDGET):
            raise W08RegistryError("W-08 resource schema is incomplete")
        if set(self.use_fields) != {
            "consumer_key", "request_key", "selected_candidate_key",
            "evidence_keys", "directional_choice_key",
        }:
            raise W08RegistryError("W-08 Use schema is incomplete")


_TRACE_FIELDS = (
    "source_key",
    "scope_key",
    "version_key",
    "owner_key",
    "logical_clock",
    "candidate_key",
    "evidence_key",
    "lifecycle",
    "support_keys",
    "refute_keys",
    "supersedes_key",
    "use_key",
    "outcome_key",
)
_USE_FIELDS = (
    "consumer_key",
    "request_key",
    "selected_candidate_key",
    "evidence_keys",
    "directional_choice_key",
)


W08_DIMENSION_REGISTRY = {
    subtask: W08DimensionSchema(
        subtask,
        W08_DIMENSION_KEYS[index],
        W08_ABLATION_KEYS[index],
        f"W08{subtask.title().replace('_', '')}RequestV1",
        f"W08{subtask.title().replace('_', '')}ResultV1",
        _TRACE_FIELDS,
        tuple(sorted(W08_RESOURCE_BUDGET)),
        W08_CONSUMER_KEYS,
        _USE_FIELDS,
        W08_STOP_STATES,
        ("PH2_W08_TRANSACTION_OWNER",),
    )
    for index, subtask in enumerate(W08_SUBTASK_ORDER)
}

W08_PACK_REGISTRY = {
    W08_LEARNING_PACK_KEYS[0]: (
        "DiscourseRevisionQuery", "DISCOURSE_REVISION", "W-08"
    ),
    W08_LEARNING_PACK_KEYS[1]: (
        "DiscourseInformationCandidateV1",
        "LC07_DISCOURSE_INFORMATION_STRUCTURE",
        "W-08",
    ),
    W08_LEARNING_PACK_KEYS[2]: (
        "OpenSetClarificationCandidateV1", "LC08_OPEN_SET_CLARIFICATION", "W-08"
    ),
    W08_LEARNING_PACK_KEYS[3]: (
        "AttributionQuotationCandidateV1",
        "LC14_ATTRIBUTION_QUOTATION_PERSPECTIVE",
        "W-08",
    ),
    W08_LEARNING_PACK_KEYS[4]: (
        "RAW_SOURCE_OBSERVATION_V1", "D-02-SOURCE-ZHWIKIPEDIA-V1", "W-08"
    ),
    W08_LEARNING_PACK_KEYS[5]: (
        "RAW_SOURCE_OBSERVATION_V1", "D-02-SOURCE-ZHWIKTIONARY-V1", "W-03"
    ),
}


@dataclass(frozen=True)
class W08RegistryAuditReport:
    observation_count: int
    teacher_evidence_count: int
    source_ref_count: int
    pack_counts: tuple[tuple[str, int], ...]
    sample_roles: tuple[str, ...]
    payload_kinds: tuple[str, ...]
    definitive_truth_claims: int
    expected_or_label_fields: int


def _pack_key_for(record, source_by_key: dict[object, object]) -> str:
    source = source_by_key.get(record.source_ref_key)
    if source is None:
        raise W08RegistryError("W-08 Observation points to unknown SourceRef")
    candidates = []
    for pack_key in W08_LEARNING_PACK_KEYS:
        expected_source = pack_key.split("--", 1)[0]
        if source.source_key == expected_source:
            candidates.append(pack_key)
    if source.source_key == "AUTHORED_CC0_V1":
        by_substage = {
            schema[1]: key for key, schema in W08_PACK_REGISTRY.items()
            if key.startswith("AUTHORED_CC0_V1")
        }
        pack_key = by_substage.get(record.substage)
        if pack_key is None:
            raise W08RegistryError("W-08 authored Observation is not registered")
        return pack_key
    if len(candidates) != 1:
        raise W08RegistryError("W-08 source pack identity is ambiguous")
    return candidates[0]


def audit_w08_registry_payload(payload: W08TrainingPayload) -> W08RegistryAuditReport:
    """核验六 pack schema、四态样本、零 label/expected 和来源闭合。"""
    if not isinstance(payload, W08TrainingPayload):
        raise W08RegistryError("W-08 registry requires W08TrainingPayload")
    source_by_key = {item.stable_key: item for item in payload.source_refs}
    observation_by_key = {item.stable_key: item for item in payload.observations}
    if len(source_by_key) != len(payload.source_refs):
        raise W08RegistryError("W-08 SourceRef stable key is duplicated")
    counts = {key: 0 for key in W08_LEARNING_PACK_KEYS}
    roles: set[str] = set()
    kinds: set[str] = set()
    truth_claims = 0
    label_fields = 0
    for observation in payload.observations:
        pack_key = _pack_key_for(observation, source_by_key)
        expected_kind, expected_substage, expected_stage = W08_PACK_REGISTRY[pack_key]
        if (
            observation.payload_kind != expected_kind
            or observation.substage != expected_substage
            or observation.w_stage != expected_stage
            or observation.split != "train"
        ):
            raise W08RegistryError("W-08 Observation does not match pack registry")
        value = observation.typed_payload.to_value()
        if not isinstance(value, dict):
            raise W08RegistryError("W-08 typed payload must be an object")
        encoded_keys = {str(key).lower() for key in value}
        label_fields += len(encoded_keys & {"expected", "label", "evaluator_label"})
        if expected_kind == "RAW_SOURCE_OBSERVATION_V1":
            if (
                value.get("definitive_truth_authoritative") != 0
                or value.get("raw_observation_append_only") != 1
            ):
                raise W08RegistryError("external text was promoted to definitive truth")
            truth_claims += value.get("definitive_truth_authoritative", 0)
        else:
            if value.get("selection_state") not in {None, "UNSELECTED"}:
                raise W08RegistryError("train Observation contains adopted selection")
        counts[pack_key] += 1
        roles.add(observation.sample_role)
        kinds.add(observation.payload_kind)
    if any(count == 0 for count in counts.values()):
        raise W08RegistryError("W-08 registered train pack is empty")
    for evidence in payload.teacher_evidence:
        observation = observation_by_key.get(evidence.observation_key)
        if observation is None or evidence.source_ref_key != observation.source_ref_key:
            raise W08RegistryError("W-08 Evidence is not bound to current Observation")
        if evidence.visible_from_stage not in {"W-03", "W-08"}:
            raise W08RegistryError("W-08 Evidence visibility stage is invalid")
    if len(payload.teacher_evidence) != len(payload.observations):
        raise W08RegistryError("W-08 train Evidence coverage is incomplete")
    if not {"support", "refute", "conflict", "supersede"}.issubset(roles):
        raise W08RegistryError("W-08 lifecycle sample roles are incomplete")
    if label_fields:
        raise W08RegistryError("W-08 train payload contains expected/label field")
    return W08RegistryAuditReport(
        len(payload.observations),
        len(payload.teacher_evidence),
        len(payload.source_refs),
        tuple((key, counts[key]) for key in W08_LEARNING_PACK_KEYS),
        tuple(sorted(roles)),
        tuple(sorted(kinds)),
        truth_claims,
        label_fields,
    )


__all__ = [
    "W08_DIMENSION_REGISTRY",
    "W08_PACK_REGISTRY",
    "W08DimensionSchema",
    "W08RegistryAuditReport",
    "W08RegistryError",
    "audit_w08_registry_payload",
]
