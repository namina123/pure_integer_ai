"""W-08 篇章 facade 的六 pack train schema 与投影隔离审计。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.experiments.ph2_w08_payload import W08TrainingPayload
from pure_integer_ai.experiments.ph2_w08_registry import (
    W08RegistryError,
    audit_w08_registry_payload,
)
from pure_integer_ai.experiments.ph2_w08_variation import (
    W08VariationError,
    learn_w08_variation,
)


class W08DiscourseTrainingError(ValueError):
    """W-08 篇章 train schema、Evidence identity 或投影隔离不闭合。"""


@dataclass(frozen=True)
class W08DiscourseTrainingAudit:
    observation_count: int
    evidence_binding_count: int
    discourse_revision_count: int
    discourse_information_count: int
    open_set_clarification_count: int
    attribution_quotation_count: int
    raw_source_count: int
    reference_plan_count: int
    parser_revision_count: int
    proposition_candidate_count: int
    discourse_relation_count: int
    information_structure_count: int
    qud_candidate_count: int
    open_obligation_count: int
    attribution_candidate_count: int
    quotation_span_count: int
    reported_projection_violation_count: int
    raw_truth_claim_count: int
    authored_answer_read_count: int

    def __post_init__(self) -> None:
        values = tuple(getattr(self, name) for name in self.__dataclass_fields__)
        if any(type(item) is not int or item < 0 for item in values):
            raise W08DiscourseTrainingError("discourse audit count is invalid")
        if (
            self.observation_count != self.evidence_binding_count
            or sum((
                self.discourse_revision_count,
                self.discourse_information_count,
                self.open_set_clarification_count,
                self.attribution_quotation_count,
                self.raw_source_count,
            )) != self.observation_count
        ):
            raise W08DiscourseTrainingError("discourse train inventory is incomplete")
        required_positive = (
            self.reference_plan_count,
            self.parser_revision_count,
            self.proposition_candidate_count,
            self.discourse_relation_count,
            self.information_structure_count,
            self.qud_candidate_count,
            self.open_obligation_count,
            self.attribution_candidate_count,
        )
        if any(item <= 0 for item in required_positive):
            raise W08DiscourseTrainingError("discourse typed operation coverage is incomplete")
        if (
            self.reported_projection_violation_count
            or self.raw_truth_claim_count
            or self.authored_answer_read_count
        ):
            raise W08DiscourseTrainingError("private answer/truth leaked into discourse training")


def _list(value: object, *, where: str) -> list:
    if not isinstance(value, list):
        raise W08DiscourseTrainingError(f"{where} must be a list")
    return value


def audit_w08_discourse_training(
    payload: W08TrainingPayload,
) -> W08DiscourseTrainingAudit:
    """只读取 Observation schema 和 Evidence identity；不读 authored 答案。"""
    try:
        registry = audit_w08_registry_payload(payload)
        learning = learn_w08_variation(payload)
    except (W08RegistryError, W08VariationError) as error:
        raise W08DiscourseTrainingError("discourse payload audit failed") from error

    counts = {
        "DiscourseRevisionQuery": 0,
        "DiscourseInformationCandidateV1": 0,
        "OpenSetClarificationCandidateV1": 0,
        "AttributionQuotationCandidateV1": 0,
        "RAW_SOURCE_OBSERVATION_V1": 0,
    }
    reference_plans = 0
    parser_revisions = 0
    propositions = 0
    relations = 0
    information = 0
    qud = 0
    obligations = 0
    attributions = 0
    quotations = 0
    projection_violations = 0
    truth_claims = 0
    for observation in payload.observations:
        value = observation.typed_payload.to_value()
        if observation.payload_kind not in counts:
            raise W08DiscourseTrainingError("discourse payload kind is not registered")
        counts[observation.payload_kind] += 1
        if observation.payload_kind == "DiscourseRevisionQuery":
            reference_plans += value.get("reference_plan") is not None
            parser_revisions += value.get("parser_revision_plan") is not None
        elif observation.payload_kind == "DiscourseInformationCandidateV1":
            propositions += len(_list(
                value.get("proposition_candidates"),
                where="discourse proposition candidates",
            ))
            relations += len(_list(
                value.get("discourse_relations"),
                where="discourse relations",
            ))
            information += len(_list(
                value.get("information_structure"),
                where="information structure",
            ))
            qud += len(_list(value.get("qud_candidates"), where="QUD candidates"))
            obligations += len(_list(
                value.get("presupposition_obligations"),
                where="presupposition obligations",
            ))
        elif observation.payload_kind == "OpenSetClarificationCandidateV1":
            obligations += len(_list(
                value.get("missing_information_obligations"),
                where="missing information obligations",
            ))
        elif observation.payload_kind == "AttributionQuotationCandidateV1":
            raw_propositions = _list(
                value.get("proposition_candidates"),
                where="reported proposition candidates",
            )
            raw_attributions = _list(
                value.get("attribution_candidates"),
                where="attribution candidates",
            )
            raw_quotations = _list(
                value.get("quotation_spans"),
                where="quotation spans",
            )
            propositions += len(raw_propositions)
            attributions += len(raw_attributions)
            quotations += len(raw_quotations)
            projection_violations += sum(
                item.get("current_projection_allowed") != 0
                for item in (*raw_propositions, *raw_attributions)
            )
        else:
            truth_claims += value.get("definitive_truth_authoritative", 0)

    return W08DiscourseTrainingAudit(
        registry.observation_count,
        len(learning.evidence_bindings),
        counts["DiscourseRevisionQuery"],
        counts["DiscourseInformationCandidateV1"],
        counts["OpenSetClarificationCandidateV1"],
        counts["AttributionQuotationCandidateV1"],
        counts["RAW_SOURCE_OBSERVATION_V1"],
        reference_plans,
        parser_revisions,
        propositions,
        relations,
        information,
        qud,
        obligations,
        attributions,
        quotations,
        projection_violations,
        truth_claims,
        0,
    )


__all__ = [
    "W08DiscourseTrainingAudit",
    "W08DiscourseTrainingError",
    "audit_w08_discourse_training",
]
