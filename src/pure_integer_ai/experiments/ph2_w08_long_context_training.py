"""把 W08 可见训练表层编译为 typed 长上下文资料包。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from pure_integer_ai.experiments.ph2_w08_payload import W08TrainingPayload
from pure_integer_ai.experiments.ph2_w08_recompute_training import (
    W08RecomputeTrainingError,
    audit_w08_recompute_training,
)
from pure_integer_ai.experiments.ph2_w08_registry import (
    W08RegistryError,
    audit_w08_registry_payload,
)
from pure_integer_ai.experiments.ph2_w05_contract import digest_value


_MATERIAL_KINDS = (
    "DiscourseRevisionQuery",
    "DiscourseInformationCandidateV1",
    "OpenSetClarificationCandidateV1",
    "AttributionQuotationCandidateV1",
)


class W08LongContextTrainingError(ValueError):
    """W08-05 仅训练资料或可见 schema 未闭合。"""


def _key(value: object, *, where: str) -> tuple[int, ...]:
    if not isinstance(value, tuple) or not value or any(type(item) is not int for item in value):
        raise W08LongContextTrainingError(f"{where} is not a strict integer key")
    return value


@dataclass(frozen=True, order=True)
class W08LongContextMaterialItem:
    """一条带 typed 绝对位置 metadata 的 W08 可见表层。"""

    observation_key: tuple[int, ...]
    source_ref_key: tuple[int, ...]
    payload_kind: str
    surface: str
    section_ordinal: int
    paragraph_ordinal: int
    proposition_ordinal: int
    start: int
    end: int

    def __post_init__(self) -> None:
        _key(self.observation_key, where="long-context observation")
        _key(self.source_ref_key, where="long-context source")
        if self.payload_kind not in _MATERIAL_KINDS:
            raise W08LongContextTrainingError("material payload kind is not visible")
        if not isinstance(self.surface, str) or not self.surface:
            raise W08LongContextTrainingError("material surface is empty")
        for name in (
            "section_ordinal",
            "paragraph_ordinal",
            "proposition_ordinal",
            "start",
            "end",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise W08LongContextTrainingError(f"material {name} is invalid")
        if self.start >= self.end or self.end - self.start != len(self.surface):
            raise W08LongContextTrainingError("material absolute surface range is invalid")


@dataclass(frozen=True)
class W08LongContextTrainingAudit:
    observation_count: int
    evidence_binding_count: int
    material_item_count: int
    source_surface_count: int
    paragraph_span_count: int
    revision_occurrence_count: int
    information_relation_count: int
    clarification_obligation_count: int
    attribution_count: int
    quotation_span_count: int
    multi_center_signal: int
    cold_relevant_signal: int
    conflict_signal: int
    clarification_signal: int
    budget_signal: int
    authored_answer_read_count: int = 0

    def __post_init__(self) -> None:
        fields = tuple(getattr(self, name) for name in self.__dataclass_fields__)
        if any(type(value) is not int or value < 0 for value in fields):
            raise W08LongContextTrainingError("long-context audit count is invalid")
        if self.observation_count != self.evidence_binding_count:
            raise W08LongContextTrainingError("Observation/Evidence coverage drifted")
        if self.material_item_count <= 0 or self.source_surface_count <= 0:
            raise W08LongContextTrainingError("visible W08 material is incomplete")
        if min(
            self.paragraph_span_count,
            self.revision_occurrence_count,
            self.information_relation_count,
            self.clarification_obligation_count,
            self.budget_signal,
        ) <= 0:
            raise W08LongContextTrainingError("long-context typed operation coverage is incomplete")
        if self.multi_center_signal != 1 or self.cold_relevant_signal != 1:
            raise W08LongContextTrainingError("multi-center/cold-page signal is missing")
        if self.conflict_signal != 1 or self.clarification_signal != 1:
            raise W08LongContextTrainingError("conflict/clarification signal is missing")
        if self.authored_answer_read_count != 0:
            raise W08LongContextTrainingError("authored answer was read by long-context training")

    def stable_key(self) -> tuple[int, ...]:
        return digest_value({
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        })


@dataclass(frozen=True)
class W08LongContextTrainingBundle:
    audit: W08LongContextTrainingAudit
    material: tuple[W08LongContextMaterialItem, ...]
    material_key: tuple[int, ...]
    document_text: str

    def __post_init__(self) -> None:
        if not isinstance(self.audit, W08LongContextTrainingAudit):
            raise TypeError("long-context training audit type is invalid")
        if not isinstance(self.material, tuple) or not self.material:
            raise W08LongContextTrainingError("long-context material is empty")
        if len(self.material) != self.audit.material_item_count:
            raise W08LongContextTrainingError("material/audit count drifted")
        if not isinstance(self.document_text, str) or not self.document_text:
            raise W08LongContextTrainingError("long-context document text is empty")
        if tuple(item.end for item in self.material)[-1] != len(self.document_text):
            raise W08LongContextTrainingError("material does not cover document text")
        _key(self.material_key, where="long-context material key")


def _surface(value: dict, *, where: str) -> str:
    if "surface" in value:
        surface = value["surface"]
    else:
        observed = value.get("observed_surface")
        if not isinstance(observed, dict):
            raise W08LongContextTrainingError(f"{where} lacks a visible surface")
        surface = observed.get("text")
        expected = observed.get("sha256")
        if not isinstance(expected, str) or hashlib.sha256(surface.encode("utf-8")).hexdigest() != expected:
            raise W08LongContextTrainingError(f"{where} surface digest drifted")
    if not isinstance(surface, str) or not surface:
        raise W08LongContextTrainingError(f"{where} surface is empty")
    return surface


def compile_w08_long_context_training(
    payload: W08TrainingPayload,
) -> W08LongContextTrainingBundle:
    """只审计安全字段，并按 typed 顺序拼接可见表层。"""
    try:
        registry = audit_w08_registry_payload(payload)
        recompute = audit_w08_recompute_training(payload)
    except (W08RegistryError, W08RecomputeTrainingError) as error:
        raise W08LongContextTrainingError("W08-05 train payload audit failed") from error

    records: list[tuple[object, str, dict]] = []
    source_surface_count = 0
    paragraph_span_count = 0
    revision_occurrence_count = 0
    information_relation_count = 0
    clarification_obligation_count = 0
    attribution_count = 0
    quotation_span_count = 0
    multi_center = 0
    cold_relevant = 0
    conflict = 0
    clarification = 0
    budget = 0
    section_by_kind = {kind: ordinal for ordinal, kind in enumerate(_MATERIAL_KINDS)}
    for observation in sorted(payload.observations, key=lambda item: item.stable_key):
        value = observation.typed_payload.to_value()
        if observation.payload_kind == "RAW_SOURCE_OBSERVATION_V1":
            source_surface_count += 1
            continue
        if observation.payload_kind not in _MATERIAL_KINDS:
            raise W08LongContextTrainingError("unregistered payload entered material")
        if not isinstance(value, dict):
            raise W08LongContextTrainingError("visible Observation payload is not an object")
        text = _surface(value, where=observation.payload_kind)
        records.append((observation, text, value))
        if observation.payload_kind == "DiscourseRevisionQuery":
            paragraph_spans = value.get("paragraph_spans")
            occurrences = value.get("occurrences")
            if not isinstance(paragraph_spans, list) or not isinstance(occurrences, list):
                raise W08LongContextTrainingError("revision hierarchy fields are missing")
            paragraph_span_count += len(paragraph_spans)
            revision_occurrence_count += len(occurrences)
            cold_relevant |= int(bool(paragraph_spans))
        elif observation.payload_kind == "DiscourseInformationCandidateV1":
            relations = value.get("discourse_relations")
            propositions = value.get("proposition_candidates")
            if not isinstance(relations, list) or not isinstance(propositions, list):
                raise W08LongContextTrainingError("information hierarchy fields are missing")
            information_relation_count += len(relations)
            multi_center |= int(len(propositions) > 1)
        elif observation.payload_kind == "OpenSetClarificationCandidateV1":
            obligations = value.get("missing_information_obligations")
            if not isinstance(obligations, list):
                raise W08LongContextTrainingError("clarification obligations are missing")
            clarification_obligation_count += len(obligations)
            clarification |= int(bool(obligations))
        elif observation.payload_kind == "AttributionQuotationCandidateV1":
            attributions = value.get("attribution_candidates")
            quotations = value.get("quotation_spans")
            if not isinstance(attributions, list) or not isinstance(quotations, list):
                raise W08LongContextTrainingError("attribution hierarchy fields are missing")
            attribution_count += len(attributions)
            quotation_span_count += len(quotations)
        budget |= int(isinstance(value.get("resource_budget"), dict))
        conflict |= int(observation.sample_role == "conflict")

    if not records:
        raise W08LongContextTrainingError("no visible W08 material records")
    material: list[W08LongContextMaterialItem] = []
    text_parts: list[str] = []
    cursor = 0
    for ordinal, (observation, surface, value) in enumerate(records):
        if text_parts:
            text_parts.append("\n")
            cursor += 1
        start = cursor
        text_parts.append(surface)
        cursor += len(surface)
        material.append(W08LongContextMaterialItem(
            tuple(observation.stable_key.components),
            tuple(observation.source_ref_key.components),
            observation.payload_kind,
            surface,
            section_by_kind[observation.payload_kind],
            ordinal,
            ordinal,
            start,
            cursor,
        ))
    document_text = "".join(text_parts)
    audit = W08LongContextTrainingAudit(
        registry.observation_count,
        registry.teacher_evidence_count,
        len(material),
        source_surface_count,
        paragraph_span_count,
        revision_occurrence_count,
        information_relation_count,
        clarification_obligation_count,
        attribution_count,
        quotation_span_count,
        multi_center,
        cold_relevant,
        conflict,
        clarification,
        budget,
        0,
    )
    material_key = digest_value({
        "audit": list(audit.stable_key()),
        "items": [
            {
                "kind": item.payload_kind,
                "key": list(item.observation_key),
                "surface_sha256": hashlib.sha256(item.surface.encode("utf-8")).hexdigest(),
            }
            for item in material
        ],
    })
    return W08LongContextTrainingBundle(audit, tuple(material), material_key, document_text)


__all__ = [
    "W08LongContextMaterialItem",
    "W08LongContextTrainingAudit",
    "W08LongContextTrainingBundle",
    "W08LongContextTrainingError",
    "compile_w08_long_context_training",
]
