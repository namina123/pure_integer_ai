"""不读取 W09 课程 pack，构建 W08 可见的 P3-Ia 资料。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

from pure_integer_ai.experiments.ph2_dataset_contract import StableRecordKey
from pure_integer_ai.experiments.ph2_w05_contract import digest_value
from pure_integer_ai.experiments.ph2_w08_long_context_training import (
    W08LongContextTrainingBundle,
    W08LongContextTrainingError,
    compile_w08_long_context_training,
)
from pure_integer_ai.experiments.ph2_w08_payload import W08TrainingPayload


W08_P3IA_CASE_FAMILY = "W08_VISIBLE_PARAPHRASE_HIERARCHY_RECALL_V1"
W08_P3IA_PARAPHRASE_REASON_KEY = (80806, 101, 1)


class W08P3IaTrainingError(ValueError):
    """W08 可见 P3-Ia 资料不完整或越过数据边界。"""


def _key(value: object, *, where: str) -> tuple[int, ...]:
    if not isinstance(value, tuple) or not value or any(
        type(item) is not int for item in value
    ):
        raise W08P3IaTrainingError(f"{where} is not a strict integer key")
    return value


def _record_key(domain: int, value: object) -> StableRecordKey:
    digest = digest_value(value)
    return StableRecordKey((domain, *(item + 1 for item in digest[:12])))


@dataclass(frozen=True, order=True)
class W08P3IaTrainingCase:
    """由一条 W08 train 记录机械形成的改写到精确引文 case。"""

    case_key: StableRecordKey
    observation_key: tuple[int, ...]
    source_ref_key: tuple[int, ...]
    exact_quote_key: StableRecordKey
    paraphrase_quote_key: StableRecordKey
    feature_key: StableRecordKey
    exact_surface: str
    paraphrase_surface: str
    exact_surface_sha256: str
    paraphrase_surface_sha256: str
    citation_start: int
    citation_end: int
    paraphrase_start: int
    paraphrase_end: int

    def __post_init__(self) -> None:
        for name in (
            "case_key",
            "exact_quote_key",
            "paraphrase_quote_key",
            "feature_key",
        ):
            if not isinstance(getattr(self, name), StableRecordKey):
                raise TypeError(f"P3-Ia {name} type is invalid")
        _key(self.observation_key, where="P3-Ia observation")
        _key(self.source_ref_key, where="P3-Ia source")
        if (
            not self.exact_surface
            or not self.paraphrase_surface
            or self.exact_surface == self.paraphrase_surface
        ):
            raise W08P3IaTrainingError("P3-Ia paraphrase pair is not distinct")
        for surface, expected in (
            (self.exact_surface, self.exact_surface_sha256),
            (self.paraphrase_surface, self.paraphrase_surface_sha256),
        ):
            if hashlib.sha256(surface.encode("utf-8")).hexdigest() != expected:
                raise W08P3IaTrainingError("P3-Ia surface digest drifted")
        for start, end, surface in (
            (self.citation_start, self.citation_end, self.exact_surface),
            (self.paraphrase_start, self.paraphrase_end, self.paraphrase_surface),
        ):
            if (
                type(start) is not int
                or type(end) is not int
                or start < 0
                or start >= end
                or end - start != len(surface)
            ):
                raise W08P3IaTrainingError("P3-Ia absolute span is invalid")

    def stable_key(self) -> tuple[int, ...]:
        return digest_value(
            {
                "case": list(self.case_key.components),
                "observation": list(self.observation_key),
                "source": list(self.source_ref_key),
                "exact_quote": list(self.exact_quote_key.components),
                "paraphrase_quote": list(self.paraphrase_quote_key.components),
                "feature": list(self.feature_key.components),
                "exact_sha256": self.exact_surface_sha256,
                "paraphrase_sha256": self.paraphrase_surface_sha256,
                "citation": [self.citation_start, self.citation_end],
                "paraphrase": [self.paraphrase_start, self.paraphrase_end],
            }
        )


@dataclass(frozen=True)
class W08P3IaTrainingAudit:
    observation_count: int
    evidence_binding_count: int
    attribution_observation_count: int
    quotation_span_count: int
    exact_quote_count: int
    paraphrase_link_count: int
    case_count: int
    authored_answer_read_count: int = 0
    future_pack_read_count: int = 0
    evaluator_label_read_count: int = 0
    host_learning_write_count: int = 0
    memory_learning_write_count: int = 0

    def __post_init__(self) -> None:
        values = tuple(getattr(self, name) for name in self.__dataclass_fields__)
        if any(type(value) is not int or value < 0 for value in values):
            raise W08P3IaTrainingError("P3-Ia training audit count is invalid")
        if self.observation_count != self.evidence_binding_count:
            raise W08P3IaTrainingError("P3-Ia Observation/Evidence binding drifted")
        if min(
            self.attribution_observation_count,
            self.quotation_span_count,
            self.exact_quote_count,
            self.paraphrase_link_count,
            self.case_count,
        ) <= 0:
            raise W08P3IaTrainingError("P3-Ia visible material is not decidable")
        if self.case_count != self.paraphrase_link_count:
            raise W08P3IaTrainingError("P3-Ia paraphrase cases were manually filtered")
        if any(
            (
                self.authored_answer_read_count,
                self.future_pack_read_count,
                self.evaluator_label_read_count,
                self.host_learning_write_count,
                self.memory_learning_write_count,
            )
        ):
            raise W08P3IaTrainingError("P3-Ia training crossed a forbidden boundary")

    def stable_key(self) -> tuple[int, ...]:
        return digest_value(
            {name: getattr(self, name) for name in self.__dataclass_fields__}
        )


@dataclass(frozen=True)
class W08P3IaTrainingBundle:
    family_key: str
    long_context: W08LongContextTrainingBundle
    cases: tuple[W08P3IaTrainingCase, ...]
    audit: W08P3IaTrainingAudit
    family_identity: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.family_key != W08_P3IA_CASE_FAMILY:
            raise W08P3IaTrainingError("P3-Ia case family drifted")
        if not isinstance(self.long_context, W08LongContextTrainingBundle):
            raise TypeError("P3-Ia long-context material type is invalid")
        if (
            not isinstance(self.cases, tuple)
            or not self.cases
            or any(not isinstance(item, W08P3IaTrainingCase) for item in self.cases)
            or self.cases != tuple(sorted(self.cases))
        ):
            raise W08P3IaTrainingError("P3-Ia cases are not canonical")
        if len({item.case_key for item in self.cases}) != len(self.cases):
            raise W08P3IaTrainingError("P3-Ia case identity is duplicated")
        if not isinstance(self.audit, W08P3IaTrainingAudit):
            raise TypeError("P3-Ia training audit type is invalid")
        if len(self.cases) != self.audit.case_count:
            raise W08P3IaTrainingError("P3-Ia case/audit count drifted")
        _key(self.family_identity, where="P3-Ia family identity")


def _span(value: object, surface: str) -> dict:
    if not isinstance(value, dict):
        raise W08P3IaTrainingError("P3-Ia quotation span is not an object")
    required = {
        "attribution_id",
        "boundary_state",
        "candidate_version",
        "end",
        "exact_text",
        "paraphrase_of_quote_id",
        "quote_id",
        "start",
        "surface_sha256",
        "version_kind",
    }
    if set(value) != required:
        raise W08P3IaTrainingError("P3-Ia quotation span schema drifted")
    text = value["exact_text"]
    start = value["start"]
    end = value["end"]
    if (
        not isinstance(text, str)
        or not text
        or type(start) is not int
        or type(end) is not int
        or start < 0
        or start >= end
        or end > len(surface)
        or surface[start:end] != text
        or hashlib.sha256(text.encode("utf-8")).hexdigest()
        != value["surface_sha256"]
    ):
        raise W08P3IaTrainingError("P3-Ia quotation span content drifted")
    return value


def compile_w08_p3ia_training(payload: W08TrainingPayload) -> W08P3IaTrainingBundle:
    """把每条 W08 可见改写关系编译成预注册 case。"""
    if not isinstance(payload, W08TrainingPayload):
        raise TypeError("P3-Ia compiler requires W08TrainingPayload")
    try:
        long_context = compile_w08_long_context_training(payload)
    except W08LongContextTrainingError as error:
        raise W08P3IaTrainingError("P3-Ia parent material audit failed") from error
    material_by_observation = {
        item.observation_key: item for item in long_context.material
    }
    cases: list[W08P3IaTrainingCase] = []
    attribution_count = 0
    quotation_count = 0
    exact_count = 0
    paraphrase_count = 0
    for observation in sorted(payload.observations, key=lambda item: item.stable_key):
        if observation.payload_kind != "AttributionQuotationCandidateV1":
            continue
        attribution_count += 1
        value = observation.typed_payload.to_value()
        if not isinstance(value, dict):
            raise W08P3IaTrainingError("P3-Ia attribution payload is not an object")
        material = material_by_observation.get(tuple(observation.stable_key.components))
        if material is None:
            raise W08P3IaTrainingError("P3-Ia attribution lacks visible material")
        surface = material.surface
        spans_raw = value.get("quotation_spans")
        if not isinstance(spans_raw, list):
            raise W08P3IaTrainingError("P3-Ia quotation inventory is missing")
        spans = tuple(_span(item, surface) for item in spans_raw)
        quotation_count += len(spans)
        exact_count += sum(item["version_kind"] == "EXACT_QUOTE" for item in spans)
        by_id = {item["quote_id"]: item for item in spans}
        if len(by_id) != len(spans):
            raise W08P3IaTrainingError("P3-Ia quotation id is duplicated")
        for paraphrase in spans:
            if paraphrase["version_kind"] != "PARAPHRASE":
                continue
            paraphrase_count += 1
            parent_id = paraphrase["paraphrase_of_quote_id"]
            exact = by_id.get(parent_id)
            if (
                not parent_id
                or exact is None
                or exact["version_kind"] != "EXACT_QUOTE"
                or exact["boundary_state"] != "MATCH"
                or paraphrase["boundary_state"] != "MATCH"
            ):
                raise W08P3IaTrainingError("P3-Ia paraphrase link is not supported")
            identity = {
                "family": W08_P3IA_CASE_FAMILY,
                "observation": list(observation.stable_key.components),
                "exact_quote_id": parent_id,
                "paraphrase_quote_id": paraphrase["quote_id"],
                "exact_sha256": exact["surface_sha256"],
                "paraphrase_sha256": paraphrase["surface_sha256"],
            }
            cases.append(
                W08P3IaTrainingCase(
                    _record_key(8080601, identity),
                    tuple(observation.stable_key.components),
                    tuple(observation.source_ref_key.components),
                    _record_key(8080602, {**identity, "kind": "exact"}),
                    _record_key(8080603, {**identity, "kind": "paraphrase"}),
                    _record_key(8080604, {**identity, "kind": "feature"}),
                    exact["exact_text"],
                    paraphrase["exact_text"],
                    exact["surface_sha256"],
                    paraphrase["surface_sha256"],
                    material.start + exact["start"],
                    material.start + exact["end"],
                    material.start + paraphrase["start"],
                    material.start + paraphrase["end"],
                )
            )
    ordered = tuple(sorted(cases))
    audit = W08P3IaTrainingAudit(
        long_context.audit.observation_count,
        long_context.audit.evidence_binding_count,
        attribution_count,
        quotation_count,
        exact_count,
        paraphrase_count,
        len(ordered),
    )
    family_identity = digest_value(
        {
            "family": W08_P3IA_CASE_FAMILY,
            "parent_material": list(long_context.material_key),
            "audit": list(audit.stable_key()),
            "cases": [list(item.stable_key()) for item in ordered],
        }
    )
    return W08P3IaTrainingBundle(
        W08_P3IA_CASE_FAMILY,
        long_context,
        ordered,
        audit,
        family_identity,
    )


__all__ = [
    "W08P3IaTrainingAudit",
    "W08P3IaTrainingBundle",
    "W08P3IaTrainingCase",
    "W08P3IaTrainingError",
    "W08_P3IA_CASE_FAMILY",
    "W08_P3IA_PARAPHRASE_REASON_KEY",
    "compile_w08_p3ia_training",
]
