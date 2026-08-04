"""W08-04 对 63 条 train Evidence 的局部 revision 安全字段审计。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.experiments.ph2_w08_discourse_training import (
    W08DiscourseTrainingError,
    audit_w08_discourse_training,
)
from pure_integer_ai.experiments.ph2_w08_payload import W08TrainingPayload
from pure_integer_ai.experiments.ph2_w08_recompute_contract import (
    W08_REVISION_MAPPING_SHAPES,
)


class W08RecomputeTrainingError(ValueError):
    """局部 revision train schema、影响集或安全 bit 不闭合。"""


@dataclass(frozen=True)
class W08RecomputeTrainingAudit:
    observation_count: int
    evidence_binding_count: int
    discourse_revision_count: int
    parser_revision_plan_count: int
    reference_plan_count: int
    impacted_reference_query_count: int
    source_conflict_count: int
    later_correction_count: int
    affected_occurrence_count: int
    unaffected_occurrence_count: int
    recompute_query_count: int
    observed_mapping_shapes: tuple[str, ...]
    parent_mapping_shapes: tuple[str, ...]
    old_occurrence_rewrite_count: int
    unaffected_recompute_count: int
    whole_document_recompute_count: int
    fifo_authority_count: int
    surface_cue_authority_count: int
    authored_answer_read_count: int

    def __post_init__(self) -> None:
        count_fields = tuple(
            getattr(self, name)
            for name in self.__dataclass_fields__
            if not name.endswith("mapping_shapes")
        )
        if any(type(item) is not int or item < 0 for item in count_fields):
            raise W08RecomputeTrainingError("recompute train count is invalid")
        if self.observation_count != self.evidence_binding_count:
            raise W08RecomputeTrainingError("recompute Observation/Evidence drifted")
        if min(
            self.discourse_revision_count,
            self.parser_revision_plan_count,
            self.reference_plan_count,
            self.impacted_reference_query_count,
            self.source_conflict_count,
            self.later_correction_count,
            self.affected_occurrence_count,
            self.unaffected_occurrence_count,
            self.recompute_query_count,
        ) <= 0:
            raise W08RecomputeTrainingError("recompute train coverage is incomplete")
        if (
            not isinstance(self.observed_mapping_shapes, tuple)
            or not self.observed_mapping_shapes
            or any(
                item not in W08_REVISION_MAPPING_SHAPES
                for item in self.observed_mapping_shapes
            )
        ):
            raise W08RecomputeTrainingError("observed mapping shapes are invalid")
        if self.parent_mapping_shapes != W08_REVISION_MAPPING_SHAPES:
            raise W08RecomputeTrainingError("A-08 parent mapping inventory drifted")
        if any((
            self.old_occurrence_rewrite_count,
            self.unaffected_recompute_count,
            self.whole_document_recompute_count,
            self.fifo_authority_count,
            self.surface_cue_authority_count,
            self.authored_answer_read_count,
        )):
            raise W08RecomputeTrainingError("recompute train used a forbidden shortcut")


def _list(value: object, *, where: str) -> list:
    if not isinstance(value, list):
        raise W08RecomputeTrainingError(f"{where} must be a list")
    return value


def audit_w08_recompute_training(
    payload: W08TrainingPayload,
) -> W08RecomputeTrainingAudit:
    """只读 compiled revision/reference plan，不访问 authored expected/label。"""
    try:
        discourse = audit_w08_discourse_training(payload)
    except W08DiscourseTrainingError as error:
        raise W08RecomputeTrainingError("recompute payload audit failed") from error

    parser_plans = 0
    reference_plans = 0
    impacted_reference_queries = 0
    source_conflicts = 0
    later_corrections = 0
    affected_occurrences = 0
    unaffected_occurrences = 0
    recompute_queries = 0
    shapes = set()
    old_rewrites = 0
    unaffected_recomputes = 0
    whole_document_recomputes = 0
    fifo_authority = 0
    surface_authority = 0
    for observation in payload.observations:
        if observation.payload_kind != "DiscourseRevisionQuery":
            continue
        value = observation.typed_payload.to_value()
        variant = value.get("variant_kind")
        source_conflicts += variant == "SOURCE_CONFLICT"
        later_corrections += variant == "LATER_REINTERPRETATION"
        old_rewrites += value.get("old_occurrences_rewritten", 0)
        unaffected_recomputes += value.get("unaffected_recomputed", 0)
        whole_document_recomputes += value.get("whole_document_recomputed", 0)
        fifo_authority += value.get("bare_reference_fifo_authoritative", 0)
        surface_authority += value.get("surface_cue_authoritative", 0)

        reference = value.get("reference_plan")
        if reference is not None:
            if not isinstance(reference, dict):
                raise W08RecomputeTrainingError("reference plan type is invalid")
            reference_plans += 1
            impacted_reference_queries += len(_list(
                reference.get("impacted_query_ids"),
                where="reference impacted queries",
            ))

        parser = value.get("parser_revision_plan")
        if parser is None:
            continue
        if not isinstance(parser, dict):
            raise W08RecomputeTrainingError("parser revision plan type is invalid")
        parser_plans += 1
        affected = _list(
            parser.get("affected_occurrences"),
            where="parser affected occurrences",
        )
        unaffected = _list(
            parser.get("unaffected_occurrences"),
            where="parser unaffected occurrences",
        )
        queries = _list(
            parser.get("recompute_query_ids"),
            where="parser recompute queries",
        )
        affected_occurrences += len(affected)
        unaffected_occurrences += len(unaffected)
        recompute_queries += len(queries)
        single_targets: dict[tuple[int, ...], int] = {}
        for mapping in affected:
            if not isinstance(mapping, dict):
                raise W08RecomputeTrainingError("parser mapping type is invalid")
            targets = tuple(
                tuple(item)
                for item in _list(
                    mapping.get("replacement_occurrence_keys"),
                    where="parser mapping replacements",
                )
            )
            if not targets:
                shapes.add("OLD_TO_ZERO")
            elif len(targets) == 1:
                shapes.add("OLD_TO_ONE")
                single_targets[targets[0]] = single_targets.get(targets[0], 0) + 1
            else:
                shapes.add("ONE_TO_MANY")
        if any(count > 1 for count in single_targets.values()):
            shapes.add("MANY_TO_ONE")

    return W08RecomputeTrainingAudit(
        discourse.observation_count,
        discourse.evidence_binding_count,
        discourse.discourse_revision_count,
        parser_plans,
        reference_plans,
        impacted_reference_queries,
        source_conflicts,
        later_corrections,
        affected_occurrences,
        unaffected_occurrences,
        recompute_queries,
        tuple(sorted(shapes)),
        W08_REVISION_MAPPING_SHAPES,
        old_rewrites,
        unaffected_recomputes,
        whole_document_recomputes,
        fifo_authority,
        surface_authority,
        0,
    )


__all__ = [
    "W08RecomputeTrainingAudit",
    "W08RecomputeTrainingError",
    "audit_w08_recompute_training",
]
