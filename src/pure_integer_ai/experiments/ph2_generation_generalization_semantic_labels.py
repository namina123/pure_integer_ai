"""GG-03 open-surface semantic labels and candidate-independent projections."""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
    exact_dict,
    sha256_text,
)
from pure_integer_ai.experiments.ph2_generation_generalization_contract import (
    INDEPENDENT_VERIFIER_REQUIREMENTS,
)
from pure_integer_ai.experiments.ph2_generation_generalization_evaluation_family_identity import (
    GenerationGeneralizationEvaluationFamilyError,
    generation_generalization_sha256_bytes,
)
from pure_integer_ai.experiments.ph2_generation_generalization_evaluation_observation import (
    GenerationGeneralizationEvaluationObservation,
)
from pure_integer_ai.experiments.ph2_generation_generalization_evaluation_runner import (
    GenerationGeneralizationEvaluationActualRun,
    generation_generalization_evaluation_requirements,
)
from pure_integer_ai.experiments.ph2_grounded_answer_course import (
    CARRIER_KINDS,
    RESPONSE_ACTS,
    SurfaceRealization,
)
from pure_integer_ai.experiments.ph2_grounded_response_act_planning import (
    GroundedResponseActPlanningBuild,
    compile_grounded_answer_planning,
    compile_grounded_answer_reference_planning,
    compile_grounded_response_act_planning,
)


SEMANTIC_LABEL_ARTIFACT_KIND = "PH2_GG03_FORMAL_SEMANTIC_LABEL_V2"
SEMANTIC_PROJECTION_KIND = "PH2_GG03_SURFACE_SEMANTIC_PROJECTION_V1"
SEMANTIC_PROJECTION_FIELDS = (
    "carrier_kind",
    "response_act",
    "scope_id",
    "claim_ids",
    "cited_source_ids",
)
_PROJECTION_RECORD_FIELDS = frozenset({
    "artifact_kind", "carrier_kind", "cited_source_ids", "claim_ids",
    "format_version", "response_act", "scope_id",
})
_LABEL_RECORD_FIELDS = frozenset({
    "artifact_kind", "expected_semantic_sha256", "format_version",
    "observation_stable_key_sha256", "requirements", "split",
})


def _text_ids(
        values: tuple[str, ...], *, where: str, ordered: bool,
        ) -> tuple[str, ...]:
    if (not isinstance(values, tuple)
            or any(not isinstance(item, str) or not item
                   or item.strip() != item for item in values)
            or len(values) != len(set(values))):
        raise GenerationGeneralizationEvaluationFamilyError(
            f"{where} 非法或重复")
    if not ordered and values != tuple(sorted(values)):
        raise GenerationGeneralizationEvaluationFamilyError(
            f"{where} 必须 canonical 排序")
    return values


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True, order=True)
class GenerationGeneralizationSemanticProjection:
    """A surface-independent answer meaning recovered from visible output."""

    carrier_kind: str
    response_act: str
    scope_id: int
    claim_ids: tuple[str, ...]
    cited_source_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.carrier_kind not in CARRIER_KINDS:
            raise GenerationGeneralizationEvaluationFamilyError(
                "GG-03 semantic carrier kind 未注册")
        if self.response_act not in RESPONSE_ACTS:
            raise GenerationGeneralizationEvaluationFamilyError(
                "GG-03 semantic response act 未注册")
        if type(self.scope_id) is not int or self.scope_id <= 0:
            raise GenerationGeneralizationEvaluationFamilyError(
                "GG-03 semantic scope 必须为正整数")
        _text_ids(self.claim_ids, where="GG-03 semantic claim ids", ordered=True)
        _text_ids(
            self.cited_source_ids,
            where="GG-03 semantic cited source ids",
            ordered=False,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_kind": SEMANTIC_PROJECTION_KIND,
            "carrier_kind": self.carrier_kind,
            "cited_source_ids": list(self.cited_source_ids),
            "claim_ids": list(self.claim_ids),
            "format_version": 1,
            "response_act": self.response_act,
            "scope_id": self.scope_id,
        }

    def sha256(self) -> str:
        return generation_generalization_sha256_bytes(
            canonical_json_bytes(self.to_dict()))

    @classmethod
    def from_dict(
            cls, value: object,
            ) -> "GenerationGeneralizationSemanticProjection":
        raw = exact_dict(
            value, _PROJECTION_RECORD_FIELDS,
            where="GG-03 semantic projection",
        )
        if (raw["artifact_kind"] != SEMANTIC_PROJECTION_KIND
                or raw["format_version"] != 1
                or not isinstance(raw["claim_ids"], list)
                or not isinstance(raw["cited_source_ids"], list)):
            raise GenerationGeneralizationEvaluationFamilyError(
                "GG-03 semantic projection kind/version/list 漂移")
        return cls(
            str(raw["carrier_kind"]),
            str(raw["response_act"]),
            raw["scope_id"],
            tuple(str(item) for item in raw["claim_ids"]),
            tuple(str(item) for item in raw["cited_source_ids"]),
        )


def semantic_projection_from_realization(
        realization: SurfaceRealization,
        ) -> GenerationGeneralizationSemanticProjection:
    """Drop wording and realization identity while preserving typed meaning."""
    if not isinstance(realization, SurfaceRealization):
        raise TypeError("GG-03 semantic realization 类型错误")
    return GenerationGeneralizationSemanticProjection(
        realization.carrier_kind,
        realization.response_act,
        realization.scope_id,
        realization.claim_ids,
        tuple(sorted(realization.cited_source_ids)),
    )


def build_expected_generation_generalization_semantic_projection(
        observation: GenerationGeneralizationEvaluationObservation,
        *, carrier_kind: str = "PLAIN_TEXT",
        ) -> GenerationGeneralizationSemanticProjection:
    """Build the owner-side meaning commitment without selecting a candidate."""
    if not isinstance(
            observation, GenerationGeneralizationEvaluationObservation):
        raise TypeError("GG-03 semantic Observation 类型错误")
    plan = observation.question.answer_plan
    return GenerationGeneralizationSemanticProjection(
        carrier_kind,
        plan.response_act,
        observation.question.response_scope_id,
        plan.ordered_claim_ids,
        tuple(sorted(plan.citation_source_ids)),
    )


def _planning_for_actual(
        run: GenerationGeneralizationEvaluationActualRun,
        ) -> GroundedResponseActPlanningBuild:
    observation = run.observation
    branch = run.parse_request.branch
    if observation.reference_course is not None:
        return compile_grounded_answer_reference_planning(observation, branch)
    if observation.question.answer_plan.response_act == "ANSWER":
        return compile_grounded_answer_planning(observation, branch)
    return compile_grounded_response_act_planning(observation, branch)


def build_actual_generation_generalization_semantic_projection(
        run: GenerationGeneralizationEvaluationActualRun,
        *, carrier_kind: str = "PLAIN_TEXT",
        ) -> GenerationGeneralizationSemanticProjection | None:
    """Project parsed output into the same owner-side typed meaning domain."""
    if not isinstance(run, GenerationGeneralizationEvaluationActualRun):
        raise TypeError("GG-03 semantic actual run 类型错误")
    parsed = run.postcheck.parsed
    if not parsed.succeeded or parsed.observation is None:
        return None
    planning = _planning_for_actual(run)
    candidate_ids = {
        item.candidate.stable_key(): item.proposition_id
        for item in planning.candidate_bindings
    }
    source_ids = {
        item.source: item.source_id for item in planning.source_bindings
    }
    if run.execution.preview is None:
        return None
    sentences = run.execution.preview.request.structure.syntax.sentences
    emitted_keys = tuple(
        key for sentence in sentences for key in sentence.proposition_keys)
    recovered_keys = {
        item.candidate_key for item in parsed.observation.propositions}
    if set(emitted_keys) != recovered_keys:
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 semantic emitted/recovered candidate 漂移")
    try:
        claim_ids = tuple(candidate_ids[key] for key in emitted_keys)
        cited_source_ids = tuple(sorted(
            source_ids[source]
            for source in parsed.observation.cited_sources
        ))
    except KeyError as error:
        raise GenerationGeneralizationEvaluationFamilyError(
            "GG-03 semantic projection 含 planning 外身份") from error
    return GenerationGeneralizationSemanticProjection(
        carrier_kind,
        run.observation.question.answer_plan.response_act,
        parsed.observation.scope.local_id,
        claim_ids,
        cited_source_ids,
    )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True, order=True)
class GenerationGeneralizationSemanticLabelRecord:
    """Private expected-meaning commitment for one held-out Observation."""

    observation_stable_key_sha256: str
    expected_semantic_sha256: str
    requirements: tuple[str, ...]

    def __post_init__(self) -> None:
        sha256_text(
            self.observation_stable_key_sha256,
            where="GG-03 semantic label Observation SHA",
        )
        sha256_text(
            self.expected_semantic_sha256,
            where="GG-03 expected semantic SHA",
        )
        if (not isinstance(self.requirements, tuple) or not self.requirements
                or self.requirements != tuple(
                    item for item in INDEPENDENT_VERIFIER_REQUIREMENTS
                    if item in self.requirements)):
            raise GenerationGeneralizationEvaluationFamilyError(
                "GG-03 semantic label requirement 顺序非法")

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_kind": SEMANTIC_LABEL_ARTIFACT_KIND,
            "expected_semantic_sha256": self.expected_semantic_sha256,
            "format_version": 2,
            "observation_stable_key_sha256": (
                self.observation_stable_key_sha256),
            "requirements": list(self.requirements),
            "split": "held_out",
        }

    def verdict_for_projection(
            self,
            projection: GenerationGeneralizationSemanticProjection | None,
            ) -> str:
        """Return PASS for equality, FAIL for a different meaning, NE if absent."""
        if projection is None:
            return "NE"
        if not isinstance(
                projection, GenerationGeneralizationSemanticProjection):
            raise TypeError("GG-03 semantic predicted projection 类型错误")
        return (
            "PASS"
            if projection.sha256() == self.expected_semantic_sha256
            else "FAIL"
        )

    @classmethod
    def from_dict(
            cls, value: object,
            ) -> "GenerationGeneralizationSemanticLabelRecord":
        raw = exact_dict(
            value, _LABEL_RECORD_FIELDS,
            where="GG-03 semantic label",
        )
        if (raw["artifact_kind"] != SEMANTIC_LABEL_ARTIFACT_KIND
                or raw["format_version"] != 2
                or raw["split"] != "held_out"
                or not isinstance(raw["requirements"], list)):
            raise GenerationGeneralizationEvaluationFamilyError(
                "GG-03 semantic label kind/version/split 漂移")
        return cls(
            str(raw["observation_stable_key_sha256"]),
            str(raw["expected_semantic_sha256"]),
            tuple(str(item) for item in raw["requirements"]),
        )


def build_generation_generalization_semantic_label_record(
        observation: GenerationGeneralizationEvaluationObservation,
        *, carrier_kind: str = "PLAIN_TEXT",
        ) -> GenerationGeneralizationSemanticLabelRecord:
    """Build an owner commitment from meaning fields, never from candidate output."""
    expected = build_expected_generation_generalization_semantic_projection(
        observation, carrier_kind=carrier_kind)
    return GenerationGeneralizationSemanticLabelRecord(
        generation_generalization_sha256_bytes(canonical_json_bytes(
            list(observation.stable_key()))),
        expected.sha256(),
        generation_generalization_evaluation_requirements(observation),
    )


def generation_generalization_semantic_verdict_contract_sha256() -> str:
    """Freeze the V2 PASS/FAIL/NE rule independently from any wording."""
    return generation_generalization_sha256_bytes(canonical_json_bytes({
        "expected_label_kind": SEMANTIC_LABEL_ARTIFACT_KIND,
        "fail_condition": "PARSED_SEMANTIC_PROJECTION_DIFFERS",
        "ne_condition": "SEMANTIC_PROJECTION_UNAVAILABLE",
        "pass_condition": "PARSED_SEMANTIC_PROJECTION_EQUALS_EXPECTED",
        "projection_fields": list(SEMANTIC_PROJECTION_FIELDS),
        "projection_kind": SEMANTIC_PROJECTION_KIND,
        "status_precedence": ["FAIL", "NE", "PASS"],
        "version": 1,
    }))


__all__ = [
    "SEMANTIC_LABEL_ARTIFACT_KIND",
    "SEMANTIC_PROJECTION_FIELDS",
    "SEMANTIC_PROJECTION_KIND",
    "GenerationGeneralizationSemanticLabelRecord",
    "GenerationGeneralizationSemanticProjection",
    "build_actual_generation_generalization_semantic_projection",
    "build_expected_generation_generalization_semantic_projection",
    "build_generation_generalization_semantic_label_record",
    "generation_generalization_semantic_verdict_contract_sha256",
    "semantic_projection_from_realization",
]
