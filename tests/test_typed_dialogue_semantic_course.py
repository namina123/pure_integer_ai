"""Typed 对话课程进入正式 S-02 的定向回归。"""
from dataclasses import replace

from tests.test_l05b2b_semantic_course_runtime import _fixture, _protocol

from pure_integer_ai.experiments.language_semantic_runtime import (
    LanguageSemanticCourseRuntime,
)
from pure_integer_ai.experiments.ph2_dataset_contract import CanonicalJsonObject
from pure_integer_ai.experiments.typed_dialogue_semantic_course import (
    TypedDialogueSemanticMapper,
)


def _adoption_payload() -> CanonicalJsonObject:
    return CanonicalJsonObject.from_value({
        "candidate_propositions": [
            {"candidate_id": "a", "state": {"support": 1, "refute": 0}},
            {"candidate_id": "b", "state": {"support": 0, "refute": 1}},
        ],
    })


def test_typed_payload_forms_real_semantic_lesson_and_keeps_ordinary_surface_out():
    backend, ctx, _runtime, _mapper, item, payload, observed = _fixture()
    try:
        typed_runtime = LanguageSemanticCourseRuntime(
            ctx, _protocol(TypedDialogueSemanticMapper((21401, 99), 1)))
        typed_item = replace(
            item,
            typed_payload=_adoption_payload(),
            payload_kind="GenerationAdoptionPostcheckQuery",
        )
        run = typed_runtime.process(ctx, typed_item, payload, observed)
        assert run.decision.lesson is not None
        assert len(run.build.propositions) == 2
        assert len({item.evidence_id for item in run.evidence}) == 2
        # Semantic H-00/H-04 history must survive runtime reconstruction;
        # a resumed Stage 3 must not depend on the original Python ledger.
        restored_runtime = LanguageSemanticCourseRuntime(
            ctx, _protocol(TypedDialogueSemanticMapper((21401, 99), 1)))
        assert restored_runtime.ledger.state_key() == typed_runtime.ledger.state_key()
        ordinary = typed_runtime.process(ctx, item, payload, observed)
        assert ordinary.decision.lesson is None
        assert ordinary.request is None
    finally:
        backend.close()


def test_negative_generalization_payload_is_not_positive_lesson():
    backend, ctx, _runtime, _mapper, item, payload, observed = _fixture()
    try:
        typed_runtime = LanguageSemanticCourseRuntime(
            ctx, _protocol(TypedDialogueSemanticMapper((21401, 99), 1)))
        negative = CanonicalJsonObject.from_value({
            "sample_family": "NEGATIVE",
            "choice_candidates": [{"candidate_id": "x"}],
        })
        run = typed_runtime.process(
            ctx,
            replace(item, typed_payload=negative,
                    payload_kind="GenerationGeneralizationCandidateV1"),
            payload,
            observed,
        )
        assert run.decision.lesson is None
        assert run.request is None
    finally:
        backend.close()


def test_authored_adoption_conflict_is_not_positive_lesson():
    backend, ctx, _runtime, _mapper, item, payload, observed = _fixture()
    try:
        typed_runtime = LanguageSemanticCourseRuntime(
            ctx, _protocol(TypedDialogueSemanticMapper((21401, 99), 1)))
        conflict = CanonicalJsonObject.from_value({
            "task_kind": "ADOPTION",
            "candidate_propositions": [{
                "candidate_id": "conflicted",
                "state": {"support": 1, "refute": 1},
            }],
        })
        run = typed_runtime.process(
            ctx,
            replace(item, typed_payload=conflict,
                    payload_kind="GenerationAdoptionPostcheckQuery"),
            payload,
            observed,
        )
        assert run.decision.lesson is None
        assert run.request is None
    finally:
        backend.close()
