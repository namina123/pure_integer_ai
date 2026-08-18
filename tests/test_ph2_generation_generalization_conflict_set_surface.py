"""Public typed generation/parser probes for GG03 CONFLICT_SET."""
from dataclasses import replace

from pure_integer_ai.experiments.ph2_generation_generalization_conflict_set_contract import (
    CONFLICT_SET_FAIL,
    CONFLICT_SET_NE,
    CONFLICT_SET_PASS,
    ConflictSetEvidence,
    build_conflict_set_plan,
)
from pure_integer_ai.experiments.ph2_generation_generalization_conflict_set_surface import (
    classify_conflict_set_surface,
    generate_conflict_set_sentences,
)


def _plan():
    return build_conflict_set_plan(
        scope_id=901,
        claim_ids=("claim-b", "claim-a"),
        evidence=(
            ConflictSetEvidence("e1", "claim-a", "source-b", 901, 1, 0),
            ConflictSetEvidence("e2", "claim-a", "source-a", 901, 0, 1),
            ConflictSetEvidence("e3", "claim-b", "source-c", 901, 1, 1),
            ConflictSetEvidence("e4", "claim-b", "source-d", 901, 0, 1),
        ),
    )


def test_conflict_set_typed_generation_and_parser_pass() -> None:
    plan = _plan()
    sentences = generate_conflict_set_sentences(
        plan, ("命题B存在来源冲突。", "命题A存在来源冲突。"))
    result = classify_conflict_set_surface(plan, sentences)
    assert result.status == CONFLICT_SET_PASS
    assert result.projection == plan.projection
    assert result.sentence_count == 2


def test_conflict_set_parser_distinguishes_order_and_missing_sentence() -> None:
    plan = _plan()
    sentences = generate_conflict_set_sentences(
        plan, ("命题B存在来源冲突。", "命题A存在来源冲突。"))
    reversed_sentences = tuple(reversed(sentences))
    assert classify_conflict_set_surface(
        plan, reversed_sentences).status == CONFLICT_SET_NE

    wrong_order = (
        replace(sentences[0], claim_id="claim-a"),
        replace(sentences[1], claim_id="claim-b"),
    )
    assert classify_conflict_set_surface(
        plan, wrong_order).status == CONFLICT_SET_FAIL
    source_loss = replace(sentences[0], source_ids=("source-a",))
    assert classify_conflict_set_surface(
        plan, (source_loss, sentences[1])).status == CONFLICT_SET_FAIL
    assert classify_conflict_set_surface(plan, None).status == CONFLICT_SET_NE
