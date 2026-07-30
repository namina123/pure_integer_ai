"""PH2 W-03 target Sense 到 surface 的生成闭环。"""
from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    GenerationChoiceOutcomeRef,
    GenerationChoiceUseRef,
    LosslessIntegerKey,
)
from pure_integer_ai.experiments.ph2_w03_generation import (
    W03_GENERATION_ADOPTED,
    W03_GENERATION_CLARIFY,
    W03_GENERATION_HARD_CASES,
    W03_GENERATION_OUTCOME_NEUTRAL,
    W03_GENERATION_OUTCOME_REFUTE,
    W03_GENERATION_OUTCOME_SUPPORT,
    W03_GENERATION_READY,
    W03_GENERATION_REJECTED,
    W03_GENERATION_UNKNOWN,
    W03ExpressionConstraints,
    W03GenerationCaseResult,
    W03GenerationError,
    W03GenerationRequest,
    build_w03_generation_runtime,
    run_w03_generation_hard_conjunct,
)
from tests.test_w03_stage2_understanding import _authored_evidence, _runtime


def _surface(runtime, value: str):
    """返回唯一正式 surface 候选，不按 stable order 猜测。"""
    matches = tuple(
        item for item in runtime.output.candidates
        if item.anchor.extracted.surface == value
    )
    if len(matches) != 1:
        raise AssertionError(f"缺唯一测试 surface: {value}")
    return matches[0]


def _request(candidate, *, seed: int, context=...):
    """只从 typed target identity 构造请求，不携带 expected surface。"""
    selected_context = candidate.context if context is ... else context
    return W03GenerationRequest(
        LosslessIntegerKey((30304, seed)),
        candidate.sense,
        candidate.concept,
        selected_context,
        candidate.anchor.branch,
        W03ExpressionConstraints(True, True, 64),
        candidate.source_ref,
        document_scope(candidate.source_ref),
    )


def test_target_sense_exposes_multiple_legal_surfaces_without_unique_expected():
    """同 concept/context/branch 的两个 active surface 都合法且 Sense 不合并。"""
    backend, understanding = _runtime()
    try:
        understanding.apply_all_evidence()
        runtime = build_w03_generation_runtime(understanding)
        target = _surface(understanding, "把门打开")

        choice = runtime.choose(_request(target, seed=1))

        assert choice.status == W03_GENERATION_READY
        assert choice.selected is None
        assert {item.surface for item in choice.options} == {
            "把门打开", "把门再次打开",
        }
        assert len({item.sense for item in choice.options}) == 2
        assert {item.concept for item in choice.options} == {target.concept}
        assert {item.context for item in choice.options} == {target.context}
        assert {item.branch for item in choice.options} == {
            target.anchor.branch,
        }
        assert "expected_surface" not in {
            item.name for item in fields(W03GenerationRequest)
        }
    finally:
        backend.close()


def test_homograph_identity_and_missing_context_fail_closed_without_sort_first():
    """银行多义保留独立 Sense；缺 context 时澄清且不返回排序首项。"""
    backend, understanding = _runtime()
    try:
        understanding.apply_evidence(_authored_evidence(1))
        understanding.apply_evidence(_authored_evidence(2))
        finance = understanding.candidate_for_observation(
            _authored_evidence(1).observation.stable_key)[0]
        river = understanding.candidate_for_observation(
            _authored_evidence(2).observation.stable_key)[0]
        runtime = build_w03_generation_runtime(understanding)

        clarify = runtime.choose(_request(finance, seed=2, context=None))
        exact = runtime.choose(_request(finance, seed=3))

        assert finance.anchor.extracted.surface == river.anchor.extracted.surface
        assert finance.sense != river.sense
        assert finance.concept != river.concept
        assert clarify.status == W03_GENERATION_CLARIFY
        assert clarify.options == ()
        assert clarify.selected is None
        assert exact.status == W03_GENERATION_READY
        assert {item.sense for item in exact.options} == {finance.sense}
        assert river.sense not in {item.sense for item in exact.options}
    finally:
        backend.close()


def test_choice_adoption_rejection_use_and_outcome_are_independent_ledgers():
    """每个 option 有独立 decision/Use/outcome，不按整句统一奖惩。"""
    backend, understanding = _runtime()
    try:
        understanding.apply_all_evidence()
        runtime = build_w03_generation_runtime(understanding)
        target = _surface(understanding, "把门打开")
        choice = runtime.choose(_request(target, seed=4))
        adopted_key = next(
            item.stable_key() for item in choice.options
            if item.surface == "把门打开"
        )

        uses = runtime.adopt(choice, (adopted_key,))
        outcomes = tuple(runtime.verify_use(item) for item in uses)

        assert tuple(item.action for item in runtime.decisions) == (
            W03_GENERATION_ADOPTED,
            W03_GENERATION_REJECTED,
        )
        assert len(runtime.choices) == 1
        assert len(runtime.decisions) == len(choice.options) == len(runtime.uses)
        assert all(isinstance(item.ref, GenerationChoiceUseRef) for item in uses)
        assert all(
            item.ref.selection_key.components == item.decision.option.stable_key()
            for item in uses
        )
        assert {item.verdict for item in outcomes} == {
            W03_GENERATION_OUTCOME_SUPPORT,
            W03_GENERATION_OUTCOME_NEUTRAL,
        }
        assert all(
            isinstance(item.ref, GenerationChoiceOutcomeRef)
            and item.ref.use_key == item.use.ref.use_key
            for item in outcomes
        )
        assert len({item.ref.outcome_key for item in outcomes}) == len(outcomes)
    finally:
        backend.close()


def test_supersede_removes_old_surface_and_reverification_refutes_old_use():
    """旧采用历史保留，但 revision 后旧 Sense 不再可生成且旧 Use 转 refute。"""
    backend, understanding = _runtime()
    try:
        old_binding = _authored_evidence(1)
        revision_binding = _authored_evidence(4)
        old = understanding.candidate_for_observation(
            old_binding.observation.stable_key)[0]
        revision = understanding.candidate_for_observation(
            revision_binding.observation.stable_key)[0]
        understanding.apply_evidence(old_binding)
        runtime = build_w03_generation_runtime(understanding)
        before = runtime.choose(_request(old, seed=5))
        old_use = runtime.adopt(
            before, (before.options[0].stable_key(),))[0]
        supported = runtime.verify_use(old_use)

        understanding.apply_evidence(revision_binding)
        after_old = runtime.choose(_request(old, seed=6))
        after_revision = runtime.choose(_request(revision, seed=7))
        refuted = runtime.verify_use(old_use)

        assert supported.verdict == W03_GENERATION_OUTCOME_SUPPORT
        assert after_old.status == W03_GENERATION_UNKNOWN
        assert after_old.options == ()
        assert after_revision.status == W03_GENERATION_READY
        assert {item.sense for item in after_revision.options} == {
            revision.sense,
        }
        assert refuted.verdict == W03_GENERATION_OUTCOME_REFUTE
        assert supported.ref.outcome_key != refuted.ref.outcome_key
        assert supported.use.ref.use_key == refuted.use.ref.use_key
        assert tuple(item.verdict for item in runtime.outcomes) == (
            W03_GENERATION_OUTCOME_SUPPORT,
            W03_GENERATION_OUTCOME_REFUTE,
        )
    finally:
        backend.close()


def test_generation_hard_conjunct_is_one_of_one_and_ablations_are_fail_not_ne():
    """五类独立 case 全过才 1/1；两个真实连接任一关闭均硬失败。"""
    cases = tuple(
        W03GenerationCaseResult(name, True, LosslessIntegerKey((30304, 40 + i)))
        for i, name in enumerate(W03_GENERATION_HARD_CASES)
    )

    passed = run_w03_generation_hard_conjunct(
        cases,
        sense_consumer_connected=True,
        choice_bridge_connected=True,
    )
    sense_disabled = run_w03_generation_hard_conjunct(
        cases,
        sense_consumer_connected=False,
        choice_bridge_connected=True,
    )
    bridge_disabled = run_w03_generation_hard_conjunct(
        cases,
        sense_consumer_connected=True,
        choice_bridge_connected=False,
    )

    assert (passed.passed, passed.required, passed.fail_count, passed.ne_count) == (
        1, 1, 0, 0,
    )
    assert (sense_disabled.passed, sense_disabled.fail_count,
            sense_disabled.ne_count) == (0, 1, 0)
    assert (bridge_disabled.passed, bridge_disabled.fail_count,
            bridge_disabled.ne_count) == (0, 1, 0)
    assert passed.status == "PASS"
    assert sense_disabled.status == bridge_disabled.status == "FAIL"


def test_generation_rejects_foreign_option_duplicate_request_and_disabled_bridge():
    """跨 choice option、重复 request 和关闭 bridge 均在 ledger 写前拒绝。"""
    backend, understanding = _runtime()
    try:
        understanding.apply_all_evidence()
        target = _surface(understanding, "把门打开")
        runtime = build_w03_generation_runtime(understanding)
        choice = runtime.choose(_request(target, seed=8))
        with pytest.raises(W03GenerationError, match="option"):
            runtime.adopt(choice, ((999, 0),))
        assert runtime.decisions == runtime.uses == ()
        with pytest.raises(W03GenerationError, match="重复"):
            runtime.choose(choice.request)

        disabled = build_w03_generation_runtime(
            understanding, choice_bridge_connected=False)
        disabled_choice = disabled.choose(_request(target, seed=9))
        assert disabled.adopt(
            disabled_choice,
            (disabled_choice.options[0].stable_key(),),
        ) == ()
        assert disabled.decisions == disabled.uses == disabled.outcomes == ()
    finally:
        backend.close()


def test_generation_source_has_no_expected_evaluator_legacy_or_filesystem_route():
    """W03-G 仅消费 typed inventory/active projection，不接 expected 或旧表。"""
    repository = Path(__file__).resolve().parents[1]
    source = "".join(
        (repository / "src/pure_integer_ai/experiments" / name).read_text(
            encoding="utf-8")
        for name in (
            "ph2_w03_generation.py",
            "ph2_w03_generation_contract.py",
        )
    )

    assert "expected_surface" not in source
    assert "EvaluatorLabelRecord" not in source
    assert "language_sense_candidate_runtime" not in source
    assert "storage.sense_candidates" not in source
    assert "pathlib" not in source
    assert "transaction" not in source.lower()
    assert ".consumer.lookup(" in source
