"""E-02/E-03 六路 TRAIN readback、exact evidence 与状态分型专项。"""
from __future__ import annotations

from dataclasses import replace

from pure_integer_ai.experiments.ph2_generation_generalization_executable_train_rehearsal import (
    GenerationGeneralizationTrainRehearsal,
)
from pure_integer_ai.experiments.ph2_generation_generalization_train_closure import (
    READBACK_COVERAGE_KINDS,
    build_generation_generalization_train_course_closure,
)
from pure_integer_ai.experiments.verification_orchestration import (
    VERDICT_REFUTE,
)

from tests.test_ph2_generation_generalization_reference_rehearsal import (
    _rehearse_reference_requirement,
)
from tests.test_ph2_grounded_answer_connector_runtime import (
    _rehearse_answer_requirement,
)
from tests.test_ph2_grounded_answer_response_act_runtime import (
    _rehearse_response_act,
)


def _full_rehearsal() -> GenerationGeneralizationTrainRehearsal:
    """运行冻结六项 TRAIN case，并按 catalog 顺序返回 actual rehearsal。"""
    course, _planning, addressee, _run, _structure, _source = (
        _rehearse_reference_requirement("ADDRESSEE_RECOVERABILITY", 21))
    _, _, communicative, _ = _rehearse_response_act(
        "COMMUNICATIVE_TASK", "CLARIFY", 21)
    _, _, readback, _ = _rehearse_answer_requirement(
        "INDEPENDENT_UNDERSTANDING_READBACK", 21)
    _, _, legal, _ = _rehearse_answer_requirement(
        "LEGAL_OBJECT_COMPOSITION", 22)
    _, _, source, _ = _rehearse_response_act(
        "SOURCE_UNCERTAINTY_CITATION", "CONFLICT", 22)
    _, _planning, structure, _run, _structure, _source = (
        _rehearse_reference_requirement("STRUCTURE_SLOT_ORDER", 22))
    return GenerationGeneralizationTrainRehearsal(
        course,
        (addressee, communicative, readback, legal, source, structure),
    )


def _with_first_requirement_refuted(
        rehearsal: GenerationGeneralizationTrainRehearsal,
        ) -> GenerationGeneralizationTrainRehearsal:
    """只改写首项独立 result/report verdict，保留其 exact claim 与 route。"""
    item = rehearsal.items[0]
    old_result = item.verification.result
    new_result = replace(old_result, verdict=VERDICT_REFUTE)
    reports = tuple(
        replace(
            report,
            results=tuple(
                new_result if result == old_result else result
                for result in report.results),
        )
        for report in item.verification_reports
    )
    failed_item = replace(
        item,
        verification_reports=reports,
        verification=replace(item.verification, result=new_result),
    )
    return GenerationGeneralizationTrainRehearsal(
        rehearsal.course,
        (failed_item, *rehearsal.items[1:]),
    )


def test_train_course_closure_covers_readback_and_distinguishes_fail_ne():
    """完整六路 PASS；独立 refute 为 FAIL；缺第六项为 NE。"""
    rehearsal = _full_rehearsal()
    closure = build_generation_generalization_train_course_closure(rehearsal)

    assert closure.status == "PASS_TRAIN_COURSE_CLOSURE"
    assert closure.complete == 1
    assert closure.readback_coverage == READBACK_COVERAGE_KINDS
    assert all(audit.readback_complete for audit in closure.audits)
    assert all(audit.claim_keys for audit in closure.audits)
    assert all(audit.evidence_keys for audit in closure.audits)
    assert len({
        audit.item.execution.stable_key() for audit in closure.audits}) == 6

    failed = build_generation_generalization_train_course_closure(
        _with_first_requirement_refuted(rehearsal))
    assert failed.status == "FAIL_TRAIN_VERIFIER_CONJUNCTION"
    assert failed.complete == 0

    partial = GenerationGeneralizationTrainRehearsal(
        rehearsal.course, rehearsal.items[:-1])
    missing = build_generation_generalization_train_course_closure(partial)
    assert missing.status == "NE_TRAIN_REQUIREMENT_INPUT_MISSING"
    assert missing.complete == 0
