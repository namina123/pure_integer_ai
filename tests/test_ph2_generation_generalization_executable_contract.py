"""E-00 actual generation/readback/六路独立 verifier 聚焦合同。"""
from __future__ import annotations

from dataclasses import replace

import pytest

from pure_integer_ai.cognition.shared.generation_verification import (
    GenerationSurfaceParseRequest,
)
from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    LosslessIntegerKey,
)
from pure_integer_ai.experiments.ph2_generation_generalization_contract import (
    INDEPENDENT_VERIFIER_REQUIREMENTS,
)
from pure_integer_ai.experiments.ph2_generation_generalization_executable_contract import (
    GenerationGeneralizationExecutableContractError,
    GenerationGeneralizationExecutableEvidence,
    GenerationGeneralizationIndependentVerification,
)
from pure_integer_ai.experiments.verification_orchestration import (
    VERDICT_REFUTE,
    VERDICT_UNKNOWN,
    VerificationReport,
)

from tests.test_ph2_grounded_answer_reference_compile import (
    _run_reference_strategy,
)


def _result_for_route(report, route):
    """从只读 report 精确回读一个 dimension/verifier 结果。"""
    return next(
        item for item in report.results
        if item.dimension == route.dimension and item.verifier == route.verifier)


def _replace_report_result(reports, prior, revised):
    """在保持 report 只读和排序的前提下替换一个测试结果。"""
    return tuple(
        VerificationReport(
            report.read_only,
            tuple(revised if item == prior else item for item in report.results),
        )
        if prior in report.results else report
        for report in reports
    )


def test_executable_contract_binds_actual_run_and_keeps_hard_conjunction():
    """真实 output/readback 可 PASS；refute、缺路和广播分别收紧。"""
    completed = _run_reference_strategy("ANTECEDENT_REFERENCE")
    uses = completed[1]
    gg02 = completed[4]
    run = uses.reference.run
    assert run.generation is not None and run.postcheck is not None
    choice = uses.reference.choice_after
    use = uses.reference.use
    parse_request = GenerationSurfaceParseRequest.from_execution(
        run.generation)
    reference_protocol = gg02.verification.protocol.by_name()
    reference_report = gg02.verification.report
    postcheck = run.postcheck
    postcheck_results = {
        dimension: next(
            item for item in postcheck.report.results
            if item.dimension == dimension)
        for dimension in postcheck.protocol.dimensions()
    }
    mapped_results = {
        "ADDRESSEE_RECOVERABILITY": _result_for_route(
            reference_report,
            reference_protocol["REFERENCE_UNIQUE_RESOLUTION"].route,
        ),
        "COMMUNICATIVE_TASK": _result_for_route(
            reference_report,
            reference_protocol["TASK_STANCE_EXECUTION"].route,
        ),
        "INDEPENDENT_UNDERSTANDING_READBACK": postcheck_results[
            postcheck.protocol.proposition_dimension],
        "LEGAL_OBJECT_COMPOSITION": _result_for_route(
            reference_report,
            reference_protocol["CONTENT_CANDIDATE_COVERAGE"].route,
        ),
        "SOURCE_UNCERTAINTY_CITATION": postcheck_results[
            postcheck.protocol.source_dimension],
        "STRUCTURE_SLOT_ORDER": _result_for_route(
            reference_report,
            reference_protocol["STRUCTURE_EXECUTION"].route,
        ),
    }
    verifications = tuple(
        GenerationGeneralizationIndependentVerification(
            requirement,
            LosslessIntegerKey(mapped_results[requirement].claim_keys[0]),
            mapped_results[requirement],
        )
        for requirement in INDEPENDENT_VERIFIER_REQUIREMENTS
    )
    reports = (postcheck.report, reference_report)
    evidence = GenerationGeneralizationExecutableEvidence(
        choice,
        use,
        run.generation,
        parse_request,
        postcheck,
        reports,
        verifications,
        (21200, 1),
    )

    assert evidence.runtime_status == "PASS_EXECUTABLE_LAYER_CONJUNCTION"
    assert evidence.ready_for_label_comparison == 1
    assert evidence.stable_key() == evidence.stable_key()

    missing = replace(evidence, verifications=verifications[:-1])
    assert missing.runtime_status == "NE_INDEPENDENT_LAYER_INPUT_MISSING"
    assert missing.ready_for_label_comparison == 0

    prior = verifications[0]
    refuted_result = replace(prior.result, verdict=VERDICT_REFUTE)
    refuted = replace(
        evidence,
        verification_reports=_replace_report_result(
            reports, prior.result, refuted_result),
        verifications=(
            replace(prior, result=refuted_result),
            *verifications[1:],
        ),
    )
    assert refuted.runtime_status == "FAIL_INDEPENDENT_LAYER_CONJUNCTION"

    unknown_result = replace(prior.result, verdict=VERDICT_UNKNOWN)
    unknown = replace(
        evidence,
        verification_reports=_replace_report_result(
            reports, prior.result, unknown_result),
        verifications=(
            replace(prior, result=unknown_result),
            *verifications[1:],
        ),
    )
    assert unknown.runtime_status == (
        "NE_INDEPENDENT_LAYER_INPUT_INDETERMINATE")

    with pytest.raises(
            GenerationGeneralizationExecutableContractError,
            match="input 或 route 发生广播"):
        replace(
            evidence,
            verifications=(
                verifications[0],
                replace(
                    verifications[1],
                    input_key=verifications[0].input_key,
                    result=verifications[0].result,
                ),
                *verifications[2:],
            ),
        )
