"""FT16 已学问题特征注册表的全局阶段分派。"""
from __future__ import annotations

from pure_integer_ai.experiments.ph2_w03_w04_w05_question_feature_catalog import (
    run_raw_question_feature_answer,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_question_feature_registry_contract import (
    QUESTION_FEATURE_DISPATCH_PHASES,
    QUESTION_FEATURE_INTERPRETATION_STATUSES,
    QUESTION_FEATURE_REGISTRY_EXPRESSION_BOUNDARY,
    QUESTION_FEATURE_REGISTRY_SHA256,
    RawQuestionFeatureDispatchTrace,
    RawQuestionFeatureInterpretation,
    RawQuestionFeatureRegistry,
    RawQuestionFeatureRegistryAnswerResult,
    RawQuestionFeatureRegistryEntry,
    W03W04W05QuestionFeatureRegistryError,
    build_raw_question_feature_registry,
    raw_question_feature_trace_interpretation,
    resolve_question_feature_interpretations,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question_alias import (
    run_question_feature_predicate_alias_answer,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question_contract import (
    RawQuestionRequest,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question_implicit import (
    run_implicit_predicate_question_answer,
)


def project_raw_question_feature_registry_phase(
        registry: RawQuestionFeatureRegistry,
        request: RawQuestionRequest,
        traces: tuple[RawQuestionFeatureDispatchTrace, ...],
        phase: str,
        ) -> RawQuestionFeatureRegistryAnswerResult | None:
    statuses = tuple(item.status_at(phase) for item in traces)
    if all(item == "UNKNOWN" for item in statuses):
        return None
    answered = tuple(
        item for item in traces if item.status_at(phase) == "ANSWER")
    interpretations = tuple(sorted(
        {
            value.sha256(): value
            for value in (
                raw_question_feature_trace_interpretation(item, phase)
                for item in answered)
        }.values(),
        key=RawQuestionFeatureInterpretation.sha256,
    ))
    resolution = resolve_question_feature_interpretations(interpretations)
    if "CLARIFY" in statuses or resolution == "AMBIGUOUS":
        return RawQuestionFeatureRegistryAnswerResult(
            registry.identity_sha256,
            request,
            "CLARIFY",
            None,
            phase,
            interpretations,
            None,
            traces,
        )
    if resolution != "SELECTED":
        raise W03W04W05QuestionFeatureRegistryError(
            "decisive registry phase lacks an answer interpretation")
    selected = min(answered, key=lambda item: item.entry_sha256)
    return RawQuestionFeatureRegistryAnswerResult(
        registry.identity_sha256,
        request,
        "ANSWER",
        selected.answer_at(phase),
        phase,
        interpretations,
        selected.entry_sha256,
        traces,
    )


def run_raw_question_feature_registry_answer(
        registry: RawQuestionFeatureRegistry,
        request: RawQuestionRequest,
        ) -> RawQuestionFeatureRegistryAnswerResult:
    """按全局阶段分派，绝不先执行单个目录的完整流水线。"""
    if (not isinstance(registry, RawQuestionFeatureRegistry)
            or not isinstance(request, RawQuestionRequest)):
        raise TypeError("question feature registry dispatch inputs are invalid")
    traces = tuple(
        RawQuestionFeatureDispatchTrace(
            entry.sha256(),
            run_raw_question_feature_answer(entry.feature_catalog, request),
            None,
            None,
        )
        for entry in registry.entries
    )
    decision = project_raw_question_feature_registry_phase(
        registry, request, traces, "EXACT")
    if decision is not None:
        return decision

    traces = tuple(
        RawQuestionFeatureDispatchTrace(
            entry.sha256(),
            prior.exact_result,
            run_question_feature_predicate_alias_answer(
                entry.alias_bridge,
                entry.feature_catalog,
                request,
            ),
            None,
        )
        for entry, prior in zip(registry.entries, traces)
    )
    decision = project_raw_question_feature_registry_phase(
        registry, request, traces, "ALIAS")
    if decision is not None:
        return decision

    traces = tuple(
        RawQuestionFeatureDispatchTrace(
            entry.sha256(),
            prior.exact_result,
            prior.alias_result,
            run_implicit_predicate_question_answer(
                entry.alias_bridge,
                entry.implicit_bundle,
                entry.feature_catalog.w03_batch,
                entry.feature_catalog.w04_batch,
                entry.feature_catalog.w05_batch,
                request,
                overlay_validation_sha256=(
                    entry.feature_catalog.overlay_validation_sha256),
            ),
        )
        for entry, prior in zip(registry.entries, traces)
    )
    decision = project_raw_question_feature_registry_phase(
        registry, request, traces, "IMPLICIT")
    if decision is not None:
        return decision
    return RawQuestionFeatureRegistryAnswerResult(
        registry.identity_sha256,
        request,
        "UNKNOWN",
        None,
        None,
        (),
        None,
        traces,
    )


__all__ = [
    "QUESTION_FEATURE_DISPATCH_PHASES",
    "QUESTION_FEATURE_INTERPRETATION_STATUSES",
    "QUESTION_FEATURE_REGISTRY_EXPRESSION_BOUNDARY",
    "QUESTION_FEATURE_REGISTRY_SHA256",
    "RawQuestionFeatureDispatchTrace",
    "RawQuestionFeatureInterpretation",
    "RawQuestionFeatureRegistry",
    "RawQuestionFeatureRegistryAnswerResult",
    "RawQuestionFeatureRegistryEntry",
    "W03W04W05QuestionFeatureRegistryError",
    "build_raw_question_feature_registry",
    "project_raw_question_feature_registry_phase",
    "resolve_question_feature_interpretations",
    "run_raw_question_feature_registry_answer",
]
