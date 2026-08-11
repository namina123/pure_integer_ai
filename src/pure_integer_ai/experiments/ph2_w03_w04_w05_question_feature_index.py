"""在冻结 FT16 已学注册表上执行 FT17 索引分派。"""
from __future__ import annotations

from bisect import bisect_left

from pure_integer_ai.experiments.ph2_w03_w04_w05_question_feature_catalog import (
    run_raw_question_feature_answer,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_question_feature_index_contract import (
    QUESTION_FEATURE_INDEX_EXPRESSION_BOUNDARY,
    QUESTION_FEATURE_INDEX_SHA256,
    RawQuestionFeatureAliasFrameRow,
    RawQuestionFeatureIndex,
    RawQuestionFeatureIndexPosting,
    RawQuestionFeatureIndexRow,
    W03W04W05QuestionFeatureIndexError,
    build_raw_question_feature_index,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_question_feature_registry import (
    project_raw_question_feature_registry_phase,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_question_feature_registry_contract import (
    QUESTION_FEATURE_DISPATCH_PHASES,
    RawQuestionFeatureDispatchTrace,
    RawQuestionFeatureRegistryAnswerResult,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question_alias import (
    continue_question_feature_predicate_alias_answer,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question_alias_contract import (
    RawQuestionPredicateAliasAnswerResult,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question_contract import (
    RawQuestionAnswerResult,
    RawQuestionRequest,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question_implicit import (
    RawQuestionImplicitPredicateAnswerResult,
    continue_implicit_predicate_question_answer,
)


def lookup_indexed_question_feature_entries(
        index: RawQuestionFeatureIndex,
        phase: str,
        request: RawQuestionRequest,
        ) -> tuple[str, ...]:
    """只返回一个全局阶段可到达的注册项。"""
    if (not isinstance(index, RawQuestionFeatureIndex)
            or not isinstance(request, RawQuestionRequest)):
        raise TypeError("indexed question feature lookup inputs are invalid")
    rows = index.rows_at(phase)
    row_ordinal = bisect_left(
        rows,
        request.question_surface,
        key=lambda item: item.question_surface,
    )
    direct = ()
    if (row_ordinal < len(rows)
            and rows[row_ordinal].question_surface
            == request.question_surface):
        direct = _posting_entries(
            rows[row_ordinal].postings,
            request.source_record_key,
        )
    if phase != "ALIAS":
        return direct
    structural = {
        entry_sha
        for row in index.alias_frame_rows
        if _matches_alias_frame(row, request.question_surface)
        for entry_sha in _posting_entries(
            row.postings,
            request.source_record_key,
        )
    }
    return tuple(sorted({*direct, *structural}))


def _posting_entries(
        postings: tuple[RawQuestionFeatureIndexPosting, ...],
        source_record_key: tuple[int, ...] | None,
        ) -> tuple[str, ...]:
    if source_record_key is None:
        return tuple(sorted({
            entry_sha
            for posting in postings
            for entry_sha in posting.entry_sha256s
        }))
    posting_ordinal = bisect_left(
        postings,
        source_record_key,
        key=lambda item: item.source_record_key,
    )
    if (posting_ordinal >= len(postings)
            or postings[posting_ordinal].source_record_key
            != source_record_key):
        return ()
    return postings[posting_ordinal].entry_sha256s


def _matches_alias_frame(
        row: RawQuestionFeatureAliasFrameRow,
        question_surface: str,
        ) -> bool:
    if (not question_surface.startswith(row.prefix_surface)
            or not question_surface.endswith(row.suffix_surface)
            or len(question_surface)
            <= len(row.prefix_surface) + len(row.suffix_surface)):
        return False
    end = (
        len(question_surface) - len(row.suffix_surface)
        if row.suffix_surface else None
    )
    alias_surface = question_surface[len(row.prefix_surface):end]
    return (
        bool(alias_surface)
        and alias_surface.strip() == alias_surface
        and alias_surface != row.predicate_surface
    )


def _unknown_exact(request: RawQuestionRequest) -> RawQuestionAnswerResult:
    return RawQuestionAnswerResult(
        request,
        "UNKNOWN",
        None,
        (),
        None,
        None,
    )


def _unknown_alias(
        request: RawQuestionRequest,
        exact_result: RawQuestionAnswerResult,
        ) -> RawQuestionPredicateAliasAnswerResult:
    return RawQuestionPredicateAliasAnswerResult(
        request,
        "UNKNOWN",
        None,
        exact_result,
        (),
        None,
        None,
    )


def _unknown_implicit(
        request: RawQuestionRequest,
        predicate_result: RawQuestionPredicateAliasAnswerResult,
        ) -> RawQuestionImplicitPredicateAnswerResult:
    implicit_result = _unknown_exact(request)
    return RawQuestionImplicitPredicateAnswerResult(
        request,
        "UNKNOWN",
        None,
        predicate_result,
        implicit_result,
    )


def run_indexed_raw_question_feature_registry_answer(
        index: RawQuestionFeatureIndex,
        request: RawQuestionRequest,
        ) -> RawQuestionFeatureRegistryAnswerResult:
    """只运行索引候选，同时保留完整 FT16 轨迹。"""
    if (not isinstance(index, RawQuestionFeatureIndex)
            or not isinstance(request, RawQuestionRequest)):
        raise TypeError("indexed question feature dispatch inputs are invalid")
    registry = index.registry
    candidates = set(lookup_indexed_question_feature_entries(
        index, "EXACT", request))
    traces = tuple(
        RawQuestionFeatureDispatchTrace(
            entry.sha256(),
            run_raw_question_feature_answer(entry.feature_catalog, request)
            if entry.sha256() in candidates else _unknown_exact(request),
            None,
            None,
        )
        for entry in registry.entries
    )
    decision = project_raw_question_feature_registry_phase(
        registry, request, traces, "EXACT")
    if decision is not None:
        return decision

    candidates = set(lookup_indexed_question_feature_entries(
        index, "ALIAS", request))
    traces = tuple(
        RawQuestionFeatureDispatchTrace(
            entry.sha256(),
            prior.exact_result,
            continue_question_feature_predicate_alias_answer(
                entry.alias_bridge,
                entry.feature_catalog,
                request,
                prior.exact_result,
            ) if entry.sha256() in candidates else _unknown_alias(
                request, prior.exact_result),
            None,
        )
        for entry, prior in zip(registry.entries, traces)
    )
    decision = project_raw_question_feature_registry_phase(
        registry, request, traces, "ALIAS")
    if decision is not None:
        return decision

    candidates = set(lookup_indexed_question_feature_entries(
        index, "IMPLICIT", request))
    traces = tuple(
        RawQuestionFeatureDispatchTrace(
            entry.sha256(),
            prior.exact_result,
            prior.alias_result,
            continue_implicit_predicate_question_answer(
                entry.alias_bridge,
                entry.implicit_bundle,
                request,
                prior.alias_result,
            ) if entry.sha256() in candidates else _unknown_implicit(
                request, prior.alias_result),
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
    "QUESTION_FEATURE_INDEX_EXPRESSION_BOUNDARY",
    "QUESTION_FEATURE_INDEX_SHA256",
    "RawQuestionFeatureAliasFrameRow",
    "RawQuestionFeatureIndex",
    "RawQuestionFeatureIndexPosting",
    "RawQuestionFeatureIndexRow",
    "W03W04W05QuestionFeatureIndexError",
    "build_raw_question_feature_index",
    "lookup_indexed_question_feature_entries",
    "run_indexed_raw_question_feature_registry_answer",
]
