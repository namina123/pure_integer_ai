"""在 FT17 上执行 FT18 注册项内构式候选分派。"""
from __future__ import annotations

from bisect import bisect_left

from pure_integer_ai.experiments.ph2_w03_w04_w05_question_construction_index_contract import (
    QUESTION_CONSTRUCTION_INDEX_EXPRESSION_BOUNDARY,
    QUESTION_CONSTRUCTION_INDEX_SHA256,
    RawQuestionConstructionAliasFrameRow,
    RawQuestionConstructionCandidateSet,
    RawQuestionConstructionIndex,
    RawQuestionConstructionIndexPosting,
    RawQuestionConstructionIndexRow,
    W03W04W05QuestionConstructionIndexError,
    build_raw_question_construction_index,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_question_feature_catalog import (
    run_raw_question_feature_candidate_answer,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_question_feature_index import (
    _unknown_alias,
    _unknown_exact,
    _unknown_implicit,
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
    continue_question_feature_predicate_alias_candidate_answer,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question_contract import (
    RawQuestionConstruction,
    RawQuestionRequest,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question_implicit import (
    continue_implicit_predicate_question_candidate_answer,
)


def _matches_alias_frame(
        row: RawQuestionConstructionAliasFrameRow,
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


def merge_indexed_question_construction_candidates(
        values: tuple[RawQuestionConstructionCandidateSet, ...],
        ) -> tuple[RawQuestionConstructionCandidateSet, ...]:
    grouped: dict[str, dict[str, RawQuestionConstruction]] = {}
    for value in values:
        target = grouped.setdefault(value.entry_sha256, {})
        for construction in value.constructions:
            target[construction.sha256()] = construction
    return tuple(
        RawQuestionConstructionCandidateSet(
            entry_sha,
            tuple(by_sha[key] for key in sorted(by_sha)),
        )
        for entry_sha, by_sha in sorted(grouped.items())
    )


def _posting_candidates(
        postings: tuple[RawQuestionConstructionIndexPosting, ...],
        source_record_key: tuple[int, ...] | None,
        ) -> tuple[RawQuestionConstructionCandidateSet, ...]:
    if source_record_key is None:
        return merge_indexed_question_construction_candidates(tuple(
            candidate
            for posting in postings
            for candidate in posting.candidates
        ))
    posting_ordinal = bisect_left(
        postings,
        source_record_key,
        key=lambda item: item.source_record_key,
    )
    if (posting_ordinal >= len(postings)
            or postings[posting_ordinal].source_record_key
            != source_record_key):
        return ()
    return postings[posting_ordinal].candidates


def lookup_indexed_question_feature_direct_constructions(
        index: RawQuestionConstructionIndex,
        phase: str,
        request: RawQuestionRequest,
        ) -> tuple[RawQuestionConstructionCandidateSet, ...]:
    """只查询精确表层投递行，不访问 alias frame。"""
    if (not isinstance(index, RawQuestionConstructionIndex)
            or not isinstance(request, RawQuestionRequest)):
        raise TypeError("indexed direct construction lookup inputs are invalid")
    rows = index.rows_at(phase)
    row_ordinal = bisect_left(
        rows,
        request.question_surface,
        key=lambda item: item.question_surface,
    )
    if (row_ordinal >= len(rows)
            or rows[row_ordinal].question_surface
            != request.question_surface):
        return ()
    return _posting_candidates(
        rows[row_ordinal].postings,
        request.source_record_key,
    )


def lookup_indexed_question_feature_alias_frame_constructions(
        index: RawQuestionConstructionIndex,
        request: RawQuestionRequest,
        frame_ordinals: tuple[int, ...] | None = None,
        ) -> tuple[RawQuestionConstructionCandidateSet, ...]:
    """只在给定 alias frame ordinal 中执行既有结构匹配。"""
    if (not isinstance(index, RawQuestionConstructionIndex)
            or not isinstance(request, RawQuestionRequest)):
        raise TypeError("indexed alias frame lookup inputs are invalid")
    if frame_ordinals is None:
        ordinals = tuple(range(len(index.alias_frame_rows)))
    else:
        if (not isinstance(frame_ordinals, tuple)
                or any(type(item) is not int for item in frame_ordinals)
                or frame_ordinals != tuple(sorted(set(frame_ordinals)))
                or any(item < 0 or item >= len(index.alias_frame_rows)
                       for item in frame_ordinals)):
            raise W03W04W05QuestionConstructionIndexError(
                "alias frame ordinals are not canonical")
        ordinals = frame_ordinals
    structural = tuple(
        candidate
        for ordinal in ordinals
        for row in (index.alias_frame_rows[ordinal],)
        if _matches_alias_frame(row, request.question_surface)
        for candidate in _posting_candidates(
            row.postings,
            request.source_record_key,
        )
    )
    return merge_indexed_question_construction_candidates(structural)


def lookup_indexed_question_feature_constructions(
        index: RawQuestionConstructionIndex,
        phase: str,
        request: RawQuestionRequest,
        ) -> tuple[RawQuestionConstructionCandidateSet, ...]:
    """返回一个全局阶段可到达的注册项与构式候选。"""
    if (not isinstance(index, RawQuestionConstructionIndex)
            or not isinstance(request, RawQuestionRequest)):
        raise TypeError("indexed construction lookup inputs are invalid")
    direct = lookup_indexed_question_feature_direct_constructions(
        index,
        phase,
        request,
    )
    if phase != "ALIAS":
        return direct
    structural = lookup_indexed_question_feature_alias_frame_constructions(
        index,
        request,
    )
    return merge_indexed_question_construction_candidates(
        (*direct, *structural))


def indexed_question_feature_candidate_counts(
        index: RawQuestionConstructionIndex,
        phase: str,
        request: RawQuestionRequest,
        ) -> tuple[int, int]:
    """给出确定性的注册项与内部构式候选数。"""
    candidates = lookup_indexed_question_feature_constructions(
        index,
        phase,
        request,
    )
    return (
        len(candidates),
        sum(len(item.constructions) for item in candidates),
    )


def _by_entry(
        values: tuple[RawQuestionConstructionCandidateSet, ...],
        ) -> dict[str, tuple[RawQuestionConstruction, ...]]:
    return {
        item.entry_sha256: item.constructions
        for item in values
    }


def lookup_indexed_alias_normalization_constructions(
        index: RawQuestionConstructionIndex,
        entry_sha256: str,
        request: RawQuestionRequest,
        alias_candidates: tuple[RawQuestionConstruction, ...],
        candidate_source: object,
        candidate_lookup,
        ) -> tuple[RawQuestionConstruction, ...]:
    values = {}
    for construction in alias_candidates:
        exact = candidate_lookup(
            candidate_source,
            "EXACT",
            RawQuestionRequest(
                construction.question_surface,
                request.source_record_key,
            ),
        )
        for item in exact:
            if item.entry_sha256 == entry_sha256:
                for candidate in item.constructions:
                    values[candidate.sha256()] = candidate
    if not values:
        raise W03W04W05QuestionConstructionIndexError(
            "alias normalization lost its exact construction")
    return tuple(values[key] for key in sorted(values))


def run_indexed_question_construction_registry_answer(
        index: RawQuestionConstructionIndex,
        request: RawQuestionRequest,
        ) -> RawQuestionFeatureRegistryAnswerResult:
    """只执行注册项内构式候选，同时保留完整 FT16 轨迹。"""
    return run_indexed_question_construction_registry_answer_with_lookup(
        index,
        request,
        index,
        lookup_indexed_question_feature_constructions,
    )


def run_indexed_question_construction_registry_answer_with_lookup(
        index: RawQuestionConstructionIndex,
        request: RawQuestionRequest,
        candidate_source: object,
        candidate_lookup,
        ) -> RawQuestionFeatureRegistryAnswerResult:
    """复用 FT18 三阶段运行时，但由调用方提供候选发现。"""
    if (not isinstance(index, RawQuestionConstructionIndex)
            or not isinstance(request, RawQuestionRequest)
            or not callable(candidate_lookup)):
        raise TypeError("indexed construction dispatch inputs are invalid")
    registry = index.registry
    candidates = _by_entry(candidate_lookup(
        candidate_source, "EXACT", request))
    traces = tuple(
        RawQuestionFeatureDispatchTrace(
            entry.sha256(),
            run_raw_question_feature_candidate_answer(
                entry.feature_catalog,
                request,
                candidates[entry.sha256()],
            ) if entry.sha256() in candidates else _unknown_exact(request),
            None,
            None,
        )
        for entry in registry.entries
    )
    decision = project_raw_question_feature_registry_phase(
        registry, request, traces, "EXACT")
    if decision is not None:
        return decision

    candidates = _by_entry(candidate_lookup(
        candidate_source, "ALIAS", request))
    traces = tuple(
        RawQuestionFeatureDispatchTrace(
            entry.sha256(),
            prior.exact_result,
            continue_question_feature_predicate_alias_candidate_answer(
                entry.alias_bridge,
                entry.feature_catalog,
                request,
                prior.exact_result,
                candidates[entry.sha256()],
                lookup_indexed_alias_normalization_constructions(
                    index,
                    entry.sha256(),
                    request,
                    candidates[entry.sha256()],
                    candidate_source,
                    candidate_lookup,
                ),
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

    candidates = _by_entry(candidate_lookup(
        candidate_source, "IMPLICIT", request))
    traces = tuple(
        RawQuestionFeatureDispatchTrace(
            entry.sha256(),
            prior.exact_result,
            prior.alias_result,
            continue_implicit_predicate_question_candidate_answer(
                entry.alias_bridge,
                entry.implicit_bundle,
                request,
                prior.alias_result,
                candidates[entry.sha256()],
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
    "QUESTION_CONSTRUCTION_INDEX_EXPRESSION_BOUNDARY",
    "QUESTION_CONSTRUCTION_INDEX_SHA256",
    "QUESTION_FEATURE_DISPATCH_PHASES",
    "RawQuestionConstructionAliasFrameRow",
    "RawQuestionConstructionCandidateSet",
    "RawQuestionConstructionIndex",
    "RawQuestionConstructionIndexPosting",
    "RawQuestionConstructionIndexRow",
    "W03W04W05QuestionConstructionIndexError",
    "build_raw_question_construction_index",
    "indexed_question_feature_candidate_counts",
    "lookup_indexed_question_feature_alias_frame_constructions",
    "lookup_indexed_alias_normalization_constructions",
    "lookup_indexed_question_feature_constructions",
    "lookup_indexed_question_feature_direct_constructions",
    "merge_indexed_question_construction_candidates",
    "run_indexed_question_construction_registry_answer",
    "run_indexed_question_construction_registry_answer_with_lookup",
]
