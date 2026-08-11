"""执行 FT20 候选限定热路径，并按需恢复 FT16 完整审计结果。"""
from __future__ import annotations

from bisect import bisect_left

from pure_integer_ai.experiments.ph2_w03_w04_w05_question_alias_frame_anchor import (
    lookup_indexed_alias_frame_anchor_constructions,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_question_construction_index import (
    lookup_indexed_alias_normalization_constructions,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_question_feature_catalog import (
    run_raw_question_feature_candidate_answer,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_question_feature_index import (
    _unknown_alias,
    _unknown_exact,
    _unknown_implicit,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_question_feature_registry_contract import (
    RawQuestionFeatureDispatchTrace,
    RawQuestionFeatureRegistryAnswerResult,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_question_sparse_dispatch_contract import (
    QUESTION_SPARSE_DISPATCH_EXPRESSION_BOUNDARY,
    QUESTION_SPARSE_DISPATCH_SHA256,
    RawQuestionSparseDecision,
    RawQuestionSparseDispatchEntry,
    RawQuestionSparseDispatchIndex,
    RawQuestionSparseDispatchProbe,
    RawQuestionSparseExecutionRecord,
    RawQuestionSparsePhaseVisit,
    W03W04W05QuestionSparseDispatchError,
    build_raw_question_sparse_dispatch_index,
    project_sparse_question_phase_decision,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question_alias import (
    continue_question_feature_predicate_alias_candidate_answer,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question_contract import (
    RawQuestionRequest,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question_implicit import (
    continue_implicit_predicate_question_candidate_answer,
)


def _entry_at(
        index: RawQuestionSparseDispatchIndex,
        entry_sha256: str,
        ) -> RawQuestionSparseDispatchEntry:
    ordinal = bisect_left(
        index.entries,
        entry_sha256,
        key=lambda item: item.entry_sha256,
    )
    if (ordinal >= len(index.entries)
            or index.entries[ordinal].entry_sha256 != entry_sha256):
        raise W03W04W05QuestionSparseDispatchError(
            "candidate entry escaped sparse dispatch index")
    return index.entries[ordinal]


def _record(
        index: RawQuestionSparseDispatchIndex,
        request: RawQuestionRequest,
        decision: RawQuestionSparseDecision,
        visits: list[RawQuestionSparsePhaseVisit],
        traces: dict[str, RawQuestionFeatureDispatchTrace],
        ) -> RawQuestionSparseExecutionRecord:
    return RawQuestionSparseExecutionRecord(
        index.identity_sha256,
        index.registry_identity_sha256,
        request,
        decision,
        tuple(visits),
        tuple(traces[key] for key in sorted(traces)),
    )


def run_sparse_question_dispatch(
        index: RawQuestionSparseDispatchIndex,
        request: RawQuestionRequest,
        ) -> RawQuestionSparseExecutionRecord:
    """只执行各阶段 candidate entries，不创建非候选 UNKNOWN 轨迹。"""
    if (not isinstance(index, RawQuestionSparseDispatchIndex)
            or not isinstance(request, RawQuestionRequest)):
        raise TypeError("sparse question dispatch inputs are invalid")
    candidate_source = index.anchor_index
    construction_index = candidate_source.construction_index
    visits: list[RawQuestionSparsePhaseVisit] = []
    traces: dict[str, RawQuestionFeatureDispatchTrace] = {}

    exact_candidates = lookup_indexed_alias_frame_anchor_constructions(
        candidate_source, "EXACT", request)
    visits.append(RawQuestionSparsePhaseVisit(
        "EXACT",
        tuple(item.entry_sha256 for item in exact_candidates),
        sum(len(item.constructions) for item in exact_candidates),
    ))
    exact_phase = []
    for candidate in exact_candidates:
        row = _entry_at(index, candidate.entry_sha256)
        trace = RawQuestionFeatureDispatchTrace(
            row.entry_sha256,
            run_raw_question_feature_candidate_answer(
                row.entry.feature_catalog,
                request,
                candidate.constructions,
            ),
            None,
            None,
        )
        traces[row.entry_sha256] = trace
        exact_phase.append(trace)
    decision = project_sparse_question_phase_decision(
        tuple(exact_phase), "EXACT")
    if decision is not None:
        return _record(index, request, decision, visits, traces)

    alias_candidates = lookup_indexed_alias_frame_anchor_constructions(
        candidate_source, "ALIAS", request)
    visits.append(RawQuestionSparsePhaseVisit(
        "ALIAS",
        tuple(item.entry_sha256 for item in alias_candidates),
        sum(len(item.constructions) for item in alias_candidates),
    ))
    alias_phase = []
    for candidate in alias_candidates:
        row = _entry_at(index, candidate.entry_sha256)
        prior = traces.get(row.entry_sha256)
        exact_result = (
            _unknown_exact(request)
            if prior is None else prior.exact_result
        )
        alias_result = continue_question_feature_predicate_alias_candidate_answer(
            row.entry.alias_bridge,
            row.entry.feature_catalog,
            request,
            exact_result,
            candidate.constructions,
            lookup_indexed_alias_normalization_constructions(
                construction_index,
                row.entry_sha256,
                request,
                candidate.constructions,
                candidate_source,
                lookup_indexed_alias_frame_anchor_constructions,
            ),
        )
        trace = RawQuestionFeatureDispatchTrace(
            row.entry_sha256,
            exact_result,
            alias_result,
            None,
        )
        traces[row.entry_sha256] = trace
        alias_phase.append(trace)
    decision = project_sparse_question_phase_decision(
        tuple(alias_phase), "ALIAS")
    if decision is not None:
        return _record(index, request, decision, visits, traces)

    implicit_candidates = lookup_indexed_alias_frame_anchor_constructions(
        candidate_source, "IMPLICIT", request)
    visits.append(RawQuestionSparsePhaseVisit(
        "IMPLICIT",
        tuple(item.entry_sha256 for item in implicit_candidates),
        sum(len(item.constructions) for item in implicit_candidates),
    ))
    implicit_phase = []
    for candidate in implicit_candidates:
        row = _entry_at(index, candidate.entry_sha256)
        prior = traces.get(row.entry_sha256)
        exact_result = (
            _unknown_exact(request)
            if prior is None else prior.exact_result
        )
        alias_result = (
            _unknown_alias(request, exact_result)
            if prior is None or prior.alias_result is None
            else prior.alias_result
        )
        implicit_result = continue_implicit_predicate_question_candidate_answer(
            row.entry.alias_bridge,
            row.entry.implicit_bundle,
            request,
            alias_result,
            candidate.constructions,
        )
        trace = RawQuestionFeatureDispatchTrace(
            row.entry_sha256,
            exact_result,
            alias_result,
            implicit_result,
        )
        traces[row.entry_sha256] = trace
        implicit_phase.append(trace)
    decision = project_sparse_question_phase_decision(
        tuple(implicit_phase), "IMPLICIT")
    if decision is None:
        decision = RawQuestionSparseDecision(
            "UNKNOWN", None, None, (), None)
    return _record(index, request, decision, visits, traces)


def project_sparse_question_dispatch_audit(
        index: RawQuestionSparseDispatchIndex,
        record: RawQuestionSparseExecutionRecord,
        ) -> RawQuestionFeatureRegistryAnswerResult:
    """按需补齐所有非候选 UNKNOWN，恢复 FT16 完整审计投影。"""
    if (not isinstance(index, RawQuestionSparseDispatchIndex)
            or not isinstance(record, RawQuestionSparseExecutionRecord)):
        raise TypeError("sparse question audit projection inputs are invalid")
    if (record.dispatch_index_identity_sha256 != index.identity_sha256
            or record.registry_identity_sha256
            != index.registry_identity_sha256):
        raise W03W04W05QuestionSparseDispatchError(
            "sparse execution record escaped its dispatch index")
    sparse = {item.entry_sha256: item for item in record.traces}
    phase = record.decision.decisive_phase
    alias_depth = phase in {"ALIAS", "IMPLICIT", None}
    implicit_depth = phase in {"IMPLICIT", None}
    traces = []
    for row in index.entries:
        prior = sparse.get(row.entry_sha256)
        exact_result = (
            _unknown_exact(record.request)
            if prior is None else prior.exact_result
        )
        alias_result = None
        if alias_depth:
            alias_result = (
                _unknown_alias(record.request, exact_result)
                if prior is None or prior.alias_result is None
                else prior.alias_result
            )
        implicit_result = None
        if implicit_depth:
            assert alias_result is not None
            implicit_result = (
                _unknown_implicit(record.request, alias_result)
                if prior is None or prior.implicit_result is None
                else prior.implicit_result
            )
        traces.append(RawQuestionFeatureDispatchTrace(
            row.entry_sha256,
            exact_result,
            alias_result,
            implicit_result,
        ))
    decision = record.decision
    return RawQuestionFeatureRegistryAnswerResult(
        record.registry_identity_sha256,
        record.request,
        decision.status,
        decision.answer_surface,
        decision.decisive_phase,
        decision.interpretations,
        decision.selected_entry_sha256,
        tuple(traces),
    )


def sparse_question_dispatch_probe(
        index: RawQuestionSparseDispatchIndex,
        record: RawQuestionSparseExecutionRecord,
        ) -> RawQuestionSparseDispatchProbe:
    """返回 visited entry、phase result、construction 与审计规模。"""
    if (not isinstance(index, RawQuestionSparseDispatchIndex)
            or not isinstance(record, RawQuestionSparseExecutionRecord)):
        raise TypeError("sparse question dispatch probe inputs are invalid")
    if record.dispatch_index_identity_sha256 != index.identity_sha256:
        raise W03W04W05QuestionSparseDispatchError(
            "sparse probe record escaped its dispatch index")
    return RawQuestionSparseDispatchProbe(
        tuple(item.entry_sha256 for item in record.traces),
        tuple(len(item.entry_sha256s) for item in record.phase_visits),
        tuple(item.construction_count for item in record.phase_visits),
        len(record.traces),
        sum(item.alias_result is not None for item in record.traces),
        sum(item.implicit_result is not None for item in record.traces),
        sum(len(item.entry_sha256s) for item in record.phase_visits),
        len(record.traces),
        len(index.entries),
    )


def run_sparse_question_dispatch_registry_answer(
        index: RawQuestionSparseDispatchIndex,
        request: RawQuestionRequest,
        ) -> RawQuestionFeatureRegistryAnswerResult:
    """兼容既有完整结果门面，但将全量 UNKNOWN 构造推迟到投影时。"""
    return project_sparse_question_dispatch_audit(
        index,
        run_sparse_question_dispatch(index, request),
    )


__all__ = [
    "QUESTION_SPARSE_DISPATCH_EXPRESSION_BOUNDARY",
    "QUESTION_SPARSE_DISPATCH_SHA256",
    "RawQuestionSparseDecision",
    "RawQuestionSparseDispatchEntry",
    "RawQuestionSparseDispatchIndex",
    "RawQuestionSparseDispatchProbe",
    "RawQuestionSparseExecutionRecord",
    "RawQuestionSparsePhaseVisit",
    "W03W04W05QuestionSparseDispatchError",
    "build_raw_question_sparse_dispatch_index",
    "project_sparse_question_dispatch_audit",
    "project_sparse_question_phase_decision",
    "run_sparse_question_dispatch",
    "run_sparse_question_dispatch_registry_answer",
    "sparse_question_dispatch_probe",
]
