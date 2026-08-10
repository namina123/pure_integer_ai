"""Read-only source-bound queries over the learned public W-04 graph."""
from __future__ import annotations

import hashlib
from typing import Any

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_w04_adapter import (
    W04PrimitiveSurfaceCandidate,
    adapt_w04_training_payload,
)
from pure_integer_ai.experiments.ph2_w04_generation import (
    build_w04_generation_runtime,
)
from pure_integer_ai.experiments.ph2_w04_generation_contract import (
    W04_GENERATION_READY,
    W04GenerationRequest,
)
from pure_integer_ai.experiments.ph2_w04_learning import (
    W04PrimitiveSurfaceLearningRuntime,
    build_w04_learning_runtime,
)
from pure_integer_ai.experiments.ph2_w04_reasoning import (
    build_w04_reasoning_runtime,
)
from pure_integer_ai.experiments.ph2_w04_understanding import (
    W04_UNDERSTANDING_MULTI,
    W04_UNDERSTANDING_UNIQUE,
    W04_UNDERSTANDING_UNKNOWN,
    build_w04_understanding_runtime,
)
from pure_integer_ai.experiments.ph2_w04_v2_public_query_contract import (
    W04_V2_PUBLIC_QUERY_GENERATION_NOT_RUN,
    W04_V2_PUBLIC_QUERY_REASONING_NOT_RUN,
    W04V2PublicCandidateProjection,
    W04V2PublicGenerationProjection,
    W04V2PublicQuery,
    W04V2PublicQueryError,
    W04V2PublicQueryResult,
)
from pure_integer_ai.experiments.ph2_w04_v2_public_source import (
    W04V2PublicEvaluationBatch,
    w04_v2_public_training_payload,
)
from pure_integer_ai.storage.backend import DictBackend


def _sha(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _source_commitment(candidate: W04PrimitiveSurfaceCandidate) -> str:
    return _sha(candidate.source_record.to_dict())


def _state_signature(runtime: W04PrimitiveSurfaceLearningRuntime) -> str:
    report = runtime.report()
    return _sha({
        "active": [
            list(item.candidate.stable_key())
            for item in runtime.active_candidates()
        ],
        "registered": [
            list(item.candidate.stable_key())
            for item in runtime.registered_candidates()
        ],
        "report": {
            "account_count": report.account_count,
            "active_candidate_count": report.active_candidate_count,
            "candidate_count": report.candidate_count,
            "conflict_candidate_count": report.conflict_candidate_count,
            "evidence_application_count": report.evidence_application_count,
            "superseded_candidate_count": report.superseded_candidate_count,
            "unknown_candidate_count": report.unknown_candidate_count,
        },
        "superseded": [
            list(item.candidate.stable_key())
            for item in runtime.superseded_candidates()
        ],
    })


def _candidate_projection(
        candidate: W04PrimitiveSurfaceCandidate,
        *,
        active: set[object],
        superseded: set[object],
        ) -> W04V2PublicCandidateProjection:
    return W04V2PublicCandidateProjection(
        candidate.surface_form,
        candidate.context_text,
        candidate.primitive_registry,
        candidate.primitive_kind,
        candidate.candidate.stable_key(),
        candidate.source_record.stable_key.stable_key(),
        candidate.source_record.source_key,
        _source_commitment(candidate),
        candidate.source_record.license_id,
        int(candidate.candidate in active),
        int(candidate.candidate in superseded),
    )


def _generation_projection(
        runtime: W04PrimitiveSurfaceLearningRuntime,
        selected: W04PrimitiveSurfaceCandidate,
        ) -> tuple[str, tuple[W04V2PublicGenerationProjection, ...]]:
    generation = build_w04_generation_runtime(runtime)
    choice = generation.choose(W04GenerationRequest(
        selected.primitive_registry,
        selected.primitive_kind,
        selected.context_text,
        True,
    ))
    options = tuple(
        W04V2PublicGenerationProjection(
            item.surface_form,
            item.candidate.primitive_registry,
            item.candidate.primitive_kind,
            item.candidate.candidate.stable_key(),
            item.candidate.source_record.stable_key.stable_key(),
            _source_commitment(item.candidate),
        )
        for item in choice.options
    )
    return choice.status, options


def project_w04_v2_public_query(
        runtime: W04PrimitiveSurfaceLearningRuntime,
        query: W04V2PublicQuery,
        *,
        source_binding_sha256: str,
        ) -> W04V2PublicQueryResult:
    """Project one query from an existing runtime without changing its state."""
    if (not isinstance(runtime, W04PrimitiveSurfaceLearningRuntime)
            or not isinstance(query, W04V2PublicQuery)
            or not isinstance(source_binding_sha256, str)
            or len(source_binding_sha256) != 64):
        raise TypeError("public W-04 query projection inputs are invalid")
    registered = tuple(
        item for item in runtime.registered_candidates()
        if item.surface_form == query.surface
        and (query.context_text is None
             or item.context_text == query.context_text)
    )
    active_candidates = runtime.active_candidates()
    active_keys = {item.candidate for item in active_candidates}
    superseded_keys = {
        item.candidate for item in runtime.superseded_candidates()}
    active_matches = tuple(
        item for item in registered if item.candidate in active_keys)
    selected: W04PrimitiveSurfaceCandidate | None = None
    if query.context_text is not None:
        resolution = build_w04_understanding_runtime(runtime).resolve(
            query.surface, query.context_text)
        status = resolution.status
        selected = resolution.selected
    elif not active_matches:
        status = W04_UNDERSTANDING_UNKNOWN
    elif len(active_matches) == 1:
        status = W04_UNDERSTANDING_UNIQUE
        selected = active_matches[0]
    else:
        status = W04_UNDERSTANDING_MULTI

    candidates = tuple(
        _candidate_projection(
            item, active=active_keys, superseded=superseded_keys)
        for item in registered
    )
    selected_registry = None
    selected_kind = None
    reasoning_status = W04_V2_PUBLIC_QUERY_REASONING_NOT_RUN
    generation_status = W04_V2_PUBLIC_QUERY_GENERATION_NOT_RUN
    generation_options: tuple[W04V2PublicGenerationProjection, ...] = ()
    if status == W04_UNDERSTANDING_UNIQUE and selected is not None:
        selected_registry = selected.primitive_registry
        selected_kind = selected.primitive_kind
        reasoning = build_w04_reasoning_runtime(runtime).authorize(
            selected_registry, selected_kind)
        reasoning_status = reasoning.status
        if query.allow_generation:
            generation_status, generation_options = _generation_projection(
                runtime, selected)
            if (generation_status == W04_GENERATION_READY
                    and not generation_options):
                raise W04V2PublicQueryError(
                    "READY generation returned no source-bound option")
    return W04V2PublicQueryResult(
        query,
        status,
        candidates,
        selected_registry,
        selected_kind,
        int(status == W04_UNDERSTANDING_MULTI),
        reasoning_status,
        generation_status,
        generation_options,
        source_binding_sha256,
    )


def run_w04_v2_public_query(
        batch: W04V2PublicEvaluationBatch,
        query: W04V2PublicQuery,
        ) -> W04V2PublicQueryResult:
    """Run one query through the shared-load multi-query public entrypoint."""
    if (not isinstance(batch, W04V2PublicEvaluationBatch)
            or not isinstance(query, W04V2PublicQuery)):
        raise TypeError("public W-04 query inputs are invalid")
    return run_w04_v2_public_queries(batch, (query,))[0]


def run_w04_v2_public_queries(
        batch: W04V2PublicEvaluationBatch,
        queries: tuple[W04V2PublicQuery, ...],
        ) -> tuple[W04V2PublicQueryResult, ...]:
    """Load the learned W-04 graph once and run a bounded read-only batch."""
    if (not isinstance(batch, W04V2PublicEvaluationBatch)
            or not isinstance(queries, tuple) or not queries
            or any(not isinstance(item, W04V2PublicQuery)
                   for item in queries)):
        raise TypeError("public W-04 query batch inputs are invalid")
    output = adapt_w04_training_payload(
        w04_v2_public_training_payload(batch))
    backend = DictBackend()
    try:
        runtime = build_w04_learning_runtime(backend, output)
        before = _state_signature(runtime)
        results = tuple(
            project_w04_v2_public_query(
                runtime,
                query,
                source_binding_sha256=batch.source_binding.sha256(),
            )
            for query in queries
        )
        after = _state_signature(runtime)
        if before != after:
            raise W04V2PublicQueryError(
                "public W-04 query batch mutated learned state")
        return results
    finally:
        backend.close()


__all__ = [
    "project_w04_v2_public_query",
    "run_w04_v2_public_queries",
    "run_w04_v2_public_query",
]
