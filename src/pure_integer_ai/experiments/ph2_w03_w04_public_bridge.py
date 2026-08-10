"""Public W-03 -> W-04 bridge authorized only by prerequisite stable keys."""
from __future__ import annotations

from pure_integer_ai.experiments.ph2_w03_v2_public_query import (
    W03V2PublicQuery,
    W03V2PublicQueryResult,
    run_w03_v2_public_queries,
)
from pure_integer_ai.experiments.ph2_w03_v2_public_source import (
    W03V2PublicEvaluationBatch,
)
from pure_integer_ai.experiments.ph2_w03_w04_public_bridge_contract import (
    W03W04PublicBridgeLink,
    W03W04PublicBridgeQuery,
    W03W04PublicBridgeResult,
)
from pure_integer_ai.experiments.ph2_w04_v2_public_query import (
    run_w04_v2_public_queries,
)
from pure_integer_ai.experiments.ph2_w04_v2_public_query_contract import (
    W04V2PublicQuery,
    W04V2PublicQueryResult,
)
from pure_integer_ai.experiments.ph2_w04_v2_public_source import (
    W04V2PublicEvaluationBatch,
)


def _matching_observation(
        batch,
        *,
        source_ref_key: tuple[int, ...],
        surface: str,
        context_text: str,
        stage: str,
        ):
    matches = []
    for pair in batch.pairs:
        observation = pair.observation
        if (observation.source_ref_key.stable_key() != source_ref_key
                or observation.w_stage != stage):
            continue
        payload = observation.typed_payload.to_value()
        observed_surface = payload.get(
            "surface" if stage == "W-03" else "surface_form")
        if (observed_surface == surface
                and payload.get("context") == context_text):
            matches.append(observation)
    return tuple(matches)


def _link(
        w03_batch: W03V2PublicEvaluationBatch,
        w04_batch: W04V2PublicEvaluationBatch,
        query: W03W04PublicBridgeQuery,
        w03: W03V2PublicQueryResult,
        w04: W04V2PublicQueryResult,
        ) -> W03W04PublicBridgeLink | None:
    if w03.status != "UNIQUE" or w04.status != "UNIQUE":
        return None
    w03_candidates = tuple(
        item for item in w03.candidates
        if item.active == 1 and item.sense_key == w03.selected_sense_key)
    w04_candidates = tuple(
        item for item in w04.candidates
        if (item.active == 1
            and item.primitive_registry == w04.selected_primitive_registry
            and item.primitive_kind == w04.selected_primitive_kind))
    if len(w03_candidates) != 1 or len(w04_candidates) != 1:
        return None
    sense = w03_candidates[0]
    primitive = w04_candidates[0]
    if (sense.source_ref_key != primitive.source_ref_key
            or sense.source_commitment != primitive.source_commitment):
        return None
    w03_observations = _matching_observation(
        w03_batch,
        source_ref_key=sense.source_ref_key,
        surface=query.surface,
        context_text=query.context_text,
        stage="W-03",
    )
    w04_observations = _matching_observation(
        w04_batch,
        source_ref_key=primitive.source_ref_key,
        surface=query.surface,
        context_text=query.context_text,
        stage="W-04",
    )
    if len(w03_observations) != 1 or len(w04_observations) != 1:
        return None
    w03_observation = w03_observations[0]
    w04_observation = w04_observations[0]
    if w04_observation.prerequisite_keys != (w03_observation.stable_key,):
        return None
    assert w03.selected_sense_key is not None
    assert w04.selected_primitive_registry is not None
    assert w04.selected_primitive_kind is not None
    selected_projection = w03_candidates[0]
    return W03W04PublicBridgeLink(
        sense.source_ref_key,
        sense.source_commitment,
        w03_observation.stable_key.stable_key(),
        w04_observation.stable_key.stable_key(),
        w03.selected_sense_key,
        selected_projection.concept_key,
        w04.selected_primitive_registry,
        w04.selected_primitive_kind,
    )


def project_w03_w04_public_bridge(
        w03_batch: W03V2PublicEvaluationBatch,
        w04_batch: W04V2PublicEvaluationBatch,
        query: W03W04PublicBridgeQuery,
        w03: W03V2PublicQueryResult,
        w04: W04V2PublicQueryResult,
        ) -> W03W04PublicBridgeResult:
    """Join two already projected stage results through explicit prerequisites."""
    if (not isinstance(w03_batch, W03V2PublicEvaluationBatch)
            or not isinstance(w04_batch, W04V2PublicEvaluationBatch)
            or not isinstance(query, W03W04PublicBridgeQuery)
            or not isinstance(w03, W03V2PublicQueryResult)
            or not isinstance(w04, W04V2PublicQueryResult)):
        raise TypeError("public bridge projection inputs are invalid")
    link = _link(w03_batch, w04_batch, query, w03, w04)
    if link is not None:
        status = "BRIDGED"
    elif w03.status in {"AMBIGUOUS", "CLARIFY"} or w04.status == "MULTI":
        status = "CLARIFY"
    else:
        status = "UNKNOWN"
    return W03W04PublicBridgeResult(
        query,
        status,
        w03,
        w04,
        link,
        w03_batch.source_binding.sha256(),
        w04_batch.source_binding.sha256(),
    )


def run_w03_w04_public_bridge_query(
        w03_batch: W03V2PublicEvaluationBatch,
        w04_batch: W04V2PublicEvaluationBatch,
        query: W03W04PublicBridgeQuery,
        ) -> W03W04PublicBridgeResult:
    """Run one bridge query through the shared-load batch entrypoint."""
    if not isinstance(query, W03W04PublicBridgeQuery):
        raise TypeError("public bridge query is invalid")
    return run_w03_w04_public_bridge_queries(
        w03_batch, w04_batch, (query,))[0]


def run_w03_w04_public_bridge_queries(
        w03_batch: W03V2PublicEvaluationBatch,
        w04_batch: W04V2PublicEvaluationBatch,
        queries: tuple[W03W04PublicBridgeQuery, ...],
        ) -> tuple[W03W04PublicBridgeResult, ...]:
    """Load each stage once, then bridge a bounded exact-query batch."""
    if (not isinstance(w03_batch, W03V2PublicEvaluationBatch)
            or not isinstance(w04_batch, W04V2PublicEvaluationBatch)
            or not isinstance(queries, tuple) or not queries
            or any(not isinstance(item, W03W04PublicBridgeQuery)
                   for item in queries)):
        raise TypeError("public bridge batch inputs are invalid")
    w03_results = run_w03_v2_public_queries(
        w03_batch,
        tuple(W03V2PublicQuery(
            item.surface, item.context_text, item.language, 0)
            for item in queries),
    )
    w04_results = run_w04_v2_public_queries(
        w04_batch,
        tuple(W04V2PublicQuery(
            item.surface, item.context_text, item.allow_generation)
            for item in queries),
    )
    return tuple(
        project_w03_w04_public_bridge(
            w03_batch, w04_batch, query, w03, w04)
        for query, w03, w04 in zip(
            queries, w03_results, w04_results, strict=True)
    )


__all__ = [
    "project_w03_w04_public_bridge",
    "run_w03_w04_public_bridge_queries",
    "run_w03_w04_public_bridge_query",
]
