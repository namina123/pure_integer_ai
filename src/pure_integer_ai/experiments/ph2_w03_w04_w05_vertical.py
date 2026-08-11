"""Compose public W-03/W-04/W-05 only through two exact prerequisite links."""
from __future__ import annotations

from pure_integer_ai.experiments.ph2_w03_v2_public_source import (
    W03V2PublicEvaluationBatch,
)
from pure_integer_ai.experiments.ph2_w03_v2_public_query import (
    W03V2PublicQuery,
    run_w03_v2_public_queries,
)
from pure_integer_ai.experiments.ph2_w03_w04_public_bridge import (
    project_w03_w04_public_bridge,
)
from pure_integer_ai.experiments.ph2_w03_w04_public_bridge_contract import (
    W03W04PublicBridgeQuery,
    W03W04PublicBridgeResult,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_vertical_contract import (
    W03W04W05VerticalError,
    W03W04W05VerticalLink,
    W03W04W05VerticalQuery,
    W03W04W05VerticalResult,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_vertical_overlay_registry import (
    AUTHORIZED_VERTICAL_OVERLAY_VALIDATION_SHA256S,
)
from pure_integer_ai.experiments.ph2_w04_v2_public_source import (
    W04V2PublicEvaluationBatch,
)
from pure_integer_ai.experiments.ph2_w04_v2_public_query import (
    run_w04_v2_public_queries,
)
from pure_integer_ai.experiments.ph2_w04_v2_public_query_contract import (
    W04V2PublicQuery,
)
from pure_integer_ai.experiments.ph2_w04_w05_public_bridge import (
    project_w04_w05_public_bridge,
)
from pure_integer_ai.experiments.ph2_w04_w05_public_bridge_contract import (
    W04W05PublicBridgeQuery,
    W04W05PublicBridgeResult,
)
from pure_integer_ai.experiments.ph2_w05_v2_public_source import (
    W05V2PublicEvaluationBatch,
)
from pure_integer_ai.experiments.ph2_w05_v2_public_query import (
    run_w05_v2_public_queries,
)
from pure_integer_ai.experiments.ph2_w05_v2_public_query_contract import (
    W05V2PublicQuery,
)


def _join(
        w03_w04: W03W04PublicBridgeResult,
        w04_w05: W04W05PublicBridgeResult,
        ) -> W03W04W05VerticalLink | None:
    """Require the same source and exact W-04 Observation in both links."""
    if (w03_w04.status != "BRIDGED" or w03_w04.link is None
            or w04_w05.status != "BRIDGED" or w04_w05.link is None):
        return None
    left = w03_w04.link
    right = w04_w05.link
    if (left.source_ref_key != right.source_record_key
            or left.source_commitment != right.source_commitment
            or left.w04_observation_key != right.w04_observation_key
            or left.primitive_registry != right.primitive_registry
            or left.primitive_kind != right.primitive_kind):
        return None
    return W03W04W05VerticalLink(
        left.source_ref_key,
        left.source_commitment,
        left.w03_observation_key,
        left.w04_observation_key,
        right.w05_observation_key,
        left.sense_key,
        left.concept_key,
        left.primitive_registry,
        left.primitive_kind,
        right.proposition_key,
        right.predicate_key,
        right.predicate_occurrence_key,
    )


def project_w03_w04_w05_vertical(
        query: W03W04W05VerticalQuery,
        w03_w04: W03W04PublicBridgeResult,
        w04_w05: W04W05PublicBridgeResult,
        *,
        overlay_validation_sha256: str,
        ) -> W03W04W05VerticalResult:
    """Project the total chain without surface- or primitive-based fallback."""
    if (not isinstance(query, W03W04W05VerticalQuery)
            or not isinstance(w03_w04, W03W04PublicBridgeResult)
            or not isinstance(w04_w05, W04W05PublicBridgeResult)):
        raise TypeError("vertical projection inputs are invalid")
    if overlay_validation_sha256 not in (
            AUTHORIZED_VERTICAL_OVERLAY_VALIDATION_SHA256S):
        raise W03W04W05VerticalError(
            "vertical projection is not bound to the frozen overlay")
    link = _join(w03_w04, w04_w05)
    if link is not None:
        status = "BRIDGED"
    elif w03_w04.status == "CLARIFY" or w04_w05.status == "CLARIFY":
        status = "CLARIFY"
    else:
        status = "UNKNOWN"
    return W03W04W05VerticalResult(
        query,
        status,
        w03_w04,
        w04_w05,
        link,
        overlay_validation_sha256,
    )


def run_w03_w04_w05_vertical_query(
        w03_batch: W03V2PublicEvaluationBatch,
        w04_batch: W04V2PublicEvaluationBatch,
        w05_batch: W05V2PublicEvaluationBatch,
        query: W03W04W05VerticalQuery,
        *,
        overlay_validation_sha256: str,
        ) -> W03W04W05VerticalResult:
    """Run one vertical query through the identity-compatible batch path."""
    if (not isinstance(w03_batch, W03V2PublicEvaluationBatch)
            or not isinstance(w04_batch, W04V2PublicEvaluationBatch)
            or not isinstance(w05_batch, W05V2PublicEvaluationBatch)
            or not isinstance(query, W03W04W05VerticalQuery)):
        raise TypeError("vertical query inputs are invalid")
    return run_w03_w04_w05_vertical_queries(
        w03_batch,
        w04_batch,
        w05_batch,
        (query,),
        overlay_validation_sha256=overlay_validation_sha256,
    )[0]


def run_w03_w04_w05_vertical_queries(
        w03_batch: W03V2PublicEvaluationBatch,
        w04_batch: W04V2PublicEvaluationBatch,
        w05_batch: W05V2PublicEvaluationBatch,
        queries: tuple[W03W04W05VerticalQuery, ...],
        *,
        overlay_validation_sha256: str,
        ) -> tuple[W03W04W05VerticalResult, ...]:
    """Load each learned stage once and preserve singleton request identities."""
    if (not isinstance(w03_batch, W03V2PublicEvaluationBatch)
            or not isinstance(w04_batch, W04V2PublicEvaluationBatch)
            or not isinstance(w05_batch, W05V2PublicEvaluationBatch)
            or not isinstance(queries, tuple) or not queries
            or any(not isinstance(item, W03W04W05VerticalQuery)
                   for item in queries)):
        raise TypeError("vertical query batch inputs are invalid")
    w03_queries = tuple(W03V2PublicQuery(
        item.surface,
        item.context_text,
        item.language,
        0,
    ) for item in queries)
    w04_queries = tuple(W04V2PublicQuery(
        item.surface,
        item.context_text,
        item.allow_generation,
    ) for item in queries)
    w05_queries = tuple(W05V2PublicQuery(
        item.proposition_surface,
        allow_generation=item.allow_generation,
    ) for item in queries)
    w03_results = run_w03_v2_public_queries(w03_batch, w03_queries)
    w04_results = run_w04_v2_public_queries(w04_batch, w04_queries)
    w05_results = run_w05_v2_public_queries(
        w05_batch,
        w05_queries,
        request_ordinals=(1,) * len(queries),
    )
    values = []
    for query, w03, w04, w05 in zip(
            queries, w03_results, w04_results, w05_results, strict=True):
        w03_w04_query = W03W04PublicBridgeQuery(
            query.surface,
            query.context_text,
            query.language,
            query.allow_generation,
        )
        w04_w05_query = W04W05PublicBridgeQuery(
            query.surface,
            query.context_text,
            query.proposition_surface,
            query.allow_generation,
        )
        values.append(project_w03_w04_w05_vertical(
            query,
            project_w03_w04_public_bridge(
                w03_batch, w04_batch, w03_w04_query, w03, w04),
            project_w04_w05_public_bridge(
                w04_batch, w05_batch, w04_w05_query, w04, w05),
            overlay_validation_sha256=overlay_validation_sha256,
        ))
    return tuple(values)


__all__ = [
    "project_w03_w04_w05_vertical",
    "run_w03_w04_w05_vertical_queries",
    "run_w03_w04_w05_vertical_query",
]
