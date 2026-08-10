"""只由同源 prerequisite 与 predicate occurrence 授权的 W-04 到 W-05 bridge。"""
from __future__ import annotations

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
from pure_integer_ai.experiments.ph2_w04_w05_public_bridge_contract import (
    W04W05PublicBridgeLink,
    W04W05PublicBridgeQuery,
    W04W05PublicBridgeResult,
)
from pure_integer_ai.experiments.ph2_w05_v2_public_query import (
    run_w05_v2_public_queries,
)
from pure_integer_ai.experiments.ph2_w05_v2_public_query_contract import (
    W05V2PublicQuery,
    W05V2PublicQueryResult,
)
from pure_integer_ai.experiments.ph2_w05_v2_public_source import (
    W05V2PublicEvaluationBatch,
)


def _matching_observation(
        batch,
        *,
        source_record_key: tuple[int, ...],
        surface: str,
        context_text: str | None,
        stage: str,
        ):
    matches = []
    for pair in batch.pairs:
        observation = pair.observation
        if (observation.source_ref_key.stable_key() != source_record_key
                or observation.w_stage != stage):
            continue
        payload = observation.typed_payload.to_value()
        if stage == "W-04":
            matched = (
                payload.get("surface_form") == surface
                and payload.get("context") == context_text
            )
        else:
            matched = payload.get("surface") == surface
        if matched:
            matches.append(observation)
    return tuple(matches)


def _link(
        w04_batch: W04V2PublicEvaluationBatch,
        w05_batch: W05V2PublicEvaluationBatch,
        query: W04W05PublicBridgeQuery,
        w04: W04V2PublicQueryResult,
        w05: W05V2PublicQueryResult,
        ) -> W04W05PublicBridgeLink | None:
    if w04.status != "UNIQUE" or w05.status != "UNIQUE":
        return None
    w04_candidates = tuple(
        item for item in w04.candidates
        if (item.active == 1
            and item.primitive_registry == w04.selected_primitive_registry
            and item.primitive_kind == w04.selected_primitive_kind))
    w05_candidates = tuple(
        item for item in w05.candidates
        if (item.lifecycle_status == "ACTIVE"
            and item.proposition_key == w05.selected_proposition_key
            and item.reasoning_status == "AUTHORIZED"))
    if len(w04_candidates) != 1 or len(w05_candidates) != 1:
        return None
    primitive = w04_candidates[0]
    proposition = w05_candidates[0]
    if (primitive.source_ref_key != proposition.source_record_key
            or primitive.source_commitment != proposition.source_commitment):
        return None
    predicate_occurrences = tuple(
        item for item in proposition.occurrences
        if item.identity_key == proposition.source_anchor_key)
    if (len(predicate_occurrences) != 1
            or predicate_occurrences[0].surface_fragment
            != query.primitive_surface):
        return None
    w04_observations = _matching_observation(
        w04_batch,
        source_record_key=primitive.source_ref_key,
        surface=query.primitive_surface,
        context_text=query.context_text,
        stage="W-04",
    )
    w05_observations = _matching_observation(
        w05_batch,
        source_record_key=proposition.source_record_key,
        surface=query.proposition_surface,
        context_text=None,
        stage="W-05",
    )
    if len(w04_observations) != 1 or len(w05_observations) != 1:
        return None
    w04_observation = w04_observations[0]
    w05_observation = w05_observations[0]
    if w05_observation.prerequisite_keys != (w04_observation.stable_key,):
        return None
    assert w04.selected_primitive_registry is not None
    assert w04.selected_primitive_kind is not None
    assert w05.selected_proposition_key is not None
    return W04W05PublicBridgeLink(
        primitive.source_ref_key,
        primitive.source_commitment,
        w04_observation.stable_key.stable_key(),
        w05_observation.stable_key.stable_key(),
        w04.selected_primitive_registry,
        w04.selected_primitive_kind,
        w05.selected_proposition_key,
        proposition.predicate_key,
        proposition.source_anchor_key,
        proposition.context_key,
        proposition.occurrence_order,
        tuple(item.identity_key for item in proposition.role_bindings),
    )


def project_w04_w05_public_bridge(
        w04_batch: W04V2PublicEvaluationBatch,
        w05_batch: W05V2PublicEvaluationBatch,
        query: W04W05PublicBridgeQuery,
        w04: W04V2PublicQueryResult,
        w05: W05V2PublicQueryResult,
        ) -> W04W05PublicBridgeResult:
    """只通过显式 prerequisite 把两个已投影 stage result 连接起来。"""
    if (not isinstance(w04_batch, W04V2PublicEvaluationBatch)
            or not isinstance(w05_batch, W05V2PublicEvaluationBatch)
            or not isinstance(query, W04W05PublicBridgeQuery)
            or not isinstance(w04, W04V2PublicQueryResult)
            or not isinstance(w05, W05V2PublicQueryResult)):
        raise TypeError("W04/W05 public bridge 输入非法")
    link = _link(w04_batch, w05_batch, query, w04, w05)
    if link is not None:
        status = "BRIDGED"
    elif w04.status == "MULTI" or w05.status == "MULTI":
        status = "CLARIFY"
    else:
        status = "UNKNOWN"
    return W04W05PublicBridgeResult(
        query,
        status,
        w04,
        w05,
        link,
        w04_batch.source_binding.sha256(),
        w05_batch.source_binding.sha256(),
    )


def run_w04_w05_public_bridge_query(
        w04_batch: W04V2PublicEvaluationBatch,
        w05_batch: W05V2PublicEvaluationBatch,
        query: W04W05PublicBridgeQuery,
        ) -> W04W05PublicBridgeResult:
    """通过共享装载的 batch 入口运行一个 bridge query。"""
    if not isinstance(query, W04W05PublicBridgeQuery):
        raise TypeError("W04/W05 public bridge query 非法")
    return run_w04_w05_public_bridge_queries(
        w04_batch, w05_batch, (query,))[0]


def run_w04_w05_public_bridge_queries(
        w04_batch: W04V2PublicEvaluationBatch,
        w05_batch: W05V2PublicEvaluationBatch,
        queries: tuple[W04W05PublicBridgeQuery, ...],
        ) -> tuple[W04W05PublicBridgeResult, ...]:
    """每个 stage 只装载一次，再运行有界精确 bridge query 批次。"""
    if (not isinstance(w04_batch, W04V2PublicEvaluationBatch)
            or not isinstance(w05_batch, W05V2PublicEvaluationBatch)
            or not isinstance(queries, tuple) or not queries
            or any(not isinstance(item, W04W05PublicBridgeQuery)
                   for item in queries)):
        raise TypeError("W04/W05 public bridge batch 输入非法")
    w04_results = run_w04_v2_public_queries(
        w04_batch,
        tuple(W04V2PublicQuery(
            item.primitive_surface,
            item.context_text,
            item.allow_generation,
        ) for item in queries),
    )
    w05_results = run_w05_v2_public_queries(
        w05_batch,
        tuple(W05V2PublicQuery(
            item.proposition_surface,
            allow_generation=item.allow_generation,
        ) for item in queries),
    )
    return tuple(
        project_w04_w05_public_bridge(
            w04_batch, w05_batch, query, w04, w05)
        for query, w04, w05 in zip(
            queries, w04_results, w05_results, strict=True)
    )


__all__ = [
    "project_w04_w05_public_bridge",
    "run_w04_w05_public_bridge_queries",
    "run_w04_w05_public_bridge_query",
]
