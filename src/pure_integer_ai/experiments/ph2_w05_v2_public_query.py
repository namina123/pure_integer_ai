"""在已学习公开 W-05 图上的来源绑定只读查询。"""
from __future__ import annotations

import hashlib
from typing import Any

from pure_integer_ai.cognition.shared.identity import (
    concept_identity,
    language_branch_identity,
)
from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    GenerationExpressionConstraints,
    LosslessIntegerKey,
)
from pure_integer_ai.experiments.ph2_w05_adapter import (
    W05_IDENTITY_VERSIONS,
    W05AtomicPropositionCandidate,
    adapt_w05_training_payload,
)
from pure_integer_ai.experiments.ph2_w05_generation import (
    build_w05_generation_runtime,
    generation_request_for_candidate,
)
from pure_integer_ai.experiments.ph2_w05_learning import (
    W05AtomicPropositionLearningRuntime,
    build_w05_learning_runtime,
)
from pure_integer_ai.experiments.ph2_w05_reasoning import (
    W05_REASONING_AUTHORIZED,
    W05_REASONING_CONFLICT,
    W05_REASONING_REJECTED,
    W05_REASONING_SUPERSEDED,
    build_w05_reasoning_runtime,
    reasoning_request_for_candidate,
)
from pure_integer_ai.experiments.ph2_w05_understanding import (
    build_w05_understanding_runtime,
    understanding_request_for_candidate,
)
from pure_integer_ai.experiments.ph2_w05_v2_public_query_contract import (
    W05_V2_PUBLIC_QUERY_GENERATION_NOT_RUN,
    W05V2PublicCandidateProjection,
    W05V2PublicGenerationProjection,
    W05V2PublicOccurrenceProjection,
    W05V2PublicQuery,
    W05V2PublicQueryError,
    W05V2PublicQueryResult,
    W05V2PublicRoleBindingProjection,
)
from pure_integer_ai.experiments.ph2_w05_v2_public_source import (
    W05V2PublicEvaluationBatch,
    w05_v2_public_training_payload,
)
from pure_integer_ai.storage.backend import DictBackend


_PUBLIC_QUERY_NAMESPACE = 50551
_SELECTED_REASONING_NOT_RUN = "NOT_RUN"


def _sha(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _request_key(
        query_ordinal: int,
        consumer_kind: int,
        candidate_ordinal: int,
        ) -> LosslessIntegerKey:
    return LosslessIntegerKey((
        _PUBLIC_QUERY_NAMESPACE,
        query_ordinal,
        consumer_kind,
        candidate_ordinal,
    ))


def _source_commitment(candidate: W05AtomicPropositionCandidate) -> str:
    return _sha(candidate.source_record.to_dict())


def _state_signature(runtime: W05AtomicPropositionLearningRuntime) -> str:
    report = runtime.report()
    return _sha({
        "active": [
            list(item.candidate.stable_key())
            for item in runtime.active_candidates()
        ],
        "applications": [
            {
                "accounts": [
                    {
                        "candidate": list(account.candidate.stable_key()),
                        "derived_supersede": int(account.derived_supersede),
                        "stance": account.stance,
                    }
                    for account in application.accounts
                ],
                "teacher": list(
                    application.binding.teacher_record.stable_key.stable_key()),
            }
            for application in runtime.applications()
        ],
        "registered": [
            {
                "candidate": list(item.candidate.stable_key()),
                "context": list(
                    item.proposition_definition.context.stable_key()),
                "occurrences": [
                    list(value.identity.stable_key())
                    for value in item.occurrences
                ],
                "roles": [
                    list(value.stable_key())
                    for value in item.role_binding_identities()
                ],
                "source": list(item.source_ref.stable_key()),
            }
            for item in runtime.registered_candidates()
        ],
        "report": {
            "account_count": report.account_count,
            "active_candidate_count": report.active_candidate_count,
            "candidate_count": report.candidate_count,
            "conflict_candidate_count": report.conflict_candidate_count,
            "evidence_application_count": report.evidence_application_count,
            "occurrence_count": report.occurrence_count,
            "role_binding_count": report.role_binding_count,
            "superseded_candidate_count": report.superseded_candidate_count,
            "unknown_candidate_count": report.unknown_candidate_count,
        },
        "superseded": [
            list(item.candidate.stable_key())
            for item in runtime.superseded_candidates()
        ],
    })


def _occurrence_projection(
        candidate: W05AtomicPropositionCandidate,
        ) -> tuple[W05V2PublicOccurrenceProjection, ...]:
    return tuple(
        W05V2PublicOccurrenceProjection(
            item.identity.stable_key(),
            item.semantic_object.stable_key(),
            item.start,
            item.end,
            item.ordinal,
            item.surface_fragment,
        )
        for item in candidate.occurrences
    )


def _role_projection(
        candidate: W05AtomicPropositionCandidate,
        ) -> tuple[W05V2PublicRoleBindingProjection, ...]:
    proposition = candidate.proposition_definition.proposition
    return tuple(
        W05V2PublicRoleBindingProjection(
            item.identity_for(proposition).stable_key(),
            item.role.stable_key(),
            item.filler.stable_key(),
            item.ordinal,
        )
        for item in candidate.proposition_definition.canonical_bindings()
    )


def _lifecycle(
        *,
        active: bool,
        superseded: bool,
        reasoning_status: str,
        ) -> str:
    if superseded or reasoning_status == W05_REASONING_SUPERSEDED:
        return "SUPERSEDED"
    if reasoning_status == W05_REASONING_CONFLICT:
        return "CONFLICT"
    if active and reasoning_status == W05_REASONING_AUTHORIZED:
        return "ACTIVE"
    if reasoning_status == W05_REASONING_REJECTED:
        return "REFUTED"
    return "UNKNOWN"


def _candidate_projection(
        candidate: W05AtomicPropositionCandidate,
        *,
        active: bool,
        superseded: bool,
        understanding_status: str,
        reasoning_status: str,
        evidence_count: int,
        ) -> W05V2PublicCandidateProjection:
    definition = candidate.proposition_definition
    return W05V2PublicCandidateProjection(
        candidate.surface,
        candidate.candidate.stable_key(),
        definition.predicate.stable_key(),
        definition.source_anchor.stable_key(),
        definition.context.stable_key(),
        tuple(item.stable_key() for item in candidate.occurrence_order),
        _occurrence_projection(candidate),
        _role_projection(candidate),
        candidate.source_ref.stable_key(),
        candidate.source_record.stable_key.stable_key(),
        candidate.source_record.source_key,
        _source_commitment(candidate),
        candidate.source_record.license_id,
        _lifecycle(
            active=active,
            superseded=superseded,
            reasoning_status=reasoning_status,
        ),
        int(active),
        int(superseded),
        understanding_status,
        reasoning_status,
        evidence_count,
    )


def _generation_projection(
        runtime: W05AtomicPropositionLearningRuntime,
        selected: W05AtomicPropositionCandidate,
        *,
        request_ordinal: int,
        ) -> tuple[str, tuple[W05V2PublicGenerationProjection, ...]]:
    branch = language_branch_identity(
        (_PUBLIC_QUERY_NAMESPACE, request_ordinal, 1),
        versions=W05_IDENTITY_VERSIONS,
    )
    uncertainty = concept_identity(
        (_PUBLIC_QUERY_NAMESPACE, request_ordinal, 2),
        versions=W05_IDENTITY_VERSIONS,
    )
    constraints = GenerationExpressionConstraints(
        branch, (), (), 0, 0, 0, 4096)
    generation = build_w05_generation_runtime(runtime)
    choice = generation.choose(generation_request_for_candidate(
        selected,
        request_key=_request_key(request_ordinal, 3, 1),
        uncertainty=uncertainty,
        constraints=constraints,
    ))
    candidate_by_id = {
        item.candidate: item for item in runtime.registered_candidates()}
    target_commitment = _source_commitment(selected)
    options = []
    for item in choice.options:
        construction_source = candidate_by_id.get(
            item.construction_source_candidate)
        if construction_source is None:
            raise W05V2PublicQueryError(
                "generation option references an unknown construction source")
        options.append(W05V2PublicGenerationProjection(
            item.surface,
            item.construction.stable_key(),
            item.construction_source_candidate.stable_key(),
            construction_source.source_ref.stable_key(),
            _source_commitment(construction_source),
            item.target_proposition.stable_key(),
            item.target_predicate.stable_key(),
            item.source.stable_key(),
            target_commitment,
            item.context.stable_key(),
            tuple(value.stable_key() for value in item.occurrence_order),
            tuple(value.stable_key() for value in item.role_bindings),
        ))
    return choice.status, tuple(options)


def project_w05_v2_public_query(
        runtime: W05AtomicPropositionLearningRuntime,
        query: W05V2PublicQuery,
        *,
        source_binding_sha256: str,
        record_commitment: str,
        request_ordinal: int,
        ) -> W05V2PublicQueryResult:
    """从既有 runtime 投影一个查询，不改变学习状态。"""
    if (not isinstance(runtime, W05AtomicPropositionLearningRuntime)
            or not isinstance(query, W05V2PublicQuery)
            or not isinstance(source_binding_sha256, str)
            or len(source_binding_sha256) != 64
            or not isinstance(record_commitment, str)
            or len(record_commitment) != 64
            or type(request_ordinal) is not int
            or request_ordinal <= 0):
        raise TypeError("public W-05 query projection inputs are invalid")
    registered = tuple(
        item for item in runtime.registered_candidates()
        if item.surface == query.surface
        and (query.source_ref_key is None
             or item.source_ref.stable_key() == query.source_ref_key)
    )
    active_keys = {item.candidate for item in runtime.active_candidates()}
    superseded_keys = {
        item.candidate for item in runtime.superseded_candidates()}
    understanding = build_w05_understanding_runtime(runtime)
    reasoning = build_w05_reasoning_runtime(runtime)
    projections = []
    reasoning_by_candidate: dict[object, str] = {}
    for candidate_ordinal, candidate in enumerate(registered, start=1):
        resolution = understanding.resolve(understanding_request_for_candidate(
            candidate,
            request_key=_request_key(
                request_ordinal, 1, candidate_ordinal),
        ))
        use = reasoning.authorize(reasoning_request_for_candidate(
            candidate,
            request_key=_request_key(
                request_ordinal, 2, candidate_ordinal),
        ))
        reasoning_by_candidate[candidate.candidate] = use.status
        projections.append(_candidate_projection(
            candidate,
            active=candidate.candidate in active_keys,
            superseded=candidate.candidate in superseded_keys,
            understanding_status=resolution.status,
            reasoning_status=use.status,
            evidence_count=len(use.evidence_keys),
        ))

    active_matches = tuple(
        item for item in registered if item.candidate in active_keys)
    selected: W05AtomicPropositionCandidate | None = None
    if len(active_matches) == 1:
        selected = active_matches[0]
        status = "UNIQUE"
    elif len(active_matches) > 1:
        status = "MULTI"
    elif any(
            item.lifecycle_status == "CONFLICT" for item in projections):
        status = "CONFLICT"
    else:
        status = "UNKNOWN"

    selected_key = None
    selected_reasoning = _SELECTED_REASONING_NOT_RUN
    generation_status = W05_V2_PUBLIC_QUERY_GENERATION_NOT_RUN
    generation_options: tuple[W05V2PublicGenerationProjection, ...] = ()
    if selected is not None:
        selected_key = selected.candidate.stable_key()
        selected_reasoning = reasoning_by_candidate[selected.candidate]
        if selected_reasoning != W05_REASONING_AUTHORIZED:
            raise W05V2PublicQueryError(
                "active Proposition is not authorized by current Evidence")
        if query.allow_generation:
            generation_status, generation_options = _generation_projection(
                runtime, selected, request_ordinal=request_ordinal)

    return W05V2PublicQueryResult(
        query,
        status,
        tuple(projections),
        selected_key,
        int(status == "MULTI"),
        selected_reasoning,
        generation_status,
        generation_options,
        source_binding_sha256,
        record_commitment,
    )


def run_w05_v2_public_query(
        batch: W05V2PublicEvaluationBatch,
        query: W05V2PublicQuery,
        ) -> W05V2PublicQueryResult:
    """通过共享装载的多查询公开入口运行单个查询。"""
    if (not isinstance(batch, W05V2PublicEvaluationBatch)
            or not isinstance(query, W05V2PublicQuery)):
        raise TypeError("public W-05 query inputs are invalid")
    return run_w05_v2_public_queries(batch, (query,))[0]


def run_w05_v2_public_queries(
        batch: W05V2PublicEvaluationBatch,
        queries: tuple[W05V2PublicQuery, ...],
        *,
        request_ordinals: tuple[int, ...] | None = None,
        ) -> tuple[W05V2PublicQueryResult, ...]:
    """只装载一次已学习 W-05 图并运行有界只读批次。"""
    if (not isinstance(batch, W05V2PublicEvaluationBatch)
            or not isinstance(queries, tuple) or not queries
            or any(not isinstance(item, W05V2PublicQuery)
                   for item in queries)
            or (request_ordinals is not None
                and (not isinstance(request_ordinals, tuple)
                     or len(request_ordinals) != len(queries)
                     or any(type(item) is not int or item <= 0
                            for item in request_ordinals)))):
        raise TypeError("public W-05 query batch inputs are invalid")
    ordinals = (
        tuple(range(1, len(queries) + 1))
        if request_ordinals is None else request_ordinals
    )
    output = adapt_w05_training_payload(
        w05_v2_public_training_payload(batch))
    backend = DictBackend()
    try:
        runtime = build_w05_learning_runtime(backend, output)
        before = _state_signature(runtime)
        results = tuple(
            project_w05_v2_public_query(
                runtime,
                query,
                source_binding_sha256=batch.source_binding.sha256(),
                record_commitment=batch.record_commitment,
                request_ordinal=ordinal,
            )
            for ordinal, query in zip(ordinals, queries, strict=True)
        )
        after = _state_signature(runtime)
        if before != after:
            raise W05V2PublicQueryError(
                "public W-05 query batch mutated learned state")
        return results
    finally:
        backend.close()


__all__ = [
    "project_w05_v2_public_query",
    "run_w05_v2_public_queries",
    "run_w05_v2_public_query",
]
