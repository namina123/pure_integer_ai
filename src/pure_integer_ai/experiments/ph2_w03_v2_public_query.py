"""Source-bound public W-03 sense query and authorized generation projection."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.experiments.formal_train import make_train_context
from pure_integer_ai.experiments.ph2_d03_contract_core import (
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    LosslessIntegerKey,
)
from pure_integer_ai.experiments.ph2_w03_adapter import (
    W03SenseCandidateEnvelope,
    adapt_w03_training_payload,
)
from pure_integer_ai.experiments.ph2_w03_generation import (
    W03_GENERATION_READY,
    W03ExpressionConstraints,
    W03GenerationRequest,
    build_w03_generation_runtime,
)
from pure_integer_ai.experiments.ph2_w03_understanding import (
    W03_UNDERSTANDING_AMBIGUOUS,
    W03_UNDERSTANDING_CLARIFY,
    W03_UNDERSTANDING_UNIQUE,
    W03_UNDERSTANDING_UNKNOWN,
    W03UnderstandingRuntime,
    build_w03_understanding_runtime,
)
from pure_integer_ai.experiments.ph2_w03_v2_public_source import (
    W03V2PublicEvaluationBatch,
)
from pure_integer_ai.storage.backend import DictBackend


W03_V2_PUBLIC_QUERY_VERSION = "PH2-W03-V2-PUBLIC-QUERY-V1"
W03_V2_PUBLIC_QUERY_GENERATION_NOT_RUN = "NOT_RUN"
W03_V2_PUBLIC_QUERY_STATUSES = (
    W03_UNDERSTANDING_UNIQUE,
    W03_UNDERSTANDING_AMBIGUOUS,
    W03_UNDERSTANDING_CLARIFY,
    W03_UNDERSTANDING_UNKNOWN,
)


# object-model: exception
class W03V2PublicQueryError(ValueError):
    """A public query or its source-bound projection is inconsistent."""


def _sha(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _text(value: object, *, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise W03V2PublicQueryError(f"{where} must be nonempty trimmed text")
    return value


def _strict_key(value: tuple[int, ...], *, where: str) -> tuple[int, ...]:
    if (not isinstance(value, tuple) or not value
            or any(type(item) is not int for item in value)):
        raise W03V2PublicQueryError(f"{where} is not a strict integer key")
    return value


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W03V2PublicQuery:
    """Exact external surface/context query; no expected answer is supplied."""

    surface: str
    context_text: str | None = None
    language: str = "zh"
    allow_generation: int = 1

    def __post_init__(self) -> None:
        _text(self.surface, where="query surface")
        _text(self.language, where="query language")
        if self.context_text is not None:
            _text(self.context_text, where="query context")
        if self.allow_generation not in {0, 1}:
            raise W03V2PublicQueryError(
                "query allow_generation must be zero or one")

    def to_dict(self) -> dict[str, object]:
        return {
            "allow_generation": self.allow_generation,
            "context_text": self.context_text,
            "language": self.language,
            "surface": self.surface,
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W03V2PublicCandidateProjection:
    """One learned Sense/Concept with its exact redistributable SourceRef."""

    surface: str
    context_text: str | None
    sense_key: tuple[int, ...]
    concept_key: tuple[int, ...]
    source_ref_key: tuple[int, ...]
    source_key: str
    source_commitment: str
    license_id: str
    active: int

    def __post_init__(self) -> None:
        _text(self.surface, where="candidate surface")
        if self.context_text is not None:
            _text(self.context_text, where="candidate context")
        for name in ("sense_key", "concept_key", "source_ref_key"):
            _strict_key(getattr(self, name), where=f"candidate {name}")
        _text(self.source_key, where="candidate source_key")
        _text(self.license_id, where="candidate license_id")
        if (not isinstance(self.source_commitment, str)
                or len(self.source_commitment) != 64
                or self.active not in {0, 1}):
            raise W03V2PublicQueryError("candidate projection drifted")

    def to_dict(self) -> dict[str, object]:
        return {
            "active": self.active,
            "concept_key": list(self.concept_key),
            "context_text": self.context_text,
            "license_id": self.license_id,
            "sense_key": list(self.sense_key),
            "source_commitment": self.source_commitment,
            "source_key": self.source_key,
            "source_ref_key": list(self.source_ref_key),
            "surface": self.surface,
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W03V2PublicGenerationProjection:
    """One option authorized by the current active Sense projection."""

    surface: str
    sense_key: tuple[int, ...]
    concept_key: tuple[int, ...]
    source_ref_key: tuple[int, ...]
    source_commitment: str

    def __post_init__(self) -> None:
        _text(self.surface, where="generation surface")
        for name in ("sense_key", "concept_key", "source_ref_key"):
            _strict_key(getattr(self, name), where=f"generation {name}")
        if (not isinstance(self.source_commitment, str)
                or len(self.source_commitment) != 64):
            raise W03V2PublicQueryError("generation source commitment drifted")

    def to_dict(self) -> dict[str, object]:
        return {
            "concept_key": list(self.concept_key),
            "sense_key": list(self.sense_key),
            "source_commitment": self.source_commitment,
            "source_ref_key": list(self.source_ref_key),
            "surface": self.surface,
        }


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class W03V2PublicQueryResult:
    """Safe external result without teacher labels or a prewritten answer."""

    query: W03V2PublicQuery
    status: str
    candidates: tuple[W03V2PublicCandidateProjection, ...]
    selected_sense_key: tuple[int, ...] | None
    clarify_required: int
    generation_status: str
    generation_options: tuple[W03V2PublicGenerationProjection, ...]
    source_binding_sha256: str
    experimental: int = 1
    formal_mastery_claim: int = 0
    w03_started: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.query, W03V2PublicQuery):
            raise W03V2PublicQueryError("query result request drifted")
        if self.status not in W03_V2_PUBLIC_QUERY_STATUSES:
            raise W03V2PublicQueryError("query result status drifted")
        if (not isinstance(self.candidates, tuple)
                or any(not isinstance(item, W03V2PublicCandidateProjection)
                       for item in self.candidates)
                or not isinstance(self.generation_options, tuple)
                or any(not isinstance(item, W03V2PublicGenerationProjection)
                       for item in self.generation_options)):
            raise W03V2PublicQueryError("query result projection type drifted")
        if self.selected_sense_key is not None:
            _strict_key(self.selected_sense_key, where="selected sense")
        if self.clarify_required != int(self.status in {
                W03_UNDERSTANDING_AMBIGUOUS,
                W03_UNDERSTANDING_CLARIFY,
                }):
            raise W03V2PublicQueryError("query clarify flag drifted")
        if (not isinstance(self.generation_status, str)
                or not self.generation_status
                or not isinstance(self.source_binding_sha256, str)
                or len(self.source_binding_sha256) != 64
                or (self.experimental, self.formal_mastery_claim,
                    self.w03_started) != (1, 0, 0)):
            raise W03V2PublicQueryError("query result boundary drifted")
        if self.status == W03_UNDERSTANDING_UNIQUE:
            if self.selected_sense_key is None:
                raise W03V2PublicQueryError("UNIQUE query lacks selected Sense")
        elif self.selected_sense_key is not None:
            raise W03V2PublicQueryError("non-UNIQUE query selected a Sense")
        if (self.generation_status == W03_V2_PUBLIC_QUERY_GENERATION_NOT_RUN
                and self.generation_options):
            raise W03V2PublicQueryError(
                "NOT_RUN generation cannot publish options")

    def to_dict(self) -> dict[str, object]:
        return {
            "candidates": [item.to_dict() for item in self.candidates],
            "clarify_required": self.clarify_required,
            "experimental": self.experimental,
            "formal_mastery_claim": self.formal_mastery_claim,
            "generation_options": [
                item.to_dict() for item in self.generation_options],
            "generation_status": self.generation_status,
            "query": self.query.to_dict(),
            "selected_sense_key": (
                None if self.selected_sense_key is None
                else list(self.selected_sense_key)),
            "source_binding_sha256": self.source_binding_sha256,
            "status": self.status,
            "w03_started": self.w03_started,
        }

    def sha256(self) -> str:
        return _sha(self.to_dict())


def _candidate_context_text(
        candidate: W03SenseCandidateEnvelope,
        ) -> str | None:
    anchor = candidate.anchor.extracted.provenance.to_value()
    for key in ("context", "definition_text"):
        value = anchor.get(key)
        if isinstance(value, str) and value:
            return value
    payload = candidate.observation.typed_payload.to_value()
    observed = payload.get("observed_surface")
    if isinstance(observed, dict):
        value = observed.get("text")
        if isinstance(value, str) and value:
            return value
    return None


def _candidate_projection(
        runtime: W03UnderstandingRuntime,
        candidate: W03SenseCandidateEnvelope,
        ) -> W03V2PublicCandidateProjection:
    active = any(
        item.sense == candidate.sense
        for item in runtime.consumer.lookup(
            candidate.anchor.atom, context=candidate.context)
    )
    return W03V2PublicCandidateProjection(
        candidate.anchor.extracted.surface,
        _candidate_context_text(candidate),
        candidate.sense.stable_key(),
        candidate.concept.stable_key(),
        candidate.source_record.stable_key.stable_key(),
        candidate.source_record.source_key,
        _sha(candidate.source_record.to_dict()),
        candidate.source_record.license_id,
        int(active),
    )


def _generation_projection(
        runtime: W03UnderstandingRuntime,
        candidate: W03SenseCandidateEnvelope,
        ordinal: int,
        ) -> tuple[str, tuple[W03V2PublicGenerationProjection, ...]]:
    generation = build_w03_generation_runtime(runtime)
    request = W03GenerationRequest(
        LosslessIntegerKey((303_220, ordinal)),
        candidate.sense,
        candidate.concept,
        candidate.context,
        candidate.anchor.branch,
        W03ExpressionConstraints(True, True, 256),
        candidate.source_ref,
        document_scope(candidate.source_ref),
    )
    choice = generation.choose(request)
    source_by_sense = {
        item.sense: item.source_record for item in runtime.output.candidates}
    options = tuple(
        W03V2PublicGenerationProjection(
            item.surface,
            item.sense.stable_key(),
            item.concept.stable_key(),
            source_by_sense[item.sense].stable_key.stable_key(),
            _sha(source_by_sense[item.sense].to_dict()),
        )
        for item in choice.options
    )
    return choice.status, options


def project_w03_v2_public_query(
        runtime: W03UnderstandingRuntime,
        query: W03V2PublicQuery,
        *,
        source_binding_sha256: str,
        ) -> W03V2PublicQueryResult:
    """Project one query from an existing runtime without changing its state."""
    if (not isinstance(runtime, W03UnderstandingRuntime)
            or not isinstance(query, W03V2PublicQuery)
            or not isinstance(source_binding_sha256, str)
            or len(source_binding_sha256) != 64):
        raise TypeError("public W-03 query projection inputs are invalid")
    surface_matches = tuple(sorted(
        (
            item for item in runtime.output.candidates
            if (item.anchor.extracted.surface == query.surface
                and item.anchor.extracted.branch_language == query.language)
        ),
        key=lambda item: item.sense.stable_key(),
    ))
    if not surface_matches:
        return W03V2PublicQueryResult(
            query, W03_UNDERSTANDING_UNKNOWN, (), None, 0,
            W03_V2_PUBLIC_QUERY_GENERATION_NOT_RUN, (),
            source_binding_sha256,
        )
    atoms = {item.anchor.atom for item in surface_matches}
    if len(atoms) != 1:
        raise W03V2PublicQueryError(
            "one surface/language mapped to multiple atom identities")
    atom = surface_matches[0].anchor.atom
    if query.context_text is None:
        resolution = runtime.resolve(atom)
        visible = resolution.candidates or surface_matches
        status = resolution.status
        selected = resolution.selected
        clarify = int(resolution.clarify_required)
    else:
        context_matches = tuple(
            item for item in surface_matches
            if _candidate_context_text(item) == query.context_text)
        contexts = {item.context for item in context_matches}
        if len(contexts) == 1:
            resolution = runtime.resolve(atom, context=next(iter(contexts)))
            visible = resolution.candidates or context_matches
            status = resolution.status
            selected = resolution.selected
            clarify = int(resolution.clarify_required)
        elif len(contexts) > 1:
            visible = context_matches
            status = W03_UNDERSTANDING_CLARIFY
            selected = None
            clarify = 1
        else:
            visible = surface_matches
            distinct_senses = {item.concept for item in surface_matches}
            status = (
                W03_UNDERSTANDING_CLARIFY
                if len(distinct_senses) > 1
                else W03_UNDERSTANDING_UNKNOWN
            )
            selected = None
            clarify = int(status == W03_UNDERSTANDING_CLARIFY)
    candidates = tuple(
        _candidate_projection(runtime, item) for item in visible)
    generation_status = W03_V2_PUBLIC_QUERY_GENERATION_NOT_RUN
    generation_options: tuple[W03V2PublicGenerationProjection, ...] = ()
    selected_key = None
    if status == W03_UNDERSTANDING_UNIQUE and selected is not None:
        selected_key = selected.sense.stable_key()
        selected_candidate = next(
            item for item in runtime.output.candidates
            if item.sense == selected.sense)
        if query.allow_generation:
            generation_status, generation_options = _generation_projection(
                runtime, selected_candidate, 1)
            if (generation_status == W03_GENERATION_READY
                    and not generation_options):
                raise W03V2PublicQueryError(
                    "READY generation returned no source-bound option")
    return W03V2PublicQueryResult(
        query,
        status,
        candidates,
        selected_key,
        clarify,
        generation_status,
        generation_options,
        source_binding_sha256,
    )


def run_w03_v2_public_query(
        batch: W03V2PublicEvaluationBatch,
        query: W03V2PublicQuery,
        ) -> W03V2PublicQueryResult:
    """Run one query through the shared-load multi-query public entrypoint."""
    if (not isinstance(batch, W03V2PublicEvaluationBatch)
            or not isinstance(query, W03V2PublicQuery)):
        raise TypeError("public W-03 query inputs are invalid")
    return run_w03_v2_public_queries(batch, (query,))[0]


def run_w03_v2_public_queries(
        batch: W03V2PublicEvaluationBatch,
        queries: tuple[W03V2PublicQuery, ...],
        ) -> tuple[W03V2PublicQueryResult, ...]:
    """Load the learned graph once and project a bounded read-only query batch."""
    if (not isinstance(batch, W03V2PublicEvaluationBatch)
            or not isinstance(queries, tuple) or not queries
            or any(not isinstance(item, W03V2PublicQuery)
                   for item in queries)):
        raise TypeError("public W-03 query batch inputs are invalid")
    output = adapt_w03_training_payload(batch.training_payload())
    backend = DictBackend()
    try:
        context = make_train_context(backend)
        runtime = build_w03_understanding_runtime(
            output, context.graph_ontology)
        runtime.apply_all_evidence()
        before = runtime.candidate_runtime_state_key()
        results = tuple(
            project_w03_v2_public_query(
                runtime,
                query,
                source_binding_sha256=batch.source_binding.sha256(),
            )
            for query in queries
        )
        after = runtime.candidate_runtime_state_key()
        if before != after:
            raise W03V2PublicQueryError("public query batch mutated learned state")
        return results
    finally:
        backend.close()


__all__ = [
    "W03_V2_PUBLIC_QUERY_GENERATION_NOT_RUN",
    "W03_V2_PUBLIC_QUERY_STATUSES",
    "W03_V2_PUBLIC_QUERY_VERSION",
    "W03V2PublicCandidateProjection",
    "W03V2PublicGenerationProjection",
    "W03V2PublicQuery",
    "W03V2PublicQueryError",
    "W03V2PublicQueryResult",
    "project_w03_v2_public_query",
    "run_w03_v2_public_query",
    "run_w03_v2_public_queries",
]
