"""Join dynamic natural role hypotheses with independent, licensed teacher truth.

Syntax supplies neither entity identity nor support. A source grant and the
complete teacher relation must independently close before any Core write.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from pure_integer_ai.cognition.shared.hypothesis import EVIDENCE_SUPPORT
from pure_integer_ai.cognition.shared.identity import SourceRef
from pure_integer_ai.cognition.understanding.query_open_roles import (
    OpenRelationCandidate, OpenRoleBudgetExceeded, OpenRoleFrame,
    match_open_roles, pack_record,
)
from pure_integer_ai.experiments.independent_relation_fact_source import (
    IndependentRelationFactSource, verify_independent_fact_source_course,
)
from pure_integer_ai.experiments.integer_relation_graph_course import (
    IntegerRelationGraphCourse, integer_relation_graph_course_to_w06_adapter,
)
from pure_integer_ai.experiments.ph2_dataset_io import DatasetArtifactIOError


ADMISSION_WITNESS = 92035
ADMISSION_READY = "READY_FOR_INDEPENDENT_CORE_TRAINING"


@dataclass(frozen=True, slots=True)
class IndependentFactGrant:
    """Host-authorized input sources plus an audited, separate teacher course."""

    course: IntegerRelationGraphCourse
    source_digest_key: tuple[int, ...]
    input_sources: tuple[SourceRef, ...]
    source: IndependentRelationFactSource

    def stable_key(self) -> tuple[int, ...]:
        return pack_record(92036, self.course.source_identity, self.course.license_identity,
                           self.source_digest_key, *(source.stable_key() for source in self.input_sources))


def grant_independent_fact_inputs(
        source: IndependentRelationFactSource, input_sources: tuple[SourceRef, ...],
        ) -> IndependentFactGrant:
    """Called only by a source-preparation host, never by query/evaluator labels."""
    if type(source) is not IndependentRelationFactSource:
        raise TypeError("fact grant requires an audited independent teacher source")
    audit, course = source.source_audit, source.course
    manifest = source.pack.manifest
    if (audit.get("status") != "PASS_SOURCE_PACK_ONLY"
            or audit.get("course_scope") != "TRAIN_ONLY"
            or audit.get("evaluator_count") != 0
            or audit.get("teacher_count") != len(course.records)
            or audit.get("license_id") != "CC0-1.0"
            or manifest.splits != ("train",)
            or course.license_identity != tuple(map(ord, "CC0-1.0"))
            or course.source_identity != (
                92034, *manifest.stable_key.components, *source.source_digest_key)):
        raise ValueError("fact source/permission/teacher ownership is not closed")
    if (type(input_sources) is not tuple or not input_sources
            or any(type(item) is not SourceRef for item in input_sources)
            or len(set(input_sources)) != len(input_sources)):
        raise ValueError("fact grant requires unique explicit input sources")
    verify_independent_fact_source_course(source)
    result = IndependentFactGrant(course, source.source_digest_key, input_sources, source)
    result.stable_key()
    return result


def _meaning(parse: OpenRelationCandidate) -> tuple:
    return (parse.frame.predicate, tuple(
        (span.role, span.ordinal, span.start, span.end, span.values) for span in parse.bindings))


def _matches(parse: OpenRelationCandidate, candidate) -> bool:
    if parse.frame.predicate != candidate.proposition.predicate.stable_key():
        return False
    bindings = {item.role.stable_key(): item for item in candidate.proposition.bindings}
    endpoints = {item.identity.stable_key(): item for item in candidate.endpoints}
    if set(bindings) != set(parse.frame.roles) or len(bindings) != len(parse.bindings):
        return False
    for span in parse.bindings:
        binding = bindings[span.role]
        endpoint = endpoints.get(binding.filler.stable_key())
        if endpoint is None or tuple(map(ord, endpoint.surface_fragment)) != span.values:
            return False
    return True


@dataclass(frozen=True, slots=True)
class RelationFactAdmission:
    """No admitted course is exposed on UNKNOWN/CLARIFY/REFUSE paths."""

    status: str
    reason: str
    source: SourceRef
    parses: tuple[OpenRelationCandidate, ...] = ()
    grant: IndependentFactGrant | None = None
    course: IntegerRelationGraphCourse | None = None
    frames: tuple[OpenRoleFrame, ...] = ()

    def stable_key(self) -> tuple[int, ...]:
        """Full syntax, source, permission, teacher payload and relation identity."""
        if self.status != ADMISSION_READY or self.course is None or self.grant is None:
            raise ValueError("only independently admitted facts have a Core witness")
        return pack_record(
            ADMISSION_WITNESS, self.source.stable_key(), self.grant.stable_key(),
            pack_record(92041, *(frame.stable_key() for frame in self.frames)),
            pack_record(92037, *(parse.stable_key() for parse in self.parses)),
            pack_record(92038, *(row.stable_key() for row in self.course.records)))


def admit_independent_relation_fact(
        surface: str, source: SourceRef, frames: tuple[OpenRoleFrame, ...], *,
        grant: IndependentFactGrant | None,
        max_parse_steps: int = 65536, max_witness_integers: int = 65536,
        ) -> RelationFactAdmission:
    """Parse complete input, resolve ambiguity, then check separate teacher truth."""
    if type(source) is not SourceRef or not isinstance(surface, str) or not surface:
        raise ValueError("fact admission requires a natural surface and source identity")
    if (type(frames) is not tuple or any(type(frame) is not OpenRoleFrame for frame in frames)
            or type(max_witness_integers) is not int or max_witness_integers <= 0):
        raise ValueError("fact admission frame or witness budget is invalid")
    def reject(status, reason, parses=()):
        return RelationFactAdmission(status, reason, source, parses)
    if grant is None:
        return reject("UNKNOWN", "NO_INDEPENDENT_TEACHER_EVIDENCE")
    if type(grant) is not IndependentFactGrant or source not in grant.input_sources:
        return reject("REFUSE", "INPUT_SOURCE_NOT_AUTHORIZED")
    try:
        checked = grant_independent_fact_inputs(grant.source, grant.input_sources)
        if checked != grant:
            return reject("UNKNOWN", "INDEPENDENT_GRANT_DRIFT")
    except (OSError, TypeError, ValueError, DatasetArtifactIOError):
        return reject("UNKNOWN", "INDEPENDENT_SOURCE_REVALIDATION_FAILED")
    frames = tuple(sorted(set(frames), key=lambda item: item.stable_key()))
    try:
        parses = match_open_roles(frames, tuple(map(ord, surface)), source.stable_key(),
                                  max_steps=max_parse_steps)
    except OpenRoleBudgetExceeded:
        return reject("UNKNOWN", "COMPLETE_PARSE_BUDGET_EXCEEDED")
    meanings = {_meaning(parse) for parse in parses}
    if not meanings:
        return reject("UNKNOWN", "NO_COMPLETE_TRAINED_ROLE_PARSE")
    if len(meanings) != 1:
        return reject("CLARIFY", "MULTIPLE_COMPLETE_ROLE_INTERPRETATIONS", parses)
    # Adapter validation consumes independent teacher fields; parse contains none.
    adapter = integer_relation_graph_course_to_w06_adapter(
        grant.course, require_full_curriculum=False)
    candidates = tuple(item for item in adapter.candidates if _matches(parses[0], item))
    propositions = {item.proposition.proposition for item in candidates}
    if not propositions:
        return reject("UNKNOWN", "NO_INDEPENDENT_FACT_FOR_CURRENT_ROLES", parses)
    if len(propositions) != 1:
        return reject("CLARIFY", "INDEPENDENT_ENTITY_IDENTITY_AMBIGUOUS", parses)
    # A candidate may have multiple independent observation envelopes. Inspect
    # every teacher account for this proposition, including later refutations.
    observation_keys = {item.observation.stable_key for item in adapter.observations
                        if item.candidate.proposition.proposition in propositions}
    evidence = tuple(item for item in adapter.evidence if item.candidate in propositions)
    if ({item.observation.stable_key for item in evidence} != observation_keys
            or any(item.stances != (EVIDENCE_SUPPORT,) or item.withdrawal_level
                   or item.supersedes_observation_key is not None for item in evidence)):
        return reject("UNKNOWN", "INDEPENDENT_EVIDENCE_NOT_UNCONTESTED_SUPPORT", parses)
    records = tuple(row for row in grant.course.records
                    if row.observation_key in {key.components for key in observation_keys})
    course = replace(grant.course, records=records)
    result = RelationFactAdmission(ADMISSION_READY, "INDEPENDENT_SOURCE_ROLE_EVIDENCE_JOIN",
                                   source, parses, grant, course, frames)
    if len(result.stable_key()) > max_witness_integers:
        return reject("UNKNOWN", "FULL_ADMISSION_WITNESS_BUDGET_EXCEEDED", parses)
    return result


def learn_independent_relation_fact_admission(backend, context, admission: RelationFactAdmission):
    """The real W-06 learner persists the complete admission witness in Evidence."""
    from pure_integer_ai.experiments.conversation_typed_relation_bridge import _materialize_relation_spans
    from pure_integer_ai.experiments.ph2_w06_learning import build_w06_learning_runtime

    witness = admission.stable_key()  # Reject every non-admitted result before writes.
    if not admission.parses:
        raise ValueError("independent fact admission has no complete role witness")
    repeated = admit_independent_relation_fact(
        "".join(map(chr, admission.parses[0].values)), admission.source,
        admission.frames,
        grant=admission.grant, max_witness_integers=len(witness))
    if repeated != admission:
        raise ValueError("independent fact admission drifted before Core consumption")
    adapter = integer_relation_graph_course_to_w06_adapter(
        admission.course, require_full_curriculum=False)
    adapter = replace(adapter, evidence=tuple(
        replace(binding, reason_key=pack_record(92039, witness, binding.reason_key))
        for binding in adapter.evidence))
    context.graph_ontology.enable_physical_statement_projection()
    runtime = build_w06_learning_runtime(backend, adapter, context=context)
    _materialize_relation_spans(context, runtime, adapter)
    return runtime


__all__ = [
    "ADMISSION_READY", "IndependentFactGrant", "RelationFactAdmission",
    "grant_independent_fact_inputs", "admit_independent_relation_fact",
    "learn_independent_relation_fact_admission",
]
