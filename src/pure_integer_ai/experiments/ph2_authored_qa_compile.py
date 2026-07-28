"""把 D-02E QA seed 编译为真实 QuestionRequest 和候选采用依据。"""
from __future__ import annotations

import hashlib

from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    CorpusVersion,
    CurriculumVersion,
    ParserVersion,
    PrimitiveVersion,
    SourceRef,
    VersionBundle,
    concept_identity,
    minimal_instruction_identity,
    occurrence_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.logic_executor import LogicEvidenceState
from pure_integer_ai.cognition.shared.question_answer import QuestionRequest
from pure_integer_ai.cognition.shared.scope_identity import (
    document_scope,
    query_scope,
)
from pure_integer_ai.cognition.shared.semantic_object import (
    context_scope_identity,
    proposition_identity,
)
from pure_integer_ai.cognition.shared.typed_binding import BoundProposition
from pure_integer_ai.experiments.ph2_authored_course_common import (
    AuthoredCompiledSeed,
)
from pure_integer_ai.experiments.ph2_authored_qa_schema import AuthoredQASeed
from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    canonical_json_bytes,
)


_COURSE_SOURCE_KIND = 211
_COURSE_NAMESPACE = 21101
_VERSIONS = VersionBundle(
    CorpusVersion(1),
    ParserVersion(1),
    PrimitiveVersion(1),
    CurriculumVersion(1),
)


def _stable_positive_int(namespace: str, value: str) -> int:
    """把 QA family/seed/kind 压为稳定正整数身份段。"""
    payload = canonical_json_bytes({"namespace": namespace, "value": value})
    result = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    result &= (1 << 63) - 1
    return result if result > 0 else 1


def _identity_key(value) -> list[int]:
    """把来源、scope 或一等对象投影为规范整数列表。"""
    return list(value.stable_key())


def _source(seed: AuthoredQASeed) -> SourceRef:
    """构造同 family 隔离、同 seed 稳定的 QA 来源。"""
    return SourceRef(
        _COURSE_SOURCE_KIND,
        _stable_positive_int("qa-family", seed.family),
        _stable_positive_int("qa-seed", seed.seed_id),
        GLOBAL_OWNER_SCOPE,
        _VERSIONS,
    )


def _bound(source: SourceRef, item, *, family: int) -> BoundProposition:
    """从显式 target/candidate seed 构造来源化 BoundProposition。"""
    return BoundProposition(
        proposition_identity(source, (family, item.proposition_local_id)),
        minimal_instruction_identity(
            (_COURSE_NAMESPACE, 10, family), versions=_VERSIONS),
        concept_identity(
            (_COURSE_NAMESPACE, 20, item.predicate_kind),
            versions=_VERSIONS,
        ),
        structure_concept_identity(
            (_COURSE_NAMESPACE, 30, family), versions=_VERSIONS),
        occurrence_identity(
            source,
            start=item.start,
            end=item.end,
            ordinal=item.ordinal,
        ),
        context_scope_identity(source, (1, 1)),
        (),
        (),
        (),
    )


def _bound_payload(bound: BoundProposition) -> dict:
    """投影 QA target/candidate 的完整 typed bound identity。"""
    return {
        "applied_variable_keys": [
            _identity_key(item) for item in bound.applied_variables],
        "bindings": [],
        "context_key": _identity_key(bound.context),
        "instruction_key": _identity_key(bound.instruction),
        "introduced_binder_keys": [
            _identity_key(item) for item in bound.introduced_binders],
        "predicate_key": _identity_key(bound.predicate),
        "source_anchor_key": _identity_key(bound.source_anchor),
        "structure_key": _identity_key(bound.structure),
        "template_key": _identity_key(bound.template),
    }


def _reference_payload(seed: AuthoredQASeed, source: SourceRef) -> dict | None:
    """投影 A-01 candidate/adopted 集合和 singleton winner。"""
    resolution = seed.reference_resolution
    if resolution is None:
        return None
    occurrences = {
        item.occurrence_id: occurrence_identity(
            source,
            start=item.start,
            end=item.end,
            ordinal=item.ordinal,
        )
        for item in seed.occurrences
    }
    return {
        "adopted_occurrence_ids": list(resolution.adopted_occurrence_ids),
        "adopted_occurrence_keys": [
            _identity_key(occurrences[item])
            for item in resolution.adopted_occurrence_ids
        ],
        "candidate_occurrence_ids": list(
            resolution.candidate_occurrence_ids),
        "candidate_occurrence_keys": [
            _identity_key(occurrences[item])
            for item in resolution.candidate_occurrence_ids
        ],
        "reference_occurrence_id": resolution.reference_occurrence_id,
        "reference_occurrence_key": _identity_key(
            occurrences[resolution.reference_occurrence_id]),
        "singleton_winner_required": 1,
        "stable_order_tiebreaker": 0,
        "winner_occurrence_id": resolution.winner_occurrence_id,
        "winner_occurrence_key": (
            None if not resolution.winner_occurrence_id
            else _identity_key(
                occurrences[resolution.winner_occurrence_id])
        ),
    }


def compile_qa_seed(seed: AuthoredQASeed) -> AuthoredCompiledSeed:
    """生成 F-00 request、候选 Proposition、Evidence 采用和 reference 边界。"""
    if not isinstance(seed, AuthoredQASeed):
        raise TypeError("compile_qa_seed 需要 AuthoredQASeed")
    source = _source(seed)
    document = document_scope(source)
    evidence_scope = query_scope(
        seed.query_scope_local_id, parent=document)
    response_scope = query_scope(
        seed.response_scope_local_id, parent=document)
    target = _bound(source, seed.target, family=1)
    kind_component = _stable_positive_int("qa-kind", seed.question_kind)
    request = QuestionRequest(
        minimal_instruction_identity(
            (_COURSE_NAMESPACE, 1, kind_component), versions=_VERSIONS),
        minimal_instruction_identity(
            (_COURSE_NAMESPACE, 2, kind_component), versions=_VERSIONS),
        minimal_instruction_identity(
            (_COURSE_NAMESPACE, 3, kind_component), versions=_VERSIONS),
        target,
        LogicEvidenceState(
            bool(seed.required_support), bool(seed.required_refute)),
        evidence_scope,
        response_scope,
        (1, seed.logical_order, kind_component),
    )
    candidates = []
    for item in seed.candidates:
        matches_target = (
            item.proposition_local_id == seed.target.proposition_local_id
            and item.predicate_kind == seed.target.predicate_kind
            and item.start == seed.target.start
            and item.end == seed.target.end
            and item.ordinal == seed.target.ordinal
        )
        bound = target if matches_target else _bound(source, item, family=2)
        candidates.append({
            "bound_proposition": _bound_payload(bound),
            "candidate_id": item.candidate_id,
            "candidate_key": _identity_key(bound),
            "competition_key": [1, item.competition_id],
            "evidence_ids": list(item.evidence_ids),
            "matches_request_target": int(matches_target),
            "state": {
                "refute": item.evidence_refute,
                "support": item.evidence_support,
            },
        })
    payload = {
        "candidate_propositions": candidates,
        "consumer_request": {
            "max_candidates": seed.consumer_request.max_candidates,
            "max_context_chars": seed.consumer_request.max_context_chars,
            "max_evidence_items": seed.consumer_request.max_evidence_items,
            "max_occurrences": seed.consumer_request.max_occurrences,
        },
        "context_surface": seed.context_surface,
        "fixture_authoritative": 0,
        "nonempty_output_authoritative": 0,
        "question_kind": seed.question_kind,
        "question_request": {
            "evidence_scope_key": _identity_key(request.evidence_scope),
            "goal_kind_key": _identity_key(request.goal_kind),
            "intent_key": _identity_key(request.intent),
            "query_kind_key": _identity_key(request.query_kind),
            "request_key": _identity_key(request),
            "required_state": {
                "refute": int(request.required.refute),
                "support": int(request.required.support),
            },
            "response_scope_key": _identity_key(request.response_scope),
            "target": _bound_payload(request.target),
            "trace": list(request.trace),
        },
        "question_surface": seed.question_surface,
        "reference_resolution": _reference_payload(seed, source),
        "route_registered": int(seed.route_status == "REGISTERED"),
        "route_status": seed.route_status,
        "selection_basis": {
            "candidate_evidence_state_authoritative": 1,
            "conflict_precedes_answer": 1,
            "expected_authoritative": 0,
            "required_direction_authoritative": 1,
            "teacher_authoritative": 0,
            "unique_reference_winner_required": 1,
        },
        "source_ref_key": _identity_key(source),
        "surface_cue_authoritative": 0,
    }
    typed_payload = CanonicalJsonObject.from_value(payload)
    return AuthoredCompiledSeed(
        seed.seed_id,
        seed.family,
        seed.template_family,
        seed.label_owner,
        seed.split,
        seed.sample_role,
        "QuestionExecutionQuery",
        typed_payload,
        seed.expected_state,
        seed.expected_payload,
        seed.perturbation_kind,
        seed.supersedes_seed_id,
        seed.logical_order,
        (seed.seed_id, payload),
        (
            seed.context_surface,
            seed.question_surface,
            payload["question_request"],
            payload["candidate_propositions"],
            payload["reference_resolution"],
        ),
        (
            "typed_qa_v1",
            seed.question_kind,
            seed.route_status,
            len(seed.candidates),
            len(seed.occurrences),
        ),
    )


__all__ = ["compile_qa_seed"]
