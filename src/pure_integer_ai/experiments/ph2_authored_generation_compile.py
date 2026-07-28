"""把 D-02E generation seed 编译为 adoption 与 G-04 source requirement payload。"""
from __future__ import annotations

import hashlib

from pure_integer_ai.cognition.shared.generation_content import (
    AnswerContentDecision,
    AnswerContentProtocol,
)
from pure_integer_ai.cognition.shared.generation_verification import (
    GenerationSourceRequirement,
)
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
from pure_integer_ai.cognition.shared.scope_identity import (
    document_scope,
    query_scope,
)
from pure_integer_ai.cognition.shared.semantic_object import (
    context_scope_identity,
    proposition_identity,
)
from pure_integer_ai.cognition.shared.typed_binding import BoundProposition
from pure_integer_ai.crosscut.determinism.fingerprint import (
    integer_tuple_fingerprint,
)
from pure_integer_ai.experiments.ph2_authored_course_common import (
    AuthoredCompiledSeed,
)
from pure_integer_ai.experiments.ph2_authored_generation_schema import (
    AuthoredGenerationSeed,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    canonical_json_bytes,
)


_COURSE_SOURCE_KIND = 212
_EVIDENCE_SOURCE_KIND = 213
_COURSE_NAMESPACE = 21201
_VERSIONS = VersionBundle(
    CorpusVersion(1),
    ParserVersion(1),
    PrimitiveVersion(1),
    CurriculumVersion(1),
)
_STANCE_ORDER = ("ANSWER", "CLARIFY", "UNKNOWN", "REFUSE", "CONFLICT")


def _stable_positive_int(namespace: str, value: str) -> int:
    """把 generation family/seed 压为稳定正整数身份段。"""
    payload = canonical_json_bytes({"namespace": namespace, "value": value})
    result = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    result &= (1 << 63) - 1
    return result if result > 0 else 1


def _identity_key(value) -> list[int]:
    """把一等对象、来源、scope 或 typed 结构投影为整数列表。"""
    return list(value.stable_key())


def _source(seed: AuthoredGenerationSeed) -> SourceRef:
    """构造 generation candidate 的归属来源。"""
    return SourceRef(
        _COURSE_SOURCE_KIND,
        _stable_positive_int("generation-family", seed.family),
        _stable_positive_int("generation-seed", seed.seed_id),
        GLOBAL_OWNER_SCOPE,
        _VERSIONS,
    )


def _evidence_source(seed: AuthoredGenerationSeed, local_id: int) -> SourceRef:
    """构造候选实际 Evidence 的独立来源引用。"""
    return SourceRef(
        _EVIDENCE_SOURCE_KIND,
        _stable_positive_int("generation-evidence-family", seed.family),
        local_id,
        GLOBAL_OWNER_SCOPE,
        _VERSIONS,
    )


def _bound(source: SourceRef, item) -> BoundProposition:
    """从显式候选构造来源化 BoundProposition。"""
    return BoundProposition(
        proposition_identity(source, (1, item.proposition_local_id)),
        minimal_instruction_identity(
            (_COURSE_NAMESPACE, 10, 1), versions=_VERSIONS),
        concept_identity(
            (_COURSE_NAMESPACE, 20, item.predicate_kind),
            versions=_VERSIONS,
        ),
        structure_concept_identity(
            (_COURSE_NAMESPACE, 30, 1), versions=_VERSIONS),
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
    """投影无 Binder/Role 的 generation candidate。"""
    return {
        "applied_variable_keys": [],
        "bindings": [],
        "context_key": _identity_key(bound.context),
        "instruction_key": _identity_key(bound.instruction),
        "introduced_binder_keys": [],
        "predicate_key": _identity_key(bound.predicate),
        "source_anchor_key": _identity_key(bound.source_anchor),
        "structure_key": _identity_key(bound.structure),
        "template_key": _identity_key(bound.template),
    }


def _protocol() -> AnswerContentProtocol:
    """构造五种开放 stance 的真实 G-01 protocol。"""
    identities = tuple(minimal_instruction_identity(
        (_COURSE_NAMESPACE, 40, index), versions=_VERSIONS)
        for index in range(1, 6))
    return AnswerContentProtocol(*identities)


def compile_generation_seed(
        seed: AuthoredGenerationSeed) -> AuthoredCompiledSeed:
    """生成 stance/content adoption、同次 key 和 source/citation/trust 观察。"""
    if not isinstance(seed, AuthoredGenerationSeed):
        raise TypeError("compile_generation_seed 需要 AuthoredGenerationSeed")
    source = _source(seed)
    scope = query_scope(
        seed.response_scope_local_id, parent=document_scope(source))
    protocol = _protocol()
    stance_by_name = dict(zip(_STANCE_ORDER, protocol.stances()))
    candidate_rows = []
    candidate_keys = {}
    evidence_sources = {}
    for item in seed.candidates:
        bound = _bound(source, item)
        candidate_key = bound.stable_key()
        candidate_keys[item.candidate_id] = candidate_key
        evidence_sources[item.candidate_id] = {
            local_id: _evidence_source(seed, local_id)
            for local_id in item.evidence_source_ids
        }
        candidate_rows.append({
            "bound_proposition": _bound_payload(bound),
            "candidate_id": item.candidate_id,
            "candidate_key": list(candidate_key),
            "evidence_source_ids": list(item.evidence_source_ids),
            "evidence_source_keys": [
                _identity_key(evidence_sources[item.candidate_id][local_id])
                for local_id in item.evidence_source_ids
            ],
            "state": {
                "refute": item.evidence_refute,
                "support": item.evidence_support,
            },
            "owner_scope_key": _identity_key(scope),
            "owner_source_key": _identity_key(source),
        })
    run_key = integer_tuple_fingerprint(
        (*source.stable_key(), seed.same_run_local_id),
        domain="ph2.generation.same-run.v1",
    )
    selected_keys = tuple(
        candidate_keys[item] for item in seed.selected_candidate_ids)
    decision = None
    if seed.postcheck_enabled:
        decision = AnswerContentDecision(
            stance_by_name[seed.stance],
            minimal_instruction_identity(
                (_COURSE_NAMESPACE, 50, seed.logical_order), versions=_VERSIONS),
            selected_keys,
            (),
            (1, *run_key, seed.logical_order),
        )
    requirement_by_id = {
        item.candidate_id: item for item in seed.source_requirements}
    requirement_rows = []
    for candidate_id in seed.selected_candidate_ids:
        raw = requirement_by_id.get(candidate_id)
        if raw is None:
            continue
        by_local_id = evidence_sources[candidate_id]
        requirement = GenerationSourceRequirement(
            candidate_keys[candidate_id],
            source,
            scope,
            bool(raw.citation_required),
            bool(raw.trust_required),
            (1, *run_key, seed.logical_order),
            tuple(by_local_id.values()),
        )
        requirement_rows.append({
            "candidate_id": candidate_id,
            "cited_source_ids": list(raw.cited_source_ids),
            "cited_source_keys": [
                _identity_key(by_local_id[item])
                for item in raw.cited_source_ids
            ],
            "refuted_source_ids": list(raw.refuted_source_ids),
            "refuted_source_keys": [
                _identity_key(by_local_id[item])
                for item in raw.refuted_source_ids
            ],
            "requirement": {
                "candidate_key": list(requirement.candidate_key),
                "citation_required": int(requirement.citation_required),
                "evidence_source_keys": [
                    _identity_key(item) for item in requirement.evidence_sources
                ],
                "requirement_key": _identity_key(requirement),
                "scope_key": _identity_key(requirement.scope),
                "source_key": _identity_key(requirement.source),
                "trace": list(requirement.trace),
                "trust_required": int(requirement.trust_required),
            },
            "source_match": raw.source_match,
            "trusted_source_ids": list(raw.trusted_source_ids),
            "trusted_source_keys": [
                _identity_key(by_local_id[item])
                for item in raw.trusted_source_ids
            ],
        })
    payload = {
        "adoption_request": {
            "candidate_keys": [
                item["candidate_key"] for item in candidate_rows],
            "protocol_key": _identity_key(protocol),
            "same_run_key": list(run_key),
        },
        "candidate_propositions": candidate_rows,
        "consumer_request": {
            "max_candidates": seed.consumer_request.max_candidates,
            "max_evidence_sources": seed.consumer_request.max_evidence_sources,
            "max_postcheck_checks": seed.consumer_request.max_postcheck_checks,
            "max_surface_units": seed.consumer_request.max_surface_units,
        },
        "context_surface": seed.context_surface,
        "expected_authoritative": 0,
        "fixture_authoritative": 0,
        "memory_commit_enabled": 0,
        "memory_evidence_enabled": 0,
        "nonempty_surface_authoritative": 0,
        "postcheck": {
            "enabled": seed.postcheck_enabled,
            "prior_adoption": (
                None if decision is None else {
                    "reason_key": _identity_key(decision.reason),
                    "selected_artifact_keys": [],
                    "selected_candidate_ids": list(
                        seed.selected_candidate_ids),
                    "selected_candidate_keys": [
                        list(item)
                        for item in decision.selected_candidate_keys
                    ],
                    "stance": seed.stance,
                    "stance_key": _identity_key(decision.stance),
                    "trace": list(decision.trace),
                }
            ),
            "renderer_complete": seed.renderer_complete,
            "requirements": requirement_rows,
            "same_run_key": list(run_key),
            "surface_units": seed.surface_units,
        },
        "renderer_success_is_postcheck_pass": 0,
        "response_scope_key": _identity_key(scope),
        "source_ref_key": _identity_key(source),
        "surface_cue_authoritative": 0,
        "task_kind": (
            "POSTCHECK" if seed.postcheck_enabled else "ADOPTION"),
        "teacher_authoritative": 0,
    }
    typed_payload = CanonicalJsonObject.from_value(payload)
    return AuthoredCompiledSeed(
        seed.seed_id,
        seed.family,
        seed.template_family,
        seed.label_owner,
        seed.split,
        seed.sample_role,
        "GenerationAdoptionPostcheckQuery",
        typed_payload,
        seed.expected_state,
        seed.expected_payload,
        seed.perturbation_kind,
        seed.supersedes_seed_id,
        seed.logical_order,
        (seed.seed_id, payload),
        (
            seed.context_surface,
            payload["candidate_propositions"],
            payload["adoption_request"],
            payload["postcheck"],
        ),
        (
            "typed_generation_adoption_postcheck_v1",
            seed.postcheck_enabled,
            len(seed.candidates),
        ),
    )


__all__ = ["compile_generation_seed"]
