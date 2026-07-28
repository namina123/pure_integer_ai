"""把已核准 W-05 seed 投影为现役纯整数语义对象和 D-02 typed payload。"""
from __future__ import annotations

import hashlib

from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    OBJECT_ENTITY,
    CorpusVersion,
    CurriculumVersion,
    ObjectIdentity,
    ParserVersion,
    PrimitiveVersion,
    SourceRef,
    VersionBundle,
    concept_identity,
    occurrence_identity,
)
from pure_integer_ai.cognition.shared.semantic_object import (
    AtomicPropositionDefinition,
    AtomicRoleBinding,
    context_scope_identity,
    entity_identity,
    event_identity,
    proposition_identity,
    role_identity,
)
from pure_integer_ai.experiments.ph2_authored_atomic_schema import (
    AuthoredAtomicSeed,
)
from pure_integer_ai.experiments.ph2_authored_course_common import (
    AuthoredCompiledSeed,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    canonical_json_bytes,
)


_COURSE_SOURCE_KIND = 205
_COURSE_NAMESPACE = 20501
_VERSIONS = VersionBundle(
    CorpusVersion(1),
    ParserVersion(1),
    PrimitiveVersion(1),
    CurriculumVersion(1),
)


def _stable_positive_int(namespace: str, value: str) -> int:
    """把来源 family/seed id 压为稳定正整数，不把自然语言 cue 写入 registry。"""
    payload = canonical_json_bytes({"namespace": namespace, "value": value})
    result = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    result &= (1 << 63) - 1
    return result if result > 0 else 1


def _identity_key(identity: ObjectIdentity) -> list[int]:
    """把已由现役构造器核验的一等对象身份投影为规范整数列表。"""
    return list(identity.stable_key())


def compile_atomic_seed(seed: AuthoredAtomicSeed) -> AuthoredCompiledSeed:
    """构造一条候选 Proposition、RoleBinding、ContextScope 和 occurrence payload。"""
    if not isinstance(seed, AuthoredAtomicSeed):
        raise TypeError("compile_atomic_seed 需要 AuthoredAtomicSeed")
    source = SourceRef(
        _COURSE_SOURCE_KIND,
        _stable_positive_int("atomic-family", seed.family),
        _stable_positive_int("atomic-seed", seed.seed_id),
        GLOBAL_OWNER_SCOPE,
        _VERSIONS,
    )
    occurrence_identities = {
        item.occurrence_id: occurrence_identity(
            source,
            start=item.start,
            end=item.end,
            ordinal=item.ordinal,
        )
        for item in seed.occurrences
    }
    semantic_identities = {
        item.occurrence_id: (
            entity_identity(source, (1, item.semantic_local_id))
            if item.semantic_kind == OBJECT_ENTITY
            else event_identity(source, (1, item.semantic_local_id))
        )
        for item in seed.occurrences
    }
    proposition = proposition_identity(source, (1, 1))
    predicate = concept_identity(
        (_COURSE_NAMESPACE, 1, seed.predicate_kind),
        versions=_VERSIONS,
    )
    context = context_scope_identity(source, (1, seed.context_local_id))
    bindings = tuple(
        AtomicRoleBinding(
            role_identity(
                (_COURSE_NAMESPACE, 1, item.role_kind),
                versions=_VERSIONS,
            ),
            semantic_identities[item.filler_occurrence_id],
            item.ordinal,
        )
        for item in seed.bindings
    )
    definition = AtomicPropositionDefinition(
        proposition,
        predicate,
        occurrence_identities[seed.predicate_occurrence_id],
        context,
        bindings,
    )
    occurrence_payload = []
    occurrence_by_id = {item.occurrence_id: item for item in seed.occurrences}
    for occurrence_id in seed.occurrence_order:
        item = occurrence_by_id[occurrence_id]
        occurrence_payload.append({
            "end": item.end,
            "identity_key": _identity_key(occurrence_identities[occurrence_id]),
            "ordinal": item.ordinal,
            "semantic_key": _identity_key(semantic_identities[occurrence_id]),
            "semantic_kind": item.semantic_kind,
            "start": item.start,
            "surface_fragment": item.surface_fragment,
        })
    role_payload = []
    for binding in definition.canonical_bindings():
        role_payload.append({
            "binding_key": _identity_key(
                binding.identity_for(definition.proposition)),
            "filler_key": _identity_key(binding.filler),
            "ordinal": binding.ordinal,
            "role_key": _identity_key(binding.role),
        })
    typed_payload = CanonicalJsonObject.from_value({
        "candidate_definition": {
            "context_key": _identity_key(definition.context),
            "predicate_key": _identity_key(definition.predicate),
            "proposition_key": _identity_key(definition.proposition),
            "role_bindings": role_payload,
            "source_anchor_key": _identity_key(definition.source_anchor),
        },
        "occurrence_order": [
            _identity_key(occurrence_identities[item])
            for item in seed.occurrence_order
        ],
        "occurrences": occurrence_payload,
        "query_kind": "occurrence_role_atomic_proposition",
        "surface": seed.surface,
    })
    payload_value = typed_payload.to_value()
    return AuthoredCompiledSeed(
        seed.seed_id,
        seed.family,
        seed.template_family,
        seed.label_owner,
        seed.split,
        seed.sample_role,
        "AtomicPropositionQuery",
        typed_payload,
        seed.expected_state,
        seed.expected_payload,
        seed.perturbation_kind,
        seed.supersedes_seed_id,
        seed.logical_order,
        (seed.surface, payload_value),
        (seed.surface, payload_value["candidate_definition"]),
        (
            "atomic_proposition_query_v1",
            len(seed.occurrences),
            len(seed.bindings),
            seed.perturbation_kind,
        ),
    )


__all__ = ["compile_atomic_seed"]
