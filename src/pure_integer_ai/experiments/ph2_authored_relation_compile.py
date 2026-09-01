"""把已核准 D-02C relation seed 编译为现役 RelationSchema 和命题 payload。"""
from __future__ import annotations

import hashlib

from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    OBJECT_CONCEPT,
    OBJECT_ENTITY,
    OBJECT_EVENT,
    OBJECT_OCCURRENCE,
    OBJECT_PROPOSITION,
    OBJECT_SET_EXPR,
    CorpusVersion,
    CurriculumVersion,
    ObjectIdentity,
    ParserVersion,
    PrimitiveVersion,
    SourceRef,
    VersionBundle,
    concept_identity,
    minimal_instruction_identity,
    occurrence_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.semantic_object import (
    AtomicPropositionDefinition,
    AtomicRoleBinding,
    context_scope_identity,
    entity_identity,
    event_identity,
    proposition_identity,
    role_identity,
    set_expr_identity,
)
from pure_integer_ai.cognition.shared.typed_relation import (
    RelationSchema,
    RelationSlotSchema,
    SameKindConstraint,
)
from pure_integer_ai.experiments.ph2_authored_course_common import (
    AuthoredCompiledSeed,
)
from pure_integer_ai.experiments.ph2_authored_relation_schema import (
    AuthoredRelationSeed,
    DIRECTION_SYMMETRIC,
    REQUEST_MEREOLOGY_QUERY,
    REQUEST_PROPERTY_SELECTION,
    REQUEST_SYMMETRIC_PAIR_QUERY,
    REQUEST_EVENT_TIME_VERIFICATION,
    RELATION_EVENT_AFTER,
    RELATION_EVENT_BEFORE,
    RELATION_EVENT_SAME,
    RELATION_EVENT_UNKNOWN,
    REQUEST_CAUSAL_VERIFICATION,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    canonical_json_bytes,
)


_COURSE_SOURCE_KIND = 206
_COURSE_NAMESPACE = 20601
_VERSIONS = VersionBundle(
    CorpusVersion(1),
    ParserVersion(1),
    PrimitiveVersion(1),
    CurriculumVersion(1),
)


def _stable_positive_int(namespace: str, value: str) -> int:
    """把 relation family/seed id 压为稳定正整数身份段。"""
    payload = canonical_json_bytes({"namespace": namespace, "value": value})
    result = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    result &= (1 << 63) - 1
    return result if result > 0 else 1


def _identity_key(identity: ObjectIdentity) -> list[int]:
    """把一等对象身份投影为规范严格整数列表。"""
    return list(identity.stable_key())


def authored_relation_identity(kind: int) -> ObjectIdentity:
    """按冻结 D-02C 坐标恢复一等 relation identity。"""
    if type(kind) is not int or kind <= 0:
        raise ValueError("authored relation kind 必须是正严格整数")
    return concept_identity(
        (_COURSE_NAMESPACE, 1, kind), versions=_VERSIONS)


def authored_relation_schema_identity(
        kind: int,
        *,
        slot_signature: tuple[tuple[int, tuple[int, ...]], ...] | None = None,
        ) -> ObjectIdentity:
    """恢复一等 schema identity，并把严格 slot 形状纳入身份。

    ``schema_kind`` 是课程级语义坐标，但一个 typed schema 的允许对象类型
    也是定义的一部分。反向/扰动样本可能在同一 relation family 中改变
    endpoint 类型；若仍只按 ``schema_kind`` 编址，会把两个不同 schema
    错误地合并。保留无签名调用的冻结旧身份，同时让编译后的 payload 对
    不同 slot 形状拥有可复现且独立的一等身份。
    """
    if type(kind) is not int or kind <= 0:
        raise ValueError("authored schema kind 必须是正严格整数")
    components = [_COURSE_NAMESPACE, 2, kind]
    if slot_signature is not None:
        if (not isinstance(slot_signature, tuple)
                or any(not isinstance(role, int)
                       or not isinstance(kinds, tuple)
                       or not kinds
                       or any(type(item) is not int for item in kinds)
                       for role, kinds in slot_signature)):
            raise ValueError("authored schema slot_signature 非法")
        normalized_signature = tuple(sorted(
            ((role, tuple(sorted(kinds))) for role, kinds in slot_signature),
            key=lambda item: item[0],
        ))
        shape_token = _stable_positive_int(
            "relation-schema-shape",
            "|".join(
                f"{role}:{','.join(str(item) for item in kinds)}"
                for role, kinds in normalized_signature),
        )
        components.extend((3, shape_token))
    return structure_concept_identity(
        tuple(components), versions=_VERSIONS)


def authored_relation_role_identity(kind: int) -> ObjectIdentity:
    """按冻结 D-02C 坐标恢复一等 Role identity。"""
    if type(kind) is not int or kind <= 0:
        raise ValueError("authored Role kind 必须是正严格整数")
    return role_identity(
        (_COURSE_NAMESPACE, 3, kind), versions=_VERSIONS)


def authored_relation_rule_identity(kind: int) -> ObjectIdentity:
    """按冻结 D-02C 坐标恢复一等 MinimalInstruction rule identity。"""
    if type(kind) is not int or kind <= 0:
        raise ValueError("authored rule kind 必须是正严格整数")
    return minimal_instruction_identity(
        (_COURSE_NAMESPACE, 5, kind), versions=_VERSIONS)


def _endpoint_identity(seed, source: SourceRef) -> ObjectIdentity:
    """按 endpoint object kind 调用现役身份构造器，不走裸 ObjectIdentity 拼装。"""
    if seed.object_kind == OBJECT_CONCEPT:
        assert seed.local_id is not None
        return concept_identity(
            (_COURSE_NAMESPACE, 10, seed.local_id), versions=_VERSIONS)
    if seed.object_kind == OBJECT_ENTITY:
        assert seed.local_id is not None
        return entity_identity(source, (10, seed.local_id))
    if seed.object_kind == OBJECT_EVENT:
        assert seed.local_id is not None
        return event_identity(source, (10, seed.local_id))
    if seed.object_kind == OBJECT_OCCURRENCE:
        return occurrence_identity(
            source, start=seed.start, end=seed.end, ordinal=seed.ordinal)
    if seed.object_kind == OBJECT_PROPOSITION:
        assert seed.local_id is not None
        return proposition_identity(source, (10, seed.local_id))
    if seed.object_kind == OBJECT_SET_EXPR:
        assert seed.local_id is not None
        return set_expr_identity(source, (10, seed.local_id))
    raise ValueError("relation endpoint kind 未由 typed compiler 支持")


def _allowed_kinds(
        seed: AuthoredRelationSeed, binding, endpoint, *,
        use_relation_profiles: bool = True,
        ) -> frozenset[int]:
    """恢复 relation profile 声明的稳定 Role 类型集合。

    authored seed 允许省略重复的 ``allowed_object_kinds``；此时不能把
    当前 endpoint 的实际类型误当作 schema 定义，否则同一 schema 会因
    反向/替换样本而漂移。W-06 profile 是类型合同的唯一注册源；对尚未
    注册的历史课程保留旧的单 endpoint 回退，避免改变其既有编译语义。
    """
    # Keep an intentional TYPE_MISMATCH payload self-consistent so it can be
    # round-tripped as data.  W-06 performs the semantic rejection against the
    # registered profile before candidate formation.
    if seed.perturbation_kind == "TYPE_MISMATCH":
        return frozenset({endpoint.object_kind})
    if binding.allowed_object_kinds:
        return frozenset(binding.allowed_object_kinds)
    if not use_relation_profiles:
        return frozenset({endpoint.object_kind})
    try:
        from pure_integer_ai.experiments.ph2_w06_source_semantic import (
            W06_RELATION_PROFILES,
        )
        profile = W06_RELATION_PROFILES.get(seed.relation_family)
        if profile is not None:
            return profile.allowed_for_role(binding.role_kind)
    except (ImportError, KeyError):
        pass
    return frozenset({endpoint.object_kind})


def compile_relation_seed(
        seed: AuthoredRelationSeed, *,
        rational_role_values: tuple[tuple[int, int, int], ...] = (),
        use_relation_profiles: bool = True,
        ) -> AuthoredCompiledSeed:
    """生成 candidate relation、schema、RoleBinding 和 consumer request payload。"""
    if not isinstance(seed, AuthoredRelationSeed):
        raise TypeError("compile_relation_seed 需要 AuthoredRelationSeed")
    source = SourceRef(
        _COURSE_SOURCE_KIND,
        _stable_positive_int("relation-family", seed.family),
        _stable_positive_int("relation-seed", seed.seed_id),
        GLOBAL_OWNER_SCOPE,
        _VERSIONS,
    )
    relation = authored_relation_identity(seed.relation_kind)
    endpoints = {
        item.endpoint_id: _endpoint_identity(item, source)
        for item in seed.endpoints
    }
    roles = {
        item.role_kind: authored_relation_role_identity(item.role_kind)
        for item in seed.bindings
    }
    slots = tuple(
        RelationSlotSchema(
            roles[item.role_kind],
            _allowed_kinds(
                seed,
                item,
                endpoints[item.endpoint_id],
                use_relation_profiles=use_relation_profiles,
            ),
            1,
            1,
        )
        for item in seed.bindings
    )
    slot_signature = tuple(
        (item.role_kind, tuple(sorted(slot.allowed_object_kinds)))
        for item, slot in zip(seed.bindings, slots)
    )
    # Keep the frozen schema identity for profile-conforming seeds (this
    # preserves existing W-06 artifacts).  Only a genuinely non-profile
    # shape gets an identity extension; that prevents collisions without
    # rewriting compatible published packs.
    profile_shape = None
    if use_relation_profiles:
        try:
            from pure_integer_ai.experiments.ph2_w06_source_semantic import (
                W06_RELATION_PROFILES,
            )
            profile = W06_RELATION_PROFILES.get(seed.relation_family)
            if profile is not None:
                profile_shape = tuple(sorted(
                    (role, tuple(sorted(allowed)))
                    for role, allowed in profile.role_object_kinds))
        except ImportError:
            profile_shape = None
    normalized_shape = tuple(sorted(slot_signature))
    schema_identity = authored_relation_schema_identity(
        seed.schema_kind,
        slot_signature=slot_signature
        if profile_shape is not None and normalized_shape != profile_shape
        else None,
    )
    constraints = ()
    if seed.directionality == DIRECTION_SYMMETRIC:
        constraints = (SameKindConstraint(
            structure_concept_identity(
                (_COURSE_NAMESPACE, 4, seed.relation_kind),
                versions=_VERSIONS,
            ),
            tuple(roles[item.role_kind] for item in seed.bindings),
        ),)
    relation_schema = RelationSchema(
        schema_identity,
        relation,
        slots,
        constraints,
    )
    proposition = proposition_identity(source, (1, 1))
    definition = AtomicPropositionDefinition(
        proposition,
        relation,
        occurrence_identity(
            source,
            start=seed.anchor.start,
            end=seed.anchor.end,
            ordinal=seed.anchor.ordinal,
        ),
        context_scope_identity(source, (1, seed.context_local_id)),
        tuple(
            AtomicRoleBinding(
                roles[item.role_kind],
                endpoints[item.endpoint_id],
                item.ordinal,
            )
            for item in seed.bindings
        ),
    )
    # TYPE_MISMATCH is an intentional negative proposal.  Preserve its full
    # typed payload so W-06 can route it through the schema-rejection firewall;
    # validating it here would erase the very evidence the adapter must audit.
    if seed.perturbation_kind != "TYPE_MISMATCH":
        relation_schema.validate_definition(definition)
    endpoint_payload = [
        {
            "end": item.end,
            "endpoint_key": _identity_key(endpoints[item.endpoint_id]),
            "object_kind": item.object_kind,
            "ordinal": item.ordinal,
            "start": item.start,
            "surface_fragment": item.surface_fragment,
        }
        for item in seed.endpoints
    ]
    binding_payload = [
        {
            "binding_key": _identity_key(
                binding.identity_for(definition.proposition)),
            "filler_key": _identity_key(binding.filler),
            "ordinal": binding.ordinal,
            "role_key": _identity_key(binding.role),
        }
        for binding in definition.canonical_bindings()
    ]
    slot_payload = [
        {
            "allowed_object_kinds": sorted(slot.allowed_object_kinds),
            "max_count": slot.max_count,
            "min_count": slot.min_count,
            "role_key": _identity_key(slot.role),
        }
        for slot in relation_schema.slots
    ]
    constraint_payload = [
        {
            "constraint_key": _identity_key(item.constraint),
            "role_keys": [_identity_key(role) for role in item.roles],
        }
        for item in relation_schema.same_kind_constraints
    ]
    request = seed.consumer_request
    if request.request_kind == REQUEST_CAUSAL_VERIFICATION:
        assert request.max_evidence_requests is not None
        assert request.max_relations is not None
        assert request.max_witness_inputs is not None
        consumer_payload = {
            "budget": {
                "max_evidence_requests": request.max_evidence_requests,
                "max_relations": request.max_relations,
                "max_witness_inputs": request.max_witness_inputs,
            },
            "candidate_endpoints": {
                "cause_key": _identity_key(
                    endpoints[request.origin_endpoint_id]),
                "effect_key": _identity_key(
                    endpoints[request.effect_endpoint_id]),
            },
            "causal_relation_key": _identity_key(relation_schema.relation),
            "request_kind": request.request_kind,
            "temporal_relations": [
                _identity_key(authored_relation_identity(kind))
                for kind in (
                    RELATION_EVENT_BEFORE,
                    RELATION_EVENT_AFTER,
                    RELATION_EVENT_SAME,
                    RELATION_EVENT_UNKNOWN,
                )
            ],
        }
    elif request.request_kind == REQUEST_EVENT_TIME_VERIFICATION:
        assert request.max_evidence_requests is not None
        assert request.max_relations is not None
        consumer_payload = {
            "budget": {
                "max_evidence_requests": request.max_evidence_requests,
                "max_relations": request.max_relations,
            },
            "candidate_endpoints": {
                "object_key": _identity_key(
                    endpoints[request.object_endpoint_id]),
                "subject_key": _identity_key(
                    endpoints[request.origin_endpoint_id]),
            },
            "relations": [_identity_key(relation_schema.relation)],
            "request_kind": request.request_kind,
        }
    elif request.request_kind == REQUEST_SYMMETRIC_PAIR_QUERY:
        assert request.max_options is not None
        assert request.max_total_direct_facts is not None
        consumer_payload = {
            "budget": {
                "max_direct_facts": request.max_facts,
                "max_options": request.max_options,
                "max_total_direct_facts": request.max_total_direct_facts,
            },
            "pattern": {
                "counterpart_key": _identity_key(
                    endpoints[request.counterpart_endpoint_id]),
                "endpoint_key": _identity_key(
                    endpoints[request.origin_endpoint_id]),
                "relation_key": _identity_key(relation_schema.relation),
            },
            "request_kind": request.request_kind,
        }
    elif request.request_kind == REQUEST_MEREOLOGY_QUERY:
        assert request.max_options is not None
        assert request.max_closure_statements is not None
        assert request.max_rule_applications is not None
        consumer_payload = {
            "budget": {
                "max_closure_statements": request.max_closure_statements,
                "max_direct_facts": request.max_facts,
                "max_options": request.max_options,
                "max_rule_applications": request.max_rule_applications,
            },
            "pattern": {
                "part_key": _identity_key(
                    endpoints[request.origin_endpoint_id]),
                "relation_key": _identity_key(relation_schema.relation),
                "whole_key": _identity_key(
                    endpoints[request.whole_endpoint_id]),
            },
            "request_kind": request.request_kind,
        }
    elif request.request_kind == REQUEST_PROPERTY_SELECTION:
        assert request.max_options is not None
        consumer_payload = {
            "budget": {
                "max_direct_facts": request.max_facts,
                "max_options": request.max_options,
            },
            "pattern": {
                "attribute_key": _identity_key(
                    endpoints[request.attribute_endpoint_id]),
                "subject_key": _identity_key(
                    endpoints[request.origin_endpoint_id]),
            },
            "request_kind": request.request_kind,
        }
    else:
        assert request.max_states is not None
        assert request.max_routes is not None
        consumer_payload = {
            "budget": {
                "max_facts": request.max_facts,
                "max_routes": request.max_routes,
                "max_states": request.max_states,
            },
            "origin_key": _identity_key(
                endpoints[request.origin_endpoint_id]),
            "request_kind": request.request_kind,
            "target_object_kinds": list(request.target_object_kinds),
        }
    payload_value = {
        "candidate_definition": {
            "context_key": _identity_key(definition.context),
            "predicate_key": _identity_key(definition.predicate),
            "proposition_key": _identity_key(definition.proposition),
            "role_bindings": binding_payload,
            "source_anchor_key": _identity_key(definition.source_anchor),
        },
        "consumer_request": consumer_payload,
        "directionality": seed.directionality,
        "endpoints": endpoint_payload,
        "query_kind": "typed_relation_candidate",
        "relation_family": seed.relation_family,
        "relation_schema": {
            "constraints": constraint_payload,
            "relation_key": _identity_key(relation_schema.relation),
            "schema_key": _identity_key(relation_schema.schema),
            "slots": slot_payload,
        },
        "surface": seed.surface,
    }
    if not isinstance(rational_role_values, tuple):
        raise TypeError("rational_role_values 必须是 tuple")
    rational_payload = []
    seen_rational_roles = set()
    binding_by_role = {
        item.role_kind: item for item in seed.bindings
    }
    for item in rational_role_values:
        if (not isinstance(item, tuple) or len(item) != 3
                or any(type(value) is not int for value in item)):
            raise ValueError("rational role value 必须是三个严格整数")
        role_kind, numerator, denominator = item
        if role_kind in seen_rational_roles or role_kind not in binding_by_role:
            raise ValueError("rational role 未绑定或重复")
        if denominator <= 0:
            raise ValueError("rational role denominator 必须为正整数")
        seen_rational_roles.add(role_kind)
        binding = binding_by_role[role_kind]
        rational_payload.append({
            "den": denominator,
            "filler_key": _identity_key(endpoints[binding.endpoint_id]),
            "num": numerator,
            "role_key": _identity_key(roles[role_kind]),
        })
    if rational_payload:
        payload_value["rational_role_values"] = sorted(
            rational_payload,
            key=lambda item: item["role_key"],
        )
    typed_payload = CanonicalJsonObject.from_value(payload_value)
    payload_value = typed_payload.to_value()
    return AuthoredCompiledSeed(
        seed.seed_id,
        seed.family,
        seed.template_family,
        seed.label_owner,
        seed.split,
        seed.sample_role,
        "TypedRelationQuery",
        typed_payload,
        seed.expected_state,
        seed.expected_payload,
        seed.perturbation_kind,
        seed.supersedes_seed_id,
        seed.logical_order,
        (seed.surface, payload_value),
        (seed.surface, payload_value["candidate_definition"]),
        (
            "typed_relation_query_v1",
            seed.relation_kind,
            seed.schema_kind,
            seed.directionality,
            len(seed.endpoints),
            seed.perturbation_kind,
        ),
    )


__all__ = [
    "authored_relation_identity",
    "authored_relation_role_identity",
    "authored_relation_rule_identity",
    "authored_relation_schema_identity",
    "compile_relation_seed",
]
