"""把 D-02D 量词 seed 编译为 Binder、有限域和 bound operator payload。"""
from __future__ import annotations

import hashlib

from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
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
from pure_integer_ai.cognition.shared.logic_candidate import (
    LogicOperatorCandidateProtocol,
    LogicOperatorCandidateSpec,
)
from pure_integer_ai.cognition.shared.logic_executor import (
    ExistentialOperator,
    FiniteQuantifierDomain,
    LogicOperatorDefinition,
    OperatorSlot,
    QuantifierDefinition,
    UniversalOperator,
)
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.cognition.shared.semantic_object import (
    binder_identity,
    context_scope_identity,
    entity_identity,
    proposition_identity,
    set_expr_identity,
    variable_identity,
)
from pure_integer_ai.cognition.shared.typed_binding import (
    BoundProposition,
    BoundRoleBinding,
    TypedValue,
)
from pure_integer_ai.experiments.ph2_authored_course_common import (
    AuthoredCompiledSeed,
)
from pure_integer_ai.experiments.ph2_authored_logic_compile import (
    authored_logic_instruction_identity,
    authored_logic_operator_identity,
    authored_logic_role_identity,
    authored_logic_structure_identity,
)
from pure_integer_ai.experiments.ph2_authored_logic_schema import (
    OPERATOR_EXISTS,
    OPERATOR_FORALL,
)
from pure_integer_ai.experiments.ph2_authored_quantifier_schema import (
    AuthoredQuantifierSeed,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    canonical_json_bytes,
)


_COURSE_SOURCE_KIND = 208
_COURSE_NAMESPACE = 20801
_VERSIONS = VersionBundle(
    CorpusVersion(1),
    ParserVersion(1),
    PrimitiveVersion(1),
    CurriculumVersion(1),
)


def _stable_positive_int(namespace: str, value: str) -> int:
    """把 quantifier family/seed id 压为稳定正整数身份段。"""
    payload = canonical_json_bytes({"namespace": namespace, "value": value})
    result = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    result &= (1 << 63) - 1
    return result if result > 0 else 1


def _identity_key(identity) -> list[int]:
    """把一等对象、SourceRef 或 scope 投影为规范严格整数列表。"""
    return list(identity.stable_key())


def authored_quantifier_value_type(kind: int) -> ObjectIdentity:
    """按冻结课程坐标恢复 domain value 的一等类型 Concept。"""
    if type(kind) is not int or kind <= 0:
        raise ValueError("quantifier value type kind 必须是正严格整数")
    return concept_identity(
        (_COURSE_NAMESPACE, 4, kind), versions=_VERSIONS)


def _candidate_protocol() -> LogicOperatorCandidateProtocol:
    """返回量词候选 structure/instruction/slot 的冻结图协议。"""
    return LogicOperatorCandidateProtocol(
        concept_identity((_COURSE_NAMESPACE, 5, 1), versions=_VERSIONS),
        concept_identity((_COURSE_NAMESPACE, 5, 2), versions=_VERSIONS),
        concept_identity((_COURSE_NAMESPACE, 5, 3), versions=_VERSIONS),
    )


def _handler(operator_kind: int):
    """按显式量词坐标返回 EXISTS/FORALL handler。"""
    if operator_kind == OPERATOR_EXISTS:
        return ExistentialOperator()
    if operator_kind == OPERATOR_FORALL:
        return UniversalOperator()
    raise ValueError("quantifier operator 尚无 compiler handler")


def _bound_payload(bound: BoundProposition) -> dict:
    """递归投影量词 BoundProposition 与 Variable filler。"""
    bindings = []
    for item in bound.bindings:
        if isinstance(item.filler, ObjectIdentity):
            filler = {
                "identity_key": _identity_key(item.filler),
                "kind": "identity",
            }
        else:
            filler = {
                "bound": _bound_payload(item.filler),
                "kind": "bound_proposition",
            }
        bindings.append({
            "filler": filler,
            "ordinal": item.ordinal,
            "role_key": _identity_key(item.role),
        })
    return {
        "applied_variable_keys": [
            _identity_key(item) for item in bound.applied_variables],
        "bindings": bindings,
        "context_key": _identity_key(bound.context),
        "instruction_key": _identity_key(bound.instruction),
        "introduced_binder_keys": [
            _identity_key(item) for item in bound.introduced_binders],
        "predicate_key": _identity_key(bound.predicate),
        "source_anchor_key": _identity_key(bound.source_anchor),
        "structure_key": _identity_key(bound.structure),
        "template_key": _identity_key(bound.template),
    }


def compile_quantifier_seed(
        seed: AuthoredQuantifierSeed) -> AuthoredCompiledSeed:
    """生成 quantifier definition、有限域、bound root 和执行请求 payload。"""
    if not isinstance(seed, AuthoredQuantifierSeed):
        raise TypeError("compile_quantifier_seed 需要 AuthoredQuantifierSeed")
    source = SourceRef(
        _COURSE_SOURCE_KIND,
        _stable_positive_int("quantifier-family", seed.family),
        _stable_positive_int("quantifier-seed", seed.seed_id),
        GLOBAL_OWNER_SCOPE,
        _VERSIONS,
    )
    structure = authored_logic_structure_identity(seed.structure_kind)
    instruction = authored_logic_instruction_identity(seed.instruction_kind)
    predicate = authored_logic_operator_identity(seed.operator_kind)
    body_role = authored_logic_role_identity(seed.body_role_kind)
    value_role = authored_logic_role_identity(seed.value_role_kind)
    context = context_scope_identity(source, (1, seed.context_local_id))
    value_type = authored_quantifier_value_type(seed.value_type_kind)
    binder = binder_identity(source, (1, seed.binder_local_id))
    variable = variable_identity(
        binder, (1, seed.variable_local_id), value_type)
    body = BoundProposition(
        proposition_identity(source, (1, seed.body.local_id)),
        minimal_instruction_identity(
            (_COURSE_NAMESPACE, 10, 1), versions=_VERSIONS),
        concept_identity(
            (_COURSE_NAMESPACE, 11, seed.body.local_id), versions=_VERSIONS),
        structure_concept_identity(
            (_COURSE_NAMESPACE, 12, seed.body.local_id), versions=_VERSIONS),
        occurrence_identity(
            source,
            start=seed.body.start,
            end=seed.body.end,
            ordinal=seed.body.ordinal,
        ),
        context,
        (),
        (BoundRoleBinding(value_role, variable, 0),),
        (),
    )
    root = BoundProposition(
        proposition_identity(source, (2, 1)),
        instruction,
        predicate,
        structure,
        occurrence_identity(
            source,
            start=seed.anchor.start,
            end=seed.anchor.end,
            ordinal=seed.anchor.ordinal,
        ),
        context,
        (binder,),
        (BoundRoleBinding(body_role, body, 0),),
        (),
    )
    definition = LogicOperatorDefinition(
        structure,
        instruction,
        (OperatorSlot(body_role, 0),),
        _handler(seed.operator_kind),
    )
    typed_values = tuple(TypedValue(
        entity_identity(source, (3, item.local_id)),
        authored_quantifier_value_type(item.actual_type_kind),
    ) for item in seed.domain.values)
    closure_evidence = tuple(proposition_identity(
        source, (4, item))
        for item in seed.domain.closure_evidence_local_ids)
    domain = FiniteQuantifierDomain(
        set_expr_identity(source, (1, seed.domain.domain_local_id)),
        typed_values,
        bool(seed.domain.closed),
        closure_evidence,
    )
    quantifier = QuantifierDefinition(
        binder,
        variable,
        OperatorSlot(body_role, 0),
        domain,
    )
    protocol = _candidate_protocol()
    spec = LogicOperatorCandidateSpec(
        root.template,
        definition,
        (_COURSE_NAMESPACE, 20, seed.operator_kind, seed.structure_kind),
        (source,),
    )
    candidate_definition = spec.candidate_definition(protocol)
    request = seed.consumer_request
    forall_semantics = ({
        "closed_empty_domain_vacuous_support": 1,
        "explicit_counterexample_requires_closed_domain": 0,
        "open_domain_current_all_support_is_true": 0,
    } if seed.operator_kind == OPERATOR_FORALL else {})
    typed_payload = CanonicalJsonObject.from_value({
        "bound_root": _bound_payload(root),
        "candidate_protocol": {
            "instruction_predicate_key": _identity_key(
                protocol.instruction_predicate),
            "slot_predicate_key": _identity_key(protocol.slot_predicate),
            "structure_predicate_key": _identity_key(
                protocol.structure_predicate),
        },
        "candidate_spec": {
            "candidate_key": _identity_key(spec.candidate),
            "competition_key": list(spec.competition_key),
            "forming_source_keys": [
                _identity_key(item) for item in spec.forming_sources],
            "graph_binding_count": len(candidate_definition.bindings),
        },
        "closed_domain_requires_evidence": 1,
        "consumer_request": {
            "budget": {
                "max_branches": request.max_branches,
                "max_depth": request.max_depth,
                "max_domain_values": request.max_domain_values,
                "max_steps": request.max_steps,
            },
            "request_kind": request.request_kind,
            "root_key": _identity_key(root.template),
            "scope_key": _identity_key(document_scope(source)),
        },
        "current_absence_is_counterexample": 0,
        "open_domain_exhaustion_is_false": 0,
        **forall_semantics,
        "operator_definition": {
            "instruction_key": _identity_key(definition.instruction),
            "slots": [{
                "ordinal": item.ordinal,
                "role_key": _identity_key(item.role),
            } for item in definition.slots],
            "structure_key": _identity_key(definition.structure),
        },
        "operator_family": seed.operator_family,
        "operator_kind": seed.operator_kind,
        "quantifier_definition": {
            "binder_key": _identity_key(quantifier.binder),
            "body_slot": {
                "ordinal": quantifier.body_slot.ordinal,
                "role_key": _identity_key(quantifier.body_slot.role),
            },
            "domain": {
                "closed": int(quantifier.domain.closed),
                "closure_evidence_keys": [
                    _identity_key(item)
                    for item in quantifier.domain.closure_evidence
                ],
                "domain_key": _identity_key(quantifier.domain.domain),
                "values": [{
                    "type_key": _identity_key(item.value_type),
                    "value_key": _identity_key(item.value),
                } for item in quantifier.domain.values],
            },
            "value_type_key": _identity_key(value_type),
            "variable_key": _identity_key(quantifier.variable),
        },
        "query_kind": "typed_quantifier_candidate",
        "surface": seed.surface,
        "surface_cue_authoritative": 0,
        "value_evidence": [{
            "refute": seed_value.evidence_refute,
            "support": seed_value.evidence_support,
            "value_key": _identity_key(item.value),
        } for item, seed_value in zip(typed_values, seed.domain.values)],
    })
    payload_value = typed_payload.to_value()
    return AuthoredCompiledSeed(
        seed.seed_id,
        seed.family,
        seed.template_family,
        seed.label_owner,
        seed.split,
        seed.sample_role,
        "QuantifierExecutionQuery",
        typed_payload,
        seed.expected_state,
        seed.expected_payload,
        seed.perturbation_kind,
        seed.supersedes_seed_id,
        seed.logical_order,
        (seed.surface, payload_value),
        (
            seed.surface,
            payload_value["bound_root"],
            payload_value["quantifier_definition"],
            payload_value["consumer_request"],
        ),
        (
            "typed_quantifier_v1",
            seed.operator_kind,
            seed.structure_kind,
            seed.instruction_kind,
            len(seed.domain.values),
            seed.domain.closed,
            seed.perturbation_kind,
        ),
    )


__all__ = [
    "authored_quantifier_value_type",
    "compile_quantifier_seed",
]
