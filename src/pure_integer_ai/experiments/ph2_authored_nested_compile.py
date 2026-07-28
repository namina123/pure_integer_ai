"""把 D-02D nested chain 编译为异构 bound tree、layer candidate 和 scope trace。"""
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
    LogicEvidenceState,
    LogicOperatorDefinition,
    ModalOperator,
    ModalResolution,
    NegationOperator,
    OperatorSlot,
    QuantifierDefinition,
    UniversalOperator,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    SCOPE_DOCUMENT,
    SCOPE_GENERATION,
    SCOPE_QUERY,
    ScopeIdentity,
    document_scope,
    generation_scope,
    query_scope,
)
from pure_integer_ai.cognition.shared.semantic_object import (
    binder_identity,
    context_scope_identity,
    entity_identity,
    proposition_identity,
    role_identity,
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
from pure_integer_ai.experiments.ph2_authored_modal_schema import (
    RESOLVER_RESOLVED,
)
from pure_integer_ai.experiments.ph2_authored_nested_schema import (
    AuthoredNestedSeed,
)
from pure_integer_ai.experiments.ph2_authored_quantifier_compile import (
    authored_quantifier_value_type,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    canonical_json_bytes,
)


_COURSE_SOURCE_KIND = 209
_COURSE_NAMESPACE = 20901
_VERSIONS = VersionBundle(
    CorpusVersion(1),
    ParserVersion(1),
    PrimitiveVersion(1),
    CurriculumVersion(1),
)


def _stable_positive_int(namespace: str, value: str) -> int:
    """把 nested family/seed id 压为稳定正整数身份段。"""
    payload = canonical_json_bytes({"namespace": namespace, "value": value})
    result = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    result &= (1 << 63) - 1
    return result if result > 0 else 1


def _identity_key(identity) -> list[int]:
    """把一等对象、SourceRef 或 scope 投影为规范严格整数列表。"""
    return list(identity.stable_key())


def _candidate_protocol() -> LogicOperatorCandidateProtocol:
    """返回 nested layer candidate 的冻结图协议。"""
    return LogicOperatorCandidateProtocol(
        concept_identity((_COURSE_NAMESPACE, 5, 1), versions=_VERSIONS),
        concept_identity((_COURSE_NAMESPACE, 5, 2), versions=_VERSIONS),
        concept_identity((_COURSE_NAMESPACE, 5, 3), versions=_VERSIONS),
    )


def _handler(operator_family: str):
    """按显式 nested operator family 返回现役 handler。"""
    if operator_family == "NOT":
        return NegationOperator()
    if operator_family == "MODAL":
        return ModalOperator()
    if operator_family == "EXISTS":
        return ExistentialOperator()
    if operator_family == "FORALL":
        return UniversalOperator()
    raise ValueError("nested operator 尚无 compiler handler")


def _bound_payload(bound: BoundProposition) -> dict:
    """递归投影 nested BoundProposition 与 Variable filler。"""
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


def _modal_scope(resolver, input_scope: ScopeIdentity) -> ScopeIdentity:
    """按 nested modal resolver 构造同 source 输出 scope。"""
    if resolver.scope_kind == SCOPE_DOCUMENT:
        return input_scope
    if resolver.scope_kind == SCOPE_QUERY:
        return query_scope(resolver.scope_local_id, parent=input_scope)
    if resolver.scope_kind == SCOPE_GENERATION:
        return generation_scope(resolver.scope_local_id, parent=input_scope)
    raise ValueError("nested modal scope kind 尚无 compiler handler")


def _modal_plan(layer, source: SourceRef, input_scope: ScopeIdentity) -> dict:
    """构造一层 modal 的受限 resolver payload。"""
    resolver = layer.modal_resolver
    assert resolver is not None
    output_scope = None
    if resolver.status == RESOLVER_RESOLVED:
        output_scope = _modal_scope(resolver, input_scope)
        resolution = ModalResolution(
            LogicEvidenceState(
                bool(resolver.resolution_support),
                bool(resolver.resolution_refute),
            ),
            source,
            output_scope,
            resolver.evidence_ids,
        )
        state = {
            "refute": int(resolution.state.refute),
            "support": int(resolution.state.support),
        }
        evidence_ids = list(resolution.evidence_ids)
    else:
        state = {"refute": 0, "support": 0}
        evidence_ids = []
    return {
        "evidence_ids": evidence_ids,
        "input_scope_key": _identity_key(input_scope),
        "output_scope_key": (
            None if output_scope is None else _identity_key(output_scope)),
        "resolution_state": state,
        "source_key": _identity_key(source),
        "source_unchanged": 1,
        "status": resolver.status,
    }


def compile_nested_seed(seed: AuthoredNestedSeed) -> AuthoredCompiledSeed:
    """生成 nested bound tree、多 layer candidate、Binder/domain 和 scope trace。"""
    if not isinstance(seed, AuthoredNestedSeed):
        raise TypeError("compile_nested_seed 需要 AuthoredNestedSeed")
    source = SourceRef(
        _COURSE_SOURCE_KIND,
        _stable_positive_int("nested-family", seed.family),
        _stable_positive_int("nested-seed", seed.seed_id),
        GLOBAL_OWNER_SCOPE,
        _VERSIONS,
    )
    context = context_scope_identity(source, (1, seed.context_local_id))
    input_scope = document_scope(source)
    quantifier = seed.quantifier
    binder = None
    variable = None
    value_type = None
    domain = None
    quantifier_definition = None
    if quantifier is not None:
        value_type = authored_quantifier_value_type(
            quantifier.value_type_kind)
        binder = binder_identity(source, (1, quantifier.binder_local_id))
        variable = variable_identity(
            binder, (1, quantifier.variable_local_id), value_type)
        typed_values = tuple(TypedValue(
            entity_identity(source, (3, item.local_id)),
            authored_quantifier_value_type(item.actual_type_kind),
        ) for item in quantifier.domain.values)
        closure_evidence = tuple(proposition_identity(
            source, (4, item))
            for item in quantifier.domain.closure_evidence_local_ids)
        domain = FiniteQuantifierDomain(
            set_expr_identity(
                source, (1, quantifier.domain.domain_local_id)),
            typed_values,
            bool(quantifier.domain.closed),
            closure_evidence,
        )
    leaf_bindings = ()
    if quantifier is not None:
        assert variable is not None
        leaf_bindings = (BoundRoleBinding(
            authored_logic_role_identity(quantifier.value_role_kind),
            variable,
            0,
        ),)
    current = BoundProposition(
        proposition_identity(source, (1, seed.leaf.local_id)),
        minimal_instruction_identity(
            (_COURSE_NAMESPACE, 10, 1), versions=_VERSIONS),
        concept_identity(
            (_COURSE_NAMESPACE, 11, seed.leaf.local_id),
            versions=_VERSIONS,
        ),
        structure_concept_identity(
            (_COURSE_NAMESPACE, 12, seed.leaf.local_id),
            versions=_VERSIONS,
        ),
        occurrence_identity(
            source,
            start=seed.leaf.start,
            end=seed.leaf.end,
            ordinal=seed.leaf.ordinal,
        ),
        context,
        (),
        leaf_bindings,
        (),
    )
    protocol = _candidate_protocol()
    layer_rows = []
    trace_rows = []
    for reverse_index, layer in enumerate(reversed(seed.layers), start=1):
        role = authored_logic_role_identity(layer.role_kind)
        structure = authored_logic_structure_identity(layer.structure_kind)
        instruction = authored_logic_instruction_identity(
            layer.instruction_kind)
        predicate = authored_logic_operator_identity(layer.operator_kind)
        definition = LogicOperatorDefinition(
            structure,
            instruction,
            (OperatorSlot(role, 0),),
            _handler(layer.operator_family),
        )
        introduced = ()
        if quantifier is not None and layer.layer_id == quantifier.layer_id:
            assert binder is not None and domain is not None
            introduced = (binder,)
            quantifier_definition = QuantifierDefinition(
                binder,
                variable,
                OperatorSlot(role, 0),
                domain,
            )
        current = BoundProposition(
            proposition_identity(source, (2, reverse_index)),
            instruction,
            predicate,
            structure,
            occurrence_identity(
                source,
                start=layer.anchor.start,
                end=layer.anchor.end,
                ordinal=layer.anchor.ordinal,
            ),
            context,
            introduced,
            (BoundRoleBinding(role, current, 0),),
            (),
        )
        spec = LogicOperatorCandidateSpec(
            current.template,
            definition,
            (
                _COURSE_NAMESPACE,
                20,
                layer.operator_kind,
                layer.structure_kind,
                reverse_index,
            ),
            (source,),
        )
        candidate_definition = spec.candidate_definition(protocol)
        modal_plan = (
            _modal_plan(layer, source, input_scope)
            if layer.operator_family == "MODAL" else None)
        quantifier_payload = None
        if quantifier is not None and layer.layer_id == quantifier.layer_id:
            assert quantifier_definition is not None
            assert value_type is not None
            quantifier_payload = {
                "binder_key": _identity_key(quantifier_definition.binder),
                "body_slot": {
                    "ordinal": quantifier_definition.body_slot.ordinal,
                    "role_key": _identity_key(
                        quantifier_definition.body_slot.role),
                },
                "domain": {
                    "closed": int(quantifier_definition.domain.closed),
                    "closure_evidence_keys": [
                        _identity_key(item)
                        for item in quantifier_definition.domain.closure_evidence
                    ],
                    "domain_key": _identity_key(
                        quantifier_definition.domain.domain),
                    "values": [{
                        "type_key": _identity_key(item.value_type),
                        "value_key": _identity_key(item.value),
                    } for item in quantifier_definition.domain.values],
                },
                "value_evidence": [{
                    "refute": source_value.evidence_refute,
                    "support": source_value.evidence_support,
                    "value_key": _identity_key(item.value),
                } for item, source_value in zip(
                    quantifier_definition.domain.values,
                    quantifier.domain.values,
                )],
                "value_role_key": _identity_key(
                    authored_logic_role_identity(quantifier.value_role_kind)),
                "value_type_key": _identity_key(value_type),
                "variable_key": _identity_key(
                    quantifier_definition.variable),
            }
        row = {
            "bound_key": _identity_key(current.template),
            "candidate_available": layer.candidate_available,
            "candidate_spec": {
                "candidate_key": _identity_key(spec.candidate),
                "competition_key": list(spec.competition_key),
                "forming_source_keys": [
                    _identity_key(item) for item in spec.forming_sources],
                "graph_binding_count": len(candidate_definition.bindings),
            },
            "layer_id": layer.layer_id,
            "modal_resolution_plan": modal_plan,
            "operator_definition": {
                "instruction_key": _identity_key(definition.instruction),
                "slots": [{
                    "ordinal": item.ordinal,
                    "role_key": _identity_key(item.role),
                } for item in definition.slots],
                "structure_key": _identity_key(definition.structure),
            },
            "operator_family": layer.operator_family,
            "operator_kind": layer.operator_kind,
            "quantifier_definition": quantifier_payload,
        }
        layer_rows.append(row)
        if not layer.candidate_available:
            trace_scope = None
        elif modal_plan is not None:
            trace_scope = modal_plan["output_scope_key"]
        else:
            trace_scope = _identity_key(input_scope)
        trace_rows.append({
            "candidate_available": layer.candidate_available,
            "layer_id": layer.layer_id,
            "scope_key": trace_scope,
        })
    layer_rows.reverse()
    typed_payload = CanonicalJsonObject.from_value({
        "bound_root": _bound_payload(current),
        "candidate_protocol": {
            "instruction_predicate_key": _identity_key(
                protocol.instruction_predicate),
            "slot_predicate_key": _identity_key(protocol.slot_predicate),
            "structure_predicate_key": _identity_key(
                protocol.structure_predicate),
        },
        "closed_world_assumed": 0,
        "consumer_request": {
            "budget": {
                "max_branches": seed.consumer_request.max_branches,
                "max_depth": seed.consumer_request.max_depth,
                "max_domain_values": seed.consumer_request.max_domain_values,
                "max_resolver_calls": seed.consumer_request.max_resolver_calls,
                "max_steps": seed.consumer_request.max_steps,
            },
            "root_key": _identity_key(current.template),
            "scope_key": _identity_key(input_scope),
        },
        "derivation_order": [item.layer_id for item in reversed(seed.layers)],
        "derivation_trace": trace_rows,
        "layers": layer_rows,
        "leaf_evidence": {
            "refute": seed.leaf.evidence_refute,
            "support": seed.leaf.evidence_support,
            "template_key": _identity_key(
                proposition_identity(source, (1, seed.leaf.local_id))),
        },
        "query_kind": "typed_nested_scope_candidate",
        "same_source_required": 1,
        "scope_order_authoritative": 1,
        "surface": seed.surface,
        "surface_cue_authoritative": 0,
    })
    payload_value = typed_payload.to_value()
    return AuthoredCompiledSeed(
        seed.seed_id,
        seed.family,
        seed.template_family,
        seed.label_owner,
        seed.split,
        seed.sample_role,
        "NestedScopeExecutionQuery",
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
            payload_value["layers"],
            payload_value["consumer_request"],
        ),
        (
            "typed_nested_scope_v1",
            tuple(item.operator_kind for item in seed.layers),
            len(seed.layers),
            0 if seed.quantifier is None else 1,
            seed.perturbation_kind,
        ),
    )


__all__ = ["compile_nested_seed"]
