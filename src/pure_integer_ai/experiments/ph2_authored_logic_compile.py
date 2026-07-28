"""把已核准 D-02D logic seed 编译为现役 operator 与 bound payload。"""
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
    ConjunctionOperator,
    ConditionOperator,
    DisjunctionOperator,
    LogicOperatorDefinition,
    ModalOperator,
    NegationOperator,
    OperatorSlot,
)
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.cognition.shared.semantic_object import (
    context_scope_identity,
    proposition_identity,
    role_identity,
)
from pure_integer_ai.cognition.shared.typed_binding import (
    BoundProposition,
    BoundRoleBinding,
)
from pure_integer_ai.experiments.ph2_authored_course_common import (
    AuthoredCompiledSeed,
)
from pure_integer_ai.experiments.ph2_authored_logic_schema import (
    OPERATOR_AND,
    OPERATOR_CONDITION,
    OPERATOR_MODAL,
    OPERATOR_NOT,
    OPERATOR_OR,
    AuthoredLogicSeed,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    canonical_json_bytes,
)


_COURSE_SOURCE_KIND = 207
_COURSE_NAMESPACE = 20701
_VERSIONS = VersionBundle(
    CorpusVersion(1),
    ParserVersion(1),
    PrimitiveVersion(1),
    CurriculumVersion(1),
)


def _stable_positive_int(namespace: str, value: str) -> int:
    """把 logic family/seed id 压为稳定正整数身份段。"""
    payload = canonical_json_bytes({"namespace": namespace, "value": value})
    result = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    result &= (1 << 63) - 1
    return result if result > 0 else 1


def _identity_key(identity) -> list[int]:
    """把一等对象、SourceRef 或 scope 投影为规范严格整数列表。"""
    return list(identity.stable_key())


def authored_logic_operator_identity(kind: int) -> ObjectIdentity:
    """按冻结 D-02D 坐标恢复一等 operator Concept。"""
    if type(kind) is not int or kind <= 0:
        raise ValueError("authored logic operator kind 必须是正严格整数")
    return concept_identity(
        (_COURSE_NAMESPACE, 1, kind), versions=_VERSIONS)


def authored_logic_structure_identity(kind: int) -> ObjectIdentity:
    """按冻结 D-02D 坐标恢复一等 StructureConcept。"""
    if type(kind) is not int or kind <= 0:
        raise ValueError("authored logic structure kind 必须是正严格整数")
    return structure_concept_identity(
        (_COURSE_NAMESPACE, 2, kind), versions=_VERSIONS)


def authored_logic_instruction_identity(kind: int) -> ObjectIdentity:
    """按冻结 D-02D 坐标恢复一等 MinimalInstruction。"""
    if type(kind) is not int or kind <= 0:
        raise ValueError("authored logic instruction kind 必须是正严格整数")
    return minimal_instruction_identity(
        (_COURSE_NAMESPACE, 3, kind), versions=_VERSIONS)


def authored_logic_role_identity(kind: int) -> ObjectIdentity:
    """按冻结 D-02D 坐标恢复一等逻辑 Role。"""
    if type(kind) is not int or kind <= 0:
        raise ValueError("authored logic Role kind 必须是正严格整数")
    return role_identity(
        (_COURSE_NAMESPACE, 4, kind), versions=_VERSIONS)


def _candidate_protocol() -> LogicOperatorCandidateProtocol:
    """返回逻辑候选 structure/instruction/slot 的冻结图协议。"""
    return LogicOperatorCandidateProtocol(
        concept_identity((_COURSE_NAMESPACE, 5, 1), versions=_VERSIONS),
        concept_identity((_COURSE_NAMESPACE, 5, 2), versions=_VERSIONS),
        concept_identity((_COURSE_NAMESPACE, 5, 3), versions=_VERSIONS),
    )


def _handler(operator_kind: int):
    """按显式 operator 坐标返回当前已注册的纯执行 handler。"""
    if operator_kind == OPERATOR_NOT:
        return NegationOperator()
    if operator_kind == OPERATOR_AND:
        return ConjunctionOperator()
    if operator_kind == OPERATOR_OR:
        return DisjunctionOperator()
    if operator_kind == OPERATOR_CONDITION:
        return ConditionOperator()
    if operator_kind == OPERATOR_MODAL:
        return ModalOperator()
    raise ValueError("authored logic operator 尚无 compiler handler")


def _bound_payload(bound: BoundProposition) -> dict:
    """递归投影不可物化 BoundProposition，保留 slot 和嵌套边界。"""
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


def compile_logic_seed(seed: AuthoredLogicSeed) -> AuthoredCompiledSeed:
    """生成 operator definition、候选、嵌套 bound root 和执行请求 payload。"""
    if not isinstance(seed, AuthoredLogicSeed):
        raise TypeError("compile_logic_seed 需要 AuthoredLogicSeed")
    source = SourceRef(
        _COURSE_SOURCE_KIND,
        _stable_positive_int("logic-family", seed.family),
        _stable_positive_int("logic-seed", seed.seed_id),
        GLOBAL_OWNER_SCOPE,
        _VERSIONS,
    )
    structure = authored_logic_structure_identity(seed.structure_kind)
    instruction = authored_logic_instruction_identity(seed.instruction_kind)
    predicate = authored_logic_operator_identity(seed.operator_kind)
    roles = {
        item.role_kind: authored_logic_role_identity(item.role_kind)
        for item in seed.bindings
    }
    definition = LogicOperatorDefinition(
        structure,
        instruction,
        tuple(OperatorSlot(
            roles[item.role_kind], item.ordinal) for item in seed.bindings),
        _handler(seed.operator_kind),
    )
    context = context_scope_identity(source, (1, seed.context_local_id))
    operand_by_id = {item.operand_id: item for item in seed.operands}
    bound_operands = {}
    for operand in seed.operands:
        bound_operands[operand.operand_id] = BoundProposition(
            proposition_identity(source, (1, operand.local_id)),
            minimal_instruction_identity(
                (_COURSE_NAMESPACE, 10, 1), versions=_VERSIONS),
            concept_identity(
                (_COURSE_NAMESPACE, 11, operand.local_id),
                versions=_VERSIONS,
            ),
            structure_concept_identity(
                (_COURSE_NAMESPACE, 12, operand.local_id),
                versions=_VERSIONS,
            ),
            occurrence_identity(
                source,
                start=operand.start,
                end=operand.end,
                ordinal=operand.ordinal,
            ),
            context,
            (),
            (),
            (),
        )
    current = None
    for depth in range(1, seed.nesting_depth + 1):
        if depth == 1:
            bindings = tuple(BoundRoleBinding(
                roles[item.role_kind],
                bound_operands[item.operand_id],
                item.ordinal,
            ) for item in seed.bindings)
        else:
            if current is None or len(seed.bindings) != 1:
                raise ValueError("嵌套 logic compiler 当前只允许一元 operator")
            item = seed.bindings[0]
            bindings = (BoundRoleBinding(
                roles[item.role_kind], current, item.ordinal),)
        current = BoundProposition(
            proposition_identity(source, (2, depth)),
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
            (),
            bindings,
            (),
        )
    assert current is not None
    protocol = _candidate_protocol()
    spec = LogicOperatorCandidateSpec(
        current.template,
        definition,
        (_COURSE_NAMESPACE, 20, seed.operator_kind, seed.structure_kind),
        (source,),
    )
    candidate_definition = spec.candidate_definition(protocol)
    request = seed.consumer_request
    typed_payload = CanonicalJsonObject.from_value({
        "bound_root": _bound_payload(current),
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
        "closed_world_assumed": 0,
        "consumer_request": {
            "budget": {
                "max_branches": request.max_branches,
                "max_depth": request.max_depth,
                "max_steps": request.max_steps,
            },
            "request_kind": request.request_kind,
            "root_key": _identity_key(current.template),
            "scope_key": _identity_key(document_scope(source)),
        },
        "nesting_depth": seed.nesting_depth,
        "operand_evidence": [{
            "refute": operand_by_id[item.operand_id].evidence_refute,
            "support": operand_by_id[item.operand_id].evidence_support,
            "template_key": _identity_key(
                bound_operands[item.operand_id].template),
        } for item in seed.bindings],
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
        "query_kind": "typed_logic_operator_candidate",
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
        "LogicExecutionQuery",
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
            payload_value["consumer_request"],
        ),
        (
            "typed_logic_operator_v1",
            seed.operator_kind,
            seed.structure_kind,
            seed.instruction_kind,
            seed.nesting_depth,
            len(seed.operands),
            seed.perturbation_kind,
        ),
    )


__all__ = [
    "authored_logic_instruction_identity",
    "authored_logic_operator_identity",
    "authored_logic_role_identity",
    "authored_logic_structure_identity",
    "compile_logic_seed",
]
