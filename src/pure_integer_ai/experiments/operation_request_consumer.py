"""图内请求变量、实际 Core 来源和同次 QueryState 的共享消费门。"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.identity import ObjectIdentity
from pure_integer_ai.cognition.shared.integer_operation_execution import READY
from pure_integer_ai.cognition.shared.query_driver import unvisited_frontier
from pure_integer_ai.cognition.shared.query_state import (
    ROOT_EXPANDED, ROOT_PENDING, TERMINATION_ANSWER_CLOSED, FrontierEntry, QueryAnchor, QueryRoot,
)
from pure_integer_ai.cognition.shared.memory_event import MEMORY_OBJECT_EVIDENCE
from pure_integer_ai.cognition.understanding.query_operation_requests import (
    BINDING, OperationRequestFrame, match_operation_request_frames,
)


def validate_operation_request_frames(frames, witnesses):
    """reader 必须精确对应当前许可程序的每个 external variable，模式 id 无授权作用。"""
    for frame in frames:
        selected = tuple(witness for witness in witnesses if witness.program.operation_key == frame.operation_key)
        if len(selected) != 1:
            raise ValueError("request frame has no unique current semantic program")
        witness, = selected
        if (frame.source_identity != witness.source_identity or frame.license_identity != witness.license_identity
                or {row[0] for row in frame.readers} != set(witness.program.external_variables)):
            raise ValueError("request frame permission or complete external-variable contract drifted")
        for variable, predicate, role, kinds in frame.readers:
            slots = tuple(slot for atom in witness.program.atoms if atom.schema.relation.stable_key() == predicate
                          for semantic_role, coordinate in atom.role_variables if coordinate == variable
                          and semantic_role.stable_key() == role for slot in atom.schema.slots if slot.role == semantic_role)
            if not slots or any(not set(kinds) <= set(slot.allowed_object_kinds) for slot in slots):
                raise ValueError("request reader differs from its actual program schema role/type")


def validate_operation_input_permissions(witness, input_groups):
    """只核验实际声明此operation的输入来源，无关课程不能授权或冒充其来源。"""
    operation = witness.program.operation_key
    for frames, metadata in input_groups:
        if metadata is not None and any(frame.operation_key == operation for frame in frames):
            if (witness.source_identity != tuple(metadata["source_identity"])
                    or witness.license_identity != tuple(metadata["license_identity"])):
                raise ValueError("semantic operation program source permission drifted")


def request_variables_from_readers(witness, structure, facts, core):
    """逐项验证完整跨度和 alias 物理来源，当前输入不能借同名或旧变量命中。"""
    if core is None or len(structure.operation_request_bindings) != 1:
        return None
    binding, = structure.operation_request_bindings
    frame = OperationRequestFrame.from_stable_key(binding.frame_key)
    validate_operation_request_frames((frame,), (witness,))
    if match_operation_request_frames((frame,), structure) != (binding,):
        return None
    by_key = {fact.proposition_key: fact for fact in facts}
    surfaces = {fact.proposition.stable_key(): fact for fact in core.active_surface_facts()}
    members = {member.stable_key(): member for member in structure.semantic_candidates}
    variables = []
    for (variable, predicate, role, kinds), (_variable, target, span, keys) in zip(frame.readers, binding.readings):
        if target[0] not in kinds:
            return None
        for key in keys:
            member = members.get(key)
            fact = None if member is None else by_key.get(member.proposition_key)
            surface = None if member is None else surfaces.get(member.proposition_key)
            if (fact is None or surface is None or member.projection_kind != 1 or member.span_ref != span
                    or member.candidate_key != target or member.predicate_key != predicate or member.role_key != role
                    or member.source_ref != (surface.source_hash,)
                    or not set(fact.candidate.forming_sources) <= set(witness.authorized_forming_sources)
                    or fact.definition.predicate.stable_key() != predicate
                    or not any(row.role.stable_key() == role and row.filler.stable_key() == target
                               for row in fact.definition.bindings)):
                return None
        variables.append((variable, ObjectIdentity.from_stable_key(target)))
    return tuple(sorted(variables))


def operation_request_query_seeds(structure, query_key, owner_weight):
    """完整 request 的当前对象成为 Core 查询根；根不携带事实真值。"""
    if not structure.operation_request_bindings:
        return ()
    binding, = structure.operation_request_bindings
    frame = OperationRequestFrame.from_stable_key(binding.frame_key)
    if match_operation_request_frames((frame,), structure) != (binding,):
        raise ValueError("request Core root differs from its complete current input")
    targets = tuple(sorted({row[1] for row in binding.readings}))
    return tuple((
        QueryRoot(1, target, ROOT_PENDING, owner_weight,
                  source_ref=structure.source_ref, scope_key=query_key),
        QueryAnchor(1, target, kind=BINDING, support=1, scope_key=query_key),
        FrontierEntry((1, BINDING, *target), 1, target_key=target, root_key=target,
                      scope_key=query_key, source_ref=structure.source_ref,
                      owner_weight=owner_weight, required_slot_gain=2, relation_fit=2),
    ) for target in targets)


def close_operation_request_selection(owner, structure, state, execution, *, logical_conflict):
    """仅完整程序 proof 解决实际选中 claims 的竞争；独立冲突和未消费 Memory 不放行。"""
    if (not structure.operation_request_bindings or execution is None or execution.status != READY
            or logical_conflict or unvisited_frontier(state.frontier, state.visited)
            or any(root.status == ROOT_PENDING for root in state.roots)):
        return state
    from pure_integer_ai.experiments.operation_output_query import execution_proof_consumed
    if (len(structure.operation_request_bindings) != 1 or state.query_key != structure.source_ref
            or execution.query_key != state.query_key or not execution_proof_consumed(execution, state)):
        return state
    request, = structure.operation_request_bindings
    if (request.source_ref != state.query_key or request.operation_key != execution.witness.program.operation_key
            or any(not any(root.owner_space == 1 and root.root_key == target
                           and root.status == ROOT_EXPANDED and root.source_ref == structure.source_ref
                           and root.scope_key == state.query_key for root in state.roots)
                   for target in {row[1] for row in request.readings})):
        return state
    expected = {item.evidence_key for item in owner._memory_evidence}
    consumed = {item.filler_key for item in state.bindings if item.space == 2 and item.role_key == (MEMORY_OBJECT_EVIDENCE,)}
    if not expected <= consumed:
        return state
    return state.with_(termination=TERMINATION_ANSWER_CLOSED, required_slots_open=0,
        evidence_closed=1, generation_ready=1, conflict_open=0,
        best_candidate_key=execution.outputs[0][0], best_score=max(1, state.best_score),
        second_candidate_key=(), second_score=0)
