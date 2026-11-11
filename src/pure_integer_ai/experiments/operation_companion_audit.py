"""独立冷核 operation 的真实 Companion 来源阶段，不调用查询或生成器。"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.identity import ObjectIdentity
from pure_integer_ai.cognition.shared.integer_operation_execution import READY
from pure_integer_ai.cognition.shared.query_state import (
    ROOT_EXPANDED, ROOT_PENDING, BindingEntry, EvidenceEntry, QueryRoot, VisitedKey,
)
from pure_integer_ai.cognition.understanding.query_open_roles import unpack_record
from pure_integer_ai.experiments.graph_query_trace_contract import GraphQueryTraceEvidence
from pure_integer_ai.experiments.operation_companion_query import (
    COMPANION_MEMBERS, COMPANION_PHASE, COMPANION_PREMISE_ROLE, COMPANION_PROOF_ROLE,
)
from pure_integer_ai.experiments.query_memory_ohe_audit import _unpack
from pure_integer_ai.experiments.trained_query_evidence import (
    core_query_evidence, current_query_evidence, dialogue_query_evidence,
)


def _evidence(key):
    """解码公开 QueryState Evidence 记录，拒绝缺尾或多尾。"""
    fields = _unpack(key, 3, 6)
    return EvidenceEntry(fields[0], fields[1], key[0], key[1], key[2], *fields[2:])


def _counter(row):
    """冷审只接受原始 trace 明确保存的四项非负计数。"""
    value = row.get("counters")
    if (not isinstance(value, (tuple, list)) or len(value) != 4
            or any(type(item) is not int or item < 0 for item in value)):
        raise ValueError("Companion audit requires original query step counters")
    return tuple(value)


def verify_saved_operation_companion(core, execution, value):
    """从物理帧/H-00 独立重建全部前提，并核原 trace、预算、hop 与 Plan 引用。"""
    trace = GraphQueryTraceEvidence(value)
    if trace.version != 4 or execution.status != READY:
        raise ValueError("Companion audit requires a READY execution and the complete v4 trace")
    state = value["state_key"]
    query = trace.resolve_ref(state["query_ref"])
    if query != execution.query_key or state.get("active_spaces") != [1, 2, 3]:
        raise ValueError("Companion audit found a different query or inactive shared space")

    def keys(row, field):
        """只解引用原始 artifact 成员，不补造阶段状态。"""
        refs = row.get(field)
        if not isinstance(refs, (tuple, list)):
            raise ValueError("Companion trace lost a required state component")
        return tuple(trace.resolve_ref(ref) for ref in refs)

    evidence = tuple(_evidence(key) for key in keys(state, "evidence"))
    phases = tuple(item for item in evidence if item.space == 3
                   and item.evidence_key == (COMPANION_PHASE, *query))
    if len(phases) != 1:
        raise ValueError("Companion requires one exact phase Evidence")
    phase = phases[0]
    permission = execution.witness.source.stable_key()
    if (phase.source_ref != permission or phase.scope_key != query or phase.polarity != 3
            or phase.trust != 1 or phase.supersedes_key
            or phase.hypothesis_key != execution.witness.program.operation_key):
        raise ValueError("Companion phase source, scope, or operation drifted")
    fields = unpack_record(phase.payload_key, COMPANION_PHASE)
    if len(fields) != 4 or fields[0] != execution.stable_key():
        raise ValueError("Companion phase lacks the full current execution")
    before, after = fields[1:3]
    members = unpack_record(fields[3], COMPANION_MEMBERS)
    premises = tuple(sorted({key for match in execution.matches for key in match}))
    if not premises or len(premises) > 64 or len(before) != 4 or len(after) != 4:
        raise ValueError("Companion phase lacks bounded premises or counters")
    frames = {frame.proposition.stable_key(): frame for frame in core.active_surface_frames()}
    facts = {fact.proposition.stable_key(): fact for fact in core.active_surface_facts()}
    bindings, roots, visited = (set(keys(state, name)) for name in ("bindings", "roots", "visited"))
    proof = BindingEntry(COMPANION_PROOF_ROLE, execution.stable_key(), space=3, scope_key=query).stable_key()
    if proof not in bindings:
        raise ValueError("Companion phase lacks its complete execution binding")
    expected_members, costs, premise_roots = [], [], []
    steps = value["frontier_trace"]["steps"]
    phase_positions = [index for index, row in enumerate(steps)
                       if phase.stable_key() in keys(row, "evidence_refs")]
    if not phase_positions:
        raise ValueError("Companion phase was never recorded in the original query trace")
    end = phase_positions[0]
    start = end - len(premises) - 2
    if start < 0 or _counter(steps[start]) != before or _counter(steps[start + 1]) != before:
        raise ValueError("Companion phase has no exact original entry state")
    current = before
    for index, key in enumerate(premises):
        frame, fact = frames.get(key), facts.get(key)
        if frame is None or fact is None:
            raise ValueError("Companion premise has no physical source frame or active fact")
        source = core.generation_input(ObjectIdentity.from_stable_key(key))
        definition = source.proposition.definition
        history = core.evidence_history(definition.proposition)
        if (frame.proposition != definition.proposition or frame.predicate != definition.predicate
                or fact.predicate != definition.predicate or frame.source_hash != fact.source_hash
                or len(fact.bindings) != len(definition.bindings)
                or {(binding.role, binding.filler) for binding in fact.bindings}
                    != {(binding.role, binding.filler) for binding in definition.bindings}
                or len(frame.roles) != len(definition.bindings)
                or set(frame.roles) != {binding.role for binding in definition.bindings}
                or len(frame.gaps) != len(frame.roles) + 1 or not history):
            raise ValueError("Companion physical premise roles, source, or Evidence are incomplete")
        expected_core = current_query_evidence(core_query_evidence(definition.proposition, history,
            core.context.scoped_identity_store.registry))
        actual_core = current_query_evidence(tuple(item for item in evidence if item.space == 1
            and item.hypothesis_key == key and (item.evidence_key[:1] == (1,) or item.polarity != 3)))
        if set(expected_core) != set(actual_core):
            raise ValueError("Companion premise differs from physical current Core/H-00 Evidence")
        source_evidence = dialogue_query_evidence(definition.proposition, frame, source, None)
        binding = BindingEntry(COMPANION_PREMISE_ROLE, key, space=3, scope_key=query).stable_key()
        candidates = []
        for root_key in roots:
            parts = _unpack(root_key, 3, 3)
            if (root_key[0] == 3 and parts == (key, permission, query)
                    and root_key[1] == ROOT_EXPANDED):
                candidates.append(QueryRoot(3, key, ROOT_EXPANDED, root_key[2], permission, query))
        if (source_evidence not in evidence or binding not in bindings
                or VisitedKey(3, key).stable_key() not in visited or len(candidates) != 1):
            raise ValueError("Companion premise lacks its full source, binding, expanded root, or visit")
        root = candidates[0]
        if QueryRoot(3, key, ROOT_PENDING, root.priority_weight, permission, query).stable_key() not in keys(
                steps[start + 1], "root_refs"):
            raise ValueError("Companion source root was never pending in this query")
        cost = 2 + len(history)
        current = before[0] + 1, current[1] + 1, current[2] + 2, current[3] + cost
        row = steps[start + index + 2]
        if (_counter(row) != current or source_evidence.stable_key() not in keys(row, "evidence_refs")
                or binding not in keys(row, "binding_refs")
                or root.stable_key() not in keys(row, "root_refs")):
            raise ValueError("Companion per-premise read does not match the original state transition")
        matching_hops = [hop for hop in value["hops"] if hop["space"] == 3
            and trace.resolve_ref(hop["proposition_ref"]) == key
            and trace.resolve_ref(hop["predicate_ref"]) == frame.predicate.stable_key()
            and hop["source_hash"] == frame.source_hash and hop["depth"] == before[0] + 1
            and trace.resolve_ref(hop["from_filler_ref"]) == (3, *key)
            and trace.resolve_ref(hop["to_filler_ref"]) == (1, *key)
            and trace.resolve_ref(hop["surface_values_ref"]) == ()]
        if len(matching_hops) != 1:
            raise ValueError("Companion premise lacks one actual current source hop")
        expected_members.append(source_evidence.stable_key())
        costs.append(cost)
        premise_roots.append(root.stable_key())
    if (members != tuple(expected_members) or after != current or _counter(steps[end]) != after
            or proof not in keys(steps[end], "binding_refs")):
        raise ValueError("Companion phase has incomplete members or incorrect final counters")
    scalars, budget = state["scalars"], state["budget"]
    if (len(scalars) != 17 or len(budget) != 4
            or any(end_value < used for end_value, used in zip(
                (scalars[0], scalars[2], scalars[3], scalars[4]), after, strict=True))
            or any(used > limit for used, limit in zip(after,
                (budget[3], budget[0], budget[1], budget[2]), strict=True))):
        raise ValueError("Companion stage exceeded or escaped the original query budget")
    plan = value.get("response_plan")
    plan_evidence = set() if not isinstance(plan, dict) else {
        tuple(item) for item in plan.get("evidence_refs", ())}
    if not {phase.stable_key(), *expected_members} <= plan_evidence:
        raise ValueError("Companion full phase and premise Evidence did not reach ResponsePlan")
    return {"status": "MATCHED_PHYSICAL_COMPANION_PREMISES_QUERYSTATE_AND_PLAN",
            "premise_count": len(premises), "source_read_count": sum(costs),
            "counters_before": list(before), "counters_after": list(after),
            "original_trace_step_range": [start, end],
            "query_reexecution_count": 0, "generation_reexecution_count": 0}
