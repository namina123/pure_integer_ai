"""当前 operation 的全部实际前提经过同一查询的 Companion 来源读取。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.identity import ObjectIdentity
from pure_integer_ai.cognition.shared.query_state import (
    ROOT_EXPANDED, ROOT_PENDING, TERMINATION_BUDGET_EXHAUSTED,
    TERMINATION_CLARIFY_MISSING_BINDING, TERMINATION_OPEN, BindingEntry, EvidenceEntry,
    FrontierEntry, QueryAnchor, QueryRoot, VisitedKey,
)
from pure_integer_ai.cognition.understanding.query_open_roles import pack_record, unpack_record
from pure_integer_ai.experiments.operation_output_query import execution_proof_consumed
from pure_integer_ai.experiments.relation_runtime_phase import observed_relation_phase
from pure_integer_ai.experiments.trained_relation_graph_runtime import TrainedRelationGraphError
from pure_integer_ai.experiments.trained_query_evidence import (
    core_query_evidence, current_query_evidence, dialogue_query_evidence,
)


COMPANION_PHASE = 92510
COMPANION_MEMBERS = 92511
COMPANION_PROOF_ROLE = (COMPANION_PHASE, 1)
COMPANION_PREMISE_ROLE = (COMPANION_PHASE, 2)


@dataclass(frozen=True, slots=True)
class OperationCompanionRead:
    """真实来源读取的整数 hop 载体；不携带回答表面。"""

    proposition: tuple[int, ...]
    predicate: tuple[int, ...]
    source_hash: int
    depth: int


def _counters(state):
    """阶段计数固定为 depth、node、edge、read，沿用原请求预算。"""
    return state.depth, state.node_count, state.edge_count, state.read_count


@observed_relation_phase("companion")
def consume_operation_companion(core, frames, facts, execution, state, *, owner_weight=1, tracer=None):
    """逐前提读取来源帧及当前 Evidence；缺帧或预算不足不能获得生成许可。"""
    if not execution_proof_consumed(execution, state):
        return state, ()
    premises = tuple(sorted({key for match in execution.matches for key in match}))
    if not premises:
        return state, ()
    before = _counters(state)
    depth = state.depth + 1
    reads, members = [], []
    original = state

    def record(current, reason):
        """将真实来源阶段交给原查询 tracer，不创建旁路 QueryState。"""
        if tracer is not None:
            tracer.record(current, reason)

    def fail(current, budget=False):
        """保留已发生读取及未闭合来源根，关闭所有部分回答。"""
        reason = TERMINATION_BUDGET_EXHAUSTED if budget else TERMINATION_CLARIFY_MISSING_BINDING
        current = current.with_(termination=reason, evidence_closed=0, generation_ready=0,
            required_slots_open=max(1, current.required_slots_open))
        record(current, reason)
        return current, tuple(reads)

    record(state, state.termination)
    if (depth > state.budget.max_depth
            or state.node_count + len(premises) > state.budget.max_nodes
            or state.edge_count + 2 * len(premises) > state.budget.max_edges
            or state.read_count + 3 * len(premises) > state.budget.max_reads):
        return fail(state, True)
    permission_source = execution.witness.source.stable_key()
    entries = tuple(FrontierEntry((3, COMPANION_PHASE, *key), 3, target_key=key, root_key=key,
        source_ref=permission_source, scope_key=state.query_key, owner_weight=owner_weight,
        depth=depth - 1, evidence_support=1) for key in premises)
    state = state.with_(termination=TERMINATION_OPEN, generation_ready=0,
        roots=(*state.roots, *(QueryRoot(3, key, ROOT_PENDING, owner_weight,
            source_ref=permission_source, scope_key=state.query_key) for key in premises)),
        anchors=(*state.anchors, *(QueryAnchor(3, key, kind=COMPANION_PHASE,
            scope_key=state.query_key) for key in premises)), frontier=(*state.frontier, *entries))
    record(state, TERMINATION_OPEN)
    for key, edge in zip(premises, entries, strict=True):
        # 物理帧索引与当前 Core input 各读一次，H-00 的每个记录单独计费。
        if state.read_count + 2 > state.budget.max_reads:
            return fail(state, True)
        state = state.with_(read_count=state.read_count + 2)
        frame, fact = frames.get(key), facts.get(key)
        if frame is None or fact is None:
            return fail(state)
        try:
            source = core.generation_input(ObjectIdentity.from_stable_key(key))
            history = core.evidence_history(source.proposition.definition.proposition)
        except TrainedRelationGraphError:
            return fail(state)
        if not history:
            return fail(state)
        if state.read_count + len(history) > state.budget.max_reads:
            return fail(state, True)
        state = state.with_(read_count=state.read_count + len(history))
        definition = source.proposition.definition
        if (frame.proposition != definition.proposition or frame.predicate != definition.predicate
                or fact.proposition != definition.proposition or fact.predicate != definition.predicate
                or frame.source_hash != fact.source_hash
                or len(fact.bindings) != len(definition.bindings)
                or {(binding.role, binding.filler) for binding in fact.bindings}
                    != {(binding.role, binding.filler) for binding in definition.bindings}
                or len(frame.roles) != len(definition.bindings)
                or set(frame.roles) != {binding.role for binding in definition.bindings}
                or len(frame.gaps) != len(frame.roles) + 1):
            return fail(state)
        expected = current_query_evidence(core_query_evidence(definition.proposition, history,
            core.context.scoped_identity_store.registry))
        actual = current_query_evidence(tuple(item for item in state.evidence
            if item.space == 1 and item.hypothesis_key == key
            and (item.evidence_key[:1] == (1,) or item.polarity != 3)))
        if not expected or set(expected) != set(actual):
            return fail(state)
        evidence = dialogue_query_evidence(definition.proposition, frame, source, None)
        binding = BindingEntry(COMPANION_PREMISE_ROLE, key, space=3, scope_key=state.query_key)
        state = state.with_(depth=depth, node_count=state.node_count + 1,
            edge_count=state.edge_count + 2,
            frontier=tuple(item for item in state.frontier if item != edge),
            roots=tuple(root.expanded() if root.owner_space == 3 and root.root_key == key
                and root.source_ref == permission_source and root.scope_key == state.query_key else root
                for root in state.roots),
            visited=(*state.visited, VisitedKey(3, key)),
            bindings=tuple(sorted(set((*state.bindings, binding)), key=lambda row: row.stable_key())),
            evidence=tuple(sorted(set((*state.evidence, evidence)), key=lambda row: row.stable_key())))
        members.append(evidence.stable_key())
        reads.append(OperationCompanionRead(key, frame.predicate.stable_key(), frame.source_hash, depth))
        record(state, TERMINATION_OPEN)
    phase = EvidenceEntry(permission_source, execution.witness.program.operation_key, space=3, polarity=3,
        evidence_key=(COMPANION_PHASE, *state.query_key), scope_key=state.query_key,
        payload_key=pack_record(COMPANION_PHASE, execution.stable_key(), before, _counters(state),
            pack_record(COMPANION_MEMBERS, *members)))
    binding = BindingEntry(COMPANION_PROOF_ROLE, execution.stable_key(), space=3, scope_key=state.query_key)
    state = state.with_(termination=original.termination, generation_ready=original.generation_ready,
        bindings=tuple(sorted(set((*state.bindings, binding)), key=lambda row: row.stable_key())),
        evidence=tuple(sorted(set((*state.evidence, phase)), key=lambda row: row.stable_key())))
    record(state, state.termination)
    return state, tuple(reads)


def operation_companion_consumed(execution, state):
    """生成门只接受同次实际执行的完整来源阶段，Memory 镜像不满足该门。"""
    if not execution_proof_consumed(execution, state):
        return False
    proof = BindingEntry(COMPANION_PROOF_ROLE, execution.stable_key(), space=3, scope_key=state.query_key)
    if proof not in state.bindings:
        return False
    phases = tuple(item for item in state.evidence if item.space == 3
        and item.evidence_key == (COMPANION_PHASE, *state.query_key)
        and item.scope_key == state.query_key and item.source_ref == execution.witness.source.stable_key()
        and item.hypothesis_key == execution.witness.program.operation_key and item.polarity == 3)
    if len(phases) != 1:
        return False
    try:
        fields = unpack_record(phases[0].payload_key, COMPANION_PHASE)
        if len(fields) != 4 or fields[0] != execution.stable_key():
            return False
        members = unpack_record(fields[3], COMPANION_MEMBERS)
    except ValueError:
        return False
    premises = tuple(sorted({key for match in execution.matches for key in match}))
    before, after = fields[1:3]
    if (not premises or len(before) != 4 or len(after) != 4
            or after[0] != before[0] + 1 or after[1] != before[1] + len(premises)
            or after[2] != before[2] + 2 * len(premises)
            or after[3] < before[3] + 3 * len(premises)
            or any(current < consumed for current, consumed in zip(_counters(state), after, strict=True))
            or any(consumed > limit for consumed, limit in zip(after,
                (state.budget.max_depth, state.budget.max_nodes, state.budget.max_edges, state.budget.max_reads), strict=True))):
        return False
    expected = tuple(item.stable_key() for key in premises for item in state.evidence
        if item.space == 3 and item.hypothesis_key == key and item.evidence_key[:1] == (2,)
        and item.payload_key[-1:] == (0,))
    return (len(members) == len(premises) and members == expected
        and all(BindingEntry(COMPANION_PREMISE_ROLE, key, space=3, scope_key=state.query_key) in state.bindings
            and VisitedKey(3, key) in state.visited
            and any(root.owner_space == 3 and root.root_key == key and root.status == ROOT_EXPANDED
                and root.scope_key == state.query_key and root.source_ref == execution.witness.source.stable_key()
                for root in state.roots) for key in premises))
