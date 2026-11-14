"""普通完整陈述的事实捕获资格；不虚构问式或操作。"""
from pure_integer_ai.cognition.shared.query_state import BindingEntry, SPACE_CORE
from pure_integer_ai.cognition.understanding.query_input_structure import (
    PROJECTION_FILLER, PROJECTION_PREDICATE, STRUCTURE_CLOSED,
)
from pure_integer_ai.cognition.understanding.query_memory_candidates import frame_key
from pure_integer_ai.cognition.understanding.query_relation_coverage import has_uncovered_closed_relation_modifier
from pure_integer_ai.experiments.graph_query_trace_contract import GraphQueryTraceEvidence
from pure_integer_ai.experiments.query_memory_ohe_audit import _unpack
from pure_integer_ai.experiments.query_relation_arbitration import (
    _surface_equivalent_relation_tie, core_relation_endpoint_identity,
)
from pure_integer_ai.experiments.symmetric_input_order import (
    _projection_evidence, symmetric_input_order_witnesses, verify_saved_symmetric_input_order,
)

ASSERTION_CAPTURE_VERSION = 92481


def delivered_assertion_capture_witness(structure, facts, result):
    """从本次完整输入及实际单事实Plan恢复资格，不消费期望答案。"""
    if structure is None or any(getattr(structure, field) for field in (
            'operation_keys', 'relation_question_bindings', 'operation_request_bindings',
            'assertion_scope_bindings', 'discourse_compositions', 'discourse_gaps',
            'pending_relation_composition_bindings')) or has_uncovered_closed_relation_modifier(structure):
        raise ValueError('plain assertion capture requires a complete unscoped input')
    trace = GraphQueryTraceEvidence(result)
    if (result.get('termination') != 1 or result.get('operation_keys')
            or trace.operation_contracts() or result.get('generation', {}).get('kind') != 1
            or trace.resolve_ref(result['query_ref']) != structure.source_ref
            or trace.resolve_ref(result['input_structure']['stable_key']['integer_key_ref']) != structure.stable_key()):
        raise ValueError('plain assertion capture differs from the actual complete query')
    claims = tuple(tuple(key) for key in result['response_plan']['claim_refs'])
    if len(claims) != 1 or claims[0] != trace.resolve_ref(result['state_key']['scalars'][11]):
        raise ValueError('plain assertion capture requires the unique actual primary claim')
    claim, = claims
    selected = tuple(item for item in structure.relation_candidates if item.proposition_key == claim)
    if len(selected) != 1:
        raise ValueError('plain assertion lacks one actual source projection')
    relation, = selected
    if (relation.state != STRUCTURE_CLOSED or relation.required_open or relation.conflict_count
            or relation.support_count != relation.required_count or relation.predicate_coverage != 1):
        raise ValueError('plain assertion selected projection is not complete')
    fact = facts.get(claim)
    if fact is None or fact.proposition.stable_key() != claim or fact.predicate.stable_key() != relation.predicate_key:
        raise ValueError('plain assertion lacks the original complete Core fact')
    expected = {item.role.stable_key(): item.filler.stable_key() for item in fact.bindings}
    members = tuple(item for item in structure.semantic_candidates
        if item.proposition_key == claim and item.predicate_key == relation.predicate_key
        and item.source_ref == relation.source_ref and item.span_ref in relation.span_refs)
    fillers = tuple(item for item in members if item.projection_kind == PROJECTION_FILLER)
    predicates = tuple(item for item in members if item.projection_kind == PROJECTION_PREDICATE)
    if (len(expected) != len(fact.bindings) or len(fillers) != len(expected)
            or {item.role_key: item.candidate_key for item in fillers} != expected
            or len(members) != relation.required_count or relation.required_count != len(expected) + 1
            or len(predicates) != 1 or predicates[0].candidate_key != relation.predicate_key):
        raise ValueError('plain assertion full physical roles differ from Core')
    witnesses = symmetric_input_order_witnesses(structure, facts)
    verify_saved_symmetric_input_order(trace, structure, facts)
    explained = {item.relation.stable_key() for item in witnesses}
    closed = []
    for candidate in structure.relation_candidates:
        other = facts.get(candidate.proposition_key)
        if (other is None or candidate.source_ref != (other.source_hash,)
                or other.predicate.stable_key() != fact.predicate.stable_key()
                or core_relation_endpoint_identity(other) != core_relation_endpoint_identity(fact)):
            raise ValueError('plain assertion contains different physical Core alternatives')
        if candidate.stable_key() in explained:
            continue
        if (candidate.state != STRUCTURE_CLOSED or candidate.required_open or candidate.conflict_count
                or candidate.required_count != candidate.support_count):
            raise ValueError('plain assertion retains an unexplained incomplete alternative')
        closed.append(candidate)
    if len(closed) > 1 and not _surface_equivalent_relation_tie(structure, closed):
        raise ValueError('plain assertion has distinct complete observation topologies')
    bindings = {trace.resolve_ref(ref) for ref in result['state_key']['bindings']}
    conflicts = {key for key in bindings if key[-1]}
    if conflicts != {item.conflict_binding().stable_key() for item in witnesses}:
        raise ValueError('plain assertion retains unexplained binding conflicts')
    evidence = {trace.resolve_ref(ref) for ref in result['state_key']['evidence']}
    originals = ((2, relation), *((1, item) for item in fillers))
    if (any(_projection_evidence(item, kind, structure.source_ref).stable_key() not in evidence
            for kind, item in originals)
            or any(BindingEntry(role, filler, space=SPACE_CORE, scope_key=structure.source_ref).stable_key()
                not in bindings for role, filler in ((relation.predicate_key, claim), *expected.items()))):
        raise ValueError('plain assertion lost original projection evidence or bindings')
    positive = tuple(_unpack(key, 3, 6) for key in evidence if key[:2] == (SPACE_CORE, 1))
    if not any(parts[0] == (fact.source_hash,) and parts[1] == claim for parts in positive):
        raise ValueError('plain assertion lacks current positive Core evidence')
    return claim, frame_key(ASSERTION_CAPTURE_VERSION, (1,), structure.source_ref,
                           structure.stable_key(), relation.stable_key())
