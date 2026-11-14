"""以完整整数见证解释对称关系的来源局部顺序，不修改原投影。"""
from dataclasses import dataclass

from pure_integer_ai.cognition.shared.identity import ObjectIdentity
from pure_integer_ai.cognition.shared.query_state import BindingEntry, EvidenceEntry, SPACE_CORE
from pure_integer_ai.cognition.understanding.query_input_structure import (
    INPUT_STRUCTURE_VERSION, PROJECTION_FILLER, PROJECTION_PREDICATE,
    STRUCTURE_CLOSED, STRUCTURE_ORDER_CONFLICT, InputRelationCandidate, InputSemanticCandidate,
)
from pure_integer_ai.cognition.understanding.query_memory_candidates import frame_key
from pure_integer_ai.experiments.query_relation_arbitration import (
    _surface_symmetric_relation_topology_key, _trained_symmetric_relation_role_pair,
)

SYMMETRIC_INPUT_ORDER_VERSION = 92480


def _projection_evidence(item, kind, query_key):
    """重建原UNKNOWN投影；等价见证不能代替任何原始证据。"""
    key = item.stable_key()
    return EvidenceEntry(item.source_ref, item.proposition_key, space=SPACE_CORE,
        polarity=3, trust=1, evidence_key=frame_key(91569, (kind,), key),
        scope_key=query_key, payload_key=key)


@dataclass(frozen=True, slots=True)
class SymmetricInputOrderWitness:
    """查询局部值记录；完整原候选、成员和规范证据均可跨语言重建。"""

    query_key: tuple[int, ...]
    relation: InputRelationCandidate
    support: InputRelationCandidate
    members: tuple[InputSemanticCandidate, ...]
    support_members: tuple[InputSemanticCandidate, ...]

    def stable_key(self):
        """版本化有界整数记录，保持两条独立物理来源。"""
        return frame_key(SYMMETRIC_INPUT_ORDER_VERSION, (1,), self.query_key,
            self.relation.stable_key(), self.support.stable_key(),
            *(item.stable_key() for item in (*self.members, *self.support_members)))

    def query_evidence(self):
        """见证仅有UNKNOWN立场，不授予命题真值或生成许可。"""
        key = self.stable_key()
        return EvidenceEntry(self.relation.source_ref, self.relation.proposition_key,
            space=SPACE_CORE, polarity=3, trust=1, evidence_key=key,
            scope_key=self.query_key, payload_key=key)

    def conflict_binding(self):
        """保留原来源局部冲突绑定，消费者另行核验解释。"""
        return BindingEntry(self.relation.predicate_key, self.relation.proposition_key,
            space=SPACE_CORE, scope_key=self.query_key, conflict_kept=1)

    def original_evidence(self):
        """返回见证成立所需的两条原关系和全部端点UNKNOWN证据。"""
        return tuple(_projection_evidence(item, kind, self.query_key)
            for kind, items in ((2, (self.relation, self.support)),
                (1, tuple(item for item in (*self.members, *self.support_members)
                          if item.projection_kind == PROJECTION_FILLER)))
            for item in items)

    def original_bindings(self):
        """两条原关系和全部原角色绑定缺一不可，原反序位始终保留。"""
        return (self.conflict_binding(), BindingEntry(self.support.predicate_key,
            self.support.proposition_key, space=SPACE_CORE, scope_key=self.query_key),
            *(BindingEntry(item.role_key, item.candidate_key, space=SPACE_CORE,
                scope_key=self.query_key) for item in (*self.members, *self.support_members)
                if item.projection_kind == PROJECTION_FILLER))


def _complete_members(structure, relation, facts):
    """核对精确Core来源、三成员拓扑及当前物理跨度；不猜测开放成员。"""
    roles = _trained_symmetric_relation_role_pair(relation.predicate_key)
    fact = facts.get(relation.proposition_key)
    if (roles is None or fact is None or relation.required_count != 3
            or relation.support_count != 3 or relation.required_open
            or relation.role_coverage != 2 or relation.predicate_coverage != 1
            or relation.source_ref != (fact.source_hash,)
            or fact.proposition.stable_key() != relation.proposition_key
            or fact.predicate.stable_key() != relation.predicate_key):
        return None
    bindings = tuple(fact.bindings)
    fillers = {item.role.stable_key(): item.filler.stable_key() for item in bindings}
    if len(bindings) != 2 or set(fillers) != roles or len(set(fillers.values())) != 2:
        return None
    members = tuple(sorted((item for item in structure.semantic_candidates
        if item.proposition_key == relation.proposition_key
        and item.predicate_key == relation.predicate_key
        and item.source_ref == relation.source_ref and item.span_ref in relation.span_refs),
        key=lambda item: item.member_ordinal))
    if len(members) != 3 or tuple(item.member_ordinal for item in members) != (0, 1, 2):
        return None
    topology = [INPUT_STRUCTURE_VERSION, 3]
    spans = {item.ref_key: item for item in structure.spans}
    endpoint_roles = set()
    predicate_count = 0
    for item in members:
        span = spans.get(item.span_ref)
        if (span is None or span.source_ref != structure.source_ref
                or span.values != tuple(token.value for token in structure.tokens[span.start:span.end])):
            return None
        topology.extend((item.member_ordinal, item.projection_kind, item.object_kind,
                         len(item.role_key), *item.role_key))
        if item.projection_kind == PROJECTION_FILLER:
            if item.role_key not in roles or fillers[item.role_key] != item.candidate_key:
                return None
            endpoint_roles.add(item.role_key)
        elif (item.projection_kind == PROJECTION_PREDICATE and not item.role_key
                and item.candidate_key == relation.predicate_key):
            predicate_count += 1
        else:
            return None
    if (tuple(topology) != relation.structure_key or endpoint_roles != set(roles)
            or predicate_count != 1 or len(set(item.span_ref for item in members)) != 3
            or tuple(sorted(item.span_ref for item in members)) != relation.span_refs):
        return None
    return members, tuple(sorted(fillers.values()))


def _ordered(members):
    """判断当前完整跨度的非重叠顺序，禁止相同跨度重复消费。"""
    return all(left.span_ref[3] <= right.span_ref[2]
               for left, right in zip(members, members[1:]))


def symmetric_input_order_witnesses(structure, facts):
    """只解释具有原闭合支持来源的严格双端点反序；其它冲突保持。"""
    if structure is None or any(getattr(structure, field) for field in (
            'operation_keys', 'relation_question_bindings', 'operation_request_bindings',
            'assertion_scope_bindings', 'discourse_compositions', 'discourse_gaps',
            'pending_relation_composition_bindings')):
        return ()
    rows = []
    for relation in structure.relation_candidates:
        parts = _complete_members(structure, relation, facts)
        if parts is not None:
            rows.append((relation, *parts))
    result = []
    for relation, members, endpoints in rows:
        if relation.state != STRUCTURE_ORDER_CONFLICT or relation.conflict_count != 1 or _ordered(members):
            continue
        swapped = list(members)
        positions = [index for index, item in enumerate(members) if item.projection_kind == PROJECTION_FILLER]
        left, right = positions
        swapped[left], swapped[right] = swapped[right], swapped[left]
        if not _ordered(swapped):
            continue
        roles = _trained_symmetric_relation_role_pair(relation.predicate_key)
        topology = _surface_symmetric_relation_topology_key(structure, relation, roles)
        identity = ObjectIdentity.from_stable_key(relation.proposition_key)
        supports = []
        for support, support_members, support_endpoints in rows:
            if support.state != STRUCTURE_CLOSED or support.conflict_count or not _ordered(support_members):
                continue
            other = ObjectIdentity.from_stable_key(support.proposition_key)
            if (support.predicate_key != relation.predicate_key or support_endpoints != endpoints
                    or identity.owner != other.owner or identity.versions != other.versions):
                continue
            other_topology = _surface_symmetric_relation_topology_key(structure, support, roles)
            if topology[:9] + topology[11:] == other_topology[:9] + other_topology[11:]:
                supports.append((support, support_members))
        if supports:
            support, support_members = min(supports, key=lambda pair: pair[0].stable_key())
            result.append(SymmetricInputOrderWitness(structure.source_ref, relation, support,
                                                    members, support_members))
    return tuple(sorted(result, key=lambda item: item.stable_key()))


def unresolved_input_conflict_bindings(state, witnesses):
    """原冲突Binding及全部原证据、见证均存在时才解释这一项来源局部反序。"""
    if not witnesses:
        return tuple(item for item in state.bindings if item.conflict_kept == 1)
    evidence = {item.stable_key() for item in state.evidence
        if item.space == SPACE_CORE and item.polarity == 3 and item.scope_key == state.query_key
        and item.evidence_key[:1] in ((91569,), (SYMMETRIC_INPUT_ORDER_VERSION,))}
    bindings = {item.stable_key() for item in state.bindings
                if item.space == SPACE_CORE and item.scope_key == state.query_key}
    explained = {item.conflict_binding().stable_key() for item in witnesses
        if item.query_key == state.query_key and item.query_evidence().stable_key() in evidence
        and all(original.stable_key() in evidence for original in item.original_evidence())
        and all(original.stable_key() in bindings for original in item.original_bindings())}
    return tuple(item for item in state.bindings
        if item.conflict_kept == 1 and item.stable_key() not in explained)


def verify_saved_symmetric_input_order(trace, structure, facts):
    """冷重建全部见证并核对原保留记录；禁止调用生产查询或生成。"""
    witnesses = symmetric_input_order_witnesses(structure, facts)
    evidence = set()
    actual = set()
    for ref in trace.trace['state_key']['evidence']:
        key = trace.resolve_ref(ref)
        evidence.add(key)
        cursor = 3
        for _ in range(2):
            if cursor >= len(key) or key[cursor] > len(key) - cursor - 1:
                raise ValueError('saved symmetric input order evidence framing differs')
            cursor += 1 + key[cursor]
        if cursor >= len(key):
            raise ValueError('saved symmetric input order evidence marker is missing')
        if key[cursor] > 0 and key[cursor + 1] == SYMMETRIC_INPUT_ORDER_VERSION:
            actual.add(key)
    if actual != {item.query_evidence().stable_key() for item in witnesses}:
        raise ValueError('saved symmetric input order witnesses differ from cold sources')
    bindings = {trace.resolve_ref(ref) for ref in trace.trace['state_key']['bindings']}
    for item in witnesses:
        if (any(original.stable_key() not in bindings for original in item.original_bindings())
                or any(original.stable_key() not in evidence for original in item.original_evidence())):
            raise ValueError('saved symmetric input order lost original conflict evidence')
    return len(witnesses)
