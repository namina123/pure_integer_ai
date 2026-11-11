"""由完整真实历史支持集合约束指代来源一致性，不提供事实或目标采用授权。"""
from dataclasses import dataclass

from pure_integer_ai.cognition.understanding.query_input_structure import (
    INPUT_STRUCTURE_VERSION, PROJECTION_FILLER,
)
from pure_integer_ai.cognition.understanding.query_open_roles import pack_record
from pure_integer_ai.cognition.understanding.query_provenance_identity import (
    semantic_candidate_identity_key,
)


REFERENCE_SOURCE_CONSENSUS_VERSION = 91656


def _question_structure_topology(structure_key, role_order):
    """只为已识别且完整覆盖 Role 的拓扑改用当前问式序，不交换角色配对。"""
    if len(structure_key) < 2 or structure_key[0] != INPUT_STRUCTURE_VERSION:
        return structure_key
    count, cursor, records = structure_key[1], 2, []
    if type(count) is not int or not 0 < count <= (len(structure_key) - 2) // 4:
        return structure_key
    for _index in range(count):
        if cursor + 4 > len(structure_key):
            return structure_key
        ordinal, kind, object_kind, size = structure_key[cursor:cursor + 4]
        if size < 0 or cursor + 4 + size > len(structure_key):
            return structure_key
        role = structure_key[cursor + 4:cursor + 4 + size]
        records.append((ordinal, kind, object_kind, role))
        cursor += 4 + size
    fillers = tuple(item[3] for item in records if item[1] == PROJECTION_FILLER)
    if (cursor != len(structure_key) or len(fillers) != len(role_order)
            or len(set(fillers)) != len(fillers) or set(fillers) != set(role_order)):
        return structure_key
    canonical = tuple(sorted(pack_record(1,
        (role_order.index(role) if kind == PROJECTION_FILLER else ordinal, kind, object_kind), role)
        for ordinal, kind, object_kind, role in records))
    return pack_record(REFERENCE_SOURCE_CONSENSUS_VERSION, (4,), *canonical)


@dataclass(frozen=True, slots=True)
class GroundedReferenceSource:
    """一个候选的完整来源、当前指代集合及全部历史角色支持。"""

    binding_key: tuple[int, ...]
    reference_candidate_key: tuple[int, ...]
    antecedent_keys: tuple[tuple[int, ...], ...]
    contract_key: tuple[int, ...]

    def __post_init__(self):
        """保留完整规范支持集合，不以频次、最近出现或一个历史代表替代。"""
        if (type(self.antecedent_keys) is not tuple or not self.antecedent_keys
                or self.antecedent_keys != tuple(sorted(set(self.antecedent_keys)))):
            raise ValueError("来源一致候选缺少完整规范历史支持集合")
        for key in self.antecedent_keys:
            if not key:
                raise ValueError("来源一致候选含空历史身份")
            pack_record(REFERENCE_SOURCE_CONSENSUS_VERSION, key)

    def stable_key(self) -> tuple[int, ...]:
        """保留可逆整数成员，不用摘要替代候选本体。"""
        return pack_record(REFERENCE_SOURCE_CONSENSUS_VERSION, (1,), self.binding_key,
                           self.reference_candidate_key,
                           pack_record(1, *self.antecedent_keys), self.contract_key)


@dataclass(frozen=True, slots=True)
class ReferenceSourceConsensus:
    """全部来源共享一项指代解释；代表只决定执行顺序，不是事实 Evidence。"""

    members: tuple[GroundedReferenceSource, ...]

    def __post_init__(self) -> None:
        """要求成员规范且合同与完整历史支持集合一致。"""
        if (len(self.members) < 2 or self.members != tuple(sorted(
                set(self.members), key=lambda item: item.stable_key()))
                or len({item.binding_key for item in self.members}) != len(self.members)
                or len({(item.antecedent_keys, item.contract_key) for item in self.members}) != 1):
            raise ValueError("来源一致记录必须保留全部规范且同义的历史约束成员")

    @property
    def selected_binding_key(self) -> tuple[int, ...]:
        """仅从已证明一致的成员中选确定性的物理执行代表。"""
        return min(item.binding_key for item in self.members)

    def stable_key(self) -> tuple[int, ...]:
        """将全部候选、历史角色及代表写入当前 QueryState 的可逆记录。"""
        return pack_record(REFERENCE_SOURCE_CONSENSUS_VERSION, (2,),
                           self.selected_binding_key, *(item.stable_key() for item in self.members))


def grounded_reference_source(
        structure, binding, parts, dynamic_span, reference_candidates,
        antecedent_keys: tuple[tuple[int, ...], ...],
        ) -> GroundedReferenceSource:
    """比较完整有序声明、当前语法和全部历史支持，保留物理来源作为成员。"""
    frame, relation, fact, answer, dynamic_roles = parts
    if not frame.is_reference_slot or len(dynamic_roles) != 1:
        raise ValueError("来源一致候选必须是已核验的唯一显式指代槽")
    full_open = tuple(item for item in structure.open_relations
                      if item.frame.proposition == relation.proposition_key
                      and item.frame.predicate == frame.predicate_key
                      and item.frame.source_ref == relation.source_ref)
    if len(full_open) != 1:
        raise ValueError("来源一致候选缺少唯一完整问式角色序")
    role_order = full_open[0].frame.roles
    spans = {item.ref_key: item for item in structure.spans}
    endpoints = tuple(pack_record(1, item.role.stable_key(),
                                 semantic_candidate_identity_key(item.filler.stable_key()))
                      for item in sorted(fact.bindings, key=lambda item: item.role.stable_key()))
    projections = tuple(sorted(pack_record(1,
        (spans[item.span_ref].start, spans[item.span_ref].end,
         item.projection_kind, item.object_kind), item.role_key,
        item.span_ref, semantic_candidate_identity_key(item.candidate_key))
        for item in structure.semantic_candidates
        if item.proposition_key == relation.proposition_key
        and item.source_ref == relation.source_ref and item.span_ref in relation.span_refs))
    contract = pack_record(REFERENCE_SOURCE_CONSENSUS_VERSION, (3,),
        binding.operation_key, frame.predicate_key, pack_record(1, *frame.roles),
        binding.answer_role_key, frame.wh_values, (frame.slot_mode,), pack_record(1, *frame.gaps),
        pack_record(1, *binding.slot_span_refs),
        _question_structure_topology(relation.structure_key, role_order), relation.carrier_key,
        pack_record(1, *relation.span_refs),
        (relation.required_count, relation.support_count, relation.role_coverage,
         relation.predicate_coverage, relation.required_open, relation.state, relation.conflict_count),
        pack_record(1, *endpoints), pack_record(1, *projections), answer, dynamic_roles[0],
        (fact.proposition.object_kind,), fact.proposition.owner.stable_key(),
        fact.proposition.versions.stable_key(), (dynamic_span.start, dynamic_span.end),
        dynamic_span.values, structure.source_ref)
    return GroundedReferenceSource(binding.stable_key(), reference_candidates.stable_key(),
                                   antecedent_keys, contract)


def reference_source_consensus(members) -> ReferenceSourceConsensus | None:
    """不同目标、历史角色或任一合同漂移均保留竞争。"""
    values = tuple(sorted(set(members), key=lambda item: item.stable_key()))
    if (len(values) < 2 or len({item.binding_key for item in values}) != len(values)
            or len({(item.antecedent_keys, item.contract_key) for item in values}) != 1):
        return None
    return ReferenceSourceConsensus(values)
