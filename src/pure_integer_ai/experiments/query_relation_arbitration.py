"""关系来源与对称端点的纯整数仲裁，不更改图身份或证据。"""
from __future__ import annotations

from pure_integer_ai.cognition.understanding.query_input_structure import (
    INPUT_STRUCTURE_VERSION, PROJECTION_FILLER, PROJECTION_PREDICATE,
    STRUCTURE_CLOSED, STRUCTURE_OPEN, QueryInputStructure,
)
from pure_integer_ai.cognition.understanding.query_provenance_identity import (
    semantic_candidate_identity_key,
)
from pure_integer_ai.experiments.ph2_authored_relation_compile import (
    authored_relation_identity, authored_relation_role_identity,
)
from pure_integer_ai.experiments.ph2_authored_relation_schema import (
    RELATION_PURE_ALIAS, RELATION_SIMILAR, RELATION_ANTONYM,
    ROLE_ALIAS_LEFT, ROLE_ALIAS_RIGHT, ROLE_SIMILAR_LEFT, ROLE_SIMILAR_RIGHT,
    ROLE_ANTONYM_LEFT, ROLE_ANTONYM_RIGHT,
)
from pure_integer_ai.experiments.query_pending_reference import (
    pending_reference_relation_keys,
)


def _surface_relation_topology_key(
        input_structure, candidate, *, include_endpoint_identity: bool = False):
    """建立保留角色、跨度及状态的观察拓扑键，不改动图对象与来源证据。"""
    members = tuple(sorted({
        (
            item.member_ordinal,
            item.projection_kind,
            item.object_kind,
            (semantic_candidate_identity_key(item.candidate_key)
             if include_endpoint_identity else ()),
            item.role_key,
            item.span_ref,
        )
        for item in input_structure.semantic_candidates
        if (item.proposition_key == candidate.proposition_key
                and item.predicate_key == candidate.predicate_key
                and item.source_ref == candidate.source_ref
                and item.span_ref in candidate.span_refs)
    }))
    return (
        candidate.predicate_key,
        candidate.structure_key,
        candidate.carrier_key,
        candidate.span_refs,
        candidate.required_count,
        candidate.support_count,
        candidate.role_coverage,
        candidate.predicate_coverage,
        candidate.required_open,
        candidate.state,
        candidate.conflict_count,
        members,
    )


def _trained_symmetric_relation_role_pair(predicate_key):
    """只返回冻结训练编译器授权的对称关系 Role 对，不从自然词面猜测。"""
    pairs = (
        (
            authored_relation_identity(RELATION_PURE_ALIAS).stable_key(),
            frozenset({
                authored_relation_role_identity(ROLE_ALIAS_LEFT).stable_key(),
                authored_relation_role_identity(ROLE_ALIAS_RIGHT).stable_key(),
            }),
        ),
        (
            authored_relation_identity(RELATION_SIMILAR).stable_key(),
            frozenset({
                authored_relation_role_identity(ROLE_SIMILAR_LEFT).stable_key(),
                authored_relation_role_identity(ROLE_SIMILAR_RIGHT).stable_key(),
            }),
        ),
        (
            authored_relation_identity(RELATION_ANTONYM).stable_key(),
            frozenset({
                authored_relation_role_identity(ROLE_ANTONYM_LEFT).stable_key(),
                authored_relation_role_identity(ROLE_ANTONYM_RIGHT).stable_key(),
            }),
        ),
    )
    for relation, roles in pairs:
        if predicate_key == relation:
            return roles
    return None


def _symmetric_structure_topology_key(structure_key, role_keys):
    """仅消除合法输入拓扑中授权对称端点的 Role/ordinal 配对。"""
    if not role_keys:
        return structure_key
    # ``TrainedInputStructure.topology_key`` encodes member ordinal before the
    # Role key.  A symmetric relation's two endpoint ordinals are precisely
    # the order that the typed rule authorizes us to forget.  Normalize only
    # that recognized topology form; opaque semantic StructureConcept keys and
    # unrelated structure variants remain byte-for-byte distinct.
    if (isinstance(structure_key, tuple)
            and len(structure_key) >= 2
            and structure_key[0] == INPUT_STRUCTURE_VERSION):
        member_count = structure_key[1]
        cursor = 2
        records = []
        valid = (type(member_count) is int
                 and 0 <= member_count <= (len(structure_key) - 2) // 4)
        for _ in range(member_count if valid else 0):
            if cursor + 4 > len(structure_key):
                valid = False
                break
            ordinal, projection_kind, object_kind, role_size = (
                structure_key[cursor:cursor + 4])
            if (any(type(value) is not int or value < 0
                    for value in (ordinal, projection_kind,
                                  object_kind, role_size))
                    or cursor + 4 + role_size > len(structure_key)):
                valid = False
                break
            role = tuple(structure_key[cursor + 4:cursor + 4 + role_size])
            if any(type(value) is not int or value < 0 for value in role):
                valid = False
                break
            records.append((ordinal, projection_kind, object_kind, role))
            cursor += 4 + role_size
        if valid and cursor != len(structure_key):
            valid = False
        structure_fillers = tuple(
            item for item in records
            if item[1] == PROJECTION_FILLER and item[3] in role_keys)
        if (valid and len(structure_fillers) == 2
                and {item[3] for item in structure_fillers} == set(role_keys)):
            normalized_records = []
            for ordinal, projection_kind, object_kind, role in records:
                if (projection_kind == PROJECTION_FILLER
                        and role in role_keys):
                    # Keep sequence position and object kind, while dropping
                    # only the endpoint ordinal/Role pairing authorized by
                    # the typed symmetric relation rule.
                    normalized_records.append((
                        0, projection_kind, object_kind))
                else:
                    normalized_records.append((
                        1, ordinal, projection_kind, object_kind, role))
            structure_key = (
                INPUT_STRUCTURE_VERSION,
                member_count,
                tuple(normalized_records),
            )
    return structure_key


def _surface_symmetric_relation_topology_key(input_structure, candidate,
                                             role_keys):
    """仅规范化授权对称端点的顺序，保留端点身份、观察跨度和其他结构。"""
    members = []
    filler_count = 0
    valid_roles = set()
    for item in input_structure.semantic_candidates:
        if not (
                item.proposition_key == candidate.proposition_key
                and item.predicate_key == candidate.predicate_key
                and item.source_ref == candidate.source_ref
                and item.span_ref in candidate.span_refs):
            continue
        if item.projection_kind == PROJECTION_FILLER:
            filler_count += 1
            valid_roles.add(item.role_key)
            members.append((
                item.projection_kind,
                item.object_kind,
                semantic_candidate_identity_key(item.candidate_key),
                item.span_ref,
            ))
            continue
        # The predicate cue has no endpoint role and remains ordered as a
        # syntax member.  Unknown projection kinds are retained verbatim and
        # therefore cannot accidentally become a symmetric tie.
        if item.projection_kind == PROJECTION_PREDICATE:
            members.append((
                item.projection_kind, item.object_kind, item.span_ref))
        else:
            members.append((
                item.projection_kind,
                item.object_kind,
                item.role_key,
                item.span_ref,
            ))
    if filler_count != 2 or valid_roles != set(role_keys):
        return _surface_relation_topology_key(input_structure, candidate)
    structure_key = _symmetric_structure_topology_key(
        candidate.structure_key, role_keys)
    return (
        candidate.predicate_key,
        structure_key,
        candidate.carrier_key,
        candidate.span_refs,
        candidate.required_count,
        candidate.support_count,
        candidate.role_coverage,
        candidate.predicate_coverage,
        candidate.required_open,
        candidate.state,
        candidate.conflict_count,
        tuple(sorted(members)),
    )


def _surface_open_relation_topology_key(input_structure, candidate):
    """建立开放 WH 的观察拓扑；完整 Core 端点在竞争值中另行核验。"""
    members = tuple(sorted({
        (item.member_ordinal, item.projection_kind, item.object_kind,
         item.span_ref)
        for item in input_structure.semantic_candidates
        if (item.proposition_key == candidate.proposition_key
                and item.predicate_key == candidate.predicate_key
                and item.source_ref == candidate.source_ref
                and item.span_ref in candidate.span_refs)
    }))
    return (
        candidate.predicate_key,
        candidate.carrier_key,
        candidate.span_refs,
        candidate.required_count,
        candidate.role_coverage,
        candidate.predicate_coverage,
        candidate.required_open,
        candidate.state,
        candidate.conflict_count,
        members,
    )


def _surface_equivalent_relation_tie(input_structure, candidates) -> bool:
    """识别完整正关系的同形来源平局，保留图身份及不同拓扑的歧义。"""
    values = tuple(candidates)
    if len(values) < 2:
        return False
    if any(
            item.state != STRUCTURE_CLOSED
            or item.required_open != 0
            or item.conflict_count != 0
            or item.predicate_coverage != 1
            or item.required_count != item.support_count
            for item in values):
        return False
    predicate_keys = {item.predicate_key for item in values}
    role_keys = None
    if len(predicate_keys) == 1:
        role_keys = _trained_symmetric_relation_role_pair(
            next(iter(predicate_keys)))
    if role_keys is not None:
        topology = {
            _surface_symmetric_relation_topology_key(
                input_structure, item, role_keys)
            for item in values
        }
    else:
        topology = {
            _surface_relation_topology_key(input_structure, item)
            for item in values
        }
    return len(topology) == 1



def core_relation_endpoint_identity(fact) -> tuple[
        tuple[tuple[int, ...], tuple[int, ...]], ...]:
    """返回 Core 完整绑定的查询比较键，不合并物理来源或图对象。

    只有冻结 typed symmetric 关系且完整 Role 对精确匹配时才消除两端
    配对顺序；所有声明语义、对象类型、owner 和版本仍由身份键保留。
    其他关系保留原有的有序 Role/filler 绑定。
    """
    bindings = tuple(fact.bindings)
    roles = tuple(item.role.stable_key() for item in bindings)
    if (len(bindings) != len(set(roles))
            or any(not item.filler.stable_key() for item in bindings)):
        return ()
    values = tuple(semantic_candidate_identity_key(item.filler.stable_key())
                   for item in bindings)
    symmetric_roles = _trained_symmetric_relation_role_pair(
        fact.predicate.stable_key())
    if (symmetric_roles is not None and len(bindings) == 2
            and set(roles) == symmetric_roles):
        return tuple(zip(sorted(roles), sorted(values)))
    return tuple(zip(roles, values))


def select_equivalent_slot_question_bindings(validated):
    """从已逐项验证的普通 WH 候选中选取完全等价的物理代表。

    每项为 `(binding, parts)`，parts 是既有 Core 槽验证器的完整结果。
    核对完整有序端点、答案角色及语法/跨度/载体/拓扑；仅已授权的
    对称关系可消除端点配对顺序。动态指代和真实语义差异保持歧义。
    """
    records = tuple(validated)
    if len(records) < 2:
        return tuple(binding for binding, _parts in records)
    contracts = set()
    for binding, parts in records:
        frame, relation, fact, answer_key, dynamic_roles = parts
        roles = _trained_symmetric_relation_role_pair(frame.predicate_key)
        if (dynamic_roles or frame.is_reference_slot
                or binding.answer_role_key != frame.answer_role_key):
            return ()
        fact_fillers = {
            item.role.stable_key(): item.filler.stable_key()
            for item in fact.bindings}
        endpoints = core_relation_endpoint_identity(fact)
        if (not endpoints or len(fact.bindings) != len(fact_fillers)
                or frame.roles != tuple(item.role.stable_key()
                    for item in fact.bindings
                    if item.role.stable_key() != binding.answer_role_key)
                or fact.predicate.stable_key() != frame.predicate_key
                or fact_fillers.get(binding.answer_role_key) != answer_key):
            return ()
        if roles is not None and (
                len(fact.bindings) != 2 or set(fact_fillers) != roles):
            return ()
        contracts.add((
            binding.operation_key, frame.predicate_key,
            (() if roles is not None else (frame.roles, binding.answer_role_key)),
            frame.gaps, frame.wh_values, frame.slot_mode,
            binding.slot_span_refs,
            tuple((span_ref, semantic_candidate_identity_key(fact_fillers[role]))
                  for role, span_ref in zip(frame.roles, binding.slot_span_refs)),
            endpoints,
            semantic_candidate_identity_key(answer_key),
            _symmetric_structure_topology_key(relation.structure_key, roles),
            relation.carrier_key, relation.span_refs,
            relation.required_count, relation.support_count,
            relation.role_coverage, relation.predicate_coverage,
            relation.required_open, relation.state, relation.conflict_count,
        ))
    if len(contracts) != 1:
        return ()
    return (min((binding for binding, _parts in records),
                key=lambda item: item.stable_key()),)

def input_relation_competition(
        structure: QueryInputStructure,
        core_fact_index,
        *,
        question_predicate_keys: frozenset[tuple[int, ...]] | None,
        question_frames=(),
        ) -> bool:
    """比较同一局部输入拓扑的完整 Core 端点，提前保留真实歧义。"""
    relation_by_key = {
        item.stable_key(): item for item in structure.relation_candidates}
    pending_references = pending_reference_relation_keys(
        structure, core_fact_index, question_frames)

    def scoped(groups):
        """仅恢复当前组合范围中仍存在的候选，不扩大查询作用域。"""
        for keys in groups:
            values = tuple(
                relation_by_key[key] for key in keys
                if key in relation_by_key)
            if values:
                yield values

    if structure.discourse_compositions:
        candidate_groups = tuple(
            group for composition in structure.discourse_compositions
            for group in composition.operand_relation_keys)
    elif structure.discourse_gaps:
        candidate_groups = tuple(
            group for gap in structure.discourse_gaps
            for group in (gap.left_relation_keys,
                          gap.right_relation_keys))
    else:
        candidate_groups = (
            tuple(item.stable_key() for item in structure.relation_candidates),)

    for candidates in scoped(candidate_groups):
        by_surface: dict[tuple[object, ...], set[tuple[object, ...]]] = {}
        for candidate in candidates:
            if candidate.stable_key() in pending_references:
                continue
            fact = core_fact_index.get(candidate.proposition_key)
            if fact is None or candidate.source_ref != (fact.source_hash,):
                continue
            if fact.predicate.stable_key() != candidate.predicate_key:
                continue
            endpoint_identity = core_relation_endpoint_identity(fact)
            if not endpoint_identity:
                continue
            complete = bool(
                candidate.state == STRUCTURE_CLOSED
                and candidate.required_open == 0
                and candidate.support_count == candidate.required_count
                and candidate.conflict_count == 0)
            wh_slot = bool(
                question_predicate_keys
                and candidate.predicate_key in question_predicate_keys
                and candidate.state == STRUCTURE_OPEN
                and candidate.predicate_coverage == 1
                and candidate.required_open > 0
                and candidate.required_count == len(fact.bindings) + 1
                and candidate.conflict_count == 0)
            if not (complete or wh_slot):
                continue
            surface_key = (
                _surface_open_relation_topology_key(structure, candidate)
                if wh_slot else _surface_relation_topology_key(
                    structure, candidate, include_endpoint_identity=False))
            by_surface.setdefault(surface_key, set()).add(
                (endpoint_identity, candidate.predicate_key))
        if any(len(identities) > 1 for identities in by_surface.values()):
            return True
    return False
