"""在查询消费边界聚合物理重复来源，保留不同声明及其完整投影。"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from pure_integer_ai.cognition.understanding.query_input_structure import (
    QueryInputStructure, relation_candidate_semantic_key,
)
from pure_integer_ai.cognition.understanding.query_provenance_identity import (
    semantic_candidate_identity_key,
)


def _relation_scope(item) -> tuple[tuple[int, ...], ...]:
    """以完整命题、谓词和训练来源限定成员归属。"""
    return item.proposition_key, item.predicate_key, item.source_ref


def _source_provenance_key(
        source_ref: tuple[int, ...],
        provenance: Mapping[tuple[int, ...], tuple[int, ...]],
        ) -> tuple[int, ...]:
    """Return a verified source proof, isolating unproved references."""
    if not provenance:
        return source_ref
    return provenance.get(source_ref, (0, *source_ref))


def _semantic_scope(
        item,
        provenance: Mapping[tuple[int, ...], tuple[int, ...]],
        ) -> tuple[object, ...]:
    proposition = getattr(item, "proposition_key", None)
    if proposition is None:
        proposition = item.proposition
    predicate = getattr(item, "predicate_key", None)
    if predicate is None:
        predicate = item.predicate
    return (
        proposition,
        predicate,
        _source_provenance_key(item.source_ref, provenance),
    )


def aggregate_input_provenance_duplicates(
        structure: QueryInputStructure,
        *,
        source_provenance: Mapping[
            tuple[int, ...], tuple[int, ...]] | None = None,
        ) -> QueryInputStructure:
    """按完整输入拓扑选取物理代表；不同候选的共享成员不能相互删去。

    开放框架还必须携带所属关系的语义分组，防止相同框架包络把不同
    filler 声明合成一个候选。未观测到关系投影的框架保留原始来源身份。
    已绑定请求的 reader 引用保持精确，不跨对象改写其目标或证明。
    """
    if source_provenance is not None and not isinstance(
            source_provenance, Mapping):
        raise TypeError("source_provenance 必须是 source_ref 到整数 proof 的映射")
    merged_provenance = dict(structure.source_provenance_keys)
    if source_provenance:
        merged_provenance.update(source_provenance)
    provenance_items = tuple(sorted(
        merged_provenance.items(), key=lambda item: item[0]))
    if (not structure.relation_candidates and not structure.open_relations
            and provenance_items == structure.source_provenance_keys):
        return structure
    if provenance_items != structure.source_provenance_keys:
        structure = replace(
            structure, source_provenance_keys=provenance_items)
    provenance = dict(provenance_items)
    relation_key_map = {}
    retained_relations = []
    groups = {}
    for candidate in structure.relation_candidates:
        # relation_candidate_semantic_key includes the verified proof once it
        # is present on the structure.  Keeping the grouping here explicit
        # makes the duplicate boundary auditable and deterministic.
        key = relation_candidate_semantic_key(structure, candidate)
        groups.setdefault(key, []).append(candidate)
    for _key, members in sorted(groups.items(), key=lambda item: item[0]):
        representative = members[0]
        retained_relations.append(representative)
        for member in sorted(members, key=lambda item: item.stable_key()):
            relation_key_map[member.stable_key()] = representative.stable_key()
    retained_relations = tuple(sorted(
        retained_relations, key=lambda item: item.stable_key()))

    all_refs = {}
    retained_refs = {}
    context_by_scope = {}
    for relation in structure.relation_candidates:
        scope = _relation_scope(relation)
        all_refs.setdefault(scope, set()).update(relation.span_refs)
        context_by_scope.setdefault(_semantic_scope(relation, provenance), set()).add(
            relation_candidate_semantic_key(structure, relation))
    for relation in retained_relations:
        retained_refs.setdefault(_relation_scope(relation), set()).update(
            relation.span_refs)
    request_projection_keys = {
        projection
        for binding in structure.operation_request_bindings
        for _variable, _target, _span, projections in binding.readings
        for projection in projections
    }
    semantic = tuple(
        item for item in structure.semantic_candidates
        if (item.span_ref not in all_refs.get(_relation_scope(item), ())
            or item.span_ref in retained_refs.get(_relation_scope(item), ())
            or item.stable_key() in request_projection_keys))
    # 每条保留关系仍须拥有自己完整的成员；只删已丢弃物理关系的投影。
    # 不按局部 proposition id 再次全局去重，否则不同端点的候选会失去
    # 共享 predicate/subject，第二次调用也会改变第一次的仲裁结果。
    retained_carrier_scopes = {
        (item.structure_key,
         _source_provenance_key(item.source_ref, provenance))
        for item in retained_relations}
    all_carrier_scopes = {
        (item.structure_key,
         _source_provenance_key(item.source_ref, provenance))
        for item in structure.relation_candidates}
    carriers = tuple(
        item for item in structure.carrier_candidates
        if ((item.structure_key,
             _source_provenance_key(item.source_ref, provenance))
            not in all_carrier_scopes
            or (item.structure_key,
                _source_provenance_key(item.source_ref, provenance))
            in retained_carrier_scopes))

    open_representatives = {}
    for candidate in structure.open_relations:
        frame = candidate.frame
        scope = frame.proposition, frame.predicate, frame.source_ref
        semantic_scope = _semantic_scope(frame, provenance)
        contexts = context_by_scope.get(semantic_scope)
        relation_context = ((1, tuple(sorted(contexts))) if contexts
                            else (0, frame.proposition,
                                  _source_provenance_key(
                                      frame.source_ref, provenance)))
        key = (
            semantic_candidate_identity_key(frame.proposition),
            frame.predicate, frame.roles, frame.gaps, frame.envelope,
            _source_provenance_key(candidate.source_ref, provenance),
            candidate.values,
            tuple((item.role, item.source_ref, item.ordinal,
                   item.start, item.end, item.values)
                  for item in candidate.bindings),
            relation_context,
        )
        # 框架和完整成员必须取同一物理代表，供后续精确图 join 回读。
        rank = (scope not in retained_refs, candidate.stable_key())
        prior = open_representatives.get(key)
        if prior is None or rank < prior[0]:
            open_representatives[key] = rank, candidate
    open_relations = tuple(sorted(
        (item[1] for item in open_representatives.values()),
        key=lambda item: item.stable_key()))

    def remap_group(group):
        """同步重写组合中的关系引用，保持当前结构引用闭合。"""
        return tuple(sorted({relation_key_map.get(key, key) for key in group}))

    compositions = tuple(sorted({
        replace(item, operand_relation_keys=tuple(
            remap_group(group) for group in item.operand_relation_keys))
        for item in structure.discourse_compositions
    }, key=lambda item: item.stable_key()))
    gaps = tuple(sorted({
        replace(item, left_relation_keys=remap_group(item.left_relation_keys),
                right_relation_keys=remap_group(item.right_relation_keys))
        for item in structure.discourse_gaps
    }, key=lambda item: item.stable_key()))
    questions = tuple(sorted({
        replace(item, relation_key=relation_key_map.get(
            item.relation_key, item.relation_key))
        for item in structure.relation_question_bindings
    }, key=lambda item: item.stable_key()))
    return replace(
        structure, relation_candidates=retained_relations,
        semantic_candidates=semantic, carrier_candidates=carriers,
        open_relations=open_relations, discourse_compositions=compositions,
        discourse_gaps=gaps, relation_question_bindings=questions,
    )
