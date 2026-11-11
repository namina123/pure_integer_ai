"""来源聚合必须保留真实声明竞争、各自成员和重复调用后的结构闭合。"""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from pure_integer_ai.cognition.shared.identity import (
    ObjectIdentity, OwnerScope, SourceRef, VersionBundle,
)
from pure_integer_ai.cognition.shared.semantic_object import (
    entity_identity, proposition_identity, role_identity, set_expr_identity,
)
from pure_integer_ai.cognition.understanding.query_input_provenance import (
    aggregate_input_provenance_duplicates,
)
from pure_integer_ai.cognition.understanding.query_input_structure import (
    PROJECTION_FILLER, PROJECTION_PREDICATE, STRUCTURE_CLOSED,
    InputRelationQuestionBinding, TrainedInputDiscourseOperator,
    TrainedInputMember, TrainedInputStructure, TrainedInputStructureProjector,
    relation_candidate_semantic_groups,
)
from pure_integer_ai.cognition.understanding.query_open_roles import OpenRoleFrame
from pure_integer_ai.cognition.understanding.query_operation_requests import (
    FRAME, OperationRequestFrame, attach_operation_request,
)
from pure_integer_ai.cognition.understanding.query_provenance_identity import (
    semantic_candidate_identity_key,
)
from pure_integer_ai.experiments.trained_graph_query_bridge import (
    TrainedGraphQueryBridge, _surface_equivalent_relation_tie,
)
from pure_integer_ai.experiments.ph2_authored_relation_compile import (
    authored_relation_identity, authored_relation_role_identity,
)
from pure_integer_ai.experiments.ph2_authored_relation_schema import (
    RELATION_SIMILAR, ROLE_SIMILAR_LEFT, ROLE_SIMILAR_RIGHT,
)


PREDICATE = ObjectIdentity(4, (800, 4))
ROLES = role_identity((7,)), role_identity((8,))


def _source(row, **kwargs):
    """生成有独立物理来源的最小测试声明。"""
    return SourceRef(206, row, 0, kwargs.get('owner', OwnerScope()),
                     kwargs.get('versions', VersionBundle()))


def _structure(row, target=11):
    """共享主体与词面，目标局部声明可独立变化。"""
    source = _source(row)
    prop = proposition_identity(source, (1, 1)).stable_key()
    predicate = PREDICATE.stable_key()
    source_ref = (row,)
    subject = entity_identity(source, (10, 13))
    target = set_expr_identity(source, (10, target))
    members = (
        TrainedInputMember((41,), subject.stable_key(), subject.object_kind,
            PROJECTION_FILLER, prop, predicate, source_ref, 0, 1, 0, ROLES[0].stable_key()),
        TrainedInputMember((43,), predicate, 4, PROJECTION_PREDICATE,
            prop, predicate, source_ref, 1, 2, 1),
        TrainedInputMember((42,), target.stable_key(), target.object_kind,
            PROJECTION_FILLER, prop, predicate, source_ref, 2, 3, 2, ROLES[1].stable_key()),
    )
    return TrainedInputStructure(prop, predicate, source_ref, (1,), members)


def _project(*structures, values=(41, 43, 42), operators=()):
    """真实整数投影器产生关系、开放框架和成员引用。"""
    frames = tuple(OpenRoleFrame(
        item.proposition_key, item.predicate_key, item.source_ref,
        tuple(role.stable_key() for role in ROLES), ((), (43,), ()), (0, 3))
        for item in structures)
    return TrainedInputStructureProjector(
        tuple(structures), open_role_frames=frames,
        discourse_operators=operators).project(values)


def _members(structure, relation):
    """取一条保留关系的全部精确归属成员。"""
    return tuple(item for item in structure.semantic_candidates
        if (item.proposition_key, item.predicate_key, item.source_ref)
        == (relation.proposition_key, relation.predicate_key, relation.source_ref)
        and item.span_ref in relation.span_refs)


def test_physical_duplicates_share_one_complete_relation_and_open_frame():
    projected = _project(_structure(101), _structure(102), _structure(103))
    aggregated = aggregate_input_provenance_duplicates(projected)
    assert len(aggregated.relation_candidates) == 1
    relation, = aggregated.relation_candidates
    assert len(_members(aggregated, relation)) == 3
    frame, = aggregated.open_relations
    assert frame.frame.proposition == relation.proposition_key
    assert frame.frame.source_ref == relation.source_ref
    assert aggregate_input_provenance_duplicates(aggregated) == aggregated


def test_distinct_endpoint_keeps_both_complete_relations_and_open_frames():
    projected = _project(_structure(101, 11), _structure(102, 14), _structure(103, 14))
    aggregated = aggregate_input_provenance_duplicates(projected)
    assert len(aggregated.relation_candidates) == 2
    assert len(aggregated.open_relations) == 2
    assert len(aggregated.semantic_candidates) == 6
    assert all(len(_members(aggregated, relation)) == 3
               for relation in aggregated.relation_candidates)
    assert len(relation_candidate_semantic_groups(aggregated)) == 2
    assert aggregate_input_provenance_duplicates(aggregated) == aggregated

    # 真实范围过滤器不能再因开放框架只剩物理代表而删除另一声明。
    bridge = object.__new__(TrainedGraphQueryBridge)
    bridge._relation_question_open_frames = (projected.open_relations[0].frame,)
    bridge.pending_relation_compositions = ()
    bridge.relation_question_frames = ()
    bridge.input_discourse_conjunctions = ()
    bridge.core_fact_index = {}
    for relation in aggregated.relation_candidates:
        members = _members(aggregated, relation)
        bindings = tuple(SimpleNamespace(
            role=ObjectIdentity.from_stable_key(item.role_key),
            filler=ObjectIdentity.from_stable_key(item.candidate_key))
            for item in members if item.projection_kind == PROJECTION_FILLER)
        bridge.core_fact_index[relation.proposition_key] = SimpleNamespace(
            source_hash=relation.source_ref[0], bindings=bindings)
    filtered = bridge._filter_relation_question_open_scope(aggregated)
    assert len(filtered.relation_candidates) == 2
    assert len(filtered.semantic_candidates) == 6


def test_same_surface_positive_tie_keeps_identity_rows_but_is_answer_equivalent():
    """同一输入拓扑的正来源可闭合，图身份仍保持分离。"""
    same_surface = _project(_structure(101, 11), _structure(102, 11))
    assert len(same_surface.relation_candidates) == 2
    assert _surface_equivalent_relation_tie(
        same_surface, same_surface.relation_candidates)

    distinct_surface = replace(
        same_surface,
        relation_candidates=tuple(sorted((
            same_surface.relation_candidates[0],
            replace(
                same_surface.relation_candidates[1],
                structure_key=(999, *same_surface.relation_candidates[1].structure_key),
            ),
        ), key=lambda item: item.stable_key())),
    )
    assert len(distinct_surface.relation_candidates) == 2
    assert not _surface_equivalent_relation_tie(
        distinct_surface, distinct_surface.relation_candidates)


def test_composition_and_question_references_follow_retained_relation():
    projected = _project(_structure(101), _structure(102),
        values=(41, 43, 42, 99),
        operators=(TrainedInputDiscourseOperator((900, 1), (99,)),))
    relation = projected.relation_candidates[-1]
    slots = tuple(item.span_ref for item in _members(projected, relation)
                  if item.projection_kind == PROJECTION_FILLER)
    binding = InputRelationQuestionBinding((901, 1), (900, 1),
                                           relation.stable_key(), slots)
    projected = replace(projected, relation_question_bindings=(binding,))
    aggregated = aggregate_input_provenance_duplicates(projected)
    keys = {item.stable_key() for item in aggregated.relation_candidates}
    assert len(keys) == 1
    assert aggregated.relation_question_bindings[0].relation_key in keys
    assert all(key in keys for composition in aggregated.discourse_compositions
               for group in composition.operand_relation_keys for key in group)
    assert aggregate_input_provenance_duplicates(aggregated) == aggregated


def test_bound_request_keeps_every_exact_reader_projection():
    """共享端点的多来源 reader 在关系去重后仍须完整且可重新核验。"""
    projected = _project(_structure(101), _structure(102))
    shared = next(item.candidate_key for item in projected.semantic_candidates
                  if item.member_ordinal == 0)
    projected = replace(projected, semantic_candidates=tuple(sorted((
        replace(item, candidate_key=shared) if item.member_ordinal == 0 else item
        for item in projected.semantic_candidates), key=lambda item: item.stable_key())))
    digest = tuple(range(32))
    source_id = int.from_bytes(bytes(digest[:8]), 'big') or 1
    frame = OperationRequestFrame(
        (902, 1), SourceRef(FRAME, source_id, 0, OwnerScope(), VersionBundle()),
        digest, (903, 1), (904, 1),
        ((1, PREDICATE.stable_key(), ROLES[0].stable_key(),
          (ObjectIdentity.from_stable_key(shared).object_kind,)),),
        ((), (43, 42)))
    attached = attach_operation_request((frame,), projected)
    binding, = attached.operation_request_bindings
    assert binding.readings[0][1] == shared
    assert len(binding.readings[0][3]) == 2
    aggregated = aggregate_input_provenance_duplicates(attached)
    assert len(aggregated.relation_candidates) == 1
    assert aggregated.operation_request_bindings == (binding,)
    projections = {item.stable_key() for item in aggregated.semantic_candidates}
    assert set(binding.readings[0][3]) <= projections
    assert attach_operation_request((frame,), aggregated) == aggregated
    assert aggregate_input_provenance_duplicates(aggregated) == aggregated


@pytest.mark.parametrize('change', ('role', 'span', 'state', 'structure'))
def test_role_span_state_and_required_topology_are_not_physical_duplicates(change):
    projected = _project(_structure(101), _structure(102))
    candidate = projected.relation_candidates[-1]
    if change == 'role':
        semantic = tuple(sorted((
            replace(item, role_key=ROLES[1].stable_key())
            if item.proposition_key == candidate.proposition_key and item.member_ordinal == 0
            else item for item in projected.semantic_candidates), key=lambda item: item.stable_key()))
        projected = replace(projected, semantic_candidates=semantic)
    else:
        if change == 'span':
            changed = replace(candidate, span_refs=candidate.span_refs[:-1])
        elif change == 'state':
            changed = replace(candidate, support_count=2, required_open=1, state=1)
        else:
            changed = replace(candidate, structure_key=(99, *candidate.structure_key))
        projected = replace(projected, relation_candidates=tuple(sorted((
            changed if item == candidate else item for item in projected.relation_candidates),
            key=lambda item: item.stable_key())))
    assert len(aggregate_input_provenance_duplicates(projected).relation_candidates) == 2


def test_identity_normalization_preserves_owner_versions_kind_and_full_declaration():
    baseline = entity_identity(_source(101), (10, 11))
    duplicate = entity_identity(_source(102), (10, 11))
    assert semantic_candidate_identity_key(baseline.stable_key()) == semantic_candidate_identity_key(duplicate.stable_key())
    changed_sources = (
        _source(102, owner=OwnerScope(1, 2, 3, 4)),
        _source(102, versions=replace(VersionBundle(), corpus=type(baseline.versions.corpus)(2))),
    )
    different = [entity_identity(source, (10, 11)) for source in changed_sources]
    different += [entity_identity(_source(102), (10, 14)),
                  entity_identity(_source(102), (11, 11)),
                  set_expr_identity(_source(102), (10, 11))]
    assert all(semantic_candidate_identity_key(item.stable_key())
               != semantic_candidate_identity_key(baseline.stable_key()) for item in different)
    concept = replace(baseline, object_kind=4)
    assert semantic_candidate_identity_key(concept.stable_key()) == (0, *concept.stable_key())


def test_verified_source_proof_merges_duplicate_relation_rows():
    """来源证明进入关系键后，重复物化只保留一个消费候选。"""
    projected = _project(_structure(101, 11), _structure(102, 14))
    with_proof = replace(
        projected,
        source_provenance_keys=(((101,), (1, 900)), ((102,), (1, 900))),
    )
    aggregated = aggregate_input_provenance_duplicates(with_proof)
    assert len(aggregated.relation_candidates) == 1
    assert len(aggregated.open_relations) == 1
    # The full graph rows remain available as evidence members under the
    # retained physical representative.
    assert len(aggregated.semantic_candidates) == 3
    assert aggregate_input_provenance_duplicates(aggregated) == aggregated


def test_proof_aware_relation_key_keeps_structural_competitors_separate():
    """来源证明不能 erase role, span, state, or structure conflicts."""
    projected = _project(_structure(101), _structure(102))
    proof = (((101,), (1, 901)), ((102,), (1, 901)))
    candidate = projected.relation_candidates[-1]
    variants = (
        replace(
            projected,
            source_provenance_keys=proof,
            relation_candidates=tuple(sorted((
                projected.relation_candidates[0],
                replace(candidate, span_refs=candidate.span_refs[:-1]),
            ), key=lambda item: item.stable_key())),
        ),
        replace(
            projected,
            source_provenance_keys=proof,
            relation_candidates=tuple(sorted((
                projected.relation_candidates[0],
                replace(candidate, support_count=2, required_open=1, state=1),
            ), key=lambda item: item.stable_key())),
        ),
        replace(
            projected,
            source_provenance_keys=proof,
            relation_candidates=tuple(sorted((
                projected.relation_candidates[0],
                replace(candidate, structure_key=(999, *candidate.structure_key)),
            ), key=lambda item: item.stable_key())),
        ),
    )
    assert all(len(aggregate_input_provenance_duplicates(item).relation_candidates) == 2
               for item in variants)


def test_trained_similar_tie_accepts_only_authorized_reversed_roles():
    """SIMILAR 的 typed symmetric rule 允许反向 Role，但保留端点冲突。"""
    predicate = authored_relation_identity(RELATION_SIMILAR).stable_key()
    left = authored_relation_role_identity(ROLE_SIMILAR_LEFT).stable_key()
    right = authored_relation_role_identity(ROLE_SIMILAR_RIGHT).stable_key()

    def convert(structure, *, reverse):
        members = []
        filler_ordinal = 0
        for item in structure.semantic_candidates:
            role = item.role_key
            if item.projection_kind == PROJECTION_FILLER:
                pair = (right, left) if reverse else (left, right)
                role = pair[filler_ordinal]
                filler_ordinal += 1
            members.append(replace(item, predicate_key=predicate, role_key=role))
        # Exercise the real projector topology key: reversing the typed Roles
        # also reverses the role/ordinal pairing in ``structure_key``.  The
        # symmetric arbitration gate may normalize that endpoint order only.
        pair = (right, left) if reverse else (left, right)
        base_structure_key = structure.relation_candidates[0].structure_key
        cursor = 2
        filler_ordinal = 0
        converted_structure_key = [base_structure_key[0], base_structure_key[1]]
        for _ in range(base_structure_key[1]):
            ordinal, projection_kind, object_kind, role_size = (
                base_structure_key[cursor:cursor + 4])
            cursor += 4
            role = base_structure_key[cursor:cursor + role_size]
            cursor += role_size
            if projection_kind == PROJECTION_FILLER:
                role = pair[filler_ordinal]
                filler_ordinal += 1
            converted_structure_key.extend(
                (ordinal, projection_kind, object_kind, len(role), *role))
        converted_structure_key = tuple(converted_structure_key)
        relations = tuple(
            replace(item, predicate_key=predicate,
                    structure_key=converted_structure_key)
            for item in structure.relation_candidates)
        return replace(
            structure,
            semantic_candidates=tuple(sorted(
                members, key=lambda item: item.stable_key())),
            relation_candidates=tuple(sorted(
                relations, key=lambda item: item.stable_key())),
            open_relations=(),
        )

    forward = convert(_project(_structure(101)), reverse=False)
    reverse = convert(_project(_structure(102)), reverse=True)
    merged = replace(
        forward,
        semantic_candidates=tuple(sorted(
            (*forward.semantic_candidates, *reverse.semantic_candidates),
            key=lambda item: item.stable_key())),
        relation_candidates=tuple(sorted(
            (*forward.relation_candidates, *reverse.relation_candidates),
            key=lambda item: item.stable_key())),
    )
    assert _surface_equivalent_relation_tie(
        merged, merged.relation_candidates)

    changed = list(reverse.semantic_candidates)
    filler = next(index for index, item in enumerate(changed)
                  if item.projection_kind == PROJECTION_FILLER)
    changed_key = (*changed[filler].candidate_key[:-1],
                   changed[filler].candidate_key[-1] + 999)
    changed[filler] = replace(
        changed[filler], candidate_key=changed_key)
    changed_structure = replace(
        reverse,
        semantic_candidates=tuple(sorted(
            changed, key=lambda item: item.stable_key())))
    conflict = replace(
        forward,
        semantic_candidates=tuple(sorted(
            (*forward.semantic_candidates, *changed_structure.semantic_candidates),
            key=lambda item: item.stable_key())),
        relation_candidates=tuple(sorted(
            (*forward.relation_candidates, *changed_structure.relation_candidates),
            key=lambda item: item.stable_key())),
    )
    assert not _surface_equivalent_relation_tie(
        conflict, conflict.relation_candidates)
