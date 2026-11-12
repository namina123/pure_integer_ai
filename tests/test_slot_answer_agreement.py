"""跨谓词WH只能在完整对象、全槽和训练语法一致时选择物理代表。"""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from pure_integer_ai.cognition.shared.identity import OwnerScope, SourceRef, VersionBundle
from pure_integer_ai.cognition.shared.semantic_object import entity_identity, proposition_identity
from pure_integer_ai.cognition.understanding.query_input_structure import (
    PROJECTION_FILLER, PROJECTION_PREDICATE, STRUCTURE_ORDER_CONFLICT, TrainedInputMember,
    TrainedInputStructure, TrainedInputStructureProjector,
)
from pure_integer_ai.cognition.understanding.query_open_roles import OpenRoleFrame
from pure_integer_ai.cognition.understanding.query_relation_questions import (
    TrainedRelationQuestionFrame, match_relation_slot_question_frames,
)
from pure_integer_ai.experiments.ph2_authored_relation_compile import (
    authored_relation_identity, authored_relation_role_identity,
)
from pure_integer_ai.experiments.ph2_authored_relation_schema import (
    RELATION_CAUSES, RELATION_SIMILAR, ROLE_CAUSE, ROLE_EFFECT, ROLE_SIMILAR_LEFT, ROLE_SIMILAR_RIGHT,
)
from pure_integer_ai.experiments.query_relation_arbitration import select_equivalent_slot_question_bindings
from pure_integer_ai.experiments.slot_answer_agreement import select_convergent_slot_answer
from pure_integer_ai.experiments.trained_graph_query_bridge import TrainedGraphQueryBridge


def _context():
    """两个不同来源/谓词的真实整数投影，共享完整对象而非同名文本。"""
    common = SourceRef(206,99,0,OwnerScope(),VersionBundle())
    known, answer = (entity_identity(common,(10,value)) for value in (10,11))
    structures, questions, opens, facts = [], [], [], {}
    for index, kind, role_ids in ((101,RELATION_CAUSES,(ROLE_CAUSE,ROLE_EFFECT)),
                                  (102,RELATION_SIMILAR,(ROLE_SIMILAR_LEFT,ROLE_SIMILAR_RIGHT))):
        source = SourceRef(206,index,0,OwnerScope(),VersionBundle())
        proposition, predicate = proposition_identity(source,(1,1)), authored_relation_identity(kind)
        roles = tuple(authored_relation_role_identity(value) for value in role_ids)
        prop, pred, ref = proposition.stable_key(), predicate.stable_key(), (index,)
        members = (
            TrainedInputMember((11,),known.stable_key(),known.object_kind,PROJECTION_FILLER,
                prop,pred,ref,0,1,0,roles[0].stable_key()),
            TrainedInputMember((33,),pred,predicate.object_kind,PROJECTION_PREDICATE,
                prop,pred,ref,1,2,1),
            TrainedInputMember((22,),answer.stable_key(),answer.object_kind,PROJECTION_FILLER,
                prop,pred,ref,2,3,2,roles[1].stable_key()))
        structures.append(TrainedInputStructure(prop,pred,ref,(7,),members))
        question = TrainedRelationQuestionFrame((92002,index),(92004,1),pred,
            (roles[0].stable_key(),),((88,33),()),roles[1].stable_key(),(88,))
        questions.append(question)
        opens.append(OpenRoleFrame(prop,pred,ref,question.roles,question.gaps,(0,3)))
        facts[prop] = SimpleNamespace(proposition=proposition,predicate=predicate,source_hash=index,
            bindings=tuple(SimpleNamespace(role=role,filler=filler)
                for role,filler in zip(roles,(known,answer))))
    structure = TrainedInputStructureProjector(tuple(structures),open_role_frames=tuple(opens)).project((88,33,11))
    bridge = object.__new__(TrainedGraphQueryBridge)
    bridge.relation_question_frames = tuple(questions)
    bridge._relation_question_open_frames = tuple(opens)
    bridge.pending_relation_compositions = ()
    bridge.input_discourse_conjunctions = ()
    bridge.core_fact_index = facts
    alternatives = match_relation_slot_question_frames(tuple(questions),structure,allow_exact_alternatives=True)
    assert len(alternatives)==2
    records = tuple((binding,bridge._relation_slot_question_parts(structure,binding)) for binding in alternatives)
    assert all(parts is not None and not parts[4] for _,parts in records)
    return bridge,structure,records


@pytest.mark.parametrize('higher_key', (False, True))
def test_complete_question_predicate_witness_precedes_local_predicate_span(higher_key):
    """两种身份次序都优先完整gap证明，不从关系种类或目标答案选代表。"""
    bridge, structure, records = _context()
    ordered = sorted(records, key=lambda row: row[0].stable_key())
    binding, parts = ordered[int(higher_key)]
    relation = parts[1]
    predicate_spans = set(relation.span_refs) - set(binding.slot_span_refs)
    assert len(predicate_spans) == 1
    original = replace(structure,
        semantic_candidates=tuple(item for item in structure.semantic_candidates
            if not (item.proposition_key == relation.proposition_key
                    and item.projection_kind == PROJECTION_PREDICATE)),
        relation_candidates=tuple(sorted((replace(item,
            span_refs=tuple(ref for ref in item.span_refs if ref not in predicate_spans),
            predicate_coverage=0, support_count=item.role_coverage,
            required_open=item.required_count-item.role_coverage)
            if item.stable_key() == relation.stable_key() else item
            for item in structure.relation_candidates), key=lambda item: item.stable_key())))
    promoted = bridge._close_relation_question_predicate_scope(original)
    alternatives = match_relation_slot_question_frames(bridge.relation_question_frames,
        promoted, allow_exact_alternatives=True)
    validated = tuple((item, bridge._relation_slot_question_parts(promoted, item))
                      for item in alternatives)
    assert len(validated) == 2 and all(row[1] is not None for row in validated)
    expected, = (item for item in alternatives if item.frame_key == binding.frame_key)
    assert select_convergent_slot_answer(validated) == (expected,)
    assert select_convergent_slot_answer(tuple(reversed(validated))) == (expected,)
    assert bridge._attach_relation_question_binding(original).relation_question_bindings == (expected,)


def test_different_relations_can_answer_one_object_without_becoming_equivalent():
    bridge,structure,records = _context()
    expected = min((binding for binding,_ in records),key=lambda item:item.stable_key())
    before = tuple(structure.relation_candidates),tuple(bridge.core_fact_index)
    assert select_convergent_slot_answer(records)==(expected,)
    assert select_equivalent_slot_question_bindings(records)==(expected,)
    result = bridge._attach_relation_question_binding(structure)
    assert result.relation_question_bindings==(expected,)
    assert (tuple(structure.relation_candidates),tuple(bridge.core_fact_index))==before
    assert len({parts[0].predicate_key for _,parts in records})==2


@pytest.mark.parametrize('drift', (
    'answer','known','operation','syntax','span','hidden','reference','dynamic',
    'state','source','binding','predicate','same_predicate',
))
def test_agreement_rejects_full_identity_scope_source_and_role_drift(drift):
    _,_,records = _context()
    records = list(records)
    binding,parts = records[1]
    frame,relation,original,answer,dynamic = parts
    fact = SimpleNamespace(**vars(original))
    if drift in ('answer','known'):
        index = 1 if drift=='answer' else 0
        row = fact.bindings[index]
        # 同一可见整数词形也不能替代另一来源的完整对象。
        alternate = entity_identity(SourceRef(206,999,0,OwnerScope(),VersionBundle()),(10,11 if index else 10))
        values = list(fact.bindings)
        values[index] = SimpleNamespace(role=row.role,filler=alternate)
        fact.bindings = tuple(values)
        if index: answer=alternate.stable_key()
    elif drift=='operation':
        frame=replace(frame,operation_key=(92004,9))
        binding=replace(binding,frame_key=frame.stable_key(),operation_key=frame.operation_key)
    elif drift=='syntax':
        frame=replace(frame,gaps=((88,34),()))
        binding=replace(binding,frame_key=frame.stable_key())
    elif drift=='span':
        binding=replace(binding,slot_span_refs=((777,),))
    elif drift=='hidden':
        fact.bindings=(*fact.bindings,SimpleNamespace(
            role=authored_relation_role_identity(ROLE_CAUSE),filler=fact.bindings[0].filler))
    elif drift=='reference':
        frame=replace(frame,slot_mode=1)
        binding=replace(binding,frame_key=frame.stable_key())
    elif drift=='dynamic':
        dynamic=frame.roles
    elif drift=='state':
        relation=replace(relation,conflict_count=1,state=STRUCTURE_ORDER_CONFLICT)
        binding=replace(binding,relation_key=relation.stable_key())
    elif drift=='source':
        fact.source_hash+=1
    elif drift=='binding':
        binding=replace(binding,frame_key=(999,))
    elif drift=='predicate':
        fact.predicate=authored_relation_identity(RELATION_CAUSES)
    else:
        frame=replace(frame,predicate_key=records[0][1][0].predicate_key)
    records[1]=(binding,(frame,relation,fact,answer,dynamic))
    assert select_convergent_slot_answer(records)==()


def test_incomplete_validation_never_creates_a_question_binding():
    _,_,records = _context()
    assert select_convergent_slot_answer(())==()
    assert select_convergent_slot_answer(records[:1])==()
    assert select_convergent_slot_answer(((records[0][0],None),records[1]))==()
