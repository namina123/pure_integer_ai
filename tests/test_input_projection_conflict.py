"""真实整数投影竞争只能表达未知，不得清除竞争或读入事实。"""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from pure_integer_ai.cognition.shared.query_state import (
    BindingEntry, EvidenceEntry, ROOT_EMPTY, ROOT_EXPANDED, SPACE_CORE,
    TERMINATION_CLARIFY_CONFLICT,
)
from pure_integer_ai.cognition.shared.query_driver import ExpansionOutcome
from pure_integer_ai.cognition.understanding.query_input_structure import PROJECTION_FILLER
from pure_integer_ai.cognition.understanding.query_memory_candidates import frame_key
from pure_integer_ai.experiments.input_projection_conflict import (
    conflicting_input_filler_groups, projection_conflict_keys_match,
    retained_projection_conflict_allowed,
)
from pure_integer_ai.experiments.trained_input_clarification import (
    clarification_failure_termination, clarification_query_state, run_input_clarification,
)
from pure_integer_ai.experiments.input_conflict_clarification_audit import verify_input_conflict_clarification
from tests.test_query_relation_coverage import _projector
from tests.test_trained_input_clarification import _query


def _fixture():
    """同一训练角色出现两个不同已知filler，保留完整原候选和UNKNOWN证据。"""
    structure = _projector().project((11, 77, 22))
    original = structure.semantic_candidates[0]
    competing = replace(original, candidate_key=(4, 999))
    structure = replace(structure, semantic_candidates=tuple(sorted(
        (*structure.semantic_candidates, competing), key=lambda x:x.stable_key())))
    state, frontier, hypotheses, nodes = _query()
    state = state.with_(query_key=structure.source_ref)
    fillers = tuple(x for x in structure.semantic_candidates if x.projection_kind == PROJECTION_FILLER)
    groups = conflicting_input_filler_groups(fillers)
    bindings = tuple(BindingEntry(x.role_key,x.candidate_key,space=SPACE_CORE,
        scope_key=state.query_key,conflict_kept=int((x.proposition_key,x.source_ref,x.role_key) in groups)) for x in fillers)
    evidence = tuple(EvidenceEntry(x.source_ref,x.proposition_key,space=SPACE_CORE,polarity=3,
        evidence_key=frame_key(91569,(kind,),x.stable_key()),scope_key=state.query_key,payload_key=x.stable_key())
        for kind,members in ((1,fillers),(2,structure.relation_candidates)) for x in members)
    return structure,state.with_(bindings=bindings,evidence=evidence),frontier,hypotheses,nodes


def test_source_exact_competition_retains_every_original_binding_and_unknown_evidence():
    structure,state,frontier,hypotheses,nodes = _fixture()
    before = state.stable_key()
    assert retained_projection_conflict_allowed(state,structure)
    admitted = clarification_query_state(state,frontier,hypotheses,nodes,input_structure=structure)
    assert admitted is not None
    assert set(state.bindings) <= set(admitted.bindings) and state.evidence == admitted.evidence
    assert state.stable_key() == before and admitted.budget == state.budget
    assert sum(x.conflict_kept for x in admitted.bindings) == 2
    assert retained_projection_conflict_allowed(admitted,structure,restricted=True)
    assert not retained_projection_conflict_allowed(state,structure,restricted=True)


@pytest.mark.parametrize('drift',('source','query','binding_extra','binding_missing','binding_clear',
    'evidence_extra','evidence_missing','support','refute','evidence_scope','payload','role','bound_request'))
def test_competition_contract_rejects_all_member_source_truth_and_scope_drift(drift):
    structure,state,_,_,_ = _fixture()
    if drift == 'source':structure = replace(structure,semantic_candidates=tuple(
        replace(x,source_ref=(999,)) for x in structure.semantic_candidates))
    elif drift == 'query':state=state.with_(query_key=(999,))
    elif drift == 'binding_extra':state=state.with_(bindings=(*state.bindings,BindingEntry((2,),(3,),conflict_kept=1)))
    elif drift == 'binding_missing':state=state.with_(bindings=state.bindings[1:])
    elif drift == 'binding_clear':state=state.with_(bindings=tuple(replace(x,conflict_kept=0) for x in state.bindings))
    elif drift == 'evidence_extra':state=state.with_(evidence=(*state.evidence,EvidenceEntry((1,),(2,),space=1,polarity=3)))
    elif drift == 'evidence_missing':state=state.with_(evidence=state.evidence[1:])
    elif drift in ('support','refute','evidence_scope','payload'):
        edits={'support':{'polarity':1},'refute':{'polarity':2},'evidence_scope':{'scope_key':(999,)},'payload':{'payload_key':(999,)}}
        state=state.with_(evidence=(replace(state.evidence[0],**edits[drift]),*state.evidence[1:]))
    elif drift == 'role':state=state.with_(bindings=(replace(state.bindings[0],role_key=(999,)),*state.bindings[1:]))
    else:structure=SimpleNamespace(**{field:getattr(structure,field) for field in (
        'source_ref','relation_question_bindings','operation_request_bindings','assertion_scope_bindings',
        'discourse_compositions','discourse_gaps','pending_relation_composition_bindings')},operation_keys=((999,),))
    assert not retained_projection_conflict_allowed(state,structure)


def test_cold_conflict_audit_accepts_exact_retained_projection_and_rejects_extra_binding():
    structure,state,frontier,hypotheses,nodes = _fixture()
    admitted=clarification_query_state(state,frontier,hypotheses,nodes,input_structure=structure)
    roots=tuple(replace(x,status=ROOT_EXPANDED) if x.status!=ROOT_EMPTY else x for x in admitted.roots)
    scalars=[0]*17
    scalars[8],scalars[15],scalars[16]=1,1,2
    value={'state_key':{'scalars':scalars},'termination':2,'query_ref':1,'input_structure':structure.integer_trace()}
    trace=SimpleNamespace(trace=value,resolve_ref=lambda ref:structure.source_ref,hops=lambda:())
    original=deepcopy(value)
    keys={x.stable_key() for x in admitted.bindings}
    result=verify_input_conflict_clarification(trace,structure,{},(),roots=roots,evidence=state.evidence,bindings=keys,error=ValueError)
    assert result['status'].startswith('MATCHED_RETAINED_INPUT_PROJECTION_CONFLICT') and trace.trace==original
    keys.add(BindingEntry((1,),(2,),conflict_kept=1).stable_key())
    with pytest.raises(ValueError):
        verify_input_conflict_clarification(trace,structure,{},(),roots=roots,evidence=state.evidence,bindings=keys,error=ValueError)


@pytest.mark.parametrize('generated',(False,True))
def test_shared_clarification_run_retains_conflicts_and_never_expands_core(generated):
    """真正query_driver只展开准入根，最终竞争及全部原证据始终保留。"""
    structure,state,frontier,hypotheses,nodes = _fixture()
    generic=SimpleNamespace(response_plan=SimpleNamespace(claim_refs=(),claim_evidence_refs=()),graph_surfaces=())
    hops=[]
    expander=SimpleNamespace(input_structure=structure,hops=hops,generic_response_generations=())

    def expand(current,edge):
        """只推进当前观察与connector，保留全部竞争而非伪造闭合。"""
        assert edge.owner_space != SPACE_CORE
        hops.append(SimpleNamespace(space=edge.owner_space))
        if generated and edge.owner_space==3:expander.generic_response_generations=(generic,)
        updated=current.with_(roots=tuple(replace(x,status=ROOT_EXPANDED)
            if (x.owner_space,x.root_key)==(edge.owner_space,edge.root_key) else x for x in current.roots),
            frontier=tuple(x for x in current.frontier if x!=edge),
            read_count=current.read_count+1,
            node_count=current.node_count+1,conflict_open=1)
        return ExpansionOutcome(updated,edge,1)

    expander.expand=expand
    result,response=run_input_clarification(state,frontier,hypotheses,nodes,expander=expander,
        tracer=SimpleNamespace(record=lambda *_:None))
    assert result.termination==TERMINATION_CLARIFY_CONFLICT and result.conflict_open==1
    assert not result.evidence_closed and result.required_slots_open>0
    assert state.evidence==result.evidence and set(state.bindings)<=set(result.bindings)
    assert response is (generic if generated else None) and len(hops)==2


@pytest.mark.parametrize('drift',('none','binding_missing','scope','evidence_missing','support','payload','relation_state'))
def test_original_relation_order_projection_is_reconstructed_without_clearing_conflict(drift):
    """关系次序竞争使用原predicate→proposition绑定，缺任一原证明即关闭。"""
    structure=_projector().project((22,33,11))
    relation,=structure.relation_candidates
    assert relation.state==3 and relation.conflict_count==1
    base,frontier,hypotheses,nodes=_query()
    source=structure.source_ref
    binding=BindingEntry(relation.predicate_key,relation.proposition_key,space=SPACE_CORE,scope_key=source,conflict_kept=1)
    evidence=tuple(EvidenceEntry(x.source_ref,x.proposition_key,space=1,polarity=3,
        evidence_key=frame_key(91569,(kind,),x.stable_key()),scope_key=source,payload_key=x.stable_key())
        for kind,members in ((1,tuple(x for x in structure.semantic_candidates if x.projection_kind==PROJECTION_FILLER)),
                             (2,structure.relation_candidates)) for x in members)
    state=base.with_(query_key=source,bindings=(binding,),evidence=evidence)
    if drift=='binding_missing':state=state.with_(bindings=())
    elif drift=='scope':state=state.with_(bindings=(replace(binding,scope_key=(999,)),))
    elif drift=='evidence_missing':state=state.with_(evidence=evidence[:-1])
    elif drift=='support':state=state.with_(evidence=(*evidence[:-1],replace(evidence[-1],polarity=1)))
    elif drift=='payload':state=state.with_(evidence=(*evidence[:-1],replace(evidence[-1],payload_key=(999,))))
    elif drift=='relation_state':structure=replace(structure,relation_candidates=(replace(relation,state=2,conflict_count=0),))
    assert retained_projection_conflict_allowed(state,structure)==(drift=='none')
    admitted=clarification_query_state(state,frontier,hypotheses,nodes,input_structure=structure)
    if drift=='none':
        assert admitted is not None and binding in admitted.bindings
        assert admitted.evidence==evidence and admitted.budget==state.budget


@pytest.mark.parametrize(('termination','expected'),((0,3),(1,3),(2,2),(3,3),(4,3),(5,5),(6,6),(7,3),(8,3)))
def test_outer_unbound_response_preserves_conflict_and_resource_failure(termination,expected):
    """内层真实冲突不能因外层缺绑定入口被抹去，原状态不得改写。"""
    _,state,_,_,_=_fixture()
    state=state.with_(termination=termination)
    before=state.stable_key()
    assert clarification_failure_termination(state)==expected
    assert state.stable_key()==before
