"""桥接补齐交付来源不能重新授权原输入内未命中的全部分支。"""
from types import SimpleNamespace
import sys
from pathlib import Path
from dataclasses import replace
import pytest

ROOT=Path(__file__).resolve().parents[1]
from tests.test_memory_content_qualification import _append,_candidates,MATCHED,OTHER,UNROUTED
from tests.test_memory_delivered_recall import _output
from pure_integer_ai.cognition.shared.identity import SourceRef
from pure_integer_ai.cognition.shared.memory_event import INTAKE_DERIVED_HYPOTHESIS
from pure_integer_ai.cognition.shared.query_state import BindingEntry,QueryRoot,ROOT_EXPANDED,SPACE_MEMORY
from pure_integer_ai.cognition.shared.memory_record_identity import observation_reference_source
from pure_integer_ai.cognition.understanding.query_open_roles import unpack_record
from pure_integer_ai.experiments.trained_dialogue_memory_graph import TrainedDialogueMemoryGraph
from pure_integer_ai.experiments.trained_graph_query_bridge import (
    _root_response_context_inputs,_validate_response_context_memory_closure,
)
from pure_integer_ai.experiments.trained_response_context import DeliveredGenericResponseContext


def _context(memory,input_ref,output_ref):
    """从本次实际输出唯一Episode及物理声明恢复来源context，完全不读表面。"""
    episode,=memory.response_episodes_for(output_ref)
    proof=episode.selected_path_ref.value()
    source=observation_reference_source(output_ref)
    manifest=memory.intake.require_current_manifest(source)
    hypothesis,=(item.object_ref for item in manifest.bindings if item.binding_kind==INTAKE_DERIVED_HYPOTHESIS)
    active=memory.active_structures(candidate_keys=(),observation_refs=(output_ref,),context_depth_limit=0)
    observed,=active.observations
    h,=active.hypotheses
    return DeliveredGenericResponseContext(output_ref,hypothesis,observed.source_ref,observed.scope_key,
        proof,input_ref,episode.intent_ref.value(),h.evidence_keys,0)


def test_bridge_context_input_keeps_content_matches_without_reopening_siblings(tmp_path):
    """必须经过真实桥接helper，不能只检查下层active_structures。"""
    with TrainedDialogueMemoryGraph(tmp_path/'bridge.sqlite3',session_id=82) as memory:
        old=_append(memory,1)
        output,_=_output(memory,SimpleNamespace(source=observation_reference_source(old)),1)
        _output(memory,memory.append_graph_input(((72,2),)),2)
        current=_append(memory,3)
        active=memory.active_structures(candidate_keys=(MATCHED,),observation_refs=(current,),
            context_depth_limit=1,recall_delivered_outputs=True)
        context=_context(memory,old,output)
        rooted=_root_response_context_inputs(memory,active,(context,))
        _validate_response_context_memory_closure(rooted,(context,))
        assert _candidates(rooted,old)=={MATCHED}
        assert _candidates(rooted,current)=={MATCHED,OTHER,UNROUTED}


def test_context_only_input_adds_physical_source_without_new_semantic_hypotheses(tmp_path):
    """没有内容路由的原输入只证明输出来源，不会成为新的猜测或答案来源。"""
    with TrainedDialogueMemoryGraph(tmp_path/'source.sqlite3',session_id=83) as memory:
        old=_append(memory,1)
        output,_=_output(memory,SimpleNamespace(source=observation_reference_source(old)),1)
        context=_context(memory,old,output)
        active=memory.active_structures(candidate_keys=(),observation_refs=(output,),context_depth_limit=0)
        rooted=_root_response_context_inputs(memory,active,(context,))
        _validate_response_context_memory_closure(rooted,(context,))
        assert old.stable_key() in {item.observation_key for item in rooted.observations}
        assert _candidates(rooted,old)==set()
        assert rooted.hypotheses==active.hypotheses
        assert rooted.evidence==active.evidence


@pytest.mark.parametrize('consumed',('none','input','output'))
def test_roleless_context_source_qualification_follows_actual_physical_observation_reads(tmp_path,consumed):
    """真实无角色交付的候选不自授权；输入O或输出O被读取都要求完整来源核验。"""
    from pure_integer_ai.experiments.memory_provenance_inputs import consumed_response_contexts
    with TrainedDialogueMemoryGraph(tmp_path/'roleless.sqlite3',session_id=85) as memory:
        inp=_append(memory,1)
        out,_=_output(memory,SimpleNamespace(source=observation_reference_source(inp)),1)
        context=_context(memory,inp,out)
        assert not context.role_occurrences
        read_ref={'input':inp,'output':out}.get(consumed)
        hops=() if read_ref is None else (SimpleNamespace(
            space=SPACE_MEMORY,predicate_key=(1,),proposition_key=read_ref.stable_key()),)
        selected=consumed_response_contexts((context,),SimpleNamespace(hops=lambda:hops),set())
        assert selected==(() if consumed=='none' else (context,))


@pytest.mark.parametrize('mutation',('valid','missing_root','foreign_source','missing_binding'))
def test_cold_source_observation_requires_exact_physical_root_and_binding(tmp_path,mutation):
    """冷审来源O不能只相信context或Plan，缺根/错来源/缺绑定都必须拒绝。"""
    from pure_integer_ai.experiments.memory_provenance_inputs import verify_context_input_observations
    with TrainedDialogueMemoryGraph(tmp_path/'cold.sqlite3',session_id=84) as memory:
        inp=_append(memory,1)
        out,_=_output(memory,SimpleNamespace(source=observation_reference_source(inp)),1)
        context=_context(memory,inp,out)
        source=observation_reference_source(inp)
        active=memory.active_structures(candidate_keys=(),observation_refs=(inp,),context_depth_limit=0)
        observed,=active.observations
        root=QueryRoot(SPACE_MEMORY,inp.stable_key(),status=ROOT_EXPANDED,priority_weight=1,
            source_ref=observed.source_ref,scope_key=observed.scope_key)
        roots=[] if mutation=='missing_root' else [root]
        if mutation=='foreign_source':
            roots=[replace(root,source_ref=(observed.source_hash+1,*source.stable_key()))]
        binding=BindingEntry((1,),inp.stable_key(),SPACE_MEMORY,observed.scope_key).stable_key()
        bindings=set() if mutation=='missing_binding' else {binding}
        def declaration(ref):
            """只读原会话完整声明，测试不构造替代来源。"""
            rows=tuple(item for item in memory.intake.event_log.query(
                access=memory.candidate_index.access,object_ref=ref) if item.event.is_declaration)
            assert len(rows)==1
            return rows[0]
        def verify():
            """调用现役冷审核，所有变体只改变待审查询证明。"""
            return verify_context_input_observations((context,),roots=roots,bindings=bindings,
                declaration=declaration,observation=lambda value:memory.intake.result_for_source(value).observation_ref,
                source_hash=lambda value:observed.source_hash if value==source else 0,error=ValueError)
        if mutation=='valid':
            assert verify()=={(inp.stable_key(),(1,),observed.source_hash)}
        else:
            with pytest.raises(ValueError,match='exact current Observation root/binding'):
                verify()
