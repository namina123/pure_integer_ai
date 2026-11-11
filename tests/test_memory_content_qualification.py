"""内容召回须逐假设授权，来源相同不能跨授无路由的句法分支。"""
from pure_integer_ai.cognition.shared.identity import OBJECT_CONTEXT_SCOPE, ObjectIdentity
from pure_integer_ai.cognition.shared.memory_event import MemoryLinkedRef
from pure_integer_ai.cognition.understanding.memory_intake import HypothesisIntakeDraft, ObservationIntakeDraft
from pure_integer_ai.cognition.understanding.query_memory_candidates import frame_key
from pure_integer_ai.experiments.trained_dialogue_memory_graph import TrainedDialogueMemoryGraph


MATCHED=frame_key(91502,(11,),(72,1))
OTHER=frame_key(91502,(11,),(72,2))
UNROUTED=frame_key(91503,(6,),(91,3))


def _append(memory, ordinal):
    """三个真实来源内分支共享Observation，但只有一个获得当前内容路由。"""
    context=MemoryLinkedRef.object(ObjectIdentity(OBJECT_CONTEXT_SCOPE,
        (91501,memory.session_id,1),memory.owner,memory.candidate_index.source.versions))
    hypotheses=tuple(HypothesisIntakeDraft((91,ordinal,index),(91,1),candidate,
        (91,index),3,reason_key=(91,4)) for index,candidate in enumerate((MATCHED,OTHER,UNROUTED),1))
    result=memory.append('有来源输入'+str(ordinal),speaker_kind=1,observation_draft=ObservationIntakeDraft(
        (91,ordinal),context,hypotheses=hypotheses))
    return memory.intake.result_for_source(result.source).observation_ref


def _candidates(active, observation):
    """按完整Observation身份恢复候选集，不以数量代替语义成员。"""
    return {item.candidate_key for item in active.hypotheses
        if item.observation_key==observation.stable_key()}


def test_bounded_content_recall_keeps_actual_matches_and_complete_current_branches(tmp_path):
    """两个历史来源的真命中都保留；无路由兄弟只留在当前显式锚点。"""
    with TrainedDialogueMemoryGraph(tmp_path/'bounded.sqlite3',session_id=79) as memory:
        old=(_append(memory,1),_append(memory,2))
        current=_append(memory,3)
        before=memory.backend.count('memory_event')
        active=memory.active_structures(candidate_keys=(MATCHED,),observation_refs=(current,),context_depth_limit=0)
        for ref in old:
            assert _candidates(active,ref)=={MATCHED}
        assert _candidates(active,current)=={MATCHED,OTHER,UNROUTED}
        for hypothesis in active.hypotheses:
            assert set(hypothesis.evidence_keys)=={item.evidence_key for item in active.evidence
                if item.hypothesis_key==hypothesis.hypothesis_key}
        assert memory.backend.count('memory_event')==before
        explicit=memory.active_structures(candidate_keys=(MATCHED,),
            observation_refs=(*old,current),context_depth_limit=0)
        assert all(_candidates(explicit,ref)=={MATCHED,OTHER,UNROUTED} for ref in (*old,current))


def test_unbounded_diagnostic_retains_original_source_local_topology(tmp_path):
    """诊断仍保留无路由兄弟；没有删除权威历史或改变旧诊断边界。"""
    with TrainedDialogueMemoryGraph(tmp_path/'diagnostic.sqlite3',session_id=79) as memory:
        old=_append(memory,1)
        current=_append(memory,2)
        active=memory.active_structures(candidate_keys=(MATCHED,),observation_refs=(current,),context_depth_limit=None)
        assert _candidates(active,old)=={MATCHED,UNROUTED}
        assert _candidates(active,current)=={MATCHED,OTHER,UNROUTED}
