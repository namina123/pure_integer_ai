"""实际交付反向边恢复来源，不能重新扩大内容命中输入的兄弟假设。"""
from pathlib import Path
from types import SimpleNamespace
import sys

from tests.test_memory_content_qualification import _append, _candidates, MATCHED
from tests.test_memory_delivered_recall import _output
from pure_integer_ai.cognition.shared.memory_record_identity import observation_reference_source
from pure_integer_ai.experiments.trained_dialogue_memory_graph import TrainedDialogueMemoryGraph


def test_actual_output_input_edge_does_not_reauthorize_unmatched_input_siblings(tmp_path):
    """真实proof/Episode恢复较早输出后，原输入仍只有实际当前内容命中的H/E。"""
    with TrainedDialogueMemoryGraph(tmp_path/'reverse.sqlite3',session_id=79) as memory:
        old=_append(memory,1)
        actual,_=_output(memory,SimpleNamespace(source=observation_reference_source(old)),1)
        noise=memory.append_graph_input(((72,2),))
        _output(memory,noise,2)
        current=_append(memory,3)
        active=memory.active_structures(candidate_keys=(MATCHED,),observation_refs=(current,),
            context_depth_limit=1,recall_delivered_outputs=True)
        assert actual.stable_key() in {item.observation_key for item in active.observations}
        assert _candidates(active,old)=={MATCHED}
