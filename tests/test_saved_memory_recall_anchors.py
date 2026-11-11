"""当前结构召回锚点必须来自真实 Memory 声明，不能由历史 context 自报。"""
from types import SimpleNamespace

import pytest

from pure_integer_ai.cognition.shared.memory_event import HypothesisPayload
from pure_integer_ai.cognition.understanding.query_input_structure import QueryInputStructure
from pure_integer_ai.cognition.understanding.query_memory_candidates import (
    frame_key, memory_candidate_routes, memory_input_candidate_keys,
)
from pure_integer_ai.experiments.query_memory_ohe_audit import QueryMemoryOheEvidenceError
from pure_integer_ai.experiments.saved_memory_recall_anchors import read_saved_recall_observation_anchors
from pure_integer_ai.experiments.trained_dialogue_memory_graph import TrainedDialogueMemoryGraph


def _input(key):
    """使用纯结构输入建立候选，测试不需要语言或模型 fixture。"""
    return QueryInputStructure((1,), (), (1,), (), (), (), (), (), (), graph_object_keys=(key,))


@pytest.fixture
def recalled(tmp_path):
    """保存先前、当前及未来的同结构输入，保留全部声明与 manifest。"""
    with TrainedDialogueMemoryGraph(tmp_path / "anchors.sqlite3", session_id=31) as memory:
        prepared = _input((72, 1))
        prior = memory.append_graph_input(prepared.graph_object_keys)
        current = memory.append_graph_input(prepared.graph_object_keys)
        future = memory.append_graph_input(prepared.graph_object_keys)
        events = memory.intake.event_log
        def declarations(ref):
            """只读精确引用的原物理声明。"""
            return tuple(item for item in events.query(access=memory.candidate_index.access, object_ref=ref)
                         if item.event.is_declaration)
        candidates = memory_input_candidate_keys(prepared, session_id=31)
        candidate = next(key for key in candidates if memory_candidate_routes(key))
        shared = memory_candidate_routes(candidate)[0]
        def route(append):
            """从实际 intake 的候选声明恢复完整召回路由。"""
            intake = memory.intake.result_for_source(append.source)
            ref = next(ref for ref in intake.hypothesis_refs
                       if isinstance(declarations(ref)[0].event.payload, HypothesisPayload)
                       and declarations(ref)[0].event.payload.hypothesis.candidate_key == candidate)
            return frame_key(1, candidate, shared, ref.stable_key())
        current_ref = memory.intake.result_for_source(current.source).observation_ref
        def read(routes, **options):
            """锚点恢复只接收原 route 与独立当前候选。"""
            return read_saved_recall_observation_anchors(memory,
                SimpleNamespace(memory_routes=lambda: routes), input_observation=current_ref,
                input_candidate_keys=options.get("candidates", candidates),
                declaration_reader=options.get("declarations", declarations))
        yield prior, current, future, route, read


def test_only_actual_current_candidate_match_can_root_a_prior_observation(recalled):
    """历史距离可从已核验的当前内容命中重新起算，同一来源只保留一个锚点。"""
    prior, _current, _future, route, read = recalled
    rows = read((route(prior), route(prior)))
    assert len(rows) == 1 and rows[0].event.payload.source == prior.source
    assert read(()) == ()


def test_future_matching_observation_cannot_change_an_original_query(recalled):
    """相同结构出现在旧输入之后，也不得成为旧查询锚点。"""
    _prior, _current, future, route, read = recalled
    with pytest.raises(QueryMemoryOheEvidenceError, match="future"):
        read((route(future),))


def test_saved_route_cannot_authorize_its_own_input_candidate(recalled):
    """待审 route 不能绕过独立恢复的当前输入候选。"""
    prior, _current, _future, route, read = recalled
    with pytest.raises(QueryMemoryOheEvidenceError, match="current input"):
        read((route(prior),), candidates=())


def test_anchor_requires_the_original_unique_declaration(recalled):
    """索引或 trace 中存在引用不等于物理声明存在。"""
    prior, _current, _future, route, read = recalled
    with pytest.raises(QueryMemoryOheEvidenceError, match="unique scoped declaration"):
        read((route(prior),), declarations=lambda _ref: ())
