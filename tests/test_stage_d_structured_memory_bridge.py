"""Stage D：结构化 Memory O/H/E 与统一三图库查询纵切。"""
from __future__ import annotations

import json

from pure_integer_ai.cognition.shared.hypothesis import EvidenceRecord
from pure_integer_ai.cognition.shared.memory_event import (
    MEMORY_EVENT_EVIDENCE,
    MEMORY_EVENT_HYPOTHESIS,
    MEMORY_EVENT_OBSERVATION,
)
from pure_integer_ai.cognition.shared.query_state import (
    ROOT_EMPTY,
    ROOT_EXPANDED,
    SPACE_CORE,
    SPACE_DIALOGUE,
    SPACE_MEMORY,
    TERMINATION_ANSWER_CLOSED,
)
from pure_integer_ai.cognition.understanding.query_memory_candidates import (
    MEMORY_CANDIDATE_VERSION,
    MEMORY_CATEGORY_OPEN_RELATION,
    MemoryTopicCandidate,
    frame_key,
    memory_candidate_routes,
    memory_candidate_category,
    memory_topic_candidate,
)
from pure_integer_ai.cognition.understanding.query_open_roles import OPEN_SCHEMA_VERSION, OpenRoleFrame
from pure_integer_ai.experiments.trained_dialogue_memory_graph import (
    TrainedDialogueMemoryGraph,
    memory_input_candidate_keys,
)
from pure_integer_ai.experiments.trained_memory_candidate_index import (
    CANDIDATE_INDEX_TABLE,
)
from pure_integer_ai.experiments.trained_graph_query_bridge import (
    TrainedGraphQueryBridge,
)

from test_stage_b_core_query_bridge import _trained_database


def exercise_topic_switch(bridge, *, capture):
    """在实际模型 owner 上运行双话题及返回原话题，所有回答仍走生产 bridge。"""
    first = "车轮是汽车的一部分"
    second = "启明星就是晨星"
    projected = [bridge.input_projector.project(tuple(map(ord, text)))
                 for text in (first, second)]
    candidates = [memory_input_candidate_keys(item) for item in projected]
    legacy_topics = [{frame_key(MEMORY_CANDIDATE_VERSION, (7,),
                               row.structure_key, row.carrier_key)
                      for row in item.relation_candidates} for item in projected]
    assert legacy_topics[0] & legacy_topics[1]
    routes = [{route for key in keys for route in memory_candidate_routes(key)}
              for keys in candidates]
    assert routes[0] and routes[1] and not routes[0] & routes[1]
    for key in legacy_topics[0] | legacy_topics[1]:
        assert not memory_candidate_routes(key)
    for keys in candidates:
        topics = [memory_topic_candidate(key) for key in keys]
        assert any(topic is not None and topic.members for topic in topics)
        for topic in topics:
            if topic is not None:
                assert MemoryTopicCandidate.from_stable_key(topic.stable_key()) == topic

    initial = bridge.memory.active_structures()
    first_append = bridge.memory.append(first, speaker_kind=1)
    first_result = bridge.query(first)
    capture("first", first_result)
    assert first_result["termination"] == TERMINATION_ANSWER_CLOSED
    assert first_result["response_surface"]
    # 未训练内容现可形成开放角色假设，但仍不能把句法解析当作事实闭合。
    bridge.memory.append("书页是书本的一部分", speaker_kind=1)
    unbound = bridge.query("书页是书本的一部分")
    capture("unbound_content", unbound)
    assert not unbound["response_surface"]
    assert unbound["input_structure"]["open_relations"]
    assert all(item["status"] == ROOT_EXPANDED for item in unbound["roots"]
               if item["space"] == SPACE_MEMORY)
    assert not unbound["termination_state"]["evidence_closed"]
    bridge.memory.append(second, speaker_kind=1)
    second_result = bridge.query(second)
    capture("second", second_result)
    assert second_result["termination"] == TERMINATION_ANSWER_CLOSED
    assert second_result["response_surface"]
    refuted = bridge.memory.retract(second, speaker_kind=1)
    conflict = bridge.query(second)
    capture("second_refuted", conflict)
    assert conflict["termination_state"]["conflict_open"] == 1
    assert not conflict["response_surface"]
    returned = bridge.query(first)
    capture("returned_first", returned)
    assert returned["termination"] == TERMINATION_ANSWER_CLOSED
    assert returned["response_surface"] == first_result["response_surface"]
    recalled = bridge.memory.active_structures(candidate_keys=candidates[0])
    assert [item.turn_seq for item in recalled.observations] == [first_append.turn_seq]
    assert all(item.source_ref[0] != refuted.source_hash for item in recalled.evidence)
    all_memory = bridge.memory.active_structures()
    assert set(initial.observations) <= set(all_memory.observations)
    assert set(initial.hypotheses) <= set(all_memory.hypotheses)
    assert set(initial.evidence) <= set(all_memory.evidence)
    for result in (first_result, second_result, returned):
        assert {item["space"] for item in result["evidence"]} == {
            SPACE_CORE, SPACE_MEMORY, SPACE_DIALOGUE}
        assert result["response_plan"]["memory_refs"]
        assert result["response_plan"]["claim_refs"] == [
            result["termination_state"]["best_candidate"]]
        assert result["memory_competitions"]
    path, owner = bridge.memory.path, bridge.memory.owner
    bridge.memory.close()
    bridge.memory = TrainedDialogueMemoryGraph(
        path, tenant_id=owner.tenant_id, user_id=owner.user_id,
        session_id=owner.session_id, input_projector=bridge.input_projector)
    assert bridge.memory.active_structures() == all_memory
    restored = bridge.query(first)
    capture("cold_first", restored)
    assert restored == returned
    return {
        "old_observations_preserved": len(initial.observations),
        "old_hypotheses_preserved": len(initial.hypotheses),
        "old_evidence_preserved": len(initial.evidence),
        "observations": len(all_memory.observations),
        "hypotheses": len(all_memory.hypotheses),
        "evidence": len(all_memory.evidence),
        "cold_equal": 1,
        "uncovered_content_queries": 1,
        "broad_free_dialogue_complete": 0,
    }


def test_memory_topics_do_not_equate_shared_syntax(tmp_path):
    """同句法不同概念的反对证据不跨话题，返回原话题与冷恢复均使用完整图。"""
    database = _trained_database(tmp_path)
    with TrainedGraphQueryBridge(
            database, memory_database=tmp_path / "topic_memory.sqlite3",
            session_id=29) as bridge:
        exercise_topic_switch(bridge, capture=lambda _name, _result: None)


def test_structured_memory_ohe_is_parallel_and_replayable(tmp_path):
    """同一训练投影产生 O/H/E，查询沿三层 Memory 结构边并可重启恢复。"""
    database = _trained_database(tmp_path)
    memory_db = tmp_path / "structured_memory.sqlite3"
    surface = "麻雀集合包含于鸟类集合"

    with TrainedGraphQueryBridge(
            database, memory_database=memory_db, session_id=19) as bridge:
        assert bridge.memory is not None
        appended = bridge.memory.append(surface, speaker_kind=1)
        assert appended.observation_key
        assert appended.hypothesis_keys
        assert appended.evidence_keys
        before = bridge.memory.active_structures()
        assert len(before.observations) == 1
        assert len(before.hypotheses) == len(appended.hypothesis_keys)
        assert len(before.evidence) == len(appended.evidence_keys)

        result = bridge.query(surface, minimum_depth=2, max_depth=8)
        assert {item["space"] for item in result["roots"]} == {
            SPACE_CORE, SPACE_MEMORY, SPACE_DIALOGUE}
        assert any(
            item["space"] == SPACE_MEMORY
            and item["status"] == ROOT_EXPANDED
            for item in result["roots"])
        assert {item["space"] for item in result["evidence"]} >= {
            SPACE_CORE, SPACE_MEMORY, SPACE_DIALOGUE}
        memory_predicates = {
            tuple(item["predicate"])
            for item in result["hops"] if item["space"] == SPACE_MEMORY
        }
        assert {
            (MEMORY_EVENT_OBSERVATION,),
            (MEMORY_EVENT_HYPOTHESIS,),
            (MEMORY_EVENT_EVIDENCE,),
        } <= memory_predicates
        assert any(
            item["space"] == SPACE_CORE for item in result["hops"])
        assert any(
            item["space"] == SPACE_DIALOGUE for item in result["hops"])
        assert result["active_spaces"] == [
            SPACE_CORE, SPACE_MEMORY, SPACE_DIALOGUE]
        assert surface not in json.dumps(result, ensure_ascii=False, sort_keys=True)

    with TrainedDialogueMemoryGraph(
            memory_db, session_id=19,
            input_projector=bridge.input_projector) as restored:
        after = restored.active_structures()
        assert after == before

    # Same code points with a topology absent from the trained projector do not
    # become a Memory semantic route merely because the source surface overlaps.
    with TrainedGraphQueryBridge(
            database, memory_database=memory_db, session_id=19) as bridge:
        old = bridge.memory.append("麻雀麻雀麻雀", speaker_kind=1)
        assert old.hypothesis_keys
        unmatched = bridge.query("麻雀麻雀麻雀", minimum_depth=1, max_depth=3)
        memory_roots = [
            item for item in unmatched["roots"] if item["space"] == SPACE_MEMORY]
        assert memory_roots
        assert all(item["status"] == ROOT_EMPTY for item in memory_roots)
        assert not any(item["space"] == SPACE_MEMORY
                       for item in unmatched["hops"])


def test_memory_refute_stays_in_parallel_trace_and_recency_is_turn_based(tmp_path):
    """跨越旧热区的冲突仍沿三图传播，派生索引重建不改变证据与结构路径。"""
    database = _trained_database(tmp_path)
    memory_db = tmp_path / "conflicting_memory.sqlite3"
    surface = "麻雀集合包含于鸟类集合"

    with TrainedGraphQueryBridge(
            database, memory_database=memory_db, session_id=23) as bridge:
        assert bridge.memory is not None
        supported = bridge.memory.append(surface, speaker_kind=1)
        for ordinal in range(65):
            bridge.memory.append(f"独立话题记录 {ordinal}", speaker_kind=1)
        refuted = bridge.memory.retract(surface, speaker_kind=1)
        candidates = memory_input_candidate_keys(bridge.input_projector.project(
            tuple(ord(value) for value in surface)))
        snapshot = bridge.memory.active_structures(candidate_keys=candidates)
        assert [item.turn_seq for item in snapshot.observations] == [1, 67]

        by_candidate = {}
        for item in snapshot.hypotheses:
            by_candidate.setdefault(item.candidate_key, []).append(item)
        assert any(
            len(items) >= 2
            and len({item.competition_key for item in items}) == 1
            for items in by_candidate.values())
        stance_by_hypothesis = {
            item.hypothesis_key: {
                evidence.stance for evidence in snapshot.evidence
                if evidence.hypothesis_key == item.hypothesis_key
            }
            for item in snapshot.hypotheses
        }
        assert set().union(*stance_by_hypothesis.values()) >= {1, 2}
        assert supported.turn_seq < refuted.turn_seq

        result = bridge.query(surface, minimum_depth=2, max_depth=8)
        assert result["termination"] != TERMINATION_ANSWER_CLOSED
        assert result["termination_state"]["conflict_open"] == 1
        assert result["response_surface"] == ""
        assert {item["polarity"] for item in result["evidence"]} >= {1, 2}
        for space in (SPACE_MEMORY, SPACE_DIALOGUE):
            memory_evidence = [item for item in result["evidence"]
                               if item["space"] == space and item["evidence_key"]]
            assert {item["polarity"] for item in memory_evidence} == {1, 2, 3}
            assert all(item["scope"] and item["payload_key"] for item in memory_evidence)
        open_keys = {item.hypothesis_key for item in snapshot.hypotheses
                     if memory_candidate_category(item.candidate_key) == MEMORY_CATEGORY_OPEN_RELATION}
        assert open_keys
        assert {item.stance for item in snapshot.evidence
                if item.hypothesis_key in open_keys} == {2, 3}
        assert result["memory_routes"]

        # 只出现一个概念时，沿共同命题引用仍能激活该来源的其他结构假设。
        partial = bridge.input_projector.project(tuple(ord(value) for value in "麻雀集合"))
        partial_candidates = memory_input_candidate_keys(partial)
        related = bridge.memory.active_structures(candidate_keys=partial_candidates)
        assert any(item.candidate_key not in partial_candidates for item in related.hypotheses)
        assert related.match_routes

        initial = result["frontier_trace"][0]
        discourse_priorities = [
            row for row in initial["frontier_priority"]
            if row[0] == SPACE_DIALOGUE and row[6] == 2
        ]
        assert discourse_priorities
        recencies = {row[8] for row in discourse_priorities}
        assert min(recencies) == supported.turn_seq
        assert max(recencies) == refuted.turn_seq
        assert all(
            all(type(value) is int and value >= 0 for value in row)
            for step in result["frontier_trace"]
            for row in step["frontier_priority"])

        # 新来源可给旧假设追加未知证据；它的来源不能被替换成旧观察来源。
        first_hypothesis = snapshot.hypotheses[0]
        unknown = bridge.memory.append_evidence(
            first_hypothesis.hypothesis_key, source=refuted.source, stance=3,
            reason_key=(29031, 1), detail=(29031, 2))
        assert unknown.source_ref == (refuted.source_hash, *refuted.source.stable_key())
        # 通过显式 supersedes 修订所有反对证据，原证据仍必须完整可回读。
        for evidence in snapshot.evidence:
            if evidence.stance == 2:
                bridge.memory.append_evidence(
                    evidence.hypothesis_key, source=refuted.source, stance=1,
                    reason_key=(29032, 1), detail=(29032, 2),
                    supersedes_key=evidence.evidence_key)
        revised = bridge.query(surface, minimum_depth=2, max_depth=8)
        assert revised["termination_state"]["conflict_open"] == 0
        assert revised["termination"] == TERMINATION_ANSWER_CLOSED
        assert revised["response_surface"]
        assert revised["response_plan"]["memory_refs"]
        assert revised["response_plan"]["scope_and_time"]
        assert len(revised["response_plan"]["evidence_refs"]) >= len(revised["evidence"])
        assert revised["response_plan"]["discourse_links"]
        assert revised["response_plan"]["claim_refs"] == [
            revised["termination_state"]["best_candidate"]]
        core_evidence = [item for item in revised["evidence"] if item["space"] == SPACE_CORE]
        assert core_evidence
        for item in core_evidence:
            if item["hypothesis"][0] == OPEN_SCHEMA_VERSION:
                frame = OpenRoleFrame.from_stable_key(tuple(item["payload_key"]))
                assert tuple(item["hypothesis"]) == frame.schema_key()
                assert item["polarity"] == 1
                continue
            restored_record = EvidenceRecord.from_stable_key(tuple(item["payload_key"]))
            assert item["polarity"] == restored_record.stance
            assert tuple(item["scope"]) == restored_record.hypothesis.scope.stable_key()
            assert tuple(item["source_ref"][1:]) == restored_record.source.stable_key()
        assert {item["polarity"] for item in revised["evidence"]
                if item["space"] == SPACE_MEMORY} == {1, 2, 3}
        old_keys = {item.evidence_key for item in snapshot.evidence}
        assert old_keys <= {tuple(item["evidence_key"]) for item in revised["evidence"]}
        assert any(item["supersedes"] for item in revised["evidence"])
        snapshot = bridge.memory.active_structures(candidate_keys=candidates)
        result = revised

        projector = bridge.input_projector
        selector = bridge.memory.candidate_index.selector
        # 只移除测试会话内可重建的定位索引；保留 cursor 和所有权威 O/H/E。
        bridge.memory.backend.delete(CANDIDATE_INDEX_TABLE, selector)
        bridge.memory.close()
        bridge.memory = TrainedDialogueMemoryGraph(
            memory_db, session_id=23, input_projector=projector)
        assert bridge.memory.active_structures(candidate_keys=candidates) == snapshot
        replay = bridge.query(surface, minimum_depth=2, max_depth=8)
        assert replay == result
