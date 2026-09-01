"""当前公开 authored relation 到正式 W-06 训练 owner 的接线专项。"""
from __future__ import annotations

from pathlib import Path
from io import BytesIO
import json

from pure_integer_ai.experiments.conversation_typed_relation_bridge import (
    build_authored_w06_adapter,
    build_authored_w06_learning_runtime,
)
from pure_integer_ai.experiments.ph2_w06_learning import (
    W06RelationLearningRuntime,
    build_w06_learning_runtime,
)
from pure_integer_ai.experiments.trained_relation_graph_runtime import (
    GRAPH_RELATION_CONFLICT,
    TrainedRelationGraphRuntime,
)
from pure_integer_ai.experiments.run_trained_relation_graph_terminal import (
    run_trained_relation_graph_terminal,
)
from pure_integer_ai.experiments.trained_dialogue_memory_graph import (
    DIALOGUE_MEMORY_POSTING_TABLE,
    TrainedDialogueMemoryGraph,
)
from pure_integer_ai.cognition.shared.hypothesis import LIFECYCLE_SUPERSEDED
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.experiments.language_protocol_runtime import (
    install_language_graph_protocols,
)
from pure_integer_ai.experiments.run_conversation_training import (
    dialogue_semantic_protocols,
)
from pure_integer_ai.storage.backend import DictBackend, SQLiteBackend


ROOT = Path(__file__).resolve().parents[1]


def _relation_courses() -> tuple[Path, ...]:
    """返回现役七类 authored W-06 训练输入。"""
    root = ROOT / "data" / "ph2"
    return tuple(sorted(root.glob("authored_relation_*_w06_seed_v2.jsonl.sample"))) + tuple(
        sorted(
            path for path in root.glob("authored_relation_*_seed_v1.jsonl.sample")
            if path.name != "authored_relation_alias_refers_seed_v1.jsonl.sample"
        )
    )


def test_w06_owner_restores_h00_h04_from_current_authored_courses(tmp_path):
    """W-06 重开必须从 Core 历史恢复，不能只依赖候选图终态。"""
    adapter = build_authored_w06_adapter(
        _relation_courses(), tmp_path / "typed-relation-pack")
    backend = DictBackend()
    try:
        first = build_w06_learning_runtime(backend, adapter)
        mismatches = []
        for hypothesis in first.learning.engine.ledger.hypotheses():
            definition = first.learning.engine.definition(hypothesis)
            candidate = first.candidate_graph.ontology.resolve(
                definition.candidate)
            history = (() if candidate is None
                       else first.candidate_graph.history(candidate))
            projection = (None if not history
                          else first.candidate_graph.project(candidate))
            lifecycle = first.learning.engine.ledger.snapshot(
                hypothesis).lifecycle
            if ((lifecycle == LIFECYCLE_SUPERSEDED)
                    != (projection is not None and projection.state
                        == first.projection_protocol.superseded_state)):
                mismatches.append((
                    definition.candidate.stable_key(),
                    lifecycle,
                    ("NONE" if projection is None else
                     "ACTIVE" if projection.state
                     == first.projection_protocol.active_state else
                     "INACTIVE" if projection.state
                     == first.projection_protocol.inactive_state else
                     "SUPERSEDED"),
                ))
        assert not mismatches
        restored = W06RelationLearningRuntime(backend)
        assert restored.learning.state_key() == first.learning.state_key()
        assert restored.learning.report() == first.learning.report()
    finally:
        backend.close()


def test_trained_relation_graph_reopens_read_only_without_course_or_qa(tmp_path):
    """关闭训练 owner 后只凭 SQLite 恢复 active 命题并执行稀疏 filler 查询。"""
    database = tmp_path / "training.sqlite3"
    backend = SQLiteBackend(str(database))
    try:
        context = make_train_context(backend)
        _semantic, occurrence, span = dialogue_semantic_protocols()
        install_language_graph_protocols(
            context,
            occurrence_protocol=occurrence,
            span_protocol=span,
        )
        trained = build_authored_w06_learning_runtime(
            backend,
            context,
            _relation_courses(),
            tmp_path / "typed-relation-pack",
        )
        backend.commit()
        expected_active = tuple(
            item.proposition.proposition
            for item in trained.active_candidates()
        )
        expected_surfaces = {}
        for candidate in trained.active_candidates():
            endpoint_by_identity = {
                endpoint.identity: endpoint
                for endpoint in candidate.endpoints
            }
            for binding in candidate.proposition.canonical_bindings():
                expected_surfaces[(
                    candidate.proposition.proposition,
                    binding.role,
                )] = endpoint_by_identity[binding.filler].surface_fragment
    finally:
        backend.close()

    with TrainedRelationGraphRuntime(database) as runtime:
        assert runtime.backend.read_only is True
        assert runtime.snapshot.candidate_identities == tuple(sorted(
            expected_active, key=lambda item: item.stable_key()))
        for fact in runtime.active_surface_facts():
            for binding in fact.bindings:
                assert binding.surface == expected_surfaces[(
                    fact.proposition, binding.role)]
        first = runtime.active_propositions()[0]
        filler = first.definition.canonical_bindings()[0].filler
        matches = runtime.lookup_active_by_filler(filler)
        assert first.definition.proposition in {
            item.definition.proposition for item in matches}
        assert all(
            item.definition.proposition in runtime.snapshot.candidate_identities
            for item in matches)
        answer = None
        surface_fact = None
        for candidate_fact in runtime.active_surface_facts():
            held_out_order = tuple(reversed(candidate_fact.bindings))
            query = " / ".join((
                *(item.surface for item in held_out_order),
                candidate_fact.cue,
            ))
            candidate_answer = runtime.respond(query)
            if (candidate_answer is not None
                    and candidate_answer.predicate == candidate_fact.predicate
                    and set(candidate_answer.matched_fillers) == {
                        item.filler for item in candidate_fact.bindings}):
                answer = candidate_answer
                surface_fact = candidate_fact
                break
        assert answer is not None
        assert surface_fact is not None
        assert answer.surface.strip()
        assert answer.generation.surface == answer.surface
        assert answer.generation.slot_count == len(surface_fact.bindings)
        assert 0 < answer.fact_reads <= len(runtime.active_surface_facts())
        assert all(
            item.surface in answer.surface for item in surface_fact.bindings)
        compatible_foreign_frames = tuple(
            frame for frame in runtime.active_surface_frames()
            if frame.proposition != surface_fact.proposition
            and frame.predicate == surface_fact.predicate
            and set(frame.roles) == {
                item.role for item in surface_fact.bindings})
        if compatible_foreign_frames:
            assert answer.generation.frame_proposition != surface_fact.proposition
        assert runtime.active_surface_frames()
        assert set(answer.matched_fillers) == {
            item.filler for item in surface_fact.bindings}
        for candidate_fact in runtime.active_surface_facts():
            complete_query = " / ".join((
                candidate_fact.cue,
                *(item.surface for item in candidate_fact.bindings),
            ))
            assert runtime.respond(complete_query) is not None
        conflict_checked = False
        facts = runtime.active_surface_facts()
        for left in facts:
            for right in facts:
                left_surfaces = {item.surface for item in left.bindings}
                right_surfaces = {item.surface for item in right.bindings}
                if (left.predicate != right.predicate
                        or left.proposition == right.proposition
                        or left_surfaces & right_surfaces
                        or len(left.bindings) < 2
                        or len(right.bindings) < 2):
                    continue
                conflict_query = " / ".join((
                    left.cue,
                    left.bindings[0].surface,
                    right.bindings[-1].surface,
                ))
                assert runtime.respond(conflict_query) is None
                assert runtime.query(conflict_query).result_code == (
                    GRAPH_RELATION_CONFLICT)
                conflict_checked = True
                break
            if conflict_checked:
                break
        assert conflict_checked
        try:
            runtime.backend.insert("concept_node", {
                "space_id": 999,
                "local_id": 999,
                "type": 1,
                "tier": 1,
            })
        except RuntimeError as error:
            assert "禁止写入" in str(error)
        else:
            raise AssertionError("只读训练图 owner 意外允许写入")

    fallback = tmp_path / "fallback-surfaces.txt"
    fallback.write_text("我暂时无法根据现有信息确定答案。\n", encoding="utf-8")
    memory_database = tmp_path / "dialogue-memory.sqlite3"
    metrics_path = tmp_path / "dialogue-metrics.json"
    first_input = BytesIO(
        b'{"id":"one","op":"turn","text":"\xe6\x88\x91\xe5\x8f\xab\xe5\xb0\x8f\xe6\x98\x8e\xef\xbc\x8c\xe4\xbd\x8f\xe5\x9c\xa8\xe5\x8c\x97\xe4\xba\xac\xe3\x80\x82"}\n'
        b'{"id":"stop","op":"quit"}\n')
    first_output = BytesIO()
    assert run_trained_relation_graph_terminal(
        training_database=database,
        fallback_surfaces=("我暂时无法根据现有信息确定答案。",),
        memory_database=memory_database,
        memory_session_id=73,
        input_stream=first_input,
        output_stream=first_output,
        metrics_output=metrics_path,
        protocol_stream=True,
    ) == 0
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["turn_count"] == 1
    assert metrics["route_counts"] == {
        "boundary": 1, "core_graph": 0,
        "dialogue_graph": 0, "memory_graph": 0}
    assert metrics["latency_p50_us"] <= metrics["latency_p95_us"]
    assert metrics["peak_working_set_bytes"] > 0
    with TrainedDialogueMemoryGraph(
            memory_database, session_id=73) as recorded:
        speaker_rows = recorded.backend.select(
            DIALOGUE_MEMORY_POSTING_TABLE,
            order_by="turn_seq",
        )
        assert {row["speaker_kind"] for row in speaker_rows} == {1, 2}
    second_output = BytesIO()
    assert run_trained_relation_graph_terminal(
        training_database=database,
        fallback_surfaces=("我暂时无法根据现有信息确定答案。",),
        memory_database=memory_database,
        memory_session_id=73,
        input_stream=BytesIO(
            b'{"id":"two","op":"turn","text":"\xe6\x88\x91\xe5\x8f\xab\xe5\xb0\x8f\xe6\x98\x8e\xef\xbc\x8c\xe4\xbd\x8f\xe5\x9c\xa8\xe5\x8c\x97\xe4\xba\xac\xe5\x90\x97\xef\xbc\x9f"}\n'),
        output_stream=second_output,
        protocol_stream=True,
    ) == 0
    response = json.loads(second_output.getvalue().decode("utf-8"))
    assert response["id"] == "two"
    assert response["text"] == "我叫小明，住在北京。"
    assert response["source"]["kind"] == "interaction_memory_graph"
    assert response["source"]["source_hash"] > 0
