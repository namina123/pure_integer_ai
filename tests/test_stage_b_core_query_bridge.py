"""三图库并行回归：共同 roots/frontier、结构证据与生成。

桥测试验证：Core relation、Memory Observation、Dialogue realization frame
在同一 QueryState 中登记并实际扩展，trace 带 roots/evidence/共同 frontier/
visited/终止状态。Memory 原文不进入查询 trace，生成不做 successor 回放。
"""
from __future__ import annotations

import json
from pathlib import Path

from pure_integer_ai.cognition.shared.query_state import (
    ROOT_EMPTY,
    ROOT_EXPANDED,
    SPACE_CORE,
    SPACE_DIALOGUE,
    SPACE_MEMORY,
    TERMINATION_ANSWER_CLOSED,
)
from pure_integer_ai.cognition.shared.graph_ontology import (
    relation_concept_identity,
)
from pure_integer_ai.cognition.shared.hypothesis import HypothesisKey
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    OBJECT_ENTITY,
    SourceRef,
    VersionBundle,
    concept_identity,
    minimal_instruction_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.cognition.shared.semantic_graph import (
    AtomicPropositionPredicates,
    SemanticGraph,
)
from pure_integer_ai.cognition.shared.semantic_object import role_identity
from pure_integer_ai.cognition.understanding.occurrence_index import (
    OccurrenceIndex,
    OccurrenceProtocol,
)
from pure_integer_ai.cognition.understanding.query_input_structure import (
    CARRIER_SEMANTIC_GRAPH,
    STRUCTURE_CLOSED,
    TrainedInputStructureProjector,
)
from pure_integer_ai.cognition.understanding.query_structure_adapter import (
    SemanticCandidateBindingSpan,
    SemanticCandidateInputView,
)
from pure_integer_ai.cognition.understanding.semantic_builder import (
    SemanticBindingSpec,
    SemanticBuildPlan,
    SemanticBuilderProtocol,
    SemanticCandidateBuilder,
    SemanticFillerSpec,
    SemanticObjectSpec,
    SemanticPropositionSpec,
)
from pure_integer_ai.cognition.understanding.semantic_builder_graph import (
    SemanticBuilderTracePredicates,
    SemanticCandidateGraphAdapter,
)
from pure_integer_ai.cognition.understanding.span_index import (
    SpanIndex,
    SpanProtocol,
)

from pure_integer_ai.experiments.conversation_typed_relation_bridge import (
    build_authored_w06_learning_runtime,
)
from pure_integer_ai.experiments.language_protocol_runtime import (
    install_language_graph_protocols,
)
from pure_integer_ai.experiments.run_conversation_training import (
    dialogue_semantic_protocols,
)
from pure_integer_ai.experiments.trained_dialogue_memory_graph import (
    TrainedDialogueMemoryGraph,
)
from pure_integer_ai.experiments.trained_graph_query_bridge import (
    TrainedGraphQueryBridge,
)
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.storage.backend import DictBackend, SQLiteBackend
from pure_integer_ai.storage.edge_store import EPI_STRUCTURED, SOURCE_BARE_TEXT

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


def _trained_database(tmp_path: Path) -> Path:
    """消费公开 relation samples 训练 W-06 并落盘 SQLite。"""
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
        build_authored_w06_learning_runtime(
            backend,
            context,
            _relation_courses(),
            tmp_path / "typed-relation-pack",
        )
        backend.commit()
    finally:
        backend.close()
    return database


def _semantic_input_view() -> SemanticCandidateInputView:
    """构造已物化 S-02 候选和其已核验整数 Span 输入视图。"""
    backend = DictBackend()
    try:
        context = make_train_context(backend)
        source = SourceRef(
            SOURCE_BARE_TEXT, 29001, 1, GLOBAL_OWNER_SCOPE, VersionBundle())
        scope = document_scope(source)
        occurrences = OccurrenceIndex(
            context.graph_ontology,
            context.scoped_identity_store,
            OccurrenceProtocol((29002, 1), (29002, 2)),
        )
        spans = SpanIndex(
            context.graph_ontology,
            context.scoped_identity_store,
            SpanProtocol((29003, 1), (29003, 2), (29003, 3), (29003, 4)),
            occurrences,
        )
        anchor = spans.ensure_ref(
            source=source, raw_text="甲是", scope=scope, members=((0, 2),))
        upstream = HypothesisKey(
            (29004, 1), (29004, 1), (29004, 2), scope, source)
        builder = SemanticCandidateBuilder(
            spans,
            SemanticBuilderProtocol(
                minimal_instruction_identity((29005, 1)), (29005, 2)),
            occurrences,
        )
        plan = SemanticBuildPlan(
            upstream,
            (29006, 1),
            (SemanticObjectSpec(OBJECT_ENTITY, (1,)),),
            (SemanticPropositionSpec(
                (1,),
                (29006, 2),
                concept_identity((29007, 1)),
                structure_concept_identity((29008, 1)),
                (SemanticBindingSpec(
                    role_identity((29009, 1)),
                    SemanticFillerSpec(
                        local_ref=SemanticObjectSpec(
                            OBJECT_ENTITY, (1,)).local_ref),
                ),),
            ),),
        )
        build = builder.compile(anchor, plan)
        refs = tuple(context.graph_ontology.materialize(
            relation_concept_identity((29010, ordinal)))
            for ordinal in range(1, 10))
        graph = SemanticCandidateGraphAdapter(
            SemanticGraph(
                context.graph_ontology,
                AtomicPropositionPredicates(*refs[:6]),
            ),
            SemanticBuilderTracePredicates(*refs[6:]),
        )
        candidate = graph.materialize(
            build,
            provenance_kind=SOURCE_BARE_TEXT,
            epistemic_origin=EPI_STRUCTURED,
        ).candidates[0]
        binding = candidate.atomic.definition.canonical_bindings()[0]
        return SemanticCandidateInputView(
            candidate,
            29011,
            (ord("是"),),
            1,
            2,
            (SemanticCandidateBindingSpan(
                binding.role.stable_key(),
                binding.filler.stable_key(),
                (ord("甲"),),
                0,
                1,
                binding.ordinal,
            ),),
        )
    finally:
        backend.close()


def test_three_space_roots_share_one_query_state(tmp_path):
    """三类结构根共同扩展；权重只改变次序且不屏蔽低权重图。"""
    database = _trained_database(tmp_path)
    memory_db = tmp_path / "session_memory.sqlite3"
    semantic_view = _semantic_input_view()
    with TrainedDialogueMemoryGraph(memory_db, session_id=7) as memory:
        memory.append("用户在讨论汽车相关话题。", speaker_kind=1)
        memory.append("我最近在查汽车构造资料。", speaker_kind=1)
    with TrainedGraphQueryBridge(
            database,
            memory_database=memory_db,
            session_id=7,
            semantic_input_views=(semantic_view,),
    ) as bridge:
        assert bridge.memory is not None
        result = bridge.query(
            "麻雀集合包含于鸟类集合", minimum_depth=2, max_depth=3)
        core_surfaces = {
            "".join(chr(value) for value in item["surface_values"])
            for item in result["hops"] if item["space"] == SPACE_CORE}
        assert "小麻雀" in core_surfaces
        assert "鸟类集合" in result["response_surface"]
        assert result["format"] == "PURE_INTEGER_TRAINED_GRAPH_QUERY_TRACE_V3"
        assert result["active_spaces"] == [
            SPACE_CORE, SPACE_MEMORY, SPACE_DIALOGUE]
        assert {item["space"] for item in result["roots"]} == {
            SPACE_CORE, SPACE_MEMORY, SPACE_DIALOGUE}
        # These pre-Stage-D records were appended without the bridge's trained
        # projector, so their generic hypotheses are intentionally not a
        # semantic Memory match.  Memory still participates through an explicit
        # empty root while Core and Dialogue close the learned relation.
        assert all(
            item["status"] == (
                ROOT_EXPANDED if item["space"] != SPACE_MEMORY else ROOT_EMPTY)
            for item in result["roots"])
        assert {item["space"] for item in result["evidence"]} == {
            SPACE_CORE, SPACE_DIALOGUE}
        expanded_spaces = [item[0] for item in result["expanded_edges"]]
        assert expanded_spaces[0] == SPACE_CORE
        assert expanded_spaces.index(SPACE_DIALOGUE) > expanded_spaces.index(
            SPACE_CORE)
        assert set(expanded_spaces) == {
            SPACE_CORE, SPACE_DIALOGUE}
        assert result["termination"] == TERMINATION_ANSWER_CLOSED
        assert result["depth"] >= 2
        assert result["termination_state"]["required_slots_open"] == 0
        assert result["termination_state"]["evidence_closed"] == 1
        assert result["termination_state"]["conflict_open"] == 0
        assert result["termination_state"]["marginal_gain"] > 0
        assert result["termination_state"]["cycle_hit"] == 0
        assert result["termination_state"]["generation_ready"] == 1
        assert (result["termination_state"]["best_score"]
                > result["termination_state"]["second_score"])
        assert result["frontier_trace"]
        # Memory root 只携带 Observation/SourceRef 整数身份，不泄漏历史正文。
        encoded = json.dumps(result, ensure_ascii=False, sort_keys=True)
        assert "汽车" not in encoded
        # S-02 graph candidate 走同一 projector/QueryState，不进入独立 fallback。
        semantic = bridge.query("甲-是", minimum_depth=1, max_depth=2)
        assert any(
            item["carrier"] == [CARRIER_SEMANTIC_GRAPH]
            for item in semantic["input_structure"]["carrier_candidates"])
        structure = semantic_view.candidate.structure.stable_key()
        semantic_relations = tuple(
            item for item in semantic["input_structure"]["relation_candidates"]
            if item["structure"] == list(structure))
        assert len(semantic_relations) == 1
        assert semantic_relations[0]["state"] == STRUCTURE_CLOSED
        assert any(
            item["space"] == SPACE_CORE and item["kind"] == 2
            and item["ref"] == list(structure)
            for item in semantic["anchors"])
        assert any(
            item["space"] == SPACE_CORE and item["status"] == ROOT_EXPANDED
            for item in semantic["roots"])
        assert semantic["response_surface"] == ""

        # 稳定 PURE_ALIAS 允许表述互换，但不合并节点或跳过共同查询证据。
        rewritten = "启明星就是晨星"
        values = tuple(ord(value) for value in rewritten)
        baseline = TrainedInputStructureProjector(bridge.input_projector.structures).project(values)
        projected = bridge.input_projector.project(values)
        closed = {item.proposition_key for item in projected.relation_candidates
                  if item.state == STRUCTURE_CLOSED}
        assert closed - {item.proposition_key for item in baseline.relation_candidates
                         if item.state == STRUCTURE_CLOSED}
        assert all(not projected.concept_evidence_closed(key, frozenset()) for key in closed)
        bridge.memory.append(rewritten, speaker_kind=1)
        linked = bridge.query(rewritten, minimum_depth=2, max_depth=8)
        assert linked["termination"] == TERMINATION_ANSWER_CLOSED
        assert linked["input_structure"]["concept_routes"]
        assert {item["space"] for item in linked["evidence"]} == {
            SPACE_CORE, SPACE_MEMORY, SPACE_DIALOGUE}
        assert linked["response_plan"]["claim_refs"] == [
            linked["termination_state"]["best_candidate"]]


def test_bridge_replayable_and_evidence_closed(tmp_path):
    """同一输入在同一训练库上的 trace 与 stable key 可重放。"""
    database = _trained_database(tmp_path)
    first = None
    for _ in range(2):
        with TrainedGraphQueryBridge(database) as bridge:
            result = bridge.query(
                "麻雀集合包含于鸟类集合",
                minimum_depth=1,
                max_depth=3,
            )
        if first is None:
            first = result
        else:
            assert first["hops"] == result["hops"]
            assert first["state_key"] == result["state_key"]
    assert all(item["source_hash"] > 0 for item in first["evidence"])
    assert first["termination"] == TERMINATION_ANSWER_CLOSED


def test_equal_relation_candidates_do_not_select_by_stable_key(tmp_path):
    """仅有共享概念且两个关系同分时保留路径但不任取表层回答。"""
    database = _trained_database(tmp_path)
    with TrainedGraphQueryBridge(database) as bridge:
        result = bridge.query("麻雀集合", minimum_depth=1, max_depth=3)
    assert result["hops"]
    assert result["termination"] != TERMINATION_ANSWER_CLOSED
    assert result["response_plan"] is None
    assert result["response_surface"] == ""
    assert (result["termination_state"]["best_score"]
            == result["termination_state"]["second_score"])


def test_no_anchor_returns_clarify_missing_binding(tmp_path):
    """无概念 anchor 的普通语言返回 CLARIFY_MISSING_BINDING，不近邻。"""
    database = _trained_database(tmp_path)
    with TrainedGraphQueryBridge(database) as bridge:
        result = bridge.query("火星轨道参数是什么？", minimum_depth=1)
        assert result["termination_reason"] in {
            "CLARIFY_MISSING_BINDING", "NO_CONCEPT_ANCHOR"}
        assert result["hops"] == []
        assert {item["space"] for item in result["roots"]} == {
            SPACE_CORE, SPACE_MEMORY, SPACE_DIALOGUE}
        assert all(item["status"] == ROOT_EMPTY for item in result["roots"])
