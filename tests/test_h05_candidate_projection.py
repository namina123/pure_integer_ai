"""H-05 候选图投影、生命周期和恢复边界专项。"""
from __future__ import annotations

import pytest

from pure_integer_ai.cognition.shared.candidate_projection import (
    CandidateProjectionError,
    CandidateProjectionGraph,
    CandidateProjectionProtocol,
    EvidenceCandidateProjector,
)
from pure_integer_ai.cognition.shared.evidence_candidate import (
    CandidateBinding,
    CandidateVerification,
    EvidenceCandidateDefinition,
    EvidenceCandidateEngine,
    EvidenceCandidateProtocol,
)
from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EvidenceRecord,
)
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    ObjectIdentity,
    SourceRef,
    VersionBundle,
    concept_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.experiments.formal_train import make_train_context
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import EPI_STRUCTURED, SOURCE_BARE_TEXT
from pure_integer_ai.storage.graph_statement import GRAPH_STATEMENT_TABLE
from pure_integer_ai.training.cursor import DUMP_TABLES, dump_run, load_run


def _source(source_id: int) -> SourceRef:
    """构造同一 owner/version 下的独立测试来源。"""
    return SourceRef(
        17,
        source_id,
        0,
        GLOBAL_OWNER_SCOPE,
        VersionBundle(),
    )


def _engine() -> EvidenceCandidateEngine:
    """构造形成条件和 aggregate manifest 均由协议注入的候选 owner。"""
    aggregate = _source(900)
    return EvidenceCandidateEngine(EvidenceCandidateProtocol(
        hypothesis_kind_key=(101, 1),
        formation_reason_key=(101, 2),
        aggregate_source=aggregate,
        aggregate_scope=document_scope(aggregate),
        minimum_forming_sources=2,
    ))


def _definition(
        local_key: int, *, competition: tuple[int, ...] = (201, 1),
        predicate: ObjectIdentity | None = None,
        ) -> EvidenceCandidateDefinition:
    """构造带两个动态图 binding 的结构候选定义。"""
    return EvidenceCandidateDefinition(
        candidate=structure_concept_identity((301, local_key)),
        competition_key=competition,
        bindings=(
            CandidateBinding(
                predicate or concept_identity((401, 1)),
                concept_identity((501, local_key)),
            ),
            CandidateBinding(
                concept_identity((401, 2)),
                concept_identity((601, 1)),
            ),
        ),
        forming_sources=(_source(1), _source(2)),
    )


def _projection_protocol() -> CandidateProjectionProtocol:
    """构造全部关系、状态和事件 kind 均不同的开放投影协议。"""
    identities = tuple(concept_identity((701, index)) for index in range(14))
    return CandidateProjectionProtocol(
        event_candidate=identities[0],
        event_kind=identities[1],
        event_from_state=identities[2],
        event_to_state=identities[3],
        event_hypothesis=identities[4],
        event_replacement=identities[5],
        inactive_state=identities[6],
        active_state=identities[7],
        superseded_state=identities[8],
        promotion_kind=identities[9],
        refresh_kind=identities[10],
        demotion_kind=identities[11],
        supersede_kind=identities[12],
        event_namespace_key=(801, 1),
    )


def _graph(backend: DictBackend, protocol: CandidateProjectionProtocol):
    """在正式训练图 facade 上构造 H-05 投影读写器。"""
    context = make_train_context(backend)
    return context, CandidateProjectionGraph(context.graph_ontology, protocol)


def _support(
        engine: EvidenceCandidateEngine, hypothesis, *, source_id: int,
        timestamp_seq: int):
    """先冻结独立预测，再揭示 support 并提交最新 H-04 决策。"""
    observation = _source(source_id)
    prediction = engine.predict(
        hypothesis,
        observation=observation,
        scope=document_scope(observation),
        event_key=(901, source_id),
        visible_inputs=(concept_identity((1001, source_id)),),
        predicted=concept_identity((1101, 1)),
    )
    engine.reveal(
        prediction,
        CandidateVerification(
            EVIDENCE_SUPPORT,
            (1201, source_id),
            _source(700 + source_id),
            concept_identity((1301, 1)),
            (1401, 1),
            (1501, source_id),
        ),
        timestamp_seq=timestamp_seq,
    )
    return engine.resolve(hypothesis, timestamp_seq=timestamp_seq + 1)


def _projector(backend: DictBackend):
    """构造候选内核、正式图、投影器和一个已登记候选。"""
    engine = _engine()
    definition = _definition(1)
    hypothesis = engine.register(definition, timestamp_base=1)
    protocol = _projection_protocol()
    context, graph = _graph(backend, protocol)
    return engine, definition, hypothesis, protocol, context, graph, (
        EvidenceCandidateProjector(engine, graph))


def _statement_count(backend: DictBackend) -> int:
    """返回 append-only 图 statement 行数供非法批次零写断言。"""
    return len(backend.select(GRAPH_STATEMENT_TABLE))


def test_forming_definition_is_recoverable_but_not_active_for_binding():
    """forming 定义可以写图，缺 support/H-04/lifecycle 时消费者必须返回空。"""
    backend = DictBackend()
    try:
        engine, definition, hypothesis, _, _, graph, _ = _projector(backend)
        graph.define(
            definition,
            hypothesis,
            provenance_kind=SOURCE_BARE_TEXT,
            epistemic_origin=EPI_STRUCTURED,
        )

        assert engine.active(hypothesis) is None
        assert graph.active_for_binding(definition.bindings[0]) == ()
        assert graph.read_definition(hypothesis).definition == definition
    finally:
        backend.close()


def test_promotion_refresh_demotion_and_repromotion_keep_current_trace():
    """状态链允许证明刷新和降级后再晋升，并始终保存最新 Evidence/decision。"""
    backend = DictBackend()
    try:
        engine, definition, hypothesis, protocol, _, graph, projector = (
            _projector(backend))
        first_decision = _support(
            engine, hypothesis, source_id=3, timestamp_seq=20)
        promoted = projector.promote(
            hypothesis,
            timestamp_seq=30,
            provenance_kind=SOURCE_BARE_TEXT,
            epistemic_origin=EPI_STRUCTURED,
        )
        assert promoted.state == protocol.active_state

        second_decision = _support(
            engine, hypothesis, source_id=4, timestamp_seq=40)
        refreshed = projector.promote(
            hypothesis,
            timestamp_seq=50,
            provenance_kind=SOURCE_BARE_TEXT,
            epistemic_origin=EPI_STRUCTURED,
        )
        assert len(refreshed.history) == 2
        assert refreshed.history[-1].definition.event_kind == protocol.refresh_kind
        assert refreshed.history[-1].definition.decision_key == (
            second_decision.stable_key())
        assert refreshed.history[0].definition.decision_key == (
            first_decision.stable_key())

        negative_source = _source(5)
        prediction = engine.predict(
            hypothesis,
            observation=negative_source,
            scope=document_scope(negative_source),
            event_key=(901, 5),
            visible_inputs=(concept_identity((1001, 5)),),
            predicted=concept_identity((1101, 1)),
        )
        refute = engine.reveal(
            prediction,
            CandidateVerification(
                EVIDENCE_REFUTE,
                (1201, 5),
                _source(705),
                concept_identity((1301, 1)),
                (1401, 1),
                (1501, 5),
            ),
            timestamp_seq=60,
        )
        engine.resolve(hypothesis, timestamp_seq=61)
        demoted = projector.demote(
            hypothesis,
            timestamp_seq=70,
            provenance_kind=SOURCE_BARE_TEXT,
            epistemic_origin=EPI_STRUCTURED,
        )
        assert demoted.state == protocol.inactive_state
        assert graph.active_for_binding(definition.bindings[0]) == ()

        engine.ledger.append_evidence(EvidenceRecord(
            refute.evidence_id + 1,
            hypothesis,
            EVIDENCE_SUPPORT,
            (1201, 6),
            _source(706),
            80,
            payload=(1601, 1),
            supersedes_evidence_id=refute.evidence_id,
        ))
        engine.resolve(hypothesis, timestamp_seq=81)
        reprojection = projector.promote(
            hypothesis,
            timestamp_seq=90,
            provenance_kind=SOURCE_BARE_TEXT,
            epistemic_origin=EPI_STRUCTURED,
        )
        assert reprojection.state == protocol.active_state
        assert len(reprojection.history) == 4
        assert graph.active_for_binding(definition.bindings[0]) == (
            reprojection,)
    finally:
        backend.close()


def test_supersede_requires_h00_replacement_and_preserves_graph_target():
    """替代只接受 H-00 已确认的同竞争组 active replacement。"""
    backend = DictBackend()
    try:
        engine, _, first, protocol, _, graph, projector = _projector(backend)
        second_definition = _definition(2)
        second = engine.register(second_definition, timestamp_base=3)
        _support(engine, first, source_id=3, timestamp_seq=20)
        _support(engine, second, source_id=4, timestamp_seq=30)
        engine.resolve(first, timestamp_seq=40)
        projector.promote(
            first,
            timestamp_seq=50,
            provenance_kind=SOURCE_BARE_TEXT,
            epistemic_origin=EPI_STRUCTURED,
        )
        projector.promote(
            second,
            timestamp_seq=51,
            provenance_kind=SOURCE_BARE_TEXT,
            epistemic_origin=EPI_STRUCTURED,
        )

        negative_source = _source(5)
        prediction = engine.predict(
            first,
            observation=negative_source,
            scope=document_scope(negative_source),
            event_key=(901, 5),
            visible_inputs=(concept_identity((1001, 5)),),
            predicted=concept_identity((1101, 1)),
        )
        engine.reveal(
            prediction,
            CandidateVerification(
                EVIDENCE_REFUTE,
                (1201, 5),
                _source(705),
                concept_identity((1301, 1)),
                (1401, 1),
                (1501, 5),
            ),
            timestamp_seq=60,
        )
        engine.resolve(first, timestamp_seq=61, replacement=second)
        projection = projector.supersede(
            first,
            second,
            timestamp_seq=70,
            provenance_kind=SOURCE_BARE_TEXT,
            epistemic_origin=EPI_STRUCTURED,
        )

        assert projection.state == protocol.superseded_state
        assert projection.replacement == second_definition.candidate
        assert graph.active_for_binding(_definition(1).bindings[0]) == ()
    finally:
        backend.close()


def test_cache_clear_and_dump_load_restore_active_projection(tmp_path):
    """运行缓存和 backend 重建后都只靠图恢复相同 active typed 候选。"""
    backend = DictBackend()
    try:
        engine, definition, hypothesis, protocol, context, graph, projector = (
            _projector(backend))
        _support(engine, hypothesis, source_id=3, timestamp_seq=20)
        expected = projector.promote(
            hypothesis,
            timestamp_seq=30,
            provenance_kind=SOURCE_BARE_TEXT,
            epistemic_origin=EPI_STRUCTURED,
        )
        context.graph_ontology.clear_runtime_caches()
        rebuilt = CandidateProjectionGraph(context.graph_ontology, protocol)
        cached = rebuilt.active_for_binding(definition.bindings[0])
        assert len(cached) == 1
        assert cached[0].candidate.definition == expected.candidate.definition
        assert cached[0].history == expected.history

        dump_run(
            backend,
            str(tmp_path),
            "h05_projection",
            spaces=[context.space_id],
            tables=DUMP_TABLES,
        )
    finally:
        backend.close()

    restored_backend = DictBackend()
    try:
        restored_context = make_train_context(restored_backend)
        assert load_run(
            restored_backend, str(tmp_path), "h05_projection") == [1]
        restored_graph = CandidateProjectionGraph(
            restored_context.graph_ontology, protocol)
        restored = restored_graph.active_for_binding(definition.bindings[0])
        assert len(restored) == 1
        assert restored[0].candidate.definition == expected.candidate.definition
        assert restored[0].state == protocol.active_state
        assert restored[0].history == expected.history
    finally:
        restored_backend.close()


def test_binding_protocol_collision_and_partial_definition_write_fail_closed():
    """协议复用和部分拓扑均在追加新 statement 前失败。"""
    backend = DictBackend()
    try:
        engine = _engine()
        protocol = _projection_protocol()
        _, graph = _graph(backend, protocol)
        colliding = _definition(1, predicate=protocol.event_candidate)
        colliding_hypothesis = engine.register(colliding)
        baseline = _statement_count(backend)
        with pytest.raises(ValueError, match="lifecycle predicate"):
            graph.define(
                colliding,
                colliding_hypothesis,
                provenance_kind=SOURCE_BARE_TEXT,
                epistemic_origin=EPI_STRUCTURED,
            )
        assert _statement_count(backend) == baseline

        definition = _definition(2)
        hypothesis = engine.register(definition)
        candidate = graph.ontology.materialize(definition.candidate)
        first_binding = definition.bindings[0]
        graph.ontology.relate(
            graph.ontology.materialize(first_binding.predicate),
            candidate,
            graph.ontology.materialize(first_binding.value),
            scope=hypothesis.scope,
            provenance_kind=SOURCE_BARE_TEXT,
            epistemic_origin=EPI_STRUCTURED,
            qualifiers=(first_binding.ordinal,),
        )
        partial_count = _statement_count(backend)
        with pytest.raises(CandidateProjectionError, match="部分图拓扑"):
            graph.define(
                definition,
                hypothesis,
                provenance_kind=SOURCE_BARE_TEXT,
                epistemic_origin=EPI_STRUCTURED,
            )
        assert _statement_count(backend) == partial_count
    finally:
        backend.close()


def test_legacy_like_binding_cannot_activate_without_lifecycle_event():
    """任意旧 tier 或宽边拓扑都不能替代 typed promotion Event。"""
    backend = DictBackend()
    try:
        _, definition, hypothesis, _, _, graph, _ = _projector(backend)
        graph.define(
            definition,
            hypothesis,
            provenance_kind=SOURCE_BARE_TEXT,
            epistemic_origin=EPI_STRUCTURED,
        )
        legacy_predicate = graph.ontology.materialize(
            concept_identity((1701, 1)))
        graph.ontology.relate(
            legacy_predicate,
            graph.ontology.resolve(definition.candidate),
            graph.ontology.materialize(concept_identity((1801, 1))),
            scope=hypothesis.scope,
            provenance_kind=SOURCE_BARE_TEXT,
            epistemic_origin=EPI_STRUCTURED,
        )

        assert graph.active_for_binding(definition.bindings[0]) == ()
    finally:
        backend.close()
