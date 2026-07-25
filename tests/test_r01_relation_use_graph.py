"""R-01 PH2 Core Use Event 的持久化、恢复、隔离和篡改对抗测试。"""
from __future__ import annotations

import pytest

from dataclasses import replace

from pure_integer_ai.cognition.shared.alias_resolution import (
    AliasResolutionSelector,
)
from pure_integer_ai.cognition.shared.generation_surface import (
    GenerationSurfaceAttribution,
)
from pure_integer_ai.cognition.shared.hypothesis import HypothesisKey
from pure_integer_ai.cognition.shared.identity import (
    OwnerScope,
    VISIBILITY_SESSION,
    concept_identity,
    language_branch_identity,
    minimal_instruction_identity,
    representation_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.relation_use import (
    RelationUseContext,
    RelationUseDefinition,
    RelationUseGraph,
    RelationUseGraphProtocol,
    RelationUseIntegrityError,
    RelationUseOwner,
    RelationUseWriteMetadata,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    SCOPE_DOCUMENT,
    document_scope,
    episode_scope,
    make_scope,
    query_scope,
)
from pure_integer_ai.cognition.shared.semantic_object import (
    proposition_identity,
    semantic_source,
)
from pure_integer_ai.experiments.evaluation_isolation import (
    clone_backend,
    clone_train_context,
)
from pure_integer_ai.experiments.alias_relation_runtime import (
    AliasRelationRuntime,
)
from pure_integer_ai.experiments.formal_train import make_train_context
from pure_integer_ai.experiments.generation_surface_runtime import (
    GenerationSurfaceRuntime,
)
from pure_integer_ai.experiments.relation_closure_runtime import (
    RelationClosureIncompleteError,
    RelationClosureRuntime,
)
from pure_integer_ai.storage.backend import DictBackend, SQLiteBackend
from pure_integer_ai.storage.assertion_record import (
    ASSERTION_QUALIFIER_TABLE,
    ASSERTION_RECORD_TABLE,
)
from pure_integer_ai.storage.edge_store import EPI_STRUCTURED, SOURCE_BARE_TEXT
from pure_integer_ai.storage.graph_statement import GRAPH_STATEMENT_TABLE

from tests.test_r00_relation_closure import (
    _cloned_graphs,
    _fixture,
    _recognition,
    _source,
)
from tests.test_g03_generation_surface import (
    _alias_fixture,
    _manual_execution,
    _request_for,
    _structure_plan,
    _templates,
)


_BASE = 18100


def _protocol(seed: int = _BASE) -> RelationUseGraphProtocol:
    """构造不写死消费者或用途语义的开放 Use 图协议。"""
    identities = tuple(
        concept_identity((seed, ordinal)) for ordinal in range(1, 9))
    return RelationUseGraphProtocol(
        *identities,
        (seed, 9),
    )


def _metadata() -> RelationUseWriteMetadata:
    """构造测试注入的 Core 训练来源元数据。"""
    return RelationUseWriteMetadata(
        SOURCE_BARE_TEXT,
        EPI_STRUCTURED,
        content_version=3,
        qualifiers=(_BASE, 10),
    )


def _definition(
        source_id: int, *, use_key: tuple[int, ...] = (_BASE, 20),
        decision_suffix: int = 0,
        read_only_recovered: bool = False,
        ) -> RelationUseDefinition:
    """构造来源化 query、独立 relation Proposition 和完整 H-00/H-04 归因。"""
    query_source = _source(source_id)
    fact_source = _source(source_id + 100)
    aggregate = _source(900)
    hypothesis = HypothesisKey(
        (_BASE, 21),
        (_BASE, 22, source_id),
        (_BASE, 23),
        document_scope(aggregate),
        aggregate,
    )
    return RelationUseDefinition(
        use_key,
        RelationUseContext(
            query_source,
            document_scope(query_source),
            concept_identity((_BASE, 24)),
            minimal_instruction_identity((_BASE, 25)),
        ),
        proposition_identity(fact_source, (_BASE, 26, source_id)),
        hypothesis,
        (
            (_BASE, 27, source_id),
            (_BASE, 28, source_id),
        ),
        (_BASE, 29, source_id, decision_suffix),
        read_only_recovered,
    )


@pytest.mark.parametrize("backend_type", (DictBackend, SQLiteBackend))
def test_core_use_graph_roundtrips_on_dict_and_sqlite(backend_type):
    """Dict 与 SQLite 都能从图恢复完整 Use，且重放不增加 statement。"""
    backend = backend_type()
    try:
        ctx = make_train_context(backend)
        graph = RelationUseGraph(ctx.graph_ontology, _protocol())
        owner = RelationUseOwner(graph, _metadata())
        definition = _definition(1)

        first = owner.append_many((definition,))[0]
        statement_count = backend.count(GRAPH_STATEMENT_TABLE)
        second = owner.append_many((definition,))[0]
        recovered = RelationUseOwner(
            RelationUseGraph(ctx.graph_ontology, _protocol()),
            _metadata(),
        )

        assert first == second
        assert recovered.history() == (first,)
        assert backend.count(GRAPH_STATEMENT_TABLE) == statement_count
        assert first.definition.evidence_keys == definition.evidence_keys
        assert first.definition.decision_key == definition.decision_key
    finally:
        backend.close()


def test_local_use_key_is_scoped_by_full_query_context_and_conflicts_are_atomic():
    """相同局部键可跨 context 使用，同一路由竞争则在任何新增 statement 前失败。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        owner = RelationUseOwner(
            RelationUseGraph(ctx.graph_ontology, _protocol()),
            _metadata(),
        )
        first = _definition(2)
        second = _definition(3, use_key=first.use_key)
        owner.append_many((first, second))
        before = backend.count(GRAPH_STATEMENT_TABLE)

        with pytest.raises(RelationUseIntegrityError, match="不同采用事实"):
            owner.append_many((
                _definition(
                    2,
                    use_key=first.use_key,
                    decision_suffix=1,
                ),
            ))

        assert len(owner.history()) == 2
        assert owner.history()[0].event != owner.history()[1].event
        assert backend.count(GRAPH_STATEMENT_TABLE) == before
    finally:
        backend.close()


def test_core_use_v06_keeps_knowledge_source_and_writes_only_eval_scope():
    """V-06 clone 的 Use Event 保留知识来源，运行 assertion 仅属于评测 scope。"""
    host_backend = DictBackend()
    evaluation_backend = None
    try:
        host_context = make_train_context(host_backend)
        host_before = host_backend.snapshot()
        evaluation_backend = clone_backend(host_backend)
        evaluation_context = clone_train_context(
            host_context,
            evaluation_backend,
            label="relation-use-owner-only-scope",
        )
        if evaluation_context.scope_owner is None:
            raise RuntimeError("V-06 clone 缺少独立评测 owner")
        evaluation_document = make_scope(
            SCOPE_DOCUMENT,
            _BASE + 31,
            owner=evaluation_context.scope_owner,
        )
        evaluation_episode = episode_scope(
            _BASE + 32,
            parent=evaluation_document,
        )
        evaluation_query = query_scope(
            _BASE + 33,
            parent=evaluation_episode,
        )
        definition = _definition(31)
        context = replace(definition.context, scope=evaluation_query)
        definition = replace(definition, context=context)
        graph = RelationUseGraph(
            evaluation_context.graph_ontology,
            _protocol(),
        )
        materialized = RelationUseOwner(graph, _metadata()).append_many((
            definition,
        ))[0]

        assert evaluation_query.source is None
        assert context.source != evaluation_query.source
        assert RelationUseContext.from_stable_key(context.stable_key()) == context
        assert semantic_source(materialized.event) == context.source
        statements = graph.ontology.statements(subject=materialized.event_ref)
        assert statements
        assert all(item.assertion.scope == evaluation_query for item in statements)
        assert host_backend.snapshot() == host_before

        with pytest.raises(ValueError, match="显式 source scope"):
            RelationUseContext(
                context.source,
                document_scope(_source(32)),
                context.consumer,
                context.purpose,
            )
    finally:
        if evaluation_backend is not None:
            evaluation_backend.close()
        host_backend.close()


def test_incremental_append_uses_event_lookup_instead_of_rescanning_history(
        monkeypatch):
    """续写只按 Event 路由局部查询，禁止每次提交扫描全部历史导致 O(n²)。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        graph = RelationUseGraph(ctx.graph_ontology, _protocol())
        owner = RelationUseOwner(graph, _metadata())
        owner.append_many((_definition(30),))

        def _unexpected_history_scan():
            """一旦增量预检错误回退全历史扫描就立即失败。"""
            raise AssertionError("增量 Core Use 提交不得扫描全部历史")

        monkeypatch.setattr(graph, "history", _unexpected_history_scan)
        owner.append_many((_definition(31),))

        assert len(owner.history()) == 2
    finally:
        backend.close()


def test_partial_snapshot_tamper_fails_closed_instead_of_repairing_history():
    """删除 snapshot 后同一路由不得被静默补写或当作未发生。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        graph = RelationUseGraph(ctx.graph_ontology, _protocol())
        definition = _definition(4)
        materialized = RelationUseOwner(
            graph, _metadata()).append_many((definition,))[0]
        snapshot_ref = ctx.graph_ontology.resolve(
            graph.protocol.event_snapshot)
        event_ref = ctx.graph_ontology.resolve(materialized.event)
        snapshot_rows = ctx.graph_ontology.statements(
            predicate=snapshot_ref,
            subject=event_ref,
        )
        assert len(snapshot_rows) == 1
        snapshot_hash = snapshot_rows[0].assertion_hash
        raw = backend.snapshot()
        raw[GRAPH_STATEMENT_TABLE] = [
            row for row in raw[GRAPH_STATEMENT_TABLE]
            if row["assertion_hash"] != snapshot_hash
        ]
        raw[ASSERTION_RECORD_TABLE] = [
            row for row in raw[ASSERTION_RECORD_TABLE]
            if row["identity_hash"] != snapshot_hash
        ]
        raw[ASSERTION_QUALIFIER_TABLE] = [
            row for row in raw[ASSERTION_QUALIFIER_TABLE]
            if row["identity_hash"] != snapshot_hash
        ]
        backend.load_snapshot(raw)
        ctx.graph_ontology.clear_runtime_caches()
        damaged = RelationUseOwner(
            RelationUseGraph(ctx.graph_ontology, _protocol()),
            _metadata(),
        )

        with pytest.raises(RelationUseIntegrityError, match="snapshot"):
            damaged.append_many((definition,))
    finally:
        backend.close()


def test_relation_runtime_requires_context_and_v06_recovers_independent_owner():
    """正式 owner 缺 context 必须失败，V-06 从 clone 图恢复后只写克隆历史。"""
    fixture = _fixture()
    cloned_backend = None
    try:
        fixture.runtime.form(fixture.spec)
        fixture.runtime.recognize(
            _recognition(fixture, 5, stance="support"))
        owner = RelationUseOwner(
            RelationUseGraph(fixture.semantic_graph.ontology, _protocol()),
            _metadata(),
        )
        runtime = RelationClosureRuntime(
            fixture.candidate_runtime,
            fixture.semantic_graph,
            fixture.consumer,
            fixture.runtime.protocol,
            owner,
        )
        proposition = fixture.spec.proposition.proposition
        context = _definition(5).context

        with pytest.raises(RelationClosureIncompleteError, match="Context"):
            runtime.consume(proposition, use_key=(_BASE, 30))
        host_use = runtime.consume(
            proposition,
            use_key=(_BASE, 30),
            context=context,
        )
        assert host_use.event is not None
        assert len(owner.history()) == 1

        cloned_backend, semantic_graph, candidate_graph = _cloned_graphs(fixture)
        cloned = runtime.clone_for_evaluation(
            semantic_graph,
            candidate_graph,
        )
        assert cloned.use_owner is not owner
        assert cloned.use_owner.state_key() == owner.state_key()
        cloned.consume(
            proposition,
            use_key=(_BASE, 30),
            context=_definition(6).context,
        )

        assert len(owner.history()) == 1
        assert len(cloned.use_owner.history()) == 2
    finally:
        if cloned_backend is not None:
            cloned_backend.close()
        fixture.close()


def test_clone_backend_preserves_core_use_event_without_memory_use():
    """跨 run Core clone 恢复 Event 本体，不依赖宿主 owner 的进程字典。"""
    backend = DictBackend()
    cloned_backend = None
    try:
        ctx = make_train_context(backend)
        protocol = _protocol()
        owner = RelationUseOwner(
            RelationUseGraph(ctx.graph_ontology, protocol),
            _metadata(),
        )
        materialized = owner.append_many((
            _definition(7, read_only_recovered=True),
        ))[0]

        cloned_backend = clone_backend(backend)
        cloned_ctx = make_train_context(cloned_backend)
        cloned_owner = RelationUseOwner(
            RelationUseGraph(cloned_ctx.graph_ontology, protocol),
            _metadata(),
        )

        assert cloned_owner.history()[0].definition == materialized.definition
        assert cloned_owner.history()[0].event == materialized.event
        assert cloned_owner.graph.ontology is not owner.graph.ontology
        assert cloned_owner.history()[0].event_ref == materialized.event_ref
    finally:
        if cloned_backend is not None:
            cloned_backend.close()
        backend.close()


def test_generation_surface_derives_use_context_from_goal_and_attribution():
    """G-03 不由宿主硬编码 context，而从 query goal 与理论归因传给 R-01。"""
    branch = language_branch_identity((_BASE, 40))
    structure = _structure_plan(branch)
    first, second = _templates(structure)
    family = (_BASE, 41)
    fixture = _alias_fixture(
        branch,
        (
            (first, representation_identity(family, (0x7532,))),
            (second, representation_identity(family, (0x5E8F,))),
        ),
    )
    try:
        owner = RelationUseOwner(
            RelationUseGraph(fixture.semantic_graph.ontology, _protocol()),
            _metadata(),
        )
        closure = RelationClosureRuntime(
            fixture.closure.candidate_runtime,
            fixture.semantic_graph,
            fixture.closure.consumer,
            fixture.closure.protocol,
            owner,
        )
        alias = AliasRelationRuntime(
            closure,
            AliasResolutionSelector(fixture.protocol),
        )
        request = _request_for(
            structure,
            _manual_execution(structure),
            alias,
        )
        attribution = GenerationSurfaceAttribution(
            structure_concept_identity((_BASE, 42)),
            _definition(8).hypothesis,
            minimal_instruction_identity((_BASE, 43)),
        )
        request = replace(request, attribution=attribution)

        run = GenerationSurfaceRuntime(alias).plan(request)

        assert run.complete
        assert len(owner.history()) == 2
        for item in owner.history():
            assert item.definition.context == RelationUseContext(
                request.structure.selection.request.goal.source,
                request.structure.selection.request.goal.scope,
                attribution.theory,
                attribution.purpose,
            )
    finally:
        fixture.close()
