"""A-03 同原文跨 ParserVersion revision、归档和事务回滚测试。"""
from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from pure_integer_ai.cognition.shared.formal_artifact import ArtifactSchema
from pure_integer_ai.cognition.shared.graph_ontology import (
    relation_concept_identity,
)
from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EvidenceRecord,
    HypothesisKey,
    HypothesisLedger,
    LIFECYCLE_ACTIVE,
    LIFECYCLE_ARCHIVED,
)
from pure_integer_ai.cognition.shared.hypothesis_resolution import (
    HypothesisResolver,
)
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    ParserVersion,
    SourceRef,
    VersionBundle,
    concept_identity,
    minimal_instruction_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.parser_revision import (
    ParserAnchorRevision,
    ParserHypothesisRevision,
    ParserRevisionError,
    ParserRevisionGraph,
    ParserRevisionProtocol,
    ParserRevisionRequest,
)
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.cognition.shared.semantic_graph import (
    AtomicPropositionPredicates,
    SemanticGraph,
)
from pure_integer_ai.cognition.shared.semantic_object import (
    entity_identity,
    role_identity,
)
from pure_integer_ai.cognition.shared.training_hypothesis import (
    TrainingHypothesisEventSink,
    TrainingHypothesisHistoryProtocol,
)
from pure_integer_ai.cognition.understanding.occurrence_index import (
    OccurrenceIndex,
    OccurrenceProtocol,
)
from pure_integer_ai.cognition.understanding.semantic_builder import (
    SemanticBindingSpec,
    SemanticBuildPlan,
    SemanticBuilderProtocol,
    SemanticCandidateBuilder,
    SemanticCandidateLedgerAdapter,
    SemanticFillerSpec,
    SemanticGuidanceEvidence,
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
from pure_integer_ai.experiments.parser_revision_runtime import (
    ParserRevisionRuntime,
    ParserRevisionRuntimeError,
    ParserRevisionTrainingSink,
)
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.storage.backend import (
    DictBackend,
    SQLiteBackend,
    StorageBackend,
)
from pure_integer_ai.storage.edge_store import SOURCE_BARE_TEXT
from pure_integer_ai.storage.source_record import SourceRecordRepository
from pure_integer_ai.storage.telemetry import collect_backend_telemetry


@dataclass
class _World:
    """集中保存一次真实 S-02 双 parser fixture 的运行对象。"""

    backend: StorageBackend
    runtime: ParserRevisionRuntime
    graph: ParserRevisionGraph
    ledger: HypothesisLedger
    resolver: HypothesisResolver
    request: ParserRevisionRequest
    old_source: SourceRef
    new_source: SourceRef
    old_hypotheses: tuple[HypothesisKey, ...]
    new_hypothesis: HypothesisKey
    unaffected: HypothesisKey


def _source(parser: int) -> SourceRef:
    """构造除 ParserVersion 外完全相同的来源身份。"""
    return SourceRef(
        SOURCE_BARE_TEXT,
        24301,
        1,
        GLOBAL_OWNER_SCOPE,
        VersionBundle(parser=ParserVersion(parser)),
    )


def _revision_protocol(*, max_targets: int = 4) -> ParserRevisionProtocol:
    """注入 A-03 Artifact 分类、图关系、来源元数据和局部预算。"""
    kinds = tuple(concept_identity((24310, item)) for item in range(1, 4))
    relations = tuple(
        relation_concept_identity((24311, item)) for item in range(1, 7))
    return ParserRevisionProtocol(
        kinds[0],
        kinds[1],
        kinds[2],
        ArtifactSchema(
            concept_identity((24312, 1)),
            concept_identity((24312, 2)),
        ),
        *relations,
        7,
        2,
        1,
        (13, 17),
        8,
        8,
        max_targets,
    )


def _semantic_graph(context) -> SemanticCandidateGraphAdapter:
    """在当前 ontology 上装配互异的 S-00 定义边和 S-02 provenance 边。"""
    refs = tuple(
        context.graph_ontology.materialize(
            relation_concept_identity((24320, item)))
        for item in range(1, 10)
    )
    return SemanticCandidateGraphAdapter(
        SemanticGraph(
            context.graph_ontology,
            AtomicPropositionPredicates(*refs[:6]),
        ),
        SemanticBuilderTracePredicates(*refs[6:]),
    )


def _plan(source: SourceRef, *, variant: int, competition: int) -> SemanticBuildPlan:
    """构造一个由 S-02 编译的来源化 Proposition 候选计划。"""
    upstream = HypothesisKey(
        (24330, 1),
        (24330, variant),
        (24330, competition),
        document_scope(source),
        source,
    )
    return SemanticBuildPlan(
        upstream,
        (24331, variant),
        (),
        (SemanticPropositionSpec(
            (24332, variant),
            (24333, competition),
            concept_identity((24334, variant)),
            structure_concept_identity((24335, 1)),
            (SemanticBindingSpec(
                role_identity((24336, 1)),
                SemanticFillerSpec(
                    external=entity_identity(source, (variant,))),
            ),),
        ),),
    )


def _world(*, raw_new: str = "甲乙", new_evidence: bool = True,
           max_targets: int = 4, old_count: int = 2,
           backend_kind: str = "dict", training_sink: bool = False) -> _World:
    """建立同一 Core backend 上的 SourceRecord、Occurrence/Span、S-02 和 A-03。"""
    if type(old_count) is not int or old_count <= 0:
        raise ValueError("A-03 fixture old_count 必须为严格正整数")
    if backend_kind == "dict":
        backend = DictBackend()
    elif backend_kind == "sqlite":
        backend = SQLiteBackend(":memory:")
    else:
        raise ValueError("未知 A-03 fixture backend")
    context = make_train_context(backend)
    repository = SourceRecordRepository(backend)
    old_source = _source(1)
    new_source = _source(2)
    repository.put(old_source.stable_key(), "甲乙")
    repository.put(new_source.stable_key(), raw_new)

    occurrences = OccurrenceIndex(
        context.graph_ontology,
        context.scoped_identity_store,
        OccurrenceProtocol((24340, 1), (24340, 2)),
    )
    spans = SpanIndex(
        context.graph_ontology,
        context.scoped_identity_store,
        SpanProtocol((24341, 1), (24341, 2), (24341, 3), (24341, 4)),
        occurrences,
    )
    old_occurrence = occurrences.record(
        source=old_source,
        raw_text="甲乙",
        scope=document_scope(old_source),
        start=0,
        end=1,
        ordinal=0,
        segment_index=0,
        local_index=0,
        document_index=0,
    ).occurrence
    new_occurrence = occurrences.record(
        source=new_source,
        raw_text=raw_new,
        scope=document_scope(new_source),
        start=0,
        end=1,
        ordinal=0,
        segment_index=0,
        local_index=0,
        document_index=0,
    ).occurrence
    old_occurrence_second = occurrences.record(
        source=old_source,
        raw_text="甲乙",
        scope=document_scope(old_source),
        start=1,
        end=2,
        ordinal=0,
        segment_index=0,
        local_index=1,
        document_index=1,
    ).occurrence
    new_occurrence_second = occurrences.record(
        source=new_source,
        raw_text=raw_new,
        scope=document_scope(new_source),
        start=1,
        end=2,
        ordinal=0,
        segment_index=0,
        local_index=1,
        document_index=1,
    ).occurrence
    old_span = spans.ensure_ref(
        source=old_source,
        raw_text="甲乙",
        scope=document_scope(old_source),
        members=((0, 2),),
    )
    new_span = spans.ensure_ref(
        source=new_source,
        raw_text=raw_new,
        scope=document_scope(new_source),
        members=((0, len(raw_new)),),
    )

    builder = SemanticCandidateBuilder(
        spans,
        SemanticBuilderProtocol(
            minimal_instruction_identity((24342, 1)),
            (24342, 2),
        ),
        occurrences,
    )
    old_results = tuple(
        builder.compile(
            old_span,
            _plan(old_source, variant=index + 1, competition=1),
        )
        for index in range(old_count)
    )
    new_result = builder.compile(
        new_span,
        _plan(new_source, variant=old_count + 1, competition=1),
    )
    unaffected_result = builder.compile(
        old_span,
        _plan(old_source, variant=old_count + 2, competition=2),
    )
    semantic_graph = _semantic_graph(context)
    for result in (*old_results, new_result, unaffected_result):
        semantic_graph.materialize(
            result,
            provenance_kind=5,
            epistemic_origin=1,
            content_version=1,
        )

    sink = None
    if training_sink:
        if context.training_candidate_history is None:
            raise RuntimeError("A-03 fixture 缺少 Core training history")
        routed = []
        for ordinal, source in enumerate((old_source, new_source), start=1):
            protocol = TrainingHypothesisHistoryProtocol(
                (24344, ordinal),
                (24342, 2),
                source,
                document_scope(source),
            )
            routed.append((source, TrainingHypothesisEventSink(
                context.training_candidate_history, protocol)))
        sink = ParserRevisionTrainingSink(tuple(routed))
    ledger = HypothesisLedger(sink)
    adapter = SemanticCandidateLedgerAdapter(ledger)
    evidence_id = 1000
    for result in (*old_results, unaffected_result):
        hypothesis = result.propositions[0].hypothesis
        adapter.register_unknown(result, (SemanticGuidanceEvidence(
            hypothesis,
            evidence_id,
            (24343, evidence_id),
            result.source,
            1,
        ),))
        evidence_id += 1
    new_hypothesis = new_result.propositions[0].hypothesis
    guidance = ()
    if new_evidence:
        guidance = (SemanticGuidanceEvidence(
            new_hypothesis,
            2001,
            (24343, 2001),
            new_source,
            2,
        ),)
    adapter.register_unknown(new_result, guidance)
    hypotheses = tuple(
        result.propositions[0].hypothesis for result in old_results)
    unaffected = unaffected_result.propositions[0].hypothesis
    for hypothesis in (*hypotheses, new_hypothesis, unaffected):
        context.graph_ontology.materialize(hypothesis.object_identity())

    protocol = _revision_protocol(max_targets=max_targets)
    graph = ParserRevisionGraph(context.graph_ontology, protocol)
    resolver = HypothesisResolver(ledger, sink=sink)
    request = ParserRevisionRequest(
        old_source,
        new_source,
        document_scope(old_source),
        document_scope(new_source),
        (24350, 1),
        (
            ParserAnchorRevision(
                context.graph_ontology.identity_of(old_occurrence),
                (
                    context.graph_ontology.identity_of(new_occurrence),
                    context.graph_ontology.identity_of(
                        new_occurrence_second),
                ),
            ),
            ParserAnchorRevision(
                context.graph_ontology.identity_of(old_occurrence_second),
                (context.graph_ontology.identity_of(
                    new_occurrence_second),),
            ),
            ParserAnchorRevision(
                context.graph_ontology.identity_of(old_span),
                (),
            ),
        ),
        tuple(
            ParserHypothesisRevision(
                hypothesis,
                (new_hypothesis,),
                EvidenceRecord(
                    3001 + index,
                    hypothesis,
                    EVIDENCE_REFUTE,
                    (24351, index),
                    new_source,
                    10,
                ),
            )
            for index, hypothesis in enumerate(hypotheses)
        ),
        (concept_identity((24352, 1)),),
        minimal_instruction_identity((24353, 1)),
        11,
        (24354, 1),
    )
    runtime = ParserRevisionRuntime(repository, graph, ledger, resolver)
    return _World(
        backend,
        runtime,
        graph,
        ledger,
        resolver,
        request,
        old_source,
        new_source,
        hypotheses,
        new_hypothesis,
        unaffected,
    )


@pytest.mark.parametrize("backend_kind", ("dict", "sqlite"))
def test_real_s02_revision_archives_old_and_exact_replay_is_zero_write(
        backend_kind):
    """真实 S-02 产物跨版本映射；旧候选归档、新候选保留，精确重放零新增。"""
    world = _world(
        backend_kind=backend_kind,
        training_sink=backend_kind == "sqlite",
    )
    try:
        unaffected_before = (
            world.ledger.snapshot(world.unaffected),
            world.ledger.evidence_history(world.unaffected),
        )
        result = world.runtime.apply(world.request)

        assert not result.replayed
        assert len(result.decisions) == 1
        assert len(result.materialized.anchor_mappings) == 3
        assert all(
            world.ledger.snapshot(item).lifecycle == LIFECYCLE_ARCHIVED
            for item in world.old_hypotheses)
        assert world.ledger.snapshot(
            world.new_hypothesis).lifecycle == LIFECYCLE_ACTIVE
        assert all(
            transition.replacement is None
            for item in world.old_hypotheses
            for transition in world.ledger.transition_history(item))
        assert world.graph.lineages()[0].old_source == world.old_source
        assert world.graph.lineages()[0].new_source == world.new_source
        assert unaffected_before == (
            world.ledger.snapshot(world.unaffected),
            world.ledger.evidence_history(world.unaffected),
        )

        backend_before = world.backend.recovery_state_snapshot()
        ledger_before = world.ledger.state_key()
        resolver_before = world.resolver.state_key()
        replay = world.runtime.apply(world.request)
        assert replay.replayed
        assert world.backend.recovery_state_snapshot() == backend_before
        assert world.ledger.state_key() == ledger_before
        assert world.resolver.state_key() == resolver_before
    finally:
        world.backend.close()


def test_source_text_missing_old_candidate_and_new_evidence_fail_before_write():
    """原文漂移、旧竞争组漏项和新候选无 Evidence 均在 revision 首写前失败。"""
    cases = (
        (_world(raw_new="甲丙"), None, "原文"),
        (_world(), "omit", "遗漏"),
        (_world(new_evidence=False), None, "Evidence"),
    )
    for world, mutation, message in cases:
        try:
            request = world.request
            if mutation == "omit":
                request = replace(
                    request, hypotheses=request.hypotheses[:1])
            backend_before = world.backend.recovery_state_snapshot()
            ledger_before = world.ledger.state_key()
            resolver_before = world.resolver.state_key()
            with pytest.raises(ParserRevisionRuntimeError, match=message):
                world.runtime.apply(request)
            assert world.backend.recovery_state_snapshot() == backend_before
            assert world.ledger.state_key() == ledger_before
            assert world.resolver.state_key() == resolver_before
        finally:
            world.backend.close()


def test_anchor_source_lineage_and_target_budget_fail_closed():
    """错误 anchor 来源、跨 lineage 与单 mapping 目标超预算不得留下图或历史。"""
    world = _world(max_targets=1)
    try:
        with pytest.raises(ParserRevisionError, match="new anchor"):
            replace(
                world.request,
                anchors=(ParserAnchorRevision(
                    world.request.anchors[0].old,
                    (world.request.anchors[0].old,),
                ),),
            )
        with pytest.raises(ParserRevisionError, match="lineage"):
            replace(
                world.request,
                new_source=SourceRef(
                    SOURCE_BARE_TEXT,
                    99999,
                    1,
                    GLOBAL_OWNER_SCOPE,
                    VersionBundle(parser=ParserVersion(2)),
                ),
            )

        oversized = replace(
            world.request,
            anchors=(ParserAnchorRevision(
                world.request.anchors[0].old,
                (
                    world.request.anchors[0].replacements[0],
                    world.request.anchors[1].replacements[0],
                ),
            ),),
        )
        before = world.backend.recovery_state_snapshot()
        with pytest.raises(ParserRevisionError, match="target 超预算"):
            world.runtime.apply(oversized)
        assert world.backend.recovery_state_snapshot() == before
    finally:
        world.backend.close()


def test_partial_graph_and_commit_failure_preserve_call_state(monkeypatch):
    """既有半图 fail closed；图写后 H-04 异常会回滚 backend、ledger 和 resolver。"""
    partial = _world()
    try:
        partial.graph.ontology.materialize(
            partial.request.revision_identity(partial.graph.protocol))
        before = partial.backend.recovery_state_snapshot()
        with pytest.raises(ParserRevisionError, match="缺少 mapping"):
            partial.runtime.apply(partial.request)
        assert partial.backend.recovery_state_snapshot() == before
    finally:
        partial.backend.close()

    world = _world()
    try:
        backend_before = world.backend.recovery_state_snapshot()
        ledger_before = world.ledger.state_key()
        resolver_before = world.resolver.state_key()

        def fail_resolve(*args, **kwargs):
            """模拟 graph 已写而宿主 H-04 提交失败。"""
            del args, kwargs
            raise RuntimeError("injected H-04 failure")

        monkeypatch.setattr(world.resolver, "resolve", fail_resolve)
        with pytest.raises(RuntimeError, match="injected"):
            world.runtime.apply(world.request)
        assert world.backend.recovery_state_snapshot() == backend_before
        assert world.ledger.state_key() == ledger_before
        assert world.resolver.state_key() == resolver_before
        assert world.graph.preflight(world.request) is None
    finally:
        world.backend.close()


def test_evaluation_clone_commits_only_to_clone():
    """A-03 evaluation clone 独立写自身 backend 和历史，宿主保持逐表、逐事件不变。"""
    world = _world()
    try:
        backend_before = world.backend.recovery_state_snapshot()
        ledger_before = world.ledger.state_key()
        resolver_before = world.resolver.state_key()
        cloned = world.runtime.clone_for_evaluation()
        result = cloned.apply(world.request)

        assert not result.replayed
        assert world.backend.recovery_state_snapshot() == backend_before
        assert world.ledger.state_key() == ledger_before
        assert world.resolver.state_key() == resolver_before
        assert cloned.graph.lineages()
        assert all(
            cloned.ledger.snapshot(item).lifecycle == LIFECYCLE_ARCHIVED
            for item in world.old_hypotheses)
        cloned.graph.ontology.backend.close()
    finally:
        world.backend.close()


def test_v02_mapping_growth_is_linear_and_hash_seed_independent():
    """V-02 synthetic：影响集翻倍时图/history 操作和新增行不呈二次增长。"""
    points = []
    snapshots = []
    for count in (2, 4, 8):
        world = _world(old_count=count)
        try:
            before = world.backend.snapshot()
            with collect_backend_telemetry() as telemetry:
                result = world.runtime.apply(world.request)
            operations = telemetry.operation_snapshot()
            calls = sum(value[0] for value in operations.values())
            after = world.backend.snapshot()
            growth = sum(
                len(after[table]) - len(before[table])
                for table in after)
            points.append((count, calls, growth, result.stable_key()))
            if count == 4:
                snapshots.append(after)
        finally:
            world.backend.close()

    call_first = points[1][1] - points[0][1]
    call_second = points[2][1] - points[1][1]
    growth_first = points[1][2] - points[0][2]
    growth_second = points[2][2] - points[1][2]
    assert call_first > 0
    assert call_second <= call_first * 3
    assert growth_first > 0
    assert growth_second <= growth_first * 3

    replay = _world(old_count=4)
    try:
        replay.runtime.apply(replay.request)
        snapshots.append(replay.backend.snapshot())
    finally:
        replay.backend.close()
    assert snapshots[0] == snapshots[1]
