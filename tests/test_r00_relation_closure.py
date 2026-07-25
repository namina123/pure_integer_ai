"""R-00 统一关系闭环、故障消融、恢复和 held-out 隔离测试。"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from pure_integer_ai.cognition.shared.alias_resolution import (
    AliasResolutionProtocol,
    AliasResolutionResult,
    AliasResolutionSelector,
    AliasRoute,
    AliasRouteSearchBudget,
    AliasRouteSearchExhausted,
    AliasRouteStep,
    ReferenceResolutionQuery,
    SurfaceRealizationQuery,
)
from pure_integer_ai.cognition.shared.candidate_projection import (
    CandidateProjectionGraph,
    CandidateProjectionProtocol,
)
from pure_integer_ai.cognition.shared.candidate_runtime import (
    CandidateHistoryUnavailableError,
    CandidateLearningRuntime,
    CandidateProjectionMetadata,
)
from pure_integer_ai.cognition.shared.candidate_verifier import (
    IndependentObjectVerifier,
    IndependentVerifierProtocol,
    RevealedObjectObservation,
)
from pure_integer_ai.cognition.shared.evidence_candidate import (
    EvidenceCandidateEngine,
    EvidenceCandidateProtocol,
)
from pure_integer_ai.cognition.shared.graph_ontology import (
    relation_concept_identity,
)
from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EVIDENCE_UNKNOWN,
)
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    OBJECT_CONCEPT,
    OBJECT_ENTITY,
    OBJECT_LANGUAGE_BRANCH,
    OBJECT_OCCURRENCE,
    OBJECT_REPRESENTATION,
    SourceRef,
    VersionBundle,
    concept_identity,
    language_branch_identity,
    minimal_instruction_identity,
    occurrence_identity,
    representation_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.relation_algebra import RelationAlgebra
from pure_integer_ai.cognition.shared.relation_closure import (
    ActiveRelationClosureConsumer,
    RelationClosureCandidateSpec,
    RelationClosureField,
    RelationClosureProtocol,
)
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.cognition.shared.semantic_graph import (
    AtomicPropositionPredicates,
    SemanticGraph,
)
from pure_integer_ai.cognition.shared.semantic_object import (
    AtomicPropositionDefinition,
    AtomicRoleBinding,
    context_scope_identity,
    entity_identity,
    proposition_identity,
    role_identity,
)
from pure_integer_ai.cognition.shared.typed_relation import (
    RelationSchema,
    RelationSchemaError,
    RelationSlotSchema,
    SymmetricRule,
)
from pure_integer_ai.experiments.evaluation_isolation import clone_backend
from pure_integer_ai.experiments.evaluation_protocol import ProtocolKey
from pure_integer_ai.experiments.alias_relation_runtime import (
    AliasRelationRuntime,
)
from pure_integer_ai.experiments.formal_train import make_train_context
from pure_integer_ai.experiments.relation_closure_runtime import (
    RelationClosureIncompleteError,
    RelationClosurePerformanceCounters,
    RelationClosurePerformanceWindow,
    RelationClosureRecognitionInput,
    RelationClosureRuntime,
)
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import EPI_STRUCTURED, SOURCE_BARE_TEXT


def _source(source_id: int) -> SourceRef:
    """构造共享 owner/version、来源身份彼此独立的测试 SourceRef。"""
    return SourceRef(
        131,
        source_id,
        0,
        GLOBAL_OWNER_SCOPE,
        VersionBundle(),
    )


def _semantic_predicate_identities() -> tuple:
    """返回 S-00 原子命题拓扑使用的六个注入 predicate 身份。"""
    return tuple(
        relation_concept_identity((8100, ordinal))
        for ordinal in range(1, 7)
    )


def _semantic_graph(ontology) -> SemanticGraph:
    """在给定 ontology 上安装可跨 dump/clone 重建的 S-00 协议。"""
    refs = tuple(
        ontology.materialize(identity)
        for identity in _semantic_predicate_identities()
    )
    return SemanticGraph(ontology, AtomicPropositionPredicates(*refs))


def _projection_protocol() -> CandidateProjectionProtocol:
    """构造互不复用字段的 H-05 lifecycle 图协议。"""
    values = tuple(concept_identity((8200, ordinal)) for ordinal in range(13))
    return CandidateProjectionProtocol(
        values[0],
        values[1],
        values[2],
        values[3],
        values[4],
        values[5],
        values[6],
        values[7],
        values[8],
        values[9],
        values[10],
        values[11],
        values[12],
        (8201, 1),
    )


def _evidence_protocol() -> EvidenceCandidateProtocol:
    """构造要求两个 forming 来源的独立 aggregate H-05 owner。"""
    aggregate = _source(900)
    return EvidenceCandidateProtocol(
        (8300, 1),
        (8300, 2),
        aggregate,
        document_scope(aggregate),
        2,
    )


def _verifier() -> IndependentObjectVerifier:
    """构造不读取候选图、只消费显式 reveal 的三态 verifier。"""
    return IndependentObjectVerifier(IndependentVerifierProtocol(
        concept_identity((8400, 1)),
        (8400, 2),
        (8400, 3),
        (8400, 4),
        (8400, 5),
    ))


def _relation_protocol() -> RelationClosureProtocol:
    """构造 relation/schema 两个互异的候选字段协议。"""
    return RelationClosureProtocol(
        RelationClosureField(concept_identity((8500, 1))),
        RelationClosureField(concept_identity((8500, 2))),
    )


def _candidate_runtime(graph: CandidateProjectionGraph) -> CandidateLearningRuntime:
    """把共享 H-05 engine、verifier 和候选图装成关系 owner。"""
    return CandidateLearningRuntime(
        EvidenceCandidateEngine(_evidence_protocol()),
        graph,
        _verifier(),
        CandidateProjectionMetadata(
            SOURCE_BARE_TEXT,
            EPI_STRUCTURED,
        ),
    )


def _definition_and_schema(source: SourceRef, *, family: int = 8600):
    """构造一个完全注入 relation/Role/schema 的二元 Entity 命题。"""
    relation = relation_concept_identity((family, 1))
    left_role = role_identity((family, 2))
    right_role = role_identity((family, 3))
    schema = RelationSchema(
        structure_concept_identity((family, 4)),
        relation,
        (
            RelationSlotSchema(
                left_role, frozenset({OBJECT_ENTITY}), 1, 1),
            RelationSlotSchema(
                right_role, frozenset({OBJECT_ENTITY}), 1, 1),
        ),
    )
    definition = AtomicPropositionDefinition(
        proposition_identity(source, (family, 5)),
        relation,
        occurrence_identity(source, start=0, end=2, ordinal=0),
        context_scope_identity(source, (family, 6)),
        (
            AtomicRoleBinding(
                left_role, entity_identity(source, (family, 7))),
            AtomicRoleBinding(
                right_role, entity_identity(source, (family, 8))),
        ),
    )
    return definition, schema


@dataclass
class _ClosureFixture:
    """集中保存一套可关闭、可 clone 的 R-00 测试设施。"""

    backend: DictBackend
    semantic_graph: SemanticGraph
    candidate_graph: CandidateProjectionGraph
    candidate_runtime: CandidateLearningRuntime
    consumer: ActiveRelationClosureConsumer
    runtime: RelationClosureRuntime
    spec: RelationClosureCandidateSpec

    def close(self) -> None:
        """关闭当前测试后端。"""
        self.backend.close()


def _fixture() -> _ClosureFixture:
    """创建已物化 S-00 命题、但尚未 forming 的完整 R-00 环境。"""
    backend = DictBackend()
    ctx = make_train_context(backend)
    semantic_graph = _semantic_graph(ctx.graph_ontology)
    candidate_graph = CandidateProjectionGraph(
        ctx.graph_ontology, _projection_protocol())
    candidate_runtime = _candidate_runtime(candidate_graph)
    definition, schema = _definition_and_schema(_source(1))
    semantic_graph.define_atomic(
        definition,
        scope=document_scope(definition.source),
        provenance_kind=SOURCE_BARE_TEXT,
        epistemic_origin=EPI_STRUCTURED,
    )
    protocol = _relation_protocol()
    consumer = ActiveRelationClosureConsumer(
        semantic_graph,
        candidate_graph,
        protocol,
        (schema,),
        engine=candidate_runtime.engine,
    )
    runtime = RelationClosureRuntime(
        candidate_runtime,
        semantic_graph,
        consumer,
        protocol,
    )
    spec = RelationClosureCandidateSpec(
        definition,
        schema,
        (8700, 1),
        (_source(2), _source(3)),
    )
    return _ClosureFixture(
        backend,
        semantic_graph,
        candidate_graph,
        candidate_runtime,
        consumer,
        runtime,
        spec,
    )


def _recognition(
        fixture: _ClosureFixture, source_id: int, *,
        stance: str, archive_refuted: bool = False,
        ) -> RelationClosureRecognitionInput:
    """构造带 held-out partition、Occurrence anchor 和显式三态 reveal 的输入。"""
    observation = _source(source_id)
    anchor = occurrence_identity(
        observation, start=0, end=2, ordinal=0)
    proposition = fixture.spec.proposition.proposition
    supported = (proposition,) if stance == "support" else ()
    refuted = (proposition,) if stance == "refute" else ()
    return RelationClosureRecognitionInput(
        proposition,
        observation,
        document_scope(observation),
        ProtocolKey((8800, source_id)),
        (8801, source_id),
        anchor,
        (anchor,),
        RevealedObjectObservation(
            observation,
            document_scope(observation),
            (8801, source_id),
            _source(700 + source_id),
            supported_targets=supported,
            refuted_targets=refuted,
            trace=(8802, source_id),
        ),
        archive_refuted=archive_refuted,
    )


def _cloned_graphs(fixture: _ClosureFixture):
    """复制正式后端，并在副本上重建同一 SemanticGraph/CandidateGraph facade。"""
    backend = clone_backend(fixture.backend)
    ctx = make_train_context(backend)
    semantic_graph = _semantic_graph(ctx.graph_ontology)
    candidate_graph = CandidateProjectionGraph(
        ctx.graph_ontology,
        fixture.candidate_graph.protocol,
    )
    return backend, semantic_graph, candidate_graph


def test_full_relation_closure_requires_real_consumer_and_reports_each_stage():
    """完整链必须真实经过 writer、Evidence、H-04、active 投影和 use 归因。"""
    fixture = _fixture()
    try:
        formation = fixture.runtime.form(fixture.spec)
        assert formation.hypothesis == (
            fixture.spec.candidate_definition(_relation_protocol())
            .hypothesis(_evidence_protocol()))
        with pytest.raises(RelationClosureIncompleteError, match="evidence"):
            fixture.runtime.require_complete(fixture.spec)

        trace = fixture.runtime.recognize(
            _recognition(fixture, 4, stance="support"))
        assert trace.outcome.verification.stance == EVIDENCE_SUPPORT
        assert trace.active_fact is not None
        algebra_fact = trace.active_fact.as_algebra_fact()
        left_role = fixture.spec.schema.slots[0].role
        right_role = fixture.spec.schema.slots[1].role
        derived = RelationAlgebra((fixture.spec.schema,)).derive_active_candidates(
            (algebra_fact,),
            (SymmetricRule(
                minimal_instruction_identity((8850, 1)),
                fixture.spec.proposition.predicate,
                left_role,
                right_role,
            ),),
        )
        assert len(derived) == 1
        assert derived[0].premises[0].hypothesis == formation.hypothesis
        assert derived[0].premises[0].support_evidence_ids == (
            algebra_fact.snapshot.support_evidence_ids)
        assert fixture.consumer.lookup_relation(
            fixture.spec.proposition.predicate,
            schema=fixture.spec.schema.schema,
        ) == (trace.active_fact,)
        with pytest.raises(RelationClosureIncompleteError, match="consumer"):
            fixture.runtime.require_complete(fixture.spec)

        use = fixture.runtime.consume(
            fixture.spec.proposition.proposition,
            use_key=(8900, 1),
        )
        audit = fixture.runtime.require_complete(fixture.spec)
        assert audit.complete is True
        assert use.evidence_keys == trace.active_fact.evidence_keys
        assert use.decision_key == trace.active_fact.decision_key

        before = RelationClosurePerformanceCounters(10, 20, 30, 40)
        after = RelationClosurePerformanceCounters(13, 25, 32, 47)
        window = RelationClosurePerformanceWindow.between(
            before, after, elapsed_ns=1000)
        report = fixture.runtime.report(window)
        assert report.formation_count == 1
        assert report.recognition_count == 1
        assert report.support_count == 1
        assert report.refute_count == 0
        assert report.unknown_count == 0
        assert report.consumer_use_count == 1
        assert report.performance.backend_queries == 3
        assert report.performance.graph_statements == 7
    finally:
        fixture.close()


def test_missing_reveal_target_stays_unknown_and_never_becomes_false():
    """verifier 未覆盖 Proposition 时只追加 unknown，不因图缺边晋升或反驳。"""
    fixture = _fixture()
    try:
        fixture.runtime.form(fixture.spec)
        trace = fixture.runtime.recognize(
            _recognition(fixture, 5, stance="unknown"))
        audit = fixture.runtime.audit(fixture.spec)
        assert trace.outcome.verification.stance == EVIDENCE_UNKNOWN
        assert trace.outcome.projection is None
        assert fixture.consumer.lookup_proposition(
            fixture.spec.proposition.proposition) == ()
        assert audit.writer_defined is True
        assert audit.recognition_evidence is True
        assert audit.resolver_adopted is False
        assert audit.active_projection is False
        assert "resolver" in audit.missing
    finally:
        fixture.close()


def test_typed_schema_semantics_are_not_shared_by_same_shaped_relations():
    """相同二元 shape 的另一 relation/schema 不能替换当前候选语义。"""
    fixture = _fixture()
    try:
        other_definition, other_schema = _definition_and_schema(
            _source(10), family=8610)
        with pytest.raises(RelationSchemaError, match="predicate"):
            RelationClosureCandidateSpec(
                fixture.spec.proposition,
                other_schema,
                (8700, 2),
                (_source(11), _source(12)),
            )
        other_spec = RelationClosureCandidateSpec(
            other_definition,
            other_schema,
            (8700, 2),
            (_source(11), _source(12)),
        )
        current = fixture.spec.candidate_definition(_relation_protocol())
        other = other_spec.candidate_definition(_relation_protocol())
        assert current.competition_key != other.competition_key
        assert current.bindings != other.bindings
        assert current.stable_key() != other.stable_key()
        assert fixture.runtime.audit(fixture.spec).writer_defined is False
    finally:
        fixture.close()


def test_writer_disconnection_makes_probe_fail_even_when_engine_registered(
        monkeypatch):
    """切断候选图 writer 后，H-05 内存登记和 forming 计数不能冒充闭环。"""
    fixture = _fixture()
    try:
        monkeypatch.setattr(
            fixture.candidate_graph,
            "define",
            lambda *args, **kwargs: None,
        )
        fixture.runtime.form(fixture.spec)
        audit = fixture.runtime.audit(fixture.spec)
        assert audit.writer_defined is False
        assert "writer" in audit.missing
        with pytest.raises(RelationClosureIncompleteError, match="writer"):
            fixture.runtime.require_complete(fixture.spec)
        with pytest.raises(LookupError, match="active"):
            fixture.runtime.consume(
                fixture.spec.proposition.proposition,
                use_key=(8910, 1),
            )
    finally:
        fixture.close()


def test_promoter_disconnection_makes_probe_fail_after_support(
        monkeypatch):
    """切断 H-04 到图投影后，support 与 adopted 仍不能被 consumer 读取。"""
    fixture = _fixture()
    try:
        formation = fixture.runtime.form(fixture.spec)
        monkeypatch.setattr(
            fixture.candidate_runtime,
            "sync_competition",
            lambda hypothesis, timestamp_seq: ((hypothesis, None),),
        )
        trace = fixture.runtime.recognize(
            _recognition(fixture, 6, stance="support"))
        audit = fixture.runtime.audit(fixture.spec)
        assert trace.outcome.verification.stance == EVIDENCE_SUPPORT
        assert trace.outcome.projection is None
        assert fixture.candidate_runtime.engine.active(
            formation.hypothesis) is not None
        assert audit.resolver_adopted is True
        assert audit.active_projection is False
        assert "projection" in audit.missing
        with pytest.raises(RelationClosureIncompleteError, match="projection"):
            fixture.runtime.require_complete(fixture.spec)
    finally:
        fixture.close()


def test_consumer_disconnection_makes_probe_fail_after_active_projection(
        monkeypatch):
    """切断正式 consumer 后，active 图边和生命周期翻转仍不能算实际采用。"""
    fixture = _fixture()
    try:
        fixture.runtime.form(fixture.spec)
        fixture.runtime.recognize(
            _recognition(fixture, 7, stance="support"))
        monkeypatch.setattr(
            fixture.consumer,
            "require_proposition",
            lambda proposition: (_ for _ in ()).throw(
                LookupError("consumer disconnected")),
        )
        with pytest.raises(LookupError, match="disconnected"):
            fixture.runtime.consume(
                fixture.spec.proposition.proposition,
                use_key=(8920, 1),
            )
        audit = fixture.runtime.audit(fixture.spec)
        assert audit.writer_defined is True
        assert audit.recognition_evidence is True
        assert audit.resolver_adopted is True
        assert audit.active_projection is True
        assert audit.consumer_used is False
        with pytest.raises(RelationClosureIncompleteError, match="consumer"):
            fixture.runtime.require_complete(fixture.spec)
    finally:
        fixture.close()


def test_graph_recovery_is_read_only_and_preserves_evidence_decision_trace():
    """backend clone 后可只读恢复 active 事实，但缺持久历史时不能伪续写 H-00。"""
    fixture = _fixture()
    clone = None
    try:
        fixture.runtime.form(fixture.spec)
        fixture.runtime.recognize(
            _recognition(fixture, 8, stance="support"))
        live = fixture.consumer.require_proposition(
            fixture.spec.proposition.proposition)
        clone, semantic_graph, candidate_graph = _cloned_graphs(fixture)
        recovered_consumer = ActiveRelationClosureConsumer(
            semantic_graph,
            candidate_graph,
            _relation_protocol(),
            (fixture.spec.schema,),
            engine=None,
        )
        recovered = recovered_consumer.require_proposition(
            fixture.spec.proposition.proposition)
        assert recovered.read_only_recovered is True
        assert recovered.proposition == live.proposition
        assert recovered.schema == live.schema
        assert recovered.evidence_keys == live.evidence_keys
        assert recovered.decision_key == live.decision_key

        fresh_candidate = _candidate_runtime(candidate_graph)
        live_consumer = ActiveRelationClosureConsumer(
            semantic_graph,
            candidate_graph,
            _relation_protocol(),
            (fixture.spec.schema,),
            engine=fresh_candidate.engine,
        )
        fresh_runtime = RelationClosureRuntime(
            fresh_candidate,
            semantic_graph,
            live_consumer,
            _relation_protocol(),
        )
        with pytest.raises(
                CandidateHistoryUnavailableError,
                match="持久历史"):
            fresh_runtime.form(fixture.spec)
    finally:
        if clone is not None:
            clone.close()
        fixture.close()


def test_held_out_clone_can_demote_without_mutating_training_owner():
    """held-out 副本追加负证据和降级时，正式 backend、owner 和 active fact 不变。"""
    fixture = _fixture()
    clone = None
    try:
        fixture.runtime.form(fixture.spec)
        fixture.runtime.recognize(
            _recognition(fixture, 9, stance="support"))
        host_backend = fixture.backend.snapshot()
        host_state = fixture.runtime.state_key()
        clone, semantic_graph, candidate_graph = _cloned_graphs(fixture)
        evaluation = fixture.runtime.clone_for_evaluation(
            semantic_graph,
            candidate_graph,
        )
        trace = evaluation.recognize(_recognition(
            fixture,
            10,
            stance="refute",
            archive_refuted=True,
        ))
        assert trace.outcome.verification.stance == EVIDENCE_REFUTE
        assert evaluation.consumer.lookup_proposition(
            fixture.spec.proposition.proposition) == ()
        assert fixture.consumer.require_proposition(
            fixture.spec.proposition.proposition).active_candidate is not None
        assert fixture.backend.snapshot() == host_backend
        assert fixture.runtime.state_key() == host_state
    finally:
        if clone is not None:
            clone.close()
        fixture.close()


def test_relation_consume_many_is_atomic_before_any_use_write():
    """同批 use key 冲突必须在写账前失败，不能留下部分 consumer use。"""
    fixture = _fixture()
    try:
        fixture.runtime.form(fixture.spec)
        fixture.runtime.recognize(
            _recognition(fixture, 20, stance="support"))
        proposition = fixture.spec.proposition.proposition

        with pytest.raises(
                RelationClosureIncompleteError, match="同批"):
            fixture.runtime.consume_many((
                (proposition, (8990, 1)),
                (proposition, (8990, 1)),
            ))

        assert fixture.runtime.audit(fixture.spec).consumer_used is False
    finally:
        fixture.close()


def test_relation_batch_rejects_last_duplicate_before_any_write():
    """forming 或 recognition 末项重复时，整批必须在首写前失败。"""
    fixture = _fixture()
    try:
        before_form = fixture.backend.snapshot(), fixture.runtime.state_key()
        with pytest.raises(
                RelationClosureIncompleteError, match="同批 relation forming"):
            fixture.runtime.form_many((
                (fixture.spec, 0),
                (fixture.spec, 0),
            ))
        assert (fixture.backend.snapshot(), fixture.runtime.state_key()) == (
            before_form)

        fixture.runtime.form(fixture.spec)
        recognition = _recognition(fixture, 21, stance="support")
        before_recognition = (
            fixture.backend.snapshot(), fixture.runtime.state_key())
        with pytest.raises(
                RelationClosureIncompleteError, match="同批 relation recognition"):
            fixture.runtime.recognize_many_at((
                (recognition, 10, 11, 12),
                (recognition, 13, 14, 15),
            ))
        assert (fixture.backend.snapshot(), fixture.runtime.state_key()) == (
            before_recognition)
    finally:
        fixture.close()


def _r01_definition(
        source: SourceRef,
        *,
        family: int,
        relation,
        schema_identity,
        role_fillers: tuple,
        ):
    """按 filler 实际对象类型构造开放 n 元 R-01 relation fact。"""
    schema = RelationSchema(
        schema_identity,
        relation,
        tuple(
            RelationSlotSchema(
                role,
                frozenset({filler.object_kind}),
                1,
                1,
            )
            for role, filler in role_fillers
        ),
    )
    definition = AtomicPropositionDefinition(
        proposition_identity(source, (family, 1)),
        relation,
        occurrence_identity(source, start=0, end=1, ordinal=0),
        context_scope_identity(source, (family, 2)),
        tuple(
            AtomicRoleBinding(role, filler)
            for role, filler in role_fillers
        ),
    )
    return definition, schema


def _r01_recognition(spec, source_id: int):
    """为任意 R-01 spec 构造独立 support reveal。"""
    observation = _source(source_id)
    anchor = occurrence_identity(
        observation, start=0, end=1, ordinal=0)
    proposition = spec.proposition.proposition
    return RelationClosureRecognitionInput(
        proposition,
        observation,
        document_scope(observation),
        ProtocolKey((9200, source_id)),
        (9201, source_id),
        anchor,
        (anchor,),
        RevealedObjectObservation(
            observation,
            document_scope(observation),
            (9201, source_id),
            _source(700 + source_id),
            supported_targets=(proposition,),
            trace=(9202, source_id),
        ),
    )


@dataclass
class _R01Fixture:
    """保存 R-01 route 测试的 R-00 owner、事实和图对象。"""

    backend: DictBackend
    semantic_graph: SemanticGraph
    candidate_graph: CandidateProjectionGraph
    closure: RelationClosureRuntime
    alias_runtime: AliasRelationRuntime
    protocol: AliasResolutionProtocol
    facts: dict
    specs: dict
    objects: dict

    def close(self) -> None:
        """关闭测试后端。"""
        self.backend.close()


def _r01_fixture() -> _R01Fixture:
    """构造 alias、refers 和两个 realizes 选项的完整 active R-00 闭环。"""
    backend = DictBackend()
    ctx = make_train_context(backend)
    semantic_graph = _semantic_graph(ctx.graph_ontology)
    candidate_graph = CandidateProjectionGraph(
        ctx.graph_ontology, _projection_protocol())
    candidate_runtime = _candidate_runtime(candidate_graph)
    closure_protocol = _relation_protocol()

    alias_relation = relation_concept_identity((9300, 1))
    refers_relation = relation_concept_identity((9300, 2))
    realizes_relation = relation_concept_identity((9300, 3))
    alias_roles = (role_identity((9301, 1)), role_identity((9301, 2)))
    refers_roles = (role_identity((9301, 3)), role_identity((9301, 4)))
    realizes_roles = (
        role_identity((9301, 5)),
        role_identity((9301, 6)),
        role_identity((9301, 7)),
    )
    alias_schema = structure_concept_identity((9302, 1))
    refers_schema = structure_concept_identity((9302, 2))
    realizes_schema = structure_concept_identity((9302, 3))

    concept_a = concept_identity((9303, 1))
    concept_b = concept_identity((9303, 2))
    concept_c = concept_identity((9303, 3))
    concept_d = concept_identity((9303, 4))
    concept_e = concept_identity((9303, 5))
    branch = language_branch_identity((9304, 1))
    representation_one = representation_identity(
        (9305, 1), (0x7532, 0x4E00))
    representation_two = representation_identity(
        (9305, 1), (0x7532, 0x4E8C))
    reference_occurrence = occurrence_identity(
        _source(930), start=0, end=1, ordinal=0)

    definitions = {}
    definitions["alias"] = _r01_definition(
        _source(31),
        family=9310,
        relation=alias_relation,
        schema_identity=alias_schema,
        role_fillers=(
            (alias_roles[0], concept_a),
            (alias_roles[1], concept_b),
        ),
    )
    definitions["refers"] = _r01_definition(
        _source(32),
        family=9320,
        relation=refers_relation,
        schema_identity=refers_schema,
        role_fillers=(
            (refers_roles[0], reference_occurrence),
            (refers_roles[1], concept_a),
        ),
    )
    definitions["real_b"] = _r01_definition(
        _source(33),
        family=9330,
        relation=realizes_relation,
        schema_identity=realizes_schema,
        role_fillers=(
            (realizes_roles[0], concept_b),
            (realizes_roles[1], representation_one),
            (realizes_roles[2], branch),
        ),
    )
    definitions["real_a"] = _r01_definition(
        _source(34),
        family=9340,
        relation=realizes_relation,
        schema_identity=realizes_schema,
        role_fillers=(
            (realizes_roles[0], concept_a),
            (realizes_roles[1], representation_two),
            (realizes_roles[2], branch),
        ),
    )
    definitions["real_c"] = _r01_definition(
        _source(35),
        family=9350,
        relation=realizes_relation,
        schema_identity=realizes_schema,
        role_fillers=(
            (realizes_roles[0], concept_c),
            (realizes_roles[1], representation_one),
            (realizes_roles[2], branch),
        ),
    )
    definitions["alias_de"] = _r01_definition(
        _source(36),
        family=9360,
        relation=alias_relation,
        schema_identity=alias_schema,
        role_fillers=(
            (alias_roles[0], concept_d),
            (alias_roles[1], concept_e),
        ),
    )
    definitions["real_e"] = _r01_definition(
        _source(37),
        family=9370,
        relation=realizes_relation,
        schema_identity=realizes_schema,
        role_fillers=(
            (realizes_roles[0], concept_e),
            (realizes_roles[1], representation_one),
            (realizes_roles[2], branch),
        ),
    )
    schemas = tuple({
        schema.schema: schema
        for _, schema in definitions.values()
    }.values())
    for definition, _ in definitions.values():
        semantic_graph.define_atomic(
            definition,
            scope=document_scope(definition.source),
            provenance_kind=SOURCE_BARE_TEXT,
            epistemic_origin=EPI_STRUCTURED,
        )
    inactive_realization, _ = _r01_definition(
        _source(38),
        family=9380,
        relation=realizes_relation,
        schema_identity=realizes_schema,
        role_fillers=(
            (realizes_roles[0], concept_c),
            (realizes_roles[1], representation_two),
            (realizes_roles[2], branch),
        ),
    )
    semantic_graph.define_atomic(
        inactive_realization,
        scope=document_scope(inactive_realization.source),
        provenance_kind=SOURCE_BARE_TEXT,
        epistemic_origin=EPI_STRUCTURED,
    )
    consumer = ActiveRelationClosureConsumer(
        semantic_graph,
        candidate_graph,
        closure_protocol,
        schemas,
        engine=candidate_runtime.engine,
    )
    closure = RelationClosureRuntime(
        candidate_runtime,
        semantic_graph,
        consumer,
        closure_protocol,
    )
    specs = {}
    facts = {}
    for index, name in enumerate(sorted(definitions), start=1):
        definition, schema = definitions[name]
        spec = RelationClosureCandidateSpec(
            definition,
            schema,
            (9360, index),
            (_source(40), _source(41)),
        )
        specs[name] = spec
        closure.form(spec)
        trace = closure.recognize(_r01_recognition(spec, 50 + index))
        assert trace.active_fact is not None
        facts[name] = trace.active_fact

    protocol = AliasResolutionProtocol(
        alias_relation,
        (alias_schema,),
        *alias_roles,
        minimal_instruction_identity((9370, 1)),
        refers_relation,
        (refers_schema,),
        *refers_roles,
        minimal_instruction_identity((9370, 2)),
        realizes_relation,
        (realizes_schema,),
        *realizes_roles,
        minimal_instruction_identity((9370, 3)),
        minimal_instruction_identity((9370, 4)),
        minimal_instruction_identity((9370, 5)),
        minimal_instruction_identity((9370, 6)),
    )
    selector = AliasResolutionSelector(protocol)
    return _R01Fixture(
        backend,
        semantic_graph,
        candidate_graph,
        closure,
        AliasRelationRuntime(closure, selector),
        protocol,
        facts,
        specs,
        {
            "a": concept_a,
            "b": concept_b,
            "c": concept_c,
            "d": concept_d,
            "e": concept_e,
            "branch": branch,
            "rep1": representation_one,
            "rep2": representation_two,
            "occurrence": reference_occurrence,
        },
    )


def test_r01_reference_and_surface_selection_use_active_typed_routes():
    """方向同指和跨 alias 词形选择保留 R-00 Evidence/use 完整归因。"""
    fixture = _r01_fixture()
    try:
        p = fixture.protocol
        o = fixture.objects
        budget = AliasRouteSearchBudget(20, 20, 20)
        reference = fixture.alias_runtime.resolve_reference(
            o["occurrence"],
            target_kinds=(OBJECT_CONCEPT,),
            budget=budget,
            use_key=(9380, 1),
        )
        assert reference.result.outcome == p.selected_outcome
        assert reference.result.selected.value == o["a"]
        assert len(reference.relation_uses) == 1
        assert fixture.closure.require_complete(
            fixture.specs["refers"]).complete

        surface_route = AliasRoute(o["d"], (
            AliasRouteStep(
                p.alias_step,
                fixture.facts["alias_de"],
                o["d"],
                o["e"],
            ),
            AliasRouteStep(
                p.realizes_step,
                fixture.facts["real_e"],
                o["e"],
                o["rep1"],
            ),
        ))
        surface = fixture.alias_runtime.select_surface(
            o["d"],
            o["branch"],
            budget=budget,
            use_key=(9380, 2),
        )
        assert surface.result.outcome == p.selected_outcome
        assert surface.result.selected.value == o["rep1"]
        assert len(surface.relation_uses) == 2
        assert fixture.closure.require_complete(
            fixture.specs["alias_de"]).complete
        assert fixture.closure.require_complete(
            fixture.specs["real_e"]).complete

        other_branch = language_branch_identity((9381, 1))
        with pytest.raises(ValueError, match="分支"):
            fixture.alias_runtime.selector.select_surface(
                SurfaceRealizationQuery(
                    o["d"], other_branch, (surface_route,)))

        missing = fixture.alias_runtime.select_surface(
            o["rep2"],
            o["branch"],
            budget=budget,
            use_key=(9380, 3),
        )
        assert missing.result.outcome == p.missing_outcome
        assert not missing.relation_uses
    finally:
        fixture.close()


def test_r01_preserves_word_form_ambiguity_and_homograph_target_identity():
    """多 Representation 不私选，同一 Representation 也不合并不同语义起点。"""
    fixture = _r01_fixture()
    try:
        p = fixture.protocol
        o = fixture.objects
        budget = AliasRouteSearchBudget(20, 20, 20)
        ambiguous = fixture.alias_runtime.select_surface(
            o["a"],
            o["branch"],
            budget=budget,
            use_key=(9390, 1),
        )
        assert ambiguous.result.outcome == p.ambiguous_outcome
        assert ambiguous.result.selected is None
        assert {item.value for item in ambiguous.result.options} == {
            o["rep1"], o["rep2"]}
        assert not ambiguous.relation_uses
        assert fixture.closure.audit(
            fixture.specs["real_a"]).consumer_used is False

        selected = fixture.alias_runtime.select_surface(
            o["c"],
            o["branch"],
            budget=budget,
            use_key=(9390, 2),
        )
        assert selected.result.origin == o["c"]
        assert selected.result.selected.value == o["rep1"]
        assert selected.result.origin != o["a"]
        assert {
            fact.proposition.proposition
            for fact in selected.discovery.considered_facts
        } == {fixture.specs["real_c"].proposition.proposition}

        with pytest.raises(ValueError, match="outcome"):
            AliasResolutionResult(
                p,
                p.selected_outcome,
                o["a"],
                o["branch"],
                ambiguous.result.options,
            )
        with pytest.raises(ValueError, match="origin"):
            AliasResolutionResult(
                p,
                p.ambiguous_outcome,
                o["c"],
                o["branch"],
                ambiguous.result.options,
            )
    finally:
        fixture.close()


def test_r01_active_discovery_budget_failure_cannot_commit_partial_choice():
    """候选 route 超预算时必须整体失败，不能把首个词形伪装成唯一结果。"""
    fixture = _r01_fixture()
    try:
        before = fixture.alias_runtime.state_key()
        with pytest.raises(AliasRouteSearchExhausted, match="route 数"):
            fixture.alias_runtime.select_surface(
                fixture.objects["a"],
                fixture.objects["branch"],
                budget=AliasRouteSearchBudget(20, 20, 1),
                use_key=(9395, 1),
            )
        assert fixture.alias_runtime.state_key() == before
        assert fixture.closure.audit(
            fixture.specs["real_a"]).consumer_used is False
        assert fixture.closure.audit(
            fixture.specs["real_b"]).consumer_used is False
    finally:
        fixture.close()


def test_r01_active_discovery_uses_local_binding_index(monkeypatch):
    """surface 发现必须按当前 filler 局部反查，不能退回全 relation 扫描。"""
    fixture = _r01_fixture()
    try:
        monkeypatch.setattr(
            fixture.closure.consumer,
            "lookup_relation",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("不得全局扫描 relation")),
        )
        selected = fixture.alias_runtime.select_surface(
            fixture.objects["c"],
            fixture.objects["branch"],
            budget=AliasRouteSearchBudget(20, 20, 20),
            use_key=(9395, 2),
        )
        assert selected.result.selected.value == fixture.objects["rep1"]
    finally:
        fixture.close()


def test_r01_surface_prefix_policy_is_explicit_and_traceable():
    """调用方可限制 alias/refers 前缀，direct-only 不得暗中解引用。"""
    fixture = _r01_fixture()
    try:
        p = fixture.protocol
        o = fixture.objects
        budget = AliasRouteSearchBudget(20, 20, 20)

        direct = fixture.alias_runtime.preview_surface(
            o["occurrence"],
            o["branch"],
            budget=budget,
            allowed_prefix_steps=(),
        )
        assert direct.result.outcome == p.missing_outcome
        assert direct.discovery.allowed_prefix_steps == ()
        assert not direct.discovery.considered_facts

        dereferenced = fixture.alias_runtime.preview_surface(
            o["occurrence"],
            o["branch"],
            budget=budget,
            allowed_prefix_steps=(p.refers_step,),
        )
        assert dereferenced.result.selected.value == o["rep2"]
        assert dereferenced.discovery.allowed_prefix_steps == (p.refers_step,)
        assert all(
            step.instruction in (p.refers_step, p.realizes_step)
            for route in dereferenced.discovery.routes
            for step in route.steps
        )

        default = fixture.alias_runtime.preview_surface(
            o["occurrence"], o["branch"], budget=budget)
        assert default.result.outcome == p.ambiguous_outcome
        assert set(default.discovery.allowed_prefix_steps) == {
            p.alias_step, p.refers_step}
    finally:
        fixture.close()


def test_r01_held_out_clone_surface_use_does_not_mutate_training_owner():
    """held-out route 发现和 use 只写克隆 owner，宿主 backend 与账本保持不变。"""
    fixture = _r01_fixture()
    clone = None
    try:
        host_backend = fixture.backend.snapshot()
        host_closure = fixture.closure.state_key()
        host_alias = fixture.alias_runtime.state_key()
        clone, semantic_graph, candidate_graph = _cloned_graphs(fixture)
        closure = fixture.closure.clone_for_evaluation(
            semantic_graph, candidate_graph)
        runtime = fixture.alias_runtime.clone_for_runtime(closure)

        selected = runtime.select_surface(
            fixture.objects["c"],
            fixture.objects["branch"],
            budget=AliasRouteSearchBudget(20, 20, 20),
            use_key=(9396, 1),
        )

        assert selected.result.selected.value == fixture.objects["rep1"]
        assert len(selected.relation_uses) == 1
        assert runtime.state_key() != host_alias
        assert fixture.backend.snapshot() == host_backend
        assert fixture.closure.state_key() == host_closure
        assert fixture.alias_runtime.state_key() == host_alias
    finally:
        if clone is not None:
            clone.close()
        fixture.close()


def test_r01_rejects_relation_kind_confusion_and_reverse_reference():
    """refers fact 不能冒充 strict alias，方向性同指也不能反向读取。"""
    fixture = _r01_fixture()
    try:
        p = fixture.protocol
        o = fixture.objects
        wrong_kind = AliasRoute(o["occurrence"], (
            AliasRouteStep(
                p.alias_step,
                fixture.facts["refers"],
                o["occurrence"],
                o["a"],
            ),
        ))
        with pytest.raises(ValueError, match="alias step"):
            fixture.alias_runtime.selector.resolve_reference(
                ReferenceResolutionQuery(
                    o["occurrence"], (wrong_kind,)))

        reverse = AliasRoute(o["a"], (
            AliasRouteStep(
                p.refers_step,
                fixture.facts["refers"],
                o["a"],
                o["occurrence"],
            ),
        ))
        with pytest.raises(ValueError, match="不得反向"):
            fixture.alias_runtime.selector.resolve_reference(
                ReferenceResolutionQuery(o["a"], (reverse,)))
    finally:
        fixture.close()
