"""R-05 双对称关系 owner、context Use、课程接线和隔离测试。"""
from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from pure_integer_ai.cognition.shared.candidate_projection import (
    CandidateProjectionGraph,
)
from pure_integer_ai.cognition.shared.candidate_runtime import (
    CandidateLearningRuntime,
    CandidateProjectionMetadata,
)
from pure_integer_ai.cognition.shared.candidate_verifier import (
    RevealedObjectObservation,
)
from pure_integer_ai.cognition.shared.evidence_candidate import (
    EvidenceCandidateEngine,
    EvidenceCandidateProtocol,
)
from pure_integer_ai.cognition.shared.graph_ontology import (
    relation_concept_identity,
)
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    OBJECT_CONCEPT,
    OBJECT_ENTITY,
    ObjectIdentity,
    SourceRef,
    VersionBundle,
    concept_identity,
    minimal_instruction_identity,
    occurrence_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.logic_executor import LogicEvidenceState
from pure_integer_ai.cognition.shared.relation_closure import (
    ActiveRelationClosureConsumer,
    RelationClosureCandidateSpec,
)
from pure_integer_ai.cognition.shared.relation_use import RelationUseContext
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.cognition.shared.semantic_object import (
    AtomicPropositionDefinition,
    AtomicRoleBinding,
    context_scope_identity,
    entity_identity,
    proposition_identity,
    role_identity,
)
from pure_integer_ai.cognition.shared.symmetric_relation import (
    SymmetricPairPattern,
    SymmetricRelationBudget,
    SymmetricRelationBudgetExceeded,
    SymmetricRelationProtocol,
)
from pure_integer_ai.cognition.shared.typed_relation import (
    IrreflexiveRule,
    RelationSchema,
    RelationSlotSchema,
    SymmetricRule,
)
from pure_integer_ai.cognition.understanding.occurrence_index import (
    OccurrenceProtocol,
)
from pure_integer_ai.experiments.collection import CollectedItem
from pure_integer_ai.experiments.evaluation_isolation import isolated_evaluation
from pure_integer_ai.experiments.evaluation_protocol import ProtocolKey
from pure_integer_ai.experiments.formal_train import (
    FormalTrainConfig,
    formal_train,
    make_train_context,
)
from pure_integer_ai.experiments.mechanism_inventory import (
    STATUS_OPT_IN,
    inventory_by_id,
    readiness_candidates,
    validate_inventory,
)
from pure_integer_ai.experiments.relation_closure_runtime import (
    RelationClosureIncompleteError,
    RelationClosureRecognitionInput,
    RelationClosureRuntime,
)
from pure_integer_ai.experiments.semantic_pair_runtime import (
    SemanticPairBudget,
    SemanticPairRoundRequest,
    SemanticPairRuntime,
    SemanticPairRuntimeError,
    install_semantic_pair_runtime,
)
from pure_integer_ai.experiments.symmetric_relation_runtime import (
    LegacySymmetricPairRecord,
    MappedLegacySymmetricPair,
    SymmetricChannelBatch,
    SymmetricPairFormationRequest,
    SymmetricPairQuery,
    SymmetricRelationChannelRuntime,
    SymmetricRelationRuntimeError,
)
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import EPI_STRUCTURED, SOURCE_BARE_TEXT
from pure_integer_ai.training.stages import STAGE1_SKELETON
from tests.test_r00_relation_closure import (
    _projection_protocol,
    _relation_protocol,
    _semantic_graph,
    _verifier,
)


def _source(source_id: int) -> SourceRef:
    """构造共享 owner/version 且来源号互异的 R-05 测试来源。"""
    return SourceRef(
        205,
        source_id,
        0,
        GLOBAL_OWNER_SCOPE,
        VersionBundle(),
    )


def _channel_protocol(
        seed: int, *, irreflexive: bool,
        ) -> SymmetricRelationProtocol:
    """构造 relation、逆序 schema slot、对称规则和可选反自反规则。"""
    relation = relation_concept_identity((10700, seed))
    left_role = role_identity((10710, seed, 1))
    right_role = role_identity((10710, seed, 2))
    schema = RelationSchema(
        structure_concept_identity((10720, seed)),
        relation,
        (
            RelationSlotSchema(
                right_role, frozenset({OBJECT_ENTITY}), 1, 1),
            RelationSlotSchema(
                left_role, frozenset({OBJECT_ENTITY}), 1, 1),
        ),
    )
    symmetric = SymmetricRule(
        minimal_instruction_identity((10730, seed, 1)),
        relation,
        left_role,
        right_role,
    )
    rule = None
    if irreflexive:
        rule = IrreflexiveRule(
            minimal_instruction_identity((10730, seed, 2)),
            relation,
            left_role,
            right_role,
        )
    return SymmetricRelationProtocol(
        schema,
        left_role,
        right_role,
        symmetric,
        rule,
    )


def _candidate_runtime(
        graph: CandidateProjectionGraph, seed: int,
        ) -> CandidateLearningRuntime:
    """以独立 hypothesis kind 构造一个 H-05 owner。"""
    aggregate = _source(900 + seed)
    return CandidateLearningRuntime(
        EvidenceCandidateEngine(EvidenceCandidateProtocol(
            (10740, seed, 1),
            (10740, seed, 2),
            aggregate,
            document_scope(aggregate),
            2,
        )),
        graph,
        _verifier(),
        CandidateProjectionMetadata(
            SOURCE_BARE_TEXT,
            EPI_STRUCTURED,
        ),
    )


def _channel_runtime(
        ctx,
        semantic_graph,
        protocol: SymmetricRelationProtocol,
        seed: int,
        budget: SymmetricRelationBudget,
        ) -> SymmetricRelationChannelRuntime:
    """在共享本体上建立一个独立 candidate/closure owner。"""
    candidate_graph = CandidateProjectionGraph(
        ctx.graph_ontology,
        _projection_protocol(),
    )
    candidate_runtime = _candidate_runtime(candidate_graph, seed)
    consumer = ActiveRelationClosureConsumer(
        semantic_graph,
        candidate_graph,
        _relation_protocol(),
        (protocol.schema,),
        engine=candidate_runtime.engine,
    )
    closure = RelationClosureRuntime(
        candidate_runtime,
        semantic_graph,
        consumer,
        _relation_protocol(),
    )
    return SymmetricRelationChannelRuntime(
        closure,
        protocol,
        budget,
    )


@dataclass
class _Fixture:
    """保存双 owner、共享语义图、测试 endpoint 和追加方法。"""

    backend: DictBackend
    ctx: object
    similar_protocol: SymmetricRelationProtocol
    antonym_protocol: SymmetricRelationProtocol
    runtime: SemanticPairRuntime
    objects: tuple[ObjectIdentity, ...]
    next_fact_id: int = 1

    def close(self) -> None:
        """关闭测试后端。"""
        self.backend.close()

    def owner(self, channel: str) -> SymmetricRelationChannelRuntime:
        """按测试 channel 名返回对应独立 owner。"""
        if channel == "similar":
            return self.runtime.similar
        if channel == "antonym":
            return self.runtime.antonym
        raise ValueError("未知测试 channel")

    def definition(
            self,
            channel: str,
            left: ObjectIdentity,
            right: ObjectIdentity,
            source: SourceRef,
            fact_id: int,
            ) -> AtomicPropositionDefinition:
        """按注入 Role 构造保留原始方向的 typed pair Proposition。"""
        protocol = self.owner(channel).protocol
        return AtomicPropositionDefinition(
            proposition_identity(source, (10750, fact_id)),
            protocol.relation,
            occurrence_identity(source, start=0, end=1, ordinal=0),
            context_scope_identity(source, (10751, fact_id)),
            (
                AtomicRoleBinding(protocol.right_role, right),
                AtomicRoleBinding(protocol.left_role, left),
            ),
        )

    def add(
            self,
            channel: str,
            left: int,
            right: int,
            *,
            stance: str,
            archive_refuted: bool = False,
            replacement: ObjectIdentity | None = None,
            competition_key: tuple[int, ...] | None = None,
            ) -> ObjectIdentity:
        """向指定独立 owner 写入一个方向化 pair 候选。"""
        fact_id = self.next_fact_id
        self.next_fact_id += 1
        owner = self.owner(channel)
        source = _source(1000 + fact_id)
        definition = self.definition(
            channel,
            self.objects[left],
            self.objects[right],
            source,
            fact_id,
        )
        spec = RelationClosureCandidateSpec(
            definition,
            owner.protocol.schema,
            competition_key or (10752, fact_id),
            (_source(2000 + fact_id * 2), _source(2001 + fact_id * 2)),
        )
        owner.semantic_graph.define_atomic(
            definition,
            scope=document_scope(source),
            provenance_kind=SOURCE_BARE_TEXT,
            epistemic_origin=EPI_STRUCTURED,
        )
        owner.relation_runtime.form(spec, timestamp_base=fact_id * 10)
        owner.relation_runtime.recognize(_recognition_for(
            definition,
            _source(3000 + fact_id),
            fact_id,
            stance=stance,
            archive_refuted=archive_refuted,
            replacement=replacement,
        ))
        return definition.proposition


def _fixture(
        *,
        channel_budget: SymmetricRelationBudget | None = None,
        total_budget: int = 100,
        ) -> _Fixture:
    """建立共享 SemanticGraph 和两个 hypothesis kind 不同的 R-00 owner。"""
    backend = DictBackend()
    ctx = make_train_context(backend)
    semantic_graph = _semantic_graph(ctx.graph_ontology)
    similar_protocol = _channel_protocol(1, irreflexive=False)
    antonym_protocol = _channel_protocol(2, irreflexive=True)
    budget = channel_budget or SymmetricRelationBudget(100, 50)
    similar = _channel_runtime(
        ctx, semantic_graph, similar_protocol, 1, budget)
    antonym = _channel_runtime(
        ctx, semantic_graph, antonym_protocol, 2, budget)
    runtime = SemanticPairRuntime(
        similar,
        antonym,
        SemanticPairBudget(total_budget),
    )
    objects = tuple(
        entity_identity(_source(100 + index), (10760, index))
        for index in range(8)
    )
    return _Fixture(
        backend,
        ctx,
        similar_protocol,
        antonym_protocol,
        runtime,
        objects,
    )


def _recognition_for(
        definition: AtomicPropositionDefinition,
        observation: SourceRef,
        seed: int,
        *,
        stance: str = "support",
        archive_refuted: bool = False,
        replacement: ObjectIdentity | None = None,
        ) -> RelationClosureRecognitionInput:
    """构造与 forming 分源的显式 support/refute recognition。"""
    anchor = occurrence_identity(
        observation,
        start=0,
        end=1,
        ordinal=0,
    )
    supported = (definition.proposition,) if stance == "support" else ()
    refuted = (definition.proposition,) if stance == "refute" else ()
    return RelationClosureRecognitionInput(
        definition.proposition,
        observation,
        document_scope(observation),
        ProtocolKey((10770, seed)),
        (10771, seed),
        anchor,
        (anchor,),
        RevealedObjectObservation(
            observation,
            document_scope(observation),
            (10771, seed),
            _source(4000 + seed),
            supported_targets=supported,
            refuted_targets=refuted,
            trace=(10772, seed),
        ),
        archive_refuted=archive_refuted,
        replacement=replacement,
    )


def _use_context(source_id: int) -> RelationUseContext:
    """构造带 consumer/purpose 的完整来源化采用 context。"""
    source = _source(source_id)
    return RelationUseContext(
        source,
        document_scope(source),
        concept_identity((10780, source_id, 1)),
        concept_identity((10780, source_id, 2)),
    )


def test_two_independent_kinds_and_symmetric_read_without_physical_double_write():
    """两个 channel kind/ledger 独立，反向查询复用同一 direct Proposition。"""
    fixture = _fixture()
    try:
        similar_prop = fixture.add("similar", 0, 1, stance="support")
        antonym_prop = fixture.add("antonym", 0, 1, stance="support")
        assert fixture.runtime.similar.hypothesis_kind != (
            fixture.runtime.antonym.hypothesis_kind)

        reverse = fixture.runtime.query_similar(SymmetricPairQuery(
            SymmetricPairPattern(fixture.objects[1], fixture.objects[0]),
            use_key=(10790, 1),
            context=_use_context(5000),
        ))
        assert reverse.selection.evaluations[0].state == LogicEvidenceState(
            True, False)
        assert reverse.selection.evaluations[0].evidence[0].left == (
            fixture.objects[0])
        assert {item.proposition for item in reverse.uses} == {similar_prop}
        assert len(fixture.runtime.similar.relation_runtime.formation_traces()) == 1

        antonym = fixture.runtime.query_antonym(SymmetricPairQuery(
            SymmetricPairPattern(fixture.objects[1], fixture.objects[0])
        ))
        assert antonym.selection.evaluations[0].state == LogicEvidenceState(
            True, False)
        assert antonym.selection.evaluations[0].evidence[0].proposition == antonym_prop
        assert reverse.selection.evaluations[0].evidence[0].hypothesis != (
            antonym.selection.evaluations[0].evidence[0].hypothesis)
    finally:
        fixture.close()


def test_dual_runtime_rejects_shared_owner_and_duplicate_hypothesis_kind():
    """两个 relation 名称不能掩盖 facade、closure owner 或 hypothesis kind 共用。"""
    fixture = _fixture()
    try:
        with pytest.raises(SemanticPairRuntimeError, match="不得共用 facade"):
            SemanticPairRuntime(
                fixture.runtime.similar,
                fixture.runtime.similar,
                fixture.runtime.budget,
            )
        duplicate_kind = _channel_runtime(
            fixture.ctx,
            fixture.runtime.similar.semantic_graph,
            fixture.antonym_protocol,
            1,
            fixture.runtime.antonym.budget,
        )
        with pytest.raises(SemanticPairRuntimeError, match="kind 必须不同"):
            SemanticPairRuntime(
                fixture.runtime.similar,
                duplicate_kind,
                fixture.runtime.budget,
            )
        protocol = fixture.similar_protocol
        mismatched = RelationSchema(
            structure_concept_identity((10721, 99)),
            protocol.relation,
            (
                RelationSlotSchema(
                    protocol.left_role,
                    frozenset({OBJECT_ENTITY}),
                    1,
                    1,
                ),
                RelationSlotSchema(
                    protocol.right_role,
                    frozenset({OBJECT_CONCEPT}),
                    1,
                    1,
                ),
            ),
        )
        with pytest.raises(ValueError, match="对象类型必须完全相同"):
            SymmetricRelationProtocol(
                mismatched,
                protocol.left_role,
                protocol.right_role,
                protocol.symmetric_rule,
            )
    finally:
        fixture.close()


def test_opposite_directions_share_four_state_and_conflict_writes_no_use():
    """两个方向聚合同一 pair，support/refute 冲突在任一方向查询都一致。"""
    fixture = _fixture()
    try:
        fixture.add("similar", 0, 1, stance="support")
        fixture.add(
            "similar", 1, 0,
            stance="refute",
            archive_refuted=True,
        )
        for left, right in ((0, 1), (1, 0)):
            result = fixture.runtime.query_similar(SymmetricPairQuery(
                SymmetricPairPattern(
                    fixture.objects[left], fixture.objects[right]),
                use_key=(10791, left + 1),
                context=_use_context(5010 + left),
            ))
            evaluation = result.selection.evaluations[0]
            assert evaluation.state == LogicEvidenceState(True, True)
            assert {item.left for item in evaluation.evidence} == {
                fixture.objects[0], fixture.objects[1]}
            assert result.uses == ()
    finally:
        fixture.close()


def test_no_transitivity_cross_channel_alias_or_implicit_self_rule():
    """缺 pair 保持 unknown；channel 不互推；自环只按各自显式规则裁决。"""
    fixture = _fixture()
    try:
        fixture.add("similar", 0, 1, stance="support")
        fixture.add("similar", 1, 2, stance="support")
        unknown = fixture.runtime.query_similar(SymmetricPairQuery(
            SymmetricPairPattern(fixture.objects[0], fixture.objects[2])
        ))
        assert unknown.selection.evaluations[0].state == LogicEvidenceState(
            False, False)

        other_channel = fixture.runtime.query_antonym(SymmetricPairQuery(
            SymmetricPairPattern(fixture.objects[0], fixture.objects[1])
        ))
        assert other_channel.selection.evaluations[0].state == (
            LogicEvidenceState(False, False))
        assert fixture.similar_protocol.relation != fixture.antonym_protocol.relation

        fixture.add("similar", 3, 3, stance="support")
        similar_self = fixture.runtime.query_similar(SymmetricPairQuery(
            SymmetricPairPattern(fixture.objects[3], fixture.objects[3])
        ))
        assert similar_self.selection.evaluations[0].state == LogicEvidenceState(
            True, False)

        fixture.add("antonym", 4, 4, stance="support")
        antonym_self = fixture.runtime.query_antonym(SymmetricPairQuery(
            SymmetricPairPattern(fixture.objects[4], fixture.objects[4])
        ))
        assert antonym_self.selection.evaluations[0].state == LogicEvidenceState(
            True, True)
        assert fixture.antonym_protocol.irreflexive_rule is not None
    finally:
        fixture.close()


def test_discovery_returns_all_states_and_exact_use_requires_context():
    """发现查询不排名且零 Use；只有完整 pair+context 可采用纯支持前提。"""
    fixture = _fixture()
    try:
        first = fixture.add("similar", 0, 1, stance="support")
        second = fixture.add("similar", 0, 2, stance="support")
        fixture.add(
            "similar", 0, 3,
            stance="refute",
            archive_refuted=True,
        )
        discovery = fixture.runtime.query_similar(SymmetricPairQuery(
            SymmetricPairPattern(endpoint=fixture.objects[0])
        ))
        assert len(discovery.selection.evaluations) == 3
        assert len(discovery.selection.pure_supported()) == 2
        assert discovery.uses == ()
        assert {
            item.evidence[0].proposition
            for item in discovery.selection.pure_supported()
        } == {first, second}

        with pytest.raises(ValueError, match="成对提供"):
            SymmetricPairQuery(
                SymmetricPairPattern(fixture.objects[0], fixture.objects[1]),
                use_key=(10792, 1),
            )
        with pytest.raises(ValueError, match="发现查询不得写 Use"):
            SymmetricPairQuery(
                SymmetricPairPattern(endpoint=fixture.objects[0]),
                use_key=(10792, 2),
                context=_use_context(5020),
            )
        exact = fixture.runtime.query_similar(SymmetricPairQuery(
            SymmetricPairPattern(fixture.objects[0], fixture.objects[1]),
            use_key=(10792, 3),
            context=_use_context(5021),
        ))
        assert len(exact.uses) == 1
        assert exact.uses[0].context is not None
        assert exact.uses[0].context.purpose == concept_identity((10780, 5021, 2))
    finally:
        fixture.close()


class _MappedLegacy:
    """返回固定映射结果的 legacy mapper 测试替身。"""

    def __init__(self, mapped):
        self.mapped = mapped

    def map(self, record):
        """返回预设映射，不自行修正错误。"""
        return self.mapped

    def clone_for_evaluation(self):
        """复制固定映射替身。"""
        return _MappedLegacy(self.mapped)

    def state_key(self):
        """返回固定 mapper 版本键。"""
        return 10800, 1


def test_lifecycle_and_legacy_mapper_keep_channel_schema_source_scope():
    """archived refute 保留、superseded 排除，mapper 不得跨 channel 或改来源。"""
    fixture = _fixture()
    try:
        fixture.add(
            "similar", 0, 1,
            stance="refute",
            archive_refuted=True,
        )
        archived = fixture.runtime.query_similar(SymmetricPairQuery(
            SymmetricPairPattern(fixture.objects[0], fixture.objects[1])
        ))
        assert archived.selection.evaluations[0].state == LogicEvidenceState(
            False, True)

        replacement = fixture.add(
            "similar", 1, 2,
            stance="unknown",
            competition_key=(10801, 1),
        )
        fixture.add(
            "similar", 2, 3,
            stance="refute",
            replacement=replacement,
            competition_key=(10801, 1),
        )
        superseded = fixture.runtime.query_similar(SymmetricPairQuery(
            SymmetricPairPattern(fixture.objects[2], fixture.objects[3])
        ))
        assert superseded.selection.evaluations[0].state == LogicEvidenceState(
            False, False)

        source = _source(6000)
        definition = fixture.definition(
            "similar",
            fixture.objects[4],
            fixture.objects[5],
            source,
            500,
        )
        formation = SymmetricPairFormationRequest(
            RelationClosureCandidateSpec(
                definition,
                fixture.similar_protocol.schema,
                (10802, 1),
                (_source(6001), _source(6002)),
            ),
            document_scope(source),
            SOURCE_BARE_TEXT,
            EPI_STRUCTURED,
        )
        mapped = MappedLegacySymmetricPair(
            formation,
            _recognition_for(definition, source, 501),
        )
        record = LegacySymmetricPairRecord(
            (1,), (2,), (3,), source, document_scope(source), (4,),
        )
        assert fixture.runtime.similar.map_legacy(
            record, _MappedLegacy(mapped)) == mapped
        with pytest.raises(SymmetricRelationRuntimeError, match="typed schema"):
            fixture.runtime.antonym.map_legacy(record, _MappedLegacy(mapped))

        wrong_source = _source(6010)
        wrong_definition = fixture.definition(
            "similar",
            fixture.objects[4],
            fixture.objects[5],
            wrong_source,
            501,
        )
        wrong = MappedLegacySymmetricPair(
            replace(
                formation,
                spec=replace(
                    formation.spec,
                    proposition=wrong_definition,
                ),
                scope=document_scope(wrong_source),
            ),
            _recognition_for(wrong_definition, wrong_source, 502),
        )
        with pytest.raises(SymmetricRelationRuntimeError, match="不得替换"):
            fixture.runtime.similar.map_legacy(record, _MappedLegacy(wrong))
    finally:
        fixture.close()


def test_channel_total_and_option_budgets_fail_closed():
    """单 channel、双 owner 总量和发现结果预算任一耗尽都拒绝部分结果。"""
    direct = _fixture(
        channel_budget=SymmetricRelationBudget(1, 50),
        total_budget=100,
    )
    try:
        direct.add("similar", 0, 1, stance="support")
        direct.add("similar", 0, 2, stance="support")
        with pytest.raises(SymmetricRelationBudgetExceeded, match="直接事实"):
            direct.runtime.query_similar(SymmetricPairQuery(
                SymmetricPairPattern(endpoint=direct.objects[0])
            ))
    finally:
        direct.close()

    total = _fixture(total_budget=1)
    try:
        total.add("similar", 0, 1, stance="support")
        total.add("antonym", 0, 2, stance="support")
        with pytest.raises(SymmetricRelationBudgetExceeded, match="总预算"):
            total.runtime.query_similar(SymmetricPairQuery(
                SymmetricPairPattern(total.objects[0], total.objects[1])
            ))
    finally:
        total.close()

    options = _fixture(
        channel_budget=SymmetricRelationBudget(10, 1),
        total_budget=10,
    )
    try:
        options.add("similar", 0, 1, stance="support")
        options.add("similar", 0, 2, stance="support")
        with pytest.raises(SymmetricRelationBudgetExceeded, match="option"):
            options.runtime.query_similar(SymmetricPairQuery(
                SymmetricPairPattern(endpoint=options.objects[0])
            ))
    finally:
        options.close()


@dataclass(frozen=True)
class _RuntimeBuilder:
    """在任意 TrainContext 图上重建同一 R-05 双 owner 协议。"""

    similar_protocol: SymmetricRelationProtocol
    antonym_protocol: SymmetricRelationProtocol
    channel_budget: SymmetricRelationBudget
    total_budget: SemanticPairBudget
    bound: SemanticPairRuntime | None = None

    def build(self, ctx) -> SemanticPairRuntime:
        """复用同图 bound owner，否则建立共享 SemanticGraph 和两个 H-05 owner。"""
        if (self.bound is not None
                and self.bound.ontology is ctx.graph_ontology):
            return self.bound
        semantic_graph = _semantic_graph(ctx.graph_ontology)
        similar = _channel_runtime(
            ctx,
            semantic_graph,
            self.similar_protocol,
            1,
            self.channel_budget,
        )
        antonym = _channel_runtime(
            ctx,
            semantic_graph,
            self.antonym_protocol,
            2,
            self.channel_budget,
        )
        return SemanticPairRuntime(
            similar,
            antonym,
            self.total_budget,
        )

    def clone_for_evaluation(self):
        """复制不可变协议并清除宿主双 owner 引用。"""
        return _RuntimeBuilder(
            self.similar_protocol,
            self.antonym_protocol,
            self.channel_budget,
            self.total_budget,
        )

    def state_key(self):
        """返回两个协议、kind 版本和预算的完整纯整数键。"""
        return (
            10810,
            *self.similar_protocol.stable_key(),
            *self.antonym_protocol.stable_key(),
            *self.channel_budget.stable_key(),
            *self.total_budget.stable_key(),
        )


@dataclass(frozen=True)
class _EmptyCourse:
    """任何 scope 都返回空双 channel 请求的安装和隔离测试课程。"""

    version: int

    def request(self, scope, *, read_only):
        """返回不学习、不查询的空 R-05 round。"""
        return SemanticPairRoundRequest(scope)

    def similar_legacy_mapper(self):
        """声明当前课程不迁移旧 SIMILAR 输入。"""
        return None

    def antonym_legacy_mapper(self):
        """声明当前课程不迁移旧 ANTONYM 输入。"""
        return None

    def clone_for_evaluation(self):
        """复制不可变空课程。"""
        return _EmptyCourse(self.version)

    def state_key(self):
        """返回课程版本键。"""
        return 10811, self.version


@dataclass(frozen=True)
class _DriftingCourse(_EmptyCourse):
    """故意在评测 clone 中改变版本的隔离负例课程。"""

    def clone_for_evaluation(self):
        """故意改变版本，验证 runtime 拒绝评测状态漂移。"""
        return _DriftingCourse(self.version + 1)


@dataclass(frozen=True)
class _StaticCourse(_EmptyCourse):
    """在目标 scope 返回预设双 channel 请求。"""

    round_request: SemanticPairRoundRequest

    def request(self, scope, *, read_only):
        """仅在目标 scope 和写入轮返回预设请求。"""
        if scope != self.round_request.scope or read_only:
            return SemanticPairRoundRequest(scope)
        return self.round_request

    def clone_for_evaluation(self):
        """复制不可变请求和版本。"""
        return _StaticCourse(self.version, self.round_request)


def _formal_formation(
        protocol: SymmetricRelationProtocol,
        left: ObjectIdentity,
        right: ObjectIdentity,
        source: SourceRef,
        seed: int,
        *,
        timestamp_base: int = 0,
        ) -> tuple[SymmetricPairFormationRequest, RelationClosureRecognitionInput]:
    """构造与一个 formal item scope 绑定的 typed pair forming/recognition。"""
    definition = AtomicPropositionDefinition(
        proposition_identity(source, (10820, seed, 1)),
        protocol.relation,
        occurrence_identity(source, start=0, end=1, ordinal=0),
        context_scope_identity(source, (10820, seed, 2)),
        (
            AtomicRoleBinding(protocol.right_role, right),
            AtomicRoleBinding(protocol.left_role, left),
        ),
    )
    formation = SymmetricPairFormationRequest(
        RelationClosureCandidateSpec(
            definition,
            protocol.schema,
            (10820, seed, 3),
            (_source(7000 + seed * 2), _source(7001 + seed * 2)),
        ),
        document_scope(source),
        SOURCE_BARE_TEXT,
        EPI_STRUCTURED,
        timestamp_base=timestamp_base,
    )
    return formation, _recognition_for(definition, source, 700 + seed)


def test_round_preflight_high_timestamp_and_cross_channel_atomic_failure():
    """双 channel 先全量预检；逻辑序后移，任一 owner 缺 forming 时双方零写。"""
    fixture = _fixture()
    try:
        source = _source(8000)
        similar = _formal_formation(
            fixture.similar_protocol,
            fixture.objects[0],
            fixture.objects[1],
            source,
            1,
            timestamp_base=100_000,
        )
        antonym = _formal_formation(
            fixture.antonym_protocol,
            fixture.objects[2],
            fixture.objects[3],
            source,
            2,
        )
        request = SemanticPairRoundRequest(
            document_scope(source),
            similar=SymmetricChannelBatch(
                formations=(similar[0],), recognitions=(similar[1],)),
            antonym=SymmetricChannelBatch(
                formations=(antonym[0],), recognitions=(antonym[1],)),
        )
        installed = install_semantic_pair_runtime(
            fixture.ctx,
            _RuntimeBuilder(
                fixture.similar_protocol,
                fixture.antonym_protocol,
                fixture.runtime.similar.budget,
                fixture.runtime.budget,
                fixture.runtime,
            ),
            _StaticCourse(1, request),
        )
        report = installed.process(request.scope, read_only=False)
        forming_end = (
            similar[0].timestamp_base
            + len(similar[0].spec.forming_sources) - 1)
        assert report.similar.recognitions[0].outcome.evidence.timestamp_seq > (
            forming_end)
        assert report.antonym.recognitions[0].active_fact is not None

        with pytest.raises(ValueError, match="必须分开"):
            SemanticPairRoundRequest(
                request.scope,
                similar=SymmetricChannelBatch(formations=(similar[0],)),
                antonym=SymmetricChannelBatch(queries=(SymmetricPairQuery(
                    SymmetricPairPattern(
                        fixture.objects[2], fixture.objects[3]),
                ),)),
            )
    finally:
        fixture.close()

    failed = _fixture()
    try:
        source = _source(8010)
        similar = _formal_formation(
            failed.similar_protocol,
            failed.objects[0],
            failed.objects[1],
            source,
            3,
        )
        antonym = _formal_formation(
            failed.antonym_protocol,
            failed.objects[2],
            failed.objects[3],
            source,
            4,
        )
        missing = proposition_identity(source, (10821, 1))
        request = SemanticPairRoundRequest(
            document_scope(source),
            similar=SymmetricChannelBatch(
                formations=(similar[0],), recognitions=(similar[1],)),
            antonym=SymmetricChannelBatch(
                formations=(antonym[0],),
                recognitions=(replace(antonym[1], proposition=missing),),
            ),
        )
        installed = install_semantic_pair_runtime(
            failed.ctx,
            _RuntimeBuilder(
                failed.similar_protocol,
                failed.antonym_protocol,
                failed.runtime.similar.budget,
                failed.runtime.budget,
                failed.runtime,
            ),
            _StaticCourse(1, request),
        )
        baseline = failed.backend.snapshot()
        with pytest.raises(RelationClosureIncompleteError, match="缺少 forming"):
            installed.process(request.scope, read_only=False)
        assert failed.backend.snapshot() == baseline
        assert failed.runtime.similar.relation_runtime.formation_traces() == ()
        assert failed.runtime.antonym.relation_runtime.formation_traces() == ()
        assert failed.runtime.similar.semantic_graph.ontology.resolve(
            similar[0].spec.proposition.proposition
        ) is None
    finally:
        failed.close()


def test_v06_formal_installation_and_inventory_are_honest(tmp_path, monkeypatch):
    """V-06 双 ledger 守恒，formal 成对安装，legacy similar 退出 readiness。"""
    fixture = _fixture()
    try:
        fixture.add("similar", 0, 1, stance="support")
        installed = install_semantic_pair_runtime(
            fixture.ctx,
            _RuntimeBuilder(
                fixture.similar_protocol,
                fixture.antonym_protocol,
                fixture.runtime.similar.budget,
                fixture.runtime.budget,
                fixture.runtime,
            ),
            _EmptyCourse(1),
        )
        baseline = installed.state_key()
        report_count = len(fixture.ctx.semantic_pair_reports)
        with isolated_evaluation(fixture.ctx, label="r05-held-out") as eval_ctx:
            result = eval_ctx.semantic_pair_runtime.owner.query_similar(
                SymmetricPairQuery(
                    SymmetricPairPattern(
                        fixture.objects[1], fixture.objects[0]),
                    use_key=(10830, 1),
                    context=_use_context(8100),
                )
            )
            assert result.selection.pure_supported()
            assert len(result.uses) == 1
        assert installed.state_key() == baseline
        assert len(fixture.ctx.semantic_pair_reports) == report_count
    finally:
        fixture.close()

    drift = _fixture()
    try:
        install_semantic_pair_runtime(
            drift.ctx,
            _RuntimeBuilder(
                drift.similar_protocol,
                drift.antonym_protocol,
                drift.runtime.similar.budget,
                drift.runtime.budget,
                drift.runtime,
            ),
            _DriftingCourse(1),
        )
        with pytest.raises(ValueError, match="改变课程状态"):
            with isolated_evaluation(drift.ctx, label="r05-drift"):
                pass
    finally:
        drift.close()

    partial = FormalTrainConfig(
        run_dir=str(tmp_path),
        run_id="r05-partial",
        active_training_stages=(),
        persist_graph_dump=False,
        language_semantic_pair_builder=object(),
    )
    with pytest.raises(ValueError, match="必须成对配置"):
        formal_train(partial, [], backend=DictBackend())

    from pure_integer_ai.training import stages as training_stages

    monkeypatch.setattr(training_stages, "FLOOR_GRAPH_SIZE_S1", 0)
    similar_protocol = _channel_protocol(1, irreflexive=False)
    antonym_protocol = _channel_protocol(2, irreflexive=True)
    source = _source(8200)
    left = entity_identity(_source(8201), (10840, 1))
    right = entity_identity(_source(8202), (10840, 2))
    similar = _formal_formation(
        similar_protocol, left, right, source, 5)
    antonym = _formal_formation(
        antonym_protocol, left, right, source, 6)
    request = SemanticPairRoundRequest(
        document_scope(source),
        similar=SymmetricChannelBatch(
            formations=(similar[0],), recognitions=(similar[1],)),
        antonym=SymmetricChannelBatch(
            formations=(antonym[0],), recognitions=(antonym[1],)),
    )
    result = formal_train(
        FormalTrainConfig(
            run_dir=str(tmp_path),
            run_id="r05-formal",
            rounds_per_stage=1,
            active_training_stages=(STAGE1_SKELETON,),
            persist_graph_dump=False,
            language_occurrence_protocol=OccurrenceProtocol((10841, 1)),
            language_semantic_pair_builder=_RuntimeBuilder(
                similar_protocol,
                antonym_protocol,
                SymmetricRelationBudget(20, 10),
                SemanticPairBudget(30),
            ),
            language_semantic_pair_course=_StaticCourse(2, request),
        ),
        [CollectedItem(
            tokens=["近义", "反义"],
            raw_text="近义反义",
            role_seq=[1, 1],
            source=source.source_kind,
            source_ref=source,
        )],
        backend=DictBackend(),
    )
    assert result.semantic_pair_reports
    report = result.semantic_pair_reports[-1]
    assert report.similar.recognitions[0].active_fact is not None
    assert report.antonym.recognitions[0].active_fact is not None

    records = inventory_by_id()
    candidates = {item.mechanism_id for item in readiness_candidates()}
    typed = records["relation.semantic_pair_typed"]
    legacy = records["relation.similar"]
    assert typed.status == STATUS_OPT_IN
    assert typed.readiness_eligible is False
    assert "FormalTrainConfig.language_semantic_pair_builder" in typed.gates
    assert "V06" in typed.recovery[-1]
    assert "K 盘" in typed.limitation
    assert typed.mechanism_id not in candidates
    assert legacy.readiness_eligible is False
    assert legacy.mechanism_id not in candidates
    assert validate_inventory() == ()
