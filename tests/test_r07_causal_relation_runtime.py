"""R-07 typed CAUSES 独立证据、时间约束、执行和隔离验收。"""
from dataclasses import dataclass, replace

import pytest

from pure_integer_ai.cognition.shared.candidate_projection import (
    CandidateProjectionGraph,
    CandidateProjectionProtocol,
)
from pure_integer_ai.cognition.shared.candidate_runtime import (
    CandidateLearningRuntime,
    CandidateProjectionMetadata,
)
from pure_integer_ai.cognition.shared.candidate_verifier import (
    IndependentObjectVerifier,
    IndependentVerifierProtocol,
)
from pure_integer_ai.cognition.shared.causal_execution import (
    CAUSAL_EXECUTION_CONFLICTED,
    CAUSAL_EXECUTION_PREDICTED,
    CAUSAL_EXECUTION_UNKNOWN,
    CAUSAL_TEMPORAL_ACCEPTED,
    CAUSAL_TEMPORAL_CONFLICTED,
    CAUSAL_TEMPORAL_REJECTED,
    CAUSAL_TEMPORAL_UNKNOWN,
    CausalEndpointEvaluation,
    CausalEndpointProtocol,
    CausalExecutor,
    CausalTemporalAssessment,
)
from pure_integer_ai.cognition.shared.event_time import (
    EVENT_TIME_AFTER,
    EVENT_TIME_BEFORE,
    EVENT_TIME_CONFLICTED,
    EVENT_TIME_CONSISTENT,
    EVENT_TIME_EMPTY,
    EVENT_TIME_UNKNOWN,
    EventTimeFactIndex,
    EventTimeVerifier,
    ResolvedEventTimeRelation,
)
from pure_integer_ai.cognition.shared.evidence_candidate import (
    EvidenceCandidateEngine,
    EvidenceCandidateProtocol,
)
from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_SUPPORT,
    EVIDENCE_UNKNOWN,
)
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    OBJECT_PROPOSITION,
    SourceRef,
    VersionBundle,
    concept_identity,
    minimal_instruction_identity,
    occurrence_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.graph_ontology import (
    relation_concept_identity,
)
from pure_integer_ai.cognition.shared.logic_executor import (
    LogicEvaluation,
    LogicEvidenceState,
)
from pure_integer_ai.cognition.shared.order_facts import OrderFactIndex
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
    event_identity,
    proposition_identity,
    role_identity,
)
from pure_integer_ai.cognition.shared.typed_binding import BoundProposition
from pure_integer_ai.cognition.shared.typed_relation import (
    RelationSchema,
    RelationSlotSchema,
)
from pure_integer_ai.cognition.understanding.occurrence_index import (
    OccurrenceProtocol,
)
from pure_integer_ai.experiments.causal_relation_course import (
    CausalEventTimeFactRequest,
    CausalExecutionRequest,
    CausalFormationRequest,
    CausalRelationCourseRuntime,
    CausalRoundRequest,
    install_causal_relation_runtime,
)
from pure_integer_ai.experiments.causal_relation_runtime import (
    CausalIndependentWitness,
    CausalRelationRuntime,
    CausalVerificationAdapter,
    CausalVerificationArtifact,
    CausalVerificationProtocol,
    CausalVerificationRequest,
)
from pure_integer_ai.experiments.collection import (
    COLLECT_PRECEDES,
    CollectedItem,
)
from pure_integer_ai.experiments.evaluation_isolation import (
    clone_backend,
    isolated_evaluation,
)
from pure_integer_ai.experiments.evaluation_protocol import ProtocolKey
from pure_integer_ai.experiments.event_time_verification import (
    EventTimeVerificationRequest,
)
from pure_integer_ai.experiments.formal_train import (
    FormalTrainConfig,
    formal_train,
)
from pure_integer_ai.experiments.language_protocol_runtime import (
    install_language_graph_protocols,
)
from pure_integer_ai.experiments.relation_closure_runtime import (
    RelationClosureRuntime,
)
from pure_integer_ai.experiments.round_runtime import DefaultRoundRunner
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.experiments.verification_orchestration import (
    VERDICT_CONFLICTED,
    VERDICT_REFUTE,
    VERDICT_SUPPORT,
    VERDICT_UNKNOWN,
    MultiVerifierOrchestrator,
    VerificationEffect,
    VerificationEvaluation,
)
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import EPI_STRUCTURED, SOURCE_BARE_TEXT
from pure_integer_ai.training.stages import STAGE1_SKELETON


_BASE = 92000


def _source(source_id: int) -> SourceRef:
    """构造 owner/version 相同而来源身份互异的测试 SourceRef。"""
    return SourceRef(
        151,
        source_id,
        0,
        GLOBAL_OWNER_SCOPE,
        VersionBundle(),
    )


def _semantic_identities():
    """返回 S-00 六个开放图 predicate 身份。"""
    return tuple(
        relation_concept_identity((_BASE + 10, ordinal))
        for ordinal in range(1, 7)
    )


def _semantic_graph(ontology) -> SemanticGraph:
    """在指定 ontology 上重建 S-00 原子命题 facade。"""
    refs = tuple(
        ontology.materialize(identity)
        for identity in _semantic_identities()
    )
    return SemanticGraph(ontology, AtomicPropositionPredicates(*refs))


def _projection_protocol() -> CandidateProjectionProtocol:
    """构造 causal H-05 lifecycle 图协议。"""
    values = tuple(
        concept_identity((_BASE + 20, ordinal))
        for ordinal in range(13)
    )
    return CandidateProjectionProtocol(
        *values,
        (_BASE + 21, 1),
    )


def _evidence_protocol() -> EvidenceCandidateProtocol:
    """构造要求两个 forming source 的 causal H-05 owner 协议。"""
    aggregate = _source(900)
    return EvidenceCandidateProtocol(
        (_BASE + 30, 1),
        (_BASE + 30, 2),
        aggregate,
        document_scope(aggregate),
        2,
    )


def _independent_verifier() -> IndependentObjectVerifier:
    """构造只消费 causal adapter 生成 reveal 的独立三态 verifier。"""
    return IndependentObjectVerifier(IndependentVerifierProtocol(
        concept_identity((_BASE + 40, 1)),
        (_BASE + 40, 2),
        (_BASE + 40, 3),
        (_BASE + 40, 4),
        (_BASE + 40, 5),
    ))


class _EventRelationResolver:
    """按测试注入表解释 before/after relation。"""

    def __init__(self, directions) -> None:
        self.directions = dict(directions)

    def resolve(self, relation):
        """返回 relation 对应的最小方向协议。"""
        return ResolvedEventTimeRelation(
            relation,
            self.directions[relation],
            (_BASE + 50, self.directions[relation]),
        )


class _TemporalResolver:
    """只按规范化 cause/effect 方向裁决 causal 时间约束。"""

    accepted_reason = minimal_instruction_identity((_BASE + 60, 1))
    rejected_reason = minimal_instruction_identity((_BASE + 60, 2))
    unknown_reason = minimal_instruction_identity((_BASE + 60, 3))
    conflict_reason = minimal_instruction_identity((_BASE + 60, 4))

    def resolve(self, cause, effect, result):
        """一致正向 accepted，反向 rejected，缺失和冲突保持分型。"""
        hashes = tuple(
            fact.assertion_hash for fact in result.fact_set.facts)
        if result.status == EVENT_TIME_CONFLICTED:
            return CausalTemporalAssessment(
                CAUSAL_TEMPORAL_CONFLICTED,
                self.conflict_reason,
                (_BASE + 61, 4),
                hashes,
            )
        if result.status in {EVENT_TIME_EMPTY, EVENT_TIME_UNKNOWN}:
            return CausalTemporalAssessment(
                CAUSAL_TEMPORAL_UNKNOWN,
                self.unknown_reason,
                (_BASE + 61, 3),
            )
        if result.status != EVENT_TIME_CONSISTENT:
            raise ValueError("测试 temporal resolver 收到未知状态")
        if (cause, effect) in result.before_edges:
            return CausalTemporalAssessment(
                CAUSAL_TEMPORAL_ACCEPTED,
                self.accepted_reason,
                (_BASE + 61, 1),
                hashes,
            )
        if (effect, cause) in result.before_edges:
            return CausalTemporalAssessment(
                CAUSAL_TEMPORAL_REJECTED,
                self.rejected_reason,
                (_BASE + 61, 2),
                hashes,
            )
        return CausalTemporalAssessment(
            CAUSAL_TEMPORAL_UNKNOWN,
            self.unknown_reason,
            (_BASE + 61, 3),
        )


class _DisconnectedAcceptedResolver:
    """故意把互不连通但分别触及端点的时间事实误报为 accepted。"""

    def resolve(self, cause, effect, result):
        """返回引用全部事实的伪 accepted 裁决，供路径防线验收。"""
        del cause, effect
        return CausalTemporalAssessment(
            CAUSAL_TEMPORAL_ACCEPTED,
            minimal_instruction_identity((_BASE + 62, 1)),
            (_BASE + 62, 2),
            tuple(fact.assertion_hash for fact in result.fact_set.facts),
        )


@dataclass
class _Fixture:
    """集中保存一套可关闭、可 clone 的 R-07 测试设施。"""

    backend: DictBackend
    ctx: object
    semantic_graph: SemanticGraph
    candidate_graph: CandidateProjectionGraph
    relation_runtime: RelationClosureRuntime
    runtime: CausalRelationRuntime
    spec: RelationClosureCandidateSpec
    cause: object
    effect: object
    before: object
    after: object

    def close(self) -> None:
        """关闭测试后端。"""
        self.backend.close()


@dataclass(frozen=True)
class _Domain:
    """保存可在正式图和评测图上重建的 causal 测试协议。"""

    closure_protocol: RelationClosureProtocol
    schema: RelationSchema
    spec: RelationClosureCandidateSpec
    cause: object
    effect: object
    before: object
    after: object
    endpoints: CausalEndpointProtocol
    verification: CausalVerificationProtocol


@dataclass(frozen=True)
class _DomainView:
    """向 request helper 暴露课程构造所需的最小领域视图。"""

    spec: RelationClosureCandidateSpec
    cause: object
    effect: object
    before: object


def _domain() -> _Domain:
    """构造不依赖运行时图引用的完整 causal 测试领域。"""
    closure_protocol = RelationClosureProtocol(
        RelationClosureField(concept_identity((_BASE + 70, 1))),
        RelationClosureField(concept_identity((_BASE + 70, 2))),
    )
    source = _source(1)
    relation = relation_concept_identity((_BASE + 80, 1))
    cause_role = role_identity((_BASE + 80, 2))
    effect_role = role_identity((_BASE + 80, 3))
    schema = RelationSchema(
        structure_concept_identity((_BASE + 80, 4)),
        relation,
        (
            RelationSlotSchema(
                cause_role, frozenset({OBJECT_PROPOSITION}), 1, 1),
            RelationSlotSchema(
                effect_role, frozenset({OBJECT_PROPOSITION}), 1, 1),
        ),
    )
    cause = proposition_identity(source, (_BASE + 81, 1))
    effect = proposition_identity(source, (_BASE + 81, 2))
    definition = AtomicPropositionDefinition(
        proposition_identity(source, (_BASE + 81, 3)),
        relation,
        occurrence_identity(source, start=0, end=2, ordinal=0),
        context_scope_identity(source, (_BASE + 81, 4)),
        (
            AtomicRoleBinding(cause_role, cause),
            AtomicRoleBinding(effect_role, effect),
        ),
    )
    before = concept_identity((_BASE + 90, 1))
    after = concept_identity((_BASE + 90, 2))
    return _Domain(
        closure_protocol,
        schema,
        RelationClosureCandidateSpec(
            definition,
            schema,
            (_BASE + 82, 1),
            (_source(2), _source(3)),
        ),
        cause,
        effect,
        before,
        after,
        CausalEndpointProtocol(
            relation,
            cause_role,
            effect_role,
            minimal_instruction_identity((_BASE + 100, 1)),
        ),
        CausalVerificationProtocol(
            ProtocolKey((_BASE + 110, 1)),
            ProtocolKey((_BASE + 110, 2)),
            ProtocolKey((_BASE + 110, 3)),
        ),
    )


def _build_runtime(ctx, domain: _Domain):
    """在指定 TrainContext 图上装配低层 R-00/R-06/R-07 owner。"""
    semantic_graph = _semantic_graph(ctx.graph_ontology)
    candidate_graph = CandidateProjectionGraph(
        ctx.graph_ontology,
        _projection_protocol(),
    )
    candidate_runtime = CandidateLearningRuntime(
        EvidenceCandidateEngine(_evidence_protocol()),
        candidate_graph,
        _independent_verifier(),
        CandidateProjectionMetadata(SOURCE_BARE_TEXT, EPI_STRUCTURED),
    )
    consumer = ActiveRelationClosureConsumer(
        semantic_graph,
        candidate_graph,
        domain.closure_protocol,
        (domain.schema,),
        engine=candidate_runtime.engine,
    )
    relation_runtime = RelationClosureRuntime(
        candidate_runtime,
        semantic_graph,
        consumer,
        domain.closure_protocol,
    )
    event_time_facts = EventTimeFactIndex(OrderFactIndex(
        ctx.graph_ontology,
        ctx.scoped_identity_store,
    ))
    event_time_verifier = EventTimeVerifier(
        event_time_facts,
        _EventRelationResolver({
            domain.before: EVENT_TIME_BEFORE,
            domain.after: EVENT_TIME_AFTER,
        }),
    )
    runtime = CausalRelationRuntime(
        semantic_graph,
        relation_runtime,
        event_time_facts,
        event_time_verifier,
        CausalExecutor(domain.endpoints, _TemporalResolver()),
        domain.verification,
    )
    return semantic_graph, candidate_graph, relation_runtime, runtime


@dataclass(frozen=True)
class _FormalProtocol:
    """在任意 TrainContext 图上重建同一 causal schema 和 owner。"""

    domain: _Domain

    def build(self, ctx) -> CausalRelationRuntime:
        """返回绑定当前 TrainContext 图的低层 typed causal runtime。"""
        return _build_runtime(ctx, self.domain)[-1]

    def stable_key(self) -> tuple[int, ...]:
        """返回 relation、Role、schema 和 R-09 身份的完整协议键。"""
        values = [_BASE + 180]
        for identity in (
                self.domain.schema.schema,
                self.domain.endpoints.relation,
                self.domain.endpoints.cause_role,
                self.domain.endpoints.effect_role,
                self.domain.endpoints.execution_instruction):
            key = identity.stable_key()
            values.extend((len(key), *key))
        for key in (
                self.domain.verification.dimension.stable_key(),
                self.domain.verification.verifier.stable_key(),
                self.domain.verification.evidence_target_kind.stable_key()):
            values.extend((len(key), *key))
        return tuple(values)


@dataclass(frozen=True)
class _FormalCourse:
    """只按完整领域对象和来源 scope 生成 typed causal 课程请求。"""

    domain: _Domain

    def request(self, scope, *, read_only):
        """训练时形成候选，held-out 时只核验并消费已学 active 事实。"""
        if scope != document_scope(self.domain.spec.proposition.source):
            return CausalRoundRequest(scope)
        view = _DomainView(
            self.domain.spec,
            self.domain.cause,
            self.domain.effect,
            self.domain.before,
        )
        verification = _request(
            view,
            80,
            stance=EVIDENCE_SUPPORT,
            temporal_scope=scope,
        )
        cause, effect = _endpoint_evaluations(view)
        suffix = 2 if read_only else 1
        return CausalRoundRequest(
            scope,
            temporal_facts=(CausalEventTimeFactRequest(
                self.domain.before,
                self.domain.cause,
                self.domain.effect,
                scope,
                SOURCE_BARE_TEXT,
                EPI_STRUCTURED,
            ),),
            formations=() if read_only else (CausalFormationRequest(
                self.domain.spec,
                scope,
                SOURCE_BARE_TEXT,
                EPI_STRUCTURED,
            ),),
            verifications=(verification,),
            executions=(CausalExecutionRequest(
                self.domain.spec.proposition.proposition,
                verification.temporal,
                cause,
                effect,
                (_BASE + 181, suffix),
                (_BASE + 182, suffix),
            ),),
        )

    def clone_for_evaluation(self):
        """返回领域身份相同且不共享可变状态的课程副本。"""
        return _FormalCourse(self.domain)

    def state_key(self) -> tuple[int, ...]:
        """返回课程版本和候选 Proposition 的完整纯整数键。"""
        proposition = self.domain.spec.proposition.proposition.stable_key()
        return _BASE + 183, len(proposition), *proposition


class _RejectAllEventTimeFacts:
    """模拟 H-00/H-04 已撤销全部 raw 时间投影的 active 过滤器。"""

    def accepts(self, fact) -> bool:
        """拒绝所有历史 statement，证明 causal 不会绕过 active 状态。"""
        return False


@dataclass(frozen=True)
class _SharedEventTimeRuntime:
    """为联合 R-06B/R-07 安装测试提供最小共享 runtime 边界。"""

    facts: EventTimeFactIndex
    verifier: EventTimeVerifier


@dataclass(frozen=True)
class _FormalCourseWithoutTemporalWrites:
    """保留 causal forming/verification，但不再提交兼容时间写入。"""

    delegate: _FormalCourse

    def request(self, scope, *, read_only):
        """删除平行时间写入和依赖 active CAUSES 的执行请求。"""
        return replace(
            self.delegate.request(scope, read_only=read_only),
            temporal_facts=(),
            executions=(),
        )

    def clone_for_evaluation(self):
        """返回不共享课程对象的联合链评测副本。"""
        return _FormalCourseWithoutTemporalWrites(
            self.delegate.clone_for_evaluation())

    def state_key(self) -> tuple[int, ...]:
        """在原课程身份上追加联合链无写入版本标记。"""
        return *self.delegate.state_key(), _BASE + 185


def _fixture() -> _Fixture:
    """创建尚未 forming、但协议和 typed facade 完整的 causal 环境。"""
    backend = DictBackend()
    ctx = make_train_context(backend)
    domain = _domain()
    semantic_graph, candidate_graph, relation_runtime, runtime = (
        _build_runtime(ctx, domain)
    )
    return _Fixture(
        backend,
        ctx,
        semantic_graph,
        candidate_graph,
        relation_runtime,
        runtime,
        domain.spec,
        domain.cause,
        domain.effect,
        domain.before,
        domain.after,
    )


def _form(fixture: _Fixture) -> None:
    """物化 typed CAUSES 并登记 forming unknown。"""
    fixture.runtime.form(
        fixture.spec,
        scope=document_scope(fixture.spec.proposition.source),
        provenance_kind=SOURCE_BARE_TEXT,
        epistemic_origin=EPI_STRUCTURED,
    )


def _record_time(
        fixture: _Fixture, relation, *, scope=None, reverse=False,
        ):
    """在指定 scope 写入一条 cause/effect typed 时间事实。"""
    target_scope = (
        document_scope(fixture.spec.proposition.source)
        if scope is None else scope
    )
    first, second = (
        (fixture.effect, fixture.cause)
        if reverse else (fixture.cause, fixture.effect)
    )
    return fixture.runtime.event_time_facts.record(
        relation,
        first,
        second,
        scope=target_scope,
        provenance_kind=SOURCE_BARE_TEXT,
        epistemic_origin=EPI_STRUCTURED,
    )


def _request(
        fixture: _Fixture,
        observation_id: int,
        *,
        stance: int,
        temporal_scope=None,
        relations=None,
        witness_source=None,
        witness_inputs=None,
        ) -> CausalVerificationRequest:
    """构造带独立 witness、Occurrence anchor 和时间请求的 causal 输入。"""
    observation = _source(observation_id)
    scope = document_scope(observation)
    anchor = occurrence_identity(
        observation,
        start=0,
        end=2,
        ordinal=0,
    )
    return CausalVerificationRequest(
        fixture.spec,
        observation,
        scope,
        ProtocolKey((_BASE + 120, observation_id)),
        (_BASE + 121, observation_id),
        anchor,
        (anchor, fixture.cause, fixture.effect),
        CausalIndependentWitness(
            stance,
            _source(700 + observation_id)
            if witness_source is None else witness_source,
            (fixture.cause, fixture.effect)
            if witness_inputs is None else witness_inputs,
            (_BASE + 122, observation_id),
        ),
        EventTimeVerificationRequest(
            document_scope(fixture.spec.proposition.source)
            if temporal_scope is None else temporal_scope,
            (fixture.before,) if relations is None else relations,
        ),
    )


def _logic_evaluation(
        fixture: _Fixture,
        endpoint,
        *,
        support: bool,
        refute: bool,
        ordinal: int,
        ) -> LogicEvaluation:
    """构造当前 causal query 使用的 S-04 四态原子求值。"""
    source = fixture.spec.proposition.source
    bound = BoundProposition(
        endpoint,
        minimal_instruction_identity((_BASE + 130, 1)),
        relation_concept_identity((_BASE + 130, 2)),
        structure_concept_identity((_BASE + 130, 3)),
        occurrence_identity(source, start=ordinal, end=ordinal + 1, ordinal=0),
        context_scope_identity(source, (_BASE + 130, 4)),
        (),
        (),
        (),
    )
    evidence_ids = (ordinal + 1,) if support or refute else ()
    return LogicEvaluation(
        bound,
        LogicEvidenceState(support, refute),
        source,
        document_scope(source),
        evidence_ids=evidence_ids,
    )


def _endpoint_evaluations(
        fixture: _Fixture, *, cause=(True, False), effect=(False, False),
        ):
    """构造显式 endpoint-to-Proposition mapping 和 S-04 trace。"""
    instruction = minimal_instruction_identity((_BASE + 140, 1))
    return (
        CausalEndpointEvaluation(
            fixture.cause,
            _logic_evaluation(
                fixture,
                fixture.cause,
                support=cause[0],
                refute=cause[1],
                ordinal=10,
            ),
            instruction,
            (_BASE + 141, 1),
        ),
        CausalEndpointEvaluation(
            fixture.effect,
            _logic_evaluation(
                fixture,
                fixture.effect,
                support=effect[0],
                refute=effect[1],
                ordinal=20,
            ),
            instruction,
            (_BASE + 141, 2),
        ),
    )


def _formal_item() -> CollectedItem:
    """构造只提供来源和普通 token、没有 causal cue 特权的训练项。"""
    source_ref = _source(1)
    return CollectedItem(
        tokens=["甲", "乙"],
        raw_text="甲乙",
        role_seq=[1, 1],
        collect_type=COLLECT_PRECEDES,
        source=source_ref.source_kind,
        source_ref=source_ref,
    )


def _install_formal_causal(ctx, domain: _Domain) -> CausalRelationCourseRuntime:
    """安装来源 occurrence 协议和成对注入的 R-07 protocol/course。"""
    install_language_graph_protocols(
        ctx,
        occurrence_protocol=OccurrenceProtocol((_BASE + 184, 1)),
    )
    return install_causal_relation_runtime(
        ctx,
        _FormalProtocol(domain),
        _FormalCourse(domain),
    )


def test_causal_support_requires_independent_witness_and_temporal_acceptance():
    """只读先得同 verdict 零提交，训练提交后才能执行并记录生成采用。"""
    fixture = _fixture()
    try:
        _form(fixture)
        _record_time(fixture, fixture.before)
        request = _request(
            fixture,
            10,
            stance=EVIDENCE_SUPPORT,
        )
        orchestrator = MultiVerifierOrchestrator()

        held_out = orchestrator.run(
            request,
            (fixture.runtime.registration(),),
            read_only=True,
        )
        assert held_out.results[0].verdict == VERDICT_SUPPORT
        assert held_out.results[0].committed_effects == ()
        assert fixture.relation_runtime.consumer.lookup_proposition(
            fixture.spec.proposition.proposition) == ()

        trained = orchestrator.run(
            request,
            (fixture.runtime.registration(),),
            read_only=False,
        )
        assert trained.results[0].verdict == VERDICT_SUPPORT
        assert trained.results[0].committed_effects
        cause, effect = _endpoint_evaluations(fixture)
        execution = fixture.runtime.execute(
            fixture.spec.proposition.proposition,
            request.temporal,
            cause,
            effect,
            use_key=(_BASE + 150, 1),
            generation_use_key=(_BASE + 150, 2),
        )
        assert execution.execution.status == CAUSAL_EXECUTION_PREDICTED
        assert execution.execution.predicted_effect is True
        assert execution.generation_use is not None
        assert fixture.relation_runtime.require_complete(fixture.spec).complete
    finally:
        fixture.close()


def test_time_or_witness_alone_never_supports_causal_candidate():
    """有 witness 无时间、或有时间但 witness unknown，都只能提交 unknown。"""
    first = _fixture()
    second = _fixture()
    try:
        _form(first)
        no_time = MultiVerifierOrchestrator().run(
            _request(first, 20, stance=EVIDENCE_SUPPORT),
            (first.runtime.registration(),),
            read_only=False,
        )
        assert no_time.results[0].verdict == VERDICT_UNKNOWN
        assert first.relation_runtime.consumer.lookup_proposition(
            first.spec.proposition.proposition) == ()

        _form(second)
        _record_time(second, second.before)
        no_witness = MultiVerifierOrchestrator().run(
            _request(second, 21, stance=EVIDENCE_UNKNOWN),
            (second.runtime.registration(),),
            read_only=False,
        )
        assert no_witness.results[0].verdict == VERDICT_UNKNOWN
        assert second.relation_runtime.consumer.lookup_proposition(
            second.spec.proposition.proposition) == ()
    finally:
        first.close()
        second.close()


def test_causal_witness_cannot_reuse_forming_source_or_candidate_itself():
    """forming source 和待核验 Proposition 都不能充当独立 causal witness。"""
    fixture = _fixture()
    try:
        _form(fixture)
        _record_time(fixture, fixture.before)
        forming_source = fixture.spec.forming_sources[0]
        report = MultiVerifierOrchestrator().run(
            _request(
                fixture,
                30,
                stance=EVIDENCE_SUPPORT,
                witness_source=forming_source,
            ),
            (fixture.runtime.registration(),),
            read_only=False,
        )
        assert report.results[0].verdict == VERDICT_UNKNOWN
        assert report.results[0].operational_failure == "ValueError"
        assert report.results[0].committed_effects == ()

        registration = fixture.runtime.registration()
        with pytest.raises(ValueError, match="Proposition 自身"):
            registration.evaluate(_request(
                fixture,
                31,
                stance=EVIDENCE_SUPPORT,
                witness_inputs=(fixture.spec.proposition.proposition,),
            ))
        with pytest.raises(ValueError, match="不得冒充 event-time"):
            registration.evaluate(_request(
                fixture,
                32,
                stance=EVIDENCE_SUPPORT,
                relations=(fixture.spec.proposition.predicate,),
            ))
    finally:
        fixture.close()


def test_temporal_support_requires_a_connected_assertion_path():
    """分别触及 cause/effect 但互不连通的事实不能通过 causal 时间约束。"""
    fixture = _fixture()
    try:
        _form(fixture)
        scope = document_scope(fixture.spec.proposition.source)
        left = event_identity(_source(3301), (_BASE + 123, 1))
        right = event_identity(_source(3302), (_BASE + 123, 2))
        fixture.runtime.event_time_facts.record(
            fixture.before,
            fixture.cause,
            left,
            scope=scope,
            provenance_kind=SOURCE_BARE_TEXT,
            epistemic_origin=EPI_STRUCTURED,
        )
        fixture.runtime.event_time_facts.record(
            fixture.before,
            right,
            fixture.effect,
            scope=scope,
            provenance_kind=SOURCE_BARE_TEXT,
            epistemic_origin=EPI_STRUCTURED,
        )
        adapter = CausalVerificationAdapter(
            fixture.relation_runtime,
            fixture.runtime.event_time_verifier,
            _DisconnectedAcceptedResolver(),
            fixture.runtime.executor.protocol,
            fixture.runtime.verification_protocol,
        )

        with pytest.raises(ValueError, match="完整方向路径"):
            adapter.registration().evaluate(_request(
                fixture,
                33,
                stance=EVIDENCE_SUPPORT,
            ))
        assert fixture.relation_runtime.consumer.lookup_proposition(
            fixture.spec.proposition.proposition) == ()
    finally:
        fixture.close()


def test_causal_artifact_and_commit_reject_cross_stage_drift_before_write():
    """artifact 或 proposed effect 被替换时，commit 必须在 recognition 写入前失败。"""
    fixture = _fixture()
    try:
        _form(fixture)
        _record_time(fixture, fixture.before)
        request = _request(fixture, 34, stance=EVIDENCE_SUPPORT)
        registration = fixture.runtime.registration()
        evaluation = registration.evaluate(request)
        artifact = evaluation.artifact
        assert isinstance(artifact, CausalVerificationArtifact)

        with pytest.raises(ValueError, match="stance/verdict"):
            replace(artifact, verdict=VERDICT_UNKNOWN)

        forged = VerificationEvaluation(
            evaluation.verdict,
            evaluation.claim_keys,
            (),
            evaluation.detail,
            evaluation.source,
            evaluation.scope,
            artifact,
        )
        baseline = fixture.relation_runtime.state_key()
        assert registration.commit is not None
        with pytest.raises(ValueError, match="evaluation 与 artifact"):
            registration.commit(forged)
        assert fixture.relation_runtime.state_key() == baseline

        wrong_effect = VerificationEffect(
            ProtocolKey((_BASE + 124, 1)),
            artifact.effect.target_kind,
            artifact.effect.target_key,
        )
        wrong_artifact = replace(artifact, effect=wrong_effect)
        wrong_evaluation = VerificationEvaluation(
            evaluation.verdict,
            evaluation.claim_keys,
            (wrong_effect,),
            evaluation.detail,
            evaluation.source,
            evaluation.scope,
            wrong_artifact,
        )
        with pytest.raises(ValueError, match="维度或目标类型"):
            registration.commit(wrong_evaluation)
        assert fixture.relation_runtime.state_key() == baseline
    finally:
        fixture.close()


def test_reverse_temporal_constraint_refutes_and_demotes_active_causes():
    """独立 support 建活后，另一 scope 的反向时间约束形成 refute 并退出消费者。"""
    fixture = _fixture()
    try:
        _form(fixture)
        forward_scope = document_scope(fixture.spec.proposition.source)
        _record_time(fixture, fixture.before, scope=forward_scope)
        orchestrator = MultiVerifierOrchestrator()
        orchestrator.run(
            _request(
                fixture,
                40,
                stance=EVIDENCE_SUPPORT,
                temporal_scope=forward_scope,
            ),
            (fixture.runtime.registration(),),
            read_only=False,
        )
        assert fixture.relation_runtime.consumer.lookup_proposition(
            fixture.spec.proposition.proposition)

        reverse_source = _source(4040)
        reverse_scope = document_scope(reverse_source)
        _record_time(
            fixture,
            fixture.after,
            scope=reverse_scope,
        )
        refuted = orchestrator.run(
            _request(
                fixture,
                41,
                stance=EVIDENCE_SUPPORT,
                temporal_scope=reverse_scope,
                relations=(fixture.after,),
            ),
            (fixture.runtime.registration(),),
            read_only=False,
        )
        assert refuted.results[0].verdict == VERDICT_REFUTE
        assert fixture.relation_runtime.consumer.lookup_proposition(
            fixture.spec.proposition.proposition) == ()
    finally:
        fixture.close()


def test_temporal_conflict_is_reported_without_support_commit():
    """冲突时间图得到 causal conflicted verdict，但只提交 unknown Evidence。"""
    fixture = _fixture()
    try:
        _form(fixture)
        scope = document_scope(fixture.spec.proposition.source)
        _record_time(fixture, fixture.before, scope=scope)
        _record_time(fixture, fixture.before, scope=scope, reverse=True)
        report = MultiVerifierOrchestrator().run(
            _request(
                fixture,
                50,
                stance=EVIDENCE_SUPPORT,
                temporal_scope=scope,
            ),
            (fixture.runtime.registration(),),
            read_only=False,
        )
        assert report.results[0].verdict == VERDICT_CONFLICTED
        artifact = report.results[0].artifact
        assert artifact.evidence_stance == EVIDENCE_UNKNOWN
        assert fixture.relation_runtime.consumer.lookup_proposition(
            fixture.spec.proposition.proposition) == ()
    finally:
        fixture.close()


def test_causal_execution_preserves_unknown_and_effect_conflict():
    """cause 未支持不预测；effect 被反驳时保留 provisional/refute 冲突。"""
    fixture = _fixture()
    try:
        _form(fixture)
        _record_time(fixture, fixture.before)
        request = _request(fixture, 60, stance=EVIDENCE_SUPPORT)
        MultiVerifierOrchestrator().run(
            request,
            (fixture.runtime.registration(),),
            read_only=False,
        )

        unknown_cause, unknown_effect = _endpoint_evaluations(
            fixture,
            cause=(False, False),
        )
        with pytest.raises(ValueError, match="不得进入 causal 生成采用"):
            fixture.runtime.execute(
                fixture.spec.proposition.proposition,
                request.temporal,
                unknown_cause,
                unknown_effect,
                use_key=(_BASE + 160, 0),
                generation_use_key=(_BASE + 160, 9),
            )
        assert fixture.relation_runtime.audit(
            fixture.spec).consumer_used is False
        unknown = fixture.runtime.execute(
            fixture.spec.proposition.proposition,
            request.temporal,
            unknown_cause,
            unknown_effect,
            use_key=(_BASE + 160, 1),
        )
        assert unknown.execution.status == CAUSAL_EXECUTION_UNKNOWN
        assert unknown.execution.predicted_effect is False

        cause, refuted_effect = _endpoint_evaluations(
            fixture,
            effect=(False, True),
        )
        conflicted = fixture.runtime.execute(
            fixture.spec.proposition.proposition,
            request.temporal,
            cause,
            refuted_effect,
            use_key=(_BASE + 160, 2),
            generation_use_key=(_BASE + 160, 3),
        )
        assert conflicted.execution.status == CAUSAL_EXECUTION_CONFLICTED
        assert conflicted.execution.effect_state.support is True
        assert conflicted.execution.effect_state.refute is True
    finally:
        fixture.close()


def test_causal_execution_key_covers_binding_trace_and_collision_is_atomic():
    """endpoint 映射变化必须改变归因键，同 use key 冲突不得留下半写账。"""
    fixture = _fixture()
    try:
        _form(fixture)
        _record_time(fixture, fixture.before)
        request = _request(fixture, 61, stance=EVIDENCE_SUPPORT)
        MultiVerifierOrchestrator().run(
            request,
            (fixture.runtime.registration(),),
            read_only=False,
        )
        cause, effect = _endpoint_evaluations(fixture)
        first = fixture.runtime.execute(
            fixture.spec.proposition.proposition,
            request.temporal,
            cause,
            effect,
            use_key=(_BASE + 161, 1),
            generation_use_key=(_BASE + 161, 2),
        )
        changed_cause = replace(cause, trace=(_BASE + 161, 3))
        fact = fixture.relation_runtime.consumer.require_proposition(
            fixture.spec.proposition.proposition)
        temporal_result = fixture.runtime.event_time_verifier.verify(
            request.temporal.relations,
            scope=request.temporal.scope,
        )
        changed_result = fixture.runtime.executor.execute(
            fact,
            temporal_result,
            changed_cause,
            effect,
        )
        fixture.runtime.adapter.validate_temporal_support(
            changed_cause.endpoint,
            effect.endpoint,
            temporal_result,
            changed_result.temporal_assessment,
        )
        assert changed_result.stable_key() != first.execution.stable_key()

        baseline = fixture.runtime.state_key()
        with pytest.raises(RuntimeError, match="同一 causal execution use_key"):
            fixture.runtime.execute(
                fixture.spec.proposition.proposition,
                request.temporal,
                changed_cause,
                effect,
                use_key=(_BASE + 161, 1),
                generation_use_key=(_BASE + 161, 4),
            )
        assert fixture.runtime.state_key() == baseline
    finally:
        fixture.close()


def test_causal_clone_read_only_verdict_and_execution_do_not_pollute_host():
    """克隆只读核验和 held-out 执行可用，宿主 relation/use 状态保持不变。"""
    fixture = _fixture()
    cloned_backend = None
    try:
        _form(fixture)
        _record_time(fixture, fixture.before)
        training_request = _request(
            fixture,
            70,
            stance=EVIDENCE_SUPPORT,
        )
        MultiVerifierOrchestrator().run(
            training_request,
            (fixture.runtime.registration(),),
            read_only=False,
        )
        baseline = fixture.runtime.state_key()

        cloned_backend = clone_backend(fixture.backend)
        cloned_ctx = make_train_context(cloned_backend)
        cloned_semantic = _semantic_graph(cloned_ctx.graph_ontology)
        cloned_candidates = CandidateProjectionGraph(
            cloned_ctx.graph_ontology,
            fixture.candidate_graph.protocol,
        )
        cloned_event_facts = EventTimeFactIndex(OrderFactIndex(
            cloned_ctx.graph_ontology,
            cloned_ctx.scoped_identity_store,
        ))
        cloned = fixture.runtime.clone_for_evaluation(
            cloned_semantic,
            cloned_candidates,
            cloned_event_facts,
        )
        held_out = MultiVerifierOrchestrator().run(
            _request(fixture, 71, stance=EVIDENCE_SUPPORT),
            (cloned.registration(),),
            read_only=True,
        )
        assert held_out.results[0].verdict == VERDICT_SUPPORT
        cause, effect = _endpoint_evaluations(fixture)
        executed = cloned.execute(
            fixture.spec.proposition.proposition,
            training_request.temporal,
            cause,
            effect,
            use_key=(_BASE + 170, 1),
            generation_use_key=(_BASE + 170, 2),
        )
        assert executed.execution.status == CAUSAL_EXECUTION_PREDICTED
        assert fixture.runtime.state_key() == baseline
    finally:
        fixture.close()
        if cloned_backend is not None:
            cloned_backend.close()


def test_formal_round_and_v06_clone_use_only_typed_causal_course_requests():
    """正式 round 跑通全链，held-out clone 只读核验和执行且宿主零污染。"""
    backend = DictBackend()
    try:
        domain = _domain()
        ctx = make_train_context(backend)
        runtime = _install_formal_causal(ctx, domain)
        runner = DefaultRoundRunner()
        runner.run_round(ctx, _formal_item(), STAGE1_SKELETON, 1)

        trained = ctx.causal_relation_reports[-1]
        assert trained.read_only is False
        assert trained.formations
        assert trained.verifications[0].results[0].verdict == VERDICT_SUPPORT
        assert trained.verifications[0].results[0].committed_effects
        assert trained.executions[0].execution.status == CAUSAL_EXECUTION_PREDICTED
        assert trained.executions[0].generation_use is not None
        baseline = runtime.state_key()
        report_count = len(ctx.causal_relation_reports)

        with isolated_evaluation(ctx, label="r07-held-out") as eval_ctx:
            runner.run_round(
                eval_ctx,
                _formal_item(),
                STAGE1_SKELETON,
                2,
            )
            held = eval_ctx.causal_relation_reports[-1]
            assert held.read_only is True
            assert held.formations == ()
            assert held.verifications[0].results[0].verdict == VERDICT_SUPPORT
            assert held.verifications[0].results[0].committed_effects == ()
            assert held.executions[0].execution.status == (
                CAUSAL_EXECUTION_PREDICTED)

        assert runtime.state_key() == baseline
        assert len(ctx.causal_relation_reports) == report_count
    finally:
        backend.close()


def test_formal_train_installs_and_returns_causal_reports(tmp_path, monkeypatch):
    """顶层 formal_train 成对安装 R-07 协议和课程并返回真实执行报告。"""
    from pure_integer_ai.training import stages as training_stages

    monkeypatch.setattr(training_stages, "FLOOR_GRAPH_SIZE_S1", 0)
    domain = _domain()
    result = formal_train(
        FormalTrainConfig(
            run_dir=str(tmp_path),
            run_id="r07-formal",
            rounds_per_stage=1,
            active_training_stages=(STAGE1_SKELETON,),
            language_occurrence_protocol=OccurrenceProtocol(
                (_BASE + 184, 1)),
            language_causal_protocol=_FormalProtocol(domain),
            language_causal_course=_FormalCourse(domain),
        ),
        [_formal_item()],
        backend=DictBackend(),
        runner=DefaultRoundRunner(),
    )

    assert result.causal_relation_reports
    report = result.causal_relation_reports[-1]
    assert report.verifications[0].results[0].verdict == VERDICT_SUPPORT
    assert report.executions[0].execution.status == CAUSAL_EXECUTION_PREDICTED
    assert report.executions[0].generation_use is not None


def test_joint_r06b_r07_reuses_filtered_verifier_and_blocks_parallel_time():
    """联合链拒绝平行 writer，裸 statement 也不能绕过 active filter。"""
    backend = DictBackend()
    cloned_backend = None
    try:
        domain = _domain()
        ctx = make_train_context(backend)
        scope = document_scope(domain.spec.proposition.source)
        shared_facts = EventTimeFactIndex(OrderFactIndex(
            ctx.graph_ontology,
            ctx.scoped_identity_store,
        ))
        shared_facts.record(
            domain.before,
            domain.cause,
            domain.effect,
            scope=scope,
            provenance_kind=SOURCE_BARE_TEXT,
            epistemic_origin=EPI_STRUCTURED,
        )
        shared_verifier = EventTimeVerifier(
            shared_facts,
            _EventRelationResolver({
                domain.before: EVENT_TIME_BEFORE,
                domain.after: EVENT_TIME_AFTER,
            }),
            _RejectAllEventTimeFacts(),
        )
        ctx.event_time_relation_runtime = _SharedEventTimeRuntime(
            shared_facts,
            shared_verifier,
        )
        runtime = install_causal_relation_runtime(
            ctx,
            _FormalProtocol(domain),
            _FormalCourse(domain),
        )
        assert runtime.shared_event_time is True
        assert runtime.owner.event_time_facts is shared_facts
        assert runtime.owner.event_time_verifier is shared_verifier

        baseline = runtime.state_key()
        with pytest.raises(ValueError, match="不得直写平行 temporal fact"):
            runtime.process(scope, read_only=False)
        assert runtime.state_key() == baseline
        assert shared_verifier.verify(
            (domain.before,), scope=scope).status == EVENT_TIME_EMPTY

        filtered_runtime = CausalRelationCourseRuntime(
            runtime.owner,
            runtime.protocol_key,
            _FormalCourseWithoutTemporalWrites(_FormalCourse(domain)),
            shared_event_time=True,
        )
        report = filtered_runtime.process(scope, read_only=False)
        assert report.temporal_facts == ()
        assert report.verifications[0].results[0].verdict == VERDICT_UNKNOWN
        assert runtime.owner.relation_runtime.consumer.lookup_proposition(
            domain.spec.proposition.proposition) == ()

        cloned_backend = clone_backend(backend)
        cloned_ctx = make_train_context(cloned_backend)
        cloned_facts = EventTimeFactIndex(OrderFactIndex(
            cloned_ctx.graph_ontology,
            cloned_ctx.scoped_identity_store,
        ))
        cloned_verifier = EventTimeVerifier(
            cloned_facts,
            shared_verifier.resolver,
            _RejectAllEventTimeFacts(),
        )
        cloned_ctx.event_time_relation_runtime = _SharedEventTimeRuntime(
            cloned_facts,
            cloned_verifier,
        )
        cloned_runtime = filtered_runtime.clone_for_context(cloned_ctx)
        assert cloned_runtime.shared_event_time is True
        assert cloned_runtime.owner.event_time_facts is cloned_facts
        assert cloned_runtime.owner.event_time_verifier is cloned_verifier
    finally:
        backend.close()
        if cloned_backend is not None:
            cloned_backend.close()
