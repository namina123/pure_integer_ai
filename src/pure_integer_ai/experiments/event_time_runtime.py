"""R-06B 基于 S-02 Evidence 的事件时间生产、投影和核验闭环。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pure_integer_ai.cognition.shared.event_time import (
    EventTimeFactIndex,
    EventTimeVerifier,
    ResolvedEventTimeRelation,
)
from pure_integer_ai.cognition.shared.hypothesis import (
    EPISTEMIC_SUPPORTED,
    EvidenceRecord,
    HypothesisKey,
    HypothesisLedger,
    LIFECYCLE_ACTIVE,
)
from pure_integer_ai.cognition.shared.hypothesis_resolution import (
    HypothesisResolver,
    ResolverDecision,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONCEPT,
    OBJECT_EVENT,
    OBJECT_PROPOSITION,
    OBJECT_ROLE,
    ObjectIdentity,
    SourceRef,
)
from pure_integer_ai.cognition.shared.order_facts import (
    OrderFact,
    OrderFactIndex,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    LogicalTimestamp,
    ScopeIdentity,
)
from pure_integer_ai.cognition.shared.training_hypothesis import (
    TrainingHypothesisEventSink,
    TrainingHypothesisHistoryProtocol,
)
from pure_integer_ai.crosscut.determinism.hasher import Hasher
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.experiments.event_time_verification import (
    EventTimeVerificationAdapter,
    EventTimeVerificationProtocol,
    EventTimeVerificationRequest,
)
from pure_integer_ai.experiments.language_semantic_runtime import (
    LanguageSemanticCourseRun,
)
from pure_integer_ai.experiments.train_context import TrainContext
from pure_integer_ai.experiments.verification_orchestration import (
    MultiVerifierOrchestrator,
    VerificationReport,
)


_CLAIM_VERSION = 1
_COMPETITION_VERSION = 1
_PROJECTION_VERSION = 1
_EVIDENCE_PAYLOAD_VERSION = 1
_EVIDENCE_HASHER = Hasher("event_time_runtime.evidence.v1")
_ENDPOINT_KINDS = frozenset({OBJECT_EVENT, OBJECT_PROPOSITION})


class EventTimeProductionError(RuntimeError):
    """事件时间课程、上游 Evidence 或 active 投影不一致。"""


def _strict_key(
        value: tuple[int, ...], *, where: str,
        nonnegative: bool = False,
        ) -> tuple[int, ...]:
    """校验并返回非空严格整数协议键。"""
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{where} 必须是非空整数 tuple")
    assert_int(*value, _where=where)
    if any(type(item) is not int for item in value):
        raise TypeError(f"{where} 必须只含严格 int")
    if nonnegative and any(item < 0 for item in value):
        raise ValueError(f"{where} 必须只含非负整数")
    return value


def _pack(value: tuple[int, ...]) -> tuple[int, ...]:
    """给可变长稳定键增加长度边界。"""
    return len(value), *value


def _take(
        values: tuple[int, ...], cursor: int, *, where: str,
        ) -> tuple[tuple[int, ...], int]:
    """从长度前缀整数流读取一个非空键段。"""
    if cursor >= len(values):
        raise ValueError(f"{where} 缺少长度")
    size = values[cursor]
    cursor += 1
    if type(size) is not int or size <= 0 or cursor + size > len(values):
        raise ValueError(f"{where} 长度非法")
    return values[cursor:cursor + size], cursor + size


def _typed_identity(
        value: ObjectIdentity, *, where: str,
        object_kind: int | None = None,
        ) -> ObjectIdentity:
    """核验一等对象及其可选类型，不读取名称或组件语义。"""
    if not isinstance(value, ObjectIdentity):
        raise TypeError(f"{where} 必须是 ObjectIdentity")
    if object_kind is not None and value.object_kind != object_kind:
        raise ValueError(f"{where} 对象类型非法")
    return value


@dataclass(frozen=True)
class EventTimeClaimKey:
    """一个来源 scope 中由 S-02 命题定义的完整时间候选。"""

    scope: ScopeIdentity
    relation: ObjectIdentity
    subject_role: ObjectIdentity
    subject_ordinal: int
    subject: ObjectIdentity
    object_role: ObjectIdentity
    object_ordinal: int
    object_identity: ObjectIdentity
    proposition: ObjectIdentity
    context: ObjectIdentity

    def __post_init__(self) -> None:
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("EventTimeClaimKey.scope 类型非法")
        _typed_identity(
            self.relation,
            where="EventTimeClaimKey.relation",
            object_kind=OBJECT_CONCEPT,
        )
        _typed_identity(
            self.subject_role,
            where="EventTimeClaimKey.subject_role",
            object_kind=OBJECT_ROLE,
        )
        _typed_identity(
            self.object_role,
            where="EventTimeClaimKey.object_role",
            object_kind=OBJECT_ROLE,
        )
        if self.subject_role == self.object_role:
            raise ValueError("event-time subject/object Role 不得相同")
        assert_int(
            self.subject_ordinal,
            self.object_ordinal,
            _where="EventTimeClaimKey ordinals",
        )
        if (type(self.subject_ordinal) is not int
                or type(self.object_ordinal) is not int
                or self.subject_ordinal < 0
                or self.object_ordinal < 0):
            raise ValueError("event-time Role ordinal 必须为非负严格整数")
        for label, endpoint in (
                ("subject", self.subject),
                ("object", self.object_identity)):
            _typed_identity(endpoint, where=f"EventTimeClaimKey.{label}")
            if endpoint.object_kind not in _ENDPOINT_KINDS:
                raise ValueError("event-time 端点必须是 Event 或 Proposition")
        if self.subject == self.object_identity:
            raise ValueError("event-time 候选不得形成自环")
        _typed_identity(
            self.proposition,
            where="EventTimeClaimKey.proposition",
            object_kind=OBJECT_PROPOSITION,
        )
        _typed_identity(self.context, where="EventTimeClaimKey.context")
        if (self.subject.owner != self.object_identity.owner
                or self.subject.owner != self.scope.owner):
            raise ValueError("event-time 端点与来源 scope owner 不一致")
        if (self.proposition.owner != self.scope.owner
                or self.context.owner != self.scope.owner):
            raise ValueError("event-time Proposition/context 与来源 scope owner 不一致")

    def stable_key(self) -> tuple[int, ...]:
        """展开候选全部本体、Role、scope 和上下文身份。"""
        return (
            _CLAIM_VERSION,
            *_pack(self.scope.stable_key()),
            *_pack(self.relation.stable_key()),
            *_pack(self.subject_role.stable_key()),
            self.subject_ordinal,
            *_pack(self.subject.stable_key()),
            *_pack(self.object_role.stable_key()),
            self.object_ordinal,
            *_pack(self.object_identity.stable_key()),
            *_pack(self.proposition.stable_key()),
            *_pack(self.context.stable_key()),
        )

    @classmethod
    def from_stable_key(
            cls, key: tuple[int, ...],
            ) -> "EventTimeClaimKey":
        """从完整整数键恢复候选，拒绝截断和尾随字段。"""
        key = _strict_key(key, where="EventTimeClaimKey.stable_key")
        if key[0] != _CLAIM_VERSION:
            raise ValueError("EventTimeClaimKey 版本非法")
        scope_key, cursor = _take(key, 1, where="claim.scope")
        relation_key, cursor = _take(key, cursor, where="claim.relation")
        subject_role_key, cursor = _take(
            key, cursor, where="claim.subject_role")
        if cursor >= len(key):
            raise ValueError("claim 缺少 subject ordinal")
        subject_ordinal = key[cursor]
        subject_key, cursor = _take(
            key, cursor + 1, where="claim.subject")
        object_role_key, cursor = _take(
            key, cursor, where="claim.object_role")
        if cursor >= len(key):
            raise ValueError("claim 缺少 object ordinal")
        object_ordinal = key[cursor]
        object_key, cursor = _take(
            key, cursor + 1, where="claim.object")
        proposition_key, cursor = _take(
            key, cursor, where="claim.proposition")
        context_key, cursor = _take(
            key, cursor, where="claim.context")
        if cursor != len(key):
            raise ValueError("EventTimeClaimKey 含尾随字段")
        restored = cls(
            ScopeIdentity.from_stable_key(scope_key),
            ObjectIdentity.from_stable_key(relation_key),
            ObjectIdentity.from_stable_key(subject_role_key),
            subject_ordinal,
            ObjectIdentity.from_stable_key(subject_key),
            ObjectIdentity.from_stable_key(object_role_key),
            object_ordinal,
            ObjectIdentity.from_stable_key(object_key),
            ObjectIdentity.from_stable_key(proposition_key),
            ObjectIdentity.from_stable_key(context_key),
        )
        if restored.stable_key() != key:
            raise ValueError("EventTimeClaimKey 不能规范化重放")
        return restored


@dataclass(frozen=True)
class EventTimeEvidenceRequest:
    """课程引用同轮 S-02 命题和上游 Evidence 的时间观察。"""

    relation: ObjectIdentity
    proposition: ObjectIdentity
    subject_role: ObjectIdentity
    subject_ordinal: int
    subject: ObjectIdentity
    object_role: ObjectIdentity
    object_ordinal: int
    object_identity: ObjectIdentity
    upstream_evidence_id: int
    supersedes_assertion_hash: int = 0
    supersede_timestamp: LogicalTimestamp | None = None

    def __post_init__(self) -> None:
        _typed_identity(
            self.relation,
            where="event-time request relation",
            object_kind=OBJECT_CONCEPT,
        )
        _typed_identity(
            self.proposition,
            where="event-time request proposition",
            object_kind=OBJECT_PROPOSITION,
        )
        _typed_identity(
            self.subject_role,
            where="event-time request subject Role",
            object_kind=OBJECT_ROLE,
        )
        _typed_identity(
            self.object_role,
            where="event-time request object Role",
            object_kind=OBJECT_ROLE,
        )
        for endpoint in (self.subject, self.object_identity):
            _typed_identity(endpoint, where="event-time request endpoint")
            if endpoint.object_kind not in _ENDPOINT_KINDS:
                raise ValueError("event-time request 端点类型非法")
        assert_int(
            self.subject_ordinal,
            self.object_ordinal,
            self.upstream_evidence_id,
            self.supersedes_assertion_hash,
            _where="EventTimeEvidenceRequest",
        )
        if (type(self.subject_ordinal) is not int
                or type(self.object_ordinal) is not int
                or self.subject_ordinal < 0
                or self.object_ordinal < 0):
            raise ValueError("event-time request ordinal 非法")
        if (type(self.upstream_evidence_id) is not int
                or self.upstream_evidence_id <= 0):
            raise ValueError("upstream_evidence_id 必须为严格正整数")
        if (type(self.supersedes_assertion_hash) is not int
                or self.supersedes_assertion_hash < 0):
            raise ValueError("supersedes_assertion_hash 必须为非负整数")
        if ((self.supersedes_assertion_hash == 0)
                != (self.supersede_timestamp is None)):
            raise ValueError("assertion supersede hash 与 timestamp 必须成对")
        if (self.supersede_timestamp is not None
                and not isinstance(
                    self.supersede_timestamp, LogicalTimestamp)):
            raise TypeError("supersede_timestamp 类型非法")


@dataclass(frozen=True)
class EventTimeCourseInput:
    """提供给课程 mapper 的同轮 S-02 产物和只读边界。"""

    scope: ScopeIdentity
    semantic_run: LanguageSemanticCourseRun
    read_only: bool

    def __post_init__(self) -> None:
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("event-time course scope 类型非法")
        if not isinstance(self.semantic_run, LanguageSemanticCourseRun):
            raise TypeError("event-time course semantic_run 类型非法")
        if type(self.read_only) is not bool:
            raise TypeError("event-time course read_only 必须是 bool")
        semantic_input = self.semantic_run.input_value
        if (semantic_input.occurrence_scope != self.scope
                or semantic_input.source != self.scope.source):
            raise ValueError("event-time course 与 S-02 来源 scope 不一致")
        if semantic_input.read_only != self.read_only:
            raise ValueError("event-time course 与 S-02 只读状态不一致")


@dataclass(frozen=True)
class EventTimeRoundRequest:
    """一个来源 scope 下的时间 Evidence 和 R-09 核验请求。"""

    scope: ScopeIdentity
    evidence: tuple[EventTimeEvidenceRequest, ...] = ()
    verifications: tuple[EventTimeVerificationRequest, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("event-time round scope 类型非法")
        if (not isinstance(self.evidence, tuple)
                or any(not isinstance(item, EventTimeEvidenceRequest)
                       for item in self.evidence)):
            raise TypeError("event-time round evidence 类型非法")
        if (not isinstance(self.verifications, tuple)
                or any(not isinstance(item, EventTimeVerificationRequest)
                       for item in self.verifications)):
            raise TypeError("event-time round verifications 类型非法")
        if any(item.scope != self.scope for item in self.verifications):
            raise ValueError("event-time verification 必须绑定当前 scope")
        routes = tuple(
            (
                item.relation,
                item.proposition,
                item.subject_role,
                item.subject_ordinal,
                item.subject,
                item.object_role,
                item.object_ordinal,
                item.object_identity,
                item.upstream_evidence_id,
            )
            for item in self.evidence
        )
        if len(set(routes)) != len(routes):
            raise ValueError("同一 event-time Evidence 路由不得重复")


@dataclass(frozen=True)
class EventTimeProductionProtocol:
    """声明事件时间 H-00、历史、投影限定项和 R-09 协议。"""

    hypothesis_kind: tuple[int, ...]
    aggregate_source: SourceRef
    aggregate_scope: ScopeIdentity
    history_namespace: tuple[int, ...]
    competition_namespace: tuple[int, ...]
    evidence_reason_key: tuple[int, ...]
    evidence_namespace: tuple[int, ...]
    projection_qualifier_key: tuple[int, ...]
    verification: EventTimeVerificationProtocol
    provenance_kind: int
    epistemic_origin: int = 0
    content_version: int = 0

    def __post_init__(self) -> None:
        _strict_key(self.hypothesis_kind, where="event-time hypothesis kind")
        _strict_key(self.history_namespace, where="event-time history namespace")
        _strict_key(
            self.competition_namespace,
            where="event-time competition namespace",
        )
        _strict_key(
            self.evidence_reason_key,
            where="event-time evidence reason",
        )
        _strict_key(
            self.evidence_namespace,
            where="event-time evidence namespace",
        )
        _strict_key(
            self.projection_qualifier_key,
            where="event-time projection qualifier",
            nonnegative=True,
        )
        if not isinstance(self.aggregate_source, SourceRef):
            raise TypeError("event-time aggregate_source 类型非法")
        if (not isinstance(self.aggregate_scope, ScopeIdentity)
                or self.aggregate_scope.source != self.aggregate_source):
            raise ValueError("event-time aggregate_scope 必须指向 aggregate_source")
        if not isinstance(self.verification, EventTimeVerificationProtocol):
            raise TypeError("event-time verification protocol 类型非法")
        assert_int(
            self.provenance_kind,
            self.epistemic_origin,
            self.content_version,
            _where="EventTimeProductionProtocol metadata",
        )
        if type(self.provenance_kind) is not int or self.provenance_kind <= 0:
            raise ValueError("event-time provenance_kind 必须为正整数")
        if (type(self.epistemic_origin) is not int
                or type(self.content_version) is not int
                or self.epistemic_origin < 0
                or self.content_version < 0):
            raise ValueError("event-time metadata 必须为非负严格整数")

    def history_protocol(self) -> TrainingHypothesisHistoryProtocol:
        """建立与物理 backend 无关的 M-03 Core 训练历史边界。"""
        return TrainingHypothesisHistoryProtocol(
            self.history_namespace,
            self.hypothesis_kind,
            self.aggregate_source,
            self.aggregate_scope,
        )

    def stable_key(self) -> tuple[int, ...]:
        """返回生产协议全部开放键、来源和写入元数据。"""
        values: list[int] = [1]
        for key in (
                self.hypothesis_kind,
                self.aggregate_source.stable_key(),
                self.aggregate_scope.stable_key(),
                self.history_namespace,
                self.competition_namespace,
                self.evidence_reason_key,
                self.evidence_namespace,
                self.projection_qualifier_key,
                self.verification.dimension.stable_key(),
                self.verification.verifier.stable_key()):
            values.extend(_pack(key))
        values.extend((
            self.provenance_kind,
            self.epistemic_origin,
            self.content_version,
        ))
        return tuple(values)


@runtime_checkable
class EventTimeProductionSemantics(Protocol):
    """提供 relation 方向解释、评测克隆和可比较状态。"""

    def resolve(
            self, relation: ObjectIdentity,
            ) -> ResolvedEventTimeRelation:
        """把开放 relation 解释为最小方向执行协议。"""
        ...

    def clone_for_evaluation(self) -> "EventTimeProductionSemantics":
        """返回不共享可变语义注册表的评测副本。"""
        ...

    def state_key(self) -> tuple[int, ...]:
        """返回 relation 注册和版本的完整整数状态。"""
        ...


@runtime_checkable
class EventTimeProductionCourse(Protocol):
    """把同轮 S-02 产物映射为时间 Evidence 和核验请求。"""

    def request(self, value: EventTimeCourseInput) -> EventTimeRoundRequest:
        """返回当前来源 scope 的全部 typed 时间请求。"""
        ...

    def clone_for_evaluation(self) -> "EventTimeProductionCourse":
        """返回不共享可变课程状态的评测副本。"""
        ...

    def state_key(self) -> tuple[int, ...]:
        """返回 mapper、版本和来源策略的完整整数状态。"""
        ...


@dataclass(frozen=True)
class EventTimeEvidenceTrace:
    """保存上游 Evidence、独立时间 Evidence、决策和可选投影。"""

    request: EventTimeEvidenceRequest
    hypothesis: HypothesisKey
    upstream_evidence: EvidenceRecord
    evidence: EvidenceRecord
    decision: ResolverDecision
    fact: OrderFact | None
    duplicate: bool

    def __post_init__(self) -> None:
        if not isinstance(self.request, EventTimeEvidenceRequest):
            raise TypeError("event-time trace request 类型非法")
        if not isinstance(self.hypothesis, HypothesisKey):
            raise TypeError("event-time trace hypothesis 类型非法")
        if (not isinstance(self.upstream_evidence, EvidenceRecord)
                or not isinstance(self.evidence, EvidenceRecord)):
            raise TypeError("event-time trace Evidence 类型非法")
        if not isinstance(self.decision, ResolverDecision):
            raise TypeError("event-time trace decision 类型非法")
        if self.fact is not None and not isinstance(self.fact, OrderFact):
            raise TypeError("event-time trace fact 类型非法")
        if type(self.duplicate) is not bool:
            raise TypeError("event-time trace duplicate 必须是 bool")


@dataclass(frozen=True)
class EventTimeRoundReport:
    """一个来源 scope 的时间学习、投影和分维核验结果。"""

    scope: ScopeIdentity
    read_only: bool
    evidence: tuple[EventTimeEvidenceTrace, ...]
    verifications: tuple[VerificationReport, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("event-time report scope 类型非法")
        if type(self.read_only) is not bool:
            raise TypeError("event-time report read_only 必须是 bool")
        if (not isinstance(self.evidence, tuple)
                or any(not isinstance(item, EventTimeEvidenceTrace)
                       for item in self.evidence)):
            raise TypeError("event-time report evidence 类型非法")
        if (not isinstance(self.verifications, tuple)
                or any(not isinstance(item, VerificationReport)
                       for item in self.verifications)):
            raise TypeError("event-time report verifications 类型非法")
        if self.read_only and self.evidence:
            raise ValueError("read-only event-time report 不得携带训练 Evidence")


@dataclass(frozen=True)
class _PreparedEvidence:
    """一次写前已完成 S-02、H-00 和 H-04 纯校验的内部请求。"""

    request: EventTimeEvidenceRequest
    claim: EventTimeClaimKey
    hypothesis: HypothesisKey
    upstream: EvidenceRecord
    semantic_trace: tuple[int, ...]
    evidence: EvidenceRecord
    decision: ResolverDecision
    duplicate: bool


@dataclass(frozen=True)
class _ValidatedEvidenceInput:
    """已完成无状态 S-02 交叉核验、可进入局部 H-00 试算的输入。"""

    request: EventTimeEvidenceRequest
    claim: EventTimeClaimKey
    hypothesis: HypothesisKey
    upstream: EvidenceRecord
    semantic_trace: tuple[int, ...]


class _EventTimeHypothesisAcceptance:
    """从可逆 assertion 限定项回查当前 H-00/H-04 采用状态。"""

    def __init__(
            self,
            protocol: EventTimeProductionProtocol,
            ledger: HypothesisLedger,
            resolver: HypothesisResolver,
            facts: EventTimeFactIndex,
            ) -> None:
        self.protocol = protocol
        self.ledger = ledger
        self.resolver = resolver
        self.facts = facts

    def qualifiers(self, hypothesis: HypothesisKey) -> tuple[int, ...]:
        """编码投影协议和完整 Hypothesis，不用摘要替代身份。"""
        key = hypothesis.stable_key()
        return (
            _PROJECTION_VERSION,
            *_pack(self.protocol.projection_qualifier_key),
            *_pack(key),
        )

    def accepts(self, fact: OrderFact) -> bool:
        """只接受限定项、候选键、事实端点和最新 adopted 状态全一致的投影。"""
        hypothesis = self._hypothesis_from(fact)
        if hypothesis is None:
            return False
        if not self.protocol.history_protocol().accepts(hypothesis):
            raise EventTimeProductionError("event-time 投影引用了其他历史协议")
        claim = EventTimeClaimKey.from_stable_key(hypothesis.candidate_key)
        self._require_fact_claim(fact, claim)
        snapshot = self.ledger.snapshot(hypothesis)
        decisions = self.resolver.decision_history(hypothesis)
        if not decisions:
            return False
        return (
            snapshot.lifecycle == LIFECYCLE_ACTIVE
            and snapshot.epistemic_status == EPISTEMIC_SUPPORTED
            and hypothesis in decisions[-1].adopted_hypotheses
        )

    def _hypothesis_from(self, fact: OrderFact) -> HypothesisKey | None:
        """识别本协议限定项；其他领域 statement 返回不可用。"""
        qualifiers = fact.statement.assertion.qualifiers
        if not qualifiers or qualifiers[0] != _PROJECTION_VERSION:
            return None
        try:
            protocol_key, cursor = _take(
                qualifiers, 1, where="event-time projection protocol")
        except ValueError:
            return None
        if protocol_key != self.protocol.projection_qualifier_key:
            return None
        hypothesis_key, cursor = _take(
            qualifiers, cursor, where="event-time projection hypothesis")
        if cursor != len(qualifiers):
            raise EventTimeProductionError("event-time 投影限定项含尾随字段")
        return HypothesisKey.from_stable_key(hypothesis_key)

    def _require_fact_claim(
            self, fact: OrderFact, claim: EventTimeClaimKey,
            ) -> None:
        """核验 raw statement 没有替换候选 relation、端点或 scope。"""
        statement = fact.statement
        if statement.assertion.scope != claim.scope:
            raise EventTimeProductionError("event-time 投影 scope 与候选不一致")
        if statement.assertion.subject != statement.subject:
            raise EventTimeProductionError("event-time 投影 subject assertion 漂移")
        if statement.assertion.object != statement.object:
            raise EventTimeProductionError("event-time 投影 object assertion 漂移")
        ontology = self.facts.ontology
        if ontology.identity_of(statement.predicate) != claim.relation:
            raise EventTimeProductionError("event-time 投影 relation 与候选不一致")
        if ontology.identity_of(statement.subject) != claim.subject:
            raise EventTimeProductionError("event-time 投影 subject 与候选不一致")
        if ontology.identity_of(statement.object) != claim.object_identity:
            raise EventTimeProductionError("event-time 投影 object 与候选不一致")


class EventTimeProductionRuntime:
    """执行 S-02 Evidence 映射、持久 H-00/H-04、active 投影和 R-09。"""

    def __init__(
            self,
            ctx: TrainContext,
            protocol: EventTimeProductionProtocol,
            semantics: EventTimeProductionSemantics,
            course: EventTimeProductionCourse,
            ) -> None:
        if not isinstance(ctx, TrainContext):
            raise TypeError("event-time runtime ctx 类型非法")
        if not isinstance(protocol, EventTimeProductionProtocol):
            raise TypeError("event-time runtime protocol 类型非法")
        if not isinstance(semantics, EventTimeProductionSemantics):
            raise TypeError("event-time semantics 协议不完整")
        if not isinstance(course, EventTimeProductionCourse):
            raise TypeError("event-time course 协议不完整")
        _strict_key(semantics.state_key(), where="event-time semantics state")
        _strict_key(course.state_key(), where="event-time course state")
        semantic_runtime = ctx.language_semantic_course_runtime
        if semantic_runtime is None:
            raise ValueError("event-time production 需要正式 S-02 semantic runtime")
        if ctx.training_candidate_history is None:
            raise ValueError("event-time production 缺少 M-03 Core 训练历史")
        self.protocol = protocol
        self.semantics = semantics
        self.course = course
        self.semantic_runtime = semantic_runtime
        self.history_sink = TrainingHypothesisEventSink(
            ctx.training_candidate_history,
            protocol.history_protocol(),
        )
        self.ledger = self.history_sink.load_ledger(attach_sink=True)
        self.resolver = HypothesisResolver.from_history(
            self.ledger,
            self.history_sink.load_decisions(),
            sink=self.history_sink,
        )
        entries = ctx.training_candidate_history.entries(
            protocol.history_protocol())
        self._logical_clock = max(
            (item.event_seq for item in entries),
            default=0,
        )
        self.facts = EventTimeFactIndex(OrderFactIndex(
            ctx.graph_ontology,
            ctx.scoped_identity_store,
        ))
        self.acceptance = _EventTimeHypothesisAcceptance(
            protocol,
            self.ledger,
            self.resolver,
            self.facts,
        )
        self.verifier = EventTimeVerifier(
            self.facts,
            semantics,
            self.acceptance,
        )
        self.adapter = EventTimeVerificationAdapter(
            self.verifier,
            protocol.verification,
        )

    def process(
            self,
            scope: ScopeIdentity,
            semantic_run: LanguageSemanticCourseRun,
            *,
            read_only: bool,
            ) -> EventTimeRoundReport:
        """处理同轮语义产物；训练可写，held-out 只核验克隆状态。"""
        value = EventTimeCourseInput(scope, semantic_run, read_only)
        request = self.course.request(value)
        if not isinstance(request, EventTimeRoundRequest):
            raise TypeError("event-time course.request 返回类型非法")
        if request.scope != scope:
            raise ValueError("event-time course 替换了 round scope")
        if read_only and request.evidence:
            raise ValueError("held-out event-time 请求不得现场写 Evidence")
        traces = () if read_only else self._learn(request, semantic_run)
        orchestrator = MultiVerifierOrchestrator()
        verifications = tuple(
            orchestrator.run(
                item,
                (self.adapter.registration(),),
                read_only=read_only,
            )
            for item in request.verifications
        )
        return EventTimeRoundReport(
            scope,
            read_only,
            traces,
            verifications,
        )

    def clone_for_context(
            self, ctx: TrainContext,
            ) -> "EventTimeProductionRuntime":
        """在克隆语义 runtime、图和训练历史上重建独立 R-06B owner。"""
        semantics = self.semantics.clone_for_evaluation()
        course = self.course.clone_for_evaluation()
        if not isinstance(semantics, EventTimeProductionSemantics):
            raise TypeError("event-time semantics clone 协议不完整")
        if not isinstance(course, EventTimeProductionCourse):
            raise TypeError("event-time course clone 协议不完整")
        return EventTimeProductionRuntime(
            ctx,
            self.protocol,
            semantics,
            course,
        )

    def evidence_count(self) -> int:
        """返回 event-time H-00 中含历史替代事件的 Evidence 总数。"""
        return sum(
            len(self.ledger.evidence_history(hypothesis))
            for hypothesis in self.ledger.hypotheses()
        )

    def state_key(self) -> tuple:
        """返回协议、课程、语义、H-00/H-04 和逻辑序的完整状态。"""
        return (
            self.protocol.stable_key(),
            self.semantics.state_key(),
            self.course.state_key(),
            self.ledger.state_key(),
            self.resolver.state_key(),
            self._logical_clock,
        )

    def _learn(
            self,
            request: EventTimeRoundRequest,
            semantic_run: LanguageSemanticCourseRun,
            ) -> tuple[EventTimeEvidenceTrace, ...]:
        """整批预检后提交 Evidence，再只投影最终 adopted 的时间事实。"""
        prepared, final_clock = self._prepare_batch(
            request.evidence,
            semantic_run,
        )
        for item in prepared:
            if item.duplicate:
                continue
            self.ledger.register(item.hypothesis)
            self.ledger.append_evidence(item.evidence)
            decision = self.resolver.resolve(
                item.hypothesis,
                timestamp_seq=item.decision.timestamp_seq,
            )
            if decision != item.decision:
                raise EventTimeProductionError("event-time H-04 提交结果与预检不一致")
        self._logical_clock = final_clock

        fact_by_hypothesis: dict[HypothesisKey, OrderFact | None] = {}
        request_by_hypothesis: dict[HypothesisKey, EventTimeEvidenceRequest] = {}
        claim_by_hypothesis: dict[HypothesisKey, EventTimeClaimKey] = {}
        for item in prepared:
            prior_request = request_by_hypothesis.get(item.hypothesis)
            if (prior_request is not None
                    and (prior_request.supersedes_assertion_hash
                         != item.request.supersedes_assertion_hash
                         or prior_request.supersede_timestamp
                         != item.request.supersede_timestamp)):
                raise ValueError("同一 event-time 候选声明了竞争 assertion supersede")
            request_by_hypothesis[item.hypothesis] = item.request
            claim_by_hypothesis[item.hypothesis] = item.claim

        for hypothesis in sorted(
                claim_by_hypothesis,
                key=HypothesisKey.stable_key):
            if not self._is_adopted(hypothesis):
                fact_by_hypothesis[hypothesis] = None
                continue
            claim = claim_by_hypothesis[hypothesis]
            fact = self.facts.record_existing(
                claim.relation,
                claim.subject,
                claim.object_identity,
                scope=claim.scope,
                provenance_kind=self.protocol.provenance_kind,
                epistemic_origin=self.protocol.epistemic_origin,
                content_version=self.protocol.content_version,
                qualifiers=self.acceptance.qualifiers(hypothesis),
            )
            supersede_request = request_by_hypothesis[hypothesis]
            if supersede_request.supersedes_assertion_hash:
                old = self.facts.fact_by_assertion_hash(
                    supersede_request.supersedes_assertion_hash)
                self.facts.supersede(
                    old,
                    fact,
                    supersede_request.supersede_timestamp,
                )
            fact_by_hypothesis[hypothesis] = fact

        return tuple(
            EventTimeEvidenceTrace(
                item.request,
                item.hypothesis,
                item.upstream,
                item.evidence,
                self._latest_decision(item.hypothesis),
                fact_by_hypothesis[item.hypothesis],
                item.duplicate,
            )
            for item in prepared
        )

    def _prepare_batch(
            self,
            requests: tuple[EventTimeEvidenceRequest, ...],
            semantic_run: LanguageSemanticCourseRun,
            ) -> tuple[tuple[_PreparedEvidence, ...], int]:
        """只克隆本批竞争组，核验全部 Evidence 和 H-04 决策顺序。"""
        if not requests:
            return (), self._logical_clock
        validated: list[_ValidatedEvidenceInput] = []
        for request in requests:
            claim, upstream, semantic_trace = self._claim_and_upstream(
                request,
                semantic_run,
            )
            validated.append(_ValidatedEvidenceInput(
                request,
                claim,
                self._hypothesis(claim),
                upstream,
                semantic_trace,
            ))
        anchors = tuple(item.hypothesis for item in validated)
        probe_ledger = self.ledger.clone_competitions(anchors)
        probe_resolver = self.resolver.clone_competitions(
            ledger=probe_ledger,
            anchors=anchors,
        )
        next_clock = self._logical_clock
        prepared: list[_PreparedEvidence] = []
        for item in validated:
            evidence_id = self._evidence_id(
                item.hypothesis,
                item.upstream,
            )
            existing = self._evidence_if_present(
                probe_ledger,
                item.hypothesis,
                evidence_id,
            )
            if existing is not None:
                expected = self._event_evidence(
                    item.hypothesis,
                    item.upstream,
                    item.claim,
                    item.semantic_trace,
                    existing.timestamp_seq,
                    probe_ledger,
                )
                if existing != expected:
                    raise EventTimeProductionError(
                        "同一 event-time Evidence id 绑定不同完整事件")
                decision = self._latest_decision_from(
                    probe_resolver,
                    item.hypothesis,
                )
                prepared.append(_PreparedEvidence(
                    item.request,
                    item.claim,
                    item.hypothesis,
                    item.upstream,
                    item.semantic_trace,
                    existing,
                    decision,
                    True,
                ))
                continue
            next_clock += 1
            probe_ledger.register(item.hypothesis)
            evidence = self._event_evidence(
                item.hypothesis,
                item.upstream,
                item.claim,
                item.semantic_trace,
                next_clock,
                probe_ledger,
            )
            probe_ledger.append_evidence(evidence)
            next_clock += 1
            decision = probe_resolver.resolve(
                item.hypothesis,
                timestamp_seq=next_clock,
            )
            prepared.append(_PreparedEvidence(
                item.request,
                item.claim,
                item.hypothesis,
                item.upstream,
                item.semantic_trace,
                evidence,
                decision,
                False,
            ))
        return tuple(prepared), next_clock

    def _claim_and_upstream(
            self,
            request: EventTimeEvidenceRequest,
            semantic_run: LanguageSemanticCourseRun,
            ) -> tuple[EventTimeClaimKey, EvidenceRecord, tuple[int, ...]]:
        """核验请求正好引用同轮 S-02 predicate、RoleBinding 和 active Evidence。"""
        materialized = semantic_run.materialized
        if materialized is None:
            raise ValueError("event-time 训练请求缺少同轮 S-02 物化产物")
        matches = tuple(
            candidate
            for candidate in materialized.candidates
            if candidate.atomic.definition.proposition == request.proposition
        )
        if len(matches) != 1:
            raise ValueError("event-time Proposition 未在同轮 S-02 中唯一物化")
        candidate = matches[0]
        definition = candidate.atomic.definition
        build = semantic_run.build
        if build is None:
            raise ValueError("event-time 训练请求缺少同轮 S-02 build 产物")
        built_matches = tuple(
            item
            for item in build.propositions
            if item.definition.proposition == request.proposition
        )
        if len(built_matches) != 1:
            raise ValueError("event-time Proposition 未在同轮 S-02 build 中唯一出现")
        built = built_matches[0]
        if (built.definition != definition
                or built.upstream_hypothesis.object_identity()
                != candidate.upstream_hypothesis):
            raise ValueError("event-time S-02 build 与 materialized trace 不一致")
        proposition_hypothesis = built.hypothesis
        if definition.predicate != request.relation:
            raise ValueError("event-time relation 必须等于 S-02 Proposition predicate")
        self._require_binding(
            definition.bindings,
            request.subject_role,
            request.subject_ordinal,
            request.subject,
            label="subject",
        )
        self._require_binding(
            definition.bindings,
            request.object_role,
            request.object_ordinal,
            request.object_identity,
            label="object",
        )
        if request.subject == request.object_identity:
            raise ValueError("event-time S-02 Role 绑定不得形成自环")
        upstream_matches = tuple(
            evidence
            for evidence in semantic_run.evidence
            if (evidence.evidence_id == request.upstream_evidence_id
                and evidence.hypothesis == proposition_hypothesis)
        )
        if len(upstream_matches) != 1:
            raise ValueError("event-time 请求未引用同轮 Proposition Evidence")
        upstream = upstream_matches[0]
        snapshot = self.semantic_runtime.ledger.snapshot(
            proposition_hypothesis)
        active_ids = frozenset((
            *snapshot.support_evidence_ids,
            *snapshot.refute_evidence_ids,
            *snapshot.unknown_evidence_ids,
        ))
        if (snapshot.lifecycle != LIFECYCLE_ACTIVE
                or upstream.evidence_id not in active_ids):
            raise ValueError("event-time 上游 Evidence 当前不是 active")
        claim = EventTimeClaimKey(
            semantic_run.input_value.occurrence_scope,
            request.relation,
            request.subject_role,
            request.subject_ordinal,
            request.subject,
            request.object_role,
            request.object_ordinal,
            request.object_identity,
            request.proposition,
            definition.context,
        )
        if (request.supersede_timestamp is not None
                and request.supersede_timestamp.clock.scope != claim.scope):
            raise ValueError("event-time assertion supersede 时钟必须归当前 scope")
        semantic_trace = tuple(sorted((
            *candidate.atomic.assertion_hashes,
            *candidate.trace_assertion_hashes,
        )))
        return claim, upstream, semantic_trace

    @staticmethod
    def _require_binding(
            bindings,
            role: ObjectIdentity,
            ordinal: int,
            filler: ObjectIdentity,
            *,
            label: str,
            ) -> None:
        """要求 S-02 原子命题存在唯一且精确的 Role/ordinal/filler 绑定。"""
        matches = tuple(
            item
            for item in bindings
            if item.role == role and item.ordinal == ordinal
        )
        if len(matches) != 1 or matches[0].filler != filler:
            raise ValueError(f"event-time {label} RoleBinding 与 S-02 不一致")

    def _hypothesis(self, claim: EventTimeClaimKey) -> HypothesisKey:
        """把来源化时间候选映射到独立 aggregate H-00 竞争组。"""
        endpoints = sorted(
            (claim.subject.stable_key(), claim.object_identity.stable_key()),
        )
        competition = (
            _COMPETITION_VERSION,
            *_pack(self.protocol.competition_namespace),
            *_pack(claim.scope.stable_key()),
            *_pack(claim.context.stable_key()),
            *_pack(endpoints[0]),
            *_pack(endpoints[1]),
        )
        return HypothesisKey(
            self.protocol.hypothesis_kind,
            claim.stable_key(),
            competition,
            self.protocol.aggregate_scope,
            self.protocol.aggregate_source,
        )

    def _event_evidence(
            self,
            hypothesis: HypothesisKey,
            upstream: EvidenceRecord,
            claim: EventTimeClaimKey,
            semantic_trace: tuple[int, ...],
            timestamp_seq: int,
            ledger: HypothesisLedger,
            ) -> EvidenceRecord:
        """继承上游立场并保存完整 S-02 Evidence、候选和替代链。"""
        supersedes = 0
        if upstream.supersedes_evidence_id:
            prior = self._semantic_evidence(
                upstream.hypothesis,
                upstream.supersedes_evidence_id,
            )
            prior_id = self._evidence_id(hypothesis, prior)
            if self._evidence_if_present(
                    ledger, hypothesis, prior_id) is not None:
                supersedes = prior_id
        upstream_key = upstream.stable_key()
        claim_key = claim.stable_key()
        payload = (
            _EVIDENCE_PAYLOAD_VERSION,
            *_pack(claim_key),
            *_pack(upstream_key),
            *_pack(semantic_trace),
        )
        return EvidenceRecord(
            self._evidence_id(hypothesis, upstream),
            hypothesis,
            upstream.stance,
            self.protocol.evidence_reason_key,
            upstream.source,
            timestamp_seq,
            payload,
            supersedes,
        )

    def _semantic_evidence(
            self, hypothesis: HypothesisKey, evidence_id: int,
            ) -> EvidenceRecord:
        """从 S-02 ledger 恢复指定旧 Evidence，拒绝跨候选引用。"""
        matches = tuple(
            item
            for item in self.semantic_runtime.ledger.evidence_history(
                hypothesis)
            if item.evidence_id == evidence_id
        )
        if len(matches) != 1:
            raise ValueError("S-02 supersede 目标 Evidence 缺失或不唯一")
        return matches[0]

    def _evidence_id(
            self, hypothesis: HypothesisKey, upstream: EvidenceRecord,
            ) -> int:
        """从注入 namespace、完整候选和上游事件生成确定性 Evidence id。"""
        values = (
            *_pack(self.protocol.evidence_namespace),
            *_pack(hypothesis.stable_key()),
            *_pack(upstream.stable_key()),
        )
        return _EVIDENCE_HASHER.h63(values) or 1

    @staticmethod
    def _evidence_if_present(
            ledger: HypothesisLedger,
            hypothesis: HypothesisKey,
            evidence_id: int,
            ) -> EvidenceRecord | None:
        """按候选和 id 返回既有 Evidence；候选未登记时返回空。"""
        if not ledger.has_hypothesis(hypothesis):
            return None
        matches = tuple(
            item
            for item in ledger.evidence_history(hypothesis)
            if item.evidence_id == evidence_id
        )
        if len(matches) > 1:
            raise EventTimeProductionError("event-time ledger 含重复 Evidence id")
        return None if not matches else matches[0]

    def _is_adopted(self, hypothesis: HypothesisKey) -> bool:
        """按最新 H-00 快照和 H-04 决策判断候选是否可投影。"""
        snapshot = self.ledger.snapshot(hypothesis)
        decision = self._latest_decision(hypothesis)
        return (
            snapshot.lifecycle == LIFECYCLE_ACTIVE
            and snapshot.epistemic_status == EPISTEMIC_SUPPORTED
            and hypothesis in decision.adopted_hypotheses
        )

    def _latest_decision(self, hypothesis: HypothesisKey) -> ResolverDecision:
        """返回正式 resolver 中一个竞争组的最新完整决策。"""
        return self._latest_decision_from(self.resolver, hypothesis)

    @staticmethod
    def _latest_decision_from(
            resolver: HypothesisResolver,
            hypothesis: HypothesisKey,
            ) -> ResolverDecision:
        """返回指定 resolver 的最新决策，缺失时 fail closed。"""
        decisions = resolver.decision_history(hypothesis)
        if not decisions:
            raise EventTimeProductionError("event-time Hypothesis 缺少 H-04 决策")
        return decisions[-1]


def install_event_time_relation_runtime(
        ctx: TrainContext,
        protocol: EventTimeProductionProtocol,
        semantics: EventTimeProductionSemantics,
        course: EventTimeProductionCourse,
        ) -> EventTimeProductionRuntime:
    """在正式 TrainContext 安装 S-02 后的独立 R-06B production owner。"""
    if ctx.event_time_relation_runtime is not None:
        raise ValueError("TrainContext 已安装 event-time relation runtime")
    runtime = EventTimeProductionRuntime(
        ctx,
        protocol,
        semantics,
        course,
    )
    ctx.event_time_relation_runtime = runtime
    return runtime


__all__ = [
    "EventTimeClaimKey",
    "EventTimeCourseInput",
    "EventTimeEvidenceRequest",
    "EventTimeEvidenceTrace",
    "EventTimeProductionCourse",
    "EventTimeProductionError",
    "EventTimeProductionProtocol",
    "EventTimeProductionRuntime",
    "EventTimeProductionSemantics",
    "EventTimeRoundReport",
    "EventTimeRoundRequest",
    "install_event_time_relation_runtime",
]
