"""L-06 occurrence 顺序到 H-06/S-07 消费的独立生产 runtime。

该模块只编排已冻结的 typed 接口。结构、slot、Role、constraint、verifier、H-04
scorer、S-07 图协议和消费语义均由课程注入，宿主不读取词面或旧顺序兼容链猜规则。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pure_integer_ai.cognition.shared.hypothesis import (
    EPISTEMIC_SUPPORTED,
    EvidenceRecord,
    HypothesisKey,
)
from pure_integer_ai.cognition.shared.hypothesis_resolution import (
    ArchiveDirective,
    ReplacementDirective,
    ResolverDecision,
    TypedResolverScorer,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONCEPT,
    ObjectIdentity,
)
from pure_integer_ai.cognition.shared.order_hypothesis import (
    OrderAssessment,
    OrderEvidenceResult,
    OrderHypothesisEngine,
    OrderLearningProtocol,
    OrderPattern,
)
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.cognition.shared.structure_order import (
    StructureOrderGraph,
    StructureOrderGraphPredicates,
)
from pure_integer_ai.cognition.shared.structure_order_consumer import (
    PositionedStructureSlotValue,
    ResolvedStructureOrderConstraint,
    StructureOrderConsumer,
    StructureOrderConsumerProtocol,
    StructureOrderLinearizationResult,
    StructureOrderParseResult,
    StructureOrderSearchBudget,
    StructureSlotValue,
)
from pure_integer_ai.cognition.shared.structure_order_lifecycle import (
    StructureOrderLifecycleGraph,
    StructureOrderLifecycleProtocol,
)
from pure_integer_ai.cognition.understanding.order_constraint_promotion import (
    OrderConstraintPromoter,
    StructureOrderPromotionPlan,
    StructureOrderPromotionResult,
)
from pure_integer_ai.cognition.understanding.order_hypothesis_adapter import (
    MappedOrderObservation,
    OccurrenceOrderHypothesisAdapter,
    OccurrenceOrderStep,
    TypedOrderProjection,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int


PRECEDENCE_LIFECYCLE_NONE = 0
PRECEDENCE_LIFECYCLE_PROMOTED = 1
PRECEDENCE_LIFECYCLE_ACTIVE_REUSED = 2
PRECEDENCE_LIFECYCLE_DEMOTED = 3
PRECEDENCE_LIFECYCLE_INACTIVE = 4


def _strict_key(value: tuple[int, ...], *, where: str) -> tuple[int, ...]:
    """校验并返回非空严格整数协议键。"""
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{where} 必须是非空整数 tuple")
    assert_int(*value, _where=where)
    if any(type(item) is not int for item in value):
        raise TypeError(f"{where} 必须只含严格 int")
    return value


def _concepts(
        values: tuple[ObjectIdentity, ...], *, where: str,
        expected_count: int,
        ) -> tuple[ObjectIdentity, ...]:
    """校验固定 schema 槽数量，但不解释各一等 relation 的领域含义。"""
    if not isinstance(values, tuple) or len(values) != expected_count:
        raise ValueError(f"{where} 必须包含 {expected_count} 个对象")
    if any(not isinstance(item, ObjectIdentity) for item in values):
        raise TypeError(f"{where} 只能包含 ObjectIdentity")
    if any(item.object_kind != OBJECT_CONCEPT for item in values):
        raise ValueError(f"{where} 必须全部是一等 Concept")
    if len(set(values)) != len(values):
        raise ValueError(f"{where} 不得重复")
    return values


@dataclass(frozen=True)
class PrecedenceRelationProtocol:
    """定义可跨图重建的 H-06/S-07 开放协议和写入元数据。"""

    learning: OrderLearningProtocol
    order_predicates: tuple[ObjectIdentity, ...]
    lifecycle_predicates: tuple[ObjectIdentity, ...]
    lifecycle_states_and_kinds: tuple[ObjectIdentity, ...]
    lifecycle_event_namespace: tuple[int, ...]
    consumer: StructureOrderConsumerProtocol
    provenance_kind: int
    epistemic_origin: int = 0
    content_version: int = 0
    qualifiers: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.learning, OrderLearningProtocol):
            raise TypeError("learning 必须是 OrderLearningProtocol")
        order = _concepts(
            self.order_predicates,
            where="order_predicates",
            expected_count=19,
        )
        lifecycle = _concepts(
            self.lifecycle_predicates,
            where="lifecycle_predicates",
            expected_count=6,
        )
        states = _concepts(
            self.lifecycle_states_and_kinds,
            where="lifecycle_states_and_kinds",
            expected_count=6,
        )
        if len(set((*order, *lifecycle, *states))) != 31:
            raise ValueError("R-06 图 predicate、状态和事件 kind 必须互不相同")
        _strict_key(
            self.lifecycle_event_namespace,
            where="lifecycle_event_namespace",
        )
        if not isinstance(self.consumer, StructureOrderConsumerProtocol):
            raise TypeError("consumer 必须是 StructureOrderConsumerProtocol")
        assert_int(
            self.provenance_kind,
            self.epistemic_origin,
            self.content_version,
            *self.qualifiers,
            _where="PrecedenceRelationProtocol metadata",
        )
        if (type(self.provenance_kind) is not int
                or self.provenance_kind <= 0):
            raise ValueError("provenance_kind 必须为严格正整数")
        if type(self.epistemic_origin) is not int or self.epistemic_origin < 0:
            raise ValueError("epistemic_origin 必须为非负严格整数")
        if type(self.content_version) is not int or self.content_version < 0:
            raise ValueError("content_version 必须为非负严格整数")
        if not isinstance(self.qualifiers, tuple) or any(
                type(item) is not int for item in self.qualifiers):
            raise TypeError("qualifiers 必须是严格整数 tuple")

    def stable_key(self) -> tuple:
        """返回全部注入协议和写入元数据的可比较状态。"""
        return (
            self.learning.stable_key(),
            tuple(item.stable_key() for item in self.order_predicates),
            tuple(item.stable_key() for item in self.lifecycle_predicates),
            tuple(
                item.stable_key()
                for item in self.lifecycle_states_and_kinds
            ),
            self.lifecycle_event_namespace,
            tuple(
                reason.stable_key()
                for reason in (
                    self.consumer.invalid_assignment,
                    self.consumer.applicability_unknown,
                    self.consumer.missing_slot,
                    self.consumer.constraint_cycle,
                    self.consumer.constraint_conflict,
                    self.consumer.constraint_violation,
                    self.consumer.budget_exhausted,
                )
            ),
            self.provenance_kind,
            self.epistemic_origin,
            self.content_version,
            self.qualifiers,
        )


@dataclass(frozen=True)
class PrecedenceResolutionPlan:
    """课程为一次 H-04 解析显式提供的 scorer 和退出指令。"""

    scorers: tuple[TypedResolverScorer, ...] = ()
    replacements: tuple[ReplacementDirective, ...] = ()
    archives: tuple[ArchiveDirective, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.scorers, tuple):
            raise TypeError("scorers 必须是 tuple")
        if not isinstance(self.replacements, tuple) or any(
                not isinstance(item, ReplacementDirective)
                for item in self.replacements):
            raise TypeError("replacements 类型非法")
        if not isinstance(self.archives, tuple) or any(
                not isinstance(item, ArchiveDirective)
                for item in self.archives):
            raise TypeError("archives 类型非法")


@dataclass(frozen=True)
class PrecedenceConsumptionPlan:
    """课程为一次 parse 和 linearize 提供的 typed slot 输入。"""

    assignments: tuple[PositionedStructureSlotValue, ...]
    values: tuple[StructureSlotValue, ...]
    context: tuple[ObjectIdentity, ...]
    budget: StructureOrderSearchBudget

    def __post_init__(self) -> None:
        if not isinstance(self.assignments, tuple) or not self.assignments:
            raise ValueError("assignments 必须是非空 tuple")
        if any(not isinstance(item, PositionedStructureSlotValue)
               for item in self.assignments):
            raise TypeError("assignments 类型非法")
        if not isinstance(self.values, tuple) or not self.values:
            raise ValueError("values 必须是非空 tuple")
        if any(not isinstance(item, StructureSlotValue) for item in self.values):
            raise TypeError("values 类型非法")
        if not isinstance(self.context, tuple) or any(
                not isinstance(item, ObjectIdentity) for item in self.context):
            raise TypeError("context 类型非法")
        if not isinstance(self.budget, StructureOrderSearchBudget):
            raise TypeError("budget 必须是 StructureOrderSearchBudget")


@dataclass(frozen=True)
class PrecedenceEvidenceTrace:
    """保存一条 mapped observation 及其已提交 H-06 Evidence。"""

    mapped: MappedOrderObservation
    result: OrderEvidenceResult

    def __post_init__(self) -> None:
        if not isinstance(self.mapped, MappedOrderObservation):
            raise TypeError("mapped 必须是 MappedOrderObservation")
        if not isinstance(self.result, OrderEvidenceResult):
            raise TypeError("result 必须是 OrderEvidenceResult")


@runtime_checkable
class PrecedenceRelationCourse(Protocol):
    """课程注入结构映射、验证、解析策略、图定义和消费语义。"""

    def map_step(
            self, step: OccurrenceOrderStep,
            ) -> tuple[TypedOrderProjection, ...]:
        """把来源事实映射为零个或多个 typed slot 模式。"""
        ...

    def assess(
            self, pattern: OrderPattern,
            observation,
            ) -> OrderAssessment:
        """对单条真实 observation 返回 support/refute/unknown。"""
        ...

    def supersedes_evidence_id(
            self,
            mapped: MappedOrderObservation,
            prior: tuple[PrecedenceEvidenceTrace, ...],
            ) -> int:
        """显式选择本观察替代的旧 Evidence；无替代时返回零。"""
        ...

    def resolution_plan(
            self,
            mapped: MappedOrderObservation,
            evidence: OrderEvidenceResult,
            ) -> PrecedenceResolutionPlan:
        """返回本次 H-04 解析所需 scorer、replacement 和 archive。"""
        ...

    def promotion_plan(
            self,
            pattern: OrderPattern,
            hypothesis: HypothesisKey,
            ) -> StructureOrderPromotionPlan:
        """把 H-06 模式映射为完整 S-07 slot schema 和 constraint。"""
        ...

    def consumption_plan(
            self,
            mapped: MappedOrderObservation,
            ) -> PrecedenceConsumptionPlan:
        """提供不依赖 filler 词面的 parse 和 generation 输入。"""
        ...

    def resolve(
            self,
            definition,
            context: tuple[ObjectIdentity, ...],
            ) -> ResolvedStructureOrderConstraint:
        """解释图内 modality、condition、exception 和参数。"""
        ...

    def clone_for_evaluation(self) -> "PrecedenceRelationCourse":
        """返回不共享可变课程状态的评测副本。"""
        ...

    def state_key(self) -> tuple:
        """返回课程可比较状态供 V-06 零污染核验。"""
        ...


@dataclass(frozen=True)
class PrecedenceObservationReport:
    """一条 observation 的学习、生命周期和消费者联合报告。"""

    mapped: MappedOrderObservation
    read_only: bool
    duplicate: bool
    evidence: OrderEvidenceResult | None
    decision: ResolverDecision | None
    promotion: StructureOrderPromotionResult | None
    lifecycle_action: int
    parse: StructureOrderParseResult | None
    linearization: StructureOrderLinearizationResult | None

    def __post_init__(self) -> None:
        if not isinstance(self.mapped, MappedOrderObservation):
            raise TypeError("report.mapped 类型非法")
        if type(self.read_only) is not bool or type(self.duplicate) is not bool:
            raise TypeError("report read_only/duplicate 必须是 bool")
        assert_int(
            self.lifecycle_action,
            _where="PrecedenceObservationReport.lifecycle_action",
        )
        if self.lifecycle_action not in {
                PRECEDENCE_LIFECYCLE_NONE,
                PRECEDENCE_LIFECYCLE_PROMOTED,
                PRECEDENCE_LIFECYCLE_ACTIVE_REUSED,
                PRECEDENCE_LIFECYCLE_DEMOTED,
                PRECEDENCE_LIFECYCLE_INACTIVE}:
            raise ValueError("precedence lifecycle action 未注册")
        if self.read_only and any((
                self.evidence is not None,
                self.decision is not None,
                self.promotion is not None,
                self.lifecycle_action in {
                    PRECEDENCE_LIFECYCLE_PROMOTED,
                    PRECEDENCE_LIFECYCLE_DEMOTED,
                },
                )):
            raise ValueError("只读 precedence report 不得携带训练写入")


@dataclass(frozen=True)
class PrecedenceRoundReport:
    """一个 document scope 内全部 typed 顺序 observation 的报告。"""

    scope: ScopeIdentity
    read_only: bool
    observations: tuple[PrecedenceObservationReport, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("round report scope 必须是 ScopeIdentity")
        if type(self.read_only) is not bool:
            raise TypeError("round report read_only 必须是 bool")
        if not isinstance(self.observations, tuple) or any(
                not isinstance(item, PrecedenceObservationReport)
                for item in self.observations):
            raise TypeError("round report observations 类型非法")


class PrecedenceRelationRuntime:
    """显式驱动 occurrence fact、Evidence、约束生命周期和消费闭环。"""

    def __init__(
            self,
            ontology,
            occurrence_order_reader,
            protocol: PrecedenceRelationProtocol,
            course: PrecedenceRelationCourse,
            *,
            engine: OrderHypothesisEngine | None = None,
            next_timestamp_seq: int = 1,
            ) -> None:
        if not isinstance(protocol, PrecedenceRelationProtocol):
            raise TypeError("protocol 必须是 PrecedenceRelationProtocol")
        if not isinstance(course, PrecedenceRelationCourse):
            raise TypeError("course 必须实现 PrecedenceRelationCourse")
        assert_int(next_timestamp_seq, _where="next_timestamp_seq")
        if type(next_timestamp_seq) is not int or next_timestamp_seq <= 0:
            raise ValueError("next_timestamp_seq 必须为严格正整数")
        self.ontology = ontology
        self.protocol = protocol
        self.course = course
        self.adapter = OccurrenceOrderHypothesisAdapter(
            occurrence_order_reader)
        self.engine = (
            OrderHypothesisEngine(protocol.learning)
            if engine is None else engine
        )
        if (not isinstance(self.engine, OrderHypothesisEngine)
                or self.engine.protocol != protocol.learning):
            raise ValueError("engine 必须绑定同一个 OrderLearningProtocol")
        self.order_graph, self.lifecycle = self._build_graphs()
        self.promoter = OrderConstraintPromoter(
            self.engine, self.order_graph, self.lifecycle)
        self.consumer = StructureOrderConsumer(
            self.lifecycle, course, protocol.consumer)
        self._next_timestamp_seq = next_timestamp_seq
        self._traces: list[PrecedenceEvidenceTrace] = []
        self._trace_by_observation: dict[
            tuple[int, ...], PrecedenceEvidenceTrace] = {}

    def process(
            self, scope: ScopeIdentity, *, read_only: bool,
            ) -> PrecedenceRoundReport:
        """处理一个来源 scope；训练写 Evidence，评测只消费克隆状态。"""
        if not isinstance(scope, ScopeIdentity):
            raise TypeError("precedence scope 必须是 ScopeIdentity")
        if type(read_only) is not bool:
            raise TypeError("read_only 必须是 bool")
        mapped = tuple(sorted(
            self.adapter.project(scope, self.course.map_step),
            key=lambda item: (
                item.pattern.stable_key(),
                item.observation.stable_key(),
            ),
        ))
        reports = tuple(
            self._process_observation(item, read_only=read_only)
            for item in mapped
        )
        return PrecedenceRoundReport(scope, read_only, reports)

    def clone_for_context(self, ctx) -> "PrecedenceRelationRuntime":
        """复制 H-06 和课程状态，并在评测克隆图上重建 S-07 facade。"""
        if ctx.occurrence_order_reader is None:
            raise ValueError("评测 clone 缺少 occurrence order reader")
        cloned_course = self.course.clone_for_evaluation()
        if not isinstance(cloned_course, PrecedenceRelationCourse):
            raise TypeError("course clone 未实现 PrecedenceRelationCourse")
        cloned = PrecedenceRelationRuntime(
            ctx.graph_ontology,
            ctx.occurrence_order_reader,
            self.protocol,
            cloned_course,
            engine=self.engine.clone(),
            next_timestamp_seq=self._next_timestamp_seq,
        )
        cloned._traces = list(self._traces)
        cloned._trace_by_observation = dict(self._trace_by_observation)
        return cloned

    def evidence_count(self) -> int:
        """返回 runtime 已提交且包含历史替代事件的 Evidence 数。"""
        return len(self._traces)

    def state_key(self) -> tuple:
        """返回 H-06、课程、时钟和 observation 路由的完整可比较状态。"""
        return (
            self.protocol.stable_key(),
            self.engine.state_key(),
            self.course.state_key(),
            self._next_timestamp_seq,
            tuple(
                (
                    trace.mapped.pattern.stable_key(),
                    trace.mapped.observation.stable_key(),
                    trace.result.evidence.stable_key(),
                )
                for trace in self._traces
            ),
        )

    def _build_graphs(
            self,
            ) -> tuple[StructureOrderGraph, StructureOrderLifecycleGraph]:
        """从一等身份在当前 ontology 上重建可 dump/load 的两层 facade。"""
        order_refs = tuple(
            self.ontology.materialize(identity)
            for identity in self.protocol.order_predicates
        )
        lifecycle_refs = tuple(
            self.ontology.materialize(identity)
            for identity in self.protocol.lifecycle_predicates
        )
        for identity in self.protocol.lifecycle_states_and_kinds:
            self.ontology.materialize(identity)
        order_graph = StructureOrderGraph(
            self.ontology,
            StructureOrderGraphPredicates(*order_refs),
        )
        lifecycle_protocol = StructureOrderLifecycleProtocol(
            *lifecycle_refs,
            *self.protocol.lifecycle_states_and_kinds,
            self.protocol.lifecycle_event_namespace,
        )
        return (
            order_graph,
            StructureOrderLifecycleGraph(order_graph, lifecycle_protocol),
        )

    def _process_observation(
            self, mapped: MappedOrderObservation, *, read_only: bool,
            ) -> PrecedenceObservationReport:
        """对单条映射执行显式学习或只读 held-out 消费。"""
        pattern = mapped.pattern
        hypothesis = self.engine.hypothesis_for(pattern)
        if read_only:
            plan, consumption = self._prepare_course(mapped, hypothesis)
            return self._consume_report(
                mapped,
                plan=plan,
                consumption=consumption,
                evidence=None,
                decision=None,
                promotion=None,
                lifecycle_action=PRECEDENCE_LIFECYCLE_NONE,
                duplicate=False,
                read_only=True,
            )

        prior = tuple(
            trace for trace in self._traces
            if trace.result.hypothesis == hypothesis
        )
        supersedes = self.course.supersedes_evidence_id(mapped, prior)
        assert_int(supersedes, _where="supersedes_evidence_id")
        if type(supersedes) is not int or supersedes < 0:
            raise ValueError("supersedes_evidence_id 必须为非负严格整数")
        observation_key = self._observation_key(mapped)
        existing = self._trace_by_observation.get(observation_key)
        if existing is not None and supersedes == 0:
            plan, consumption = self._prepare_course(mapped, hypothesis)
            return self._consume_report(
                mapped,
                plan=plan,
                consumption=consumption,
                evidence=existing.result,
                decision=None,
                promotion=None,
                lifecycle_action=PRECEDENCE_LIFECYCLE_NONE,
                duplicate=True,
                read_only=False,
            )

        plan, consumption = self._prepare_course(mapped, hypothesis)
        assessment = self.course.assess(pattern, mapped.observation)
        if not isinstance(assessment, OrderAssessment):
            raise TypeError("course.assess 返回类型错误")

        def cached_verifier(candidate, observation):
            """只向 H-06 返回本次已预检的单次 assessment。"""
            if candidate != pattern or observation != mapped.observation:
                raise ValueError("cached assessment 收到不同 observation")
            return assessment

        evidence = self.engine.accumulate(
            pattern,
            mapped.observation,
            cached_verifier,
            timestamp_seq=self._advance(),
            supersedes_evidence_id=supersedes,
        )
        trace = PrecedenceEvidenceTrace(mapped, evidence)
        self._traces.append(trace)
        self._trace_by_observation[observation_key] = trace
        resolution = self.course.resolution_plan(mapped, evidence)
        if not isinstance(resolution, PrecedenceResolutionPlan):
            raise TypeError("course.resolution_plan 返回类型错误")
        decision = self.engine.resolve(
            pattern,
            timestamp_seq=self._advance(),
            scorers=resolution.scorers,
            replacements=resolution.replacements,
            archives=resolution.archives,
        )
        promotion, lifecycle_action = self._synchronize_lifecycle(
            plan, decision)
        return self._consume_report(
            mapped,
            plan=plan,
            consumption=consumption,
            evidence=evidence,
            decision=decision,
            promotion=promotion,
            lifecycle_action=lifecycle_action,
            duplicate=False,
            read_only=False,
        )

    def _synchronize_lifecycle(
            self,
            plan: StructureOrderPromotionPlan,
            decision: ResolverDecision,
            ) -> tuple[
                StructureOrderPromotionResult | None,
                int,
                ]:
        """按当前 H-00/H-04 状态显式晋升、复用或降级 S-07 约束。"""
        hypothesis = plan.constraint.hypothesis
        snapshot = self.engine.ledger.snapshot(hypothesis)
        constraint_ref = self.ontology.resolve(plan.constraint.constraint)
        active = False
        if constraint_ref is not None:
            active = (
                self.lifecycle.project(constraint_ref).state
                == self.lifecycle.protocol.active_state
            )
        adopted = hypothesis in decision.adopted_hypotheses
        if snapshot.epistemic_status == EPISTEMIC_SUPPORTED and adopted:
            promotion = self.promoter.promote(
                plan,
                decision,
                timestamp_seq=self._advance(),
                provenance_kind=self.protocol.provenance_kind,
                epistemic_origin=self.protocol.epistemic_origin,
                content_version=self.protocol.content_version,
                qualifiers=self.protocol.qualifiers,
            )
            return (
                promotion,
                PRECEDENCE_LIFECYCLE_ACTIVE_REUSED
                if active else PRECEDENCE_LIFECYCLE_PROMOTED,
            )
        if active:
            self.promoter.demote(
                plan.constraint.constraint,
                decision,
                timestamp_seq=self._advance(),
                provenance_kind=self.protocol.provenance_kind,
                epistemic_origin=self.protocol.epistemic_origin,
                content_version=self.protocol.content_version,
                qualifiers=self.protocol.qualifiers,
            )
            return None, PRECEDENCE_LIFECYCLE_DEMOTED
        return None, PRECEDENCE_LIFECYCLE_INACTIVE

    def _consume_report(
            self,
            mapped: MappedOrderObservation,
            *,
            plan: StructureOrderPromotionPlan,
            consumption: PrecedenceConsumptionPlan,
            evidence: OrderEvidenceResult | None,
            decision: ResolverDecision | None,
            promotion: StructureOrderPromotionResult | None,
            lifecycle_action: int,
            duplicate: bool,
            read_only: bool,
            ) -> PrecedenceObservationReport:
        """只在目标 constraint 当前 active 时执行 parse 和 linearize。"""
        constraint_ref = self.ontology.resolve(plan.constraint.constraint)
        if constraint_ref is None:
            return PrecedenceObservationReport(
                mapped,
                read_only,
                duplicate,
                evidence,
                decision,
                promotion,
                lifecycle_action,
                None,
                None,
            )
        projection = self.lifecycle.project(constraint_ref)
        if projection.state != self.lifecycle.protocol.active_state:
            return PrecedenceObservationReport(
                mapped,
                read_only,
                duplicate,
                evidence,
                decision,
                promotion,
                lifecycle_action,
                None,
                None,
            )
        structure_ref = self.ontology.resolve(mapped.pattern.structure_candidate)
        if structure_ref is None:
            raise RuntimeError("active constraint 缺少 StructureConcept")
        parsed = self.consumer.parse(
            structure_ref,
            consumption.assignments,
            context=consumption.context,
            budget=consumption.budget,
        )
        linearized = self.consumer.linearize(
            structure_ref,
            consumption.values,
            context=consumption.context,
            budget=consumption.budget,
        )
        return PrecedenceObservationReport(
            mapped,
            read_only,
            duplicate,
            evidence,
            decision,
            promotion,
            lifecycle_action,
            parsed,
            linearized,
        )

    def _prepare_course(
            self,
            mapped: MappedOrderObservation,
            hypothesis: HypothesisKey,
            ) -> tuple[
                StructureOrderPromotionPlan,
                PrecedenceConsumptionPlan,
                ]:
        """在写 Evidence 前纯校验课程的图 schema、消费输入和 resolver 输出。"""
        plan = self.course.promotion_plan(mapped.pattern, hypothesis)
        if not isinstance(plan, StructureOrderPromotionPlan):
            raise TypeError("course.promotion_plan 返回类型错误")
        if plan.constraint.hypothesis != hypothesis:
            raise ValueError("promotion plan 替换了 H-06 Hypothesis")
        consumption = self.course.consumption_plan(mapped)
        if not isinstance(consumption, PrecedenceConsumptionPlan):
            raise TypeError("course.consumption_plan 返回类型错误")
        resolved = self.course.resolve(
            plan.constraint,
            consumption.context,
        )
        if not isinstance(resolved, ResolvedStructureOrderConstraint):
            raise TypeError("course.resolve 返回类型错误")
        if resolved.constraint != plan.constraint.constraint:
            raise ValueError("course.resolve 替换了 constraint 身份")
        if {resolved.before_slot, resolved.after_slot} != {
                plan.constraint.first_slot, plan.constraint.second_slot}:
            raise ValueError("course.resolve 替换了 constraint slot pair")
        return plan, consumption

    def _advance(self) -> int:
        """分配严格递增的运行内逻辑序，不读取墙钟。"""
        current = self._next_timestamp_seq
        self._next_timestamp_seq += 1
        return current

    @staticmethod
    def _observation_key(
            mapped: MappedOrderObservation,
            ) -> tuple[int, ...]:
        """组合模式和来源 observation 完整键用于幂等重放。"""
        pattern = mapped.pattern.stable_key()
        observation = mapped.observation.stable_key()
        return (
            len(pattern),
            *pattern,
            len(observation),
            *observation,
        )


def install_precedence_relation_runtime(
        ctx,
        protocol: PrecedenceRelationProtocol,
        course: PrecedenceRelationCourse,
        ) -> PrecedenceRelationRuntime:
    """在正式 TrainContext 上安装 R-06 runtime，并拒绝缺失 L-06 reader。"""
    if ctx.occurrence_order_reader is None:
        raise ValueError("R-06 runtime 必须先安装 L-06 occurrence order reader")
    runtime = PrecedenceRelationRuntime(
        ctx.graph_ontology,
        ctx.occurrence_order_reader,
        protocol,
        course,
    )
    ctx.precedence_relation_runtime = runtime
    return runtime


__all__ = [
    "PRECEDENCE_LIFECYCLE_ACTIVE_REUSED",
    "PRECEDENCE_LIFECYCLE_DEMOTED",
    "PRECEDENCE_LIFECYCLE_INACTIVE",
    "PRECEDENCE_LIFECYCLE_NONE",
    "PRECEDENCE_LIFECYCLE_PROMOTED",
    "PrecedenceConsumptionPlan",
    "PrecedenceEvidenceTrace",
    "PrecedenceObservationReport",
    "PrecedenceRelationCourse",
    "PrecedenceRelationProtocol",
    "PrecedenceRelationRuntime",
    "PrecedenceResolutionPlan",
    "PrecedenceRoundReport",
    "install_precedence_relation_runtime",
]
