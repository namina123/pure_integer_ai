"""S-07 一等结构顺序约束、生命周期和 typed 消费者对抗测试。"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from pure_integer_ai.cognition.shared.hypothesis import (
    EPISTEMIC_CONFLICTED,
    EPISTEMIC_SUPPORTED,
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EVIDENCE_UNKNOWN,
)
from pure_integer_ai.cognition.shared.hypothesis_resolution import (
    ReplacementDirective,
)
from pure_integer_ai.cognition.shared.generation_structure_execution import (
    GenerationStructureExecutionPlan,
    GenerationStructureExecutionPlanner,
    GenerationStructureExecutionRequest,
    SentenceStructureExecutionBudget,
)
from pure_integer_ai.cognition.shared.generation_structure_plan import (
    PlannedSentence,
    PropositionSlotFiller,
    SyntaxLinearizationObligation,
    SyntaxPlan,
)
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    SourceRef,
    VersionBundle,
    concept_identity,
    language_branch_identity,
    minimal_instruction_identity,
    occurrence_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.order_hypothesis import (
    OrderAssessment,
    OrderHypothesisEngine,
    OrderLearningProtocol,
    OrderObservation,
    OrderPattern,
)
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.cognition.shared.semantic_object import (
    proposition_identity,
    role_identity,
)
from pure_integer_ai.cognition.shared.structure_order import (
    StructureOrderConstraintDefinition,
    StructureOrderGraph,
    StructureOrderGraphPredicates,
    StructureOrderParameterDefinition,
    StructureOrderTopologyError,
    StructureSlotDefinition,
)
from pure_integer_ai.cognition.shared.structure_order_consumer import (
    ORDER_APPLICABILITY_UNKNOWN,
    ORDER_APPLICABLE,
    ORDER_CONSUMER_ACCEPTED,
    ORDER_CONSUMER_REJECTED,
    ORDER_CONSUMER_UNKNOWN,
    ORDER_NOT_APPLICABLE,
    PositionedStructureSlotValue,
    ResolvedStructureOrderConstraint,
    StructureOrderConsumer,
    StructureOrderConsumerProtocol,
    StructureOrderSearchBudget,
    StructureSlotValue,
)
from pure_integer_ai.cognition.shared.structure_order_lifecycle import (
    StructureOrderLifecycleError,
    StructureOrderLifecycleGraph,
    StructureOrderLifecycleProtocol,
)
from pure_integer_ai.cognition.understanding.order_constraint_promotion import (
    OrderConstraintPromoter,
    OrderConstraintPromotionError,
    StructureOrderPromotionPlan,
)
from pure_integer_ai.cognition.shared.typed_binding import BoundProposition
from pure_integer_ai.experiments.formal_train import make_train_context
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.training.cursor import DUMP_TABLES, dump_run, load_run


_BASE = 27100


def _source(document_id: int, *, source_id: int = _BASE + 1) -> SourceRef:
    """构造 H-06 aggregate 或真实 observation SourceRef。"""
    return SourceRef(
        _BASE + 2,
        source_id,
        document_id,
        GLOBAL_OWNER_SCOPE,
        VersionBundle(),
    )


def _learning_protocol(*, source_id: int = _BASE + 3) -> OrderLearningProtocol:
    """构造与具体 observation 分离的版本化顺序学习协议。"""
    source = _source(0, source_id=source_id)
    return OrderLearningProtocol(
        (_BASE + 4, 1),
        (_BASE + 5, 1),
        (_BASE + 5, 2),
        (_BASE + 5, 3),
        (_BASE + 6, 1),
        source,
        document_scope(source),
    )


@dataclass(frozen=True)
class _Domain:
    """测试课程注入的一等结构、slot、上下文和 modality 身份。"""

    language: object
    order_kind: object
    family: object
    structure: object
    slots: tuple
    context: object
    condition: object
    exception: object
    required: object
    preferred: object
    optional: object
    value_type: object


def _domain(*, variant: int = 1, slot_count: int = 3) -> _Domain:
    """为一个结构建立彼此独立且不含词面的图身份。"""
    return _Domain(
        language_branch_identity((_BASE + 10, variant)),
        concept_identity((_BASE + 11, variant)),
        structure_concept_identity((_BASE + 12, variant)),
        structure_concept_identity((_BASE + 13, variant)),
        tuple(
            structure_concept_identity((_BASE + 14, variant, index))
            for index in range(slot_count)
        ),
        concept_identity((_BASE + 15, variant)),
        concept_identity((_BASE + 16, variant)),
        concept_identity((_BASE + 17, variant)),
        concept_identity((_BASE + 18, variant, 1)),
        concept_identity((_BASE + 18, variant, 2)),
        concept_identity((_BASE + 18, variant, 3)),
        concept_identity((_BASE + 19, variant)),
    )


def _pattern(
        domain: _Domain, *, first: int = 0, second: int = 1,
        kind: int = 1, conditional: bool = False) -> OrderPattern:
    """构造 slot pair 规范化但 constraint kind 可替换的 H-06 模式。"""
    left, right = sorted(
        (domain.slots[first], domain.slots[second]),
        key=lambda item: item.stable_key(),
    )
    return OrderPattern(
        domain.language,
        domain.order_kind,
        domain.family,
        domain.structure,
        left,
        right,
        concept_identity((_BASE + 20, kind)),
        domain.context,
        (domain.condition,) if conditional else (),
    )


def _observation(
        pattern: OrderPattern, event: int, *, reverse: bool = False,
        ) -> OrderObservation:
    """用来源化 occurrence 构造与词汇身份无关的 typed 顺序观察。"""
    source = _source(event, source_id=_BASE + 30)
    first_position, second_position = ((1, 0) if reverse else (0, 1))
    return OrderObservation(
        source,
        document_scope(source),
        (_BASE + 31, event),
        pattern.language_branch,
        pattern.structure_family,
        pattern.structure_candidate,
        pattern.first_slot,
        pattern.second_slot,
        pattern.context,
        pattern.conditions,
        occurrence_identity(source, start=event * 2, end=event * 2 + 1,
                            ordinal=0),
        occurrence_identity(source, start=event * 2 + 1,
                            end=event * 2 + 2, ordinal=0),
        first_position,
        second_position,
        (first_position, second_position),
    )


def _assessment(stance: int, detail: int):
    """返回由测试课程显式控制的三态顺序 verifier。"""
    return lambda _pattern, _observation: OrderAssessment(
        stance, (_BASE + 32, detail))


def _support(
        engine: OrderHypothesisEngine, pattern: OrderPattern, event: int,
        ):
    """追加一条支持 Evidence，并提交同竞争组 H-04 决策。"""
    result = engine.accumulate(
        pattern,
        _observation(pattern, event),
        _assessment(EVIDENCE_SUPPORT, event),
        timestamp_seq=event,
    )
    decision = engine.resolve(pattern, timestamp_seq=event + 1)
    return result, decision


@dataclass(frozen=True)
class _Graphs:
    """同一 GraphOntology 上的结构图、生命周期和协议身份。"""

    context: object
    order_graph: StructureOrderGraph
    lifecycle: StructureOrderLifecycleGraph
    order_predicate_identities: tuple
    lifecycle_predicate_identities: tuple
    states_and_kinds: tuple


def _graphs(backend) -> _Graphs:
    """物化开放 predicate/state/kind 后装配 S-07 两层图 facade。"""
    context = make_train_context(backend)
    ontology = context.graph_ontology
    predicate_identities = tuple(
        concept_identity((_BASE + 40, index)) for index in range(25))
    refs = tuple(ontology.materialize(item) for item in predicate_identities)
    order_predicates = StructureOrderGraphPredicates(*refs[:19])
    order_graph = StructureOrderGraph(ontology, order_predicates)
    states_and_kinds = tuple(
        concept_identity((_BASE + 41, index)) for index in range(6))
    for identity in states_and_kinds:
        ontology.materialize(identity)
    lifecycle_protocol = StructureOrderLifecycleProtocol(
        *refs[19:],
        *states_and_kinds,
        (_BASE + 42, 1),
    )
    lifecycle = StructureOrderLifecycleGraph(
        order_graph, lifecycle_protocol)
    return _Graphs(
        context,
        order_graph,
        lifecycle,
        predicate_identities[:19],
        predicate_identities[19:],
        states_and_kinds,
    )


def _slot_definitions(domain: _Domain) -> tuple[StructureSlotDefinition, ...]:
    """为每个 slot 注入独立 Role，并共享一个显式 value type。"""
    return tuple(
        StructureSlotDefinition(
            domain.structure,
            slot,
            role_identity((_BASE + 50, index)),
            domain.value_type,
        )
        for index, slot in enumerate(domain.slots)
    )


def _plan(
        engine: OrderHypothesisEngine, pattern: OrderPattern,
        domain: _Domain, *, instance: int,
        modality=None, exceptions: tuple = (),
        parameter: bool = False,
        ) -> StructureOrderPromotionPlan:
    """把 H-06 模式映射为显式 S-07 slot schema 和约束实例。"""
    constraint = structure_concept_identity((_BASE + 60, instance))
    parameters = ()
    if parameter:
        parameters = (StructureOrderParameterDefinition(
            structure_concept_identity((_BASE + 61, instance)),
            role_identity((_BASE + 62, instance)),
            concept_identity((_BASE + 63, instance)),
        ),)
    definition = StructureOrderConstraintDefinition(
        constraint,
        pattern.language_branch,
        pattern.structure_family,
        pattern.structure_candidate,
        pattern.first_slot,
        pattern.second_slot,
        pattern.order_kind,
        pattern.constraint,
        domain.required if modality is None else modality,
        pattern.context,
        tuple(sorted(pattern.conditions, key=lambda item: item.stable_key())),
        tuple(sorted(exceptions, key=lambda item: item.stable_key())),
        parameters,
        engine.hypothesis_for(pattern),
    )
    return StructureOrderPromotionPlan(_slot_definitions(domain), definition)


def _promote(
        promoter: OrderConstraintPromoter,
        plan: StructureOrderPromotionPlan,
        decision,
        timestamp: int,
        ):
    """用统一开放元数据执行一次测试晋升。"""
    return promoter.promote(
        plan,
        decision,
        timestamp_seq=timestamp,
        provenance_kind=_BASE + 70,
        qualifiers=(_BASE + 71,),
    )


def test_structure_constraint_round_trip_survives_cache_clear():
    """结构成员、Role、参数、Hypothesis、scope 和 Event 可只从图恢复。"""
    backend = DictBackend()
    try:
        graphs = _graphs(backend)
        engine = OrderHypothesisEngine(_learning_protocol())
        domain = _domain()
        pattern = _pattern(domain, conditional=True)
        support, decision = _support(engine, pattern, 1)
        plan = _plan(
            engine,
            pattern,
            domain,
            instance=1,
            exceptions=(domain.exception,),
            parameter=True,
        )
        promoter = OrderConstraintPromoter(
            engine, graphs.order_graph, graphs.lifecycle)

        written = _promote(promoter, plan, decision, 3)
        structure_ref = graphs.context.graph_ontology.resolve(domain.structure)
        restored = graphs.order_graph.read_structure(structure_ref)

        assert tuple(item.definition for item in restored.slots) == (
            _slot_definitions(domain))
        assert len(restored.constraints) == 1
        assert restored.constraints[0].definition == plan.constraint
        assert written.lifecycle.definition.evidence_keys == (
            support.evidence.stable_key(),)
        assert written.lifecycle.definition.hypothesis.observation == (
            _learning_protocol().aggregate_source)
        assert graphs.lifecycle.project(
            written.constraint.constraint).state == (
                graphs.lifecycle.protocol.active_state)

        graphs.context.graph_ontology.clear_runtime_caches()
        structure_ref = graphs.context.graph_ontology.resolve(domain.structure)
        after_clear = graphs.order_graph.read_structure(structure_ref)
        projection = graphs.lifecycle.project(
            graphs.context.graph_ontology.resolve(plan.constraint.constraint))
        assert tuple(item.definition for item in after_clear.constraints) == (
            plan.constraint,)
        assert projection.history[0].definition == written.lifecycle.definition
    finally:
        backend.close()


def test_partial_structure_topology_fails_without_repair():
    """已有成员边但缺 language/family 时，define 不得静默补成完整结构。"""
    backend = DictBackend()
    try:
        graphs = _graphs(backend)
        engine = OrderHypothesisEngine(_learning_protocol())
        domain = _domain()
        pattern = _pattern(domain)
        _support(engine, pattern, 1)
        plan = _plan(engine, pattern, domain, instance=2)
        ontology = graphs.context.graph_ontology
        structure = ontology.materialize(domain.structure)
        slot = ontology.materialize(domain.slots[0])
        ontology.relate(
            graphs.order_graph.predicates.structure_slot,
            structure,
            slot,
            scope=plan.constraint.scope,
            provenance_kind=_BASE + 70,
        )
        before = backend.snapshot()

        with pytest.raises(StructureOrderTopologyError, match="缺少"):
            graphs.order_graph.define_constraint(
                plan.slots,
                plan.constraint,
                scope=plan.constraint.scope,
                provenance_kind=_BASE + 70,
            )

        assert backend.snapshot() == before
    finally:
        backend.close()


def test_constraint_cannot_be_attached_to_a_second_structure():
    """额外成员边不得使一个 constraint 在第二个 StructureConcept 中伪装有效。"""
    backend = DictBackend()
    try:
        graphs = _graphs(backend)
        engine = OrderHypothesisEngine(_learning_protocol())
        domain = _domain()
        pattern = _pattern(domain)
        _, decision = _support(engine, pattern, 1)
        plan = _plan(engine, pattern, domain, instance=21)
        promoter = OrderConstraintPromoter(
            engine, graphs.order_graph, graphs.lifecycle)
        written = _promote(promoter, plan, decision, 3)
        ontology = graphs.context.graph_ontology
        foreign = structure_concept_identity((_BASE + 64, 1))
        foreign_ref = ontology.materialize(foreign)
        ontology.relate(
            graphs.order_graph.predicates.structure_language,
            foreign_ref,
            ontology.resolve(domain.language),
            scope=plan.constraint.scope,
            provenance_kind=_BASE + 70,
        )
        ontology.relate(
            graphs.order_graph.predicates.structure_family,
            foreign_ref,
            ontology.resolve(domain.family),
            scope=plan.constraint.scope,
            provenance_kind=_BASE + 70,
        )
        ontology.relate(
            graphs.order_graph.predicates.structure_constraint,
            foreign_ref,
            written.constraint.constraint,
            scope=plan.constraint.scope,
            provenance_kind=_BASE + 70,
        )

        with pytest.raises(StructureOrderTopologyError, match="多个 structure"):
            graphs.order_graph.read_constraint(written.constraint.constraint)
        with pytest.raises(StructureOrderTopologyError, match="constraint"):
            graphs.order_graph.read_structure(foreign_ref)
    finally:
        backend.close()


def test_promotion_requires_supported_and_committed_current_decision():
    """unknown、未提交决策和被后续 Evidence 陈旧化的决策都不能晋升。"""
    backend = DictBackend()
    try:
        graphs = _graphs(backend)
        engine = OrderHypothesisEngine(_learning_protocol())
        domain = _domain()
        pattern = _pattern(domain)
        engine.register_pattern(pattern)
        plan = _plan(engine, pattern, domain, instance=3)
        promoter = OrderConstraintPromoter(
            engine, graphs.order_graph, graphs.lifecycle)
        unknown_decision = engine.resolve(pattern, timestamp_seq=1)
        with pytest.raises(OrderConstraintPromotionError, match="supported"):
            _promote(promoter, plan, unknown_decision, 2)

        support = engine.accumulate(
            pattern,
            _observation(pattern, 2),
            _assessment(EVIDENCE_SUPPORT, 2),
            timestamp_seq=2,
        )
        simulated = engine.resolve(pattern, timestamp_seq=3, commit=False)
        with pytest.raises(OrderConstraintPromotionError, match="未提交"):
            _promote(promoter, plan, simulated, 4)

        committed = engine.resolve(pattern, timestamp_seq=3)
        engine.accumulate(
            pattern,
            _observation(pattern, 3),
            _assessment(EVIDENCE_REFUTE, 3),
            timestamp_seq=4,
        )
        assert engine.ledger.snapshot(
            support.hypothesis).epistemic_status == EPISTEMIC_CONFLICTED
        with pytest.raises(OrderConstraintPromotionError, match="supported"):
            _promote(promoter, plan, committed, 5)
    finally:
        backend.close()


def test_demotion_and_repromotion_append_state_history():
    """反例使约束降级，替代反例后可再次晋升且旧 Event 全部保留。"""
    backend = DictBackend()
    try:
        graphs = _graphs(backend)
        engine = OrderHypothesisEngine(_learning_protocol())
        domain = _domain()
        pattern = _pattern(domain)
        _, first_decision = _support(engine, pattern, 1)
        plan = _plan(engine, pattern, domain, instance=4)
        promoter = OrderConstraintPromoter(
            engine, graphs.order_graph, graphs.lifecycle)
        written = _promote(promoter, plan, first_decision, 3)

        counter_observation = _observation(pattern, 2, reverse=True)
        counter = engine.accumulate(
            pattern,
            counter_observation,
            _assessment(EVIDENCE_REFUTE, 2),
            timestamp_seq=4,
        )
        demotion_decision = engine.resolve(pattern, timestamp_seq=5)
        promoter.demote(
            plan.constraint.constraint,
            demotion_decision,
            timestamp_seq=6,
            provenance_kind=_BASE + 70,
            qualifiers=(_BASE + 71,),
        )
        assert graphs.lifecycle.project(
            written.constraint.constraint).state == (
                graphs.lifecycle.protocol.inactive_state)

        engine.accumulate(
            pattern,
            counter_observation,
            _assessment(EVIDENCE_UNKNOWN, 3),
            timestamp_seq=7,
            supersedes_evidence_id=counter.evidence.evidence_id,
        )
        assert engine.ledger.snapshot(
            plan.constraint.hypothesis).epistemic_status == EPISTEMIC_SUPPORTED
        repromotion_decision = engine.resolve(pattern, timestamp_seq=8)
        _promote(promoter, plan, repromotion_decision, 9)
        projection = graphs.lifecycle.project(written.constraint.constraint)

        assert projection.state == graphs.lifecycle.protocol.active_state
        assert tuple(
            item.definition.to_state for item in projection.history) == (
                graphs.lifecycle.protocol.active_state,
                graphs.lifecycle.protocol.inactive_state,
                graphs.lifecycle.protocol.active_state,
            )
    finally:
        backend.close()


def test_supersede_requires_h00_replacement_and_changes_active_projection():
    """只有 H-00 同竞争组 replacement 能终止旧 constraint，图历史仍可回读。"""
    backend = DictBackend()
    try:
        graphs = _graphs(backend)
        engine = OrderHypothesisEngine(_learning_protocol())
        domain = _domain()
        old_pattern = _pattern(domain, kind=1)
        new_pattern = _pattern(domain, kind=2)
        old_support, old_decision = _support(engine, old_pattern, 1)
        old_plan = _plan(engine, old_pattern, domain, instance=5)
        new_support = engine.accumulate(
            new_pattern,
            _observation(new_pattern, 2),
            _assessment(EVIDENCE_SUPPORT, 2),
            timestamp_seq=3,
        )
        joint_decision = engine.resolve(new_pattern, timestamp_seq=4)
        promoter = OrderConstraintPromoter(
            engine, graphs.order_graph, graphs.lifecycle)
        _promote(promoter, old_plan, old_decision, 2)
        new_plan = _plan(engine, new_pattern, domain, instance=6)
        _promote(promoter, new_plan, joint_decision, 5)

        with pytest.raises(OrderConstraintPromotionError, match="尚未 superseded"):
            promoter.supersede(
                old_plan.constraint.constraint,
                new_plan.constraint.constraint,
                joint_decision,
                timestamp_seq=6,
                provenance_kind=_BASE + 70,
            )

        refute = engine.accumulate(
            old_pattern,
            old_support.observation,
            _assessment(EVIDENCE_REFUTE, 4),
            timestamp_seq=6,
            supersedes_evidence_id=old_support.evidence.evidence_id,
        )
        replacement_decision = engine.resolve(
            old_pattern,
            timestamp_seq=7,
            replacements=(ReplacementDirective(
                old_support.hypothesis,
                new_support.hypothesis,
                refute.evidence.evidence_id,
            ),),
        )
        event = promoter.supersede(
            old_plan.constraint.constraint,
            new_plan.constraint.constraint,
            replacement_decision,
            timestamp_seq=8,
            provenance_kind=_BASE + 70,
        )
        structure = graphs.context.graph_ontology.resolve(domain.structure)
        active = graphs.lifecycle.active_constraints(structure)

        assert event.definition.replacement == new_plan.constraint.constraint
        assert tuple(item.constraint.definition.constraint for item in active) == (
            new_plan.constraint.constraint,)
        old_projection = graphs.lifecycle.project(
            graphs.context.graph_ontology.resolve(
                old_plan.constraint.constraint))
        assert old_projection.state == graphs.lifecycle.protocol.superseded_state
        with pytest.raises(StructureOrderLifecycleError, match="from_state"):
            graphs.lifecycle.append(
                graphs.lifecycle.make_event(
                    old_plan.constraint.constraint,
                    event_kind=graphs.lifecycle.protocol.promotion_kind,
                    from_state=graphs.lifecycle.protocol.inactive_state,
                    to_state=graphs.lifecycle.protocol.active_state,
                    hypothesis=old_plan.constraint.hypothesis,
                    evidence_keys=event.definition.evidence_keys,
                    decision_key=replacement_decision.stable_key(),
                    timestamp_seq=9,
                ),
                scope=old_plan.constraint.scope,
                provenance_kind=_BASE + 70,
            )
    finally:
        backend.close()


@dataclass(frozen=True)
class _ResolvedRule:
    """测试 resolver 的注入控制，不依赖 constraint 名称或键值解析。"""

    before: object
    after: object
    enforced: bool
    allow_missing: bool
    weight: int
    minimum_gap: int
    maximum_gap: int | None


class _SemanticsResolver:
    """按测试课程显式注册表解释 modality、条件和例外。"""

    def __init__(self, rules: dict, *, applies_reason, skipped_reason,
                 unknown_reason) -> None:
        self.rules = rules
        self.applies_reason = applies_reason
        self.skipped_reason = skipped_reason
        self.unknown_reason = unknown_reason

    def resolve(self, definition, context):
        """显式 context 缺 condition 时 unknown，命中 exception 时跳过。"""
        rule = self.rules[definition.constraint]
        context_set = frozenset(context)
        if any(item in context_set for item in definition.exceptions):
            applicability = ORDER_NOT_APPLICABLE
            reason = self.skipped_reason
        elif any(item not in context_set for item in definition.conditions):
            applicability = ORDER_APPLICABILITY_UNKNOWN
            reason = self.unknown_reason
        else:
            applicability = ORDER_APPLICABLE
            reason = self.applies_reason
        return ResolvedStructureOrderConstraint(
            definition.constraint,
            applicability,
            rule.before,
            rule.after,
            rule.enforced,
            rule.allow_missing,
            rule.weight,
            rule.minimum_gap,
            rule.maximum_gap,
            reason,
        )


def _consumer_protocol() -> StructureOrderConsumerProtocol:
    """构造互异的一等 MinimalInstruction reason。"""
    return StructureOrderConsumerProtocol(*tuple(
        minimal_instruction_identity((_BASE + 80, index))
        for index in range(7)
    ))


def _semantic_reasons():
    """返回 resolver 自身的 applies/skipped/unknown reason。"""
    return tuple(
        minimal_instruction_identity((_BASE + 81, index))
        for index in range(3)
    )


def _active_plan(
        engine, promoter, domain, pattern, *, event: int,
        instance: int, modality=None, exceptions=(),
        ) -> StructureOrderPromotionPlan:
    """累计、解析并晋升一个供 typed consumer 使用的约束。"""
    _, decision = _support(engine, pattern, event)
    plan = _plan(
        engine,
        pattern,
        domain,
        instance=instance,
        modality=modality,
        exceptions=exceptions,
    )
    _promote(promoter, plan, decision, event + 2)
    return plan


def _generation_syntax(
        domain: _Domain,
        promotion: StructureOrderPromotionPlan,
        *,
        constraints: tuple | None = None,
        context: tuple = (),
        mutate_slots=None,
        ) -> SyntaxPlan:
    """构造只依赖 typed Proposition/slot 的最小 G-02 SyntaxPlan。"""
    source = _source(_BASE + 200, source_id=_BASE + 201)
    scope = document_scope(source)
    bound = BoundProposition(
        proposition_identity(source, (_BASE + 202, 1)),
        minimal_instruction_identity((_BASE + 203, 1)),
        concept_identity((_BASE + 204, 1)),
        domain.structure,
        occurrence_identity(source, start=0, end=1, ordinal=0),
        concept_identity((_BASE + 205, 1)),
        (),
        (),
        (),
    )
    slots = promotion.slots
    if mutate_slots is not None:
        slots = mutate_slots(slots)
    values = tuple(
        StructureSlotValue(
            slot.slot,
            bound.template if index == 0
            else concept_identity((_BASE + 206, index)),
        )
        for index, slot in enumerate(slots)
    )
    candidate_key = (_BASE + 207, 1)
    sentence_identity = structure_concept_identity((_BASE + 208, 1))
    sentence = PlannedSentence(
        sentence_identity,
        domain.structure,
        0,
        (candidate_key,),
        slots,
        values,
        (PropositionSlotFiller(candidate_key, bound, values[0]),),
        minimal_instruction_identity((_BASE + 209, 1)),
        source,
        scope,
    )
    obligation = SyntaxLinearizationObligation(
        sentence_identity,
        domain.structure,
        sentence.values,
        ((promotion.constraint.constraint,)
         if constraints is None else constraints),
        context,
        minimal_instruction_identity((_BASE + 210, 1)),
        source,
        scope,
    )
    return SyntaxPlan((_BASE + 211, 1), (sentence,), (), (obligation,))


def test_l05b1_executes_g02_syntax_from_active_s07_projection_read_only():
    """G-02 句法只经 active graph schema/constraint 线性化并保留 lifecycle trace。"""
    backend = DictBackend()
    try:
        graphs = _graphs(backend)
        engine = OrderHypothesisEngine(_learning_protocol())
        domain = _domain(slot_count=2)
        promoter = OrderConstraintPromoter(
            engine, graphs.order_graph, graphs.lifecycle)
        promotion = _active_plan(
            engine,
            promoter,
            domain,
            _pattern(domain, kind=30),
            event=30,
            instance=30,
        )
        applies, skipped, unknown = _semantic_reasons()
        resolver = _SemanticsResolver({
            promotion.constraint.constraint: _ResolvedRule(
                domain.slots[1], domain.slots[0], True, False, 0, 0, None),
        }, applies_reason=applies, skipped_reason=skipped,
           unknown_reason=unknown)
        consumer = StructureOrderConsumer(
            graphs.lifecycle, resolver, _consumer_protocol())
        syntax = _generation_syntax(domain, promotion)
        request = GenerationStructureExecutionRequest(
            syntax,
            (SentenceStructureExecutionBudget(
                syntax.sentences[0].sentence,
                StructureOrderSearchBudget(20),
            ),),
        )
        before = {
            table: backend.count(table) for table in DUMP_TABLES
        }

        execution = GenerationStructureExecutionPlanner(
            graphs.lifecycle, consumer).execute(request)

        after = {
            table: backend.count(table) for table in DUMP_TABLES
        }
        sentence = execution.sentences[0]
        assert execution.complete
        assert tuple(item.slot for item in sentence.result.values) == (
            domain.slots[1], domain.slots[0])
        assert tuple(item.role for item in sentence.graph_slots) == tuple(
            item.role for item in sorted(
                _slot_definitions(domain),
                key=lambda item: item.slot.stable_key(),
            ))
        assert sentence.active_constraints[0].history
        assert before == after
        with pytest.raises(ValueError, match="逐点覆盖"):
            GenerationStructureExecutionPlan(request, ())
    finally:
        backend.close()


def test_l05b1_rejects_constraint_or_role_schema_drift():
    """漏报 active constraint 或替换 Role 都不能回退到旧 role_seq 猜测。"""
    backend = DictBackend()
    try:
        graphs = _graphs(backend)
        engine = OrderHypothesisEngine(_learning_protocol())
        domain = _domain(slot_count=2)
        promoter = OrderConstraintPromoter(
            engine, graphs.order_graph, graphs.lifecycle)
        promotion = _active_plan(
            engine,
            promoter,
            domain,
            _pattern(domain, kind=31),
            event=31,
            instance=31,
        )
        applies, skipped, unknown = _semantic_reasons()
        resolver = _SemanticsResolver({
            promotion.constraint.constraint: _ResolvedRule(
                domain.slots[0], domain.slots[1], True, False, 0, 0, None),
        }, applies_reason=applies, skipped_reason=skipped,
           unknown_reason=unknown)
        consumer = StructureOrderConsumer(
            graphs.lifecycle, resolver, _consumer_protocol())
        planner = GenerationStructureExecutionPlanner(
            graphs.lifecycle, consumer)

        missing = _generation_syntax(
            domain, promotion, constraints=())
        missing_request = GenerationStructureExecutionRequest(
            missing,
            (SentenceStructureExecutionBudget(
                missing.sentences[0].sentence,
                StructureOrderSearchBudget(20),
            ),),
        )
        with pytest.raises(ValueError, match="active S-07 constraint"):
            planner.execute(missing_request)

        def replace_role(slots):
            """只替换 Role，保持 slot/value identity 不变以检验 schema 双向核对。"""
            first = slots[0]
            return (
                StructureSlotDefinition(
                    first.structure,
                    first.slot,
                    role_identity((_BASE + 212, 1)),
                    first.value_type,
                ),
                *slots[1:],
            )

        drifted = _generation_syntax(
            domain, promotion, mutate_slots=replace_role)
        drifted_request = GenerationStructureExecutionRequest(
            drifted,
            (SentenceStructureExecutionBudget(
                drifted.sentences[0].sentence,
                StructureOrderSearchBudget(20),
            ),),
        )
        with pytest.raises(ValueError, match="graph schema"):
            planner.execute(drifted_request)
    finally:
        backend.close()


def test_consumer_uses_conditions_distance_and_content_independent_slots():
    """条件存在时邻接必要约束生效，换 filler 不改变结果，自由序不被硬拒绝。"""
    backend = DictBackend()
    try:
        graphs = _graphs(backend)
        engine = OrderHypothesisEngine(_learning_protocol())
        domain = _domain(slot_count=2)
        promoter = OrderConstraintPromoter(
            engine, graphs.order_graph, graphs.lifecycle)
        required = _active_plan(
            engine,
            promoter,
            domain,
            _pattern(domain, kind=10, conditional=True),
            event=1,
            instance=10,
        )
        optional = _active_plan(
            engine,
            promoter,
            domain,
            _pattern(domain, kind=11),
            event=4,
            instance=11,
            modality=domain.optional,
        )
        applies, skipped, unknown = _semantic_reasons()
        resolver = _SemanticsResolver({
            required.constraint.constraint: _ResolvedRule(
                domain.slots[0], domain.slots[1], True, False, 0, 0, 0),
            optional.constraint.constraint: _ResolvedRule(
                domain.slots[1], domain.slots[0], False, True, 0, 0, None),
        }, applies_reason=applies, skipped_reason=skipped,
           unknown_reason=unknown)
        consumer = StructureOrderConsumer(
            graphs.lifecycle, resolver, _consumer_protocol())
        structure = graphs.context.graph_ontology.resolve(domain.structure)
        first_fillers = (
            concept_identity((_BASE + 90, 1)),
            concept_identity((_BASE + 90, 2)),
        )
        second_fillers = (
            concept_identity((_BASE + 91, 1)),
            concept_identity((_BASE + 91, 2)),
        )

        for fillers in (first_fillers, second_fillers):
            accepted = consumer.parse(
                structure,
                tuple(
                    PositionedStructureSlotValue(
                        StructureSlotValue(slot, filler), index)
                    for index, (slot, filler) in enumerate(
                        zip(domain.slots, fillers, strict=True))
                ),
                context=(domain.condition,),
                budget=StructureOrderSearchBudget(20),
            )
            assert accepted.status == ORDER_CONSUMER_ACCEPTED

        non_adjacent = consumer.parse(
            structure,
            (
                PositionedStructureSlotValue(
                    StructureSlotValue(domain.slots[0], first_fillers[0]), 0),
                PositionedStructureSlotValue(
                    StructureSlotValue(domain.slots[1], first_fillers[1]), 2),
            ),
            context=(domain.condition,),
            budget=StructureOrderSearchBudget(20),
        )
        missing_condition = consumer.parse(
            structure,
            tuple(
                PositionedStructureSlotValue(
                    StructureSlotValue(slot, filler), index)
                for index, (slot, filler) in enumerate(
                    zip(domain.slots, first_fillers, strict=True))
            ),
            context=(),
            budget=StructureOrderSearchBudget(20),
        )

        assert non_adjacent.status == ORDER_CONSUMER_REJECTED
        assert missing_condition.status == ORDER_CONSUMER_UNKNOWN
    finally:
        backend.close()


def test_consumer_exception_skips_hard_rule_and_generation_preserves_partial_order():
    """显式 exception 关闭必要边；正常 context 下生成只重排受约束部分。"""
    backend = DictBackend()
    try:
        graphs = _graphs(backend)
        engine = OrderHypothesisEngine(_learning_protocol())
        domain = _domain(variant=2)
        promoter = OrderConstraintPromoter(
            engine, graphs.order_graph, graphs.lifecycle)
        plan = _active_plan(
            engine,
            promoter,
            domain,
            _pattern(domain, kind=20),
            event=1,
            instance=20,
            exceptions=(domain.exception,),
        )
        applies, skipped, unknown = _semantic_reasons()
        resolver = _SemanticsResolver({
            plan.constraint.constraint: _ResolvedRule(
                domain.slots[0], domain.slots[1], True, False, 0, 0, None),
        }, applies_reason=applies, skipped_reason=skipped,
           unknown_reason=unknown)
        consumer = StructureOrderConsumer(
            graphs.lifecycle, resolver, _consumer_protocol())
        structure = graphs.context.graph_ontology.resolve(domain.structure)
        fillers = tuple(
            concept_identity((_BASE + 92, index)) for index in range(3))
        base = tuple(
            StructureSlotValue(domain.slots[index], fillers[index])
            for index in (2, 1, 0)
        )

        normal = consumer.linearize(
            structure,
            base,
            context=(),
            budget=StructureOrderSearchBudget(30),
        )
        excepted = consumer.linearize(
            structure,
            base,
            context=(domain.exception,),
            budget=StructureOrderSearchBudget(30),
        )

        assert normal.status == ORDER_CONSUMER_ACCEPTED
        assert tuple(item.slot for item in normal.values) == (
            domain.slots[2], domain.slots[0], domain.slots[1])
        assert excepted.status == ORDER_CONSUMER_ACCEPTED
        assert excepted.values == base
    finally:
        backend.close()


def test_consumer_hard_cycle_and_preference_budget_fail_closed():
    """相反必要边返回 cycle unknown，未证完偏好最优时返回 budget unknown。"""
    backend = DictBackend()
    try:
        graphs = _graphs(backend)
        engine = OrderHypothesisEngine(_learning_protocol())
        domain = _domain(variant=3)
        promoter = OrderConstraintPromoter(
            engine, graphs.order_graph, graphs.lifecycle)
        forward = _active_plan(
            engine, promoter, domain, _pattern(domain, kind=30),
            event=1, instance=30)
        reverse = _active_plan(
            engine, promoter, domain, _pattern(domain, kind=31),
            event=4, instance=31)
        preferred = _active_plan(
            engine,
            promoter,
            domain,
            _pattern(domain, first=1, second=2, kind=32),
            event=7,
            instance=32,
            modality=domain.preferred,
        )
        applies, skipped, unknown = _semantic_reasons()
        common = dict(
            applies_reason=applies,
            skipped_reason=skipped,
            unknown_reason=unknown,
        )
        cycle_resolver = _SemanticsResolver({
            forward.constraint.constraint: _ResolvedRule(
                domain.slots[0], domain.slots[1], True, False, 0, 0, None),
            reverse.constraint.constraint: _ResolvedRule(
                domain.slots[1], domain.slots[0], True, False, 0, 0, None),
            preferred.constraint.constraint: _ResolvedRule(
                domain.slots[2], domain.slots[1], False, True, 1, 0, None),
        }, **common)
        consumer = StructureOrderConsumer(
            graphs.lifecycle, cycle_resolver, _consumer_protocol())
        structure = graphs.context.graph_ontology.resolve(domain.structure)
        values = tuple(
            StructureSlotValue(
                slot, concept_identity((_BASE + 93, index)))
            for index, slot in enumerate(domain.slots)
        )

        cycle = consumer.linearize(
            structure,
            values,
            context=(),
            budget=StructureOrderSearchBudget(100),
        )
        assert cycle.status == ORDER_CONSUMER_UNKNOWN
        assert cycle.reasons == (_consumer_protocol().constraint_cycle,)

        no_cycle_resolver = _SemanticsResolver({
            forward.constraint.constraint: _ResolvedRule(
                domain.slots[0], domain.slots[1], False, True, 0, 0, None),
            reverse.constraint.constraint: _ResolvedRule(
                domain.slots[1], domain.slots[0], False, True, 0, 0, None),
            preferred.constraint.constraint: _ResolvedRule(
                domain.slots[2], domain.slots[1], False, True, 1, 0, None),
        }, **common)
        budgeted = StructureOrderConsumer(
            graphs.lifecycle, no_cycle_resolver, _consumer_protocol()).linearize(
                structure,
                values,
                context=(),
                budget=StructureOrderSearchBudget(1),
            )
        assert budgeted.status == ORDER_CONSUMER_UNKNOWN
        assert budgeted.reasons == (_consumer_protocol().budget_exhausted,)
        assert budgeted.explored_states == 1
    finally:
        backend.close()


def test_parse_does_not_optimize_preferences_under_feasibility_budget():
    """解析只核必要约束可满足，不能因偏好排列未穷尽而错误 unknown。"""
    backend = DictBackend()
    try:
        graphs = _graphs(backend)
        engine = OrderHypothesisEngine(_learning_protocol())
        domain = _domain(variant=5)
        promoter = OrderConstraintPromoter(
            engine, graphs.order_graph, graphs.lifecycle)
        preferred = _active_plan(
            engine,
            promoter,
            domain,
            _pattern(domain, first=0, second=2, kind=50),
            event=1,
            instance=50,
            modality=domain.preferred,
        )
        applies, skipped, unknown = _semantic_reasons()
        resolver = _SemanticsResolver({
            preferred.constraint.constraint: _ResolvedRule(
                domain.slots[2], domain.slots[0], False, True, 1, 0, None),
        }, applies_reason=applies, skipped_reason=skipped,
           unknown_reason=unknown)
        consumer = StructureOrderConsumer(
            graphs.lifecycle, resolver, _consumer_protocol())
        structure = graphs.context.graph_ontology.resolve(domain.structure)
        assignments = tuple(
            PositionedStructureSlotValue(
                StructureSlotValue(
                    slot, concept_identity((_BASE + 94, index))),
                index,
            )
            for index, slot in enumerate(domain.slots)
        )

        result = consumer.parse(
            structure,
            assignments,
            context=(),
            budget=StructureOrderSearchBudget(4),
        )

        assert result.status == ORDER_CONSUMER_ACCEPTED
        assert result.evaluations[0].satisfied is False
    finally:
        backend.close()


def test_dump_load_recovers_structure_and_lifecycle_without_h06_engine(tmp_path):
    """dump/load 后只凭图对象和 statement 重建 active constraint 与 Event 历史。"""
    first_backend = DictBackend()
    try:
        first = _graphs(first_backend)
        engine = OrderHypothesisEngine(_learning_protocol())
        domain = _domain(variant=4, slot_count=2)
        pattern = _pattern(domain, kind=40)
        _, decision = _support(engine, pattern, 1)
        plan = _plan(engine, pattern, domain, instance=40)
        promoter = OrderConstraintPromoter(
            engine, first.order_graph, first.lifecycle)
        written = _promote(promoter, plan, decision, 3)
        expected_history = tuple(
            item.definition
            for item in first.lifecycle.project(
                written.constraint.constraint).history
        )
        dump_run(
            first_backend,
            str(tmp_path),
            "run_s07",
            spaces=[first.context.core_space.space_id],
            tables=DUMP_TABLES,
        )
    finally:
        first_backend.close()

    second_backend = DictBackend()
    try:
        context = make_train_context(second_backend)
        assert load_run(second_backend, str(tmp_path), "run_s07") == [1]
        ontology = context.graph_ontology
        order_refs = tuple(
            ontology.resolve(item) for item in first.order_predicate_identities)
        lifecycle_refs = tuple(
            ontology.resolve(item)
            for item in first.lifecycle_predicate_identities
        )
        order_graph = StructureOrderGraph(
            ontology, StructureOrderGraphPredicates(*order_refs))
        protocol = StructureOrderLifecycleProtocol(
            *lifecycle_refs,
            *first.states_and_kinds,
            (_BASE + 42, 1),
        )
        lifecycle = StructureOrderLifecycleGraph(order_graph, protocol)
        constraint_ref = ontology.resolve(plan.constraint.constraint)
        projection = lifecycle.project(constraint_ref)

        assert projection.constraint.definition == plan.constraint
        assert tuple(item.definition for item in projection.history) == (
            expected_history)
        assert projection.state == protocol.active_state
    finally:
        second_backend.close()
