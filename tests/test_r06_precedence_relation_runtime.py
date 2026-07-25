"""R-06 occurrence→Hypothesis→StructureConcept→消费者生产闭环测试。"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
)
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    ParserVersion,
    SourceRef,
    VersionBundle,
    concept_identity,
    language_branch_identity,
    minimal_instruction_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.order_hypothesis import (
    OrderAssessment,
    OrderLearningProtocol,
    OrderPattern,
)
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.cognition.shared.semantic_object import role_identity
from pure_integer_ai.cognition.shared.structure_order import (
    StructureOrderConstraintDefinition,
    StructureSlotDefinition,
)
from pure_integer_ai.cognition.shared.structure_order_consumer import (
    ORDER_APPLICABLE,
    ORDER_CONSUMER_ACCEPTED,
    PositionedStructureSlotValue,
    ResolvedStructureOrderConstraint,
    StructureOrderConsumerProtocol,
    StructureOrderSearchBudget,
    StructureSlotValue,
)
from pure_integer_ai.cognition.understanding.occurrence_index import (
    OccurrenceProtocol,
)
from pure_integer_ai.cognition.understanding.occurrence_order import (
    OccurrenceOrderProtocol,
)
from pure_integer_ai.cognition.understanding.order_constraint_promotion import (
    StructureOrderPromotionPlan,
)
from pure_integer_ai.cognition.understanding.order_hypothesis_adapter import (
    TypedOrderProjection,
)
from pure_integer_ai.experiments.collection import (
    COLLECT_PRECEDES,
    CollectedItem,
)
from pure_integer_ai.experiments.evaluation_isolation import isolated_evaluation
from pure_integer_ai.experiments.formal_train import (
    FormalTrainConfig,
    formal_train,
    make_train_context,
)
from pure_integer_ai.experiments.language_protocol_runtime import (
    install_language_graph_protocols,
)
from pure_integer_ai.experiments.precedence_relation_runtime import (
    PRECEDENCE_LIFECYCLE_DEMOTED,
    PRECEDENCE_LIFECYCLE_PROMOTED,
    PrecedenceConsumptionPlan,
    PrecedenceRelationProtocol,
    PrecedenceResolutionPlan,
    install_precedence_relation_runtime,
)
from pure_integer_ai.experiments.round_runtime import DefaultRoundRunner
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import SOURCE_BARE_TEXT
from pure_integer_ai.training.cursor import dump_run, load_run
from pure_integer_ai.training.stages import STAGE1_SKELETON


_BASE = 31400


@dataclass(frozen=True)
class _Domain:
    """测试课程使用的全部一等结构身份。"""

    language: object
    order_kind: object
    family: object
    structure: object
    slots: tuple
    context: object
    constraint_kind: object
    constraint_instance: object
    modality: object
    value_type: object
    roles: tuple
    applies_reason: object


def _domain() -> _Domain:
    """构造不含词面和位置编码的共享结构定义。"""
    return _Domain(
        language_branch_identity((_BASE + 1, 1)),
        concept_identity((_BASE + 2, 1)),
        structure_concept_identity((_BASE + 3, 1)),
        structure_concept_identity((_BASE + 4, 1)),
        (
            structure_concept_identity((_BASE + 5, 1)),
            structure_concept_identity((_BASE + 5, 2)),
        ),
        concept_identity((_BASE + 6, 1)),
        concept_identity((_BASE + 7, 1)),
        structure_concept_identity((_BASE + 8, 1)),
        concept_identity((_BASE + 9, 1)),
        concept_identity((_BASE + 10, 1)),
        (
            role_identity((_BASE + 11, 1)),
            role_identity((_BASE + 11, 2)),
        ),
        minimal_instruction_identity((_BASE + 12, 1)),
    )


def _source(
        document_id: int, *, parser: int = 1,
        source_id: int = _BASE + 20,
        ) -> SourceRef:
    """构造可用 parser version 显式修正的真实文档来源。"""
    return SourceRef(
        SOURCE_BARE_TEXT,
        source_id,
        document_id,
        GLOBAL_OWNER_SCOPE,
        VersionBundle(parser=ParserVersion(parser)),
    )


def _learning_protocol() -> OrderLearningProtocol:
    """构造与真实 observation 分离的聚合学习 manifest。"""
    aggregate = SourceRef(
        _BASE + 21,
        _BASE + 22,
        0,
        GLOBAL_OWNER_SCOPE,
        VersionBundle(),
    )
    return OrderLearningProtocol(
        (_BASE + 23, 1),
        (_BASE + 24, 1),
        (_BASE + 24, 2),
        (_BASE + 24, 3),
        (_BASE + 25, 1),
        aggregate,
        document_scope(aggregate),
    )


def _consumer_protocol() -> StructureOrderConsumerProtocol:
    """构造七个互异的一等消费者失败 reason。"""
    return StructureOrderConsumerProtocol(*tuple(
        minimal_instruction_identity((_BASE + 30, index))
        for index in range(7)
    ))


def _precedence_protocol() -> PrecedenceRelationProtocol:
    """构造可跨评测 clone 和 dump/load 重建的开放图协议。"""
    return PrecedenceRelationProtocol(
        _learning_protocol(),
        tuple(
            concept_identity((_BASE + 40, index))
            for index in range(19)
        ),
        tuple(
            concept_identity((_BASE + 41, index))
            for index in range(6)
        ),
        tuple(
            concept_identity((_BASE + 42, index))
            for index in range(6)
        ),
        (_BASE + 43, 1),
        _consumer_protocol(),
        provenance_kind=_BASE + 44,
        qualifiers=(_BASE + 45,),
    )


class _Course:
    """按显式文档集合控制反例，并以 slot 而非词面定义顺序。"""

    def __init__(self, domain: _Domain, *, reverse_documents=()) -> None:
        self.domain = domain
        self.reverse_documents = frozenset(reverse_documents)
        first, second = sorted(
            domain.slots, key=lambda item: item.stable_key())
        self.pattern = OrderPattern(
            domain.language,
            domain.order_kind,
            domain.family,
            domain.structure,
            first,
            second,
            domain.constraint_kind,
            domain.context,
        )

    def map_step(self, step):
        """只映射首个相邻事实，方向由课程显式文档标记提供。"""
        if step.index != 0:
            return ()
        source = step.fact.statement.assertion.scope.source
        endpoint_order = (
            (1, 0)
            if source.document_id in self.reverse_documents
            and source.versions.parser.value == 1
            else (0, 1)
        )
        return (TypedOrderProjection(self.pattern, endpoint_order),)

    def assess(self, _pattern, observation):
        """依据 mapper 已声明的 slot 端点位置返回三态中的支持或反驳。"""
        stance = (
            EVIDENCE_SUPPORT
            if observation.first_position < observation.second_position
            else EVIDENCE_REFUTE
        )
        return OrderAssessment(
            stance,
            (_BASE + 50, observation.source.document_id,
             observation.source.versions.parser.value),
        )

    def supersedes_evidence_id(self, mapped, prior):
        """同文档 parser 升级时显式替代旧版本 Evidence。"""
        current = mapped.observation.source
        for trace in reversed(prior):
            previous = trace.mapped.observation.source
            if (previous.source_id == current.source_id
                    and previous.document_id == current.document_id
                    and previous.versions.parser != current.versions.parser):
                return trace.result.evidence.evidence_id
        return 0

    def resolution_plan(self, _mapped, _evidence):
        """本测试无竞争 scorer 或退出指令，保持 H-04 默认三态裁决。"""
        return PrecedenceResolutionPlan()

    def promotion_plan(self, pattern, hypothesis):
        """把共享模式映射为两个一等 slot、Role 和必要约束实例。"""
        slots = tuple(
            StructureSlotDefinition(
                self.domain.structure,
                slot,
                role,
                self.domain.value_type,
            )
            for slot, role in zip(
                self.domain.slots, self.domain.roles, strict=True)
        )
        definition = StructureOrderConstraintDefinition(
            self.domain.constraint_instance,
            pattern.language_branch,
            pattern.structure_family,
            pattern.structure_candidate,
            pattern.first_slot,
            pattern.second_slot,
            pattern.order_kind,
            pattern.constraint,
            self.domain.modality,
            pattern.context,
            (),
            (),
            (),
            hypothesis,
        )
        return StructureOrderPromotionPlan(slots, definition)

    def consumption_plan(self, mapped):
        """用真实 occurrence 作 filler，并故意给生成传入反向基序。"""
        observation = mapped.observation
        first = StructureSlotValue(
            mapped.pattern.first_slot,
            observation.first_occurrence,
        )
        second = StructureSlotValue(
            mapped.pattern.second_slot,
            observation.second_occurrence,
        )
        return PrecedenceConsumptionPlan(
            (
                PositionedStructureSlotValue(
                    first, observation.first_position),
                PositionedStructureSlotValue(
                    second, observation.second_position),
            ),
            (second, first),
            (),
            StructureOrderSearchBudget(16),
        )

    def resolve(self, definition, _context):
        """把课程的一等 constraint 解释为 first_slot 必须先于 second_slot。"""
        return ResolvedStructureOrderConstraint(
            definition.constraint,
            ORDER_APPLICABLE,
            definition.first_slot,
            definition.second_slot,
            True,
            False,
            0,
            0,
            0,
            self.domain.applies_reason,
        )

    def clone_for_evaluation(self):
        """返回共享不可变本体但不共享课程容器的评测副本。"""
        return _Course(
            self.domain,
            reverse_documents=tuple(self.reverse_documents),
        )

    def state_key(self):
        """返回课程方向标记和共享模式的完整稳定状态。"""
        return (
            tuple(sorted(self.reverse_documents)),
            self.pattern.stable_key(),
        )


def _install(ctx, course):
    """安装 L-03、L-06 和 R-06 runtime。"""
    install_language_graph_protocols(
        ctx,
        occurrence_protocol=OccurrenceProtocol((_BASE + 60, 1)),
        occurrence_order_protocol=OccurrenceOrderProtocol((_BASE + 60, 2)),
    )
    return install_precedence_relation_runtime(
        ctx,
        _precedence_protocol(),
        course,
    )


def _item(tokens, source):
    """构造不依赖旧 role 语义的两 token 来源项。"""
    return CollectedItem(
        tokens=list(tokens),
        raw_text="".join(tokens),
        role_seq=[1, 1],
        collect_type=COLLECT_PRECEDES,
        source=SOURCE_BARE_TEXT,
        source_ref=source,
    )


def test_runtime_reuses_slots_across_vocabulary_and_replay_is_idempotent():
    """换词共享同一 Hypothesis，parse/linearize 真消费且重放不增 Evidence。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        runtime = _install(ctx, _Course(_domain()))
        runner = DefaultRoundRunner()

        runner.run_round(ctx, _item(("甲", "乙"), _source(1)),
                         STAGE1_SKELETON, 1)
        first = ctx.precedence_relation_reports[-1].observations[0]
        runner.run_round(ctx, _item(("丙", "丁"), _source(2)),
                         STAGE1_SKELETON, 2)
        held = ctx.precedence_relation_reports[-1].observations[0]
        runner.run_round(ctx, _item(("丙", "丁"), _source(2)),
                         STAGE1_SKELETON, 3)
        replay = ctx.precedence_relation_reports[-1].observations[0]

        assert first.lifecycle_action == PRECEDENCE_LIFECYCLE_PROMOTED
        assert first.parse.status == ORDER_CONSUMER_ACCEPTED
        assert first.linearization.status == ORDER_CONSUMER_ACCEPTED
        assert held.evidence.hypothesis == first.evidence.hypothesis
        assert held.parse.status == ORDER_CONSUMER_ACCEPTED
        assert tuple(item.slot for item in held.linearization.values) == (
            runtime.course.pattern.first_slot,
            runtime.course.pattern.second_slot,
        )
        assert replay.duplicate is True
        assert runtime.evidence_count() == 2
    finally:
        backend.close()


def test_counterexample_demotes_and_parser_correction_repromotes():
    """反向 observation 保留冲突并降级，显式 parser supersede 后重新晋升。"""
    backend = DictBackend()
    try:
        domain = _domain()
        ctx = make_train_context(backend)
        runtime = _install(ctx, _Course(domain, reverse_documents=(3,)))
        runner = DefaultRoundRunner()
        runner.run_round(ctx, _item(("甲", "乙"), _source(1)),
                         STAGE1_SKELETON, 1)
        runner.run_round(ctx, _item(("戊", "己"), _source(3, parser=1)),
                         STAGE1_SKELETON, 2)
        counter = ctx.precedence_relation_reports[-1].observations[0]

        assert counter.lifecycle_action == PRECEDENCE_LIFECYCLE_DEMOTED
        assert counter.parse is None
        assert counter.linearization is None

        ctx.work_memory.end_session()
        runner.run_round(ctx, _item(("戊", "己"), _source(3, parser=2)),
                         STAGE1_SKELETON, 3)
        corrected = ctx.precedence_relation_reports[-1].observations[0]

        assert corrected.evidence.evidence.supersedes_evidence_id == (
            counter.evidence.evidence.evidence_id)
        assert corrected.lifecycle_action == PRECEDENCE_LIFECYCLE_PROMOTED
        assert corrected.parse.status == ORDER_CONSUMER_ACCEPTED
        assert corrected.linearization.status == ORDER_CONSUMER_ACCEPTED
        assert runtime.evidence_count() == 3
    finally:
        backend.close()


def test_evaluation_clone_consumes_held_out_without_host_pollution():
    """held-out 新词只读复用 active 结构，宿主 H-06、课程、图和报告零变化。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        runtime = _install(ctx, _Course(_domain()))
        runner = DefaultRoundRunner()
        runner.run_round(ctx, _item(("甲", "乙"), _source(1)),
                         STAGE1_SKELETON, 1)
        baseline = runtime.state_key()
        report_count = len(ctx.precedence_relation_reports)

        with isolated_evaluation(ctx, label="r06-held-out") as eval_ctx:
            runner.run_round(
                eval_ctx,
                _item(("新", "词"), _source(9)),
                STAGE1_SKELETON,
                2,
            )
            report = eval_ctx.precedence_relation_reports[-1]
            observation = report.observations[0]
            assert report.read_only is True
            assert observation.evidence is None
            assert observation.parse.status == ORDER_CONSUMER_ACCEPTED
            assert observation.linearization.status == ORDER_CONSUMER_ACCEPTED

        assert runtime.state_key() == baseline
        assert len(ctx.precedence_relation_reports) == report_count
    finally:
        backend.close()


def test_structure_consumer_behavior_survives_graph_dump_load(tmp_path):
    """S-07 图约束恢复后无需 H-06 续写即可只读解析和线性化。"""
    backend = DictBackend()
    restored_backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        course = _Course(_domain())
        _install(ctx, course)
        runner = DefaultRoundRunner()
        runner.run_round(ctx, _item(("甲", "乙"), _source(1)),
                         STAGE1_SKELETON, 1)
        dump_run(
            backend,
            str(tmp_path),
            "r06",
            spaces=[ctx.space_id],
        )

        restored = make_train_context(restored_backend)
        load_run(restored_backend, str(tmp_path), "r06")
        restored_runtime = _install(restored, _Course(_domain()))
        restored.precedence_relation_runtime = None
        held_item = _item(("新", "词"), _source(8))
        runner.run_round(restored, held_item, STAGE1_SKELETON, 2)
        restored.precedence_relation_runtime = restored_runtime
        report = restored_runtime.process(
            document_scope(held_item.source_ref),
            read_only=True,
        )

        assert report.observations[0].parse.status == ORDER_CONSUMER_ACCEPTED
        assert report.observations[0].linearization.status == (
            ORDER_CONSUMER_ACCEPTED)
        assert restored_runtime.evidence_count() == 0
    finally:
        backend.close()
        restored_backend.close()


def test_formal_train_installs_and_reports_precedence_runtime(
        tmp_path, monkeypatch):
    """顶层训练入口成对安装 R-06 协议和课程并返回真实闭环报告。"""
    from pure_integer_ai.training import stages as training_stages

    monkeypatch.setattr(training_stages, "FLOOR_GRAPH_SIZE_S1", 0)
    result = formal_train(
        FormalTrainConfig(
            run_dir=str(tmp_path),
            run_id="r06-formal",
            rounds_per_stage=1,
            active_training_stages=(STAGE1_SKELETON,),
            language_occurrence_protocol=OccurrenceProtocol((_BASE + 60, 1)),
            language_occurrence_order_protocol=OccurrenceOrderProtocol(
                (_BASE + 60, 2)),
            language_precedence_protocol=_precedence_protocol(),
            language_precedence_course=_Course(_domain()),
        ),
        [_item(("甲", "乙"), _source(1))],
        backend=DictBackend(),
        runner=DefaultRoundRunner(),
    )

    assert result.precedence_evidence_count == 1
    assert result.precedence_relation_reports
    observation = result.precedence_relation_reports[-1].observations[0]
    assert observation.parse.status == ORDER_CONSUMER_ACCEPTED
    assert observation.linearization.status == ORDER_CONSUMER_ACCEPTED


def test_invalid_course_schema_fails_before_h06_evidence_commit():
    """非法 promotion schema 不得留下只有 H-06、没有下游定义的课程半写。"""

    class _InvalidCourse(_Course):
        """只破坏 promotion 输出类型，其余课程协议保持合法。"""

        def promotion_plan(self, pattern, hypothesis):
            """返回非法对象以验证 runtime 的写前纯校验。"""
            return (pattern, hypothesis)

    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        runtime = _install(ctx, _InvalidCourse(_domain()))
        baseline = runtime.engine.state_key()

        with pytest.raises(TypeError, match="promotion_plan"):
            DefaultRoundRunner().run_round(
                ctx,
                _item(("甲", "乙"), _source(1)),
                STAGE1_SKELETON,
                1,
            )

        assert runtime.evidence_count() == 0
        assert runtime.engine.state_key() == baseline
    finally:
        backend.close()
