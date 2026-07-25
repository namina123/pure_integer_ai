"""R-06B S-02 Evidence 到 active event-time 投影的生产对抗测试。"""
from __future__ import annotations

from dataclasses import replace

import pytest

from pure_integer_ai.cognition.shared.event_time import (
    EVENT_TIME_BEFORE,
    EVENT_TIME_CONSISTENT,
    EVENT_TIME_EMPTY,
    ResolvedEventTimeRelation,
)
from pure_integer_ai.cognition.shared.hypothesis import (
    EPISTEMIC_CONFLICTED,
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    HypothesisKey,
)
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    OBJECT_EVENT,
    OBJECT_PROPOSITION,
    SourceRef,
    VersionBundle,
    concept_identity,
    language_branch_identity,
    minimal_instruction_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.logic_executor import LogicEvidenceState
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.cognition.shared.semantic_object import (
    context_scope_identity,
    role_identity,
)
from pure_integer_ai.cognition.shared.typed_binding import BindingEnvironment
from pure_integer_ai.cognition.understanding.occurrence_index import (
    OccurrenceIndex,
    OccurrenceProtocol,
)
from pure_integer_ai.cognition.understanding.segmentation_span import (
    SegmentationSpanProtocol,
)
from pure_integer_ai.cognition.understanding.semantic_builder import (
    LocalSemanticRef,
    SemanticBindingSpec,
    SemanticBuildPlan,
    SemanticFillerSpec,
    SemanticObjectSpec,
    SemanticPropositionSpec,
)
from pure_integer_ai.cognition.understanding.span_index import (
    SpanIndex,
    SpanProtocol,
)
from pure_integer_ai.experiments.collection import CollectedItem
from pure_integer_ai.experiments.evaluation_isolation import isolated_evaluation
from pure_integer_ai.experiments.evaluation_protocol import ProtocolKey
from pure_integer_ai.experiments.event_time_runtime import (
    EventTimeEvidenceRequest,
    EventTimeProductionProtocol,
    EventTimeRoundRequest,
    install_event_time_relation_runtime,
)
from pure_integer_ai.experiments.event_time_verification import (
    EventTimeVerificationProtocol,
    EventTimeVerificationRequest,
)
from pure_integer_ai.experiments.formal_train import (
    FormalTrainConfig,
    formal_train,
    make_train_context,
)
from pure_integer_ai.experiments.language_semantic_course import (
    LanguageSemanticCourseDecision,
    LanguageSemanticLesson,
    SemanticCourseEvidenceSpec,
    SemanticCourseTemplateScope,
)
from pure_integer_ai.experiments.language_semantic_runtime import (
    LanguageSemanticCourseRun,
    install_language_semantic_course_runtime,
)
from pure_integer_ai.experiments.round_runtime import DefaultRoundRunner
from pure_integer_ai.experiments.verification_orchestration import (
    VERDICT_SUPPORT,
)
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import EPI_STRUCTURED, SOURCE_BARE_TEXT
from pure_integer_ai.training.cursor import dump_run, load_run
from pure_integer_ai.training.stages import STAGE1_SKELETON
from tests.test_l05b2b_semantic_course_runtime import _protocol as _semantic_protocol


_BASE = 33300
_EVENT_FIRST = LocalSemanticRef(OBJECT_EVENT, (1,))
_EVENT_SECOND = LocalSemanticRef(OBJECT_EVENT, (2,))
_PROPOSITION = LocalSemanticRef(OBJECT_PROPOSITION, (3,))
_RELATION = concept_identity((_BASE + 1, 1))
_SUBJECT_ROLE = role_identity((_BASE + 2, 1))
_OBJECT_ROLE = role_identity((_BASE + 2, 2))


def _source(document_id: int = 1) -> SourceRef:
    """构造来源化语言课程文档。"""
    return SourceRef(
        SOURCE_BARE_TEXT,
        _BASE + 3,
        document_id,
        GLOBAL_OWNER_SCOPE,
        VersionBundle(),
    )


def _aggregate_source() -> SourceRef:
    """构造只承载 event-time H-00 历史的聚合来源。"""
    return SourceRef(
        _BASE + 4,
        _BASE + 5,
        0,
        GLOBAL_OWNER_SCOPE,
        VersionBundle(),
    )


def _event_protocol() -> EventTimeProductionProtocol:
    """注入 H-00、M-03、R-09 和 assertion 限定项协议。"""
    aggregate = _aggregate_source()
    return EventTimeProductionProtocol(
        hypothesis_kind=(_BASE + 10, 1),
        aggregate_source=aggregate,
        aggregate_scope=document_scope(aggregate),
        history_namespace=(_BASE + 11, 1),
        competition_namespace=(_BASE + 12, 1),
        evidence_reason_key=(_BASE + 13, 1),
        evidence_namespace=(_BASE + 14, 1),
        projection_qualifier_key=(_BASE + 15, 1),
        verification=EventTimeVerificationProtocol(
            ProtocolKey((_BASE + 16, 1)),
            ProtocolKey((_BASE + 16, 2)),
        ),
        provenance_kind=SOURCE_BARE_TEXT,
        epistemic_origin=EPI_STRUCTURED,
        content_version=1,
    )


class _Semantics:
    """只从注入注册表解释 relation 方向。"""

    def __init__(self, directions=None) -> None:
        self.directions = dict(
            {_RELATION: EVENT_TIME_BEFORE}
            if directions is None else directions
        )

    def resolve(self, relation):
        """返回 relation 自身、注入方向和审计键。"""
        direction = self.directions[relation]
        return ResolvedEventTimeRelation(
            relation,
            direction,
            (_BASE + 17, direction),
        )

    def clone_for_evaluation(self):
        """复制关系注册表，后续修改互不共享。"""
        return _Semantics(self.directions)

    def state_key(self):
        """展开全部 relation 和方向，不使用对象地址。"""
        values = [_BASE + 18, len(self.directions)]
        for relation, direction in sorted(
                self.directions.items(),
                key=lambda item: item[0].stable_key()):
            key = relation.stable_key()
            values.extend((len(key), *key, direction))
        return tuple(values)


class _SemanticMapper:
    """按当前 stance 生成同构双 Event 命题课程。"""

    def __init__(self, stance=EVIDENCE_SUPPORT, evidence_id=_BASE + 30) -> None:
        self.stance = stance
        self.evidence_id = evidence_id
        self.supersedes_evidence_id = 0
        self.calls = 0

    def map(self, input_value):
        """从 typed occurrence 选择锚点，不读取词面或 token 位置语义。"""
        self.calls += 1
        reason = minimal_instruction_identity((_BASE + 31, 1))
        if input_value.read_only:
            return LanguageSemanticCourseDecision(
                reason,
                (_BASE + 31, self.calls, 0),
            )
        if not input_value.occurrences:
            raise ValueError("测试语义课程缺少 occurrence anchor")
        lesson = _lesson(
            input_value.source,
            input_value.occurrences[0],
            stance=self.stance,
            evidence_id=self.evidence_id,
            supersedes_evidence_id=self.supersedes_evidence_id,
        )
        return LanguageSemanticCourseDecision(
            reason,
            (_BASE + 31, self.calls, 1),
            lesson,
        )

    def clone_for_evaluation(self):
        """复制课程模式与调用水位。"""
        cloned = _SemanticMapper(self.stance, self.evidence_id)
        cloned.supersedes_evidence_id = self.supersedes_evidence_id
        cloned.calls = self.calls
        return cloned

    def state_key(self):
        """返回 stance、Evidence 版本和调用水位。"""
        return (
            _BASE + 32,
            self.stance,
            self.evidence_id,
            self.supersedes_evidence_id,
            self.calls,
        )


def _lesson(
        source: SourceRef,
        anchor,
        *,
        stance: int,
        evidence_id: int,
        supersedes_evidence_id: int,
        ) -> LanguageSemanticLesson:
    """构造 predicate 与两个开放 Role 均为一等对象的时间命题。"""
    scope = document_scope(source)
    upstream = HypothesisKey(
        (_BASE + 40, 1),
        (_BASE + 40, 2),
        (_BASE + 40, 3),
        scope,
        source,
    )
    first = SemanticObjectSpec(OBJECT_EVENT, _EVENT_FIRST.local_key)
    second = SemanticObjectSpec(OBJECT_EVENT, _EVENT_SECOND.local_key)
    proposition = SemanticPropositionSpec(
        _PROPOSITION.local_key,
        (_BASE + 41, 1),
        _RELATION,
        structure_concept_identity((_BASE + 42, 1)),
        (
            SemanticBindingSpec(
                _SUBJECT_ROLE,
                SemanticFillerSpec(local_ref=first.local_ref),
            ),
            SemanticBindingSpec(
                _OBJECT_ROLE,
                SemanticFillerSpec(local_ref=second.local_ref),
            ),
        ),
    )
    plan = SemanticBuildPlan(
        upstream,
        (_BASE + 43, 1),
        (first, second),
        (proposition,),
    )
    evidence = SemanticCourseEvidenceSpec(
        _PROPOSITION,
        evidence_id,
        stance,
        (_BASE + 44, stance),
        source,
        evidence_id,
        (_BASE + 45, stance),
        supersedes_evidence_id,
    )
    return LanguageSemanticLesson(
        anchor,
        plan,
        (evidence,),
        (SemanticCourseTemplateScope(
            _PROPOSITION,
            context_scope_identity(source, (_BASE + 46, 1)),
        ),),
        BindingEnvironment(),
        _PROPOSITION,
        (_PROPOSITION,),
        minimal_instruction_identity((_BASE + 47, 1)),
        LogicEvidenceState(
            stance == EVIDENCE_SUPPORT,
            stance == EVIDENCE_REFUTE,
        ),
        language_branch_identity((_BASE + 48, 1)),
    )


class _EventCourse:
    """从同轮 S-02 原子定义恢复 Role filler 并形成 typed 请求。"""

    def __init__(self, *, tamper_role=False, tamper_endpoint=False) -> None:
        self.tamper_role = tamper_role
        self.tamper_endpoint = tamper_endpoint
        self.calls = 0

    def request(self, value):
        """训练引用同轮 Evidence，held-out 只请求已有投影核验。"""
        self.calls += 1
        verification = EventTimeVerificationRequest(
            value.scope,
            (_RELATION,),
        )
        if value.read_only:
            return EventTimeRoundRequest(
                value.scope,
                verifications=(verification,),
            )
        run = value.semantic_run
        if run.materialized is None or len(run.materialized.candidates) != 1:
            raise ValueError("测试 event-time 课程需要唯一 S-02 candidate")
        candidate = run.materialized.candidates[0]
        definition = candidate.atomic.definition
        fillers = {
            (binding.role, binding.ordinal): binding.filler
            for binding in definition.bindings
        }
        subject = fillers[(_SUBJECT_ROLE, 0)]
        object_identity = fillers[(_OBJECT_ROLE, 0)]
        if self.tamper_endpoint:
            object_identity = subject
        object_role = (
            role_identity((_BASE + 49, 1))
            if self.tamper_role else _OBJECT_ROLE
        )
        request = EventTimeEvidenceRequest(
            _RELATION,
            definition.proposition,
            _SUBJECT_ROLE,
            0,
            subject,
            object_role,
            0,
            object_identity,
            run.evidence[0].evidence_id,
        )
        return EventTimeRoundRequest(
            value.scope,
            (request,),
            (verification,),
        )

    def clone_for_evaluation(self):
        """复制课程策略和调用水位。"""
        cloned = _EventCourse(
            tamper_role=self.tamper_role,
            tamper_endpoint=self.tamper_endpoint,
        )
        cloned.calls = self.calls
        return cloned

    def state_key(self):
        """返回对抗开关和调用水位的严格整数状态。"""
        return (
            _BASE + 50,
            1 if self.tamper_role else 0,
            1 if self.tamper_endpoint else 0,
            self.calls,
        )


def _span_protocol() -> SegmentationSpanProtocol:
    """安装 S-02 所需但本测试不解释边界语义的 Span 地基。"""
    return SegmentationSpanProtocol(
        SpanProtocol(
            (_BASE + 60, 1),
            (_BASE + 60, 2),
            (_BASE + 60, 3),
            (_BASE + 60, 4),
        ),
        (_BASE + 61, 1),
        (_BASE + 61, 2),
        (_BASE + 61, 3),
        (_BASE + 61, 4),
    )


def _item(source: SourceRef) -> CollectedItem:
    """构造只提供来源和表层观察、不携带时间标签的语言项。"""
    return CollectedItem(
        tokens=["甲", "乙"],
        raw_text="甲乙",
        role_seq=[1, 1],
        source=SOURCE_BARE_TEXT,
        source_ref=source,
    )


def _install(
        ctx,
        semantic_mapper: _SemanticMapper,
        event_course: _EventCourse,
        ):
    """在同一图上安装 occurrence/span、S-02 和 R-06B owner。"""
    ctx.occurrence_index = OccurrenceIndex(
        ctx.graph_ontology,
        ctx.scoped_identity_store,
        OccurrenceProtocol((_BASE + 70, 1)),
    )
    span = _span_protocol().span_protocol
    ctx.span_index = SpanIndex(
        ctx.graph_ontology,
        ctx.scoped_identity_store,
        span,
        ctx.occurrence_index,
    )
    install_language_semantic_course_runtime(
        ctx,
        _semantic_protocol(semantic_mapper),
    )
    return install_event_time_relation_runtime(
        ctx,
        _event_protocol(),
        _Semantics(),
        event_course,
    )


def _read_only_run(run: LanguageSemanticCourseRun) -> LanguageSemanticCourseRun:
    """从训练 run 构造不携带 lesson 或训练产物的只读同源输入。"""
    return LanguageSemanticCourseRun(
        replace(run.input_value, read_only=True),
        LanguageSemanticCourseDecision(
            minimal_instruction_identity((_BASE + 80, 1)),
            (_BASE + 80, 2),
        ),
    )


def test_formal_train_installs_s02_event_time_evidence_and_r09(tmp_path, monkeypatch):
    """顶层训练从同轮 S-02 RoleBinding 写 Evidence、active fact 和 R-09 报告。"""
    from pure_integer_ai.training import stages as training_stages

    monkeypatch.setattr(training_stages, "FLOOR_GRAPH_SIZE_S1", 0)
    semantic_mapper = _SemanticMapper()
    event_course = _EventCourse()
    result = formal_train(
        FormalTrainConfig(
            run_dir=str(tmp_path),
            run_id="r06b-formal",
            rounds_per_stage=1,
            active_training_stages=(STAGE1_SKELETON,),
            language_occurrence_protocol=OccurrenceProtocol(
                (_BASE + 70, 1)),
            language_span_protocol=_span_protocol(),
            language_semantic_course_protocol=_semantic_protocol(
                semantic_mapper),
            language_event_time_protocol=_event_protocol(),
            language_event_time_semantics=_Semantics(),
            language_event_time_course=event_course,
        ),
        [_item(_source())],
        backend=DictBackend(),
        runner=DefaultRoundRunner(),
    )

    assert result.event_time_evidence_count == 1
    report = result.event_time_relation_reports[-1]
    assert report.evidence[0].fact is not None
    dimension = _event_protocol().verification.dimension
    assert report.verifications[0].dimension_results(
        dimension)[0].verdict == VERDICT_SUPPORT


@pytest.mark.parametrize("tamper", ["role", "endpoint"])
def test_production_writer_rejects_non_s02_role_or_endpoint(tamper):
    """课程替换 Role 或端点时，在 event-time H-00 和事实写入前失败。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        runtime = _install(
            ctx,
            _SemanticMapper(),
            _EventCourse(
                tamper_role=tamper == "role",
                tamper_endpoint=tamper == "endpoint",
            ),
        )

        with pytest.raises(ValueError, match="RoleBinding|自环"):
            DefaultRoundRunner().run_round(
                ctx,
                _item(_source()),
                STAGE1_SKELETON,
                1,
            )

        assert runtime.evidence_count() == 0
        assert runtime.facts.read(
            (_RELATION,),
            scope=document_scope(_source()),
            active_only=False,
        ).facts == ()
    finally:
        backend.close()


def test_refute_evidence_deactivates_but_does_not_delete_raw_projection():
    """新增反驳使 H-00 conflicted，历史 statement 保留但 verifier 不再消费。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        semantic_mapper = _SemanticMapper()
        runtime = _install(ctx, semantic_mapper, _EventCourse())
        runner = DefaultRoundRunner()
        source = _source()
        runner.run_round(ctx, _item(source), STAGE1_SKELETON, 1)
        first_trace = ctx.event_time_relation_reports[-1].evidence[0]

        semantic_mapper.stance = EVIDENCE_REFUTE
        semantic_mapper.evidence_id = _BASE + 31
        runner.run_round(ctx, _item(source), STAGE1_SKELETON, 2)

        snapshot = runtime.ledger.snapshot(first_trace.hypothesis)
        assert snapshot.epistemic_status == EPISTEMIC_CONFLICTED
        assert len(runtime.facts.read(
            (_RELATION,),
            scope=document_scope(source),
            active_only=False,
        ).facts) == 1
        verified = runtime.verifier.verify(
            (_RELATION,),
            scope=document_scope(source),
        )
        assert verified.status == EVENT_TIME_EMPTY
        assert verified.fact_set.facts == ()
    finally:
        backend.close()


def test_incremental_preflight_never_clones_full_event_time_history():
    """历史增长后新增来源只克隆相关竞争组，不复制完整 H-00/H-04。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        semantic_mapper = _SemanticMapper()
        runtime = _install(ctx, semantic_mapper, _EventCourse())
        runner = DefaultRoundRunner()
        runner.run_round(
            ctx,
            _item(_source(1)),
            STAGE1_SKELETON,
            1,
        )

        def reject_full_clone(*args, **kwargs):
            """全量 clone 一旦被增量路径调用就立即暴露性能回退。"""
            raise AssertionError("增量 event-time 预检不得全量 clone")

        class RejectGlobalIteration(dict):
            """保留按键访问，但禁止局部预检遍历无关全局历史。"""

            def items(self):
                """任何全表 items 扫描都视为增量复杂度回退。"""
                raise AssertionError("增量 event-time 预检不得全表扫描")

        runtime.ledger.clone = reject_full_clone
        runtime.resolver.clone = reject_full_clone
        runtime.ledger._superseded_evidence = RejectGlobalIteration(
            runtime.ledger._superseded_evidence)
        runtime.resolver._decisions = RejectGlobalIteration(
            runtime.resolver._decisions)
        semantic_mapper.evidence_id = _BASE + 130
        runner.run_round(
            ctx,
            _item(_source(2)),
            STAGE1_SKELETON,
            2,
        )

        assert runtime.evidence_count() == 2
        assert len(runtime.ledger.hypotheses()) == 2
    finally:
        backend.close()


def test_evaluation_clone_rebuilds_active_filter_without_host_pollution():
    """V-06 在克隆历史和图上只读核验，宿主课程、H-00 和报告不变。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        runtime = _install(ctx, _SemanticMapper(), _EventCourse())
        source = _source()
        DefaultRoundRunner().run_round(
            ctx,
            _item(source),
            STAGE1_SKELETON,
            1,
        )
        semantic_run = ctx.language_semantic_course_reports[-1]
        baseline = runtime.state_key()
        report_count = len(ctx.event_time_relation_reports)

        with isolated_evaluation(ctx, label="r06b-held-out") as eval_ctx:
            report = eval_ctx.event_time_relation_runtime.process(
                document_scope(source),
                _read_only_run(semantic_run),
                read_only=True,
            )
            assert report.evidence == ()
            assert report.verifications[0].dimension_results(
                _event_protocol().verification.dimension,
            )[0].verdict == VERDICT_SUPPORT

        assert runtime.state_key() == baseline
        assert len(ctx.event_time_relation_reports) == report_count
    finally:
        backend.close()


def test_dump_load_restores_event_time_h00_h04_and_active_projection(tmp_path):
    """图和 M-03 历史恢复后，无需重放 S-02 lesson 即可核验 adopted 时间事实。"""
    backend = DictBackend()
    restored_backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        runtime = _install(ctx, _SemanticMapper(), _EventCourse())
        source = _source()
        DefaultRoundRunner().run_round(
            ctx,
            _item(source),
            STAGE1_SKELETON,
            1,
        )
        dump_run(
            backend,
            str(tmp_path),
            "r06b",
            spaces=[ctx.space_id],
        )

        restored_ctx = make_train_context(restored_backend)
        load_run(restored_backend, str(tmp_path), "r06b")
        restored = _install(
            restored_ctx,
            _SemanticMapper(),
            _EventCourse(),
        )
        result = restored.verifier.verify(
            (_RELATION,),
            scope=document_scope(source),
        )

        assert restored.evidence_count() == runtime.evidence_count() == 1
        assert result.status == EVENT_TIME_CONSISTENT
        assert len(result.fact_set.facts) == 1
    finally:
        backend.close()
        restored_backend.close()
