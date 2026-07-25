"""F-00 经 L-03 指代 occurrence 和 L-05B2B 恢复目标的纵切测试。"""
from __future__ import annotations

from dataclasses import dataclass, replace

from pure_integer_ai.cognition.shared.identity import (
    concept_identity,
    language_branch_identity,
    minimal_instruction_identity,
)
from pure_integer_ai.cognition.shared.logic_executor import LogicEvidenceState
from pure_integer_ai.cognition.shared.scope_identity import (
    query_scope,
    session_scope,
)
from pure_integer_ai.cognition.understanding.refers_occurrence import (
    resolve_pronoun_occurrence,
)
from pure_integer_ai.cognition.understanding.occurrence_reference import (
    OccurrenceReferenceProtocol,
    OccurrenceReferenceRuntime,
)
from pure_integer_ai.experiments.language_semantic_course import (
    LanguageSemanticCourseInput,
)
from pure_integer_ai.experiments.language_semantic_query import (
    LanguageSemanticQueryDecision,
    LanguageSemanticQueryProtocol,
)
from pure_integer_ai.experiments.reference_question_runtime import (
    OccurrenceReferenceSemanticResolver,
    SemanticReferenceQuestionExecutor,
)
from tests.test_f00_question_answer_runtime import (
    _fixture as _question_fixture,
    _rendered_text,
)
from tests.test_l05b2b_semantic_course_runtime import (
    _fixture as _semantic_fixture,
    _lesson,
)


_BASE = 19600


class _ReferenceQueryMapper:
    """只在 antecedent anchor 恢复出唯一 active ground 候选时建立请求。"""

    def __init__(self, branch):
        """绑定调用方注入的目标语言分支。"""
        self.branch = branch

    def map(self, input_value):
        """选择唯一 ground 候选，不读取词面、expected 或容器排序首项。"""
        if len(input_value.candidates) != 1:
            return LanguageSemanticQueryDecision(
                minimal_instruction_identity((_BASE + 1, 1)),
                (_BASE + 1, 2),
            )
        candidate = input_value.candidates[0]
        if not candidate.ground:
            return LanguageSemanticQueryDecision(
                minimal_instruction_identity((_BASE + 1, 3)),
                (_BASE + 1, 4),
            )
        return LanguageSemanticQueryDecision(
            minimal_instruction_identity((_BASE + 1, 5)),
            (_BASE + 1, 6),
            candidate.hypothesis,
            (candidate.hypothesis,),
            minimal_instruction_identity((_BASE + 1, 7)),
            LogicEvidenceState(True, False),
            target_branch=self.branch,
        )

    def clone_for_evaluation(self):
        """返回同配置无状态 mapper，避免评测共享调用历史。"""
        return _ReferenceQueryMapper(self.branch)

    def state_key(self):
        """返回固定协议版本键，不包含 Python 对象身份。"""
        return _BASE + 1, 8, *self.branch.stable_key()


@dataclass
class _ReferenceFixture:
    """保存指代解析、语义恢复和 F-00 纵切所需 owner。"""

    backend: object
    ctx: object
    semantic_runtime: object
    current: LanguageSemanticCourseInput
    antecedent: object
    target: object
    branch: object
    candidate_occurrences: tuple
    reference_runtime: OccurrenceReferenceRuntime

    def close(self):
        """按 A-09 逆序关闭活动边界，再关闭测试后端。"""
        self.ctx.work_memory.end_episode()
        self.ctx.work_memory.end_document()
        self.ctx.work_memory.end_session()
        self.backend.close()


def _reference_fixture(*, ambiguous: bool = False) -> _ReferenceFixture:
    """构造真实 legacy 指代选择及一个或两个同 ref 前文 occurrence。"""
    branch = language_branch_identity((_BASE + 3, 1))
    protocol = LanguageSemanticQueryProtocol(_ReferenceQueryMapper(branch))
    (backend, ctx, semantic_runtime, course_mapper,
     item, payload, observed) = _semantic_fixture(
        protocol,
        raw_text="甲甲它" if ambiguous else "甲它",
    )
    source = payload.source_ref
    occurrence_scope = payload.occurrence_scope_identity
    work_memory = ctx.work_memory
    work_memory.begin_session(session_scope(
        _BASE + 20,
        owner=source.owner,
        versions=source.versions,
        source=source,
    ))
    work_memory.begin_document(occurrence_scope)
    work_memory.begin_episode(payload.scope_identity)
    antecedent = observed.occurrence_refs[0]
    legacy = ctx.concept_index.ensure("甲", space_id=ctx.space_id)
    ctx.occurrence_index.record(
        source=source,
        raw_text="甲甲它" if ambiguous else "甲它",
        scope=occurrence_scope,
        start=0,
        end=1,
        ordinal=0,
        segment_index=0,
        local_index=0,
        document_index=0,
        legacy_candidates=(legacy,),
    )
    course_mapper.decision = replace(
        course_mapper.decision,
        lesson=_lesson(source, antecedent),
    )
    training = semantic_runtime.process(
        ctx,
        item,
        payload,
        observed,
    )
    assert training.request is not None

    candidates = [antecedent]
    if ambiguous:
        duplicate = ctx.occurrence_index.record(
            source=source,
            raw_text="甲甲它",
            scope=occurrence_scope,
            start=1,
            end=2,
            ordinal=0,
            segment_index=1,
            local_index=0,
            document_index=1,
            legacy_candidates=(legacy,),
        ).occurrence
        candidates.append(duplicate)
    ctx.work_memory.push_segment(0, [legacy])
    selected = resolve_pronoun_occurrence(
        ctx.edge_store,
        ctx.concept_index,
        "它",
        work_memory=ctx.work_memory,
        memory_space_id=ctx.space_id,
        timestamp_seq=1,
    )
    assert selected == legacy
    reference_index = 2 if ambiguous else 1
    reference = ctx.occurrence_index.record(
        source=source,
        raw_text="甲甲它" if ambiguous else "甲它",
        scope=occurrence_scope,
        start=reference_index,
        end=reference_index + 1,
        ordinal=0,
        segment_index=2,
        local_index=0,
        document_index=reference_index,
        legacy_candidates=(selected,),
    ).occurrence
    current = LanguageSemanticCourseInput(
        source,
        occurrence_scope,
        payload.scope_identity,
        (reference,),
        (),
        (),
        True,
    )
    reference_runtime = OccurrenceReferenceRuntime(
        ctx.occurrence_index,
        work_memory,
        OccurrenceReferenceProtocol(
            (_BASE + 21, 1),
            concept_identity((_BASE + 21, 2)),
            minimal_instruction_identity((_BASE + 21, 3)),
            16,
            16,
            16,
            8,
        ),
    )
    return _ReferenceFixture(
        backend,
        ctx,
        semantic_runtime,
        current,
        antecedent,
        training.request.goal.proposition,
        branch,
        tuple(candidates),
        reference_runtime,
    )


def _executor_factory(resolver, current):
    """返回把同一 reference resolver 绑定到 F-00 route 的 factory。"""
    def build(route):
        """为当前 QuestionAnswerRuntime 建立 typed 指代 executor。"""
        return SemanticReferenceQuestionExecutor(
            resolver,
            current,
            route=route,
            executed_reason=minimal_instruction_identity((_BASE + 2, 1)),
        )

    return build


def test_f00_reference_resolves_typed_antecedent_and_generates_without_write(
        monkeypatch,
        ):
    """真实指代 ref 经 typed antecedent 恢复 Proposition，并完成 F-00 生成。"""
    world = _reference_fixture()
    question = None
    try:
        resolver = OccurrenceReferenceSemanticResolver(
            world.ctx.occurrence_index,
            world.semantic_runtime.query_runtime,
            world.reference_runtime,
            world.candidate_occurrences,
            timestamp_seq=1,
        )
        resolutions = []
        original_resolve = resolver.resolve

        def observed_resolve(current):
            """记录 F-00 同次使用的指代恢复结果，不改变 owner 行为。"""
            resolution = original_resolve(current)
            resolutions.append(resolution)
            return resolution

        monkeypatch.setattr(resolver, "resolve", observed_resolve)
        response_scope = query_scope(
            1,
            parent=world.current.runtime_scope,
        )
        question = _question_fixture(
            world=(world.current.source, response_scope, world.target),
            executor_factory=_executor_factory(resolver, world.current),
            answer_text="指代结论",
            target_branch=world.branch,
        )
        before_backend = world.backend.snapshot()
        before_semantic = world.semantic_runtime.state_key()
        before_reference = world.reference_runtime.state_key()

        run = question.runtime.run(question.request)

        assert len(resolutions) == 1
        resolution = resolutions[0]
        assert resolution.reference_occurrence == world.current.occurrences[0]
        assert resolution.antecedent_occurrence == world.antecedent
        assert resolution.semantic is not None
        assert resolution.semantic.request is not None
        assert resolution.semantic.request.goal.proposition == world.target
        assert run.complete
        assert run.status == question.content.answer
        assert run.query_result.candidates[0].proposition == world.target
        assert run.query_result.candidates[0].evidence
        assert _rendered_text(question, run) == "指代结论"
        assert world.semantic_runtime.state_key() == before_semantic
        assert world.reference_runtime.state_key() == before_reference
        assert world.backend.snapshot() == before_backend
    finally:
        if question is not None:
            question.close()
        world.close()


def test_f00_reference_multiple_antecedent_occurrences_stays_unknown():
    """同一 legacy ref 命中多个前文 occurrence 时不得按稳定序私选命题。"""
    world = _reference_fixture(ambiguous=True)
    question = None
    try:
        resolver = OccurrenceReferenceSemanticResolver(
            world.ctx.occurrence_index,
            world.semantic_runtime.query_runtime,
            world.reference_runtime,
            world.candidate_occurrences,
            timestamp_seq=1,
        )
        response_scope = query_scope(
            1,
            parent=world.current.runtime_scope,
        )
        question = _question_fixture(
            world=(world.current.source, response_scope, world.target),
            executor_factory=_executor_factory(resolver, world.current),
            target_branch=world.branch,
        )
        before_backend = world.backend.snapshot()
        before_semantic = world.semantic_runtime.state_key()
        before_reference = world.reference_runtime.state_key()

        run = question.runtime.run(question.request)

        assert run.complete
        assert run.status == question.content.unknown
        assert run.query_result.candidates == ()
        assert _rendered_text(question, run) == "未知"
        assert world.semantic_runtime.state_key() == before_semantic
        assert world.reference_runtime.state_key() == before_reference
        assert world.backend.snapshot() == before_backend
    finally:
        if question is not None:
            question.close()
        world.close()
