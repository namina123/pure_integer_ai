"""F-00 对 R-08 有限逻辑派生结果的真实问答与生成纵切测试。"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.identity import (
    minimal_instruction_identity,
)
from pure_integer_ai.cognition.shared.logic_executor import LogicEvidenceState
from pure_integer_ai.cognition.shared.scope_identity import (
    document_scope,
    query_scope,
)
from pure_integer_ai.experiments.logic_question_runtime import (
    LogicClosureQuestionExecutor,
)
from tests.test_f00_question_answer_runtime import (
    _fixture as _question_fixture,
    _rendered_text,
)
from tests.test_r08_logic_closure import _production_fixture


_BASE = 19500


def _executor_factory(runtime):
    """返回把真实 R-08 owner 绑定到当前 F-00 route 的 executor factory。"""
    def build(route):
        """为当前 QuestionAnswerRuntime route 建立只读有限逻辑 executor。"""
        return LogicClosureQuestionExecutor(
            runtime,
            route=route,
            executed_reason=minimal_instruction_identity((_BASE + 1, 1)),
        )

    return build


def test_f00_logic_derivation_preserves_adoption_and_generates_without_write(
        monkeypatch,
        ):
    """真实激活 NOT 后只读派生 refute，并以同次候选完成 G-00 至 G-03。"""
    ctx, world, logic_runtime = _production_fixture()
    question = None
    try:
        document = document_scope(world.source)
        activation = logic_runtime.process(document, read_only=False)
        assert activation.executions[0].execution.adoptions

        reports = []
        original_process = logic_runtime.process

        def observed_process(scope, *, read_only):
            """记录 F-00 实际消费的只读 R-08 报告，不替换领域结果。"""
            report = original_process(scope, read_only=read_only)
            reports.append(report)
            return report

        monkeypatch.setattr(logic_runtime, "process", observed_process)
        response_scope = query_scope(1, parent=document)
        question = _question_fixture(
            world=(world.source, response_scope, world.root),
            executor_factory=_executor_factory(logic_runtime),
            required=LogicEvidenceState(False, True),
            answer_text="逻辑结论",
        )
        before_owner = logic_runtime.state_key()
        before_backend = ctx.backend.snapshot()

        run = question.runtime.run(question.request)

        assert len(reports) == 1
        report = reports[0]
        assert report.read_only
        assert report.scope == response_scope
        assert report.formations == ()
        assert report.recognitions == ()
        assert len(report.executions) == 1
        bundle = report.executions[0]
        assert bundle.execution.evaluation.derivation
        assert bundle.execution.adoptions
        assert bundle.candidate.proposition == world.root
        assert bundle.candidate.source == world.source
        assert bundle.candidate.scope == response_scope
        assert bundle.candidate.evidence == bundle.evidence

        assert run.complete
        assert run.status == question.content.answer
        assert run.query_result.candidates[0] is bundle.candidate
        assert run.planning_request.candidates[0] is bundle.candidate
        assert run.selection.selected_candidate_keys
        assert _rendered_text(question, run) == "逻辑结论"
        assert logic_runtime.state_key() == before_owner
        assert ctx.backend.snapshot() == before_backend
    finally:
        if question is not None:
            question.close()
        ctx.backend.close()
