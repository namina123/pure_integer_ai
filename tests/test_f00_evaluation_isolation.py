"""F-00 在 V-06 clone 内执行有限逻辑与 G-04 的宿主零写测试。"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.logic_executor import LogicEvidenceState
from pure_integer_ai.cognition.shared.scope_identity import (
    document_scope,
    query_scope,
)
from pure_integer_ai.experiments.evaluation_isolation import isolated_evaluation
from pure_integer_ai.experiments.evaluation_protocol import (
    ProbeOutcome,
    evaluate_probe,
)
from tests.test_f00_generation_postcheck import _postcheck_owners
from tests.test_f00_logic_question_runtime import _executor_factory
from tests.test_f00_question_answer_runtime import (
    _fixture as _question_fixture,
    _rendered_text,
)
from tests.test_r08_logic_closure import _production_fixture
from tests.test_v00_evaluation_protocol import _complete_plan


def test_f00_logic_and_g04_run_in_v06_clone_without_host_write():
    """V-00 探针在 V-06 clone 内执行 F-00/G-04，宿主状态必须不变。"""
    ctx, world, host_runtime = _production_fixture()
    question = None
    try:
        document = document_scope(world.source)
        host_runtime.process(document, read_only=False)
        host_backend = ctx.backend.snapshot()
        host_owner = host_runtime.state_key()
        plan, _items = _complete_plan(full_coverage=False)
        assignment = plan.assignments[2]
        dimension = plan.protocol.required_dimensions[0]

        def host_state():
            """返回 V-00 必须保持不变的宿主持久态、owner 和生命周期。"""
            memory = ctx.work_memory
            return (
                ctx.backend.snapshot(),
                host_runtime.state_key(),
                tuple(ctx.logic_closure_reports),
                memory.round_id,
                tuple(memory.produced_refs),
                tuple(memory.prior_topic_refs),
                memory.active_session_scope,
                memory.active_document_scope,
                memory.active_episode_scope,
                memory.active_query_scope,
                memory.active_generation_scope,
            )

        def run_f00_probe() -> ProbeOutcome:
            """在隔离 clone 内执行真实 R-08、F-00 和 G-04 纵切。"""
            nonlocal question
            with isolated_evaluation(ctx, label="f00-logic-g04") as eval_ctx:
                eval_runtime = eval_ctx.logic_closure_runtime
                assert eval_runtime is not None
                response_scope = query_scope(1, parent=document)
                mapper, postchecker, _, _, _ = _postcheck_owners()
                question = _question_fixture(
                    world=(world.source, response_scope, world.root),
                    executor_factory=_executor_factory(eval_runtime),
                    required=LogicEvidenceState(False, True),
                    answer_text="隔离逻辑结论",
                    postcheck_mapper=mapper,
                    postchecker=postchecker,
                )
                eval_backend = eval_ctx.backend.snapshot()
                eval_owner = eval_runtime.state_key()

                run = question.runtime.run(question.request)

                assert run.complete
                assert run.postcheck is not None
                assert run.postcheck.complete
                assert _rendered_text(question, run) == "隔离逻辑结论"
                assert eval_runtime.state_key() == eval_owner
                assert eval_ctx.backend.snapshot() == eval_backend
                return ProbeOutcome(True, value=len(run.trace))

        observation = evaluate_probe(
            plan,
            assignment,
            dimension,
            run_f00_probe,
            state_reader=host_state,
        )

        assert observation.identity == assignment.identity
        assert observation.outcome.passed is True
        assert observation.evidence == plan.protocol.statistical_evidence
        assert host_runtime.state_key() == host_owner
        assert ctx.backend.snapshot() == host_backend
    finally:
        if question is not None:
            question.close()
        ctx.backend.close()
