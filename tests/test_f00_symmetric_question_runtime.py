"""F-00 对 R-05 aggregate Evidence 的来源守恒关系问答测试。"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.identity import minimal_instruction_identity
from pure_integer_ai.cognition.shared.scope_identity import (
    document_scope,
    query_scope,
)
from pure_integer_ai.cognition.shared.semantic_object import semantic_source
from pure_integer_ai.cognition.shared.typed_binding import (
    BindingEnvironment,
    BindingFailureProtocol,
    PropositionSubstituter,
    PropositionTemplateGraph,
    ScopedPropositionTemplate,
    SubstitutionProtocol,
)
from pure_integer_ai.experiments.symmetric_question_runtime import (
    SymmetricPairQuestionExecutor,
)
from tests.test_f00_question_answer_runtime import (
    _fixture as _question_fixture,
    _rendered_text,
)
from tests.test_r05_semantic_pair_runtime import (
    _fixture as _relation_fixture,
    _source as _relation_source,
)


_BASE = 19400


def _bound(definition, structure):
    """把 R-05 原子命题定义转换为无变量的运行期 BoundProposition。"""
    failures = BindingFailureProtocol(*tuple(
        minimal_instruction_identity((_BASE + 1, index))
        for index in range(1, 10)
    ))
    graph = PropositionTemplateGraph((
        ScopedPropositionTemplate(definition, structure),
    ))
    return PropositionSubstituter(SubstitutionProtocol(
        minimal_instruction_identity((_BASE + 2, 1)),
        failures,
    )).substitute(
        definition.proposition,
        graph,
        BindingEnvironment(),
    )


def _materialized_bound(relation, proposition):
    """从 R-05 权威语义图恢复直接关系命题并建立 bound view。"""
    owner = relation.runtime.similar
    proposition_ref = owner.semantic_graph.ontology.resolve(proposition)
    assert proposition_ref is not None
    definition = owner.semantic_graph.read_atomic(proposition_ref).definition
    return _bound(definition, owner.protocol.schema.schema)


def _executor_factory(owner):
    """返回把同一 R-05 owner 绑定到 F-00 route 的 executor factory。"""
    def build(route):
        """为当前 QuestionAnswerRuntime route 建立只读关系 executor。"""
        return SymmetricPairQuestionExecutor(
            owner,
            route=route,
            executed_reason=minimal_instruction_identity((_BASE + 3, 1)),
        )

    return build


def test_f00_symmetric_relation_preserves_aggregate_evidence_and_generates():
    """关系答案保留 aggregate Hypothesis 与 recognition 来源并走真实生成链。"""
    relation = _relation_fixture()
    question = None
    try:
        proposition = relation.add("similar", 0, 1, stance="support")
        target = _materialized_bound(relation, proposition)
        knowledge_source = semantic_source(target.template)
        response_scope = query_scope(
            1, parent=document_scope(knowledge_source))
        question = _question_fixture(
            world=(knowledge_source, response_scope, target),
            executor_factory=_executor_factory(relation.runtime.similar),
        )
        before = relation.runtime.similar.state_key()

        run = question.runtime.run(question.request)

        assert run.complete
        assert run.status == question.content.answer
        candidate = run.query_result.candidates[0]
        assert candidate.proposition == target
        assert candidate.evidence
        assert candidate.evidence[0].hypothesis.observation != knowledge_source
        assert _rendered_text(question, run) == "事实"
        assert relation.runtime.similar.state_key() == before
    finally:
        if question is not None:
            question.close()
        relation.close()


def test_f00_symmetric_reverse_fact_without_derived_evidence_stays_unknown():
    """只有反向直接事实而无当前命题派生 Evidence 时不得伪造关系答案。"""
    relation = _relation_fixture()
    question = None
    try:
        relation.add("similar", 0, 1, stance="support")
        source = _relation_source(9901)
        reverse = relation.definition(
            "similar",
            relation.objects[1],
            relation.objects[0],
            source,
            9901,
        )
        target = _bound(
            reverse, relation.similar_protocol.schema.schema)
        response_scope = query_scope(1, parent=document_scope(source))
        question = _question_fixture(
            world=(source, response_scope, target),
            executor_factory=_executor_factory(relation.runtime.similar),
        )
        before = relation.runtime.similar.state_key()

        run = question.runtime.run(question.request)

        assert run.complete
        assert run.status == question.content.unknown
        assert run.query_result.candidates == ()
        assert _rendered_text(question, run) == "未知"
        assert relation.runtime.similar.state_key() == before
    finally:
        if question is not None:
            question.close()
        relation.close()
