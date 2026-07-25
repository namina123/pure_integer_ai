"""F-00 answer 与 unknown response-act 的同次 G-04 复核测试。"""
from __future__ import annotations

import pytest

from pure_integer_ai.cognition.shared.hypothesis import EVIDENCE_SUPPORT
from pure_integer_ai.experiments.generation_verification_runtime import (
    GenerationPostcheckRuntime,
)
from pure_integer_ai.experiments.question_answer_runtime import (
    EvidenceQuestionPostcheckMapper,
)
from pure_integer_ai.experiments.verification_orchestration import (
    VERDICT_SUPPORT,
)
from tests.test_f00_question_answer_runtime import (
    _fixture as _question_fixture,
    _rendered_text,
)
from tests.test_g04_generation_postcheck import (
    _ExecutionParser,
    _StaticVerifier,
    _protocol,
)


_BASE = 19700


class _RecordingPostcheckMapper:
    """让受限 parser 只按实际输出键读取预登记观察，再委托生产 mapper。"""

    def __init__(self, parser) -> None:
        """绑定独立 parser 和项目级证据问答 postcheck mapper。"""
        self.parser = parser
        self.delegate = EvidenceQuestionPostcheckMapper(
            (_BASE + 1, 1),
            citation_required=True,
            trust_required=True,
        )

    def build(self, request, query, result, generation):
        """登记实际 renderer 输出的 typed 观察，并建立同次 G-04 请求。"""
        selected_keys = set(
            generation.surface.preview.request.structure
            .selection.selected_candidate_keys)
        cited_sources = tuple({
            source
            for candidate in generation.plan.request.candidates
            if candidate.stable_key() in selected_keys
            for source in candidate.citation_sources
        })
        self.parser.record(generation, cited_sources=cited_sources)
        return self.delegate.build(request, query, result, generation)


def _postcheck_owners():
    """装配六维 G-04 runtime，并返回可检查的独立 parser/verifier。"""
    parser = _ExecutionParser()
    structure = _StaticVerifier(VERDICT_SUPPORT, 1)
    source = _StaticVerifier(VERDICT_SUPPORT, 2)
    runtime = GenerationPostcheckRuntime(
        _protocol(),
        parser,
        structure,
        source,
    )
    return _RecordingPostcheckMapper(parser), runtime, parser, structure, source


@pytest.mark.parametrize(
    ("stances", "expected_text", "source_calls"),
    (
        ((EVIDENCE_SUPPORT,), "事实", 1),
        ((), "未知", 0),
    ),
)
def test_f00_generation_runs_same_execution_through_g04(
        stances,
        expected_text,
        source_calls,
        ):
    """有答案和 unknown 都复核同次 surface，空命题不伪造来源要求。"""
    mapper, postchecker, parser, structure, source = _postcheck_owners()
    fixture = _question_fixture(
        *stances,
        postcheck_mapper=mapper,
        postchecker=postchecker,
    )
    try:
        before_ledger = fixture.ledger.state_key()

        run = fixture.runtime.run(fixture.request)

        assert run.generation is not None
        assert run.postcheck is not None
        assert run.postcheck.request.execution is run.generation
        assert run.postcheck.report.read_only
        assert run.postcheck.complete
        assert run.complete
        assert _rendered_text(fixture, run) == expected_text
        assert parser.calls == 1
        assert structure.calls == 1
        assert source.calls == source_calls
        assert fixture.ledger.state_key() == before_ledger
        if stances:
            assert run.postcheck.request.source_requirements
            selected = {
                item.stable_key(): item
                for item in run.planning_request.candidates
            }
            for requirement in run.postcheck.request.source_requirements:
                assert requirement.evidence_sources == (
                    selected[requirement.candidate_key].citation_sources)
        else:
            assert run.postcheck.request.source_requirements == ()
            assert run.selection.selected_candidate_keys == ()
    finally:
        fixture.close()
