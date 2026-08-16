"""grounded CLARIFY 经 response-act、S-07、R-01、parser 与 G-04 的专项。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.generation_content import (
    AnswerContentProtocol,
    AnswerContentSelector,
)
from pure_integer_ai.cognition.shared.generation_plan import (
    GenerationPlanningRequest,
)
from pure_integer_ai.cognition.shared.generation_structure_plan import (
    GenerationStructureLayerProtocol,
)
from pure_integer_ai.cognition.shared.identity import (
    language_branch_identity,
    minimal_instruction_identity,
)
from pure_integer_ai.cognition.shared.question_answer import (
    EvidenceAnswerPolicy,
    EvidenceAnswerPolicyProtocol,
    QuestionRequest,
)
from pure_integer_ai.cognition.shared.representation_rendering import (
    UnicodeRepresentationRenderer,
)
from pure_integer_ai.experiments.ph2_grounded_answer_compile import (
    compile_grounded_answer_training_records,
)
from pure_integer_ai.experiments.ph2_grounded_answer_course import (
    read_grounded_answer_episodes,
)
from pure_integer_ai.experiments.ph2_grounded_answer_learning import (
    PATTERN_LITERAL,
    learn_grounded_answer_surface_model,
)
from pure_integer_ai.experiments.ph2_grounded_answer_response_act_compile import (
    GroundedResponseActCompileTarget,
    compile_grounded_response_act_patterns,
)
from pure_integer_ai.experiments.ph2_grounded_answer_response_act_choice_use import (
    GroundedResponseActLexicalAdoptionLedger,
    GroundedResponseActLexicalUseError,
)
from pure_integer_ai.experiments.ph2_grounded_answer_response_act_parser import (
    GroundedResponseActParserProtocol,
    GroundedResponseActStructureVerifier,
    GroundedResponseActTaskVerifier,
)
from pure_integer_ai.experiments.ph2_grounded_answer_response_act_runtime_factory import (
    GroundedResponseActRunLocalBuild,
    GroundedResponseActRunLocalComponents,
    GroundedResponseActRunLocalFactory,
)
from pure_integer_ai.experiments.question_answer_runtime import (
    QuestionAnswerProtocol,
)
from pure_integer_ai.experiments.verification_orchestration import (
    VERDICT_SUPPORT,
)
from pure_integer_ai.storage.backend import DictBackend

from tests.test_g01_generation_content import _T, _request
from tests.test_g02_generation_structure_plan import _plan_protocol
from tests.test_g03_generation_surface import _alias_fixture, _surface_protocol
from tests.test_g04_generation_postcheck import (
    _StaticVerifier,
    _protocol as _postcheck_protocol,
)
from tests.test_s07_structure_order import _graphs


_BASE = 20963
_SAMPLE = Path("data/ph2/grounded_answer_train_v1.jsonl.sample")


class _AliasFactory:
    """按 selected variant 建立 stance 到 learned literal 的独占 R-01 图。"""

    def __init__(self, branch) -> None:
        self.branch = branch
        self.fixture = None

    def build(self, variant):
        """只物化当前 stance/Representation，不读取 pattern id 返回文本。"""
        if self.fixture is not None:
            raise RuntimeError("response-act alias factory 不得重复 build")
        self.fixture = _alias_fixture(
            self.branch,
            ((variant.template.stance, variant.representation),),
        )
        return self.fixture.runtime


def test_clarify_runs_as_learned_response_act_with_selected_candidates():
    """CLARIFY 保留歧义候选，但实际生成零命题并由 G-04 恢复任务。"""
    episodes = read_grounded_answer_episodes(_SAMPLE)
    clarify_question = next(
        item.question for item in episodes
        if item.question.answer_plan.response_act == "CLARIFY")
    model, _report = learn_grounded_answer_surface_model(
        compile_grounded_answer_training_records(_SAMPLE))
    branch = language_branch_identity((_BASE, 1))
    raw_planning, _source, _scope, _bound = _request(states=(_T, _T))
    planning = GenerationPlanningRequest(
        replace(raw_planning.goal, target_branch=branch),
        raw_planning.candidates,
    )
    content_protocol = AnswerContentProtocol(*tuple(
        minimal_instruction_identity((_BASE, 2, index))
        for index in range(1, 6)
    ))
    policy_protocol = EvidenceAnswerPolicyProtocol(*tuple(
        minimal_instruction_identity((_BASE, 3, index))
        for index in range(1, 5)
    ))
    selector = AnswerContentSelector(
        content_protocol,
        EvidenceAnswerPolicy(content_protocol, policy_protocol),
    )
    selection = selector.select(planning)
    assert selection.stance == content_protocol.clarify
    assert len(selection.selected_candidate_keys) == 2

    target = GroundedResponseActCompileTarget(
        "CLARIFY",
        content_protocol.clarify,
        branch,
        (_BASE, 4),
    )
    compilation = compile_grounded_response_act_patterns(model, target)
    selected_pattern = compilation.variants[0]
    learned_literal = next(
        pattern.parts[0].literal for pattern in model.patterns
        if (pattern.pattern_id == selected_pattern.pattern_id
            and pattern.parts[0].kind == PATTERN_LITERAL)
    )
    backend = DictBackend()
    alias_factory = _AliasFactory(branch)
    try:
        graphs = _graphs(backend)
        renderer_identity = minimal_instruction_identity((_BASE, 5, 1))
        renderer = UnicodeRepresentationRenderer(
            target.representation_family, renderer_identity)
        components = GroundedResponseActRunLocalComponents(
            selector,
            _plan_protocol(_BASE + 10),
            GenerationStructureLayerProtocol(*tuple(
                minimal_instruction_identity((_BASE, 6, index))
                for index in range(1, 4)
            )),
            _surface_protocol(_BASE + 20),
            alias_factory,
            renderer,
            renderer_identity,
            _postcheck_protocol(),
            GroundedResponseActStructureVerifier(
                minimal_instruction_identity((_BASE, 7, 1)),
                minimal_instruction_identity((_BASE, 7, 2)),
            ),
            _StaticVerifier(VERDICT_SUPPORT, 1),
            GroundedResponseActTaskVerifier(
                minimal_instruction_identity((_BASE, 8, 1)),
                minimal_instruction_identity((_BASE, 8, 2)),
            ),
            QuestionAnswerProtocol(*tuple(
                minimal_instruction_identity((_BASE, 9, index))
                for index in range(1, 4)
            )),
        )
        query_kind = minimal_instruction_identity((_BASE, 10, 1))
        route = minimal_instruction_identity((_BASE, 10, 2))
        factory = GroundedResponseActRunLocalFactory(
            graphs.lifecycle, components)
        installation = factory.build(GroundedResponseActRunLocalBuild(
            model,
            clarify_question,
            target,
            planning,
            selected_pattern.pattern_id,
            GroundedResponseActParserProtocol(*tuple(
                minimal_instruction_identity((_BASE, 11, index))
                for index in range(1, 4)
            )),
            query_kind,
            route,
            minimal_instruction_identity((_BASE, 12, 1)),
            (_BASE, 12, 2),
        ))
        question_request = QuestionRequest(
            query_kind,
            minimal_instruction_identity((_BASE, 13, 1)),
            planning.goal.goal_kind,
            planning.goal.proposition,
            planning.goal.required,
            planning.goal.scope,
            planning.goal.scope,
            (_BASE, 13, 2),
            branch,
            tuple(item.proposition for item in planning.candidates),
        )

        run = installation.runtime.run(question_request)
        ledger = GroundedResponseActLexicalAdoptionLedger(installation)
        lexical_use = ledger.adopt(run)

        assert run.complete
        assert run.selection is not None
        assert len(run.selection.selected_candidate_keys) == 2
        assert run.generation is not None and run.generation.complete
        structure = run.generation.surface.preview.request.structure
        assert structure.selection.selected_candidate_keys == (
            run.selection.selected_candidate_keys)
        assert {item.candidate_key for item in structure.propositions.propositions} == (
            set(run.selection.selected_candidate_keys))
        assert len(structure.syntax.sentences) == 1
        assert structure.syntax.sentences[0].proposition_keys == ()
        assert structure.syntax.suppressed_candidate_keys == (
            run.selection.selected_candidate_keys)
        assert structure.syntax.sentences[0].values[0].filler == (
            content_protocol.clarify)
        execution = run.generation.surface.preview.request.execution
        assert execution.complete
        assert execution.sentences[0].active_constraints == ()
        assert renderer.text(run.generation.rendered) == learned_literal
        assert len(run.generation.surface.adoptions) == 1
        assert lexical_use.choice_before == installation.lexical_choice
        assert lexical_use.adoptions == run.generation.surface.adoptions
        assert lexical_use.choice_after.exact_uses == (lexical_use.use,)
        assert lexical_use.use.scope == planning.goal.scope
        assert ledger.records == (lexical_use,)
        with pytest.raises(
                GroundedResponseActLexicalUseError,
                match="不得重复登记"):
            ledger.adopt(run)
        assert run.postcheck is not None and run.postcheck.complete
        observation = run.postcheck.parsed.observation
        assert observation is not None
        assert observation.stance == content_protocol.clarify
        assert observation.propositions == ()
        assert observation.representations == run.generation.representations
        assert observation.task_observations[0].result_key == (
            selected_pattern.task_result_key)
        applicable = run.postcheck.report.applicable_results()
        assert all(item.verdict == VERDICT_SUPPORT for item in applicable)
    finally:
        if alias_factory.fixture is not None:
            alias_factory.fixture.close()
        backend.close()
