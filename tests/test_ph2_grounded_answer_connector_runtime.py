"""grounded-answer pattern 经真实 connector、executor 与 G-04 的专项。"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.generation_content import (
    GenerationContentLayerResolver,
    GenerationStanceLayerResolver,
)
from pure_integer_ai.cognition.shared.generation_execution import (
    TypedGenerationExecutor,
)
from pure_integer_ai.cognition.shared.generation_plan import (
    GenerationLayerRegistration,
    GenerationPlanner,
)
from pure_integer_ai.cognition.shared.generation_structure_plan import (
    GenerationDiscourseLayerResolver,
    GenerationPropositionLayerResolver,
    GenerationStructureLayerProtocol,
    GenerationSyntaxLayerResolver,
)
from pure_integer_ai.cognition.shared.question_answer import (
    QuestionExecutionResult,
    QuestionRequest,
)
from pure_integer_ai.cognition.shared.identity import (
    minimal_instruction_identity,
)
from pure_integer_ai.cognition.shared.representation_rendering import (
    UnicodeRepresentationRenderer,
)
from pure_integer_ai.experiments.generation_surface_runtime import (
    GenerationSurfaceLayerResolver,
    GenerationSurfaceRuntime,
)
from pure_integer_ai.experiments.generation_verification_runtime import (
    GenerationPostcheckRuntime,
)
from pure_integer_ai.experiments.question_answer_runtime import (
    EvidenceQuestionPostcheckMapper,
    QuestionAnswerProtocol,
    QuestionAnswerRuntime,
    QuestionRouteRegistration,
)
from pure_integer_ai.experiments.ph2_grounded_answer_connector import (
    GroundedAnswerConnectorTarget,
    build_grounded_answer_connector,
    compile_grounded_answer_connectors,
)
from pure_integer_ai.experiments.ph2_grounded_answer_parser import (
    GroundedAnswerParserProtocol,
    GroundedAnswerSurfaceParser,
    build_grounded_answer_parser_catalog,
)
from pure_integer_ai.experiments.ph2_grounded_answer_order import (
    install_grounded_answer_order_course,
)
from pure_integer_ai.experiments.verification_orchestration import (
    VERDICT_SUPPORT,
)
from pure_integer_ai.storage.backend import DictBackend

from tests.test_g02_generation_structure_plan import (
    _plan_protocol,
    _selection,
)
from tests.test_g03_generation_surface import (
    _alias_fixture,
    _surface_protocol,
)
from tests.test_g04_generation_postcheck import (
    _StaticVerifier,
    _protocol as _postcheck_protocol,
)
from tests.test_ph2_grounded_answer_course import (
    _connector_question_and_candidate,
)
from tests.test_s07_structure_order import (
    _graphs,
)


_BASE = 20930


class _QuestionExecutor:
    """把已构造的 typed 候选作为查询结果返回，不读取 surface label。"""

    def __init__(self, route, candidate) -> None:
        self.route = route
        self.candidate = candidate

    def execute(self, query):
        """保持同一目标、response scope 和真实 Evidence 候选。"""
        if query.route != self.route:
            raise ValueError("grounded question route 漂移")
        return QuestionExecutionResult(
            query,
            minimal_instruction_identity((_BASE + 20, 1)),
            (self.candidate,),
            (_BASE + 20, 2),
        )


def _execution_planner(backend, variant):
    """把编译得到的相邻 part 顺序提升为真实 active S-07 约束。"""
    graphs = _graphs(backend)
    installation = install_grounded_answer_order_course(
        variant, graphs.lifecycle)
    assert len(installation.promotions) == len(variant.order_requirements)
    assert installation.evidence_count == (
        len(variant.order_requirements)
        * len(variant.option.support_teacher_keys)
    )
    return installation.execution_planner


def test_grounded_variant_runs_through_typed_executor_parser_and_g04():
    """已学 literal/claim 槽必须由真实 G-02/G-03 输出并经 G-04 恢复。"""
    model, question, planning, candidate, branch = (
        _connector_question_and_candidate())
    surface_protocol = _surface_protocol(_BASE + 10)
    family = (_BASE + 11, 1)
    compilation = compile_grounded_answer_connectors(
        model,
        question,
        GroundedAnswerConnectorTarget(
            candidate.proposition, branch, family),
        surface_protocol,
    )
    selected_pattern = next(
        pattern for pattern in model.patterns
        if any(part.literal == "档案显示，" for part in pattern.parts)
    )
    selected = compilation.select(selected_pattern.pattern_id)
    variant, connector = build_grounded_answer_connector(
        compilation, selected.option.pattern_id, surface_protocol)
    backend = DictBackend()
    alias = None
    try:
        selection, selector, _content_protocol = _selection(planning)
        execution_planner = _execution_planner(backend, variant)
        structure_planner = connector.structure_planner()
        surface_builder = connector.surface_request_builder(execution_planner)
        alias = _alias_fixture(
            branch,
            tuple((item.filler, item.representation)
                  for item in variant.aliases),
        )
        surface_runtime = GenerationSurfaceRuntime(alias.runtime)
        plan_protocol = _plan_protocol(_BASE + 12)
        structure_protocol = GenerationStructureLayerProtocol(*tuple(
            minimal_instruction_identity((_BASE + 13, index))
            for index in range(1, 4)
        ))
        registrations = (
            GenerationLayerRegistration(
                plan_protocol.stance_layer,
                GenerationStanceLayerResolver(plan_protocol, selector),
            ),
            GenerationLayerRegistration(
                plan_protocol.content_layer,
                GenerationContentLayerResolver(plan_protocol, selector),
            ),
            GenerationLayerRegistration(
                plan_protocol.discourse_layer,
                GenerationDiscourseLayerResolver(
                    plan_protocol,
                    structure_protocol,
                    selector,
                    structure_planner,
                ),
            ),
            GenerationLayerRegistration(
                plan_protocol.proposition_layer,
                GenerationPropositionLayerResolver(
                    plan_protocol,
                    structure_protocol,
                    selector,
                    structure_planner,
                ),
            ),
            GenerationLayerRegistration(
                plan_protocol.syntax_layer,
                GenerationSyntaxLayerResolver(
                    plan_protocol,
                    structure_protocol,
                    selector,
                    structure_planner,
                ),
            ),
            GenerationLayerRegistration(
                plan_protocol.surface_layer,
                GenerationSurfaceLayerResolver(
                    plan_protocol,
                    selector,
                    structure_planner,
                    surface_builder,
                    surface_runtime,
                    commit=False,
                ),
            ),
        )
        renderer_identity = minimal_instruction_identity((_BASE + 14, 1))
        renderer = UnicodeRepresentationRenderer(family, renderer_identity)
        executor = TypedGenerationExecutor(
            GenerationPlanner(plan_protocol, registrations),
            renderer,
            surface_runtime,
        )

        catalog = build_grounded_answer_parser_catalog(
            compilation, candidate, renderer_identity)
        parser_protocol = GroundedAnswerParserProtocol(
            *tuple(minimal_instruction_identity((_BASE + 15, index))
                   for index in range(1, 6)),
            selection.stance,
        )
        parser = GroundedAnswerSurfaceParser(parser_protocol, catalog)
        postchecker = GenerationPostcheckRuntime(
            _postcheck_protocol(),
            parser,
            _StaticVerifier(VERDICT_SUPPORT, 1),
            _StaticVerifier(VERDICT_SUPPORT, 2),
        )
        query_kind = minimal_instruction_identity((_BASE + 16, 1))
        route = minimal_instruction_identity((_BASE + 16, 2))
        request = QuestionRequest(
            query_kind,
            minimal_instruction_identity((_BASE + 16, 3)),
            planning.goal.goal_kind,
            planning.goal.proposition,
            planning.goal.required,
            candidate.scope,
            candidate.scope,
            (_BASE + 16, 4),
            branch,
        )
        runtime = QuestionAnswerRuntime(
            QuestionAnswerProtocol(*tuple(
                minimal_instruction_identity((_BASE + 17, index))
                for index in range(1, 4)
            )),
            (QuestionRouteRegistration(
                query_kind,
                route,
                _QuestionExecutor(route, candidate),
            ),),
            selector,
            executor,
            postcheck_mapper=EvidenceQuestionPostcheckMapper(
                (_BASE + 18, 1),
                citation_required=True,
                trust_required=True,
            ),
            postchecker=postchecker,
        )

        run = runtime.run(request)

        assert run.complete
        assert run.generation is not None
        assert run.generation.rendered is not None
        assert run.postcheck is not None and run.postcheck.complete
        assert run.postcheck.parsed.observation.representations == (
            run.generation.representations)
        assert renderer.text(run.generation.rendered) == (
            "档案显示，云岭站西门于2026年启用。")
    finally:
        if alias is not None:
            alias.close()
        backend.close()
