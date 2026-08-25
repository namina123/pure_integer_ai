"""grounded-answer 显式 pattern 的 run-local 问答装配入口。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pure_integer_ai.cognition.shared.generation_content import (
    AnswerContentSelector,
    GenerationContentLayerResolver,
    GenerationStanceLayerResolver,
)
from pure_integer_ai.cognition.shared.generation_execution import (
    TypedGenerationExecutor,
)
from pure_integer_ai.cognition.shared.generation_plan import (
    GenerationCandidate,
    GenerationLayerRegistration,
    GenerationPlanner,
    GenerationPlanningRequest,
    GenerationPlanProtocol,
)
from pure_integer_ai.cognition.shared.generation_structure_plan import (
    GenerationDiscourseLayerResolver,
    GenerationPropositionLayerResolver,
    GenerationStructureLayerProtocol,
    GenerationSyntaxLayerResolver,
)
from pure_integer_ai.cognition.shared.generation_surface import (
    GenerationSurfaceAttribution,
    GenerationSurfaceProtocol,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_MINIMAL_INSTRUCTION,
    ObjectIdentity,
)
from pure_integer_ai.cognition.shared.question_answer import (
    QuestionExecutionResult,
    QuestionQuery,
)
from pure_integer_ai.cognition.shared.structure_order_lifecycle import (
    StructureOrderLifecycleGraph,
)
from pure_integer_ai.experiments.alias_relation_runtime import (
    AliasRelationRuntime,
)
from pure_integer_ai.experiments.generation_surface_runtime import (
    GenerationSurfaceLayerResolver,
    GenerationSurfaceRuntime,
)
from pure_integer_ai.experiments.generation_verification_runtime import (
    GenerationPostcheckProtocol,
    GenerationPostcheckRuntime,
)
from pure_integer_ai.experiments.language_generation_connector import (
    LanguageGenerationConnector,
)
from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    GenerationChoiceHypothesis,
)
from pure_integer_ai.experiments.ph2_grounded_answer_choice import (
    build_grounded_answer_lexical_choice,
)
from pure_integer_ai.experiments.ph2_grounded_answer_connector import (
    GroundedAnswerClaimInput,
    GroundedAnswerConnectorCompilation,
    GroundedAnswerConnectorTarget,
    GroundedAnswerConnectorVariant,
    GroundedAnswerStructureSelection,
    build_grounded_answer_connector,
    compile_grounded_answer_connectors,
    select_grounded_answer_structure,
)
from pure_integer_ai.experiments.ph2_grounded_answer_course import (
    GroundedQuestionEpisode,
)
from pure_integer_ai.experiments.ph2_grounded_answer_learning import (
    GroundedAnswerSurfaceModel,
)
from pure_integer_ai.experiments.ph2_grounded_answer_order import (
    GroundedAnswerOrderInstallation,
    install_grounded_answer_order_course,
)
from pure_integer_ai.experiments.ph2_grounded_answer_parser import (
    GroundedAnswerParserCatalog,
    GroundedAnswerParserProtocol,
    GroundedAnswerSurfaceParser,
    build_grounded_answer_parser_catalog,
)
from pure_integer_ai.experiments.question_answer_runtime import (
    QuestionAnswerProtocol,
    QuestionAnswerRuntime,
    QuestionRouteRegistration,
)


# object-model: exception
class GroundedAnswerRunLocalFactoryError(ValueError):
    """run-local 组件、显式 pattern 或同次 planning 发生漂移。"""


class GroundedAnswerAliasRuntimeFactory(Protocol):
    """为已显式选择的 variant 建立本次独占 R-01 owner。"""

    def build(
            self,
            variant: GroundedAnswerConnectorVariant,
            ) -> AliasRelationRuntime:
        """返回已覆盖 variant 全部 alias requirement 的 runtime。"""
        ...


def _instruction(value: ObjectIdentity, *, where: str) -> ObjectIdentity:
    """核验 route/reason 使用一等 MinimalInstruction。"""
    if not isinstance(value, ObjectIdentity):
        raise TypeError(f"{where} 类型错误")
    if value.object_kind != OBJECT_MINIMAL_INSTRUCTION:
        raise GroundedAnswerRunLocalFactoryError(
            f"{where} 必须是 MinimalInstruction")
    return value


def _strict_key(value: tuple[int, ...], *, where: str) -> tuple[int, ...]:
    """核验 run-local trace 是非空严格整数 tuple。"""
    if (not isinstance(value, tuple) or not value
            or any(type(item) is not int for item in value)):
        raise GroundedAnswerRunLocalFactoryError(
            f"{where} 必须是非空严格整数 tuple")
    return value


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GroundedAnswerRunLocalComponents:
    """一次 run 独占的 G-01、R-01、renderer、G-04 与 question 组件。"""

    selector: AnswerContentSelector
    plan_protocol: GenerationPlanProtocol
    structure_protocol: GenerationStructureLayerProtocol
    alias_factory: GroundedAnswerAliasRuntimeFactory
    renderer: object
    renderer_identity: ObjectIdentity
    postcheck_protocol: GenerationPostcheckProtocol
    structure_verifier: object
    source_verifier: object
    question_protocol: QuestionAnswerProtocol
    postcheck_mapper: object
    artifact_verifier: object | None = None
    task_verifier: object | None = None
    surface_attributions: tuple[GenerationSurfaceAttribution, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.selector, AnswerContentSelector):
            raise TypeError("grounded factory selector 类型错误")
        if not isinstance(self.plan_protocol, GenerationPlanProtocol):
            raise TypeError("grounded factory plan protocol 类型错误")
        if not isinstance(
                self.structure_protocol, GenerationStructureLayerProtocol):
            raise TypeError("grounded factory structure protocol 类型错误")
        if not hasattr(self.alias_factory, "build"):
            raise TypeError("grounded factory alias factory 缺少 build")
        if not hasattr(self.renderer, "render"):
            raise TypeError("grounded factory renderer 缺少 render")
        _instruction(self.renderer_identity, where="grounded renderer identity")
        if not isinstance(self.postcheck_protocol, GenerationPostcheckProtocol):
            raise TypeError("grounded factory postcheck protocol 类型错误")
        for label, verifier in (
                ("structure verifier", self.structure_verifier),
                ("source verifier", self.source_verifier)):
            if not hasattr(verifier, "verify"):
                raise TypeError(f"grounded factory {label} 缺少 verify")
        if (self.artifact_verifier is not None
                and not hasattr(self.artifact_verifier, "verify")):
            raise TypeError("grounded factory artifact verifier 缺少 verify")
        if (self.task_verifier is not None
                and not hasattr(self.task_verifier, "verify")):
            raise TypeError("grounded factory task verifier 缺少 verify")
        if not isinstance(self.question_protocol, QuestionAnswerProtocol):
            raise TypeError("grounded factory question protocol 类型错误")
        if not hasattr(self.postcheck_mapper, "build"):
            raise TypeError("grounded factory postcheck mapper 缺少 build")
        if (not isinstance(self.surface_attributions, tuple)
                or any(not isinstance(item, GenerationSurfaceAttribution)
                       for item in self.surface_attributions)):
            raise TypeError("grounded factory surface attributions 类型错误")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GroundedAnswerRunLocalBuild:
    """显式 pattern 与同次 planning/query route 的不可变装配请求。"""

    model: GroundedAnswerSurfaceModel
    question: GroundedQuestionEpisode | GroundedAnswerClaimInput
    target: GroundedAnswerConnectorTarget
    planning: GenerationPlanningRequest
    candidate: GenerationCandidate
    structure_id: int
    pattern_id: int
    parser_protocol: GroundedAnswerParserProtocol
    query_kind: ObjectIdentity
    route: ObjectIdentity
    execution_reason: ObjectIdentity
    execution_trace: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.model, GroundedAnswerSurfaceModel):
            raise TypeError("grounded build model 类型错误")
        if not isinstance(self.question, (GroundedQuestionEpisode,
                                          GroundedAnswerClaimInput)):
            raise TypeError("grounded build question 类型错误")
        if not isinstance(self.target, GroundedAnswerConnectorTarget):
            raise TypeError("grounded build target 类型错误")
        if not isinstance(self.planning, GenerationPlanningRequest):
            raise TypeError("grounded build planning 类型错误")
        if not isinstance(self.candidate, GenerationCandidate):
            raise TypeError("grounded build candidate 类型错误")
        if self.candidate not in self.planning.candidates:
            raise GroundedAnswerRunLocalFactoryError(
                "grounded build candidate 不属于 planning")
        if self.target.proposition != self.candidate.proposition:
            raise GroundedAnswerRunLocalFactoryError(
                "grounded target 替换了 planning candidate Proposition")
        if self.planning.goal.target_branch != self.target.language_branch:
            raise GroundedAnswerRunLocalFactoryError(
                "grounded target branch 与 planning goal 漂移")
        if type(self.structure_id) is not int or self.structure_id <= 0:
            raise GroundedAnswerRunLocalFactoryError(
                "grounded build structure id 非法")
        if type(self.pattern_id) is not int or self.pattern_id <= 0:
            raise GroundedAnswerRunLocalFactoryError(
                "grounded build pattern id 非法")
        if not isinstance(self.parser_protocol, GroundedAnswerParserProtocol):
            raise TypeError("grounded build parser protocol 类型错误")
        for label, value in (
                ("query kind", self.query_kind),
                ("route", self.route),
                ("execution reason", self.execution_reason)):
            _instruction(value, where=f"grounded {label}")
        _strict_key(self.execution_trace, where="grounded execution trace")


# object-model: runtime
class GroundedAnswerQuestionExecutor:
    """只返回本 factory 绑定的同次 planning candidates。"""

    def __init__(
            self,
            route: ObjectIdentity,
            planning: GenerationPlanningRequest,
            reason: ObjectIdentity,
            trace: tuple[int, ...],
            ) -> None:
        self.route = _instruction(route, where="grounded executor route")
        if not isinstance(planning, GenerationPlanningRequest):
            raise TypeError("grounded executor planning 类型错误")
        self.planning = planning
        self.reason = _instruction(reason, where="grounded executor reason")
        self.trace = _strict_key(trace, where="grounded executor trace")

    def execute(self, query: QuestionQuery) -> QuestionExecutionResult:
        """拒绝 query/goal 漂移，不读取 surface、answer plan 或 teacher label。"""
        if not isinstance(query, QuestionQuery):
            raise TypeError("grounded executor query 类型错误")
        if query.route != self.route:
            raise GroundedAnswerRunLocalFactoryError(
                "grounded executor route 漂移")
        result = QuestionExecutionResult(
            query,
            self.reason,
            self.planning.candidates,
            self.trace,
        )
        if result.planning_request() != self.planning:
            raise GroundedAnswerRunLocalFactoryError(
                "grounded run-local query 替换了冻结 planning goal")
        return result


# object-model: runtime-bundle; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GroundedAnswerRunLocalInstallation:
    """显式 variant 到 QuestionAnswerRuntime 的完整同次装配。"""

    compilation: GroundedAnswerConnectorCompilation
    structure_selection: GroundedAnswerStructureSelection
    variant: GroundedAnswerConnectorVariant
    planning: GenerationPlanningRequest
    candidate: GenerationCandidate
    connector: LanguageGenerationConnector
    alias: AliasRelationRuntime
    order: GroundedAnswerOrderInstallation
    parser_catalog: GroundedAnswerParserCatalog
    parser: GroundedAnswerSurfaceParser
    lexical_choice: GenerationChoiceHypothesis
    executor: TypedGenerationExecutor
    runtime: QuestionAnswerRuntime

    def __post_init__(self) -> None:
        if not isinstance(
                self.compilation, GroundedAnswerConnectorCompilation):
            raise TypeError("grounded installation compilation 类型错误")
        if not isinstance(
                self.structure_selection, GroundedAnswerStructureSelection):
            raise TypeError("grounded installation structure selection 类型错误")
        if (self.structure_selection.options != self.compilation.structures
                or self.structure_selection.selected.structure_id
                != self.variant.option.structure_id
                or self.variant.option.pattern_id not in (
                    self.structure_selection.selected.pattern_ids)):
            raise GroundedAnswerRunLocalFactoryError(
                "grounded installation structure/lexical 选择次序漂移")
        if not isinstance(self.variant, GroundedAnswerConnectorVariant):
            raise TypeError("grounded installation variant 类型错误")
        if self.compilation.select(
                self.variant.option.pattern_id) != self.variant:
            raise GroundedAnswerRunLocalFactoryError(
                "grounded installation variant 不属于 compilation")
        if not isinstance(self.planning, GenerationPlanningRequest):
            raise TypeError("grounded installation planning 类型错误")
        if not isinstance(self.candidate, GenerationCandidate):
            raise TypeError("grounded installation candidate 类型错误")
        if (self.candidate not in self.planning.candidates
                or self.candidate.proposition.structure
                != self.variant.template.proposition_structure
                or self.candidate.proposition.predicate
                != self.variant.template.predicate
                or self.planning.goal.target_branch
                != self.variant.template.language_branch):
            raise GroundedAnswerRunLocalFactoryError(
                "grounded installation 冻结 planning/candidate 与 variant 漂移")
        if not isinstance(self.connector, LanguageGenerationConnector):
            raise TypeError("grounded installation connector 类型错误")
        if self.connector.registry.templates != (self.variant.template,):
            raise GroundedAnswerRunLocalFactoryError(
                "grounded installation connector 未精确选择单一 variant")
        if not isinstance(self.alias, AliasRelationRuntime):
            raise TypeError("grounded installation alias 类型错误")
        if not isinstance(self.order, GroundedAnswerOrderInstallation):
            raise TypeError("grounded installation order 类型错误")
        if self.order.variant != self.variant:
            raise GroundedAnswerRunLocalFactoryError(
                "grounded installation S-07 替换了 variant")
        if not isinstance(self.parser_catalog, GroundedAnswerParserCatalog):
            raise TypeError("grounded installation parser catalog 类型错误")
        if not isinstance(self.parser, GroundedAnswerSurfaceParser):
            raise TypeError("grounded installation parser 类型错误")
        if self.parser.catalog != self.parser_catalog:
            raise GroundedAnswerRunLocalFactoryError(
                "grounded installation parser 替换了 catalog")
        if not isinstance(self.lexical_choice, GenerationChoiceHypothesis):
            raise TypeError("grounded installation lexical choice 类型错误")
        if (self.lexical_choice.choice_kind
                != "LEXICAL_REALIZATION_CHOICE"
                or self.lexical_choice.selected_object
                != self.variant.template.connector
                or self.lexical_choice.exact_uses
                or self.lexical_choice.typed_outcomes):
            raise GroundedAnswerRunLocalFactoryError(
                "grounded installation GG-01 lexical 边界漂移")
        if not isinstance(self.executor, TypedGenerationExecutor):
            raise TypeError("grounded installation executor 类型错误")
        if not isinstance(self.runtime, QuestionAnswerRuntime):
            raise TypeError("grounded installation runtime 类型错误")


# object-model: factory
class GroundedAnswerRunLocalFactory:
    """在既有 run-local owners 上原子装配一个显式 grounded variant。"""

    def __init__(
            self,
            surface_protocol: GenerationSurfaceProtocol,
            lifecycle: StructureOrderLifecycleGraph,
            components: GroundedAnswerRunLocalComponents,
            ) -> None:
        if not isinstance(surface_protocol, GenerationSurfaceProtocol):
            raise TypeError("grounded factory surface protocol 类型错误")
        if not isinstance(lifecycle, StructureOrderLifecycleGraph):
            raise TypeError("grounded factory lifecycle 类型错误")
        if not isinstance(components, GroundedAnswerRunLocalComponents):
            raise TypeError("grounded factory components 类型错误")
        self.surface_protocol = surface_protocol
        self.lifecycle = lifecycle
        self.components = components

    @staticmethod
    def _preflight_aliases(
            variant: GroundedAnswerConnectorVariant,
            alias: AliasRelationRuntime,
            ) -> None:
        """零写证明 R-01 对每个 slot 恰好恢复编译要求的 Representation。"""
        policies = {
            item.slot: item
            for template in variant.runtime_policy.templates
            for item in template.surface
        }
        directives = {item.slot: item for item in variant.template.surface}
        if (set(policies) != {item.slot for item in variant.aliases}
                or set(directives) != {item.slot for item in variant.aliases}):
            raise GroundedAnswerRunLocalFactoryError(
                "grounded alias preflight slot 覆盖漂移")
        for requirement in variant.aliases:
            policy = policies[requirement.slot]
            if policy.surface_budget is None:
                raise GroundedAnswerRunLocalFactoryError(
                    "grounded alias preflight 缺 surface budget")
            proposal = alias.preview_surface(
                requirement.filler,
                variant.template.language_branch,
                budget=policy.surface_budget,
                allowed_prefix_steps=(
                    directives[requirement.slot].surface_prefix_steps),
                expected_value=requirement.representation,
            )
            selected = proposal.result.selected
            if selected is None or selected.value != requirement.representation:
                raise GroundedAnswerRunLocalFactoryError(
                    "grounded alias factory 未提供唯一预期 Representation: "
                    f"slot={requirement.slot.stable_key()} "
                    f"expected={requirement.representation.stable_key()} "
                    f"outcome={proposal.result.outcome.stable_key()} "
                    f"options={tuple(item.value.stable_key() for item in proposal.result.options)}")

    def build(
            self,
            request: GroundedAnswerRunLocalBuild,
            ) -> GroundedAnswerRunLocalInstallation:
        """显式选择一个 pattern 后装配 G-00 至 G-04；不接默认 CLI。"""
        if not isinstance(request, GroundedAnswerRunLocalBuild):
            raise TypeError("grounded factory build request 类型错误")
        components = self.components
        selection = components.selector.select(request.planning)
        if request.parser_protocol.answer_stance != selection.stance:
            raise GroundedAnswerRunLocalFactoryError(
                "grounded parser stance 与 G-01 selection 漂移")
        if selection.selected_candidate_keys != (request.candidate.stable_key(),):
            raise GroundedAnswerRunLocalFactoryError(
                "grounded run-local 只能为 G-01 精确选中的单 candidate 生成")
        compilation = compile_grounded_answer_connectors(
            request.model,
            request.question,
            request.target,
            self.surface_protocol,
        )
        structure_selection = select_grounded_answer_structure(
            compilation, request.structure_id)
        variant, connector = build_grounded_answer_connector(
            compilation,
            structure_selection.selected.structure_id,
            request.pattern_id,
            self.surface_protocol,
        )
        if components.surface_attributions:
            connector = LanguageGenerationConnector(
                connector.registry,
                connector.runtime_policy,
                connector.surface_protocol,
                components.surface_attributions,
                connector.discourse_declarations,
                connector.anaphora_declarations,
            )
        alias = components.alias_factory.build(variant)
        if not isinstance(alias, AliasRelationRuntime):
            raise TypeError("grounded alias factory 返回类型错误")
        self._preflight_aliases(variant, alias)
        order = install_grounded_answer_order_course(
            variant, self.lifecycle)
        structure_planner = connector.structure_planner()
        surface_runtime = GenerationSurfaceRuntime(
            alias,
            {item.slot: item.representation for item in variant.aliases},
        )
        surface_builder = connector.surface_request_builder(
            order.execution_planner)
        protocol = components.plan_protocol
        structure_protocol = components.structure_protocol
        selector = components.selector
        registrations = (
            GenerationLayerRegistration(
                protocol.stance_layer,
                GenerationStanceLayerResolver(protocol, selector),
            ),
            GenerationLayerRegistration(
                protocol.content_layer,
                GenerationContentLayerResolver(protocol, selector),
            ),
            GenerationLayerRegistration(
                protocol.discourse_layer,
                GenerationDiscourseLayerResolver(
                    protocol,
                    structure_protocol,
                    selector,
                    structure_planner,
                ),
            ),
            GenerationLayerRegistration(
                protocol.proposition_layer,
                GenerationPropositionLayerResolver(
                    protocol,
                    structure_protocol,
                    selector,
                    structure_planner,
                ),
            ),
            GenerationLayerRegistration(
                protocol.syntax_layer,
                GenerationSyntaxLayerResolver(
                    protocol,
                    structure_protocol,
                    selector,
                    structure_planner,
                ),
            ),
            GenerationLayerRegistration(
                protocol.surface_layer,
                GenerationSurfaceLayerResolver(
                    protocol,
                    selector,
                    structure_planner,
                    surface_builder,
                    surface_runtime,
                    commit=False,
                ),
            ),
        )
        executor = TypedGenerationExecutor(
            GenerationPlanner(protocol, registrations),
            components.renderer,
            surface_runtime,
        )
        parser_catalog = build_grounded_answer_parser_catalog(
            compilation,
            request.candidate,
            components.renderer_identity,
        )
        parser = GroundedAnswerSurfaceParser(
            request.parser_protocol, parser_catalog)
        postchecker = GenerationPostcheckRuntime(
            components.postcheck_protocol,
            parser,
            components.structure_verifier,
            components.source_verifier,
            artifact_verifier=components.artifact_verifier,
            task_verifier=components.task_verifier,
        )
        question_executor = GroundedAnswerQuestionExecutor(
            request.route,
            request.planning,
            request.execution_reason,
            request.execution_trace,
        )
        runtime = QuestionAnswerRuntime(
            components.question_protocol,
            (QuestionRouteRegistration(
                request.query_kind,
                request.route,
                question_executor,
            ),),
            selector,
            executor,
            postcheck_mapper=components.postcheck_mapper,
            postchecker=postchecker,
        )
        lexical_choice = build_grounded_answer_lexical_choice(
            variant, request.candidate)
        return GroundedAnswerRunLocalInstallation(
            compilation,
            structure_selection,
            variant,
            request.planning,
            request.candidate,
            connector,
            alias,
            order,
            parser_catalog,
            parser,
            lexical_choice,
            executor,
            runtime,
        )


__all__ = [
    "GroundedAnswerAliasRuntimeFactory",
    "GroundedAnswerQuestionExecutor",
    "GroundedAnswerRunLocalBuild",
    "GroundedAnswerRunLocalComponents",
    "GroundedAnswerRunLocalFactory",
    "GroundedAnswerRunLocalFactoryError",
    "GroundedAnswerRunLocalInstallation",
]
