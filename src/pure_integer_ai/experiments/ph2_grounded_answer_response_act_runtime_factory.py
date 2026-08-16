"""grounded literal-only response act 的 run-local G-00 至 G-04 装配。"""
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
    GenerationLayerRegistration,
    GenerationPlanner,
    GenerationPlanningRequest,
    GenerationPlanProtocol,
)
from pure_integer_ai.cognition.shared.generation_response import (
    ResponseActDiscourseRouter,
    ResponseActGenerationRegistry,
    ResponseActPropositionRouter,
    ResponseActSyntaxRouter,
)
from pure_integer_ai.cognition.shared.generation_structure_execution import (
    GenerationStructureExecutionPlan,
    GenerationStructureExecutionPlanner,
    GenerationStructureExecutionRequest,
    SentenceStructureExecutionBudget,
)
from pure_integer_ai.cognition.shared.generation_structure_plan import (
    GenerationDiscourseLayerResolver,
    GenerationPropositionLayerResolver,
    GenerationStructureLayerProtocol,
    GenerationStructurePlan,
    GenerationStructurePlanner,
    GenerationSyntaxLayerResolver,
)
from pure_integer_ai.cognition.shared.generation_surface import (
    GenerationSurfaceProtocol,
    SurfaceSlotDirective,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_MINIMAL_INSTRUCTION,
    ObjectIdentity,
    minimal_instruction_identity,
)
from pure_integer_ai.cognition.shared.question_answer import (
    QuestionExecutionResult,
    QuestionQuery,
)
from pure_integer_ai.cognition.shared.structure_order import (
    StructureOrderConstraintDefinition,
)
from pure_integer_ai.cognition.shared.structure_order_consumer import (
    StructureOrderConsumer,
    StructureOrderConsumerProtocol,
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
    TypedGenerationSurfaceRequestBuilder,
)
from pure_integer_ai.experiments.generation_verification_runtime import (
    GenerationPostcheckProtocol,
    GenerationPostcheckRuntime,
)
from pure_integer_ai.experiments.ph2_grounded_answer_course import (
    GroundedQuestionEpisode,
)
from pure_integer_ai.experiments.ph2_grounded_answer_learning import (
    GroundedAnswerSurfaceModel,
)
from pure_integer_ai.experiments.ph2_grounded_answer_response_act_compile import (
    GroundedResponseActCompilation,
    GroundedResponseActCompileTarget,
    GroundedResponseActVariant,
    compile_grounded_response_act_patterns,
)
from pure_integer_ai.experiments.ph2_grounded_answer_response_act_choice_use import (
    build_grounded_response_act_lexical_choice,
)
from pure_integer_ai.experiments.ph2_grounded_answer_response_act_parser import (
    GroundedResponseActParserCatalog,
    GroundedResponseActParserProtocol,
    GroundedResponseActPostcheckMapper,
    GroundedResponseActSurfaceParser,
    build_grounded_response_act_parser_catalog,
)
from pure_integer_ai.experiments.question_answer_runtime import (
    QuestionAnswerProtocol,
    QuestionAnswerRuntime,
    QuestionRouteRegistration,
)
from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    GenerationChoiceHypothesis,
)


_NAMESPACE = 20962


# object-model: exception
class GroundedResponseActRunLocalFactoryError(ValueError):
    """response-act run-local 组件、课程或同次 planning 发生漂移。"""


def _instruction(value: ObjectIdentity, *, where: str) -> ObjectIdentity:
    """核验 route/reason/renderer 使用 MinimalInstruction。"""
    if (not isinstance(value, ObjectIdentity)
            or value.object_kind != OBJECT_MINIMAL_INSTRUCTION):
        raise GroundedResponseActRunLocalFactoryError(
            f"{where} 必须是 MinimalInstruction")
    return value


def _strict_key(value: tuple[int, ...], *, where: str) -> tuple[int, ...]:
    """核验非空严格整数 trace。"""
    if (not isinstance(value, tuple) or not value
            or any(type(item) is not int for item in value)):
        raise GroundedResponseActRunLocalFactoryError(
            f"{where} 必须是非空严格整数 tuple")
    return value


class GroundedResponseActAliasRuntimeFactory(Protocol):
    """为已显式选择的 response-act variant 建立独占 R-01 owner。"""

    def build(self, variant: GroundedResponseActVariant) -> AliasRelationRuntime:
        """返回只覆盖 stance 到 learned Representation 的 runtime。"""
        ...


class _UnavailableMapper:
    """response-act 未匹配时 fail closed，禁止生成侧静默 fallback。"""

    def plan(self, *args):
        """拒绝当前 installation 之外的 stance。"""
        del args
        raise GroundedResponseActRunLocalFactoryError(
            "当前 response-act installation 不支持普通命题 fallback")


class _NoConstraintResolver:
    """无约束单槽结构的 resolver；若被调用即说明图状态漂移。"""

    def resolve(
            self,
            definition: StructureOrderConstraintDefinition,
            context: tuple[ObjectIdentity, ...],
            ):
        """无约束 schema 不应产生 active constraint 解析请求。"""
        del definition, context
        raise GroundedResponseActRunLocalFactoryError(
            "无约束 response-act structure 出现 active constraint")


# object-model: mapper; state=immutable
@dataclass(frozen=True, slots=True)
class _ResponseActExecutionRequestMapper:
    """为唯一 response-act sentence 注入有限 S-07 搜索预算。"""

    variant: GroundedResponseActVariant

    def build(
            self,
            structure: GenerationStructurePlan,
            ) -> GenerationStructureExecutionRequest:
        """逐句绑定同一 variant 的固定整数预算。"""
        if not isinstance(structure, GenerationStructurePlan):
            raise TypeError("response-act execution mapper 输入类型错误")
        return GenerationStructureExecutionRequest(
            structure.syntax,
            tuple(
                SentenceStructureExecutionBudget(
                    sentence.address,
                    self.variant.order_budget,
                )
                for sentence in structure.syntax.sentences
            ),
        )


# object-model: mapper; state=immutable
@dataclass(frozen=True, slots=True)
class _ResponseActSurfaceDirectiveMapper:
    """把 stance slot 显式映射到 learned literal 的 R-01 surface 查询。"""

    variant: GroundedResponseActVariant
    protocol: GenerationSurfaceProtocol

    def plan(
            self,
            structure: GenerationStructurePlan,
            execution: GenerationStructureExecutionPlan,
            branch: ObjectIdentity,
            ) -> tuple[SurfaceSlotDirective, ...]:
        """核验单句单槽后产生独占 use key，不按 pattern id 返回文本。"""
        if not isinstance(structure, GenerationStructurePlan):
            raise TypeError("response-act surface mapper structure 类型错误")
        if (not isinstance(execution, GenerationStructureExecutionPlan)
                or execution.request.syntax != structure.syntax
                or not execution.complete):
            raise GroundedResponseActRunLocalFactoryError(
                "response-act surface mapper 收到不完整 S-07 execution")
        template = self.variant.template
        sentences = structure.syntax.sentences
        if (branch != template.branch or len(sentences) != 1
                or len(sentences[0].values) != 1):
            raise GroundedResponseActRunLocalFactoryError(
                "response-act surface 单句单槽合同漂移")
        sentence = sentences[0]
        value = sentence.values[0]
        if (sentence.sentence != template.sentence
                or sentence.structure != template.slot.structure
                or value.slot != template.slot.slot
                or value.filler != template.stance):
            raise GroundedResponseActRunLocalFactoryError(
                "response-act surface template/stance 漂移")
        use_key = (
            _NAMESPACE,
            1,
            *structure.selection.stable_key(),
            *template.stance.stable_key(),
            *value.slot.stable_key(),
            *self.variant.use_key_suffix,
        )
        return (SurfaceSlotDirective(
            sentence.address,
            value.slot,
            self.protocol.emit_action,
            self.variant.surface_instruction,
            (1, self.variant.pattern_id),
            (),
            self.variant.surface_budget,
            use_key,
        ),)


# object-model: runtime
class GroundedResponseActQuestionExecutor:
    """只返回 factory 绑定的同次 planning candidates。"""

    def __init__(
            self,
            route: ObjectIdentity,
            planning: GenerationPlanningRequest,
            reason: ObjectIdentity,
            trace: tuple[int, ...],
            ) -> None:
        self.route = _instruction(route, where="response-act executor route")
        if not isinstance(planning, GenerationPlanningRequest):
            raise TypeError("response-act executor planning 类型错误")
        self.planning = planning
        self.reason = _instruction(reason, where="response-act executor reason")
        self.trace = _strict_key(trace, where="response-act executor trace")

    def execute(self, query: QuestionQuery) -> QuestionExecutionResult:
        """拒绝 query/goal 漂移，不读取 surface 或 teacher label。"""
        if not isinstance(query, QuestionQuery):
            raise TypeError("response-act executor query 类型错误")
        if query.route != self.route:
            raise GroundedResponseActRunLocalFactoryError(
                "response-act executor route 漂移")
        result = QuestionExecutionResult(
            query, self.reason, self.planning.candidates, self.trace)
        if result.planning_request() != self.planning:
            raise GroundedResponseActRunLocalFactoryError(
                "response-act executor 替换了 planning goal")
        return result


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GroundedResponseActRunLocalComponents:
    """一次 run 独占的 G-01、R-01、renderer、G-04 和 question 组件。"""

    selector: AnswerContentSelector
    plan_protocol: GenerationPlanProtocol
    structure_protocol: GenerationStructureLayerProtocol
    surface_protocol: GenerationSurfaceProtocol
    alias_factory: GroundedResponseActAliasRuntimeFactory
    renderer: object
    renderer_identity: ObjectIdentity
    postcheck_protocol: GenerationPostcheckProtocol
    structure_verifier: object
    source_verifier: object
    task_verifier: object
    question_protocol: QuestionAnswerProtocol

    def __post_init__(self) -> None:
        if not isinstance(self.selector, AnswerContentSelector):
            raise TypeError("response-act selector 类型错误")
        if not isinstance(self.plan_protocol, GenerationPlanProtocol):
            raise TypeError("response-act plan protocol 类型错误")
        if not isinstance(
                self.structure_protocol, GenerationStructureLayerProtocol):
            raise TypeError("response-act structure protocol 类型错误")
        if not isinstance(self.surface_protocol, GenerationSurfaceProtocol):
            raise TypeError("response-act surface protocol 类型错误")
        if not hasattr(self.alias_factory, "build"):
            raise TypeError("response-act alias factory 缺少 build")
        if not hasattr(self.renderer, "render"):
            raise TypeError("response-act renderer 缺少 render")
        _instruction(
            self.renderer_identity, where="response-act renderer identity")
        if not isinstance(self.postcheck_protocol, GenerationPostcheckProtocol):
            raise TypeError("response-act postcheck protocol 类型错误")
        for label, verifier in (
                ("structure", self.structure_verifier),
                ("source", self.source_verifier),
                ("task", self.task_verifier)):
            if not hasattr(verifier, "verify"):
                raise TypeError(f"response-act {label} verifier 缺少 verify")
        if not isinstance(self.question_protocol, QuestionAnswerProtocol):
            raise TypeError("response-act question protocol 类型错误")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GroundedResponseActRunLocalBuild:
    """显式 response-act pattern 与同次 planning/query 的装配请求。"""

    model: GroundedAnswerSurfaceModel
    question: GroundedQuestionEpisode
    target: GroundedResponseActCompileTarget
    planning: GenerationPlanningRequest
    pattern_id: int
    parser_protocol: GroundedResponseActParserProtocol
    query_kind: ObjectIdentity
    route: ObjectIdentity
    execution_reason: ObjectIdentity
    execution_trace: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.model, GroundedAnswerSurfaceModel):
            raise TypeError("response-act build model 类型错误")
        if not isinstance(self.question, GroundedQuestionEpisode):
            raise TypeError("response-act build question 类型错误")
        if not isinstance(self.target, GroundedResponseActCompileTarget):
            raise TypeError("response-act build target 类型错误")
        if self.question.answer_plan.response_act != self.target.response_act:
            raise GroundedResponseActRunLocalFactoryError(
                "response-act question/target 漂移")
        if self.question.answer_plan.ordered_claim_ids:
            raise GroundedResponseActRunLocalFactoryError(
                "response-act question 不得携带 ordered claim")
        if not isinstance(self.planning, GenerationPlanningRequest):
            raise TypeError("response-act build planning 类型错误")
        if self.planning.goal.target_branch != self.target.language_branch:
            raise GroundedResponseActRunLocalFactoryError(
                "response-act planning branch 漂移")
        if type(self.pattern_id) is not int or self.pattern_id <= 0:
            raise GroundedResponseActRunLocalFactoryError(
                "response-act selected pattern id 非法")
        if not isinstance(
                self.parser_protocol, GroundedResponseActParserProtocol):
            raise TypeError("response-act parser protocol 类型错误")
        for label, value in (
                ("query kind", self.query_kind),
                ("route", self.route),
                ("execution reason", self.execution_reason)):
            _instruction(value, where=f"response-act {label}")
        _strict_key(
            self.execution_trace, where="response-act execution trace")


# object-model: runtime-bundle; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GroundedResponseActRunLocalInstallation:
    """selected learned response act 到真实 G-00 至 G-04 的完整装配。"""

    compilation: GroundedResponseActCompilation
    variant: GroundedResponseActVariant
    planning: GenerationPlanningRequest
    alias: AliasRelationRuntime
    parser_catalog: GroundedResponseActParserCatalog
    parser: GroundedResponseActSurfaceParser
    lexical_choice: GenerationChoiceHypothesis
    executor: TypedGenerationExecutor
    runtime: QuestionAnswerRuntime

    def __post_init__(self) -> None:
        if not isinstance(self.compilation, GroundedResponseActCompilation):
            raise TypeError("response-act installation compilation 类型错误")
        if self.compilation.select(self.variant.pattern_id) != self.variant:
            raise GroundedResponseActRunLocalFactoryError(
                "response-act installation variant 不属于 compilation")
        if not isinstance(self.planning, GenerationPlanningRequest):
            raise TypeError("response-act installation planning 类型错误")
        if not isinstance(self.alias, AliasRelationRuntime):
            raise TypeError("response-act installation alias 类型错误")
        if not isinstance(
                self.parser_catalog, GroundedResponseActParserCatalog):
            raise TypeError("response-act parser catalog 类型错误")
        if not isinstance(self.parser, GroundedResponseActSurfaceParser):
            raise TypeError("response-act parser 类型错误")
        if not isinstance(self.lexical_choice, GenerationChoiceHypothesis):
            raise TypeError("response-act lexical choice 类型错误")
        if (self.lexical_choice.selected_object != self.variant.template.sentence
                or self.lexical_choice.target_obligation
                != self.variant.template.stance
                or self.lexical_choice.authorized_scope
                != self.planning.goal.scope
                or self.lexical_choice.exact_uses
                or self.lexical_choice.typed_outcomes):
            raise GroundedResponseActRunLocalFactoryError(
                "response-act installation lexical choice 漂移")
        if not isinstance(self.executor, TypedGenerationExecutor):
            raise TypeError("response-act executor 类型错误")
        if not isinstance(self.runtime, QuestionAnswerRuntime):
            raise TypeError("response-act runtime 类型错误")


# object-model: factory
class GroundedResponseActRunLocalFactory:
    """原子装配一个 learned non-answer pattern，不接默认 broad-QA CLI。"""

    def __init__(
            self,
            lifecycle: StructureOrderLifecycleGraph,
            components: GroundedResponseActRunLocalComponents,
            ) -> None:
        if not isinstance(lifecycle, StructureOrderLifecycleGraph):
            raise TypeError("response-act lifecycle 类型错误")
        if not isinstance(components, GroundedResponseActRunLocalComponents):
            raise TypeError("response-act components 类型错误")
        self.lifecycle = lifecycle
        self.components = components

    @staticmethod
    def _preflight_alias(
            variant: GroundedResponseActVariant,
            alias: AliasRelationRuntime,
            ) -> None:
        """零写证明 stance 经独占 R-01 route 唯一恢复 learned Representation。"""
        proposal = alias.preview_surface(
            variant.template.stance,
            variant.template.branch,
            budget=variant.surface_budget,
            allowed_prefix_steps=(),
        )
        selected = proposal.result.selected
        if selected is None or selected.value != variant.representation:
            raise GroundedResponseActRunLocalFactoryError(
                "response-act alias 未唯一恢复 learned Representation")

    def build(
            self,
            request: GroundedResponseActRunLocalBuild,
            ) -> GroundedResponseActRunLocalInstallation:
        """显式选择 pattern 后装配 response-act G-00 至 G-04。"""
        if not isinstance(request, GroundedResponseActRunLocalBuild):
            raise TypeError("response-act factory build request 类型错误")
        components = self.components
        selection = components.selector.select(request.planning)
        if selection.stance != request.target.stance:
            raise GroundedResponseActRunLocalFactoryError(
                "response-act target stance 与 G-01 selection 漂移")
        compilation = compile_grounded_response_act_patterns(
            request.model, request.target)
        variant = compilation.select(request.pattern_id)
        alias = components.alias_factory.build(variant)
        if not isinstance(alias, AliasRelationRuntime):
            raise TypeError("response-act alias factory 返回类型错误")
        self._preflight_alias(variant, alias)

        self.lifecycle.order_graph.define_structure(
            (variant.template.slot,),
            variant.template.branch,
            variant.structure_family,
            scope=request.planning.goal.scope,
            provenance_kind=_NAMESPACE,
            qualifiers=(variant.pattern_id,),
        )
        consumer = StructureOrderConsumer(
            self.lifecycle,
            _NoConstraintResolver(),
            StructureOrderConsumerProtocol(*tuple(
                _instruction(
                    minimal_instruction_identity(
                        (_NAMESPACE, 2, variant.pattern_id, index),
                        owner=variant.template.branch.owner,
                        versions=variant.template.branch.versions,
                    ),
                    where="response-act S-07 consumer reason",
                )
                for index in range(1, 8)
            )),
        )
        execution_planner = GenerationStructureExecutionPlanner(
            self.lifecycle, consumer)
        registry = ResponseActGenerationRegistry((variant.template,))
        unavailable = _UnavailableMapper()
        structure_planner = GenerationStructurePlanner(
            ResponseActDiscourseRouter(unavailable, registry),
            ResponseActPropositionRouter(unavailable, registry),
            ResponseActSyntaxRouter(unavailable, registry),
        )
        surface_runtime = GenerationSurfaceRuntime(alias)
        surface_builder = TypedGenerationSurfaceRequestBuilder(
            components.surface_protocol,
            execution_planner,
            _ResponseActExecutionRequestMapper(variant),
            _ResponseActSurfaceDirectiveMapper(
                variant, components.surface_protocol),
        )
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
        catalog = build_grounded_response_act_parser_catalog(
            compilation,
            components.renderer_identity,
            request.planning.goal.source,
            request.planning.goal.scope,
        )
        parser = GroundedResponseActSurfaceParser(
            request.parser_protocol, catalog)
        postchecker = GenerationPostcheckRuntime(
            components.postcheck_protocol,
            parser,
            components.structure_verifier,
            components.source_verifier,
            task_verifier=components.task_verifier,
        )
        question_executor = GroundedResponseActQuestionExecutor(
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
            postcheck_mapper=GroundedResponseActPostcheckMapper(
                variant, (_NAMESPACE, 3)),
            postchecker=postchecker,
        )
        lexical_choice = build_grounded_response_act_lexical_choice(
            variant, request.planning)
        return GroundedResponseActRunLocalInstallation(
            compilation,
            variant,
            request.planning,
            alias,
            catalog,
            parser,
            lexical_choice,
            executor,
            runtime,
        )


__all__ = [
    "GroundedResponseActAliasRuntimeFactory",
    "GroundedResponseActQuestionExecutor",
    "GroundedResponseActRunLocalBuild",
    "GroundedResponseActRunLocalComponents",
    "GroundedResponseActRunLocalFactory",
    "GroundedResponseActRunLocalFactoryError",
    "GroundedResponseActRunLocalInstallation",
]
