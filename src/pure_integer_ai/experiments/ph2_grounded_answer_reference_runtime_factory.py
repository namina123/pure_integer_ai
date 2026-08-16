"""装配 grounded reference 双句 compilation 的 run-local G-00 至 G-04。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

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
    GenerationSyntaxLayerResolver,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_MINIMAL_INSTRUCTION,
    ObjectIdentity,
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
    GenerationPostcheckRuntime,
)
from pure_integer_ai.experiments.ph2_grounded_answer_parser import (
    GroundedAnswerParserProtocol,
)
from pure_integer_ai.experiments.ph2_grounded_answer_reference_compile import (
    GroundedAnswerReferenceCompilation,
)
from pure_integer_ai.experiments.ph2_grounded_answer_reference_order import (
    GroundedAnswerReferenceOrderInstallation,
    install_grounded_answer_reference_order,
)
from pure_integer_ai.experiments.ph2_grounded_answer_reference_parser import (
    GroundedAnswerReferenceParserCatalog,
    GroundedAnswerReferenceSurfaceParser,
    build_grounded_answer_reference_parser_catalog,
)
from pure_integer_ai.experiments.ph2_grounded_answer_runtime_factory import (
    GroundedAnswerQuestionExecutor,
    GroundedAnswerRunLocalComponents,
)
from pure_integer_ai.experiments.question_answer_runtime import (
    QuestionAnswerRuntime,
    QuestionRouteRegistration,
)


# object-model: exception
class GroundedAnswerReferenceRunLocalFactoryError(ValueError):
    """reference compilation、组件或同次 planning 发生漂移。"""


class GroundedAnswerReferenceAliasRuntimeFactory(Protocol):
    """为一个完整双句 compilation 建立独占 R-01 owner。"""

    def build(
            self,
            compilation: GroundedAnswerReferenceCompilation,
            ) -> AliasRelationRuntime:
        """返回覆盖全部 direct surface 与 reference relation 的 runtime。"""
        ...


def _instruction(value: ObjectIdentity, *, where: str) -> ObjectIdentity:
    """核验 query/route/reason 使用 MinimalInstruction。"""
    if (not isinstance(value, ObjectIdentity)
            or value.object_kind != OBJECT_MINIMAL_INSTRUCTION):
        raise GroundedAnswerReferenceRunLocalFactoryError(
            f"{where} 必须是 MinimalInstruction")
    return value


def _strict_key(value: tuple[int, ...], *, where: str) -> tuple[int, ...]:
    """核验非空严格整数 trace。"""
    if (not isinstance(value, tuple) or not value
            or any(type(item) is not int for item in value)):
        raise GroundedAnswerReferenceRunLocalFactoryError(
            f"{where} 必须是非空严格整数 tuple")
    return value


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GroundedAnswerReferenceRunLocalBuild:
    """绑定已编译双句课程与本次 question route。"""

    compilation: GroundedAnswerReferenceCompilation
    parser_protocol: GroundedAnswerParserProtocol
    query_kind: ObjectIdentity
    route: ObjectIdentity
    execution_reason: ObjectIdentity
    execution_trace: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(
                self.compilation, GroundedAnswerReferenceCompilation):
            raise TypeError("reference build compilation 类型错误")
        if not isinstance(self.parser_protocol, GroundedAnswerParserProtocol):
            raise TypeError("reference build parser protocol 类型错误")
        for label, value in (
                ("query kind", self.query_kind),
                ("route", self.route),
                ("execution reason", self.execution_reason)):
            _instruction(value, where=f"reference {label}")
        _strict_key(self.execution_trace, where="reference execution trace")


# object-model: runtime-bundle; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GroundedAnswerReferenceRunLocalInstallation:
    """双句 connector、R-01、S-07、parser 和 question runtime。"""

    compilation: GroundedAnswerReferenceCompilation
    alias: AliasRelationRuntime
    order: GroundedAnswerReferenceOrderInstallation
    parser_catalog: GroundedAnswerReferenceParserCatalog
    parser: GroundedAnswerReferenceSurfaceParser
    executor: TypedGenerationExecutor
    runtime: QuestionAnswerRuntime

    def __post_init__(self) -> None:
        if not isinstance(
                self.compilation, GroundedAnswerReferenceCompilation):
            raise TypeError("reference installation compilation 类型错误")
        if not isinstance(self.alias, AliasRelationRuntime):
            raise TypeError("reference installation alias 类型错误")
        if not isinstance(
                self.order, GroundedAnswerReferenceOrderInstallation):
            raise TypeError("reference installation order 类型错误")
        if self.order.compilation != self.compilation:
            raise GroundedAnswerReferenceRunLocalFactoryError(
                "reference order 替换了 compilation")
        if not isinstance(
                self.parser_catalog, GroundedAnswerReferenceParserCatalog):
            raise TypeError("reference installation parser catalog 类型错误")
        if self.parser_catalog.compilation != self.compilation:
            raise GroundedAnswerReferenceRunLocalFactoryError(
                "reference parser catalog 替换了 compilation")
        if not isinstance(self.parser, GroundedAnswerReferenceSurfaceParser):
            raise TypeError("reference installation parser 类型错误")
        if self.parser.catalog != self.parser_catalog:
            raise GroundedAnswerReferenceRunLocalFactoryError(
                "reference parser 替换了 catalog")
        if not isinstance(self.executor, TypedGenerationExecutor):
            raise TypeError("reference installation executor 类型错误")
        if not isinstance(self.runtime, QuestionAnswerRuntime):
            raise TypeError("reference installation runtime 类型错误")


# object-model: factory
class GroundedAnswerReferenceRunLocalFactory:
    """原子装配一个已编译 reference strategy；不接 choice/Use。"""

    def __init__(
            self,
            lifecycle: StructureOrderLifecycleGraph,
            components: GroundedAnswerRunLocalComponents,
            ) -> None:
        if not isinstance(lifecycle, StructureOrderLifecycleGraph):
            raise TypeError("reference factory lifecycle 类型错误")
        if not isinstance(components, GroundedAnswerRunLocalComponents):
            raise TypeError("reference factory components 类型错误")
        self.lifecycle = lifecycle
        self.components = components

    @staticmethod
    def _preflight(
            compilation: GroundedAnswerReferenceCompilation,
            selection: object,
            alias: AliasRelationRuntime,
            ) -> None:
        """零写核对全部 direct surface 与 expected antecedent。"""
        connector = compilation.connector
        policies = {
            item.slot: item
            for template in connector.runtime_policy.templates
            for item in template.surface
        }
        directives = {
            item.slot: item
            for template in connector.registry.templates
            for item in template.surface
        }
        aliases = tuple(
            item
            for sentence in compilation.sentences
            for item in sentence.aliases
        )
        if (set(policies) != {item.slot for item in aliases}
                or set(directives) != {item.slot for item in aliases}):
            raise GroundedAnswerReferenceRunLocalFactoryError(
                "reference preflight slot 覆盖漂移")
        for requirement in aliases:
            policy = policies[requirement.slot]
            if policy.surface_budget is None:
                raise GroundedAnswerReferenceRunLocalFactoryError(
                    "reference preflight 缺 surface budget")
            proposal = alias.preview_surface(
                requirement.filler,
                connector.registry.templates[0].language_branch,
                budget=policy.surface_budget,
                allowed_prefix_steps=(
                    directives[requirement.slot].surface_prefix_steps),
            )
            selected = proposal.result.selected
            if selected is None or selected.value != requirement.representation:
                raise GroundedAnswerReferenceRunLocalFactoryError(
                    "reference alias factory 未提供唯一 direct Representation")
        structure = connector.structure_planner().plan(selection)
        if len(structure.syntax.sentences) != 2:
            raise GroundedAnswerReferenceRunLocalFactoryError(
                "reference structure 未形成两个 sentence")
        if compilation.strategy == "ANTECEDENT_REFERENCE":
            if len(structure.syntax.anaphora) != 1:
                raise GroundedAnswerReferenceRunLocalFactoryError(
                    "antecedent strategy 未形成唯一 requirement")
            policy = policies[compilation.reference_slot]
            if policy.reference_budget is None:
                raise GroundedAnswerReferenceRunLocalFactoryError(
                    "antecedent strategy 缺 reference budget")
            expected = compilation.claims[0].candidate.proposition.template
            proposal = alias.preview_reference(
                compilation.reference_origin,
                target_kinds=(expected.object_kind,),
                budget=policy.reference_budget,
            )
            if (len(proposal.result.options) != 1
                    or proposal.result.selected.value != expected):
                raise GroundedAnswerReferenceRunLocalFactoryError(
                    "reference alias factory 未唯一命中 expected antecedent")

    def build(
            self,
            request: GroundedAnswerReferenceRunLocalBuild,
            ) -> GroundedAnswerReferenceRunLocalInstallation:
        """装配并冻结双 candidate G-00 至 G-04；不形成 GG-01 choice。"""
        if not isinstance(request, GroundedAnswerReferenceRunLocalBuild):
            raise TypeError("reference factory build request 类型错误")
        compilation = request.compilation
        candidates = tuple(item.candidate for item in compilation.claims)
        components = self.components
        planning = compilation.planning
        selection = components.selector.select(planning)
        if set(selection.selected_candidate_keys) != {
                item.stable_key() for item in candidates}:
            raise GroundedAnswerReferenceRunLocalFactoryError(
                "reference selector 未精确选择全部 candidates")
        if request.parser_protocol.answer_stance != selection.stance:
            raise GroundedAnswerReferenceRunLocalFactoryError(
                "reference parser stance 与 G-01 selection 漂移")
        alias = components.alias_factory.build(compilation)
        if not isinstance(alias, AliasRelationRuntime):
            raise TypeError("reference alias factory 返回类型错误")
        self._preflight(compilation, selection, alias)
        order = install_grounded_answer_reference_order(
            compilation, self.lifecycle)
        connector = compilation.connector
        structure_planner = connector.structure_planner()
        surface_runtime = GenerationSurfaceRuntime(alias)
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
                    protocol, structure_protocol, selector, structure_planner),
            ),
            GenerationLayerRegistration(
                protocol.proposition_layer,
                GenerationPropositionLayerResolver(
                    protocol, structure_protocol, selector, structure_planner),
            ),
            GenerationLayerRegistration(
                protocol.syntax_layer,
                GenerationSyntaxLayerResolver(
                    protocol, structure_protocol, selector, structure_planner),
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
        parser_catalog = build_grounded_answer_reference_parser_catalog(
            compilation, components.renderer_identity)
        parser = GroundedAnswerReferenceSurfaceParser(
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
            planning,
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
        return GroundedAnswerReferenceRunLocalInstallation(
            compilation,
            alias,
            order,
            parser_catalog,
            parser,
            executor,
            runtime,
        )
__all__ = [
    "GroundedAnswerReferenceAliasRuntimeFactory",
    "GroundedAnswerReferenceRunLocalBuild",
    "GroundedAnswerReferenceRunLocalFactory",
    "GroundedAnswerReferenceRunLocalFactoryError",
    "GroundedAnswerReferenceRunLocalInstallation",
]
