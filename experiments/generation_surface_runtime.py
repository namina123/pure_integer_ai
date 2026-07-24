"""G-03 surface preview、R-01 原子采用和 G-00 第六层接线。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from pure_integer_ai.cognition.shared.generation_content import (
    AnswerContentSelector,
    ContentArtifactAttachment,
)
from pure_integer_ai.cognition.shared.generation_plan import (
    GenerationLayerDecision,
    GenerationLayerResult,
    GenerationPlanProtocol,
    GenerationPlanningRequest,
)
from pure_integer_ai.cognition.shared.generation_structure_execution import (
    GenerationStructureExecutionPlan,
    GenerationStructureExecutionPlanner,
    GenerationStructureExecutionRequest,
)
from pure_integer_ai.cognition.shared.generation_structure_plan import (
    GenerationSentenceInstance,
    GenerationStructurePlan,
    GenerationStructurePlanner,
)
from pure_integer_ai.cognition.shared.generation_surface import (
    GenerationSurfaceAttribution,
    GenerationSurfaceSentenceAttribution,
    GenerationSurfacePlan,
    GenerationSurfacePreview,
    GenerationSurfaceProtocol,
    GenerationSurfaceRequest,
    SurfaceAdoption,
    SurfaceSlotDirective,
    SurfaceSlotPreview,
)
from pure_integer_ai.cognition.shared.identity import ObjectIdentity
from pure_integer_ai.cognition.shared.relation_use import RelationUseContext
from pure_integer_ai.experiments.alias_relation_runtime import (
    AliasRelationRuntime,
    AliasResolutionUse,
)


def _packed(key: tuple[int, ...]) -> tuple[int, ...]:
    """为可变长稳定键增加长度边界。"""
    return len(key), *key


class StructureExecutionRequestMapper(Protocol):
    """为完整 G-02 structure plan 注入逐句 L-05B1 搜索预算。"""

    def build(
            self, structure: GenerationStructurePlan,
            ) -> GenerationStructureExecutionRequest:
        """返回与 structure.syntax 完全一致的执行请求。"""
        ...


class SurfaceDirectiveMapper(Protocol):
    """为 L-05B1 结果中的全部 planned slot 注入 emit/silent 指令。"""

    def plan(
            self,
            structure: GenerationStructurePlan,
            execution: GenerationStructureExecutionPlan,
            branch: ObjectIdentity,
            ) -> tuple[SurfaceSlotDirective, ...]:
        """返回逐 slot 指令、R-01 预算、use key 和 mapper trace。"""
        ...


class SurfaceAttributionMapper(Protocol):
    """为当前 G-02 结构返回可选的一等理论/Hypothesis 归属。"""

    def attribution(
            self,
            structure: GenerationStructurePlan,
            ) -> GenerationSurfaceAttribution | None:
        """返回显式归属；无候选生命周期的 generic surface 可返回空。"""
        ...


class SentenceSurfaceAttributionMapper(Protocol):
    """按运行期句实例返回 connector 理论、Hypothesis 和 purpose 归属。"""

    def attributions_for(
            self,
            structure: GenerationStructurePlan,
            ) -> tuple[GenerationSurfaceSentenceAttribution, ...]:
        """返回精确覆盖每个 planned sentence 的逐句归属。"""
        ...


class GenerationSurfaceRequestBuilder(Protocol):
    """把重建的 G-02 structure plan 汇合为完整 G-03 请求。"""

    def build(
            self, structure: GenerationStructurePlan,
            ) -> GenerationSurfaceRequest:
        """只从 typed plan 和注入 mapper 构造 surface 请求。"""
        ...


class TypedGenerationSurfaceRequestBuilder:
    """执行 L-05B1 并调用注入 directive mapper，不读取 legacy 生成链。"""

    def __init__(
            self,
            protocol: GenerationSurfaceProtocol,
            execution_planner: GenerationStructureExecutionPlanner,
            execution_requests: StructureExecutionRequestMapper,
            directives: SurfaceDirectiveMapper,
            attribution_mapper: SurfaceAttributionMapper | None = None,
            sentence_attribution_mapper: SentenceSurfaceAttributionMapper
            | None = None,
            ) -> None:
        if not isinstance(protocol, GenerationSurfaceProtocol):
            raise TypeError("surface request builder protocol 类型错误")
        if not isinstance(
                execution_planner, GenerationStructureExecutionPlanner):
            raise TypeError("surface request builder execution planner 类型错误")
        if not hasattr(execution_requests, "build"):
            raise TypeError("execution request mapper 必须实现 build")
        if not hasattr(directives, "plan"):
            raise TypeError("surface directive mapper 必须实现 plan")
        if (attribution_mapper is not None
                and not hasattr(attribution_mapper, "attribution")):
            raise TypeError("surface attribution mapper 必须实现 attribution")
        if (sentence_attribution_mapper is not None
                and not hasattr(
                    sentence_attribution_mapper,
                    "attributions_for",
                )):
            raise TypeError("surface sentence attribution mapper 必须实现 attributions_for")
        self._protocol = protocol
        self._execution_planner = execution_planner
        self._execution_requests = execution_requests
        self._directives = directives
        self._attribution_mapper = attribution_mapper
        self._sentence_attribution_mapper = sentence_attribution_mapper

    def build(
            self, structure: GenerationStructurePlan,
            ) -> GenerationSurfaceRequest:
        """按目标分支执行结构顺序，并要求 mapper 完整覆盖全部 slot。"""
        if not isinstance(structure, GenerationStructurePlan):
            raise TypeError("surface request builder structure 类型错误")
        branch = structure.selection.request.goal.target_branch
        if branch is None:
            raise ValueError("G-03 request builder 缺目标 LanguageBranch")
        execution_request = self._execution_requests.build(structure)
        if not isinstance(
                execution_request, GenerationStructureExecutionRequest):
            raise TypeError("execution request mapper 返回类型错误")
        if execution_request.syntax != structure.syntax:
            raise ValueError("execution request mapper 替换了 G-02 SyntaxPlan")
        execution = self._execution_planner.execute(execution_request)
        directives = ()
        if execution.complete:
            directives = self._directives.plan(structure, execution, branch)
            if not isinstance(directives, tuple):
                raise TypeError("surface directive mapper 必须返回 tuple")
        attribution = None
        if self._attribution_mapper is not None:
            attribution = self._attribution_mapper.attribution(structure)
            if (attribution is not None
                    and not isinstance(
                        attribution, GenerationSurfaceAttribution)):
                raise TypeError("surface attribution mapper 返回类型错误")
        sentence_attributions = ()
        if self._sentence_attribution_mapper is not None:
            sentence_attributions = (
                self._sentence_attribution_mapper.attributions_for(structure))
            if (not isinstance(sentence_attributions, tuple)
                    or any(not isinstance(
                        item, GenerationSurfaceSentenceAttribution)
                           for item in sentence_attributions)):
                raise TypeError("surface sentence attribution mapper 返回类型错误")
        return GenerationSurfaceRequest(
            self._protocol,
            structure,
            execution,
            branch,
            directives,
            attribution,
            sentence_attributions,
        )


@dataclass(frozen=True)
class GenerationSurfaceRun:
    """一次 G-03 无写入 preview 和可选完整采用计划。"""

    preview: GenerationSurfacePreview
    plan: GenerationSurfacePlan | None

    def __post_init__(self) -> None:
        if not isinstance(self.preview, GenerationSurfacePreview):
            raise TypeError("surface run preview 类型错误")
        if self.plan is not None:
            if not isinstance(self.plan, GenerationSurfacePlan):
                raise TypeError("surface run plan 类型错误")
            if self.plan.preview != self.preview:
                raise ValueError("surface run plan 未绑定当前 preview")
        if self.preview.complete != (self.plan is not None):
            raise ValueError("surface run complete 状态与 plan 不一致")

    @property
    def complete(self) -> bool:
        """返回是否完成全部 slot 并提交采用账。"""
        return self.plan is not None

    def stable_key(self) -> tuple[int, ...]:
        """返回 preview 和可选完整计划键。"""
        result = [
            *_packed(self.preview.stable_key()),
            0 if self.plan is None else 1,
        ]
        if self.plan is not None:
            result.extend(_packed(self.plan.stable_key()))
        return tuple(result)


class GenerationSurfaceRuntime:
    """先完成全部 R-01 preview，再一次性提交词形和照应采用账。"""

    def __init__(self, alias: AliasRelationRuntime) -> None:
        if not isinstance(alias, AliasRelationRuntime):
            raise TypeError("surface runtime alias 类型错误")
        self._alias = alias

    def preview(
            self, request: GenerationSurfaceRequest,
            ) -> GenerationSurfacePreview:
        """按 L-05B1 顺序解析全部 slot；首个失败后未来 slot 零查询。"""
        if not isinstance(request, GenerationSurfaceRequest):
            raise TypeError("surface runtime request 类型错误")
        protocol = request.protocol
        if not request.execution.complete:
            return GenerationSurfacePreview(
                request, protocol.structure_incomplete_reason, ())
        directives = request.directive_map()
        antecedents = request.antecedent_map()
        registered_prefix_steps = set(
            self._alias.selector.protocol.surface_prefix_steps())
        slots: list[SurfaceSlotPreview] = []
        for sentence, value in request.ordered_values():
            key = sentence, value.slot
            directive = directives[key]
            if directive.action == protocol.silent_action:
                slots.append(SurfaceSlotPreview(directive, value))
                continue
            if any(
                    step not in registered_prefix_steps
                    for step in directive.surface_prefix_steps):
                raise ValueError("surface directive 使用了未注册 R-01 prefix step")
            if directive.surface_budget is None:
                raise RuntimeError("emit directive 丢失已核验 surface budget")
            antecedent_key = ()
            antecedent = None
            reference = None
            expected = antecedents.get(key)
            if expected is not None:
                if directive.reference_budget is None:
                    raise RuntimeError("anaphora directive 丢失已核验 reference budget")
                antecedent_key, antecedent = expected
                reference = self._alias.preview_reference(
                    value.filler,
                    target_kinds=(antecedent.object_kind,),
                    budget=directive.reference_budget,
                )
                current = SurfaceSlotPreview(
                    directive,
                    value,
                    antecedent_key,
                    antecedent,
                    reference,
                )
                slots.append(current)
                if not reference.result.options:
                    return GenerationSurfacePreview(
                        request,
                        protocol.reference_missing_reason,
                        tuple(slots),
                    )
                if len(reference.result.options) > 1:
                    return GenerationSurfacePreview(
                        request,
                        protocol.reference_ambiguous_reason,
                        tuple(slots),
                    )
                if reference.result.selected.value != antecedent:
                    return GenerationSurfacePreview(
                        request,
                        protocol.reference_mismatch_reason,
                        tuple(slots),
                    )
                slots.pop()
            surface = self._alias.preview_surface(
                value.filler,
                request.branch,
                budget=directive.surface_budget,
                allowed_prefix_steps=directive.surface_prefix_steps,
            )
            representation = (
                None
                if surface.result.selected is None
                else surface.result.selected.value
            )
            slots.append(SurfaceSlotPreview(
                directive,
                value,
                antecedent_key,
                antecedent,
                reference,
                surface,
                representation,
            ))
            if not surface.result.options:
                return GenerationSurfacePreview(
                    request,
                    protocol.surface_missing_reason,
                    tuple(slots),
                )
            if len(surface.result.options) > 1:
                return GenerationSurfacePreview(
                    request,
                    protocol.surface_ambiguous_reason,
                    tuple(slots),
                )
        return GenerationSurfacePreview(
            request, protocol.complete_reason, tuple(slots))

    def plan(
            self, request: GenerationSurfaceRequest,
            ) -> GenerationSurfaceRun:
        """preview 全部成功后原子提交所有 reference/surface proposal。"""
        preview = self.preview(request)
        if not preview.complete:
            return GenerationSurfaceRun(preview, None)
        return GenerationSurfaceRun(preview, self.commit(preview))

    def commit(
            self, preview: GenerationSurfacePreview,
            ) -> GenerationSurfacePlan:
        """原子提交一个完整 preview 的全部 R-01 proposal，并绑定实际 use trace。"""
        if not isinstance(preview, GenerationSurfacePreview):
            raise TypeError("surface commit preview 类型错误")
        if not preview.complete:
            raise ValueError("失败 surface preview 不得提交部分采用账")
        sentence_attributions = preview.request.sentence_attribution_map()
        legacy_attribution = preview.request.attribution
        goal = preview.request.structure.selection.request.goal

        def use_context_for(sentence) -> RelationUseContext | None:
            """按句实例构造独立 Core Use context，保留旧单句兼容归属。"""
            attribution = sentence_attributions.get(sentence)
            if attribution is not None:
                return RelationUseContext(
                    goal.source,
                    goal.scope,
                    attribution.theory,
                    attribution.purpose,
                    attribution.hypothesis,
                    sentence.stable_key(),
                )
            if legacy_attribution is None:
                return None
            if isinstance(sentence, GenerationSentenceInstance):
                return RelationUseContext(
                    goal.source,
                    goal.scope,
                    legacy_attribution.theory,
                    legacy_attribution.purpose,
                    legacy_attribution.hypothesis,
                    sentence.stable_key(),
                )
            return RelationUseContext(
                goal.source,
                goal.scope,
                legacy_attribution.theory,
                legacy_attribution.purpose,
            )

        commit_requests = []
        metadata = []
        for slot in preview.slots:
            item_context = use_context_for(slot.directive.sentence)
            if slot.reference is not None:
                commit_requests.append((
                    slot.reference,
                    slot.directive.reference_use_key,
                    item_context,
                ))
                metadata.append((
                    slot.directive.sentence,
                    slot.value.slot,
                    slot.reference,
                    slot.directive.reference_use_key,
                ))
            if slot.surface is not None:
                commit_requests.append((
                    slot.surface,
                    slot.directive.surface_use_key,
                    item_context,
                ))
                metadata.append((
                    slot.directive.sentence,
                    slot.value.slot,
                    slot.surface,
                    slot.directive.surface_use_key,
                ))
        uses: tuple[AliasResolutionUse, ...] = ()
        if commit_requests:
            uses = self._alias.commit_many(tuple(commit_requests))
        if len(uses) != len(metadata):
            raise RuntimeError("R-01 批量采用数量与 surface proposal 不一致")
        adoptions = tuple(
            SurfaceAdoption(
                sentence,
                slot,
                proposal,
                use_key,
                use.stable_key(),
            )
            for (sentence, slot, proposal, use_key), use
            in zip(metadata, uses)
        )
        return GenerationSurfacePlan(preview, adoptions)


class GenerationSurfaceLayerResolver:
    """独立重算 G-01/G-02，再执行 typed G-03 作为 G-00 第六层。"""

    def __init__(
            self,
            planner_protocol: GenerationPlanProtocol,
            selector: AnswerContentSelector,
            structure_planner: GenerationStructurePlanner,
            request_builder: GenerationSurfaceRequestBuilder,
            runtime: GenerationSurfaceRuntime,
            artifacts: Sequence[ContentArtifactAttachment] = (),
            *,
            commit: bool = True,
            ) -> None:
        if not isinstance(planner_protocol, GenerationPlanProtocol):
            raise TypeError("surface layer planner protocol 类型错误")
        if not isinstance(selector, AnswerContentSelector):
            raise TypeError("surface layer selector 类型错误")
        if not isinstance(structure_planner, GenerationStructurePlanner):
            raise TypeError("surface layer structure planner 类型错误")
        if not hasattr(request_builder, "build"):
            raise TypeError("surface request builder 必须实现 build")
        if not isinstance(runtime, GenerationSurfaceRuntime):
            raise TypeError("surface layer runtime 类型错误")
        if not isinstance(artifacts, Sequence):
            raise TypeError("surface layer artifacts 必须是 Sequence")
        if type(commit) is not bool:
            raise TypeError("surface layer commit 必须是严格 bool")
        self._planner_protocol = planner_protocol
        self._selector = selector
        self._structure_planner = structure_planner
        self._request_builder = request_builder
        self._runtime = runtime
        self._artifacts = tuple(artifacts)
        self._commit = commit

    def resolve(
            self,
            request: GenerationPlanningRequest,
            prior: tuple[GenerationLayerResult, ...],
            ) -> GenerationLayerDecision:
        """核验前五层独立重算结果，并返回 complete 或分型 failed surface。"""
        if len(prior) != 5:
            raise ValueError("surface layer 必须接收 G-00 前五层结果")
        expected_layers = self._planner_protocol.layers()[:5]
        if tuple(item.layer for item in prior) != expected_layers:
            raise ValueError("surface layer 上游顺序不匹配")
        if any(item.outcome != self._planner_protocol.complete for item in prior):
            raise ValueError("surface layer 只接受 complete 上游")
        selection = self._selector.select(request, self._artifacts)
        discourse = self._structure_planner.plan_discourse(selection)
        propositions = self._structure_planner.plan_propositions(
            selection, discourse)
        syntax = self._structure_planner.plan_syntax(
            selection, discourse, propositions)
        expected_payloads = (
            selection.stable_key(),
            selection.stable_key(),
            discourse.stable_key(),
            propositions.stable_key(),
            syntax.stable_key(),
        )
        if tuple(item.payload for item in prior) != expected_payloads:
            raise ValueError("surface layer 独立重算结果与上游 payload 不一致")
        structure = GenerationStructurePlan(
            selection, discourse, propositions, syntax)
        surface_request = self._request_builder.build(structure)
        if not isinstance(surface_request, GenerationSurfaceRequest):
            raise TypeError("surface request builder 返回类型错误")
        if surface_request.structure != structure:
            raise ValueError("surface request builder 替换了重建 structure plan")
        if self._commit:
            run = self._runtime.plan(surface_request)
            preview = run.preview
            artifact = run.plan if run.plan is not None else preview
            trace = run.stable_key()
        else:
            preview = self._runtime.preview(surface_request)
            artifact = preview
            trace = preview.stable_key()
        outcome = (
            self._planner_protocol.complete
            if preview.complete else self._planner_protocol.failed
        )
        return GenerationLayerDecision(
            self._planner_protocol.surface_layer,
            outcome,
            preview.reason,
            selection.selected_candidate_keys,
            artifact.stable_key(),
            trace,
            artifact,
        )


__all__ = [
    "GenerationSurfaceLayerResolver",
    "GenerationSurfaceRequestBuilder",
    "GenerationSurfaceRun",
    "GenerationSurfaceRuntime",
    "StructureExecutionRequestMapper",
    "SurfaceAttributionMapper",
    "SurfaceDirectiveMapper",
    "TypedGenerationSurfaceRequestBuilder",
]
