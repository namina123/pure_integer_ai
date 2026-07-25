"""G-03 typed surface 请求、预览和采用计划的纯运行期契约。

本模块不查询图也不写 use。它只约束 G-02/L-05B1 的有序 slot 如何显式选择
emit 或 silent，并要求词形与照应结果可回溯到 R-01 完整 route proposal。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.alias_resolution import (
    AliasResolutionProposal,
    AliasRouteSearchBudget,
    ReferenceRouteDiscovery,
    SurfaceRouteDiscovery,
)
from pure_integer_ai.cognition.shared.generation_structure_execution import (
    GenerationStructureExecutionPlan,
)
from pure_integer_ai.cognition.shared.generation_structure_plan import (
    GenerationSentenceInstance,
    GenerationStructurePlan,
    generation_sentence_address_key,
)
from pure_integer_ai.cognition.shared.hypothesis import HypothesisKey
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_LANGUAGE_BRANCH,
    OBJECT_MINIMAL_INSTRUCTION,
    OBJECT_STRUCTURE_CONCEPT,
    ObjectIdentity,
)
from pure_integer_ai.cognition.shared.structure_order_consumer import (
    StructureSlotValue,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int


def _packed(key: tuple[int, ...]) -> tuple[int, ...]:
    """为可变长稳定键增加长度边界。"""
    return len(key), *key


def _identity(
        value: ObjectIdentity, *, label: str, kind: int | None = None,
        ) -> ObjectIdentity:
    """核验一等对象及可选宿主对象类型。"""
    if not isinstance(value, ObjectIdentity):
        raise TypeError(f"{label} 必须是 ObjectIdentity")
    if kind is not None and value.object_kind != kind:
        raise ValueError(f"{label} 对象类型不匹配")
    return value


def _strict_key(
        value: tuple[int, ...], *, label: str, allow_empty: bool = False,
        ) -> tuple[int, ...]:
    """核验运行期 trace/use key 是严格整数 tuple。"""
    if not isinstance(value, tuple):
        raise TypeError(f"{label} 必须是 tuple")
    if not value:
        if allow_empty:
            return value
        raise ValueError(f"{label} 不能为空")
    assert_int(*value, _where=label)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{label} 必须使用严格整数")
    return value


def _sentence_address(
        value: ObjectIdentity | GenerationSentenceInstance,
        *,
        label: str,
        ) -> ObjectIdentity | GenerationSentenceInstance:
    """核验 surface 地址仍指向模板句式或来源化运行期句实例。"""
    generation_sentence_address_key(value, label=label)
    return value


@dataclass(frozen=True)
class GenerationSurfaceAttribution:
    """把一次 surface artifact 显式归属到一等理论和完整 Hypothesis。"""

    theory: ObjectIdentity
    hypothesis: HypothesisKey
    purpose: ObjectIdentity

    def __post_init__(self) -> None:
        _identity(self.theory, label="surface attribution theory")
        if not isinstance(self.hypothesis, HypothesisKey):
            raise TypeError("surface attribution hypothesis 类型错误")
        _identity(
            self.purpose,
            label="surface attribution purpose",
            kind=OBJECT_MINIMAL_INSTRUCTION,
        )

    def stable_key(self) -> tuple[int, ...]:
        """返回理论对象和来源化 Hypothesis 的完整稳定键。"""
        return (
            *_packed(self.theory.stable_key()),
            *_packed(self.hypothesis.stable_key()),
            *_packed(self.purpose.stable_key()),
        )


@dataclass(frozen=True)
class GenerationSurfaceSentenceAttribution:
    """把一个运行期句实例精确归属到 connector 理论和 Hypothesis。"""

    sentence: GenerationSentenceInstance
    theory: ObjectIdentity
    hypothesis: HypothesisKey
    purpose: ObjectIdentity

    def __post_init__(self) -> None:
        """拒绝把模板句式或无候选的抽象归属充作逐句反馈。"""
        if not isinstance(self.sentence, GenerationSentenceInstance):
            raise TypeError("surface sentence attribution 必须绑定运行期句实例")
        _identity(self.theory, label="surface sentence attribution theory")
        if not isinstance(self.hypothesis, HypothesisKey):
            raise TypeError("surface sentence attribution hypothesis 类型错误")
        _identity(
            self.purpose,
            label="surface sentence attribution purpose",
            kind=OBJECT_MINIMAL_INSTRUCTION,
        )

    def stable_key(self) -> tuple[int, ...]:
        """返回句实例、理论、Hypothesis 和 purpose 的完整键。"""
        return (
            *_packed(self.sentence.stable_key()),
            *_packed(self.theory.stable_key()),
            *_packed(self.hypothesis.stable_key()),
            *_packed(self.purpose.stable_key()),
        )

@dataclass(frozen=True)
class GenerationSurfaceProtocol:
    """注入 emit/silent 动作和 surface 失败原因身份。"""

    emit_action: ObjectIdentity
    silent_action: ObjectIdentity
    complete_reason: ObjectIdentity
    structure_incomplete_reason: ObjectIdentity
    surface_missing_reason: ObjectIdentity
    surface_ambiguous_reason: ObjectIdentity
    reference_missing_reason: ObjectIdentity
    reference_ambiguous_reason: ObjectIdentity
    reference_mismatch_reason: ObjectIdentity

    def __post_init__(self) -> None:
        identities = self.actions() + self.reasons()
        if len(set(identities)) != len(identities):
            raise ValueError("surface action/reason 身份必须互不相同")
        for identity in identities:
            _identity(
                identity,
                label="surface action/reason",
                kind=OBJECT_MINIMAL_INSTRUCTION,
            )

    def actions(self) -> tuple[ObjectIdentity, ...]:
        """返回 emit 和 silent 两个注入动作。"""
        return self.emit_action, self.silent_action

    def reasons(self) -> tuple[ObjectIdentity, ...]:
        """返回完成和六类分型失败原因。"""
        return (
            self.complete_reason,
            self.structure_incomplete_reason,
            self.surface_missing_reason,
            self.surface_ambiguous_reason,
            self.reference_missing_reason,
            self.reference_ambiguous_reason,
            self.reference_mismatch_reason,
        )

    def stable_key(self) -> tuple[int, ...]:
        """返回全部动作和原因一等身份。"""
        result: list[int] = []
        for identity in self.actions() + self.reasons():
            result.extend(_packed(identity.stable_key()))
        return tuple(result)


@dataclass(frozen=True)
class SurfaceSlotDirective:
    """为一个 planned slot 注入 emit/silent 和 R-01 搜索/采用预算。"""

    sentence: ObjectIdentity | GenerationSentenceInstance
    slot: ObjectIdentity
    action: ObjectIdentity
    instruction: ObjectIdentity
    trace: tuple[int, ...]
    surface_prefix_steps: tuple[ObjectIdentity, ...]
    surface_budget: AliasRouteSearchBudget | None = None
    surface_use_key: tuple[int, ...] = ()
    reference_budget: AliasRouteSearchBudget | None = None
    reference_use_key: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        _sentence_address(
            self.sentence,
            label="surface directive sentence",
        )
        _identity(
            self.slot,
            label="surface directive slot",
            kind=OBJECT_STRUCTURE_CONCEPT,
        )
        _identity(
            self.action,
            label="surface directive action",
            kind=OBJECT_MINIMAL_INSTRUCTION,
        )
        _identity(
            self.instruction,
            label="surface directive instruction",
            kind=OBJECT_MINIMAL_INSTRUCTION,
        )
        _strict_key(self.trace, label="surface directive trace")
        if not isinstance(self.surface_prefix_steps, tuple):
            raise TypeError("surface directive prefix steps 必须是 tuple")
        if len(set(self.surface_prefix_steps)) != len(self.surface_prefix_steps):
            raise ValueError("surface directive prefix step 不得重复")
        for step in self.surface_prefix_steps:
            _identity(
                step,
                label="surface directive prefix step",
                kind=OBJECT_MINIMAL_INSTRUCTION,
            )
        object.__setattr__(self, "surface_prefix_steps", tuple(sorted(
            self.surface_prefix_steps, key=ObjectIdentity.stable_key)))
        if (self.surface_budget is not None
                and not isinstance(self.surface_budget, AliasRouteSearchBudget)):
            raise TypeError("surface directive surface_budget 类型错误")
        if (self.reference_budget is not None
                and not isinstance(self.reference_budget, AliasRouteSearchBudget)):
            raise TypeError("surface directive reference_budget 类型错误")
        _strict_key(
            self.surface_use_key,
            label="surface directive surface_use_key",
            allow_empty=True,
        )
        _strict_key(
            self.reference_use_key,
            label="surface directive reference_use_key",
            allow_empty=True,
        )

    def stable_key(self) -> tuple[int, ...]:
        """返回 slot、动作、预算、use key 和 mapper trace。"""
        result = [
            *_packed(generation_sentence_address_key(
                self.sentence,
                label="surface directive sentence",
            )),
            *_packed(self.slot.stable_key()),
            *_packed(self.action.stable_key()),
            *_packed(self.instruction.stable_key()),
            *_packed(self.trace),
            len(self.surface_prefix_steps),
            *(value for step in self.surface_prefix_steps
              for value in _packed(step.stable_key())),
            0 if self.surface_budget is None else 1,
        ]
        if self.surface_budget is not None:
            result.extend(self.surface_budget.stable_key())
        result.extend(_packed(self.surface_use_key))
        result.append(0 if self.reference_budget is None else 1)
        if self.reference_budget is not None:
            result.extend(self.reference_budget.stable_key())
        result.extend(_packed(self.reference_use_key))
        return tuple(result)


@dataclass(frozen=True)
class GenerationSurfaceRequest:
    """汇合 G-02、L-05B1、目标分支和逐 slot surface 指令。"""

    protocol: GenerationSurfaceProtocol
    structure: GenerationStructurePlan
    execution: GenerationStructureExecutionPlan
    branch: ObjectIdentity
    directives: tuple[SurfaceSlotDirective, ...]
    attribution: GenerationSurfaceAttribution | None = None
    sentence_attributions: tuple[GenerationSurfaceSentenceAttribution, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.protocol, GenerationSurfaceProtocol):
            raise TypeError("surface request protocol 类型错误")
        if not isinstance(self.structure, GenerationStructurePlan):
            raise TypeError("surface request structure 类型错误")
        if not isinstance(self.execution, GenerationStructureExecutionPlan):
            raise TypeError("surface request execution 类型错误")
        if self.execution.request.syntax != self.structure.syntax:
            raise ValueError("surface request 的 G-02/L-05B1 SyntaxPlan 不一致")
        _identity(
            self.branch,
            label="surface request branch",
            kind=OBJECT_LANGUAGE_BRANCH,
        )
        target_branch = self.structure.selection.request.goal.target_branch
        if target_branch is None:
            raise ValueError("G-03 surface request 缺少目标 LanguageBranch")
        if target_branch != self.branch:
            raise ValueError("surface request branch 与 generation goal 不一致")
        if not isinstance(self.directives, tuple) or any(
                not isinstance(item, SurfaceSlotDirective)
                for item in self.directives):
            raise TypeError("surface request directives 类型错误")
        if (self.attribution is not None
                and not isinstance(
                    self.attribution, GenerationSurfaceAttribution)):
            raise TypeError("surface request attribution 类型错误")
        if (not isinstance(self.sentence_attributions, tuple)
                or any(not isinstance(item, GenerationSurfaceSentenceAttribution)
                       for item in self.sentence_attributions)):
            raise TypeError("surface request sentence_attributions 类型错误")
        if not self.execution.complete:
            if self.directives or self.sentence_attributions:
                raise ValueError("incomplete structure execution 不得提前规划 surface directive")
            return
        by_key = {(item.sentence, item.slot): item for item in self.directives}
        if len(by_key) != len(self.directives):
            raise ValueError("同一 sentence/slot 不得重复注入 surface directive")
        expected_keys = tuple(
            (sentence.address, value.slot)
            for sentence in self.structure.syntax.sentences
            for value in sentence.values
        )
        if set(by_key) != set(expected_keys):
            raise ValueError("surface directive 必须精确覆盖全部 planned slot value")
        anaphora = {
            (item.address, item.slot): item
            for item in self.structure.syntax.anaphora
        }
        use_keys: list[tuple[int, ...]] = []
        emitted_sentences: set[ObjectIdentity] = set()
        for key in expected_keys:
            directive = by_key[key]
            if directive.action not in self.protocol.actions():
                raise ValueError("surface directive action 未在 protocol 注册")
            requirement = anaphora.get(key)
            if directive.action == self.protocol.emit_action:
                if (directive.surface_budget is None
                        or not directive.surface_use_key):
                    raise ValueError("emit directive 必须注入 surface 预算和 use key")
                emitted_sentences.add(directive.sentence)
                use_keys.append(directive.surface_use_key)
                if requirement is None:
                    if (directive.reference_budget is not None
                            or directive.reference_use_key):
                        raise ValueError("非 anaphora slot 不得注入 reference 采用")
                else:
                    if (directive.reference_budget is None
                            or not directive.reference_use_key):
                        raise ValueError("anaphora emit 必须注入 reference 预算和 use key")
                    use_keys.append(directive.reference_use_key)
            else:
                if requirement is not None:
                    raise ValueError("anaphora slot 不得声明 silent")
                if any((
                        bool(directive.surface_prefix_steps),
                        directive.surface_budget is not None,
                        bool(directive.surface_use_key),
                        directive.reference_budget is not None,
                        bool(directive.reference_use_key))):
                    raise ValueError("silent directive 不得携带 R-01 预算或 use key")
        if len(set(use_keys)) != len(use_keys):
            raise ValueError("surface request 的全部 R-01 use key 必须唯一")
        expected_sentences = {
            item.address for item in self.structure.syntax.sentences}
        if emitted_sentences != expected_sentences:
            raise ValueError("每个 planned sentence 必须至少包含一个 emit slot")
        sentence_attributions = {
            item.sentence: item for item in self.sentence_attributions}
        if len(sentence_attributions) != len(self.sentence_attributions):
            raise ValueError("同一运行期句实例不得重复归属")
        if sentence_attributions:
            if set(sentence_attributions) != expected_sentences:
                raise ValueError("逐句归属必须精确覆盖全部 planned sentence")
            if any(not isinstance(item, GenerationSentenceInstance)
                   for item in expected_sentences):
                raise ValueError("逐句归属只接受运行期句实例地址")
            if self.attribution is not None:
                if len(expected_sentences) != 1:
                    raise ValueError("多句 surface 不得复用整次 attribution")
                sentence_attribution = next(iter(sentence_attributions.values()))
                if (self.attribution.theory != sentence_attribution.theory
                        or self.attribution.hypothesis
                        != sentence_attribution.hypothesis
                        or self.attribution.purpose
                        != sentence_attribution.purpose):
                    raise ValueError("旧整次 attribution 与逐句归属不一致")
        elif self.attribution is not None and len(expected_sentences) != 1:
            raise ValueError("多句 surface 不得复用整次 attribution")
        object.__setattr__(self, "directives", tuple(
            by_key[key] for key in expected_keys))
        object.__setattr__(self, "sentence_attributions", tuple(
            sentence_attributions[item.address]
            for item in self.structure.syntax.sentences
            if item.address in sentence_attributions
        ))

    def directive_map(self) -> dict[
            tuple[ObjectIdentity, ObjectIdentity], SurfaceSlotDirective]:
        """按 sentence/slot 返回完整 directive 映射。"""
        return {
            (item.sentence, item.slot): item for item in self.directives}

    def value_map(self) -> dict[
            tuple[ObjectIdentity, ObjectIdentity], StructureSlotValue]:
        """按 sentence/slot 返回 G-02 planned value。"""
        return {
            (sentence.address, value.slot): value
            for sentence in self.structure.syntax.sentences
            for value in sentence.values
        }

    def antecedent_map(self) -> dict[
            tuple[ObjectIdentity, ObjectIdentity],
            tuple[tuple[int, ...], ObjectIdentity]]:
        """把 anaphora slot 映射到 antecedent candidate 和 Proposition template。"""
        propositions = {
            item.candidate_key: item.proposition.template
            for item in self.structure.propositions.propositions
        }
        return {
            (item.address, item.slot): (
                item.antecedent_candidate_key,
                propositions[item.antecedent_candidate_key],
            )
            for item in self.structure.syntax.anaphora
        }

    def ordered_values(self) -> tuple[
            tuple[ObjectIdentity, StructureSlotValue], ...]:
        """按 L-05B1 accepted 线性化结果返回句子和值；未完成时返回空。"""
        if not self.execution.complete:
            return ()
        return tuple(
            (sentence.obligation.address, value)
            for sentence in self.execution.sentences
            for value in sentence.result.values
        )

    def stable_key(self) -> tuple[int, ...]:
        """返回协议、三层输入、目标分支和全部 directive。"""
        result = [
            *_packed(self.protocol.stable_key()),
            *_packed(self.structure.stable_key()),
            *_packed(self.execution.stable_key()),
            *_packed(self.branch.stable_key()),
            len(self.directives),
        ]
        for directive in self.directives:
            result.extend(_packed(directive.stable_key()))
        result.append(0 if self.attribution is None else 1)
        if self.attribution is not None:
            result.extend(_packed(self.attribution.stable_key()))
        result.append(len(self.sentence_attributions))
        for attribution in self.sentence_attributions:
            result.extend(_packed(attribution.stable_key()))
        return tuple(result)

    def sentence_attribution_map(self) -> dict[
            GenerationSentenceInstance, GenerationSurfaceSentenceAttribution]:
        """按运行期句实例返回精确 connector 归属。"""
        return {item.sentence: item for item in self.sentence_attributions}


@dataclass(frozen=True)
class SurfaceSlotPreview:
    """一个有序 slot 的 silent 或 R-01 reference/surface 无写入结果。"""

    directive: SurfaceSlotDirective
    value: StructureSlotValue
    antecedent_candidate_key: tuple[int, ...] = ()
    antecedent: ObjectIdentity | None = None
    reference: AliasResolutionProposal | None = None
    surface: AliasResolutionProposal | None = None
    representation: ObjectIdentity | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.directive, SurfaceSlotDirective):
            raise TypeError("surface slot preview directive 类型错误")
        if not isinstance(self.value, StructureSlotValue):
            raise TypeError("surface slot preview value 类型错误")
        if self.directive.slot != self.value.slot:
            raise ValueError("surface slot preview directive/value slot 不一致")
        _strict_key(
            self.antecedent_candidate_key,
            label="surface antecedent candidate key",
            allow_empty=True,
        )
        if self.antecedent is not None:
            _identity(self.antecedent, label="surface antecedent")
        if self.reference is not None:
            if not isinstance(self.reference, AliasResolutionProposal):
                raise TypeError("surface reference proposal 类型错误")
            if (self.reference.result.origin != self.value.filler
                    or self.reference.result.branch is not None):
                raise ValueError("reference proposal 未绑定当前 slot filler")
            if not isinstance(
                    self.reference.discovery, ReferenceRouteDiscovery):
                raise ValueError("reference proposal discovery 类型错误")
            if (self.antecedent is not None
                    and self.reference.discovery.target_kinds
                    != (self.antecedent.object_kind,)):
                raise ValueError("reference proposal 未精确请求 antecedent 对象类型")
        if self.surface is not None:
            if not isinstance(self.surface, AliasResolutionProposal):
                raise TypeError("surface proposal 类型错误")
            if (self.surface.result.origin != self.value.filler
                    or self.surface.result.branch is None):
                raise ValueError("surface proposal 未绑定当前 slot filler/branch")
            if not isinstance(self.surface.discovery, SurfaceRouteDiscovery):
                raise ValueError("surface proposal discovery 类型错误")
            if (self.surface.discovery.allowed_prefix_steps
                    != self.directive.surface_prefix_steps):
                raise ValueError("surface proposal prefix 策略与 directive 不一致")
        if self.representation is not None:
            _identity(self.representation, label="surface representation")
            if (self.surface is None
                    or self.surface.result.selected is None
                    or self.surface.result.selected.value != self.representation):
                raise ValueError("representation 未由唯一 surface proposal 产生")
        if bool(self.antecedent_candidate_key) != (self.antecedent is not None):
            raise ValueError("antecedent candidate/template 必须同时存在或同时缺失")

    def stable_key(self) -> tuple[int, ...]:
        """返回 directive、value、照应、R-01 proposal 和 Representation。"""
        result = [
            *_packed(self.directive.stable_key()),
            *_packed((
                *_packed(self.value.slot.stable_key()),
                *_packed(self.value.filler.stable_key()),
            )),
            1 if self.antecedent is not None else 0,
        ]
        if self.antecedent is not None:
            result.extend(_packed(self.antecedent_candidate_key))
            result.extend(_packed(self.antecedent.stable_key()))
        for proposal in (self.reference, self.surface):
            result.append(0 if proposal is None else 1)
            if proposal is not None:
                result.extend(_packed(proposal.stable_key()))
        result.append(0 if self.representation is None else 1)
        if self.representation is not None:
            result.extend(_packed(self.representation.stable_key()))
        return tuple(result)


def _proposal_failure(proposal: AliasResolutionProposal) -> str | None:
    """按 option 数量返回 missing、ambiguous 或空。"""
    count = len(proposal.result.options)
    if count == 0:
        return "missing"
    if count > 1:
        return "ambiguous"
    return None


@dataclass(frozen=True)
class GenerationSurfacePreview:
    """无写入 surface 规划前缀；失败时保留首个分型失败 slot。"""

    request: GenerationSurfaceRequest
    reason: ObjectIdentity
    slots: tuple[SurfaceSlotPreview, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.request, GenerationSurfaceRequest):
            raise TypeError("surface preview request 类型错误")
        _identity(
            self.reason,
            label="surface preview reason",
            kind=OBJECT_MINIMAL_INSTRUCTION,
        )
        if self.reason not in self.request.protocol.reasons():
            raise ValueError("surface preview reason 未在 protocol 注册")
        if not isinstance(self.slots, tuple) or any(
                not isinstance(item, SurfaceSlotPreview) for item in self.slots):
            raise TypeError("surface preview slots 类型错误")
        if self.reason == self.request.protocol.structure_incomplete_reason:
            if self.request.execution.complete or self.slots:
                raise ValueError("structure incomplete preview 必须零 slot")
            return
        if not self.request.execution.complete:
            raise ValueError("未完成 structure execution 只能返回对应失败原因")
        ordered = self.request.ordered_values()
        keys = tuple(
            (slot.directive.sentence, slot.value.slot) for slot in self.slots)
        expected = tuple(
            (sentence, value.slot) for sentence, value in ordered)
        if keys != expected[:len(keys)]:
            raise ValueError("surface preview slots 必须是 L-05B1 顺序前缀")
        if self.reason == self.request.protocol.complete_reason:
            if len(self.slots) != len(expected):
                raise ValueError("complete surface preview 必须覆盖全部有序 slot")
            for slot in self.slots:
                self._require_success(slot)
            return
        if not self.slots:
            raise ValueError("surface relation 失败必须保存失败 slot")
        for slot in self.slots[:-1]:
            self._require_success(slot)
        self._require_failure(self.slots[-1])

    @property
    def complete(self) -> bool:
        """仅在全部有序 slot 成功且 reason 为 complete 时返回真。"""
        return self.reason == self.request.protocol.complete_reason

    @property
    def representations(self) -> tuple[ObjectIdentity, ...]:
        """按 L-05B1 slot 顺序返回完整 preview 的 emitted Representation。"""
        if not self.complete:
            raise ValueError("失败 surface preview 不得暴露部分 Representation 序列")
        return tuple(
            item.representation
            for item in self.slots
            if item.representation is not None
        )

    def _require_success(self, slot: SurfaceSlotPreview) -> None:
        """核验 silent 或 emit slot 已完成全部必要 R-01 唯一选择。"""
        protocol = self.request.protocol
        if slot.directive.action == protocol.silent_action:
            if any((
                    slot.antecedent is not None,
                    slot.reference is not None,
                    slot.surface is not None,
                    slot.representation is not None)):
                raise ValueError("silent slot 不得携带 surface 结果")
            return
        if slot.directive.action != protocol.emit_action:
            raise ValueError("surface slot action 非法")
        if slot.surface is None or slot.representation is None:
            raise ValueError("emit slot 缺唯一 surface 结果")
        self._require_surface_query(slot)
        if _proposal_failure(slot.surface) is not None:
            raise ValueError("emit slot surface proposal 未唯一选择")
        expected = self.request.antecedent_map().get(
            (slot.directive.sentence, slot.value.slot))
        if expected is None:
            if slot.reference is not None or slot.antecedent is not None:
                raise ValueError("非 anaphora slot 不得携带 reference")
            return
        self._require_reference_success(slot, expected)

    def _require_reference_success(
            self,
            slot: SurfaceSlotPreview,
            expected: tuple[tuple[int, ...], ObjectIdentity],
            ) -> None:
        """核验 anaphora proposal 精确请求并唯一命中计划中的 antecedent。"""
        if (slot.antecedent_candidate_key != expected[0]
                or slot.antecedent != expected[1]
                or slot.reference is None
                or slot.reference.discovery.target_kinds
                != (expected[1].object_kind,)
                or _proposal_failure(slot.reference) is not None
                or slot.reference.result.selected.value != expected[1]):
            raise ValueError("anaphora slot 未唯一命中 antecedent")

    def _require_surface_query(self, slot: SurfaceSlotPreview) -> None:
        """核验 surface proposal 精确绑定当前 generation goal 的目标分支。"""
        if (slot.surface is None
                or slot.surface.result.branch != self.request.branch
                or slot.surface.discovery.branch != self.request.branch
                or slot.surface.discovery.allowed_prefix_steps
                != slot.directive.surface_prefix_steps):
            raise ValueError("surface proposal 未绑定 generation goal 目标分支")

    def _require_failure(self, slot: SurfaceSlotPreview) -> None:
        """核验末 slot 的 proposal 状态与分型 failure reason 一致。"""
        protocol = self.request.protocol
        if slot.directive.action != protocol.emit_action:
            raise ValueError("surface relation 失败只能发生在 emit slot")
        expected_antecedent = self.request.antecedent_map().get(
            (slot.directive.sentence, slot.value.slot))
        if self.reason in (
                protocol.reference_missing_reason,
                protocol.reference_ambiguous_reason,
                protocol.reference_mismatch_reason):
            if expected_antecedent is None or slot.reference is None:
                raise ValueError("reference failure 缺 anaphora proposal")
            if (slot.antecedent_candidate_key != expected_antecedent[0]
                    or slot.antecedent != expected_antecedent[1]
                    or slot.reference.discovery.target_kinds
                    != (expected_antecedent[1].object_kind,)):
                raise ValueError("reference failure 未绑定计划中的 antecedent")
            if slot.surface is not None or slot.representation is not None:
                raise ValueError("reference failure 后不得继续 surface 选择")
            failure = _proposal_failure(slot.reference)
            if (self.reason == protocol.reference_missing_reason
                    and failure != "missing"):
                raise ValueError("reference missing reason 与 proposal 不一致")
            if (self.reason == protocol.reference_ambiguous_reason
                    and failure != "ambiguous"):
                raise ValueError("reference ambiguous reason 与 proposal 不一致")
            if self.reason == protocol.reference_mismatch_reason:
                if (failure is not None
                        or slot.reference.result.selected.value
                        == expected_antecedent[1]):
                    raise ValueError("reference mismatch reason 与 proposal 不一致")
            return
        if slot.surface is None:
            raise ValueError("surface failure 缺 surface proposal")
        self._require_surface_query(slot)
        if expected_antecedent is None:
            if slot.reference is not None or slot.antecedent is not None:
                raise ValueError("非 anaphora surface failure 不得携带 reference")
        else:
            self._require_reference_success(slot, expected_antecedent)
        failure = _proposal_failure(slot.surface)
        if (self.reason == protocol.surface_missing_reason
                and failure != "missing"):
            raise ValueError("surface missing reason 与 proposal 不一致")
        if (self.reason == protocol.surface_ambiguous_reason
                and failure != "ambiguous"):
            raise ValueError("surface ambiguous reason 与 proposal 不一致")

    def stable_key(self) -> tuple[int, ...]:
        """返回请求、分型 reason 和已执行 slot 前缀。"""
        result = [
            *_packed(self.request.stable_key()),
            *_packed(self.reason.stable_key()),
            len(self.slots),
        ]
        for slot in self.slots:
            result.extend(_packed(slot.stable_key()))
        return tuple(result)


@dataclass(frozen=True)
class SurfaceAdoption:
    """一个 reference 或 surface proposal 提交后的完整 use 稳定键。"""

    sentence: ObjectIdentity | GenerationSentenceInstance
    slot: ObjectIdentity
    proposal: AliasResolutionProposal
    use_key: tuple[int, ...]
    use_stable_key: tuple[int, ...]

    def __post_init__(self) -> None:
        _sentence_address(self.sentence, label="surface adoption sentence")
        _identity(self.slot, label="surface adoption slot")
        if not isinstance(self.proposal, AliasResolutionProposal):
            raise TypeError("surface adoption proposal 类型错误")
        _strict_key(self.use_key, label="surface adoption use_key")
        _strict_key(
            self.use_stable_key, label="surface adoption use_stable_key")

    def stable_key(self) -> tuple[int, ...]:
        """返回 slot、proposal 和实际采用账键。"""
        return (
            *_packed(generation_sentence_address_key(
                self.sentence,
                label="surface adoption sentence",
            )),
            *_packed(self.slot.stable_key()),
            *_packed(self.proposal.stable_key()),
            *_packed(self.use_key),
            *_packed(self.use_stable_key),
        )


@dataclass(frozen=True)
class GenerationSurfacePlan:
    """完整 surface preview 及其一次性提交的全部 R-01 采用 trace。"""

    preview: GenerationSurfacePreview
    adoptions: tuple[SurfaceAdoption, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.preview, GenerationSurfacePreview):
            raise TypeError("surface plan preview 类型错误")
        if not self.preview.complete:
            raise ValueError("surface plan 只能由 complete preview 构造")
        if not isinstance(self.adoptions, tuple) or any(
                not isinstance(item, SurfaceAdoption)
                for item in self.adoptions):
            raise TypeError("surface plan adoptions 类型错误")
        expected = []
        for slot in self.preview.slots:
            if slot.reference is not None:
                expected.append((
                    slot.directive.sentence,
                    slot.value.slot,
                    slot.reference,
                    slot.directive.reference_use_key,
                ))
            if slot.surface is not None:
                expected.append((
                    slot.directive.sentence,
                    slot.value.slot,
                    slot.surface,
                    slot.directive.surface_use_key,
                ))
        actual = tuple(
            (item.sentence, item.slot, item.proposal, item.use_key)
            for item in self.adoptions)
        if actual != tuple(expected):
            raise ValueError("surface plan adoption 未逐点覆盖全部 R-01 proposal")

    @property
    def representations(self) -> tuple[ObjectIdentity, ...]:
        """按 L-05B1 slot 顺序返回全部 emitted Representation。"""
        return self.preview.representations

    def stable_key(self) -> tuple[int, ...]:
        """返回完整 preview 和全部实际采用 trace。"""
        result = [*_packed(self.preview.stable_key()), len(self.adoptions)]
        for adoption in self.adoptions:
            result.extend(_packed(adoption.stable_key()))
        return tuple(result)


__all__ = [
    "GenerationSurfaceAttribution",
    "GenerationSurfaceSentenceAttribution",
    "GenerationSurfacePlan",
    "GenerationSurfacePreview",
    "GenerationSurfaceProtocol",
    "GenerationSurfaceRequest",
    "SurfaceAdoption",
    "SurfaceSlotDirective",
    "SurfaceSlotPreview",
]
