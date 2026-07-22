"""把 typed 语义命题接到 S-07 结构和 R-01 表示选择。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.alias_resolution import (
    AliasRouteSearchBudget,
)
from pure_integer_ai.cognition.shared.generation_content import (
    AnswerContentSelection,
)
from pure_integer_ai.cognition.shared.generation_structure_execution import (
    GenerationStructureExecutionPlan,
    GenerationStructureExecutionPlanner,
    GenerationStructureExecutionRequest,
    SentenceStructureExecutionBudget,
)
from pure_integer_ai.cognition.shared.generation_structure_plan import (
    DiscoursePlan,
    GenerationStructurePlan,
    GenerationStructurePlanner,
    PlannedProposition,
    PlannedSentence,
    PropositionPlan,
    PropositionSlotFiller,
    SyntaxLinearizationObligation,
    SyntaxPlan,
)
from pure_integer_ai.cognition.shared.generation_surface import (
    GenerationSurfaceAttribution,
    GenerationSurfaceProtocol,
    SurfaceSlotDirective,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONCEPT,
    OBJECT_LANGUAGE_BRANCH,
    OBJECT_MINIMAL_INSTRUCTION,
    OBJECT_ROLE,
    OBJECT_STRUCTURE_CONCEPT,
    ObjectIdentity,
)
from pure_integer_ai.cognition.shared.structure_order import (
    StructureSlotDefinition,
)
from pure_integer_ai.cognition.shared.structure_order_consumer import (
    StructureOrderSearchBudget,
    StructureSlotValue,
)
from pure_integer_ai.cognition.shared.typed_binding import BoundProposition
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.experiments.generation_surface_runtime import (
    TypedGenerationSurfaceRequestBuilder,
)


class LanguageGenerationConnectorError(RuntimeError):
    """课程结构模板缺失、歧义或不能无损绑定当前命题。"""


def _packed(key: tuple[int, ...]) -> tuple[int, ...]:
    """为可变长稳定键增加长度边界。"""
    return len(key), *key


def _strict_key(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """核验非空严格整数键。"""
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{label} 必须是非空整数 tuple")
    assert_int(*value, _where=label)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{label} 必须使用严格整数")
    return value


def _identity(
        value: ObjectIdentity,
        *,
        label: str,
        kind: int | None = None,
        ) -> ObjectIdentity:
    """核验一等对象及可选对象类型。"""
    if not isinstance(value, ObjectIdentity):
        raise TypeError(f"{label} 必须是 ObjectIdentity")
    if kind is not None and value.object_kind != kind:
        raise ValueError(f"{label} 对象类型错误")
    return value


@dataclass(frozen=True)
class LanguageConnectorOrdinalDefinition:
    """把图内序标识符映射为 BoundRoleBinding 使用的严格整数坐标。"""

    instruction: ObjectIdentity
    ordinal: int

    def __post_init__(self) -> None:
        _identity(
            self.instruction,
            label="connector ordinal instruction",
            kind=OBJECT_MINIMAL_INSTRUCTION,
        )
        assert_int(
            self.ordinal,
            _where="LanguageConnectorOrdinalDefinition.ordinal",
        )
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ValueError("connector ordinal 必须为非负严格整数")

    def stable_key(self) -> tuple[int, ...]:
        """返回序标识符及其注入式执行坐标。"""
        return *_packed(self.instruction.stable_key()), self.ordinal


@dataclass(frozen=True)
class LanguageConnectorValueProtocol:
    """注入四种最小槽值读取指令，不解释任何语言或 relation 名称。"""

    proposition_source: ObjectIdentity
    predicate_source: ObjectIdentity
    role_filler_source: ObjectIdentity
    constant_source: ObjectIdentity
    ordinals: tuple[LanguageConnectorOrdinalDefinition, ...] = ()

    def __post_init__(self) -> None:
        values = self.sources()
        if len(set(values)) != len(values):
            raise ValueError("connector 槽值读取指令必须互不相同")
        for value in values:
            _identity(
                value,
                label="connector 槽值读取指令",
                kind=OBJECT_MINIMAL_INSTRUCTION,
            )
        if not isinstance(self.ordinals, tuple) or any(
                not isinstance(item, LanguageConnectorOrdinalDefinition)
                for item in self.ordinals):
            raise TypeError("connector ordinal protocol 必须是定义 tuple")
        if len({item.instruction for item in self.ordinals}) != len(
                self.ordinals):
            raise ValueError("connector ordinal instruction 不得重复")
        if len({item.ordinal for item in self.ordinals}) != len(self.ordinals):
            raise ValueError("connector ordinal 坐标不得重复")
        if set(values) & {item.instruction for item in self.ordinals}:
            raise ValueError("connector source 与 ordinal instruction 必须互异")
        object.__setattr__(self, "ordinals", tuple(sorted(
            self.ordinals, key=lambda item: item.ordinal)))

    def sources(self) -> tuple[ObjectIdentity, ...]:
        """返回全部开放读取指令。"""
        return (
            self.proposition_source,
            self.predicate_source,
            self.role_filler_source,
            self.constant_source,
        )

    def stable_key(self) -> tuple[int, ...]:
        """返回读取指令和序标识符执行协议的完整身份。"""
        result = [
            value
            for source in self.sources()
            for value in _packed(source.stable_key())
        ]
        result.append(len(self.ordinals))
        for ordinal in self.ordinals:
            result.extend(_packed(ordinal.stable_key()))
        return tuple(result)

    def ordinal_value(self, instruction: ObjectIdentity) -> int:
        """执行一个已注入序标识符；未知标识符不得从 components 猜数值。"""
        matches = tuple(
            item.ordinal for item in self.ordinals
            if item.instruction == instruction
        )
        if len(matches) != 1:
            raise LanguageGenerationConnectorError(
                "connector ordinal instruction 未唯一注册")
        return matches[0]


@dataclass(frozen=True)
class LanguageConnectorSlotBinding:
    """声明一个 S-07 slot 从 typed 命题的哪个结构位置取值。"""

    binding: ObjectIdentity
    slot: ObjectIdentity
    source: ObjectIdentity
    role: ObjectIdentity | None = None
    ordinal: ObjectIdentity | None = None
    constant: ObjectIdentity | None = None

    def __post_init__(self) -> None:
        _identity(
            self.binding,
            label="connector slot binding identity",
            kind=OBJECT_STRUCTURE_CONCEPT,
        )
        _identity(
            self.slot,
            label="connector slot binding slot",
            kind=OBJECT_STRUCTURE_CONCEPT,
        )
        _identity(
            self.source,
            label="connector slot binding source",
            kind=OBJECT_MINIMAL_INSTRUCTION,
        )
        if self.role is not None:
            _identity(
                self.role,
                label="connector slot binding role",
                kind=OBJECT_ROLE,
            )
        if self.constant is not None:
            _identity(self.constant, label="connector slot binding constant")
        if self.ordinal is not None:
            _identity(
                self.ordinal,
                label="connector slot binding ordinal",
                kind=OBJECT_MINIMAL_INSTRUCTION,
            )

    def stable_key(self) -> tuple[int, ...]:
        """返回一等 binding、slot、读取指令及可选 Role/序/常量完整键。"""
        role_key = () if self.role is None else self.role.stable_key()
        ordinal_key = (
            () if self.ordinal is None else self.ordinal.stable_key())
        constant_key = (
            () if self.constant is None else self.constant.stable_key())
        return (
            *_packed(self.binding.stable_key()),
            *_packed(self.slot.stable_key()),
            *_packed(self.source.stable_key()),
            *_packed(role_key),
            *_packed(ordinal_key),
            *_packed(constant_key),
        )


@dataclass(frozen=True)
class LanguageConnectorSurfaceDirective:
    """声明一个 connector slot 的 emit/silent 作用和表示路由步骤。"""

    directive: ObjectIdentity
    slot: ObjectIdentity
    action: ObjectIdentity
    instruction: ObjectIdentity
    prefix_route: ObjectIdentity
    surface_prefix_steps: tuple[ObjectIdentity, ...]

    def __post_init__(self) -> None:
        _identity(
            self.directive,
            label="connector surface directive",
            kind=OBJECT_STRUCTURE_CONCEPT,
        )
        _identity(
            self.slot,
            label="connector surface slot",
            kind=OBJECT_STRUCTURE_CONCEPT,
        )
        for label, value in (
                ("action", self.action),
                ("instruction", self.instruction)):
            _identity(
                value,
                label=f"connector surface {label}",
                kind=OBJECT_MINIMAL_INSTRUCTION,
            )
        _identity(
            self.prefix_route,
            label="connector surface prefix route",
            kind=OBJECT_STRUCTURE_CONCEPT,
        )
        if not isinstance(self.surface_prefix_steps, tuple):
            raise TypeError("connector surface prefix steps 必须是 tuple")
        if len(set(self.surface_prefix_steps)) != len(
                self.surface_prefix_steps):
            raise ValueError("connector surface prefix step 不得重复")
        for step in self.surface_prefix_steps:
            _identity(
                step,
                label="connector surface prefix step",
                kind=OBJECT_MINIMAL_INSTRUCTION,
            )
        object.__setattr__(self, "surface_prefix_steps", tuple(sorted(
            self.surface_prefix_steps,
            key=ObjectIdentity.stable_key,
        )))

    def stable_key(self) -> tuple[int, ...]:
        """返回只含语言理论和表示路由作用的逐槽稳定键。"""
        return (
            *_packed(self.directive.stable_key()),
            *_packed(self.slot.stable_key()),
            *_packed(self.action.stable_key()),
            *_packed(self.instruction.stable_key()),
            *_packed(self.prefix_route.stable_key()),
            len(self.surface_prefix_steps),
            *(value for step in self.surface_prefix_steps
              for value in _packed(step.stable_key())),
        )


@dataclass(frozen=True)
class LanguageConnectorSurfaceRuntimePolicy:
    """保存逐槽 trace、R-01 搜索预算和 Use key 后缀。"""

    slot: ObjectIdentity
    trace: tuple[int, ...]
    surface_budget: AliasRouteSearchBudget | None
    use_key_suffix: tuple[int, ...]

    def __post_init__(self) -> None:
        _identity(
            self.slot,
            label="connector runtime surface slot",
            kind=OBJECT_STRUCTURE_CONCEPT,
        )
        _strict_key(self.trace, label="connector runtime surface trace")
        if (self.surface_budget is not None
                and not isinstance(self.surface_budget,
                                   AliasRouteSearchBudget)):
            raise TypeError("connector runtime surface budget 类型错误")
        _strict_key(
            self.use_key_suffix,
            label="connector runtime use key suffix",
        )

    def stable_key(self) -> tuple[int, ...]:
        """返回逐槽运行策略键，不把它并入语言理论身份。"""
        budget_key = (
            ()
            if self.surface_budget is None
            else self.surface_budget.stable_key()
        )
        return (
            *_packed(self.slot.stable_key()),
            *_packed(self.trace),
            *_packed(budget_key),
            *_packed(self.use_key_suffix),
        )


@dataclass(frozen=True)
class LanguageGenerationConnectorTemplate:
    """把一个精确 semantic predicate/structure 映射到图内句法结构。"""

    connector: ObjectIdentity
    language_branch: ObjectIdentity
    proposition_structure: ObjectIdentity
    predicate: ObjectIdentity
    sentence: ObjectIdentity
    structure: ObjectIdentity
    slots: tuple[StructureSlotDefinition, ...]
    bindings: tuple[LanguageConnectorSlotBinding, ...]
    constraint_set: ObjectIdentity
    constraints: tuple[ObjectIdentity, ...]
    context_set: ObjectIdentity
    context: tuple[ObjectIdentity, ...]
    boundary: ObjectIdentity
    linearization_reason: ObjectIdentity
    surface: tuple[LanguageConnectorSurfaceDirective, ...]

    def __post_init__(self) -> None:
        for label, value, kind in (
                ("connector", self.connector, OBJECT_STRUCTURE_CONCEPT),
                ("language branch", self.language_branch,
                 OBJECT_LANGUAGE_BRANCH),
                ("proposition structure", self.proposition_structure,
                 OBJECT_STRUCTURE_CONCEPT),
                ("predicate", self.predicate, OBJECT_CONCEPT),
                ("sentence", self.sentence, OBJECT_STRUCTURE_CONCEPT),
                ("surface structure", self.structure,
                 OBJECT_STRUCTURE_CONCEPT),
                ("constraint set", self.constraint_set,
                 OBJECT_STRUCTURE_CONCEPT),
                ("context set", self.context_set,
                 OBJECT_STRUCTURE_CONCEPT),
                ("boundary", self.boundary, OBJECT_MINIMAL_INSTRUCTION),
                ("linearization reason", self.linearization_reason,
                 OBJECT_MINIMAL_INSTRUCTION)):
            _identity(value, label=f"connector {label}", kind=kind)
        if (not isinstance(self.slots, tuple) or not self.slots
                or any(not isinstance(item, StructureSlotDefinition)
                       for item in self.slots)):
            raise TypeError("connector slots 必须是非空 StructureSlotDefinition tuple")
        if any(item.structure != self.structure for item in self.slots):
            raise ValueError("connector slot 必须属于模板 surface structure")
        slot_ids = tuple(item.slot for item in self.slots)
        if len(set(slot_ids)) != len(slot_ids):
            raise ValueError("connector template 不得重复 slot")
        for label, values, expected_type in (
                ("bindings", self.bindings, LanguageConnectorSlotBinding),
                ("surface", self.surface,
                 LanguageConnectorSurfaceDirective)):
            if not isinstance(values, tuple) or any(
                    not isinstance(item, expected_type) for item in values):
                raise TypeError(f"connector {label} 类型错误")
            if {item.slot for item in values} != set(slot_ids):
                raise ValueError(f"connector {label} 必须精确覆盖全部 slot")
            if len({item.slot for item in values}) != len(values):
                raise ValueError(f"connector {label} 不得重复 slot")
        if len({item.binding for item in self.bindings}) != len(self.bindings):
            raise ValueError("connector binding 一等身份不得重复")
        if len({item.directive for item in self.surface}) != len(self.surface):
            raise ValueError("connector surface directive 身份不得重复")
        if len({item.prefix_route for item in self.surface}) != len(
                self.surface):
            raise ValueError("connector surface prefix route 身份不得重复")
        owned_structures = (
            self.constraint_set,
            self.context_set,
            *(item.binding for item in self.bindings),
            *(item.directive for item in self.surface),
            *(item.prefix_route for item in self.surface),
        )
        if len(set(owned_structures)) != len(owned_structures):
            raise ValueError("connector 内部结构身份不得跨职责复用")
        if not isinstance(self.constraints, tuple):
            raise TypeError("connector constraints 必须是 tuple")
        for constraint in self.constraints:
            _identity(
                constraint,
                label="connector constraint",
                kind=OBJECT_STRUCTURE_CONCEPT,
            )
        if len(set(self.constraints)) != len(self.constraints):
            raise ValueError("connector constraint 不得重复")
        if not isinstance(self.context, tuple):
            raise TypeError("connector context 必须是 ObjectIdentity tuple")
        for value in self.context:
            _identity(value, label="connector context")
        if len(set(self.context)) != len(self.context):
            raise ValueError("connector context 不得重复")
        object.__setattr__(self, "slots", tuple(sorted(
            self.slots, key=lambda item: item.slot.stable_key())))
        object.__setattr__(self, "bindings", tuple(sorted(
            self.bindings, key=lambda item: item.slot.stable_key())))
        object.__setattr__(self, "constraints", tuple(sorted(
            self.constraints, key=ObjectIdentity.stable_key)))
        object.__setattr__(self, "context", tuple(sorted(
            self.context, key=ObjectIdentity.stable_key)))
        object.__setattr__(self, "surface", tuple(sorted(
            self.surface, key=lambda item: item.slot.stable_key())))

    def match_key(self) -> tuple[ObjectIdentity, ObjectIdentity, ObjectIdentity]:
        """返回目标分支、semantic structure 和 predicate 的精确匹配键。"""
        return (
            self.language_branch,
            self.proposition_structure,
            self.predicate,
        )

    def stable_key(self) -> tuple[int, ...]:
        """返回 connector 语言理论、结构、槽值和 surface 作用键。"""
        result = [
            *_packed(self.connector.stable_key()),
            *_packed(self.language_branch.stable_key()),
            *_packed(self.proposition_structure.stable_key()),
            *_packed(self.predicate.stable_key()),
            *_packed(self.sentence.stable_key()),
            *_packed(self.structure.stable_key()),
            len(self.slots),
        ]
        for slot in self.slots:
            result.extend(_packed(slot.structure.stable_key()))
            result.extend(_packed(slot.slot.stable_key()))
            result.extend(_packed(slot.role.stable_key()))
            result.extend(_packed(slot.value_type.stable_key()))
        for values in (self.bindings, self.surface):
            result.append(len(values))
            for value in values:
                result.extend(_packed(value.stable_key()))
        result.extend(_packed(self.constraint_set.stable_key()))
        result.append(len(self.constraints))
        for value in self.constraints:
            result.extend(_packed(value.stable_key()))
        result.extend(_packed(self.context_set.stable_key()))
        result.append(len(self.context))
        for value in self.context:
            result.extend(_packed(value.stable_key()))
        result.extend(_packed(self.boundary.stable_key()))
        result.extend(_packed(self.linearization_reason.stable_key()))
        return tuple(result)


class LanguageGenerationConnectorRegistry:
    """按精确 typed 命题匹配唯一课程模板，并解析全部槽值。"""

    def __init__(
            self,
            value_protocol: LanguageConnectorValueProtocol,
            templates: tuple[LanguageGenerationConnectorTemplate, ...],
            ) -> None:
        if not isinstance(value_protocol, LanguageConnectorValueProtocol):
            raise TypeError("connector value protocol 类型错误")
        if (not isinstance(templates, tuple) or not templates
                or any(not isinstance(item,
                                       LanguageGenerationConnectorTemplate)
                       for item in templates)):
            raise TypeError("connector templates 必须是非空模板 tuple")
        if len({item.connector for item in templates}) != len(templates):
            raise ValueError("connector template identity 不得重复")
        self.value_protocol = value_protocol
        self.templates = tuple(sorted(
            templates, key=lambda item: item.connector.stable_key()))
        for template in self.templates:
            self._validate_sources(template)

    def _validate_sources(
            self, template: LanguageGenerationConnectorTemplate) -> None:
        """按注入最小指令核验 Role、常量和命题槽位形状。"""
        protocol = self.value_protocol
        proposition_count = 0
        for binding in template.bindings:
            if binding.source not in protocol.sources():
                raise ValueError("connector binding 使用未注册读取指令")
            if binding.source == protocol.proposition_source:
                proposition_count += 1
                if (binding.role is not None or binding.ordinal is not None
                        or binding.constant is not None):
                    raise ValueError(
                        "非 Role filler 槽位不得声明 Role、序或常量")
            elif binding.source == protocol.predicate_source:
                if (binding.role is not None or binding.ordinal is not None
                        or binding.constant is not None):
                    raise ValueError(
                        "非 Role filler 槽位不得声明 Role、序或常量")
            elif binding.source == protocol.role_filler_source:
                if (binding.role is None or binding.ordinal is None
                        or binding.constant is not None):
                    raise ValueError("Role filler 槽位必须只声明 Role 和序标识符")
                protocol.ordinal_value(binding.ordinal)
            else:
                if (binding.role is not None or binding.ordinal is not None
                        or binding.constant is None):
                    raise ValueError("常量槽位必须只声明一等常量")
        if proposition_count != 1:
            raise ValueError("connector template 必须恰有一个命题本体槽位")

    def match(
            self,
            selection: AnswerContentSelection,
            ) -> tuple[LanguageGenerationConnectorTemplate, object]:
        """返回唯一模板和候选；多命题或多模板时拒绝私下排序。"""
        if not isinstance(selection, AnswerContentSelection):
            raise TypeError("connector registry 只接受 AnswerContentSelection")
        selected = tuple(
            candidate
            for candidate in selection.request.candidates
            if candidate.stable_key() in set(
                selection.selected_candidate_keys)
        )
        if len(selected) != 1:
            raise LanguageGenerationConnectorError(
                "connector 首个生产契约要求 G-01 显式收敛为一个命题")
        candidate = selected[0]
        branch = selection.request.goal.target_branch
        if branch is None:
            raise LanguageGenerationConnectorError(
                "connector 缺少目标 LanguageBranch")
        key = (
            branch,
            candidate.proposition.structure,
            candidate.proposition.predicate,
        )
        matches = tuple(
            template for template in self.templates
            if template.match_key() == key
        )
        if not matches:
            raise LanguageGenerationConnectorError(
                "当前 predicate/structure/LanguageBranch 没有课程模板")
        if len(matches) != 1:
            raise LanguageGenerationConnectorError(
                "当前 predicate/structure/LanguageBranch 存在歧义模板")
        return matches[0], candidate

    def values(
            self,
            template: LanguageGenerationConnectorTemplate,
            proposition: BoundProposition,
            ) -> tuple[StructureSlotValue, ...]:
        """按模板最小指令无损读取命题本体、predicate、Role filler 或常量。"""
        if not isinstance(template, LanguageGenerationConnectorTemplate):
            raise TypeError("connector values template 类型错误")
        if not isinstance(proposition, BoundProposition):
            raise TypeError("connector values proposition 类型错误")
        protocol = self.value_protocol
        result = []
        for binding in template.bindings:
            if binding.source == protocol.proposition_source:
                value = proposition.template
            elif binding.source == protocol.predicate_source:
                value = proposition.predicate
            elif binding.source == protocol.role_filler_source:
                ordinal = protocol.ordinal_value(binding.ordinal)
                matches = tuple(
                    item for item in proposition.bindings
                    if item.role == binding.role
                    and item.ordinal == ordinal
                )
                if len(matches) != 1:
                    raise LanguageGenerationConnectorError(
                        "connector Role+ordinal 未唯一绑定")
                filler = matches[0].filler
                if isinstance(filler, BoundProposition):
                    raise LanguageGenerationConnectorError(
                        "嵌套命题 filler 必须由独立显式读取指令处理")
                value = filler
            else:
                value = binding.constant
            if not isinstance(value, ObjectIdentity):
                raise LanguageGenerationConnectorError(
                    "connector 槽值未解析为一等对象")
            result.append(StructureSlotValue(binding.slot, value))
        return tuple(result)

    def stable_key(self) -> tuple[int, ...]:
        """返回全部语言理论模板和读取协议，不含运行策略。"""
        result = [
            *_packed(self.value_protocol.stable_key()),
            len(self.templates),
        ]
        for template in self.templates:
            result.extend(_packed(template.stable_key()))
        return tuple(result)


@dataclass(frozen=True)
class LanguageConnectorTemplateRuntimePolicy:
    """把一个 connector 定义关联到完整的逐槽运行策略。"""

    connector: ObjectIdentity
    surface: tuple[LanguageConnectorSurfaceRuntimePolicy, ...]

    def __post_init__(self) -> None:
        _identity(
            self.connector,
            label="connector runtime template",
            kind=OBJECT_STRUCTURE_CONCEPT,
        )
        if (not isinstance(self.surface, tuple) or not self.surface
                or any(not isinstance(
                    item, LanguageConnectorSurfaceRuntimePolicy)
                       for item in self.surface)):
            raise TypeError("connector runtime surface 必须是非空策略 tuple")
        if len({item.slot for item in self.surface}) != len(self.surface):
            raise ValueError("connector runtime surface 不得重复 slot")
        object.__setattr__(self, "surface", tuple(sorted(
            self.surface, key=lambda item: item.slot.stable_key())))

    def stable_key(self) -> tuple[int, ...]:
        """返回模板身份和全部逐槽运行策略。"""
        return (
            *_packed(self.connector.stable_key()),
            len(self.surface),
            *(value for item in self.surface
              for value in _packed(item.stable_key())),
        )


@dataclass(frozen=True)
class LanguageGenerationConnectorRuntimePolicy:
    """保存一次 connector 装配使用的搜索预算、trace 和 Use 键策略。"""

    use_key_namespace: tuple[int, ...]
    order_budget: StructureOrderSearchBudget
    templates: tuple[LanguageConnectorTemplateRuntimePolicy, ...]

    def __post_init__(self) -> None:
        _strict_key(
            self.use_key_namespace,
            label="connector runtime use key namespace",
        )
        if not isinstance(self.order_budget, StructureOrderSearchBudget):
            raise TypeError("connector runtime order budget 类型错误")
        if (not isinstance(self.templates, tuple) or not self.templates
                or any(not isinstance(
                    item, LanguageConnectorTemplateRuntimePolicy)
                       for item in self.templates)):
            raise TypeError("connector runtime templates 必须是非空策略 tuple")
        if len({item.connector for item in self.templates}) != len(
                self.templates):
            raise ValueError("connector runtime template 不得重复")
        object.__setattr__(self, "templates", tuple(sorted(
            self.templates, key=lambda item: item.connector.stable_key())))

    def template_for(
            self,
            connector: ObjectIdentity,
            ) -> LanguageConnectorTemplateRuntimePolicy:
        """按 connector 一等身份返回唯一运行策略。"""
        matches = tuple(
            item for item in self.templates
            if item.connector == connector
        )
        if len(matches) != 1:
            raise LanguageGenerationConnectorError(
                "connector 没有唯一运行策略")
        return matches[0]

    def stable_key(self) -> tuple[int, ...]:
        """返回全部 run-local 预算、trace 和 Use 键配置。"""
        return (
            *_packed(self.use_key_namespace),
            *_packed((self.order_budget.max_states,)),
            len(self.templates),
            *(value for item in self.templates
              for value in _packed(item.stable_key())),
        )


class LanguageConnectorDiscourseMapper:
    """为单一已选命题保留 unresolved obligation 和模板 context。"""

    def __init__(self, registry: LanguageGenerationConnectorRegistry) -> None:
        self.registry = registry

    def plan(self, selection: AnswerContentSelection) -> DiscoursePlan:
        """建立不猜表层顺序的单命题 discourse。"""
        template, candidate = self.registry.match(selection)
        questions = (
            () if candidate.reasoning is None
            else candidate.reasoning.unresolved
        )
        return DiscoursePlan(
            selection.stable_key(),
            selection.selected_candidate_keys,
            (),
            questions,
            template.context,
        )


class LanguageConnectorPropositionMapper:
    """逐字段复制 G-01 selected candidate 的命题与 active Evidence。"""

    def plan(
            self,
            selection: AnswerContentSelection,
            discourse: DiscoursePlan,
            ) -> PropositionPlan:
        """形成不丢失 H-00 Hypothesis/Evidence 的 PropositionPlan。"""
        if discourse.selection_key != selection.stable_key():
            raise ValueError("connector proposition mapper 收到漂移 discourse")
        selected = set(selection.selected_candidate_keys)
        propositions = tuple(
            PlannedProposition(
                candidate.stable_key(),
                candidate.proposition,
                candidate.state,
                candidate.source,
                candidate.scope,
                candidate.evidence,
                candidate.hypotheses,
                (),
            )
            for candidate in selection.request.candidates
            if candidate.stable_key() in selected
        )
        return PropositionPlan(selection.stable_key(), propositions)


class LanguageConnectorSyntaxMapper:
    """把唯一 typed 命题按课程模板投影为 S-07 slot/value 义务。"""

    def __init__(self, registry: LanguageGenerationConnectorRegistry) -> None:
        self.registry = registry

    def plan(
            self,
            selection: AnswerContentSelection,
            discourse: DiscoursePlan,
            propositions: PropositionPlan,
            ) -> SyntaxPlan:
        """生成只含一等对象和 active constraint 身份的句法计划。"""
        template, candidate = self.registry.match(selection)
        if discourse.selection_key != selection.stable_key():
            raise ValueError("connector syntax mapper 收到漂移 discourse")
        planned = {item.candidate_key: item for item in propositions.propositions}
        key = candidate.stable_key()
        if set(planned) != {key} or planned[key].proposition != (
                candidate.proposition):
            raise ValueError("connector syntax mapper 收到漂移 PropositionPlan")
        values = self.registry.values(template, candidate.proposition)
        proposition_values = tuple(
            value
            for binding, value in zip(template.bindings, values)
            if binding.source == self.registry.value_protocol.proposition_source
        )
        if len(proposition_values) != 1:
            raise RuntimeError("connector 命题本体槽位预检失效")
        sentence = PlannedSentence(
            template.sentence,
            template.structure,
            0,
            (key,),
            template.slots,
            values,
            (PropositionSlotFiller(
                key,
                candidate.proposition,
                proposition_values[0],
            ),),
            template.boundary,
            candidate.source,
            candidate.scope,
        )
        obligation = SyntaxLinearizationObligation(
            template.sentence,
            template.structure,
            sentence.values,
            template.constraints,
            template.context,
            template.linearization_reason,
            candidate.source,
            candidate.scope,
        )
        return SyntaxPlan(
            selection.stable_key(),
            (sentence,),
            (),
            (obligation,),
        )


class LanguageConnectorExecutionRequestMapper:
    """为 connector 生成的每个句子注入相同的有限 S-07 搜索预算。"""

    def __init__(self, budget: StructureOrderSearchBudget) -> None:
        if not isinstance(budget, StructureOrderSearchBudget):
            raise TypeError("connector order budget 类型错误")
        self.budget = budget

    def build(
            self,
            structure: GenerationStructurePlan,
            ) -> GenerationStructureExecutionRequest:
        """逐句建立预算，不替换 G-02 SyntaxPlan。"""
        if not isinstance(structure, GenerationStructurePlan):
            raise TypeError("connector execution mapper 需要 GenerationStructurePlan")
        return GenerationStructureExecutionRequest(
            structure.syntax,
            tuple(
                SentenceStructureExecutionBudget(
                    sentence.sentence,
                    self.budget,
                )
                for sentence in structure.syntax.sentences
            ),
        )


class LanguageConnectorSurfaceDirectiveMapper:
    """按同一课程模板完整产生 R-01 emit/silent 指令。"""

    def __init__(
            self,
            registry: LanguageGenerationConnectorRegistry,
            runtime_policy: LanguageGenerationConnectorRuntimePolicy,
            protocol: GenerationSurfaceProtocol,
            ) -> None:
        if not isinstance(
                runtime_policy, LanguageGenerationConnectorRuntimePolicy):
            raise TypeError("connector surface runtime policy 类型错误")
        if not isinstance(protocol, GenerationSurfaceProtocol):
            raise TypeError("connector surface protocol 类型错误")
        self.registry = registry
        self.runtime_policy = runtime_policy
        self.protocol = protocol

    def plan(
            self,
            structure: GenerationStructurePlan,
            execution: GenerationStructureExecutionPlan,
            branch: ObjectIdentity,
            ) -> tuple[SurfaceSlotDirective, ...]:
        """绑定当前 selection/scope 生成唯一 use key，不读取词面或 Unicode。"""
        if not isinstance(structure, GenerationStructurePlan):
            raise TypeError("connector surface mapper 需要 GenerationStructurePlan")
        if not isinstance(execution, GenerationStructureExecutionPlan):
            raise TypeError(
                "connector surface mapper 需要 GenerationStructureExecutionPlan")
        _identity(
            branch,
            label="connector surface mapper branch",
            kind=OBJECT_LANGUAGE_BRANCH,
        )
        if execution.request.syntax != structure.syntax:
            raise ValueError("connector surface mapper 收到漂移 S-07 execution")
        if not execution.complete:
            raise ValueError("connector surface mapper 只接受完整 S-07 execution")
        template, _candidate = self.registry.match(structure.selection)
        if branch != template.language_branch:
            raise ValueError("connector surface mapper 目标分支漂移")
        sentence = structure.syntax.sentences
        if len(sentence) != 1 or sentence[0].sentence != template.sentence:
            raise ValueError("connector surface mapper sentence 漂移")
        runtime = self.runtime_policy.template_for(template.connector)
        theory_by_slot = {item.slot: item for item in template.surface}
        runtime_by_slot = {item.slot: item for item in runtime.surface}
        if set(theory_by_slot) != set(runtime_by_slot):
            raise ValueError("connector surface 理论与运行策略 slot 不一致")
        result = []
        for value in sentence[0].values:
            theory = theory_by_slot[value.slot]
            policy = runtime_by_slot[value.slot]
            if theory.action not in self.protocol.actions():
                raise ValueError("connector surface action 未注册")
            emitted = theory.action == self.protocol.emit_action
            if emitted != (policy.surface_budget is not None):
                raise ValueError("connector emit/silent 与 R-01 预算不一致")
            use_key = ()
            if emitted:
                use_key = (
                    *self.runtime_policy.use_key_namespace,
                    *_packed(structure.selection.stable_key()),
                    *_packed(template.connector.stable_key()),
                    *_packed(template.sentence.stable_key()),
                    *_packed(value.slot.stable_key()),
                    *_packed(policy.use_key_suffix),
                )
            result.append(SurfaceSlotDirective(
                template.sentence,
                value.slot,
                theory.action,
                theory.instruction,
                policy.trace,
                theory.surface_prefix_steps,
                policy.surface_budget,
                use_key,
            ))
        return tuple(result)


class LanguageConnectorSurfaceAttributionMapper:
    """按当前唯一模板注入 exact connector Hypothesis，不从 Use key 反推。"""

    def __init__(
            self,
            registry: LanguageGenerationConnectorRegistry,
            attributions: tuple[GenerationSurfaceAttribution, ...],
            ) -> None:
        if not isinstance(registry, LanguageGenerationConnectorRegistry):
            raise TypeError("connector attribution registry 类型错误")
        if not isinstance(attributions, tuple) or any(
                not isinstance(item, GenerationSurfaceAttribution)
                for item in attributions):
            raise TypeError("connector attributions 类型错误")
        mapping = {item.theory: item for item in attributions}
        if len(mapping) != len(attributions):
            raise ValueError("同一 connector 不得重复归属 Hypothesis")
        template_ids = {item.connector for item in registry.templates}
        if mapping and set(mapping) != template_ids:
            raise ValueError("connector attribution 必须精确覆盖全部模板")
        self.registry = registry
        self.attributions = mapping

    def attribution(
            self,
            structure: GenerationStructurePlan,
            ) -> GenerationSurfaceAttribution | None:
        """返回匹配模板的显式归属；无生命周期装配时保留 generic 空归属。"""
        template, _candidate = self.registry.match(structure.selection)
        return self.attributions.get(template.connector)


class LanguageGenerationConnector:
    """装配共享 G-02 mapper 与执行 S-07/R-01 的 G-03 request builder。"""

    def __init__(
            self,
            registry: LanguageGenerationConnectorRegistry,
            runtime_policy: LanguageGenerationConnectorRuntimePolicy,
            surface_protocol: GenerationSurfaceProtocol,
            attributions: tuple[GenerationSurfaceAttribution, ...] = (),
            ) -> None:
        if not isinstance(registry, LanguageGenerationConnectorRegistry):
            raise TypeError("language generation connector registry 类型错误")
        if not isinstance(
                runtime_policy, LanguageGenerationConnectorRuntimePolicy):
            raise TypeError("language generation connector runtime policy 类型错误")
        if not isinstance(surface_protocol, GenerationSurfaceProtocol):
            raise TypeError("language generation connector surface protocol 类型错误")
        self.registry = registry
        self.runtime_policy = runtime_policy
        self.surface_protocol = surface_protocol
        self.attribution_mapper = LanguageConnectorSurfaceAttributionMapper(
            registry,
            attributions,
        )
        self._validate_runtime_policy()

    def _validate_runtime_policy(self) -> None:
        """双向核验每个理论模板及 slot 都有且只有一个运行策略。"""
        definitions = {item.connector: item for item in self.registry.templates}
        policies = {
            item.connector: item for item in self.runtime_policy.templates}
        if set(definitions) != set(policies):
            raise ValueError("connector 理论模板与运行策略未双向覆盖")
        actions = self.surface_protocol.actions()
        for connector, template in definitions.items():
            runtime = policies[connector]
            theory_by_slot = {item.slot: item for item in template.surface}
            runtime_by_slot = {item.slot: item for item in runtime.surface}
            if set(theory_by_slot) != set(runtime_by_slot):
                raise ValueError("connector 理论 slot 与运行策略未双向覆盖")
            for slot, theory in theory_by_slot.items():
                if theory.action not in actions:
                    raise ValueError("connector surface action 未注册")
                emitted = theory.action == self.surface_protocol.emit_action
                if emitted != (
                        runtime_by_slot[slot].surface_budget is not None):
                    raise ValueError("connector emit/silent 与 R-01 预算不一致")

    def structure_planner(self) -> GenerationStructurePlanner:
        """返回可直接安装到 G-00 三层 resolver 的 typed 结构 planner。"""
        return GenerationStructurePlanner(
            LanguageConnectorDiscourseMapper(self.registry),
            LanguageConnectorPropositionMapper(),
            LanguageConnectorSyntaxMapper(self.registry),
        )

    def surface_request_builder(
            self,
            execution_planner: GenerationStructureExecutionPlanner,
            ) -> TypedGenerationSurfaceRequestBuilder:
        """返回真实调用 S-07 并向 G-03/R-01 提交完整指令的 builder。"""
        if not isinstance(
                execution_planner, GenerationStructureExecutionPlanner):
            raise TypeError("connector execution planner 类型错误")
        return TypedGenerationSurfaceRequestBuilder(
            self.surface_protocol,
            execution_planner,
            LanguageConnectorExecutionRequestMapper(
                self.runtime_policy.order_budget),
            LanguageConnectorSurfaceDirectiveMapper(
                self.registry,
                self.runtime_policy,
                self.surface_protocol,
            ),
            self.attribution_mapper,
        )

    def stable_key(self) -> tuple[int, ...]:
        """返回语言理论、run-local 策略和 surface 协议完整配置键。"""
        return (
            *_packed(self.registry.stable_key()),
            *_packed(self.runtime_policy.stable_key()),
            *_packed(self.surface_protocol.stable_key()),
            len(self.attribution_mapper.attributions),
            *(value
              for item in sorted(
                  self.attribution_mapper.attributions.values(),
                  key=lambda current: current.theory.stable_key(),
              )
              for value in _packed(item.stable_key())),
        )


__all__ = [
    "LanguageConnectorDiscourseMapper",
    "LanguageConnectorExecutionRequestMapper",
    "LanguageConnectorOrdinalDefinition",
    "LanguageConnectorPropositionMapper",
    "LanguageConnectorSlotBinding",
    "LanguageConnectorSurfaceDirective",
    "LanguageConnectorSurfaceDirectiveMapper",
    "LanguageConnectorSurfaceAttributionMapper",
    "LanguageConnectorSurfaceRuntimePolicy",
    "LanguageConnectorTemplateRuntimePolicy",
    "LanguageConnectorSyntaxMapper",
    "LanguageConnectorValueProtocol",
    "LanguageGenerationConnector",
    "LanguageGenerationConnectorError",
    "LanguageGenerationConnectorRegistry",
    "LanguageGenerationConnectorRuntimePolicy",
    "LanguageGenerationConnectorTemplate",
]
