"""把 typed 语义命题接到 S-07 结构和 R-01 表示选择。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

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
    DiscourseDependency,
    DiscoursePlan,
    GenerationSentenceInstance,
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
    GenerationSurfaceSentenceAttribution,
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
    SourceRef,
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


@dataclass(frozen=True)
class LanguageConnectorDiscourseDeclaration:
    """课程或图读取器注入的一次多命题篇章依赖声明。"""

    candidate_keys: tuple[tuple[int, ...], ...]
    dependencies: tuple[DiscourseDependency, ...]
    source: SourceRef
    trace: tuple[int, ...]

    def __post_init__(self) -> None:
        """要求声明完整列出候选、依赖一等关系和可追溯来源。"""
        if not isinstance(self.candidate_keys, tuple) or not self.candidate_keys:
            raise ValueError("connector discourse candidate_keys 必须是非空 tuple")
        for key in self.candidate_keys:
            _strict_key(key, label="connector discourse candidate key")
        if len(set(self.candidate_keys)) != len(self.candidate_keys):
            raise ValueError("connector discourse candidate key 不得重复")
        if (not isinstance(self.dependencies, tuple)
                or any(not isinstance(item, DiscourseDependency)
                       for item in self.dependencies)):
            raise TypeError("connector discourse dependencies 类型错误")
        if len({item.stable_key() for item in self.dependencies}) != len(
                self.dependencies):
            raise ValueError("connector discourse dependency 不得重复")
        if not isinstance(self.source, SourceRef):
            raise TypeError("connector discourse source 类型错误")
        _strict_key(self.trace, label="connector discourse trace")
        object.__setattr__(self, "candidate_keys", tuple(sorted(
            self.candidate_keys)))
        object.__setattr__(self, "dependencies", tuple(sorted(
            self.dependencies, key=lambda item: item.stable_key())))

    def stable_key(self) -> tuple[int, ...]:
        """返回候选集合、依赖声明、来源和 trace 的完整键。"""
        result = [len(self.candidate_keys)]
        for key in self.candidate_keys:
            result.extend(_packed(key))
        result.append(len(self.dependencies))
        for dependency in self.dependencies:
            result.extend(_packed(dependency.stable_key()))
        result.extend(_packed(self.source.stable_key()))
        result.extend(_packed(self.trace))
        return tuple(result)


class LanguageConnectorDiscourseDeclarationProvider(Protocol):
    """为当前 G-01 selection 返回课程或图读取器声明的篇章依赖。"""

    def declaration(
            self,
            selection: AnswerContentSelection,
            ) -> LanguageConnectorDiscourseDeclaration | None:
        """返回精确覆盖当前 selection 的声明；无匹配时返回空。"""
        ...

    def state_key(self) -> tuple[int, ...]:
        """返回不可变声明集或读取器版本的完整状态键。"""
        ...

    def clone_for_evaluation(
            self,
            ) -> "LanguageConnectorDiscourseDeclarationProvider":
        """返回不共享可变读取状态、且保持同一声明配置的评测副本。"""
        ...


@dataclass(frozen=True)
class StaticLanguageConnectorDiscourseDeclarations:
    """供 PH1 注入和测试使用的不可变篇章声明集合。"""

    declarations: tuple[LanguageConnectorDiscourseDeclaration, ...]

    def __post_init__(self) -> None:
        """拒绝两个声明覆盖同一候选集，避免由存储顺序私选。"""
        if (not isinstance(self.declarations, tuple)
                or any(not isinstance(item, LanguageConnectorDiscourseDeclaration)
                       for item in self.declarations)):
            raise TypeError("connector discourse declarations 类型错误")
        keys = tuple(item.candidate_keys for item in self.declarations)
        if len(set(keys)) != len(keys):
            raise ValueError("同一 connector candidate 集不得重复声明篇章顺序")
        object.__setattr__(self, "declarations", tuple(sorted(
            self.declarations, key=lambda item: item.stable_key())))

    def declaration(
            self,
            selection: AnswerContentSelection,
            ) -> LanguageConnectorDiscourseDeclaration | None:
        """仅按当前 selected candidate 集精确匹配，不按顺序或前缀猜测。"""
        if not isinstance(selection, AnswerContentSelection):
            raise TypeError("connector discourse selection 类型错误")
        selected = tuple(sorted(selection.selected_candidate_keys))
        matches = tuple(
            item for item in self.declarations
            if item.candidate_keys == selected
        )
        if len(matches) > 1:
            raise LanguageGenerationConnectorError(
                "当前 selection 存在多个篇章声明")
        return None if not matches else matches[0]

    def state_key(self) -> tuple[int, ...]:
        """返回全部声明的内容锁定键。"""
        result = [len(self.declarations)]
        for declaration in self.declarations:
            result.extend(_packed(declaration.stable_key()))
        return tuple(result)

    def clone_for_evaluation(self) -> "StaticLanguageConnectorDiscourseDeclarations":
        """为 V-06 返回独立不可变声明容器，不复用宿主 provider 实例。"""
        return StaticLanguageConnectorDiscourseDeclarations(self.declarations)


@dataclass(frozen=True)
class BoundPropositionDiscourseDependency:
    """以课程中的精确 BoundProposition 声明一个篇章依赖端点。"""

    before: BoundProposition
    after: BoundProposition
    relation: ObjectIdentity
    reason: ObjectIdentity
    trace: tuple[int, ...]

    def __post_init__(self) -> None:
        """核验依赖不依赖运行期 candidate key 或宿主 query scope。"""
        if not isinstance(self.before, BoundProposition):
            raise TypeError("篇章模板 before 必须是 BoundProposition")
        if not isinstance(self.after, BoundProposition):
            raise TypeError("篇章模板 after 必须是 BoundProposition")
        if self.before == self.after:
            raise ValueError("篇章模板 dependency 不得自环")
        _identity(
            self.relation,
            label="篇章模板 relation",
            kind=OBJECT_STRUCTURE_CONCEPT,
        )
        _identity(
            self.reason,
            label="篇章模板 reason",
            kind=OBJECT_MINIMAL_INSTRUCTION,
        )
        _strict_key(self.trace, label="篇章模板 dependency trace")

    def stable_key(self) -> tuple[int, ...]:
        """返回两个命题端点及一等 relation/reason 的内容锁键。"""
        return (
            *_packed(self.before.stable_key()),
            *_packed(self.after.stable_key()),
            *_packed(self.relation.stable_key()),
            *_packed(self.reason.stable_key()),
            *_packed(self.trace),
        )


@dataclass(frozen=True)
class BoundPropositionDiscourseDeclaration:
    """保存可跨 query scope 复用的课程篇章声明模板。"""

    propositions: tuple[BoundProposition, ...]
    dependencies: tuple[BoundPropositionDiscourseDependency, ...]
    source: SourceRef
    trace: tuple[int, ...]

    def __post_init__(self) -> None:
        """冻结精确命题集并拒绝竞争端点或不完整的课程声明。"""
        if (not isinstance(self.propositions, tuple)
                or not self.propositions
                or any(not isinstance(item, BoundProposition)
                       for item in self.propositions)):
            raise TypeError("篇章模板 propositions 必须是非空 BoundProposition tuple")
        keys = tuple(item.stable_key() for item in self.propositions)
        if len(set(keys)) != len(keys):
            raise ValueError("篇章模板 proposition 不得重复")
        if (not isinstance(self.dependencies, tuple)
                or any(not isinstance(item, BoundPropositionDiscourseDependency)
                       for item in self.dependencies)):
            raise TypeError("篇章模板 dependencies 类型错误")
        if len({item.stable_key() for item in self.dependencies}) != len(
                self.dependencies):
            raise ValueError("篇章模板 dependency 不得重复")
        declared = set(self.propositions)
        if any(item.before not in declared or item.after not in declared
               for item in self.dependencies):
            raise ValueError("篇章模板 dependency 端点必须属于当前命题集")
        if not isinstance(self.source, SourceRef):
            raise TypeError("篇章模板 source 类型错误")
        _strict_key(self.trace, label="篇章模板 trace")
        object.__setattr__(self, "propositions", tuple(sorted(
            self.propositions,
            key=lambda item: item.stable_key(),
        )))
        object.__setattr__(self, "dependencies", tuple(sorted(
            self.dependencies,
            key=lambda item: item.stable_key(),
        )))

    @property
    def proposition_keys(self) -> tuple[tuple[int, ...], ...]:
        """返回排序后的课程命题键，不包含运行期 candidate scope。"""
        return tuple(item.stable_key() for item in self.propositions)

    def instantiate(
            self,
            candidate_keys_by_proposition: dict[tuple[int, ...], tuple[int, ...]],
            ) -> LanguageConnectorDiscourseDeclaration:
        """把当前 selection 的唯一 candidate key 投影为一次运行期篇章声明。"""
        if not isinstance(candidate_keys_by_proposition, dict):
            raise TypeError("篇章模板 candidate 映射必须是 dict")
        expected = set(self.proposition_keys)
        if set(candidate_keys_by_proposition) != expected:
            raise LanguageGenerationConnectorError(
                "篇章模板未精确覆盖当前 selected Proposition 集")
        candidate_keys = []
        for proposition_key in self.proposition_keys:
            candidate_key = candidate_keys_by_proposition[proposition_key]
            _strict_key(candidate_key, label="篇章模板 candidate key")
            candidate_keys.append(candidate_key)
        if len(set(candidate_keys)) != len(candidate_keys):
            raise LanguageGenerationConnectorError(
                "同一 BoundProposition 不得映射多个或重复 candidate")
        dependencies = tuple(
            DiscourseDependency(
                candidate_keys_by_proposition[item.before.stable_key()],
                candidate_keys_by_proposition[item.after.stable_key()],
                item.relation,
                item.reason,
                item.trace,
            )
            for item in self.dependencies
        )
        return LanguageConnectorDiscourseDeclaration(
            tuple(candidate_keys),
            dependencies,
            self.source,
            self.trace,
        )

    def stable_key(self) -> tuple[int, ...]:
        """返回命题、依赖及来源追溯的完整课程内容键。"""
        result = [len(self.propositions)]
        for proposition in self.propositions:
            result.extend(_packed(proposition.stable_key()))
        result.append(len(self.dependencies))
        for dependency in self.dependencies:
            result.extend(_packed(dependency.stable_key()))
        result.extend(_packed(self.source.stable_key()))
        result.extend(_packed(self.trace))
        return tuple(result)


@dataclass(frozen=True)
class BoundPropositionDiscourseDeclarations:
    """按课程 BoundProposition 集实例化当前 query 的来源化篇章声明。"""

    declarations: tuple[BoundPropositionDiscourseDeclaration, ...]

    def __post_init__(self) -> None:
        """拒绝两个课程声明竞争同一精确命题集合。"""
        if (not isinstance(self.declarations, tuple)
                or any(not isinstance(item, BoundPropositionDiscourseDeclaration)
                       for item in self.declarations)):
            raise TypeError("BoundProposition 篇章声明类型错误")
        keys = tuple(item.proposition_keys for item in self.declarations)
        if len(set(keys)) != len(keys):
            raise ValueError("同一 BoundProposition 集不得重复声明篇章顺序")
        object.__setattr__(self, "declarations", tuple(sorted(
            self.declarations,
            key=lambda item: item.stable_key(),
        )))

    def declaration(
            self,
            selection: AnswerContentSelection,
            ) -> LanguageConnectorDiscourseDeclaration | None:
        """仅在 selected Proposition 与课程模板一一对应时生成本次声明。"""
        if not isinstance(selection, AnswerContentSelection):
            raise TypeError("篇章模板 selection 类型错误")
        selected_keys = set(selection.selected_candidate_keys)
        candidates = {
            item.stable_key(): item
            for item in selection.request.candidates
            if item.stable_key() in selected_keys
        }
        if set(candidates) != selected_keys:
            raise LanguageGenerationConnectorError(
                "篇章模板 selection candidate 不可恢复")
        candidate_keys_by_proposition = {}
        for candidate in candidates.values():
            proposition_key = candidate.proposition.stable_key()
            existing = candidate_keys_by_proposition.get(proposition_key)
            if existing is not None and existing != candidate.stable_key():
                raise LanguageGenerationConnectorError(
                    "同一 BoundProposition 命中多个 selected candidate")
            candidate_keys_by_proposition[proposition_key] = candidate.stable_key()
        proposition_keys = tuple(sorted(candidate_keys_by_proposition))
        matches = tuple(
            item for item in self.declarations
            if item.proposition_keys == proposition_keys
        )
        if len(matches) > 1:
            raise LanguageGenerationConnectorError(
                "当前 selected Proposition 存在多个篇章模板")
        return (
            None if not matches
            else matches[0].instantiate(candidate_keys_by_proposition)
        )

    def state_key(self) -> tuple[int, ...]:
        """返回所有课程模板的不可变内容锁键。"""
        result = [len(self.declarations)]
        for declaration in self.declarations:
            result.extend(_packed(declaration.stable_key()))
        return tuple(result)

    def clone_for_evaluation(self) -> "BoundPropositionDiscourseDeclarations":
        """为 V-06 重建独立的不可变声明 provider 容器。"""
        return BoundPropositionDiscourseDeclarations(self.declarations)


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
        selected = self.selected_candidates(selection)
        if len(selected) != 1:
            raise LanguageGenerationConnectorError(
                "单命题 match 不得私选多命题 selection")
        return self.match_candidate(selection, selected[0])

    def selected_candidates(
            self,
            selection: AnswerContentSelection,
            ) -> tuple[object, ...]:
        """返回 G-01 全部显式选择的候选，不赋予容器顺序篇章含义。"""
        if not isinstance(selection, AnswerContentSelection):
            raise TypeError("connector registry 只接受 AnswerContentSelection")
        selected_keys = set(selection.selected_candidate_keys)
        selected = tuple(
            candidate
            for candidate in selection.request.candidates
            if candidate.stable_key() in selected_keys
        )
        if len(selected) != len(selected_keys):
            raise LanguageGenerationConnectorError(
                "connector selection 含请求外或重复 candidate")
        if {candidate.stable_key() for candidate in selected} != selected_keys:
            raise LanguageGenerationConnectorError(
                "connector selection candidate 身份漂移")
        return selected

    def match_candidate(
            self,
            selection: AnswerContentSelection,
            candidate: object,
            ) -> tuple[LanguageGenerationConnectorTemplate, object]:
        """为一个已选候选精确匹配唯一模板，拒绝按模板顺序补全。"""
        selected = self.selected_candidates(selection)
        candidate_key = getattr(candidate, "stable_key", None)
        if not callable(candidate_key):
            raise TypeError("connector candidate 缺少稳定身份")
        key = candidate_key()
        matches = tuple(
            item for item in selected if item.stable_key() == key)
        if len(matches) != 1 or matches[0] != candidate:
            raise LanguageGenerationConnectorError(
                "connector candidate 不属于当前精确 selection")
        match_key = self.match_key_for_candidate(selection, candidate)
        matches = tuple(
            template for template in self.templates
            if template.match_key() == match_key
        )
        if not matches:
            raise LanguageGenerationConnectorError(
                "当前 predicate/structure/LanguageBranch 没有课程模板")
        if len(matches) != 1:
            raise LanguageGenerationConnectorError(
                "当前 predicate/structure/LanguageBranch 存在歧义模板")
        return matches[0], candidate

    def match_key_for_candidate(
            self,
            selection: AnswerContentSelection,
            candidate: object,
            ) -> tuple[ObjectIdentity, ObjectIdentity, ObjectIdentity]:
        """从一个已选命题恢复 LanguageBranch、Structure 和 predicate 匹配键。"""
        if not isinstance(selection, AnswerContentSelection):
            raise TypeError("connector registry 只接受 AnswerContentSelection")
        branch = selection.request.goal.target_branch
        if branch is None:
            raise LanguageGenerationConnectorError(
                "connector 缺少目标 LanguageBranch")
        proposition = getattr(candidate, "proposition", None)
        if not isinstance(proposition, BoundProposition):
            raise TypeError("connector candidate 缺少 BoundProposition")
        return branch, proposition.structure, proposition.predicate

    def match_input(
            self,
            selection: AnswerContentSelection,
            ) -> tuple[
                tuple[ObjectIdentity, ObjectIdentity, ObjectIdentity],
                object,
            ]:
        """从 G-01 实际选择恢复精确 connector 匹配键和唯一命题候选。"""
        selected = self.selected_candidates(selection)
        if len(selected) != 1:
            raise LanguageGenerationConnectorError(
                "connector 首个生产契约要求 G-01 显式收敛为一个命题")
        candidate = selected[0]
        return self.match_key_for_candidate(selection, candidate), candidate

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
    """保留逐候选模板 context，并只消费显式篇章声明确定句序。"""

    def __init__(
            self,
            registry: LanguageGenerationConnectorRegistry,
            declarations: LanguageConnectorDiscourseDeclarationProvider | None = None,
            ) -> None:
        """绑定模板注册表和可注入声明读取器，不从候选容器顺序推篇章。"""
        if not isinstance(registry, LanguageGenerationConnectorRegistry):
            raise TypeError("connector discourse registry 类型错误")
        if declarations is not None and any(
                not hasattr(declarations, name)
                for name in ("declaration", "state_key")):
            raise TypeError("connector discourse declaration provider 协议不完整")
        self.registry = registry
        self.declarations = declarations

    def plan(self, selection: AnswerContentSelection) -> DiscoursePlan:
        """建立所有候选的篇章计划，多命题必须有唯一可证顺序声明。"""
        selected = self.registry.selected_candidates(selection)
        matched = tuple(
            self.registry.match_candidate(selection, candidate)
            for candidate in selected
        )
        declaration = (
            None if self.declarations is None
            else self.declarations.declaration(selection)
        )
        candidate_keys = tuple(sorted(
            candidate.stable_key() for _template, candidate in matched))
        if len(matched) > 1 and declaration is None:
            raise LanguageGenerationConnectorError(
                "多命题 connector 必须注入来源化篇章声明")
        if declaration is not None:
            if declaration.candidate_keys != candidate_keys:
                raise LanguageGenerationConnectorError(
                    "篇章声明未精确覆盖当前 selected candidate 集")
            dependencies = declaration.dependencies
            declaration_source = declaration.source
            declaration_trace = declaration.trace
        else:
            dependencies = ()
            declaration_source = None
            declaration_trace = ()
        questions = tuple(
            obligation
            for _template, candidate in matched
            if candidate.reasoning is not None
            for obligation in candidate.reasoning.unresolved
        )
        if len(set(questions)) != len(questions):
            raise LanguageGenerationConnectorError(
                "不同 selected candidate 重复声明同一 open question")
        context = tuple(sorted(
            {
                value
                for template, _candidate in matched
                for value in template.context
            },
            key=ObjectIdentity.stable_key,
        ))
        return DiscoursePlan(
            selection.stable_key(),
            candidate_keys,
            dependencies,
            questions,
            context,
            require_unique_order=bool(matched),
            declaration_source=declaration_source,
            declaration_trace=declaration_trace,
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
    """逐候选按课程模板投影为带运行期句实例的 S-07 义务。"""

    def __init__(self, registry: LanguageGenerationConnectorRegistry) -> None:
        self.registry = registry

    def plan(
            self,
            selection: AnswerContentSelection,
            discourse: DiscoursePlan,
            propositions: PropositionPlan,
    ) -> SyntaxPlan:
        """按唯一篇章拓扑序建立逐候选句实例，不复用模板句式地址。"""
        if discourse.selection_key != selection.stable_key():
            raise ValueError("connector syntax mapper 收到漂移 discourse")
        planned = {item.candidate_key: item for item in propositions.propositions}
        selected = {
            candidate.stable_key(): candidate
            for candidate in self.registry.selected_candidates(selection)
        }
        if set(planned) != set(selected):
            raise ValueError("connector syntax mapper 收到漂移 PropositionPlan")
        if set(discourse.topological_order) != set(selected):
            raise RuntimeError("connector discourse 拓扑顺序未覆盖当前 selection")
        sentences = []
        obligations = []
        for ordinal, key in enumerate(discourse.topological_order):
            candidate = selected.get(key)
            if candidate is None:
                raise ValueError("connector discourse 引用了 selection 外 candidate")
            template, matched_candidate = self.registry.match_candidate(
                selection, candidate)
            if matched_candidate != candidate:
                raise RuntimeError("connector 匹配替换了 selected candidate")
            if planned[key].proposition != candidate.proposition:
                raise ValueError("connector syntax mapper 丢失 BoundProposition")
            values = self.registry.values(template, candidate.proposition)
            proposition_values = tuple(
                value
                for binding, value in zip(template.bindings, values)
                if binding.source == self.registry.value_protocol.proposition_source
            )
            if len(proposition_values) != 1:
                raise RuntimeError("connector 命题本体槽位预检失效")
            instance = GenerationSentenceInstance(
                template.sentence,
                key,
                ordinal,
                candidate.source,
                candidate.scope,
            )
            sentence = PlannedSentence(
                template.sentence,
                template.structure,
                ordinal,
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
                instance=instance,
            )
            sentences.append(sentence)
            obligations.append(SyntaxLinearizationObligation(
                template.sentence,
                template.structure,
                sentence.values,
                template.constraints,
                template.context,
                template.linearization_reason,
                candidate.source,
                candidate.scope,
                instance,
            ))
        return SyntaxPlan(
            selection.stable_key(),
            tuple(sentences),
            (),
            tuple(obligations),
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
                    sentence.address,
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
        """逐句绑定 exact 模板归属并生成不共享的 R-01 use key。"""
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
        selected = {
            candidate.stable_key(): candidate
            for candidate in self.registry.selected_candidates(
                structure.selection)
        }
        result = []
        for sentence in structure.syntax.sentences:
            if not isinstance(sentence.instance, GenerationSentenceInstance):
                raise ValueError("connector surface mapper 缺少运行期句实例")
            candidate = selected.get(sentence.instance.candidate_key)
            if candidate is None:
                raise ValueError("connector surface mapper 句实例引用 selection 外 candidate")
            template, matched_candidate = self.registry.match_candidate(
                structure.selection,
                candidate,
            )
            if (matched_candidate != candidate
                    or sentence.instance.template != template.sentence
                    or sentence.sentence != template.sentence
                    or sentence.structure != template.structure):
                raise ValueError("connector surface mapper sentence/template 漂移")
            if branch != template.language_branch:
                raise ValueError("connector surface mapper 目标分支漂移")
            runtime = self.runtime_policy.template_for(template.connector)
            theory_by_slot = {item.slot: item for item in template.surface}
            runtime_by_slot = {item.slot: item for item in runtime.surface}
            if set(theory_by_slot) != set(runtime_by_slot):
                raise ValueError("connector surface 理论与运行策略 slot 不一致")
            for value in sentence.values:
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
                        *_packed(sentence.instance.stable_key()),
                        *_packed(value.slot.stable_key()),
                        *_packed(policy.use_key_suffix),
                    )
                result.append(SurfaceSlotDirective(
                    sentence.instance,
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
        """仅兼容单句 caller；多句必须转由逐句归属 mapper 处理。"""
        if len(structure.syntax.sentences) != 1:
            return None
        sentence = structure.syntax.sentences[0]
        if not isinstance(sentence.instance, GenerationSentenceInstance):
            return None
        selected = {
            candidate.stable_key(): candidate
            for candidate in self.registry.selected_candidates(
                structure.selection)
        }
        candidate = selected.get(sentence.instance.candidate_key)
        if candidate is None:
            raise LanguageGenerationConnectorError(
                "surface attribution 句实例不属于当前 selection")
        template, _matched = self.registry.match_candidate(
            structure.selection,
            candidate,
        )
        return self.attributions.get(template.connector)


class LanguageConnectorSurfaceSentenceAttributionMapper:
    """为每个运行期句实例绑定 exact connector/Hypothesis/purpose。"""

    def __init__(
            self,
            registry: LanguageGenerationConnectorRegistry,
            attributions: tuple[GenerationSurfaceAttribution, ...],
            ) -> None:
        """建立理论到生命周期归属的只读映射，拒绝遗漏和重复。"""
        self._legacy = LanguageConnectorSurfaceAttributionMapper(
            registry,
            attributions,
        )
        self.registry = registry
        self.attributions = self._legacy.attributions

    def attributions_for(
            self,
            structure: GenerationStructurePlan,
            ) -> tuple[GenerationSurfaceSentenceAttribution, ...]:
        """逐句读取 exact 模板，不以 stable sort 或共享 theory 推断归属。"""
        if not isinstance(structure, GenerationStructurePlan):
            raise TypeError("connector sentence attribution 需要 GenerationStructurePlan")
        if not self.attributions:
            return ()
        selected = {
            candidate.stable_key(): candidate
            for candidate in self.registry.selected_candidates(
                structure.selection)
        }
        result = []
        for sentence in structure.syntax.sentences:
            instance = sentence.instance
            if not isinstance(instance, GenerationSentenceInstance):
                raise LanguageGenerationConnectorError(
                    "connector sentence attribution 缺少运行期句实例")
            candidate = selected.get(instance.candidate_key)
            if candidate is None:
                raise LanguageGenerationConnectorError(
                    "connector sentence attribution 句实例引用 selection 外 candidate")
            template, matched_candidate = self.registry.match_candidate(
                structure.selection,
                candidate,
            )
            attribution = self.attributions.get(template.connector)
            if attribution is None:
                raise LanguageGenerationConnectorError(
                    "connector sentence attribution 缺少理论 Hypothesis")
            if (matched_candidate != candidate
                    or instance.template != template.sentence
                    or sentence.proposition_keys != (instance.candidate_key,)):
                raise LanguageGenerationConnectorError(
                    "connector sentence attribution 与模板或 candidate 漂移")
            result.append(GenerationSurfaceSentenceAttribution(
                instance,
                attribution.theory,
                attribution.hypothesis,
                attribution.purpose,
            ))
        return tuple(result)


class LanguageGenerationConnector:
    """装配共享 G-02 mapper 与执行 S-07/R-01 的 G-03 request builder。"""

    def __init__(
            self,
            registry: LanguageGenerationConnectorRegistry,
            runtime_policy: LanguageGenerationConnectorRuntimePolicy,
            surface_protocol: GenerationSurfaceProtocol,
            attributions: tuple[GenerationSurfaceAttribution, ...] = (),
            discourse_declarations: LanguageConnectorDiscourseDeclarationProvider
            | None = None,
            ) -> None:
        if not isinstance(registry, LanguageGenerationConnectorRegistry):
            raise TypeError("language generation connector registry 类型错误")
        if not isinstance(
                runtime_policy, LanguageGenerationConnectorRuntimePolicy):
            raise TypeError("language generation connector runtime policy 类型错误")
        if not isinstance(surface_protocol, GenerationSurfaceProtocol):
            raise TypeError("language generation connector surface protocol 类型错误")
        if discourse_declarations is not None and any(
                not hasattr(discourse_declarations, name)
                for name in ("declaration", "state_key")):
            raise TypeError("language generation connector discourse provider 协议不完整")
        self.registry = registry
        self.runtime_policy = runtime_policy
        self.surface_protocol = surface_protocol
        self.discourse_declarations = discourse_declarations
        self.attribution_mapper = LanguageConnectorSurfaceAttributionMapper(
            registry,
            attributions,
        )
        self.sentence_attribution_mapper = (
            LanguageConnectorSurfaceSentenceAttributionMapper(
                registry,
                attributions,
            )
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
            LanguageConnectorDiscourseMapper(
                self.registry,
                self.discourse_declarations,
            ),
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
            self.sentence_attribution_mapper,
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
            0 if self.discourse_declarations is None else 1,
            *(() if self.discourse_declarations is None
              else _packed(self.discourse_declarations.state_key())),
        )


__all__ = [
    "BoundPropositionDiscourseDeclaration",
    "BoundPropositionDiscourseDeclarations",
    "BoundPropositionDiscourseDependency",
    "LanguageConnectorDiscourseDeclaration",
    "LanguageConnectorDiscourseDeclarationProvider",
    "LanguageConnectorDiscourseMapper",
    "LanguageConnectorExecutionRequestMapper",
    "LanguageConnectorOrdinalDefinition",
    "LanguageConnectorPropositionMapper",
    "LanguageConnectorSlotBinding",
    "LanguageConnectorSurfaceDirective",
    "LanguageConnectorSurfaceDirectiveMapper",
    "LanguageConnectorSurfaceAttributionMapper",
    "LanguageConnectorSurfaceSentenceAttributionMapper",
    "LanguageConnectorSurfaceRuntimePolicy",
    "LanguageConnectorTemplateRuntimePolicy",
    "LanguageConnectorSyntaxMapper",
    "LanguageConnectorValueProtocol",
    "LanguageGenerationConnector",
    "LanguageGenerationConnectorError",
    "LanguageGenerationConnectorRegistry",
    "LanguageGenerationConnectorRuntimePolicy",
    "LanguageGenerationConnectorTemplate",
    "StaticLanguageConnectorDiscourseDeclarations",
]
