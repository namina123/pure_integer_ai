"""S-03 typed BindingFrame、作用域环境和捕获规避 substitution。

Binder、Variable、Role 和 Proposition template 使用 S-00 一等身份；BindingFrame、当前值和
bound proposition 只作为运行期不可变视图存在。本模块不写图、不持久化，也不根据名称、位置
或旧槽位编号猜变量身份。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_BINDER,
    OBJECT_CONCEPT,
    OBJECT_MINIMAL_INSTRUCTION,
    OBJECT_PROPOSITION,
    OBJECT_ROLE,
    OBJECT_STRUCTURE_CONCEPT,
    OBJECT_VARIABLE,
    ObjectIdentity,
    object_contracts_by_kind,
)
from pure_integer_ai.cognition.shared.semantic_object import (
    AtomicPropositionDefinition,
    SEMANTIC_OBJECT_KINDS,
    describe_variable,
    semantic_source,
    validate_semantic_identity,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int


_BOUND_VIEW_KEY_VERSION = 1
_TYPE_KINDS = frozenset({
    OBJECT_CONCEPT,
    OBJECT_STRUCTURE_CONCEPT,
    OBJECT_ROLE,
})


def _packed(key: tuple[int, ...]) -> tuple[int, ...]:
    """给运行期稳定键的可变长分段添加长度，避免拼接歧义。"""
    return len(key), *key


def _require_kind(
        identity: ObjectIdentity, expected: int, *, label: str,
        ) -> ObjectIdentity:
    """核验一等对象的精确 object kind，不读取名称或表层表示。"""
    if not isinstance(identity, ObjectIdentity):
        raise TypeError(f"{label} 必须是 ObjectIdentity")
    if identity.object_kind != expected:
        raise ValueError(f"{label} 对象类型不匹配")
    return identity


def _require_type(identity: ObjectIdentity, *, label: str) -> ObjectIdentity:
    """核验 Variable 类型属于 S-00 开放概念层并通过完整身份校验。"""
    if not isinstance(identity, ObjectIdentity):
        raise TypeError(f"{label} 必须是 ObjectIdentity")
    if identity.object_kind not in _TYPE_KINDS:
        raise ValueError(f"{label} 必须是 Concept/StructureConcept/Role")
    validate_semantic_identity(identity)
    return identity


def _require_authoritative(
        identity: ObjectIdentity, *, label: str,
        ) -> ObjectIdentity:
    """拒绝 legacy projection 或损坏的完整身份充当运行期绑定值。"""
    if not isinstance(identity, ObjectIdentity):
        raise TypeError(f"{label} 必须是 ObjectIdentity")
    contract = object_contracts_by_kind().get(identity.object_kind)
    if contract is None or not contract.authoritative_identity:
        raise ValueError(f"{label} 必须是权威对象身份")
    if ObjectIdentity.from_stable_key(identity.stable_key()) != identity:
        raise ValueError(f"{label} 完整身份不能稳定 round-trip")
    if (identity.object_kind in SEMANTIC_OBJECT_KINDS
            or identity.object_kind in {
                OBJECT_CONCEPT, OBJECT_STRUCTURE_CONCEPT}):
        validate_semantic_identity(identity)
    return identity


@dataclass(frozen=True)
class BindingFailureProtocol:
    """为每类失败注入 MinimalInstruction reason，不冻结 reason 数值或文字。"""

    duplicate_variable: ObjectIdentity
    binder_mismatch: ObjectIdentity
    type_rejected: ObjectIdentity
    type_unknown: ObjectIdentity
    unbound_variable: ObjectIdentity
    scope_conflict: ObjectIdentity
    template_missing: ObjectIdentity
    proposition_cycle: ObjectIdentity
    legacy_mapping_missing: ObjectIdentity

    def reasons(self) -> tuple[ObjectIdentity, ...]:
        """按机制槽位返回全部 reason，供统一分型和互异检查。"""
        return (
            self.duplicate_variable,
            self.binder_mismatch,
            self.type_rejected,
            self.type_unknown,
            self.unbound_variable,
            self.scope_conflict,
            self.template_missing,
            self.proposition_cycle,
            self.legacy_mapping_missing,
        )

    def __post_init__(self) -> None:
        if len(set(self.reasons())) != len(self.reasons()):
            raise ValueError("binding failure reason 必须互不相同")
        for reason in self.reasons():
            _require_kind(
                reason, OBJECT_MINIMAL_INSTRUCTION,
                label="binding failure reason")


@dataclass(frozen=True)
class BindingFailure:
    """不依赖异常文字的 typed 失败结果，可由上层 trace 保存完整身份。"""

    reason: ObjectIdentity
    variable: ObjectIdentity | None = None
    binder: ObjectIdentity | None = None
    expected_type: ObjectIdentity | None = None
    actual_type: ObjectIdentity | None = None
    proposition: ObjectIdentity | None = None
    details: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        _require_kind(
            self.reason, OBJECT_MINIMAL_INSTRUCTION,
            label="binding failure reason")
        if self.variable is not None:
            describe_variable(self.variable)
        if self.binder is not None:
            _require_kind(self.binder, OBJECT_BINDER, label="failure binder")
        if self.expected_type is not None:
            _require_type(self.expected_type, label="failure expected_type")
        if self.actual_type is not None:
            _require_type(self.actual_type, label="failure actual_type")
        if self.proposition is not None:
            _require_kind(
                self.proposition, OBJECT_PROPOSITION,
                label="failure proposition")
        if not isinstance(self.details, tuple):
            raise TypeError("failure details 必须是整数 tuple")
        assert_int(*self.details, _where="BindingFailure.details")
        if any(type(item) is not int for item in self.details):
            raise ValueError("failure details 必须使用严格整数")


class TypedBindingError(ValueError):
    """携带一等 MinimalInstruction reason 和相关对象全键的绑定失败。"""

    def __init__(self, failure: BindingFailure) -> None:
        if not isinstance(failure, BindingFailure):
            raise TypeError("failure 必须是 BindingFailure")
        self.failure = failure
        super().__init__("typed binding/substitution 失败")


@dataclass(frozen=True)
class TypeCompatibilityResult:
    """类型 resolver 的三态结果；support 只保存调用方注入的一等依据。"""

    expected_type: ObjectIdentity
    actual_type: ObjectIdentity
    compatible: bool | None
    support: tuple[ObjectIdentity, ...] = ()

    def __post_init__(self) -> None:
        _require_type(self.expected_type, label="expected_type")
        _require_type(self.actual_type, label="actual_type")
        if self.compatible is not None and type(self.compatible) is not bool:
            raise TypeError("compatible 必须是 bool 或 None")
        if not isinstance(self.support, tuple):
            raise TypeError("type support 必须是 ObjectIdentity tuple")
        for identity in self.support:
            _require_authoritative(identity, label="type support")


class TypeCompatibilityResolver(Protocol):
    """由类型关系或受限 verifier 实现的定向三态兼容协议。"""

    def resolve(
            self, expected_type: ObjectIdentity,
            actual_type: ObjectIdentity,
            ) -> TypeCompatibilityResult:
        """判断 actual 是否可填 expected；缺依据必须返回 unknown。"""
        ...


class ExactTypeCompatibilityResolver:
    """默认只承认完整类型身份相等，其他组合保持 unknown。"""

    def resolve(
            self, expected_type: ObjectIdentity,
            actual_type: ObjectIdentity,
            ) -> TypeCompatibilityResult:
        """按完整 ObjectIdentity 比较类型，不解释名称或关系层级。"""
        _require_type(expected_type, label="expected_type")
        _require_type(actual_type, label="actual_type")
        compatible = True if expected_type == actual_type else None
        return TypeCompatibilityResult(
            expected_type, actual_type, compatible)


@dataclass(frozen=True)
class TypedValue:
    """运行期绑定值及调用方声明的一等类型；它不是 Core 属性行。"""

    value: ObjectIdentity
    value_type: ObjectIdentity

    def __post_init__(self) -> None:
        _require_authoritative(self.value, label="typed value")
        _require_type(self.value_type, label="typed value type")
        if self.value.object_kind == OBJECT_VARIABLE:
            descriptor = describe_variable(self.value)
            if descriptor.value_type != self.value_type:
                raise ValueError("Variable 值的声明类型与自身完整身份不一致")

    def stable_key(self) -> tuple[int, ...]:
        """返回只用于运行期确定性比较的完整值和类型键。"""
        value_key = self.value.stable_key()
        type_key = self.value_type.stable_key()
        return (*_packed(value_key), *_packed(type_key))


@dataclass(frozen=True)
class TypedBindingAssignment:
    """一个已由 resolver 明确通过的精确 Variable 到 TypedValue 赋值。"""

    variable: ObjectIdentity
    value: TypedValue
    type_check: TypeCompatibilityResult

    def __post_init__(self) -> None:
        descriptor = describe_variable(self.variable)
        if not isinstance(self.value, TypedValue):
            raise TypeError("assignment value 必须是 TypedValue")
        if not isinstance(self.type_check, TypeCompatibilityResult):
            raise TypeError("type_check 必须是 TypeCompatibilityResult")
        if self.type_check.compatible is not True:
            raise ValueError("assignment 只能保存已明确通过的类型检查")
        if (self.type_check.expected_type != descriptor.value_type
                or self.type_check.actual_type != self.value.value_type):
            raise ValueError("assignment 类型检查与 Variable/TypedValue 不一致")

    @classmethod
    def create(
            cls, variable: ObjectIdentity, value: TypedValue, *,
            resolver: TypeCompatibilityResolver,
            failures: BindingFailureProtocol,
            ) -> "TypedBindingAssignment":
        """执行注入式类型检查，unknown 与明确拒绝使用不同 reason fail closed。"""
        descriptor = describe_variable(variable)
        if not isinstance(value, TypedValue):
            raise TypeError("value 必须是 TypedValue")
        result = resolver.resolve(descriptor.value_type, value.value_type)
        if not isinstance(result, TypeCompatibilityResult):
            raise TypeError("type resolver 必须返回 TypeCompatibilityResult")
        if (result.expected_type != descriptor.value_type
                or result.actual_type != value.value_type):
            raise ValueError("type resolver 返回了其他类型对的结果")
        if result.compatible is not True:
            reason = (
                failures.type_rejected
                if result.compatible is False
                else failures.type_unknown
            )
            raise TypedBindingError(BindingFailure(
                reason,
                variable=variable,
                binder=descriptor.binder,
                expected_type=descriptor.value_type,
                actual_type=value.value_type,
            ))
        return cls(variable, value, result)

    def stable_key(self) -> tuple[int, ...]:
        """返回 Variable、当前值、类型和兼容 support 的完整运行期键。"""
        variable_key = self.variable.stable_key()
        value_key = self.value.stable_key()
        result: list[int] = [
            *_packed(variable_key),
            *_packed(value_key),
            len(self.type_check.support),
        ]
        for support in self.type_check.support:
            result.extend(_packed(support.stable_key()))
        return tuple(result)


@dataclass(frozen=True)
class BindingFrame:
    """同一精确 Binder 下的确定性运行期赋值集合。"""

    binder: ObjectIdentity
    assignments: tuple[TypedBindingAssignment, ...]

    def __post_init__(self) -> None:
        _require_kind(self.binder, OBJECT_BINDER, label="binding frame binder")
        semantic_source(self.binder)
        if not isinstance(self.assignments, tuple):
            raise TypeError("assignments 必须是 TypedBindingAssignment tuple")
        if any(not isinstance(item, TypedBindingAssignment)
               for item in self.assignments):
            raise TypeError("assignments 只能包含 TypedBindingAssignment")
        seen: set[ObjectIdentity] = set()
        for assignment in self.assignments:
            descriptor = describe_variable(assignment.variable)
            if descriptor.binder != self.binder:
                raise ValueError("BindingFrame 混入其他 Binder 的 Variable")
            if assignment.variable in seen:
                raise ValueError("BindingFrame 重复 Variable")
            seen.add(assignment.variable)
        object.__setattr__(self, "assignments", tuple(sorted(
            self.assignments,
            key=lambda item: item.variable.stable_key(),
        )))

    @classmethod
    def create(
            cls, binder: ObjectIdentity,
            assignments: tuple[TypedBindingAssignment, ...], *,
            failures: BindingFailureProtocol,
            ) -> "BindingFrame":
        """先给重复变量和 Binder 错配生成结构化 reason，再构造规范 frame。"""
        _require_kind(binder, OBJECT_BINDER, label="binding frame binder")
        if not isinstance(assignments, tuple):
            raise TypeError("assignments 必须是 TypedBindingAssignment tuple")
        seen: set[ObjectIdentity] = set()
        for assignment in assignments:
            if not isinstance(assignment, TypedBindingAssignment):
                raise TypeError("assignments 只能包含 TypedBindingAssignment")
            descriptor = describe_variable(assignment.variable)
            if descriptor.binder != binder:
                raise TypedBindingError(BindingFailure(
                    failures.binder_mismatch,
                    variable=assignment.variable,
                    binder=binder,
                ))
            if assignment.variable in seen:
                raise TypedBindingError(BindingFailure(
                    failures.duplicate_variable,
                    variable=assignment.variable,
                    binder=binder,
                ))
            seen.add(assignment.variable)
        return cls(binder, assignments)

    def assignment_for(
            self, variable: ObjectIdentity,
            ) -> TypedBindingAssignment | None:
        """按完整 Variable identity 查赋值，不比较 local key 或表层名称。"""
        describe_variable(variable)
        for assignment in self.assignments:
            if assignment.variable == variable:
                return assignment
        return None

    def stable_key(self) -> tuple[int, ...]:
        """返回只用于运行期回放诊断的 Binder 和赋值全键。"""
        binder_key = self.binder.stable_key()
        result: list[int] = [*_packed(binder_key), len(self.assignments)]
        for assignment in self.assignments:
            result.extend(_packed(assignment.stable_key()))
        return tuple(result)


@dataclass(frozen=True)
class BindingEnvironment:
    """按 outer 到 inner 顺序保存 frame 的不可变作用域栈。"""

    frames: tuple[BindingFrame, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.frames, tuple):
            raise TypeError("frames 必须是 BindingFrame tuple")
        if any(not isinstance(frame, BindingFrame) for frame in self.frames):
            raise TypeError("frames 只能包含 BindingFrame")
        binders = tuple(frame.binder for frame in self.frames)
        if len(set(binders)) != len(binders):
            raise ValueError("同一 Binder 不得在环境栈重复出现")

    def push(
            self, frame: BindingFrame, *,
            failures: BindingFailureProtocol,
            ) -> "BindingEnvironment":
        """返回加入 inner frame 的新环境，重复 Binder 以结构化 scope reason 失败。"""
        if not isinstance(frame, BindingFrame):
            raise TypeError("frame 必须是 BindingFrame")
        if any(existing.binder == frame.binder for existing in self.frames):
            raise TypedBindingError(BindingFailure(
                failures.scope_conflict,
                binder=frame.binder,
            ))
        return BindingEnvironment((*self.frames, frame))

    def has_binder(self, binder: ObjectIdentity) -> bool:
        """判断精确 Binder 是否在当前环境，不按来源内局部键折叠。"""
        _require_kind(binder, OBJECT_BINDER, label="environment binder")
        return any(frame.binder == binder for frame in self.frames)

    def lookup(
            self, variable: ObjectIdentity,
            ) -> TypedBindingAssignment | None:
        """从 inner 到 outer 查精确 Binder/Variable；同 local key 不参与匹配。"""
        descriptor = describe_variable(variable)
        for frame in reversed(self.frames):
            if frame.binder != descriptor.binder:
                continue
            return frame.assignment_for(variable)
        return None

    def resolve(
            self, variable: ObjectIdentity, *,
            failures: BindingFailureProtocol,
            ) -> TypedBindingAssignment:
        """返回精确赋值；Binder 不在栈或 frame 未赋该变量都返回 unbound reason。"""
        descriptor = describe_variable(variable)
        assignment = self.lookup(variable)
        if assignment is None:
            raise TypedBindingError(BindingFailure(
                failures.unbound_variable,
                variable=variable,
                binder=descriptor.binder,
            ))
        return assignment

    def stable_key(self) -> tuple[int, ...]:
        """按作用域栈顺序返回完整 frame 键，供确定性回放比较。"""
        result: list[int] = [len(self.frames)]
        for frame in self.frames:
            result.extend(_packed(frame.stable_key()))
        return tuple(result)


@dataclass(frozen=True)
class SubstitutionProtocol:
    """注入 substitution MinimalInstruction 和全部结构化失败 reason。"""

    instruction: ObjectIdentity
    failures: BindingFailureProtocol

    def __post_init__(self) -> None:
        _require_kind(
            self.instruction, OBJECT_MINIMAL_INSTRUCTION,
            label="substitution instruction")
        if not isinstance(self.failures, BindingFailureProtocol):
            raise TypeError("failures 必须是 BindingFailureProtocol")


@dataclass(frozen=True)
class ScopedPropositionTemplate:
    """原子 Proposition template 及其显式引入的 Binder 集合。"""

    definition: AtomicPropositionDefinition
    structure: ObjectIdentity
    introduced_binders: tuple[ObjectIdentity, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.definition, AtomicPropositionDefinition):
            raise TypeError("definition 必须是 AtomicPropositionDefinition")
        _require_kind(
            self.structure, OBJECT_STRUCTURE_CONCEPT,
            label="template structure")
        validate_semantic_identity(self.structure)
        if not isinstance(self.introduced_binders, tuple):
            raise TypeError("introduced_binders 必须是 Binder tuple")
        if len(set(self.introduced_binders)) != len(self.introduced_binders):
            raise ValueError("同一 template 不得重复引入 Binder")
        for binder in self.introduced_binders:
            _require_kind(binder, OBJECT_BINDER, label="template binder")
            if semantic_source(binder) != self.definition.source:
                raise ValueError("template Binder 与 Proposition 来源不一致")
        object.__setattr__(self, "introduced_binders", tuple(sorted(
            self.introduced_binders,
            key=lambda item: item.stable_key(),
        )))


@dataclass(frozen=True)
class PropositionTemplateGraph:
    """供一次 substitution 只读的 Proposition template 索引。"""

    templates: tuple[ScopedPropositionTemplate, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.templates, tuple) or not self.templates:
            raise ValueError("template graph 至少需要一个 Proposition")
        if any(not isinstance(item, ScopedPropositionTemplate)
               for item in self.templates):
            raise TypeError("templates 只能包含 ScopedPropositionTemplate")
        identities = tuple(
            item.definition.proposition for item in self.templates)
        if len(set(identities)) != len(identities):
            raise ValueError("template graph 不得重复 Proposition 身份")
        introduced = tuple(
            binder
            for item in self.templates
            for binder in item.introduced_binders
        )
        if len(set(introduced)) != len(introduced):
            raise ValueError("同一 Binder 不得由多个 Proposition template 引入")
        object.__setattr__(self, "templates", tuple(sorted(
            self.templates,
            key=lambda item: item.definition.proposition.stable_key(),
        )))

    def get(
            self, proposition: ObjectIdentity,
            ) -> ScopedPropositionTemplate | None:
        """按完整 Proposition identity 返回 template；未知嵌套命题保持 opaque。"""
        _require_kind(proposition, OBJECT_PROPOSITION, label="template lookup")
        for template in self.templates:
            if template.definition.proposition == proposition:
                return template
        return None

    def lexical_binders(self) -> frozenset[ObjectIdentity]:
        """返回图内显式引入的 Binder 集，用于区分词法声明和自由声明。"""
        return frozenset(
            binder
            for template in self.templates
            for binder in template.introduced_binders
        )


@dataclass(frozen=True)
class BoundRoleBinding:
    """bound view 中保留精确 Role/ordinal 和对象或嵌套命题 filler。"""

    role: ObjectIdentity
    filler: ObjectIdentity | "BoundProposition"
    ordinal: int = 0

    def __post_init__(self) -> None:
        _require_kind(self.role, OBJECT_ROLE, label="bound role")
        if isinstance(self.filler, ObjectIdentity):
            _require_authoritative(self.filler, label="bound filler")
        elif not isinstance(self.filler, BoundProposition):
            raise TypeError("bound filler 必须是 ObjectIdentity 或 BoundProposition")
        assert_int(self.ordinal, _where="BoundRoleBinding.ordinal")
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ValueError("bound ordinal 必须为非负严格整数")

    def stable_key(self) -> tuple[int, ...]:
        """返回 Role、ordinal 和分型 filler 的递归运行期键。"""
        role_key = self.role.stable_key()
        if isinstance(self.filler, ObjectIdentity):
            filler_key = self.filler.stable_key()
            filler_tag = 1
        else:
            filler_key = self.filler.stable_key()
            filler_tag = 2
        return (
            *_packed(role_key),
            self.ordinal,
            filler_tag,
            *_packed(filler_key),
        )


@dataclass(frozen=True)
class BoundProposition:
    """不可物化的运行期命题视图，保存 template、指令、scope 与替换结果。"""

    template: ObjectIdentity
    instruction: ObjectIdentity
    predicate: ObjectIdentity
    structure: ObjectIdentity
    source_anchor: ObjectIdentity
    context: ObjectIdentity
    introduced_binders: tuple[ObjectIdentity, ...]
    bindings: tuple[BoundRoleBinding, ...]
    applied_variables: tuple[ObjectIdentity, ...]

    def __post_init__(self) -> None:
        _require_kind(self.template, OBJECT_PROPOSITION, label="bound template")
        _require_kind(
            self.instruction, OBJECT_MINIMAL_INSTRUCTION,
            label="bound instruction")
        _require_kind(self.predicate, OBJECT_CONCEPT, label="bound predicate")
        _require_kind(
            self.structure, OBJECT_STRUCTURE_CONCEPT,
            label="bound structure")
        validate_semantic_identity(self.structure)
        if not isinstance(self.introduced_binders, tuple):
            raise TypeError("introduced_binders 必须是 Binder tuple")
        for binder in self.introduced_binders:
            _require_kind(binder, OBJECT_BINDER, label="bound binder")
        if not isinstance(self.bindings, tuple):
            raise TypeError("bindings 必须是 BoundRoleBinding tuple")
        if any(not isinstance(item, BoundRoleBinding) for item in self.bindings):
            raise TypeError("bindings 只能包含 BoundRoleBinding")
        slots = tuple((item.role, item.ordinal) for item in self.bindings)
        if len(set(slots)) != len(slots):
            raise ValueError("bound view 同一 Role+ordinal 不得重复")
        if not isinstance(self.applied_variables, tuple):
            raise TypeError("applied_variables 必须是 Variable tuple")
        for variable in self.applied_variables:
            describe_variable(variable)
        if len(set(self.applied_variables)) != len(self.applied_variables):
            raise ValueError("applied_variables 不得重复")
        object.__setattr__(self, "bindings", tuple(sorted(
            self.bindings,
            key=lambda item: (item.role.stable_key(), item.ordinal),
        )))
        object.__setattr__(self, "applied_variables", tuple(sorted(
            self.applied_variables,
            key=lambda item: item.stable_key(),
        )))

    def stable_key(self) -> tuple[int, ...]:
        """递归返回完整 bound view 键；该键只作运行期比较，不是 Core 对象身份。"""
        template_key = self.template.stable_key()
        instruction_key = self.instruction.stable_key()
        predicate_key = self.predicate.stable_key()
        structure_key = self.structure.stable_key()
        anchor_key = self.source_anchor.stable_key()
        context_key = self.context.stable_key()
        result: list[int] = [
            _BOUND_VIEW_KEY_VERSION,
            *_packed(template_key),
            *_packed(instruction_key),
            *_packed(predicate_key),
            *_packed(structure_key),
            *_packed(anchor_key),
            *_packed(context_key),
            len(self.introduced_binders),
        ]
        for binder in self.introduced_binders:
            result.extend(_packed(binder.stable_key()))
        result.append(len(self.bindings))
        for binding in self.bindings:
            result.extend(_packed(binding.stable_key()))
        result.append(len(self.applied_variables))
        for variable in self.applied_variables:
            result.extend(_packed(variable.stable_key()))
        return tuple(result)


class PropositionSubstituter:
    """按精确 Variable/Binder identity 递归形成纯运行期 bound proposition。"""

    def __init__(self, protocol: SubstitutionProtocol) -> None:
        if not isinstance(protocol, SubstitutionProtocol):
            raise TypeError("protocol 必须是 SubstitutionProtocol")
        self.protocol = protocol

    def substitute(
            self, root: ObjectIdentity, graph: PropositionTemplateGraph,
            environment: BindingEnvironment,
            inherited_binders: tuple[ObjectIdentity, ...] = (),
            ) -> BoundProposition:
        """替换精确 Variable 并按显式祖先 Binder 继续校验 template 词法作用域。"""
        _require_kind(root, OBJECT_PROPOSITION, label="substitution root")
        if not isinstance(graph, PropositionTemplateGraph):
            raise TypeError("graph 必须是 PropositionTemplateGraph")
        if not isinstance(environment, BindingEnvironment):
            raise TypeError("environment 必须是 BindingEnvironment")
        if not isinstance(inherited_binders, tuple):
            raise TypeError("inherited_binders 必须是 Binder tuple")
        if len(set(inherited_binders)) != len(inherited_binders):
            raise ValueError("inherited_binders 不得重复")
        for binder in inherited_binders:
            _require_kind(binder, OBJECT_BINDER, label="inherited binder")
            if not environment.has_binder(binder):
                raise ValueError("inherited Binder 必须在 BindingEnvironment 中")
        if graph.get(root) is None:
            raise TypedBindingError(BindingFailure(
                self.protocol.failures.template_missing,
                proposition=root,
            ))
        lexical_binders = graph.lexical_binders()
        active_templates: set[ObjectIdentity] = set()
        memo: dict[
            tuple[ObjectIdentity, frozenset[ObjectIdentity]],
            BoundProposition,
        ] = {}

        def require_visible(
                variable: ObjectIdentity,
                visible_binders: frozenset[ObjectIdentity], *,
                proposition: ObjectIdentity,
                ) -> None:
            """拒绝引用图内已声明但在当前递归路径不可见的 Variable。"""
            descriptor = describe_variable(variable)
            if (descriptor.binder in lexical_binders
                    and descriptor.binder not in visible_binders):
                raise TypedBindingError(BindingFailure(
                    self.protocol.failures.scope_conflict,
                    variable=variable,
                    binder=descriptor.binder,
                    proposition=proposition,
                ))

        def visit(
                proposition: ObjectIdentity,
                inherited_binders: frozenset[ObjectIdentity],
                ) -> BoundProposition:
            """按词法路径递归绑定 template，memo 不跨不同 Binder 可见域复用。"""
            template = graph.get(proposition)
            if template is None:
                raise TypedBindingError(BindingFailure(
                    self.protocol.failures.template_missing,
                    proposition=proposition,
                ))
            visible_binders = inherited_binders.union(
                template.introduced_binders)
            memo_key = proposition, visible_binders
            cached = memo.get(memo_key)
            if cached is not None:
                return cached
            if proposition in active_templates:
                raise TypedBindingError(BindingFailure(
                    self.protocol.failures.proposition_cycle,
                    proposition=proposition,
                ))
            active_templates.add(proposition)
            applied: set[ObjectIdentity] = set()
            bindings: list[BoundRoleBinding] = []
            try:
                for binding in template.definition.canonical_bindings():
                    filler: ObjectIdentity | BoundProposition = binding.filler
                    if binding.filler.object_kind == OBJECT_VARIABLE:
                        require_visible(
                            binding.filler, visible_binders,
                            proposition=proposition)
                        assignment = environment.lookup(binding.filler)
                        if assignment is not None:
                            filler = assignment.value.value
                            if filler.object_kind == OBJECT_VARIABLE:
                                require_visible(
                                    filler, visible_binders,
                                    proposition=proposition)
                            applied.add(binding.filler)
                    elif binding.filler.object_kind == OBJECT_PROPOSITION:
                        nested = graph.get(binding.filler)
                        if nested is not None:
                            filler = visit(
                                binding.filler, visible_binders)
                    bindings.append(BoundRoleBinding(
                        binding.role, filler, binding.ordinal))
            finally:
                active_templates.remove(proposition)
            bound = BoundProposition(
                template.definition.proposition,
                self.protocol.instruction,
                template.definition.predicate,
                template.structure,
                template.definition.source_anchor,
                template.definition.context,
                template.introduced_binders,
                tuple(bindings),
                tuple(applied),
            )
            memo[memo_key] = bound
            return bound

        return visit(root, frozenset(inherited_binders))


__all__ = [
    "BindingEnvironment",
    "BindingFailure",
    "BindingFailureProtocol",
    "BindingFrame",
    "BoundProposition",
    "BoundRoleBinding",
    "ExactTypeCompatibilityResolver",
    "PropositionSubstituter",
    "PropositionTemplateGraph",
    "ScopedPropositionTemplate",
    "SubstitutionProtocol",
    "TypeCompatibilityResolver",
    "TypeCompatibilityResult",
    "TypedBindingAssignment",
    "TypedBindingError",
    "TypedValue",
]
