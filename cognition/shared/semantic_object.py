"""S-00 统一语义对象、来源身份和认识论投影协议。

本模块只定义领域对象，不猜自然语言角色、predicate 或逻辑操作。稳定含义继续由
Concept/StructureConcept 承载；一次来源中的实体、事件、命题、集合表达式、变量、
绑定器、语义上下文和角色绑定均使用独立的一等对象。命题存在只表示候选内容存在，
当前认识论状态必须从 H-00 Evidence 派生。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.hypothesis import (
    HypothesisKey,
    HypothesisLedger,
    HypothesisSnapshot,
)
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    OBJECT_BINDER,
    OBJECT_CONCEPT,
    OBJECT_CONTEXT_SCOPE,
    OBJECT_ENTITY,
    OBJECT_EVENT,
    OBJECT_OCCURRENCE,
    OBJECT_PROPOSITION,
    OBJECT_ROLE,
    OBJECT_ROLE_BINDING,
    OBJECT_SET_EXPR,
    OBJECT_SPAN,
    OBJECT_STRUCTURE_CONCEPT,
    OBJECT_VARIABLE,
    ObjectIdentity,
    OwnerScope,
    SourceRef,
    VersionBundle,
    object_contracts_by_kind,
)
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.crosscut.guards.int_blocker import assert_int


_SEMANTIC_IDENTITY_VERSION = 1
_SOURCE_KEY_SIZE = len(SourceRef(
    1, 1, 0, GLOBAL_OWNER_SCOPE, VersionBundle()).stable_key())

_SOURCE_BOUND_KINDS = frozenset({
    OBJECT_ENTITY,
    OBJECT_EVENT,
    OBJECT_PROPOSITION,
    OBJECT_SET_EXPR,
    OBJECT_VARIABLE,
    OBJECT_BINDER,
    OBJECT_CONTEXT_SCOPE,
    OBJECT_ROLE_BINDING,
})

_ROLE_FILLER_KINDS = frozenset(
    contract.object_kind
    for contract in object_contracts_by_kind().values()
    if contract.authoritative_identity
)

_VARIABLE_TYPE_KINDS = frozenset({
    OBJECT_CONCEPT,
    OBJECT_STRUCTURE_CONCEPT,
    OBJECT_ROLE,
})

SEMANTIC_OBJECT_KINDS = frozenset({
    *_SOURCE_BOUND_KINDS,
    OBJECT_ROLE,
})


def _strict_key(value: tuple[int, ...], *, where: str) -> tuple[int, ...]:
    """校验调用方注入的开放稳定键，拒绝空键、布尔值和整数子类。"""
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{where} 必须是非空整数 tuple")
    assert_int(*value, _where=where)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{where} 必须使用严格整数")
    return value


def _strict_nonnegative(value: int, *, where: str) -> int:
    """校验身份中的 ordinal，禁止布尔值和负数进入稳定键。"""
    assert_int(value, _where=where)
    if type(value) is not int or value < 0:
        raise ValueError(f"{where} 必须为非负严格整数")
    return value


def _packed(key: tuple[int, ...]) -> tuple[int, ...]:
    """给可变长完整身份段增加长度前缀，避免拼接歧义。"""
    return len(key), *key


def _take_packed(
        components: tuple[int, ...], cursor: int, *, label: str
        ) -> tuple[tuple[int, ...], int]:
    """从稳定键读取一个非空长度前缀段，并返回新游标。"""
    if cursor >= len(components):
        raise ValueError(f"语义对象缺少 {label} 长度")
    size = components[cursor]
    cursor += 1
    if size <= 0 or cursor + size > len(components):
        raise ValueError(f"语义对象 {label} 长度非法")
    return components[cursor:cursor + size], cursor + size


def _source_prefix(identity: ObjectIdentity) -> SourceRef:
    """只核验来源化语义对象的版本、SourceRef 前缀和 owner/version。"""
    if not isinstance(identity, ObjectIdentity):
        raise TypeError("语义身份必须是 ObjectIdentity")
    if identity.object_kind not in _SOURCE_BOUND_KINDS:
        raise ValueError("该对象不是来源化语义对象")
    components = identity.components
    if (len(components) <= _SOURCE_KEY_SIZE
            or components[0] != _SEMANTIC_IDENTITY_VERSION):
        raise ValueError("来源化语义对象身份版本或长度非法")
    source = SourceRef.from_stable_key(
        components[1:1 + _SOURCE_KEY_SIZE])
    if source.owner != identity.owner or source.versions != identity.versions:
        raise ValueError("语义对象 owner/version 与 SourceRef 不一致")
    return source


def _validate_role(identity: ObjectIdentity) -> None:
    """核验 Role 的版本和唯一开放声明键布局。"""
    if identity.object_kind != OBJECT_ROLE:
        raise ValueError("语义角色身份必须是 Role")
    components = identity.components
    if not components or components[0] != _SEMANTIC_IDENTITY_VERSION:
        raise ValueError("Role 身份版本非法")
    key, cursor = _take_packed(components, 1, label="role key")
    _strict_key(key, where="Role.role_key")
    if cursor != len(components):
        raise ValueError("Role 身份含尾随数据")


def _validate_source_declaration(identity: ObjectIdentity) -> SourceRef:
    """核验普通来源化声明仅含完整 SourceRef 和一个开放声明键。"""
    source = _source_prefix(identity)
    components = identity.components
    key, cursor = _take_packed(
        components, 1 + _SOURCE_KEY_SIZE, label="declaration key")
    _strict_key(key, where="semantic declaration key")
    if cursor != len(components):
        raise ValueError("来源化语义声明含尾随数据")
    return source


def _source_bound_identity(
        object_kind: int, source: SourceRef, declaration_key: tuple[int, ...],
        *, where: str) -> ObjectIdentity:
    """构造来源化语义声明身份，Evidence 和运行节点号不得进入声明键。"""
    if not isinstance(source, SourceRef):
        raise TypeError(f"{where}.source 必须是 SourceRef")
    key = _strict_key(declaration_key, where=f"{where}.declaration_key")
    return ObjectIdentity(
        object_kind,
        (_SEMANTIC_IDENTITY_VERSION, *source.stable_key(), *_packed(key)),
        source.owner,
        source.versions,
    )


def entity_identity(source: SourceRef,
                    entity_key: tuple[int, ...]) -> ObjectIdentity:
    """构造一次来源中的语义实体身份，不与稳定 Concept 或 token occurrence 合并。"""
    return _source_bound_identity(
        OBJECT_ENTITY, source, entity_key, where="entity_identity")


def event_identity(source: SourceRef,
                   event_key: tuple[int, ...]) -> ObjectIdentity:
    """构造一次来源中的语义事件身份，参与角色另由 RoleBinding 表达。"""
    return _source_bound_identity(
        OBJECT_EVENT, source, event_key, where="event_identity")


def proposition_identity(source: SourceRef,
                         proposition_key: tuple[int, ...]) -> ObjectIdentity:
    """构造命题候选声明身份，不把真值、Evidence 或角色内容塞入本体键。"""
    return _source_bound_identity(
        OBJECT_PROPOSITION, source, proposition_key,
        where="proposition_identity")


def set_expr_identity(source: SourceRef,
                      expression_key: tuple[int, ...]) -> ObjectIdentity:
    """构造来源化集合表达式身份，关系代数和执行语义由后续 S 线图定义。"""
    return _source_bound_identity(
        OBJECT_SET_EXPR, source, expression_key, where="set_expr_identity")


def binder_identity(source: SourceRef,
                    binder_key: tuple[int, ...]) -> ObjectIdentity:
    """构造来源化 Binder 声明身份，不在宿主代码写死量词或 lambda 类别。"""
    return _source_bound_identity(
        OBJECT_BINDER, source, binder_key, where="binder_identity")


def context_scope_identity(source: SourceRef,
                           context_key: tuple[int, ...]) -> ObjectIdentity:
    """构造语义 ContextScope；它是图对象，不替代运行期 ScopeIdentity。"""
    return _source_bound_identity(
        OBJECT_CONTEXT_SCOPE, source, context_key,
        where="context_scope_identity")


def role_identity(
        role_key: tuple[int, ...], *,
        owner: OwnerScope = GLOBAL_OWNER_SCOPE,
        versions: VersionBundle = VersionBundle()) -> ObjectIdentity:
    """构造开放 typed Role 概念身份，具体角色意义必须由图或课程注入。"""
    key = _strict_key(role_key, where="role_identity.role_key")
    return ObjectIdentity(
        OBJECT_ROLE,
        (_SEMANTIC_IDENTITY_VERSION, *_packed(key)),
        owner,
        versions,
    )


def variable_identity(
        binder: ObjectIdentity, variable_key: tuple[int, ...],
        value_type: ObjectIdentity) -> ObjectIdentity:
    """构造 typed Variable，完整保存 Binder、局部声明键和类型概念身份。"""
    if binder.object_kind != OBJECT_BINDER:
        raise ValueError("variable_identity.binder 必须是 Binder 身份")
    if value_type.object_kind not in _VARIABLE_TYPE_KINDS:
        raise ValueError("variable_identity.value_type 必须是概念层类型身份")
    key = _strict_key(
        variable_key, where="variable_identity.variable_key")
    source = semantic_source(binder)
    validate_semantic_identity(value_type)
    binder_key = binder.stable_key()
    type_key = value_type.stable_key()
    return ObjectIdentity(
        OBJECT_VARIABLE,
        (
            _SEMANTIC_IDENTITY_VERSION,
            *source.stable_key(),
            *_packed(binder_key),
            *_packed(key),
            *_packed(type_key),
        ),
        source.owner,
        source.versions,
    )


@dataclass(frozen=True)
class VariableDescriptor:
    """从完整 Variable 身份恢复的来源、Binder、局部键和一等类型。"""

    identity: ObjectIdentity
    source: SourceRef
    binder: ObjectIdentity
    local_key: tuple[int, ...]
    value_type: ObjectIdentity


def role_binding_identity(
        proposition: ObjectIdentity, role: ObjectIdentity,
        filler: ObjectIdentity, *, ordinal: int) -> ObjectIdentity:
    """构造一等 n 元角色绑定身份，完整保存命题、角色、填充值和同角色序号。"""
    if proposition.object_kind != OBJECT_PROPOSITION:
        raise ValueError("role_binding_identity.proposition 必须是 Proposition")
    if role.object_kind != OBJECT_ROLE:
        raise ValueError("role_binding_identity.role 必须是 Role")
    if filler.object_kind not in _ROLE_FILLER_KINDS:
        raise ValueError("role_binding_identity.filler 类型不能承担语义角色")
    _strict_nonnegative(ordinal, where="role_binding_identity.ordinal")
    source = semantic_source(proposition)
    validate_semantic_identity(role)
    _validate_role_filler(filler)
    return ObjectIdentity(
        OBJECT_ROLE_BINDING,
        (
            _SEMANTIC_IDENTITY_VERSION,
            *source.stable_key(),
            *_packed(proposition.stable_key()),
            *_packed(role.stable_key()),
            *_packed(filler.stable_key()),
            ordinal,
        ),
        source.owner,
        source.versions,
    )


def semantic_source(identity: ObjectIdentity) -> SourceRef:
    """完整核验来源化语义对象布局并恢复 SourceRef。"""
    if identity.object_kind in {
            OBJECT_ENTITY, OBJECT_EVENT, OBJECT_PROPOSITION,
            OBJECT_SET_EXPR, OBJECT_BINDER, OBJECT_CONTEXT_SCOPE}:
        return _validate_source_declaration(identity)
    if identity.object_kind == OBJECT_VARIABLE:
        return _validate_variable(identity)
    if identity.object_kind == OBJECT_ROLE_BINDING:
        source, _, _, _, _ = _parse_role_binding(identity)
        return source
    raise ValueError("该对象不是来源化语义对象")


def _validate_variable(identity: ObjectIdentity) -> SourceRef:
    """核验 Variable 的 Binder、局部声明键和类型概念均以完整键保存。"""
    return _describe_variable(identity).source


def _describe_variable(identity: ObjectIdentity) -> VariableDescriptor:
    """解析并重建 Variable 全键，返回后续 binding 可消费的只读说明。"""
    source = _source_prefix(identity)
    components = identity.components
    cursor = 1 + _SOURCE_KEY_SIZE
    binder_key, cursor = _take_packed(
        components, cursor, label="variable binder")
    variable_key, cursor = _take_packed(
        components, cursor, label="variable key")
    type_key, cursor = _take_packed(
        components, cursor, label="variable type")
    if cursor != len(components):
        raise ValueError("Variable 身份含尾随数据")
    binder = ObjectIdentity.from_stable_key(binder_key)
    value_type = ObjectIdentity.from_stable_key(type_key)
    if binder.object_kind != OBJECT_BINDER or semantic_source(binder) != source:
        raise ValueError("Variable Binder 类型或来源不一致")
    if value_type.object_kind not in _VARIABLE_TYPE_KINDS:
        raise ValueError("Variable 类型身份不在概念层")
    validate_semantic_identity(value_type)
    _strict_key(variable_key, where="Variable.variable_key")
    expected = variable_identity(binder, variable_key, value_type)
    if expected != identity:
        raise ValueError("Variable 嵌套身份与外层身份不一致")
    return VariableDescriptor(
        identity, source, binder, variable_key, value_type)


def describe_variable(identity: ObjectIdentity) -> VariableDescriptor:
    """公开恢复 typed Variable 声明，拒绝类型标签正确但嵌套全键损坏的对象。"""
    if not isinstance(identity, ObjectIdentity):
        raise TypeError("describe_variable 需要 ObjectIdentity")
    if identity.object_kind != OBJECT_VARIABLE:
        raise ValueError("describe_variable 需要 Variable 身份")
    return _describe_variable(identity)


def _parse_role_binding(
        identity: ObjectIdentity
        ) -> tuple[SourceRef, ObjectIdentity, ObjectIdentity,
                   ObjectIdentity, int]:
    """完整解析 RoleBinding，返回来源、命题、角色、filler 和 ordinal。"""
    if identity.object_kind != OBJECT_ROLE_BINDING:
        raise ValueError("RoleBinding 解析需要 RoleBinding 身份")
    source = _source_prefix(identity)
    components = identity.components
    cursor = 1 + _SOURCE_KEY_SIZE
    proposition_key, cursor = _take_packed(
        components, cursor, label="binding proposition")
    role_key, cursor = _take_packed(
        components, cursor, label="binding role")
    filler_key, cursor = _take_packed(
        components, cursor, label="binding filler")
    if cursor + 1 != len(components):
        raise ValueError("RoleBinding ordinal 缺失或身份含尾随数据")
    proposition = ObjectIdentity.from_stable_key(proposition_key)
    role = ObjectIdentity.from_stable_key(role_key)
    filler = ObjectIdentity.from_stable_key(filler_key)
    ordinal = _strict_nonnegative(
        components[cursor], where="role_binding_ordinal")
    if proposition.object_kind != OBJECT_PROPOSITION:
        raise ValueError("RoleBinding proposition 类型非法")
    if semantic_source(proposition) != source:
        raise ValueError("RoleBinding 与 Proposition 来源不一致")
    _validate_role(role)
    _validate_role_filler(filler)
    expected = role_binding_identity(
        proposition, role, filler, ordinal=ordinal)
    if expected != identity:
        raise ValueError("RoleBinding 嵌套身份与外层身份不一致")
    return source, proposition, role, filler, ordinal


def validate_semantic_identity(identity: ObjectIdentity) -> ObjectIdentity:
    """在图持久化边界核验 S-00 分型身份，防止只贴 object_kind 标签。"""
    if not isinstance(identity, ObjectIdentity):
        raise TypeError("validate_semantic_identity 需要 ObjectIdentity")
    if identity.object_kind == OBJECT_ROLE:
        _validate_role(identity)
        return identity
    if identity.object_kind in _SOURCE_BOUND_KINDS:
        semantic_source(identity)
        return identity
    if identity.object_kind in {OBJECT_CONCEPT, OBJECT_STRUCTURE_CONCEPT}:
        return identity
    raise ValueError("对象不属于 S-00 语义身份或概念层类型")


def _validate_role_filler(identity: ObjectIdentity) -> ObjectIdentity:
    """允许完整权威对象参与角色，并拒绝 legacy compatibility projection。"""
    if not isinstance(identity, ObjectIdentity):
        raise TypeError("RoleBinding filler 必须是 ObjectIdentity")
    if identity.object_kind not in _ROLE_FILLER_KINDS:
        raise ValueError("RoleBinding filler 不是权威对象身份")
    if (identity.object_kind in SEMANTIC_OBJECT_KINDS
            or identity.object_kind in {
                OBJECT_CONCEPT, OBJECT_STRUCTURE_CONCEPT}):
        validate_semantic_identity(identity)
    return identity


def role_binding_ordinal(identity: ObjectIdentity) -> int:
    """完整解析 RoleBinding 嵌套身份并返回同角色 ordinal，拒绝截断和尾随。"""
    _, _, _, _, ordinal = _parse_role_binding(identity)
    return ordinal


@dataclass(frozen=True, order=True)
class AtomicRoleBinding:
    """原子命题中的一个开放角色赋值；角色名称和作用均不由宿主字段决定。"""

    role: ObjectIdentity
    filler: ObjectIdentity
    ordinal: int = 0

    def __post_init__(self) -> None:
        if self.role.object_kind != OBJECT_ROLE:
            raise ValueError("AtomicRoleBinding.role 必须是 Role")
        if self.filler.object_kind not in _ROLE_FILLER_KINDS:
            raise ValueError("AtomicRoleBinding.filler 类型不能承担语义角色")
        validate_semantic_identity(self.role)
        _validate_role_filler(self.filler)
        _strict_nonnegative(
            self.ordinal, where="AtomicRoleBinding.ordinal")

    def identity_for(self, proposition: ObjectIdentity) -> ObjectIdentity:
        """为给定命题构造一等 RoleBinding 身份。"""
        return role_binding_identity(
            proposition, self.role, self.filler, ordinal=self.ordinal)


@dataclass(frozen=True)
class AtomicPropositionDefinition:
    """一个来源化原子命题的完整图定义，不含真值或 Evidence aggregate。"""

    proposition: ObjectIdentity
    predicate: ObjectIdentity
    source_anchor: ObjectIdentity
    context: ObjectIdentity
    bindings: tuple[AtomicRoleBinding, ...]

    def __post_init__(self) -> None:
        if self.proposition.object_kind != OBJECT_PROPOSITION:
            raise ValueError("proposition 必须是 Proposition 身份")
        if self.predicate.object_kind != OBJECT_CONCEPT:
            raise ValueError("predicate 必须是通用 Concept 身份")
        if self.source_anchor.object_kind not in {
                OBJECT_OCCURRENCE, OBJECT_SPAN}:
            raise ValueError("source_anchor 必须是 Occurrence 或 Span")
        if self.context.object_kind != OBJECT_CONTEXT_SCOPE:
            raise ValueError("context 必须是语义 ContextScope 身份")
        validate_semantic_identity(self.proposition)
        validate_semantic_identity(self.context)
        if not isinstance(self.bindings, tuple):
            raise TypeError("bindings 必须是 AtomicRoleBinding tuple")
        if any(not isinstance(item, AtomicRoleBinding)
               for item in self.bindings):
            raise TypeError("bindings 只能包含 AtomicRoleBinding")

        source = semantic_source(self.proposition)
        anchor_source = SourceRef.from_stable_key(
            self.source_anchor.components[:_SOURCE_KEY_SIZE])
        if anchor_source != source:
            raise ValueError("Proposition 与 source anchor 必须来自同一 SourceRef")
        context_source = semantic_source(self.context)
        if (context_source.owner != source.owner
                or context_source.versions != source.versions):
            raise ValueError("ContextScope 与 Proposition owner/version 不一致")

        seen_slots: set[tuple[ObjectIdentity, int]] = set()
        seen_bindings: set[ObjectIdentity] = set()
        for binding in self.bindings:
            slot = binding.role, binding.ordinal
            if slot in seen_slots:
                raise ValueError("同一 Role 和 ordinal 不得绑定多个 filler")
            seen_slots.add(slot)
            binding_identity = binding.identity_for(self.proposition)
            if binding_identity in seen_bindings:
                raise ValueError("原子命题不得重复同一 RoleBinding")
            seen_bindings.add(binding_identity)
        object.__setattr__(self, "bindings", tuple(sorted(
            self.bindings,
            key=lambda item: item.identity_for(
                self.proposition).stable_key(),
        )))

    @property
    def source(self) -> SourceRef:
        """返回命题声明中保存的完整 SourceRef。"""
        return semantic_source(self.proposition)

    def canonical_bindings(self) -> tuple[AtomicRoleBinding, ...]:
        """按完整 RoleBinding 身份返回确定顺序，不依赖调用方输入顺序。"""
        return self.bindings


def proposition_hypothesis_key(
        proposition: ObjectIdentity, *, hypothesis_kind: tuple[int, ...],
        competition_key: tuple[int, ...], scope: ScopeIdentity
        ) -> HypothesisKey:
    """把 Proposition 完整身份映射到 H-00 候选，不使用 hash 或表层摘要。"""
    if proposition.object_kind != OBJECT_PROPOSITION:
        raise ValueError("proposition_hypothesis_key 需要 Proposition 身份")
    source = semantic_source(proposition)
    if not isinstance(scope, ScopeIdentity):
        raise TypeError("scope 必须是 ScopeIdentity")
    return HypothesisKey(
        _strict_key(hypothesis_kind, where="hypothesis_kind"),
        proposition.stable_key(),
        _strict_key(competition_key, where="competition_key"),
        scope,
        source,
    )


@dataclass(frozen=True)
class PropositionKnowledge:
    """命题定义与 H-00 当前派生状态的只读汇合，不复制 Evidence 真源。"""

    definition: AtomicPropositionDefinition
    snapshot: HypothesisSnapshot

    def __post_init__(self) -> None:
        hypothesis = self.snapshot.hypothesis
        if hypothesis.candidate_key != self.definition.proposition.stable_key():
            raise ValueError("Hypothesis candidate 与 Proposition 身份不一致")
        if hypothesis.observation != self.definition.source:
            raise ValueError("Hypothesis observation 与 Proposition 来源不一致")

    @property
    def scope(self) -> ScopeIdentity:
        """返回 Evidence 聚合使用的运行/来源 scope。"""
        return self.snapshot.hypothesis.scope

    @property
    def source(self) -> SourceRef:
        """返回命题和 Evidence 共同绑定的完整来源。"""
        return self.definition.source


def project_proposition_knowledge(
        definition: AtomicPropositionDefinition,
        hypothesis: HypothesisKey,
        ledger: HypothesisLedger) -> PropositionKnowledge:
    """从 H-00 事件真源派生命题状态；未登记候选必须失败而非默认判真。"""
    if not isinstance(definition, AtomicPropositionDefinition):
        raise TypeError("definition 必须是 AtomicPropositionDefinition")
    if not isinstance(hypothesis, HypothesisKey):
        raise TypeError("hypothesis 必须是 HypothesisKey")
    if not isinstance(ledger, HypothesisLedger):
        raise TypeError("ledger 必须是 HypothesisLedger")
    if hypothesis.candidate_key != definition.proposition.stable_key():
        raise ValueError("Hypothesis candidate 与 Proposition 身份不一致")
    if hypothesis.observation != definition.source:
        raise ValueError("Hypothesis observation 与 Proposition 来源不一致")
    return PropositionKnowledge(definition, ledger.snapshot(hypothesis))


__all__ = [
    "AtomicPropositionDefinition",
    "AtomicRoleBinding",
    "PropositionKnowledge",
    "SEMANTIC_OBJECT_KINDS",
    "VariableDescriptor",
    "binder_identity",
    "context_scope_identity",
    "describe_variable",
    "entity_identity",
    "event_identity",
    "project_proposition_knowledge",
    "proposition_hypothesis_key",
    "proposition_identity",
    "role_binding_identity",
    "role_binding_ordinal",
    "role_identity",
    "semantic_source",
    "set_expr_identity",
    "variable_identity",
    "validate_semantic_identity",
]
