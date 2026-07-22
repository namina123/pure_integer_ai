"""S-04 注入式复合逻辑执行器。

本模块只组合调用方注册的 StructureConcept、Role、MinimalInstruction 和 resolver。
它不从名称、predicate、位置或图缺边猜逻辑；原子状态、模态结果和有限域完备性均
必须由调用方提供。输出是本次运行的四态证据视图和 source-bearing derivation trace，
不物化为 Core 命题，也不宣称世界终极真值。
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

from pure_integer_ai.cognition.shared.hypothesis import (
    EPISTEMIC_CONFLICTED,
    EPISTEMIC_REFUTED,
    EPISTEMIC_SUPPORTED,
    EPISTEMIC_UNKNOWN,
    HypothesisKey,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_BINDER,
    OBJECT_MINIMAL_INSTRUCTION,
    OBJECT_PROPOSITION,
    OBJECT_ROLE,
    OBJECT_SET_EXPR,
    OBJECT_STRUCTURE_CONCEPT,
    OBJECT_VARIABLE,
    ObjectIdentity,
    SourceRef,
)
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.cognition.shared.semantic_object import (
    semantic_source,
    validate_semantic_identity,
)
from pure_integer_ai.cognition.shared.typed_binding import (
    BindingFailure,
    BindingFailureProtocol,
    BindingEnvironment,
    BindingFrame,
    BoundProposition,
    PropositionSubstituter,
    PropositionTemplateGraph,
    SubstitutionProtocol,
    TypeCompatibilityResolver,
    TypedBindingAssignment,
    TypedBindingError,
    TypedValue,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int


def _require_kind(
        identity: ObjectIdentity, expected: int, *, label: str,
        ) -> ObjectIdentity:
    """核验一等身份的 object kind，不读取名字或表层表示。"""
    if not isinstance(identity, ObjectIdentity):
        raise TypeError(f"{label} 必须是 ObjectIdentity")
    if identity.object_kind != expected:
        raise ValueError(f"{label} 对象类型不匹配")
    return identity


def _packed(key: tuple[int, ...]) -> tuple[int, ...]:
    """为运行期稳定键编码可变长完整身份分段。"""
    return len(key), *key


def _strict_int_tuple(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """核验开放整数元数据，禁止 bool、字符串和浮点进入 trace。"""
    if not isinstance(value, tuple):
        raise TypeError(f"{label} 必须是整数 tuple")
    assert_int(*value, _where=label)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{label} 必须使用严格整数")
    return value


STATE_PROVISIONAL = EPISTEMIC_SUPPORTED
STATE_REFUTED = EPISTEMIC_REFUTED
STATE_UNKNOWN = EPISTEMIC_UNKNOWN
STATE_CONFLICTED = EPISTEMIC_CONFLICTED
_STATES = frozenset({
    STATE_PROVISIONAL,
    STATE_REFUTED,
    STATE_UNKNOWN,
    STATE_CONFLICTED,
})


@dataclass(frozen=True)
class LogicEvidenceState:
    """以 support/refute 两个证据位表示 provisional/refuted/unknown/conflicted。"""

    support: bool
    refute: bool

    def __post_init__(self) -> None:
        if type(self.support) is not bool or type(self.refute) is not bool:
            raise TypeError("LogicEvidenceState 位必须是严格 bool")

    @property
    def status(self) -> int:
        """返回既有 H-00 四态编码；supported 在 S-04 中只表示 provisional。"""
        if self.support and self.refute:
            return STATE_CONFLICTED
        if self.support:
            return STATE_PROVISIONAL
        if self.refute:
            return STATE_REFUTED
        return STATE_UNKNOWN

    @classmethod
    def from_status(cls, status: int) -> "LogicEvidenceState":
        """把 H-00 状态转换为四态证据位，拒绝未知状态。"""
        if status == STATE_PROVISIONAL:
            return cls(True, False)
        if status == STATE_REFUTED:
            return cls(False, True)
        if status == STATE_CONFLICTED:
            return cls(True, True)
        if status == STATE_UNKNOWN:
            return cls(False, False)
        raise ValueError("未知逻辑证据状态")

    def negate(self) -> "LogicEvidenceState":
        """交换支持和反驳证据位，不把 unknown 或 conflicted 压成二值。"""
        return LogicEvidenceState(self.refute, self.support)

    def stable_key(self) -> tuple[int, ...]:
        """返回确定性离散状态键。"""
        return int(self.support), int(self.refute)


@dataclass(frozen=True)
class LogicFailureProtocol:
    """为结构缺失、域不完备和 resolver 失败注入互异 MinimalInstruction reason。"""

    operand_missing: ObjectIdentity
    operand_type: ObjectIdentity
    atom_unknown: ObjectIdentity
    domain_missing: ObjectIdentity
    domain_incomplete: ObjectIdentity
    modal_unknown: ObjectIdentity
    binding_failure: ObjectIdentity
    scope_conflict: ObjectIdentity
    evaluation_cycle: ObjectIdentity

    def __post_init__(self) -> None:
        reasons = self.reasons()
        if len(set(reasons)) != len(reasons):
            raise ValueError("logic failure reason 必须互不相同")
        for reason in reasons:
            _require_kind(
                reason, OBJECT_MINIMAL_INSTRUCTION,
                label="logic failure reason")

    def reasons(self) -> tuple[ObjectIdentity, ...]:
        """返回全部机制失败槽位，供审计和统一 trace 分型。"""
        return (
            self.operand_missing,
            self.operand_type,
            self.atom_unknown,
            self.domain_missing,
            self.domain_incomplete,
            self.modal_unknown,
            self.binding_failure,
            self.scope_conflict,
            self.evaluation_cycle,
        )


@dataclass(frozen=True)
class LogicFailure:
    """一次 fail-closed 结构失败，保留完整命题、结构、Role 和可选绑定原因。"""

    reason: ObjectIdentity
    proposition: ObjectIdentity
    structure: ObjectIdentity
    role: ObjectIdentity | None = None
    details: tuple[int, ...] = ()
    binding_failure: BindingFailure | None = None

    def __post_init__(self) -> None:
        _require_kind(
            self.reason, OBJECT_MINIMAL_INSTRUCTION,
            label="logic failure reason")
        _require_kind(
            self.proposition, OBJECT_PROPOSITION,
            label="logic failure proposition")
        _require_kind(
            self.structure, OBJECT_STRUCTURE_CONCEPT,
            label="logic failure structure")
        validate_semantic_identity(self.structure)
        if self.role is not None:
            _require_kind(self.role, OBJECT_ROLE, label="logic failure role")
        _strict_int_tuple(self.details, label="logic failure details")
        if self.binding_failure is not None and not isinstance(
                self.binding_failure, BindingFailure):
            raise TypeError("binding_failure 必须是 BindingFailure")

    def stable_key(self) -> tuple[int, ...]:
        """返回失败原因和完整上下文的稳定键，不依赖异常文字。"""
        reason = self.reason.stable_key()
        proposition = self.proposition.stable_key()
        structure = self.structure.stable_key()
        role = () if self.role is None else self.role.stable_key()
        binding_key: list[int] = []
        if self.binding_failure is not None:
            binding_key.extend(_packed(
                self.binding_failure.reason.stable_key()))
            for identity in (
                    self.binding_failure.variable,
                    self.binding_failure.binder,
                    self.binding_failure.expected_type,
                    self.binding_failure.actual_type,
                    self.binding_failure.proposition):
                key = () if identity is None else identity.stable_key()
                binding_key.extend(_packed(key))
            binding_key.extend(_packed(self.binding_failure.details))
        return (
            *_packed(reason),
            *_packed(proposition),
            *_packed(structure),
            *_packed(role),
            *_strict_int_tuple(self.details, label="logic failure details"),
            *_packed(tuple(binding_key)),
        )


@dataclass(frozen=True)
class LogicAtomEvidence:
    """原子 resolver 注入的证据状态，不把图中 Proposition 存在当作真。"""

    proposition: ObjectIdentity
    state: LogicEvidenceState
    source: SourceRef
    scope: ScopeIdentity
    hypothesis: HypothesisKey | None = None
    support_evidence_ids: tuple[int, ...] = ()
    refute_evidence_ids: tuple[int, ...] = ()
    unknown_evidence_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        _require_kind(
            self.proposition, OBJECT_PROPOSITION,
            label="atom evidence proposition")
        if not isinstance(self.state, LogicEvidenceState):
            raise TypeError("atom evidence state 必须是 LogicEvidenceState")
        if not isinstance(self.source, SourceRef):
            raise TypeError("atom evidence source 必须是 SourceRef")
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("atom evidence scope 必须是 ScopeIdentity")
        if self.hypothesis is not None and not isinstance(
                self.hypothesis, HypothesisKey):
            raise TypeError("atom evidence hypothesis 必须是 HypothesisKey")
        for label, ids in (
                ("support_evidence_ids", self.support_evidence_ids),
                ("refute_evidence_ids", self.refute_evidence_ids),
                ("unknown_evidence_ids", self.unknown_evidence_ids)):
            _strict_int_tuple(ids, label=label)
            if any(item <= 0 for item in ids):
                raise ValueError(f"{label} 必须全部为正整数")
        all_ids = (
            *self.support_evidence_ids,
            *self.refute_evidence_ids,
            *self.unknown_evidence_ids,
        )
        if len(set(all_ids)) != len(all_ids):
            raise ValueError("同一 atom Evidence id 不得跨证据位重复")
        if bool(self.support_evidence_ids) != self.state.support:
            raise ValueError("support Evidence 与 atom support 状态不一致")
        if bool(self.refute_evidence_ids) != self.state.refute:
            raise ValueError("refute Evidence 与 atom refute 状态不一致")
        if self.hypothesis is not None:
            if self.hypothesis.observation != self.source:
                raise ValueError("atom hypothesis 与 evidence source 不一致")
            if self.hypothesis.scope != self.scope:
                raise ValueError("atom hypothesis 与 evidence scope 不一致")
        object.__setattr__(
            self, "support_evidence_ids", tuple(sorted(self.support_evidence_ids)))
        object.__setattr__(
            self, "refute_evidence_ids", tuple(sorted(self.refute_evidence_ids)))
        object.__setattr__(
            self, "unknown_evidence_ids", tuple(sorted(self.unknown_evidence_ids)))

    def stable_key(self) -> tuple[int, ...]:
        """返回完整原子证据身份和 Evidence id 键。"""
        hypothesis = () if self.hypothesis is None else self.hypothesis.stable_key()
        return (
            *_packed(self.proposition.stable_key()),
            *self.state.stable_key(),
            *_packed(self.source.stable_key()),
            *_packed(self.scope.stable_key()),
            *_packed(hypothesis),
            *_packed(self.support_evidence_ids),
            *_packed(self.refute_evidence_ids),
            *_packed(self.unknown_evidence_ids),
        )


class AtomEvidenceResolver(Protocol):
    """按完整 Bound Proposition 身份读取原子 H-00/verifier 证据。"""

    def resolve(
            self, proposition: BoundProposition, *, source: SourceRef,
            scope: ScopeIdentity,
            ) -> LogicAtomEvidence | None:
        """未见原子证据必须返回 None，由执行器生成 unknown。"""
        ...


class MappingAtomEvidenceResolver:
    """测试和受限调用方使用的完整 Proposition identity 到证据映射。"""

    def __init__(self, evidence: tuple[LogicAtomEvidence, ...]) -> None:
        if not isinstance(evidence, tuple):
            raise TypeError("evidence 必须是 LogicAtomEvidence tuple")
        if any(not isinstance(item, LogicAtomEvidence) for item in evidence):
            raise TypeError("evidence 含非法项")
        keys = tuple(item.proposition for item in evidence)
        if len(set(keys)) != len(keys):
            raise ValueError("同一 Proposition 不得映射多个 atom evidence")
        self._evidence = {item.proposition: item for item in evidence}

    def resolve(
            self, proposition: BoundProposition, *, source: SourceRef,
            scope: ScopeIdentity,
            ) -> LogicAtomEvidence | None:
        """只按完整 template identity 命中，并拒绝来源或 scope 漂移。"""
        evidence = self._evidence.get(proposition.template)
        if evidence is None:
            return None
        if evidence.source != source or evidence.scope != scope:
            raise ValueError("atom evidence 与当前执行 source/scope 不一致")
        if evidence.proposition != proposition.template:
            raise ValueError("atom evidence 与 Bound Proposition 不一致")
        return evidence


@dataclass(frozen=True)
class OperatorSlot:
    """由调用方显式指定的逻辑子命题 Role/ordinal 槽。"""

    role: ObjectIdentity
    ordinal: int = 0

    def __post_init__(self) -> None:
        _require_kind(self.role, OBJECT_ROLE, label="operator slot role")
        assert_int(self.ordinal, _where="OperatorSlot.ordinal")
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ValueError("OperatorSlot.ordinal 必须是非负严格整数")

    def stable_key(self) -> tuple[int, ...]:
        """返回 Role/ordinal 槽键，保留调用方声明的 slot tuple 顺序。"""
        return (*self.role.stable_key(), self.ordinal)


class LogicOperatorHandler(Protocol):
    """一个由调用方注入到 StructureConcept 的逻辑执行行为。"""

    def apply(
            self, executor: "LogicExecutor",
            definition: "LogicOperatorDefinition",
            proposition: BoundProposition,
            context: "LogicEvaluationContext",
            ) -> "LogicEvaluation":
        """消费显式 slot/domain/resolver，不从名称猜 operator。"""
        ...


@dataclass(frozen=True)
class LogicOperatorDefinition:
    """StructureConcept 到逻辑 handler 的注入注册，不包含语言 surface。"""

    structure: ObjectIdentity
    instruction: ObjectIdentity
    slots: tuple[OperatorSlot, ...]
    handler: LogicOperatorHandler

    def __post_init__(self) -> None:
        _require_kind(
            self.structure, OBJECT_STRUCTURE_CONCEPT,
            label="operator structure")
        validate_semantic_identity(self.structure)
        _require_kind(
            self.instruction, OBJECT_MINIMAL_INSTRUCTION,
            label="operator instruction")
        if not isinstance(self.slots, tuple) or not self.slots:
            raise ValueError("operator slots 不能为空")
        if any(not isinstance(item, OperatorSlot) for item in self.slots):
            raise TypeError("operator slots 只能包含 OperatorSlot")
        slots = tuple((item.role, item.ordinal) for item in self.slots)
        if len(set(slots)) != len(slots):
            raise ValueError("operator slots 不得重复 Role/ordinal")
        if not hasattr(self.handler, "apply"):
            raise TypeError("handler 必须实现 LogicOperatorHandler")

    def stable_key(self) -> tuple[int, ...]:
        """返回结构、指令和调用方声明槽位的确定性键。"""
        result = [
            *_packed(self.structure.stable_key()),
            *_packed(self.instruction.stable_key()),
            len(self.slots),
        ]
        for slot in self.slots:
            key = slot.stable_key()
            result.extend(_packed(key))
        return tuple(result)


class LogicOperatorRegistry:
    """按完整 StructureConcept identity 查找一次执行所用的 operator 定义。"""

    def __init__(self, definitions: tuple[LogicOperatorDefinition, ...]) -> None:
        if not isinstance(definitions, tuple):
            raise TypeError("operator definitions 必须是 tuple")
        if any(not isinstance(item, LogicOperatorDefinition)
               for item in definitions):
            raise TypeError("definitions 只能包含 LogicOperatorDefinition")
        keys = tuple(item.structure for item in definitions)
        if len(set(keys)) != len(keys):
            raise ValueError("同一 StructureConcept 不得注册多个 operator")
        self._definitions = {item.structure: item for item in definitions}

    def get(self, structure: ObjectIdentity) -> LogicOperatorDefinition | None:
        """按完整 StructureConcept 查找；未注册结构由 atom resolver 处理。"""
        _require_kind(structure, OBJECT_STRUCTURE_CONCEPT, label="operator lookup")
        return self._definitions.get(structure)


@dataclass(frozen=True)
class FiniteQuantifierDomain:
    """运行期 typed 有限域；values 不写入 Core，closed 由显式来源/验证器声明。"""

    domain: ObjectIdentity
    values: tuple[TypedValue, ...]
    closed: bool
    closure_evidence: tuple[ObjectIdentity, ...] = ()

    def __post_init__(self) -> None:
        _require_kind(self.domain, OBJECT_SET_EXPR, label="quantifier domain")
        semantic_source(self.domain)
        if not isinstance(self.values, tuple):
            raise TypeError("quantifier values 必须是 TypedValue tuple")
        if any(not isinstance(item, TypedValue) for item in self.values):
            raise TypeError("quantifier values 只能包含 TypedValue")
        keys = tuple(item.stable_key() for item in self.values)
        if len(set(keys)) != len(keys):
            raise ValueError("quantifier domain 不得重复同一 TypedValue")
        if type(self.closed) is not bool:
            raise TypeError("quantifier domain.closed 必须是严格 bool")
        if not isinstance(self.closure_evidence, tuple):
            raise TypeError("closure_evidence 必须是 ObjectIdentity tuple")
        for evidence in self.closure_evidence:
            if not isinstance(evidence, ObjectIdentity):
                raise TypeError("closure_evidence 只能包含 ObjectIdentity")
        if len(set(self.closure_evidence)) != len(self.closure_evidence):
            raise ValueError("closure_evidence 不得重复")
        if self.closed and not self.closure_evidence:
            raise ValueError("closed 有限域必须带显式 closure evidence")
        object.__setattr__(self, "values", tuple(sorted(
            self.values, key=lambda item: item.stable_key())))
        object.__setattr__(self, "closure_evidence", tuple(sorted(
            self.closure_evidence, key=lambda item: item.stable_key())))

    def stable_key(self) -> tuple[int, ...]:
        """返回域身份、值身份、完备声明和 closure evidence 的运行期键。"""
        result = [*_packed(self.domain.stable_key()), int(self.closed)]
        result.append(len(self.values))
        for value in self.values:
            result.extend(_packed(value.stable_key()))
        result.append(len(self.closure_evidence))
        for evidence in self.closure_evidence:
            result.extend(_packed(evidence.stable_key()))
        return tuple(result)


@dataclass(frozen=True)
class QuantifierDefinition:
    """由调用方注入 binder、typed Variable、body slot 和有限域。"""

    binder: ObjectIdentity
    variable: ObjectIdentity
    body_slot: OperatorSlot
    domain: FiniteQuantifierDomain

    def __post_init__(self) -> None:
        _require_kind(self.binder, OBJECT_BINDER, label="quantifier binder")
        _require_kind(self.variable, OBJECT_VARIABLE, label="quantifier variable")
        _require_kind(
            self.body_slot.role, OBJECT_ROLE,
            label="quantifier body role")
        if not isinstance(self.domain, FiniteQuantifierDomain):
            raise TypeError("quantifier domain 必须是 FiniteQuantifierDomain")
        from pure_integer_ai.cognition.shared.semantic_object import describe_variable
        if describe_variable(self.variable).binder != self.binder:
            raise ValueError("quantifier Variable 与 Binder 不一致")
        if semantic_source(self.binder) != semantic_source(self.domain.domain):
            raise ValueError("quantifier Binder 与 domain 来源不一致")


class QuantifierResolver(Protocol):
    """按当前命题和执行上下文注入 Binder、Variable 与有限域。"""

    def resolve(
            self, operator: LogicOperatorDefinition,
            proposition: BoundProposition,
            context: "LogicEvaluationContext",
            ) -> QuantifierDefinition | None:
        """当前 scope 没有可审计有限域时返回 None，执行器保持 unknown。"""
        ...


@dataclass(frozen=True)
class ModalResolution:
    """注入式 modal resolver 的受限结果，可改变执行 scope 但不能伪造 source。"""

    state: LogicEvidenceState
    source: SourceRef
    scope: ScopeIdentity
    evidence_ids: tuple[int, ...] = ()
    hypotheses: tuple[HypothesisKey, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.state, LogicEvidenceState):
            raise TypeError("modal state 必须是 LogicEvidenceState")
        if not isinstance(self.source, SourceRef):
            raise TypeError("modal source 必须是 SourceRef")
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("modal scope 必须是 ScopeIdentity")
        _strict_int_tuple(self.evidence_ids, label="modal evidence_ids")
        if any(item <= 0 for item in self.evidence_ids):
            raise ValueError("modal evidence_ids 必须全部为正整数")
        if self.state.status != STATE_UNKNOWN and not self.evidence_ids:
            raise ValueError("非 unknown modal 结果必须带 Evidence id")
        if any(not isinstance(item, HypothesisKey) for item in self.hypotheses):
            raise TypeError("modal hypotheses 只能包含 HypothesisKey")
        for hypothesis in self.hypotheses:
            if hypothesis.observation != self.source:
                raise ValueError("modal hypothesis 与 resolution source 不一致")
            if hypothesis.scope != self.scope:
                raise ValueError("modal hypothesis 与 resolution scope 不一致")
        object.__setattr__(
            self, "evidence_ids", tuple(sorted(set(self.evidence_ids))))
        object.__setattr__(
            self, "hypotheses", tuple(sorted(set(self.hypotheses))))


class ModalResolver(Protocol):
    """调用方提供的受限 modal/world verifier。"""

    def resolve(
            self, operator: LogicOperatorDefinition,
            child: "LogicEvaluation",
            context: "LogicEvaluationContext",
            ) -> ModalResolution | None:
        """无 modal 依据返回 None，执行器保持 unknown。"""
        ...


@dataclass(frozen=True)
class LogicBranchResult:
    """有限量化一条 typed value 分支的状态和来源。"""

    ordinal: int
    value: TypedValue
    proposition: ObjectIdentity
    state: LogicEvidenceState
    source: SourceRef
    scope: ScopeIdentity
    evaluation_key: tuple[int, ...]
    assignment: TypedBindingAssignment | None = None

    def __post_init__(self) -> None:
        assert_int(self.ordinal, _where="logic branch ordinal")
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ValueError("logic branch ordinal 必须是非负严格整数")
        if not isinstance(self.value, TypedValue):
            raise TypeError("branch value 必须是 TypedValue")
        _require_kind(self.proposition, OBJECT_PROPOSITION, label="branch proposition")
        if not isinstance(self.state, LogicEvidenceState):
            raise TypeError("branch state 必须是 LogicEvidenceState")
        if not isinstance(self.source, SourceRef):
            raise TypeError("branch source 必须是 SourceRef")
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("branch scope 必须是 ScopeIdentity")
        _strict_int_tuple(self.evaluation_key, label="branch evaluation_key")
        if self.assignment is not None:
            if not isinstance(self.assignment, TypedBindingAssignment):
                raise TypeError("branch assignment 类型错误")
            if self.assignment.value != self.value:
                raise ValueError("branch assignment 与 branch value 不一致")

    def stable_key(self) -> tuple[int, ...]:
        """返回量化分支的完整 typed value、命题和状态键。"""
        return (
            self.ordinal,
            *_packed(self.value.stable_key()),
            *_packed(self.proposition.stable_key()),
            *self.state.stable_key(),
            *_packed(self.source.stable_key()),
            *_packed(self.scope.stable_key()),
            *_packed(self.evaluation_key),
            *_packed(
                () if self.assignment is None
                else self.assignment.stable_key()),
        )


@dataclass(frozen=True)
class LogicDerivationStep:
    """一次复合、量化或 modal 应用的 source-bearing 推导步骤。"""

    operator: ObjectIdentity
    instruction: ObjectIdentity
    proposition: ObjectIdentity
    premises: tuple[ObjectIdentity, ...]
    result: LogicEvidenceState
    source: SourceRef
    scope: ScopeIdentity
    evidence_ids: tuple[int, ...] = ()
    hypotheses: tuple[HypothesisKey, ...] = ()
    branch_key: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        _require_kind(self.operator, OBJECT_STRUCTURE_CONCEPT, label="derivation operator")
        _require_kind(
            self.instruction, OBJECT_MINIMAL_INSTRUCTION,
            label="derivation instruction")
        _require_kind(self.proposition, OBJECT_PROPOSITION, label="derivation proposition")
        if not isinstance(self.premises, tuple):
            raise TypeError("derivation premises 必须是 Proposition tuple")
        if any(not isinstance(item, ObjectIdentity)
               or item.object_kind != OBJECT_PROPOSITION
               for item in self.premises):
            raise ValueError("derivation premises 必须是 Proposition")
        if not isinstance(self.result, LogicEvidenceState):
            raise TypeError("derivation result 必须是 LogicEvidenceState")
        if not isinstance(self.source, SourceRef):
            raise TypeError("derivation source 必须是 SourceRef")
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("derivation scope 必须是 ScopeIdentity")
        _strict_int_tuple(self.evidence_ids, label="derivation evidence_ids")
        if any(item <= 0 for item in self.evidence_ids):
            raise ValueError("derivation evidence_ids 必须全部为正整数")
        if any(not isinstance(item, HypothesisKey) for item in self.hypotheses):
            raise TypeError("derivation hypotheses 只能包含 HypothesisKey")
        _strict_int_tuple(self.branch_key, label="derivation branch_key")
        object.__setattr__(
            self, "evidence_ids", tuple(sorted(set(self.evidence_ids))))
        object.__setattr__(
            self, "hypotheses", tuple(sorted(set(self.hypotheses))))

    def stable_key(self) -> tuple[int, ...]:
        """返回步骤所有身份、前提、状态和作用域的确定性键。"""
        result = [
            *_packed(self.operator.stable_key()),
            *_packed(self.instruction.stable_key()),
            *_packed(self.proposition.stable_key()),
            len(self.premises),
        ]
        for premise in self.premises:
            result.extend(_packed(premise.stable_key()))
        result.extend(self.result.stable_key())
        result.extend(_packed(self.source.stable_key()))
        result.extend(_packed(self.scope.stable_key()))
        result.extend(_packed(self.evidence_ids))
        result.append(len(self.hypotheses))
        for hypothesis in self.hypotheses:
            result.extend(_packed(hypothesis.stable_key()))
        result.extend(_packed(self.branch_key))
        return tuple(result)


@dataclass(frozen=True)
class LogicEvaluation:
    """一次根命题求值的四态结果、Evidence 汇总、分支和完整 trace。"""

    proposition: BoundProposition
    state: LogicEvidenceState
    source: SourceRef
    scope: ScopeIdentity
    derivation: tuple[LogicDerivationStep, ...] = ()
    branches: tuple[LogicBranchResult, ...] = ()
    evidence_ids: tuple[int, ...] = ()
    hypotheses: tuple[HypothesisKey, ...] = ()
    failures: tuple[LogicFailure, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.proposition, BoundProposition):
            raise TypeError("evaluation proposition 必须是 BoundProposition")
        if not isinstance(self.state, LogicEvidenceState):
            raise TypeError("evaluation state 必须是 LogicEvidenceState")
        if not isinstance(self.source, SourceRef):
            raise TypeError("evaluation source 必须是 SourceRef")
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("evaluation scope 必须是 ScopeIdentity")
        if any(not isinstance(item, LogicDerivationStep)
               for item in self.derivation):
            raise TypeError("derivation 只能包含 LogicDerivationStep")
        if any(not isinstance(item, LogicBranchResult) for item in self.branches):
            raise TypeError("branches 只能包含 LogicBranchResult")
        _strict_int_tuple(self.evidence_ids, label="evaluation evidence_ids")
        if any(item <= 0 for item in self.evidence_ids):
            raise ValueError("evaluation evidence_ids 必须全部为正整数")
        if any(not isinstance(item, HypothesisKey) for item in self.hypotheses):
            raise TypeError("evaluation hypotheses 只能包含 HypothesisKey")
        if any(not isinstance(item, LogicFailure) for item in self.failures):
            raise TypeError("failures 只能包含 LogicFailure")
        object.__setattr__(self, "branches", tuple(sorted(
            self.branches, key=lambda item: item.stable_key())))
        object.__setattr__(self, "evidence_ids", tuple(sorted(set(self.evidence_ids))))
        object.__setattr__(self, "hypotheses", tuple(sorted(set(self.hypotheses))))
        object.__setattr__(self, "failures", tuple(sorted(
            self.failures, key=lambda item: item.stable_key())))

    @property
    def status(self) -> int:
        """返回兼容旧 H-00 的四态编码，不把结果解释成 definitive truth。"""
        return self.state.status

    def stable_key(self) -> tuple[int, ...]:
        """返回完整运行期结果键；该键不是 Core object identity。"""
        result = [
            *_packed(self.proposition.stable_key()),
            *self.state.stable_key(),
            *_packed(self.source.stable_key()),
            *_packed(self.scope.stable_key()),
            len(self.derivation),
        ]
        for item in self.derivation:
            result.extend(_packed(item.stable_key()))
        result.append(len(self.branches))
        for item in self.branches:
            result.extend(_packed(item.stable_key()))
        result.extend(_packed(self.evidence_ids))
        result.append(len(self.hypotheses))
        for item in self.hypotheses:
            result.extend(_packed(item.stable_key()))
        result.append(len(self.failures))
        for item in self.failures:
            result.extend(_packed(item.stable_key()))
        return tuple(result)


@dataclass(frozen=True)
class LogicEvaluationContext:
    """一次递归求值共享的 source/scope、graph、environment 和 resolver。"""

    source: SourceRef
    scope: ScopeIdentity
    graph: PropositionTemplateGraph
    environment: BindingEnvironment
    failures: LogicFailureProtocol
    atom_resolver: AtomEvidenceResolver
    operators: LogicOperatorRegistry
    substituter: PropositionSubstituter
    type_resolver: TypeCompatibilityResolver
    quantifier_resolver: QuantifierResolver | None = None
    modal_resolver: ModalResolver | None = None
    inherited_binders: tuple[ObjectIdentity, ...] = ()
    active_keys: frozenset[tuple[int, ...]] = frozenset()

    def __post_init__(self) -> None:
        if not isinstance(self.source, SourceRef):
            raise TypeError("logic context source 必须是 SourceRef")
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("logic context scope 必须是 ScopeIdentity")
        if not isinstance(self.graph, PropositionTemplateGraph):
            raise TypeError("logic context graph 类型错误")
        if not isinstance(self.environment, BindingEnvironment):
            raise TypeError("logic context environment 类型错误")
        if not isinstance(self.failures, LogicFailureProtocol):
            raise TypeError("logic context failures 类型错误")
        if not hasattr(self.atom_resolver, "resolve"):
            raise TypeError("atom_resolver 必须实现 resolve")
        if not isinstance(self.operators, LogicOperatorRegistry):
            raise TypeError("operators 类型错误")
        if not isinstance(self.substituter, PropositionSubstituter):
            raise TypeError("substituter 类型错误")
        if not hasattr(self.type_resolver, "resolve"):
            raise TypeError("type_resolver 必须实现 resolve")
        if (self.quantifier_resolver is not None
                and not hasattr(self.quantifier_resolver, "resolve")):
            raise TypeError("quantifier_resolver 必须实现 resolve")
        if not isinstance(self.inherited_binders, tuple):
            raise TypeError("inherited_binders 必须是 Binder tuple")
        if len(set(self.inherited_binders)) != len(self.inherited_binders):
            raise ValueError("inherited_binders 不得重复")
        for binder in self.inherited_binders:
            _require_kind(binder, OBJECT_BINDER, label="logic inherited binder")
            if not self.environment.has_binder(binder):
                raise ValueError("logic inherited Binder 必须在 environment 中")
        if not isinstance(self.active_keys, frozenset):
            raise TypeError("active_keys 必须是 frozenset")
        for key in self.active_keys:
            _strict_int_tuple(key, label="logic active key")


def _combine_and(states: tuple[LogicEvidenceState, ...]) -> LogicEvidenceState:
    """按证据位执行开放世界 AND，不把 unknown 当 false。"""
    if not states:
        return LogicEvidenceState(False, False)
    return LogicEvidenceState(
        all(item.support for item in states),
        any(item.refute for item in states),
    )


def _combine_or(states: tuple[LogicEvidenceState, ...]) -> LogicEvidenceState:
    """按证据位执行开放世界 OR，不把 unknown 当 true。"""
    if not states:
        return LogicEvidenceState(False, False)
    return LogicEvidenceState(
        any(item.support for item in states),
        all(item.refute for item in states),
    )


def _combine_condition(
        antecedent: LogicEvidenceState,
        consequent: LogicEvidenceState,
        ) -> LogicEvidenceState:
    """按注入 slot 顺序执行四态 material implication 的证据位。"""
    return LogicEvidenceState(
        antecedent.refute or consequent.support,
        antecedent.support and consequent.refute,
    )


class _StandardOperator:
    """标准复合 operator 的共享 slot 提取和 trace 组装工具。"""

    @staticmethod
    def operand_views(
            executor: "LogicExecutor", definition: LogicOperatorDefinition,
            proposition: BoundProposition, context: LogicEvaluationContext,
            ) -> tuple[BoundProposition, ...] | LogicEvaluation:
        """按显式 Role/ordinal 槽读取子命题视图，不提前执行其内容。"""
        values: list[BoundProposition] = []
        for slot in definition.slots:
            matches = tuple(
                item for item in proposition.bindings
                if item.role == slot.role and item.ordinal == slot.ordinal)
            if len(matches) != 1:
                return executor.failure_evaluation(
                    proposition,
                    context,
                    context.failures.operand_missing,
                    role=slot.role,
                    details=(slot.ordinal, len(matches)),
                )
            filler = matches[0].filler
            if not isinstance(filler, BoundProposition):
                return executor.failure_evaluation(
                    proposition,
                    context,
                    context.failures.operand_type,
                    role=slot.role,
                    details=(slot.ordinal, filler.object_kind)
                    if isinstance(filler, ObjectIdentity) else (slot.ordinal,),
                )
            values.append(filler)
        return tuple(values)

    @staticmethod
    def operands(
            executor: "LogicExecutor", definition: LogicOperatorDefinition,
            proposition: BoundProposition, context: LogicEvaluationContext,
            ) -> tuple[LogicEvaluation, ...] | LogicEvaluation:
        """读取并求值全部显式子命题，保留 definition 声明的槽位顺序。"""
        views = _StandardOperator.operand_views(
            executor, definition, proposition, context)
        if isinstance(views, LogicEvaluation):
            return views
        return tuple(executor._evaluate(view, context) for view in views)

    @staticmethod
    def result(
            executor: "LogicExecutor", definition: LogicOperatorDefinition,
            proposition: BoundProposition, context: LogicEvaluationContext,
            state: LogicEvidenceState,
            premises: tuple[LogicEvaluation, ...], *,
            branches: tuple[LogicBranchResult, ...] = (),
            extra_failures: tuple[LogicFailure, ...] = (),
            branch_key: tuple[int, ...] = (),
            source: SourceRef | None = None,
            scope: ScopeIdentity | None = None,
            evidence_ids: tuple[int, ...] = (),
            hypotheses: tuple[HypothesisKey, ...] = (),
            ) -> LogicEvaluation:
        """合并前提来源和 trace，并追加当前 operator derivation step。"""
        source = context.source if source is None else source
        scope = context.scope if scope is None else scope
        all_evidence = set(evidence_ids)
        all_hypotheses = set(hypotheses)
        derivation: list[LogicDerivationStep] = []
        failures: list[LogicFailure] = list(extra_failures)
        premise_ids: list[ObjectIdentity] = []
        for premise in premises:
            all_evidence.update(premise.evidence_ids)
            all_hypotheses.update(premise.hypotheses)
            derivation.extend(premise.derivation)
            failures.extend(premise.failures)
            premise_ids.append(premise.proposition.template)
        derivation.append(LogicDerivationStep(
            definition.structure,
            definition.instruction,
            proposition.template,
            tuple(premise_ids),
            state,
            source,
            scope,
            tuple(sorted(all_evidence)),
            tuple(sorted(all_hypotheses)),
            branch_key,
        ))
        return LogicEvaluation(
            proposition,
            state,
            source,
            scope,
            tuple(derivation),
            branches,
            tuple(sorted(all_evidence)),
            tuple(sorted(all_hypotheses)),
            tuple(failures),
        )


class NegationOperator:
    """对单个显式子命题执行证据位交换。"""

    def apply(self, executor, definition, proposition, context):
        """执行 NOT 形状，不绑定任何 predicate 或语言词。"""
        operands = _StandardOperator.operands(
            executor, definition, proposition, context)
        if isinstance(operands, LogicEvaluation):
            return operands
        if len(operands) != 1:
            return executor.failure_evaluation(
                proposition, context, context.failures.operand_missing,
                details=(len(operands),))
        return _StandardOperator.result(
            executor, definition, proposition, context,
            operands[0].state.negate(), operands)


class ConjunctionOperator:
    """按所有显式子命题的 evidence bit 执行 AND 形状。"""

    def apply(self, executor, definition, proposition, context):
        """执行合取，不把 unknown 或 conflicted 压成二值。"""
        operands = _StandardOperator.operands(
            executor, definition, proposition, context)
        if isinstance(operands, LogicEvaluation):
            return operands
        return _StandardOperator.result(
            executor, definition, proposition, context,
            _combine_and(tuple(item.state for item in operands)), operands)


class DisjunctionOperator:
    """按所有显式子命题的 evidence bit 执行 OR 形状。"""

    def apply(self, executor, definition, proposition, context):
        """执行析取，不把缺证据误作支持。"""
        operands = _StandardOperator.operands(
            executor, definition, proposition, context)
        if isinstance(operands, LogicEvaluation):
            return operands
        return _StandardOperator.result(
            executor, definition, proposition, context,
            _combine_or(tuple(item.state for item in operands)), operands)


class ConditionOperator:
    """按调用方 slot 顺序执行二元 CONDITION/material implication。"""

    def apply(self, executor, definition, proposition, context):
        """执行前件到后件的四态蕴含，slot 顺序由 definition 显式声明。"""
        operands = _StandardOperator.operands(
            executor, definition, proposition, context)
        if isinstance(operands, LogicEvaluation):
            return operands
        if len(operands) != 2:
            return executor.failure_evaluation(
                proposition, context, context.failures.operand_missing,
                details=(len(operands),))
        return _StandardOperator.result(
            executor, definition, proposition, context,
            _combine_condition(operands[0].state, operands[1].state), operands)


class ModalOperator:
    """把显式单子命题交给调用方受限 modal resolver。"""

    def apply(self, executor, definition, proposition, context):
        """无 resolver 或无 modal 依据返回 unknown，不伪造 modal truth。"""
        if context.modal_resolver is None:
            return executor.failure_evaluation(
                proposition, context, context.failures.modal_unknown)
        operands = _StandardOperator.operands(
            executor, definition, proposition, context)
        if isinstance(operands, LogicEvaluation):
            return operands
        if len(operands) != 1:
            return executor.failure_evaluation(
                proposition, context, context.failures.operand_missing,
                details=(len(operands),))
        resolution = context.modal_resolver.resolve(
            definition, operands[0], context)
        if resolution is None:
            return executor.failure_evaluation(
                proposition, context, context.failures.modal_unknown)
        if resolution.source != context.source:
            raise ValueError("modal resolver 不得改变 source")
        return _StandardOperator.result(
            executor, definition, proposition, context,
            resolution.state, operands,
            source=resolution.source,
            scope=resolution.scope,
            evidence_ids=resolution.evidence_ids,
            hypotheses=resolution.hypotheses,
        )


class _QuantifierOperator:
    """共享有限域分支 substitution、branch trace 和四态聚合逻辑。"""

    def apply(self, executor, definition, proposition, context):
        """逐个 typed domain value 建立临时 frame，不把当前值写入 Core。"""
        if context.quantifier_resolver is None:
            return executor.failure_evaluation(
                proposition,
                context,
                context.failures.domain_missing,
            )
        quantifier = context.quantifier_resolver.resolve(
            definition, proposition, context)
        if quantifier is None:
            return executor.failure_evaluation(
                proposition,
                context,
                context.failures.domain_missing,
            )
        if not isinstance(quantifier, QuantifierDefinition):
            raise TypeError("quantifier resolver 必须返回 QuantifierDefinition 或 None")
        if semantic_source(quantifier.binder) != context.source:
            raise ValueError("quantifier definition 与 execution source 不一致")
        if definition.slots != (quantifier.body_slot,):
            return executor.failure_evaluation(
                proposition,
                context,
                context.failures.operand_missing,
                role=quantifier.body_slot.role,
                details=(len(definition.slots),),
            )
        if quantifier.binder not in proposition.introduced_binders:
            return executor.failure_evaluation(
                proposition,
                context,
                context.failures.scope_conflict,
                details=(len(proposition.introduced_binders),),
            )
        body_values = _StandardOperator.operand_views(
            executor, definition, proposition, context)
        if isinstance(body_values, LogicEvaluation):
            return body_values
        body = body_values[0]
        domain = quantifier.domain
        branches: list[LogicBranchResult] = []
        branch_evaluations: list[LogicEvaluation] = []
        branch_failures: list[LogicFailure] = []
        for ordinal, value in enumerate(domain.values):
            assignment: TypedBindingAssignment | None = None
            try:
                assignment = TypedBindingAssignment.create(
                    quantifier.variable,
                    value,
                    resolver=context.type_resolver,
                    failures=executor._binding_failures,
                )
                frame = BindingFrame.create(
                    quantifier.binder,
                    (assignment,),
                    failures=executor._binding_failures,
                )
                branch_environment = context.environment.push(
                    frame, failures=executor._binding_failures)
                inherited_binders = (
                    *context.inherited_binders,
                    quantifier.binder,
                )
                substituted = context.substituter.substitute(
                    body.template,
                    context.graph,
                    branch_environment,
                    inherited_binders=inherited_binders,
                )
                branch_context = LogicEvaluationContext(
                    source=context.source,
                    scope=context.scope,
                    graph=context.graph,
                    environment=branch_environment,
                    failures=context.failures,
                    atom_resolver=context.atom_resolver,
                    operators=context.operators,
                    substituter=context.substituter,
                    type_resolver=context.type_resolver,
                    quantifier_resolver=context.quantifier_resolver,
                    modal_resolver=context.modal_resolver,
                    inherited_binders=inherited_binders,
                    active_keys=context.active_keys,
                )
                evaluated = executor._evaluate(substituted, branch_context)
            except TypedBindingError as error:
                reason = context.failures.binding_failure
                if error.failure.reason == executor._binding_failures.scope_conflict:
                    reason = context.failures.scope_conflict
                evaluated = executor.failure_evaluation(
                    body,
                    context,
                    reason,
                    details=(ordinal,),
                    binding_failure=error.failure,
                )
            branches.append(LogicBranchResult(
                ordinal,
                value,
                evaluated.proposition.template,
                evaluated.state,
                evaluated.source,
                evaluated.scope,
                evaluated.stable_key(),
                assignment,
            ))
            branch_evaluations.append(evaluated)
        states = tuple(item.state for item in branch_evaluations)
        result_state, incomplete = self.aggregate(states, domain.closed)
        if incomplete:
            branch_failures.append(LogicFailure(
                context.failures.domain_incomplete,
                proposition.template,
                definition.structure,
            ))
        return _StandardOperator.result(
            executor, definition, proposition, context,
            result_state,
            tuple(branch_evaluations),
            branches=tuple(branches),
            extra_failures=tuple(branch_failures),
            branch_key=domain.stable_key(),
        )

    def aggregate(
            self, states: tuple[LogicEvidenceState, ...], closed: bool,
            ) -> tuple[LogicEvidenceState, bool]:
        """由 EXISTS/FORALL 子类给出量化聚合，closed=false 时保留 unknown。"""
        raise NotImplementedError


class ExistentialOperator(_QuantifierOperator):
    """在显式有限域上执行存在量化的 witness 聚合。"""

    def aggregate(self, states, closed):
        """任一支持是 witness；全反驳只在域闭合时反驳，否则 unknown。"""
        if any(item.support for item in states):
            return LogicEvidenceState(
                True, closed and all(item.refute for item in states)), False
        if closed and all(item.refute for item in states):
            return LogicEvidenceState(False, True), False
        return LogicEvidenceState(False, False), not closed


class UniversalOperator(_QuantifierOperator):
    """在显式有限域上执行全称量化的反例/完整支持聚合。"""

    def aggregate(self, states, closed):
        """任一明确反驳是反例；全支持只在域闭合时支持，否则 unknown。"""
        if any(item.refute for item in states):
            return LogicEvidenceState(
                closed and all(item.support for item in states), True), False
        if closed and all(item.support for item in states):
            return LogicEvidenceState(True, False), False
        return LogicEvidenceState(False, False), not closed


class LogicExecutor:
    """对 Bound Proposition 执行注入式原子、复合、量化和 modal 求值。"""

    def __init__(
            self, operators: LogicOperatorRegistry,
            atom_resolver: AtomEvidenceResolver,
            failures: LogicFailureProtocol,
            substitution: SubstitutionProtocol,
            type_resolver: TypeCompatibilityResolver,
            binding_failures: BindingFailureProtocol,
            ) -> None:
        if not isinstance(operators, LogicOperatorRegistry):
            raise TypeError("operators 必须是 LogicOperatorRegistry")
        if not hasattr(atom_resolver, "resolve"):
            raise TypeError("atom_resolver 必须实现 resolve")
        if not isinstance(failures, LogicFailureProtocol):
            raise TypeError("failures 必须是 LogicFailureProtocol")
        if not isinstance(substitution, SubstitutionProtocol):
            raise TypeError("substitution 必须是 SubstitutionProtocol")
        if not hasattr(type_resolver, "resolve"):
            raise TypeError("type_resolver 必须实现 resolve")
        if not isinstance(binding_failures, BindingFailureProtocol):
            raise TypeError("binding_failures 必须是 BindingFailureProtocol")
        if substitution.failures != binding_failures:
            raise ValueError("substitution 与 executor 必须共享 BindingFailureProtocol")
        self._operators = operators
        self._atom_resolver = atom_resolver
        self._failures = failures
        self._substituter = PropositionSubstituter(substitution)
        self._type_resolver = type_resolver
        self._binding_failures = binding_failures

    def evaluate(
            self, root: BoundProposition, *, source: SourceRef,
            scope: ScopeIdentity, graph: PropositionTemplateGraph,
            environment: BindingEnvironment,
            quantifier_resolver: QuantifierResolver | None = None,
            modal_resolver: ModalResolver | None = None,
            inherited_binders: tuple[ObjectIdentity, ...] = (),
            ) -> LogicEvaluation:
        """从显式 source/scope/environment 和祖先 Binder 开始纯递归求值。"""
        if not isinstance(root, BoundProposition):
            raise TypeError("root 必须是 BoundProposition")
        if not isinstance(source, SourceRef):
            raise TypeError("source 必须是 SourceRef")
        if not isinstance(scope, ScopeIdentity):
            raise TypeError("scope 必须是 ScopeIdentity")
        context = LogicEvaluationContext(
            source=source,
            scope=scope,
            graph=graph,
            environment=environment,
            failures=self._failures,
            atom_resolver=self._atom_resolver,
            operators=self._operators,
            substituter=self._substituter,
            type_resolver=self._type_resolver,
            quantifier_resolver=quantifier_resolver,
            modal_resolver=modal_resolver,
            inherited_binders=inherited_binders,
        )
        if semantic_source(root.template) != source:
            raise ValueError("root Proposition 与 execution source 不一致")
        template = graph.get(root.template)
        if template is None:
            raise ValueError("root Bound Proposition 不在 template graph 中")
        if root.instruction != self._substituter.protocol.instruction:
            raise ValueError("root Bound Proposition 的 substitution 指令不一致")
        if root.structure != template.structure:
            raise ValueError("root Bound Proposition 与图中 StructureConcept 不一致")
        return self._evaluate(root, context)

    def _evaluate(
            self, proposition: BoundProposition,
            context: LogicEvaluationContext,
            ) -> LogicEvaluation:
        """递归求值单个 bound view；memo 仅保存已完成的不可变结果。"""
        if semantic_source(proposition.template) != context.source:
            raise ValueError("nested Proposition 与 execution source 不一致")
        definition = context.operators.get(proposition.structure)
        if definition is None:
            atom = context.atom_resolver.resolve(
                proposition, source=context.source, scope=context.scope)
            if atom is None:
                return self.failure_evaluation(
                    proposition, context, context.failures.atom_unknown)
            if atom.proposition != proposition.template:
                raise ValueError("atom resolver 返回了其他 Proposition")
            if atom.source != context.source or atom.scope != context.scope:
                raise ValueError("atom evidence 与当前执行 source/scope 不一致")
            return LogicEvaluation(
                proposition,
                atom.state,
                atom.source,
                atom.scope,
                (),
                (),
                tuple(sorted({
                    *atom.support_evidence_ids,
                    *atom.refute_evidence_ids,
                    *atom.unknown_evidence_ids,
                })),
                () if atom.hypothesis is None else (atom.hypothesis,),
                (),
            )
        active_key = proposition.stable_key()
        if active_key in context.active_keys:
            return self.failure_evaluation(
                proposition, context, context.failures.evaluation_cycle)
        active_context = replace(
            context,
            active_keys=context.active_keys.union((active_key,)),
        )
        return definition.handler.apply(
            self, definition, proposition, active_context)

    def failure_evaluation(
            self, proposition: BoundProposition,
            context: LogicEvaluationContext,
            reason: ObjectIdentity, *,
            role: ObjectIdentity | None = None,
            details: tuple[int, ...] = (),
            binding_failure: BindingFailure | None = None,
            ) -> LogicEvaluation:
        """把结构或 resolver 缺口转为 unknown，并留下可审计 failure trace。"""
        failure = LogicFailure(
            reason,
            proposition.template,
            proposition.structure,
            role,
            details,
            binding_failure,
        )
        return LogicEvaluation(
            proposition,
            LogicEvidenceState(False, False),
            context.source,
            context.scope,
            (),
            (),
            (),
            (),
            (failure,),
        )

__all__ = [
    "AtomEvidenceResolver",
    "ConjunctionOperator",
    "ConditionOperator",
    "DisjunctionOperator",
    "ExistentialOperator",
    "FiniteQuantifierDomain",
    "LogicAtomEvidence",
    "LogicDerivationStep",
    "LogicEvaluation",
    "LogicEvaluationContext",
    "LogicEvidenceState",
    "LogicExecutor",
    "LogicFailure",
    "LogicFailureProtocol",
    "LogicOperatorDefinition",
    "LogicOperatorHandler",
    "LogicOperatorRegistry",
    "LogicBranchResult",
    "MappingAtomEvidenceResolver",
    "ModalOperator",
    "ModalResolution",
    "ModalResolver",
    "NegationOperator",
    "OperatorSlot",
    "QuantifierDefinition",
    "QuantifierResolver",
    "STATE_CONFLICTED",
    "STATE_PROVISIONAL",
    "STATE_REFUTED",
    "STATE_UNKNOWN",
    "UniversalOperator",
]
