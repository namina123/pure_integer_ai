"""R-05 对称语义 pair 的 typed 协议、四态聚合和有界查询。

本模块不声明近义、反义、同指或代数逆等具体语义。调用方必须注入 relation、
schema、两个 endpoint Role、SymmetricRule 和可选 IrreflexiveRule；宿主只把两个
方向聚合为同一逻辑 pair，不物理双写，也不提供传递、替换或身份提升。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.hypothesis import (
    EPISTEMIC_SUPPORTED,
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EvidenceRecord,
    HypothesisKey,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONCEPT,
    OBJECT_MINIMAL_INSTRUCTION,
    OBJECT_ROLE,
    ObjectIdentity,
    SourceRef,
    object_contracts_by_kind,
)
from pure_integer_ai.cognition.shared.logic_executor import LogicEvidenceState
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.cognition.shared.semantic_object import semantic_source
from pure_integer_ai.cognition.shared.typed_relation import (
    IrreflexiveRule,
    RelationSchema,
    RelationSchemaError,
    SymmetricRule,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int


class SymmetricRelationError(RuntimeError):
    """对称关系协议、Evidence 或查询状态不完整。"""


class SymmetricRelationBudgetExceeded(SymmetricRelationError):
    """对称关系查询超过调用方预算，拒绝返回部分结果。"""


def _packed(value: tuple[int, ...]) -> tuple[int, ...]:
    """为可变长完整键增加长度边界。"""
    return len(value), *value


def _authoritative(
        identity: ObjectIdentity, *, label: str,
        ) -> ObjectIdentity:
    """要求 endpoint 具有权威对象契约，不能使用派生索引冒充本体。"""
    if not isinstance(identity, ObjectIdentity):
        raise TypeError(f"{label} 必须是 ObjectIdentity")
    contract = object_contracts_by_kind().get(identity.object_kind)
    if contract is None or not contract.authoritative_identity:
        raise ValueError(f"{label} 缺少权威 object contract")
    return identity


@dataclass(frozen=True)
class SymmetricRelationProtocol:
    """一个对称 relation 的 schema、两个 Role 和显式规则身份。"""

    schema: RelationSchema
    left_role: ObjectIdentity
    right_role: ObjectIdentity
    symmetric_rule: SymmetricRule
    irreflexive_rule: IrreflexiveRule | None = None

    def __post_init__(self) -> None:
        """要求 schema 与规则恰好绑定两个互异且单值的 endpoint Role。"""
        if not isinstance(self.schema, RelationSchema):
            raise TypeError("symmetric relation schema 类型错误")
        if self.left_role == self.right_role:
            raise RelationSchemaError("symmetric endpoint Role 必须不同")
        if len(self.schema.slots) != 2:
            raise RelationSchemaError("symmetric schema 必须恰好声明两个 Role")
        if {slot.role for slot in self.schema.slots} != set(self.roles()):
            raise RelationSchemaError("symmetric schema Role 与协议不一致")
        for role in self.roles():
            if (not isinstance(role, ObjectIdentity)
                    or role.object_kind != OBJECT_ROLE):
                raise RelationSchemaError("symmetric endpoint 必须是一等 Role")
            slot = self.schema.slot(role)
            if slot is None or (slot.min_count, slot.max_count) != (1, 1):
                raise RelationSchemaError("symmetric endpoint Role 必须恰好出现一次")
        left_slot = self.schema.slot(self.left_role)
        right_slot = self.schema.slot(self.right_role)
        if (left_slot is None or right_slot is None
                or left_slot.allowed_object_kinds
                != right_slot.allowed_object_kinds):
            raise RelationSchemaError(
                "对称关系两个 Role 的允许对象类型必须完全相同")
        if not isinstance(self.symmetric_rule, SymmetricRule):
            raise TypeError("symmetric relation 缺少 SymmetricRule")
        if (self.symmetric_rule.relation != self.relation
                or {self.symmetric_rule.left_role,
                    self.symmetric_rule.right_role} != set(self.roles())):
            raise RelationSchemaError("SymmetricRule 与 relation/Role 协议不一致")
        if self.irreflexive_rule is not None:
            if not isinstance(self.irreflexive_rule, IrreflexiveRule):
                raise TypeError("irreflexive_rule 类型错误")
            if (self.irreflexive_rule.relation != self.relation
                    or {self.irreflexive_rule.left_role,
                        self.irreflexive_rule.right_role} != set(self.roles())):
                raise RelationSchemaError(
                    "IrreflexiveRule 与 relation/Role 协议不一致")

    @property
    def relation(self) -> ObjectIdentity:
        """返回 schema 绑定的一等关系身份。"""
        return self.schema.relation

    def roles(self) -> tuple[ObjectIdentity, ObjectIdentity]:
        """按直接 Proposition 的 left/right 字段顺序返回 Role。"""
        return self.left_role, self.right_role

    def pair(
            self, left: ObjectIdentity, right: ObjectIdentity,
            ) -> "SymmetricPair":
        """按 schema 校验两个 endpoint 并形成 canonical 逻辑 pair。"""
        for role, filler in (
                (self.left_role, left), (self.right_role, right)):
            _authoritative(filler, label="symmetric endpoint")
            slot = self.schema.slot(role)
            if slot is None or filler.object_kind not in slot.allowed_object_kinds:
                raise RelationSchemaError("symmetric endpoint 类型不满足 schema")
        return self.validate_pair(SymmetricPair(self.relation, left, right))

    def validate_pair(self, pair: "SymmetricPair") -> "SymmetricPair":
        """核验 canonical pair 属于当前 relation 且端点类型可双向成立。"""
        if not isinstance(pair, SymmetricPair):
            raise TypeError("symmetric pair 类型错误")
        if pair.relation != self.relation:
            raise SymmetricRelationError("pair relation 与协议不一致")
        left_slot = self.schema.slot(self.left_role)
        right_slot = self.schema.slot(self.right_role)
        if left_slot is None or right_slot is None:
            raise RelationSchemaError("symmetric schema 缺少 endpoint slot")
        for filler in pair.endpoints():
            if (filler.object_kind not in left_slot.allowed_object_kinds
                    or filler.object_kind not in right_slot.allowed_object_kinds):
                raise RelationSchemaError(
                    "对称 pair 的端点类型不能在两个 Role 间交换")
        return pair

    def stable_key(self) -> tuple[int, ...]:
        """返回 relation、schema、Role 和规则身份的完整状态键。"""
        result = [
            *_packed(self.schema.schema.stable_key()),
            *_packed(self.relation.stable_key()),
            *_packed(self.left_role.stable_key()),
            *_packed(self.right_role.stable_key()),
            *_packed(self.symmetric_rule.rule.stable_key()),
        ]
        if self.irreflexive_rule is None:
            result.append(0)
        else:
            result.extend((
                1,
                *_packed(self.irreflexive_rule.rule.stable_key()),
            ))
        return tuple(result)


@dataclass(frozen=True)
class SymmetricRelationBudget:
    """限制单 channel 的直接事实和单次返回 pair 数量。"""

    max_direct_facts: int
    max_options: int

    def __post_init__(self) -> None:
        """要求两个预算均为严格正整数。"""
        assert_int(
            self.max_direct_facts,
            self.max_options,
            _where="SymmetricRelationBudget",
        )
        if (type(self.max_direct_facts) is not int
                or self.max_direct_facts <= 0
                or type(self.max_options) is not int
                or self.max_options <= 0):
            raise ValueError("symmetric relation budget 必须为严格正整数")

    def stable_key(self) -> tuple[int, ...]:
        """返回直接事实和查询选项预算。"""
        return self.max_direct_facts, self.max_options


@dataclass(frozen=True)
class SymmetricPair:
    """一个 relation 下按完整 identity 排序的无向逻辑 pair。"""

    relation: ObjectIdentity
    first: ObjectIdentity
    second: ObjectIdentity

    def __post_init__(self) -> None:
        """核验 relation/endpoint，并稳定规范化 endpoint 顺序。"""
        if (not isinstance(self.relation, ObjectIdentity)
                or self.relation.object_kind != OBJECT_CONCEPT):
            raise TypeError("symmetric pair relation 必须是一等 Concept")
        _authoritative(self.first, label="symmetric pair first")
        _authoritative(self.second, label="symmetric pair second")
        if self.second.stable_key() < self.first.stable_key():
            first, second = self.second, self.first
            object.__setattr__(self, "first", first)
            object.__setattr__(self, "second", second)

    def endpoints(self) -> tuple[ObjectIdentity, ObjectIdentity]:
        """按 canonical identity 顺序返回两个 endpoint。"""
        return self.first, self.second

    def contains(self, endpoint: ObjectIdentity) -> bool:
        """判断一个完整对象身份是否为 pair endpoint。"""
        _authoritative(endpoint, label="symmetric pair endpoint")
        return endpoint == self.first or endpoint == self.second

    def stable_key(self) -> tuple[int, ...]:
        """返回 relation 和 canonical endpoint 的完整键。"""
        return (
            *_packed(self.relation.stable_key()),
            *_packed(self.first.stable_key()),
            *_packed(self.second.stable_key()),
        )


@dataclass(frozen=True)
class SymmetricPairPattern:
    """按任一 endpoint 发现 pair，或完整指定两个 endpoint 精确查询。"""

    endpoint: ObjectIdentity | None = None
    counterpart: ObjectIdentity | None = None

    def __post_init__(self) -> None:
        """核验可选 endpoint，并拒绝缺少 anchor 的 counterpart。"""
        if self.endpoint is not None:
            _authoritative(self.endpoint, label="symmetric pattern endpoint")
        if self.counterpart is not None:
            _authoritative(
                self.counterpart, label="symmetric pattern counterpart")
            if self.endpoint is None:
                raise ValueError("counterpart 必须配套 endpoint")

    def exact_pair(
            self, protocol: SymmetricRelationProtocol,
            ) -> SymmetricPair | None:
        """完整指定两个 endpoint 时返回协议校验后的 canonical pair。"""
        if not isinstance(protocol, SymmetricRelationProtocol):
            raise TypeError("symmetric pattern protocol 类型错误")
        if self.endpoint is None or self.counterpart is None:
            return None
        return protocol.pair(self.endpoint, self.counterpart)

    def matches(self, pair: SymmetricPair) -> bool:
        """按 anchor/counterpart 无向匹配一个 canonical pair。"""
        if not isinstance(pair, SymmetricPair):
            raise TypeError("symmetric pattern pair 类型错误")
        if self.endpoint is None:
            return True
        if not pair.contains(self.endpoint):
            return False
        if self.counterpart is None:
            return True
        expected = SymmetricPair(pair.relation, self.endpoint, self.counterpart)
        return pair == expected

    def stable_key(self) -> tuple[int, ...]:
        """返回两个可选查询 endpoint 的完整键。"""
        result = []
        for identity in (self.endpoint, self.counterpart):
            key = () if identity is None else identity.stable_key()
            result.extend(_packed(key))
        return tuple(result)


@dataclass(frozen=True)
class SymmetricPairEvidence:
    """一个 R-00 候选的原始方向、canonical pair 和当前直接 Evidence。"""

    pair: SymmetricPair
    left: ObjectIdentity
    right: ObjectIdentity
    proposition: ObjectIdentity
    semantic_context: ObjectIdentity
    scope: ScopeIdentity
    hypothesis: HypothesisKey
    state: LogicEvidenceState
    evidence: tuple[EvidenceRecord, ...]
    forming_sources: tuple[SourceRef, ...]
    active_supported: bool

    def __post_init__(self) -> None:
        """核验原始方向、来源、Hypothesis、Evidence 和 active 投影一致。"""
        if not isinstance(self.pair, SymmetricPair):
            raise TypeError("symmetric evidence pair 类型错误")
        _authoritative(self.left, label="symmetric evidence left")
        _authoritative(self.right, label="symmetric evidence right")
        if SymmetricPair(self.pair.relation, self.left, self.right) != self.pair:
            raise ValueError("symmetric Evidence 原始方向与 canonical pair 不一致")
        if not isinstance(self.proposition, ObjectIdentity):
            raise TypeError("symmetric evidence proposition 类型错误")
        if not isinstance(self.semantic_context, ObjectIdentity):
            raise TypeError("symmetric evidence semantic_context 类型错误")
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("symmetric evidence scope 类型错误")
        proposition_source = semantic_source(self.proposition)
        if (semantic_source(self.semantic_context) != proposition_source
                or self.scope.source != proposition_source):
            raise ValueError("symmetric Evidence context/scope 必须绑定 Proposition 来源")
        if not isinstance(self.hypothesis, HypothesisKey):
            raise TypeError("symmetric evidence hypothesis 类型错误")
        if not isinstance(self.state, LogicEvidenceState):
            raise TypeError("symmetric evidence state 类型错误")
        if not isinstance(self.evidence, tuple) or any(
                not isinstance(item, EvidenceRecord) for item in self.evidence):
            raise TypeError("symmetric evidence records 类型错误")
        if any(item.hypothesis != self.hypothesis for item in self.evidence):
            raise ValueError("symmetric Evidence 属于其他 Hypothesis")
        if not isinstance(self.forming_sources, tuple) or any(
                not isinstance(item, SourceRef) for item in self.forming_sources):
            raise TypeError("symmetric forming_sources 类型错误")
        if len(set(self.forming_sources)) != len(self.forming_sources):
            raise ValueError("symmetric forming_sources 不得重复")
        if type(self.active_supported) is not bool:
            raise TypeError("symmetric active_supported 必须是严格 bool")
        support = any(item.stance == EVIDENCE_SUPPORT for item in self.evidence)
        refute = any(item.stance == EVIDENCE_REFUTE for item in self.evidence)
        if (support, refute) != (self.state.support, self.state.refute):
            raise ValueError("symmetric Evidence 与四态不一致")
        if self.active_supported and self.state.status != EPISTEMIC_SUPPORTED:
            raise ValueError("active_supported 只能标记纯支持候选")

    def stable_key(self) -> tuple[int, ...]:
        """返回 pair、原始方向、命题、Evidence 和来源完整键。"""
        result = [
            *_packed(self.pair.stable_key()),
            *_packed(self.left.stable_key()),
            *_packed(self.right.stable_key()),
            *_packed(self.proposition.stable_key()),
            *_packed(self.semantic_context.stable_key()),
            *_packed(self.scope.stable_key()),
            *_packed(self.hypothesis.stable_key()),
            *self.state.stable_key(),
            int(self.active_supported),
            len(self.evidence),
        ]
        for record in self.evidence:
            result.extend(_packed(record.stable_key()))
        result.append(len(self.forming_sources))
        for source in self.forming_sources:
            result.extend(_packed(source.stable_key()))
        return tuple(result)


@dataclass(frozen=True)
class SymmetricPairKnowledge:
    """一次查询从单个 R-00 owner 读取的不可变 pair Evidence 集。"""

    evidence: tuple[SymmetricPairEvidence, ...]

    def __post_init__(self) -> None:
        """按完整 Evidence 键拒绝重复并稳定排序。"""
        if not isinstance(self.evidence, tuple) or any(
                not isinstance(item, SymmetricPairEvidence)
                for item in self.evidence):
            raise TypeError("symmetric pair knowledge 类型错误")
        keyed = {item.stable_key(): item for item in self.evidence}
        if len(keyed) != len(self.evidence):
            raise ValueError("symmetric pair knowledge 不得重复 Evidence")
        object.__setattr__(self, "evidence", tuple(
            keyed[key] for key in sorted(keyed)))


@dataclass(frozen=True)
class SymmetricRuleRefute:
    """显式反自反规则对 active supported 自 pair 形成的 provisional 反驳。"""

    rule: ObjectIdentity
    premises: tuple[SymmetricPairEvidence, ...]

    def __post_init__(self) -> None:
        """要求规则为最小指令，且前提均是同一 active supported 自 pair。"""
        if (not isinstance(self.rule, ObjectIdentity)
                or self.rule.object_kind != OBJECT_MINIMAL_INSTRUCTION):
            raise TypeError("symmetric rule refute 缺少最小指令身份")
        if not isinstance(self.premises, tuple) or not self.premises:
            raise ValueError("symmetric rule refute 必须保留直接前提")
        if any(not isinstance(item, SymmetricPairEvidence)
               or not item.active_supported for item in self.premises):
            raise ValueError("symmetric rule refute 只能引用 active supported 前提")
        pairs = {item.pair for item in self.premises}
        if len(pairs) != 1:
            raise ValueError("symmetric rule refute 前提必须属于同一 pair")
        pair = next(iter(pairs))
        if pair.first != pair.second:
            raise ValueError("IrreflexiveRule 只能反驳自 pair")

    @property
    def pair(self) -> SymmetricPair:
        """返回被规则反驳的 canonical 自 pair。"""
        return self.premises[0].pair

    def stable_key(self) -> tuple[int, ...]:
        """返回规则和全部直接前提的完整审计键。"""
        result = [*_packed(self.rule.stable_key()), len(self.premises)]
        for premise in self.premises:
            result.extend(_packed(premise.stable_key()))
        return tuple(result)


@dataclass(frozen=True)
class SymmetricPairEvaluation:
    """一个 canonical pair 的多方向、多来源四态聚合结果。"""

    pair: SymmetricPair
    state: LogicEvidenceState
    evidence: tuple[SymmetricPairEvidence, ...]
    rule_refutes: tuple[SymmetricRuleRefute, ...] = ()

    def __post_init__(self) -> None:
        """核验 Evidence 属于当前 pair 且证据位与状态一致。"""
        if not isinstance(self.pair, SymmetricPair):
            raise TypeError("symmetric evaluation pair 类型错误")
        if not isinstance(self.state, LogicEvidenceState):
            raise TypeError("symmetric evaluation state 类型错误")
        if not isinstance(self.evidence, tuple) or any(
                not isinstance(item, SymmetricPairEvidence)
                or item.pair != self.pair for item in self.evidence):
            raise TypeError("symmetric evaluation evidence 类型错误")
        if not isinstance(self.rule_refutes, tuple) or any(
                not isinstance(item, SymmetricRuleRefute)
                or item.pair != self.pair for item in self.rule_refutes):
            raise TypeError("symmetric evaluation rule_refutes 类型错误")
        support = any(item.state.support for item in self.evidence)
        refute = bool(self.rule_refutes) or any(
            item.state.refute for item in self.evidence)
        if (support, refute) != (self.state.support, self.state.refute):
            raise ValueError("symmetric evaluation 与 Evidence 四态不一致")

    def active_premises(self) -> tuple[SymmetricPairEvidence, ...]:
        """返回当前 pair 全部 active supported 直接事实。"""
        return tuple(item for item in self.evidence if item.active_supported)


@dataclass(frozen=True)
class SymmetricPairSelection:
    """一次精确或发现查询的完整四态 pair 集合。"""

    pattern: SymmetricPairPattern
    evaluations: tuple[SymmetricPairEvaluation, ...]

    def __post_init__(self) -> None:
        """要求结果稳定有序、互不重复且全部匹配查询模式。"""
        if not isinstance(self.pattern, SymmetricPairPattern):
            raise TypeError("symmetric selection pattern 类型错误")
        if not isinstance(self.evaluations, tuple) or any(
                not isinstance(item, SymmetricPairEvaluation)
                for item in self.evaluations):
            raise TypeError("symmetric selection evaluations 类型错误")
        pairs = tuple(item.pair for item in self.evaluations)
        if any(not self.pattern.matches(pair) for pair in pairs):
            raise ValueError("symmetric selection 包含模式外 pair")
        if len(set(pairs)) != len(pairs):
            raise ValueError("symmetric selection 不得重复 pair")
        if pairs != tuple(sorted(pairs, key=SymmetricPair.stable_key)):
            raise ValueError("symmetric selection 顺序不稳定")

    def pure_supported(self) -> tuple[SymmetricPairEvaluation, ...]:
        """返回全部纯支持 pair，不按频次或 Evidence 数私选。"""
        return tuple(
            item for item in self.evaluations
            if item.state.status == EPISTEMIC_SUPPORTED
        )


class SymmetricRelationEngine:
    """在单个 R-00 owner 快照上执行对称 pair 四态查询。"""

    def __init__(
            self,
            protocol: SymmetricRelationProtocol,
            budget: SymmetricRelationBudget,
            knowledge: SymmetricPairKnowledge,
            ) -> None:
        """绑定协议、预算和当前直接 Evidence，并预构造 pair 索引。"""
        if not isinstance(protocol, SymmetricRelationProtocol):
            raise TypeError("symmetric relation protocol 类型错误")
        if not isinstance(budget, SymmetricRelationBudget):
            raise TypeError("symmetric relation budget 类型错误")
        if not isinstance(knowledge, SymmetricPairKnowledge):
            raise TypeError("symmetric pair knowledge 类型错误")
        if len(knowledge.evidence) > budget.max_direct_facts:
            raise SymmetricRelationBudgetExceeded(
                "symmetric relation 直接事实预算耗尽")
        self.protocol = protocol
        self.budget = budget
        self.knowledge = knowledge
        grouped: dict[SymmetricPair, list[SymmetricPairEvidence]] = {}
        for item in knowledge.evidence:
            protocol.validate_pair(item.pair)
            grouped.setdefault(item.pair, []).append(item)
        self._by_pair = {
            pair: tuple(sorted(values, key=SymmetricPairEvidence.stable_key))
            for pair, values in grouped.items()
        }

    def evaluate(self, pair: SymmetricPair) -> SymmetricPairEvaluation:
        """聚合一个 canonical pair 的两个方向和全部来源四态。"""
        pair = self.protocol.validate_pair(pair)
        evidence = self._by_pair.get(pair, ())
        rule_refutes = ()
        active = tuple(item for item in evidence if item.active_supported)
        if (self.protocol.irreflexive_rule is not None
                and pair.first == pair.second
                and active):
            rule_refutes = (SymmetricRuleRefute(
                self.protocol.irreflexive_rule.rule,
                active,
            ),)
        return SymmetricPairEvaluation(
            pair,
            LogicEvidenceState(
                any(item.state.support for item in evidence),
                bool(rule_refutes)
                or any(item.state.refute for item in evidence),
            ),
            evidence,
            rule_refutes,
        )

    def select(self, pattern: SymmetricPairPattern) -> SymmetricPairSelection:
        """返回全部匹配 pair；精确缺 pair 时显式返回 unknown。"""
        if not isinstance(pattern, SymmetricPairPattern):
            raise TypeError("symmetric select pattern 类型错误")
        pairs = {
            pair for pair in self._by_pair if pattern.matches(pair)
        }
        exact = pattern.exact_pair(self.protocol)
        if exact is not None:
            pairs.add(exact)
        ordered = tuple(sorted(pairs, key=SymmetricPair.stable_key))
        if len(ordered) > self.budget.max_options:
            raise SymmetricRelationBudgetExceeded(
                "symmetric relation option 预算耗尽")
        return SymmetricPairSelection(
            pattern,
            tuple(self.evaluate(pair) for pair in ordered),
        )


__all__ = [
    "SymmetricPair",
    "SymmetricPairEvaluation",
    "SymmetricPairEvidence",
    "SymmetricPairKnowledge",
    "SymmetricPairPattern",
    "SymmetricPairSelection",
    "SymmetricRelationBudget",
    "SymmetricRelationBudgetExceeded",
    "SymmetricRelationEngine",
    "SymmetricRelationError",
    "SymmetricRelationProtocol",
    "SymmetricRuleRefute",
]
