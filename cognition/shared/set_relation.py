"""R-02 集合关系的四态闭包、来源化证明和有限域执行。

本模块不认识关系名称、旧边类型或自然语言记号。调用方必须注入完整 relation、
Role、schema 和 MinimalInstruction；运行期只消费 R-00 提供的当前 Evidence 视图，
派生结果不写回 Core，也不把 provisional 结果宣称为开放世界终极真值。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

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
    OBJECT_SET_EXPR,
    ObjectIdentity,
    SourceRef,
    object_contracts_by_kind,
)
from pure_integer_ai.cognition.shared.logic_executor import (
    FiniteQuantifierDomain,
    LogicEvidenceState,
)
from pure_integer_ai.cognition.shared.typed_binding import TypedValue
from pure_integer_ai.cognition.shared.typed_relation import (
    RelationSchema,
    RelationSchemaError,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int


class SetRelationError(RuntimeError):
    """集合关系协议、证明或有限域不完整。"""


class SetRelationBudgetExceeded(SetRelationError):
    """集合关系查询超过调用方注入预算，拒绝返回部分闭包。"""


def _strict_key(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """校验开放稳定键只含非空严格整数。"""
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{label} 必须是非空整数 tuple")
    assert_int(*value, _where=label)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{label} 必须使用严格整数")
    return value


def _packed(value: tuple[int, ...]) -> tuple[int, ...]:
    """为可变长完整键增加长度边界。"""
    return len(value), *value


def _require_kind(
        identity: ObjectIdentity, expected: int, *, label: str,
        ) -> ObjectIdentity:
    """核验一等对象类型，不读取对象名称或局部编号。"""
    if not isinstance(identity, ObjectIdentity):
        raise TypeError(f"{label} 必须是 ObjectIdentity")
    if identity.object_kind != expected:
        raise ValueError(f"{label} 对象类型不匹配")
    return identity


def _authoritative(identity: ObjectIdentity, *, label: str) -> ObjectIdentity:
    """要求对象类型具有权威 identity contract，可承担 MEMBER 左端点。"""
    if not isinstance(identity, ObjectIdentity):
        raise TypeError(f"{label} 必须是 ObjectIdentity")
    contract = object_contracts_by_kind().get(identity.object_kind)
    if contract is None or not contract.authoritative_identity:
        raise ValueError(f"{label} 缺少权威 object contract")
    return identity


def _slot(schema: RelationSchema, role: ObjectIdentity):
    """读取 schema 中唯一 Role slot，缺失时立即拒绝协议。"""
    _require_kind(role, OBJECT_ROLE, label="set relation role")
    slot = schema.slot(role)
    if slot is None:
        raise RelationSchemaError("set relation Role 不在 schema 中")
    return slot


@dataclass(frozen=True)
class SetBinaryRelationProtocol:
    """一个二元集合关系的 relation、schema 和有方向 Role。"""

    schema: RelationSchema
    left_role: ObjectIdentity
    right_role: ObjectIdentity

    def __post_init__(self) -> None:
        """要求 schema 恰有两个单值 Role，避免匿名额外参数改变语义。"""
        if not isinstance(self.schema, RelationSchema):
            raise TypeError("binary set relation schema 类型错误")
        if self.left_role == self.right_role:
            raise ValueError("binary set relation 左右 Role 不得相同")
        left = _slot(self.schema, self.left_role)
        right = _slot(self.schema, self.right_role)
        if len(self.schema.slots) != 2:
            raise RelationSchemaError("binary set relation schema 必须恰有两个 Role")
        if any((item.min_count, item.max_count) != (1, 1)
               for item in (left, right)):
            raise RelationSchemaError("binary set relation Role 必须恰好出现一次")

    @property
    def relation(self) -> ObjectIdentity:
        """返回 schema 绑定的一等 relation Concept。"""
        return self.schema.relation

    def stable_key(self) -> tuple[int, ...]:
        """返回 relation、schema 和左右 Role 的完整协议键。"""
        return (
            *_packed(self.schema.schema.stable_key()),
            *_packed(self.relation.stable_key()),
            *_packed(self.left_role.stable_key()),
            *_packed(self.right_role.stable_key()),
        )


@dataclass(frozen=True)
class SetUnaryRelationProtocol:
    """一个单参数集合声明的 relation、schema 和 SetExpr Role。"""

    schema: RelationSchema
    value_role: ObjectIdentity

    def __post_init__(self) -> None:
        """要求声明 schema 恰有一个单值 SetExpr Role。"""
        if not isinstance(self.schema, RelationSchema):
            raise TypeError("unary set relation schema 类型错误")
        value = _slot(self.schema, self.value_role)
        if len(self.schema.slots) != 1:
            raise RelationSchemaError("unary set relation schema 必须恰有一个 Role")
        if (value.min_count, value.max_count) != (1, 1):
            raise RelationSchemaError("unary set relation Role 必须恰好出现一次")
        if value.allowed_object_kinds != frozenset({OBJECT_SET_EXPR}):
            raise RelationSchemaError("unary set relation 只能接收 SetExpr")

    @property
    def relation(self) -> ObjectIdentity:
        """返回 schema 绑定的一等 relation Concept。"""
        return self.schema.relation

    def stable_key(self) -> tuple[int, ...]:
        """返回 relation、schema 和参数 Role 的完整协议键。"""
        return (
            *_packed(self.schema.schema.stable_key()),
            *_packed(self.relation.stable_key()),
            *_packed(self.value_role.stable_key()),
        )


@dataclass(frozen=True)
class SetRelationRules:
    """R-02 所有直接投影、代数和量化步骤的一等最小指令。"""

    direct_evidence: ObjectIdentity
    subset_reflexive: ObjectIdentity
    subset_transitive: ObjectIdentity
    proper_to_subset: ObjectIdentity
    proper_irreflexive: ObjectIdentity
    proper_from_inequality: ObjectIdentity
    proper_transitive: ObjectIdentity
    proper_then_subset: ObjectIdentity
    subset_then_proper: ObjectIdentity
    equal_identity: ObjectIdentity
    equal_refutes_proper: ObjectIdentity
    member_lift: ObjectIdentity
    exists_aggregate: ObjectIdentity
    forall_aggregate: ObjectIdentity

    def __post_init__(self) -> None:
        """要求每条执行规则使用互异 MinimalInstruction，防止 proof 混型。"""
        identities = self.identities()
        if len(set(identities)) != len(identities):
            raise ValueError("set relation MinimalInstruction 必须互不相同")
        for identity in identities:
            _require_kind(
                identity,
                OBJECT_MINIMAL_INSTRUCTION,
                label="set relation rule",
            )

    def identities(self) -> tuple[ObjectIdentity, ...]:
        """按协议槽位返回全部执行规则身份。"""
        return tuple(self.__dict__.values())

    def stable_key(self) -> tuple[int, ...]:
        """返回全部规则身份的无哈希完整键。"""
        result: list[int] = []
        for identity in self.identities():
            result.extend(_packed(identity.stable_key()))
        return tuple(result)


@dataclass(frozen=True)
class SetRelationProtocol:
    """SUBSET_EQ、PROPER_SUBSET、MEMBER、EQUAL 和闭域声明协议。"""

    subset_eq: SetBinaryRelationProtocol
    proper_subset: SetBinaryRelationProtocol
    member: SetBinaryRelationProtocol
    equal: SetBinaryRelationProtocol
    closed_domain: SetUnaryRelationProtocol
    rules: SetRelationRules

    def __post_init__(self) -> None:
        """核验五类关系互异且各自 schema 的端点 object kind 正确。"""
        binaries = (
            self.subset_eq,
            self.proper_subset,
            self.member,
            self.equal,
        )
        if any(not isinstance(item, SetBinaryRelationProtocol)
               for item in binaries):
            raise TypeError("set relation binary protocol 类型错误")
        if not isinstance(self.closed_domain, SetUnaryRelationProtocol):
            raise TypeError("closed_domain protocol 类型错误")
        if not isinstance(self.rules, SetRelationRules):
            raise TypeError("set relation rules 类型错误")
        all_relations = tuple(item.relation for item in binaries) + (
            self.closed_domain.relation,
        )
        if len(set(all_relations)) != len(all_relations):
            raise ValueError("set relation 五类 relation 必须互不相同")
        all_schemas = tuple(item.schema.schema for item in binaries) + (
            self.closed_domain.schema.schema,
        )
        if len(set(all_schemas)) != len(all_schemas):
            raise ValueError("set relation 五类 schema 必须互不相同")
        for binary in (self.subset_eq, self.proper_subset, self.equal):
            self._require_set_pair(binary)
        member_left = _slot(self.member.schema, self.member.left_role)
        member_right = _slot(self.member.schema, self.member.right_role)
        if member_right.allowed_object_kinds != frozenset({OBJECT_SET_EXPR}):
            raise RelationSchemaError("MEMBER 右端点只能是 SetExpr")
        contracts = object_contracts_by_kind()
        if any(
                kind not in contracts
                or not contracts[kind].authoritative_identity
                for kind in member_left.allowed_object_kinds):
            raise RelationSchemaError("MEMBER 左端点必须具有权威 object contract")

    @staticmethod
    def _require_set_pair(protocol: SetBinaryRelationProtocol) -> None:
        """要求集合二元关系的两个 Role 都只接受 SetExpr。"""
        for role in (protocol.left_role, protocol.right_role):
            if _slot(protocol.schema, role).allowed_object_kinds != frozenset({
                    OBJECT_SET_EXPR}):
                raise RelationSchemaError("集合二元关系两端都必须是 SetExpr")

    def relation_protocol(
            self, relation: ObjectIdentity,
            ) -> SetBinaryRelationProtocol | SetUnaryRelationProtocol | None:
        """按完整 relation identity 返回对应协议，未知关系不做形状兼容。"""
        for protocol in (
                self.subset_eq,
                self.proper_subset,
                self.member,
                self.equal,
                self.closed_domain):
            if protocol.relation == relation:
                return protocol
        return None

    def validate_statement(
            self, statement: "SetRelationStatement",
            ) -> "SetRelationStatement":
        """按 relation 的注入 schema 核验运行期 statement 端点类型。"""
        if not isinstance(statement, SetRelationStatement):
            raise TypeError("set relation statement 类型错误")
        protocol = self.relation_protocol(statement.relation)
        if protocol is None:
            raise SetRelationError("statement relation 未在 R-02 协议注册")
        if isinstance(protocol, SetUnaryRelationProtocol):
            if statement.right is not None:
                raise SetRelationError("单参数集合声明不得携带右端点")
            if statement.left.object_kind != OBJECT_SET_EXPR:
                raise SetRelationError("闭域声明端点必须是 SetExpr")
            return statement
        if statement.right is None:
            raise SetRelationError("二元集合关系缺少右端点")
        left_slot = _slot(protocol.schema, protocol.left_role)
        right_slot = _slot(protocol.schema, protocol.right_role)
        if statement.left.object_kind not in left_slot.allowed_object_kinds:
            raise SetRelationError("statement 左端点 object kind 不满足 schema")
        if statement.right.object_kind not in right_slot.allowed_object_kinds:
            raise SetRelationError("statement 右端点 object kind 不满足 schema")
        return statement

    def stable_key(self) -> tuple[int, ...]:
        """返回五类 schema/Role 和全部执行规则的完整协议键。"""
        result: list[int] = []
        for protocol in (
                self.subset_eq,
                self.proper_subset,
                self.member,
                self.equal,
                self.closed_domain):
            result.extend(_packed(protocol.stable_key()))
        result.extend(_packed(self.rules.stable_key()))
        return tuple(result)


@dataclass(frozen=True)
class SetRelationBudget:
    """一次集合闭包允许读取和派生的离散资源上限。"""

    max_facts: int
    max_states: int
    max_proofs: int
    max_premises: int
    max_domain_members: int

    def __post_init__(self) -> None:
        """要求所有预算均为严格正整数，不在模块内提供隐式默认值。"""
        values = tuple(self.__dict__.values())
        assert_int(*values, _where="SetRelationBudget")
        if any(type(item) is not int or item <= 0 for item in values):
            raise ValueError("set relation budget 必须全部为严格正整数")

    def stable_key(self) -> tuple[int, ...]:
        """返回预算各维度的纯整数键。"""
        return tuple(self.__dict__.values())


@dataclass(frozen=True)
class SetRelationStatement:
    """一个不要求已物化 Proposition 的集合关系内容。"""

    relation: ObjectIdentity
    left: ObjectIdentity
    right: ObjectIdentity | None = None

    def __post_init__(self) -> None:
        """核验 relation 与端点均为权威一等对象。"""
        _require_kind(self.relation, OBJECT_CONCEPT, label="set relation")
        _authoritative(self.left, label="set relation left")
        if self.right is not None:
            _authoritative(self.right, label="set relation right")

    def stable_key(self) -> tuple[int, ...]:
        """返回 relation 和完整有向端点内容键。"""
        right = () if self.right is None else self.right.stable_key()
        return (
            *_packed(self.relation.stable_key()),
            *_packed(self.left.stable_key()),
            *_packed(right),
        )


@dataclass(frozen=True)
class SetRelationEvidence:
    """从一个 R-00 forming 候选投影出的当前四态直接证据。"""

    statement: SetRelationStatement
    proposition: ObjectIdentity
    hypothesis: HypothesisKey
    state: LogicEvidenceState
    evidence: tuple[EvidenceRecord, ...]
    forming_sources: tuple[SourceRef, ...]
    active_supported: bool

    def __post_init__(self) -> None:
        """核验直接 Evidence、Hypothesis、形成来源和 active 状态彼此一致。"""
        if not isinstance(self.statement, SetRelationStatement):
            raise TypeError("set relation evidence statement 类型错误")
        if not isinstance(self.proposition, ObjectIdentity):
            raise TypeError("set relation evidence proposition 类型错误")
        if not isinstance(self.hypothesis, HypothesisKey):
            raise TypeError("set relation evidence hypothesis 类型错误")
        if not isinstance(self.state, LogicEvidenceState):
            raise TypeError("set relation evidence state 类型错误")
        if not isinstance(self.evidence, tuple):
            raise TypeError("set relation evidence records 必须是 tuple")
        if any(not isinstance(item, EvidenceRecord) for item in self.evidence):
            raise TypeError("set relation evidence record 类型错误")
        if any(item.hypothesis != self.hypothesis for item in self.evidence):
            raise ValueError("set relation Evidence 属于其他 Hypothesis")
        if not isinstance(self.forming_sources, tuple):
            raise TypeError("forming_sources 必须是 tuple")
        if any(not isinstance(item, SourceRef) for item in self.forming_sources):
            raise TypeError("forming_sources 元素类型错误")
        if len(set(self.forming_sources)) != len(self.forming_sources):
            raise ValueError("forming_sources 不得重复")
        if type(self.active_supported) is not bool:
            raise TypeError("active_supported 必须是严格 bool")
        support = any(item.stance == EVIDENCE_SUPPORT for item in self.evidence)
        refute = any(item.stance == EVIDENCE_REFUTE for item in self.evidence)
        if (support, refute) != (self.state.support, self.state.refute):
            raise ValueError("直接 Evidence 与四态证据位不一致")
        if self.active_supported and self.state.status != EPISTEMIC_SUPPORTED:
            raise ValueError("active_supported 只能标记纯 supported 候选")

    def stable_key(self) -> tuple[int, ...]:
        """返回关系内容、命题、Hypothesis 和当前 Evidence 的完整键。"""
        result = [
            *_packed(self.statement.stable_key()),
            *_packed(self.proposition.stable_key()),
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
class SetRelationProof:
    """一个支持或反驳结论的规则节点，可递归回到全部直接 Evidence。"""

    rule: ObjectIdentity
    conclusion: SetRelationStatement
    stance: int
    direct: tuple[SetRelationEvidence, ...] = ()
    antecedents: tuple["SetRelationProof", ...] = ()

    def __post_init__(self) -> None:
        """核验规则、立场、直接来源和前提 proof 的类型与方向。"""
        _require_kind(self.rule, OBJECT_MINIMAL_INSTRUCTION, label="proof rule")
        if not isinstance(self.conclusion, SetRelationStatement):
            raise TypeError("proof conclusion 类型错误")
        if self.stance not in (EVIDENCE_SUPPORT, EVIDENCE_REFUTE):
            raise ValueError("proof stance 只能是 support 或 refute")
        if not isinstance(self.direct, tuple):
            raise TypeError("proof direct 必须是 tuple")
        if any(not isinstance(item, SetRelationEvidence) for item in self.direct):
            raise TypeError("proof direct 元素类型错误")
        if not isinstance(self.antecedents, tuple):
            raise TypeError("proof antecedents 必须是 tuple")
        if any(not isinstance(item, SetRelationProof)
               for item in self.antecedents):
            raise TypeError("proof antecedent 类型错误")

    def direct_evidence(self) -> tuple[SetRelationEvidence, ...]:
        """递归收集并去重全部叶子 Evidence，供 Use 和来源审计。"""
        collected = {item.stable_key(): item for item in self.direct}
        for antecedent in self.antecedents:
            for item in antecedent.direct_evidence():
                collected[item.stable_key()] = item
        return tuple(collected[key] for key in sorted(collected))

    def depth(self) -> int:
        """返回 proof 树深度，公理和直接 Evidence 深度均为一。"""
        if not self.antecedents:
            return 1
        return 1 + max(item.depth() for item in self.antecedents)

    def stable_key(self) -> tuple[int, ...]:
        """展开规则、结论、直接 Evidence 和递归前提的完整 proof 键。"""
        result = [
            *_packed(self.rule.stable_key()),
            *_packed(self.conclusion.stable_key()),
            self.stance,
            len(self.direct),
        ]
        for item in self.direct:
            result.extend(_packed(item.stable_key()))
        result.append(len(self.antecedents))
        for item in self.antecedents:
            result.extend(_packed(item.stable_key()))
        return tuple(result)


@dataclass(frozen=True)
class SetRelationEvaluation:
    """一个集合关系 statement 的四态 provisional 结果和全部已选 proof。"""

    statement: SetRelationStatement
    state: LogicEvidenceState
    proofs: tuple[SetRelationProof, ...]

    def __post_init__(self) -> None:
        """要求 proof 均指向当前 statement，且证据位与 proof 立场一致。"""
        if not isinstance(self.statement, SetRelationStatement):
            raise TypeError("set relation evaluation statement 类型错误")
        if not isinstance(self.state, LogicEvidenceState):
            raise TypeError("set relation evaluation state 类型错误")
        if not isinstance(self.proofs, tuple):
            raise TypeError("set relation evaluation proofs 必须是 tuple")
        if any(item.conclusion != self.statement for item in self.proofs):
            raise ValueError("set relation proof 指向其他结论")
        support = any(item.stance == EVIDENCE_SUPPORT for item in self.proofs)
        refute = any(item.stance == EVIDENCE_REFUTE for item in self.proofs)
        if (support, refute) != (self.state.support, self.state.refute):
            raise ValueError("set relation proofs 与四态结果不一致")

    def active_premises(self) -> tuple[SetRelationEvidence, ...]:
        """返回实际参与 proof 且当前可由 R-00 consumer 采用的直接事实。"""
        premises: dict[tuple[int, ...], SetRelationEvidence] = {}
        for proof in self.proofs:
            for evidence in proof.direct_evidence():
                if evidence.active_supported:
                    premises[evidence.proposition.stable_key()] = evidence
        return tuple(premises[key] for key in sorted(premises))

    def stable_key(self) -> tuple[int, ...]:
        """返回 statement、四态位和全部 proof 的确定性键。"""
        result = [
            *_packed(self.statement.stable_key()),
            *self.state.stable_key(),
            len(self.proofs),
        ]
        for proof in self.proofs:
            result.extend(_packed(proof.stable_key()))
        return tuple(result)


@dataclass(frozen=True)
class SetRelationKnowledge:
    """一次查询批次从 R-00 读取的不可变直接 Evidence 集。"""

    evidence: tuple[SetRelationEvidence, ...]

    def __post_init__(self) -> None:
        """按完整 Evidence 键去重并稳定排序。"""
        if not isinstance(self.evidence, tuple):
            raise TypeError("set relation knowledge evidence 必须是 tuple")
        if any(not isinstance(item, SetRelationEvidence)
               for item in self.evidence):
            raise TypeError("set relation knowledge 元素类型错误")
        keyed = {item.stable_key(): item for item in self.evidence}
        if len(keyed) != len(self.evidence):
            raise ValueError("set relation knowledge 不得重复 Evidence")
        object.__setattr__(
            self,
            "evidence",
            tuple(keyed[key] for key in sorted(keyed)),
        )


@runtime_checkable
class MemberTypeResolver(Protocol):
    """把 MEMBER 左端点映射到运行期量词绑定使用的一等类型。"""

    def resolve(self, member: ObjectIdentity) -> ObjectIdentity | None:
        """缺少可审计类型时返回 None，有限域构造必须 fail closed。"""
        ...

    def clone_for_evaluation(self) -> "MemberTypeResolver":
        """返回不共享可变状态的评测 resolver。"""
        ...

    def state_key(self) -> tuple[int, ...]:
        """返回类型映射协议和内容的完整整数状态键。"""
        ...


class MappingMemberTypeResolver:
    """按完整成员身份读取调用方提供的一等类型映射。"""

    def __init__(
            self,
            entries: tuple[tuple[ObjectIdentity, ObjectIdentity], ...],
            ) -> None:
        """核验成员唯一且类型为 Concept、StructureConcept 或 Role。"""
        if not isinstance(entries, tuple):
            raise TypeError("member type entries 必须是 tuple")
        mapping: dict[ObjectIdentity, ObjectIdentity] = {}
        for entry in entries:
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise TypeError("member type entry 必须是 member/type 对")
            member, value_type = entry
            _authoritative(member, label="member type member")
            TypedValue(member, value_type)
            prior = mapping.get(member)
            if prior is not None and prior != value_type:
                raise ValueError("同一成员不得映射多个类型")
            mapping[member] = value_type
        self._mapping = mapping

    def resolve(self, member: ObjectIdentity) -> ObjectIdentity | None:
        """按完整成员身份返回类型，未声明成员保持 None。"""
        _authoritative(member, label="member type lookup")
        return self._mapping.get(member)

    def clone_for_evaluation(self) -> "MappingMemberTypeResolver":
        """复制不可变成员类型映射，避免评测依赖宿主对象引用。"""
        return MappingMemberTypeResolver(tuple(self._mapping.items()))

    def state_key(self) -> tuple[int, ...]:
        """返回按完整成员身份排序的类型映射键。"""
        result: list[int] = []
        for member, value_type in sorted(
                self._mapping.items(),
                key=lambda item: item[0].stable_key()):
            result.extend(_packed(member.stable_key()))
            result.extend(_packed(value_type.stable_key()))
        return tuple(result)


@dataclass(frozen=True)
class SetRelationDomainResult:
    """由 CLOSED_DOMAIN 和 MEMBER 四态结果构造的有限域及来源视图。"""

    domain: FiniteQuantifierDomain
    closure: SetRelationEvaluation
    members: tuple[tuple[TypedValue, SetRelationEvaluation], ...]

    def __post_init__(self) -> None:
        """要求成员值与 MEMBER evaluation 左端点逐项一致且不重复。"""
        if not isinstance(self.domain, FiniteQuantifierDomain):
            raise TypeError("set relation domain 类型错误")
        if not isinstance(self.closure, SetRelationEvaluation):
            raise TypeError("set relation closure evaluation 类型错误")
        if not isinstance(self.members, tuple):
            raise TypeError("set relation domain members 必须是 tuple")
        values = []
        for item in self.members:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError("set relation domain member 必须是 value/evaluation 对")
            value, evaluation = item
            if not isinstance(value, TypedValue):
                raise TypeError("set relation domain value 类型错误")
            if not isinstance(evaluation, SetRelationEvaluation):
                raise TypeError("set relation domain evaluation 类型错误")
            if evaluation.statement.left != value.value:
                raise ValueError("MEMBER evaluation 与 TypedValue 不一致")
            values.append(value)
        if len({item.stable_key() for item in values}) != len(values):
            raise ValueError("set relation domain 不得重复成员")
        if tuple(sorted(values, key=TypedValue.stable_key)) != self.domain.values:
            raise ValueError("set relation domain values 与成员结果不一致")

    def active_premises(self) -> tuple[SetRelationEvidence, ...]:
        """返回闭域声明和 MEMBER proof 中实际采用的全部 active 前提。"""
        premises = {
            item.proposition.stable_key(): item
            for item in self.closure.active_premises()
        }
        for _value, evaluation in self.members:
            for item in evaluation.active_premises():
                premises[item.proposition.stable_key()] = item
        return tuple(premises[key] for key in sorted(premises))


@dataclass(frozen=True)
class SetQuantifierBranch:
    """一个有限域值对应的原子或复合 body 四态结果。"""

    value: TypedValue
    state: LogicEvidenceState
    trace: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """核验分支值、四态和调用方 trace 使用纯整数。"""
        if not isinstance(self.value, TypedValue):
            raise TypeError("quantifier branch value 类型错误")
        if not isinstance(self.state, LogicEvidenceState):
            raise TypeError("quantifier branch state 类型错误")
        if not isinstance(self.trace, tuple):
            raise TypeError("quantifier branch trace 必须是 tuple")
        assert_int(*self.trace, _where="SetQuantifierBranch.trace")
        if any(type(item) is not int for item in self.trace):
            raise ValueError("quantifier branch trace 必须使用严格整数")


@dataclass(frozen=True)
class SetQuantifierEvaluation:
    """有限域 EXISTS/FORALL 的四态 provisional 聚合结果。"""

    rule: ObjectIdentity
    state: LogicEvidenceState
    domain: SetRelationDomainResult
    branches: tuple[SetQuantifierBranch, ...]
    provisional: bool = True

    def __post_init__(self) -> None:
        """核验规则、域、分支和 provisional 边界没有被调用方篡改。"""
        _require_kind(self.rule, OBJECT_MINIMAL_INSTRUCTION, label="quantifier rule")
        if not isinstance(self.state, LogicEvidenceState):
            raise TypeError("quantifier evaluation state 类型错误")
        if not isinstance(self.domain, SetRelationDomainResult):
            raise TypeError("quantifier evaluation domain 类型错误")
        if not isinstance(self.branches, tuple):
            raise TypeError("quantifier evaluation branches 必须是 tuple")
        if any(not isinstance(item, SetQuantifierBranch)
               for item in self.branches):
            raise TypeError("quantifier evaluation branch 类型错误")
        if self.provisional is not True:
            raise ValueError("R-02 有限域量化只能产生 provisional 结果")


class SetRelationEngine:
    """在一个不可变 R-00 快照上执行有界集合闭包和有限域查询。"""

    def __init__(
            self,
            protocol: SetRelationProtocol,
            budget: SetRelationBudget,
            knowledge: SetRelationKnowledge,
            member_type_resolver: MemberTypeResolver | None = None,
            ) -> None:
        """核验协议和预算后一次构造可复用的确定性闭包。"""
        if not isinstance(protocol, SetRelationProtocol):
            raise TypeError("set relation protocol 类型错误")
        if not isinstance(budget, SetRelationBudget):
            raise TypeError("set relation budget 类型错误")
        if not isinstance(knowledge, SetRelationKnowledge):
            raise TypeError("set relation knowledge 类型错误")
        if (member_type_resolver is not None
                and not isinstance(member_type_resolver, MemberTypeResolver)):
            raise TypeError("member_type_resolver 未实现 MemberTypeResolver")
        if len(knowledge.evidence) > budget.max_facts:
            raise SetRelationBudgetExceeded("set relation 直接事实预算耗尽")
        self.protocol = protocol
        self.budget = budget
        self.knowledge = knowledge
        self.member_type_resolver = member_type_resolver
        self._proof_count = 0
        self._state_keys: set[tuple[int, ...]] = set()
        self._direct_support: dict[
            SetRelationStatement, list[SetRelationProof]] = {}
        self._direct_refute: dict[
            SetRelationStatement, list[SetRelationProof]] = {}
        self._index_direct()
        self._subset = self._build_subset_closure()
        self._proper = self._build_proper_closure()
        self._member = self._build_member_closure()

    def _index_direct(self) -> None:
        """把当前 active support 与显式 refute 分开索引，不从缺边制造证据。"""
        for evidence in self.knowledge.evidence:
            self.protocol.validate_statement(evidence.statement)
            if evidence.active_supported:
                proof = self._proof(
                    self.protocol.rules.direct_evidence,
                    evidence.statement,
                    EVIDENCE_SUPPORT,
                    direct=(evidence,),
                )
                self._direct_support.setdefault(
                    evidence.statement, []).append(proof)
            if evidence.state.refute:
                proof = self._proof(
                    self.protocol.rules.direct_evidence,
                    evidence.statement,
                    EVIDENCE_REFUTE,
                    direct=(evidence,),
                )
                self._direct_refute.setdefault(
                    evidence.statement, []).append(proof)
        for index in (self._direct_support, self._direct_refute):
            for statement, proofs in index.items():
                index[statement] = sorted(proofs, key=SetRelationProof.stable_key)

    def _proof(
            self,
            rule: ObjectIdentity,
            conclusion: SetRelationStatement,
            stance: int,
            *,
            direct: tuple[SetRelationEvidence, ...] = (),
            antecedents: tuple[SetRelationProof, ...] = (),
            ) -> SetRelationProof:
        """构造一个 proof 并同时执行总量和叶子前提预算。"""
        proof = SetRelationProof(
            rule,
            conclusion,
            stance,
            direct,
            antecedents,
        )
        self._proof_count += 1
        if self._proof_count > self.budget.max_proofs:
            raise SetRelationBudgetExceeded("set relation proof 预算耗尽")
        if len(proof.direct_evidence()) > self.budget.max_premises:
            raise SetRelationBudgetExceeded("set relation premise 预算耗尽")
        return proof

    def _track_state(self, statement: SetRelationStatement) -> None:
        """登记一个派生内容状态，超过预算时拒绝整个闭包。"""
        key = statement.stable_key()
        self._state_keys.add(key)
        if len(self._state_keys) > self.budget.max_states:
            raise SetRelationBudgetExceeded("set relation state 预算耗尽")

    @staticmethod
    def _proof_score(proof: SetRelationProof) -> tuple:
        """优先较少直接前提、较浅 proof，再按完整键确定唯一证明。"""
        return (
            len(proof.direct_evidence()),
            proof.depth(),
            proof.stable_key(),
        )

    def _set_best(
            self,
            target: dict[SetRelationStatement, SetRelationProof],
            proof: SetRelationProof,
            ) -> bool:
        """只保留同一派生内容的确定性最佳 proof，并报告是否发生更新。"""
        existing = target.get(proof.conclusion)
        if existing is not None and self._proof_score(existing) <= self._proof_score(
                proof):
            return False
        if existing is None:
            self._track_state(proof.conclusion)
        target[proof.conclusion] = proof
        return True

    def _direct_best(
            self,
            relation: ObjectIdentity,
            *,
            stance: int,
            ) -> dict[SetRelationStatement, SetRelationProof]:
        """选择指定 relation 每个直接内容的最佳 support 或 refute proof。"""
        source = (
            self._direct_support
            if stance == EVIDENCE_SUPPORT
            else self._direct_refute
        )
        return {
            statement: min(proofs, key=self._proof_score)
            for statement, proofs in source.items()
            if statement.relation == relation
        }

    def _build_subset_closure(
            self,
            ) -> dict[SetRelationStatement, SetRelationProof]:
        """由直接 SUBSET_EQ、严格子集降格和传递规则构造非自环闭包。"""
        subset_relation = self.protocol.subset_eq.relation
        proper_relation = self.protocol.proper_subset.relation
        closure = self._direct_best(
            subset_relation,
            stance=EVIDENCE_SUPPORT,
        )
        for statement, proof in self._direct_best(
                proper_relation,
                stance=EVIDENCE_SUPPORT,
                ).items():
            conclusion = SetRelationStatement(
                subset_relation,
                statement.left,
                statement.right,
            )
            self._set_best(closure, self._proof(
                self.protocol.rules.proper_to_subset,
                conclusion,
                EVIDENCE_SUPPORT,
                antecedents=(proof,),
            ))
        changed = True
        while changed:
            changed = False
            items = tuple(sorted(
                closure.items(),
                key=lambda item: item[0].stable_key(),
            ))
            for first, first_proof in items:
                for second, second_proof in items:
                    if first.right != second.left:
                        continue
                    if first.left == second.right:
                        continue
                    conclusion = SetRelationStatement(
                        subset_relation,
                        first.left,
                        second.right,
                    )
                    proof = self._proof(
                        self.protocol.rules.subset_transitive,
                        conclusion,
                        EVIDENCE_SUPPORT,
                        antecedents=(first_proof, second_proof),
                    )
                    changed = self._set_best(closure, proof) or changed
        return closure

    def _equal_support(
            self, left: ObjectIdentity, right: ObjectIdentity,
            ) -> SetRelationProof | None:
        """返回显式 active EQUAL 或同一身份公理的最佳支持 proof。"""
        statement = SetRelationStatement(
            self.protocol.equal.relation,
            left,
            right,
        )
        proofs = list(self._direct_support.get(statement, ()))
        if left == right:
            proofs.append(self._proof(
                self.protocol.rules.equal_identity,
                statement,
                EVIDENCE_SUPPORT,
            ))
        return None if not proofs else min(proofs, key=self._proof_score)

    def _build_proper_closure(
            self,
            ) -> dict[SetRelationStatement, SetRelationProof]:
        """构造严格子集直接、非等、传递和两种混合传递闭包。"""
        proper_relation = self.protocol.proper_subset.relation
        equal_relation = self.protocol.equal.relation
        closure = self._direct_best(
            proper_relation,
            stance=EVIDENCE_SUPPORT,
        )
        equal_refutes = self._direct_best(
            equal_relation,
            stance=EVIDENCE_REFUTE,
        )
        for subset_statement, subset_proof in self._subset.items():
            if subset_statement.left == subset_statement.right:
                continue
            equal_statement = SetRelationStatement(
                equal_relation,
                subset_statement.left,
                subset_statement.right,
            )
            unequal = equal_refutes.get(equal_statement)
            if unequal is None:
                continue
            conclusion = SetRelationStatement(
                proper_relation,
                subset_statement.left,
                subset_statement.right,
            )
            self._set_best(closure, self._proof(
                self.protocol.rules.proper_from_inequality,
                conclusion,
                EVIDENCE_SUPPORT,
                antecedents=(subset_proof, unequal),
            ))

        changed = True
        while changed:
            changed = False
            proper_items = tuple(sorted(
                closure.items(),
                key=lambda item: item[0].stable_key(),
            ))
            subset_items = tuple(sorted(
                self._subset.items(),
                key=lambda item: item[0].stable_key(),
            ))
            for first, first_proof in proper_items:
                for second, second_proof in proper_items:
                    if first.right != second.left or first.left == second.right:
                        continue
                    conclusion = SetRelationStatement(
                        proper_relation,
                        first.left,
                        second.right,
                    )
                    changed = self._set_best(closure, self._proof(
                        self.protocol.rules.proper_transitive,
                        conclusion,
                        EVIDENCE_SUPPORT,
                        antecedents=(first_proof, second_proof),
                    )) or changed
            proper_items = tuple(sorted(
                closure.items(),
                key=lambda item: item[0].stable_key(),
            ))
            for proper_statement, proper_proof in proper_items:
                for subset_statement, subset_proof in subset_items:
                    if (proper_statement.right == subset_statement.left
                            and proper_statement.left != subset_statement.right):
                        conclusion = SetRelationStatement(
                            proper_relation,
                            proper_statement.left,
                            subset_statement.right,
                        )
                        changed = self._set_best(closure, self._proof(
                            self.protocol.rules.proper_then_subset,
                            conclusion,
                            EVIDENCE_SUPPORT,
                            antecedents=(proper_proof, subset_proof),
                        )) or changed
                    if (subset_statement.right == proper_statement.left
                            and subset_statement.left != proper_statement.right):
                        conclusion = SetRelationStatement(
                            proper_relation,
                            subset_statement.left,
                            proper_statement.right,
                        )
                        changed = self._set_best(closure, self._proof(
                            self.protocol.rules.subset_then_proper,
                            conclusion,
                            EVIDENCE_SUPPORT,
                            antecedents=(subset_proof, proper_proof),
                        )) or changed
        return closure

    def _build_member_closure(
            self,
            ) -> dict[SetRelationStatement, SetRelationProof]:
        """只沿 SetExpr 的 SUBSET_EQ 正向提升 MEMBER，不生成反向关系。"""
        member_relation = self.protocol.member.relation
        closure = self._direct_best(
            member_relation,
            stance=EVIDENCE_SUPPORT,
        )
        direct_members = tuple(sorted(
            closure.items(),
            key=lambda item: item[0].stable_key(),
        ))
        subset_items = tuple(sorted(
            self._subset.items(),
            key=lambda item: item[0].stable_key(),
        ))
        for member_statement, member_proof in direct_members:
            for subset_statement, subset_proof in subset_items:
                if member_statement.right != subset_statement.left:
                    continue
                conclusion = SetRelationStatement(
                    member_relation,
                    member_statement.left,
                    subset_statement.right,
                )
                self._set_best(closure, self._proof(
                    self.protocol.rules.member_lift,
                    conclusion,
                    EVIDENCE_SUPPORT,
                    antecedents=(member_proof, subset_proof),
                ))
        return closure

    def evaluate(
            self, statement: SetRelationStatement,
            ) -> SetRelationEvaluation:
        """返回一个 typed statement 的四态结果，普通缺边始终保持 unknown。"""
        self.protocol.validate_statement(statement)
        support = list(self._direct_support.get(statement, ()))
        refute = list(self._direct_refute.get(statement, ()))
        if statement.relation == self.protocol.subset_eq.relation:
            if statement.left == statement.right:
                support.append(self._proof(
                    self.protocol.rules.subset_reflexive,
                    statement,
                    EVIDENCE_SUPPORT,
                ))
            derived = self._subset.get(statement)
            if derived is not None:
                support.append(derived)
        elif statement.relation == self.protocol.proper_subset.relation:
            derived = self._proper.get(statement)
            if derived is not None:
                support.append(derived)
            if statement.left == statement.right:
                refute.append(self._proof(
                    self.protocol.rules.proper_irreflexive,
                    statement,
                    EVIDENCE_REFUTE,
                ))
            equal = self._equal_support(statement.left, statement.right)
            if equal is not None:
                refute.append(self._proof(
                    self.protocol.rules.equal_refutes_proper,
                    statement,
                    EVIDENCE_REFUTE,
                    antecedents=(equal,),
                ))
        elif statement.relation == self.protocol.member.relation:
            derived = self._member.get(statement)
            if derived is not None:
                support.append(derived)
        elif statement.relation == self.protocol.equal.relation:
            equal = self._equal_support(statement.left, statement.right)
            if equal is not None:
                support.append(equal)
        proofs = {
            item.stable_key(): item
            for item in (*support, *refute)
        }
        ordered = tuple(proofs[key] for key in sorted(proofs))
        return SetRelationEvaluation(
            statement,
            LogicEvidenceState(bool(support), bool(refute)),
            ordered,
        )

    def finite_domain(
            self, domain: ObjectIdentity,
            ) -> SetRelationDomainResult:
        """由当前 CLOSED_DOMAIN 和支持 MEMBER 构造 typed 有限域。"""
        _require_kind(domain, OBJECT_SET_EXPR, label="finite domain")
        closure_statement = SetRelationStatement(
            self.protocol.closed_domain.relation,
            domain,
        )
        closure = self.evaluate(closure_statement)
        member_statements = tuple(sorted(
            (
                statement
                for statement in self._member
                if statement.right == domain
            ),
            key=SetRelationStatement.stable_key,
        ))
        if len(member_statements) > self.budget.max_domain_members:
            raise SetRelationBudgetExceeded("finite domain member 预算耗尽")
        members: list[tuple[TypedValue, SetRelationEvaluation]] = []
        for statement in member_statements:
            evaluation = self.evaluate(statement)
            if not evaluation.state.support:
                continue
            if self.member_type_resolver is None:
                raise SetRelationError("finite domain 缺少 member type resolver")
            value_type = self.member_type_resolver.resolve(statement.left)
            if value_type is None:
                raise SetRelationError("finite domain 成员缺少可审计一等类型")
            members.append((TypedValue(statement.left, value_type), evaluation))
        members.sort(key=lambda item: item[0].stable_key())
        closed = (
            closure.state.support
            and not closure.state.refute
            and all(not evaluation.state.refute
                    for _value, evaluation in members)
        )
        closure_evidence = ()
        if closed:
            closure_evidence = tuple(
                item.proposition
                for item in closure.active_premises()
            )
            if not closure_evidence:
                raise SetRelationError("closed domain 缺少 active 闭域 Proposition")
        finite = FiniteQuantifierDomain(
            domain,
            tuple(item[0] for item in members),
            closed,
            closure_evidence,
        )
        return SetRelationDomainResult(finite, closure, tuple(members))

    def quantify_exists(
            self,
            domain: SetRelationDomainResult,
            branches: tuple[SetQuantifierBranch, ...],
            ) -> SetQuantifierEvaluation:
        """在已知有限域上聚合 EXISTS；开放域只允许明确 witness 支持。"""
        normalized = self._validate_branches(domain, branches)
        states = tuple(item.state for item in normalized)
        if any(item.support for item in states):
            state = LogicEvidenceState(
                True,
                domain.domain.closed and all(item.refute for item in states),
            )
        elif domain.domain.closed and all(item.refute for item in states):
            state = LogicEvidenceState(False, True)
        else:
            state = LogicEvidenceState(False, False)
        return SetQuantifierEvaluation(
            self.protocol.rules.exists_aggregate,
            state,
            domain,
            normalized,
        )

    def quantify_forall(
            self,
            domain: SetRelationDomainResult,
            branches: tuple[SetQuantifierBranch, ...],
            ) -> SetQuantifierEvaluation:
        """在已知有限域上聚合 FORALL；开放域不得由已见样本全支持而通过。"""
        normalized = self._validate_branches(domain, branches)
        states = tuple(item.state for item in normalized)
        if any(item.refute for item in states):
            state = LogicEvidenceState(
                domain.domain.closed and all(item.support for item in states),
                True,
            )
        elif domain.domain.closed and all(item.support for item in states):
            state = LogicEvidenceState(True, False)
        else:
            state = LogicEvidenceState(False, False)
        return SetQuantifierEvaluation(
            self.protocol.rules.forall_aggregate,
            state,
            domain,
            normalized,
        )

    @staticmethod
    def _validate_branches(
            domain: SetRelationDomainResult,
            branches: tuple[SetQuantifierBranch, ...],
            ) -> tuple[SetQuantifierBranch, ...]:
        """要求量词分支与有限域值一一对应，不允许漏值或添加隐藏值。"""
        if not isinstance(domain, SetRelationDomainResult):
            raise TypeError("quantifier domain result 类型错误")
        if not isinstance(branches, tuple):
            raise TypeError("quantifier branches 必须是 tuple")
        if any(not isinstance(item, SetQuantifierBranch) for item in branches):
            raise TypeError("quantifier branch 类型错误")
        keys = tuple(item.value.stable_key() for item in branches)
        if len(set(keys)) != len(keys):
            raise ValueError("quantifier branches 不得重复有限域值")
        expected = tuple(item.stable_key() for item in domain.domain.values)
        if tuple(sorted(keys)) != expected:
            raise ValueError("quantifier branches 必须完整覆盖有限域")
        return tuple(sorted(branches, key=lambda item: item.value.stable_key()))


__all__ = [
    "MappingMemberTypeResolver",
    "MemberTypeResolver",
    "SetBinaryRelationProtocol",
    "SetQuantifierBranch",
    "SetQuantifierEvaluation",
    "SetRelationBudget",
    "SetRelationBudgetExceeded",
    "SetRelationDomainResult",
    "SetRelationEngine",
    "SetRelationError",
    "SetRelationEvaluation",
    "SetRelationEvidence",
    "SetRelationKnowledge",
    "SetRelationProof",
    "SetRelationProtocol",
    "SetRelationRules",
    "SetRelationStatement",
    "SetUnaryRelationProtocol",
]
