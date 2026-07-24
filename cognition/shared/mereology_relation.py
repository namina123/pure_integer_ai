"""R-04 部分整体关系族、显式规则闭包和四态查询协议。

本模块不声明任何具体部分整体关系、自然语言分类或默认传递律。调用方必须注入
relation、schema、part/whole Role 和最小规则身份；所有派生证明最终只引用 R-00
提供的 active supported 直接事实，不会自动形成新的 H-00 事实。
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq

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
    CompositionRule,
    InverseRule,
    IrreflexiveRule,
    RelationSchema,
    RelationSchemaError,
    TransitiveRule,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int


class MereologyRelationError(RuntimeError):
    """部分整体协议、证明或查询状态不完整。"""


class MereologyBudgetExceeded(MereologyRelationError):
    """部分整体查询超过调用方预算，拒绝返回部分闭包。"""


def _packed(value: tuple[int, ...]) -> tuple[int, ...]:
    """为可变长完整键增加长度边界。"""
    return len(value), *value


def _strict_key(
        value: tuple[int, ...], *, label: str,
        allow_empty: bool = False,
        ) -> tuple[int, ...]:
    """校验开放协议键只包含严格整数。"""
    if not isinstance(value, tuple) or (not value and not allow_empty):
        raise ValueError(f"{label} 必须是整数 tuple")
    assert_int(*value, _where=label)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{label} 必须使用严格整数")
    return value


def _authoritative(
        identity: ObjectIdentity, *, label: str,
        ) -> ObjectIdentity:
    """要求端点具有权威对象契约，不能使用派生索引冒充本体。"""
    if not isinstance(identity, ObjectIdentity):
        raise TypeError(f"{label} 必须是 ObjectIdentity")
    contract = object_contracts_by_kind().get(identity.object_kind)
    if contract is None or not contract.authoritative_identity:
        raise ValueError(f"{label} 缺少权威 object contract")
    return identity


def _rule_key(rule) -> tuple[int, ...]:
    """展开规则中的全部一等身份，避免依赖名称或哈希排序。"""
    if not isinstance(rule, (
            TransitiveRule, CompositionRule, InverseRule,
            IrreflexiveRule)):
        raise TypeError("mereology 包含未支持的规则类型")
    result: list[int] = []
    for value in rule.__dict__.values():
        if isinstance(value, ObjectIdentity):
            result.extend(_packed(value.stable_key()))
    return tuple(result)


@dataclass(frozen=True)
class MereologyRelationProtocol:
    """一个具体部分整体 relation 的 schema 与 canonical Role。"""

    schema: RelationSchema
    part_role: ObjectIdentity
    whole_role: ObjectIdentity

    def __post_init__(self) -> None:
        """要求 schema 恰好声明两个互异、单值且有名的端点 Role。"""
        if not isinstance(self.schema, RelationSchema):
            raise TypeError("mereology relation schema 类型错误")
        if self.part_role == self.whole_role:
            raise RelationSchemaError("part Role 与 whole Role 必须不同")
        if len(self.schema.slots) != 2:
            raise RelationSchemaError("mereology schema 必须恰好声明两个 Role")
        if {slot.role for slot in self.schema.slots} != {
                self.part_role, self.whole_role}:
            raise RelationSchemaError("mereology schema Role 与协议不一致")
        for role in (self.part_role, self.whole_role):
            if not isinstance(role, ObjectIdentity):
                raise TypeError("mereology Role 类型错误")
            if role.object_kind != OBJECT_ROLE:
                raise RelationSchemaError("mereology Role 必须是一等 Role")
            slot = self.schema.slot(role)
            if slot is None or (slot.min_count, slot.max_count) != (1, 1):
                raise RelationSchemaError("mereology Role 必须恰好出现一次")

    @property
    def relation(self) -> ObjectIdentity:
        """返回 schema 绑定的一等关系身份。"""
        return self.schema.relation

    def roles(self) -> tuple[ObjectIdentity, ObjectIdentity]:
        """按 canonical part、whole 顺序返回完整 Role。"""
        return self.part_role, self.whole_role

    def stable_key(self) -> tuple[int, ...]:
        """返回 relation、schema 和 canonical Role 的完整状态键。"""
        return (
            *_packed(self.schema.schema.stable_key()),
            *_packed(self.relation.stable_key()),
            *_packed(self.part_role.stable_key()),
            *_packed(self.whole_role.stable_key()),
        )


@dataclass(frozen=True)
class MereologyProtocol:
    """部分整体关系族和全部显式允许的代数规则。"""

    relations: tuple[MereologyRelationProtocol, ...]
    transitive_rules: tuple[TransitiveRule, ...] = ()
    composition_rules: tuple[CompositionRule, ...] = ()
    inverse_rules: tuple[InverseRule, ...] = ()
    irreflexive_rules: tuple[IrreflexiveRule, ...] = ()

    def __post_init__(self) -> None:
        """核验关系身份唯一，并确保每条规则只引用已注入 relation/Role。"""
        if not isinstance(self.relations, tuple) or not self.relations:
            raise ValueError("mereology protocol 至少需要一个 relation")
        if any(not isinstance(item, MereologyRelationProtocol)
               for item in self.relations):
            raise TypeError("mereology relations 类型错误")
        relation_ids = tuple(item.relation for item in self.relations)
        schema_ids = tuple(item.schema.schema for item in self.relations)
        if len(set(relation_ids)) != len(relation_ids):
            raise ValueError("mereology relation 身份不得重复")
        if len(set(schema_ids)) != len(schema_ids):
            raise ValueError("mereology schema 身份不得重复")
        object.__setattr__(self, "relations", tuple(sorted(
            self.relations, key=MereologyRelationProtocol.stable_key)))
        groups = (
            (self.transitive_rules, TransitiveRule),
            (self.composition_rules, CompositionRule),
            (self.inverse_rules, InverseRule),
            (self.irreflexive_rules, IrreflexiveRule),
        )
        for values, expected in groups:
            if not isinstance(values, tuple) or any(
                    not isinstance(item, expected) for item in values):
                raise TypeError("mereology rule group 类型错误")
            if len({_rule_key(item) for item in values}) != len(values):
                raise ValueError("mereology rule 不得重复")
        for rule in self.transitive_rules:
            self._validate_role_pair(
                rule.relation, rule.left_role, rule.right_role)
        for rule in self.irreflexive_rules:
            self._validate_role_pair(
                rule.relation, rule.left_role, rule.right_role)
        for rule in self.inverse_rules:
            self._validate_role_pair(
                rule.premise_relation,
                rule.premise_left_role,
                rule.premise_right_role,
            )
            self._validate_role_pair(
                rule.result_relation,
                rule.result_left_role,
                rule.result_right_role,
            )
        for rule in self.composition_rules:
            self._validate_role_pair(
                rule.first_relation,
                rule.first_input_role,
                rule.first_join_role,
            )
            self._validate_role_pair(
                rule.second_relation,
                rule.second_join_role,
                rule.second_output_role,
            )
            self._validate_role_pair(
                rule.result_relation,
                rule.result_input_role,
                rule.result_output_role,
            )
        for name in (
                "transitive_rules", "composition_rules",
                "inverse_rules", "irreflexive_rules"):
            object.__setattr__(self, name, tuple(sorted(
                getattr(self, name), key=_rule_key)))

    def relation_protocol(
            self, relation: ObjectIdentity,
            ) -> MereologyRelationProtocol | None:
        """按完整 relation 身份读取协议，不按名称或二元 shape 降级。"""
        return next((
            item for item in self.relations if item.relation == relation
        ), None)

    def require_relation(
            self, relation: ObjectIdentity,
            ) -> MereologyRelationProtocol:
        """读取已注册 relation，未知关系立即失败。"""
        protocol = self.relation_protocol(relation)
        if protocol is None:
            raise MereologyRelationError("relation 未注册为 mereology variant")
        return protocol

    def validate_statement(
            self, statement: "MereologyStatement",
            ) -> "MereologyStatement":
        """按目标 relation schema 核验 canonical part/whole 端点类型。"""
        if not isinstance(statement, MereologyStatement):
            raise TypeError("mereology statement 类型错误")
        relation = self.require_relation(statement.relation)
        for role, filler in (
                (relation.part_role, statement.part),
                (relation.whole_role, statement.whole)):
            slot = relation.schema.slot(role)
            if slot is None or filler.object_kind not in slot.allowed_object_kinds:
                raise RelationSchemaError("mereology statement 端点类型不满足 schema")
        return statement

    def _validate_role_pair(
            self, relation: ObjectIdentity,
            left_role: ObjectIdentity,
            right_role: ObjectIdentity,
            ) -> None:
        """要求规则端点恰好使用目标 relation 注入的两个 Role。"""
        protocol = self.require_relation(relation)
        if {left_role, right_role} != set(protocol.roles()):
            raise RelationSchemaError("mereology rule Role 与 relation 协议不一致")

    def stable_key(self) -> tuple[int, ...]:
        """返回关系族和全部显式规则的完整状态键。"""
        result = [1, len(self.relations)]
        for relation in self.relations:
            result.extend(_packed(relation.stable_key()))
        for tag, rules in enumerate((
                self.transitive_rules,
                self.composition_rules,
                self.inverse_rules,
                self.irreflexive_rules,
                ), start=1):
            result.extend((tag, len(rules)))
            for rule in rules:
                result.extend(_packed(_rule_key(rule)))
        return tuple(result)


@dataclass(frozen=True)
class MereologyBudget:
    """限制直接事实、闭包规模、规则应用和返回结果数量。"""

    max_direct_facts: int
    max_closure_statements: int
    max_rule_applications: int
    max_options: int

    def __post_init__(self) -> None:
        """要求所有预算均为严格正整数。"""
        values = (
            self.max_direct_facts,
            self.max_closure_statements,
            self.max_rule_applications,
            self.max_options,
        )
        assert_int(*values, _where="MereologyBudget")
        if any(type(item) is not int or item <= 0 for item in values):
            raise ValueError("mereology budget 必须为严格正整数")

    def stable_key(self) -> tuple[int, ...]:
        """返回四个预算维度的完整整数键。"""
        return (
            self.max_direct_facts,
            self.max_closure_statements,
            self.max_rule_applications,
            self.max_options,
        )


@dataclass(frozen=True)
class MereologyStatement:
    """一个 canonical relation+part+whole 内容，不以绑定顺序表达方向。"""

    relation: ObjectIdentity
    part: ObjectIdentity
    whole: ObjectIdentity

    def __post_init__(self) -> None:
        """核验 relation 和两个端点均为一等权威对象。"""
        if not isinstance(self.relation, ObjectIdentity):
            raise TypeError("mereology relation 类型错误")
        if self.relation.object_kind != OBJECT_CONCEPT:
            raise ValueError("mereology relation 必须是一等 Concept")
        _authoritative(self.part, label="mereology part")
        _authoritative(self.whole, label="mereology whole")

    def stable_key(self) -> tuple[int, ...]:
        """返回 relation、part 和 whole 的无歧义完整键。"""
        return (
            *_packed(self.relation.stable_key()),
            *_packed(self.part.stable_key()),
            *_packed(self.whole.stable_key()),
        )


@dataclass(frozen=True)
class MereologyPattern:
    """按 relation、part、whole 任意组合过滤发现结果。"""

    relation: ObjectIdentity | None = None
    part: ObjectIdentity | None = None
    whole: ObjectIdentity | None = None

    def __post_init__(self) -> None:
        """核验可选过滤身份，允许全空模式发现全部预算内结果。"""
        if self.relation is not None:
            if (not isinstance(self.relation, ObjectIdentity)
                    or self.relation.object_kind != OBJECT_CONCEPT):
                raise TypeError("mereology pattern relation 类型错误")
        if self.part is not None:
            _authoritative(self.part, label="mereology pattern part")
        if self.whole is not None:
            _authoritative(self.whole, label="mereology pattern whole")

    def matches(self, statement: MereologyStatement) -> bool:
        """按非空过滤字段匹配一个 canonical statement。"""
        if not isinstance(statement, MereologyStatement):
            raise TypeError("mereology pattern statement 类型错误")
        return (
            (self.relation is None or statement.relation == self.relation)
            and (self.part is None or statement.part == self.part)
            and (self.whole is None or statement.whole == self.whole)
        )

    def exact_statement(self) -> MereologyStatement | None:
        """完整指定三个字段时返回精确 statement，否则返回 None。"""
        if self.relation is None or self.part is None or self.whole is None:
            return None
        return MereologyStatement(self.relation, self.part, self.whole)

    def stable_key(self) -> tuple[int, ...]:
        """返回三个可选过滤字段的完整键。"""
        result = []
        for identity in (self.relation, self.part, self.whole):
            key = () if identity is None else identity.stable_key()
            result.extend(_packed(key))
        return tuple(result)


@dataclass(frozen=True)
class MereologyEvidence:
    """从一个 R-00 forming 候选恢复的当前部分整体直接证据。"""

    statement: MereologyStatement
    proposition: ObjectIdentity
    semantic_context: ObjectIdentity
    scope: ScopeIdentity
    hypothesis: HypothesisKey
    state: LogicEvidenceState
    evidence: tuple[EvidenceRecord, ...]
    forming_sources: tuple[SourceRef, ...]
    active_supported: bool

    def __post_init__(self) -> None:
        """核验 Evidence、Hypothesis、来源和 active 投影一致。"""
        if not isinstance(self.statement, MereologyStatement):
            raise TypeError("mereology evidence statement 类型错误")
        if not isinstance(self.proposition, ObjectIdentity):
            raise TypeError("mereology evidence proposition 类型错误")
        if not isinstance(self.semantic_context, ObjectIdentity):
            raise TypeError("mereology evidence semantic_context 类型错误")
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("mereology evidence scope 类型错误")
        proposition_source = semantic_source(self.proposition)
        if (semantic_source(self.semantic_context) != proposition_source
                or self.scope.source != proposition_source):
            raise ValueError("mereology Evidence context/scope 必须绑定 Proposition 来源")
        if not isinstance(self.hypothesis, HypothesisKey):
            raise TypeError("mereology evidence hypothesis 类型错误")
        if not isinstance(self.state, LogicEvidenceState):
            raise TypeError("mereology evidence state 类型错误")
        if not isinstance(self.evidence, tuple) or any(
                not isinstance(item, EvidenceRecord) for item in self.evidence):
            raise TypeError("mereology evidence records 类型错误")
        if any(item.hypothesis != self.hypothesis for item in self.evidence):
            raise ValueError("mereology Evidence 属于其他 Hypothesis")
        if not isinstance(self.forming_sources, tuple) or any(
                not isinstance(item, SourceRef) for item in self.forming_sources):
            raise TypeError("mereology forming_sources 类型错误")
        if len(set(self.forming_sources)) != len(self.forming_sources):
            raise ValueError("mereology forming_sources 不得重复")
        if type(self.active_supported) is not bool:
            raise TypeError("mereology active_supported 必须是严格 bool")
        support = any(item.stance == EVIDENCE_SUPPORT for item in self.evidence)
        refute = any(item.stance == EVIDENCE_REFUTE for item in self.evidence)
        if (support, refute) != (self.state.support, self.state.refute):
            raise ValueError("mereology Evidence 与四态不一致")
        if self.active_supported and self.state.status != EPISTEMIC_SUPPORTED:
            raise ValueError("active_supported 只能标记纯支持候选")

    def stable_key(self) -> tuple[int, ...]:
        """返回 statement、命题、Evidence 和来源的完整键。"""
        result = [
            *_packed(self.statement.stable_key()),
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
class MereologyKnowledge:
    """一次查询从 R-00 读取的不可变部分整体 Evidence 集。"""

    evidence: tuple[MereologyEvidence, ...]

    def __post_init__(self) -> None:
        """按完整 Evidence 键拒绝重复并稳定排序。"""
        if not isinstance(self.evidence, tuple) or any(
                not isinstance(item, MereologyEvidence) for item in self.evidence):
            raise TypeError("mereology knowledge 类型错误")
        keyed = {item.stable_key(): item for item in self.evidence}
        if len(keyed) != len(self.evidence):
            raise ValueError("mereology knowledge 不得重复 Evidence")
        object.__setattr__(self, "evidence", tuple(
            keyed[key] for key in sorted(keyed)))


@dataclass(frozen=True)
class MereologyRuleApplication:
    """一次显式规则应用及其输入、输出 statement。"""

    rule: ObjectIdentity
    premises: tuple[MereologyStatement, ...]
    result: MereologyStatement

    def __post_init__(self) -> None:
        """核验规则身份、非空前提和结果类型。"""
        if (not isinstance(self.rule, ObjectIdentity)
                or self.rule.object_kind != OBJECT_MINIMAL_INSTRUCTION):
            raise TypeError("mereology rule application 缺少最小指令身份")
        if not isinstance(self.premises, tuple) or not self.premises:
            raise ValueError("mereology rule application 必须保留前提")
        if any(not isinstance(item, MereologyStatement)
               for item in self.premises):
            raise TypeError("mereology rule application 前提类型错误")
        if not isinstance(self.result, MereologyStatement):
            raise TypeError("mereology rule application result 类型错误")

    def stable_key(self) -> tuple[int, ...]:
        """返回规则、前提顺序和结果的完整审计键。"""
        result = [*_packed(self.rule.stable_key()), len(self.premises)]
        for premise in self.premises:
            result.extend(_packed(premise.stable_key()))
        result.extend(_packed(self.result.stable_key()))
        return tuple(result)


@dataclass(frozen=True)
class MereologySupportProof:
    """一个支持证明，叶子全部是 active supported 直接 Evidence。"""

    statement: MereologyStatement
    direct_premises: tuple[MereologyEvidence, ...]
    applications: tuple[MereologyRuleApplication, ...] = ()

    def __post_init__(self) -> None:
        """核验直接前提真实 active，并稳定拒绝重复证明元素。"""
        if not isinstance(self.statement, MereologyStatement):
            raise TypeError("mereology support proof statement 类型错误")
        if not isinstance(self.direct_premises, tuple) or not self.direct_premises:
            raise ValueError("mereology support proof 必须保留直接前提")
        if any(not isinstance(item, MereologyEvidence)
               or not item.active_supported for item in self.direct_premises):
            raise ValueError("mereology proof 只能引用 active supported 直接事实")
        if len({item.stable_key() for item in self.direct_premises}) != len(
                self.direct_premises):
            raise ValueError("mereology proof 不得重复直接前提")
        if not isinstance(self.applications, tuple) or any(
                not isinstance(item, MereologyRuleApplication)
                for item in self.applications):
            raise TypeError("mereology proof applications 类型错误")
        if len({item.stable_key() for item in self.applications}) != len(
                self.applications):
            raise ValueError("mereology proof 不得重复规则应用")
        if self.applications and self.applications[-1].result != self.statement:
            raise ValueError("mereology proof 最后规则结果与 statement 不一致")
        if not self.applications and any(
                item.statement != self.statement for item in self.direct_premises):
            raise ValueError("直接 proof 的 Evidence 必须属于当前 statement")

    def stable_key(self) -> tuple[int, ...]:
        """展开 statement、直接前提和完整规则链。"""
        result = [
            *_packed(self.statement.stable_key()),
            len(self.direct_premises),
        ]
        for premise in self.direct_premises:
            result.extend(_packed(premise.stable_key()))
        result.append(len(self.applications))
        for application in self.applications:
            result.extend(_packed(application.stable_key()))
        return tuple(result)

    def selection_key(self) -> tuple:
        """优先直接或较短规则链，再以完整证明键稳定裁决。"""
        return len(self.applications), self.stable_key()


@dataclass(frozen=True)
class MereologyRuleRefute:
    """显式反自反规则对一个 active 支持自环形成的 provisional 反驳。"""

    rule: ObjectIdentity
    proof: MereologySupportProof

    def __post_init__(self) -> None:
        """要求规则为最小指令且证明结果确为自环。"""
        if (not isinstance(self.rule, ObjectIdentity)
                or self.rule.object_kind != OBJECT_MINIMAL_INSTRUCTION):
            raise TypeError("mereology rule refute 缺少最小指令身份")
        if not isinstance(self.proof, MereologySupportProof):
            raise TypeError("mereology rule refute proof 类型错误")
        if self.proof.statement.part != self.proof.statement.whole:
            raise ValueError("irreflexive refute 只能针对自环")

    @property
    def statement(self) -> MereologyStatement:
        """返回被反自反规则反驳的 canonical statement。"""
        return self.proof.statement

    def stable_key(self) -> tuple[int, ...]:
        """返回规则和支持证明的完整审计键。"""
        return (
            *_packed(self.rule.stable_key()),
            *_packed(self.proof.stable_key()),
        )


@dataclass(frozen=True)
class MereologyEvaluation:
    """一个完整 statement 的直接、派生和规则反驳四态。"""

    statement: MereologyStatement
    state: LogicEvidenceState
    direct_evidence: tuple[MereologyEvidence, ...]
    support_proof: MereologySupportProof | None
    rule_refutes: tuple[MereologyRuleRefute, ...]

    def __post_init__(self) -> None:
        """核验全部证据属于当前 statement 且证据位与状态一致。"""
        if not isinstance(self.statement, MereologyStatement):
            raise TypeError("mereology evaluation statement 类型错误")
        if not isinstance(self.state, LogicEvidenceState):
            raise TypeError("mereology evaluation state 类型错误")
        if not isinstance(self.direct_evidence, tuple) or any(
                not isinstance(item, MereologyEvidence)
                or item.statement != self.statement
                for item in self.direct_evidence):
            raise TypeError("mereology evaluation direct evidence 类型错误")
        if self.support_proof is not None and (
                not isinstance(self.support_proof, MereologySupportProof)
                or self.support_proof.statement != self.statement):
            raise TypeError("mereology evaluation support proof 类型错误")
        if not isinstance(self.rule_refutes, tuple) or any(
                not isinstance(item, MereologyRuleRefute)
                or item.statement != self.statement
                for item in self.rule_refutes):
            raise TypeError("mereology evaluation rule refute 类型错误")
        support = self.support_proof is not None or any(
            item.state.support for item in self.direct_evidence)
        refute = bool(self.rule_refutes) or any(
            item.state.refute for item in self.direct_evidence)
        if (support, refute) != (self.state.support, self.state.refute):
            raise ValueError("mereology evaluation 与证据四态不一致")

    def active_premises(self) -> tuple[MereologyEvidence, ...]:
        """返回当前 provisional 支持所依赖的全部 active 直接事实。"""
        if self.support_proof is None:
            return ()
        return self.support_proof.direct_premises


@dataclass(frozen=True)
class MereologySelection:
    """一次精确或发现查询的完整四态结果集合。"""

    pattern: MereologyPattern
    evaluations: tuple[MereologyEvaluation, ...]

    def __post_init__(self) -> None:
        """要求结果稳定有序、互不重复且全部匹配查询模式。"""
        if not isinstance(self.pattern, MereologyPattern):
            raise TypeError("mereology selection pattern 类型错误")
        if not isinstance(self.evaluations, tuple) or any(
                not isinstance(item, MereologyEvaluation)
                for item in self.evaluations):
            raise TypeError("mereology selection evaluations 类型错误")
        statements = tuple(item.statement for item in self.evaluations)
        if any(not self.pattern.matches(item) for item in statements):
            raise ValueError("mereology selection 包含模式外 statement")
        if len(set(statements)) != len(statements):
            raise ValueError("mereology selection 不得重复 statement")
        if statements != tuple(sorted(
                statements, key=MereologyStatement.stable_key)):
            raise ValueError("mereology selection 顺序不稳定")

    def pure_supported(self) -> tuple[MereologyEvaluation, ...]:
        """返回全部纯支持结果；多 part 或多 whole 不视为歧义。"""
        return tuple(
            item for item in self.evaluations
            if item.state.status == EPISTEMIC_SUPPORTED
        )


class MereologyRelationEngine:
    """在不可变 R-00 快照上执行显式规则 fixpoint 和四态查询。"""

    def __init__(
            self,
            protocol: MereologyProtocol,
            budget: MereologyBudget,
            knowledge: MereologyKnowledge,
            ) -> None:
        """绑定协议和预算，并完整计算闭包；失败时不暴露前缀结果。"""
        if not isinstance(protocol, MereologyProtocol):
            raise TypeError("mereology protocol 类型错误")
        if not isinstance(budget, MereologyBudget):
            raise TypeError("mereology budget 类型错误")
        if not isinstance(knowledge, MereologyKnowledge):
            raise TypeError("mereology knowledge 类型错误")
        if len(knowledge.evidence) > budget.max_direct_facts:
            raise MereologyBudgetExceeded("mereology 直接事实预算耗尽")
        self.protocol = protocol
        self.budget = budget
        self.knowledge = knowledge
        direct: dict[MereologyStatement, list[MereologyEvidence]] = {}
        for item in knowledge.evidence:
            self.protocol.validate_statement(item.statement)
            direct.setdefault(item.statement, []).append(item)
        self._direct = {
            statement: tuple(sorted(values, key=MereologyEvidence.stable_key))
            for statement, values in direct.items()
        }
        self._proofs, self._rule_refutes = self._close()

    def evaluate(self, statement: MereologyStatement) -> MereologyEvaluation:
        """聚合一个完整 statement 的直接、派生和规则反驳四态。"""
        if not isinstance(statement, MereologyStatement):
            raise TypeError("mereology evaluate statement 类型错误")
        self.protocol.validate_statement(statement)
        direct = self._direct.get(statement, ())
        proof = self._proofs.get(statement)
        refutes = self._rule_refutes.get(statement, ())
        return MereologyEvaluation(
            statement,
            LogicEvidenceState(
                proof is not None or any(item.state.support for item in direct),
                bool(refutes) or any(item.state.refute for item in direct),
            ),
            direct,
            proof,
            refutes,
        )

    def select(self, pattern: MereologyPattern) -> MereologySelection:
        """返回全部匹配四态结果；精确缺边显式返回 unknown。"""
        if not isinstance(pattern, MereologyPattern):
            raise TypeError("mereology select pattern 类型错误")
        if pattern.relation is not None:
            self.protocol.require_relation(pattern.relation)
        statements = {
            item for item in set(self._direct) | set(self._proofs)
            | set(self._rule_refutes)
            if pattern.matches(item)
        }
        exact = pattern.exact_statement()
        if exact is not None:
            statements.add(exact)
        ordered = tuple(sorted(statements, key=MereologyStatement.stable_key))
        if len(ordered) > self.budget.max_options:
            raise MereologyBudgetExceeded("mereology option 预算耗尽")
        return MereologySelection(
            pattern,
            tuple(self.evaluate(item) for item in ordered),
        )

    def _close(self) -> tuple[dict, dict]:
        """以 active supported 直接事实为叶子执行确定性有界 fixpoint。"""
        proofs: dict[MereologyStatement, MereologySupportProof] = {}
        for statement, evidence in self._direct.items():
            active = tuple(item for item in evidence if item.active_supported)
            if active:
                proofs[statement] = MereologySupportProof(statement, active)
        if len(proofs) > self.budget.max_closure_statements:
            raise MereologyBudgetExceeded("mereology 闭包 statement 预算耗尽")

        queue = []
        serial = 0
        for statement in sorted(proofs, key=MereologyStatement.stable_key):
            proof = proofs[statement]
            heapq.heappush(queue, (
                proof.selection_key(), statement.stable_key(), serial, statement))
            serial += 1
        seen_applications: set[tuple[int, ...]] = set()
        applications = 0

        def offer(
                rule: ObjectIdentity,
                premise_proofs: tuple[MereologySupportProof, ...],
                result: MereologyStatement,
                ) -> None:
            """记录唯一规则应用，并按最短完整证明更新闭包队列。"""
            nonlocal applications, serial
            application = MereologyRuleApplication(
                rule,
                tuple(item.statement for item in premise_proofs),
                result,
            )
            direct_map = {
                item.stable_key(): item
                for proof in premise_proofs
                for item in proof.direct_premises
            }
            prior_apps: list[MereologyRuleApplication] = []
            prior_keys = set()
            for proof in premise_proofs:
                for item in proof.applications:
                    key = item.stable_key()
                    if key not in prior_keys:
                        prior_keys.add(key)
                        prior_apps.append(item)
            if application.stable_key() in prior_keys:
                return
            prior_apps.append(application)
            candidate = MereologySupportProof(
                result,
                tuple(direct_map[key] for key in sorted(direct_map)),
                tuple(prior_apps),
            )
            application_key = candidate.stable_key()
            if application_key in seen_applications:
                return
            seen_applications.add(application_key)
            applications += 1
            if applications > self.budget.max_rule_applications:
                raise MereologyBudgetExceeded("mereology 规则应用预算耗尽")
            current = proofs.get(result)
            if current is not None and not current.applications:
                return
            if current is not None and current.selection_key() <= candidate.selection_key():
                return
            is_new = current is None
            proofs[result] = candidate
            if is_new and len(proofs) > self.budget.max_closure_statements:
                raise MereologyBudgetExceeded("mereology 闭包 statement 预算耗尽")
            heapq.heappush(queue, (
                candidate.selection_key(), result.stable_key(), serial, result))
            serial += 1

        while queue:
            proof_key, _statement_key, _serial, statement = heapq.heappop(queue)
            current = proofs.get(statement)
            if current is None or current.selection_key() != proof_key:
                continue
            snapshot = tuple(
                proofs[item]
                for item in sorted(proofs, key=MereologyStatement.stable_key)
            )
            for rule in self.protocol.inverse_rules:
                if statement.relation != rule.premise_relation:
                    continue
                left = self._filler(statement, rule.premise_left_role)
                right = self._filler(statement, rule.premise_right_role)
                result = self._statement_from_values(
                    rule.result_relation,
                    rule.result_left_role, right,
                    rule.result_right_role, left,
                )
                offer(rule.rule, (current,), result)
            for rule in self.protocol.transitive_rules:
                if statement.relation != rule.relation:
                    continue
                for other in snapshot:
                    if other.statement.relation != rule.relation:
                        continue
                    self._offer_join(
                        rule.rule,
                        current, other,
                        rule.left_role, rule.right_role,
                        rule.left_role, rule.right_role,
                        rule.relation,
                        rule.left_role, rule.right_role,
                        offer,
                    )
                    self._offer_join(
                        rule.rule,
                        other, current,
                        rule.left_role, rule.right_role,
                        rule.left_role, rule.right_role,
                        rule.relation,
                        rule.left_role, rule.right_role,
                        offer,
                    )
            for rule in self.protocol.composition_rules:
                for other in snapshot:
                    if (statement.relation == rule.first_relation
                            and other.statement.relation == rule.second_relation):
                        self._offer_join(
                            rule.rule,
                            current, other,
                            rule.first_input_role, rule.first_join_role,
                            rule.second_join_role, rule.second_output_role,
                            rule.result_relation,
                            rule.result_input_role, rule.result_output_role,
                            offer,
                        )
                    if (other.statement.relation == rule.first_relation
                            and statement.relation == rule.second_relation):
                        self._offer_join(
                            rule.rule,
                            other, current,
                            rule.first_input_role, rule.first_join_role,
                            rule.second_join_role, rule.second_output_role,
                            rule.result_relation,
                            rule.result_input_role, rule.result_output_role,
                            offer,
                        )

        refutes: dict[MereologyStatement, tuple[MereologyRuleRefute, ...]] = {}
        for rule in self.protocol.irreflexive_rules:
            for statement in sorted(proofs, key=MereologyStatement.stable_key):
                if statement.relation != rule.relation:
                    continue
                if (self._filler(statement, rule.left_role)
                        != self._filler(statement, rule.right_role)):
                    continue
                item = MereologyRuleRefute(rule.rule, proofs[statement])
                key = item.stable_key()
                if key in seen_applications:
                    continue
                seen_applications.add(key)
                applications += 1
                if applications > self.budget.max_rule_applications:
                    raise MereologyBudgetExceeded("mereology 规则应用预算耗尽")
                refutes.setdefault(statement, ())
                refutes[statement] = tuple(sorted(
                    (*refutes[statement], item),
                    key=MereologyRuleRefute.stable_key,
                ))
        return proofs, refutes

    def _filler(
            self, statement: MereologyStatement, role: ObjectIdentity,
            ) -> ObjectIdentity:
        """按 relation 协议解释有名 Role，不依赖 binding 或字段顺序。"""
        protocol = self.protocol.require_relation(statement.relation)
        if role == protocol.part_role:
            return statement.part
        if role == protocol.whole_role:
            return statement.whole
        raise MereologyRelationError("Role 不属于 statement relation")

    def _statement_from_values(
            self,
            relation: ObjectIdentity,
            first_role: ObjectIdentity,
            first: ObjectIdentity,
            second_role: ObjectIdentity,
            second: ObjectIdentity,
            ) -> MereologyStatement:
        """把规则输出 Role 重新规范化为 canonical part/whole statement。"""
        protocol = self.protocol.require_relation(relation)
        values = {first_role: first, second_role: second}
        if set(values) != set(protocol.roles()):
            raise MereologyRelationError("规则输出 Role 与 relation 协议不一致")
        return self.protocol.validate_statement(MereologyStatement(
            relation,
            values[protocol.part_role],
            values[protocol.whole_role],
        ))

    def _offer_join(
            self,
            rule: ObjectIdentity,
            first: MereologySupportProof,
            second: MereologySupportProof,
            first_input_role: ObjectIdentity,
            first_join_role: ObjectIdentity,
            second_join_role: ObjectIdentity,
            second_output_role: ObjectIdentity,
            result_relation: ObjectIdentity,
            result_input_role: ObjectIdentity,
            result_output_role: ObjectIdentity,
            offer,
            ) -> None:
        """仅在显式 join Role 的完整 filler 相等时提交二元规则候选。"""
        join = self._filler(first.statement, first_join_role)
        if join != self._filler(second.statement, second_join_role):
            return
        result = self._statement_from_values(
            result_relation,
            result_input_role,
            self._filler(first.statement, first_input_role),
            result_output_role,
            self._filler(second.statement, second_output_role),
        )
        offer(rule, (first, second), result)


__all__ = [
    "MereologyBudget",
    "MereologyBudgetExceeded",
    "MereologyEvaluation",
    "MereologyEvidence",
    "MereologyKnowledge",
    "MereologyPattern",
    "MereologyProtocol",
    "MereologyRelationEngine",
    "MereologyRelationError",
    "MereologyRelationProtocol",
    "MereologyRuleApplication",
    "MereologyRuleRefute",
    "MereologySelection",
    "MereologyStatement",
    "MereologySupportProof",
]
