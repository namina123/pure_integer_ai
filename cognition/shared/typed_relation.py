"""S-01 带类型关系事实、schema、规则身份和派生候选协议。

本模块不声明任何具体关系、角色或代数律。关系和角色都是调用方注入的一等
图身份；宿主代码只提供类型、基数和最小规则形状的通用校验。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.evidence_candidate import (
    EvidenceCandidateDefinition,
)
from pure_integer_ai.cognition.shared.hypothesis import (
    EPISTEMIC_SUPPORTED,
    LIFECYCLE_ACTIVE,
    HypothesisKey,
    HypothesisSnapshot,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONCEPT,
    OBJECT_MINIMAL_INSTRUCTION,
    OBJECT_PROPOSITION,
    OBJECT_ROLE,
    OBJECT_STRUCTURE_CONCEPT,
    ObjectIdentity,
    object_contracts_by_kind,
)
from pure_integer_ai.cognition.shared.semantic_object import (
    AtomicPropositionDefinition,
    AtomicRoleBinding,
    PropositionKnowledge,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int


class RelationSchemaError(ValueError):
    """关系事实或 schema 不满足 typed 协议。"""


def _require_identity_kind(
        identity: ObjectIdentity, expected: int, *, label: str,
        ) -> ObjectIdentity:
    """核验一等图身份的宿主对象类型，不解释其具体图中含义。"""
    if not isinstance(identity, ObjectIdentity):
        raise TypeError(f"{label} 必须是 ObjectIdentity")
    if identity.object_kind != expected:
        raise RelationSchemaError(f"{label} 对象类型不匹配")
    return identity


@dataclass(frozen=True)
class RelationSlotSchema:
    """一个注入 Role 的允许对象类型和基数约束。"""

    role: ObjectIdentity
    allowed_object_kinds: frozenset[int]
    min_count: int
    max_count: int | None

    def __post_init__(self) -> None:
        _require_identity_kind(self.role, OBJECT_ROLE, label="slot role")
        if (not isinstance(self.allowed_object_kinds, frozenset)
                or not self.allowed_object_kinds):
            raise RelationSchemaError(
                "allowed_object_kinds 必须是非空 frozenset")
        assert_int(
            *self.allowed_object_kinds,
            self.min_count,
            _where="RelationSlotSchema",
        )
        if any(type(item) is not int for item in self.allowed_object_kinds):
            raise RelationSchemaError("允许对象类型必须是严格整数")
        contracts = object_contracts_by_kind()
        if any(
                item not in contracts
                or not contracts[item].authoritative_identity
                for item in self.allowed_object_kinds):
            raise RelationSchemaError(
                "允许对象类型必须具有权威 object contract")
        if type(self.min_count) is not int or self.min_count < 0:
            raise RelationSchemaError("min_count 必须为非负严格整数")
        if self.max_count is not None:
            assert_int(self.max_count, _where="RelationSlotSchema.max_count")
            if (type(self.max_count) is not int
                    or self.max_count < self.min_count):
                raise RelationSchemaError(
                    "max_count 必须不小于 min_count")


@dataclass(frozen=True)
class SameKindConstraint:
    """要求若干 Role 的实际 filler 使用相同对象类型。"""

    constraint: ObjectIdentity
    roles: tuple[ObjectIdentity, ...]

    def __post_init__(self) -> None:
        _require_identity_kind(
            self.constraint, OBJECT_STRUCTURE_CONCEPT,
            label="same-kind constraint")
        if not isinstance(self.roles, tuple) or len(self.roles) < 2:
            raise RelationSchemaError("same-kind 约束至少需要两个 Role")
        if len(set(self.roles)) != len(self.roles):
            raise RelationSchemaError("same-kind 约束不得重复 Role")
        for role in self.roles:
            _require_identity_kind(role, OBJECT_ROLE, label="constraint role")


@dataclass(frozen=True)
class RelationSchema:
    """一个一等关系 Concept 的开放 n 元角色协议。"""

    schema: ObjectIdentity
    relation: ObjectIdentity
    slots: tuple[RelationSlotSchema, ...]
    same_kind_constraints: tuple[SameKindConstraint, ...] = ()

    def __post_init__(self) -> None:
        _require_identity_kind(
            self.schema, OBJECT_STRUCTURE_CONCEPT, label="relation schema")
        _require_identity_kind(
            self.relation, OBJECT_CONCEPT, label="relation")
        if not isinstance(self.slots, tuple) or not self.slots:
            raise RelationSchemaError("relation schema 至少需要一个 slot")
        if any(not isinstance(slot, RelationSlotSchema)
               for slot in self.slots):
            raise TypeError("slots 只能包含 RelationSlotSchema")
        roles = tuple(slot.role for slot in self.slots)
        if len(set(roles)) != len(roles):
            raise RelationSchemaError("relation schema 不得重复声明 Role")
        if not isinstance(self.same_kind_constraints, tuple):
            raise TypeError("same_kind_constraints 必须是 tuple")
        if any(not isinstance(item, SameKindConstraint)
               for item in self.same_kind_constraints):
            raise TypeError(
                "same_kind_constraints 只能包含 SameKindConstraint")
        role_set = frozenset(roles)
        for constraint in self.same_kind_constraints:
            if not set(constraint.roles).issubset(role_set):
                raise RelationSchemaError(
                    "same-kind constraint 引用了 schema 外的 Role")

    def slot(self, role: ObjectIdentity) -> RelationSlotSchema | None:
        """按完整 Role 身份查询 slot，不按名称、局部键或位置降级。"""
        return next((slot for slot in self.slots if slot.role == role), None)

    def validate_bindings(
            self, bindings: tuple[AtomicRoleBinding, ...],
            ) -> tuple[AtomicRoleBinding, ...]:
        """校验角色、filler 类型、基数和跨角色同型约束。"""
        if not isinstance(bindings, tuple):
            raise TypeError("bindings 必须是 AtomicRoleBinding tuple")
        if any(not isinstance(binding, AtomicRoleBinding)
               for binding in bindings):
            raise TypeError("bindings 只能包含 AtomicRoleBinding")

        by_role: dict[ObjectIdentity, list[AtomicRoleBinding]] = {
            slot.role: [] for slot in self.slots}
        for binding in bindings:
            slot = self.slot(binding.role)
            if slot is None:
                raise RelationSchemaError("事实包含 schema 未声明的 Role")
            if binding.filler.object_kind not in slot.allowed_object_kinds:
                raise RelationSchemaError("RoleBinding filler 类型不合法")
            by_role[binding.role].append(binding)

        for slot in self.slots:
            count = len(by_role[slot.role])
            if count < slot.min_count:
                raise RelationSchemaError("RoleBinding 数量低于最小基数")
            if slot.max_count is not None and count > slot.max_count:
                raise RelationSchemaError("RoleBinding 数量超过最大基数")

        for constraint in self.same_kind_constraints:
            kinds = {
                binding.filler.object_kind
                for role in constraint.roles
                for binding in by_role[role]
            }
            if len(kinds) > 1:
                raise RelationSchemaError("same-kind constraint 不满足")
        return tuple(sorted(bindings, key=_binding_key))

    def validate_definition(
            self, definition: AtomicPropositionDefinition,
            ) -> AtomicPropositionDefinition:
        """核验原子命题 predicate 与 schema 关系身份完全一致。"""
        if not isinstance(definition, AtomicPropositionDefinition):
            raise TypeError("definition 必须是 AtomicPropositionDefinition")
        if definition.predicate != self.relation:
            raise RelationSchemaError("命题 predicate 与 relation schema 不一致")
        self.validate_bindings(definition.bindings)
        return definition


@dataclass(frozen=True)
class ActiveSupportedRelationFact:
    """经过 H-00 核验的 active supported typed 关系事实。"""

    definition: AtomicPropositionDefinition
    snapshot: HypothesisSnapshot
    candidate_definition: EvidenceCandidateDefinition | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.definition, AtomicPropositionDefinition):
            raise TypeError("definition 必须是 AtomicPropositionDefinition")
        if not isinstance(self.snapshot, HypothesisSnapshot):
            raise TypeError("snapshot 必须是 HypothesisSnapshot")
        if self.candidate_definition is None:
            if self.snapshot.hypothesis.candidate_key != (
                    self.definition.proposition.stable_key()):
                raise RelationSchemaError("事实 Hypothesis 与 Proposition 不一致")
            if self.snapshot.hypothesis.observation != self.definition.source:
                raise RelationSchemaError("事实 Hypothesis 与来源不一致")
        else:
            if not isinstance(
                    self.candidate_definition, EvidenceCandidateDefinition):
                raise TypeError("candidate_definition 类型非法")
            hypothesis = self.snapshot.hypothesis
            candidate = self.candidate_definition
            if (candidate.candidate != self.definition.proposition
                    or hypothesis.candidate_key != candidate.stable_key()
                    or hypothesis.competition_key != candidate.competition_key):
                raise RelationSchemaError(
                    "H-05 候选定义与 Proposition/Hypothesis 不一致")
            if (candidate.candidate.owner != hypothesis.observation.owner
                    or candidate.candidate.versions
                    != hypothesis.observation.versions):
                raise RelationSchemaError(
                    "H-05 aggregate owner/version 与 Proposition 不一致")
        if (self.snapshot.lifecycle != LIFECYCLE_ACTIVE
                or self.snapshot.epistemic_status != EPISTEMIC_SUPPORTED):
            raise RelationSchemaError(
                "关系代数只接受 active supported 事实")

    @classmethod
    def from_knowledge(
            cls, knowledge: PropositionKnowledge,
            ) -> "ActiveSupportedRelationFact":
        """从命题知识投影构造关系事实，未知、冲突或退出状态直接失败。"""
        if not isinstance(knowledge, PropositionKnowledge):
            raise TypeError("knowledge 必须是 PropositionKnowledge")
        return cls(knowledge.definition, knowledge.snapshot)


def active_supported_relation_facts(
        knowledge: tuple[PropositionKnowledge, ...],
        ) -> tuple[ActiveSupportedRelationFact, ...]:
    """只选择 active supported 输入，未知、冲突和历史状态均不进入闭包。"""
    facts: list[ActiveSupportedRelationFact] = []
    for item in knowledge:
        if not isinstance(item, PropositionKnowledge):
            raise TypeError("knowledge 只能包含 PropositionKnowledge")
        if (item.snapshot.lifecycle == LIFECYCLE_ACTIVE
                and item.snapshot.epistemic_status == EPISTEMIC_SUPPORTED):
            facts.append(ActiveSupportedRelationFact.from_knowledge(item))
    return tuple(sorted(
        facts, key=lambda item: item.definition.proposition.stable_key()))


@dataclass(frozen=True)
class RelationPremise:
    """派生候选引用的完整 Proposition、Hypothesis 和支持 Evidence。"""

    proposition: ObjectIdentity
    hypothesis: HypothesisKey
    support_evidence_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        _require_identity_kind(
            self.proposition, OBJECT_PROPOSITION, label="premise proposition")
        if not isinstance(self.hypothesis, HypothesisKey):
            raise TypeError("premise hypothesis 必须是 HypothesisKey")
        if self.hypothesis.candidate_key != self.proposition.stable_key():
            try:
                candidate = EvidenceCandidateDefinition.from_stable_key(
                    self.hypothesis.candidate_key)
            except (TypeError, ValueError) as exc:
                raise RelationSchemaError(
                    "premise Hypothesis 与 Proposition 不一致") from exc
            if (candidate.candidate != self.proposition
                    or candidate.competition_key
                    != self.hypothesis.competition_key
                    or candidate.candidate.owner
                    != self.hypothesis.observation.owner
                    or candidate.candidate.versions
                    != self.hypothesis.observation.versions):
                raise RelationSchemaError(
                    "premise H-05 候选与 Proposition 不一致")
        assert_int(
            *self.support_evidence_ids,
            _where="RelationPremise.support_evidence_ids")
        if (not self.support_evidence_ids
                or any(type(item) is not int or item <= 0
                       for item in self.support_evidence_ids)):
            raise RelationSchemaError("premise 必须保留正整数支持 Evidence ID")

    @classmethod
    def from_fact(cls, fact: ActiveSupportedRelationFact) -> "RelationPremise":
        """从 active supported 事实复制完整审计引用，不复制 Evidence 内容。"""
        return cls(
            fact.definition.proposition,
            fact.snapshot.hypothesis,
            fact.snapshot.support_evidence_ids,
        )

    def stable_key(self) -> tuple[int, ...]:
        """返回不依赖哈希截断的完整派生前提键。"""
        proposition = self.proposition.stable_key()
        hypothesis = self.hypothesis.stable_key()
        return (
            len(proposition), *proposition,
            len(hypothesis), *hypothesis,
            len(self.support_evidence_ids), *self.support_evidence_ids,
        )


@dataclass(frozen=True)
class DerivedRelationCandidate:
    """尚未获得独立 Evidence 的关系代数派生候选。"""

    relation: ObjectIdentity
    bindings: tuple[AtomicRoleBinding, ...]
    rule: ObjectIdentity
    schemas: tuple[ObjectIdentity, ...]
    premises: tuple[RelationPremise, ...]

    def __post_init__(self) -> None:
        _require_identity_kind(self.relation, OBJECT_CONCEPT, label="result relation")
        _require_identity_kind(
            self.rule, OBJECT_MINIMAL_INSTRUCTION, label="derivation rule")
        if not isinstance(self.bindings, tuple) or not self.bindings:
            raise RelationSchemaError("派生候选必须含 RoleBinding")
        if any(not isinstance(item, AtomicRoleBinding)
               for item in self.bindings):
            raise TypeError("派生 bindings 类型非法")
        if not isinstance(self.schemas, tuple) or not self.schemas:
            raise RelationSchemaError("派生候选必须保存 schema 身份")
        for schema in self.schemas:
            _require_identity_kind(
                schema, OBJECT_STRUCTURE_CONCEPT, label="derivation schema")
        if not isinstance(self.premises, tuple) or not self.premises:
            raise RelationSchemaError("派生候选必须保存 active supported 前提")
        if any(not isinstance(item, RelationPremise)
               for item in self.premises):
            raise TypeError("派生 premises 类型非法")
        object.__setattr__(
            self, "bindings", tuple(sorted(self.bindings, key=_binding_key)))

    def stable_key(self) -> tuple[int, ...]:
        """展开规则、schema、完整前提和结果角色，不把 hash 当身份。"""
        relation = self.relation.stable_key()
        rule = self.rule.stable_key()
        schema_keys = tuple(schema.stable_key() for schema in self.schemas)
        binding_keys = tuple(_binding_key(binding) for binding in self.bindings)
        premise_keys = tuple(premise.stable_key() for premise in self.premises)
        result: list[int] = [len(relation), *relation, len(rule), *rule]
        result.append(len(schema_keys))
        for key in schema_keys:
            result.extend((len(key), *key))
        result.append(len(binding_keys))
        for key in binding_keys:
            result.extend((len(key), *key))
        result.append(len(premise_keys))
        for key in premise_keys:
            result.extend((len(key), *key))
        return tuple(result)


def _binding_key(binding: AtomicRoleBinding) -> tuple[int, ...]:
    """按完整 Role、filler 和 ordinal 展开无 Proposition 的角色赋值键。"""
    role = binding.role.stable_key()
    filler = binding.filler.stable_key()
    return (
        len(role), *role,
        len(filler), *filler,
        binding.ordinal,
    )


def _validate_rule_identity(rule: ObjectIdentity) -> None:
    """要求执行规则使用符号域最小指令身份，而非名称或宿主枚举。"""
    _require_identity_kind(
        rule, OBJECT_MINIMAL_INSTRUCTION, label="relation rule")


def _validate_relation_role_pair(
        relation: ObjectIdentity, left_role: ObjectIdentity,
        right_role: ObjectIdentity,
        ) -> None:
    """核验二元规则绑定的完整 relation 和两个不同 Role 身份。"""
    _require_identity_kind(relation, OBJECT_CONCEPT, label="rule relation")
    _require_identity_kind(left_role, OBJECT_ROLE, label="left role")
    _require_identity_kind(right_role, OBJECT_ROLE, label="right role")
    if left_role == right_role:
        raise RelationSchemaError("二元规则的左右 Role 必须不同")


@dataclass(frozen=True)
class TransitiveRule:
    """注入 relation 与端点 Role 的传递规则。"""

    rule: ObjectIdentity
    relation: ObjectIdentity
    left_role: ObjectIdentity
    right_role: ObjectIdentity

    def __post_init__(self) -> None:
        _validate_rule_identity(self.rule)
        _validate_relation_role_pair(
            self.relation, self.left_role, self.right_role)


@dataclass(frozen=True)
class SymmetricRule:
    """注入 relation 与端点 Role 的对称规则。"""

    rule: ObjectIdentity
    relation: ObjectIdentity
    left_role: ObjectIdentity
    right_role: ObjectIdentity

    def __post_init__(self) -> None:
        _validate_rule_identity(self.rule)
        _validate_relation_role_pair(
            self.relation, self.left_role, self.right_role)


@dataclass(frozen=True)
class InverseRule:
    """把一个关系的端点映射到另一个注入关系的端点。"""

    rule: ObjectIdentity
    premise_relation: ObjectIdentity
    premise_left_role: ObjectIdentity
    premise_right_role: ObjectIdentity
    result_relation: ObjectIdentity
    result_left_role: ObjectIdentity
    result_right_role: ObjectIdentity

    def __post_init__(self) -> None:
        _validate_rule_identity(self.rule)
        _validate_relation_role_pair(
            self.premise_relation,
            self.premise_left_role,
            self.premise_right_role,
        )
        _validate_relation_role_pair(
            self.result_relation,
            self.result_left_role,
            self.result_right_role,
        )


@dataclass(frozen=True)
class CompositionRule:
    """把两个注入关系按显式 join Role 复合到结果关系。"""

    rule: ObjectIdentity
    first_relation: ObjectIdentity
    first_input_role: ObjectIdentity
    first_join_role: ObjectIdentity
    second_relation: ObjectIdentity
    second_join_role: ObjectIdentity
    second_output_role: ObjectIdentity
    result_relation: ObjectIdentity
    result_input_role: ObjectIdentity
    result_output_role: ObjectIdentity

    def __post_init__(self) -> None:
        _validate_rule_identity(self.rule)
        _validate_relation_role_pair(
            self.first_relation,
            self.first_input_role,
            self.first_join_role,
        )
        _validate_relation_role_pair(
            self.second_relation,
            self.second_join_role,
            self.second_output_role,
        )
        _validate_relation_role_pair(
            self.result_relation,
            self.result_input_role,
            self.result_output_role,
        )


@dataclass(frozen=True)
class ReflexiveRule:
    """从显式 seed 关系提取有限论域对象并形成自反候选。"""

    rule: ObjectIdentity
    seed_relation: ObjectIdentity
    seed_role: ObjectIdentity
    result_relation: ObjectIdentity
    result_left_role: ObjectIdentity
    result_right_role: ObjectIdentity

    def __post_init__(self) -> None:
        _validate_rule_identity(self.rule)
        _require_identity_kind(
            self.seed_relation, OBJECT_CONCEPT, label="seed relation")
        _require_identity_kind(self.seed_role, OBJECT_ROLE, label="seed role")
        _validate_relation_role_pair(
            self.result_relation,
            self.result_left_role,
            self.result_right_role,
        )


@dataclass(frozen=True)
class IrreflexiveRule:
    """检测显式关系事实中的自环，不直接把违规候选升级成反证。"""

    rule: ObjectIdentity
    relation: ObjectIdentity
    left_role: ObjectIdentity
    right_role: ObjectIdentity

    def __post_init__(self) -> None:
        _validate_rule_identity(self.rule)
        _validate_relation_role_pair(
            self.relation, self.left_role, self.right_role)


@dataclass(frozen=True)
class IrreflexiveViolation:
    """反自反规则发现的待审计冲突候选，不等于 definitive refute。"""

    rule: ObjectIdentity
    schema: ObjectIdentity
    premise: RelationPremise
    filler: ObjectIdentity

    def __post_init__(self) -> None:
        _validate_rule_identity(self.rule)
        _require_identity_kind(
            self.schema, OBJECT_STRUCTURE_CONCEPT,
            label="irreflexive schema")
        if not isinstance(self.premise, RelationPremise):
            raise TypeError("irreflexive premise 必须是 RelationPremise")
        if not isinstance(self.filler, ObjectIdentity):
            raise TypeError("irreflexive filler 必须是 ObjectIdentity")

    def stable_key(self) -> tuple[int, ...]:
        """展开规则、schema、前提和自环 filler 的完整审计身份。"""
        rule = self.rule.stable_key()
        schema = self.schema.stable_key()
        premise = self.premise.stable_key()
        filler = self.filler.stable_key()
        return (
            len(rule), *rule,
            len(schema), *schema,
            len(premise), *premise,
            len(filler), *filler,
        )


RelationRule = (
    TransitiveRule | SymmetricRule | InverseRule | CompositionRule
    | ReflexiveRule
)


__all__ = [
    "ActiveSupportedRelationFact",
    "CompositionRule",
    "DerivedRelationCandidate",
    "InverseRule",
    "IrreflexiveRule",
    "IrreflexiveViolation",
    "ReflexiveRule",
    "RelationPremise",
    "RelationRule",
    "RelationSchema",
    "RelationSchemaError",
    "RelationSlotSchema",
    "SameKindConstraint",
    "SymmetricRule",
    "TransitiveRule",
    "active_supported_relation_facts",
]
