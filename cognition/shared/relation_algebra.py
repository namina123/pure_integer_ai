"""S-01 注入式 typed relation 代数执行器。

执行器只识别显式规则对象的最小运算形状，不根据 relation surface、枚举值或
二元外形猜测规律。所有派生均追溯到 active supported 原子命题；输出仍是候选，
不会自动获得 H-00 支持状态。
"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.identity import ObjectIdentity
from pure_integer_ai.cognition.shared.semantic_object import (
    AtomicPropositionDefinition,
    AtomicRoleBinding,
    PropositionKnowledge,
)
from pure_integer_ai.cognition.shared.typed_relation import (
    ActiveSupportedRelationFact,
    CompositionRule,
    DerivedRelationCandidate,
    InverseRule,
    IrreflexiveRule,
    IrreflexiveViolation,
    ReflexiveRule,
    RelationPremise,
    RelationRule,
    RelationSchema,
    RelationSchemaError,
    SymmetricRule,
    TransitiveRule,
    active_supported_relation_facts,
)


class RelationAlgebra:
    """在一组完整 relation schema 上执行显式注入的最小代数规则。"""

    def __init__(self, schemas: tuple[RelationSchema, ...]) -> None:
        if not isinstance(schemas, tuple) or not schemas:
            raise RelationSchemaError("RelationAlgebra 至少需要一个 schema")
        if any(not isinstance(schema, RelationSchema) for schema in schemas):
            raise TypeError("schemas 只能包含 RelationSchema")
        relations = tuple(schema.relation for schema in schemas)
        schema_identities = tuple(schema.schema for schema in schemas)
        if len(set(relations)) != len(relations):
            raise RelationSchemaError("同一 relation 只能有一个 active schema")
        if len(set(schema_identities)) != len(schema_identities):
            raise RelationSchemaError("relation schema 身份不得重复")
        self._schemas = {schema.relation: schema for schema in schemas}

    def validate_definition(
            self, definition: AtomicPropositionDefinition,
            ) -> AtomicPropositionDefinition:
        """用 predicate 的完整 relation 身份选择 schema 并执行严格校验。"""
        schema = self._schemas.get(definition.predicate)
        if schema is None:
            raise RelationSchemaError("命题 relation 没有 active schema")
        return schema.validate_definition(definition)

    def derive_candidates(
            self, knowledge: tuple[PropositionKnowledge, ...],
            rules: tuple[RelationRule, ...],
            ) -> tuple[DerivedRelationCandidate, ...]:
        """从 active supported 且 schema 合法的事实生成可审计派生候选。"""
        if not isinstance(knowledge, tuple):
            raise TypeError("knowledge 必须是 PropositionKnowledge tuple")
        if not isinstance(rules, tuple):
            raise TypeError("rules 必须是 RelationRule tuple")
        return self.derive_active_candidates(
            active_supported_relation_facts(knowledge), rules)

    def derive_active_candidates(
            self, facts: tuple[ActiveSupportedRelationFact, ...],
            rules: tuple[RelationRule, ...],
            ) -> tuple[DerivedRelationCandidate, ...]:
        """消费已经核验的 active facts，供 H-05 关系闭环复用同一代数。"""
        if not isinstance(facts, tuple):
            raise TypeError("facts 必须是 ActiveSupportedRelationFact tuple")
        if any(not isinstance(item, ActiveSupportedRelationFact)
               for item in facts):
            raise TypeError("facts 只能包含 ActiveSupportedRelationFact")
        if not isinstance(rules, tuple):
            raise TypeError("rules 必须是 RelationRule tuple")
        facts = self._validated_facts(facts)
        base_content = frozenset(_fact_content_key(fact) for fact in facts)
        candidates: dict[tuple[int, ...], DerivedRelationCandidate] = {}
        for rule in sorted(rules, key=_rule_key):
            if isinstance(rule, TransitiveRule):
                produced = self._derive_transitive(rule, facts, base_content)
            elif isinstance(rule, SymmetricRule):
                produced = self._derive_symmetric(rule, facts, base_content)
            elif isinstance(rule, InverseRule):
                produced = self._derive_inverse(rule, facts, base_content)
            elif isinstance(rule, CompositionRule):
                produced = self._derive_composition(rule, facts, base_content)
            elif isinstance(rule, ReflexiveRule):
                produced = self._derive_reflexive(rule, facts, base_content)
            else:
                raise TypeError("rules 包含未注册的关系规则类型")
            for candidate in produced:
                candidates[candidate.stable_key()] = candidate
        return tuple(candidates[key] for key in sorted(candidates))

    def irreflexive_violations(
            self, knowledge: tuple[PropositionKnowledge, ...],
            rules: tuple[IrreflexiveRule, ...],
            ) -> tuple[IrreflexiveViolation, ...]:
        """报告 active supported 自环，不把结构违规直接写成 refute Evidence。"""
        if not isinstance(knowledge, tuple) or not isinstance(rules, tuple):
            raise TypeError("knowledge 和 rules 必须是 tuple")
        return self.irreflexive_active_violations(
            active_supported_relation_facts(knowledge), rules)

    def irreflexive_active_violations(
            self, facts: tuple[ActiveSupportedRelationFact, ...],
            rules: tuple[IrreflexiveRule, ...],
            ) -> tuple[IrreflexiveViolation, ...]:
        """在已核验 active facts 上执行反自反审计，不改写 Evidence。"""
        if not isinstance(facts, tuple) or not isinstance(rules, tuple):
            raise TypeError("facts 和 rules 必须是 tuple")
        if any(not isinstance(item, ActiveSupportedRelationFact)
               for item in facts):
            raise TypeError("facts 只能包含 ActiveSupportedRelationFact")
        facts = self._validated_facts(facts)
        violations: list[IrreflexiveViolation] = []
        for rule in sorted(rules, key=_rule_key):
            if not isinstance(rule, IrreflexiveRule):
                raise TypeError("rules 只能包含 IrreflexiveRule")
            schema = self._schemas.get(rule.relation)
            if schema is None:
                continue
            for fact in self._facts_for(facts, rule.relation):
                left = _single_filler(fact, rule.left_role)
                right = _single_filler(fact, rule.right_role)
                if left is None or right is None or left != right:
                    continue
                violations.append(IrreflexiveViolation(
                    rule.rule,
                    schema.schema,
                    RelationPremise.from_fact(fact),
                    left,
                ))
        return tuple(sorted(
            violations,
            key=lambda item: item.stable_key(),
        ))

    def _validated_facts(
            self, facts: tuple[ActiveSupportedRelationFact, ...],
            ) -> tuple[ActiveSupportedRelationFact, ...]:
        """过滤无 schema 或 schema 不匹配事实，使错误形状产生零派生。"""
        accepted: list[ActiveSupportedRelationFact] = []
        for fact in facts:
            schema = self._schemas.get(fact.definition.predicate)
            if schema is None:
                continue
            try:
                schema.validate_definition(fact.definition)
            except RelationSchemaError:
                continue
            accepted.append(fact)
        return tuple(sorted(accepted, key=_fact_key))

    @staticmethod
    def _facts_for(
            facts: tuple[ActiveSupportedRelationFact, ...],
            relation: ObjectIdentity,
            ) -> tuple[ActiveSupportedRelationFact, ...]:
        """按完整 relation identity 选择事实，不对相同 shape 做兼容匹配。"""
        return tuple(
            fact for fact in facts
            if fact.definition.predicate == relation
        )

    def _candidate(
            self, *, relation: ObjectIdentity,
            bindings: tuple[AtomicRoleBinding, ...],
            rule: ObjectIdentity,
            premise_relations: tuple[ObjectIdentity, ...],
            premise_facts: tuple[ActiveSupportedRelationFact, ...],
            base_content: frozenset[tuple[int, ...]],
            ) -> DerivedRelationCandidate | None:
        """校验结果 schema、去除已有事实，并保留规则和全部支持前提。"""
        result_schema = self._schemas.get(relation)
        premise_schemas = tuple(
            self._schemas.get(item) for item in premise_relations)
        if result_schema is None or any(item is None for item in premise_schemas):
            return None
        try:
            canonical_bindings = result_schema.validate_bindings(bindings)
        except RelationSchemaError:
            return None
        content = _content_key(relation, canonical_bindings)
        if content in base_content:
            return None
        schemas = tuple(item.schema for item in premise_schemas)
        if result_schema.relation not in premise_relations:
            schemas = (*schemas, result_schema.schema)
        premises = tuple(RelationPremise.from_fact(fact)
                         for fact in premise_facts)
        return DerivedRelationCandidate(
            relation,
            canonical_bindings,
            rule,
            schemas,
            premises,
        )

    def _derive_symmetric(
            self, rule: SymmetricRule,
            facts: tuple[ActiveSupportedRelationFact, ...],
            base_content: frozenset[tuple[int, ...]],
            ) -> tuple[DerivedRelationCandidate, ...]:
        """对匹配关系交换显式左右 Role，不复制其他关系的规律。"""
        result: list[DerivedRelationCandidate] = []
        for fact in self._facts_for(facts, rule.relation):
            left = _single_filler(fact, rule.left_role)
            right = _single_filler(fact, rule.right_role)
            if left is None or right is None:
                continue
            candidate = self._candidate(
                relation=rule.relation,
                bindings=(
                    AtomicRoleBinding(rule.left_role, right, 0),
                    AtomicRoleBinding(rule.right_role, left, 0),
                ),
                rule=rule.rule,
                premise_relations=(rule.relation,),
                premise_facts=(fact,),
                base_content=base_content,
            )
            if candidate is not None:
                result.append(candidate)
        return tuple(result)

    def _derive_inverse(
            self, rule: InverseRule,
            facts: tuple[ActiveSupportedRelationFact, ...],
            base_content: frozenset[tuple[int, ...]],
            ) -> tuple[DerivedRelationCandidate, ...]:
        """按规则声明把 premise 端点反向映射到结果 relation。"""
        result: list[DerivedRelationCandidate] = []
        for fact in self._facts_for(facts, rule.premise_relation):
            left = _single_filler(fact, rule.premise_left_role)
            right = _single_filler(fact, rule.premise_right_role)
            if left is None or right is None:
                continue
            candidate = self._candidate(
                relation=rule.result_relation,
                bindings=(
                    AtomicRoleBinding(rule.result_left_role, right, 0),
                    AtomicRoleBinding(rule.result_right_role, left, 0),
                ),
                rule=rule.rule,
                premise_relations=(
                    rule.premise_relation, rule.result_relation),
                premise_facts=(fact,),
                base_content=base_content,
            )
            if candidate is not None:
                result.append(candidate)
        return tuple(result)

    def _derive_composition(
            self, rule: CompositionRule,
            facts: tuple[ActiveSupportedRelationFact, ...],
            base_content: frozenset[tuple[int, ...]],
            ) -> tuple[DerivedRelationCandidate, ...]:
        """只在两个显式 join Role 的完整 filler identity 相等时复合。"""
        first_facts = self._facts_for(facts, rule.first_relation)
        second_facts = self._facts_for(facts, rule.second_relation)
        result: list[DerivedRelationCandidate] = []
        for first in first_facts:
            start = _single_filler(first, rule.first_input_role)
            join = _single_filler(first, rule.first_join_role)
            if start is None or join is None:
                continue
            for second in second_facts:
                second_join = _single_filler(second, rule.second_join_role)
                end = _single_filler(second, rule.second_output_role)
                if second_join is None or end is None or join != second_join:
                    continue
                candidate = self._candidate(
                    relation=rule.result_relation,
                    bindings=(
                        AtomicRoleBinding(rule.result_input_role, start, 0),
                        AtomicRoleBinding(rule.result_output_role, end, 0),
                    ),
                    rule=rule.rule,
                    premise_relations=(
                        rule.first_relation,
                        rule.second_relation,
                        rule.result_relation,
                    ),
                    premise_facts=(first, second),
                    base_content=base_content,
                )
                if candidate is not None:
                    result.append(candidate)
        return tuple(result)

    def _derive_reflexive(
            self, rule: ReflexiveRule,
            facts: tuple[ActiveSupportedRelationFact, ...],
            base_content: frozenset[tuple[int, ...]],
            ) -> tuple[DerivedRelationCandidate, ...]:
        """从显式 seed 事实限定有限论域，不扫描全图或凭空制造对象。"""
        result: list[DerivedRelationCandidate] = []
        for fact in self._facts_for(facts, rule.seed_relation):
            filler = _single_filler(fact, rule.seed_role)
            if filler is None:
                continue
            candidate = self._candidate(
                relation=rule.result_relation,
                bindings=(
                    AtomicRoleBinding(rule.result_left_role, filler, 0),
                    AtomicRoleBinding(rule.result_right_role, filler, 0),
                ),
                rule=rule.rule,
                premise_relations=(
                    rule.seed_relation, rule.result_relation),
                premise_facts=(fact,),
                base_content=base_content,
            )
            if candidate is not None:
                result.append(candidate)
        return tuple(result)

    def _derive_transitive(
            self, rule: TransitiveRule,
            facts: tuple[ActiveSupportedRelationFact, ...],
            base_content: frozenset[tuple[int, ...]],
            ) -> tuple[DerivedRelationCandidate, ...]:
        """计算有限端点闭包，路径前提始终展开为 active supported 基础事实。"""
        relation_facts = self._facts_for(facts, rule.relation)
        direct: dict[
            tuple[ObjectIdentity, ObjectIdentity],
            tuple[ActiveSupportedRelationFact, ...],
        ] = {}
        for fact in relation_facts:
            left = _single_filler(fact, rule.left_role)
            right = _single_filler(fact, rule.right_role)
            if left is None or right is None:
                continue
            key = left, right
            path = (fact,)
            if key not in direct or _path_key(path) < _path_key(direct[key]):
                direct[key] = path

        paths = dict(direct)
        changed = True
        while changed:
            changed = False
            snapshot = tuple(sorted(paths.items(), key=_path_item_key))
            for (start, middle), first_path in snapshot:
                for (second_middle, end), second_path in snapshot:
                    if middle != second_middle:
                        continue
                    combined = (*first_path, *second_path)
                    key = start, end
                    existing = paths.get(key)
                    if (existing is None
                            or _path_key(combined) < _path_key(existing)):
                        paths[key] = combined
                        changed = True

        result: list[DerivedRelationCandidate] = []
        for (start, end), path in sorted(paths.items(), key=_path_item_key):
            if (start, end) in direct or len(path) < 2:
                continue
            candidate = self._candidate(
                relation=rule.relation,
                bindings=(
                    AtomicRoleBinding(rule.left_role, start, 0),
                    AtomicRoleBinding(rule.right_role, end, 0),
                ),
                rule=rule.rule,
                premise_relations=(rule.relation,),
                premise_facts=path,
                base_content=base_content,
            )
            if candidate is not None:
                result.append(candidate)
        return tuple(result)


def _single_filler(
        fact: ActiveSupportedRelationFact, role: ObjectIdentity,
        ) -> ObjectIdentity | None:
    """只在事实恰有一个指定 RoleBinding 时返回 filler。"""
    matches = tuple(
        binding.filler for binding in fact.definition.bindings
        if binding.role == role
    )
    return matches[0] if len(matches) == 1 else None


def _fact_key(fact: ActiveSupportedRelationFact) -> tuple[int, ...]:
    """按完整 Proposition 和 Hypothesis 身份排序 active supported 事实。"""
    proposition = fact.definition.proposition.stable_key()
    hypothesis = fact.snapshot.hypothesis.stable_key()
    return len(proposition), *proposition, len(hypothesis), *hypothesis


def _binding_key(binding: AtomicRoleBinding) -> tuple[int, ...]:
    """展开结果角色赋值，供内容去重和确定性排序。"""
    role = binding.role.stable_key()
    filler = binding.filler.stable_key()
    return len(role), *role, len(filler), *filler, binding.ordinal


def _content_key(
        relation: ObjectIdentity,
        bindings: tuple[AtomicRoleBinding, ...],
        ) -> tuple[int, ...]:
    """返回 relation 加全部角色赋值的无哈希内容身份。"""
    relation_key = relation.stable_key()
    binding_keys = tuple(sorted(_binding_key(item) for item in bindings))
    result: list[int] = [len(relation_key), *relation_key, len(binding_keys)]
    for key in binding_keys:
        result.extend((len(key), *key))
    return tuple(result)


def _fact_content_key(fact: ActiveSupportedRelationFact) -> tuple[int, ...]:
    """投影基础事实的 relation 内容键，不混入 Proposition 来源身份。"""
    return _content_key(
        fact.definition.predicate, fact.definition.bindings)


def _path_key(
        path: tuple[ActiveSupportedRelationFact, ...],
        ) -> tuple[int, tuple[tuple[int, ...], ...]]:
    """优先较短路径，再按完整事实身份确定闭包的唯一审计路径。"""
    return len(path), tuple(_fact_key(fact) for fact in path)


def _path_item_key(item):
    """按完整端点和路径键稳定排序闭包状态。"""
    (left, right), path = item
    return left.stable_key(), right.stable_key(), _path_key(path)


def _rule_key(rule) -> tuple[int, ...]:
    """展开规则对象中的全部一等身份，避免只按 rule hash 或名称排序。"""
    if not isinstance(rule, (
            TransitiveRule, SymmetricRule, InverseRule,
            CompositionRule, ReflexiveRule, IrreflexiveRule)):
        raise TypeError("未知 relation rule")
    identities = tuple(
        value for value in rule.__dict__.values()
        if isinstance(value, ObjectIdentity)
    )
    result: list[int] = []
    for identity in identities:
        key = identity.stable_key()
        result.extend((len(key), *key))
    return tuple(result)


__all__ = ["RelationAlgebra"]
