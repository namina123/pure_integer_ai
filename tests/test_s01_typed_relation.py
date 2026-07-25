"""S-01 开放 relation schema、typed 关系代数和派生来源测试。"""
from __future__ import annotations

import pytest

from pure_integer_ai.cognition.shared.graph_ontology import (
    relation_concept_identity,
)
from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EvidenceRecord,
    HypothesisLedger,
)
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    OBJECT_CONCEPT,
    OBJECT_ENTITY,
    OBJECT_EVENT,
    OBJECT_SET_EXPR,
    SourceRef,
    VersionBundle,
    concept_identity,
    minimal_instruction_identity,
    occurrence_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.relation_algebra import RelationAlgebra
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.cognition.shared.semantic_object import (
    AtomicPropositionDefinition,
    AtomicRoleBinding,
    context_scope_identity,
    entity_identity,
    event_identity,
    project_proposition_knowledge,
    proposition_hypothesis_key,
    proposition_identity,
    role_identity,
    set_expr_identity,
)
from pure_integer_ai.cognition.shared.typed_relation import (
    ActiveSupportedRelationFact,
    CompositionRule,
    InverseRule,
    IrreflexiveRule,
    ReflexiveRule,
    RelationSchema,
    RelationSchemaError,
    RelationSlotSchema,
    SameKindConstraint,
    SymmetricRule,
    TransitiveRule,
)


def _source(document_id: int = 1) -> SourceRef:
    """构造 S-01 事实共享的完整测试来源。"""
    return SourceRef(
        81, 83, document_id, GLOBAL_OWNER_SCOPE, VersionBundle())


def _binary_schema(
        key: int, relation, left_role, right_role,
        *, left_kinds=frozenset({OBJECT_SET_EXPR}),
        right_kinds=frozenset({OBJECT_SET_EXPR}),
        constraints=(),
        ) -> RelationSchema:
    """构造完全注入的二元 schema，不依赖 relation 名称。"""
    return RelationSchema(
        structure_concept_identity((1200, key)),
        relation,
        (
            RelationSlotSchema(left_role, left_kinds, 1, 1),
            RelationSlotSchema(right_role, right_kinds, 1, 1),
        ),
        tuple(constraints),
    )


def _definition(
        source: SourceRef, ordinal: int, relation, bindings,
        ) -> AtomicPropositionDefinition:
    """构造一个来源化 typed relation 原子命题。"""
    return AtomicPropositionDefinition(
        proposition_identity(source, (1300, ordinal)),
        relation,
        occurrence_identity(
            source, start=ordinal * 2, end=ordinal * 2 + 1,
            ordinal=ordinal),
        context_scope_identity(source, (1301, 1)),
        tuple(bindings),
    )


def _knowledge(definition, evidence_id: int, *, state: str = "supported"):
    """通过 H-00 ledger 形成 supported、unknown 或 conflicted 命题知识。"""
    hypothesis = proposition_hypothesis_key(
        definition.proposition,
        hypothesis_kind=(1400, 1),
        competition_key=(1400, evidence_id),
        scope=document_scope(definition.source),
    )
    ledger = HypothesisLedger()
    ledger.register(hypothesis)
    if state in {"supported", "conflicted"}:
        ledger.append_evidence(EvidenceRecord(
            evidence_id,
            hypothesis,
            EVIDENCE_SUPPORT,
            (1401, 1),
            definition.source,
            evidence_id,
        ))
    if state == "conflicted":
        ledger.append_evidence(EvidenceRecord(
            evidence_id + 1000,
            hypothesis,
            EVIDENCE_REFUTE,
            (1401, 2),
            definition.source,
            evidence_id + 1000,
        ))
    return project_proposition_knowledge(definition, hypothesis, ledger)


def _fillers(candidate):
    """按完整 Role 身份投影候选 filler，供测试核对端点。"""
    return {binding.role: binding.filler for binding in candidate.bindings}


def test_member_schema_is_directional_and_not_mereology_or_subset():
    """MEMBER 只接受个体到 SetExpr，反向或集合到集合均不合法。"""
    source = _source()
    relation = relation_concept_identity((1500, 1))
    member_role = role_identity((1500, 2))
    set_role = role_identity((1500, 3))
    schema = _binary_schema(
        1, relation, member_role, set_role,
        left_kinds=frozenset({OBJECT_ENTITY}),
        right_kinds=frozenset({OBJECT_SET_EXPR}),
    )
    individual = entity_identity(source, (1500, 4))
    collection = set_expr_identity(source, (1500, 5))
    valid = _definition(source, 1, relation, (
        AtomicRoleBinding(member_role, individual),
        AtomicRoleBinding(set_role, collection),
    ))
    assert schema.validate_definition(valid) is valid

    reversed_fact = _definition(source, 2, relation, (
        AtomicRoleBinding(member_role, collection),
        AtomicRoleBinding(set_role, individual),
    ))
    with pytest.raises(RelationSchemaError, match="filler 类型"):
        schema.validate_definition(reversed_fact)


def test_property_is_open_nary_and_event_relation_has_independent_types():
    """PROPERTY 可注入三元角色，EVENT relation 独立要求 Event 参数。"""
    source = _source()
    subject_role = role_identity((1510, 1))
    attribute_role = role_identity((1510, 2))
    value_role = role_identity((1510, 3))
    property_relation = relation_concept_identity((1510, 4))
    property_schema = RelationSchema(
        structure_concept_identity((1510, 5)),
        property_relation,
        (
            RelationSlotSchema(
                subject_role, frozenset({OBJECT_ENTITY}), 1, 1),
            RelationSlotSchema(
                attribute_role, frozenset({OBJECT_CONCEPT}), 1, 1),
            RelationSlotSchema(
                value_role,
                frozenset({OBJECT_ENTITY, OBJECT_CONCEPT}), 1, 1),
        ),
    )
    property_fact = _definition(source, 3, property_relation, (
        AtomicRoleBinding(subject_role, entity_identity(source, (1510, 6))),
        AtomicRoleBinding(attribute_role, concept_identity((1510, 7))),
        AtomicRoleBinding(value_role, concept_identity((1510, 8))),
    ))
    assert property_schema.validate_definition(property_fact) is property_fact

    cause_role = role_identity((1510, 9))
    effect_role = role_identity((1510, 10))
    event_relation = relation_concept_identity((1510, 11))
    event_schema = _binary_schema(
        2, event_relation, cause_role, effect_role,
        left_kinds=frozenset({OBJECT_EVENT}),
        right_kinds=frozenset({OBJECT_EVENT}),
    )
    event_fact = _definition(source, 4, event_relation, (
        AtomicRoleBinding(cause_role, event_identity(source, (1510, 12))),
        AtomicRoleBinding(effect_role, event_identity(source, (1510, 13))),
    ))
    assert event_schema.validate_definition(event_fact) is event_fact
    with pytest.raises(RelationSchemaError, match="filler 类型"):
        event_schema.validate_definition(_definition(
            source, 5, event_relation, (
                AtomicRoleBinding(
                    cause_role, entity_identity(source, (1510, 14))),
                AtomicRoleBinding(
                    effect_role, event_identity(source, (1510, 15))),
            )))


def test_alias_same_kind_constraint_does_not_merge_object_identities():
    """ALIAS 可要求同型，但两个端点身份始终分别保存。"""
    source = _source()
    relation = relation_concept_identity((1520, 1))
    left_role = role_identity((1520, 2))
    right_role = role_identity((1520, 3))
    constraint = SameKindConstraint(
        structure_concept_identity((1520, 4)),
        (left_role, right_role),
    )
    schema = _binary_schema(
        3, relation, left_role, right_role,
        left_kinds=frozenset({OBJECT_CONCEPT, OBJECT_ENTITY}),
        right_kinds=frozenset({OBJECT_CONCEPT, OBJECT_ENTITY}),
        constraints=(constraint,),
    )
    first = concept_identity((1520, 5))
    second = concept_identity((1520, 6))
    definition = _definition(source, 6, relation, (
        AtomicRoleBinding(left_role, first),
        AtomicRoleBinding(right_role, second),
    ))
    schema.validate_definition(definition)
    assert first != second
    assert set(_fillers(definition).values()) == {first, second}

    with pytest.raises(RelationSchemaError, match="same-kind"):
        schema.validate_definition(_definition(source, 7, relation, (
            AtomicRoleBinding(left_role, first),
            AtomicRoleBinding(
                right_role, entity_identity(source, (1520, 7))),
        )))


def test_schema_rejects_missing_extra_and_over_cardinality_roles():
    """缺 Role、schema 外 Role 和同 Role 超基数都必须失败。"""
    source = _source()
    relation = relation_concept_identity((1525, 1))
    left_role = role_identity((1525, 2))
    right_role = role_identity((1525, 3))
    extra_role = role_identity((1525, 4))
    schema = _binary_schema(15, relation, left_role, right_role)
    first = set_expr_identity(source, (1525, 5))
    second = set_expr_identity(source, (1525, 6))

    with pytest.raises(RelationSchemaError, match="最小基数"):
        schema.validate_definition(_definition(source, 8, relation, (
            AtomicRoleBinding(left_role, first),
        )))
    with pytest.raises(RelationSchemaError, match="未声明"):
        schema.validate_definition(_definition(source, 9, relation, (
            AtomicRoleBinding(left_role, first),
            AtomicRoleBinding(right_role, second),
            AtomicRoleBinding(extra_role, first),
        )))
    with pytest.raises(RelationSchemaError, match="最大基数"):
        schema.validate_definition(_definition(source, 10, relation, (
            AtomicRoleBinding(left_role, first, 0),
            AtomicRoleBinding(left_role, second, 1),
            AtomicRoleBinding(right_role, second),
        )))

    with pytest.raises(RelationSchemaError, match="权威 object contract"):
        RelationSlotSchema(left_role, frozenset({99999}), 1, 1)


def test_transitive_closure_preserves_rule_schemas_and_full_supported_premises():
    """三跳闭包只用 supported 基础事实，并保存规则、schema、命题和 Evidence。"""
    source = _source()
    relation = relation_concept_identity((1530, 1))
    left_role = role_identity((1530, 2))
    right_role = role_identity((1530, 3))
    schema = _binary_schema(4, relation, left_role, right_role)
    algebra = RelationAlgebra((schema,))
    values = tuple(set_expr_identity(source, (1530, item))
                   for item in range(4, 8))
    definitions = tuple(
        _definition(source, 10 + index, relation, (
            AtomicRoleBinding(left_role, values[index]),
            AtomicRoleBinding(right_role, values[index + 1]),
        ))
        for index in range(3)
    )
    knowledge = tuple(
        _knowledge(definition, 10 + index)
        for index, definition in enumerate(definitions)
    )
    rule_identity = minimal_instruction_identity((1530, 20))
    candidates = algebra.derive_candidates(
        knowledge,
        (TransitiveRule(
            rule_identity, relation, left_role, right_role),),
    )
    endpoints = {
        (_fillers(item)[left_role], _fillers(item)[right_role]): item
        for item in candidates
    }
    assert set(endpoints) == {
        (values[0], values[2]),
        (values[0], values[3]),
        (values[1], values[3]),
    }
    longest = endpoints[(values[0], values[3])]
    assert longest.rule == rule_identity
    assert longest.schemas == (schema.schema,)
    assert tuple(item.proposition for item in longest.premises) == tuple(
        definition.proposition for definition in definitions)
    assert tuple(item.support_evidence_ids for item in longest.premises) == (
        (10,), (11,), (12,))


def test_unknown_and_conflicted_facts_never_enter_closure():
    """unknown/conflicted 命题不能因图中存在而被关系代数选中。"""
    source = _source()
    relation = relation_concept_identity((1540, 1))
    left_role = role_identity((1540, 2))
    right_role = role_identity((1540, 3))
    schema = _binary_schema(5, relation, left_role, right_role)
    values = tuple(set_expr_identity(source, (1540, item))
                   for item in range(4, 7))
    first = _definition(source, 20, relation, (
        AtomicRoleBinding(left_role, values[0]),
        AtomicRoleBinding(right_role, values[1]),
    ))
    second = _definition(source, 21, relation, (
        AtomicRoleBinding(left_role, values[1]),
        AtomicRoleBinding(right_role, values[2]),
    ))
    unknown = _knowledge(first, 20, state="unknown")
    conflicted = _knowledge(second, 21, state="conflicted")
    with pytest.raises(RelationSchemaError, match="active supported"):
        ActiveSupportedRelationFact.from_knowledge(conflicted)
    assert RelationAlgebra((schema,)).derive_candidates(
        (unknown, conflicted),
        (TransitiveRule(
            minimal_instruction_identity((1540, 8)),
            relation, left_role, right_role),),
    ) == ()


def test_member_and_same_shape_relation_do_not_inherit_subset_rule():
    """共享二元 shape 不会让 MEMBER 或其他 relation 自动获得传递性。"""
    source = _source()
    subset = relation_concept_identity((1550, 1))
    member = relation_concept_identity((1550, 2))
    left_role = role_identity((1550, 3))
    right_role = role_identity((1550, 4))
    subset_schema = _binary_schema(6, subset, left_role, right_role)
    member_schema = _binary_schema(7, member, left_role, right_role)
    values = tuple(set_expr_identity(source, (1550, item))
                   for item in range(5, 8))
    member_knowledge = tuple(
        _knowledge(_definition(source, 30 + index, member, (
            AtomicRoleBinding(left_role, values[index]),
            AtomicRoleBinding(right_role, values[index + 1]),
        )), 30 + index)
        for index in range(2)
    )
    algebra = RelationAlgebra((subset_schema, member_schema))
    assert algebra.derive_candidates(
        member_knowledge,
        (TransitiveRule(
            minimal_instruction_identity((1550, 9)),
            subset, left_role, right_role),),
    ) == ()


def test_inverse_and_composition_require_exact_relation_and_role_identities():
    """inverse/compose 只按完整 relation 与 Role 接线，错配时零派生。"""
    source = _source()
    relation_a = relation_concept_identity((1560, 1))
    relation_b = relation_concept_identity((1560, 2))
    relation_c = relation_concept_identity((1560, 3))
    a_left, a_right = role_identity((1560, 4)), role_identity((1560, 5))
    b_left, b_right = role_identity((1560, 6)), role_identity((1560, 7))
    c_left, c_right = role_identity((1560, 8)), role_identity((1560, 9))
    schemas = (
        _binary_schema(8, relation_a, a_left, a_right),
        _binary_schema(9, relation_b, b_left, b_right),
        _binary_schema(10, relation_c, c_left, c_right),
    )
    algebra = RelationAlgebra(schemas)
    first, join, end = tuple(
        set_expr_identity(source, (1560, item)) for item in range(10, 13))
    fact_a = _knowledge(_definition(source, 40, relation_a, (
        AtomicRoleBinding(a_left, first),
        AtomicRoleBinding(a_right, join),
    )), 40)
    fact_b = _knowledge(_definition(source, 41, relation_b, (
        AtomicRoleBinding(b_left, join),
        AtomicRoleBinding(b_right, end),
    )), 41)

    inverse = InverseRule(
        minimal_instruction_identity((1560, 20)),
        relation_a, a_left, a_right,
        relation_b, b_left, b_right,
    )
    inverse_candidates = algebra.derive_candidates((fact_a,), (inverse,))
    assert len(inverse_candidates) == 1
    assert _fillers(inverse_candidates[0]) == {b_left: join, b_right: first}

    compose = CompositionRule(
        minimal_instruction_identity((1560, 21)),
        relation_a, a_left, a_right,
        relation_b, b_left, b_right,
        relation_c, c_left, c_right,
    )
    composed = algebra.derive_candidates((fact_a, fact_b), (compose,))
    assert len(composed) == 1
    assert _fillers(composed[0]) == {c_left: first, c_right: end}

    wrong_role = role_identity((1560, 99))
    mismatched = CompositionRule(
        minimal_instruction_identity((1560, 22)),
        relation_a, wrong_role, a_right,
        relation_b, b_left, b_right,
        relation_c, c_left, c_right,
    )
    assert algebra.derive_candidates(
        (fact_a, fact_b), (mismatched,)) == ()


def test_relation_identity_replacement_requires_replaced_schema():
    """只换 relation identity 而不换 schema 时不能沿旧 shape 执行。"""
    source = _source()
    old_relation = relation_concept_identity((1570, 1))
    new_relation = relation_concept_identity((1570, 2))
    left_role = role_identity((1570, 3))
    right_role = role_identity((1570, 4))
    old_schema = _binary_schema(11, old_relation, left_role, right_role)
    values = tuple(set_expr_identity(source, (1570, item))
                   for item in range(5, 8))
    knowledge = tuple(
        _knowledge(_definition(source, 50 + index, new_relation, (
            AtomicRoleBinding(left_role, values[index]),
            AtomicRoleBinding(right_role, values[index + 1]),
        )), 50 + index)
        for index in range(2)
    )
    rule = TransitiveRule(
        minimal_instruction_identity((1570, 9)),
        new_relation, left_role, right_role)
    assert RelationAlgebra((old_schema,)).derive_candidates(
        knowledge, (rule,)) == ()

    new_schema = _binary_schema(12, new_relation, left_role, right_role)
    assert len(RelationAlgebra((new_schema,)).derive_candidates(
        knowledge, (rule,))) == 1


def test_symmetric_reflexive_and_irreflexive_rules_are_explicit():
    """对称、自反和反自反均需独立规则身份，且自反论域来自 supported seed。"""
    source = _source()
    relation = relation_concept_identity((1580, 1))
    seed_relation = relation_concept_identity((1580, 2))
    left_role = role_identity((1580, 3))
    right_role = role_identity((1580, 4))
    seed_role = role_identity((1580, 5))
    relation_schema = _binary_schema(13, relation, left_role, right_role)
    seed_schema = RelationSchema(
        structure_concept_identity((1580, 6)),
        seed_relation,
        (RelationSlotSchema(
            seed_role, frozenset({OBJECT_SET_EXPR}), 1, 1),),
    )
    algebra = RelationAlgebra((relation_schema, seed_schema))
    first = set_expr_identity(source, (1580, 7))
    second = set_expr_identity(source, (1580, 8))
    relation_fact = _knowledge(_definition(source, 60, relation, (
        AtomicRoleBinding(left_role, first),
        AtomicRoleBinding(right_role, second),
    )), 60)
    seed_fact = _knowledge(_definition(source, 61, seed_relation, (
        AtomicRoleBinding(seed_role, first),
    )), 61)

    symmetric = SymmetricRule(
        minimal_instruction_identity((1580, 10)),
        relation, left_role, right_role)
    reflexive = ReflexiveRule(
        minimal_instruction_identity((1580, 11)),
        seed_relation, seed_role,
        relation, left_role, right_role)
    candidates = algebra.derive_candidates(
        (relation_fact, seed_fact), (symmetric, reflexive))
    assert {
        (_fillers(item)[left_role], _fillers(item)[right_role])
        for item in candidates
    } == {(second, first), (first, first)}

    self_fact = _knowledge(_definition(source, 62, relation, (
        AtomicRoleBinding(left_role, first),
        AtomicRoleBinding(right_role, first),
    )), 62)
    violations = algebra.irreflexive_violations(
        (relation_fact, self_fact),
        (IrreflexiveRule(
            minimal_instruction_identity((1580, 12)),
            relation, left_role, right_role),),
    )
    assert len(violations) == 1
    assert violations[0].filler == first
    assert violations[0].premise.proposition == self_fact.definition.proposition


def test_candidate_output_is_deterministic_under_input_and_rule_permutation():
    """输入事实和规则顺序不改变派生候选完整稳定键。"""
    source = _source()
    relation = relation_concept_identity((1590, 1))
    left_role = role_identity((1590, 2))
    right_role = role_identity((1590, 3))
    schema = _binary_schema(14, relation, left_role, right_role)
    values = tuple(set_expr_identity(source, (1590, item))
                   for item in range(4, 7))
    knowledge = tuple(
        _knowledge(_definition(source, 70 + index, relation, (
            AtomicRoleBinding(left_role, values[index]),
            AtomicRoleBinding(right_role, values[index + 1]),
        )), 70 + index)
        for index in range(2)
    )
    rules = (
        TransitiveRule(
            minimal_instruction_identity((1590, 8)),
            relation, left_role, right_role),
        SymmetricRule(
            minimal_instruction_identity((1590, 9)),
            relation, left_role, right_role),
    )
    algebra = RelationAlgebra((schema,))
    first = algebra.derive_candidates(knowledge, rules)
    second = algebra.derive_candidates(
        tuple(reversed(knowledge)), tuple(reversed(rules)))
    assert tuple(item.stable_key() for item in first) == tuple(
        item.stable_key() for item in second)
