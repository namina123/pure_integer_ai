"""S-00 统一语义对象、n 元角色拓扑和 Evidence 分离测试。"""
from __future__ import annotations

import pytest

from pure_integer_ai.cognition.shared.graph_ontology import (
    relation_concept_identity,
)
from pure_integer_ai.cognition.shared.hypothesis import (
    EPISTEMIC_CONFLICTED,
    EPISTEMIC_SUPPORTED,
    EPISTEMIC_UNKNOWN,
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EvidenceRecord,
    HypothesisLedger,
)
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    OBJECT_BINDER,
    OBJECT_CONTEXT_SCOPE,
    OBJECT_ENTITY,
    OBJECT_EVENT,
    OBJECT_PROPOSITION,
    OBJECT_ROLE,
    OBJECT_ROLE_BINDING,
    OBJECT_SET_EXPR,
    OBJECT_VARIABLE,
    ObjectIdentity,
    SourceRef,
    VersionBundle,
    concept_identity,
    legacy_character_identity,
    minimal_instruction_identity,
    object_contracts_by_kind,
    occurrence_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.cognition.shared.semantic_graph import (
    AtomicPropositionPredicates,
    SemanticGraph,
    SemanticTopologyError,
)
from pure_integer_ai.cognition.shared.semantic_object import (
    AtomicPropositionDefinition,
    AtomicRoleBinding,
    binder_identity,
    context_scope_identity,
    entity_identity,
    event_identity,
    project_proposition_knowledge,
    proposition_hypothesis_key,
    proposition_identity,
    role_binding_identity,
    role_binding_ordinal,
    role_identity,
    semantic_source,
    set_expr_identity,
    variable_identity,
)
from pure_integer_ai.experiments.formal_train import make_train_context
from pure_integer_ai.storage.backend import DictBackend, SQLiteBackend
from pure_integer_ai.storage.edge_store import EPI_STRUCTURED, SOURCE_BARE_TEXT
from pure_integer_ai.training.cursor import DUMP_TABLES, dump_run, load_run


def _backend(kind: str):
    """为语义图契约创建独立两类后端。"""
    if kind == "dict":
        return DictBackend()
    if kind == "sqlite":
        return SQLiteBackend()
    raise ValueError(kind)


def _source(document_id: int = 1) -> SourceRef:
    """构造稳定测试来源，document_id 用于验证跨来源不合并。"""
    return SourceRef(
        71, 73, document_id, GLOBAL_OWNER_SCOPE, VersionBundle())


def _predicates(ontology, *, family: int = 8100):
    """物化一套完全注入的协议 predicate，不依赖宿主关系枚举。"""
    refs = tuple(
        ontology.materialize(
            relation_concept_identity((family, ordinal)))
        for ordinal in range(1, 7)
    )
    return AtomicPropositionPredicates(*refs)


def _definition(source: SourceRef, *, reverse: bool = False):
    """构造含 speaker/time/place 三类普通开放 Role 的原子命题。"""
    proposition = proposition_identity(source, (101, 1))
    speaker = AtomicRoleBinding(
        role_identity((201, 11)), entity_identity(source, (301, 1)), 0)
    time = AtomicRoleBinding(
        role_identity((201, 12)), entity_identity(source, (301, 2)), 0)
    place = AtomicRoleBinding(
        role_identity((201, 13)), entity_identity(source, (301, 3)), 0)
    bindings = (speaker, time, place)
    if reverse:
        bindings = tuple(reversed(bindings))
    return AtomicPropositionDefinition(
        proposition,
        concept_identity((401, 9)),
        occurrence_identity(source, start=0, end=3, ordinal=0),
        context_scope_identity(source, (501, 7)),
        bindings,
    )


@pytest.mark.parametrize("kind", ["dict", "sqlite"])
def test_atomic_proposition_roundtrips_open_roles_without_naked_filler_edges(
        kind: str):
    """三类限定角色走同一 RoleBinding 协议，清缓存和乱序重放保持同一图。"""
    backend = _backend(kind)
    try:
        ctx = make_train_context(backend)
        ontology = ctx.graph_ontology
        predicates = _predicates(ontology)
        graph = SemanticGraph(ontology, predicates)
        source = _source()
        scope = document_scope(source)
        definition = _definition(source)

        materialized = graph.define_atomic(
            definition,
            scope=scope,
            provenance_kind=SOURCE_BARE_TEXT,
            epistemic_origin=EPI_STRUCTURED,
            content_version=3,
            qualifiers=(17, 19),
        )
        before = backend.snapshot()
        assert materialized.definition == _definition(source, reverse=True)
        assert materialized.scope == scope
        assert materialized.definition.source == source
        assert len(materialized.bindings) == 3
        assert all(binding.identity.object_kind == OBJECT_ROLE_BINDING
                   for binding in materialized.bindings)
        for binding in materialized.bindings:
            assert ontology.statements(
                subject=materialized.proposition,
                object_ref=binding.filler,
            ) == ()

        ontology.clear_runtime_caches()
        restored = SemanticGraph(ontology, predicates).read_atomic(
            materialized.proposition)
        assert restored == materialized
        replayed = SemanticGraph(ontology, predicates).define_atomic(
            _definition(source, reverse=True),
            scope=scope,
            provenance_kind=SOURCE_BARE_TEXT,
            epistemic_origin=EPI_STRUCTURED,
            content_version=3,
            qualifiers=(17, 19),
        )
        assert replayed == materialized
        assert backend.snapshot() == before
    finally:
        backend.close()


def test_proposition_existence_is_not_truth_and_status_comes_from_h00():
    """图中命题存在不自动造 Evidence；支持和反驳由 H-00 独立聚合。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        source = _source()
        definition = _definition(source)
        graph = SemanticGraph(ctx.graph_ontology, _predicates(ctx.graph_ontology))
        graph.define_atomic(
            definition,
            scope=document_scope(source),
            provenance_kind=SOURCE_BARE_TEXT,
        )
        hypothesis = proposition_hypothesis_key(
            definition.proposition,
            hypothesis_kind=(601, 1),
            competition_key=(601, 2),
            scope=document_scope(source),
        )
        ledger = HypothesisLedger()
        with pytest.raises(KeyError, match="尚未登记"):
            project_proposition_knowledge(definition, hypothesis, ledger)

        ledger.register(hypothesis)
        unknown = project_proposition_knowledge(
            definition, hypothesis, ledger)
        assert unknown.snapshot.epistemic_status == EPISTEMIC_UNKNOWN
        ledger.append_evidence(EvidenceRecord(
            1, hypothesis, EVIDENCE_SUPPORT, (701, 1), source, 1))
        supported = project_proposition_knowledge(
            definition, hypothesis, ledger)
        assert supported.snapshot.epistemic_status == EPISTEMIC_SUPPORTED
        ledger.append_evidence(EvidenceRecord(
            2, hypothesis, EVIDENCE_REFUTE, (701, 2), source, 2))
        conflicted = project_proposition_knowledge(
            definition, hypothesis, ledger)
        assert conflicted.snapshot.epistemic_status == EPISTEMIC_CONFLICTED
        assert conflicted.source == source
        assert conflicted.scope == document_scope(source)
    finally:
        backend.close()


def test_semantic_identity_families_are_typed_source_bound_and_round_trip():
    """概念层 Role 与来源化实例分离，typed Variable 保存 Binder 和类型全键。"""
    first = _source(1)
    second = _source(2)
    role = role_identity((801, 1))
    value_type = concept_identity((801, 2))
    binder = binder_identity(first, (801, 3))
    variable = variable_identity(binder, (801, 4), value_type)
    entity = entity_identity(first, (801, 5))
    event = event_identity(first, (801, 6))
    proposition = proposition_identity(first, (801, 7))
    set_expr = set_expr_identity(first, (801, 8))
    context = context_scope_identity(first, (801, 9))
    binding = role_binding_identity(
        proposition, role, entity, ordinal=2)

    identities = (
        role, binder, variable, entity, event,
        proposition, set_expr, context, binding,
    )
    assert tuple(identity.object_kind for identity in identities) == (
        OBJECT_ROLE,
        OBJECT_BINDER,
        OBJECT_VARIABLE,
        OBJECT_ENTITY,
        OBJECT_EVENT,
        OBJECT_PROPOSITION,
        OBJECT_SET_EXPR,
        OBJECT_CONTEXT_SCOPE,
        OBJECT_ROLE_BINDING,
    )
    assert all(ObjectIdentity.from_stable_key(identity.stable_key()) == identity
               for identity in identities)
    assert semantic_source(variable) == first
    assert semantic_source(binding) == first
    assert role_binding_ordinal(binding) == 2
    assert proposition != proposition_identity(second, (801, 7))
    assert context != context_scope_identity(second, (801, 9))
    assert role == role_identity((801, 1))
    contracts = object_contracts_by_kind()
    assert all(contracts[identity.object_kind].persistence_owner
               == "storage.graph_object" for identity in identities)


def test_role_binding_parser_rejects_corrupt_nested_lengths():
    """RoleBinding 不能只读尾部 ordinal 而忽略被截断或伪造的完整嵌套身份。"""
    source = _source()
    binding = role_binding_identity(
        proposition_identity(source, (901, 1)),
        role_identity((901, 2)),
        entity_identity(source, (901, 3)),
        ordinal=0,
    )
    corrupt = list(binding.components)
    corrupt[12] += 1
    corrupt_identity = ObjectIdentity(
        OBJECT_ROLE_BINDING,
        tuple(corrupt),
        binding.owner,
        binding.versions,
    )
    with pytest.raises(ValueError):
        role_binding_ordinal(corrupt_identity)

    malformed_entity = entity_identity(source, (901, 4))
    malformed_components = list(malformed_entity.components)
    malformed_components[12] += 1
    malformed_identity = ObjectIdentity(
        OBJECT_ENTITY,
        tuple(malformed_components),
        malformed_entity.owner,
        malformed_entity.versions,
    )
    backend = DictBackend()
    try:
        ontology = make_train_context(backend).graph_ontology
        with pytest.raises(ValueError, match="长度"):
            ontology.materialize(malformed_identity)
        assert ontology.resolve(malformed_identity) is None
    finally:
        backend.close()


def test_role_binding_accepts_authoritative_graph_objects_not_legacy_projection():
    """通用 RoleBinding 保持开放，具体关系类型由后续 relation schema 收窄。"""
    source = _source()
    proposition = proposition_identity(source, (905, 1))
    role = role_identity((905, 2))
    fillers = (
        concept_identity((905, 3)),
        structure_concept_identity((905, 4)),
        minimal_instruction_identity((905, 5)),
    )
    for ordinal, filler in enumerate(fillers):
        binding = AtomicRoleBinding(role, filler, ordinal)
        assert role_binding_ordinal(
            binding.identity_for(proposition)) == ordinal

    with pytest.raises(ValueError, match="权威|承担语义角色"):
        AtomicRoleBinding(
            role,
            legacy_character_identity(65, language=1),
            0,
        )


def test_partial_existing_role_binding_fails_before_proposition_statements():
    """孤立 RoleBinding 的半条定义必须预检失败，不能先写 Proposition 基槽。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        ontology = ctx.graph_ontology
        predicates = _predicates(ontology)
        graph = SemanticGraph(ontology, predicates)
        source = _source()
        definition = _definition(source)
        first_binding = definition.bindings[0]
        binding_ref = ontology.materialize(
            first_binding.identity_for(definition.proposition))
        role_ref = ontology.materialize(first_binding.role)
        ontology.relate(
            predicates.binding_role,
            binding_ref,
            role_ref,
            scope=document_scope(source),
            provenance_kind=SOURCE_BARE_TEXT,
        )
        before = backend.snapshot()

        with pytest.raises(SemanticTopologyError, match="部分定义"):
            graph.define_atomic(
                definition,
                scope=document_scope(source),
                provenance_kind=SOURCE_BARE_TEXT,
            )
        assert backend.snapshot() == before
        proposition_ref = ontology.resolve(definition.proposition)
        assert proposition_ref is None
    finally:
        backend.close()


def test_competing_topology_and_legacy_concept_cannot_masquerade_as_proposition():
    """多值 predicate 和旧 Concept 节点都不能被首行选择或类型降级掩盖。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        ontology = ctx.graph_ontology
        predicates = _predicates(ontology)
        graph = SemanticGraph(ontology, predicates)
        source = _source()
        materialized = graph.define_atomic(
            _definition(source),
            scope=document_scope(source),
            provenance_kind=SOURCE_BARE_TEXT,
        )
        competing = ontology.materialize(concept_identity((1001, 99)))
        ontology.relate(
            predicates.proposition_predicate,
            materialized.proposition,
            competing,
            scope=document_scope(source),
            provenance_kind=SOURCE_BARE_TEXT,
        )
        with pytest.raises(SemanticTopologyError, match="恰有一条"):
            graph.read_atomic(materialized.proposition)

        legacy_like = ontology.materialize(concept_identity((95, 95, 112, 114, 111, 112)))
        with pytest.raises(ValueError, match="Proposition"):
            graph.read_atomic(legacy_like)
    finally:
        backend.close()


@pytest.mark.parametrize("kind", ["dict", "sqlite"])
def test_semantic_graph_roundtrips_through_authoritative_dump(kind: str, tmp_path):
    """语义对象只依赖通用权威图表，完整 dump/load 后无需私有语义表即可恢复。"""
    source = _source()
    definition = _definition(source)
    first_backend = _backend(kind)
    try:
        first = make_train_context(first_backend)
        predicate_identities = tuple(
            relation_concept_identity((1100, ordinal))
            for ordinal in range(1, 7)
        )
        predicates = AtomicPropositionPredicates(*tuple(
            first.graph_ontology.materialize(identity)
            for identity in predicate_identities
        ))
        materialized = SemanticGraph(
            first.graph_ontology, predicates).define_atomic(
                definition,
                scope=document_scope(source),
                provenance_kind=SOURCE_BARE_TEXT,
                epistemic_origin=EPI_STRUCTURED,
            )
        dump_run(
            first_backend,
            str(tmp_path),
            "run_s00",
            spaces=[first.space_id],
            tables=DUMP_TABLES,
        )
    finally:
        first_backend.close()

    second_backend = _backend(kind)
    try:
        second = make_train_context(second_backend)
        assert load_run(second_backend, str(tmp_path), "run_s00") == [1]
        restored_predicates = AtomicPropositionPredicates(*tuple(
            second.graph_ontology.resolve(identity)
            for identity in predicate_identities
        ))
        proposition_ref = second.graph_ontology.resolve(
            definition.proposition)
        assert proposition_ref is not None
        restored = SemanticGraph(
            second.graph_ontology, restored_predicates).read_atomic(
                proposition_ref)
        assert restored.definition == definition
        assert restored.assertion_hashes == materialized.assertion_hashes
    finally:
        second_backend.close()
