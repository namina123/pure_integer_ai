"""U-01 一等图对象、动态 predicate 和结构整体契约测试。"""
from __future__ import annotations

import pytest

from pure_integer_ai.cognition.shared.graph_ontology import (
    GraphOntology,
    relation_concept_identity,
)
from pure_integer_ai.cognition.shared.hypothesis import HypothesisKey
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    OBJECT_CONCEPT,
    OBJECT_LEGACY_CHARACTER,
    ObjectIdentity,
    SourceRef,
    TypedRef,
    VersionBundle,
    language_atom_identity,
    language_branch_identity,
    legacy_character_identity,
    minimal_instruction_identity,
    occurrence_identity,
    representation_identity,
    span_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    document_scope,
    session_scope,
)
from pure_integer_ai.cognition.shared.scoped_persistence import ScopedIdentityStore
from pure_integer_ai.experiments.formal_train import make_train_context
from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.backend import DictBackend, SQLiteBackend
from pure_integer_ai.storage.discipline import AppendOnlyViolation
from pure_integer_ai.storage.edge_store import EPI_STRUCTURED, SOURCE_BARE_TEXT
from pure_integer_ai.storage.graph_object import GRAPH_OBJECT_TABLE
from pure_integer_ai.storage.graph_object import GraphObjectIntegrityError
from pure_integer_ai.storage.graph_object_identity import (
    GRAPH_HYPOTHESIS_GROUP_COMPONENT_TABLE,
    GRAPH_HYPOTHESIS_GROUP_TABLE,
    GRAPH_OBJECT_CODEC_GENERIC_V1,
    GRAPH_OBJECT_CODEC_HYPOTHESIS_V1,
    GRAPH_OBJECT_CODEC_OCCURRENCE_SOURCE_V1,
    GRAPH_OBJECT_CODEC_SPAN_SOURCE_V1,
    GRAPH_OBJECT_COMPONENT_TABLE,
    GRAPH_OBJECT_IDENTITY_COLUMNS,
)
from pure_integer_ai.storage.graph_statement import (
    GRAPH_STATEMENT_TABLE,
    GraphStatementIntegrityError,
)
from pure_integer_ai.storage.assertion_identity import (
    IDENTITY_ASSERTION,
    IDENTITY_GRAPH_OBJECT,
    IDENTITY_HEADER_TABLE,
    IDENTITY_PART_TABLE,
    IdentityIncompleteError,
)
from pure_integer_ai.storage.assertion_record import ASSERTION_RECORD_TABLE
from pure_integer_ai.storage.source_record import SourceRecordRepository
from pure_integer_ai.storage.spaces.registry import (
    SPACE_TYPE_CORE,
    SpaceRegistry,
)
from pure_integer_ai.storage.telemetry import (
    BackendTelemetryCollector,
    collect_backend_telemetry,
)
from pure_integer_ai.training.cursor import DUMP_TABLES, dump_run, load_run


def _backend(kind: str):
    """为两后端契约测试创建独立空存储。"""
    if kind == "dict":
        return DictBackend()
    if kind == "sqlite":
        return SQLiteBackend()
    raise ValueError(kind)


def _relate(ontology: GraphOntology, predicate: TypedRef,
            subject: TypedRef, object_ref: TypedRef, *, scope_id: int = 1):
    """以统一来源写入测试 statement，避免每个断言重复协议参数。"""
    return ontology.relate(
        predicate,
        subject,
        object_ref,
        scope=session_scope(scope_id),
        provenance_kind=SOURCE_BARE_TEXT,
        epistemic_origin=EPI_STRUCTURED,
    )


def _legacy_statement_row(statement) -> dict[str, int]:
    """把领域 statement 展开为迁移前冗余物理行，供兼容对抗测试使用。"""
    return {
        "assertion_hash": statement.assertion_hash,
        "predicate_identity_hash": statement.predicate_identity_hash,
        "predicate_space_id": statement.predicate.space_id,
        "predicate_local_id": statement.predicate.local_id,
        "object_kind_from": statement.subject.object_kind,
        "space_id_from": statement.subject.space_id,
        "local_id_from": statement.subject.local_id,
        "object_kind_to": statement.object.object_kind,
        "space_id_to": statement.object.space_id,
        "local_id_to": statement.object.local_id,
        "scope_hash": statement.scope_hash,
    }


@pytest.mark.parametrize("kind", ["dict", "sqlite"])
def test_representation_relations_are_many_to_many_without_host_enums(kind: str):
    backend = _backend(kind)
    try:
        ontology = make_train_context(backend).graph_ontology
        branch_a = language_branch_identity((101,))
        branch_b = language_branch_identity((202,))
        branch_a_ref = ontology.materialize(branch_a)
        branch_b_ref = ontology.materialize(branch_b)
        atom_a = ontology.materialize(language_atom_identity(branch_a, (1,)))
        atom_b = ontology.materialize(language_atom_identity(branch_b, (7,)))
        representation_a = ontology.materialize(
            representation_identity((9001,), (20320, 22909)))
        representation_b = ontology.materialize(
            representation_identity((9001,), (20320,)))
        predicate = ontology.materialize(
            relation_concept_identity((70001, 1)))
        branch_predicate = ontology.materialize(
            relation_concept_identity((70001, 2)))

        _relate(ontology, branch_predicate, branch_a_ref, atom_a)
        _relate(ontology, branch_predicate, branch_b_ref, atom_b)
        _relate(ontology, predicate, atom_a, representation_a)
        _relate(ontology, predicate, atom_a, representation_b)
        _relate(ontology, predicate, atom_b, representation_a)

        assert set(ontology.follow(atom_a, (predicate,))) == {
            representation_a, representation_b}
        assert ontology.follow(atom_b, (predicate,)) == (representation_a,)
        assert ontology.follow(
            branch_a_ref, (branch_predicate,)) == (atom_a,)
        assert len(ontology.statements(
            predicate=predicate, object_ref=representation_a)) == 2
    finally:
        backend.close()


@pytest.mark.parametrize("kind", ["dict", "sqlite"])
def test_structure_is_recoverable_as_whole_with_member_role_order_and_scope(
        kind: str):
    backend = _backend(kind)
    try:
        ontology = make_train_context(backend).graph_ontology
        root = ontology.materialize(structure_concept_identity((81, 1)))
        slot = ontology.materialize(structure_concept_identity((81, 2)))
        child = ontology.materialize(structure_concept_identity((81, 3)))
        source = SourceRef(
            3, 5, 8, GLOBAL_OWNER_SCOPE, VersionBundle())
        occurrence = ontology.materialize(
            occurrence_identity(source, start=0, end=1, ordinal=0))
        span = ontology.materialize(
            span_identity(source, members=((0, 1),)))
        role = ontology.materialize(ObjectIdentity(OBJECT_CONCEPT, (8101,)))
        order = ontology.materialize(ObjectIdentity(OBJECT_CONCEPT, (8102, 1)))
        semantic_scope = ontology.materialize(
            ObjectIdentity(OBJECT_CONCEPT, (8103,)))

        member_predicate = ontology.materialize(
            relation_concept_identity((8201,)))
        role_predicate = ontology.materialize(
            relation_concept_identity((8202,)))
        order_predicate = ontology.materialize(
            relation_concept_identity((8203,)))
        scope_predicate = ontology.materialize(
            relation_concept_identity((8204,)))
        instance_predicate = ontology.materialize(
            relation_concept_identity((8205,)))

        _relate(ontology, member_predicate, root, slot)
        _relate(ontology, member_predicate, root, child)
        _relate(ontology, member_predicate, slot, occurrence)
        _relate(ontology, role_predicate, slot, role)
        _relate(ontology, order_predicate, slot, order)
        _relate(ontology, scope_predicate, root, semantic_scope)
        _relate(ontology, instance_predicate, occurrence, root)
        _relate(ontology, instance_predicate, span, root)

        assert set(ontology.follow(root, (member_predicate,))) == {slot, child}
        assert ontology.follow(
            root, (member_predicate, member_predicate)) == (occurrence,)
        assert ontology.follow(slot, (role_predicate,)) == (role,)
        assert ontology.follow(slot, (order_predicate,)) == (order,)
        assert ontology.follow(root, (scope_predicate,)) == (semantic_scope,)
        assert ontology.follow(occurrence, (instance_predicate,)) == (root,)
        assert ontology.follow(span, (instance_predicate,)) == (root,)
        assert ontology.identity_of(root) == structure_concept_identity((81, 1))
    finally:
        backend.close()


@pytest.mark.parametrize("kind", ["dict", "sqlite"])
def test_full_graph_path_does_not_require_minimal_instruction(kind: str):
    backend = _backend(kind)
    try:
        ontology = make_train_context(backend).graph_ontology
        atom = ontology.materialize(language_atom_identity(
            language_branch_identity((301,)), (1,)))
        representation = ontology.materialize(
            representation_identity((9101,), (46,)))
        representation_predicate = ontology.materialize(
            relation_concept_identity((9102,)))
        _relate(ontology, representation_predicate, atom, representation)

        before = ontology.follow(atom, (representation_predicate,))
        instruction = ontology.materialize(
            minimal_instruction_identity((9901,)))
        binding_predicate = ontology.materialize(
            relation_concept_identity((9902,)))
        _relate(
            ontology, binding_predicate,
            representation_predicate, instruction)
        after = ontology.follow(atom, (representation_predicate,))

        assert before == after == (representation,)
        assert ontology.follow(
            representation_predicate, (binding_predicate,)) == (instruction,)
    finally:
        backend.close()


@pytest.mark.parametrize("kind", ["dict", "sqlite"])
def test_graph_ontology_rejects_legacy_and_unmaterialized_refs(kind: str):
    backend = _backend(kind)
    try:
        ontology = make_train_context(backend).graph_ontology
        with pytest.raises(ValueError, match="兼容投影"):
            ontology.materialize(legacy_character_identity(65, language=1))
        assert legacy_character_identity(
            65, language=1).object_kind == OBJECT_LEGACY_CHARACTER

        subject = ontology.materialize(structure_concept_identity((1,)))
        object_ref = ontology.materialize(structure_concept_identity((2,)))
        fake_predicate = TypedRef(OBJECT_CONCEPT, ontology.space_id, 99999)
        with pytest.raises(Exception, match="唯一对象身份映射"):
            _relate(ontology, fake_predicate, subject, object_ref)
    finally:
        backend.close()


@pytest.mark.parametrize("kind", ["dict", "sqlite"])
def test_graph_ontology_roundtrips_through_authoritative_dump(
        kind: str, tmp_path):
    first_backend = _backend(kind)
    try:
        first = make_train_context(first_backend)
        ontology = first.graph_ontology
        atom_identity = language_atom_identity(
            language_branch_identity((401,)), (9,))
        representation_key = representation_identity((9201,), (65, 66))
        predicate_key = relation_concept_identity((9202,))
        atom = ontology.materialize(atom_identity)
        representation = ontology.materialize(representation_key)
        predicate = ontology.materialize(predicate_key)
        statement = _relate(ontology, predicate, atom, representation)

        dump_run(
            first_backend,
            str(tmp_path),
            "run_u01",
            spaces=[first.space_id],
            tables=DUMP_TABLES,
        )
    finally:
        first_backend.close()

    second_backend = _backend(kind)
    try:
        second = make_train_context(second_backend)
        assert load_run(second_backend, str(tmp_path), "run_u01") == [1]
        restored_atom = second.graph_ontology.resolve(atom_identity)
        restored_representation = second.graph_ontology.resolve(
            representation_key)
        restored_predicate = second.graph_ontology.resolve(predicate_key)
        assert restored_atom is not None
        assert restored_representation is not None
        assert restored_predicate is not None
        restored = second.graph_ontology.statements(
            predicate=restored_predicate,
            subject=restored_atom,
            object_ref=restored_representation,
        )
        assert len(restored) == 1
        assert restored[0].assertion_hash == statement.assertion_hash
    finally:
        second_backend.close()


def test_graph_tables_are_core_append_only_and_in_formal_dump():
    backend = DictBackend()
    try:
        bootstrap(backend)
        registry = SpaceRegistry(backend)
        space_id = registry.register(SPACE_TYPE_CORE, "core")
        scoped = ScopedIdentityStore(backend)
        ontology = GraphOntology(
            backend,
            space_id=space_id,
            space_identity=SpaceRegistry.identity_for(SPACE_TYPE_CORE, "core"),
            scoped_identities=scoped,
        )
        ontology.materialize(structure_concept_identity((1,)))
        with pytest.raises(AppendOnlyViolation):
            backend.delete(GRAPH_OBJECT_TABLE, where={"space_id": space_id})
        with pytest.raises(AppendOnlyViolation):
            backend.delete(GRAPH_STATEMENT_TABLE, where={"space_id_from": space_id})
        assert GRAPH_OBJECT_TABLE in DUMP_TABLES
        assert GRAPH_OBJECT_COMPONENT_TABLE in DUMP_TABLES
        assert GRAPH_HYPOTHESIS_GROUP_TABLE in DUMP_TABLES
        assert GRAPH_HYPOTHESIS_GROUP_COMPONENT_TABLE in DUMP_TABLES
        assert GRAPH_STATEMENT_TABLE in DUMP_TABLES
    finally:
        backend.close()


@pytest.mark.parametrize("kind", ["dict", "sqlite"])
def test_graph_object_identity_uses_self_header_and_open_overflow(kind: str):
    """普通对象固定槽加 overflow 必须恢复同一完整键且不再写通用身份行。"""
    backend = _backend(kind)
    try:
        ctx = make_train_context(backend)
        identity = representation_identity(
            (1, 2, 3, 4), tuple(range(100, 112)))
        ref = ctx.graph_ontology.materialize(identity)
        identity_hash = ctx.graph_ontology.identity_hash_of(ref)
        row = backend.select(GRAPH_OBJECT_TABLE, where={
            "identity_hash": identity_hash,
        })[0]
        assert row["identity_codec"] == GRAPH_OBJECT_CODEC_GENERIC_V1
        assert backend.select(IDENTITY_HEADER_TABLE, where={
            "identity_kind": IDENTITY_GRAPH_OBJECT,
        }) == []
        assert backend.select(IDENTITY_PART_TABLE, where={
            "identity_kind": IDENTITY_GRAPH_OBJECT,
        }) == []
        assert len(backend.select(GRAPH_OBJECT_COMPONENT_TABLE, where={
            "identity_hash": identity_hash,
        })) == len(identity.components) - 8
        ctx.graph_ontology.clear_runtime_caches()
        assert ctx.graph_ontology.identity_of(ref) == identity
    finally:
        backend.close()


@pytest.mark.parametrize("kind", ["dict", "sqlite"])
def test_source_graph_object_codecs_reuse_source_identity(kind: str):
    """来源已登记时 Occurrence/Span 只保存 source hash 和必要结构字段。"""
    backend = _backend(kind)
    try:
        ctx = make_train_context(backend)
        source = SourceRef(
            4, 31, 7, GLOBAL_OWNER_SCOPE, VersionBundle())
        SourceRecordRepository(
            backend, registry=ctx.scoped_identity_store.registry,
        ).put(source.stable_key(), "甲乙丙")
        occurrence_identity_value = occurrence_identity(
            source, start=0, end=1, ordinal=0)
        span_identity_value = span_identity(
            source, members=((0, 1),), ordinal=0)
        occurrence = ctx.graph_ontology.materialize(
            occurrence_identity_value)
        span = ctx.graph_ontology.materialize(span_identity_value)
        occurrence_hash = ctx.graph_ontology.identity_hash_of(occurrence)
        span_hash = ctx.graph_ontology.identity_hash_of(span)
        occurrence_row = backend.select(GRAPH_OBJECT_TABLE, where={
            "identity_hash": occurrence_hash,
        })[0]
        span_row = backend.select(GRAPH_OBJECT_TABLE, where={
            "identity_hash": span_hash,
        })[0]
        assert occurrence_row["identity_codec"] == (
            GRAPH_OBJECT_CODEC_OCCURRENCE_SOURCE_V1)
        assert span_row["identity_codec"] == (
            GRAPH_OBJECT_CODEC_SPAN_SOURCE_V1)
        assert backend.select(GRAPH_OBJECT_COMPONENT_TABLE, where={
            "identity_hash": occurrence_hash,
        }) == []
        assert backend.select(GRAPH_OBJECT_COMPONENT_TABLE, where={
            "identity_hash": span_hash,
        }) == []
        ctx.graph_ontology.clear_runtime_caches()
        assert ctx.graph_ontology.identity_of(
            occurrence) == occurrence_identity_value
        assert ctx.graph_ontology.identity_of(span) == span_identity_value
    finally:
        backend.close()


@pytest.mark.parametrize("kind", ["dict", "sqlite"])
def test_hypothesis_codec_deduplicates_competition_group(kind: str):
    """同竞争组只保存一次 kind/competition，候选仅保存可逆 suffix。"""
    backend = _backend(kind)
    try:
        ctx = make_train_context(backend)
        source = SourceRef(
            4, 41, 9, GLOBAL_OWNER_SCOPE, VersionBundle())
        SourceRecordRepository(
            backend, registry=ctx.scoped_identity_store.registry,
        ).put(source.stable_key(), "候选竞争组")
        scope = document_scope(source)
        ctx.scoped_identity_store.register_scope(scope)
        competition_key = tuple(range(20, 40))
        hypotheses = tuple(HypothesisKey(
            (7001, 1),
            (*competition_key, marker, 0, 1, 2, 3, 4, 5, 6),
            competition_key,
            scope,
            source,
        ) for marker in (101, 202))
        refs = tuple(ctx.graph_ontology.materialize(
            hypothesis.object_identity()) for hypothesis in hypotheses)
        assert backend.count(GRAPH_HYPOTHESIS_GROUP_TABLE) == 1
        assert backend.count(
            GRAPH_HYPOTHESIS_GROUP_COMPONENT_TABLE) == (
                len(hypotheses[0].hypothesis_kind)
                + len(competition_key))
        for ref, hypothesis in zip(refs, hypotheses):
            identity_hash = ctx.graph_ontology.identity_hash_of(ref)
            row = backend.select(GRAPH_OBJECT_TABLE, where={
                "identity_hash": identity_hash,
            })[0]
            assert row["identity_codec"] == GRAPH_OBJECT_CODEC_HYPOTHESIS_V1
            assert row["codec_value_2"] == len(competition_key)
            assert backend.count(GRAPH_OBJECT_COMPONENT_TABLE, where={
                "identity_hash": identity_hash,
            }) == 3
        ctx.graph_ontology.clear_runtime_caches()
        for ref, hypothesis in zip(refs, hypotheses):
            assert ctx.graph_ontology.identity_of(
                ref) == hypothesis.object_identity()
    finally:
        backend.close()


def test_graph_object_codec_rejects_missing_overflow_after_cache_clear():
    """缺失普通对象 overflow 后不得由旧运行期身份缓存掩盖损坏。"""
    backend = _backend("dict")
    ctx = make_train_context(backend)
    identity = representation_identity(
        (1, 2, 3, 4), tuple(range(100, 112)))
    ref = ctx.graph_ontology.materialize(identity)
    snapshot = backend.snapshot()
    snapshot[GRAPH_OBJECT_COMPONENT_TABLE] = (
        snapshot[GRAPH_OBJECT_COMPONENT_TABLE][:-1])
    restored_backend = _backend("dict")
    restored_ctx = make_train_context(restored_backend)
    restored_backend.load_snapshot(snapshot)
    restored_ctx.graph_ontology.clear_runtime_caches()
    with pytest.raises(IdentityIncompleteError, match="component"):
        restored_ctx.graph_ontology.resolve(identity)


def test_graph_object_codec_reads_legacy_header_and_flat_parts():
    """旧 graph_object 映射加通用 header/parts 必须恢复同一对象身份。"""
    backend = _backend("dict")
    ctx = make_train_context(backend)
    identity = structure_concept_identity((8811,))
    ref = ctx.graph_ontology.materialize(identity)
    identity_hash = ctx.graph_ontology.identity_hash_of(ref)
    snapshot = backend.snapshot()
    graph_row = next(row for row in snapshot[GRAPH_OBJECT_TABLE]
                     if row["identity_hash"] == identity_hash)
    for name, _ in GRAPH_OBJECT_IDENTITY_COLUMNS:
        graph_row.pop(name)
    snapshot[GRAPH_OBJECT_COMPONENT_TABLE] = [
        row for row in snapshot[GRAPH_OBJECT_COMPONENT_TABLE]
        if row["identity_hash"] != identity_hash
    ]
    snapshot[IDENTITY_HEADER_TABLE].append({
        "identity_kind": IDENTITY_GRAPH_OBJECT,
        "identity_hash": identity_hash,
        "key_size": len(identity.stable_key()),
        "parent_hash": 0,
        "ordinal": 0,
    })
    snapshot[IDENTITY_PART_TABLE].extend({
        "identity_kind": IDENTITY_GRAPH_OBJECT,
        "identity_hash": identity_hash,
        "part_index": index,
        "part_value": value,
    } for index, value in enumerate(identity.stable_key()))

    restored_backend = _backend("dict")
    restored_ctx = make_train_context(restored_backend)
    restored_backend.load_snapshot(snapshot)
    assert restored_ctx.graph_ontology.resolve(identity) == ref


@pytest.mark.parametrize("corruption", ["group_missing", "unused_slot"])
def test_graph_object_codec_rejects_noncanonical_payload(corruption: str):
    """竞争组缺项或未用固定槽非零都不能恢复为另一个合法物理表示。"""
    backend = _backend("dict")
    ctx = make_train_context(backend)
    if corruption == "group_missing":
        source = SourceRef(
            4, 51, 11, GLOBAL_OWNER_SCOPE, VersionBundle())
        SourceRecordRepository(
            backend, registry=ctx.scoped_identity_store.registry,
        ).put(source.stable_key(), "组损坏")
        scope = document_scope(source)
        hypothesis = HypothesisKey(
            (8001,),
            (1, 2, 3, 4, 5, 6, 7),
            (1, 2, 3),
            scope,
            source,
        )
        identity = hypothesis.object_identity()
    else:
        identity = structure_concept_identity((1,))
    ctx.graph_ontology.materialize(identity)
    snapshot = backend.snapshot()
    if corruption == "group_missing":
        snapshot[GRAPH_HYPOTHESIS_GROUP_COMPONENT_TABLE] = (
            snapshot[GRAPH_HYPOTHESIS_GROUP_COMPONENT_TABLE][:-1])
    else:
        row = next(row for row in snapshot[GRAPH_OBJECT_TABLE]
                   if row["object_kind"] == identity.object_kind
                   and row["component_size"] == 1
                   and row["codec_value_0"] == 1)
        row["codec_value_7"] = 9
    restored_backend = _backend("dict")
    restored_ctx = make_train_context(restored_backend)
    restored_backend.load_snapshot(snapshot)
    restored_ctx.graph_ontology.clear_runtime_caches()
    with pytest.raises(IdentityIncompleteError):
        restored_ctx.graph_ontology.resolve(identity)


def test_runtime_identity_cache_is_state_neutral_and_explicitly_invalidated():
    """缓存不得改状态；外部故障注入后清缓存必须重新触发完整性核验。"""
    backend = DictBackend()
    try:
        ontology = make_train_context(backend).graph_ontology
        identity = structure_concept_identity((8801,))
        ref = ontology.materialize(identity)
        before = backend.snapshot()

        for _ in range(20):
            assert ontology.identity_of(ref) == identity
            assert ontology.resolve(identity) == ref
        assert backend.snapshot() == before

        row = backend.select(GRAPH_OBJECT_TABLE, where={
            "space_id": ref.space_id,
            "local_id": ref.local_id,
        })[0]
        backend.insert(GRAPH_OBJECT_TABLE, row)
        assert ontology.identity_of(ref) == identity
        ontology.clear_runtime_caches()
        with pytest.raises(GraphObjectIntegrityError, match="唯一对象身份"):
            ontology.identity_of(ref)
    finally:
        backend.close()


def test_statement_projection_accepts_exact_physical_index_and_rejects_drift():
    """assertion 为权威；精确物理索引可恢复，任一字段漂移必须失败。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        ontology = ctx.graph_ontology
        predicate = ontology.materialize(relation_concept_identity((9901,)))
        subject = ontology.materialize(structure_concept_identity((9902,)))
        object_ref = ontology.materialize(structure_concept_identity((9903,)))
        statement = _relate(ontology, predicate, subject, object_ref)
        before = backend.snapshot()
        assert _relate(
            ontology, predicate, subject, object_ref).assertion_hash == (
                statement.assertion_hash)
        assert backend.snapshot() == before
        assert backend.select(GRAPH_STATEMENT_TABLE, where=None) == []

        backend.insert(GRAPH_STATEMENT_TABLE, _legacy_statement_row(statement))
        ontology.clear_runtime_caches()
        assert ontology.statements(
            predicate=predicate, subject=subject) == (statement,)

        row = _legacy_statement_row(statement)
        row["assertion_hash"] += 1
        backend.insert(GRAPH_STATEMENT_TABLE, row)
        ontology.clear_runtime_caches()
        with pytest.raises(GraphStatementIntegrityError):
            ontology.statements(predicate=predicate, subject=subject)
    finally:
        backend.close()


def test_empty_legacy_statement_namespace_avoids_redundant_queries():
    """新格式只检查一次旧表为空，后续写入和查询不得再访问旧冗余行。"""
    backend = DictBackend()
    try:
        ontology = make_train_context(backend).graph_ontology
        predicate = ontology.materialize(relation_concept_identity((99010,)))
        subject = ontology.materialize(structure_concept_identity((99011,)))
        object_ref = ontology.materialize(structure_concept_identity((99012,)))
        collector = BackendTelemetryCollector()

        with collect_backend_telemetry(collector):
            statement = _relate(ontology, predicate, subject, object_ref)
            for _ in range(5):
                assert ontology.statements(
                    predicate=predicate,
                    subject=subject,
                    object_ref=object_ref,
                ) == (statement,)

        operations = [
            row
            for row in collector.to_json()["table_operations"]
            if row["table"] == GRAPH_STATEMENT_TABLE
        ]
        assert operations == [{
            "table": GRAPH_STATEMENT_TABLE,
            "operation": "count",
            "calls": 1,
            "rows": 0,
            "failures": 0,
        }]
    finally:
        backend.close()


def test_legacy_statement_row_round_trips_and_duplicate_fails_closed():
    """旧 role=generic 快照只接受唯一吻合 statement 行，重复行不得被投影掩盖。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        ontology = ctx.graph_ontology
        predicate_identity = relation_concept_identity((9907,))
        subject_identity = structure_concept_identity((9908,))
        object_identity = structure_concept_identity((9909,))
        predicate = ontology.materialize(predicate_identity)
        subject = ontology.materialize(subject_identity)
        object_ref = ontology.materialize(object_identity)
        statement = _relate(ontology, predicate, subject, object_ref)
        row = _legacy_statement_row(statement)
        snapshot = backend.snapshot()
        assertion_record = None
        for record in snapshot[ASSERTION_RECORD_TABLE]:
            if record["identity_hash"] == statement.assertion_hash:
                record["assertion_role"] = 0
                assertion_record = record
        assert assertion_record is not None
        snapshot[IDENTITY_HEADER_TABLE].append({
            "identity_kind": IDENTITY_ASSERTION,
            "identity_hash": statement.assertion_hash,
            "key_size": len(statement.assertion.stable_key()),
            "parent_hash": assertion_record["scope_hash"],
            "ordinal": 0,
        })
        snapshot[GRAPH_STATEMENT_TABLE] = [row]
        backend.load_snapshot(snapshot)
        ontology.clear_runtime_caches()

        restored_predicate = ontology.resolve(predicate_identity)
        restored_subject = ontology.resolve(subject_identity)
        restored_object = ontology.resolve(object_identity)
        assert restored_predicate is not None
        assert restored_subject is not None
        assert restored_object is not None
        assert ontology.statements(
            predicate=restored_predicate,
            subject=restored_subject,
            object_ref=restored_object,
        ) == (statement,)

        backend.insert(GRAPH_STATEMENT_TABLE, row)
        ontology.clear_runtime_caches()
        with pytest.raises(GraphStatementIntegrityError, match="重复"):
            ontology.statements(predicate=restored_predicate)
    finally:
        backend.close()


def test_statement_fast_and_strict_paths_have_identical_physical_snapshot():
    """同一输入经空命名空间快路或逐项严格路径必须产生相同物理快照。"""
    backends = (DictBackend(), DictBackend())
    try:
        contexts = tuple(make_train_context(backend) for backend in backends)
        refs = []
        for ctx in contexts:
            ontology = ctx.graph_ontology
            refs.append((
                ontology.materialize(relation_concept_identity((9904,))),
                ontology.materialize(structure_concept_identity((9905,))),
                ontology.materialize(structure_concept_identity((9906,))),
            ))
        strict_ctx = contexts[1]
        strict_ctx.scoped_identity_store._fresh_assertion_namespace = False
        strict_ctx.graph_ontology._statements._fresh_namespace = False

        for ctx, (predicate, subject, object_ref) in zip(contexts, refs):
            _relate(ctx.graph_ontology, predicate, subject, object_ref)

        assert backends[0].snapshot() == backends[1].snapshot()
    finally:
        for backend in backends:
            backend.close()
