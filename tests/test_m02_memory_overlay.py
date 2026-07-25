"""M-02 owner scope、Core identity 复用和 Memory overlay 对抗测试。"""
from __future__ import annotations

import copy

import pytest

from pure_integer_ai.cognition.shared.graph_ontology import (
    GraphOntology,
    relation_concept_identity,
)
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    OBJECT_CONCEPT,
    OwnerScope,
    TypedRef,
    VISIBILITY_SESSION,
    VISIBILITY_TENANT,
    VISIBILITY_USER,
    VersionBundle,
    concept_identity,
)
from pure_integer_ai.cognition.shared.memory_overlay import (
    CoreIdentityBoundaryError,
    CoreIdentityCatalog,
    MemoryAccessContext,
    MemoryOverlay,
    MemoryOverlayQueryError,
    MemoryOverlayRelation,
)
from pure_integer_ai.cognition.shared.scope_identity import session_scope
from pure_integer_ai.cognition.shared.scoped_persistence import ScopedIdentityStore
from pure_integer_ai.experiments.evaluation_isolation import (
    clone_backend,
    clone_train_context,
)
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.storage.backend import DictBackend, SQLiteBackend
from pure_integer_ai.storage.discipline import AppendOnlyViolation
from pure_integer_ai.storage.edge_store import EPI_STRUCTURED, SOURCE_BARE_TEXT
from pure_integer_ai.storage.memory_overlay import (
    MEMORY_OVERLAY_TABLE,
    MemoryOverlayIntegrityError,
)
from pure_integer_ai.storage.spaces.registry import (
    SPACE_TYPE_CORE,
    SpaceRegistry,
)
from pure_integer_ai.training.cursor import DUMP_TABLES, dump_run, load_run


def _backend(kind: str):
    """为双后端契约测试创建独立空存储。"""
    if kind == "dict":
        return DictBackend()
    if kind == "sqlite":
        return SQLiteBackend()
    raise ValueError(kind)


def _refs(ctx):
    """在正式 Core 中物化三个不依赖表层文字的关系端点。"""
    ontology = ctx.graph_ontology
    predicate = ontology.materialize(relation_concept_identity((92001,)))
    subject = ontology.materialize(concept_identity((92002,)))
    object_ref = ontology.materialize(concept_identity((92003,)))
    return predicate, subject, object_ref


def _relation(overlay: MemoryOverlay, refs, owner: OwnerScope, *,
              scope_id: int, qualifier: int = 0) -> MemoryOverlayRelation:
    """用统一来源构造显式 owner 的测试 overlay 关系。"""
    predicate, subject, object_ref = refs
    return MemoryOverlayRelation(
        overlay.memory_space_identity,
        owner,
        VersionBundle(),
        predicate,
        subject,
        object_ref,
        session_scope(scope_id, owner=owner),
        SOURCE_BARE_TEXT,
        EPI_STRUCTURED,
        1,
        (qualifier,),
    )


@pytest.mark.parametrize("kind", ["dict", "sqlite"])
def test_overlay_reuses_cross_core_identity_without_core_writes(kind: str):
    """跨 Core 空间关系只保存 typed ref，不能复制概念或新增 Core statement/edge。"""
    backend = _backend(kind)
    try:
        ctx = make_train_context(backend)
        registry = ctx.memory_interact.registry
        second_space_id = registry.register(SPACE_TYPE_CORE, "core_second")
        second_scoped_identities = ScopedIdentityStore(backend)
        second_ontology = GraphOntology(
            backend,
            space_id=second_space_id,
            space_identity=SpaceRegistry.identity_for(
                SPACE_TYPE_CORE, "core_second"),
            scoped_identities=second_scoped_identities,
        )
        predicate = ctx.graph_ontology.materialize(
            relation_concept_identity((93001,)))
        subject = ctx.graph_ontology.materialize(concept_identity((93002,)))
        object_ref = second_ontology.materialize(concept_identity((93003,)))
        catalog = CoreIdentityCatalog((ctx.graph_ontology, second_ontology))
        overlay = MemoryOverlay(
            registry,
            backend,
            ctx.memory_interact.space_id,
            ctx.scoped_identity_store,
            catalog,
        )
        owner = OwnerScope(7, 8, 9, VISIBILITY_SESSION)
        relation = _relation(
            overlay, (predicate, subject, object_ref), owner,
            scope_id=9, qualifier=31)
        core_counts = {
            table: backend.count(table)
            for table in ("concept_node", "graph_object", "graph_statement", "edge")
        }

        first = overlay.add(relation)
        second = overlay.add(relation)

        assert first == second
        assert backend.count(MEMORY_OVERLAY_TABLE) == 1
        assert {
            table: backend.count(table)
            for table in core_counts
        } == core_counts
        visible = overlay.query(access=MemoryAccessContext(7, 8, 9))
        assert visible == (first,)
        assert visible[0].relation.subject == subject
        assert visible[0].relation.object_ref == object_ref
        assert visible[0].relation.owner == owner

        fake_memory_ref = TypedRef(
            OBJECT_CONCEPT, ctx.memory_interact.space_id, 1)
        invalid = _relation(
            overlay, (predicate, subject, fake_memory_ref), owner,
            scope_id=9, qualifier=32)
        with pytest.raises(CoreIdentityBoundaryError, match="不属于已登记 Core"):
            overlay.add(invalid)
        assert backend.count(MEMORY_OVERLAY_TABLE) == 1
    finally:
        backend.close()


@pytest.mark.parametrize("kind", ["dict", "sqlite"])
def test_memory_space_and_owner_are_part_of_overlay_identity(kind: str):
    """同一 Core 三元组在不同 Memory 空间或 owner 下必须形成不同关系身份。"""
    backend = _backend(kind)
    try:
        ctx = make_train_context(backend)
        refs = _refs(ctx)
        owner_a = OwnerScope(11, 12, 13, VISIBILITY_SESSION)
        owner_b = OwnerScope(11, 12, 14, VISIBILITY_SESSION)
        interact_a = ctx.memory_interact_overlay.add(
            _relation(ctx.memory_interact_overlay, refs, owner_a,
                      scope_id=13, qualifier=1))
        interact_b = ctx.memory_interact_overlay.add(
            _relation(ctx.memory_interact_overlay, refs, owner_b,
                      scope_id=14, qualifier=1))
        read_a = ctx.memory_read_overlay.add(
            _relation(ctx.memory_read_overlay, refs, owner_a,
                      scope_id=13, qualifier=1))

        assert len({
            interact_a.identity_hash,
            interact_b.identity_hash,
            read_a.identity_hash,
        }) == 3
        assert ctx.memory_interact_overlay.query(
            access=MemoryAccessContext(11, 12, 13)) == (interact_a,)
        assert ctx.memory_read_overlay.query(
            access=MemoryAccessContext(11, 12, 13)) == (read_a,)
        assert backend.count(MEMORY_OVERLAY_TABLE) == 3
    finally:
        backend.close()


@pytest.mark.parametrize("kind", ["dict", "sqlite"])
def test_overlay_visibility_is_hierarchical_and_cross_user_closed(kind: str):
    """GLOBAL/TENANT/USER/SESSION 逐级可见，其他用户和其他 session 不得读取私有关系。"""
    backend = _backend(kind)
    try:
        ctx = make_train_context(backend)
        overlay = ctx.memory_interact_overlay
        refs = _refs(ctx)
        owners = (
            GLOBAL_OWNER_SCOPE,
            OwnerScope(21, visibility=VISIBILITY_TENANT),
            OwnerScope(21, 22, visibility=VISIBILITY_USER),
            OwnerScope(21, 22, 23, VISIBILITY_SESSION),
        )
        entries = tuple(
            overlay.add(_relation(
                overlay, refs, owner, scope_id=100 + index,
                qualifier=index))
            for index, owner in enumerate(owners)
        )

        assert overlay.query(access=MemoryAccessContext()) == entries[:1]
        assert overlay.query(access=MemoryAccessContext(21)) == entries[:2]
        assert overlay.query(access=MemoryAccessContext(21, 22)) == entries[:3]
        assert overlay.query(
            access=MemoryAccessContext(21, 22, 23)) == entries
        assert overlay.query(
            access=MemoryAccessContext(21, 99, 23)) == entries[:2]
        assert overlay.query(
            access=MemoryAccessContext(21, 22, 99)) == entries[:3]
        assert overlay.read(
            entries[3].identity_hash,
            access=MemoryAccessContext(21, 22, 99),
        ) is None
        with pytest.raises(MemoryOverlayQueryError, match="必须显式提供"):
            overlay.query(access=None)
    finally:
        backend.close()


@pytest.mark.parametrize("kind", ["dict", "sqlite"])
def test_memory_null_session_is_exact_not_wildcard(kind: str):
    """session_id=None 只能命中阅读层 NULL，不能退化成不加过滤条件。"""
    backend = _backend(kind)
    try:
        ctx = make_train_context(backend)
        memory = ctx.memory_read
        memory.put(memory.new_local_id(), 101, session_id=None)
        memory.put(memory.new_local_id(), 102, session_id=31)
        memory.put(memory.new_local_id(), 103, session_id=32)

        assert [row["content_hash"] for row in memory.query_by_session(None)] == [101]
        assert [row["content_hash"] for row in memory.query_by_session(31)] == [102]
        assert [row["content_hash"] for row in memory.query_by_session(32)] == [103]
        with pytest.raises(ValueError, match="严格正整数"):
            memory.query_by_session(0)
    finally:
        backend.close()


@pytest.mark.parametrize("kind", ["dict", "sqlite"])
def test_overlay_roundtrips_through_authoritative_dump(kind: str, tmp_path):
    """终 dump/load 后 Memory 空间、owner、scope、端点和限定项必须完整恢复。"""
    first_backend = _backend(kind)
    try:
        first = make_train_context(first_backend)
        refs = _refs(first)
        owner = OwnerScope(41, 42, 43, VISIBILITY_SESSION)
        expected = first.memory_interact_overlay.add(
            _relation(first.memory_interact_overlay, refs, owner,
                      scope_id=43, qualifier=77))
        dump_run(
            first_backend,
            str(tmp_path),
            "m02_overlay",
            spaces=[
                first.space_id,
                first.memory_read.space_id,
                first.memory_interact.space_id,
            ],
            tables=DUMP_TABLES,
        )
    finally:
        first_backend.close()

    second_backend = _backend(kind)
    try:
        second = make_train_context(second_backend)
        assert load_run(
            second_backend, str(tmp_path), "m02_overlay") == [1, 2, 3]
        restored = second.memory_interact_overlay.query(
            access=MemoryAccessContext(41, 42, 43))
        assert restored == (expected,)
        assert restored[0].relation.owner == owner
        assert restored[0].relation.scope.owner == owner
        assert restored[0].relation.qualifiers == (77,)
        assert MEMORY_OVERLAY_TABLE in DUMP_TABLES
    finally:
        second_backend.close()


def test_evaluation_clone_preserves_acl_and_uses_another_session():
    """V-06 clone 保留 host 私有关系但评测 session 不可见，clone 写入也不回流 host。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        refs = _refs(ctx)
        host_owner = OwnerScope(51, 52, 53, VISIBILITY_SESSION)
        host_entry = ctx.memory_interact_overlay.add(
            _relation(ctx.memory_interact_overlay, refs, host_owner,
                      scope_id=53, qualifier=1))
        cloned_backend = clone_backend(backend)
        try:
            cloned = clone_train_context(ctx, cloned_backend, label="m02")
            eval_owner = cloned.scope_owner
            eval_access = MemoryAccessContext(
                eval_owner.tenant_id,
                eval_owner.user_id,
                eval_owner.session_id,
            )
            assert eval_owner.session_id != host_owner.session_id
            assert cloned.memory_interact_overlay.query(
                access=eval_access) == ()
            assert cloned.memory_interact_overlay.query(
                access=MemoryAccessContext(51, 52, 53)) == (host_entry,)

            cloned_entry = cloned.memory_interact_overlay.add(
                _relation(
                    cloned.memory_interact_overlay,
                    refs,
                    eval_owner,
                    scope_id=eval_owner.session_id,
                    qualifier=2,
                ))
            assert cloned.memory_interact_overlay.query(
                access=eval_access) == (cloned_entry,)
            assert backend.count(MEMORY_OVERLAY_TABLE) == 1
            assert cloned_backend.count(MEMORY_OVERLAY_TABLE) == 2
        finally:
            cloned_backend.close()
    finally:
        backend.close()


@pytest.mark.parametrize("kind", ["dict", "sqlite"])
@pytest.mark.parametrize("corruption", ["owner", "duplicate"])
def test_overlay_corruption_fails_closed(kind: str, corruption: str):
    """owner 漂移和重复主记录不得被运行期缓存或 ACL 过滤掩盖。"""
    backend = _backend(kind)
    try:
        ctx = make_train_context(backend)
        refs = _refs(ctx)
        owner = OwnerScope(61, 62, 63, VISIBILITY_SESSION)
        ctx.memory_interact_overlay.add(
            _relation(ctx.memory_interact_overlay, refs, owner,
                      scope_id=63, qualifier=1))
        snapshot = backend.snapshot()
        rows = snapshot[MEMORY_OVERLAY_TABLE]
        if corruption == "owner":
            rows[0]["owner_user_id"] = 99
            access = MemoryAccessContext(61, 99, 63)
        else:
            rows.append(copy.deepcopy(rows[0]))
            access = MemoryAccessContext(61, 62, 63)
        backend.load_snapshot(snapshot)
        ctx.memory_interact_overlay.clear_runtime_caches()

        with pytest.raises(MemoryOverlayIntegrityError):
            ctx.memory_interact_overlay.query(access=access)
    finally:
        backend.close()


def test_overlay_table_is_core_append_only():
    """Memory overlay 是可恢复核心记录，禁止原地改 owner 或删除关系。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        entry = ctx.memory_interact_overlay.add(
            _relation(
                ctx.memory_interact_overlay,
                _refs(ctx),
                GLOBAL_OWNER_SCOPE,
                scope_id=1,
                qualifier=1,
            ))
        with pytest.raises(AppendOnlyViolation):
            backend.update(
                MEMORY_OVERLAY_TABLE,
                where={"identity_hash": entry.identity_hash},
                set_={"owner_user_id": 7},
            )
        with pytest.raises(AppendOnlyViolation):
            backend.delete(
                MEMORY_OVERLAY_TABLE,
                where={"identity_hash": entry.identity_hash},
            )
    finally:
        backend.close()
