"""B-05 scope、assertion、逻辑时钟和身份索引对抗测试。"""
from __future__ import annotations

import pytest

from pure_integer_ai.cognition.shared.identity import (
    CorpusVersion,
    CurriculumVersion,
    GLOBAL_OWNER_SCOPE,
    ParserVersion,
    PrimitiveVersion,
    SourceRef,
    VersionBundle,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    CLOCK_OBSERVATION,
    LogicalClock,
    LogicalClockIdentity,
    LogicalTimestamp,
    concept_assertion,
    document_scope,
    episode_scope,
    query_scope,
)
from pure_integer_ai.cognition.shared.scoped_persistence import ScopedIdentityStore
from pure_integer_ai.storage import bootstrap
from pure_integer_ai.storage.assertion_identity import (
    IDENTITY_ASSERTION,
    IDENTITY_HEADER_TABLE,
    IDENTITY_PART_TABLE,
    IDENTITY_SCOPE,
    IDENTITY_TIMESTAMP,
    ExternalIdentityKey,
    IdentityCollisionError,
    IdentityIncompleteError,
    IntegerIdentityRegistry,
    LegacyAssertionAmbiguity,
    SupersedeConflictError,
)
from pure_integer_ai.storage.assertion_record import (
    ASSERTION_QUALIFIER_TABLE,
    ASSERTION_RECORD_TABLE,
)
from pure_integer_ai.storage.backend import DictBackend, SQLiteBackend
from pure_integer_ai.storage.telemetry import (
    BackendTelemetryCollector,
    collect_backend_telemetry,
)
from pure_integer_ai.storage.discipline import AppendOnlyViolation
from pure_integer_ai.storage.edge_store import (
    EPI_CUE,
    EPI_STRUCTURED,
    SOURCE_BARE_TEXT,
    SOURCE_QA,
    EdgeStore,
)
from pure_integer_ai.storage.edge_types import EDGE_CAUSES
from pure_integer_ai.training.cursor import DUMP_TABLES, dump_run, load_run
from pure_integer_ai.experiments.collection import (
    CollectedItem,
    InMemorySource,
    collect_corpus,
)
from pure_integer_ai.experiments.corpus_identity import (
    assign_corpus_source_refs,
    ensure_item_scope,
)
from pure_integer_ai.experiments.formal_train import (
    DefaultRoundRunner,
    _build_space_ctx,
    make_train_context,
)
from pure_integer_ai.cognition.shared.types import (
    InputPayload,
    Segment,
    STAGE_TRAINING,
)
from pure_integer_ai.cognition.process.reward_propagate import _fetch_edge_row
from pure_integer_ai.cognition.understanding.observe import observe


def _versions(*, parser: int = 2) -> VersionBundle:
    return VersionBundle(
        CorpusVersion(1),
        ParserVersion(parser),
        PrimitiveVersion(3),
        CurriculumVersion(4),
    )


def _source(*, source_id: int = 10, document_id: int = 20,
            parser: int = 2) -> SourceRef:
    return SourceRef(
        SOURCE_BARE_TEXT,
        source_id,
        document_id,
        GLOBAL_OWNER_SCOPE,
        _versions(parser=parser),
    )


def _backend(kind: str):
    backend = DictBackend() if kind == "dict" else SQLiteBackend(":memory:")
    bootstrap(backend)
    return backend


def test_scope_roundtrip_preserves_source_parent_owner_and_versions():
    document = document_scope(_source())
    episode = episode_scope(7, parent=document)
    query = query_scope(9, parent=episode)
    restored = type(query).from_stable_key(query.stable_key())
    assert restored == query
    assert restored.parent == episode
    assert restored.source == document.source


def test_scope_rejects_source_and_parent_isolation_mismatch():
    document = document_scope(_source(parser=2))
    changed_parser = _source(parser=3)
    with pytest.raises(ValueError, match="versions"):
        episode_scope(1, parent=document, source=changed_parser)


def test_logical_clock_is_owned_and_only_restores_forward():
    first_scope = document_scope(_source(document_id=1))
    second_scope = document_scope(_source(document_id=2))
    first_id = LogicalClockIdentity(first_scope, CLOCK_OBSERVATION)
    second_id = LogicalClockIdentity(second_scope, CLOCK_OBSERVATION)
    clock = LogicalClock(first_id)
    first = clock.advance()
    third = clock.advance(2)
    assert first.seq == 1
    assert third.seq == 3
    with pytest.raises(ValueError, match="其他 clock"):
        clock.restore(LogicalTimestamp(second_id, 4))
    with pytest.raises(ValueError, match="倒退"):
        clock.restore(first)


def test_assertion_identity_separates_source_parser_scope_and_occurrence():
    endpoint_a = (1, 11)
    endpoint_b = (1, 12)
    scope_a = document_scope(_source(source_id=10, parser=2))
    scope_b = document_scope(_source(source_id=11, parser=2))
    scope_c = document_scope(_source(source_id=10, parser=3))
    base = concept_assertion(
        EDGE_CAUSES, endpoint_a, endpoint_b,
        scope=scope_a,
        provenance_kind=SOURCE_BARE_TEXT,
        epistemic_origin=EPI_STRUCTURED,
        qualifiers=(0, 1, 2),
    )
    assertions = {
        base,
        concept_assertion(
            EDGE_CAUSES, endpoint_a, endpoint_b,
            scope=scope_b,
            provenance_kind=SOURCE_BARE_TEXT,
            epistemic_origin=EPI_STRUCTURED,
            qualifiers=(0, 1, 2),
        ),
        concept_assertion(
            EDGE_CAUSES, endpoint_a, endpoint_b,
            scope=scope_c,
            provenance_kind=SOURCE_BARE_TEXT,
            epistemic_origin=EPI_STRUCTURED,
            qualifiers=(0, 1, 2),
        ),
        concept_assertion(
            EDGE_CAUSES, endpoint_a, endpoint_b,
            scope=scope_a,
            provenance_kind=SOURCE_BARE_TEXT,
            epistemic_origin=EPI_CUE,
            qualifiers=(0, 1, 2),
        ),
        concept_assertion(
            EDGE_CAUSES, endpoint_a, endpoint_b,
            scope=scope_a,
            provenance_kind=SOURCE_BARE_TEXT,
            epistemic_origin=EPI_STRUCTURED,
            qualifiers=(0, 3, 4),
        ),
    }
    assert len(assertions) == 5
    assert type(base).from_stable_key(base.stable_key()) == base


def test_corpus_identity_keeps_equal_documents_as_distinct_occurrences():
    first = CollectedItem(tokens=["同", "文"])
    second = CollectedItem(tokens=["同", "文"])
    assign_corpus_source_refs((first, second), source_namespace="fixture-a")
    assert first.source_ref is not None
    assert second.source_ref is not None
    assert first.source_ref.source_id == second.source_ref.source_id
    assert first.source_ref.document_id == 0
    assert second.source_ref.document_id == 1

    backend = _backend("dict")
    store = ScopedIdentityStore(backend)
    first_scope, first_hash = ensure_item_scope(first, store)
    second_scope, second_hash = ensure_item_scope(second, store)
    assert first_scope != second_scope
    assert first_hash != second_hash


def test_collection_source_namespace_separates_equal_text_sources():
    report = collect_corpus([
        InMemorySource([CollectedItem(tokens=["同", "文"])], name="source-a"),
        InMemorySource([CollectedItem(tokens=["同", "文"])], name="source-b"),
    ])
    assert len(report.items) == 2
    assert report.items[0].source_ref is not None
    assert report.items[1].source_ref is not None
    assert report.items[0].source_ref != report.items[1].source_ref


def test_anonymous_item_identity_is_stable_across_backends():
    first = CollectedItem(tokens=["跨", "进程"])
    second = CollectedItem(tokens=["跨", "进程"])
    assign_corpus_source_refs((first,), source_namespace="stable-source")
    assign_corpus_source_refs((second,), source_namespace="stable-source")
    first_backend = _backend("dict")
    second_backend = _backend("sqlite")
    try:
        first_scope, first_hash = ensure_item_scope(
            first, ScopedIdentityStore(first_backend))
        second_scope, second_hash = ensure_item_scope(
            second, ScopedIdentityStore(second_backend))
        assert first_scope == second_scope
        assert first_hash == second_hash
    finally:
        first_backend.close()
        second_backend.close()


def test_identity_hash_cache_preserves_strict_integer_validation():
    """纯函数 hash 可复用，但等值的布尔伪整数仍必须被拒绝。"""
    backend = _backend("dict")
    try:
        registry = IntegerIdentityRegistry(backend)
        first = registry.identity_hash(IDENTITY_SCOPE, (1, 2, 3))
        assert registry.identity_hash(IDENTITY_SCOPE, (1, 2, 3)) == first
        with pytest.raises((TypeError, ValueError)):
            registry.identity_hash(IDENTITY_SCOPE, (True, 2, 3))
    finally:
        backend.close()


def test_convenience_observe_resumes_scoped_clock_instead_of_resetting():
    backend = _backend("dict")
    ctx = make_train_context(backend)
    item = CollectedItem(tokens=["甲", "乙"])
    assign_corpus_source_refs((item,), source_namespace="clock-source")
    scope, _scope_hash = ensure_item_scope(item, ctx.scoped_identity_store)
    raw = InputPayload(
        segments=[Segment(seg_id=0, tokens=["甲", "乙"])],
        source=SOURCE_BARE_TEXT,
        stage=STAGE_TRAINING,
        scope_identity=scope,
    )
    sctx = _build_space_ctx(ctx)
    observe(raw, sctx, concept_index=ctx.concept_index,
            work_memory=ctx.work_memory)
    observe(raw, sctx, concept_index=ctx.concept_index,
            work_memory=ctx.work_memory)
    resumed = ctx.scoped_identity_store.resume_clock(
        LogicalClockIdentity(scope, CLOCK_OBSERVATION))
    assert resumed.current_seq == 4
    timestamps = backend.select(IDENTITY_HEADER_TABLE, where={
        "identity_kind": IDENTITY_TIMESTAMP,
    })
    assert sorted(row["ordinal"] for row in timestamps) == [2, 4]


def test_observe_registers_same_endpoint_causes_per_episode_scope():
    backend = _backend("dict")
    ctx = make_train_context(backend)
    item = CollectedItem(
        tokens=["雨", "地湿"],
        causal_pairs=[(0, 1)],
        source=SOURCE_BARE_TEXT,
    )
    assign_corpus_source_refs((item,), source_namespace="causes-source")
    runner = DefaultRoundRunner()
    runner.run_round(ctx, item, STAGE_TRAINING, 10)
    runner.run_round(ctx, item, STAGE_TRAINING, 11)
    records = backend.select(ASSERTION_RECORD_TABLE)
    assertions = [
        ctx.scoped_identity_store.load_assertion(row["identity_hash"])
        for row in records
    ]
    assert len(assertions) == 2
    assert len({assertion.scope for assertion in assertions}) == 2
    assert {assertion.qualifiers for assertion in assertions} == {(0, 0, 1)}
    assert all(assertion.scope.parent is not None for assertion in assertions)


@pytest.mark.parametrize("kind", ["dict", "sqlite"])
def test_registry_roundtrip_supersede_and_snapshot(kind: str):
    backend = _backend(kind)
    try:
        store = ScopedIdentityStore(backend)
        scope = document_scope(_source())
        old = concept_assertion(
            EDGE_CAUSES, (1, 1), (1, 2),
            scope=scope,
            provenance_kind=SOURCE_BARE_TEXT,
            epistemic_origin=EPI_STRUCTURED,
            content_version=1,
            qualifiers=(0, 1, 2),
        )
        new = concept_assertion(
            EDGE_CAUSES, (1, 1), (1, 2),
            scope=scope,
            provenance_kind=SOURCE_BARE_TEXT,
            epistemic_origin=EPI_STRUCTURED,
            content_version=2,
            qualifiers=(0, 1, 2),
        )
        clock_id = LogicalClockIdentity(scope, CLOCK_OBSERVATION)
        timestamp = LogicalClock(clock_id).advance()
        old_hash = store.register_assertion(old)
        assert store.register_assertion(old) == old_hash
        assert store.load_assertion(old_hash) == old
        event_hash = store.supersede(old, new, timestamp)
        assert event_hash > 0
        events = store.registry.superseding_events(old_hash)
        assert len(events) == 1

        snapshot = backend.snapshot()
        restored_backend = _backend(kind)
        try:
            restored_backend.load_snapshot(snapshot)
            restored = ScopedIdentityStore(restored_backend)
            assert restored.load_assertion(old_hash) == old
            assert restored.registry.superseding_events(old_hash) == events
            resumed = restored.resume_clock(clock_id)
            assert resumed.current_seq == timestamp.seq
        finally:
            restored_backend.close()
    finally:
        backend.close()


@pytest.mark.parametrize("kind", ["dict", "sqlite"])
def test_identity_registry_roundtrips_through_authoritative_cursor_dump(
        kind: str, tmp_path):
    first_backend = _backend(kind)
    try:
        store = ScopedIdentityStore(first_backend)
        scope = document_scope(_source())
        assertion = concept_assertion(
            EDGE_CAUSES, (1, 1), (1, 2),
            scope=scope,
            provenance_kind=SOURCE_BARE_TEXT,
            epistemic_origin=EPI_STRUCTURED,
        )
        assertion_hash = store.register_assertion(assertion)
        dump_run(
            first_backend,
            str(tmp_path),
            "run_a",
            spaces=[1],
            tables=DUMP_TABLES,
        )
    finally:
        first_backend.close()

    assert (tmp_path / "run_a" / "global_identity.dump").is_file()
    second_backend = _backend(kind)
    try:
        assert load_run(second_backend, str(tmp_path), "run_a") == [1]
        restored = ScopedIdentityStore(second_backend)
        assert restored.load_assertion(assertion_hash) == assertion
    finally:
        second_backend.close()


@pytest.mark.parametrize("kind", ["dict", "sqlite"])
def test_normalized_assertion_preserves_legacy_key_hash_and_reduces_rows(kind):
    """正规化编码必须保持旧完整键和 hash，同时不再平铺 assertion parts。"""
    assertion = concept_assertion(
        EDGE_CAUSES, (1, 1), (1, 2),
        scope=episode_scope(7, parent=document_scope(_source())),
        provenance_kind=SOURCE_BARE_TEXT,
        epistemic_origin=EPI_STRUCTURED,
        content_version=3,
        qualifiers=(5, 3, 5),
    )
    normalized_backend = _backend(kind)
    legacy_backend = _backend(kind)
    try:
        normalized = ScopedIdentityStore(normalized_backend)
        normalized_hash = normalized.register_assertion(assertion)
        assert normalized.registry.read_key(
            IDENTITY_ASSERTION, normalized_hash) == assertion.stable_key()
        assert normalized_backend.select(IDENTITY_PART_TABLE, where={
            "identity_kind": IDENTITY_ASSERTION,
            "identity_hash": normalized_hash,
        }) == []
        assert len(normalized_backend.select(ASSERTION_RECORD_TABLE, where={
            "identity_hash": normalized_hash,
        })) == 1
        assert normalized_backend.select(IDENTITY_HEADER_TABLE, where={
            "identity_kind": IDENTITY_ASSERTION,
            "identity_hash": normalized_hash,
        }) == []
        assert normalized_backend.select(ASSERTION_QUALIFIER_TABLE, where={
            "identity_hash": normalized_hash,
        }) == []
        record = normalized_backend.select(ASSERTION_RECORD_TABLE, where={
            "identity_hash": normalized_hash,
        })[0]
        assert tuple(record[f"qualifier_{index}"] for index in range(3)) == (
            5, 3, 5)
        assert 1 < len(assertion.stable_key())

        legacy = IntegerIdentityRegistry(legacy_backend)
        legacy_scope_hash = legacy.register(
            IDENTITY_SCOPE, assertion.scope.stable_key())
        legacy_hash = legacy.register(
            IDENTITY_ASSERTION,
            assertion.stable_key(),
            parent_hash=legacy_scope_hash,
        )
        assert legacy_hash == normalized_hash
        assert legacy.read_key(
            IDENTITY_ASSERTION, legacy_hash) == assertion.stable_key()
    finally:
        normalized_backend.close()
        legacy_backend.close()


@pytest.mark.parametrize("corruption", ["missing", "gap", "duplicate"])
def test_normalized_assertion_rejects_qualifier_corruption(corruption):
    """限定项缺失、ordinal 断裂或重复都不能被排序和 limit 掩盖。"""
    backend = _backend("dict")
    store = ScopedIdentityStore(backend)
    assertion = concept_assertion(
        EDGE_CAUSES, (1, 1), (1, 2),
        scope=document_scope(_source()),
        provenance_kind=SOURCE_BARE_TEXT,
        qualifiers=(11, 22, 33, 44, 55),
    )
    assertion_hash = store.register_assertion(assertion)
    snapshot = backend.snapshot()
    qualifiers = snapshot[ASSERTION_QUALIFIER_TABLE]
    if corruption == "missing":
        snapshot[ASSERTION_QUALIFIER_TABLE] = qualifiers[:1]
    elif corruption == "gap":
        qualifiers[1]["ordinal"] = 5
    else:
        qualifiers.append(dict(qualifiers[0]))

    restored_backend = _backend("dict")
    restored_backend.load_snapshot(snapshot)
    with pytest.raises(IdentityIncompleteError, match="qualifier"):
        IntegerIdentityRegistry(restored_backend).read_key(
            IDENTITY_ASSERTION, assertion_hash)


def test_normalized_assertion_accepts_self_header_and_rejects_mixed_encoding():
    """主记录可自带 header；旧 header 兼容但不得与 flat parts 混存。"""
    assertion = concept_assertion(
        EDGE_CAUSES, (1, 1), (1, 2),
        scope=document_scope(_source()),
        provenance_kind=SOURCE_BARE_TEXT,
    )

    header_only_backend = _backend("dict")
    header_only_store = ScopedIdentityStore(header_only_backend)
    header_only_hash = header_only_store.register_assertion(assertion)
    header_only_snapshot = header_only_backend.snapshot()
    header_record = header_only_snapshot[ASSERTION_RECORD_TABLE][0]
    header_only_snapshot[IDENTITY_HEADER_TABLE].append({
        "identity_kind": IDENTITY_ASSERTION,
        "identity_hash": header_only_hash,
        "key_size": len(assertion.stable_key()),
        "parent_hash": header_record["scope_hash"],
        "ordinal": header_record["assertion_role"],
    })
    header_only_snapshot[ASSERTION_RECORD_TABLE] = []
    header_only_restored = _backend("dict")
    header_only_restored.load_snapshot(header_only_snapshot)
    with pytest.raises(IdentityIncompleteError, match="没有完整键载荷"):
        IntegerIdentityRegistry(header_only_restored).read_key(
            IDENTITY_ASSERTION, header_only_hash)

    record_only_backend = _backend("dict")
    record_only_store = ScopedIdentityStore(record_only_backend)
    record_only_hash = record_only_store.register_assertion(assertion)
    metadata = record_only_store.registry.metadata(
        IDENTITY_ASSERTION, record_only_hash)
    assert metadata.key_size == len(assertion.stable_key())
    assert metadata.parent_hash > 0
    assert metadata.ordinal == 0

    mixed_backend = _backend("dict")
    mixed_store = ScopedIdentityStore(mixed_backend)
    mixed_hash = mixed_store.register_assertion(assertion)
    mixed_record = mixed_backend.select(ASSERTION_RECORD_TABLE, where={
        "identity_hash": mixed_hash,
    })[0]
    mixed_backend.insert(IDENTITY_HEADER_TABLE, {
        "identity_kind": IDENTITY_ASSERTION,
        "identity_hash": mixed_hash,
        "key_size": len(assertion.stable_key()),
        "parent_hash": mixed_record["scope_hash"],
        "ordinal": mixed_record["assertion_role"],
    })
    mixed_backend.insert(IDENTITY_PART_TABLE, {
        "identity_kind": IDENTITY_ASSERTION,
        "identity_hash": mixed_hash,
        "part_index": 0,
        "part_value": assertion.stable_key()[0],
    })
    mixed_store.clear_runtime_caches()
    with pytest.raises(IdentityIncompleteError, match="同时存在"):
        mixed_store.load_assertion(mixed_hash)


@pytest.mark.parametrize("kind", ["dict", "sqlite"])
def test_normalized_assertion_reads_legacy_qualifier_codec_and_header(kind):
    """旧 header 加全子表限定项格式必须恢复同一完整键和 hash。"""
    backend = _backend(kind)
    restored_backend = _backend(kind)
    try:
        store = ScopedIdentityStore(backend)
        assertion = concept_assertion(
            EDGE_CAUSES, (1, 1), (1, 2),
            scope=document_scope(_source()),
            provenance_kind=SOURCE_BARE_TEXT,
            qualifiers=(0, -2, 7, 9),
        )
        assertion_hash = store.register_assertion(assertion)
        snapshot = backend.snapshot()
        record = snapshot[ASSERTION_RECORD_TABLE][0]
        for index in range(3):
            record.pop(f"qualifier_{index}")
        snapshot[ASSERTION_QUALIFIER_TABLE] = [
            {
                "identity_hash": assertion_hash,
                "ordinal": ordinal,
                "qualifier_value": value,
            }
            for ordinal, value in enumerate(assertion.qualifiers)
        ]
        snapshot[IDENTITY_HEADER_TABLE].append({
            "identity_kind": IDENTITY_ASSERTION,
            "identity_hash": assertion_hash,
            "key_size": len(assertion.stable_key()),
            "parent_hash": record["scope_hash"],
            "ordinal": record["assertion_role"],
        })
        restored_backend.load_snapshot(snapshot)
        restored = ScopedIdentityStore(restored_backend)
        assert restored.load_assertion(assertion_hash) == assertion
        assert restored.registry.find(
            IDENTITY_ASSERTION, assertion.stable_key()) == assertion_hash
    finally:
        backend.close()
        restored_backend.close()


def test_normalized_assertion_rejects_conflicting_legacy_header():
    """旧通用 header 与 assertion 主记录的 parent 或角色冲突时必须拒绝。"""
    backend = _backend("dict")
    store = ScopedIdentityStore(backend)
    assertion = concept_assertion(
        EDGE_CAUSES, (1, 1), (1, 2),
        scope=document_scope(_source()),
        provenance_kind=SOURCE_BARE_TEXT,
    )
    assertion_hash = store.register_assertion(assertion)
    snapshot = backend.snapshot()
    record = snapshot[ASSERTION_RECORD_TABLE][0]
    snapshot[IDENTITY_HEADER_TABLE].append({
        "identity_kind": IDENTITY_ASSERTION,
        "identity_hash": assertion_hash,
        "key_size": len(assertion.stable_key()),
        "parent_hash": record["scope_hash"],
        "ordinal": record["assertion_role"] + 1,
    })
    restored_backend = _backend("dict")
    restored_backend.load_snapshot(snapshot)
    with pytest.raises(IdentityIncompleteError, match="ordinal"):
        IntegerIdentityRegistry(restored_backend).read_key(
            IDENTITY_ASSERTION, assertion_hash)


def test_normalized_assertion_rejects_inline_codec_corruption():
    """部分缺列、未用槽非零和回退到内联 ordinal 的 overflow 均失败。"""
    assertion = concept_assertion(
        EDGE_CAUSES, (1, 1), (1, 2),
        scope=document_scope(_source()),
        provenance_kind=SOURCE_BARE_TEXT,
        qualifiers=(0, -2, 7, 9),
    )
    for corruption in ("partial", "unused", "overflow"):
        backend = _backend("dict")
        store = ScopedIdentityStore(backend)
        assertion_hash = store.register_assertion(assertion)
        snapshot = backend.snapshot()
        record = snapshot[ASSERTION_RECORD_TABLE][0]
        if corruption == "partial":
            record.pop("qualifier_2")
        elif corruption == "unused":
            record["qualifier_size"] = 2
            record["qualifier_2"] = 8
            snapshot[ASSERTION_QUALIFIER_TABLE] = []
        else:
            snapshot[ASSERTION_QUALIFIER_TABLE][0]["ordinal"] = 0
        restored_backend = _backend("dict")
        restored_backend.load_snapshot(snapshot)
        with pytest.raises(IdentityIncompleteError, match="qualifier"):
            IntegerIdentityRegistry(restored_backend).read_key(
                IDENTITY_ASSERTION, assertion_hash)


def test_external_record_without_header_requires_explicit_self_heading():
    """未获 self-headed 授权的外部 kind 缺通用 header 时仍视为损坏。"""
    backend = _backend("dict")
    registry = IntegerIdentityRegistry(backend)
    identity_kind = 99
    key = (1, 2, 3)
    identity_hash = registry.identity_hash(identity_kind, key)
    registry.register_external_key_resolver(
        identity_kind,
        lambda candidate_hash: (
            ExternalIdentityKey(key, parent_hash=7, ordinal=0)
            if candidate_hash == identity_hash else None
        ),
    )
    with pytest.raises(IdentityIncompleteError, match="无 header"):
        registry.read_key(identity_kind, identity_hash)


def test_normalized_assertion_fails_closed_on_hash_collision():
    """正规化记录仍以完整键判碰撞，不能因主表较窄而接受同 hash 异义。"""
    backend = _backend("dict")
    store = ScopedIdentityStore(backend, hasher=_ConstantHasher())
    scope = document_scope(_source())
    first = concept_assertion(
        EDGE_CAUSES, (1, 1), (1, 2), scope=scope,
        provenance_kind=SOURCE_BARE_TEXT)
    second = concept_assertion(
        EDGE_CAUSES, (1, 1), (1, 3), scope=scope,
        provenance_kind=SOURCE_BARE_TEXT)
    assert store.register_assertion(first) == 7
    with pytest.raises(IdentityCollisionError):
        store.register_assertion(second)


def test_assertion_runtime_cache_does_not_change_normalized_snapshot():
    """清空并重建运行期缓存不得向正规化物理表追加任何行。"""
    backend = _backend("dict")
    store = ScopedIdentityStore(backend)
    assertion = concept_assertion(
        EDGE_CAUSES, (1, 1), (1, 2),
        scope=document_scope(_source()),
        provenance_kind=SOURCE_BARE_TEXT,
        qualifiers=(1, 2, 3),
    )
    assertion_hash = store.register_assertion(assertion)
    before = backend.snapshot()
    store.clear_runtime_caches()
    assert store.load_assertion(assertion_hash) == assertion
    assert backend.snapshot() == before


def test_fresh_assertion_namespace_avoids_redundant_select_readbacks():
    """空命名空间批量首写只做一次空表声明，不为每条 assertion 重复读十八次。"""
    backend = _backend("dict")
    store = ScopedIdentityStore(backend)
    scope = document_scope(_source())
    store.register_scope(scope)
    assertion = concept_assertion(
        EDGE_CAUSES, (1, 1), (1, 2), scope=scope,
        provenance_kind=SOURCE_BARE_TEXT,
        qualifiers=(1, 2),
    )
    collector = BackendTelemetryCollector()
    with collect_backend_telemetry(collector):
        store.register_assertion(assertion)
    query_rows = [
        row for row in collector.to_json()["by_dimension"]["query"]
        if row["value"] == "assertion.register"
    ]
    assert len(query_rows) == 1
    operations = query_rows[0]["operations"]
    assert sum(row["calls"] for row in operations
               if row["operation"] == "count") == 4
    assert all(row["operation"] != "select" for row in operations)


def test_store_created_before_load_uses_strict_existing_assertion_path():
    """store 先构造后载入时不得误把恢复数据当作空命名空间快路。"""
    source_backend = _backend("dict")
    source_store = ScopedIdentityStore(source_backend)
    assertion = concept_assertion(
        EDGE_CAUSES, (1, 1), (1, 2),
        scope=document_scope(_source()),
        provenance_kind=SOURCE_BARE_TEXT,
        qualifiers=(7,),
    )
    assertion_hash = source_store.register_assertion(assertion)
    snapshot = source_backend.snapshot()

    restored_backend = _backend("dict")
    restored_store = ScopedIdentityStore(restored_backend)
    restored_backend.load_snapshot(snapshot)
    assert restored_store.register_assertion(assertion) == assertion_hash
    assert restored_store.load_assertion(assertion_hash) == assertion
    assert restored_backend.snapshot() == snapshot


@pytest.mark.parametrize("kind", ["dict", "sqlite"])
def test_identity_tables_are_append_only(kind: str):
    backend = _backend(kind)
    try:
        store = ScopedIdentityStore(backend)
        scope_hash = store.register_scope(document_scope(_source()))
        with pytest.raises(AppendOnlyViolation):
            backend.update(
                IDENTITY_HEADER_TABLE,
                where={"identity_hash": scope_hash},
                set_={"ordinal": 2},
            )
        with pytest.raises(AppendOnlyViolation):
            backend.delete(
                IDENTITY_PART_TABLE,
                where={"identity_hash": scope_hash},
            )
    finally:
        backend.close()


class _ConstantHasher:
    """对抗测试用固定哈希器，强制制造索引碰撞。"""

    def h63(self, value) -> int:
        return 7


def test_registry_fails_closed_on_hash_collision():
    backend = _backend("dict")
    registry = IntegerIdentityRegistry(backend, hasher=_ConstantHasher())
    registry.register(IDENTITY_SCOPE, document_scope(_source(document_id=1)).stable_key())
    with pytest.raises(IdentityCollisionError):
        registry.register(
            IDENTITY_SCOPE,
            document_scope(_source(document_id=2)).stable_key(),
        )


def test_registry_fails_closed_on_orphan_part_and_duplicate_header():
    backend = _backend("dict")
    registry = IntegerIdentityRegistry(backend, hasher=_ConstantHasher())
    backend.insert(IDENTITY_PART_TABLE, {
        "identity_kind": IDENTITY_SCOPE,
        "identity_hash": 7,
        "part_index": 0,
        "part_value": 1,
    })
    with pytest.raises(IdentityIncompleteError, match="孤儿"):
        registry.register(IDENTITY_SCOPE, (1, 2, 3))

    other = _backend("dict")
    other_registry = IntegerIdentityRegistry(other)
    scope = document_scope(_source())
    scope_hash = other_registry.register(IDENTITY_SCOPE, scope.stable_key())
    metadata = other_registry.metadata(IDENTITY_SCOPE, scope_hash)
    other.insert(IDENTITY_HEADER_TABLE, {
        "identity_kind": metadata.identity_kind,
        "identity_hash": metadata.identity_hash,
        "key_size": metadata.key_size,
        "parent_hash": metadata.parent_hash,
        "ordinal": metadata.ordinal,
    })
    with pytest.raises(IdentityIncompleteError, match="header 数量"):
        other_registry.read_key(IDENTITY_SCOPE, scope_hash)


def test_supersede_rejects_competing_target():
    backend = _backend("dict")
    store = ScopedIdentityStore(backend)
    scope = document_scope(_source())
    old = concept_assertion(
        EDGE_CAUSES, (1, 1), (1, 2), scope=scope,
        provenance_kind=SOURCE_BARE_TEXT)
    first = concept_assertion(
        EDGE_CAUSES, (1, 1), (1, 3), scope=scope,
        provenance_kind=SOURCE_BARE_TEXT)
    second = concept_assertion(
        EDGE_CAUSES, (1, 1), (1, 4), scope=scope,
        provenance_kind=SOURCE_BARE_TEXT)
    clock = LogicalClock(LogicalClockIdentity(scope, CLOCK_OBSERVATION))
    store.supersede(old, first, clock.advance())
    with pytest.raises(SupersedeConflictError):
        store.supersede(old, second, clock.advance())


def test_legacy_edge_migration_requires_unique_wide_row():
    backend = _backend("dict")
    edge_store = EdgeStore(backend)
    scoped = ScopedIdentityStore(backend)
    scope = document_scope(_source())
    edge_store.add(
        space_id_from=1, local_id_from=1,
        space_id_to=1, local_id_to=2,
        edge_type=EDGE_CAUSES,
        source=SOURCE_BARE_TEXT,
        epistemic_origin=EPI_STRUCTURED,
    )
    assertion = scoped.assertion_from_legacy_edge(
        scope=scope,
        subject_ref=(1, 1),
        object_ref=(1, 2),
        relation_kind=EDGE_CAUSES,
    )
    assert assertion is not None
    assert assertion.scope == scope
    edge_store.add(
        space_id_from=1, local_id_from=1,
        space_id_to=1, local_id_to=2,
        edge_type=EDGE_CAUSES,
        source=SOURCE_QA,
        epistemic_origin=EPI_STRUCTURED,
    )
    with pytest.raises(LegacyAssertionAmbiguity):
        scoped.assertion_from_legacy_edge(
            scope=scope,
            subject_ref=(1, 1),
            object_ref=(1, 2),
            relation_kind=EDGE_CAUSES,
        )


@pytest.mark.parametrize("backend_kind", ["dict", "sqlite"])
def test_legacy_edge_read_and_updates_reject_ambiguous_rows_without_mutation(
        backend_kind):
    """同端点不同来源的旧边不可首行命中，也不可被五元宽键批量更新。"""
    backend = _backend(backend_kind)
    edge_store = EdgeStore(backend)
    common = {
        "space_id_from": 1,
        "local_id_from": 1,
        "space_id_to": 1,
        "local_id_to": 2,
        "edge_type": EDGE_CAUSES,
    }
    edge_store.add(
        **common, source=SOURCE_BARE_TEXT,
        epistemic_origin=EPI_STRUCTURED,
    )
    edge_store.add(
        **common, source=SOURCE_QA,
        epistemic_origin=EPI_CUE,
    )
    before = backend.select("edge", where=common)

    operations = (
        lambda: edge_store.get(**common),
        lambda: _fetch_edge_row(
            edge_store, (1, 1, 1, 2, EDGE_CAUSES)),
        lambda: edge_store.set_tier(**common, new_tier=1),
        lambda: edge_store.add_strength(**common, delta=1),
        lambda: edge_store.record_success(**common, success=True),
        lambda: edge_store.record_episode_result(
            **common, sn_delta=1, tn_delta=1, strength_delta=1),
    )
    for operation in operations:
        with pytest.raises(LegacyAssertionAmbiguity):
            operation()
        assert backend.select("edge", where=common) == before


@pytest.mark.parametrize("backend_kind", ["dict", "sqlite"])
def test_unique_legacy_edge_updates_exact_row_with_nullable_columns(backend_kind):
    """唯一旧边即使含空字段，也能在两种后端按完整行精确更新。"""
    backend = _backend(backend_kind)
    edge_store = EdgeStore(backend)
    common = {
        "space_id_from": 1,
        "local_id_from": 1,
        "space_id_to": 1,
        "local_id_to": 2,
        "edge_type": EDGE_CAUSES,
    }
    edge_store.add(
        **common, source=SOURCE_BARE_TEXT,
        epistemic_origin=EPI_STRUCTURED,
    )
    edge_store.record_episode_result(
        **common, sn_delta=1, tn_delta=1, strength_delta=1)
    row = edge_store.get(**common)
    assert row is not None
    assert (row["sn"], row["tn"], row["strength"]) == (1, 1, 2)
