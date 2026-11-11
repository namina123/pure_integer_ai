"""限定项物理打包、旧库兼容及真实 connector 图恢复验证。"""
from __future__ import annotations

import pytest

from pure_integer_ai.cognition.shared.scope_identity import concept_assertion, document_scope
from pure_integer_ai.cognition.shared.scoped_persistence import ScopedIdentityStore
from pure_integer_ai.storage.assertion_identity import (
    IDENTITY_ASSERTION, IdentityCollisionError, IdentityIncompleteError, integer_identity_hash,
    register_assertion_identity_tables,
)
from pure_integer_ai.storage.assertion_record import (
    ASSERTION_QUALIFIER_TABLE, ASSERTION_RECORD_TABLE, AssertionRecordError,
    register_assertion_record_tables,
)
from pure_integer_ai.storage.assertion_qualifier_codec import (
    decode_assertion_qualifiers, encode_qualifier_overflow,
)
from pure_integer_ai.storage.backend import DictBackend, SQLiteBackend
from pure_integer_ai.storage.edge_types import EDGE_CAUSES
from pure_integer_ai.storage.integer_codec import IntegerCodecError
from pure_integer_ai.training.cursor import DUMP_TABLES, dump_run, load_run

from tests.test_scope_assertion_identity import _source


def _backend(path=None, *, read_only=False):
    """只注册身份和断言五表，测试真正的 registry 恢复边界。"""
    backend = (DictBackend() if path is None
               else SQLiteBackend(str(path), read_only=read_only))
    register_assertion_identity_tables(backend)
    register_assertion_record_tables(backend)
    return backend


def _assertion(qualifiers):
    """构造带来源 scope 的完整领域断言，不伪造 hash。"""
    return concept_assertion(
        EDGE_CAUSES, (1, 1), (1, 2), scope=document_scope(_source()),
        provenance_kind=1, epistemic_origin=2, content_version=3,
        qualifiers=qualifiers)


def _physical(data):
    """把人为损坏的字节流装入物理 word，直接针对 decoder 合同。"""
    return ((-2, len(data)), (-1, 1), *tuple(
        (3 + start // 7, int.from_bytes(data[start:start + 7], "little"))
        for start in range(0, len(data), 7)))


def test_packed_fixed_vector_and_threshold():
    """冻结 32 项固定向量，31 项仍保存逐整数 overflow。"""
    values = (91516, 1, -7) + (0,) * 29
    expected = ((-2, 30), (-1, 1), (3, 29), (4, 0), (5, 0), (6, 0), (7, 0))
    assert encode_qualifier_overflow(values) == expected
    assert decode_assertion_qualifiers(32, values[:3], expected) == values
    short = values[:-1]
    assert encode_qualifier_overflow(short) == tuple(enumerate(short[3:], 3))


@pytest.mark.parametrize("values", [
    (), (0,), (-2, 0), (91516, 1, 7), tuple(range(31)), tuple(range(119)),
    (1, 2, 3) + (0, -1, 63, -64, 64, -65) * 32,
    (1, 2, 3) + (1 << 120,) * 29,
    (1, 2, 3) + (-(1 << 80),) + (0,) * 60,
    (0,) * 5000,
])
def test_complete_roundtrip_and_legacy_read(values):
    """覆盖符号、宽整数及超过新写门的历史读取，不删减完整值。"""
    inline = tuple(values[index] if index < len(values) else 0 for index in range(3))
    assert decode_assertion_qualifiers(
        len(values), inline, encode_qualifier_overflow(values)) == values
    assert decode_assertion_qualifiers(
        len(values), (None,) * 3, tuple(enumerate(values))) == values
    assert decode_assertion_qualifiers(
        len(values), inline, tuple(enumerate(values[3:], 3))) == values


def test_wide_integers_without_row_profit_keep_old_representation():
    """元素数达阈值但 word 行数不减少时保持旧表示。"""
    values = (1 << 120,) * 32
    assert encode_qualifier_overflow(values) == tuple(enumerate(values[3:], 3))


@pytest.mark.parametrize("parts", [
    ((-3, 30), (-1, 1), (3, 29)),
    ((-2, 30), (3, 29)),
    ((-1, 1), (3, 29)),
    ((-2, 30), (-1, 2), (3, 29)),
    ((-2, 0), (-1, 1), (3, 29)),
    ((-2, -1), (-1, 1), (3, 29)),
    ((-2, 10**100), (-1, 1), (3, 29)),
    ((-2, 7 * 27), (-1, 1), (3, 29)),
    ((-2, 30), (-1, 1), (3, 29), (4, 0), (5, 0), (6, 0)),
    ((-2, 30), (-1, 1), (3, 29), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0)),
    ((-2, 30), (-1, 1), (3, 29), (4, 0), (4, 0), (6, 0), (7, 0)),
    ((-2, 30), (-1, 1), (3, 29), (4, 0), (6, 0), (7, 0), (8, 0)),
    ((-2, 30), (-1, 1), (3, 29), (4, -1), (5, 0), (6, 0), (7, 0)),
    ((-2, 30), (-1, 1), (3, 29), (4, 1 << 56), (5, 0), (6, 0), (7, 0)),
    ((-2, 30), (-1, 1), (3, 29), (4, 0), (5, 0), (6, 0), (7, 1 << 16)),
    ((-2, 30), (-1, True), (3, 29)),
])
def test_packed_corruption_rejected(parts):
    """marker、范围、序、填充和恶意长度必须在字节恢复之前拒绝。"""
    with pytest.raises(IntegerCodecError):
        decode_assertion_qualifiers(32, (1, 2, 3), parts)


@pytest.mark.parametrize("data", [
    b"\x9d\x00" + b"\x00" * 29,
    b"\x1d\x80\x00" + b"\x00" * 28,
    b"\x1d" + b"\x00" * 28,
    b"\x1c" + b"\x00" * 29,
    b"\x1d" + b"\x00" * 29 + b"\x00",
])
def test_noncanonical_or_incomplete_stream_rejected(data):
    """规范 varint、元素计数、截断和尾随都不能退回旧表示。"""
    with pytest.raises(IntegerCodecError):
        decode_assertion_qualifiers(32, (1, 2, 3), _physical(data))


@pytest.mark.parametrize("size,inline,parts", [
    (True, (0, 0, 0), ()), (-1, (0, 0, 0), ()),
    (0, (0, 0), ()), (0, (0, None, 0), ()),
    (0, (0, 0, True), ()), (1, (2, 3, 0), ()),
    (4, (1, 2, 3), ((4, -2),)),
    (4, (None,) * 3, ((-2, 1), (-1, 1), (3, 1), (4, 0))),
    (31, (1, 2, 3), ((-2, 29), (-1, 1), (3, 28))),
    (32, (None,) * 3, ((-2, 30), (-1, 1), (3, 29))),
])
def test_legacy_inline_and_packed_cannot_mix(size, inline, parts):
    """旧格式的负值与新格式负 ordinal 分离，半内联和小 packed 拒绝。"""
    with pytest.raises(IntegerCodecError):
        decode_assertion_qualifiers(size, inline, parts)


@pytest.mark.parametrize("legacy", ["all_parts", "inline", "packed"])
def test_sqlite_reopen_preserves_hash_role_metadata_and_idempotence(tmp_path, legacy):
    """不同历史表示关闭重开后走真实 registry，幂等登记不迁移物理行。"""
    assertion = _assertion((91516, 1, 7) + tuple(range(116)))
    first = _backend()
    try:
        identity_hash = ScopedIdentityStore(first).register_assertion(assertion, assertion_role=1)
        snapshot = first.snapshot()
    finally:
        first.close()
    if legacy != "packed":
        start = 0 if legacy == "all_parts" else 3
        if start == 0:
            for index in range(3):
                snapshot[ASSERTION_RECORD_TABLE][0][f"qualifier_{index}"] = None
        snapshot[ASSERTION_QUALIFIER_TABLE] = [
            {"identity_hash": identity_hash, "ordinal": index, "qualifier_value": value}
            for index, value in enumerate(assertion.qualifiers[start:], start)]
    path = tmp_path / "assertions.sqlite3"
    backend = _backend(path)
    try:
        backend.load_snapshot(snapshot)
        backend.commit()
    finally:
        backend.close()
    backend = _backend(path, read_only=True)
    try:
        store = ScopedIdentityStore(backend)
        assert store.load_assertion(identity_hash) == assertion
        assert identity_hash == integer_identity_hash(IDENTITY_ASSERTION, assertion.stable_key())
        metadata = store.registry.metadata(IDENTITY_ASSERTION, identity_hash)
        assert metadata.ordinal == 1
        assert metadata.parent_hash == store.assertion_records.read(identity_hash).scope_hash
        assert backend._conn.total_changes == 0
    finally:
        backend.close()
    backend = _backend(path)
    try:
        before = backend.snapshot()
        assert ScopedIdentityStore(backend).register_assertion(assertion, assertion_role=1) == identity_hash
        assert backend.snapshot() == before
    finally:
        backend.close()


@pytest.mark.parametrize("corruption", ["codec", "valid_other_value", "inline", "orphan"])
def test_registry_detects_packed_damage_and_original_hash_mismatch(corruption):
    """经真实 owner 映射损坏；合法但不同的完整内容仍被原 hash 拒绝。"""
    first, second = _backend(), _backend()
    try:
        assertion = _assertion((91516, 1, 7) + (0,) * 29)
        identity_hash = ScopedIdentityStore(first).register_assertion(assertion)
        snapshot = first.snapshot()
        if corruption == "codec":
            next(row for row in snapshot[ASSERTION_QUALIFIER_TABLE] if row["ordinal"] == -1)["qualifier_value"] = 2
        elif corruption == "valid_other_value":
            next(row for row in snapshot[ASSERTION_QUALIFIER_TABLE] if row["ordinal"] == 3)["qualifier_value"] += 2 << 8
        elif corruption == "inline":
            snapshot[ASSERTION_RECORD_TABLE][0]["qualifier_0"] += 1
        else:
            snapshot[ASSERTION_RECORD_TABLE] = []
        second.load_snapshot(snapshot)
        expected_error = (IdentityCollisionError
                          if corruption in {"valid_other_value", "inline"}
                          else IdentityIncompleteError)
        with pytest.raises(expected_error):
            ScopedIdentityStore(second).load_assertion(identity_hash)
    finally:
        second.close()
        first.close()


def test_packed_write_does_not_bypass_logical_qualifier_limit():
    """高度可压缩也不能绕过 owner 的 4096 项写入门。"""
    backend = _backend()
    try:
        with pytest.raises(AssertionRecordError, match="预算"):
            ScopedIdentityStore(backend).register_assertion(_assertion((0,) * 4097))
        assert backend.count(ASSERTION_RECORD_TABLE) == 0
        assert backend.count(ASSERTION_QUALIFIER_TABLE) == 0
    finally:
        backend.close()


def test_cursor_dump_load_preserves_packed_qualifiers(tmp_path):
    """权威 cursor 保留负 ordinal，并以新 owner 恢复完整领域断言。"""
    from pure_integer_ai.experiments.formal_train import make_train_context

    first, second = DictBackend(), DictBackend()
    try:
        make_train_context(first)
        make_train_context(second)
        assertion = _assertion((91516, 1, 7) + tuple(range(116)))
        identity_hash = ScopedIdentityStore(first).register_assertion(assertion, assertion_role=1)
        dump_run(first, str(tmp_path), "packed_qualifiers", spaces=[1], tables=DUMP_TABLES)
        assert load_run(second, str(tmp_path), "packed_qualifiers") == [1]
        assert ScopedIdentityStore(second).load_assertion(identity_hash) == assertion
        assert second.select(ASSERTION_QUALIFIER_TABLE, where={"identity_hash": identity_hash})
    finally:
        second.close()
        first.close()


def test_connector_graph_recovers_complete_long_qualifiers_after_cache_rebuild():
    """真实理论图多边打包后清缓存，仍恢复完整模板、scope 和限定项。"""
    from tests.test_l05b2b_language_generation_connector_graph import _fixture

    backend = DictBackend()
    try:
        graphs, promotion, connector, _identities, graph = _fixture(backend, variant=109)
        definition = connector.registry.templates[0]
        qualifiers = (91516, 1, 7) + tuple(range(116))
        written = graph.materialize(
            definition, scope=promotion.constraint.scope, provenance_kind=7,
            epistemic_origin=8, content_version=9, qualifiers=qualifiers)
        packed = backend.select(ASSERTION_QUALIFIER_TABLE, where={"ordinal": -1})
        assert len(packed) > 1
        before = backend.snapshot()
        graphs.context.graph_ontology.clear_runtime_caches()
        rebuilt = type(graph)(graph.ontology, graphs.order_graph, graph.predicates,
                              connector.registry.value_protocol)
        restored = rebuilt.read(definition.connector)
        assert restored == written
        assert restored.qualifiers == qualifiers
        assert backend.snapshot() == before
    finally:
        backend.close()
