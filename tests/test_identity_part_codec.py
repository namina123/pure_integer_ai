"""通用身份键打包的规范向量、兼容恢复和完整性拒绝。"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from pure_integer_ai.experiments.formal_train import make_train_context
from pure_integer_ai.storage.assertion_identity import (
    IDENTITY_HEADER_TABLE,
    IDENTITY_PART_TABLE,
    ExternalIdentityKey,
    IdentityCollisionError,
    IdentityIncompleteError,
    IntegerIdentityRegistry,
    integer_identity_hash,
    register_assertion_identity_tables,
)
from pure_integer_ai.storage.backend import DictBackend, SQLiteBackend
from pure_integer_ai.storage.identity_part_codec import (
    decode_identity_parts,
    pack_identity_parts,
)
from pure_integer_ai.storage.integer_codec import IntegerCodecError, encode_integer_tuple
from pure_integer_ai.training.cursor import DUMP_TABLES, dump_run, load_run


_KIND = 91536


def _registry(backend, *, hasher=None):
    """仅注册通用身份表，确保编码不依赖领域 facade。"""
    register_assertion_identity_tables(backend)
    return IntegerIdentityRegistry(backend, hasher=hasher)


def _rows(identity_hash, parts):
    """把物理向量转换为 backend 行，用于兼容与损坏注入。"""
    return [{"identity_kind": _KIND, "identity_hash": identity_hash,
             "part_index": index, "part_value": value} for index, value in parts]


def _legacy(backend, key, *, parent_hash=19, ordinal=23):
    """按历史合同写旧表示，不调用新编码选择器。"""
    identity_hash = integer_identity_hash(_KIND, key)
    backend.insert_many(IDENTITY_PART_TABLE, _rows(identity_hash, enumerate(key)))
    backend.insert(IDENTITY_HEADER_TABLE, {
        "identity_kind": _KIND, "identity_hash": identity_hash, "key_size": len(key),
        "parent_hash": parent_hash, "ordinal": ordinal})
    return identity_hash


def _stream_parts(data):
    """由给定字节构造 word，独立注入 varint 和计数损坏。"""
    return ((-2, len(data)), (-1, 1), *(
        (index, int.from_bytes(data[start:start + 7], "little"))
        for index, start in enumerate(range(0, len(data), 7))))


def test_zero_key_has_frozen_integer_word_vector():
    """128 个零固定为 130 字节与 19 word，首 word 为 384。"""
    key = (0,) * 128
    parts = ((-2, 130), (-1, 1), (0, 384), *((index, 0) for index in range(1, 19)))
    assert pack_identity_parts(key) == parts
    assert decode_identity_parts(128, parts) == key


def test_signed_and_wide_integer_values_are_not_narrowed():
    """物理 word 有界，但恢复的完整整数保留符号与任意宽度。"""
    key = (0,) * 128 + (-1, 1, -(1 << 63), (1 << 63) - 1, -(1 << 257), 1 << 257)
    parts = pack_identity_parts(key)
    assert parts is not None and len(parts) < len(key)
    assert decode_identity_parts(len(key), parts) == key


@pytest.mark.parametrize("backend_type", (DictBackend, SQLiteBackend))
def test_batch_registry_reads_validate_keys_and_preserve_input_order(backend_type):
    backend = (backend_type(":memory:") if backend_type is SQLiteBackend
               else backend_type())
    try:
        registry = _registry(backend)
        keys = ((101, 202), (0,) * 128, tuple(range(-70, 70)))
        hashes = tuple(registry.register(_KIND, key) for key in keys)

        values = (*hashes[:2], *(10**12 + n for n in range(905)),
                  hashes[2], hashes[0])
        rows = backend.select_in(
            IDENTITY_HEADER_TABLE,
            "identity_hash",
            values,
            where={"identity_kind": _KIND},
            order_by="identity_hash",
        )

        assert tuple(row["identity_hash"] for row in rows) == tuple(sorted(hashes))
        assert registry.read_keys(_KIND, (hashes[2], hashes[0], hashes[2])) == (
            keys[2], keys[0], keys[2])
        with pytest.raises(KeyError):
            registry.read_keys(_KIND, (hashes[0], 10**15))
    finally:
        backend.close()


def test_registry_read_scope_reuses_only_verified_keys_and_clears_on_exit(
        monkeypatch):
    backend = DictBackend()
    try:
        registry = _registry(backend)
        key = tuple(range(128))
        identity_hash = registry.register(_KIND, key)
        original_select = backend.select
        identity_selects = []

        def count_identity_selects(table, *args, **kwargs):
            if table in {IDENTITY_HEADER_TABLE, IDENTITY_PART_TABLE}:
                identity_selects.append(table)
            return original_select(table, *args, **kwargs)

        monkeypatch.setattr(backend, "select", count_identity_selects)
        with registry.read_scope():
            assert registry.read_key(_KIND, identity_hash) == key
            with registry.read_scope():
                assert registry.read_keys(
                    _KIND, (identity_hash, identity_hash)) == (key, key)
            assert len(identity_selects) == 2
        assert not registry._scoped_read_keys

        snapshot = backend.snapshot()
        snapshot[IDENTITY_PART_TABLE] = snapshot[IDENTITY_PART_TABLE][:-1]
        backend.load_snapshot(snapshot)
        with pytest.raises(IdentityIncompleteError):
            registry.read_key(_KIND, identity_hash)
    finally:
        backend.close()


@pytest.mark.parametrize("batch", (False, True))
def test_registry_read_scope_rechecks_cached_key_hash(batch):
    backend = DictBackend()
    try:
        registry = _registry(backend)
        key = (101, 202, 303)
        identity_hash = registry.register(_KIND, key)
        with registry.read_scope():
            assert registry.read_key(_KIND, identity_hash) == key
            registry._scoped_read_keys[(
                _KIND, identity_hash)] = (404, 505, 606)
            with pytest.raises(IdentityCollisionError):
                if batch:
                    registry.read_keys(_KIND, (identity_hash,))
                else:
                    registry.read_key(_KIND, identity_hash)
    finally:
        backend.close()


@pytest.mark.parametrize("key", [(0,) * 127, ((1 << 63) - 1,) * 128])
def test_small_or_unprofitable_keys_keep_flat_rows(key):
    """阈值及收益是通用整数规则，不能为了 codec 改写小键或增加行数。"""
    backend = DictBackend()
    try:
        registry = _registry(backend)
        identity_hash = registry.register(_KIND, key)
        assert pack_identity_parts(key) is None
        assert backend.select(IDENTITY_PART_TABLE, order_by="part_index") == _rows(
            identity_hash, enumerate(key))
        assert registry.read_key(_KIND, identity_hash) == key
    finally:
        backend.close()


@pytest.mark.parametrize("legacy", [False, True])
def test_reopen_is_read_only_and_registration_preserves_physical_rows(tmp_path, legacy):
    """旧新大键关闭后均可只读恢复，幂等登记不迁移旧表示。"""
    path = tmp_path / "identities.sqlite3"
    key = tuple(index % 31 - 15 for index in range(512))
    backend = SQLiteBackend(str(path))
    try:
        registry = _registry(backend)
        identity_hash = (_legacy(backend, key) if legacy else
                         registry.register(_KIND, key, parent_hash=19, ordinal=23))
        before = backend.snapshot()
        assert registry.find(_KIND, key) == identity_hash
        assert registry.register(_KIND, key, parent_hash=19, ordinal=23) == identity_hash
        assert backend.snapshot() == before
        assert identity_hash == integer_identity_hash(_KIND, key)
        assert len(before[IDENTITY_PART_TABLE]) == (len(key) if legacy else len(pack_identity_parts(key)))
        backend.commit()
    finally:
        backend.close()
    before_stamp = (path.stat().st_size, path.stat().st_mtime_ns)
    cold_backend = SQLiteBackend(str(path), read_only=True)
    try:
        cold = _registry(cold_backend)
        assert cold.read_key(_KIND, identity_hash) == key
        metadata = cold.metadata(_KIND, identity_hash)
        assert (metadata.key_size, metadata.parent_hash, metadata.ordinal) == (512, 19, 23)
        assert cold.register(_KIND, key, parent_hash=19, ordinal=23) == identity_hash
        assert cold_backend._conn.total_changes == 0
    finally:
        cold_backend.close()
    assert (path.stat().st_size, path.stat().st_mtime_ns) == before_stamp


@pytest.mark.parametrize("corruption", [
    "missing_word", "duplicate_word", "word_index", "negative_word", "wide_word",
    "padding", "missing_size", "missing_version", "duplicate_marker", "unknown_version",
    "unknown_negative", "zero_bytes", "huge_bytes", "count", "small_key_size",
    "noncanonical", "truncated", "trailing", "flat_mixed", "boolean_index", "boolean_word",
])
def test_cold_registry_rejects_damaged_parts(corruption):
    """新 owner 必须拒绝缺块、marker、混合格式、非规范流与严格类型破坏。"""
    backend = DictBackend()
    cold_backend = DictBackend()
    try:
        registry = _registry(backend)
        key = (0,) * 128
        identity_hash = registry.register(_KIND, key)
        snapshot = backend.snapshot()
        parts = snapshot[IDENTITY_PART_TABLE]
        header = snapshot[IDENTITY_HEADER_TABLE][0]
        if corruption == "missing_word":
            parts.pop()
        elif corruption == "duplicate_word":
            parts.append(dict(parts[-1]))
        elif corruption == "word_index":
            parts[2]["part_index"] = 1
        elif corruption == "negative_word":
            parts[2]["part_value"] = -1
        elif corruption == "wide_word":
            parts[2]["part_value"] = 1 << 56
        elif corruption == "padding":
            parts[-1]["part_value"] = 1 << 32
        elif corruption == "missing_size":
            parts.pop(0)
        elif corruption == "missing_version":
            parts.pop(1)
        elif corruption == "duplicate_marker":
            parts.append(dict(parts[0]))
        elif corruption == "unknown_version":
            parts[1]["part_value"] = 2
        elif corruption == "unknown_negative":
            parts[0]["part_index"] = -3
        elif corruption == "zero_bytes":
            parts[0]["part_value"] = 0
        elif corruption == "huge_bytes":
            parts[0]["part_value"] = 1 << 62
        elif corruption == "count":
            header["key_size"] += 1
        elif corruption == "small_key_size":
            header["key_size"] = 127
        elif corruption in {"noncanonical", "truncated", "trailing"}:
            data = encode_integer_tuple(key)
            data = (data[:1] + b"\x81\x00" + data[2:] if corruption == "noncanonical"
                    else data[:-1] if corruption == "truncated" else data + b"\x00")
            snapshot[IDENTITY_PART_TABLE] = _rows(identity_hash, _stream_parts(data))
        elif corruption == "flat_mixed":
            parts.extend(_rows(identity_hash, enumerate(key)))
        elif corruption == "boolean_index":
            parts[2]["part_index"] = False
        elif corruption == "boolean_word":
            parts[-1]["part_value"] = False
        cold = _registry(cold_backend)
        cold_backend.load_snapshot(snapshot)
        with pytest.raises(IdentityIncompleteError):
            cold.read_key(_KIND, identity_hash)
    finally:
        cold_backend.close()
        backend.close()


def test_complete_packed_content_still_requires_original_hash():
    """流可完整解码时也必须重算原 hash，不能把物理正确当成身份正确。"""
    backend = DictBackend()
    cold_backend = DictBackend()
    try:
        identity_hash = _registry(backend).register(_KIND, (0,) * 128)
        snapshot = backend.snapshot()
        snapshot[IDENTITY_PART_TABLE][2]["part_value"] ^= 2 << 16
        cold = _registry(cold_backend)
        cold_backend.load_snapshot(snapshot)
        with pytest.raises(IdentityCollisionError):
            cold.read_key(_KIND, identity_hash)
    finally:
        cold_backend.close()
        backend.close()


def test_collision_metadata_and_external_mixing_remain_fail_closed():
    """完整键碰撞、metadata 冲突和外部记录混存均不能因打包绕过。"""
    backend = DictBackend()
    try:
        registry = _registry(backend, hasher=SimpleNamespace(h63=lambda value: 17))
        key = (0,) * 128
        registry.register(_KIND, key, parent_hash=19, ordinal=23)
        before = backend.snapshot()
        with pytest.raises(IdentityCollisionError):
            registry.register(_KIND, (*key[:-1], 1), parent_hash=19, ordinal=23)
        with pytest.raises(IdentityIncompleteError):
            registry.register(_KIND, key, parent_hash=20, ordinal=23)
        assert backend.snapshot() == before
        registry.register_external_key_resolver(_KIND, lambda value: ExternalIdentityKey(key))
        with pytest.raises(IdentityIncompleteError, match="同时存在"):
            registry.read_key(_KIND, 17)
    finally:
        backend.close()


def test_legacy_reader_rejects_discontinuous_indices_and_packed_without_profit():
    """旧表示不放宽连续 index，新表示必须满足严格行数收益。"""
    with pytest.raises(IntegerCodecError):
        decode_identity_parts(2, ((0, -1), (2, 1)))
    with pytest.raises(IntegerCodecError):
        decode_identity_parts(128, ((-2, 7 * 126), (-1, 1), (0, 0)))


def test_cursor_dump_load_preserves_mixed_old_new_complete_keys(tmp_path):
    """权威 cursor dump 原样保留负 index，跨 owner 恢复旧新格式和元数据。"""
    first = DictBackend()
    second = DictBackend()
    try:
        make_train_context(first)
        make_train_context(second)
        registry = IntegerIdentityRegistry(first)
        old_key, new_key = (0,) * 128, tuple(range(512))
        old_hash = _legacy(first, old_key)
        new_hash = registry.register(_KIND, new_key, parent_hash=old_hash, ordinal=31)
        dump_run(first, str(tmp_path), "mixed_identity", spaces=[1], tables=DUMP_TABLES)
        assert load_run(second, str(tmp_path), "mixed_identity") == [1]
        cold = IntegerIdentityRegistry(second)
        assert cold.read_key(_KIND, old_hash) == old_key
        assert cold.read_key(_KIND, new_hash) == new_key
        metadata = cold.metadata(_KIND, new_hash)
        assert (metadata.parent_hash, metadata.ordinal) == (old_hash, 31)
    finally:
        second.close()
        first.close()
