"""大图对象整数打包的规范向量、冷恢复和真实 registry 完整性边界。"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from pure_integer_ai.cognition.shared.identity import OBJECT_CONCEPT, ObjectIdentity
from pure_integer_ai.experiments.formal_train import make_train_context
from pure_integer_ai.storage.assertion_identity import (
    IDENTITY_GRAPH_OBJECT,
    IDENTITY_PART_TABLE,
    IdentityCollisionError,
    IdentityIncompleteError,
    IntegerIdentityRegistry,
    register_assertion_identity_tables,
)
from pure_integer_ai.storage.backend import DictBackend, SQLiteBackend
from pure_integer_ai.storage.graph_object import (
    GRAPH_OBJECT_TABLE,
    GraphObjectRepository,
    register_graph_object_table,
)
from pure_integer_ai.storage.graph_object_identity import (
    GRAPH_OBJECT_CODEC_GENERIC_BLOCK_V1,
    GRAPH_OBJECT_CODEC_GENERIC_V1,
    GRAPH_OBJECT_COMPONENT_BLOCK_TABLE,
    GRAPH_OBJECT_COMPONENT_TABLE,
    GraphObjectIdentitySpec,
)
from pure_integer_ai.storage.graph_object_packed import (
    GRAPH_OBJECT_CODEC_GENERIC_PACKED_V1,
    decode_packed_components,
    encode_packed_components,
)
from pure_integer_ai.storage.integer_codec import IntegerCodecError, encode_integer_tuple
from pure_integer_ai.training.cursor import DUMP_TABLES, dump_run, load_run


def _identity() -> ObjectIdentity:
    """构造含正负整数且末 word 有填充的大对象，不依赖自然语言内容。"""
    return ObjectIdentity(OBJECT_CONCEPT, tuple(index % 31 - 15 for index in range(512)))


def _block_identity() -> ObjectIdentity:
    """构造触发固定整数分块 codec 的大对象。"""
    return ObjectIdentity(
        OBJECT_CONCEPT,
        tuple(index % 31 - 15 for index in range(5000)),
    )


def _repository(backend, *, registry=None) -> GraphObjectRepository:
    """仅注册实际 GraphObjectRepository 所需的现有 schema。"""
    register_assertion_identity_tables(backend)
    register_graph_object_table(backend)
    return GraphObjectRepository(backend, registry=registry)


def _add(repository, spec, *, local_id=1):
    """通过真实 registry 写入一条稳定空间映射，避免测试绕过 hash 校验。"""
    return repository.add(spec, object_kind=spec.stable_key[0], space_id=1,
                          local_id=local_id, space_identity_key=(1, 2, 3))


def _legacy_spec(identity) -> GraphObjectIdentitySpec:
    """显式构造旧大 generic 格式，验证新代码无须重写旧库。"""
    return GraphObjectIdentitySpec(
        identity.stable_key(), GRAPH_OBJECT_CODEC_GENERIC_V1, 0,
        identity.components[:8], tuple(enumerate(identity.components[8:], start=8)))


def test_zero_sequence_has_frozen_cross_language_word_vector():
    """128 个零冻结为 130 字节，首 word=384，余 word 全零。"""
    encoded = encode_packed_components((0,) * 128)
    assert encoded == ((130, 384, 0, 0, 0, 0, 0, 0),
                       tuple((ordinal, 0) for ordinal in range(7, 19)))
    assert decode_packed_components(128, 0, *encoded) == (0,) * 128


def test_signed_boundaries_and_wide_integers_restore_without_narrowing():
    """打包不截断负数、有符号边界或超出宿主 64 位的完整整数。"""
    values = (0,) * 128 + (-1, 1, -(1 << 63), (1 << 63) - 1, -(1 << 257), 1 << 257)
    encoded = encode_packed_components(values)
    assert encoded is not None
    assert decode_packed_components(len(values), 0, *encoded) == values


def test_small_or_hash_heavy_keys_keep_legacy_representation():
    """小键和 varint 膨胀键不因打包增加子行，旧格式仍是合法新写选择。"""
    for values in ((0,) * 127, ((1 << 63) - 1,) * 128):
        assert encode_packed_components(values) is None
        spec = GraphObjectIdentitySpec.generic(ObjectIdentity(OBJECT_CONCEPT, values).stable_key())
        assert spec.identity_codec == GRAPH_OBJECT_CODEC_GENERIC_V1


def test_large_generic_key_uses_integer_blocks_and_roundtrips():
    """超大对象按固定整数列分块，完整键和 hash 校验仍闭合。"""
    backend = DictBackend()
    try:
        repository = _repository(backend)
        identity = _block_identity()
        spec = GraphObjectIdentitySpec.generic(identity.stable_key())
        assert spec.identity_codec == GRAPH_OBJECT_CODEC_GENERIC_BLOCK_V1
        assert spec.overflow_components == ()
        assert spec.overflow_blocks
        record = _add(repository, spec)
        repository.clear_runtime_caches()
        assert repository.registry.read_key(
            IDENTITY_GRAPH_OBJECT, record.identity_hash) == identity.stable_key()
        assert backend.count(GRAPH_OBJECT_COMPONENT_TABLE) == 0
        assert backend.count(GRAPH_OBJECT_COMPONENT_BLOCK_TABLE) == len(
            spec.overflow_blocks)
    finally:
        backend.close()


@pytest.mark.parametrize("corruption", ["missing", "ordinal", "padding", "mixed"])
def test_block_reader_rejects_incomplete_or_mixed_representation(corruption):
    """分块缺失、断序、非零填充和旧表混入均必须 fail closed。"""
    backend = DictBackend()
    try:
        repository = _repository(backend)
        identity = _block_identity()
        record = _add(repository, GraphObjectIdentitySpec.generic(
            identity.stable_key()))
        snapshot = backend.snapshot()
        blocks = snapshot[GRAPH_OBJECT_COMPONENT_BLOCK_TABLE]
        if corruption == "missing":
            blocks.pop()
        elif corruption == "ordinal":
            blocks[0]["block_ordinal"] = 1
        elif corruption == "padding":
            blocks[-1]["component_value_31"] = 1
        else:
            snapshot[GRAPH_OBJECT_COMPONENT_TABLE] = [{
                "identity_hash": record.identity_hash,
                "component_ordinal": 8,
                "component_value": 1,
            }]
        restored = DictBackend()
        try:
            cold = _repository(restored)
            restored.load_snapshot(snapshot)
            with pytest.raises((IdentityIncompleteError, IdentityCollisionError)):
                cold.registry.read_key(IDENTITY_GRAPH_OBJECT, record.identity_hash)
        finally:
            restored.close()
    finally:
        backend.close()


@pytest.mark.parametrize("codec", ["legacy", "packed"])
def test_file_reopen_reads_complete_key_hash_and_mapping_without_writes(tmp_path, codec):
    """新旧大键均在关闭后只读恢复，重新登记不会覆盖旧物理表示。"""
    identity = _identity()
    path = tmp_path / "objects.sqlite3"
    backend = SQLiteBackend(str(path))
    try:
        repository = _repository(backend)
        spec = (_legacy_spec(identity) if codec == "legacy"
                else GraphObjectIdentitySpec.generic(identity.stable_key()))
        record = _add(repository, spec)
        assert record.identity_hash == repository.registry.identity_hash(
            IDENTITY_GRAPH_OBJECT, identity.stable_key())
        repository.clear_runtime_caches()
        assert _add(repository, GraphObjectIdentitySpec.generic(identity.stable_key())) == record
        assert backend.select(GRAPH_OBJECT_TABLE)[0]["identity_codec"] == spec.identity_codec
        backend.commit()
    finally:
        backend.close()
    before = (path.read_bytes(), path.stat().st_mtime_ns)
    backend = SQLiteBackend(str(path), read_only=True)
    try:
        repository = _repository(backend)
        assert repository.registry.read_key(IDENTITY_GRAPH_OBJECT, record.identity_hash) == identity.stable_key()
        assert repository.find_identity(identity.stable_key()) == record.identity_hash
        assert repository.read(record.identity_hash) == record
        assert repository.read_by_ref(1, 1) == record
    finally:
        backend.close()
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before


@pytest.mark.parametrize("kind", ["dict", "sqlite"])
def test_ontology_dump_load_recovers_packed_object_membership(tmp_path, kind):
    """新格式经生产 ontology 和权威 dump/load 保留完整整数身份。"""
    factory = DictBackend if kind == "dict" else SQLiteBackend
    first_backend = factory()
    try:
        first = make_train_context(first_backend)
        identity = _identity()
        ref = first.graph_ontology.materialize(identity)
        identity_hash = first.graph_ontology.identity_hash_of(ref)
        assert first_backend.select(GRAPH_OBJECT_TABLE, where={"identity_hash": identity_hash})[0][
            "identity_codec"] == GRAPH_OBJECT_CODEC_GENERIC_PACKED_V1
        assert first_backend.select(IDENTITY_PART_TABLE, where={"identity_kind": IDENTITY_GRAPH_OBJECT}) == []
        dump_run(first_backend, str(tmp_path), "packed", spaces=[first.space_id], tables=DUMP_TABLES)
    finally:
        first_backend.close()
    second_backend = factory()
    try:
        second = make_train_context(second_backend)
        assert load_run(second_backend, str(tmp_path), "packed") == [1]
        assert second.graph_ontology.resolve(identity) == ref
        assert second.graph_ontology.identity_of(ref) == identity
        assert second.graph_ontology.identity_hash_of(ref) == identity_hash
    finally:
        second_backend.close()


def _replace_stream(snapshot, row, data):
    """给故障注入重建物理 word，使测试能独立破坏规范流和分帧字段。"""
    words = tuple(int.from_bytes(data[start:start + 7], "little") for start in range(0, len(data), 7))
    values = (len(data), *words[:7], *((0,) * max(7 - len(words), 0)))
    for index, value in enumerate(values):
        row[f"codec_value_{index}"] = value
    snapshot[GRAPH_OBJECT_COMPONENT_TABLE] = [
        {"identity_hash": row["identity_hash"], "component_ordinal": ordinal, "component_value": value}
        for ordinal, value in enumerate(words[7:], start=7)]


@pytest.mark.parametrize("corruption", [
    "missing_word", "duplicate_word", "ordinal_drift", "negative_word", "oversized_word",
    "padding", "count", "noncanonical", "trailing_byte", "ref", "flat_parts", "legacy_children",
])
def test_cold_registry_rejects_corrupted_or_mixed_representation(corruption):
    """独立新 owner 必须拒绝缺块、混合、非规范和完整键计数破坏。"""
    backend = DictBackend()
    repository = _repository(backend)
    identity = _identity()
    record = _add(repository, GraphObjectIdentitySpec.generic(identity.stable_key()))
    snapshot = backend.snapshot()
    row = snapshot[GRAPH_OBJECT_TABLE][0]
    parts = snapshot[GRAPH_OBJECT_COMPONENT_TABLE]
    if corruption == "missing_word":
        parts.pop()
    elif corruption == "duplicate_word":
        parts.append(dict(parts[-1]))
    elif corruption == "ordinal_drift":
        parts[0]["component_ordinal"] += 1
    elif corruption == "negative_word":
        parts[0]["component_value"] = -1
    elif corruption == "oversized_word":
        parts[0]["component_value"] = 1 << 56
    elif corruption == "padding":
        parts[-1]["component_value"] += 1 << (8 * (row["codec_value_0"] % 7))
    elif corruption == "count":
        row["component_size"] += 1
    elif corruption in {"noncanonical", "trailing_byte"}:
        data = encode_integer_tuple(identity.components)
        data = (data[:1] + bytes((data[1] | 128, 0)) + data[2:]
                if corruption == "noncanonical" else data + b"\x00")
        _replace_stream(snapshot, row, data)
    elif corruption == "ref":
        row["codec_ref_hash"] = 1
    elif corruption == "flat_parts":
        snapshot[IDENTITY_PART_TABLE].append({"identity_kind": IDENTITY_GRAPH_OBJECT,
            "identity_hash": record.identity_hash, "part_index": 0, "part_value": 1})
    elif corruption == "legacy_children":
        snapshot[GRAPH_OBJECT_COMPONENT_TABLE] = [
            {"identity_hash": record.identity_hash, "component_ordinal": ordinal, "component_value": value}
            for ordinal, value in enumerate(identity.components[8:], start=8)]
    restored = DictBackend()
    try:
        cold = _repository(restored)
        restored.load_snapshot(snapshot)
        with pytest.raises(IdentityIncompleteError):
            cold.registry.read_key(IDENTITY_GRAPH_OBJECT, record.identity_hash)
    finally:
        restored.close()
        backend.close()


def test_packed_payload_cannot_bypass_full_hash_or_collision_check():
    """即使物理流完整可解码，hash 不吻合或碰撞也必须拒绝。"""
    backend = DictBackend()
    try:
        registry = IntegerIdentityRegistry(backend, hasher=SimpleNamespace(h63=lambda key: 17))
        repository = _repository(backend, registry=registry)
        identity = _identity()
        record = _add(repository, GraphObjectIdentitySpec.generic(identity.stable_key()))
        changed = ObjectIdentity(OBJECT_CONCEPT, (*identity.components[:-1], 63))
        with pytest.raises(IdentityCollisionError):
            _add(repository, GraphObjectIdentitySpec.generic(changed.stable_key()), local_id=2)
        assert backend.count(GRAPH_OBJECT_TABLE) == 1
        snapshot = backend.snapshot()
    finally:
        backend.close()
    restored = DictBackend()
    try:
        cold = _repository(restored)
        restored.load_snapshot(snapshot)
        with pytest.raises(IdentityCollisionError):
            cold.registry.read_key(IDENTITY_GRAPH_OBJECT, record.identity_hash)
    finally:
        restored.close()


def test_lookup_does_not_construct_a_physical_encoding_plan(monkeypatch):
    """只读键查找不得因大键额外运行物理编码器或创建子载荷。"""
    backend = DictBackend()
    try:
        repository = _repository(backend)
        identity = _identity()
        record = _add(repository, GraphObjectIdentitySpec.generic(identity.stable_key()))
        repository.clear_runtime_caches()
        def forbidden_encoding(components):
            """以显式失败捕获查找过程意外执行编码。"""
            raise AssertionError("lookup encoded a physical payload")
        monkeypatch.setattr("pure_integer_ai.storage.graph_object_identity.encode_packed_components",
                            forbidden_encoding)
        assert repository.find_identity(identity.stable_key()) == record.identity_hash
    finally:
        backend.close()


def test_packed_reader_rejects_unused_slot_and_pointless_format():
    """固定槽填充和无收益格式不能成为额外可接受的物理表示。"""
    with pytest.raises(IntegerCodecError):
        decode_packed_components(128, 0, (1, 0, 1, 0, 0, 0, 0, 0), ())
    with pytest.raises(IntegerCodecError):
        decode_packed_components(128, 0, (1282, 0, 0, 0, 0, 0, 0, 0), ())
