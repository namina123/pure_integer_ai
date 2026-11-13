"""Lossless large-key storage preserves all integers and the original V1 bounds."""
from __future__ import annotations

import pytest

from pure_integer_ai.storage.assertion_identity import (
    IntegerIdentityRegistry, integer_identity_hash, register_assertion_identity_tables,
)
from pure_integer_ai.storage.backend import SQLiteBackend
from pure_integer_ai.storage.identity_part_codec import decode_identity_parts, pack_identity_parts
from pure_integer_ai.storage.integer_codec import IntegerCodecError, encode_integer_tuple


def _key():
    return tuple(range(-32, 32)) * 128


def test_block_codec_preserves_all_members_and_reduces_v1_rows():
    key = _key()
    parts = pack_identity_parts(key)
    data = encode_integer_tuple(key)
    assert parts[:1] == ((-3, len(data)),)
    assert parts[2] == (-1, 2)
    assert len(parts) < (len(data) + 6) // 7 + 2 < len(key)
    assert decode_identity_parts(len(key), parts) == key
    assert pack_identity_parts(key) == parts


def test_small_key_frozen_v1_vector_is_unchanged():
    parts = pack_identity_parts((0,) * 128)
    assert parts[:3] == ((-2, 130), (-1, 1), (0, 384))
    assert len(parts) == 21


def test_large_legacy_v1_words_remain_readable():
    key = _key()
    data = encode_integer_tuple(key)
    parts = ((-2, len(data)), (-1, 1), *(
        (n, int.from_bytes(data[start:start + 7], 'little'))
        for n, start in enumerate(range(0, len(data), 7))))
    assert decode_identity_parts(len(key), parts) == key


@pytest.mark.parametrize('corruption', (
    'decoded_budget', 'stored_budget', 'missing_word', 'duplicate_word', 'word_index',
    'negative_word', 'wide_word', 'version', 'marker', 'payload', 'count', 'boolean',
))
def test_block_corruption_and_expansion_bombs_fail_closed(corruption):
    key = _key()
    parts = list(pack_identity_parts(key))
    size = len(key)
    if corruption == 'decoded_budget':
        parts[0] = (-3, 1 << 62)
    elif corruption == 'stored_budget':
        parts[1] = (-2, 1 << 62)
    elif corruption == 'missing_word':
        parts.pop()
    elif corruption == 'duplicate_word':
        parts.append(parts[-1])
    elif corruption == 'word_index':
        parts[3] = (1, parts[3][1])
    elif corruption == 'negative_word':
        parts[3] = (0, -1)
    elif corruption == 'wide_word':
        parts[3] = (0, 1 << 56)
    elif corruption == 'version':
        parts[2] = (-1, 3)
    elif corruption == 'marker':
        parts[1] = (-4, parts[1][1])
    elif corruption == 'payload':
        parts[3] = (0, parts[3][1] ^ 1)
    elif corruption == 'count':
        size += 1
    else:
        parts[3] = (False, parts[3][1])
    with pytest.raises(IntegerCodecError):
        decode_identity_parts(size, tuple(parts))


def test_cold_sqlite_reopen_preserves_hash_members_and_rows(tmp_path):
    path = tmp_path / 'identity.sqlite3'
    key = _key()
    backend = SQLiteBackend(str(path))
    try:
        register_assertion_identity_tables(backend)
        registry = IntegerIdentityRegistry(backend)
        identity_hash = registry.register(91536, key, parent_hash=11, ordinal=7)
        assert identity_hash == integer_identity_hash(91536, key)
        before = backend.snapshot()
        assert registry.register(91536, key, parent_hash=11, ordinal=7) == identity_hash
        assert backend.snapshot() == before
        backend.commit()
    finally:
        backend.close()
    backend = SQLiteBackend(str(path), read_only=True)
    try:
        register_assertion_identity_tables(backend)
        cold = IntegerIdentityRegistry(backend)
        assert cold.read_key(91536, identity_hash) == key
        assert cold.metadata(91536, identity_hash).parent_hash == 11
        assert backend._conn.total_changes == 0
    finally:
        backend.close()
