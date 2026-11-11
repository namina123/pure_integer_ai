"""固定字节向量与损坏输入约束物理编码，不运行历史查询。"""
import hashlib

import pytest

from pure_integer_ai.storage.trace_byte_codec import (
    BLOCK_BYTES, MAGIC, TraceByteCodecError, decode_trace_bytes,
    encode_trace_bytes, iter_encoded_trace_bytes,
)


@pytest.mark.parametrize("raw,hexadecimal", [
    (b"A", "5049414c5a34423101000000000000000100008041"),
    (b"a" * 20, "5049414c5a34423114000000000000000a0000001a610100506161616161"),
])
def test_frozen_physical_vectors(raw, hexadecimal):
    encoded = bytes.fromhex(hexadecimal)
    assert encode_trace_bytes(raw) == encoded
    assert decode_trace_bytes(encoded, max_decoded_bytes=len(raw)) == raw


@pytest.mark.parametrize("size", [12, BLOCK_BYTES - 1, BLOCK_BYTES, BLOCK_BYTES + 1, BLOCK_BYTES * 2 + 1])
def test_block_boundaries_and_streamed_encoding(size):
    raw = (b"claim:1234;source:5678;" * (size // 22 + 2))[:size]
    checked = []
    streamed = b"".join(iter_encoded_trace_bytes(raw, checkpoint=lambda: checked.append(1)))
    assert len(checked) == (size + BLOCK_BYTES - 1) // BLOCK_BYTES
    assert streamed == encode_trace_bytes(raw)
    assert decode_trace_bytes(streamed, max_decoded_bytes=size) == raw


def test_incompressible_blocks_use_the_unique_raw_choice():
    raw = b"".join(hashlib.sha256(index.to_bytes(4, "little")).digest()
                   for index in range(BLOCK_BYTES // 32))
    encoded = encode_trace_bytes(raw)
    assert encoded[16:20] == (len(raw) | (1 << 31)).to_bytes(4, "little")
    assert decode_trace_bytes(encoded, max_decoded_bytes=len(raw)) == raw


@pytest.mark.parametrize("payload,raw_flag,expected", [
    (b"\x10a\x00\x00", 0, 20),  # offset 零。
    (b"\x10a\x02\x00", 0, 20),  # offset 跨当前块。
    (b"\xf0\xff", 0, 20),       # literal 扩展超出预算。
    (b"\x1fa\x01\x00\xff", 0, 20),
    (b"a" * 20, 1, 20),         # 可压缩块不可伪装为原样。
    (b"a", 1, 20),              # 原样长度不足。
])
def test_damaged_or_noncanonical_blocks_are_rejected(payload, raw_flag, expected):
    encoded = MAGIC + expected.to_bytes(8, "little")
    encoded += (len(payload) | (raw_flag << 31)).to_bytes(4, "little") + payload
    with pytest.raises(TraceByteCodecError):
        decode_trace_bytes(encoded, max_decoded_bytes=expected)


@pytest.mark.parametrize("mutation", ["budget", "truncated", "trailing", "header", "zero"])
def test_stream_envelope_and_decoded_budget_are_enforced(mutation):
    encoded = encode_trace_bytes(b"a" * 20)
    limit = 20
    if mutation == "budget":
        limit = 19
    elif mutation == "truncated":
        encoded = encoded[:-1]
    elif mutation == "trailing":
        encoded += b"\x00"
    elif mutation == "header":
        encoded = b"X" + encoded[1:]
    else:
        encoded = MAGIC + b"\x00" * 8
    with pytest.raises(TraceByteCodecError):
        decode_trace_bytes(encoded, max_decoded_bytes=limit)
