"""整数流单遍规范解码的独立字节向量和非规范输入边界。"""
from __future__ import annotations

from itertools import product

import pytest

from pure_integer_ai.storage.integer_codec import (
    IntegerCodecError,
    decode_integer_tuple,
    encode_integer_tuple,
)


@pytest.mark.parametrize(
    ("data", "values"),
    (
        (b"\x00", ()),
        (b"\x06\x00\x01\x02\x7f\x80\x01\x81\x01", (0, -1, 1, -64, 64, -65)),
        (b"\x7f" + b"\x00" * 127, (0,) * 127),
        (b"\x80\x01" + b"\x00" * 128, (0,) * 128),
        (b"\x02" + b"\xff" * 73 + b"\x01" + b"\x80" * 73 + b"\x02",
         (-(1 << 511), 1 << 511)),
    ),
)
def test_canonical_decoder_preserves_frozen_bytes_and_full_integers(data, values):
    """规范向量独立给出，覆盖计数进位、正负边界及任意精度。"""
    assert decode_integer_tuple(data) == values
    assert encode_integer_tuple(values) == data


@pytest.mark.parametrize(
    "data",
    (
        b"", b"\x80", b"\x80\x00", b"\x81\x00\x00",
        b"\x00\x00", b"\x01", b"\x02\x00", b"\x01\x80",
        b"\x01\x80\x00", b"\x01\x81\x00", b"\x01\x00\x00",
        b"\x01" + b"\xff" * 73 + b"\x00",
        b"\x01" + b"\x80" * 73,
    ),
)
def test_canonical_decoder_rejects_incomplete_overlong_and_trailing_bytes(data):
    """计数与元素均须最短且完整，声明数量与全部物理字节严格对应。"""
    with pytest.raises(IntegerCodecError):
        decode_integer_tuple(data)


def test_all_one_and_two_byte_streams_have_exactly_the_canonical_members():
    """穷举短流；唯一合法流为零元素或单个单字节zigzag元素。"""
    accepted = {}
    for size in (1, 2):
        for items in product(range(256), repeat=size):
            data = bytes(items)
            try:
                accepted[data] = decode_integer_tuple(data)
            except IntegerCodecError:
                pass
    expected = {b"\x00": ()}
    for value in range(64):
        expected[bytes((1, 2 * value))] = (value,)
        expected[bytes((1, 2 * value + 1))] = (-value - 1,)
    assert accepted == expected


@pytest.mark.parametrize("data", (None, "\x00", bytearray(b"\x00"), memoryview(b"\x00")))
def test_canonical_decoder_requires_nonempty_bytes(data):
    """宿主容器不能绕过既有bytes输入合同。"""
    with pytest.raises(IntegerCodecError):
        decode_integer_tuple(data)
