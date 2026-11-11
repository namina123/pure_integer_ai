"""Assertion 限定项的历史全子表、内联溢出和七字节 word 物理协议。"""
from __future__ import annotations

from pure_integer_ai.storage.integer_codec import (
    IntegerCodecError,
    decode_integer_tuple,
    encode_integer_tuple,
    strict_integer_tuple,
)


INLINE_QUALIFIER_COUNT = 3
QUALIFIER_CODEC_PACKED_V1 = 1
QUALIFIER_PACKED_MIN_SIZE = 32
_BYTE_COUNT_ORDINAL = -2
_CODEC_ORDINAL = -1
_WORD_BYTES = 7
_MAX_WORD = (1 << (_WORD_BYTES * 8)) - 1


def encode_qualifier_overflow(
        qualifiers: tuple[int, ...],
        ) -> tuple[tuple[int, int], ...]:
    """保留前三项内联，仅在长限定项严格减少子行时选择 packed v1。"""
    strict_integer_tuple(qualifiers, label="assertion qualifiers", empty=True)
    overflow = qualifiers[INLINE_QUALIFIER_COUNT:]
    if len(qualifiers) >= QUALIFIER_PACKED_MIN_SIZE:
        data = encode_integer_tuple(overflow)
        word_count = (len(data) + _WORD_BYTES - 1) // _WORD_BYTES
        if word_count + 2 < len(overflow):
            return (
                (_BYTE_COUNT_ORDINAL, len(data)),
                (_CODEC_ORDINAL, QUALIFIER_CODEC_PACKED_V1),
                *tuple(
                    (index, int.from_bytes(data[start:start + _WORD_BYTES], "little"))
                    for index, start in enumerate(
                        range(0, len(data), _WORD_BYTES),
                        start=INLINE_QUALIFIER_COUNT)),
            )
    return tuple(enumerate(overflow, start=INLINE_QUALIFIER_COUNT))


def decode_assertion_qualifiers(
        qualifier_size: int,
        inline_values: tuple[int | None, ...],
        parts: tuple[tuple[int, int], ...],
        ) -> tuple[int, ...]:
    """严格区分三种完整表示，拒绝半内联、混存和任何不连续的物理行。"""
    if type(qualifier_size) is not int or qualifier_size < 0:
        raise IntegerCodecError("qualifier_size 必须为非负严格整数")
    if (not isinstance(inline_values, tuple)
            or len(inline_values) != INLINE_QUALIFIER_COUNT):
        raise IntegerCodecError("qualifier 内联槽数量不完整")
    if not isinstance(parts, tuple):
        raise IntegerCodecError("qualifier 子行必须为 tuple")
    for part in parts:
        if (not isinstance(part, tuple) or len(part) != 2
                or any(type(value) is not int for value in part)):
            raise IntegerCodecError("qualifier 子行必须为两个严格整数")
    if all(value is None for value in inline_values):
        return _decode_integer_parts(qualifier_size, 0, parts)
    if any(type(value) is not int for value in inline_values):
        raise IntegerCodecError("qualifier 内联列不完整或不是严格整数")
    inline_size = min(qualifier_size, INLINE_QUALIFIER_COUNT)
    if any(inline_values[index] != 0
           for index in range(inline_size, INLINE_QUALIFIER_COUNT)):
        raise IntegerCodecError("未使用 qualifier 内联槽非零")
    inline = inline_values[:inline_size]
    overflow_size = max(qualifier_size - INLINE_QUALIFIER_COUNT, 0)
    if parts and parts[0][0] < 0:
        overflow = _decode_packed_parts(qualifier_size, parts)
    else:
        overflow = _decode_integer_parts(
            overflow_size, INLINE_QUALIFIER_COUNT, parts)
    return (*inline, *overflow)


def _decode_integer_parts(
        size: int, first_ordinal: int,
        parts: tuple[tuple[int, int], ...],
        ) -> tuple[int, ...]:
    """恢复逐整数旧表示，负值允许但 ordinal 必须精确连续。"""
    if len(parts) != size:
        raise IntegerCodecError("qualifier 数量与 qualifier_size 不一致")
    for ordinal, (actual, _) in enumerate(parts, start=first_ordinal):
        if actual != ordinal:
            raise IntegerCodecError("qualifier ordinal 不连续")
    return tuple(value for _, value in parts)


def _decode_packed_parts(
        qualifier_size: int, parts: tuple[tuple[int, int], ...],
        ) -> tuple[int, ...]:
    """先核 marker、收益和精确行数，再有限分配并恢复规范溢出段。"""
    if qualifier_size < QUALIFIER_PACKED_MIN_SIZE:
        raise IntegerCodecError("packed qualifier_size 小于格式阈值")
    if (len(parts) < 3 or parts[0][0] != _BYTE_COUNT_ORDINAL
            or parts[1] != (_CODEC_ORDINAL, QUALIFIER_CODEC_PACKED_V1)):
        raise IntegerCodecError("packed qualifier marker 或 codec 不完整")
    byte_count = parts[0][1]
    if byte_count <= 0:
        raise IntegerCodecError("packed qualifier 字节数必须为正整数")
    word_count = (byte_count + _WORD_BYTES - 1) // _WORD_BYTES
    overflow_size = qualifier_size - INLINE_QUALIFIER_COUNT
    if word_count + 2 >= overflow_size:
        raise IntegerCodecError("packed qualifier 未减少子行总数")
    if len(parts) != word_count + 2:
        raise IntegerCodecError("packed qualifier word 数量与字节数不一致")
    data = bytearray()
    remaining = byte_count
    for ordinal, (actual, word) in enumerate(
            parts[2:], start=INLINE_QUALIFIER_COUNT):
        if actual != ordinal:
            raise IntegerCodecError("packed qualifier word ordinal 断序或重复")
        if not 0 <= word <= _MAX_WORD:
            raise IntegerCodecError("packed qualifier word 必须位于 0..2**56-1")
        width = min(_WORD_BYTES, remaining)
        if word >> (width * 8):
            raise IntegerCodecError("packed qualifier 末 word 高位填充必须为零")
        data.extend(word.to_bytes(width, "little"))
        remaining -= width
    overflow = decode_integer_tuple(bytes(data))
    if len(overflow) != overflow_size:
        raise IntegerCodecError("packed qualifier 元素数与 qualifier_size 不一致")
    return overflow
