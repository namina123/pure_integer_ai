"""通用完整身份键的逐整数旧表示与同表七字节 word 表示。"""
from __future__ import annotations

from pure_integer_ai.storage.integer_codec import (
    IntegerCodecError,
    decode_integer_tuple,
    encode_integer_tuple,
    strict_integer_tuple,
)


IDENTITY_PART_CODEC_PACKED_V1 = 1
IDENTITY_PART_PACKED_MIN_SIZE = 128
_WORD_BYTES = 7
_MAX_WORD = (1 << (_WORD_BYTES * 8)) - 1
_BYTE_COUNT_INDEX = -2
_CODEC_INDEX = -1


def pack_identity_parts(
        key: tuple[int, ...],
        ) -> tuple[tuple[int, int], ...] | None:
    """仅在大键严格减少总行数时返回 marker 和 word，否则沿用逐整数表示。"""
    strict_integer_tuple(key, label="identity part key")
    if len(key) < IDENTITY_PART_PACKED_MIN_SIZE:
        return None
    data = encode_integer_tuple(key)
    word_count = (len(data) + _WORD_BYTES - 1) // _WORD_BYTES
    if word_count + 2 >= len(key):
        return None
    words = tuple(
        (index, int.from_bytes(data[start:start + _WORD_BYTES], "little"))
        for index, start in enumerate(range(0, len(data), _WORD_BYTES)))
    return ((_BYTE_COUNT_INDEX, len(data)),
            (_CODEC_INDEX, IDENTITY_PART_CODEC_PACKED_V1), *words)


def decode_identity_parts(
        key_size: int, parts: tuple[tuple[int, int], ...],
        ) -> tuple[int, ...]:
    """从有序物理行严格恢复完整键，未知负 index 与混合表示一律拒绝。"""
    if type(key_size) is not int or key_size <= 0:
        raise IntegerCodecError("identity key_size 必须为正整数")
    if not isinstance(parts, tuple) or not parts:
        raise IntegerCodecError("identity header 没有完整键载荷")
    for part in parts:
        if (not isinstance(part, tuple) or len(part) != 2
                or any(type(value) is not int for value in part)):
            raise IntegerCodecError("identity part 必须为两个严格整数")
    if parts[0][0] < 0:
        return _decode_packed_parts(key_size, parts)
    if len(parts) != key_size:
        raise IntegerCodecError("identity part 数量与 key_size 不一致")
    for expected, (index, _) in enumerate(parts):
        if index != expected:
            raise IntegerCodecError("identity part_index 不连续")
    return tuple(value for _, value in parts)


def _decode_packed_parts(
        key_size: int, parts: tuple[tuple[int, int], ...],
        ) -> tuple[int, ...]:
    """先核元数据和物理边界，再恢复规范字节，禁止按未核验长度分配。"""
    if key_size < IDENTITY_PART_PACKED_MIN_SIZE:
        raise IntegerCodecError("packed identity key_size 小于格式阈值")
    if (len(parts) < 3 or parts[0][0] != _BYTE_COUNT_INDEX
            or parts[1] != (_CODEC_INDEX, IDENTITY_PART_CODEC_PACKED_V1)):
        raise IntegerCodecError("packed identity marker 或 codec 不完整")
    byte_count = parts[0][1]
    if byte_count <= 0:
        raise IntegerCodecError("packed identity 字节数必须为正整数")
    word_count = (byte_count + _WORD_BYTES - 1) // _WORD_BYTES
    if word_count + 2 >= key_size:
        raise IntegerCodecError("packed identity 未减少 part 总行数")
    if len(parts) != word_count + 2:
        raise IntegerCodecError("packed identity word 数量与字节数不一致")
    data = bytearray()
    remaining = byte_count
    for expected, (index, word) in enumerate(parts[2:]):
        if index != expected:
            raise IntegerCodecError("packed identity word index 断序或重复")
        if not 0 <= word <= _MAX_WORD:
            raise IntegerCodecError("packed identity word 必须位于 0..2**56-1")
        width = min(_WORD_BYTES, remaining)
        if word >> (width * 8):
            raise IntegerCodecError("packed identity 末 word 高位填充必须为零")
        data.extend(word.to_bytes(width, "little"))
        remaining -= width
    key = decode_integer_tuple(bytes(data))
    if len(key) != key_size:
        raise IntegerCodecError("packed identity key_size 与规范流计数不一致")
    return key
