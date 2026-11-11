"""大 generic GraphObject 的规范 varint 流与七字节整数 word 物理格式。

本模块只处理有限整数序列，不接触数据库、来源语义、对象 hash 或运行时缓存。
"""
from __future__ import annotations

from pure_integer_ai.storage.integer_codec import (
    IntegerCodecError,
    decode_integer_tuple,
    encode_integer_tuple,
    strict_integer_tuple,
)


GRAPH_OBJECT_CODEC_GENERIC_PACKED_V1 = 5
PACKED_MIN_COMPONENTS = 128
PACKED_WORD_BYTES = 7
PACKED_INLINE_WORDS = 7
_MAX_WORD = (1 << (PACKED_WORD_BYTES * 8)) - 1


def encode_packed_components(
        components: tuple[int, ...],
        ) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]] | None:
    """仅为满足阈值且严格减少子行的大键生成固定槽和连续 word 子载荷。"""
    strict_integer_tuple(components, label="packed graph components")
    if len(components) < PACKED_MIN_COMPONENTS:
        return None
    data = encode_integer_tuple(components)
    word_count = (len(data) + PACKED_WORD_BYTES - 1) // PACKED_WORD_BYTES
    if max(word_count - PACKED_INLINE_WORDS, 0) >= len(components) - 8:
        return None
    words = tuple(int.from_bytes(data[start:start + PACKED_WORD_BYTES], "little")
                  for start in range(0, len(data), PACKED_WORD_BYTES))
    inline = words[:PACKED_INLINE_WORDS]
    values = (len(data), *inline, *((0,) * (PACKED_INLINE_WORDS - len(inline))))
    overflow = tuple(enumerate(words[PACKED_INLINE_WORDS:],
                               start=PACKED_INLINE_WORDS))
    return values, overflow


def _word(value: int) -> int:
    """要求物理 word 是可跨有符号整数列保存的非负七字节整数。"""
    if type(value) is not int or not 0 <= value <= _MAX_WORD:
        raise IntegerCodecError("packed graph word 必须位于 0..2**56-1")
    return value


def decode_packed_components(
        component_size: int, ref_hash: int, values: tuple[int, ...],
        overflow: tuple[tuple[int, int], ...],
        ) -> tuple[int, ...]:
    """严格恢复完整整数序列，拒绝缺块、混合表示、填充和非规范 varint。"""
    if type(component_size) is not int or component_size < PACKED_MIN_COMPONENTS:
        raise IntegerCodecError("packed graph component_size 小于格式阈值")
    if type(ref_hash) is not int or ref_hash != 0:
        raise IntegerCodecError("packed graph ref 必须为零")
    strict_integer_tuple(values, label="packed graph codec values")
    if len(values) != PACKED_INLINE_WORDS + 1 or values[0] <= 0:
        raise IntegerCodecError("packed graph 固定槽或字节数不完整")
    byte_count = values[0]
    word_count = (byte_count + PACKED_WORD_BYTES - 1) // PACKED_WORD_BYTES
    overflow_count = max(word_count - PACKED_INLINE_WORDS, 0)
    if overflow_count >= component_size - 8:
        raise IntegerCodecError("packed graph 未减少 component 子行")
    if not isinstance(overflow, tuple) or len(overflow) != overflow_count:
        raise IntegerCodecError("packed graph component word 数量不一致")
    inline_count = min(word_count, PACKED_INLINE_WORDS)
    if any(value != 0 for value in values[1 + inline_count:]):
        raise IntegerCodecError("packed graph 未使用固定槽必须为零")
    words = [_word(value) for value in values[1:1 + inline_count]]
    for ordinal, part in enumerate(overflow, start=PACKED_INLINE_WORDS):
        if not isinstance(part, tuple) or len(part) != 2:
            raise IntegerCodecError("packed graph component 记录形状错误")
        if type(part[0]) is not int or part[0] != ordinal:
            raise IntegerCodecError("packed graph component ordinal 断序或重复")
        words.append(_word(part[1]))
    data = bytearray()
    remaining = byte_count
    for word in words:
        width = min(PACKED_WORD_BYTES, remaining)
        if word >> (width * 8):
            raise IntegerCodecError("packed graph 末 word 高位填充必须为零")
        data.extend(word.to_bytes(width, "little"))
        remaining -= width
    components = decode_integer_tuple(bytes(data))
    if len(components) != component_size:
        raise IntegerCodecError("packed graph component_size 与规范流计数不一致")
    return components
