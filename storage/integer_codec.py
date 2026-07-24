"""存储协议共用的规范整数流编码和分帧读取。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.crosscut.guards.int_blocker import assert_int


class IntegerCodecError(ValueError):
    """整数流版本、varint、分帧或规范编码不完整。"""


def strict_integer_tuple(
        value: tuple[int, ...], *, label: str, empty: bool = False,
        ) -> tuple[int, ...]:
    """核验协议值为严格整数 tuple，并按调用方要求处理空值。"""
    if not isinstance(value, tuple) or (not empty and not value):
        raise ValueError(f"{label} 必须是{'可空' if empty else '非空'}整数 tuple")
    if value:
        assert_int(*value, _where=label)
        if any(type(item) is not int for item in value):
            raise ValueError(f"{label} 必须使用严格整数")
    return value


def encode_integer_tuple(values: tuple[int, ...]) -> bytes:
    """把严格整数 tuple 编成带元素计数的规范 zigzag varint 字节。"""
    strict_integer_tuple(values, label="integer codec values", empty=True)
    encoded = bytearray()
    _append_unsigned(encoded, len(values))
    for value in values:
        unsigned = value * 2 if value >= 0 else (-value * 2) - 1
        _append_unsigned(encoded, unsigned)
    return bytes(encoded)


def decode_integer_tuple(data: bytes) -> tuple[int, ...]:
    """解码规范整数流，并拒绝截断、尾随字节和非最短 varint。"""
    if not isinstance(data, bytes) or not data:
        raise IntegerCodecError("integer codec data 必须是非空 bytes")
    cursor = 0
    size, cursor = _read_unsigned(data, cursor)
    result: list[int] = []
    for _ in range(size):
        unsigned, cursor = _read_unsigned(data, cursor)
        result.append(
            unsigned // 2 if unsigned % 2 == 0
            else -((unsigned + 1) // 2)
        )
    if cursor != len(data):
        raise IntegerCodecError("integer codec data 存在尾随字节")
    restored = tuple(result)
    if encode_integer_tuple(restored) != data:
        raise IntegerCodecError("integer codec data 不是规范编码")
    return restored


def pack_key(result: list[int], value: tuple[int, ...]) -> None:
    """把可变长度整数键按长度分帧追加到目标流。"""
    strict_integer_tuple(value, label="packed integer key", empty=True)
    result.extend((len(value), *value))


def _append_unsigned(target: bytearray, value: int) -> None:
    """按最短 unsigned varint 形式追加一个非负整数。"""
    if type(value) is not int or value < 0:
        raise ValueError("unsigned varint 只能编码非负严格整数")
    while value >= 128:
        target.append((value & 127) | 128)
        value >>= 7
    target.append(value)


def _read_unsigned(data: bytes, cursor: int) -> tuple[int, int]:
    """从指定位置读取一个最短 unsigned varint 及其后继位置。"""
    start = cursor
    value = 0
    shift = 0
    while True:
        if cursor >= len(data):
            raise IntegerCodecError("unsigned varint 被截断")
        byte = data[cursor]
        cursor += 1
        value |= (byte & 127) << shift
        if byte < 128:
            break
        shift += 7
    canonical = bytearray()
    _append_unsigned(canonical, value)
    if bytes(canonical) != data[start:cursor]:
        raise IntegerCodecError("unsigned varint 不是最短编码")
    return value, cursor


@dataclass
class IntegerStreamReader:
    """在已解码整数 tuple 上执行带边界检查的顺序读取。"""

    values: tuple[int, ...]
    cursor: int = 0

    def __post_init__(self) -> None:
        """核验输入流和初始游标，禁止越界起步。"""
        strict_integer_tuple(
            self.values, label="integer stream reader values", empty=True)
        if type(self.cursor) is not int or not 0 <= self.cursor <= len(self.values):
            raise ValueError("integer stream reader cursor 越界")

    def read(self, *, label: str) -> int:
        """读取一个严格整数，流已结束时报告字段名。"""
        if self.cursor >= len(self.values):
            raise IntegerCodecError(f"整数流缺少字段: {label}")
        value = self.values[self.cursor]
        self.cursor += 1
        return value

    def read_nonnegative(self, *, label: str) -> int:
        """读取一个非负严格整数。"""
        value = self.read(label=label)
        if value < 0:
            raise IntegerCodecError(f"{label} 必须是非负整数")
        return value

    def read_positive(self, *, label: str) -> int:
        """读取一个正严格整数。"""
        value = self.read(label=label)
        if value <= 0:
            raise IntegerCodecError(f"{label} 必须是正整数")
        return value

    def read_key(self, *, label: str, empty: bool = False) -> tuple[int, ...]:
        """读取一个长度分帧整数键并核验是否允许为空。"""
        size = self.read_nonnegative(label=f"{label}.size")
        end = self.cursor + size
        if end > len(self.values):
            raise IntegerCodecError(f"{label} 被截断")
        value = self.values[self.cursor:end]
        self.cursor = end
        return strict_integer_tuple(value, label=label, empty=empty)

    def finish(self) -> None:
        """要求调用方已经消费完整流，拒绝未知尾字段。"""
        if self.cursor != len(self.values):
            raise IntegerCodecError("整数流存在未消费尾字段")


__all__ = [
    "IntegerCodecError",
    "IntegerStreamReader",
    "decode_integer_tuple",
    "encode_integer_tuple",
    "pack_key",
    "strict_integer_tuple",
]
