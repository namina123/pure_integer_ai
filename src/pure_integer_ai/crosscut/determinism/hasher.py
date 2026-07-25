"""crosscut.determinism.hasher — 带种子的 FNV-1a 63-bit 哈希 + canonical 编码。

Hasher(seed)：FNV-1a 64-bit 带种子哈希；canonical 编码（int/bytes/str/tuple/
Rational/FixedQuotient/dataclass）保证跨宿主 bit 一致。

h() 返全 64-bit 无符号；h63() 掩 63-bit（存 SQLite INTEGER 一律用 h63——SQLite
INTEGER 是有符号 64-bit，≥2^63 会变负数，掩 63-bit 保证跨宿主一致）。统一契约：存库用 h63。
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

from pure_integer_ai.crosscut.integer.valtypes import Rational, FixedQuotient

_MASK64 = (1 << 64) - 1
_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3


# ---- canonical 编码 ----

def _encode_int(n: int) -> bytes:
    nbytes = max(1, (n.bit_length() + 8) // 8)
    return b"\x02" + n.to_bytes(nbytes, "big", signed=True)


def _append_encoded(out: bytearray, value: Any) -> None:
    """把一个值的 canonical 编码追加到缓冲区，避免递归 bytes 拼接复制。"""
    if type(value) is int:
        out.extend(_encode_int(value))
        return
    if isinstance(value, bool):
        out.extend(b"\x01\x01" if value else b"\x01\x00")
        return
    if isinstance(value, int):
        out.extend(_encode_int(value))
        return
    if isinstance(value, bytes):
        out.extend(b"\x03")
        out.extend(len(value).to_bytes(8, "big"))
        out.extend(value)
        return
    if isinstance(value, str):
        encoded = value.encode("utf-8")
        out.extend(b"\x04")
        out.extend(len(encoded).to_bytes(8, "big"))
        out.extend(encoded)
        return
    if value is None:
        out.extend(b"\x06")
        return
    if isinstance(value, Rational):
        out.extend(b"\x07")
        out.extend(_encode_int(value.num))
        out.extend(_encode_int(value.den))
        return
    if isinstance(value, FixedQuotient):
        out.extend(b"\x08")
        out.extend(_encode_int(value.M))
        out.extend(_encode_int(value.r))
        out.extend(_encode_int(value.k))
        out.extend(_encode_int(value.b))
        return
    if isinstance(value, (tuple, list)):
        out.extend(b"\x05")
        out.extend(len(value).to_bytes(8, "big"))
        for item in value:
            _append_encoded(out, item)
        return
    if isinstance(value, dict):
        # canonical：按键排序（dict 插入序跨宿主不保证一致·排序保 bit 一致）
        encoded_keys = sorted(
            ((_encode(key), key) for key in value),
            key=lambda item: item[0],
        )
        out.extend(b"\x0A")
        out.extend(len(value).to_bytes(8, "big"))
        for encoded_key, key in encoded_keys:
            out.extend(encoded_key)
            _append_encoded(out, value[key])
        return
    if dataclasses.is_dataclass(value):
        out.extend(b"\x09")
        _append_encoded(out, tuple(
            getattr(value, field.name) for field in dataclasses.fields(value)
        ))
        return
    raise TypeError(f"Hasher 不支持编码类型: {type(value)!r}")


def _encode(value: Any) -> bytes:
    """返回与历史格式逐字节相同的 canonical 编码。"""
    out = bytearray()
    _append_encoded(out, value)
    return bytes(out)


def _fnv1a(data: bytes, h: int) -> int:
    for byte in data:
        h ^= byte
        h = (h * _FNV_PRIME) & _MASK64
    return h


def _fnv1a_int(value: int, state: int) -> int:
    """把严格整数按既有 canonical 编码直接续入 FNV 状态。"""
    state ^= 0x02
    state = (state * _FNV_PRIME) & _MASK64
    size = max(1, (value.bit_length() + 8) // 8)
    return _fnv1a(value.to_bytes(size, "big", signed=True), state)


def _fnv1a_tuple_header(size: int, state: int) -> int:
    """把 tuple 类型标记和八字节长度按既有格式续入 FNV 状态。"""
    state ^= 0x05
    state = (state * _FNV_PRIME) & _MASK64
    return _fnv1a(size.to_bytes(8, "big"), state)


@dataclass(frozen=True)
class TupleHashPrefix:
    """保存 tuple 公共前缀的 FNV 状态，并对剩余项续算同一稳定哈希。"""

    _state: int
    _remaining_items: int

    def h(self, suffix: tuple[Any, ...]) -> int:
        """续算完整 64-bit 哈希；suffix 数量必须补足预声明 tuple。"""
        if not isinstance(suffix, tuple):
            raise TypeError("tuple 哈希 suffix 必须是 tuple")
        if len(suffix) != self._remaining_items:
            raise ValueError("tuple 哈希 suffix 数量与预声明长度不一致")
        state = self._state
        for item in suffix:
            state = _fnv1a(_encode(item), state)
        return state

    def h63(self, suffix: tuple[Any, ...]) -> int:
        """续算并返回可存 SQLite INTEGER 的 63-bit 稳定哈希。"""
        return self.h(suffix) & ((1 << 63) - 1)


class Hasher:
    """带种子的 FNV-1a 64-bit 哈希。h(value) 不改状态，纯函数式。"""

    def __init__(self, seed: Any):
        self._iv = _fnv1a(_encode(seed), _FNV_OFFSET)

    def h(self, value: Any) -> int:
        return _fnv1a(_encode(value), self._iv)

    def h63(self, value: Any) -> int:
        """掩 63-bit（存 SQLite INTEGER 用，跨宿主一致）。"""
        return _fnv1a(_encode(value), self._iv) & ((1 << 63) - 1)

    def h63_tagged_int_tuple(
            self, tag: int, values: tuple[int, ...]) -> int:
        """流式计算 ``(tag, values)``，结果与通用 canonical 编码逐位相同。"""
        if type(tag) is not int:
            raise TypeError("tag 必须是严格整数")
        if not isinstance(values, tuple):
            raise TypeError("values 必须是严格整数元组")
        for index, value in enumerate(values):
            if type(value) is not int:
                raise TypeError(f"values[{index}] 必须是严格整数")
        state = _fnv1a_tuple_header(2, self._iv)
        state = _fnv1a_int(tag, state)
        state = _fnv1a_tuple_header(len(values), state)
        for value in values:
            state = _fnv1a_int(value, state)
        return state & ((1 << 63) - 1)

    def prepare_tuple_prefix(
            self, total_items: int,
            prefix: tuple[Any, ...]) -> TupleHashPrefix:
        """预哈希 tuple 公共前缀，后续不同 suffix 仍得到原 canonical 哈希。"""
        if type(total_items) is not int or total_items < 0:
            raise ValueError("tuple 总项数必须为非负严格整数")
        if not isinstance(prefix, tuple):
            raise TypeError("tuple 哈希 prefix 必须是 tuple")
        if len(prefix) > total_items:
            raise ValueError("tuple 哈希 prefix 不得长于总项数")
        state = _fnv1a(
            b"\x05" + total_items.to_bytes(8, "big"),
            self._iv,
        )
        for item in prefix:
            state = _fnv1a(_encode(item), state)
        return TupleHashPrefix(state, total_items - len(prefix))
