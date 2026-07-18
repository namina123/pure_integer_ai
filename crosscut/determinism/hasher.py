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


def _encode(value: Any) -> bytes:
    if isinstance(value, bool):
        return b"\x01" + (b"\x01" if value else b"\x00")
    if isinstance(value, int):
        return _encode_int(value)
    if isinstance(value, bytes):
        return b"\x03" + len(value).to_bytes(8, "big") + value
    if isinstance(value, str):
        b = value.encode("utf-8")
        return b"\x04" + len(b).to_bytes(8, "big") + b
    if value is None:
        return b"\x06"
    if isinstance(value, Rational):
        return b"\x07" + _encode_int(value.num) + _encode_int(value.den)
    if isinstance(value, FixedQuotient):
        return (b"\x08" + _encode_int(value.M) + _encode_int(value.r)
                + _encode_int(value.k) + _encode_int(value.b))
    if isinstance(value, (tuple, list)):
        out = b"\x05" + len(value).to_bytes(8, "big")
        for item in value:
            out += _encode(item)
        return out
    if isinstance(value, dict):
        # canonical：按键排序（dict 插入序跨宿主不保证一致·排序保 bit 一致）
        out = b"\x0A" + len(value).to_bytes(8, "big")
        for k in sorted(value.keys(), key=lambda x: _encode(x)):
            out += _encode(k) + _encode(value[k])
        return out
    if dataclasses.is_dataclass(value):
        return b"\x09" + _encode(tuple(getattr(value, f.name)
                                        for f in dataclasses.fields(value)))
    raise TypeError(f"Hasher 不支持编码类型: {type(value)!r}")


def _fnv1a(data: bytes, h: int) -> int:
    for byte in data:
        h ^= byte
        h = (h * _FNV_PRIME) & _MASK64
    return h


class Hasher:
    """带种子的 FNV-1a 64-bit 哈希。h(value) 不改状态，纯函数式。"""

    def __init__(self, seed: Any):
        self._iv = _fnv1a(_encode(seed), _FNV_OFFSET)

    def h(self, value: Any) -> int:
        return _fnv1a(_encode(value), self._iv)

    def h63(self, value: Any) -> int:
        """掩 63-bit（存 SQLite INTEGER 用，跨宿主一致）。"""
        return _fnv1a(_encode(value), self._iv) & ((1 << 63) - 1)
