"""纯整数内容指纹编码等价性与严格类型专项。"""
from __future__ import annotations

import hashlib

import pytest

from pure_integer_ai.crosscut.determinism.fingerprint import (
    integer_tuple_fingerprint,
)


def _reference_fingerprint(
        values: tuple[int, ...], *, domain: str,
        ) -> tuple[int, ...]:
    """按冻结 v1 字节协议独立编码，防止快速路径改变现役 SHA。"""
    domain_bytes = domain.encode("utf-8")
    digest = hashlib.sha256()
    digest.update(b"pure_integer_ai.integer_tuple_fingerprint.v1\x00")
    digest.update(len(domain_bytes).to_bytes(8, "big"))
    digest.update(domain_bytes)
    digest.update(len(values).to_bytes(8, "big"))
    for value in values:
        size = max(1, (value.bit_length() + 8) // 8)
        digest.update(size.to_bytes(8, "big"))
        digest.update(value.to_bytes(size, "big", signed=True))
    return 1, len(values), *digest.digest()


def test_integer_tuple_fingerprint_fast_path_preserves_v1_transport():
    """小整数缓存、符号边界和任意精度整数必须逐字节等价。"""
    boundaries = (
        -(1 << 80), -32769, -32768, -129, -128, -1, 0, 1,
        127, 128, 255, 256, 32767, 32768, 1 << 80,
    )
    values = tuple(
        (*boundaries, *(index % 256 for index in range(20_000))))
    expected = _reference_fingerprint(values, domain="指纹等价性.v1")
    assert integer_tuple_fingerprint(
        values, domain="指纹等价性.v1") == expected


def test_integer_tuple_fingerprint_still_rejects_non_strict_integer():
    """快速路径不得把 bool 或其他整数相似值放入核心指纹。"""
    with pytest.raises(TypeError, match=r"values\[1\]"):
        integer_tuple_fingerprint((1, False), domain="strict.v1")
