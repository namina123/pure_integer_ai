"""为运行期聚合对象提供域分离、固定长度的纯整数内容引用。"""
from __future__ import annotations

import hashlib


INTEGER_TUPLE_FINGERPRINT_VERSION = 1


def integer_tuple_fingerprint(
        values: tuple[int, ...],
        *,
        domain: str,
        ) -> tuple[int, ...]:
    """把严格整数元组流式编码为带版本和原长度的 SHA-256 内容引用。"""
    if not isinstance(values, tuple):
        raise TypeError("fingerprint values 必须是 tuple")
    if not isinstance(domain, str) or not domain:
        raise ValueError("fingerprint domain 必须是非空字符串")
    domain_bytes = domain.encode("utf-8")
    digest = hashlib.sha256()
    digest.update(b"pure_integer_ai.integer_tuple_fingerprint.v1\x00")
    digest.update(len(domain_bytes).to_bytes(8, "big"))
    digest.update(domain_bytes)
    digest.update(len(values).to_bytes(8, "big"))
    for index, value in enumerate(values):
        if type(value) is not int:
            raise TypeError(f"fingerprint values[{index}] 必须是严格整数")
        size = max(1, (value.bit_length() + 8) // 8)
        encoded = value.to_bytes(size, "big", signed=True)
        digest.update(size.to_bytes(8, "big"))
        digest.update(encoded)
    return (
        INTEGER_TUPLE_FINGERPRINT_VERSION,
        len(values),
        *digest.digest(),
    )


__all__ = [
    "INTEGER_TUPLE_FINGERPRINT_VERSION",
    "integer_tuple_fingerprint",
]
