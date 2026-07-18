"""crosscut.integer.unicode_codec — 码点（unicode ordinal）边缘编解码。

文本 = unicode 码点（整数）的有序数组（用户从始至终不变量·计算机数据本身即整数序数集）。
本模块是**核心↔外缘**的唯一 chr/ord 边缘协议：核心只存/算码点整数（纯整数铁律守）·
chr/ord 仅在此 + I/O 边缘发生（不入核心计算）。

  encode(text)  : str  → tuple[int, ...]   （每字符 ord·有序·不可变）
  decode(cps)   : 序列 → str               （每码点 chr·拼接）

**为何 tuple 非 list**：hashable + 不可变 → caller 无法 mutate 破 bit-identical（list 可变若被下游
改写会引入跨引用别名 bug）。返 tuple 是"纯整数 + 确定性"铁律的操作化。

**BMP 外字符**（如 𝄞 U+1D11E）：Python PEP 393 下 str 是码点序列（非 UTF-16 单元）·`ord(ch)` 一对一
返完整码点·`chr(cp)` 一对一还原·**无 surrogate pair 特殊处理**（避 archive 旧 3-byte serialize 丢
BMP 外字符的坑）。跨宿主一致（码点是 unicode 标准·非宿主编码）。

**重写非搬**：`_archive/mock_v1/utils/unicode_codec.py` 的 hash_sequence/serialize/serialize_compact
不复刻（content_hash 已由 `Hasher.h63` 统一·本模块只管编解码边缘）。

铁律：纯整数（返码点 int）/ 确定性（同输入同输出·无随机）/ 跨宿主一致（码点标准·非编码依赖）/
只 stdlib（chr/ord 内建）/ 单向依赖（L1 crosscut·零业务依赖）。
"""
from __future__ import annotations

from typing import Iterable


def encode(text: str) -> tuple[int, ...]:
    """文本 → 码点有序整数组（每字符 ord·I/O 边缘协议·核心存算皆整数）。

    确定性：同 str 同 tuple（ord 是确定性映射）。空串 → ()。
    """
    if not isinstance(text, str):
        raise TypeError(f"unicode_codec.encode 仅接受 str·got {type(text).__name__}")
    return tuple(ord(ch) for ch in text)


def decode(codepoints: Iterable[int]) -> str:
    """码点有序整数组 → 文本（每码点 chr·I/O 边缘协议·核心存算皆整数）。

    确定性：同序列同 str（chr 是确定性映射）。空序列 → ""。
    注意：caller（surface_of resolver）负责把"无 correspondence 行"映射为 None·**不应对空序列
    调 decode 当 None**（decode(())→"" 是合法空串·非 None·混用会破 judge J2s truthiness 不变量）。
    """
    return "".join(chr(int(cp)) for cp in codepoints)
