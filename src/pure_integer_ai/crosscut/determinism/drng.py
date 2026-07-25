"""crosscut.determinism.drng — 确定性 RNG（splitmix64·唯一随机入口·禁 random）。

splitmix64：seed → 64-bit 整数流。纯整数运算，跨宿主 bit 一致。
所有"带种子随机"调用方只经它取随机（核心无 random）。
"""
from __future__ import annotations

_MASK64 = (1 << 64) - 1


class DRNG:
    """splitmix64：seed → 64-bit 整数流。纯整数运算，跨宿主 bit 一致。"""

    _GAMMA = 0x9E3779B97F4A7C15

    def __init__(self, seed: int) -> None:
        self.state = seed & _MASK64

    def next(self) -> int:
        self.state = (self.state + self._GAMMA) & _MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
        z = (z ^ (z >> 31)) & _MASK64
        return z

    def randbelow(self, n: int) -> int:
        """[0, n) 内的确定性整数。"""
        if n <= 0:
            raise ValueError("randbelow: n 须为正")
        return self.next() % n
