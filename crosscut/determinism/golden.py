"""crosscut.determinism.golden — append-only golden 快照库（只增·序号单调）。

assert_reproducible 通过则写 golden 三元组 (seq, seed_hash, value_hash)。
永不覆盖/删除；完整性校验序号单调递增无重复。
"""
from __future__ import annotations

from typing import Any

from pure_integer_ai.crosscut.determinism.hasher import Hasher


class _Golden:
    def __init__(self) -> None:
        self._records: list[tuple[int, int, int]] = []
        self._seq = 0

    def record(self, seed: Any, value: Any, value_hash: int) -> int:
        """只追加，返回序号。永不覆盖/删除。"""
        self._seq += 1
        self._records.append((self._seq, Hasher(0).h63(seed), value_hash))
        return self._seq

    def count(self) -> int:
        return len(self._records)

    def verify(self) -> bool:
        """序号单调递增（append-only 完整性）。"""
        seqs = [s for s, _, _ in self._records]
        return seqs == sorted(seqs) and len(seqs) == len(set(seqs))


golden = _Golden()
