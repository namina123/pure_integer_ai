"""storage.spaces.registry — 空间注册 + A5 space_name_hash（§十五决策1）。

space 表：space_id(编址键) / type(空间类型) / type_hash / name_hash（A5 整数编址）。
文本名入伴随库（守"文本不入核心"）；核心/记忆纯整数空间表只存 hash 整数。
编址仍 (space_id, local_id) 整数 tuple（吸收保留）；hash 作 space 身份标识非编址键。
每空间 id_pool 自增（backend.next_id）。
"""
from __future__ import annotations

from typing import Any

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.crosscut.determinism.hasher import Hasher
from pure_integer_ai.storage import discipline as disc
from pure_integer_ai.storage.backend import StorageBackend, TYPE_INT

SPACE_TYPE_CORE = 1
SPACE_TYPE_MEMORY = 2
SPACE_TYPE_COMPANION = 3

_SPACE_COLUMNS = [
    ("space_id", TYPE_INT),
    ("type", TYPE_INT),
    ("type_hash", TYPE_INT),
    ("name_hash", TYPE_INT),
]


def register_space_table(backend: StorageBackend) -> None:
    backend.register_table(
        "space", _SPACE_COLUMNS,
        disc.DISC_APPEND_ONLY, [("space_id",), ("type_hash", "name_hash")],
        core=True,
    )


class SpaceRegistry:
    """空间注册表。每空间分配 space_id + A5 hash；文本名经伴随库反查（守纯整数）。"""

    def __init__(self, backend: StorageBackend) -> None:
        self._b = backend
        self.backend = backend  # 暴露供 Space 持有
        self._next_space_id = 0

    def _alloc_space_id(self) -> int:
        self._next_space_id += 1
        return self._next_space_id

    @staticmethod
    def _hashes(space_type: int, name: str) -> tuple[int, int]:
        """A5：type_hash = Hasher.h63(space_type)；name_hash = Hasher.h63(name)。
        文本 name 仅用于 hash·不入核心表（守"文本不入核心"）。"""
        h = Hasher("pure_integer_ai.space.v1")
        return h.h63(space_type), h.h63(name)

    def register(self, space_type: int, name: str) -> int:
        """注册一个空间·返回 space_id。文本 name 仅 hash·不入核心表。"""
        assert_int(space_type, _where="SpaceRegistry.register.type")
        if space_type not in (SPACE_TYPE_CORE, SPACE_TYPE_MEMORY, SPACE_TYPE_COMPANION):
            raise ValueError(f"未知空间类型: {space_type}")
        space_id = self._alloc_space_id()
        type_hash, name_hash = self._hashes(space_type, name)
        self._b.insert("space", {
            "space_id": space_id, "type": space_type,
            "type_hash": type_hash, "name_hash": name_hash,
        })
        return space_id

    def get(self, space_id: int) -> dict[str, Any] | None:
        rows = self._b.select("space", where={"space_id": space_id}, limit=1)
        return rows[0] if rows else None

    def lookup_by_hash(self, type_hash: int, name_hash: int) -> int | None:
        """经 hash 反查 space_id（文本名经伴随库反查·此处只整数 hash）。"""
        rows = self._b.select("space",
                              where={"type_hash": type_hash, "name_hash": name_hash},
                              limit=1)
        return rows[0]["space_id"] if rows else None

    def all_spaces(self) -> list[dict[str, Any]]:
        return self._b.select("space", order_by="space_id")
