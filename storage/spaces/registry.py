"""storage.spaces.registry — 空间注册 + A5 space_name_hash（§十五决策1）。

space 表：space_id(编址键) / type(空间类型) / type_hash / name_hash（A5 整数编址）。
文本名入伴随库（守"文本不入核心"）；核心/记忆纯整数空间表只存 hash 整数。
编址仍 (space_id, local_id) 整数 tuple（吸收保留）；hash 作 space 身份标识非编址键。
每空间 id_pool 自增（backend.next_id）。
"""
from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True, order=True)
class SpaceIdentity:
    """空间类型和名称哈希组成的稳定整数身份。"""

    space_type: int
    type_hash: int
    name_hash: int

    def __post_init__(self) -> None:
        """拒绝未知空间类型、负哈希和布尔值进入稳定空间身份。"""
        assert_int(
            self.space_type, self.type_hash, self.name_hash,
            _where="SpaceIdentity",
        )
        if any(type(value) is not int for value in self.stable_key()):
            raise ValueError("SpaceIdentity 必须使用严格整数")
        if self.space_type not in {
                SPACE_TYPE_CORE, SPACE_TYPE_MEMORY, SPACE_TYPE_COMPANION}:
            raise ValueError("SpaceIdentity.space_type 未注册")
        if self.type_hash < 0 or self.name_hash < 0:
            raise ValueError("SpaceIdentity 哈希不得为负数")

    def stable_key(self) -> tuple[int, int, int]:
        """返回空间稳定键。"""
        return self.space_type, self.type_hash, self.name_hash


def register_space_table(backend: StorageBackend) -> None:
    backend.register_table(
        "space", _SPACE_COLUMNS,
        disc.DISC_APPEND_ONLY, [("space_id",), ("type_hash", "name_hash")],
        core=True,
        recovery_key=("space_id",),
    )


class SpaceRegistry:
    """空间注册表。每空间分配 space_id + A5 hash；文本名经伴随库反查（守纯整数）。"""

    def __init__(self, backend: StorageBackend) -> None:
        """绑定后端，并从既有空间行恢复分配水位。"""
        self._b = backend
        self.backend = backend  # 暴露供 Space 持有
        rows = backend.select(
            "space", order_by="space_id", descending=True, limit=1)
        self._next_space_id = rows[0]["space_id"] if rows else 0

    def _alloc_space_id(self) -> int:
        """按本地与持久层较高水位分配，避免多 registry 交错碰撞。"""
        rows = self._b.select(
            "space", order_by="space_id", descending=True, limit=1)
        persisted = rows[0]["space_id"] if rows else 0
        self._next_space_id = max(self._next_space_id, persisted) + 1
        return self._next_space_id

    @staticmethod
    def _hashes(space_type: int, name: str) -> tuple[int, int]:
        """A5：type_hash = Hasher.h63(space_type)；name_hash = Hasher.h63(name)。
        文本 name 仅用于 hash·不入核心表（守"文本不入核心"）。"""
        h = Hasher("pure_integer_ai.space.v1")
        return h.h63(space_type), h.h63(name)

    @classmethod
    def identity_for(cls, space_type: int, name: str) -> SpaceIdentity:
        """计算空间稳定身份，不分配运行时 space_id。"""
        assert_int(space_type, _where="SpaceRegistry.identity_for.type")
        if space_type not in (SPACE_TYPE_CORE, SPACE_TYPE_MEMORY, SPACE_TYPE_COMPANION):
            raise ValueError(f"未知空间类型: {space_type}")
        type_hash, name_hash = cls._hashes(space_type, name)
        return SpaceIdentity(space_type, type_hash, name_hash)

    def register(self, space_type: int, name: str) -> int:
        """按稳定身份幂等注册空间，并返回不冲突的运行时 space_id。"""
        assert_int(space_type, _where="SpaceRegistry.register.type")
        if space_type not in (SPACE_TYPE_CORE, SPACE_TYPE_MEMORY, SPACE_TYPE_COMPANION):
            raise ValueError(f"未知空间类型: {space_type}")
        identity = self.identity_for(space_type, name)
        existing = self._b.select("space", where={
            "type": identity.space_type,
            "type_hash": identity.type_hash,
            "name_hash": identity.name_hash,
        })
        if len(existing) > 1:
            raise ValueError("空间稳定身份存在重复注册行")
        if existing:
            return existing[0]["space_id"]
        space_id = self._alloc_space_id()
        self._b.insert("space", {
            "space_id": space_id, "type": space_type,
            "type_hash": identity.type_hash, "name_hash": identity.name_hash,
        })
        return space_id

    def get(self, space_id: int) -> dict[str, Any] | None:
        rows = self._b.select("space", where={"space_id": space_id}, limit=1)
        return rows[0] if rows else None

    def identity(self, space_id: int) -> SpaceIdentity:
        """严格恢复唯一空间身份，拒绝缺行、重复编址和损坏字段。"""
        assert_int(space_id, _where="SpaceRegistry.identity.space_id")
        if type(space_id) is not int or space_id <= 0:
            raise ValueError("space_id 必须为严格正整数")
        rows = self._b.select("space", where={"space_id": space_id})
        if len(rows) != 1:
            raise ValueError("space_id 没有唯一注册行")
        row = rows[0]
        return SpaceIdentity(
            row.get("type"), row.get("type_hash"), row.get("name_hash"))

    def lookup_by_hash(self, type_hash: int, name_hash: int) -> int | None:
        """经 hash 反查 space_id（文本名经伴随库反查·此处只整数 hash）。"""
        rows = self._b.select("space",
                              where={"type_hash": type_hash, "name_hash": name_hash},
                              limit=1)
        return rows[0]["space_id"] if rows else None

    def all_spaces(self) -> list[dict[str, Any]]:
        return self._b.select("space", order_by="space_id")
