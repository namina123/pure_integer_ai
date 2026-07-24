"""storage.spaces.companion — 伴随库（§十五决策1·非整数合法）。

最低优先度·**接受非整数**（原输入文本/未验证内容·TEXT 介质）·sign=0 隔离不参与默认边计算/检索。
形式与存储介质可与核心/记忆不同（TEXT 文件库）。一般用户只存原输入；高级可选存未验证记忆。
惰性挂载·不进核心纯整数热路径·按 space_id 索引·可任意多个。

text_assoc TEXT 表：伴随空间正当形态（非可移植性债·决策1 伴随非整数合法化）。
经 backend 抽象注册为非核心扩展表（register_extension_table·L1·allow_text）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pure_integer_ai.crosscut.determinism.hasher import Hasher
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage import discipline as disc
from pure_integer_ai.storage.backend import (
    StorageBackend, TYPE_INT, TYPE_TEXT, register_extension_table,
)
from pure_integer_ai.storage.edge_types import EDGE_QUARANTINE_LINK
from pure_integer_ai.storage.node_store import TIER_SHADOW
from pure_integer_ai.storage.spaces.registry import (
    SPACE_TYPE_COMPANION,
    SpaceIdentity,
    SpaceRegistry,
)

# text_assoc：伴随文本关联表（非整数 TEXT·合法·非核心扩展表）
TEXT_ASSOC_TABLE = "text_assoc"
_TEXT_ASSOC_COLUMNS = [
    ("space_id", TYPE_INT),
    ("assoc_id", TYPE_INT),
    ("text_hash", TYPE_INT),   # Hasher.h63(text)·整数索引（纯整数热路径只扫 hash）
    ("text", TYPE_TEXT),       # 原输入文本（TEXT·伴随非整数合法）
    ("meta", TYPE_INT),        # 整数元信息（来源标记等）
]
_TEXT_ASSOC_INDEXES = [
    ("space_id",),
    ("space_id", "assoc_id"),
    ("space_id", "text_hash"),
]


class CompanionIntegrityError(RuntimeError):
    """伴随空间身份或 assoc 行出现缺失、重复和冲突。"""


@dataclass(frozen=True, order=True)
class CompanionAssocIdentity:
    """由稳定空间身份和局部 assoc 序号组成的非图引用。"""

    space: SpaceIdentity
    assoc_id: int

    def __post_init__(self) -> None:
        assert_int(self.assoc_id, _where="CompanionAssocIdentity.assoc_id")
        if self.space.space_type != SPACE_TYPE_COMPANION or self.assoc_id <= 0:
            raise ValueError("Companion assoc 身份非法")

    def stable_key(self) -> tuple[int, int, int, int]:
        """返回不依赖运行时 space_id 的稳定整数键。"""
        return (*self.space.stable_key(), self.assoc_id)


def register_companion_table(backend: StorageBackend) -> None:
    """注册 text_assoc 为非核心扩展表（L1·allow_text·非整数合法）。"""
    register_extension_table(backend, TEXT_ASSOC_TABLE, _TEXT_ASSOC_COLUMNS,
                             discipline=disc.DISC_APPEND_ONLY,
                             indexes=_TEXT_ASSOC_INDEXES,
                             recovery_key=("space_id", "assoc_id"))


class CompanionSpace:
    """伴随库（非整数 TEXT·sign=0 隔离·惰性挂载·可任意多个）。

    伴随项留档不删（append-only 守检疫可回溯）；伴随→记忆检疫闸靠跨 space 边关联非移动。
    """

    def __init__(self, registry: SpaceRegistry, backend: StorageBackend,
                 space_id: int) -> None:
        """绑定已注册 Companion 空间并恢复当前 assoc 水位。"""
        if registry.backend is not backend:
            raise CompanionIntegrityError("Companion registry 与 backend 不一致")
        assert_int(space_id, _where="CompanionSpace.space_id")
        if type(space_id) is not int or space_id <= 0:
            raise ValueError("Companion space_id 必须为严格正整数")
        self.registry = registry
        self.backend = backend
        self.space_id = space_id
        space_rows = backend.select("space", where={"space_id": space_id})
        if len(space_rows) != 1:
            raise CompanionIntegrityError("CompanionSpace 没有唯一空间注册行")
        row = space_rows[0]
        identity_values = (
            row.get("type"), row.get("type_hash"), row.get("name_hash"))
        try:
            assert_int(*identity_values, _where="CompanionSpace.identity")
        except (TypeError, ValueError) as exc:
            raise CompanionIntegrityError("Companion 空间稳定身份非法") from exc
        if (any(type(value) is not int for value in identity_values)
                or row["type"] != SPACE_TYPE_COMPANION):
            raise CompanionIntegrityError("CompanionSpace 未绑定有效空间注册行")
        self.identity = SpaceIdentity(
            row["type"], row["type_hash"], row["name_hash"])
        rows = backend.select(
            TEXT_ASSOC_TABLE,
            where={"space_id": space_id},
            order_by="assoc_id",
            descending=True,
            limit=1,
        )
        self._next_assoc = rows[0]["assoc_id"] if rows else 0

    @classmethod
    def create(cls, registry: SpaceRegistry, name: str) -> "CompanionSpace":
        """按稳定空间名幂等创建或重新挂载 Companion。"""
        sid = registry.register(SPACE_TYPE_COMPANION, name)
        return cls(registry, registry.backend, sid)

    def _alloc_assoc_id(self) -> int:
        """与持久层最高 assoc 水位对齐后分配，支持多实例交错写。"""
        rows = self.backend.select(
            TEXT_ASSOC_TABLE,
            where={"space_id": self.space_id},
            order_by="assoc_id",
            descending=True,
            limit=1,
        )
        persisted = rows[0]["assoc_id"] if rows else 0
        self._next_assoc = max(self._next_assoc, persisted) + 1
        return self._next_assoc

    def put_text(self, text: str, meta: int = 0) -> int:
        """存原输入文本·返回 assoc_id。text_hash 供纯整数热路径索引（不扫文本）。"""
        if not isinstance(text, str):
            raise TypeError("Companion text 必须是字符串")
        assert_int(meta, _where="CompanionSpace.put_text.meta")
        if type(meta) is not int:
            raise ValueError("Companion meta 必须是严格整数")
        aid = self._alloc_assoc_id()
        th = Hasher("pure_integer_ai.companion.v1").h63(text)
        self.backend.insert(TEXT_ASSOC_TABLE, {
            "space_id": self.space_id, "assoc_id": aid,
            "text_hash": th, "text": text, "meta": meta,
        })
        return aid

    def assoc_identity(self, assoc_id: int) -> CompanionAssocIdentity:
        """返回经唯一行核验的稳定 assoc 身份。"""
        self.read(assoc_id)
        return CompanionAssocIdentity(self.identity, assoc_id)

    def read(self, assoc_id: int) -> dict[str, Any]:
        """按局部序号回读唯一伴随项，并核验文本 hash。"""
        assert_int(assoc_id, _where="CompanionSpace.read.assoc_id")
        if type(assoc_id) is not int or assoc_id <= 0:
            raise ValueError("assoc_id 必须为严格正整数")
        rows = self.backend.select(TEXT_ASSOC_TABLE, where={
            "space_id": self.space_id,
            "assoc_id": assoc_id,
        })
        if len(rows) != 1:
            raise CompanionIntegrityError("Companion assoc 没有唯一记录")
        row = rows[0]
        values = (
            row.get("space_id"), row.get("assoc_id"),
            row.get("text_hash"), row.get("meta"))
        try:
            assert_int(*values, _where="CompanionSpace.read.row")
        except (TypeError, ValueError) as exc:
            raise CompanionIntegrityError("Companion assoc 整数字段非法") from exc
        if (any(type(value) is not int for value in values)
                or not isinstance(row.get("text"), str)):
            raise CompanionIntegrityError("Companion assoc 存储类型非法")
        if Hasher("pure_integer_ai.companion.v1").h63(row["text"]) != row["text_hash"]:
            raise CompanionIntegrityError("Companion assoc 文本 hash 不一致")
        return row

    def lookup_by_hash(self, text_hash: int) -> list[dict[str, Any]]:
        """经 hash 反查文本（纯整数热路径先算 hash 再反查·不扫全文本）。"""
        return self.backend.select(TEXT_ASSOC_TABLE,
                                   where={"space_id": self.space_id,
                                          "text_hash": text_hash})

    def all_items(self) -> list[dict[str, Any]]:
        """按 assoc 序返回当前 Companion 的全部留档项。"""
        return self.backend.select(TEXT_ASSOC_TABLE,
                                   where={"space_id": self.space_id},
                                   order_by="assoc_id")


def build_quarantine_link(edge_store: Any, *, from_companion: tuple[int, int],
                          to_memory: tuple[int, int]) -> None:
    """C9-ter 跨 space 检疫关联边（伴随→记忆·检疫过闸留档可回溯）。

    字段契约（C9-ter·edge_store 复用既有 edge 表不单建）：
      edge_type=EDGE_QUARANTINE_LINK / source=SOURCE_QUARANTINE / tier=SHADOW（检疫留档非已验证语义边·不进默认 A1/PR）
      / strength=0（检疫关联非学习对象）/ sn=tn=0（不接 reward 反传）/ memory_time_attach=NULL（跨 space 结构关联非记忆时序经验）
      / order_index=role=NULL / epistemic_origin=subtype=NULL。
    insert-only 守规：strength/tier 恒定不 update（靠调用方守规非 discipline 强制·disc=DISC_MUTABLE_MONOTONE 允许 insert+update 拒 delete）。

    caller：伴随 staging 候选（sign=0）检疫过闸晋升记忆项时调·记忆空间激活（C9-quater 两层 dump）后真触发。
    M10 第一刀 11a：MemorySpace 已生产实例化（formal_train make_train_context 挂 TrainContext）·但
    build_quarantine_link caller 仍 defer（检疫过闸晋升编排器 = promote 层·M10 第二刀 11b）·constructor 就绪 + 单测覆盖。
    """
    from pure_integer_ai.storage.edge_store import SOURCE_QUARANTINE
    csid, clid = from_companion
    msid, mlid = to_memory
    edge_store.add(
        space_id_from=csid, local_id_from=clid,
        space_id_to=msid, local_id_to=mlid,
        edge_type=EDGE_QUARANTINE_LINK, strength=0,
        source=SOURCE_QUARANTINE, tier=TIER_SHADOW,
    )
