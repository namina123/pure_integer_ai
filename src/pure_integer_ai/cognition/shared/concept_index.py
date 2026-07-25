"""cognition.shared.concept_index — 概念身份索引（hash→local_id·per-space dedup）。

concept_node 表无 text/hash 列（决策3扩列只 modality_marker/tier·纯整数概念点）。
概念身份靠 **content_hash**（Hasher.h63 of surface/QID）·per-space dedup：
  同 space 内同 hash → 同概念点（不重建·tier 单调升）；不同 space 同 hash → 不同点
  （核心 vs 记忆是不同空间·正确·永不合并节点 A8）。

  surface 文本 → 伴随库（text_assoc·"文本不入核心"）·概念点只存整数。

  **跨 run identity 持久化（Task #475·§8.7-idx）**：_index in-memory run-scoped·load_run
  不重建 → 载入算子不可 inline + observe 续训后建重复概念点（latent corrupt）。修：concept_identity
  扩展表持久化 (space_id, local_id, content_hash)·lazy per-space 重建（_ensure_space_loaded·首次
  access 触发·post-load_run）。ensure 新建后 record_concept_identity（best-effort·bare fixture skip）。

_ensure_concept 是 observe 4 入口共用原语（§十一缺口#3·OOV 一出现就建概念点）。
"""
from __future__ import annotations

from typing import Any

from pure_integer_ai.crosscut.determinism.hasher import Hasher
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.crosscut.integer.unicode_codec import encode as _encode_codepoints
from pure_integer_ai.storage.backend import StorageBackend
from pure_integer_ai.storage.node_store import NodeStore, TIER_PRIMARY, TIER_SHADOW
from pure_integer_ai.storage.concept_identity import (
    load_space_identity, record_concept_identity,
)
from pure_integer_ai.storage.concept_correspondence import (
    record_correspondence as _record_correspondence, CORR_ORDINAL,
)

_CONCEPT_HASHER = Hasher("pure_integer_ai.concept.v1")


class ConceptIdentityConflict(ValueError):
    """旧 surface 身份已绑定到不同节点类型时拒绝静默复用。"""


def content_hash(surface: str | int) -> int:
    """surface → 整数 hash（str 走 h63·int 直接用·QID/synset 整数 id）。"""
    if isinstance(surface, int):
        return _CONCEPT_HASHER.h63(surface)
    return _CONCEPT_HASHER.h63(surface)


class ConceptIndex:
    """概念身份索引（per-space hash→local_id·dedup·tier 单调）。

    持有 NodeStore + 各 space 的 hash→local_id 映射。companion 可选（surface 文本留档）。
    跨 run identity：lazy per-space 从 concept_identity 表重建（_ensure_space_loaded·首次 access）。
    """

    def __init__(self, backend: StorageBackend, companion: Any = None) -> None:
        self._b = backend
        self._nodes = NodeStore(backend)
        self._companion = companion  # CompanionSpace | None（surface 文本留档）
        # per-space: {space_id: {content_hash: local_id}}
        self._index: dict[int, dict[int, int]] = {}
        # 已 lazy 重建的 space 集（首次 access per space 触发 _ensure_space_loaded·once）
        self._loaded_spaces: set[int] = set()

    def _ensure_space_loaded(self, space_id: int) -> None:
        """lazy 从 concept_identity 表重建 _index[space_id]（once per space·post-load_run 跨 run identity）。

        首次 access 该 space（ensure/lookup）时扫 concept_identity WHERE space_id → 补 _index[space]。
        表未注册（bare fixture）→ load_space_identity 返空·_index 由 ensure 内存建（向后兼容 bit-identical）。
        后续 ensure 直写 _index + concept_identity 同步（_loaded_spaces 守 once·无重复扫）。
        """
        if space_id in self._loaded_spaces:
            return
        self._loaded_spaces.add(space_id)
        loaded = load_space_identity(self._b, space_id)
        if loaded:
            space_map = self._index.setdefault(space_id, {})
            for ch, lid in loaded.items():
                space_map.setdefault(ch, lid)   # first-wins（内存既有优先·lazy 载补缺·无重复 hash 故序无关）

    def ensure(self, surface: str | int, *, space_id: int,
               tier: int = TIER_SHADOW, node_type: int = 1) -> tuple[int, int]:
        """_ensure_concept：dedup 建概念点·返回 (space_id, local_id)。

        同 space 同 hash → 返既有（tier 单调升 = max(旧, 新)·MUTABLE_MONOTONE）。
        surface 文本（str）入伴随库留档（若有 companion）。
        新建概念点 → record_concept_identity 持久化（跨 run 重建·best-effort·§8.7-idx）。

        modality 不在节点列（§7.7.1 路径 B·modality_marker 迁 abstract_mark MARK_MODALITY）·
        caller 需标模态 → ensure 后显式 set_mark(MARK_MODALITY, modality)（首版无 caller·bit-identical）。
        """
        assert_int(space_id, tier, node_type,
                   _where="ConceptIndex.ensure")
        self._ensure_space_loaded(space_id)   # lazy 重建（post-load_run·载入概念点 dedup）
        ch = content_hash(surface)
        space_map = self._index.setdefault(space_id, {})
        if ch in space_map:
            lid = space_map[ch]
            row = self._nodes.get(space_id, lid)
            if row is None:
                raise ConceptIdentityConflict("surface 身份指向不存在的概念节点")
            # 旧 API 历史上不把 node_type 纳入身份；新代码须走 ensure_typed 或分型图对象。
            # tier 单调升（§十二⑤·概念点 tier=max 其边 tier）
            self._bump_tier(space_id, lid, tier)
            return (space_id, lid)
        # 新建概念点
        lid = self._b.next_id(space_id)
        self._nodes.put(space_id, lid, node_type=node_type,
                        tier=tier)
        space_map[ch] = lid
        # 跨 run identity 持久化（§8.7-idx·best-effort·bare fixture 表未注册则 skip 向后兼容）
        record_concept_identity(self._b, space_id=space_id, local_id=lid, content_hash=ch)
        # P0a：码点序数对应（surface=str 时写·best-effort·bare fixture skip·bit-identical additive）。
        # 一个抽象概念(local_id) ↔ 一条 ordinal 对应（码点有序数组）·surface_of resolver 读之产文本。
        # UNGATED always-write（同 record_concept_identity·纯 additive 数据无既有消费者·读写 gate 分离
        # 守 dump 完整性：gate OFF 亦写→续训 dump 含文本·gate 只控读消费）。int surface（QID/synset）不写。
        if isinstance(surface, str):
            _record_correspondence(self._b, space_id=space_id, local_id=lid,
                                   corr_kind=CORR_ORDINAL,
                                   codepoints=_encode_codepoints(surface))
        # surface 文本入伴随库留档（若有 companion·"文本不入核心"）
        if self._companion is not None and isinstance(surface, str):
            self._companion.put_text(surface)
        return (space_id, lid)

    def ensure_typed(self, surface: str | int, *, space_id: int,
                     tier: int = TIER_SHADOW,
                     node_type: int = 1) -> tuple[int, int]:
        """严格创建旧节点；同 surface 已绑定异型节点时拒绝复用。"""
        existing = self.lookup(surface, space_id)
        if existing is not None:
            row = self._nodes.get(*existing)
            if row is None:
                raise ConceptIdentityConflict("surface 身份指向不存在的概念节点")
            if row["type"] != node_type:
                raise ConceptIdentityConflict(
                    "同一旧 surface 身份不得同时充当不同节点类型；"
                    "必须使用分型对象或显式迁移桥")
        return self.ensure(
            surface, space_id=space_id, tier=tier, node_type=node_type)

    def lookup(self, surface: str | int, space_id: int) -> tuple[int, int] | None:
        """查既有概念点（不建）。lazy 重建守跨 run identity（载入算子可 inline·§8.7-idx）。"""
        self._ensure_space_loaded(space_id)   # lazy 重建（post-load_run·载入算子 lookup 命中）
        ch = content_hash(surface)
        lid = self._index.get(space_id, {}).get(ch)
        return (space_id, lid) if lid is not None else None

    def lookup_typed(self, surface: str | int, space_id: int, *,
                     node_type: int) -> tuple[int, int] | None:
        """按旧 surface 和节点类型查找；同键异型时 fail closed。"""
        assert_int(space_id, node_type, _where="ConceptIndex.lookup_typed")
        ref = self.lookup(surface, space_id)
        if ref is None:
            return None
        row = self._nodes.get(*ref)
        if row is None:
            raise ConceptIdentityConflict("surface 身份指向不存在的概念节点")
        if row["type"] != node_type:
            raise ConceptIdentityConflict(
                "旧 surface 已绑定其他节点类型，不能按请求类型静默读取")
        return ref

    def _bump_tier(self, space_id: int, local_id: int, new_tier: int) -> None:
        """tier 单调升（MUTABLE_MONOTONE·只升不降·NodeStore.set_tier 守）。"""
        if new_tier <= TIER_SHADOW:
            return  # SHADOW 是最低·无需升
        try:
            self._nodes.set_tier(space_id, local_id, new_tier)
        except Exception:
            # set_tier 内部已守单调（降级抛 MonotoneViolation）·此处忽略降级尝试
            pass


__all__ = ["ConceptIdentityConflict", "ConceptIndex", "content_hash"]
