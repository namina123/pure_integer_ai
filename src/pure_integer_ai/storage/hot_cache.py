"""storage.hot_cache — HotCache 查询结果 LRU 缓存（§十五决策7·修 defer_indexes 必修）。

HotCache 是**查询结果 LRU 缓存装饰器**（非分页器·决策7 设计层债补第3条认对载体）：
装饰 backend.select·同 (table, where, where_gt, order_by, limit) 命中返缓存·
写透传失效同表缓存。纯整数 LRU·key 含全维度。

**决策7必修 bug（已修）**：ensure_index 现在带 defer_indexes kwarg（旧 hot_cache.py:133 缺
→ 接 prod 传 defer_indexes=True 直接 TypeError）。本实现 ensure_index 经 backend·
defer_indexes 透传·批量加载时延迟建索引。

接线 defer（决策7）：HotCache 骨架留·接线随 cognition 层热路径（Stage 3+）·Stage 1 提供机制。
真分页载体是 cognition 层 HotZone（非本模块·决策7第3条）。
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Any

from pure_integer_ai.storage.backend import StorageBackend


class HotCache:
    """查询结果 LRU 缓存装饰器（装饰 backend.select·写透传失效）。

    capacity: 缓存条数上限（纯整数 LRU·cap 驱逐最久未用）。
    """

    def __init__(self, backend: StorageBackend, capacity: int = 4096) -> None:
        self._b = backend
        self._cap = capacity
        self._cache: OrderedDict[tuple, list[dict[str, Any]]] = OrderedDict()
        # 表 → 是否有缓存条目（写透传失效用·同表写即失效该表所有缓存）
        self._table_keys: dict[str, set[tuple]] = {}

    def select(self, table: str, where: dict[str, Any] | None = None,
               where_gt: dict[str, int] | None = None,
               order_by: str | None = None, *, descending: bool = False,
               limit: int | None = None) -> list[dict[str, Any]]:
        """经缓存的 select：命中返缓存拷贝·未命中走 backend 并缓存。"""
        key = self._key(table, where, where_gt, order_by, descending, limit)
        if key in self._cache:
            self._cache.move_to_end(key)  # LRU touch
            return [dict(r) for r in self._cache[key]]  # 拷贝防外部篡改缓存
        rows = self._b.select(table, where, where_gt, order_by,
                              descending=descending, limit=limit)
        self._put(key, table, rows)
        return rows

    def invalidate(self, table: str) -> None:
        """写透传失效：同表写后失效该表所有缓存条目。"""
        for k in self._table_keys.get(table, ()):
            self._cache.pop(k, None)
        self._table_keys.pop(table, None)

    def ensure_index(self, table: str, columns: tuple[str, ...],
                     *, defer_indexes: bool = False) -> None:
        """建索引·defer_indexes 透传 backend（决策7必修 bug 已修·接 prod 无 TypeError）。"""
        self._b.ensure_index(table, columns, defer_indexes=defer_indexes)
        # 建索引不改变数据·缓存仍有效（索引是查询加速·非数据变更）

    def _key(self, table, where, where_gt, order_by, descending, limit) -> tuple:
        return (table, _freeze(where), _freeze(where_gt), order_by,
                descending, limit)

    def _put(self, key, table, rows) -> None:
        self._cache[key] = rows
        self._table_keys.setdefault(table, set()).add(key)
        if len(self._cache) > self._cap:
            old_key, _ = self._cache.popitem(last=False)  # LRU 驱逐
            # 清 table_keys 索引
            for ks in self._table_keys.values():
                ks.discard(old_key)

    # 写透传：装饰 insert/update/delete·先执行再失效同表缓存
    def insert(self, table: str, row: dict[str, Any]) -> None:
        self._b.insert(table, row)
        self.invalidate(table)

    def update(self, table: str, where: dict[str, Any],
               set_: dict[str, Any]) -> int:
        n = self._b.update(table, where, set_)
        self.invalidate(table)
        return n

    def delete(self, table: str, where: dict[str, Any]) -> int:
        n = self._b.delete(table, where)
        self.invalidate(table)
        return n


def _freeze(obj: Any) -> tuple:
    """把 where/where_gt dict 冻结为可哈希 key（排序 tuple）。None → ()。"""
    if obj is None:
        return ()
    return tuple(sorted(obj.items()))
