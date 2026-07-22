"""storage.backend — 存储抽象协议 + DictBackend + SQLiteBackend（§十五决策8·重来最稳资产）。

StorageBackend 协议**不暴露任何 SQL**：
  register_table / ensure_index(defer_indexes=) / insert / update / select / delete / next_id
  where 仅等值；where_gt 单列范围；order_by 单列；limit 整数。无 JOIN/GROUP BY/OR。
cognition 层全经此抽象，绝不写 raw SQL（§7.3 可移植性债正面已偿付）。

纯整数：_validate_row 拒 float（核心表纯整数铁律）。
纪律下推 backend：register_table 挂 DISC，update/delete 经 check_write 闸门。
确定性有序读（A10）：order_by 是显式参数；DictBackend dict 保插入序（Python 3.7+ dict 有序）；
SQLiteBackend order_by 映射 SQL ORDER BY。同 order_by 两后端返同序（续训可复现）。

两实现：
  DictBackend   内存 dict（保插入序）·测试/确定性·零宿主依赖
  SQLiteBackend sqlite3 首版宿主·可换（决策8）
"""
from __future__ import annotations

import sqlite3
from typing import Any, Iterable, Protocol, runtime_checkable

from pure_integer_ai.crosscut.guards.float_guard import assert_no_float, FloatViolation
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage import discipline as disc
from pure_integer_ai.storage.edge_types import EDGE_IS_A   # leaf 模块（无 backend import·无环）·perf #1144 IS_A gen counter
from pure_integer_ai.storage.telemetry import active_backend_telemetry

# 列类型 → SQLite affinity（DictBackend 不用·仅 SQLiteBackend）。
TYPE_INT = "INT"
TYPE_TEXT = "TEXT"
_SQLITE_AFFINITY = {TYPE_INT: "INTEGER", TYPE_TEXT: "TEXT"}


class CoreStrViolation(disc.DisciplineViolation):
    """核心表行含 str 违例（文本不入核心·入伴随库·守纯整数）。"""


def _validate_row(row: dict[str, Any], *, allow_text: bool = False) -> None:
    """拒 float 值（纯整数铁律·raise FloatViolation 与 crosscut 守卫一致）。
    TEXT 列允许 str（仅伴随/审计非核心·allow_text 时）；核心表拒 str。"""
    for v in row.values():
        if isinstance(v, float):
            raise FloatViolation(f"行含 float 被拒（纯整数铁律）: {v!r}")
        if isinstance(v, str) and not allow_text:
            # 核心表不应存 str（文本入伴随库）。允许 None（可空列）。
            raise CoreStrViolation(
                f"核心表行含 str 被拒（文本不入核心·入伴随库）: {v!r}"
            )


@runtime_checkable
class StorageBackend(Protocol):
    """存储后端协议（无 raw SQL·纪律下推·纯整数）。"""

    def register_table(self, table: str, columns: list[tuple[str, str]],
                       discipline: int, indexes: list[tuple[str, ...]] = (),
                       *, core: bool = False, defer_indexes: bool = False) -> None: ...
    def ensure_index(self, table: str, columns: tuple[str, ...],
                     *, defer_indexes: bool = False) -> None: ...
    def insert(self, table: str, row: dict[str, Any]) -> None: ...
    def update(self, table: str, where: dict[str, Any],
               set_: dict[str, Any]) -> int: ...
    def select(self, table: str, where: dict[str, Any] | None = None,
               where_gt: dict[str, int] | None = None,
               order_by: str | None = None, *, descending: bool = False,
               limit: int | None = None) -> list[dict[str, Any]]: ...
    def count(self, table: str, where: dict[str, Any] | None = None) -> int: ...
    def delete(self, table: str, where: dict[str, Any]) -> int: ...
    def next_id(self, space_id: int) -> int: ...
    def commit(self) -> None: ...
    def close(self) -> None: ...


def register_extension_table(backend: StorageBackend, table: str,
                             columns: list[tuple[str, str]],
                             discipline: int = disc.DISC_NONE,
                             indexes: list[tuple[str, ...]] = ()) -> None:
    """非核心扩展表注册（L1 迁移·14 越层表归此·§十五决策8 残债收口）。

    core=False；挂纪律；guard 守。cognition 层扩展表经此注册，不直接建核心表。
    """
    backend.register_table(table, columns, discipline, indexes, core=False)


# ---- _BaseBackend：纪律闸门 + _validate_row + 表注册元数据 ----

class _BaseBackend:
    """backend 基类：持有表元数据（列名/纪律/core 标志）+ 写闸门。

    子类实现 _do_create_table / _do_ensure_index / _do_insert / _do_update /
    _do_select / _do_delete / _do_next_id 的物理存储。
    """

    def __init__(self) -> None:
        self._tables: dict[str, dict[str, Any]] = {}
        # id_pool: space_id → next_local_id（每空间自增·决策1编址）
        self._id_pool: dict[int, int] = {}
        # isa_edge_gen: space_id → IS_A 拓扑版本号（perf #1144·bump on EDGE_IS_A insert·ancestor_map cache O(1) 命中用）
        self._isa_edge_gen: dict[int, int] = {}

    # -- 表元数据 --
    def register_table(self, table: str, columns: list[tuple[str, str]],
                       discipline: int, indexes: list[tuple[str, ...]] = (),
                       *, core: bool = False, defer_indexes: bool = False) -> None:
        if table in self._tables:
            return  # 幂等
        col_names = [c[0] for c in columns]
        is_core = core or disc.is_core(table)
        self._tables[table] = {
            "columns": col_names,
            "col_types": {c[0]: c[1] for c in columns},
            "discipline": discipline,
            "core": is_core,
            "indexes": [tuple(index) for index in indexes],
        }
        self._do_create_table(table, columns)
        for idx in indexes:
            self._do_ensure_index(table, idx, defer_indexes=defer_indexes)

    def ensure_index(self, table: str, columns: tuple[str, ...],
                     *, defer_indexes: bool = False) -> None:
        if table not in self._tables:
            raise KeyError(f"ensure_index: 未注册表 {table!r}")
        self._do_ensure_index(table, columns, defer_indexes=defer_indexes)
        normalized = tuple(columns)
        indexes = self._tables[table].setdefault("indexes", [])
        if normalized not in indexes:
            indexes.append(normalized)

    def _meta(self, table: str) -> dict[str, Any]:
        m = self._tables.get(table)
        if m is None:
            raise KeyError(f"未注册表 {table!r}")
        return m

    # -- 写操作（经纪律闸门） --
    def insert(self, table: str, row: dict[str, Any]) -> None:
        telemetry = active_backend_telemetry()
        m = self._meta(table)
        allow_text = any(t == TYPE_TEXT for t in m["col_types"].values())
        _validate_row(row, allow_text=allow_text)
        disc.check_write(table, "insert", m["discipline"], m["core"])
        try:
            self._do_insert(table, row)
        except BaseException:
            if telemetry is not None:
                telemetry.record("insert", table, failed=True)
            raise
        if telemetry is not None:
            telemetry.record("insert", table, rows=1)
        # perf #1144：IS_A 拓扑版本 bump（ancestor_map cache O(1) 命中信号）。IS_A 拓扑仅经 insert 变
        # （update 只动 tier/strength/sn/tn 非拓扑·edge_store:320/361/373/379/419·IS_A reward-inert 无 reward update·
        # 无 delete）→ 任一 IS_A insert = 拓扑变。gen-match ⟺ ancestor_map 不变（bit-identical·abstraction.build_isa_ancestor_map 读）。
        if table == "edge" and row.get("edge_type") == EDGE_IS_A:
            _sid = row.get("space_id_from")
            if _sid is not None:
                self._isa_edge_gen[_sid] = self._isa_edge_gen.get(_sid, 0) + 1

    def update(self, table: str, where: dict[str, Any],
               set_: dict[str, Any]) -> int:
        telemetry = active_backend_telemetry()
        m = self._meta(table)
        allow_text = any(t == TYPE_TEXT for t in m["col_types"].values())
        # 增量元组 ("+=", n) 的 n 须纯整数；其余值经 _validate_row 拒 float/str
        for k, v in set_.items():
            if isinstance(v, tuple) and len(v) == 2 and v[0] == "+=":
                assert_int(v[1], _where=f"update increment {k}")
            else:
                _validate_row({k: v}, allow_text=allow_text)
        disc.check_write(table, "update", m["discipline"], m["core"])
        try:
            affected = self._do_update(table, where, set_)
        except BaseException:
            if telemetry is not None:
                telemetry.record("update", table, failed=True)
            raise
        if telemetry is not None:
            telemetry.record("update", table, rows=affected)
        return affected

    def delete(self, table: str, where: dict[str, Any]) -> int:
        telemetry = active_backend_telemetry()
        m = self._meta(table)
        disc.check_write(table, "delete", m["discipline"], m["core"])
        try:
            affected = self._do_delete(table, where)
        except BaseException:
            if telemetry is not None:
                telemetry.record("delete", table, failed=True)
            raise
        if telemetry is not None:
            telemetry.record("delete", table, rows=affected)
        return affected

    def select(self, table: str, where: dict[str, Any] | None = None,
               where_gt: dict[str, int] | None = None,
               order_by: str | None = None, *, descending: bool = False,
               limit: int | None = None) -> list[dict[str, Any]]:
        telemetry = active_backend_telemetry()
        self._meta(table)  # 存在性检查
        try:
            rows = self._do_select(
                table, where, where_gt, order_by, descending, limit)
        except BaseException:
            if telemetry is not None:
                telemetry.record("select", table, failed=True)
            raise
        if telemetry is not None:
            telemetry.record("select", table, rows=len(rows))
        return rows

    def count(self, table: str, where: dict[str, Any] | None = None) -> int:
        """计数匹配行（**零 copy**·dedup 存在性检查 / size 函数用·替代 select+len）。

        与 len(select(table, where)) **同整数 bit-identical**（同 where 等值过滤集）·但**不 copy 行**
        （select 每行 dict(r) copy·count 直返计数）·per-item dedup（add_cooccurs_dedup/add_precedes_dedup
        每潜在边一次存在性 select）+ size 函数（_graph_size/_edge_count select 全表只为 len）省海量 copy。
        不支持 where_gt/order_by/limit（dedup/size 不需·需者用 select）。纯读无 discipline 闸门（同 select）。
        """
        telemetry = active_backend_telemetry()
        self._meta(table)  # 存在性检查
        try:
            rows = self._do_count(table, where)
        except BaseException:
            if telemetry is not None:
                telemetry.record("count", table, failed=True)
            raise
        if telemetry is not None:
            telemetry.record("count", table, rows=rows)
        return rows

    def isa_edge_generation(self, space_id: int) -> int:
        """IS_A 拓扑版本号（perf cache signal·bump on EDGE_IS_A insert·#1144）。

        ancestor_map cache O(1) 命中用：IS_A 拓扑（child→parent 边集）**仅经 insert 变**——update 只动
        tier/strength/sn/tn 非拓扑字段（edge_store:320/361/373/379/419 where 含 edge_type·全验无 from/to/edge_type 改）·
        IS_A reward-inert（effective_weight:82 assert 挡 PR 外·reward_propagate:133 CAUSES-only）故无 reward update·
        全仓零 delete("edge")。∴ 任一 IS_A insert = 拓扑变 = bump。ancestor_map 仅依赖 from/to 拓扑·非 tier/strength·
        故 gen-match ⟺ ancestor_map 不变（bit-identical）。返 0 = 该 space 无 IS_A insert（首建/不支持·caller 退化 rebuild）。
        """
        return self._isa_edge_gen.get(space_id, 0)

    def next_id(self, space_id: int) -> int:
        """每空间 id_pool 自增（决策1 编址 (space_id, local_id)）。纯整数·确定。"""
        assert_int(space_id, _where="next_id.space_id")
        nid = self._id_pool.get(space_id, 0) + 1
        self._id_pool[space_id] = nid
        return nid

    def advance_id_pool(self, space_id: int, floor: int) -> None:
        """续训 load 后推高 id_pool 水位（序列7·防新分配撞已载 local_id）。

        next_id 从内存 _id_pool 自增·非存储表·load_run 载入已分配 concept_node 后须 rebaseline
        _id_pool[space] ≥ 已载 max local_id·否则续训新分配从 1 起→撞已载节点（DictBackend 静默
        dup 行 corrupt / SQLiteBackend PK crash·latent 续训 bug）。monotone max（多次调取最高·不降）。
        纯整数·确定（floor 序无关·max 不依赖扫描序·bit-identical）。
        """
        assert_int(space_id, floor, _where="advance_id_pool")
        if floor > self._id_pool.get(space_id, 0):
            self._id_pool[space_id] = floor

    def commit(self) -> None:
        pass

    def close(self) -> None:
        pass

    # -- 子类实现 --
    def _do_create_table(self, table, columns): raise NotImplementedError
    def _do_ensure_index(self, table, columns, *, defer_indexes=False): raise NotImplementedError
    def _do_insert(self, table, row): raise NotImplementedError
    def _do_update(self, table, where, set_): raise NotImplementedError
    def _do_select(self, table, where, where_gt, order_by, descending, limit): raise NotImplementedError
    def _do_count(self, table, where): raise NotImplementedError
    def _do_delete(self, table, where): raise NotImplementedError


# ---- DictBackend：内存 dict 保插入序（测试/确定性·零宿主） ----

class DictBackend(_BaseBackend):
    """内存 dict backend（保插入序·A10 确定性有序读基础）。

    内存索引（2026-07-09·cProfile 真修·解 #734 scaling）：register_table/ensure_index 声明的索引
    现真正落内存哈希桶（mirror SQLiteBackend ix_edge_* 等·两后端同 index 集·同 query 同结果）。
    _do_select/_do_update 优先用**覆盖索引**（索引 cols ⊆ where.keys·取列数最多=最选择性）缩候选·
    无覆盖则退全表扫。**纯 perf·桶内保插入序 → 同序同结果 → bit-identical-safe 无需 gate**
    （非 hub_set cache 改语义·cProfile 证 compute_hub_set 非瓶颈·总收口 §三簇2 hub_set cache 错修降级）。

    桶存**行 dict 引用**（非位置·_do_delete repack _data 不破引用）。维护不变量：**桶内序恒 == 插入序**
    （insert 追加双处保序 / update 索引列变 + delete + load_snapshot 从 _data rebuild 保序）。
    更新非索引列（strength/sn/tn ·热路径）只就地改·零 rebuild·O(候选)。
    """

    def __init__(self) -> None:
        super().__init__()
        self._data: dict[str, list[dict[str, Any]]] = {}
        # _idx: table → {cols_tuple → {value_tuple → [行 dict 引用·插入序]}}
        self._idx: dict[str, dict[tuple[str, ...], dict[tuple, list[dict[str, Any]]]]] = {}
        # _covering_idx_cache: (table, frozenset(where_keys)) → best_cols tuple | None（perf·
        # 索引 setup 后不变 → best_cols 纯函数·cache 命中跳迭代找索引·_do_ensure_index 清）。
        # **前提不变量**：idx cols 只增不减（仅 _do_ensure_index 加 cols·无单 cols 删除路径）·
        # 故 cache 的 best_cols 永指有效 cols。若未来加"删单 cols"功能须同时清本 cache（避 KeyError）。
        # best_cols 只影响 perf（选最选择性索引缩候选）不影响 correctness（caller 全 where 过滤保同集同序）。
        self._covering_idx_cache: dict[tuple[str, frozenset], tuple[str, ...] | None] = {}

    def _do_create_table(self, table, columns):
        self._data[table] = []

    def _do_ensure_index(self, table, columns, *, defer_indexes=False):
        if defer_indexes:
            return  # 批量加载延迟建索引（镜像 SQLiteBackend·决策7）
        idx = self._idx.setdefault(table, {})
        if columns in idx:
            return  # 幂等（镜像 SQLiteBackend CREATE INDEX IF NOT EXISTS）
        idx[columns] = {}
        self._rebuild_index(table, columns)   # 从已存行建（register 后已 insert / load_snapshot 后 ensure）
        self._covering_idx_cache.clear()   # 新索引改变 best_cols 选择 → 清 cache（setup 期·罕）

    def _row_idx_key(self, row: dict[str, Any], cols: tuple[str, ...]) -> tuple:
        """行 → 索引键（cols 顺序取值·缺列 None·确定可哈希）。"""
        return tuple(row.get(c) for c in cols)

    def _rebuild_index(self, table: str, cols: tuple[str, ...]) -> None:
        """从 _data[table] 重建单索引桶（插入序·确定）。

        update(索引列变)/delete/load_snapshot 用·保桶内序 == 插入序 → bit-identical。
        """
        bucket: dict[tuple, list[dict[str, Any]]] = {}
        for r in self._data[table]:
            bucket.setdefault(self._row_idx_key(r, cols), []).append(r)
        self._idx[table][cols] = bucket

    def _covering_candidates(self, table: str,
                             where: dict[str, Any] | None
                             ) -> list[dict[str, Any]] | None:
        """覆盖索引命中候选行（插入序）·无 where 或无覆盖索引返 None（caller 退全表扫）。

        覆盖判据：索引 cols ⊆ where.keys·取列数最多者（最选择性·候选最少）。桶内保插入序·
        caller 仍按全 where + where_gt 过滤剩余列 → 结果与全表扫**同集同序**（bit-identical）。
        """
        if not where:
            return None
        idx = self._idx.get(table)
        if not idx:
            return None
        where_keys = frozenset(where)
        # best covering cols 缓存（perf·索引 setup 后不变 -> best_cols 纯函数·命中跳迭代找索引）。
        # sentinel 0=未算（区分 None=算过但无覆盖索引）·_do_ensure_index 新索引时清。
        best_cols = self._covering_idx_cache.get((table, where_keys), 0)
        if best_cols == 0:
            best_cols = None
            _best_len = -1
            for cols in idx:   # idx 插入序·first-max-wins（同原 best_cols is None or len> 语义）
                if frozenset(cols) <= where_keys and len(cols) > _best_len:
                    best_cols = cols
                    _best_len = len(cols)
            self._covering_idx_cache[(table, where_keys)] = best_cols
        if best_cols is None:
            return None
        bucket = idx[best_cols]
        return bucket.get(tuple(where[c] for c in best_cols), [])

    def _do_insert(self, table, row):
        r = dict(row)
        self._data[table].append(r)
        idx = self._idx.get(table)
        if idx:
            # 追加到每索引桶尾·保桶内序 == 插入序（与 rebuild 等价·增量维护）
            for cols, bucket in idx.items():
                bucket.setdefault(self._row_idx_key(r, cols), []).append(r)

    def _do_update(self, table, where, set_):
        data = self._data[table]
        idx = self._idx.get(table)
        cands = self._covering_candidates(table, where)
        rows = cands if cands is not None else data
        # set_ 是否改了任何索引列（是→受影响索引须 rebuild 保桶序 == 插入序）
        dirty = {c for cols in (idx or {}) for c in cols} & set(set_)
        n = 0
        for r in rows:
            if where and not all(r.get(k) == v for k, v in where.items()):
                continue
            for k, v in set_.items():
                if isinstance(v, tuple) and len(v) == 2 and v[0] == "+=":
                    r[k] = (r.get(k) or 0) + v[1]
                else:
                    r[k] = v
            n += 1
        if n and dirty and idx is not None:
            # 仅 rebuild 列被改的索引（如 set_tier 改 tier 只 rebuild (tier,)·热路径 strength/sn/tn dirty=空零 rebuild）
            for cols in list(idx):
                if set(cols) & dirty:
                    self._rebuild_index(table, cols)
        return n

    def _do_select(self, table, where, where_gt, order_by, descending, limit):
        cands = self._covering_candidates(table, where)
        src = cands if cands is not None else self._data[table]
        out: list[dict[str, Any]] = []
        for r in src:
            if where and not all(r.get(k) == v for k, v in where.items()):
                continue
            if where_gt and not all(r.get(k, 0) > v for k, v in where_gt.items()):
                continue
            out.append(dict(r))
        if order_by is not None:
            # None 值排序稳定：放末尾（升序）/首（降序）保持确定
            out.sort(key=lambda r: (r.get(order_by) is None, r.get(order_by)),
                     reverse=descending)
        if limit is not None:
            out = out[:limit]
        return out

    def _do_count(self, table, where):
        # 零 copy 计数（镜像 _do_select 的 _covering_candidates 缩候选·但不 dict(r)·只 sum(1)）。
        # where=None → len(_data)（O(1)·size 函数用）·有覆盖索引 → bucket 内过滤·否则全表扫计数。
        # 与 len(_do_select(...)) 同集同计数 bit-identical（同 where 等值过滤）。
        cands = self._covering_candidates(table, where)
        if cands is None:
            src = self._data[table]
            if not where:
                return len(src)
            return sum(1 for r in src if all(r.get(k) == v for k, v in where.items()))
        # covering 命中（where 必非空·_covering_candidates 无 where 返 None）·仍按全 where 过滤
        # （覆盖索引 cols 可能 ⊂ where keys·镜像 _do_select 保同集）。
        return sum(1 for r in cands if all(r.get(k) == v for k, v in where.items()))

    def _do_delete(self, table, where):
        data = self._data[table]
        kept = [r for r in data if not all(r.get(k) == v for k, v in where.items())]
        deleted = len(data) - len(kept)
        self._data[table] = kept
        if deleted and self._idx.get(table):
            # _data repack 致位置失效·桶存引用仍真·但桶内集变了→从新 _data rebuild 全索引保一致
            for cols in list(self._idx[table]):
                self._rebuild_index(table, cols)
        return deleted

    # 持久化/恢复（dump 续训·DictBackend 内存态快照）
    def snapshot(self) -> dict[str, list[dict[str, Any]]]:
        return {t: [dict(r) for r in rows] for t, rows in self._data.items()}

    def load_snapshot(self, snap: dict[str, list[dict[str, Any]]]) -> None:
        self._data = {t: [dict(r) for r in rows] for t, rows in snap.items()}
        # _data 已换·旧桶引用指向旧行（已弃）→ 清已不存在的表 + 对仍存在表从新 _data rebuild 全索引
        for t in [t for t in self._idx if t not in self._data]:
            del self._idx[t]
        for table, idx in self._idx.items():
            for cols in list(idx):
                self._rebuild_index(table, cols)


# ---- SQLiteBackend：sqlite3 首版宿主（可换·决策8） ----

class SQLiteBackend(_BaseBackend):
    """sqlite3 backend。SQL 仅在本模块内部生成·绝不暴露给 cognition 层。"""

    def __init__(self, path: str = ":memory:") -> None:
        super().__init__()
        self._conn = sqlite3.connect(path)
        self._conn.execute("PRAGMA foreign_keys=ON")

    def _q(self, name: str) -> str:
        return '"' + name.replace('"', '""') + '"'

    def _equality_clauses(self, where: dict[str, Any] | None
                          ) -> tuple[list[str], list[Any]]:
        """把结构化等值条件转换为 SQL；None 必须使用 IS NULL。"""
        clauses: list[str] = []
        params: list[Any] = []
        for column, value in (where or {}).items():
            if value is None:
                clauses.append(f"{self._q(column)} IS NULL")
            else:
                clauses.append(f"{self._q(column)}=?")
                params.append(value)
        return clauses, params

    def _do_create_table(self, table, columns):
        cols = ", ".join(
            f"{self._q(c)} {_SQLITE_AFFINITY.get(t, 'INTEGER')}" for c, t in columns
        )
        sql = f"CREATE TABLE IF NOT EXISTS {self._q(table)} ({cols})"
        self._conn.execute(sql)

    def _do_ensure_index(self, table, columns, *, defer_indexes=False):
        if defer_indexes:
            return  # 批量加载时延迟建索引（性能·决策7）
        idx_name = "idx_" + table + "_" + "_".join(columns)
        cols = ", ".join(self._q(c) for c in columns)
        self._conn.execute(
            f"CREATE INDEX IF NOT EXISTS {self._q(idx_name)} ON {self._q(table)} ({cols})"
        )

    def _do_insert(self, table, row):
        cols = ", ".join(self._q(c) for c in row)
        ph = ", ".join("?" for _ in row)
        self._conn.execute(
            f"INSERT INTO {self._q(table)} ({cols}) VALUES ({ph})",
            tuple(row.values()),
        )

    def _do_update(self, table, where, set_):
        set_parts, set_params = [], []
        for c, v in set_.items():
            if isinstance(v, tuple) and len(v) == 2 and v[0] == "+=":
                set_parts.append(f"{self._q(c)}={self._q(c)}+?")
                set_params.append(v[1])
            else:
                set_parts.append(f"{self._q(c)}=?")
                set_params.append(v)
        set_clause = ", ".join(set_parts)
        where_parts, where_params = self._equality_clauses(where)
        where_clause = " AND ".join(where_parts) or "1=1"
        cur = self._conn.execute(
            f"UPDATE {self._q(table)} SET {set_clause} WHERE {where_clause}",
            tuple(set_params) + tuple(where_params),
        )
        return cur.rowcount

    def _do_select(self, table, where, where_gt, order_by, descending, limit):
        clauses, params = self._equality_clauses(where)
        if where_gt:
            for k, v in where_gt.items():
                clauses.append(f"{self._q(k)}>?")
                params.append(v)
        where_sql = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        order_sql = ""
        if order_by is not None:
            order_sql = f" ORDER BY {self._q(order_by)} IS NULL, {self._q(order_by)}"
            if descending:
                order_sql = f" ORDER BY {self._q(order_by)} IS NULL DESC, {self._q(order_by)} DESC"
        limit_sql = f" LIMIT {int(limit)}" if limit is not None else ""
        cur = self._conn.execute(
            f"SELECT * FROM {self._q(table)}{where_sql}{order_sql}{limit_sql}",
            tuple(params),
        )
        col_names = [d[0] for d in cur.description]
        return [dict(zip(col_names, r)) for r in cur.fetchall()]

    def _do_count(self, table, where):
        # SELECT COUNT(*)（零行 fetch·镜像 DictBackend._do_count 语义·dedup/size 用）。
        clauses, params = self._equality_clauses(where)
        where_sql = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        cur = self._conn.execute(
            f"SELECT COUNT(*) FROM {self._q(table)}{where_sql}", tuple(params))
        return cur.fetchone()[0]

    def _do_delete(self, table, where):
        where_parts, where_params = self._equality_clauses(where)
        where_clause = " AND ".join(where_parts) or "1=1"
        cur = self._conn.execute(
            f"DELETE FROM {self._q(table)} WHERE {where_clause}",
            tuple(where_params),
        )
        return cur.rowcount

    def commit(self) -> None:
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # 持久化/恢复（pre_flight snapshot/rollback·对称 DictBackend.snapshot/load_snapshot）

    def snapshot(self) -> dict[str, list[dict[str, Any]]]:
        """全表快照（表名→行列表·同 DictBackend.snapshot 格式·pre_flight rollback 用）。

        遍历已注册 _tables ·每表 SELECT * → dict 列表（行序 = rowid 序 = 插入序·确定 bit-identical）。
        纯读·零 discipline 闸门（绕 check_write·同 DictBackend 直接读 _data）。含空表 []（register 但
        无 insert·对称 DictBackend.snapshot 亦含空表·trial 期空表 insert 的 rollback 也清零）。

        铁律：纯整数（行值全 int / TEXT 列 str）/ 确定性（rowid 序确定 = 插入序）。
        """
        return {t: self._do_select(t, None, None, None, False, None)
                for t in self._tables}

    def load_snapshot(self, snap: dict[str, list[dict[str, Any]]]) -> None:
        """从快照恢复（清空 + 重插·pre_flight rollback 用·同 DictBackend.load_snapshot 语义）。

        `with self._conn` 事务包裹（sqlite3 Connection 上下文·自动 commit 成功 / rollback 异常·partial
        失败不留脏）：清空 snap 中所有表（_do_delete(table, {})→DELETE FROM·where 1=1·绕 discipline 闸门·
        同 DictBackend 直接替换 _data）+ 重新插入 snapshot 行（_do_insert·行序 = snap 序·rowid 递增确定）。

        _tables/schema 不动（trial 期 schema 固定·pre_flight rollback 5 状态不含 _tables·formal_train:1180+）。

        铁律：纯整数 / 确定性（snap 行序→重插 rowid 序）/ fail-loud（异常 ROLLBACK 抛·不留脏）。
        """
        with self._conn:   # 自动 commit（成功）/ rollback（异常）·事务原子
            for table, rows in snap.items():
                self._do_delete(table, {})   # 清空（绕 discipline·trial 增量归零）
                for row in rows:
                    self._do_insert(table, row)   # 重插（行序 = snap 序·rowid 递增确定）
