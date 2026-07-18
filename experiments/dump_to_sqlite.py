"""experiments.dump_to_sqlite — 便携 SQLite 导出（portable artifact·可换）。

把 backend 内存态图 / per-space dump 文件 → 单个 SQLite 文件（可查询·可分享·可归档）。
**不是权威 dump**（权威 dump 是 cursor.dump_run 的 per-space JSON 行·bit-identical·E1）·
  本模块是便携 artifact：把图快照装进一个 SQLite 文件供外部检视/调试/分享。

  dump_to_sqlite(src_backend, sqlite_path, *, tables)   内存 backend → SQLite 文件
  export_run_dump_to_sqlite(run_dir, run_id, sqlite_path)  per-space dump 文件 → SQLite 文件

**确定性有序 insert**：行按 json sort_keys 序确定排序后批量 insert（bit-identical·跨宿主一致）·
  非 source backend 物理存储序（DictBackend/SQLiteBackend 序可能不同·sort 归一）。

**纯整数**：行经 _validate_row（核心表拒 float/str·伴随 TEXT 合法）·SQLite 文件无浮点。

铁律：纯整数（行经 backend 纪律闸门）/ 确定性（json sort_keys 序·bit-identical）/ append-only
  （导出只写新文件不改源）/ 便携（SQLite 单文件·跨宿主·可查询）。
诚实边界：dump_to_sqlite 是便携 artifact 非权威 dump（权威在 cursor.dump_run·E1）/
  导出是快照非语义（stable≠correct）/ 仅导出指定表（默认 DUMP_TABLES 图快照·非全 backend）。
"""
from __future__ import annotations

import json
import os
from typing import Any

from pure_integer_ai.storage.backend import StorageBackend, SQLiteBackend, DictBackend
from pure_integer_ai.storage import bootstrap
from pure_integer_ai.training.cursor import DUMP_TABLES, load_run


def _sorted_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """确定性排序（json sort_keys 序·bit-identical·跨宿主一致·非物理存储序）。"""
    return sorted(rows, key=lambda r: json.dumps(r, ensure_ascii=False,
                                                 sort_keys=True, default=str))


def dump_to_sqlite(src: StorageBackend, sqlite_path: str, *,
                   tables: tuple[str, ...] = DUMP_TABLES) -> str:
    """内存 backend → SQLite 文件（便携 artifact·默认导出 DUMP_TABLES 图快照）。

    新建 SQLiteBackend(sqlite_path) · bootstrap 注册核心表 · 按 json sort_keys 序批量 insert。
    返 sqlite_path。源 backend 不改（append-only 导出）。
    """
    os.makedirs(os.path.dirname(os.path.abspath(sqlite_path)), exist_ok=True)
    if os.path.exists(sqlite_path):
        os.remove(sqlite_path)   # 导出新建文件（非 append·便携 artifact 覆盖旧）
    dst = SQLiteBackend(sqlite_path)
    try:
        bootstrap(dst)
        for table in tables:
            rows = src.select(table, where=None)
            for r in _sorted_rows(rows):
                dst.insert(table, r)
        dst.commit()
    finally:
        dst.close()
    return sqlite_path


def export_run_dump_to_sqlite(run_dir: str, run_id: str,
                              sqlite_path: str) -> str:
    """per-space dump 文件 → SQLite 文件（便携 artifact·读 cursor.dump_run 产物）。

    load_run 读 space_*.dump JSON 行 → DictBackend → dump_to_sqlite → SQLite 文件。
    分享 dump 文件集 → 单 SQLite 文件供外部检视（便携）。
    """
    tmp = DictBackend()
    try:
        bootstrap(tmp)
        load_run(tmp, run_dir, run_id)
        return dump_to_sqlite(tmp, sqlite_path)
    finally:
        tmp.close()
