"""storage.paths — per-space dump 路径（§十五 C5 + 决策1·三空间物理分开）。

**用户铁律**：记忆两层必须文件层级物理层级分开（否则无法多会话/复制固定给别人）→
两 space_id 独立 dump/复制/迁移。三空间（核心/记忆/伴随）物理分开 dump。

C5：per-space dump filter + per-space 文件路径。dump 按 space_id 分文件·
续训 load 按 space_id 取文件·跨会话复制 = 复制某 space 的 dump 文件。

路径形式（确定性·跨宿主一致）：<run_dir>/<run_id>/space_<space_id>.dump
- run_id 是新 run 标识（几百G不重训红线·新 run_id）
- 每 space 一个文件·物理分开·可独立复制/迁移
- 记忆阅读层/交互层各 space_id 各文件（两层物理分开）
"""
from __future__ import annotations

import os

from pure_integer_ai.crosscut.guards.int_blocker import assert_int


def space_dump_path(run_dir: str, run_id: str, space_id: int) -> str:
    """单 space 的 dump 文件路径（per-space·物理分开·C5）。"""
    assert_int(space_id, _where="space_dump_path.space_id")
    return os.path.join(run_dir, run_id, f"space_{space_id}.dump")


def run_dir_of(run_dir: str, run_id: str) -> str:
    """某 run 的目录（所有 space dump 文件在此·per-space 分文件）。"""
    return os.path.join(run_dir, run_id)


def list_space_dumps(run_dir: str, run_id: str) -> list[int]:
    """列出某 run 下已 dump 的 space_id 列表（升序·确定性）。"""
    d = run_dir_of(run_dir, run_id)
    if not os.path.isdir(d):
        return []
    out: list[int] = []
    for fn in os.listdir(d):
        if fn.startswith("space_") and fn.endswith(".dump"):
            try:
                out.append(int(fn[len("space_"):-len(".dump")]))
            except ValueError:
                continue
    return sorted(out)


def filter_rows_for_space(rows: list[dict], space_id: int,
                          *, space_col: str = "space_id") -> list[dict]:
    """C5 per-space dump filter：从全量行中筛某 space 的行。

    edge 跨 space 复合键（space_id_from/to）特殊：两端任一属此 space 则属此 space 的 dump
    （跨 space 边在两端 space 的 dump 各留一份·append-only 可回溯·非跨 space 移动）。
    """
    out: list[dict] = []
    for r in rows:
        if _row_belongs_to_space(r, space_id, space_col):
            out.append(r)
    return out


def _row_belongs_to_space(row: dict, space_id: int,
                          space_col: str) -> bool:
    """行是否属此 space。edge 跨 space 边：from 或 to 任一属此 space 即属。"""
    if space_col in row and row[space_col] == space_id:
        return True
    # edge 跨 space 复合键
    if "space_id_from" in row and row.get("space_id_from") == space_id:
        return True
    if "space_id_to" in row and row.get("space_id_to") == space_id:
        return True
    return False


def ensure_run_dir(run_dir: str, run_id: str) -> str:
    """确保 run 目录存在·返回路径。"""
    d = run_dir_of(run_dir, run_id)
    os.makedirs(d, exist_ok=True)
    return d
