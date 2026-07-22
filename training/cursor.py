"""training.cursor — dump 续训（per-space dump·新 run_id·几百G不重训红线·§十二 E1/E4/E8）。

dump_run(backend, run_dir, run_id, *, spaces) —— 终 dump（per-space·复用 paths.py·权威 base·E1）
load_run(backend, run_dir, run_id) —— 续训 load（新 run_id 从终 dump 起·E8）
CursorState / cursor_resume —— stage-skip 续训（E8·跳已完成 skippable 阶段）

**E1 两类 checkpoint 不 conflate**：
  跨 run 续训 base = 上一 run 终 dump（run 正常结束产完整快照·权威·确定性·bit-identical 可复现）
  within-run checkpoint = 崩溃恢复专用（本 run 内从崩溃点续跑·非权威·绝不作跨 run 续训 base）
  → cursor 只产/读终 dump·within-run checkpoint 是另一机制（append_log 切片·E2·非本模块职责）。

**E4 replay 覆盖率前置校验**：续训 load 终 dump 前校验 teacher replay 覆盖率 ≥ 阈值·未达标禁续训
  （防 miss→None 静默降级破 bit 可复现·§十一 E4）。

**E8 cursor resume stage-skip**：续训用 cursor resume 跳过已完成 skippable 阶段 + load 既有终 dump 图
  只增量喂新语料·非 skippable 阶段（reward 闭环须重标定权重 H2）显式标注不跳。

**几百G不重训红线**：每正式 run 新 run_id·续训从终 dump 起·已完成阶段不重跑（不部分白训）。

per-space dump 形态（复用 paths.py C5）：
  <run_dir>/<run_id>/space_<sid>.dump = JSON 行（该 space 的所有表行·filter_rows_for_space 过滤）
  跨 space 边在两端 space dump 各留一份（append-only 可回溯·非跨 space 移动·§十五决策1）。
  <run_dir>/<run_id>/global_identity.dump = 不属于单一 space 的身份索引完整快照。

铁律：纯整数（dump 行纯整·伴随 TEXT 合法·JSON 序列化无浮点）/ 确定性（per-space 文件+行序确定·bit-identical）/
  append-only（load 不改 dump 文件·续训写新 run_id）/ 几百G不重训（新 run_id·终 dump base）。
诚实边界：dump 是存储快照非语义（stable≠correct）/ 续训跨 run 须重标定 reward 权重 H2（非 skippable）/
  within-run 崩溃恢复是另一机制（E2 append_log·本模块只管终 dump 跨 run）。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage.backend import StorageBackend
from pure_integer_ai.storage import paths
from pure_integer_ai.storage.assertion_identity import (
    ASSERTION_SUPERSEDE_TABLE,
    IDENTITY_HEADER_TABLE,
    IDENTITY_PART_TABLE,
)
from pure_integer_ai.storage.assertion_record import (
    ASSERTION_QUALIFIER_TABLE,
    ASSERTION_RECORD_TABLE,
)
from pure_integer_ai.storage.graph_object import GRAPH_OBJECT_TABLE
from pure_integer_ai.storage.graph_object_identity import (
    GRAPH_HYPOTHESIS_GROUP_COMPONENT_TABLE,
    GRAPH_HYPOTHESIS_GROUP_TABLE,
    GRAPH_OBJECT_COMPONENT_TABLE,
)
from pure_integer_ai.storage.graph_statement import GRAPH_STATEMENT_TABLE
from pure_integer_ai.storage.memory_overlay import MEMORY_OVERLAY_TABLE
from pure_integer_ai.storage.memory_event import (
    MEMORY_EVENT_PART_TABLE,
    MEMORY_EVENT_TABLE,
)
from pure_integer_ai.storage.occurrence import (
    OCCURRENCE_CANDIDATE_TABLE,
    OCCURRENCE_TABLE,
)
from pure_integer_ai.storage.source_record import SOURCE_RECORD_TABLE
from pure_integer_ai.storage.span import SPAN_MEMBER_TABLE, SPAN_TABLE
from pure_integer_ai.storage.spaces.companion import TEXT_ASSOC_TABLE

# 续训 replay 覆盖率阈值（E4·未达标禁续训·防 miss→None 静默降级破可复现）
# B7 放宽（2026-07-03）：首版 1/1（100%）致 --resume 实际不可用（真实语料任一 teacher miss 即 raise）。
# 改 9/10（90%）= 允许小量 miss（E4 优雅降级·caller skip None annotation）·仍拦 <90% 系统性 miss
# （错教师/空录制/全错位）。check_replay_coverage 已参数化（min_num/min_den）·oracle/生产可按语料调
# （趋严回 1/1 或更松）。recorded*den >= total*num → 9/10 命中即放行。
REPLAY_COVERAGE_MIN_NUM = 9   # recorded/total ≥ 9/10（90%·B7 放宽·首版 1/1 致 --resume 不可用）
REPLAY_COVERAGE_MIN_DEN = 10

# 终 dump 涉及的核心表（per-space filter·cognition 经 backend 抽象访问的表）
DUMP_TABLES: tuple[str, ...] = (
    "concept_node", "edge", "def_array", "memory_item",
    GRAPH_OBJECT_TABLE, GRAPH_OBJECT_COMPONENT_TABLE,
    GRAPH_HYPOTHESIS_GROUP_TABLE,
    GRAPH_HYPOTHESIS_GROUP_COMPONENT_TABLE,
    GRAPH_STATEMENT_TABLE,
    MEMORY_OVERLAY_TABLE,
    MEMORY_EVENT_TABLE,
    MEMORY_EVENT_PART_TABLE,
    OCCURRENCE_TABLE, OCCURRENCE_CANDIDATE_TABLE, SOURCE_RECORD_TABLE,
    SPAN_TABLE, SPAN_MEMBER_TABLE,
    TEXT_ASSOC_TABLE,
    IDENTITY_HEADER_TABLE, IDENTITY_PART_TABLE, ASSERTION_SUPERSEDE_TABLE,
    ASSERTION_RECORD_TABLE, ASSERTION_QUALIFIER_TABLE,
)

GLOBAL_DUMP_TABLES: frozenset[str] = frozenset({
    IDENTITY_HEADER_TABLE,
    IDENTITY_PART_TABLE,
    ASSERTION_SUPERSEDE_TABLE,
    ASSERTION_RECORD_TABLE,
    ASSERTION_QUALIFIER_TABLE,
    SOURCE_RECORD_TABLE,
    GRAPH_OBJECT_COMPONENT_TABLE,
    GRAPH_HYPOTHESIS_GROUP_TABLE,
    GRAPH_HYPOTHESIS_GROUP_COMPONENT_TABLE,
})


@dataclass
class CursorState:
    """续训游标（E8 stage-skip·新 run_id 从终 dump 起）。

    base_run_id   续训 base 的终 dump run_id（上一 run 正常结束·权威 base·E1）
    run_id        本 run 新 run_id（几百G不重训红线·新 run_id）
    completed     已完成阶段集（skippable·续训跳过·不重跑不部分白训）
    non_skippable 须重标定阶段集（reward 闭环须重标定权重 H2·不跳）
    """
    base_run_id: str
    run_id: str
    completed: set[int] = field(default_factory=set)
    non_skippable: set[int] = field(default_factory=set)


# ---- 终 dump（per-space·E1 权威 base） ----

def _table_rows(backend: StorageBackend, table: str) -> list[dict[str, Any]]:
    """取表全部行（确定性·无 where·backend.select 全扫）。"""
    return backend.select(table, where=None)


def _serialize_value(v: Any) -> Any:
    """JSON 序列化预处理：None/int/str 原样·其他转 str（防 tuple 等非 JSON 类型）。"""
    if v is None or isinstance(v, (int, str, bool)):
        return v
    return str(v)


def dump_run(backend: StorageBackend, run_dir: str, run_id: str,
             *, spaces: list[int],
             tables: tuple[str, ...] = DUMP_TABLES) -> list[int]:
    """终 dump（per-space·复用 paths.py·E1 权威 base）。

    每 space 一个文件 space_<sid>.dump（JSON 行·该 space 的所有表行过滤后）。
    返已 dump 的 space_id 列表（升序·确定性）。run 正常结束才调此（产终 dump·跨 run 续训 base）。
    """
    paths.ensure_run_dir(run_dir, run_id)
    dumped: list[int] = []
    for sid in sorted(spaces):
        assert_int(sid, _where="dump_run.space_id")
        path = paths.space_dump_path(run_dir, run_id, sid)
        with open(path, "w", encoding="utf-8") as f:
            for table in tables:
                if table in GLOBAL_DUMP_TABLES:
                    continue
                rows = paths.filter_rows_for_space(_table_rows(backend, table), sid)
                for r in rows:
                    record = {"_table": table,
                              **{k: _serialize_value(v) for k, v in r.items()}}
                    f.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                    f.write("\n")
        dumped.append(sid)
    global_tables = tuple(table for table in tables if table in GLOBAL_DUMP_TABLES)
    if global_tables:
        global_path = paths.global_identity_dump_path(run_dir, run_id)
        with open(global_path, "w", encoding="utf-8") as f:
            for table in global_tables:
                for row in _table_rows(backend, table):
                    record = {"_table": table,
                              **{k: _serialize_value(v) for k, v in row.items()}}
                    f.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                    f.write("\n")
    return dumped


def load_run(backend: StorageBackend, run_dir: str, run_id: str) -> list[int]:
    """续训 load（新 run_id 从终 dump 起·E8）。

    读 space_*.dump 文件·按 _table 还原行 insert 到 backend。返已 load 的 space_id 列表（升序）。
    幂等：重复 load 同一 space 会重插（caller 须用空 backend·续训新 run_id 从空起）。

    **id_pool rebaseline（序列7·修 latent 续训 id-collision bug）**：next_id 从内存 _id_pool 自增·
    非存储表·load 后须推高水位 ≥ 已载 max local_id·否则续训新分配从 1 起撞已载节点（DictBackend 静默
    dup / SQLite 无 PK 亦静默·latent 续训 corrupt）。**自分配 local_id 的表**：concept_node
    （ConceptIndex.ensure 经 next_id）+ memory_item（MemorySpace.new_local_id 经 next_id）——皆 space_id+
    local_id 列。故**载入逐行跟踪**凡含 space_id+local_id 的行的 per-space max（含 composes_attr/def_array
    等引用表·引用 local_id ≤ 分配 max·harmless 不影响正确水位）→ advance_id_pool 推高。逐行跟踪 robust
    （未来新自分配表自动覆盖·无须硬编码表名·对抗审计 Finding2 修：原 concept_node-only 漏 memory_item）。
    max 序无关 bit-identical。
    """
    sids = paths.list_space_dumps(run_dir, run_id)
    floor_by_space: dict[int, int] = {}   # id_pool rebaseline 跟踪（per-space max local_id·序列7 Gap2）

    def load_file(path: str) -> None:
        """载入单个确定性 dump 文件，并同步 id_pool 恢复水位。"""
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                table = record.pop("_table", None)
                if table is None:
                    continue
                row = {k: v for k, v in record.items()}
                backend.insert(table, row)
                lid_r = row.get("local_id")
                if lid_r is not None and "space_id" in row:
                    sid_r = row["space_id"]
                    if lid_r > floor_by_space.get(sid_r, 0):
                        floor_by_space[sid_r] = lid_r

    for sid in sids:
        path = paths.space_dump_path(run_dir, run_id, sid)
        if not os.path.isfile(path):
            continue
        load_file(path)
    global_path = paths.global_identity_dump_path(run_dir, run_id)
    if os.path.isfile(global_path):
        load_file(global_path)
    for sid_r, floor in sorted(floor_by_space.items()):
        backend.advance_id_pool(sid_r, floor)
    return sids


# ---- E4 replay 覆盖率前置校验 ----

def check_replay_coverage(teacher: Any, needed: list[tuple[int, tuple]],
                          *, min_num: int = REPLAY_COVERAGE_MIN_NUM,
                          min_den: int = REPLAY_COVERAGE_MIN_DEN) -> bool:
    """续训前置：teacher replay 覆盖率 ≥ 阈值才允许续训（E4·防 miss→None 静默降级）。

    needed = [(kind, args), ...] 本续训 run 预期 replay 的调用集。
    """
    recorded, total = teacher.replay_coverage(needed)
    if total == 0:
        return True   # 无需 replay 调用·放行
    # recorded/total ≥ min_num/min_den
    return recorded * min_den >= total * min_num


# ---- E8 cursor resume stage-skip ----

def cursor_resume(state: CursorState, stages: list[int],
                  *, skippable: "set[int] | frozenset[int]") -> list[int]:
    """续训 stage-skip（E8·跳已完成 skippable 阶段·非 skippable 不跳）。

    返本次续训须跑的阶段列表（已完成 skippable 跳过·非 skippable 保留须重跑）。
    skippable = 纯 observe 累积阶段（产物纯图累积·已存终 dump）·非 skippable = 依赖分布标定（须重标 H2）。
    """
    todo: list[int] = []
    for st in stages:
        if st in state.completed and st in skippable:
            continue   # 已完成 skippable·跳过（不重跑·不部分白训）
        todo.append(st)
    return todo


def mark_completed(state: CursorState, stage: int, *, skippable: bool) -> None:
    """标记阶段完成（skippable 进 completed·非 skippable 进 non_skippable·E8）。"""
    if skippable:
        state.completed.add(stage)
    else:
        state.non_skippable.add(stage)


def is_stage_skippable(stage: int, skippable: "set[int] | frozenset[int]") -> bool:
    """阶段是否 skippable（E8·纯 observe 累积 vs 分布标定）。"""
    return stage in skippable
