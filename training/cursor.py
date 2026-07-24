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

V-03 恢复包形态：
  <run_dir>/<run_id>/space_<sid>.dump = 带 segment 头的有序 JSON 行。
  <run_dir>/<run_id>/global_identity.dump = 无权威 space_id 列的全局表 segment。
  <run_dir>/<run_id>/cursor.json = 与图同次封存的阶段游标。
  <run_dir>/<run_id>/run.manifest.json + seal = schema、版本、依赖、epoch、
  read fence、行数和校验真值源。manifest 最后发布，已发布 run 不覆盖。

跨 space 边仍在两端 segment 保留物理副本，load 按原表 ordinal + 完整行核验
只去除跨 segment 副本，不合并表内合法重复行。重复 load 精确包为零写；
目标部分或漂移时 fail closed，中途异常恢复数据、id pool 和 IS_A 代次。

铁律：纯整数（dump 行纯整·伴随 TEXT 合法·JSON 序列化无浮点）/ 确定性（per-space 文件+行序确定·bit-identical）/
  append-only（load 不改 dump 文件·续训写新 run_id）/ 几百G不重训（新 run_id·终 dump base）。
诚实边界：dump 是完整存储快照非语义（stable≠correct）/ 续训跨 run 须重标定
  reward 权重 H2（非 skippable）/ 增量 segment、真分页、物理迁移和 within-run 崩溃点
  由 K-02/M-10 在同一 SegmentDependency 与故障边界上继续实现。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage.backend import StorageBackend
from pure_integer_ai.storage.recovery_package import (
    load_recovery_package,
    publish_recovery_package,
    registered_space_ids,
)
from pure_integer_ai.storage.recovery_protocol import (
    RecoveryDependency,
    RecoveryFaultInjector,
    RecoveryLoadResult,
    RecoveryMigrationRegistry,
)
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
from pure_integer_ai.storage.training_candidate_event import (
    TRAINING_CANDIDATE_EVENT_PART_TABLE,
    TRAINING_CANDIDATE_EVENT_TABLE,
)

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
    TRAINING_CANDIDATE_EVENT_TABLE,
    TRAINING_CANDIDATE_EVENT_PART_TABLE,
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


def _version_key(value: object | None) -> tuple[int, ...]:
    """将 VersionBundle 或显式整数 tuple 转为恢复依赖键。"""
    if value is None:
        return ()
    if isinstance(value, tuple):
        key = value
    else:
        stable_key = getattr(value, "stable_key", None)
        if not callable(stable_key):
            raise TypeError("versions 必须是整数 tuple 或实现 stable_key")
        key = stable_key()
    if not isinstance(key, tuple) or any(type(item) is not int for item in key):
        raise TypeError("versions stable_key 必须是严格整数 tuple")
    return key


def cursor_state_payload(state: CursorState) -> dict[str, Any]:
    """将训练游标转为与 run manifest 同次发布的确定 payload。"""
    if not isinstance(state, CursorState):
        raise TypeError("cursor state 类型错误")
    return {
        "base_run_id": state.base_run_id,
        "run_id": state.run_id,
        "completed": sorted(state.completed),
        "non_skippable": sorted(state.non_skippable),
    }


def cursor_state_from_payload(
        payload: dict[str, Any] | None,
        *, fallback_run_id: str,
        ) -> CursorState | None:
    """从已封存 payload 严格恢复训练游标。"""
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise TypeError("cursor payload 必须是 dict")
    try:
        base_run_id = payload.get("base_run_id", fallback_run_id)
        run_id = payload.get("run_id", fallback_run_id)
        completed = payload.get("completed", [])
        non_skippable = payload.get("non_skippable", [])
    except AttributeError as exc:
        raise TypeError("cursor payload 字段非法") from exc
    if (not isinstance(base_run_id, str) or not base_run_id
            or not isinstance(run_id, str) or not run_id):
        raise ValueError("cursor run_id 必须是非空字符串")
    if (not isinstance(completed, list)
            or not isinstance(non_skippable, list)
            or any(type(item) is not int for item in completed + non_skippable)):
        raise ValueError("cursor 阶段集必须是严格整数列表")
    return CursorState(
        base_run_id=base_run_id,
        run_id=run_id,
        completed=set(completed),
        non_skippable=set(non_skippable),
    )


def dump_run(backend: StorageBackend, run_dir: str, run_id: str,
             *, spaces: list[int] | None,
             tables: tuple[str, ...] | None = DUMP_TABLES,
             include_registered_tables: bool = False,
             require_all_spaces: bool = False,
             versions: object | None = None,
             dependencies: tuple[RecoveryDependency, ...] = (),
             publish_epoch: int = 1,
             cursor_state: CursorState | None = None,
             fault_injector: RecoveryFaultInjector | None = None,
             ) -> list[int]:
    """原子发布带 schema、segment、依赖和可选游标的终 dump。

    所有 segment 先写 staging，manifest 和独立封印最后写入，只有完整
    预检可读的包才会以单次目录替换发布。已发布 run 永不覆盖。
    """
    resolved_spaces = (list(registered_space_ids(backend))
                       if spaces is None else list(spaces))
    for sid in resolved_spaces:
        assert_int(sid, _where="dump_run.space_id")
    manifest = publish_recovery_package(
        backend,
        run_dir,
        run_id,
        spaces=tuple(resolved_spaces),
        tables=tables,
        include_registered_tables=include_registered_tables,
        require_all_spaces=require_all_spaces,
        version_key=_version_key(versions),
        dependencies=dependencies,
        publish_epoch=publish_epoch,
        cursor_payload=(None if cursor_state is None
                        else cursor_state_payload(cursor_state)),
        fault_injector=fault_injector,
    )
    return list(manifest.space_ids)


def load_run_package(
        backend: StorageBackend,
        run_dir: str,
        run_id: str,
        *,
        expected_versions: object | None = None,
        expected_dependencies: tuple[RecoveryDependency, ...] | None = None,
        expected_publish_epoch: int | None = None,
        migrations: RecoveryMigrationRegistry | None = None,
        fault_injector: RecoveryFaultInjector | None = None,
        ) -> RecoveryLoadResult:
    """预检后幂等加载恢复包，返回空间、游标和实际写表。

    加载前核验 manifest 封印、schema、依赖、epoch、segment 校验、
    read fence 和跨空间副本。目标表只允许为空或与包完全一致，
    因此精确重放是零写幂等，部分或漂移状态 fail closed。
    """
    return load_recovery_package(
        backend,
        run_dir,
        run_id,
        expected_version_key=(None if expected_versions is None
                              else _version_key(expected_versions)),
        expected_dependencies=expected_dependencies,
        expected_publish_epoch=expected_publish_epoch,
        migrations=migrations,
        fault_injector=fault_injector,
    )


def load_run(
        backend: StorageBackend,
        run_dir: str,
        run_id: str,
        **kwargs: Any,
        ) -> list[int]:
    """保留旧返回形状的恢复入口，实际执行 V-03 完整协议。"""
    return list(load_run_package(
        backend, run_dir, run_id, **kwargs).space_ids)


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
