"""storage.discipline — 写纪律枚举 + 违例 + 核心表集合（§十五决策6/8 执行点）。

三档纪律（下推 backend 执行）：
  DISC_NONE             自由（insert/update/delete·非核心扩展表）
  DISC_APPEND_ONLY      只增（insert only·拒 update/delete）
  DISC_MUTABLE_MONOTONE 可变单调（insert + update·拒 delete·单调性由 store 层守 delta>0）

核心表（CORE_TABLES）：append-only 原则——DELETE 一律拒（核心永不删·改走冷区脱离）；
UPDATE 只允许 MUTABLE_MONOTONE 核心表（edge / memory_item.status flip / node version）。
非核心扩展表（register_extension_table）按注册纪律守。

register_extension_table：非核心表注册（L1 迁移·14 越层表归此），挂纪律·guard 守。
"""
from __future__ import annotations


# ---- 纪律枚举 ----
DISC_NONE = 0
DISC_APPEND_ONLY = 1
DISC_MUTABLE_MONOTONE = 2

DISC_NAMES = {
    DISC_NONE: "NONE",
    DISC_APPEND_ONLY: "APPEND_ONLY",
    DISC_MUTABLE_MONOTONE: "MUTABLE_MONOTONE",
}


class DisciplineViolation(PermissionError):
    """写纪律违例（基类）。"""


class AppendOnlyViolation(DisciplineViolation):
    """append-only 违例（UPDATE/DELETE 只增核心表）。"""


class MonotoneViolation(DisciplineViolation):
    """单调违例（MUTABLE_MONOTONE 列只升不降·delta<0·由 store 层抛）。"""


# ---- 核心表集合（单一真相源·与各 store register 一致） ----
# 核心表 = 纯整数主存储 + 审计。append-only 原则：DELETE 拒·UPDATE 仅可变纪律表。
CORE_TABLES: frozenset[str] = frozenset({
    "space", "id_pool",
    "concept_node", "def_array", "assoc_table", "outward_index",
    "edge", "memory_item",
    "audit_event", "archived",
    "identity_header", "identity_part", "assertion_supersede",
    "assertion_record", "assertion_qualifier",
    "graph_object", "graph_object_component", "graph_statement",
    "graph_hypothesis_group", "graph_hypothesis_group_component",
    "memory_overlay_relation",
    "memory_event", "memory_event_part",
    "training_candidate_event", "training_candidate_event_part",
    "span", "span_member",
})

# 可变纪律核心表：允许 UPDATE（其单调/前移纪律由 store 层 monotone 守卫守）。
# edge: strength/sn/tn/belief 可调（MUTABLE_MONOTONE）；memory_item.status flip；
# concept_node.version_head 前移。其余核心表 APPEND_ONLY。
MUTABLE_CORE_TABLES: frozenset[str] = frozenset({
    "edge", "memory_item", "concept_node",
})


def is_core(table: str) -> bool:
    return table in CORE_TABLES


def default_discipline(table: str) -> int:
    """核心表默认纪律：可变纪律表→MUTABLE_MONOTONE，其余核心→APPEND_ONLY。
    非核心表默认 DISC_NONE（register_extension_table 可覆盖）。"""
    if table in MUTABLE_CORE_TABLES:
        return DISC_MUTABLE_MONOTONE
    if table in CORE_TABLES:
        return DISC_APPEND_ONLY
    return DISC_NONE


def check_write(table: str, op: str, discipline: int, is_core_table: bool) -> None:
    """写闸门（backend update/delete 前调·纯函数分类器）。

    op ∈ {"insert","update","delete"}。
    - insert：始终放行。
    - delete：核心表一律拒（append-only·核心永不删）；非核心按纪律（APPEND_ONLY 拒）。
    - update：APPEND_ONLY 拒；MUTABLE_MONOTONE / NONE 放行（单调值由 store 层守）。
    """
    if op == "insert":
        return
    if op == "delete":
        if is_core_table:
            raise AppendOnlyViolation(
                f"append-only 违例：DELETE 核心表 {table!r} 被拒（核心永不删·改走冷区脱离）"
            )
        if discipline == DISC_APPEND_ONLY:
            raise AppendOnlyViolation(
                f"append-only 违例：DELETE 只增表 {table!r} 被拒"
            )
        return
    # op == "update"
    if discipline == DISC_APPEND_ONLY:
        raise AppendOnlyViolation(
            f"append-only 违例：UPDATE 只增表 {table!r} 被拒（改走 INSERT/版本化）"
        )
    # MUTABLE_MONOTONE / NONE：放行（单调性由 store 层守）
    return
