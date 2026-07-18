"""cognition.understanding.modification_direction — G2 修饰方向A（ 的-cue head/modifier 统计·source+read-time）。

doc/重来_G2_修饰方向A_设计_2026-07-15.md。中文 "X 的 Y" — X=modifier（的 前）·Y=head（的 后）·head 更
salient（"红色的苹果"→苹果 head·红色 modifier）。G2 方向 A = source 统计每概念作 head vs modifier 次数·
read-time（dispatch_slot combine）给 head 加偏好 bonus（gate 守·bit-identical）。

  ① modification_hist 专用表（非 edge 避 N×膨胀·schema: (space_id, local_id, role, count)·镜像 position_hist）
  ② source write（observe token loop·2-token lookback· 的 at ti-1 → head=ti·modifier=ti-2·gate-independent）
  ③ read-time（dispatch_slot combine 第三亚轴 head_pref·gate MODIFIER_DIRECTION_MODE·cap HEAD_PREF_CAP=9）

**真消费者**：dispatch_slot generate 选词（slot_dispatch.py:166-170）·head 词获偏好·corpus_zh 富 的-修饰
（"最多样化 的 生物"·"不同 的 形态"）→ 真信号源·非 theater。

**为何 gate-independent write（bit-identical）**：表 modification_hist 唯一消费者是 gated head_pref_score
（dispatch_slot _mod_gate）·gate OFF 表 populated 但 inert（零 ungated 读）→ 无可观察行为变 → bit-identical
（同 position_hist observe_position 无条件累加证明）。累加 deterministic（同 corpus 同 head/modifier 计数）。

**诚实边界（#479 truth 墙）**：本机制只统计 的-位置（head=的 后·modifier=的 前）·非语义中心语判定
（"红色的苹果"系统不判"苹果是真正 head"·只统计 的-位置 head_count）。语义中心语 = truth 墙·stable≠correct。
30-40% 行为表面（ 的-位置统计）·长尾（语义中心语）撞墙。方向 B（硬排除 modifier）defer。

铁律：纯整数（role/count/bonus 全整）/ bit-identical（gate OFF combine 不变 + 表 inert + deterministic 累加）/
  不写死（ 的 检测复用 cue_words.is_property_attr_marker·非新硬编码词表）/ append-only（DISC_MUTABLE_MONOTONE
  count += 单调·表 append-only 语义）/ 单向依赖（L4 understanding → storage + crosscut·向下）。
依赖方向：cognition.understanding → storage（register_extension_table）+ crosscut·单向向下。
"""
from __future__ import annotations

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage.backend import StorageBackend, register_extension_table
from pure_integer_ai.storage import discipline as disc

# modification_hist 表名（专用表·非 edge·doc ①避 N×膨胀·镜像 position_hist）
MODIFICATION_HIST_TABLE = "modification_hist"

# 角色码： 的 前 X=modifier· 的 后 Y=head（非 0 区分未置位）
ROLE_MODIFIER = 1   # 的 前 X（修饰语）
ROLE_HEAD = 2       # 的 后 Y（中心语）

# head 偏好 bonus cap（read-time 亚轴 tiebreak·同 PR_SLOT_BONUS_CAP=3 量级·守 _cap_sp 999 联合 cap·不颠覆 collide 主轴）
HEAD_PREF_CAP = 9

# modification_hist 列：(space_id, local_id, role, count)·PK 四列·doc ①
_MODIFICATION_HIST_COLUMNS = [
    ("space_id", "INT"),
    ("local_id", "INT"),
    ("role", "INT"),          # ROLE_MODIFIER(1) / ROLE_HEAD(2)
    ("count", "INT"),         # 该角色出现次数（单调累加）
]
_MODIFICATION_HIST_INDEXES = [
    ("space_id", "local_id"),   # 避 O(N²) 全表扫（同 position_hist 教训）
]


def register_modification_hist(backend: StorageBackend) -> None:
    """注册 modification_hist 表（启动调一次·cognition 扩展表·core=False·镜像 register_position_hist）。

    挂 DISC_MUTABLE_MONOTONE（count 经 update += 单调增·append-only 语义）。
    """
    register_extension_table(
        backend, MODIFICATION_HIST_TABLE, _MODIFICATION_HIST_COLUMNS,
        discipline=disc.DISC_MUTABLE_MONOTONE,
        indexes=_MODIFICATION_HIST_INDEXES,
    )


def _bump(backend: StorageBackend, ref: tuple[int, int], role: int) -> None:
    """ref 作 role（HEAD/MODIFIER）出现一次·幂等累加 count（query-then-update/insert·同 observe_position）。"""
    sid, lid = ref
    assert_int(sid, lid, role, _where="modification_direction._bump.args")
    rows = backend.select(MODIFICATION_HIST_TABLE,
                          where={"space_id": sid, "local_id": lid, "role": role})
    if rows:
        backend.update(MODIFICATION_HIST_TABLE,
                       where={"space_id": sid, "local_id": lid, "role": role},
                       set_={"count": ("+=", 1)})
    else:
        backend.insert(MODIFICATION_HIST_TABLE, {
            "space_id": sid, "local_id": lid, "role": role, "count": 1,
        })


def observe_modification(backend: StorageBackend,
                         head_ref: tuple[int, int],
                         modifier_ref: tuple[int, int]) -> None:
    """观察一次 "modifier 的 head" 结构·累加 head 的 ROLE_HEAD + modifier 的 ROLE_MODIFIER（doc ②source write）。

    head_ref     的 后 Y（中心语）概念 ref。
    modifier_ref 的 前 X（修饰语）概念 ref。
    幂等：同 (ref, role) 行 → count += 1；否则 insert count=1。head==modifier（罕见自指）仍各累加一次（无害）。
    """
    _bump(backend, head_ref, ROLE_HEAD)
    _bump(backend, modifier_ref, ROLE_MODIFIER)


def head_preference(backend: StorageBackend,
                    ref: tuple[int, int]) -> tuple[int, int]:
    """读概念作 head vs modifier 的次数 → (head_count, modifier_count)（确定性·空返 (0,0)）。

    caller（head_pref_score）用 head_count - modifier_count 算偏好 bonus。
    """
    sid, lid = ref
    assert_int(sid, lid, _where="head_preference.args")
    rows = backend.select(MODIFICATION_HIST_TABLE,
                          where={"space_id": sid, "local_id": lid})
    head_count = 0
    mod_count = 0
    for r in rows:
        role = int(r["role"])
        if role == ROLE_HEAD:
            head_count = int(r["count"])
        elif role == ROLE_MODIFIER:
            mod_count = int(r["count"])
    return head_count, mod_count
