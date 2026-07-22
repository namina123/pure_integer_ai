"""cognition.understanding.emergent_role — 主导度闸（缺口#1·去 SVO 写死唯一合法替代）。

doc/重来·主线重审与重画.md:580 防塌C6 落盘：role 来自 emergent_role（位置桶+结构特征）的具体机制。

  ① position_hist 专用表（非 edge 避 N×膨胀·schema: (concept_ref, position, count)）
  ② dominant_pos = argmax(count·tiebreak 最小位置码)
  ③ 主导度闸 dominance = mx*1000//total ≥ MIN_DOMINANCE(500) 才采信 dominant_pos
  ④ 主导不足 → 混合桶（ROLE_BUCKET_BASE+offset·不污染主导桶）
  ⑤ 冷启动 ROLE_SUBJECT 兜底（position_hist 空）
  ⑥ 防碎片垮塌（无闸 245 碎片 vs SVO 4 桶·主导度闸是去 SVO 写死合法替代）

**分层定位**（取证核实·非阻塞闭合但阻塞缺口#1 目标）：
  - SUBJECT 兜底（⑤）= 阻塞下限（解致命6·role_seq 空致 generate 产空→reward 腿空转）。
  - 主导度闸+混合桶（③④）= 缺口#1 目标（去 SVO 写死合法替代·无闸则位置桶碎片垮塌 > SVO）。
  两者皆建（以优为主·不妥协）。

**与消费侧**：emergent_role 计算只产 role 整数值（role_scheme）·caller（observe 预处理 / collection）
  把 role_seq 填进 Segment.role_seq → attach_role_seq 挂结构概念点 → generate 读 role_seq 逐槽分派。
  observe_position 在 observe 段处理时调（每 token 累加位置直方图）。

铁律：纯整数（position/count/dominance×1000）/ 确定性（argmax+tiebreak 最小位置·纯函数·
  bit-identical）/ 不写死（位置桶涌现非 SVO 模板）/ append-only（position_hist count 经
  DISC_MUTABLE_MONOTONE update +=·表本身 append-only）/ fail-loud。
依赖方向：cognition.understanding → storage（register_extension_table）+ crosscut·单向向下。
"""
from __future__ import annotations

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage.backend import StorageBackend, register_extension_table
from pure_integer_ai.storage import discipline as disc
from pure_integer_ai.cognition.understanding.role_scheme import (
    ROLE_SUBJECT, MIN_DOMINANCE, MAX_POSITION,
    position_bucket, mixed_bucket, is_role_value,
)

# position_hist 表名（专用表·非 edge·doc ①避 N×膨胀）
POSITION_HIST_TABLE = "position_hist"

# position_hist 列：(space_id, local_id, position, count)·PK 三列·doc ①
_POSITION_HIST_COLUMNS = [
    ("space_id", "INT"),
    ("local_id", "INT"),
    ("position", "INT"),     # role 位置码（token 在句中序·0-based）
    ("count", "INT"),        # 在该位置出现次数
]
_POSITION_HIST_INDEXES = [
    ("space_id", "local_id"),   # 避 O(N²) 全表扫（legacy 教训·40 段 31s→82.8s）
]


def register_position_hist(backend: StorageBackend) -> None:
    """注册 position_hist 表（启动调一次·cognition 扩展表·core=False）。

    挂 DISC_MUTABLE_MONOTONE（count 经 update += 单调增·append-only 语义）。
    """
    register_extension_table(
        backend, POSITION_HIST_TABLE, _POSITION_HIST_COLUMNS,
        discipline=disc.DISC_MUTABLE_MONOTONE,
        indexes=_POSITION_HIST_INDEXES,
    )


def observe_position(backend: StorageBackend,
                     ref: tuple[int, int], position: int) -> None:
    """观察概念在某句位置出现·累加位置直方图（doc ①）。

    ref      (space_id, local_id) 概念引用。
    position token 在句中序（0-based·须 ∈ [0, MAX_POSITION]）。
    幂等累加：已有 (ref, position) 行 → count += 1；否则 insert count=1。
    """
    sid, lid = ref
    assert_int(sid, lid, position, _where="observe_position.args")
    if not (0 <= position <= MAX_POSITION):
        raise ValueError(f"position 越界 [0,{MAX_POSITION}]: {position}")
    rows = backend.select(POSITION_HIST_TABLE,
                          where={"space_id": sid, "local_id": lid, "position": position})
    if rows:
        backend.update(POSITION_HIST_TABLE,
                       where={"space_id": sid, "local_id": lid, "position": position},
                       set_={"count": ("+=", 1)})
    else:
        backend.insert(POSITION_HIST_TABLE, {
            "space_id": sid, "local_id": lid,
            "position": position, "count": 1,
        })


def position_histogram(backend: StorageBackend,
                       ref: tuple[int, int]) -> list[tuple[int, int]]:
    """读概念的 (position, count) 直方图·按 position 升序（确定性）。空返 []。"""
    sid, lid = ref
    rows = backend.select(POSITION_HIST_TABLE,
                          where={"space_id": sid, "local_id": lid},
                          order_by="position")
    return [(int(r["position"]), int(r["count"])) for r in rows]


class PositionHistogramState:
    """持有单条 observe 流水线已核验的位置直方图派生状态。

    状态只缓存从同一 backend 读取的完整桶，并且所有后续累加都必须经本实例写入，
    因而可以在不改变“整段先读角色、再写当前位置”的前提下增量维护。调用方若绕过
    本实例直接改动 ``position_hist``，必须先显式调用 :meth:`invalidate` 再次读取。
    """

    def __init__(self, backend: StorageBackend) -> None:
        """绑定权威 backend；缓存不跨 backend、run 或流水线实例共享。"""
        self._backend = backend
        self._histograms: dict[tuple[int, int], dict[int, int]] = {}

    def invalidate(self, ref: tuple[int, int] | None = None) -> None:
        """失效一个概念桶；未指定 ref 时清空本流水线的全部派生状态。"""
        if ref is None:
            self._histograms.clear()
            return
        sid, lid = ref
        assert_int(sid, lid, _where="PositionHistogramState.invalidate")
        self._histograms.pop((sid, lid), None)

    def _load(self, ref: tuple[int, int]) -> dict[int, int]:
        """首次读取完整桶并拒绝重复 position 行，随后返回实例私有可变副本。"""
        sid, lid = ref
        assert_int(sid, lid, _where="PositionHistogramState._load")
        key = (sid, lid)
        cached = self._histograms.get(key)
        if cached is not None:
            return cached
        loaded: dict[int, int] = {}
        for position, count in position_histogram(self._backend, key):
            if position in loaded:
                raise RuntimeError(
                    f"position_hist 同一概念存在重复位置行: ref={key!r}, position={position}")
            loaded[position] = count
        self._histograms[key] = loaded
        return loaded

    def histogram(self, ref: tuple[int, int]) -> list[tuple[int, int]]:
        """返回按位置升序的当前桶快照，禁止调用方修改实例内部缓存。"""
        hist = self._load(ref)
        return sorted(hist.items())

    def role(self, ref: tuple[int, int]) -> int:
        """依据当前已核验桶计算一个概念的角色，不触发额外写入。"""
        return role_from_histogram(self.histogram(ref))

    def roles_for_tokens(self, refs: list[tuple[int, int]]) -> list[int]:
        """按输入顺序计算整段角色；同段重复概念共享写入前的同一桶状态。"""
        return [self.role(ref) for ref in refs]

    def observe(self, ref: tuple[int, int], position: int) -> None:
        """累加一次位置观察，并在权威写入成功后同步更新实例派生状态。"""
        sid, lid = ref
        assert_int(sid, lid, position, _where="PositionHistogramState.observe")
        if not (0 <= position <= MAX_POSITION):
            raise ValueError(f"position 越界 [0,{MAX_POSITION}]: {position}")
        hist = self._load((sid, lid))
        where = {"space_id": sid, "local_id": lid, "position": position}
        if position in hist:
            affected = self._backend.update(
                POSITION_HIST_TABLE, where=where, set_={"count": ("+=", 1)})
            if affected != 1:
                self.invalidate((sid, lid))
                raise RuntimeError(
                    f"position_hist 已核验行更新数应为 1，实际为 {affected}: ref={(sid, lid)!r}, "
                    f"position={position}")
            hist[position] += 1
            return
        self._backend.insert(POSITION_HIST_TABLE, {
            "space_id": sid,
            "local_id": lid,
            "position": position,
            "count": 1,
        })
        hist[position] = 1


def dominant_position(hist: list[tuple[int, int]]) -> tuple[int, int] | None:
    """argmax(count·tiebreak 最小位置码) → (dominant_pos, count)·空返 None（doc ②）。

    确定性：count 平局取最小 position（list 按 position 升序·首个 max 即最小位置）。
    """
    if not hist:
        return None
    # hist 已按 position 升序（position_histogram 保证）·首个 max 即最小位置 tiebreak
    pos, cnt = max(hist, key=lambda pc: pc[1])
    return pos, cnt


def dominance(count: int, total: int) -> int:
    """主导度 = mx*1000//total（doc ③·纯整数百分比×1000）。total=0 → 0。"""
    assert_int(count, total, _where="dominance.args")
    if total <= 0:
        return 0
    return count * 1000 // total


def role_from_histogram(hist: list[tuple[int, int]]) -> int:
    """把一个确定性位置直方图投影为 role 整数值。"""
    dom = dominant_position(hist)
    if dom is None:
        return ROLE_SUBJECT
    dom_pos, mx = dom
    total = sum(count for _, count in hist)
    if dominance(mx, total) >= MIN_DOMINANCE:
        return position_bucket(dom_pos)
    return mixed_bucket()


def emergent_role(backend: StorageBackend, ref: tuple[int, int]) -> int:
    """计算概念的 emergent_role（doc ⑤③④·位置桶+主导度闸+混合桶+冷启动兜底）。

    返 role 整数值（role_scheme·is_role_value 守）：
      position_hist 空 → ROLE_SUBJECT（冷启动兜底·doc ⑤）
      dominance ≥ MIN_DOMINANCE → position_bucket(dominant_pos)（位置桶·doc ③）
      dominance < MIN_DOMINANCE → mixed_bucket()（混合桶·doc ④·不污染主导桶）
    """
    return role_from_histogram(position_histogram(backend, ref))


def role_seq_for_tokens(backend: StorageBackend,
                        refs: list[tuple[int, int]]) -> list[int]:
    """逐 token 算 emergent_role → role_seq（对齐 refs 序·缺口#1 role 生产入口）。

    caller（observe 预处理 / collection）调此填 Segment.role_seq。
    每 token role = emergent_role(该概念)·冷启动全 SUBJECT（position_hist 空·退化态·解致命6）。
    """
    return [emergent_role(backend, r) for r in refs]


__all__ = [
    "POSITION_HIST_TABLE",
    "register_position_hist",
    "observe_position",
    "position_histogram",
    "PositionHistogramState",
    "dominant_position",
    "dominance",
    "role_from_histogram",
    "emergent_role",
    "role_seq_for_tokens",
    # re-export role_scheme 守门
    "is_role_value",
]
