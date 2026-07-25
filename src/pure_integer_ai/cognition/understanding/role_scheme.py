"""cognition.understanding.role_scheme — role 整数 scheme（缺口#1·emergent_role 主导度闸载体）。

§十一缺口#1（doc/重来·主线重审与重画.md:573-580）：role 不建独立边·落"结构概念点 role_seq 属性"；
role 来自 emergent_role（位置桶+结构特征·去 SVO 写死）。role 整数 scheme 是 emergent_role 的产出值域。

**role 整数 namespace（高段独立区间·避撞）**：
  role 值不与 edge_type（1-127）/ KIND（1-104·含 KIND_PLACEMENT=101）/ opcode symbol_id（OPCODE_BASE|·≥2^60）
  撞值。用高段独立基址 ROLE_BASE = 1 << 50（远超既有 namespace·审计读码无歧义）。
  legacy 用 ROLE_SUBJECT=101 / ROLE_BUCKET_BASE=300 与 KIND_PLACEMENT=101 撞值——主线不复刻·用高段。

  ROLE_SUBJECT      = ROLE_BASE | 0   # 冷启动兜底（position_hist 空·全 SUBJECT 退化态·doc ⑤）
  ROLE_BUCKET_BASE  = ROLE_BASE | 0x10000   # 位置桶基址·dominant_pos 作 offset（主导度≥闸）
  混合桶 offset     = ROLE_BUCKET_BASE + 0x10000   # 主导不足逃生通道·不污染主导桶（doc ④⑥）

  设计 doc line580 用符号 ROLE_SUBJECT/ROLE_BUCKET_BASE/MIN_DOMINANCE·MIN_DOMINANCE=500/1000(50%)已钉死·
  ROLE_SUBJECT/ROLE_BUCKET_BASE 整数值主线设计未钉死（legacy 101/300 撞 KIND_PLACEMENT）→ 本模块定值补设计。

**与 EDGE_ROLE_STAT(13) 的分工**（设计留白·本模块澄清）：
  position_hist 专用表 = 运行时 role 生产载体（纯统计 argmax·不接 reward·位置桶涌现·doc ①明示非 edge 避 N×膨胀）。
  EDGE_ROLE_STAT(13) 边 = D 类学习型信号边（概念→主/宾槽偏好 symbol·接 reward·断奶后涌现·defer·首版不激活）。
  二者并行不重叠：position_hist 先建（role 生产）；ROLE_STAT 边 defer（reward 强化偏好·日后接线）。

铁律：纯整数（position/count/dominance×1000·role 值高段）/ 不写死（位置桶涌现非 SVO 模板·
  MIN_DOMINANCE 阈值 oracle 标非硬编码语义）/ fail-loud（assert_int 守 role 值）。
"""
from __future__ import annotations

from pure_integer_ai.crosscut.guards.int_blocker import assert_int

# ---- role 整数 namespace（高段独立区间·避撞 edge_type/KIND/opcode） ----
# 1<<50 远超 edge_type(1-127) / KIND(1-104) / variable/param index(<2^60 低段)。
# legacy ROLE_SUBJECT=101 撞 KIND_PLACEMENT=101→主线不复刻·用高段独立基址。
ROLE_BASE = 1 << 50

ROLE_SUBJECT = ROLE_BASE | 0                       # 冷启动兜底（doc ⑤·position_hist 空时返此）
ROLE_BUCKET_BASE = ROLE_BASE | (1 << 16)           # 位置桶基址·dominant_pos 作 offset（主导度≥闸·doc ③④）
_BUCKET_OFFSET_WIDTH = 1 << 16                     # 位置桶 offset 区间宽度（dominant_pos < 此值·主导桶与混合桶不撞）
MIXED_BUCKET_OFFSET = ROLE_BUCKET_BASE + _BUCKET_OFFSET_WIDTH  # 混合桶 offset（主导不足逃生通道·doc ④⑥）

# 主导度闸阈值（doc ③钉死·500/1000=50%·oracle 标非写死语义）
MIN_DOMINANCE = 500

# position 直方图 position 上限（防病态·位置码 < _BUCKET_OFFSET_WIDTH·首版句长受限·远小于此）
MAX_POSITION = _BUCKET_OFFSET_WIDTH - 1


def position_bucket(dominant_pos: int) -> int:
    """主导位置 → 位置桶 role 值（ROLE_BUCKET_BASE + dominant_pos·doc ④）。

    dominant_pos 须 ∈ [0, MAX_POSITION]（位置码·主导桶 offset 不撞混合桶）。
    """
    assert_int(dominant_pos, _where="position_bucket.dominant_pos")
    if not (0 <= dominant_pos <= MAX_POSITION):
        raise ValueError(f"dominant_pos 越界 [0,{MAX_POSITION}]: {dominant_pos}")
    return ROLE_BUCKET_BASE + dominant_pos


def mixed_bucket() -> int:
    """主导不足 → 混合桶 role 值（doc ④⑥·逃生通道·不污染主导桶）。"""
    return MIXED_BUCKET_OFFSET


def is_role_value(v: int) -> bool:
    """是否合法 role 值（SUBJECT / 位置桶 / 混合桶）·守 attach_role_seq 入口。"""
    assert_int(v, _where="is_role_value")
    if v == ROLE_SUBJECT:
        return True
    if ROLE_BUCKET_BASE <= v < ROLE_BUCKET_BASE + _BUCKET_OFFSET_WIDTH:
        return True   # 位置桶（主导度≥闸）
    if v == MIXED_BUCKET_OFFSET:
        return True   # 混合桶（主导不足）
    return False
