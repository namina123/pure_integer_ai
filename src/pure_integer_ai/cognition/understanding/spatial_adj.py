"""cognition.understanding.spatial_adj — 模块2-ter SPATIAL_ADJ 空间邻接建边（I1·§7.4.2）。

空间静态模态（2D/3D/动画帧内）**无步进阶段**（闭环 = observe 建拓扑 → judge 拓扑匹配·
非 observe→步进→judge）。空间邻接是无向图拓扑·非 DAG 序（§3.3）。

  - SPATIAL_ADJ strength 恒 = 1·不接 reward（结构真值·同 PRECEDES·§7.4.2）
  - 无向图：from/to 对称建边 (a,b)+(b,a)·邻接有环不进 DAG 传递闭包（§3.3）
  - 不进 head_types{PRECEDES,CAUSES}·PR 邻接不读（空间模态无步进不消费 PR）
  - 邻接判据：2D=ℚ²±1 邻接 / 3D=体素 6/18/26 邻接 / 动画帧内=同帧空间邻接

**首版语言 only·空间模态骨架 defer**（I1 消解"声称非defer交付defer"·伪代码层兑现骨架）。
本模块提供接口 + 2D ℚ²±1 邻接判据（纯整数·§7.4.1 避超越数 C7）·非空流实现随模态扩展阶段。
"""
from __future__ import annotations

from pure_integer_ai.storage.edge_store import EdgeStore
from pure_integer_ai.storage.node_store import TIER_PRIMARY
from pure_integer_ai.cognition.shared.edge_types import EDGE_SPATIAL_ADJ

SPATIAL_ADJ_STRENGTH = 1   # §7.4.2 结构真值·不接 reward


def build_spatial_adj(edge_store: EdgeStore,
                      primitives: list[tuple[tuple[int, int], tuple[int, int]]],
                      *, source: int, space_id: int) -> int:
    """SPATIAL_ADJ 无向邻接建边（对称·strength=1·不接 reward）。

    primitives : [(ref_a, ref_b), ...] 邻接对（调用方按模态邻接判据生成·2D ℚ²±1 / 3D 体素）。
    返回建边数（对称·每对 2 条）。
    首版：语言 only 时 primitives 为空·此函数 no-op（骨架 defer·I1）。
    概念点模态标由 caller observe 建 concept 后挂 abstract_mark MARK_MODALITY（§7.7.1 路径 B·非边参数）。
    """
    n = 0
    for a, b in primitives:
        if a == b:
            continue
        for (frm, to) in ((a, b), (b, a)):  # 无向对称
            edge_store.add(
                space_id_from=frm[0], local_id_from=frm[1],
                space_id_to=to[0], local_id_to=to[1],
                edge_type=EDGE_SPATIAL_ADJ, strength=SPATIAL_ADJ_STRENGTH,
                source=source, epistemic_origin=None,
                order_index=None, role=None, tier=TIER_PRIMARY,
            )
            n += 1
    return n


def adj_2d_rational(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """2D ℚ²±1 邻接判据（纯整数坐标·§7.4.1 避超越数 C7）。

    a, b 是 (x, y) 整数坐标（ℚ² 的整数表示）。邻接 = 曼哈顿距离 == 1（4-邻接）。
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
