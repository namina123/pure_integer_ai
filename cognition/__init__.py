"""cognition — 认知层（上层·经 storage 抽象访问·三大建模）。

依赖 storage + crosscut + numeric + vm + algorithm。绝不写 raw SQL（只经 backend 抽象）。
  shared/        跨卷共享（types/edge_types/work_memory/concept_index）
  understanding/ 卷一理解建模（observe 建图·9 模块）
  process/       卷二过程建模（Stage 4·步进产 path·死路产负·reward 落 CAUSES）
  result/        卷三结果建模（Stage 5·reward=ΠG·ΣwJ·防塌三柱·假收敛）

卷一（本 Stage）= 输入层拿到数据 → observe 建图（概念点 + 边 + 落点分流）。
"""
from __future__ import annotations
