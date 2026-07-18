"""algorithm — 自建算法原语层（§十五决策9·系统内自建无一外包·依赖 storage+crosscut+numeric）。

决策9 裁定 A2/A3/A4 全部系统内自建无一外包（"外包作废"≠不实现算法）。
  a2_topology      A2 拓扑分层（Kahn·按头分发·PRECEDES+Kahn 随 §7.1 地基）
  a3_personal_rank A3 PR 求解（(I-αA)x=(1-α)e·多种子 wrapper·B1 主精确/B2 迭代兜底/B3 LU defer）
  a4_alignment     A4 结构对齐（LCS·pairwise 折叠）
  closure          transitive_closure（按 edge_type 分发·闭包纯净性·CLOSURE 派生不存储）

纯整数·不依赖 numpy/scipy/networkx（§五库依赖）。只参考原理非搬代码（旧代码非重写权威）。
"""
from __future__ import annotations
