"""cognition.result — 卷三结果建模（judge + 防塌闭环验收·§十四）。

三大建模闭环的收口卷：过程建模(卷二)产 DAG-path 后，沿拓扑序生成输出 →
judge 算 reward=ΠG·ΣwJ → reward 反传(回卷二模块8) → 防塌三柱闭环验收 + 收敛判据。

模块清单：
  1 generate       路径填槽主框架 + 回放逐槽原语（沿 DAG 拓扑序多部分编排）
  2 slot_dispatch  逐槽分派（概念填槽 vs 记忆序列回放·target_lang 偏好）
  3 judge          四判据合成 reward=ΠG·ΣwJ（J1覆盖/J2意图/J3因果/J4闭合 + G3b/G5 写回门）
  4 anti_collapse  防塌三柱闭环验收（结构judge/真负通路/探索压力）
  5 convergence    收敛判据（含负通路活跃·假收敛识别）
  6 tri_space      三空间协同三时间尺度（核心快环/记忆中环/伴随慢环）

graph_view：卷三读图统一接口（read_role_seq/activate_candidates/read_memory_sequence/
  collide_score）——卷三经此抽象访问持久图，不直接碰 backend schema（守最少耦合）。

卷二↔卷三接口定稿（A1）：卷二步进产完整 DAG-path 后一次性交卷三 judge（非交替·
J4 闭合性检查非逐步·交替破 bit-identical）。卷二 j4_closure_check 占位 true。
"""
from __future__ import annotations
