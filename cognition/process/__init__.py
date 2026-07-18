"""cognition.process — 卷二过程建模（步进产路径 + reward 反传·§十三过程建模）。

模块（doc/伪代码分析/卷二_过程建模.md）：
  1. a2_stepper       A2 拓扑分层步进调度（Kahn + 按头 AND/OR 分发·HeadStepper）
  2. a3_pr_wrapper    A3 PR 多种子 wrapper（B1 精确/B2 兜底/cache·线性性零损失）
  3. a4_align         A4 结构对齐（pairwise LCS 折叠 + coverage_overlap）
  4. dag_path         DAG-path 步进主控（衔接 A2/A3/A4 + attractor + WorkMemory）
  5. attractor        attractor 动态演化（松入严留 + cap K）
  6. dead_end         死路检测（三条件：无后继/前驱全不active/步数上限）
  7. effective_weight effective_weight = strength × rate（H4·率进 PR 主权重）
  8. reward_propagate reward 反传通道（CAUSES 头 + 5 落点·episode 级 R1）
  9. episode          episode 主循环 orchestrator（M5 wiring 单点化）
  10. structure_discover 结构发现最小闭环（§八序列1·多样本 COMPOSES→等长对齐抽骨架→落
      struct_ref+COMPOSES[ATTR_ORIGIN=discovered]→复用 inline+β+vm_proof/coverage_overlap 消费·
      生成核心+多模态根基·doc/重来_结构发现设计补充.md）

依赖：storage（经抽象）/ algorithm（A2/A3/A4 自建原语）/ crosscut（纯整数/确定性）。
铁律：纯整数 audit_float / 无墙钟（order_index 隐含步序·timestamp_seq 记忆时序）/
  MUTABLE_MONOTONE（sn/strength 单调）/ append-only（台账/记忆）/ 确定性 bit-identical。
"""
