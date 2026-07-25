"""numeric — 数值关联轴层（依赖 crosscut）。

模块：
  symbol_domain        TYPE_AXIS / OPCODE_* / VARIABLE / PARAM + opcode↔symbol 桥
  concept_numeric      概念在某轴的代分数定点值 (M, B^k) · axis_symbol_id 区分图即程序/数值关联
  credit_sink          数值来源 reward 信用汇聚（append-only·单调）
  numeric_relatedness  大常数−交叉积差值归一 · 跨轴=0 接地墙守门

铁律：纯整数 / 跨轴=0 接地墙（数值关联只在同轴内·跨轴无数值关系）/ append-only 信用。

【诚实标注】symbol_domain 的完整 opcode 集随 VM（Stage 2·§九 2a 图即程序）填实；
Stage 0 落轴框架 + 桥机制 + 最小 opcode 占位，不预先固化可能在 VM 设计期调整的 opcode 语义。
"""
