"""crosscut — 横切原语层（最底层·零业务依赖）。

三子层（内部单向：guards ← integer ← determinism）：
  guards/       禁浮点守卫（运行时 assert + AST CI lint + int blocker）·叶层零依赖
  integer/      纯整数数学（longdiv/isqrt/fixed_point/compare/rational）
  determinism/  确定性/审计（Hasher/DRNG/golden/assert_reproducible/audit_event）

铁律：纯整数（audit_float 双层）/ 核心无墙钟 / 确定性 bit-identical / append-only。
"""
