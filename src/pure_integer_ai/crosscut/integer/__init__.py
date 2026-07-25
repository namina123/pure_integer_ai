"""crosscut.integer — 纯整数数学（依赖 guards·零业务）。

模块（内部依赖序：constants ← valtypes ← bigint ← algebraic_fraction ←
isqrt/compare/rational ← fixed_point）：
  constants         BASE/DEFAULT_K 等数值常量单一真相源
  valtypes          Rational/FixedQuotient 不可变值类型 + 不变量声明
  bigint            base 2^30 limb 大数器（移植锚点）
  algebraic_fraction  longdiv 代分数引擎（不变量 a·B^k==M·b+r）+ mod 原子
  isqrt             纯整数开方（牛顿）+ SqrtRef 守 M²≤n·S²<(M+1)²
  compare           cross_compare 交叉积比序（零误差·全系统比序唯一路径）
  rational          Rational 算术（闭运算·eq 走交叉积）
  fixed_point       FixedQuotient 定点算术 + rational_div
  unicode_codec     码点（ord/chr）边缘编解码（核心↔外缘唯一 chr/ord 协议·文本=码点有序数组）

铁律：纯整数（入口 assert_no_float）/ 不变量断言式门 / 跨宿主 bit 一致。
"""
