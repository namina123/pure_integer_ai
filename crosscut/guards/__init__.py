"""crosscut.guards — 禁浮点守卫（叶层·零依赖）。

双层守卫（§9.2 A7）：
  - 运行时层：assert_no_float / int_only（DEBUG 可关，生产热路径省开销）
  - 源码层：AST CI lint（不可关·真扫 float 字面量 + float()/round() 调用
    + time.time/datetime/random 及 import · §9.2 实施约束）

【偏离规划草稿的诚实标注】
  落地规划§一把 audit_float 列在 determinism/ 下。但 integer/ 的纯整数入口需要
  assert_no_float 守门，若 audit_float 在 determinism/ 则形成 integer↔determinism
  循环（determinism.cross_radix 又依赖 integer.longdiv）。故把浮点守卫下沉到 guards/
  叶层，依赖序干净为 guards ← integer ← determinism。算法与不变量不变，仅归层调整。
"""
from pure_integer_ai.crosscut.guards.float_guard import (
    DEBUG,
    FloatViolation,
    assert_no_float,
    int_only,
    scan_source,
    scan_file,
    scan_module,
)
from pure_integer_ai.crosscut.guards.lint import (
    no_float_check,
    import_direction_check,
    run_lint,
)

__all__ = [
    "DEBUG",
    "FloatViolation",
    "assert_no_float",
    "int_only",
    "scan_source",
    "scan_file",
    "scan_module",
    "no_float_check",
    "import_direction_check",
    "run_lint",
]
