"""crosscut.integer.constants — 数值地基层集中常量（全系统数值行为单一事实源）。

硬约束1（核心永远有理数域）支撑点：本模块自身零浮点。所有数值常量集中于此，
避免魔法数散落、便于审计与移植。
"""
from __future__ import annotations

# 代分数进制：base 2^30。大数器天然 base——使代分数统一分母 B^k 退化为
# 算术右移 M >> (30*k)，跨宿主 bit 一致（移植 C/Rust 同算法保确定性）。
BASE_BITS = 30
BASE = 1 << BASE_BITS  # = 2**30 = 1073741824

# 代分数默认精度位数。分母 = BASE ** DEFAULT_K = 2**240。
DEFAULT_K = 8

# 实现标记。'python'（任意精度先行）；移植 C/Rust 时改为 'c'/'rust'，
# 但必须用相同算法保证跨宿主 bit 一致（bigint.to_limbs/bit_eq 为对拍锚点）。
IMPL = "python"

# 定点区间比较策略：两个定点 M/B^k 真值区间重叠（差 < 1/B^k，无法定序）时，
# 回退到零误差交叉积比较（见 compare.cross_compare）。
ON_INTERVAL_OVERLAP = "fallback_to_cross_product"
