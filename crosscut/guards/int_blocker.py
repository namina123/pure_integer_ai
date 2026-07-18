"""crosscut.guards.int_blocker — 第三层纯整数守卫（类型 assert）。

双层之外第三层（§9.2「三层叠加」）：在 setter / Instruction 构造 / opcode dispatch
边界做类型 assert，拒绝非 int 进入核心数据结构。与 float_guard（值级 float）互补：
float_guard 拦 float 实例；int_blocker 拦 bool 之外的"非 int 数值"（如未来宿主泄漏的
Decimal/复杂对象），并给 vm Instruction 构造提供类型闸门。

Stage 0 提供原子；Instruction/dispatch 闸门随 vm（Stage 2）填实。
"""
from __future__ import annotations

from typing import Any


class IntViolation(AssertionError):
    """非整数入核心违例（AssertionError 子类）。"""


def assert_int(*values: Any, _where: str = "") -> None:
    """断言每个 value 都是 int（bool 是 int 子类，单独放行——逻辑布尔不是数值）。

    严格拒绝 float（float_guard 已守·float 非 int 子类·下方 not isinstance(int) 已拦）
    与其他非 int 类型。

    **perf fast path**（2.3M+ 调/训练 run）：`type(v) is int` 身份比较一次短路纯 int 常见 case
    （非 bool/子类）·省 2 isinstance。bit-identical：纯 int type-is-int True->continue（同原
    bool False+int True 路径）；bool type-is-int False->isinstance(bool) True->continue（同）；
    float/其他 type-is-int False->bool False->not isinstance(int) True->raise（同·原 dead float
    check 不可达已删）。
    """
    for v in values:
        if type(v) is int:   # fast path：精确 int（非 bool/子类）·常见 case 一次身份比较
            continue
        if isinstance(v, bool):
            continue  # bool 是逻辑值，不是数值；允许作标志位
        if not isinstance(v, int):
            raise IntViolation(
                f"non-int {_where and f'({_where}) '}: {type(v).__name__}={v!r}"
            )


def require_int(value: Any, *, name: str = "value") -> int:
    """断言 value 为 int 并返回（用于 setter/构造器入口）。bool 放行。"""
    assert_int(value, _where=name)
    return value  # type: ignore[return-value]
