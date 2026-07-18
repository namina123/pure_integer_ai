"""crosscut.guards.float_guard — 禁浮点守卫（运行时层 + 浮点 AST 扫描）。

硬约束1「核心零浮点」的执行点之一（运行时层）。

提供：
- assert_no_float(*values)：值级断言，任何 float 实例即抛 FloatViolation。
- @int_only：装饰器，DEBUG 模式下断言纯整数入口的入参/返回值无 float。
- scan_source / scan_file / scan_module：AST 浮点扫描（float 字面量 + float()/round()
  调用）。time/datetime/random 的扫描见 lint.py（CI 门）。

双层（§9.2 A7）：
  - 运行时层（本模块 DEBUG）：可在生产热路径 export ZERO_AI_AUDIT_FLOAT=0 关闭，
    省去 isinstance 重复检查开销。
  - 源码层（lint.py AST CI）：独立于本开关，永远守源码层禁浮点 + 禁墙钟 + 禁随机。
  故关 DEBUG 只去运行时重复检查，不破纯整数铁律。

DEBUG 是模块级全局、调用点 live-read（assert_no_float/int_only 每次调读 DEBUG），
故亦可运行时直接赋值 float_guard.DEBUG=False 覆盖（gate live-read 纪律）。
"""
from __future__ import annotations

import ast
import functools
import os
from typing import Any, Callable

# 默认 ON（守测试/dev/CI）。生产热路径 export ZERO_AI_AUDIT_FLOAT=0 关运行时守卫。
# 须在 import 本模块前设 env（run 脚本 export 先于 import）。
DEBUG = os.environ.get("ZERO_AI_AUDIT_FLOAT", "1") != "0"

# 被禁的"浮点制造"调用名（AST Call 的 func 为这些 Name/Attribute 即违规）。
_FLOAT_CALLS = frozenset({"float", "round"})


class FloatViolation(AssertionError):
    """禁浮点违例（AssertionError 子类，故 assert 语义统一）。"""


def assert_no_float(*values: Any, _where: str = "") -> None:
    """值级守卫：任一 value 为 float 实例即抛。bool 是 int 子类，不算 float。"""
    if not DEBUG:
        return
    for v in values:
        if isinstance(v, float):
            raise FloatViolation(
                f"float detected{_where and f' ({_where})'}: {v!r}"
            )


def int_only(fn: Callable[..., Any]) -> Callable[..., Any]:
    """装饰器：DEBUG 下断言入参与返回值无 float（用于纯整数入口）。"""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if DEBUG:
            for a in args:
                assert_no_float(a, _where=f"arg of {fn.__name__}")
            for v in kwargs.values():
                assert_no_float(v, _where=f"kwarg of {fn.__name__}")
        r = fn(*args, **kwargs)
        if DEBUG:
            assert_no_float(r, _where=f"return of {fn.__name__}")
        return r

    return wrapper


# ---- AST 浮点源码扫描 ----

def _float_violations_of_tree(tree: ast.AST, src_label: str) -> list[str]:
    vios: list[str] = []
    for node in ast.walk(tree):
        # float 字面量
        if isinstance(node, ast.Constant) and isinstance(node.value, float):
            vios.append(
                f"{src_label}: float literal {node.value!r} at line "
                f"{getattr(node, 'lineno', '?')}"
            )
        # float()/round() 调用
        if isinstance(node, ast.Call):
            fname: str | None = None
            f = node.func
            if isinstance(f, ast.Name):
                fname = f.id
            elif isinstance(f, ast.Attribute):
                fname = f.attr
            if fname in _FLOAT_CALLS:
                vios.append(
                    f"{src_label}: {fname}() call at line "
                    f"{getattr(node, 'lineno', '?')}"
                )
    return vios


def scan_source(source: str) -> list[str]:
    """扫描源码字符串的浮点违例，返回违例列表（空 = 干净）。"""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"<unparseable source>: {e}"]
    return _float_violations_of_tree(tree, "<source>")


def scan_file(path: str) -> list[str]:
    """扫描 .py 文件的浮点违例，返回违例列表。"""
    with open(path, "r", encoding="utf-8") as fh:
        src = fh.read()
    try:
        tree = ast.parse(src, filename=path)
    except SyntaxError as e:
        return [f"{path}: <unparseable>: {e}"]
    return _float_violations_of_tree(tree, os.path.basename(path))


def scan_module(module: Any) -> list[str]:
    """扫描模块对象（用 __file__）；也可传路径或源码字符串（自适应）。"""
    f = getattr(module, "__file__", None)
    if f:
        return scan_file(f)
    if isinstance(module, str):
        if module.endswith(".py") or os.path.exists(module):
            return scan_file(module)
        return scan_source(module)
    raise TypeError(f"scan_module: 不支持的参数类型 {type(module)!r}")
