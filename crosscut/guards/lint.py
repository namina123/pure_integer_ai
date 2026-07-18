"""crosscut.guards.lint — AST CI 门（源码层·不可关）。

§9.2 实施约束：AST CI 必须真扫，非只扫 float 字面量。本模块扫四类违例：

1. 浮点：float 字面量 + float()/round() 调用（委托 float_guard.scan_file）。
2. 墙钟/随机：time.time/time_ns/monotonic*/perf_counter、datetime、random 的
   import 与调用——核心无墙钟（timestamp_seq AUTOINCREMENT 是唯一时间源），
   DRNG 是唯一随机入口（禁 random）。**真扫 import + 调用，非只字面量。**
3. 依赖方向：模块只能 import 同层或更低层 pure_integer_ai 子包（依赖只向下）。
4. （Stage 0 预留）危险词：专利敏感词在对外文件出现告警（术语脱钩表·实施期扩）。

CI 入口 run_lint(root)：四类全 clean 则 exit 0，否则 exit 1 并打印违例。
"""
from __future__ import annotations

import ast
import os
import sys

from pure_integer_ai.crosscut.guards import float_guard

# ---- 墙钟/随机禁用名 ----
# import 这些模块即违例（核心无墙钟·DRNG 唯一随机入口）。
_FORBIDDEN_MODULES = frozenset({"time", "datetime", "random"})
# 调用这些属性即违例（防 `import time as _t; _t.time()` 之类绕过——属性名扫描兜底）。
_FORBIDDEN_ATTRS = frozenset({
    "time", "time_ns", "monotonic", "monotonic_ns", "perf_counter",
    "perf_counter_ns", "strftime", "strptime", "now", "today", "utcnow",
    "random", "randint", "randrange", "choice", "shuffle", "seed", "gauss",
    "uniform", "sample",
})


def _walk_py_files(root: str, whitelist: tuple[str, ...] = ()):
    root_abs = os.path.abspath(root)
    wl_abs = [os.path.abspath(os.path.join(root_abs, w)) for w in whitelist]
    for dirpath, dirnames, filenames in os.walk(root_abs):
        ap = os.path.abspath(dirpath)
        if any(ap == w or ap.startswith(w + os.sep) for w in wl_abs):
            dirnames[:] = []
            continue
        for fn in filenames:
            if fn.endswith(".py"):
                yield os.path.join(dirpath, fn)


def _forbidden_violations(tree: ast.AST, src_label: str) -> list[str]:
    """扫墙钟/随机 import 与调用（§9.2 真扫 import + 调用）。"""
    vios: list[str] = []
    for node in ast.walk(tree):
        # import time / import datetime / import random
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in _FORBIDDEN_MODULES:
                    vios.append(
                        f"{src_label}: 禁用模块 import {alias.name} at line "
                        f"{getattr(node, 'lineno', '?')}"
                    )
        # from time import ... / from datetime import ...
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            top = mod.split(".")[0]
            if top in _FORBIDDEN_MODULES:
                vios.append(
                    f"{src_label}: 禁用模块 from {mod} import at line "
                    f"{getattr(node, 'lineno', '?')}"
                )
        # time.time() / datetime.now() / random() 调用（属性名兜底）
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute) and f.attr in _FORBIDDEN_ATTRS:
                vios.append(
                    f"{src_label}: 禁用调用 .{f.attr}() at line "
                    f"{getattr(node, 'lineno', '?')}"
                )
            elif isinstance(f, ast.Name) and f.id in _FORBIDDEN_ATTRS:
                vios.append(
                    f"{src_label}: 禁用调用 {f.id}() at line "
                    f"{getattr(node, 'lineno', '?')}"
                )
    return vios


def forbidden_check(root: str, whitelist: tuple[str, ...] = ()) -> dict[str, list[str]]:
    """扫 root 下 .py 的墙钟/随机违例。返回 {filepath: [msgs]}（空 = 干净）。"""
    violations: dict[str, list[str]] = {}
    for path in _walk_py_files(root, whitelist):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                tree = ast.parse(fh.read(), filename=path)
        except (SyntaxError, OSError):
            continue
        vios = _forbidden_violations(tree, os.path.basename(path))
        if vios:
            violations[path] = vios
    return violations


def no_float_check(root: str, whitelist: tuple[str, ...] = ()) -> dict[str, list[str]]:
    """扫 root 下 .py 的浮点违例（委托 float_guard.scan_file）。"""
    violations: dict[str, list[str]] = {}
    for path in _walk_py_files(root, whitelist):
        vios = float_guard.scan_file(path)
        if vios:
            violations[path] = vios
    return violations


# ---- 依赖只向下门（硬约束 6） ----
# 子包 → 层级（crosscut 最底；experiments 最高）。config 零依赖不约束。
_LAYER: dict[str, int] = {
    "crosscut": 0,
    "numeric": 1,
    "storage": 2,
    "vm": 3,
    "algorithm": 4,
    "cognition": 5,
    "teacher": 6,
    "training": 7,
    "experiments": 8,
}


def _file_layer(path: str, root: str) -> int | None:
    """文件所属层级（按相对 root 的首段子包）。顶层文件（无子包）返 -1（不约束）。"""
    rel = os.path.relpath(path, root)
    parts = rel.split(os.sep)
    if len(parts) < 2:
        return -1
    return _LAYER.get(parts[0])


def _imported_pure_integer_ai_segs(tree: ast.AST) -> list[tuple[str, int]]:
    """提取 AST 中所有 pure_integer_ai.<seg>... 导入的首段 seg + 行号。"""
    out: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if len(parts) >= 2 and parts[0] == "pure_integer_ai":
                    out.append((parts[1], getattr(node, "lineno", 0)))
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            parts = mod.split(".")
            if len(parts) >= 2 and parts[0] == "pure_integer_ai":
                out.append((parts[1], getattr(node, "lineno", 0)))
    return out


def import_direction_check(root: str, whitelist: tuple[str, ...] = ()
                           ) -> dict[str, list[str]]:
    """模块只能 import 同层或更低层 pure_integer_ai 子包（依赖只向下）。

    违例 = 导入了严格更高层的 pure_integer_ai 子包（向上耦合）。同层与向下放行。
    """
    violations: dict[str, list[str]] = {}
    for path in _walk_py_files(root, whitelist):
        src_layer = _file_layer(path, root)
        if src_layer is None or src_layer < 0:
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                tree = ast.parse(fh.read(), filename=path)
        except (SyntaxError, OSError):
            continue
        for seg, lineno in _imported_pure_integer_ai_segs(tree):
            tgt_layer = _LAYER.get(seg)
            if tgt_layer is None:
                continue
            if tgt_layer > src_layer:
                violations.setdefault(path, []).append(
                    f"L{lineno}: 向上耦合——L{src_layer} import "
                    f"pure_integer_ai.{seg}(L{tgt_layer})，违反依赖只向下"
                )
    return violations


def run_lint(root: str, whitelist: tuple[str, ...] = ()) -> int:
    """CI 入口：四类全 clean 则 exit 0，否则 exit 1 并打印违例。"""
    exit_code = 0

    vios = no_float_check(root, whitelist)
    if vios:
        for path, msgs in vios.items():
            rel = os.path.relpath(path, root)
            print(f"{rel}:")
            for m in msgs:
                print(f"  {m}")
        print(f"no_float_lint: FAIL（{len(vios)} 个文件含浮点）")
        exit_code = 1
    else:
        print("no_float_lint: clean")

    fv = forbidden_check(root, whitelist)
    if fv:
        for path, msgs in fv.items():
            rel = os.path.relpath(path, root)
            print(f"{rel}:")
            for m in msgs:
                print(f"  {m}")
        print(f"forbidden_call_lint: FAIL（{len(fv)} 个文件含墙钟/随机）")
        exit_code = 1
    else:
        print("forbidden_call_lint: clean（无 time/datetime/random）")

    dv = import_direction_check(root, whitelist)
    if dv:
        for path, msgs in dv.items():
            rel = os.path.relpath(path, root)
            print(f"{rel}:")
            for m in msgs:
                print(f"  {m}")
        print(f"import_direction_lint: FAIL（{len(dv)} 个文件向上耦合）")
        exit_code = 1
    else:
        print("import_direction_lint: clean（依赖只向下）")

    return exit_code


def _default_main() -> int:
    """默认扫 pure_integer_ai 包核心（白名单 tests/_archive）。"""
    import pure_integer_ai
    pkg = os.path.dirname(pure_integer_ai.__file__)
    return run_lint(pkg, whitelist=())


if __name__ == "__main__":  # python -m pure_integer_ai.crosscut.guards.lint
    sys.exit(_default_main())
