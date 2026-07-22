"""crosscut.guards.lint — AST CI 门（源码层·不可关）。

§9.2 实施约束：AST CI 必须真扫，非只扫 float 字面量。本模块扫五类违例：

1. 浮点：float 字面量 + float()/round() 调用（委托 float_guard.scan_file）。
2. 墙钟/随机：time.time/time_ns/monotonic*/perf_counter、datetime、random 的
   import 与调用——核心无墙钟（timestamp_seq AUTOINCREMENT 是唯一时间源），
   DRNG 是唯一随机入口（禁 random）。**真扫 import + 调用，非只字面量。**
3. 依赖方向：模块只能 import 同层或更低层 pure_integer_ai 子包（依赖只向下）。
4. 解释语言：解释性注释和文档字符串必须使用中文，公式、协议名和标识符短语放行。
5. （Stage 0 预留）危险词：专利敏感词在对外文件出现告警（术语脱钩表·实施期扩）。

CI 入口 run_lint(root)：全部守卫 clean 则 exit 0，否则 exit 1 并打印违例。
"""
from __future__ import annotations

import ast
import io
import os
import re
import sys
import tokenize

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


# ---- 解释性注释和文档字符串中文门 ----

_CJK_RE = re.compile(r"[\u3400-\u9fff]")
_ENGLISH_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_NATURAL_LINKERS = frozenset({
    "a", "an", "the", "this", "that", "these", "those", "and", "or",
    "is", "are", "was", "were", "be", "for", "from", "to", "of",
    "with", "without", "when", "where", "why", "how", "into", "after",
    "before", "during", "through", "if", "each", "all", "any", "only",
    "should", "must", "can", "cannot", "returns", "reads", "writes",
    "stores", "loads", "creates", "keeps", "allows", "ensures", "means",
})
_STRONG_EXPLANATION_WORDS = frozenset({
    "this", "that", "these", "those", "because", "should", "must",
    "returns", "reads", "writes", "stores", "loads", "creates", "keeps",
    "allows", "ensures", "represents", "explains",
})
_COMMENT_DIRECTIVES = (
    "noqa", "type: ignore", "pragma:", "pylint:", "ruff:", "fmt:",
    "coding:", "coding=", "pyright:", "mypy:",
)
_PYTHON_COMMENT_PREFIXES = (
    "from ", "import ", "def ", "class ", "return ", "raise ", "yield ",
)


def _first_explanation_line(text: str) -> str:
    """取文档字符串首个非空说明行。"""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _looks_like_english_explanation(text: str, *, docstring: bool) -> bool:
    """保守识别纯英文自然语言，避免把公式和协议短语当解释。"""
    summary = _first_explanation_line(text) if docstring else text.strip()
    if not summary or _CJK_RE.search(summary):
        return False
    lowered = summary.lower()
    if not docstring and (
            lowered.startswith(_COMMENT_DIRECTIVES)
            or lowered.startswith(_PYTHON_COMMENT_PREFIXES)):
        return False
    if not docstring and re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_.:()]*[。.]?", summary):
        return False
    words = _ENGLISH_WORD_RE.findall(summary)
    lowercase_words = [
        word.lower() for word in words if len(word) > 1 and word.islower()
    ]
    if len(lowercase_words) < 4:
        return False
    word_set = set(lowercase_words)
    if not (word_set & _NATURAL_LINKERS):
        return False

    # 多个代码符号通常表示公式、调用签名或协议结构；强解释词仍必须使用中文。
    code_markers = sum(marker in summary for marker in (
        "->", "→", "=", "_", "(", ")", "[", "]", "{", "}", "::",
        "\\", "/",
    ))
    if code_markers >= 2 and not (word_set & _STRONG_EXPLANATION_WORDS):
        return False
    if docstring:
        return (summary.rstrip().endswith((".", "!", "?"))
                or len(lowercase_words) >= 6)
    return True


def chinese_explanation_check(root: str, whitelist: tuple[str, ...] = ()
                              ) -> dict[str, list[str]]:
    """扫描纯英文解释性注释和文档字符串。"""
    violations: dict[str, list[str]] = {}
    for path in _walk_py_files(root, whitelist):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                source = fh.read()
            tree = ast.parse(source, filename=path)
        except (SyntaxError, OSError):
            continue

        for node in ast.walk(tree):
            if not isinstance(node, (
                    ast.Module, ast.ClassDef, ast.FunctionDef,
                    ast.AsyncFunctionDef)):
                continue
            doc = ast.get_docstring(node, clean=False)
            if doc is None or not _looks_like_english_explanation(
                    doc, docstring=True):
                continue
            line = getattr(node.body[0], "lineno", getattr(node, "lineno", 1))
            violations.setdefault(path, []).append(
                f"L{line}: 解释性文档字符串必须使用中文"
            )

        try:
            tokens = tokenize.generate_tokens(io.StringIO(source).readline)
            for token in tokens:
                if token.type != tokenize.COMMENT:
                    continue
                comment = token.string[1:].strip()
                if _looks_like_english_explanation(comment, docstring=False):
                    violations.setdefault(path, []).append(
                        f"L{token.start[0]}: 解释性注释必须使用中文"
                    )
        except tokenize.TokenError:
            continue
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
    """CI 入口：全部守卫 clean 则 exit 0，否则 exit 1 并打印违例。"""
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

    cv = chinese_explanation_check(root, whitelist)
    if cv:
        for path, msgs in cv.items():
            rel = os.path.relpath(path, root)
            print(f"{rel}:")
            for m in msgs:
                print(f"  {m}")
        print(f"chinese_explanation_lint: FAIL（{len(cv)} 个文件含英文解释）")
        exit_code = 1
    else:
        print("chinese_explanation_lint: clean（解释性注释和文档字符串使用中文）")

    return exit_code


def _default_main() -> int:
    """默认扫 pure_integer_ai 包核心（白名单 tests/_archive）。"""
    import pure_integer_ai
    pkg = os.path.dirname(pure_integer_ai.__file__)
    return run_lint(pkg, whitelist=())


if __name__ == "__main__":  # python -m pure_integer_ai.crosscut.guards.lint
    sys.exit(_default_main())
