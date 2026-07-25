#!/usr/bin/env python3
"""统计项目核心 Python 区域的有效物理代码行。"""

from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, dataclass
import io
import json
from pathlib import Path
import sys
import tokenize
from typing import Iterable, Sequence


DEFAULT_EXCLUDED_DIRS = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "benchmarks",
        "data",
        "docs",
        "examples",
        "experiments",
        "paper",
        "scratch",
        "tests",
        "venv",
    }
)


@dataclass(frozen=True)
class LineStats:
    """保存一个文件或区域中互斥的物理行分类统计。"""

    files: int = 0
    physical: int = 0
    effective: int = 0
    comments: int = 0
    docstrings: int = 0
    blank: int = 0
    other: int = 0

    def __add__(self, other: "LineStats") -> "LineStats":
        """逐字段合并两份统计，保持各行分类互斥。"""

        return LineStats(
            files=self.files + other.files,
            physical=self.physical + other.physical,
            effective=self.effective + other.effective,
            comments=self.comments + other.comments,
            docstrings=self.docstrings + other.docstrings,
            blank=self.blank + other.blank,
            other=self.other + other.other,
        )


@dataclass(frozen=True)
class FileResult:
    """保存单个文件的相对路径、所属区域和行统计。"""

    path: str
    area: str
    stats: LineStats


def _docstring_nodes(tree: ast.AST) -> tuple[ast.Constant, ...]:
    """提取模块、类和函数体首语句中的真实文档字符串节点。"""

    owners = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    found: list[ast.Constant] = []
    for node in ast.walk(tree):
        if not isinstance(node, owners) or not node.body:
            continue
        first = node.body[0]
        if not isinstance(first, ast.Expr):
            continue
        value = first.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            found.append(value)
    return tuple(found)


def _is_docstring_token(
    token: tokenize.TokenInfo,
    nodes: Sequence[ast.Constant],
) -> bool:
    """判断字符串词元是否对应 AST 已确认的文档字符串。"""

    if token.type != tokenize.STRING:
        return False
    for node in nodes:
        end_lineno = node.end_lineno or node.lineno
        if token.start[0] == node.lineno and token.end[0] == end_lineno:
            return True
    return False


def count_file(path: Path) -> LineStats:
    """解析一个 Python 文件并统计互斥的有效、注释、文档串和空白行。"""

    source = tokenize.open(path).read()
    lines = source.splitlines()
    physical_lines = set(range(1, len(lines) + 1))
    blank_lines = {index for index, line in enumerate(lines, 1) if not line.strip()}

    tree = ast.parse(source, filename=str(path))
    doc_nodes = _docstring_nodes(tree)
    code_lines: set[int] = set()
    comment_lines: set[int] = set()
    docstring_lines: set[int] = set()
    ignored_tokens = {
        tokenize.ENCODING,
        tokenize.ENDMARKER,
        tokenize.INDENT,
        tokenize.DEDENT,
        tokenize.NEWLINE,
        tokenize.NL,
    }

    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        start_line, end_line = token.start[0], token.end[0]
        covered = set(range(start_line, end_line + 1)) & physical_lines
        if token.type == tokenize.COMMENT:
            comment_lines.update(covered)
        elif _is_docstring_token(token, doc_nodes):
            docstring_lines.update(covered)
        elif token.type not in ignored_tokens:
            code_lines.update(covered)

    docstring_only = docstring_lines - code_lines
    comment_only = comment_lines - code_lines - docstring_only
    blank_only = blank_lines - code_lines - docstring_only - comment_only
    classified = code_lines | docstring_only | comment_only | blank_only
    other_lines = physical_lines - classified

    return LineStats(
        files=1,
        physical=len(physical_lines),
        effective=len(code_lines),
        comments=len(comment_only),
        docstrings=len(docstring_only),
        blank=len(blank_only),
        other=len(other_lines),
    )


def _contains_excluded_part(path: Path, excluded: frozenset[str]) -> bool:
    """检查相对路径中是否出现应排除的目录名。"""

    return any(part in excluded for part in path.parts[:-1])


def _discover_default_roots(project_root: Path, excluded: frozenset[str]) -> list[Path]:
    """自动发现项目根下含 Python 文件的非实验运行包。"""

    roots: list[Path] = []
    for child in sorted(project_root.iterdir(), key=lambda item: item.name):
        if not child.is_dir() or child.name in excluded:
            continue
        if (child / "__init__.py").is_file():
            roots.append(child)
    return roots


def discover_files(
    project_root: Path,
    requested_paths: Sequence[str],
    excluded: frozenset[str],
) -> tuple[list[Path], list[Path]]:
    """按显式路径或自动发现规则返回统计根和去重后的 Python 文件。"""

    if requested_paths:
        roots = [
            path if path.is_absolute() else project_root / path
            for path in map(Path, requested_paths)
        ]
    else:
        roots = _discover_default_roots(project_root, excluded)

    missing = [path for path in roots if not path.exists()]
    if missing:
        rendered = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"统计路径不存在: {rendered}")

    files: set[Path] = set()
    for root in roots:
        candidates: Iterable[Path]
        if root.is_file():
            candidates = (root,) if root.suffix == ".py" else ()
        else:
            candidates = root.rglob("*.py")
        for candidate in candidates:
            relative = candidate.relative_to(project_root)
            if not _contains_excluded_part(relative, excluded):
                files.add(candidate.resolve())
    return roots, sorted(files)


def _area_for(path: Path, project_root: Path) -> str:
    """把文件归入项目根下的首层区域，供汇总展示。"""

    relative = path.relative_to(project_root)
    return relative.parts[0] if len(relative.parts) > 1 else "."


def collect_results(files: Sequence[Path], project_root: Path) -> list[FileResult]:
    """统计全部文件，并保留每个文件的区域和项目相对路径。"""

    results: list[FileResult] = []
    for path in files:
        relative = path.relative_to(project_root)
        results.append(
            FileResult(
                path=relative.as_posix(),
                area=_area_for(path, project_root),
                stats=count_file(path),
            )
        )
    return results


def aggregate_by_area(results: Sequence[FileResult]) -> dict[str, LineStats]:
    """按首层源码区域汇总单文件统计。"""

    areas: dict[str, LineStats] = {}
    for result in results:
        areas[result.area] = areas.get(result.area, LineStats()) + result.stats
    return dict(sorted(areas.items()))


def _sum_stats(stats: Iterable[LineStats]) -> LineStats:
    """合计一组行统计。"""

    total = LineStats()
    for item in stats:
        total += item
    return total


def _format_table(areas: dict[str, LineStats], total: LineStats) -> str:
    """生成适合终端阅读的中文等宽统计表。"""

    headers = ("区域", "文件", "有效代码", "物理行", "纯注释", "文档串", "空白", "其他")
    rows = [
        (
            area,
            str(stats.files),
            str(stats.effective),
            str(stats.physical),
            str(stats.comments),
            str(stats.docstrings),
            str(stats.blank),
            str(stats.other),
        )
        for area, stats in areas.items()
    ]
    rows.append(
        (
            "合计",
            str(total.files),
            str(total.effective),
            str(total.physical),
            str(total.comments),
            str(total.docstrings),
            str(total.blank),
            str(total.other),
        )
    )
    widths = [max(len(row[index]) for row in [headers, *rows]) for index in range(len(headers))]

    def render(row: Sequence[str]) -> str:
        """按列宽渲染一行表格。"""

        return "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))

    separator = "  ".join("-" * width for width in widths)
    return "\n".join([render(headers), separator, *(render(row) for row in rows)])


def _json_payload(
    project_root: Path,
    roots: Sequence[Path],
    areas: dict[str, LineStats],
    total: LineStats,
    results: Sequence[FileResult],
    list_files: bool,
) -> dict[str, object]:
    """构造稳定的机器可读统计结果。"""

    payload: dict[str, object] = {
        "project_root": str(project_root),
        "scan_roots": [str(path) for path in roots],
        "definition": "含至少一个非注释、非文档字符串的有效词元的物理行",
        "areas": {area: asdict(stats) for area, stats in areas.items()},
        "total": asdict(total),
    }
    if list_files:
        payload["files"] = [
            {"path": result.path, "area": result.area, **asdict(result.stats)}
            for result in results
        ]
    return payload


def build_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器并公开统计口径的可配置项。"""

    workspace_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="统计核心 Python 区域的有效物理代码行。",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="相对项目根或绝对统计路径；省略时自动发现非实验运行包。",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=workspace_root / "src" / "pure_integer_ai",
        help="项目根目录，默认使用工程根下的 pure_integer_ai。",
    )
    parser.add_argument(
        "--exclude-dir",
        action="append",
        default=[],
        help="额外排除的目录名，可重复传入。",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="输出格式。",
    )
    parser.add_argument(
        "--list-files",
        action="store_true",
        help="JSON 输出中附带单文件统计。",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """执行目录发现、代码行统计和结果输出，解析失败时返回非零状态。"""

    args = build_parser().parse_args(argv)
    project_root = args.project_root.resolve()
    if not project_root.is_dir():
        print(f"项目根目录不存在: {project_root}", file=sys.stderr)
        return 2

    excluded = DEFAULT_EXCLUDED_DIRS | frozenset(args.exclude_dir)
    try:
        roots, files = discover_files(project_root, args.paths, excluded)
        results = collect_results(files, project_root)
    except (FileNotFoundError, SyntaxError, tokenize.TokenError, UnicodeError) as error:
        print(f"统计失败: {error}", file=sys.stderr)
        return 1

    areas = aggregate_by_area(results)
    total = _sum_stats(areas.values())
    if args.format == "json":
        payload = _json_payload(
            project_root,
            roots,
            areas,
            total,
            results,
            args.list_files,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print("口径: 含至少一个非注释、非文档字符串的有效词元的物理行")
        print(f"项目: {project_root}")
        print(_format_table(areas, total))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


