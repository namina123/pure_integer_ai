"""审计生产源码对象模型，并阻止既存债务继续增长。"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Iterator


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = REPOSITORY_ROOT / "src" / "pure_integer_ai"
DEFAULT_BASELINE = Path(__file__).with_name("object_model_baseline_v1.json")
BASELINE_SCHEMA = "pure_integer_ai.object_model_baseline"
BASELINE_VERSION = 1
MODEL_KINDS = frozenset({"value", "lifecycle", "protocol", "exception"})
BASELINE_FIELDS = frozenset({
    "id", "dataclass", "frozen", "slots", "protocol", "exception",
})
MARKER_RE = re.compile(
    r"^\s*#\s*object-model:\s*"
    r"(value|lifecycle|protocol|exception)(?:\s*;\s*(.*))?\s*$"
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _iter_class_defs(
        node: ast.AST,
        scope: tuple[str, ...] = (),
        ) -> Iterator[tuple[ast.ClassDef, str]]:
    """按词法作用域枚举类定义，保留函数内局部类的稳定身份。"""
    if isinstance(node, ast.ClassDef):
        qualname = ".".join((*scope, node.name))
        yield node, qualname
        nested_scope = (*scope, node.name)
        for child in node.body:
            yield from _iter_class_defs(child, nested_scope)
        return
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        nested_scope = (*scope, node.name, "<locals>")
        for child in node.body:
            yield from _iter_class_defs(child, nested_scope)
        return
    for child in ast.iter_child_nodes(node):
        yield from _iter_class_defs(child, scope)


def _expr_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _expr_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    if isinstance(node, ast.Subscript):
        return _expr_name(node.value)
    return ""


def _decorator_call(node: ast.expr) -> tuple[str, tuple[ast.keyword, ...]]:
    if isinstance(node, ast.Call):
        return _expr_name(node.func), tuple(node.keywords)
    return _expr_name(node), ()


def _literal_true_keyword(keywords: tuple[ast.keyword, ...], name: str) -> bool:
    for keyword in keywords:
        if keyword.arg != name:
            continue
        return (
            isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
        )
    return False


def _dataclass_shape(node: ast.ClassDef) -> tuple[bool, bool, bool]:
    for decorator in node.decorator_list:
        name, keywords = _decorator_call(decorator)
        if name.rsplit(".", 1)[-1] != "dataclass":
            continue
        return (
            True,
            _literal_true_keyword(keywords, "frozen"),
            _literal_true_keyword(keywords, "slots"),
        )
    return False, False, False


def _marker_for(
        lines: tuple[str, ...],
        node: ast.ClassDef,
        ) -> tuple[dict[str, str] | None, str | None]:
    start_line = min(
        (item.lineno for item in node.decorator_list),
        default=node.lineno,
    )
    marker_line = start_line - 1
    if marker_line < 1:
        return None, None
    source = lines[marker_line - 1]
    match = MARKER_RE.fullmatch(source)
    if match is None:
        if "object-model:" in source:
            return None, f"L{marker_line}: object-model 声明格式无效"
        return None, None

    marker = {"kind": match.group(1)}
    raw_fields = match.group(2)
    if not raw_fields:
        return marker, None
    for raw_field in raw_fields.split(";"):
        key, separator, value = raw_field.strip().partition("=")
        if not separator or not key or not value.strip():
            return None, f"L{marker_line}: object-model 元数据格式无效"
        if key in marker:
            return None, f"L{marker_line}: object-model 元数据 {key} 重复"
        marker[key] = value.strip()
    return marker, None


def _resolve_local_base(
        base: str,
        qualname: str,
        local_by_qualname: dict[str, dict[str, object]],
        ) -> dict[str, object] | None:
    if not base:
        return None
    if "." in base:
        return local_by_qualname.get(base)
    parent = qualname.split(".")[:-1]
    while parent:
        candidate = ".".join((*parent, base))
        if candidate in local_by_qualname:
            return local_by_qualname[candidate]
        parent.pop()
    return local_by_qualname.get(base)


def _apply_inheritance_families(facts: list[dict[str, object]]) -> None:
    by_path: dict[str, dict[str, dict[str, object]]] = {}
    for fact in facts:
        path = str(fact["path"])
        by_path.setdefault(path, {})[str(fact["qualname"])] = fact

    changed = True
    while changed:
        changed = False
        for fact in facts:
            local = by_path[str(fact["path"])]
            for base in fact["bases"]:
                base_name = str(base)
                leaf = base_name.rsplit(".", 1)[-1]
                parent = _resolve_local_base(
                    base_name, str(fact["qualname"]), local)
                protocol = (
                    leaf == "Protocol"
                    or leaf.endswith("Protocol")
                    or bool(parent and parent["protocol"])
                )
                exception = (
                    leaf in {"BaseException", "Exception"}
                    or leaf.endswith("Error")
                    or leaf.endswith("Exception")
                    or bool(parent and parent["exception"])
                )
                if protocol and not fact["protocol"]:
                    fact["protocol"] = True
                    changed = True
                if exception and not fact["exception"]:
                    fact["exception"] = True
                    changed = True


def scan_source(source_root: Path) -> list[dict[str, object]]:
    """扫描生产源码，返回带声明和结构形状的确定性类台账。"""
    source_root = source_root.resolve()
    if not source_root.is_dir():
        raise ValueError(f"源码目录不存在: {source_root}")

    facts: list[dict[str, object]] = []
    occurrences: dict[tuple[str, str], int] = {}
    for path in sorted(source_root.rglob("*.py")):
        relative = path.relative_to(source_root).as_posix()
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=relative)
        except (OSError, UnicodeError, SyntaxError) as error:
            raise ValueError(f"无法解析 {relative}: {error}") from error
        lines = tuple(source.splitlines())
        for node, qualname in _iter_class_defs(tree):
            key = relative, qualname
            ordinal = occurrences.get(key, 0) + 1
            occurrences[key] = ordinal
            suffix = "" if ordinal == 1 else f"#{ordinal}"
            dataclass, frozen, slots = _dataclass_shape(node)
            marker, marker_error = _marker_for(lines, node)
            facts.append({
                "id": f"{relative}::{qualname}{suffix}",
                "path": relative,
                "qualname": qualname,
                "line": node.lineno,
                "bases": tuple(_expr_name(base) for base in node.bases),
                "dataclass": dataclass,
                "frozen": frozen,
                "slots": slots,
                "protocol": False,
                "exception": False,
                "marker": marker,
                "marker_error": marker_error,
            })
    _apply_inheritance_families(facts)
    return sorted(facts, key=lambda item: str(item["id"]))


def _baseline_shape(fact: dict[str, object]) -> dict[str, object]:
    return {
        "id": fact["id"],
        "dataclass": bool(fact["dataclass"]),
        "frozen": bool(fact["frozen"]),
        "slots": bool(fact["slots"]),
        "protocol": bool(fact["protocol"]),
        "exception": bool(fact["exception"]),
    }


def build_baseline(source_root: Path) -> dict[str, object]:
    classes = [_baseline_shape(fact) for fact in scan_source(source_root)]
    return {
        "schema": BASELINE_SCHEMA,
        "version": BASELINE_VERSION,
        "scope": "src/pure_integer_ai/**/*.py",
        "class_count": len(classes),
        "inventory_sha256": _sha256(classes),
        "classes": classes,
    }


def create_baseline(source_root: Path, baseline_path: Path) -> dict[str, object]:
    """首次创建冻结台账；拒绝覆盖，避免把新增债务静默洗入基线。"""
    baseline = build_baseline(source_root)
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    with baseline_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("{\n")
        for field in (
                "schema", "version", "scope", "class_count",
                "inventory_sha256"):
            encoded = json.dumps(baseline[field], ensure_ascii=False)
            stream.write(f"  {json.dumps(field)}: {encoded},\n")
        stream.write('  "classes": [\n')
        classes = baseline["classes"]
        for index, item in enumerate(classes):
            suffix = "," if index + 1 < len(classes) else ""
            encoded = json.dumps(
                item, ensure_ascii=False, separators=(",", ":"))
            stream.write(f"    {encoded}{suffix}\n")
        stream.write("  ]\n}\n")
    return baseline


def load_baseline(baseline_path: Path) -> dict[str, dict[str, object]]:
    try:
        value = json.loads(baseline_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"无法读取对象模型 baseline: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("对象模型 baseline 顶层必须是对象")
    if value.get("schema") != BASELINE_SCHEMA:
        raise ValueError("对象模型 baseline schema 不匹配")
    if value.get("version") != BASELINE_VERSION:
        raise ValueError("对象模型 baseline version 不匹配")
    classes = value.get("classes")
    if not isinstance(classes, list):
        raise ValueError("对象模型 baseline classes 必须是数组")
    if value.get("class_count") != len(classes):
        raise ValueError("对象模型 baseline class_count 不闭合")
    if value.get("inventory_sha256") != _sha256(classes):
        raise ValueError("对象模型 baseline inventory_sha256 不闭合")

    by_id: dict[str, dict[str, object]] = {}
    previous = ""
    for item in classes:
        if not isinstance(item, dict) or frozenset(item) != BASELINE_FIELDS:
            raise ValueError("对象模型 baseline class 字段不合法")
        identity = item.get("id")
        if not isinstance(identity, str) or not identity or identity <= previous:
            raise ValueError("对象模型 baseline class 身份未严格排序或重复")
        if any(type(item[field]) is not bool for field in BASELINE_FIELDS - {"id"}):
            raise ValueError(f"对象模型 baseline shape 不是严格 bool: {identity}")
        by_id[identity] = item
        previous = identity
    return by_id


def _validate_marker(fact: dict[str, object]) -> list[str]:
    marker = fact["marker"]
    if not isinstance(marker, dict):
        return ["缺少 object-model 声明"]
    kind = marker.get("kind")
    if kind not in MODEL_KINDS:
        return ["object-model kind 不合法"]
    fields = frozenset(marker) - {"kind"}
    problems: list[str] = []
    if kind == "lifecycle":
        missing = {"owner", "cleanup"} - fields
        unknown = fields - {"owner", "cleanup"}
        if missing:
            problems.append(
                "lifecycle 缺少元数据: " + ", ".join(sorted(missing)))
        if unknown:
            problems.append(
                "lifecycle 含未知元数据: " + ", ".join(sorted(unknown)))
    elif fields:
        problems.append(
            f"{kind} 不接受元数据: " + ", ".join(sorted(fields)))

    if kind == "value":
        if not fact["dataclass"]:
            problems.append("value 必须使用 @dataclass")
        if not fact["frozen"]:
            problems.append("value 必须显式 frozen=True")
        if not fact["slots"]:
            problems.append("value 必须显式 slots=True")
        if fact["protocol"] or fact["exception"]:
            problems.append("value 不能同时属于 protocol/exception")
    elif kind == "protocol" and not fact["protocol"]:
        problems.append("protocol 声明必须继承 Protocol 家族")
    elif kind == "exception" and not fact["exception"]:
        problems.append("exception 声明必须继承 Exception 家族")
    elif kind == "lifecycle" and (fact["protocol"] or fact["exception"]):
        problems.append("lifecycle 不能同时属于 protocol/exception")
    return problems


def _legacy_shape_regressions(
        current: dict[str, object],
        baseline: dict[str, object],
        ) -> list[str]:
    problems: list[str] = []
    for field, label in (
        ("dataclass", "dataclass 身份"),
        ("protocol", "protocol 身份"),
        ("exception", "exception 身份"),
    ):
        if bool(current[field]) != bool(baseline[field]):
            problems.append(f"既存类的 {label} 已变更但未显式归类")
    if baseline["frozen"] and not current["frozen"]:
        problems.append("既存 frozen dataclass 被降级为可变")
    if baseline["slots"] and not current["slots"]:
        problems.append("既存 slots dataclass 被移除 slots")
    return problems


def check_object_model(
        source_root: Path = DEFAULT_SOURCE_ROOT,
        baseline_path: Path = DEFAULT_BASELINE,
        ) -> tuple[list[str], int]:
    baseline = load_baseline(baseline_path)
    facts = scan_source(source_root)
    violations: list[str] = []
    for fact in facts:
        prefix = f"{fact['path']}:L{fact['line']}:{fact['qualname']}"
        marker_error = fact["marker_error"]
        if marker_error:
            violations.append(f"{prefix}: {marker_error}")
            continue
        legacy = baseline.get(str(fact["id"]))
        if fact["marker"] is not None:
            if legacy is not None:
                violations.append(
                    f"{prefix}: 已声明类仍在 legacy baseline；"
                    "请从 baseline 删除该条目并重算摘要"
                )
                continue
            for problem in _validate_marker(fact):
                violations.append(f"{prefix}: {problem}")
            continue
        if legacy is None:
            violations.append(f"{prefix}: 缺少 object-model 声明")
            continue
        for problem in _legacy_shape_regressions(fact, legacy):
            violations.append(f"{prefix}: {problem}")
    return violations, len(facts)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="阻止生产源码对象模型债务超过冻结 baseline。"
    )
    parser.add_argument(
        "--source-root", type=Path, default=DEFAULT_SOURCE_ROOT,
        help="待扫描的生产源码根目录",
    )
    parser.add_argument(
        "--baseline", type=Path, default=DEFAULT_BASELINE,
        help="冻结 legacy baseline 路径",
    )
    parser.add_argument(
        "--create-baseline", action="store_true",
        help="首次创建 baseline；目标存在时拒绝覆盖",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")
    args = _build_parser().parse_args(argv)
    try:
        if args.create_baseline:
            baseline = create_baseline(args.source_root, args.baseline)
            print(
                "object_model_lint: baseline created "
                f"({baseline['class_count']} classes, "
                f"sha256={baseline['inventory_sha256']})"
            )
            return 0
        violations, class_count = check_object_model(
            args.source_root, args.baseline)
    except (ValueError, FileExistsError) as error:
        print(f"object_model_lint: ERROR: {error}")
        return 2
    if violations:
        for violation in violations:
            print(violation)
        print(
            "object_model_lint: FAIL "
            f"({len(violations)} violations across {class_count} classes)"
        )
        return 1
    print(f"object_model_lint: clean ({class_count} classes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
