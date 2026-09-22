"""Combine storage, reachability, and route audits into a direction gate.

The audit is read-only.  It does not import the project package, select source
text from SQLite, or mutate a training run.  Its output is a compact decision
record used to prevent expensive training from starting on a dirty or
semantically incomplete lineage.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any


FORMAT = "PURE_INTEGER_PERFORMANCE_DIRECTION_AUDIT_V1"
_PROJECT_TOP = "pure_integer_ai"
_STDLIB = frozenset(getattr(sys, "stdlib_module_names", ()))


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"audit report must be an object: {path}")
    return value


def _module_name(path: Path, root: Path) -> str:
    return ".".join((_PROJECT_TOP, *path.relative_to(root).with_suffix("").parts))


def _external_imports(source_root: Path, mainline: dict[str, Any]) -> dict[str, Any]:
    modules = {
        item.get("module"): item
        for item in mainline.get("modules", ())
        if isinstance(item, dict) and isinstance(item.get("module"), str)
    }
    all_imports: set[tuple[str, str, int]] = set()
    production_imports: set[tuple[str, str, int]] = set()
    for path in sorted(source_root.rglob("*.py"), key=lambda item: str(item)):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, UnicodeDecodeError, SyntaxError):
            continue
        module = _module_name(path, source_root)
        production = bool(modules.get(module, {}).get("production_reachable"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names.append(node.module)
            for name in names:
                top = name.split(".", 1)[0]
                if top in _STDLIB or top == _PROJECT_TOP:
                    continue
                item = (module, top, int(node.lineno))
                all_imports.add(item)
                if production:
                    production_imports.add(item)
    encode = lambda values: [
        {"module": module, "top_level": top, "line": line}
        for module, top, line in sorted(values)
    ]
    return {
        "all_nonstdlib_imports": encode(all_imports),
        "production_reachable_nonstdlib_imports": encode(production_imports),
    }


def _sqlite_summary(report: dict[str, Any]) -> dict[str, Any]:
    databases = []
    duplicate_tables: list[dict[str, Any]] = []
    for item in report.get("databases", ()):
        if not isinstance(item, dict):
            continue
        tables = item.get("tables", {})
        if not isinstance(tables, dict):
            tables = {}
        for table, facts in tables.items():
            if not isinstance(facts, dict):
                continue
            groups = int(facts.get("natural_duplicate_groups", 0) or 0)
            excess = int(facts.get("natural_duplicate_excess", 0) or 0)
            if groups or excess:
                duplicate_tables.append({
                    "database": item.get("database"),
                    "table": table,
                    "natural_duplicate_groups": groups,
                    "natural_duplicate_excess": excess,
                })
        databases.append({
            "database": item.get("database"),
            "bytes": int(item.get("bytes", 0) or 0),
            "page_count": int(item.get("page_count", 0) or 0),
            "page_size": int(item.get("page_size", 0) or 0),
            "freelist_count": int(item.get("freelist_count", 0) or 0),
            "elapsed_seconds": item.get("elapsed_seconds"),
        })
    return {
        "database_count": len(databases),
        "databases": databases,
        "natural_duplicate_tables": duplicate_tables,
        "natural_duplicate_free": int(not duplicate_tables),
    }


def build_report(
        *, storage: dict[str, Any], sqlite: dict[str, Any],
        mainline: dict[str, Any], functional: dict[str, Any],
        capability: dict[str, Any], source_root: Path,
        generation_correction: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
    feature = mainline.get("feature_summary", {}).get("discourse_topic", {})
    if not isinstance(feature, dict):
        feature = {}
    route_count = int(capability.get("route_count", 0) or 0)
    route_closed = int(capability.get("route_count_closed", 0) or 0)
    production_imports = _external_imports(source_root, mainline)
    duplicate_bytes = int(storage.get("duplicate_excess_bytes", 0) or 0)
    blockers: list[str] = []
    if duplicate_bytes:
        blockers.append("HISTORICAL_PHYSICAL_DUPLICATES_MUST_NOT_FEED_NEW_TRAINING")
    if not sqlite["natural_duplicate_free"]:
        blockers.append("SQLITE_NATURAL_KEY_DUPLICATES")
    if production_imports["production_reachable_nonstdlib_imports"]:
        blockers.append("PRODUCTION_NONSTDLIB_IMPORT")
    if int(feature.get("integrated_candidate_count", 0) or 0) <= 0:
        blockers.append("DISCOURSE_TOPIC_NOT_ON_PRODUCTION_QUERYSTATE_PATH")
    if route_count <= 0 or route_closed < route_count:
        blockers.append("GENERATION_ROUTE_CLOSURE_INCOMPLETE")
    status = "AUDIT_COMPLETE_DIRECTION_CORRECTED" if blockers else "AUDIT_COMPLETE"
    return {
        "format": FORMAT,
        "schema_version": 1,
        "read_only": 1,
        "source_text_selected": 0,
        "source_root": str(source_root.resolve()),
        "storage": {
            "file_count": int(storage.get("file_count", 0) or 0),
            "total_bytes": int(storage.get("total_bytes", 0) or 0),
            "unique_content_bytes": int(storage.get("unique_content_bytes", 0) or 0),
            "duplicate_excess_bytes": duplicate_bytes,
            "duplicate_group_count": int(storage.get("duplicate_group_count", 0) or 0),
            "top_duplicate_groups": [
                {
                    "size": int(item.get("size", 0) or 0),
                    "count": int(item.get("count", 0) or 0),
                    "duplicate_excess_bytes": int(
                        item.get("duplicate_excess_bytes", 0) or 0),
                    "roles": item.get("roles", ()),
                    "locations": item.get("locations", ()),
                }
                for item in sorted(
                    storage.get("duplicate_groups", ()),
                    key=lambda value: int(value.get("duplicate_excess_bytes", 0) or 0),
                    reverse=True,
                )[:12]
                if isinstance(item, dict)
            ],
        },
        "sqlite": sqlite,
        "mainline": {
            "module_count": int(mainline.get("module_count", 0) or 0),
            "reachable_module_count": int(mainline.get("reachable_module_count", 0) or 0),
            "parse_error_count": int(mainline.get("parse_error_count", 0) or 0),
            "feature_summary": mainline.get("feature_summary", {}),
        },
        "functional": {
            "mechanism_count": int(functional.get("mechanism_count", 0) or 0),
            "status_counts": functional.get("status_counts", {}),
        },
        "capability": {
            "active_fact_count": int(capability.get("active_fact_count", 0) or 0),
            "generation_registered_count": int(
                capability.get("generation_registered_count", 0) or 0),
            "route_count": route_count,
            "route_count_closed": route_closed,
        },
        "historical_performance": (
            generation_correction.get("historical_performance_constraints", {})
            if isinstance(generation_correction, dict) else {}
        ),
        "imports": production_imports,
        "direction": {
            "status": status,
            "training_start_allowed": int(not blockers),
            "blockers": blockers,
            "next_slice": (
                "integrate_work_memory_discourse_into_default_query_state"
                if "DISCOURSE_TOPIC_NOT_ON_PRODUCTION_QUERYSTATE_PATH" in blockers
                else "continue_bounded_language_structure_course"
            ),
            "storage_policy": (
                "freeze_current_lineage_and_archive_historical_duplicates_to_F"
                if duplicate_bytes else "retain_single_canonical_training_lineage"
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-report", type=Path, required=True)
    parser.add_argument("--sqlite-report", type=Path, required=True)
    parser.add_argument("--mainline-report", type=Path, required=True)
    parser.add_argument("--functional-report", type=Path, required=True)
    parser.add_argument("--capability-report", type=Path, required=True)
    parser.add_argument("--generation-correction-report", type=Path)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    source_root = args.source_root.resolve(strict=True)
    if not source_root.is_dir():
        raise ValueError("source-root must be a directory")
    report = build_report(
        storage=_read(args.storage_report),
        sqlite=_sqlite_summary(_read(args.sqlite_report)),
        mainline=_read(args.mainline_report),
        functional=_read(args.functional_report),
        capability=_read(args.capability_report),
        source_root=source_root,
        generation_correction=(
            _read(args.generation_correction_report)
            if args.generation_correction_report is not None else None
        ),
    )
    target = args.output.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="ascii",
        newline="\n",
    )
    print(json.dumps(report["direction"], ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_report", "main"]
