"""Audit historical mechanisms against the current dialogue contracts.

The mechanism inventory is intentionally descriptive.  This read-only audit
cross-checks its writers/readers with the import reachability report and with
the executable contract markers observed in each module.  A facility is not
called integrated merely because its module imports or its tests exist.
"""
from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path

from pure_integer_ai.experiments.audit_mainline_integration import (
    ENTRY_SCOPES,
    audit,
)
from pure_integer_ai.experiments.mechanism_inventory import inventory_json


CONTRACT_KEYS = (
    "query_state", "binding", "evidence", "visited", "termination",
    "response_plan",
)


def _module_ref(value: object) -> str | None:
    """Normalize an inventory writer/reader reference to a package module."""
    if not isinstance(value, str) or not value.strip():
        return None
    module = value.split(":", 1)[0].strip()
    if module.startswith("pure_integer_ai."):
        return module
    return "pure_integer_ai." + module


def _module_map(report: dict[str, object]) -> dict[str, dict[str, object]]:
    return {
        item["module"]: item
        for item in report["modules"]
        if isinstance(item, dict) and isinstance(item.get("module"), str)
    }


def _shortest_path(
        graph: dict[str, tuple[str, ...]],
        roots: tuple[str, ...],
        targets: tuple[str, ...],
        ) -> tuple[str, ...]:
    """Return one deterministic import path from a scope root to a reader."""
    wanted = set(targets)
    pending = deque(
        (root, (root,)) for root in sorted(roots) if root in graph
    )
    visited = set()
    while pending:
        module, path = pending.popleft()
        if module in visited:
            continue
        visited.add(module)
        if module in wanted:
            return path
        for imported in graph.get(module, ()):
            if imported not in visited:
                pending.append((imported, (*path, imported)))
    return ()


def audit_functional_contracts(source_root: str | Path) -> dict[str, object]:
    """Return a conservative four-stage integration report.

    The stages are intentionally independent: a training writer does not
    imply a graph contract, and a production reader does not imply generation
    eligibility.  ``closed`` therefore requires both a training-side writer
    and a production reader, executable QueryState/evidence/visited/
    termination markers, and an executable ResponsePlan marker.
    """
    reachability = audit(source_root)
    modules = _module_map(reachability)
    import_graph = {
        module: tuple(sorted(
            item for item in row.get("imports", ()) if item in modules
        ))
        for module, row in modules.items()
    }
    rows: list[dict[str, object]] = []
    for record in inventory_json():
        writers = tuple(sorted({
            module for module in (_module_ref(item) for item in record["writers"])
            if module is not None
        }))
        readers = tuple(sorted({
            module for module in (_module_ref(item) for item in record["readers"])
            if module is not None
        }))
        writer_rows = tuple(modules[item] for item in writers if item in modules)
        reader_rows = tuple(modules[item] for item in readers if item in modules)
        training_producers = tuple(sorted(
            item["module"] for item in writer_rows
            if "training" in item.get("reachable_scopes", ())
        ))
        production_readers = tuple(sorted(
            item["module"] for item in reader_rows
            if item.get("production_reachable")
        ))
        production_eager_readers = tuple(sorted(
            item["module"] for item in reader_rows
            if item.get("production_eager_reachable")
        ))
        query_readers = tuple(sorted(
            item["module"] for item in reader_rows
            if "query" in item.get("eager_reachable_scopes", ())
        ))
        scoped_readers = {
            scope: tuple(sorted(
                item["module"] for item in reader_rows
                if scope in item.get("reachable_scopes", ())
            ))
            for scope in ENTRY_SCOPES
        }
        scoped_writers = {
            scope: tuple(sorted(
                item["module"] for item in writer_rows
                if scope in item.get("reachable_scopes", ())
            ))
            for scope in ENTRY_SCOPES
        }
        route_paths = {
            scope: _shortest_path(
                import_graph, tuple(ENTRY_SCOPES[scope]), readers)
            for scope, readers in scoped_readers.items()
            if readers
        }
        route_observed_contracts = {
            scope: {
                name: tuple(
                    module for module in path
                    if name in modules[module].get(
                        "contracts_observed", ()))
                for name in CONTRACT_KEYS
            }
            for scope, path in route_paths.items()
        }
        route_missing_contracts = {
            scope: tuple(
                name for name in CONTRACT_KEYS
                if not observed[name])
            for scope, observed in route_observed_contracts.items()
        }
        observed_contracts = {
            name: tuple(sorted({
                item["module"] for item in (*writer_rows, *reader_rows)
                if name in item.get("contracts_observed", ())
            }))
            for name in CONTRACT_KEYS
        }
        missing = tuple(name for name in CONTRACT_KEYS
                        if not observed_contracts[name])
        has_training = bool(training_producers)
        has_production = bool(production_readers)
        has_query = bool(query_readers)
        has_core_contract = all(
            route_observed_contracts.get("query", {}).get(name)
            for name in (
                "query_state", "binding", "evidence", "visited",
                "termination"))
        has_generation_contract = any(
            route_observed_contracts.get(scope, {}).get("response_plan")
            for scope in ("generation", "terminal", "release"))
        inventory_status = str(record.get("status", ""))
        if inventory_status.casefold() == "dead" and not writers and not readers:
            # A deliberately retired legacy ID is not an unwired production
            # gap.  Its replacement must be audited through the active graph
            # contracts above, while the old route remains unreachable.
            status = "declared_dead"
        elif has_training and has_production and has_core_contract and has_generation_contract:
            status = "closed"
        elif has_production and has_query:
            status = "production_partial"
        elif has_training:
            status = "training_only"
        elif writers or readers:
            status = "declared_unreachable"
        else:
            status = "unwired"
        has_dialogue_reader = bool(
            scoped_readers["query"] or scoped_readers["generation"])
        has_terminal_reader = bool(scoped_readers["terminal"])
        if has_dialogue_reader and training_producers:
            route_status = "dialogue_mainline"
        elif has_dialogue_reader:
            route_status = "dialogue_read_only"
        elif has_terminal_reader or scoped_readers["release"]:
            route_status = "production_support"
        elif training_producers:
            route_status = "training_only"
        elif writers or readers:
            route_status = "isolated_declared"
        else:
            route_status = "unwired"
        if status == "closed":
            integration_disposition = "mainline_closed"
        elif status == "declared_dead":
            integration_disposition = "retired"
        elif (record["readiness_eligible"]
              and route_status in {"training_only", "isolated_declared"}):
            integration_disposition = "review_required"
        elif route_status == "dialogue_read_only":
            integration_disposition = "read_only_candidate"
        elif route_status == "training_only":
            integration_disposition = "training_only_by_contract"
        else:
            integration_disposition = "partial_route"
        rows.append({
            "mechanism_id": record["mechanism_id"],
            "inventory_status": record["status"],
            "readiness_eligible": int(bool(record["readiness_eligible"])),
            "writers": list(writers),
            "readers": list(readers),
            "training_producers": list(training_producers),
            "production_readers": list(production_readers),
            "production_eager_readers": list(production_eager_readers),
            "query_readers": list(query_readers),
            "scoped_readers": {
                scope: list(values)
                for scope, values in scoped_readers.items()
            },
            "scoped_writers": {
                scope: list(values)
                for scope, values in scoped_writers.items()
            },
            "route_paths": {
                scope: list(path) for scope, path in route_paths.items()
            },
            "route_status": route_status,
            "route_observed_contracts": {
                scope: {
                    name: list(values)
                    for name, values in observed.items()
                }
                for scope, observed in route_observed_contracts.items()
            },
            "route_missing_contracts": {
                scope: list(values)
                for scope, values in route_missing_contracts.items()
            },
            "route_core_contract_closed": int(has_core_contract),
            "route_generation_contract_closed": int(
                has_generation_contract),
            "integration_disposition": integration_disposition,
            "observed_contracts": {
                name: list(values) for name, values in observed_contracts.items()
            },
            "missing_contracts": list(missing),
            "status": status,
            "limitation": record["limitation"],
        })
    counts: dict[str, int] = {}
    route_counts: dict[str, int] = {}
    disposition_counts: dict[str, int] = {}
    for row in rows:
        status = row["status"]
        counts[status] = counts.get(status, 0) + 1
        route_status = row["route_status"]
        route_counts[route_status] = route_counts.get(route_status, 0) + 1
        disposition = row["integration_disposition"]
        disposition_counts[disposition] = (
            disposition_counts.get(disposition, 0) + 1)
    return {
        "schema_version": 1,
        "protocol": "PURE_INTEGER_AI_FUNCTIONAL_CONTRACT_AUDIT_V1",
        "source_root": str(Path(source_root).resolve()),
        "mechanism_count": len(rows),
        "status_counts": counts,
        "route_status_counts": route_counts,
        "integration_disposition_counts": disposition_counts,
        "contract_keys": list(CONTRACT_KEYS),
        "reachability_summary": {
            "module_count": reachability["module_count"],
            "reachable_module_count": reachability["reachable_module_count"],
            "eager_reachable_scope_counts": reachability["eager_reachable_scope_counts"],
        },
        "mechanisms": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root", type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_functional_contracts(args.source_root)
    target = args.output.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, ensure_ascii=True, sort_keys=True,
                   separators=(",", ":")),
        encoding="ascii",
    )
    print(json.dumps({
        "mechanism_count": report["mechanism_count"],
        "status_counts": report["status_counts"],
        "route_status_counts": report["route_status_counts"],
    }, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["audit_functional_contracts", "main"]
