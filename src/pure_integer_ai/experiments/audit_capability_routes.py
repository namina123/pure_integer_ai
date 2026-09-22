"""Read-only audit for the active relation capability route contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pure_integer_ai.experiments.trained_graph_query_bridge import (
    TrainedGraphQueryBridge,
)


def audit(database: str | Path) -> dict[str, object]:
    path = Path(database).resolve(strict=True)
    with TrainedGraphQueryBridge(path) as bridge:
        routes = bridge.relation_routes.all()
        return {
            "protocol": "PURE_INTEGER_AI_RELATION_CAPABILITY_ROUTE_AUDIT_V1",
            "read_only": 1,
            "database": str(path),
            "active_fact_count": len(bridge.facts),
            "route_count": len(routes),
            # This field is a count, not a boolean.  The previous boolean
            # encoding made a complete 17-route graph look like 1/17 to the
            # aggregate direction gate.
            "route_count_closed": sum(
                item.generation_registered for item in routes),
            "route_registry_closed": int(len(routes) == len(bridge.facts)),
            "generation_registered_count": sum(
                item.generation_registered for item in routes),
            "routes": [item.trace() for item in routes],
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = audit(args.database)
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(
        json.dumps(report, ensure_ascii=True, sort_keys=True,
                   separators=(",", ":")),
        encoding="ascii",
    )
    print(json.dumps({
        key: report[key] for key in (
            "active_fact_count", "route_count", "route_count_closed",
            "generation_registered_count",
        )
    }, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["audit", "main"]
