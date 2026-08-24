"""运行独立表层 held-out 开发评估并写入 K 盘摘要。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pure_integer_ai.experiments.conversation_trained_surface_heldout import (
    run_trained_surface_heldout,
    write_heldout_surface_report,
)
from pure_integer_ai.experiments.conversation_trained_surface_runtime import (
    load_trained_surface_runtime,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run independent trained-surface held-out evaluation")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--training-run-root", required=True)
    parser.add_argument("--pack-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    root = Path(args.project_root).resolve()
    runtime = load_trained_surface_runtime(
        project_root=root,
        training_run_root=args.training_run_root,
        expected_pack_sha256=args.pack_sha256,
    )
    report = run_trained_surface_heldout(runtime, root)
    output = write_heldout_surface_report(report, args.output)
    print(json.dumps({
        "output": output,
        "status": report.status,
        "total_cases": report.total_cases,
        "passed_cases": report.passed_cases,
        "failed_cases": report.failed_cases,
        "long_cases": report.long_cases,
        "baseline_no_consumer_cases": report.baseline_no_consumer_cases,
        "ready": bool(report.ready),
        "run_id": report.run_id,
        "pack_sha256": report.pack_sha256,
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
