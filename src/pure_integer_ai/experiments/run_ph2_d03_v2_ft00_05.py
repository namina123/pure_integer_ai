"""Run and immutably publish the public FT00-05 P0/P1 baseline."""
from __future__ import annotations

import argparse
from pathlib import Path

from pure_integer_ai.experiments.ph2_d03_v2_scale_baseline import (
    run_ft00_05_scale_baseline,
    write_ft00_05_report,
)


DEFAULT_REPORT = (
    "data/ph2/manifests/d03_v2/"
    "ph2_d03_v2_ft00_05_scale_baseline_v1.json"
)


def main() -> int:
    """Run the exact two-point workload and print its canonical identity."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", default=".")
    parser.add_argument("--output", default=DEFAULT_REPORT)
    arguments = parser.parse_args()
    root = Path(arguments.repository_root).resolve()
    output = Path(arguments.output)
    if not output.is_absolute():
        output = root / output
    report = run_ft00_05_scale_baseline(root)
    write_ft00_05_report(report, output)
    print(f"status={report.status}")
    print(f"report_sha256={report.sha256()}")
    for point in report.points:
        print(
            f"{point.scale_key}={point.target_records}:"
            f"db={point.database_bytes}:query={point.query_rows}:"
            f"resume={int(point.fresh_digest == point.resume_digest)}"
        )
    return 0 if report.status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
