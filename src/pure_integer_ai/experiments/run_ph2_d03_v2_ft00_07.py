"""Run and publish the PH2-D03-V2 FT00-07 public release gate."""
from __future__ import annotations

from pathlib import Path

from pure_integer_ai.experiments.ph2_d03_v2_ft00_release import (
    FT00_RELEASE_GATE_PATH,
    run_ft00_release_gate,
    write_ft00_release_gate,
)


def main() -> int:
    repository = Path(__file__).resolve().parents[3]
    report = run_ft00_release_gate(repository)
    target = repository / Path(*FT00_RELEASE_GATE_PATH.split("/"))
    write_ft00_release_gate(report, target)
    print(f"path={target}")
    print(f"report_sha256={report.sha256()}")
    print(f"checks={len(report.checks)}/{len(report.check_order)}")
    print(f"status={report.status}")
    print("formal_training_runs=0")
    print("formal_private_evaluation_runs=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
