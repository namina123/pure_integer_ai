"""Publish the payload-free PH2-D03-V2 FT00-06 boundary manifest."""
from __future__ import annotations

from pathlib import Path
import sys

from pure_integer_ai.experiments.ph2_d03_v2_evaluator_contract import (
    publish_v2_evaluator_boundary_contract,
    read_v2_evaluator_boundary_contract,
)


def main() -> int:
    repository = Path(__file__).resolve().parents[3]
    target = publish_v2_evaluator_boundary_contract(repository)
    contract = read_v2_evaluator_boundary_contract(repository, target)
    print(f"path={target}")
    print(f"boundary_sha256={contract.sha256()}")
    print("formal_private_evaluation_runs=0")
    print("private_payload_reads=0")
    print("status=EVALUATOR_BOUNDARY_FROZEN")
    return 0


if __name__ == "__main__":
    sys.exit(main())
