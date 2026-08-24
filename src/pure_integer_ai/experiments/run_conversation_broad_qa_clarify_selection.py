"""运行真实 broad-QA 多候选显式选择切片。"""
from __future__ import annotations

import argparse
import json

from pure_integer_ai.experiments.conversation_broad_qa_clarify_selection import (
    build_clarify_selection_report,
    write_clarify_selection_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="real broad QA clarify selection")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--pack-dir", required=True)
    parser.add_argument("--database", required=True)
    parser.add_argument("--training-run-root", required=True)
    parser.add_argument("--item-id", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = build_clarify_selection_report(
        project_root=args.project_root,
        pack_dir=args.pack_dir,
        database_path=args.database,
        training_run_root=args.training_run_root,
        item_id=args.item_id,
    )
    output = write_clarify_selection_report(report, args.output)
    print(json.dumps({
        "output": output,
        "status": report["status"],
        "selection_contract": report["selection_contract"],
        "baseline_candidate_count": report["baseline_candidate_count"],
        "replay_bit_identical": report["replay_bit_identical"],
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
