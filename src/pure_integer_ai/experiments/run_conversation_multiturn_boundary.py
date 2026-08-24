"""运行真实多轮焦点边界切片。"""
from __future__ import annotations

import argparse
import json

from pure_integer_ai.experiments.conversation_multiturn_boundary import (
    build_multiturn_boundary_report,
    write_multiturn_boundary_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="real multi-turn boundary slice")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--pack-dir", required=True)
    parser.add_argument("--database", required=True)
    parser.add_argument("--training-run-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = build_multiturn_boundary_report(
        project_root=args.project_root,
        pack_dir=args.pack_dir,
        database_path=args.database,
        training_run_root=args.training_run_root,
    )
    output = write_multiturn_boundary_report(report, args.output)
    print(json.dumps({
        "output": output,
        "status": report["status"],
        "focus_contract": report["focus_contract"],
        "hot_history_lengths": report["hot_history_lengths"],
        "replay_bit_identical": report["replay_bit_identical"],
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
