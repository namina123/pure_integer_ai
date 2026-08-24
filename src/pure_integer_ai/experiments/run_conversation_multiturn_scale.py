"""运行公开多轮对话开发切片并写入 K 盘摘要。"""
from __future__ import annotations

import argparse
import json

from pure_integer_ai.experiments.conversation_multiturn_scale import (
    build_multiturn_scale_report,
    write_multiturn_scale_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run public multiturn dialogue slice")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--database", required=True)
    parser.add_argument("--training-run-root", required=True)
    parser.add_argument("--pack-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = build_multiturn_scale_report(
        project_root=args.project_root,
        database_path=args.database,
        training_run_root=args.training_run_root,
        expected_pack_sha256=args.pack_sha256,
    )
    output = write_multiturn_scale_report(report, args.output)
    print(json.dumps({
        "output": output,
        "status": report.status,
        "question_count": report.question_count,
        "answer_count": report.answer_count,
        "unknown_count": report.unknown_count,
        "clarify_count": report.clarify_count,
        "long_answer_count": report.long_answer_count,
        "trained_surface_used_count": report.trained_surface_used_count,
        "focus_injection_count": report.focus_injection_count,
        "focus_not_crossed_unknown": bool(report.focus_not_crossed_unknown),
        "replay_bit_identical": report.replay_bit_identical,
        "turns_sha256": report.turns_sha256,
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
