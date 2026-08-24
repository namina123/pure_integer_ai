"""运行公开 24 问广域问答主证据审计。"""
from __future__ import annotations

import argparse
import json

from pure_integer_ai.experiments.broad_qa_dev_surface_audit import (
    build_broad_qa_dev_audit,
    write_broad_qa_dev_audit,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit broad QA primary surfaces")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--database", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = build_broad_qa_dev_audit(
        project_root=args.project_root, database_path=args.database)
    output = write_broad_qa_dev_audit(report, args.output)
    print(json.dumps({
        "output": output,
        "status": report.status,
        "question_count": report.question_count,
        "answer_count": report.answer_count,
        "unknown_count": report.unknown_count,
        "clarify_count": report.clarify_count,
        "evidence_expected_count": report.evidence_expected_count,
        "evidence_hit_count": report.evidence_hit_count,
        "primary_surface_clean_count": report.primary_surface_clean_count,
        "long_answer_count": report.long_answer_count,
        "replay_bit_identical": report.replay_bit_identical,
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
