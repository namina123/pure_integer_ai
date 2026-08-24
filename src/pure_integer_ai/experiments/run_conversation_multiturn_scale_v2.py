"""运行扩大来源覆盖的多轮问答开发切片。"""
from __future__ import annotations

import argparse
import json

from pure_integer_ai.experiments.conversation_multiturn_scale_v2 import (
    build_expanded_multiturn_report,
    write_expanded_multiturn_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run expanded multiturn dialogue slice")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--database", required=True)
    parser.add_argument("--training-run-root", required=True)
    parser.add_argument("--pack-sha256", required=True)
    parser.add_argument("--extra-training-course", action="append", default=[])
    parser.add_argument("--extra-variant-course", action="append", default=[])
    parser.add_argument("--extra-variant-evidence", action="append", default=[])
    parser.add_argument("--extra-order-course", action="append", default=[])
    parser.add_argument("--extra-order-evidence", action="append", default=[])
    parser.add_argument("--variant-probe-input", action="append", default=[])
    parser.add_argument("--order-probe-value", action="append", default=[])
    parser.add_argument("--order-probe-role", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = build_expanded_multiturn_report(
        project_root=args.project_root,
        database_path=args.database,
        training_run_root=args.training_run_root,
        expected_pack_sha256=args.pack_sha256,
        extra_training_course_paths=tuple(args.extra_training_course),
        extra_variant_course_paths=tuple(args.extra_variant_course),
        extra_variant_evidence_paths=tuple(args.extra_variant_evidence),
        extra_order_course_paths=tuple(args.extra_order_course),
        extra_order_evidence_paths=tuple(args.extra_order_evidence),
        variant_probe_inputs=tuple(args.variant_probe_input),
        order_probe_values=tuple(args.order_probe_value),
        order_probe_roles=tuple(args.order_probe_role),
    )
    output = write_expanded_multiturn_report(report, args.output)
    print(json.dumps({
        "output": output,
        "status": report.status,
        "scenario_count": report.scenario_count,
        "question_count": report.question_count,
        "answer_count": report.answer_count,
        "unknown_count": report.unknown_count,
        "clarify_count": report.clarify_count,
        "long_answer_count": report.long_answer_count,
        "evidence_expected_count": report.evidence_expected_count,
        "evidence_hit_count": report.evidence_hit_count,
        "trained_surface_used_count": report.trained_surface_used_count,
        "focus_injection_count": report.focus_injection_count,
        "focus_not_crossed_unknown": bool(report.focus_not_crossed_unknown),
        "replay_bit_identical": report.replay_bit_identical,
        "turns_sha256": report.turns_sha256,
        "surface_variant_probe_used_count": report.surface_variant_probe_used_count,
        "surface_order_probe_used_count": report.surface_order_probe_used_count,
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
