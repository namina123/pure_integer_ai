"""运行公开 100 问训练前/后对照。"""
from __future__ import annotations

import argparse
import json

from pure_integer_ai.experiments.conversation_broad_qa_training_contrast import (
    build_conversation_broad_qa_training_contrast,
    write_conversation_broad_qa_training_contrast,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="100-question training contrast")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--pack-dir", required=True)
    parser.add_argument("--database", required=True)
    parser.add_argument("--training-run-root", required=True)
    parser.add_argument("--item-id", action="append", default=[],
                        help="可选小切片 item_id；不提供时运行完整 dev pack")
    parser.add_argument("--extra-training-course", action="append", default=[])
    parser.add_argument("--extra-obligation-course", action="append", default=[])
    parser.add_argument("--extra-relation-evidence-course", action="append", default=[])
    parser.add_argument("--extra-relation-role-evidence-course", action="append", default=[])
    parser.add_argument("--extra-relation-marker-evidence-course", action="append", default=[])
    parser.add_argument("--extra-relation-answer-frame-course", action="append", default=[])
    parser.add_argument("--extra-variant-course", action="append", default=[])
    parser.add_argument("--extra-variant-evidence", action="append", default=[])
    parser.add_argument("--extra-order-course", action="append", default=[])
    parser.add_argument("--extra-order-evidence", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    value = build_conversation_broad_qa_training_contrast(
        project_root=args.project_root,
        pack_dir=args.pack_dir,
        database_path=args.database,
        training_run_root=args.training_run_root,
        item_ids=tuple(args.item_id),
        extra_training_course_paths=tuple(args.extra_training_course),
        extra_obligation_course_paths=tuple(args.extra_obligation_course),
        extra_relation_evidence_course_paths=tuple(
            args.extra_relation_evidence_course),
        extra_relation_role_evidence_course_paths=tuple(
            args.extra_relation_role_evidence_course),
        extra_relation_marker_evidence_course_paths=tuple(
            args.extra_relation_marker_evidence_course),
        extra_relation_answer_frame_course_paths=tuple(
            args.extra_relation_answer_frame_course),
        extra_variant_course_paths=tuple(args.extra_variant_course),
        extra_variant_evidence_paths=tuple(args.extra_variant_evidence),
        extra_order_course_paths=tuple(args.extra_order_course),
        extra_order_evidence_paths=tuple(args.extra_order_evidence),
    )
    output = write_conversation_broad_qa_training_contrast(value, args.output)
    print(json.dumps({
        "output": output,
        "baseline": value["baseline"],
        "trained": value["trained"],
        "delta": value["delta"],
        "display_changed_count": value["display_changed_count"],
        "trained_surface_consumer_used_count": (
            value["trained_surface_consumer_used_count"]),
        "replay_bit_identical": value["replay_bit_identical"],
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
