"""运行公开 100 问广域对话规模审计。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pure_integer_ai.experiments.conversation_broad_qa_scale_audit import (
    build_conversation_broad_qa_scale_audit,
    write_conversation_broad_qa_scale_audit,
)
from pure_integer_ai.experiments.ph2_broad_qa_relation_evidence_learning import (
    learn_relation_evidence_model,
)
from pure_integer_ai.experiments.ph2_broad_qa_relation_role_evidence_learning import (
    learn_relation_role_evidence_model,
)
from pure_integer_ai.experiments.ph2_broad_qa_relation_marker_evidence_learning import (
    learn_relation_marker_evidence_model,
)
from pure_integer_ai.experiments.ph2_broad_qa_relation_answer_frame_learning import (
    learn_relation_answer_frame_model,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit 100-question dialogue scale")
    parser.add_argument("--pack-dir", required=True)
    parser.add_argument("--database", required=True)
    parser.add_argument("--extra-relation-evidence-course", action="append", default=[])
    parser.add_argument("--extra-relation-role-evidence-course", action="append", default=[])
    parser.add_argument("--extra-relation-marker-evidence-course", action="append", default=[])
    parser.add_argument("--extra-relation-answer-frame-course", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    value = build_conversation_broad_qa_scale_audit(
        pack_dir=args.pack_dir, database_path=args.database,
        learned_relation_evidence_model=(
            learn_relation_evidence_model(tuple(Path(item) for item in
                                              args.extra_relation_evidence_course))
            if args.extra_relation_evidence_course else None),
        learned_relation_role_evidence_model=(
            learn_relation_role_evidence_model(tuple(Path(item) for item in
                                                   args.extra_relation_role_evidence_course))
            if args.extra_relation_role_evidence_course else None),
        learned_relation_marker_evidence_model=(
            learn_relation_marker_evidence_model(tuple(Path(item) for item in
                                                      args.extra_relation_marker_evidence_course))
            if args.extra_relation_marker_evidence_course else None),
        learned_relation_answer_frame_model=(
            learn_relation_answer_frame_model(tuple(Path(item) for item in
                                               args.extra_relation_answer_frame_course))
            if args.extra_relation_answer_frame_course else None),
    )
    output = write_conversation_broad_qa_scale_audit(value, args.output)
    print(json.dumps({
        "output": output,
        "question_count": value["question_count"],
        "status_counts": value["status_counts"],
        "readable_surface_count": value["readable_surface_count"],
        "complete_sentence_count": value["complete_sentence_count"],
        "long_surface_count": value["long_surface_count"],
        "surface_gold_hit_count": value["surface_gold_hit_count"],
        "evidence_gold_hit_count": value["evidence_gold_hit_count"],
        "failure_counts": value["failure_counts"],
        "replay_bit_identical": value["replay_bit_identical"],
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
