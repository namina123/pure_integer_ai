"""执行公开对话规模展示并把摘要写入 K 盘。"""
from __future__ import annotations

import argparse
import json

from pure_integer_ai.experiments.conversation_dialogue_scale_showcase import (
    build_dialogue_scale_showcase,
    write_dialogue_scale_showcase,
)


def main(argv: list[str] | None = None) -> int:
    """运行一次可复跑展示；不写入训练数据库。"""
    parser = argparse.ArgumentParser(description="Dialogue scale showcase")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--database", required=True)
    parser.add_argument("--training-run-root", default=None)
    parser.add_argument("--extra-training-course", action="append", default=[],
                        help="可选公开训练课程；不改变默认 v6 pack")
    parser.add_argument("--extra-variant-course", action="append", default=[],
                        help="可选表层变体课程；必须和 evidence 一一对应")
    parser.add_argument("--extra-variant-evidence", action="append", default=[],
                        help="可选表层变体 evidence；必须和 course 一一对应")
    parser.add_argument("--variant-probe-input", action="append", default=[],
                        help="显式 G7 表层变体 probe 输入；不读取答案标签")
    parser.add_argument("--extra-order-course", action="append", default=[],
                        help="可选 G9 语序课程；必须和 evidence 一一对应")
    parser.add_argument("--extra-order-evidence", action="append", default=[],
                        help="可选 G9 语序 evidence；必须和 course 一一对应")
    parser.add_argument("--order-probe-value", action="append", default=[],
                        help="显式 G9 typed slot value，按 --order-probe-role 对齐")
    parser.add_argument("--order-probe-role", action="append", default=[],
                        help="显式 G9 typed role，按 --order-probe-value 对齐")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    value = build_dialogue_scale_showcase(
        project_root=args.project_root,
        database_path=args.database,
        training_run_root=args.training_run_root,
        extra_training_course_paths=tuple(args.extra_training_course),
        extra_variant_course_paths=tuple(args.extra_variant_course),
        extra_variant_evidence_paths=tuple(args.extra_variant_evidence),
        variant_probe_inputs=tuple(args.variant_probe_input),
        extra_order_course_paths=tuple(args.extra_order_course),
        extra_order_evidence_paths=tuple(args.extra_order_evidence),
        order_probe_values=tuple(args.order_probe_value),
        order_probe_roles=tuple(args.order_probe_role),
    )
    path = write_dialogue_scale_showcase(value, args.output)
    print(json.dumps({
        "output": path,
        "question_count": value["question_count"],
        "long_question_count": value["long_question_count"],
        "status_counts": value["status_counts"],
        "source_bound_answer_count": value["source_bound_answer_count"],
        "replay_bit_identical": value["replay_bit_identical"],
        "turns_sha256": value["turns_sha256"],
        "pack_sha256": value["pack"]["pack_sha256"],
        "training_run_id": (
            None if value["training_observation"] is None
            else value["training_observation"]["run_id"]),
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
