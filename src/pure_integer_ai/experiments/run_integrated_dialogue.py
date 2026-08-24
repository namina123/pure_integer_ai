"""公开完整句窄域 + Wikipedia 广域来源约束问答的交互入口。"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

from pure_integer_ai.experiments.conversation_broad_qa_runtime import (
    BroadDialogueState,
    answer_broad_dialogue_turn,
)
from pure_integer_ai.experiments.conversation_public_sentence_demo import (
    build_public_sentence_demo_catalog,
    run_public_sentence_demo_bytes,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_sparse_qa_snapshot import (
    load_or_rebuild_public_sparse_qa_runtime,
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
def _render(turn) -> str:
    if turn.status == "ANSWER" and turn.answer is not None:
        text = turn.display_answer or turn.answer
        if turn.source_title is not None:
            text += f"\n来源：{turn.source_title}"
        if turn.source_url is not None:
            text += f"\n{turn.source_url}"
        return text
    if turn.status == "CLARIFY":
        return "存在多个接近候选，请补充更明确的实体或限定。"
    if turn.status == "CONFLICT":
        return "来源之间存在冲突，暂不直接回答。"
    return "暂未找到有来源约束的回答。"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Integrated public dialogue")
    parser.add_argument("--database", required=True)
    parser.add_argument("--training-run-root", default=None)
    parser.add_argument("--extra-relation-evidence-course", action="append",
                        default=[], help="opt-in 公开关系证据课程")
    parser.add_argument("--extra-relation-role-evidence-course", action="append",
                        default=[], help="opt-in 公开 role/span 课程")
    parser.add_argument("--extra-relation-marker-evidence-course", action="append",
                        default=[], help="opt-in 公开 marker/value 课程")
    parser.add_argument("--extra-relation-answer-frame-course", action="append",
                        default=[], help="opt-in 公开回答句面课程")
    parser.add_argument("question", nargs="?")
    args = parser.parse_args(argv)
    connection = sqlite3.connect(f"file:{args.database}?mode=ro", uri=True)
    state = BroadDialogueState((1, 1, 1))
    narrow_runtime = load_or_rebuild_public_sparse_qa_runtime()
    narrow_catalog = build_public_sentence_demo_catalog(narrow_runtime)
    trained_surface = None
    if args.training_run_root is not None:
        from pure_integer_ai.experiments.conversation_dialogue_scale_showcase import (
            load_dialogue_training_pack,
        )
        from pure_integer_ai.experiments.run_conversation_training import (
            default_course_paths,
        )
        from pure_integer_ai.experiments.conversation_trained_surface_runtime import (
            load_trained_surface_runtime,
        )
        pack = load_dialogue_training_pack(default_course_paths("."))
        trained_surface = load_trained_surface_runtime(
            project_root=".", training_run_root=args.training_run_root,
            expected_pack_sha256=pack.pack_sha256,
        )

    relation_evidence_model = (
        learn_relation_evidence_model(tuple(
            Path(item).resolve() for item in args.extra_relation_evidence_course))
        if args.extra_relation_evidence_course else None)
    relation_role_evidence_model = (
        learn_relation_role_evidence_model(tuple(
            Path(item).resolve()
            for item in args.extra_relation_role_evidence_course))
        if args.extra_relation_role_evidence_course else None)
    relation_marker_evidence_model = (
        learn_relation_marker_evidence_model(tuple(
            Path(item).resolve()
            for item in args.extra_relation_marker_evidence_course))
        if args.extra_relation_marker_evidence_course else None)
    relation_answer_frame_model = (
        learn_relation_answer_frame_model(tuple(
            Path(item).resolve()
            for item in args.extra_relation_answer_frame_course))
        if args.extra_relation_answer_frame_course else None)

    def narrow_answer(question: str) -> tuple[str, str] | None:
        result = run_public_sentence_demo_bytes(
            narrow_runtime, narrow_catalog, question.encode("utf-8"))
        if result.generated_proposition_surface is None:
            return None
        surface = result.generated_proposition_surface
        if trained_surface is not None:
            rendered = trained_surface.render(surface, response_act="ANSWER")
            if rendered.used:
                surface = rendered.surface
        return surface, "ANSWER"

    try:
        questions = [args.question] if args.question is not None else None
        if questions is None:
            for raw in sys.stdin:
                value = raw.rstrip("\r\n")
                if value in {":quit", ":exit"}:
                    break
                if not value.strip():
                    continue
                state, turn = answer_broad_dialogue_turn(
                    state, value, connection, narrow_answer=narrow_answer,
                    learned_relation_evidence_model=relation_evidence_model,
                    learned_relation_role_evidence_model=(
                        relation_role_evidence_model),
                    learned_relation_marker_evidence_model=(
                        relation_marker_evidence_model),
                    learned_relation_answer_frame_model=(
                        relation_answer_frame_model))
                print(f"系统> {_render(turn)}", flush=True)
        else:
            state, turn = answer_broad_dialogue_turn(
                state, questions[0], connection, narrow_answer=narrow_answer,
                learned_relation_evidence_model=relation_evidence_model,
                learned_relation_role_evidence_model=(
                    relation_role_evidence_model),
                learned_relation_marker_evidence_model=(
                    relation_marker_evidence_model),
                learned_relation_answer_frame_model=(
                    relation_answer_frame_model))
            print(_render(turn))
    finally:
        connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
