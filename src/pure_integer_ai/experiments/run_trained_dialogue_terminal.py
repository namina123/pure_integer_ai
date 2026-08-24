"""K 盘训练状态驱动的公开交互终端。

这是面向真实训练结果的轻量 host 入口：QA SQLite 以只读方式打开，训练 run
只读加载，回答仍由既有窄域/广域查询和表层组织 runtime 产生。Core、Runtime
和训练账本不会因交互写入；会话热区仅存在当前进程，可随进程结束丢弃。
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sqlite3
import sys
from typing import BinaryIO

from pure_integer_ai.experiments.conversation_broad_qa_runtime import (
    BroadDialogueState,
    answer_broad_dialogue_turn,
)
from pure_integer_ai.experiments.conversation_dialogue_scale_showcase import (
    load_training_observation,
)
from pure_integer_ai.experiments.conversation_public_sentence_demo import (
    build_public_sentence_demo_catalog,
    run_public_sentence_demo_bytes,
)
from pure_integer_ai.experiments.conversation_training_pack import (
    load_dialogue_training_pack,
)
from pure_integer_ai.experiments.conversation_trained_surface_runtime import (
    TrainedSurfaceRuntime,
    load_trained_surface_runtime,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_sparse_qa_snapshot import (
    load_or_rebuild_public_sparse_qa_runtime,
)
from pure_integer_ai.experiments.run_conversation_training import (
    default_course_paths,
)


def _require_k_file(value: str | Path, *, label: str) -> Path:
    """核验只读外部索引位于 K 盘且确实存在。"""
    path = Path(value).resolve()
    if path.drive.upper() != "K:" or not path.is_file():
        raise ValueError(f"{label} 必须是 K 盘已存在文件")
    return path


def _narrow_answer(runtime, catalog, trained_surface: TrainedSurfaceRuntime | None):
    """建立只读窄域消费者；未命中交给广域来源查询。"""
    def answer(question: str) -> tuple[str, str] | None:
        result = run_public_sentence_demo_bytes(
            runtime, catalog, question.encode("utf-8"))
        if result.generated_proposition_surface is None:
            return None
        surface = result.generated_proposition_surface
        if trained_surface is not None:
            rendered = trained_surface.render(surface, response_act="ANSWER")
            if rendered.used:
                surface = rendered.surface
        return surface, "ANSWER"
    return answer


def _display(turn) -> str:
    """将既有 turn 的答案载荷投影为人可读的一行，不改动 turn 语义。"""
    if turn.display_answer:
        answer = turn.display_answer
    elif turn.status == "CLARIFY":
        answer = "请补充问题的范围或限定条件。"
    elif turn.status == "REPAIR":
        answer = "当前问题需要补充信息后才能回答。"
    else:
        answer = "当前公开资料无法确认这个问题。"
    if turn.source_title:
        source = f"来源：{turn.source_title}"
        if turn.source_url:
            source += f"（{turn.source_url}）"
        return f"{answer}\n{source}"
    return answer


def run_trained_dialogue_terminal(
        *,
        project_root: str | Path,
        qa_database: str | Path,
        training_run_root: str | Path | None = None,
        extra_course_paths: tuple[str | Path, ...] = (),
        extra_variant_course_paths: tuple[str | Path, ...] = (),
        extra_variant_evidence_paths: tuple[str | Path, ...] = (),
        input_stream: BinaryIO | None = None,
        output_stream: BinaryIO | None = None,
        ) -> int:
    """运行可回放的只读交互会话，``:quit`` 结束。"""
    database = _require_k_file(qa_database, label="qa_database")
    root = Path(project_root).resolve()
    trained_surface = None
    if training_run_root is not None:
        run_root = Path(training_run_root).resolve()
        if run_root.drive.upper() != "K:" or not run_root.is_dir():
            raise ValueError("training_run_root 必须是 K 盘已存在目录")
        course_paths = (*default_course_paths(root), *tuple(
            Path(item).resolve() for item in extra_course_paths))
        pack = load_dialogue_training_pack(course_paths)
        load_training_observation(
            run_root, expected_pack_sha256=pack.pack_sha256)
        trained_surface = load_trained_surface_runtime(
            project_root=root,
            training_run_root=run_root,
            expected_pack_sha256=pack.pack_sha256,
            extra_variant_course_paths=tuple(
                Path(item).resolve() for item in extra_variant_course_paths),
            extra_variant_evidence_paths=tuple(
                Path(item).resolve() for item in extra_variant_evidence_paths),
        )
    sparse_runtime = load_or_rebuild_public_sparse_qa_runtime()
    narrow = _narrow_answer(
        sparse_runtime,
        build_public_sentence_demo_catalog(sparse_runtime),
        trained_surface,
    )
    stream_in = sys.stdin.buffer if input_stream is None else input_stream
    stream_out = sys.stdout.buffer if output_stream is None else output_stream
    state = BroadDialogueState((1, 1, 8))
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        while True:
            stream_out.write("你> ".encode("utf-8"))
            stream_out.flush()
            raw = stream_in.readline()
            if raw == b"" or raw.rstrip(b"\r\n") in {b":quit", b":exit"}:
                break
            payload = raw.rstrip(b"\r\n")
            try:
                question = payload.decode("utf-8")
            except UnicodeDecodeError:
                stream_out.write("系统> 输入必须是 UTF-8 文本。\n".encode("utf-8"))
                stream_out.flush()
                continue
            if not question.strip():
                continue
            state, turn = answer_broad_dialogue_turn(
                state, question, connection, narrow_answer=narrow)
            stream_out.write(("系统> " + _display(turn) + "\n").encode("utf-8"))
            stream_out.flush()
    finally:
        connection.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="run read-only trained dialogue terminal")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--qa-database", required=True)
    parser.add_argument("--training-run-root", default=None)
    parser.add_argument("--extra-course", action="append", default=[])
    parser.add_argument("--variant-course", action="append", default=[])
    parser.add_argument("--variant-evidence", action="append", default=[])
    args = parser.parse_args(argv)
    return run_trained_dialogue_terminal(
        project_root=args.project_root,
        qa_database=args.qa_database,
        training_run_root=args.training_run_root,
        extra_course_paths=tuple(args.extra_course),
        extra_variant_course_paths=tuple(args.variant_course),
        extra_variant_evidence_paths=tuple(args.variant_evidence),
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "run_trained_dialogue_terminal"]
