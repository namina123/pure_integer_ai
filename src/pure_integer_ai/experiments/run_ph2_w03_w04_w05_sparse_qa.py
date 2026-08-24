"""Command-line probe for the reusable public FT22 sparse QA runtime."""
from __future__ import annotations

import argparse
import json
import sys
from typing import TextIO

from pure_integer_ai.experiments.ph2_w03_w04_w05_raw_question_contract import (
    RawQuestionRequest,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_sparse_qa_runtime import (
    run_sparse_qa_query,
    run_sparse_qa_sentence,
    run_sparse_qa_queries,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_sparse_qa_runtime_contract import (
    SparseQARuntime,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_sparse_qa_session import (
    advance_sparse_qa_session,
    finish_sparse_qa_session,
    iter_sparse_qa_jsonl_session,
    start_sparse_qa_session,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_sparse_qa_snapshot import (
    load_or_rebuild_public_sparse_qa_runtime,
)


def _source_ref(value: str) -> tuple[int, ...]:
    parts = tuple(item.strip() for item in value.split(","))
    if not parts or any(not item for item in parts):
        raise argparse.ArgumentTypeError(
            "SourceRef must be a comma-separated integer key")
    try:
        return tuple(int(item) for item in parts)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "SourceRef must be a comma-separated integer key") from error


def _repeat_count(value: str) -> int:
    try:
        count = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("repeat must be an integer") from error
    if not 1 <= count <= 10000:
        raise argparse.ArgumentTypeError("repeat must be between 1 and 10000")
    return count


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Query the experimental public W03-W05 learned runtime through "
            "the FT20 sparse hot path."
        ),
    )
    parser.add_argument("question", nargs="?", help="raw question surface")
    parser.add_argument(
        "--source-ref",
        type=_source_ref,
        default=None,
        help="optional comma-separated integer SourceRef key",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="include the complete FT16 trace projection",
    )
    parser.add_argument(
        "--repeat",
        type=_repeat_count,
        default=1,
        help="repeat the same warm query on one built runtime",
    )
    parser.add_argument(
        "--jsonl",
        action="store_true",
        help="read multiple question objects from stdin using one runtime",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="run a human-facing line-oriented short-QA shell",
    )
    parser.add_argument(
        "--interactive-sentence",
        action="store_true",
        help="show learned complete proposition surfaces in a line shell",
    )
    return parser


def _emit(value: object, stream: TextIO) -> None:
    stream.write(json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ))
    stream.write("\n")
    stream.flush()


def _interactive_reply(status: str, answer_surface: str | None) -> str:
    """Render only a runtime result; this shell never invents a reply."""
    if status == "ANSWER":
        if answer_surface is None:
            raise RuntimeError("ANSWER result has no answer surface")
        return answer_surface
    return f"[{status}]"


def _run_interactive_short_qa(
        runtime: SparseQARuntime,
        input_stream: TextIO,
        output_stream: TextIO,
        ) -> None:
    """Reuse one read-only sparse runtime for a human terminal session."""
    while True:
        output_stream.write("你> ")
        output_stream.flush()
        line = input_stream.readline()
        if line == "":
            return
        question = line.rstrip("\r\n")
        if question in {":quit", ":exit"}:
            return
        try:
            request = RawQuestionRequest(question)
        except ValueError:
            reply = "[INVALID_QUESTION]"
        else:
            result = run_sparse_qa_query(runtime, request)
            reply = _interactive_reply(result.status, result.answer_surface)
        output_stream.write(f"系统> {reply}\n")
        output_stream.flush()


def _run_interactive_sentence_qa(
        runtime: SparseQARuntime,
        input_stream: TextIO,
        output_stream: TextIO,
        ) -> None:
    """展示 W03-W05 proof 的完整命题句；不持久化任何终端历史。"""
    while True:
        output_stream.write("你> ")
        output_stream.flush()
        line = input_stream.readline()
        if line == "":
            return
        question = line.rstrip("\r\n")
        if question in {":quit", ":exit"}:
            return
        try:
            request = RawQuestionRequest(question)
        except ValueError:
            reply = "[INVALID_QUESTION]"
        else:
            result = run_sparse_qa_sentence(runtime, request)
            reply = (
                result.generated_proposition_surface
                if result.query_result.status == "ANSWER"
                else f"[{result.query_result.status}]"
            )
            if reply is None:
                raise RuntimeError("ANSWER sentence projection has no surface")
        output_stream.write(f"系统> {reply}\n")
        output_stream.flush()


def main(
        argv: list[str] | None = None,
        *,
        stdin: TextIO | None = None,
        stdout: TextIO | None = None,
        ) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    input_stream = sys.stdin if stdin is None else stdin
    output_stream = sys.stdout if stdout is None else stdout
    if args.interactive or args.interactive_sentence:
        if (args.question is not None or args.source_ref is not None
                or args.audit or args.repeat != 1 or args.jsonl
                or (args.interactive and args.interactive_sentence)):
            parser.error(
                "interactive shells cannot be combined with a positional question, "
                "--source-ref, --audit, --repeat, --jsonl, or each other"
            )
        runtime = load_or_rebuild_public_sparse_qa_runtime()
        if args.interactive:
            _run_interactive_short_qa(runtime, input_stream, output_stream)
        else:
            _run_interactive_sentence_qa(runtime, input_stream, output_stream)
        return 0
    if args.jsonl:
        if (args.question is not None or args.source_ref is not None
                or args.audit or args.repeat != 1):
            parser.error(
                "--jsonl cannot be combined with a positional question, "
                "--source-ref, --audit, or --repeat"
            )
        runtime = load_or_rebuild_public_sparse_qa_runtime()
        state = start_sparse_qa_session(runtime)
        for record in iter_sparse_qa_jsonl_session(runtime, input_stream):
            _emit(record.to_dict(), output_stream)
            state = advance_sparse_qa_session(state, record)
        probe = finish_sparse_qa_session(state)
        _emit({"kind": "SESSION_PROBE", "probe": probe.to_dict()},
              output_stream)
        return 0
    if args.question is None:
        parser.error("question is required unless --jsonl is used")
    runtime = load_or_rebuild_public_sparse_qa_runtime()
    request = RawQuestionRequest(args.question, args.source_ref)
    batch = run_sparse_qa_queries(
        runtime,
        (request,) * args.repeat,
        audit=args.audit,
    )
    payload = {
        "probe": batch.probe.to_dict(),
        "result": batch.results[0].to_dict(),
    }
    _emit(payload, output_stream)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
