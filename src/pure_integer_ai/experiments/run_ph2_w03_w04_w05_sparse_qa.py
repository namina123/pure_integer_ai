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
    build_public_sparse_qa_runtime,
    run_sparse_qa_queries,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_sparse_qa_session import (
    advance_sparse_qa_session,
    finish_sparse_qa_session,
    iter_sparse_qa_jsonl_session,
    start_sparse_qa_session,
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
    if args.jsonl:
        if (args.question is not None or args.source_ref is not None
                or args.audit or args.repeat != 1):
            parser.error(
                "--jsonl cannot be combined with a positional question, "
                "--source-ref, --audit, or --repeat"
            )
        runtime = build_public_sparse_qa_runtime()
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
    runtime = build_public_sparse_qa_runtime()
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
