"""构建和查询来源约束中文广域问答 V0。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time
from typing import TextIO

from pure_integer_ai.experiments.ph2_broad_qa_index import (
    build_broad_qa_index,
)
from pure_integer_ai.experiments.ph2_broad_qa_interactive import (
    render_broad_qa_text,
)
from pure_integer_ai.experiments.ph2_broad_qa_query import query_broad_qa
from pure_integer_ai.experiments.ph2_broad_qa_sharded import (
    build_broad_qa_sharded_index,
)
from pure_integer_ai.experiments.ph2_broad_qa_selection import (
    derive_broad_qa_selection_prefix,
    build_broad_qa_selection,
    profile_broad_qa_selection,
    read_broad_qa_selection,
    write_broad_qa_selection,
)
from pure_integer_ai.experiments.ph2_mediawiki_snapshot import (
    read_mediawiki_dump_snapshot,
)


def _positive(value: str) -> int:
    """把命令行文本解析为正严格整数。"""
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be an integer") from error
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def _worker_count(value: str) -> int:
    """只接受完成确定性验证的有界 worker 数。"""
    result = _positive(value)
    if result not in {1, 2, 4}:
        raise argparse.ArgumentTypeError("workers must be 1, 2, or 4")
    return result


def _work_path(value: str) -> Path:
    """本机强制 K 盘；无盘符平台接受调用方显式绝对工作路径。"""
    raw = Path(value)
    if not raw.is_absolute():
        raise argparse.ArgumentTypeError("large-data paths must be absolute")
    path = raw.resolve()
    if sys.platform == "win32" and path.drive.casefold() != "k:":
        raise argparse.ArgumentTypeError("large-data paths must be on K:")
    return path


def _parser() -> argparse.ArgumentParser:
    """构造选择、构建、机器查询和人类问答子命令。"""
    parser = argparse.ArgumentParser(
        description="Build or query the source-bound broad QA preview.")
    subcommands = parser.add_subparsers(dest="command", required=True)
    for name in ("select", "prefix", "build", "build-sharded"):
        command = subcommands.add_parser(name)
        command.add_argument("--run-root", type=_work_path, required=True)
        command.add_argument(
            "--snapshot-manifest", type=Path, required=True)
        command.add_argument("--index", type=_work_path, required=True)
        command.add_argument("--selection", type=_work_path, required=True)
        command.add_argument(
            "--candidate-count", type=_positive, required=True)
        if name == "prefix":
            command.add_argument(
                "--parent-selection", type=_work_path, required=True)
        if name in {"build", "build-sharded"}:
            command.add_argument("--xml", type=_work_path, required=True)
            command.add_argument("--database", type=_work_path, required=True)
            command.add_argument("--page-count", type=_positive, required=True)
            command.add_argument(
                "--workers", type=_worker_count, default=1)
            if name == "build-sharded":
                command.add_argument("--shard-root", type=_work_path, required=True)
                command.add_argument(
                    "--max-blocks-per-shard", type=_positive, default=512)
                command.add_argument(
                    "--max-new-projection-shards", type=_positive)
                command.add_argument(
                    "--max-new-posting-shards", type=_positive)
                command.add_argument(
                    "--no-publish", action="store_true")
                command.add_argument(
                    "--discard-unsealed", action="store_true")
    for name in ("query", "ask"):
        query = subcommands.add_parser(name)
        query.add_argument("--run-root", type=_work_path, required=True)
        query.add_argument("--database", type=_work_path, required=True)
        query.add_argument("question", nargs="?" if name == "ask" else None)
        if name == "ask":
            query.add_argument(
                "--audit", action="store_true",
                help="Emit the complete canonical JSON result.")
    return parser


def _validate_run_paths(args: argparse.Namespace) -> None:
    """确保所有大数据输入输出均位于调用方显式 K 盘 run root。"""
    root = args.run_root.resolve()
    if not root.is_dir():
        raise SystemExit("run-root must be an existing directory")
    if args.command in {"query", "ask"}:
        names = ("database",)
    elif args.command in {"select", "prefix"}:
        names = ("index", "selection")
        if args.command == "prefix":
            names = names + ("parent_selection",)
    elif args.command == "build":
        names = ("index", "xml", "selection", "database")
    else:
        names = (
            "index", "xml", "selection", "database", "shard_root")
    for name in names:
        path = getattr(args, name).resolve()
        if not path.is_relative_to(root):
            raise SystemExit(f"{name} must stay within run-root")


def _emit(value: object, stream: TextIO) -> None:
    """输出排序、紧凑且保留中文的单行 JSON。"""
    stream.write(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    stream.write("\n")
    stream.flush()


def _ask_questions(
        database_path: Path,
        questions: tuple[str, ...],
        *, audit: bool,
        stream: TextIO,
        ) -> int:
    """在同一只读连接上回答多题，并选择文本或完整审计输出。"""
    if (not questions or any(
            not isinstance(item, str) or not item.strip()
            for item in questions)):
        raise SystemExit("ask requires a question argument or non-empty stdin")
    database = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
    try:
        for ordinal, question in enumerate(questions):
            result = query_broad_qa(database, question.strip())
            if audit:
                _emit(result.to_dict(), stream)
            else:
                if ordinal:
                    stream.write("\n\n")
                stream.write(render_broad_qa_text(result))
                stream.write("\n")
                stream.flush()
    finally:
        database.close()
    return 0


def _stdin_questions(stream: TextIO) -> tuple[str, ...]:
    """读取 UTF-8 文本行，并移除首行可能存在的 BOM。"""
    result = []
    for raw in stream:
        value = raw.strip()
        if not result:
            value = value.lstrip("\ufeff")
        if value:
            result.append(value)
    return tuple(result)


def main(
        argv: list[str] | None = None,
        *,
        stdout: TextIO | None = None,
        stdin: TextIO | None = None,
        ) -> int:
    """执行不可覆盖构建、机器查询或只读交互问答。"""
    args = _parser().parse_args(argv)
    _validate_run_paths(args)
    stream = sys.stdout if stdout is None else stdout
    input_stream = sys.stdin if stdin is None else stdin
    if args.command == "query":
        database = sqlite3.connect(f"file:{args.database}?mode=ro", uri=True)
        try:
            result = query_broad_qa(database, args.question)
        finally:
            database.close()
        _emit(result.to_dict(), stream)
        return 0
    if args.command == "ask":
        questions = (
            (args.question,) if args.question is not None
            else _stdin_questions(input_stream)
        )
        return _ask_questions(
            args.database, questions, audit=args.audit, stream=stream)
    manifest_path = args.snapshot_manifest.resolve()
    snapshot_bytes = manifest_path.read_bytes()
    snapshot = read_mediawiki_dump_snapshot(manifest_path)
    snapshot_sha256 = hashlib.sha256(snapshot_bytes).hexdigest()
    selection_started_ns = time.perf_counter_ns()
    selection_reused = int(args.selection.exists())
    if args.command == "prefix":
        parent = read_broad_qa_selection(args.parent_selection)
        if args.candidate_count > parent.requested_page_count:
            raise SystemExit("prefix count cannot exceed parent selection")
        selection = derive_broad_qa_selection_prefix(
            parent, requested_page_count=args.candidate_count)
        write_broad_qa_selection(selection, args.selection)
        selection_reused = 0
    elif selection_reused:
        selection = read_broad_qa_selection(args.selection)
        xml_identity = next(
            item for item in snapshot.raw_files if item.role == "XML")
        index_identity = next(
            item for item in snapshot.raw_files if item.role == "INDEX")
        if (selection.snapshot_manifest_sha256 != snapshot_sha256
                or selection.snapshot_id != snapshot.snapshot_id
                or selection.requested_page_count != args.candidate_count
                or selection.xml_local_sha256 != xml_identity.local_sha256
                or selection.index_local_sha256
                != index_identity.local_sha256):
            raise SystemExit("existing selection identity does not match build")
    else:
        selection = build_broad_qa_selection(
            snapshot,
            index_path=args.index,
            snapshot_manifest_sha256=snapshot_sha256,
            requested_page_count=args.candidate_count,
        )
        write_broad_qa_selection(selection, args.selection)
    selection_elapsed_ns = max(
        1, time.perf_counter_ns() - selection_started_ns)
    selection_profile = profile_broad_qa_selection(selection)
    if args.command in {"select", "prefix"}:
        _emit({
            **selection_profile,
            "selection_elapsed_ns": selection_elapsed_ns,
            "selection_reused": selection_reused,
        }, stream)
        return 0
    if args.page_count > args.candidate_count:
        raise SystemExit("page-count cannot exceed candidate-count")
    if args.command == "build":
        report = build_broad_qa_index(
            selection,
            xml_path=args.xml,
            database_path=args.database,
            accepted_page_limit=args.page_count,
            worker_count=args.workers,
        )
    else:
        report = build_broad_qa_sharded_index(
            selection,
            xml_path=args.xml,
            shard_root=args.shard_root,
            database_path=args.database,
            accepted_page_count=args.page_count,
            max_blocks_per_shard=args.max_blocks_per_shard,
            worker_count=args.workers,
            max_new_projection_shards=args.max_new_projection_shards,
            max_new_posting_shards=args.max_new_posting_shards,
            publish=not args.no_publish,
            discard_unsealed=args.discard_unsealed,
        )
    report["selection_elapsed_ns"] = selection_elapsed_ns
    report["selection_reused"] = selection_reused
    report["selection_profile"] = selection_profile
    _emit(report, stream)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
