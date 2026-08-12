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
from pure_integer_ai.experiments.ph2_broad_qa_query import query_broad_qa
from pure_integer_ai.experiments.ph2_broad_qa_selection import (
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
    """构造 select/build/query 三个显式子命令。"""
    parser = argparse.ArgumentParser(
        description="Build or query the source-bound broad QA preview.")
    subcommands = parser.add_subparsers(dest="command", required=True)
    for name in ("select", "build"):
        command = subcommands.add_parser(name)
        command.add_argument("--run-root", type=_work_path, required=True)
        command.add_argument(
            "--snapshot-manifest", type=Path, required=True)
        command.add_argument("--index", type=_work_path, required=True)
        command.add_argument("--selection", type=_work_path, required=True)
        command.add_argument(
            "--candidate-count", type=_positive, required=True)
        if name == "build":
            command.add_argument("--xml", type=_work_path, required=True)
            command.add_argument("--database", type=_work_path, required=True)
            command.add_argument("--page-count", type=_positive, required=True)
            command.add_argument(
                "--workers", type=_worker_count, default=1)
    query = subcommands.add_parser("query")
    query.add_argument("--run-root", type=_work_path, required=True)
    query.add_argument("--database", type=_work_path, required=True)
    query.add_argument("question")
    return parser


def _validate_run_paths(args: argparse.Namespace) -> None:
    """确保所有大数据输入输出均位于调用方显式 K 盘 run root。"""
    root = args.run_root.resolve()
    if not root.is_dir():
        raise SystemExit("run-root must be an existing directory")
    if args.command == "query":
        names = ("database",)
    elif args.command == "select":
        names = ("index", "selection")
    else:
        names = ("index", "xml", "selection", "database")
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


def main(
        argv: list[str] | None = None,
        *,
        stdout: TextIO | None = None,
        ) -> int:
    """执行不可覆盖构建或只读单问题查询。"""
    args = _parser().parse_args(argv)
    _validate_run_paths(args)
    stream = sys.stdout if stdout is None else stdout
    if args.command == "query":
        database = sqlite3.connect(f"file:{args.database}?mode=ro", uri=True)
        try:
            result = query_broad_qa(database, args.question)
        finally:
            database.close()
        _emit(result.to_dict(), stream)
        return 0
    manifest_path = args.snapshot_manifest.resolve()
    snapshot_bytes = manifest_path.read_bytes()
    snapshot = read_mediawiki_dump_snapshot(manifest_path)
    snapshot_sha256 = hashlib.sha256(snapshot_bytes).hexdigest()
    selection_started_ns = time.perf_counter_ns()
    selection_reused = int(args.selection.exists())
    if selection_reused:
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
    if args.command == "select":
        _emit({
            **selection_profile,
            "selection_elapsed_ns": selection_elapsed_ns,
            "selection_reused": selection_reused,
        }, stream)
        return 0
    if args.page_count > args.candidate_count:
        raise SystemExit("page-count cannot exceed candidate-count")
    report = build_broad_qa_index(
        selection,
        xml_path=args.xml,
        database_path=args.database,
        accepted_page_limit=args.page_count,
        worker_count=args.workers,
    )
    report["selection_elapsed_ns"] = selection_elapsed_ns
    report["selection_reused"] = selection_reused
    report["selection_profile"] = selection_profile
    _emit(report, stream)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
