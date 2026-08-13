"""运行自然标题锚定的来源约束检索与证据选择联合评测。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    load_external_qa_sources,
    official_external_qa_sources,
)
from pure_integer_ai.experiments.ph2_broad_qa_index import build_broad_qa_index
from pure_integer_ai.experiments.ph2_broad_qa_joint_eval import (
    augment_broad_qa_index,
    freeze_joint_source_pack,
    predict_joint_retrieval,
    read_joint_source_targets,
    resolve_joint_source_aliases,
    score_joint_retrieval,
)
from pure_integer_ai.experiments.ph2_broad_qa_selection import (
    build_broad_qa_target_selection,
    read_broad_qa_target_selection,
    write_broad_qa_target_selection,
)
from pure_integer_ai.experiments.ph2_mediawiki_snapshot import (
    read_mediawiki_dump_snapshot,
)


def _work_path(value: str) -> Path:
    """要求大数据、评测和索引 artifact 使用显式绝对 K 盘路径。"""
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("joint evaluation paths must be absolute")
    resolved = path.resolve()
    if sys.platform == "win32" and resolved.drive.casefold() != "k:":
        raise argparse.ArgumentTypeError("joint evaluation paths must be on K:")
    return resolved


def _worker_count(value: str) -> int:
    """只接受已验证为确定性的有界 worker 数。"""
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("workers must be an integer") from error
    if result not in {1, 2, 4}:
        raise argparse.ArgumentTypeError("workers must be 1, 2, or 4")
    return result


def _parser() -> argparse.ArgumentParser:
    """构造联合 family 的六个显式、可恢复子命令。"""
    parser = argparse.ArgumentParser(
        description="Freeze and run source-bound retrieval + evidence eval.")
    commands = parser.add_subparsers(dest="command", required=True)

    freeze = commands.add_parser("freeze")
    freeze.add_argument("--run-root", type=_work_path, required=True)
    freeze.add_argument("--cmrc-root", type=_work_path, required=True)
    freeze.add_argument("--drcd-root", type=_work_path, required=True)
    freeze.add_argument("--prior-pack", type=_work_path, required=True)
    freeze.add_argument("--target", type=_work_path, required=True)

    select = commands.add_parser("select")
    select.add_argument("--run-root", type=_work_path, required=True)
    select.add_argument("--snapshot-manifest", type=Path, required=True)
    select.add_argument("--index", type=_work_path, required=True)
    select.add_argument("--source-targets", type=_work_path, required=True)
    select.add_argument("--selection", type=_work_path, required=True)

    build = commands.add_parser("build-target")
    build.add_argument("--run-root", type=_work_path, required=True)
    build.add_argument("--xml", type=_work_path, required=True)
    build.add_argument("--selection", type=_work_path, required=True)
    build.add_argument("--database", type=_work_path, required=True)
    build.add_argument("--workers", type=_worker_count, default=4)

    resolve = commands.add_parser("resolve-aliases")
    resolve.add_argument("--run-root", type=_work_path, required=True)
    resolve.add_argument("--snapshot-manifest", type=Path, required=True)
    resolve.add_argument("--index", type=_work_path, required=True)
    resolve.add_argument("--xml", type=_work_path, required=True)
    resolve.add_argument("--source-targets", type=_work_path, required=True)
    resolve.add_argument("--initial-selection", type=_work_path, required=True)
    resolve.add_argument("--terminal-selection", type=_work_path, required=True)
    resolve.add_argument("--aliases", type=_work_path, required=True)
    resolve.add_argument("--workers", type=_worker_count, default=4)

    augment = commands.add_parser("augment")
    augment.add_argument("--run-root", type=_work_path, required=True)
    augment.add_argument("--base-database", type=_work_path, required=True)
    augment.add_argument("--base-sha256", required=True)
    augment.add_argument("--target-database", type=_work_path, required=True)
    augment.add_argument("--selection", type=_work_path, required=True)
    augment.add_argument("--aliases", type=_work_path, required=True)
    augment.add_argument("--database", type=_work_path, required=True)

    predict = commands.add_parser("predict")
    predict.add_argument("--run-root", type=_work_path, required=True)
    predict.add_argument("--questions", type=_work_path, required=True)
    predict.add_argument("--database", type=_work_path, required=True)
    predict.add_argument("--predictions", type=_work_path, required=True)

    score = commands.add_parser("score")
    score.add_argument("--run-root", type=_work_path, required=True)
    score.add_argument("--questions", type=_work_path, required=True)
    score.add_argument("--predictions", type=_work_path, required=True)
    score.add_argument("--labels", type=_work_path, required=True)
    score.add_argument("--selection", type=_work_path, required=True)
    score.add_argument("--aliases", type=_work_path, required=True)
    score.add_argument("--database", type=_work_path, required=True)
    score.add_argument("--aggregate", type=_work_path, required=True)
    score.add_argument(
        "--scope", choices=("DEVELOPMENT", "FORMAL_HELD_OUT"),
        required=True)
    return parser


def _validate_paths(args: argparse.Namespace) -> None:
    """确保所有 K 盘路径都严格位于调用方声明的 run root。"""
    root = args.run_root.resolve()
    if not root.is_dir():
        raise SystemExit("run-root must be an existing directory")
    names = {
        "freeze": ("cmrc_root", "drcd_root", "prior_pack", "target"),
        "select": ("index", "source_targets", "selection"),
        "build-target": ("xml", "selection", "database"),
        "resolve-aliases": (
            "index", "xml", "source_targets", "initial_selection",
            "terminal_selection", "aliases"),
        "augment": (
            "base_database", "target_database", "selection", "aliases",
            "database"),
        "predict": ("questions", "database", "predictions"),
        "score": (
            "questions", "predictions", "labels", "selection", "aliases",
            "database", "aggregate"),
    }[args.command]
    for name in names:
        if not getattr(args, name).resolve().is_relative_to(root):
            raise SystemExit(f"{name} must stay within run-root")


def _emit(value: object) -> None:
    """输出排序、紧凑且保留中文的单行 JSON。"""
    sys.stdout.write(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    sys.stdout.write("\n")
    sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    """执行一项不可覆盖的联合评测构建、预测或评分操作。"""
    args = _parser().parse_args(argv)
    _validate_paths(args)
    if args.command == "freeze":
        items, source_report = load_external_qa_sources(
            official_external_qa_sources(args.cmrc_root, args.drcd_root))
        report = freeze_joint_source_pack(
            items,
            prior_question_paths=(
                args.prior_pack / "dev.questions.jsonl",
                args.prior_pack / "held_out.questions.jsonl"),
            target_dir=args.target,
            source_report=source_report)
    elif args.command == "select":
        snapshot_path = args.snapshot_manifest.resolve()
        snapshot_payload = snapshot_path.read_bytes()
        snapshot = read_mediawiki_dump_snapshot(snapshot_path)
        selection = build_broad_qa_target_selection(
            snapshot, index_path=args.index,
            snapshot_manifest_sha256=hashlib.sha256(
                snapshot_payload).hexdigest(),
            target_titles=read_joint_source_targets(args.source_targets))
        write_broad_qa_target_selection(selection, args.selection)
        report = {
            "index_entry_count": selection.index_entry_count,
            "matched_title_count": len(selection.selected_pages),
            "missing_title_count": len(selection.missing_title_keys),
            "selection_sha256": selection.sha256(),
            "target_title_count": selection.target_title_count,
        }
    elif args.command == "resolve-aliases":
        snapshot_path = args.snapshot_manifest.resolve()
        snapshot_payload = snapshot_path.read_bytes()
        terminal_selection, report = resolve_joint_source_aliases(
            read_mediawiki_dump_snapshot(snapshot_path),
            read_broad_qa_target_selection(args.initial_selection),
            read_joint_source_targets(args.source_targets),
            snapshot_manifest_sha256=hashlib.sha256(
                snapshot_payload).hexdigest(),
            index_path=args.index, xml_path=args.xml,
            alias_path=args.aliases, worker_count=args.workers)
        write_broad_qa_target_selection(
            terminal_selection, args.terminal_selection)
    elif args.command == "build-target":
        selection = read_broad_qa_target_selection(args.selection)
        report = build_broad_qa_index(
            selection, xml_path=args.xml, database_path=args.database,
            accepted_page_limit=None, worker_count=args.workers)
        report["target_selection_sha256"] = selection.sha256()
        report["unmatched_index_title_count"] = len(
            selection.missing_title_keys)
    elif args.command == "augment":
        selection = read_broad_qa_target_selection(args.selection)
        report = augment_broad_qa_index(
            args.base_database, args.target_database,
            output_database_path=args.database,
            base_expected_sha256=args.base_sha256,
            target_selection_sha256=selection.sha256(),
            alias_path=args.aliases)
    elif args.command == "predict":
        report = predict_joint_retrieval(
            args.questions, args.database,
            predictions_path=args.predictions)
    else:
        report = score_joint_retrieval(
            args.questions, args.predictions, args.labels,
            read_broad_qa_target_selection(args.selection), args.database,
            alias_path=args.aliases, aggregate_path=args.aggregate,
            scope=args.scope)
    _emit(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
