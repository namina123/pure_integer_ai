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
from pure_integer_ai.experiments.ph2_broad_qa_formal_protocol import (
    publish_formal_algorithm_freeze,
    publish_formal_run_intent,
    publish_formal_run_outcome,
)
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
from pure_integer_ai.experiments.ph2_broad_qa_source_aligned_family import (
    derive_source_aligned_runtime_sources,
    freeze_source_aligned_joint_pack,
)
from pure_integer_ai.experiments.ph2_broad_qa_source_alignment import (
    build_source_alignment_census,
    freeze_source_alignment_candidates,
    read_consumed_title_keys,
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
    """构造联合评测与来源对齐的显式、可恢复子命令。"""
    parser = argparse.ArgumentParser(
        description="Freeze and run source-bound retrieval + evidence eval.")
    commands = parser.add_subparsers(dest="command", required=True)

    freeze = commands.add_parser("freeze")
    freeze.add_argument("--run-root", type=_work_path, required=True)
    freeze.add_argument("--cmrc-root", type=_work_path, required=True)
    freeze.add_argument("--drcd-root", type=_work_path, required=True)
    freeze.add_argument("--prior-pack", type=_work_path, required=True)
    freeze.add_argument(
        "--prior-source-targets", type=_work_path, action="append",
        required=True,
        help="Repeat for every consumed joint-family source_targets.jsonl.")
    freeze.add_argument("--target", type=_work_path, required=True)

    alignment_candidates = commands.add_parser(
        "freeze-alignment-candidates")
    alignment_candidates.add_argument(
        "--run-root", type=_work_path, required=True)
    alignment_candidates.add_argument(
        "--cmrc-root", type=_work_path, required=True)
    alignment_candidates.add_argument(
        "--drcd-root", type=_work_path, required=True)
    alignment_candidates.add_argument(
        "--prior-pack", type=_work_path, required=True)
    alignment_candidates.add_argument(
        "--prior-source-targets", type=_work_path, action="append",
        required=True)
    alignment_candidates.add_argument(
        "--target", type=_work_path, required=True)

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
    predict.add_argument("--formal-freeze", type=_work_path)
    predict.add_argument("--formal-intent", type=_work_path)
    predict.add_argument("--repository-root", type=Path)

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
    score.add_argument("--formal-freeze", type=_work_path)
    score.add_argument("--formal-intent", type=_work_path)
    score.add_argument("--repository-root", type=Path)

    formal_freeze = commands.add_parser("freeze-formal-algorithm")
    formal_freeze.add_argument("--run-root", type=_work_path, required=True)
    formal_freeze.add_argument("--family-root", type=_work_path, required=True)
    formal_freeze.add_argument(
        "--candidate-manifest", type=_work_path, required=True)
    formal_freeze.add_argument("--census", type=_work_path, required=True)
    formal_freeze.add_argument(
        "--census-manifest", type=_work_path, required=True)
    formal_freeze.add_argument(
        "--dev-aggregate", type=_work_path, required=True)
    formal_freeze.add_argument("--database", type=_work_path, required=True)
    formal_freeze.add_argument("--aliases", type=_work_path, required=True)
    formal_freeze.add_argument("--selection", type=_work_path, required=True)
    formal_freeze.add_argument(
        "--runtime-source-manifest", type=_work_path, required=True)
    formal_freeze.add_argument(
        "--predictions", type=_work_path, required=True)
    formal_freeze.add_argument("--aggregate", type=_work_path, required=True)
    formal_freeze.add_argument(
        "--repository-root", type=Path, required=True)

    formal_intent = commands.add_parser("claim-formal-run")
    formal_intent.add_argument("--run-root", type=_work_path, required=True)
    formal_intent.add_argument("--formal-freeze", type=_work_path, required=True)
    formal_intent.add_argument(
        "--repository-root", type=Path, required=True)

    formal_outcome = commands.add_parser("publish-formal-outcome")
    formal_outcome.add_argument("--run-root", type=_work_path, required=True)
    formal_outcome.add_argument("--formal-freeze", type=_work_path, required=True)
    formal_outcome.add_argument("--formal-intent", type=_work_path, required=True)
    formal_outcome.add_argument("--aggregate", type=_work_path, required=True)
    formal_outcome.add_argument(
        "--repository-root", type=Path, required=True)

    census = commands.add_parser("alignment-census")
    census.add_argument("--run-root", type=_work_path, required=True)
    census.add_argument("--candidates", type=_work_path, required=True)
    census.add_argument("--aliases", type=_work_path, required=True)
    census.add_argument("--selection", type=_work_path, required=True)
    census.add_argument("--xml", type=_work_path, required=True)
    census.add_argument("--census", type=_work_path, required=True)
    census.add_argument("--manifest", type=_work_path, required=True)
    census.add_argument("--workers", type=_worker_count, default=4)

    aligned_freeze = commands.add_parser("freeze-source-aligned")
    aligned_freeze.add_argument(
        "--run-root", type=_work_path, required=True)
    aligned_freeze.add_argument(
        "--cmrc-root", type=_work_path, required=True)
    aligned_freeze.add_argument(
        "--drcd-root", type=_work_path, required=True)
    aligned_freeze.add_argument(
        "--candidates", type=_work_path, required=True)
    aligned_freeze.add_argument(
        "--candidate-manifest", type=_work_path, required=True)
    aligned_freeze.add_argument(
        "--census", type=_work_path, required=True)
    aligned_freeze.add_argument(
        "--census-manifest", type=_work_path, required=True)
    aligned_freeze.add_argument(
        "--target", type=_work_path, required=True)

    aligned_runtime = commands.add_parser("derive-aligned-runtime-sources")
    aligned_runtime.add_argument(
        "--run-root", type=_work_path, required=True)
    aligned_runtime.add_argument(
        "--source-targets", type=_work_path, required=True)
    aligned_runtime.add_argument(
        "--population-aliases", type=_work_path, required=True)
    aligned_runtime.add_argument(
        "--population-selection", type=_work_path, required=True)
    aligned_runtime.add_argument(
        "--aliases", type=_work_path, required=True)
    aligned_runtime.add_argument(
        "--selection", type=_work_path, required=True)
    aligned_runtime.add_argument(
        "--manifest", type=_work_path, required=True)
    return parser


def _validate_paths(args: argparse.Namespace) -> None:
    """确保所有 K 盘路径都严格位于调用方声明的 run root。"""
    root = args.run_root.resolve()
    if not root.is_dir():
        raise SystemExit("run-root must be an existing directory")
    names = {
        "freeze": ("cmrc_root", "drcd_root", "prior_pack", "target"),
        "freeze-alignment-candidates": (
            "cmrc_root", "drcd_root", "prior_pack", "target"),
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
        "alignment-census": (
            "candidates", "aliases", "selection", "xml", "census",
            "manifest"),
        "freeze-source-aligned": (
            "cmrc_root", "drcd_root", "candidates", "candidate_manifest",
            "census", "census_manifest", "target"),
        "derive-aligned-runtime-sources": (
            "source_targets", "population_aliases", "population_selection",
            "aliases", "selection", "manifest"),
        "freeze-formal-algorithm": (
            "family_root", "candidate_manifest", "census",
            "census_manifest", "dev_aggregate", "database", "aliases",
            "selection", "runtime_source_manifest", "predictions",
            "aggregate"),
        "claim-formal-run": ("formal_freeze",),
        "publish-formal-outcome": (
            "formal_freeze", "formal_intent", "aggregate"),
    }[args.command]
    for name in names:
        if not getattr(args, name).resolve().is_relative_to(root):
            raise SystemExit(f"{name} must stay within run-root")
    if args.command in {"freeze", "freeze-alignment-candidates"}:
        for path in args.prior_source_targets:
            if not path.resolve().is_relative_to(root):
                raise SystemExit(
                    "prior_source_targets must stay within run-root")


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
            prior_source_target_paths=args.prior_source_targets,
            target_dir=args.target,
            source_report=source_report)
    elif args.command == "freeze-alignment-candidates":
        items, source_report = load_external_qa_sources(
            official_external_qa_sources(args.cmrc_root, args.drcd_root))
        consumed_paths = (
            args.prior_pack / "dev.questions.jsonl",
            args.prior_pack / "held_out.questions.jsonl",
            *args.prior_source_targets,
        )
        report = freeze_source_alignment_candidates(
            items,
            excluded_title_keys=read_consumed_title_keys(
                prior_question_paths=consumed_paths[:2],
                prior_source_target_paths=args.prior_source_targets),
            excluded_title_source_paths=consumed_paths,
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
            predictions_path=args.predictions,
            formal_run_root=(
                args.run_root if args.formal_freeze is not None else None),
            formal_freeze_path=args.formal_freeze,
            formal_intent_path=args.formal_intent,
            repository_root=args.repository_root)
    elif args.command == "score":
        report = score_joint_retrieval(
            args.questions, args.predictions, args.labels,
            read_broad_qa_target_selection(args.selection), args.database,
            alias_path=args.aliases, aggregate_path=args.aggregate,
            scope=args.scope,
            formal_run_root=(
                args.run_root if args.scope == "FORMAL_HELD_OUT" else None),
            formal_freeze_path=args.formal_freeze,
            formal_intent_path=args.formal_intent,
            formal_selection_path=(
                args.selection if args.scope == "FORMAL_HELD_OUT" else None),
            repository_root=args.repository_root)
    elif args.command == "alignment-census":
        report = build_source_alignment_census(
            args.candidates, args.aliases,
            read_broad_qa_target_selection(args.selection),
            xml_path=args.xml, census_path=args.census,
            manifest_path=args.manifest, worker_count=args.workers)
    elif args.command == "freeze-source-aligned":
        items, source_report = load_external_qa_sources(
            official_external_qa_sources(args.cmrc_root, args.drcd_root))
        report = freeze_source_aligned_joint_pack(
            items, candidates_path=args.candidates,
            census_path=args.census,
            census_manifest_path=args.census_manifest,
            candidate_manifest_path=args.candidate_manifest,
            target_dir=args.target, source_report=source_report)
    elif args.command == "derive-aligned-runtime-sources":
        report = derive_source_aligned_runtime_sources(
            args.source_targets, args.population_aliases,
            read_broad_qa_target_selection(args.population_selection),
            aliases_path=args.aliases,
            terminal_selection_path=args.selection,
            manifest_path=args.manifest)
    elif args.command == "freeze-formal-algorithm":
        report = publish_formal_algorithm_freeze(
            args.run_root, args.family_root,
            candidate_manifest_path=args.candidate_manifest,
            census_path=args.census,
            census_manifest_path=args.census_manifest,
            dev_aggregate_path=args.dev_aggregate,
            database_path=args.database, alias_path=args.aliases,
            terminal_selection_path=args.selection,
            runtime_source_manifest_path=args.runtime_source_manifest,
            predictions_path=args.predictions,
            aggregate_path=args.aggregate,
            repository_root=args.repository_root)
    elif args.command == "claim-formal-run":
        report = publish_formal_run_intent(
            args.run_root, args.formal_freeze,
            repository_root=args.repository_root)
    else:
        report = publish_formal_run_outcome(
            args.run_root, args.formal_freeze, args.formal_intent,
            args.aggregate, repository_root=args.repository_root)
    _emit(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
