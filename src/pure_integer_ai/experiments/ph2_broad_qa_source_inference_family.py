"""冻结来源内归纳 family 的未消费 review roster。

本模块只按既有来源对齐状态、未消费标题和稳定 item 身份选样。它不会把
``GOLD_ABSENT`` 自动解释为可推导或冲突；这些判断必须在 roster 冻结后由完整
decision ledger 覆盖，并为每个非抽取结论提供来源或推导合同承诺。
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Iterable

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
    ExternalQaItem,
    load_external_qa_sources,
    official_external_qa_sources,
)
from pure_integer_ai.experiments.ph2_broad_qa_joint_eval import (
    read_joint_source_targets,
)
from pure_integer_ai.experiments.ph2_broad_qa_source_alignment import (
    SOURCE_ALIGNED_STATUS,
    read_source_alignment_candidates,
    read_source_alignment_census,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
)


SOURCE_INFERENCE_ROSTER_KIND = (
    "PH2_BROAD_QA_SOURCE_INFERENCE_REVIEW_ROSTER_V1")
SOURCE_INFERENCE_ROSTER_RECORD_KIND = (
    "PH2_BROAD_QA_SOURCE_INFERENCE_REVIEW_ROSTER_RECORD_V1")
SOURCE_INFERENCE_REVIEW_PAYLOAD_KIND = (
    "PH2_BROAD_QA_SOURCE_INFERENCE_REVIEW_PAYLOAD_V1")
SOURCE_INFERENCE_ROSTER_SELECTION_RULE = (
    "EXCLUDE_CONSUMED_TITLE_THEN_ALIGNMENT_BUCKET_THEN_SOURCE_BALANCE_"
    "THEN_UNIQUE_TITLE_THEN_ITEM_SHA256_V1")
SOURCE_INFERENCE_ASSIGNMENTS = (
    "EXTRACTIVE_CANDIDATE",
    "NON_EXTRACTIVE_REVIEW",
)
SOURCE_INFERENCE_DECISIONS = (
    "EXTRACTIVE",
    "SOURCE_DERIVABLE",
    "SOURCE_CONFLICT",
    "REJECT",
)
_NON_EXTRACTIVE_STATUS = "GOLD_ABSENT_FROM_TERMINAL_REVISION"


def _sha256_file(path: Path) -> str:
    """流式计算输入或冻结 artifact 的 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _write_jsonl(
        path: Path,
        records: Iterable[dict[str, object]],
        ) -> int:
    """不可覆盖地写规范 JSONL 并返回记录数。"""
    if path.exists():
        raise BroadQaExternalDataError(
            f"source inference roster 禁止覆盖: {path.name}")
    count = 0
    with path.open("xb") as handle:
        for record in records:
            handle.write(canonical_json_line(record))
            count += 1
    return count


def _within(root: Path, path: Path, *, label: str) -> Path:
    """解析工作路径并要求其始终位于显式 run root 内。"""
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise BroadQaExternalDataError(f"{label} 必须位于 run root 内")
    return resolved


def _source_balanced_selection(
        records: tuple[dict[str, object], ...],
        *,
        quota: int,
        used_titles: set[str],
        ) -> tuple[dict[str, object], ...]:
    """按来源等额、标题唯一和 item SHA 顺序选择一个机械 bucket。"""
    if type(quota) is not int or quota <= 0:
        raise BroadQaExternalDataError("source inference roster quota 非法")
    by_source: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        by_source[str(record["source_key"])].append(record)
    sources = tuple(sorted(source for source, values in by_source.items() if values))
    if not sources:
        raise BroadQaExternalDataError("source inference roster bucket 为空")
    base, remainder = divmod(quota, len(sources))
    selected = []
    for source_index, source in enumerate(sources):
        source_quota = base + int(source_index < remainder)
        picked = 0
        for record in sorted(by_source[source], key=lambda item: item["item_id"]):
            title_key = str(record["title_key"])
            if title_key in used_titles:
                continue
            selected.append(record)
            used_titles.add(title_key)
            picked += 1
            if picked == source_quota:
                break
        if picked != source_quota:
            raise BroadQaExternalDataError(
                f"source inference roster 来源库存不足: {source}")
    return tuple(selected)


def freeze_source_inference_review_roster(
        items: Iterable[ExternalQaItem],
        *,
        run_root: str | Path,
        candidates_path: str | Path,
        source_census_path: str | Path,
        consumed_source_target_paths: Iterable[str | Path],
        target_dir: str | Path,
        source_report: dict[str, object],
        extractive_quota: int = 20,
        non_extractive_review_quota: int = 60,
        target_stratum_quota: int = 15,
        ) -> dict[str, object]:
    """冻结两类机械 roster，等待独立四态 review ledger 全覆盖。"""
    root = Path(run_root).resolve()
    if not root.is_dir():
        raise BroadQaExternalDataError("source inference run root 不存在")
    candidate_file = _within(
        root, Path(candidates_path), label="candidates_path")
    census_file = _within(
        root, Path(source_census_path), label="source_census_path")
    target = _within(root, Path(target_dir), label="target_dir")
    consumed_paths = tuple(
        _within(root, Path(path), label="consumed_source_target")
        for path in consumed_source_target_paths)
    if (not consumed_paths or target.exists()
            or type(target_stratum_quota) is not int
            or target_stratum_quota <= 0
            or target_stratum_quota > extractive_quota
            or target_stratum_quota * 2 > non_extractive_review_quota):
        raise BroadQaExternalDataError(
            "source inference roster 路径或配额边界非法")

    candidates = read_source_alignment_candidates(candidate_file)
    census = read_source_alignment_census(census_file)
    candidate_by_id = {item["item_id"]: item for item in candidates}
    census_by_id = {item["item_id"]: item for item in census}
    official_by_id = {item.item_id: item for item in items}
    if (set(candidate_by_id) != set(census_by_id)
            or not set(candidate_by_id).issubset(official_by_id)):
        raise BroadQaExternalDataError(
            "source inference roster 输入 inventory 漂移")

    consumed_titles = set()
    for path in consumed_paths:
        consumed_titles.update(read_joint_source_targets(path))
    if not consumed_titles:
        raise BroadQaExternalDataError(
            "source inference consumed title inventory 为空")

    buckets: dict[str, list[dict[str, object]]] = {
        assignment: [] for assignment in SOURCE_INFERENCE_ASSIGNMENTS}
    excluded_consumed_count = 0
    ignored_status_counts: Counter[str] = Counter()
    for item_id in sorted(candidate_by_id):
        candidate = candidate_by_id[item_id]
        source_record = census_by_id[item_id]
        official = official_by_id[item_id]
        if (candidate["source_key"] != official.source_key
                or candidate["title_key"] != official.title_key
                or tuple(candidate["gold_answers"]) != official.gold_answers
                or source_record["source_key"] != official.source_key
                or source_record["title_key"] != official.title_key):
            raise BroadQaExternalDataError(
                "source inference official/census binding 漂移")
        if official.title_key in consumed_titles:
            excluded_consumed_count += 1
            continue
        status = str(source_record["status"])
        if status == SOURCE_ALIGNED_STATUS:
            assignment = "EXTRACTIVE_CANDIDATE"
        elif status == _NON_EXTRACTIVE_STATUS:
            assignment = "NON_EXTRACTIVE_REVIEW"
        else:
            ignored_status_counts[status] += 1
            continue
        buckets[assignment].append({
            "assignment": assignment,
            "format_version": 1,
            "item_id": official.item_id,
            "question_sha256": hashlib.sha256(
                official.question.encode("utf-8")).hexdigest(),
            "record_kind": SOURCE_INFERENCE_ROSTER_RECORD_KIND,
            "source_alignment_status": status,
            "source_key": official.source_key,
            "terminal_page_id": source_record["terminal_page_id"],
            "terminal_revision_id": source_record["terminal_revision_id"],
            "title_key": official.title_key,
        })

    used_titles: set[str] = set()
    selected = (
        *_source_balanced_selection(
            tuple(buckets["EXTRACTIVE_CANDIDATE"]),
            quota=extractive_quota,
            used_titles=used_titles,
        ),
        *_source_balanced_selection(
            tuple(buckets["NON_EXTRACTIVE_REVIEW"]),
            quota=non_extractive_review_quota,
            used_titles=used_titles,
        ),
    )
    if (len(selected) != extractive_quota + non_extractive_review_quota
            or len(used_titles) != len(selected)):
        raise BroadQaExternalDataError(
            "source inference roster selection 未闭合")

    target.mkdir(parents=True)
    roster_path = target / "review.roster.jsonl"
    payload_path = target / "review.payload.jsonl"
    roster_count = _write_jsonl(roster_path, selected)
    payload_count = _write_jsonl(payload_path, ({
        "assignment": record["assignment"],
        "context": official_by_id[str(record["item_id"])].context,
        "context_sha256": hashlib.sha256(
            official_by_id[str(record["item_id"])].context.encode("utf-8")
        ).hexdigest(),
        "format_version": 1,
        "gold_answers": list(
            official_by_id[str(record["item_id"])].gold_answers),
        "item_id": record["item_id"],
        "license_id": official_by_id[str(record["item_id"])].license_id,
        "question": official_by_id[str(record["item_id"])].question,
        "record_kind": SOURCE_INFERENCE_REVIEW_PAYLOAD_KIND,
        "source_key": record["source_key"],
        "source_partition": official_by_id[
            str(record["item_id"])].source_partition,
        "source_question_id": official_by_id[
            str(record["item_id"])].source_question_id,
        "source_revision": official_by_id[
            str(record["item_id"])].source_revision,
        "terminal_page_id": record["terminal_page_id"],
        "terminal_revision_id": record["terminal_revision_id"],
        "title": official_by_id[str(record["item_id"])].title,
        "title_key": record["title_key"],
        "upstream_url": official_by_id[str(record["item_id"])].upstream_url,
    } for record in selected))
    if roster_count != payload_count or roster_count != len(selected):
        raise BroadQaExternalDataError(
            "source inference roster/payload count 漂移")

    assignment_counts = Counter(str(item["assignment"]) for item in selected)
    source_assignment_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for item in selected:
        source_assignment_counts[str(item["source_key"])][
            str(item["assignment"])] += 1
    manifest = {
        "artifact_kind": SOURCE_INFERENCE_ROSTER_KIND,
        "artifacts": [
            {
                "bytes": roster_path.stat().st_size,
                "record_count": roster_count,
                "role": "review_roster_without_labels",
                "sha256": _sha256_file(roster_path),
            },
            {
                "bytes": payload_path.stat().st_size,
                "record_count": payload_count,
                "role": "private_development_review_payload",
                "sha256": _sha256_file(payload_path),
            },
        ],
        "assignment_counts": {
            key: assignment_counts[key] for key in SOURCE_INFERENCE_ASSIGNMENTS},
        "candidate_count": len(candidates),
        "candidates_sha256": _sha256_file(candidate_file),
        "consumed_source_targets": [
            {"sha256": _sha256_file(path)} for path in consumed_paths],
        "consumed_title_count": len(consumed_titles),
        "decision_ledger_contract": {
            "allowed_decisions": list(SOURCE_INFERENCE_DECISIONS),
            "exact_roster_coverage_required": 1,
            "extractive_requires_source_aligned": 1,
            "source_conflict_requires_distinct_source_commitments": 1,
            "source_derivable_requires_inference_record_sha256": 1,
            "unreviewed_item_selection_forbidden": 1,
        },
        "excluded_consumed_item_count": excluded_consumed_count,
        "format_version": 1,
        "ignored_status_counts": dict(sorted(ignored_status_counts.items())),
        "non_extractive_review_quota": non_extractive_review_quota,
        "question_count": len(selected),
        "selection_rule": SOURCE_INFERENCE_ROSTER_SELECTION_RULE,
        "source_assignment_counts": {
            source: {
                assignment: values[assignment]
                for assignment in SOURCE_INFERENCE_ASSIGNMENTS
            }
            for source, values in sorted(source_assignment_counts.items())
        },
        "source_census_sha256": _sha256_file(census_file),
        "source_report": source_report,
        "status": "FROZEN_UNREVIEWED_NOT_RUN",
        "target_stratum_quota": target_stratum_quota,
        "title_count": len(used_titles),
        "title_domain_overlap_count": 0,
    }
    manifest_path = target / "manifest.json"
    manifest_path.write_bytes(canonical_json_line(manifest))
    return {**manifest, "manifest_sha256": _sha256_file(manifest_path)}


def _positive(value: str) -> int:
    """把 CLI 配额解析为正严格整数。"""
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be an integer") from error
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def _work_path(value: str) -> Path:
    """要求实际大数据路径是显式绝对 K 盘路径。"""
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("work paths must be absolute")
    resolved = path.resolve()
    if sys.platform == "win32" and resolved.drive.casefold() != "k:":
        raise argparse.ArgumentTypeError("work paths must be on K:")
    return resolved


def main(argv: list[str] | None = None) -> int:
    """从官方来源和已冻结 source-alignment 发布未消费 review roster。"""
    parser = argparse.ArgumentParser(
        description="Freeze an unconsumed source-inference review roster.")
    parser.add_argument("--run-root", type=_work_path, required=True)
    parser.add_argument("--cmrc-root", type=_work_path, required=True)
    parser.add_argument("--drcd-root", type=_work_path, required=True)
    parser.add_argument("--candidates", type=_work_path, required=True)
    parser.add_argument("--source-census", type=_work_path, required=True)
    parser.add_argument(
        "--consumed-source-target", type=_work_path,
        action="append", required=True)
    parser.add_argument("--target-dir", type=_work_path, required=True)
    parser.add_argument("--extractive-quota", type=_positive, default=20)
    parser.add_argument(
        "--non-extractive-review-quota", type=_positive, default=60)
    parser.add_argument("--target-stratum-quota", type=_positive, default=15)
    args = parser.parse_args(argv)
    sources = official_external_qa_sources(args.cmrc_root, args.drcd_root)
    items, source_report = load_external_qa_sources(sources)
    report = freeze_source_inference_review_roster(
        items,
        run_root=args.run_root,
        candidates_path=args.candidates,
        source_census_path=args.source_census,
        consumed_source_target_paths=args.consumed_source_target,
        target_dir=args.target_dir,
        source_report=source_report,
        extractive_quota=args.extractive_quota,
        non_extractive_review_quota=args.non_extractive_review_quota,
        target_stratum_quota=args.target_stratum_quota,
    )
    sys.stdout.write(json.dumps(
        report, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SOURCE_INFERENCE_ASSIGNMENTS",
    "SOURCE_INFERENCE_DECISIONS",
    "SOURCE_INFERENCE_REVIEW_PAYLOAD_KIND",
    "SOURCE_INFERENCE_ROSTER_KIND",
    "SOURCE_INFERENCE_ROSTER_RECORD_KIND",
    "SOURCE_INFERENCE_ROSTER_SELECTION_RULE",
    "freeze_source_inference_review_roster",
    "main",
]
