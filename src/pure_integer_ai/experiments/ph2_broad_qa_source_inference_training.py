"""冻结来源归纳 learner 的独立训练总体。

本模块只根据既有 source-alignment 状态建立机械训练桶。它在 learner 读取任何
样本前发布 roster、完整训练 payload、未来评测排除清单和 manifest；不会预写
operator、predicate、role、decision 或 mastery 标签。
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
    normalize_external_text,
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
from pure_integer_ai.experiments.ph2_broad_qa_source_inference_family import (
    SOURCE_INFERENCE_ROSTER_RECORD_KIND,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
)


SOURCE_INFERENCE_TRAINING_KIND = (
    "PH2_BROAD_QA_SOURCE_INFERENCE_TRAINING_ROSTER_V1")
SOURCE_INFERENCE_TRAINING_RECORD_KIND = (
    "PH2_BROAD_QA_SOURCE_INFERENCE_TRAINING_ROSTER_RECORD_V1")
SOURCE_INFERENCE_TRAINING_PAYLOAD_KIND = (
    "PH2_BROAD_QA_SOURCE_INFERENCE_TRAINING_PAYLOAD_V1")
SOURCE_INFERENCE_TRAINING_EXCLUSION_KIND = (
    "PH2_BROAD_QA_SOURCE_INFERENCE_TRAINING_EXCLUSION_V1")
SOURCE_INFERENCE_TRAINING_SELECTION_RULE = (
    "EXCLUDE_ALL_CONSUMED_AND_DISCOVERY_V1_TITLE_ITEM_THEN_ALIGNMENT_BUCKET_"
    "THEN_CMRC_DRCD_EQUAL_QUOTA_THEN_UNIQUE_TITLE_THEN_ITEM_SHA256_V1")
SOURCE_INFERENCE_TRAINING_SOURCES = ("CMRC2018", "DRCD")
SOURCE_INFERENCE_TRAINING_ASSIGNMENTS = (
    "EXTRACTIVE_REFERENCE",
    "PASSAGE_COVERAGE_REVIEW",
    "RAW_WIKITEXT_REVIEW",
    "NON_EXTRACTIVE_DISCOVERY",
)
SOURCE_INFERENCE_OPERATOR_FAMILIES = (
    "NORMALIZATION_EQUIVALENCE",
    "SOURCE_SPAN_SELECTION",
    "PARENTHETICAL_EXPANSION",
    "ENUMERATION_MEMBER_SELECTION",
    "EXPLICIT_UNIT_ERA_FORMAT_MAPPING",
    "FINITE_ROLE_COMPOSITION",
)
_STATUS_ASSIGNMENTS = {
    SOURCE_ALIGNED_STATUS: "EXTRACTIVE_REFERENCE",
    "GOLD_PRESENT_OUTSIDE_PROJECTED_PASSAGES": "PASSAGE_COVERAGE_REVIEW",
    "GOLD_ONLY_IN_RAW_WIKITEXT": "RAW_WIKITEXT_REVIEW",
    "GOLD_ABSENT_FROM_TERMINAL_REVISION": "NON_EXTRACTIVE_DISCOVERY",
}


def _sha256_file(path: Path) -> str:
    """流式计算冻结输入或 artifact 的 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _within(root: Path, path: str | Path, *, label: str) -> Path:
    """解析路径并要求它始终位于显式 K 盘 run root 内。"""
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(root):
        raise BroadQaExternalDataError(f"{label} 必须位于 run root 内")
    return resolved


def _write_jsonl(
        path: Path,
        records: Iterable[dict[str, object]],
        ) -> int:
    """不可覆盖地写规范 JSONL，并返回记录数。"""
    if path.exists():
        raise BroadQaExternalDataError(
            f"source inference training 禁止覆盖: {path.name}")
    count = 0
    with path.open("xb") as handle:
        for record in records:
            handle.write(canonical_json_line(record))
            count += 1
    return count


def _read_excluded_roster(
        path: Path,
        ) -> tuple[tuple[str, str], ...]:
    """严格回读已消费 discovery roster 的 item/title 双域。"""
    expected_fields = {
        "assignment", "format_version", "item_id", "question_sha256",
        "record_kind", "source_alignment_status", "source_key",
        "terminal_page_id", "terminal_revision_id", "title_key",
    }
    values = []
    identities = set()
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for line_number, line in enumerate(handle, start=1):
                value = json.loads(line)
                item_id = value.get("item_id") if isinstance(value, dict) else None
                title_key = (
                    value.get("title_key") if isinstance(value, dict) else None)
                if (not line.endswith("\n") or not isinstance(value, dict)
                        or canonical_json_line(value) != line.encode("utf-8")
                        or set(value) != expected_fields
                        or value.get("format_version") != 1
                        or value.get("record_kind")
                        != SOURCE_INFERENCE_ROSTER_RECORD_KIND
                        or not isinstance(item_id, str) or len(item_id) != 64
                        or any(character not in "0123456789abcdef"
                               for character in item_id)
                        or item_id in identities
                        or not isinstance(title_key, str) or not title_key
                        or normalize_external_text(title_key) != title_key):
                    raise BroadQaExternalDataError(
                        f"excluded source inference roster 漂移: {line_number}")
                identities.add(item_id)
                values.append((item_id, title_key))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "excluded source inference roster 不可读") from error
    if not values:
        raise BroadQaExternalDataError(
            "excluded source inference roster 为空")
    return tuple(values)


def _select_training_records(
        buckets: dict[str, dict[str, list[dict[str, object]]]],
        *,
        quotas_per_source: dict[str, int],
        ) -> tuple[dict[str, object], ...]:
    """按机械桶与来源等额选样，并保持全局标题唯一。"""
    if (set(quotas_per_source) != set(SOURCE_INFERENCE_TRAINING_ASSIGNMENTS)
            or any(type(value) is not int or value <= 0
                   for value in quotas_per_source.values())):
        raise BroadQaExternalDataError(
            "source inference training 配额非法")
    used_titles = set()
    selected = []
    for assignment in SOURCE_INFERENCE_TRAINING_ASSIGNMENTS:
        quota = quotas_per_source[assignment]
        for source in SOURCE_INFERENCE_TRAINING_SOURCES:
            picked = 0
            for record in sorted(
                    buckets[assignment][source],
                    key=lambda item: str(item["item_id"])):
                title_key = str(record["title_key"])
                if title_key in used_titles:
                    continue
                selected.append(record)
                used_titles.add(title_key)
                picked += 1
                if picked == quota:
                    break
            if picked != quota:
                raise BroadQaExternalDataError(
                    "source inference training 来源/机械桶库存不足: "
                    f"{source}/{assignment}")
    return tuple(selected)


def freeze_source_inference_training_roster(
        items: Iterable[ExternalQaItem],
        *,
        run_root: str | Path,
        candidates_path: str | Path,
        source_census_path: str | Path,
        consumed_source_target_paths: Iterable[str | Path],
        excluded_discovery_roster_paths: Iterable[str | Path],
        target_dir: str | Path,
        source_report: dict[str, object],
        extractive_per_source: int = 20,
        passage_coverage_per_source: int = 20,
        raw_wikitext_per_source: int = 10,
        non_extractive_per_source: int = 30,
        ) -> dict[str, object]:
    """冻结独立训练总体及未来评测必须消费的双互斥清单。"""
    root = Path(run_root).resolve()
    if not root.is_dir():
        raise BroadQaExternalDataError(
            "source inference training run root 不存在")
    candidate_file = _within(
        root, candidates_path, label="candidates_path")
    census_file = _within(
        root, source_census_path, label="source_census_path")
    target = _within(root, target_dir, label="target_dir")
    consumed_paths = tuple(
        _within(root, path, label="consumed_source_target")
        for path in consumed_source_target_paths)
    discovery_paths = tuple(
        _within(root, path, label="excluded_discovery_roster")
        for path in excluded_discovery_roster_paths)
    if not consumed_paths or not discovery_paths or target.exists():
        raise BroadQaExternalDataError(
            "source inference training 路径边界非法")

    quotas = {
        "EXTRACTIVE_REFERENCE": extractive_per_source,
        "PASSAGE_COVERAGE_REVIEW": passage_coverage_per_source,
        "RAW_WIKITEXT_REVIEW": raw_wikitext_per_source,
        "NON_EXTRACTIVE_DISCOVERY": non_extractive_per_source,
    }
    if any(type(value) is not int or value <= 0 for value in quotas.values()):
        raise BroadQaExternalDataError(
            "source inference training 配额非法")

    candidates = read_source_alignment_candidates(candidate_file)
    census = read_source_alignment_census(census_file)
    candidate_by_id = {str(value["item_id"]): value for value in candidates}
    census_by_id = {str(value["item_id"]): value for value in census}
    official_by_id = {item.item_id: item for item in items}
    if (set(candidate_by_id) != set(census_by_id)
            or not set(candidate_by_id).issubset(official_by_id)):
        raise BroadQaExternalDataError(
            "source inference training 输入 inventory 漂移")

    excluded_titles = set()
    for path in consumed_paths:
        excluded_titles.update(read_joint_source_targets(path))
    excluded_items = set()
    for path in discovery_paths:
        for item_id, title_key in _read_excluded_roster(path):
            excluded_items.add(item_id)
            excluded_titles.add(title_key)
    if not excluded_titles or not excluded_items:
        raise BroadQaExternalDataError(
            "source inference training 排除 inventory 为空")

    buckets: dict[str, dict[str, list[dict[str, object]]]] = {
        assignment: {source: [] for source in SOURCE_INFERENCE_TRAINING_SOURCES}
        for assignment in SOURCE_INFERENCE_TRAINING_ASSIGNMENTS
    }
    ignored_status_counts: Counter[str] = Counter()
    excluded_title_item_count = 0
    excluded_item_id_count = 0
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
                "source inference training official/census binding 漂移")
        if item_id in excluded_items:
            excluded_item_id_count += 1
            continue
        if official.title_key in excluded_titles:
            excluded_title_item_count += 1
            continue
        assignment = _STATUS_ASSIGNMENTS.get(str(source_record["status"]))
        if assignment is None:
            ignored_status_counts[str(source_record["status"])] += 1
            continue
        if official.source_key not in SOURCE_INFERENCE_TRAINING_SOURCES:
            raise BroadQaExternalDataError(
                "source inference training 来源域漂移")
        buckets[assignment][official.source_key].append({
            "format_version": 1,
            "item_id": item_id,
            "question_sha256": hashlib.sha256(
                official.question.encode("utf-8")).hexdigest(),
            "record_kind": SOURCE_INFERENCE_TRAINING_RECORD_KIND,
            "source_alignment_status": source_record["status"],
            "source_key": official.source_key,
            "terminal_page_id": source_record["terminal_page_id"],
            "terminal_revision_id": source_record["terminal_revision_id"],
            "title_key": official.title_key,
            "training_assignment": assignment,
        })

    selected = _select_training_records(
        buckets, quotas_per_source=quotas)
    expected_count = sum(quotas.values()) * len(
        SOURCE_INFERENCE_TRAINING_SOURCES)
    if (len(selected) != expected_count
            or len({str(item["title_key"]) for item in selected})
            != expected_count
            or len({str(item["item_id"]) for item in selected})
            != expected_count):
        raise BroadQaExternalDataError(
            "source inference training selection 未闭合")

    target.mkdir(parents=True)
    roster_path = target / "train.roster.jsonl"
    payload_path = target / "train.payload.jsonl"
    exclusion_path = target / "evaluation-exclusion.inventory.jsonl"
    roster_count = _write_jsonl(roster_path, selected)
    payload_count = _write_jsonl(payload_path, ({
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
        "record_kind": SOURCE_INFERENCE_TRAINING_PAYLOAD_KIND,
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
        "training_assignment": record["training_assignment"],
        "upstream_url": official_by_id[str(record["item_id"])].upstream_url,
    } for record in selected))
    exclusion_count = _write_jsonl(exclusion_path, ({
        "format_version": 1,
        "item_id": record["item_id"],
        "record_kind": SOURCE_INFERENCE_TRAINING_EXCLUSION_KIND,
        "title_key": record["title_key"],
    } for record in selected))
    if not roster_count == payload_count == exclusion_count == expected_count:
        raise BroadQaExternalDataError(
            "source inference training artifact count 漂移")

    assignment_counts = Counter(
        str(item["training_assignment"]) for item in selected)
    source_assignment_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for record in selected:
        source_assignment_counts[str(record["source_key"])][
            str(record["training_assignment"])] += 1
    manifest = {
        "artifact_kind": SOURCE_INFERENCE_TRAINING_KIND,
        "artifacts": [
            {
                "bytes": roster_path.stat().st_size,
                "record_count": roster_count,
                "role": "training_roster_without_semantic_labels",
                "sha256": _sha256_file(roster_path),
            },
            {
                "bytes": payload_path.stat().st_size,
                "record_count": payload_count,
                "role": "training_payload_frozen_before_learner_read",
                "sha256": _sha256_file(payload_path),
            },
            {
                "bytes": exclusion_path.stat().st_size,
                "record_count": exclusion_count,
                "role": "future_evaluation_item_title_exclusion",
                "sha256": _sha256_file(exclusion_path),
            },
        ],
        "assignment_counts": {
            assignment: assignment_counts[assignment]
            for assignment in SOURCE_INFERENCE_TRAINING_ASSIGNMENTS
        },
        "candidate_count": len(candidates),
        "candidates_sha256": _sha256_file(candidate_file),
        "excluded_discovery_rosters": [
            {"sha256": _sha256_file(path)} for path in discovery_paths],
        "excluded_item_id_match_count": excluded_item_id_count,
        "excluded_item_inventory_count": len(excluded_items),
        "excluded_source_targets": [
            {"sha256": _sha256_file(path)} for path in consumed_paths],
        "excluded_title_count": len(excluded_titles),
        "excluded_title_match_count": excluded_title_item_count,
        "format_version": 1,
        "future_evaluation_contract": {
            "exclusion_inventory_required": 1,
            "item_overlap_allowed": 0,
            "title_overlap_allowed": 0,
            "evaluation_freeze_after_rule_pack_only": 1,
        },
        "ignored_status_counts": dict(sorted(ignored_status_counts.items())),
        "learner_read_count_at_freeze": 0,
        "operator_discovery_contract": {
            "allowed_operator_families": list(
                SOURCE_INFERENCE_OPERATOR_FAMILIES),
            "defeater_required": 1,
            "negative_examples_required": 1,
            "operator_family_preassigned_count": 0,
            "positive_examples_required": 1,
            "rule_evidence_record_required": 1,
            "scope_required": 1,
            "unlisted_operator_allowed": 0,
        },
        "question_count": expected_count,
        "selection_rule": SOURCE_INFERENCE_TRAINING_SELECTION_RULE,
        "semantic_labels_written": 0,
        "source_assignment_counts": {
            source: {
                assignment: values[assignment]
                for assignment in SOURCE_INFERENCE_TRAINING_ASSIGNMENTS
            }
            for source, values in sorted(source_assignment_counts.items())
        },
        "source_census_sha256": _sha256_file(census_file),
        "source_counts": {
            source: sum(source_assignment_counts[source].values())
            for source in SOURCE_INFERENCE_TRAINING_SOURCES
        },
        "source_report": source_report,
        "status": "FROZEN_NOT_READ_NOT_LEARNED",
        "title_count": expected_count,
    }
    manifest_path = target / "manifest.json"
    manifest_path.write_bytes(canonical_json_line(manifest))
    return {**manifest, "manifest_sha256": _sha256_file(manifest_path)}


def read_source_inference_training_exclusions(
        path: str | Path,
        ) -> tuple[tuple[str, str], ...]:
    """严格回读未来评测必须排除的 training item/title 对。"""
    target = Path(path).resolve()
    values = []
    item_ids = set()
    title_keys = set()
    try:
        with target.open("r", encoding="utf-8", newline="") as handle:
            for line_number, line in enumerate(handle, start=1):
                value = json.loads(line)
                if (not line.endswith("\n") or not isinstance(value, dict)
                        or canonical_json_line(value) != line.encode("utf-8")
                        or set(value) != {
                            "format_version", "item_id", "record_kind",
                            "title_key"}
                        or value.get("format_version") != 1
                        or value.get("record_kind")
                        != SOURCE_INFERENCE_TRAINING_EXCLUSION_KIND
                        or not isinstance(value.get("item_id"), str)
                        or len(str(value["item_id"])) != 64
                        or any(character not in "0123456789abcdef"
                               for character in str(value["item_id"]))
                        or value["item_id"] in item_ids
                        or not isinstance(value.get("title_key"), str)
                        or not value["title_key"]
                        or normalize_external_text(str(value["title_key"]))
                        != value["title_key"]
                        or value["title_key"] in title_keys):
                    raise BroadQaExternalDataError(
                        f"source inference training exclusion 漂移: "
                        f"{line_number}")
                item_ids.add(str(value["item_id"]))
                title_keys.add(str(value["title_key"]))
                values.append((str(value["item_id"]), str(value["title_key"])))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "source inference training exclusion 不可读") from error
    if not values:
        raise BroadQaExternalDataError(
            "source inference training exclusion 为空")
    return tuple(values)


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
    """要求大数据输入输出使用显式绝对 K 盘路径。"""
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("work paths must be absolute")
    resolved = path.resolve()
    if sys.platform == "win32" and resolved.drive.casefold() != "k:":
        raise argparse.ArgumentTypeError("work paths must be on K:")
    return resolved


def main(argv: list[str] | None = None) -> int:
    """从官方来源与冻结 census 发布独立训练 roster。"""
    parser = argparse.ArgumentParser(
        description="Freeze an independent source-inference training roster.")
    parser.add_argument("--run-root", type=_work_path, required=True)
    parser.add_argument("--cmrc-root", type=_work_path, required=True)
    parser.add_argument("--drcd-root", type=_work_path, required=True)
    parser.add_argument("--candidates", type=_work_path, required=True)
    parser.add_argument("--source-census", type=_work_path, required=True)
    parser.add_argument(
        "--consumed-source-target", type=_work_path,
        action="append", required=True)
    parser.add_argument(
        "--excluded-discovery-roster", type=_work_path,
        action="append", required=True)
    parser.add_argument("--target-dir", type=_work_path, required=True)
    parser.add_argument("--extractive-per-source", type=_positive, default=20)
    parser.add_argument(
        "--passage-coverage-per-source", type=_positive, default=20)
    parser.add_argument(
        "--raw-wikitext-per-source", type=_positive, default=10)
    parser.add_argument(
        "--non-extractive-per-source", type=_positive, default=30)
    args = parser.parse_args(argv)
    sources = official_external_qa_sources(args.cmrc_root, args.drcd_root)
    items, source_report = load_external_qa_sources(sources)
    report = freeze_source_inference_training_roster(
        items,
        run_root=args.run_root,
        candidates_path=args.candidates,
        source_census_path=args.source_census,
        consumed_source_target_paths=args.consumed_source_target,
        excluded_discovery_roster_paths=args.excluded_discovery_roster,
        target_dir=args.target_dir,
        source_report=source_report,
        extractive_per_source=args.extractive_per_source,
        passage_coverage_per_source=args.passage_coverage_per_source,
        raw_wikitext_per_source=args.raw_wikitext_per_source,
        non_extractive_per_source=args.non_extractive_per_source,
    )
    sys.stdout.write(json.dumps(
        report, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SOURCE_INFERENCE_OPERATOR_FAMILIES",
    "SOURCE_INFERENCE_TRAINING_ASSIGNMENTS",
    "SOURCE_INFERENCE_TRAINING_EXCLUSION_KIND",
    "SOURCE_INFERENCE_TRAINING_KIND",
    "SOURCE_INFERENCE_TRAINING_PAYLOAD_KIND",
    "SOURCE_INFERENCE_TRAINING_RECORD_KIND",
    "SOURCE_INFERENCE_TRAINING_SELECTION_RULE",
    "freeze_source_inference_training_roster",
    "main",
    "read_source_inference_training_exclusions",
]
