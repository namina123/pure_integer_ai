"""在问答运行前冻结外部题目与 Wikipedia 终页的版本对齐合同。"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Iterable

from pure_integer_ai.experiments.ph2_broad_qa_contract import (
    BroadQaTargetSelectionManifest,
)
from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
    ExternalQaItem,
    external_title_surfaces,
    normalize_external_text,
)
from pure_integer_ai.experiments.ph2_broad_qa_joint_eval import (
    JOINT_TARGET_KIND,
    read_joint_source_aliases,
)
from pure_integer_ai.experiments.ph2_broad_qa_source import (
    iter_broad_qa_selected_page_inspections,
    project_broad_qa_passages,
    project_broad_qa_plain_text,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


SOURCE_ALIGNMENT_CANDIDATE_PACK_KIND = (
    "PH2_BROAD_QA_SOURCE_ALIGNMENT_CANDIDATE_PACK_V1"
)
SOURCE_ALIGNMENT_CANDIDATE_KIND = (
    "PH2_BROAD_QA_SOURCE_ALIGNMENT_CANDIDATE_V1"
)
SOURCE_ALIGNMENT_CENSUS_KIND = "PH2_BROAD_QA_SOURCE_ALIGNMENT_CENSUS_V1"
SOURCE_ALIGNMENT_CENSUS_RECORD_KIND = (
    "PH2_BROAD_QA_SOURCE_ALIGNMENT_CENSUS_RECORD_V1"
)
SOURCE_ALIGNMENT_SELECTION_RULE = (
    "EXCLUDE_CONSUMED_TITLES_THEN_NATURAL_TITLE_ANCHOR_"
    "THEN_SOURCE_COVERAGE_CENSUS_THEN_TITLE_BUCKET_ITEM_SHA256_V1"
)
SOURCE_ALIGNED_STATUS = "SOURCE_ALIGNED"


def _sha256_file(path: Path) -> str:
    """流式计算不可覆盖 artifact 的 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _write_jsonl(path: Path, records: Iterable[dict[str, object]]) -> int:
    """不可覆盖地写规范 JSONL，并返回记录数量。"""
    if path.exists():
        raise BroadQaExternalDataError(
            f"source alignment artifact 禁止覆盖: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("xb") as handle:
        for record in records:
            handle.write(canonical_json_line(record))
            count += 1
    return count


def _read_jsonl(path: Path, *, record_kind: str) -> tuple[dict, ...]:
    """严格回读规范换行、唯一 item id 的 JSONL。"""
    if not path.is_file():
        raise BroadQaExternalDataError(
            f"source alignment input 缺失: {path.name}")
    values = []
    identities = set()
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.endswith("\n") or not line.strip():
                    raise BroadQaExternalDataError(
                        f"source alignment JSONL 换行非法: {line_number}")
                value = json.loads(line)
                identity = value.get("item_id") if isinstance(value, dict) else None
                if (not isinstance(value, dict)
                        or value.get("record_kind") != record_kind
                        or not isinstance(identity, str) or not identity
                        or identity in identities):
                    raise BroadQaExternalDataError(
                        f"source alignment record 漂移: {line_number}")
                identities.add(identity)
                values.append(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "source alignment JSONL 非法") from error
    if not values:
        raise BroadQaExternalDataError("source alignment JSONL 为空")
    return tuple(values)


def read_consumed_title_keys(
        *,
        prior_question_paths: Iterable[str | Path],
        prior_source_target_paths: Iterable[str | Path],
        ) -> tuple[str, ...]:
    """只从已消费 questions/target ledger 汇总标题域，不读取标签。"""
    keys = set()
    for raw_path in prior_question_paths:
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise BroadQaExternalDataError("consumed question input 缺失")
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                for line in handle:
                    value = json.loads(line)
                    title = value.get("title") if isinstance(value, dict) else None
                    if (not isinstance(title, str) or not title
                            or value.get("record_kind")
                            != "PH2_BROAD_QA_EXTERNAL_QUESTION_V1"):
                        raise BroadQaExternalDataError(
                            "consumed question schema 漂移")
                    keys.add(normalize_external_text(title))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise BroadQaExternalDataError(
                "consumed question JSONL 非法") from error
    for raw_path in prior_source_target_paths:
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise BroadQaExternalDataError("consumed source target 缺失")
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                for line in handle:
                    value = json.loads(line)
                    key = value.get("title_key") if isinstance(value, dict) else None
                    if (not isinstance(key, str) or not key
                            or normalize_external_text(key) != key
                            or value.get("record_kind") != JOINT_TARGET_KIND):
                        raise BroadQaExternalDataError(
                            "consumed source target schema 漂移")
                    keys.add(key)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise BroadQaExternalDataError(
                "consumed source target JSONL 非法") from error
    if not keys:
        raise BroadQaExternalDataError("consumed title inventory 为空")
    return tuple(sorted(keys))


def select_source_alignment_candidates(
        items: Iterable[ExternalQaItem],
        *,
        excluded_title_keys: Iterable[str],
        ) -> tuple[ExternalQaItem, ...]:
    """冻结自然含标题且未进入任何已消费 family 的完整候选总体。"""
    excluded = frozenset(excluded_title_keys)
    if (not excluded or any(not isinstance(item, str) or not item
                            for item in excluded)):
        raise BroadQaExternalDataError("source alignment excluded titles 非法")
    selected = tuple(sorted((
        item for item in items
        if item.title_key not in excluded
        and item.title_key in normalize_external_text(item.question)
    ), key=lambda item: item.item_id))
    if not selected or len({item.item_id for item in selected}) != len(selected):
        raise BroadQaExternalDataError("source alignment candidate inventory 非法")
    return selected


def freeze_source_alignment_candidates(
        items: Iterable[ExternalQaItem],
        *,
        excluded_title_keys: Iterable[str],
        excluded_title_source_paths: Iterable[str | Path] = (),
        target_dir: str | Path,
        source_report: dict[str, object],
        ) -> dict[str, object]:
    """发布完整候选、标题目标和 manifest，作为来源扫描唯一输入。"""
    excluded = tuple(sorted(set(excluded_title_keys)))
    excluded_sources = tuple(
        Path(item).resolve() for item in excluded_title_source_paths)
    if (not excluded_sources
            or any(not item.is_file() for item in excluded_sources)):
        raise BroadQaExternalDataError(
            "source alignment excluded source inventory 非法")
    candidates = select_source_alignment_candidates(
        items, excluded_title_keys=excluded)
    target = Path(target_dir).resolve()
    if target.exists():
        raise BroadQaExternalDataError("source alignment target 已存在")
    target.mkdir(parents=True)
    candidate_path = target / "candidates.jsonl"
    count = _write_jsonl(candidate_path, ({
        "format_version": 1,
        "gold_answers": list(item.gold_answers),
        "item_id": item.item_id,
        "license_id": item.license_id,
        "record_kind": SOURCE_ALIGNMENT_CANDIDATE_KIND,
        "source_key": item.source_key,
        "source_partition": item.source_partition,
        "source_question_id": item.source_question_id,
        "source_revision": item.source_revision,
        "title": item.title,
        "title_key": item.title_key,
        "upstream_url": item.upstream_url,
    } for item in candidates))
    surfaces: dict[str, set[str]] = defaultdict(set)
    for item in candidates:
        surfaces[item.title_key].update(external_title_surfaces(item.title))
    target_path = target / "source_targets.jsonl"
    target_count = _write_jsonl(target_path, ({
        "format_version": 1,
        "record_kind": JOINT_TARGET_KIND,
        "surfaces": sorted(surfaces[key]),
        "title_key": key,
    } for key in sorted(surfaces)))
    excluded_payload = canonical_json_line({"title_keys": list(excluded)})
    manifest = {
        "artifact_kind": SOURCE_ALIGNMENT_CANDIDATE_PACK_KIND,
        "artifacts": [
            {"bytes": candidate_path.stat().st_size,
             "record_count": count, "role": "candidates",
             "sha256": _sha256_file(candidate_path)},
            {"bytes": target_path.stat().st_size,
             "record_count": target_count, "role": "source_targets",
             "sha256": _sha256_file(target_path)},
        ],
        "candidate_count": count,
        "excluded_title_count": len(excluded),
        "excluded_title_sources": [
            {"sha256": _sha256_file(path)} for path in excluded_sources],
        "excluded_titles_sha256": hashlib.sha256(
            excluded_payload).hexdigest(),
        "format_version": 1,
        "selection_rule": SOURCE_ALIGNMENT_SELECTION_RULE,
        "source_counts": dict(sorted(Counter(
            item.source_key for item in candidates).items())),
        "source_report": source_report,
        "source_target_count": target_count,
        "status": "FROZEN_NOT_SCANNED",
    }
    manifest_path = target / "manifest.json"
    manifest_path.write_bytes(canonical_json_line(manifest))
    return {**manifest, "manifest_sha256": _sha256_file(manifest_path)}


def read_source_alignment_candidates(path: str | Path) -> tuple[dict, ...]:
    """严格回读候选记录，并核验答案和标题身份。"""
    values = _read_jsonl(
        Path(path).resolve(), record_kind=SOURCE_ALIGNMENT_CANDIDATE_KIND)
    expected = {
        "format_version", "gold_answers", "item_id", "license_id",
        "record_kind", "source_key", "source_partition",
        "source_question_id", "source_revision", "title", "title_key",
        "upstream_url",
    }
    for value in values:
        if (set(value) != expected or value["format_version"] != 1
                or not isinstance(value["gold_answers"], list)
                or not value["gold_answers"]
                or any(not isinstance(item, str) or not item
                       for item in value["gold_answers"])
                or any(not normalize_external_text(item)
                       for item in value["gold_answers"])
                or normalize_external_text(value["title"])
                != value["title_key"]):
            raise BroadQaExternalDataError(
                "source alignment candidate schema 漂移")
    return values


def _normalized_contains(
        normalized_text: str,
        answers: Iterable[str],
        ) -> bool:
    """核验已规范化来源文本是否包含任一规范化金答案。"""
    if not isinstance(normalized_text, str):
        raise TypeError("normalized source text 必须是字符串")
    return any(
        normalize_external_text(answer) in normalized_text
        for answer in answers)


def _source_alignment_status(
        *, raw_hit: bool, full_text_hit: bool, projected_passage_hit: bool,
        ) -> str:
    """把 raw、整页可见文本与索引 passage 的覆盖组合归为唯一状态。"""
    if any(type(item) is not bool for item in (
            raw_hit, full_text_hit, projected_passage_hit)):
        raise TypeError("source alignment hit flags 必须为 bool")
    if full_text_hit and projected_passage_hit:
        return SOURCE_ALIGNED_STATUS
    if full_text_hit:
        return "GOLD_PRESENT_OUTSIDE_PROJECTED_PASSAGES"
    if projected_passage_hit:
        return "PASSAGE_PROJECTION_DIVERGES_FROM_FULL_PAGE"
    if raw_hit:
        return "GOLD_ONLY_IN_RAW_WIKITEXT"
    return "GOLD_ABSENT_FROM_TERMINAL_REVISION"


def build_source_alignment_census(
        candidates_path: str | Path,
        aliases_path: str | Path,
        terminal_selection: BroadQaTargetSelectionManifest,
        *,
        xml_path: str | Path,
        census_path: str | Path,
        manifest_path: str | Path,
        worker_count: int = 4,
        ) -> dict[str, object]:
    """扫描一次终页，分账标题缺失、终页缺失与投影可答覆盖。"""
    candidates_file = Path(candidates_path).resolve()
    aliases_file = Path(aliases_path).resolve()
    output = Path(census_path).resolve()
    manifest_output = Path(manifest_path).resolve()
    if (output.exists() or manifest_output.exists()
            or not isinstance(terminal_selection,
                              BroadQaTargetSelectionManifest)):
        raise BroadQaExternalDataError("source alignment census 路径非法")
    candidates = read_source_alignment_candidates(candidates_file)
    aliases = {
        item["title_key"]: item
        for item in read_joint_source_aliases(aliases_file)
    }
    if set(aliases) != {item["title_key"] for item in candidates}:
        raise BroadQaExternalDataError("source alignment alias inventory 漂移")
    page_texts = {}
    for inspection in iter_broad_qa_selected_page_inspections(
            terminal_selection.selected_pages,
            xml_path=xml_path,
            source_key=terminal_selection.source_key,
            xml_compressed_size_bytes=(
                terminal_selection.xml_compressed_size_bytes),
            worker_count=worker_count):
        if inspection.redirect_title:
            raise BroadQaExternalDataError(
                "source alignment terminal page 仍为 redirect")
        passages = project_broad_qa_passages(inspection.wikitext)
        page_texts[inspection.page_id] = (
            inspection.revision_id,
            normalize_external_text(inspection.wikitext),
            normalize_external_text(
                project_broad_qa_plain_text(inspection.wikitext)),
            normalize_external_text(
                "\n".join(item.text for item in passages)),
        )
    expected_pages = {
        item["terminal_page_id"] for item in aliases.values()
        if item["status"] == "RESOLVED"
    }
    if set(page_texts) != expected_pages:
        raise BroadQaExternalDataError(
            "source alignment terminal page inventory 未闭合")
    records = []
    status_counts: Counter[str] = Counter()
    source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for candidate in candidates:
        alias = aliases[candidate["title_key"]]
        base = {
            "format_version": 1,
            "item_id": candidate["item_id"],
            "record_kind": SOURCE_ALIGNMENT_CENSUS_RECORD_KIND,
            "source_key": candidate["source_key"],
            "title_key": candidate["title_key"],
        }
        if alias["status"] != "RESOLVED":
            record = {
                **base,
                "failure_code": alias["failure_code"],
                "status": "SOURCE_ALIAS_MISSING",
            }
        else:
            page_id = alias["terminal_page_id"]
            revision_id, raw_text, full_text, projected_text = page_texts[
                page_id]
            if revision_id != alias["terminal_revision_id"]:
                raise BroadQaExternalDataError(
                    "source alignment terminal revision 漂移")
            raw_hit = _normalized_contains(
                raw_text, candidate["gold_answers"])
            full_hit = _normalized_contains(
                full_text, candidate["gold_answers"])
            projected_hit = _normalized_contains(
                projected_text, candidate["gold_answers"])
            status = _source_alignment_status(
                raw_hit=raw_hit, full_text_hit=full_hit,
                projected_passage_hit=projected_hit)
            record = {
                **base,
                "status": status,
                "terminal_page_id": page_id,
                "terminal_revision_id": revision_id,
                "terminal_title": alias["terminal_title"],
            }
        records.append(record)
        status_counts[record["status"]] += 1
        source_counts[candidate["source_key"]][record["status"]] += 1
        source_counts[candidate["source_key"]]["candidate_count"] += 1
    count = _write_jsonl(output, records)
    aligned_count = status_counts[SOURCE_ALIGNED_STATUS]
    manifest = {
        "alias_sha256": _sha256_file(aliases_file),
        "artifact_kind": SOURCE_ALIGNMENT_CENSUS_KIND,
        "candidate_count": count,
        "candidates_sha256": _sha256_file(candidates_file),
        "census_bytes": output.stat().st_size,
        "census_sha256": _sha256_file(output),
        "format_version": 1,
        "source_aligned_count": aligned_count,
        "source_aligned_ppm": aligned_count * 1_000_000 // count,
        "per_source": {
            key: dict(sorted(values.items()))
            for key, values in sorted(source_counts.items())
        },
        "status": "FROZEN_NOT_USED_FOR_QA",
        "status_counts": dict(sorted(status_counts.items())),
        "terminal_selection_sha256": terminal_selection.sha256(),
    }
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_bytes(canonical_json_line(manifest))
    return {
        **manifest,
        "manifest_sha256": _sha256_file(manifest_output),
    }


def read_source_alignment_census(path: str | Path) -> tuple[dict, ...]:
    """严格回读来源版本对齐 census。"""
    values = _read_jsonl(
        Path(path).resolve(), record_kind=SOURCE_ALIGNMENT_CENSUS_RECORD_KIND)
    common = {
        "format_version", "item_id", "record_kind", "source_key",
        "status", "title_key",
    }
    for value in values:
        status = value.get("status")
        expected = (
            common | {"failure_code"}
            if status == "SOURCE_ALIAS_MISSING"
            else common | {
                "terminal_page_id", "terminal_revision_id", "terminal_title"}
        )
        if (set(value) != expected or value["format_version"] != 1
                or status not in {
                    SOURCE_ALIGNED_STATUS,
                    "SOURCE_ALIAS_MISSING",
                    "GOLD_PRESENT_OUTSIDE_PROJECTED_PASSAGES",
                    "PASSAGE_PROJECTION_DIVERGES_FROM_FULL_PAGE",
                    "GOLD_ONLY_IN_RAW_WIKITEXT",
                    "GOLD_ABSENT_FROM_TERMINAL_REVISION",
                }):
            raise BroadQaExternalDataError(
                "source alignment census schema 漂移")
    return values


__all__ = [
    "SOURCE_ALIGNED_STATUS",
    "SOURCE_ALIGNMENT_CANDIDATE_KIND",
    "SOURCE_ALIGNMENT_CANDIDATE_PACK_KIND",
    "SOURCE_ALIGNMENT_CENSUS_KIND",
    "SOURCE_ALIGNMENT_CENSUS_RECORD_KIND",
    "SOURCE_ALIGNMENT_SELECTION_RULE",
    "build_source_alignment_census",
    "freeze_source_alignment_candidates",
    "read_consumed_title_keys",
    "read_source_alignment_candidates",
    "read_source_alignment_census",
    "select_source_alignment_candidates",
]
