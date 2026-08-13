"""冻结未消费来源对齐问题的交互维度 census。"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sysconfig
from typing import Iterable

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
    ExternalQaItem,
    external_title_surfaces,
)
from pure_integer_ai.experiments.ph2_broad_qa_joint_eval import (
    JOINT_LABEL_KIND,
    JOINT_QUESTION_KIND,
    JOINT_TARGET_KIND,
    JOINT_THRESHOLDS,
    read_joint_source_targets,
)
from pure_integer_ai.experiments.ph2_broad_qa_question_slots import (
    load_broad_qa_question_slots,
)
from pure_integer_ai.experiments.ph2_broad_qa_source_alignment import (
    SOURCE_ALIGNED_STATUS,
    read_source_alignment_candidates,
    read_source_alignment_census,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
)


REPOSITORY = Path(__file__).resolve().parents[3]
DIMENSION_RELATIVE_PATH = Path(
    "data/ph2/broad_qa_interactive_dimensions_v1.json")
DIMENSION_ARTIFACT_PATH = REPOSITORY / DIMENSION_RELATIVE_PATH
DIMENSION_DISTRIBUTION_SUBDIRECTORY = Path("share/pure_integer_ai")
DIMENSION_ARTIFACT_SHA256 = (
    "ddfc63c9bdac62d71f74d75148459975276521e1059b0b7846b95f33f301118e")
INTERACTIVE_CENSUS_KIND = "PH2_BROAD_QA_INTERACTIVE_DIMENSION_CENSUS_V1"
INTERACTIVE_CENSUS_RECORD_KIND = (
    "PH2_BROAD_QA_INTERACTIVE_DIMENSION_CENSUS_RECORD_V1"
)
INTERACTIVE_CENSUS_RULE = (
    "SOURCE_ALIGNED_MINUS_CONSUMED_TITLE_THEN_PRIMARY_DIMENSION_V1"
)
INTERACTIVE_DEVELOPMENT_PACK_KIND = (
    "PH2_BROAD_QA_INTERACTIVE_DEVELOPMENT_PACK_V1"
)
INTERACTIVE_DIMENSION_RECORD_KIND = (
    "PH2_BROAD_QA_INTERACTIVE_DEVELOPMENT_DIMENSION_V1"
)
INTERACTIVE_DEVELOPMENT_SELECTION_RULE = (
    "PRIMARY_DIMENSION_THEN_AVAILABLE_SOURCE_BALANCE_THEN_"
    "UNIQUE_TITLE_THEN_ITEM_SHA256_V1"
)
_DIMENSIONS = ("CAUSE", "COMPARISON", "TIME", "QUANTITY", "RELATION")


def _sha256_file(path: Path) -> str:
    """流式计算冻结输入或输出文件的 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _dimension_artifact_path(path: str | Path | None) -> Path:
    """解析 checkout 或安装后 data scheme 中的冻结规则。"""
    if path is not None:
        return Path(path).resolve()
    roots = [REPOSITORY]
    data_root = sysconfig.get_path("data")
    if data_root:
        roots.append(Path(data_root) / DIMENSION_DISTRIBUTION_SUBDIRECTORY)
    return next(
        (candidate for root in roots
         if (candidate := (root / DIMENSION_RELATIVE_PATH).resolve()).is_file()),
        DIMENSION_ARTIFACT_PATH,
    )


def load_interactive_dimension_rules(
        path: str | Path | None = None,
        ) -> dict[str, object]:
    """严格加载公开 CC0 交互维度规则 artifact。"""
    source = _dimension_artifact_path(path)
    try:
        payload = source.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError("interactive dimension artifact 非法") from error
    if (hashlib.sha256(payload).hexdigest() != DIMENSION_ARTIFACT_SHA256
            or not isinstance(value, dict)
            or canonical_json_line(value) != payload
            or set(value) != {
                "artifact_kind", "comparison_surfaces", "format_version",
                "license_id", "primary_dimension_order", "source_identity"}
            or value["artifact_kind"]
            != "PH2_BROAD_QA_INTERACTIVE_DIMENSIONS_V1"
            or value["format_version"] != 1
            or value["license_id"] != "CC0-1.0"
            or value["primary_dimension_order"] != list(_DIMENSIONS)
            or not isinstance(value["comparison_surfaces"], list)
            or not value["comparison_surfaces"]
            or any(not isinstance(item, str) or not item
                   for item in value["comparison_surfaces"])):
        raise BroadQaExternalDataError("interactive dimension artifact 漂移")
    return value


def classify_interactive_dimension(
        question: str,
        *, rules: dict[str, object] | None = None,
        ) -> str:
    """按冻结优先序把问题归入一个主交互维度。"""
    if not isinstance(question, str) or not question.strip():
        raise BroadQaExternalDataError("interactive dimension question 非法")
    value = load_interactive_dimension_rules() if rules is None else rules
    if (not isinstance(value, dict)
            or value.get("primary_dimension_order") != list(_DIMENSIONS)):
        raise BroadQaExternalDataError("interactive dimension rules 非法")
    answer_kinds = set(load_broad_qa_question_slots().answer_kinds(question))
    if "CAUSE" in answer_kinds:
        return "CAUSE"
    if any(surface in question for surface in value["comparison_surfaces"]):
        return "COMPARISON"
    if "TIME" in answer_kinds:
        return "TIME"
    if "QUANTITY" in answer_kinds:
        return "QUANTITY"
    return "RELATION"


def _write_census(
        path: Path,
        records: Iterable[dict[str, object]],
        ) -> int:
    """不可覆盖地发布规范 census JSONL。"""
    if path.exists():
        raise BroadQaExternalDataError("interactive census 禁止覆盖")
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("xb") as handle:
        for record in records:
            handle.write(canonical_json_line(record))
            count += 1
    return count


def _read_interactive_census(path: Path) -> tuple[dict[str, object], ...]:
    """严格回读主维度 census，拒绝 schema 或身份漂移。"""
    if not path.is_file():
        raise BroadQaExternalDataError("interactive census 缺失")
    values = []
    identities = set()
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for line_number, line in enumerate(handle, start=1):
                value = json.loads(line)
                expected = {
                    "dimension", "format_version", "item_id",
                    "question_sha256", "record_kind", "source_key",
                    "terminal_page_id", "terminal_revision_id", "title_key",
                }
                identity = value.get("item_id") if isinstance(value, dict) else None
                if (not line.endswith("\n") or not isinstance(value, dict)
                        or set(value) != expected
                        or value["format_version"] != 1
                        or value["record_kind"] != INTERACTIVE_CENSUS_RECORD_KIND
                        or value["dimension"] not in _DIMENSIONS
                        or not isinstance(identity, str) or not identity
                        or identity in identities
                        or not isinstance(value["question_sha256"], str)
                        or len(value["question_sha256"]) != 64):
                    raise BroadQaExternalDataError(
                        f"interactive census record 漂移: {line_number}")
                identities.add(identity)
                values.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError("interactive census 非法") from error
    if not values:
        raise BroadQaExternalDataError("interactive census 为空")
    return tuple(values)


def _balanced_dimension_selection(
        records: Iterable[dict[str, object]],
        *, dimension_quota: int,
        ) -> tuple[dict[str, object], ...]:
    """按有库存的来源等额选择，并在全部维度间保持标题唯一。"""
    if type(dimension_quota) is not int or dimension_quota <= 0:
        raise BroadQaExternalDataError("interactive dimension quota 非法")
    by_dimension_source: dict[
        str, dict[str, list[dict[str, object]]]
    ] = defaultdict(lambda: defaultdict(list))
    for record in records:
        by_dimension_source[str(record["dimension"])][
            str(record["source_key"])].append(record)
    selected = []
    used_titles = set()
    for dimension in _DIMENSIONS:
        source_records = by_dimension_source.get(dimension, {})
        sources = tuple(sorted(
            source for source, values in source_records.items() if values))
        if not sources:
            raise BroadQaExternalDataError(
                f"interactive dimension 无库存: {dimension}")
        base, remainder = divmod(dimension_quota, len(sources))
        source_quotas = {
            source: base + int(ordinal < remainder)
            for ordinal, source in enumerate(sources)
        }
        for source in sources:
            if source_quotas[source] == 0:
                continue
            picked = 0
            for record in sorted(
                    source_records[source], key=lambda item: item["item_id"]):
                title_key = record["title_key"]
                if title_key in used_titles:
                    continue
                selected.append(record)
                used_titles.add(title_key)
                picked += 1
                if picked == source_quotas[source]:
                    break
            if picked != source_quotas[source]:
                raise BroadQaExternalDataError(
                    f"interactive dimension/source 库存不足: {dimension}/{source}")
    return tuple(selected)


def freeze_interactive_development_pack(
        items: Iterable[ExternalQaItem],
        *,
        census_path: str | Path,
        census_manifest_path: str | Path,
        target_dir: str | Path,
        source_report: dict[str, object],
        dimension_quota: int = 20,
        ) -> dict[str, object]:
    """从冻结未消费 census 发布五维正向开发集，不创建 held-out。"""
    census_file = Path(census_path).resolve()
    census_manifest_file = Path(census_manifest_path).resolve()
    target = Path(target_dir).resolve()
    if target.exists():
        raise BroadQaExternalDataError("interactive development target 已存在")
    census_records = _read_interactive_census(census_file)
    try:
        manifest_payload = census_manifest_file.read_bytes()
        census_manifest = json.loads(manifest_payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "interactive census manifest 非法") from error
    if (canonical_json_line(census_manifest) != manifest_payload
            or census_manifest.get("artifact_kind") != INTERACTIVE_CENSUS_KIND
            or census_manifest.get("census_sha256") != _sha256_file(census_file)
            or census_manifest.get("rules_sha256")
            != DIMENSION_ARTIFACT_SHA256
            or census_manifest.get("remaining_source_aligned_count")
            != len(census_records)):
        raise BroadQaExternalDataError(
            "interactive census manifest commitment 漂移")
    item_by_id = {item.item_id: item for item in items}
    if not {record["item_id"] for record in census_records}.issubset(item_by_id):
        raise BroadQaExternalDataError("interactive official inventory 漂移")
    for record in census_records:
        item = item_by_id[record["item_id"]]
        if (item.source_key != record["source_key"]
                or item.title_key != record["title_key"]
                or hashlib.sha256(item.question.encode("utf-8")).hexdigest()
                != record["question_sha256"]):
            raise BroadQaExternalDataError(
                "interactive census official binding 漂移")
    selected_records = _balanced_dimension_selection(
        census_records, dimension_quota=dimension_quota)
    selected = tuple(
        (record, item_by_id[record["item_id"]])
        for record in selected_records)
    target.mkdir(parents=True)
    questions_path = target / "dev.questions.jsonl"
    labels_path = target / "dev.labels.jsonl"
    dimensions_path = target / "dev.dimensions.jsonl"
    targets_path = target / "source_targets.jsonl"
    _write_census(questions_path, ({
        "format_version": 1,
        "item_id": item.item_id,
        "license_id": item.license_id,
        "question": item.question,
        "record_kind": JOINT_QUESTION_KIND,
        "source_key": item.source_key,
        "source_partition": item.source_partition,
        "source_question_id": item.source_question_id,
        "source_revision": item.source_revision,
        "split": "dev",
        "upstream_url": item.upstream_url,
    } for _, item in selected))
    _write_census(labels_path, ({
        "expected_title_key": item.title_key,
        "format_version": 1,
        "gold_answers": list(item.gold_answers),
        "item_id": item.item_id,
        "record_kind": JOINT_LABEL_KIND,
        "split": "dev",
    } for _, item in selected))
    _write_census(dimensions_path, ({
        "dimension": record["dimension"],
        "format_version": 1,
        "item_id": item.item_id,
        "record_kind": INTERACTIVE_DIMENSION_RECORD_KIND,
    } for record, item in selected))
    title_surfaces: dict[str, set[str]] = defaultdict(set)
    for _, item in selected:
        title_surfaces[item.title_key].update(external_title_surfaces(item.title))
    _write_census(targets_path, ({
        "format_version": 1,
        "record_kind": JOINT_TARGET_KIND,
        "surfaces": sorted(title_surfaces[key]),
        "title_key": key,
    } for key in sorted(title_surfaces)))
    paths = (
        ("dev_questions", questions_path),
        ("dev_labels", labels_path),
        ("dev_dimensions", dimensions_path),
        ("source_targets", targets_path),
    )
    dimension_counts = Counter(
        str(record["dimension"]) for record, _ in selected)
    source_counts = Counter(item.source_key for _, item in selected)
    source_dimension_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for record, item in selected:
        source_dimension_counts[item.source_key][str(record["dimension"])] += 1
    expected_count = dimension_quota * len(_DIMENSIONS)
    if (len(selected) != expected_count
            or len(title_surfaces) != expected_count
            or any(dimension_counts[dimension] != dimension_quota
                   for dimension in _DIMENSIONS)):
        raise BroadQaExternalDataError(
            "interactive development selection 未闭合")
    manifest = {
        "artifact_kind": INTERACTIVE_DEVELOPMENT_PACK_KIND,
        "artifacts": [{
            "bytes": path.stat().st_size,
            "record_count": expected_count,
            "role": role,
            "sha256": _sha256_file(path),
        } for role, path in paths],
        "census_manifest_sha256": _sha256_file(census_manifest_file),
        "census_sha256": _sha256_file(census_file),
        "dimension_counts": {
            dimension: dimension_counts[dimension]
            for dimension in _DIMENSIONS
        },
        "dimension_quota": dimension_quota,
        "format_version": 1,
        "question_count": expected_count,
        "selection_rule": INTERACTIVE_DEVELOPMENT_SELECTION_RULE,
        "source_counts": dict(sorted(source_counts.items())),
        "source_dimension_counts": {
            source: {
                dimension: values[dimension]
                for dimension in _DIMENSIONS
            }
            for source, values in sorted(source_dimension_counts.items())
        },
        "source_report": source_report,
        "source_target_count": len(title_surfaces),
        "status": "FROZEN_NOT_RUN",
        "thresholds": JOINT_THRESHOLDS,
        "title_count": len(title_surfaces),
        "title_domain_overlap_count": 0,
    }
    pack_manifest_path = target / "manifest.json"
    pack_manifest_path.write_bytes(canonical_json_line(manifest))
    return {
        **manifest,
        "manifest_sha256": _sha256_file(pack_manifest_path),
    }


def build_interactive_dimension_census(
        items: Iterable[ExternalQaItem],
        *,
        candidates_path: str | Path,
        source_census_path: str | Path,
        consumed_source_target_paths: Iterable[str | Path],
        census_path: str | Path,
        manifest_path: str | Path,
        source_report: dict[str, object],
        rules_path: str | Path | None = None,
        ) -> dict[str, object]:
    """对剩余 SOURCE_ALIGNED 总体发布主维度库存，不运行问答。"""
    candidate_file = Path(candidates_path).resolve()
    source_census_file = Path(source_census_path).resolve()
    census_output = Path(census_path).resolve()
    manifest_output = Path(manifest_path).resolve()
    rules_file = _dimension_artifact_path(rules_path)
    if (census_output.exists() or manifest_output.exists()
            or census_output.parent != manifest_output.parent):
        raise BroadQaExternalDataError("interactive census 输出边界非法")
    candidates = read_source_alignment_candidates(candidate_file)
    source_census = read_source_alignment_census(source_census_file)
    candidate_by_id = {item["item_id"]: item for item in candidates}
    census_by_id = {item["item_id"]: item for item in source_census}
    if set(candidate_by_id) != set(census_by_id):
        raise BroadQaExternalDataError("interactive source census inventory 漂移")
    item_by_id = {item.item_id: item for item in items}
    if not set(candidate_by_id).issubset(item_by_id):
        raise BroadQaExternalDataError("interactive official inventory 漂移")
    consumed_paths = tuple(Path(item).resolve()
                           for item in consumed_source_target_paths)
    if not consumed_paths:
        raise BroadQaExternalDataError("interactive consumed title 输入为空")
    consumed_titles = set()
    for path in consumed_paths:
        consumed_titles.update(read_joint_source_targets(path))
    rules = load_interactive_dimension_rules(rules_file)
    counts: Counter[str] = Counter()
    per_source: dict[str, Counter[str]] = defaultdict(Counter)
    eligible_records = []
    excluded_consumed_count = 0
    for item_id in sorted(candidate_by_id):
        source_record = census_by_id[item_id]
        if source_record["status"] != SOURCE_ALIGNED_STATUS:
            continue
        item = item_by_id[item_id]
        if item.title_key in consumed_titles:
            excluded_consumed_count += 1
            continue
        dimension = classify_interactive_dimension(item.question, rules=rules)
        counts[dimension] += 1
        per_source[item.source_key][dimension] += 1
        eligible_records.append({
            "dimension": dimension,
            "format_version": 1,
            "item_id": item.item_id,
            "question_sha256": hashlib.sha256(
                item.question.encode("utf-8")).hexdigest(),
            "record_kind": INTERACTIVE_CENSUS_RECORD_KIND,
            "source_key": item.source_key,
            "terminal_page_id": source_record["terminal_page_id"],
            "terminal_revision_id": source_record["terminal_revision_id"],
            "title_key": item.title_key,
        })
    if (not eligible_records or set(counts) != set(_DIMENSIONS)
            or sum(counts.values()) != len(eligible_records)):
        raise BroadQaExternalDataError("interactive dimension inventory 不闭合")
    record_count = _write_census(census_output, eligible_records)
    if record_count != len(eligible_records):
        raise BroadQaExternalDataError("interactive census write count 漂移")
    manifest = {
        "artifact_kind": INTERACTIVE_CENSUS_KIND,
        "candidate_count": len(candidates),
        "candidates_sha256": _sha256_file(candidate_file),
        "consumed_source_targets": [
            {"sha256": _sha256_file(path)} for path in consumed_paths
        ],
        "consumed_title_count": len(consumed_titles),
        "census_bytes": census_output.stat().st_size,
        "census_sha256": _sha256_file(census_output),
        "dimension_counts": {
            dimension: counts[dimension] for dimension in _DIMENSIONS
        },
        "excluded_consumed_item_count": excluded_consumed_count,
        "format_version": 1,
        "per_source": {
            source: {
                dimension: values[dimension] for dimension in _DIMENSIONS
            }
            for source, values in sorted(per_source.items())
        },
        "remaining_source_aligned_count": len(eligible_records),
        "rules_sha256": _sha256_file(rules_file),
        "selection_rule": INTERACTIVE_CENSUS_RULE,
        "source_census_sha256": _sha256_file(source_census_file),
        "source_report": source_report,
        "status": "FROZEN_NOT_USED_FOR_QA",
    }
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with manifest_output.open("xb") as handle:
            handle.write(canonical_json_line(manifest))
    except FileExistsError as error:
        raise BroadQaExternalDataError(
            "interactive census manifest 禁止覆盖") from error
    return {**manifest, "manifest_sha256": _sha256_file(manifest_output)}


__all__ = [
    "DIMENSION_ARTIFACT_PATH",
    "DIMENSION_ARTIFACT_SHA256",
    "INTERACTIVE_CENSUS_KIND",
    "INTERACTIVE_CENSUS_RECORD_KIND",
    "INTERACTIVE_DEVELOPMENT_PACK_KIND",
    "INTERACTIVE_DIMENSION_RECORD_KIND",
    "build_interactive_dimension_census",
    "classify_interactive_dimension",
    "freeze_interactive_development_pack",
    "load_interactive_dimension_rules",
]
