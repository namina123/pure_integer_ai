"""从冻结来源对齐 census 发布新联合 family 及其运行时来源子集。"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Iterable

from pure_integer_ai.experiments.ph2_broad_qa_contract import (
    BroadQaSelectedPage,
    BroadQaTargetSelectionManifest,
)
from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
    ExternalQaItem,
    external_title_surfaces,
    select_external_source_pack,
)
from pure_integer_ai.experiments.ph2_broad_qa_joint_eval import (
    JOINT_LABEL_KIND,
    JOINT_QUESTION_KIND,
    JOINT_TARGET_KIND,
    JOINT_THRESHOLDS,
    read_joint_source_aliases,
    read_joint_source_targets,
)
from pure_integer_ai.experiments.ph2_broad_qa_source_alignment import (
    SOURCE_ALIGNED_STATUS,
    SOURCE_ALIGNMENT_SELECTION_RULE,
    read_source_alignment_candidates,
    read_source_alignment_census,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


SOURCE_ALIGNED_JOINT_PACK_KIND = (
    "PH2_BROAD_QA_SOURCE_ALIGNED_JOINT_RETRIEVAL_EVIDENCE_PACK_V1"
)
SOURCE_ALIGNED_RUNTIME_SOURCE_KIND = (
    "PH2_BROAD_QA_SOURCE_ALIGNED_RUNTIME_SOURCE_PACK_V1"
)


def _sha256_file(path: Path) -> str:
    """流式计算 family 输入或 artifact 的 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _write_jsonl(path: Path, records: Iterable[dict[str, object]]) -> int:
    """不可覆盖地写规范 family JSONL。"""
    if path.exists():
        raise BroadQaExternalDataError(
            f"source aligned family artifact 禁止覆盖: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("xb") as handle:
        for record in records:
            handle.write(canonical_json_line(record))
            count += 1
    return count


def _publish_or_verify(path: Path, payload: bytes) -> None:
    """幂等发布规范字节；已有不同内容时 fail closed。"""
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise BroadQaExternalDataError(
                f"source aligned artifact 已存在且漂移: {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


def freeze_source_aligned_joint_pack(
        items: Iterable[ExternalQaItem],
        *,
        candidates_path: str | Path,
        census_path: str | Path,
        census_manifest_path: str | Path,
        candidate_manifest_path: str | Path,
        target_dir: str | Path,
        source_report: dict[str, object],
        dev_per_source: int = 100,
        held_out_per_source: int = 150,
        ) -> dict[str, object]:
    """只从冻结 SOURCE_ALIGNED 总体选择新 family，不读取预测结果。"""
    candidate_file = Path(candidates_path).resolve()
    census_file = Path(census_path).resolve()
    census_manifest = Path(census_manifest_path).resolve()
    candidate_manifest = Path(candidate_manifest_path).resolve()
    candidate_values = read_source_alignment_candidates(candidate_file)
    census_values = read_source_alignment_census(census_file)
    candidate_ids = {item["item_id"] for item in candidate_values}
    if {item["item_id"] for item in census_values} != candidate_ids:
        raise BroadQaExternalDataError(
            "source aligned family census inventory 漂移")
    item_by_id = {item.item_id: item for item in items}
    if not candidate_ids.issubset(item_by_id):
        raise BroadQaExternalDataError(
            "source aligned family 官方来源 inventory 漂移")
    eligible_ids = {
        item["item_id"] for item in census_values
        if item["status"] == SOURCE_ALIGNED_STATUS
    }
    eligible = tuple(item_by_id[item_id] for item_id in sorted(eligible_ids))
    selected = select_external_source_pack(
        eligible, dev_per_source=dev_per_source,
        held_out_per_source=held_out_per_source)
    target = Path(target_dir).resolve()
    if target.exists():
        raise BroadQaExternalDataError("source aligned family target 已存在")
    target.mkdir(parents=True)
    artifacts = []
    titles: dict[str, set[str]] = defaultdict(set)
    for split in ("dev", "held_out"):
        values = selected[split]
        question_path = target / f"{split}.questions.jsonl"
        label_path = target / f"{split}.labels.jsonl"
        question_count = _write_jsonl(question_path, ({
            "format_version": 1,
            "item_id": item.item_id,
            "license_id": item.license_id,
            "question": item.question,
            "record_kind": JOINT_QUESTION_KIND,
            "source_key": item.source_key,
            "source_partition": item.source_partition,
            "source_question_id": item.source_question_id,
            "source_revision": item.source_revision,
            "split": split,
            "upstream_url": item.upstream_url,
        } for item in values))
        label_count = _write_jsonl(label_path, ({
            "expected_title_key": item.title_key,
            "format_version": 1,
            "gold_answers": list(item.gold_answers),
            "item_id": item.item_id,
            "record_kind": JOINT_LABEL_KIND,
            "split": split,
        } for item in values))
        if question_count != label_count or question_count != len(values):
            raise BroadQaExternalDataError(
                "source aligned family split inventory 未闭合")
        for item in values:
            titles[item.title_key].update(external_title_surfaces(item.title))
        for role, path in (("questions", question_path),
                           ("labels", label_path)):
            artifacts.append({
                "bytes": path.stat().st_size,
                "record_count": len(values),
                "role": f"{split}_{role}",
                "sha256": _sha256_file(path),
            })
    targets_path = target / "source_targets.jsonl"
    target_count = _write_jsonl(targets_path, ({
        "format_version": 1,
        "record_kind": JOINT_TARGET_KIND,
        "surfaces": sorted(titles[key]),
        "title_key": key,
    } for key in sorted(titles)))
    artifacts.append({
        "bytes": targets_path.stat().st_size,
        "record_count": target_count,
        "role": "source_targets",
        "sha256": _sha256_file(targets_path),
    })
    census_manifest_value = json.loads(
        census_manifest.read_text(encoding="utf-8"))
    if (census_manifest_value.get("census_sha256")
            != _sha256_file(census_file)
            or census_manifest_value.get("candidates_sha256")
            != _sha256_file(candidate_file)):
        raise BroadQaExternalDataError(
            "source aligned family census commitment 漂移")
    manifest = {
        "artifact_kind": SOURCE_ALIGNED_JOINT_PACK_KIND,
        "artifacts": artifacts,
        "candidate_manifest_sha256": _sha256_file(candidate_manifest),
        "census_manifest_sha256": _sha256_file(census_manifest),
        "census_sha256": _sha256_file(census_file),
        "format_version": 1,
        "population_candidate_count": len(candidate_values),
        "population_source_aligned_count": len(eligible_ids),
        "population_source_aligned_ppm": (
            len(eligible_ids) * 1_000_000 // len(candidate_values)),
        "selection_rule": SOURCE_ALIGNMENT_SELECTION_RULE,
        "source_report": source_report,
        "source_target_count": target_count,
        "splits": {
            split: {
                "question_count": len(selected[split]),
                "source_counts": dict(sorted(Counter(
                    item.source_key for item in selected[split]).items())),
                "title_count": len({item.title_key for item in selected[split]}),
            }
            for split in ("dev", "held_out")
        },
        "status": "FROZEN_NOT_RUN",
        "thresholds": JOINT_THRESHOLDS,
        "title_domain_overlap_count": 0,
    }
    if (census_manifest_value.get("source_aligned_count")
            != len(eligible_ids)):
        raise BroadQaExternalDataError(
            "source aligned family census manifest 漂移")
    manifest_path = target / "manifest.json"
    manifest_path.write_bytes(canonical_json_line(manifest))
    return {**manifest, "manifest_sha256": _sha256_file(manifest_path)}


def derive_source_aligned_runtime_sources(
        source_targets_path: str | Path,
        population_aliases_path: str | Path,
        population_terminal_selection: BroadQaTargetSelectionManifest,
        *,
        aliases_path: str | Path,
        terminal_selection_path: str | Path,
        manifest_path: str | Path,
        ) -> dict[str, object]:
    """从全体 census 来源包确定性派生 family 所需 alias 与终页子集。"""
    if not isinstance(population_terminal_selection,
                      BroadQaTargetSelectionManifest):
        raise BroadQaExternalDataError(
            "source aligned runtime population selection 非法")
    targets_file = Path(source_targets_path).resolve()
    population_aliases_file = Path(population_aliases_path).resolve()
    aliases_output = Path(aliases_path).resolve()
    selection_output = Path(terminal_selection_path).resolve()
    manifest_output = Path(manifest_path).resolve()
    source_targets = read_joint_source_targets(targets_file)
    population_aliases = {
        item["title_key"]: item
        for item in read_joint_source_aliases(population_aliases_file)
    }
    records = []
    terminal_page_ids = set()
    for key in sorted(source_targets):
        alias = population_aliases.get(key)
        if alias is None or alias["status"] != "RESOLVED":
            raise BroadQaExternalDataError(
                "source aligned family 含未解析来源")
        records.append({
            **alias,
            "original_surfaces": list(source_targets[key]),
        })
        terminal_page_ids.add(alias["terminal_page_id"])
    pages_by_id = {
        item.page_id: item
        for item in population_terminal_selection.selected_pages
    }
    if not terminal_page_ids.issubset(pages_by_id):
        raise BroadQaExternalDataError(
            "source aligned runtime terminal page 缺失")
    pages = tuple(sorted(
        (pages_by_id[page_id] for page_id in terminal_page_ids),
        key=lambda item: (item.rank_sha256, item.page_id, item.title)))
    selected = tuple(BroadQaSelectedPage(
        ordinal, item.rank_sha256, item.title, item.title_sha256,
        item.page_id, item.index_line_number, item.compressed_block_offset,
        item.compressed_block_end_offset)
        for ordinal, item in enumerate(pages, start=1))
    terminal_payload = canonical_json_line({
        "terminal_pages": [
            {"page_id": item.page_id, "title": item.title}
            for item in selected
        ],
    })
    terminal_selection = BroadQaTargetSelectionManifest(
        population_terminal_selection.source_key,
        population_terminal_selection.snapshot_id,
        population_terminal_selection.snapshot_manifest_sha256,
        population_terminal_selection.index_local_sha256,
        population_terminal_selection.index_upstream_sha1,
        population_terminal_selection.xml_local_sha256,
        population_terminal_selection.xml_compressed_size_bytes,
        population_terminal_selection.index_entry_count,
        len(selected), hashlib.sha256(terminal_payload).hexdigest(),
        selected, ())
    alias_payload = b"".join(canonical_json_line(item) for item in records)
    selection_payload = terminal_selection.canonical_bytes()
    manifest = {
        "alias_count": len(records),
        "alias_sha256": hashlib.sha256(alias_payload).hexdigest(),
        "artifact_kind": SOURCE_ALIGNED_RUNTIME_SOURCE_KIND,
        "format_version": 1,
        "population_alias_sha256": _sha256_file(population_aliases_file),
        "population_terminal_selection_sha256": (
            population_terminal_selection.sha256()),
        "source_targets_sha256": _sha256_file(targets_file),
        "status": "FROZEN_NOT_INDEXED",
        "terminal_page_count": len(selected),
        "terminal_selection_sha256": terminal_selection.sha256(),
    }
    manifest_payload = canonical_json_line(manifest)
    _publish_or_verify(aliases_output, alias_payload)
    _publish_or_verify(selection_output, selection_payload)
    _publish_or_verify(manifest_output, manifest_payload)
    return {
        **manifest,
        "manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
    }


__all__ = [
    "SOURCE_ALIGNED_JOINT_PACK_KIND",
    "SOURCE_ALIGNED_RUNTIME_SOURCE_KIND",
    "derive_source_aligned_runtime_sources",
    "freeze_source_aligned_joint_pack",
]
