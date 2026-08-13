"""冻结并运行自然标题锚定的检索与证据选择联合评测。"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import time
from typing import Iterable

from pure_integer_ai.experiments.ph2_broad_qa_contract import (
    BroadQaSelectedPage,
    BroadQaTargetSelectionManifest,
)
from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
    ExternalQaItem,
    external_title_surfaces,
    normalize_external_text,
    select_external_successor_pack,
)
from pure_integer_ai.experiments.ph2_broad_qa_query import (
    answer_broad_qa_candidates,
    retrieve_broad_qa_candidates,
)
from pure_integer_ai.experiments.ph2_broad_qa_selection import (
    build_broad_qa_target_selection,
)
from pure_integer_ai.experiments.ph2_broad_qa_source import (
    iter_broad_qa_selected_page_inspections,
)
from pure_integer_ai.experiments.ph2_mediawiki_snapshot import (
    MediaWikiDumpSnapshotManifest,
)
from pure_integer_ai.experiments.ph2_broad_qa_index import broad_qa_terms
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line
from pure_integer_ai.storage.integer_codec import (
    decode_integer_tuple,
    encode_integer_tuple,
)


JOINT_PACK_KIND = "PH2_BROAD_QA_JOINT_RETRIEVAL_EVIDENCE_PACK_V1"
JOINT_SUCCESSOR_PACK_KIND = (
    "PH2_BROAD_QA_JOINT_RETRIEVAL_EVIDENCE_PACK_V2"
)
JOINT_QUESTION_KIND = "PH2_BROAD_QA_JOINT_QUESTION_V1"
JOINT_LABEL_KIND = "PH2_BROAD_QA_JOINT_LABEL_V1"
JOINT_TARGET_KIND = "PH2_BROAD_QA_JOINT_SOURCE_TARGET_V1"
JOINT_PREDICTION_KIND = "PH2_BROAD_QA_JOINT_PREDICTION_V1"
JOINT_AGGREGATE_KIND = "PH2_BROAD_QA_JOINT_AGGREGATE_V2"
JOINT_ALIAS_KIND = "PH2_BROAD_QA_JOINT_SOURCE_ALIAS_V1"
JOINT_SELECTION_RULE = (
    "EXCLUDE_PRIOR_TITLE_DOMAIN_THEN_NATURAL_TITLE_ANCHOR_"
    "TITLE_BUCKET_ITEM_SHA256_V1"
)
JOINT_SUCCESSOR_SELECTION_RULE = (
    "EXCLUDE_PRIOR_QUESTION_AND_SOURCE_TARGET_TITLE_DOMAINS_THEN_"
    "NATURAL_TITLE_ANCHOR_TITLE_BUCKET_ITEM_SHA256_V2"
)
JOINT_THRESHOLDS = {
    "minimum_evidence_hit_ppm": 600_000,
    "minimum_recall_at_20_ppm": 800_000,
    "minimum_top1_source_hit_ppm": 700_000,
    "required_answer_citation_valid_ppm": 1_000_000,
}


def _sha256_file(path: Path) -> str:
    """流式计算冻结输入、数据库或结果文件的 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path, *, expected_kind: str) -> tuple[dict, ...]:
    """读取身份唯一的严格 JSONL，并核对 record kind。"""
    if not path.is_file():
        raise BroadQaExternalDataError(f"joint input 缺失: {path.name}")
    values = []
    identities = set()
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.endswith("\n") or not line.strip():
                    raise BroadQaExternalDataError(
                        f"joint JSONL 换行非法: {line_number}")
                value = json.loads(line)
                if (not isinstance(value, dict)
                        or value.get("record_kind") != expected_kind):
                    raise BroadQaExternalDataError(
                        f"joint JSONL record 漂移: {line_number}")
                identity = value.get("item_id", value.get("title_key"))
                if (not isinstance(identity, str) or not identity
                        or identity in identities):
                    raise BroadQaExternalDataError(
                        f"joint JSONL identity 漂移: {line_number}")
                identities.add(identity)
                values.append(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError("joint JSONL 非法") from error
    if not values:
        raise BroadQaExternalDataError("joint JSONL 为空")
    return tuple(values)


def _write_jsonl(path: Path, records: Iterable[dict[str, object]]) -> int:
    """不可覆盖地写规范 JSONL，并返回记录数量。"""
    if path.exists():
        raise BroadQaExternalDataError(f"joint artifact 禁止覆盖: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("xb") as handle:
        for record in records:
            handle.write(canonical_json_line(record))
            count += 1
    return count


def _prior_title_keys(paths: Iterable[str | Path]) -> tuple[str, ...]:
    """从已消费 questions 读取标题域，不读取其独立 labels。"""
    keys = set()
    identities = set()
    for raw_path in paths:
        path = Path(raw_path).resolve()
        for value in _read_jsonl(
                path, expected_kind="PH2_BROAD_QA_EXTERNAL_QUESTION_V1"):
            if (not isinstance(value.get("title"), str)
                    or not isinstance(value.get("item_id"), str)
                    or value["item_id"] in identities):
                raise BroadQaExternalDataError("prior external inventory 漂移")
            identities.add(value["item_id"])
            keys.add(normalize_external_text(value["title"]))
    if not keys:
        raise BroadQaExternalDataError("prior title domain 为空")
    return tuple(sorted(keys))


def _prior_source_target_keys(paths: Iterable[str | Path]) -> tuple[str, ...]:
    """从前代公开 target ledger 读取标题域，不接触问题标签。"""
    keys = set()
    for raw_path in paths:
        path = Path(raw_path).resolve()
        for value in _read_jsonl(path, expected_kind=JOINT_TARGET_KIND):
            key = value.get("title_key")
            if (not isinstance(key, str) or not key
                    or normalize_external_text(key) != key):
                raise BroadQaExternalDataError(
                    "prior source target title key 漂移")
            keys.add(key)
    return tuple(sorted(keys))


def freeze_joint_source_pack(
        items: Iterable[ExternalQaItem],
        *,
        prior_question_paths: Iterable[str | Path],
        prior_source_target_paths: Iterable[str | Path] = (),
        target_dir: str | Path,
        source_report: dict[str, object],
        dev_per_source: int = 100,
        held_out_per_source: int = 150,
        ) -> dict[str, object]:
    """先排除旧标题域，再冻结自然包含标题的独立 200/300 问 family。"""
    target = Path(target_dir).resolve()
    if target.exists():
        raise BroadQaExternalDataError("joint freeze target 已存在")
    prior_paths = tuple(Path(item).resolve() for item in prior_question_paths)
    prior_target_paths = tuple(
        Path(item).resolve() for item in prior_source_target_paths)
    question_excluded = _prior_title_keys(prior_paths)
    target_excluded = _prior_source_target_keys(prior_target_paths)
    excluded = tuple(sorted(set(question_excluded) | set(target_excluded)))
    anchored = tuple(
        item for item in items
        if item.title_key in normalize_external_text(item.question)
    )
    selected = select_external_successor_pack(
        anchored, excluded_title_keys=excluded,
        dev_per_source=dev_per_source,
        held_out_per_source=held_out_per_source)
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
            raise BroadQaExternalDataError("joint split inventory 未闭合")
        for item in values:
            titles[item.title_key].update(external_title_surfaces(item.title))
        for role, path in (("questions", question_path), ("labels", label_path)):
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
    excluded_payload = canonical_json_line({"title_keys": list(excluded)})
    manifest = {
        "artifact_kind": (
            JOINT_SUCCESSOR_PACK_KIND if prior_target_paths
            else JOINT_PACK_KIND),
        "artifacts": artifacts,
        "excluded_prior_question_files": [
            {"sha256": _sha256_file(path)} for path in prior_paths],
        "excluded_prior_title_count": len(excluded),
        "excluded_prior_titles_sha256": hashlib.sha256(
            excluded_payload).hexdigest(),
        "format_version": 1,
        "selection_rule": (
            JOINT_SUCCESSOR_SELECTION_RULE if prior_target_paths
            else JOINT_SELECTION_RULE),
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
    if prior_target_paths:
        manifest["excluded_prior_question_title_count"] = len(
            question_excluded)
        manifest["excluded_prior_source_target_files"] = [
            {"sha256": _sha256_file(path)} for path in prior_target_paths]
        manifest["excluded_prior_source_target_title_count"] = len(
            target_excluded)
    manifest_path = target / "manifest.json"
    manifest_path.write_bytes(canonical_json_line(manifest))
    return {**manifest, "manifest_sha256": _sha256_file(manifest_path)}


def read_joint_source_targets(
        path: str | Path,
        ) -> dict[str, tuple[str, ...]]:
    """回读与题目映射分离的全局来源标题 inventory。"""
    values = _read_jsonl(Path(path).resolve(), expected_kind=JOINT_TARGET_KIND)
    result = {}
    for value in values:
        if (set(value) != {
                "format_version", "record_kind", "surfaces", "title_key"}
                or value["format_version"] != 1
                or not isinstance(value["surfaces"], list)
                or not value["surfaces"]
                or any(not isinstance(item, str) or not item
                       for item in value["surfaces"])
                or value["surfaces"] != sorted(set(value["surfaces"]))):
            raise BroadQaExternalDataError("joint source target 漂移")
        result[value["title_key"]] = tuple(value["surfaces"])
    return result


def resolve_joint_source_aliases(
        snapshot: MediaWikiDumpSnapshotManifest,
        initial_selection: BroadQaTargetSelectionManifest,
        source_targets: dict[str, tuple[str, ...]],
        *,
        snapshot_manifest_sha256: str,
        index_path: str | Path,
        xml_path: str | Path,
        alias_path: str | Path,
        worker_count: int = 4,
        max_redirect_depth: int = 8,
        ) -> tuple[BroadQaTargetSelectionManifest, dict[str, object]]:
    """解析冻结重定向链，发布 alias ledger 并返回终页 selection。"""
    if (not isinstance(snapshot, MediaWikiDumpSnapshotManifest)
            or not isinstance(initial_selection, BroadQaTargetSelectionManifest)
            or set(source_targets) != (
                {normalize_external_text(item.title)
                 for item in initial_selection.selected_pages}
                | set(initial_selection.missing_title_keys))
            or type(max_redirect_depth) is not int
            or not 1 <= max_redirect_depth <= 32):
        raise BroadQaExternalDataError("joint alias resolution 输入漂移")
    output = Path(alias_path).resolve()
    if output.exists():
        raise BroadQaExternalDataError("joint alias ledger 禁止覆盖")
    selected_by_key = {
        normalize_external_text(item.title): item
        for item in initial_selection.selected_pages
    }
    inspection_cache = {}
    resolution: dict[str, dict[str, object]] = {}
    resolved_pages = {}
    active = {
        key: {
            "chain": [selected_by_key[key].title],
            "page": selected_by_key[key],
            "seen_page_ids": {selected_by_key[key].page_id},
        }
        for key in sorted(selected_by_key)
    }
    for key in initial_selection.missing_title_keys:
        resolution[key] = {
            "chain": list(source_targets[key]),
            "failure_code": "SOURCE_TITLE_NOT_IN_SNAPSHOT",
            "status": "MISSING",
        }
    for _depth in range(max_redirect_depth + 1):
        if not active:
            break
        pages_to_inspect = tuple(sorted(
            {state["page"].page_id: state["page"] for state in active.values()
             if state["page"].page_id not in inspection_cache}.values(),
            key=lambda item: item.ordinal))
        if pages_to_inspect:
            for inspection in iter_broad_qa_selected_page_inspections(
                    pages_to_inspect, xml_path=xml_path,
                    source_key=initial_selection.source_key,
                    xml_compressed_size_bytes=(
                        initial_selection.xml_compressed_size_bytes),
                    worker_count=worker_count):
                inspection_cache[inspection.page_id] = inspection
        redirect_targets: dict[str, set[str]] = defaultdict(set)
        next_needed: dict[str, str] = {}
        finished = []
        for original_key, state in active.items():
            page = state["page"]
            inspection = inspection_cache[page.page_id]
            if not inspection.redirect_title:
                resolution[original_key] = {
                    "chain": state["chain"],
                    "status": "RESOLVED",
                    "terminal_page_id": page.page_id,
                    "terminal_revision_id": inspection.revision_id,
                    "terminal_title": inspection.title,
                    "terminal_title_key": normalize_external_text(
                        inspection.title),
                }
                resolved_pages[original_key] = page
                finished.append(original_key)
                continue
            target_key = inspection.redirect_title
            state["chain"].append(inspection.redirect_title)
            redirect_targets[target_key].add(inspection.redirect_title)
            next_needed[original_key] = target_key
        for key in finished:
            active.pop(key)
        if not next_needed:
            continue
        next_selection = build_broad_qa_target_selection(
            snapshot, index_path=index_path,
            snapshot_manifest_sha256=snapshot_manifest_sha256,
            target_titles={
                key: tuple(sorted(values))
                for key, values in redirect_targets.items()
            })
        next_by_key = {item.title: item for item in next_selection.selected_pages}
        for original_key, target_key in next_needed.items():
            page = next_by_key.get(target_key)
            if page is None:
                resolution[original_key] = {
                    "chain": active[original_key]["chain"],
                    "failure_code": "REDIRECT_TARGET_NOT_IN_SNAPSHOT",
                    "status": "MISSING",
                }
                active.pop(original_key)
            elif page.page_id in active[original_key]["seen_page_ids"]:
                resolution[original_key] = {
                    "chain": active[original_key]["chain"],
                    "failure_code": "REDIRECT_CYCLE",
                    "status": "MISSING",
                }
                active.pop(original_key)
            else:
                active[original_key]["page"] = page
                active[original_key]["seen_page_ids"].add(page.page_id)
    for original_key, state in active.items():
        resolution[original_key] = {
            "chain": state["chain"],
            "failure_code": "REDIRECT_DEPTH_EXCEEDED",
            "status": "MISSING",
        }
    terminal_pages = {
        page.page_id: page for page in resolved_pages.values()
    }
    if not terminal_pages:
        raise BroadQaExternalDataError("joint alias 没有可解析终页")
    ordered_terminal = sorted(
        terminal_pages.values(), key=lambda item: (
            item.rank_sha256, item.page_id, item.title))
    selected_terminal = tuple(BroadQaSelectedPage(
        ordinal, item.rank_sha256, item.title, item.title_sha256,
        item.page_id, item.index_line_number, item.compressed_block_offset,
        item.compressed_block_end_offset)
        for ordinal, item in enumerate(ordered_terminal, start=1))
    terminal_payload = canonical_json_line({
        "terminal_pages": [
            {"page_id": item.page_id, "title": item.title}
            for item in selected_terminal
        ],
    })
    terminal_selection = BroadQaTargetSelectionManifest(
        initial_selection.source_key, initial_selection.snapshot_id,
        initial_selection.snapshot_manifest_sha256,
        initial_selection.index_local_sha256,
        initial_selection.index_upstream_sha1,
        initial_selection.xml_local_sha256,
        initial_selection.xml_compressed_size_bytes,
        initial_selection.index_entry_count, len(selected_terminal),
        hashlib.sha256(terminal_payload).hexdigest(), selected_terminal, ())
    records = []
    for key in sorted(source_targets):
        record = resolution[key]
        if record["status"] == "RESOLVED":
            terminal = resolved_pages.get(key)
            if terminal is None:
                raise BroadQaExternalDataError("joint terminal selection 未闭合")
            record = {
                **record,
                "terminal_page_id": terminal.page_id,
            }
        records.append({
            **record,
            "format_version": 1,
            "original_surfaces": list(source_targets[key]),
            "record_kind": JOINT_ALIAS_KIND,
            "title_key": key,
        })
    count = _write_jsonl(output, records)
    status_counts = Counter(item["status"] for item in records)
    report = {
        "alias_count": count,
        "alias_sha256": _sha256_file(output),
        "resolved_count": status_counts["RESOLVED"],
        "status_counts": dict(sorted(status_counts.items())),
        "terminal_page_count": len(terminal_selection.selected_pages),
        "terminal_selection_sha256": terminal_selection.sha256(),
    }
    return terminal_selection, report


def read_joint_source_aliases(path: str | Path) -> tuple[dict, ...]:
    """严格回读来源 alias ledger。"""
    values = _read_jsonl(Path(path).resolve(), expected_kind=JOINT_ALIAS_KIND)
    for value in values:
        common = {
            "chain", "format_version", "original_surfaces", "record_kind",
            "status", "title_key",
        }
        if (value.get("status") == "RESOLVED"
                and set(value) != common | {
                    "terminal_page_id", "terminal_revision_id",
                    "terminal_title", "terminal_title_key"}):
            raise BroadQaExternalDataError("joint resolved alias schema 漂移")
        if (value.get("status") == "MISSING"
                and set(value) != common | {"failure_code"}):
            raise BroadQaExternalDataError("joint missing alias schema 漂移")
        if (value.get("format_version") != 1
                or not isinstance(value.get("chain"), list)
                or not isinstance(value.get("original_surfaces"), list)):
            raise BroadQaExternalDataError("joint alias value 漂移")
    return values


def _restore_postings(payload: bytes) -> tuple[int, ...]:
    """恢复 SQLite posting 中的严格递增 passage id。"""
    current = 0
    values = []
    for delta in decode_integer_tuple(payload):
        if delta <= 0:
            raise BroadQaExternalDataError("joint posting delta 非正")
        current += delta
        values.append(current)
    return tuple(values)


def _encode_postings(values: Iterable[int]) -> bytes:
    """把严格递增 passage id 编码为 canonical delta-varint。"""
    ordered = tuple(values)
    if (not ordered or ordered != tuple(sorted(set(ordered)))
            or ordered[0] <= 0):
        raise BroadQaExternalDataError("joint posting inventory 非法")
    prior = 0
    deltas = []
    for value in ordered:
        deltas.append(value - prior)
        prior = value
    return encode_integer_tuple(tuple(deltas))


def augment_broad_qa_index(
        base_database_path: str | Path,
        target_database_path: str | Path,
        *,
        output_database_path: str | Path,
        base_expected_sha256: str,
        target_selection_sha256: str,
        alias_path: str | Path | None = None,
        ) -> dict[str, object]:
    """把冻结目标页追加到随机 20k 索引，生成单一联合检索库。"""
    base = Path(base_database_path).resolve()
    source = Path(target_database_path).resolve()
    output = Path(output_database_path).resolve()
    partial = output.with_name(output.name + ".partial")
    if (output.exists() or partial.exists() or not base.is_file()
            or not source.is_file() or len(base_expected_sha256) != 64
            or len(target_selection_sha256) != 64):
        raise BroadQaExternalDataError("joint augmentation 路径/身份非法")
    if _sha256_file(base) != base_expected_sha256:
        raise BroadQaExternalDataError("joint base database SHA 漂移")
    output.parent.mkdir(parents=True, exist_ok=True)
    started_ns = time.perf_counter_ns()
    shutil.copyfile(base, partial)
    target = sqlite3.connect(str(partial))
    extra = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    try:
        if target.execute("PRAGMA integrity_check").fetchone() != ("ok",):
            raise BroadQaExternalDataError("joint base integrity 失败")
        if extra.execute("PRAGMA integrity_check").fetchone() != ("ok",):
            raise BroadQaExternalDataError("joint target integrity 失败")
        base_metadata = dict(target.execute("SELECT key,value FROM metadata"))
        extra_metadata = dict(extra.execute("SELECT key,value FROM metadata"))
        if (base_metadata["snapshot_id"] != extra_metadata["snapshot_id"]
                or base_metadata["source_key"] != extra_metadata["source_key"]):
            raise BroadQaExternalDataError("joint index snapshot 漂移")
        existing_pages = {
            int(row[0]) for row in target.execute("SELECT page_id FROM document")}
        next_doc_id = int(target.execute(
            "SELECT COALESCE(MAX(doc_id),0) FROM document").fetchone()[0]) + 1
        next_passage_id = int(target.execute(
            "SELECT COALESCE(MAX(passage_id),0) FROM passage").fetchone()[0]) + 1
        passage_map = {}
        added_documents = 0
        added_passages = 0
        for row in extra.execute("""
                SELECT doc_id,title,page_id,revision_id,timestamp,
                       contributor_json,text_sha256
                FROM document ORDER BY doc_id
                """):
            if int(row[2]) in existing_pages:
                continue
            old_doc_id = int(row[0])
            new_doc_id = next_doc_id
            next_doc_id += 1
            target.execute(
                "INSERT INTO document VALUES(?,?,?,?,?,?,?)",
                (new_doc_id, *row[1:]))
            for passage in extra.execute("""
                    SELECT passage_id,ordinal,raw_start,raw_end,raw_sha256,
                           text,text_sha256,section_title
                    FROM passage WHERE doc_id=? ORDER BY passage_id
                    """, (old_doc_id,)):
                old_passage_id = int(passage[0])
                new_passage_id = next_passage_id
                next_passage_id += 1
                passage_map[old_passage_id] = new_passage_id
                target.execute(
                    "INSERT INTO passage VALUES(?,?,?,?,?,?,?,?,?)",
                    (new_passage_id, new_doc_id, *passage[1:]))
                added_passages += 1
            existing_pages.add(int(row[2]))
            added_documents += 1
        affected_terms = 0
        for term, frequency, payload in extra.execute(
                "SELECT term,document_frequency,passage_deltas "
                "FROM posting ORDER BY term"):
            mapped = tuple(
                passage_map[item] for item in _restore_postings(payload)
                if item in passage_map)
            if not mapped:
                continue
            affected_terms += 1
            prior = target.execute(
                "SELECT document_frequency,passage_deltas FROM posting "
                "WHERE term=?", (term,)).fetchone()
            if prior is None:
                target.execute(
                    "INSERT INTO posting VALUES(?,?,?)",
                    (term, len(mapped), _encode_postings(mapped)))
            else:
                old_values = _restore_postings(prior[1])
                merged = old_values + mapped
                target.execute(
                    "UPDATE posting SET document_frequency=?,passage_deltas=? "
                    "WHERE term=?",
                    (len(merged), _encode_postings(merged), term))
        alias_term_count = 0
        if alias_path is not None:
            target.execute("""
                CREATE TABLE alias(
                    surface TEXT NOT NULL,
                    doc_id INTEGER NOT NULL,
                    PRIMARY KEY(surface,doc_id)
                )
            """)
            target.execute("CREATE INDEX alias_doc ON alias(doc_id,surface)")
            target.execute("""
                CREATE TABLE alias_term(
                    term TEXT NOT NULL,
                    surface TEXT NOT NULL,
                    doc_id INTEGER NOT NULL,
                    PRIMARY KEY(term,surface,doc_id)
                )
            """)
            target.execute(
                "CREATE INDEX alias_term_lookup "
                "ON alias_term(term,surface,doc_id)")
            alias_postings: dict[str, set[int]] = defaultdict(set)
            for alias in read_joint_source_aliases(alias_path):
                if alias["status"] != "RESOLVED":
                    continue
                document_row = target.execute(
                    "SELECT doc_id FROM document WHERE page_id=?",
                    (alias["terminal_page_id"],)).fetchone()
                if document_row is None:
                    continue
                passage_ids = tuple(row[0] for row in target.execute(
                    "SELECT p.passage_id FROM passage AS p "
                    "JOIN document AS d ON d.doc_id=p.doc_id "
                    "WHERE d.page_id=? ORDER BY p.passage_id",
                    (alias["terminal_page_id"],)))
                for surface in alias["original_surfaces"]:
                    target.execute(
                        "INSERT OR IGNORE INTO alias VALUES(?,?)",
                        (surface, document_row[0]))
                    for term in broad_qa_terms(surface):
                        target.execute(
                            "INSERT OR IGNORE INTO alias_term VALUES(?,?,?)",
                            (term, surface, document_row[0]))
                        alias_postings[term].update(passage_ids)
            alias_term_count = len(alias_postings)
            for term, values in sorted(alias_postings.items()):
                prior = target.execute(
                    "SELECT passage_deltas FROM posting WHERE term=?",
                    (term,)).fetchone()
                merged = set(values)
                if prior is not None:
                    merged.update(_restore_postings(prior[0]))
                ordered = tuple(sorted(merged))
                if prior is None:
                    target.execute(
                        "INSERT INTO posting VALUES(?,?,?)",
                        (term, len(ordered), _encode_postings(ordered)))
                else:
                    target.execute(
                        "UPDATE posting SET document_frequency=?,"
                        "passage_deltas=? WHERE term=?",
                        (len(ordered), _encode_postings(ordered), term))
        accepted = int(base_metadata["accepted_page_count"]) + added_documents
        passages = int(base_metadata["passage_count"]) + added_passages
        term_count = int(target.execute(
            "SELECT COUNT(*) FROM posting").fetchone()[0])
        union_identity = hashlib.sha256(canonical_json_line({
            "base_database_sha256": base_expected_sha256,
            "target_selection_sha256": target_selection_sha256,
        })).hexdigest()
        target.execute(
            "UPDATE metadata SET value=? WHERE key='accepted_page_count'",
            (str(accepted),))
        target.execute(
            "UPDATE metadata SET value=? WHERE key='passage_count'",
            (str(passages),))
        target.execute(
            "UPDATE metadata SET value=? WHERE key='term_count'",
            (str(term_count),))
        target.execute(
            "UPDATE metadata SET value=? WHERE key='selection_sha256'",
            (union_identity,))
        target.commit()
        target.execute("VACUUM")
        if target.execute("PRAGMA integrity_check").fetchone() != ("ok",):
            raise BroadQaExternalDataError("joint augmented integrity 失败")
    except Exception:
        target.close()
        extra.close()
        if partial.exists():
            partial.unlink()
        raise
    else:
        target.close()
        extra.close()
    partial.replace(output)
    return {
        "added_document_count": added_documents,
        "added_passage_count": added_passages,
        "affected_term_count": affected_terms,
        "alias_term_count": alias_term_count,
        "base_database_sha256": base_expected_sha256,
        "database_bytes": output.stat().st_size,
        "database_sha256": _sha256_file(output),
        "elapsed_ns": max(1, time.perf_counter_ns() - started_ns),
        "final_document_count": accepted,
        "final_passage_count": passages,
        "final_term_count": term_count,
        "target_selection_sha256": target_selection_sha256,
    }


def _percentile(values: list[int], percentile: int) -> int:
    """用 nearest-rank 计算确定性的整数延迟分位数。"""
    if not values or percentile not in {50, 95}:
        raise BroadQaExternalDataError("joint percentile 输入非法")
    ordered = sorted(values)
    index = max(0, (len(ordered) * percentile + 99) // 100 - 1)
    return ordered[index]


def predict_joint_retrieval(
        questions_path: str | Path,
        database_path: str | Path,
        *,
        predictions_path: str | Path,
        max_candidate_passages: int = 20,
        ) -> dict[str, object]:
    """只读 questions 与联合索引，输出候选、回答和查询资源轨迹。"""
    questions_file = Path(questions_path).resolve()
    database_file = Path(database_path).resolve()
    output = Path(predictions_path).resolve()
    questions = _read_jsonl(questions_file, expected_kind=JOINT_QUESTION_KIND)
    connection = sqlite3.connect(f"file:{database_file}?mode=ro", uri=True)
    elapsed_values = []
    records = []
    try:
        for question in questions:
            expected = {
                "format_version", "item_id", "license_id", "question",
                "record_kind", "source_key", "source_partition",
                "source_question_id", "source_revision", "split",
                "upstream_url",
            }
            if (set(question) != expected or question["format_version"] != 1
                    or not isinstance(question["question"], str)
                    or not question["question"]):
                raise BroadQaExternalDataError("joint question schema 漂移")
            started_ns = time.perf_counter_ns()
            candidates, trace = retrieve_broad_qa_candidates(
                connection, question["question"],
                max_candidate_passages=max_candidate_passages)
            result = answer_broad_qa_candidates(
                question["question"], candidates, trace)
            elapsed_ns = max(1, time.perf_counter_ns() - started_ns)
            elapsed_values.append(elapsed_ns)
            records.append({
                "candidate_document_count": trace.candidate_document_count,
                "candidates": [{
                    "page_id": item.page_id,
                    "passage_id": item.passage_id,
                    "revision_id": item.revision_id,
                    "title": item.title,
                } for item in candidates],
                "format_version": 1,
                "item_id": question["item_id"],
                "matched_query_term_count": trace.matched_query_term_count,
                "posting_visit_count": trace.posting_visit_count,
                "query_elapsed_ns": elapsed_ns,
                "record_kind": JOINT_PREDICTION_KIND,
                "result": result.to_dict(),
                "source_key": question["source_key"],
                "split": question["split"],
            })
    finally:
        connection.close()
    count = _write_jsonl(output, records)
    return {
        "database_sha256": _sha256_file(database_file),
        "prediction_count": count,
        "predictions_bytes": output.stat().st_size,
        "predictions_sha256": _sha256_file(output),
        "query_p50_ns": _percentile(elapsed_values, 50),
        "query_p95_ns": _percentile(elapsed_values, 95),
        "questions_sha256": _sha256_file(questions_file),
    }


def _ppm(numerator: int, denominator: int) -> int:
    """以整数百万分率表达联合评测比例。"""
    return 0 if denominator == 0 else numerator * 1_000_000 // denominator


def _page_contains_gold_answer(
        database: sqlite3.Connection,
        page_id: int,
        gold_answers: Iterable[str],
        ) -> bool:
    """核验冻结终页的任一 passage 是否实际包含规范化金答案。"""
    normalized = tuple(
        normalize_external_text(item) for item in gold_answers
        if isinstance(item, str) and item)
    if not normalized:
        raise BroadQaExternalDataError("joint gold answer inventory 非法")
    return any(
        answer in normalize_external_text(row[0])
        for row in database.execute("""
            SELECT p.text FROM passage AS p
            JOIN document AS d ON d.doc_id=p.doc_id
            WHERE d.page_id=? ORDER BY p.ordinal
        """, (page_id,))
        for answer in normalized)


def score_joint_retrieval(
        questions_path: str | Path,
        predictions_path: str | Path,
        labels_path: str | Path,
        target_selection: BroadQaTargetSelectionManifest,
        database_path: str | Path,
        *,
        alias_path: str | Path,
        aggregate_path: str | Path,
        scope: str,
        ) -> dict[str, object]:
    """独立读取 labels，聚合联合检索、证据、引用和失败类型。"""
    if (scope not in {"DEVELOPMENT", "FORMAL_HELD_OUT"}
            or not isinstance(target_selection, BroadQaTargetSelectionManifest)):
        raise BroadQaExternalDataError("joint score scope/selection 非法")
    question_file = Path(questions_path).resolve()
    prediction_file = Path(predictions_path).resolve()
    label_file = Path(labels_path).resolve()
    database_file = Path(database_path).resolve()
    target = Path(aggregate_path).resolve()
    if target.exists():
        raise BroadQaExternalDataError("joint aggregate 禁止覆盖")
    questions = _read_jsonl(question_file, expected_kind=JOINT_QUESTION_KIND)
    predictions = _read_jsonl(
        prediction_file, expected_kind=JOINT_PREDICTION_KIND)
    labels = _read_jsonl(label_file, expected_kind=JOINT_LABEL_KIND)
    question_by_id = {item["item_id"]: item for item in questions}
    prediction_by_id = {item["item_id"]: item for item in predictions}
    label_by_id = {item["item_id"]: item for item in labels}
    inventory = set(question_by_id)
    if set(prediction_by_id) != inventory or set(label_by_id) != inventory:
        raise BroadQaExternalDataError("joint score inventory 不一致")
    aliases = {
        item["title_key"]: item
        for item in read_joint_source_aliases(alias_path)
    }
    database = sqlite3.connect(f"file:{database_file}?mode=ro", uri=True)
    recall_hits = 0
    top1_hits = 0
    answer_count = 0
    citation_valid = 0
    evidence_hits = 0
    source_page_gold_coverage = 0
    status_counts: Counter[str] = Counter()
    failure_counts: Counter[str] = Counter()
    per_source: dict[str, Counter[str]] = defaultdict(Counter)
    elapsed_values = []
    candidate_reads = []
    try:
        for item_id in sorted(inventory):
            question = question_by_id[item_id]
            prediction = prediction_by_id[item_id]
            label = label_by_id[item_id]
            if (set(label) != {
                    "expected_title_key", "format_version", "gold_answers",
                    "item_id", "record_kind", "split"}
                    or label["format_version"] != 1
                    or label["split"] != question["split"]
                    or prediction["split"] != question["split"]
                    or prediction["source_key"] != question["source_key"]
                    or not isinstance(label["gold_answers"], list)
                    or not label["gold_answers"]):
                raise BroadQaExternalDataError("joint score record 漂移")
            expected_key = label["expected_title_key"]
            alias = aliases.get(expected_key)
            expected = (
                None if alias is None or alias["status"] != "RESOLVED"
                else (alias["terminal_page_id"], alias["terminal_title"]))
            source_covered = (
                expected is not None
                and _page_contains_gold_answer(
                    database, expected[0], label["gold_answers"]))
            source_page_gold_coverage += int(source_covered)
            candidates = prediction.get("candidates")
            result = prediction.get("result")
            if (not isinstance(candidates, list)
                    or len(candidates) > 20 or not isinstance(result, dict)):
                raise BroadQaExternalDataError("joint prediction schema 漂移")
            recall = expected is not None and any(
                candidate.get("page_id") == expected[0]
                for candidate in candidates)
            top1 = expected is not None and bool(candidates) and (
                candidates[0].get("page_id") == expected[0])
            status = result.get("status")
            status_counts[str(status)] += 1
            answer = status == "ANSWER"
            answer_count += int(answer)
            valid = False
            citations = result.get("citations")
            selected_evidence = []
            if (answer and isinstance(citations, list)
                    and 1 <= len(citations) <= 4):
                citation_validity = []
                for citation in citations:
                    if not isinstance(citation, dict):
                        citation_validity.append(False)
                        continue
                    row = database.execute("""
                        SELECT p.text,d.title,d.revision_id
                        FROM passage AS p
                        JOIN document AS d ON d.doc_id=p.doc_id
                        WHERE d.page_id=? AND d.revision_id=?
                          AND p.raw_start=? AND p.raw_end=? AND p.raw_sha256=?
                        """, (
                            citation.get("page_id"),
                            citation.get("revision_id"),
                            citation.get("evidence_raw_start"),
                            citation.get("evidence_raw_end"),
                            citation.get("evidence_raw_sha256"),
                        )).fetchone()
                    selected_text = citation.get("selected_text")
                    item_valid = (
                        row is not None
                        and row[0] == citation.get("evidence_text")
                        and row[1] == citation.get("title")
                        and row[2] == citation.get("revision_id")
                        and isinstance(selected_text, str)
                        and selected_text
                        and selected_text in row[0])
                    citation_validity.append(item_valid)
                    if item_valid:
                        selected_evidence.append(selected_text)
                valid = (all(citation_validity)
                         and isinstance(result.get("answer"), str)
                         and result["answer"] == "\n".join(selected_evidence))
            source_correct = (
                valid and expected is not None
                and all(citation.get("page_id") == expected[0]
                        for citation in citations))
            evidence_hit = source_correct and any(
                normalize_external_text(answer_text)
                in normalize_external_text("\n".join(selected_evidence))
                for answer_text in label["gold_answers"]
                if isinstance(answer_text, str) and answer_text)
            recall_hits += int(recall)
            top1_hits += int(top1)
            citation_valid += int(valid)
            evidence_hits += int(evidence_hit)
            elapsed_values.append(int(prediction["query_elapsed_ns"]))
            candidate_reads.append(len(candidates))
            source = question["source_key"]
            counters = per_source[source]
            counters["question_count"] += 1
            counters["recall_at_20_count"] += int(recall)
            counters["top1_source_hit_count"] += int(top1)
            counters["evidence_hit_count"] += int(evidence_hit)
            counters["source_page_gold_coverage_count"] += int(
                source_covered)
            if alias is None or alias["status"] != "RESOLVED":
                failure_counts[
                    "SOURCE_ALIAS_" + str(
                        None if alias is None else alias["failure_code"])
                ] += 1
            elif expected is not None and database.execute(
                    "SELECT 1 FROM document WHERE page_id=?", (expected[0],)
                    ).fetchone() is None:
                failure_counts["SOURCE_PAGE_NOT_PROJECTED"] += 1
            elif not source_covered:
                failure_counts["SOURCE_GOLD_ABSENT_FROM_SNAPSHOT"] += 1
            elif not recall:
                failure_counts["RETRIEVAL_MISS_AT_20"] += 1
            elif not top1:
                failure_counts["TOP1_SOURCE_MISS"] += 1
            elif not answer:
                failure_counts["NON_ANSWER"] += 1
            elif not valid:
                failure_counts["CITATION_INVALID"] += 1
            elif not evidence_hit:
                failure_counts["GOLD_NOT_IN_EVIDENCE"] += 1
    finally:
        database.close()
    total = len(inventory)
    recall_ppm = _ppm(recall_hits, total)
    top1_ppm = _ppm(top1_hits, total)
    citation_ppm = _ppm(citation_valid, answer_count)
    evidence_ppm = _ppm(evidence_hits, total)
    source_coverage_ppm = _ppm(source_page_gold_coverage, total)
    conditional_evidence_ppm = _ppm(
        evidence_hits, source_page_gold_coverage)
    passed = (
        recall_ppm >= JOINT_THRESHOLDS["minimum_recall_at_20_ppm"]
        and top1_ppm >= JOINT_THRESHOLDS["minimum_top1_source_hit_ppm"]
        and citation_ppm
        == JOINT_THRESHOLDS["required_answer_citation_valid_ppm"]
        and evidence_ppm >= JOINT_THRESHOLDS["minimum_evidence_hit_ppm"]
    )
    aggregate = {
        "answer_citation_valid_count": citation_valid,
        "answer_citation_valid_ppm": citation_ppm,
        "answer_count": answer_count,
        "artifact_kind": JOINT_AGGREGATE_KIND,
        "candidate_read_p50": _percentile(candidate_reads, 50),
        "candidate_read_p95": _percentile(candidate_reads, 95),
        "database_sha256": _sha256_file(database_file),
        "conditional_evidence_hit_ppm": conditional_evidence_ppm,
        "evidence_hit_count": evidence_hits,
        "evidence_hit_ppm": evidence_ppm,
        "failure_counts": dict(sorted(failure_counts.items())),
        "format_version": 1,
        "labels_sha256": _sha256_file(label_file),
        "per_source": {
            key: {
                **dict(sorted(values.items())),
                "evidence_hit_ppm": _ppm(
                    values["evidence_hit_count"], values["question_count"]),
                "conditional_evidence_hit_ppm": _ppm(
                    values["evidence_hit_count"],
                    values["source_page_gold_coverage_count"]),
                "recall_at_20_ppm": _ppm(
                    values["recall_at_20_count"], values["question_count"]),
                "top1_source_hit_ppm": _ppm(
                    values["top1_source_hit_count"], values["question_count"]),
            }
            for key, values in sorted(per_source.items())
        },
        "predictions_sha256": _sha256_file(prediction_file),
        "query_p50_ns": _percentile(elapsed_values, 50),
        "query_p95_ns": _percentile(elapsed_values, 95),
        "question_count": total,
        "questions_sha256": _sha256_file(question_file),
        "recall_at_20_count": recall_hits,
        "recall_at_20_ppm": recall_ppm,
        "scope": scope,
        "source_page_gold_coverage_count": source_page_gold_coverage,
        "source_page_gold_coverage_ppm": source_coverage_ppm,
        "status": "PASS" if passed else "FAIL",
        "status_counts": dict(sorted(status_counts.items())),
        "target_selection_sha256": target_selection.sha256(),
        "alias_sha256": _sha256_file(Path(alias_path).resolve()),
        "thresholds": JOINT_THRESHOLDS,
        "top1_source_hit_count": top1_hits,
        "top1_source_hit_ppm": top1_ppm,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(canonical_json_line(aggregate))
    return {**aggregate, "aggregate_sha256": _sha256_file(target)}


__all__ = [
    "JOINT_AGGREGATE_KIND",
    "JOINT_ALIAS_KIND",
    "JOINT_SUCCESSOR_PACK_KIND",
    "JOINT_SUCCESSOR_SELECTION_RULE",
    "JOINT_LABEL_KIND",
    "JOINT_PACK_KIND",
    "JOINT_PREDICTION_KIND",
    "JOINT_QUESTION_KIND",
    "JOINT_SELECTION_RULE",
    "JOINT_TARGET_KIND",
    "JOINT_THRESHOLDS",
    "augment_broad_qa_index",
    "freeze_joint_source_pack",
    "predict_joint_retrieval",
    "read_joint_source_targets",
    "read_joint_source_aliases",
    "resolve_joint_source_aliases",
    "score_joint_retrieval",
]
