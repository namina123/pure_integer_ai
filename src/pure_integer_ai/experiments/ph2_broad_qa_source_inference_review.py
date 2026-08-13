"""物化来源内归纳 roster 的终页 review dossier。

本模块只读取已冻结 roster、官方问题 payload、terminal selection 和 Wikipedia
压缩块，为人工或后续合同化 review 提供完整来源证据。它不产生四态 decision，
不运行问答，也不修改既有 roster artifact。
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Iterable

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
    normalize_external_text,
)
from pure_integer_ai.experiments.ph2_broad_qa_selection import (
    read_broad_qa_target_selection,
)
from pure_integer_ai.experiments.ph2_broad_qa_source import (
    iter_broad_qa_selected_page_inspections,
    project_broad_qa_passages,
    project_broad_qa_plain_text,
)
from pure_integer_ai.experiments.ph2_broad_qa_source_dossier import (
    materialize_terminal_sources,
)
from pure_integer_ai.experiments.ph2_broad_qa_source_inference_family import (
    SOURCE_INFERENCE_ASSIGNMENTS,
    SOURCE_INFERENCE_REVIEW_PAYLOAD_KIND,
    SOURCE_INFERENCE_ROSTER_KIND,
    SOURCE_INFERENCE_ROSTER_RECORD_KIND,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
)


SOURCE_INFERENCE_DOSSIER_KIND = (
    "PH2_BROAD_QA_SOURCE_INFERENCE_REVIEW_DOSSIER_V1")
SOURCE_INFERENCE_DOSSIER_RECORD_KIND = (
    "PH2_BROAD_QA_SOURCE_INFERENCE_REVIEW_DOSSIER_RECORD_V1")


def _sha256_file(path: Path) -> str:
    """流式计算冻结输入或 dossier artifact 的 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _within(root: Path, path: str | Path, *, label: str) -> Path:
    """要求所有大数据输入输出均位于显式 run root 内。"""
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(root):
        raise BroadQaExternalDataError(f"{label} 必须位于 run root 内")
    return resolved


def _read_jsonl(
        path: Path,
        *,
        record_kind: str,
        expected_fields: set[str],
        ) -> tuple[dict[str, object], ...]:
    """严格回读规范 JSONL、精确字段和唯一 item identity。"""
    if not path.is_file():
        raise BroadQaExternalDataError("source inference review 输入缺失")
    values = []
    identities = set()
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for line_number, line in enumerate(handle, start=1):
                value = json.loads(line)
                item_id = value.get("item_id") if isinstance(value, dict) else None
                if (not line.endswith("\n") or not isinstance(value, dict)
                        or canonical_json_line(value) != line.encode("utf-8")
                        or set(value) != expected_fields
                        or value.get("record_kind") != record_kind
                        or not isinstance(item_id, str) or not item_id
                        or item_id in identities):
                    raise BroadQaExternalDataError(
                        f"source inference review record 漂移: {line_number}")
                identities.add(item_id)
                values.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "source inference review JSONL 非法") from error
    if not values:
        raise BroadQaExternalDataError("source inference review JSONL 为空")
    return tuple(values)


def _read_roster(path: Path) -> tuple[dict[str, object], ...]:
    """回读不含标签的机械 roster 并核验 assignment/status 边界。"""
    values = _read_jsonl(path, record_kind=SOURCE_INFERENCE_ROSTER_RECORD_KIND,
                         expected_fields={
                             "assignment", "format_version", "item_id",
                             "question_sha256", "record_kind",
                             "source_alignment_status", "source_key",
                             "terminal_page_id", "terminal_revision_id",
                             "title_key",
                         })
    for value in values:
        if (value["format_version"] != 1
                or value["assignment"] not in SOURCE_INFERENCE_ASSIGNMENTS
                or (value["assignment"] == "EXTRACTIVE_CANDIDATE")
                != (value["source_alignment_status"] == "SOURCE_ALIGNED")
                or type(value["terminal_page_id"]) is not int
                or value["terminal_page_id"] <= 0
                or type(value["terminal_revision_id"]) is not int
                or value["terminal_revision_id"] <= 0):
            raise BroadQaExternalDataError(
                "source inference roster 内容漂移")
    return values


def _read_payload(path: Path) -> tuple[dict[str, object], ...]:
    """回读隔离 review payload 并核验问题、上下文和标题承诺。"""
    values = _read_jsonl(path, record_kind=SOURCE_INFERENCE_REVIEW_PAYLOAD_KIND,
                         expected_fields={
                             "assignment", "context", "context_sha256",
                             "format_version", "gold_answers", "item_id",
                             "license_id", "question", "record_kind",
                             "source_key", "source_partition",
                             "source_question_id", "source_revision",
                             "terminal_page_id", "terminal_revision_id",
                             "title", "title_key", "upstream_url",
                         })
    for value in values:
        context = value["context"]
        question = value["question"]
        if (value["format_version"] != 1
                or value["assignment"] not in SOURCE_INFERENCE_ASSIGNMENTS
                or not isinstance(context, str) or not context
                or hashlib.sha256(context.encode("utf-8")).hexdigest()
                != value["context_sha256"]
                or not isinstance(question, str) or not question
                or not isinstance(value["gold_answers"], list)
                or not value["gold_answers"]
                or normalize_external_text(str(value["title"]))
                != value["title_key"]):
            raise BroadQaExternalDataError(
                "source inference review payload 漂移")
    return values


def _manifest_artifact_sha(
        manifest: dict[str, object],
        *,
        role: str,
        ) -> str:
    """从 roster manifest 取得唯一 artifact role 的 SHA-256。"""
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise BroadQaExternalDataError("source inference roster manifest 非法")
    matches = tuple(
        item for item in artifacts
        if isinstance(item, dict) and item.get("role") == role)
    if (len(matches) != 1 or not isinstance(matches[0].get("sha256"), str)):
        raise BroadQaExternalDataError(
            "source inference roster artifact role 漂移")
    return str(matches[0]["sha256"])


def _read_roster_manifest(path: Path) -> dict[str, object]:
    """严格回读 roster manifest 的规范字节和冻结状态。"""
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "source inference roster manifest 不可读") from error
    if (not isinstance(value, dict) or canonical_json_line(value) != payload
            or value.get("artifact_kind") != SOURCE_INFERENCE_ROSTER_KIND
            or value.get("format_version") != 1
            or value.get("status") != "FROZEN_UNREVIEWED_NOT_RUN"):
        raise BroadQaExternalDataError(
            "source inference roster manifest 漂移")
    return value


def publish_source_inference_review_dossier(
        *,
        run_root: str | Path,
        roster_manifest_path: str | Path,
        roster_path: str | Path,
        review_payload_path: str | Path,
        terminal_selection_path: str | Path,
        xml_path: str | Path,
        target_dir: str | Path,
        worker_count: int = 4,
        ) -> dict[str, object]:
    """按固定 roster 物化终页全文和 passage 证据，不产生语义 decision。"""
    root = Path(run_root).resolve()
    if not root.is_dir():
        raise BroadQaExternalDataError("source inference review run root 不存在")
    roster_manifest_file = _within(
        root, roster_manifest_path, label="roster_manifest_path")
    roster_file = _within(root, roster_path, label="roster_path")
    payload_file = _within(
        root, review_payload_path, label="review_payload_path")
    selection_file = _within(
        root, terminal_selection_path, label="terminal_selection_path")
    xml_file = _within(root, xml_path, label="xml_path")
    target = _within(root, target_dir, label="target_dir")
    if target.exists():
        raise BroadQaExternalDataError(
            "source inference review dossier target 已存在")

    manifest = _read_roster_manifest(roster_manifest_file)
    if (_sha256_file(roster_file) != _manifest_artifact_sha(
            manifest, role="review_roster_without_labels")
            or _sha256_file(payload_file) != _manifest_artifact_sha(
                manifest, role="private_development_review_payload")):
        raise BroadQaExternalDataError(
            "source inference roster artifact commitment 漂移")
    roster = _read_roster(roster_file)
    payload = _read_payload(payload_file)
    roster_by_id = {str(item["item_id"]): item for item in roster}
    payload_by_id = {str(item["item_id"]): item for item in payload}
    if set(roster_by_id) != set(payload_by_id):
        raise BroadQaExternalDataError(
            "source inference roster/payload inventory 漂移")
    for item_id, record in roster_by_id.items():
        review = payload_by_id[item_id]
        if (review["assignment"] != record["assignment"]
                or review["source_key"] != record["source_key"]
                or review["title_key"] != record["title_key"]
                or review["terminal_page_id"] != record["terminal_page_id"]
                or review["terminal_revision_id"]
                != record["terminal_revision_id"]
                or hashlib.sha256(
                    str(review["question"]).encode("utf-8")).hexdigest()
                != record["question_sha256"]):
            raise BroadQaExternalDataError(
                "source inference roster/payload binding 漂移")

    selection = read_broad_qa_target_selection(selection_file)
    required_page_revisions = {}
    for item in roster:
        page_id = int(item["terminal_page_id"])
        revision_id = int(item["terminal_revision_id"])
        prior = required_page_revisions.setdefault(page_id, revision_id)
        if prior != revision_id:
            raise BroadQaExternalDataError(
                "source inference terminal revision inventory 冲突")
    terminal_pages = materialize_terminal_sources(
        selection,
        required_page_revisions=required_page_revisions,
        xml_path=xml_file,
        worker_count=worker_count,
        inspection_reader=iter_broad_qa_selected_page_inspections,
        plain_text_projector=project_broad_qa_plain_text,
        passage_projector=project_broad_qa_passages,
    )

    records = []
    for item_id in sorted(roster_by_id):
        roster_record = roster_by_id[item_id]
        review = payload_by_id[item_id]
        terminal = terminal_pages[int(roster_record["terminal_page_id"])]
        if terminal["revision_id"] != roster_record["terminal_revision_id"]:
            raise BroadQaExternalDataError(
                "source inference terminal revision 漂移")
        records.append({
            "assignment": roster_record["assignment"],
            "format_version": 1,
            "item_id": item_id,
            "record_kind": SOURCE_INFERENCE_DOSSIER_RECORD_KIND,
            "review_source": {
                "context": review["context"],
                "context_sha256": review["context_sha256"],
                "gold_answers": review["gold_answers"],
                "license_id": review["license_id"],
                "question": review["question"],
                "source_key": review["source_key"],
                "source_partition": review["source_partition"],
                "source_question_id": review["source_question_id"],
                "source_revision": review["source_revision"],
                "title": review["title"],
                "upstream_url": review["upstream_url"],
            },
            "roster_commitment": {
                "question_sha256": roster_record["question_sha256"],
                "source_alignment_status": roster_record[
                    "source_alignment_status"],
                "title_key": roster_record["title_key"],
            },
            "terminal_source": terminal,
        })

    target.mkdir(parents=True)
    dossier_path = target / "review.dossier.jsonl"
    count = _write_dossier(dossier_path, records)
    if count != len(roster):
        raise BroadQaExternalDataError(
            "source inference dossier record count 漂移")
    assignment_counts = Counter(str(item["assignment"]) for item in records)
    dossier_manifest = {
        "artifact_kind": SOURCE_INFERENCE_DOSSIER_KIND,
        "assignment_counts": {
            key: assignment_counts[key] for key in SOURCE_INFERENCE_ASSIGNMENTS},
        "dossier_bytes": dossier_path.stat().st_size,
        "dossier_record_count": count,
        "dossier_sha256": _sha256_file(dossier_path),
        "format_version": 1,
        "production_query_runs": 0,
        "review_decisions_written": 0,
        "roster_manifest_sha256": _sha256_file(roster_manifest_file),
        "roster_sha256": _sha256_file(roster_file),
        "status": "MATERIALIZED_UNREVIEWED_NOT_RUN",
        "terminal_page_count": len(terminal_pages),
        "terminal_selection_sha256": selection.sha256(),
        "wikipedia_xml_local_sha256": selection.xml_local_sha256,
        "wikipedia_xml_size_bytes": selection.xml_compressed_size_bytes,
    }
    dossier_manifest_path = target / "manifest.json"
    dossier_manifest_path.write_bytes(canonical_json_line(dossier_manifest))
    return {
        **dossier_manifest,
        "manifest_sha256": _sha256_file(dossier_manifest_path),
    }


def _write_dossier(
        path: Path,
        records: Iterable[dict[str, object]],
        ) -> int:
    """不可覆盖地写完整 review dossier。"""
    if path.exists():
        raise BroadQaExternalDataError(
            "source inference review dossier 禁止覆盖")
    count = 0
    with path.open("xb") as handle:
        for record in records:
            handle.write(canonical_json_line(record))
            count += 1
    return count


def _work_path(value: str) -> Path:
    """要求 review 大数据路径为显式绝对 K 盘路径。"""
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("work paths must be absolute")
    resolved = path.resolve()
    if sys.platform == "win32" and resolved.drive.casefold() != "k:":
        raise argparse.ArgumentTypeError("work paths must be on K:")
    return resolved


def main(argv: list[str] | None = None) -> int:
    """从冻结 roster 和 terminal selection 发布完整 review dossier。"""
    parser = argparse.ArgumentParser(
        description="Materialize source evidence for a frozen review roster.")
    parser.add_argument("--run-root", type=_work_path, required=True)
    parser.add_argument("--roster-manifest", type=_work_path, required=True)
    parser.add_argument("--roster", type=_work_path, required=True)
    parser.add_argument("--review-payload", type=_work_path, required=True)
    parser.add_argument("--terminal-selection", type=_work_path, required=True)
    parser.add_argument("--xml", type=_work_path, required=True)
    parser.add_argument("--target-dir", type=_work_path, required=True)
    parser.add_argument(
        "--workers", type=int, choices=(1, 2, 4), default=4)
    args = parser.parse_args(argv)
    report = publish_source_inference_review_dossier(
        run_root=args.run_root,
        roster_manifest_path=args.roster_manifest,
        roster_path=args.roster,
        review_payload_path=args.review_payload,
        terminal_selection_path=args.terminal_selection,
        xml_path=args.xml,
        target_dir=args.target_dir,
        worker_count=args.workers,
    )
    sys.stdout.write(json.dumps(
        report, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SOURCE_INFERENCE_DOSSIER_KIND",
    "SOURCE_INFERENCE_DOSSIER_RECORD_KIND",
    "main",
    "publish_source_inference_review_dossier",
]
