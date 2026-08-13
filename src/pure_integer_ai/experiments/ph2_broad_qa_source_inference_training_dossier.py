"""物化来源归纳训练总体的完整终页 dossier。

本模块只读取冻结 roster/payload 及终页 selection。它保留训练问题和来源证据，
但不决定 operator、不写规则、不调用 learner，也不形成 mastery。
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
from pure_integer_ai.experiments.ph2_broad_qa_source_dossier import (
    materialize_terminal_sources,
)
from pure_integer_ai.experiments.ph2_broad_qa_source_inference_training import (
    SOURCE_INFERENCE_TRAINING_ASSIGNMENTS,
    SOURCE_INFERENCE_TRAINING_KIND,
    SOURCE_INFERENCE_TRAINING_PAYLOAD_KIND,
    SOURCE_INFERENCE_TRAINING_RECORD_KIND,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


SOURCE_INFERENCE_TRAINING_DOSSIER_KIND = (
    "PH2_BROAD_QA_SOURCE_INFERENCE_TRAINING_DOSSIER_V1")
SOURCE_INFERENCE_TRAINING_DOSSIER_RECORD_KIND = (
    "PH2_BROAD_QA_SOURCE_INFERENCE_TRAINING_DOSSIER_RECORD_V1")


def _sha256_file(path: Path) -> str:
    """流式计算冻结输入或 dossier 的 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _within(root: Path, path: str | Path, *, label: str) -> Path:
    """要求训练大数据输入输出始终位于显式 run root 内。"""
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
    """严格回读规范 JSONL、精确字段及唯一 item identity。"""
    values = []
    identities = set()
    try:
        with path.open("rb") as handle:
            for line_number, payload in enumerate(handle, start=1):
                value = json.loads(payload)
                item_id = value.get("item_id") if isinstance(value, dict) else None
                if (not isinstance(value, dict)
                        or canonical_json_line(value) != payload
                        or set(value) != expected_fields
                        or value.get("record_kind") != record_kind
                        or value.get("format_version") != 1
                        or not isinstance(item_id, str) or len(item_id) != 64
                        or any(character not in "0123456789abcdef"
                               for character in item_id)
                        or item_id in identities):
                    raise BroadQaExternalDataError(
                        f"source inference training dossier 输入漂移: "
                        f"{line_number}")
                identities.add(item_id)
                values.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "source inference training dossier JSONL 不可读") from error
    if not values:
        raise BroadQaExternalDataError(
            "source inference training dossier JSONL 为空")
    return tuple(values)


def _read_roster(path: Path) -> tuple[dict[str, object], ...]:
    """回读不含语义标签的训练 roster。"""
    values = _read_jsonl(
        path,
        record_kind=SOURCE_INFERENCE_TRAINING_RECORD_KIND,
        expected_fields={
            "format_version", "item_id", "question_sha256", "record_kind",
            "source_alignment_status", "source_key", "terminal_page_id",
            "terminal_revision_id", "title_key", "training_assignment",
        },
    )
    for value in values:
        if (value["training_assignment"]
                not in SOURCE_INFERENCE_TRAINING_ASSIGNMENTS
                or type(value["terminal_page_id"]) is not int
                or value["terminal_page_id"] <= 0
                or type(value["terminal_revision_id"]) is not int
                or value["terminal_revision_id"] <= 0
                or not isinstance(value["title_key"], str)
                or normalize_external_text(str(value["title_key"]))
                != value["title_key"]):
            raise BroadQaExternalDataError(
                "source inference training roster 内容漂移")
    return values


def _read_payload(path: Path) -> tuple[dict[str, object], ...]:
    """回读完整训练 payload，但不作语义分类。"""
    values = _read_jsonl(
        path,
        record_kind=SOURCE_INFERENCE_TRAINING_PAYLOAD_KIND,
        expected_fields={
            "context", "context_sha256", "format_version", "gold_answers",
            "item_id", "license_id", "question", "record_kind",
            "source_key", "source_partition", "source_question_id",
            "source_revision", "terminal_page_id", "terminal_revision_id",
            "title", "title_key", "training_assignment", "upstream_url",
        },
    )
    for value in values:
        context = value["context"]
        question = value["question"]
        gold_answers = value["gold_answers"]
        if (value["training_assignment"]
                not in SOURCE_INFERENCE_TRAINING_ASSIGNMENTS
                or not isinstance(context, str) or not context
                or hashlib.sha256(context.encode("utf-8")).hexdigest()
                != value["context_sha256"]
                or not isinstance(question, str) or not question
                or not isinstance(gold_answers, list) or not gold_answers
                or any(not isinstance(item, str) or not item
                       for item in gold_answers)
                or normalize_external_text(str(value["title"]))
                != value["title_key"]):
            raise BroadQaExternalDataError(
                "source inference training payload 内容漂移")
    return values


def _manifest_artifact_sha(
        manifest: dict[str, object],
        *,
        role: str,
        ) -> str:
    """从 training manifest 取得唯一 artifact role 的 SHA。"""
    artifacts = manifest.get("artifacts")
    matches = tuple(
        value for value in artifacts if isinstance(value, dict)
        and value.get("role") == role
    ) if isinstance(artifacts, list) else ()
    if (len(matches) != 1
            or not isinstance(matches[0].get("sha256"), str)):
        raise BroadQaExternalDataError(
            "source inference training manifest artifact 漂移")
    return str(matches[0]["sha256"])


def _read_training_manifest(path: Path) -> dict[str, object]:
    """严格回读 learner 尚未读取的冻结训练 manifest。"""
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "source inference training manifest 不可读") from error
    if (not isinstance(value, dict) or canonical_json_line(value) != payload
            or value.get("artifact_kind") != SOURCE_INFERENCE_TRAINING_KIND
            or value.get("format_version") != 1
            or value.get("status") != "FROZEN_NOT_READ_NOT_LEARNED"
            or value.get("learner_read_count_at_freeze") != 0
            or value.get("semantic_labels_written") != 0):
        raise BroadQaExternalDataError(
            "source inference training manifest 漂移")
    return value


def read_source_inference_training_dossier(
        path: str | Path,
        ) -> tuple[dict[str, object], ...]:
    """严格回读完整训练 dossier，供后续只读 census 使用。"""
    values = _read_jsonl(
        Path(path).resolve(),
        record_kind=SOURCE_INFERENCE_TRAINING_DOSSIER_RECORD_KIND,
        expected_fields={
            "format_version", "item_id", "record_kind",
            "roster_commitment", "terminal_source", "training_assignment",
            "training_source",
        },
    )
    for value in values:
        terminal = value["terminal_source"]
        training_source = value["training_source"]
        commitment = value["roster_commitment"]
        expected_terminal_fields = {
            "attribution", "contributor", "license_id", "page_id",
            "passages", "plain_text", "plain_text_sha256", "revision_id",
            "revision_timestamp", "snapshot_id", "source_url", "title",
            "wikitext", "wikitext_sha256",
        }
        expected_training_fields = {
            "context", "context_sha256", "gold_answers", "license_id",
            "question", "source_key", "source_partition",
            "source_question_id", "source_revision", "title", "upstream_url",
        }
        expected_commitment_fields = {
            "question_sha256", "source_alignment_status", "title_key",
        }
        if (value["training_assignment"]
                not in SOURCE_INFERENCE_TRAINING_ASSIGNMENTS
                or not isinstance(terminal, dict)
                or not isinstance(training_source, dict)
                or not isinstance(commitment, dict)
                or set(terminal) != expected_terminal_fields
                or set(training_source) != expected_training_fields
                or set(commitment) != expected_commitment_fields
                or not isinstance(terminal.get("wikitext"), str)
                or not isinstance(terminal.get("plain_text"), str)
                or not isinstance(terminal.get("passages"), list)
                or not isinstance(training_source.get("gold_answers"), list)
                or not isinstance(training_source.get("question"), str)
                or not isinstance(training_source.get("source_key"), str)
                or commitment.get("title_key")
                != normalize_external_text(str(training_source.get("title")))
                or type(terminal.get("page_id")) is not int
                or type(terminal.get("revision_id")) is not int):
            raise BroadQaExternalDataError(
                "source inference training dossier 内容漂移")
        context = str(training_source["context"])
        question = str(training_source["question"])
        wikitext = str(terminal["wikitext"])
        plain_text = str(terminal["plain_text"])
        if (hashlib.sha256(context.encode("utf-8")).hexdigest()
                != training_source["context_sha256"]
                or hashlib.sha256(question.encode("utf-8")).hexdigest()
                != commitment["question_sha256"]
                or hashlib.sha256(wikitext.encode("utf-8")).hexdigest()
                != terminal["wikitext_sha256"]
                or hashlib.sha256(plain_text.encode("utf-8")).hexdigest()
                != terminal["plain_text_sha256"]):
            raise BroadQaExternalDataError(
                "source inference training dossier hash 漂移")
        required_passage_fields = {
            "ordinal", "raw_end", "raw_sha256", "raw_start",
            "section_title", "text", "text_sha256",
        }
        seen_ordinals = set()
        for passage in terminal["passages"]:
            if (not isinstance(passage, dict)
                    or set(passage) != required_passage_fields
                    or type(passage.get("ordinal")) is not int
                    or passage["ordinal"] <= 0
                    or passage["ordinal"] in seen_ordinals
                    or type(passage.get("raw_start")) is not int
                    or type(passage.get("raw_end")) is not int
                    or not 0 <= passage["raw_start"] < passage["raw_end"]
                    <= len(wikitext)
                    or not isinstance(passage.get("text"), str)
                    or not isinstance(passage.get("section_title"), str)):
                raise BroadQaExternalDataError(
                    "source inference training dossier passage 漂移")
            seen_ordinals.add(passage["ordinal"])
            raw = wikitext[passage["raw_start"]:passage["raw_end"]]
            if (hashlib.sha256(raw.encode("utf-8")).hexdigest()
                    != passage["raw_sha256"]
                    or hashlib.sha256(
                        str(passage["text"]).encode("utf-8")).hexdigest()
                    != passage["text_sha256"]):
                raise BroadQaExternalDataError(
                    "source inference training dossier passage hash 漂移")
    return values


def publish_source_inference_training_dossier(
        *,
        run_root: str | Path,
        roster_manifest_path: str | Path,
        roster_path: str | Path,
        training_payload_path: str | Path,
        terminal_selection_path: str | Path,
        xml_path: str | Path,
        target_dir: str | Path,
        worker_count: int = 4,
        ) -> dict[str, object]:
    """物化训练样本终页全文，不调用 learner 或写入语义结果。"""
    root = Path(run_root).resolve()
    if not root.is_dir():
        raise BroadQaExternalDataError(
            "source inference training dossier run root 不存在")
    manifest_file = _within(
        root, roster_manifest_path, label="roster_manifest_path")
    roster_file = _within(root, roster_path, label="roster_path")
    payload_file = _within(
        root, training_payload_path, label="training_payload_path")
    selection_file = _within(
        root, terminal_selection_path, label="terminal_selection_path")
    xml_file = _within(root, xml_path, label="xml_path")
    target = _within(root, target_dir, label="target_dir")
    if target.exists():
        raise BroadQaExternalDataError(
            "source inference training dossier target 已存在")

    manifest = _read_training_manifest(manifest_file)
    if (_sha256_file(roster_file) != _manifest_artifact_sha(
            manifest, role="training_roster_without_semantic_labels")
            or _sha256_file(payload_file) != _manifest_artifact_sha(
                manifest, role="training_payload_frozen_before_learner_read")):
        raise BroadQaExternalDataError(
            "source inference training artifact commitment 漂移")
    roster = _read_roster(roster_file)
    payload = _read_payload(payload_file)
    roster_by_id = {str(value["item_id"]): value for value in roster}
    payload_by_id = {str(value["item_id"]): value for value in payload}
    if set(roster_by_id) != set(payload_by_id):
        raise BroadQaExternalDataError(
            "source inference training roster/payload inventory 漂移")
    required_page_revisions = {}
    for item_id, roster_record in roster_by_id.items():
        training = payload_by_id[item_id]
        if (training["training_assignment"]
                != roster_record["training_assignment"]
                or training["source_key"] != roster_record["source_key"]
                or training["title_key"] != roster_record["title_key"]
                or training["terminal_page_id"]
                != roster_record["terminal_page_id"]
                or training["terminal_revision_id"]
                != roster_record["terminal_revision_id"]
                or hashlib.sha256(
                    str(training["question"]).encode("utf-8")).hexdigest()
                != roster_record["question_sha256"]):
            raise BroadQaExternalDataError(
                "source inference training roster/payload binding 漂移")
        page_id = int(roster_record["terminal_page_id"])
        revision_id = int(roster_record["terminal_revision_id"])
        prior = required_page_revisions.setdefault(page_id, revision_id)
        if prior != revision_id:
            raise BroadQaExternalDataError(
                "source inference training terminal revision inventory 冲突")

    selection = read_broad_qa_target_selection(selection_file)
    terminal_sources = materialize_terminal_sources(
        selection,
        required_page_revisions=required_page_revisions,
        xml_path=xml_file,
        worker_count=worker_count,
    )
    records = []
    for item_id in sorted(roster_by_id):
        roster_record = roster_by_id[item_id]
        training = payload_by_id[item_id]
        records.append({
            "format_version": 1,
            "item_id": item_id,
            "record_kind": SOURCE_INFERENCE_TRAINING_DOSSIER_RECORD_KIND,
            "roster_commitment": {
                "question_sha256": roster_record["question_sha256"],
                "source_alignment_status": roster_record[
                    "source_alignment_status"],
                "title_key": roster_record["title_key"],
            },
            "terminal_source": terminal_sources[
                int(roster_record["terminal_page_id"])],
            "training_assignment": roster_record["training_assignment"],
            "training_source": {
                "context": training["context"],
                "context_sha256": training["context_sha256"],
                "gold_answers": training["gold_answers"],
                "license_id": training["license_id"],
                "question": training["question"],
                "source_key": training["source_key"],
                "source_partition": training["source_partition"],
                "source_question_id": training["source_question_id"],
                "source_revision": training["source_revision"],
                "title": training["title"],
                "upstream_url": training["upstream_url"],
            },
        })

    target.mkdir(parents=True)
    dossier_path = target / "training.dossier.jsonl"
    with dossier_path.open("xb") as handle:
        for record in records:
            handle.write(canonical_json_line(record))
    assignment_counts = Counter(
        str(value["training_assignment"]) for value in records)
    dossier_manifest = {
        "artifact_kind": SOURCE_INFERENCE_TRAINING_DOSSIER_KIND,
        "assignment_counts": {
            assignment: assignment_counts[assignment]
            for assignment in SOURCE_INFERENCE_TRAINING_ASSIGNMENTS
        },
        "dossier_bytes": dossier_path.stat().st_size,
        "dossier_record_count": len(records),
        "dossier_sha256": _sha256_file(dossier_path),
        "format_version": 1,
        "learner_read_count": 0,
        "production_query_runs": 0,
        "roster_manifest_sha256": _sha256_file(manifest_file),
        "roster_sha256": _sha256_file(roster_file),
        "rules_written": 0,
        "semantic_labels_written": 0,
        "status": "MATERIALIZED_UNREAD_UNLEARNED",
        "terminal_page_count": len(terminal_sources),
        "terminal_selection_sha256": selection.sha256(),
        "training_payload_sha256": _sha256_file(payload_file),
        "wikipedia_xml_local_sha256": selection.xml_local_sha256,
        "wikipedia_xml_size_bytes": selection.xml_compressed_size_bytes,
    }
    dossier_manifest_path = target / "manifest.json"
    dossier_manifest_path.write_bytes(canonical_json_line(dossier_manifest))
    return {
        **dossier_manifest,
        "manifest_sha256": _sha256_file(dossier_manifest_path),
    }


def _work_path(value: str) -> Path:
    """要求训练 dossier 大数据路径为显式绝对 K 盘路径。"""
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("work paths must be absolute")
    resolved = path.resolve()
    if sys.platform == "win32" and resolved.drive.casefold() != "k:":
        raise argparse.ArgumentTypeError("work paths must be on K:")
    return resolved


def main(argv: list[str] | None = None) -> int:
    """从冻结训练总体发布完整终页 dossier。"""
    parser = argparse.ArgumentParser(
        description="Materialize the source-inference training dossier.")
    parser.add_argument("--run-root", type=_work_path, required=True)
    parser.add_argument("--roster-manifest", type=_work_path, required=True)
    parser.add_argument("--roster", type=_work_path, required=True)
    parser.add_argument("--training-payload", type=_work_path, required=True)
    parser.add_argument("--terminal-selection", type=_work_path, required=True)
    parser.add_argument("--xml", type=_work_path, required=True)
    parser.add_argument("--target-dir", type=_work_path, required=True)
    parser.add_argument("--worker-count", type=int, choices=(1, 2, 4), default=4)
    args = parser.parse_args(argv)
    report = publish_source_inference_training_dossier(
        run_root=args.run_root,
        roster_manifest_path=args.roster_manifest,
        roster_path=args.roster,
        training_payload_path=args.training_payload,
        terminal_selection_path=args.terminal_selection,
        xml_path=args.xml,
        target_dir=args.target_dir,
        worker_count=args.worker_count,
    )
    sys.stdout.write(json.dumps(
        report, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SOURCE_INFERENCE_TRAINING_DOSSIER_KIND",
    "SOURCE_INFERENCE_TRAINING_DOSSIER_RECORD_KIND",
    "main",
    "publish_source_inference_training_dossier",
    "read_source_inference_training_dossier",
]
