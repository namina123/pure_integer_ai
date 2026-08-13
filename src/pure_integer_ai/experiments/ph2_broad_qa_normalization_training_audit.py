"""审计来源归纳规范化训练片是否具备可发布规则所需的对比证据。

本模块只读取 learning protocol v2 的 LEARNER/TRAIN 物理切片。它可以复算既有
机械对齐并盘点字符改写库存，但不把机械信号转成语义标签、Evidence 或规则，也不
读取 VALIDATION/RESERVE。缺少来源化非等价标签时必须保持 BLOCKED。
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
    normalize_external_text,
)
from pure_integer_ai.experiments.ph2_broad_qa_source_inference_learning_protocol import (
    read_source_inference_learning_protocol,
    read_source_inference_learning_slice,
)
from pure_integer_ai.experiments.ph2_broad_qa_source_inference_training_census import (
    MECHANICAL_SIGNAL_STATES,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


NORMALIZATION_TRAINING_AUDIT_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_TRAINING_CONTRASTIVE_AUDIT_V1")
NORMALIZATION_TRAINING_AUDIT_RECORD_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_TRAINING_CONTRASTIVE_RECORD_V1")
NORMALIZATION_TRAINING_AUDIT_STATUS = (
    "BLOCKED_INSUFFICIENT_CONTRASTIVE_EVIDENCE")
NORMALIZATION_TRAINING_AUDIT_FAMILY = "NORMALIZATION_EQUIVALENCE"


def _sha256_file(path: Path) -> str:
    """流式计算正式输入或输出文件摘要。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _sha256(value: object, *, label: str) -> str:
    """要求 artifact 承诺为小写 SHA-256。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise BroadQaExternalDataError(f"{label} 必须是 SHA-256")
    return value


def _within(root: Path, path: str | Path, *, label: str) -> Path:
    """要求 audit 输出位于显式 run root 内。"""
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(root):
        raise BroadQaExternalDataError(f"{label} 必须位于 run root 内")
    return resolved


def _project_with_raw_coordinates(
        value: str,
        ) -> tuple[str, tuple[int, ...], tuple[int, ...]]:
    """机械规范化字符串，并保留每个输出码点的原始坐标。"""
    projected = []
    starts = []
    ends = []
    for index, character in enumerate(value):
        mapped = normalize_external_text(character)
        for output in mapped:
            projected.append(output)
            starts.append(index)
            ends.append(index + 1)
    return "".join(projected), tuple(starts), tuple(ends)


def _occurrences(value: str, target: str) -> tuple[int, ...]:
    """返回允许重叠的全部非空 target 起点。"""
    if not target:
        raise BroadQaExternalDataError("normalization audit target 为空")
    starts = []
    cursor = value.find(target)
    while cursor >= 0:
        starts.append(cursor)
        cursor = value.find(target, cursor + 1)
    return tuple(starts)


def audit_normalization_training_records(
        dossier: tuple[dict[str, object], ...],
        census: tuple[dict[str, object], ...],
        ) -> tuple[tuple[dict[str, object], ...], dict[str, object]]:
    """盘点 TRAIN 的机械对齐、passage 覆盖和非语义 rewrite 库存。"""
    if (not isinstance(dossier, tuple) or not dossier
            or not isinstance(census, tuple) or len(dossier) != len(census)):
        raise BroadQaExternalDataError(
            "normalization audit dossier/census 计数非法")
    census_by_id = {str(item.get("item_id")): item for item in census}
    dossier_ids = tuple(str(item.get("item_id")) for item in dossier)
    if (len(set(dossier_ids)) != len(dossier_ids)
            or set(dossier_ids) != set(census_by_id)):
        raise BroadQaExternalDataError(
            "normalization audit dossier/census identity 漂移")

    routing_counts: Counter[str] = Counter()
    rewrite_items: defaultdict[tuple[int, int], set[str]] = defaultdict(set)
    audit_records = []
    aligned_item_count = 0
    passage_aligned_item_count = 0
    ambiguous_surface_item_count = 0
    non_position_preserving_candidate_count = 0
    for item in dossier:
        item_id = str(item["item_id"])
        census_item = census_by_id[item_id]
        routing_signal = str(census_item.get("mechanical_signal_state"))
        routing_counts[routing_signal] += 1
        terminal = item["terminal_source"]
        training = item["training_source"]
        wikitext = terminal["wikitext"]
        projected, starts, ends = _project_with_raw_coordinates(wikitext)
        candidates = {}
        for gold in training["gold_answers"]:
            normalized_gold = normalize_external_text(gold)
            for position in _occurrences(projected, normalized_gold):
                raw_start = starts[position]
                raw_end = ends[position + len(normalized_gold) - 1]
                raw = wikitext[raw_start:raw_end]
                passages = tuple(
                    passage for passage in terminal["passages"]
                    if passage["raw_start"] <= raw_start
                    and raw_end <= passage["raw_end"])
                key = (raw, normalized_gold)
                previous = candidates.get(key)
                candidate = {
                    "passage_ordinals": [
                        passage["ordinal"] for passage in passages],
                    "position_preserving": int(
                        len(raw) == len(normalized_gold)),
                    "raw_end": raw_end,
                    "raw_sha256": hashlib.sha256(
                        raw.encode("utf-8")).hexdigest(),
                    "raw_start": raw_start,
                    "rewrite_pair_count": sum(
                        left != right
                        for left, right in zip(raw, normalized_gold))
                    if len(raw) == len(normalized_gold) else 0,
                }
                if (previous is None
                        or (not previous["passage_ordinals"]
                            and candidate["passage_ordinals"])
                        or (bool(previous["passage_ordinals"])
                            == bool(candidate["passage_ordinals"])
                            and candidate["raw_start"] < previous["raw_start"])):
                    candidates[key] = candidate
        aligned = int(bool(candidates))
        passage_candidates = tuple(
            value for value in candidates.values()
            if value["passage_ordinals"])
        aligned_item_count += aligned
        passage_aligned_item_count += int(bool(passage_candidates))
        ambiguous_surface_item_count += int(len(candidates) > 1)
        for (raw, normalized), candidate in candidates.items():
            if not candidate["passage_ordinals"]:
                continue
            if len(raw) != len(normalized):
                non_position_preserving_candidate_count += 1
                continue
            for left, right in zip(raw, normalized):
                if left != right:
                    rewrite_items[(ord(left), ord(right))].add(item_id)
        audit_records.append({
            "aligned_surface_count": len(candidates),
            "format_version": 1,
            "item_id": item_id,
            "operator_family": NORMALIZATION_TRAINING_AUDIT_FAMILY,
            "passage_aligned_surface_count": len(passage_candidates),
            "record_kind": NORMALIZATION_TRAINING_AUDIT_RECORD_KIND,
            "routing_signal_state": routing_signal,
            "semantic_non_equivalence_label_count": 0,
            "source_key": training["source_key"],
        })

    rewrite_conflict_count = sum(
        len({right for left, right in rewrite_items if left == source}) > 1
        for source in {left for left, _ in rewrite_items})
    rewrite_single_item_count = sum(
        len(item_ids) == 1 for item_ids in rewrite_items.values())
    report = {
        "aligned_item_count": aligned_item_count,
        "ambiguous_surface_item_count": ambiguous_surface_item_count,
        "contrastive_refute_evidence_count": 0,
        "item_count": len(dossier),
        "non_position_preserving_candidate_count": (
            non_position_preserving_candidate_count),
        "passage_aligned_item_count": passage_aligned_item_count,
        "rewrite_conflict_count": rewrite_conflict_count,
        "rewrite_pair_count": len(rewrite_items),
        "rewrite_single_item_count": rewrite_single_item_count,
        "routing_signal_counts": {
            state: routing_counts[state] for state in MECHANICAL_SIGNAL_STATES},
        "semantic_non_equivalence_label_count": 0,
        "status": NORMALIZATION_TRAINING_AUDIT_STATUS,
    }
    return tuple(audit_records), report


def publish_normalization_training_audit(
        *,
        run_root: str | Path,
        protocol_dir: str | Path,
        target_dir: str | Path,
        ) -> dict[str, object]:
    """不可覆盖发布 TRAIN-only 对比证据审计。"""
    root = Path(run_root).resolve()
    if not root.is_dir():
        raise BroadQaExternalDataError("normalization audit run root 不存在")
    protocol_root = _within(root, protocol_dir, label="protocol_dir")
    target = _within(root, target_dir, label="target_dir")
    if target.exists():
        raise BroadQaExternalDataError("normalization audit target 已存在")
    protocol = read_source_inference_learning_protocol(
        protocol_root / "manifest.json")
    dossier, census = read_source_inference_learning_slice(
        protocol_dir=protocol_root,
        access_role="LEARNER",
        operator_family=NORMALIZATION_TRAINING_AUDIT_FAMILY,
    )
    records, report = audit_normalization_training_records(dossier, census)
    target.mkdir(parents=True)
    records_path = target / "audit.records.jsonl"
    with records_path.open("xb") as handle:
        for record in records:
            handle.write(canonical_json_line(record))
    manifest = {
        **report,
        "artifact_kind": NORMALIZATION_TRAINING_AUDIT_KIND,
        "format_version": 1,
        "learner_payload_read_count": len(dossier),
        "mastery_written": 0,
        "operator_family": NORMALIZATION_TRAINING_AUDIT_FAMILY,
        "production_query_count": 0,
        "protocol_manifest_sha256": protocol["manifest_sha256"],
        "records_bytes": records_path.stat().st_size,
        "records_sha256": _sha256_file(records_path),
        "rules_written": 0,
        "validation_payload_read_count": 0,
    }
    manifest_path = target / "manifest.json"
    manifest_path.write_bytes(canonical_json_line(manifest))
    return {
        **manifest,
        "manifest_sha256": _sha256_file(manifest_path),
    }


def read_normalization_training_audit(
        target_dir: str | Path,
        ) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
    """严格回读 audit manifest 与逐 item 规范记录。"""
    root = Path(target_dir).resolve()
    manifest_path = root / "manifest.json"
    records_path = root / "audit.records.jsonl"
    try:
        manifest_payload = manifest_path.read_bytes()
        manifest = json.loads(manifest_payload)
        records_payload = records_path.read_bytes()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError("normalization audit 不可读") from error
    expected_manifest = {
        "aligned_item_count", "ambiguous_surface_item_count", "artifact_kind",
        "contrastive_refute_evidence_count", "format_version", "item_count",
        "learner_payload_read_count", "mastery_written",
        "non_position_preserving_candidate_count", "operator_family",
        "passage_aligned_item_count", "production_query_count",
        "protocol_manifest_sha256", "records_bytes", "records_sha256",
        "rewrite_conflict_count", "rewrite_pair_count",
        "rewrite_single_item_count", "routing_signal_counts", "rules_written",
        "semantic_non_equivalence_label_count", "status",
        "validation_payload_read_count",
    }
    if (not isinstance(manifest, dict) or set(manifest) != expected_manifest
            or canonical_json_line(manifest) != manifest_payload
            or manifest["artifact_kind"] != NORMALIZATION_TRAINING_AUDIT_KIND
            or type(manifest["format_version"]) is not int
            or manifest["format_version"] != 1
            or manifest["operator_family"]
            != NORMALIZATION_TRAINING_AUDIT_FAMILY
            or manifest["status"] != NORMALIZATION_TRAINING_AUDIT_STATUS
            or type(manifest["item_count"]) is not int
            or manifest["item_count"] <= 0
            or any(type(manifest[name]) is not int or manifest[name] < 0
                   for name in (
                       "aligned_item_count", "ambiguous_surface_item_count",
                       "non_position_preserving_candidate_count",
                       "passage_aligned_item_count", "records_bytes",
                       "rewrite_conflict_count", "rewrite_pair_count",
                       "rewrite_single_item_count"))
            or any(type(manifest[name]) is not int or manifest[name] != 0
                   for name in (
                       "contrastive_refute_evidence_count", "mastery_written",
                       "production_query_count", "rules_written",
                       "semantic_non_equivalence_label_count",
                       "validation_payload_read_count"))
            or type(manifest["learner_payload_read_count"]) is not int
            or manifest["learner_payload_read_count"] != manifest["item_count"]
            or manifest["aligned_item_count"] > manifest["item_count"]
            or manifest["passage_aligned_item_count"]
            > manifest["aligned_item_count"]
            or manifest["ambiguous_surface_item_count"]
            > manifest["aligned_item_count"]
            or manifest["rewrite_conflict_count"]
            > manifest["rewrite_pair_count"]
            or manifest["rewrite_single_item_count"]
            > manifest["rewrite_pair_count"]
            or manifest["records_bytes"] != len(records_payload)
            or manifest["records_sha256"]
            != hashlib.sha256(records_payload).hexdigest()):
        raise BroadQaExternalDataError("normalization audit manifest 漂移")
    _sha256(
        manifest["protocol_manifest_sha256"],
        label="normalization audit protocol",
    )
    _sha256(manifest["records_sha256"], label="normalization audit records")
    routing_counts = manifest["routing_signal_counts"]
    if (not isinstance(routing_counts, dict)
            or set(routing_counts) != set(MECHANICAL_SIGNAL_STATES)
            or any(type(value) is not int or value < 0
                   for value in routing_counts.values())
            or sum(routing_counts.values()) != manifest["item_count"]):
        raise BroadQaExternalDataError(
            "normalization audit routing counts 漂移")
    expected_record = {
        "aligned_surface_count", "format_version", "item_id",
        "operator_family", "passage_aligned_surface_count", "record_kind",
        "routing_signal_state", "semantic_non_equivalence_label_count",
        "source_key",
    }
    records = []
    identities = set()
    for line_number, line in enumerate(records_payload.splitlines(
            keepends=True), start=1):
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise BroadQaExternalDataError(
                "normalization audit record 不可读") from error
        if (not isinstance(value, dict) or set(value) != expected_record
                or canonical_json_line(value) != line
                or type(value["format_version"]) is not int
                or value["format_version"] != 1
                or value["record_kind"]
                != NORMALIZATION_TRAINING_AUDIT_RECORD_KIND
                or value["operator_family"]
                != NORMALIZATION_TRAINING_AUDIT_FAMILY
                or not isinstance(value["source_key"], str)
                or not value["source_key"]
                or type(value["aligned_surface_count"]) is not int
                or value["aligned_surface_count"] < 0
                or type(value["passage_aligned_surface_count"]) is not int
                or not 0 <= value["passage_aligned_surface_count"]
                <= value["aligned_surface_count"]
                or value["routing_signal_state"]
                not in MECHANICAL_SIGNAL_STATES
                or type(value["semantic_non_equivalence_label_count"])
                is not int
                or value["semantic_non_equivalence_label_count"] != 0
                or not isinstance(value["item_id"], str)
                or len(value["item_id"]) != 64
                or any(character not in "0123456789abcdef"
                       for character in value["item_id"])
                or value["item_id"] in identities):
            raise BroadQaExternalDataError(
                f"normalization audit record 漂移: {line_number}")
        identities.add(value["item_id"])
        records.append(value)
    if len(records) != manifest["item_count"]:
        raise BroadQaExternalDataError("normalization audit record count 漂移")
    observed_routing = Counter(
        item["routing_signal_state"] for item in records)
    if (sum(item["aligned_surface_count"] > 0 for item in records)
            != manifest["aligned_item_count"]
            or sum(item["passage_aligned_surface_count"] > 0
                   for item in records)
            != manifest["passage_aligned_item_count"]
            or sum(item["aligned_surface_count"] > 1 for item in records)
            != manifest["ambiguous_surface_item_count"]
            or {state: observed_routing[state]
                for state in MECHANICAL_SIGNAL_STATES} != routing_counts):
        raise BroadQaExternalDataError(
            "normalization audit aggregate/records 漂移")
    return ({
        **manifest,
        "manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
    }, tuple(records))


def _work_path(value: str) -> Path:
    """要求正式 audit 路径为显式绝对 K 盘路径。"""
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("work paths must be absolute")
    resolved = path.resolve()
    if sys.platform == "win32" and resolved.drive.casefold() != "k:":
        raise argparse.ArgumentTypeError("work paths must be on K:")
    return resolved


def main(argv: list[str] | None = None) -> int:
    """发布 normalization TRAIN-only contrastive audit。"""
    parser = argparse.ArgumentParser(
        description="Audit normalization TRAIN contrastive evidence.")
    parser.add_argument("--run-root", type=_work_path, required=True)
    parser.add_argument("--protocol-dir", type=_work_path, required=True)
    parser.add_argument("--target-dir", type=_work_path, required=True)
    args = parser.parse_args(argv)
    report = publish_normalization_training_audit(
        run_root=args.run_root,
        protocol_dir=args.protocol_dir,
        target_dir=args.target_dir,
    )
    sys.stdout.write(json.dumps(
        report, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "NORMALIZATION_TRAINING_AUDIT_FAMILY",
    "NORMALIZATION_TRAINING_AUDIT_KIND",
    "NORMALIZATION_TRAINING_AUDIT_RECORD_KIND",
    "NORMALIZATION_TRAINING_AUDIT_STATUS",
    "audit_normalization_training_records",
    "main",
    "publish_normalization_training_audit",
    "read_normalization_training_audit",
]
