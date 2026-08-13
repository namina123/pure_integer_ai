"""来源归纳 worksheet 的无语义标签可行性审计。

本模块只计算终页字节能否按序组成 gold，以及是否已有在评测前绑定的规则包。
它不判断来源冲突、不生成 inference record，也不把机械可拼接性当作语义可推导。
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
    normalize_external_text,
)
from pure_integer_ai.experiments.ph2_broad_qa_source_inference_decision import (
    SOURCE_INFERENCE_REVIEW_WORKSHEET_RECORD_KIND,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


SOURCE_INFERENCE_FEASIBILITY_KIND = (
    "PH2_BROAD_QA_SOURCE_INFERENCE_FEASIBILITY_AUDIT_V1")
SOURCE_INFERENCE_FEASIBILITY_RECORD_KIND = (
    "PH2_BROAD_QA_SOURCE_INFERENCE_FEASIBILITY_RECORD_V1")
_UNREACHABLE = -1


def _sha256_file(path: Path) -> str:
    """流式计算文件 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def _read_worksheet(path: Path) -> tuple[dict[str, object], ...]:
    """严格回读未审 worksheet，拒绝预写 decision 或字段漂移。"""
    values = []
    try:
        with path.open("rb") as handle:
            for line_number, payload in enumerate(handle, 1):
                value = json.loads(payload)
                if (not isinstance(value, dict)
                        or canonical_json_line(value) != payload
                        or value.get("record_kind")
                        != SOURCE_INFERENCE_REVIEW_WORKSHEET_RECORD_KIND
                        or value.get("format_version") != 1
                        or value.get("decision") != "UNREVIEWED"):
                    raise BroadQaExternalDataError(
                        f"source inference worksheet 漂移: {line_number}")
                values.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "source inference worksheet 不可读") from error
    if not values:
        raise BroadQaExternalDataError("source inference worksheet 为空")
    return tuple(values)


def _minimum_source_segments(source: str, target: str) -> int:
    """返回按序从 source 连续片段拼成 target 所需的最少片段数。"""
    if not target:
        return _UNREACHABLE
    unreachable = len(target) + 1
    outside = [unreachable] * (len(target) + 1)
    inside = [unreachable] * (len(target) + 1)
    outside[0] = 0
    for character in source:
        next_outside = outside[:]
        next_inside = [unreachable] * (len(target) + 1)
        for index in range(len(target) + 1):
            if inside[index] < next_outside[index]:
                next_outside[index] = inside[index]
            if index >= len(target) or target[index] != character:
                continue
            if outside[index] < unreachable:
                next_inside[index + 1] = min(
                    next_inside[index + 1], outside[index] + 1)
            if inside[index] < unreachable:
                next_inside[index + 1] = min(
                    next_inside[index + 1], inside[index])
        outside, inside = next_outside, next_inside
    result = min(outside[-1], inside[-1])
    return _UNREACHABLE if result >= unreachable else result


def audit_source_inference_feasibility(
        worksheet_path: str | Path,
        *,
        target_dir: str | Path,
        prebound_rule_pack_sha256s: tuple[str, ...] = (),
        ) -> dict[str, object]:
    """发布无 decision 的字节可拼接库存与规则包就绪状态。"""
    worksheet_file = Path(worksheet_path).resolve()
    target = Path(target_dir).resolve()
    if (target.exists()
            or not isinstance(prebound_rule_pack_sha256s, tuple)
            or prebound_rule_pack_sha256s
            != tuple(sorted(set(prebound_rule_pack_sha256s)))):
        raise BroadQaExternalDataError(
            "source inference feasibility 输出或规则包 inventory 非法")
    for digest in prebound_rule_pack_sha256s:
        if (len(digest) != 64
                or any(character not in "0123456789abcdef"
                       for character in digest)):
            raise BroadQaExternalDataError(
                "source inference feasibility rule pack SHA 非法")
    worksheet = _read_worksheet(worksheet_file)
    records = []
    segment_counts: Counter[str] = Counter()
    assignment_counts: Counter[str] = Counter()
    normalized_direct_hit_count = 0
    for value in worksheet:
        passages = value.get("terminal_passages")
        gold_answers = value.get("gold_answers")
        if (not isinstance(passages, list)
                or not isinstance(gold_answers, list)
                or not gold_answers
                or any(not isinstance(item, str) or not item
                       for item in gold_answers)
                or any(not isinstance(item, dict)
                       or not isinstance(item.get("text"), str)
                       for item in passages)):
            raise BroadQaExternalDataError(
                "source inference feasibility worksheet 字段非法")
        source = normalize_external_text("".join(
            str(item["text"]) for item in passages))
        gold_values = tuple(normalize_external_text(item)
                            for item in gold_answers)
        direct_hit = int(any(item in source for item in gold_values))
        normalized_direct_hit_count += direct_hit
        minimum = min(
            _minimum_source_segments(source, item) for item in gold_values)
        segment_key = "UNREACHABLE" if minimum == _UNREACHABLE else str(minimum)
        segment_counts[segment_key] += 1
        assignment = str(value.get("assignment"))
        assignment_counts[assignment] += 1
        records.append({
            "assignment": assignment,
            "format_version": 1,
            "item_id": value["item_id"],
            "minimum_source_segment_count": (
                None if minimum == _UNREACHABLE else minimum),
            "normalized_direct_hit": direct_hit,
            "record_kind": SOURCE_INFERENCE_FEASIBILITY_RECORD_KIND,
            "semantic_decision_written": 0,
        })
    status = ("READY_FOR_INDEPENDENT_RULE_VALIDATION"
              if prebound_rule_pack_sha256s
              else "BLOCKED_NO_PREBOUND_RULE_PACK")
    target.mkdir(parents=True)
    records_path = target / "feasibility.records.jsonl"
    with records_path.open("xb") as handle:
        for record in records:
            handle.write(canonical_json_line(record))
    manifest = {
        "artifact_kind": SOURCE_INFERENCE_FEASIBILITY_KIND,
        "assignment_counts": dict(sorted(assignment_counts.items())),
        "feasibility_records_sha256": _sha256_file(records_path),
        "format_version": 1,
        "item_count": len(records),
        "normalized_direct_hit_count": normalized_direct_hit_count,
        "prebound_rule_pack_count": len(prebound_rule_pack_sha256s),
        "prebound_rule_pack_sha256s": list(prebound_rule_pack_sha256s),
        "segment_count_distribution": dict(sorted(segment_counts.items())),
        "semantic_decisions_written": 0,
        "status": status,
        "worksheet_sha256": _sha256_file(worksheet_file),
    }
    manifest_path = target / "manifest.json"
    manifest_path.write_bytes(canonical_json_line(manifest))
    return {**manifest, "manifest_sha256": _sha256_file(manifest_path)}


__all__ = [
    "SOURCE_INFERENCE_FEASIBILITY_KIND",
    "SOURCE_INFERENCE_FEASIBILITY_RECORD_KIND",
    "audit_source_inference_feasibility",
]
