"""运行标签隔离的外部证据选择，并独立聚合金答案命中。"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import time
from typing import Iterable

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
    normalize_external_text,
)
from pure_integer_ai.experiments.ph2_broad_qa_query import (
    select_broad_qa_evidence_sentence,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


PREDICTION_KIND = "PH2_BROAD_QA_EXTERNAL_EVIDENCE_PREDICTION_V1"
AGGREGATE_KIND = "PH2_BROAD_QA_EXTERNAL_EVIDENCE_AGGREGATE_V1"


def _sha256_file(path: Path) -> str:
    """流式计算已冻结输入或结果的 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path, *, expected_kind: str) -> tuple[dict, ...]:
    """逐行读取 JSON object，并拒绝空行、重复身份和 record 漂移。"""
    if not path.is_file():
        raise BroadQaExternalDataError(f"external input 缺失: {path.name}")
    result = []
    identities = set()
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.endswith("\n") or not line.strip():
                    raise BroadQaExternalDataError(
                        f"external JSONL 换行非法: {line_number}")
                value = json.loads(line)
                if (not isinstance(value, dict)
                        or value.get("record_kind") != expected_kind
                        or not isinstance(value.get("item_id"), str)
                        or value["item_id"] in identities):
                    raise BroadQaExternalDataError(
                        f"external JSONL record 漂移: {line_number}")
                identities.add(value["item_id"])
                result.append(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError("external JSONL 非法") from error
    if not result:
        raise BroadQaExternalDataError("external JSONL 为空")
    return tuple(result)


def _write_jsonl(path: Path, records: Iterable[dict[str, object]]) -> int:
    """不可覆盖地写入规范预测，并返回实际记录数。"""
    if path.exists():
        raise BroadQaExternalDataError("external prediction 禁止覆盖")
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("xb") as handle:
        for record in records:
            handle.write(canonical_json_line(record))
            count += 1
    return count


def predict_external_evidence(
        questions_path: str | Path,
        *,
        predictions_path: str | Path,
        ) -> dict[str, object]:
    """只读取 questions，选择 exact context span；本函数不接收 labels。"""
    questions = _read_jsonl(
        Path(questions_path).resolve(),
        expected_kind="PH2_BROAD_QA_EXTERNAL_QUESTION_V1")
    output = Path(predictions_path).resolve()
    started_ns = time.perf_counter_ns()
    records = []
    for question in questions:
        expected = {
            "context", "context_sha256", "format_version", "item_id",
            "license_id", "question", "record_kind", "source_key",
            "source_partition", "source_question_id", "source_revision",
            "split", "title", "upstream_url",
        }
        if (set(question) != expected or question["format_version"] != 1
                or not isinstance(question["context"], str)
                or hashlib.sha256(question["context"].encode("utf-8")).hexdigest()
                != question["context_sha256"]):
            raise BroadQaExternalDataError("external question schema/SHA 漂移")
        evidence = select_broad_qa_evidence_sentence(
            question["question"], question["context"])
        start = question["context"].find(evidence)
        if start < 0:
            raise BroadQaExternalDataError("external evidence 不是 context span")
        records.append({
            "evidence_end": start + len(evidence),
            "evidence_sha256": hashlib.sha256(
                evidence.encode("utf-8")).hexdigest(),
            "evidence_start": start,
            "evidence_text": evidence,
            "format_version": 1,
            "item_id": question["item_id"],
            "record_kind": PREDICTION_KIND,
            "source_key": question["source_key"],
            "split": question["split"],
            "status": "ANSWER",
        })
    count = _write_jsonl(output, records)
    elapsed_ns = max(1, time.perf_counter_ns() - started_ns)
    return {
        "elapsed_ns": elapsed_ns,
        "prediction_count": count,
        "predictions_bytes": output.stat().st_size,
        "predictions_sha256": _sha256_file(output),
        "questions_sha256": _sha256_file(Path(questions_path).resolve()),
    }


def _ppm(numerator: int, denominator: int) -> int:
    """以整数百万分率表达聚合比例。"""
    return 0 if denominator == 0 else numerator * 1_000_000 // denominator


def score_external_evidence(
        questions_path: str | Path,
        predictions_path: str | Path,
        labels_path: str | Path,
        *,
        aggregate_path: str | Path,
        scope: str,
        minimum_evidence_hit_ppm: int = 700_000,
        ) -> dict[str, object]:
    """在预测完成后独立读取 labels，聚合证据命中与引用完整率。"""
    if (scope not in {"DEVELOPMENT", "FORMAL_HELD_OUT"}
            or type(minimum_evidence_hit_ppm) is not int
            or not 0 <= minimum_evidence_hit_ppm <= 1_000_000):
        raise BroadQaExternalDataError("external scoring scope/threshold 非法")
    question_file = Path(questions_path).resolve()
    prediction_file = Path(predictions_path).resolve()
    label_file = Path(labels_path).resolve()
    target = Path(aggregate_path).resolve()
    if target.exists():
        raise BroadQaExternalDataError("external aggregate 禁止覆盖")
    questions = _read_jsonl(
        question_file, expected_kind="PH2_BROAD_QA_EXTERNAL_QUESTION_V1")
    predictions = _read_jsonl(prediction_file, expected_kind=PREDICTION_KIND)
    labels = _read_jsonl(
        label_file, expected_kind="PH2_BROAD_QA_EXTERNAL_LABEL_V1")
    question_by_id = {item["item_id"]: item for item in questions}
    prediction_by_id = {item["item_id"]: item for item in predictions}
    label_by_id = {item["item_id"]: item for item in labels}
    inventory = set(question_by_id)
    if (set(prediction_by_id) != inventory or set(label_by_id) != inventory):
        raise BroadQaExternalDataError("external scoring inventory 不一致")
    evidence_hits = 0
    citation_valid = 0
    per_source: dict[str, Counter[str]] = {}
    for item_id in sorted(inventory):
        question = question_by_id[item_id]
        prediction = prediction_by_id[item_id]
        label = label_by_id[item_id]
        if (set(label) != {
                "format_version", "gold_answers", "item_id", "record_kind",
                "split"} or label["format_version"] != 1
                or not isinstance(label["gold_answers"], list)
                or not label["gold_answers"]
                or prediction.get("split") != question["split"]
                or label["split"] != question["split"]
                or prediction.get("source_key") != question["source_key"]):
            raise BroadQaExternalDataError("external scoring record 漂移")
        evidence = prediction.get("evidence_text")
        start = prediction.get("evidence_start")
        end = prediction.get("evidence_end")
        valid = (
            prediction.get("status") == "ANSWER"
            and isinstance(evidence, str) and evidence
            and type(start) is int and type(end) is int
            and 0 <= start < end <= len(question["context"])
            and question["context"][start:end] == evidence
            and hashlib.sha256(evidence.encode("utf-8")).hexdigest()
            == prediction.get("evidence_sha256"))
        hit = valid and any(
            normalize_external_text(answer) in normalize_external_text(evidence)
            for answer in label["gold_answers"]
            if isinstance(answer, str) and answer)
        citation_valid += int(valid)
        evidence_hits += int(hit)
        source = question["source_key"]
        counters = per_source.setdefault(source, Counter())
        counters["question_count"] += 1
        counters["citation_valid_count"] += int(valid)
        counters["evidence_hit_count"] += int(hit)
    total = len(inventory)
    citation_ppm = _ppm(citation_valid, total)
    hit_ppm = _ppm(evidence_hits, total)
    passed = (
        citation_ppm == 1_000_000
        and hit_ppm >= minimum_evidence_hit_ppm)
    aggregate = {
        "artifact_kind": AGGREGATE_KIND,
        "citation_valid_count": citation_valid,
        "citation_valid_ppm": citation_ppm,
        "evidence_hit_count": evidence_hits,
        "evidence_hit_ppm": hit_ppm,
        "format_version": 1,
        "labels_sha256": _sha256_file(label_file),
        "minimum_evidence_hit_ppm": minimum_evidence_hit_ppm,
        "per_source": {
            key: {
                **dict(sorted(values.items())),
                "evidence_hit_ppm": _ppm(
                    values["evidence_hit_count"], values["question_count"]),
            }
            for key, values in sorted(per_source.items())
        },
        "predictions_sha256": _sha256_file(prediction_file),
        "question_count": total,
        "questions_sha256": _sha256_file(question_file),
        "scope": scope,
        "status": "PASS" if passed else "FAIL",
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(canonical_json_line(aggregate))
    return {**aggregate, "aggregate_sha256": _sha256_file(target)}


__all__ = [
    "AGGREGATE_KIND",
    "PREDICTION_KIND",
    "predict_external_evidence",
    "score_external_evidence",
]
