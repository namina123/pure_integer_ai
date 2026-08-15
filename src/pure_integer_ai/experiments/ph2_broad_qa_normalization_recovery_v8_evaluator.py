"""执行 recovery-v8 VLC fixed-denominator 六维纯 aggregate evaluator。"""
from __future__ import annotations

from collections import Counter
import hashlib

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_localization_structure import (
    localization_structure_tokens,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_candidate import (
    execute_normalization_recovery_v8_candidate_batch,
    reference_normalization_recovery_v8_candidate_batch,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_evaluation_commitment import (
    NORMALIZATION_RECOVERY_V8_DIMENSIONS,
    NORMALIZATION_RECOVERY_V8_DIMENSION_ORDER,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_label_materialization import (
    NORMALIZATION_RECOVERY_V8_EVALUATION_RECORD_KIND,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes


NORMALIZATION_RECOVERY_V8_EVALUATION_REPORT_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V8_EVALUATION_REPORT_V1")


def _sha256(value: object) -> str:
    """返回 report、dimension 或 runtime aggregate identity。"""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _dimension(
        key: str, metrics: dict[str, int], outcome: str,
        ) -> dict[str, object]:
    """构造固定顺序、只含整数指标的一个正式维度。"""
    if (key not in NORMALIZATION_RECOVERY_V8_DIMENSION_ORDER
            or outcome not in {"FAIL", "NE", "PASS"}):
        raise BroadQaExternalDataError("v8 evaluator dimension 非法")
    payload = {
        "dimension_key": key,
        "metrics": {name: metrics[name] for name in sorted(metrics)},
        "outcome": outcome,
    }
    return {**payload, "dimension_sha256": _sha256(payload)}


def _validate_records(
        records: tuple[dict[str, object], ...], *,
        commitment: dict[str, object], materialization: dict[str, object],
        ) -> None:
    """核验全分母、来源 SHA、结构和所有冻结 aggregate。"""
    denominator = commitment.get("denominator")
    buckets = denominator.get("aggregate_buckets") if isinstance(
        denominator, dict) else None
    if (not isinstance(records, tuple) or not records
            or len(records) != denominator.get("record_count")
            or len(records) != materialization.get("label_materialization_count")
            or materialization.get("evaluation_record_roster_sha256")
            != _sha256(records)
            or len({item.get("evaluation_id") for item in records}) != len(records)
            or not isinstance(buckets, dict)):
        raise BroadQaExternalDataError("v8 evaluator denominator/roster 漂移")
    for record in records:
        if (not isinstance(record, dict)
                or record.get("record_kind")
                != NORMALIZATION_RECOVERY_V8_EVALUATION_RECORD_KIND
                or not isinstance(record.get("input_text"), str)
                or not record["input_text"]
                or not isinstance(record.get("expected_output"), str)
                or not record["expected_output"]
                or not isinstance(record.get("official_source_text"), str)
                or not record["official_source_text"]
                or not isinstance(record.get("structure_tokens"), list)
                or any(not isinstance(item, str) or not item
                       for item in record["structure_tokens"])
                or tuple(record["structure_tokens"])
                != localization_structure_tokens(str(record["input_text"]))
                or any(type(record.get(name)) is not int
                       or record[name] not in {0, 1}
                       for name in ("equal_length", "identity_preservation",
                                    "single_han_difference", "structure_equal",
                                    "variable_length"))
                or record["equal_length"] + record["variable_length"] != 1
                or record["source_pack_manifest_sha256"]
                != materialization.get("vlc_source_manifest_sha256")):
            raise BroadQaExternalDataError("v8 evaluator record schema 漂移")
    checks = {
        "equal_length_count": sum(item["equal_length"] for item in records),
        "identity_count": sum(item["identity_preservation"] for item in records),
        "nonidentity_count": sum(1 - item["identity_preservation"]
                                 for item in records),
        "single_han_difference_count": sum(
            item["single_han_difference"] for item in records),
        "structure_equal_count": sum(item["structure_equal"] for item in records),
        "variable_length_count": sum(item["variable_length"] for item in records),
    }
    if any(checks[name] != buckets.get(name) for name in checks):
        raise BroadQaExternalDataError("v8 evaluator aggregate denominator 漂移")


def _queries(
        records: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, object], ...]:
    """投影 candidate 所需的标签盲 query 字段。"""
    return tuple({
        "input_text": item["input_text"],
        "official_source_text": item["official_source_text"],
        "structure_tokens": item["structure_tokens"],
    } for item in records)


def _run_candidate(
        candidate: dict[str, object], records: tuple[dict[str, object], ...],
        ) -> tuple[tuple[dict[str, object], ...],
                   tuple[dict[str, object], ...],
                   tuple[dict[str, object], ...], int]:
    """执行 indexed 两次与独立 reference，并封装异常计数。"""
    queries = _queries(records)
    try:
        first = execute_normalization_recovery_v8_candidate_batch(
            candidate, queries)
        second = execute_normalization_recovery_v8_candidate_batch(
            candidate, queries)
        reference = reference_normalization_recovery_v8_candidate_batch(
            candidate, queries)
    except (BroadQaExternalDataError, KeyError, TypeError, ValueError):
        return (), (), (), 1
    return first, second, reference, 0


def _judgement(record: dict[str, object], result: dict[str, object]) -> str:
    """把候选提交判为 EXACT、UNKNOWN 或 WRONG。"""
    output = result.get("output_text")
    if output == record["expected_output"] and result.get("behavior") == "EXACT":
        return "EXACT"
    if output == "" and result.get("behavior") == "UNKNOWN":
        return "UNKNOWN"
    return "WRONG"


def _route_dimension(
        key: str, route: str, records: tuple[dict[str, object], ...],
        results: tuple[dict[str, object], ...],
        ) -> dict[str, object]:
    """判定正字或 source-conditioned changed transfer。"""
    pairs = tuple((record, result) for record, result in zip(records, results)
                  if result.get("route_kind") == route
                  and result.get("behavior") == "EXACT")
    judgements = Counter(_judgement(record, result) for record, result in pairs)
    metrics = {
        "committed_count": len(pairs),
        "exact_count": judgements["EXACT"],
        "wrong_count": judgements["WRONG"],
    }
    if key == "ORTHOGRAPHIC_ATOM_TRANSFER":
        metrics["single_han_inventory_count"] = sum(
            item["single_han_difference"] for item in records)
    else:
        metrics["unconditioned_execution_count"] = sum(
            result.get("route_kind") == route
            and not record["official_source_text"]
            for record, result in zip(records, results))
    if len(results) != len(records):
        outcome = "NE"
    elif metrics["wrong_count"] != 0 or metrics.get(
            "unconditioned_execution_count", 0) != 0:
        outcome = "FAIL"
    elif metrics["exact_count"] >= 1:
        outcome = "PASS"
    else:
        outcome = "NE"
    return _dimension(key, metrics, outcome)


def _identity_dimension(
        records: tuple[dict[str, object], ...],
        results: tuple[dict[str, object], ...],
        ) -> dict[str, object]:
    """判定完整 identity 分母不被误改且 veto 至少命中一次。"""
    pairs = tuple((record, result) for record, result in zip(records, results)
                  if record["identity_preservation"] == 1)
    false_change = sum(
        result.get("output_text") not in {"", record["input_text"]}
        for record, result in pairs)
    veto_exact = sum(
        result.get("route_kind") == "IDENTITY_VETO"
        and _judgement(record, result) == "EXACT"
        for record, result in pairs)
    metrics = {
        "false_change_count": false_change,
        "identity_inventory_count": len(pairs),
        "identity_veto_exact_count": veto_exact,
    }
    if len(results) != len(records) or not pairs:
        outcome = "NE"
    elif false_change != 0:
        outcome = "FAIL"
    elif veto_exact >= 1:
        outcome = "PASS"
    else:
        outcome = "NE"
    return _dimension("IDENTITY_PRESERVATION", metrics, outcome)


def _structure_dimension(
        records: tuple[dict[str, object], ...],
        results: tuple[dict[str, object], ...], *, exception_count: int,
        ) -> dict[str, object]:
    """判定结构保持、无部分提交与结构承载 exact 生成。"""
    committed_structure_exact = sum(
        bool(record["structure_tokens"])
        and _judgement(record, result) == "EXACT"
        for record, result in zip(records, results))
    metrics = {
        "committed_structure_bearing_exact_count": committed_structure_exact,
        "exception_count": exception_count,
        "partial_commit_count": sum(
            int(result.get("partial_commit_count", 1)) for result in results),
        "structure_mismatch_count": sum(
            int(result.get("structure_mismatch_count", 1)) for result in results),
    }
    if (metrics["partial_commit_count"] != 0
            or metrics["structure_mismatch_count"] != 0):
        outcome = "FAIL"
    elif (exception_count != 0 or len(results) != len(records)
          or committed_structure_exact == 0):
        outcome = "NE"
    else:
        outcome = "PASS"
    return _dimension("STRUCTURE_AND_GENERATION_INTEGRITY", metrics, outcome)


def _runtime_dimension(
        first: tuple[dict[str, object], ...],
        second: tuple[dict[str, object], ...],
        reference: tuple[dict[str, object], ...], *,
        inventory_count: int, exception_count: int,
        ) -> dict[str, object]:
    """核对双执行、indexed/reference、禁用态和完整分母。"""
    triples = tuple(zip(first, second, reference))
    metrics = {
        "exception_count": exception_count,
        "executed_twice_count": len(triples),
        "indexed_reference_mismatch_count": sum(
            left != right or left != linear for left, right, linear in triples),
        "inventory_count": inventory_count,
        "production_enabled_count": sum(
            item.get("production_enabled") != 0
            for triple in triples for item in triple),
    }
    if exception_count != 0 or not triples:
        outcome = "NE"
    elif (len(triples) != inventory_count
          or metrics["indexed_reference_mismatch_count"] != 0
          or metrics["production_enabled_count"] != 0):
        outcome = "FAIL"
    else:
        outcome = "PASS"
    return _dimension("RUNTIME_INDEXED_REFERENCE_EQUIVALENCE", metrics, outcome)


def _coverage_dimension(
        records: tuple[dict[str, object], ...],
        results: tuple[dict[str, object], ...],
        ) -> dict[str, object]:
    """判定全分母 EXACT/UNKNOWN/WRONG 分账与非 identity 覆盖。"""
    judgements = tuple(_judgement(record, result)
                       for record, result in zip(records, results))
    counts = Counter(judgements)
    changed_exact = sum(
        record["identity_preservation"] == 0 and judgement == "EXACT"
        for record, judgement in zip(records, judgements))
    metrics = {
        "changed_exact_count": changed_exact,
        "exact_count": counts["EXACT"],
        "inventory_count": len(records),
        "judged_count": len(judgements),
        "unknown_count": counts["UNKNOWN"],
        "wrong_count": counts["WRONG"],
    }
    if len(results) != len(records) or len(judgements) != len(records):
        outcome = "NE"
    elif counts["WRONG"] != 0:
        outcome = "FAIL"
    elif changed_exact >= 2:
        outcome = "PASS"
    else:
        outcome = "NE"
    return _dimension("END_TO_END_COVERAGE", metrics, outcome)


def evaluate_normalization_recovery_v8_candidate(
        *, commitment: dict[str, object],
        candidate_manifest: dict[str, object], candidate: dict[str, object],
        materialization: dict[str, object],
        evaluation_records: tuple[dict[str, object], ...],
        family_freeze_manifest_sha256: str,
        ) -> dict[str, object]:
    """执行六维 formal evaluator，只返回 aggregate，不发布个体 labels。"""
    if (commitment.get("dimensions") != NORMALIZATION_RECOVERY_V8_DIMENSIONS
            or candidate_manifest.get("candidate_program_sha256")
            != candidate.get("candidate_program_sha256")
            or candidate.get("evaluation_commitment_manifest_sha256")
            != commitment.get("manifest_sha256")
            or materialization.get("evaluation_commitment_manifest_sha256")
            != commitment.get("manifest_sha256")):
        raise BroadQaExternalDataError("v8 evaluator lineage 漂移")
    _validate_records(
        evaluation_records, commitment=commitment,
        materialization=materialization)
    first, second, reference, exceptions = _run_candidate(
        candidate, evaluation_records)
    dimensions = (
        _route_dimension(
            "ORTHOGRAPHIC_ATOM_TRANSFER", "ORTHOGRAPHIC_ATOM",
            evaluation_records, first),
        _route_dimension(
            "SOURCE_CONDITIONED_LEXICAL_TRANSFER",
            "SOURCE_CONDITIONED_LEXICAL_ATOM", evaluation_records, first),
        _identity_dimension(evaluation_records, first),
        _structure_dimension(
            evaluation_records, first, exception_count=exceptions),
        _runtime_dimension(
            first, second, reference, inventory_count=len(evaluation_records),
            exception_count=exceptions),
        _coverage_dimension(evaluation_records, first),
    )
    if tuple(item["dimension_key"] for item in dimensions) != (
            NORMALIZATION_RECOVERY_V8_DIMENSION_ORDER):
        raise BroadQaExternalDataError("v8 evaluator dimension 顺序漂移")
    outcomes = tuple(str(item["outcome"]) for item in dimensions)
    overall = "FAIL" if "FAIL" in outcomes else "NE" if "NE" in outcomes else "PASS"
    runtime_aggregate = tuple({
        "behavior": item.get("behavior"),
        "output_text_sha256": hashlib.sha256(
            str(item.get("output_text", "")).encode("utf-8")).hexdigest(),
        "result_sha256": item.get("result_sha256"),
        "route_kind": item.get("route_kind"),
    } for item in first)
    report = {
        "artifact_kind": NORMALIZATION_RECOVERY_V8_EVALUATION_REPORT_KIND,
        "candidate_manifest_sha256": candidate_manifest["manifest_sha256"],
        "candidate_program_sha256": candidate["candidate_program_sha256"],
        "dimensions": list(dimensions),
        "evaluation_commitment_manifest_sha256": commitment["manifest_sha256"],
        "evaluation_run_count": 1,
        "family_freeze_manifest_sha256": family_freeze_manifest_sha256,
        "format_version": 1,
        "individual_label_publication_count": 0,
        "label_materialization_count": materialization[
            "label_materialization_count"],
        "mastery_claimed": 0,
        "overall_outcome": overall,
        "production_enabled": 0,
        "runtime_aggregate_sha256": _sha256(runtime_aggregate),
        "teacher_api_llm_call_count": 0,
        "vlc_source_payload_read_count": materialization[
            "vlc_source_payload_read_count"],
    }
    return {**report, "evaluation_report_sha256": _sha256(report)}


__all__ = [
    "NORMALIZATION_RECOVERY_V8_EVALUATION_REPORT_KIND",
    "evaluate_normalization_recovery_v8_candidate",
]
