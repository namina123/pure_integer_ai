"""执行 recovery-v9 GIMP fixed-denominator六维纯aggregate evaluator。"""
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
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_evaluator import (
    _coverage_dimension,
    _identity_dimension,
    _route_dimension,
    _runtime_dimension,
    _structure_dimension,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v9_evaluation_commitment import (
    NORMALIZATION_RECOVERY_V9_DIMENSIONS,
    NORMALIZATION_RECOVERY_V9_DIMENSION_ORDER,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v9_label_materialization import (
    NORMALIZATION_RECOVERY_V9_EVALUATION_RECORD_KIND,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes


NORMALIZATION_RECOVERY_V9_EVALUATION_REPORT_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V9_EVALUATION_REPORT_V1")


def _sha256(value: object) -> str:
    """返回report、roster或runtime aggregate identity。"""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _validate_records(
        records: tuple[dict[str, object], ...], *,
        commitment: dict[str, object], materialization: dict[str, object],
        ) -> None:
    """核验GIMP全分母、来源SHA、结构与十项aggregate。"""
    denominator = commitment.get("denominator")
    buckets = denominator.get("aggregate_buckets") if isinstance(
        denominator, dict) else None
    if (not isinstance(records, tuple) or not records
            or not isinstance(denominator, dict)
            or len(records) != denominator.get("record_count")
            or len(records) != materialization.get("label_materialization_count")
            or materialization.get("evaluation_record_roster_sha256")
            != _sha256(records)
            or len({item.get("evaluation_id") for item in records})
            != len(records)
            or not isinstance(buckets, dict)):
        raise BroadQaExternalDataError("v9 evaluator denominator/roster漂移")
    for record in records:
        binary = (
            "contains_han_both", "equal_length", "evaluation_eligible",
            "identity_preservation", "single_han_difference",
            "structure_equal", "variable_length", "within_scalar_limit",
        )
        if (not isinstance(record, dict)
                or record.get("record_kind")
                != NORMALIZATION_RECOVERY_V9_EVALUATION_RECORD_KIND
                or not isinstance(record.get("input_text"), str)
                or not record["input_text"]
                or not isinstance(record.get("expected_output"), str)
                or not record["expected_output"]
                or not isinstance(record.get("official_source_text"), str)
                or not record["official_source_text"]
                or not isinstance(record.get("source_identity"), dict)
                or not isinstance(record.get("structure_tokens"), list)
                or any(not isinstance(item, str) or not item
                       for item in record["structure_tokens"])
                or tuple(record["structure_tokens"])
                != localization_structure_tokens(str(record["input_text"]))
                or any(type(record.get(name)) is not int
                       or record[name] not in {0, 1} for name in binary)
                or record["equal_length"] + record["variable_length"] != 1
                or record["evaluation_eligible"] > record["contains_han_both"]
                or record["evaluation_eligible"] > record["structure_equal"]
                or record["source_pack_manifest_sha256"]
                != materialization.get("gimp_source_manifest_sha256")):
            raise BroadQaExternalDataError("v9 evaluator record schema漂移")
    outputs_by_input: dict[str, set[str]] = {}
    for item in records:
        outputs_by_input.setdefault(str(item["input_text"]), set()).add(
            str(item["expected_output"]))
    checks = {
        "contains_han_both_count": sum(
            item["contains_han_both"] for item in records),
        "equal_length_count": sum(item["equal_length"] for item in records),
        "evaluation_eligible_count": sum(
            item["evaluation_eligible"] for item in records),
        "identity_count": sum(
            item["identity_preservation"] for item in records),
        "input_conflict_count": sum(
            len(outputs) > 1 for outputs in outputs_by_input.values()),
        "nonidentity_count": sum(
            1 - item["identity_preservation"] for item in records),
        "single_han_difference_count": sum(
            item["single_han_difference"] for item in records),
        "structure_equal_count": sum(
            item["structure_equal"] for item in records),
        "structure_unequal_count": sum(
            1 - item["structure_equal"] for item in records),
        "variable_length_count": sum(
            item["variable_length"] for item in records),
    }
    if checks != buckets:
        raise BroadQaExternalDataError("v9 evaluator aggregate denominator漂移")


def _queries(
        records: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, object], ...]:
    """投影candidate所需的标签盲query字段。"""
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
    """执行indexed两次与独立reference，并封装异常计数。"""
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


def _judgement_counts(
        records: tuple[dict[str, object], ...],
        results: tuple[dict[str, object], ...],
        ) -> dict[str, int]:
    """形成不含个体输出的EXACT/UNKNOWN/WRONG aggregate。"""
    counts = Counter()
    for record, result in zip(records, results):
        if (result.get("behavior") == "EXACT"
                and result.get("output_text") == record["expected_output"]):
            counts["EXACT"] += 1
        elif result.get("behavior") == "UNKNOWN" and not result.get(
                "output_text"):
            counts["UNKNOWN"] += 1
        else:
            counts["WRONG"] += 1
    return {name: counts[name] for name in ("EXACT", "UNKNOWN", "WRONG")}


def evaluate_normalization_recovery_v9_candidate(
        *, commitment: dict[str, object],
        candidate_manifest: dict[str, object], candidate: dict[str, object],
        materialization: dict[str, object],
        evaluation_records: tuple[dict[str, object], ...],
        family_freeze_manifest_sha256: str,
        ) -> dict[str, object]:
    """执行六维formal evaluator，只返回aggregate与结果identity。"""
    if (commitment.get("dimensions") != NORMALIZATION_RECOVERY_V9_DIMENSIONS
            or commitment.get("dimension_order")
            != list(NORMALIZATION_RECOVERY_V9_DIMENSION_ORDER)
            or candidate_manifest.get("candidate_program_sha256")
            != candidate.get("candidate_program_sha256")
            or candidate.get("evaluation_commitment_manifest_sha256")
            != commitment.get("manifest_sha256")
            or materialization.get("evaluation_commitment_manifest_sha256")
            != commitment.get("manifest_sha256")):
        raise BroadQaExternalDataError("v9 evaluator lineage漂移")
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
            NORMALIZATION_RECOVERY_V9_DIMENSION_ORDER):
        raise BroadQaExternalDataError("v9 evaluator dimension顺序漂移")
    outcomes = tuple(str(item["outcome"]) for item in dimensions)
    overall = "FAIL" if "FAIL" in outcomes else (
        "NE" if "NE" in outcomes else "PASS")
    runtime_aggregate = tuple({
        "behavior": item.get("behavior"),
        "output_text_sha256": hashlib.sha256(
            str(item.get("output_text", "")).encode("utf-8")).hexdigest(),
        "result_sha256": item.get("result_sha256"),
        "route_kind": item.get("route_kind"),
    } for item in first)
    report = {
        "artifact_kind": NORMALIZATION_RECOVERY_V9_EVALUATION_REPORT_KIND,
        "candidate_manifest_sha256": candidate_manifest["manifest_sha256"],
        "candidate_program_sha256": candidate["candidate_program_sha256"],
        "dimensions": list(dimensions),
        "evaluation_commitment_manifest_sha256": commitment["manifest_sha256"],
        "evaluation_run_count": 1,
        "family_freeze_manifest_sha256": family_freeze_manifest_sha256,
        "format_version": 1,
        "gimp_source_payload_read_count": materialization[
            "gimp_source_payload_read_count"],
        "individual_label_publication_count": 0,
        "judgement_counts": _judgement_counts(evaluation_records, first),
        "label_materialization_count": materialization[
            "label_materialization_count"],
        "mastery_claimed": 0,
        "overall_outcome": overall,
        "production_enabled": 0,
        "runtime_aggregate_sha256": _sha256(runtime_aggregate),
        "teacher_api_llm_call_count": 0,
    }
    return {**report, "evaluation_report_sha256": _sha256(report)}


__all__ = [
    "NORMALIZATION_RECOVERY_V9_EVALUATION_REPORT_KIND",
    "evaluate_normalization_recovery_v9_candidate",
]
