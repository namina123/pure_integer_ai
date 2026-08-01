"""W-02 LC-16 九载体补充资格的独立聚合 evaluator。"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Iterable

from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from pure_integer_ai.experiments.ph2_d03_lc16_overlay_contract import (
    D03Lc16SuccessorOverlay,
)
from pure_integer_ai.experiments.ph2_w02_lc16_supplemental_contract import (
    ABLATION_ORDER,
    BEARING_DIMENSIONS,
    CASE_COUNT,
    DIRECTION_EVALUATION_COUNT,
    DIRECTIONS,
    EVALUATION_ORDER,
    GENERATION_HARD_CONJUNCT,
    HOST_DIGEST_KEYS,
    IN_SCOPE_CARRIER_KEYS,
    MAX_LOGIC_OPERATIONS,
    MAX_PAYLOAD_BYTES,
    MAX_PAYLOAD_READS,
    OVERLAY_SHA256,
    SupplementalAblationResult,
    SupplementalCarrierSummary,
    SupplementalDimensionResult,
    SupplementalDirectionResult,
    W02Lc16SupplementalError,
    W02Lc16SupplementalReport,
    W02_PARENT_RECEIPT_SHA256,
)


def _digest(value: object) -> str:
    """对公开状态结构计算 canonical SHA-256。"""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _status(values: Iterable[str]) -> str:
    """按 FAIL、NE、PASS 的固定优先级折叠状态。"""
    checked = tuple(values)
    if not checked or "FAIL" in checked:
        return "FAIL" if checked else "NE"
    if "NE" in checked:
        return "NE"
    return "PASS"


def _dimension(
        key: str, values: tuple[SupplementalDimensionResult, ...],
        *, where: str,
        ) -> SupplementalDimensionResult:
    """把一组单 case 维度结果折叠为公开计数。"""
    if not values:
        raise W02Lc16SupplementalError(f"{where} 没有结果")
    passed = sum(item.status == "PASS" for item in values)
    failed = sum(item.status == "FAIL" for item in values)
    ne = sum(item.status == "NE" for item in values)
    return SupplementalDimensionResult(
        key,
        _status(item.status for item in values),
        passed,
        len(values),
        failed,
        ne,
        _digest({
            "dimension_key": key,
            "evidence": [item.evidence_sha256 for item in values],
            "where": where,
        }),
    )


def _ablation(
        key: str,
        values: tuple[SupplementalDirectionResult, ...],
        ) -> SupplementalAblationResult:
    """按固定消融顺序折叠四维和 generation 状态。"""
    index = ABLATION_ORDER.index(key)
    target = BEARING_DIMENSIONS[index]
    dimension_statuses = []
    for dimension in BEARING_DIMENSIONS:
        statuses = tuple(
            dict(item.dimension_statuses)[dimension]
            for result in values
            for item in result.ablations
            if item.ablation_key == key
        )
        dimension_statuses.append((dimension, _status(statuses)))
    generation_statuses = tuple(
        item.generation_status
        for result in values
        for item in result.ablations
        if item.ablation_key == key
    )
    return SupplementalAblationResult(
        key,
        target,
        tuple(dimension_statuses),
        _status(generation_statuses),
        _digest({
            "ablation_key": key,
            "direction_evidence": [
                result.evidence_sha256 for result in values],
        }),
    )


def _carrier_summary(
        carrier: str,
        values: tuple[SupplementalDirectionResult, ...],
        ) -> SupplementalCarrierSummary:
    """把单 carrier 的 21 个方向结果折叠为公开摘要。"""
    dimensions = tuple(
        _dimension(
            key,
            tuple(
                dimension
                for result in values
                for dimension in result.dimensions
                if dimension.dimension_key == key
            ),
            where=f"carrier {carrier}",
        )
        for key in BEARING_DIMENSIONS
    )
    generation = _dimension(
        GENERATION_HARD_CONJUNCT,
        tuple(result.generation for result in values),
        where=f"carrier {carrier} generation",
    )
    return SupplementalCarrierSummary(
        carrier,
        len(values),
        dimensions,
        generation,
        _digest({
            "carrier_key": carrier,
            "evidence": [item.evidence_sha256 for item in values],
        }),
    )


def _expected_overlay_cases(
        overlay: D03Lc16SuccessorOverlay,
        ) -> dict[tuple[str, tuple[int, ...], str], tuple[int, ...]]:
    """从 overlay 生成唯一 carrier/case/direction owner 索引。"""
    result: dict[tuple[str, tuple[int, ...], str], tuple[int, ...]] = {}
    for course in overlay.carrier_courses:
        for case in course.cases:
            case_key = case.case_key.stable_key()
            owner_key = case.owner_key.stable_key()
            for direction in DIRECTIONS:
                key = (course.carrier_key, case_key, direction)
                if key in result:
                    raise W02Lc16SupplementalError("overlay 结果键重复")
                result[key] = owner_key
    if len(result) != DIRECTION_EVALUATION_COUNT:
        raise W02Lc16SupplementalError("overlay 方向覆盖不是 189")
    return result


def _verify_ablation_orthogonality(
        baseline: tuple[SupplementalDimensionResult, ...],
        ablations: tuple[SupplementalAblationResult, ...],
        ) -> None:
    """要求每项消融只允许击穿目标维度且不得改善其它维度。"""
    baseline_status = {
        item.dimension_key: item.status for item in baseline}
    for ablation in ablations:
        target_status = dict(ablation.dimension_statuses)[
            ablation.targeted_dimension]
        if baseline_status[ablation.targeted_dimension] == "PASS" \
                and target_status == "PASS":
            raise W02Lc16SupplementalError("消融未击穿目标维度")
        for dimension, status in ablation.dimension_statuses:
            if dimension != ablation.targeted_dimension \
                    and baseline_status[dimension] != status:
                raise W02Lc16SupplementalError("消融改变非目标维度")


def _verify_direction_ablation(result: SupplementalDirectionResult) -> None:
    """在聚合前逐方向检查消融，避免其它 carrier 掩盖局部错误。"""
    baseline = {item.dimension_key: item.status for item in result.dimensions}
    baseline[GENERATION_HARD_CONJUNCT] = result.generation.status
    for ablation in result.ablations:
        statuses = dict(ablation.dimension_statuses)
        for dimension in BEARING_DIMENSIONS:
            if dimension != ablation.targeted_dimension \
                    and statuses[dimension] != baseline[dimension]:
                raise W02Lc16SupplementalError("消融改变非目标维度")
        if (baseline[ablation.targeted_dimension] == "PASS"
                and statuses[ablation.targeted_dimension] == "PASS"):
            raise W02Lc16SupplementalError("消融未击穿目标维度")
        generation_target = ablation.targeted_dimension == BEARING_DIMENSIONS[2]
        if not generation_target and ablation.generation_status != baseline[GENERATION_HARD_CONJUNCT]:
            raise W02Lc16SupplementalError("消融改变非目标 generation")
        if (generation_target and baseline[GENERATION_HARD_CONJUNCT] == "PASS"
                and ablation.generation_status == "PASS"):
            raise W02Lc16SupplementalError("形态消融未击穿 generation")


def aggregate_w02_lc16_supplemental(
        overlay: D03Lc16SuccessorOverlay,
        results: tuple[SupplementalDirectionResult, ...],
        *,
        host_digests_before: dict[str, str],
        host_digests_after: dict[str, str],
        private_path_reads: int,
        private_payload_bytes: int,
        private_payload_reads: int,
        evaluator_label_reads: int,
        evaluator_label_writes: int = 0,
        host_write_count: int = 0,
        independent_evaluator_module_separate: int = 1,
        consumer_result_builder_reused: int = 0,
        runtime_observed: int = 1,
        ) -> W02Lc16SupplementalReport:
    """独立折叠 189 个方向结果并生成新的 supplemental receipt 对象。"""
    if not isinstance(overlay, D03Lc16SuccessorOverlay):
        raise TypeError("overlay 类型非法")
    if overlay.sha256() != OVERLAY_SHA256:
        raise W02Lc16SupplementalError("overlay SHA-256 漂移")
    expected = _expected_overlay_cases(overlay)
    if (not isinstance(results, tuple)
            or len(results) != DIRECTION_EVALUATION_COUNT):
        raise W02Lc16SupplementalError("supplemental 结果必须精确 189 条")
    seen: set[tuple[str, tuple[int, ...], str]] = set()
    for result in results:
        if not isinstance(result, SupplementalDirectionResult):
            raise TypeError("supplemental direction result 类型非法")
        key = (result.carrier_key, result.case_key, result.direction)
        if key in seen or key not in expected or result.owner_key != expected[key]:
            raise W02Lc16SupplementalError("supplemental carrier/case/owner/direction 漂移")
        _verify_direction_ablation(result)
        seen.add(key)
    if seen != set(expected):
        raise W02Lc16SupplementalError("supplemental 189 方向覆盖不完整")
    if any(result.independent_reveal_status == "BLOCKED" for result in results):
        independent_status = "BLOCKED"
    elif any(result.independent_reveal_status == "NE" for result in results):
        independent_status = "NE"
    elif any(result.independent_reveal_status == "FAIL" for result in results):
        independent_status = "FAIL"
    else:
        independent_status = "PASS"
    dimensions = tuple(
        _dimension(
            key,
            tuple(
                dimension
                for result in results
                for dimension in result.dimensions
                if dimension.dimension_key == key
            ),
            where="global supplemental",
        )
        for key in BEARING_DIMENSIONS
    )
    generation = _dimension(
        GENERATION_HARD_CONJUNCT,
        tuple(result.generation for result in results),
        where="global supplemental generation",
    )
    ablations = tuple(_ablation(key, results) for key in ABLATION_ORDER)
    _verify_ablation_orthogonality(dimensions, ablations)
    by_carrier: dict[str, list[SupplementalDirectionResult]] = defaultdict(list)
    for result in results:
        by_carrier[result.carrier_key].append(result)
    summaries = tuple(
        _carrier_summary(carrier, tuple(by_carrier[carrier]))
        for carrier in IN_SCOPE_CARRIER_KEYS
    )
    statuses = [item.status for item in dimensions] + [generation.status]
    statuses.append(independent_status)
    if runtime_observed == 0:
        report_status = "BLOCKED"
    elif "FAIL" in statuses:
        report_status = "FAIL"
    elif "NE" in statuses:
        report_status = "NE"
    elif "BLOCKED" in statuses:
        report_status = "BLOCKED"
    else:
        report_status = "PASS"
    evidence = _digest({
        "ablations": [item.evidence_sha256 for item in ablations],
        "dimensions": [item.evidence_sha256 for item in dimensions],
        "generation": generation.evidence_sha256,
        "independent_status": independent_status,
        "summaries": [item.evidence_sha256 for item in summaries],
    })
    if private_payload_bytes > MAX_PAYLOAD_BYTES \
            or private_payload_reads > MAX_PAYLOAD_READS:
        raise W02Lc16SupplementalError("supplemental payload budget 超限")
    if private_path_reads < 0 or private_payload_bytes < 0 \
            or private_payload_reads < 0 or evaluator_label_reads < 0:
        raise W02Lc16SupplementalError("supplemental read count 非法")
    return W02Lc16SupplementalReport(
        report_status,
        OVERLAY_SHA256,
        W02_PARENT_RECEIPT_SHA256,
        CASE_COUNT,
        DIRECTION_EVALUATION_COUNT,
        dimensions,
        generation,
        ablations,
        summaries,
        host_digests_before,
        host_digests_after,
        private_path_reads,
        private_payload_bytes,
        private_payload_reads,
        evaluator_label_reads,
        evaluator_label_writes,
        host_write_count,
        independent_evaluator_module_separate,
        consumer_result_builder_reused,
        runtime_observed,
        evidence,
    )


__all__ = [
    "aggregate_w02_lc16_supplemental",
]
