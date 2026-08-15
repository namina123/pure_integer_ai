"""派生 recovery-v7 variable structure plans 与 TRAIN-only LOSO。

plan 只负责把有序 structure token 视为必须原样复制的锚点，并把其间文本变为
segment obligations。当前没有 segment generator，因此 represented case 仍返回
UNKNOWN，绝不部分提交或把 representation 当作 exact capability。
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
import hashlib

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_localization_structure import (
    localization_structure_layout,
    localization_structure_layout_for_tokens,
    localization_structure_token_category,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_training_records import (
    V5_SOURCE_FAMILIES,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
)


VARIABLE_STRUCTURE_PLAN_KIND = (
    "NORMALIZATION_RECOVERY_V7_VARIABLE_STRUCTURE_PLAN_V1")
VARIABLE_STRUCTURE_LOSO_KIND = (
    "NORMALIZATION_RECOVERY_V7_VARIABLE_STRUCTURE_LOSO_V1")
VARIABLE_STRUCTURE_TARGET_SCOPE = (
    "CROSS_PRODUCT_ZH_CN_VARIABLE_STRUCTURE_OBLIGATION_V1")


def _sha256(payload: bytes) -> str:
    """返回规范 identity 或 TRAIN segment surface 的 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _text_sha256(value: str) -> str:
    """返回 UTF-8 文本 SHA。"""
    if not isinstance(value, str):
        raise BroadQaExternalDataError("v7 variable structure text 非字符串")
    return _sha256(value.encode("utf-8"))


def _record_id(identity: dict[str, object]) -> str:
    """从规范 identity 形成稳定记录 id。"""
    return _sha256(canonical_json_bytes(identity))


def _segment_records(
        input_segments: tuple[str, ...],
        output_segments: tuple[str, ...],
        ) -> tuple[dict[str, object], ...]:
    """把 structure 锚点之间的 segment 变为 hash 与 obligation。"""
    if len(input_segments) != len(output_segments):
        raise BroadQaExternalDataError(
            "v7 variable structure segment count 漂移")
    values = []
    for ordinal, (input_text, output_text) in enumerate(zip(
            input_segments, output_segments)):
        values.append({
            "identity_copy": int(input_text == output_text),
            "input_length": len(input_text),
            "input_sha256": _text_sha256(input_text),
            "ordinal": ordinal,
            "output_length": len(output_text),
            "output_sha256": _text_sha256(output_text),
            "proposal_required": int(input_text != output_text),
        })
    return tuple(values)


def derive_variable_structure_plans(
        observations: Iterable[dict[str, object]],
        ) -> tuple[tuple[dict[str, object], ...], dict[str, object]]:
    """从 structured variable TRAIN observations 派生不可执行 plans。"""
    values = []
    counters = Counter()
    seen = set()
    for item in observations:
        if not isinstance(item, dict):
            raise BroadQaExternalDataError(
                "v7 variable structure observation 非对象")
        identity_flag = item.get("identity_preservation")
        equal_length = item.get("equal_length")
        tokens = item.get("structure_tokens")
        if (type(identity_flag) is not int or identity_flag not in (0, 1)
                or type(equal_length) is not int
                or equal_length not in (0, 1)
                or not isinstance(tokens, list)
                or any(not isinstance(token, str) or not token
                       for token in tokens)):
            raise BroadQaExternalDataError(
                "v7 variable structure observation schema 漂移")
        if identity_flag == 1 or equal_length == 1 or not tokens:
            continue
        observation_id = item.get("observation_id")
        source_family = item.get("source_family")
        input_text = item.get("input_text")
        output_text = item.get("output_text")
        if (not isinstance(observation_id, str) or len(observation_id) != 64
                or observation_id in seen
                or source_family not in V5_SOURCE_FAMILIES
                or not isinstance(input_text, str) or not input_text
                or not isinstance(output_text, str) or not output_text):
            raise BroadQaExternalDataError(
                "v7 variable structure observation identity 漂移")
        seen.add(observation_id)
        expected_tokens = tuple(tokens)
        shared_input = localization_structure_layout(input_text)
        shared_output = localization_structure_layout(output_text)
        shared_parser_match = int(
            tuple(shared_input["structure_tokens"])
            == expected_tokens
            == tuple(shared_output["structure_tokens"]))
        try:
            input_layout = localization_structure_layout_for_tokens(
                input_text, expected_tokens)
            output_layout = localization_structure_layout_for_tokens(
                output_text, expected_tokens)
            ledger_layout_success = 1
        except BroadQaExternalDataError:
            input_layout = {
                "raw_tokens": (), "segments": (), "structure_tokens": ()}
            output_layout = {
                "raw_tokens": (), "segments": (), "structure_tokens": ()}
            ledger_layout_success = 0
        raw_copy_equal = int(
            ledger_layout_success == 1
            and input_layout["raw_tokens"] == output_layout["raw_tokens"])
        segments = _segment_records(
            tuple(input_layout["segments"]),
            tuple(output_layout["segments"]),
        ) if ledger_layout_success else ()
        categories = tuple(
            localization_structure_token_category(token)
            for token in expected_tokens)
        identity = {
            "observation_id": observation_id,
            "target_policy_scope": VARIABLE_STRUCTURE_TARGET_SCOPE,
        }
        eligibility = int(
            ledger_layout_success == 1
            and raw_copy_equal == 1
            and len(segments) == len(expected_tokens) + 1)
        values.append({
            **identity,
            "atomic_commit_required": 1,
            "execution_allowed": 0,
            "format_version": 1,
            "input_length": len(input_text),
            "input_sha256": _text_sha256(input_text),
            "ledger_layout_success": ledger_layout_success,
            "obligation_count": sum(
                int(segment["proposal_required"]) for segment in segments),
            "output_length": len(output_text),
            "output_sha256": _text_sha256(output_text),
            "partial_commit_allowed": 0,
            "plan_id": _record_id(identity),
            "raw_structure_copy_equal": raw_copy_equal,
            "raw_structure_token_sha256": [
                _text_sha256(token) for token in input_layout["raw_tokens"]],
            "record_kind": VARIABLE_STRUCTURE_PLAN_KIND,
            "representation_eligible": eligibility,
            "segments": list(segments),
            "source_family": source_family,
            "shared_parser_match": shared_parser_match,
            "structure_categories": list(categories),
            "structure_tokens": list(expected_tokens),
        })
        counters["plan_count"] += 1
        counters["ledger_layout_failure_count"] += int(
            not ledger_layout_success)
        counters["shared_parser_mismatch_count"] += int(
            not shared_parser_match)
        counters["raw_structure_copy_mismatch_count"] += int(
            not raw_copy_equal)
        counters["representation_eligible_count"] += eligibility
        counters["obligation_count"] += sum(
            int(segment["proposal_required"]) for segment in segments)
    if not values:
        raise BroadQaExternalDataError("v7 variable structure plan 为空")
    values.sort(key=lambda value: str(value["plan_id"]))
    return tuple(values), {
        "deferred_structure_mismatch_count": sum(
            item["representation_eligible"] == 0 for item in values),
        "ledger_layout_failure_count": counters[
            "ledger_layout_failure_count"],
        "obligation_count": counters["obligation_count"],
        "plan_count": counters["plan_count"],
        "raw_structure_copy_mismatch_count": counters[
            "raw_structure_copy_mismatch_count"],
        "representation_eligible_count": counters[
            "representation_eligible_count"],
        "shared_parser_mismatch_count": counters[
            "shared_parser_mismatch_count"],
    }


def run_variable_structure_loso(
        plans: tuple[dict[str, object], ...],
        ) -> tuple[tuple[dict[str, object], ...], dict[str, object]]:
    """四方向只学习 category signature，并把 represented output 保持 UNKNOWN。"""
    if not isinstance(plans, tuple) or not plans:
        raise BroadQaExternalDataError("v7 variable structure LOSO plans 非法")
    values = []
    aggregate = Counter()
    for held_out in V5_SOURCE_FAMILIES:
        support: dict[tuple[str, ...], set[str]] = defaultdict(set)
        for plan in plans:
            family = plan.get("source_family")
            categories = plan.get("structure_categories")
            if (family not in V5_SOURCE_FAMILIES
                    or not isinstance(categories, list)
                    or any(not isinstance(item, str) or not item
                           for item in categories)):
                raise BroadQaExternalDataError(
                    "v7 variable structure LOSO plan schema 漂移")
            if family != held_out and plan.get("representation_eligible") == 1:
                support[tuple(categories)].add(str(family))
        learned = {
            signature for signature, families in support.items()
            if len(families) >= 2}
        held_plans = [plan for plan in plans
                      if plan["source_family"] == held_out]
        represented = sum(
            plan["representation_eligible"] == 1
            and tuple(plan["structure_categories"]) in learned
            for plan in held_plans)
        ineligible = sum(plan["representation_eligible"] == 0
                         for plan in held_plans)
        unknown = len(held_plans)
        identity = {
            "held_out_source_family": held_out,
            "target_policy_scope": VARIABLE_STRUCTURE_TARGET_SCOPE,
        }
        values.append({
            **identity,
            "exact_count": 0,
            "facility_outcome": "PASS",
            "format_version": 1,
            "held_out_inventory_count": len(held_plans),
            "ineligible_layout_count": ineligible,
            "learned_category_signature_count": len(learned),
            "loso_id": _record_id(identity),
            "partial_commit_count": 0,
            "record_kind": VARIABLE_STRUCTURE_LOSO_KIND,
            "represented_count": represented,
            "result_sha256": _sha256(canonical_json_bytes({
                "exact_count": 0,
                "held_out_inventory_count": len(held_plans),
                "represented_count": represented,
                "unknown_count": unknown,
                "wrong_count": 0,
            })),
            "segment_generator_available": 0,
            "status": (
                "CAPABILITY_NE_SEGMENT_GENERATOR_MISSING"
                if represented else "CAPABILITY_NE_COVERAGE_INSUFFICIENT"),
            "unknown_count": unknown,
            "wrong_count": 0,
        })
        aggregate["inventory_count"] += len(held_plans)
        aggregate["represented_count"] += represented
        aggregate["unknown_count"] += unknown
        aggregate["ineligible_layout_count"] += ineligible
    if aggregate["inventory_count"] != len(plans):
        raise BroadQaExternalDataError(
            "v7 variable structure LOSO inventory 未闭合")
    return tuple(values), {
        "capability_outcome": "NE_SEGMENT_GENERATOR_NOT_IMPLEMENTED",
        "deferred_structure_mismatch_count": aggregate[
            "ineligible_layout_count"],
        "exact_count": 0,
        "facility_outcome": "PASS",
        "held_out_inventory_count": aggregate["inventory_count"],
        "ineligible_layout_count": aggregate["ineligible_layout_count"],
        "loso_family_count": len(V5_SOURCE_FAMILIES),
        "partial_commit_count": 0,
        "represented_count": aggregate["represented_count"],
        "unknown_count": aggregate["unknown_count"],
        "wrong_count": 0,
    }


__all__ = [
    "VARIABLE_STRUCTURE_LOSO_KIND",
    "VARIABLE_STRUCTURE_PLAN_KIND",
    "VARIABLE_STRUCTURE_TARGET_SCOPE",
    "derive_variable_structure_plans",
    "run_variable_structure_loso",
]
