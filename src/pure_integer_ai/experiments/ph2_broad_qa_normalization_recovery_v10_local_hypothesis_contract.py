"""定义 recovery-v10 局部 hypothesis 与整句提交闭合合同。"""
from __future__ import annotations

import hashlib

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_localization_structure import (
    localization_structure_tokens,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_training_records import (
    V8_TRAIN_FAMILIES,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes


NORMALIZATION_RECOVERY_V10_LOCAL_HYPOTHESIS_CONTRACT_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V10_LOCAL_HYPOTHESIS_CONTRACT_V1")
NORMALIZATION_RECOVERY_V10_LOCAL_HYPOTHESIS_CONTRACT_STATUS = (
    "TRAIN_DESIGN_DISABLED_NOT_FORMAL_NOT_PRODUCTION")

LOCAL_ORTHOGRAPHIC_HYPOTHESIS_ONLY = "LOCAL_ORTHOGRAPHIC_HYPOTHESIS_ONLY"
SOURCE_CONTEXT_EXACT_SPAN_AUTHORIZATION = (
    "SOURCE_CONTEXT_EXACT_SPAN_AUTHORIZATION")
_AUTHORIZATION_KINDS = {
    LOCAL_ORTHOGRAPHIC_HYPOTHESIS_ONLY,
    SOURCE_CONTEXT_EXACT_SPAN_AUTHORIZATION,
}


def _sha256(value: object) -> str:
    """返回 span、proposal、result 或 preflight 的规范 SHA-256。"""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha_value(value: object, *, label: str) -> str:
    """核验并返回小写 SHA-256。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(item not in "0123456789abcdef" for item in value)):
        raise BroadQaExternalDataError(
            f"v10 local hypothesis {label} 非法")
    return value


def _sha_list(value: object, *, label: str) -> list[str]:
    """核验非空、有序、唯一的 SHA 列表。"""
    if (not isinstance(value, list) or not value
            or value != sorted(set(value))):
        raise BroadQaExternalDataError(
            f"v10 local hypothesis {label} 漂移")
    return [_sha_value(item, label=label) for item in value]


def _families(value: object) -> list[str]:
    """核验非空、有序、唯一且属于 TRAIN 的来源 family。"""
    if (not isinstance(value, list) or not value
            or value != sorted(set(value))
            or any(item not in V8_TRAIN_FAMILIES for item in value)):
        raise BroadQaExternalDataError(
            "v10 local hypothesis support families 漂移")
    return value


def _strict_nonnegative(value: object, *, label: str) -> int:
    """核验严格整数的非负计数。"""
    if type(value) is not int or value < 0:
        raise BroadQaExternalDataError(
            f"v10 local hypothesis {label} 非法")
    return value


def _query(value: dict[str, object]) -> dict[str, object]:
    """核验输入、official source 与结构 token 自洽。"""
    if not isinstance(value, dict) or set(value) != {
            "input_text", "official_source_text", "structure_tokens"}:
        raise BroadQaExternalDataError(
            "v10 local hypothesis query schema 漂移")
    input_text = value.get("input_text")
    source = value.get("official_source_text")
    tokens = value.get("structure_tokens")
    if (not isinstance(input_text, str) or not input_text
            or not isinstance(source, str) or not source
            or not isinstance(tokens, list)
            or any(not isinstance(item, str) or not item for item in tokens)
            or tuple(tokens) != localization_structure_tokens(input_text)):
        raise BroadQaExternalDataError(
            "v10 local hypothesis query value 漂移")
    return value


def build_normalization_recovery_v10_local_span_hypothesis(
        *, input_start: int,
        input_end: int,
        input_text: str,
        output_text: str,
        authorization_kind: str,
        evidence_ids: list[str],
        opencc_route_ids: list[str],
        training_support_families: list[str],
        source_context_authorization_id: str,
        authorized_official_source_text: str,
        identity_veto_count: int = 0,
        conflict_count: int = 0,
        ) -> dict[str, object]:
    """构造自绑定局部 span；hypothesis-only 永不自动升级为授权。"""
    if (type(input_start) is not int or input_start < 0
            or type(input_end) is not int or input_end <= input_start
            or not isinstance(input_text, str) or not input_text
            or not isinstance(output_text, str) or not output_text
            or input_text == output_text
            or authorization_kind not in _AUTHORIZATION_KINDS):
        raise BroadQaExternalDataError(
            "v10 local hypothesis span semantic 非法")
    evidence = _sha_list(evidence_ids, label="evidence ids")
    routes = _sha_list(opencc_route_ids, label="OpenCC route ids")
    families = _families(training_support_families)
    veto_count = _strict_nonnegative(
        identity_veto_count, label="identity veto count")
    conflicts = _strict_nonnegative(conflict_count, label="conflict count")
    if authorization_kind == SOURCE_CONTEXT_EXACT_SPAN_AUTHORIZATION:
        source_authorization = _sha_value(
            source_context_authorization_id,
            label="source context authorization id")
        if (families != list(V8_TRAIN_FAMILIES)
                or not isinstance(authorized_official_source_text, str)
                or not authorized_official_source_text):
            raise BroadQaExternalDataError(
                "v10 local hypothesis authorized family 未闭合")
        authorized_source = authorized_official_source_text
    else:
        if (source_context_authorization_id != ""
                or authorized_official_source_text != ""):
            raise BroadQaExternalDataError(
                "v10 local hypothesis hypothesis-only 不得携带授权")
        source_authorization = ""
        authorized_source = ""
    payload = {
        "authorization_kind": authorization_kind,
        "authorized_official_source_text": authorized_source,
        "conflict_count": conflicts,
        "evidence_ids": evidence,
        "identity_veto_count": veto_count,
        "input_end": input_end,
        "input_start": input_start,
        "input_text": input_text,
        "offset_unit": "UNICODE_SCALAR_INDEX",
        "opencc_route_ids": routes,
        "output_text": output_text,
        "source_context_authorization_id": source_authorization,
        "training_support_families": families,
    }
    return {**payload, "span_hypothesis_id": _sha256(payload)}


def _validate_span(value: object) -> dict[str, object]:
    """重建并核验 span 自绑定身份。"""
    if not isinstance(value, dict) or set(value) != {
            "authorization_kind", "authorized_official_source_text",
            "conflict_count", "evidence_ids",
            "identity_veto_count", "input_end", "input_start", "input_text",
            "offset_unit", "opencc_route_ids", "output_text",
            "source_context_authorization_id", "span_hypothesis_id",
            "training_support_families"}:
        raise BroadQaExternalDataError(
            "v10 local hypothesis span schema 漂移")
    rebuilt = build_normalization_recovery_v10_local_span_hypothesis(
        input_start=value["input_start"],
        input_end=value["input_end"],
        input_text=value["input_text"],
        output_text=value["output_text"],
        authorization_kind=value["authorization_kind"],
        evidence_ids=value["evidence_ids"],
        opencc_route_ids=value["opencc_route_ids"],
        training_support_families=value["training_support_families"],
        source_context_authorization_id=value[
            "source_context_authorization_id"],
        authorized_official_source_text=value[
            "authorized_official_source_text"],
        identity_veto_count=value["identity_veto_count"],
        conflict_count=value["conflict_count"],
    )
    if rebuilt != value:
        raise BroadQaExternalDataError(
            "v10 local hypothesis span identity 漂移")
    return value


def execute_normalization_recovery_v10_local_hypothesis_contract(
        *, query: dict[str, object],
        span_hypotheses: tuple[dict[str, object], ...],
        ) -> dict[str, object]:
    """形成可审计局部trace；只有全部changed span闭合才提交整句。"""
    checked_query = _query(query)
    if not isinstance(span_hypotheses, tuple) or not span_hypotheses:
        raise BroadQaExternalDataError(
            "v10 local hypothesis span batch 为空")
    spans = tuple(sorted(
        (_validate_span(item) for item in span_hypotheses),
        key=lambda item: (
            int(item["input_start"]), int(item["input_end"]),
            str(item["span_hypothesis_id"]))))
    if len({item["span_hypothesis_id"] for item in spans}) != len(spans):
        raise BroadQaExternalDataError(
            "v10 local hypothesis span identity 重复")
    input_text = str(checked_query["input_text"])
    geometry_failure_count = 0
    source_slice_mismatch_count = 0
    previous_end = 0
    output_parts = []
    for span in spans:
        start = int(span["input_start"])
        end = int(span["input_end"])
        if start < previous_end or end > len(input_text):
            geometry_failure_count += 1
            continue
        if input_text[start:end] != span["input_text"]:
            source_slice_mismatch_count += 1
        output_parts.append(input_text[previous_end:start])
        output_parts.append(str(span["output_text"]))
        previous_end = end
    proposed_output = ""
    if geometry_failure_count == 0:
        output_parts.append(input_text[previous_end:])
        proposed_output = "".join(output_parts)
    authorized_count = sum(
        item["authorization_kind"]
        == SOURCE_CONTEXT_EXACT_SPAN_AUTHORIZATION for item in spans)
    hypothesis_only_count = len(spans) - authorized_count
    identity_veto_count = sum(
        int(item["identity_veto_count"]) for item in spans)
    conflict_count = sum(int(item["conflict_count"]) for item in spans)
    source_context_mismatch_count = sum(
        item["authorization_kind"]
        == SOURCE_CONTEXT_EXACT_SPAN_AUTHORIZATION
        and item["authorized_official_source_text"]
        != checked_query["official_source_text"]
        for item in spans)
    structure_mismatch_count = int(
        bool(proposed_output)
        and localization_structure_tokens(proposed_output)
        != tuple(checked_query["structure_tokens"]))
    all_changed_spans_authorized = int(authorized_count == len(spans))
    commit_allowed = int(
        geometry_failure_count == 0
        and source_slice_mismatch_count == 0
        and identity_veto_count == 0
        and conflict_count == 0
        and source_context_mismatch_count == 0
        and structure_mismatch_count == 0
        and all_changed_spans_authorized == 1)
    if geometry_failure_count:
        reason = "SPAN_GEOMETRY_REJECTED"
    elif source_slice_mismatch_count:
        reason = "SOURCE_SLICE_MISMATCH_REJECTED"
    elif identity_veto_count:
        reason = "IDENTITY_VETO_REJECTED"
    elif conflict_count:
        reason = "SPAN_CONFLICT_REJECTED"
    elif source_context_mismatch_count:
        reason = "SOURCE_CONTEXT_AUTHORIZATION_MISMATCH"
    elif not all_changed_spans_authorized:
        reason = "LOCAL_HYPOTHESIS_NOT_SENTENCE_AUTHORIZATION"
    elif structure_mismatch_count:
        reason = "STRUCTURE_PRESERVATION_HARD_GATE_REJECTED"
    else:
        reason = "ALL_CHANGED_SPANS_SOURCE_CONTEXT_AUTHORIZED"
    output_text = proposed_output if commit_allowed else ""
    payload = {
        "all_changed_spans_authorized": all_changed_spans_authorized,
        "artifact_kind": (
            NORMALIZATION_RECOVERY_V10_LOCAL_HYPOTHESIS_CONTRACT_KIND),
        "authorized_span_count": authorized_count,
        "behavior": "EXACT" if commit_allowed else "UNKNOWN",
        "changed_span_count": len(spans),
        "conflict_count": conflict_count,
        "formal_or_evaluation_payload_read_count": 0,
        "geometry_failure_count": geometry_failure_count,
        "hypothesis_only_span_count": hypothesis_only_count,
        "identity_veto_count": identity_veto_count,
        "input_text": input_text,
        "mastery_claimed": 0,
        "official_source_text": checked_query["official_source_text"],
        "output_text": output_text,
        "partial_commit_count": 0,
        "production_enabled": 0,
        "proposed_output_sha256": _sha256(proposed_output),
        "reason": reason,
        "source_slice_mismatch_count": source_slice_mismatch_count,
        "source_context_mismatch_count": source_context_mismatch_count,
        "span_hypotheses": list(spans),
        "status": NORMALIZATION_RECOVERY_V10_LOCAL_HYPOTHESIS_CONTRACT_STATUS,
        "structure_mismatch_count": structure_mismatch_count,
        "teacher_api_llm_call_count": 0,
    }
    return {**payload, "contract_result_sha256": _sha256(payload)}


def derive_normalization_recovery_v10_local_hypothesis_preflight(
        cases: tuple[dict[str, object], ...],
        ) -> dict[str, object]:
    """汇总 synthetic 合同 case，硬分离局部trace与整句answer。"""
    if not isinstance(cases, tuple) or not cases:
        raise BroadQaExternalDataError(
            "v10 local hypothesis preflight 为空")
    failures = 0
    exact_count = 0
    unknown_count = 0
    for case in cases:
        if not isinstance(case, dict) or set(case) != {
                "expected_behavior", "expected_reason", "query",
                "span_hypotheses"}:
            raise BroadQaExternalDataError(
                "v10 local hypothesis preflight case 漂移")
        result = execute_normalization_recovery_v10_local_hypothesis_contract(
            query=case["query"],
            span_hypotheses=case["span_hypotheses"],
        )
        failures += int(
            result["behavior"] != case["expected_behavior"]
            or result["reason"] != case["expected_reason"]
            or (result["behavior"] == "UNKNOWN" and bool(result["output_text"]))
            or result["partial_commit_count"] != 0
            or result["production_enabled"] != 0)
        exact_count += int(result["behavior"] == "EXACT")
        unknown_count += int(result["behavior"] == "UNKNOWN")
    payload = {
        "case_count": len(cases),
        "exact_count": exact_count,
        "failure_count": failures,
        "format_version": 1,
        "partial_commit_count": 0,
        "production_enabled": 0,
        "status": "PASS" if failures == 0 else "FAIL",
        "unknown_count": unknown_count,
    }
    return {**payload, "preflight_sha256": _sha256(payload)}


__all__ = [
    "LOCAL_ORTHOGRAPHIC_HYPOTHESIS_ONLY",
    "NORMALIZATION_RECOVERY_V10_LOCAL_HYPOTHESIS_CONTRACT_KIND",
    "NORMALIZATION_RECOVERY_V10_LOCAL_HYPOTHESIS_CONTRACT_STATUS",
    "SOURCE_CONTEXT_EXACT_SPAN_AUTHORIZATION",
    "build_normalization_recovery_v10_local_span_hypothesis",
    "derive_normalization_recovery_v10_local_hypothesis_preflight",
    "execute_normalization_recovery_v10_local_hypothesis_contract",
]
