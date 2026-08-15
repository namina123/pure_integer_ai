"""Derive recovery-v8 TRAIN-only family-LOSO cases and hard-gate results."""
from __future__ import annotations

from collections import Counter
import hashlib

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_interpreter import (
    build_normalization_recovery_v8_rule_index,
    interpret_normalization_recovery_v8_indexed,
    interpret_normalization_recovery_v8_reference,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_training_records import (
    V8_TRAIN_FAMILIES,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes


V8_TRAIN_CASE_KIND = "NORMALIZATION_RECOVERY_V8_TRAIN_CASE_V1"
V8_TRAIN_RESULT_KIND = "NORMALIZATION_RECOVERY_V8_TRAIN_RESULT_V1"
V8_TRAIN_AUDIT_CENSUS_KIND = "NORMALIZATION_RECOVERY_V8_TRAIN_AUDIT_CENSUS_V1"

V8_TRAIN_AUDIT_FILES = (
    ("train-cases.jsonl", "TRAIN_ONLY_FAMILY_LOSO_CASES", "case_id"),
    ("train-results.jsonl", "TRAIN_ONLY_DUAL_INTERPRETER_RESULTS", "result_id"),
    ("audit-census.jsonl", "TRAIN_ONLY_HARD_GATE_CENSUS", "record_kind"),
)

_AUTHORIZED_FILES = (
    "authorized-orthographic-atoms.jsonl",
    "authorized-source-conditioned-lexical-atoms.jsonl",
    "authorized-layout-morphology-obligations.jsonl",
)


def _sha(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _authorization_index(
        protocol_outputs: dict[str, tuple[dict[str, object], ...]],
        ) -> dict[str, dict[str, object]]:
    values = tuple(item for name in _AUTHORIZED_FILES
                   for item in protocol_outputs.get(name, ()))
    result = {str(item.get("authorization_id")): item for item in values}
    if len(result) != len(values) or not result:
        raise BroadQaExternalDataError("v8 audit authorization inventory 漂移")
    return result


def _authorization_case(
        authorization: dict[str, object], plan: dict[str, object],
        ) -> dict[str, object]:
    """Build one real held-out-support case from a frozen LOSO plan."""
    ledger = str(authorization["ledger_kind"])
    query = {
        "held_out_family": plan["held_out_family"],
        "input_text": "",
        "official_source_text": "",
        "query_kind": ledger,
        "structure_tokens": [],
    }
    expected_text = ""
    expected_tokens: list[str] = []
    if ledger == "ORTHOGRAPHIC_ATOM":
        query["input_text"] = authorization["input_atom"]
        expected_text = str(authorization["output_atom"])
    elif ledger == "SOURCE_CONDITIONED_LEXICAL_ATOM":
        query["input_text"] = authorization["input_text"]
        query["official_source_text"] = authorization["official_source_text"]
        expected_text = str(authorization["output_text"])
    elif ledger == "LAYOUT_MORPHOLOGY_OBLIGATION":
        expected_tokens = list(authorization["structure_tokens"])
        query["structure_tokens"] = expected_tokens
    else:
        raise BroadQaExternalDataError("v8 audit authorization ledger 漂移")
    identity = {"case_source_id": plan["loso_plan_id"], "query": query}
    return {
        "case_id": _sha(identity),
        "case_source_id": plan["loso_plan_id"],
        "case_source_kind": "AUTHORIZED_CANDIDATE_LOSO_PLAN",
        "expected_behavior": plan["expected_behavior"],
        "expected_output_structure_tokens": (
            expected_tokens if plan["expected_behavior"] == "EXACT" else []),
        "expected_output_text": (
            expected_text if plan["expected_behavior"] == "EXACT" else ""),
        "format_version": 1,
        "query": query,
        "record_kind": V8_TRAIN_CASE_KIND,
    }


def _identity_cases(
        protocol_outputs: dict[str, tuple[dict[str, object], ...]],
        ) -> tuple[dict[str, object], ...]:
    """Evaluate identity veto only where the held-out family has the case."""
    values = []
    for item in protocol_outputs.get("exact-input-control.jsonl", ()):
        outputs = item.get("outputs")
        families = item.get("support_families")
        if (item.get("candidate_status") != "MULTI_FAMILY_UNIQUE_OUTPUT"
                or not isinstance(outputs, list) or len(outputs) != 1
                or not isinstance(outputs[0], dict)
                or outputs[0].get("output_text") != item.get("input_text")):
            continue
        if not isinstance(families, list) or any(
                family not in V8_TRAIN_FAMILIES for family in families):
            raise BroadQaExternalDataError("v8 audit identity support 漂移")
        for held_out in families:
            after = [family for family in families if family != held_out]
            expected = "EXACT" if len(after) >= 2 else "UNKNOWN"
            query = {
                "held_out_family": held_out,
                "input_text": item["input_text"],
                "official_source_text": "",
                "query_kind": "IDENTITY_VETO",
                "structure_tokens": [],
            }
            identity = {"case_source_id": item["candidate_id"], "query": query}
            values.append({
                "case_id": _sha(identity),
                "case_source_id": item["candidate_id"],
                "case_source_kind": "IDENTITY_VETO_CONTROL",
                "expected_behavior": expected,
                "expected_output_structure_tokens": [],
                "expected_output_text": (
                    str(item["input_text"]) if expected == "EXACT" else ""),
                "format_version": 1,
                "query": query,
                "record_kind": V8_TRAIN_CASE_KIND,
            })
    return tuple(values)


def _cases(
        protocol_outputs: dict[str, tuple[dict[str, object], ...]],
        ) -> tuple[dict[str, object], ...]:
    """Build authorization and identity cases without no-case directions."""
    authorizations = _authorization_index(protocol_outputs)
    values = []
    for plan in protocol_outputs.get("family-loso-plan.jsonl", ()):
        if plan.get("evaluation_case_expected") != 1:
            continue
        authorization = authorizations.get(str(plan.get("authorization_id")))
        if authorization is None:
            raise BroadQaExternalDataError("v8 audit LOSO authorization 缺失")
        values.append(_authorization_case(authorization, plan))
    values.extend(_identity_cases(protocol_outputs))
    result = tuple(sorted(values, key=lambda item: str(item["case_id"])))
    if not result or len({item["case_id"] for item in result}) != len(result):
        raise BroadQaExternalDataError("v8 audit case identity 漂移")
    return result


def _judge(case: dict[str, object], result: dict[str, object]) -> str:
    """Return EXACT, UNKNOWN or WRONG against the frozen expectation."""
    expected = case["expected_behavior"]
    if expected == "UNKNOWN":
        return ("UNKNOWN" if result["behavior"] == "UNKNOWN"
                and result["output_text"] == ""
                and result["output_structure_tokens"] == [] else "WRONG")
    if expected == "EXACT":
        return ("EXACT" if result["behavior"] == "EXACT"
                and result["output_text"] == case["expected_output_text"]
                and result["output_structure_tokens"]
                == case["expected_output_structure_tokens"] else "WRONG")
    raise BroadQaExternalDataError("v8 audit expected behavior 漂移")


def _results(
        rule_outputs: dict[str, tuple[dict[str, object], ...]],
        cases: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, object], ...]:
    """Run both interpreters and retain every hard-gate fact."""
    index = build_normalization_recovery_v8_rule_index(rule_outputs)
    values = []
    for case in cases:
        exception_count = 0
        try:
            reference = interpret_normalization_recovery_v8_reference(
                rule_outputs, case["query"])
            indexed = interpret_normalization_recovery_v8_indexed(
                index, case["query"])
        except Exception as error:
            exception_count = 1
            reference = {"error_type": type(error).__name__}
            indexed = {"error_type": type(error).__name__}
        mismatch = int(reference != indexed)
        judgement = "WRONG" if exception_count else _judge(case, reference)
        identity = {"case_id": case["case_id"], "result_kind": "DUAL_INTERPRETER"}
        values.append({
            "case_id": case["case_id"],
            "exception_count": exception_count,
            "format_version": 1,
            "indexed_reference_mismatch_count": mismatch,
            "judgement": judgement,
            "record_kind": V8_TRAIN_RESULT_KIND,
            "reference_result": reference,
            "result_id": _sha(identity),
            "structure_mismatch_count": (
                int(reference.get("structure_mismatch_count", 0))
                if isinstance(reference, dict) else 1),
        })
    return tuple(values)


def _summary(
        cases: tuple[dict[str, object], ...],
        results: tuple[dict[str, object], ...],
        ) -> dict[str, object]:
    """Compute all frozen TRAIN-only acceptance gates."""
    judgement_counts = Counter(str(item["judgement"]) for item in results)
    expected_counts = Counter(str(item["expected_behavior"]) for item in cases)
    source_counts = Counter(str(item["case_source_kind"]) for item in cases)
    summary = {
        "case_count": len(cases),
        "case_source_kind_counts": dict(sorted(source_counts.items())),
        "exception_count": sum(int(item["exception_count"]) for item in results),
        "expected_behavior_counts": dict(sorted(expected_counts.items())),
        "generation_hard_conjunct_pass": int(all(
            item["judgement"] in {"EXACT", "UNKNOWN"} for item in results)),
        "indexed_reference_mismatch_count": sum(
            int(item["indexed_reference_mismatch_count"]) for item in results),
        "judgement_counts": dict(sorted(judgement_counts.items())),
        "partial_commit_count": 0,
        "result_count": len(results),
        "structure_mismatch_count": sum(
            int(item["structure_mismatch_count"]) for item in results),
        "wrong_count": judgement_counts["WRONG"],
    }
    summary["hard_gates_pass"] = int(
        summary["wrong_count"] == 0
        and summary["indexed_reference_mismatch_count"] == 0
        and summary["partial_commit_count"] == 0
        and summary["structure_mismatch_count"] == 0
        and summary["exception_count"] == 0
        and summary["generation_hard_conjunct_pass"] == 1)
    return summary


def derive_normalization_recovery_v8_training_audit(
        *, protocol_outputs: dict[str, tuple[dict[str, object], ...]],
        rule_outputs: dict[str, tuple[dict[str, object], ...]],
        ) -> tuple[
            dict[str, tuple[dict[str, object], ...]], dict[str, object]]:
    """Derive cases, dual-interpreter results and a hard-gate census."""
    cases = _cases(protocol_outputs)
    results = _results(rule_outputs, cases)
    summary = _summary(cases, results)
    census = ({
        **summary,
        "format_version": 1,
        "record_kind": V8_TRAIN_AUDIT_CENSUS_KIND,
    },)
    return {
        "train-cases.jsonl": cases,
        "train-results.jsonl": results,
        "audit-census.jsonl": census,
    }, summary


__all__ = [
    "V8_TRAIN_AUDIT_FILES",
    "derive_normalization_recovery_v8_training_audit",
]
