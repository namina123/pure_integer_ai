"""Read case-level evaluator labels without admitting them to training views."""
from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

from pure_integer_ai.experiments.integer_relation_connected_course import (
    ConnectedCaseAssignment,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    parse_canonical_json_bytes,
)


_CASE_FIELDS = frozenset({"case_id", "input", "operation", "seed_ids", "split"})
_LABEL_FIELDS = frozenset({"case_id", "policy", "seed_ids"})
_POLICIES = frozenset({
    "DIRECT_ONLY", "DIRECT_PLUS_UNKNOWN", "UNKNOWN",
    "COMPOSED_REQUIRE_RULE", "CLARIFY_CONFLICT",
})
_SPLIT_CODES = {"train": 1, "held_out": 2, "negative": 3, "conflict": 4}


class IntegerRelationCaseLabelAuditError(ValueError):
    """The evaluator's case labels are incomplete or leak across ownership."""


def _rows(path: Path, fields: frozenset[str]) -> tuple[dict, ...]:
    raw = path.read_bytes()
    if not raw or not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise IntegerRelationCaseLabelAuditError("case/label source newline drifted")
    result = []
    for number, line in enumerate(raw.splitlines(), 1):
        try:
            row = parse_canonical_json_bytes(line, require_object=True)
        except ValueError as error:
            raise IntegerRelationCaseLabelAuditError(
                f"case/label line {number} is not canonical") from error
        if set(row) != fields:
            raise IntegerRelationCaseLabelAuditError(
                f"case/label line {number} fields drifted")
        result.append(row)
    return tuple(result)


def audit_integer_relation_case_labels(
        case_path: str | Path,
        label_path: str | Path,
        assignments: Sequence[ConnectedCaseAssignment],
        *,
        expected_case_sha256: str,
        ) -> dict[str, object]:
    """Audit coverage and owner separation, never returning a label table."""
    case_file = Path(case_path).resolve()
    label_file = Path(label_path).resolve()
    if (label_file.parent.name != "evaluator"
            or label_file.parent.parent.name != "owners"
            or label_file.parent.parent.parent.name != "ph2"
            or label_file.parent.parent.parent.parent.name != "data"):
        raise IntegerRelationCaseLabelAuditError(
            "case labels must have a separate data/ph2/owners/evaluator owner")
    case_hash = hashlib.sha256(case_file.read_bytes()).hexdigest()
    if case_hash != expected_case_sha256:
        raise IntegerRelationCaseLabelAuditError("case source SHA-256 drifted")
    cases = _rows(case_file, _CASE_FIELDS)
    labels = _rows(label_file, _LABEL_FIELDS)
    if (not isinstance(assignments, (tuple, list))
            or len(assignments) != len(cases)
            or any(not isinstance(item, ConnectedCaseAssignment)
                   for item in assignments)):
        raise IntegerRelationCaseLabelAuditError("integer assignments do not cover cases")
    case_by_id = {}
    for number, (case, assignment) in enumerate(zip(cases, assignments, strict=True), 1):
        case_id = case["case_id"]
        if (not isinstance(case_id, str) or not case_id
                or case_id in case_by_id
                or assignment.case_key != (92002, number)
                or assignment.split_code != _SPLIT_CODES.get(case["split"])
                or assignment.input_values != tuple(map(ord, case["input"]))
                or len(assignment.observation_keys) != len(case["seed_ids"])):
            raise IntegerRelationCaseLabelAuditError("case/assignment binding drifted")
        case_by_id[case_id] = case
    label_ids = set()
    policy_counts = {policy: 0 for policy in sorted(_POLICIES)}
    for label in labels:
        case_id = label["case_id"]
        case = case_by_id.get(case_id)
        if (case is None or case_id in label_ids
                or case["split"] == "train"
                or label["seed_ids"] != case["seed_ids"]
                or label["policy"] not in _POLICIES):
            raise IntegerRelationCaseLabelAuditError(
                "evaluator label is duplicated, misplaced, or not case-bound")
        label_ids.add(case_id)
        policy_counts[label["policy"]] += 1
        if (case["split"] == "negative" and label["policy"] != "UNKNOWN"
                or case["split"] == "conflict"
                and label["policy"] != "CLARIFY_CONFLICT"):
            raise IntegerRelationCaseLabelAuditError(
                "negative/conflict case policy does not fail closed")
    expected_ids = {case_id for case_id, case in case_by_id.items()
                    if case["split"] != "train"}
    if label_ids != expected_ids:
        raise IntegerRelationCaseLabelAuditError(
            "evaluator labels do not exactly cover evaluation cases")
    label_hash = hashlib.sha256(label_file.read_bytes()).hexdigest()
    return {
        "status": "PASS_EVALUATOR_CASE_LABEL_OWNERSHIP_ONLY",
        "case_sha256": case_hash,
        "label_sha256": label_hash,
        "label_owner_identity": [92005, int(label_hash[:15], 16)],
        "train_label_count": 0,
        "evaluation_label_count": len(labels),
        "policy_counts": {key: value for key, value in policy_counts.items() if value},
        "semantic_behavior_qualification": "UNKNOWN_requires_independent_production_audit",
        "free_dialogue_complete": 0,
    }


__all__ = [
    "IntegerRelationCaseLabelAuditError", "audit_integer_relation_case_labels",
]
