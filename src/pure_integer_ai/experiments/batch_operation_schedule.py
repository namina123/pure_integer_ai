"""Fail-closed operation assignment contract for batch production runs.

The production runner must preserve the operation specified for each case.
Restoring eight integer cues or rotating them over arbitrary facts cannot
establish eight behaviors. This module validates explicit assignments and
corresponding results; semantic behavior remains UNKNOWN until independent
production traces show a causal operation edit and response difference. The
older connector-only fixture intentionally uses one gate operation.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal


BATCH_OPERATION_SCHEDULE_FORMAT = "PURE_INTEGER_BATCH_OPERATION_SCHEDULE_V1"


class BatchOperationScheduleError(ValueError):
    """The durable operation assignment/result contract is invalid."""


def _key(value: object, field: str) -> tuple[int, ...]:
    if (not isinstance(value, (tuple, list)) or not value
            or any(type(item) is not int or item < 0 for item in value)):
        raise BatchOperationScheduleError(
            f"{field} requires a non-empty non-negative integer key")
    return tuple(value)


def _case_number(value: object, field: str = "case") -> int:
    if type(value) is not int or value <= 0:
        raise BatchOperationScheduleError(f"{field} requires a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class BatchOperationAssignment:
    """The exact operation input persisted for one production case."""

    case: int
    proposition_key: tuple[int, ...]
    operation_key: tuple[int, ...]
    operation_cue: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "case", _case_number(self.case))
        object.__setattr__(
            self, "proposition_key", _key(self.proposition_key, "proposition_key"))
        object.__setattr__(
            self, "operation_key", _key(self.operation_key, "operation_key"))
        object.__setattr__(
            self, "operation_cue", _key(self.operation_cue, "operation_cue"))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "BatchOperationAssignment":
        if not isinstance(value, Mapping):
            raise BatchOperationScheduleError("assignment must be a mapping")
        return cls(
            _case_number(value.get("case")),
            _key(value.get("proposition_key"), "proposition_key"),
            _key(value.get("operation_key"), "operation_key"),
            _key(value.get("operation_cue"), "operation_cue"),
        )

    def to_value(self) -> dict[str, object]:
        return {
            "case": self.case,
            "proposition_key": list(self.proposition_key),
            "operation_key": list(self.operation_key),
            "operation_cue": list(self.operation_cue),
        }


@dataclass(frozen=True, slots=True)
class BatchOperationScheduleAudit:
    """Assignment coverage without a semantic-operation claim."""

    status: Literal["PASS_ASSIGNMENT_CONTRACT"]
    assignments: tuple[BatchOperationAssignment, ...]
    known_operation_count: int
    covered_operation_count: int
    uncovered_operation_keys: tuple[tuple[int, ...], ...]
    behavior_qualification: Literal[
        "UNKNOWN_requires_independent_production_behavior_evidence"
    ] = "UNKNOWN_requires_independent_production_behavior_evidence"

    def to_value(self) -> dict[str, object]:
        return {
            "status": self.status,
            "assignments": [item.to_value() for item in self.assignments],
            "known_operation_count": self.known_operation_count,
            "covered_operation_count": self.covered_operation_count,
            "uncovered_operation_keys": [list(item)
                                         for item in self.uncovered_operation_keys],
            "behavior_qualification": self.behavior_qualification,
        }


def validate_batch_operation_assignments(
        assignments: Sequence[BatchOperationAssignment | Mapping[str, Any]],
        operation_cues: Mapping[tuple[int, ...], tuple[int, ...]], *,
        minimum_operations: int = 8,
        require_all_known_operations: bool = True,
        ) -> BatchOperationScheduleAudit:
    """Validate exact case-to-operation assignment and cue identity.

    The input is the durable schedule that must be written before workers are
    spawned.  ``require_all_known_operations`` is true for the current Wave 1
    operation graph; callers may disable it only for a deliberately smaller
    preflight, which still cannot report semantic behavior.
    """
    if type(minimum_operations) is not int or minimum_operations <= 0:
        raise BatchOperationScheduleError("minimum_operations must be positive")
    if not isinstance(operation_cues, Mapping) or not operation_cues:
        raise BatchOperationScheduleError("operation cue mapping is empty")
    normalized_cues: dict[tuple[int, ...], tuple[int, ...]] = {}
    for raw_key, raw_cue in operation_cues.items():
        key = _key(raw_key, "operation cue key")
        cue = _key(raw_cue, "operation cue")
        if key in normalized_cues and normalized_cues[key] != cue:
            raise BatchOperationScheduleError(
                "one operation key has multiple integer cues")
        normalized_cues[key] = cue
    if len(normalized_cues) < minimum_operations:
        raise BatchOperationScheduleError(
            "operation graph has fewer than the required operation count")
    if not isinstance(assignments, Sequence) or not assignments:
        raise BatchOperationScheduleError("operation assignment schedule is empty")

    parsed = tuple(
        item if isinstance(item, BatchOperationAssignment)
        else BatchOperationAssignment.from_mapping(item)
        for item in assignments
    )
    cases = tuple(item.case for item in parsed)
    if cases != tuple(range(1, len(parsed) + 1)):
        raise BatchOperationScheduleError(
            "operation assignments must be a contiguous case prefix")
    if len({item.proposition_key for item in parsed}) != len(parsed):
        raise BatchOperationScheduleError(
            "one proposition cannot be assigned to multiple cases")
    covered: set[tuple[int, ...]] = set()
    for item in parsed:
        expected = normalized_cues.get(item.operation_key)
        if expected is None:
            raise BatchOperationScheduleError(
                f"case {item.case} references an unknown operation key")
        if item.operation_cue != expected:
            raise BatchOperationScheduleError(
                f"case {item.case} operation cue does not match its key")
        covered.add(item.operation_key)
    uncovered = tuple(sorted(set(normalized_cues) - covered))
    if require_all_known_operations and uncovered:
        raise BatchOperationScheduleError(
            "operation assignment schedule does not cover every known operation")
    return BatchOperationScheduleAudit(
        status="PASS_ASSIGNMENT_CONTRACT",
        assignments=parsed,
        known_operation_count=len(normalized_cues),
        covered_operation_count=len(covered),
        uncovered_operation_keys=uncovered,
    )


def build_batch_operation_assignments(
        proposition_keys: Sequence[Sequence[int]],
        operation_cues: Mapping[tuple[int, ...], tuple[int, ...]], *,
        operation_keys: Sequence[Sequence[int]],
        minimum_operations: int = 8,
        ) -> tuple[BatchOperationAssignment, ...]:
    """Compile caller-frozen proposition/operation pairs, never invent pairs.

    ``operation_keys`` comes from the frozen evaluation case specification,
    not a counter, a sorted cue position, or runtime inference metadata. This
    contract checks route consistency only; compatibility between a relation
    and its requested behavior requires independent behavioral evidence.
    """
    if type(minimum_operations) is not int or minimum_operations <= 0:
        raise BatchOperationScheduleError("minimum_operations must be positive")
    if not isinstance(proposition_keys, Sequence) or not proposition_keys:
        raise BatchOperationScheduleError("proposition key schedule is empty")
    normalized_propositions = tuple(
        _key(value, "proposition_key") for value in proposition_keys)
    if len(set(normalized_propositions)) != len(normalized_propositions):
        raise BatchOperationScheduleError(
            "one proposition cannot be assigned to multiple cases")
    if not isinstance(operation_cues, Mapping) or not operation_cues:
        raise BatchOperationScheduleError("operation cue mapping is empty")
    normalized_cues: dict[tuple[int, ...], tuple[int, ...]] = {}
    for raw_key, raw_cue in operation_cues.items():
        key = _key(raw_key, "operation cue key")
        cue = _key(raw_cue, "operation cue")
        if key in normalized_cues and normalized_cues[key] != cue:
            raise BatchOperationScheduleError(
                "one operation key has multiple integer cues")
        normalized_cues[key] = cue
    if len(normalized_cues) < minimum_operations:
        raise BatchOperationScheduleError(
            "operation graph has fewer than the required operation count")
    if (not isinstance(operation_keys, Sequence)
            or len(operation_keys) != len(normalized_propositions)):
        raise BatchOperationScheduleError(
            "every proposition requires an explicit operation assignment")
    expected_operations = tuple(_key(value, "operation_key") for value in operation_keys)
    for operation in expected_operations:
        if operation not in normalized_cues:
            raise BatchOperationScheduleError("assignment references an unknown operation key")
    assignments = tuple(
        BatchOperationAssignment(
            case=index + 1,
            proposition_key=proposition,
            operation_key=operation,
            operation_cue=normalized_cues[operation],
        )
        for index, (proposition, operation) in enumerate(zip(
            normalized_propositions, expected_operations))
    )
    validate_batch_operation_assignments(
        assignments, normalized_cues, minimum_operations=minimum_operations)
    return assignments


def build_batch_operation_schedule_manifest(
        *, identity: str,
        assignments: Sequence[BatchOperationAssignment],
        operation_cues: Mapping[tuple[int, ...], tuple[int, ...]],
        minimum_operations: int = 8,
        ) -> dict[str, object]:
    """Create the immutable JSON-shaped schedule written before worker spawn."""
    if type(identity) is not str or not identity:
        raise BatchOperationScheduleError("schedule identity must be nonempty text")
    audit = validate_batch_operation_assignments(
        assignments, operation_cues, minimum_operations=minimum_operations)
    cues = {
        _key(key, "operation cue key"): _key(cue, "operation cue")
        for key, cue in operation_cues.items()
    }
    return {
        "format": BATCH_OPERATION_SCHEDULE_FORMAT,
        "identity": identity,
        "assignments": [item.to_value() for item in audit.assignments],
        "operation_cues": [
            {"operation_key": list(key), "operation_cue": list(cues[key])}
            for key in sorted(cues)
        ],
        "known_operation_count": audit.known_operation_count,
        "covered_operation_count": audit.covered_operation_count,
        "uncovered_operation_keys": [list(item)
                                     for item in audit.uncovered_operation_keys],
        "behavior_qualification": audit.behavior_qualification,
        "free_dialogue_complete": 0,
    }


def validate_batch_operation_schedule_manifest(
        manifest: Mapping[str, Any], *, identity: str,
        minimum_operations: int = 8,
        ) -> BatchOperationScheduleAudit:
    """Cold-validate a saved schedule without touching a graph or database."""
    if type(identity) is not str or not identity:
        raise BatchOperationScheduleError("schedule identity must be nonempty text")
    if not isinstance(manifest, Mapping):
        raise BatchOperationScheduleError("schedule manifest must be a mapping")
    if manifest.get("format") != BATCH_OPERATION_SCHEDULE_FORMAT \
            or manifest.get("identity") != identity:
        raise BatchOperationScheduleError("schedule manifest identity/format drift")
    if type(manifest.get("free_dialogue_complete")) is not int \
            or manifest["free_dialogue_complete"] != 0:
        raise BatchOperationScheduleError("schedule free_dialogue_complete must remain zero")
    raw_cues = manifest.get("operation_cues")
    if not isinstance(raw_cues, list):
        raise BatchOperationScheduleError("schedule operation cues are missing")
    cues: dict[tuple[int, ...], tuple[int, ...]] = {}
    for row in raw_cues:
        if not isinstance(row, Mapping):
            raise BatchOperationScheduleError("schedule operation cue row is invalid")
        key = _key(row.get("operation_key"), "operation cue key")
        cue = _key(row.get("operation_cue"), "operation cue")
        if key in cues:
            raise BatchOperationScheduleError("schedule operation key is duplicated")
        cues[key] = cue
    raw_assignments = manifest.get("assignments")
    if not isinstance(raw_assignments, list):
        raise BatchOperationScheduleError("schedule assignments are missing")
    audit = validate_batch_operation_assignments(
        raw_assignments, cues, minimum_operations=minimum_operations)
    for field, expected in (
            ("known_operation_count", audit.known_operation_count),
            ("covered_operation_count", audit.covered_operation_count),
            ("uncovered_operation_keys", [list(item)
                                           for item in audit.uncovered_operation_keys])):
        if manifest.get(field) != expected:
            raise BatchOperationScheduleError(f"schedule {field} drift")
    if manifest.get("behavior_qualification") != audit.behavior_qualification:
        raise BatchOperationScheduleError("schedule behavior qualification drift")
    return audit


def validate_batch_operation_results(
        assignments: Sequence[BatchOperationAssignment],
        results: Sequence[Mapping[str, Any]],
        ) -> tuple[dict[str, object], ...]:
    """Check that each compact result consumed the operation assigned to it.

    This is a routing/result integrity check only.  It does not infer that an
    operation is semantically distinct from another operation with a different
    integer key.
    """
    if not isinstance(assignments, Sequence) or not assignments:
        raise BatchOperationScheduleError("result audit has no assignments")
    if not isinstance(results, Sequence):
        raise BatchOperationScheduleError("result audit results must be a sequence")
    by_case: dict[int, Mapping[str, Any]] = {}
    for raw in results:
        if not isinstance(raw, Mapping):
            raise BatchOperationScheduleError("result must be a mapping")
        case = _case_number(raw.get("case"))
        if case in by_case:
            raise BatchOperationScheduleError("duplicate operation result case")
        by_case[case] = raw
    expected_cases = tuple(item.case for item in assignments)
    if tuple(sorted(by_case)) != expected_cases:
        raise BatchOperationScheduleError("operation result cases do not match assignments")
    audited = []
    for assignment in assignments:
        result = by_case[assignment.case]
        result_operation = _key(result.get("operation_key"), "result.operation_key")
        if result_operation != assignment.operation_key:
            raise BatchOperationScheduleError(
                f"case {assignment.case} result operation key drift")
        result_proposition = _key(
            result.get("proposition_key"), "result.proposition_key")
        if result_proposition != assignment.proposition_key:
            raise BatchOperationScheduleError(
                f"case {assignment.case} result proposition drift")
        inferred = result.get("operation_keys_inferred")
        consumed = result.get("operation_keys_consumed")
        if not isinstance(inferred, list) or not isinstance(consumed, list):
            raise BatchOperationScheduleError(
                f"case {assignment.case} lacks operation consumption evidence")
        inferred_keys = tuple(_key(item, "operation_keys_inferred") for item in inferred)
        consumed_keys = tuple(_key(item, "operation_keys_consumed") for item in consumed)
        if inferred_keys != (assignment.operation_key,):
            raise BatchOperationScheduleError(
                f"case {assignment.case} inferred operation differs from assignment")
        if consumed_keys != (assignment.operation_key,):
            raise BatchOperationScheduleError(
                f"case {assignment.case} consumed operation differs from assignment")
        audited.append({
            "case": assignment.case,
            "operation_key": list(assignment.operation_key),
            "proposition_key": list(assignment.proposition_key),
            "operation_assignment_verified": 1,
            "semantic_behavior": "UNKNOWN_requires_independent_counterfactual_audit",
        })
    return tuple(audited)


__all__ = [
    "BATCH_OPERATION_SCHEDULE_FORMAT",
    "BatchOperationAssignment",
    "BatchOperationScheduleAudit",
    "BatchOperationScheduleError",
    "build_batch_operation_assignments",
    "build_batch_operation_schedule_manifest",
    "validate_batch_operation_assignments",
    "validate_batch_operation_schedule_manifest",
    "validate_batch_operation_results",
]
