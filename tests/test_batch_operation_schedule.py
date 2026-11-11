from __future__ import annotations

import pytest

from pure_integer_ai.experiments.batch_operation_schedule import (
    BatchOperationAssignment,
    BatchOperationScheduleError,
    build_batch_operation_assignments,
    build_batch_operation_schedule_manifest,
    validate_batch_operation_assignments,
    validate_batch_operation_schedule_manifest,
    validate_batch_operation_results,
)


def _cues() -> dict[tuple[int, ...], tuple[int, ...]]:
    return { (100 + i,): (200 + i,) for i in range(8) }


def _assignments() -> tuple[BatchOperationAssignment, ...]:
    cues = _cues()
    return tuple(
        BatchOperationAssignment(
            case=i + 1,
            proposition_key=(900 + i,),
            operation_key=(100 + i,),
            operation_cue=cues[(100 + i,)],
        )
        for i in range(8)
    )


def _results(assignments: tuple[BatchOperationAssignment, ...]) -> tuple[dict, ...]:
    return tuple({
        "case": item.case,
        "proposition_key": list(item.proposition_key),
        "operation_key": list(item.operation_key),
        "operation_keys_inferred": [list(item.operation_key)],
        "operation_keys_consumed": [list(item.operation_key)],
    } for item in assignments)


def test_assignment_contract_covers_all_known_operations_without_semantic_claim() -> None:
    result = validate_batch_operation_assignments(_assignments(), _cues())
    assert result.status == "PASS_ASSIGNMENT_CONTRACT"
    assert result.covered_operation_count == 8
    assert result.uncovered_operation_keys == ()
    assert result.behavior_qualification.startswith("UNKNOWN")


def test_assignment_rejects_the_first_cue_reused_for_every_case() -> None:
    cues = _cues()
    assignments = list(_assignments())
    assignments[1] = BatchOperationAssignment(
        case=2, proposition_key=(901,), operation_key=(101,), operation_cue=cues[(100,)]
    )
    with pytest.raises(BatchOperationScheduleError, match="cue does not match"):
        validate_batch_operation_assignments(assignments, cues)


def test_assignment_rejects_missing_operation_coverage() -> None:
    with pytest.raises(BatchOperationScheduleError, match="does not cover every"):
        validate_batch_operation_assignments(_assignments()[:-1], _cues())


def test_result_contract_rejects_runner_that_consumed_the_first_operation() -> None:
    assignments = _assignments()
    results = list(_results(assignments))
    results[1]["operation_key"] = [100]
    results[1]["operation_keys_inferred"] = [[100]]
    results[1]["operation_keys_consumed"] = [[100]]
    with pytest.raises(BatchOperationScheduleError, match="result operation key drift"):
        validate_batch_operation_results(assignments, results)


def test_result_contract_preserves_unknown_semantic_qualification() -> None:
    audited = validate_batch_operation_results(_assignments(), _results(_assignments()))
    assert len(audited) == 8
    assert all(item["operation_assignment_verified"] == 1 for item in audited)
    assert all(item["semantic_behavior"].startswith("UNKNOWN") for item in audited)


def test_builder_preserves_explicit_case_operations_including_nonrotating_order() -> None:
    propositions = tuple((900 + i,) for i in range(10))
    operations = ((103,), (103,), (107,), (101,), (106,),
                  (100,), (104,), (105,), (102,), (107,))
    assignments = build_batch_operation_assignments(
        propositions, _cues(), operation_keys=operations)
    assert tuple(item.case for item in assignments) == tuple(range(1, 11))
    assert tuple(item.operation_key for item in assignments) == operations
    audit = validate_batch_operation_assignments(assignments, _cues())
    assert audit.covered_operation_count == 8


def test_builder_rejects_a_batch_that_cannot_cover_all_operations() -> None:
    with pytest.raises(BatchOperationScheduleError, match="does not cover every"):
        build_batch_operation_assignments(
            tuple((900 + i,) for i in range(7)), _cues(),
            operation_keys=tuple((100 + i,) for i in range(7)))


def test_builder_rejects_duplicate_propositions_before_worker_spawn() -> None:
    with pytest.raises(BatchOperationScheduleError, match="multiple cases"):
        build_batch_operation_assignments(((900,), (900,)) + tuple(
            (901 + i,) for i in range(7)), _cues(),
            operation_keys=tuple((100 + i % 8,) for i in range(9)))


@pytest.mark.parametrize("operations", [(), ((100,),), ((100,),) * 7,
                                          ((100,),) * 8, ((999,),) * 8])
def test_builder_rejects_incomplete_unknown_or_first_cue_assignments(operations) -> None:
    with pytest.raises(BatchOperationScheduleError):
        build_batch_operation_assignments(
            tuple((900 + i,) for i in range(8)), _cues(), operation_keys=operations)


def test_schedule_manifest_round_trips_without_semantic_claim() -> None:
    assignments = _assignments()
    manifest = build_batch_operation_schedule_manifest(
        identity="run-a", assignments=assignments, operation_cues=_cues())
    audit = validate_batch_operation_schedule_manifest(manifest, identity="run-a")
    assert audit.assignments == assignments
    assert manifest["behavior_qualification"].startswith("UNKNOWN")
    assert manifest["free_dialogue_complete"] == 0


@pytest.mark.parametrize("field,new_value", [
    ("identity", "run-b"),
    ("known_operation_count", 9),
    ("covered_operation_count", 7),
    ("free_dialogue_complete", True),
    ("behavior_qualification", "PASS"),
])
def test_schedule_manifest_rejects_identity_summary_or_status_drift(
        field: str, new_value: object) -> None:
    manifest = build_batch_operation_schedule_manifest(
        identity="run-a", assignments=_assignments(), operation_cues=_cues())
    manifest[field] = new_value
    with pytest.raises(BatchOperationScheduleError):
        validate_batch_operation_schedule_manifest(manifest, identity="run-a")


def test_schedule_manifest_rejects_cue_or_assignment_mutation() -> None:
    manifest = build_batch_operation_schedule_manifest(
        identity="run-a", assignments=_assignments(), operation_cues=_cues())
    manifest["operation_cues"][0]["operation_cue"] = [999]
    with pytest.raises(BatchOperationScheduleError, match="cue does not match"):
        validate_batch_operation_schedule_manifest(manifest, identity="run-a")
