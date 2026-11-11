"""The phase contract records integer evidence and stays fail-closed."""
import pytest

from pure_integer_ai.experiments.batch_phase_evidence import (
    PHASES, BatchPhaseEvidenceError, BatchPhaseRecorder, PhaseEvidence,
)


class Counter:
    def __init__(self, *values):
        self.values = iter(values)

    def __call__(self):
        return next(self.values)


def test_records_wall_cpu_and_monotonic_resource_snapshots():
    wall = Counter(10, 22)
    cpu = Counter(100, 106)
    resources = iter((
        {"working_set_bytes": 3, "process_peak_working_set_bytes": 4},
        {"working_set_bytes": 5, "process_peak_working_set_bytes": 8},
    ))
    recorder = BatchPhaseRecorder(clock=wall, cpu_clock=cpu,
                                  resource_snapshot=lambda: next(resources),
                                  required_phases=("query",))
    with recorder.phase("query"):
        pass
    report = recorder.report()
    row = report["phases"][0]
    assert row["elapsed_ns"] == 12
    assert row["cpu_elapsed_ns"] == 6
    assert row["working_set_started_bytes"] == 3
    assert row["working_set_ended_bytes"] == 5
    assert row["process_peak_started_bytes"] == 4
    assert row["process_peak_ended_bytes"] == 8
    assert report["status"] == "PASS"
    assert report["cpu_qualification"] == "OBSERVED"
    assert report["resource_qualification"] == "OBSERVED"
    assert report["process_peak_qualification"] == "OBSERVED"
    for field in ("phase_peak_qualification", "owner_qualification",
                  "durable_recovery_qualification", "production_chain_qualification",
                  "language_qualification", "runtime_cost_causality"):
        assert report[field].startswith("UNKNOWN")
    assert report["free_dialogue_complete"] == 0


def test_failures_are_recorded_and_open_intervals_cannot_finalize():
    recorder = BatchPhaseRecorder(clock=Counter(10, 20, 30, 40))
    with pytest.raises(RuntimeError):
        with recorder.phase("delivery"):
            raise RuntimeError("synthetic")
    assert recorder.report()["status"] == "FAIL"
    assert recorder.report()["phases"][0]["operation_failure_type"] == "RuntimeError"

    recorder.begin("query")
    with pytest.raises(BatchPhaseEvidenceError, match="open phase"):
        recorder.report()


def test_rejects_nested_and_unknown_phases():
    recorder = BatchPhaseRecorder(clock=Counter(10, 20, 30))
    with pytest.raises(BatchPhaseEvidenceError, match="nested"):
        recorder.begin("query")
        recorder.begin("delivery")
    recorder.end()
    with pytest.raises(BatchPhaseEvidenceError, match="frozen"):
        recorder.begin("not_a_phase")
    with pytest.raises(BatchPhaseEvidenceError, match="no phase"):
        recorder.end()


def phase_row(**overrides):
    values = {
        "name": "query", "ordinal": 1, "started_ns": 10, "ended_ns": 20,
        "elapsed_ns": 10, "cpu_started_ns": None, "cpu_ended_ns": None,
        "cpu_elapsed_ns": None, "working_set_started_bytes": None,
        "working_set_ended_bytes": None, "process_peak_started_bytes": None,
        "process_peak_ended_bytes": None, "status": "PASS",
    }
    values.update(overrides)
    return PhaseEvidence(**values)


@pytest.mark.parametrize("overrides", [
    {"ordinal": 0},
    {"ordinal": True},
    {"elapsed_ns": True, "started_ns": 1, "ended_ns": 2},
    {"ended_ns": 9},
    {"cpu_started_ns": True},
    {"cpu_ended_ns": -1},
    {"cpu_started_ns": 1, "cpu_ended_ns": None, "cpu_elapsed_ns": 0},
    {"cpu_started_ns": 1, "cpu_ended_ns": 2, "cpu_elapsed_ns": True},
    {"cpu_started_ns": 2, "cpu_ended_ns": 1, "cpu_elapsed_ns": 0},
    {"working_set_started_bytes": 5, "process_peak_started_bytes": 4},
    {"working_set_ended_bytes": 5, "process_peak_ended_bytes": 4},
    {"process_peak_started_bytes": 8, "process_peak_ended_bytes": 7},
    {"process_peak_ended_bytes": True},
    {"operation_failure_type": "RuntimeError"},
])
def test_phase_rows_reject_inconsistent_or_noninteger_evidence(overrides):
    with pytest.raises(BatchPhaseEvidenceError):
        phase_row(**overrides)


@pytest.mark.parametrize("snapshot", [
    {},
    {"working_set_bytes": None, "process_peak_working_set_bytes": None},
    {"peak_working_set_bytes": 99},
])
def test_provider_presence_and_ambiguous_peak_field_do_not_qualify(snapshot):
    recorder = BatchPhaseRecorder(clock=Counter(10, 20),
                                  resource_snapshot=lambda: snapshot)
    with recorder.phase("query"):
        pass
    report = recorder.report()
    assert report["status"] == "UNKNOWN"
    assert report["recorded_intervals_status"] == "PASS"
    assert report["required_phase_coverage"] == "UNKNOWN"
    assert report["resource_qualification"] == "UNKNOWN"
    assert report["process_peak_qualification"] == "UNKNOWN"


def test_missing_end_resource_is_partial_and_never_filled_from_start():
    resources = iter((
        {"working_set_bytes": 3, "process_peak_working_set_bytes": 4}, {},
    ))
    recorder = BatchPhaseRecorder(clock=Counter(10, 20),
                                  resource_snapshot=lambda: next(resources))
    with recorder.phase("query"):
        pass
    report = recorder.report()
    assert report["resource_qualification"] == "PARTIAL"
    assert report["process_peak_qualification"] == "PARTIAL"
    assert report["working_set_endpoint_count"] == 1
    assert report["phases"][0]["process_peak_ended_bytes"] is None


def test_required_stage_coverage_is_checked_but_does_not_assert_call_sites():
    recorder = BatchPhaseRecorder(clock=Counter(10, 20), required_phases=PHASES)
    with recorder.phase("query"):
        pass
    report = recorder.report()
    assert report["status"] == "FAIL"
    assert report["recorded_intervals_status"] == "PASS"
    assert report["required_phase_coverage"] == "FAIL"
    assert report["missing_required_phase_names"] == [name for name in PHASES if name != "query"]
    assert report["production_chain_qualification"].startswith("UNKNOWN")


def test_empty_report_has_no_success_or_observed_measurements():
    report = BatchPhaseRecorder(cpu_clock=Counter(), resource_snapshot=lambda: {}).report()
    assert report["status"] == "UNKNOWN"
    assert report["phase_count"] == 0
    assert report["cpu_qualification"] == "UNKNOWN"
    assert report["resource_qualification"] == "UNKNOWN"
    assert report["process_peak_qualification"] == "UNKNOWN"


def test_collection_failure_does_not_mask_original_operation_error():
    calls = 0

    def resources():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise LookupError("endpoint unavailable")
        return {}

    recorder = BatchPhaseRecorder(clock=Counter(10, 20), resource_snapshot=resources)
    original = RuntimeError("operation failed")
    with pytest.raises(RuntimeError) as raised:
        with recorder.phase("delivery"):
            raise original
    assert raised.value is original
    assert "LookupError" in original.__notes__[0]
    report = recorder.report()
    assert report["status"] == "FAIL"
    assert report["phase_count"] == 0
    failure = report["collection_failures"][0]
    assert failure["boundary"] == "end"
    assert failure["error_type"] == "LookupError"
    assert failure["operation_failure_type"] == "RuntimeError"
    assert failure["available_samples"]["ended_ns"] == 20
    assert calls == 2


def test_begin_collection_failure_is_retained_without_leaving_open_phase():
    recorder = BatchPhaseRecorder(clock=Counter(10), cpu_clock=Counter(True))
    with pytest.raises(BatchPhaseEvidenceError, match="cpu_started_ns"):
        recorder.begin("query")
    report = recorder.report()
    assert report["status"] == "FAIL"
    failure = report["collection_failures"][0]
    assert failure["boundary"] == "begin"
    assert failure["available_samples"]["started_ns"] == 10
    with pytest.raises(BatchPhaseEvidenceError, match="no phase"):
        recorder.end()


@pytest.mark.parametrize("end", [10, 9])
def test_nonadvancing_clock_fails_and_preserves_unfinished_attempt(end):
    recorder = BatchPhaseRecorder(clock=Counter(10, end))
    recorder.begin("query")
    with pytest.raises(BatchPhaseEvidenceError, match="did not advance"):
        recorder.end()
    report = recorder.report()
    assert report["status"] == "FAIL"
    assert report["phase_count"] == 0
    assert report["collection_failures"][0]["available_samples"]["ended_ns"] == end


def test_clock_cannot_reverse_between_valid_intervals():
    recorder = BatchPhaseRecorder(clock=Counter(10, 20, 19))
    with recorder.phase("query"):
        pass
    with pytest.raises(BatchPhaseEvidenceError, match="between intervals"):
        recorder.begin("delivery")
    assert recorder.report()["status"] == "FAIL"


@pytest.mark.parametrize("required", [(), ("query", "query"), ("not_a_phase",), ["query"]])
def test_rejects_invalid_required_stage_contract(required):
    with pytest.raises(BatchPhaseEvidenceError):
        BatchPhaseRecorder(required_phases=required)


def test_zero_cpu_elapsed_is_valid_and_partial_cpu_endpoint_is_preserved():
    row = phase_row(cpu_started_ns=2, cpu_ended_ns=2, cpu_elapsed_ns=0)
    assert row.cpu_elapsed_ns == 0
    row = phase_row(cpu_started_ns=2)
    assert row.cpu_started_ns == 2
    assert row.cpu_elapsed_ns is None


@pytest.mark.parametrize("resources", [
    ({"working_set_bytes": 5, "process_peak_working_set_bytes": 4}, {}),
    ({"process_peak_working_set_bytes": 8}, {"process_peak_working_set_bytes": 7}),
])
def test_resource_provider_cannot_hide_invalid_peak_by_taking_max(resources):
    snapshots = iter(resources)
    recorder = BatchPhaseRecorder(clock=Counter(10, 20),
                                  resource_snapshot=lambda: next(snapshots))
    with pytest.raises(BatchPhaseEvidenceError, match="peak"):
        with recorder.phase("query"):
            pass
    assert recorder.report()["status"] == "FAIL"


def test_report_failure_samples_cannot_be_mutated_by_consumer():
    recorder = BatchPhaseRecorder(clock=Counter(10, 10))
    recorder.begin("query")
    with pytest.raises(BatchPhaseEvidenceError):
        recorder.end()
    report = recorder.report()
    report["collection_failures"][0]["available_samples"]["started_ns"] = 999
    assert recorder.report()["collection_failures"][0]["available_samples"]["started_ns"] == 10
