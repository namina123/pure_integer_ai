"""Crash/reopen, corruption and failure boundaries for the independent journal."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from pure_integer_ai.experiments.batch_phase_evidence import BatchPhaseEvidenceError
from pure_integer_ai.experiments.batch_phase_host import ProcessResourceEndpoint
from pure_integer_ai.experiments.batch_phase_journal import (
    DurableBatchPhaseRecorder, audit_phase_journal,
)
import pure_integer_ai.experiments.batch_phase_journal as journal_module


CONTEXT = dict(run_identity="contract-only", attempt_key="one", case_ordinal=2)


def endpoint(wall, cpu, **overrides):
    values = dict(pid=17, creation_epoch_ns=100, sampled_started_ns=wall,
                  sampled_ended_ns=wall + 1, sampled_epoch_ns=1000 + wall,
                  cpu_ns=cpu, working_set_bytes=20, process_peak_working_set_bytes=30)
    values.update(overrides)
    return ProcessResourceEndpoint(**values)


def recorder(tmp_path, *samples, required=("query",)):
    values = iter(samples)
    return DurableBatchPhaseRecorder(tmp_path / "attempt", **CONTEXT,
                                     required_phases=required, sampler=lambda: next(values))


def audit(path, **overrides):
    kwargs = dict(**CONTEXT, required_phases=("query",))
    kwargs.update(overrides)
    return audit_phase_journal(path, **kwargs)


def test_begin_is_visible_before_body_and_end_reopens_without_application_calls(tmp_path):
    value = recorder(tmp_path, endpoint(10, 1), endpoint(30, 5))
    with value.phase("query"):
        paths = list(value.directory.iterdir())
        assert len(paths) == 1
        assert json.loads(paths[0].read_bytes())["event"] == "BEGIN"
        in_progress = audit(value.directory)
        assert in_progress["status"] == "UNKNOWN"
        assert in_progress["incomplete_boundary"]["phase"] == "query"
    result = audit(value.directory)
    assert result["status"] == "PASS"
    assert result["event_count"] == 2
    assert result["completed_intervals"][0]["cpu_elapsed_ns"] == 4
    assert result["owner_identity"] == {"pid": 17, "creation_epoch_ns": 100}
    assert result["incomplete_boundary"] is None
    for field in ("application_recovery_qualification", "phase_peak_qualification",
                  "production_chain_qualification", "runtime_cost_causality", "language_qualification"):
        assert result[field].startswith("UNKNOWN")
    assert result["free_dialogue_complete"] == 0
    assert value.report()["in_memory_intervals"]["cpu_qualification"] == "OBSERVED"


def test_does_not_reuse_existing_attempt_or_invent_missing_phase_coverage(tmp_path):
    value = recorder(tmp_path, endpoint(10, 1), endpoint(30, 5), required=("query", "delivery"))
    with value.phase("query"):
        pass
    assert value.report()["journal"]["status"] == "FAIL"
    assert value.report()["journal"]["missing_required_phase_names"] == ["delivery"]
    before = {path.name: path.read_bytes() for path in value.directory.iterdir()}
    with pytest.raises(FileExistsError):
        recorder(tmp_path, endpoint(50, 8))
    assert before == {path.name: path.read_bytes() for path in value.directory.iterdir()}


def test_missing_resource_at_end_is_preserved_as_unknown_not_start_fill(tmp_path):
    value = recorder(tmp_path, endpoint(10, 1), endpoint(
        30, 5, working_set_bytes=None, process_peak_working_set_bytes=None, memory_error="denied"))
    with value.phase("query"):
        pass
    result = value.report()
    assert result["in_memory_intervals"]["resource_qualification"] == "PARTIAL"
    row = result["journal"]["completed_intervals"][0]
    assert row["working_set_ended_bytes"] is None
    assert row["process_peak_ended_bytes"] is None
    assert json.loads((value.directory / "00000002.json").read_bytes())["endpoint"]["memory_error"] == "denied"


@pytest.mark.parametrize("second", [
    endpoint(30, 5, creation_epoch_ns=101), endpoint(30, 5, pid=18),
    endpoint(9, 5), endpoint(30, 0), endpoint(30, 5, process_peak_working_set_bytes=29),
])
def test_identity_clock_cpu_and_peak_drift_fail_closed_with_original_end_attempt(tmp_path, second):
    value = recorder(tmp_path, endpoint(10, 1), second)
    value.begin("query")
    with pytest.raises(BatchPhaseEvidenceError):
        value.end()
    original = json.loads((value.directory / "00000002.json").read_bytes())
    assert original["event"] == "END_COLLECTION_FAILURE"
    assert original["endpoint"] == second.to_value()
    assert original["collection_failures"][0]["boundary"] == "end"
    assert audit(value.directory)["status"] == "FAIL"


def test_begin_collection_failure_and_later_attempt_are_both_kept(tmp_path):
    value = recorder(tmp_path, object(), endpoint(10, 1), endpoint(30, 5))
    with pytest.raises(BatchPhaseEvidenceError):
        value.begin("query")
    with value.phase("query"):
        pass
    result = value.report()["journal"]
    assert result["status"] == "FAIL"
    assert result["event_count"] == 3
    assert result["completed_intervals"][0]["ordinal"] == 2
    assert result["collection_failures"][0]["event"] == "BEGIN_COLLECTION_FAILURE"


def test_business_failure_survives_end_sampling_failure_and_is_persisted(tmp_path):
    value = recorder(tmp_path, endpoint(10, 1))
    original = RuntimeError("business failed")
    with pytest.raises(RuntimeError) as raised:
        with value.phase("query"):
            raise original
    assert raised.value is original
    assert any("StopIteration" in note for note in original.__notes__)
    result = audit(value.directory)
    assert result["status"] == "FAIL"
    assert result["collection_failures"][0]["operation_failure_type"] == "RuntimeError"
    assert result["collection_failures"][0]["endpoint"] is None


def test_business_failure_has_fail_interval_even_when_collection_succeeds(tmp_path):
    value = recorder(tmp_path, endpoint(10, 1), endpoint(30, 5))
    with pytest.raises(LookupError):
        with value.phase("query"):
            raise LookupError("failed")
    result = audit(value.directory)
    assert result["status"] == "FAIL"
    assert result["completed_intervals"][0]["operation_failure_type"] == "LookupError"


def test_fsync_failure_blocks_body_and_forbids_reusing_writer(tmp_path, monkeypatch):
    value = recorder(tmp_path, endpoint(10, 1))
    executed = False

    def fail(_):
        raise OSError("disk unavailable")

    monkeypatch.setattr(journal_module.os, "fsync", fail)
    with pytest.raises(OSError):
        with value.phase("query"):
            executed = True
    assert not executed
    assert audit(value.directory)["status"] != "PASS"
    with pytest.raises(BatchPhaseEvidenceError, match="write failed"):
        value.report()
    with pytest.raises(BatchPhaseEvidenceError, match="not reusable"):
        value.begin("query")


@pytest.mark.parametrize("change", ["torn", "digest", "identity", "bool_case", "interval", "gap", "duplicate"])
def test_original_files_are_not_repaired_when_journal_is_corrupt(tmp_path, change):
    value = recorder(tmp_path, endpoint(10, 1), endpoint(30, 5))
    with value.phase("query"):
        pass
    path = value.directory / "00000002.json"
    event = json.loads(path.read_bytes())
    if change == "torn":
        path.write_bytes(b'{"format":')
    elif change == "gap":
        path.rename(value.directory / "00000003.json")
    elif change == "duplicate":
        path.write_bytes(b'{"format":"x","format":"x"}')
    else:
        if change == "digest":
            event["previous_sha256"] = "0" * 64
        elif change == "identity":
            event["attempt_key"] = "other"
        elif change == "bool_case":
            event["case_ordinal"] = True
        elif change == "interval":
            event["interval"]["cpu_elapsed_ns"] = 0
        path.write_text(json.dumps(event), encoding="utf-8")
    before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in value.directory.iterdir()}
    assert audit(value.directory)["status"] == "FAIL"
    assert before == {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in value.directory.iterdir()}


@pytest.mark.parametrize("field,new", [("run_identity", "other"), ("attempt_key", "other"), ("case_ordinal", 3)])
def test_reader_requires_exact_run_attempt_and_case_binding(tmp_path, field, new):
    value = recorder(tmp_path, endpoint(10, 1), endpoint(30, 5))
    with value.phase("query"):
        pass
    assert audit(value.directory, **{field: new})["status"] == "FAIL"


def test_empty_or_missing_directory_does_not_qualify(tmp_path):
    assert audit(tmp_path)["status"] == "UNKNOWN"
    assert audit(tmp_path / "absent")["status"] == "FAIL"
    assert not (tmp_path / "absent").exists()


@pytest.mark.parametrize("required", [None, (), ["query"], ("query", "query"), ("unknown",)])
def test_invalid_frozen_coverage_is_refused_before_creating_attempt(tmp_path, required):
    with pytest.raises(BatchPhaseEvidenceError):
        recorder(tmp_path, required=required)
    assert not (tmp_path / "attempt").exists()


def test_real_stage_labels_require_independent_intervals_and_missing_labels_fail(tmp_path):
    required = ("memory_init", "memory_recover", "core", "response_plan", "generation", "memory_commit")
    samples = tuple(endpoint(10 + offset * 20, 1 + offset) for offset in range(12))
    value = recorder(tmp_path, *samples, required=required)
    for name in required:
        with value.phase(name):
            pass
    result = value.report()["journal"]
    assert result["status"] == "PASS"
    assert [row["name"] for row in result["completed_intervals"]] == list(required)
    assert result["production_chain_qualification"].startswith("UNKNOWN")


@pytest.mark.skipif(os.name != "nt", reason="real Windows abrupt-exit endpoint")
def test_abrupt_process_exit_leaves_fsynced_begin_readable_in_fresh_cpython(tmp_path):
    directory = tmp_path / "abrupt"
    source_root = str(Path(__file__).resolve().parents[1] / "src")
    script = (
        "import os, sys; from pathlib import Path; "
        "from pure_integer_ai.experiments.batch_phase_journal import DurableBatchPhaseRecorder; "
        "value = DurableBatchPhaseRecorder(Path(sys.argv[1]), run_identity='contract-only', "
        "attempt_key='one', case_ordinal=2, required_phases=('query',)); "
        "value.begin('query'); os._exit(23)"
    )
    env = dict(os.environ, PYTHONPATH=source_root, PYTHONDONTWRITEBYTECODE="1")
    child = subprocess.run([sys.executable, "-c", script, str(directory)], env=env,
                           capture_output=True, text=True, timeout=20)
    assert child.returncode == 23, child.stderr
    read_script = (
        "import json, sys; from pathlib import Path; "
        "from pure_integer_ai.experiments.batch_phase_journal import audit_phase_journal; "
        "print(json.dumps(audit_phase_journal(Path(sys.argv[1]), run_identity='contract-only', "
        "attempt_key='one', case_ordinal=2, required_phases=('query',))))"
    )
    reader = subprocess.run([sys.executable, "-c", read_script, str(directory)], env=env,
                            capture_output=True, text=True, timeout=20)
    assert reader.returncode == 0, reader.stderr
    result = json.loads(reader.stdout)
    assert result["status"] == "UNKNOWN"
    assert result["event_count"] == 1
    assert result["incomplete_boundary"]["phase"] == "query"
    assert result["owner_identity"]["creation_epoch_ns"] > 0
    assert result["application_recovery_qualification"].startswith("UNKNOWN")
