"""Actual current-process counters and strict missing/invalid endpoint handling."""
import os

import pytest

from pure_integer_ai.experiments.batch_phase_evidence import BatchPhaseEvidenceError
from pure_integer_ai.experiments.batch_phase_host import (
    ProcessResourceEndpoint, current_process_resource_endpoint,
)
from pure_integer_ai.experiments.batch_run_recovery import probe_process


def endpoint(**overrides):
    values = dict(pid=17, creation_epoch_ns=100, sampled_started_ns=10,
                  sampled_ended_ns=11, sampled_epoch_ns=1000, cpu_ns=5,
                  working_set_bytes=20, process_peak_working_set_bytes=30)
    values.update(overrides)
    return ProcessResourceEndpoint(**values)


@pytest.mark.parametrize("overrides", [
    {"pid": True}, {"pid": 0}, {"pid": 0x100000000},
    {"creation_epoch_ns": None}, {"sampled_epoch_ns": True},
    {"sampled_started_ns": -1}, {"sampled_ended_ns": 9},
    {"cpu_ns": True}, {"working_set_bytes": -1},
    {"process_peak_working_set_bytes": True},
    {"process_peak_working_set_bytes": 19},
    {"memory_error": ""}, {"memory_error": "denied"},
])
def test_endpoint_rejects_invalid_identity_counters_and_memory_errors(overrides):
    with pytest.raises(BatchPhaseEvidenceError):
        endpoint(**overrides)


def test_missing_memory_retains_error_without_fake_zero_counters():
    value = endpoint(working_set_bytes=None, process_peak_working_set_bytes=None,
                     memory_error="GetProcessMemoryInfo_error_5")
    exported = value.to_value()
    assert exported["working_set_bytes"] is None
    assert exported["process_peak_working_set_bytes"] is None
    assert exported["memory_error"].endswith("_5")
    exported["pid"] = 99
    assert value.pid == 17


@pytest.mark.skipif(os.name != "nt", reason="Win32 counters")
def test_real_windows_sampler_binds_current_process_creation_and_separate_counters():
    first = current_process_resource_endpoint()
    sum(range(10000))
    second = current_process_resource_endpoint()
    probe = probe_process(os.getpid())
    assert first.pid == second.pid == os.getpid()
    assert probe.status == "LIVE"
    assert first.creation_epoch_ns == second.creation_epoch_ns == probe.creation_epoch_ns
    assert second.sampled_started_ns >= first.sampled_ended_ns
    assert second.cpu_ns >= first.cpu_ns
    assert first.working_set_bytes > 0
    assert first.process_peak_working_set_bytes >= first.working_set_bytes
    assert second.process_peak_working_set_bytes >= first.process_peak_working_set_bytes
    assert first.memory_error is None
