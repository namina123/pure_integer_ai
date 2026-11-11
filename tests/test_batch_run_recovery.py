"""Resume safety checks; these never open a training or Memory database."""
from __future__ import annotations

import os

import pytest

from pure_integer_ai.experiments.batch_run_recovery import (
    ProcessProbe,
    probe_process,
    require_inactive_run_owners,
)


def test_real_process_probe_leaves_the_current_process_alive() -> None:
    result = probe_process(os.getpid())
    assert result.pid == os.getpid()
    assert result.status == "LIVE"
    if os.name == "nt":
        assert result.creation_epoch_ns is not None


@pytest.mark.parametrize("live_pid", [101, 102])
def test_either_live_owner_refuses_duplicate_resume(live_pid: int) -> None:
    prior = {"status": "CASE_WORKER_RUNNING", "pid": 101, "worker_pid": 102}

    def probe(pid: int) -> ProcessProbe:
        return ProcessProbe(pid, "LIVE" if pid == live_pid else "EXITED")

    with pytest.raises(RuntimeError, match="still active"):
        require_inactive_run_owners(prior, probe=probe)


def test_unknown_owner_cannot_authorize_resume() -> None:
    prior = {"status": "CASE_RUNNING", "pid": 101}
    with pytest.raises(RuntimeError, match="liveness is unknown"):
        require_inactive_run_owners(
            prior, probe=lambda pid: ProcessProbe(pid, "UNKNOWN"))


def test_resume_requires_both_owners_to_have_exited() -> None:
    prior = {"status": "CASE_WORKER_RUNNING", "pid": 101, "worker_pid": 102}
    evidence = require_inactive_run_owners(
        prior, probe=lambda pid: ProcessProbe(pid, "EXITED"))
    assert tuple(item.pid for item in evidence) == (101, 102)


@pytest.mark.parametrize("pid", [None, True, 0, -1, 0x100000000])
def test_missing_or_invalid_owner_prevents_resume(pid: object) -> None:
    prior = {"status": "CASE_RUNNING", "pid": pid}
    with pytest.raises(RuntimeError, match="is unknown"):
        require_inactive_run_owners(prior)


def test_mismatched_process_probe_is_rejected() -> None:
    prior = {"status": "CASE_RUNNING", "pid": 101}
    with pytest.raises(RuntimeError, match="identity/status drift"):
        require_inactive_run_owners(
            prior, probe=lambda pid: ProcessProbe(pid + 1, "EXITED"))
