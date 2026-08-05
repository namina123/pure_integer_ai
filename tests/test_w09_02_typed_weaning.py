"""W09-02 typed teacher-exit runtime 专项。"""
from dataclasses import dataclass
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_w09_contract import (
    make_w09_request,
    open_w09_frozen_contract,
)
from pure_integer_ai.experiments.ph2_w09_firewall import W09PayloadFirewall
from pure_integer_ai.experiments.ph2_w09_types import (
    TeacherExitPhase,
    W09ResourceAudit,
    W09WindowIdentity,
)
from pure_integer_ai.experiments.ph2_w09_weaning import (
    W09TypedWeaningRuntime,
    W09FrozenTeacherEvidenceSource,
    W09WeaningError,
    W09_ZERO_CALL_WINDOWS_PENDING,
    make_w09_typed_weaning_protocol_from_contract,
    w09_commitment,
)


@dataclass
class _Context:
    backend: object
    teacher: object


class _Backend:
    def __init__(self):
        self.value = 0

    def snapshot(self):
        return {"value": self.value}


class _Teacher:
    call_count = 0


class _Source:
    def __init__(self, records):
        self.records = records

    def read(self, _ctx, _report):
        return self.records


class _Dev:
    def calibrate(self, _ctx, _report):
        return ("dev-record",)


class _Shadow:
    def __init__(self):
        self.count = 0

    def record(self, _ctx, _errors):
        self.count += 1
        return 1


class _Report:
    complete = True
    outcomes = ("shadow-error",)

    def stable_key(self):
        return (7, 8, 9)


def _runtime(tmp_path):
    del tmp_path
    context = open_w09_frozen_contract(Path(__file__).parents[1])
    training = ("frozen-teacher-evidence",)
    protocol = make_w09_typed_weaning_protocol_from_contract(
        context,
        candidate_identity=w09_commitment("candidate"),
        input_commitment=w09_commitment(training),
        threshold_key=(1, 2, 3),
    )
    backend = _Backend()
    shadow = _Shadow()
    runtime = W09TypedWeaningRuntime(
        protocol,
        training_material_source=_Source(training),
        dev_calibrator=_Dev(),
        shadow_auditor=shadow,
        frozen_contract=context,
    )
    return runtime, protocol, _Context(backend, _Teacher())


def test_three_pre_window_phases_are_real_and_zero_windows_are_pending(tmp_path):
    runtime, protocol, ctx = _runtime(tmp_path)
    report = runtime.run(ctx, _Report())
    assert report.complete is False
    assert report.blockers == (W09_ZERO_CALL_WINDOWS_PENDING,)
    assert tuple(item.phase for item in report.phase_audits) == (
        TeacherExitPhase.TRAINING_MATERIAL_SOURCE,
        TeacherExitPhase.DEV_CALIBRATION_ONLY,
        TeacherExitPhase.SHADOW_ERROR_ONLY,
    )
    assert all(item.teacher_call_count == 0 for item in report.phase_audits)
    assert all(item.host_write_count == 0 for item in report.phase_audits)

    for ordinal, input_commitment in enumerate(protocol.window_input_commitments, 1):
        identity = W09WindowIdentity(
            TeacherExitPhase.ZERO_CALL_WINDOW,
            ordinal,
            input_commitment,
            protocol.candidate_identity,
            0,
            tuple((key, w09_commitment((key, ordinal)))
                  for key in ("UNDERSTANDING", "REASONING", "GENERATION")),
            W09ResourceAudit.zero(),
            w09_commitment(("rollback", ordinal)),
        )
        runtime.execute_zero_call_window(ctx, identity)
    assert runtime.run(ctx, _Report()).complete is True


def test_window_input_drift_and_reuse_fail_closed(tmp_path):
    runtime, protocol, ctx = _runtime(tmp_path)
    runtime.run(ctx, _Report())
    identity = W09WindowIdentity(
        TeacherExitPhase.ZERO_CALL_WINDOW,
        1,
        w09_commitment("wrong-input"),
        protocol.candidate_identity,
        0,
        tuple((key, w09_commitment((key, 1)))
              for key in ("UNDERSTANDING", "REASONING", "GENERATION")),
        W09ResourceAudit.zero(),
        w09_commitment(("rollback", 1)),
    )
    with pytest.raises(W09WeaningError):
        runtime.execute_zero_call_window(ctx, identity)

    identity = W09WindowIdentity(
        TeacherExitPhase.ZERO_CALL_WINDOW,
        1,
        protocol.window_input_commitments[0],
        protocol.candidate_identity,
        0,
        tuple((key, w09_commitment((key, 1)))
              for key in ("UNDERSTANDING", "REASONING", "GENERATION")),
        W09ResourceAudit.zero(),
        w09_commitment(("rollback", 1)),
    )
    runtime.execute_zero_call_window(ctx, identity)
    with pytest.raises(W09WeaningError):
        runtime.execute_zero_call_window(ctx, identity)


def test_dev_and_shadow_host_or_output_write_is_rejected(tmp_path):
    runtime, _, ctx = _runtime(tmp_path)

    class _BadDev(_Dev):
        def calibrate(self, current, report):
            current.backend.value += 1
            return super().calibrate(current, report)

    runtime.dev_calibrator = _BadDev()
    with pytest.raises(W09WeaningError):
        runtime.run(ctx, _Report())

    runtime, _, ctx = _runtime(tmp_path)

    class _BadShadow(_Shadow):
        def record(self, current, errors):
            current.backend.value += 1
            return super().record(current, errors)

    runtime.shadow_auditor = _BadShadow()
    with pytest.raises(W09WeaningError):
        runtime.run(ctx, _Report())


def test_zero_call_window_teacher_call_is_rejected(tmp_path):
    runtime, protocol, ctx = _runtime(tmp_path)
    runtime.run(ctx, _Report())

    def calls_teacher():
        ctx.teacher.call_count += 1

    identity = W09WindowIdentity(
        TeacherExitPhase.ZERO_CALL_WINDOW,
        1,
        protocol.window_input_commitments[0],
        protocol.candidate_identity,
        0,
        tuple((key, w09_commitment((key, 1)))
              for key in ("UNDERSTANDING", "REASONING", "GENERATION")),
        W09ResourceAudit.zero(),
        w09_commitment(("rollback", 1)),
    )
    with pytest.raises(W09WeaningError):
        runtime.execute_zero_call_window(ctx, identity, calls_teacher)


def test_frozen_teacher_evidence_source_accepts_teacher_owned_expected_payload(tmp_path):
    del tmp_path
    root = Path(__file__).parents[1]
    context = open_w09_frozen_contract(root)
    firewall = W09PayloadFirewall.open(root, context, make_w09_request(context))
    payload = firewall.read_training_payload()
    source = W09FrozenTeacherEvidenceSource(context, payload)
    records = source.read(None, None)
    assert len(records) == 309
    assert source.read_count == 1
    assert source.state_key()
