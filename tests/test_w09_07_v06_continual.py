"""W09-07 V-06 隔离 clone、三零调用窗口和后续学习专项。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_d03_lc16_overlay_specs import SCOPE_KEYS
from pure_integer_ai.experiments.ph2_w09_authority import (
    W09_CARRIER_KEYS,
    W09_CONSUMER_KEYS,
)
from pure_integer_ai.experiments.ph2_w09_contract import (
    make_w09_request,
    open_w09_frozen_contract,
)
from pure_integer_ai.experiments.ph2_w09_cumulative import (
    open_w09_cumulative_runtime,
)
from pure_integer_ai.experiments.ph2_w09_dimensional import (
    open_w09_dimensional_runtime,
)
from pure_integer_ai.experiments.ph2_w09_firewall import W09PayloadFirewall
from pure_integer_ai.experiments.ph2_w09_types import (
    W09ConsumerChoice,
    W09ConsumerRequest,
    W09DirectionalResult,
    W09ResultState,
    W09UseOutcome,
    W09VerifierResult,
)
from pure_integer_ai.experiments.ph2_w09_v06_contract import (
    W09V06Error,
    W09V06HostSnapshot,
    W09V06LearningRecord,
    W09V06Probe,
    W09V06ProbeCriterion,
    W09V06ProbeOwner,
    W09V06Protocol,
    W09V06WindowPlan,
    W09_V06_ABLATION_KEYS,
    w09_v06_commitment,
    w09_v06_key,
)
from pure_integer_ai.experiments.ph2_w09_v06_runtime import (
    W09V06HostCandidate,
    W09V06Runtime,
    record_w09_v06_continual_cells,
)
from pure_integer_ai.experiments.ph2_w09_weaning import (
    W09TypedWeaningRuntime,
    make_w09_typed_weaning_protocol,
    w09_commitment,
)


ROOT = Path(__file__).parents[1]


def _key(value: object) -> tuple[int, ...]:
    """为 fixture 生成域分离的稳定 32-byte identity。"""
    return w09_v06_key(value)


def _host() -> W09V06HostCandidate:
    """构造覆盖 Core/Memory/Use/Evidence/assessment/cursor/report 的 host。"""
    snapshot = W09V06HostSnapshot(
        _key(("host", "core")),
        _key(("host", "memory")),
        _key(("host", "use")),
        _key(("host", "evidence")),
        _key(("host", "assessment")),
        _key(("host", "cursor")),
        _key(("host", "report")),
        41,
        0,
        0,
        0,
    )
    return W09V06HostCandidate(
        w09_v06_commitment(("W09-07", "candidate")),
        snapshot,
    )


def _window(ordinal: int, carriers: tuple[str, ...]) -> W09V06WindowPlan:
    """构造九条 good/error Evidence 与九条未参与学习 probe。"""
    probes = []
    good_records = []
    error_records = []
    for carrier in carriers:
        for consumer in W09_CONSUMER_KEYS:
            cell = (ordinal, carrier, consumer)
            family_key = _key(("family", *cell))
            good_candidate = _key(("candidate", "good", *cell))
            bad_candidate = _key(("candidate", "bad", *cell))
            baseline_candidate = _key(("candidate", "baseline", *cell))
            criterion = W09V06ProbeCriterion(
                _key(("criterion", *cell)),
                (good_candidate,),
            )
            probes.append(W09V06Probe(
                ordinal,
                carrier,
                consumer,
                family_key,
                _key(("probe", *cell)),
                _key(("probe-content", *cell)),
                baseline_candidate,
                criterion,
            ))
            good_records.append(W09V06LearningRecord(
                ordinal,
                carrier,
                consumer,
                family_key,
                _key(("learning-content", "good", *cell)),
                _key(("source", "good", *cell)),
                _key(("evidence", "good", *cell)),
                good_candidate,
                10,
            ))
            error_records.append(W09V06LearningRecord(
                ordinal,
                carrier,
                consumer,
                family_key,
                _key(("learning-content", "bad", *cell)),
                _key(("source", "bad", *cell)),
                _key(("evidence", "bad", *cell)),
                bad_candidate,
                20,
            ))
    owner = W09V06ProbeOwner(
        _key(("probe-owner", ordinal)),
        tuple(probes),
    )
    return W09V06WindowPlan(
        ordinal,
        _key(("threshold", ordinal)),
        5,
        tuple(good_records),
        tuple(error_records),
        owner,
    )


def _protocol(host: W09V06HostCandidate) -> W09V06Protocol:
    """把九 carrier 严格分到三个不交叠连续窗口。"""
    return W09V06Protocol(
        host.candidate_identity,
        host.snapshot().core_state_key,
        (
            _window(1, W09_CARRIER_KEYS[0:3]),
            _window(2, W09_CARRIER_KEYS[3:6]),
            _window(3, W09_CARRIER_KEYS[6:9]),
        ),
    )


def _directional(label: object, consumer: str) -> W09DirectionalResult:
    """为七个历史 scope 构造与 V06 identity 不共享的当前 retention result。"""
    request_key = _key(("retention", label, "request"))
    choice_key = _key(("retention", label, "choice"))
    candidate_key = _key(("retention", label, "candidate"))
    use_key = _key(("retention", label, "use"))
    outcome_key = _key(("retention", label, "outcome"))
    return W09DirectionalResult(
        W09ConsumerRequest(
            consumer,
            request_key,
            w09_v06_commitment(("retention", label, "input")),
        ),
        W09ConsumerChoice(
            consumer,
            request_key,
            choice_key,
            candidate_key,
        ),
        W09UseOutcome(
            consumer,
            request_key,
            choice_key,
            candidate_key,
            use_key,
            outcome_key,
            "RESOLVED",
        ),
        W09VerifierResult(
            consumer,
            request_key,
            use_key,
            outcome_key,
            _key(("retention", label, "verifier")),
            W09ResultState.PASS,
            "NONE",
        ),
    )


def _dimensional_runtime():
    """打开真实 W09 authority/firewall/cumulative/dimensional public runtime。"""
    context = open_w09_frozen_contract(ROOT)
    payload = W09PayloadFirewall.open(
        ROOT,
        context,
        make_w09_request(context),
    ).read_training_payload()
    cumulative = open_w09_cumulative_runtime(ROOT, context)
    cumulative.ingest_training_payload(payload)
    for carrier in W09_CARRIER_KEYS:
        for consumer in W09_CONSUMER_KEYS:
            cumulative.consume_directional(
                carrier,
                consumer,
                _directional(("cumulative", carrier, consumer), consumer),
            )
    return open_w09_dimensional_runtime(ROOT, cumulative)


def test_one_clone_three_windows_improve_unlearned_probes_and_rollback_errors():
    """三窗口必须各自 0->1000，错误采用 9->0，host/Core 和正式状态不变。"""
    host = _host()
    host_before = host.snapshot()
    runtime = W09V06Runtime(host, _protocol(host))
    report = runtime.run()

    assert report.clone_count == 1
    assert report.host_before == report.host_after == host_before
    assert report.clone_receipt.source_write_count == 0
    assert report.clone_receipt.isolated_learning_write_count > 0
    assert report.clone_receipt.pre_learning_output_identity != (
        report.clone_receipt.post_learning_output_identity)
    assert len(report.windows) == 3
    assert len(report.cells) == 27
    assert report.consumer_difference_count == 27
    assert report.directional_difference_counts == (
        ("UNDERSTANDING", 9),
        ("REASONING", 9),
        ("GENERATION", 9),
    )
    assert tuple(item.cell_key for item in report.cells) == tuple(
        (carrier, consumer)
        for carrier in W09_CARRIER_KEYS
        for consumer in W09_CONSUMER_KEYS
    )
    for window in report.windows:
        assert (window.metric_before.permille, window.metric_after.permille) == (0, 1000)
        assert (
            window.erroneous_adoption_before_rollback,
            window.erroneous_adoption_after_rollback,
        ) == (9, 0)
        assert window.clone_core_before == window.clone_core_after == host_before.core_state_key
        assert window.memory_write_count == 18
        assert window.evidence_write_count == 18
        assert window.rollback_event_count == 9
        assert window.teacher_call_count == 0
        assert window.api_call_count == 0
        assert window.llm_call_count == 0
        assert window.host_write_count == 0
    assert report.public_status == "PUBLIC_BOUNDED_NOT_FORMAL"
    assert report.pre_wean_language_learning_capability_evidenced == 0
    assert report.language_capability_mastered == 0
    assert report.language_readiness == 0
    assert report.stable_key()


def test_actual_v06_cells_replace_placeholder_continual_evidence_in_jlc():
    """27 个 V06 实际 result/delta 接入 216-cell J-LC，而非任意 before/after 占位。"""
    host = _host()
    v06_report = W09V06Runtime(host, _protocol(host)).run()
    dimensional = _dimensional_runtime()
    for scope in SCOPE_KEYS:
        if scope == "RETENTION_CONTINUAL_LEARNING":
            continue
        for carrier in W09_CARRIER_KEYS:
            for consumer in W09_CONSUMER_KEYS:
                dimensional.record_cell(
                    scope,
                    carrier,
                    consumer,
                    _directional((scope, carrier, consumer), consumer),
                )
    record_w09_v06_continual_cells(dimensional, v06_report)
    report = dimensional.report()
    assert report.retention_cell_count == 189
    assert report.continual_learning_cell_count == 27
    assert report.dimensional_status == "PUBLIC_BOUNDED_PASS"
    assert report.j_lc_w09_state == "PUBLIC_BOUNDED_NOT_FORMAL"
    assert report.language_capability_mastered == 0
    assert report.language_readiness == 0


class _Source:
    """给 typed weaning 前置 phase 提供冻结训练材料。"""

    def __init__(self, records: tuple[object, ...]) -> None:
        self.records = records

    def read(self, _ctx: object, _report: object) -> tuple[object, ...]:
        """只读返回冻结材料。"""
        return self.records


class _Dev:
    """提供只读 dev 校准输入。"""

    def calibrate(self, _ctx: object, _report: object) -> tuple[str, ...]:
        """返回一个不含 evaluator 字段的 dev identity。"""
        return ("dev-observation",)


class _Shadow:
    """只持有独立 shadow error commitment 计数。"""

    def __init__(self) -> None:
        self.count = 0

    def record(self, _ctx: object, _errors: object) -> int:
        """追加一条 shadow 审计，不反馈 Candidate。"""
        self.count += 1
        return 1

    @property
    def audit_write_count(self) -> int:
        """返回独立 shadow 审计写数。"""
        return self.count


class _Stage4:
    """提供 typed weaning 所需的稳定完整 stage4 report。"""

    complete = True
    outcomes = ("shadow-error",)

    def stable_key(self) -> tuple[int, ...]:
        """返回固定 stage4 输出 identity。"""
        return (7, 8, 9)


def test_real_windows_are_measured_inside_typed_weaning_zero_call_boundary():
    """typed runtime 必须包住真实 clone 操作并完成三个连续窗口，而非事后登记。"""
    host = _host()
    protocol = _protocol(host)
    training = ("frozen-teacher-evidence",)
    typed_protocol = make_w09_typed_weaning_protocol(
        authority_sha256=w09_commitment("authority"),
        registry_identity=_key("registry"),
        candidate_identity=host.candidate_identity,
        input_commitment=w09_commitment(training),
        threshold_key=_key("typed-threshold"),
        window_input_commitments=protocol.window_input_commitments,
    )
    typed = W09TypedWeaningRuntime(
        typed_protocol,
        training_material_source=_Source(training),
        dev_calibrator=_Dev(),
        shadow_auditor=_Shadow(),
    )
    assert typed.run(host, _Stage4()).complete is False
    v06 = W09V06Runtime(host, protocol).run(typed_weaning_runtime=typed)
    ready = typed.run(host, _Stage4())
    assert ready.complete is True
    assert ready.windows == v06.window_identities
    assert tuple(item.window_ordinal for item in ready.windows) == (1, 2, 3)
    assert all(item.teacher_call_count == 0 for item in ready.windows)


def test_memory_use_rollback_and_fixed_core_ablations_are_measured():
    """四项消融必须从 fresh clone 量测到目标击穿，不能静态伪造状态。"""
    host = _host()
    runtime = W09V06Runtime(host, _protocol(host))
    reports = {key: runtime.ablate(key) for key in W09_V06_ABLATION_KEYS}

    assert reports["NEW_MEMORY_WRITE"].memory_write_count == 0
    assert reports["NEW_MEMORY_WRITE"].metric_improvement == 0
    assert reports["USE_OUTCOME_CONSUMER"].use_write_count == 0
    assert reports["USE_OUTCOME_CONSUMER"].metric_improvement == 0
    assert reports["FIXED_CORE_REPLAY"].metric_improvement == 0
    rollback = reports["ROLLBACK_CONSUMER"]
    assert rollback.erroneous_adoption_before_rollback == 9
    assert rollback.erroneous_adoption_after_rollback == 9
    assert rollback.target_dimension_key == "W-09-ROLLBACK"
    assert all(item.target_status == "FAIL" for item in reports.values())
    assert host.snapshot() == runtime.host.snapshot()


def test_teacher_source_identity_leakage_and_canonical_replay_fail_closed():
    """teacher 来源与 learning/probe identity 重叠必须拒绝；同输入 fresh 结果 bit-identical。"""
    host = _host()
    protocol = _protocol(host)
    record = protocol.windows[0].learning_records[0]
    with pytest.raises(W09V06Error):
        replace(record, provenance_kind="TEACHER_EVIDENCE")

    probe = protocol.windows[0].probe_owner.probes[0]
    collided = replace(
        probe,
        content_key=record.content_key,
    )
    probes = (collided, *protocol.windows[0].probe_owner.probes[1:])
    owner = replace(protocol.windows[0].probe_owner, probes=probes)
    with pytest.raises(W09V06Error):
        replace(protocol.windows[0], probe_owner=owner, input_commitment="")

    first = W09V06Runtime(host, protocol).run()
    second = W09V06Runtime(host, protocol).run()
    assert first.stable_key() == second.stable_key()
    assert first.window_identities == second.window_identities
