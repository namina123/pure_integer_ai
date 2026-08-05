"""W09-07 V-06 一次隔离 clone、三窗口学习和 rollback runtime。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pure_integer_ai.experiments.ph2_w05_contract import digest_value
from pure_integer_ai.experiments.ph2_w09_authority import (
    W09_CARRIER_KEYS,
    W09_CONSUMER_KEYS,
    W09_RESOURCE_BUDGET,
)
from pure_integer_ai.experiments.ph2_w09_dimensional import (
    W09ContinualLearningEvidence,
    W09DimensionalRuntime,
    W09_RETENTION_SCOPE_KEY,
)
from pure_integer_ai.experiments.ph2_w09_rollback import W09RollbackLedger
from pure_integer_ai.experiments.ph2_w09_types import (
    W09CloneReceipt,
    W09ConsumerChoice,
    W09ConsumerRequest,
    W09DirectionalResult,
    W09ResourceAudit,
    W09ResultState,
    W09UseOutcome,
    W09VerifierResult,
)
from pure_integer_ai.experiments.ph2_w09_v06_contract import (
    W09V06AblationReport,
    W09V06CellEvidence,
    W09V06Error,
    W09V06HostSnapshot,
    W09V06LearningRecord,
    W09V06Metric,
    W09V06Probe,
    W09V06Protocol,
    W09V06Report,
    W09V06WindowPlan,
    W09V06WindowReport,
    W09_V06_ABLATION_KEYS,
    W09_V06_PUBLIC_STATUS,
    w09_v06_commitment,
    w09_v06_key,
)


@dataclass
class W09V06HostCandidate:
    """只暴露完整快照和正式 Candidate identity 的待封存 host。"""

    candidate_identity: str
    snapshot_state: W09V06HostSnapshot

    def __post_init__(self) -> None:
        """校验 host identity 和完整快照类型。"""
        if (
            not isinstance(self.candidate_identity, str)
            or len(self.candidate_identity) != 64
            or any(item not in "0123456789abcdef" for item in self.candidate_identity)
        ):
            raise W09V06Error("W09-07 host candidate identity is invalid")
        if not isinstance(self.snapshot_state, W09V06HostSnapshot):
            raise W09V06Error("W09-07 host snapshot is invalid")

    def snapshot(self) -> W09V06HostSnapshot:
        """返回 host 的完整不可变快照，不触发读写副作用。"""
        return self.snapshot_state

    def state_key(self) -> tuple[int, ...]:
        """供 typed weaning 零写检查读取完整 host identity。"""
        return self.snapshot_state.stable_key()

    @property
    def teacher_calls(self) -> int:
        """返回 host teacher 调用计数。"""
        return self.snapshot_state.teacher_call_count

    @property
    def api_call_count(self) -> int:
        """返回 host API 调用计数。"""
        return self.snapshot_state.api_call_count

    @property
    def llm_call_count(self) -> int:
        """返回 host LLM 调用计数。"""
        return self.snapshot_state.llm_call_count


@dataclass(frozen=True)
class _CloneMemoryRecord:
    """把一个学习记录绑定到实际 append-only Use/outcome 身份。"""

    record: W09V06LearningRecord
    use_key: tuple[int, ...] | None
    outcome_key: tuple[int, ...] | None

    def stable_key(self) -> tuple[int, ...]:
        """返回 Memory record 与采用事件的联合 identity。"""
        return digest_value({
            "outcome": [] if self.outcome_key is None else list(self.outcome_key),
            "record": list(self.record.stable_key()),
            "use": [] if self.use_key is None else list(self.use_key),
        })


@dataclass(frozen=True)
class _ProbeSelection:
    """记录候选形成阶段输出；criterion 仅在形成完成后由 probe owner 使用。"""

    probe: W09V06Probe
    candidate_key: tuple[int, ...]
    active_record: _CloneMemoryRecord | None


@dataclass(frozen=True)
class _WindowMeasurement:
    """承载正常窗口和消融共用的真实执行量，不自行发布 PASS。"""

    state_before: tuple[int, ...]
    state_after: tuple[int, ...]
    core_before: tuple[int, ...]
    core_after: tuple[int, ...]
    metric_before: W09V06Metric
    metric_after: W09V06Metric
    erroneous_before: int
    erroneous_after: int
    data_candidate_count: int
    memory_write_count: int
    evidence_write_count: int
    use_write_count: int
    outcome_write_count: int
    rollback_event_count: int
    resource_audit: W09ResourceAudit
    rollback_audit_sha256: str
    cells: tuple[W09V06CellEvidence, ...]


class _W09V06Clone:
    """只写自身 Memory/Evidence/Use/outcome 的确定性隔离 clone。"""

    def __init__(
            self,
            protocol: W09V06Protocol,
            host_snapshot: W09V06HostSnapshot,
            *,
            memory_enabled: bool = True,
            use_enabled: bool = True,
            rollback_enabled: bool = True,
            fixed_core_replay: bool = False,
            ) -> None:
        """从 host commitment 创建一次 clone，不取得任何 host 可变对象引用。"""
        self.protocol = protocol
        self.clone_identity = protocol.clone_identity
        self.core_state_key = host_snapshot.core_state_key
        self.logical_clock = host_snapshot.logical_clock
        self.memory_enabled = memory_enabled
        self.use_enabled = use_enabled
        self.rollback_enabled = rollback_enabled
        self.fixed_core_replay = fixed_core_replay
        self.rollback_ledger = W09RollbackLedger(self.core_state_key)
        self.memory_records: list[_CloneMemoryRecord] = []
        self.probe_audits: list[tuple[int, ...]] = []
        self.output_records: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        self.memory_write_count = 0
        self.evidence_write_count = 0
        self.use_write_count = 0
        self.outcome_write_count = 0
        self.rollback_event_count = 0

    def state_key(self) -> tuple[int, ...]:
        """返回 Core、append-only log、Memory、probe Use 和 clone clock 状态。"""
        return digest_value({
            "clock": self.logical_clock,
            "clone": self.clone_identity,
            "core": list(self.core_state_key),
            "ledger": list(self.rollback_ledger.state_key()),
            "memory": [list(item.stable_key()) for item in self.memory_records],
            "outputs": [
                [list(probe), list(candidate)]
                for probe, candidate in self.output_records
            ],
            "probe_audits": [list(item) for item in self.probe_audits],
        })

    def output_commitment(self) -> str:
        """返回 clone 已实际形成的输出集合 identity。"""
        return w09_v06_commitment({
            "clone": self.clone_identity,
            "outputs": [
                [list(probe), list(candidate)]
                for probe, candidate in self.output_records
            ],
            "state": list(self.state_key()),
        })

    def _tick(self) -> None:
        """只推进 clone 自有逻辑时钟。"""
        self.logical_clock += 1

    def ingest(self, record: W09V06LearningRecord) -> None:
        """data-only 追加 Observation/Evidence/Memory/Use/outcome，不读取 probe 判据。"""
        scope_key = f"{record.carrier_key}:{record.consumer_key}"
        self.rollback_ledger.append(
            "OBSERVATION",
            record.source_key,
            scope_key,
        )
        self.rollback_ledger.append(
            "EVIDENCE",
            record.evidence_key,
            scope_key,
            depends_on=(record.source_key,),
        )
        self.evidence_write_count += 1
        use_key: tuple[int, ...] | None = None
        outcome_key: tuple[int, ...] | None = None
        if self.use_enabled:
            use_key = w09_v06_key({
                "kind": "LEARNING_USE",
                "record": list(record.stable_key()),
            })
            outcome_key = w09_v06_key({
                "kind": "LEARNING_OUTCOME",
                "record": list(record.stable_key()),
                "use": list(use_key),
            })
            self.rollback_ledger.append(
                "USE_OUTCOME",
                use_key,
                scope_key,
                depends_on=(record.evidence_key,),
            )
            self.rollback_ledger.append(
                "USE_OUTCOME",
                outcome_key,
                scope_key,
                depends_on=(use_key,),
            )
            self.use_write_count += 1
            self.outcome_write_count += 1
        if self.memory_enabled:
            self.memory_records.append(_CloneMemoryRecord(
                record,
                use_key,
                outcome_key,
            ))
            self.memory_write_count += 1
        self._tick()

    def _active_records(
            self,
            probe: W09V06Probe,
            threshold: int,
            ) -> tuple[_CloneMemoryRecord, ...]:
        """从当前有效 Memory 中选择同 family/direction 且已真实 Use 的记录。"""
        invalidated = set(self.rollback_ledger.evaluate().invalidated_keys)
        return tuple(
            item
            for item in self.memory_records
            if item.record.family_key == probe.family_key
            and item.record.consumer_key == probe.consumer_key
            and item.record.carrier_key == probe.carrier_key
            and item.record.strength >= threshold
            and item.use_key is not None
            and item.outcome_key is not None
            and not {
                item.record.source_key,
                item.record.evidence_key,
                item.use_key,
                item.outcome_key,
            }.intersection(invalidated)
        )

    def select(self, probe: W09V06Probe, threshold: int) -> _ProbeSelection:
        """只从 clone Memory 形成候选；固定 Core replay 明确忽略所有 Memory。"""
        if self.fixed_core_replay:
            return _ProbeSelection(probe, probe.baseline_candidate_key, None)
        records = self._active_records(probe, threshold)
        if not records:
            return _ProbeSelection(probe, probe.baseline_candidate_key, None)
        selected = max(
            records,
            key=lambda item: (
                item.record.strength,
                item.record.candidate_key,
                item.record.evidence_key,
            ),
        )
        return _ProbeSelection(probe, selected.record.candidate_key, selected)

    def measure(
            self,
            probes: tuple[W09V06Probe, ...],
            threshold: int,
            ) -> tuple[W09V06Metric, tuple[_ProbeSelection, ...]]:
        """先完成全部候选形成，再交给独立 criterion 逐项验收。"""
        selections = tuple(self.select(item, threshold) for item in probes)
        passed = sum(
            item.probe.criterion.accepts(item.candidate_key)
            for item in selections
        )
        return W09V06Metric(
            passed,
            len(selections),
            passed * 1000 // len(selections),
        ), selections

    def rollback_error(self, record: W09V06LearningRecord) -> None:
        """追加错误采用反证和 SourceRef retract，由 dependency closure 局部失效。"""
        if not self.rollback_enabled:
            return
        memory = next(
            (
                item for item in reversed(self.memory_records)
                if item.record.evidence_key == record.evidence_key
            ),
            None,
        )
        if memory is None or memory.outcome_key is None:
            return
        scope_key = f"{record.carrier_key}:{record.consumer_key}"
        correction_key = w09_v06_key({
            "kind": "NEGATIVE_USE_OUTCOME",
            "outcome": list(memory.outcome_key),
            "source": list(record.source_key),
        })
        self.rollback_ledger.append(
            "USE_OUTCOME",
            correction_key,
            scope_key,
            depends_on=(memory.outcome_key,),
        )
        self.outcome_write_count += 1
        retract_key = w09_v06_key({
            "kind": "SOURCE_RETRACT",
            "source": list(record.source_key),
            "window": record.window_ordinal,
        })
        self.rollback_ledger.retract_source(
            record.source_key,
            retract_key,
            scope_key,
        )
        self.rollback_event_count += 1
        self._tick()

    def record_probe_result(
            self,
            selection: _ProbeSelection,
            before_state_key: tuple[int, ...],
            source_evidence_key: tuple[int, ...],
            ) -> W09V06CellEvidence:
        """把最终 held-out 输出写为 clone-only Use/outcome，并形成实际 continual cell。"""
        probe = selection.probe
        candidate_key = selection.candidate_key
        request_key = w09_v06_key({
            "kind": "PROBE_REQUEST",
            "probe": list(probe.probe_key),
        })
        choice_key = w09_v06_key({
            "candidate": list(candidate_key),
            "kind": "PROBE_CHOICE",
            "request": list(request_key),
        })
        use_key = w09_v06_key({
            "choice": list(choice_key),
            "kind": "PROBE_USE",
            "probe": list(probe.probe_key),
        })
        outcome_key = w09_v06_key({
            "candidate": list(candidate_key),
            "kind": "PROBE_OUTCOME",
            "use": list(use_key),
        })
        passed = probe.criterion.accepts(candidate_key)
        verifier_key = w09_v06_key({
            "criterion": list(probe.criterion.criterion_key),
            "kind": "PROBE_VERIFIER",
            "outcome": list(outcome_key),
            "passed": int(passed),
        })
        result = W09DirectionalResult(
            W09ConsumerRequest(
                probe.consumer_key,
                request_key,
                bytes(probe.input_key()).hex(),
            ),
            W09ConsumerChoice(
                probe.consumer_key,
                request_key,
                choice_key,
                candidate_key,
            ),
            W09UseOutcome(
                probe.consumer_key,
                request_key,
                choice_key,
                candidate_key,
                use_key,
                outcome_key,
                "RESOLVED",
            ),
            W09VerifierResult(
                probe.consumer_key,
                request_key,
                use_key,
                outcome_key,
                verifier_key,
                W09ResultState.PASS if passed else W09ResultState.FAIL,
                "NONE" if passed else "HELD_OUT_CRITERION_NOT_MET",
            ),
        )
        self.probe_audits.extend((use_key, outcome_key))
        self.output_records.append((probe.probe_key, candidate_key))
        self.use_write_count += 1
        self.outcome_write_count += 1
        self._tick()
        after_state_key = w09_v06_key({
            "candidate": list(candidate_key),
            "clone_state": list(self.state_key()),
            "probe": list(probe.probe_key),
            "source_evidence": list(source_evidence_key),
        })
        continual = W09ContinualLearningEvidence(
            before_state_key,
            after_state_key,
            source_evidence_key,
            use_key,
            outcome_key,
            0,
            0,
        )
        return W09V06CellEvidence(
            probe.window_ordinal,
            probe.carrier_key,
            probe.consumer_key,
            result,
            continual,
        )


class W09V06Runtime:
    """在一次 clone 中顺序执行三个独立 teacher-zero 后续学习窗口。"""

    def __init__(self, host: W09V06HostCandidate, protocol: W09V06Protocol) -> None:
        """绑定待封存 host 和预注册协议，不创建或写入 clone。"""
        if not isinstance(host, W09V06HostCandidate):
            raise W09V06Error("W09-07 host type is invalid")
        if not isinstance(protocol, W09V06Protocol):
            raise W09V06Error("W09-07 protocol type is invalid")
        if (
            host.candidate_identity != protocol.candidate_identity
            or host.snapshot().core_state_key != protocol.core_state_key
        ):
            raise W09V06Error("W09-07 host/protocol identity drifted")
        self.host = host
        self.protocol = protocol
        self._report: W09V06Report | None = None

    def state_key(self) -> tuple[int, ...]:
        """返回 host、协议和可选完成报告的 runtime identity。"""
        return digest_value({
            "host": list(self.host.snapshot().stable_key()),
            "protocol": list(self.protocol.stable_key()),
            "report": [] if self._report is None else list(self._report.stable_key()),
        })

    @staticmethod
    def _before_cell_state(probe: W09V06Probe) -> tuple[int, ...]:
        """形成一个未学习 probe 的独立 baseline state identity。"""
        return w09_v06_key({
            "baseline": list(probe.baseline_candidate_key),
            "kind": "UNLEARNED_HELD_OUT_STATE",
            "probe": list(probe.probe_key),
        })

    @staticmethod
    def _resource_audit(
            plan: W09V06WindowPlan,
            clone: _W09V06Clone,
            event_start: int,
            ) -> W09ResourceAudit:
        """从实际窗口写入、判据执行和失效闭包形成八项资源账。"""
        event_delta = len(clone.rollback_ledger.events) - event_start
        invalidated = len(clone.rollback_ledger.evaluate().invalidated_keys)
        used = {
            "max_checkpoint_count": 1,
            "max_logic_operations": event_delta + len(plan.probe_owner.probes) * 4,
            "max_payload_bytes": 0,
            "max_payload_gets": 0,
            "max_recompute_objects": invalidated,
            "max_records": (
                len(plan.learning_records)
                + len(plan.error_records)
                + len(plan.probe_owner.probes)
            ),
            "max_segments": len(plan.probe_owner.carrier_keys),
            "max_workers": plan.worker_count,
        }
        return W09ResourceAudit(
            tuple(sorted(used.items())),
            tuple(sorted(W09_RESOURCE_BUDGET.items())),
        )

    def _execute_window(
            self,
            clone: _W09V06Clone,
            plan: W09V06WindowPlan,
            *,
            publish_cells: bool,
            ) -> _WindowMeasurement:
        """执行 good adoption、错误采用、局部 rollback 和独立 probe 复验。"""
        host_before = self.host.snapshot()
        state_before = clone.state_key()
        core_before = clone.core_state_key
        counters_before = (
            clone.memory_write_count,
            clone.evidence_write_count,
            clone.use_write_count,
            clone.outcome_write_count,
            clone.rollback_event_count,
        )
        event_start = len(clone.rollback_ledger.events)
        metric_before, baseline = clone.measure(
            plan.probe_owner.probes,
            plan.adoption_threshold,
        )
        before_states = {
            item.probe.probe_key: self._before_cell_state(item.probe)
            for item in baseline
        }
        for record in plan.learning_records:
            clone.ingest(record)
        learned_metric, _ = clone.measure(
            plan.probe_owner.probes,
            plan.adoption_threshold,
        )
        for record in plan.error_records:
            clone.ingest(record)
        error_metric, _ = clone.measure(
            plan.probe_owner.probes,
            plan.adoption_threshold,
        )
        erroneous_before = error_metric.total - error_metric.passed
        for record in plan.error_records:
            clone.rollback_error(record)
        metric_after, final_selections = clone.measure(
            plan.probe_owner.probes,
            plan.adoption_threshold,
        )
        erroneous_after = metric_after.total - metric_after.passed
        cells: tuple[W09V06CellEvidence, ...] = ()
        if publish_cells:
            if learned_metric.passed != learned_metric.total:
                raise W09V06Error("W09-07 good non-teacher Evidence was not adopted")
            good_by_cell = {item.cell_key: item for item in plan.learning_records}
            cells = tuple(
                clone.record_probe_result(
                    selection,
                    before_states[selection.probe.probe_key],
                    good_by_cell[selection.probe.cell_key].evidence_key,
                )
                for selection in final_selections
            )
        resource_audit = self._resource_audit(plan, clone, event_start)
        evaluation = clone.rollback_ledger.evaluate()
        rollback_audit = w09_v06_commitment({
            "invalidated": [list(item) for item in evaluation.invalidated_keys],
            "ledger": list(evaluation.event_log_key),
            "preserved": [list(item) for item in evaluation.preserved_keys],
            "window": plan.window_ordinal,
        })
        counters_after = (
            clone.memory_write_count,
            clone.evidence_write_count,
            clone.use_write_count,
            clone.outcome_write_count,
            clone.rollback_event_count,
        )
        deltas = tuple(
            after - before
            for before, after in zip(counters_before, counters_after)
        )
        host_after = self.host.snapshot()
        if host_before != host_after:
            raise W09V06Error("W09-07 clone operation changed host state")
        return _WindowMeasurement(
            state_before,
            clone.state_key(),
            core_before,
            clone.core_state_key,
            metric_before,
            metric_after,
            erroneous_before,
            erroneous_after,
            len(plan.learning_records) + len(plan.error_records),
            deltas[0],
            deltas[1],
            deltas[2],
            deltas[3],
            deltas[4],
            resource_audit,
            rollback_audit,
            cells,
        )

    def _window_report(
            self,
            clone: _W09V06Clone,
            plan: W09V06WindowPlan,
            ) -> W09V06WindowReport:
        """把一次正常实际执行封装为强不变量窗口报告。"""
        host_before = self.host.snapshot()
        measurement = self._execute_window(clone, plan, publish_cells=True)
        host_after = self.host.snapshot()
        return W09V06WindowReport(
            plan.window_ordinal,
            plan.input_commitment,
            plan.threshold_key,
            self.protocol.candidate_identity,
            clone.clone_identity,
            host_before,
            host_after,
            measurement.core_before,
            measurement.core_after,
            measurement.state_before,
            measurement.state_after,
            measurement.metric_before,
            measurement.metric_after,
            measurement.erroneous_before,
            measurement.erroneous_after,
            measurement.data_candidate_count,
            measurement.memory_write_count,
            measurement.evidence_write_count,
            measurement.use_write_count,
            measurement.outcome_write_count,
            measurement.rollback_event_count,
            0,
            0,
            0,
            0,
            measurement.resource_audit,
            measurement.rollback_audit_sha256,
            measurement.cells,
        )

    def run(self, *, typed_weaning_runtime: Any = None) -> W09V06Report:
        """顺序执行三个窗口；可让 typed runtime 包住每次真实操作并登记 identity。"""
        if self._report is not None:
            if self.host.snapshot() != self._report.host_after:
                raise W09V06Error("W09-07 completed host drifted")
            return self._report
        host_before = self.host.snapshot()
        if typed_weaning_runtime is not None:
            typed_protocol = getattr(typed_weaning_runtime, "protocol", None)
            if (
                getattr(typed_protocol, "candidate_identity", None)
                != self.protocol.candidate_identity
                or getattr(typed_protocol, "window_input_commitments", None)
                != self.protocol.window_input_commitments
            ):
                raise W09V06Error("W09-07 typed weaning protocol drifted")
            measured_entry = getattr(
                typed_weaning_runtime,
                "execute_measured_zero_call_window",
                None,
            )
            if not callable(measured_entry):
                raise W09V06Error("W09-07 typed weaning measured entry is missing")
        clone = _W09V06Clone(self.protocol, host_before)
        pre_output = clone.output_commitment()
        window_reports: list[W09V06WindowReport] = []
        for plan in self.protocol.windows:
            holder: list[W09V06WindowReport] = []

            def operation(current: W09V06WindowPlan = plan) -> object:
                """执行当前真实窗口并把其实际 identity 返回给 typed runtime。"""
                report = self._window_report(clone, current)
                holder.append(report)
                return report.window_identity

            if typed_weaning_runtime is None:
                operation()
            else:
                typed_weaning_runtime.execute_measured_zero_call_window(
                    self.host,
                    operation,
                )
            if len(holder) != 1:
                raise W09V06Error("W09-07 window operation was not executed exactly once")
            window_reports.append(holder[0])
        host_after = self.host.snapshot()
        if host_before != host_after:
            raise W09V06Error("W09-07 three-window run changed host")
        cells = tuple(
            cell
            for report in window_reports
            for cell in report.cells
        )
        clone_receipt = W09CloneReceipt(
            self.protocol.candidate_identity,
            clone.clone_identity,
            pre_output,
            clone.output_commitment(),
            sum(
                report.memory_write_count
                + report.evidence_write_count
                + report.use_write_count
                + report.outcome_write_count
                for report in window_reports
            ),
            0,
        )
        self._report = W09V06Report(
            self.protocol.stable_key(),
            host_before,
            host_after,
            clone_receipt,
            tuple(window_reports),
            cells,
            1,
            0,
            0,
            0,
            0,
            W09_V06_PUBLIC_STATUS,
            0,
            0,
            0,
        )
        return self._report

    def ablate(self, component_key: str) -> W09V06AblationReport:
        """在 fresh clone 上真实禁用一个组件并量测学习或 rollback 击穿。"""
        if component_key not in W09_V06_ABLATION_KEYS:
            raise W09V06Error("W09-07 ablation component is invalid")
        host_before = self.host.snapshot()
        clone = _W09V06Clone(
            self.protocol,
            host_before,
            memory_enabled=component_key != "NEW_MEMORY_WRITE",
            use_enabled=component_key != "USE_OUTCOME_CONSUMER",
            rollback_enabled=component_key != "ROLLBACK_CONSUMER",
            fixed_core_replay=component_key == "FIXED_CORE_REPLAY",
        )
        measurement = self._execute_window(
            clone,
            self.protocol.windows[0],
            publish_cells=False,
        )
        if self.host.snapshot() != host_before:
            raise W09V06Error("W09-07 ablation changed host")
        target = (
            "W-09-ROLLBACK"
            if component_key == "ROLLBACK_CONSUMER"
            else "W-09-V06_CLONE"
        )
        return W09V06AblationReport(
            component_key,
            target,
            measurement.memory_write_count,
            measurement.use_write_count,
            max(
                0,
                measurement.metric_after.permille
                - measurement.metric_before.permille,
            ),
            measurement.erroneous_before,
            measurement.erroneous_after,
            "FAIL",
            0,
        )


def record_w09_v06_continual_cells(
        runtime: W09DimensionalRuntime,
        report: W09V06Report,
        ) -> None:
    """把 27 个实际窗口 cell 接入 W09-04 dimensional runtime。"""
    if not isinstance(runtime, W09DimensionalRuntime):
        raise W09V06Error("W09-07 dimensional runtime type is invalid")
    if not isinstance(report, W09V06Report):
        raise W09V06Error("W09-07 report type is invalid")
    by_key = {item.cell_key: item for item in report.cells}
    for carrier_key in W09_CARRIER_KEYS:
        for consumer_key in W09_CONSUMER_KEYS:
            cell = by_key[(carrier_key, consumer_key)]
            runtime.record_cell(
                W09_RETENTION_SCOPE_KEY,
                carrier_key,
                consumer_key,
                cell.result,
                continual_learning=cell.continual_learning,
            )


__all__ = [
    "W09V06HostCandidate",
    "W09V06Runtime",
    "record_w09_v06_continual_cells",
]
