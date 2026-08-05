"""W09-07 V-06 隔离后续学习、连续窗口和实际 cell 证据合同。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.experiments.ph2_w05_contract import digest_value
from pure_integer_ai.experiments.ph2_w09_authority import (
    W09_CARRIER_KEYS,
    W09_CONSUMER_KEYS,
    W09_DIMENSION_KEYS,
    W09_RESOURCE_BUDGET,
)
from pure_integer_ai.experiments.ph2_w09_dimensional import (
    W09ContinualLearningEvidence,
)
from pure_integer_ai.experiments.ph2_w09_types import (
    TeacherExitPhase,
    W09CloneReceipt,
    W09DirectionalResult,
    W09ResourceAudit,
    W09ResultState,
    W09WindowIdentity,
)


W09_V06_PROVENANCE_KIND = "NON_TEACHER_EVIDENCE"
W09_V06_PUBLIC_STATUS = "PUBLIC_BOUNDED_NOT_FORMAL"
W09_V06_WINDOW_STATUS = "PUBLIC_BOUNDED_PASS"
W09_V06_ABLATION_KEYS = (
    "NEW_MEMORY_WRITE",
    "USE_OUTCOME_CONSUMER",
    "ROLLBACK_CONSUMER",
    "FIXED_CORE_REPLAY",
)


class W09V06Error(RuntimeError):
    """W09-07 clone、窗口、学习 delta 或隔离不变量发生错误。"""


def _key(value: object, *, where: str) -> tuple[int, ...]:
    """校验一个固定 32-byte 的稳定整数身份。"""
    if (
        not isinstance(value, tuple)
        or len(value) != 32
        or any(type(item) is not int or not 0 <= item <= 255 for item in value)
    ):
        raise W09V06Error(f"{where} identity is invalid")
    return value


def _sha(value: object, *, where: str) -> str:
    """校验一个小写 SHA-256 commitment。"""
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(item not in "0123456789abcdef" for item in value)
    ):
        raise W09V06Error(f"{where} commitment is invalid")
    return value


def w09_v06_key(value: object) -> tuple[int, ...]:
    """把只含公开稳定字段的值编码为 W09-07 identity。"""
    return digest_value(value)


def w09_v06_commitment(value: object) -> str:
    """把公开稳定值编码为 SHA-256 commitment。"""
    return bytes(digest_value(value)).hex()


@dataclass(frozen=True)
class W09V06HostSnapshot:
    """冻结 host Core、学习载体、评测状态、游标、报告和调用计数。"""

    core_state_key: tuple[int, ...]
    memory_state_key: tuple[int, ...]
    use_state_key: tuple[int, ...]
    evidence_state_key: tuple[int, ...]
    assessment_state_key: tuple[int, ...]
    cursor_state_key: tuple[int, ...]
    report_state_key: tuple[int, ...]
    logical_clock: int
    teacher_call_count: int = 0
    api_call_count: int = 0
    llm_call_count: int = 0

    def __post_init__(self) -> None:
        """要求 host 快照覆盖全部隔离字段且计数严格非负。"""
        for name in (
            "core_state_key",
            "memory_state_key",
            "use_state_key",
            "evidence_state_key",
            "assessment_state_key",
            "cursor_state_key",
            "report_state_key",
        ):
            _key(getattr(self, name), where=name)
        counters = (
            self.logical_clock,
            self.teacher_call_count,
            self.api_call_count,
            self.llm_call_count,
        )
        if any(type(item) is not int or item < 0 for item in counters):
            raise W09V06Error("W09-07 host counter is invalid")

    def stable_key(self) -> tuple[int, ...]:
        """返回覆盖所有 host 隔离字段的 canonical identity。"""
        return digest_value({
            "api_calls": self.api_call_count,
            "assessment": list(self.assessment_state_key),
            "clock": self.logical_clock,
            "core": list(self.core_state_key),
            "cursor": list(self.cursor_state_key),
            "evidence": list(self.evidence_state_key),
            "llm_calls": self.llm_call_count,
            "memory": list(self.memory_state_key),
            "report": list(self.report_state_key),
            "teacher_calls": self.teacher_call_count,
            "use": list(self.use_state_key),
        })


@dataclass(frozen=True)
class W09V06LearningRecord:
    """一条 data-only、非 teacher、可形成候选的 clone 学习 Evidence。"""

    window_ordinal: int
    carrier_key: str
    consumer_key: str
    family_key: tuple[int, ...]
    content_key: tuple[int, ...]
    source_key: tuple[int, ...]
    evidence_key: tuple[int, ...]
    candidate_key: tuple[int, ...]
    strength: int
    provenance_kind: str = W09_V06_PROVENANCE_KIND

    def __post_init__(self) -> None:
        """拒绝 teacher 来源、非法方向、空强度或不稳定身份。"""
        if type(self.window_ordinal) is not int or not 1 <= self.window_ordinal <= 3:
            raise W09V06Error("W09-07 learning window is invalid")
        if self.carrier_key not in W09_CARRIER_KEYS:
            raise W09V06Error("W09-07 learning carrier is invalid")
        if self.consumer_key not in W09_CONSUMER_KEYS:
            raise W09V06Error("W09-07 learning consumer is invalid")
        for name in (
            "family_key",
            "content_key",
            "source_key",
            "evidence_key",
            "candidate_key",
        ):
            _key(getattr(self, name), where=name)
        if type(self.strength) is not int or self.strength <= 0:
            raise W09V06Error("W09-07 learning strength is invalid")
        if self.provenance_kind != W09_V06_PROVENANCE_KIND:
            raise W09V06Error("W09-07 learning source is not non-teacher Evidence")

    @property
    def cell_key(self) -> tuple[str, str]:
        """返回 carrier×consumer 学习单元。"""
        return self.carrier_key, self.consumer_key

    def stable_key(self) -> tuple[int, ...]:
        """返回不含 surface 或 evaluator 判据的学习记录身份。"""
        return digest_value({
            "candidate": list(self.candidate_key),
            "carrier": self.carrier_key,
            "consumer": self.consumer_key,
            "content": list(self.content_key),
            "evidence": list(self.evidence_key),
            "family": list(self.family_key),
            "provenance": self.provenance_kind,
            "source": list(self.source_key),
            "strength": self.strength,
            "window": self.window_ordinal,
        })


@dataclass(frozen=True)
class W09V06ProbeCriterion:
    """由独立 probe owner 持有、学习路径不可读取的验收判据。"""

    criterion_key: tuple[int, ...]
    admitted_candidate_keys: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        """要求判据非空、规范排序且每个候选身份有效。"""
        _key(self.criterion_key, where="probe criterion")
        if (
            not isinstance(self.admitted_candidate_keys, tuple)
            or not self.admitted_candidate_keys
            or self.admitted_candidate_keys
            != tuple(sorted(set(self.admitted_candidate_keys)))
        ):
            raise W09V06Error("W09-07 probe criterion candidates are invalid")
        for item in self.admitted_candidate_keys:
            _key(item, where="probe admitted candidate")

    def accepts(self, candidate_key: tuple[int, ...]) -> bool:
        """只在 probe 阶段判断一个已形成输出，不参与候选排序。"""
        _key(candidate_key, where="probe output")
        return candidate_key in self.admitted_candidate_keys

    def stable_key(self) -> tuple[int, ...]:
        """返回 evaluator-owned 判据 identity。"""
        return digest_value({
            "admitted": [list(item) for item in self.admitted_candidate_keys],
            "criterion": list(self.criterion_key),
        })


@dataclass(frozen=True)
class W09V06Probe:
    """一条未参与学习、与学习内容身份分离的 held-out probe。"""

    window_ordinal: int
    carrier_key: str
    consumer_key: str
    family_key: tuple[int, ...]
    probe_key: tuple[int, ...]
    content_key: tuple[int, ...]
    baseline_candidate_key: tuple[int, ...]
    criterion: W09V06ProbeCriterion

    def __post_init__(self) -> None:
        """校验 probe 身份、方向和未学习基线不能预先通过。"""
        if type(self.window_ordinal) is not int or not 1 <= self.window_ordinal <= 3:
            raise W09V06Error("W09-07 probe window is invalid")
        if self.carrier_key not in W09_CARRIER_KEYS:
            raise W09V06Error("W09-07 probe carrier is invalid")
        if self.consumer_key not in W09_CONSUMER_KEYS:
            raise W09V06Error("W09-07 probe consumer is invalid")
        for name in (
            "family_key",
            "probe_key",
            "content_key",
            "baseline_candidate_key",
        ):
            _key(getattr(self, name), where=name)
        if not isinstance(self.criterion, W09V06ProbeCriterion):
            raise W09V06Error("W09-07 probe criterion is invalid")
        if self.criterion.accepts(self.baseline_candidate_key):
            raise W09V06Error("W09-07 probe baseline already satisfies criterion")

    @property
    def cell_key(self) -> tuple[str, str]:
        """返回 carrier×consumer probe 单元。"""
        return self.carrier_key, self.consumer_key

    def input_key(self) -> tuple[int, ...]:
        """返回不含 criterion 的学习侧不可见 probe input identity。"""
        return digest_value({
            "carrier": self.carrier_key,
            "consumer": self.consumer_key,
            "content": list(self.content_key),
            "family": list(self.family_key),
            "probe": list(self.probe_key),
            "window": self.window_ordinal,
        })

    def stable_key(self) -> tuple[int, ...]:
        """返回 probe 输入、基线和独立 criterion 的联合身份。"""
        return digest_value({
            "baseline": list(self.baseline_candidate_key),
            "criterion": list(self.criterion.stable_key()),
            "input": list(self.input_key()),
        })


@dataclass(frozen=True)
class W09V06ProbeOwner:
    """独立持有一个窗口的九个 probe 及判据。"""

    owner_key: tuple[int, ...]
    probes: tuple[W09V06Probe, ...]

    def __post_init__(self) -> None:
        """要求三 carrier×三 consumer 完整、规范且身份不共享。"""
        _key(self.owner_key, where="probe owner")
        if len(self.probes) != 9 or any(
            not isinstance(item, W09V06Probe) for item in self.probes
        ):
            raise W09V06Error("W09-07 probe owner requires nine probes")
        carriers = tuple(dict.fromkeys(item.carrier_key for item in self.probes))
        expected = tuple(
            (carrier, consumer)
            for carrier in carriers
            for consumer in W09_CONSUMER_KEYS
        )
        if len(carriers) != 3 or tuple(item.cell_key for item in self.probes) != expected:
            raise W09V06Error("W09-07 probe owner coverage is incomplete")
        identities = tuple(
            key
            for item in self.probes
            for key in (
                item.probe_key,
                item.content_key,
                item.criterion.criterion_key,
            )
        )
        if len(set(identities)) != len(identities):
            raise W09V06Error("W09-07 probe owner shares identity")

    @property
    def carrier_keys(self) -> tuple[str, ...]:
        """返回本窗口按预注册顺序覆盖的三个 carrier。"""
        return tuple(dict.fromkeys(item.carrier_key for item in self.probes))

    def stable_key(self) -> tuple[int, ...]:
        """返回独立 probe owner 与九条 probe 的 identity。"""
        return digest_value({
            "owner": list(self.owner_key),
            "probes": [list(item.stable_key()) for item in self.probes],
        })


@dataclass(frozen=True)
class W09V06WindowPlan:
    """冻结一个窗口的 input、threshold、budget、学习批次和 probe owner。"""

    window_ordinal: int
    threshold_key: tuple[int, ...]
    adoption_threshold: int
    learning_records: tuple[W09V06LearningRecord, ...]
    error_records: tuple[W09V06LearningRecord, ...]
    probe_owner: W09V06ProbeOwner
    input_commitment: str = ""
    worker_count: int = 1
    resource_limits: tuple[tuple[str, int], ...] = tuple(
        sorted(W09_RESOURCE_BUDGET.items())
    )

    def __post_init__(self) -> None:
        """核验九 cell、内容隔离、候选效果和完整预注册边界。"""
        if type(self.window_ordinal) is not int or not 1 <= self.window_ordinal <= 3:
            raise W09V06Error("W09-07 window ordinal is invalid")
        _key(self.threshold_key, where="window threshold")
        if type(self.adoption_threshold) is not int or self.adoption_threshold <= 0:
            raise W09V06Error("W09-07 adoption threshold is invalid")
        if self.worker_count not in (1, 2, 4):
            raise W09V06Error("W09-07 worker count is invalid")
        if self.resource_limits != tuple(sorted(W09_RESOURCE_BUDGET.items())):
            raise W09V06Error("W09-07 resource limits drifted")
        if not isinstance(self.probe_owner, W09V06ProbeOwner):
            raise W09V06Error("W09-07 probe owner is invalid")
        expected = tuple(item.cell_key for item in self.probe_owner.probes)
        for records, kind in (
            (self.learning_records, "learning"),
            (self.error_records, "error"),
        ):
            if (
                len(records) != 9
                or any(not isinstance(item, W09V06LearningRecord) for item in records)
                or tuple(item.cell_key for item in records) != expected
                or any(item.window_ordinal != self.window_ordinal for item in records)
            ):
                raise W09V06Error(f"W09-07 {kind} records do not cover nine cells")
        if any(item.window_ordinal != self.window_ordinal for item in self.probe_owner.probes):
            raise W09V06Error("W09-07 probe window drifted")
        probe_by_cell = {item.cell_key: item for item in self.probe_owner.probes}
        good_by_cell = {item.cell_key: item for item in self.learning_records}
        bad_by_cell = {item.cell_key: item for item in self.error_records}
        for cell in expected:
            probe = probe_by_cell[cell]
            good = good_by_cell[cell]
            bad = bad_by_cell[cell]
            if good.family_key != probe.family_key or bad.family_key != probe.family_key:
                raise W09V06Error("W09-07 learning/probe family drifted")
            if not probe.criterion.accepts(good.candidate_key):
                raise W09V06Error("W09-07 non-teacher Evidence cannot satisfy probe")
            if probe.criterion.accepts(bad.candidate_key):
                raise W09V06Error("W09-07 error Evidence does not create an error")
            if good.strength < self.adoption_threshold or bad.strength <= good.strength:
                raise W09V06Error("W09-07 Evidence strength cannot prove adoption/rollback")
        learning_ids = tuple(
            key
            for item in (*self.learning_records, *self.error_records)
            for key in (item.content_key, item.source_key, item.evidence_key)
        )
        probe_ids = tuple(
            key
            for item in self.probe_owner.probes
            for key in (
                item.content_key,
                item.probe_key,
                item.criterion.criterion_key,
            )
        )
        if (
            len(set(learning_ids)) != len(learning_ids)
            or len(set(probe_ids)) != len(probe_ids)
            or set(learning_ids).intersection(probe_ids)
        ):
            raise W09V06Error("W09-07 learning and held-out identities are not isolated")
        calculated = self._input_commitment()
        if not self.input_commitment:
            object.__setattr__(self, "input_commitment", calculated)
        elif _sha(self.input_commitment, where="window input") != calculated:
            raise W09V06Error("W09-07 window input commitment drifted")

    def _input_commitment(self) -> str:
        """只从 data-only 学习输入和 probe input 身份计算窗口 commitment。"""
        return w09_v06_commitment({
            "error_records": [list(item.stable_key()) for item in self.error_records],
            "learning_records": [
                list(item.stable_key()) for item in self.learning_records
            ],
            "probe_inputs": [
                list(item.input_key()) for item in self.probe_owner.probes
            ],
            "threshold": list(self.threshold_key),
            "threshold_value": self.adoption_threshold,
            "window": self.window_ordinal,
            "worker_count": self.worker_count,
        })

    def stable_key(self) -> tuple[int, ...]:
        """返回完整窗口预注册 identity。"""
        return digest_value({
            "input": self.input_commitment,
            "limits": dict(self.resource_limits),
            "probe_owner": list(self.probe_owner.stable_key()),
            "threshold": list(self.threshold_key),
            "threshold_value": self.adoption_threshold,
            "window": self.window_ordinal,
            "worker_count": self.worker_count,
        })


@dataclass(frozen=True)
class W09V06Protocol:
    """冻结一次 clone、三个连续窗口和九 carrier 的不交叠输入。"""

    candidate_identity: str
    core_state_key: tuple[int, ...]
    windows: tuple[W09V06WindowPlan, ...]
    clone_owner_key: str = "PH2_W09_V06_ISOLATED_CLONE_OWNER"

    def __post_init__(self) -> None:
        """要求三个窗口连续、全 carrier 覆盖且任何内容身份不跨窗共享。"""
        _sha(self.candidate_identity, where="candidate")
        _key(self.core_state_key, where="candidate Core")
        if tuple(item.window_ordinal for item in self.windows) != (1, 2, 3):
            raise W09V06Error("W09-07 requires three contiguous windows")
        if any(not isinstance(item, W09V06WindowPlan) for item in self.windows):
            raise W09V06Error("W09-07 window plan type is invalid")
        carriers = tuple(
            carrier
            for window in self.windows
            for carrier in window.probe_owner.carrier_keys
        )
        if carriers != W09_CARRIER_KEYS:
            raise W09V06Error("W09-07 nine-carrier window partition drifted")
        records = tuple(
            record
            for window in self.windows
            for record in (*window.learning_records, *window.error_records)
        )
        probes = tuple(
            probe for window in self.windows for probe in window.probe_owner.probes
        )
        unique_groups = (
            tuple(item.content_key for item in records),
            tuple(item.source_key for item in records),
            tuple(item.evidence_key for item in records),
            tuple(item.candidate_key for item in records),
            tuple(item.content_key for item in probes),
            tuple(item.probe_key for item in probes),
            tuple(item.baseline_candidate_key for item in probes),
            tuple(item.criterion.criterion_key for item in probes),
        )
        if any(len(set(group)) != len(group) for group in unique_groups):
            raise W09V06Error("W09-07 windows share protected learning/probe identity")
        family_counts: dict[tuple[int, ...], int] = {}
        for window in self.windows:
            for probe in window.probe_owner.probes:
                family_counts[probe.family_key] = family_counts.get(probe.family_key, 0) + 1
        family_windows = {
            family: {
                window.window_ordinal
                for window in self.windows
                for probe in window.probe_owner.probes
                if probe.family_key == family
            }
            for family in family_counts
        }
        if any(len(value) != 1 for value in family_windows.values()):
            raise W09V06Error("W09-07 windows share a revealed learning family")
        if self.clone_owner_key != "PH2_W09_V06_ISOLATED_CLONE_OWNER":
            raise W09V06Error("W09-07 clone owner drifted")

    @property
    def window_input_commitments(self) -> tuple[str, ...]:
        """返回与 typed weaning protocol 对接的三个输入 commitment。"""
        return tuple(item.input_commitment for item in self.windows)

    @property
    def clone_identity(self) -> str:
        """从 host、Core、owner 和完整协议确定唯一隔离 clone identity。"""
        return w09_v06_commitment({
            "candidate": self.candidate_identity,
            "core": list(self.core_state_key),
            "owner": self.clone_owner_key,
            "protocol": list(self.stable_key()),
        })

    def stable_key(self) -> tuple[int, ...]:
        """返回三窗口 V-06 协议 identity。"""
        return digest_value({
            "candidate": self.candidate_identity,
            "core": list(self.core_state_key),
            "owner": self.clone_owner_key,
            "windows": [list(item.stable_key()) for item in self.windows],
        })


@dataclass(frozen=True)
class W09V06Metric:
    """未参与学习 probe 的逐项通过数和纯整数千分比。"""

    passed: int
    total: int
    permille: int

    def __post_init__(self) -> None:
        """要求 metric 从逐项计数精确派生。"""
        if (
            type(self.passed) is not int
            or type(self.total) is not int
            or type(self.permille) is not int
            or self.total <= 0
            or not 0 <= self.passed <= self.total
            or self.permille != self.passed * 1000 // self.total
        ):
            raise W09V06Error("W09-07 held-out metric is invalid")


@dataclass(frozen=True)
class W09V06CellEvidence:
    """由实际窗口输出形成的一个 carrier×consumer continual cell。"""

    window_ordinal: int
    carrier_key: str
    consumer_key: str
    result: W09DirectionalResult
    continual_learning: W09ContinualLearningEvidence

    def __post_init__(self) -> None:
        """要求 cell 方向、PASS 结果和实际 Use/outcome delta 完全绑定。"""
        if type(self.window_ordinal) is not int or not 1 <= self.window_ordinal <= 3:
            raise W09V06Error("W09-07 cell window is invalid")
        if self.carrier_key not in W09_CARRIER_KEYS or self.consumer_key not in W09_CONSUMER_KEYS:
            raise W09V06Error("W09-07 cell carrier/consumer is invalid")
        if not isinstance(self.result, W09DirectionalResult):
            raise W09V06Error("W09-07 cell result is invalid")
        if not isinstance(self.continual_learning, W09ContinualLearningEvidence):
            raise W09V06Error("W09-07 cell learning evidence is invalid")
        if (
            self.result.request.consumer_key != self.consumer_key
            or self.result.verifier.status is not W09ResultState.PASS
            or self.continual_learning.use_key != self.result.use_outcome.use_key
            or self.continual_learning.outcome_key != self.result.use_outcome.outcome_key
        ):
            raise W09V06Error("W09-07 cell result is not bound to learning delta")

    @property
    def cell_key(self) -> tuple[str, str]:
        """返回 carrier×consumer identity。"""
        return self.carrier_key, self.consumer_key

    def stable_key(self) -> tuple[int, ...]:
        """返回实际 result 与 continual delta 的联合 identity。"""
        return digest_value({
            "carrier": self.carrier_key,
            "consumer": self.consumer_key,
            "learning": list(self.continual_learning.stable_key()),
            "result": list(self.result.verifier.verifier_key),
            "window": self.window_ordinal,
        })


@dataclass(frozen=True)
class W09V06WindowReport:
    """一个真实零调用窗口的学习、held-out、rollback 与隔离报告。"""

    window_ordinal: int
    input_commitment: str
    threshold_key: tuple[int, ...]
    candidate_identity: str
    clone_identity: str
    host_before: W09V06HostSnapshot
    host_after: W09V06HostSnapshot
    clone_core_before: tuple[int, ...]
    clone_core_after: tuple[int, ...]
    clone_state_before: tuple[int, ...]
    clone_state_after: tuple[int, ...]
    metric_before: W09V06Metric
    metric_after: W09V06Metric
    erroneous_adoption_before_rollback: int
    erroneous_adoption_after_rollback: int
    data_candidate_count: int
    memory_write_count: int
    evidence_write_count: int
    use_write_count: int
    outcome_write_count: int
    rollback_event_count: int
    teacher_call_count: int
    api_call_count: int
    llm_call_count: int
    host_write_count: int
    resource_audit: W09ResourceAudit
    rollback_audit_sha256: str
    cells: tuple[W09V06CellEvidence, ...]
    status: str = W09_V06_WINDOW_STATUS

    def __post_init__(self) -> None:
        """要求 held-out 真改善、错误采用下降、Core/host 不变和九 cell 全通过。"""
        if type(self.window_ordinal) is not int or not 1 <= self.window_ordinal <= 3:
            raise W09V06Error("W09-07 window report ordinal is invalid")
        _sha(self.input_commitment, where="window report input")
        _key(self.threshold_key, where="window report threshold")
        _sha(self.candidate_identity, where="window report candidate")
        _sha(self.clone_identity, where="window report clone")
        if not isinstance(self.host_before, W09V06HostSnapshot) or self.host_before != self.host_after:
            raise W09V06Error("W09-07 window changed host state")
        for name in (
            "clone_core_before",
            "clone_core_after",
            "clone_state_before",
            "clone_state_after",
        ):
            _key(getattr(self, name), where=name)
        if self.clone_core_before != self.clone_core_after:
            raise W09V06Error("W09-07 window changed clone Core")
        if self.clone_state_before == self.clone_state_after:
            raise W09V06Error("W09-07 window did not change clone learning state")
        if not isinstance(self.metric_before, W09V06Metric) or not isinstance(self.metric_after, W09V06Metric):
            raise W09V06Error("W09-07 window metric type is invalid")
        if (
            self.metric_before.total != 9
            or self.metric_after.total != 9
            or self.metric_after.passed != 9
            or self.metric_after.permille <= self.metric_before.permille
        ):
            raise W09V06Error("W09-07 held-out did not improve")
        counters = (
            self.erroneous_adoption_before_rollback,
            self.erroneous_adoption_after_rollback,
            self.data_candidate_count,
            self.memory_write_count,
            self.evidence_write_count,
            self.use_write_count,
            self.outcome_write_count,
            self.rollback_event_count,
            self.teacher_call_count,
            self.api_call_count,
            self.llm_call_count,
            self.host_write_count,
        )
        if any(type(item) is not int or item < 0 for item in counters):
            raise W09V06Error("W09-07 window counter is invalid")
        if (
            self.erroneous_adoption_before_rollback <= 0
            or self.erroneous_adoption_after_rollback
            >= self.erroneous_adoption_before_rollback
            or self.erroneous_adoption_after_rollback != 0
            or min(
                self.data_candidate_count,
                self.memory_write_count,
                self.evidence_write_count,
                self.use_write_count,
                self.outcome_write_count,
                self.rollback_event_count,
            ) <= 0
        ):
            raise W09V06Error("W09-07 learning/rollback counters do not prove change")
        if any(item != 0 for item in (
            self.teacher_call_count,
            self.api_call_count,
            self.llm_call_count,
            self.host_write_count,
        )):
            raise W09V06Error("W09-07 window crossed live-call or host isolation")
        if not isinstance(self.resource_audit, W09ResourceAudit):
            raise W09V06Error("W09-07 window resource audit is invalid")
        _sha(self.rollback_audit_sha256, where="window rollback audit")
        if len(self.cells) != 9 or any(
            not isinstance(item, W09V06CellEvidence) for item in self.cells
        ):
            raise W09V06Error("W09-07 window does not expose nine actual cells")
        if any(item.window_ordinal != self.window_ordinal for item in self.cells):
            raise W09V06Error("W09-07 window report mixes cell windows")
        carriers = tuple(dict.fromkeys(item.carrier_key for item in self.cells))
        expected = tuple(
            (carrier, consumer)
            for carrier in carriers
            for consumer in W09_CONSUMER_KEYS
        )
        if len(carriers) != 3 or tuple(item.cell_key for item in self.cells) != expected:
            raise W09V06Error("W09-07 window cell coverage drifted")
        if self.status != W09_V06_WINDOW_STATUS:
            raise W09V06Error("W09-07 window public status is invalid")

    @property
    def window_identity(self) -> W09WindowIdentity:
        """投影为 typed weaning runtime 消费的真实窗口 identity。"""
        outputs = tuple(
            (
                consumer,
                w09_v06_commitment({
                    "consumer": consumer,
                    "outputs": [
                        list(item.result.use_outcome.outcome_key)
                        for item in self.cells
                        if item.consumer_key == consumer
                    ],
                    "window": self.window_ordinal,
                }),
            )
            for consumer in W09_CONSUMER_KEYS
        )
        return W09WindowIdentity(
            TeacherExitPhase.ZERO_CALL_WINDOW,
            self.window_ordinal,
            self.input_commitment,
            self.candidate_identity,
            self.teacher_call_count,
            outputs,
            self.resource_audit,
            self.rollback_audit_sha256,
        )

    def stable_key(self) -> tuple[int, ...]:
        """返回完整真实窗口报告 identity。"""
        return digest_value({
            "after": list(self.clone_state_after),
            "before": list(self.clone_state_before),
            "cells": [list(item.stable_key()) for item in self.cells],
            "clone": self.clone_identity,
            "input": self.input_commitment,
            "metric_after": self.metric_after.permille,
            "metric_before": self.metric_before.permille,
            "resource": list(self.resource_audit.stable_key()),
            "rollback": self.rollback_audit_sha256,
            "threshold": list(self.threshold_key),
            "window": self.window_ordinal,
        })


@dataclass(frozen=True)
class W09V06AblationReport:
    """从真实执行量测 Memory/Use/rollback/fixed-Core 消融击穿。"""

    component_key: str
    target_dimension_key: str
    memory_write_count: int
    use_write_count: int
    metric_improvement: int
    erroneous_adoption_before_rollback: int
    erroneous_adoption_after_rollback: int
    target_status: str
    unrelated_dimension_failure_count: int

    def __post_init__(self) -> None:
        """要求每种消融按实际量只击穿其预注册维度。"""
        if self.component_key not in W09_V06_ABLATION_KEYS:
            raise W09V06Error("W09-07 ablation component is invalid")
        expected_target = (
            W09_DIMENSION_KEYS[2]
            if self.component_key == "ROLLBACK_CONSUMER"
            else W09_DIMENSION_KEYS[4]
        )
        if self.target_dimension_key != expected_target:
            raise W09V06Error("W09-07 ablation target drifted")
        counters = (
            self.memory_write_count,
            self.use_write_count,
            self.metric_improvement,
            self.erroneous_adoption_before_rollback,
            self.erroneous_adoption_after_rollback,
            self.unrelated_dimension_failure_count,
        )
        if any(type(item) is not int or item < 0 for item in counters):
            raise W09V06Error("W09-07 ablation counter is invalid")
        if self.component_key == "NEW_MEMORY_WRITE" and (
            self.memory_write_count != 0 or self.metric_improvement != 0
        ):
            raise W09V06Error("W09-07 Memory ablation did not remove learning")
        if self.component_key == "USE_OUTCOME_CONSUMER" and (
            self.use_write_count != 0 or self.metric_improvement != 0
        ):
            raise W09V06Error("W09-07 Use ablation did not remove adoption")
        if self.component_key == "FIXED_CORE_REPLAY" and self.metric_improvement != 0:
            raise W09V06Error("W09-07 fixed Core replay still claims improvement")
        if self.component_key == "ROLLBACK_CONSUMER" and (
            self.erroneous_adoption_before_rollback <= 0
            or self.erroneous_adoption_after_rollback
            != self.erroneous_adoption_before_rollback
        ):
            raise W09V06Error("W09-07 rollback ablation did not preserve error adoption")
        if self.target_status != "FAIL" or self.unrelated_dimension_failure_count != 0:
            raise W09V06Error("W09-07 ablation is not an orthogonal failure")


@dataclass(frozen=True)
class W09V06Report:
    """一次 clone、三个窗口、27 cells 和 public NOT_FORMAL 总报告。"""

    protocol_key: tuple[int, ...]
    host_before: W09V06HostSnapshot
    host_after: W09V06HostSnapshot
    clone_receipt: W09CloneReceipt
    windows: tuple[W09V06WindowReport, ...]
    cells: tuple[W09V06CellEvidence, ...]
    clone_count: int
    teacher_call_count: int
    api_call_count: int
    llm_call_count: int
    host_write_count: int
    public_status: str
    pre_wean_language_learning_capability_evidenced: int
    language_capability_mastered: int
    language_readiness: int

    def __post_init__(self) -> None:
        """合取一次 clone、连续窗口、27 独立 cells 和未正式发布状态。"""
        _key(self.protocol_key, where="V06 protocol")
        if self.host_before != self.host_after:
            raise W09V06Error("W09-07 report changed host")
        if not isinstance(self.clone_receipt, W09CloneReceipt):
            raise W09V06Error("W09-07 clone receipt is invalid")
        if tuple(item.window_ordinal for item in self.windows) != (1, 2, 3):
            raise W09V06Error("W09-07 report windows are not contiguous")
        if any(not isinstance(item, W09V06WindowReport) for item in self.windows):
            raise W09V06Error("W09-07 window report type is invalid")
        flattened = tuple(
            cell for window in self.windows for cell in window.cells
        )
        if flattened != self.cells:
            raise W09V06Error("W09-07 aggregate cells do not match window evidence")
        if any(
            window.candidate_identity != self.clone_receipt.source_identity
            or window.clone_identity != self.clone_receipt.clone_identity
            or window.host_before != self.host_before
            or window.host_after != self.host_after
            or window.clone_core_before != self.host_before.core_state_key
            or window.clone_core_after != self.host_before.core_state_key
            for window in self.windows
        ):
            raise W09V06Error("W09-07 window/host/clone identity drifted")
        if any(
            current.clone_state_after != following.clone_state_before
            for current, following in zip(self.windows, self.windows[1:])
        ):
            raise W09V06Error("W09-07 windows are not one continuous clone history")
        expected_cells = tuple(
            (carrier, consumer)
            for carrier in W09_CARRIER_KEYS
            for consumer in W09_CONSUMER_KEYS
        )
        if (
            len(self.cells) != 27
            or tuple(item.cell_key for item in self.cells) != expected_cells
            or any(not isinstance(item, W09V06CellEvidence) for item in self.cells)
        ):
            raise W09V06Error("W09-07 report does not contain 27 actual cells")
        identities = tuple(
            key
            for item in self.cells
            for key in (
                item.continual_learning.before_state_key,
                item.continual_learning.after_state_key,
                item.continual_learning.source_evidence_key,
            )
        )
        if len(set(identities)) != 81:
            raise W09V06Error("W09-07 continual cells share learning identity")
        result_identities = tuple(
            key
            for item in self.cells
            for key in (
                item.result.request.request_key,
                item.result.choice.choice_key,
                item.result.use_outcome.use_key,
                item.result.use_outcome.outcome_key,
                item.result.verifier.verifier_key,
            )
        )
        if len(set(result_identities)) != 135:
            raise W09V06Error("W09-07 continual cells share result identity")
        if self.clone_count != 1:
            raise W09V06Error("W09-07 must create exactly one clone")
        if any(item != 0 for item in (
            self.teacher_call_count,
            self.api_call_count,
            self.llm_call_count,
            self.host_write_count,
        )):
            raise W09V06Error("W09-07 report crossed live-call or host isolation")
        if self.public_status != W09_V06_PUBLIC_STATUS:
            raise W09V06Error("W09-07 report was promoted before formal evaluation")
        if any(item != 0 for item in (
            self.pre_wean_language_learning_capability_evidenced,
            self.language_capability_mastered,
            self.language_readiness,
        )):
            raise W09V06Error("W09-07 public report published a formal state")

    @property
    def window_identities(self) -> tuple[W09WindowIdentity, ...]:
        """返回三个实际运行产生的 typed zero-call identity。"""
        return tuple(item.window_identity for item in self.windows)

    @property
    def consumer_difference_count(self) -> int:
        """返回由真实学习产生 before/after 差异的 U/R/G cell 数。"""
        return sum(
            item.continual_learning.before_state_key
            != item.continual_learning.after_state_key
            for item in self.cells
        )

    @property
    def directional_difference_counts(self) -> tuple[tuple[str, int], ...]:
        """逐 U/R/G 返回可归因学习差异数，不以综合均值代替方向。"""
        return tuple(
            (
                consumer,
                sum(
                    item.consumer_key == consumer
                    and item.continual_learning.before_state_key
                    != item.continual_learning.after_state_key
                    for item in self.cells
                ),
            )
            for consumer in W09_CONSUMER_KEYS
        )

    def stable_key(self) -> tuple[int, ...]:
        """返回 W09-07 public bounded canonical identity。"""
        return digest_value({
            "cells": [list(item.stable_key()) for item in self.cells],
            "clone": self.clone_receipt.clone_identity,
            "host": list(self.host_before.stable_key()),
            "protocol": list(self.protocol_key),
            "status": self.public_status,
            "windows": [list(item.stable_key()) for item in self.windows],
        })


__all__ = [
    "W09V06AblationReport",
    "W09V06CellEvidence",
    "W09V06Error",
    "W09V06HostSnapshot",
    "W09V06LearningRecord",
    "W09V06Metric",
    "W09V06Probe",
    "W09V06ProbeCriterion",
    "W09V06ProbeOwner",
    "W09V06Protocol",
    "W09V06Report",
    "W09V06WindowPlan",
    "W09V06WindowReport",
    "W09_V06_ABLATION_KEYS",
    "W09_V06_PROVENANCE_KIND",
    "W09_V06_PUBLIC_STATUS",
    "w09_v06_commitment",
    "w09_v06_key",
]
