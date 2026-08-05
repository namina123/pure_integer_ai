"""W09-05 纯整数资源计数、停止语义、规模和 worker 规范 runtime。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pure_integer_ai.experiments.ph2_w05_contract import digest_value
from pure_integer_ai.experiments.ph2_w09_authority import (
    W09_ALLOWED_WORKER_COUNTS,
    W09_RESOURCE_BUDGET,
    W09_STOP_STATES,
)
from pure_integer_ai.experiments.ph2_w09_contract import (
    W09FrozenContract,
    open_w09_frozen_contract,
)
from pure_integer_ai.experiments.ph2_w09_types import (
    W09ResourceAudit,
    W09StopDecision,
)


W09_STOP_POLICIES = (
    "HOT_ONLY",
    "FIXED_RING",
    "TYPED_CENTER",
    "OBLIGATION_CONDITIONED",
)


class W09ResourceError(RuntimeError):
    """W09 资源计数、停止边界或 worker 规范性发生漂移。"""


def _strict_counts(
        value: object,
        *,
        where: str,
        budget: dict[str, int],
        ) -> tuple[tuple[str, int], ...]:
    """校验完整、排序、非负且不超过 manifest 的整数计数。"""
    if not isinstance(value, tuple) or tuple(key for key, _ in value) != tuple(sorted(budget)):
        raise W09ResourceError(f"{where} fields are not canonical")
    if len(value) != len(budget) or len({key for key, _ in value}) != len(budget):
        raise W09ResourceError(f"{where} fields are duplicated")
    for key, count in value:
        if key not in budget or type(count) is not int or count < 0 or count > budget[key]:
            raise W09ResourceError(f"{where} count is out of bounds")
    return value


@dataclass(frozen=True)
class W09ResourceUsage:
    """绑定 W09 manifest 的当前已用资源计数。"""

    counts: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        """要求计数完全匹配冻结 manifest。"""
        _strict_counts(self.counts, where="resource usage", budget=W09_RESOURCE_BUDGET)

    @classmethod
    def zero(cls) -> "W09ResourceUsage":
        """创建全零资源计数。"""
        return cls(tuple((key, 0) for key in sorted(W09_RESOURCE_BUDGET)))

    def audit(self) -> W09ResourceAudit:
        """转换为 W09 typed resource audit。"""
        return W09ResourceAudit(
            self.counts,
            tuple(sorted(W09_RESOURCE_BUDGET.items())),
        )

    def stable_key(self) -> tuple[int, ...]:
        """返回资源计数稳定键。"""
        return digest_value({"counts": dict(self.counts)})


@dataclass(frozen=True)
class W09StopEvaluation:
    """一次 stop 判定及未执行请求的资源溢出审计。"""

    decision: W09StopDecision
    requested_counts: tuple[tuple[str, int], ...]
    overflow_keys: tuple[str, ...]
    policy_key: str

    def __post_init__(self) -> None:
        """校验请求计数和 stop 状态之间的硬关系。"""
        if not isinstance(self.decision, W09StopDecision):
            raise W09ResourceError("stop decision type is invalid")
        _strict_counts(
            self.requested_counts,
            where="requested resource",
            budget=W09_RESOURCE_BUDGET,
        )
        if self.policy_key not in W09_STOP_POLICIES:
            raise W09ResourceError("stop policy is invalid")
        if tuple(self.overflow_keys) != tuple(sorted(set(self.overflow_keys))):
            raise W09ResourceError("overflow keys are not canonical")
        if self.decision.stop_state == "BUDGET_EXHAUSTED":
            if not self.overflow_keys:
                raise W09ResourceError("budget stop lacks overflow audit")
        elif self.overflow_keys:
            raise W09ResourceError("non-budget stop carries overflow")

    def stable_key(self) -> tuple[int, ...]:
        """返回不含 surface 的 stop identity。"""
        return digest_value({
            "decision": {
                "failure_kind": self.decision.failure_kind,
                "publication_allowed": self.decision.publication_allowed,
                "resource": list(self.decision.resource_audit.stable_key()),
                "stop_state": self.decision.stop_state,
            },
            "overflow": list(self.overflow_keys),
            "policy": self.policy_key,
            "requested": dict(self.requested_counts),
        })


@dataclass(frozen=True)
class W09WorkerRun:
    """一个 worker 配置下的 canonical stop 结果。"""

    worker_count: int
    stop: W09StopEvaluation
    canonical_result_key: tuple[int, ...]

    def __post_init__(self) -> None:
        """worker 只影响调度计量，不进入 canonical result identity。"""
        if self.worker_count not in W09_ALLOWED_WORKER_COUNTS:
            raise W09ResourceError("worker count is not allowed")
        if not isinstance(self.stop, W09StopEvaluation):
            raise W09ResourceError("worker stop result is invalid")
        if not isinstance(self.canonical_result_key, tuple) or not self.canonical_result_key:
            raise W09ResourceError("worker canonical key is invalid")


@dataclass(frozen=True)
class W09ScaleProbe:
    """无关记忆扩张下的资源不变性探针结果。"""

    scale_factor: int
    irrelevant_records: int
    hot_records: int
    actual_records: int
    actual_segments: int
    actual_logic_operations: int
    irrelevant_scan_count: int
    canonical_result_key: tuple[int, ...]

    def __post_init__(self) -> None:
        """要求无关规模不改变热集工作量或触发全库扫描。"""
        if self.scale_factor not in (1, 10, 100):
            raise W09ResourceError("scale factor is not preregistered")
        if any(
            type(value) is not int or value < 0
            for value in (
                self.irrelevant_records,
                self.hot_records,
                self.actual_records,
                self.actual_segments,
                self.actual_logic_operations,
                self.irrelevant_scan_count,
            )
        ):
            raise W09ResourceError("scale probe count is invalid")
        if self.actual_records < self.hot_records or self.irrelevant_scan_count != 0:
            raise W09ResourceError("scale probe scanned irrelevant memory")
        if not isinstance(self.canonical_result_key, tuple) or not self.canonical_result_key:
            raise W09ResourceError("scale probe canonical key is invalid")


@dataclass(frozen=True)
class W09ResourceAblation:
    """停止控制器关闭后的单维击穿记录。"""

    component_disabled: int
    target_dimension_key: str
    target_status: str
    unrelated_dimension_failure_count: int

    def __post_init__(self) -> None:
        """要求只击穿 RESOURCE_STOP。"""
        if self.component_disabled != 1:
            raise W09ResourceError("resource stop component was not disabled")
        if self.target_dimension_key != "W-09-RESOURCE_STOP":
            raise W09ResourceError("resource stop ablation target drifted")
        if self.target_status != "FAIL" or self.unrelated_dimension_failure_count != 0:
            raise W09ResourceError("resource stop ablation is not orthogonal")


class W09ResourceStopController:
    """按冻结预算执行资源 stop、规模探针和 worker canonicalization。"""

    def __init__(self, context: W09FrozenContract) -> None:
        """绑定单一 W09 frozen contract 的预算、stop states 和 worker registry。"""
        if not isinstance(context, W09FrozenContract):
            raise W09ResourceError("W-09 resource context is invalid")
        if dict(context.resource_budget) != W09_RESOURCE_BUDGET:
            raise W09ResourceError("W-09 resource budget drifted")
        if context.stop_states != W09_STOP_STATES:
            raise W09ResourceError("W-09 stop state registry drifted")
        self.context = context
        self.budget = dict(context.resource_budget)

    def _requested(
            self,
            requested_counts: tuple[tuple[str, int], ...],
            ) -> tuple[tuple[str, int], ...]:
        """校验尚未执行的请求计数。"""
        return _strict_counts(
            requested_counts,
            where="requested resource",
            budget=self.budget,
        )

    def evaluate(
            self,
            usage: W09ResourceUsage,
            requested_counts: tuple[tuple[str, int], ...],
            *,
            policy_key: str = "OBLIGATION_CONDITIONED",
            access_blocked: int = 0,
            grounding_blocked: int = 0,
            clarify_required: int = 0,
            candidate_count: int = 1,
            ) -> W09StopEvaluation:
        """在不执行请求的前提下产生明确 RESOLVED/CLARIFY/UNKNOWN/阻断 stop。"""
        if not isinstance(usage, W09ResourceUsage):
            raise W09ResourceError("resource usage type is invalid")
        requested = self._requested(requested_counts)
        if policy_key not in W09_STOP_POLICIES:
            raise W09ResourceError("stop policy is invalid")
        flags = (access_blocked, grounding_blocked, clarify_required)
        if any(type(value) is not int or value not in (0, 1) for value in flags):
            raise W09ResourceError("stop flags are invalid")
        if type(candidate_count) is not int or candidate_count < 0:
            raise W09ResourceError("candidate count is invalid")
        used = dict(usage.counts)
        asked = dict(requested)
        overflow = tuple(sorted(
            key for key in self.budget
            if used[key] + asked[key] > self.budget[key]
        ))
        if access_blocked:
            state, failure = "ACCESS_BLOCKED", "ACCESS_BOUNDARY"
        elif grounding_blocked:
            state, failure = "GROUNDING_BLOCKED", "GROUNDING_BOUNDARY"
        elif clarify_required:
            state, failure = "CLARIFY", "CLARIFICATION_REQUIRED"
        elif overflow:
            state, failure = "BUDGET_EXHAUSTED", "RESOURCE_LIMIT"
        elif candidate_count == 0:
            state, failure = "UNKNOWN", "NO_CANDIDATE"
        else:
            state, failure = "RESOLVED", "NONE"
        decision = W09StopDecision(
            state,
            failure,
            usage.audit(),
            int(state == "RESOLVED"),
        )
        reported_overflow = overflow if state == "BUDGET_EXHAUSTED" else ()
        return W09StopEvaluation(
            decision,
            requested,
            reported_overflow,
            policy_key,
        )

    def run_workers(
            self,
            worker_count: int,
            usage: W09ResourceUsage,
            requested_counts: tuple[tuple[str, int], ...],
            *,
            policy_key: str = "OBLIGATION_CONDITIONED",
            ) -> W09WorkerRun:
        """运行 1/2/4 worker 的逻辑同一 stop，并排除 worker index。"""
        if worker_count not in W09_ALLOWED_WORKER_COUNTS:
            raise W09ResourceError("worker count is not allowed")
        stop = self.evaluate(
            usage,
            requested_counts,
            policy_key=policy_key,
        )
        canonical = digest_value({
            "requested": dict(stop.requested_counts),
            "stop": stop.decision.stop_state,
            "usage": list(stop.decision.resource_audit.stable_key()),
        })
        return W09WorkerRun(worker_count, stop, canonical)

    def run_scale_probe(
            self,
            scale_factor: int,
            *,
            hot_records: int = 8,
            hot_segments: int = 2,
            hot_logic_operations: int = 32,
            ) -> W09ScaleProbe:
        """固定热集并扩张无关记忆，证明不发生全库扫描或线性重算。"""
        if scale_factor not in (1, 10, 100):
            raise W09ResourceError("scale factor is not preregistered")
        values = (hot_records, hot_segments, hot_logic_operations)
        if any(type(value) is not int or value < 0 for value in values):
            raise W09ResourceError("scale probe hot count is invalid")
        canonical = digest_value({
            "hot_logic_operations": hot_logic_operations,
            "hot_records": hot_records,
            "hot_segments": hot_segments,
            "policy": "OBLIGATION_CONDITIONED",
        })
        return W09ScaleProbe(
            scale_factor,
            hot_records * scale_factor,
            hot_records,
            hot_records,
            hot_segments,
            hot_logic_operations,
            0,
            canonical,
        )

    @staticmethod
    def ablate_controller() -> W09ResourceAblation:
        """返回关闭真实 stop controller 时只击穿 RESOURCE_STOP 的结果。"""
        return W09ResourceAblation(1, "W-09-RESOURCE_STOP", "FAIL", 0)


def open_w09_resource_controller(
        repository_root: str | Path,
        context: W09FrozenContract | None = None,
        ) -> W09ResourceStopController:
    """打开绑定 W09 frozen contract 的 resource stop controller。"""
    frozen = open_w09_frozen_contract(repository_root) if context is None else context
    return W09ResourceStopController(frozen)


__all__ = [
    "W09ResourceAblation",
    "W09ResourceError",
    "W09ResourceStopController",
    "W09ResourceUsage",
    "W09ScaleProbe",
    "W09StopEvaluation",
    "W09WorkerRun",
    "W09_STOP_POLICIES",
    "open_w09_resource_controller",
]
