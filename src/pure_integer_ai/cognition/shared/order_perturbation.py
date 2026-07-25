"""把 H-02A 完整扰动 trace 转换为 H-06 aggregate Hypothesis Evidence。

H-02A 通用 engine 的候选绑定单一观察来源，而 H-06 候选绑定课程 manifest；本 adapter
不放宽前者的来源隔离。它核验 trace 与真实顺序观察同源且原对象一致，再把完整 trace
作为 verifier 详情的一部分交给 H-06 owner ledger。
"""
from __future__ import annotations

from typing import Protocol

from pure_integer_ai.cognition.shared.hypothesis import EVIDENCE_REFUTE
from pure_integer_ai.cognition.shared.order_hypothesis import (
    OrderAssessment,
    OrderEvidenceResult,
    OrderHypothesisEngine,
    OrderObservation,
    OrderPattern,
)
from pure_integer_ai.cognition.shared.perturbation import PerturbationTrace

_ORDER_PERTURBATION_DETAIL_VERSION = 1


class OrderPerturbationVerifier(Protocol):
    """领域 verifier 判定一个完整扰动是否支持、反驳或无法裁决顺序模式。"""

    def __call__(
            self, pattern: OrderPattern,
            observation: OrderObservation,
            trace: PerturbationTrace,
            ) -> OrderAssessment: ...


class OrderPerturbationAdapter:
    """在 H-06 来源观察与 H-02A trace 之间执行无语义猜测的完整性桥接。"""

    def __init__(self, engine: OrderHypothesisEngine) -> None:
        """绑定顺序 Hypothesis owner；adapter 自身不另建 Evidence ledger。"""
        if not isinstance(engine, OrderHypothesisEngine):
            raise TypeError("engine 必须是 OrderHypothesisEngine")
        self.engine = engine

    def evaluate(
            self, pattern: OrderPattern,
            observation: OrderObservation,
            trace: PerturbationTrace,
            verifier: OrderPerturbationVerifier, *,
            timestamp_seq: int,
            supersedes_evidence_id: int = 0,
            ) -> OrderEvidenceResult:
        """核验来源和原对象后，把完整 trace 作为 H-06 定向 Evidence 详情累计。"""
        if not isinstance(pattern, OrderPattern):
            raise TypeError("pattern 必须是 OrderPattern")
        if not isinstance(observation, OrderObservation):
            raise TypeError("observation 必须是 OrderObservation")
        if not isinstance(trace, PerturbationTrace):
            raise TypeError("trace 必须是 PerturbationTrace")
        if trace.source != observation.source or trace.scope != observation.scope:
            raise ValueError("顺序扰动 trace 必须与观察使用同一来源和 scope")
        if trace.original != (
                observation.first_occurrence,
                observation.second_occurrence):
            raise ValueError("顺序扰动 trace.original 与观察的规范 slot 对象不一致")
        if not callable(verifier):
            raise TypeError("verifier 必须可调用")
        assessment = verifier(pattern, observation, trace)
        if not isinstance(assessment, OrderAssessment):
            raise TypeError("verifier 必须返回 OrderAssessment")
        if trace.is_duplicate_diagnostic and assessment.stance == EVIDENCE_REFUTE:
            raise ValueError("同源重复诊断不得作为顺序反例")
        trace_key = trace.stable_key()
        detail = (
            _ORDER_PERTURBATION_DETAIL_VERSION,
            len(assessment.detail_key),
            *assessment.detail_key,
            len(trace_key),
            *trace_key,
        )
        wrapped = OrderAssessment(assessment.stance, detail)
        return self.engine.accumulate(
            pattern,
            observation,
            lambda _pattern, _observation: wrapped,
            timestamp_seq=timestamp_seq,
            supersedes_evidence_id=supersedes_evidence_id,
        )


__all__ = [
    "OrderPerturbationAdapter",
    "OrderPerturbationVerifier",
]
