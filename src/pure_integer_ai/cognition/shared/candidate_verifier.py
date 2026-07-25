"""H-05 不读取候选图的独立 typed prediction verifier。

revealed observation 只显式列出支持目标和定向反驳目标；未列出的目标保持 unknown，
因此图缺边、未覆盖类别或另一个已知目标都不会被自动解释为 false。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.evidence_candidate import (
    CandidatePrediction,
    CandidateVerification,
)
from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EVIDENCE_UNKNOWN,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONCEPT,
    ObjectIdentity,
    SourceRef,
)
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.crosscut.guards.int_blocker import assert_int


def _strict_key(value, *, where: str) -> tuple[int, ...]:
    """校验 verifier 协议和 reveal trace 使用的开放整数键。"""
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{where} 必须是非空整数 tuple")
    assert_int(*value, _where=where)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{where} 必须使用严格整数")
    return value


@dataclass(frozen=True)
class IndependentVerifierProtocol:
    """注入 verifier 身份、版本和三态 reason，不解释领域关系。"""

    authority: ObjectIdentity
    authority_version: tuple[int, ...]
    support_reason_key: tuple[int, ...]
    refute_reason_key: tuple[int, ...]
    unknown_reason_key: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.authority, ObjectIdentity):
            raise TypeError("verifier authority 必须是 ObjectIdentity")
        if self.authority.object_kind != OBJECT_CONCEPT:
            raise ValueError("verifier authority 必须是一等 Concept")
        for name, value in (
                ("authority_version", self.authority_version),
                ("support_reason_key", self.support_reason_key),
                ("refute_reason_key", self.refute_reason_key),
                ("unknown_reason_key", self.unknown_reason_key)):
            _strict_key(value, where=f"IndependentVerifierProtocol.{name}")
        if len({
                self.support_reason_key,
                self.refute_reason_key,
                self.unknown_reason_key}) != 3:
            raise ValueError("support/refute/unknown reason 必须互不相同")


@dataclass(frozen=True)
class RevealedObjectObservation:
    """预测冻结后由独立来源揭示的显式支持、反驳和核验 trace。"""

    observation: SourceRef
    scope: ScopeIdentity
    event_key: tuple[int, ...]
    verifier_source: SourceRef
    supported_targets: tuple[ObjectIdentity, ...] = ()
    refuted_targets: tuple[ObjectIdentity, ...] = ()
    trace: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.observation, SourceRef):
            raise TypeError("reveal observation 必须是 SourceRef")
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("reveal scope 必须是 ScopeIdentity")
        if self.scope.source != self.observation:
            raise ValueError("reveal scope 必须指向同一 observation")
        _strict_key(self.event_key, where="RevealedObjectObservation.event_key")
        if not isinstance(self.verifier_source, SourceRef):
            raise TypeError("verifier_source 必须是 SourceRef")
        for name, targets in (
                ("supported_targets", self.supported_targets),
                ("refuted_targets", self.refuted_targets)):
            if not isinstance(targets, tuple):
                raise TypeError(f"{name} 必须是 ObjectIdentity tuple")
            if any(not isinstance(item, ObjectIdentity) for item in targets):
                raise TypeError(f"{name} 只能包含 ObjectIdentity")
            if len(set(targets)) != len(targets):
                raise ValueError(f"{name} 不得重复目标")
        if set(self.supported_targets) & set(self.refuted_targets):
            raise ValueError("同一 reveal 不得同时支持和反驳同一目标")
        if not isinstance(self.trace, tuple):
            raise TypeError("reveal trace 必须是整数 tuple")
        assert_int(*self.trace, _where="RevealedObjectObservation.trace")
        if any(type(item) is not int for item in self.trace):
            raise ValueError("reveal trace 必须使用严格整数")


class IndependentObjectVerifier:
    """只比较冻结 prediction 与显式 reveal，不持有图、候选 owner 或 legacy 表。"""

    def __init__(self, protocol: IndependentVerifierProtocol) -> None:
        if not isinstance(protocol, IndependentVerifierProtocol):
            raise TypeError("protocol 必须是 IndependentVerifierProtocol")
        self.protocol = protocol

    def verify(
            self, prediction: CandidatePrediction,
            revealed: RevealedObjectObservation) -> CandidateVerification:
        """按显式支持/反驳集合返回三态核验，未覆盖目标保持 unknown。"""
        if not isinstance(prediction, CandidatePrediction):
            raise TypeError("prediction 必须是 CandidatePrediction")
        if not isinstance(revealed, RevealedObjectObservation):
            raise TypeError("revealed 必须是 RevealedObjectObservation")
        if (prediction.observation != revealed.observation
                or prediction.scope != revealed.scope
                or prediction.event_key != revealed.event_key):
            raise ValueError("reveal 与冻结 prediction 的来源、scope 或事件不一致")
        if prediction.predicted in revealed.supported_targets:
            stance = EVIDENCE_SUPPORT
            reason = self.protocol.support_reason_key
        elif prediction.predicted in revealed.refuted_targets:
            stance = EVIDENCE_REFUTE
            reason = self.protocol.refute_reason_key
        else:
            stance = EVIDENCE_UNKNOWN
            reason = self.protocol.unknown_reason_key
        return CandidateVerification(
            stance,
            reason,
            revealed.verifier_source,
            self.protocol.authority,
            self.protocol.authority_version,
            revealed.trace,
        )


__all__ = [
    "IndependentObjectVerifier",
    "IndependentVerifierProtocol",
    "RevealedObjectObservation",
]
