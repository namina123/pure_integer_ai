"""typed language generation 的 episode 与多维 reward 信号。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.identity import SourceRef
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.experiments.generation_production_runtime import (
    ProductionGenerationRun,
)
from pure_integer_ai.experiments.evaluation_protocol import ProtocolKey
from pure_integer_ai.experiments.verification_orchestration import (
    APPLICABILITY_APPLICABLE,
    APPLICABILITY_NOT_APPLICABLE,
    APPLICABILITY_UNKNOWN,
    VERDICT_CONFLICTED,
    VERDICT_REFUTE,
    VERDICT_SUPPORT,
    VERDICT_UNKNOWN,
    VerificationReport,
    VerificationResult,
)

_VALID_APPLICABILITY = frozenset({
    APPLICABILITY_NOT_APPLICABLE,
    APPLICABILITY_APPLICABLE,
    APPLICABILITY_UNKNOWN,
})
_VALID_VERDICTS = frozenset({
    VERDICT_SUPPORT,
    VERDICT_REFUTE,
    VERDICT_UNKNOWN,
    VERDICT_CONFLICTED,
})


def _packed(key: tuple[int, ...]) -> tuple[int, ...]:
    """为可变长稳定键增加长度边界。"""
    return len(key), *key


@dataclass(frozen=True)
class TypedLanguageRewardSignal:
    """单一 verifier 的 applicability、verdict、claim 和理由信号。"""

    dimension: ProtocolKey
    verifier: ProtocolKey
    applicability: int
    verdict: int
    claim_keys: tuple[tuple[int, ...], ...]
    proposed_effect_keys: tuple[tuple[int, ...], ...]
    detail: tuple[int, ...]
    source: SourceRef | None
    scope: ScopeIdentity | None
    operational_failure: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.dimension, ProtocolKey):
            raise TypeError("typed reward dimension 必须是 ProtocolKey")
        if not isinstance(self.verifier, ProtocolKey):
            raise TypeError("typed reward verifier 必须是 ProtocolKey")
        if self.applicability not in _VALID_APPLICABILITY:
            raise ValueError("typed reward applicability 未注册")
        if self.verdict not in _VALID_VERDICTS:
            raise ValueError("typed reward verdict 未注册")
        if not isinstance(self.claim_keys, tuple) or any(
                not isinstance(item, tuple) or not item
                or any(type(value) is not int for value in item)
                for item in self.claim_keys):
            raise ValueError("typed reward claim_keys 必须是非空严格整数键 tuple")
        if not isinstance(self.proposed_effect_keys, tuple) or any(
                not isinstance(item, tuple) or not item
                or any(type(value) is not int for value in item)
                for item in self.proposed_effect_keys):
            raise ValueError("typed reward effect keys 类型错误")
        if not isinstance(self.detail, tuple) or any(
                type(item) is not int for item in self.detail):
            raise ValueError("typed reward detail 必须是严格整数 tuple")
        if self.source is not None and not isinstance(self.source, SourceRef):
            raise TypeError("typed reward source 类型错误")
        if self.scope is not None and not isinstance(self.scope, ScopeIdentity):
            raise TypeError("typed reward scope 类型错误")
        if self.operational_failure is not None and (
                not isinstance(self.operational_failure, str)
                or not self.operational_failure):
            raise ValueError("typed reward operational_failure 类型错误")
        assert_int(
            self.applicability,
            self.verdict,
            *(value for key in self.claim_keys for value in key),
            *(value for key in self.proposed_effect_keys for value in key),
            *self.detail,
            _where="TypedLanguageRewardSignal",
        )

    @classmethod
    def from_verification(
            cls, result: VerificationResult,
            ) -> "TypedLanguageRewardSignal":
        """从分维结果逐字段复制，不合并或重解释 verdict。"""
        if not isinstance(result, VerificationResult):
            raise TypeError("typed reward 只能投影 VerificationResult")
        return cls(
            result.dimension,
            result.verifier,
            result.applicability,
            result.verdict,
            result.claim_keys,
            tuple(item.stable_key() for item in result.proposed_effects),
            result.detail,
            result.source,
            result.scope,
            result.operational_failure,
        )

    def stable_key(self) -> tuple[int, ...]:
        """返回单维信号的完整稳定键。"""
        result = [
            *_packed(self.dimension.stable_key()),
            *_packed(self.verifier.stable_key()),
            self.applicability,
            self.verdict,
            len(self.claim_keys),
        ]
        for key in self.claim_keys:
            result.extend(_packed(key))
        result.append(len(self.proposed_effect_keys))
        for key in self.proposed_effect_keys:
            result.extend(_packed(key))
        result.extend(_packed(self.detail))
        source_key = () if self.source is None else self.source.stable_key()
        scope_key = () if self.scope is None else self.scope.stable_key()
        failure_key = (() if self.operational_failure is None else tuple(
            ord(item) for item in self.operational_failure))
        result.extend(_packed(source_key))
        result.extend(_packed(scope_key))
        result.extend(_packed(failure_key))
        return tuple(result)


@dataclass(frozen=True)
class TypedLanguageEpisode:
    """一次 typed generation 及全部适用 verifier 的多维信号聚合。"""

    round_id: int
    source: SourceRef
    scope: ScopeIdentity
    production: ProductionGenerationRun
    signals: tuple[TypedLanguageRewardSignal, ...]
    read_only: bool
    supplemental_verification: VerificationReport | None = None

    def __post_init__(self) -> None:
        assert_int(self.round_id, _where="TypedLanguageEpisode.round_id")
        if type(self.round_id) is not int or self.round_id < 0:
            raise ValueError("typed language episode round_id 必须为非负严格整数")
        if not isinstance(self.source, SourceRef):
            raise TypeError("typed language episode source 类型错误")
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("typed language episode scope 类型错误")
        if not isinstance(self.production, ProductionGenerationRun):
            raise TypeError("typed language episode production 类型错误")
        if not isinstance(self.signals, tuple) or any(
                not isinstance(item, TypedLanguageRewardSignal)
                for item in self.signals):
            raise TypeError("typed language episode signals 类型错误")
        if len({item.dimension for item in self.signals}) != len(self.signals):
            raise ValueError("typed language episode 不得重复 reward dimension")
        if type(self.read_only) is not bool:
            raise TypeError("typed language episode read_only 必须是严格 bool")
        supplemental = self.supplemental_verification
        if supplemental is not None:
            if not isinstance(supplemental, VerificationReport):
                raise TypeError("typed language supplemental verification 类型错误")
            if supplemental.read_only != self.read_only:
                raise ValueError("typed language supplemental report 只读状态漂移")
            if any(item.committed_effects for item in supplemental.results):
                raise ValueError("typed language supplemental report 不得提交 effect")
            for result in supplemental.results:
                if result.source is not None and result.source != self.source:
                    raise ValueError("typed language supplemental verifier 来源漂移")
                if result.scope is not None and result.scope != self.scope:
                    raise ValueError("typed language supplemental verifier scope 漂移")
        request = self.production.decision.request
        if request is not None:
            if request.goal.source != self.source:
                raise ValueError("typed language episode 替换了请求知识来源")
            if request.goal.scope.parent != self.scope:
                raise ValueError("typed language episode 请求未绑定当前 episode scope")
        postcheck = self.production.postcheck
        results = [] if postcheck is None else list(postcheck.report.results)
        if supplemental is not None:
            results.extend(supplemental.results)
        expected = tuple(sorted(
            (TypedLanguageRewardSignal.from_verification(item)
             for item in results),
            key=lambda item: (
                item.dimension.stable_key(),
                item.verifier.stable_key(),
            ),
        ))
        if self.signals != expected:
            raise ValueError("typed reward signals 未逐点覆盖同次全部 verifier 结果")

    @classmethod
    def from_production(
            cls,
            round_id: int,
            source: SourceRef,
            scope: ScopeIdentity,
            production: ProductionGenerationRun,
            *,
            read_only: bool,
            supplemental_verification: VerificationReport | None = None,
            ) -> "TypedLanguageEpisode":
        """从同次 production、G-04 和适用 verifier 构造分维 episode。"""
        results = (
            []
            if production.postcheck is None
            else list(production.postcheck.report.results)
        )
        if supplemental_verification is not None:
            if not isinstance(supplemental_verification, VerificationReport):
                raise TypeError("typed language supplemental verification 类型错误")
            results.extend(supplemental_verification.results)
        signals = tuple(sorted(
            (TypedLanguageRewardSignal.from_verification(item)
             for item in results),
            key=lambda item: (
                item.dimension.stable_key(),
                item.verifier.stable_key(),
            ),
        ))
        return cls(
            round_id,
            source,
            scope,
            production,
            signals,
            read_only,
            supplemental_verification,
        )

    @property
    def generation_complete(self) -> bool:
        """返回 G-00 至 G-03 是否完整，不替代各 G-04 verdict。"""
        return self.production.complete

    @property
    def postcheck_complete(self) -> bool | None:
        """返回 G-04 聚合完成状态，仅供流程观察，不作为标量 reward。"""
        return self.production.postcheck_complete

    def stable_key(self) -> tuple[int, ...]:
        """返回来源、scope、production 和逐维信号的完整键。"""
        result = [
            self.round_id,
            *_packed(self.source.stable_key()),
            *_packed(self.scope.stable_key()),
            *_packed(self.production.stable_key()),
            int(self.read_only),
            len(self.signals),
        ]
        for signal in self.signals:
            result.extend(_packed(signal.stable_key()))
        supplemental = self.supplemental_verification
        if supplemental is None:
            result.extend((0,))
        else:
            result.extend((1, int(supplemental.read_only), len(
                supplemental.results)))
            for verification in supplemental.results:
                signal = TypedLanguageRewardSignal.from_verification(
                    verification)
                result.extend(_packed(signal.stable_key()))
        return tuple(result)


__all__ = [
    "TypedLanguageEpisode",
    "TypedLanguageRewardSignal",
]
