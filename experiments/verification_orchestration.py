"""开放、多维且无综合标量的 verifier 编排协议。"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.cognition.shared.types import SourceRef
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.experiments.evaluation_protocol import ProtocolKey

APPLICABILITY_NOT_APPLICABLE = 0
APPLICABILITY_APPLICABLE = 1
APPLICABILITY_UNKNOWN = 2

VERDICT_SUPPORT = 1
VERDICT_REFUTE = 2
VERDICT_UNKNOWN = 3
VERDICT_CONFLICTED = 4

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


class VerificationProtocolError(RuntimeError):
    '''verifier 注册、结果或 effect 违反多维隔离协议。'''


def _strict_key(value: tuple[int, ...], *, where: str) -> tuple[int, ...]:
    '''核验并返回可直接比较的完整整数键。'''
    if not isinstance(value, tuple) or not value:
        raise ValueError(f'{where} 必须是非空整数 tuple')
    assert_int(*value, _where=where)
    if any(type(component) is not int for component in value):
        raise TypeError(f'{where} 必须只含严格 int')
    return value


@dataclass(frozen=True)
class VerificationEffect:
    '''声明一个 verifier 准备或已经写入的分维对象引用。'''

    dimension: ProtocolKey
    target_kind: ProtocolKey
    target_key: tuple[int, ...]

    def __post_init__(self) -> None:
        '''核验 effect 使用完整注入键且不依赖运行时对象地址。'''
        if not isinstance(self.dimension, ProtocolKey):
            raise TypeError('effect.dimension 必须是 ProtocolKey')
        if not isinstance(self.target_kind, ProtocolKey):
            raise TypeError('effect.target_kind 必须是 ProtocolKey')
        _strict_key(self.target_key, where='effect.target_key')

    def stable_key(self) -> tuple[int, ...]:
        '''返回不依赖运行时对象地址的完整 effect 键。'''
        dimension = self.dimension.stable_key()
        target_kind = self.target_kind.stable_key()
        return (
            len(dimension),
            *dimension,
            len(target_kind),
            *target_kind,
            len(self.target_key),
            *self.target_key,
        )


def _strict_effect_tuple(
        value: tuple[VerificationEffect, ...], *,
        where: str,
        ) -> tuple[tuple[int, ...], ...]:
    '''核验 effect 容器、元素类型和完整键均严格且不重复。'''
    if not isinstance(value, tuple):
        raise TypeError(f'{where} 必须是 tuple')
    keys = []
    for index, effect in enumerate(value):
        if not isinstance(effect, VerificationEffect):
            raise TypeError(
                f'{where}[{index}] 必须是 VerificationEffect')
        keys.append(effect.stable_key())
    if len(set(keys)) != len(keys):
        raise ValueError(f'{where} 不得重复')
    return tuple(keys)


@dataclass(frozen=True)
class VerificationEvaluation:
    '''一个 verifier 的纯评估结果，尚未提交长期状态。'''

    verdict: int
    claim_keys: tuple[tuple[int, ...], ...] = ()
    proposed_effects: tuple[VerificationEffect, ...] = ()
    detail: tuple[int, ...] = ()
    source: SourceRef | None = None
    scope: ScopeIdentity | None = None
    artifact: Any = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        '''核验 verdict、claim、effect 和来源身份。'''
        assert_int(self.verdict, _where='VerificationEvaluation.verdict')
        if self.verdict not in _VALID_VERDICTS:
            raise ValueError('verification verdict 非法')
        if not isinstance(self.claim_keys, tuple):
            raise TypeError('claim_keys 必须是 tuple')
        for index, claim_key in enumerate(self.claim_keys):
            _strict_key(claim_key, where=f'claim_keys[{index}]')
        if len(set(self.claim_keys)) != len(self.claim_keys):
            raise ValueError('同一 verifier 结果不得重复 claim key')
        _strict_effect_tuple(
            self.proposed_effects,
            where='proposed_effects',
        )
        if not isinstance(self.detail, tuple):
            raise TypeError('verification detail 必须是 tuple')
        if self.detail:
            _strict_key(self.detail, where='verification detail')
        if self.source is not None and not isinstance(self.source, SourceRef):
            raise TypeError('verification source 必须是 SourceRef')
        if self.scope is not None and not isinstance(
                self.scope, ScopeIdentity):
            raise TypeError('verification scope 必须是 ScopeIdentity')


VerificationApplicability = Callable[[Any], bool]
VerificationEvaluator = Callable[[Any], VerificationEvaluation]
VerificationCommitter = Callable[
    [VerificationEvaluation],
    Sequence[VerificationEffect],
]


@dataclass(frozen=True)
class VerifierRegistration:
    '''把一个开放 verifier 绑定到独立维度和允许写入的目标类型。'''

    dimension: ProtocolKey
    verifier: ProtocolKey
    applies: VerificationApplicability
    evaluate: VerificationEvaluator
    allowed_target_kinds: tuple[ProtocolKey, ...] = ()
    commit: VerificationCommitter | None = None

    def __post_init__(self) -> None:
        '''核验 registration 的开放键、回调和目标类型集合。'''
        if not isinstance(self.dimension, ProtocolKey):
            raise TypeError('registration.dimension 必须是 ProtocolKey')
        if not isinstance(self.verifier, ProtocolKey):
            raise TypeError('registration.verifier 必须是 ProtocolKey')
        if not callable(self.applies) or not callable(self.evaluate):
            raise TypeError('registration applies/evaluate 必须可调用')
        if self.commit is not None and not callable(self.commit):
            raise TypeError('registration.commit 必须可调用')
        if not isinstance(self.allowed_target_kinds, tuple):
            raise TypeError('allowed_target_kinds 必须是 tuple')
        keys = []
        for target_kind in self.allowed_target_kinds:
            if not isinstance(target_kind, ProtocolKey):
                raise TypeError(
                    'allowed_target_kinds 必须只含 ProtocolKey')
            keys.append(target_kind.stable_key())
        if len(set(keys)) != len(keys):
            raise ValueError('allowed_target_kinds 不得重复')

    def stable_key(self) -> tuple[int, ...]:
        '''返回不受注册输入顺序影响的排序键。'''
        dimension = self.dimension.stable_key()
        verifier = self.verifier.stable_key()
        return (
            len(dimension),
            *dimension,
            len(verifier),
            *verifier,
        )


@dataclass(frozen=True)
class VerificationResult:
    '''保存单个 verifier 的适用性、verdict、归因和运行失败。'''

    dimension: ProtocolKey
    verifier: ProtocolKey
    applicability: int
    verdict: int
    claim_keys: tuple[tuple[int, ...], ...] = ()
    proposed_effects: tuple[VerificationEffect, ...] = ()
    committed_effects: tuple[VerificationEffect, ...] = ()
    detail: tuple[int, ...] = ()
    source: SourceRef | None = None
    scope: ScopeIdentity | None = None
    operational_failure: str | None = None
    artifact: Any = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        '''保持适用性、认识论 verdict 和运行失败三者正交。'''
        if not isinstance(self.dimension, ProtocolKey):
            raise TypeError('result.dimension 必须是 ProtocolKey')
        if not isinstance(self.verifier, ProtocolKey):
            raise TypeError('result.verifier 必须是 ProtocolKey')
        assert_int(
            self.applicability,
            self.verdict,
            _where='VerificationResult',
        )
        if self.applicability not in _VALID_APPLICABILITY:
            raise ValueError('verification applicability 非法')
        if self.verdict not in _VALID_VERDICTS:
            raise ValueError('verification verdict 非法')
        if not isinstance(self.claim_keys, tuple):
            raise TypeError('result.claim_keys 必须是 tuple')
        for index, claim_key in enumerate(self.claim_keys):
            _strict_key(claim_key, where=f'result.claim_keys[{index}]')
        if len(set(self.claim_keys)) != len(self.claim_keys):
            raise ValueError('result.claim_keys 不得重复')
        proposed_keys = _strict_effect_tuple(
            self.proposed_effects,
            where='result.proposed_effects',
        )
        committed_keys = _strict_effect_tuple(
            self.committed_effects,
            where='result.committed_effects',
        )
        if not isinstance(self.detail, tuple):
            raise TypeError('result.detail 必须是 tuple')
        if self.detail:
            _strict_key(self.detail, where='result.detail')
        if self.source is not None and not isinstance(self.source, SourceRef):
            raise TypeError('result.source 必须是 SourceRef')
        if self.scope is not None and not isinstance(
                self.scope, ScopeIdentity):
            raise TypeError('result.scope 必须是 ScopeIdentity')
        if self.applicability != APPLICABILITY_APPLICABLE:
            if self.claim_keys or self.proposed_effects or self.committed_effects:
                raise ValueError(
                    '非 applicable 结果不得携带 claim 或 effect')
            if self.verdict != VERDICT_UNKNOWN:
                raise ValueError(
                    '非 applicable 结果只能保持 unknown verdict')
        proposed = set(proposed_keys)
        committed = set(committed_keys)
        if not committed.issubset(proposed):
            raise ValueError(
                'committed effect 必须来自 proposed effect')
        for effect in (*self.proposed_effects, *self.committed_effects):
            if effect.dimension != self.dimension:
                raise ValueError(
                    'effect 不得跨 verification dimension')
        if self.operational_failure is not None:
            if not isinstance(self.operational_failure, str):
                raise TypeError('operational_failure 必须是 str')
            if not self.operational_failure:
                raise ValueError('operational_failure 不得为空')

    def verdict_key(self) -> tuple[Any, ...]:
        '''返回不含提交副作用的训练/只读评测比较键。'''
        return (
            self.dimension.stable_key(),
            self.verifier.stable_key(),
            self.applicability,
            self.verdict,
            self.claim_keys,
            tuple(
                effect.stable_key()
                for effect in self.proposed_effects
            ),
            self.detail,
            None if self.source is None else self.source.stable_key(),
        )


@dataclass(frozen=True)
class VerificationReport:
    '''保存全部注册 verifier 的分维结果，不提供综合 score 或 reward。'''

    read_only: bool
    results: tuple[VerificationResult, ...]

    def __post_init__(self) -> None:
        '''核验结果唯一、有序，并锁定只读零提交。'''
        if type(self.read_only) is not bool:
            raise TypeError('VerificationReport.read_only 必须是 bool')
        if not isinstance(self.results, tuple):
            raise TypeError('VerificationReport.results 必须是 tuple')
        keys = []
        for index, result in enumerate(self.results):
            if not isinstance(result, VerificationResult):
                raise TypeError(
                    f'VerificationReport.results[{index}] '
                    '必须是 VerificationResult')
            dimension = result.dimension.stable_key()
            verifier = result.verifier.stable_key()
            keys.append((
                len(dimension),
                *dimension,
                len(verifier),
                *verifier,
            ))
        if len(set(keys)) != len(keys):
            raise ValueError(
                'VerificationReport 不得重复 dimension/verifier')
        if keys != sorted(keys):
            raise ValueError(
                'VerificationReport.results 必须按完整键排序')
        if self.read_only and any(
                result.committed_effects for result in self.results):
            raise ValueError(
                '只读 verification report 不得含 committed effect')

    def applicable_results(self) -> tuple[VerificationResult, ...]:
        '''返回已确认适用的全部 verifier 结果。'''
        return tuple(
            result for result in self.results
            if result.applicability == APPLICABILITY_APPLICABLE
        )

    def dimension_results(
            self,
            dimension: ProtocolKey,
            ) -> tuple[VerificationResult, ...]:
        '''返回一个注入维度下的全部 verifier 结果。'''
        if not isinstance(dimension, ProtocolKey):
            raise TypeError('dimension 必须是 ProtocolKey')
        return tuple(
            result for result in self.results
            if result.dimension == dimension
        )

    def verdict_key(self) -> tuple[tuple[Any, ...], ...]:
        '''返回可比较训练与只读评测结果的分维 verdict 集。'''
        return tuple(result.verdict_key() for result in self.results)

    def detached(self) -> 'VerificationReport':
        '''移除临时 episode/output artifact，保留可持久记录的分维结果。'''
        return VerificationReport(
            self.read_only,
            tuple(replace(result, artifact=None) for result in self.results),
        )


class MultiVerifierOrchestrator:
    '''按规范键运行全部 verifier，并在评估结束后分维提交 effect。'''

    def _validate_effects(
            self,
            registration: VerifierRegistration,
            effects: Sequence[VerificationEffect],
            ) -> tuple[VerificationEffect, ...]:
        '''核验 effect 未跨维度、未越过目标类型声明且没有重复。'''
        result = tuple(effects)
        allowed = {
            target_kind.stable_key()
            for target_kind in registration.allowed_target_kinds
        }
        seen = set()
        for effect in result:
            if not isinstance(effect, VerificationEffect):
                raise VerificationProtocolError(
                    'verifier effect 类型非法')
            if effect.dimension != registration.dimension:
                raise VerificationProtocolError(
                    'verifier effect 试图跨 dimension 写入')
            if effect.target_kind.stable_key() not in allowed:
                raise VerificationProtocolError(
                    'verifier effect 目标类型未在 registration 声明')
            key = effect.stable_key()
            if key in seen:
                raise VerificationProtocolError(
                    'verifier effect 重复')
            seen.add(key)
        return result

    def run(
            self,
            request: Any,
            registrations: Sequence[VerifierRegistration],
            *,
            read_only: bool,
            ) -> VerificationReport:
        '''运行全部注册项；单维失败被记录但不会中止其他维度。'''
        if type(read_only) is not bool:
            raise TypeError('read_only 必须是 bool')
        if not isinstance(registrations, Sequence):
            raise TypeError('registrations 必须是 Sequence')
        registered = tuple(registrations)
        for index, registration in enumerate(registered):
            if not isinstance(registration, VerifierRegistration):
                raise TypeError(
                    f'registrations[{index}] 必须是 VerifierRegistration')
        ordered = sorted(
            registered,
            key=lambda item: item.stable_key(),
        )
        keys = [registration.stable_key() for registration in ordered]
        if len(set(keys)) != len(keys):
            raise VerificationProtocolError(
                '同一 dimension/verifier 不得重复注册')

        evaluated: list[
            tuple[
                VerifierRegistration,
                VerificationEvaluation | None,
                VerificationResult,
            ]
        ] = []
        for registration in ordered:
            try:
                applicable = registration.applies(request)
                if type(applicable) is not bool:
                    raise TypeError('verifier applies 必须返回 bool')
            except Exception as exc:
                result = VerificationResult(
                    dimension=registration.dimension,
                    verifier=registration.verifier,
                    applicability=APPLICABILITY_UNKNOWN,
                    verdict=VERDICT_UNKNOWN,
                    operational_failure=type(exc).__name__,
                )
                evaluated.append((registration, None, result))
                continue
            if not applicable:
                result = VerificationResult(
                    dimension=registration.dimension,
                    verifier=registration.verifier,
                    applicability=APPLICABILITY_NOT_APPLICABLE,
                    verdict=VERDICT_UNKNOWN,
                )
                evaluated.append((registration, None, result))
                continue
            try:
                evaluation = registration.evaluate(request)
                if not isinstance(evaluation, VerificationEvaluation):
                    raise TypeError(
                        'verifier evaluate 必须返回 VerificationEvaluation')
                proposed = self._validate_effects(
                    registration,
                    evaluation.proposed_effects,
                )
                result = VerificationResult(
                    dimension=registration.dimension,
                    verifier=registration.verifier,
                    applicability=APPLICABILITY_APPLICABLE,
                    verdict=evaluation.verdict,
                    claim_keys=evaluation.claim_keys,
                    proposed_effects=proposed,
                    detail=evaluation.detail,
                    source=evaluation.source,
                    scope=evaluation.scope,
                    artifact=evaluation.artifact,
                )
                evaluated.append((registration, evaluation, result))
            except Exception as exc:
                result = VerificationResult(
                    dimension=registration.dimension,
                    verifier=registration.verifier,
                    applicability=APPLICABILITY_APPLICABLE,
                    verdict=VERDICT_UNKNOWN,
                    operational_failure=type(exc).__name__,
                )
                evaluated.append((registration, None, result))

        finalized = []
        for registration, evaluation, result in evaluated:
            if (read_only or evaluation is None
                    or registration.commit is None
                    or result.operational_failure is not None):
                finalized.append(result)
                continue
            try:
                committed = self._validate_effects(
                    registration,
                    registration.commit(evaluation),
                )
                proposed = {
                    effect.stable_key()
                    for effect in result.proposed_effects
                }
                if any(effect.stable_key() not in proposed
                       for effect in committed):
                    raise VerificationProtocolError(
                        'committer 返回了未提议的 effect')
                finalized.append(VerificationResult(
                    dimension=result.dimension,
                    verifier=result.verifier,
                    applicability=result.applicability,
                    verdict=result.verdict,
                    claim_keys=result.claim_keys,
                    proposed_effects=result.proposed_effects,
                    committed_effects=committed,
                    detail=result.detail,
                    source=result.source,
                    scope=result.scope,
                    artifact=result.artifact,
                ))
            except Exception as exc:
                finalized.append(VerificationResult(
                    dimension=result.dimension,
                    verifier=result.verifier,
                    applicability=result.applicability,
                    verdict=result.verdict,
                    claim_keys=result.claim_keys,
                    proposed_effects=result.proposed_effects,
                    detail=result.detail,
                    source=result.source,
                    scope=result.scope,
                    operational_failure=type(exc).__name__,
                    artifact=result.artifact,
                ))
        return VerificationReport(read_only, tuple(finalized))


__all__ = [
    'APPLICABILITY_APPLICABLE',
    'APPLICABILITY_NOT_APPLICABLE',
    'APPLICABILITY_UNKNOWN',
    'MultiVerifierOrchestrator',
    'VERDICT_CONFLICTED',
    'VERDICT_REFUTE',
    'VERDICT_SUPPORT',
    'VERDICT_UNKNOWN',
    'VerificationEffect',
    'VerificationEvaluation',
    'VerificationProtocolError',
    'VerificationReport',
    'VerificationResult',
    'VerifierRegistration',
]
