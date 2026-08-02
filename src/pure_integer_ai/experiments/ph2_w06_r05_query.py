"""W06-R05 Understanding/Reasoning 共用的 typed pair+channel 查询编排。"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.logic_executor import LogicEvidenceState
from pure_integer_ai.cognition.shared.symmetric_relation import (
    SymmetricPairEvaluation,
)
from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    LosslessIntegerKey,
)
from pure_integer_ai.experiments.ph2_w06_r05_contract import (
    W06R05ContractError,
    W06R05PairOutcome,
    W06R05PairQuery,
    W06R05PairResolution,
    W06R05PairUse,
    W06_R05_CONFLICT,
    W06_R05_CONSUMERS,
    W06_R05_OUTCOME_REFUTE,
    W06_R05_OUTCOME_SUPPORT,
    W06_R05_REFUTED,
    W06_R05_SUPPORTED,
    W06_R05_UNKNOWN,
    pack_key,
)
from pure_integer_ai.experiments.ph2_w06_r05_shared import (
    W06R05View,
    W06_R05_RUNTIME_NAMESPACE,
)


_STATE_STATUS = {
    (True, False): W06_R05_SUPPORTED,
    (False, True): W06_R05_REFUTED,
    (True, True): W06_R05_CONFLICT,
    (False, False): W06_R05_UNKNOWN,
}


class W06R05QueryRuntime:
    """分别保存 U/R 历史，同时复用同一 symmetric pair 和 exact Use。"""

    def __init__(self, view: W06R05View, consumer: str) -> None:
        """绑定共享 view 和当前 consumer 名称。"""
        if not isinstance(view, W06R05View):
            raise TypeError("R05 query view 类型非法")
        if consumer not in W06_R05_CONSUMERS:
            raise W06R05ContractError("R05 query consumer 未注册")
        self.view = view
        self.consumer = consumer
        self._request_keys: set[LosslessIntegerKey] = set()
        self._resolutions: list[W06R05PairResolution] = []
        self._uses: list[W06R05PairUse] = []
        self._outcomes: list[W06R05PairOutcome] = []

    @property
    def resolutions(self) -> tuple[W06R05PairResolution, ...]:
        """返回已执行 resolution 历史。"""
        return tuple(self._resolutions)

    @property
    def uses(self) -> tuple[W06R05PairUse, ...]:
        """返回已采用 exact Use 历史。"""
        return tuple(self._uses)

    @property
    def outcomes(self) -> tuple[W06R05PairOutcome, ...]:
        """返回已重验 outcome 历史。"""
        return tuple(self._outcomes)

    @staticmethod
    def _attribution(
            evaluation: SymmetricPairEvaluation,
            status: str,
            ):
        """从 evaluation 恢复当前裁决公开的 Proposition 与 Evidence。"""
        if status == W06_R05_UNKNOWN:
            return (), ()
        direct = (
            evaluation.active_premises()
            if status == W06_R05_SUPPORTED
            else evaluation.evidence
        )
        propositions = tuple(sorted(
            {item.proposition for item in direct},
            key=lambda item: item.stable_key(),
        ))
        evidence_keys = tuple(sorted({
            record.stable_key()
            for item in direct
            for record in item.evidence
        }))
        return propositions, evidence_keys

    def preview(
            self, request: W06R05PairQuery,
            ) -> W06R05PairResolution:
        """无写入地执行四态 pair+channel 查询。"""
        if not isinstance(request, W06R05PairQuery):
            raise TypeError("R05 pair query 类型非法")
        evaluation = self.view.evaluate(request)
        if not self.view.protocol.query_ready(self.consumer, request.channel):
            status = W06_R05_UNKNOWN
            propositions = ()
            evidence_keys = ()
        else:
            state = evaluation.state
            status = _STATE_STATUS[(state.support, state.refute)]
            propositions, evidence_keys = self._attribution(evaluation, status)
        return W06R05PairResolution(
            self.consumer,
            request,
            status,
            evaluation,
            propositions,
            evidence_keys,
            LosslessIntegerKey((
                W06_R05_RUNTIME_NAMESPACE,
                100 + W06_R05_CONSUMERS.index(self.consumer) * 20,
                (W06_R05_SUPPORTED, W06_R05_REFUTED,
                 W06_R05_CONFLICT, W06_R05_UNKNOWN).index(status) + 1,
                *self.view.protocol.stable_key(),
            )),
        )

    def resolve(
            self, request: W06R05PairQuery,
            ) -> W06R05PairResolution:
        """执行一次有状态 U/R 查询，不允许 request key 重放。"""
        if request.request_key in self._request_keys:
            raise W06R05ContractError("重复 R05 pair query request key")
        resolution = self.preview(request)
        self._request_keys.add(request.request_key)
        self._resolutions.append(resolution)
        return resolution

    def adopt(self, resolution: W06R05PairResolution) -> W06R05PairUse:
        """重算 SUPPORTED pair 并原子提交全部 active premise Use。"""
        if resolution not in self._resolutions:
            raise W06R05ContractError("pair resolution 不属于当前 runtime")
        if resolution.status != W06_R05_SUPPORTED:
            raise W06R05ContractError("只有 SUPPORTED pair resolution 可采用")
        if any(item.resolution == resolution for item in self._uses):
            raise W06R05ContractError("同一 pair resolution 不得重复采用")
        current = self.preview(resolution.request)
        if current.stable_key() != resolution.stable_key():
            raise W06R05ContractError("采用前 pair resolution 已漂移")
        use_key = LosslessIntegerKey((
            W06_R05_RUNTIME_NAMESPACE,
            200 + W06_R05_CONSUMERS.index(self.consumer) * 20,
            len(self._uses) + 1,
            *pack_key(resolution.stable_key()),
        ))
        committed = self.view.commit(
            resolution.request, use_key, self.consumer)
        use = W06R05PairUse(resolution, committed.uses, use_key)
        self._uses.append(use)
        return use

    def verify(self, use: W06R05PairUse) -> W06R05PairOutcome:
        """按 current channel 重验历史 proof，withdrawal 后拒绝 stale Use。"""
        if use not in self._uses:
            raise W06R05ContractError("pair Use 不属于当前 runtime")
        current = self.preview(use.resolution.request)
        supported = (
            current.status == W06_R05_SUPPORTED
            and current.propositions == use.resolution.propositions
            and current.evidence_keys == use.resolution.evidence_keys
        )
        outcome = W06R05PairOutcome(
            use,
            W06_R05_OUTCOME_SUPPORT if supported else W06_R05_OUTCOME_REFUTE,
            current.status,
            LosslessIntegerKey((
                W06_R05_RUNTIME_NAMESPACE,
                300 + W06_R05_CONSUMERS.index(self.consumer) * 20,
                len(self._outcomes) + 1,
                *pack_key(use.use_key.components),
            )),
        )
        self._outcomes.append(outcome)
        return outcome

    def state_key(self) -> tuple:
        """返回 consumer、resolution、Use 和 outcome 的稳定状态。"""
        return (
            W06_R05_CONSUMERS.index(self.consumer) + 1,
            tuple(item.stable_key() for item in self._resolutions),
            tuple(item.use_key.components for item in self._uses),
            tuple(item.outcome_key.components for item in self._outcomes),
        )


__all__ = ["W06R05QueryRuntime"]
