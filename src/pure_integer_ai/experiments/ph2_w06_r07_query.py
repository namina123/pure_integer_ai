"""W06-R07 Understanding/Reasoning 共用的 direct CAUSES 查询编排。"""
from __future__ import annotations

from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    LosslessIntegerKey,
)
from pure_integer_ai.experiments.ph2_w06_r07_contract import (
    W06R07CausalOutcome,
    W06R07CausalQuery,
    W06R07CausalResolution,
    W06R07CausalUse,
    W06R07ContractError,
    W06_R07_CONFLICT,
    W06_R07_CONSUMERS,
    W06_R07_OUTCOME_REFUTE,
    W06_R07_OUTCOME_SUPPORT,
    W06_R07_REFUTED,
    W06_R07_RUNTIME_NAMESPACE,
    W06_R07_SUPPORTED,
    W06_R07_UNKNOWN,
    pack_key,
)
from pure_integer_ai.experiments.ph2_w06_r07_shared import W06R07View


_STATE_STATUS = {
    (True, False): W06_R07_SUPPORTED,
    (False, True): W06_R07_REFUTED,
    (True, True): W06_R07_CONFLICT,
    (False, False): W06_R07_UNKNOWN,
}


class W06R07QueryRuntime:
    """分别保存 U/R 历史，同时复用同一 direct CAUSES view。"""

    def __init__(self, view: W06R07View, consumer: str) -> None:
        if not isinstance(view, W06R07View):
            raise TypeError("R07 query view 类型非法")
        if consumer not in W06_R07_CONSUMERS:
            raise W06R07ContractError("R07 query consumer 未注册")
        self.view = view
        self.consumer = consumer
        self._request_keys: set[LosslessIntegerKey] = set()
        self._resolutions: list[W06R07CausalResolution] = []
        self._uses: list[W06R07CausalUse] = []
        self._outcomes: list[W06R07CausalOutcome] = []

    @property
    def resolutions(self) -> tuple[W06R07CausalResolution, ...]:
        return tuple(self._resolutions)

    @property
    def uses(self) -> tuple[W06R07CausalUse, ...]:
        return tuple(self._uses)

    @property
    def outcomes(self) -> tuple[W06R07CausalOutcome, ...]:
        return tuple(self._outcomes)

    def preview(
            self, request: W06R07CausalQuery,
            ) -> W06R07CausalResolution:
        if not isinstance(request, W06R07CausalQuery):
            raise TypeError("R07 causal query 类型非法")
        evaluation = self.view.evaluate(request)
        if not self.view.protocol.query_ready(self.consumer):
            status = W06_R07_UNKNOWN
            propositions = ()
            evidence_keys = ()
        else:
            status = _STATE_STATUS[(
                evaluation.state.support, evaluation.state.refute)]
            if status == W06_R07_SUPPORTED:
                propositions = evaluation.active_propositions
                evidence_keys = evaluation.evidence_keys
            elif status in {W06_R07_REFUTED, W06_R07_CONFLICT}:
                propositions = evaluation.matched_propositions
                evidence_keys = evaluation.evidence_keys
            else:
                propositions = ()
                evidence_keys = ()
        return W06R07CausalResolution(
            self.consumer,
            request,
            status,
            evaluation,
            propositions,
            evidence_keys,
            LosslessIntegerKey((
                W06_R07_RUNTIME_NAMESPACE,
                100 + W06_R07_CONSUMERS.index(self.consumer) * 20,
                (W06_R07_SUPPORTED, W06_R07_REFUTED,
                 W06_R07_CONFLICT, W06_R07_UNKNOWN).index(status) + 1,
                *self.view.protocol.stable_key(),
            )),
        )

    def resolve(
            self, request: W06R07CausalQuery,
            ) -> W06R07CausalResolution:
        if request.request_key in self._request_keys:
            raise W06R07ContractError("重复 R07 causal query request key")
        resolution = self.preview(request)
        self._request_keys.add(request.request_key)
        self._resolutions.append(resolution)
        return resolution

    def adopt(
            self,
            resolution: W06R07CausalResolution,
            ) -> W06R07CausalUse:
        if resolution not in self._resolutions:
            raise W06R07ContractError("causal resolution 不属于当前 runtime")
        if resolution.status != W06_R07_SUPPORTED:
            raise W06R07ContractError("只有 SUPPORTED causal resolution 可采用")
        if any(item.resolution == resolution for item in self._uses):
            raise W06R07ContractError("同一 causal resolution 不得重复采用")
        current = self.preview(resolution.request)
        if current.stable_key() != resolution.stable_key():
            raise W06R07ContractError("采用前 causal resolution 已漂移")
        use_key = LosslessIntegerKey((
            W06_R07_RUNTIME_NAMESPACE,
            200 + W06_R07_CONSUMERS.index(self.consumer) * 20,
            len(self._uses) + 1,
            *pack_key(resolution.stable_key()),
        ))
        committed = self.view.commit(
            resolution.request, use_key, self.consumer)
        use = W06R07CausalUse(resolution, committed, use_key)
        self._uses.append(use)
        return use

    def verify(self, use: W06R07CausalUse) -> W06R07CausalOutcome:
        if use not in self._uses:
            raise W06R07ContractError("causal Use 不属于当前 runtime")
        current = self.preview(use.resolution.request)
        supported = current.stable_key() == use.resolution.stable_key()
        outcome = W06R07CausalOutcome(
            use,
            W06_R07_OUTCOME_SUPPORT if supported else W06_R07_OUTCOME_REFUTE,
            current.status,
            LosslessIntegerKey((
                W06_R07_RUNTIME_NAMESPACE,
                300 + W06_R07_CONSUMERS.index(self.consumer) * 20,
                len(self._outcomes) + 1,
                *pack_key(use.use_key.components),
            )),
        )
        self._outcomes.append(outcome)
        return outcome

    def state_key(self) -> tuple:
        return (
            W06_R07_CONSUMERS.index(self.consumer) + 1,
            tuple(item.stable_key() for item in self._resolutions),
            tuple(item.use_key.components for item in self._uses),
            tuple(item.outcome_key.components for item in self._outcomes),
        )


__all__ = ["W06R07QueryRuntime"]
