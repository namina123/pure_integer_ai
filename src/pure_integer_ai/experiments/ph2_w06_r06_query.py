"""W06-R06 Understanding/Reasoning 共用的 event-time 查询编排。"""
from __future__ import annotations

from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    LosslessIntegerKey,
)
from pure_integer_ai.experiments.ph2_w06_r06_contract import (
    W06R06ContractError,
    W06R06EventTimeOutcome,
    W06R06EventTimeQuery,
    W06R06EventTimeResolution,
    W06R06EventTimeUse,
    W06_R06_CONFLICT,
    W06_R06_CONSUMERS,
    W06_R06_OUTCOME_REFUTE,
    W06_R06_OUTCOME_SUPPORT,
    W06_R06_REFUTED,
    W06_R06_RUNTIME_NAMESPACE,
    W06_R06_SUPPORTED,
    W06_R06_UNKNOWN,
    pack_key,
)
from pure_integer_ai.experiments.ph2_w06_r06_shared import W06R06View


_STATE_STATUS = {
    (True, False): W06_R06_SUPPORTED,
    (False, True): W06_R06_REFUTED,
    (True, True): W06_R06_CONFLICT,
    (False, False): W06_R06_UNKNOWN,
}


class W06R06QueryRuntime:
    """分别保存 U/R 历史，同时复用同一 event-time view 和 exact Use。"""

    def __init__(self, view: W06R06View, consumer: str) -> None:
        if not isinstance(view, W06R06View):
            raise TypeError("R06 query view 类型非法")
        if consumer not in W06_R06_CONSUMERS:
            raise W06R06ContractError("R06 query consumer 未注册")
        self.view = view
        self.consumer = consumer
        self._request_keys: set[LosslessIntegerKey] = set()
        self._resolutions: list[W06R06EventTimeResolution] = []
        self._uses: list[W06R06EventTimeUse] = []
        self._outcomes: list[W06R06EventTimeOutcome] = []

    @property
    def resolutions(self) -> tuple[W06R06EventTimeResolution, ...]:
        return tuple(self._resolutions)

    @property
    def uses(self) -> tuple[W06R06EventTimeUse, ...]:
        return tuple(self._uses)

    @property
    def outcomes(self) -> tuple[W06R06EventTimeOutcome, ...]:
        return tuple(self._outcomes)

    def preview(
            self, request: W06R06EventTimeQuery,
            ) -> W06R06EventTimeResolution:
        """无写入执行 raw family + qualifier 的 event-time 四态查询。"""
        if not isinstance(request, W06R06EventTimeQuery):
            raise TypeError("R06 event-time query 类型非法")
        evaluation = self.view.evaluate(request)
        if not self.view.protocol.query_ready(
                self.consumer, request.relation_family):
            status = W06_R06_UNKNOWN
            propositions = ()
            evidence_keys = ()
        else:
            status = _STATE_STATUS[(
                evaluation.state.support, evaluation.state.refute)]
            if status == W06_R06_SUPPORTED:
                propositions = evaluation.active_propositions
                evidence_keys = evaluation.evidence_keys
            elif status in {W06_R06_REFUTED, W06_R06_CONFLICT}:
                propositions = evaluation.matched_propositions
                evidence_keys = evaluation.evidence_keys
            elif evaluation.explicit_unknown:
                propositions = evaluation.matched_propositions
                evidence_keys = evaluation.evidence_keys
            else:
                propositions = ()
                evidence_keys = ()
        return W06R06EventTimeResolution(
            self.consumer,
            request,
            status,
            evaluation,
            propositions,
            evidence_keys,
            LosslessIntegerKey((
                W06_R06_RUNTIME_NAMESPACE,
                100 + W06_R06_CONSUMERS.index(self.consumer) * 20,
                (W06_R06_SUPPORTED, W06_R06_REFUTED,
                 W06_R06_CONFLICT, W06_R06_UNKNOWN).index(status) + 1,
                *self.view.protocol.stable_key(),
            )),
        )

    def resolve(
            self, request: W06R06EventTimeQuery,
            ) -> W06R06EventTimeResolution:
        if request.request_key in self._request_keys:
            raise W06R06ContractError("重复 R06 event-time query request key")
        resolution = self.preview(request)
        self._request_keys.add(request.request_key)
        self._resolutions.append(resolution)
        return resolution

    def adopt(
            self,
            resolution: W06R06EventTimeResolution,
            ) -> W06R06EventTimeUse:
        if resolution not in self._resolutions:
            raise W06R06ContractError("event-time resolution 不属于当前 runtime")
        if resolution.status != W06_R06_SUPPORTED:
            raise W06R06ContractError("只有 SUPPORTED event-time resolution 可采用")
        if any(item.resolution == resolution for item in self._uses):
            raise W06R06ContractError("同一 event-time resolution 不得重复采用")
        current = self.preview(resolution.request)
        if current.stable_key() != resolution.stable_key():
            raise W06R06ContractError("采用前 event-time resolution 已漂移")
        use_key = LosslessIntegerKey((
            W06_R06_RUNTIME_NAMESPACE,
            200 + W06_R06_CONSUMERS.index(self.consumer) * 20,
            len(self._uses) + 1,
            *pack_key(resolution.stable_key()),
        ))
        committed = self.view.commit(
            resolution.request, use_key, self.consumer)
        use = W06R06EventTimeUse(resolution, committed, use_key)
        self._uses.append(use)
        return use

    def verify(self, use: W06R06EventTimeUse) -> W06R06EventTimeOutcome:
        """按 current qualifier view 重验，withdrawal 后拒绝 stale Use。"""
        if use not in self._uses:
            raise W06R06ContractError("event-time Use 不属于当前 runtime")
        current = self.preview(use.resolution.request)
        supported = current.stable_key() == use.resolution.stable_key()
        outcome = W06R06EventTimeOutcome(
            use,
            W06_R06_OUTCOME_SUPPORT if supported else W06_R06_OUTCOME_REFUTE,
            current.status,
            LosslessIntegerKey((
                W06_R06_RUNTIME_NAMESPACE,
                300 + W06_R06_CONSUMERS.index(self.consumer) * 20,
                len(self._outcomes) + 1,
                *pack_key(use.use_key.components),
            )),
        )
        self._outcomes.append(outcome)
        return outcome

    def state_key(self) -> tuple:
        return (
            W06_R06_CONSUMERS.index(self.consumer) + 1,
            tuple(item.stable_key() for item in self._resolutions),
            tuple(item.use_key.components for item in self._uses),
            tuple(item.outcome_key.components for item in self._outcomes),
        )


__all__ = ["W06R06QueryRuntime"]
