"""W06-R04 Understanding/Reasoning 共用的 typed 部分整体查询编排。"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.logic_executor import LogicEvidenceState
from pure_integer_ai.cognition.shared.mereology_relation import (
    MereologyEvaluation,
)
from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    LosslessIntegerKey,
)
from pure_integer_ai.experiments.ph2_w06_r04_contract import (
    W06R04ContractError,
    W06R04MereologyOutcome,
    W06R04MereologyQuery,
    W06R04MereologyResolution,
    W06R04MereologyUse,
    W06_R04_CONFLICT,
    W06_R04_CONSUMERS,
    W06_R04_OUTCOME_REFUTE,
    W06_R04_OUTCOME_SUPPORT,
    W06_R04_REFUTED,
    W06_R04_SUPPORTED,
    W06_R04_UNKNOWN,
    pack_key,
)
from pure_integer_ai.experiments.ph2_w06_r04_shared import (
    W06R04View,
    W06_R04_RUNTIME_NAMESPACE,
)


_STATE_STATUS = {
    (True, False): W06_R04_SUPPORTED,
    (False, True): W06_R04_REFUTED,
    (True, True): W06_R04_CONFLICT,
    (False, False): W06_R04_UNKNOWN,
}


class W06R04QueryRuntime:
    """分别保存 U/R 历史，同时复用同一 R-04 查询和 exact Use 规则。"""

    def __init__(self, view: W06R04View, consumer: str) -> None:
        """绑定共享 view 和当前 consumer 名称。"""
        if not isinstance(view, W06R04View):
            raise TypeError("R04 query view 类型非法")
        if consumer not in W06_R04_CONSUMERS:
            raise W06R04ContractError("R04 query consumer 未注册")
        self.view = view
        self.consumer = consumer
        self._request_keys: set[LosslessIntegerKey] = set()
        self._resolutions: list[W06R04MereologyResolution] = []
        self._uses: list[W06R04MereologyUse] = []
        self._outcomes: list[W06R04MereologyOutcome] = []

    @property
    def resolutions(self) -> tuple[W06R04MereologyResolution, ...]:
        """返回已执行 resolution 历史。"""
        return tuple(self._resolutions)

    @property
    def uses(self) -> tuple[W06R04MereologyUse, ...]:
        """返回已采用 exact Use 历史。"""
        return tuple(self._uses)

    @property
    def outcomes(self) -> tuple[W06R04MereologyOutcome, ...]:
        """返回已重验 outcome 历史。"""
        return tuple(self._outcomes)

    @staticmethod
    def _proof_attribution(
            evaluation: MereologyEvaluation,
            status: str,
            ):
        """从 evaluation 恢复当前裁决需要公开的 Proposition 与 Evidence。"""
        if status == W06_R04_UNKNOWN:
            return (), ()
        if status == W06_R04_SUPPORTED:
            direct = evaluation.active_premises()
        else:
            direct = evaluation.direct_evidence
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
            self, request: W06R04MereologyQuery,
            ) -> W06R04MereologyResolution:
        """无写入地执行四态部分整体查询。"""
        if not isinstance(request, W06R04MereologyQuery):
            raise TypeError("R04 mereology query 类型非法")
        ready = self.view.protocol.query_ready(self.consumer)
        if not ready:
            evaluation = self.view.evaluate(request)
            status = W06_R04_UNKNOWN
            propositions = ()
            evidence_keys = ()
        else:
            evaluation = self.view.evaluate(request)
            state = evaluation.state
            status = _STATE_STATUS[(state.support, state.refute)]
            propositions, evidence_keys = self._proof_attribution(
                evaluation, status)
        return W06R04MereologyResolution(
            self.consumer,
            request,
            status,
            evaluation,
            propositions,
            evidence_keys,
            LosslessIntegerKey((
                W06_R04_RUNTIME_NAMESPACE,
                100 + W06_R04_CONSUMERS.index(self.consumer) * 20,
                (W06_R04_SUPPORTED, W06_R04_REFUTED,
                 W06_R04_CONFLICT, W06_R04_UNKNOWN).index(status) + 1,
                *self.view.protocol.stable_key(),
            )),
        )

    def resolve(
            self, request: W06R04MereologyQuery,
            ) -> W06R04MereologyResolution:
        """执行一次有状态 U/R 查询，不允许 request key 重放。"""
        if request.request_key in self._request_keys:
            raise W06R04ContractError("重复 R04 mereology query request key")
        resolution = self.preview(request)
        self._request_keys.add(request.request_key)
        self._resolutions.append(resolution)
        return resolution

    def adopt(
            self,
            resolution: W06R04MereologyResolution,
            ) -> W06R04MereologyUse:
        """重算 SUPPORTED proof 并原子提交全部 active 前提 Use。"""
        if resolution not in self._resolutions:
            raise W06R04ContractError("mereology resolution 不属于当前 runtime")
        if resolution.status != W06_R04_SUPPORTED:
            raise W06R04ContractError("只有 SUPPORTED mereology resolution 可采用")
        if any(item.resolution == resolution for item in self._uses):
            raise W06R04ContractError("同一 mereology resolution 不得重复采用")
        current = self.preview(resolution.request)
        if current.stable_key() != resolution.stable_key():
            raise W06R04ContractError("采用前 mereology resolution 已漂移")
        use_key = LosslessIntegerKey((
            W06_R04_RUNTIME_NAMESPACE,
            200 + W06_R04_CONSUMERS.index(self.consumer) * 20,
            len(self._uses) + 1,
            *pack_key(resolution.stable_key()),
        ))
        committed = self.view.commit(resolution.request, use_key)
        use = W06R04MereologyUse(resolution, committed.uses, use_key)
        self._uses.append(use)
        return use

    def verify(self, use: W06R04MereologyUse) -> W06R04MereologyOutcome:
        """按 current closure 重验历史 proof，withdrawal 后不得保持 SUPPORT。"""
        if use not in self._uses:
            raise W06R04ContractError("mereology Use 不属于当前 runtime")
        current = self.preview(use.resolution.request)
        supported = current.stable_key() == use.resolution.stable_key()
        outcome = W06R04MereologyOutcome(
            use,
            W06_R04_OUTCOME_SUPPORT if supported else W06_R04_OUTCOME_REFUTE,
            current.status,
            LosslessIntegerKey((
                W06_R04_RUNTIME_NAMESPACE,
                300 + W06_R04_CONSUMERS.index(self.consumer) * 20,
                len(self._outcomes) + 1,
                *pack_key(use.use_key.components),
            )),
        )
        self._outcomes.append(outcome)
        return outcome

    def state_key(self) -> tuple:
        """返回 consumer、resolution、Use 和 outcome 的稳定状态。"""
        return (
            W06_R04_CONSUMERS.index(self.consumer) + 1,
            tuple(item.stable_key() for item in self._resolutions),
            tuple(item.use_key.components for item in self._uses),
            tuple(item.outcome_key.components for item in self._outcomes),
        )


__all__ = ["W06R04QueryRuntime"]
