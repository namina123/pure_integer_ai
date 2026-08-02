"""W06-R02 Understanding/Reasoning 共用的 typed 集合查询编排。"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.logic_executor import LogicEvidenceState
from pure_integer_ai.cognition.shared.set_relation import SetRelationEvaluation
from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    LosslessIntegerKey,
)
from pure_integer_ai.experiments.ph2_w06_r02_contract import (
    W06R02ContractError,
    W06R02SetOutcome,
    W06R02SetQuery,
    W06R02SetResolution,
    W06R02SetUse,
    W06_R02_CONFLICT,
    W06_R02_CONSUMERS,
    W06_R02_OUTCOME_REFUTE,
    W06_R02_OUTCOME_SUPPORT,
    W06_R02_REFUTED,
    W06_R02_SUPPORTED,
    W06_R02_UNKNOWN,
    pack_key,
)
from pure_integer_ai.experiments.ph2_w06_r02_shared import (
    W06R02View,
    W06_R02_RUNTIME_NAMESPACE,
)


_STATE_STATUS = {
    (True, False): W06_R02_SUPPORTED,
    (False, True): W06_R02_REFUTED,
    (True, True): W06_R02_CONFLICT,
    (False, False): W06_R02_UNKNOWN,
}


class W06R02QueryRuntime:
    """分别保存 U/R 历史，同时复用同一 R-02 查询和 exact Use 规则。"""

    def __init__(self, view: W06R02View, consumer: str) -> None:
        if not isinstance(view, W06R02View):
            raise TypeError("R02 query view 类型非法")
        if consumer not in W06_R02_CONSUMERS:
            raise W06R02ContractError("R02 query consumer 未注册")
        self.view = view
        self.consumer = consumer
        self._request_keys: set[LosslessIntegerKey] = set()
        self._resolutions: list[W06R02SetResolution] = []
        self._uses: list[W06R02SetUse] = []
        self._outcomes: list[W06R02SetOutcome] = []

    @property
    def resolutions(self) -> tuple[W06R02SetResolution, ...]:
        return tuple(self._resolutions)

    @property
    def uses(self) -> tuple[W06R02SetUse, ...]:
        return tuple(self._uses)

    @property
    def outcomes(self) -> tuple[W06R02SetOutcome, ...]:
        return tuple(self._outcomes)

    @staticmethod
    def _proof_attribution(evaluation: SetRelationEvaluation):
        """从选中 proof 递归恢复全部直接 proposition 与 Evidence。"""
        direct = {
            item.stable_key(): item
            for proof in evaluation.proofs
            for item in proof.direct_evidence()
        }
        propositions = tuple(sorted(
            {item.proposition for item in direct.values()},
            key=lambda item: item.stable_key(),
        ))
        evidence_keys = tuple(sorted({
            record.stable_key()
            for item in direct.values()
            for record in item.evidence
        }))
        return propositions, evidence_keys

    def preview(self, request: W06R02SetQuery) -> W06R02SetResolution:
        """无写入地执行四态集合查询，并保留 current conflict 分态。"""
        if not isinstance(request, W06R02SetQuery):
            raise TypeError("R02 set query 类型非法")
        ready = self.view.protocol.query_ready(self.consumer)
        if not ready:
            evaluation = SetRelationEvaluation(
                self.view.statement_for(request),
                LogicEvidenceState(False, False),
                (),
            )
            status = W06_R02_UNKNOWN
            propositions = ()
            evidence_keys = ()
        else:
            evaluation = self.view.evaluate(request).evaluation
            conflicts = self.view.current_conflicts(request)
            if conflicts:
                status = W06_R02_CONFLICT
                propositions = tuple(sorted(
                    {item.proposition.proposition for item in conflicts},
                    key=lambda item: item.stable_key(),
                ))
                evidence_keys = tuple(sorted({
                    record.stable_key()
                    for candidate in conflicts
                    for record in self.view.learning.snapshot_for(
                        candidate.proposition.proposition).evidence
                }))
            else:
                state = evaluation.state
                status = _STATE_STATUS[(state.support, state.refute)]
                propositions, evidence_keys = self._proof_attribution(evaluation)
                if status == W06_R02_UNKNOWN:
                    propositions = ()
                    evidence_keys = ()
        return W06R02SetResolution(
            self.consumer,
            request,
            status,
            evaluation,
            propositions,
            evidence_keys,
            LosslessIntegerKey((
                W06_R02_RUNTIME_NAMESPACE,
                100 + W06_R02_CONSUMERS.index(self.consumer) * 20,
                (W06_R02_SUPPORTED, W06_R02_REFUTED,
                 W06_R02_CONFLICT, W06_R02_UNKNOWN).index(status) + 1,
                *self.view.protocol.stable_key(),
            )),
        )

    def resolve(self, request: W06R02SetQuery) -> W06R02SetResolution:
        """执行一次有状态 U/R 查询，不允许 request key 重放。"""
        if request.request_key in self._request_keys:
            raise W06R02ContractError("重复 R02 set query request key")
        resolution = self.preview(request)
        self._request_keys.add(request.request_key)
        self._resolutions.append(resolution)
        return resolution

    def adopt(self, resolution: W06R02SetResolution) -> W06R02SetUse:
        """重算 SUPPORTED proof 并原子提交全部 active 前提 Use。"""
        if resolution not in self._resolutions:
            raise W06R02ContractError("set resolution 不属于当前 runtime")
        if resolution.status != W06_R02_SUPPORTED:
            raise W06R02ContractError("只有 SUPPORTED set resolution 可采用")
        if any(item.resolution == resolution for item in self._uses):
            raise W06R02ContractError("同一 set resolution 不得重复采用")
        current = self.preview(resolution.request)
        if current.stable_key() != resolution.stable_key():
            raise W06R02ContractError("采用前 set resolution 已漂移")
        use_key = LosslessIntegerKey((
            W06_R02_RUNTIME_NAMESPACE,
            200 + W06_R02_CONSUMERS.index(self.consumer) * 20,
            len(self._uses) + 1,
            *pack_key(resolution.stable_key()),
        ))
        committed = self.view.commit(resolution.request, use_key)
        use = W06R02SetUse(resolution, committed.uses, use_key)
        self._uses.append(use)
        return use

    def verify(self, use: W06R02SetUse) -> W06R02SetOutcome:
        """按 current closure 重验历史 proof，withdrawal 后不得保持 SUPPORT。"""
        if use not in self._uses:
            raise W06R02ContractError("set Use 不属于当前 runtime")
        current = self.preview(use.resolution.request)
        supported = current.stable_key() == use.resolution.stable_key()
        outcome = W06R02SetOutcome(
            use,
            W06_R02_OUTCOME_SUPPORT if supported else W06_R02_OUTCOME_REFUTE,
            current.status,
            LosslessIntegerKey((
                W06_R02_RUNTIME_NAMESPACE,
                300 + W06_R02_CONSUMERS.index(self.consumer) * 20,
                len(self._outcomes) + 1,
                *pack_key(use.use_key.components),
            )),
        )
        self._outcomes.append(outcome)
        return outcome

    def state_key(self) -> tuple:
        return (
            W06_R02_CONSUMERS.index(self.consumer) + 1,
            tuple(item.stable_key() for item in self._resolutions),
            tuple(item.use_key.components for item in self._uses),
            tuple(item.outcome_key.components for item in self._outcomes),
        )


__all__ = ["W06R02QueryRuntime"]
