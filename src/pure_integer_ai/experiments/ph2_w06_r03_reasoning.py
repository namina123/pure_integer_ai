"""W06-R03 完整六维 PROPERTY claim 的 Reasoning consumer。"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.identity import ObjectIdentity
from pure_integer_ai.cognition.shared.property_relation import (
    PropertySelection,
    PropertySelectionOption,
)
from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    LosslessIntegerKey,
)
from pure_integer_ai.experiments.ph2_w06_r03_contract import (
    W06R03ContractError,
    W06R03ReasoningOutcome,
    W06R03ReasoningRequest,
    W06R03ReasoningResolution,
    W06R03ReasoningUse,
    W06_R03_OUTCOME_REFUTE,
    W06_R03_OUTCOME_SUPPORT,
    W06_R03_REASONING_CONFLICT,
    W06_R03_REASONING_REFUTED,
    W06_R03_REASONING_STATUSES,
    W06_R03_REASONING_SUPPORTED,
    W06_R03_REASONING_UNRESOLVED,
    pack_key,
)
from pure_integer_ai.experiments.ph2_w06_r03_shared import (
    W06R03View,
    W06_R03_RUNTIME_NAMESPACE,
)


def _attribution(
        options: tuple[PropertySelectionOption, ...],
        ) -> tuple[tuple[ObjectIdentity, ...], tuple[tuple[int, ...], ...]]:
    """从 PROPERTY options 恢复原始 Proposition 与 Evidence stable key。"""
    propositions = tuple(sorted({
        item.proposition
        for option in options
        for item in option.evaluation.evidence
    }, key=ObjectIdentity.stable_key))
    evidence = tuple(sorted({
        record.stable_key()
        for option in options
        for item in option.evaluation.evidence
        for record in item.evidence
    }))
    return propositions, evidence


class W06R03ReasoningRuntime:
    """按完整六维 claim 区分支持、反驳、冲突和未决。"""

    def __init__(self, view: W06R03View) -> None:
        """绑定共享 R03 view 并初始化 append-only 裁决历史。"""
        if not isinstance(view, W06R03View):
            raise TypeError("R03 Reasoning view 类型非法")
        self.view = view
        self._request_keys: set[LosslessIntegerKey] = set()
        self._resolutions: list[W06R03ReasoningResolution] = []
        self._uses: list[W06R03ReasoningUse] = []
        self._outcomes: list[W06R03ReasoningOutcome] = []

    @property
    def resolutions(self) -> tuple[W06R03ReasoningResolution, ...]:
        """返回本 runtime 的裁决历史。"""
        return tuple(self._resolutions)

    @property
    def uses(self) -> tuple[W06R03ReasoningUse, ...]:
        """返回 SUPPORTED 结论的 exact Use。"""
        return tuple(self._uses)

    @property
    def outcomes(self) -> tuple[W06R03ReasoningOutcome, ...]:
        """返回 append-only 的 current 重验结果。"""
        return tuple(self._outcomes)

    def preview(
            self,
            request: W06R03ReasoningRequest,
            ) -> W06R03ReasoningResolution:
        """无写入地对完整 PROPERTY claim 做 direct fact 裁决。"""
        pattern = self.view.exact_pattern(request.claim)
        if self.view.protocol.reasoning_ready():
            selection = self.view.select(pattern, request.budget).selection
        else:
            selection = PropertySelection(pattern, ())
        option = next(
            (item for item in selection.options
             if item.claim == request.claim),
            None,
        )
        exposed: tuple[PropertySelectionOption, ...]
        if option is None:
            status = W06_R03_REASONING_UNRESOLVED
            exposed = ()
        elif option.evaluation.state.support and option.evaluation.state.refute:
            status = W06_R03_REASONING_CONFLICT
            exposed = (option,)
        elif option.evaluation.state.support:
            status = W06_R03_REASONING_SUPPORTED
            exposed = (option,)
        elif option.evaluation.state.refute:
            status = W06_R03_REASONING_REFUTED
            exposed = (option,)
        else:
            status = W06_R03_REASONING_UNRESOLVED
            exposed = ()
        propositions, evidence = _attribution(exposed)
        return W06R03ReasoningResolution(
            request,
            status,
            selection,
            propositions,
            evidence,
            LosslessIntegerKey((
                W06_R03_RUNTIME_NAMESPACE,
                400 + W06_R03_REASONING_STATUSES.index(status),
                *self.view.protocol.stable_key(),
            )),
        )

    def resolve(
            self,
            request: W06R03ReasoningRequest,
            ) -> W06R03ReasoningResolution:
        """执行有状态裁决，不从 cue、共现或属性路径推理。"""
        if not isinstance(request, W06R03ReasoningRequest):
            raise TypeError("R03 reasoning request 类型非法")
        if request.request_key in self._request_keys:
            raise W06R03ContractError("重复 R03 reasoning request key")
        resolution = self.preview(request)
        self._request_keys.add(request.request_key)
        self._resolutions.append(resolution)
        return resolution

    def adopt(
            self,
            resolution: W06R03ReasoningResolution,
            ) -> W06R03ReasoningUse:
        """重验 SUPPORTED 结论并提交全部 active premise Use。"""
        if resolution not in self._resolutions:
            raise W06R03ContractError("reasoning resolution 不属于 runtime")
        if resolution.status != W06_R03_REASONING_SUPPORTED:
            raise W06R03ContractError("只有 SUPPORTED reasoning 可采用")
        if any(item.resolution == resolution for item in self._uses):
            raise W06R03ContractError("同一 reasoning resolution 不得重复采用")
        current = self.preview(resolution.request)
        if current.stable_key() != resolution.stable_key():
            raise W06R03ContractError("采用前 reasoning resolution 已漂移")
        use_key = LosslessIntegerKey((
            W06_R03_RUNTIME_NAMESPACE,
            500,
            len(self._uses) + 1,
            *pack_key(resolution.stable_key()),
        ))
        result = self.view.select(
            resolution.selection.pattern,
            resolution.request.budget,
            use_key.components,
        )
        selected = result.selection.selected()
        if selected is None or selected.claim != resolution.request.claim:
            raise W06R03ContractError("PROPERTY reasoning adoption 未恢复支持 claim")
        use = W06R03ReasoningUse(resolution, result.uses, use_key)
        self._uses.append(use)
        return use

    def verify(
            self,
            use: W06R03ReasoningUse,
            ) -> W06R03ReasoningOutcome:
        """按 current direct fact 重验历史裁决，撤回后必须 REFUTE。"""
        if use not in self._uses:
            raise W06R03ContractError("reasoning Use 不属于 runtime")
        current = self.preview(use.resolution.request)
        supported = (
            current.status == W06_R03_REASONING_SUPPORTED
            and current.propositions == use.resolution.propositions
            and current.evidence_keys == use.resolution.evidence_keys
        )
        outcome = W06R03ReasoningOutcome(
            use,
            W06_R03_OUTCOME_SUPPORT if supported else W06_R03_OUTCOME_REFUTE,
            current.status,
            LosslessIntegerKey((
                W06_R03_RUNTIME_NAMESPACE,
                600,
                len(self._outcomes) + 1,
                *pack_key(use.use_key.components),
            )),
        )
        self._outcomes.append(outcome)
        return outcome

    def state_key(self) -> tuple:
        """返回裁决、Use 与 outcome 的稳定状态。"""
        return (
            tuple(item.stable_key() for item in self._resolutions),
            tuple(item.use_key.components for item in self._uses),
            tuple(item.outcome_key.components for item in self._outcomes),
        )


__all__ = ["W06R03ReasoningRuntime"]
