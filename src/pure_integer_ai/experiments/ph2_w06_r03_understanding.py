"""W06-R03 subject/attribute 到 PROPERTY 值的 Understanding consumer。"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.identity import ObjectIdentity
from pure_integer_ai.cognition.shared.property_relation import (
    PropertyPattern,
    PropertySelection,
    PropertySelectionOption,
)
from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    LosslessIntegerKey,
)
from pure_integer_ai.experiments.ph2_w06_r03_contract import (
    W06R03ContractError,
    W06R03UnderstandingOutcome,
    W06R03UnderstandingRequest,
    W06R03UnderstandingResolution,
    W06R03UnderstandingUse,
    W06_R03_OUTCOME_REFUTE,
    W06_R03_OUTCOME_SUPPORT,
    W06_R03_UNDERSTANDING_CLARIFY,
    W06_R03_UNDERSTANDING_CONFLICT,
    W06_R03_UNDERSTANDING_MULTI,
    W06_R03_UNDERSTANDING_STATUSES,
    W06_R03_UNDERSTANDING_UNIQUE,
    W06_R03_UNDERSTANDING_UNKNOWN,
    pack_key,
)
from pure_integer_ai.experiments.ph2_w06_r03_shared import (
    W06R03View,
    W06_R03_RUNTIME_NAMESPACE,
)


def _attribution(
        options: tuple[PropertySelectionOption, ...],
        ) -> tuple[tuple[ObjectIdentity, ...], tuple[tuple[int, ...], ...]]:
    """从 PROPERTY options 恢复全部原 Proposition 与 Evidence stable key。"""
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


class W06R03UnderstandingRuntime:
    """形成 UNIQUE/MULTI/CLARIFY/CONFLICT/UNKNOWN 与 exact Use。"""

    def __init__(self, view: W06R03View) -> None:
        """绑定共享 R03 view 并初始化 append-only consumer 历史。"""
        if not isinstance(view, W06R03View):
            raise TypeError("R03 Understanding view 类型非法")
        self.view = view
        self._request_keys: set[LosslessIntegerKey] = set()
        self._resolutions: list[W06R03UnderstandingResolution] = []
        self._uses: list[W06R03UnderstandingUse] = []
        self._outcomes: list[W06R03UnderstandingOutcome] = []

    @property
    def resolutions(self) -> tuple[W06R03UnderstandingResolution, ...]:
        """返回本 runtime 的查询历史。"""
        return tuple(self._resolutions)

    @property
    def uses(self) -> tuple[W06R03UnderstandingUse, ...]:
        """返回已采用的 UNIQUE PROPERTY 选择。"""
        return tuple(self._uses)

    @property
    def outcomes(self) -> tuple[W06R03UnderstandingOutcome, ...]:
        """返回 append-only 的 current 重验结果。"""
        return tuple(self._outcomes)

    def preview(
            self,
            request: W06R03UnderstandingRequest,
            ) -> W06R03UnderstandingResolution:
        """无写入地按 subject/attribute 查询 direct PROPERTY 当前事实。"""
        pattern = PropertyPattern(request.subject, request.attribute)
        if self.view.protocol.understanding_ready():
            selection = self.view.select(pattern, request.budget).selection
        else:
            selection = PropertySelection(pattern, ())
        conflicts = tuple(
            item for item in selection.options
            if item.evaluation.state.support and item.evaluation.state.refute
        )
        selected_option = selection.selected()
        support_options = selection.support_bearing()
        if conflicts:
            status = W06_R03_UNDERSTANDING_CONFLICT
            exposed = conflicts
            selected = None
        elif selected_option is not None:
            status = W06_R03_UNDERSTANDING_UNIQUE
            exposed = (selected_option,)
            selected = selected_option.claim
        elif len(support_options) > 1 and request.allow_multiple:
            status = W06_R03_UNDERSTANDING_MULTI
            exposed = support_options
            selected = None
        elif len(support_options) > 1:
            status = W06_R03_UNDERSTANDING_CLARIFY
            exposed = ()
            selected = None
        else:
            status = W06_R03_UNDERSTANDING_UNKNOWN
            exposed = ()
            selected = None
        options = tuple(sorted(
            (item.claim for item in exposed),
            key=lambda item: item.stable_key(),
        ))
        propositions, evidence = _attribution(exposed)
        return W06R03UnderstandingResolution(
            request,
            status,
            options,
            selected,
            selection,
            propositions,
            evidence,
            LosslessIntegerKey((
                W06_R03_RUNTIME_NAMESPACE,
                100 + W06_R03_UNDERSTANDING_STATUSES.index(status),
                *self.view.protocol.stable_key(),
            )),
        )

    def resolve(
            self,
            request: W06R03UnderstandingRequest,
            ) -> W06R03UnderstandingResolution:
        """执行有状态查询并拒绝 request key 重放。"""
        if not isinstance(request, W06R03UnderstandingRequest):
            raise TypeError("R03 understanding request 类型非法")
        if request.request_key in self._request_keys:
            raise W06R03ContractError("重复 R03 understanding request key")
        resolution = self.preview(request)
        self._request_keys.add(request.request_key)
        self._resolutions.append(resolution)
        return resolution

    def adopt(
            self,
            resolution: W06R03UnderstandingResolution,
            ) -> W06R03UnderstandingUse:
        """重验 UNIQUE 选择并经 PropertyRelationRuntime 提交 exact Use。"""
        if resolution not in self._resolutions:
            raise W06R03ContractError("understanding resolution 不属于 runtime")
        if (resolution.status != W06_R03_UNDERSTANDING_UNIQUE
                or resolution.selected is None):
            raise W06R03ContractError("只有 UNIQUE understanding 可采用")
        if any(item.resolution == resolution for item in self._uses):
            raise W06R03ContractError("同一 understanding resolution 不得重复采用")
        current = self.preview(resolution.request)
        if current.stable_key() != resolution.stable_key():
            raise W06R03ContractError("采用前 understanding resolution 已漂移")
        use_key = LosslessIntegerKey((
            W06_R03_RUNTIME_NAMESPACE,
            200,
            len(self._uses) + 1,
            *pack_key(resolution.stable_key()),
        ))
        result = self.view.select(
            resolution.selection.pattern,
            resolution.request.budget,
            use_key.components,
        )
        selected = result.selection.selected()
        if selected is None or selected.claim != resolution.selected:
            raise W06R03ContractError("PROPERTY adoption 未恢复唯一选择")
        use = W06R03UnderstandingUse(
            resolution, result.uses, use_key)
        self._uses.append(use)
        return use

    def verify(
            self,
            use: W06R03UnderstandingUse,
            ) -> W06R03UnderstandingOutcome:
        """按 current direct fact 重验历史选择，撤回后必须 REFUTE。"""
        if use not in self._uses:
            raise W06R03ContractError("understanding Use 不属于 runtime")
        current = self.preview(use.resolution.request)
        supported = (
            current.status == W06_R03_UNDERSTANDING_UNIQUE
            and current.selected == use.resolution.selected
            and current.propositions == use.resolution.propositions
            and current.evidence_keys == use.resolution.evidence_keys
        )
        outcome = W06R03UnderstandingOutcome(
            use,
            W06_R03_OUTCOME_SUPPORT if supported else W06_R03_OUTCOME_REFUTE,
            current.status,
            LosslessIntegerKey((
                W06_R03_RUNTIME_NAMESPACE,
                300,
                len(self._outcomes) + 1,
                *pack_key(use.use_key.components),
            )),
        )
        self._outcomes.append(outcome)
        return outcome

    def state_key(self) -> tuple:
        """返回 resolution、Use 与 outcome 的完整稳定状态。"""
        return (
            tuple(item.stable_key() for item in self._resolutions),
            tuple(item.use_key.components for item in self._uses),
            tuple(item.outcome_key.components for item in self._outcomes),
        )


__all__ = ["W06R03UnderstandingRuntime"]
