"""W-07 Understanding 与 Reasoning 的独立 exact Use consumer。"""
from __future__ import annotations

from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    LosslessIntegerKey,
)
from pure_integer_ai.experiments.ph2_w07_logic_contract import (
    W07LogicContractError,
    W07LogicOutcome,
    W07LogicRequest,
    W07LogicResolution,
    W07LogicUse,
    pack_key,
)
from pure_integer_ai.experiments.ph2_w07_logic_shared import (
    W07LogicView,
    W07_LOGIC_RUNTIME_NAMESPACE,
)


def _status(execution) -> str:
    if execution is None:
        return "NO_ADOPTION"
    return {
        (True, False): "SUPPORTED",
        (False, True): "REFUTED",
        (False, False): "UNKNOWN",
        (True, True): "CONFLICT",
    }[(execution.evaluation.state.support, execution.evaluation.state.refute)]


class W07LogicConsumerRuntime:
    """一个 consumer 一个 request/Use/outcome ledger，不做共享 fanout。"""

    def __init__(self, view: W07LogicView, consumer: str) -> None:
        if not isinstance(view, W07LogicView):
            raise TypeError("W-07 consumer view 类型非法")
        if consumer not in {"UNDERSTANDING", "REASONING"}:
            raise W07LogicContractError("W-07 consumer 必须是 U 或 R")
        self.view = view
        self.consumer = consumer
        self._request_keys: set[LosslessIntegerKey] = set()
        self._resolutions: list[W07LogicResolution] = []
        self._uses: list[W07LogicUse] = []
        self._outcomes: list[W07LogicOutcome] = []

    @property
    def resolutions(self) -> tuple[W07LogicResolution, ...]:
        return tuple(self._resolutions)

    @property
    def uses(self) -> tuple[W07LogicUse, ...]:
        return tuple(self._uses)

    @property
    def outcomes(self) -> tuple[W07LogicOutcome, ...]:
        return tuple(self._outcomes)

    def preview(self, request: W07LogicRequest) -> W07LogicResolution:
        if not isinstance(request, W07LogicRequest):
            raise TypeError("W-07 logic preview request 类型非法")
        execution = (
            self.view.execute(request)
            if self.view.protocol.connected(self.consumer) else None)
        status = _status(execution)
        return W07LogicResolution(
            self.consumer,
            request,
            status,
            execution,
            LosslessIntegerKey((
                W07_LOGIC_RUNTIME_NAMESPACE,
                100 if self.consumer == "UNDERSTANDING" else 110,
                1 + (
                    "SUPPORTED", "REFUTED", "UNKNOWN", "CONFLICT",
                    "NO_ADOPTION",
                ).index(status),
                *self.view.protocol.stable_key(),
            )),
        )

    def resolve(self, request: W07LogicRequest) -> W07LogicResolution:
        if request.request_key in self._request_keys:
            raise W07LogicContractError("重复 W-07 logic request key")
        resolution = self.preview(request)
        self._request_keys.add(request.request_key)
        self._resolutions.append(resolution)
        return resolution

    def adopt(self, resolution: W07LogicResolution) -> W07LogicUse:
        if resolution not in self._resolutions:
            raise W07LogicContractError("logic resolution 不属于当前 consumer")
        if resolution.execution is None:
            raise W07LogicContractError("NO_ADOPTION resolution 不得形成 Use")
        if any(item.resolution == resolution for item in self._uses):
            raise W07LogicContractError("同一 logic resolution 不得重复采用")
        use = W07LogicUse(
            resolution,
            LosslessIntegerKey((
                W07_LOGIC_RUNTIME_NAMESPACE,
                200 if self.consumer == "UNDERSTANDING" else 210,
                len(self._uses) + 1,
                *pack_key(resolution.stable_key()),
            )),
        )
        self._uses.append(use)
        return use

    def verify(self, use: W07LogicUse) -> W07LogicOutcome:
        if use not in self._uses:
            raise W07LogicContractError("logic Use 不属于当前 consumer")
        current = self.preview(use.resolution.request)
        supported = (
            current.execution is not None
            and use.resolution.execution is not None
            and current.execution.stable_key()
            == use.resolution.execution.stable_key()
        )
        outcome = W07LogicOutcome(
            use,
            "SUPPORT" if supported else "REFUTE",
            current.status,
            LosslessIntegerKey((
                W07_LOGIC_RUNTIME_NAMESPACE,
                300 if self.consumer == "UNDERSTANDING" else 310,
                len(self._outcomes) + 1,
                *pack_key(use.use_key.components),
            )),
        )
        self._outcomes.append(outcome)
        return outcome

    def state_key(self) -> tuple:
        return (
            self.consumer,
            tuple(item.stable_key() for item in self._resolutions),
            tuple(item.use_key.components for item in self._uses),
            tuple(item.outcome_key.components for item in self._outcomes),
        )


class W07LogicUnderstandingRuntime(W07LogicConsumerRuntime):
    def __init__(self, view: W07LogicView) -> None:
        super().__init__(view, "UNDERSTANDING")


class W07LogicReasoningRuntime(W07LogicConsumerRuntime):
    def __init__(self, view: W07LogicView) -> None:
        super().__init__(view, "REASONING")


__all__ = [
    "W07LogicConsumerRuntime",
    "W07LogicReasoningRuntime",
    "W07LogicUnderstandingRuntime",
]
