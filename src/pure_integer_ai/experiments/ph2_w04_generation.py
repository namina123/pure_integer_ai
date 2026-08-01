"""W-04 generation consumer：primitive 坐标到合法 surface 选择。"""
from __future__ import annotations

from pure_integer_ai.experiments.ph2_w04_generation_contract import (
    W04_GENERATION_CLARIFY,
    W04_GENERATION_READY,
    W04_GENERATION_UNKNOWN,
    W04GenerationChoice,
    W04GenerationOption,
    W04GenerationOutcome,
    W04GenerationRequest,
    W04GenerationUse,
)
from pure_integer_ai.experiments.ph2_w04_learning import (
    W04PrimitiveSurfaceLearningRuntime,
)


class W04GenerationRuntime:
    """只从 active W-04 Evidence 选择 surface，不接收 evaluator expected。"""

    def __init__(self, learning: W04PrimitiveSurfaceLearningRuntime) -> None:
        if not isinstance(learning, W04PrimitiveSurfaceLearningRuntime):
            raise TypeError("learning 必须是 W04PrimitiveSurfaceLearningRuntime")
        self.learning = learning

    def choose(self, request: W04GenerationRequest) -> W04GenerationChoice:
        """为 primitive/context 返回全部合法 surface option。"""
        options = tuple(sorted((
            W04GenerationOption(item.surface_form, item)
            for item in self.learning.active_candidates()
            if item.primitive_registry == request.primitive_registry
            and item.primitive_kind == request.primitive_kind
            and item.context_text == request.context_text
        ), key=lambda item: item.surface_form))
        if not options:
            status = W04_GENERATION_UNKNOWN
        elif len(options) > 1 and not request.allow_multiple:
            status = W04_GENERATION_CLARIFY
            options = ()
        else:
            status = W04_GENERATION_READY
        return W04GenerationChoice(request, status, options)

    def adopt(
            self,
            choice: W04GenerationChoice,
            selected: tuple[W04GenerationOption, ...],
            ) -> W04GenerationUse:
        """写独立 Use 对象；调用方必须显式选择 option。"""
        return W04GenerationUse(choice, selected)

    def verify_use(self, use: W04GenerationUse) -> W04GenerationOutcome:
        """当前最小 verifier 只确认 Use 仍绑定 active option。"""
        active = set(self.learning.active_candidates())
        verdict = (
            "SUPPORT" if all(item.candidate in active for item in use.selected_options)
            else "REFUTE"
        )
        return W04GenerationOutcome(use, verdict)


def build_w04_generation_runtime(
        learning: W04PrimitiveSurfaceLearningRuntime,
        ) -> W04GenerationRuntime:
    """构建 W-04 generation consumer。"""
    return W04GenerationRuntime(learning)


__all__ = ["W04GenerationRuntime", "build_w04_generation_runtime"]
