"""W-04 understanding consumer：surface/context 到 primitive 候选。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.experiments.ph2_w04_adapter import (
    W04PrimitiveSurfaceCandidate,
)
from pure_integer_ai.experiments.ph2_w04_learning import (
    W04PrimitiveSurfaceLearningRuntime,
)


W04_UNDERSTANDING_UNIQUE = "UNIQUE"
W04_UNDERSTANDING_MULTI = "MULTI"
W04_UNDERSTANDING_UNKNOWN = "UNKNOWN"
W04_UNDERSTANDING_CONFLICT = "CONFLICT"
W04_UNDERSTANDING_CLARIFY = "CLARIFY"


class W04UnderstandingError(RuntimeError):
    """W-04 understanding 查询状态或候选集合非法。"""


@dataclass(frozen=True)
class W04PrimitiveResolution:
    """一次 surface/context 查询的完整候选和严格结果。"""

    status: str
    surface_form: str
    context_text: str
    candidates: tuple[W04PrimitiveSurfaceCandidate, ...]
    selected: W04PrimitiveSurfaceCandidate | None
    clarify_required: bool

    def __post_init__(self) -> None:
        if self.status not in {
            W04_UNDERSTANDING_UNIQUE,
            W04_UNDERSTANDING_MULTI,
            W04_UNDERSTANDING_UNKNOWN,
            W04_UNDERSTANDING_CONFLICT,
            W04_UNDERSTANDING_CLARIFY,
        }:
            raise W04UnderstandingError("understanding status 非法")
        if self.status == W04_UNDERSTANDING_UNIQUE:
            if len(self.candidates) != 1 or self.selected != self.candidates[0]:
                raise W04UnderstandingError("UNIQUE 必须且只能采用一个 active primitive")
        elif self.selected is not None:
            raise W04UnderstandingError("非 UNIQUE 结果不得私选 primitive")
        if self.clarify_required != (
                self.status in {W04_UNDERSTANDING_MULTI, W04_UNDERSTANDING_CLARIFY}):
            raise W04UnderstandingError("clarify_required 与查询状态不一致")


class W04UnderstandingRuntime:
    """只消费 active W-04 lifecycle 投影，不按字符串硬绑定 primitive。"""

    def __init__(self, learning: W04PrimitiveSurfaceLearningRuntime) -> None:
        if not isinstance(learning, W04PrimitiveSurfaceLearningRuntime):
            raise TypeError("learning 必须是 W04PrimitiveSurfaceLearningRuntime")
        self.learning = learning

    def resolve(self, surface_form: str, context_text: str) -> W04PrimitiveResolution:
        """返回完整 active 候选，多候选时要求澄清。"""
        active = tuple(
            item for item in self.learning.active_candidates()
            if item.surface_form == surface_form and item.context_text == context_text
        )
        if not active:
            status = W04_UNDERSTANDING_UNKNOWN
            selected = None
        elif len(active) == 1:
            status = W04_UNDERSTANDING_UNIQUE
            selected = active[0]
        else:
            status = W04_UNDERSTANDING_MULTI
            selected = None
        return W04PrimitiveResolution(
            status,
            surface_form,
            context_text,
            tuple(sorted(active, key=lambda item: item.candidate.stable_key())),
            selected,
            status == W04_UNDERSTANDING_MULTI,
        )


def build_w04_understanding_runtime(
        learning: W04PrimitiveSurfaceLearningRuntime,
        ) -> W04UnderstandingRuntime:
    """构建 W-04 understanding consumer。"""
    return W04UnderstandingRuntime(learning)


__all__ = [
    "W04PrimitiveResolution",
    "W04UnderstandingError",
    "W04UnderstandingRuntime",
    "W04_UNDERSTANDING_CLARIFY",
    "W04_UNDERSTANDING_CONFLICT",
    "W04_UNDERSTANDING_MULTI",
    "W04_UNDERSTANDING_UNIQUE",
    "W04_UNDERSTANDING_UNKNOWN",
    "build_w04_understanding_runtime",
]
