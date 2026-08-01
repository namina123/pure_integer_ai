"""W-04 reasoning consumer：只消费已授权 primitive 坐标和 Evidence。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.experiments.ph2_w04_adapter import (
    W04PrimitiveSurfaceCandidate,
)
from pure_integer_ai.experiments.ph2_w04_learning import (
    W04PrimitiveSurfaceLearningRuntime,
)


W04_REASONING_AUTHORIZED = "AUTHORIZED"
W04_REASONING_REJECTED = "REJECTED"


@dataclass(frozen=True)
class W04ReasoningUse:
    """推理侧对一个 primitive 坐标的受限采用记录。"""

    primitive_registry: str
    primitive_kind: int
    status: str
    evidence_count: int


class W04ReasoningRuntime:
    """不把外部关系事实当自产证明，只检查 W-04 active Evidence。"""

    def __init__(self, learning: W04PrimitiveSurfaceLearningRuntime) -> None:
        if not isinstance(learning, W04PrimitiveSurfaceLearningRuntime):
            raise TypeError("learning 必须是 W04PrimitiveSurfaceLearningRuntime")
        self.learning = learning

    def authorize(
            self,
            primitive_registry: str,
            primitive_kind: int,
            ) -> W04ReasoningUse:
        """返回 primitive 坐标是否已有 active W-04 Evidence 授权。"""
        active = tuple(
            item for item in self.learning.active_candidates()
            if item.primitive_registry == primitive_registry
            and item.primitive_kind == primitive_kind
        )
        return W04ReasoningUse(
            primitive_registry,
            primitive_kind,
            W04_REASONING_AUTHORIZED if active else W04_REASONING_REJECTED,
            len(active),
        )


def build_w04_reasoning_runtime(
        learning: W04PrimitiveSurfaceLearningRuntime,
        ) -> W04ReasoningRuntime:
    """构建 W-04 reasoning consumer。"""
    return W04ReasoningRuntime(learning)


__all__ = [
    "W04ReasoningRuntime",
    "W04ReasoningUse",
    "W04_REASONING_AUTHORIZED",
    "W04_REASONING_REJECTED",
    "build_w04_reasoning_runtime",
]
