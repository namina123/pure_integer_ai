"""W-04 primitive-to-surface 生成 choice/use/outcome 合同。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.experiments.ph2_w04_adapter import (
    W04PrimitiveSurfaceCandidate,
)


W04_GENERATION_READY = "READY"
W04_GENERATION_UNKNOWN = "UNKNOWN"
W04_GENERATION_CLARIFY = "CLARIFY"
W04_GENERATION_REJECTED = "REJECTED"
W04_GENERATION_STATUSES = {
    W04_GENERATION_READY,
    W04_GENERATION_UNKNOWN,
    W04_GENERATION_CLARIFY,
    W04_GENERATION_REJECTED,
}


class W04GenerationError(RuntimeError):
    """W-04 generation choice、Use 或 outcome 非法。"""


@dataclass(frozen=True)
class W04GenerationRequest:
    """从目标 primitive、context 和约束发起的生成请求。"""

    primitive_registry: str
    primitive_kind: int
    context_text: str
    allow_multiple: bool


@dataclass(frozen=True)
class W04GenerationOption:
    """一个经 active Evidence 授权的合法 surface 选项。"""

    surface_form: str
    candidate: W04PrimitiveSurfaceCandidate


@dataclass(frozen=True)
class W04GenerationChoice:
    """不私选首项的完整合法 option 集。"""

    request: W04GenerationRequest
    status: str
    options: tuple[W04GenerationOption, ...]

    def __post_init__(self) -> None:
        if self.status not in W04_GENERATION_STATUSES:
            raise W04GenerationError("generation status 非法")
        if self.status == W04_GENERATION_READY and not self.options:
            raise W04GenerationError("READY choice 缺合法 option")
        if self.status != W04_GENERATION_READY and self.options:
            raise W04GenerationError("非 READY choice 不得泄漏 option")


@dataclass(frozen=True)
class W04GenerationUse:
    """采用一个或多个精确 surface option 的 Use。"""

    choice: W04GenerationChoice
    selected_options: tuple[W04GenerationOption, ...]

    def __post_init__(self) -> None:
        if self.choice.status != W04_GENERATION_READY:
            raise W04GenerationError("只有 READY choice 可形成 Use")
        if (not self.selected_options
                or any(item not in self.choice.options for item in self.selected_options)):
            raise W04GenerationError("Use 必须选择 choice 内 option")


@dataclass(frozen=True)
class W04GenerationOutcome:
    """当前 active Evidence 对 Use 的 outcome。"""

    use: W04GenerationUse
    verdict: str


__all__ = [
    "W04_GENERATION_CLARIFY",
    "W04_GENERATION_READY",
    "W04_GENERATION_REJECTED",
    "W04_GENERATION_STATUSES",
    "W04_GENERATION_UNKNOWN",
    "W04GenerationChoice",
    "W04GenerationError",
    "W04GenerationOption",
    "W04GenerationOutcome",
    "W04GenerationRequest",
    "W04GenerationUse",
]
