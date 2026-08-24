"""T1-G4：source-qualified response-act 到 DLG-RAW-16 只读 shadow 的薄适配器。

适配器只核对已确认的 source/context/family、proposition 和 response-act，然后调用既有
表层结构 shadow。它不改变默认 terminal、会话状态或 Memory；任何不一致在进入表层前拒绝。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.experiments.conversation_dlg_raw16_surface_shadow import (
    SurfaceShadowPlan,
    SurfaceShadowResult,
    SurfaceShadowError,
    run_surface_shadow,
)
from pure_integer_ai.experiments.conversation_dlg_raw16_surface_structure_learner import (
    SurfaceStructureModel,
)
from pure_integer_ai.experiments.conversation_raw_proposition_consumer import (
    RawPropositionConsumerResult,
)


RAW_T1_SHADOW_ADAPTER_PROTOCOL_V1 = 1


class RawT1ShadowAdapterError(ValueError):
    """T1 qualification 与表层 shadow plan 不能无损对齐。"""


@dataclass(frozen=True, slots=True)
class RawT1ShadowAdapterResult:
    """source-qualified shadow 结果；``replaced`` 永远为 0。"""

    consumer: RawPropositionConsumerResult
    shadow: SurfaceShadowResult
    trace: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.consumer, RawPropositionConsumerResult):
            raise TypeError("adapter.consumer 类型错误")
        if not isinstance(self.shadow, SurfaceShadowResult):
            raise TypeError("adapter.shadow 类型错误")
        if not isinstance(self.trace, tuple) or not self.trace \
                or any(type(item) is not int or item < 0 for item in self.trace):
            raise RawT1ShadowAdapterError("adapter.trace 非法")

    @property
    def replaced(self) -> int:
        """旁路 shadow 不替换旧答案。"""
        return 0

    def canonical_record(self) -> tuple[int, ...]:
        """返回 source-qualified shadow 的整数回放记录。"""
        consumer_record = self.consumer.integer_record
        shadow_record = self.shadow.canonical_record()
        return (
            RAW_T1_SHADOW_ADAPTER_PROTOCOL_V1, self.replaced,
            len(consumer_record), *consumer_record,
            len(shadow_record), *shadow_record,
            len(self.trace), *self.trace,
        )


def run_raw_t1_shadow_adapter(
        model: SurfaceStructureModel,
        consumer: RawPropositionConsumerResult,
        plan: SurfaceShadowPlan,
        ) -> RawT1ShadowAdapterResult:
    """核对 T1 obligation 后运行现有 DLG-RAW-16 只读表层 shadow。"""
    if not isinstance(model, SurfaceStructureModel):
        raise TypeError("adapter.model 类型错误")
    if not isinstance(consumer, RawPropositionConsumerResult):
        raise TypeError("adapter.consumer 类型错误")
    if not isinstance(plan, SurfaceShadowPlan):
        raise TypeError("adapter.plan 类型错误")
    if plan.dialogue_act != consumer.response_act:
        raise RawT1ShadowAdapterError("response-act obligation 与 shadow plan 不一致")
    if consumer.source_id not in plan.authorized_source_ids:
        raise RawT1ShadowAdapterError("shadow plan 未授权 qualification source")
    if plan.context_id != consumer.context_id or plan.family_id != consumer.family_id:
        raise RawT1ShadowAdapterError("shadow plan context/family 与 qualification 漂移")
    if consumer.state == "SUPPORTED":
        if consumer.proposition_id not in plan.required_proposition_ids:
            raise RawT1ShadowAdapterError("ANSWER shadow 未绑定 qualified proposition")
    elif plan.required_proposition_ids:
        raise RawT1ShadowAdapterError(
            f"{consumer.state} shadow 不得携带 required proposition")
    try:
        shadow = run_surface_shadow(model, plan)
    except SurfaceShadowError as error:
        raise RawT1ShadowAdapterError("表层 shadow plan 拒绝 qualification") from error
    values = (*consumer.integer_record, *shadow.canonical_record())
    if not values or any(type(value) is not int or value < 0 for value in values):
        raise RawT1ShadowAdapterError("adapter trace 非法")
    trace = tuple(values)
    return RawT1ShadowAdapterResult(consumer, shadow, trace)


__all__ = [
    "RAW_T1_SHADOW_ADAPTER_PROTOCOL_V1", "RawT1ShadowAdapterError",
    "RawT1ShadowAdapterResult", "run_raw_t1_shadow_adapter",
]
