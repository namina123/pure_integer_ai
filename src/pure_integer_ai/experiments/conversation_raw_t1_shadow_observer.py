"""T1-G31：三态 relation/surface shadow 的 developer-only 只读观察投影。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.experiments.conversation_raw_t1_surface_plan import (
    UnifiedFocusSurfaceResult,
)


RAW_T1_SHADOW_OBSERVER_PROTOCOL_V1 = 1


class RawT1ShadowObserverError(ValueError):
    """shadow observer 输入或零 claim 不变量非法。"""


@dataclass(frozen=True, slots=True)
class RawT1ShadowObservation:
    """不推进状态、不写介质的观察记录。"""

    response_act: str
    state: str
    focus_revision: int
    replaced: int
    surface_scalars: tuple[int, ...]
    surface_u8: tuple[int, ...]
    plan_selected: int
    trace: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.response_act not in {"ANSWER", "UNKNOWN", "CLARIFY"}:
            raise RawT1ShadowObserverError("response_act 未注册")
        if self.state not in {"SUPPORTED", "UNKNOWN", "CONFLICT"}:
            raise RawT1ShadowObserverError("state 未注册")
        if type(self.focus_revision) is not int or self.focus_revision <= 0:
            raise RawT1ShadowObserverError("focus_revision 非法")
        if self.replaced != 0:
            raise RawT1ShadowObserverError("observer 只允许 replaced=0")
        for name, values, high in (
                ("surface_scalars", self.surface_scalars, 0x10FFFF),
                ("surface_u8", self.surface_u8, 255),
                ("trace", self.trace, None)):
            if not isinstance(values, tuple) or any(
                    type(item) is not int or item < 0 or
                    (high is not None and item > high) for item in values):
                raise RawT1ShadowObserverError(f"{name} 非法")
        if self.plan_selected not in (0, 1):
            raise RawT1ShadowObserverError("plan_selected 必须为 0/1")
        if self.response_act == "ANSWER":
            if not self.surface_scalars or not self.surface_u8 or self.plan_selected != 1:
                raise RawT1ShadowObserverError("ANSWER observer 缺 surface/plan")
        elif self.surface_scalars or self.surface_u8 or self.plan_selected != 0:
            raise RawT1ShadowObserverError("非 ANSWER observer 必须零 surface/plan")

    @property
    def surface(self) -> str | None:
        return ("".join(chr(value) for value in self.surface_scalars)
                if self.surface_scalars else None)

    def canonical_record(self) -> tuple[int, ...]:
        return (RAW_T1_SHADOW_OBSERVER_PROTOCOL_V1,
                len(self.response_act), *(ord(item) for item in self.response_act),
                len(self.state), *(ord(item) for item in self.state),
                self.focus_revision, self.replaced, self.plan_selected,
                len(self.surface_scalars), *self.surface_scalars,
                len(self.surface_u8), *self.surface_u8,
                len(self.trace), *self.trace)


def observe_raw_t1_shadow_result(
        result: UnifiedFocusSurfaceResult,
        ) -> RawT1ShadowObservation:
    """从已完成 shadow result 读取观察值，不修改 result 或任何外部状态。"""
    if type(result) is not UnifiedFocusSurfaceResult:
        raise TypeError("observer 需要 UnifiedFocusSurfaceResult")
    consumer = result.dialogue_turn.adapter_result.consumer
    if result.surface is None:
        scalars: tuple[int, ...] = ()
        output_u8: tuple[int, ...] = ()
        selected = 0
    else:
        scalars = result.plan_result.output_scalars
        output_u8 = result.plan_result.output_bytes
        selected = 1
    return RawT1ShadowObservation(
        consumer.response_act, consumer.state,
        result.dialogue_turn.after.focus_revision,
        result.replaced, scalars, output_u8, selected,
        result.canonical_record(),
    )


def render_raw_t1_shadow_observation_zh(
        observation: RawT1ShadowObservation,
        ) -> str:
    """将只读 observation 渲染成人类可读中文，不生成命令或写回。"""
    if type(observation) is not RawT1ShadowObservation:
        raise TypeError("renderer 需要 RawT1ShadowObservation")
    surface = observation.surface if observation.surface is not None else "（无表层输出）"
    return ("T1 developer-only shadow 观察（只读）\n"
            f"行为：{observation.response_act}；状态：{observation.state}；"
            f"focus revision：{observation.focus_revision}\n"
            f"表层：{surface}\n"
            f"replaced：{observation.replaced}；plan_selected：{observation.plan_selected}\n"
            "说明：该观察不代表自动理解、现实真值或开放域问答。")


__all__ = [
    "RAW_T1_SHADOW_OBSERVER_PROTOCOL_V1",
    "RawT1ShadowObservation", "RawT1ShadowObserverError",
    "observe_raw_t1_shadow_result", "render_raw_t1_shadow_observation_zh",
]
