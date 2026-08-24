"""T1-G8：把 G7 多表层选择接到 G6 focus shadow。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.experiments.conversation_dlg_raw16_surface_shadow import (
    SurfaceShadowPlan,
)
from pure_integer_ai.experiments.conversation_dlg_raw16_surface_structure_learner import (
    SurfaceStructureRequest,
)
from pure_integer_ai.experiments.conversation_raw_t1_shadow_adapter import (
    RawT1ShadowAdapterResult,
)
from pure_integer_ai.experiments.conversation_raw_t1_shadow_dialogue import (
    RawT1ShadowDialogueState,
    RawT1ShadowDialogueTurn,
    run_raw_t1_shadow_dialogue_turn,
)
from pure_integer_ai.experiments.conversation_raw_t1_surface_variants import (
    VARIANT_SELECTED,
    SurfaceVariantModel,
    SurfaceVariantResult,
    realize_surface_variants,
)
from pure_integer_ai.experiments.conversation_raw_t1_surface_order import (
    ORDER_SELECTED,
    SurfaceOrderModel,
    SurfaceOrderResult,
    realize_surface_order,
)


T1_G8_FOCUS_SURFACE_PROTOCOL_V1 = 1


class FocusSurfaceShadowError(ValueError):
    """G7 surface 与 G6 focus/qualification 不能无损对齐。"""


class FocusOrderShadowError(ValueError):
    """G9 role-order surface 与 G6 focus/qualification 不能无损对齐。"""


# object-model: value; representation=struct; interop=T1-G8
@dataclass(frozen=True, slots=True)
class FocusSurfaceShadowResult:
    dialogue_turn: RawT1ShadowDialogueTurn
    variant_result: SurfaceVariantResult | None

    def __post_init__(self) -> None:
        if not isinstance(self.dialogue_turn, RawT1ShadowDialogueTurn):
            raise TypeError("dialogue_turn 类型错误")
        if self.dialogue_turn.adapter_result.replaced != 0:
            raise FocusSurfaceShadowError("G8 shadow 不得替换旧答案")
        consumer = self.dialogue_turn.adapter_result.consumer
        if consumer.response_act == "ANSWER":
            if (not isinstance(self.variant_result, SurfaceVariantResult)
                    or self.variant_result.status_code != VARIANT_SELECTED):
                raise FocusSurfaceShadowError("ANSWER 必须有 selected G7 variant")
        elif self.variant_result is not None:
            raise FocusSurfaceShadowError("UNKNOWN/CLARIFY 不得携带表层 claim")

    @property
    def replaced(self) -> int:
        return 0

    @property
    def surface(self) -> str | None:
        return None if self.variant_result is None else self.variant_result.surface

    def canonical_record(self) -> tuple[int, ...]:
        turn = self.dialogue_turn.canonical_record()
        result = [T1_G8_FOCUS_SURFACE_PROTOCOL_V1, self.replaced,
                  len(turn), *turn]
        if self.variant_result is None:
            result.append(0)
        else:
            record = self.variant_result.canonical_record()
            result.extend((len(record), *record))
        return tuple(result)


# object-model: value; representation=struct; interop=T1-G9
@dataclass(frozen=True, slots=True)
class FocusOrderShadowResult:
    dialogue_turn: RawT1ShadowDialogueTurn
    order_result: SurfaceOrderResult | None

    def __post_init__(self) -> None:
        if not isinstance(self.dialogue_turn, RawT1ShadowDialogueTurn):
            raise TypeError("dialogue_turn 类型错误")
        if self.dialogue_turn.adapter_result.replaced != 0:
            raise FocusOrderShadowError("G9 shadow 不得替换旧答案")
        consumer = self.dialogue_turn.adapter_result.consumer
        if consumer.response_act == "ANSWER":
            if (not isinstance(self.order_result, SurfaceOrderResult)
                    or self.order_result.status_code != ORDER_SELECTED):
                raise FocusOrderShadowError("ANSWER 必须有 selected G9 order")
        elif self.order_result is not None:
            raise FocusOrderShadowError("UNKNOWN/CLARIFY 不得携带表层 claim")

    @property
    def replaced(self) -> int:
        return 0

    @property
    def surface(self) -> str | None:
        return None if self.order_result is None else self.order_result.surface

    def canonical_record(self) -> tuple[int, ...]:
        turn = self.dialogue_turn.canonical_record()
        result = [T1_G8_FOCUS_SURFACE_PROTOCOL_V1, self.replaced,
                  len(turn), *turn]
        if self.order_result is None:
            result.append(0)
        else:
            record = self.order_result.canonical_record()
            result.extend((len(record), *record))
        return tuple(result)


def _variant_request(plan: SurfaceShadowPlan, selection_ordinal: int) -> SurfaceStructureRequest:
    if type(selection_ordinal) is not int or selection_ordinal < 0:
        raise FocusSurfaceShadowError("selection_ordinal 必须是非负严格整数")
    return SurfaceStructureRequest(
        plan.semantic, plan.dialogue_act, plan.register, plan.ordered_roles,
        plan.min_chars, plan.max_chars, selection_ordinal,
        plan.authorized_source_ids[0], plan.context_id, plan.family_id,
        plan.slot_values,
    )


def run_focus_surface_shadow_turn(
        state: RawT1ShadowDialogueState,
        adapter_result: RawT1ShadowAdapterResult,
        variant_model: SurfaceVariantModel,
        *,
        selection_ordinal: int = 0,
        ) -> FocusSurfaceShadowResult:
    """推进 G6 focus，并只为 ANSWER 选择 G7 variant。"""
    if not isinstance(variant_model, SurfaceVariantModel):
        raise TypeError("variant_model 类型错误")
    turn = run_raw_t1_shadow_dialogue_turn(state, adapter_result)
    consumer = adapter_result.consumer
    if consumer.response_act == "ANSWER":
        result = realize_surface_variants(
            variant_model,
            _variant_request(adapter_result.shadow.plan, selection_ordinal),
        )
        return FocusSurfaceShadowResult(turn, result)
    if consumer.response_act not in {"UNKNOWN", "CLARIFY"}:
        raise FocusSurfaceShadowError("G8 response-act 未注册")
    if adapter_result.shadow.plan.required_proposition_ids:
        raise FocusSurfaceShadowError("UNKNOWN/CLARIFY obligation 不得携带 claim")
    return FocusSurfaceShadowResult(turn, None)


def run_focus_order_shadow_turn(
        state: RawT1ShadowDialogueState,
        adapter_result: RawT1ShadowAdapterResult,
        order_model: SurfaceOrderModel,
        ) -> FocusOrderShadowResult:
    """推进 G6 focus，并只为 ANSWER 选择 G9 role-order surface。"""
    if not isinstance(order_model, SurfaceOrderModel):
        raise TypeError("order_model 类型错误")
    turn = run_raw_t1_shadow_dialogue_turn(state, adapter_result)
    consumer = adapter_result.consumer
    if consumer.response_act == "ANSWER":
        result = realize_surface_order(
            order_model,
            _variant_request(adapter_result.shadow.plan, 0),
        )
        return FocusOrderShadowResult(turn, result)
    if consumer.response_act not in {"UNKNOWN", "CLARIFY"}:
        raise FocusOrderShadowError("G9 response-act 未注册")
    if adapter_result.shadow.plan.required_proposition_ids:
        raise FocusOrderShadowError("UNKNOWN/CLARIFY obligation 不得携带 claim")
    return FocusOrderShadowResult(turn, None)


__all__ = [
    "T1_G8_FOCUS_SURFACE_PROTOCOL_V1", "FocusOrderShadowError",
    "FocusOrderShadowResult", "FocusSurfaceShadowError", "FocusSurfaceShadowResult",
    "run_focus_order_shadow_turn", "run_focus_surface_shadow_turn",
]
