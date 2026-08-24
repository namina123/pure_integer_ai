"""T1-G28：G27 relation admission 到 G10 surface/focus shadow 的薄桥接。

本模块不创建新表层算法，只把已闭合的 G27 consumer 交给 G4 adapter 与 G10 unified focus。
默认 terminal、Memory 和持久状态均不触碰。
"""
from __future__ import annotations

from pure_integer_ai.experiments.conversation_dlg_raw16_surface_shadow import (
    SurfaceShadowPlan,
)
from pure_integer_ai.experiments.conversation_dlg_raw16_surface_structure_learner import (
    SurfaceStructureModel,
)
from pure_integer_ai.experiments.conversation_raw_t1_shadow_adapter import (
    run_raw_t1_shadow_adapter,
)
from pure_integer_ai.experiments.conversation_raw_t1_shadow_dialogue import (
    RawT1ShadowDialogueState,
)
from pure_integer_ai.experiments.conversation_raw_t1_exact_relation_admission import (
    RawT1ExactRelationAdmission,
)
from pure_integer_ai.experiments.conversation_raw_t1_surface_plan import (
    SurfacePlanModel,
    UnifiedFocusSurfaceResult,
    run_unified_focus_surface_turn,
)


RAW_T1_RELATION_SURFACE_SHADOW_PROTOCOL_V1 = 1


class RawT1RelationSurfaceShadowError(ValueError):
    """G27 consumer 到 G10 shadow 桥接失败。"""


def run_exact_relation_surface_shadow(
        admission: RawT1ExactRelationAdmission,
        legacy_structure_model: SurfaceStructureModel,
        shadow_plan: SurfaceShadowPlan,
        focus_state: RawT1ShadowDialogueState,
        surface_plan_model: SurfacePlanModel,
        *,
        selection_ordinal: int = 0,
        ) -> UnifiedFocusSurfaceResult:
    """把 G27 consumer 送入 G4/G10/G6；只返回 shadow 结果。"""
    if type(admission) is not RawT1ExactRelationAdmission:
        raise TypeError("surface shadow 需要 RawT1ExactRelationAdmission")
    if type(legacy_structure_model) is not SurfaceStructureModel:
        raise TypeError("surface shadow 需要 legacy structure model")
    if type(shadow_plan) is not SurfaceShadowPlan:
        raise TypeError("surface shadow 需要 SurfaceShadowPlan")
    if type(focus_state) is not RawT1ShadowDialogueState:
        raise TypeError("surface shadow 需要 focus state")
    if type(surface_plan_model) is not SurfacePlanModel:
        raise TypeError("surface shadow 需要 unified surface plan model")
    if type(selection_ordinal) is not int or selection_ordinal < 0:
        raise RawT1RelationSurfaceShadowError("selection_ordinal 必须是非负严格整数")
    try:
        adapter = run_raw_t1_shadow_adapter(
            legacy_structure_model, admission.consumer, shadow_plan)
        return run_unified_focus_surface_turn(
            focus_state, adapter, surface_plan_model,
            selection_ordinal=selection_ordinal,
        )
    except (TypeError, ValueError, RuntimeError) as error:
        raise RawT1RelationSurfaceShadowError("relation surface shadow 失败") from error


__all__ = [
    "RAW_T1_RELATION_SURFACE_SHADOW_PROTOCOL_V1",
    "RawT1RelationSurfaceShadowError",
    "run_exact_relation_surface_shadow",
]
