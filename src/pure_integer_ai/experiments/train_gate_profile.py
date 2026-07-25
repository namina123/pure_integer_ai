"""正式训练生产路径使用的 context-local gate 配置。"""
from __future__ import annotations

import contextvars

from pure_integer_ai.config import gates


_PRODUCTION_TRAINING_GATE_OVERRIDES: dict[str, bool] = {
    "SENSE_LOOKUP_MODE": True,
    "CUE_EXTRACTOR_MODE": True,
    "ORDINAL_SURFACE_MODE": True,
    "DISPATCH_TOKEN_CHAIN_MODE": True,
    "OUTPUT_WORD_REWARD_MODE": True,
    "G5_C_CONSOLIDATE_MODE": True,
    "EMERGENT_RELATION_HYPOTHESIS_MODE": True,
    "EMERGENT_RELATION_FEED_MODE": True,
    "EMERGENT_RELATION_CUE_READBACK_MODE": True,
    "REALIZES_MODE": True,
    "CUE_CLUSTER_MODE": True,
    "ORACLE_PROMOTE_MODE": True,
    "CORRESPONDENCE_SLOT_MODE": True,
    "COMPOSES_COMBINE_MODE": True,
    "CUE_SLOT_FILL_MODE": True,
    "SLOT_LCA_CONSTRAINT_MODE": True,
    "OPERATOR_D11_READBACK_MODE": True,
    "MODAL_D11_READBACK_MODE": True,
    "NEGATION_D11_READBACK_MODE": True,
    "CAUSES_REWARD_DOMAIN_FILTER_MODE": True,
    "SIMILAR_SLOT_MODE": True,
    "PRONOUN_SLOT_MODE": True,
    "SELECTION_PREF_MODE": True,
    "M1_INTENT_CLASSIFY_MODE": True,
    "COOCCURS_WINDOW_MODE": True,
    "COOCCURS_DEDUP_MODE": True,
    "PRECEDES_DEDUP_MODE": True,
    "CAUSES_DEDUP_MODE": True,
    "HOTZONE_MODE": True,
    "PR_B2_LARGE_N_MODE": True,
    "PRECEDES_OR_MODE": True,
    "PRECEDES_OI_MODE": True,
    "PRONOUN_INTRASEG_MODE": True,
    "MODIFIER_DIRECTION_MODE": True,
    "PRONOUN_RESOLVE_COUNT_MODE": True,
    "EXCLUDE_FUNCTION_MODE": True,
    "TIME_SEQ_PROOF_MODE": True,
    "NUMERIC_PROOF_MODE": True,
    "UNIVERSAL_PROOF_MODE": True,
    "EXISTENTIAL_PROOF_MODE": True,
    "COMPARISON_PROOF_MODE": True,
    "PROPOSITION_MODE": True,
    "NEGATION_MODE": True,
    "MODALITY_MODE": True,
    "DEGREE_MODE": True,
}


def production_training_gate_overrides() -> dict[str, bool]:
    """返回正式训练 gate 配置副本，避免调用方形成可变共享状态。"""
    return dict(_PRODUCTION_TRAINING_GATE_OVERRIDES)


def push_production_training_gates(
        ) -> contextvars.Token[tuple[dict[str, bool], ...]]:
    """把正式训练 gate 配置压入当前执行上下文并返回复位 token。"""
    return gates.push_gate_overrides(production_training_gate_overrides())


def reset_production_training_gates(
        token: contextvars.Token[tuple[dict[str, bool], ...]]) -> None:
    """精确复位由 ``push_production_training_gates`` 压入的配置。"""
    gates.reset_gate_overrides(token)


__all__ = [
    "production_training_gate_overrides",
    "push_production_training_gates",
    "reset_production_training_gates",
]
