"""发现当前样本上全部适用的 legacy verifier adapter。"""
from __future__ import annotations

from collections.abc import Sequence

from pure_integer_ai.cognition.shared.types import (
    MODALITY_ARITH,
    MODALITY_CODE,
    MODALITY_LANGUAGE,
    Segment,
)
from pure_integer_ai.config import gates
from pure_integer_ai.experiments.collection import CollectedItem
from pure_integer_ai.experiments.evaluation_protocol import ProtocolKey

VERIFY_ROUTE_NONE = 0
VERIFY_ROUTE_COMPOSES = 1
VERIFY_ROUTE_NUMERIC = 2
VERIFY_ROUTE_COMPARISON = 3
VERIFY_ROUTE_UNIVERSAL = 4
VERIFY_ROUTE_EXISTENTIAL = 5
VERIFY_ROUTE_OCCURRENCE_ORDER = 6

# 仅保留旧 import 兼容；该 adapter 不表示现实事件时间。
VERIFY_ROUTE_TIME = VERIFY_ROUTE_OCCURRENCE_ORDER

_LANGUAGE_ROUTE_ORDER = (
    VERIFY_ROUTE_NUMERIC,
    VERIFY_ROUTE_COMPARISON,
    VERIFY_ROUTE_UNIVERSAL,
    VERIFY_ROUTE_EXISTENTIAL,
    VERIFY_ROUTE_OCCURRENCE_ORDER,
)


def is_verify_modality(modality: int) -> bool:
    """判断代码或算术模态是否必须走 COMPOSES verifier。"""
    return modality in (MODALITY_CODE, MODALITY_ARITH)


def verification_dimension_key(route: int) -> ProtocolKey:
    """把 legacy route 映射到开放的完整维度键。"""
    if route not in (VERIFY_ROUTE_COMPOSES, *_LANGUAGE_ROUTE_ORDER):
        raise ValueError("未知 legacy verification route")
    return ProtocolKey((route,))


def verification_verifier_key(route: int) -> ProtocolKey:
    """返回 legacy adapter 的版本化 verifier 身份。"""
    if route not in (VERIFY_ROUTE_COMPOSES, *_LANGUAGE_ROUTE_ORDER):
        raise ValueError("未知 legacy verification route")
    return ProtocolKey((1, route))


def _language_route_applies(
        route: int,
        segments: Sequence[Segment],
        ) -> bool:
    """按 gate 和已解析声明判断一个 legacy 语言 adapter 是否适用。"""
    if route == VERIFY_ROUTE_NUMERIC:
        return (
            gates.NUMERIC_PROOF_MODE
            and any(segment.numeric_claims for segment in segments)
        )
    if route == VERIFY_ROUTE_COMPARISON:
        return (
            gates.COMPARISON_PROOF_MODE
            and any(segment.comparison_claims for segment in segments)
        )
    if route == VERIFY_ROUTE_UNIVERSAL:
        return (
            gates.UNIVERSAL_PROOF_MODE
            and any(segment.universal_claims for segment in segments)
        )
    if route == VERIFY_ROUTE_EXISTENTIAL:
        return (
            gates.EXISTENTIAL_PROOF_MODE
            and any(segment.existential_claims for segment in segments)
        )
    if route == VERIFY_ROUTE_OCCURRENCE_ORDER:
        return (
            gates.TIME_SEQ_PROOF_MODE
            and any(segment.precedes_pairs for segment in segments)
        )
    raise ValueError("未知 legacy language verification route")


def select_verification_routes(
        item: CollectedItem,
        segments: Sequence[Segment],
        ) -> tuple[int, ...]:
    """返回全部适用 adapter；不得以第一个命中覆盖后续维度。"""
    if is_verify_modality(item.modality):
        return (VERIFY_ROUTE_COMPOSES,)
    if item.modality != MODALITY_LANGUAGE:
        return ()
    return tuple(
        route
        for route in _LANGUAGE_ROUTE_ORDER
        if _language_route_applies(route, segments)
    )


__all__ = [
    "VERIFY_ROUTE_COMPARISON",
    "VERIFY_ROUTE_COMPOSES",
    "VERIFY_ROUTE_EXISTENTIAL",
    "VERIFY_ROUTE_NONE",
    "VERIFY_ROUTE_NUMERIC",
    "VERIFY_ROUTE_OCCURRENCE_ORDER",
    "VERIFY_ROUTE_TIME",
    "VERIFY_ROUTE_UNIVERSAL",
    "is_verify_modality",
    "select_verification_routes",
    "verification_dimension_key",
    "verification_verifier_key",
]
