"""W-06 relation 来源 Span 到语义图端点的冻结整数协议。"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.graph_ontology import (
    relation_concept_identity,
)
from pure_integer_ai.cognition.shared.identity import ObjectIdentity


W06_SPAN_ENDPOINT_RELATION_KEY = (60606, 960, 1)
W06_SPAN_ANCHOR_RELATION_KEY = (60606, 960, 2)


def w06_span_endpoint_predicate() -> ObjectIdentity:
    """返回 Span 到 RoleBinding 的开放关系身份。"""
    return relation_concept_identity(W06_SPAN_ENDPOINT_RELATION_KEY)


def w06_span_anchor_predicate() -> ObjectIdentity:
    """返回 relation anchor Span 到 Proposition 的开放关系身份。"""
    return relation_concept_identity(W06_SPAN_ANCHOR_RELATION_KEY)


__all__ = [
    "W06_SPAN_ANCHOR_RELATION_KEY",
    "W06_SPAN_ENDPOINT_RELATION_KEY",
    "w06_span_anchor_predicate",
    "w06_span_endpoint_predicate",
]
