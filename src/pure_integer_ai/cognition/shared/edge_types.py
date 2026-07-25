"""cognition.shared.edge_types — 边类型枚举（cognition 面向别名·源头在 storage）。

边类型常量归 storage/edge_types.py（algorithm/vm 下层需引·守依赖律）。
本模块 re-export 供 cognition 层引用（cognition/shared 是卷一/二/三公共依赖点）。
"""
from __future__ import annotations

from pure_integer_ai.storage.edge_types import (
    EDGE_PRECEDES, EDGE_CAUSES, EDGE_IS_A, EDGE_PROPERTY, EDGE_CONDITION,
    EDGE_REFERS_TO, EDGE_COOCCURS, EDGE_COMPOSES, EDGE_T_STEP,
    EDGE_SPATIAL_ADJ, EDGE_QUARANTINE_LINK, EDGE_CLOSURE,
    EDGE_CALLS, EDGE_INSTANTIATES, EDGE_TOPO_GENERALIZES, EDGE_STRUCT_BIND,
    EDGE_IMPLEMENTS_BY, EDGE_RELATION_SIGNAL, EDGE_FUNCTION_CLASS, EDGE_ROLE_STAT,
    EDGE_SIMILAR,
    EDGE_TYPE_NAME,
    REGISTERED_EDGE_TYPES, is_registered_edge_type,
)

__all__ = [
    "EDGE_PRECEDES", "EDGE_CAUSES", "EDGE_IS_A", "EDGE_PROPERTY", "EDGE_CONDITION",
    "EDGE_REFERS_TO", "EDGE_COOCCURS", "EDGE_COMPOSES", "EDGE_T_STEP",
    "EDGE_SPATIAL_ADJ", "EDGE_QUARANTINE_LINK", "EDGE_CLOSURE",
    "EDGE_CALLS", "EDGE_INSTANTIATES", "EDGE_TOPO_GENERALIZES", "EDGE_STRUCT_BIND",
    "EDGE_IMPLEMENTS_BY", "EDGE_RELATION_SIGNAL", "EDGE_FUNCTION_CLASS", "EDGE_ROLE_STAT",
    "EDGE_SIMILAR",
    "EDGE_TYPE_NAME",
    "REGISTERED_EDGE_TYPES", "is_registered_edge_type",
]
