"""语言结构学习路径的机器可读状态。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, TYPE_CHECKING

from pure_integer_ai.cognition.result.graph_view import ConceptGraph
from pure_integer_ai.storage.backend import StorageBackend
from pure_integer_ai.storage.composes_attr import (
    ATTR_RELATION_PRIMITIVE,
    read_composes_attrs,
)
from pure_integer_ai.storage.edge_store import SOURCE_BARE_TEXT
from pure_integer_ai.storage.edge_types import EDGE_RELATION_SIGNAL
from pure_integer_ai.storage.node_store import TIER_PRIMARY, TIER_SHADOW

if TYPE_CHECKING:
    from pure_integer_ai.cognition.process.structure_discover import DiscoveredOperator
    from pure_integer_ai.cognition.shared.types import ConceptRef


@dataclass
class LanguageStructureStateStats:
    """一次正式训练调用后的算子和已学习 D:11 状态。"""

    operators_total: int = 0
    operators_new: int = 0
    cue_bearing_operators: int = 0
    realizes_operators: int = 0
    realizes_cue_operators: int = 0
    d11_shadow_edges: int = 0
    d11_primary_edges: int = 0

    def to_json(self) -> dict[str, int]:
        return {
            "operators_total": self.operators_total,
            "operators_new": self.operators_new,
            "cue_bearing_operators": self.cue_bearing_operators,
            "realizes_operators": self.realizes_operators,
            "realizes_cue_operators": self.realizes_cue_operators,
            "d11_shadow_edges": self.d11_shadow_edges,
            "d11_primary_edges": self.d11_primary_edges,
        }


def measure_language_structure_state(
        backend: StorageBackend, graph: ConceptGraph,
        operators: Iterable[DiscoveredOperator], *,
        new_operator_refs: frozenset[ConceptRef], space_id: int,
        ) -> LanguageStructureStateStats:
    """测量语言发现池和已学习关系信号层级。

    D:11 只计裸文本关系信号，不计教师 boot 信号，也不计入算子、模态、符号或动作目标。
    """
    unique_ops: dict[ConceptRef, DiscoveredOperator] = {}
    for operator in operators:
        unique_ops.setdefault(operator.skeleton_ref, operator)

    state = LanguageStructureStateStats(
        operators_total=len(unique_ops),
        operators_new=sum(1 for ref in unique_ops if ref in new_operator_refs),
    )
    for operator in unique_ops.values():
        cue_bearing = any(
            cue is not None for cue in graph.read_cue_sig(operator.skeleton_ref))
        realizes = graph.rel_kind_of_skeleton(operator.skeleton_ref) != 0
        if cue_bearing:
            state.cue_bearing_operators += 1
        if realizes:
            state.realizes_operators += 1
        if cue_bearing and realizes:
            state.realizes_cue_operators += 1

    rows = backend.select("edge", where={
        "space_id_from": space_id,
        "edge_type": EDGE_RELATION_SIGNAL,
        "source": SOURCE_BARE_TEXT,
    })
    for row in rows:
        target = (row["space_id_to"], row["local_id_to"])
        attrs = read_composes_attrs(backend, target)
        if attrs.get(ATTR_RELATION_PRIMITIVE, (0, 0))[0] == 0:
            continue
        if row.get("tier") == TIER_SHADOW:
            state.d11_shadow_edges += 1
        elif row.get("tier") == TIER_PRIMARY:
            state.d11_primary_edges += 1
    return state
