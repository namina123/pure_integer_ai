"""把 L-02 分词 Hypothesis 物化为 L-04 递归 Span 图。

每个候选拥有独立根 Span 和完整 root-to-part 路径；同来源同几何的 part 与 atomic
共享 Span 本体，并通过一等 StructureConcept 表达不同结构角色。只有当前 winner 关联 occurrence。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.graph_ontology import GraphStatement
from pure_integer_ai.cognition.shared.hypothesis import HypothesisKey
from pure_integer_ai.cognition.shared.identity import (
    TypedRef,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.scope_identity import LogicalTimestamp
from pure_integer_ai.cognition.understanding.segmentation_hypothesis import (
    SegmentationHypothesisCandidate,
    SegmentationResult,
)
from pure_integer_ai.cognition.understanding.span_index import (
    SpanIndex,
    SpanProtocol,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int


def _structure_key(value, *, where: str) -> tuple[int, ...]:
    """校验调用方注入的一等 StructureConcept 稳定键。"""
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{where} 必须是非空整数 tuple")
    assert_int(*value, _where=where)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{where} 必须使用严格整数")
    return value


@dataclass(frozen=True)
class SegmentationSpanProtocol:
    """注入分词 Span 的关系、层级结构和共享形状命名空间。"""

    span_protocol: SpanProtocol
    document_structure_key: tuple[int, ...]
    part_structure_key: tuple[int, ...]
    candidate_shape_namespace_key: tuple[int, ...]
    atomic_structure_key: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.span_protocol, SpanProtocol):
            raise TypeError("span_protocol 必须是 SpanProtocol")
        keys = [
            _structure_key(
                self.document_structure_key,
                where="SegmentationSpanProtocol.document_structure_key",
            ),
            _structure_key(
                self.part_structure_key,
                where="SegmentationSpanProtocol.part_structure_key",
            ),
            _structure_key(
                self.candidate_shape_namespace_key,
                where="SegmentationSpanProtocol.candidate_shape_namespace_key",
            ),
        ]
        if self.atomic_structure_key is not None:
            keys.append(_structure_key(
                self.atomic_structure_key,
                where="SegmentationSpanProtocol.atomic_structure_key",
            ))
        if len(set(keys)) != len(keys):
            raise ValueError("分词 Span 的结构键和形状命名空间必须互不相同")


@dataclass(frozen=True)
class SegmentationSpanCandidate:
    """一个分词 Hypothesis 对应的根 Span、part Span 和候选断言。"""

    hypothesis: HypothesisKey
    root: TypedRef
    parts: tuple[TypedRef, ...]
    candidate_statement: GraphStatement


@dataclass(frozen=True)
class SegmentationSpanResult:
    """一次完整分词 lattice 的 Span 物化结果。"""

    candidates: tuple[SegmentationSpanCandidate, ...]
    atomic_spans: tuple[TypedRef, ...]
    span_refs: tuple[TypedRef, ...]
    statement_hashes: tuple[int, ...]


class SegmentationSpanMaterializer:
    """把分词候选转换为来源化递归 Span，并同步边界替代事件。"""

    def __init__(self, spans: SpanIndex,
                 protocol: SegmentationSpanProtocol) -> None:
        if not isinstance(spans, SpanIndex):
            raise TypeError("spans 必须是 SpanIndex")
        if not isinstance(protocol, SegmentationSpanProtocol):
            raise TypeError("protocol 必须是 SegmentationSpanProtocol")
        if spans.protocol != protocol.span_protocol:
            raise ValueError("SpanIndex 与 SegmentationSpanProtocol 关系协议不一致")
        self.spans = spans
        self.protocol = protocol
        self._candidates: dict[HypothesisKey, SegmentationSpanCandidate] = {}
        self._result_cache: dict[tuple, SegmentationSpanResult] = {}

    def materialize(
            self, result: SegmentationResult, *,
            occurrence_refs: tuple[TypedRef, ...] = (),
            ) -> SegmentationSpanResult:
        """物化全部候选；只把 occurrence 绑定到当前 selected 候选。"""
        if not isinstance(result, SegmentationResult):
            raise TypeError("result 必须是 SegmentationResult")
        if not isinstance(occurrence_refs, tuple):
            raise TypeError("occurrence_refs 必须是 TypedRef tuple")
        if not result.candidates:
            if occurrence_refs:
                raise ValueError("无分词候选时不得携带 occurrence")
            return SegmentationSpanResult((), (), (), ())
        cache_key = self._cache_key(result, occurrence_refs)
        cached = self._result_cache.get(cache_key)
        if cached is not None:
            return cached
        source, scope = self._source_scope(result.candidates)
        raw_text = result.text
        span_refs: list[TypedRef] = []
        statement_hashes: list[int] = []
        atomic_by_range: dict[tuple[int, int], TypedRef] = {}
        atomic_refs: list[TypedRef] = []

        if self.protocol.atomic_structure_key is not None:
            atomic_structure = structure_concept_identity(
                self.protocol.atomic_structure_key)
            for index in range(len(raw_text)):
                members = ((index, index + 1),)
                atomic = self.spans.ensure_ref(
                    source=source,
                    raw_text=raw_text,
                    scope=scope,
                    members=members,
                    ordinal=0,
                    structures=(atomic_structure,),
                )
                atomic_by_range[(index, index + 1)] = atomic
                atomic_refs.append(atomic)
                span_refs.append(atomic)

        document_structure = structure_concept_identity(
            self.protocol.document_structure_key)
        part_structure = structure_concept_identity(
            self.protocol.part_structure_key)
        materialized: list[SegmentationSpanCandidate] = []
        selected_parts: tuple[TypedRef, ...] | None = None
        for candidate in sorted(
                result.candidates,
                key=lambda item: item.hypothesis.stable_key()):
            root_members = ((0, len(raw_text)),)
            shape_structure = structure_concept_identity(
                self._shape_key(candidate))
            root_ordinal = self.spans.ensure_role_ordinal(
                candidate.hypothesis.object_identity())
            root = self.spans.ensure_ref(
                source=source,
                raw_text=raw_text,
                scope=scope,
                members=root_members,
                ordinal=root_ordinal,
                structures=(document_structure, shape_structure),
            )
            span_refs.append(root)
            part_refs: list[TypedRef] = []
            for part_ordinal, part in enumerate(candidate.segmentation.parts):
                members = ((part.start, part.end),)
                part_ref = self.spans.ensure_ref(
                    source=source,
                    raw_text=raw_text,
                    scope=scope,
                    members=members,
                    ordinal=0,
                    structures=(part_structure,),
                )
                part_refs.append(part_ref)
                span_refs.append(part_ref)
                constituent = self.spans.add_constituent(
                    root,
                    part_ref,
                    member_ordinal=part_ordinal,
                )
                statement_hashes.append(constituent.assertion_hash)
                for atomic_ordinal, index in enumerate(
                        range(part.start, part.end)):
                    atomic = atomic_by_range.get((index, index + 1))
                    if atomic is None or atomic == part_ref:
                        continue
                    atomic_link = self.spans.add_constituent(
                        part_ref,
                        atomic,
                        member_ordinal=atomic_ordinal,
                    )
                    statement_hashes.append(atomic_link.assertion_hash)
            candidate_statement = self.spans.add_candidate(
                candidate.hypothesis,
                root,
            )
            statement_hashes.append(candidate_statement.assertion_hash)
            item = SegmentationSpanCandidate(
                candidate.hypothesis,
                root,
                tuple(part_refs),
                candidate_statement,
            )
            self._candidates[candidate.hypothesis] = item
            materialized.append(item)
            if candidate.hypothesis == result.selected_hypothesis:
                selected_parts = item.parts

        self._link_selected_occurrences(
            selected_parts,
            occurrence_refs,
            statement_hashes,
        )
        unique_refs = tuple(dict.fromkeys(span_refs))
        materialization = SegmentationSpanResult(
            tuple(materialized),
            tuple(atomic_refs),
            unique_refs,
            tuple(sorted(set(statement_hashes))),
        )
        self._result_cache[cache_key] = materialization
        return materialization

    def supersede_candidate(
            self, old: HypothesisKey, new: HypothesisKey,
            timestamp: LogicalTimestamp) -> int:
        """同步边界 Hypothesis 的 replacement 到图内 candidate assertion。"""
        self.validate_candidate_supersede(old, new, timestamp)
        return self.spans.supersede_candidate(old, new, timestamp)

    def validate_candidate_supersede(
            self, old: HypothesisKey, new: HypothesisKey,
            timestamp: LogicalTimestamp) -> None:
        """在 Hypothesis ledger 转换前核验图内候选可安全替代。"""
        if old not in self._candidates or new not in self._candidates:
            raise LookupError("替代前必须先物化两个分词 Span 候选")
        self.spans.validate_candidate_supersede(old, new, timestamp)

    def candidate(
            self, hypothesis: HypothesisKey
            ) -> SegmentationSpanCandidate | None:
        """返回已物化候选，未物化时返回 None。"""
        return self._candidates.get(hypothesis)

    def clone_for_context(
            self, spans: SpanIndex) -> "SegmentationSpanMaterializer":
        """在评测 clone 的 SpanIndex 上重建独立适配器。"""
        return SegmentationSpanMaterializer(spans, self.protocol)

    def _link_selected_occurrences(
            self, selected_parts: tuple[TypedRef, ...] | None,
            occurrence_refs: tuple[TypedRef, ...],
            statement_hashes: list[int]) -> None:
        """核对 winner part 与 occurrence 精确同位后追加关系。"""
        if selected_parts is None:
            if occurrence_refs:
                raise ValueError("无 selected 分词候选时不得携带 occurrence")
            return
        if not occurrence_refs:
            return
        if len(selected_parts) != len(occurrence_refs):
            raise ValueError("selected Span part 数量与 occurrence 数量不一致")
        if self.spans.occurrence_index is None and occurrence_refs:
            raise RuntimeError("关联 occurrence 前必须装配 OccurrenceIndex")
        for part, occurrence in zip(selected_parts, occurrence_refs):
            if not isinstance(occurrence, TypedRef):
                raise TypeError("occurrence_refs 只能包含 TypedRef")
            part_members = self.spans.members_of(part)
            occurrence_record = self.spans.occurrence_index.read(occurrence)
            expected_members = ((
                occurrence_record.start,
                occurrence_record.end,
            ),)
            if part_members != expected_members:
                raise ValueError("selected Span part 与 occurrence 边界不一致")
            statement = self.spans.add_occurrence(
                part,
                occurrence,
                member_ordinal=0,
            )
            statement_hashes.append(statement.assertion_hash)

    def _shape_key(
            self, candidate: SegmentationHypothesisCandidate
            ) -> tuple[int, ...]:
        """生成只含平坦拓扑、不含 surface 和区间长度的共享结构键。"""
        namespace = self.protocol.candidate_shape_namespace_key
        return (
            len(namespace),
            *namespace,
            len(candidate.segmentation.parts),
        )

    @staticmethod
    def _cache_key(
            result: SegmentationResult,
            occurrence_refs: tuple[TypedRef, ...]) -> tuple:
        """构造不含 Evidence 快照的稳定物化键，避免跨 round 重复查询。"""
        return (
            tuple(
                candidate.hypothesis.stable_key()
                for candidate in sorted(
                    result.candidates,
                    key=lambda item: item.hypothesis.stable_key())
            ),
            () if result.selected_hypothesis is None
            else result.selected_hypothesis.stable_key(),
            tuple(
                hypothesis.stable_key()
                for hypothesis in result.adopted_hypotheses
            ),
            tuple(ref.stable_key() for ref in occurrence_refs),
        )

    @staticmethod
    def _source_scope(
            candidates: tuple[SegmentationHypothesisCandidate, ...]
            ):
        """核验整个 lattice 的 observation 和 scope 唯一一致。"""
        first = candidates[0].hypothesis
        for candidate in candidates[1:]:
            hypothesis = candidate.hypothesis
            if (hypothesis.observation != first.observation
                    or hypothesis.scope != first.scope):
                raise ValueError("一个分词 lattice 不得混入多个来源或 scope")
        return first.observation, first.scope


__all__ = [
    "SegmentationSpanCandidate",
    "SegmentationSpanMaterializer",
    "SegmentationSpanProtocol",
    "SegmentationSpanResult",
]
