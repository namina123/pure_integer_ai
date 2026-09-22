"""把训练后关系图的整数 Span 视图适配为查询输入结构。

该适配器只消费图运行时已经恢复的命题、predicate、RoleBinding 和来源区间。
它不解析文本、不建立词表，也不把来源正文写入查询结构；文本表层仅作为当前
Span 的整数值，用于 ``TrainedInputStructureProjector`` 的候选定位。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

from pure_integer_ai.cognition.understanding.query_input_structure import (
    CARRIER_GRAPH_SOURCE,
    CARRIER_SEMANTIC_GRAPH,
    PROJECTION_FILLER,
    PROJECTION_PREDICATE,
    TrainedInputMember,
    TrainedInputStructure,
)
from pure_integer_ai.cognition.understanding.semantic_builder_graph import (
    MaterializedSemanticCandidate,
)


class RelationBindingView(Protocol):
    """活动关系 RoleBinding 的最小只读视图。"""

    role: object
    filler: object
    surface: str
    start: int
    end: int


class RelationSurfaceView(Protocol):
    """活动关系表层/图身份的最小只读视图。"""

    proposition: object
    predicate: object
    cue: str
    bindings: tuple[RelationBindingView, ...]
    source_hash: int
    cue_start: int
    cue_end: int


class QueryStructureAdapterError(ValueError):
    """活动关系到输入结构的整数适配失败。"""


def _identity_key(value: object, *, where: str) -> tuple[int, ...]:
    stable_key = getattr(value, "stable_key", None)
    if stable_key is None or not callable(stable_key):
        raise QueryStructureAdapterError(f"{where} 缺少 stable_key")
    key = stable_key()
    if type(key) is not tuple or not key or any(
            type(item) is not int or item < 0 for item in key):
        raise QueryStructureAdapterError(f"{where}.stable_key 必须是非负整数 tuple")
    return key


def _strict_key(value: tuple[int, ...], *, where: str,
                allow_empty: bool = False) -> tuple[int, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{where} 必须是 tuple")
    if not allow_empty and not value:
        raise ValueError(f"{where} 不能为空")
    if any(type(item) is not int or item < 0 for item in value):
        raise ValueError(f"{where} 只能包含非负严格整数")
    return value


@dataclass(frozen=True, slots=True)
class SemanticCandidateBindingSpan:
    """已物化 S-02 RoleBinding 在训练 Span 图中的纯整数投影。"""

    role_key: tuple[int, ...]
    filler_key: tuple[int, ...]
    values: tuple[int, ...]
    start: int
    end: int
    binding_ordinal: int

    def __post_init__(self) -> None:
        for name in ("role_key", "filler_key", "values"):
            _strict_key(getattr(self, name), where=f"binding span.{name}")
        for name in ("start", "end", "binding_ordinal"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"binding span.{name} 必须是非负严格整数")
        if self.end <= self.start or self.end - self.start != len(self.values):
            raise ValueError("binding span 区间与 values 不闭合")

    def stable_key(self) -> tuple[int, ...]:
        return (
            self.start,
            self.end,
            self.binding_ordinal,
            len(self.role_key),
            *self.role_key,
            len(self.filler_key),
            *self.filler_key,
            len(self.values),
            *self.values,
        )


@dataclass(frozen=True, slots=True)
class SemanticCandidateInputView:
    """`SemanticCandidateGraphAdapter` 恢复结果的只读输入 Span 视图。

    S-02 候选本身不携带来源正文或 Span 表层；宿主必须显式提供已核验的
    predicate/RoleBinding 整数 Span。该视图只形成开放解析候选，绝不添加
    Evidence 或把候选当作可回答的事实。
    """

    candidate: MaterializedSemanticCandidate
    source_hash: int
    predicate_values: tuple[int, ...]
    predicate_start: int
    predicate_end: int
    bindings: tuple[SemanticCandidateBindingSpan, ...]
    carrier_key: tuple[int, ...] = (CARRIER_SEMANTIC_GRAPH,)

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, MaterializedSemanticCandidate):
            raise TypeError("semantic input view 必须携带 MaterializedSemanticCandidate")
        if type(self.source_hash) is not int or self.source_hash <= 0:
            raise ValueError("semantic input view.source_hash 必须为正整数")
        _strict_key(
            self.predicate_values,
            where="semantic input view.predicate_values",
        )
        _strict_key(self.carrier_key, where="semantic input view.carrier_key")
        if (type(self.predicate_start) is not int
                or type(self.predicate_end) is not int
                or self.predicate_start < 0
                or self.predicate_end <= self.predicate_start
                or self.predicate_end - self.predicate_start
                != len(self.predicate_values)):
            raise ValueError("semantic input predicate Span 不闭合")
        if (type(self.bindings) is not tuple or not self.bindings
                or any(not isinstance(item, SemanticCandidateBindingSpan)
                       for item in self.bindings)):
            raise TypeError("semantic input view.bindings 类型非法")
        expected = tuple(sorted(
            self.bindings, key=lambda item: item.stable_key()))
        if self.bindings != expected or len(set(self.bindings)) != len(self.bindings):
            raise ValueError("semantic input view.bindings 必须规范排序去重")

    def stable_key(self) -> tuple[int, ...]:
        definition = self.candidate.atomic.definition
        result = [
            self.source_hash,
            len(definition.proposition.stable_key()),
            *definition.proposition.stable_key(),
            len(self.predicate_values),
            *self.predicate_values,
            self.predicate_start,
            self.predicate_end,
            len(self.carrier_key),
            *self.carrier_key,
            len(self.bindings),
        ]
        for item in self.bindings:
            key = item.stable_key()
            result.extend((len(key), *key))
        return tuple(result)


class RelationSurfaceStructureAdapter:
    """从活动关系图恢复 carrier-neutral 的训练输入结构。"""

    @staticmethod
    def from_facts(
            facts: Iterable[RelationSurfaceView],
            ) -> tuple[TrainedInputStructure, ...]:
        structures: list[TrainedInputStructure] = []
        for fact in facts:
            proposition = _identity_key(fact.proposition, where="fact.proposition")
            predicate = _identity_key(fact.predicate, where="fact.predicate")
            if type(fact.source_hash) is not int or fact.source_hash <= 0:
                raise QueryStructureAdapterError("fact.source_hash 必须为正整数")
            if type(fact.cue) is not str or not fact.cue:
                raise QueryStructureAdapterError("fact.cue 必须为非空字符串")
            if type(fact.cue_start) is not int or type(fact.cue_end) is not int:
                raise QueryStructureAdapterError("fact cue 区间必须是严格整数")
            if fact.cue_end <= fact.cue_start:
                raise QueryStructureAdapterError("fact cue 区间必须为正向区间")
            members: list[TrainedInputMember] = [TrainedInputMember(
                tuple(ord(value) for value in fact.cue),
                predicate,
                predicate[0],
                PROJECTION_PREDICATE,
                proposition,
                predicate,
                (fact.source_hash,),
                fact.cue_start,
                fact.cue_end,
                0,
            )]
            for ordinal, binding in enumerate(fact.bindings, start=1):
                role = _identity_key(binding.role, where="binding.role")
                filler = _identity_key(binding.filler, where="binding.filler")
                if type(binding.surface) is not str or not binding.surface:
                    raise QueryStructureAdapterError(
                        "binding.surface 必须为非空字符串")
                if (type(binding.start) is not int
                        or type(binding.end) is not int
                        or binding.end <= binding.start):
                    raise QueryStructureAdapterError(
                        "binding span 必须为正向严格整数区间")
                members.append(TrainedInputMember(
                    tuple(ord(value) for value in binding.surface),
                    filler,
                    filler[0],
                    PROJECTION_FILLER,
                    proposition,
                    predicate,
                    (fact.source_hash,),
                    binding.start,
                    binding.end,
                    ordinal,
                    role,
                ))
            members.sort(key=lambda item: (
                item.trained_start,
                item.trained_end,
                item.projection_kind,
                item.member_ordinal,
                item.stable_key(),
            ))
            normalized = tuple(
                TrainedInputMember(
                    item.values,
                    item.candidate_key,
                    item.object_kind,
                    item.projection_kind,
                    item.proposition_key,
                    item.predicate_key,
                    item.source_ref,
                    item.trained_start,
                    item.trained_end,
                    ordinal,
                    item.role_key,
                )
                for ordinal, item in enumerate(members)
            )
            structures.append(TrainedInputStructure(
                proposition,
                predicate,
                (fact.source_hash,),
                (CARRIER_GRAPH_SOURCE,),
                normalized,
            ))
        return tuple(sorted(set(structures), key=lambda item: item.stable_key()))


class SemanticCandidateStructureAdapter:
    """从已恢复 S-02 候选和显式整数 Span 建立开放输入结构。"""

    @staticmethod
    def from_views(
            views: tuple[SemanticCandidateInputView, ...],
            ) -> tuple[TrainedInputStructure, ...]:
        if (type(views) is not tuple
                or any(not isinstance(item, SemanticCandidateInputView)
                       for item in views)):
            raise TypeError("semantic candidate input views 必须是 tuple")
        structures: list[TrainedInputStructure] = []
        for view in views:
            definition = view.candidate.atomic.definition
            proposition = _identity_key(
                definition.proposition, where="candidate.proposition")
            predicate = _identity_key(
                definition.predicate, where="candidate.predicate")
            structure = _identity_key(
                view.candidate.structure, where="candidate.structure")
            source_ref = _identity_key(
                definition.source, where="candidate.source")
            expected_bindings = {
                (
                    _identity_key(item.role, where="candidate.binding.role"),
                    _identity_key(item.filler, where="candidate.binding.filler"),
                    item.ordinal,
                )
                for item in definition.canonical_bindings()
            }
            actual_bindings = {
                (item.role_key, item.filler_key, item.binding_ordinal)
                for item in view.bindings
            }
            if actual_bindings != expected_bindings:
                raise QueryStructureAdapterError(
                    "semantic input binding Span 与 S-02 RoleBinding 不一致")
            members: list[TrainedInputMember] = [TrainedInputMember(
                view.predicate_values,
                predicate,
                predicate[0],
                PROJECTION_PREDICATE,
                proposition,
                predicate,
                source_ref,
                view.predicate_start,
                view.predicate_end,
                0,
            )]
            for ordinal, binding in enumerate(view.bindings, start=1):
                members.append(TrainedInputMember(
                    binding.values,
                    binding.filler_key,
                    binding.filler_key[0],
                    PROJECTION_FILLER,
                    proposition,
                    predicate,
                    source_ref,
                    binding.start,
                    binding.end,
                    ordinal,
                    binding.role_key,
                ))
            members.sort(key=lambda item: (
                item.trained_start,
                item.trained_end,
                item.projection_kind,
                item.member_ordinal,
                item.stable_key(),
            ))
            normalized = tuple(
                TrainedInputMember(
                    item.values,
                    item.candidate_key,
                    item.object_kind,
                    item.projection_kind,
                    item.proposition_key,
                    item.predicate_key,
                    item.source_ref,
                    item.trained_start,
                    item.trained_end,
                    ordinal,
                    item.role_key,
                )
                for ordinal, item in enumerate(members)
            )
            structures.append(TrainedInputStructure(
                proposition,
                predicate,
                source_ref,
                view.carrier_key,
                normalized,
                structure,
            ))
        return tuple(sorted(set(structures), key=lambda item: item.stable_key()))


__all__ = [
    "QueryStructureAdapterError",
    "RelationSurfaceStructureAdapter",
    "RelationBindingView",
    "RelationSurfaceView",
    "SemanticCandidateBindingSpan",
    "SemanticCandidateInputView",
    "SemanticCandidateStructureAdapter",
]
