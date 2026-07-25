"""语言对象的分型 ensure/lookup 入口。

本模块只组装共享身份构造器和 GraphOntology，不解释具体语言、词形或关系含义。
调用方必须注入稳定整数键、来源、scope 和 predicate；surface 不进入权威对象身份。
"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.graph_ontology import (
    GraphOntology,
    GraphStatement,
    relation_concept_identity,
)
from pure_integer_ai.cognition.shared.identity import (
    ObjectIdentity,
    SourceRef,
    TypedRef,
    concept_identity,
    language_atom_identity,
    language_branch_identity,
    occurrence_identity,
    sense_identity,
)
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.storage.node_store import TIER_SHADOW


class LanguageObjectIndex:
    """通过唯一图本体入口创建和查找语言相关分型对象。"""

    def __init__(self, ontology: GraphOntology) -> None:
        self._ontology = ontology

    @property
    def ontology(self) -> GraphOntology:
        """返回承载权威对象和 statement 的图本体。"""
        return self._ontology

    def ensure_branch(self, branch_key: tuple[int, ...], *,
                      tier: int = TIER_SHADOW) -> TypedRef:
        """按注入键幂等物化语言分支，不使用语言名称或 surface。"""
        return self._ontology.materialize(
            language_branch_identity(branch_key), tier=tier)

    def lookup_branch(self, branch_key: tuple[int, ...]) -> TypedRef | None:
        """只读查找语言分支，不存在时不登记身份。"""
        return self._ontology.resolve(language_branch_identity(branch_key))

    def ensure_atom(self, branch: ObjectIdentity | TypedRef,
                    atom_key: tuple[int, ...], *,
                    tier: int = TIER_SHADOW) -> TypedRef:
        """在指定分支内物化纯语言原子，atom_key 不得由 surface 代替。"""
        identity = language_atom_identity(
            self._branch_identity(branch), atom_key)
        return self._ontology.materialize(identity, tier=tier)

    def lookup_atom(self, branch: ObjectIdentity | TypedRef,
                    atom_key: tuple[int, ...]) -> TypedRef | None:
        """只读查找指定分支内的纯语言原子。"""
        identity = language_atom_identity(
            self._branch_identity(branch), atom_key)
        return self._ontology.resolve(identity)

    def ensure_concept(self, concept_key: tuple[int, ...], *,
                       tier: int = TIER_SHADOW) -> TypedRef:
        """按非 surface 注入键物化通用概念。"""
        return self._ontology.materialize(
            concept_identity(concept_key), tier=tier)

    def lookup_concept(self, concept_key: tuple[int, ...]) -> TypedRef | None:
        """只读查找按稳定键登记的通用概念。"""
        return self._ontology.resolve(concept_identity(concept_key))

    def ensure_sense(self, source: SourceRef, *,
                     sense_key: tuple[int, ...],
                     tier: int = TIER_SHADOW) -> TypedRef:
        """物化来源化词义；同表层词可以关联多个独立 sense。"""
        return self._ontology.materialize(
            sense_identity(source, sense_key=sense_key), tier=tier)

    def lookup_sense(self, source: SourceRef, *,
                     sense_key: tuple[int, ...]) -> TypedRef | None:
        """只读查找来源化词义，不以 token surface 查表。"""
        return self._ontology.resolve(
            sense_identity(source, sense_key=sense_key))

    def ensure_occurrence(self, source: SourceRef, *, start: int, end: int,
                          ordinal: int, tier: int = TIER_SHADOW) -> TypedRef:
        """物化一次来源内 occurrence；重复概念出现不会共享身份。"""
        return self._ontology.materialize(
            occurrence_identity(
                source, start=start, end=end, ordinal=ordinal),
            tier=tier,
        )

    def lookup_occurrence(self, source: SourceRef, *, start: int, end: int,
                          ordinal: int) -> TypedRef | None:
        """只读查找来源内 occurrence。"""
        return self._ontology.resolve(occurrence_identity(
            source, start=start, end=end, ordinal=ordinal))

    def relate(self, relation_key: tuple[int, ...], subject: TypedRef,
               object_ref: TypedRef, *, scope: ScopeIdentity,
               provenance_kind: int, epistemic_origin: int = 0,
               content_version: int = 0,
               qualifiers: tuple[int, ...] = ()) -> GraphStatement:
        """用调用方注入的动态 predicate 连接分型对象。"""
        predicate = self._ontology.materialize(
            relation_concept_identity(relation_key))
        return self._ontology.relate(
            predicate,
            subject,
            object_ref,
            scope=scope,
            provenance_kind=provenance_kind,
            epistemic_origin=epistemic_origin,
            content_version=content_version,
            qualifiers=qualifiers,
        )

    def _branch_identity(
            self, branch: ObjectIdentity | TypedRef) -> ObjectIdentity:
        """把分支身份或分型引用统一还原为经图核验的身份。"""
        if isinstance(branch, TypedRef):
            return self._ontology.identity_of(branch)
        if isinstance(branch, ObjectIdentity):
            return branch
        raise TypeError("branch 必须是 ObjectIdentity 或 TypedRef")


__all__ = ["LanguageObjectIndex"]
