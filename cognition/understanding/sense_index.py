"""来源化 Sense 的图内索引和旧 surface 候选迁移入口。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.graph_ontology import (
    relation_concept_identity,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONCEPT,
    OBJECT_LANGUAGE_ATOM,
    OBJECT_SENSE,
    SourceRef,
    TypedRef,
)
from pure_integer_ai.cognition.shared.language_object_index import (
    LanguageObjectIndex,
)
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.storage.node_store import TIER_SHADOW
from pure_integer_ai.storage.sense_candidates import (
    SenseLegacyBridgeConflict,
    read_sense_candidates,
    record_legacy_sense_bridge,
    sense_surface_hash,
)


@dataclass(frozen=True)
class SenseBinding:
    """语言原子、来源化 Sense 与非 surface 概念的分型绑定。"""

    atom: TypedRef
    sense: TypedRef
    concept: TypedRef


@dataclass(frozen=True)
class LegacySenseSpec:
    """调用方为一个旧候选显式提供的稳定迁移键。"""

    legacy_ref: tuple[int, int]
    sense_key: tuple[int, ...]
    concept_key: tuple[int, ...]


class SenseIndex:
    """用注入 predicate 管理 atom->Sense->Concept，不读取 surface 猜语义。"""

    def __init__(self, backend, language_objects: LanguageObjectIndex, *,
                 atom_sense_relation_key: tuple[int, ...],
                 sense_concept_relation_key: tuple[int, ...]) -> None:
        self._backend = backend
        self._objects = language_objects
        self._atom_sense_relation_key = atom_sense_relation_key
        self._sense_concept_relation_key = sense_concept_relation_key

    def ensure(self, atom: TypedRef, source: SourceRef, *,
               sense_key: tuple[int, ...], concept_key: tuple[int, ...],
               scope: ScopeIdentity, provenance_kind: int,
               epistemic_origin: int = 0,
               content_version: int = 0,
               tier: int = TIER_SHADOW) -> SenseBinding:
        """幂等创建来源化 Sense、概念及两段图关系。"""
        atom_identity = self._objects.ontology.identity_of(atom)
        if atom_identity.object_kind != OBJECT_LANGUAGE_ATOM:
            raise ValueError("Sense 只能由已物化 LanguageAtom 承载")
        sense = self._objects.ensure_sense(
            source, sense_key=sense_key, tier=tier)
        concept = self._objects.ensure_concept(concept_key, tier=tier)
        self._objects.relate(
            self._atom_sense_relation_key,
            atom,
            sense,
            scope=scope,
            provenance_kind=provenance_kind,
            epistemic_origin=epistemic_origin,
            content_version=content_version,
        )
        self._objects.relate(
            self._sense_concept_relation_key,
            sense,
            concept,
            scope=scope,
            provenance_kind=provenance_kind,
            epistemic_origin=epistemic_origin,
            content_version=content_version,
        )
        return SenseBinding(atom, sense, concept)

    def lookup(self, atom: TypedRef) -> tuple[SenseBinding, ...]:
        """从权威图恢复 atom 的全部 Sense 和概念候选，不按首项塌缩。"""
        atom_identity = self._objects.ontology.identity_of(atom)
        if atom_identity.object_kind != OBJECT_LANGUAGE_ATOM:
            raise ValueError("Sense 只能由已物化 LanguageAtom 承载")
        atom_sense = self._objects.ontology.resolve(
            relation_concept_identity(self._atom_sense_relation_key))
        sense_concept = self._objects.ontology.resolve(
            relation_concept_identity(self._sense_concept_relation_key))
        if atom_sense is None or sense_concept is None:
            return ()
        bindings: list[SenseBinding] = []
        for sense in self._objects.ontology.follow(atom, (atom_sense,)):
            if sense.object_kind != OBJECT_SENSE:
                raise SenseLegacyBridgeConflict(
                    "atom 的 sense predicate 指向了非 Sense 对象")
            for concept in self._objects.ontology.follow(
                    sense, (sense_concept,)):
                if concept.object_kind != OBJECT_CONCEPT:
                    raise SenseLegacyBridgeConflict(
                        "Sense 的概念 predicate 指向了非 Concept 对象")
                bindings.append(SenseBinding(atom, sense, concept))
        return tuple(sorted(
            bindings,
            key=lambda item: (
                item.sense.stable_key(), item.concept.stable_key()),
        ))

    def migrate_legacy(self, surface: str, atom: TypedRef,
                       source: SourceRef, *,
                       specs: tuple[LegacySenseSpec, ...],
                       scope: ScopeIdentity, provenance_kind: int,
                       epistemic_origin: int = 0,
                       content_version: int = 0,
                       tier: int = TIER_SHADOW) -> tuple[SenseBinding, ...]:
        """全集迁移旧 surface 候选；缺项、多项或错 ref 均 fail closed。"""
        if not isinstance(surface, str) or not surface:
            raise ValueError("旧 sense surface 必须是非空字符串")
        candidates = read_sense_candidates(
            self._backend, atom.space_id, sense_surface_hash(surface))
        legacy_refs = {item[0] for item in candidates}
        if len(legacy_refs) != len(candidates):
            raise SenseLegacyBridgeConflict("旧 sense 候选表含重复 ref")
        spec_by_ref = {spec.legacy_ref: spec for spec in specs}
        if len(spec_by_ref) != len(specs):
            raise SenseLegacyBridgeConflict("旧 sense 迁移规格含重复 ref")
        if legacy_refs != set(spec_by_ref):
            raise SenseLegacyBridgeConflict(
                "迁移规格必须精确覆盖旧 surface 的全部 sense 候选")
        bindings: list[SenseBinding] = []
        for legacy_ref in sorted(legacy_refs):
            spec = spec_by_ref[legacy_ref]
            binding = self.ensure(
                atom,
                source,
                sense_key=spec.sense_key,
                concept_key=spec.concept_key,
                scope=scope,
                provenance_kind=provenance_kind,
                epistemic_origin=epistemic_origin,
                content_version=content_version,
                tier=tier,
            )
            for object_ref in (binding.sense, binding.concept):
                record_legacy_sense_bridge(
                    self._backend,
                    legacy_ref=legacy_ref,
                    object_ref=(
                        object_ref.object_kind,
                        object_ref.space_id,
                        object_ref.local_id,
                    ),
                )
            bindings.append(binding)
        return tuple(bindings)


__all__ = [
    "LegacySenseSpec",
    "SenseBinding",
    "SenseIndex",
]
