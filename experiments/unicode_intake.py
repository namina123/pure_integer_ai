"""将只读 UCD 属性按需接入 Unicode sequence Representation 图对象。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.graph_ontology import GraphOntology
from pure_integer_ai.cognition.shared.identity import (
    CorpusVersion,
    CurriculumVersion,
    ParserVersion,
    PrimitiveVersion,
    TypedRef,
    VersionBundle,
)
from pure_integer_ai.cognition.shared.scope_identity import make_scope
from pure_integer_ai.cognition.shared.unicode_representation import (
    UnicodeSequenceMaterializer,
)
from pure_integer_ai.cognition.shared.types import Segment
from pure_integer_ai.crosscut.determinism.hasher import Hasher
from pure_integer_ai.experiments.data_manifest import RawDatasetManifest
from pure_integer_ai.experiments.ucd_adapter import (
    BINDING_EXTERNAL_PROPERTY_RELATION,
    BINDING_UCD_EPISTEMIC_ORIGIN,
    BINDING_UCD_PROVENANCE_KIND,
    BINDING_UCD_SCOPE_KIND,
    BINDING_UNICODE_SEQUENCE_FAMILY,
    UcdReadOnlyAdapter,
)


_SCOPE_HASHER = Hasher("pure_integer_ai.ucd.manifest.scope.v1")


@dataclass(frozen=True)
class UnicodeIntakeResult:
    """一次 token 表示摄入的确定性增长摘要。"""

    token_count: int
    unique_sequence_count: int
    sequence_refs: tuple[TypedRef, ...]
    property_link_count: int


class UnicodeIntake:
    """共享只读 UCD 索引，并把每个 token 作为独立表示概念摄入。"""

    def __init__(self, ontology: GraphOntology,
                 adapter: UcdReadOnlyAdapter) -> None:
        self._adapter = adapter
        manifest = adapter.manifest
        self._materializer = UnicodeSequenceMaterializer(
            ontology,
            family_key=manifest.binding(BINDING_UNICODE_SEQUENCE_FAMILY),
            external_property_relation_key=manifest.binding(
                BINDING_EXTERNAL_PROPERTY_RELATION),
        )
        self._provenance_kind = self._single_binding(
            manifest, BINDING_UCD_PROVENANCE_KIND, positive=True)
        self._epistemic_origin = self._single_binding(
            manifest, BINDING_UCD_EPISTEMIC_ORIGIN, positive=False)
        scope_kind = self._single_binding(
            manifest, BINDING_UCD_SCOPE_KIND, positive=True)
        manifest_hash = _SCOPE_HASHER.h63(manifest.sha256())
        if manifest_hash == 0:
            manifest_hash = 1
        self._scope = make_scope(
            scope_kind,
            manifest_hash,
            versions=VersionBundle(
                CorpusVersion(manifest_hash),
                ParserVersion(manifest.parser_version),
                PrimitiveVersion(0),
                CurriculumVersion(0),
            ),
        )
        self._sequence_cache: dict[tuple[int, ...], tuple[TypedRef, int]] = {}

    @staticmethod
    def _single_binding(manifest: RawDatasetManifest, name: str, *,
                        positive: bool) -> int:
        """读取单整数 manifest 绑定并校验范围。"""
        values = manifest.binding(name)
        if len(values) != 1:
            raise ValueError(f"manifest 绑定 {name!r} 必须只有一个整数")
        value = values[0]
        if positive and value <= 0:
            raise ValueError(f"manifest 绑定 {name!r} 必须为正整数")
        return value

    def clone_for_ontology(self, ontology: GraphOntology) -> "UnicodeIntake":
        """复用不可变只读索引，为评测沙箱绑定独立图写入口。"""
        return UnicodeIntake(ontology, self._adapter)

    def observe_segments(self,
                         segments: list[Segment]) -> UnicodeIntakeResult:
        """按首次出现顺序摄入 token 序列；不创建语言原子或结构角色。"""
        tokens = [token for segment in segments for token in segment.tokens]
        unique: dict[tuple[int, ...], None] = {}
        for token in tokens:
            codepoints = tuple(ord(character) for character in token)
            if codepoints:
                unique.setdefault(codepoints, None)

        refs: list[TypedRef] = []
        link_count = 0
        for codepoints in unique:
            cached = self._sequence_cache.get(codepoints)
            if cached is not None:
                refs.append(cached[0])
                link_count += cached[1]
                continue
            evidence_keys: list[tuple[int, ...]] = []
            for index, codepoint in enumerate(codepoints):
                for record in self._adapter.properties_for(codepoint):
                    evidence_keys.append(record.integer_key(
                        parser_version=self._adapter.manifest.parser_version,
                        sequence_index=index,
                    ))
            sequence_ref, links = self._materializer.materialize_with_properties(
                codepoints,
                tuple(sorted(evidence_keys)),
                scope=self._scope,
                provenance_kind=self._provenance_kind,
                epistemic_origin=self._epistemic_origin,
            )
            refs.append(sequence_ref)
            link_count += len(links)
            self._sequence_cache[codepoints] = (sequence_ref, len(links))
        return UnicodeIntakeResult(
            len(tokens), len(unique), tuple(refs), link_count)


__all__ = ["UnicodeIntake", "UnicodeIntakeResult"]
