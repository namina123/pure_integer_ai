"""一等 occurrence 的领域写入、图关系和可回源读取入口。

Occurrence 身份只由 SourceRef、码点 span 和同位 ordinal 决定。segment/local/document
位置、speaker 和解释候选是可核验详情或图关系，不能反向替代 occurrence 本体身份。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.graph_ontology import (
    GraphObjectIntegrityError,
    GraphOntology,
    relation_concept_identity,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_OCCURRENCE,
    ObjectIdentity,
    SourceRef,
    TypedRef,
    occurrence_identity,
)
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.cognition.shared.scoped_persistence import ScopedIdentityStore
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.storage.node_store import TIER_SHADOW
from pure_integer_ai.storage.occurrence import (
    OccurrenceCandidateStorage,
    OccurrenceStorageRecord,
    OccurrenceStore,
)
from pure_integer_ai.storage.source_record import SourceRecordRepository


def _protocol_key(value, *, where: str) -> tuple[int, ...]:
    """校验由课程或程序注入的开放图关系键。"""
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{where} 必须是非空整数 tuple")
    assert_int(*value, _where=where)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{where} 必须使用严格整数")
    return value


@dataclass(frozen=True)
class OccurrenceProtocol:
    """Occurrence 到候选和可选 speaker 的动态图 predicate 协议。"""

    candidate_relation_key: tuple[int, ...]
    speaker_relation_key: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        _protocol_key(
            self.candidate_relation_key,
            where="OccurrenceProtocol.candidate_relation_key",
        )
        if self.speaker_relation_key is not None:
            _protocol_key(
                self.speaker_relation_key,
                where="OccurrenceProtocol.speaker_relation_key",
            )


@dataclass(frozen=True)
class OccurrenceCandidate:
    """一个 typed 图候选或显式 legacy 节点候选。"""

    ordinal: int
    typed_ref: TypedRef | None = None
    legacy_ref: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        assert_int(self.ordinal, _where="OccurrenceCandidate.ordinal")
        if self.ordinal < 0:
            raise ValueError("OccurrenceCandidate.ordinal 不得为负")
        if (self.typed_ref is None) == (self.legacy_ref is None):
            raise ValueError("OccurrenceCandidate 必须且只能携带一种端点")
        if self.typed_ref is not None and not isinstance(self.typed_ref, TypedRef):
            raise TypeError("OccurrenceCandidate.typed_ref 必须是 TypedRef")
        if self.legacy_ref is not None:
            if (not isinstance(self.legacy_ref, tuple)
                    or len(self.legacy_ref) != 2):
                raise ValueError("legacy_ref 必须是二元节点引用")
            assert_int(*self.legacy_ref, _where="OccurrenceCandidate.legacy_ref")
            if min(self.legacy_ref) <= 0:
                raise ValueError("legacy_ref 编址必须为正")


@dataclass(frozen=True)
class OccurrenceRecord:
    """可从 occurrence 回到来源原文、解析位置和候选端点的完整读模型。"""

    occurrence: TypedRef
    source: SourceRef
    scope: ScopeIdentity
    raw_text: str
    surface: str
    start: int
    end: int
    ordinal: int
    segment_index: int
    local_index: int
    document_index: int
    parser_version: int
    speaker: TypedRef | None
    candidates: tuple[OccurrenceCandidate, ...]


class OccurrenceIndex:
    """组装图本体、SourceRecord 和 occurrence 详情表的唯一领域入口。"""

    def __init__(
            self, ontology: GraphOntology,
            scoped_identities: ScopedIdentityStore,
            protocol: OccurrenceProtocol,
            ) -> None:
        if not isinstance(ontology, GraphOntology):
            raise TypeError("ontology 必须是 GraphOntology")
        if not isinstance(scoped_identities, ScopedIdentityStore):
            raise TypeError("scoped_identities 必须是 ScopedIdentityStore")
        if not isinstance(protocol, OccurrenceProtocol):
            raise TypeError("protocol 必须是 OccurrenceProtocol")
        self.ontology = ontology
        self.scoped_identities = scoped_identities
        self.protocol = protocol
        self._sources = SourceRecordRepository(
            ontology.backend, registry=scoped_identities.registry)
        self._occurrences = OccurrenceStore(ontology.backend)
        self._source_cache: dict[SourceRef, tuple[int, str]] = {}
        self._occurrence_cache: dict[ObjectIdentity, TypedRef] = {}
        self._candidate_links: set[tuple] = set()
        self._speaker_links: set[tuple] = set()
        self._candidate_predicate: TypedRef | None = None
        self._speaker_predicate: TypedRef | None = None

    def ensure_source(self, source: SourceRef, raw_text: str) -> int:
        """幂等留档来源原文，并缓存同一 SourceRef 的精确文本绑定。"""
        if not isinstance(source, SourceRef):
            raise TypeError("source 必须是 SourceRef")
        if not isinstance(raw_text, str):
            raise TypeError("raw_text 必须是字符串")
        cached = self._source_cache.get(source)
        if cached is not None:
            if cached[1] != raw_text:
                raise ValueError("同一 SourceRef 在当前上下文绑定了不同原文")
            return cached[0]
        record = self._sources.put(source.stable_key(), raw_text)
        self._source_cache[source] = (record.source_hash, raw_text)
        return record.source_hash

    def record(
            self, *, source: SourceRef, raw_text: str,
            scope: ScopeIdentity, start: int, end: int, ordinal: int,
            segment_index: int, local_index: int, document_index: int,
            speaker: ObjectIdentity | TypedRef | None = None,
            typed_candidates: tuple[TypedRef, ...] = (),
            legacy_candidates: tuple[tuple[int, int], ...] = (),
            ) -> OccurrenceRecord:
        """物化 occurrence，保存精确位置，并追加 typed/legacy 候选桥。"""
        if not isinstance(scope, ScopeIdentity):
            raise TypeError("scope 必须是 ScopeIdentity")
        if scope.source != source:
            raise ValueError("occurrence scope 必须指向同一 SourceRef")
        assert_int(
            start,
            end,
            ordinal,
            segment_index,
            local_index,
            document_index,
            _where="OccurrenceIndex.record",
        )
        if (start < 0 or end < start or end > len(raw_text) or ordinal < 0
                or segment_index < 0 or local_index < 0
                or document_index < 0):
            raise ValueError("occurrence span 或局部位置非法")
        source_hash = self.ensure_source(source, raw_text)
        identity = occurrence_identity(
            source, start=start, end=end, ordinal=ordinal)
        occurrence = self._occurrence_cache.get(identity)
        if occurrence is None:
            occurrence = self.ontology.materialize(identity, tier=TIER_SHADOW)
            self._occurrence_cache[identity] = occurrence
        speaker_ref = self._speaker_ref(speaker)
        scope_hash = self.scoped_identities.register_scope(scope)
        self._occurrences.add(OccurrenceStorageRecord(
            occurrence.space_id,
            occurrence.local_id,
            source_hash,
            scope_hash,
            start,
            end,
            ordinal,
            segment_index,
            local_index,
            document_index,
            source.versions.parser.value,
            0 if speaker_ref is None else speaker_ref.object_kind,
            0 if speaker_ref is None else speaker_ref.space_id,
            0 if speaker_ref is None else speaker_ref.local_id,
        ))
        self._record_speaker_relation(
            occurrence, speaker_ref, scope=scope, source=source)
        ordinal_cursor = 0
        for candidate in self._unique_typed_candidates(typed_candidates):
            self._record_candidate(
                occurrence,
                candidate,
                ordinal_cursor,
                scope=scope,
                source=source,
            )
            ordinal_cursor += 1
        for candidate in self._unique_legacy_candidates(legacy_candidates):
            self._occurrences.add_candidate(OccurrenceCandidateStorage(
                occurrence.space_id,
                occurrence.local_id,
                ordinal_cursor,
                0,
                candidate[0],
                candidate[1],
            ))
            ordinal_cursor += 1
        return self.read(occurrence)

    def read(self, occurrence: TypedRef) -> OccurrenceRecord:
        """完整核验 occurrence 本体、来源、scope、原文 span 和候选端点。"""
        identity = self.ontology.identity_of(occurrence)
        if identity.object_kind != OBJECT_OCCURRENCE:
            raise ValueError("read 需要一等 occurrence 引用")
        stored = self._occurrences.read(
            occurrence.space_id, occurrence.local_id)
        source_record = self._sources.read(stored.source_hash)
        source = SourceRef.from_stable_key(source_record.source_key)
        expected = occurrence_identity(
            source,
            start=stored.start,
            end=stored.end,
            ordinal=stored.ordinal,
        )
        if expected != identity:
            raise ValueError("occurrence 图身份与详情表不一致")
        if stored.end > len(source_record.raw_text):
            raise ValueError("occurrence span 超出来源原文")
        scope = self.scoped_identities.load_scope(stored.scope_hash)
        if scope.source != source:
            raise ValueError("occurrence scope 与来源记录不一致")
        if stored.parser_version != source.versions.parser.value:
            raise ValueError("occurrence parser version 与 SourceRef 不一致")
        speaker = None
        if stored.speaker_object_kind:
            speaker = self.ontology.typed_ref_for_node(
                stored.speaker_space_id, stored.speaker_local_id)
            if speaker.object_kind != stored.speaker_object_kind:
                raise ValueError("occurrence speaker 类型与图对象不一致")
        candidates: list[OccurrenceCandidate] = []
        for candidate in self._occurrences.candidates(
                occurrence.space_id, occurrence.local_id):
            if candidate.candidate_object_kind == 0:
                candidates.append(OccurrenceCandidate(
                    candidate.candidate_ordinal,
                    legacy_ref=(
                        candidate.candidate_space_id,
                        candidate.candidate_local_id,
                    ),
                ))
                continue
            typed = self.ontology.typed_ref_for_node(
                candidate.candidate_space_id,
                candidate.candidate_local_id,
            )
            if typed.object_kind != candidate.candidate_object_kind:
                raise ValueError("occurrence candidate 类型与图对象不一致")
            candidates.append(OccurrenceCandidate(
                candidate.candidate_ordinal,
                typed_ref=typed,
            ))
        return OccurrenceRecord(
            occurrence,
            source,
            scope,
            source_record.raw_text,
            source_record.raw_text[stored.start:stored.end],
            stored.start,
            stored.end,
            stored.ordinal,
            stored.segment_index,
            stored.local_index,
            stored.document_index,
            stored.parser_version,
            speaker,
            tuple(candidates),
        )

    def typed_candidate_for_node(
            self, ref: tuple[int, int]) -> TypedRef | None:
        """把已物化权威图节点恢复为 TypedRef；legacy 节点返回 None。"""
        try:
            return self.ontology.typed_ref_for_node(ref[0], ref[1])
        except GraphObjectIntegrityError:
            return None

    def occurrence_count(self) -> int:
        """返回当前后端已保存的唯一 occurrence 数。"""
        return self._occurrences.occurrence_count()

    def source_count(self) -> int:
        """返回当前后端已保存的唯一 SourceRecord 数。"""
        return self._sources.source_count()

    @property
    def source_repository(self) -> SourceRecordRepository:
        """暴露同一 identity registry 下的来源仓库，供断奶后 intake 编排复用。"""
        return self._sources

    def clone_for_context(
            self, ontology: GraphOntology,
            scoped_identities: ScopedIdentityStore,
            ) -> "OccurrenceIndex":
        """在评测 clone 的 backend/ontology 上重建独立索引和缓存。"""
        return OccurrenceIndex(ontology, scoped_identities, self.protocol)

    def _speaker_ref(
            self, speaker: ObjectIdentity | TypedRef | None
            ) -> TypedRef | None:
        """把可选 speaker 身份物化或核验为图内分型端点。"""
        if speaker is None:
            return None
        if isinstance(speaker, ObjectIdentity):
            return self.ontology.materialize(speaker, tier=TIER_SHADOW)
        if isinstance(speaker, TypedRef):
            self.ontology.identity_of(speaker)
            return speaker
        raise TypeError("speaker 必须是 ObjectIdentity、TypedRef 或 None")

    def _record_speaker_relation(
            self, occurrence: TypedRef, speaker: TypedRef | None, *,
            scope: ScopeIdentity, source: SourceRef) -> None:
        """按注入 predicate 把 occurrence 与 speaker 关联；未配置时只存详情端点。"""
        if speaker is None or self.protocol.speaker_relation_key is None:
            return
        link_key = (occurrence, speaker)
        if link_key in self._speaker_links:
            return
        if self._speaker_predicate is None:
            self._speaker_predicate = self.ontology.materialize(
                relation_concept_identity(self.protocol.speaker_relation_key))
        self.ontology.relate(
            self._speaker_predicate,
            occurrence,
            speaker,
            scope=scope,
            provenance_kind=source.source_kind,
            content_version=source.versions.parser.value,
        )
        self._speaker_links.add(link_key)

    def _record_candidate(
            self, occurrence: TypedRef, candidate: TypedRef,
            candidate_ordinal: int, *, scope: ScopeIdentity,
            source: SourceRef) -> None:
        """保存 typed 候选索引，并用注入 predicate 写来源化图关系。"""
        self.ontology.identity_of(candidate)
        storage = OccurrenceCandidateStorage(
            occurrence.space_id,
            occurrence.local_id,
            candidate_ordinal,
            candidate.object_kind,
            candidate.space_id,
            candidate.local_id,
        )
        self._occurrences.add_candidate(storage)
        link_key = (occurrence, candidate, candidate_ordinal)
        if link_key in self._candidate_links:
            return
        if self._candidate_predicate is None:
            self._candidate_predicate = self.ontology.materialize(
                relation_concept_identity(
                    self.protocol.candidate_relation_key))
        self.ontology.relate(
            self._candidate_predicate,
            occurrence,
            candidate,
            scope=scope,
            provenance_kind=source.source_kind,
            content_version=source.versions.parser.value,
            qualifiers=(candidate_ordinal,),
        )
        self._candidate_links.add(link_key)

    @staticmethod
    def _unique_typed_candidates(
            candidates: tuple[TypedRef, ...]) -> tuple[TypedRef, ...]:
        """按调用方顺序去重 typed 候选，不引入分数或对象类型偏好。"""
        out: list[TypedRef] = []
        seen: set[TypedRef] = set()
        for candidate in candidates:
            if not isinstance(candidate, TypedRef):
                raise TypeError("typed_candidates 只能包含 TypedRef")
            if candidate not in seen:
                seen.add(candidate)
                out.append(candidate)
        return tuple(out)

    @staticmethod
    def _unique_legacy_candidates(
            candidates: tuple[tuple[int, int], ...]
            ) -> tuple[tuple[int, int], ...]:
        """按调用方顺序去重 legacy 节点，并保持其非权威标记。"""
        out: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for candidate in candidates:
            if not isinstance(candidate, tuple) or len(candidate) != 2:
                raise ValueError("legacy_candidates 必须是二元节点引用")
            assert_int(*candidate, _where="OccurrenceIndex.legacy_candidates")
            if min(candidate) <= 0:
                raise ValueError("legacy candidate 编址必须为正")
            if candidate not in seen:
                seen.add(candidate)
                out.append(candidate)
        return tuple(out)


__all__ = [
    "OccurrenceCandidate",
    "OccurrenceIndex",
    "OccurrenceProtocol",
    "OccurrenceRecord",
]
