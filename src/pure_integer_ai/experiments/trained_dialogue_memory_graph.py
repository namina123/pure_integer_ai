"""把运行时对话表层写入 interaction Memory 图并执行有界跨进程召回。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pure_integer_ai.cognition.shared.identity import (
    CorpusVersion,
    CurriculumVersion,
    OBJECT_CONTEXT_SCOPE,
    ObjectIdentity,
    OwnerScope,
    ParserVersion,
    PrimitiveVersion,
    SourceRef,
    VersionBundle,
    VISIBILITY_SESSION,
)
from pure_integer_ai.cognition.shared.memory_event import MemoryLinkedRef
from pure_integer_ai.cognition.shared.memory_event_log import MemoryEventLog
from pure_integer_ai.cognition.shared.memory_overlay import CoreIdentityCatalog
from pure_integer_ai.cognition.understanding.memory_intake import (
    MemorySourceIntake,
    ObservationIntakeDraft,
    interaction_intake_policy,
)
from pure_integer_ai.cognition.understanding.source_intake import (
    SourceIntake,
    SourceSlice,
)
from pure_integer_ai.crosscut.determinism.hasher import Hasher
from pure_integer_ai.storage import discipline as disc
from pure_integer_ai.storage.assertion_identity import (
    register_assertion_identity_tables,
)
from pure_integer_ai.storage.assertion_record import (
    register_assertion_record_tables,
)
from pure_integer_ai.storage.backend import (
    SQLiteBackend,
    TYPE_INT,
    register_extension_table,
)
from pure_integer_ai.storage.graph_object import register_graph_object_table
from pure_integer_ai.storage.graph_statement import (
    register_graph_statement_table,
)
from pure_integer_ai.storage.memory_event import register_memory_event_table
from pure_integer_ai.cognition.shared.graph_ontology import GraphOntology
from pure_integer_ai.cognition.shared.scoped_persistence import (
    ScopedIdentityStore,
)
from pure_integer_ai.storage.source_record import (
    SourceRecordRepository,
    register_source_record_table,
)
from pure_integer_ai.storage.spaces.companion import (
    CompanionSpace,
    register_companion_table,
)
from pure_integer_ai.storage.spaces.memory_space import MemorySpace
from pure_integer_ai.storage.spaces.registry import (
    SPACE_TYPE_CORE,
    SpaceRegistry,
    register_space_table,
)


DIALOGUE_MEMORY_POSTING_TABLE = "dialogue_memory_posting"
_DIALOGUE_MEMORY_NAMESPACE = 91501
_DIALOGUE_MEMORY_SOURCE_KIND = 91501
_FEATURE_HASHER = Hasher("trained_dialogue_memory.feature.v1")
_VERSIONS = VersionBundle(
    CorpusVersion(1), ParserVersion(1),
    PrimitiveVersion(1), CurriculumVersion(1))
_MAX_POSTINGS_PER_FEATURE = 64
_MAX_CANDIDATES = 32


def register_dialogue_memory_posting_table(backend) -> None:
    """注册 Memory 图的可丢弃整数检索投影。"""
    register_extension_table(
        backend,
        DIALOGUE_MEMORY_POSTING_TABLE,
        [
            ("source_hash", TYPE_INT),
            ("feature_hash", TYPE_INT),
            ("feature_width", TYPE_INT),
            ("feature_ordinal", TYPE_INT),
            ("turn_seq", TYPE_INT),
            ("speaker_kind", TYPE_INT),
            ("tenant_id", TYPE_INT),
            ("user_id", TYPE_INT),
            ("session_id", TYPE_INT),
        ],
        discipline=disc.DISC_APPEND_ONLY,
        indexes=[
            ("source_hash",),
            ("tenant_id", "user_id", "session_id",
             "feature_hash", "feature_width"),
            ("turn_seq",),
        ],
        recovery_key=("source_hash", "feature_ordinal"),
    )


def _register_interaction_memory_tables(backend) -> None:
    """只注册 session interaction Memory 承重路径实际使用的表。"""
    register_space_table(backend)
    register_companion_table(backend)
    register_assertion_identity_tables(backend)
    register_assertion_record_tables(backend)
    register_graph_object_table(backend)
    register_graph_statement_table(backend)
    register_memory_event_table(backend)
    register_source_record_table(backend)


def _build_interaction_memory_intake(backend) -> MemorySourceIntake:
    """最小装配 Companion、SourceRecord 与 interaction Memory 权威 owner。"""
    _register_interaction_memory_tables(backend)
    registry = SpaceRegistry(backend)
    core_space_id = registry.register(SPACE_TYPE_CORE, "core")
    companion = CompanionSpace.create(registry, "companion")
    interaction = MemorySpace.create(registry, "memory_interact")
    identities = ScopedIdentityStore(backend)
    ontology = GraphOntology(
        backend,
        space_id=core_space_id,
        space_identity=SpaceRegistry.identity_for(SPACE_TYPE_CORE, "core"),
        scoped_identities=identities,
    )
    catalog = CoreIdentityCatalog((ontology,))
    events = MemoryEventLog(
        registry,
        backend,
        interaction.space_id,
        identities,
        catalog,
    )
    repository = SourceRecordRepository(
        backend, registry=identities.registry)
    return MemorySourceIntake(
        SourceIntake(repository, companion),
        events,
        interaction_intake_policy(),
    )


@dataclass(frozen=True)
class DialogueMemoryAppend:
    """一次表层进入 interaction Memory 图后的稳定结果。"""

    source: SourceRef
    source_hash: int
    turn_seq: int
    speaker_kind: int
    posting_count: int
    observation_key: tuple[int, ...]


@dataclass(frozen=True)
class DialogueMemoryRecall:
    """一次经活动 Memory manifest 核验后的表层召回。"""

    surface: str
    source: SourceRef
    source_hash: int
    turn_seq: int
    speaker_kind: int
    similarity_permille: int
    candidate_count: int
    posting_reads: int


@dataclass(frozen=True)
class DialogueMemoryTurn:
    """从当前 owner 的活动 Memory manifest 恢复的一轮对话。"""

    surface: str
    source: SourceRef
    source_hash: int
    turn_seq: int
    speaker_kind: int


@dataclass(frozen=True)
class _DialogueObservationParser:
    """把完整 SourceSlice 映射为不伪造命题的 Memory Observation。"""

    owner: OwnerScope
    session_id: int
    speaker_kind: int

    def parse(self, source: SourceSlice) -> ObservationIntakeDraft:
        """保留来源和会话上下文；未知语义不形成 Hypothesis。"""
        if not isinstance(source, SourceSlice) or not source.text:
            raise ValueError("dialogue Memory parser 需要非空完整来源切片")
        context = ObjectIdentity(
            OBJECT_CONTEXT_SCOPE,
            (_DIALOGUE_MEMORY_NAMESPACE, self.session_id,
             self.speaker_kind),
            self.owner,
            source.source.versions,
        )
        return ObservationIntakeDraft(
            (_DIALOGUE_MEMORY_NAMESPACE, source.source.document_id),
            MemoryLinkedRef.object(context),
        )


# object-model: resource_owner; representation=runtime; interop=dialogue-memory-graph-v1
class TrainedDialogueMemoryGraph:
    """拥有可写 session SQLite，并把每次召回约束到活动 Memory 图来源。"""

    __slots__ = (
        "path", "backend", "intake", "owner", "session_id",
        "source_id",
        "_next_turn_seq",
    )

    def __init__(
            self,
            database: str | Path,
            *,
            tenant_id: int = 1,
            user_id: int = 1,
            session_id: int = 1,
            ) -> None:
        """打开或创建 session Memory 图；身份参数必须为严格正整数。"""
        for label, value in (
                ("tenant_id", tenant_id),
                ("user_id", user_id),
                ("session_id", session_id)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{label} 必须是严格正整数")
        self.path = Path(database).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.backend = SQLiteBackend(str(self.path))
        try:
            self.intake = _build_interaction_memory_intake(self.backend)
            register_dialogue_memory_posting_table(self.backend)
            self.owner = OwnerScope(
                tenant_id, user_id, session_id, VISIBILITY_SESSION)
            self.session_id = session_id
            self.source_id = (
                Hasher("trained_dialogue_memory.session_source.v1").h63(
                    self.owner.stable_key()) or 1)
            self._next_turn_seq = self._restore_next_turn_seq()
        except BaseException:
            self.backend.close()
            raise

    def close(self) -> None:
        """提交当前 append-only Memory 图并关闭 owner。"""
        backend = getattr(self, "backend", None)
        if backend is not None:
            backend.commit()
            backend.close()
            self.backend = None

    def __enter__(self) -> "TrainedDialogueMemoryGraph":
        """返回当前 session Memory owner。"""
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        """结束 session Memory owner 生命周期。"""
        self.close()

    def append(self, surface: str, *, speaker_kind: int) -> DialogueMemoryAppend:
        """把一轮表层写入 SourceRecord、Companion 和 interaction Memory 图。"""
        if type(surface) is not str or not surface.strip():
            raise ValueError("Memory 表层必须是非空文本")
        if type(speaker_kind) is not int or speaker_kind <= 0:
            raise ValueError("speaker_kind 必须是严格正整数")
        turn_seq = self._next_turn_seq
        source = SourceRef(
            _DIALOGUE_MEMORY_SOURCE_KIND,
            self.source_id,
            turn_seq,
            self.owner,
            _VERSIONS,
        )
        result = self.intake.ingest(
            source,
            surface,
            license_id="runtime-dialogue-v1",
            batch_id=turn_seq,
            parser=_DialogueObservationParser(
                self.owner, self.session_id, speaker_kind),
        )
        posting_count = self._ensure_postings(
            result.source_record.source_hash,
            surface,
            turn_seq=turn_seq,
            speaker_kind=speaker_kind,
        )
        self.backend.commit()
        self._next_turn_seq += 1
        return DialogueMemoryAppend(
            source,
            result.source_record.source_hash,
            turn_seq,
            speaker_kind,
            posting_count,
            result.observation_ref.object_key,
        )

    def recall(
            self,
            surface: str,
            *,
            minimum_similarity_permille: int = 500,
            speaker_kind: int | None = None,
            ) -> DialogueMemoryRecall | None:
        """按有界 posting 查询相关表层，并要求来源仍有活动 Memory manifest。"""
        if type(surface) is not str or not surface.strip():
            return None
        if (type(minimum_similarity_permille) is not int
                or not 0 <= minimum_similarity_permille <= 1000):
            raise ValueError("Memory similarity 必须是 0..1000 整数")
        if (speaker_kind is not None
                and (type(speaker_kind) is not int or speaker_kind <= 0)):
            raise ValueError("Memory speaker_kind 必须是正整数或 None")
        query_features = self._features(surface)
        hits: dict[int, int] = {}
        posting_reads = 0
        # 长特征优先；每个特征的 page-in 都有固定上限，不随总 Memory 线性扫描。
        for feature, width in sorted(
                query_features, key=lambda item: (-item[1], item[0])):
            where = {
                "tenant_id": self.owner.tenant_id,
                "user_id": self.owner.user_id,
                "session_id": self.owner.session_id,
                "feature_hash": feature,
                "feature_width": width,
            }
            if speaker_kind is not None:
                where["speaker_kind"] = speaker_kind
            rows = self.backend.select(
                DIALOGUE_MEMORY_POSTING_TABLE,
                where=where,
                order_by="turn_seq",
                descending=True,
                limit=_MAX_POSTINGS_PER_FEATURE,
            )
            posting_reads += len(rows)
            for row in rows:
                source_hash = row["source_hash"]
                hits[source_hash] = hits.get(source_hash, 0) + width
        shortlist = tuple(sorted(
            hits,
            key=lambda source_hash: (-hits[source_hash], -source_hash),
        )[:_MAX_CANDIDATES])
        repository = self.intake.source_intake.repository
        query_exact = self._exact_features(surface)
        ranked = []
        for source_hash in shortlist:
            record = repository.read(source_hash)
            source = SourceRef.from_stable_key(record.source_key)
            if source.owner != self.owner or record.raw_text.strip() == surface.strip():
                continue
            # manifest 是 Memory 图对来源当前可见性的权威证明；posting 不能替代它。
            self.intake.require_current_manifest(source)
            candidate_exact = self._exact_features(record.raw_text)
            overlap = len(query_exact.intersection(candidate_exact))
            score = (2000 * overlap) // max(
                1, len(query_exact) + len(candidate_exact))
            if score < minimum_similarity_permille:
                continue
            posting_where = {"source_hash": source_hash}
            if speaker_kind is not None:
                posting_where["speaker_kind"] = speaker_kind
            posting_rows = self.backend.select(
                DIALOGUE_MEMORY_POSTING_TABLE, where=posting_where)
            if not posting_rows:
                raise RuntimeError("Memory 来源缺少整数 posting")
            turn_seq = posting_rows[0]["turn_seq"]
            candidate_speaker_kind = posting_rows[0]["speaker_kind"]
            if any(
                row["turn_seq"] != turn_seq
                or row["speaker_kind"] != candidate_speaker_kind
                for row in posting_rows):
                raise RuntimeError("Memory posting 来源元数据漂移")
            ranked.append((
                score,
                overlap,
                turn_seq,
                record,
                source,
                candidate_speaker_kind,
            ))
        ranked.sort(key=lambda item: (-item[0], -item[1], -item[2]))
        if not ranked:
            return None
        if (len(ranked) > 1
                and ranked[0][:2] == ranked[1][:2]
                and ranked[0][3].raw_text != ranked[1][3].raw_text):
            return None
        best = ranked[0]
        return DialogueMemoryRecall(
            best[3].raw_text,
            best[4],
            best[3].source_hash,
            best[2],
            best[5],
            best[0],
            len(shortlist),
            posting_reads,
        )

    def recent_turns(self, *, limit: int = 6) -> tuple[DialogueMemoryTurn, ...]:
        """按逻辑轮次返回当前 owner 最近活动表层，供对话图热区使用。"""
        if type(limit) is not int or not 1 <= limit <= 64:
            raise ValueError("recent Memory limit 必须是 1..64 整数")
        rows = self.backend.select(
            "source_record",
            where={
                "source_kind": _DIALOGUE_MEMORY_SOURCE_KIND,
                "source_id": self.source_id,
            },
            order_by="document_id",
            descending=True,
            limit=limit,
        )
        repository = self.intake.source_intake.repository
        result = []
        for row in reversed(rows):
            record = repository.read(row["source_hash"])
            source = SourceRef.from_stable_key(record.source_key)
            if source.owner != self.owner:
                raise RuntimeError("recent Memory 来源跨 owner")
            self.intake.require_current_manifest(source)
            postings = self.backend.select(
                DIALOGUE_MEMORY_POSTING_TABLE,
                where={"source_hash": record.source_hash})
            if not postings:
                raise RuntimeError("recent Memory 来源缺少 posting")
            turn_seq = postings[0]["turn_seq"]
            speaker_kind = postings[0]["speaker_kind"]
            if (turn_seq != source.document_id
                    or any(item["turn_seq"] != turn_seq
                           or item["speaker_kind"] != speaker_kind
                           for item in postings)):
                raise RuntimeError("recent Memory posting 元数据漂移")
            result.append(DialogueMemoryTurn(
                record.raw_text,
                source,
                record.source_hash,
                turn_seq,
                speaker_kind,
            ))
        return tuple(result)

    def _restore_next_turn_seq(self) -> int:
        """从当前 session 的来源行恢复下一逻辑序，不使用墙钟。"""
        rows = self.backend.select(
            "source_record",
            where={
                "source_kind": _DIALOGUE_MEMORY_SOURCE_KIND,
                "source_id": self.source_id,
            },
            order_by="document_id",
            descending=True,
            limit=1,
        )
        return 1 if not rows else int(rows[0]["document_id"]) + 1

    def _ensure_postings(
            self,
            source_hash: int,
            surface: str,
            *,
            turn_seq: int,
            speaker_kind: int,
            ) -> int:
        """幂等补全派生 posting；已有冲突行时 fail closed。"""
        features = sorted(self._features(surface), key=lambda item: (
            item[1], item[0]))
        expected = tuple(
            (
                source_hash, feature, width, ordinal, turn_seq, speaker_kind,
                self.owner.tenant_id, self.owner.user_id, self.owner.session_id,
            )
            for ordinal, (feature, width) in enumerate(features)
        )
        rows = self.backend.select(
            DIALOGUE_MEMORY_POSTING_TABLE,
            where={"source_hash": source_hash})
        existing = {
            (
                row["source_hash"], row["feature_hash"],
                row["feature_width"], row["feature_ordinal"],
                row["turn_seq"], row["speaker_kind"],
                row["tenant_id"], row["user_id"], row["session_id"],
            )
            for row in rows
        }
        if len(existing) != len(rows) or not existing.issubset(set(expected)):
            raise RuntimeError("Memory posting 存在重复或冲突")
        for row in expected:
            if row in existing:
                continue
            self.backend.insert(DIALOGUE_MEMORY_POSTING_TABLE, {
                "source_hash": row[0],
                "feature_hash": row[1],
                "feature_width": row[2],
                "feature_ordinal": row[3],
                "turn_seq": row[4],
                "speaker_kind": row[5],
                "tenant_id": row[6],
                "user_id": row[7],
                "session_id": row[8],
            })
        return len(expected)

    @staticmethod
    def _exact_features(surface: str) -> frozenset[tuple[int, ...]]:
        """返回 1..3 宽度的原始码点特征，供 hash 命中后的碰撞核验。"""
        values = tuple(ord(character) for character in surface.strip())
        return frozenset(
            values[offset:offset + width]
            for width in (1, 2, 3)
            for offset in range(max(0, len(values) - width + 1))
        )

    @classmethod
    def _features(cls, surface: str) -> frozenset[tuple[int, int]]:
        """把原始码点特征投影为 SQLite 安全正整数 hash 与宽度。"""
        return frozenset(
            ((_FEATURE_HASHER.h63(item) or 1), len(item))
            for item in cls._exact_features(surface)
        )


__all__ = [
    "DIALOGUE_MEMORY_POSTING_TABLE",
    "DialogueMemoryAppend",
    "DialogueMemoryRecall",
    "DialogueMemoryTurn",
    "TrainedDialogueMemoryGraph",
    "register_dialogue_memory_posting_table",
]
