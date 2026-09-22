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
from pure_integer_ai.cognition.shared.memory_event import (
    EpisodePayload,
    EvidencePayload,
    HypothesisPayload,
    INTAKE_DERIVED_EVIDENCE,
    INTAKE_DERIVED_HYPOTHESIS,
    INTAKE_DERIVED_OBSERVATION,
    MEMORY_EVENT_EVIDENCE,
    MEMORY_EVENT_EPISODE,
    MEMORY_EVENT_HYPOTHESIS,
    MEMORY_EVENT_OBSERVATION,
    MEMORY_OBJECT_EVIDENCE,
    MEMORY_OBJECT_HYPOTHESIS,
    MEMORY_OBJECT_OBSERVATION,
    MemoryEvent,
    MemoryObjectRef,
    memory_object_ref,
    ObservationPayload,
    MemoryLinkedRef,
)
from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EVIDENCE_UNKNOWN,
)
from pure_integer_ai.cognition.shared.memory_event_log import (
    MaterializedMemoryEvent,
    MemoryEventLog,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    CLOCK_MEMORY_OBSERVED,
    LogicalClockIdentity,
    document_scope,
)
from pure_integer_ai.cognition.shared.memory_overlay import CoreIdentityCatalog
from pure_integer_ai.cognition.understanding.memory_intake import (
    HypothesisIntakeDraft,
    MemorySourceIntake,
    ObservationIntakeDraft,
    interaction_intake_policy,
)
from pure_integer_ai.cognition.understanding.artifact_query_input import (
    ArtifactMemoryObservationDraft,
    ArtifactQueryInput,
)
from pure_integer_ai.cognition.understanding.query_input_structure import (
    QueryInputStructure,
    TrainedInputStructureProjector,
    graph_query_input_structure,
)
from pure_integer_ai.cognition.understanding.query_memory_candidates import (
    MEMORY_CATEGORY_DISCOURSE as _MEMORY_CATEGORY_DISCOURSE,
    MEMORY_CATEGORY_OPEN_RELATION,
    MEMORY_CATEGORY_TIME as _MEMORY_CATEGORY_TIME,
    MEMORY_GENERIC_CANDIDATE_VERSION as _MEMORY_GENERIC_CANDIDATE_VERSION,
    frame_key as _frame_key,
    memory_candidate_category as _memory_candidate_category,
    memory_candidate_routes,
    memory_input_candidate_keys,
    memory_topic_input_proof,
    memory_relation_candidate_key,
)
from pure_integer_ai.cognition.understanding.source_intake import (
    SourceIntake,
    SourceSlice,
)
from pure_integer_ai.crosscut.determinism.hasher import Hasher
from pure_integer_ai.storage import discipline as disc
from pure_integer_ai.storage.assertion_identity import (
    IDENTITY_SOURCE_RECORD,
    register_assertion_identity_tables,
)
from pure_integer_ai.experiments.trained_memory_candidate_index import (
    MemoryCandidateIndex,
    evidence_target_route,
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
from pure_integer_ai.storage.node_store import register_node_tables
from pure_integer_ai.storage.graph_statement import (
    register_graph_statement_table,
)
from pure_integer_ai.storage.memory_event import register_memory_event_table
from pure_integer_ai.cognition.shared.graph_ontology import GraphOntology
from pure_integer_ai.experiments.trained_response_occurrence import build_response_occurrence_index
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

# Memory hypothesis keys are open, source-scoped candidates.  These integer
# namespaces describe only the structural role proposed by the input graph;
# they never impersonate a Core TypedRef or preserve the source surface.
_MEMORY_HYPOTHESIS_NAMESPACE = 91504
_MEMORY_LINEAGE_NAMESPACE = 91505
_MEMORY_COMPETITION_NAMESPACE = 91506
_MEMORY_EVIDENCE_NAMESPACE = 91507
_MEMORY_CONCEPT_PROOF_NAMESPACE = 91508
_MEMORY_RELATION_MARKER_NAMESPACE = 91536
_MEMORY_RELATION_PROOF_NAMESPACE = 91537

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
    register_node_tables(backend)
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
    hypothesis_keys: tuple[tuple[int, ...], ...]
    evidence_keys: tuple[tuple[int, ...], ...]


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


def _integer_key(value: tuple[int, ...], *, label: str) -> None:
    if (type(value) is not tuple or not value
            or any(type(item) is not int or item < 0 for item in value)):
        raise ValueError(f"{label} 必须是非空非负严格整数 tuple")


@dataclass(frozen=True, slots=True)
class MemoryObservationStructure:
    """活动 Observation 的无表层整数恢复行。"""

    observation_key: tuple[int, ...]
    source_ref: tuple[int, ...]
    scope_key: tuple[int, ...]
    source_hash: int
    turn_seq: int
    speaker_kind: int
    hypothesis_keys: tuple[tuple[int, ...], ...]
    relation_refs: tuple[MemoryLinkedRef, ...] = ()
    context_depth: int = 0

    def __post_init__(self) -> None:
        for label in ("observation_key", "source_ref", "scope_key"):
            _integer_key(getattr(self, label), label=label)
        if (type(self.source_hash) is not int or self.source_hash <= 0
                or type(self.turn_seq) is not int or self.turn_seq <= 0
                or type(self.speaker_kind) is not int or self.speaker_kind <= 0):
            raise ValueError("Memory Observation 数值字段必须是正严格整数")
        if (type(self.hypothesis_keys) is not tuple
                or not self.hypothesis_keys
                or any(type(item) is not tuple for item in self.hypothesis_keys)):
            raise ValueError("Memory Observation 必须保留 Hypothesis 整数引用")
        for item in self.hypothesis_keys:
            _integer_key(item, label="hypothesis_key")
        if self.hypothesis_keys != tuple(sorted(set(self.hypothesis_keys))):
            raise ValueError("Memory Observation Hypothesis 引用必须排序去重")
        if (type(self.relation_refs) is not tuple
                or any(not isinstance(item, MemoryLinkedRef) for item in self.relation_refs)
                or type(self.context_depth) is not int or self.context_depth < 0):
            raise ValueError("Memory Observation 关系引用或上下文深度非法")


@dataclass(frozen=True, slots=True)
class MemoryHypothesisStructure:
    """活动 Hypothesis 及其 Observation/Evidence 连接，不含原文。"""

    observation_key: tuple[int, ...]
    hypothesis_key: tuple[int, ...]
    hypothesis_kind: tuple[int, ...]
    candidate_key: tuple[int, ...]
    competition_key: tuple[int, ...]
    source_ref: tuple[int, ...]
    scope_key: tuple[int, ...]
    evidence_keys: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        for label in (
                "observation_key", "hypothesis_key", "hypothesis_kind",
                "candidate_key", "competition_key", "source_ref", "scope_key"):
            _integer_key(getattr(self, label), label=label)
        if (type(self.evidence_keys) is not tuple or not self.evidence_keys
                or any(type(item) is not tuple for item in self.evidence_keys)):
            raise ValueError("Memory Hypothesis 必须保留 Evidence 整数引用")
        for item in self.evidence_keys:
            _integer_key(item, label="evidence_key")
        if self.evidence_keys != tuple(sorted(set(self.evidence_keys))):
            raise ValueError("Memory Hypothesis Evidence 引用必须排序去重")


@dataclass(frozen=True, slots=True)
class MemoryEvidenceStructure:
    """活动 Evidence 的实际 stance、来源和结构细节。"""

    evidence_key: tuple[int, ...]
    hypothesis_key: tuple[int, ...]
    source_ref: tuple[int, ...]
    scope_key: tuple[int, ...]
    stance: int
    detail: tuple[int, ...]
    payload_key: tuple[int, ...]
    supersedes_key: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        for label in (
                "evidence_key", "hypothesis_key", "source_ref", "scope_key"):
            _integer_key(getattr(self, label), label=label)
        if self.stance not in {1, 2, 3}:
            raise ValueError("Memory Evidence stance 未注册")
        if (type(self.detail) is not tuple
                or any(type(item) is not int or item < 0
                       for item in self.detail)):
            raise ValueError("Memory Evidence detail 必须是非负严格整数 tuple")
        _integer_key(self.payload_key, label="Memory Evidence payload_key")
        if self.supersedes_key:
            _integer_key(self.supersedes_key, label="Memory Evidence supersedes_key")


@dataclass(frozen=True, slots=True)
class MemoryActiveStructures:
    """一次 ACL/manifest 核验后恢复的 O/H/E 整数图快照。"""

    observations: tuple[MemoryObservationStructure, ...]
    hypotheses: tuple[MemoryHypothesisStructure, ...]
    evidence: tuple[MemoryEvidenceStructure, ...]
    match_routes: tuple[tuple[int, ...], ...] = ()

    def __post_init__(self) -> None:
        for label, expected in (
                ("observations", MemoryObservationStructure),
                ("hypotheses", MemoryHypothesisStructure),
                ("evidence", MemoryEvidenceStructure)):
            rows = getattr(self, label)
            if (type(rows) is not tuple
                    or any(not isinstance(item, expected) for item in rows)):
                raise TypeError(f"MemoryActiveStructures.{label} 类型非法")
        if self.observations != tuple(sorted(
                self.observations,
                key=lambda item: (item.turn_seq, item.observation_key))):
            raise ValueError("Memory Observation 行必须按逻辑轮次排序")
        if self.hypotheses != tuple(sorted(
                self.hypotheses, key=lambda item: item.hypothesis_key)):
            raise ValueError("Memory Hypothesis 行必须按整数键排序")
        if self.evidence != tuple(sorted(
                self.evidence, key=lambda item: item.evidence_key)):
            raise ValueError("Memory Evidence 行必须按整数键排序")
        for route in self.match_routes:
            _integer_key(route, label="Memory match route")


@dataclass(frozen=True)
class _DialogueObservationParser:
    """把完整 SourceSlice 映射为来源化 O/H/E 结构候选。"""

    owner: OwnerScope
    session_id: int
    speaker_kind: int
    input_projector: TrainedInputStructureProjector | None = None
    stance: int = EVIDENCE_SUPPORT
    context_refs: tuple[MemoryObjectRef, ...] = ()
    input_structure: QueryInputStructure | None = None

    def __post_init__(self) -> None:
        if type(self.stance) is not int or self.stance not in {
                EVIDENCE_SUPPORT, EVIDENCE_REFUTE, EVIDENCE_UNKNOWN}:
            raise ValueError("Memory Evidence stance 未注册")

    def _hypotheses(
            self, source: SourceSlice,
            *, input_structure=None,
            ) -> tuple[HypothesisIntakeDraft, ...]:
        """以训练结构形成开放假设；无投影器时只产生不可匹配的通用结构。"""
        if input_structure is None and self.input_projector is not None:
            input_structure = self.input_projector.project(tuple(
                ord(value) for value in source.text))
        if self.input_structure is not None:
            candidate_rows = tuple(
                (_memory_candidate_category(candidate), candidate)
                for candidate in memory_input_candidate_keys(
                    input_structure, session_id=self.session_id,
                    speaker_kind=self.speaker_kind))
        elif self.input_projector is None:
            candidate_rows = (
                (_MEMORY_CATEGORY_TIME, _frame_key(
                    _MEMORY_GENERIC_CANDIDATE_VERSION,
                    (_MEMORY_CATEGORY_TIME,),
                    (self.session_id, self.speaker_kind))),
                (_MEMORY_CATEGORY_DISCOURSE, _frame_key(
                    _MEMORY_GENERIC_CANDIDATE_VERSION,
                    (_MEMORY_CATEGORY_DISCOURSE,),
                    (self.session_id, self.speaker_kind))),
            )
        else:
            candidate_rows = tuple(
                (_memory_candidate_category(candidate), candidate)
                for candidate in memory_input_candidate_keys(
                        input_structure, session_id=self.session_id,
                        speaker_kind=self.speaker_kind))
            # Inputs with no trained filler still need a Dialogue context even
            # when an open-role hypothesis was also formed.  Keep both roots
            # in the same O/H/E observation: the open relation carries the
            # structural UNKNOWN, while the generic context supplies the
            # trained response-act route.  Generic identities are namespace
            # separated and can never become a Core semantic match.
            has_memory_structure = any(
                candidate_key[0] != _MEMORY_GENERIC_CANDIDATE_VERSION
                for _category, candidate_key in candidate_rows)
            if not has_memory_structure:
                candidate_rows = (
                    *candidate_rows,
                    (_MEMORY_CATEGORY_TIME, _frame_key(
                        _MEMORY_GENERIC_CANDIDATE_VERSION,
                        (_MEMORY_CATEGORY_TIME,),
                        input_structure.source_ref)),
                    (_MEMORY_CATEGORY_DISCOURSE, _frame_key(
                        _MEMORY_GENERIC_CANDIDATE_VERSION,
                        (_MEMORY_CATEGORY_DISCOURSE,),
                        input_structure.source_ref)),
                )
        # A generic candidate may already be supplied by the input projector;
        # adding the dialogue context must be idempotent for the same source.
        # Deduplicate by its semantic integer identity before assigning a
        # lineage.  Otherwise the same O/H pair receives different ordinal
        # lineages and MemoryIntake correctly rejects it as duplicate data.
        unique_rows = {}
        for category, candidate_key in candidate_rows:
            unique_rows[(category, candidate_key)] = (category, candidate_key)
        candidate_rows = tuple(
            unique_rows[key] for key in sorted(unique_rows))
        result = []
        for ordinal, (category, candidate_key) in enumerate(candidate_rows):
            proof = ()
            if input_structure is not None:
                topic_proof = memory_topic_input_proof(candidate_key, input_structure)
                if topic_proof:
                    proof = (topic_proof,)
                if category == 5:
                    relation_rows = tuple(
                        item for item in input_structure.relation_candidates
                        if memory_relation_candidate_key(item) == candidate_key)
                    if len(relation_rows) != 1:
                        raise ValueError("Memory 关系候选无法唯一回读")
                    relation = relation_rows[0]
                    members = tuple(
                        item.stable_key()
                        for item in input_structure.semantic_candidates
                        if item.proposition_key == relation.proposition_key
                        and item.source_ref == relation.source_ref)
                    proof += (_frame_key(
                        _MEMORY_RELATION_PROOF_NAMESPACE,
                        relation.stable_key(),
                        _frame_key(_MEMORY_RELATION_MARKER_NAMESPACE,
                                   relation.predicate_key,
                                   relation.structure_key,
                                   relation.carrier_key),
                        _frame_key(1, *members),
                        _frame_key(2, *relation.span_refs),
                    ),)
            if input_structure is not None and input_structure.concept_routes:
                routes = frozenset(memory_candidate_routes(candidate_key))
                related = tuple(item for item in input_structure.concept_routes
                                if (_frame_key(2, item.target_key) in routes
                                    or _frame_key(3, item.target_proposition_key) in routes))
                if related:
                    proof += (_frame_key(
                        _MEMORY_CONCEPT_PROOF_NAMESPACE,
                        _frame_key(1, *(item.stable_key() for item in related)),
                        _frame_key(2, *(item.stable_key()
                                        for item in input_structure.concept_links))),)
            result.append(HypothesisIntakeDraft(
                _frame_key(
                    _MEMORY_LINEAGE_NAMESPACE,
                    (source.source.document_id, category),
                    candidate_key),
                (_MEMORY_HYPOTHESIS_NAMESPACE, category),
                candidate_key,
                _frame_key(
                    _MEMORY_COMPETITION_NAMESPACE,
                    (category,),
                    candidate_key),
                EVIDENCE_UNKNOWN if self.stance == EVIDENCE_SUPPORT and (
                    category == MEMORY_CATEGORY_OPEN_RELATION
                    or candidate_key[0] == _MEMORY_GENERIC_CANDIDATE_VERSION
                ) else self.stance,
                reason_key=_frame_key(
                    _MEMORY_EVIDENCE_NAMESPACE,
                    (source.source.document_id, category),
                    candidate_key),
                detail=_frame_key(
                    _MEMORY_EVIDENCE_NAMESPACE,
                    (self.session_id, self.speaker_kind, category),
                    candidate_key, *proof),
            ))
        return tuple(result)

    @staticmethod
    def _relation_occurrences(
            input_structure,
            context: MemoryLinkedRef,
            ) -> tuple[MemoryLinkedRef, ...]:
        """把当前输入的关系 marker/role/proposition 作为一等图引用保留。"""
        keys: set[tuple[int, ...]] = set(input_structure.graph_object_keys)
        for relation in input_structure.relation_candidates:
            keys.update((
                relation.proposition_key,
                relation.predicate_key,
                relation.structure_key,
                relation.carrier_key,
            ))
        # semantic filler/role 是关系槽位的完整成员；只保留整数身份，
        # 不把输入正文或课程表层复制进 Observation。
        for member in input_structure.semantic_candidates:
            if member.projection_kind == 1:
                keys.update((
                    member.candidate_key,
                    member.role_key,
                    member.proposition_key,
                    member.predicate_key,
                ))
        refs = [context]
        for key in sorted(keys):
            try:
                refs.append(MemoryLinkedRef.object(
                    ObjectIdentity.from_stable_key(key)))
            except (TypeError, ValueError):
                # 非 ObjectIdentity 的旧投影不冒充关系端点；候选本身仍
                # 通过 Hypothesis/Evidence 完整保存并可回读。
                continue
        return tuple(sorted(set(refs), key=lambda item: item.stable_key()))

    def parse(self, source: SourceSlice) -> ObservationIntakeDraft:
        """保留来源和会话上下文；只写开放候选，不伪造 Core 命题。"""
        if not isinstance(source, SourceSlice):
            raise ValueError("dialogue Memory parser 需要完整来源切片")
        if (not source.text and (self.input_structure is None
                                or not self.input_structure.graph_object_keys)):
            raise ValueError("空来源必须携带 graph-only 输入结构")
        context = ObjectIdentity(
            OBJECT_CONTEXT_SCOPE,
            (_DIALOGUE_MEMORY_NAMESPACE, self.session_id,
             self.speaker_kind),
            self.owner,
            source.source.versions,
        )
        input_structure = self.input_structure
        if input_structure is not None:
            if (not isinstance(input_structure, QueryInputStructure)
                    or input_structure.token_values != tuple(
                        ord(value) for value in source.text)):
                raise ValueError("显式输入结构与 Memory 来源整数不一致")
        elif self.input_projector is not None:
            input_structure = self.input_projector.project(tuple(
                ord(value) for value in source.text))
        relation_occurrences = (self._relation_occurrences(
            input_structure, MemoryLinkedRef.object(context))
            if input_structure is not None else
            (MemoryLinkedRef.object(context),))
        return ObservationIntakeDraft(
            (_DIALOGUE_MEMORY_NAMESPACE, source.source.document_id),
            MemoryLinkedRef.object(context),
            relation_occurrences=tuple(sorted((
                *relation_occurrences,
                *(MemoryLinkedRef.memory(ref) for ref in self.context_refs),
            ), key=lambda item: item.stable_key())),
            hypotheses=self._hypotheses(source, input_structure=input_structure),
        )


@dataclass(frozen=True)
class _GeneratedObservationParser:
    """已完成生成的结构草案直接摄入；来源正文不重新派生事实。"""

    draft: ObservationIntakeDraft

    def parse(self, source: SourceSlice) -> ObservationIntakeDraft:
        if not isinstance(source, SourceSlice) or not source.text:
            raise ValueError("生成观察需要实际输出来源")
        context = self.draft.context.value()
        if not isinstance(context, ObjectIdentity) or context.owner != source.source.owner:
            raise ValueError("生成观察与输出来源 owner 不一致")
        return self.draft


@dataclass
class _ArtifactObservationParser:
    """把已验证的 carrier projection 写成当前 session 的 O/H/E，不重解析原文。"""

    artifact_input: ArtifactQueryInput
    session_id: int
    speaker_kind: int
    materialized: ArtifactMemoryObservationDraft | None = None

    def parse(self, source: SourceSlice) -> ObservationIntakeDraft:
        """只允许 Unicode carrier 通过既有 SourceRecord 边界，OCTET 不作伪文本转换。"""
        if not isinstance(source, SourceSlice) or not source.text:
            raise ValueError("artifact Memory parser 需要非空 Unicode 来源切片")
        self.materialized = self.artifact_input.memory_observation_draft(
            source.source, session_id=self.session_id,
            speaker_kind=self.speaker_kind)
        return self.materialized.draft


# object-model: resource_owner; representation=runtime; interop=dialogue-memory-graph-v1
class TrainedDialogueMemoryGraph:
    """拥有可写 session SQLite，并把每次召回约束到活动 Memory 图来源。"""

    __slots__ = (
        "path", "backend", "intake", "owner", "session_id",
        "source_id", "input_projector",
        "_next_turn_seq", "candidate_index", "occurrence_index",
    )

    def __init__(
            self,
            database: str | Path,
            *,
            tenant_id: int = 1,
            user_id: int = 1,
            session_id: int = 1,
            input_projector: TrainedInputStructureProjector | None = None,
            ) -> None:
        """打开或创建 session Memory 图；身份参数必须为严格正整数。"""
        for label, value in (
                ("tenant_id", tenant_id),
                ("user_id", user_id),
                ("session_id", session_id)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{label} 必须是严格正整数")
        if (input_projector is not None
                and not isinstance(input_projector,
                                   TrainedInputStructureProjector)):
            raise TypeError("input_projector 必须是 TrainedInputStructureProjector 或 None")
        self.path = Path(database).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.backend = SQLiteBackend(str(self.path))
        try:
            self.intake = _build_interaction_memory_intake(self.backend)
            self.occurrence_index = build_response_occurrence_index(self.intake.event_log)
            register_dialogue_memory_posting_table(self.backend)
            self.owner = OwnerScope(
                tenant_id, user_id, session_id, VISIBILITY_SESSION)
            self.session_id = session_id
            self.input_projector = input_projector
            self.source_id = (
                Hasher("trained_dialogue_memory.session_source.v1").h63(
                    self.owner.stable_key()) or 1)
            self._next_turn_seq = self._restore_next_turn_seq()
            self.candidate_index = MemoryCandidateIndex(
                self.intake.event_log,
                SourceRef(_DIALOGUE_MEMORY_SOURCE_KIND, self.source_id, 0,
                          self.owner, _VERSIONS),
                memory_candidate_routes,
            )
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

    def append(
            self,
            surface: str,
            *,
            speaker_kind: int,
            stance: int = EVIDENCE_SUPPORT,
            observation_draft: ObservationIntakeDraft | None = None,
            input_structure: QueryInputStructure | None = None,
            ) -> DialogueMemoryAppend:
        """把一轮表层写入 O/H/E；stance 是追加 Evidence，不覆盖旧记录。"""
        if type(surface) is not str or not surface.strip():
            raise ValueError("Memory 表层必须是非空文本")
        if type(speaker_kind) is not int or speaker_kind <= 0:
            raise ValueError("speaker_kind 必须是严格正整数")
        if type(stance) is not int or stance not in {
                EVIDENCE_SUPPORT, EVIDENCE_REFUTE, EVIDENCE_UNKNOWN}:
            raise ValueError("Memory Evidence stance 未注册")
        if observation_draft is not None and not isinstance(observation_draft, ObservationIntakeDraft):
            raise TypeError("显式生成观察必须是 ObservationIntakeDraft")
        if input_structure is not None and not isinstance(input_structure, QueryInputStructure):
            raise TypeError("input_structure 必须是 QueryInputStructure 或 None")
        if observation_draft is not None and input_structure is not None:
            raise ValueError("observation_draft 与 input_structure 不得同时提供")
        parser = (_GeneratedObservationParser(observation_draft)
                  if observation_draft is not None else _DialogueObservationParser(
                      self.owner, self.session_id, speaker_kind, self.input_projector, stance,
                      self._preceding_response_refs() if speaker_kind == 1 else (),
                      input_structure))
        appended, _result = self._append_with_parser(
            surface, speaker_kind=speaker_kind, parser=parser)
        return appended

    def append_graph_input(
            self,
            graph_object_keys: tuple[tuple[int, ...], ...],
            *,
            speaker_kind: int = 1,
        ) -> DialogueMemoryAppend:
        """Persist a graph-only interaction Observation using integer identity only."""
        if type(speaker_kind) is not int or speaker_kind <= 0:
            raise ValueError("speaker_kind 必须是严格正整数")
        structure = graph_query_input_structure(graph_object_keys)
        parser = _DialogueObservationParser(
            self.owner,
            self.session_id,
            speaker_kind,
            self.input_projector,
            EVIDENCE_SUPPORT,
            self._preceding_response_refs() if speaker_kind == 1 else (),
            structure,
        )
        appended, _result = self._append_with_parser(
            "", speaker_kind=speaker_kind, parser=parser)
        return appended

    def append_artifact(
            self,
            artifact_input: ArtifactQueryInput,
            *, speaker_kind: int,
            ) -> DialogueMemoryAppend:
        """把 carrier projection 作为一等交互输入持久化，禁止 OCTET 走文本存储旁路。"""
        if not isinstance(artifact_input, ArtifactQueryInput):
            raise TypeError("artifact_input 必须是 ArtifactQueryInput")
        if type(speaker_kind) is not int or speaker_kind <= 0:
            raise ValueError("speaker_kind 必须是严格正整数")
        surface = artifact_input.unicode_surface()
        if not surface.strip():
            raise ValueError("artifact Memory 不接受空白 carrier surface")
        parser = _ArtifactObservationParser(
            artifact_input, self.session_id, speaker_kind)
        appended, result = self._append_with_parser(
            surface, speaker_kind=speaker_kind, parser=parser)
        materialized = parser.materialized
        if materialized is None:
            raise RuntimeError("artifact Memory parser 没有产生 O/H/E 草案")
        if len(result.hypothesis_refs) != len(materialized.evidence_tails):
            raise RuntimeError("artifact Memory hypothesis/evidence 映射漂移")
        # ingest 保留第一条 Evidence；余下每条按原 stance 追加，不能汇总掉冲突。
        for hypothesis_ref, tails in zip(
                result.hypothesis_refs, materialized.evidence_tails):
            for evidence in tails:
                self.append_evidence(
                    hypothesis_ref.stable_key(), source=appended.source,
                    stance=evidence.stance,
                    reason_key=_frame_key(
                        91542, (1,), artifact_input.envelope.identity.stable_key(),
                        evidence.stable_key()),
                    detail=_frame_key(
                        91542, (2,), artifact_input.envelope.identity.stable_key(),
                        evidence.stable_key()),
                )
        return appended

    def _append_with_parser(
            self, surface: str, *, speaker_kind: int, parser,
            ) -> tuple[DialogueMemoryAppend, object]:
        """复用单一来源摄入和 posting 边界，避免 artifact 另建 session 写路径。"""
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
            parser=parser,
        )
        posting_count = self._ensure_postings(
            result.source_record.source_hash,
            surface,
            turn_seq=turn_seq,
            speaker_kind=speaker_kind,
        )
        self.candidate_index.synchronize()
        self.backend.commit()
        self._next_turn_seq += 1
        return DialogueMemoryAppend(
            source,
            result.source_record.source_hash,
            turn_seq,
            speaker_kind,
            posting_count,
            result.observation_ref.object_key,
            tuple(item.object_key for item in result.hypothesis_refs),
            tuple(item.object_key for item in result.evidence_refs),
        ), result

    def _preceding_response_refs(self) -> tuple[MemoryObjectRef, ...]:
        """沿已持久 Episode 的真实输出端点承接会话，不读取最近回复正文。"""
        access = self.candidate_index.access
        candidates = []
        for item in self.intake.event_log.query(access=access, event_kind=MEMORY_EVENT_EPISODE):
            payload = item.event.payload
            if (not isinstance(payload, EpisodePayload) or payload.output_ref is None
                    or item.event.object_ref.owner != self.owner):
                continue
            ref = payload.output_ref.value()
            if not isinstance(ref, MemoryObjectRef) or ref.object_kind != MEMORY_OBJECT_OBSERVATION:
                continue
            output = self._event_payload(self.intake.event_log.query(
                access=access, event_kind=MEMORY_EVENT_OBSERVATION, object_ref=ref),
                ObservationPayload, label="preceding output")
            if (output.source.owner != self.owner or output.source.source_id != self.source_id
                    or output.source.source_kind != _DIALOGUE_MEMORY_SOURCE_KIND):
                raise ValueError("前序回应不属于当前会话来源")
            self.intake.require_current_manifest(output.source)
            candidates.append((output.source.document_id, ref))
        if not candidates:
            return ()
        latest = max(seq for seq, _ in candidates)
        return tuple(sorted({ref for seq, ref in candidates if seq == latest},
                            key=lambda ref: ref.stable_key()))

    def retract(self, surface: str, *, speaker_kind: int) -> DialogueMemoryAppend:
        """追加一条反对 Evidence；不删除或覆盖先前 Observation/Hypothesis。"""
        return self.append(
            surface,
            speaker_kind=speaker_kind,
            stance=EVIDENCE_REFUTE,
        )

    def append_evidence(
            self, hypothesis_key: tuple[int, ...], *, source: SourceRef,
            stance: int, reason_key: tuple[int, ...], detail: tuple[int, ...] = (),
            supersedes_key: tuple[int, ...] = (),
            ) -> MemoryEvidenceStructure:
        """向既有假设追加来源化 Evidence；显式替代也保留原证据及其连接。"""
        hypothesis_ref = MemoryObjectRef.from_stable_key(hypothesis_key)
        if (not isinstance(source, SourceRef) or source.owner != self.owner
                or hypothesis_ref.owner != self.owner
                or hypothesis_ref.memory_space != self.intake.event_log.memory_space_identity):
            raise ValueError("追加 Memory Evidence 必须属于当前 owner 和 Memory 空间")
        self.intake.require_current_manifest(source)
        if type(stance) is not int or stance not in {
                EVIDENCE_SUPPORT, EVIDENCE_REFUTE, EVIDENCE_UNKNOWN}:
            raise ValueError("Memory Evidence stance 未注册")
        _integer_key(reason_key, label="Memory Evidence reason_key")
        scope = document_scope(source)
        clock = self.intake.event_log.scoped_identities.resume_clock(
            LogicalClockIdentity(scope, CLOCK_MEMORY_OBSERVED))
        payload = EvidencePayload(
            hypothesis_ref, stance, None, reason_key, source, None, detail,
            MemoryObjectRef.from_stable_key(supersedes_key) if supersedes_key else None,
            clock.advance(),
        )
        ref = memory_object_ref(
            self.intake.event_log.memory_space_identity, MEMORY_OBJECT_EVIDENCE,
            payload.stable_key(), owner=source.owner, versions=source.versions)
        event = self.intake.event_log.append(MemoryEvent(
            MEMORY_EVENT_EVIDENCE, ref, scope, payload))
        self.candidate_index.synchronize()
        self.backend.commit()
        return self._restore_evidence(event)

    def _restore_evidence(self, event: MaterializedMemoryEvent) -> MemoryEvidenceStructure:
        """从证据自身恢复来源、范围、完整载荷和替代引用，不借用假设来源。"""
        payload = event.event.payload
        if not isinstance(payload, EvidencePayload) or payload.source is None:
            raise RuntimeError("当前会话 Evidence 恢复需要显式 SourceRef")
        self.intake.require_current_manifest(payload.source)
        source_hash = self.intake.event_log.scoped_identities.registry.find(
            IDENTITY_SOURCE_RECORD, payload.source.stable_key())
        if source_hash is None:
            raise RuntimeError("Memory Evidence 来源身份缺失")
        return MemoryEvidenceStructure(
            event.event.object_ref.stable_key(), payload.hypothesis_ref.stable_key(),
            (source_hash, *payload.source.stable_key()), event.event.scope.stable_key(),
            payload.stance, payload.detail, payload.stable_key(),
            () if payload.supersedes_ref is None else payload.supersedes_ref.stable_key(),
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
                if record.raw_text == "":
                    # Graph-only turns have no surface posting by design;
                    # their O/H/E remains available through active_structures.
                    continue
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

    @staticmethod
    def _event_payload(
            entries: tuple[object, ...], payload_type: type,
            *, label: str,
            ) -> object:
        """要求一个对象引用只对应一条同类型声明事件。"""
        payloads = tuple(
            item.event.payload for item in entries
            if isinstance(item.event.payload, payload_type))
        if len(payloads) != 1:
            raise RuntimeError(f"Memory {label} 声明事件不唯一")
        return payloads[0]

    def active_structures(
            self, *, limit: int | None = None,
            candidate_keys: tuple[tuple[int, ...], ...] | None = None,
            observation_refs: tuple[MemoryObjectRef, ...] = (),
            ) -> MemoryActiveStructures:
        """按共同图节点定位全历史 O/H/E；limit 只供显式诊断预览使用。"""
        if limit is not None and (type(limit) is not int or limit <= 0):
            raise ValueError("active Memory limit 必须是正整数或 None")
        if (candidate_keys is not None or observation_refs) and limit is not None:
            raise ValueError("结构语义查询不得截断相关历史")
        if (type(observation_refs) is not tuple
                or any(not isinstance(ref, MemoryObjectRef)
                       or ref.object_kind != MEMORY_OBJECT_OBSERVATION for ref in observation_refs)):
            raise TypeError("当前输入锚点必须是实际 Observation 引用")
        access = self.intake._access(SourceRef(
            _DIALOGUE_MEMORY_SOURCE_KIND,
            self.source_id,
            1,
            self.owner,
            _VERSIONS,
        ))
        matched = None
        query_routes: dict[tuple[int, ...], set[tuple[int, ...]]] = {}
        if candidate_keys is None:
            observation_events = self.intake.event_log.query(
                access=access, event_kind=MEMORY_EVENT_OBSERVATION)
        else:
            for candidate in candidate_keys:
                for route in memory_candidate_routes(candidate):
                    query_routes.setdefault(route, set()).add(candidate)
            matched: dict[SourceRef, set[tuple[int, ...]]] = {}
            for event in self.candidate_index.matching_events(tuple(query_routes)):
                hypothesis = event.event.payload.hypothesis
                matched.setdefault(hypothesis.observation, set()).add(
                    event.event.object_ref.stable_key())
            observation_events = []
            for source in sorted(matched):
                manifest = self.intake.require_current_manifest(source)
                for binding in manifest.bindings:
                    if binding.binding_kind == INTAKE_DERIVED_OBSERVATION:
                        observation_events.extend(self.intake.event_log.query(
                            access=access, event_kind=MEMORY_EVENT_OBSERVATION,
                            object_ref=binding.object_ref))
        observation_events, context_depths, context_sources, context_routes = self._observation_closure(
            observation_events, observation_refs, access)
        rows = []
        match_routes = set(context_routes)
        for materialized in observation_events:
            payload = materialized.event.payload
            if not isinstance(payload, ObservationPayload):
                raise RuntimeError("Memory Observation event payload 类型漂移")
            source = payload.source
            if (source.owner != self.owner
                    or source.source_kind != _DIALOGUE_MEMORY_SOURCE_KIND
                    or source.source_id != self.source_id):
                continue
            manifest = self.intake.require_current_manifest(source)
            by_kind = {
                kind: tuple(binding.object_ref for binding in manifest.bindings
                            if binding.binding_kind == kind)
                for kind in (INTAKE_DERIVED_OBSERVATION, INTAKE_DERIVED_HYPOTHESIS,
                             INTAKE_DERIVED_EVIDENCE)
            }
            if by_kind[INTAKE_DERIVED_OBSERVATION] != (materialized.event.object_ref,):
                raise RuntimeError("Memory manifest Observation 引用漂移")
            context = payload.context.value()
            if (not isinstance(context, ObjectIdentity)
                    or context.object_kind != OBJECT_CONTEXT_SCOPE
                    or context.owner != self.owner
                    or context.versions != source.versions
                    or len(context.components) != 3
                    or context.components[:2] != (
                        _DIALOGUE_MEMORY_NAMESPACE, self.session_id)):
                raise RuntimeError("Memory Observation context 不可恢复")
            speaker_kind = context.components[2]
            if type(speaker_kind) is not int or speaker_kind <= 0:
                raise RuntimeError("Memory Observation speaker_kind 非法")
            if manifest.source != source:
                raise RuntimeError("Memory manifest source 漂移")
            source_hash = self.intake.event_log.scoped_identities.registry.find(
                IDENTITY_SOURCE_RECORD, source.stable_key())
            if source_hash is None:
                raise RuntimeError("Memory Observation 来源身份缺失")
            source_ref = (source_hash, *source.stable_key())
            scope_key = materialized.event.scope.stable_key()
            hypotheses = []
            evidence = []
            hypothesis_payloads = {}
            for ref in by_kind[INTAKE_DERIVED_HYPOTHESIS]:
                restored = self._event_payload(
                    self.intake.event_log.query(
                        access=access, event_kind=MEMORY_EVENT_HYPOTHESIS, object_ref=ref),
                    HypothesisPayload, label="Hypothesis")
                candidate = restored.hypothesis.candidate_key
                if (matched is None or source in context_sources
                        or ref.stable_key() in matched.get(source, ())
                        or not memory_candidate_routes(candidate)):
                    hypothesis_payloads[ref] = restored
                    if matched is not None and ref.stable_key() not in matched.get(source, ()):
                        # 先由内容命中 Observation，再沿同来源结构边恢复旧拓扑。
                        # 此路由不把拓扑等同于话题，也不丢弃旧 O/H/E。
                        match_routes.add(_frame_key(
                            2, materialized.event.object_ref.stable_key(), ref.stable_key()))
            hypothesis_refs = tuple(hypothesis_payloads)
            evidence_by_hypothesis: dict[tuple[int, ...], list[MemoryEvidenceStructure]] = {}
            for event in self.candidate_index.matching_events(tuple(
                    evidence_target_route(ref.stable_key()) for ref in hypothesis_refs)):
                restored_evidence = self._restore_evidence(event)
                evidence_by_hypothesis.setdefault(
                    restored_evidence.hypothesis_key, []).append(restored_evidence)
            selected_lineages = {
                binding.lineage_key for binding in manifest.bindings
                if binding.binding_kind == INTAKE_DERIVED_HYPOTHESIS
                and binding.object_ref in hypothesis_refs}
            expected_evidence = {
                binding.object_ref.stable_key() for binding in manifest.bindings
                if binding.binding_kind == INTAKE_DERIVED_EVIDENCE
                and binding.lineage_key in selected_lineages}
            actual_evidence = {
                item.evidence_key for group in evidence_by_hypothesis.values() for item in group}
            if not expected_evidence <= actual_evidence:
                raise RuntimeError("Memory Evidence 索引没有恢复完整 intake 证据")
            for hypothesis_ref in hypothesis_refs:
                hypothesis_payload = hypothesis_payloads[hypothesis_ref]
                hypothesis = hypothesis_payload.hypothesis
                if hypothesis.observation != source:
                    raise RuntimeError("Memory Hypothesis observation 漂移")
                for route in memory_candidate_routes(hypothesis.candidate_key):
                    for input_candidate in query_routes.get(route, ()):
                        match_routes.add(_frame_key(
                            1, input_candidate, route, hypothesis_ref.stable_key()))
                linked = evidence_by_hypothesis.get(hypothesis_ref.stable_key(), ())
                linked_evidence = [item.evidence_key for item in linked]
                if any(item.supersedes_key and item.supersedes_key not in linked_evidence
                       for item in linked):
                    raise RuntimeError("Memory Evidence 替代链缺少原始证据")
                evidence.extend(linked)
                if not linked_evidence:
                    raise RuntimeError("Memory Hypothesis 缺少来源 Evidence")
                hypotheses.append(MemoryHypothesisStructure(
                    materialized.event.object_ref.stable_key(),
                    hypothesis_ref.stable_key(),
                    hypothesis.hypothesis_kind,
                    hypothesis.candidate_key,
                    hypothesis.competition_key,
                    source_ref,
                    scope_key,
                    tuple(sorted(linked_evidence)),
                ))
            if not hypotheses:
                raise RuntimeError("Memory Observation 缺少结构 Hypothesis")
            rows.append((
                MemoryObservationStructure(
                    materialized.event.object_ref.stable_key(),
                    source_ref,
                    scope_key,
                    source_hash,
                    source.document_id,
                    speaker_kind,
                    tuple(sorted(item.hypothesis_key for item in hypotheses)),
                    payload.relation_occurrences,
                    context_depths.get(source, 0),
                ),
                tuple(hypotheses),
                tuple(evidence),
            ))
        rows.sort(key=lambda item: (
            item[0].turn_seq, item[0].observation_key))
        selected = rows if limit is None else rows[-limit:]
        return MemoryActiveStructures(
            tuple(item[0] for item in selected),
            tuple(sorted(
                (hypothesis for _observation, hypotheses, _evidence in selected
                 for hypothesis in hypotheses),
                key=lambda item: item.hypothesis_key)),
            tuple(sorted(
                (evidence for _observation, _hypotheses, evidence_rows in selected
                 for evidence in evidence_rows),
                key=lambda item: item.evidence_key)),
            tuple(sorted(match_routes)),
        )

    def _observation_closure(self, matched_events, observation_refs, access):
        """完整恢复显式观察边的来源闭包；上下文身份不升级为事实匹配。"""
        events = self.intake.event_log
        found = {item.event.object_ref: item for item in matched_events}
        depths = {item.event.payload.source: 0 for item in matched_events}
        context_sources = set()
        routes = set()
        queue = []
        for ref in observation_refs:
            if ref.owner != self.owner or ref.memory_space != events.memory_space_identity:
                raise ValueError("输入 Observation 跨会话或跨 Memory 空间")
            rows = events.query(access=access, event_kind=MEMORY_EVENT_OBSERVATION, object_ref=ref)
            self._event_payload(rows, ObservationPayload, label="input anchor")
            found[ref] = rows[0]
            depths[rows[0].event.payload.source] = 0
            context_sources.add(rows[0].event.payload.source)
            routes.add(_frame_key(3, ref.stable_key()))
        queue.extend(found.values())
        visited = set()
        cursor = 0
        while cursor < len(queue):
            current = queue[cursor]
            cursor += 1
            if current.event.object_ref in visited:
                continue
            visited.add(current.event.object_ref)
            payload = current.event.payload
            for link in payload.relation_occurrences:
                ref = link.value()
                if not isinstance(ref, MemoryObjectRef) or ref.object_kind not in {
                        MEMORY_OBJECT_OBSERVATION, MEMORY_OBJECT_HYPOTHESIS, MEMORY_OBJECT_EVIDENCE}:
                    continue
                if ref.owner != self.owner or ref.memory_space != events.memory_space_identity:
                    raise ValueError("上下文观察边跨会话或跨 Memory 空间")
                target = ref
                if target.object_kind == MEMORY_OBJECT_EVIDENCE:
                    evidence = self._event_payload(events.query(access=access,
                        event_kind=MEMORY_EVENT_EVIDENCE, object_ref=target), EvidencePayload, label="context evidence")
                    target = evidence.hypothesis_ref
                if target.object_kind == MEMORY_OBJECT_HYPOTHESIS:
                    hypothesis = self._event_payload(events.query(access=access,
                        event_kind=MEMORY_EVENT_HYPOTHESIS, object_ref=target), HypothesisPayload, label="context hypothesis")
                    manifest = self.intake.require_current_manifest(hypothesis.hypothesis.observation)
                    target = next(binding.object_ref for binding in manifest.bindings
                                  if binding.binding_kind == INTAKE_DERIVED_OBSERVATION)
                target_events = events.query(access=access, event_kind=MEMORY_EVENT_OBSERVATION, object_ref=target)
                target_payload = self._event_payload(target_events, ObservationPayload, label="context observation")
                if (target_payload.source.source_id != self.source_id
                        or target_payload.source.source_kind != _DIALOGUE_MEMORY_SOURCE_KIND
                        or target_payload.source.document_id > payload.source.document_id):
                    raise ValueError("上下文边必须指向已有当前会话观察")
                context_sources.add(target_payload.source)
                routes.add(_frame_key(4, current.event.object_ref.stable_key(), ref.stable_key(), target.stable_key()))
                distance = depths[payload.source] + 1
                depths[target_payload.source] = min(depths.get(target_payload.source, distance), distance)
                if target not in found:
                    found[target] = target_events[0]
                    queue.append(target_events[0])
        return tuple(found.values()), depths, context_sources, tuple(sorted(routes))

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
    "MemoryActiveStructures",
    "MemoryEvidenceStructure",
    "MemoryHypothesisStructure",
    "MemoryObservationStructure",
    "DialogueMemoryAppend",
    "DialogueMemoryRecall",
    "DialogueMemoryTurn",
    "TrainedDialogueMemoryGraph",
    "memory_input_candidate_keys",
    "register_dialogue_memory_posting_table",
]
