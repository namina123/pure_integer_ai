"""普通多轮对话到核心后继命题的训练与稀疏查询接线。

训练侧只从显式 speaker/content span、Occurrence、SourceRef 和顺序事实建模；
查询侧使用纯整数 codepoint posting 找候选，再逐码点回源核验并从 response
Occurrence 端点恢复表层。它不是外置 QA 库，也不在代码中保存问题或答案。
"""
from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
import sqlite3
import unicodedata

from pure_integer_ai.cognition.shared.graph_ontology import (
    relation_concept_identity,
)
from pure_integer_ai.cognition.shared.dialogue_pipeline import (
    DIALOGUE_RESULT_CLARIFICATION,
    DIALOGUE_RESULT_EXACT,
    DIALOGUE_RESULT_RESPONSE_CLASS,
    DIALOGUE_RESULT_TRANSFER,
    DialoguePipelineTrace,
    integer_token_features,
    integer_token_values,
    transfer_dialogue_surface,
)
from pure_integer_ai.cognition.shared.hypothesis import (
    EPISTEMIC_SUPPORTED,
    EVIDENCE_SUPPORT,
    EvidenceRecord,
    HypothesisKey,
    LIFECYCLE_ACTIVE,
)
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    SourceRef,
    TypedRef,
    VersionBundle,
    concept_identity,
)
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.cognition.shared.semantic_graph import (
    AtomicPropositionPredicates,
    SemanticGraph,
)
from pure_integer_ai.cognition.shared.semantic_object import (
    AtomicPropositionDefinition,
    AtomicRoleBinding,
    context_scope_identity,
    proposition_identity,
    role_identity,
)
from pure_integer_ai.cognition.shared.training_hypothesis import (
    TrainingHypothesisEventSink,
    TrainingHypothesisHistoryProtocol,
)
from pure_integer_ai.cognition.shared.types import InputPayload, ObserveResult
from pure_integer_ai.crosscut.determinism.hasher import Hasher
from pure_integer_ai.experiments.collection import CollectedItem
from pure_integer_ai.experiments.train_context import TrainContext
from pure_integer_ai.storage.dialogue_successor import (
    DIALOGUE_SUCCESSOR_FEATURE_TABLE,
    DIALOGUE_SUCCESSOR_TABLE,
    DialogueSuccessorFeature,
    DialogueSuccessorProjection,
    DialogueSuccessorProjectionStore,
    FEATURE_CURRENT_TURN,
    FEATURE_HISTORY_TURN,
)


_GRAPH_DIGEST = Hasher("dialogue.successor.graph.assertions.v1")
_EVIDENCE_ID = Hasher("dialogue.successor.evidence.v1")
_FEATURE_HASH = Hasher("pure_integer_ai.concept.v1")
_SUCCESSOR_RESPONSE_CACHE_SIZE = 128
_GRAPH_DIALOGUE_MAX_RESPONSE_CODEPOINTS = 512
_GRAPH_DIALOGUE_MAX_CANDIDATES = 64
_GRAPH_DIALOGUE_MAX_HOT_CANDIDATES = 128
_GRAPH_DIALOGUE_MAX_POSTINGS_PER_FEATURE = 256
_GRAPH_DIALOGUE_MAX_EXACT_LOOKUP_CODEPOINTS = 96
_GRAPH_DIALOGUE_MAX_EXACT_CANDIDATES = 512
_GRAPH_DIALOGUE_MAX_ANCHOR_QUERIES = 12


def _integer_tokens(surface: str) -> tuple[tuple[int, ...], ...]:
    """按通用 Unicode 边界拆分输入，不携带任何语言词表或转换规则。

    空白与标点形成边界，字母/数字/标记连续运行保持为一个 token；其余
    符号按单码点保留。这样 CJK 等无空格文字仍会留下可组合的码点序列，
    而拉丁词、数字和代码标记不会被强制逐字拆散。
    """
    return integer_token_values(surface)


def _integer_token_features(tokens: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    """从 token 序列派生一至三元整数片段，长片段优先而不依赖语言。"""
    return integer_token_features(tokens)


def _integer_token_unigrams(tokens: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    """仅生成单 token 整数片段，供大规模启动倒排使用。"""
    return tuple((1, len(token), *token) for token in tokens)


def _render_integer_tokens(tokens: tuple[tuple[int, ...], ...]) -> str:
    """结果阶段仅从已选图路径的整数 token 序列重建可读表层。"""
    return "".join(chr(value) for token in tokens for value in token)


def _contains_token_sequence(
        values: tuple[tuple[int, ...], ...],
        target: tuple[tuple[int, ...], ...],
        ) -> bool:
    """按严格 token 序判断连续包含，不使用语言规则或集合近似。"""
    if not target or len(target) > len(values):
        return False
    return any(
        values[start:start + len(target)] == target
        for start in range(len(values) - len(target) + 1)
    )


@dataclass(frozen=True, slots=True)
class DialogueSuccessorProtocol:
    """注入后继 predicate、Role、图槽和 H-00 协议的整数命名空间。"""

    namespace: tuple[int, ...]
    provenance_kind: int
    version: int = 1

    def __post_init__(self) -> None:
        if (not isinstance(self.namespace, tuple) or not self.namespace
                or any(type(item) is not int or item < 0
                       for item in self.namespace)
                or type(self.provenance_kind) is not int
                or self.provenance_kind <= 0
                or type(self.version) is not int or self.version <= 0):
            raise ValueError("dialogue successor protocol 非法")

    @property
    def hypothesis_kind(self) -> tuple[int, ...]:
        return (*self.namespace, 90, self.version)


@dataclass(frozen=True, slots=True)
class DialogueSuccessorTrainingRun:
    """一次结构化课程项实际写入或精确重放的结果。"""

    proposition: TypedRef
    evidence_id: int
    current_features: int
    history_features: int
    response_codepoints: int
    replayed: bool


@dataclass(frozen=True, slots=True)
class DialogueSuccessorAnswer:
    """从核心后继图回源得到的回答及有界查询证据。"""

    surface: str
    source_hash: int
    proposition_ref: tuple[int, int]
    similarity_permille: int
    history_similarity_permille: int
    candidate_count: int
    posting_rows_read: int


# Compatibility name for older callers; the implementation is the shared
# cognition protocol so public and experiment runtimes consume one value type.
GraphDialogueTrace = DialoguePipelineTrace


def _graph_understanding_stage(
        surface: str,
        ) -> tuple[tuple[int, ...], ...]:
    """理解阶段：以通用 Unicode token 把输入投影为整数图查询单元。"""
    tokens = _integer_tokens(surface)
    if not tokens:
        raise ValueError("graph dialogue 理解阶段没有 token")
    return tokens


def _graph_result_stage(
        tokens: tuple[tuple[int, ...], ...],
        learned_surface: str,
        ) -> tuple[tuple[int, ...], str]:
    """结果阶段：核验并恢复已选 occurrence 表层，不添写任何语言。"""
    if type(learned_surface) is not str or _integer_tokens(learned_surface) != tokens:
        raise RuntimeError("dialogue successor 结果 token/occurrence 漂移")
    surface = learned_surface[:_GRAPH_DIALOGUE_MAX_RESPONSE_CODEPOINTS]
    result = _integer_tokens(surface)
    if not surface.strip():
        raise RuntimeError("dialogue successor 结果 token 为空")
    return result, surface


@dataclass(frozen=True, slots=True)
class GraphDialogueAnswer:
    """由 successor occurrence 图组合得到的自由对话结果。"""

    surface: str
    source_hash: int
    proposition_ref: tuple[int, int]
    confidence_permille: int
    trace: GraphDialogueTrace
    posting_rows_read: int


@dataclass(frozen=True, slots=True)
class _ObservedCodepoint:
    """一个输入码点及其权威 occurrence 位置。"""

    value: str
    occurrence: TypedRef
    occurrence_codepoint_ordinal: int
    turn_ordinal: int


@dataclass(frozen=True, slots=True)
class _GraphDialogueCandidate:
    """运行时从 occurrence 端点恢复的单条 successor 图路径。"""

    key: tuple[int, int]
    source_hash: int
    current_surface: str
    response_surface: str
    current_tokens: tuple[tuple[int, ...], ...]
    response_tokens: tuple[tuple[int, ...], ...]
    current_features: tuple[tuple[int, ...], ...]
    response_features: tuple[tuple[int, ...], ...]


class DialogueSuccessorTrainingRuntime:
    """把普通对话 turn 物化为 SemanticGraph 命题、Evidence 和稀疏索引。"""

    def __init__(self, ctx: TrainContext,
                 protocol: DialogueSuccessorProtocol) -> None:
        if not isinstance(ctx, TrainContext):
            raise TypeError("ctx 必须是 TrainContext")
        if not isinstance(protocol, DialogueSuccessorProtocol):
            raise TypeError("protocol 类型错误")
        if ctx.occurrence_index is None or ctx.occurrence_order_writer is None:
            raise ValueError("dialogue successor 需要 occurrence 与来源顺序设施")
        self.protocol = protocol
        namespace = protocol.namespace
        predicate_refs = tuple(ctx.graph_ontology.materialize(
            relation_concept_identity((*namespace, 10, ordinal)))
            for ordinal in range(1, 7))
        self.graph = SemanticGraph(
            ctx.graph_ontology,
            AtomicPropositionPredicates(*predicate_refs),
        )
        self.successor_predicate = concept_identity((*namespace, 20, 1))
        self.roles = tuple(
            role_identity((*namespace, 30, ordinal))
            for ordinal in range(1, 5)
        )
        aggregate_source = SourceRef(
            21402, 112, 0, GLOBAL_OWNER_SCOPE, VersionBundle())
        history_protocol = TrainingHypothesisHistoryProtocol(
            namespace=(*namespace, 80, protocol.version),
            hypothesis_kind=protocol.hypothesis_kind,
            aggregate_source=aggregate_source,
            aggregate_scope=document_scope(aggregate_source),
            allow_source_variants=True,
        )
        if ctx.training_candidate_history is None:
            raise RuntimeError("dialogue successor 缺少 Core training history")
        self.ledger = TrainingHypothesisEventSink(
            ctx.training_candidate_history,
            history_protocol,
        ).load_ledger(attach_sink=True)
        self.store = DialogueSuccessorProjectionStore(ctx.backend)

    def process(
            self,
            ctx: TrainContext,
            item: CollectedItem,
            input_payload: InputPayload,
            observation: ObserveResult,
            ) -> DialogueSuccessorTrainingRun | None:
        """只消费显式结构化 turn，并在任何写入前核验端点和支持 Evidence。"""
        if not item.dialogue_content_spans:
            return None
        if ctx.scope_owner is not None:
            return None
        if (input_payload.source_ref is None
                or input_payload.occurrence_scope_identity is None
                or input_payload.raw_text is None):
            raise ValueError("dialogue successor 输入缺少 SourceRef/scope/raw_text")
        codepoints = self._codepoints(item, input_payload, observation)
        response_span = item.dialogue_content_spans[-1]
        current_span = item.dialogue_content_spans[-2]
        response = tuple(item for item in codepoints
                         if item.turn_ordinal == response_span.turn_ordinal)
        current = tuple(item for item in codepoints
                        if item.turn_ordinal == current_span.turn_ordinal)
        history = tuple(item for item in codepoints
                        if item.turn_ordinal < current_span.turn_ordinal)
        if not response or not current:
            raise ValueError("dialogue successor 当前 turn 或回答 turn 没有 occurrence")
        source = input_payload.source_ref
        ontology = ctx.graph_ontology
        current_start = ontology.identity_of(current[0].occurrence)
        current_end = ontology.identity_of(current[-1].occurrence)
        response_start = ontology.identity_of(response[0].occurrence)
        response_end = ontology.identity_of(response[-1].occurrence)
        proposition_identity_value = proposition_identity(
            source,
            (*self.protocol.namespace, 100,
             response_span.turn_ordinal,
             *response_start.stable_key(),
             *response_end.stable_key()),
        )
        definition = AtomicPropositionDefinition(
            proposition_identity_value,
            self.successor_predicate,
            current_start,
            context_scope_identity(
                source,
                (*self.protocol.namespace, 101,
                 response_span.turn_ordinal)),
            tuple(AtomicRoleBinding(role, filler) for role, filler in zip(
                self.roles,
                (current_start, current_end, response_start, response_end),
            )),
        )
        scope = input_payload.occurrence_scope_identity
        self.graph.preflight_atomic(
            definition,
            scope=scope,
            provenance_kind=self.protocol.provenance_kind,
            content_version=self.protocol.version,
            qualifiers=(self.protocol.version,
                        current_span.turn_ordinal,
                        response_span.turn_ordinal),
        )
        hypothesis = HypothesisKey(
            self.protocol.hypothesis_kind,
            (*self.protocol.namespace, 110,
             *proposition_identity_value.stable_key()),
            (*self.protocol.namespace, 111,
             current_span.turn_ordinal),
            scope,
            source,
        )
        evidence_id = _EVIDENCE_ID.h63((
            source.stable_key(), proposition_identity_value.stable_key())) or 1
        evidence = EvidenceRecord(
            evidence_id,
            hypothesis,
            EVIDENCE_SUPPORT,
            (*self.protocol.namespace, 112, self.protocol.version),
            source,
            response_span.turn_ordinal,
            payload=(len(current), len(history), len(response)),
        )
        ledger_probe = self.ledger.clone()
        ledger_probe.register(hypothesis)
        ledger_probe.append_evidence(evidence)
        snapshot = ledger_probe.snapshot(hypothesis)
        if (snapshot.lifecycle != LIFECYCLE_ACTIVE
                or snapshot.epistemic_status != EPISTEMIC_SUPPORTED):
            raise RuntimeError("dialogue successor 候选未得到 active support")
        materialized = self.graph.define_atomic(
            definition,
            scope=scope,
            provenance_kind=self.protocol.provenance_kind,
            content_version=self.protocol.version,
            qualifiers=(self.protocol.version,
                        current_span.turn_ordinal,
                        response_span.turn_ordinal),
        )
        self.ledger.register(hypothesis)
        self.ledger.append_evidence(evidence)
        assertion_digest = _GRAPH_DIGEST.h63(
            materialized.assertion_hashes) or 1
        source_hash = ctx.occurrence_index.ensure_source(
            source, input_payload.raw_text)
        features = self._features(
            materialized.proposition, current, history,
            response_span.turn_ordinal)
        projection = DialogueSuccessorProjection(
            self.protocol.version,
            materialized.proposition.space_id,
            materialized.proposition.local_id,
            source_hash,
            current[0].occurrence.space_id,
            current[0].occurrence.local_id,
            current[-1].occurrence.space_id,
            current[-1].occurrence.local_id,
            response[0].occurrence.space_id,
            response[0].occurrence.local_id,
            response[-1].occurrence.space_id,
            response[-1].occurrence.local_id,
            len(current),
            len(history),
            len(item.dialogue_content_spans) - 1,
            response_span.turn_ordinal,
            len(materialized.assertion_hashes),
            assertion_digest,
            evidence_id,
        )
        replayed = self.store.preflight(projection, features)
        self.store.record(projection, features)
        return DialogueSuccessorTrainingRun(
            materialized.proposition,
            evidence_id,
            len(current),
            len(history),
            len(response),
            replayed,
        )

    @staticmethod
    def _codepoints(
            item: CollectedItem,
            payload: InputPayload,
            observation: ObserveResult,
            ) -> tuple[_ObservedCodepoint, ...]:
        """把 token occurrence 展开为码点 posting，保留 occurrence 内偏移。"""
        if len(payload.segments) != len(observation.segment_occurrence_refs):
            raise ValueError("dialogue successor segment/occurrence 数量漂移")
        content_by_turn = {
            span.turn_ordinal: span for span in item.dialogue_content_spans}
        result = []
        for segment, occurrences in zip(
                payload.segments, observation.segment_occurrence_refs):
            if len(segment.token_spans) != len(occurrences):
                raise ValueError("dialogue successor token/occurrence 数量漂移")
            content = content_by_turn.get(segment.dialogue_turn_ordinal)
            if content is None:
                continue
            for occurrence, (start, end) in zip(
                    occurrences, segment.token_spans):
                if start < content.start or end > content.end:
                    continue
                surface = payload.raw_text[start:end]
                for ordinal, value in enumerate(surface):
                    result.append(_ObservedCodepoint(
                        value, occurrence, ordinal,
                        segment.dialogue_turn_ordinal))
        return tuple(result)

    @staticmethod
    def _features(
            proposition: TypedRef,
            current: tuple[_ObservedCodepoint, ...],
            history: tuple[_ObservedCodepoint, ...],
            response_turn: int,
            ) -> tuple[DialogueSuccessorFeature, ...]:
        """从当前热区和更深历史生成分层 posting，不复制表层正文。"""
        result = []
        for kind, values in (
                (FEATURE_CURRENT_TURN, current),
                (FEATURE_HISTORY_TURN, history)):
            for ordinal, item in enumerate(values):
                result.append(DialogueSuccessorFeature(
                    proposition.space_id,
                    proposition.local_id,
                    kind,
                    ordinal,
                    _FEATURE_HASH.h63(item.value) or 1,
                    item.occurrence.space_id,
                    item.occurrence.local_id,
                    item.occurrence_codepoint_ordinal,
                    response_turn - item.turn_ordinal,
                ))
        return tuple(result)

    def counts(self) -> tuple[int, int]:
        """返回当前训练库内后继命题和稀疏 posting 数量。"""
        return self.store.counts()


def install_dialogue_successor_runtime(
        ctx: TrainContext,
        protocol: DialogueSuccessorProtocol,
        ) -> DialogueSuccessorTrainingRuntime:
    """把普通对话学习 runtime 安装到正式 TrainContext。"""
    runtime = DialogueSuccessorTrainingRuntime(ctx, protocol)
    ctx.dialogue_successor_runtime = runtime
    return runtime


class SqliteDialogueSuccessorRuntime:
    """从训练 SQLite 只读查询后继投影，并回源恢复回答正文。"""

    def __init__(self, database: str | Path, *, shortlist_limit: int = 32,
                 graph_dialogue: bool = False) -> None:
        path = Path(database).resolve()
        if not path.is_file():
            raise ValueError("dialogue successor database 不存在")
        if type(shortlist_limit) is not int or shortlist_limit <= 0:
            raise ValueError("shortlist_limit 必须是正整数")
        if type(graph_dialogue) is not bool:
            raise TypeError("graph_dialogue 必须是严格 bool")
        self.path = path
        self.shortlist_limit = shortlist_limit
        self.connection = sqlite3.connect(
            f"file:{path.as_posix()}?mode=ro", uri=True)
        self.connection.row_factory = sqlite3.Row
        # 只缓存已经通过 occurrence/source 原文核验的候选；缓存的是离散
        # 字符与位置，不是独立答案表。这样重复问题和相近问题不会再次把
        # 同一批 source raw_text 从 SQLite 复制到 Python。
        self._feature_cache: dict[
            tuple[int, int], tuple[tuple[int, int, str], ...]] = {}
        self._feature_cache_order: list[tuple[int, int]] = []
        self._feature_cache_limit = 128
        # Per-process decision cache.  The key retains only the six history
        # surfaces that the scoring algorithm actually consumes; values are
        # immutable answers or an explicit miss.  SQLite remains authoritative.
        self._response_cache: OrderedDict[
            tuple[object, ...], DialogueSuccessorAnswer | None] = OrderedDict()
        tables = {row[0] for row in self.connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        required = {
            DIALOGUE_SUCCESSOR_TABLE,
            DIALOGUE_SUCCESSOR_FEATURE_TABLE,
            "occurrence",
            "source_record",
        }
        if not required.issubset(tables):
            self.close()
            raise ValueError("training database 缺少 dialogue successor 表")
        self._hash_index: dict[int, dict[int, dict[tuple[int, int], int]]] = {
            FEATURE_CURRENT_TURN: {}, FEATURE_HISTORY_TURN: {}}
        self._projection_lengths: dict[tuple[int, int], tuple[int, int]] = {}
        if not graph_dialogue:
            self._load_integer_posting_index()
        # Strict graph mode keeps only projection identities and an integer
        # posting index resident.  Source text and token tuples are cold data;
        # they are page'd in only for shortlisted candidates and retained in a
        # bounded LRU cache.
        self._graph_candidate_count = 0
        self._graph_candidate_cache: OrderedDict[
            tuple[int, int], _GraphDialogueCandidate] = OrderedDict()
        self._graph_feature_index: OrderedDict[
            int, tuple[tuple[int, int], ...]] = OrderedDict()
        if graph_dialogue:
            self._graph_candidate_count = int(self.connection.execute(
                f'SELECT COUNT(*) FROM "{DIALOGUE_SUCCESSOR_TABLE}"'
            ).fetchone()[0])
            if self._graph_candidate_count <= 0:
                raise RuntimeError("dialogue successor 图没有可组合路径")

    def close(self) -> None:
        """关闭当前只读 SQLite owner。"""
        connection = getattr(self, "connection", None)
        if connection is not None:
            connection.close()
            self.connection = None

    def count(self) -> int:
        """返回模型内可查询后继命题数。"""
        row = self.connection.execute(
            f'SELECT COUNT(*) FROM "{DIALOGUE_SUCCESSOR_TABLE}"').fetchone()
        return int(row[0])

    def _load_graph_postings(
            self, feature_hashes: tuple[int, ...],
            ) -> tuple[tuple[int, int], ...]:
        """按查询从 SQLite 倒排页入少量候选，不预载全库 posting。"""
        wanted = tuple(dict.fromkeys(int(value) for value in feature_hashes))
        if not wanted:
            return ()
        # Keep one occurrence per queried feature.  The caller uses this
        # multiplicity as the first integer evidence signal; collapsing to a
        # set here made every candidate tie and could discard exact turns from
        # the bounded shortlist.
        per_feature: dict[int, tuple[tuple[int, int], ...]] = {}
        missing: list[int] = []
        for feature_hash in wanted:
            cached = self._graph_feature_index.get(feature_hash)
            if cached is None:
                missing.append(feature_hash)
            else:
                per_feature[feature_hash] = cached
                self._graph_feature_index.move_to_end(feature_hash)
        for offset in range(0, len(missing), 400):
            chunk = tuple(missing[offset:offset + 400])
            placeholders = ",".join("?" for _ in chunk)
            rows = self.connection.execute(
                f'''SELECT DISTINCT feature_hash, proposition_space_id,
                           proposition_local_id
                    FROM "{DIALOGUE_SUCCESSOR_FEATURE_TABLE}"
                    WHERE feature_kind=? AND feature_hash IN ({placeholders})
                    ORDER BY feature_hash, proposition_space_id,
                             proposition_local_id''',
                (FEATURE_CURRENT_TURN, *chunk),
            ).fetchall()
            grouped: dict[int, list[tuple[int, int]]] = {
                value: [] for value in chunk}
            for row in rows:
                feature_hash = int(row[0])
                key = (int(row[1]), int(row[2]))
                values = grouped.setdefault(feature_hash, [])
                if len(values) < _GRAPH_DIALOGUE_MAX_POSTINGS_PER_FEATURE:
                    values.append(key)
            for feature_hash in chunk:
                cached = tuple(grouped.get(feature_hash, ()))
                self._graph_feature_index[feature_hash] = cached
                self._graph_feature_index.move_to_end(feature_hash)
                per_feature[feature_hash] = cached
        while len(self._graph_feature_index) > _GRAPH_DIALOGUE_MAX_HOT_CANDIDATES:
            self._graph_feature_index.popitem(last=False)
        return tuple(
            key
            for feature_hash in wanted
            for key in per_feature.get(feature_hash, ())
        )

    def _materialize_graph_candidate(
            self, key: tuple[int, int]) -> _GraphDialogueCandidate:
        """按 proposition 键冷读 occurrence/source，并维护有界热缓存。"""
        cached = self._graph_candidate_cache.get(key)
        if cached is not None:
            self._graph_candidate_cache.move_to_end(key)
            return cached
        if (not isinstance(key, tuple) or len(key) != 2
                or any(type(item) is not int or item <= 0 for item in key)):
            raise ValueError("dialogue successor proposition 键非法")
        row = self.connection.execute(
            f'''SELECT p.source_hash,
                       cs.start, ce.end, rs.start, re.end,
                       sc.raw_text, sr.raw_text,
                       cs.source_hash, ce.source_hash,
                       rs.source_hash, re.source_hash
                FROM "{DIALOGUE_SUCCESSOR_TABLE}" AS p
                JOIN occurrence AS cs
                  ON cs.space_id=p.current_start_space_id
                 AND cs.local_id=p.current_start_local_id
                JOIN occurrence AS ce
                  ON ce.space_id=p.current_end_space_id
                 AND ce.local_id=p.current_end_local_id
                JOIN occurrence AS rs
                  ON rs.space_id=p.response_start_space_id
                 AND rs.local_id=p.response_start_local_id
                JOIN occurrence AS re
                  ON re.space_id=p.response_end_space_id
                 AND re.local_id=p.response_end_local_id
                JOIN source_record AS sc ON sc.source_hash=cs.source_hash
                JOIN source_record AS sr ON sr.source_hash=rs.source_hash
                WHERE p.proposition_space_id=? AND p.proposition_local_id=?''',
            key,
        ).fetchone()
        if row is None:
            raise RuntimeError("dialogue successor projection 端点缺失")
        source_hash = int(row[0])
        if any(source_hash != int(row[index]) for index in range(7, 11)):
            raise RuntimeError("dialogue successor occurrence 来源漂移")
        current_start, current_end = int(row[1]), int(row[2])
        response_start, response_end = int(row[3]), int(row[4])
        current_raw, response_raw = row[5], row[6]
        if (not isinstance(current_raw, str)
                or not isinstance(response_raw, str)
                or current_start < 0 or current_end <= current_start
                or current_end > len(current_raw)
                or response_start < 0 or response_end <= response_start
                or response_end > len(response_raw)):
            raise RuntimeError("dialogue successor occurrence span 越界")
        current_surface = current_raw[current_start:current_end].strip()
        response_surface = response_raw[response_start:response_end].strip()
        current_tokens = _integer_tokens(current_surface)
        response_tokens = _integer_tokens(response_surface)
        if not current_surface or not response_surface or not current_tokens \
                or not response_tokens:
            raise RuntimeError("dialogue successor 图路径表层为空")
        candidate = _GraphDialogueCandidate(
            key, source_hash, current_surface, response_surface,
            current_tokens, response_tokens,
            tuple(dict.fromkeys((
                *_integer_token_unigrams(current_tokens),
                *_integer_token_features(current_tokens),
            ))),
            _integer_token_features(response_tokens),
        )
        self._graph_candidate_cache[key] = candidate
        self._graph_candidate_cache.move_to_end(key)
        while len(self._graph_candidate_cache) > _GRAPH_DIALOGUE_MAX_HOT_CANDIDATES:
            self._graph_candidate_cache.popitem(last=False)
        return candidate

    def _exact_graph_candidate_keys(
            self,
            feature_hashes: tuple[int, ...],
            codepoint_count: int,
            ) -> tuple[tuple[int, int], ...]:
        """按当前 turn 的整数多重集寻找精确图候选。

        普通 posting 读取必须有界，但常见短句的单码点 posting 可能被前
        256 条截断。这里仅对短查询执行一次持久化聚合：候选的当前特征
        数量和每个整数 hash 的出现次数都必须与查询相同，随后仍由
        ``_materialize_graph_candidate`` 逐 occurrence 回源确认表层。
        """
        if (not feature_hashes or type(codepoint_count) is not int
                or codepoint_count <= 0
                or codepoint_count > _GRAPH_DIALOGUE_MAX_EXACT_LOOKUP_CODEPOINTS):
            return ()
        counts = Counter(int(value) for value in feature_hashes)
        hashes = tuple(counts)
        placeholders = ",".join("?" for _ in hashes)
        clauses = []
        params: list[int] = [
            FEATURE_CURRENT_TURN, codepoint_count, *hashes,
            codepoint_count, len(hashes),
        ]
        for feature_hash, count in counts.items():
            clauses.append(
                "SUM(CASE WHEN f.feature_hash=? THEN 1 ELSE 0 END)=?")
            params.extend((feature_hash, count))
        rows = self.connection.execute(
            f'''SELECT f.proposition_space_id, f.proposition_local_id
                FROM "{DIALOGUE_SUCCESSOR_FEATURE_TABLE}" AS f
                JOIN "{DIALOGUE_SUCCESSOR_TABLE}" AS p
                  ON p.proposition_space_id=f.proposition_space_id
                 AND p.proposition_local_id=f.proposition_local_id
                WHERE f.feature_kind=?
                  AND p.current_feature_count=?
                  AND f.feature_hash IN ({placeholders})
                GROUP BY f.proposition_space_id, f.proposition_local_id
                HAVING COUNT(*)=? AND COUNT(DISTINCT f.feature_hash)=?
                   AND {' AND '.join(clauses)}
                ORDER BY f.proposition_space_id, f.proposition_local_id
                LIMIT ?''',
            (*params, _GRAPH_DIALOGUE_MAX_EXACT_CANDIDATES),
        ).fetchall()
        return tuple((int(row[0]), int(row[1])) for row in rows)

    def _anchor_graph_candidate_keys(
            self,
            query_tokens: tuple[tuple[int, ...], ...],
            ) -> tuple[tuple[int, int], ...]:
        """查找覆盖当前输入至少三分之二的完整已学连续片段。

        这里只补足高频码点 posting 截断造成的不可达候选。每个候选仍须由
        occurrence 回源，并且其完整 current token 序必须等于查询中的连续片段；
        最长宽度一旦有结果就停止，避免短常用片段接管较长输入。
        """
        token_count = len(query_tokens)
        if token_count <= 1:
            return ()
        minimum_width = max(1, (token_count * 2 + 2) // 3)
        query_count = 0
        for width in range(token_count - 1, minimum_width - 1, -1):
            found: dict[tuple[int, int], None] = {}
            for start in range(token_count - width + 1):
                if query_count >= _GRAPH_DIALOGUE_MAX_ANCHOR_QUERIES:
                    break
                query_count += 1
                fragment = query_tokens[start:start + width]
                codepoint_count = sum(len(token) for token in fragment)
                if codepoint_count < 2:
                    continue
                hashes = tuple(
                    _FEATURE_HASH.h63(chr(value)) or 1
                    for token in fragment for value in token)
                for key in self._exact_graph_candidate_keys(
                        hashes, codepoint_count):
                    candidate = self._materialize_graph_candidate(key)
                    if candidate.current_tokens == fragment:
                        found[key] = None
            if found:
                return tuple(found)
            if query_count >= _GRAPH_DIALOGUE_MAX_ANCHOR_QUERIES:
                break
        return ()

    @staticmethod
    def _query_clarification(
            ranked: list[tuple[int, int, int, int, int,
                               _GraphDialogueCandidate]],
            query_tokens: tuple[tuple[int, ...], ...],
            query_features: set[tuple[int, ...]],
            ) -> _GraphDialogueCandidate | None:
        """只从与当前输入相连的回答节点选择疑问式澄清。"""
        marked = []
        for item in ranked:
            # A clarification must retain a learned multi-token relation on
            # the current turn.  A single shared codepoint is insufficient:
            # it would turn arbitrary unknown text into an unrelated answer.
            if item[2] <= 0:
                continue
            candidate = item[5]
            if (not _contains_token_sequence(
                    query_tokens, candidate.current_tokens)
                    or 3 * len(candidate.current_tokens)
                    < 2 * len(query_tokens)):
                continue
            response_shared = sum(
                1 for feature in query_features.intersection(
                    candidate.response_features)
                if feature and feature[0] > 1)
            if response_shared <= 0:
                continue
            if any("QUESTION" in unicodedata.name(value, "")
                   for value in candidate.response_surface):
                marked.append((response_shared, item))
        if not marked:
            return None
        marked.sort(key=lambda value: (
            -value[0], -value[1][2], -value[1][1], -value[1][3],
            value[1][4], value[1][5].key))
        return marked[0][1][5]

    @staticmethod
    def _response_similarity(
            left: _GraphDialogueCandidate,
            right: _GraphDialogueCandidate,
            ) -> int:
        """计算两条回答路径的长整数片段 Dice，相同表层视为满分。"""
        if left.response_surface == right.response_surface:
            return 1000
        left_features = {
            value for value in left.response_features
            if value and value[0] > 1}
        right_features = {
            value for value in right.response_features
            if value and value[0] > 1}
        denominator = len(left_features) + len(right_features)
        if denominator == 0:
            return 0
        return (2000 * len(left_features & right_features)) // denominator

    def _transfer_answer(
            self,
            query_tokens: tuple[tuple[int, ...], ...],
            current: str,
            ranked: list[tuple[int, int, int, int, int,
                               _GraphDialogueCandidate]],
            candidate_count: int,
            ) -> GraphDialogueAnswer | None:
        """把查询中的新变项代入已学 successor 结构，不复制近邻整句。"""
        transformed = []
        for row in ranked:
            if row[0] or row[1] < 500 or row[2] <= 0:
                continue
            candidate = row[5]
            transfer = transfer_dialogue_surface(
                current,
                candidate.current_surface,
                candidate.response_surface,
            )
            if transfer is None:
                continue
            confidence = min(
                950,
                max(600, row[1])
                + min(250, transfer.anchor_count * 20
                      + transfer.replacement_count * 50),
            )
            transformed.append((
                confidence,
                transfer.anchor_count,
                transfer.replacement_count,
                row[2],
                -len(transfer.surface),
                candidate,
                transfer,
            ))
        if not transformed:
            return None
        transformed.sort(key=lambda item: (
            -item[0], -item[1], -item[2], -item[3], -item[4],
            item[5].key,
        ))
        best = transformed[0]
        candidate = best[5]
        transfer = best[6]
        return GraphDialogueAnswer(
            transfer.surface,
            candidate.source_hash,
            candidate.key,
            best[0],
            GraphDialogueTrace(
                query_tokens,
                len(ranked),
                candidate.key,
                0,
                transfer.result_tokens,
                best[0],
                DIALOGUE_RESULT_TRANSFER,
                transfer.replacement_count,
                1,
            ),
            candidate_count,
        )

    def _response_class_answer(
            self,
            query_tokens: tuple[tuple[int, ...], ...],
            ranked: list[tuple[int, int, int, int, int,
                               _GraphDialogueCandidate]],
            candidate_count: int,
            ) -> GraphDialogueAnswer | None:
        """由完整输入锚点的多来源后继形成响应类，拒绝字符近邻聚类。"""
        by_current: dict[
            tuple[tuple[int, ...], ...],
            list[tuple[int, int, int, int, int, _GraphDialogueCandidate]],
        ] = {}
        for row in ranked:
            candidate = row[5]
            current_tokens = candidate.current_tokens
            if (row[0] or not current_tokens
                    or len(current_tokens) >= len(query_tokens)
                    or 3 * len(current_tokens) < 2 * len(query_tokens)
                    or not _contains_token_sequence(
                        query_tokens, current_tokens)
                    or not self._publishable_response(
                        candidate.response_surface)):
                continue
            by_current.setdefault(current_tokens, []).append(row)
        classes = []
        for current_tokens, rows in by_current.items():
            by_response: dict[str, list[_GraphDialogueCandidate]] = {}
            for row in rows:
                candidate = row[5]
                by_response.setdefault(
                    candidate.response_surface, []).append(candidate)
            for response_surface, candidates in by_response.items():
                sources = {candidate.source_hash for candidate in candidates}
                if len(sources) < 2:
                    continue
                candidate = min(candidates, key=lambda item: item.key)
                coverage = 1000 * len(current_tokens) // len(query_tokens)
                classes.append((
                    coverage,
                    len(sources),
                    len(current_tokens),
                    -len(response_surface),
                    candidate,
                ))
        if not classes:
            return None
        classes.sort(key=lambda item: (
            -item[0], -item[1], -item[2], -item[3],
            tuple(ord(value) for value in item[4].response_surface),
            item[4].key,
        ))
        selected = classes[0]
        candidate = selected[4]
        confidence = min(
            950,
            selected[0] + min(200, (selected[1] - 1) * 50),
        )
        result_tokens, surface = _graph_result_stage(
            candidate.response_tokens, candidate.response_surface)
        return GraphDialogueAnswer(
            surface,
            candidate.source_hash,
            candidate.key,
            confidence,
            GraphDialogueTrace(
                query_tokens,
                len(ranked),
                candidate.key,
                0,
                result_tokens,
                confidence,
                DIALOGUE_RESULT_RESPONSE_CLASS,
                0,
                selected[1],
            ),
            candidate_count,
        )

    @staticmethod
    def _malformed_surface(surface: str) -> bool:
        """仅把非法 Unicode 标量或替换码点视为混乱码型输入。"""
        if any(
            ord(value) == 0xFFFD
            or unicodedata.category(value) in {"Cs", "Cn"}
            for value in surface):
            return True
        # Number-letter symbols mixed with ordinary letters are code-like
        # noise for this boundary.  The rule is category based and carries
        # no language vocabulary or script table.
        categories = tuple(unicodedata.category(value) for value in surface)
        return len(surface) > 1 and "Nl" in categories and any(
            category.startswith("L") and category != "Nl"
            for category in categories)

    def _low_evidence_answer(
            self,
            query_tokens: tuple[tuple[int, ...], ...],
            ranked: list[tuple[int, int, int, int, int,
                               _GraphDialogueCandidate]],
            query_features: set[tuple[int, ...]],
            candidate_count: int,
            current: str,
            ) -> GraphDialogueAnswer | None:
        """自然语言低证据只输出已训练图澄清；混乱码型保持无结果。"""
        if self._malformed_surface(current):
            return None
        candidate = self._query_clarification(
            ranked, query_tokens, query_features)
        # A ranked list can contain ordinary response candidates without a
        # valid clarification path.  Natural-language input must still be
        # answered by a learned clarification structure in that case; tying
        # this fallback to ``not ranked`` made the public strict route raise
        # for normal prose whenever an unrelated candidate shared a feature.
        if candidate is None:
            candidate = self._structural_clarification_candidate(
                sum(len(token) for token in query_tokens),
            )
        if candidate is None:
            return None
        return GraphDialogueAnswer(
            _graph_result_stage(
                candidate.response_tokens, candidate.response_surface)[1],
            candidate.source_hash, candidate.key, 0,
            GraphDialogueTrace(query_tokens, len(ranked), candidate.key, 0,
                               candidate.response_tokens, 0,
                               DIALOGUE_RESULT_CLARIFICATION, 0, 1),
            candidate_count)

    def _structural_clarification_candidate(
            self, codepoint_count: int,
            ) -> _GraphDialogueCandidate | None:
        """从已训练 Dialogue 图按输入长度取一个澄清路径。

        完全没有共享整数片段时，不能把任意近邻句当作回答；但自然语言
        输入本身也不应暴露开发态 UNKNOWN。这里仍只从 Dialogue 图的
        occurrence 端点冷读候选，并以回答中的已学习疑问结构作最终约束。
        查询范围由训练时记录的当前输入码点数界定，结果固定排序且有界。
        """
        if (type(codepoint_count) is not int or codepoint_count <= 0
                or codepoint_count > _GRAPH_DIALOGUE_MAX_RESPONSE_CODEPOINTS):
            return None
        rows = self.connection.execute(
            f'''SELECT proposition_space_id, proposition_local_id
                FROM "{DIALOGUE_SUCCESSOR_TABLE}"
                WHERE current_feature_count>0
                ORDER BY ABS(current_feature_count-?),
                         proposition_space_id, proposition_local_id
                LIMIT ?''',
            (codepoint_count, _GRAPH_DIALOGUE_MAX_EXACT_CANDIDATES),
        ).fetchall()
        candidates = []
        for row in rows:
            candidate = self._materialize_graph_candidate(
                (int(row[0]), int(row[1])),
            )
            if any("QUESTION" in unicodedata.name(value, "")
                   for value in candidate.response_surface):
                candidates.append(candidate)
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda item: (
                len(item.response_tokens),
                tuple(ord(value) for value in item.response_surface),
                item.key,
            ),
        )

    @staticmethod
    def _publishable_response(surface: str) -> bool:
        """只拒绝空表层；代码、Markdown 和各语言文字均是可学习结构。"""
        return bool(surface.strip())

    def respond_graph(
            self,
            current: str,
            *,
            history: tuple[tuple[int, str], ...] = (),
            minimum_confidence_permille: int = 700,
            ) -> GraphDialogueAnswer | None:
        """执行输入拆分→图路径理解/过程选择→结果组合的自由对话链。

        该路径不读取旧字符 posting，也不返回随机表层。候选、回答和 trace
        均由训练 SQLite 的 occurrence/图投影恢复；无候选时返回 ``None``，
        由上层决定是否以正式协议错误结束。
        """
        if type(current) is not str or not current.strip():
            raise ValueError("current 必须是非空文本")
        if self._graph_candidate_count <= 0:
            raise RuntimeError("graph dialogue runtime 未启用")
        if (type(minimum_confidence_permille) is not int
                or not 0 <= minimum_confidence_permille <= 1000):
            raise ValueError("minimum_confidence_permille 必须是 0..1000")
        query_tokens = _graph_understanding_stage(current)
        query_features = _integer_token_features(query_tokens)
        if not query_features:
            return self._low_evidence_answer(
                query_tokens, [], set(), 0, current)
        query_features = tuple(dict.fromkeys(
            (*_integer_token_unigrams(query_tokens), *query_features)))
        # 冷索引只按持久化码点 hash 建立候选范围；真正的 token/n-gram
        # 结构与 occurrence 正文在有限 shortlist 内再核验。这样启动和
        # 常规查询都不会把全部候选表层加载到内存。
        query_hash_sequence = tuple(
            _FEATURE_HASH.h63(chr(value)) or 1
            for token in query_tokens for value in token)
        query_hashes = tuple(dict.fromkeys(query_hash_sequence))
        hit_counts: Counter[tuple[int, int]] = Counter()
        for key in self._load_graph_postings(query_hashes):
            hit_counts[key] += 1
        exact_keys = self._exact_graph_candidate_keys(
            query_hash_sequence,
            sum(len(token) for token in query_tokens),
        )
        candidate_keys = list(exact_keys)
        if not exact_keys:
            anchor_keys = self._anchor_graph_candidate_keys(query_tokens)
            candidate_keys = list(dict.fromkeys((
                *anchor_keys,
                *sorted(
                hit_counts,
                key=lambda key: (-hit_counts[key], key),
                )[:_GRAPH_DIALOGUE_MAX_HOT_CANDIDATES],
            )))
        if not candidate_keys:
            return self._low_evidence_answer(
                query_tokens, [], set(query_features), 0, current)
        history_tokens = tuple(
            token for _speaker, surface in history[-6:]
            for token in _integer_tokens(surface))
        history_features = set((*_integer_token_unigrams(history_tokens),
                                *_integer_token_features(history_tokens)))
        query_feature_set = set(query_features)
        ranked: list[tuple[int, int, int, int, int, _GraphDialogueCandidate]] = []
        # 倒排已经把范围限制在命中至少一个 token/n-gram 的候选；不能再按
        # proposition 序号截前 N 条，否则早期的“你好”等合法路径会被任意
        # 排序截掉。仅在排序结果阶段保留有限候选，保证计算和输出有界。
        for ordinal, key in enumerate(candidate_keys):
            candidate = self._materialize_graph_candidate(key)
            candidate_features = set(candidate.current_features)
            shared = len(query_feature_set & candidate_features)
            if shared <= 0:
                continue
            long_shared = sum(1 for item in query_feature_set & candidate_features
                              if item and item[0] > 1)
            history_shared = (len(history_features & set(candidate.current_features))
                              if history_features else 0)
            # 长 token 片段、当前覆盖和上下文热区均为整数证据；不使用浮点。
            exact = int(candidate.current_surface == current.strip())
            if exact:
                confidence = 1000
            else:
                # The query must cover a learned structural fragment.  A raw
                # character overlap is insufficient for a multi-token turn;
                # without a shared n-gram it remains an in-graph clarify,
                # never a guessed answer.  All scores are integer evidence.
                query_coverage = (1000 * shared
                                  // max(1, len(query_feature_set)))
                structure_bonus = min(250, long_shared * 125)
                history_bonus = min(150, history_shared * 50)
                confidence = min(
                    1000,
                    query_coverage + structure_bonus + history_bonus,
                )
                if len(query_tokens) > 1 and long_shared == 0:
                    confidence = min(confidence, 599)
            ranked.append((exact, confidence, long_shared, history_shared,
                           -ordinal, candidate))
        if not ranked:
            return self._low_evidence_answer(
                query_tokens, [], set(query_features), len(candidate_keys),
                current)
        ranked.sort(key=lambda item: (-item[0], -item[1], -item[2], -item[3], item[4]))
        publishable = tuple(item for item in ranked
                            if self._publishable_response(item[5].response_surface))
        if publishable:
            ranked = list(publishable)
        best = ranked[0]
        # 对完全相同的已学输入，按图中回答路径的重复支持做统计聚合，
        # 不以 proposition 序号随机挑一条表层。并列时以整数 token 序稳定决胜。
        if best[0]:
            exact_rows = tuple(item for item in ranked
                               if item[0] and self._publishable_response(
                                   item[5].response_surface))
            if not exact_rows:
                exact_rows = tuple(item for item in ranked if item[0])
            response_counts = Counter(item[5].response_surface
                                      for item in exact_rows)
            winner_surface, _winner_count = min(
                response_counts.items(),
                key=lambda item: (
                    -item[1], len(item[0]),
                    tuple(ord(value) for value in item[0])))
            best = min((item for item in exact_rows
                        if item[5].response_surface == winner_surface),
                       key=lambda item: item[4])
        else:
            ranked = ranked[:_GRAPH_DIALOGUE_MAX_CANDIDATES]
            best = ranked[0]
        # The current trained projection restores complete learned turns; it
        # does not yet provide a proved compositional generator.  Therefore a
        # non-exact input can only yield an in-graph clarification.  Never
        # publish a merely similar occurrence as if it answered the user's
        # sentence, even when the integer overlap score saturates.
        if not best[0]:
            transferred = self._transfer_answer(
                query_tokens, current, ranked, len(candidate_keys))
            if transferred is not None:
                return transferred
            response_class = self._response_class_answer(
                query_tokens, ranked, len(candidate_keys))
            if response_class is not None:
                return response_class
            candidate = self._query_clarification(
                ranked, query_tokens, query_feature_set)
            if candidate is None:
                return self._low_evidence_answer(
                    query_tokens, ranked, query_feature_set,
                    len(candidate_keys), current)
            return GraphDialogueAnswer(
                _graph_result_stage(
                    candidate.response_tokens, candidate.response_surface)[1],
                candidate.source_hash, candidate.key,
                0, GraphDialogueTrace(query_tokens, len(ranked), candidate.key, 0,
                                      candidate.response_tokens, 0,
                                      DIALOGUE_RESULT_CLARIFICATION, 0, 1),
                len(candidate_keys))
        if best[1] < minimum_confidence_permille:
            candidate = self._query_clarification(
                ranked, query_tokens, query_feature_set)
            if candidate is None:
                return self._low_evidence_answer(
                    query_tokens, ranked, query_feature_set,
                    len(candidate_keys), current)
            return GraphDialogueAnswer(
                _graph_result_stage(
                    candidate.response_tokens, candidate.response_surface)[1],
                candidate.source_hash, candidate.key,
                0, GraphDialogueTrace(query_tokens, len(ranked), candidate.key, 0,
                                      candidate.response_tokens, 0,
                                      DIALOGUE_RESULT_CLARIFICATION, 0, 1),
                len(candidate_keys))
        if (not best[0] and len(ranked) > 1
                and best[1] == ranked[1][1]
                and best[2:4] == ranked[1][2:4]):
            candidate = self._query_clarification(
                ranked, query_tokens, query_feature_set)
            if candidate is None:
                return self._low_evidence_answer(
                    query_tokens, ranked, query_feature_set,
                    len(candidate_keys), current)
            return GraphDialogueAnswer(
                _graph_result_stage(
                    candidate.response_tokens, candidate.response_surface)[1],
                candidate.source_hash, candidate.key,
                0, GraphDialogueTrace(query_tokens, len(ranked), candidate.key, 0,
                                      candidate.response_tokens, 0,
                                      DIALOGUE_RESULT_CLARIFICATION, 0, 1),
                len(candidate_keys))
        response_tokens = best[5].response_tokens
        # 结果阶段沿 occurrence 恢复的回答 token 组合成输出；原始字符只在
        # 该图路径已选定后用于渲染，不作为候选裁决依据。
        result_tokens, surface = _graph_result_stage(
            response_tokens, best[5].response_surface)
        return GraphDialogueAnswer(
            surface,
            best[5].source_hash,
            best[5].key,
            best[1],
            GraphDialogueTrace(
                query_tokens,
                len(ranked),
                best[5].key,
                best[0],
                result_tokens,
                best[1],
                DIALOGUE_RESULT_EXACT,
                0,
                1,
            ),
            len(candidate_keys),
        )

    def _load_integer_posting_index(self) -> None:
        """启动时把稀疏整数 posting 聚合到有界 runtime 倒排。

        SQLite 仍是唯一权威来源；该索引只是可丢弃的查询缓存。它避免每个
        用户问题重复执行 ``GROUP BY`` 和临时 B-tree，删除后可由同一数据库
        确定性重建，且只保存整数 key/count，不复制正文。
        """
        rows = self.connection.execute(
            f'''SELECT feature_kind, feature_hash,
                       proposition_space_id, proposition_local_id, COUNT(*)
                FROM "{DIALOGUE_SUCCESSOR_FEATURE_TABLE}"
                GROUP BY feature_kind, feature_hash,
                         proposition_space_id, proposition_local_id''').fetchall()
        for row in rows:
            kind = int(row[0])
            if kind not in self._hash_index:
                raise RuntimeError("dialogue successor feature kind 未注册")
            feature_hash = int(row[1])
            key = (int(row[2]), int(row[3]))
            self._hash_index[kind].setdefault(feature_hash, {})[key] = int(row[4])
        projection_rows = self.connection.execute(
            f'''SELECT proposition_space_id, proposition_local_id,
                       current_feature_count, history_feature_count
                FROM "{DIALOGUE_SUCCESSOR_TABLE}"''').fetchall()
        self._projection_lengths = {
            (int(row[0]), int(row[1])): (int(row[2]), int(row[3]))
            for row in projection_rows}

    def respond(
            self,
            current: str,
            *,
            history: tuple[tuple[int, str], ...] = (),
            minimum_similarity_permille: int = 700,
            minimum_margin_permille: int = 80,
            ) -> DialogueSuccessorAnswer | None:
        """中心热区优先查询；弱相关或近似并列候选保持 UNKNOWN。"""
        if type(current) is not str or not current.strip():
            raise ValueError("current 必须是非空文本")
        if (not isinstance(history, tuple)
                or any(not isinstance(row, tuple) or len(row) != 2
                       or type(row[0]) is not int or type(row[1]) is not str
                       for row in history)):
            raise TypeError("history 必须是 (speaker, surface) tuple")
        for value in (minimum_similarity_permille, minimum_margin_permille):
            if type(value) is not int or not 0 <= value <= 1000:
                raise ValueError("similarity/margin 必须是 0..1000 整数")
        cache_key = (
            current, history[-6:], minimum_similarity_permille,
            minimum_margin_permille)
        if cache_key in self._response_cache:
            cached = self._response_cache[cache_key]
            self._response_cache.move_to_end(cache_key)
            return cached
        current_values = tuple(current.strip())
        current_hash_counter = Counter(
            _FEATURE_HASH.h63(value) or 1 for value in current_values)
        shortlist_rows = self._shortlist_candidates(current_hash_counter)
        if not shortlist_rows:
            self._remember_response(cache_key, None)
            return None
        keys = tuple((int(row[0]), int(row[1])) for row in shortlist_rows)
        history_values = tuple(
            value for _speaker, surface in history[-6:]
            for value in surface)
        history_hash_counter = Counter(
            _FEATURE_HASH.h63(value) or 1 for value in history_values)
        # shortlist 已经由 feature_hash 索引完成；这里只聚合命中的 hash，
        # 并从 projection 读取候选总长度。不得为评分再次扫描候选全部 posting。
        candidate_lengths = self._candidate_lengths(keys)
        current_matches = self._feature_hash_counts(
            keys, FEATURE_CURRENT_TURN, tuple(current_hash_counter))
        history_matches = self._feature_hash_counts(
            keys, FEATURE_HISTORY_TURN, tuple(history_hash_counter))
        ranked = []
        for key in keys:
            current_overlap = sum(
                min(current_hash_counter[value], count)
                for value, count in current_matches.get(key, {}).items())
            history_overlap = sum(
                min(history_hash_counter[value], count)
                for value, count in history_matches.get(key, {}).items())
            current_length, history_length = candidate_lengths[key]
            current_score = self._dice_counts(
                current_overlap, len(current_values), current_length)
            history_score = (
                self._dice_counts(
                    history_overlap, len(history_values), history_length)
                if history_values and history_length else 0)
            combined = current_score * 4 + history_score
            ranked.append((combined, current_score, history_score,
                           key))
        ranked.sort(key=lambda row: (-row[0], -row[1], -row[2], row[3]))
        best = ranked[0]
        if best[1] < minimum_similarity_permille:
            self._remember_response(cache_key, None)
            return None
        verify_keys = tuple(row[3] for row in ranked[:2])
        verified, posting_rows = self._candidate_features(verify_keys)
        actual_scores: dict[tuple[int, int], tuple[int, int, tuple[str, ...]]] = {}
        current_counter = Counter(current_values)
        history_counter = Counter(history_values)
        for key in verify_keys:
            values = verified[key]
            current_candidate = tuple(
                value for kind, _ordinal, value in values
                if kind == FEATURE_CURRENT_TURN)
            history_candidate = tuple(
                value for kind, _ordinal, value in values
                if kind == FEATURE_HISTORY_TURN)
            actual_scores[key] = (
                self._dice_permille(current_counter, Counter(current_candidate)),
                self._dice_permille(history_counter, Counter(history_candidate))
                if history_values and history_candidate else 0,
                current_candidate,
            )
        verified_best = actual_scores[best[3]]
        if (verified_best[0] != best[1]
                or verified_best[1] != best[2]
                or (len(ranked) > 1
                    and actual_scores[ranked[1][3]][:2]
                    != (ranked[1][1], ranked[1][2]))):
            # 64-bit hash 碰撞或异常数据漂移时保留旧的完整裁决路径，
            # 不让优化改变来源核验后的排名。
            candidate_features, extra_rows = self._candidate_features(keys)
            posting_rows += extra_rows
            reranked = []
            for key in keys:
                current_candidate = tuple(
                    value for kind, _ordinal, value in candidate_features[key]
                    if kind == FEATURE_CURRENT_TURN)
                history_candidate = tuple(
                    value for kind, _ordinal, value in candidate_features[key]
                    if kind == FEATURE_HISTORY_TURN)
                current_score = self._dice_permille(
                    current_counter, Counter(current_candidate))
                history_score = (
                    self._dice_permille(
                        history_counter, Counter(history_candidate))
                    if history_values and history_candidate else 0)
                reranked.append((current_score * 4 + history_score,
                                 current_score, history_score,
                                 current_candidate, key))
            reranked.sort(key=lambda row: (-row[0], -row[1], -row[2], row[4]))
            best = reranked[0]
            ranked = reranked
            exact_current = best[3] == current_values
        else:
            exact_current = (verified_best[2] == current_values)
        if len(ranked) > 1:
            margin = (best[0] - ranked[1][0]) // 5
            if not exact_current and margin < minimum_margin_permille:
                self._remember_response(cache_key, None)
                return None
        best_key = best[4] if len(best) == 5 else best[3]
        surface, source_hash = self._response_surface(best_key)
        answer = DialogueSuccessorAnswer(
            surface,
            source_hash,
            best_key,
            best[1],
            best[2],
            len(keys),
            posting_rows,
        )
        self._remember_response(cache_key, answer)
        return answer

    def _remember_response(
            self, key: tuple[object, ...],
            value: DialogueSuccessorAnswer | None,
            ) -> None:
        """Store one bounded decision without changing persistent state."""
        self._response_cache[key] = value
        self._response_cache.move_to_end(key)
        while len(self._response_cache) > _SUCCESSOR_RESPONSE_CACHE_SIZE:
            self._response_cache.popitem(last=False)

    def _shortlist_candidates(
            self,
            current_hash_counter: Counter,
            ) -> tuple[sqlite3.Row, ...]:
        """用内存整数倒排完成多重集命中和候选长度读取。"""
        scores: dict[tuple[int, int], int] = {}
        index = self._hash_index[FEATURE_CURRENT_TURN]
        for feature_hash, wanted in sorted(current_hash_counter.items()):
            for key, hits in index.get(feature_hash, {}).items():
                scores[key] = scores.get(key, 0) + min(wanted, hits)
        ranked = sorted(
            ((overlap, key, *self._projection_lengths[key])
             for key, overlap in scores.items()
             if key in self._projection_lengths),
            key=lambda row: (-row[0], row[1]),
        )[:self.shortlist_limit]
        return tuple((key[0], key[1], overlap, current_length, history_length)
                     for overlap, key, current_length, history_length in ranked)

    def _candidate_lengths(
            self,
            keys: tuple[tuple[int, int], ...],
            ) -> dict[tuple[int, int], tuple[int, int]]:
        """从后继投影读取当前/历史特征总长度，不读取正文。"""
        result = {key: self._projection_lengths[key] for key in keys
                  if key in self._projection_lengths}
        if len(result) != len(keys):
            raise RuntimeError("dialogue successor projection 长度缺失")
        return result

    def _feature_hash_counts(
            self,
            keys: tuple[tuple[int, int], ...],
            feature_kind: int,
            hashes: tuple[int, ...],
            ) -> dict[tuple[int, int], Counter]:
        """只聚合输入 hash 命中的 posting，返回候选 hash 计数。"""
        if not hashes:
            return {}
        result: dict[tuple[int, int], Counter] = {}
        index = self._hash_index[feature_kind]
        wanted = set(hashes)
        for feature_hash in wanted:
            for key in keys:
                count = index.get(feature_hash, {}).get(key)
                if count is not None:
                    result.setdefault(key, Counter())[feature_hash] = count
        return result

    def _candidate_features(
            self,
            keys: tuple[tuple[int, int], ...],
            ) -> tuple[dict[tuple[int, int], tuple[tuple[int, int, str], ...]], int]:
        """回读 shortlist posting，并只对每个 occurrence 一次传输正文。

        旧实现把同一个 ``source_record.raw_text`` 随每个码点 posting 重复
        从 SQLite 复制，导致精确问题也出现数万次字符串传输。现在先回读
        posting 的 occurrence 引用，再批量回读去重后的 occurrence/source，
        最后仍逐码点核验 hash，权威数据与拒答语义不变。
        """
        result: dict[tuple[int, int], tuple[tuple[int, int, str], ...]] = {}
        missing = tuple(key for key in keys if key not in self._feature_cache)
        if not missing:
            for key in keys:
                result[key] = self._feature_cache[key]
            return result, 0
        clauses = " OR ".join(
            "(f.proposition_space_id=? AND f.proposition_local_id=?)"
            for _key in missing)
        parameters = tuple(value for key in missing for value in key)
        rows = self.connection.execute(
            f'''SELECT f.proposition_space_id, f.proposition_local_id,
                       f.feature_kind, f.feature_ordinal, f.feature_hash,
                       f.occurrence_codepoint_ordinal,
                       f.occurrence_space_id, f.occurrence_local_id
                FROM "{DIALOGUE_SUCCESSOR_FEATURE_TABLE}" AS f
                WHERE {clauses}
                ORDER BY f.proposition_space_id, f.proposition_local_id,
                         f.feature_kind, f.feature_ordinal''',
            parameters,
        ).fetchall()
        occurrences = tuple(sorted({
            (int(row[6]), int(row[7])) for row in rows}))
        if not occurrences:
            raise RuntimeError("dialogue successor shortlist 缺少 feature")
        # SQLite 对 OR 表达式有固定深度上限；固定小批量读取仍保持
        # candidate -> occurrence 的确定性映射，并避免一次性搬运大正文。
        occurrence_rows = []
        for offset in range(0, len(occurrences), 400):
            chunk = occurrences[offset:offset + 400]
            occurrence_clauses = " OR ".join(
                "(o.space_id=? AND o.local_id=?)" for _key in chunk)
            occurrence_parameters = tuple(
                value for key in chunk for value in key)
            occurrence_rows.extend(self.connection.execute(
                f'''SELECT o.space_id, o.local_id, o.source_hash,
                           o.start, o.end, s.raw_text
                    FROM occurrence AS o
                    JOIN source_record AS s ON s.source_hash=o.source_hash
                    WHERE {occurrence_clauses}''',
                occurrence_parameters,
            ).fetchall())
        occurrence_values: dict[tuple[int, int], tuple[str, int, int]] = {}
        for row in occurrence_rows:
            occurrence_values[(int(row[0]), int(row[1]))] = (
                row[5], int(row[3]), int(row[4]))
        if len(occurrence_values) != len(occurrences):
            raise RuntimeError("dialogue successor occurrence/source 缺失")
        grouped: dict[tuple[int, int], list[tuple[int, int, str]]] = {
            key: [] for key in missing}
        for row in rows:
            occurrence = (int(row[6]), int(row[7]))
            raw_text, start, end = occurrence_values[occurrence]
            surface = raw_text[start:end]
            offset = int(row[5])
            if offset < 0 or offset >= len(surface):
                raise RuntimeError("dialogue successor occurrence 码点偏移越界")
            value = surface[offset]
            if (_FEATURE_HASH.h63(value) or 1) != int(row[4]):
                raise RuntimeError("dialogue successor feature hash 与原文漂移")
            grouped[(int(row[0]), int(row[1]))].append(
                (int(row[2]), int(row[3]), value))
        for key, values in grouped.items():
            if not values:
                raise RuntimeError("dialogue successor shortlist 缺少 feature")
            cached = tuple(values)
            self._feature_cache[key] = cached
            self._feature_cache_order.append(key)
            result[key] = cached
        while len(self._feature_cache_order) > self._feature_cache_limit:
            stale = self._feature_cache_order.pop(0)
            self._feature_cache.pop(stale, None)
        for key in keys:
            result.setdefault(key, self._feature_cache[key])
        return result, len(rows) + len(occurrence_rows)

    def _response_surface(self, key: tuple[int, int]) -> tuple[str, int]:
        """由已索引 response occurrence 端点和 SourceRecord 恢复正文。"""
        row = self.connection.execute(
            f'''SELECT p.source_hash, first.start, last.end, s.raw_text,
                       first.source_hash, last.source_hash
                FROM "{DIALOGUE_SUCCESSOR_TABLE}" AS p
                JOIN occurrence AS first
                  ON first.space_id=p.response_start_space_id
                 AND first.local_id=p.response_start_local_id
                JOIN occurrence AS last
                  ON last.space_id=p.response_end_space_id
                 AND last.local_id=p.response_end_local_id
                JOIN source_record AS s ON s.source_hash=p.source_hash
                WHERE p.proposition_space_id=? AND p.proposition_local_id=?''',
            key,
        ).fetchone()
        if row is None or int(row[0]) != int(row[4]) or int(row[0]) != int(row[5]):
            raise RuntimeError("dialogue successor response 端点来源漂移")
        start, end, raw_text = int(row[1]), int(row[2]), row[3]
        if start < 0 or end <= start or end > len(raw_text):
            raise RuntimeError("dialogue successor response span 越界")
        surface = raw_text[start:end]
        if not surface.strip():
            raise RuntimeError("dialogue successor response 为空")
        return surface.strip(), int(row[0])

    @staticmethod
    def _dice_permille(left: Counter, right: Counter) -> int:
        """计算两个离散码点多重集的整数 Dice，不引入浮点状态。"""
        overlap = sum(min(count, right.get(value, 0))
                      for value, count in left.items())
        return SqliteDialogueSuccessorRuntime._dice_counts(
            overlap, sum(left.values()), sum(right.values()))

    @staticmethod
    def _dice_counts(overlap: int, left_length: int,
                     right_length: int) -> int:
        """用已聚合的整数计数计算 Dice，避免构造候选字符 Counter。"""
        denominator = left_length + right_length
        if denominator == 0:
            return 0
        return (2000 * overlap) // denominator


__all__ = [
    "DialogueSuccessorAnswer",
    "GraphDialogueAnswer",
    "GraphDialogueTrace",
    "DialogueSuccessorProtocol",
    "DialogueSuccessorTrainingRun",
    "DialogueSuccessorTrainingRuntime",
    "SqliteDialogueSuccessorRuntime",
    "install_dialogue_successor_runtime",
]
