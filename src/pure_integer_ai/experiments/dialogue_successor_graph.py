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

from pure_integer_ai.cognition.shared.graph_ontology import (
    relation_concept_identity,
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


@dataclass(frozen=True, slots=True)
class _ObservedCodepoint:
    """一个输入码点及其权威 occurrence 位置。"""

    value: str
    occurrence: TypedRef
    occurrence_codepoint_ordinal: int
    turn_ordinal: int


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

    def __init__(self, database: str | Path, *, shortlist_limit: int = 32) -> None:
        path = Path(database).resolve()
        if not path.is_file():
            raise ValueError("dialogue successor database 不存在")
        if type(shortlist_limit) is not int or shortlist_limit <= 0:
            raise ValueError("shortlist_limit 必须是正整数")
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
        self._load_integer_posting_index()

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
    "DialogueSuccessorProtocol",
    "DialogueSuccessorTrainingRun",
    "DialogueSuccessorTrainingRuntime",
    "SqliteDialogueSuccessorRuntime",
    "install_dialogue_successor_runtime",
]
