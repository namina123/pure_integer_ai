"""从冻结训练 SQLite 恢复可查询的 active typed relation 图。

本模块只消费训练后的 Core 图、候选历史和一等语义关系。课程、QA 数据库、
表层答案索引和语言专用转换均不在依赖边界内。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pure_integer_ai.cognition.shared.identity import ObjectIdentity
from pure_integer_ai.cognition.shared.semantic_graph import (
    MaterializedAtomicProposition,
)
from pure_integer_ai.experiments.ph2_w06_span_graph_protocol import (
    w06_span_anchor_predicate,
    w06_span_endpoint_predicate,
)
from pure_integer_ai.experiments.ph2_w06_learning import (
    W06RelationLearningRuntime,
)
from pure_integer_ai.experiments.ph2_w06_adapter import (
    w06_directionality_binding_predicate,
    w06_directionality_value,
)
from pure_integer_ai.experiments.ph2_authored_relation_schema import (
    DIRECTION_FORWARD,
    DIRECTION_SYMMETRIC,
)
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.storage.backend import SQLiteBackend
from pure_integer_ai.storage.source_record import SourceRecordRepository
from pure_integer_ai.storage.span import SpanStore


class TrainedRelationGraphError(RuntimeError):
    """训练图缺失、候选历史不闭合或只读恢复期间出现写入。"""


GRAPH_RELATION_MISS = 1
GRAPH_RELATION_ANSWER = 2
GRAPH_RELATION_CONFLICT = 3


@dataclass(frozen=True)
class ActiveRelationGraphSnapshot:
    """一次只读恢复得到的 active 命题集合及其候选身份。"""

    propositions: tuple[MaterializedAtomicProposition, ...]
    candidate_identities: tuple[ObjectIdentity, ...]


@dataclass(frozen=True)
class RelationSurfaceBinding:
    """一个 RoleBinding 在来源 Span 中的可核验表层片段。"""

    role: ObjectIdentity
    filler: ObjectIdentity
    surface: str
    source_hash: int
    start: int
    end: int


@dataclass(frozen=True)
class ActiveRelationSurface:
    """active 命题的图身份、relation cue、端点和来源表层证据。"""

    proposition: ObjectIdentity
    predicate: ObjectIdentity
    cue: str
    bindings: tuple[RelationSurfaceBinding, ...]
    evidence_surface: str
    source_hash: int
    cue_start: int
    cue_end: int


@dataclass(frozen=True)
class RelationSurfaceFrame:
    """从训练后 Span 图恢复的有序角色槽位与已学 literal 间隔。"""

    proposition: ObjectIdentity
    predicate: ObjectIdentity
    roles: tuple[ObjectIdentity, ...]
    gaps: tuple[str, ...]
    source_hash: int
    envelope_start: int
    envelope_end: int


@dataclass(frozen=True)
class GraphRelationGeneration:
    """由目标 RoleBinding 值和图内已学框架组合得到的表层。"""

    surface: str
    frame_proposition: ObjectIdentity
    frame_source_hash: int
    slot_count: int


@dataclass(frozen=True)
class GraphRelationAnswer:
    """一次由输入 cue/端点命中 active 图后得到的关系回答。"""

    surface: str
    proposition: ObjectIdentity
    predicate: ObjectIdentity
    matched_fillers: tuple[ObjectIdentity, ...]
    cue_matched: bool
    candidate_count: int
    generation: GraphRelationGeneration
    fact_reads: int
    source_hash: int


@dataclass(frozen=True)
class GraphRelationDecision:
    """区分图未命中、图回答和图内竞争的纯整数查询结果。"""

    result_code: int
    answer: GraphRelationAnswer | None

    def __post_init__(self) -> None:
        if self.result_code not in {
                GRAPH_RELATION_MISS,
                GRAPH_RELATION_ANSWER,
                GRAPH_RELATION_CONFLICT}:
            raise ValueError("relation query result_code 未注册")
        if ((self.result_code == GRAPH_RELATION_ANSWER)
                != (self.answer is not None)):
            raise ValueError("relation query result 与 answer 不闭合")


class TrainedRelationGraphRuntime:
    """拥有只读训练库句柄，并提供只从 active 图状态派生的关系查询。"""

    def __init__(self, database: str | Path) -> None:
        """只读打开一个训练 SQLite，并严格恢复 W-06 图和 H-00/H-04 历史。"""
        path = Path(database).resolve()
        if not path.is_file():
            raise ValueError("训练图 SQLite 不存在")
        self.path = path
        self.backend = SQLiteBackend(str(path), read_only=True)
        try:
            self.context = make_train_context(self.backend)
            self.owner = W06RelationLearningRuntime(
                self.backend, context=self.context)
            self._anchor_predicate = self.context.graph_ontology.resolve(
                w06_span_anchor_predicate())
            if self._anchor_predicate is None:
                raise TrainedRelationGraphError(
                    "训练图缺少 W-06 relation anchor 协议")
            self._snapshot = self._restore_active_snapshot()
            self._surface_facts = self._restore_surface_facts()
            self._surface_frames = self._restore_surface_frames()
            self._surface_start_postings = self._build_surface_start_postings()
        except BaseException:
            self.backend.close()
            raise

    def close(self) -> None:
        """关闭只读训练图 owner；重复关闭不改变模型状态。"""
        backend = getattr(self, "backend", None)
        if backend is not None:
            backend.close()
            self.backend = None

    def __enter__(self) -> "TrainedRelationGraphRuntime":
        """返回当前只读 owner。"""
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        """退出 owner 生命周期时关闭 SQLite。"""
        self.close()

    @property
    def snapshot(self) -> ActiveRelationGraphSnapshot:
        """返回启动时从权威图和候选历史恢复的不可变 active 快照。"""
        return self._snapshot

    def active_propositions(self) -> tuple[MaterializedAtomicProposition, ...]:
        """返回全部 active 原子命题，不包含 inactive 或 superseded 候选。"""
        return self._snapshot.propositions

    def active_surface_facts(self) -> tuple[ActiveRelationSurface, ...]:
        """返回从 Span/RoleBinding 图恢复的 active 关系表层视图。"""
        return self._surface_facts

    def active_surface_frames(self) -> tuple[RelationSurfaceFrame, ...]:
        """返回只从训练 SQLite 图恢复的可组合表层框架。"""
        return self._surface_frames

    def respond(self, text: str) -> GraphRelationAnswer | None:
        """兼容返回图回答；未命中或图内竞争都不猜测。"""
        return self.query(text).answer

    def query(self, text: str) -> GraphRelationDecision:
        """查询 active 图并保留未命中与明确竞争之间的路由差异。"""
        if type(text) is not str or not text.strip():
            raise ValueError("关系查询输入必须是非空文本")
        query = tuple(ord(character) for character in text)
        candidate_ordinals: set[int] = set()
        for codepoint in frozenset(query):
            candidate_ordinals.update(
                self._surface_start_postings.get(codepoint, ()))
        ranked: list[tuple[int, int, int, tuple[int, ...],
                           ActiveRelationSurface,
                           tuple[ObjectIdentity, ...], bool]] = []
        for ordinal in sorted(candidate_ordinals):
            fact = self._surface_facts[ordinal]
            cue = tuple(ord(character) for character in fact.cue)
            cue_matched = self._contains(query, cue)
            matched = tuple(
                binding.filler
                for binding in fact.bindings
                if self._contains(
                    query,
                    tuple(ord(character) for character in binding.surface),
                )
            )
            if not matched:
                continue
            matched_length = sum(
                len(binding.surface)
                for binding in fact.bindings
                if binding.filler in matched
            )
            score = len(matched) * 1_000_000 + matched_length * 1_000
            if cue_matched:
                score += len(fact.cue) * 10 + 1
            ranked.append((
                score,
                len(matched),
                matched_length,
                fact.proposition.stable_key(),
                fact,
                matched,
                cue_matched,
            ))
        if not ranked:
            return GraphRelationDecision(GRAPH_RELATION_MISS, None)
        ranked.sort(key=lambda item: (-item[0], item[3]))
        best = ranked[0]
        # 多来源可支持同一个 predicate/Role/filler 事实；这种同分不是语义
        # 冲突。只有最佳分组内出现不同关系签名时才保持无回答。
        tied = tuple(item for item in ranked if item[:3] == best[:3])
        if len({self._query_signature(item[4]) for item in tied}) > 1:
            return GraphRelationDecision(GRAPH_RELATION_CONFLICT, None)
        recognized_surfaces = {
            binding.surface
            for item in ranked
            if item[4].predicate == best[4].predicate
            for binding in item[4].bindings
            if binding.filler in item[5]
        }
        if not recognized_surfaces.issubset({
                binding.surface for binding in best[4].bindings}):
            return GraphRelationDecision(GRAPH_RELATION_CONFLICT, None)
        generation = self._generate_surface(best[4])
        return GraphRelationDecision(GRAPH_RELATION_ANSWER, GraphRelationAnswer(
            generation.surface,
            best[4].proposition,
            best[4].predicate,
            best[5],
            best[6],
            len(ranked),
            generation,
            len(candidate_ordinals),
            best[4].source_hash,
        ))

    def lookup_active_by_filler(
            self,
            filler: ObjectIdentity,
            ) -> tuple[MaterializedAtomicProposition, ...]:
        """按一个一等 filler 稀疏反查 active 命题，并完整核验 RoleBinding。"""
        if not isinstance(filler, ObjectIdentity):
            raise TypeError("filler 必须是 ObjectIdentity")
        rows = self.owner.semantic_graph.lookup_atomic_by_filler(filler)
        active = set(self._snapshot.candidate_identities)
        return tuple(
            item for item in rows
            if item.definition.proposition in active
        )

    def lookup_active_by_binding(
            self,
            predicate: ObjectIdentity,
            role: ObjectIdentity,
            filler: ObjectIdentity,
            ) -> tuple[MaterializedAtomicProposition, ...]:
        """按 predicate/Role/filler 三元约束反查 active 命题。"""
        for label, value in (
                ("predicate", predicate),
                ("role", role),
                ("filler", filler)):
            if not isinstance(value, ObjectIdentity):
                raise TypeError(f"{label} 必须是 ObjectIdentity")
        rows = self.owner.semantic_graph.lookup_atomic_by_binding(
            predicate, role, filler)
        active = set(self._snapshot.candidate_identities)
        return tuple(
            item for item in rows
            if item.definition.proposition in active
        )

    def _restore_active_snapshot(self) -> ActiveRelationGraphSnapshot:
        """联合 H-00/H-04、候选 lifecycle 图和 S-00 拓扑恢复 active 集。"""
        engine = self.owner.learning.engine
        active_state = self.owner.projection_protocol.active_state
        ontology = self.context.graph_ontology
        anchored = frozenset(
            ontology.identity_of(statement.object)
            for statement in ontology.statements(
                predicate=self._anchor_predicate)
        )
        if not anchored:
            raise TrainedRelationGraphError("训练图没有 W-06 relation anchor")
        propositions: list[MaterializedAtomicProposition] = []
        identities: list[ObjectIdentity] = []
        directionalities: list[int] = []
        direction_predicate = w06_directionality_binding_predicate()
        direction_values = {
            w06_directionality_value(DIRECTION_FORWARD): DIRECTION_FORWARD,
            w06_directionality_value(DIRECTION_SYMMETRIC): DIRECTION_SYMMETRIC,
        }
        for hypothesis in engine.ledger.hypotheses():
            definition = engine.definition(hypothesis)
            if definition.candidate not in anchored:
                continue
            candidate = self.context.graph_ontology.resolve(definition.candidate)
            if candidate is None:
                raise TrainedRelationGraphError("候选历史引用了未物化图对象")
            history = self.owner.candidate_graph.history(candidate)
            if not history:
                # Forming 只建立 H-00 候选；在独立 Evidence 到来前不存在
                # H-05 消费投影，因此既不是 active，也不是损坏的事件链。
                continue
            projection = self.owner.candidate_graph.project(candidate)
            if projection.state != active_state:
                continue
            atomic = self.owner.semantic_graph.read_atomic(candidate)
            if atomic.definition.proposition != definition.candidate:
                raise TrainedRelationGraphError("active 候选与语义命题身份漂移")
            direction = tuple(
                binding.value for binding in definition.bindings
                if binding.predicate == direction_predicate)
            if len(direction) != 1 or direction[0] not in direction_values:
                raise TrainedRelationGraphError("active relation 方向性字段不闭合")
            identities.append(definition.candidate)
            propositions.append(atomic)
            directionalities.append(direction_values[direction[0]])
        ordered = sorted(
            zip(identities, propositions, directionalities, strict=True),
            key=lambda item: item[0].stable_key(),
        )
        if not ordered:
            raise TrainedRelationGraphError("训练图没有可消费的 active typed relation")
        self._directionality_by_proposition = {
            item[0]: item[2] for item in ordered
        }
        return ActiveRelationGraphSnapshot(
            tuple(item[1] for item in ordered),
            tuple(item[0] for item in ordered),
        )

    def _restore_surface_facts(self) -> tuple[ActiveRelationSurface, ...]:
        """从 Span statement 和 SourceRecord 恢复命题 cue/端点，不读取课程。"""
        ontology = self.context.graph_ontology
        endpoint_predicate = ontology.resolve(w06_span_endpoint_predicate())
        if endpoint_predicate is None:
            raise TrainedRelationGraphError("训练图缺少 W-06 relation Span 协议")
        spans = SpanStore(self.backend)
        sources = SourceRecordRepository(
            self.backend,
            registry=self.context.scoped_identity_store.registry,
        )
        facts: list[ActiveRelationSurface] = []
        for atomic in self._snapshot.propositions:
            anchor_links = ontology.statements(
                predicate=self._anchor_predicate,
                object_ref=atomic.proposition,
            )
            if len(anchor_links) != 1:
                raise TrainedRelationGraphError(
                    "active 命题没有唯一 relation anchor Span")
            cue, source_hash, cue_start, cue_end, evidence = self._span_surface(
                anchor_links[0].subject, spans, sources)
            bindings: list[RelationSurfaceBinding] = []
            for binding in atomic.bindings:
                endpoint_links = ontology.statements(
                    predicate=endpoint_predicate,
                    object_ref=binding.ref,
                )
                if len(endpoint_links) != 1:
                    raise TrainedRelationGraphError(
                        "active RoleBinding 没有唯一 endpoint Span")
                surface, endpoint_source, start, end, endpoint_evidence = (
                    self._span_surface(
                        endpoint_links[0].subject, spans, sources))
                if endpoint_source != source_hash or endpoint_evidence != evidence:
                    raise TrainedRelationGraphError(
                        "relation cue 与 endpoint 没有共同来源")
                bindings.append(RelationSurfaceBinding(
                    ontology.identity_of(binding.role),
                    ontology.identity_of(binding.filler),
                    surface,
                    endpoint_source,
                    start,
                    end,
                ))
            facts.append(ActiveRelationSurface(
                atomic.definition.proposition,
                atomic.definition.predicate,
                cue,
                tuple(sorted(bindings, key=lambda item: (
                    item.start, item.end, item.role.stable_key()))),
                evidence,
                source_hash,
                cue_start,
                cue_end,
            ))
        return tuple(sorted(
            facts, key=lambda item: item.proposition.stable_key()))

    def _restore_surface_frames(self) -> tuple[RelationSurfaceFrame, ...]:
        """把图内 endpoint Span 投影为槽位框架；不读取训练课程。"""
        frames = []
        for fact in self._surface_facts:
            bindings = tuple(sorted(
                fact.bindings,
                key=lambda item: (
                    item.start, item.end, item.role.stable_key()),
            ))
            if not bindings:
                raise TrainedRelationGraphError("active relation 没有可生成角色")
            envelope_start = min(
                fact.cue_start, *(item.start for item in bindings))
            envelope_end = max(
                fact.cue_end, *(item.end for item in bindings))
            cursor = envelope_start
            gaps = []
            roles = []
            for binding in bindings:
                if (binding.start < cursor
                        or binding.end > envelope_end
                        or envelope_end > len(fact.evidence_surface)):
                    raise TrainedRelationGraphError("relation endpoint Span 重叠或越界")
                if fact.evidence_surface[binding.start:binding.end] != binding.surface:
                    raise TrainedRelationGraphError(
                        "非连续 endpoint Span 暂不能形成确定性表层框架")
                gaps.append(fact.evidence_surface[cursor:binding.start])
                roles.append(binding.role)
                cursor = binding.end
            gaps.append(fact.evidence_surface[cursor:envelope_end])
            if not (0 <= fact.cue_start < fact.cue_end
                    <= len(fact.evidence_surface)
                    and fact.evidence_surface[
                        fact.cue_start:fact.cue_end] == fact.cue):
                raise TrainedRelationGraphError("relation cue Span 不能核验表层")
            frames.append(RelationSurfaceFrame(
                fact.proposition,
                fact.predicate,
                tuple(roles),
                tuple(gaps),
                fact.source_hash,
                envelope_start,
                envelope_end,
            ))
        return tuple(sorted(
            frames, key=lambda item: item.proposition.stable_key()))

    def _build_surface_start_postings(self) -> dict[int, tuple[int, ...]]:
        """按 cue/endpoint 首码点建立可重建稀疏候选索引。"""
        postings: dict[int, set[int]] = {}
        for ordinal, fact in enumerate(self._surface_facts):
            surfaces = (fact.cue, *(item.surface for item in fact.bindings))
            for surface in surfaces:
                values = tuple(ord(character) for character in surface)
                if not values:
                    raise TrainedRelationGraphError("关系表层索引遇到空码点序列")
                postings.setdefault(values[0], set()).add(ordinal)
        return {
            codepoint: tuple(sorted(ordinals))
            for codepoint, ordinals in postings.items()
        }

    def _generate_surface(
            self, fact: ActiveRelationSurface,
            ) -> GraphRelationGeneration:
        """以同 predicate/role 的已学框架重填目标图 RoleBinding。"""
        by_role: dict[ObjectIdentity, str] = {}
        for binding in fact.bindings:
            prior = by_role.get(binding.role)
            if prior is not None and prior != binding.surface:
                raise TrainedRelationGraphError("同一 relation role 存在竞争表层")
            by_role[binding.role] = binding.surface
        compatible = tuple(
            frame for frame in self._surface_frames
            if frame.predicate == fact.predicate
            and len(frame.roles) == len(by_role)
            and set(frame.roles) == set(by_role)
            and len(set(frame.roles)) == len(frame.roles)
        )
        if not compatible:
            raise TrainedRelationGraphError("active relation 缺少兼容生成框架")
        # 相同关系若学到多个表层结构，优先使用另一命题的框架，明确执行
        # RoleBinding 重填而非复制当前来源句；只有单框架时才使用自身框架。
        ordered = tuple(sorted(
            compatible,
            key=lambda item: (
                item.proposition == fact.proposition,
                item.proposition.stable_key()),
        ))
        frame = ordered[0]
        pieces = [frame.gaps[0]]
        for index, role in enumerate(frame.roles):
            pieces.append(by_role[role])
            pieces.append(frame.gaps[index + 1])
        surface = "".join(pieces)
        if not surface.strip():
            raise TrainedRelationGraphError("relation 图生成了空表层")
        return GraphRelationGeneration(
            surface,
            frame.proposition,
            frame.source_hash,
            len(frame.roles),
        )

    @staticmethod
    def _span_surface(span, spans: SpanStore,
                      sources: SourceRecordRepository
                      ) -> tuple[str, int, int, int, str]:
        """完整核验一个 Span，并返回片段、来源 hash、包络和来源表层。"""
        record, members = spans.read(span.space_id, span.local_id)
        source = sources.read(record.source_hash)
        if any(end > len(source.raw_text) for _start, end in members):
            raise TrainedRelationGraphError("Span member 越出来源表层")
        surface = "".join(
            source.raw_text[start:end] for start, end in members)
        if not surface:
            raise TrainedRelationGraphError("relation Span 表层为空")
        return (
            surface,
            record.source_hash,
            record.envelope_start,
            record.envelope_end,
            source.raw_text,
        )

    @staticmethod
    def _contains(haystack: tuple[int, ...], needle: tuple[int, ...]) -> bool:
        """按严格码点序列执行连续包含查询，不使用语言或 Unicode 隐式规则。"""
        if not needle or len(needle) > len(haystack):
            return False
        stop = len(haystack) - len(needle) + 1
        return any(
            haystack[offset:offset + len(needle)] == needle
            for offset in range(stop)
        )

    def _query_signature(self, fact: ActiveRelationSurface) -> tuple[int, ...]:
        """按图内方向性和查询可见表层区分真正竞争的关系解释。"""
        directionality = self._directionality_by_proposition.get(
            fact.proposition)
        if directionality not in {DIRECTION_FORWARD, DIRECTION_SYMMETRIC}:
            raise TrainedRelationGraphError("active relation 缺少已训练方向性")
        values = [
            *fact.predicate.stable_key(), directionality, len(fact.bindings)]
        if directionality == DIRECTION_SYMMETRIC:
            rows = tuple(sorted(
                tuple(ord(character) for character in binding.surface)
                for binding in fact.bindings
            ))
            for surface in rows:
                values.extend((len(surface), *surface))
            return tuple(values)
        for binding in sorted(
                fact.bindings, key=lambda item: item.role.stable_key()):
            role = binding.role.stable_key()
            surface = tuple(ord(character) for character in binding.surface)
            values.extend((len(role), *role, len(surface), *surface))
        return tuple(values)


__all__ = [
    "ActiveRelationGraphSnapshot",
    "ActiveRelationSurface",
    "GRAPH_RELATION_ANSWER",
    "GRAPH_RELATION_CONFLICT",
    "GRAPH_RELATION_MISS",
    "GraphRelationGeneration",
    "GraphRelationAnswer",
    "GraphRelationDecision",
    "RelationSurfaceFrame",
    "RelationSurfaceBinding",
    "TrainedRelationGraphError",
    "TrainedRelationGraphRuntime",
]
