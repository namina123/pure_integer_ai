"""用生产 A-08 owner 执行 F-01 parser revision 设施场景。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.formal_artifact import ArtifactSchema
from pure_integer_ai.cognition.shared.graph_ontology import (
    relation_concept_identity,
)
from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EvidenceRecord,
    HypothesisKey,
)
from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    ParserVersion,
    SourceRef,
    VersionBundle,
    concept_identity,
    minimal_instruction_identity,
)
from pure_integer_ai.cognition.shared.memory_batch import (
    MemoryBatchRuntimeConfig,
    install_memory_batch_runtimes,
)
from pure_integer_ai.cognition.shared.memory_event import (
    EpisodePayload,
    MEMORY_EVENT_EPISODE,
    MEMORY_EVENT_USE,
    MEMORY_OBJECT_EPISODE,
    MEMORY_OBJECT_USE,
    MemoryEvent,
    MemoryLinkedRef,
    UsePayload,
    memory_object_ref,
)
from pure_integer_ai.cognition.shared.memory_overlay import MemoryAccessContext
from pure_integer_ai.cognition.shared.parser_revision import (
    ParserHypothesisRevision,
    ParserRevisionGraph,
    ParserRevisionProtocol,
    ParserRevisionRequest,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    CLOCK_MEMORY_CREATED,
    CLOCK_MEMORY_USED,
    LogicalClockIdentity,
    document_scope,
    episode_scope,
    session_scope,
)
from pure_integer_ai.cognition.understanding.memory_intake import (
    HypothesisIntakeDraft,
    ObservationIntakeDraft,
)
from pure_integer_ai.experiments.evaluation_protocol import CanonicalIdentity
from pure_integer_ai.experiments.memory_reparse_runtime import (
    MemoryParserRevisionRuntime,
)
from pure_integer_ai.experiments.post_weaning_runtime import (
    CoreCanonicalStateReader,
)
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.storage.backend import DictBackend
from pure_integer_ai.storage.edge_store import SOURCE_BARE_TEXT
from pure_integer_ai.storage.memory_batch import (
    MEMORY_BATCH_CORE_DEPENDENCY_KEY,
)
from pure_integer_ai.storage.placement import (
    TemperatureProfile,
    TemperatureTier,
)
from pure_integer_ai.storage.sealed_segment import SegmentBudget
from pure_integer_ai.storage.segment_dependency import SegmentDependency


_TEXT = "甲乙丙丁"
_LICENSE = "license-f01-reparse"
_ACCESS = MemoryAccessContext(0, 0, 0)
_PROFILE = TemperatureProfile(
    (52_800, 1),
    (
        TemperatureTier((52_800, 1), 0),
        TemperatureTier((52_800, 2), 1),
    ),
)


@dataclass(frozen=True)
class ReparseEvidence:
    """保存 A-08 首写、历史保留、Core 不变和精确重放证据。"""

    hypothesis_count: int
    preserved_use_count: int
    core_before: CanonicalIdentity
    core_after: CanonicalIdentity
    replay_idempotent: bool
    replay_before: CanonicalIdentity
    replay_after: CanonicalIdentity

    def __post_init__(self) -> None:
        """拒绝空结果、Core 漂移或并非逐字节幂等的重放。"""
        if type(self.hypothesis_count) is not int or self.hypothesis_count <= 0:
            raise ValueError("F-01 A-08 缺少新 Hypothesis")
        if (type(self.preserved_use_count) is not int
                or self.preserved_use_count <= 0):
            raise ValueError("F-01 A-08 缺少保留 Use")
        if not isinstance(self.core_before, CanonicalIdentity):
            raise TypeError("F-01 A-08 core_before 类型错误")
        if not isinstance(self.core_after, CanonicalIdentity):
            raise TypeError("F-01 A-08 core_after 类型错误")
        if self.core_before != self.core_after:
            raise ValueError("F-01 A-08 改写了 Core")
        if type(self.replay_idempotent) is not bool:
            raise TypeError("F-01 A-08 replay 标记类型错误")
        if self.replay_before != self.replay_after:
            raise ValueError("F-01 A-08 replay 改写了持久状态")


def _source(parser: int) -> SourceRef:
    """构造只有 ParserVersion 不同的两个公开来源版本。"""
    return SourceRef(
        SOURCE_BARE_TEXT,
        52_801,
        1,
        GLOBAL_OWNER_SCOPE,
        VersionBundle(parser=ParserVersion(parser)),
    )


def _batch_config() -> MemoryBatchRuntimeConfig:
    """给长期 Memory 首写注入固定 M-10 依赖和有界预算。"""
    return MemoryBatchRuntimeConfig(
        _PROFILE,
        (52_800, 1),
        SegmentDependency(
            MEMORY_BATCH_CORE_DEPENDENCY_KEY,
            (52_802, 1),
            (52_802, 2),
        ),
        SegmentBudget(16, 2_000_000),
        SegmentBudget(128, 8_000_000),
    )


def _revision_protocol() -> ParserRevisionProtocol:
    """冻结 A-03 对象种类、映射关系和映射预算。"""
    kinds = tuple(concept_identity((52_810, item)) for item in range(1, 4))
    relations = tuple(
        relation_concept_identity((52_811, item)) for item in range(1, 7)
    )
    return ParserRevisionProtocol(
        kinds[0],
        kinds[1],
        kinds[2],
        ArtifactSchema(
            concept_identity((52_812, 1)),
            concept_identity((52_812, 2)),
        ),
        *relations,
        7,
        2,
        1,
        (13, 17),
        8,
        8,
        4,
    )


def _hypothesis(source: SourceRef, candidate: int) -> HypothesisKey:
    """形成与 parser 草案一致的候选身份。"""
    return HypothesisKey(
        (52_820, 1),
        (52_821, candidate),
        (52_822, candidate),
        document_scope(source),
        source,
    )


def _context_ref(source: SourceRef) -> MemoryLinkedRef:
    """形成来源版本化的 observation 上下文。"""
    return MemoryLinkedRef.object(minimal_instruction_identity(
        (52_830, source.versions.parser.value),
        owner=source.owner,
        versions=source.versions,
    ))


def _signal_ref(source: SourceRef, candidate: int) -> MemoryLinkedRef:
    """形成候选各自独立的来源化证据信号。"""
    return MemoryLinkedRef.object(minimal_instruction_identity(
        (52_831, candidate),
        owner=source.owner,
        versions=source.versions,
    ))


class _OldParser:
    """产生覆盖唯一、拆分、合并和删除端点的五个旧候选。"""

    def __init__(self, source: SourceRef) -> None:
        self.source = source

    def parse(self, source_slice: object) -> ObservationIntakeDraft:
        """返回五个互异 lineage 的旧草案。"""
        del source_slice
        return ObservationIntakeDraft(
            (52_840, 1),
            _context_ref(self.source),
            hypotheses=tuple(
                HypothesisIntakeDraft(
                    (52_841, candidate),
                    (52_820, 1),
                    (52_821, candidate),
                    (52_822, candidate),
                    1,
                    signal_ref=_signal_ref(self.source, candidate),
                )
                for candidate in range(1, 6)
            ),
        )


class _NewParser:
    """产生一对一、拆分和合并对应的四个新候选。"""

    def __init__(self, source: SourceRef) -> None:
        self.source = source
        self.calls = 0

    def parse(self, source_slice: object) -> ObservationIntakeDraft:
        """执行一次真实 parse 并返回四个新草案。"""
        del source_slice
        self.calls += 1
        lineages = (
            (52_841, 1),
            (52_842, 6),
            (52_842, 7),
            (52_842, 8),
        )
        return ObservationIntakeDraft(
            (52_840, 2),
            _context_ref(self.source),
            hypotheses=tuple(
                HypothesisIntakeDraft(
                    lineages[index],
                    (52_820, 1),
                    (52_821, candidate),
                    (52_822, candidate),
                    1,
                    signal_ref=_signal_ref(self.source, candidate),
                )
                for index, candidate in enumerate(range(6, 10))
            ),
        )


class _RejectParser:
    """证明 exact replay 不会再次调用 parser。"""

    def parse(self, source_slice: object) -> ObservationIntakeDraft:
        """任何调用都表示 replay 没有走已提交 manifest。"""
        del source_slice
        raise RuntimeError("F-01 A-08 replay 不得重跑 parser")


def _append_use(ctx: object, source: SourceRef, old_result: object) -> object:
    """为首个旧候选追加一条带输出 Episode 的真实 Use。"""
    event_log = ctx.memory_read_events
    document = document_scope(source)
    episode = episode_scope(1, parent=document)
    session = session_scope(
        1,
        owner=source.owner,
        versions=source.versions,
    )
    output = ctx.graph_ontology.materialize(
        minimal_instruction_identity((52_850, 1)))
    created = event_log.scoped_identities.resume_clock(
        LogicalClockIdentity(episode, CLOCK_MEMORY_CREATED)).advance()
    episode_payload = EpisodePayload(
        old_result.observation_ref,
        None,
        (),
        None,
        MemoryLinkedRef.core(output),
        (old_result.hypothesis_refs[0],),
        (),
        None,
        1,
        session,
        created,
    )
    episode_ref = memory_object_ref(
        event_log.memory_space_identity,
        MEMORY_OBJECT_EPISODE,
        episode_payload.stable_key(),
        owner=source.owner,
        versions=source.versions,
    )
    event_log.append(MemoryEvent(
        MEMORY_EVENT_EPISODE,
        episode_ref,
        episode,
        episode_payload,
    ))
    used_at = event_log.scoped_identities.resume_clock(
        LogicalClockIdentity(episode, CLOCK_MEMORY_USED)).advance()
    use_payload = UsePayload(
        old_result.hypothesis_refs[0],
        episode_ref,
        MemoryLinkedRef.core(output),
        None,
        used_at,
    )
    use_ref = memory_object_ref(
        event_log.memory_space_identity,
        MEMORY_OBJECT_USE,
        use_payload.identity_key(),
        owner=source.owner,
        versions=source.versions,
    )
    event_log.append(MemoryEvent(
        MEMORY_EVENT_USE,
        use_ref,
        episode,
        use_payload,
    ))
    ctx.memory_read_aggregates.rebuild_dirty(access=_ACCESS)
    return use_ref


def run_reparse_evidence() -> ReparseEvidence:
    """运行 A-08 首写与重放，并返回直接测量的生产证据。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend, companion=True)
        install_memory_batch_runtimes(ctx, _batch_config())
        old_source = _source(1)
        new_source = _source(2)
        source_intake = ctx.memory_read_intake.source_intake
        source_intake.ensure(
            old_source, _TEXT, license_id=_LICENSE, batch_id=101)
        source_intake.ensure(
            new_source, _TEXT, license_id=_LICENSE, batch_id=102)
        old_result = ctx.memory_read_intake.ingest(
            old_source,
            _TEXT,
            license_id=_LICENSE,
            batch_id=101,
            parser=_OldParser(old_source),
        )
        use_ref = _append_use(ctx, old_source, old_result)
        old_hypotheses = tuple(
            _hypothesis(old_source, item) for item in range(1, 6))
        new_hypotheses = tuple(
            _hypothesis(new_source, item) for item in range(6, 10))
        graph = ParserRevisionGraph(ctx.graph_ontology, _revision_protocol())
        for hypothesis in (*old_hypotheses, *new_hypotheses):
            ctx.graph_ontology.materialize(hypothesis.object_identity())
        dimension = concept_identity((52_860, 1))
        reason = minimal_instruction_identity((52_861, 1))
        ctx.graph_ontology.materialize(dimension)
        ctx.graph_ontology.materialize(reason)
        replacements = (
            (new_hypotheses[0],),
            (new_hypotheses[1], new_hypotheses[2]),
            (new_hypotheses[3],),
            (new_hypotheses[3],),
            (),
        )
        request = ParserRevisionRequest(
            old_source,
            new_source,
            document_scope(old_source),
            document_scope(new_source),
            (52_862, 1),
            (),
            tuple(
                ParserHypothesisRevision(
                    hypothesis,
                    replacements[index],
                    EvidenceRecord(
                        52_870 + index,
                        hypothesis,
                        EVIDENCE_REFUTE,
                        (52_871, index),
                        new_source,
                        10 + index,
                    ),
                )
                for index, hypothesis in enumerate(old_hypotheses)
            ),
            (dimension,),
            reason,
            20,
            (52_872, 1),
        )
        graph.materialize(request)
        runtime = MemoryParserRevisionRuntime(
            ctx, graph, ctx.memory_read_intake)
        core_reader = CoreCanonicalStateReader(ctx)
        core_before = CanonicalIdentity.from_value(core_reader.read())
        parser = _NewParser(new_source)
        result = runtime.apply(
            request,
            raw_text=_TEXT,
            license_id=_LICENSE,
            batch_id=102,
            parser=parser,
        )
        core_after = CanonicalIdentity.from_value(core_reader.read())
        replay_before = CanonicalIdentity.from_value(
            backend.recovery_state_snapshot())
        replay = runtime.apply(
            request,
            raw_text=_TEXT,
            license_id=_LICENSE,
            batch_id=102,
            parser=_RejectParser(),
        )
        replay_after = CanonicalIdentity.from_value(
            backend.recovery_state_snapshot())
        return ReparseEvidence(
            len(result.new_hypothesis_refs),
            len(result.preserved_use_refs),
            core_before,
            core_after,
            (
                parser.calls == 1
                and result.replayed is False
                and result.preserved_use_refs == (use_ref,)
                and replay.replayed is True
                and replay.preserved_use_refs == (use_ref,)
                and replay_before == replay_after
            ),
            replay_before,
            replay_after,
        )
    finally:
        backend.close()


__all__ = ["ReparseEvidence", "run_reparse_evidence"]
