"""M-05 来源摄入、失败隔离、parser 替代和通道 ACL 对抗。"""
from __future__ import annotations

import pytest

from pure_integer_ai.cognition.shared.identity import (
    CorpusVersion,
    CurriculumVersion,
    OBJECT_CONTEXT_SCOPE,
    OBJECT_MINIMAL_INSTRUCTION,
    ObjectIdentity,
    OwnerScope,
    ParserVersion,
    PrimitiveVersion,
    SourceRef,
    VersionBundle,
    VISIBILITY_GLOBAL,
    VISIBILITY_SESSION,
)
from pure_integer_ai.cognition.shared.memory_event import (
    INTAKE_OUTCOME_FAILURE,
    INTAKE_OUTCOME_SUCCESS,
    MEMORY_EVENT_DERIVATION,
    MEMORY_EVENT_EVIDENCE,
    MEMORY_EVENT_HYPOTHESIS,
    MEMORY_EVENT_INTAKE_MANIFEST,
    MEMORY_EVENT_OBSERVATION,
    MEMORY_EVENT_PARSE_FAILURE,
    MEMORY_EVENT_RETENTION,
    MemoryLinkedRef,
    LIFECYCLE_ACTIVE,
)
from pure_integer_ai.cognition.shared.memory_aggregate import (
    MEMORY_EVIDENCE_PROVISIONAL,
)
from pure_integer_ai.cognition.shared.memory_overlay import MemoryAccessContext
from pure_integer_ai.cognition.understanding.memory_intake import (
    HypothesisIntakeDraft,
    MemoryIntakeIntegrityError,
    MemorySourceIntake,
    ObservationIntakeDraft,
    ParseFailureDraft,
    interaction_intake_policy,
    reading_intake_policy,
)
from pure_integer_ai.cognition.understanding.source_intake import SourceIntake
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.experiments.evaluation_isolation import (
    clone_backend,
    clone_train_context,
)
from pure_integer_ai.storage.backend import DictBackend, SQLiteBackend
from pure_integer_ai.storage.source_record import SourceRecordRepository
from pure_integer_ai.storage.spaces.companion import CompanionSpace
from pure_integer_ai.storage.spaces.registry import SpaceRegistry


def _source(parser: int, *, owner: OwnerScope | None = None) -> SourceRef:
    """构造同一文档的显式 parser 版本来源。"""
    return SourceRef(
        71, 9, 0, owner or OwnerScope(),
        VersionBundle(
            CorpusVersion(3), ParserVersion(parser),
            PrimitiveVersion(2), CurriculumVersion(4),
        ),
    )


def _intake(ctx, *, interaction: bool = False) -> MemorySourceIntake:
    """在测试上下文中装配独立 SourceIntake 和指定 Memory 通道。"""
    registry = SpaceRegistry(ctx.backend)
    companion = CompanionSpace.create(registry, "companion")
    repository = SourceRecordRepository(
        ctx.backend, registry=ctx.scoped_identity_store.registry)
    source_intake = SourceIntake(repository, companion)
    return MemorySourceIntake(
        source_intake,
        ctx.memory_interact_events if interaction
        else ctx.memory_read_events,
        interaction_intake_policy() if interaction
        else reading_intake_policy(),
    )


def _context_ref(source: SourceRef) -> MemoryLinkedRef:
    """构造不触发 Core 写入的非编址上下文引用。"""
    return MemoryLinkedRef.object(ObjectIdentity(
        OBJECT_CONTEXT_SCOPE, (601,), source.owner, source.versions))


def _signal(source: SourceRef) -> MemoryLinkedRef:
    """构造由调用方注入的最小失败/证据类型引用。"""
    return MemoryLinkedRef.object(ObjectIdentity(
        OBJECT_MINIMAL_INSTRUCTION, (602,), source.owner, source.versions))


class _SuccessParser:
    """返回一个候选的纯解析器，并记录是否在 SourceRecord 后才被调用。"""

    def __init__(self, repository, source):
        self.repository = repository
        self.source = source
        self.seen_source = False
        self.calls = 0

    def parse(self, source_slice):
        """只读取切片并返回草案，不访问图和 Memory。"""
        self.calls += 1
        self.seen_source = self.repository.find(self.source.stable_key()) is not None
        return ObservationIntakeDraft(
            (701,),
            _context_ref(self.source),
            hypotheses=(HypothesisIntakeDraft(
                (702,), (703,), (704,), (705,), 1,
                signal_ref=_signal(self.source),
            ),),
        )


@pytest.mark.parametrize("backend_type", [DictBackend, SQLiteBackend])
def test_success_is_source_first_idempotent_and_not_consolidated(backend_type):
    """成功摄入先落来源，重复调用不重解析且不把文档频次当巩固。"""
    backend = backend_type()
    try:
        ctx = make_train_context(backend, companion=True)
        source = _source(1)
        registry = SpaceRegistry(backend)
        companion = CompanionSpace.create(registry, "companion")
        repository = SourceRecordRepository(
            backend, registry=ctx.scoped_identity_store.registry)
        intake = MemorySourceIntake(
            SourceIntake(repository, companion),
            ctx.memory_read_events,
            reading_intake_policy(),
        )
        parser = _SuccessParser(repository, source)
        graph_object_count = len(backend.select("graph_object", where=None))
        materialize_calls = 0

        def materialize(draft):
            """核验 Core materialize 位于纯解析后、首个 Memory event 前。"""
            nonlocal materialize_calls
            materialize_calls += 1
            assert parser.calls == 1 and parser.seen_source
            assert ctx.memory_read_events.query(
                access=MemoryAccessContext(0, 0, 0)) == ()
            return draft

        first = intake.ingest(
            source, "来源文本", license_id="license-a", batch_id=11,
            parser=parser, materialize=materialize)
        second = intake.ingest(
            source, "来源文本", license_id="license-a", batch_id=11,
            parser=parser, materialize=materialize)
        assert first.outcome_kind == INTAKE_OUTCOME_SUCCESS
        assert second == first
        assert parser.calls == 1 and parser.seen_source
        assert materialize_calls == 1
        assert repository.source_count() == 1
        assert len(backend.select("graph_object", where=None)) == graph_object_count
        assert first.source_record.batch_id == 11
        assert first.source_record.parser_version == 1
        assert first.hypothesis_refs[0].versions.parser.value == 1
        access = MemoryAccessContext(0, 0, 0)
        events = ctx.memory_read_events.query(access=access)
        assert [item.event.event_kind for item in events].count(
            MEMORY_EVENT_OBSERVATION) == 1
        assert [item.event.event_kind for item in events].count(
            MEMORY_EVENT_HYPOTHESIS) == 1
        assert [item.event.event_kind for item in events].count(
            MEMORY_EVENT_EVIDENCE) == 1
        assert [item.event.event_kind for item in events].count(
            MEMORY_EVENT_INTAKE_MANIFEST) == 1
        assert MEMORY_EVENT_RETENTION not in {
            item.event.event_kind for item in events}
        aggregate = ctx.memory_read_aggregates.rebuild_all(access=access)
        assert aggregate.processed_hypothesis_count == 1
        record = ctx.memory_read_aggregates.read(
            first.hypothesis_refs[0], access=access)
        assert record is not None
        assert record.support_count == 1
        assert record.independent_source_count == 1
        assert record.use_count == 0
        assert record.evidence_state == MEMORY_EVIDENCE_PROVISIONAL
    finally:
        backend.close()


def test_parse_failure_keeps_source_and_writes_no_observation():
    """结构化解析失败只写 SourceRecord、失败事件和 manifest。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend, companion=True)
        source = _source(1)
        intake = _intake(ctx)

        class Failed:
            """返回注入式失败类别的 parser。"""

            def parse(self, source_slice):
                """失败发生在任何 Core materialize 之前。"""
                return ParseFailureDraft(
                    (801,), _signal(source), (802, 803))

        result = intake.ingest(
            source, "坏解析", license_id="license-b", batch_id=12,
            parser=Failed(),
            materialize=lambda draft: pytest.fail("失败解析不得调用 Core materialize"))
        assert result.outcome_kind == INTAKE_OUTCOME_FAILURE
        assert result.failure_ref is not None
        access = MemoryAccessContext(0, 0, 0)
        events = ctx.memory_read_events.query(access=access)
        kinds = [item.event.event_kind for item in events]
        assert MEMORY_EVENT_PARSE_FAILURE in kinds
        assert MEMORY_EVENT_INTAKE_MANIFEST in kinds
        assert MEMORY_EVENT_OBSERVATION not in kinds
        assert MEMORY_EVENT_HYPOTHESIS not in kinds
        assert MEMORY_EVENT_EVIDENCE not in kinds
    finally:
        backend.close()


def test_parser_exception_needs_injected_failure_classifier():
    """宿主不解释异常文字；调用方分类后才写结构化失败事件。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend, companion=True)
        source = _source(1)
        intake = _intake(ctx)

        class Raises:
            """模拟 parser 内部抛出不带领域类型的异常。"""

            def parse(self, source_slice):
                """抛出异常，领域类别留给注入 classifier。"""
                raise ValueError("自由文本不得进入事件")

        result = intake.ingest(
            source, "异常解析", license_id="license-e", batch_id=17,
            parser=Raises(),
            failure_classifier=lambda exc: ParseFailureDraft(
                (811,), _signal(source), (812,)),
        )
        assert result.outcome_kind == INTAKE_OUTCOME_FAILURE
        assert result.failure_ref is not None
    finally:
        backend.close()


@pytest.mark.parametrize("backend_type", [DictBackend, SQLiteBackend])
def test_reparse_reuses_companion_and_supersedes_old_derived_objects(
        backend_type):
    """更高 parser 版本生成新对象，旧 Observation/Hypothesis/Evidence 全部退出。"""
    backend = backend_type()
    try:
        ctx = make_train_context(backend, companion=True)
        intake = _intake(ctx)
        old = _source(1)
        new = _source(2)

        class Parser:
            """按 parser 版本产生不同候选身份但共用 lineage。"""

            def __init__(self, source, candidate):
                self.source = source
                self.candidate = candidate

            def parse(self, source_slice):
                """返回同一 lineage 的新 parser 结果。"""
                return ObservationIntakeDraft(
                    (901,), _context_ref(self.source), hypotheses=(
                        HypothesisIntakeDraft(
                            (902,), (903,), (self.candidate,), (905,), 1,
                            signal_ref=_signal(self.source),
                        ),
                    ))

        first = intake.ingest(
            old, "同一来源", license_id="license-c", batch_id=13,
            parser=Parser(old, 1))
        second = intake.ingest(
            new, "同一来源", license_id="license-c", batch_id=14,
            parser=Parser(new, 2), supersedes_source=old)
        repeated = intake.ingest(
            new, "同一来源", license_id="license-c", batch_id=14,
            parser=Parser(new, 999), supersedes_source=old)
        assert repeated == second
        assert len(second.superseded_refs) == 4
        assert first.source_record.companion_assoc_id == (
            second.source_record.companion_assoc_id)
        access = MemoryAccessContext(0, 0, 0)
        derivations = ctx.memory_read_events.query(
            access=access, event_kind=MEMORY_EVENT_DERIVATION)
        assert len(derivations) == 4
        old_aggregate = ctx.memory_read_aggregates.rebuild_all(access=access)
        assert old_aggregate.processed_hypothesis_count == 2
        old_record = ctx.memory_read_aggregates.read(
            first.hypothesis_refs[0], access=access)
        new_record = ctx.memory_read_aggregates.read(
            second.hypothesis_refs[0], access=access)
        assert old_record is not None
        assert old_record.lifecycle_state != LIFECYCLE_ACTIVE
        assert old_record.support_count == 0
        assert new_record is not None and new_record.support_count == 1
    finally:
        backend.close()


def test_reparse_requires_explicit_current_manifest_predecessor():
    """重解析拒绝省略、跳过或在幂等重放中漂移当前 manifest 前驱。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend, companion=True)
        intake = _intake(ctx)
        first_source = _source(1)
        second_source = _source(2)
        third_source = _source(3)

        first = intake.ingest(
            first_source, "谱系来源", license_id="license-lineage",
            batch_id=21,
            parser=_SuccessParser(
                SourceRecordRepository(
                    backend, registry=ctx.scoped_identity_store.registry),
                first_source,
            ),
        )
        access = MemoryAccessContext(0, 0, 0)
        first_event_count = len(ctx.memory_read_events.query(access=access))

        with pytest.raises(
                MemoryIntakeIntegrityError,
                match="重新解析必须显式声明"):
            intake.ingest(
                second_source, "谱系来源", license_id="license-lineage",
                batch_id=22,
                parser=_SuccessParser(
                    SourceRecordRepository(
                        backend,
                        registry=ctx.scoped_identity_store.registry),
                    second_source,
                ),
            )
        assert len(ctx.memory_read_events.query(access=access)) == first_event_count

        second = intake.ingest(
            second_source, "谱系来源", license_id="license-lineage",
            batch_id=22,
            parser=_SuccessParser(
                SourceRecordRepository(
                    backend, registry=ctx.scoped_identity_store.registry),
                second_source,
            ),
            supersedes_source=first_source,
        )
        with pytest.raises(
                MemoryIntakeIntegrityError,
                match="不是同谱系当前活跃"):
            intake.ingest(
                third_source, "谱系来源", license_id="license-lineage",
                batch_id=23,
                parser=_SuccessParser(
                    SourceRecordRepository(
                        backend,
                        registry=ctx.scoped_identity_store.registry),
                    third_source,
                ),
                supersedes_source=first_source,
            )

        third = intake.ingest(
            third_source, "谱系来源", license_id="license-lineage",
            batch_id=23,
            parser=_SuccessParser(
                SourceRecordRepository(
                    backend, registry=ctx.scoped_identity_store.registry),
                third_source,
            ),
            supersedes_source=second_source,
        )
        assert first.manifest_ref != second.manifest_ref != third.manifest_ref
        with pytest.raises(
                MemoryIntakeIntegrityError,
                match="既有 manifest 漂移"):
            intake.ingest(
                third_source, "谱系来源", license_id="license-lineage",
                batch_id=23,
                parser=_SuccessParser(
                    SourceRecordRepository(
                        backend,
                        registry=ctx.scoped_identity_store.registry),
                    third_source,
                ),
                supersedes_source=first_source,
            )
    finally:
        backend.close()


def test_reading_and_interaction_use_different_visibility_policies():
    """阅读层拒绝 session，交互层拒绝 global，协议本体保持相同。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend, companion=True)
        global_source = _source(1)
        session_source = _source(
            1, owner=OwnerScope(1, 2, 3, VISIBILITY_SESSION))
        parser = _SuccessParser(None, global_source)
        with pytest.raises(MemoryIntakeIntegrityError):
            _intake(ctx).ingest(
                session_source, "交互", license_id="license-d", batch_id=15,
                parser=parser)
        with pytest.raises(MemoryIntakeIntegrityError):
            _intake(ctx, interaction=True).ingest(
                global_source, "阅读", license_id="license-d", batch_id=16,
                parser=parser)

        class SessionParser:
            """为合法交互来源返回与阅读层同型的 Observation 草案。"""

            def parse(self, source_slice):
                """只改变来源 owner，不改变共享摄入协议。"""
                return ObservationIntakeDraft(
                    (951,), _context_ref(session_source))

        result = _intake(ctx, interaction=True).ingest(
            session_source, "交互", license_id="license-d", batch_id=16,
            parser=SessionParser())
        assert result.outcome_kind == INTAKE_OUTCOME_SUCCESS
        assert ctx.memory_interact_events.query(
            access=MemoryAccessContext(1, 2, 3),
            event_kind=MEMORY_EVENT_OBSERVATION,
        )
    finally:
        backend.close()


def test_train_context_and_v06_install_independent_intake_protocols():
    """Companion 上下文默认装配 M-05，V-06 在独立后端重建 writer。"""
    backend = DictBackend()
    cloned_backend = None
    try:
        ctx = make_train_context(backend, companion=True)
        assert ctx.memory_read_intake is not None
        assert ctx.memory_interact_intake is not None
        source = _source(1)

        class Parser:
            """为 clone 恢复测试写入一个最小成功 manifest。"""

            def parse(self, source_slice):
                """返回不含候选的 Observation 草案。"""
                return ObservationIntakeDraft(
                    (1001,), _context_ref(source))

        host_result = ctx.memory_read_intake.ingest(
            source, "clone 来源", license_id="license-f", batch_id=18,
            parser=Parser())
        host_event_count = len(ctx.memory_read_events.query(
            access=MemoryAccessContext(0, 0, 0)))
        cloned_backend = clone_backend(backend)
        cloned = clone_train_context(ctx, cloned_backend, label="m05-clone")
        assert cloned.memory_read_intake is not None
        assert cloned.memory_interact_intake is not None
        assert cloned.memory_read_intake is not ctx.memory_read_intake
        assert cloned.memory_read_intake.event_log.backend is cloned_backend
        assert ctx.memory_read_intake.event_log.backend is backend

        class FailParser:
            """若 clone 未恢复 manifest，则立即使测试失败。"""

            def parse(self, source_slice):
                """禁止 clone 对已有来源重新解析。"""
                pytest.fail("clone 应从 manifest 恢复而非重解析")

        clone_result = cloned.memory_read_intake.ingest(
            source, "clone 来源", license_id="license-f", batch_id=18,
            parser=FailParser())
        assert clone_result == host_result
        assert len(ctx.memory_read_events.query(
            access=MemoryAccessContext(0, 0, 0))) == host_event_count
    finally:
        if cloned_backend is not None:
            cloned_backend.close()
        backend.close()
