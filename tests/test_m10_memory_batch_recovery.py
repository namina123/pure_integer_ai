"""M-10 联合 batch、K-02 activation、rollback 和独立 Memory 包对抗。"""
from __future__ import annotations

from pathlib import Path

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
)
from pure_integer_ai.cognition.shared.memory_batch import (
    FAULT_MEMORY_BATCH_AFTER_ACTIVATION,
    FAULT_MEMORY_BATCH_AFTER_EVENT,
    FAULT_MEMORY_BATCH_AFTER_LINK,
    FAULT_MEMORY_BATCH_AFTER_PROJECTION,
    FAULT_MEMORY_BATCH_AFTER_ROLLBACK_PROJECTION,
    FAULT_MEMORY_BATCH_AFTER_ROLLBACK_RECEIPT,
    FAULT_MEMORY_BATCH_AFTER_STAGE,
    FAULT_MEMORY_BATCH_BEFORE_ACTIVATION,
    FAULT_MEMORY_GROUP_AFTER_COMMIT,
    FAULT_MEMORY_GROUP_AFTER_PROJECTION,
    FAULT_MEMORY_GROUP_AFTER_UNIT,
    MemoryBatchRuntimeConfig,
    install_memory_batch_runtimes,
)
from pure_integer_ai.cognition.shared.memory_event import (
    MemoryEvent,
    MemoryLinkedRef,
)
from pure_integer_ai.cognition.shared.memory_overlay import MemoryAccessContext
from pure_integer_ai.cognition.understanding.memory_intake import (
    HypothesisIntakeDraft,
    ObservationIntakeDraft,
)
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.storage.backend import DictBackend, SQLiteBackend
from pure_integer_ai.storage.memory_batch import (
    MEMORY_BATCH_CORE_DEPENDENCY_KEY,
    memory_batch_hash,
    source_record_dependency,
)
from pure_integer_ai.storage.memory_event import MEMORY_EVENT_TABLE
from pure_integer_ai.storage.memory_recovery import (
    load_memory_recovery_package,
    publish_memory_recovery_package,
    recovery_manifest_dependency,
)
from pure_integer_ai.storage.placement import (
    TemperatureProfile,
    TemperatureTier,
)
from pure_integer_ai.storage.recovery_package import (
    load_recovery_package,
    publish_recovery_package,
    registered_space_ids,
)
from pure_integer_ai.storage.recovery_protocol import (
    RecoveryConflictError,
    RecoveryIntegrityError,
)
from pure_integer_ai.storage.sealed_segment import SegmentBudget
from pure_integer_ai.storage.segment_dependency import SegmentDependency
from pure_integer_ai.storage.segment_repository import BackendObjectRepository
from pure_integer_ai.storage.source_record import SOURCE_RECORD_TABLE
from pure_integer_ai.storage.write_guard import RuntimeWriteGuardError
from pure_integer_ai.training.cursor import CursorState, mark_completed


_PROFILE = TemperatureProfile(
    (910, 1),
    (
        TemperatureTier((910, 1), 0),
        TemperatureTier((910, 2), 1),
    ),
)
_HOT = (910, 1)
_ACCESS = MemoryAccessContext(0, 0, 0)


def _core_dependency(value: int = 1) -> SegmentDependency:
    """构造测试用完整 Core 基线依赖。"""
    return SegmentDependency(
        MEMORY_BATCH_CORE_DEPENDENCY_KEY,
        (1, value),
        (2, value),
    )


def _config(dependency: SegmentDependency) -> MemoryBatchRuntimeConfig:
    """构造不写死于生产代码的测试温层和批次预算。"""
    return MemoryBatchRuntimeConfig(
        _PROFILE,
        _HOT,
        dependency,
        SegmentBudget(8, 1_000_000),
        SegmentBudget(64, 2_000_000),
    )


def _source(source_id: int, *, parser: int = 1) -> SourceRef:
    """构造稳定但彼此独立的全局来源。"""
    return SourceRef(
        71,
        source_id,
        source_id,
        OwnerScope(),
        VersionBundle(
            CorpusVersion(1),
            ParserVersion(parser),
            PrimitiveVersion(1),
            CurriculumVersion(1),
        ),
    )


class _Parser:
    """返回一个来源化候选，并记录是否被 staged 恢复绕过。"""

    def __init__(self, source: SourceRef, candidate: int) -> None:
        """绑定来源和候选身份。"""
        self.source = source
        self.candidate = candidate
        self.calls = 0

    def parse(self, source_slice):
        """把完整来源切片转为一个 Observation/Hypothesis 草案。"""
        self.calls += 1
        context = MemoryLinkedRef.object(ObjectIdentity(
            OBJECT_CONTEXT_SCOPE,
            (1000 + self.candidate,),
            self.source.owner,
            self.source.versions,
        ))
        signal = MemoryLinkedRef.object(ObjectIdentity(
            OBJECT_MINIMAL_INSTRUCTION,
            (2000 + self.candidate,),
            self.source.owner,
            self.source.versions,
        ))
        return ObservationIntakeDraft(
            (3000 + self.candidate,),
            context,
            hypotheses=(HypothesisIntakeDraft(
                (4000 + self.candidate,),
                (5000 + self.candidate,),
                (6000 + self.candidate,),
                (7000 + self.candidate,),
                1,
                signal_ref=signal,
            ),),
        )


class _FailOnce:
    """在指定 M-10 故障点首次命中时中断。"""

    def __init__(self, point: int, *, event_ordinal: int | None = None) -> None:
        """绑定故障点和可选事件序号。"""
        self.point = point
        self.event_ordinal = event_ordinal
        self.triggered = False

    def hit(self, point: int, context: dict[str, int]) -> None:
        """忽略其他边界，并确保只抛出一次。"""
        if self.triggered or point != self.point:
            return
        if (self.event_ordinal is not None
                and context.get("event_ordinal") != self.event_ordinal):
            return
        self.triggered = True
        raise RuntimeError(f"M-10 fault {point}")


class _RejectParser:
    """满足 parser 协议，但被调用即证明恢复路径错误。"""

    def parse(self, source_slice):
        """拒绝任何重复解析。"""
        pytest.fail("恢复后的 staged 单元不应重新调用 parser")


def _context(backend, dependency: SegmentDependency):
    """装配带 Companion、双 Memory 和正式 K-02 batch runtime 的上下文。"""
    ctx = make_train_context(backend, companion=True)
    install_memory_batch_runtimes(ctx, _config(dependency))
    return ctx


@pytest.mark.parametrize("backend_type", (DictBackend, SQLiteBackend))
@pytest.mark.parametrize("point", (
    FAULT_MEMORY_BATCH_AFTER_STAGE,
    FAULT_MEMORY_BATCH_AFTER_LINK,
    FAULT_MEMORY_BATCH_AFTER_EVENT,
    FAULT_MEMORY_BATCH_AFTER_PROJECTION,
    FAULT_MEMORY_BATCH_BEFORE_ACTIVATION,
    FAULT_MEMORY_BATCH_AFTER_ACTIVATION,
))
def test_faults_leave_only_old_or_complete_new_view(backend_type, point):
    """activation 前故障保持旧视图，activation 后故障保持完整新视图。"""
    backend = backend_type()
    try:
        ctx = _context(backend, _core_dependency())
        source = _source(1)
        parser = _Parser(source, 1)
        fault = _FailOnce(
            point,
            event_ordinal=(1 if point == FAULT_MEMORY_BATCH_AFTER_EVENT
                           else None),
        )
        with pytest.raises(RuntimeError, match="M-10 fault"):
            ctx.memory_read_intake.ingest(
                source,
                "批次来源",
                license_id="license-m10",
                batch_id=101,
                parser=parser,
                batch_fault_injector=fault,
            )
        visible = ctx.memory_read_events.query(access=_ACCESS)
        expected = 4 if point == FAULT_MEMORY_BATCH_AFTER_ACTIVATION else 0
        assert len(visible) == expected

        result = ctx.memory_read_intake.ingest(
            source,
            "批次来源",
            license_id="license-m10",
            batch_id=101,
            parser=parser,
        )
        assert parser.calls == 1
        assert len(ctx.memory_read_events.query(access=_ACCESS)) == 4
        aggregate = ctx.memory_read_aggregates.read(
            result.hypothesis_refs[0], access=_ACCESS)
        assert aggregate is not None and aggregate.support_count == 1
    finally:
        backend.close()


def test_sqlite_file_restart_rolls_forward_without_reparsing(tmp_path: Path):
    """文件后端在事件中段崩溃后从 staged segment 恢复并发布完整 activation。"""
    path = str(tmp_path / "memory_batch.sqlite3")
    dependency = _core_dependency()
    source = _source(2)
    parser = _Parser(source, 2)
    backend = SQLiteBackend(path)
    ctx = _context(backend, dependency)
    with pytest.raises(RuntimeError):
        ctx.memory_read_intake.ingest(
            source,
            "重启来源",
            license_id="license-restart",
            batch_id=102,
            parser=parser,
            batch_fault_injector=_FailOnce(
                FAULT_MEMORY_BATCH_AFTER_EVENT,
                event_ordinal=1,
            ),
        )
    backend.commit()
    backend.close()

    restored_backend = SQLiteBackend(path)
    try:
        restored = _context(restored_backend, dependency)
        assert len(restored.memory_read_events.query(access=_ACCESS)) == 4
        repeated = restored.memory_read_intake.ingest(
            source,
            "重启来源",
            license_id="license-restart",
            batch_id=102,
            parser=_RejectParser(),
        )
        assert repeated.outcome_kind == 1
        assert parser.calls == 1
    finally:
        restored_backend.close()


def test_half_group_rolls_back_and_never_marks_cursor_complete():
    """同 batch 后续单元失败时撤销先前 activation，cursor 保持未完成。"""
    backend = DictBackend()
    try:
        ctx = _context(backend, _core_dependency())
        first = _source(3)
        cursor = CursorState("base", "run")

        def first_action():
            """发布第一个来源单元。"""
            return ctx.memory_read_intake.ingest(
                first,
                "第一来源",
                license_id="license-group",
                batch_id=103,
                parser=_Parser(first, 3),
            )

        def failed_action():
            """模拟同一来源批次的后续单元失败。"""
            raise RuntimeError("second unit failed")

        with pytest.raises(RuntimeError, match="second unit failed"):
            ctx.memory_batch_coordinator.execute(
                103,
                (first_action, failed_action),
                cursor_commit=lambda: mark_completed(
                    cursor, 99, skippable=True),
            )
        assert 99 not in cursor.completed
        assert ctx.memory_read_events.query(access=_ACCESS) == ()
        assert ctx.memory_read_aggregates.query(access=_ACCESS) == ()
    finally:
        backend.close()


def test_rollback_hides_view_without_reusing_physical_timeline():
    """回滚可清空逻辑视图，但新批次必须越过仍留档的物理 timeline。"""
    backend = DictBackend()
    try:
        ctx = _context(backend, _core_dependency())
        first = _source(4)
        second = _source(5)
        first_result = ctx.memory_read_intake.ingest(
            first,
            "待回滚",
            license_id="license-rollback",
            batch_id=104,
            parser=_Parser(first, 4),
        )
        ctx.memory_read_batch_runtime.rollback_batch(104)
        assert ctx.memory_read_events.timeline_watermark() is None
        assert ctx.memory_read_events.physical_timeline_watermark().seq == 4
        assert ctx.memory_read_aggregates.read(
            first_result.hypothesis_refs[0], access=_ACCESS) is None

        second_result = ctx.memory_read_intake.ingest(
            second,
            "保留来源",
            license_id="license-rollback",
            batch_id=105,
            parser=_Parser(second, 5),
        )
        assert ctx.memory_read_events.timeline_watermark().seq == 8
        assert ctx.memory_read_events.physical_timeline_watermark().seq == 8
        records = ctx.memory_read_aggregates.query(access=_ACCESS)
        assert len(records) == 1
        assert ctx.memory_read_aggregates.read(
            second_result.hypothesis_refs[0], access=_ACCESS) == records[0]
    finally:
        backend.close()


def test_pending_batch_reserves_physical_timeline_with_bounded_watermark_lookup(
        monkeypatch):
    """隐藏批次仍占用物理序，后续批次只经倒序有限索引越过该水位。"""
    backend = DictBackend()
    try:
        ctx = _context(backend, _core_dependency())
        original_select = backend.select
        watermark_reads = []

        def monitored_select(
                table, where=None, where_gt=None, order_by=None, *,
                descending=False, limit=None):
            """记录 timeline 最大值查询，并保持 backend 协议透明转发。"""
            if table == MEMORY_EVENT_TABLE and order_by == "timeline_seq":
                watermark_reads.append((where, descending, limit))
            return original_select(
                table,
                where,
                where_gt,
                order_by,
                descending=descending,
                limit=limit,
            )

        monkeypatch.setattr(backend, "select", monitored_select)
        hidden = _source(6)
        visible = _source(7)
        ctx.memory_batch_visibility.receipts.begin_group(106)
        ctx.memory_read_intake.ingest(
            hidden,
            "隐藏来源",
            license_id="license-pending-timeline",
            batch_id=106,
            parser=_Parser(hidden, 6),
        )
        assert ctx.memory_read_events.timeline_watermark() is None
        assert ctx.memory_read_events.physical_timeline_watermark().seq == 4

        ctx.memory_read_intake.ingest(
            visible,
            "可见来源",
            license_id="license-pending-timeline",
            batch_id=107,
            parser=_Parser(visible, 7),
        )
        assert ctx.memory_read_events.timeline_watermark().seq == 8
        rows = original_select(MEMORY_EVENT_TABLE)
        assert tuple(sorted(row["timeline_seq"] for row in rows)) == tuple(
            range(1, 9))
        assert watermark_reads
        assert all(
            where == {"space_id": ctx.memory_read_events.memory_space_id}
            and descending is True
            and limit == 2
            for where, descending, limit in watermark_reads
        )
    finally:
        backend.close()


def test_batch_activation_changes_projection_state_without_reusing_events():
    """pending 转 active 时物理事件不重写，但 batch 逻辑状态必须使投影失效。"""
    backend = DictBackend()
    try:
        ctx = _context(backend, _core_dependency())
        source = _source(8)
        parser = _Parser(source, 8)
        with pytest.raises(RuntimeError, match="M-10 fault"):
            ctx.memory_read_intake.ingest(
                source,
                "待激活来源",
                license_id="license-projection-state",
                batch_id=108,
                parser=parser,
                batch_fault_injector=_FailOnce(
                    FAULT_MEMORY_BATCH_BEFORE_ACTIVATION),
            )
        pending_state = ctx.memory_read_events.projection_state_key()
        assert ctx.memory_read_events.timeline_watermark() is None

        ctx.memory_read_intake.ingest(
            source,
            "待激活来源",
            license_id="license-projection-state",
            batch_id=108,
            parser=parser,
        )
        active_state = ctx.memory_read_events.projection_state_key()
        assert parser.calls == 1
        assert pending_state[0] == active_state[0] == 4
        assert pending_state != active_state
        assert len(ctx.memory_read_events.query(access=_ACCESS)) == 4
    finally:
        backend.close()


@pytest.mark.parametrize("backend_type", (DictBackend, SQLiteBackend))
@pytest.mark.parametrize("point", (
    FAULT_MEMORY_BATCH_AFTER_ROLLBACK_PROJECTION,
    FAULT_MEMORY_BATCH_AFTER_ROLLBACK_RECEIPT,
))
def test_rollback_fault_is_either_old_or_complete_hidden(backend_type, point):
    """rollback 投影后故障恢复旧视图，receipt 后故障保持完整隐藏。"""
    backend = backend_type()
    try:
        ctx = _context(backend, _core_dependency())
        source = _source(9)
        result = ctx.memory_read_intake.ingest(
            source,
            "回滚故障",
            license_id="license-rollback-fault",
            batch_id=109,
            parser=_Parser(source, 9),
        )
        with pytest.raises(RuntimeError, match="M-10 fault"):
            ctx.memory_read_batch_runtime.rollback_batch(
                109,
                fault_injector=_FailOnce(point),
            )
        expected = (
            4
            if point == FAULT_MEMORY_BATCH_AFTER_ROLLBACK_PROJECTION
            else 0
        )
        assert len(ctx.memory_read_events.query(access=_ACCESS)) == expected
        aggregate = ctx.memory_read_aggregates.read(
            result.hypothesis_refs[0], access=_ACCESS)
        assert (aggregate is not None) == bool(expected)

        ctx.memory_read_batch_runtime.rollback_batch(109)
        assert ctx.memory_read_events.query(access=_ACCESS) == ()
        assert ctx.memory_read_aggregates.query(access=_ACCESS) == ()
    finally:
        backend.close()


@pytest.mark.parametrize("point, expected", (
    (FAULT_MEMORY_GROUP_AFTER_UNIT, 0),
    (FAULT_MEMORY_GROUP_AFTER_PROJECTION, 0),
    (FAULT_MEMORY_GROUP_AFTER_COMMIT, 8),
))
def test_group_fault_is_either_old_or_complete_new(point, expected):
    """组级任一故障点只能留下完整旧视图或完整新视图。"""
    backend = DictBackend()
    try:
        ctx = _context(backend, _core_dependency())
        first = _source(10)
        second = _source(11)
        first_parser = _Parser(first, 10)
        second_parser = _Parser(second, 11)
        cursor = CursorState("base", "group-fault")

        def first_action():
            """摄入组内第一个来源。"""
            return ctx.memory_read_intake.ingest(
                first,
                "组来源一",
                license_id="license-group-fault",
                batch_id=110,
                parser=first_parser,
            )

        def second_action():
            """摄入组内第二个来源。"""
            return ctx.memory_read_intake.ingest(
                second,
                "组来源二",
                license_id="license-group-fault",
                batch_id=110,
                parser=second_parser,
            )

        with pytest.raises(RuntimeError, match="M-10 fault"):
            ctx.memory_batch_coordinator.execute(
                110,
                (first_action, second_action),
                cursor_commit=lambda: mark_completed(
                    cursor, 110, skippable=True),
                fault_injector=_FailOnce(point),
            )
        assert len(ctx.memory_read_events.query(access=_ACCESS)) == expected
        assert 110 not in cursor.completed
        if expected:
            assert len(ctx.memory_read_aggregates.query(access=_ACCESS)) == 2
        else:
            assert ctx.memory_read_aggregates.query(access=_ACCESS) == ()
    finally:
        backend.close()


def test_sqlite_restart_keeps_pending_group_hidden_then_resumes(tmp_path: Path):
    """组内单元之间崩溃后保持旧视图，重跑 coordinator 后整体提交。"""
    path = str(tmp_path / "memory_group.sqlite3")
    dependency = _core_dependency()
    first = _source(12)
    second = _source(13)
    first_parser = _Parser(first, 12)

    backend = SQLiteBackend(path)
    ctx = _context(backend, dependency)
    ctx.memory_batch_visibility.receipts.begin_group(111)
    ctx.memory_read_intake.ingest(
        first,
        "崩溃前来源",
        license_id="license-group-restart",
        batch_id=111,
        parser=first_parser,
    )
    assert ctx.memory_read_events.query(access=_ACCESS) == ()
    backend.commit()
    backend.close()

    restored_backend = SQLiteBackend(path)
    try:
        restored = _context(restored_backend, dependency)
        assert restored.memory_read_events.query(access=_ACCESS) == ()
        assert restored.memory_read_aggregates.query(access=_ACCESS) == ()
        second_parser = _Parser(second, 13)
        cursor = CursorState("base", "group-restart")

        def first_action():
            """从 staged 单元恢复第一个来源，不允许重跑 parser。"""
            return restored.memory_read_intake.ingest(
                first,
                "崩溃前来源",
                license_id="license-group-restart",
                batch_id=111,
                parser=_RejectParser(),
            )

        def second_action():
            """摄入重启后尚未执行的第二个来源。"""
            return restored.memory_read_intake.ingest(
                second,
                "重启后来源",
                license_id="license-group-restart",
                batch_id=111,
                parser=second_parser,
            )

        restored.memory_batch_coordinator.execute(
            111,
            (first_action, second_action),
            cursor_commit=lambda: mark_completed(
                cursor, 111, skippable=True),
        )
        assert first_parser.calls == 1
        assert second_parser.calls == 1
        assert len(restored.memory_read_events.query(access=_ACCESS)) == 8
        assert len(restored.memory_read_aggregates.query(access=_ACCESS)) == 2
        assert 111 in cursor.completed
    finally:
        restored_backend.close()


def test_group_commit_and_rollback_hot_path_do_not_scan_all_staged_segments(
        monkeypatch):
    """正常组提交与回滚只查当前 batch 索引，不解封全部历史段。"""
    backend = DictBackend()
    try:
        ctx = _context(backend, _core_dependency())
        first = _source(14)
        second = _source(15)
        monkeypatch.setattr(
            ctx.memory_batch_visibility.receipts,
            "staged_batches",
            lambda: pytest.fail("组热路径不得扫描全部 staged segment"),
        )

        def first_action():
            """摄入当前批次第一个来源。"""
            return ctx.memory_read_intake.ingest(
                first,
                "索引来源一",
                license_id="license-indexed-group",
                batch_id=112,
                parser=_Parser(first, 14),
            )

        def second_action():
            """摄入当前批次第二个来源。"""
            return ctx.memory_read_intake.ingest(
                second,
                "索引来源二",
                license_id="license-indexed-group",
                batch_id=112,
                parser=_Parser(second, 15),
            )

        ctx.memory_batch_coordinator.execute(
            112,
            (first_action, second_action),
            cursor_commit=lambda: None,
        )
        assert len(ctx.memory_read_events.query(access=_ACCESS)) == 8
        ctx.memory_batch_coordinator.rollback_batch(112)
        assert ctx.memory_read_events.query(access=_ACCESS) == ()
        assert ctx.memory_read_aggregates.query(access=_ACCESS) == ()
    finally:
        backend.close()


def test_same_immutable_events_remain_visible_through_another_batch():
    """同一物理事件被两批引用时，回滚一批不删除另一批可见性。"""
    backend = DictBackend()
    try:
        ctx = _context(backend, _core_dependency())
        first = _source(16)
        second = _source(17)
        result = ctx.memory_read_intake.ingest(
            first,
            "共享事件来源一",
            license_id="license-shared-event",
            batch_id=113,
            parser=_Parser(first, 16),
        )
        first_hash = memory_batch_hash(
            ctx.memory_read_events.memory_space_identity.stable_key(),
            first.stable_key(),
            113,
        )
        staged = ctx.memory_batch_visibility.receipts.staged(first_hash)
        assert staged is not None
        events = tuple(
            MemoryEvent.from_stable_key(key) for key in staged.event_keys)
        second_record = ctx.memory_read_intake.source_intake.ensure(
            second,
            "共享事件来源二",
            license_id="license-shared-event",
            batch_id=114,
        )
        ctx.memory_read_batch_runtime.publish(
            second,
            114,
            events,
            source_dependency=source_record_dependency(second_record),
        )
        assert len(ctx.memory_read_events.query(access=_ACCESS)) == 4
        assert all(
            len(ctx.memory_batch_visibility.links.for_event(
                item.event_hash)) == 2
            for item in ctx.memory_read_events.query(access=_ACCESS)
        )

        ctx.memory_read_batch_runtime.rollback_batch(113)
        assert len(ctx.memory_read_events.query(access=_ACCESS)) == 4
        aggregate = ctx.memory_read_aggregates.read(
            result.hypothesis_refs[0], access=_ACCESS)
        assert aggregate is not None and aggregate.support_count == 1

        ctx.memory_read_batch_runtime.rollback_batch(114)
        assert ctx.memory_read_events.query(access=_ACCESS) == ()
        assert ctx.memory_read_aggregates.query(access=_ACCESS) == ()
    finally:
        backend.close()


def test_memory_package_rejects_non_companion_dependency(tmp_path: Path):
    """独立 Memory 包把非 Companion 空间作为伴随依赖时 fail closed。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend, companion=True)
        BackendObjectRepository(backend)
        baseline = publish_recovery_package(
            backend,
            str(tmp_path),
            "invalid-companion-base",
            spaces=registered_space_ids(backend),
            include_registered_tables=True,
            require_all_spaces=True,
        )
        with pytest.raises(RecoveryIntegrityError, match="非 Companion"):
            publish_memory_recovery_package(
                backend,
                str(tmp_path),
                "invalid-companion-memory",
                memory_space_ids=(ctx.memory_read.space_id,),
                companion_space_ids=(ctx.memory_interact.space_id,),
                core_dependency=recovery_manifest_dependency(baseline),
            )
        with pytest.raises(RecoveryIntegrityError, match="完整 Memory 空间集"):
            publish_memory_recovery_package(
                backend,
                str(tmp_path),
                "partial-memory-package",
                memory_space_ids=(ctx.memory_read.space_id,),
                companion_space_ids=(
                    ctx.concept_index._companion.space_id,),
                core_dependency=recovery_manifest_dependency(baseline),
            )
    finally:
        backend.close()


def test_materialize_callback_cannot_hide_core_writes():
    """M-05 materialize 只允许纯草案转换，后端写在落表前被拒绝。"""
    backend = DictBackend()
    try:
        ctx = _context(backend, _core_dependency())
        source = _source(6)

        def writes_core(draft):
            """故意尝试从回调修改 Core 空间注册表。"""
            backend.insert("space", {})
            return draft

        with pytest.raises(RuntimeWriteGuardError):
            ctx.memory_read_intake.ingest(
                source,
                "只读回调",
                license_id="license-guard",
                batch_id=106,
                parser=_Parser(source, 6),
                materialize=writes_core,
            )
        assert ctx.memory_read_events.query(access=_ACCESS) == ()
    finally:
        backend.close()


def test_memory_package_merges_after_core_without_duplication(tmp_path: Path):
    """独立 Memory 包要求 Core 依赖，重复合并零写且恢复消费者行为。"""
    source_backend = DictBackend()
    target_backend = DictBackend()
    try:
        source_ctx = make_train_context(source_backend, companion=True)
        baseline = publish_recovery_package(
            source_backend,
            str(tmp_path),
            "core-base",
            spaces=registered_space_ids(source_backend),
            include_registered_tables=True,
            require_all_spaces=True,
        )
        dependency = recovery_manifest_dependency(baseline)
        install_memory_batch_runtimes(source_ctx, _config(dependency))
        source = _source(7)
        result = source_ctx.memory_read_intake.ingest(
            source,
            "独立包来源",
            license_id="license-package",
            batch_id=107,
            parser=_Parser(source, 7),
        )
        companion_id = source_ctx.concept_index._companion.space_id
        publish_memory_recovery_package(
            source_backend,
            str(tmp_path),
            "memory-delta",
            memory_space_ids=(
                source_ctx.memory_read.space_id,
                source_ctx.memory_interact.space_id,
            ),
            companion_space_ids=(companion_id,),
            core_dependency=dependency,
        )

        target_ctx = make_train_context(target_backend, companion=True)
        BackendObjectRepository(target_backend)
        load_recovery_package(
            target_backend,
            str(tmp_path),
            "core-base",
        )
        with pytest.raises(RecoveryIntegrityError, match="Core dependency"):
            load_memory_recovery_package(
                target_backend,
                str(tmp_path),
                "memory-delta",
                available_dependencies=(),
            )
        first = load_memory_recovery_package(
            target_backend,
            str(tmp_path),
            "memory-delta",
            available_dependencies=(dependency,),
        )
        second = load_memory_recovery_package(
            target_backend,
            str(tmp_path),
            "memory-delta",
            available_dependencies=(dependency,),
        )
        assert first.loaded_tables
        assert second.loaded_tables == ()
        install_memory_batch_runtimes(target_ctx, _config(dependency))
        assert len(target_ctx.memory_read_events.query(access=_ACCESS)) == 4
        aggregate = target_ctx.memory_read_aggregates.read(
            result.hypothesis_refs[0], access=_ACCESS)
        assert aggregate is not None and aggregate.support_count == 1
    finally:
        source_backend.close()
        target_backend.close()


def test_memory_package_same_recovery_key_drift_rolls_back_load(tmp_path: Path):
    """目标同 source_hash 异内容时合并 fail closed，其他表保持加载前状态。"""
    source_backend = DictBackend()
    target_backend = DictBackend()
    try:
        source_ctx = make_train_context(source_backend, companion=True)
        baseline = publish_recovery_package(
            source_backend,
            str(tmp_path),
            "conflict-base",
            spaces=registered_space_ids(source_backend),
            include_registered_tables=True,
            require_all_spaces=True,
        )
        dependency = recovery_manifest_dependency(baseline)
        install_memory_batch_runtimes(source_ctx, _config(dependency))
        source = _source(8)
        source_ctx.memory_read_intake.ingest(
            source,
            "真实内容",
            license_id="license-conflict",
            batch_id=108,
            parser=_Parser(source, 8),
        )
        publish_memory_recovery_package(
            source_backend,
            str(tmp_path),
            "conflict-memory",
            memory_space_ids=(
                source_ctx.memory_read.space_id,
                source_ctx.memory_interact.space_id,
            ),
            companion_space_ids=(
                source_ctx.concept_index._companion.space_id,),
            core_dependency=dependency,
        )

        make_train_context(target_backend, companion=True)
        BackendObjectRepository(target_backend)
        load_recovery_package(
            target_backend,
            str(tmp_path),
            "conflict-base",
        )
        source_row = source_backend.select(
            SOURCE_RECORD_TABLE, where=None)[0]
        target_backend.insert(SOURCE_RECORD_TABLE, {
            **source_row,
            "raw_text": "漂移内容",
        })
        baseline_state = target_backend.recovery_state_snapshot()
        with pytest.raises(RecoveryConflictError, match="recovery_key"):
            load_memory_recovery_package(
                target_backend,
                str(tmp_path),
                "conflict-memory",
                available_dependencies=(dependency,),
            )
        assert target_backend.recovery_state_snapshot() == baseline_state
    finally:
        source_backend.close()
        target_backend.close()
