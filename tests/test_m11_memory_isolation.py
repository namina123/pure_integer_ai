"""M-11 全对象隔离、逻辑导出、遗忘事务和恢复对抗。"""
from __future__ import annotations

from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.graph_ontology import (
    relation_concept_identity,
)
from pure_integer_ai.cognition.shared.identity import (
    CorpusVersion,
    CurriculumVersion,
    GLOBAL_OWNER_SCOPE,
    OBJECT_CONTEXT_SCOPE,
    OBJECT_MINIMAL_INSTRUCTION,
    ObjectIdentity,
    OwnerScope,
    ParserVersion,
    PrimitiveVersion,
    SourceRef,
    VersionBundle,
    VISIBILITY_SESSION,
    VISIBILITY_TENANT,
    VISIBILITY_USER,
    concept_identity,
)
from pure_integer_ai.cognition.shared.memory_batch import (
    MemoryBatchRuntimeConfig,
    install_memory_batch_runtimes,
)
from pure_integer_ai.cognition.shared.memory_event import MemoryLinkedRef
from pure_integer_ai.cognition.shared.memory_overlay import (
    MemoryAccessContext,
    MemoryOverlayRelation,
)
from pure_integer_ai.cognition.shared.memory_owner import (
    OWNER_SELECTION_EXACT,
    OWNER_SELECTION_SUBTREE,
    MemoryLayerCandidate,
    MemoryManagementContext,
    MemoryOwnerSelector,
    select_memory_layers,
)
from pure_integer_ai.cognition.shared.scope_identity import session_scope
from pure_integer_ai.cognition.understanding.memory_intake import (
    HypothesisIntakeDraft,
    ObservationIntakeDraft,
)
from pure_integer_ai.cognition.understanding.source_intake import (
    SourceIntakeIntegrityError,
)
from pure_integer_ai.experiments.evaluation_isolation import isolated_evaluation
from pure_integer_ai.experiments.memory_isolation_runtime import (
    FAULT_MEMORY_FORGET_AFTER_COMMIT,
    FAULT_MEMORY_FORGET_AFTER_PROJECTION,
    FAULT_MEMORY_FORGET_AFTER_STAGE,
    MemoryIsolationError,
    install_memory_isolation_runtime,
)
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.storage.backend import DictBackend, SQLiteBackend
from pure_integer_ai.storage.edge_store import EPI_STRUCTURED, SOURCE_BARE_TEXT
from pure_integer_ai.storage.memory_batch import MEMORY_BATCH_CORE_DEPENDENCY_KEY
from pure_integer_ai.storage.memory_event import MEMORY_EVENT_TABLE
from pure_integer_ai.storage.memory_overlay import MEMORY_OVERLAY_TABLE
from pure_integer_ai.storage.placement import TemperatureProfile, TemperatureTier
from pure_integer_ai.storage.sealed_segment import SegmentBudget
from pure_integer_ai.storage.segment_dependency import SegmentDependency
from pure_integer_ai.storage.source_record import SOURCE_RECORD_TABLE
from pure_integer_ai.storage.spaces.companion import TEXT_ASSOC_TABLE


_PROFILE = TemperatureProfile(
    (911, 1),
    (
        TemperatureTier((911, 1), 0),
        TemperatureTier((911, 2), 1),
    ),
)
_HOT = (911, 1)
_ADMIN = GLOBAL_OWNER_SCOPE


def _dependency() -> SegmentDependency:
    """构造 M-10/M-11 测试共享的完整 Core 依赖。"""
    return SegmentDependency(
        MEMORY_BATCH_CORE_DEPENDENCY_KEY,
        (11, 1),
        (12, 1),
    )


def _config() -> MemoryBatchRuntimeConfig:
    """构造由测试注入的温层和对象预算。"""
    return MemoryBatchRuntimeConfig(
        _PROFILE,
        _HOT,
        _dependency(),
        SegmentBudget(8, 1_000_000),
        SegmentBudget(64, 2_000_000),
    )


class _Authorizer:
    """用稳定配置键控制测试中的跨 owner 管理请求。"""

    def __init__(self, *, allowed: bool = True, revision: int = 1) -> None:
        """绑定授权结果和配置修订号。"""
        self.allowed = allowed
        self.revision = revision

    def state_key(self) -> tuple[int, ...]:
        """返回不包含进程地址的授权配置键。"""
        return 911, self.revision, int(self.allowed)

    def authorize(
            self,
            actor: OwnerScope,
            selector: MemoryOwnerSelector,
            ) -> bool:
        """只允许显式测试管理员执行已构造的 owner 选择。"""
        return self.allowed and actor == _ADMIN and isinstance(
            selector, MemoryOwnerSelector)

    def clone_for_context(self, ctx) -> "_Authorizer":
        """为 V-06 clone 返回配置相同但对象独立的授权器。"""
        del ctx
        return _Authorizer(allowed=self.allowed, revision=self.revision)


class _Parser:
    """把一个来源转换为唯一 Observation/Hypothesis/Evidence 闭包。"""

    def __init__(self, source: SourceRef, candidate: int) -> None:
        """绑定来源和不与其他 fixture 冲突的候选编号。"""
        self.source = source
        self.candidate = candidate
        self.calls = 0

    def parse(self, source_slice):
        """从来源切片生成不写 Core 的结构化草案。"""
        assert source_slice.source == self.source
        self.calls += 1
        context = MemoryLinkedRef.object(ObjectIdentity(
            OBJECT_CONTEXT_SCOPE,
            (10_000 + self.candidate,),
            self.source.owner,
            self.source.versions,
        ))
        signal = MemoryLinkedRef.object(ObjectIdentity(
            OBJECT_MINIMAL_INSTRUCTION,
            (20_000 + self.candidate,),
            self.source.owner,
            self.source.versions,
        ))
        return ObservationIntakeDraft(
            (30_000 + self.candidate,),
            context,
            hypotheses=(HypothesisIntakeDraft(
                (40_000 + self.candidate,),
                (50_000 + self.candidate,),
                (60_000 + self.candidate,),
                (70_000 + self.candidate,),
                1,
                signal_ref=signal,
            ),),
        )


class _FailOnce:
    """在指定 M-11 边界首次命中时中断。"""

    def __init__(self, point: int) -> None:
        """绑定唯一故障点。"""
        self.point = point
        self.triggered = False

    def hit(self, point: int, context: dict[str, int]) -> None:
        """忽略其他边界，命中后抛出一次异常。"""
        assert context["operation_hash"] > 0
        if self.triggered or point != self.point:
            return
        self.triggered = True
        raise RuntimeError(f"M-11 fault {point}")


def _source(
        source_id: int,
        owner: OwnerScope,
        *,
        parser: int = 1,
        ) -> SourceRef:
    """构造 owner 和 parser 版本都进入完整键的来源。"""
    return SourceRef(
        71,
        source_id,
        source_id,
        owner,
        VersionBundle(
            CorpusVersion(1),
            ParserVersion(parser),
            PrimitiveVersion(1),
            CurriculumVersion(1),
        ),
    )


def _context(backend, *, authorizer: _Authorizer | None = None):
    """装配 Companion、M-10 batch 和 M-11 isolation runtime。"""
    ctx = make_train_context(backend, companion=True)
    install_memory_batch_runtimes(ctx, _config())
    install_memory_isolation_runtime(ctx, authorizer or _Authorizer())
    return ctx


def _ingest(ctx, source: SourceRef, candidate: int):
    """按 owner visibility 路由到正式阅读或交互 Memory 摄入。"""
    intake = (
        ctx.memory_interact_intake
        if source.owner.visibility == VISIBILITY_SESSION
        else ctx.memory_read_intake
    )
    parser = _Parser(source, candidate)
    result = intake.ingest(
        source,
        f"来源-{candidate}",
        license_id="license-m11",
        batch_id=1_000 + candidate,
        parser=parser,
    )
    assert parser.calls == 1
    return result


def _management(
        owner: OwnerScope,
        selection_kind: int,
        *,
        key: tuple[int, ...] = (911, 1, 1),
        ) -> MemoryManagementContext:
    """构造与普通读取完全分离的管理请求。"""
    return MemoryManagementContext(
        _ADMIN,
        MemoryOwnerSelector(owner, selection_kind),
        key,
    )


def _refs(ctx):
    """在 Core 中物化 overlay 所需的三个稳定 typed ref。"""
    return (
        ctx.graph_ontology.materialize(relation_concept_identity((91_101,))),
        ctx.graph_ontology.materialize(concept_identity((91_102,))),
        ctx.graph_ontology.materialize(concept_identity((91_103,))),
    )


def _overlay_relation(ctx, owner: OwnerScope, qualifier: int):
    """构造 owner 完整进入身份的 Memory-local overlay 关系。"""
    predicate, subject, object_ref = _refs(ctx)
    return MemoryOverlayRelation(
        ctx.memory_interact_overlay.memory_space_identity,
        owner,
        VersionBundle(),
        predicate,
        subject,
        object_ref,
        session_scope(90_000 + qualifier, owner=owner),
        SOURCE_BARE_TEXT,
        EPI_STRUCTURED,
        1,
        (qualifier,),
    )


def test_acl_managed_export_and_session_zero_are_closed():
    """普通读取不跨 session，管理 exact/subtree 只在授权后开放目标闭包。"""
    backend = DictBackend()
    try:
        ctx = _context(backend)
        tenant = OwnerScope(1, visibility=VISIBILITY_TENANT)
        user = OwnerScope(1, 2, visibility=VISIBILITY_USER)
        session_a = OwnerScope(1, 2, 3, VISIBILITY_SESSION)
        session_b = OwnerScope(1, 2, 4, VISIBILITY_SESSION)
        other_user = OwnerScope(1, 9, 5, VISIBILITY_SESSION)
        for candidate, owner in enumerate((
                GLOBAL_OWNER_SCOPE, tenant, user, session_a, session_b,
                other_user), start=1):
            _ingest(ctx, _source(candidate, owner), candidate)

        no_session = ctx.memory_isolation_runtime.export(
            MemoryAccessContext(1, 2, 0))
        assert len(no_session.events) == 12
        assert len(no_session.sources) == 3
        assert all(item.event.object_ref.owner.visibility != VISIBILITY_SESSION
                   for item in no_session.events)

        session_export = ctx.memory_isolation_runtime.export(
            MemoryAccessContext(1, 2, 3))
        assert len(session_export.events) == 16
        assert {item.event.object_ref.owner for item in session_export.events} == {
            GLOBAL_OWNER_SCOPE, tenant, user, session_a,
        }

        exact = ctx.memory_isolation_runtime.export_managed(
            _management(session_b, OWNER_SELECTION_EXACT))
        assert len(exact.events) == 4
        assert {item.event.object_ref.owner for item in exact.events} == {session_b}
        assert len(exact.sources) == 1 and len(exact.companions) == 1

        subtree = ctx.memory_isolation_runtime.export_managed(
            _management(user, OWNER_SELECTION_SUBTREE))
        assert len(subtree.events) == 12
        assert {item.event.object_ref.owner for item in subtree.events} == {
            user, session_a, session_b,
        }

        with pytest.raises(MemoryIsolationError, match="state key 漂移"):
            ctx.memory_isolation_runtime.export_managed(
                _management(session_a, OWNER_SELECTION_EXACT, key=(911, 9, 1)))
    finally:
        backend.close()


def test_layer_selection_uses_only_open_shadow_keys():
    """层覆盖只按调用方提供的 shadow key 选最高可见 owner 并保留轨迹。"""
    access = MemoryAccessContext(1, 2, 3)
    shared = (101,)
    candidates = (
        MemoryLayerCandidate(GLOBAL_OWNER_SCOPE, shared, (1,)),
        MemoryLayerCandidate(
            OwnerScope(1, visibility=VISIBILITY_TENANT), shared, (2,)),
        MemoryLayerCandidate(
            OwnerScope(1, 2, visibility=VISIBILITY_USER), shared, (3,)),
        MemoryLayerCandidate(
            OwnerScope(1, 2, 3, VISIBILITY_SESSION), shared, (4,)),
        MemoryLayerCandidate(GLOBAL_OWNER_SCOPE, (202,), (5,)),
    )
    selected = select_memory_layers(candidates, access=access)
    assert tuple(item.value_key for item in selected.selected) == ((4,), (5,))
    assert tuple(item.value_key for item in selected.shadowed) == (
        (1,), (2,), (3,))


def test_unauthorized_management_fails_closed():
    """缺少策略允许时，跨 owner 导出和遗忘都不得退化为普通 ACL。"""
    backend = DictBackend()
    try:
        ctx = _context(backend, authorizer=_Authorizer(allowed=False))
        session = OwnerScope(2, 3, 4, VISIBILITY_SESSION)
        _ingest(ctx, _source(10, session), 10)
        context = _management(
            session, OWNER_SELECTION_EXACT, key=(911, 1, 0))
        with pytest.raises(MemoryIsolationError, match="未获授权"):
            ctx.memory_isolation_runtime.export_managed(context)
        with pytest.raises(MemoryIsolationError, match="未获授权"):
            ctx.memory_isolation_runtime.forget(context, reason_key=(1,))
    finally:
        backend.close()


@pytest.mark.parametrize("point", (
    FAULT_MEMORY_FORGET_AFTER_STAGE,
    FAULT_MEMORY_FORGET_AFTER_PROJECTION,
    FAULT_MEMORY_FORGET_AFTER_COMMIT,
))
def test_forget_faults_leave_only_old_or_new_view(point: int):
    """set、投影和 commit 三边界故障只能暴露完整旧视图或完整新视图。"""
    backend = DictBackend()
    try:
        ctx = _context(backend)
        session = OwnerScope(3, 4, 5, VISIBILITY_SESSION)
        result = _ingest(ctx, _source(20, session), 20)
        ctx.memory_interact_overlay.add(_overlay_relation(ctx, session, 20))
        access = MemoryAccessContext(3, 4, 5)
        context = _management(session, OWNER_SELECTION_EXACT)
        with pytest.raises(RuntimeError, match="M-11 fault"):
            ctx.memory_isolation_runtime.forget(
                context,
                reason_key=(20,),
                fault_injector=_FailOnce(point),
            )

        exported = ctx.memory_isolation_runtime.export(access)
        expected = 0 if point == FAULT_MEMORY_FORGET_AFTER_COMMIT else 4
        assert len(exported.events) == expected
        assert len(exported.overlays) == (0 if expected == 0 else 1)
        aggregate = ctx.memory_interact_aggregates.read(
            result.hypothesis_refs[0], access=access)
        assert (aggregate is None) == (expected == 0)

        recovered = ctx.memory_isolation_runtime.recover_pending()
        if point == FAULT_MEMORY_FORGET_AFTER_COMMIT:
            assert recovered == ()
        else:
            assert len(recovered) == 1
        assert ctx.memory_isolation_runtime.export(access).events == ()
        assert ctx.memory_interact_aggregates.read(
            result.hypothesis_refs[0], access=access) is None
    finally:
        backend.close()


def test_forget_hides_logical_closure_but_keeps_physical_rows_and_allows_new_key():
    """遗忘退出全逻辑闭包，物理留档不删，新完整 parser 键可重新学习。"""
    backend = DictBackend()
    try:
        ctx = _context(backend)
        session = OwnerScope(5, 6, 7, VISIBILITY_SESSION)
        old_source = _source(30, session, parser=1)
        old = _ingest(ctx, old_source, 30)
        old_assoc = old.source_record.companion_assoc_id
        overlay = ctx.memory_interact_overlay.add(
            _overlay_relation(ctx, session, 30))
        physical_counts = {
            table: backend.count(table)
            for table in (
                MEMORY_EVENT_TABLE,
                MEMORY_OVERLAY_TABLE,
                SOURCE_RECORD_TABLE,
                TEXT_ASSOC_TABLE,
            )
        }
        result = ctx.memory_isolation_runtime.forget(
            _management(session, OWNER_SELECTION_EXACT),
            reason_key=(30,),
        )
        assert result.event_count == 4
        assert result.overlay_count == 1
        assert result.source_count == 1
        assert result.companion_count == 1
        assert ctx.memory_isolation_runtime.export(
            MemoryAccessContext(5, 6, 7)).events == ()
        assert ctx.memory_interact_overlay.read(
            overlay.identity_hash,
            access=MemoryAccessContext(5, 6, 7),
        ) is None
        assert {
            table: backend.count(table) for table in physical_counts
        } == physical_counts
        assert ctx.memory_interact_intake.source_intake.repository.find(
            old_source.stable_key()) is not None
        assert ctx.memory_interact_intake.source_intake.companion.read(
            old_assoc)["text"] == "来源-30"

        with pytest.raises(SourceIntakeIntegrityError, match="不得按同一身份重放"):
            ctx.memory_interact_intake.ingest(
                old_source,
                "来源-30",
                license_id="license-m11",
                batch_id=1_030,
                parser=_Parser(old_source, 30),
            )

        new_source = _source(30, session, parser=2)
        new = _ingest(ctx, new_source, 31)
        assert new.source_record.companion_assoc_id != old_assoc
        exported = ctx.memory_isolation_runtime.export(
            MemoryAccessContext(5, 6, 7))
        assert len(exported.events) == 4
        assert tuple(item.source_key for item in exported.sources) == (
            new_source.stable_key(),)
        assert tuple(item.assoc_id for item in exported.companions) == (
            new.source_record.companion_assoc_id,)
    finally:
        backend.close()


def test_forget_changes_only_descriptor_scoped_visibility_state():
    """遗忘提交不增加 Memory 事件，也不得因共享 location epoch 改写 batch 状态。"""
    backend = DictBackend()
    try:
        ctx = _context(backend)
        session = OwnerScope(5, 6, 8, VISIBILITY_SESSION)
        _ingest(ctx, _source(32, session), 32)
        event_log = ctx.memory_interact_events
        before_projection = event_log.projection_state_key()
        before_batch = ctx.memory_batch_visibility.state_key()
        before_forget = ctx.memory_forget_visibility.state_key()

        ctx.memory_isolation_runtime.forget(
            _management(session, OWNER_SELECTION_EXACT),
            reason_key=(32,),
        )

        after_projection = event_log.projection_state_key()
        assert before_projection[0] == after_projection[0] == 4
        assert ctx.memory_batch_visibility.state_key() == before_batch
        assert ctx.memory_forget_visibility.state_key() != before_forget
        assert after_projection != before_projection
    finally:
        backend.close()


def test_sqlite_restart_recovers_subtree_owners_from_staged_targets(
        tmp_path: Path):
    """subtree pending forget 重启后从 set 恢复全部实际 owner 并提交新投影。"""
    path = str(tmp_path / "memory_forget.sqlite3")
    user = OwnerScope(7, 8, visibility=VISIBILITY_USER)
    session_a = OwnerScope(7, 8, 9, VISIBILITY_SESSION)
    session_b = OwnerScope(7, 8, 10, VISIBILITY_SESSION)

    backend = SQLiteBackend(path)
    ctx = _context(backend)
    results = tuple(
        _ingest(ctx, _source(40 + index, owner), 40 + index)
        for index, owner in enumerate((user, session_a, session_b))
    )
    with pytest.raises(RuntimeError, match="M-11 fault"):
        ctx.memory_isolation_runtime.forget(
            _management(user, OWNER_SELECTION_SUBTREE),
            reason_key=(40,),
            fault_injector=_FailOnce(FAULT_MEMORY_FORGET_AFTER_STAGE),
        )
    assert len(ctx.memory_isolation_runtime.export(
        MemoryAccessContext(7, 8, 9)).events) == 8
    backend.commit()
    backend.close()

    restored_backend = SQLiteBackend(path)
    try:
        restored = _context(restored_backend)
        assert restored.memory_isolation_runtime.export(
            MemoryAccessContext(7, 8, 9)).events == ()
        assert restored.memory_isolation_runtime.export(
            MemoryAccessContext(7, 8, 10)).events == ()
        for result, access in zip(results, (
                MemoryAccessContext(7, 8, 0),
                MemoryAccessContext(7, 8, 9),
                MemoryAccessContext(7, 8, 10))):
            aggregate = (
                restored.memory_interact_aggregates
                if result.hypothesis_refs[0].memory_space
                == restored.memory_interact_events.memory_space_identity
                else restored.memory_read_aggregates
            )
            assert aggregate.read(
                result.hypothesis_refs[0], access=access) is None
    finally:
        restored_backend.close()


def test_v06_clone_owns_independent_forget_visibility():
    """评测 clone 可遗忘自己的逻辑视图，但不得共享或改变宿主 M-11 状态。"""
    backend = DictBackend()
    try:
        ctx = _context(backend)
        session = OwnerScope(9, 10, 11, VISIBILITY_SESSION)
        _ingest(ctx, _source(50, session), 50)
        access = MemoryAccessContext(9, 10, 11)
        assert len(ctx.memory_isolation_runtime.export(access).events) == 4

        with isolated_evaluation(ctx, label="m11-isolation") as cloned:
            assert cloned.memory_isolation_runtime is not ctx.memory_isolation_runtime
            assert cloned.memory_forget_visibility is not ctx.memory_forget_visibility
            assert len(cloned.memory_isolation_runtime.export(access).events) == 4
            cloned.memory_isolation_runtime.forget(
                _management(session, OWNER_SELECTION_EXACT),
                reason_key=(50,),
            )
            assert cloned.memory_isolation_runtime.export(access).events == ()

        assert len(ctx.memory_isolation_runtime.export(access).events) == 4
    finally:
        backend.close()
