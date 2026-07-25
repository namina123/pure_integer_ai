"""M-06 当前 typed 输入到 Memory activation request 的回归与对抗测试。"""
from __future__ import annotations

import pytest

from pure_integer_ai.cognition.shared.identity import (
    ObjectIdentity,
    OwnerScope,
    SourceRef,
    VISIBILITY_SESSION,
    VersionBundle,
    concept_identity,
    minimal_instruction_identity,
    occurrence_identity,
    span_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.semantic_object import (
    proposition_identity,
)
from pure_integer_ai.cognition.shared.memory_overlay import (
    MemoryAccessContext,
)
from pure_integer_ai.cognition.shared.memory_query import (
    MemoryCurrentQuery,
    MemoryQueryDefinition,
    MemoryQueryProtocol,
    MemoryQueryRoles,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    CLOCK_QUERY,
    LogicalClock,
    LogicalClockIdentity,
    document_scope,
    episode_scope,
    query_scope,
    session_scope,
)
from pure_integer_ai.experiments.evaluation_isolation import (
    isolated_evaluation,
)
from pure_integer_ai.experiments.memory_query_runtime import (
    install_memory_query_runtime,
)
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.storage.backend import DictBackend, SQLiteBackend
from pure_integer_ai.storage.edge_store import SOURCE_BARE_TEXT


def _source(document_id: int = 1) -> SourceRef:
    """构造带 session owner 的来源，使 ACL 对抗可观察。"""
    owner = OwnerScope(1, 2, 3, VISIBILITY_SESSION)
    return SourceRef(
        SOURCE_BARE_TEXT,
        6100,
        document_id,
        owner,
        VersionBundle(),
    )


def _open_query(ctx, source: SourceRef, *, local_id: int = 1):
    """打开 A-09 session/document/episode/query 生命周期并返回 query scope。"""
    session = session_scope(
        1,
        owner=source.owner,
        versions=source.versions,
        source=source,
    )
    document = document_scope(source, parent=session)
    episode = episode_scope(local_id, parent=document)
    query = query_scope(local_id, parent=episode)
    ctx.work_memory.begin_session(session)
    ctx.work_memory.begin_document(document)
    ctx.work_memory.begin_episode(episode)
    ctx.work_memory.begin_query(query)
    return query


def _close_query(ctx) -> None:
    """按逆序关闭测试打开的全部 WorkMemory 生命周期。"""
    ctx.work_memory.end_query()
    ctx.work_memory.end_episode()
    ctx.work_memory.end_document()
    ctx.work_memory.end_session()


def _instruction(source: SourceRef, value: int) -> ObjectIdentity:
    """构造协议注入的 MinimalInstruction 身份。"""
    return minimal_instruction_identity(
        (value,), owner=source.owner, versions=source.versions)


def _protocol(source: SourceRef) -> MemoryQueryProtocol:
    """构造六种注入 query kind，不在实现中写入领域语义枚举。"""
    roles = MemoryQueryRoles(*(
        _instruction(source, value)
        for value in range(7001, 7009)
    ))
    definitions = (
        MemoryQueryDefinition(
            _instruction(source, 7101), (7201,),
            (roles.occurrence, roles.span, roles.domain), 3),
        MemoryQueryDefinition(
            _instruction(source, 7102), (7202,),
            (roles.span, roles.structure), 5),
        MemoryQueryDefinition(
            _instruction(source, 7103), (7203,),
            (roles.semantic_object, roles.intent), 7),
        MemoryQueryDefinition(
            _instruction(source, 7104), (7204,),
            (roles.occurrence, roles.speaker), 11),
        MemoryQueryDefinition(
            _instruction(source, 7105), (7205,),
            (roles.task, roles.speaker), 13),
        MemoryQueryDefinition(
            _instruction(source, 7106), (7206,),
            (roles.task, roles.semantic_object), 17),
    )
    return MemoryQueryProtocol(roles, definitions)


def _current(ctx, source: SourceRef, scope, *, ordinal: int = 0):
    """物化当前 occurrence/span/S-02 对象/结构并构造完整 query 输入。"""
    ontology = ctx.graph_ontology
    occurrence = ontology.materialize(occurrence_identity(
        source,
        start=ordinal,
        end=ordinal + 1,
        ordinal=ordinal,
    ))
    span = ontology.materialize(span_identity(
        source,
        members=((ordinal, ordinal + 1),),
        ordinal=ordinal,
    ))
    semantic = ontology.materialize(proposition_identity(
        source, (7301, ordinal + 1)))
    structure = ontology.materialize(structure_concept_identity(
        (7302, ordinal + 1),
        owner=source.owner,
        versions=source.versions,
    ))
    timestamp = LogicalClock(
        LogicalClockIdentity(scope, CLOCK_QUERY)).advance()
    return MemoryCurrentQuery(
        scope,
        source,
        timestamp,
        (occurrence,),
        (span,),
        (semantic,),
        (structure,),
        concept_identity(
            (7303,), owner=source.owner, versions=source.versions),
        concept_identity(
            (7304,), owner=source.owner, versions=source.versions),
        concept_identity(
            (7305,), owner=source.owner, versions=source.versions),
        concept_identity(
            (7306,), owner=source.owner, versions=source.versions),
    )


@pytest.mark.parametrize("backend_type", [DictBackend, SQLiteBackend])
def test_current_typed_input_compiles_six_injected_requests_without_memory_read(
        backend_type):
    """当前 typed 输入产出六类独立 request，且编译器不读写 M-04 聚合。"""
    backend = backend_type()
    try:
        ctx = make_train_context(backend)
        source = _source()
        runtime = install_memory_query_runtime(
            ctx,
            _protocol(source),
            aggregates=ctx.memory_interact_aggregates,
        )
        scope = _open_query(ctx, source)
        current = _current(ctx, source, scope)
        before = backend.snapshot()

        def forbidden_aggregate_read(*args, **kwargs):
            """若 M-06 越界读取 aggregate，则立即使测试失败。"""
            del args, kwargs
            raise AssertionError("M-06 不得在编译期读取 aggregate")

        ctx.memory_interact_aggregates.query = forbidden_aggregate_read
        compiled = runtime.compile(
            current,
            access=MemoryAccessContext(1, 2, 3),
        )

        assert len(compiled.requests) == 6
        assert {item.hypothesis_kind for item in compiled.requests} == {
            (7201,), (7202,), (7203,), (7204,), (7205,), (7206,),
        }
        assert all(item.scope == scope for item in compiled.requests)
        assert all(item.source == source for item in compiled.requests)
        assert all(item.logical_timestamp == current.logical_timestamp
                   for item in compiled.requests)
        assert [item.budget for item in compiled.requests] == [3, 5, 7, 11, 13, 17]
        assert backend.snapshot() == before
    finally:
        if "ctx" in locals() and ctx.work_memory.active_query_scope is not None:
            _close_query(ctx)
        backend.close()


def test_current_input_and_scope_change_request_but_history_replay_does_not():
    """检索键只依赖当前输入和 scope，不受上一 episode 工作记忆残留控制。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        source = _source()
        runtime = install_memory_query_runtime(
            ctx,
            _protocol(source),
            aggregates=ctx.memory_interact_aggregates,
        )
        scope = _open_query(ctx, source)
        current = _current(ctx, source, scope)
        access = MemoryAccessContext(1, 2, 3)
        first = runtime.compile(current, access=access)
        ctx.work_memory.replay_candidates[:] = [(999,)]
        ctx.work_memory.pr_vector[(1, 2)] = 1000
        repeated = runtime.compile(current, access=access)
        assert repeated.stable_key() == first.stable_key()

        next_scope = query_scope(2, parent=scope.parent)
        changed = _current(ctx, source, next_scope, ordinal=1)
        direct = runtime.compiler.compile(changed, access=access)
        assert direct.stable_key() != first.stable_key()
        assert direct.requests[0].reasons[0].anchors != (
            first.requests[0].reasons[0].anchors)
    finally:
        if "ctx" in locals() and ctx.work_memory.active_query_scope is not None:
            _close_query(ctx)
        backend.close()


def test_runtime_rejects_wrong_scope_and_acl_without_memory_write():
    """生命周期 scope 或 ACL 不匹配时 fail closed，backend 状态保持不变。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        source = _source()
        runtime = install_memory_query_runtime(
            ctx,
            _protocol(source),
            aggregates=ctx.memory_interact_aggregates,
        )
        scope = _open_query(ctx, source)
        current = _current(ctx, source, scope)
        before = backend.snapshot()
        with pytest.raises(PermissionError, match="不可读取"):
            runtime.compile(current, access=MemoryAccessContext(1, 2, 4))
        assert backend.snapshot() == before

        wrong_scope = query_scope(2, parent=scope.parent)
        wrong = _current(ctx, source, wrong_scope)
        with pytest.raises(ValueError, match="活动 WorkMemory query"):
            runtime.compile(wrong, access=MemoryAccessContext(1, 2, 3))
        assert backend.snapshot() == before
    finally:
        if "ctx" in locals() and ctx.work_memory.active_query_scope is not None:
            _close_query(ctx)
        backend.close()


def test_v06_clone_rebinds_query_compiler_to_isolated_memory_aggregate():
    """V-06 clone 必须保留协议但不能复用宿主的 aggregate facade。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        source = _source()
        runtime = install_memory_query_runtime(
            ctx,
            _protocol(source),
            aggregates=ctx.memory_interact_aggregates,
        )
        with isolated_evaluation(ctx, label="m06-memory-query") as cloned:
            assert cloned.memory_query_runtime is not runtime
            assert (cloned.memory_query_runtime.compiler.aggregates
                    is cloned.memory_interact_aggregates)
            assert (cloned.memory_query_runtime.compiler.aggregates
                    is not runtime.compiler.aggregates)
            assert cloned.memory_query_runtime.state_key() == runtime.state_key()
    finally:
        backend.close()
