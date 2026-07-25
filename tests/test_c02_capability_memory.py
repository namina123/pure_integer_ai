"""C-02 Capability 进入 Memory 前的对象分型、失败归因和 typed trace 测试。"""
from __future__ import annotations

from dataclasses import replace

import pytest

from pure_integer_ai.cognition.shared.artifact_binding import (
    ArtifactBindingChoice,
    ArtifactBindingEndpoint,
    ArtifactBindingRequest,
    ArtifactBindingValue,
)
from pure_integer_ai.cognition.shared.capability_activation import (
    CapabilityActivationMapper,
    CapabilityObligationProjection,
)
from pure_integer_ai.cognition.shared.formal_artifact import (
    ArtifactSchema,
    artifact_identity,
)
from pure_integer_ai.cognition.shared.identity import (
    concept_identity,
    minimal_instruction_identity,
)
from pure_integer_ai.cognition.shared.memory_event import (
    MEMORY_EVENT_ARTIFACT,
    MEMORY_EVENT_CAPABILITY,
    MEMORY_EVENT_CAPABILITY_ATTEMPT_OUTCOME,
    MEMORY_EVENT_USE,
    MEMORY_OBJECT_ARTIFACT,
    MEMORY_OBJECT_CAPABILITY,
    MEMORY_OBJECT_HYPOTHESIS,
    ArtifactPayload,
    CapabilityAttemptOutcomePayload,
    CapabilityPayload,
    MemoryEvent,
    MemoryLinkedRef,
    memory_object_ref,
)
from pure_integer_ai.cognition.shared.capability_memory import (
    VerifiedCapabilityContract,
)
from pure_integer_ai.cognition.shared.capability_resolver import (
    CapabilityActivationAssessment,
    CapabilityResolverRoute,
)
from pure_integer_ai.cognition.shared.memory_batch import (
    FAULT_MEMORY_BATCH_AFTER_EVENT,
    install_memory_batch_runtimes,
)
from pure_integer_ai.cognition.shared.memory_overlay import MemoryAccessContext
from pure_integer_ai.cognition.shared.memory_query import (
    MemoryQueryDefinition,
    MemoryQueryProtocol,
)
from pure_integer_ai.cognition.shared.memory_resolver import (
    RESOLUTION_ORIGIN_MEMORY,
    ActivationScore,
    ActivationScoreReason,
    ResolvedCandidate,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    CLOCK_MEMORY_CREATED,
    CLOCK_MEMORY_USED,
    LogicalClock,
    LogicalClockIdentity,
    LogicalTimestamp,
    document_scope,
    episode_scope,
)
from pure_integer_ai.experiments.capability_memory_runtime import (
    CapabilityMemoryRuntime,
)
from pure_integer_ai.experiments.capability_execution_runtime import (
    CapabilityExecutionProtocol,
    CapabilityExecutionRuntime,
)
from pure_integer_ai.experiments.capability_verification_runtime import (
    CapabilityVerificationRuntime,
)
from pure_integer_ai.experiments.memory_query_runtime import (
    install_memory_query_runtime,
)
from pure_integer_ai.experiments.evaluation_isolation import isolated_evaluation
from pure_integer_ai.experiments.memory_use_runtime import (
    install_memory_use_runtime,
)
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.storage.backend import DictBackend, SQLiteBackend
from pure_integer_ai.storage.memory_batch import (
    MEMORY_BATCH_SOURCE_DEPENDENCY_KEY,
)
from pure_integer_ai.storage.segment_dependency import SegmentDependency
from pure_integer_ai.cognition.understanding.struct_bind_typed_adapter import (
    TypedStructBindEndpoint,
)

from test_c01_capability_verification import (
    _candidate,
    _held_out,
)
from test_a06_artifact_binding_runtime import _runtime_case
from test_a10_attractor_state import (
    _core_refs as _attractor_core_refs,
    _current as _attractor_current,
    _goals as _attractor_goals,
    _install_a10,
    _install_resolver,
    _memory_protocol,
    _setup as _attractor_setup,
)
from test_m03_memory_event import (
    _append_observation,
    _core_refs,
    _scopes,
    _source,
    _timestamp,
)
from test_m10_memory_batch_recovery import (
    _FailOnce,
    _config,
    _core_dependency,
)
from test_s06_formal_artifact import _artifact, _bridge, _case
from test_m06_memory_query import (
    _close_query,
    _current,
    _open_query,
    _protocol,
)


_ACCESS = MemoryAccessContext(1, 2, 3)


class _CapabilityScorer:
    """测试用纯整数评分器，记录 route 传入的目标局部使用统计。"""

    def __init__(self):
        """初始化空调用记录。"""
        self.calls = []

    def assess(self, request, capability, usage):
        """按成功减失败形成整数分，并注入能力类型竞争键。"""
        self.calls.append((request, capability, usage))
        value = 100 + usage.use_count * 10 - usage.failure_count * 20
        return CapabilityActivationAssessment(
            request.hypothesis_kind,
            ActivationScore(
                value,
                (ActivationScoreReason((26000, 1), value),),
            ),
        )

    def clone_for_context(self, ctx):
        """为 V-06 返回无共享调用记录的同协议评分器。"""
        del ctx
        return _CapabilityScorer()

    def state_key(self):
        """返回测试评分协议版本。"""
        return (1, 26000)


class _FirstCapabilityObligation:
    """测试 selector：显式选择调用方传入的第一个当前义务。"""

    def select(self, request, candidate, obligations):
        """只返回既有首项义务，不从 Capability 伪造新命题。"""
        del request, candidate
        return (CapabilityObligationProjection(obligations[0], 0),)

    def clone_for_context(self, ctx):
        """返回无状态同协议 selector。"""
        del ctx
        return _FirstCapabilityObligation()

    def state_key(self):
        """返回 selector 版本。"""
        return (1, 26001)


def _verified_report():
    """执行真实 C-01 held-out 并返回明确 verified 报告。"""
    candidate = _candidate()
    return CapabilityVerificationRuntime(_bridge(_case())).verify(
        candidate, _held_out(candidate))


def test_verified_contract_requires_nonempty_provenance_fingerprint():
    """直接构造也不得绕过 verified provenance 内容引用。"""
    contract = VerifiedCapabilityContract.from_report(_verified_report())

    with pytest.raises(ValueError, match="verified provenance"):
        replace(contract, provenance_ref=())


def test_capability_execution_protocol_stable_key_has_segment_boundaries():
    """执行协议的三个开放身份必须分别带长度，避免拼接歧义。"""
    source = _source()
    protocol = CapabilityExecutionProtocol(
        minimal_instruction_identity(
            (26039, 1), owner=source.owner, versions=source.versions),
        MemoryLinkedRef.object(concept_identity(
            (26039, 2), owner=source.owner, versions=source.versions)),
        MemoryLinkedRef.object(concept_identity(
            (26039, 3), owner=source.owner, versions=source.versions)),
    )
    fields = (
        protocol.consumer.stable_key(),
        protocol.influence_kind.stable_key(),
        protocol.failure_outcome_kind.stable_key(),
    )
    expected = [1]
    for field in fields:
        expected.extend((len(field), *field))

    assert protocol.stable_key() == tuple(expected)


def _source_dependency(batch_id: int) -> SegmentDependency:
    """构造绑定当前测试批次的完整来源依赖。"""
    return SegmentDependency(
        MEMORY_BATCH_SOURCE_DEPENDENCY_KEY,
        (26040, batch_id),
        (26041, batch_id),
    )


def _created_at(report, seq: int = 1):
    """在 candidate 来源的 episode scope 内构造创建逻辑时间。"""
    source = report.candidate.proposal.source
    episode = episode_scope(26042, parent=document_scope(source))
    return LogicalClock(
        LogicalClockIdentity(episode, CLOCK_MEMORY_CREATED), seq - 1,
    ).advance()


def _capability_binding_request(
        binding_runtime,
        original: ArtifactBindingRequest,
        definition,
        proposition,
        scope,
        ) -> ArtifactBindingRequest:
    """用现有真实 STRUCT_BIND 槽位为恢复后的 Capability definition 重建请求。"""
    source_endpoints = (original.endpoints[0], original.endpoints[2])
    target_endpoints = tuple(
        ArtifactBindingEndpoint(
            original.endpoints[index * 2 + 1].slot_ref,
            parameter.variable,
        )
        for index, parameter in enumerate(definition.parameters)
    )
    endpoints = tuple(
        item
        for pair in zip(source_endpoints, target_endpoints)
        for item in pair
    )
    typed_endpoints = tuple(
        TypedStructBindEndpoint(item.slot_ref, item.variable)
        for item in endpoints
    )
    correspondences = tuple(
        binding_runtime.adapter.read_from(
            binding_runtime.edge_store,
            source.slot_ref,
            typed_endpoints,
        ).correspondences[0]
        for source in source_endpoints
    )
    choices = tuple(
        ArtifactBindingChoice(
            source_endpoints[index],
            target_endpoints[index],
            correspondences[index].stable_key(),
            original.choices[index].reason,
            (26090, index + 1),
        )
        for index in range(len(target_endpoints))
    )
    values = tuple(
        ArtifactBindingValue(
            source_endpoints[index],
            _artifact(
                original.source,
                scope,
                value.artifact.artifact_kind,
                value.artifact.schema,
                26110 + index,
                value.artifact.payload,
            ),
        )
        for index, value in enumerate(original.values)
    )
    expected = None
    if original.expected is not None:
        expected = _artifact(
            original.source,
            scope,
            original.expected.artifact_kind,
            original.expected.schema,
            26119,
            original.expected.payload,
        )
    return ArtifactBindingRequest(
        proposition,
        definition,
        original.source,
        scope,
        (26091, 1),
        endpoints,
        values,
        choices,
        expected,
    )


def _execution_setup(*, failing: bool):
    """组装真实 M-06/M-07/A-10/A-06/M-08 Capability 纵切。"""
    expected = (6, 1) if failing else (5, 1)
    backend, ctx, formal, binding_runtime, original = _runtime_case(
        expected_payload=expected)
    source = formal["source"]
    _close_query(ctx)
    scope = _open_query(ctx, source)
    query_runtime, _ = _install_resolver(
        ctx,
        _memory_protocol(source),
        _attractor_core_refs(ctx)[1],
    )
    attractor_runtime, _ = _install_a10(
        ctx,
        source,
        prefer_matching_document=False,
        max_agenda_entries=2,
        max_consumptions=2,
    )
    compilation = query_runtime.compile(
        _attractor_current(ctx, source, scope),
        access=MemoryAccessContext(0, 0, 0),
    )
    goals = _attractor_goals(source, scope)
    install_memory_batch_runtimes(ctx, _config(_core_dependency(26100)))
    report = _verified_report()
    assert report.candidate.proposal.source == source
    published = CapabilityMemoryRuntime(ctx).publish_verified(
        report,
        batch_id=26101,
        source_dependency=_source_dependency(26101),
        created_at=_created_at(report),
    )
    scorer = _CapabilityScorer()
    route = CapabilityResolverRoute(ctx.memory_interact_events, scorer)
    ctx.memory_resolver_runtime.register_route(route)
    mapper = CapabilityActivationMapper(
        _FirstCapabilityObligation(),
        activation_kind=minimal_instruction_identity(
            (26102, 1), owner=source.owner, versions=source.versions),
        score_reason=minimal_instruction_identity(
            (26102, 2), owner=source.owner, versions=source.versions),
        capability_dependency_role=minimal_instruction_identity(
            (26102, 3), owner=source.owner, versions=source.versions),
    )
    attractor_runtime.register_mapper_route(mapper)
    capability_request = replace(
        compilation.requests[0],
        hypothesis_kind=report.candidate.proposal.capability_kind.stable_key(),
        budget=1,
        memory_object_kind=MEMORY_OBJECT_CAPABILITY,
    )
    capability_compilation = replace(
        compilation, requests=(capability_request,))
    state = attractor_runtime.resolve_and_activate(
        capability_compilation, (goals[0],))
    request = _capability_binding_request(
        binding_runtime,
        original,
        report.candidate.proposal.definition,
        goals[0].proposition.template,
        scope,
    )
    observation = _append_observation(ctx, source, _core_refs(ctx))
    memory_use = install_memory_use_runtime(ctx)
    execution = CapabilityExecutionRuntime(
        ctx,
        binding_runtime,
        memory_use,
        CapabilityExecutionProtocol(
            minimal_instruction_identity(
                (26103, 1), owner=source.owner, versions=source.versions),
            MemoryLinkedRef.object(concept_identity(
                (26103, 2), owner=source.owner, versions=source.versions)),
            MemoryLinkedRef.object(concept_identity(
                (26103, 3), owner=source.owner, versions=source.versions)),
        ),
    )
    used_at = LogicalTimestamp(state.current_timestamp.clock, 2)
    capability_scope = published.recovered.capability_event.event.scope
    failed_at = LogicalTimestamp(
        LogicalClockIdentity(capability_scope, CLOCK_MEMORY_USED), 1)
    return (
        backend,
        ctx,
        route,
        capability_request,
        execution,
        request,
        observation.event.object_ref,
        used_at,
        failed_at,
        published,
    )


def _append_capability(ctx, source):
    """追加最小 Artifact 与 Capability 声明，返回能力事件及声明 scope。"""
    refs = _core_refs(ctx)
    observation = _append_observation(ctx, source, refs)
    _, _, episode = _scopes(source)
    artifact_kind = concept_identity(
        (26002, 1), owner=source.owner, versions=source.versions)
    value_type = concept_identity(
        (26003, 1), owner=source.owner, versions=source.versions)
    unit = concept_identity(
        (26004, 1), owner=source.owner, versions=source.versions)
    artifact = artifact_identity(
        source,
        artifact_kind,
        ArtifactSchema(value_type, unit),
        (26005, 1),
        (26006, 1),
        episode,
    )
    artifact_payload = ArtifactPayload(
        artifact,
        observation.event.object_ref,
        _timestamp(episode, CLOCK_MEMORY_CREATED, 4),
    )
    artifact_ref = memory_object_ref(
        ctx.memory_interact_events.memory_space_identity,
        MEMORY_OBJECT_ARTIFACT,
        artifact.stable_key(),
        owner=source.owner,
        versions=source.versions,
    )
    ctx.memory_interact_events.append(MemoryEvent(
        MEMORY_EVENT_ARTIFACT,
        artifact_ref,
        episode,
        artifact_payload,
    ))
    capability = CapabilityPayload(
        MemoryLinkedRef.core(refs[0]),
        artifact_ref,
        (26001, 1),
        (),
        _timestamp(episode, CLOCK_MEMORY_CREATED, 5),
    )
    capability_ref = memory_object_ref(
        ctx.memory_interact_events.memory_space_identity,
        MEMORY_OBJECT_CAPABILITY,
        capability.stable_key(),
        owner=source.owner,
        versions=source.versions,
    )
    event = ctx.memory_interact_events.append(MemoryEvent(
        MEMORY_EVENT_CAPABILITY,
        capability_ref,
        episode,
        capability,
    ))
    return event, episode


def test_m06_defaults_to_hypothesis_and_carries_injected_capability_kind():
    """旧定义保持 Hypothesis；Capability 定义显式编译对象种类和类型键。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        source = _source()
        base = _protocol(source)
        definition = MemoryQueryDefinition(
            minimal_instruction_identity(
                (26010, 1), owner=source.owner, versions=source.versions),
            (26011, 1),
            (base.roles.task,),
            2,
            MEMORY_OBJECT_CAPABILITY,
        )
        protocol = MemoryQueryProtocol(base.roles, (definition,))
        runtime = install_memory_query_runtime(
            ctx, protocol, aggregates=ctx.memory_interact_aggregates)
        scope = _open_query(ctx, source)
        compiled = runtime.compile(
            _current(ctx, source, scope), access=_ACCESS)

        assert all(item.memory_object_kind == MEMORY_OBJECT_HYPOTHESIS
                   for item in base.definitions)
        assert len(compiled.requests) == 1
        assert compiled.requests[0].memory_object_kind == (
            MEMORY_OBJECT_CAPABILITY)
        assert compiled.requests[0].hypothesis_kind == (26011, 1)
        assert compiled.requests[0].stable_key() != replace(
            compiled.requests[0],
            memory_object_kind=MEMORY_OBJECT_HYPOTHESIS,
        ).stable_key()
    finally:
        if "ctx" in locals() and ctx.work_memory.active_query_scope is not None:
            _close_query(ctx)
        backend.close()


@pytest.mark.parametrize("backend_type", [DictBackend, SQLiteBackend])
def test_capability_attempt_outcome_round_trips_without_becoming_use(
        backend_type):
    """失败尝试只追加到目标 Capability，双后端恢复后仍保持独立事件种类。"""
    backend = backend_type()
    try:
        ctx = make_train_context(backend)
        source = _source()
        capability_event, episode = _append_capability(ctx, source)
        query_kind = minimal_instruction_identity(
            (26020, 1), owner=source.owner, versions=source.versions)
        outcome_kind = concept_identity(
            (26021, 1), owner=source.owner, versions=source.versions)
        payload = CapabilityAttemptOutcomePayload(
            capability_event.event.object_ref,
            MemoryLinkedRef.object(query_kind),
            (26022, 1),
            (26023, 1),
            MemoryLinkedRef.object(outcome_kind),
            None,
            _timestamp(episode, CLOCK_MEMORY_USED, 6),
        )
        appended = ctx.memory_interact_events.append(MemoryEvent(
            MEMORY_EVENT_CAPABILITY_ATTEMPT_OUTCOME,
            capability_event.event.object_ref,
            episode,
            payload,
        ))
        ctx.memory_interact_events.clear_runtime_caches()
        restored = ctx.memory_interact_events.read(
            appended.event_hash, access=_ACCESS)

        assert restored is not None
        assert restored.event == appended.event
        assert restored.event.is_declaration is False
        assert restored.event.payload.target_ref == (
            capability_event.event.object_ref)
        assert ctx.memory_interact_events.query(
            access=_ACCESS,
            event_kind=MEMORY_EVENT_CAPABILITY_ATTEMPT_OUTCOME,
        ) == (restored,)
    finally:
        backend.close()


def test_capability_resolved_candidate_has_no_fake_hypothesis_fields():
    """Capability typed 结果保留声明，不允许伪造 aggregate、Hypothesis 或来源分账。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend)
        source = _source()
        capability_event, episode = _append_capability(ctx, source)
        capability = capability_event.event.payload
        query = _open_query(ctx, source)
        candidate = ResolvedCandidate(
            RESOLUTION_ORIGIN_MEMORY,
            capability_event.event.object_ref.stable_key(),
            capability.capability_kind.stable_key(),
            query,
            episode,
            None,
            capability_event.event.object_ref,
            None,
            None,
            (source,),
            (),
            10,
            (ActivationScoreReason((26030, 1), 10),),
            None,
            capability,
        )

        assert candidate.capability == capability
        assert candidate.hypothesis is None
        assert candidate.aggregate is None
        assert candidate.memory_source_traces == ()
        with pytest.raises(ValueError, match="不得伪造 Hypothesis"):
            replace(candidate, diversity=object())
    finally:
        if "ctx" in locals() and ctx.work_memory.active_query_scope is not None:
            _close_query(ctx)
        backend.close()


@pytest.mark.parametrize("backend_type", [DictBackend, SQLiteBackend])
def test_verified_contract_publishes_program_and_capability_atomically(
        backend_type):
    """C-01 verified 报告经同一 batch 发布，清缓存后仍恢复完整 definition。"""
    backend = backend_type()
    try:
        ctx = make_train_context(backend, companion=True)
        install_memory_batch_runtimes(ctx, _config(_core_dependency(26050)))
        runtime = CapabilityMemoryRuntime(ctx)
        report = _verified_report()
        result = runtime.publish_verified(
            report,
            batch_id=26051,
            source_dependency=_source_dependency(26051),
            created_at=_created_at(report),
        )
        runtime.event_log.clear_runtime_caches()
        recovered = runtime.recover(
            result.capability_ref,
            access=MemoryAccessContext(0, 0, 0),
        )

        assert len(result.batch.materialized) == 2
        assert tuple(item.event.event_kind for item in result.batch.materialized) == (
            MEMORY_EVENT_ARTIFACT, MEMORY_EVENT_CAPABILITY)
        assert recovered.definition == report.candidate.proposal.definition
        assert recovered.contract.state == report.candidate.status_protocol.verified
        assert recovered.contract.report_key == report.stable_key()
        assert recovered.payload.program_ref == recovered.artifact_event.event.object_ref
    finally:
        backend.close()


def test_half_written_capability_is_hidden_until_batch_roll_forward():
    """program 物理写入后中断时两项都不可见，恢复 activation 后才同时出现。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend, companion=True)
        install_memory_batch_runtimes(ctx, _config(_core_dependency(26060)))
        runtime = CapabilityMemoryRuntime(ctx)
        report = _verified_report()
        dependency = _source_dependency(26061)
        with pytest.raises(RuntimeError, match="M-10 fault"):
            runtime.publish_verified(
                report,
                batch_id=26061,
                source_dependency=dependency,
                created_at=_created_at(report),
                fault_injector=_FailOnce(
                    FAULT_MEMORY_BATCH_AFTER_EVENT, event_ordinal=0),
            )
        assert runtime.event_log.query(
            access=MemoryAccessContext(0, 0, 0),
            event_kind=MEMORY_EVENT_ARTIFACT,
        ) == ()
        assert runtime.event_log.query(
            access=MemoryAccessContext(0, 0, 0),
            event_kind=MEMORY_EVENT_CAPABILITY,
        ) == ()

        completed = runtime.batch_runtime.recover_unit(
            report.candidate.proposal.source,
            26061,
            source_dependency=dependency,
        )
        assert completed is not None
        assert len(runtime.event_log.query(
            access=MemoryAccessContext(0, 0, 0),
            event_kind=MEMORY_EVENT_ARTIFACT,
        )) == 1
        assert len(runtime.event_log.query(
            access=MemoryAccessContext(0, 0, 0),
            event_kind=MEMORY_EVENT_CAPABILITY,
        )) == 1
    finally:
        backend.close()


def test_verified_contract_rejects_state_or_provenance_drift():
    """C-00 原契约和篡改 verified provenance 都不能伪装为 C-02 contract。"""
    report = _verified_report()
    contract = VerifiedCapabilityContract.from_report(report)
    with pytest.raises(ValueError, match="版本"):
        VerifiedCapabilityContract.from_stable_key(
            report.candidate.proposal.contract_key())
    drift = (*contract.stable_key()[:-1], contract.stable_key()[-1] + 1)
    with pytest.raises(ValueError, match="provenance"):
        VerifiedCapabilityContract.from_stable_key(drift)


def test_sqlite_restart_recovers_same_verified_capability(tmp_path):
    """关闭并重开 SQLite 后，batch receipt 与两项声明仍恢复为同一能力。"""
    path = str(tmp_path / "c02-capability.sqlite3")
    config = _config(_core_dependency(26070))
    report = _verified_report()
    first_backend = SQLiteBackend(path)
    try:
        first_ctx = make_train_context(first_backend, companion=True)
        install_memory_batch_runtimes(first_ctx, config)
        published = CapabilityMemoryRuntime(first_ctx).publish_verified(
            report,
            batch_id=26071,
            source_dependency=_source_dependency(26071),
            created_at=_created_at(report),
        )
        capability_ref = published.capability_ref
    finally:
        first_backend.close()

    second_backend = SQLiteBackend(path)
    try:
        second_ctx = make_train_context(second_backend, companion=True)
        install_memory_batch_runtimes(second_ctx, config)
        recovered = CapabilityMemoryRuntime(second_ctx).recover(
            capability_ref,
            access=MemoryAccessContext(0, 0, 0),
        )
        assert recovered.definition == report.candidate.proposal.definition
        assert recovered.contract.report_key == report.stable_key()
    finally:
        second_backend.close()


def test_registered_capability_route_preserves_hypothesis_results_and_resolves():
    """未注册时 fail closed；注册后召回能力且原 Hypothesis 结果 bit-identical。"""
    setup = _attractor_setup()
    backend, ctx, source, _, _, compilation, _ = setup
    try:
        before = ctx.memory_resolver_runtime.resolve(compilation)
        install_memory_batch_runtimes(ctx, _config(_core_dependency(26080)))
        report = _verified_report()
        published = CapabilityMemoryRuntime(ctx).publish_verified(
            report,
            batch_id=26081,
            source_dependency=_source_dependency(26081),
            created_at=_created_at(report),
        )
        capability_request = replace(
            compilation.requests[0],
            hypothesis_kind=report.candidate.proposal.capability_kind.stable_key(),
            budget=1,
            memory_object_kind=MEMORY_OBJECT_CAPABILITY,
        )
        capability_compilation = replace(
            compilation, requests=(capability_request,))
        with pytest.raises(ValueError, match="未安装 resolver route"):
            ctx.memory_resolver_runtime.resolve(capability_compilation)

        scorer = _CapabilityScorer()
        ctx.memory_resolver_runtime.register_route(CapabilityResolverRoute(
            ctx.memory_interact_events, scorer))
        after = ctx.memory_resolver_runtime.resolve(compilation)
        resolved = ctx.memory_resolver_runtime.resolve(capability_compilation)

        assert after.stable_key() == before.stable_key()
        assert len(resolved.sets) == 1
        assert resolved.sets[0].considered_count == 1
        assert resolved.sets[0].candidates[0].memory_ref == (
            published.capability_ref)
        assert resolved.sets[0].candidates[0].capability == (
            published.recovered.payload)
        assert resolved.sets[0].candidates[0].hypothesis is None
        assert scorer.calls[0][2].stable_key() == (0, 0, 0, 0)
    finally:
        _close_query(ctx)
        backend.close()


def test_capability_frontier_success_consumes_then_writes_exact_use():
    """真实 A-06 成功后 A-10 才 consumed，并为目标 Capability 写唯一 M-08 Use。"""
    setup = _execution_setup(failing=False)
    (backend, ctx, route, capability_request, execution, request,
     observation_ref, used_at, failed_at, published) = setup
    try:
        result = execution.execute_frontier(
            request,
            input_observation_ref=observation_ref,
            used_at=used_at,
            failed_at=failed_at,
        )

        assert result.binding_run.succeeded is True
        assert result.processing.decision.disposition == (
            ctx.attractor_runtime.protocol.consumed)
        assert result.use is not None
        assert result.use.use.event.payload.memory_ref == published.capability_ref
        assert result.failure is None
        assert len(ctx.memory_interact_events.query(
            access=MemoryAccessContext(0, 0, 0),
            event_kind=MEMORY_EVENT_USE,
        )) == 1
        assert ctx.memory_interact_events.query(
            access=MemoryAccessContext(0, 0, 0),
            event_kind=MEMORY_EVENT_CAPABILITY_ATTEMPT_OUTCOME,
        ) == ()
        route.resolve(capability_request)
        assert route.score_provider.calls[-1][2].stable_key() == (1, 0, 2, 0)
    finally:
        _close_query(ctx)
        backend.close()


def test_capability_formal_failure_suspends_and_only_updates_target():
    """verifier 明确拒绝时零 Use，A-10 suspended，并只追加目标能力失败结果。"""
    setup = _execution_setup(failing=True)
    (backend, ctx, route, capability_request, execution, request,
     observation_ref, used_at, failed_at, published) = setup
    try:
        result = execution.execute_frontier(
            request,
            input_observation_ref=observation_ref,
            used_at=used_at,
            failed_at=failed_at,
        )

        assert result.binding_run.succeeded is False
        assert result.binding_run.result.verification is not None
        assert result.binding_run.result.verification.accepted is False
        assert result.processing.decision.disposition == (
            ctx.attractor_runtime.protocol.suspended)
        assert result.use is None
        assert result.failure is not None
        assert result.failure.event.object_ref == published.capability_ref
        assert ctx.memory_interact_events.query(
            access=MemoryAccessContext(0, 0, 0),
            event_kind=MEMORY_EVENT_USE,
        ) == ()
        failures = ctx.memory_interact_events.query(
            access=MemoryAccessContext(0, 0, 0),
            event_kind=MEMORY_EVENT_CAPABILITY_ATTEMPT_OUTCOME,
        )
        assert failures == (result.failure,)
        route.resolve(capability_request)
        assert route.score_provider.calls[-1][2].stable_key() == (0, 1, 0, 1)
    finally:
        _close_query(ctx)
        backend.close()


def test_capability_definition_drift_stops_before_a06_and_a10_commit():
    """恢复 definition 被替换时不得执行、提交 processing、Use 或失败修正。"""
    setup = _execution_setup(failing=False)
    (backend, ctx, _, _, execution, request,
     observation_ref, used_at, failed_at, _) = setup
    try:
        drift_definition = replace(
            request.definition,
            verifier=request.definition.executor,
        )
        drift = replace(request, definition=drift_definition)
        with pytest.raises(ValueError, match="恢复 Capability 漂移"):
            execution.execute_frontier(
                drift,
                input_observation_ref=observation_ref,
                used_at=used_at,
                failed_at=failed_at,
            )

        state = ctx.work_memory.require_attractor_state()
        assert state.processing_traces() == ()
        assert ctx.work_memory.query_artifact_results == []
        assert ctx.memory_interact_events.query(
            access=MemoryAccessContext(0, 0, 0),
            event_kind=MEMORY_EVENT_USE,
        ) == ()
        assert ctx.memory_interact_events.query(
            access=MemoryAccessContext(0, 0, 0),
            event_kind=MEMORY_EVENT_CAPABILITY_ATTEMPT_OUTCOME,
        ) == ()
    finally:
        _close_query(ctx)
        backend.close()


def test_capability_routes_clone_into_v06_without_host_write():
    """V-06 重绑 Capability resolver/mapper/M-08，进入和退出均不改宿主。"""
    setup = _execution_setup(failing=False)
    (backend, ctx, _, _, _, _, _, _, _, _) = setup
    try:
        host_before = backend.snapshot()
        with isolated_evaluation(ctx, label="c02-capability-routes") as clone:
            assert clone.memory_resolver_runtime is not (
                ctx.memory_resolver_runtime)
            assert clone.memory_resolver_runtime.state_key() == (
                ctx.memory_resolver_runtime.state_key())
            assert clone.attractor_runtime is not ctx.attractor_runtime
            assert clone.attractor_runtime.state_key() == (
                ctx.attractor_runtime.state_key())
            assert clone.memory_use_runtime is not ctx.memory_use_runtime
            assert clone.memory_use_runtime.state_key() == (
                ctx.memory_use_runtime.state_key())
        assert backend.snapshot() == host_before
    finally:
        _close_query(ctx)
        backend.close()
