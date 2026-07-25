"""A-05 来源准入、来源簇计数、M-10 回滚与 K-04 恢复对抗。"""
from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from pure_integer_ai.crosscut.determinism.fingerprint import (
    integer_tuple_fingerprint,
)
from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EvidenceRecord,
    HypothesisKey,
    HypothesisLedger,
)
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
    TypedRef,
    VISIBILITY_SESSION,
    VersionBundle,
)
from pure_integer_ai.cognition.shared.memory_aggregate import (
    MEMORY_EVIDENCE_CONFLICTED,
    MEMORY_EVIDENCE_CORROBORATED,
)
from pure_integer_ai.cognition.shared.memory_batch import (
    FAULT_MEMORY_GROUP_AFTER_UNIT,
    install_memory_batch_runtimes,
)
from pure_integer_ai.cognition.shared.memory_event import MemoryLinkedRef
from pure_integer_ai.cognition.shared.memory_hot_set import (
    decode_memory_candidate,
    encode_memory_candidate,
)
from pure_integer_ai.cognition.shared.memory_hypothesis import (
    MemoryHypothesisEventSink,
)
from pure_integer_ai.cognition.shared.memory_overlay import MemoryAccessContext
from pure_integer_ai.cognition.shared.memory_resolver import MemoryCandidateBundle
from pure_integer_ai.cognition.shared.memory_resolver_engine import (
    MemoryOverlayResolver,
)
from pure_integer_ai.cognition.shared.post_weaning import PostWeaningIntakeRequest
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.cognition.shared.source_trust import (
    SOURCE_ADMISSION_ACCEPTED,
    SOURCE_ADMISSION_REJECTED,
    SourceTrustAssessment,
)
from pure_integer_ai.cognition.understanding.memory_intake import (
    HypothesisIntakeDraft,
    ObservationIntakeDraft,
)
from pure_integer_ai.experiments.evaluation_isolation import (
    clone_backend,
    clone_train_context,
)
from pure_integer_ai.experiments.post_weaning_runtime import CoreCanonicalStateReader
from pure_integer_ai.experiments.source_trust_runtime import (
    SourceAdmissionError,
    SourceAdmissionRejected,
    install_source_admission_runtime,
)
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.storage.backend import DictBackend, SQLiteBackend
from pure_integer_ai.storage.memory_event import MEMORY_EVENT_TABLE
from pure_integer_ai.storage.source_record import SOURCE_RECORD_TABLE
from pure_integer_ai.storage.source_trust import SOURCE_TRUST_ASSESSMENT_TABLE
from pure_integer_ai.storage.spaces.companion import TEXT_ASSOC_TABLE
from pure_integer_ai.storage.telemetry import collect_backend_telemetry
from pure_integer_ai.experiments.v02_run_store import canonical_json_bytes

from test_k04_memory_hot_set import _batch_config
from test_m03_memory_event import _core_refs
from test_m04_memory_aggregate import _TwoSourcePolicy
from test_m06_memory_query import (
    _close_query,
    _current,
    _open_query,
    _source as _query_source,
)
from test_m07_memory_resolver import (
    _BaselineProvider,
    _CurrentContextScorer,
    _DistinctSourcePolicy,
    _IndexFilterProvider,
    _protocol,
)


def _versions() -> VersionBundle:
    """返回 A-05 对抗使用的固定版本束。"""
    return VersionBundle(
        CorpusVersion(1),
        ParserVersion(1),
        PrimitiveVersion(1),
        CurriculumVersion(1),
    )


def _source(
        source_id: int,
        *,
        document_id: int | None = None,
        source_kind: int = 71,
        owner: OwnerScope | None = None,
        ) -> SourceRef:
    """构造可独立改变完整文档和来源 kind 的全局 SourceRef。"""
    return SourceRef(
        source_kind,
        source_id,
        source_id if document_id is None else document_id,
        OwnerScope() if owner is None else owner,
        _versions(),
    )


def _route(value: int) -> ObjectIdentity:
    """构造调用方注入的一等摄入 route。"""
    return ObjectIdentity(
        OBJECT_MINIMAL_INSTRUCTION, (value,), OwnerScope(), _versions())


class _Parser:
    """把来源切片转为一个不写 Core 的 Memory 候选。"""

    def __init__(self, source: SourceRef, candidate: int) -> None:
        """绑定预期来源和开放候选身份。"""
        self.source = source
        self.candidate = candidate

    def parse(self, source_slice):
        """核验 SourceRef 后返回最小 Observation/Hypothesis 草案。"""
        if source_slice.source != self.source:
            raise ValueError("A-05 parser 收到其他来源")
        context = MemoryLinkedRef.object(ObjectIdentity(
            OBJECT_CONTEXT_SCOPE,
            (21000 + self.candidate,),
            self.source.owner,
            self.source.versions,
        ))
        signal = MemoryLinkedRef.object(ObjectIdentity(
            OBJECT_MINIMAL_INSTRUCTION,
            (22000 + self.candidate,),
            self.source.owner,
            self.source.versions,
        ))
        return ObservationIntakeDraft(
            (23000 + self.candidate,),
            context,
            hypotheses=(HypothesisIntakeDraft(
                (7201,),
                (self.candidate,),
                (24000,),
                (25000 + self.candidate,),
                EVIDENCE_SUPPORT,
                signal_ref=signal,
            ),),
        )


class _SourcePolicy:
    """受控测试 policy，把开放来源映射到注入图身份和来源簇。"""

    def __init__(
            self,
            refs: tuple[TypedRef, ...],
            clusters: dict[tuple[int, ...], tuple[int, ...]] | None = None,
            ) -> None:
        """绑定图内类别以及按完整 SourceRef 注入的来源簇。"""
        self.refs = refs
        self.clusters = clusters or {}

    def state_key(self) -> tuple[int, ...]:
        """返回 refs 与来源簇配置组成的完整版本化状态。"""
        result = [1, len(self.refs)]
        for ref in self.refs:
            key = ref.stable_key()
            result.extend((len(key), *key))
        cluster_state = [len(self.clusters)]
        for source_key, cluster_key in sorted(self.clusters.items()):
            cluster_state.extend((len(source_key), *source_key))
            cluster_state.extend((len(cluster_key), *cluster_key))
        cluster_ref = integer_tuple_fingerprint(
            tuple(cluster_state),
            domain="tests.a05.source_policy.clusters.v1",
        )
        result.extend((len(cluster_ref), *cluster_ref))
        return tuple(result)

    def assess(self, request) -> SourceTrustAssessment:
        """按许可、source kind 和异常 source id 给出确定性准入裁决。"""
        accepted = (
            request.license_id == "license-ok"
            and request.source.source_kind == 71
            and request.source.source_id != 999
        )
        decision = (
            SOURCE_ADMISSION_ACCEPTED if accepted
            else SOURCE_ADMISSION_REJECTED)
        cluster = self.clusters.get(
            request.source.stable_key(), request.source.stable_key())
        return SourceTrustAssessment(
            request.stable_key(),
            self.state_key(),
            decision,
            cluster,
            (
                request.source.versions.corpus.value,
                request.source.versions.parser.value,
            ),
            self.refs[0],
            self.refs[1],
            self.refs[2],
            (self.refs[3],),
            () if accepted else (self.refs[4],),
            (26000, decision),
        )

    def clone_for_context(self, ctx):
        """回读克隆图中的全部依据并返回无可变运行态副本。"""
        for ref in self.refs:
            ctx.graph_ontology.identity_of(ref)
        return _SourcePolicy(self.refs, dict(self.clusters))


def _install(
        ctx,
        *,
        clusters: dict[tuple[int, ...], tuple[int, ...]] | None = None,
        policy: _SourcePolicy | None = None,
        ):
    """安装 M-10、物化 policy 图身份并返回 A-05 runtime 与三类 route。"""
    if ctx.memory_batch_coordinator is None:
        install_memory_batch_runtimes(ctx, _batch_config())
    routes = (_route(201), _route(202), _route(203))
    if policy is None:
        from pure_integer_ai.cognition.shared.graph_ontology import (
            relation_concept_identity,
        )
        refs = tuple(
            ctx.graph_ontology.materialize(
                relation_concept_identity((27000 + index,)))
            for index in range(5)
        )
        policy = _SourcePolicy(refs, clusters)
    runtime = install_source_admission_runtime(
        ctx,
        policy,
        reading_route=routes[0],
        interaction_route=routes[1],
        record_only_routes=(routes[2],),
    )
    return runtime, routes, policy


def _request(
        route: ObjectIdentity,
        source: SourceRef,
        *,
        batch_id: int,
        license_id: str = "license-ok",
        parser=None,
        raw_text: str = "来源正文",
        ) -> PostWeaningIntakeRequest:
    """构造带完整许可、批次和 trace 的准入请求。"""
    return PostWeaningIntakeRequest(
        route,
        source,
        raw_text,
        license_id,
        batch_id,
        parser=parser,
        trace=(28000, batch_id),
    )


@pytest.mark.parametrize("backend_type", (DictBackend, SQLiteBackend))
@pytest.mark.parametrize(
    "bad_request",
    (
        lambda route: _request(route, _source(2), batch_id=701, license_id="deny"),
        lambda route: _request(route, _source(2, source_kind=72), batch_id=701),
        lambda route: _request(route, _source(999), batch_id=701),
    ),
)
def test_batch_preflight_rejections_are_physical_zero_write(
        backend_type, bad_request):
    """许可、未知 kind 或异常来源位于组内第二项时，整批仍须物理零写。"""
    backend = backend_type()
    try:
        ctx = make_train_context(backend, companion=True)
        runtime, routes, _ = _install(ctx)
        good_source = _source(1)
        good = _request(
            routes[0],
            good_source,
            batch_id=701,
            parser=_Parser(good_source, 1),
        )
        bad = bad_request(routes[1])
        before = backend.recovery_state_snapshot()
        cursor = []
        with pytest.raises(SourceAdmissionRejected):
            runtime.ingest_batch(
                (good, bad), cursor_commit=lambda: cursor.append(1))
        assert backend.recovery_state_snapshot() == before
        assert cursor == []
    finally:
        backend.close()


class _DriftingPolicy(_SourcePolicy):
    """同一请求第二次返回不同来源簇，模拟非确定 assessment。"""

    def __init__(self, refs):
        """建立调用计数但保持公开 state_key 不变。"""
        super().__init__(refs)
        self.calls = 0

    def assess(self, request):
        """交替来源簇，触发 runtime 的同请求漂移保护。"""
        self.calls += 1
        assessment = super().assess(request)
        return replace(
            assessment,
            source_cluster_key=(29000, self.calls),
        )

    def clone_for_context(self, ctx):
        """克隆时只复用图身份，不复制调用计数。"""
        for ref in self.refs:
            ctx.graph_ontology.identity_of(ref)
        return _DriftingPolicy(self.refs)


def test_policy_drift_and_forged_graph_ref_fail_before_persistence():
    """assessment 漂移和伪造图引用都必须在 SourceRecord 前拒绝。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend, companion=True)
        runtime, routes, base_policy = _install(ctx)
        request = _request(routes[2], _source(31), batch_id=731)
        before = backend.recovery_state_snapshot()

        drifting = _DriftingPolicy(base_policy.refs)
        runtime.policy = drifting
        runtime.preflight(request)
        with pytest.raises(SourceAdmissionError, match="漂移"):
            runtime.preflight(request)
        assert backend.recovery_state_snapshot() == before

        forged = TypedRef(
            base_policy.refs[0].object_kind,
            base_policy.refs[0].space_id,
            999_999,
            base_policy.refs[0].owner,
            base_policy.refs[0].versions,
        )
        runtime.policy = _SourcePolicy((
            forged,
            *base_policy.refs[1:],
        ))
        with pytest.raises((KeyError, RuntimeError)):
            runtime.preflight(_request(routes[2], _source(32), batch_id=732))
        assert backend.recovery_state_snapshot() == before
    finally:
        backend.close()


class _FailSecondUnit:
    """在 M-10 完成第二个单元但尚未组提交时中断。"""

    def hit(self, point: int, context: dict[str, int]) -> None:
        """只在 after-unit 的 ordinal=1 抛出一次固定异常。"""
        if (point == FAULT_MEMORY_GROUP_AFTER_UNIT
                and context.get("unit_ordinal") == 1):
            raise RuntimeError("A-05 second unit fault")


@pytest.mark.parametrize("backend_type", (DictBackend, SQLiteBackend))
def test_second_unit_fault_hides_memory_keeps_audit_and_restart_rebuilds(
        backend_type):
    """组故障保留已准入审计，但 Memory/cursor 不可见且逻辑重启可重建。"""
    backend = backend_type()
    try:
        ctx = make_train_context(backend, companion=True)
        runtime, routes, policy = _install(ctx)
        first_source = _source(41)
        second_source = _source(
            42,
            owner=OwnerScope(1, 2, 3, VISIBILITY_SESSION),
        )
        requests = (
            _request(
                routes[0], first_source, batch_id=741,
                parser=_Parser(first_source, 1)),
            _request(
                routes[1], second_source, batch_id=741,
                parser=_Parser(second_source, 2)),
        )
        core_before = CoreCanonicalStateReader(ctx).read()
        cursor = []
        with pytest.raises(RuntimeError, match="second unit"):
            runtime.ingest_batch(
                requests,
                cursor_commit=lambda: cursor.append(741),
                fault_injector=_FailSecondUnit(),
            )
        reading_access = MemoryAccessContext(0, 0, 0)
        interaction_access = MemoryAccessContext(1, 2, 3)
        assert cursor == []
        assert CoreCanonicalStateReader(ctx).read() == core_before
        assert backend.count(SOURCE_RECORD_TABLE, where=None) == 2
        assert backend.count(SOURCE_TRUST_ASSESSMENT_TABLE, where=None) == 2
        assert backend.count(TEXT_ASSOC_TABLE, where=None) == 2
        assert ctx.memory_read_events.query(access=reading_access) == ()
        assert ctx.memory_interact_events.query(access=interaction_access) == ()

        restored = make_train_context(backend, companion=True)
        install_memory_batch_runtimes(restored, _batch_config())
        restored_runtime = install_source_admission_runtime(
            restored,
            policy.clone_for_context(restored),
            reading_route=routes[0],
            interaction_route=routes[1],
            record_only_routes=(routes[2],),
        )
        restored.memory_batch_coordinator.recover_groups()
        assert restored.memory_read_events.query(access=reading_access) == ()
        assert restored.memory_interact_events.query(
            access=interaction_access) == ()
        assert restored.memory_read_aggregates.rebuild_all(
            access=reading_access).aggregate_count == 0
        assert restored.memory_interact_aggregates.rebuild_all(
            access=interaction_access).aggregate_count == 0
        assert restored_runtime.records.find(
            first_source.stable_key()) is not None
        assert restored_runtime.records.find(
            second_source.stable_key()) is not None
    finally:
        backend.close()


def _hypothesis(source: SourceRef, candidate: int) -> HypothesisKey:
    """构造可被多个完整来源共同支持或反对的共享 Hypothesis。"""
    return HypothesisKey(
        (7201,),
        (candidate,),
        (31000,),
        document_scope(source),
        source,
    )


def _hypothesis_ref(ctx, hypothesis: HypothesisKey):
    """按完整 Hypothesis 声明恢复对应 MemoryObjectRef。"""
    for item in ctx.memory_interact_events.query(
            access=MemoryAccessContext(0, 0, 0)):
        if getattr(item.event.payload, "hypothesis", None) == hypothesis:
            return item.event.object_ref
    raise AssertionError("A-05 测试缺少 Hypothesis 声明")


@pytest.mark.parametrize("backend_type", (DictBackend, SQLiteBackend))
def test_cluster_counts_conflict_m07_and_k04_round_trip(backend_type):
    """同簇多文档只计一源，异簇同文可独立，冲突和冷热 trace 均保留。"""
    backend = backend_type()
    try:
        ctx = make_train_context(backend, companion=True)
        source_a = _source(51, document_id=501)
        source_b = _source(52, document_id=502)
        source_c = _source(53, document_id=503)
        clusters = {
            source_a.stable_key(): (32001,),
            source_b.stable_key(): (32001,),
            source_c.stable_key(): (32002,),
        }
        runtime, routes, _ = _install(ctx, clusters=clusters)
        for source, batch_id, raw_text in (
                (source_a, 751, "同文"),
                (source_b, 752, "另一文"),
                (source_c, 753, "同文")):
            runtime.admit_record_only(_request(
                routes[2], source, batch_id=batch_id, raw_text=raw_text))
        backend.commit()

        corroborated = _hypothesis(source_a, 1)
        conflicted = _hypothesis(source_a, 2)
        ledger = HypothesisLedger(
            MemoryHypothesisEventSink(ctx.memory_interact_events))
        ledger.register(corroborated)
        ledger.register(conflicted)
        for evidence_id, source, seq in (
                (1, source_a, 1),
                (2, source_b, 2),
                (3, source_c, 3)):
            ledger.append_evidence(EvidenceRecord(
                evidence_id,
                corroborated,
                EVIDENCE_SUPPORT,
                (33000 + evidence_id,),
                source,
                seq,
            ))
        ledger.append_evidence(EvidenceRecord(
            4, conflicted, EVIDENCE_SUPPORT, (33004,), source_a, 4))
        ledger.append_evidence(EvidenceRecord(
            5, conflicted, EVIDENCE_REFUTE, (33005,), source_b, 5))

        access = MemoryAccessContext(0, 0, 0)
        ctx.memory_interact_aggregates.policy = _TwoSourcePolicy()
        ctx.memory_interact_aggregates.rebuild_dirty(access=access)
        corroborated_aggregate = ctx.memory_interact_aggregates.read(
            _hypothesis_ref(ctx, corroborated), access=access)
        conflicted_aggregate = ctx.memory_interact_aggregates.read(
            _hypothesis_ref(ctx, conflicted), access=access)
        assert corroborated_aggregate is not None
        assert corroborated_aggregate.support_count == 3
        assert corroborated_aggregate.independent_source_count == 2
        assert corroborated_aggregate.support_source_count == 2
        assert corroborated_aggregate.evidence_state == MEMORY_EVIDENCE_CORROBORATED
        assert conflicted_aggregate is not None
        assert conflicted_aggregate.independent_source_count == 1
        assert conflicted_aggregate.support_source_count == 1
        assert conflicted_aggregate.contradict_source_count == 1
        assert conflicted_aggregate.evidence_state == MEMORY_EVIDENCE_CONFLICTED

        resolver = MemoryOverlayResolver(
            ctx.memory_interact_aggregates,
            ctx.core_identity_catalog,
            _BaselineProvider(_core_refs(ctx)[1]),
            _IndexFilterProvider(),
            _CurrentContextScorer(),
            _DistinctSourcePolicy(),
        )
        bundle = resolver.load_bundle(corroborated_aggregate, access=access)
        assert len(bundle.sources) == 3
        assert {
            item.source_cluster_key for item in bundle.source_traces
        } == {(32001,), (32002,)}
        assert all(item.source_assessment_key for item in bundle.source_traces)

        projection_key = (34000, 1)
        decoded = decode_memory_candidate(
            projection_key,
            encode_memory_candidate(projection_key, bundle),
        )
        assert decoded == bundle

        query_source = _query_source(document_id=1)
        from pure_integer_ai.experiments.memory_query_runtime import (
            install_memory_query_runtime,
        )
        query_runtime = install_memory_query_runtime(
            ctx,
            _protocol(query_source),
            aggregates=ctx.memory_interact_aggregates,
        )
        scope = _open_query(ctx, query_source)
        compilation = query_runtime.compile(
            _current(ctx, query_source, scope),
            access=MemoryAccessContext(1, 2, 3),
        )
        request = compilation.requests[0]
        candidate = resolver.candidate_from_bundle(request, decoded)
        assert candidate.diversity.distinct_source_count == 2

        traces = list(decoded.source_traces)
        traces[0] = replace(traces[0], source_cluster_key=(32002,))
        tampered = MemoryCandidateBundle(
            decoded.hypothesis_ref,
            decoded.hypothesis,
            decoded.aggregate,
            decoded.sources,
            tuple(sorted(traces, key=lambda item: item.stable_key())),
        )
        with pytest.raises(RuntimeError, match="持久化映射漂移"):
            resolver.candidate_from_bundle(request, tampered)
        _close_query(ctx)
    finally:
        backend.close()


@pytest.mark.parametrize("backend_type", (DictBackend, SQLiteBackend))
def test_v06_clone_preserves_keys_and_never_writes_host(backend_type):
    """V-06 克隆恢复同一 mapping，后续准入只写克隆后端。"""
    backend = backend_type()
    cloned_backend = None
    cloned = None
    try:
        ctx = make_train_context(backend, companion=True)
        source = _source(61)
        runtime, routes, _ = _install(
            ctx, clusters={source.stable_key(): (35001,)})
        runtime.admit_record_only(_request(
            routes[2], source, batch_id=761))
        backend.commit()
        host_before = backend.recovery_state_snapshot()

        cloned_backend = clone_backend(backend)
        cloned = clone_train_context(
            ctx,
            cloned_backend,
            label="a05-source-trust-clone",
        )
        assert cloned.source_trust_runtime.state_key() == runtime.state_key()
        assert cloned.source_trust_records.find(
            source.stable_key()) == ctx.source_trust_records.find(
                source.stable_key())
        clone_source = _source(62)
        cloned.source_trust_runtime.admit_record_only(_request(
            routes[2], clone_source, batch_id=762))
        cloned.backend.commit()
        assert cloned.source_trust_records.find(
            clone_source.stable_key()) is not None
        assert backend.recovery_state_snapshot() == host_before
        assert ctx.source_trust_records.find(clone_source.stable_key()) is None
    finally:
        if cloned_backend is not None:
            cloned_backend.close()
        backend.close()


def _a05_v02_probe(count: int) -> tuple[int, int, int, str]:
    """测量来源准入、来源簇聚合和候选页编码的确定性增长。"""
    backend = DictBackend()
    try:
        ctx = make_train_context(backend, companion=True)
        sources = tuple(_source(700 + index) for index in range(count))
        clusters = {
            source.stable_key(): (36000, index)
            for index, source in enumerate(sources)
        }
        runtime, routes, _ = _install(ctx, clusters=clusters)
        before = backend.snapshot()
        with collect_backend_telemetry() as telemetry:
            for index, source in enumerate(sources):
                runtime.admit_record_only(_request(
                    routes[2],
                    source,
                    batch_id=800 + index,
                    raw_text=f"来源正文{index}",
                ))
            ledger = HypothesisLedger(
                MemoryHypothesisEventSink(ctx.memory_interact_events))
            hypothesis = _hypothesis(sources[0], 1)
            ledger.register(hypothesis)
            for index, source in enumerate(sources):
                ledger.append_evidence(EvidenceRecord(
                    40000 + index,
                    hypothesis,
                    EVIDENCE_SUPPORT,
                    (41000 + index,),
                    source,
                    index + 1,
                ))
            access = MemoryAccessContext(0, 0, 0)
            ctx.memory_interact_aggregates.policy = _TwoSourcePolicy()
            ctx.memory_interact_aggregates.rebuild_dirty(access=access)
            aggregate = ctx.memory_interact_aggregates.read(
                _hypothesis_ref(ctx, hypothesis), access=access)
            assert aggregate is not None
            resolver = MemoryOverlayResolver(
                ctx.memory_interact_aggregates,
                ctx.core_identity_catalog,
                _BaselineProvider(_core_refs(ctx)[1]),
                _IndexFilterProvider(),
                _CurrentContextScorer(),
                _DistinctSourcePolicy(),
            )
            bundle = resolver.load_bundle(aggregate, access=access)
            projection_key = (37000, count)
            encoded = encode_memory_candidate(projection_key, bundle)
            assert decode_memory_candidate(projection_key, encoded) == bundle
        operations = telemetry.operation_snapshot()
        calls = sum(item[0] for item in operations.values())
        assert sum(item[2] for item in operations.values()) == 0
        after = backend.snapshot()
        growth = sum(
            len(after[table]) - len(before.get(table, ()))
            for table in after
        )
        digest = hashlib.sha256(canonical_json_bytes(after)).hexdigest()
        return calls, growth, len(encoded.payload), digest
    finally:
        backend.close()


def test_v02_source_trust_and_candidate_page_growth_is_linear():
    """V-02 synthetic：来源表和 K-04 v2 页随来源数保持线性。"""
    points = tuple((count, *_a05_v02_probe(count)) for count in (2, 4, 8))
    for metric_index in (1, 2, 3):
        first_delta = points[1][metric_index] - points[0][metric_index]
        second_delta = points[2][metric_index] - points[1][metric_index]
        assert first_delta > 0
        assert second_delta <= first_delta * 2 + 8
    assert _a05_v02_probe(4) == points[1][1:]
