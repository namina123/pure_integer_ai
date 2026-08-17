"""R-01 版本化 alias/refers/realizes 课程、恢复和隔离对抗测试。"""
from __future__ import annotations

from dataclasses import replace

import pytest

from pure_integer_ai.cognition.shared.alias_resolution import (
    AliasRouteSearchBudget,
)
from pure_integer_ai.cognition.shared.identity import (
    concept_identity,
    minimal_instruction_identity,
)
from pure_integer_ai.cognition.shared.hypothesis import HypothesisLedger
from pure_integer_ai.cognition.shared.evidence_candidate import (
    EvidenceCandidateError,
)
from pure_integer_ai.cognition.shared.relation_use import (
    RelationUseContext,
    RelationUseGraphProtocol,
    RelationUseWriteMetadata,
)
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.experiments.alias_relation_course import (
    AliasRelationCourseEntry,
    AliasRelationCourseError,
    AliasRelationCourseLoader,
    AliasRelationCourseManifest,
    AliasRelationCourseRecognition,
    AliasRelationStatementMetadata,
)
from pure_integer_ai.experiments.evaluation_isolation import (
    clone_backend,
    isolated_evaluation,
)
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.storage.backend import DictBackend, SQLiteBackend
from pure_integer_ai.storage.edge_store import EPI_STRUCTURED, SOURCE_BARE_TEXT
from pure_integer_ai.storage.memory_event import MEMORY_EVENT_TABLE

from tests.test_r00_relation_closure import (
    _r01_fixture,
    _r01_recognition,
    _semantic_predicate_identities,
    _source,
)


_BASE = 17400


def _use_protocol(variant: int) -> RelationUseGraphProtocol:
    """构造互异的 PH2 Core relation Use 图协议。"""
    identities = tuple(
        concept_identity((_BASE + variant, ordinal))
        for ordinal in range(1, 9)
    )
    return RelationUseGraphProtocol(
        *identities,
        (_BASE + variant, 9),
    )


def _manifest(*, variant: int = 1) -> AliasRelationCourseManifest:
    """从既有 R-01 fixture 提取纯协议和 typed 课程，不复用其后端状态。"""
    fixture = _r01_fixture()
    try:
        closure = fixture.closure
        entries = []
        for ordinal, name in enumerate(sorted(fixture.specs), start=1):
            spec = fixture.specs[name]
            recognition = _r01_recognition(spec, 300 + ordinal + variant * 20)
            base = ordinal * 20
            entries.append(AliasRelationCourseEntry(
                spec,
                document_scope(spec.proposition.source),
                base,
                (AliasRelationCourseRecognition(
                    recognition,
                    base + 3,
                    base + 4,
                    base + 5,
                ),),
            ))
        return AliasRelationCourseManifest(
            1,
            (_BASE, variant),
            _semantic_predicate_identities(),
            fixture.candidate_graph.protocol,
            (_BASE + 100, variant),
            closure.candidate_runtime.engine.protocol,
            closure.candidate_runtime.verifier.protocol,
            closure.candidate_runtime.metadata,
            AliasRelationStatementMetadata(
                SOURCE_BARE_TEXT,
                EPI_STRUCTURED,
                content_version=variant,
            ),
            closure.protocol,
            closure.consumer.schemas,
            fixture.protocol,
            _use_protocol(variant),
            RelationUseWriteMetadata(
                SOURCE_BARE_TEXT,
                EPI_STRUCTURED,
                content_version=variant,
            ),
            tuple(entries),
        )
    finally:
        fixture.close()


@pytest.mark.parametrize("backend_type", (DictBackend, SQLiteBackend))
def test_relation_course_is_idempotent_and_restores_core_history(backend_type):
    """同一课程跨运行零增长恢复，并在 Dict/SQLite 上保持相同 active 集。"""
    backend = backend_type()
    try:
        manifest = _manifest(variant=1)
        loader = AliasRelationCourseLoader(manifest, manifest.sha256())
        ctx = make_train_context(backend)

        first = loader.load(ctx)
        first_snapshot = backend.snapshot()
        second = loader.load(ctx)

        assert backend.snapshot() == first_snapshot
        assert second.report == first.report
        assert first.report.active_count == len(manifest.entries)
        assert first.report.recognition_count == len(manifest.entries)
        assert first.report.relation_use_count == 0
        assert backend.count(MEMORY_EVENT_TABLE) == 0

        restarted = make_train_context(backend)
        restored = loader.load(restarted)
        assert restored.report == first.report
        assert restored.alias is not first.alias
        assert restored.alias.closure.candidate_runtime.engine is not (
            first.alias.closure.candidate_runtime.engine)
        assert backend.snapshot() == first_snapshot
    finally:
        backend.close()


def test_relation_course_forms_all_relation_kinds_and_persists_real_use():
    """课程形成 alias/refers/realizes，真实 surface 采用才追加来源化 Use Event。"""
    backend = DictBackend()
    try:
        manifest = _manifest(variant=2)
        loaded = AliasRelationCourseLoader(
            manifest, manifest.sha256()).load(make_train_context(backend))
        relations = {
            entry.spec.proposition.predicate for entry in manifest.entries
        }
        assert relations == {
            manifest.alias_protocol.alias_relation,
            manifest.alias_protocol.refers_relation,
            manifest.alias_protocol.realizes_relation,
        }

        query = _source(990)
        context = RelationUseContext(
            query,
            document_scope(query),
            minimal_instruction_identity((_BASE + 200, 1)),
            minimal_instruction_identity((_BASE + 200, 2)),
        )
        result = loaded.alias.select_surface(
            next(
                binding.filler
                for entry in manifest.entries
                if entry.spec.proposition.predicate
                == manifest.alias_protocol.realizes_relation
                for binding in entry.spec.proposition.bindings
                if binding.role
                == manifest.alias_protocol.realizes_bearer_role
            ),
            next(
                binding.filler
                for entry in manifest.entries
                if entry.spec.proposition.predicate
                == manifest.alias_protocol.realizes_relation
                for binding in entry.spec.proposition.bindings
                if binding.role
                == manifest.alias_protocol.realizes_branch_role
            ),
            budget=AliasRouteSearchBudget(100, 100, 100),
            use_key=(_BASE + 201, 1),
            allowed_prefix_steps=(),
            context=context,
        )

        assert result.result.selected is not None
        assert len(result.relation_uses) == 1
        assert result.relation_uses[0].event is not None
        assert len(loaded.alias.closure.use_owner.history()) == 1
        assert backend.count(MEMORY_EVENT_TABLE) == 0

        restored = AliasRelationCourseLoader(
            manifest, manifest.sha256()).load(make_train_context(backend))
        assert restored.report.relation_use_count == 1
        assert len(restored.alias.closure.use_owner.history()) == 1
    finally:
        backend.close()


def test_relation_course_rejects_hash_partial_protocol_and_source_leakage():
    """内容漂移、部分协议和 forming/reveal 同源均在正式课程写入前失败。"""
    backend = DictBackend()
    try:
        manifest = _manifest(variant=3)
        ctx = make_train_context(backend)
        baseline = backend.snapshot()
        with pytest.raises(AliasRelationCourseError, match="哈希漂移"):
            AliasRelationCourseLoader(manifest, "0" * 64).load(ctx)
        assert backend.snapshot() == baseline

        ctx.graph_ontology.materialize(manifest.semantic_predicates[0])
        partial = backend.snapshot()
        with pytest.raises(AliasRelationCourseError, match="部分协议"):
            AliasRelationCourseLoader(
                manifest, manifest.sha256()).load(ctx)
        assert backend.snapshot() == partial

        entry = manifest.entries[0]
        leaked_input = replace(
            entry.recognitions[0].input,
            revealed=replace(
                entry.recognitions[0].input.revealed,
                verifier_source=entry.spec.forming_sources[0],
            ),
        )
        with pytest.raises(ValueError, match="来源必须分离"):
            replace(entry, recognitions=(replace(
                entry.recognitions[0], input=leaked_input),))
    finally:
        backend.close()


def test_relation_course_factory_rebuilds_v06_owner_without_sharing():
    """V-06 克隆从复制图和 Core 历史重建独立 runtime、Use owner 与派生字典。"""
    backend = DictBackend()
    cloned_backend = None
    try:
        manifest = _manifest(variant=4)
        host_ctx = make_train_context(backend)
        loaded = AliasRelationCourseLoader(
            manifest, manifest.sha256()).load(host_ctx)
        cloned_backend = clone_backend(backend)
        cloned_ctx = make_train_context(cloned_backend)

        cloned = loaded.factory.clone_for_evaluation().build(cloned_ctx)

        assert cloned is not loaded.alias
        assert cloned.closure.semantic_graph.ontology is cloned_ctx.graph_ontology
        assert cloned.closure.use_owner is not loaded.alias.closure.use_owner
        assert cloned.closure.use_owner._uses is not (
            loaded.alias.closure.use_owner._uses)
        assert cloned._uses is not loaded.alias._uses
        assert cloned.state_key() == loaded.alias.state_key()
        assert backend.count(MEMORY_EVENT_TABLE) == 0
        assert cloned_backend.count(MEMORY_EVENT_TABLE) == 0
    finally:
        if cloned_backend is not None:
            cloned_backend.close()
        backend.close()


def test_relation_course_batch_preflight_avoids_per_entry_owner_clone(
        monkeypatch):
    """forming 与 support recognition 整批预演，owner 复制次数不随 entry 数线性增长。"""
    manifest = _manifest(variant=5)
    clone_count = 0
    original = HypothesisLedger.clone

    def counted(self):
        """统计课程加载期间完整 H-00 ledger 的复制次数。"""
        nonlocal clone_count
        clone_count += 1
        return original(self)

    monkeypatch.setattr(HypothesisLedger, "clone", counted)
    backend = DictBackend()
    try:
        AliasRelationCourseLoader(
            manifest, manifest.sha256()).load(make_train_context(backend))
        assert len(manifest.entries) >= 7
        assert clone_count <= 12
    finally:
        backend.close()


def test_relation_course_isolated_evaluation_fast_path_is_guarded_and_equal(
        monkeypatch):
    """快速路径只接受外层隔离 owner/session，且不改变课程结果。"""
    manifest = _manifest(variant=7)
    loader = AliasRelationCourseLoader(manifest, manifest.sha256())
    normal_backend = DictBackend()
    host_backend = DictBackend()
    try:
        normal_ctx = make_train_context(normal_backend)
        calls = 0
        original = AliasRelationCourseLoader._preflight_isolated

        def counted(self):
            """统计普通入口仍执行完整独立预演。"""
            nonlocal calls
            calls += 1
            return original(self)

        monkeypatch.setattr(
            AliasRelationCourseLoader, "_preflight_isolated", counted)
        normal = loader.load(normal_ctx)
        assert calls == 1

        with pytest.raises(AliasRelationCourseError, match="scope owner"):
            loader.load_for_isolated_evaluation(
                make_train_context(host_backend))

        host_ctx = make_train_context(host_backend)
        with isolated_evaluation(
                host_ctx, label="r01-course-fast-path") as eval_ctx:
            active_session = eval_ctx.work_memory.active_session_scope
            assert active_session is not None
            assert active_session.owner == eval_ctx.scope_owner
            isolated = loader.load_for_isolated_evaluation(eval_ctx)
            assert calls == 1
            assert isolated.report == normal.report
            assert isolated.alias.state_key() == normal.alias.state_key()
    finally:
        host_backend.close()
        normal_backend.close()


def test_relation_course_rejects_forming_timestamp_drift_before_write():
    """即使调用方接受新内容锁，既有 forming 逻辑序也不能被静默重解释。"""
    backend = DictBackend()
    try:
        manifest = _manifest(variant=6)
        ctx = make_train_context(backend)
        AliasRelationCourseLoader(
            manifest, manifest.sha256()).load(ctx)
        baseline = backend.snapshot()
        first = manifest.entries[0]
        drifted = replace(
            manifest,
            course_version=(*manifest.course_version, 1),
            entries=(
                replace(first, timestamp_base=first.timestamp_base + 1),
                *manifest.entries[1:],
            ),
        )

        with pytest.raises(EvidenceCandidateError, match="forming Evidence"):
            AliasRelationCourseLoader(
                drifted, drifted.sha256()).load(ctx)
        assert backend.snapshot() == baseline
    finally:
        backend.close()
