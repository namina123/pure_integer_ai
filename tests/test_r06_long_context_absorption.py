"""R-06 长输入、持久 agenda 和严格 generation continuation 反例。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    ParserVersion,
    SourceRef,
    TypedRef,
    VersionBundle,
    concept_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.cognition.shared.semantic_object import (
    context_scope_identity,
    proposition_identity,
)
from pure_integer_ai.cognition.understanding.span_index import (
    SpanIndex,
    SpanProtocol,
)
from pure_integer_ai.experiments.authorized_generation_delivery import (
    AuthorizedGenerationDeliveryAuthority,
)
from pure_integer_ai.experiments.formal_train import make_train_context
from pure_integer_ai.experiments.long_generation_checkpoint import (
    LONG_GENERATION_BUDGET_STOP,
    LONG_GENERATION_COMPLETE,
    LongGenerationCheckpointError,
    LongGenerationCheckpointStore,
    LongGenerationPageBudget,
    LongGenerationPageCommit,
    LongGenerationPlan,
    LongGenerationPlanItem,
)
from pure_integer_ai.experiments.long_input_hierarchy import (
    LongInputChunk,
    LongInputHierarchyBuilder,
    LongInputHierarchyError,
    LongInputHierarchyProtocol,
    LongInputHierarchySeed,
)
from pure_integer_ai.experiments.persistent_conversation_agenda import (
    AGENDA_OPEN,
    AGENDA_RESOLVED,
    PersistentAgendaCenter,
    PersistentConversationAgendaError,
    PersistentConversationAgendaRuntime,
    PersistentConversationAgendaStore,
)
from pure_integer_ai.experiments.ph2_dataset_contract import StableRecordKey
from pure_integer_ai.experiments.ph2_long_context_absorption_catalog import (
    MANIFEST_PATH,
    build_long_context_absorption_manifest,
)
from pure_integer_ai.experiments.ph2_long_context_absorption_contract import (
    LongContextAbsorptionContractError,
    LongContextAbsorptionEvidenceFile,
    read_long_context_absorption_manifest,
    verify_long_context_absorption_files,
    write_long_context_absorption_manifest,
)
from pure_integer_ai.storage.backend import DictBackend, SQLiteBackend
from pure_integer_ai.storage.edge_store import SOURCE_BARE_TEXT
from pure_integer_ai.storage.segment_repository import BackendObjectRepository
from tests.test_r04_authorized_center_generation_absorption import (
    _agenda,
    _authorized_generation_fixture,
    _binding,
    _runtime_fixture,
)


_BASE = 98600
_ROOT = Path(__file__).resolve().parents[1]


def _source(document_id=1, *, parser=1):
    """构造同 owner 下版本化长输入来源。"""
    return SourceRef(
        SOURCE_BARE_TEXT,
        _BASE + 1,
        document_id,
        GLOBAL_OWNER_SCOPE,
        VersionBundle(parser=ParserVersion(parser)),
    )


def _long_input_fixture(source=None):
    """返回小型中文原文、绝对层级 seed 和两种无对齐 chunk 划分。"""
    source = _source() if source is None else source
    surfaces = ("甲事实。", "乙事实。", "甲事实。", "丙事实。")
    starts = []
    text = ""
    for surface in surfaces:
        starts.append(len(text))
        text += surface
    ends = tuple(start + len(surface) for start, surface in zip(starts, surfaces))
    scope = document_scope(source)
    document = ((0, len(text)),)
    section_one = ((0, ends[2]),)
    section_two = ((starts[3], ends[3]),)
    paragraph_one = ((0, ends[1]),)
    paragraph_two = ((starts[2], ends[2]),)
    paragraph_three = section_two
    seeds = tuple(
        LongInputHierarchySeed(
            proposition_identity(source, (_BASE + 10, ordinal)),
            context_scope_identity(source, (_BASE + 11, ordinal)),
            ordinal,
            document,
            section_one if ordinal < 3 else section_two,
            (paragraph_one, paragraph_one, paragraph_two, paragraph_three)[ordinal],
            ((starts[ordinal], ends[ordinal]),),
            1,
            10 if ordinal < 3 else 20,
            (101, 101, 102, 201)[ordinal],
            1001 + ordinal,
        )
        for ordinal in range(4)
    )

    def chunks(width):
        values = tuple(
            LongInputChunk.from_text(source, start, text[start:start + width])
            for start in range(0, len(text), width)
        )
        return tuple(reversed(values))

    return text, scope, seeds, chunks(3), chunks(5)


def test_long_input_hierarchy_is_chunk_order_invariant_and_materializes_spans():
    """不同 chunk 宽度/顺序形成相同 identity，并写入真实 SpanIndex 层级。"""
    text, scope, seeds, first_chunks, second_chunks = _long_input_fixture()
    builder = LongInputHierarchyBuilder()
    first = builder.build(first_chunks, scope, seeds)
    second = builder.build(second_chunks, scope, seeds)
    stable_before_access = first.stable_key()

    assert first.stable_key() == second.stable_key()
    assert first.records[0].content_digest == first.records[2].content_digest
    assert first.records[0].paragraph_span != first.records[2].paragraph_span
    assert first.records[0].proposition_span != first.records[2].proposition_span

    backend = DictBackend()
    try:
        context = make_train_context(backend)
        spans = SpanIndex(
            context.graph_ontology,
            context.scoped_identity_store,
            SpanProtocol(
                (_BASE + 20, 1),
                (_BASE + 20, 2),
                (_BASE + 20, 3),
                (_BASE + 20, 4),
            ),
        )
        protocol = LongInputHierarchyProtocol(
            structure_concept_identity((_BASE + 21, 1)),
            structure_concept_identity((_BASE + 21, 2)),
            structure_concept_identity((_BASE + 21, 3)),
            structure_concept_identity((_BASE + 21, 4)),
        )
        refs = builder.materialize(first, first_chunks, spans, protocol)
        assert len(refs) == 4
        assert spans.span_count() == 10
        for record, (_, proposition_ref) in zip(first.records, refs):
            assert isinstance(proposition_ref, TypedRef)
            assert spans.members_of(proposition_ref) == record.proposition_members
        assert first.stable_key() == stable_before_access
    finally:
        backend.close()


def test_long_input_local_offsets_source_and_prefix_drift_fail_closed():
    """chunk-local offset、SourceRef/parser 漂移和旧 document digest 均不得静默复用。"""
    text, scope, seeds, chunks, _ = _long_input_fixture()
    builder = LongInputHierarchyBuilder()
    baseline = builder.build(chunks, scope, seeds)
    source = scope.source
    local = tuple(
        LongInputChunk.from_text(source, 0, item.text) for item in chunks)
    with pytest.raises(LongInputHierarchyError, match="空洞或重叠"):
        builder.build(local, scope, seeds)

    modified_text = "丁" + text[1:]
    modified = (
        LongInputChunk.from_text(source, 0, modified_text),)
    with pytest.raises(LongInputHierarchyError, match="digest 漂移"):
        builder.build(
            modified,
            scope,
            seeds,
            expected_document_digest=baseline.document_digest,
        )

    drift_source = _source(parser=2)
    _, drift_scope, drift_seeds, drift_chunks, _ = _long_input_fixture(
        drift_source)
    drift = builder.build(drift_chunks, drift_scope, drift_seeds)
    assert drift.stable_key() != baseline.stable_key()


def _metadata_store(path):
    """打开可跨进程重建的通用 sealed metadata repository。"""
    backend = SQLiteBackend(str(path))
    repository = BackendObjectRepository(backend)
    return backend, repository


def test_persistent_agenda_reopens_metadata_then_exact_pages_in_once(tmp_path):
    """启动只恢复 center/record metadata，随后授权 runtime 才读取唯一 segment。"""
    fixture = _runtime_fixture()
    database = tmp_path / "agenda.sqlite3"
    try:
        first = fixture.recall.centers[0]
        second = replace(
            first,
            center_key=StableRecordKey((_BASE + 30, 2)),
        )
        references = (
            PersistentAgendaCenter.from_formed_center(
                first,
                StableRecordKey((_BASE + 31, 1)),
                logical_seq=1,
            ),
            PersistentAgendaCenter.from_formed_center(
                second,
                StableRecordKey((_BASE + 31, 2)),
                dependencies=(first.center_key,),
                logical_seq=1,
            ),
        )
        backend, repository = _metadata_store(database)
        try:
            created = PersistentConversationAgendaStore(
                repository, backend.commit).create(
                    StableRecordKey((_BASE + 32, 1)), references)
            assert created.revision == 0
        finally:
            backend.close()

        before_reads = fixture.repository.segment_reads
        backend, repository = _metadata_store(database)
        try:
            store = PersistentConversationAgendaStore(
                repository, backend.commit)
            restored = store.load(StableRecordKey((_BASE + 32, 1)))
            binding = PersistentConversationAgendaRuntime.bind(
                restored, (first, second))
            assert fixture.repository.segment_reads == before_reads
            run = _agenda(
                fixture,
                binding.centers,
                (_binding(fixture, first), _binding(fixture, second)),
            )
            assert fixture.repository.segment_reads == before_reads + 1
            assert sum(
                item.receipt.physical_payload_gets for item in run.states) == 1
            assert sum(item.receipt.reused_payload for item in run.states) == 1
            advanced = store.record_authorized_run(
                restored,
                run,
                lifecycle_by_center=(
                    (first.center_key, AGENDA_RESOLVED, 2),
                    (second.center_key, AGENDA_OPEN, 2),
                ),
            )
            assert advanced.revision == 1
            assert tuple(len(item.consumer_receipt_keys)
                         for item in advanced.centers) == (1, 1)
        finally:
            backend.close()

        backend, repository = _metadata_store(database)
        try:
            restored = PersistentConversationAgendaStore(
                repository, backend.commit).load(
                    StableRecordKey((_BASE + 32, 1)))
            assert restored.revision == 1
            assert restored.centers[0].lifecycle == AGENDA_RESOLVED
            assert restored.centers[1].lifecycle == AGENDA_OPEN
            assert fixture.repository.segment_reads == before_reads + 1
        finally:
            backend.close()
    finally:
        fixture.close()


def test_missing_or_drifted_persistent_agenda_cannot_bind_payload(tmp_path):
    """无 agenda 或 record identity 漂移时，metadata 本身不能授予任何 page-in。"""
    fixture = _runtime_fixture()
    database = tmp_path / "missing-agenda.sqlite3"
    backend, repository = _metadata_store(database)
    try:
        store = PersistentConversationAgendaStore(repository, backend.commit)
        with pytest.raises(KeyError, match="agenda 不存在"):
            store.load(StableRecordKey((_BASE + 33, 1)))
        reference = PersistentAgendaCenter.from_formed_center(
            fixture.recall.centers[0],
            StableRecordKey((_BASE + 33, 2)),
        )
        agenda = store.create(
            StableRecordKey((_BASE + 33, 1)), (reference,))
        drifted = replace(
            fixture.recall.centers[0].index_entry,
            record_key=(_BASE + 33, 9),
        )
        forged = replace(fixture.recall.centers[0], index_entry=drifted)
        before = fixture.repository.segment_reads
        with pytest.raises(PersistentConversationAgendaError, match="record identity"):
            PersistentConversationAgendaRuntime.bind(agenda, (forged,))
        assert fixture.repository.segment_reads == before
    finally:
        backend.close()
        fixture.close()


def _checkpoint_store(path):
    """打开跨进程 checkpoint store。"""
    backend, repository = _metadata_store(path)
    return backend, LongGenerationCheckpointStore(repository, backend.commit)


def _page(
        checkpoint,
        decision,
        state,
        item_key,
        page_key,
        *,
        used=(),
        introduced=(),
        ):
    """构造绑定当前 checkpoint 的真实授权 generation 页。"""
    return LongGenerationPageCommit(
        page_key,
        (item_key,),
        StableRecordKey((_BASE + 40, 1)),
        StableRecordKey((_BASE + 40, 2)),
        checkpoint.revision,
        checkpoint.next_cursor,
        checkpoint.prefix_digest,
        used,
        introduced,
        decision,
        (state,),
    )


def test_generation_checkpoint_resumes_across_process_and_commits_after_g04(
        tmp_path):
    """每个局部页先严格交付后 append checkpoint，跨进程从精确 cursor 继续。"""
    fixture, qa, agenda, run, claim = _authorized_generation_fixture()
    database = tmp_path / "checkpoint.sqlite3"
    try:
        decision = AuthorizedGenerationDeliveryAuthority().authorize(
            run, (claim,))
        plan = LongGenerationPlan(
            StableRecordKey((_BASE + 41, 1)),
            (
                LongGenerationPlanItem(
                    StableRecordKey((_BASE + 41, 2)),
                    claim.proposition_key,
                ),
                LongGenerationPlanItem(
                    StableRecordKey((_BASE + 41, 3)),
                    claim.proposition_key,
                ),
            ),
        )
        antecedent = concept_identity((_BASE + 41, 4))
        backend, store = _checkpoint_store(database)
        try:
            initial = store.create(plan)
            first = store.commit_page(
                plan,
                _page(
                    initial,
                    decision,
                    agenda.states[0],
                    plan.items[0].item_key,
                    StableRecordKey((_BASE + 41, 5)),
                    introduced=(antecedent,),
                ),
                LongGenerationPageBudget(
                    len(decision.envelope.units), 1),
            )
            assert first.checkpoint.status == LONG_GENERATION_BUDGET_STOP
            assert first.checkpoint.next_cursor == 1
        finally:
            backend.close()

        backend, store = _checkpoint_store(database)
        try:
            restored = store.load(plan.answer_key)
            assert restored == first.checkpoint
            second = store.commit_page(
                plan,
                _page(
                    restored,
                    decision,
                    agenda.states[0],
                    plan.items[1].item_key,
                    StableRecordKey((_BASE + 41, 6)),
                    used=(antecedent,),
                ),
                LongGenerationPageBudget(
                    len(decision.envelope.units), 1),
            )
            assert second.complete
            assert second.checkpoint.status == LONG_GENERATION_COMPLETE
            assert second.checkpoint.next_cursor == 2
            assert second.checkpoint.prefix_digest != initial.prefix_digest
            assert second.checkpoint.committed_page_keys == (
                StableRecordKey((_BASE + 41, 5)),
                StableRecordKey((_BASE + 41, 6)),
            )
            assert len(second.checkpoint.citation_record_keys) == 2
            assert len(second.checkpoint.authorization_receipt_keys) == 2
        finally:
            backend.close()
    finally:
        qa.close()
        fixture.close()


def test_postcheck_pagein_cursor_prefix_and_antecedent_failures_are_atomic(
        tmp_path):
    """去掉 G-04/page-in/cursor/prefix/antecedent 任一门都不推进 checkpoint。"""
    fixture, qa, agenda, run, claim = _authorized_generation_fixture()
    database = tmp_path / "checkpoint-failures.sqlite3"
    backend = None
    try:
        authority = AuthorizedGenerationDeliveryAuthority()
        authorized = authority.authorize(run, (claim,))
        rejected = authority.authorize(replace(run, postcheck=None), (claim,))
        plan = LongGenerationPlan(
            StableRecordKey((_BASE + 42, 1)),
            (LongGenerationPlanItem(
                StableRecordKey((_BASE + 42, 2)),
                claim.proposition_key,
            ),),
        )
        backend, store = _checkpoint_store(database)
        initial = store.create(plan)
        budget = LongGenerationPageBudget(
            len(authorized.envelope.units), 1)
        valid = _page(
            initial,
            authorized,
            agenda.states[0],
            plan.items[0].item_key,
            StableRecordKey((_BASE + 42, 3)),
        )

        drifted_prefix = (
            (valid.expected_prefix_digest[0] + 1) % 256,
            *valid.expected_prefix_digest[1:],
        )
        failures = (
            (replace(valid, delivery=rejected), budget),
            (replace(valid, authorized_states=(agenda.stop_center(
                agenda.states[0].center.center_key).states[0],)), budget),
            (replace(valid, expected_cursor=1), budget),
            (replace(valid, expected_prefix_digest=drifted_prefix), budget),
            (replace(valid, antecedents_used=(
                concept_identity((_BASE + 42, 4)),)), budget),
            (valid, LongGenerationPageBudget(
                len(authorized.envelope.units) - 1, 1)),
        )
        for page, page_budget in failures:
            before = store.load(plan.answer_key)
            with pytest.raises(LongGenerationCheckpointError):
                store.commit_page(plan, page, page_budget)
            assert store.load(plan.answer_key) == before == initial
    finally:
        if backend is not None:
            backend.close()
        qa.close()
        fixture.close()


@pytest.fixture(scope="module")
def formal_manifest():
    """按当前仓库内容构建一次确定性 R-06 正式 manifest。"""
    return build_long_context_absorption_manifest(_ROOT)


def test_manifest_round_trip_file_identity_and_zero_execution(
        tmp_path, formal_manifest):
    """R-06 artifact 必须 canonical 回读并逐字节绑定实现和测试。"""
    target = tmp_path / "r06.json"
    assert write_long_context_absorption_manifest(
        formal_manifest, target) == target
    assert read_long_context_absorption_manifest(target) == formal_manifest
    verify_long_context_absorption_files(formal_manifest, repository_root=_ROOT)
    assert set(formal_manifest.requirement_decisions.to_value().values()) == {
        "PASS"}
    assert formal_manifest.execution_state.to_value()["formal_training_runs"] == 0
    assert formal_manifest.execution_state.to_value()["teacher_calls"] == 0


def test_manifest_rejects_fake_counterexample_coverage(formal_manifest):
    """任一长输入、agenda 或 checkpoint 反例被删都必须构造失败。"""
    coverage = formal_manifest.counterexample_coverage.to_value()
    coverage["checkpoint_prefix_drift_atomic_failure"] = 0
    from pure_integer_ai.experiments.ph2_dataset_contract import (
        CanonicalJsonObject,
    )
    with pytest.raises(
            LongContextAbsorptionContractError,
            match="counterexample coverage 漂移"):
        replace(
            formal_manifest,
            counterexample_coverage=CanonicalJsonObject.from_value(coverage),
        )


def test_manifest_rejects_file_identity_drift(formal_manifest):
    """任何三件套实现、生产依赖或测试尺寸/hash 漂移都必须失败。"""
    first = formal_manifest.evidence_files[0]
    drifted = LongContextAbsorptionEvidenceFile(
        first.relative_path,
        first.role,
        first.byte_count + 1,
        first.sha256,
    )
    manifest = replace(
        formal_manifest,
        evidence_files=(drifted, *formal_manifest.evidence_files[1:]),
    )
    with pytest.raises(
            LongContextAbsorptionContractError,
            match="evidence 文件身份漂移"):
        verify_long_context_absorption_files(manifest, repository_root=_ROOT)


def test_manifest_is_idempotent_but_non_overwritable(
        tmp_path, formal_manifest):
    """同内容可幂等重放，同版本异内容不可覆盖。"""
    target = tmp_path / "r06.json"
    write_long_context_absorption_manifest(formal_manifest, target)
    assert write_long_context_absorption_manifest(
        formal_manifest, target) == target
    target.write_bytes(b"{}\n")
    with pytest.raises(
            LongContextAbsorptionContractError,
            match="已存在且内容不同"):
        write_long_context_absorption_manifest(formal_manifest, target)


def test_stored_manifest_is_current_readable_and_deterministic(formal_manifest):
    """仓内正式 artifact 必须等于当前 builder，重复构建字节一致。"""
    stored = read_long_context_absorption_manifest(_ROOT / MANIFEST_PATH)
    rebuilt = build_long_context_absorption_manifest(_ROOT)
    assert stored == formal_manifest == rebuilt
    assert stored.canonical_bytes() == rebuilt.canonical_bytes()
    verify_long_context_absorption_files(stored, repository_root=_ROOT)
