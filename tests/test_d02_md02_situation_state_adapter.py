"""MD-02 篇章状态三件套、局部重建和不可覆盖 manifest T0。"""
from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.cognition.shared.attractor_state import (
    AttractorContextUpdate,
    AttractorDependency,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_ENTITY,
    OBJECT_EVENT,
    OBJECT_PROPOSITION,
    OwnerScope,
    ParserVersion,
    SourceRef,
    VersionBundle,
    VISIBILITY_SESSION,
    minimal_instruction_identity,
)
from pure_integer_ai.cognition.shared.memory_event import (
    MEMORY_EVENT_OBSERVATION,
    MEMORY_OBJECT_OBSERVATION,
    MemoryEvent,
    MemoryLinkedRef,
    ObservationPayload,
    memory_object_ref,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    CLOCK_MEMORY_OBSERVED,
    LogicalClock,
    LogicalClockIdentity,
    LogicalTimestamp,
    document_scope,
    episode_scope,
    query_scope,
)
from pure_integer_ai.cognition.shared.semantic_object import (
    entity_identity,
)
from pure_integer_ai.cognition.shared.situation_state import (
    CurrentSituationProjection,
    SituationEventLog,
    SituationProjectionEntry,
    SituationProjectionReplacement,
    SituationRebuildReceipt,
    SituationStateError,
)
from pure_integer_ai.cognition.shared.work_memory import WorkMemory
from pure_integer_ai.cognition.shared.work_memory_content import (
    WorkMemoryContentProtocol,
)
from pure_integer_ai.cognition.shared.work_memory_discourse import (
    WorkMemoryDiscourseProjection,
    WorkMemoryDiscourseRoles,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line
from pure_integer_ai.experiments.ph2_md02_manifest import (
    MD02AdapterManifest,
    MD02ManifestError,
    build_md02_adapter_manifest,
    read_md02_adapter_manifest,
    write_md02_adapter_manifest,
)
from pure_integer_ai.experiments.train_context import make_train_context
from pure_integer_ai.storage.backend import DictBackend
from tests.test_a02_work_memory_content import (
    _anchor,
    _definition,
    _index,
    _item,
    _record,
    _role,
    _source,
    _start,
)
from tests.test_m03_memory_event import _core_refs


REPO_ROOT = Path(__file__).resolve().parents[1]
MD01_PATH = REPO_ROOT / "data/ph2/manifests/md01_memory_dynamics_contract_v1.json"
BASELINE_PATH = REPO_ROOT / "data/ph2/manifests/language_capability_baseline_v18.json"
MANIFEST_PATH = REPO_ROOT / "data/ph2/manifests/md02_situation_state_adapter_v1.json"


def _instruction(source: SourceRef, ordinal: int):
    """建立 owner/version 对齐的一等最小指令。"""
    return minimal_instruction_identity(
        (28600, ordinal), owner=source.owner, versions=source.versions)


def _observation_event(ctx, source: SourceRef, *, seq: int) -> MemoryEvent:
    """建立由真实 MemoryEventLog 校验的 source-bound Observation。"""
    document = document_scope(source)
    episode = episode_scope(1, parent=document)
    context, concept, structure, proposition, relation = _core_refs(ctx)
    timestamp = LogicalClock(
        LogicalClockIdentity(episode, CLOCK_MEMORY_OBSERVED),
        seq - 1,
    ).advance()
    payload = ObservationPayload(
        source,
        MemoryLinkedRef.core(context),
        (concept,),
        (concept, structure),
        structure,
        (proposition,),
        (MemoryLinkedRef.core(relation),),
        timestamp,
    )
    ref = memory_object_ref(
        ctx.memory_interact_events.memory_space_identity,
        MEMORY_OBJECT_OBSERVATION,
        payload.stable_key(),
        owner=source.owner,
        versions=source.versions,
    )
    return MemoryEvent(MEMORY_EVENT_OBSERVATION, ref, episode, payload)


def _fixture():
    """建立真实 Memory/WorkMemory/G-02/A-10 复用链。"""
    backend = DictBackend()
    ctx = make_train_context(backend)
    source = _source(71)
    _graph_backend, _graph_ctx, index = _index()
    entity_role = _role(source, 31)
    event_role = _role(source, 32)
    proposition_role = _role(source, 33)
    work_memory = WorkMemory()
    work_memory.configure_content(WorkMemoryContentProtocol((
        _definition(entity_role, (OBJECT_ENTITY,), 5, 1),
        _definition(event_role, (OBJECT_EVENT,), 5, 1),
        _definition(proposition_role, (OBJECT_PROPOSITION,), 5, 1),
    ), 12))
    _session, _document, episode = _start(work_memory, source)
    scope = query_scope(9, parent=episode)
    work_memory.begin_query(scope)
    record = _record(index, source, "甲修正乙", 0)
    anchor = _anchor(index, record)
    first = work_memory.put_content(_item(
        work_memory,
        entity_role,
        entity_identity(source, (28610, 1)),
        anchor,
        1,
    ))
    from pure_integer_ai.cognition.shared.semantic_object import (
        event_identity,
        proposition_identity,
    )
    second = work_memory.put_content(_item(
        work_memory,
        event_role,
        event_identity(source, (28610, 2)),
        anchor,
        2,
    ))
    third = work_memory.put_content(_item(
        work_memory,
        proposition_role,
        proposition_identity(source, (28610, 3)),
        anchor,
        3,
    ))
    discourse = WorkMemoryDiscourseProjection(
        (28620, 1),
        (28620, 2),
        WorkMemoryDiscourseRoles(
            entity_role, proposition_role, event_role),
        (first, second, third),
    )
    dependencies = tuple(
        AttractorDependency(_instruction(source, ordinal), item.value)
        for ordinal, item in enumerate((first, second, third), start=1)
    )
    facade = SituationEventLog(
        ctx.memory_interact_events, source, scope)
    projection = CurrentSituationProjection.from_work_memory_discourse(
        facade,
        work_memory.require_content_store(),
        scope,
        discourse,
        kind_by_role={
            entity_role: "ENTITY",
            event_role: "EVENT",
            proposition_role: "PROPOSITION",
        },
        dependencies_by_content_ref={
            item.content_ref(): (dependency,)
            for item, dependency in zip((first, second, third), dependencies)
        },
    )
    return {
        "backend": backend,
        "graph_backend": _graph_backend,
        "ctx": ctx,
        "source": source,
        "scope": scope,
        "work_memory": work_memory,
        "anchor": anchor,
        "items": (first, second, third),
        "dependencies": dependencies,
        "facade": facade,
        "projection": projection,
    }


def _replacement(fixture, *, entry_index: int = 0):
    """为指定投影槽位建立单步 supersede 的新 WorkMemory item。"""
    projection = fixture["projection"]
    source = fixture["source"]
    old_entry = projection.entries()[entry_index]
    old_item = next(
        item for item in fixture["items"]
        if item.content_ref() == old_entry.content_ref)
    new_item = _item(
        fixture["work_memory"],
        old_item.role,
        entity_identity(source, (28630, entry_index + 1)),
        fixture["anchor"],
        10 + entry_index,
        supersedes=(old_item.content_ref(),),
    )
    new_dependency = AttractorDependency(
        _instruction(source, 20 + entry_index), new_item.value)
    new_entry = SituationProjectionEntry(
        old_entry.projection_key,
        old_entry.projection_kind,
        new_item.content_ref(),
        (new_dependency,),
        old_entry.revision + 1,
    )
    return old_entry, SituationProjectionReplacement(new_entry, new_item)


def _update(fixture, dependencies):
    """建立属于当前 query 的真实 A-10 后文更新。"""
    scope = fixture["scope"]
    return AttractorContextUpdate(
        scope,
        LogicalTimestamp(
            LogicalClockIdentity(scope, CLOCK_MEMORY_OBSERVED), 3),
        _instruction(fixture["source"], 40),
        tuple(dependencies),
    )


def _close(fixture) -> None:
    """关闭临时后端，不把测试写入任何正式宿主。"""
    fixture["backend"].close()
    fixture["graph_backend"].close()


def test_real_revision_is_local_and_preserves_original_event_and_bytes():
    """后文修正只重建依赖槽位，旧事件与未受影响投影逐字节不变。"""
    fixture = _fixture()
    try:
        facade = fixture["facade"]
        projection = fixture["projection"]
        original = facade.append(_observation_event(
            fixture["ctx"], fixture["source"], seq=1))
        original_bytes = original.event.stable_key()
        revision = facade.append(_observation_event(
            fixture["ctx"], fixture["source"], seq=2))
        before_bytes = projection.entry_bytes()
        old_entry, replacement = _replacement(fixture)
        receipt = projection.apply_revision(
            _update(fixture, old_entry.dependencies),
            revision,
            (replacement,),
            preserved_event_hashes=(original.event_hash,),
        )

        assert isinstance(receipt, SituationRebuildReceipt)
        assert receipt.invalidated_projection_keys == (
            old_entry.projection_key,)
        assert receipt.rebuilt_projection_keys == (
            old_entry.projection_key,)
        assert receipt.work_memory_write_count == 1
        assert receipt.host_learning_write_count == 0
        assert receipt.unaffected_bit_identical == 1
        assert receipt.original_events_preserved == 1
        after_bytes = projection.entry_bytes()
        for key in receipt.unaffected_projection_keys:
            assert after_bytes[key] == before_bytes[key]
        assert after_bytes[old_entry.projection_key] != before_bytes[
            old_entry.projection_key]
        assert facade.read(original.event_hash).event.stable_key() == original_bytes
        history = fixture["work_memory"].content_history()
        assert old_entry.content_ref in {item.content_ref() for item in history}
        assert replacement.entry.content_ref in {
            item.content_ref() for item in history}
        assert projection.work_memory is fixture[
            "work_memory"].require_content_store()
        assert facade.event_log is fixture["ctx"].memory_interact_events
    finally:
        _close(fixture)


def test_dependency_index_rebuild_removes_old_edge_and_keeps_unaffected_edges():
    """局部替换后旧 dependency 不再命中，新 dependency 精确命中同一槽位。"""
    fixture = _fixture()
    try:
        facade = fixture["facade"]
        original = facade.append(_observation_event(
            fixture["ctx"], fixture["source"], seq=1))
        revision = facade.append(_observation_event(
            fixture["ctx"], fixture["source"], seq=2))
        projection = fixture["projection"]
        old_entry, replacement = _replacement(fixture)
        unaffected_dependency = next(
            entry.dependencies[0] for entry in projection.entries()
            if entry.projection_key != old_entry.projection_key)
        projection.apply_revision(
            _update(fixture, old_entry.dependencies), revision, (replacement,),
            preserved_event_hashes=(original.event_hash,))
        assert projection.dependency_index.affected(old_entry.dependencies) == ()
        assert projection.dependency_index.affected(
            replacement.entry.dependencies) == (old_entry.projection_key,)
        assert projection.dependency_index.affected(
            (unaffected_dependency,))
    finally:
        _close(fixture)


def test_unmatched_dependency_and_incomplete_rebuild_fail_before_work_memory_write():
    """未命中或漏槽位的修正必须在真实 transient store 首写前失败。"""
    fixture = _fixture()
    try:
        facade = fixture["facade"]
        original = facade.append(_observation_event(
            fixture["ctx"], fixture["source"], seq=1))
        revision = facade.append(_observation_event(
            fixture["ctx"], fixture["source"], seq=2))
        state = fixture["work_memory"].content_state_key()
        projection_state = fixture["projection"].state_key()
        unknown = AttractorDependency(
            _instruction(fixture["source"], 99),
            entity_identity(fixture["source"], (28699, 1)),
        )
        with pytest.raises(SituationStateError, match="未命中"):
            fixture["projection"].apply_revision(
                _update(fixture, (unknown,)), revision, (),
                preserved_event_hashes=(original.event_hash,))
        assert fixture["work_memory"].content_state_key() == state
        assert fixture["projection"].state_key() == projection_state

        old_entry, replacement = _replacement(fixture)
        second_entry = next(
            item for item in fixture["projection"].entries()
            if item.projection_key != old_entry.projection_key)
        with pytest.raises(SituationStateError, match="精确覆盖"):
            fixture["projection"].apply_revision(
                _update(fixture, (
                    old_entry.dependencies[0], second_entry.dependencies[0])),
                revision,
                (replacement,),
                preserved_event_hashes=(original.event_hash,),
            )
        assert fixture["work_memory"].content_state_key() == state
        assert fixture["projection"].state_key() == projection_state
    finally:
        _close(fixture)


def test_scope_source_and_old_event_proof_fail_closed_before_projection_write():
    """scope/source 越权或缺原事件证据都不能触及当前投影。"""
    fixture = _fixture()
    try:
        facade = fixture["facade"]
        original = facade.append(_observation_event(
            fixture["ctx"], fixture["source"], seq=1))
        revision = facade.append(_observation_event(
            fixture["ctx"], fixture["source"], seq=2))
        old_entry, replacement = _replacement(fixture)
        state = fixture["work_memory"].content_state_key()
        other_scope = query_scope(
            10, parent=fixture["scope"].parent)
        bad_update = AttractorContextUpdate(
            other_scope,
            LogicalTimestamp(
                LogicalClockIdentity(other_scope, CLOCK_MEMORY_OBSERVED), 3),
            _instruction(fixture["source"], 40),
            old_entry.dependencies,
        )
        with pytest.raises(SituationStateError, match="scope 越权"):
            fixture["projection"].apply_revision(
                bad_update, revision, (replacement,),
                preserved_event_hashes=(original.event_hash,))
        with pytest.raises(SituationStateError, match="revision 之前"):
            fixture["projection"].apply_revision(
                _update(fixture, old_entry.dependencies),
                revision,
                (replacement,),
                preserved_event_hashes=(revision.event_hash,),
            )
        assert fixture["work_memory"].content_state_key() == state

        foreign_source = _source(72)
        with pytest.raises(SituationStateError, match="source/owner/version"):
            facade.append(_observation_event(
                fixture["ctx"], foreign_source, seq=1))
        with pytest.raises(
                SituationStateError, match="source/scope/owner/version"):
            SituationEventLog(
                fixture["ctx"].memory_interact_events,
                foreign_source,
                fixture["scope"],
            )
        foreign_owner_source = replace(
            fixture["source"],
            owner=OwnerScope(1, 2, 3, VISIBILITY_SESSION),
        )
        with pytest.raises(
                SituationStateError, match="source/scope/owner/version"):
            SituationEventLog(
                fixture["ctx"].memory_interact_events,
                foreign_owner_source,
                fixture["scope"],
            )
        foreign_version_source = replace(
            fixture["source"],
            versions=VersionBundle(parser=ParserVersion(1)),
        )
        with pytest.raises(
                SituationStateError, match="source/scope/owner/version"):
            SituationEventLog(
                fixture["ctx"].memory_interact_events,
                foreign_version_source,
                fixture["scope"],
            )
    finally:
        _close(fixture)


def test_replacement_must_be_single_step_same_kind_and_exact_supersede():
    """替换只能推进同槽同类一个 revision，并只 supersede 命中的前项。"""
    fixture = _fixture()
    try:
        old_entry, replacement = _replacement(fixture)
        with pytest.raises(SituationStateError, match="kind 或 revision"):
            bad = replace(replacement.entry, revision=old_entry.revision + 2)
            revision = fixture["facade"].append(_observation_event(
                fixture["ctx"], fixture["source"], seq=2))
            original = fixture["facade"].append(_observation_event(
                fixture["ctx"], fixture["source"], seq=1))
            fixture["projection"].apply_revision(
                _update(fixture, old_entry.dependencies), revision,
                (SituationProjectionReplacement(
                    bad, replacement.content_item),),
                preserved_event_hashes=(original.event_hash,),
            )
        assert old_entry in fixture["projection"].entries()
    finally:
        _close(fixture)


def test_discourse_adapter_requires_exact_role_and_dependency_maps():
    """G-02 映射不得漏 Role/dependency 或携带未消费额外项。"""
    fixture = _fixture()
    try:
        entries = fixture["projection"].entries()
        with pytest.raises(SituationStateError, match="缺 projection"):
            CurrentSituationProjection.from_work_memory_discourse(
                fixture["facade"],
                fixture["work_memory"].require_content_store(),
                fixture["scope"],
                WorkMemoryDiscourseProjection(
                    (1,), (2,),
                    WorkMemoryDiscourseRoles(*tuple(
                        item.role for item in fixture["items"])),
                    fixture["items"],
                ),
                kind_by_role={},
                dependencies_by_content_ref={
                    item.content_ref(): entry.dependencies
                    for item, entry in zip(fixture["items"], entries)},
            )
    finally:
        _close(fixture)


def _manifest() -> MD02AdapterManifest:
    """按当前不可变前置 hash 建立 MD-02 artifact。"""
    return build_md02_adapter_manifest(
        md01_manifest_relative_path=(
            "data/ph2/manifests/md01_memory_dynamics_contract_v1.json"),
        md01_manifest_sha256=hashlib.sha256(MD01_PATH.read_bytes()).hexdigest(),
        baseline_manifest_relative_path=(
            "data/ph2/manifests/language_capability_baseline_v18.json"),
        baseline_manifest_sha256=hashlib.sha256(
            BASELINE_PATH.read_bytes()).hexdigest(),
    )


def test_manifest_round_trip_nonoverwrite_and_strict_fields(tmp_path):
    """MD-02 manifest 规范回读、幂等发布并拒绝覆盖或额外字段。"""
    manifest = _manifest()
    output = tmp_path / "md02.json"
    write_md02_adapter_manifest(manifest, output)
    assert read_md02_adapter_manifest(output) == manifest
    write_md02_adapter_manifest(manifest, output)
    output.write_bytes(canonical_json_line({"damaged": 1}))
    with pytest.raises(MD02ManifestError, match="内容不同"):
        write_md02_adapter_manifest(manifest, output)
    value = manifest.to_dict()
    value["mastered"] = 1
    output.write_bytes(canonical_json_line(value))
    with pytest.raises(MD02ManifestError, match="字段不精确"):
        read_md02_adapter_manifest(output)


def test_manifest_rejects_fake_probe_host_write_and_bad_prerequisite():
    """adapter T0 不得冒充 probe，且任何 host write/坏 hash 都失败。"""
    manifest = _manifest()
    with pytest.raises(MD02ManifestError, match="probe"):
        replace(manifest, probe_status="COMPLETE")
    with pytest.raises(MD02ManifestError, match="host learning write"):
        replace(manifest, host_learning_write_count=1)
    with pytest.raises(MD02ManifestError, match="SHA-256"):
        replace(manifest, md01_manifest_sha256="bad")
    with pytest.raises(MD02ManifestError, match="安全 POSIX"):
        replace(manifest, baseline_manifest_relative_path="../private.json")


def test_manifest_lists_reused_engines_and_no_second_engine_invariant():
    """正式边界必须列出复用底座并显式禁止第二套长期引擎。"""
    manifest = _manifest()
    assert "NO_SECOND_MEMORY_DATASET_EVIDENCE_ENGINE" in manifest.invariant_keys
    assert set(manifest.reused_component_refs) >= {
        "src/pure_integer_ai/cognition/shared/attractor_state.py",
        "src/pure_integer_ai/cognition/shared/memory_event_log.py",
        "src/pure_integer_ai/cognition/shared/parser_revision.py",
        "src/pure_integer_ai/cognition/shared/work_memory_content.py",
        "src/pure_integer_ai/cognition/shared/work_memory_discourse.py",
    }
    assert all(value == 0 for value in manifest.execution_state.to_value().values())
    assert manifest.results_observed == 0


def test_repository_md02_manifest_matches_current_immutable_prerequisites():
    """正式 MD-02 artifact 必须精确绑定 MD-01 和 v18 基线。"""
    assert read_md02_adapter_manifest(MANIFEST_PATH) == _manifest()
