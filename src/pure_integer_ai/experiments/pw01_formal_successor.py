"""从已封存 PW-00A base 重建并执行 PW-01 正式受控阅读纵切。"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from pure_integer_ai.cognition.shared.formal_post_weaning import (
    FormalPostWeaningLoadRequest,
    FormalPostWeaningManifest,
)
from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
)
from pure_integer_ai.cognition.shared.memory_event import (
    MEMORY_EVENT_USE,
    MEMORY_OBJECT_OBSERVATION,
    MemoryObjectRef,
)
from pure_integer_ai.cognition.shared.memory_batch import (
    install_memory_batch_runtimes,
)
from pure_integer_ai.cognition.shared.memory_overlay import MemoryAccessContext
from pure_integer_ai.cognition.shared.post_weaning import PostWeaningIntakeRequest
from pure_integer_ai.cognition.shared.types import WEANING_POST
from pure_integer_ai.experiments.evaluation_isolation import isolated_evaluation
from pure_integer_ai.experiments.facility_readiness_scenarios import (
    _ACCESS,
    _batch_config,
    _close_outer_lifecycle,
    _core_refs,
    _install_post_weaning_consumers,
    _install_resolver,
    _observation,
    _post_weaning_manifest,
    _publish_projection,
    _query_source,
    _refresh_projection,
)
from pure_integer_ai.experiments.post_weaning_runtime import (
    CoreCanonicalStateReader,
    PostWeaningOperationRuntime,
    post_weaning_component_state_key,
)
from pure_integer_ai.experiments.pw00a_formal_runtime import (
    AUTHORITY_RECEIPT_PATH,
    _owner_identity_key,
    _owner_watermark_key,
    _resume_owner_id_pools,
)
from pure_integer_ai.experiments.pw00a_formal_transaction import (
    PW00A_EVENT_PREPARED,
    PW00A_EVENT_PUBLISHED,
    PW00AFormalEventStore,
)
from pure_integer_ai.experiments.pw00a_inference_artifact import (
    ARTIFACT_PATH as INFERENCE_ARTIFACT_PATH,
    read_pw00a_w09_inference_artifact,
)
from pure_integer_ai.experiments.pw01_controlled_reading import (
    PW01ControlledReadingParser,
    PW01_HYPOTHESIS_KIND,
    build_pw01_question_dialogue,
    install_pw01_controlled_query,
    pw01_source,
)
from pure_integer_ai.experiments.train_context import (
    TrainContext,
    make_train_context,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
)
from pure_integer_ai.storage.backend import SQLiteBackend, StorageBackend


PW01_SUPPORT_BATCH = 2026080721
PW01_REFUTE_BATCH = 2026080722


def _file_digest(root: Path, relative_path: str) -> tuple[int, ...]:
    """读取仓库内固定依赖并返回 32 字节摘要键。"""
    target = root / Path(*relative_path.replace("\\", "/").split("/"))
    return tuple(hashlib.sha256(target.read_bytes()).digest())


def stable_key_sha256(key: tuple[int, ...]) -> str:
    """把严格整数稳定键规范编码成公开安全 SHA-256。"""
    if not isinstance(key, tuple) or any(type(item) is not int for item in key):
        raise TypeError("PW-01 stable key 必须是严格整数 tuple")
    return hashlib.sha256(canonical_json_bytes(list(key))).hexdigest()


def semantic_answer_key(operation: Any) -> tuple[int, ...]:
    """提取不含新 Use 序号的回答语义身份，供 fresh/restart 比较。"""
    question = operation.result.question
    generation = question.generation
    status_key = question.status.stable_key()
    if generation is None:
        return len(status_key), *status_key, 0
    target_key = question.query.request.target.stable_key()
    rendered_key = generation.rendered.stable_key()
    return (
        len(status_key),
        *status_key,
        1,
        len(target_key),
        *target_key,
        len(rendered_key),
        *rendered_key,
    )


def _formal_manifest(
        ctx: TrainContext,
        root: Path,
        *,
        run_id: int,
        publish_epoch: int,
        ) -> FormalPostWeaningManifest:
    """从当前联邦设施形成 successor 专用正式 manifest。"""
    _, dry = _post_weaning_manifest(ctx, _query_source(1))
    _, inference_state = read_pw00a_w09_inference_artifact(root)
    request = FormalPostWeaningLoadRequest(
        run_id,
        publish_epoch,
        dry.runtime_owner,
        _file_digest(root, AUTHORITY_RECEIPT_PATH),
        _file_digest(root, INFERENCE_ARTIFACT_PATH),
        dry.routes,
        dry.probe,
        dry.budget,
        (20260807, 3, 1),
    )
    watermarks = _resume_owner_id_pools(ctx)
    return FormalPostWeaningManifest(
        request,
        tuple(bytes.fromhex(inference_state.sha256())),
        dry.core_state_key,
        dry.schema_state_key,
        dry.backend_state_key,
        dry.component_state_key,
        _owner_identity_key(ctx),
        _owner_watermark_key(ctx, watermarks),
        (20260807, 3, 2),
    )


def assemble_successor_context(
        backend: StorageBackend,
        repository_root: str | Path,
        *,
        run_id: int,
        publish_epoch: int,
        ) -> tuple[TrainContext, Any, Any, FormalPostWeaningManifest,
                   PostWeaningOperationRuntime]:
    """从 base/继承 SQLite 重建投影、联邦查询和正式操作入口。"""
    root = Path(repository_root).resolve()
    ctx = make_train_context(backend, companion=True)
    install_memory_batch_runtimes(ctx, _batch_config())
    source = _query_source(1)
    _, resolver = _install_resolver(ctx, source, _core_refs(ctx)[1])
    ctx.memory_read_aggregates.rebuild_dirty(access=_ACCESS)
    ctx.memory_interact_aggregates.rebuild_dirty(access=_ACCESS)
    projection = _publish_projection(ctx, resolver.resolver)
    projection.validate_store(ctx.tiered_segment_store)
    _install_post_weaning_consumers(ctx, source, projection)
    _post_weaning_manifest(ctx, source)
    install_pw01_controlled_query(ctx)
    ctx.backend.protect_owner_space(ctx.core_space.space_id)
    ctx.weaning_phase = WEANING_POST
    manifest = _formal_manifest(
        ctx, root, run_id=run_id, publish_epoch=publish_epoch)
    runtime = PostWeaningOperationRuntime(
        ctx,
        manifest,
        core_reader=CoreCanonicalStateReader(ctx),
        required_phase=WEANING_POST,
    )
    return ctx, source, projection, manifest, runtime


def _question(
        ctx: TrainContext,
        source: Any,
        observation: Any,
        runtime: PostWeaningOperationRuntime,
        ) -> Any:
    """执行一次 held-out 目标 3 问答并关闭所有临时资源。"""
    _close_outer_lifecycle(ctx)
    _refresh_projection(ctx)
    fixture, dialogue = build_pw01_question_dialogue(
        ctx, source, observation)
    try:
        return runtime.run_question(dialogue, fixture.request)
    finally:
        fixture.close()
        _close_outer_lifecycle(ctx)


def _intake(
        ctx: TrainContext,
        runtime: PostWeaningOperationRuntime,
        source: Any,
        *,
        stance: int,
        batch_id: int,
        lineage_id: int,
        supersedes_source: Any = None,
        ) -> Any:
    """经 successor 正式 reading route 摄入一个受控版本。"""
    return runtime.run_intake(PostWeaningIntakeRequest(
        runtime.manifest.routes.reading,
        source,
        f"PW-01 formal source {source.document_id}",
        "CC0-1.0",
        batch_id,
        parser=PW01ControlledReadingParser(source, stance, lineage_id),
        supersedes_source=supersedes_source,
        trace=(20260807, 3, lineage_id),
    ))


def _fresh_runtime_for_clone(
        clone: TrainContext,
        root: Path,
        *,
        run_id: int,
        publish_epoch: int,
        ) -> PostWeaningOperationRuntime:
    """为已装配的 V-06 clone 形成独立正式操作入口。"""
    clone.weaning_phase = WEANING_POST
    manifest = _formal_manifest(
        clone, root, run_id=run_id, publish_epoch=publish_epoch)
    return PostWeaningOperationRuntime(
        clone,
        manifest,
        core_reader=CoreCanonicalStateReader(clone),
        required_phase=WEANING_POST,
    )


def run_fresh_successor_evidence(
        ctx: TrainContext,
        source: Any,
        runtime: PostWeaningOperationRuntime,
        repository_root: str | Path,
        *,
        run_id: int,
        publish_epoch: int,
        ) -> dict[str, Any]:
    """执行读前、读后、精确消融、坏 revision 回滚与 ACL 全部对抗。"""
    root = Path(repository_root).resolve()
    observations = ctx.memory_interact_events.query(access=_ACCESS)
    observation = next(
        item for item in observations
        if item.event.object_ref.object_kind == MEMORY_OBJECT_OBSERVATION)
    observation_ref = observation.event.object_ref
    core_before = CoreCanonicalStateReader(ctx).read()
    before = _question(ctx, source, observation, runtime)
    if before.result.question.complete:
        raise RuntimeError("PW-01 held-out 在阅读前已可回答")

    learned_source = pw01_source(parser_version=1)
    intake = _intake(
        ctx,
        runtime,
        learned_source,
        stance=EVIDENCE_SUPPORT,
        batch_id=PW01_SUPPORT_BATCH,
        lineage_id=1,
    )
    ctx.memory_read_aggregates.rebuild_dirty(access=_ACCESS)

    same_user_other_session = MemoryAccessContext(1, 2, 4)
    other_user = MemoryAccessContext(1, 9, 4)
    if len(ctx.memory_read_aggregates.query(
            access=same_user_other_session,
            hypothesis_kind=PW01_HYPOTHESIS_KIND,
            source=learned_source)) != 1:
        raise RuntimeError("PW-01 用户阅读记忆没有跨其 session 保持")
    if ctx.memory_read_aggregates.query(
            access=other_user,
            hypothesis_kind=PW01_HYPOTHESIS_KIND,
            source=learned_source):
        raise RuntimeError("PW-01 阅读记忆泄漏到其他用户")
    if ctx.memory_interact_aggregates.query(
            access=same_user_other_session,
            hypothesis_kind=PW01_HYPOTHESIS_KIND):
        raise RuntimeError("PW-01 session 交互记忆泄漏到其他 session")

    with isolated_evaluation(ctx, label="pw01-formal-ablation") as clone:
        clone.work_memory.end_session()
        clone.memory_batch_coordinator.rollback_batch(PW01_SUPPORT_BATCH)
        clone.memory_read_aggregates.rebuild_dirty(access=_ACCESS)
        clone_runtime = _fresh_runtime_for_clone(
            clone, root, run_id=run_id, publish_epoch=publish_epoch)
        ablated = _question(
            clone, source, _observation(clone, observation_ref), clone_runtime)
        if ablated.result.question.complete:
            raise RuntimeError("PW-01 精确来源消融后仍可回答")

    with isolated_evaluation(ctx, label="pw01-formal-bad-revision") as clone:
        clone.work_memory.end_session()
        clone_runtime = _fresh_runtime_for_clone(
            clone, root, run_id=run_id, publish_epoch=publish_epoch)
        bad_source = pw01_source(parser_version=2)
        _intake(
            clone,
            clone_runtime,
            bad_source,
            stance=EVIDENCE_REFUTE,
            batch_id=PW01_REFUTE_BATCH,
            lineage_id=2,
            supersedes_source=learned_source,
        )
        clone.memory_read_aggregates.rebuild_dirty(access=_ACCESS)
        conflicted = _question(
            clone, source, _observation(clone, observation_ref), clone_runtime)
        if conflicted.result.question.complete:
            raise RuntimeError("PW-01 坏 revision 没有改变 held-out 行为")
        clone.memory_batch_coordinator.rollback_batch(PW01_REFUTE_BATCH)
        clone.memory_read_aggregates.rebuild_dirty(access=_ACCESS)
        restored = _question(
            clone, source, _observation(clone, observation_ref), clone_runtime)
        if not restored.result.question.complete:
            raise RuntimeError("PW-01 坏 revision 回滚后没有恢复回答")

    after = _question(ctx, source, observation, runtime)
    if not after.result.question.complete:
        raise RuntimeError("PW-01 阅读后 held-out 仍不可回答")
    if {item.trace.source for item in after.result.sources} != {learned_source}:
        raise RuntimeError("PW-01 阅读后答案没有唯一归因到新增来源")
    uses = ctx.memory_interact_events.query(
        access=_ACCESS, event_kind=MEMORY_EVENT_USE)
    if not uses:
        raise RuntimeError("PW-01 阅读后回答没有形成 Use")
    use = uses[-1].event.payload
    if (use.memory_ref.memory_space
            != ctx.memory_read_events.memory_space_identity
            or use.episode_ref.memory_space
            != ctx.memory_interact_events.memory_space_identity):
        raise RuntimeError("PW-01 跨空间 Use 所有权漂移")
    ctx.memory_read_aggregates.require_clean(access=_ACCESS)
    ctx.memory_interact_aggregates.require_clean(access=_ACCESS)
    core_after = CoreCanonicalStateReader(ctx).read()
    if core_after != core_before:
        raise RuntimeError("PW-01 正式纵切改变了 Core")
    return {
        "after_answer_sha256": stable_key_sha256(semantic_answer_key(after)),
        "before_complete": 0,
        "core_state_sha256": bytes(core_after).hex(),
        "exact_ablation_complete": 0,
        "fresh_operation_count": len(runtime.reports()),
        "intake_report_sha256": stable_key_sha256(intake.report.stable_key()),
        "learned_source_key_sha256": stable_key_sha256(
            learned_source.stable_key()),
        "observation_ref_key": list(observation_ref.stable_key()),
        "other_user_visible": 0,
        "same_user_other_session_visible": 1,
        "session_interaction_leaked": 0,
        "use_candidate_space": list(use.memory_ref.memory_space.stable_key()),
        "use_event_space": list(use.episode_ref.memory_space.stable_key()),
    }


def run_restart_successor_evidence(
        backend: SQLiteBackend,
        repository_root: str | Path,
        fresh: dict[str, Any],
        *,
        run_id: int,
        publish_epoch: int,
        ) -> dict[str, Any]:
    """真重开继承 SQLite，并验证同一新增来源仍形成相同语义回答。"""
    ctx, source, projection, manifest, runtime = assemble_successor_context(
        backend,
        repository_root,
        run_id=run_id,
        publish_epoch=publish_epoch,
    )
    observation_ref = fresh["observation_ref_key"]
    observation = _observation(
        ctx, MemoryObjectRef.from_stable_key(tuple(observation_ref)))
    resumed = _question(ctx, source, observation, runtime)
    answer_sha256 = stable_key_sha256(semantic_answer_key(resumed))
    if (not resumed.result.question.complete
            or answer_sha256 != fresh["after_answer_sha256"]
            or {item.trace.source for item in resumed.result.sources}
            != {pw01_source(parser_version=1)}):
        raise RuntimeError("PW-01 restart 没有保持同一读后回答")
    if CoreCanonicalStateReader(ctx).read() != tuple(
            bytes.fromhex(fresh["core_state_sha256"])):
        raise RuntimeError("PW-01 restart Core 摘要漂移")
    return {
        "component_state_sha256": stable_key_sha256(
            post_weaning_component_state_key(ctx)),
        "formal_manifest_sha256": stable_key_sha256(manifest.stable_key()),
        "projection_record_count": projection.record_count,
        "restart_answer_sha256": answer_sha256,
        "restart_complete": 1,
        "restart_operation_count": len(runtime.reports()),
    }


def validate_pw00a_base_events(
        backend: StorageBackend,
        base_receipt: dict[str, Any],
        ) -> None:
    """要求继承数据库保留 receipt 承诺的唯一 PW-00A 双事件。"""
    events = PW00AFormalEventStore(backend).all_events()
    expected = base_receipt.get("formal_events")
    if (len(events) != 2
            or tuple(item.event_kind for item in events) != (
                PW00A_EVENT_PREPARED, PW00A_EVENT_PUBLISHED)
            or not isinstance(expected, list)
            or len(expected) != 2):
        raise RuntimeError("PW-01 base 缺少唯一 PW-00A 正式事件")
    for event, item in zip(events, expected):
        if (event.run_id != item.get("run_id")
                or event.publish_epoch != item.get("publish_epoch")
                or event.event_seq != item.get("event_seq")
                or event.event_kind != item.get("event_kind")
                or event.manifest_sha256 != item.get("manifest_sha256")
                or event.payload_sha256 != item.get("payload_sha256")):
            raise RuntimeError("PW-01 base 的 PW-00A 事件与 receipt 漂移")


__all__ = [
    "PW01_REFUTE_BATCH",
    "PW01_SUPPORT_BATCH",
    "assemble_successor_context",
    "run_fresh_successor_evidence",
    "run_restart_successor_evidence",
    "semantic_answer_key",
    "stable_key_sha256",
    "validate_pw00a_base_events",
]
