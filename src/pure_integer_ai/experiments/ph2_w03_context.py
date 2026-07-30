"""不解码任何训练 payload 地打开 W-03 冻结 context。"""
from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath

from pure_integer_ai.experiments.ph2_d03_contract_core import (
    D03ContractError,
    D03FileIdentity,
)
from pure_integer_ai.experiments.ph2_d03_publication import (
    read_d03_publication_receipt,
)
from pure_integer_ai.experiments.ph2_d03_release_catalog import (
    FORMAL_GLOBAL_MANIFEST_PATH,
    FORMAL_RECEIPT_PATH,
)
from pure_integer_ai.experiments.ph2_d03_release_reader import (
    D03ReleaseReader,
    VisibleArtifactFile,
)
from pure_integer_ai.experiments.ph2_dataset_io import read_artifact_manifest
from pure_integer_ai.experiments.ph2_w03_continuity import (
    W03PublicationBaseline,
    W03W02ContinuityBinding,
    formal_w03_publication_baseline,
)
from pure_integer_ai.experiments.ph2_w03_contract import (
    W03_ABLATION_KEYS,
    W03_AGGREGATION_POLICY,
    W03_ALLOWED_WORKER_COUNTS,
    W03_D03_ABLATION_KEYS,
    W03_DIMENSION_KEY_MAP,
    W03_DIMENSION_KEYS,
    W03_EVALUATION_ORDER,
    W03_FORBIDDEN_WRITE_OWNERS,
    W03_FORMAL_RUN_ID,
    W03_GENERATION_HARD_CONJUNCT,
    W03_LOGICAL_CLOCK_VERSION,
    W03_OWNER_KEY,
    W03_RESOURCE_BUDGET,
    W03_STAGE_KEY,
    W03_TRAIN_PACK_KEYS,
    W03_W02_BASE_RUN_ID,
    W03_ZERO_EXECUTION_STATE,
    W03_ALLOWED_WRITE_OWNERS,
    W03ContractError,
    W03FrozenContext,
    W03PackBinding,
    W03PayloadBinding,
    strict_key,
)


D03_RECEIPT_SHA256 = "8efd5f8c559bb22f0d2587fea4d38ee94d2dc10cf13ca0f787f3489f45847aef"
D03_GLOBAL_MANIFEST_SHA256 = (
    "384329cf651ea4c5e4bc9d0b5dc4da7b22a71bc008bfabe468c86278dd9d40b6"
)
W03_STAGE_MANIFEST_SHA256 = (
    "b32a3174e9d224fb702bc378e311aa9f1d07f205feca90e80f9a81a17ecfcd3b"
)


def _overlay_path(primary: Path, dependency: Path, relative: str) -> Path:
    parts = Path(*PurePosixPath(relative).parts)
    for root in (primary, dependency):
        target = (root / parts).resolve()
        if target.is_relative_to(root) and target.is_file():
            return target
    raise W03ContractError(f"frozen W-03 metadata file is missing: {relative}")


def _file_identity(primary: Path, dependency: Path, relative: str) -> D03FileIdentity:
    path = _overlay_path(primary, dependency, relative)
    payload = path.read_bytes()
    return D03FileIdentity(relative, len(payload), hashlib.sha256(payload).hexdigest())


def _payload_binding(item: VisibleArtifactFile) -> W03PayloadBinding:
    identity = item.file_identity
    return W03PayloadBinding(
        relative_path=item.relative_path,
        pack_key=item.pack_key,
        owner_kind=identity.owner_kind,
        split=identity.split,
        record_count=identity.record_count,
        transport_size_bytes=identity.transport_size_bytes,
        transport_sha256=identity.transport_sha256,
        content_size_bytes=identity.content_size_bytes,
        content_sha256=identity.content_sha256,
        file_identity=identity,
    )


def _verify_pack_manifest(primary: Path, dependency: Path, pack) -> None:
    """逐字段闭合 pack manifest 的来源、许可、路径、owner 和 split。"""
    path = _overlay_path(primary, dependency, pack.manifest_identity.relative_path)
    payload = path.read_bytes()
    if (len(payload) != pack.manifest_identity.size_bytes
            or hashlib.sha256(payload).hexdigest() != pack.manifest_identity.sha256):
        raise W03ContractError("W-03 pack manifest identity drifted")
    manifest = read_artifact_manifest(path)
    expected_splits = tuple(
        split for split, paths in (
            ("train", pack.train_observation_paths),
            ("dev", pack.dev_observation_paths),
            ("held_out", pack.held_out_observation_paths),
        ) if paths
    )
    if (manifest.source_key != pack.source_key
            or manifest.license_partition != pack.license_id
            or manifest.record_count != pack.total_record_count
            or len(manifest.source_cluster_keys) != pack.source_cluster_count
            or manifest.splits != expected_splits
            or pack.earliest_stage not in manifest.w_stages):
        raise W03ContractError("W-03 pack source/license/count/stage drifted")
    prefix = PurePosixPath(pack.manifest_identity.relative_path).parent
    files = {
        PurePosixPath(prefix, item.relative_path).as_posix(): item
        for item in manifest.files
    }
    if set(files) != set(pack.payload_paths):
        raise W03ContractError("W-03 pack manifest does not cover its exact paths")
    expected_owner_split: dict[str, tuple[str, str | None]] = {}
    for paths, owner, split in (
            (pack.source_ref_paths, "source", None),
            (pack.train_observation_paths, "observation", "train"),
            (pack.dev_observation_paths, "observation", "dev"),
            (pack.held_out_observation_paths, "observation", "held_out"),
            ):
        expected_owner_split.update({path: (owner, split) for path in paths})
    for relative in pack.teacher_evidence_paths:
        expected_owner_split[relative] = ("teacher", files[relative].split)
    for relative in pack.evaluator_label_paths:
        expected_owner_split[relative] = ("evaluator", files[relative].split)
    if any((item.owner_kind, item.split) != expected_owner_split[relative]
           for relative, item in files.items()):
        raise W03ContractError("W-03 pack manifest owner/split mapping drifted")


def open_w03_frozen_context(
        repository_root: str | Path,
        global_manifest_path: str = FORMAL_GLOBAL_MANIFEST_PATH,
        *,
        current_remote_commit_sha1: str,
        w02_continuity: W03W02ContinuityBinding,
        publication_baseline: W03PublicationBaseline,
        backend_profile_key: tuple[int, ...],
        dependency_root: str | Path | None = None,
        ) -> W03FrozenContext:
    """只读 receipts/manifests，并保持 payload、交付和写计数为零。"""
    if not isinstance(w02_continuity, W03W02ContinuityBinding):
        raise W03ContractError("W-02 continuity binding type is invalid")
    if not isinstance(publication_baseline, W03PublicationBaseline):
        raise W03ContractError("W-03 publication baseline type is invalid")
    formal_baseline = formal_w03_publication_baseline()
    if publication_baseline.stable_key() != formal_baseline.stable_key():
        raise W03ContractError("W-03 reader-fix publication baseline drifted")
    if current_remote_commit_sha1 != publication_baseline.head_sha1:
        raise W03ContractError("current remote commit is not the green W03-00C baseline")
    strict_key(backend_profile_key, label="backend profile key")
    primary = Path(repository_root).resolve()
    dependency = (Path(dependency_root).resolve()
                  if dependency_root is not None else primary)
    try:
        d03_receipt_path = _overlay_path(primary, dependency, FORMAL_RECEIPT_PATH)
        d03_receipt = read_d03_publication_receipt(d03_receipt_path)
        reader = D03ReleaseReader.open(
            primary,
            global_manifest_path,
            dependency_root=dependency,
            require_publication=True,
        )
    except W03ContractError:
        raise
    except (D03ContractError, OSError, TypeError, ValueError, KeyError) as exc:
        raise W03ContractError(f"D-03/W-03 frozen metadata is invalid: {exc}") from exc
    receipt_identity = _file_identity(primary, dependency, FORMAL_RECEIPT_PATH)
    global_identity = _file_identity(primary, dependency, global_manifest_path)
    if (receipt_identity.sha256 != D03_RECEIPT_SHA256
            or global_identity.sha256 != D03_GLOBAL_MANIFEST_SHA256
            or d03_receipt.global_manifest_identity != global_identity
            or d03_receipt.execution_state.get("d03_published") != 1
            or d03_receipt.publication_state.d03_published != 1):
        raise W03ContractError("D-03 post-publication receipt/global identity drifted")

    stage_index = 2
    stage = reader.stages[stage_index]
    stage_reference = reader.global_manifest.stage_manifests[stage_index]
    if stage_reference.artifact_key != W03_STAGE_KEY:
        raise W03ContractError("D-03 W-03 stage reference drifted")
    stage_identity = stage_reference.file_identity
    if (stage_identity.size_bytes != 5074
            or stage_identity.sha256 != W03_STAGE_MANIFEST_SHA256):
        raise W03ContractError("W-03 stage manifest identity drifted")
    candidate_view = reader.visibility(W03_STAGE_KEY, "candidate")
    teacher_view = reader.visibility(W03_STAGE_KEY, "teacher")
    evaluator_view = reader.visibility(W03_STAGE_KEY, "evaluator")
    if (tuple(len(view.allowed_paths) for view in (
            candidate_view, teacher_view, evaluator_view)) != (12, 18, 29)
            or any(view.payload_reads != 0 or view.payload_bytes != 0
                   for view in (candidate_view, teacher_view, evaluator_view))):
        raise W03ContractError("W-03 12/18/29 zero-payload visibility drifted")
    candidate_traces = reader.visible_file_identities(W03_STAGE_KEY, "candidate")
    teacher_traces = reader.visible_file_identities(W03_STAGE_KEY, "teacher")
    evaluator_traces = reader.visible_file_identities(W03_STAGE_KEY, "evaluator")
    candidate_paths = {item.relative_path for item in candidate_traces}
    teacher_only_traces = tuple(
        item for item in teacher_traces if item.relative_path not in candidate_paths)
    if len(teacher_only_traces) != 6:
        raise W03ContractError("W-03 teacher-only binding count drifted")

    pack_catalog = {
        item.pack_key: item for item in reader.global_manifest.pack_bindings}
    packs: list[W03PackBinding] = []
    for key in W03_TRAIN_PACK_KEYS:
        pack = pack_catalog.get(key)
        if pack is None:
            raise W03ContractError("W-03 frozen train pack is missing")
        _verify_pack_manifest(primary, dependency, pack)
        packs.append(W03PackBinding(
            pack_key=pack.pack_key,
            source_key=pack.source_key,
            license_id=pack.license_id,
            earliest_stage=pack.earliest_stage,
            manifest_identity=pack.manifest_identity,
            total_record_count=pack.total_record_count,
            source_cluster_count=pack.source_cluster_count,
        ))
    evaluation = stage.evaluation_binding
    recovery = stage.recovery_binding
    release = reader.global_manifest.release_identity
    return W03FrozenContext(
        current_remote_commit_sha1=current_remote_commit_sha1,
        publication_baseline=publication_baseline,
        w02_continuity=w02_continuity,
        d03_receipt_identity=receipt_identity,
        d03_global_manifest_identity=global_identity,
        stage_manifest_identity=stage_identity,
        stage_key=stage.stage_identity.stage_key,
        stage_ordinal=stage.stage_identity.ordinal,
        prerequisite_stage_keys=stage.stage_identity.prerequisite_stage_keys,
        train_pack_keys=stage.data_visibility.train_pack_keys,
        pack_bindings=tuple(packs),
        candidate_payload_bindings=tuple(_payload_binding(item)
                                         for item in candidate_traces),
        teacher_evidence_bindings=tuple(_payload_binding(item)
                                        for item in teacher_only_traces),
        evaluator_visible_bindings=tuple(_payload_binding(item)
                                         for item in evaluator_traces),
        d03_thresholds=evaluation.thresholds,
        d03_ablation_keys=evaluation.ablation_keys,
        dimension_key_map=W03_DIMENSION_KEY_MAP,
        dimension_keys=W03_DIMENSION_KEYS,
        ablation_keys=W03_ABLATION_KEYS,
        generation_hard_conjunct=W03_GENERATION_HARD_CONJUNCT,
        evaluation_order=W03_EVALUATION_ORDER,
        aggregation_policy=W03_AGGREGATION_POLICY,
        allowed_worker_counts=recovery.allowed_worker_counts,
        failure_point_keys=recovery.failure_point_keys,
        logical_shard_count=recovery.logical_shard_count,
        merge_barrier_key=recovery.merge_barrier_key,
        cursor_version=recovery.cursor_version,
        logical_clock_version=W03_LOGICAL_CLOCK_VERSION,
        resource_budget=stage.resource_budget.to_dict(),
        version_keys=release.version_keys,
        run_id=W03_FORMAL_RUN_ID,
        parent_run_id=W03_W02_BASE_RUN_ID,
        base_run_id=W03_W02_BASE_RUN_ID,
        backend_profile_key=backend_profile_key,
        base_fence_key=w02_continuity.base_fence_key(),
        owner_key=W03_OWNER_KEY,
        allowed_write_owners=W03_ALLOWED_WRITE_OWNERS,
        forbidden_write_owners=W03_FORBIDDEN_WRITE_OWNERS,
        execution_state=dict(W03_ZERO_EXECUTION_STATE),
    )


__all__ = ["open_w03_frozen_context"]
