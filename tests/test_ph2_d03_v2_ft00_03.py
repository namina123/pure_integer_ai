"""FT00-03 v2 streaming reader、logical shard、window 和 checkpoint 专项。"""
from __future__ import annotations

from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_d03_v2_authority import (
    V2_LOGICAL_SHARD_COUNT,
    V2_RELEASE_KEY,
    V2RunIdentity,
)
from pure_integer_ai.experiments.ph2_d03_v2_registry import V2PackEntry
from pure_integer_ai.experiments.ph2_d03_v2_streaming import (
    V2LogicalShardPlan,
    V2StreamCheckpoint,
    V2StreamReader,
    V2StreamingError,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    ArtifactManifest,
    StableRecordKey,
    record_from_dict,
)
from pure_integer_ai.experiments.ph2_dataset_io import (
    ArtifactWriteSpec,
    write_artifact_manifest,
    write_record_artifact,
)


SHA0 = "0" * 64
SHA1 = "1" * 64


def _carrier(node: int) -> dict:
    """形成一个最小但未展平的 plain_text carrier。"""
    return {
        "carrier": {
            "carrier_kind": "plain_text",
            "edges": [],
            "nodes": [{
                "attributes": {},
                "node_key": [node],
                "node_kind": "content",
                "parent_node_key": None,
                "span_end": 1,
                "span_start": 0,
            }],
            "raw_text_sha256": SHA0,
            "root_node_keys": [[node]],
        },
        "language_payload": {"text": "甲"},
    }


def _source() -> dict:
    """形成来源记录。"""
    return {
        "artifact_key": [2],
        "attribution": "FT00-03 authored fixture",
        "course_version": 2,
        "dataset_key": [1],
        "format_version": 2,
        "license_id": "CC0-1.0",
        "local_sha256": SHA1,
        "official_url": "urn:ft00-03:source",
        "parser_version": 2,
        "record_kind": "source_ref",
        "record_ordinal": 1,
        "redistribution_policy": "PUBLIC",
        "revision_id": "",
        "schema_version": 2,
        "snapshot_id": "FT00-03-V1",
        "source_cluster_key": [100],
        "source_identity": "AUTHORED_CC0:1",
        "source_key": "AUTHORED_CC0",
        "source_span": {
            "document_cluster_key": [101],
            "entity_graph_cluster_key": [102],
            "locator_kind": "record",
            "locator_value": "1",
            "span_end": 1,
            "span_start": 0,
        },
        "stable_key": [1],
        "upstream_checksum": "sha256:" + SHA0,
    }


def _observation(stable: int, split: str, cluster: int) -> dict:
    """形成 train/dev Observation。"""
    return {
        "artifact_key": [2],
        "content_group_key": [cluster + 1],
        "course_version": 2,
        "dataset_key": [1],
        "dedup_cluster_key": [cluster],
        "epistemic_role": "forming",
        "format_version": 2,
        "language": "zh",
        "license_partition": "CC0-1.0",
        "logical_order": stable,
        "payload_kind": "typed_carrier",
        "perturbation_kind": "NONE",
        "prerequisite_keys": [],
        "record_kind": "observation",
        "representation": "plain_text",
        "sample_role": "support",
        "schema_version": 2,
        "shape_group_key": [cluster + 4],
        "source_ref_key": [1],
        "split": split,
        "stable_key": [stable],
        "substage": "FT01_FORMAL_FOUNDATION",
        "supersedes_key": None,
        "template_group_key": [cluster + 3],
        "typed_payload": _carrier(cluster + 5),
        "w_stage": "W-02",
    }


def _teacher() -> dict:
    """形成 train-only TeacherEvidence。"""
    return {
        "artifact_key": [2],
        "course_version": 2,
        "dataset_key": [1],
        "evidence_kind": "AUTHORED_FORM",
        "format_version": 2,
        "observation_key": [2],
        "owner_key": [9001],
        "record_kind": "teacher_evidence",
        "schema_version": 2,
        "source_ref_key": [1],
        "stable_key": [3],
        "typed_evidence": {"accepted": 1},
        "visible_from_stage": "W-02",
        "withdrawal_level": 0,
    }


def _label() -> dict:
    """形成 dev-only evaluator label。"""
    return {
        "artifact_key": [2],
        "budget_units": 1,
        "course_version": 2,
        "dataset_key": [1],
        "dimension_key": [7001],
        "evaluator_version": 2,
        "expected_payload": {"accepted": 1},
        "expected_state": "TRUE",
        "format_version": 2,
        "observation_key": [4],
        "owner_key": [9002],
        "owner_mode": "read_only",
        "record_kind": "evaluator_label",
        "schema_version": 2,
        "stable_key": [5],
        "visible_stage": "W-02",
    }


def _build_pack(root: Path) -> str:
    """写入真实 gzip payload 与 v2 manifest，返回 manifest 相对路径。"""
    pack_root = root / "data/packs/p1"
    source = record_from_dict(_source())
    train = record_from_dict(_observation(2, "train", 200))
    dev = record_from_dict(_observation(4, "dev", 300))
    teacher = record_from_dict(_teacher())
    label = record_from_dict(_label())
    cluster = (StableRecordKey((100,)),)
    specs = (
        ("source_ref", "source", "source_refs.jsonl.gz", None,
         (source,)),
        ("observation", "observation", "observations/train.jsonl.gz",
         "train", (train,)),
        ("observation", "observation", "observations/dev.jsonl.gz",
         "dev", (dev,)),
        ("teacher_evidence", "teacher",
         "owners/teacher/train.evidence.jsonl.gz", "train",
         (teacher,)),
        ("evaluator_label", "evaluator",
         "owners/evaluator/dev.labels.jsonl.gz", "dev", (label,)),
    )
    identities = []
    for kind, owner, relative, split, records in specs:
        identities.append(write_record_artifact(
            records, pack_root,
            ArtifactWriteSpec(kind, owner, relative, split, "CC0-1.0", cluster)))
    manifest = ArtifactManifest(
        2, 2, 2, 2,
        StableRecordKey((1,)), StableRecordKey((1,)), "AUTHORED_CC0", "CC0-1.0",
        "PUBLIC", 2, 2, 2, tuple(identities), ("train", "dev"), ("W-02",),
        cluster, (), "W-02")
    relative = "data/packs/p1/manifest.json"
    write_artifact_manifest(manifest, root, relative_path=relative)
    return relative


def test_stream_reader_keeps_owner_views_and_reads_record_by_record(tmp_path: Path) -> None:
    """candidate/teacher/dev/shadow 视图物理隔离，private 入口 fail closed。"""
    relative = _build_pack(tmp_path)
    entry = V2PackEntry.from_manifest(tmp_path, relative)
    reader = V2StreamReader(tmp_path, entry)
    assert [item.stable_key.components for item in reader.iter_records("candidate")] == [
        (1,), (2,)]
    assert [item.stable_key.components for item in reader.iter_records("teacher")] == [
        (1,), (2,), (3,)]
    assert [item.stable_key.components for item in reader.iter_records(
        "dev", split="dev")] == [(1,), (4,)]
    assert [item.stable_key.components for item in reader.iter_records("shadow")] == [
        (1,), (2,), (4,)]
    with pytest.raises(V2StreamingError, match="private"):
        tuple(reader.iter_records("private_evaluator"))


def test_stream_windows_resume_and_worker_independent_shards(tmp_path: Path) -> None:
    """window cursor 可恢复，logical shard 只由 stable key 决定。"""
    relative = _build_pack(tmp_path)
    entry = V2PackEntry.from_manifest(tmp_path, relative)
    reader = V2StreamReader(tmp_path, entry)
    windows = list(reader.iter_windows("candidate", window_size=1))
    assert [[item.stable_key.components for item in page.records]
            for page in windows] == [[(1,)], [(2,)]]
    assert windows[-1].complete == 1
    plan_a = V2LogicalShardPlan()
    plan_b = V2LogicalShardPlan()
    assert all(plan_a.shard_for((index,)) == plan_b.shard_for((index,))
               for index in range(1, 12))
    shard = plan_a.shard_for((2,))
    run = V2RunIdentity(
        V2_RELEASE_KEY, "W-02", "P0", 1, V2_LOGICAL_SHARD_COUNT,
        "a" * 64, "")
    checkpoint = reader.checkpoint(
        run, owner_key="PH2_V2_CANDIDATE", shard_index=shard,
        cursor_record_key=(1,), source_state_sha256="b" * 64)
    assert V2StreamCheckpoint.from_dict(checkpoint.to_dict()) == checkpoint
    resumed = list(reader.iter_from_checkpoint(
        checkpoint, run, owner_key="PH2_V2_CANDIDATE",
        source_state_sha256="b" * 64, window_size=8))
    assert [item.stable_key.components
            for page in resumed for item in page.records] == (
                [(2,)] if plan_a.shard_for((2,)) == shard else [])
    with pytest.raises(V2StreamingError, match="source state"):
        list(reader.iter_from_checkpoint(
            checkpoint, run, owner_key="PH2_V2_CANDIDATE",
            source_state_sha256="c" * 64, window_size=8))


def test_stream_rejects_transport_drift_and_checkpoint_shape(tmp_path: Path) -> None:
    """payload transport 漂移、checkpoint 未知字段和 bool 下标都拒绝。"""
    relative = _build_pack(tmp_path)
    entry = V2PackEntry.from_manifest(tmp_path, relative)
    reader = V2StreamReader(tmp_path, entry)
    payload = tmp_path / "data/packs/p1/observations/train.jsonl.gz"
    payload.write_bytes(payload.read_bytes() + b"x")
    with pytest.raises(V2StreamingError, match="transport"):
        tuple(reader.iter_records("candidate"))
    value = {
        "checkpoint_format_version": 1,
        "release_key": V2_RELEASE_KEY,
        "run_identity_sha256": "a" * 64,
        "owner_key": "PH2_V2_CANDIDATE",
        "pack_key": [1],
        "source_state_sha256": "b" * 64,
        "logical_shard_index": True,
        "cursor_record_key": [],
        "input_manifest_sha256": "c" * 64,
        "extra": 1,
    }
    with pytest.raises(V2StreamingError):
        V2StreamCheckpoint.from_dict(value)
