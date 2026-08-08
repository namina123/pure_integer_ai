"""FT00-05 streaming pack, multi-pack intake and P0/P1 report checks."""
from __future__ import annotations

from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_d03_v2_registry import (
    V2GenericTrainer,
    V2PackRegistry,
    V2TrainPackStream,
)
from pure_integer_ai.experiments.ph2_d03_v2_scale_baseline import (
    V2ScaleBaselineError,
    build_authored_scale_pack,
    read_ft00_05_report,
    validate_scale_target,
    write_ft00_05_report,
)
from pure_integer_ai.experiments.ph2_d03_v2_streaming import V2StreamReader
from pure_integer_ai.experiments.ph2_dataset_io import (
    ArtifactWriteSpec,
    DatasetArtifactIOError,
    read_artifact_manifest,
    read_record_artifact,
    write_record_artifact_streaming,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT = (
    ROOT / "data/ph2/manifests/d03_v2/"
    "ph2_d03_v2_ft00_05_scale_baseline_v1.json"
)


def test_authored_fixture_is_streamed_and_out_of_order_input_is_rejected(
        tmp_path: Path) -> None:
    """The fixture has exact 1+2N counts and streaming writes fail closed."""
    relative, digest, observations = build_authored_scale_pack(
        tmp_path, "P0", 7)
    manifest_path = tmp_path / Path(*relative.split("/"))
    manifest = read_artifact_manifest(manifest_path)
    assert manifest.sha256() == digest
    assert observations == 3
    assert manifest.record_count == 7
    observation_file = next(item for item in manifest.files
                            if item.record_kind == "observation")
    records = read_record_artifact(manifest_path.parent, observation_file)
    bad_root = tmp_path / "bad"
    with pytest.raises(DatasetArtifactIOError, match="strictly increasing|严格递增"):
        write_record_artifact_streaming(
            reversed(records), bad_root,
            ArtifactWriteSpec(
                "observation", "observation", "observations/train.jsonl.gz",
                "train", "CC0-1.0", observation_file.source_cluster_keys,
            ),
        )
    assert not (bad_root / "observations/train.jsonl.gz").exists()


def test_generic_trainer_accepts_multiple_pack_owner_namespaces(
        tmp_path: Path) -> None:
    """Independent teacher owners merge only after each pack streams and validates."""
    first, _, _ = build_authored_scale_pack(tmp_path, "P0", 7)
    second, _, _ = build_authored_scale_pack(tmp_path, "P1", 9)
    registry = V2PackRegistry.from_manifest_paths(tmp_path, (second, first))
    plan = registry.train_plan("W-08", scale_key="P0")
    streams = []
    for entry in registry.entries:
        def records(selected=entry):
            return (item.to_dict() for item in V2StreamReader(
                tmp_path, selected).iter_records("teacher"))
        streams.append(V2TrainPackStream(entry.pack_key, records))
    streams = tuple(streams)
    result = V2GenericTrainer().validate_train_streams(plan, streams)
    assert result.source_ref_count == 2
    assert result.observation_count == 7
    assert result.teacher_evidence_count == 7
    assert result.candidate_writes == result.core_writes == result.teacher_calls == 0
    with pytest.raises(Exception, match="stream"):
        V2GenericTrainer().validate_train_streams(plan, reversed(streams))


def test_generic_trainer_streams_records_through_temporary_validation_store(
        tmp_path: Path) -> None:
    """The stream lane can be replayed and does not retain a record tuple."""
    first, _, _ = build_authored_scale_pack(tmp_path, "P0", 7)
    registry = V2PackRegistry.from_manifest_paths(tmp_path, (first,))
    entry = registry.entries[0]
    plan = registry.train_plan("W-08", scale_key="P0")

    def records():
        return (item.to_dict() for item in V2StreamReader(
            tmp_path, entry).iter_records("teacher"))

    stream = V2TrainPackStream(entry.pack_key, records)
    result = V2GenericTrainer().validate_train_streams(plan, (stream,))
    replay = V2GenericTrainer().validate_train_streams(plan, (stream,))
    assert result == replay
    assert result.input_commitment
    assert result.candidate_writes == result.core_writes == result.teacher_calls == 0


def test_scale_target_stops_before_work_on_overrun_or_bool() -> None:
    """P0/P1 hard ceilings reject overflow and bool before scratch creation."""
    validate_scale_target("P0", 3_200)
    validate_scale_target("P1", 12_800)
    with pytest.raises(V2ScaleBaselineError, match="exceeds"):
        validate_scale_target("P0", 3_201)
    with pytest.raises(V2ScaleBaselineError):
        validate_scale_target("P1", True)
    with pytest.raises(V2ScaleBaselineError):
        validate_scale_target("P2", 1)


def test_repository_report_is_canonical_pass_and_immutable(tmp_path: Path) -> None:
    """The published P0/P1 evidence is strict, zero-state and non-overwritable."""
    report = read_ft00_05_report(REPORT)
    assert report.status == "PASS"
    assert report.slope.passed == 1
    assert tuple(item.target_records for item in report.points) == (3_200, 12_800)
    assert report.formal_training_runs == report.candidate_writes == 0
    assert report.core_writes == report.memory_writes == 0
    assert report.companion_writes == report.use_writes == report.teacher_calls == 0
    for point in report.points:
        assert point.fresh_digest == point.resume_digest
        assert point.query_rows <= 16
        assert point.resolved_rows <= 16
        assert point.rollback_clean == point.resource_stop_boundary == 1
        assert {item.phase_key for item in point.phases} >= {
            "pack_build", "raw_pack_scan", "typed_adaptation",
            "registry_trainer_intake", "intake_projection",
            "candidate_build_simulation", "evidence_apply",
            "checkpoint_merge_resume", "query_index_build", "query",
            "resolve", "generation", "rollback",
        }
    target = tmp_path / "report.json"
    write_ft00_05_report(report, target)
    assert read_ft00_05_report(target) == report
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(Exception):
        write_ft00_05_report(report, target)
