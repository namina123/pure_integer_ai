"""FT00-04 public source adapter, owner split, and audit checks."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_d03_v2_registry import (
    V2PackEntry,
    V2PackRegistry,
)
from pure_integer_ai.experiments.ph2_d03_v2_source_adapters import (
    V2SourceAdapterError,
    audit_d02_source_coverage,
    compile_v2_source_pack,
    read_v2_source_adapter_audit,
)
from pure_integer_ai.experiments.ph2_d03_v2_streaming import (
    V2StreamReader,
    V2StreamingError,
)


ROOT = Path(__file__).resolve().parents[1]
D02_ROOT = Path("ph2_dataset_artifacts/d02_source_pack_v1/packs")


def _copy_pack(tmp_path: Path, pack_name: str) -> str:
    relative = D02_ROOT / pack_name
    target = tmp_path / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT / relative, target)
    return (relative / "manifest.json").as_posix()


def test_repository_source_audit_preserves_snapshot_license_and_blockers() -> None:
    """All D-02 partitions are accounted for without promoting incomplete splits."""
    audit = audit_d02_source_coverage(ROOT)
    assert (audit.ready_pack_count, audit.source_only_count, audit.blocked_count) == (
        3, 3, 1)
    assert audit.formal_training_runs == audit.teacher_calls == 0
    rows = {(item.source_key, item.license_partition): item
            for item in audit.entries}
    assert rows[("CC_CEDICT_20260725", "UNRESOLVED/BLOCKED")].blocker_code == (
        "OFFICIAL_LICENSE_EVIDENCE_DIVERGENCE")
    for key in (
            ("CONCEPTNET_5_7_0", "CC-BY-4.0"),
            ("CONCEPTNET_5_7_0", "CC-BY-SA-4.0"),
            ("UD_ZH_GSDSIMP_R2_18", "CC-BY-SA-4.0")):
        assert rows[key].status == "SOURCE_ONLY"
        assert rows[key].limitation_code == "SOURCE_SPLIT_INCOMPLETE"
    for item in audit.entries:
        assert item.raw_snapshot_manifest_relative_path
        assert len(item.raw_snapshot_manifest_sha256) == 64
        assert "private" not in item.raw_snapshot_manifest_relative_path.casefold()


def test_published_source_audit_is_canonical_and_replayable() -> None:
    """The checked-in audit has an exact, immutable read path."""
    audit = read_v2_source_adapter_audit(ROOT)
    assert audit.sha256() == (
        "4e28fa2e09962ff5046f1ed92051900d007bf90c065d274637570f2a68a5ddbe")
    assert audit.to_dict()["artifact_kind"] == (
        "PH2_D03_V2_SOURCE_ADAPTER_AUDIT")


def test_ready_d02_pack_compiles_to_variable_v2_manifest_and_resumes(
        tmp_path: Path) -> None:
    """A multi-split public pack becomes a v2 manifest without fixed counts."""
    pack_name = "ZHWIKTIONARY_20260701--CC-BY-SA-4.0--source-pack-v1"
    d02_manifest = _copy_pack(tmp_path, pack_name)
    first = compile_v2_source_pack(
        tmp_path, d02_manifest, output_relative_root="v2/source_adapter")
    assert first.published is True
    assert first.audit.status == "READY"
    relative = first.audit.v2_manifest_relative_path
    entry = V2PackEntry.from_manifest(tmp_path, relative)
    assert entry.source_ref_count == 4
    assert entry.observation_counts == (("train", 2), ("held_out", 2))
    assert entry.teacher_evidence_count == 2
    assert entry.evaluator_label_counts == (("held_out", 2),)
    assert entry.total_record_count == 12
    registry = V2PackRegistry.from_manifest_paths(tmp_path, (relative,))
    assert registry.snapshot().total_record_count == 12
    assert registry.train_plan("W-03").total_input_count == 8

    second = compile_v2_source_pack(
        tmp_path, d02_manifest, output_relative_root="v2/source_adapter")
    assert second.published is False
    assert second.manifest.sha256() == first.manifest.sha256()


def test_v2_source_carrier_preserves_raw_and_owner_visibility(tmp_path: Path) -> None:
    """Raw source structure remains under document_container and labels stay hidden."""
    pack_name = "WIKIDATA_REVISION_V1--CC0-1.0--source-pack-v1"
    d02_manifest = _copy_pack(tmp_path, pack_name)
    build = compile_v2_source_pack(
        tmp_path, d02_manifest, output_relative_root="v2/source_adapter")
    entry = V2PackEntry.from_manifest(
        tmp_path, build.audit.v2_manifest_relative_path)
    reader = V2StreamReader(tmp_path, entry)
    candidate = tuple(reader.iter_records("candidate"))
    observations = [item for item in candidate
                    if getattr(item, "RECORD_KIND", "") == "observation"]
    assert observations
    payload = observations[0].typed_payload.to_value()
    assert payload["carrier"]["carrier_kind"] == "document_container"
    assert payload["language_payload"]["raw_observation"]
    assert payload["language_payload"]["source_adapter"][
        "d02_manifest_relative_path"] == d02_manifest
    assert all(getattr(item, "RECORD_KIND", "") != "evaluator_label"
               for item in candidate)
    with pytest.raises(V2StreamingError, match="private"):
        tuple(reader.iter_records("private_evaluator"))


def test_source_only_pack_is_not_relabelled_as_train(tmp_path: Path) -> None:
    """A held-out-only D-02 sample remains source-only and writes no v2 artifact."""
    pack_name = "UD_ZH_GSDSIMP_R2_18--CC-BY-SA-4.0--source-pack-v1"
    d02_manifest = _copy_pack(tmp_path, pack_name)
    with pytest.raises(V2SourceAdapterError, match="SOURCE_SPLIT_INCOMPLETE"):
        compile_v2_source_pack(
            tmp_path, d02_manifest, output_relative_root="v2/source_adapter")
    assert not (tmp_path / "v2/source_adapter").exists()
