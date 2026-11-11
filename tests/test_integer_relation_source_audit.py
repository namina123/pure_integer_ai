from __future__ import annotations

from pathlib import Path

import pytest

from pure_integer_ai.experiments.integer_relation_source_audit import (
    IntegerRelationSourceAuditError,
    audit_integer_relation_source_pack,
)
from pure_integer_ai.experiments.ph2_dataset_contract import RECORD_OBSERVATION
from pure_integer_ai.experiments.ph2_dataset_io import (
    read_artifact_manifest,
    read_record_artifact,
)


ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / ("ph2_w06_dataset_artifacts/source_semantic_overlay_v1/packs/"
               "AUTHORED_CC0_V1--CC0-1.0--causes-v1")
SOURCE = ROOT / "data/ph2/authored_relation_causes_seed_v1.jsonl.sample"


def _observations():
    manifest = read_artifact_manifest(PACK / "manifest.json")
    return tuple(
        item for identity in manifest.files
        if identity.record_kind == RECORD_OBSERVATION
        for item in read_record_artifact(PACK, identity)
    )


def test_authored_pack_is_source_bound_but_not_four_way_qualified():
    report = audit_integer_relation_source_pack(PACK, SOURCE)

    assert report["status"] == "PASS_SOURCE_PACK_ONLY"
    assert report["source_count"] == 14
    assert report["teacher_count"] == 10
    assert report["evaluator_count"] == 4
    assert report["four_way_qualification"].startswith("UNKNOWN")
    assert report["free_dialogue_complete"] == 0


def test_source_checksum_change_is_rejected(tmp_path):
    source_dir = tmp_path / "data" / "ph2"
    source_dir.mkdir(parents=True)
    altered = source_dir / SOURCE.name
    raw = SOURCE.read_bytes()
    assert b"CAUSES" in raw
    altered.write_bytes(raw.replace(b"CAUSES", b"CAUSEX", 1))

    with pytest.raises(IntegerRelationSourceAuditError, match="SourceRef"):
        audit_integer_relation_source_pack(PACK, altered)


def test_legacy_held_out_cluster_cannot_be_relabelled_four_ways():
    observations = _observations()
    train = [item.stable_key.components for item in observations
             if item.split == "train"]
    held_out = [item.stable_key.components for item in observations
                if item.split == "held_out"]
    assert len(held_out) == 4
    assignments = {
        "train": train,
        "held_out": held_out[:1],
        "negative": held_out[1:2],
        "conflict": held_out[2:],
    }

    with pytest.raises(IntegerRelationSourceAuditError, match="source cluster crosses"):
        audit_integer_relation_source_pack(PACK, SOURCE,
                                           assignments=assignments)


def test_four_way_assignment_requires_exact_pack_coverage():
    observations = _observations()
    train = [item.stable_key.components for item in observations
             if item.split == "train"]
    held_out = [item.stable_key.components for item in observations
                if item.split == "held_out"]
    assignments = {
        "train": train,
        "held_out": held_out[:1],
        "negative": held_out[1:2],
        "conflict": held_out[2:3],
    }

    with pytest.raises(IntegerRelationSourceAuditError, match="cover the pack"):
        audit_integer_relation_source_pack(PACK, SOURCE,
                                           assignments=assignments)
