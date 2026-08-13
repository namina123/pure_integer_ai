"""normalization 对比训练来源协议的资格、来源和诚实边界测试。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_contrastive_protocol import (
    NORMALIZATION_CONTRASTIVE_APPLICATION_DOMAIN,
    NORMALIZATION_CONTRASTIVE_PROTOCOL_KIND,
    NORMALIZATION_CONTRASTIVE_STATUS,
    derive_normalization_contrastive_records,
    publish_normalization_contrastive_protocol,
    read_normalization_contrastive_protocol,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_source_pack import (
    publish_normalization_source_pack,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


def test_contrastive_derivation_requires_explicit_phrase_observation() -> None:
    """同一候选只有短语目标显式相同/不同时才形成 SUPPORT/REFUTE。"""
    candidates, trials, summary = derive_normalization_contrastive_records(
        source_pack_manifest_sha256="a" * 64,
        character_payload="乾\t干\n甲\t乙\n".encode("utf-8"),
        phrase_payload="乾乾\t干乾\n甲甲\t乙甲\n".encode("utf-8"),
    )
    assert len(candidates) == 2
    assert len(trials) == 4
    assert summary == {
        "candidate_count": 2,
        "candidate_with_both_qualifications_count": 2,
        "candidate_with_refute_count": 2,
        "candidate_with_support_count": 2,
        "source_replay_refute_count": 2,
        "source_replay_support_count": 2,
        "trial_count": 4,
    }
    assert {item["qualification_kind"] for item in trials} == {
        "SOURCE_REPLAY_REFUTE", "SOURCE_REPLAY_SUPPORT"}
    assert all(item["split"] == "TRAIN_SOURCE" for item in trials)
    assert all(item["semantic_non_equivalence_label_written"] == 0
               for item in trials)
    assert all(item["defeater_written"] == 0 for item in trials)
    assert all(item["accepted_rule_written"] == 0 for item in candidates)


def test_contrastive_protocol_round_trip_has_no_evaluation_or_learning(
        tmp_path: Path,
        ) -> None:
    """正式来源可重派生回读，但不宣称 evaluation、rule 或 learner read。"""
    source_pack = tmp_path / "normalization-source-pack"
    publish_normalization_source_pack(
        run_root=tmp_path,
        target_dir=source_pack,
    )
    target = tmp_path / "normalization-contrastive-protocol"
    report = publish_normalization_contrastive_protocol(
        run_root=tmp_path,
        source_pack_dir=source_pack,
        target_dir=target,
    )
    manifest, candidates, trials = read_normalization_contrastive_protocol(
        target,
        source_pack_dir=source_pack,
    )
    assert report["manifest_sha256"] == manifest["manifest_sha256"]
    assert manifest["artifact_kind"] == NORMALIZATION_CONTRASTIVE_PROTOCOL_KIND
    assert manifest["status"] == NORMALIZATION_CONTRASTIVE_STATUS
    assert manifest["application_domain"] == (
        NORMALIZATION_CONTRASTIVE_APPLICATION_DOMAIN)
    assert manifest["evaluation_record_count"] == 0
    assert manifest["validation_record_count"] == 0
    assert manifest["reserve_record_count"] == 0
    assert manifest["learner_read_count"] == 0
    assert manifest["accepted_rules_written"] == 0
    assert manifest["rejection_records_written"] == 0
    assert manifest["defeaters_written"] == 0
    assert len(candidates) == manifest["summary"]["candidate_count"]
    assert len(trials) == manifest["summary"]["trial_count"]
    assert manifest["summary"]["candidate_with_both_qualifications_count"] == 3
    assert manifest["summary"]["source_replay_support_count"] == 146
    assert manifest["summary"]["source_replay_refute_count"] == 136
    with pytest.raises(BroadQaExternalDataError, match="target 已存在"):
        publish_normalization_contrastive_protocol(
            run_root=tmp_path,
            source_pack_dir=source_pack,
            target_dir=target,
        )


def test_contrastive_reader_rejects_record_and_zero_boundary_tamper(
        tmp_path: Path,
        ) -> None:
    """trial 字节或未学习边界被改写时，来源重放必须拒绝。"""
    source_pack = tmp_path / "normalization-source-pack"
    publish_normalization_source_pack(
        run_root=tmp_path,
        target_dir=source_pack,
    )
    target = tmp_path / "normalization-contrastive-protocol"
    publish_normalization_contrastive_protocol(
        run_root=tmp_path,
        source_pack_dir=source_pack,
        target_dir=target,
    )
    trials_path = target / "context-trials.jsonl"
    lines = trials_path.read_bytes().splitlines(keepends=True)
    trial = json.loads(lines[0])
    trial["qualification_kind"] = (
        "SOURCE_REPLAY_REFUTE"
        if trial["qualification_kind"] == "SOURCE_REPLAY_SUPPORT"
        else "SOURCE_REPLAY_SUPPORT")
    lines[0] = canonical_json_line(trial)
    trials_path.write_bytes(b"".join(lines))
    manifest_path = target / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    import hashlib
    payload = trials_path.read_bytes()
    manifest["trial_records_bytes"] = len(payload)
    manifest["trial_records_sha256"] = hashlib.sha256(payload).hexdigest()
    manifest_path.write_bytes(canonical_json_line(manifest))
    with pytest.raises(BroadQaExternalDataError, match="records/source 漂移"):
        read_normalization_contrastive_protocol(
            target,
            source_pack_dir=source_pack,
        )

    target_two = tmp_path / "normalization-contrastive-protocol-two"
    publish_normalization_contrastive_protocol(
        run_root=tmp_path,
        source_pack_dir=source_pack,
        target_dir=target_two,
    )
    manifest_path = target_two / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["evaluation_record_count"] = 1
    manifest_path.write_bytes(canonical_json_line(manifest))
    with pytest.raises(BroadQaExternalDataError, match="manifest 漂移"):
        read_normalization_contrastive_protocol(
            target_two,
            source_pack_dir=source_pack,
        )


def test_contrastive_derivation_rejects_non_position_preserving_phrase(
        ) -> None:
    """v1 不得把增删码点的短语映射猜成字符位置 trial。"""
    with pytest.raises(BroadQaExternalDataError, match="位置保持"):
        derive_normalization_contrastive_records(
            source_pack_manifest_sha256="a" * 64,
            character_payload="甲\t乙\n".encode("utf-8"),
            phrase_payload="甲甲\t乙\n".encode("utf-8"),
        )
