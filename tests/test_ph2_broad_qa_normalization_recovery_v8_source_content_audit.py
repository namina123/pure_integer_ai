"""覆盖 recovery-v8 aggregate source content feasibility artifact。"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_source_content_audit import (
    publish_normalization_recovery_v8_source_content_audit,
    read_normalization_recovery_v8_source_content_audit,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
)


def _fake_outputs():
    """构造不含surface的最小aggregate outputs。"""
    records = tuple({
        "content_outcome": outcome,
        "format_version": 1,
        "pair_surface_published": 0,
        "parser_summary": {
            "plain_pair_count": count,
            "structure_equal_count": count,
            "v8_training_eligible_pair_count": count,
        },
        "record_kind": "CONTENT",
        "source_family": family,
        "transient_pair_count": count,
    } for family, outcome, count in (
        ("BITCOIN_CORE_PROJECT", "REJECTED_ZERO_ACTIVE_COMMON_PAIR", 0),
        ("QBITTORRENT_PROJECT", "PASS_NONZERO_ACTIVE_COMMON_PAIR", 2),
        ("STELLARIUM_PROJECT", "PASS_NONZERO_ACTIVE_COMMON_PAIR", 3),
    ))
    summary = {
        "content_pass_count": 2,
        "content_rejected_count": 1,
        "license_file_read_count": 5,
        "locale_file_read_count": 26,
        "pair_surface_published": 0,
        "selected_source_family_count": 3,
        "source_pack_published_count": 0,
        "structure_equal_count": 5,
        "transient_pair_count": 5,
        "v8_training_eligible_pair_count": 5,
    }
    census = ({
        **summary,
        "format_version": 1,
        "record_kind": "CENSUS",
    },)
    return {
        "source-content.jsonl": records,
        "source-content-census.jsonl": census,
    }, summary


def test_v8_source_content_round_trip_nonoverwrite_and_tamper(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """publisher/reader逐字段一致，并拒绝覆盖和同步篡改。"""
    import pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_source_content_audit as module

    monkeypatch.setattr(module, "_require_k_root", lambda value: Path(value))
    monkeypatch.setattr(
        module, "_state", lambda **_kwargs: ((), {}))
    monkeypatch.setattr(module, "_derive", lambda *_args: _fake_outputs())
    inputs = [tmp_path / name for name in (
        "roster", "bitcoin", "qbittorrent", "stellarium")]
    for value in inputs:
        value.mkdir()
    target = tmp_path / "audit"
    published = publish_normalization_recovery_v8_source_content_audit(
        run_root=tmp_path,
        roster_dir=inputs[0],
        bitcoin_source_root=inputs[1],
        qbittorrent_source_root=inputs[2],
        stellarium_source_root=inputs[3],
        target_dir=target,
    )
    manifest, outputs = read_normalization_recovery_v8_source_content_audit(
        target,
        roster_dir=inputs[0],
        bitcoin_source_root=inputs[1],
        qbittorrent_source_root=inputs[2],
        stellarium_source_root=inputs[3],
        expected_manifest_sha256=str(published["manifest_sha256"]),
    )
    assert manifest == published
    assert outputs == _fake_outputs()[0]
    assert manifest["summary"]["content_pass_count"] == 2
    assert manifest["summary"]["content_rejected_count"] == 1
    with pytest.raises(BroadQaExternalDataError, match="input/target path"):
        publish_normalization_recovery_v8_source_content_audit(
            run_root=tmp_path,
            roster_dir=inputs[0],
            bitcoin_source_root=inputs[1],
            qbittorrent_source_root=inputs[2],
            stellarium_source_root=inputs[3],
            target_dir=target,
        )

    path = target / "source-content-census.jsonl"
    changed = canonical_json_line({
        "format_version": 1,
        "record_kind": "CHANGED",
    })
    path.write_bytes(changed)
    stored = json.loads((target / "manifest.json").read_bytes())
    artifact = next(item for item in stored["files"]
                    if item["relative_path"] == path.name)
    artifact["bytes"] = len(changed)
    artifact["sha256"] = hashlib.sha256(changed).hexdigest()
    encoded = canonical_json_line(stored)
    (target / "manifest.json").write_bytes(encoded)
    with pytest.raises(BroadQaExternalDataError, match="records 漂移"):
        read_normalization_recovery_v8_source_content_audit(
            target,
            roster_dir=inputs[0],
            bitcoin_source_root=inputs[1],
            qbittorrent_source_root=inputs[2],
            stellarium_source_root=inputs[3],
            expected_manifest_sha256=hashlib.sha256(encoded).hexdigest(),
        )


def test_v8_source_content_official_round_trip(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """提供Git外fixture时发布并重派生正式三家aggregate。"""
    values = {
        name: os.environ.get(name) for name in (
            "PURE_INTEGER_AI_V8_SOURCE_ROSTER_ROOT",
            "PURE_INTEGER_AI_V8_BITCOIN_PROJECT_ROOT",
            "PURE_INTEGER_AI_V8_QBITTORRENT_PROJECT_ROOT",
            "PURE_INTEGER_AI_V8_STELLARIUM_PROJECT_ROOT",
        )}
    if any(not value for value in values.values()):
        pytest.skip("official v8 source content fixtures are unavailable")
    import pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_source_content_audit as module

    monkeypatch.setattr(module, "_require_k_root", lambda _value: tmp_path)
    monkeypatch.setattr(
        module, "_within",
        lambda _root, value, label: Path(value).resolve())
    target = tmp_path / "audit"
    published = publish_normalization_recovery_v8_source_content_audit(
        run_root=tmp_path,
        roster_dir=values["PURE_INTEGER_AI_V8_SOURCE_ROSTER_ROOT"],
        bitcoin_source_root=values[
            "PURE_INTEGER_AI_V8_BITCOIN_PROJECT_ROOT"],
        qbittorrent_source_root=values[
            "PURE_INTEGER_AI_V8_QBITTORRENT_PROJECT_ROOT"],
        stellarium_source_root=values[
            "PURE_INTEGER_AI_V8_STELLARIUM_PROJECT_ROOT"],
        target_dir=target,
    )
    manifest, outputs = read_normalization_recovery_v8_source_content_audit(
        target,
        roster_dir=values["PURE_INTEGER_AI_V8_SOURCE_ROSTER_ROOT"],
        bitcoin_source_root=values[
            "PURE_INTEGER_AI_V8_BITCOIN_PROJECT_ROOT"],
        qbittorrent_source_root=values[
            "PURE_INTEGER_AI_V8_QBITTORRENT_PROJECT_ROOT"],
        stellarium_source_root=values[
            "PURE_INTEGER_AI_V8_STELLARIUM_PROJECT_ROOT"],
        expected_manifest_sha256=str(published["manifest_sha256"]),
    )
    assert manifest["summary"] == {
        "content_pass_count": 2,
        "content_rejected_count": 1,
        "license_file_read_count": 5,
        "locale_file_read_count": 26,
        "pair_surface_published": 0,
        "selected_source_family_count": 3,
        "source_pack_published_count": 0,
        "structure_equal_count": 30_704,
        "transient_pair_count": 30_767,
        "v8_training_eligible_pair_count": 27_339,
    }
    by_family = {item["source_family"]: item
                 for item in outputs["source-content.jsonl"]}
    assert by_family["BITCOIN_CORE_PROJECT"]["content_outcome"] == (
        "REJECTED_ZERO_ACTIVE_COMMON_PAIR")
    assert by_family["QBITTORRENT_PROJECT"]["transient_pair_count"] == 2_436
    assert by_family["STELLARIUM_PROJECT"]["transient_pair_count"] == 28_331
