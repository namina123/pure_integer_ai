"""覆盖 recovery-v7 neutral source projection 与严格回读边界。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_recovery_v7_neutral_source_projection_audit
    as audit,
)
from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_neutral_source_projection_records import (
    GODOT_SOURCE_FAMILY,
    LIBREOFFICE_SOURCE_FAMILY,
    NEUTRAL_SURFACE_UNAVAILABLE,
    THUNDERBIRD_SOURCE_FAMILY,
    VSCODE_SOURCE_FAMILY,
    derive_neutral_upstream_source_projection_records,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    canonical_json_line,
)


def _id(value: str) -> str:
    """形成稳定 synthetic SHA identity。"""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _manifest(family: str) -> dict[str, object]:
    """构造 records 层所需的最小 source manifest。"""
    return {
        "license": {
            "attribution": f"{family} contributors",
            "license_id": "MIT" if family in (
                GODOT_SOURCE_FAMILY, VSCODE_SOURCE_FAMILY) else "MPL-2.0",
        },
        "manifest_sha256": _id(f"manifest-{family}"),
        "source_family": family,
        "source_policy_scope": f"POLICY_{family}",
    }


def _godot_pair(surface: str, output: str) -> dict[str, object]:
    """构造一个 eligible Godot gettext pair。"""
    identity = {"msgctxt": "", "msgid": surface, "msgid_plural": ""}
    return {
        "equal_length": int(len(surface) == len(output)),
        "pair_id": _id(f"godot-{surface}"),
        "source_identity": identity,
        "training_eligible": 1,
        "zh_hans": {"msgstr": output, "structure_tokens": []},
        "zh_hant": {"msgstr": "開啟檔案", "structure_tokens": []},
    }


def _libreoffice_pair(surface: str, output: str) -> dict[str, object]:
    """构造一个 eligible LibreOffice gettext pair。"""
    identity = {"msgctxt": "CTX", "msgid": surface, "msgid_plural": ""}
    return {
        "equal_length": int(len(surface) == len(output)),
        "pair_id": _id(f"libreoffice-{surface}"),
        "source_identity": identity,
        "training_eligible": 1,
        "zh_hans": {"msgstr": output},
        "zh_hans_structure_tokens": [],
        "zh_hant_structure_tokens": [],
    }


def _vscode_pair(surface: str, output: str) -> dict[str, object]:
    """构造一个 eligible VS Code key-leaf pair。"""
    return {
        "equal_length": int(len(surface) == len(output)),
        "json_path": ["bundle", surface],
        "pair_id": _id(f"vscode-{surface}"),
        "training_eligible": 1,
        "translation_relative_path": "translations/a.json",
        "zh_hans_file_id": _id("hans-file"),
        "zh_hans_structure_tokens": [],
        "zh_hans_text": output,
        "zh_hant_file_id": _id("hant-file"),
        "zh_hant_structure_tokens": [],
    }


def _synthetic_material() -> tuple[
        tuple[dict[str, object], ...],
        tuple[dict[str, object], ...],
        tuple[dict[str, object], ...],
        dict[str, object],
        ]:
    """派生一个三来源 exact consensus 和 Thunderbird unavailable。"""
    surface = "Open file"
    output = "打开文件"
    return derive_neutral_upstream_source_projection_records(
        godot_manifest=_manifest(GODOT_SOURCE_FAMILY),
        godot_pairs=(_godot_pair(surface, output),),
        libreoffice_manifest=_manifest(LIBREOFFICE_SOURCE_FAMILY),
        libreoffice_pairs=(_libreoffice_pair(surface, output),),
        vscode_manifest=_manifest(VSCODE_SOURCE_FAMILY),
        vscode_pairs=(_vscode_pair(surface, output),),
        thunderbird_manifest=_manifest(THUNDERBIRD_SOURCE_FAMILY),
        thunderbird_pairs=({"pair_id": _id("thunderbird")},),
    )


def test_projection_distinguishes_adapters_and_publishes_no_surface() -> None:
    """三类 adapter 保持分账，artifact 只保留承诺而不泄露原文。"""
    families, projections, support, summary = _synthetic_material()
    assert len(families) == 4
    assert len(projections) == 3
    assert len(support) == 1
    assert support[0]["support_family_count"] == 3
    assert support[0]["output_consensus"] == 1
    assert support[0]["consensus_output_sha256"] == _id("打开文件")
    thunderbird = next(
        item for item in families
        if item["source_family"] == THUNDERBIRD_SOURCE_FAMILY)
    assert thunderbird["adapter_projection_kind"] == (
        NEUTRAL_SURFACE_UNAVAILABLE)
    assert thunderbird["projected_record_count"] == 0
    encoded = canonical_json_bytes({
        "families": families,
        "projections": projections,
        "support": support,
        "summary": summary,
    })
    assert b"Open file" not in encoded
    assert "打开文件".encode("utf-8") not in encoded


def test_projection_rejects_non_integer_binary_schema() -> None:
    """records 层拒绝 bool 冒充冻结 JSON 二值。"""
    pair = _godot_pair("Open file", "打开文件")
    pair["training_eligible"] = True
    with pytest.raises(BroadQaExternalDataError, match="非二值"):
        derive_neutral_upstream_source_projection_records(
            godot_manifest=_manifest(GODOT_SOURCE_FAMILY),
            godot_pairs=(pair,),
            libreoffice_manifest=_manifest(LIBREOFFICE_SOURCE_FAMILY),
            libreoffice_pairs=(),
            vscode_manifest=_manifest(VSCODE_SOURCE_FAMILY),
            vscode_pairs=(),
            thunderbird_manifest=_manifest(THUNDERBIRD_SOURCE_FAMILY),
            thunderbird_pairs=(),
        )


def _fake_outputs() -> tuple[
        dict[str, tuple[dict[str, object], ...]], dict[str, object]]:
    """构造 audit publisher/reader 的小型稳定输出。"""
    outputs = {
        "family-projections.jsonl": ({
            "record_kind": "FAMILY", "source_family": "A"},),
        "neutral-projections.jsonl": ({
            "projection_id": _id("projection"), "record_kind": "PROJECTION"},),
        "cross-family-support.jsonl": ({
            "record_kind": "SUPPORT", "support_id": _id("support")},),
    }
    return outputs, {
        "capability_claimed": 0,
        "projection_outcome": "PASS",
    }


def _patch_audit_inputs(
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """隔离 synthetic publisher test 与真实 K 盘 source pack。"""
    monkeypatch.setattr(audit, "_require_k_root", lambda value: Path(value))
    monkeypatch.setattr(
        audit,
        "_input_state",
        lambda **_kwargs: ({}, {}, {}),
    )
    monkeypatch.setattr(audit, "_derive", lambda _sources: _fake_outputs())


def _input_dirs(tmp_path: Path) -> list[Path]:
    """创建 publisher 所需的六个 synthetic input 目录。"""
    paths = [tmp_path / name for name in (
        "protocol", "replay", "godot", "libreoffice", "vscode",
        "thunderbird")]
    for path in paths:
        path.mkdir()
    return paths


def _publish(
        tmp_path: Path,
        inputs: list[Path],
        ) -> tuple[Path, dict[str, object]]:
    """发布小型 synthetic projection artifact。"""
    target = tmp_path / "projection"
    published = (
        audit.publish_normalization_recovery_v7_neutral_source_projection_audit(
            run_root=tmp_path,
            training_protocol_dir=inputs[0],
            source_replay_audit_dir=inputs[1],
            godot_source_pack_dir=inputs[2],
            libreoffice_source_pack_dir=inputs[3],
            vscode_source_pack_dir=inputs[4],
            thunderbird_source_pack_dir=inputs[5],
            target_dir=target,
        ))
    return target, published


def _read(
        target: Path,
        inputs: list[Path],
        manifest_sha256: str,
        ) -> tuple[dict[str, object], dict[str, tuple[dict[str, object], ...]]]:
    """严格回读 synthetic projection artifact。"""
    return audit.read_normalization_recovery_v7_neutral_source_projection_audit(
        target,
        training_protocol_dir=inputs[0],
        source_replay_audit_dir=inputs[1],
        godot_source_pack_dir=inputs[2],
        libreoffice_source_pack_dir=inputs[3],
        vscode_source_pack_dir=inputs[4],
        thunderbird_source_pack_dir=inputs[5],
        expected_manifest_sha256=manifest_sha256,
    )


def test_projection_audit_round_trip_and_nonoverwrite(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """publisher/reader 往返一致，并拒绝覆盖已存在 artifact。"""
    _patch_audit_inputs(monkeypatch)
    inputs = _input_dirs(tmp_path)
    target, published = _publish(tmp_path, inputs)
    manifest, outputs = _read(
        target, inputs, str(published["manifest_sha256"]))
    assert manifest == published
    assert manifest["status"] == (
        audit.NORMALIZATION_RECOVERY_V7_NEUTRAL_SOURCE_PROJECTION_STATUS)
    assert outputs == _fake_outputs()[0]
    assert manifest["train_surface_published_in_audit"] == 0
    with pytest.raises(BroadQaExternalDataError, match="input/target path 非法"):
        _publish(tmp_path, inputs)


def test_projection_audit_rejects_record_and_manifest_tamper(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """record 或 manifest 字段被改写时 strict reader 均封闭失败。"""
    _patch_audit_inputs(monkeypatch)
    inputs = _input_dirs(tmp_path)
    target, published = _publish(tmp_path, inputs)
    projection_path = target / "neutral-projections.jsonl"
    projection_path.write_bytes(canonical_json_line({
        "projection_id": _id("tampered"),
        "record_kind": "PROJECTION",
    }))
    with pytest.raises(BroadQaExternalDataError, match="records/inputs 漂移"):
        _read(target, inputs, str(published["manifest_sha256"]))

    projection_path.write_bytes(canonical_json_line(
        _fake_outputs()[0]["neutral-projections.jsonl"][0]))
    manifest_path = target / "manifest.json"
    stored = json.loads(manifest_path.read_bytes())
    stored["production_enabled"] = 1
    encoded = canonical_json_line(stored)
    manifest_path.write_bytes(encoded)
    with pytest.raises(BroadQaExternalDataError, match="manifest 字段漂移"):
        _read(target, inputs, hashlib.sha256(encoded).hexdigest())
