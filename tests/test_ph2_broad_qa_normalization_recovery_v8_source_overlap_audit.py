"""覆盖 recovery-v8 cross-family overlap/copy audit。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_source_overlap_audit import (
    publish_normalization_recovery_v8_source_overlap_audit,
    read_normalization_recovery_v8_source_overlap_audit,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
)


_FAMILIES = (
    "QBITTORRENT_PROJECT",
    "STELLARIUM_PROJECT",
    "KEEPASSXC_PROJECT",
)


def _manifest(family: str, *, shared_license: bool = False) -> dict[str, object]:
    """构造独立lineage、独立locale且可共享license的source-pack manifest。"""
    index = _FAMILIES.index(family) + 1
    license_sha = "a" * 64 if shared_license else str(index) * 64
    return {
        "files": [
            {
                "role": "V8_SOURCE_RAW_LICENSE_BLOB",
                "sha256": license_sha,
            },
            {
                "role": "V8_SOURCE_RAW_LOCALE_BLOB",
                "sha256": str(index + 3) * 64,
            },
        ],
        "raw_source": {
            "commit": str(index) * 40,
            "repository": f"https://github.com/example/{index}.git",
            "root_tree": str(index + 3) * 40,
        },
        "source_family": family,
        "source_family_vote_count": 1,
    }


def _pair(
        family: str,
        *,
        source: str,
        input_text: str,
        output_text: str,
        gettext: bool = False,
        ) -> dict[str, object]:
    """构造统一字段的Qt或gettext pair。"""
    surface_key = "msgstr" if gettext else "translation"
    return {
        "official_source_text": source,
        "source_family": family,
        "zh_hans": {surface_key: output_text},
        "zh_hans_structure_tokens": [],
        "zh_hant": {surface_key: input_text},
        "zh_hant_structure_tokens": [],
    }


def _state(*, copied: bool = False):
    """构造三家独立数据；可令KeePassXC成为qBittorrent完整复制。"""
    manifests = {
        "QBITTORRENT_PROJECT": _manifest(
            "QBITTORRENT_PROJECT", shared_license=True),
        "STELLARIUM_PROJECT": _manifest("STELLARIUM_PROJECT"),
        "KEEPASSXC_PROJECT": _manifest(
            "KEEPASSXC_PROJECT", shared_license=True),
    }
    qbit = _pair(
        "QBITTORRENT_PROJECT",
        source="Open",
        input_text="開啟",
        output_text="打开",
    )
    keepass = (
        {**qbit, "source_family": "KEEPASSXC_PROJECT"}
        if copied else _pair(
            "KEEPASSXC_PROJECT",
            source="Save",
            input_text="儲存",
            output_text="保存",
        ))
    pairs = {
        "QBITTORRENT_PROJECT": (qbit,),
        "STELLARIUM_PROJECT": (_pair(
            "STELLARIUM_PROJECT",
            source="Sky",
            input_text="天空",
            output_text="天空",
            gettext=True,
        ),),
        "KEEPASSXC_PROJECT": (keepass,),
    }
    return manifests, pairs


def _input_dirs(tmp_path: Path) -> tuple[Path, ...]:
    """创建publisher所需的七个synthetic输入目录。"""
    values = tuple(tmp_path / name for name in (
        "v2-roster", "v1-roster", "v1-content", "v2-content",
        "qbit", "stellarium", "keepassxc"))
    for value in values:
        value.mkdir()
    return values


def _publish(
        tmp_path: Path,
        inputs: tuple[Path, ...],
        target: Path,
        ) -> dict[str, object]:
    """调用publisher并保持测试参数集中。"""
    return publish_normalization_recovery_v8_source_overlap_audit(
        run_root=tmp_path,
        v2_roster_dir=inputs[0],
        v1_roster_dir=inputs[1],
        v1_content_audit_dir=inputs[2],
        v2_content_audit_dir=inputs[3],
        qbittorrent_source_pack_dir=inputs[4],
        stellarium_source_pack_dir=inputs[5],
        keepassxc_source_pack_dir=inputs[6],
        target_dir=target,
    )


def test_v8_source_overlap_round_trip_shared_license_and_tamper(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """共享license不失败，独立locale/lineage可PASS并拒绝同步篡改。"""
    import pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_source_overlap_audit as module

    inputs = _input_dirs(tmp_path)
    monkeypatch.setattr(module, "_require_k_root", lambda value: Path(value))
    monkeypatch.setattr(module, "_state", lambda **_kwargs: _state())
    target = tmp_path / "audit"
    published = _publish(tmp_path, inputs, target)
    manifest, outputs = read_normalization_recovery_v8_source_overlap_audit(
        target,
        v2_roster_dir=inputs[0],
        v1_roster_dir=inputs[1],
        v1_content_audit_dir=inputs[2],
        v2_content_audit_dir=inputs[3],
        qbittorrent_source_pack_dir=inputs[4],
        stellarium_source_pack_dir=inputs[5],
        keepassxc_source_pack_dir=inputs[6],
        expected_manifest_sha256=str(published["manifest_sha256"]),
    )
    assert manifest == published
    assert manifest["summary"]["hard_independence_failure_count"] == 0
    assert manifest["summary"]["license_blob_pairwise_overlap_count"] == 1
    assert manifest["summary"]["locale_blob_pairwise_overlap_count"] == 0
    assert len(outputs["source-overlap.jsonl"]) == 3
    with pytest.raises(BroadQaExternalDataError, match="input/target path"):
        _publish(tmp_path, inputs, target)

    path = target / "source-overlap-census.jsonl"
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
        read_normalization_recovery_v8_source_overlap_audit(
            target,
            v2_roster_dir=inputs[0],
            v1_roster_dir=inputs[1],
            v1_content_audit_dir=inputs[2],
            v2_content_audit_dir=inputs[3],
            qbittorrent_source_pack_dir=inputs[4],
            stellarium_source_pack_dir=inputs[5],
            keepassxc_source_pack_dir=inputs[6],
            expected_manifest_sha256=hashlib.sha256(encoded).hexdigest(),
        )


def test_v8_source_overlap_rejects_complete_smaller_family_copy(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """较小family完整semantic set被覆盖时发布REJECTED而非放宽门。"""
    import pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_source_overlap_audit as module

    inputs = _input_dirs(tmp_path)
    monkeypatch.setattr(module, "_require_k_root", lambda value: Path(value))
    monkeypatch.setattr(
        module, "_state", lambda **_kwargs: _state(copied=True))
    published = _publish(tmp_path, inputs, tmp_path / "rejected")
    assert published["summary"]["exact_full_subset_copy_pair_count"] == 1
    assert published["summary"]["hard_independence_failure_count"] == 1
    assert published["status"] == "SOURCE_INDEPENDENCE_OR_COPY_GATE_REJECTED"
