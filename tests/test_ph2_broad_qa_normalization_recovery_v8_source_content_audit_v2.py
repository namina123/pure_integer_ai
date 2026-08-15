"""覆盖 recovery-v8 roster-v2 aggregate content feasibility。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_localization_structure import (
    git_blob_sha1,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_source_content_audit_v2 import (
    publish_normalization_recovery_v8_source_content_audit_v2,
    read_normalization_recovery_v8_source_content_audit_v2,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
)


def _ts(*, language: str, translation: str, unfinished: bool = False) -> bytes:
    """构造一份单message KeePassXC-style TS。"""
    type_text = ' type="unfinished"' if unfinished else ""
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<!DOCTYPE TS>\n'
        f'<TS version="2.1" language="{language}">'
        '<context><name>Main</name><message id="open">'
        '<source>Open %1</source>'
        f'<translation{type_text}>{translation}</translation>'
        '</message></context></TS>\n'
    ).encode("utf-8")


def _blob(relative_path: str, payload: bytes) -> dict[str, object]:
    """构造与synthetic payload一致的blob commitment。"""
    return {
        "bytes": len(payload),
        "git_blob_sha1": git_blob_sha1(payload),
        "relative_path": relative_path,
    }


def _fixture_state(
        root: Path,
        *,
        unfinished: bool = False,
        ) -> tuple[
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            Path,
        ]:
    """构造两家sealed PASS与一家KeePassXC replacement。"""
    license_payload = b"synthetic license\n"
    hans = _ts(
        language="zh_CN", translation="打开 %1", unfinished=unfinished)
    hant = _ts(
        language="zh_TW", translation="開啟 %1", unfinished=unfinished)
    payloads = {
        "COPYING": license_payload,
        "share/translations/keepassxc_zh_CN.ts": hans,
        "share/translations/keepassxc_zh_TW.ts": hant,
    }
    for relative, payload in payloads.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    keepassxc = {
        "content_feasibility_outcome": "NOT_READ_ROSTER_V2_REPLACEMENT",
        "license": {
            "expression": "GPL-2.0-only OR GPL-3.0-only",
            "files": [_blob("COPYING", license_payload)],
            "primary_bytes": len(license_payload),
            "primary_sha256": hashlib.sha256(license_payload).hexdigest(),
        },
        "locale_blob_content_read_count": 0,
        "locale_file_count": 2,
        "locale_files": [
            _blob("share/translations/keepassxc_zh_CN.ts", hans),
            _blob("share/translations/keepassxc_zh_TW.ts", hant),
        ],
        "selection_status": (
            "SELECTED_V2_REPLACEMENT_TREE_LICENSE_PATH_FROZEN"),
        "source_family": "KEEPASSXC_PROJECT",
        "source_policy_scope": (
            "KEEPASSXC_ZH_TW_TO_ZH_CN_FIXED_COMMIT_V1"),
    }
    roster = [keepassxc]
    predecessor = [{
        "content_outcome": "REJECTED_ZERO_ACTIVE_COMMON_PAIR",
        "source_family": "BITCOIN_CORE_PROJECT",
        "transient_pair_count": 0,
    }]
    for family, count in (
            ("QBITTORRENT_PROJECT", 2),
            ("STELLARIUM_PROJECT", 3)):
        policy = family + "_SCOPE"
        roster.append({
            "content_feasibility_outcome": (
                "PASS_NONZERO_ACTIVE_COMMON_PAIR"),
            "license": {
                "expression": "MIT",
                "files": [_blob("COPYING", license_payload)],
            },
            "locale_file_count": 2,
            "selection_status": "INHERITED_V1_CONTENT_PASS",
            "source_family": family,
            "source_policy_scope": policy,
        })
        predecessor.append({
            "content_outcome": "PASS_NONZERO_ACTIVE_COMMON_PAIR",
            "license_expression": "MIT",
            "license_file_read_count": 1,
            "locale_file_read_count": 2,
            "parser_summary": {
                "content_outcome": "PASS_NONZERO_ACTIVE_COMMON_PAIR",
                "structure_equal_count": count,
                "v8_training_eligible_pair_count": count,
            },
            "source_family": family,
            "source_policy_scope": policy,
            "transient_pair_count": count,
        })
    return tuple(roster), tuple(predecessor), root


def _input_dirs(tmp_path: Path) -> tuple[Path, ...]:
    """创建publisher所需的四个synthetic输入目录。"""
    values = tuple(tmp_path / name for name in (
        "v2-roster", "v1-roster", "v1-content", "keepassxc"))
    for value in values:
        value.mkdir()
    return values


def test_v8_source_content_v2_round_trip_nonoverwrite_and_tamper(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """两家继承加replacement PASS可发布，并拒绝覆盖与同步篡改。"""
    import pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_source_content_audit_v2 as module

    inputs = _input_dirs(tmp_path)
    state = _fixture_state(inputs[3])
    monkeypatch.setattr(module, "_require_k_root", lambda value: Path(value))
    monkeypatch.setattr(module, "_state", lambda **_kwargs: state)
    target = tmp_path / "audit-v2"
    published = publish_normalization_recovery_v8_source_content_audit_v2(
        run_root=tmp_path,
        v2_roster_dir=inputs[0],
        v1_roster_dir=inputs[1],
        v1_content_audit_dir=inputs[2],
        keepassxc_source_root=inputs[3],
        target_dir=target,
    )
    manifest, outputs = read_normalization_recovery_v8_source_content_audit_v2(
        target,
        v2_roster_dir=inputs[0],
        v1_roster_dir=inputs[1],
        v1_content_audit_dir=inputs[2],
        keepassxc_source_root=inputs[3],
        expected_manifest_sha256=str(published["manifest_sha256"]),
    )
    assert manifest == published
    assert manifest["summary"]["content_pass_count"] == 3
    assert manifest["summary"]["replacement_blob_read_count"] == 3
    assert manifest["summary"]["transient_pair_count"] == 6
    by_family = {item["source_family"]: item
                 for item in outputs["source-content-v2.jsonl"]}
    assert by_family["KEEPASSXC_PROJECT"]["transient_pair_count"] == 1
    assert by_family["QBITTORRENT_PROJECT"][
        "content_blob_read_this_revision_count"] == 0
    with pytest.raises(BroadQaExternalDataError, match="input/target path"):
        publish_normalization_recovery_v8_source_content_audit_v2(
            run_root=tmp_path,
            v2_roster_dir=inputs[0],
            v1_roster_dir=inputs[1],
            v1_content_audit_dir=inputs[2],
            keepassxc_source_root=inputs[3],
            target_dir=target,
        )

    path = target / "source-content-census-v2.jsonl"
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
        read_normalization_recovery_v8_source_content_audit_v2(
            target,
            v2_roster_dir=inputs[0],
            v1_roster_dir=inputs[1],
            v1_content_audit_dir=inputs[2],
            keepassxc_source_root=inputs[3],
            expected_manifest_sha256=hashlib.sha256(encoded).hexdigest(),
        )


def test_v8_source_content_v2_preserves_zero_active_rejection(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """replacement全unfinished时保留REJECTED，不偷纳入非active表面。"""
    import pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_source_content_audit_v2 as module

    inputs = _input_dirs(tmp_path)
    state = _fixture_state(inputs[3], unfinished=True)
    monkeypatch.setattr(module, "_require_k_root", lambda value: Path(value))
    monkeypatch.setattr(module, "_state", lambda **_kwargs: state)
    published = publish_normalization_recovery_v8_source_content_audit_v2(
        run_root=tmp_path,
        v2_roster_dir=inputs[0],
        v1_roster_dir=inputs[1],
        v1_content_audit_dir=inputs[2],
        keepassxc_source_root=inputs[3],
        target_dir=tmp_path / "rejected",
    )
    assert published["summary"]["content_pass_count"] == 2
    assert published["summary"]["content_rejected_count"] == 1
    assert published["summary"]["transient_pair_count"] == 5
    assert published["status"] == (
        "CONTENT_FEASIBILITY_V2_REPLACEMENT_REJECTED_NOT_SOURCE_PACK")
