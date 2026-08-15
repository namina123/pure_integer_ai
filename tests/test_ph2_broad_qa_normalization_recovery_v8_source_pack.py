"""覆盖参数化 recovery-v8 source family pack。"""
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
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_source_family_records import (
    derive_normalization_recovery_v8_source_family_records,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_source_pack import (
    publish_normalization_recovery_v8_source_pack,
    read_normalization_recovery_v8_source_pack,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
)


def _ts(*, language: str, translation: str) -> bytes:
    """构造一份单message qBittorrent-style TS。"""
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<!DOCTYPE TS>\n'
        f'<TS version="2.1" language="{language}">'
        '<context><name>Main</name><message id="open">'
        '<source>Open %1</source><comment>verb</comment>'
        f'<translation>{translation}</translation>'
        '</message></context></TS>\n'
    ).encode("utf-8")


def _blob(relative_path: str, payload: bytes) -> dict[str, object]:
    """构造与synthetic source一致的blob commitment。"""
    return {
        "bytes": len(payload),
        "git_blob_sha1": git_blob_sha1(payload),
        "relative_path": relative_path,
    }


def _source_state(
        source_root: Path,
        ) -> tuple[dict[str, object], dict[str, object]]:
    """写入synthetic raw blobs并形成一致的roster/content records。"""
    license_payload = b"synthetic license\n"
    hans = _ts(language="zh", translation="打开 %1")
    hant = _ts(language="zh_TW", translation="開啟 %1")
    payloads = {
        "COPYING": license_payload,
        "src/lang/qbittorrent_zh_CN.ts": hans,
        "src/lang/qbittorrent_zh_TW.ts": hant,
    }
    for relative, payload in payloads.items():
        path = source_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    roster = {
        "commit": "1" * 40,
        "commit_date": "2026-08-15T00:00:00Z",
        "license": {
            "expression": "MIT",
            "files": [_blob("COPYING", license_payload)],
            "primary_bytes": len(license_payload),
            "primary_sha256": hashlib.sha256(license_payload).hexdigest(),
        },
        "locale_file_count": 2,
        "locale_files": [
            _blob("src/lang/qbittorrent_zh_CN.ts", hans),
            _blob("src/lang/qbittorrent_zh_TW.ts", hant),
        ],
        "parser_identity": "QT_TS_XML_V1",
        "repository": "https://github.com/example/qbittorrent.git",
        "root_tree": "2" * 40,
        "source_family": "QBITTORRENT_PROJECT",
        "source_policy_scope": "QBITTORRENT_ZH_TW_TO_ZH_CN_FIXED_COMMIT_V1",
    }
    _files, pairs, summary = (
        derive_normalization_recovery_v8_source_family_records(
            roster, payloads))
    content = {
        "content_outcome": "PASS_NONZERO_ACTIVE_COMMON_PAIR",
        "license_expression": "MIT",
        "license_file_count": 1,
        "locale_file_count": 2,
        "parser_summary": summary,
        "source_family": "QBITTORRENT_PROJECT",
        "source_policy_scope": roster["source_policy_scope"],
        "transient_pair_count": len(pairs),
    }
    return roster, content


def _input_dirs(tmp_path: Path) -> tuple[Path, ...]:
    """创建publisher所需的五个synthetic输入目录。"""
    values = tuple(tmp_path / name for name in (
        "v2-roster", "v1-roster", "v1-content", "v2-content", "source"))
    for value in values:
        value.mkdir()
    return values


def test_v8_source_pack_round_trip_self_contained_and_tamper(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """pack从内置raw重派生，拒绝覆盖及records/manifest同步篡改。"""
    import pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_source_pack as module

    inputs = _input_dirs(tmp_path)
    state = _source_state(inputs[4])
    monkeypatch.setattr(module, "_require_k_root", lambda value: Path(value))
    monkeypatch.setattr(module, "_state", lambda **_kwargs: state)
    target = tmp_path / "pack"
    published = publish_normalization_recovery_v8_source_pack(
        run_root=tmp_path,
        source_family="QBITTORRENT_PROJECT",
        v2_roster_dir=inputs[0],
        v1_roster_dir=inputs[1],
        v1_content_audit_dir=inputs[2],
        v2_content_audit_dir=inputs[3],
        source_root=inputs[4],
        target_dir=target,
    )
    with pytest.raises(BroadQaExternalDataError, match="input/target path"):
        publish_normalization_recovery_v8_source_pack(
            run_root=tmp_path,
            source_family="QBITTORRENT_PROJECT",
            v2_roster_dir=inputs[0],
            v1_roster_dir=inputs[1],
            v1_content_audit_dir=inputs[2],
            v2_content_audit_dir=inputs[3],
            source_root=inputs[4],
            target_dir=target,
        )

    (inputs[4] / "src/lang/qbittorrent_zh_CN.ts").write_bytes(b"changed")
    manifest, files, pairs, census = read_normalization_recovery_v8_source_pack(
        target,
        v2_roster_dir=inputs[0],
        v1_roster_dir=inputs[1],
        v1_content_audit_dir=inputs[2],
        v2_content_audit_dir=inputs[3],
        expected_manifest_sha256=str(published["manifest_sha256"]),
    )
    assert manifest == published
    assert len(files) == 2
    assert len(pairs) == 1
    assert census["pair_record_count"] == 1
    assert census["v8_training_eligible_pair_count"] == 1
    assert (target / "raw/src/lang/qbittorrent_zh_CN.ts").is_file()

    extra = target / "raw/extra.txt"
    extra.write_bytes(b"unselected")
    with pytest.raises(BroadQaExternalDataError, match="inventory 漂移"):
        read_normalization_recovery_v8_source_pack(
            target,
            v2_roster_dir=inputs[0],
            v1_roster_dir=inputs[1],
            v1_content_audit_dir=inputs[2],
            v2_content_audit_dir=inputs[3],
            expected_manifest_sha256=str(published["manifest_sha256"]),
        )
    extra.unlink()

    path = target / "translation-pairs.jsonl"
    values = [json.loads(line) for line in path.read_bytes().splitlines()]
    values[0]["v8_training_eligible"] = 0
    changed = b"".join(canonical_json_line(item) for item in values)
    path.write_bytes(changed)
    stored = json.loads((target / "manifest.json").read_bytes())
    artifact = next(item for item in stored["files"]
                    if item["relative_path"] == path.name)
    artifact["bytes"] = len(changed)
    artifact["sha256"] = hashlib.sha256(changed).hexdigest()
    encoded = canonical_json_line(stored)
    (target / "manifest.json").write_bytes(encoded)
    with pytest.raises(BroadQaExternalDataError, match="records 漂移"):
        read_normalization_recovery_v8_source_pack(
            target,
            v2_roster_dir=inputs[0],
            v1_roster_dir=inputs[1],
            v1_content_audit_dir=inputs[2],
            v2_content_audit_dir=inputs[3],
            expected_manifest_sha256=hashlib.sha256(encoded).hexdigest(),
        )


def test_v8_source_pack_rejects_content_aggregate_drift(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """content-v2统计与完整重派生不一致时fail closed。"""
    import pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_source_pack as module

    inputs = _input_dirs(tmp_path)
    roster, content = _source_state(inputs[4])
    changed = {
        **content,
        "transient_pair_count": int(content["transient_pair_count"]) + 1,
    }
    monkeypatch.setattr(module, "_require_k_root", lambda value: Path(value))
    monkeypatch.setattr(
        module, "_state", lambda **_kwargs: (roster, changed))
    with pytest.raises(BroadQaExternalDataError, match="aggregate 漂移"):
        publish_normalization_recovery_v8_source_pack(
            run_root=tmp_path,
            source_family="QBITTORRENT_PROJECT",
            v2_roster_dir=inputs[0],
            v1_roster_dir=inputs[1],
            v1_content_audit_dir=inputs[2],
            v2_content_audit_dir=inputs[3],
            source_root=inputs[4],
            target_dir=tmp_path / "rejected",
        )
