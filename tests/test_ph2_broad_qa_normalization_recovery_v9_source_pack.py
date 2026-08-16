"""覆盖 v9 GIMP 自包含 source pack。"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_localization_structure import (
    git_blob_sha1,
    sha256_hex,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v9_gettext_source_records import (
    derive_normalization_recovery_v9_gettext_source_records,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v9_source_pack import (
    materialize_normalization_recovery_v9_source_pairs_after_guard,
    publish_normalization_recovery_v9_source_pack,
    read_normalization_recovery_v9_source_pack,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes


def _po(language: str, msgid: str, translation: str) -> bytes:
    """构造一个含结构 token 的最小合法 PO。"""
    return (
        "# This file is distributed under the same license as the gimp package.\n"
        "msgid \"\"\n"
        "msgstr \"\"\n"
        '"Project-Id-Version: GIMP-test\\n"\n'
        f'"Language: {language}\\n"\n'
        '"Content-Type: text/plain; charset=UTF-8\\n"\n'
        "\n"
        f'msgid "{msgid}"\n'
        f'msgstr "{translation}"\n'
    ).encode("utf-8")


def _material(tmp_path: Path):
    """建立八 domain raw、roster与匹配content aggregate。"""
    license_root = tmp_path / "license"
    locale_root = tmp_path / "locale"
    license_root.mkdir()
    locale_root.mkdir()
    license_files = []
    for name, payload in (("COPYING", b"gpl\n"), ("LICENSE", b"license\n")):
        (license_root / name).write_bytes(payload)
        license_files.append({
            "bytes": len(payload),
            "git_blob_sha1": git_blob_sha1(payload),
            "relative_path": name,
            "sha256": sha256_hex(payload),
        })
    locale_files = []
    pair_specs = []
    payloads = {}
    for index in range(8):
        domain = f"po-{index}"
        values = {}
        for locale, role, translation in (
                ("zh_CN", "zh_Hans", f"打开 %s {index}"),
                ("zh_TW", "zh_Hant", f"開啟 %s {index}")):
            relative = f"{domain}/{locale}.po"
            payload = _po(locale, f"Open %s {index}", translation)
            path = locale_root / Path(relative)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            payloads[relative] = payload
            values[role] = {
                "expected_language": locale,
                "relative_path": relative,
            }
            locale_files.append({
                "bytes": len(payload),
                "domain": domain,
                "git_blob_sha1": git_blob_sha1(payload),
                "locale": locale,
                "relative_path": relative,
            })
        pair_specs.append({"domain": domain, **values})
    record = {
        "license": {
            "expression": "GPL-3.0-or-later",
            "files": license_files,
        },
        "locale_files": locale_files,
        "source_policy_scope": "TEST_SCOPE",
    }
    source_files, pairs, summary = (
        derive_normalization_recovery_v9_gettext_source_records(
            source_family="GIMP_PROJECT",
            source_policy_scope="TEST_SCOPE",
            license_expression="GPL-3.0-or-later",
            pair_specs=tuple(pair_specs),
            files=payloads,
        ))
    content = {
        "content_outcome": "PASS_NONZERO_ACTIVE_COMMON_PAIR",
        "locale_file_commitment_sha256": hashlib.sha256(
            canonical_json_bytes(source_files)).hexdigest(),
        "pair_identity_roster_sha256": hashlib.sha256(
            canonical_json_bytes([{
                "pair_id": item["pair_id"],
                "source_identity": item["source_identity"],
            } for item in pairs])).hexdigest(),
        "parser_summary": summary,
        "transient_pair_count": len(pairs),
    }
    return license_root, locale_root, record, content


def test_v9_source_pack_round_trip_is_label_free(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """pack self-contained strict reread且只发布identity与synthetic shape。"""
    import pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v9_source_pack as module

    license_root, locale_root, record, content = _material(tmp_path)
    roster_dir = tmp_path / "roster"
    content_dir = tmp_path / "content"
    roster_dir.mkdir()
    content_dir.mkdir()
    target = tmp_path / "pack"
    monkeypatch.setattr(module, "_require_k_root", lambda value: Path(value))
    monkeypatch.setattr(
        module,
        "read_normalization_recovery_v9_source_roster",
        lambda *args, **kwargs: ({}, {"source-roster.jsonl": (record,)}),
    )
    monkeypatch.setattr(
        module,
        "read_normalization_recovery_v9_source_content_aggregate",
        lambda *args, **kwargs: ({}, {"source-content.jsonl": (content,)}),
    )
    published = publish_normalization_recovery_v9_source_pack(
        run_root=tmp_path,
        roster_dir=roster_dir,
        content_audit_dir=content_dir,
        license_root=license_root,
        locale_root=locale_root,
        target_dir=target,
    )
    manifest, outputs = read_normalization_recovery_v9_source_pack(
        target,
        roster_dir=roster_dir,
        content_audit_dir=content_dir,
        expected_manifest_sha256=published["manifest_sha256"],
    )
    assert manifest == published
    assert len(outputs["source-files.jsonl"]) == 18
    assert len(outputs["pair-identities.jsonl"]) == 8
    assert len(outputs["runtime-shapes.jsonl"]) == 8
    assert all(item["synthetic_surface_only"] == 1
               for item in outputs["runtime-shapes.jsonl"])
    encoded = (target / "pair-identities.jsonl").read_text(encoding="utf-8")
    assert '"msgstr"' not in encoded
    assert '"zh_hans"' not in encoded
    assert '"zh_hant"' not in encoded
    with pytest.raises(BroadQaExternalDataError, match="guard后"):
        materialize_normalization_recovery_v9_source_pairs_after_guard(
            target,
            expected_manifest_sha256=published["manifest_sha256"],
            guard_consumed=0,
        )
    guarded_manifest, pairs, summary = (
        materialize_normalization_recovery_v9_source_pairs_after_guard(
            target,
            expected_manifest_sha256=published["manifest_sha256"],
            guard_consumed=1,
        ))
    assert guarded_manifest == manifest
    assert len(pairs) == 8
    assert summary["plain_pair_count"] == 8
