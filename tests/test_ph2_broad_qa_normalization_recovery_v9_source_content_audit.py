"""覆盖 v9 GIMP aggregate-only content audit。"""
from __future__ import annotations

from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_localization_structure import (
    git_blob_sha1,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v9_source_content_audit import (
    publish_normalization_recovery_v9_source_content_audit,
    read_normalization_recovery_v9_source_content_aggregate,
    read_normalization_recovery_v9_source_content_audit,
)


def _po(language: str, msgid: str, translation: str) -> bytes:
    """构造一个合法 active gettext 文件。"""
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


def _fixture(root: Path) -> dict[str, object]:
    """建立八 domain、十六文件的 synthetic roster/source。"""
    locale_files = []
    for index in range(8):
        domain = f"po-{index}"
        for locale, language, translation in (
                ("zh_CN", "zh_CN", f"打开{index}"),
                ("zh_TW", "zh_TW", f"開啟{index}")):
            relative = f"{domain}/{locale}.po"
            payload = _po(language, f"Open {index}", translation)
            path = root / Path(relative)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            locale_files.append({
                "bytes": len(payload),
                "domain": domain,
                "git_blob_sha1": git_blob_sha1(payload),
                "locale": locale,
                "relative_path": relative,
            })
    return {
        "label_or_translation_read_count": 0,
        "license": {"expression": "GPL-3.0-or-later"},
        "locale_blob_content_read_count": 0,
        "locale_files": locale_files,
        "locale_pair_count": 8,
        "source_family": "GIMP_PROJECT",
        "source_policy_scope": "TEST_SCOPE",
    }


def test_v9_source_content_round_trip_and_aggregate_reader(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """正式输出只含 aggregate，strict reader 可重派生同一分母。"""
    import pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v9_source_content_audit as module

    source = tmp_path / "source"
    source.mkdir()
    record = _fixture(source)
    roster = tmp_path / "roster"
    roster.mkdir()
    target = tmp_path / "audit"
    monkeypatch.setattr(module, "_require_k_root", lambda value: Path(value))
    monkeypatch.setattr(
        module,
        "read_normalization_recovery_v9_source_roster",
        lambda *args, **kwargs: ({}, {"source-roster.jsonl": (record,)}),
    )
    published = publish_normalization_recovery_v9_source_content_audit(
        run_root=tmp_path,
        roster_dir=roster,
        source_root=source,
        target_dir=target,
    )
    aggregate, aggregate_outputs = (
        read_normalization_recovery_v9_source_content_aggregate(
            target,
            expected_manifest_sha256=published["manifest_sha256"],
        ))
    strict, strict_outputs = read_normalization_recovery_v9_source_content_audit(
        target,
        roster_dir=roster,
        source_root=source,
        expected_manifest_sha256=published["manifest_sha256"],
    )
    assert aggregate == strict == published
    assert aggregate_outputs == strict_outputs
    record_output = aggregate_outputs["source-content.jsonl"][0]
    assert record_output["transient_pair_count"] == 8
    assert record_output["label_pair_surface_published"] == 0
    assert record_output["individual_label_print_count"] == 0
    assert published["summary"]["plain_pair_count"] == 8
