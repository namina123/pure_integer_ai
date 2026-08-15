"""覆盖 v9 gettext obsolete 严格分母。"""
from __future__ import annotations

from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v9_gettext_source_records import (
    derive_normalization_recovery_v9_gettext_source_records,
)


def _po(language: str, translation: str, *, obsolete: bool = False) -> bytes:
    """构造含一个 active 与一个可选 obsolete entry 的最小 PO。"""
    marker = "#~ " if obsolete else ""
    text = (
        "# This file is distributed under the same license as the gimp package.\n"
        "msgid \"\"\n"
        "msgstr \"\"\n"
        '"Project-Id-Version: GIMP-test\\n"\n'
        f'"Language: {language}\\n"\n'
        '"Content-Type: text/plain; charset=UTF-8\\n"\n'
        "\n"
        "msgid \"Open\"\n"
        f"msgstr \"{translation}\"\n"
        "\n"
        f"{marker}msgid \"Old\"\n"
        f"{marker}msgstr \"旧\"\n"
    )
    return text.encode("utf-8")


def test_v9_gettext_excludes_obsolete_from_plain_denominator() -> None:
    """任一侧 obsolete 都不能进入 held-out plain pair。"""
    files = {
        "po/zh_CN.po": _po("zh_CN", "打开", obsolete=True),
        "po/zh_TW.po": _po("zh_TW", "開啟", obsolete=True),
    }
    source_files, pairs, summary = (
        derive_normalization_recovery_v9_gettext_source_records(
            source_family="GIMP_PROJECT",
            source_policy_scope="TEST_SCOPE",
            license_expression="GPL-3.0-or-later",
            pair_specs=({
                "domain": "po",
                "zh_Hans": {
                    "expected_language": "zh_CN",
                    "relative_path": "po/zh_CN.po",
                },
                "zh_Hant": {
                    "expected_language": "zh_TW",
                    "relative_path": "po/zh_TW.po",
                },
            },),
            files=files,
        ))
    assert len(source_files) == 2
    assert len(pairs) == 1
    assert pairs[0]["source_identity"]["msgid"] == "Open"
    assert summary["common_source_identity_count"] == 2
    assert summary["excluded_any_count"] == 1
    assert summary["excluded_obsolete_count"] == 1
    assert summary["plain_pair_count"] == 1
