"""覆盖 recovery-v8 参数化 gettext source parser。"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_gettext_source_records import (
    derive_normalization_recovery_v8_gettext_source_records,
)


def _po(*, language: str, rows: tuple[tuple[str, str, str], ...]) -> bytes:
    """构造一份简化synthetic PO。"""
    values = [
        'msgid ""',
        'msgstr ""',
        f'"Language: {language}\\n"',
        '"Content-Type: text/plain; charset=UTF-8\\n"',
        '',
    ]
    for flags, source, translation in rows:
        if flags:
            values.append(f"#, {flags}")
        values.extend((
            f'msgid "{source}"',
            f'msgstr "{translation}"',
            '',
        ))
    return "\n".join(values).encode("utf-8")


def _derive(hans: bytes, hant: bytes):
    """用固定synthetic domain调用共享gettext parser。"""
    return derive_normalization_recovery_v8_gettext_source_records(
        source_family="SYNTHETIC_PROJECT",
        source_policy_scope="SYNTHETIC_ZH_TW_TO_ZH_CN_V1",
        license_expression="GPL-2.0-only",
        pair_specs=({
            "domain": "main",
            "zh_Hans": {
                "expected_language": "zh_CN",
                "relative_path": "po/main/zh_CN.po",
            },
            "zh_Hant": {
                "expected_language": "zh_TW",
                "relative_path": "po/main/zh_TW.po",
            },
        },),
        files={
            "po/main/zh_CN.po": hans,
            "po/main/zh_TW.po": hant,
        },
    )


def test_v8_gettext_parser_keeps_active_common_and_structure_facts() -> None:
    """active common pair保留msgid、surface与结构资格。"""
    files, pairs, summary = _derive(
        _po(language="zh_CN", rows=(("", "Open %s", "打开 %s"),)),
        _po(language="zh_TW", rows=(("", "Open %s", "開啟 %s"),)),
    )
    assert len(files) == 2
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair["official_source_text"] == "Open %s"
    assert pair["zh_hans"]["msgstr"] == "打开 %s"
    assert pair["zh_hant"]["msgstr"] == "開啟 %s"
    assert pair["v8_training_eligible"] == 1
    assert summary["content_outcome"] == "PASS_NONZERO_ACTIVE_COMMON_PAIR"


def test_v8_gettext_parser_excludes_fuzzy_empty_and_preserves_unequal(
        ) -> None:
    """fuzzy/empty不进pair，structure unequal仍保留但不具训练资格。"""
    _files, pairs, summary = _derive(
        _po(language="zh_CN", rows=(
            ("", "Open %s", "打开 %s"),
            ("fuzzy", "Close", "关闭"),
            ("", "Empty", ""),
        )),
        _po(language="zh_TW", rows=(
            ("", "Open %s", "開啟 {0}"),
            ("", "Close", "關閉"),
            ("", "Empty", "空"),
        )),
    )
    assert len(pairs) == 1
    assert pairs[0]["structure_equal"] == 0
    assert pairs[0]["v8_training_eligible"] == 0
    assert summary["plain_pair_count"] == 1
    assert summary["structure_equal_count"] == 0


def test_v8_gettext_parser_rejects_language_and_unselected_file() -> None:
    """header language或固定roster外文件漂移时fail closed。"""
    with pytest.raises(BroadQaExternalDataError, match="language header"):
        _derive(
            _po(language="zh_TW", rows=(("", "Open", "打开"),)),
            _po(language="zh_TW", rows=(("", "Open", "開啟"),)),
        )
    with pytest.raises(BroadQaExternalDataError, match="unselected file"):
        derive_normalization_recovery_v8_gettext_source_records(
            source_family="SYNTHETIC_PROJECT",
            source_policy_scope="SYNTHETIC_SCOPE",
            license_expression="GPL-2.0-only",
            pair_specs=({
                "domain": "main",
                "zh_Hans": {
                    "expected_language": "zh_CN",
                    "relative_path": "cn.po",
                },
                "zh_Hant": {
                    "expected_language": "zh_TW",
                    "relative_path": "tw.po",
                },
            },),
            files={
                "cn.po": _po(
                    language="zh_CN", rows=(("", "Open", "打开"),)),
                "tw.po": _po(
                    language="zh_TW", rows=(("", "Open", "開啟"),)),
                "extra.po": b"not selected",
            },
        )


def test_v8_gettext_official_stellarium_aggregate() -> None:
    """提供Git外fixture时核对11 domain完整aggregate。"""
    value = os.environ.get("PURE_INTEGER_AI_V8_STELLARIUM_SOURCE_ROOT")
    if not value:
        pytest.skip("official v8 Stellarium PO fixture is unavailable")
    root = Path(value)
    domains = tuple(sorted(path.parent.name
                           for path in root.glob("*/zh_CN.po")))
    specs = tuple({
        "domain": domain,
        "zh_Hans": {
            "expected_language": "zh_CN",
            "relative_path": f"{domain}/zh_CN.po",
        },
        "zh_Hant": {
            "expected_language": "zh_TW",
            "relative_path": f"{domain}/zh_TW.po",
        },
    } for domain in domains)
    names = tuple(
        item[role]["relative_path"]
        for item in specs for role in ("zh_Hans", "zh_Hant"))
    _files, pairs, summary = (
        derive_normalization_recovery_v8_gettext_source_records(
            source_family="STELLARIUM_PROJECT",
            source_policy_scope=(
                "STELLARIUM_ZH_TW_TO_ZH_CN_FIXED_COMMIT_V1"),
            license_expression="GPL-2.0-only",
            pair_specs=specs,
            files={name: (root / name).read_bytes() for name in names},
        ))
    assert len(domains) == 11
    assert len(pairs) == summary["plain_pair_count"] == 28_331
    assert summary["structure_equal_count"] == 28_278
    assert summary["v8_training_eligible_pair_count"] == 24_972
    assert summary["identity_pair_count"] == 3_375
    assert summary["content_outcome"] == "PASS_NONZERO_ACTIVE_COMMON_PAIR"
