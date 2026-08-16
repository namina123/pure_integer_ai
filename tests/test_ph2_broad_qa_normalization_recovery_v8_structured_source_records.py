"""覆盖 recovery-v8 参数化 Qt TS source parser。"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_structured_source_records import (
    derive_normalization_recovery_v8_qt_ts_source_records,
)


def _ts(
        *,
        language: str,
        translation: str,
        translation_type: str = "",
        source_language: str = "",
        old_source: str = "",
        ) -> bytes:
    """构造一份单 message synthetic TS。"""
    type_text = f' type="{translation_type}"' if translation_type else ""
    source_text = (
        f' sourcelanguage="{source_language}"' if source_language else "")
    old_source_text = f'<oldsource>{old_source}</oldsource>' if old_source else ""
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<!DOCTYPE TS>\n'
        f'<TS version="2.1" language="{language}"{source_text}>'
        '<context><name>Main</name><message id="open">'
        f'<source>Open %1</source>{old_source_text}<comment>verb</comment>'
        f'<translation{type_text}>{translation}</translation>'
        '</message></context></TS>\n'
    ).encode("utf-8")


def _derive(
        hans: bytes,
        hant: bytes,
        *,
        expected_source_language: str = "",
        ):
    """用固定 synthetic source identity 调共享 parser。"""
    return derive_normalization_recovery_v8_qt_ts_source_records(
        source_family="SYNTHETIC_PROJECT",
        source_policy_scope="SYNTHETIC_ZH_TW_TO_ZH_CN_V1",
        license_expression="MIT",
        pair_specs=({
            "domain": "main",
            "zh_Hans": {
                "expected_language": "zh_CN",
                "expected_source_language": expected_source_language,
                "relative_path": "locale/app_zh_CN.ts",
            },
            "zh_Hant": {
                "expected_language": "zh_TW",
                "expected_source_language": expected_source_language,
                "relative_path": "locale/app_zh_TW.ts",
            },
        },),
        files={
            "locale/app_zh_CN.ts": hans,
            "locale/app_zh_TW.ts": hant,
        },
    )


def test_v8_qt_ts_parser_preserves_official_source_and_pair_structure(
        ) -> None:
    """active common pair保留英文source、简繁surface与结构。"""
    files, pairs, summary = _derive(
        _ts(language="zh_CN", translation="打开 %1"),
        _ts(language="zh_TW", translation="開啟 %1"),
    )
    assert len(files) == 2
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair["official_source_text"] == "Open %1"
    assert pair["zh_hans"]["translation"] == "打开 %1"
    assert pair["zh_hant"]["translation"] == "開啟 %1"
    assert pair["zh_hans_structure_tokens"] == ["%1"]
    assert pair["zh_hant_structure_tokens"] == ["%1"]
    assert pair["v8_training_eligible"] == 1
    assert summary["content_outcome"] == "PASS_NONZERO_ACTIVE_COMMON_PAIR"
    assert summary["plain_pair_count"] == 1


def test_v8_qt_ts_parser_accepts_only_declared_source_language() -> None:
    """标准 sourcelanguage 只有在冻结 pair spec 显式声明时才可进入。"""
    hans = _ts(
        language="zh_CN", translation="打开 %1", source_language="en")
    hant = _ts(
        language="zh_TW", translation="開啟 %1", source_language="en")
    _files, pairs, summary = _derive(
        hans, hant, expected_source_language="en")
    assert len(pairs) == 1
    assert summary["plain_pair_count"] == 1
    with pytest.raises(BroadQaExternalDataError, match="root schema"):
        _derive(hans, hant)


def test_v8_qt_ts_parser_accepts_nonsemantic_oldsource_once() -> None:
    """oldsource 可审计但不替代当前 official source identity。"""
    _files, pairs, summary = _derive(
        _ts(language="zh_CN", translation="打开 %1", old_source="Open file"),
        _ts(language="zh_TW", translation="開啟 %1", old_source="Open file"),
    )
    assert len(pairs) == 1
    assert pairs[0]["official_source_text"] == "Open %1"
    assert summary["plain_pair_count"] == 1
    duplicated = _ts(
        language="zh_CN", translation="打开 %1", old_source="Open file").replace(
            b"<oldsource>Open file</oldsource>",
            b"<oldsource>Open file</oldsource><oldsource>Open item</oldsource>")
    with pytest.raises(BroadQaExternalDataError, match="oldsource"):
        _derive(
            duplicated,
            _ts(language="zh_TW", translation="開啟 %1"),
        )


def test_v8_qt_ts_parser_reports_zero_active_without_relaxing_unfinished(
        ) -> None:
    """双方全unfinished时形成content reject而不是偷纳入翻译。"""
    _files, pairs, summary = _derive(
        _ts(
            language="zh_CN", translation="打开 %1",
            translation_type="unfinished"),
        _ts(
            language="zh_TW", translation="開啟 %1",
            translation_type="unfinished"),
    )
    assert pairs == ()
    assert summary["content_outcome"] == "REJECTED_ZERO_ACTIVE_COMMON_PAIR"
    assert summary["plain_pair_count"] == 0
    assert summary["locale_summaries"]["zh_Hans"]["main"][
        "unfinished_count"] == 1
    assert summary["locale_summaries"]["zh_Hant"]["main"][
        "unfinished_count"] == 1


def test_v8_qt_ts_parser_rejects_entity_and_unselected_file() -> None:
    """XML entity与source roster外文件均fail closed。"""
    malicious = _ts(
        language="zh_CN", translation="打开 %1").replace(
            b"<!DOCTYPE TS>", b"<!DOCTYPE TS [<!ENTITY x 'bad'>]>")
    with pytest.raises(BroadQaExternalDataError, match="input contract"):
        _derive(
            malicious,
            _ts(language="zh_TW", translation="開啟 %1"),
        )
    with pytest.raises(BroadQaExternalDataError, match="unselected file"):
        derive_normalization_recovery_v8_qt_ts_source_records(
            source_family="SYNTHETIC_PROJECT",
            source_policy_scope="SYNTHETIC_SCOPE",
            license_expression="MIT",
            pair_specs=({
                "domain": "main",
                "zh_Hans": {
                    "expected_language": "zh_CN",
                    "relative_path": "cn.ts",
                },
                "zh_Hant": {
                    "expected_language": "zh_TW",
                    "relative_path": "tw.ts",
                },
            },),
            files={
                "cn.ts": _ts(language="zh_CN", translation="打开"),
                "tw.ts": _ts(language="zh_TW", translation="開啟"),
                "extra.ts": b"not selected",
            },
        )


def test_v8_qt_ts_official_bitcoin_and_qbittorrent_aggregate() -> None:
    """提供Git外fixture时核对Bitcoin reject与qBittorrent正覆盖。"""
    bitcoin_root = os.environ.get("PURE_INTEGER_AI_V8_BITCOIN_SOURCE_ROOT")
    qbittorrent_root = os.environ.get(
        "PURE_INTEGER_AI_V8_QBITTORRENT_SOURCE_ROOT")
    if not bitcoin_root or not qbittorrent_root:
        pytest.skip("official v8 Qt TS source fixtures are unavailable")

    bitcoin = Path(bitcoin_root)
    _files, pairs, summary = (
        derive_normalization_recovery_v8_qt_ts_source_records(
            source_family="BITCOIN_CORE_PROJECT",
            source_policy_scope=(
                "BITCOIN_ZH_TW_TO_ZH_CN_FIXED_COMMIT_V1"),
            license_expression="MIT",
            pair_specs=({
                "domain": "bitcoin",
                "zh_Hans": {
                    "expected_language": "zh_CN",
                    "relative_path": "bitcoin_zh_CN.ts",
                },
                "zh_Hant": {
                    "expected_language": "zh_TW",
                    "relative_path": "bitcoin_zh_TW.ts",
                },
            },),
            files={
                name: (bitcoin / name).read_bytes()
                for name in ("bitcoin_zh_CN.ts", "bitcoin_zh_TW.ts")
            },
        ))
    assert pairs == ()
    assert summary["content_outcome"] == "REJECTED_ZERO_ACTIVE_COMMON_PAIR"
    assert summary["locale_summaries"]["zh_Hans"]["bitcoin"] == {
        "active_plain_count": 0,
        "empty_active_translation_count": 0,
        "message_count": 1_137,
        "numerus_count": 16,
        "obsolete_count": 0,
        "unfinished_count": 1_121,
        "vanished_count": 0,
    }
    assert summary["locale_summaries"]["zh_Hant"]["bitcoin"] == {
        "active_plain_count": 0,
        "empty_active_translation_count": 0,
        "message_count": 1_142,
        "numerus_count": 16,
        "obsolete_count": 0,
        "unfinished_count": 1_126,
        "vanished_count": 0,
    }

    qbit = Path(qbittorrent_root)
    _files, pairs, summary = (
        derive_normalization_recovery_v8_qt_ts_source_records(
            source_family="QBITTORRENT_PROJECT",
            source_policy_scope=(
                "QBITTORRENT_ZH_TW_TO_ZH_CN_FIXED_COMMIT_V1"),
            license_expression=(
                "LicenseRef-qBittorrent-GPL-2.0-or-later-with-exception"),
            pair_specs=({
                "domain": "qbittorrent",
                "zh_Hans": {
                    "expected_language": "zh",
                    "relative_path": "qbittorrent_zh_CN.ts",
                },
                "zh_Hant": {
                    "expected_language": "zh_TW",
                    "relative_path": "qbittorrent_zh_TW.ts",
                },
            },),
            files={
                name: (qbit / name).read_bytes()
                for name in (
                    "qbittorrent_zh_CN.ts", "qbittorrent_zh_TW.ts")
            },
        ))
    assert summary["content_outcome"] == "PASS_NONZERO_ACTIVE_COMMON_PAIR"
    assert summary["plain_pair_count"] == len(pairs) == 2_436
    assert summary["locale_summaries"]["zh_Hans"]["qbittorrent"][
        "active_plain_count"] == 2_436
    assert summary["locale_summaries"]["zh_Hant"]["qbittorrent"][
        "active_plain_count"] == 2_436
