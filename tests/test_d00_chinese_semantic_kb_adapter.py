"""D-00C ChineseSemanticKB manifest 和只读适配器测试。"""
from __future__ import annotations

from pathlib import Path

import pytest

from pure_integer_ai.experiments.chinese_semantic_kb_adapter import (
    PARSER_DECIMAL_TAB,
    PARSER_DOCUMENT,
    PARSER_RELATION_MARKER,
    PARSER_SURFACE_LINE,
    PARSER_SYMMETRIC_AT,
    PROFILES,
    ChineseKBRecord,
    ChineseSemanticKBAdapter,
    DatasetAnomaly,
    build_manifest,
    parse_decimal_rational,
)
from pure_integer_ai.experiments.data_manifest import ManifestIntegrityError


def _valid_line(profile) -> str:
    """按外部 profile 生成一个最小合法原始记录。"""
    if profile.parser_kind == PARSER_DOCUMENT:
        return "# 来源说明"
    if profile.parser_kind == PARSER_RELATION_MARKER:
        return f"甲,{profile.relation_marker},乙"
    if profile.parser_kind == PARSER_SYMMETRIC_AT:
        return "甲@乙"
    if profile.parser_kind == PARSER_DECIMAL_TAB:
        return "很\t2.50"
    if profile.parser_kind == PARSER_SURFACE_LINE:
        return "词形"
    raise AssertionError(profile.parser_kind)


def _write_minimal_snapshot(root: Path) -> None:
    """创建覆盖全部参与文件的最小只读适配快照。"""
    for profile in PROFILES:
        path = root / profile.relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_valid_line(profile) + "\n", encoding="utf-8")


def test_manifest_and_adapter_replay_same_records_and_byte_spans(tmp_path):
    _write_minimal_snapshot(tmp_path)
    manifest, reports = build_manifest(
        tmp_path,
        dataset_version="fixture-v1",
        unicode_sequence_family=(7001,),
    )
    adapter = ChineseSemanticKBAdapter(manifest, tmp_path)
    events = tuple(adapter.iter_events())

    assert len(reports) == len(PROFILES)
    assert all(report.record_count == 1 for report in reports)
    assert all(report.anomaly_count == 0 for report in reports)
    assert all(isinstance(event, ChineseKBRecord) for event in events)
    assert all(event.span.byte_start == 0 for event in events)
    assert manifest.binding("unicode_sequence_family") == (7001,)
    adapter.verify_scan_counts()


def test_duplicate_bad_line_blank_and_whitespace_are_deterministically_isolated(
        tmp_path):
    _write_minimal_snapshot(tmp_path)
    antonym = tmp_path / "dict/反义关系库.txt"
    antonym.write_text(
        "甲@乙\n乙@甲\n坏行\n \n 丙@丁 \n",
        encoding="utf-8",
    )
    manifest, reports = build_manifest(
        tmp_path,
        dataset_version="fixture-anomaly",
        unicode_sequence_family=(7001,),
    )
    report = next(item for item in reports
                  if item.relative_path == "dict/反义关系库.txt")
    kinds = [item.kind for item in report.anomalies]

    assert report.record_count == 2
    assert kinds == [
        "duplicate_key", "malformed_record", "blank_record",
        "outer_whitespace",
    ]
    item = next(item for item in manifest.files
                if item.relative_path == report.relative_path)
    assert item.anomaly_count == 4
    assert item.record_count == 2
    events = tuple(ChineseSemanticKBAdapter(
        manifest, tmp_path).iter_events(category="antonym"))
    assert sum(isinstance(event, ChineseKBRecord) for event in events) == 2
    assert sum(isinstance(event, DatasetAnomaly) for event in events) == 4


def test_relation_marker_in_third_column_is_recovered_without_guessing(tmp_path):
    _write_minimal_snapshot(tmp_path)
    abstraction = tmp_path / "dict/抽象关系库.txt"
    abstraction.write_text(
        "丁是丁,卯是卯,抽象\n左,抽象,右,含逗号\n",
        encoding="utf-8",
    )
    manifest, reports = build_manifest(
        tmp_path,
        dataset_version="fixture-marker-position",
        unicode_sequence_family=(7001,),
    )
    report = next(item for item in reports
                  if item.relative_path == "dict/抽象关系库.txt")
    records = [event for event in ChineseSemanticKBAdapter(
        manifest, tmp_path).iter_events(category="abstraction")
               if isinstance(event, ChineseKBRecord)]

    assert report.record_count == 2
    assert report.anomaly_count == 0
    assert [record.fields for record in records] == [
        ("丁是丁", "卯是卯"),
        ("左", "右,含逗号"),
    ]


def test_relation_endpoint_may_equal_central_marker(tmp_path):
    """三列标准格式由中列定关系位置，不把同名端点误判成第二个 marker。"""
    _write_minimal_snapshot(tmp_path)
    abstraction = tmp_path / "dict/抽象关系库.txt"
    abstraction.write_text(
        "纸上谈兵,抽象,抽象\n抽象,抽象,思想\n",
        encoding="utf-8",
    )
    manifest, reports = build_manifest(
        tmp_path,
        dataset_version="fixture-marker-endpoint",
        unicode_sequence_family=(7001,),
    )
    report = next(item for item in reports
                  if item.relative_path == "dict/抽象关系库.txt")
    records = [event for event in ChineseSemanticKBAdapter(
        manifest, tmp_path).iter_events(category="abstraction")
               if isinstance(event, ChineseKBRecord)]

    assert report.record_count == 2
    assert report.anomaly_count == 0
    assert [record.fields for record in records] == [
        ("纸上谈兵", "抽象"),
        ("抽象", "思想"),
    ]


def test_decode_error_is_counted_without_polluting_neighboring_files(tmp_path):
    _write_minimal_snapshot(tmp_path)
    stoplist = tmp_path / "dict/停用词.txt"
    stoplist.write_bytes(b"ok\n\xff\nnext\n")
    manifest, reports = build_manifest(
        tmp_path,
        dataset_version="fixture-decode",
        unicode_sequence_family=(7001,),
    )
    report = next(item for item in reports
                  if item.relative_path == "dict/停用词.txt")

    assert report.record_count == 2
    assert [item.kind for item in report.anomalies] == ["decode_error"]
    assert sum(item.record_count for item in reports) == len(PROFILES) + 1
    ChineseSemanticKBAdapter(manifest, tmp_path).verify_scan_counts()


@pytest.mark.parametrize("text, expected", [
    ("2.0", (2, 1)),
    ("0.60", (3, 5)),
    ("-1.25", (-5, 4)),
    ("+3", (3, 1)),
])
def test_decimal_values_never_pass_through_float(text, expected):
    assert parse_decimal_rational(text) == expected


def test_adapter_rejects_raw_mutation_after_manifest(tmp_path):
    _write_minimal_snapshot(tmp_path)
    manifest, _ = build_manifest(
        tmp_path,
        dataset_version="fixture-mutate",
        unicode_sequence_family=(7001,),
    )
    (tmp_path / "dict/情态词.txt").write_text("变化\n", encoding="utf-8")

    with pytest.raises(ManifestIntegrityError):
        ChineseSemanticKBAdapter(manifest, tmp_path)
