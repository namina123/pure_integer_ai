"""normalization recovery v2 Firefox 来源与学习前协议冻结测试。"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import zipfile

import pytest

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_recovery_evaluation_protocol as protocol_module,
)
from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_recovery_source_pack as source_module,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_evaluation_protocol import (
    NORMALIZATION_RECOVERY_EVALUATION_DIMENSIONS,
    NORMALIZATION_RECOVERY_EVALUATION_METRIC_CONTRACT,
    NORMALIZATION_RECOVERY_EVALUATION_STATUS,
    NORMALIZATION_RECOVERY_TARGET_POLICY_SCOPE,
    publish_normalization_recovery_evaluation_protocol,
    read_normalization_recovery_evaluation_inventory_only,
    read_normalization_recovery_evaluation_manifest_only,
    read_normalization_recovery_evaluation_protocol,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_source_pack import (
    parse_normalization_recovery_firefox_archive,
    publish_normalization_recovery_source_pack,
    read_normalization_recovery_source_pack,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


def _ftl_pair() -> tuple[bytes, bytes]:
    """构造含局部、phrase、identity、context 与 Placeable 的双 locale FTL。"""
    cn = ["# synthetic zh-CN\n"]
    tw = ["# synthetic zh-TW\n"]
    stable = (
        ("條", "条"), ("頁", "页"), ("網", "网"), ("語", "语"),
        ("門", "门"), ("風", "风"), ("書", "书"), ("車", "车"),
    )
    for ordinal, (traditional, simplified) in enumerate(stable):
        cn.append(f"local-{ordinal} = 项目{simplified}{ordinal}\n")
        tw.append(f"local-{ordinal} = 项目{traditional}{ordinal}\n")
    cn.extend(("context-one = 组合\n", "context-two = 模块\n"))
    tw.extend(("context-one = 組合\n", "context-two = 模組\n"))
    for ordinal in range(80):
        cn.append(f"phrase-{ordinal} = 导入信息项目{ordinal}\n")
        tw.append(f"phrase-{ordinal} = 匯入資訊項目{ordinal}\n")
    for ordinal in range(30):
        cn.append(f"identity-{ordinal} = 保持内容{ordinal}\n")
        tw.append(f"identity-{ordinal} = 保持内容{ordinal}\n")
    cn.append("with-placeable = 项目 { $name }\n")
    tw.append("with-placeable = 項目 { $name }\n")
    return "".join(cn).encode(), "".join(tw).encode()


def _archive(*, malformed: bool = False) -> bytes:
    """构造只含许可与两个 locale 的内存 Git archive 形状。"""
    cn, tw = _ftl_pair()
    if malformed:
        cn += b"broken = {\n"
    target = io.BytesIO()
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("LICENSE", b"Mozilla Public License Version 2.0\n")
        archive.writestr("zh-CN/browser/test.ftl", cn)
        archive.writestr("zh-TW/browser/test.ftl", tw)
    return target.getvalue()


def _freeze_synthetic_identity(
        monkeypatch: pytest.MonkeyPatch,
        payload: bytes,
        ) -> None:
    """把 synthetic archive 身份注入 publisher，不放宽 parser/reader。"""
    files, pairs, summary = parse_normalization_recovery_firefox_archive(payload)
    monkeypatch.setattr(source_module, "FIREFOX_L10N_ARCHIVE_BYTES", len(payload))
    monkeypatch.setattr(
        source_module, "FIREFOX_L10N_ARCHIVE_SHA256",
        hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(
        source_module, "FIREFOX_L10N_ARCHIVE_FILE_COUNT", len(files))
    monkeypatch.setattr(
        source_module, "FIREFOX_L10N_LICENSE_BYTES",
        summary["license_bytes"])
    monkeypatch.setattr(
        source_module, "FIREFOX_L10N_LICENSE_SHA256",
        summary["license_sha256"])
    monkeypatch.setattr(
        source_module, "FIREFOX_L10N_LICENSE_BLOB_SHA1",
        summary["license_git_blob_sha1"])
    monkeypatch.setattr(
        source_module, "FIREFOX_L10N_LOCALE_TREES",
        summary["locale_trees"])
    monkeypatch.setattr(
        source_module, "FIREFOX_L10N_COMMON_PATTERN_COUNT", len(pairs))
    monkeypatch.setattr(
        source_module, "FIREFOX_L10N_STRUCTURE_EQUAL_COUNT",
        summary["structure_equal_count"])
    monkeypatch.setattr(
        source_module, "FIREFOX_L10N_PLAIN_PAIR_COUNT",
        summary["plain_pair_count"])
    locale_summaries = summary["locale_summaries"]
    monkeypatch.setattr(source_module, "FIREFOX_L10N_FTL_FILE_COUNTS", {
        key: value["ftl_file_count"] for key, value in locale_summaries.items()
    })
    monkeypatch.setattr(source_module, "FIREFOX_L10N_PATTERN_COUNTS", {
        key: value["pattern_count"] for key, value in locale_summaries.items()
    })


def _small_dimensions(monkeypatch: pytest.MonkeyPatch) -> None:
    """缩小 synthetic fixture 的库存门，不改变六维 hard conjunct。"""
    dimensions = deepcopy(NORMALIZATION_RECOVERY_EVALUATION_DIMENSIONS)
    dimensions["LOCAL_MAPPING_TRANSFER"]["local_mapping_inventory_min"] = 1
    dimensions["END_TO_END_COVERAGE"]["phrase_inventory_min"] = 8
    dimensions["END_TO_END_COVERAGE"]["identity_inventory_min"] = 4
    dimensions["INDEPENDENT_CONTEXT_TRANSFER"]["context_inventory_min"] = 0
    monkeypatch.setattr(
        protocol_module, "NORMALIZATION_RECOVERY_EVALUATION_DIMENSIONS",
        dimensions)


def _publish_source(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> Path:
    """在临时目录发布 synthetic Firefox source pack。"""
    payload = _archive()
    _freeze_synthetic_identity(monkeypatch, payload)
    monkeypatch.setattr(source_module, "_require_k_root", lambda value: Path(value))
    monkeypatch.setattr(protocol_module, "_require_k_root", lambda value: Path(value))
    _small_dimensions(monkeypatch)
    input_path = tmp_path / source_module.FIREFOX_L10N_ARCHIVE_NAME
    input_path.write_bytes(payload)
    target = tmp_path / "source-pack"
    publish_normalization_recovery_source_pack(
        run_root=tmp_path, archive_path=input_path, target_dir=target)
    return target


def test_firefox_parser_preserves_fluent_structure_and_rejects_junk() -> None:
    """parser 保留 message/attribute/Placeable 结构并对 Junk 失败关闭。"""
    files, pairs, summary = parse_normalization_recovery_firefox_archive(
        _archive())
    assert len(files) == 3
    assert summary["locale_summaries"]["zh-CN"]["ftl_file_count"] == 1
    assert summary["locale_summaries"]["zh-TW"]["ftl_file_count"] == 1
    placeable = next(item for item in pairs
                     if item["message_id"] == "with-placeable")
    assert placeable["structure_equal"] == 1
    assert placeable["plain_pair_eligible"] == 0
    assert placeable["zh_cn"]["surface_text"] is None
    with pytest.raises(BroadQaExternalDataError, match="parser Junk"):
        parse_normalization_recovery_firefox_archive(_archive(malformed=True))


def test_recovery_source_pack_round_trip_rederives_git_archive(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """source pack 从 raw archive 重派生 tree、pattern 和零消费边界。"""
    source = _publish_source(tmp_path, monkeypatch)
    manifest, files, pairs = read_normalization_recovery_source_pack(source)
    assert manifest["evaluation_run_count"] == 0
    assert manifest["prior_formal_item_read_count"] == 0
    assert manifest["recovery_training_source_read_count"] == 0
    assert manifest["reserve_payload_read_count"] == 0
    assert manifest["production_enabled"] == 0
    assert len(files) == 3
    assert pairs
    with pytest.raises(BroadQaExternalDataError, match="target 已存在"):
        publish_normalization_recovery_source_pack(
            run_root=tmp_path,
            archive_path=source / source_module.FIREFOX_L10N_ARCHIVE_NAME,
            target_dir=source,
        )


def test_source_reader_rejects_synchronized_pair_tamper(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """同步修改 pair JSONL 与 manifest 仍不能绕过 raw Fluent 重派生。"""
    source = _publish_source(tmp_path, monkeypatch)
    pair_path = source / "pattern-pairs.jsonl"
    lines = pair_path.read_bytes().splitlines(keepends=True)
    value = json.loads(lines[0])
    value["zh_cn"]["surface_text"] = "篡改"
    lines[0] = canonical_json_line(value)
    pair_path.write_bytes(b"".join(lines))
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    artifact = next(item for item in manifest["files"]
                    if item["relative_path"] == "pattern-pairs.jsonl")
    artifact["bytes"] = pair_path.stat().st_size
    artifact["sha256"] = hashlib.sha256(pair_path.read_bytes()).hexdigest()
    manifest_path.write_bytes(canonical_json_line(manifest))
    with pytest.raises(BroadQaExternalDataError, match="records/source 漂移"):
        read_normalization_recovery_source_pack(source)


def test_recovery_protocol_freezes_six_dimensions_and_label_free_reserve(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """来源、split、六维 metric 与 reserve 在 recovery TRAIN 前冻结。"""
    source = _publish_source(tmp_path, monkeypatch)
    target = tmp_path / "evaluation-protocol"
    report = publish_normalization_recovery_evaluation_protocol(
        run_root=tmp_path, source_pack_dir=source, target_dir=target)
    manifest, evaluation, reserve = (
        read_normalization_recovery_evaluation_protocol(
            target, source_pack_dir=source))
    assert report["manifest_sha256"] == manifest["manifest_sha256"]
    assert manifest["status"] == NORMALIZATION_RECOVERY_EVALUATION_STATUS
    assert manifest["target_policy_scope"] == (
        NORMALIZATION_RECOVERY_TARGET_POLICY_SCOPE)
    assert len(manifest["dimensions"]) == 6
    assert manifest["metric_contract"] == (
        NORMALIZATION_RECOVERY_EVALUATION_METRIC_CONTRACT)
    assert manifest["recovery_training_source_read_count"] == 0
    assert manifest["prior_formal_item_read_count"] == 0
    assert evaluation and reserve
    assert all("input_text" not in item and "expected_output" not in item
               and "source_pair_id" not in item for item in reserve)
    assert any(item["identity_preservation"] == 1 for item in evaluation)
    assert any("LOCAL_MAPPING_TRANSFER" in item["family_keys"]
               for item in evaluation)
    assert any("END_TO_END_COVERAGE" in item["family_keys"]
               for item in evaluation)
    with pytest.raises(BroadQaExternalDataError, match="target 已存在"):
        publish_normalization_recovery_evaluation_protocol(
            run_root=tmp_path, source_pack_dir=source, target_dir=target)


def test_protocol_reader_rejects_label_and_zero_boundary_tamper(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """evaluation label 或学习前零边界被改写时严格失败关闭。"""
    source = _publish_source(tmp_path, monkeypatch)
    target = tmp_path / "evaluation-protocol"
    publish_normalization_recovery_evaluation_protocol(
        run_root=tmp_path, source_pack_dir=source, target_dir=target)
    inventory_path = target / "evaluation.inventory.jsonl"
    lines = inventory_path.read_bytes().splitlines(keepends=True)
    value = json.loads(lines[0])
    value["expected_output"] += "改"
    lines[0] = canonical_json_line(value)
    inventory_path.write_bytes(b"".join(lines))
    manifest_path = target / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["evaluation_inventory"]["bytes"] = inventory_path.stat().st_size
    manifest["evaluation_inventory"]["sha256"] = hashlib.sha256(
        inventory_path.read_bytes()).hexdigest()
    manifest_path.write_bytes(canonical_json_line(manifest))
    with pytest.raises(BroadQaExternalDataError, match="inventory/source 漂移"):
        read_normalization_recovery_evaluation_protocol(
            target, source_pack_dir=source)

    target_two = tmp_path / "evaluation-protocol-two"
    publish_normalization_recovery_evaluation_protocol(
        run_root=tmp_path, source_pack_dir=source, target_dir=target_two)
    manifest_path = target_two / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["recovery_training_source_read_count"] = 1
    manifest_path.write_bytes(canonical_json_line(manifest))
    with pytest.raises(BroadQaExternalDataError, match="manifest 漂移"):
        read_normalization_recovery_evaluation_protocol(
            target_two, source_pack_dir=source)


def test_manifest_and_evaluation_only_readers_do_not_open_reserve(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """family freeze 与 formal runner 可保持 source/reserve 物理读取边界。"""
    source = _publish_source(tmp_path, monkeypatch)
    target = tmp_path / "evaluation-protocol"
    report = publish_normalization_recovery_evaluation_protocol(
        run_root=tmp_path, source_pack_dir=source, target_dir=target)
    manifest = read_normalization_recovery_evaluation_manifest_only(
        target, expected_manifest_sha256=report["manifest_sha256"])
    assert manifest["evaluation_run_count"] == 0
    assert manifest["reserve_payload_read_count"] == 0
    (target / "reserve.identity.jsonl").unlink()
    restored, evaluation = (
        read_normalization_recovery_evaluation_inventory_only(
            target, expected_manifest_sha256=report["manifest_sha256"]))
    assert restored["manifest_sha256"] == report["manifest_sha256"]
    assert len(evaluation) == restored["inventory_summary"]["evaluation_count"]
    with pytest.raises(BroadQaExternalDataError, match="reserve JSONL 不可读"):
        read_normalization_recovery_evaluation_protocol(
            target, source_pack_dir=source)


def test_official_firefox_archive_identity_and_inventory() -> None:
    """显式官方 fixture 的 Git tree 与完整 Fluent inventory 不漂移。"""
    configured = os.environ.get(
        "PURE_INTEGER_AI_NORMALIZATION_RECOVERY_FIREFOX_ARCHIVE")
    if not configured or not Path(configured).is_file():
        pytest.skip("official Firefox l10n archive fixture is unavailable")
    payload = Path(configured).read_bytes()
    assert len(payload) == source_module.FIREFOX_L10N_ARCHIVE_BYTES
    assert hashlib.sha256(payload).hexdigest() == (
        source_module.FIREFOX_L10N_ARCHIVE_SHA256)
    files, pairs, summary = parse_normalization_recovery_firefox_archive(payload)
    assert len(files) == source_module.FIREFOX_L10N_ARCHIVE_FILE_COUNT
    assert len(pairs) == source_module.FIREFOX_L10N_COMMON_PATTERN_COUNT
    assert summary["locale_trees"] == source_module.FIREFOX_L10N_LOCALE_TREES
    assert summary["structure_equal_count"] == (
        source_module.FIREFOX_L10N_STRUCTURE_EQUAL_COUNT)
    assert summary["plain_pair_count"] == source_module.FIREFOX_L10N_PLAIN_PAIR_COUNT
