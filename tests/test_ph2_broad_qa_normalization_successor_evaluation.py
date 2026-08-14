"""normalization successor 独立来源与学习前 evaluation 冻结测试。"""
from __future__ import annotations

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
    ph2_broad_qa_normalization_successor_evaluation_protocol as protocol_module,
)
from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_successor_source_pack as source_module,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_successor_evaluation_protocol import (
    NORMALIZATION_SUCCESSOR_EVALUATION_DIMENSIONS,
    NORMALIZATION_SUCCESSOR_EVALUATION_METRIC_CONTRACT,
    NORMALIZATION_SUCCESSOR_EVALUATION_STATUS,
    NORMALIZATION_SUCCESSOR_TARGET_POLICY_SCOPE,
    publish_normalization_successor_evaluation_protocol,
    read_normalization_successor_evaluation_inventory_only,
    read_normalization_successor_evaluation_manifest_only,
    read_normalization_successor_evaluation_protocol,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_successor_source_pack import (
    MEDIAWIKI_LICENSE_BYTES,
    MEDIAWIKI_LICENSE_SHA256,
    MEDIAWIKI_RULE_BYTES,
    MEDIAWIKI_RULE_SHA256,
    UNIHAN_ARCHIVE_BYTES,
    UNIHAN_ARCHIVE_SHA256,
    UNIHAN_LICENSE_BYTES,
    UNIHAN_LICENSE_SHA256,
    parse_normalization_mediawiki_source,
    parse_normalization_unihan_source,
    publish_normalization_successor_source_pack,
    read_normalization_successor_source_pack,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


def _unihan_zip(row_count: int = 400) -> bytes:
    """构造含足量唯一简化关系的内存 Unihan archive。"""
    lines = [
        "# Unihan synthetic\n",
        "U+3400\tkSemanticVariant\tU+3401<kMatthews\n",
    ]
    for ordinal in range(row_count):
        lines.append(
            f"U+{0x6000 + ordinal:04X}\tkSimplifiedVariant\t"
            f"U+{0x7000 + ordinal:04X}\n")
    target = io.BytesIO()
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("Unihan_Variants.txt", "".join(lines).encode())
    return target.getvalue()


def _mediawiki_php(phrase_count: int = 200) -> bytes:
    """构造五表齐全且含 context-sensitive 等长短语的 PHP fixture。"""
    tables: dict[str, list[tuple[str, str]]] = {
        "ZH_TO_HANT": [("简", "繁")],
        "ZH_TO_HANS": [],
        "ZH_TO_TW": [("字节", "位元")],
        "ZH_TO_HK": [("里面", "裏面")],
        "ZH_TO_CN": [("位址", "地址")],
    }
    for ordinal in range(phrase_count):
        suffix = chr(0x8000 + ordinal)
        replacement = chr(0x8100 + ordinal)
        first = "传" if ordinal % 2 == 0 else "傳"
        tables["ZH_TO_HANS"].append(("傳" + suffix, first + replacement))
    lines = [
        "<?php\n", "/**\n", " * synthetic generated table\n", " */\n",
        "\n", "namespace MediaWiki\\Languages\\Data;\n", "\n",
        "class ZhConversion {\n",
    ]
    for name in source_module.MEDIAWIKI_CONVERSION_TABLES:
        lines.append(f"\tpublic const {name} = [\n")
        for input_text, output_text in tables[name]:
            lines.append(f"\t\t'{input_text}' => '{output_text}',\n")
        lines.extend(("\t];\n", "\n"))
    lines.append("}\n")
    return "".join(lines).encode()


def _payloads() -> tuple[bytes, bytes, bytes, bytes]:
    """返回 synthetic official-source 与许可字节。"""
    return (
        _unihan_zip(),
        b"UNICODE LICENSE V3\nsynthetic\n",
        _mediawiki_php(),
        (b"GNU General Public License\nMediaWiki is licensed under "
         b"version 2 or later\n"),
    )


def _publish_source(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> Path:
    """在临时目录发布 synthetic source pack。"""
    payloads = _payloads()
    for name, payload in zip((
            "UNIHAN_ARCHIVE", "UNIHAN_LICENSE",
            "MEDIAWIKI_RULE", "MEDIAWIKI_LICENSE"), payloads):
        monkeypatch.setattr(source_module, f"{name}_BYTES", len(payload))
        monkeypatch.setattr(
            source_module, f"{name}_SHA256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(source_module, "_require_k_root", lambda value: Path(value))
    monkeypatch.setattr(protocol_module, "_require_k_root", lambda value: Path(value))
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    paths = []
    for name, payload in zip((
            "Unihan.zip", "UNICODE-LICENSE.txt",
            "ZhConversion.php", "COPYING"), payloads):
        path = inputs / name
        path.write_bytes(payload)
        paths.append(path)
    target = tmp_path / "source-pack"
    publish_normalization_successor_source_pack(
        run_root=tmp_path,
        unihan_archive_path=paths[0],
        unihan_license_path=paths[1],
        mediawiki_rule_path=paths[2],
        mediawiki_license_path=paths[3],
        target_dir=target,
    )
    return target


def _installed_payloads() -> tuple[bytes, bytes, bytes, bytes]:
    """仅从显式外部 fixture root 读取官方字节。"""
    configured = os.environ.get(
        "PURE_INTEGER_AI_NORMALIZATION_SUCCESSOR_SOURCE_ROOT")
    if configured:
        root = Path(configured)
        paths = tuple(root / name for name in (
            "Unihan-17.0.0.zip", "UNICODE-LICENSE.txt",
            "MediaWiki-ZhConversion.php", "MediaWiki-COPYING"))
        if all(path.is_file() for path in paths):
            return tuple(path.read_bytes() for path in paths)  # type: ignore[return-value]
    pytest.skip("official normalization successor source fixture is unavailable")


def test_successor_source_parsers_preserve_variant_and_php_structure() -> None:
    """两个 parser 保留来源结构并拒绝未知 PHP 行。"""
    unihan, unihan_summary = parse_normalization_unihan_source(_unihan_zip(4))
    assert len(unihan) == 5
    assert unihan_summary["t2s_unambiguous_eligible_count"] == 4
    assert unihan[0]["targets"][0]["source_tags"] == ["kMatthews"]
    mediawiki, mediawiki_summary = parse_normalization_mediawiki_source(
        _mediawiki_php(8))
    assert mediawiki_summary["table_counts"] == {
        "ZH_TO_HANT": 1,
        "ZH_TO_HANS": 8,
        "ZH_TO_TW": 1,
        "ZH_TO_HK": 1,
        "ZH_TO_CN": 1,
    }
    assert mediawiki_summary["phrase_counts"]["ZH_TO_HANS"] == 8
    assert any(item["input_text"].startswith("傳") for item in mediawiki)
    broken = _mediawiki_php(8).replace(
        b"class ZhConversion {\n", b"class ZhConversion {\n\teval('bad');\n")
    with pytest.raises(BroadQaExternalDataError, match="PHP 壳 syntax"):
        parse_normalization_mediawiki_source(broken)


def test_official_successor_sources_have_frozen_identity_and_counts() -> None:
    """显式官方 fixture 的字节身份和完整 parser 统计不漂移。"""
    payloads = _installed_payloads()
    expected = (
        (UNIHAN_ARCHIVE_BYTES, UNIHAN_ARCHIVE_SHA256),
        (UNIHAN_LICENSE_BYTES, UNIHAN_LICENSE_SHA256),
        (MEDIAWIKI_RULE_BYTES, MEDIAWIKI_RULE_SHA256),
        (MEDIAWIKI_LICENSE_BYTES, MEDIAWIKI_LICENSE_SHA256),
    )
    for payload, (size, digest) in zip(payloads, expected):
        assert len(payload) == size
        assert hashlib.sha256(payload).hexdigest() == digest
    unihan, unihan_summary = parse_normalization_unihan_source(payloads[0])
    mediawiki, mediawiki_summary = parse_normalization_mediawiki_source(payloads[2])
    assert len(unihan) == 17_965
    assert unihan_summary["t2s_unambiguous_eligible_count"] == 6_447
    assert len(mediawiki) == 20_198
    assert mediawiki_summary["table_counts"]["ZH_TO_HANS"] == 4_687
    assert mediawiki_summary["phrase_counts"]["ZH_TO_HANS"] == 673


def test_source_pack_round_trip_rederives_original_bytes(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """source pack 记录零消费边界并从原始 ZIP/PHP 重派生。"""
    source = _publish_source(tmp_path, monkeypatch)
    manifest, unihan, mediawiki = read_normalization_successor_source_pack(source)
    assert manifest["learned_pack_read_count"] == 0
    assert manifest["candidate_pack_read_count"] == 0
    assert manifest["failed_icu_evaluation_read_count"] == 0
    assert manifest["evaluation_run_count"] == 0
    assert manifest["production_enabled"] == 0
    assert len(unihan) == 401
    assert len(mediawiki) == 204
    with pytest.raises(BroadQaExternalDataError, match="target 已存在"):
        publish_normalization_successor_source_pack(
            run_root=tmp_path,
            unihan_archive_path=source / "Unihan-17.0.0.zip",
            unihan_license_path=source / "UNICODE-LICENSE.txt",
            mediawiki_rule_path=source / "MediaWiki-ZhConversion.php",
            mediawiki_license_path=source / "MediaWiki-COPYING",
            target_dir=source,
        )


def test_source_reader_rejects_synchronized_record_tamper(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """同步修改 parsed JSONL 与 manifest 仍不能绕过原始来源重派生。"""
    source = _publish_source(tmp_path, monkeypatch)
    path = source / "unihan-variants.jsonl"
    lines = path.read_bytes().splitlines(keepends=True)
    value = json.loads(lines[1])
    value["t2s_expected_output"] = "改"
    lines[1] = canonical_json_line(value)
    path.write_bytes(b"".join(lines))
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    file_record = next(
        item for item in manifest["files"]
        if item["relative_path"] == "unihan-variants.jsonl")
    file_record["bytes"] = path.stat().st_size
    file_record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_bytes(canonical_json_line(manifest))
    with pytest.raises(BroadQaExternalDataError, match="records/source 漂移"):
        read_normalization_successor_source_pack(source)


def test_evaluation_protocol_freezes_six_dimensions_and_label_free_reserve(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """split/metric/threshold 在学习前冻结，reserve 只含不可逆身份。"""
    source = _publish_source(tmp_path, monkeypatch)
    target = tmp_path / "evaluation-protocol"
    report = publish_normalization_successor_evaluation_protocol(
        run_root=tmp_path, source_pack_dir=source, target_dir=target)
    manifest, evaluation, reserve = (
        read_normalization_successor_evaluation_protocol(
            target, source_pack_dir=source))
    assert report["manifest_sha256"] == manifest["manifest_sha256"]
    assert manifest["status"] == NORMALIZATION_SUCCESSOR_EVALUATION_STATUS
    assert manifest["target_policy_scope"] == (
        NORMALIZATION_SUCCESSOR_TARGET_POLICY_SCOPE)
    assert manifest["dimensions"] == NORMALIZATION_SUCCESSOR_EVALUATION_DIMENSIONS
    assert manifest["metric_contract"] == (
        NORMALIZATION_SUCCESSOR_EVALUATION_METRIC_CONTRACT)
    assert len(manifest["dimensions"]) == 6
    assert manifest["learned_pack_read_count"] == 0
    assert manifest["successor_training_source_read_count"] == 0
    assert manifest["failed_icu_evaluation_read_count"] == 0
    assert evaluation and reserve
    assert all(set(item) == {
        "evaluation_id", "format_version", "record_kind",
        "source_record_id", "split"} for item in reserve)
    assert {item["source_key"] for item in evaluation} == {
        "UNICODE_UNIHAN", "MEDIAWIKI_CORE"}
    assert any(item.get("context_sensitive") == 1 for item in evaluation)
    assert all(item["target_policy_scope"] == (
        NORMALIZATION_SUCCESSOR_TARGET_POLICY_SCOPE) for item in evaluation)
    with pytest.raises(BroadQaExternalDataError, match="target 已存在"):
        publish_normalization_successor_evaluation_protocol(
            run_root=tmp_path, source_pack_dir=source, target_dir=target)


def test_evaluation_reader_rejects_label_and_zero_boundary_tamper(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """evaluation label 或学习前零边界被改写时严格失败关闭。"""
    source = _publish_source(tmp_path, monkeypatch)
    target = tmp_path / "evaluation-protocol"
    publish_normalization_successor_evaluation_protocol(
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
        read_normalization_successor_evaluation_protocol(
            target, source_pack_dir=source)

    target_two = tmp_path / "evaluation-protocol-two"
    publish_normalization_successor_evaluation_protocol(
        run_root=tmp_path, source_pack_dir=source, target_dir=target_two)
    manifest_path = target_two / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["successor_training_source_read_count"] = 1
    manifest_path.write_bytes(canonical_json_line(manifest))
    with pytest.raises(BroadQaExternalDataError, match="manifest 漂移"):
        read_normalization_successor_evaluation_protocol(
            target_two, source_pack_dir=source)


def test_manifest_and_evaluation_only_readers_do_not_open_reserve_payload(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """family freeze 与正式运行可分别保持 evaluation/reserve 读取边界。"""
    source = _publish_source(tmp_path, monkeypatch)
    target = tmp_path / "evaluation-protocol"
    report = publish_normalization_successor_evaluation_protocol(
        run_root=tmp_path, source_pack_dir=source, target_dir=target)
    manifest = read_normalization_successor_evaluation_manifest_only(
        target, expected_manifest_sha256=report["manifest_sha256"])
    assert manifest["evaluation_run_count"] == 0
    assert manifest["reserve_label_read_count"] == 0
    (target / "reserve.identity.jsonl").unlink()
    restored, evaluation = (
        read_normalization_successor_evaluation_inventory_only(
            target,
            expected_manifest_sha256=report["manifest_sha256"],
        ))
    assert restored["manifest_sha256"] == report["manifest_sha256"]
    assert len(evaluation) == restored["inventory_summary"]["evaluation_count"]
    with pytest.raises(BroadQaExternalDataError, match="reserve JSONL 不可读"):
        read_normalization_successor_evaluation_protocol(
            target, source_pack_dir=source)
