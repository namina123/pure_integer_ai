"""独立 ICU normalization 来源与未消费 evaluation 协议测试。"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_icu_evaluation_protocol import (
    NORMALIZATION_ICU_EVALUATION_DIMENSIONS,
    NORMALIZATION_ICU_EVALUATION_METRIC_CONTRACT,
    NORMALIZATION_ICU_EVALUATION_STATUS,
    derive_normalization_icu_evaluation_inventory,
    publish_normalization_icu_evaluation_protocol,
    read_normalization_icu_evaluation_protocol,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_icu_source_pack import (
    NORMALIZATION_ICU_LICENSE_BYTES,
    NORMALIZATION_ICU_LICENSE_SHA256,
    NORMALIZATION_ICU_RULE_BYTES,
    NORMALIZATION_ICU_RULE_SHA256,
    parse_normalization_icu_source,
    publish_normalization_icu_source_pack,
    read_normalization_icu_source_pack,
)
from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_icu_source_pack as source_module,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


def _installed_icu_payloads() -> tuple[bytes, bytes]:
    """仅从显式外部 fixture root 读取官方字节。"""
    configured = os.environ.get("PURE_INTEGER_AI_ICU_SOURCE_ROOT")
    if configured:
        root = Path(configured)
        rule = root / "Hans_Hant.txt"
        license_path = root / "LICENSE"
        if rule.is_file() and license_path.is_file():
            return rule.read_bytes(), license_path.read_bytes()
    pytest.skip("official ICU normalization source fixture is unavailable")


def _synthetic_payloads() -> tuple[bytes, bytes]:
    """返回足以覆盖三种 arrow、context 和 reserve 的小型来源。"""
    rules = [b"\xef\xbb\xbf# synthetic\n", "$Digits = [一二] ;\n".encode()]
    for ordinal in range(20):
        rules.append(
            f"{chr(0x4e00 + ordinal)}↔{chr(0x5000 + ordinal)};\n".encode())
    for ordinal in range(20):
        value = chr(0x5200 + ordinal)
        rules.append(f"{value}↔{value};\n".encode())
    for ordinal in range(1, 21):
        rules.append(f"简{ordinal}↔繁{ordinal};\n".encode())
    rules.extend((
        "只简←只繁;\n".encode(),
        "前简→前繁;\n".encode(),
        "$Digits { 上下文简 → 上下文繁;\n".encode(),
    ))
    license_payload = (
        "UNICODE LICENSE V3\n"
        "SPDX-License-Identifier: Unicode-3.0\n").encode()
    return b"".join(rules), license_payload


def _publish_source(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        *,
        payloads: tuple[bytes, bytes] | None = None,
        ) -> Path:
    """发布 synthetic 或调用方显式提供的 ICU 来源字节。"""
    rule_payload, license_payload = payloads or _synthetic_payloads()
    import hashlib
    monkeypatch.setattr(source_module, "NORMALIZATION_ICU_RULE_BYTES",
                        len(rule_payload))
    monkeypatch.setattr(source_module, "NORMALIZATION_ICU_RULE_SHA256",
                        hashlib.sha256(rule_payload).hexdigest())
    monkeypatch.setattr(source_module, "NORMALIZATION_ICU_LICENSE_BYTES",
                        len(license_payload))
    monkeypatch.setattr(source_module, "NORMALIZATION_ICU_LICENSE_SHA256",
                        hashlib.sha256(license_payload).hexdigest())
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    rule_path = inputs / "Hans_Hant.txt"
    license_path = inputs / "LICENSE"
    rule_path.write_bytes(rule_payload)
    license_path.write_bytes(license_payload)
    target = tmp_path / "source-pack"
    publish_normalization_icu_source_pack(
        run_root=tmp_path,
        rule_source_path=rule_path,
        license_source_path=license_path,
        target_dir=target,
    )
    return target


def test_icu_parser_preserves_lines_and_reverse_direction() -> None:
    """parser 支持跨行 statement，并严格区分三种 arrow 的 T2S 资格。"""
    payload = (b"\xef\xbb\xbf# test\n"
               + "$Digits = [一二] ;\n".encode("utf-8")
               + "简\n  ↔\n  繁;\n".encode("utf-8")
               + "简二←繁二;\n".encode("utf-8")
               + "简三→繁三;\n".encode("utf-8")
               + "$Digits { 简四 → 繁四;\n".encode("utf-8"))
    variables, rules, summary = parse_normalization_icu_source(payload)
    assert len(variables) == 1
    assert len(rules) == 4
    assert rules[0]["line_start_ordinal"] == 3
    assert rules[0]["line_end_ordinal"] == 5
    assert len(rules[0]["physical_lines"]) == 3
    assert (rules[0]["t2s_input"], rules[0]["t2s_expected_output"]) == (
        "繁", "简")
    assert (rules[1]["t2s_input"], rules[1]["t2s_expected_output"]) == (
        "繁二", "简二")
    assert rules[2]["t2s_reverse_eligible"] == 0
    assert rules[3]["has_context"] == 1
    assert rules[3]["t2s_reverse_eligible"] == 0
    assert summary["t2s_reverse_eligible_count"] == 2


def test_icu_source_pack_exact_round_trip_and_counts(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """官方 ICU 字节、许可、parser 统计和零消费边界可严格回读。"""
    rule_payload, license_payload = _installed_icu_payloads()
    assert len(rule_payload) == NORMALIZATION_ICU_RULE_BYTES
    assert len(license_payload) == NORMALIZATION_ICU_LICENSE_BYTES
    import hashlib
    assert hashlib.sha256(rule_payload).hexdigest() == NORMALIZATION_ICU_RULE_SHA256
    assert hashlib.sha256(license_payload).hexdigest() == (
        NORMALIZATION_ICU_LICENSE_SHA256)
    target = _publish_source(
        tmp_path,
        monkeypatch,
        payloads=(rule_payload, license_payload),
    )
    manifest, variables, rules = read_normalization_icu_source_pack(target)
    assert manifest["learned_pack_read_count"] == 0
    assert manifest["evaluation_run_count"] == 0
    assert manifest["production_enabled"] == 0
    assert len(variables) == 2
    assert len(rules) == 4_219
    assert manifest["summary"]["arrow_counts"] == {
        "←": 374, "→": 42, "↔": 3_803}
    assert manifest["summary"]["context_rule_count"] == 7
    assert manifest["summary"]["physical_line_count"] == 4_400
    assert manifest["summary"]["statement_count"] == 4_221
    assert manifest["summary"]["t2s_reverse_eligible_count"] == 4_177


def test_icu_source_reader_rejects_synchronized_record_tamper(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """即使同步改 record SHA 与 manifest，也必须被原始来源重派生拒绝。"""
    target = _publish_source(tmp_path, monkeypatch)
    rules_path = target / "rules.jsonl"
    lines = rules_path.read_bytes().splitlines(keepends=True)
    value = json.loads(lines[0])
    value["t2s_expected_output"] += "改"
    lines[0] = canonical_json_line(value)
    rules_path.write_bytes(b"".join(lines))
    import hashlib
    manifest_path = target / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["files"][2]["bytes"] = rules_path.stat().st_size
    manifest["files"][2]["sha256"] = hashlib.sha256(
        rules_path.read_bytes()).hexdigest()
    manifest_path.write_bytes(canonical_json_line(manifest))
    with pytest.raises(BroadQaExternalDataError, match="records/source 漂移"):
        read_normalization_icu_source_pack(target)


def test_icu_evaluation_freeze_is_unconsumed_and_reserve_has_no_labels(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """evaluation/reserve 在 learned pack 读取前冻结，reserve 只发布身份。"""
    source = _publish_source(tmp_path, monkeypatch)
    target = tmp_path / "evaluation-protocol"
    report = publish_normalization_icu_evaluation_protocol(
        run_root=tmp_path,
        source_pack_dir=source,
        target_dir=target,
    )
    manifest, evaluation, reserve = (
        read_normalization_icu_evaluation_protocol(
            target, source_pack_dir=source))
    assert report["manifest_sha256"] == manifest["manifest_sha256"]
    assert manifest["status"] == NORMALIZATION_ICU_EVALUATION_STATUS
    assert manifest["learned_pack_read_count"] == 0
    assert manifest["evaluation_run_count"] == 0
    assert manifest["mastery_claimed"] == 0
    assert manifest["production_enabled"] == 0
    assert manifest["dimensions"] == NORMALIZATION_ICU_EVALUATION_DIMENSIONS
    assert manifest["metric_contract"] == (
        NORMALIZATION_ICU_EVALUATION_METRIC_CONTRACT)
    assert manifest["dimensions"]["DIRECT_MAPPING_CONSISTENCY"] == {
        "applicable_independent_rule_count_min": 1,
        "bearing": 1,
        "independent_false_accept_count_max": 0,
        "independent_false_reject_count_max": 0,
        "independent_support_count_min": 1,
        "no_applicable_rule_outcome": "NE",
        "unresolved_independent_conflict_count_max": 0,
    }
    assert manifest["dimensions"]["CONTEXT_APPLICATION_OUTCOME"] == {
        "bearing": 1,
        "consumer_execution_required": 1,
        "context_false_accept_count_max": 0,
        "context_false_reject_count_max": 0,
        "defeater_hit_count_must_equal_negative_context_case_count": 1,
        "missing_consumer_outcome": "NE",
        "negative_context_case_count_min": 1,
        "positive_context_case_count_min": 1,
    }
    assert len(evaluation) + len(reserve) == 61
    assert evaluation and reserve
    assert all(set(value) == {
        "evaluation_id", "format_version", "record_kind",
        "source_rule_sha256", "split"} for value in reserve)
    assert {value["mapping_kind"] for value in evaluation} == {
        "CHARACTER", "IDENTITY", "PHRASE"}
    with pytest.raises(BroadQaExternalDataError, match="target 已存在"):
        publish_normalization_icu_evaluation_protocol(
            run_root=tmp_path,
            source_pack_dir=source,
            target_dir=target,
        )


def test_icu_evaluation_reader_rejects_label_and_zero_boundary_tamper(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """标签或未消费边界篡改时，协议严格失败关闭。"""
    source = _publish_source(tmp_path, monkeypatch)
    target = tmp_path / "evaluation-protocol"
    publish_normalization_icu_evaluation_protocol(
        run_root=tmp_path,
        source_pack_dir=source,
        target_dir=target,
    )
    inventory_path = target / "evaluation.inventory.jsonl"
    lines = inventory_path.read_bytes().splitlines(keepends=True)
    value = json.loads(lines[0])
    value["expected_output"] += "改"
    lines[0] = canonical_json_line(value)
    inventory_path.write_bytes(b"".join(lines))
    import hashlib
    manifest_path = target / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["evaluation_inventory"]["bytes"] = inventory_path.stat().st_size
    manifest["evaluation_inventory"]["sha256"] = hashlib.sha256(
        inventory_path.read_bytes()).hexdigest()
    manifest_path.write_bytes(canonical_json_line(manifest))
    with pytest.raises(BroadQaExternalDataError, match="inventory/source 漂移"):
        read_normalization_icu_evaluation_protocol(
            target, source_pack_dir=source)

    target_two = tmp_path / "evaluation-protocol-two"
    publish_normalization_icu_evaluation_protocol(
        run_root=tmp_path,
        source_pack_dir=source,
        target_dir=target_two,
    )
    manifest_path = target_two / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["learned_pack_read_count"] = 1
    manifest_path.write_bytes(canonical_json_line(manifest))
    with pytest.raises(BroadQaExternalDataError, match="manifest 漂移"):
        read_normalization_icu_evaluation_protocol(
            target_two, source_pack_dir=source)


def test_icu_evaluation_derivation_rejects_duplicate_reverse_input() -> None:
    """独立来源若给同一 T2S input 多个记录，协议不得猜测标签。"""
    rule = {
        "arrow": "↔",
        "byte_end": 10,
        "byte_start": 0,
        "has_context": 0,
        "line_end_ordinal": 1,
        "line_start_ordinal": 1,
        "statement_sha256": "a" * 64,
        "t2s_expected_output": "简",
        "t2s_input": "繁",
        "t2s_reverse_eligible": 1,
    }
    duplicate = {**rule, "statement_sha256": "b" * 64}
    with pytest.raises(BroadQaExternalDataError, match="eligible rule 漂移"):
        derive_normalization_icu_evaluation_inventory(
            source_pack_manifest_sha256="c" * 64,
            rules=(rule, duplicate),
        )
