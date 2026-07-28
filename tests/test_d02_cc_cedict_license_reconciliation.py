"""D-02 切片 3 CC-CEDICT 许可分歧与 LC-12 缺口 T0。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_cc_cedict_license_reconciliation import (
    HISTORICAL_MANIFEST_PATH,
    HISTORICAL_MANIFEST_SHA256,
    RECONCILIATION_MANIFEST_PATH,
    CcCedictLicenseReconciliationError,
    build_cc_cedict_lc12_supplement,
    build_cc_cedict_license_reconciliation,
    read_cc_cedict_license_reconciliation,
    verify_cc_cedict_license_reconciliation,
    write_cc_cedict_license_reconciliation,
)
from pure_integer_ai.experiments.ph2_raw_snapshot import sha256_path


MANIFEST_PATH = Path(RECONCILIATION_MANIFEST_PATH)


def test_repository_reconciliation_preserves_historical_blocker_and_divergence():
    """新对账不得覆写旧 3.0 expectation artifact 或放行当前 raw。"""
    manifest = read_cc_cedict_license_reconciliation(MANIFEST_PATH)
    current = build_cc_cedict_license_reconciliation()
    assert sha256_path(MANIFEST_PATH) == (
        "7dcccd47604a4d0eaba6e127eb19daeb25974016f19788ece062cc6d7ef1cc6a")
    assert manifest.lc12_supplement != current.lc12_supplement
    assert {item.capability_key for item in current.lc12_supplement.records
            if item.exit_state == "COURSE_FROZEN"} == {
                "ATTRIBUTION_QUOTATION_PERSPECTIVE",
                    "COMPARISON_QUANTITY_MEASURE",
                    "DISCOURSE_INFORMATION_STRUCTURE",
                    "EVALUATOR_RETENTION_RESOURCE", "EVENT_TIME_ASPECT",
                    "LAYERED_GENERATION",
                    "MORPHOLOGY_WORD_FORM",
                "MULTIWORD_CONSTRUCTION", "OPEN_SET_CONTINUAL_LEARNING",
                "PRAGMATIC_CLARIFICATION_REPAIR", "RAW_TEXT_NOISE",
                    "RECURSIVE_PARSE",
                    "REFERENCE_DISCOURSE_REVISION",
                    "TRANSFER_AXES",
                    "TYPED_LEARNING_OBJECTIVES"}
    assert sha256_path(HISTORICAL_MANIFEST_PATH) == HISTORICAL_MANIFEST_SHA256
    assert manifest.historical_license_verdict == "BLOCKED"
    assert manifest.historical_blocker_code == "LICENSE_PARTITION_MISMATCH"
    assert manifest.official_evidence_consistent == 0
    assert manifest.license_verdict == "BLOCKED"
    assert manifest.blocker_code == "OFFICIAL_LICENSE_EVIDENCE_DIVERGENCE"
    assert manifest.redistribution_policy == "BLOCKED"
    assert manifest.release_eligible == 0
    assert manifest.public_source_pack_emitted == 0
    verify_cc_cedict_license_reconciliation(manifest, Path.cwd())


def test_official_evidence_is_scope_specific_and_records_both_license_ids():
    """项目通用页 3.0 与当前下载/header 4.0 不能被压成单一许可。"""
    manifest = build_cc_cedict_license_reconciliation()
    records = {item.evidence_key: item for item in manifest.official_evidence}
    assert tuple(records) == (
        "CC_CEDICT_PROJECT_WIKI_GENERAL",
        "MDBG_CURRENT_DOWNLOAD_PAGE",
        "MDBG_SNAPSHOT_RAW_HEADER",
    )
    assert records["CC_CEDICT_PROJECT_WIKI_GENERAL"].observed_license_id == (
        "CC-BY-SA-3.0")
    assert records["MDBG_CURRENT_DOWNLOAD_PAGE"].observed_license_id == (
        "CC-BY-SA-4.0")
    assert records["MDBG_SNAPSHOT_RAW_HEADER"].payload_sha256 == (
        "c745acaa8d549e6fd3a6cadadf5481c018eef0a0e3dbb2c704c3969c9f1685d3")
    assert records["MDBG_SNAPSHOT_RAW_HEADER"].payload_size_bytes == 3965460


def test_lc12_supplement_keeps_source_gap_but_accepts_independent_courses():
    """CC 来源 gap 保留，独立 CC0 LC-02/LC-03 课程仍可诚实冻结。"""
    ledger = build_cc_cedict_lc12_supplement()
    records = {item.capability_key: item for item in ledger.records}
    assert "W02_CC_CEDICT_BLOCKED_ALTERNATIVES_PARTIAL" in (
        records["MORPHOLOGY_WORD_FORM"].external_prerequisites)
    assert "W02_CC_CEDICT_BLOCKED_ALTERNATIVES_PARTIAL" in (
        records["RAW_TEXT_NOISE"].external_prerequisites)
    assert "W03_CC_CEDICT_BLOCKED_ALTERNATIVES_PARTIAL" in (
        records["MULTIWORD_CONSTRUCTION"].external_prerequisites)
    assert "W03_CC_CEDICT_BLOCKED_ALTERNATIVES_PARTIAL" in (
        records["SOURCE_UNCERTAINTY_REALITY"].external_prerequisites)
    assert {item.capability_key for item in ledger.records
            if item.exit_state == "COURSE_FROZEN"} == {
                "ATTRIBUTION_QUOTATION_PERSPECTIVE",
                    "COMPARISON_QUANTITY_MEASURE",
                    "DISCOURSE_INFORMATION_STRUCTURE",
                    "EVALUATOR_RETENTION_RESOURCE", "EVENT_TIME_ASPECT",
                    "LAYERED_GENERATION",
                    "MORPHOLOGY_WORD_FORM",
                "MULTIWORD_CONSTRUCTION", "OPEN_SET_CONTINUAL_LEARNING",
                "PRAGMATIC_CLARIFICATION_REPAIR", "RAW_TEXT_NOISE",
                    "RECURSIVE_PARSE",
                    "REFERENCE_DISCOURSE_REVISION",
                    "TRANSFER_AXES",
                    "TYPED_LEARNING_OBJECTIVES"}
    assert records["MORPHOLOGY_WORD_FORM"].exit_state == "COURSE_FROZEN"
    assert "data/ph2/manifests/lc02_morphology_course_v1.json" in (
        records["MORPHOLOGY_WORD_FORM"].evidence_refs)
    assert records["MULTIWORD_CONSTRUCTION"].exit_state == "COURSE_FROZEN"
    assert "data/ph2/manifests/lc03_construction_course_v1.json" in (
        records["MULTIWORD_CONSTRUCTION"].evidence_refs)
    assert records["SOURCE_UNCERTAINTY_REALITY"].exit_state != "COURSE_FROZEN"


def test_alternatives_exclude_blocked_source_and_cross_source_pass_claims():
    """Wiktionary/UD/原创等是合法部分替代，不伪造同族独立 PASS。"""
    manifest = build_cc_cedict_license_reconciliation()
    alternatives = {item.stage_key: item for item in manifest.alternative_coverage}
    assert set(alternatives) == {"W-02", "W-03"}
    assert all(item.status == "PARTIAL_ALTERNATIVE"
               for item in alternatives.values())
    assert all("CC_CEDICT_20260725" not in item.source_keys
               for item in alternatives.values())
    assert "UNIFIED_SOURCE_PACK_NOT_FROZEN" in (
        alternatives["W-02"].independence_limitations)
    assert "NO_CROSS_SOURCE_PASS_YET" in (
        alternatives["W-03"].independence_limitations)


def test_reconciliation_round_trip_is_canonical_and_nonoverwriting(tmp_path):
    """许可对账可恢复，且同版本只能逐字节幂等发布。"""
    manifest = build_cc_cedict_license_reconciliation()
    output = tmp_path / "reconciliation.json"
    write_cc_cedict_license_reconciliation(manifest, output)
    assert read_cc_cedict_license_reconciliation(output) == manifest
    write_cc_cedict_license_reconciliation(manifest, output)
    output.write_bytes(b"{}\n")
    with pytest.raises(CcCedictLicenseReconciliationError, match="内容不同"):
        write_cc_cedict_license_reconciliation(manifest, output)


def test_missing_evidence_bad_hash_and_false_release_fail_closed():
    """漏官方证据、历史漂移和伪造一致/放行均不能构成 artifact。"""
    manifest = build_cc_cedict_license_reconciliation()
    with pytest.raises(CcCedictLicenseReconciliationError, match="三项齐全"):
        replace(manifest, official_evidence=manifest.official_evidence[:-1])
    with pytest.raises(CcCedictLicenseReconciliationError, match="hash 漂移"):
        replace(manifest, historical_manifest_sha256="0" * 64)
    with pytest.raises(CcCedictLicenseReconciliationError, match="不得标成一致"):
        replace(manifest, official_evidence_consistent=1)
    with pytest.raises(CcCedictLicenseReconciliationError, match="不得 release"):
        replace(manifest, release_eligible=1)
    with pytest.raises(CcCedictLicenseReconciliationError, match="fail-closed"):
        replace(manifest, license_verdict="PASS")


def test_reader_rejects_extra_fields_and_manifest_has_no_environment_capture(
        tmp_path):
    """cookie/proxy/client/私有绝对路径不进入正式 artifact。"""
    manifest = build_cc_cedict_license_reconciliation()
    output = tmp_path / "reconciliation.json"
    write_cc_cedict_license_reconciliation(manifest, output)
    payload = output.read_text(encoding="utf-8")
    lowered = payload.lower()
    for forbidden in ("cookie", "proxy", "client_ip", "php_session", "d:\\"):
        assert forbidden not in lowered
    output.write_text(payload[:-2] + ',"extra":1}\n', encoding="utf-8")
    with pytest.raises(CcCedictLicenseReconciliationError, match="字段集合"):
        read_cc_cedict_license_reconciliation(output)


def test_verify_detects_historical_manifest_byte_or_semantic_drift(tmp_path):
    """旧 BLOCKED artifact 必须由固定 hash 和语义双重承重。"""
    manifest = build_cc_cedict_license_reconciliation()
    old = Path(HISTORICAL_MANIFEST_PATH)
    copy = tmp_path / HISTORICAL_MANIFEST_PATH
    copy.parent.mkdir(parents=True)
    copy.write_bytes(old.read_bytes())
    verify_cc_cedict_license_reconciliation(manifest, tmp_path)
    copy.write_bytes(copy.read_bytes() + b" ")
    with pytest.raises(CcCedictLicenseReconciliationError, match="hash 不一致"):
        verify_cc_cedict_license_reconciliation(manifest, tmp_path)
