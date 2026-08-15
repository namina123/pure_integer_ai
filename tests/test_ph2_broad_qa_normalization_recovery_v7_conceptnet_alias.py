"""覆盖 recovery-v7 ConceptNet neutral alias 记录与严格发布边界。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_recovery_v7_conceptnet_alias_audit as audit,
)
from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_conceptnet_alias_records import (
    conceptnet_alias_evidence_record,
    derive_conceptnet_alias_routes,
    derive_neutral_phrase_inventory,
    neutral_source_phrases,
    neutral_source_units,
    normalize_conceptnet_term,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_neutral_source_projection_records import (
    GODOT_SOURCE_FAMILY,
    LIBREOFFICE_SOURCE_FAMILY,
    THUNDERBIRD_SOURCE_FAMILY,
    VSCODE_SOURCE_FAMILY,
)
from pure_integer_ai.experiments.ph2_conceptnet_adapter import (
    parse_conceptnet_assertion,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
)


def _id(value: str) -> str:
    """形成稳定 synthetic SHA identity。"""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _component(value: str) -> str:
    """按 ConceptNet assertion URI 规则包围 endpoint。"""
    item = value if value.startswith("/") else "/" + value
    return item if item.endswith("/") else item + "/"


def _assertion(
        start: str,
        end: str,
        *,
        line_number: int,
        license_text: str,
        source: str,
        weight: str = "1.0",
        ):
    """构造并严格解析一条 synthetic 英中 Synonym assertion。"""
    relation = "/r/Synonym"
    assertion_uri = "/a/[" + ",".join((
        _component(relation),
        _component(start),
        _component(end),
    )) + "]"
    metadata = json.dumps({
        "dataset": "/d/test",
        "license": license_text,
        "sources": [{"contributor": source}],
        "weight": json.loads(weight),
    }, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return parse_conceptnet_assertion(
        "\t".join((assertion_uri, relation, start, end, metadata)),
        line_number=line_number,
    )


def _neutral_material():
    """构造两个共享 open route 和一个独立 save route。"""
    rows = {
        GODOT_SOURCE_FAMILY: ({
            "_neutral_surface": "OpenFile",
            "pair_id": _id("godot-pair"),
        },),
        LIBREOFFICE_SOURCE_FAMILY: ({
            "_neutral_surface": "open_file",
            "pair_id": _id("libreoffice-pair"),
        },),
        VSCODE_SOURCE_FAMILY: ({
            "_neutral_surface": "SaveURL",
            "pair_id": _id("vscode-pair"),
        },),
    }
    return derive_neutral_phrase_inventory(rows)


def test_neutral_phrase_extraction_respects_camel_acronym_and_limit() -> None:
    """delimiter、camelCase、acronym 和数字边界形成一至四单元短语。"""
    assert neutral_source_units("openHTTPServer_URL2File") == (
        "open", "http", "server", "url", "2", "file")
    phrases = neutral_source_phrases("openHTTPServer_URL2File")
    assert "open http server url" in phrases
    assert "http server url 2" in phrases
    assert "open http server url 2" not in phrases
    assert len(phrases) == len(set(phrases))
    assert neutral_source_units("Eastern Arabic (٣ ,٢ ,١…)") == (
        "eastern", "arabic", "٣", "٢", "١")


def test_conceptnet_direction_uri_license_source_and_ambiguity() -> None:
    """双向 URI 解码后保留许可、来源承诺、整数 weight 与歧义 route。"""
    phrase_support, pair_phrases, family_records = _neutral_material()
    assertions = (
        _assertion(
            "/c/en/open_file/n",
            "/c/zh/%E6%89%93%E5%BC%80%E6%96%87%E4%BB%B6/v",
            line_number=11,
            license_text="cc:by-sa/4.0",
            source="/s/test/one",
            weight="2.5",
        ),
        _assertion(
            "/c/zh/%E5%BC%80%E5%90%AF%E6%96%87%E4%BB%B6/v",
            "/c/en/open_file/v",
            line_number=12,
            license_text="cc:by/4.0",
            source="/s/test/two",
        ),
        _assertion(
            "/c/en/save/n",
            "/c/zh/%E4%BF%9D%E5%AD%98/v",
            line_number=13,
            license_text="cc:by-sa/4.0",
            source="/s/test/three",
        ),
    )
    evidence = tuple(
        conceptnet_alias_evidence_record(
            item, phrase_support=phrase_support)
        for item in assertions
    )
    assert all(item is not None for item in evidence)
    typed_evidence = tuple(item for item in evidence if item is not None)
    first = typed_evidence[0]
    assert first["english_surface"] == "open file"
    assert first["chinese_surface"] == "打开文件"
    assert first["english_suffix"] == ["n"]
    assert first["chinese_suffix"] == ["v"]
    assert first["license_id"] == "CC-BY-SA-4.0"
    assert first["license_text"] == "cc:by-sa/4.0"
    assert first["source_cluster_sha256"] == (
        assertions[0].source_cluster_sha256)
    assert first["sources"] == [
        {"contributor": "/s/test/one"}]
    assert (first["weight_numerator"], first["weight_denominator"]) == (5, 2)
    assert first["neutral_source_families"] == [
        GODOT_SOURCE_FAMILY, LIBREOFFICE_SOURCE_FAMILY]

    routes, coverage, summary = derive_conceptnet_alias_routes(
        typed_evidence,
        phrase_support=phrase_support,
        pair_phrases=pair_phrases,
        family_records=family_records,
    )
    open_route = next(
        item for item in routes if item["english_surface"] == "open file")
    save_route = next(
        item for item in routes if item["english_surface"] == "save")
    assert open_route["chinese_variant_count"] == 2
    assert open_route["unique_chinese_surface"] == 0
    assert save_route["chinese_variant_count"] == 1
    assert save_route["unique_chinese_surface"] == 1
    assert summary["alias_evidence_count"] == 3
    assert summary["english_route_count"] == 2
    assert summary["ambiguous_english_route_count"] == 1
    assert summary["unique_english_route_count"] == 1
    assert summary["license_evidence_counts"] == {
        "CC-BY-4.0": 1,
        "CC-BY-SA-4.0": 2,
    }
    by_family = {item["source_family"]: item for item in coverage}
    assert by_family[GODOT_SOURCE_FAMILY]["pair_any_alias_count"] == 1
    assert by_family[LIBREOFFICE_SOURCE_FAMILY][
        "pair_any_alias_count"] == 1
    assert by_family[VSCODE_SOURCE_FAMILY]["pair_any_alias_count"] == 1
    assert by_family[THUNDERBIRD_SOURCE_FAMILY] == {
        "family_coverage_id": by_family[THUNDERBIRD_SOURCE_FAMILY][
            "family_coverage_id"],
        "format_version": 1,
        "matched_neutral_phrase_count": 0,
        "pair_any_alias_count": 0,
        "projected_pair_count": 0,
        "record_kind": (
            "NORMALIZATION_RECOVERY_V7_CONCEPTNET_ALIAS_FAMILY_COVERAGE_V1"),
        "source_family": THUNDERBIRD_SOURCE_FAMILY,
        "target_scope": "NEUTRAL_SOURCE_CONCEPTNET_ALIAS_V1",
        "unique_neutral_phrase_count": 0,
    }

    tampered = ({
        **typed_evidence[0],
        "chinese_surface_sha256": "0" * 64,
    }, *typed_evidence[1:])
    with pytest.raises(BroadQaExternalDataError, match="commitment"):
        derive_conceptnet_alias_routes(
            tampered,
            phrase_support=phrase_support,
            pair_phrases=pair_phrases,
            family_records=family_records,
        )


def test_conceptnet_term_decoding_fails_closed_on_bad_utf8() -> None:
    """URI percent bytes 必须形成严格 UTF-8，且英文空白/大小写规范稳定。"""
    assert normalize_conceptnet_term(
        "Open%20File", language="en") == "open file"
    assert normalize_conceptnet_term(
        "%E7%94%B2__%E4%B9%99", language="zh") == "甲  乙"
    with pytest.raises(BroadQaExternalDataError, match="UTF-8"):
        normalize_conceptnet_term("%FF", language="zh")


def _fake_outputs() -> tuple[
        dict[str, tuple[dict[str, object], ...]], dict[str, object]]:
    """构造 publisher/reader 使用的小型稳定输出。"""
    outputs = {
        "alias-evidence.jsonl": ({
            "alias_evidence_id": _id("evidence"),
            "record_kind": "EVIDENCE",
        },),
        "english-alias-routes.jsonl": ({
            "alias_route_id": _id("route"),
            "record_kind": "ROUTE",
        },),
        "family-coverage.jsonl": ({
            "family_coverage_id": _id("coverage"),
            "record_kind": "COVERAGE",
        },),
    }
    summary = {
        "alias": {"english_route_count": 1},
        "capability_claimed": 0,
        "projection_outcome": "PASS",
        "scan": {
            **audit._EXPECTED_SCAN,
            "conceptnet_raw_bytes": 3,
            "conceptnet_raw_sha256": audit.CONCEPTNET_RAW_SHA256,
            "matching_parse_anomaly_codes": {},
        },
    }
    return outputs, summary


def _patch_audit_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    """隔离 synthetic publisher test 与真实 K 盘和 3,400 万行 raw。"""
    monkeypatch.setattr(audit, "_require_k_root", lambda value: Path(value))
    monkeypatch.setattr(audit, "_read_snapshot_manifest", lambda _path: {})
    monkeypatch.setattr(
        audit, "sha256_path", lambda _path: audit.CONCEPTNET_RAW_SHA256)
    monkeypatch.setattr(audit, "_input_state", lambda **_kwargs: {})
    monkeypatch.setattr(audit, "_derive", lambda **_kwargs: _fake_outputs())
    monkeypatch.setattr(
        audit,
        "_validate_stored_outputs",
        lambda **_kwargs: _fake_outputs()[1]["alias"],
    )


def _input_paths(tmp_path: Path) -> tuple[list[Path], Path, Path]:
    """创建 publisher 所需的八个目录及 snapshot/raw 文件。"""
    paths = [tmp_path / name for name in (
        "protocol", "replay", "projection", "transformation", "godot",
        "libreoffice", "vscode", "thunderbird")]
    for path in paths:
        path.mkdir()
    snapshot = tmp_path / "snapshot.json"
    raw = tmp_path / "raw.gz"
    snapshot.write_bytes(b"{}\n")
    raw.write_bytes(b"raw")
    return paths, snapshot, raw


def _publish(
        tmp_path: Path,
        paths: list[Path],
        snapshot: Path,
        raw: Path,
        ) -> tuple[Path, dict[str, object]]:
    """发布 synthetic ConceptNet alias artifact。"""
    target = tmp_path / "alias"
    result = audit.publish_normalization_recovery_v7_conceptnet_alias_audit(
        run_root=tmp_path,
        training_protocol_dir=paths[0],
        source_replay_audit_dir=paths[1],
        neutral_source_projection_dir=paths[2],
        cross_source_transformation_dir=paths[3],
        godot_source_pack_dir=paths[4],
        libreoffice_source_pack_dir=paths[5],
        vscode_source_pack_dir=paths[6],
        thunderbird_source_pack_dir=paths[7],
        conceptnet_snapshot_manifest_path=snapshot,
        conceptnet_raw_path=raw,
        target_dir=target,
    )
    return target, result


def _read(
        target: Path,
        paths: list[Path],
        snapshot: Path,
        raw: Path,
        manifest_sha256: str,
        ):
    """严格回读 synthetic ConceptNet alias artifact。"""
    return audit.read_normalization_recovery_v7_conceptnet_alias_audit(
        target,
        training_protocol_dir=paths[0],
        source_replay_audit_dir=paths[1],
        neutral_source_projection_dir=paths[2],
        cross_source_transformation_dir=paths[3],
        godot_source_pack_dir=paths[4],
        libreoffice_source_pack_dir=paths[5],
        vscode_source_pack_dir=paths[6],
        thunderbird_source_pack_dir=paths[7],
        conceptnet_snapshot_manifest_path=snapshot,
        conceptnet_raw_path=raw,
        expected_manifest_sha256=manifest_sha256,
    )


def test_alias_audit_round_trip_nonoverwrite_and_tamper(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """publisher 往返一致、拒绝覆盖，并拒绝 record/manifest 篡改。"""
    _patch_audit_inputs(monkeypatch)
    paths, snapshot, raw = _input_paths(tmp_path)
    target, published = _publish(tmp_path, paths, snapshot, raw)
    manifest, outputs = _read(
        target, paths, snapshot, raw, str(published["manifest_sha256"]))
    assert manifest == published
    assert outputs == _fake_outputs()[0]
    assert manifest["status"] == (
        audit.NORMALIZATION_RECOVERY_V7_CONCEPTNET_ALIAS_STATUS)
    assert manifest["training_surface_materialized_on_k"] == 1
    assert manifest["public_git_training_surface_published"] == 0
    with pytest.raises(BroadQaExternalDataError, match="input/target path 非法"):
        _publish(tmp_path, paths, snapshot, raw)

    evidence_path = target / "alias-evidence.jsonl"
    evidence_path.write_bytes(canonical_json_line({
        "alias_evidence_id": _id("tampered"),
        "record_kind": "EVIDENCE",
    }))
    with pytest.raises(BroadQaExternalDataError, match="manifest 字段漂移"):
        _read(
            target, paths, snapshot, raw,
            str(published["manifest_sha256"]))

    evidence_path.write_bytes(canonical_json_line(
        _fake_outputs()[0]["alias-evidence.jsonl"][0]))
    manifest_path = target / "manifest.json"
    stored = json.loads(manifest_path.read_bytes())
    stored["production_enabled"] = 1
    encoded = canonical_json_line(stored)
    manifest_path.write_bytes(encoded)
    with pytest.raises(BroadQaExternalDataError, match="manifest 字段漂移"):
        _read(
            target, paths, snapshot, raw,
            hashlib.sha256(encoded).hexdigest())
