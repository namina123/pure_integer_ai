"""覆盖 recovery-v7 intent/semantic alignment records。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_recovery_v7_intent_semantic_alignment_audit
    as audit,
    ph2_broad_qa_normalization_recovery_v7_intent_semantic_alignment_records
    as records,
)
from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_localization_structure import (
    localization_structure_layout,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_training_records import (
    V5_SOURCE_POLICY_BY_FAMILY,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_conceptnet_alias_records import (
    CONCEPTNET_ALIAS_EVIDENCE_KIND,
    CONCEPTNET_ALIAS_ROUTE_KIND,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_neutral_source_projection_records import (
    GODOT_SOURCE_FAMILY,
    LIBREOFFICE_SOURCE_FAMILY,
    THUNDERBIRD_SOURCE_FAMILY,
    VSCODE_SOURCE_FAMILY,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_variable_structure_records import (
    derive_variable_structure_plans,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
)


def _id(value: str) -> str:
    """形成稳定 synthetic SHA identity。"""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _route(phrase: str, *, unique: int = 1) -> dict[str, object]:
    """构造最小 ConceptNet alias route。"""
    return {
        "alias_route_id": _id(f"route-{phrase}"),
        "english_surface": phrase,
        "record_kind": CONCEPTNET_ALIAS_ROUTE_KIND,
        "unique_chinese_surface": unique,
    }


def _evidence(phrase: str, suffix: list[str]) -> dict[str, object]:
    """构造最小 ConceptNet alias evidence。"""
    return {
        "english_suffix": suffix,
        "english_surface": phrase,
        "record_kind": CONCEPTNET_ALIAS_EVIDENCE_KIND,
    }


def test_structure_tokens_block_phrase_spanning_and_longest_match_wins() -> None:
    """alias 不能跨占位符拼接，同 segment 内必须最长优先。"""
    layout = localization_structure_layout("open %s file")
    route_by_phrase = {
        phrase: {
            "alias_route_id": _id(f"route-{phrase}"),
            "unique_chinese_surface": 1,
        }
        for phrase in ("open", "file", "open file")
    }
    pos_by_phrase = {phrase: ("n",) for phrase in route_by_phrase}
    facts = records._matched_alias_facts(
        tuple(layout["segments"]),
        route_by_phrase=route_by_phrase,
        pos_by_phrase=pos_by_phrase,
    )
    assert layout["structure_tokens"] == ("%s",)
    assert facts["alias_route_ids"] == (
        _id("route-open"), _id("route-file"))
    assert _id("route-open file") not in facts["alias_route_ids"]

    longest = records._matched_alias_facts(
        ("open recent file",),
        route_by_phrase={
            **route_by_phrase,
            "open recent file": {
                "alias_route_id": _id("route-open recent file"),
                "unique_chinese_surface": 1,
            },
        },
        pos_by_phrase={
            **pos_by_phrase,
            "open recent file": ("v",),
        },
    )
    assert longest["alias_route_ids"] == (
        _id("route-open recent file"),)
    assert longest["unmatched_unit_count"] == 0


def test_alias_catalog_preserves_sourced_pos_ambiguity() -> None:
    """同一 alias 的多个来源 POS 必须并存，不能任取一个标签。"""
    route_by_phrase, pos_by_phrase, summary = records._alias_catalog(
        (_evidence("open", ["v"]), _evidence("open", ["n"])),
        (_route("open"),),
        clean_inventory={"open"},
    )
    assert set(route_by_phrase) == {"open"}
    assert pos_by_phrase == {"open": ("n", "v")}
    assert summary["specified_pos_route_count"] == 1
    assert summary["single_specified_pos_route_count"] == 0
    assert summary["multi_specified_pos_route_count"] == 1


def _signature_facts(
        family: str,
        suffix: str,
        output: str,
        ) -> dict[str, object]:
    """构造 signature census 所需的离散 pair facts。"""
    return {
        "alias_pos_roles": (("v",),),
        "alias_route_ids": (_id(f"route-{suffix}"),),
        "alias_unique_chinese_flags": (1,),
        "digit_unit_count": 0,
        "observation_structure_categories": ("PRINTF",),
        "output_sha256": _id(output),
        "pair_id": _id(f"pair-{family}-{suffix}-{output}"),
        "punctuation_codepoints": (58,),
        "source_family": family,
        "source_structure_categories": ("PRINTF",),
        "structure_spacing_profile": ((1, 1),),
        "unit_count": 1,
        "unmatched_unit_count": 0,
    }


def test_signature_census_separates_consensus_from_conflict() -> None:
    """相同离散签名的同 output 与多 output 必须分别入账。"""
    pair_facts = {
        "a": _signature_facts(GODOT_SOURCE_FAMILY, "consensus", "same"),
        "b": _signature_facts(
            LIBREOFFICE_SOURCE_FAMILY, "consensus", "same"),
        "c": _signature_facts(GODOT_SOURCE_FAMILY, "conflict", "left"),
        "d": _signature_facts(
            VSCODE_SOURCE_FAMILY, "conflict", "right"),
    }
    census, _groups = records._signature_census(pair_facts)
    alias = next(item for item in census if item["signature_mode"]
                 == records.SIGNATURE_ALIAS_PUNCTUATION)
    assert alias["cross_family_signature_count"] == 2
    assert alias["cross_family_consensus_count"] == 1
    assert alias["cross_family_conflict_count"] == 1


def _observation(
        family: str,
        suffix: str,
        ) -> dict[str, object]:
    """构造可产生跨 family transformation proposal 的 observation。"""
    input_text = "舊{0}"
    output_text = "新詞{0}"
    return {
        "equal_length": 0,
        "identity_preservation": 0,
        "input_text": input_text,
        "observation_id": _id(f"observation-{family}-{suffix}"),
        "output_text": output_text,
        "source_family": family,
        "source_pair_id": _id(f"pair-{family}-{suffix}"),
        "source_policy_scope": V5_SOURCE_POLICY_BY_FAMILY[family],
        "structure_tokens": ["{0}"],
    }


def _fragment(observation: dict[str, object]) -> dict[str, object]:
    """构造 transformation model 所需的 EDIT_CORE evidence。"""
    return {
        "fragment_id": _id(f"fragment-{observation['observation_id']}"),
        "fragment_kind": "EDIT_CORE",
        "input_text": "舊",
        "observation_id": observation["observation_id"],
        "output_text": "新詞",
        "source_family": observation["source_family"],
    }


def _neutral_row(
        observation: dict[str, object],
        surface: str,
        ) -> tuple[dict[str, object], dict[str, object]]:
    """构造 transient neutral row 及其公开 projection。"""
    output_sha256 = _id(str(observation["output_text"]))
    row = {
        "_neutral_surface": surface,
        "output_sha256": output_sha256,
        "pair_id": observation["source_pair_id"],
        "source_family": observation["source_family"],
    }
    projection = {
        "neutral_surface_sha256": _id(surface),
        "output_sha256": output_sha256,
        "pair_id": observation["source_pair_id"],
        "source_family": observation["source_family"],
    }
    return row, projection


def test_loso_proposal_without_matching_semantic_authority_defers() -> None:
    """即使 transformation proposal 正确，intent signature 不同也必须拒绝。"""
    godot = _observation(GODOT_SOURCE_FAMILY, "godot")
    libreoffice = _observation(LIBREOFFICE_SOURCE_FAMILY, "libreoffice")
    vscode = _observation(VSCODE_SOURCE_FAMILY, "vscode")
    observations = (godot, libreoffice, vscode)
    plans, _summary = derive_variable_structure_plans(observations)
    godot_row, godot_projection = _neutral_row(godot, "open {0}")
    libreoffice_row, libreoffice_projection = _neutral_row(
        libreoffice, "open {0}")
    vscode_row, vscode_projection = _neutral_row(vscode, "close {0}")

    _facts, _families, _signatures, loso, summary = (
        records.derive_intent_semantic_alignment_feasibility(
            observations=observations,
            fragments=(_fragment(godot), _fragment(libreoffice)),
            plans=plans,
            neutral_projections=(
                godot_projection, libreoffice_projection, vscode_projection),
            rows_by_family={
                GODOT_SOURCE_FAMILY: (godot_row,),
                LIBREOFFICE_SOURCE_FAMILY: (libreoffice_row,),
                VSCODE_SOURCE_FAMILY: (vscode_row,),
                THUNDERBIRD_SOURCE_FAMILY: (),
            },
            alias_evidence=(
                _evidence("open", ["v"]),
                _evidence("close", ["v"]),
            ),
            alias_routes=(_route("open"), _route("close")),
        ))
    vscode_loso = next(item for item in loso
                        if item["held_out_source_family"]
                        == VSCODE_SOURCE_FAMILY)
    assert vscode_loso["pre_outcome_counts"] == {
        "EXACT": 1, "UNKNOWN": 0, "WRONG": 0}
    assert vscode_loso["authority_route_available_count"] == 0
    assert vscode_loso["authorized_count"] == 0
    assert vscode_loso["final_outcome_counts"] == {
        "EXACT": 0, "UNKNOWN": 1, "WRONG": 0}
    assert summary["capability_outcome"] == "NE_ZERO_AUTHORIZED_EXACT"


def _fake_outputs() -> tuple[
        dict[str, tuple[dict[str, object], ...]], dict[str, object]]:
    """构造 publisher/reader 所需的小型稳定 census。"""
    return {
        "fact-families.jsonl": ({
            "fact_family_id": _id("fact"), "record_kind": "FACT"},),
        "family-census.jsonl": ({
            "family_census_id": _id("family"), "record_kind": "FAMILY"},),
        "signature-census.jsonl": ({
            "record_kind": "SIGNATURE",
            "signature_census_id": _id("signature")},),
        "loso-census.jsonl": ({
            "loso_id": _id("loso"), "record_kind": "LOSO"},),
    }, {
        "alignment": {"capability_outcome": "NE_ZERO_AUTHORIZED_EXACT"},
        "audit_outcome": "FACILITY_PASS_REPRESENTATION_PARTIAL_CAPABILITY_NE",
        "raw_input_output_or_source_surface_published": 0,
    }


def _patch_audit_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    """隔离 audit synthetic test 与真实 K 盘 predecessor。"""
    monkeypatch.setattr(audit, "_require_k_root", lambda value: Path(value))
    monkeypatch.setattr(
        audit, "_input_state", lambda **_kwargs: ({}, {}, {}, {}, {}, {}, {}))
    monkeypatch.setattr(audit, "_derive", lambda **_kwargs: _fake_outputs())


def _audit_inputs(tmp_path: Path) -> tuple[list[Path], Path, Path]:
    """创建十个目录输入与两个 ConceptNet identity 文件。"""
    directories = [tmp_path / name for name in (
        "protocol", "variable", "replay", "neutral", "transformation",
        "alias", "godot", "libreoffice", "vscode", "thunderbird")]
    for path in directories:
        path.mkdir()
    snapshot = tmp_path / "snapshot.json"
    raw = tmp_path / "raw.gz"
    snapshot.write_bytes(b"{}\n")
    raw.write_bytes(b"raw")
    return directories, snapshot, raw


def _publish_audit(
        tmp_path: Path,
        directories: list[Path],
        snapshot: Path,
        raw: Path,
        ) -> tuple[Path, dict[str, object]]:
    """发布小型 synthetic intent/semantic artifact。"""
    target = tmp_path / "audit"
    manifest = audit.publish_normalization_recovery_v7_intent_semantic_alignment_audit(
        run_root=tmp_path,
        training_protocol_dir=directories[0],
        variable_structure_audit_dir=directories[1],
        source_replay_audit_dir=directories[2],
        neutral_source_projection_dir=directories[3],
        cross_source_transformation_dir=directories[4],
        conceptnet_alias_audit_dir=directories[5],
        godot_source_pack_dir=directories[6],
        libreoffice_source_pack_dir=directories[7],
        vscode_source_pack_dir=directories[8],
        thunderbird_source_pack_dir=directories[9],
        conceptnet_snapshot_manifest_path=snapshot,
        conceptnet_raw_path=raw,
        target_dir=target,
    )
    return target, manifest


def _read_audit(
        target: Path,
        directories: list[Path],
        snapshot: Path,
        raw: Path,
        manifest_sha256: str,
        ) -> tuple[
            dict[str, object],
            dict[str, tuple[dict[str, object], ...]],
        ]:
    """严格回读小型 synthetic intent/semantic artifact。"""
    return audit.read_normalization_recovery_v7_intent_semantic_alignment_audit(
        target,
        training_protocol_dir=directories[0],
        variable_structure_audit_dir=directories[1],
        source_replay_audit_dir=directories[2],
        neutral_source_projection_dir=directories[3],
        cross_source_transformation_dir=directories[4],
        conceptnet_alias_audit_dir=directories[5],
        godot_source_pack_dir=directories[6],
        libreoffice_source_pack_dir=directories[7],
        vscode_source_pack_dir=directories[8],
        thunderbird_source_pack_dir=directories[9],
        conceptnet_snapshot_manifest_path=snapshot,
        conceptnet_raw_path=raw,
        expected_manifest_sha256=manifest_sha256,
    )


def test_intent_semantic_audit_round_trip_nonoverwrite_and_tamper(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """audit 往返一致、拒绝覆盖，并拒绝 records/manifest 篡改。"""
    _patch_audit_inputs(monkeypatch)
    directories, snapshot, raw = _audit_inputs(tmp_path)
    target, published = _publish_audit(
        tmp_path, directories, snapshot, raw)
    manifest, outputs = _read_audit(
        target, directories, snapshot, raw,
        str(published["manifest_sha256"]))
    assert manifest == published
    assert outputs == _fake_outputs()[0]
    assert manifest["status"] == (
        "TRAIN_ONLY_INTENT_SEMANTIC_ALIGNMENT_NE_NOT_RUNTIME")
    with pytest.raises(BroadQaExternalDataError, match="input/target path 非法"):
        _publish_audit(tmp_path, directories, snapshot, raw)

    path = target / "loso-census.jsonl"
    path.write_bytes(canonical_json_line({
        "loso_id": _id("tampered"), "record_kind": "LOSO"}))
    with pytest.raises(BroadQaExternalDataError, match="records/inputs 漂移"):
        _read_audit(
            target, directories, snapshot, raw,
            str(published["manifest_sha256"]))

    manifest_path = target / "manifest.json"
    tampered_payload = path.read_bytes()
    stored = json.loads(manifest_path.read_bytes())
    loso_file = next(item for item in stored["files"]
                     if item["relative_path"] == "loso-census.jsonl")
    loso_file["bytes"] = len(tampered_payload)
    loso_file["sha256"] = hashlib.sha256(tampered_payload).hexdigest()
    encoded = canonical_json_line(stored)
    manifest_path.write_bytes(encoded)
    with pytest.raises(BroadQaExternalDataError, match="records/inputs 漂移"):
        _read_audit(
            target, directories, snapshot, raw,
            hashlib.sha256(encoded).hexdigest())

    path.write_bytes(canonical_json_line(
        _fake_outputs()[0]["loso-census.jsonl"][0]))
    manifest_path.write_bytes(canonical_json_line({
        key: value for key, value in published.items()
        if key != "manifest_sha256"}))
    stored = json.loads(manifest_path.read_bytes())
    stored["production_enabled"] = 1
    encoded = canonical_json_line(stored)
    manifest_path.write_bytes(encoded)
    with pytest.raises(BroadQaExternalDataError, match="manifest 字段漂移"):
        _read_audit(
            target, directories, snapshot, raw,
            hashlib.sha256(encoded).hexdigest())
