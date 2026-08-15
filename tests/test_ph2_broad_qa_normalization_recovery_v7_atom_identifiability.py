"""覆盖 recovery-v7 atom identifiable lower-bound records/source。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_recovery_v7_atom_identifiability_audit
    as audit,
)
from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_recovery_v7_atom_identifiability_sources
    as sources,
)
from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_training_records import (
    V5_SOURCE_POLICY_BY_FAMILY,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_atom_identifiability_records import (
    derive_atom_identifiability_authorizations,
    derive_atom_identifiability_feasibility,
    score_atom_identifiability_authorizations,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_neutral_source_projection_records import (
    GODOT_SOURCE_FAMILY,
    LIBREOFFICE_SOURCE_FAMILY,
    VSCODE_SOURCE_FAMILY,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_variable_structure_records import (
    derive_variable_structure_plans,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    canonical_json_line,
)


def _id(value: str) -> str:
    """形成稳定 synthetic SHA identity。"""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _observation(
        family: str,
        suffix: str,
        input_text: str,
        output_text: str,
        *,
        tokens: list[str] | None = None,
        ) -> dict[str, object]:
    """构造最小 TRAIN observation。"""
    return {
        "equal_length": int(len(input_text) == len(output_text)),
        "identity_preservation": int(input_text == output_text),
        "input_text": input_text,
        "observation_id": _id(f"observation-{family}-{suffix}"),
        "output_text": output_text,
        "source_family": family,
        "source_pair_id": _id(f"pair-{family}-{suffix}"),
        "source_policy_scope": V5_SOURCE_POLICY_BY_FAMILY[family],
        "structure_tokens": [] if tokens is None else tokens,
    }


def _fragment(
        observation: dict[str, object],
        suffix: str,
        ) -> dict[str, object]:
    """构造 frozen EDIT_CORE lexical evidence。"""
    return {
        "fragment_id": _id(f"fragment-{suffix}"),
        "fragment_kind": "EDIT_CORE",
        "input_text": "外掛程式",
        "observation_id": observation["observation_id"],
        "output_text": "插件",
        "source_family": observation["source_family"],
    }


def _material(
        *,
        source: str = "Plugin: {0}",
        held_output: str = "插件: {0}",
        ) -> tuple[
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            dict[str, str],
        ]:
    """构造两家非 held、三 occurrence 的 stable lexical route。"""
    godot_a = _observation(
        GODOT_SOURCE_FAMILY, "godot-a", "外掛程式: {0}", "插件: {0}",
        tokens=["{0}"])
    godot_b = _observation(
        GODOT_SOURCE_FAMILY, "godot-b", "外掛程式", "插件")
    libreoffice = _observation(
        LIBREOFFICE_SOURCE_FAMILY, "libreoffice",
        "外掛程式: {0}", "插件: {0}", tokens=["{0}"])
    held = _observation(
        VSCODE_SOURCE_FAMILY, "held", "外掛程式: {0}", held_output,
        tokens=["{0}"],
    )
    plans, _summary = derive_variable_structure_plans((held,))
    return (
        (godot_a, godot_b, libreoffice, held),
        (_fragment(godot_a, "godot"),
         _fragment(libreoffice, "libreoffice")),
        plans,
        {str(held["source_pair_id"]): source},
    )


def test_identifiable_lower_bound_authorizes_exact_without_label_access() -> None:
    """stable lexical + source layout identity 得到非零 exact、零 wrong。"""
    observations, fragments, plans, source_map = _material()
    authorizations, census = derive_atom_identifiability_authorizations(
        observations=observations, fragments=fragments, plans=plans,
        official_source_by_pair=source_map, opencc_routes={},
        morphology_by_form={"plugin": (("plugin", "N;SG"),)})
    assert census["authorized_proposal_count"] == 1
    assert authorizations[0]["authorization_decision"] == "AUTHORIZED"
    assert authorizations[0]["held_label_reads"] == 0
    assert authorizations[0]["stable_lexical_atom_count"] == 1

    records, scoring = score_atom_identifiability_authorizations(
        authorizations,
        labels_by_observation={
            str(observations[-1]["observation_id"]): (
                "外掛程式: {0}", "插件: {0}")})
    assert scoring["outcome_counts"] == {
        "EXACT": 1, "UNKNOWN": 0, "WRONG": 0}
    assert records[0]["surface_published"] == 0
    encoded = canonical_json_bytes(records)
    assert "外掛程式".encode("utf-8") not in encoded
    assert "插件".encode("utf-8") not in encoded


def test_authorization_is_label_blind_and_wrong_is_scored_after_freeze() -> None:
    """改变 held label 不改变 authorization，只在独立 scoring 暴露 WRONG。"""
    baseline = _material()
    changed = _material(held_output="其他: {0}")
    left, _left_census = derive_atom_identifiability_authorizations(
        observations=baseline[0], fragments=baseline[1], plans=baseline[2],
        official_source_by_pair=baseline[3], opencc_routes={},
        morphology_by_form={"plugin": (("plugin", "N;SG"),)})
    right, _right_census = derive_atom_identifiability_authorizations(
        observations=changed[0], fragments=changed[1], plans=changed[2],
        official_source_by_pair=changed[3], opencc_routes={},
        morphology_by_form={"plugin": (("plugin", "N;SG"),)})
    assert left == right
    _records, scoring = score_atom_identifiability_authorizations(
        right,
        labels_by_observation={
            str(changed[0][-1]["observation_id"]): (
                "外掛程式: {0}", "其他: {0}")})
    assert scoring["feasibility_outcome"] == "FAIL_WRONG_AUTHORIZATION"
    assert scoring["outcome_counts"]["WRONG"] == 1


def test_marked_morphology_and_layout_drift_defer() -> None:
    """plural/aspect ambiguity或 source layout 漂移必须完整 defer。"""
    plural = _material(source="Plugins: {0}")
    records, _census, summary = derive_atom_identifiability_feasibility(
        observations=plural[0], fragments=plural[1], plans=plural[2],
        official_source_by_pair=plural[3], opencc_routes={},
        morphology_by_form={"plugins": (("plugin", "N;PL"),)})
    assert summary["authorization"]["authorized_proposal_count"] == 0
    assert summary["scoring"]["outcome_counts"]["UNKNOWN"] == 1
    assert records[0]["reason_counts"] == {
        "MARKED_MORPHOLOGY_UNRESOLVED": 1}

    layout = _material(source="Plugin：{0}")
    records, _census, summary = derive_atom_identifiability_feasibility(
        observations=layout[0], fragments=layout[1], plans=layout[2],
        official_source_by_pair=layout[3], opencc_routes={},
        morphology_by_form={"plugin": (("plugin", "N;SG"),)})
    assert summary["authorization"]["authorized_proposal_count"] == 0
    assert records[0]["reason_counts"] == {
        "SOURCE_LAYOUT_NOT_PRESERVED": 1}


def test_unimorph_parser_casefolds_and_rejects_identity_drift(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """UniMorph parser 保留 analysis ambiguity，并绑定 README/license/data。"""
    readme = b"Source: Wikipedia\nLicense: https://creativecommons.org/licenses/by-sa/3.0/\n"
    legalcode = b"license"
    data = "check\tChecking\tV;V.PTCP;PRS\nplugin\tplugins\tN;PL\n".encode()
    paths = []
    for name, payload in (("README.md", readme), ("LICENSE", legalcode),
                          ("eng", data)):
        path = tmp_path / name
        path.write_bytes(payload)
        paths.append(path)
    monkeypatch.setattr(sources, "UNIMORPH_ENGLISH_README_BYTES", len(readme))
    monkeypatch.setattr(
        sources, "UNIMORPH_ENGLISH_README_SHA256", hashlib.sha256(
            readme).hexdigest())
    monkeypatch.setattr(sources, "UNIMORPH_ENGLISH_LICENSE_BYTES", len(legalcode))
    monkeypatch.setattr(
        sources, "UNIMORPH_ENGLISH_LICENSE_SHA256", hashlib.sha256(
            legalcode).hexdigest())
    monkeypatch.setattr(sources, "UNIMORPH_ENGLISH_DATA_BYTES", len(data))
    monkeypatch.setattr(
        sources, "UNIMORPH_ENGLISH_DATA_SHA256", hashlib.sha256(
            data).hexdigest())
    index, census = sources.parse_unimorph_english(
        data_path=paths[2], readme_path=paths[0], license_path=paths[1])
    assert index["checking"] == (("check", "V;V.PTCP;PRS"),)
    assert census["form_casefold_count"] == 1
    assert sources.unimorph_segment_facts("Checking", index)[0].startswith(
        "UNIMORPH_MARKED:")

    paths[2].write_bytes(data + b"x\tx\tN;SG\n")
    with pytest.raises(Exception, match="identity"):
        sources.parse_unimorph_english(
            data_path=paths[2], readme_path=paths[0], license_path=paths[1])


def _fake_audit_outputs() -> tuple[
        dict[str, tuple[dict[str, object], ...]], dict[str, object]]:
    """构造不含 source/target surface 的最小审计输出。"""
    outputs = {
        "proposal-audit.jsonl": ({
            "proposal_id": "1" * 64,
            "record_kind": "PROPOSAL",
            "surface_published": 0,
        },),
        "census.jsonl": ({
            "census_id": "2" * 64,
            "record_kind": "CENSUS",
        },),
        "source-census.jsonl": ({
            "record_kind": "SOURCE_CENSUS",
            "surface_published": 0,
        },),
    }
    return outputs, {
        "audit_outcome": (
            "ATOM_IDENTIFIABILITY_FACILITY_PASS_"
            "TRAIN_FEASIBILITY_NONZERO_EXACT_ZERO_WRONG"),
        "capability_claimed": 0,
        "identifiability": {
            "scoring": {
                "outcome_counts": {
                    "EXACT": 2, "UNKNOWN": 12, "WRONG": 0}}},
        "runtime_claimed": 0,
        "train_source_or_output_surface_published": 0,
    }


def _audit_inputs(tmp_path: Path) -> tuple[list[Path], Path, Path]:
    """创建 publisher 所需的目录输入、archive 与 target。"""
    directories = [tmp_path / name for name in (
        "protocol", "variable", "semantic", "godot", "libreoffice",
        "vscode", "thunderbird", "vscode-source", "parser", "opencc",
        "unimorph")]
    for path in directories:
        path.mkdir()
    archive = tmp_path / "vscode-source.zip"
    archive.write_bytes(b"archive")
    return directories, archive, tmp_path / "audit"


def _publish_audit(
        tmp_path: Path,
        directories: list[Path],
        archive: Path,
        target: Path,
        ) -> dict[str, object]:
    """发布 synthetic atom-identifiability artifact。"""
    return audit.publish_normalization_recovery_v7_atom_identifiability_audit(
        run_root=tmp_path,
        training_protocol_dir=directories[0],
        variable_structure_audit_dir=directories[1],
        neutral_semantic_source_audit_dir=directories[2],
        godot_source_pack_dir=directories[3],
        libreoffice_source_pack_dir=directories[4],
        vscode_source_pack_dir=directories[5],
        thunderbird_source_pack_dir=directories[6],
        vscode_source_archive_path=archive,
        vscode_source_root=directories[7],
        typescript_parser_root=directories[8],
        opencc_source_pack_dir=directories[9],
        unimorph_english_dir=directories[10],
        target_dir=target,
    )


def _read_audit(
        directories: list[Path],
        archive: Path,
        target: Path,
        manifest_sha256: str,
        ) -> tuple[dict[str, object], dict[str, tuple[dict[str, object], ...]]]:
    """严格回读 synthetic atom-identifiability artifact。"""
    return audit.read_normalization_recovery_v7_atom_identifiability_audit(
        target,
        training_protocol_dir=directories[0],
        variable_structure_audit_dir=directories[1],
        neutral_semantic_source_audit_dir=directories[2],
        godot_source_pack_dir=directories[3],
        libreoffice_source_pack_dir=directories[4],
        vscode_source_pack_dir=directories[5],
        thunderbird_source_pack_dir=directories[6],
        vscode_source_archive_path=archive,
        vscode_source_root=directories[7],
        typescript_parser_root=directories[8],
        opencc_source_pack_dir=directories[9],
        unimorph_english_dir=directories[10],
        expected_manifest_sha256=manifest_sha256,
    )


def test_atom_identifiability_audit_round_trip_nonoverwrite_and_tamper(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """artifact 往返一致，并拒绝覆盖及同步篡改。"""
    fake_state = {"unimorph_git": {
        "commit": sources.UNIMORPH_ENGLISH_COMMIT,
        "remote": sources.UNIMORPH_ENGLISH_REPOSITORY,
        "tree": sources.UNIMORPH_ENGLISH_TREE,
    }}
    monkeypatch.setattr(audit, "_require_k_root", lambda value: Path(value))
    monkeypatch.setattr(audit, "_input_state", lambda **_kwargs: fake_state)
    monkeypatch.setattr(audit, "_derive", lambda _state: _fake_audit_outputs())
    directories, archive, target = _audit_inputs(tmp_path)
    published = _publish_audit(tmp_path, directories, archive, target)
    manifest, outputs = _read_audit(
        directories, archive, target, str(published["manifest_sha256"]))
    assert manifest == published
    assert outputs == _fake_audit_outputs()[0]
    assert manifest["status"] == (
        "TRAIN_ONLY_ATOM_IDENTIFIABILITY_FEASIBILITY_PASS_NOT_RUNTIME")
    with pytest.raises(BroadQaExternalDataError, match="input/target path 非法"):
        _publish_audit(tmp_path, directories, archive, target)

    path = target / "proposal-audit.jsonl"
    changed = canonical_json_line({
        "proposal_id": "9" * 64,
        "record_kind": "PROPOSAL",
        "surface_published": 0,
    })
    path.write_bytes(changed)
    with pytest.raises(BroadQaExternalDataError, match="records/inputs 漂移"):
        _read_audit(
            directories, archive, target,
            str(published["manifest_sha256"]))

    stored = json.loads((target / "manifest.json").read_bytes())
    artifact = next(item for item in stored["files"]
                    if item["relative_path"] == path.name)
    artifact["bytes"] = len(changed)
    artifact["sha256"] = hashlib.sha256(changed).hexdigest()
    encoded = canonical_json_line(stored)
    (target / "manifest.json").write_bytes(encoded)
    with pytest.raises(BroadQaExternalDataError, match="records/inputs 漂移"):
        _read_audit(
            directories, archive, target,
            hashlib.sha256(encoded).hexdigest())
