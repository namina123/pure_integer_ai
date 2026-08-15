"""recovery-v7 neutral semantic source feasibility 专项测试。"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZipInfo

import pytest

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_recovery_v7_neutral_semantic_source_audit
    as audit,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_neutral_semantic_source_records import (
    OEWN_SOURCE_ID,
    PROPBANK_SOURCE_ID,
    SUPPORT_PROPBANK_MODAL_CUE,
    SUPPORT_PROPBANK_NEGATION_CUE,
    SUPPORT_TWO_SOURCE_ACTION_STATE,
    derive_neutral_semantic_source_feasibility,
    normalize_optional_semantic_source_text,
    parse_open_english_wordnet,
    parse_propbank_frames,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_neutral_source_projection_records import (
    GODOT_SOURCE_FAMILY,
    LIBREOFFICE_SOURCE_FAMILY,
    THUNDERBIRD_SOURCE_FAMILY,
    VSCODE_SOURCE_FAMILY,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
)


def _sha256(payload: bytes) -> str:
    """返回 synthetic source 或记录 identity。"""
    return hashlib.sha256(payload).hexdigest()


def _git_blob_sha1(payload: bytes) -> str:
    """返回 synthetic license Git blob identity。"""
    return hashlib.sha1(
        b"blob " + str(len(payload)).encode("ascii")
        + b"\x00" + payload).hexdigest()


def _oewn_payload(*, version: str = "2025", duplicate: bool = False) -> bytes:
    """构造含 action/state senses 的最小 OEWN LMF。"""
    duplicate_entry = (
        '<LexicalEntry id="entry-open"><Lemma writtenForm="open" '
        'partOfSpeech="v"/><Sense id="sense-open-2" '
        'synset="synset-open"/></LexicalEntry>' if duplicate else "")
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<LexicalResource><Lexicon id="oewn" language="en" '
        'license="https://creativecommons.org/licenses/by/4.0" '
        f'version="{version}" '
        'url="https://github.com/globalwordnet/english-wordnet">'
        '<LexicalEntry id="entry-open"><Lemma writtenForm="open" '
        'partOfSpeech="v"/><Sense id="sense-open" '
        'synset="synset-open"/></LexicalEntry>'
        '<LexicalEntry id="entry-still"><Lemma writtenForm="still" '
        'partOfSpeech="n"/><Sense id="sense-still" '
        'synset="synset-still"/></LexicalEntry>'
        f'{duplicate_entry}'
        '<Synset id="synset-open" lexfile="noun.act" partOfSpeech="n"/>'
        '<Synset id="synset-still" lexfile="noun.state" partOfSpeech="n"/>'
        '</Lexicon></LexicalResource>').encode("utf-8")


def _write_oewn(
        path: Path,
        *,
        version: str = "2025",
        duplicate: bool = False,
        ) -> Path:
    """写入 deterministic synthetic OEWN gzip。"""
    path.write_bytes(gzip.compress(
        _oewn_payload(version=version, duplicate=duplicate), mtime=0))
    return path


def _propbank_frame(
        *,
        roleset_id: str = "open.01",
        predicate: str = "open",
        include_empty: bool = True,
        ) -> bytes:
    """构造 predicate、role inventory、cross-link 与 MOD/NEG cue。"""
    empty = (
        '<alias pos="v"></alias>'
        if include_empty else "")
    empty_arg = '<arg type="ARGM-MOD"></arg>' if include_empty else ""
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<frameset>'
        f'<predicate lemma="{predicate}">'
        f'<roleset id="{roleset_id}" name="make accessible">'
        '<aliases><alias pos="v">open</alias>'
        f'{empty}</aliases>'
        '<roles><role n="0" f="PAG" descr="opener">'
        '<rolelinks><rolelink resource="VerbNet" class="open-45.1"/>'
        '</rolelinks></role>'
        '<role n="1" f="PPT" descr="thing opened"/></roles>'
        '<lexlinks><lexlink resource="FrameNet" class="Opening"/></lexlinks>'
        '<example><text>It can not open</text><propbank>'
        '<arg type="ARGM-MOD">can</arg>'
        '<arg type="ARGM-NEG">not</arg>'
        f'{empty_arg}'
        '</propbank></example>'
        '</roleset></predicate></frameset>').encode("utf-8")


def _write_propbank(
        path: Path,
        *,
        duplicate_roleset: bool = False,
        malformed_xml: bool = False,
        traversal: bool = False,
        ) -> tuple[Path, bytes]:
    """写入受控 PropBank selection ZIP。"""
    license_payload = b"synthetic CC-BY-SA-4.0 license\n"
    with ZipFile(path, "w") as archive:
        archive.writestr("LICENSE", license_payload)
        archive.writestr("README.md", b"synthetic\n")
        archive.writestr("frames/", b"")
        archive.writestr("frames/frameset.dtd", b"<!ELEMENT frameset ANY>\n")
        archive.writestr("frames/open.xml", _propbank_frame())
        if duplicate_roleset:
            archive.writestr(
                "frames/duplicate.xml",
                _propbank_frame(
                    roleset_id="open.01", predicate="unlock",
                    include_empty=False))
        if malformed_xml:
            archive.writestr("frames/broken.xml", b"<frameset><broken></frameset>")
        if traversal:
            archive.writestr("../escape.xml", b"<frameset/>")
    return path, license_payload


def test_optional_normalizer_skips_empty_schema_fields() -> None:
    """空 alias/arg 明确跳过，非空文本仍使用严格 neutral unit 合同。"""
    assert normalize_optional_semantic_source_text("") == ""
    assert normalize_optional_semantic_source_text("   ") == ""
    assert normalize_optional_semantic_source_text(None) == ""
    assert normalize_optional_semantic_source_text("OpenFile ٣") == (
        "open file ٣")


def test_oewn_parser_action_state_and_fail_closed_inputs(tmp_path: Path) -> None:
    """OEWN 保留 action/state lexfile，拒绝坏 gzip、重复 id 与版本漂移。"""
    path = _write_oewn(tmp_path / "oewn.xml.gz")
    indexes, census = parse_open_english_wordnet(path)
    assert indexes["action"] == frozenset(("open",))
    assert indexes["state"] == frozenset(("still",))
    assert census["lexical_entry_count"] == 2
    assert census["sense_count"] == 2
    assert census["parse_anomaly_count"] == 0

    bad = tmp_path / "bad.xml.gz"
    bad.write_bytes(b"not-gzip")
    with pytest.raises(BroadQaExternalDataError, match="gzip/XML 非法"):
        parse_open_english_wordnet(bad)

    duplicate = _write_oewn(
        tmp_path / "duplicate.xml.gz", duplicate=True)
    with pytest.raises(BroadQaExternalDataError, match="LexicalEntry id"):
        parse_open_english_wordnet(duplicate)

    drift = _write_oewn(tmp_path / "drift.xml.gz", version="2024")
    with pytest.raises(BroadQaExternalDataError, match="source identity 漂移"):
        parse_open_english_wordnet(drift)


def test_propbank_parser_preserves_anomalies_inventory_and_cues(
        tmp_path: Path,
        ) -> None:
    """坏 XML 与重复 roleset 可审计，空字段不污染 role/cue 索引。"""
    path, license_payload = _write_propbank(
        tmp_path / "propbank.zip",
        duplicate_roleset=True,
        malformed_xml=True,
    )
    indexes, census = parse_propbank_frames(
        path,
        expected_license_sha256=_sha256(license_payload),
        expected_license_git_blob_sha1=_git_blob_sha1(license_payload),
    )
    assert "open" in indexes["predicate"]
    assert "open" in indexes["role_inventory"]
    assert indexes["modal_cue"] == frozenset(("can",))
    assert indexes["negation_cue"] == frozenset(("not",))
    assert census["malformed_xml_file_count"] == 1
    assert census["duplicate_roleset_id_count"] == 1
    assert census["empty_alias_count"] == 1
    assert census["empty_argument_text_count"] == 1
    assert census["role_count"] == 4
    assert census["cross_link_counts"] == {
        "FRAMENET": 2, "VERBNET": 2}

    with pytest.raises(BroadQaExternalDataError, match="license identity 漂移"):
        parse_propbank_frames(
            path,
            expected_license_sha256="0" * 64,
            expected_license_git_blob_sha1=_git_blob_sha1(license_payload),
        )


def test_propbank_zip_rejects_path_escape_and_symlink(tmp_path: Path) -> None:
    """selection ZIP 拒绝目录穿越与 symlink member。"""
    path, license_payload = _write_propbank(
        tmp_path / "escape.zip", traversal=True)
    with pytest.raises(BroadQaExternalDataError, match="ZIP member 非法"):
        parse_propbank_frames(
            path,
            expected_license_sha256=_sha256(license_payload),
            expected_license_git_blob_sha1=_git_blob_sha1(license_payload),
        )

    symlink = tmp_path / "symlink.zip"
    with ZipFile(symlink, "w") as archive:
        archive.writestr("LICENSE", license_payload)
        archive.writestr("README.md", b"synthetic\n")
        info = ZipInfo("frames/open.xml")
        info.create_system = 3
        info.external_attr = (0o120777 << 16)
        archive.writestr(info, b"target")
    with pytest.raises(BroadQaExternalDataError, match="ZIP member 非法"):
        parse_propbank_frames(
            symlink,
            expected_license_sha256=_sha256(license_payload),
            expected_license_git_blob_sha1=_git_blob_sha1(license_payload),
        )


def _row(family: str, pair_id: str, surface: str) -> dict[str, object]:
    """构造只含 source-side transient surface 的 neutral row。"""
    return {
        "_neutral_surface": surface,
        "pair_id": pair_id,
        "source_family": family,
    }


def test_feasibility_separates_lexical_role_inventory_and_assignment(
        tmp_path: Path,
        ) -> None:
    """两源可形成非零 evidence，但 sense/placeholder assignment 保持为零。"""
    oewn = _write_oewn(tmp_path / "oewn.xml.gz")
    propbank, license_payload = _write_propbank(tmp_path / "propbank.zip")
    godot_pair = "1" * 64
    libre_pair = "2" * 64
    vscode_pair = "3" * 64
    rows = {
        GODOT_SOURCE_FAMILY: (
            _row(GODOT_SOURCE_FAMILY, godot_pair, "can not open"),),
        LIBREOFFICE_SOURCE_FAMILY: (
            _row(LIBREOFFICE_SOURCE_FAMILY, libre_pair, "open file"),),
        VSCODE_SOURCE_FAMILY: (
            _row(VSCODE_SOURCE_FAMILY, vscode_pair, "can open"),),
        THUNDERBIRD_SOURCE_FAMILY: (),
    }
    proposals = ({
        "held_out_source_family": GODOT_SOURCE_FAMILY,
        "pre_authorization_outcome": "EXACT",
        "source_pair_id": godot_pair,
    },)
    source_identities = {
        OEWN_SOURCE_ID: {"raw_sha256": "a" * 64},
        PROPBANK_SOURCE_ID: {"archive_sha256": "b" * 64},
    }
    candidate, census, family, proposal, fact, summary = (
        derive_neutral_semantic_source_feasibility(
            oewn_source_path=oewn,
            propbank_source_path=propbank,
            propbank_license_sha256=_sha256(license_payload),
            propbank_license_git_blob_sha1=(
                _git_blob_sha1(license_payload)),
            rows_by_family=rows,
            proposals=proposals,
            source_identities=source_identities,
        ))
    assert len(candidate) == 6
    assert len(census) == 2
    assert len(family) == 4
    assert len(proposal) == 4
    assert summary["cross_family_matched_phrase_counts"][
        SUPPORT_TWO_SOURCE_ACTION_STATE] == 1
    assert summary["cross_family_matched_phrase_counts"][
        SUPPORT_PROPBANK_MODAL_CUE] == 1
    assert summary["cross_family_matched_phrase_counts"][
        SUPPORT_PROPBANK_NEGATION_CUE] == 0
    assert summary["lexical_match_assigns_semantic_sense"] == 0
    assert summary["placeholder_role_assignment_count"] == 0
    by_fact = {item["fact_family"]: item for item in fact}
    assert by_fact["ARGUMENT_SEMANTIC_ROLE_INVENTORY"]["outcome"] == (
        "AVAILABLE_INVENTORY_ONLY_NOT_PLACEHOLDER_ASSIGNMENT")
    assert by_fact["PLACEHOLDER_ROLE_ASSIGNMENT"]["outcome"] == (
        "NE_NOT_PRESENT")
    assert by_fact["LEXICAL_SENSE_ASSIGNMENT"]["outcome"] == (
        "NE_NOT_PRESENT")


def _fake_outputs() -> tuple[
        dict[str, tuple[dict[str, object], ...]], dict[str, object]]:
    """构造 publisher/reader 的小型稳定 aggregate。"""
    return {
        "source-candidates.jsonl": ({
            "record_id": "1" * 64, "record_kind": "CANDIDATE"},),
        "source-census.jsonl": ({
            "record_id": "2" * 64, "record_kind": "SOURCE"},),
        "family-coverage.jsonl": ({
            "record_id": "3" * 64, "record_kind": "FAMILY"},),
        "proposal-coverage.jsonl": ({
            "record_id": "4" * 64, "record_kind": "PROPOSAL"},),
        "fact-families.jsonl": ({
            "record_id": "5" * 64, "record_kind": "FACT"},),
    }, {
        "audit_outcome": "SOURCE_FEASIBILITY_PASS_CAPABILITY_NE",
        "semantic_source": {
            "capability_outcome": "NE_SOURCE_FEASIBILITY_NOT_AUTHORIZATION"},
        "source_surface_published": 0,
    }


def _patch_audit(monkeypatch: pytest.MonkeyPatch) -> None:
    """隔离 audit 往返测试与真实 K 盘 source/predecessor。"""
    monkeypatch.setattr(audit, "_require_k_root", lambda value: Path(value))
    monkeypatch.setattr(
        audit, "_input_state", lambda **_kwargs: ({}, {}, {}, {}, {}))
    monkeypatch.setattr(audit, "_derive", lambda **_kwargs: _fake_outputs())


def _audit_inputs(tmp_path: Path) -> tuple[list[Path], list[Path]]:
    """创建七个目录输入和三份 source 文件。"""
    directories = [tmp_path / name for name in (
        "protocol", "variable", "alignment", "godot", "libreoffice",
        "vscode", "thunderbird")]
    for path in directories:
        path.mkdir()
    files = [tmp_path / name for name in (
        "oewn.gz", "oewn-license.md", "propbank.zip")]
    for index, path in enumerate(files):
        path.write_bytes(f"source-{index}".encode("ascii"))
    return directories, files


def _publish_audit(
        tmp_path: Path,
        directories: list[Path],
        files: list[Path],
        ) -> tuple[Path, dict[str, object]]:
    """发布 synthetic source feasibility artifact。"""
    target = tmp_path / "audit"
    manifest = audit.publish_normalization_recovery_v7_neutral_semantic_source_audit(
        run_root=tmp_path,
        training_protocol_dir=directories[0],
        variable_structure_audit_dir=directories[1],
        intent_semantic_alignment_dir=directories[2],
        godot_source_pack_dir=directories[3],
        libreoffice_source_pack_dir=directories[4],
        vscode_source_pack_dir=directories[5],
        thunderbird_source_pack_dir=directories[6],
        oewn_source_path=files[0],
        oewn_license_path=files[1],
        propbank_source_path=files[2],
        target_dir=target,
    )
    return target, manifest


def _read_audit(
        target: Path,
        directories: list[Path],
        files: list[Path],
        manifest_sha256: str,
        ) -> tuple[
            dict[str, object],
            dict[str, tuple[dict[str, object], ...]],
        ]:
    """严格回读 synthetic source feasibility artifact。"""
    return audit.read_normalization_recovery_v7_neutral_semantic_source_audit(
        target,
        training_protocol_dir=directories[0],
        variable_structure_audit_dir=directories[1],
        intent_semantic_alignment_dir=directories[2],
        godot_source_pack_dir=directories[3],
        libreoffice_source_pack_dir=directories[4],
        vscode_source_pack_dir=directories[5],
        thunderbird_source_pack_dir=directories[6],
        oewn_source_path=files[0],
        oewn_license_path=files[1],
        propbank_source_path=files[2],
        expected_manifest_sha256=manifest_sha256,
    )


def test_audit_round_trip_nonoverwrite_and_synchronized_tamper(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """audit 往返一致、拒绝覆盖，并拒绝 records+manifest 同步篡改。"""
    _patch_audit(monkeypatch)
    directories, files = _audit_inputs(tmp_path)
    target, published = _publish_audit(tmp_path, directories, files)
    manifest, outputs = _read_audit(
        target, directories, files, str(published["manifest_sha256"]))
    assert manifest == published
    assert outputs == _fake_outputs()[0]
    assert manifest["status"] == (
        "TRAIN_ONLY_NEUTRAL_SEMANTIC_SOURCE_FEASIBILITY_PASS_NOT_RUNTIME")
    with pytest.raises(BroadQaExternalDataError, match="input/target path 非法"):
        _publish_audit(tmp_path, directories, files)

    path = target / "proposal-coverage.jsonl"
    path.write_bytes(canonical_json_line({
        "record_id": "9" * 64, "record_kind": "PROPOSAL"}))
    with pytest.raises(BroadQaExternalDataError, match="records/inputs 漂移"):
        _read_audit(
            target, directories, files, str(published["manifest_sha256"]))

    stored = json.loads((target / "manifest.json").read_bytes())
    payload = path.read_bytes()
    artifact = next(
        item for item in stored["files"]
        if item["relative_path"] == "proposal-coverage.jsonl")
    artifact["bytes"] = len(payload)
    artifact["sha256"] = _sha256(payload)
    encoded = canonical_json_line(stored)
    (target / "manifest.json").write_bytes(encoded)
    with pytest.raises(BroadQaExternalDataError, match="records/inputs 漂移"):
        _read_audit(
            target, directories, files, _sha256(encoded))
