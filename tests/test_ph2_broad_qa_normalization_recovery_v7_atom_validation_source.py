"""覆盖 Audacity atom-validation source pack 与标签盲 commitment。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_recovery_v7_atom_validation_commitment
    as commitment,
)
from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_recovery_v7_atom_validation_commitment_v2
    as commitment_v2,
)
from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_recovery_v7_atom_validation_source_pack
    as source_pack,
)
from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_localization_structure import (
    git_blob_sha1,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_atom_identifiability_audit import (
    NORMALIZATION_RECOVERY_V7_ATOM_IDENTIFIABILITY_AUDIT_KIND,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_atom_validation_source_records import (
    AUDACITY_LICENSE_PATH,
    AUDACITY_SOURCE_PATHS,
    parse_audacity_atom_validation_files,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_evaluation_commitment import (
    NORMALIZATION_RECOVERY_V7_EVALUATION_COMMITMENT_KIND,
    NORMALIZATION_RECOVERY_V7_EVALUATION_COMMITMENT_STATUS,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    canonical_json_line,
)


def _license() -> bytes:
    """构造满足固定 project/default-file 声明的 synthetic license。"""
    return (
        b"Audacity is released under the GNU General Public License version 3 "
        b"(GPLv3).\n"
        b"Individual files are available under GPL version 2 (GPLv2) or "
        b"(at your option) any later version.\n"
        b"GNU GENERAL PUBLIC LICENSE\n                       Version 3\n")


def _po(
        language: str,
        *,
        plugin: str,
        ready: str,
        empty: str,
        ) -> bytes:
    """构造含 eligible、fuzzy、plural 与 empty 的 synthetic PO。"""
    text = f'''msgid ""
msgstr ""
"Language: {language}\\n"
"Content-Type: text/plain; charset=UTF-8\\n"

#: menu.cpp:1
msgctxt "menu"
msgid "Plugin"
msgstr "{plugin}"

#: status.cpp:1
msgctxt "status"
msgid "Ready"
msgstr "{ready}"

#: fuzzy.cpp:1
#, fuzzy
msgctxt "status"
msgid "Working"
msgstr "工作中"

#: plural.cpp:1
msgctxt "count"
msgid "file"
msgid_plural "files"
msgstr[0] "文件"

#: empty.cpp:1
msgctxt "empty"
msgid "Missing"
msgstr "{empty}"
'''
    return text.encode("utf-8")


def _files() -> dict[str, bytes]:
    """返回固定三文件 synthetic source material。"""
    return {
        AUDACITY_LICENSE_PATH: _license(),
        AUDACITY_SOURCE_PATHS["zh_Hans"]: _po(
            "zh_CN", plugin="插件", ready="就绪", empty=""),
        AUDACITY_SOURCE_PATHS["zh_Hant"]: _po(
            "zh_TW", plugin="外掛程式", ready="就绪", empty="缺失"),
    }


def _git_identity() -> dict[str, object]:
    """返回 source-pack manifest 所需的固定 Git identity。"""
    return {
        "branch": source_pack.AUDACITY_BRANCH,
        "commit": source_pack.AUDACITY_COMMIT,
        "commit_date": source_pack.AUDACITY_COMMIT_DATE,
        "remote": source_pack.AUDACITY_REPOSITORY_URL,
        "root_tree": source_pack.AUDACITY_ROOT_TREE,
    }


def _patch_source_material(
        monkeypatch: pytest.MonkeyPatch,
        files: dict[str, bytes],
        ) -> None:
    """用 synthetic blobs 隔离 publisher test 与真实 K checkout。"""
    monkeypatch.setattr(
        source_pack, "AUDACITY_SOURCE_BLOBS",
        {path: git_blob_sha1(payload) for path, payload in files.items()})
    monkeypatch.setattr(
        source_pack, "AUDACITY_LICENSE_BYTES", len(files[AUDACITY_LICENSE_PATH]))
    monkeypatch.setattr(
        source_pack, "AUDACITY_LICENSE_SHA256",
        hashlib.sha256(files[AUDACITY_LICENSE_PATH]).hexdigest())
    monkeypatch.setattr(
        source_pack, "_require_k_root", lambda value: Path(value))
    monkeypatch.setattr(
        source_pack, "_git_source_material",
        lambda _root: (_git_identity(), files))


def _publish_source(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> tuple[Path, dict[str, object], dict[str, bytes]]:
    """发布一份 synthetic Audacity source pack。"""
    files = _files()
    _patch_source_material(monkeypatch, files)
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    target = tmp_path / "source-pack"
    manifest = source_pack.publish_audacity_atom_validation_source_pack(
        run_root=tmp_path,
        checkout_root=checkout,
        target_dir=target,
    )
    return target, manifest, files


def _write_manifest(path: Path, value: dict[str, object]) -> str:
    """写入一份规范 synthetic manifest 并返回 SHA。"""
    path.mkdir()
    encoded = canonical_json_line(value)
    (path / "manifest.json").write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def _atom_manifest() -> dict[str, object]:
    """构造 section 90 lower-bound 的最小 manifest。"""
    return {
        "artifact_kind": NORMALIZATION_RECOVERY_V7_ATOM_IDENTIFIABILITY_AUDIT_KIND,
        "candidate_family_formal_run_count": 0,
        "mastery_claimed": 0,
        "production_enabled": 0,
        "runtime_program_published": 0,
        "status": (
            "TRAIN_ONLY_ATOM_IDENTIFIABILITY_FEASIBILITY_PASS_NOT_RUNTIME"),
        "summary": {
            "identifiability": {
                "scoring": {
                    "outcome_counts": {
                        "EXACT": 2, "UNKNOWN": 12, "WRONG": 0}}}},
        "teacher_api_llm_call_count": 0,
        "train_source_or_output_surface_published": 0,
    }


def _vlc_commitment() -> dict[str, object]:
    """构造未消费 VLC final commitment 的最小 manifest。"""
    return {
        "artifact_kind": NORMALIZATION_RECOVERY_V7_EVALUATION_COMMITMENT_KIND,
        "candidate_or_code_read_count": 0,
        "denominator": {"record_count": 3656},
        "formal_contract": {"formal_run_count_max": 1},
        "mastery_claimed": 0,
        "production_enabled": 0,
        "source_non_manifest_file_read_count": 0,
        "status": NORMALIZATION_RECOVERY_V7_EVALUATION_COMMITMENT_STATUS,
        "teacher_api_llm_call_count": 0,
        "training_source_read_count": 0,
    }


def test_audacity_parser_freezes_all_common_eligible_without_surfaces() -> None:
    """全量 selection 得到两条 pair，并让 identity 可脱离翻译 surface。"""
    source_records, pairs, summary = parse_audacity_atom_validation_files(
        _files())
    assert len(source_records) == 3
    assert len(pairs) == 2
    assert summary["plain_pair_count"] == 2
    assert summary["excluded_common_pair_counts"] == {
        "any": 3, "empty": 2, "fuzzy": 1, "obsolete": 0, "plural": 1}
    identities = tuple({
        "pair_id": item["pair_id"],
        "source_identity": item["source_identity"],
    } for item in pairs)
    encoded = canonical_json_bytes(identities)
    assert "外掛程式".encode("utf-8") not in encoded
    assert "插件".encode("utf-8") not in encoded


def test_source_pack_round_trip_nonoverwrite_and_frozen_blob_tamper(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """source pack 往返一致、拒绝覆盖并拒绝 raw blob 篡改。"""
    target, published, _raw = _publish_source(tmp_path, monkeypatch)
    manifest, source_records, inventory = (
        source_pack.read_audacity_atom_validation_source_pack(
            target,
            expected_manifest_sha256=str(published["manifest_sha256"]),
        ))
    assert manifest == published
    assert len(source_records) == 3
    assert len(inventory) == 2
    assert manifest["validation_state"]["validation_run_count"] == 0
    assert manifest["validation_state"][
        "individual_translation_surface_published_in_jsonl"] == 0
    with pytest.raises(BroadQaExternalDataError, match="input/target path 非法"):
        source_pack.publish_audacity_atom_validation_source_pack(
            run_root=tmp_path,
            checkout_root=tmp_path / "checkout",
            target_dir=target,
        )

    raw_path = target / AUDACITY_SOURCE_PATHS["zh_Hans"]
    raw_path.write_bytes(raw_path.read_bytes() + b"\n")
    with pytest.raises(BroadQaExternalDataError, match="frozen raw blob 漂移"):
        source_pack.read_audacity_atom_validation_source_pack(
            target,
            expected_manifest_sha256=str(published["manifest_sha256"]),
        )


def test_held_input_and_label_readers_are_physically_separated(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """proposal reader 不开 zh-CN；label materializer 不重开 zh-TW。"""
    target, published, _raw = _publish_source(tmp_path, monkeypatch)
    original = source_pack._committed_payload
    reads = []

    def tracked(root, manifest, *, relative_path):
        reads.append(relative_path)
        return original(root, manifest, relative_path=relative_path)

    monkeypatch.setattr(source_pack, "_committed_payload", tracked)
    manifest, held_inputs, input_reads = (
        source_pack.read_audacity_atom_validation_held_inputs_after_family_freeze(
            target,
            expected_manifest_sha256=str(published["manifest_sha256"]),
        ))
    assert manifest["manifest_sha256"] == published["manifest_sha256"]
    assert len(held_inputs) == 2
    assert input_reads["zh_hans_translation_file_read_count"] == 0
    assert AUDACITY_SOURCE_PATHS["zh_Hans"] not in reads
    assert all("output_text" not in item and "zh_hans" not in item
               for item in held_inputs)

    reads.clear()
    labels, label_reads = (
        source_pack.materialize_audacity_atom_validation_labels_after_authorization_freeze(
            target,
            expected_manifest_sha256=str(published["manifest_sha256"]),
            held_inputs=held_inputs,
        ))
    assert len(labels) == 2
    assert label_reads["zh_hans_translation_file_read_count"] == 1
    assert AUDACITY_SOURCE_PATHS["zh_Hant"] not in reads


def test_commitment_manifest_only_round_trip_and_tamper(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """commitment 只绑定三个 manifest，并拒绝同步字段篡改。"""
    source_dir, source_manifest, _raw = _publish_source(
        tmp_path, monkeypatch)
    atom_dir = tmp_path / "atom"
    atom_sha = _write_manifest(atom_dir, _atom_manifest())
    vlc_dir = tmp_path / "vlc"
    vlc_sha = _write_manifest(vlc_dir, _vlc_commitment())
    commitment_dir = tmp_path / "commitment"
    monkeypatch.setattr(
        commitment, "_require_k_root", lambda value: Path(value))
    with pytest.raises(BroadQaExternalDataError, match="commitment path 非法"):
        commitment.publish_audacity_atom_validation_commitment(
            run_root=tmp_path,
            source_pack_dir=source_dir,
            expected_source_manifest_sha256=str(
                source_manifest["manifest_sha256"]),
            atom_audit_dir=atom_dir,
            expected_atom_manifest_sha256=atom_sha,
            vlc_commitment_dir=vlc_dir,
            expected_vlc_commitment_manifest_sha256=vlc_sha,
            target_dir=source_dir / "nested-commitment",
        )
    published = commitment.publish_audacity_atom_validation_commitment(
        run_root=tmp_path,
        source_pack_dir=source_dir,
        expected_source_manifest_sha256=str(
            source_manifest["manifest_sha256"]),
        atom_audit_dir=atom_dir,
        expected_atom_manifest_sha256=atom_sha,
        vlc_commitment_dir=vlc_dir,
        expected_vlc_commitment_manifest_sha256=vlc_sha,
        target_dir=commitment_dir,
    )
    restored = commitment.read_audacity_atom_validation_commitment(
        commitment_dir,
        source_pack_dir=source_dir,
        expected_source_manifest_sha256=str(
            source_manifest["manifest_sha256"]),
        atom_audit_dir=atom_dir,
        expected_atom_manifest_sha256=atom_sha,
        vlc_commitment_dir=vlc_dir,
        expected_vlc_commitment_manifest_sha256=vlc_sha,
        expected_manifest_sha256=str(published["manifest_sha256"]),
    )
    assert restored == published
    assert restored["gates"]["exact_output_count_min"] == 1
    assert restored["gates"]["wrong_output_count_max"] == 0
    assert restored["validation_reads"] == {
        "atom_non_manifest_file_read_count": 0,
        "audacity_identity_raw_or_translation_read_count": 0,
        "source_manifest_read_count": 1,
        "vlc_commitment_manifest_read_count": 1,
        "vlc_identity_raw_or_translation_read_count": 0,
    }

    commitment_v2_dir = tmp_path / "commitment-v2"
    monkeypatch.setattr(
        commitment_v2, "_require_k_root", lambda value: Path(value))
    published_v2 = (
        commitment_v2.publish_audacity_atom_validation_commitment_v2(
            run_root=tmp_path,
            v1_commitment_dir=commitment_dir,
            expected_v1_manifest_sha256=str(published["manifest_sha256"]),
            target_dir=commitment_v2_dir,
        ))
    restored_v2 = commitment_v2.read_audacity_atom_validation_commitment_v2(
        commitment_v2_dir,
        v1_commitment_dir=commitment_dir,
        expected_v1_manifest_sha256=str(published["manifest_sha256"]),
        expected_manifest_sha256=str(published_v2["manifest_sha256"]),
    )
    assert restored_v2 == published_v2
    assert "exact_output_count_min" not in restored_v2["gates"]
    assert restored_v2["gates"][
        "authorized_changed_exact_output_count_min"] == 1
    assert restored_v2["validation_contract"][
        "identity_only_exact_satisfies_transfer_pass"] == 0
    assert restored_v2["revision"][
        "label_or_individual_record_read_count"] == 0

    path = commitment_dir / "manifest.json"
    stored = json.loads(path.read_bytes())
    stored["production_enabled"] = 1
    encoded = canonical_json_line(stored)
    path.write_bytes(encoded)
    with pytest.raises(BroadQaExternalDataError, match="fields 漂移"):
        commitment.read_audacity_atom_validation_commitment(
            commitment_dir,
            source_pack_dir=source_dir,
            expected_source_manifest_sha256=str(
                source_manifest["manifest_sha256"]),
            atom_audit_dir=atom_dir,
            expected_atom_manifest_sha256=atom_sha,
            vlc_commitment_dir=vlc_dir,
            expected_vlc_commitment_manifest_sha256=vlc_sha,
            expected_manifest_sha256=hashlib.sha256(encoded).hexdigest(),
        )
