"""D-02 闸后 UD Chinese GSDSimp r2.18 snapshot 与 parser T0。"""
from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_dataset_contract import CanonicalJsonObject
from pure_integer_ai.experiments.ph2_git_snapshot import (
    GitSnapshotError,
    GitSourceSnapshotManifest,
    GitTrackedFileManifest,
    git_blob_sha1_path,
    read_git_snapshot_manifest,
    verify_git_snapshot,
    write_git_snapshot_manifest,
)
from pure_integer_ai.experiments.ph2_ud_gsdsimp_adapter import (
    COMMIT_SHA1,
    NODE_EMPTY,
    NODE_RANGE,
    NODE_WORD,
    ConlluSentence,
    UdGsdsimpAdapterError,
    parse_conllu_node_id,
    parse_conllu_row,
    scan_ud_conllu,
)


SAMPLE_PATH = Path(
    "data/ph2/ud_zh_gsdsimp_r2_18_dev_s2_v1.conllu.sample")
MANIFEST_PATH = Path(
    "data/ph2/manifests/ud_zh_gsdsimp_r2_18.git_snapshot.json")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row(
        node_id: str,
        *,
        form: str = "甲",
        lemma: str = "甲",
        upos: str = "NOUN",
        xpos: str = "NN",
        feats: str = "_",
        head: str = "0",
        deprel: str = "root",
        deps: str = "_",
        misc: str = "_",
        ):
    return parse_conllu_row("\t".join((
        node_id, form, lemma, upos, xpos, feats,
        head, deprel, deps, misc,
    )))


def _sentence(*rows) -> ConlluSentence:
    return ConlluSentence(
        "synthetic-s1",
        "甲乙",
        (("sent_id", "synthetic-s1"), ("text", "甲乙")),
        tuple(rows),
        1,
        len(rows) + 2,
    )


def _tracked(path: Path, relative_path: str, *, kind: str, split: str):
    return GitTrackedFileManifest(
        relative_path,
        git_blob_sha1_path(path),
        _sha256(path),
        path.stat().st_size,
        kind,
        split,
        1 if split else 0,
        0,
        _sha256(path),
    )


def test_node_ids_are_strict_integer_tuples_and_never_float():
    """word/range/empty 身份完整保留为严格整数 tuple。"""
    assert parse_conllu_node_id("12").stable_key() == (NODE_WORD, 12, 0)
    assert parse_conllu_node_id("2-4").stable_key() == (NODE_RANGE, 2, 4)
    assert parse_conllu_node_id("7.3").stable_key() == (NODE_EMPTY, 7, 3)
    for bad in ("", "0", "01", "1.0", "1e0", "1-1", "01-2",
                "1-02", "1.2.3", "-1", "1-2.3"):
        with pytest.raises(UdGsdsimpAdapterError):
            parse_conllu_node_id(bad)
    with pytest.raises(UdGsdsimpAdapterError):
        parse_conllu_node_id(1.2)  # type: ignore[arg-type]


def test_ten_columns_feats_misc_deps_and_authority_bits():
    """十列 typed 注释完整保留，但 dependency 不冒充项目 Role。"""
    row = _row(
        "2",
        form="学生",
        lemma="学生",
        feats="Animacy=Hum|Number=Sing",
        head="1",
        deprel="nsubj:pass",
        deps="0:root|1:nsubj:pass",
        misc="SpaceAfter=No|Translit=xuéshēng",
    )
    assert row.feats == (("Animacy", "Hum"), ("Number", "Sing"))
    assert row.deps == ((0, 0, "root"), (1, 0, "nsubj:pass"))
    assert row.misc == (("SpaceAfter", "No"), ("Translit", "xuéshēng"))
    exported = row.to_dict()
    assert exported["dependency_label_authoritative"] == 0
    assert exported["project_role_authoritative"] == 0
    assert exported["head"] == 1
    with pytest.raises(UdGsdsimpAdapterError, match="十列"):
        parse_conllu_row("1\t甲")
    with pytest.raises(UdGsdsimpAdapterError, match="HEAD"):
        _row("1", head="01")


def test_range_empty_and_enhanced_references_are_closed_world():
    """range/empty 及 enhanced head 必须引用同 sentence 的实际 node。"""
    multiword = _row(
        "1-2", form="甲乙", lemma="_", upos="_", xpos="_",
        head="_", deprel="_",
    )
    word1 = _row("1", head="2", deprel="dep")
    word2 = _row("2", form="乙", lemma="乙", deps="2.1:dep")
    empty = _row(
        "2.1", form="他", lemma="他", head="_", deprel="_",
        deps="2:dep",
    )
    sentence = _sentence(multiword, word1, word2, empty)
    assert [row.node_id.stable_key() for row in sentence.rows] == [
        (NODE_RANGE, 1, 2),
        (NODE_WORD, 1, 0),
        (NODE_WORD, 2, 0),
        (NODE_EMPTY, 2, 1),
    ]
    with pytest.raises(UdGsdsimpAdapterError, match="覆盖缺失"):
        _sentence(
            _row("1-3", form="甲乙丙", lemma="_", upos="_", xpos="_",
                 head="_", deprel="_"),
            word1,
            word2,
        )
    with pytest.raises(UdGsdsimpAdapterError, match="enhanced DEPS"):
        _sentence(word1, _row("2", form="乙", lemma="乙", deps="3:dep"))
    with pytest.raises(UdGsdsimpAdapterError, match="minor"):
        _sentence(
            word1,
            word2,
            _row("2.2", head="_", deprel="_", deps="2:dep"),
        )


def test_sentence_rejects_discontinuous_words_duplicate_root_and_bad_head():
    """word 序列、basic root 与 HEAD 引用均 fail-closed。"""
    with pytest.raises(UdGsdsimpAdapterError, match="连续"):
        _sentence(_row("1"), _row("3", head="1", deprel="dep"))
    with pytest.raises(UdGsdsimpAdapterError, match="恰有一个"):
        _sentence(_row("1"), _row("2", form="乙", lemma="乙"))
    with pytest.raises(UdGsdsimpAdapterError, match="HEAD 引用"):
        _sentence(
            _row("1", head="3", deprel="dep"),
            _row("2", form="乙", lemma="乙"),
        )
    with pytest.raises(UdGsdsimpAdapterError, match="重复"):
        _sentence(_row("1"), _row("1", head="1", deprel="dep"))


def test_sample_scan_is_stable_and_malformed_block_is_atomic_anomaly(tmp_path):
    """公开小样双遍稳定；坏 sentence block 只记 anomaly。"""
    digest = _sha256(SAMPLE_PATH)
    first = scan_ud_conllu(
        SAMPLE_PATH,
        relative_path="zh_gsdsimp-ud-dev.conllu",
        split="dev",
        expected_sha256=digest,
    )
    second = scan_ud_conllu(
        SAMPLE_PATH,
        relative_path="zh_gsdsimp-ud-dev.conllu",
        split="dev",
        expected_sha256=digest,
    )
    assert first == second
    assert first.sentence_count == 1
    assert first.word_count == 12
    assert first.range_count == 0
    assert first.empty_count == 0
    assert first.anomaly_count == 0
    assert first.terminal_newline_present == 1

    damaged = tmp_path / "damaged.conllu"
    damaged.write_bytes(
        SAMPLE_PATH.read_bytes()
        + b"\n# sent_id = bad-s1\n# text = bad\n1\tbad\n")
    report = scan_ud_conllu(
        damaged,
        relative_path="damaged.conllu",
        split="train",
        expected_sha256=_sha256(damaged),
    )
    assert report.sentence_count == 1
    assert report.word_count == 12
    assert report.anomaly_count == 1
    assert report.event_sha256 != first.event_sha256


def test_scan_rejects_bad_split_hash_and_non_utf8(tmp_path):
    """split、文件身份与 strict UTF-8 任一漂移都阻止 scan。"""
    digest = _sha256(SAMPLE_PATH)
    with pytest.raises(UdGsdsimpAdapterError, match="split"):
        scan_ud_conllu(
            SAMPLE_PATH,
            relative_path="sample.conllu",
            split="test",
            expected_sha256=digest,
        )
    with pytest.raises(UdGsdsimpAdapterError, match="SHA-256"):
        scan_ud_conllu(
            SAMPLE_PATH,
            relative_path="sample.conllu",
            split="dev",
            expected_sha256="0" * 64,
        )
    invalid = tmp_path / "invalid.conllu"
    invalid.write_bytes(b"\xff\xfe")
    with pytest.raises(UdGsdsimpAdapterError, match="UTF-8"):
        scan_ud_conllu(
            invalid,
            relative_path="invalid.conllu",
            split="train",
            expected_sha256=_sha256(invalid),
        )


def test_git_snapshot_round_trip_blob_verification_and_no_overwrite(tmp_path):
    """commit/blob/local hash、许可与 parser report 共同封存且不可覆盖。"""
    repository = tmp_path / "repository"
    repository.mkdir()
    license_path = repository / "LICENSE.txt"
    license_path.write_text("CC-BY-SA-4.0\n", encoding="utf-8")
    data_path = repository / "dev.conllu"
    data_path.write_bytes(SAMPLE_PATH.read_bytes())
    manifest = GitSourceSnapshotManifest(
        1,
        "UD_ZH_GSDSIMP_R2_18",
        "https://github.com/UniversalDependencies/UD_Chinese-GSDSimp",
        "r2.18",
        COMMIT_SHA1,
        "CC-BY-SA-4.0",
        "LICENSE.txt",
        "PUBLIC",
        1,
        1,
        (
            _tracked(license_path, "LICENSE.txt", kind="license", split=""),
            _tracked(data_path, "dev.conllu", kind="conllu", split="dev"),
        ),
        CanonicalJsonObject.from_value({"anomaly_count": 0}),
    )
    output = tmp_path / "snapshot.json"
    write_git_snapshot_manifest(manifest, output)
    restored = read_git_snapshot_manifest(output)
    assert restored == manifest
    assert restored.sha256() == hashlib.sha256(output.read_bytes()).hexdigest()
    verify_git_snapshot(
        restored,
        repository,
        resolved_commit_sha1=COMMIT_SHA1,
        resolved_git_blob_sha1s={
            item.relative_path: item.git_blob_sha1 for item in restored.files
        },
    )
    with pytest.raises(GitSnapshotError, match="commit"):
        verify_git_snapshot(
            restored, repository, resolved_commit_sha1="0" * 40)
    damaged = replace(
        restored,
        files=(replace(restored.files[0], git_blob_sha1="0" * 40),
               restored.files[1]),
    )
    with pytest.raises(GitSnapshotError, match="blob SHA-1"):
        verify_git_snapshot(
            damaged,
            repository,
            resolved_commit_sha1=COMMIT_SHA1,
            resolved_git_blob_sha1s={
                item.relative_path: item.git_blob_sha1 for item in restored.files
            },
        )
    output.write_text("{}\n", encoding="utf-8")
    with pytest.raises(GitSnapshotError, match="内容不同"):
        write_git_snapshot_manifest(manifest, output)


def test_repository_manifest_freezes_r218_commit_files_and_zero_anomaly():
    """正式引用只记录官方 r2.18 commit/hash/统计，不复制完整 treebank。"""
    manifest = read_git_snapshot_manifest(MANIFEST_PATH)
    assert manifest.sha256() == (
        "acb724a72c6d1f0d2b5d5d91997eeaa28d8b2e2a690a7be945c1c3499d88b98a")
    assert manifest.source_key == "UD_ZH_GSDSIMP_R2_18"
    assert manifest.tag == "r2.18"
    assert manifest.commit_sha1 == COMMIT_SHA1
    assert manifest.license_id == "CC-BY-SA-4.0"
    assert manifest.license_evidence_path == "LICENSE.txt"
    assert manifest.redistribution_policy == "PUBLIC"
    assert len(manifest.files) == 7
    reports = manifest.parser_report.to_value()
    assert reports["anomaly_count"] == 0
    assert reports["sentence_count"] == 4997
    assert reports["word_count"] == 123289
    assert reports["range_count"] == 0
    assert reports["empty_count"] == 0
    assert reports["scan_passes"] == 2
    assert reports["tag_aliases_at_commit"] == ["r2.17", "r2.18"]
    assert {item.split for item in manifest.files if item.split} == {
        "train", "dev", "held_out",
    }
    assert all(item.anomaly_count == 0 for item in manifest.files)
