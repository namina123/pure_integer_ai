"""OpenCC normalization 依赖来源 pack 的严格边界测试。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_source_pack import (
    NORMALIZATION_SOURCE_FILES,
    NORMALIZATION_SOURCE_PACK_KIND,
    NORMALIZATION_SOURCE_PACK_STATUS,
    inspect_normalization_source_payloads,
    publish_normalization_source_pack,
    read_normalization_source_pack,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


def test_normalization_source_pack_round_trip_is_dependency_only(
        tmp_path: Path,
        ) -> None:
    """正式依赖文件可冻结回读，但不产生标签、规则或 learner read。"""
    target = tmp_path / "normalization-source-pack"
    report = publish_normalization_source_pack(
        run_root=tmp_path,
        target_dir=target,
    )
    restored = read_normalization_source_pack(target)
    assert restored["manifest_sha256"] == report["manifest_sha256"]
    assert restored["artifact_kind"] == NORMALIZATION_SOURCE_PACK_KIND
    assert restored["status"] == NORMALIZATION_SOURCE_PACK_STATUS
    assert restored["file_count"] == 4
    assert restored["contrastive_non_equivalence_label_count"] == 0
    assert restored["semantic_labels_written"] == 0
    assert restored["rules_written"] == 0
    assert restored["learner_read_count"] == 0
    assert restored["parsing_contract"]["conversion_dictionary_order"] == [
        "TSPhrases.txt", "TSCharacters.txt"]
    assert {item["relative_path"] for item in restored["files"]} == set(
        NORMALIZATION_SOURCE_FILES)
    with pytest.raises(BroadQaExternalDataError, match="target 已存在"):
        publish_normalization_source_pack(
            run_root=tmp_path,
            target_dir=target,
        )


def test_normalization_source_reader_rejects_file_and_manifest_tamper(
        tmp_path: Path,
        ) -> None:
    """物理字节或零学习边界被改写时严格回读失败。"""
    target = tmp_path / "normalization-source-pack"
    publish_normalization_source_pack(
        run_root=tmp_path,
        target_dir=target,
    )
    dictionary = target / "dictionary" / "TSCharacters.txt"
    dictionary.write_bytes(dictionary.read_bytes() + "測\t测\n".encode("utf-8"))
    with pytest.raises(BroadQaExternalDataError, match="commitment 漂移"):
        read_normalization_source_pack(target)

    target_two = tmp_path / "normalization-source-pack-two"
    publish_normalization_source_pack(
        run_root=tmp_path,
        target_dir=target_two,
    )
    manifest_path = target_two / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["rules_written"] = 1
    manifest_path.write_bytes(canonical_json_line(manifest))
    with pytest.raises(BroadQaExternalDataError, match="manifest 漂移"):
        read_normalization_source_pack(target_two)


def test_normalization_source_parser_rejects_config_order_and_bad_tabs(
        tmp_path: Path,
        ) -> None:
    """字典顺序和单 tab 解析合同不得由调用方放宽。"""
    target = tmp_path / "normalization-source-pack"
    publish_normalization_source_pack(
        run_root=tmp_path,
        target_dir=target,
    )
    payloads = {
        name: (target / name).read_bytes()
        for name in NORMALIZATION_SOURCE_FILES
    }
    config = json.loads(payloads["config/t2s.json"])
    config["conversion_chain"][0]["dict"]["dicts"].reverse()
    payloads["config/t2s.json"] = json.dumps(config).encode("utf-8")
    with pytest.raises(BroadQaExternalDataError, match="解析顺序漂移"):
        inspect_normalization_source_payloads(payloads)

    payloads = {
        name: (target / name).read_bytes()
        for name in NORMALIZATION_SOURCE_FILES
    }
    payloads["dictionary/TSPhrases.txt"] = b"missing-tab\n"
    with pytest.raises(BroadQaExternalDataError, match="不是单 tab"):
        inspect_normalization_source_payloads(payloads)


def test_normalization_source_pack_requires_explicit_existing_run_root(
        tmp_path: Path,
        ) -> None:
    """来源 pack 不得逃出显式 run root 或静默创建不存在的 root。"""
    missing = tmp_path / "missing"
    with pytest.raises(BroadQaExternalDataError, match="有效 run root"):
        publish_normalization_source_pack(
            run_root=missing,
            target_dir=missing / "pack",
        )
    with pytest.raises(BroadQaExternalDataError, match="有效 run root"):
        publish_normalization_source_pack(
            run_root=tmp_path,
            target_dir=tmp_path.parent / "outside-pack",
        )
