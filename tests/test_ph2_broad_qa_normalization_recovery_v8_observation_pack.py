"""覆盖 recovery-v8 三家统一 Observation pack。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_observation_pack import (
    publish_normalization_recovery_v8_observation_pack,
    read_normalization_recovery_v8_observation_pack,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
)


_FAMILIES = (
    "QBITTORRENT_PROJECT",
    "STELLARIUM_PROJECT",
    "KEEPASSXC_PROJECT",
)


def _pair(
        family: str,
        *,
        contains_han: int,
        structure_equal: int,
        within_limit: int,
        ) -> dict[str, object]:
    """构造一个eligibility facts自洽的synthetic source pair。"""
    index = _FAMILIES.index(family) + 1
    training = int(structure_equal == 1 and within_limit == 1)
    eligible = int(training == 1 and contains_han == 1)
    file_id_hans = str(index) * 64
    file_id_hant = str(index + 3) * 64
    source = f"Source {index}"
    identity = {"domain": f"domain-{index}", "source": source}
    return {
        "contains_han_both": contains_han,
        "equal_length": 0,
        "format_version": 1,
        "identity_preservation": 0,
        "license_expression": "MIT",
        "official_source_text": source,
        "pair_id": str(index + 6) * 64,
        "record_kind": "SOURCE_PAIR_V1",
        "single_han_difference": 0,
        "source_family": family,
        "source_identity": identity,
        "source_identity_sha256": str(index + 2) * 64,
        "source_policy_scope": family + "_SCOPE",
        "structure_equal": structure_equal,
        "training_eligible": training,
        "v8_training_eligible": eligible,
        "within_scalar_limit": within_limit,
        "zh_hans": {
            "source_file_id": file_id_hans,
            "source_file_sha256": str(index + 1) * 64,
            "translation": f"简体{index}",
        },
        "zh_hans_structure_tokens": ([] if structure_equal else ["%1"]),
        "zh_hant": {
            "source_file_id": file_id_hant,
            "source_file_sha256": str(index + 4) * 64,
            "translation": f"繁體{index}",
        },
        "zh_hant_structure_tokens": [],
    }


def _state():
    """构造eligible、structure/no-Han、scalar三种Observation状态。"""
    pairs = {
        "QBITTORRENT_PROJECT": (_pair(
            "QBITTORRENT_PROJECT",
            contains_han=1,
            structure_equal=1,
            within_limit=1,
        ),),
        "STELLARIUM_PROJECT": (_pair(
            "STELLARIUM_PROJECT",
            contains_han=0,
            structure_equal=0,
            within_limit=1,
        ),),
        "KEEPASSXC_PROJECT": (_pair(
            "KEEPASSXC_PROJECT",
            contains_han=1,
            structure_equal=1,
            within_limit=0,
        ),),
    }
    manifests = {}
    source_files = {}
    for index, family in enumerate(_FAMILIES, start=1):
        manifests[family] = {
            "parser_summary": {
                "locale_summaries": {
                    "zh_Hans": {"domain": {
                        "unfinished_count": index,
                    }},
                    "zh_Hant": {"domain": {
                        "unfinished_count": 0,
                    }},
                },
                "source_format_policy": {"parser": "synthetic"},
            },
        }
        source_files[family] = ({
            "file_id": str(index) * 64,
            "relative_path": f"family-{index}.txt",
        },)
    overlap = {
        "status": "THREE_INDEPENDENT_FAMILIES_NO_LOCALE_OR_SUBSET_COPY",
    }
    return manifests, source_files, pairs, overlap


def _input_dirs(tmp_path: Path) -> tuple[Path, ...]:
    """创建publisher所需的八个synthetic输入目录。"""
    values = tuple(tmp_path / name for name in (
        "v2-roster", "v1-roster", "v1-content", "v2-content",
        "overlap", "qbit", "stellarium", "keepassxc"))
    for value in values:
        value.mkdir()
    return values


def _publish(
        tmp_path: Path,
        inputs: tuple[Path, ...],
        target: Path,
        ) -> dict[str, object]:
    """调用publisher并保持测试参数集中。"""
    return publish_normalization_recovery_v8_observation_pack(
        run_root=tmp_path,
        v2_roster_dir=inputs[0],
        v1_roster_dir=inputs[1],
        v1_content_audit_dir=inputs[2],
        v2_content_audit_dir=inputs[3],
        source_overlap_dir=inputs[4],
        qbittorrent_source_pack_dir=inputs[5],
        stellarium_source_pack_dir=inputs[6],
        keepassxc_source_pack_dir=inputs[7],
        target_dir=target,
    )


def test_v8_observation_round_trip_exclusion_census_and_tamper(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """全量Observation保留三种资格状态，并拒绝覆盖与同步篡改。"""
    import pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_observation_pack as module

    inputs = _input_dirs(tmp_path)
    monkeypatch.setattr(module, "_require_k_root", lambda value: Path(value))
    monkeypatch.setattr(module, "_state", lambda **_kwargs: _state())
    target = tmp_path / "observations"
    published = _publish(tmp_path, inputs, target)
    manifest, outputs = read_normalization_recovery_v8_observation_pack(
        target,
        v2_roster_dir=inputs[0],
        v1_roster_dir=inputs[1],
        v1_content_audit_dir=inputs[2],
        v2_content_audit_dir=inputs[3],
        source_overlap_dir=inputs[4],
        qbittorrent_source_pack_dir=inputs[5],
        stellarium_source_pack_dir=inputs[6],
        keepassxc_source_pack_dir=inputs[7],
        expected_manifest_sha256=str(published["manifest_sha256"]),
    )
    assert manifest == published
    assert manifest["summary"]["observation_count"] == 3
    assert manifest["summary"]["v8_training_eligible_count"] == 1
    assert manifest["summary"]["v8_training_excluded_count"] == 2
    stellarium = outputs["stellarium-observations.jsonl"][0]
    assert stellarium["eligibility"]["exclusion_reasons"] == [
        "STRUCTURE_UNEQUAL", "NO_HAN_BOTH"]
    keepassxc = outputs["keepassxc-observations.jsonl"][0]
    assert keepassxc["eligibility"]["exclusion_reasons"] == [
        "SCALAR_LIMIT"]
    family_census = {item["source_family"]: item
                     for item in outputs["family-census.jsonl"]}
    assert family_census["QBITTORRENT_PROJECT"][
        "parser_stage_exclusion_locale_entry_counts"]["UNFINISHED"] == 1
    with pytest.raises(BroadQaExternalDataError, match="input/target path"):
        _publish(tmp_path, inputs, target)

    path = target / "qbittorrent-observations.jsonl"
    values = [json.loads(line) for line in path.read_bytes().splitlines()]
    values[0]["eligibility"]["status"] = "CHANGED"
    changed = b"".join(canonical_json_line(item) for item in values)
    path.write_bytes(changed)
    stored = json.loads((target / "manifest.json").read_bytes())
    artifact = next(item for item in stored["files"]
                    if item["relative_path"] == path.name)
    artifact["bytes"] = len(changed)
    artifact["sha256"] = hashlib.sha256(changed).hexdigest()
    encoded = canonical_json_line(stored)
    (target / "manifest.json").write_bytes(encoded)
    with pytest.raises(BroadQaExternalDataError, match="records 漂移"):
        read_normalization_recovery_v8_observation_pack(
            target,
            v2_roster_dir=inputs[0],
            v1_roster_dir=inputs[1],
            v1_content_audit_dir=inputs[2],
            v2_content_audit_dir=inputs[3],
            source_overlap_dir=inputs[4],
            qbittorrent_source_pack_dir=inputs[5],
            stellarium_source_pack_dir=inputs[6],
            keepassxc_source_pack_dir=inputs[7],
            expected_manifest_sha256=hashlib.sha256(encoded).hexdigest(),
        )
