"""覆盖 recovery-v10 五 family Observation pack。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v10_source_expansion_observation_pack import (
    publish_normalization_recovery_v10_source_expansion_observation_pack,
    read_normalization_recovery_v10_source_expansion_observation_pack,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
)


_OLD = (
    ("QBITTORRENT_PROJECT", "qbittorrent-observations.jsonl"),
    ("STELLARIUM_PROJECT", "stellarium-observations.jsonl"),
    ("KEEPASSXC_PROJECT", "keepassxc-observations.jsonl"),
)


def _source_file(family: str, index: int) -> dict[str, object]:
    """构造新family的source-file record。"""
    return {
        "file_id": f"{index:x}" * 64,
        "format_version": 1,
        "record_kind": "SYNTHETIC_SOURCE_FILE",
    }


def _pair(family: str, index: int, *, identity: int) -> dict[str, object]:
    """构造可提升为Observation的完整source pair。"""
    input_text = "相同" if identity else "開啟"
    output_text = input_text if identity else "打开"
    file_sha = f"{index + 2:x}" * 64
    locale = {
        "source_file_id": f"{index:x}" * 64,
        "source_file_sha256": file_sha,
        "translation": output_text,
    }
    return {
        "contains_han_both": 1,
        "equal_length": int(len(input_text) == len(output_text)),
        "identity_preservation": identity,
        "license_expression": "MIT",
        "official_source_text": f"Source {index}",
        "pair_id": f"{index + 4:x}" * 64,
        "record_kind": "SYNTHETIC_PAIR",
        "single_han_difference": 0,
        "source_family": family,
        "source_identity": {"source": f"Source {index}"},
        "source_identity_sha256": f"{index + 6:x}" * 64,
        "source_policy_scope": f"{family}_SCOPE",
        "structure_equal": 1,
        "training_eligible": 1,
        "v8_training_eligible": 1,
        "within_scalar_limit": 1,
        "zh_hans": locale,
        "zh_hans_structure_tokens": [],
        "zh_hant": {**locale, "translation": input_text},
        "zh_hant_structure_tokens": [],
    }


def _state():
    """构造三家旧Observation和两家新source pack的闭合state。"""
    predecessor_outputs = {
        "source-files.jsonl": tuple({
            "observation_source_file_id": f"{index + 1:x}" * 64,
            "source_family": family,
        } for index, (family, _name) in enumerate(_OLD)),
        "family-census.jsonl": tuple({
            "family_vote_count": 1,
            "identity_pair_count": int(index == 1),
            "observation_count": 1,
            "source_family": family,
            "source_file_record_count": 1,
            "v8_training_eligible_count": 1,
            "v8_training_excluded_count": 0,
        } for index, (family, _name) in enumerate(_OLD)),
        "observation-census.jsonl": ({"observation_count": 33_179},),
    }
    for index, (family, name) in enumerate(_OLD):
        predecessor_outputs[name] = ({
            "observation_id": f"{index + 1:x}" * 64,
            "source_family": family,
        },)
    manifests = {}
    source_files = {}
    pairs = {}
    for index, family in enumerate(("MIXXX_PROJECT", "MUMBLE_PROJECT"), 7):
        manifests[family] = {
            "manifest_sha256": f"{index:x}" * 64,
            "parser_summary": {
                "locale_summaries": {},
                "source_format_policy": {},
            },
        }
        source_files[family] = (_source_file(family, index),)
        pairs[family] = (_pair(family, index, identity=int(index == 8)),)
    audit_summary = {
        "cross_family_output_conflict_key_count": 1,
        "identity_changed_mixed_key_count": 1,
        "identity_pair_count": 2,
        "pair_record_count": 5,
        "source_input_collision_record_count": 1,
        "training_eligible_pair_count": 5,
        "training_excluded_pair_count": 0,
    }
    audit_manifest = {
        "files": [{
            "relative_path": "source-input-collisions.jsonl",
            "sha256": "c" * 64,
        }],
        "status": (
            "FIVE_INDEPENDENT_TRAIN_FAMILIES_COLLISIONS_FROZEN_NOT_OBSERVED"),
        "summary": audit_summary,
    }
    audit_outputs = {"source-input-collisions.jsonl": ({"id": 1},)}
    predecessor_manifest = {
        "files": [
            {"relative_path": name, "sha256": f"{index + 1:x}" * 64}
            for index, (_family, name) in enumerate(_OLD)
        ],
    }
    return (
        predecessor_manifest,
        predecessor_outputs,
        manifests,
        source_files,
        pairs,
        audit_manifest,
        audit_outputs,
    )


def _input_dirs(tmp_path: Path) -> tuple[Path, ...]:
    """创建publisher所需的六个synthetic输入目录。"""
    values = tuple(tmp_path / name for name in (
        "predecessor", "audit", "roster", "content", "mixxx", "mumble"))
    for value in values:
        value.mkdir()
    return values


def _publish(
        tmp_path: Path,
        inputs: tuple[Path, ...],
        target: Path,
        ) -> dict[str, object]:
    """集中Observation publisher参数。"""
    return publish_normalization_recovery_v10_source_expansion_observation_pack(
        run_root=tmp_path,
        predecessor_observation_dir=inputs[0],
        five_family_audit_dir=inputs[1],
        roster_dir=inputs[2],
        content_dir=inputs[3],
        mixxx_source_pack_dir=inputs[4],
        mumble_source_pack_dir=inputs[5],
        target_dir=target,
    )


def test_v10_expanded_observation_round_trip_preserves_old_and_tamper(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """旧三家记录不变，新两家提升后闭合五family分母并拒绝篡改。"""
    import pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v10_source_expansion_observation_pack as module

    inputs = _input_dirs(tmp_path)
    state = _state()
    monkeypatch.setattr(module, "_require_k_root", lambda value: Path(value))
    monkeypatch.setattr(module, "_state", lambda **_kwargs: state)
    target = tmp_path / "observation"
    published = _publish(tmp_path, inputs, target)
    manifest, outputs = (
        read_normalization_recovery_v10_source_expansion_observation_pack(
            target,
            predecessor_observation_dir=inputs[0],
            five_family_audit_dir=inputs[1],
            roster_dir=inputs[2],
            content_dir=inputs[3],
            mixxx_source_pack_dir=inputs[4],
            mumble_source_pack_dir=inputs[5],
            expected_manifest_sha256=str(published["manifest_sha256"]),
        ))
    assert manifest == published
    assert manifest["summary"]["observation_count"] == 5
    assert manifest["summary"]["v8_training_eligible_count"] == 5
    assert outputs["qbittorrent-observations.jsonl"] == (
        state[1]["qbittorrent-observations.jsonl"])
    assert len(outputs["mixxx-observations.jsonl"]) == 1
    assert len(outputs["mumble-observations.jsonl"]) == 1
    with pytest.raises(BroadQaExternalDataError, match="input/target path"):
        _publish(tmp_path, inputs, target)

    path = target / "observation-census.jsonl"
    changed = canonical_json_line({"format_version": 1, "record_kind": "X"})
    path.write_bytes(changed)
    stored = json.loads((target / "manifest.json").read_bytes())
    artifact = next(item for item in stored["files"]
                    if item["relative_path"] == path.name)
    artifact["bytes"] = len(changed)
    artifact["sha256"] = hashlib.sha256(changed).hexdigest()
    encoded = canonical_json_line(stored)
    (target / "manifest.json").write_bytes(encoded)
    with pytest.raises(BroadQaExternalDataError, match="records 漂移"):
        read_normalization_recovery_v10_source_expansion_observation_pack(
            target,
            predecessor_observation_dir=inputs[0],
            five_family_audit_dir=inputs[1],
            roster_dir=inputs[2],
            content_dir=inputs[3],
            mixxx_source_pack_dir=inputs[4],
            mumble_source_pack_dir=inputs[5],
            expected_manifest_sha256=hashlib.sha256(encoded).hexdigest(),
        )


def test_v10_expanded_observation_rejects_audit_denominator_drift(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """audit与五family Observation分母不一致时必须fail closed。"""
    import pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v10_source_expansion_observation_pack as module

    inputs = _input_dirs(tmp_path)
    state = list(_state())
    state[5] = {
        **state[5],
        "summary": {**state[5]["summary"], "pair_record_count": 6},
    }
    monkeypatch.setattr(module, "_require_k_root", lambda value: Path(value))
    monkeypatch.setattr(module, "_state", lambda **_kwargs: tuple(state))
    with pytest.raises(BroadQaExternalDataError, match="denominator 漂移"):
        _publish(tmp_path, inputs, tmp_path / "rejected")
