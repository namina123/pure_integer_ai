"""覆盖 recovery-v10 五个 TRAIN family overlap/collision audit。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v10_source_expansion_five_family_audit import (
    publish_normalization_recovery_v10_five_family_audit,
    read_normalization_recovery_v10_five_family_audit,
    read_normalization_recovery_v10_five_family_audit_aggregate,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v10_source_expansion_five_family_audit_records import (
    V10_FIVE_FAMILY_AUDIT_FAMILIES,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
)


def _manifest(family: str, index: int) -> dict[str, object]:
    """构造独立lineage、locale blob与固定pack identity。"""
    return {
        "artifact_kind": "SYNTHETIC_SOURCE_PACK",
        "files": [
            {"role": "V10_SOURCE_RAW_LICENSE_BLOB", "sha256": "a" * 64},
            {
                "role": "V10_SOURCE_RAW_LOCALE_BLOB",
                "sha256": f"{index + 1:x}" * 64,
            },
        ],
        "manifest_sha256": f"{index + 6:x}" * 64,
        "raw_source": {
            "commit": f"{index + 1:x}" * 40,
            "repository": f"https://github.com/example/{family.lower()}.git",
            "root_tree": f"{index + 6:x}" * 40,
        },
        "source_family": family,
        "source_family_vote_count": 1,
    }


def _pair(
        family: str,
        *,
        source: str,
        input_text: str,
        output_text: str,
        identity: int = 0,
        ) -> dict[str, object]:
    """构造统一字段的synthetic pair。"""
    return {
        "contains_han_both": 1,
        "equal_length": int(len(input_text) == len(output_text)),
        "identity_preservation": identity,
        "official_source_text": source,
        "single_han_difference": 0,
        "source_family": family,
        "structure_equal": 1,
        "training_eligible": 1,
        "v8_training_eligible": 1,
        "within_scalar_limit": 1,
        "zh_hans": {"translation": output_text},
        "zh_hans_structure_tokens": [],
        "zh_hant": {"translation": input_text},
        "zh_hant_structure_tokens": [],
    }


def _state(*, copied: bool = False):
    """构造五家数据，并保留一个identity/changed跨family冲突。"""
    manifests = {
        family: _manifest(family, index)
        for index, family in enumerate(V10_FIVE_FAMILY_AUDIT_FAMILIES)
    }
    pairs = {
        "QBITTORRENT_PROJECT": (
            _pair(
                "QBITTORRENT_PROJECT", source="Open", input_text="開啟",
                output_text="打开"),
            _pair(
                "QBITTORRENT_PROJECT", source="Queue", input_text="佇列",
                output_text="队列"),
        ),
        "STELLARIUM_PROJECT": (_pair(
            "STELLARIUM_PROJECT", source="Sky", input_text="天空",
            output_text="天空", identity=1),),
        "KEEPASSXC_PROJECT": (_pair(
            "KEEPASSXC_PROJECT", source="Save", input_text="儲存",
            output_text="保存"),),
        "MIXXX_PROJECT": (_pair(
            "MIXXX_PROJECT", source="Open", input_text="開啟",
            output_text="開啟", identity=1),),
        "MUMBLE_PROJECT": (
            _pair(
                "MUMBLE_PROJECT", source="Open", input_text="開啟",
                output_text="打开"),
            _pair(
                "MUMBLE_PROJECT", source="Mute", input_text="靜音",
                output_text="静音"),
        ),
    }
    if copied:
        pairs["MUMBLE_PROJECT"] = ({
            **pairs["MIXXX_PROJECT"][0],
            "source_family": "MUMBLE_PROJECT",
        },)
    return manifests, pairs


def _input_dirs(tmp_path: Path) -> tuple[Path, ...]:
    """创建publisher所需的八个synthetic输入目录。"""
    values = tuple(tmp_path / name for name in (
        "observation", "roster", "content", "qbit", "stellarium",
        "keepassxc", "mixxx", "mumble"))
    for value in values:
        value.mkdir()
    return values


def _publish(
        tmp_path: Path,
        inputs: tuple[Path, ...],
        target: Path,
        ) -> dict[str, object]:
    """集中publisher参数，避免测试重复路径样板。"""
    return publish_normalization_recovery_v10_five_family_audit(
        run_root=tmp_path,
        observation_dir=inputs[0],
        roster_dir=inputs[1],
        content_dir=inputs[2],
        qbittorrent_source_pack_dir=inputs[3],
        stellarium_source_pack_dir=inputs[4],
        keepassxc_source_pack_dir=inputs[5],
        mixxx_source_pack_dir=inputs[6],
        mumble_source_pack_dir=inputs[7],
        target_dir=target,
    )


def test_v10_five_family_audit_round_trip_collision_and_tamper(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """冻结十组pairwise与哈希化冲突账，并拒绝覆盖和同步篡改。"""
    import pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v10_source_expansion_five_family_audit as module

    inputs = _input_dirs(tmp_path)
    monkeypatch.setattr(module, "_require_k_root", lambda value: Path(value))
    monkeypatch.setattr(module, "_state", lambda **_kwargs: _state())
    target = tmp_path / "audit"
    published = _publish(tmp_path, inputs, target)
    manifest, outputs = read_normalization_recovery_v10_five_family_audit(
        target,
        observation_dir=inputs[0],
        roster_dir=inputs[1],
        content_dir=inputs[2],
        qbittorrent_source_pack_dir=inputs[3],
        stellarium_source_pack_dir=inputs[4],
        keepassxc_source_pack_dir=inputs[5],
        mixxx_source_pack_dir=inputs[6],
        mumble_source_pack_dir=inputs[7],
        expected_manifest_sha256=str(published["manifest_sha256"]),
    )
    assert manifest == published
    assert manifest["summary"]["hard_independence_failure_count"] == 0
    assert manifest["summary"]["cross_family_output_conflict_key_count"] == 1
    assert manifest["summary"]["identity_changed_mixed_key_count"] == 1
    assert len(outputs["family-census.jsonl"]) == 5
    assert len(outputs["pairwise-overlap.jsonl"]) == 10
    assert len(outputs["source-input-collisions.jsonl"]) == 1
    aggregate, aggregate_outputs = (
        read_normalization_recovery_v10_five_family_audit_aggregate(
            target,
            expected_manifest_sha256=str(published["manifest_sha256"]),
        ))
    assert aggregate == manifest
    assert aggregate_outputs == outputs
    with pytest.raises(BroadQaExternalDataError, match="input/target path"):
        _publish(tmp_path, inputs, target)

    path = target / "audit-census.jsonl"
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
        read_normalization_recovery_v10_five_family_audit(
            target,
            observation_dir=inputs[0],
            roster_dir=inputs[1],
            content_dir=inputs[2],
            qbittorrent_source_pack_dir=inputs[3],
            stellarium_source_pack_dir=inputs[4],
            keepassxc_source_pack_dir=inputs[5],
            mixxx_source_pack_dir=inputs[6],
            mumble_source_pack_dir=inputs[7],
            expected_manifest_sha256=hashlib.sha256(encoded).hexdigest(),
        )


def test_v10_five_family_audit_rejects_complete_family_copy(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """完整semantic family复制必须记为REJECTED，不能被共享锚点掩盖。"""
    import pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v10_source_expansion_five_family_audit as module

    inputs = _input_dirs(tmp_path)
    monkeypatch.setattr(module, "_require_k_root", lambda value: Path(value))
    monkeypatch.setattr(module, "_state", lambda **_kwargs: _state(copied=True))
    published = _publish(tmp_path, inputs, tmp_path / "rejected")
    assert published["summary"]["hard_independence_failure_count"] == 1
    assert published["status"] == (
        "FIVE_FAMILY_SOURCE_INDEPENDENCE_OR_COPY_GATE_REJECTED")
