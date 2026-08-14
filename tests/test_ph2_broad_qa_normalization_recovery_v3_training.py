"""normalization recovery-v3 phrase/context TRAIN 协议测试。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_recovery_v3_training_protocol as protocol,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v3_training_records import (
    derive_normalization_recovery_v3_fragments,
    derive_normalization_recovery_v3_groups,
    derive_normalization_recovery_v3_pair_observations,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


def _source_pair(
        *,
        pair_id: str,
        traditional: str,
        simplified: str,
        ) -> dict[str, object]:
    """构造最小 Thunderbird plain pair。"""
    side = {
        "file_sha256": "1" * 64,
        "source_slice_sha256": "2" * 64,
        "surface_text": "",
    }
    return {
        "attribute_id": "",
        "entry_kind": "MESSAGE",
        "message_id": pair_id,
        "pair_id": hashlib.sha256(pair_id.encode()).hexdigest(),
        "plain_pair_eligible": 1,
        "record_kind": "THUNDERBIRD_L10N_PATTERN_PAIR_V1",
        "relative_path": "test.ftl",
        "zh_cn": {**side, "surface_text": simplified},
        "zh_tw": {**side, "surface_text": traditional},
    }


def test_records_keep_cross_family_consensus_and_conflict_separate() -> None:
    """同输出可成候选，不同输出必须保留冲突且禁止无 scope 执行。"""
    thunderbird = (
        _source_pair(pair_id="one", traditional="開啟檔案", simplified="打开文件"),
        _source_pair(pair_id="two", traditional="請開啟檔案", simplified="请打开文件"),
    )
    godot = ({
        "pair_id": "3" * 64,
        "record_kind": "GODOT_EDITOR_PO_PAIR_V1",
        "source_identity": {"msgctxt": "", "msgid": "Open", "msgid_plural": ""},
        "training_eligible": 1,
        "zh_hans": {
            "entry_linenum": 1,
            "entry_semantic_sha256": "4" * 64,
            "msgstr": "打开文件",
            "structure_tokens": [],
        },
        "zh_hant": {
            "entry_linenum": 1,
            "entry_semantic_sha256": "5" * 64,
            "msgstr": "開啟檔案",
            "structure_tokens": [],
        },
    },)
    observations = derive_normalization_recovery_v3_pair_observations(
        thunderbird_manifest_sha256="a" * 64,
        thunderbird_pairs=thunderbird,
        godot_manifest_sha256="b" * 64,
        godot_pairs=godot,
    )
    fragments = derive_normalization_recovery_v3_fragments(observations)
    groups = derive_normalization_recovery_v3_groups(fragments)
    assert any(item["disposition"]
               == "CROSS_FAMILY_CONSENSUS_CANDIDATE" for item in groups)
    assert all(item["unscoped_execution_allowed"] == 0 for item in groups)
    assert all(item["negative_evidence_required_before_execution"] == 1
               for item in groups if item["disposition"].endswith("CANDIDATE"))


def _install_protocol_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    """安装不触碰真实 K 盘来源的 synthetic readers。"""
    thunderbird_pairs = (
        _source_pair(pair_id="one", traditional="開啟檔案", simplified="打开文件"),
        _source_pair(pair_id="two", traditional="請開啟檔案", simplified="请打开文件"),
    )
    godot_pairs = ({
        "pair_id": "3" * 64,
        "record_kind": "GODOT_EDITOR_PO_PAIR_V1",
        "source_identity": {"msgctxt": "", "msgid": "Open", "msgid_plural": ""},
        "training_eligible": 1,
        "zh_hans": {
            "entry_linenum": 1,
            "entry_semantic_sha256": "4" * 64,
            "msgstr": "打开文件",
            "structure_tokens": [],
        },
        "zh_hant": {
            "entry_linenum": 1,
            "entry_semantic_sha256": "5" * 64,
            "msgstr": "開啟檔案",
            "structure_tokens": [],
        },
    },)
    monkeypatch.setattr(
        protocol, "read_normalization_recovery_v3_evaluation_commitment",
        lambda *args, **kwargs: {"manifest_sha256": (
            protocol.V3_EVALUATION_COMMITMENT_MANIFEST_SHA256)},
    )
    monkeypatch.setattr(
        protocol, "read_normalization_recovery_v3_thunderbird_source_pack",
        lambda path: ({"manifest_sha256": (
            protocol.THUNDERBIRD_SOURCE_PACK_MANIFEST_SHA256)}, (), thunderbird_pairs),
    )
    monkeypatch.setattr(
        protocol, "read_normalization_recovery_v3_godot_source_pack",
        lambda path: ({"manifest_sha256": (
            protocol.GODOT_SOURCE_PACK_MANIFEST_SHA256)}, (), godot_pairs),
    )


def test_protocol_publish_learner_read_and_tamper_guard(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """protocol 不可覆盖，learner-only reader 严格绑定全部物化文件。"""
    _install_protocol_sources(monkeypatch)
    monkeypatch.setattr(protocol, "_require_k_root", lambda value: Path(value))
    paths = []
    for name in ("prior", "commitment", "thunderbird", "godot"):
        path = tmp_path / name
        path.mkdir()
        paths.append(path)
    target = tmp_path / "protocol"
    published = protocol.publish_normalization_recovery_v3_training_protocol(
        run_root=tmp_path,
        prior_evaluation_protocol_dir=paths[0],
        evaluation_commitment_dir=paths[1],
        thunderbird_source_pack_dir=paths[2],
        godot_source_pack_dir=paths[3],
        target_dir=target,
    )
    material = protocol.read_normalization_recovery_v3_learner_input(
        target, expected_manifest_sha256=published["manifest_sha256"])
    assert material[0] == published
    assert all(material[index] for index in range(1, 5))
    with pytest.raises(BroadQaExternalDataError, match="target 已存在"):
        protocol.publish_normalization_recovery_v3_training_protocol(
            run_root=tmp_path,
            prior_evaluation_protocol_dir=paths[0],
            evaluation_commitment_dir=paths[1],
            thunderbird_source_pack_dir=paths[2],
            godot_source_pack_dir=paths[3],
            target_dir=target,
        )
    group_path = target / "train.phrase-groups.jsonl"
    lines = group_path.read_bytes().splitlines(keepends=True)
    value = json.loads(lines[0])
    value["unscoped_execution_allowed"] = 1
    lines[0] = canonical_json_line(value)
    group_path.write_bytes(b"".join(lines))
    with pytest.raises(BroadQaExternalDataError, match="material 漂移"):
        protocol.read_normalization_recovery_v3_learner_input(
            target, expected_manifest_sha256=published["manifest_sha256"])
