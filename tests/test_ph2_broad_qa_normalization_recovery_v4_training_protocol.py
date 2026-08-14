"""normalization recovery-v4 三来源 TRAIN protocol 测试。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_recovery_v4_training_protocol as protocol,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


def _thunderbird_pair(
        identity: str,
        *,
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
        "message_id": identity,
        "pair_id": hashlib.sha256(identity.encode()).hexdigest(),
        "plain_pair_eligible": 1,
        "record_kind": "THUNDERBIRD_L10N_PATTERN_PAIR_V1",
        "relative_path": "test.ftl",
        "zh_cn": {**side, "surface_text": simplified},
        "zh_tw": {**side, "surface_text": traditional},
    }


def _godot_pair() -> dict[str, object]:
    """构造与其他来源共享 phrase 的最小 Godot pair。"""
    return {
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
    }


def _vscode_pair(identity: str, traditional: str, simplified: str) -> dict[str, object]:
    """构造最小 VS Code full-key-path pair。"""
    return {
        "contains_han_both": 1,
        "json_path_sha256": "6" * 64,
        "pair_id": hashlib.sha256(identity.encode()).hexdigest(),
        "record_kind": "VSCODE_LOCALIZATION_PAIR_V1",
        "structure_equal": 1,
        "training_eligible": 1,
        "translation_relative_path": "translations/main.i18n.json",
        "within_scalar_limit": 1,
        "zh_hans_file_id": "7" * 64,
        "zh_hans_structure_tokens": [],
        "zh_hans_text": simplified,
        "zh_hans_text_sha256": hashlib.sha256(
            simplified.encode()).hexdigest(),
        "zh_hant_file_id": "8" * 64,
        "zh_hant_structure_tokens": [],
        "zh_hant_text": traditional,
        "zh_hant_text_sha256": hashlib.sha256(
            traditional.encode()).hexdigest(),
    }


def _install_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    """安装不触碰真实 K 盘 artifact 的 synthetic strict readers。"""
    thunderbird_pairs = (
        _thunderbird_pair(
            "one", traditional="開啟檔案", simplified="打开文件"),
        _thunderbird_pair(
            "two", traditional="請開啟檔案", simplified="请打开文件"),
    )
    godot_pairs = (_godot_pair(),)
    vscode_pairs = (
        _vscode_pair("vscode-one", "開啟檔案", "打开文件"),
        _vscode_pair("vscode-two", "來源內容", "来源内容"),
    )
    monkeypatch.setattr(
        protocol, "read_normalization_recovery_v3_evaluation_commitment",
        lambda *args, **kwargs: {"manifest_sha256": (
            protocol.V4_EVALUATION_COMMITMENT_MANIFEST_SHA256)},
    )
    monkeypatch.setattr(
        protocol, "read_normalization_recovery_v3_thunderbird_source_pack",
        lambda path: ({"manifest_sha256": (
            protocol.V4_THUNDERBIRD_SOURCE_PACK_MANIFEST_SHA256)},
            (), thunderbird_pairs),
    )
    monkeypatch.setattr(
        protocol, "read_normalization_recovery_v3_godot_source_pack",
        lambda path: ({"manifest_sha256": (
            protocol.V4_GODOT_SOURCE_PACK_MANIFEST_SHA256)}, (), godot_pairs),
    )
    monkeypatch.setattr(
        protocol, "read_normalization_recovery_v4_vscode_source_pack",
        lambda path: ({"manifest_sha256": (
            protocol.V4_VSCODE_SOURCE_PACK_MANIFEST_SHA256)}, (), vscode_pairs),
    )


def _source_dirs(tmp_path: Path) -> tuple[Path, ...]:
    """创建 protocol path gate 所需的空输入目录。"""
    values = []
    for name in ("prior", "commitment", "thunderbird", "godot", "vscode"):
        path = tmp_path / name
        path.mkdir()
        values.append(path)
    return tuple(values)


def test_protocol_publish_learner_read_auditor_and_tamper_guard(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """v4 protocol 必须支持 learner-only 与三源重派生两条严格读路径。"""
    _install_sources(monkeypatch)
    monkeypatch.setattr(protocol, "_require_k_root", lambda value: Path(value))
    prior, commitment, thunderbird, godot, vscode = _source_dirs(tmp_path)
    target = tmp_path / "protocol"
    published = protocol.publish_normalization_recovery_v4_training_protocol(
        run_root=tmp_path,
        prior_evaluation_protocol_dir=prior,
        evaluation_commitment_dir=commitment,
        thunderbird_source_pack_dir=thunderbird,
        godot_source_pack_dir=godot,
        vscode_source_pack_dir=vscode,
        target_dir=target,
    )
    material = protocol.read_normalization_recovery_v4_learner_input(
        target, expected_manifest_sha256=published["manifest_sha256"])
    assert material[0] == published
    assert material[0]["base_rule_pack_read_count"] == 0
    assert material[0]["predecessor_rule_pack_read_count"] == 0
    assert material[0]["summary"]["source_family_count"] == 3
    assert material[0]["summary"]["cross_family_target_candidate_count"] >= 1
    audited = protocol.read_normalization_recovery_v4_training_protocol(
        target,
        expected_manifest_sha256=published["manifest_sha256"],
        thunderbird_source_pack_dir=thunderbird,
        godot_source_pack_dir=godot,
        vscode_source_pack_dir=vscode,
    )
    assert audited == published
    with pytest.raises(BroadQaExternalDataError, match="target 已存在"):
        protocol.publish_normalization_recovery_v4_training_protocol(
            run_root=tmp_path,
            prior_evaluation_protocol_dir=prior,
            evaluation_commitment_dir=commitment,
            thunderbird_source_pack_dir=thunderbird,
            godot_source_pack_dir=godot,
            vscode_source_pack_dir=vscode,
            target_dir=target,
        )
    group_path = target / "train.phrase-groups.jsonl"
    lines = group_path.read_bytes().splitlines(keepends=True)
    value = json.loads(lines[0])
    value["unscoped_execution_allowed"] = 1
    lines[0] = canonical_json_line(value)
    group_path.write_bytes(b"".join(lines))
    with pytest.raises(BroadQaExternalDataError, match="material 漂移"):
        protocol.read_normalization_recovery_v4_learner_input(
            target, expected_manifest_sha256=published["manifest_sha256"])


def test_protocol_rejects_non_k_root_before_write(tmp_path: Path) -> None:
    """正式 protocol publisher 不得回退到非 K 盘。"""
    target = tmp_path / "protocol"
    with pytest.raises(BroadQaExternalDataError, match="K 盘"):
        protocol.publish_normalization_recovery_v4_training_protocol(
            run_root=tmp_path,
            prior_evaluation_protocol_dir=tmp_path / "prior",
            evaluation_commitment_dir=tmp_path / "commitment",
            thunderbird_source_pack_dir=tmp_path / "thunderbird",
            godot_source_pack_dir=tmp_path / "godot",
            vscode_source_pack_dir=tmp_path / "vscode",
            target_dir=target,
        )
    assert not target.exists()
