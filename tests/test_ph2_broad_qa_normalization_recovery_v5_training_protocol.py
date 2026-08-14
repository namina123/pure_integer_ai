"""normalization recovery-v5 四来源 TRAIN protocol 测试。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_recovery_v5_training_protocol as protocol,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_training_records import (
    derive_normalization_recovery_v5_groups,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


_TRADITIONAL = "檔案總管"
_SIMPLIFIED = "文件管理器"


def _thunderbird_pair(identity: str) -> dict[str, object]:
    """构造最小 Thunderbird variable-length pair。"""
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
        "zh_cn": {**side, "surface_text": _SIMPLIFIED},
        "zh_tw": {**side, "surface_text": _TRADITIONAL},
    }


def _godot_pair() -> dict[str, object]:
    """构造最小 Godot variable-length pair。"""
    return {
        "pair_id": "3" * 64,
        "record_kind": "GODOT_EDITOR_PO_PAIR_V1",
        "source_identity": {
            "msgctxt": "", "msgid": "File manager", "msgid_plural": ""},
        "training_eligible": 1,
        "zh_hans": {
            "entry_linenum": 1,
            "entry_semantic_sha256": "4" * 64,
            "msgstr": _SIMPLIFIED,
            "structure_tokens": [],
        },
        "zh_hant": {
            "entry_linenum": 1,
            "entry_semantic_sha256": "5" * 64,
            "msgstr": _TRADITIONAL,
            "structure_tokens": [],
        },
    }


def _vscode_pair() -> dict[str, object]:
    """构造最小 VS Code variable-length pair。"""
    return {
        "contains_han_both": 1,
        "json_path_sha256": "6" * 64,
        "pair_id": "7" * 64,
        "record_kind": "VSCODE_LOCALIZATION_PAIR_V1",
        "structure_equal": 1,
        "training_eligible": 1,
        "translation_relative_path": "translations/main.i18n.json",
        "within_scalar_limit": 1,
        "zh_hans_file_id": "8" * 64,
        "zh_hans_structure_tokens": [],
        "zh_hans_text": _SIMPLIFIED,
        "zh_hans_text_sha256": hashlib.sha256(
            _SIMPLIFIED.encode()).hexdigest(),
        "zh_hant_file_id": "9" * 64,
        "zh_hant_structure_tokens": [],
        "zh_hant_text": _TRADITIONAL,
        "zh_hant_text_sha256": hashlib.sha256(
            _TRADITIONAL.encode()).hexdigest(),
    }


def _libreoffice_pair() -> dict[str, object]:
    """构造最小 LibreOffice variable-length pair。"""
    return {
        "pair_id": "a" * 64,
        "record_kind": "LIBREOFFICE_CUI_PO_PAIR_V1",
        "source_identity_sha256": "b" * 64,
        "structure_equal": 1,
        "training_eligible": 1,
        "within_scalar_limit": 1,
        "zh_hans_structure_tokens": [],
        "zh_hant_structure_tokens": [],
        "zh_hans": {
            "entry_semantic_sha256": "c" * 64,
            "msgstr": _SIMPLIFIED,
            "source_file_id": "d" * 64,
        },
        "zh_hant": {
            "entry_semantic_sha256": "e" * 64,
            "msgstr": _TRADITIONAL,
            "source_file_id": "f" * 64,
        },
    }


def _install_sources(
        monkeypatch: pytest.MonkeyPatch,
        calls: list[dict[str, object]],
        ) -> None:
    """安装不触碰真实 K artifact 的 synthetic strict readers。"""
    def read_commitment(*args: object, **kwargs: object) -> dict[str, object]:
        calls.append(dict(kwargs))
        return {"manifest_sha256": (
            protocol.V5_EVALUATION_COMMITMENT_MANIFEST_SHA256)}

    monkeypatch.setattr(
        protocol,
        "read_normalization_recovery_v5_evaluation_commitment",
        read_commitment,
    )
    monkeypatch.setattr(
        protocol, "read_normalization_recovery_v3_thunderbird_source_pack",
        lambda path: ({"manifest_sha256": (
            protocol.V5_THUNDERBIRD_SOURCE_PACK_MANIFEST_SHA256)}, (),
            (_thunderbird_pair("one"),)),
    )
    monkeypatch.setattr(
        protocol, "read_normalization_recovery_v3_godot_source_pack",
        lambda path: ({"manifest_sha256": (
            protocol.V5_GODOT_SOURCE_PACK_MANIFEST_SHA256)}, (),
            (_godot_pair(),)),
    )
    monkeypatch.setattr(
        protocol, "read_normalization_recovery_v4_vscode_source_pack",
        lambda path: ({"manifest_sha256": (
            protocol.V5_VSCODE_SOURCE_PACK_MANIFEST_SHA256)}, (),
            (_vscode_pair(),)),
    )
    monkeypatch.setattr(
        protocol, "read_normalization_recovery_v5_libreoffice_source_pack",
        lambda path: ({"manifest_sha256": (
            protocol.V5_LIBREOFFICE_SOURCE_PACK_MANIFEST_SHA256)}, (),
            (_libreoffice_pair(),)),
    )


def _source_dirs(tmp_path: Path) -> tuple[Path, ...]:
    """创建 protocol path gate 所需的空输入目录。"""
    values = []
    for name in (
            "commitment", "qt", "thunderbird", "godot", "vscode",
            "libreoffice"):
        path = tmp_path / name
        path.mkdir()
        values.append(path)
    return tuple(values)


def _fragment(
        identity: str,
        *,
        family: str,
        policy: str,
        ) -> dict[str, object]:
    """构造 direct authority test 使用的 variable whole-input fragment。"""
    return {
        "equal_length": 0,
        "fragment_id": hashlib.sha256(identity.encode()).hexdigest(),
        "fragment_kind": "WHOLE_INPUT",
        "input_text": _TRADITIONAL,
        "output_text": _SIMPLIFIED,
        "source_family": family,
        "source_policy_scope": policy,
    }


def test_variable_whole_input_requires_strong_cross_family_authority() -> None:
    """两 family 单例不足，两 family 各两份或三 family 才能升级 target。"""
    families = (
        ("GODOT_ENGINE_PROJECT", "GODOT_EDITOR_ZH_HANT_TO_ZH_HANS_V1"),
        ("LIBREOFFICE_PROJECT",
         "LIBREOFFICE_CUI_ZH_TW_TO_ZH_CN_FIXED_COMMIT_V1"),
        ("MICROSOFT_VSCODE_PROJECT",
         "VSCODE_ZH_HANT_TO_ZH_HANS_FIXED_COMMIT_V1"),
    )
    two = tuple(_fragment(str(index), family=family, policy=policy)
                for index, (family, policy) in enumerate(families[:2]))
    deferred = derive_normalization_recovery_v5_groups(two)[0]
    assert deferred["disposition"] == "DEFER_INSUFFICIENT_AUTHORITY"

    replicated = two + tuple(_fragment(
        f"repeat-{index}", family=family, policy=policy)
        for index, (family, policy) in enumerate(families[:2]))
    target = derive_normalization_recovery_v5_groups(replicated)[0]
    assert target["candidate_scope_kind"] == "TARGET_CROSS_FAMILY"
    assert target["authority_basis"] == (
        "VARIABLE_LENGTH_WHOLE_INPUT_STRONG_CONSENSUS")

    three = tuple(_fragment(
        f"three-{index}", family=family, policy=policy)
        for index, (family, policy) in enumerate(families))
    assert derive_normalization_recovery_v5_groups(
        three)[0]["candidate_scope_kind"] == "TARGET_CROSS_FAMILY"


def test_protocol_publish_learner_read_auditor_and_tamper_guard(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """v5 protocol 必须支持 label-blind gate、learner-only 与四源重派生。"""
    calls: list[dict[str, object]] = []
    _install_sources(monkeypatch, calls)
    monkeypatch.setattr(protocol, "_require_k_root", lambda value: Path(value))
    commitment, qt, thunderbird, godot, vscode, libreoffice = (
        _source_dirs(tmp_path))
    target = tmp_path / "protocol"
    published = protocol.publish_normalization_recovery_v5_training_protocol(
        run_root=tmp_path,
        evaluation_commitment_dir=commitment,
        qt_source_pack_dir=qt,
        thunderbird_source_pack_dir=thunderbird,
        godot_source_pack_dir=godot,
        vscode_source_pack_dir=vscode,
        libreoffice_source_pack_dir=libreoffice,
        target_dir=target,
    )
    assert len(calls) == 1
    assert calls[0]["expected_qt_source_manifest_sha256"] == (
        protocol.V5_QT_SOURCE_PACK_MANIFEST_SHA256)
    assert published["held_out_exclusion"]["qt_non_manifest_file_read_count"] == 0
    assert published["summary"]["source_family_count"] == 4
    assert published["summary"][
        "target_variable_length_whole_input_candidate_count"] >= 1
    material = protocol.read_normalization_recovery_v5_learner_input(
        target, expected_manifest_sha256=published["manifest_sha256"])
    assert material[0] == published
    assert material[0]["predecessor_rule_pack_read_count"] == 0
    audited = protocol.read_normalization_recovery_v5_training_protocol(
        target,
        expected_manifest_sha256=published["manifest_sha256"],
        thunderbird_source_pack_dir=thunderbird,
        godot_source_pack_dir=godot,
        vscode_source_pack_dir=vscode,
        libreoffice_source_pack_dir=libreoffice,
    )
    assert audited == published
    with pytest.raises(BroadQaExternalDataError, match="target 已存在"):
        protocol.publish_normalization_recovery_v5_training_protocol(
            run_root=tmp_path,
            evaluation_commitment_dir=commitment,
            qt_source_pack_dir=qt,
            thunderbird_source_pack_dir=thunderbird,
            godot_source_pack_dir=godot,
            vscode_source_pack_dir=vscode,
            libreoffice_source_pack_dir=libreoffice,
            target_dir=target,
        )
    group_path = target / "train.phrase-groups.jsonl"
    lines = group_path.read_bytes().splitlines(keepends=True)
    value = json.loads(lines[0])
    value["unscoped_execution_allowed"] = 1
    lines[0] = canonical_json_line(value)
    group_path.write_bytes(b"".join(lines))
    with pytest.raises(BroadQaExternalDataError, match="material 漂移"):
        protocol.read_normalization_recovery_v5_learner_input(
            target, expected_manifest_sha256=published["manifest_sha256"])


def test_protocol_rejects_non_k_root_before_write(tmp_path: Path) -> None:
    """正式 protocol publisher 不得回退到非 K 盘。"""
    target = tmp_path / "protocol"
    with pytest.raises(BroadQaExternalDataError, match="K 盘"):
        protocol.publish_normalization_recovery_v5_training_protocol(
            run_root=tmp_path,
            evaluation_commitment_dir=tmp_path / "commitment",
            qt_source_pack_dir=tmp_path / "qt",
            thunderbird_source_pack_dir=tmp_path / "thunderbird",
            godot_source_pack_dir=tmp_path / "godot",
            vscode_source_pack_dir=tmp_path / "vscode",
            libreoffice_source_pack_dir=tmp_path / "libreoffice",
            target_dir=target,
        )
    assert not target.exists()
