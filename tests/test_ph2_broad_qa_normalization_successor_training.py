"""normalization successor TRAIN 记录与物化协议测试。"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_successor_training_protocol as protocol_module,
)
from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_successor_training_protocol import (
    ICU_TRAIN_SOURCE_MANIFEST_SHA256,
    NORMALIZATION_SUCCESSOR_CHECKPOINT_CONTRACT,
    NORMALIZATION_SUCCESSOR_EVIDENCE_CONTRACT,
    NORMALIZATION_SUCCESSOR_RULE_PACK_CONTRACT,
    OPENCC_TRAIN_SOURCE_MANIFEST_SHA256,
    publish_normalization_successor_training_protocol,
    read_normalization_successor_learner_input,
    read_normalization_successor_training_protocol,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_successor_training_records import (
    ICU_SOURCE_POLICY_SCOPE,
    OPENCC_SOURCE_POLICY_SCOPE,
    SUCCESSOR_TARGET_POLICY_SCOPE,
    derive_icu_successor_observations,
    derive_normalization_successor_training_records,
    derive_opencc_successor_observations,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


def _icu_rule(
        input_text: str,
        expected_output: str,
        ordinal: int,
        ) -> dict[str, object]:
    """构造保留完整物理来源承诺的最小 ICU eligible rule。"""
    encoded = f"rule-{ordinal}:{input_text}>{expected_output}\n".encode()
    digest = hashlib.sha256(encoded).hexdigest()
    byte_start = ordinal * 100
    byte_end = byte_start + len(encoded)
    return {
        "byte_end": byte_end,
        "byte_start": byte_start,
        "line_end_ordinal": ordinal,
        "line_start_ordinal": ordinal,
        "physical_lines": [{
            "byte_end": byte_end,
            "byte_start": byte_start,
            "line_ordinal": ordinal,
            "line_sha256": digest,
        }],
        "statement_sha256": digest,
        "t2s_expected_output": expected_output,
        "t2s_input": input_text,
        "t2s_reverse_eligible": 1,
    }


def _synthetic_inputs() -> tuple[bytes, bytes, tuple[dict[str, object], ...]]:
    """返回同时含共识、冲突、单来源与 context replay 的两源输入。"""
    characters = "鍾\t钟\n乾\t干\n於\t于\n".encode()
    phrases = "鍾馗\t锺馗\n乾杯\t干杯\n".encode()
    rules = tuple(_icu_rule(*values, ordinal) for ordinal, values in enumerate((
        ("鍾", "钟"),
        ("乾", "干"),
        ("鐘", "钟"),
        ("鍾馗", "钟馗"),
        ("乾杯", "乾杯"),
    ), start=1))
    return characters, phrases, rules


def _derive_synthetic() -> tuple[
        tuple[dict[str, object], ...],
        tuple[dict[str, object], ...],
        tuple[dict[str, object], ...],
        dict[str, object],
        ]:
    """从 synthetic 两源输入执行完整纯记录派生。"""
    characters, phrases, rules = _synthetic_inputs()
    opencc = derive_opencc_successor_observations(
        source_pack_manifest_sha256=OPENCC_TRAIN_SOURCE_MANIFEST_SHA256,
        character_payload=characters,
        phrase_payload=phrases,
    )
    icu = derive_icu_successor_observations(
        source_pack_manifest_sha256=ICU_TRAIN_SOURCE_MANIFEST_SHA256,
        rules=rules,
    )
    return derive_normalization_successor_training_records(
        opencc_observations=opencc, icu_observations=icu)


def _publish_synthetic(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> tuple[Path, Path, Path, dict[str, object]]:
    """在临时根发布协议，同时保留 publisher 的真实物理读路径。"""
    characters, phrases, rules = _synthetic_inputs()
    opencc = tmp_path / "opencc-source"
    icu = tmp_path / "icu-source"
    (opencc / "dictionary").mkdir(parents=True)
    icu.mkdir()
    (opencc / "dictionary" / "TSCharacters.txt").write_bytes(characters)
    (opencc / "dictionary" / "TSPhrases.txt").write_bytes(phrases)
    monkeypatch.setattr(protocol_module, "_require_k_root", lambda value: Path(value))
    monkeypatch.setattr(
        protocol_module,
        "read_normalization_source_pack",
        lambda path: {"manifest_sha256": OPENCC_TRAIN_SOURCE_MANIFEST_SHA256},
    )
    monkeypatch.setattr(
        protocol_module,
        "read_normalization_icu_source_pack",
        lambda path: (
            {"manifest_sha256": ICU_TRAIN_SOURCE_MANIFEST_SHA256}, (), rules),
    )
    target = tmp_path / "training-protocol"
    report = publish_normalization_successor_training_protocol(
        run_root=tmp_path,
        opencc_source_pack_dir=opencc,
        icu_source_pack_dir=icu,
        target_dir=target,
    )
    return target, opencc, icu, report


def test_training_records_separate_consensus_conflict_and_context() -> None:
    """机械派生保留来源 policy，不把冲突或短语 override 写成全局规则。"""
    observations, groups, contexts, summary = _derive_synthetic()
    assert len(observations) == 10
    assert summary["group_kind_counts"] == {
        "CROSS_SOURCE_CONSENSUS": 2,
        "SINGLE_SOURCE": 2,
        "SOURCE_POLICY_CONFLICT": 2,
    }
    by_input = {item["input_text"]: item for item in groups}
    assert by_input["鍾"]["consensus_output"] == "钟"
    assert by_input["鍾"]["eligible_target_policy_scope"] == (
        SUCCESSOR_TARGET_POLICY_SCOPE)
    assert by_input["鍾馗"]["group_kind"] == "SOURCE_POLICY_CONFLICT"
    assert by_input["鍾馗"]["eligible_target_policy_scope"] == ""
    qualifications = {
        (item["source_policy_scope"], item["input_text"]):
        item["qualification_kind"]
        for item in contexts
    }
    assert qualifications[(OPENCC_SOURCE_POLICY_SCOPE, "鍾馗")] == (
        "SOURCE_REPLAY_OVERRIDE")
    assert qualifications[(ICU_SOURCE_POLICY_SCOPE, "鍾馗")] == (
        "SOURCE_REPLAY_SUPPORT")


def test_training_protocol_round_trip_freezes_learning_contracts(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """物化协议固定来源、顺序、证据、规则包和 checkpoint 边界。"""
    target, opencc, icu, report = _publish_synthetic(tmp_path, monkeypatch)
    restored = read_normalization_successor_training_protocol(
        target,
        opencc_source_pack_dir=opencc,
        icu_source_pack_dir=icu,
    )
    manifest, observations, groups, contexts, work = restored
    assert manifest["manifest_sha256"] == report["manifest_sha256"]
    assert len(work) == len(observations) + len(groups) + len(contexts)
    assert [item["work_ordinal"] for item in work] == list(range(len(work)))
    contract = manifest["learner_contract"]
    assert contract["evidence_contract"] == NORMALIZATION_SUCCESSOR_EVIDENCE_CONTRACT
    assert contract["rule_pack_contract"] == NORMALIZATION_SUCCESSOR_RULE_PACK_CONTRACT
    assert contract["checkpoint_contract"] == (
        NORMALIZATION_SUCCESSOR_CHECKPOINT_CONTRACT)
    assert manifest["teacher_api_llm_call_count"] == 0
    assert manifest["evaluation_or_reserve_artifact_read_count"] == 0
    assert manifest["production_enabled"] == 0
    with pytest.raises(BroadQaExternalDataError, match="target 已存在"):
        publish_normalization_successor_training_protocol(
            run_root=tmp_path,
            opencc_source_pack_dir=opencc,
            icu_source_pack_dir=icu,
            target_dir=target,
        )


def test_learner_reader_opens_only_protocol_local_material(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """learner 用外部 manifest 身份回读时不调用任一 source reader。"""
    target, _opencc, _icu, report = _publish_synthetic(tmp_path, monkeypatch)

    def unexpected_source_read(*args, **kwargs):
        raise AssertionError("learner 不得读取 source pack")

    monkeypatch.setattr(
        protocol_module, "read_normalization_source_pack", unexpected_source_read)
    monkeypatch.setattr(
        protocol_module, "read_normalization_icu_source_pack",
        unexpected_source_read)
    restored = read_normalization_successor_learner_input(
        target,
        expected_manifest_sha256=report["manifest_sha256"],
    )
    assert restored[0]["learner_contract"]["learner_source_pack_read_count"] == 0


def test_reader_rejects_synchronized_material_and_manifest_tamper(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """同步重算文件摘要仍无法越过 learner 外部冻结 manifest 身份。"""
    target, _opencc, _icu, report = _publish_synthetic(tmp_path, monkeypatch)
    path = target / "train.observations.jsonl"
    lines = path.read_bytes().splitlines(keepends=True)
    value = json.loads(lines[0])
    value["expected_output"] += "改"
    lines[0] = canonical_json_line(value)
    path.write_bytes(b"".join(lines))
    manifest_path = target / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    file_record = next(
        item for item in manifest["files"]
        if item["relative_path"] == path.name)
    file_record["bytes"] = path.stat().st_size
    file_record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_bytes(canonical_json_line(manifest))
    with pytest.raises(BroadQaExternalDataError, match="manifest identity 漂移"):
        read_normalization_successor_learner_input(
            target,
            expected_manifest_sha256=report["manifest_sha256"],
        )


def test_auditor_rejects_source_rederivation_mismatch(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """协议发布后来源字节改变时 auditor 从来源重派生并失败关闭。"""
    target, opencc, icu, _report = _publish_synthetic(tmp_path, monkeypatch)
    phrase_path = opencc / "dictionary" / "TSPhrases.txt"
    phrase_path.write_bytes(phrase_path.read_bytes().replace(
        "锺馗".encode(), "钟馗".encode()))
    with pytest.raises(BroadQaExternalDataError, match="manifest identity 漂移"):
        read_normalization_successor_training_protocol(
            target,
            opencc_source_pack_dir=opencc,
            icu_source_pack_dir=icu,
        )


def test_publisher_rejects_source_manifest_identity_mismatch(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """任一训练来源不是冻结 manifest 时不得物化 learner 输入。"""
    _target, opencc, icu, _report = _publish_synthetic(tmp_path, monkeypatch)
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    monkeypatch.setattr(
        protocol_module,
        "read_normalization_source_pack",
        lambda path: {"manifest_sha256": "f" * 64},
    )
    with pytest.raises(BroadQaExternalDataError, match="manifest identity 漂移"):
        publish_normalization_successor_training_protocol(
            run_root=tmp_path,
            opencc_source_pack_dir=opencc,
            icu_source_pack_dir=icu,
            target_dir=replacement / "protocol",
        )


def test_official_training_source_census_is_frozen() -> None:
    """显式 K 盘 fixture 的两源 census 与协议库存保持逐项一致。"""
    configured = os.environ.get(
        "PURE_INTEGER_AI_NORMALIZATION_TRAIN_SOURCE_ROOT")
    if not configured:
        pytest.skip("official normalization training source fixture is unavailable")
    root = Path(configured)
    opencc = root / "normalization-dependency-source-pack-v1"
    icu = root / "normalization-icu-source-pack-v1"
    if not opencc.is_dir() or not icu.is_dir():
        pytest.skip("official normalization training source fixture is incomplete")
    observations, groups, contexts, work, summary = (
        protocol_module._derive_from_sources(
            opencc_source_pack_dir=opencc,
            icu_source_pack_dir=icu,
        ))
    assert (len(observations), len(groups), len(contexts), len(work)) == (
        8_567, 5_563, 1_253, 15_383)
    assert summary["source_observation_counts"] == {
        OPENCC_SOURCE_POLICY_SCOPE: 4_390,
        ICU_SOURCE_POLICY_SCOPE: 4_177,
    }
    assert summary["group_kind_counts"] == {
        "CROSS_SOURCE_CONSENSUS": 2_983,
        "SINGLE_SOURCE": 2_559,
        "SOURCE_POLICY_CONFLICT": 21,
    }
    assert summary["context_qualification_counts"] == {
        f"{OPENCC_SOURCE_POLICY_SCOPE}:SOURCE_REPLAY_OVERRIDE": 258,
        f"{OPENCC_SOURCE_POLICY_SCOPE}:SOURCE_REPLAY_SUPPORT": 19,
        f"{ICU_SOURCE_POLICY_SCOPE}:SOURCE_REPLAY_OVERRIDE": 169,
        f"{ICU_SOURCE_POLICY_SCOPE}:SOURCE_REPLAY_SUPPORT": 807,
    }
