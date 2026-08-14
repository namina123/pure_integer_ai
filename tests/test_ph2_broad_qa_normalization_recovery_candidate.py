"""Normalization recovery transfer profile 与双解释器测试。"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_candidate_clone import (
    NormalizationRecoveryPhraseOverride,
    RECOVERY_TRANSFER_REGION_SCOPE,
    compile_normalization_recovery_candidate,
    execute_normalization_recovery_candidate,
    reference_normalization_recovery_candidate,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_candidate_profile import (
    derive_normalization_recovery_training_queries,
    profile_normalization_recovery_candidate,
    publish_normalization_recovery_candidate_profile,
)
from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_recovery_candidate_profile as profile_module,
)
from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_recovery_candidate_profile_reader as profile_reader_module,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_candidate_profile_reader import (
    read_normalization_recovery_candidate_profile,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_evaluation_protocol import (
    NORMALIZATION_RECOVERY_EVALUATION_STATUS,
    NORMALIZATION_RECOVERY_TARGET_POLICY_SCOPE,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_learning_records import (
    derive_normalization_recovery_learning_outputs,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_training_protocol import (
    read_normalization_recovery_learner_input,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_training_records import (
    ICU_SOURCE_POLICY_SCOPE,
    OPENCC_SOURCE_POLICY_SCOPE,
    RECOVERY_TARGET_POLICY_SCOPE,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line
from test_ph2_broad_qa_normalization_recovery_training import _publish_protocol


def _sha(value: str) -> str:
    """构造 synthetic manifest identity。"""
    return hashlib.sha256(value.encode()).hexdigest()


def _candidate_material(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ):
    """从真实 recovery synthetic protocol 派生 pack 输出与 transfer manifest。"""
    protocol, _sources, report, _calls = _publish_protocol(
        tmp_path, monkeypatch)
    values = read_normalization_recovery_learner_input(
        protocol, expected_manifest_sha256=report["manifest_sha256"])
    outputs, summary = derive_normalization_recovery_learning_outputs(
        protocol_manifest=values[0],
        roster=values[1],
        observations=values[2],
        groups=values[3],
        compositions=values[4],
        work=values[5],
    )
    pack = {
        "manifest_sha256": _sha("synthetic recovery pack"),
        "mastery_claimed": 0,
        "production_enabled": 0,
        "runtime_state": "LEARNED_PACK_DISABLED",
        "summary": summary,
    }
    evaluation = {
        "manifest_sha256": _sha("synthetic Firefox evaluation protocol"),
        "status": NORMALIZATION_RECOVERY_EVALUATION_STATUS,
        "target_policy_scope": NORMALIZATION_RECOVERY_TARGET_POLICY_SCOPE,
    }
    return evaluation, pack, outputs


def _assert_equal(program, text: str, policy: str, region: str = ""):
    """要求 indexed 与线性 reference 逐字段相等。"""
    indexed = execute_normalization_recovery_candidate(
        program, text, policy_scope=policy, regional_scope=region)
    reference = reference_normalization_recovery_candidate(
        program, text, policy_scope=policy, regional_scope=region)
    assert indexed == reference
    return indexed


def test_transfer_profile_keeps_authority_and_firefox_scope_separate(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """Firefox projection 不得改写 learner rule 的原 authority scope。"""
    evaluation, pack, outputs = _candidate_material(tmp_path, monkeypatch)
    program = compile_normalization_recovery_candidate(
        evaluation_protocol_manifest=evaluation,
        rule_pack_manifest=pack,
        outputs=outputs,
    )
    profile = program.transfer_profile
    assert profile.authority_policy_scope == RECOVERY_TARGET_POLICY_SCOPE
    assert profile.candidate_target_policy_scope == (
        NORMALIZATION_RECOVERY_TARGET_POLICY_SCOPE)
    assert profile.rule_pack_manifest_sha256 == pack["manifest_sha256"]
    assert profile.evaluation_protocol_manifest_sha256 == (
        evaluation["manifest_sha256"])
    assert program.production_enabled == 0
    assert len(program.generic_rules) == 3
    assert len(program.regional_rules) == 1
    assert len(program.source_replays) == 16
    assert len(program.phrase_overrides) == 1
    assert len(program.conflicts) == 1


def test_target_projection_region_phrase_composition_and_conflict_are_scoped(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """target exact/character、region 与 conflict 不得互相越权。"""
    evaluation, pack, outputs = _candidate_material(tmp_path, monkeypatch)
    program = compile_normalization_recovery_candidate(
        evaluation_protocol_manifest=evaluation,
        rule_pack_manifest=pack,
        outputs=outputs,
    )
    generic = _assert_equal(
        program, "甲", NORMALIZATION_RECOVERY_TARGET_POLICY_SCOPE,
        RECOVERY_TRANSFER_REGION_SCOPE)
    assert generic.output_text == "一"
    assert generic.projection_used == 1 and generic.transfer_profile_id

    missing_region = _assert_equal(
        program, "乙", NORMALIZATION_RECOVERY_TARGET_POLICY_SCOPE)
    assert missing_region.output_text == "乙"
    assert missing_region.scope_mismatch == 1
    assert missing_region.projection_used == 0

    regional = _assert_equal(
        program, "乙", NORMALIZATION_RECOVERY_TARGET_POLICY_SCOPE,
        RECOVERY_TRANSFER_REGION_SCOPE)
    assert regional.output_text == "二"
    assert regional.target_rule_ids

    composition = _assert_equal(
        program, "舊詞", NORMALIZATION_RECOVERY_TARGET_POLICY_SCOPE,
        RECOVERY_TRANSFER_REGION_SCOPE)
    assert composition.output_text == "旧词"
    assert len(composition.target_rule_ids) == 2

    conflict = _assert_equal(
        program, "丁", NORMALIZATION_RECOVERY_TARGET_POLICY_SCOPE,
        RECOVERY_TRANSFER_REGION_SCOPE)
    assert conflict.output_text == "丁"
    assert conflict.conflict_ids
    assert conflict.unscoped_conflict_blocked == 0

    unscoped = _assert_equal(program, "丁", "")
    assert unscoped.output_text == "丁"
    assert unscoped.scope_mismatch == 1
    assert unscoped.unscoped_conflict_blocked == 1


def test_source_policy_phrase_override_and_replay_do_not_upgrade_target(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """source phrase override 只在原 policy 下执行，target 仍用 authority composition。"""
    evaluation, pack, outputs = _candidate_material(tmp_path, monkeypatch)
    program = compile_normalization_recovery_candidate(
        evaluation_protocol_manifest=evaluation,
        rule_pack_manifest=pack,
        outputs=outputs,
    )
    source = _assert_equal(program, "詞舊", OPENCC_SOURCE_POLICY_SCOPE)
    assert source.output_text == "词故"
    assert source.phrase_rule_ids and source.source_evidence_ids

    other_source = _assert_equal(program, "詞舊", ICU_SOURCE_POLICY_SCOPE)
    assert other_source.output_text == "詞舊"
    assert not other_source.phrase_rule_ids

    target = _assert_equal(
        program, "詞舊", NORMALIZATION_RECOVERY_TARGET_POLICY_SCOPE,
        RECOVERY_TRANSFER_REGION_SCOPE)
    assert target.output_text == "词旧"
    assert not target.phrase_rule_ids


def test_compile_rejects_policy_rename_enabled_pack_and_phrase_drift(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """target 改名、production 启用或 phrase/replay 不一致都失败关闭。"""
    evaluation, pack, outputs = _candidate_material(tmp_path, monkeypatch)
    wrong_evaluation = {
        **evaluation,
        "target_policy_scope": RECOVERY_TARGET_POLICY_SCOPE,
    }
    with pytest.raises(BroadQaExternalDataError, match="pack/evaluation"):
        compile_normalization_recovery_candidate(
            evaluation_protocol_manifest=wrong_evaluation,
            rule_pack_manifest=pack,
            outputs=outputs,
        )
    with pytest.raises(BroadQaExternalDataError, match="pack/evaluation"):
        compile_normalization_recovery_candidate(
            evaluation_protocol_manifest=evaluation,
            rule_pack_manifest={**pack, "production_enabled": 1},
            outputs=outputs,
        )
    phrase = dict(outputs["source-phrase-rules.jsonl"][0])
    phrase["output_text"] = "错误"
    drifted = {
        **outputs,
        "source-phrase-rules.jsonl": (phrase,),
    }
    with pytest.raises(BroadQaExternalDataError, match="未闭合"):
        compile_normalization_recovery_candidate(
            evaluation_protocol_manifest=evaluation,
            rule_pack_manifest=pack,
            outputs=drifted,
        )


def test_surface_identity_phrase_can_still_refute_changed_base() -> None:
    """表面 identity 只要反驳非 identity base，仍是可执行 override。"""
    value = NormalizationRecoveryPhraseOverride(
        source_policy_scope=OPENCC_SOURCE_POLICY_SCOPE,
        input_text="乾乾",
        base_output="干干",
        output_text="乾乾",
        rule_id=_sha("identity phrase rule"),
        support_evidence_id=_sha("identity phrase support"),
        refute_evidence_id=_sha("identity phrase refute"),
    )
    assert value.input_text == value.output_text
    assert value.base_output != value.output_text
    with pytest.raises(BroadQaExternalDataError, match="phrase override"):
        NormalizationRecoveryPhraseOverride(
            source_policy_scope=OPENCC_SOURCE_POLICY_SCOPE,
            input_text="乾乾",
            base_output="乾乾",
            output_text="乾乾",
            rule_id=_sha("invalid identity phrase rule"),
            support_evidence_id=_sha("invalid identity phrase support"),
            refute_evidence_id=_sha("invalid identity phrase refute"),
        )


def test_train_only_profile_covers_all_scopes_and_matches_reference(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """profile roster 覆盖全部运行域，双解释器结果逐查询相同。"""
    evaluation, pack, outputs = _candidate_material(tmp_path, monkeypatch)
    program = compile_normalization_recovery_candidate(
        evaluation_protocol_manifest=evaluation,
        rule_pack_manifest=pack,
        outputs=outputs,
    )
    queries = derive_normalization_recovery_training_queries(program)
    assert len(queries) == 25
    report = profile_normalization_recovery_candidate(program)
    assert report["query_kind_counts"] == {
        "AUTHORITY_GENERIC": 3,
        "AUTHORITY_REGIONAL": 1,
        "SOURCE_REPLAY": 16,
        "TRANSFER_TARGET": 4,
        "UNSCOPED_CONFLICT": 1,
    }
    assert report["indexed"]["failure_count"] == 0
    assert report["reference"]["failure_count"] == 0
    assert report["reference"]["mismatch_count"] == 0
    assert report["indexed"]["result_sha256"] == (
        report["reference"]["result_sha256"])
    assert report["indexed_reference_result_bytes_equal"] == 1


def test_profile_publish_read_and_tamper_fail_closed(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """profile 不可覆盖，reader 重编译 identity 并拒绝同步改字段。"""
    evaluation, pack, outputs = _candidate_material(tmp_path, monkeypatch)
    evaluation_root = tmp_path / "evaluation"
    training_root = tmp_path / "training"
    pack_root = tmp_path / "pack"
    for path in (evaluation_root, training_root, pack_root):
        path.mkdir()
    monkeypatch.setattr(
        profile_module,
        "_require_k_root",
        lambda value: Path(value).resolve(),
    )
    monkeypatch.setattr(
        profile_module,
        "read_normalization_recovery_evaluation_manifest_only",
        lambda *args, **kwargs: evaluation,
    )
    monkeypatch.setattr(
        profile_module,
        "read_normalization_recovery_rule_pack",
        lambda *args, **kwargs: (pack, outputs),
    )
    monkeypatch.setattr(
        profile_reader_module,
        "read_normalization_recovery_evaluation_manifest_only",
        lambda *args, **kwargs: evaluation,
    )
    monkeypatch.setattr(
        profile_reader_module,
        "read_normalization_recovery_rule_pack",
        lambda *args, **kwargs: (pack, outputs),
    )
    target = tmp_path / "profile"
    training_sha = _sha("synthetic training protocol")
    report = publish_normalization_recovery_candidate_profile(
        run_root=tmp_path,
        evaluation_protocol_dir=evaluation_root,
        expected_evaluation_manifest_sha256=evaluation["manifest_sha256"],
        training_protocol_dir=training_root,
        expected_training_manifest_sha256=training_sha,
        rule_pack_dir=pack_root,
        expected_rule_pack_manifest_sha256=pack["manifest_sha256"],
        target_dir=target,
    )
    stored = read_normalization_recovery_candidate_profile(
        target,
        expected_profile_sha256=report["profile_sha256"],
        evaluation_protocol_dir=evaluation_root,
        expected_evaluation_manifest_sha256=evaluation["manifest_sha256"],
        training_protocol_dir=training_root,
        expected_training_manifest_sha256=training_sha,
        rule_pack_dir=pack_root,
        expected_rule_pack_manifest_sha256=pack["manifest_sha256"],
    )
    assert stored["candidate_program_sha256"] == (
        report["candidate_program_sha256"])
    with pytest.raises(BroadQaExternalDataError, match="target 已存在"):
        publish_normalization_recovery_candidate_profile(
            run_root=tmp_path,
            evaluation_protocol_dir=evaluation_root,
            expected_evaluation_manifest_sha256=evaluation["manifest_sha256"],
            training_protocol_dir=training_root,
            expected_training_manifest_sha256=training_sha,
            rule_pack_dir=pack_root,
            expected_rule_pack_manifest_sha256=pack["manifest_sha256"],
            target_dir=target,
        )

    profile_path = target / "profile.json"
    drifted = json.loads(profile_path.read_bytes())
    drifted["mastery_claimed"] = 1
    encoded = canonical_json_line(drifted)
    profile_path.write_bytes(encoded)
    with pytest.raises(BroadQaExternalDataError, match="固定边界"):
        read_normalization_recovery_candidate_profile(
            target,
            expected_profile_sha256=hashlib.sha256(encoded).hexdigest(),
            evaluation_protocol_dir=evaluation_root,
            expected_evaluation_manifest_sha256=evaluation["manifest_sha256"],
            training_protocol_dir=training_root,
            expected_training_manifest_sha256=training_sha,
            rule_pack_dir=pack_root,
            expected_rule_pack_manifest_sha256=pack["manifest_sha256"],
        )


def test_official_recovery_candidate_profile_identity_is_frozen() -> None:
    """显式 K 盘 fixture 的 program、query 与双解释器结果保持冻结。"""
    configured = os.environ.get(
        "PURE_INTEGER_AI_NORMALIZATION_RECOVERY_TRAIN_ROOT")
    if not configured:
        pytest.skip("official recovery candidate fixture is unavailable")
    root = Path(configured)
    profile = root / "profiles" / "normalization-recovery-candidate-training-v2"
    required = (
        profile,
        root / "normalization-recovery-evaluation-protocol-v2",
        root / "normalization-recovery-training-protocol-v2",
        root / "normalization-recovery-rule-pack-v2",
    )
    if any(not path.is_dir() for path in required):
        pytest.skip("official recovery candidate fixture is incomplete")
    report = read_normalization_recovery_candidate_profile(
        profile,
        expected_profile_sha256=(
            "cbeea89cda730ab9c14a96f8337bdc7a6faa5f887bd029a50680431f5ec7af46"),
        evaluation_protocol_dir=required[1],
        expected_evaluation_manifest_sha256=(
            "9a1aa10f2b4285e74e62a8a265967caeefbb31779faf7af2bf8c6c29f15dfb70"),
        training_protocol_dir=required[2],
        expected_training_manifest_sha256=(
            "315c2a34a026d42e7d4dedc3126acda5c24f0cca5cb49d76a9cc798de7760af9"),
        rule_pack_dir=required[3],
        expected_rule_pack_manifest_sha256=(
            "a676340af3717c078069c4c80535df66f13af43b675cdb69000d281101fcf21c"),
    )
    values = report["profile"]
    assert report["candidate_program_sha256"] == (
        "0bb299c7e9360b2d748f0cb7a49f1614330fb0c342800fbc4d613aa860d26212")
    assert report["transfer_profile_sha256"] == (
        "c95e5845fa9bd8462556afb7d8d51e137b809c653aedebcaaf76d79cbd79f3b1")
    assert values["query_count"] == 33_637
    assert values["query_roster_sha256"] == (
        "051d4f3aa169c533e35bedb70ec4b1f8513313c9cf07887e9e2ad11be0f5df9f")
    assert values["indexed"]["result_sha256"] == (
        "475030a5be3b5a1bed84f92bacce4ff285cf9fbcf2f4015a1167b98908e53280")
    assert values["reference"]["result_sha256"] == (
        values["indexed"]["result_sha256"])
    assert values["indexed"]["failure_count"] == 0
    assert values["reference"]["failure_count"] == 0
    assert values["reference"]["mismatch_count"] == 0
    assert report["evaluation_payload_read_count"] == 0
    assert report["reserve_payload_read_count"] == 0
    assert report["production_enabled"] == 0
    assert report["mastery_claimed"] == 0
