"""Recovery-v5 TRAIN-only successor simulation 测试。"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_recovery_v5_successor_simulation as simulation_module,
)
from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_recovery_v5_successor_simulation_records as records_module,
)
from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_failure_profile import (
    NORMALIZATION_RECOVERY_V5_FAILURE_PROFILE_KIND,
    NORMALIZATION_RECOVERY_V5_FAILURE_PROFILE_STATUS,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_learning_records import (
    derive_normalization_recovery_v5_learning_outputs,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_phrase_runtime import (
    compile_normalization_recovery_v5_phrase_program,
    execute_normalization_recovery_v5_phrase_program,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_successor_simulation import (
    publish_normalization_recovery_v5_successor_simulation,
    read_normalization_recovery_v5_successor_simulation,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_successor_simulation_records import (
    SUCCESSOR_STRATEGIES,
    derive_normalization_recovery_v5_successor_simulation,
    simulate_normalization_recovery_v5_successor_strategy,
)
from test_ph2_broad_qa_normalization_recovery_v5_failure_profile import (
    _profile_material,
)
from test_ph2_broad_qa_normalization_recovery_v5_learner import _material


def _sha(value: str) -> str:
    """返回 synthetic artifact identity。"""
    return hashlib.sha256(value.encode()).hexdigest()


def _inputs():
    """构造 protocol、sealed audit/profile manifest 与期望输出。"""
    material, audit_sha, audit_manifest, profile = _profile_material()
    profile_sha = _sha("v5-successor-profile")
    profile_manifest = {
        "artifact_kind": NORMALIZATION_RECOVERY_V5_FAILURE_PROFILE_KIND,
        "formal_run_count": 0,
        "mastery_claimed": 0,
        "production_enabled": 0,
        "protocol_manifest_sha256": material[0]["manifest_sha256"],
        "selection_or_threshold_changed": 0,
        "status": NORMALIZATION_RECOVERY_V5_FAILURE_PROFILE_STATUS,
        "summary": profile[3],
        "training_audit_manifest_sha256": audit_sha,
    }
    result = derive_normalization_recovery_v5_successor_simulation(
        protocol_manifest=material[0],
        observations=material[1],
        fragments=material[2],
        audit_manifest_sha256=audit_sha,
        audit_manifest=audit_manifest,
        profile_manifest_sha256=profile_sha,
        profile_manifest=profile_manifest,
    )
    return material, audit_sha, audit_manifest, profile_sha, profile_manifest, result


def test_v5_successor_relearns_once_per_family_and_keeps_fixed_denominator(
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """四策略必须共享四次 LOSO 重学，且 case 不保存 held-out surface。"""
    original = records_module.derive_normalization_recovery_v5_loso_execution
    calls = []

    def counted(**kwargs):
        """记录 LOSO 调用并转发真实 pure derivation。"""
        calls.append(kwargs["held_out_source_family"])
        return original(**kwargs)

    monkeypatch.setattr(
        records_module,
        "derive_normalization_recovery_v5_loso_execution",
        counted,
    )
    material, _audit_sha, _audit, _profile_sha, _profile, result = _inputs()
    cases, families, strategies, summary = result
    assert len(calls) == 4
    assert len(set(calls)) == 4
    assert len(cases) == len(material[1]) * len(SUCCESSOR_STRATEGIES)
    assert len(families) == 4 * len(SUCCESSOR_STRATEGIES)
    assert len(strategies) == len(SUCCESSOR_STRATEGIES)
    assert summary["loso_relearn_count"] == 4
    assert summary["strategy_replay_count"] == 16
    assert summary["selection_label_read_count"] == 0
    assert all("input_text" not in item and "expected_output" not in item
               for item in cases)
    assert all(set(item["bucket_outcome_counts"]) == {
        f"{bucket}:{outcome}"
        for bucket in (
            "IDENTITY", "CHARACTER_LOCAL", "WHOLE_INPUT_EQUAL_LENGTH",
            "WHOLE_INPUT_VARIABLE_LENGTH", "CONTEXT_HUNK")
        for outcome in ("EXACT", "UNKNOWN", "WRONG")}
        for item in families)


def _local_program_material():
    """构造只含 target local rule 的 synthetic disabled program。"""
    protocol_sha = _sha("v5-successor-local")
    material = _material(protocol_sha)
    outputs, _summary, _emissions = derive_normalization_recovery_v5_learning_outputs(
        protocol_manifest=material[0],
        observations=material[1],
        fragments=material[2],
        groups=material[3],
        work=material[4],
    )
    local_rules = tuple(
        item for item in outputs["target-phrase-rules.jsonl"]
        if item["fragment_kind"] in {"EDIT_CORE", "CONTEXT_HUNK"})
    assert local_rules
    rule_ids = {str(item["rule_id"]) for item in local_rules}
    defeaters = tuple(
        item for item in outputs["defeaters.jsonl"]
        if item["rule_id"] in rule_ids)
    overlap = tuple(
        item for item in outputs["target-overlap-index.jsonl"]
        if item["rule_id"] in rule_ids)
    program = compile_normalization_recovery_v5_phrase_program(
        rule_pack_manifest_sha256=_sha("v5-successor-local-pack"),
        target_phrase_rules=local_rules,
        source_phrase_rules=(),
        defeaters=defeaters,
        target_overlap_index=overlap,
        source_overlap_index=(),
    )
    return material, outputs, local_rules, program


def test_v5_successor_positive_context_and_atomic_obligation_are_fail_closed(
        ) -> None:
    """正 applicability 失配与未闭合 obligation 都必须整句回退。"""
    material, outputs, local_rules, program = _local_program_material()
    rule = local_rules[0]
    evidence_id = str(rule["positive_evidence_ids"][0])
    evidence = next(item for item in outputs["evidence.jsonl"]
                    if item["evidence_id"] == evidence_id)
    support = next(item for item in material[1]
                   if item["observation_id"] == evidence["observation_id"])
    baseline = execute_normalization_recovery_v5_phrase_program(
        program,
        str(support["input_text"]),
        source_family=str(support["source_family"]),
        structure_tokens=tuple(support["structure_tokens"]),
    )
    positive = simulate_normalization_recovery_v5_successor_strategy(
        strategy="LOCAL_POSITIVE_CONTEXT",
        observation=support,
        baseline_result=baseline,
        program=program,
        outputs=outputs,
        training_groups=material[3],
        held_out_source_family=str(support["source_family"]),
    )
    assert positive["decision_reason_counts"].get(
        "LOCAL_POSITIVE_CONTEXT_COMMIT", 0) > 0

    changed = {
        **support,
        "input_text": "前" + str(rule["input_text"]) + "後",
        "output_text": "前" + str(rule["input_text"]) + "後",
        "structure_tokens": [],
    }
    changed_baseline = execute_normalization_recovery_v5_phrase_program(
        program,
        str(changed["input_text"]),
        source_family=str(support["source_family"]),
    )
    guarded = simulate_normalization_recovery_v5_successor_strategy(
        strategy="LOCAL_POSITIVE_CONTEXT",
        observation=changed,
        baseline_result=changed_baseline,
        program=program,
        outputs=outputs,
        training_groups=material[3],
        held_out_source_family=str(support["source_family"]),
    )
    assert guarded["decision_reason_counts"].get(
        "POSITIVE_CONTEXT_MISS", 0) > 0

    structured = {
        **support,
        "input_text": str(rule["input_text"]),
        "output_text": str(rule["input_text"]),
        "structure_tokens": ["OPAQUE_STRUCTURE"],
    }
    structured_baseline = execute_normalization_recovery_v5_phrase_program(
        program,
        str(structured["input_text"]),
        source_family=str(support["source_family"]),
        structure_tokens=("OPAQUE_STRUCTURE",),
    )
    atomic = simulate_normalization_recovery_v5_successor_strategy(
        strategy="LOCAL_ATOMIC_COVERAGE",
        observation=structured,
        baseline_result=structured_baseline,
        program=program,
        outputs=outputs,
        training_groups=material[3],
        held_out_source_family=str(support["source_family"]),
    )
    assert atomic["output_text"] == structured["input_text"]
    assert atomic["open_obligation_count"] > 0
    assert atomic["decision_reason_counts"] == {"ATOMIC_OBLIGATION_OPEN": 1}


def _install_readers(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ):
    """把 publisher readers 固定到 synthetic TRAIN 与两个 manifest。"""
    material, audit_sha, audit_manifest, profile_sha, profile_manifest, result = (
        _inputs())
    protocol_dir = tmp_path / "protocol"
    audit_dir = tmp_path / "audit"
    profile_dir = tmp_path / "profile"
    protocol_dir.mkdir()
    audit_dir.mkdir()
    profile_dir.mkdir()
    monkeypatch.setattr(
        simulation_module,
        "read_normalization_recovery_v5_learner_input",
        lambda *args, **kwargs: material,
    )

    def manifest_reader(_root, *, expected_manifest_sha256, label):
        """按外部 SHA 返回对应 synthetic sealed manifest。"""
        if expected_manifest_sha256 == audit_sha and label == "training audit":
            return audit_manifest
        if expected_manifest_sha256 == profile_sha and label == "failure profile":
            return profile_manifest
        raise AssertionError("unexpected manifest request")

    monkeypatch.setattr(
        simulation_module, "_read_manifest_only", manifest_reader)
    monkeypatch.setattr(
        simulation_module, "_require_k_root",
        lambda value: Path(value).resolve())
    return (protocol_dir, audit_dir, profile_dir, material, audit_sha,
            profile_sha, result)


def test_v5_successor_publishes_and_strictly_rederives(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """simulation 必须不可覆盖、manifest-last 且严格回读拒绝篡改。"""
    values = _install_readers(tmp_path, monkeypatch)
    protocol_dir, audit_dir, profile_dir, material, audit_sha, profile_sha, _ = (
        values)
    target = tmp_path / "simulation"
    report = publish_normalization_recovery_v5_successor_simulation(
        run_root=tmp_path,
        protocol_dir=protocol_dir,
        expected_protocol_manifest_sha256=str(material[0]["manifest_sha256"]),
        audit_dir=audit_dir,
        expected_audit_manifest_sha256=audit_sha,
        profile_dir=profile_dir,
        expected_profile_manifest_sha256=profile_sha,
        target_dir=target,
    )
    assert report["status"] == (
        "TRAIN_ONLY_SIMULATION_COMPLETE_NOT_SELECTION_NOT_EVALUATION")
    assert report["training_audit_non_manifest_read_count"] == 0
    assert report["failure_profile_non_manifest_read_count"] == 0
    restored, outputs = read_normalization_recovery_v5_successor_simulation(
        target,
        protocol_dir=protocol_dir,
        expected_protocol_manifest_sha256=str(material[0]["manifest_sha256"]),
        audit_dir=audit_dir,
        expected_audit_manifest_sha256=audit_sha,
        profile_dir=profile_dir,
        expected_profile_manifest_sha256=profile_sha,
        expected_simulation_manifest_sha256=str(report["manifest_sha256"]),
    )
    assert restored["manifest_sha256"] == report["manifest_sha256"]
    assert len(outputs["strategy-results.jsonl"]) == 4
    with pytest.raises(BroadQaExternalDataError, match="target 已存在"):
        publish_normalization_recovery_v5_successor_simulation(
            run_root=tmp_path,
            protocol_dir=protocol_dir,
            expected_protocol_manifest_sha256=str(
                material[0]["manifest_sha256"]),
            audit_dir=audit_dir,
            expected_audit_manifest_sha256=audit_sha,
            profile_dir=profile_dir,
            expected_profile_manifest_sha256=profile_sha,
            target_dir=target,
        )
    path = target / "strategy-results.jsonl"
    path.write_bytes(path.read_bytes() + b"{}\n")
    with pytest.raises(BroadQaExternalDataError, match="重派生漂移"):
        read_normalization_recovery_v5_successor_simulation(
            target,
            protocol_dir=protocol_dir,
            expected_protocol_manifest_sha256=str(
                material[0]["manifest_sha256"]),
            audit_dir=audit_dir,
            expected_audit_manifest_sha256=audit_sha,
            profile_dir=profile_dir,
            expected_profile_manifest_sha256=profile_sha,
            expected_simulation_manifest_sha256=str(report["manifest_sha256"]),
        )


def test_v5_successor_rejects_non_k_root_before_write(tmp_path: Path) -> None:
    """正式 publisher 不得把 simulation artifact 回退到 D 盘。"""
    with pytest.raises(BroadQaExternalDataError, match="K 盘"):
        publish_normalization_recovery_v5_successor_simulation(
            run_root=tmp_path,
            protocol_dir=tmp_path / "protocol",
            expected_protocol_manifest_sha256="a" * 64,
            audit_dir=tmp_path / "audit",
            expected_audit_manifest_sha256="b" * 64,
            profile_dir=tmp_path / "profile",
            expected_profile_manifest_sha256="c" * 64,
            target_dir=tmp_path / "simulation",
        )
    assert not (tmp_path / "simulation").exists()
