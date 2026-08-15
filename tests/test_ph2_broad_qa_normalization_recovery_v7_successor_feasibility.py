"""覆盖 recovery-v7 三项 TRAIN-only successor feasibility。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_recovery_v7_successor_feasibility as artifact,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_training_records import (
    V5_SOURCE_POLICY_BY_FAMILY,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_successor_feasibility_records import (
    derive_context_scoped_local_projections,
    derive_identity_inputs,
    derive_source_policy_replay_projections,
    derive_variable_structure_projections,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
)


GODOT = "GODOT_ENGINE_PROJECT"
THUNDERBIRD = "THUNDERBIRD_PROJECT"
VSCODE = "MICROSOFT_VSCODE_PROJECT"


def _id(value: str) -> str:
    """形成测试 identity。"""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _observations() -> tuple[dict[str, object], ...]:
    """构造同 structure signature 的双 family variable observations。"""
    return (
        {
            "equal_length": 0,
            "identity_preservation": 0,
            "input_text": "開啟 %s 選項",
            "observation_id": _id("observation-a"),
            "output_text": "打开 %s",
            "source_family": GODOT,
            "structure_tokens": ["%s"],
        },
        {
            "equal_length": 0,
            "identity_preservation": 0,
            "input_text": "儲存 %s 設定",
            "observation_id": _id("observation-b"),
            "output_text": "保存 %s",
            "source_family": THUNDERBIRD,
            "structure_tokens": ["%s"],
        },
        {
            "equal_length": 0,
            "identity_preservation": 0,
            "input_text": "無結構繁體",
            "observation_id": _id("observation-c"),
            "output_text": "无结构",
            "source_family": VSCODE,
            "structure_tokens": [],
        },
    )


def _target_rules() -> tuple[dict[str, object], ...]:
    """构造一条具双 family SUPPORT 与 defeater 的 target local rule。"""
    return ({
        "candidate_id": _id("candidate-local"),
        "candidate_scope_kind": "TARGET_CROSS_FAMILY",
        "defeater_ids": [_id("defeater-local")],
        "fragment_kind": "EDIT_CORE",
        "input_text": "視窗",
        "output_text": "窗口",
        "positive_evidence_ids": [_id("evidence-a"), _id("evidence-b")],
        "rule_class": "EDIT_CORE",
        "rule_id": _id("rule-local"),
        "source_families": [GODOT, THUNDERBIRD],
    },)


def _evidence() -> tuple[dict[str, object], ...]:
    """构造两个不同 family 的正上下文 evidence。"""
    return (
        {
            "candidate_id": _id("candidate-local"),
            "context_signature": {
                "context_signature_id": _id("context-a"),
                "left_boundary": 0,
                "left_context": "開啟",
                "right_boundary": 1,
                "right_context": "",
            },
            "evidence_id": _id("evidence-a"),
            "source_family": GODOT,
            "stance": "SUPPORT",
        },
        {
            "candidate_id": _id("candidate-local"),
            "context_signature": {
                "context_signature_id": _id("context-b"),
                "left_boundary": 0,
                "left_context": "關閉",
                "right_boundary": 1,
                "right_context": "",
            },
            "evidence_id": _id("evidence-b"),
            "source_family": THUNDERBIRD,
            "stance": "SUPPORT",
        },
    )


def _identity_observations() -> tuple[dict[str, object], ...]:
    """构造 exact-input identity hard veto。"""
    return ({"input_text": "視窗", "output_text": "視窗"},)


def _variant(
        output: str,
        family: str,
        suffix: str,
        ) -> dict[str, object]:
    """构造一条 source-policy conflict variant。"""
    return {
        "fragment_ids": [_id(f"fragment-{suffix}")],
        "output_text": output,
        "source_families": [family],
        "source_policy_scopes": [V5_SOURCE_POLICY_BY_FAMILY[family]],
        "support_count": 1,
    }


def _conflicts() -> tuple[dict[str, object], ...]:
    """构造一条可 replay 与一条同 family 多输出冲突。"""
    return (
        {
            "conflict_id": _id("conflict-replayable"),
            "conflict_kind": "TRAIN_OUTPUT_CONFLICT",
            "fragment_kind": "EDIT_CORE",
            "input_text": "視窗",
            "variants": [
                _variant("窗口", GODOT, "a"),
                _variant("视窗", THUNDERBIRD, "b"),
            ],
        },
        {
            "conflict_id": _id("conflict-context"),
            "conflict_kind": "TRAIN_OUTPUT_CONFLICT",
            "fragment_kind": "WHOLE_INPUT",
            "input_text": "偏好",
            "variants": [
                _variant("首选项", GODOT, "c"),
                _variant("偏好设置", GODOT, "d"),
                _variant("偏好", VSCODE, "e"),
            ],
        },
    )


def _jsonl(path: Path, values: tuple[dict[str, object], ...]) -> dict[str, object]:
    """写入测试 JSONL 并返回物理承诺。"""
    payload = b"".join(canonical_json_line(value) for value in values)
    path.write_bytes(payload)
    return {
        "bytes": len(payload),
        "record_count": len(values),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _manifest(path: Path, value: dict[str, object]) -> str:
    """写入规范测试 manifest 并返回 SHA。"""
    payload = canonical_json_line(value)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _artifact_record(
        name: str,
        role: str,
        physical: dict[str, object],
        ) -> dict[str, object]:
    """构造 manifest file record。"""
    return {
        **physical,
        "relative_path": name,
        "role": role,
    }


def _input_tree(tmp_path: Path) -> dict[str, object]:
    """构造 publisher 所需四个 sealed synthetic input。"""
    protocol = tmp_path / "protocol"
    pack = tmp_path / "pack"
    audit = tmp_path / "audit"
    commitment = tmp_path / "commitment"
    for path in (protocol, pack, audit, commitment):
        path.mkdir()
    observation_file = _jsonl(
        protocol / "train.pair-observations.jsonl", _observations())
    protocol_value = {
        "artifact_kind": (
            "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V5_TRAINING_PROTOCOL_V1"),
        "evaluation_or_held_out_payload_read_count": 0,
        "files": [_artifact_record(
            "train.pair-observations.jsonl",
            "TRAIN_PAIR_OBSERVATIONS",
            observation_file,
        )],
        "mastery_claimed": 0,
        "production_enabled": 0,
        "status": "FROZEN_NOT_READ_NOT_LEARNED",
    }
    protocol_sha = _manifest(protocol / "manifest.json", protocol_value)
    pack_files = []
    for name, role, values in (
        ("target-phrase-rules.jsonl", "LEARNED_TARGET_PHRASE_RULES",
         _target_rules()),
        ("evidence.jsonl", "LEARNED_SCOPED_PHRASE_EVIDENCE", _evidence()),
        ("conflict-ledger.jsonl", "LEARNED_CONFLICT_LEDGER", _conflicts()),
        ("identity-observations.jsonl", "IDENTITY_PRESERVATION_AUDIT_BUCKET",
         _identity_observations()),
    ):
        pack_files.append(_artifact_record(
            name, role, _jsonl(pack / name, values)))
    pack_value = {
        "artifact_kind": "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V5_RULE_PACK_V1",
        "files": pack_files,
        "mastery_claimed": 0,
        "production_enabled": 0,
        "protocol_manifest_sha256": protocol_sha,
        "status": "FROZEN_NOT_EVALUATED_NOT_DEPLOYED",
    }
    pack_sha = _manifest(pack / "manifest.json", pack_value)
    audit_value = {
        "artifact_kind": (
            "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V6_TRAINING_AUDIT_V1"),
        "mastery_claimed": 0,
        "production_enabled": 0,
        "status": "TRAIN_ONLY_COMPLETE_NOT_FORMAL_NOT_DEPLOYED",
        "summary": {
            "audit_outcome": "FACILITY_PASS_CAPABILITY_PASS",
            "facility_failure_count": 0,
            "identity_false_change_count": 0,
        },
    }
    audit_sha = _manifest(audit / "manifest.json", audit_value)
    commitment_value = {
        "artifact_kind": (
            "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V7_EVALUATION_COMMITMENT_V1"),
        "denominator": {"label_blind": 1, "record_count": 3_656},
        "mastery_claimed": 0,
        "production_enabled": 0,
        "source_non_manifest_file_read_count": 0,
        "status": (
            "LABEL_BLIND_DENOMINATOR_AND_GATES_FROZEN_BEFORE_V7_LEARNER_CHANGE"),
        "training_source_read_count": 0,
    }
    commitment_sha = _manifest(
        commitment / "manifest.json", commitment_value)
    return {
        "audit": audit,
        "audit_sha": audit_sha,
        "commitment": commitment,
        "commitment_sha": commitment_sha,
        "pack": pack,
        "pack_sha": pack_sha,
        "protocol": protocol,
        "protocol_sha": protocol_sha,
    }


def test_three_successor_projections_keep_execution_disabled() -> None:
    """三项 projection 如实分离窄可行、正上下文和 partial replay。"""
    variable, variable_summary = derive_variable_structure_projections(
        _observations())
    source, source_summary, conflict_inputs = (
        derive_source_policy_replay_projections(_conflicts()))
    identity_inputs = derive_identity_inputs(_identity_observations())
    context, context_summary = derive_context_scoped_local_projections(
        target_rules=_target_rules(),
        evidence=_evidence(),
        identity_inputs=identity_inputs,
        conflict_inputs=conflict_inputs,
    )
    assert len(variable) == 1
    assert variable_summary["cross_family_signature_count"] == 1
    assert variable[0]["execution_allowed"] == 0
    assert context_summary["representation_feasible"] == 1
    assert context[0]["identity_veto_required"] == 1
    assert context[0]["conflict_veto_required"] == 1
    assert context[0]["atomic_whole_commit_required"] == 1
    assert all(item["execution_allowed"] == 0 for item in source)
    assert source_summary["replayable_conflict_count"] == 1
    assert source_summary[
        "context_or_source_identity_required_count"] == 1


def test_context_projection_rejects_missing_positive_evidence() -> None:
    """正上下文 evidence 缺一条时 projection fail closed。"""
    with pytest.raises(BroadQaExternalDataError, match="未闭合"):
        derive_context_scoped_local_projections(
            target_rules=_target_rules(),
            evidence=_evidence()[:1],
            identity_inputs=derive_identity_inputs(_identity_observations()),
            conflict_inputs=frozenset(),
        )


def test_feasibility_artifact_round_trip_reads_only_sealed_train_inputs(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """publisher/reader 从四个 manifest 与 TRAIN JSONL 严格重派生。"""
    inputs = _input_tree(tmp_path)
    monkeypatch.setattr(
        artifact, "V5_TRAINING_PROTOCOL_MANIFEST_SHA256",
        inputs["protocol_sha"])
    monkeypatch.setattr(
        artifact, "V5_RULE_PACK_MANIFEST_SHA256", inputs["pack_sha"])
    monkeypatch.setattr(
        artifact, "V6_TRAINING_AUDIT_MANIFEST_SHA256", inputs["audit_sha"])
    monkeypatch.setattr(
        artifact, "V7_EVALUATION_COMMITMENT_MANIFEST_SHA256",
        inputs["commitment_sha"])
    monkeypatch.setattr(artifact, "_require_k_root", lambda value: Path(value))
    target = tmp_path / "feasibility"
    published = artifact.publish_normalization_recovery_v7_successor_feasibility(
        run_root=tmp_path,
        training_protocol_dir=inputs["protocol"],
        rule_pack_dir=inputs["pack"],
        training_audit_dir=inputs["audit"],
        evaluation_commitment_dir=inputs["commitment"],
        target_dir=target,
    )
    manifest, outputs = (
        artifact.read_normalization_recovery_v7_successor_feasibility(
            target,
            training_protocol_dir=inputs["protocol"],
            rule_pack_dir=inputs["pack"],
            training_audit_dir=inputs["audit"],
            evaluation_commitment_dir=inputs["commitment"],
            expected_manifest_sha256=published["manifest_sha256"],
        ))
    assert manifest == published
    assert manifest["held_out_boundary"][
        "vlc_identity_raw_or_translation_read_count"] == 0
    assert manifest["learner_or_selection_change_count"] == 0
    assert set(outputs) == {
        "context-scoped-local-projections.jsonl",
        "source-policy-replay-projections.jsonl",
        "variable-structure-projections.jsonl",
    }
    with pytest.raises(BroadQaExternalDataError, match="target path 非法"):
        artifact.publish_normalization_recovery_v7_successor_feasibility(
            run_root=tmp_path,
            training_protocol_dir=inputs["protocol"],
            rule_pack_dir=inputs["pack"],
            training_audit_dir=inputs["audit"],
            evaluation_commitment_dir=inputs["commitment"],
            target_dir=target,
        )


def test_feasibility_reader_rejects_synchronized_output_tamper(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """修改 output 与 manifest 摘要仍不能绕过 sealed TRAIN 重派生。"""
    inputs = _input_tree(tmp_path)
    monkeypatch.setattr(
        artifact, "V5_TRAINING_PROTOCOL_MANIFEST_SHA256",
        inputs["protocol_sha"])
    monkeypatch.setattr(
        artifact, "V5_RULE_PACK_MANIFEST_SHA256", inputs["pack_sha"])
    monkeypatch.setattr(
        artifact, "V6_TRAINING_AUDIT_MANIFEST_SHA256", inputs["audit_sha"])
    monkeypatch.setattr(
        artifact, "V7_EVALUATION_COMMITMENT_MANIFEST_SHA256",
        inputs["commitment_sha"])
    monkeypatch.setattr(artifact, "_require_k_root", lambda value: Path(value))
    target = tmp_path / "feasibility"
    artifact.publish_normalization_recovery_v7_successor_feasibility(
        run_root=tmp_path,
        training_protocol_dir=inputs["protocol"],
        rule_pack_dir=inputs["pack"],
        training_audit_dir=inputs["audit"],
        evaluation_commitment_dir=inputs["commitment"],
        target_dir=target,
    )
    path = target / "variable-structure-projections.jsonl"
    value = json.loads(path.read_bytes())
    value["execution_allowed"] = 1
    path.write_bytes(canonical_json_line(value))
    manifest_path = target / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    file_record = next(
        item for item in manifest["files"]
        if item["relative_path"] == path.name)
    payload = path.read_bytes()
    file_record["bytes"] = len(payload)
    file_record["sha256"] = hashlib.sha256(payload).hexdigest()
    encoded = canonical_json_line(manifest)
    manifest_path.write_bytes(encoded)
    with pytest.raises(BroadQaExternalDataError, match="records/inputs 漂移"):
        artifact.read_normalization_recovery_v7_successor_feasibility(
            target,
            training_protocol_dir=inputs["protocol"],
            rule_pack_dir=inputs["pack"],
            training_audit_dir=inputs["audit"],
            evaluation_commitment_dir=inputs["commitment"],
            expected_manifest_sha256=hashlib.sha256(encoded).hexdigest(),
        )
