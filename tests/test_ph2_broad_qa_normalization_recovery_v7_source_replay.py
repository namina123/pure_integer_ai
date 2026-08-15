"""覆盖 recovery-v7 source identity replay 与 audit 边界。"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_recovery_v7_source_replay_audit as audit,
)
from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_training_records import (
    V5_SOURCE_POLICY_BY_FAMILY,
    derive_normalization_recovery_v5_groups,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_source_replay_execution import (
    execute_source_replay_segment_transaction,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_source_replay_program import (
    derive_source_replay_program,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_variable_structure_records import (
    derive_variable_structure_plans,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
)


def _id(value: str) -> str:
    """形成稳定 synthetic identity。"""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _observation(
        family: str,
        suffix: str,
        output_piece: str,
        *,
        commitment_suffix: str | None = None,
        identity: bool = False,
        structured: bool = True,
        ) -> dict[str, object]:
    """构造带完整 source commitment 的 observation。"""
    input_text = "前舊{0}後" if structured else "舊"
    output_text = (
        input_text if identity
        else f"前{output_piece}{{0}}後" if structured
        else output_piece)
    return {
        "equal_length": int(len(input_text) == len(output_text)),
        "identity_preservation": int(identity),
        "input_text": input_text,
        "observation_id": _id(f"observation-{family}-{suffix}"),
        "output_text": output_text,
        "source_commitment": {
            "synthetic_source_identity": commitment_suffix or suffix},
        "source_family": family,
        "source_policy_scope": V5_SOURCE_POLICY_BY_FAMILY[family],
        "structure_tokens": ["{0}"] if structured else [],
    }


def _fragment(
        observation: dict[str, object],
        suffix: str,
        output_piece: str,
        ) -> dict[str, object]:
    """构造 source identity route 所属 EDIT_CORE fragment。"""
    return {
        "equal_length": int(len(output_piece) == 1),
        "fragment_id": _id(f"fragment-{suffix}"),
        "fragment_kind": "EDIT_CORE",
        "input_text": "舊",
        "observation_id": observation["observation_id"],
        "output_text": output_piece,
        "source_family": observation["source_family"],
        "source_policy_scope": observation["source_policy_scope"],
    }


def _material(
        *,
        ambiguous: bool = False,
        identity_veto: bool = False,
        ) -> tuple[
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            dict[str, object],
            dict[str, object],
        ]:
    """构造两个 source identity route 共享 input 的 conflict group。"""
    godot = _observation(
        "GODOT_ENGINE_PROJECT", "godot", "新的",
        commitment_suffix="source-a")
    vscode = _observation(
        "MICROSOFT_VSCODE_PROJECT", "vscode", "别的",
        commitment_suffix="source-b")
    observations = [godot, vscode]
    fragments = [
        _fragment(godot, "godot", "新的"),
        _fragment(vscode, "vscode", "别的"),
    ]
    if ambiguous:
        duplicate = _observation(
            "GODOT_ENGINE_PROJECT", "godot-duplicate", "又新",
            commitment_suffix="source-a")
        observations.append(duplicate)
        fragments.append(_fragment(duplicate, "godot-duplicate", "又新"))
    if identity_veto:
        observations.append(_observation(
            "GODOT_ENGINE_PROJECT", "godot-identity", "舊",
            commitment_suffix="source-a", identity=True, structured=False))
    groups = derive_normalization_recovery_v5_groups(tuple(fragments))
    program, representations, summary = derive_source_replay_program(
        observations=tuple(observations),
        fragments=tuple(fragments),
        groups=groups,
    )
    assert len(representations) == 1
    return tuple(observations), tuple(fragments), program, summary


def test_seen_source_identity_replay_commits_inside_structure_segment() -> None:
    """同 family/policy/commitment 的唯一 route 可原子重放并保留 token。"""
    observations, _fragments, program, summary = _material()
    godot = observations[0]
    plans, _plan_summary = derive_variable_structure_plans((godot,))
    left = execute_source_replay_segment_transaction(
        observation=godot, plan=plans[0], program=program, indexed=True)
    right = execute_source_replay_segment_transaction(
        observation=godot, plan=plans[0], program=program, indexed=False)
    assert left == right
    assert left["decision"] == "COMMIT"
    assert left["output_text"] == godot["output_text"]
    assert left["structure_token_mismatch_count"] == 0
    assert summary["all_routes_unique_conflict_count"] == 1
    assert summary["seen_source_self_replay_counts"] == {
        "EXACT": 2, "UNKNOWN": 0, "WRONG": 0}


def test_unseen_family_has_no_route_and_remains_unknown() -> None:
    """显式 family/policy scope 禁止把已见 source route 跨产品套用。"""
    _observations, _fragments, program, _summary = _material()
    thunderbird = _observation(
        "THUNDERBIRD_PROJECT", "held-out", "新的",
        commitment_suffix="source-a")
    plans, _plan_summary = derive_variable_structure_plans((thunderbird,))
    left = execute_source_replay_segment_transaction(
        observation=thunderbird, plan=plans[0],
        program=program, indexed=True)
    right = execute_source_replay_segment_transaction(
        observation=thunderbird, plan=plans[0],
        program=program, indexed=False)
    assert left == right
    assert left["decision"] == "UNKNOWN_NO_SOURCE_IDENTITY_ROUTE"
    assert left["output_text"] == thunderbird["input_text"]
    assert left["route_match_count"] == 0


def test_ambiguous_identity_route_and_identity_veto_fail_closed() -> None:
    """同 commitment 多 output 或 exact identity observation 均不得重放。"""
    observations, _fragments, ambiguous_program, summary = _material(
        ambiguous=True)
    godot = observations[0]
    plans, _plan_summary = derive_variable_structure_plans((godot,))
    ambiguous = execute_source_replay_segment_transaction(
        observation=godot, plan=plans[0],
        program=ambiguous_program, indexed=True)
    assert ambiguous["decision"] == "UNKNOWN_NO_SOURCE_IDENTITY_ROUTE"
    assert summary["ambiguous_route_count"] == 1
    veto_observations, _fragments, veto_program, _summary = _material(
        identity_veto=True)
    veto_godot = veto_observations[0]
    veto_plans, _plan_summary = derive_variable_structure_plans((veto_godot,))
    veto = execute_source_replay_segment_transaction(
        observation=veto_godot, plan=veto_plans[0],
        program=veto_program, indexed=True)
    assert veto["decision"] == "UNKNOWN_NO_SOURCE_IDENTITY_ROUTE"
    assert veto["identity_veto_count"] == 1


def _manifest(path: Path, value: dict[str, object]) -> str:
    """写入规范 synthetic manifest 并返回 SHA。"""
    payload = canonical_json_line(value)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def test_source_replay_audit_round_trip_is_nonruntime_and_nonoverwrite(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """publisher/reader 分账发布并拒绝覆盖既有 artifact。"""
    protocol = tmp_path / "protocol"
    commitment = tmp_path / "commitment"
    feasibility = tmp_path / "feasibility"
    variable = tmp_path / "variable"
    context_local = tmp_path / "context-local"
    for path in (
            protocol, commitment, feasibility, variable, context_local):
        path.mkdir()
    protocol_sha = _manifest(protocol / "manifest.json", {
        "artifact_kind": (
            "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V5_TRAINING_PROTOCOL_V1"),
        "evaluation_or_held_out_payload_read_count": 0,
        "status": "FROZEN_NOT_READ_NOT_LEARNED",
    })
    commitment_sha = _manifest(commitment / "manifest.json", {
        "artifact_kind": (
            "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V7_EVALUATION_COMMITMENT_V1"),
        "denominator": {"record_count": 3_656},
        "source_non_manifest_file_read_count": 0,
        "status": (
            "LABEL_BLIND_DENOMINATOR_AND_GATES_FROZEN_BEFORE_V7_LEARNER_CHANGE"),
    })
    feasibility_sha = _manifest(feasibility / "manifest.json", {
        "artifact_kind": (
            "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V7_SUCCESSOR_FEASIBILITY_V1"),
        "status": "TRAIN_ONLY_FEASIBILITY_COMPLETE_NOT_LEARNER_NOT_RUNTIME",
        "summary": {"source_policy_replay": {
            "context_or_source_identity_required_count": 5_703,
            "replayable_conflict_count": 984,
            "train_output_conflict_count": 6_687,
        }},
    })
    variable_sha = _manifest(variable / "manifest.json", {
        "artifact_kind": (
            "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V7_VARIABLE_STRUCTURE_AUDIT_V1"),
        "status": "TRAIN_ONLY_REPRESENTATION_PASS_CAPABILITY_NE_NOT_RUNTIME",
        "summary": {"plans": {"plan_count": 3_460}},
    })
    context_sha = _manifest(context_local / "manifest.json", {
        "artifact_kind": (
            "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V7_CONTEXT_LOCAL_AUDIT_V1"),
        "runtime_program_published": 0,
        "status": "TRAIN_ONLY_CONTEXT_LOCAL_NE_NOT_RUNTIME",
        "summary": {"loso": {
            "capability_outcome": "NE_ZERO_VARIABLE_EXACT",
            "wrong_count": 0,
        }},
    })
    monkeypatch.setattr(
        audit, "V5_TRAINING_PROTOCOL_MANIFEST_SHA256", protocol_sha)
    monkeypatch.setattr(
        audit, "V7_EVALUATION_COMMITMENT_MANIFEST_SHA256", commitment_sha)
    monkeypatch.setattr(
        audit, "V7_SUCCESSOR_FEASIBILITY_MANIFEST_SHA256", feasibility_sha)
    monkeypatch.setattr(
        audit, "V7_VARIABLE_STRUCTURE_AUDIT_MANIFEST_SHA256", variable_sha)
    monkeypatch.setattr(
        audit, "V7_CONTEXT_LOCAL_AUDIT_MANIFEST_SHA256", context_sha)
    monkeypatch.setattr(audit, "_require_k_root", lambda value: Path(value))
    representation = {"record_kind": "REPRESENTATION", "value": 1}
    loso = {"record_kind": "LOSO", "value": 2}
    summary = {
        "audit_outcome": "REPLAY_PASS_UNSEEN_TRANSFER_NE",
        "loso": {"capability_outcome": "NE_ZERO_UNSEEN_FAMILY_EXACT"},
        "loso_group_rederivation_count": 4,
    }
    monkeypatch.setattr(audit, "_derive", lambda **_kwargs: ({
        "full-pack-conflict-representations.jsonl": (representation,),
        "loso-audit.jsonl": (loso,),
    }, summary))
    target = tmp_path / "audit"
    published = audit.publish_normalization_recovery_v7_source_replay_audit(
        run_root=tmp_path,
        training_protocol_dir=protocol,
        evaluation_commitment_dir=commitment,
        successor_feasibility_dir=feasibility,
        variable_structure_audit_dir=variable,
        context_local_audit_dir=context_local,
        target_dir=target,
    )
    manifest, outputs = audit.read_normalization_recovery_v7_source_replay_audit(
        target,
        training_protocol_dir=protocol,
        evaluation_commitment_dir=commitment,
        successor_feasibility_dir=feasibility,
        variable_structure_audit_dir=variable,
        context_local_audit_dir=context_local,
        expected_manifest_sha256=published["manifest_sha256"],
    )
    assert manifest == published
    assert manifest["status"] == "TRAIN_ONLY_SOURCE_REPLAY_NE_NOT_RUNTIME"
    assert outputs["loso-audit.jsonl"] == (loso,)
    assert manifest["held_out_boundary"][
        "vlc_identity_raw_or_translation_read_count"] == 0
    with pytest.raises(BroadQaExternalDataError, match="input/target path 非法"):
        audit.publish_normalization_recovery_v7_source_replay_audit(
            run_root=tmp_path,
            training_protocol_dir=protocol,
            evaluation_commitment_dir=commitment,
            successor_feasibility_dir=feasibility,
            variable_structure_audit_dir=variable,
            context_local_audit_dir=context_local,
            target_dir=target,
        )
