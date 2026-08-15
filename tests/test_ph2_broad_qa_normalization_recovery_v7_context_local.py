"""覆盖 recovery-v7 context local segment transaction 与 audit 边界。"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_recovery_v7_context_local_audit as audit,
)
from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_learning_contract import (
    NORMALIZATION_RECOVERY_V5_DEFEATER_KIND,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_context_local_execution import (
    derive_context_local_rule_representations,
    execute_context_scoped_local_transfer,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_variable_structure_records import (
    derive_variable_structure_plans,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
)


def _id(value: str) -> str:
    """形成稳定测试 identity。"""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _rule(
        *,
        input_text: str = "舊",
        output_text: str = "新的",
        suffix: str = "base",
        ) -> dict[str, object]:
    """构造满足 target local 表示合同的 synthetic rule。"""
    return {
        "application_scope": {"structure_match_required": 0},
        "candidate_id": _id(f"candidate-{suffix}"),
        "candidate_scope_kind": "TARGET_CROSS_FAMILY",
        "fragment_kind": "EDIT_CORE",
        "input_text": input_text,
        "output_text": output_text,
        "positive_evidence_ids": [_id(f"evidence-{suffix}")],
        "rule_class": "EDIT_CORE",
        "rule_id": _id(f"rule-{suffix}"),
        "source_execution_family": "",
        "source_families": [
            "GODOT_ENGINE_PROJECT",
            "MICROSOFT_VSCODE_PROJECT",
        ],
        "structure_token_variants": [],
    }


def _defeater(rule: dict[str, object], *, left: str = "禁") -> dict[str, object]:
    """构造默认不命中的合法 negative defeater。"""
    return {
        "action": "BLOCK_PHRASE_RULE_USE_BACKOFF",
        "candidate_id": rule["candidate_id"],
        "defeater_id": _id(f"defeater-{rule['rule_id']}-{left}"),
        "left_boundary": 0,
        "left_context": left,
        "production_enabled": 0,
        "record_kind": NORMALIZATION_RECOVERY_V5_DEFEATER_KIND,
        "right_boundary": 0,
        "right_context": "",
        "rule_class": "EDIT_CORE",
        "rule_id": rule["rule_id"],
    }


def _observation() -> dict[str, object]:
    """构造 token 原样复制且 segment 变长的 observation。"""
    return {
        "equal_length": 0,
        "identity_preservation": 0,
        "input_text": "前舊{0}後",
        "observation_id": _id("context-local-observation"),
        "output_text": "前新的{0}後",
        "source_family": "THUNDERBIRD_PROJECT",
        "structure_tokens": ["{0}"],
    }


def _context(
        *,
        identity_veto: bool = False,
        conflict_veto: bool = False,
        defeater_hit: bool = False,
        open_obligation: bool = False,
        overlapping: bool = False,
        ) -> dict[str, object]:
    """构造 indexed/reference 共用的只读 local execution context。"""
    rules = [_rule()]
    if overlapping:
        rules.append(_rule(
            input_text="前舊", output_text="前新的", suffix="overlap"))
    positive = {
        str(rules[0]["rule_id"]): frozenset(((0, "前", 0, "{0}後"),)),
    }
    if overlapping:
        positive[str(rules[1]["rule_id"])] = frozenset(
            ((1, "", 0, "{0}後"),))
    defeaters = {
        str(rule["rule_id"]): (_defeater(
            rule, left="前" if defeater_hit and index == 0 else "禁"),)
        for index, rule in enumerate(rules)
    }
    ordered = tuple(sorted(
        rules,
        key=lambda rule: (-len(str(rule["input_text"])), str(rule["rule_id"]))))
    buckets = {}
    for rule in ordered:
        buckets.setdefault(ord(str(rule["input_text"])[0]), []).append(rule)
    frozen_buckets = {key: tuple(value) for key, value in buckets.items()}
    obligations = {"舊": ("舊",)}
    if open_obligation:
        obligations["後"] = ("後",)
    context = {
        "conflict_inputs": frozenset({"舊"} if conflict_veto else ()),
        "defeaters_by_rule": defeaters,
        "identity_inputs": frozenset({"舊"} if identity_veto else ()),
        "obligation_buckets": obligations,
        "positive_contexts": positive,
        "rule_by_id": {str(rule["rule_id"]): rule for rule in rules},
        "source_buckets": {},
        "target_buckets": frozen_buckets,
    }
    representations, summary = derive_context_local_rule_representations(
        context=context,
        held_out_source_family="THUNDERBIRD_PROJECT",
    )
    eligible = frozenset(
        str(item["predecessor_rule_id"])
        for item in representations
        if item["status"] == "REPRESENTATION_ELIGIBLE")
    assert summary["representation_eligible_count"] == len(rules)
    return {
        **context,
        "eligible_rule_ids": eligible,
        "reference_local_rules": ordered,
    }


def _execute(context: dict[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    """用同一 plan 执行 indexed/reference。"""
    observation = _observation()
    plans, _summary = derive_variable_structure_plans((observation,))
    left = execute_context_scoped_local_transfer(
        observation=observation, plan=plans[0],
        context=context, indexed=True)
    right = execute_context_scoped_local_transfer(
        observation=observation, plan=plans[0],
        context=context, indexed=False)
    return left, right


def test_context_local_commits_only_closed_segment_transaction() -> None:
    """正上下文 rule 闭合唯一 obligation 后整句提交并原样复制 token。"""
    left, right = _execute(_context())
    assert left == right
    assert left["decision"] == "COMMIT"
    assert left["output_text"] == "前新的{0}後"
    assert left["open_obligation_count"] == 0
    assert left["partial_commit_count"] == 0
    assert left["structure_token_mismatch_count"] == 0


@pytest.mark.parametrize("veto", ("identity", "conflict", "defeater"))
def test_context_local_vetoes_run_before_proposal(veto: str) -> None:
    """identity/conflict/negative context 任一命中都 fail closed。"""
    left, right = _execute(_context(
        identity_veto=veto == "identity",
        conflict_veto=veto == "conflict",
        defeater_hit=veto == "defeater",
    ))
    assert left == right
    assert left["decision"] == "UNKNOWN_NO_PROPOSAL"
    assert left["output_text"] == _observation()["input_text"]


def test_context_local_rejects_open_or_overlapping_proposals() -> None:
    """未闭合 obligation 与重叠多解均不得形成局部混合 output。"""
    open_left, open_right = _execute(_context(open_obligation=True))
    assert open_left == open_right
    assert open_left["decision"] == "UNKNOWN_OPEN_OBLIGATION"
    assert open_left["output_text"] == _observation()["input_text"]
    overlap_left, overlap_right = _execute(_context(overlapping=True))
    assert overlap_left == overlap_right
    assert overlap_left["decision"] \
        == "UNKNOWN_OVERLAPPING_OR_AMBIGUOUS_PROPOSAL"
    assert overlap_left["output_text"] == _observation()["input_text"]


def _manifest(path: Path, value: dict[str, object]) -> str:
    """写入规范 synthetic manifest 并返回 SHA。"""
    payload = canonical_json_line(value)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def test_context_local_audit_round_trip_is_nonruntime_and_nonoverwrite(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """publisher/reader 保持 sealed 边界并拒绝覆盖既有 artifact。"""
    protocol = tmp_path / "protocol"
    commitment = tmp_path / "commitment"
    feasibility = tmp_path / "feasibility"
    variable = tmp_path / "variable"
    for path in (protocol, commitment, feasibility, variable):
        path.mkdir()
    protocol_sha = _manifest(protocol / "manifest.json", {
        "artifact_kind": (
            "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V5_TRAINING_PROTOCOL_V1"),
        "evaluation_or_held_out_payload_read_count": 0,
        "production_enabled": 0,
        "status": "FROZEN_NOT_READ_NOT_LEARNED",
    })
    commitment_sha = _manifest(commitment / "manifest.json", {
        "artifact_kind": (
            "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V7_EVALUATION_COMMITMENT_V1"),
        "denominator": {"record_count": 3_656},
        "source_non_manifest_file_read_count": 0,
        "status": (
            "LABEL_BLIND_DENOMINATOR_AND_GATES_FROZEN_BEFORE_V7_LEARNER_CHANGE"),
        "training_source_read_count": 0,
    })
    feasibility_sha = _manifest(feasibility / "manifest.json", {
        "artifact_kind": (
            "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V7_SUCCESSOR_FEASIBILITY_V1"),
        "learner_or_selection_change_count": 0,
        "runtime_program_published": 0,
        "status": "TRAIN_ONLY_FEASIBILITY_COMPLETE_NOT_LEARNER_NOT_RUNTIME",
        "summary": {"context_scoped_local_transfer": {
            "has_defeater": 500,
            "positive_nonempty_context": 472,
            "representation_feasible": 472,
            "rule_count": 500,
            "status": (
                "FEASIBLE_WITH_DEFER_AND_ATOMIC_COMMIT_IMPLEMENTATION_REQUIRED"),
            "support_closed": 500,
        }},
    })
    variable_sha = _manifest(variable / "manifest.json", {
        "artifact_kind": (
            "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V7_VARIABLE_STRUCTURE_AUDIT_V1"),
        "candidate_family_formal_run_count": 0,
        "runtime_program_published": 0,
        "status": "TRAIN_ONLY_REPRESENTATION_PASS_CAPABILITY_NE_NOT_RUNTIME",
        "summary": {"plans": {
            "obligation_count": 6_691,
            "plan_count": 3_460,
            "representation_eligible_count": 3_459,
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
    monkeypatch.setattr(audit, "_require_k_root", lambda value: Path(value))
    rule_record = {"record_kind": "RULE", "value": 1}
    loso_record = {"record_kind": "LOSO", "value": 2}
    summary = {
        "audit_outcome": "REPRESENTATION_PASS_CAPABILITY_NE",
        "frozen_full_pack_context_rule_count": 500,
        "frozen_full_pack_deferred_no_surface_context_count": 28,
        "frozen_full_pack_representation_eligible_count": 472,
        "loso": {"capability_outcome": "NE_ZERO_VARIABLE_EXACT"},
        "loso_relearning_count": 4,
    }
    monkeypatch.setattr(audit, "_derive", lambda **_kwargs: ({
        "loso-rule-representations.jsonl": (rule_record,),
        "loso-audit.jsonl": (loso_record,),
    }, summary))
    target = tmp_path / "audit"
    published = audit.publish_normalization_recovery_v7_context_local_audit(
        run_root=tmp_path,
        training_protocol_dir=protocol,
        evaluation_commitment_dir=commitment,
        successor_feasibility_dir=feasibility,
        variable_structure_audit_dir=variable,
        target_dir=target,
    )
    manifest, outputs = audit.read_normalization_recovery_v7_context_local_audit(
        target,
        training_protocol_dir=protocol,
        evaluation_commitment_dir=commitment,
        successor_feasibility_dir=feasibility,
        variable_structure_audit_dir=variable,
        expected_manifest_sha256=published["manifest_sha256"],
    )
    assert manifest == published
    assert manifest["status"] == "TRAIN_ONLY_CONTEXT_LOCAL_NE_NOT_RUNTIME"
    assert outputs["loso-audit.jsonl"] == (loso_record,)
    assert manifest["held_out_boundary"][
        "vlc_identity_raw_or_translation_read_count"] == 0
    with pytest.raises(BroadQaExternalDataError, match="input/target path 非法"):
        audit.publish_normalization_recovery_v7_context_local_audit(
            run_root=tmp_path,
            training_protocol_dir=protocol,
            evaluation_commitment_dir=commitment,
            successor_feasibility_dir=feasibility,
            variable_structure_audit_dir=variable,
            target_dir=target,
        )
