"""覆盖 recovery-v7 variable structure layout、plans 与 LOSO。"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_recovery_v7_variable_structure_audit as audit,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_localization_structure import (
    localization_structure_layout,
    localization_structure_layout_for_tokens,
    localization_structure_token_category,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v7_variable_structure_records import (
    derive_variable_structure_plans,
    run_variable_structure_loso,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_line,
)


def _id(value: str) -> str:
    """形成测试 observation identity。"""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _observation(
        family: str,
        token: str,
        suffix: str,
        ) -> dict[str, object]:
    """构造 raw token 精确复制的 variable structured observation。"""
    return {
        "equal_length": 0,
        "identity_preservation": 0,
        "input_text": f"繁體前文 {token} 較長",
        "observation_id": _id(f"observation-{suffix}"),
        "output_text": f"简体 {token}",
        "source_family": family,
        "structure_tokens": [token],
    }


def test_structure_layout_reconstructs_raw_text_and_normalizes_categories() -> None:
    """layout 同时保留 raw token、规范 identity 与可重建 segments。"""
    value = "前<b>%s</b>后"
    layout = localization_structure_layout(value)
    rebuilt = "".join(
        segment + (layout["raw_tokens"][index]
                   if index < len(layout["raw_tokens"]) else "")
        for index, segment in enumerate(layout["segments"])
    )
    assert rebuilt == value
    assert layout["structure_tokens"] == (
        "HTML_OPEN:b", "%s", "HTML_CLOSE:b")
    assert tuple(localization_structure_token_category(token)
                 for token in layout["structure_tokens"]) == (
                     "HTML_OPEN", "PERCENT_PLACEHOLDER", "HTML_CLOSE")
    guided = localization_structure_layout_for_tokens(
        "`{not_a_placeholder}`", ("`", "`"))
    assert guided["segments"] == ("", "{not_a_placeholder}", "")


def test_variable_structure_loso_represents_but_never_partially_commits() -> None:
    """三 family category 支持可表示 held-out，但无 generator 时全为 UNKNOWN。"""
    observations = (
        _observation("GODOT_ENGINE_PROJECT", "%s", "a"),
        _observation("MICROSOFT_VSCODE_PROJECT", "%1", "b"),
        _observation("THUNDERBIRD_PROJECT", "%d", "c"),
    )
    plans, plan_summary = derive_variable_structure_plans(observations)
    loso, summary = run_variable_structure_loso(plans)
    assert plan_summary["plan_count"] == 3
    assert plan_summary["representation_eligible_count"] == 3
    assert all(plan["execution_allowed"] == 0 for plan in plans)
    assert all(plan["partial_commit_allowed"] == 0 for plan in plans)
    assert summary == {
        "capability_outcome": "NE_SEGMENT_GENERATOR_NOT_IMPLEMENTED",
        "deferred_structure_mismatch_count": 0,
        "exact_count": 0,
        "facility_outcome": "PASS",
        "held_out_inventory_count": 3,
        "ineligible_layout_count": 0,
        "loso_family_count": 4,
        "partial_commit_count": 0,
        "represented_count": 3,
        "unknown_count": 3,
        "wrong_count": 0,
    }
    assert sum(item["represented_count"] for item in loso) == 3
    assert all(item["wrong_count"] == 0 for item in loso)


def test_raw_token_change_is_ineligible_instead_of_being_generated() -> None:
    """output 改写 placeholder 时 plan fail closed，不把 category 相同当复制。"""
    observation = _observation("GODOT_ENGINE_PROJECT", "%s", "mismatch")
    observation["output_text"] = "简体 %d"
    plans, plan_summary = derive_variable_structure_plans((observation,))
    _loso, summary = run_variable_structure_loso(plans)
    assert plans[0]["ledger_layout_success"] == 0
    assert plans[0]["raw_structure_copy_equal"] == 0
    assert plans[0]["representation_eligible"] == 0
    assert plan_summary["ledger_layout_failure_count"] == 1
    assert summary["deferred_structure_mismatch_count"] == 1
    assert summary["facility_outcome"] == "PASS"
    assert summary["wrong_count"] == 0


def test_ledger_guided_layout_ignores_shared_parser_overmatch() -> None:
    """代码块内 brace 不重选 adapter ledger，仍只以 backtick 形成 segments。"""
    observation = {
        "equal_length": 0,
        "identity_preservation": 0,
        "input_text": "繁體 `{not_a_placeholder}` 較長",
        "observation_id": _id("observation-ledger"),
        "output_text": "简体 `{not_a_placeholder}`",
        "source_family": "GODOT_ENGINE_PROJECT",
        "structure_tokens": ["`", "`"],
    }
    plans, summary = derive_variable_structure_plans((observation,))
    assert plans[0]["shared_parser_match"] == 0
    assert plans[0]["ledger_layout_success"] == 1
    assert plans[0]["representation_eligible"] == 1
    assert summary["shared_parser_mismatch_count"] == 1


def _write_jsonl(path: Path, values: tuple[dict[str, object], ...]) -> dict[str, object]:
    """写入测试 JSONL 并返回 manifest 物理字段。"""
    payload = b"".join(canonical_json_line(value) for value in values)
    path.write_bytes(payload)
    return {
        "bytes": len(payload),
        "record_count": len(values),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _write_manifest(path: Path, value: dict[str, object]) -> str:
    """写入规范测试 manifest 并返回 SHA。"""
    payload = canonical_json_line(value)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def test_variable_structure_audit_round_trip(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """audit 只读 observations 与两个 manifest，并严格重派生 plans/LOSO。"""
    protocol = tmp_path / "protocol"
    commitment = tmp_path / "commitment"
    feasibility = tmp_path / "feasibility"
    for path in (protocol, commitment, feasibility):
        path.mkdir()
    observations = (
        _observation("GODOT_ENGINE_PROJECT", "%s", "audit-a"),
        _observation("MICROSOFT_VSCODE_PROJECT", "%1", "audit-b"),
        _observation("THUNDERBIRD_PROJECT", "%d", "audit-c"),
    )
    physical = _write_jsonl(
        protocol / "train.pair-observations.jsonl", observations)
    protocol_sha = _write_manifest(protocol / "manifest.json", {
        "artifact_kind": (
            "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V5_TRAINING_PROTOCOL_V1"),
        "evaluation_or_held_out_payload_read_count": 0,
        "files": [{
            **physical,
            "relative_path": "train.pair-observations.jsonl",
            "role": "TRAIN_PAIR_OBSERVATIONS",
        }],
        "production_enabled": 0,
        "status": "FROZEN_NOT_READ_NOT_LEARNED",
    })
    commitment_sha = _write_manifest(commitment / "manifest.json", {
        "artifact_kind": (
            "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V7_EVALUATION_COMMITMENT_V1"),
        "denominator": {"record_count": 3_656},
        "production_enabled": 0,
        "source_non_manifest_file_read_count": 0,
        "status": (
            "LABEL_BLIND_DENOMINATOR_AND_GATES_FROZEN_BEFORE_V7_LEARNER_CHANGE"),
        "training_source_read_count": 0,
    })
    feasibility_sha = _write_manifest(feasibility / "manifest.json", {
        "artifact_kind": (
            "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V7_SUCCESSOR_FEASIBILITY_V1"),
        "learner_or_selection_change_count": 0,
        "runtime_program_published": 0,
        "status": "TRAIN_ONLY_FEASIBILITY_COMPLETE_NOT_LEARNER_NOT_RUNTIME",
        "summary": {
            "overall_outcome": (
                "FEASIBILITY_CONFIRMED_NARROW_OR_PARTIAL_IMPLEMENTATION_REQUIRED"),
        },
    })
    monkeypatch.setattr(
        audit, "V5_TRAINING_PROTOCOL_MANIFEST_SHA256", protocol_sha)
    monkeypatch.setattr(
        audit, "V7_EVALUATION_COMMITMENT_MANIFEST_SHA256", commitment_sha)
    monkeypatch.setattr(
        audit, "V7_SUCCESSOR_FEASIBILITY_MANIFEST_SHA256", feasibility_sha)
    monkeypatch.setattr(audit, "_require_k_root", lambda value: Path(value))
    target = tmp_path / "audit"
    published = audit.publish_normalization_recovery_v7_variable_structure_audit(
        run_root=tmp_path,
        training_protocol_dir=protocol,
        evaluation_commitment_dir=commitment,
        successor_feasibility_dir=feasibility,
        target_dir=target,
    )
    manifest, outputs = (
        audit.read_normalization_recovery_v7_variable_structure_audit(
            target,
            training_protocol_dir=protocol,
            evaluation_commitment_dir=commitment,
            successor_feasibility_dir=feasibility,
            expected_manifest_sha256=published["manifest_sha256"],
        ))
    assert manifest == published
    assert manifest["summary"]["audit_outcome"] \
        == "REPRESENTATION_PASS_CAPABILITY_NE"
    assert len(outputs["structure-plans.jsonl"]) == 3
    assert len(outputs["loso-audit.jsonl"]) == 4
    assert manifest["held_out_boundary"][
        "vlc_identity_raw_or_translation_read_count"] == 0
    with pytest.raises(BroadQaExternalDataError, match="input/target path 非法"):
        audit.publish_normalization_recovery_v7_variable_structure_audit(
            run_root=tmp_path,
            training_protocol_dir=protocol,
            evaluation_commitment_dir=commitment,
            successor_feasibility_dir=feasibility,
            target_dir=target,
        )
