"""Recovery-v8 LOSO learner, interpreters, rule pack and audit tests."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from pure_integer_ai.experiments import (
    ph2_broad_qa_materialized_learner_runtime as learner_runtime,
)
from pure_integer_ai.experiments import (
    ph2_broad_qa_materialized_rule_pack as pack_runtime,
)
from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_recovery_v8_learner as learner,
)
from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_recovery_v8_training_audit as audit,
)
from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_interpreter import (
    build_normalization_recovery_v8_rule_index,
    interpret_normalization_recovery_v8_indexed,
    interpret_normalization_recovery_v8_reference,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_learning_records import (
    derive_normalization_recovery_v8_learning_outputs,
    normalization_recovery_v8_output_payloads,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_rule_pack import (
    publish_normalization_recovery_v8_rule_pack,
    read_normalization_recovery_v8_rule_pack,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_training_audit_records import (
    derive_normalization_recovery_v8_training_audit,
)


_FAMILIES = (
    "KEEPASSXC_PROJECT",
    "QBITTORRENT_PROJECT",
    "STELLARIUM_PROJECT",
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _authorization(
        label: str, *, ledger: str, support: tuple[str, ...],
        ) -> dict[str, object]:
    """Build one synthetic tagged authorization record."""
    value = {
        "authorization_id": _sha(f"authorization:{label}"),
        "candidate_id": _sha(f"candidate:{label}"),
        "execution_allowed": 0,
        "learned_rule_claimed": 0,
        "ledger_kind": ledger,
        "support_families": list(support),
        "support_family_count": len(support),
    }
    if ledger == "ORTHOGRAPHIC_ATOM":
        return {**value, "input_atom": "檔", "output_atom": "档"}
    if ledger == "SOURCE_CONDITIONED_LEXICAL_ATOM":
        return {
            **value,
            "input_text": "檔案",
            "official_source_text": "File",
            "output_text": "文件",
        }
    return {**value, "structure_tokens": ["%1"]}


def _plans(authorization: dict[str, object]) -> tuple[dict[str, object], ...]:
    """Build the exact three directions required by the learner."""
    values = []
    before = authorization["support_families"]
    for held_out in _FAMILIES:
        after = [family for family in before if family != held_out]
        present = int(len(after) >= 2)
        supports = int(held_out in before)
        behavior = "EXACT" if supports and present else "UNKNOWN"
        values.append({
            "authorization_id": authorization["authorization_id"],
            "evaluation_case_expected": supports,
            "expected_behavior": behavior,
            "expected_rule_present_after_holdout": present,
            "held_out_family": held_out,
            "held_out_output_may_influence_rule_construction": 0,
            "held_out_output_read_count": 0,
            "ledger_kind": authorization["ledger_kind"],
            "loso_plan_id": _sha(
                f"plan:{authorization['authorization_id']}:{held_out}"),
            "support_families_after_holdout": after,
            "support_families_before_holdout": before,
        })
    return tuple(values)


def _protocol_material(protocol_sha: str) -> tuple[
        dict[str, object], dict[str, tuple[dict[str, object], ...]]]:
    """Build support-2/support-3 and identity-veto protocol material."""
    orthographic = _authorization(
        "orthographic", ledger="ORTHOGRAPHIC_ATOM", support=_FAMILIES[:2])
    lexical = _authorization(
        "lexical", ledger="SOURCE_CONDITIONED_LEXICAL_ATOM",
        support=_FAMILIES)
    structure = _authorization(
        "structure", ledger="LAYOUT_MORPHOLOGY_OBLIGATION",
        support=_FAMILIES)
    outputs = {
        "authorized-orthographic-atoms.jsonl": (orthographic,),
        "authorized-source-conditioned-lexical-atoms.jsonl": (lexical,),
        "authorized-layout-morphology-obligations.jsonl": (structure,),
        "exact-input-control.jsonl": ({
            "candidate_id": _sha("identity-candidate"),
            "candidate_status": "MULTI_FAMILY_UNIQUE_OUTPUT",
            "input_text": "地址",
            "outputs": [{"output_text": "地址"}],
            "support_families": list(_FAMILIES),
        },),
        "family-loso-plan.jsonl": (
            _plans(orthographic) + _plans(lexical) + _plans(structure)),
    }
    manifest = {
        "manifest_sha256": protocol_sha,
        "status": "THREE_LEDGER_PROTOCOL_FROZEN_NOT_TRAINED",
        "training_executed": 0,
        "vlc_final_read_count": 0,
    }
    return manifest, outputs


def test_learning_views_dual_interpreters_and_train_audit() -> None:
    """LOSO removal, dual interpreters and all audit hard gates must agree."""
    manifest, protocol_outputs = _protocol_material(_sha("protocol"))
    outputs, summary, work, increments = (
        derive_normalization_recovery_v8_learning_outputs(
            protocol_manifest=manifest, protocol_outputs=protocol_outputs))
    assert summary["learned_rule_count"] == 10
    assert summary["learner_work_item_count"] == 12
    assert len(work) == len(increments) == 12
    assert len(outputs["orthographic-rules.jsonl"]) == 1
    assert len(outputs["source-conditioned-lexical-rules.jsonl"]) == 3
    assert len(outputs["layout-morphology-obligations.jsonl"]) == 3
    assert len(outputs["identity-veto-rules.jsonl"]) == 3

    query = {
        "held_out_family": _FAMILIES[0],
        "input_text": "檔案",
        "official_source_text": "File",
        "query_kind": "SOURCE_CONDITIONED_LEXICAL_ATOM",
        "structure_tokens": [],
    }
    index = build_normalization_recovery_v8_rule_index(outputs)
    reference = interpret_normalization_recovery_v8_reference(outputs, query)
    indexed = interpret_normalization_recovery_v8_indexed(index, query)
    assert reference == indexed
    assert reference["behavior"] == "EXACT"
    assert reference["output_text"] == "文件"

    audit_outputs, audit_summary = derive_normalization_recovery_v8_training_audit(
        protocol_outputs=protocol_outputs, rule_outputs=outputs)
    assert audit_summary["case_count"] == 11
    assert audit_summary["expected_behavior_counts"] == {
        "EXACT": 9, "UNKNOWN": 2}
    assert audit_summary["hard_gates_pass"] == 1
    assert audit_summary["wrong_count"] == 0
    assert len(audit_outputs["train-results.jsonl"]) == 11


def test_fresh_resume_pack_and_audit_publication(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """Fresh/resume bytes, disabled pack and immutable PASS audit must hold."""
    protocol_sha = _sha("materialized-protocol")
    material = _protocol_material(protocol_sha)
    monkeypatch.setattr(
        learner, "read_normalization_recovery_v8_learner_input",
        lambda *args, **kwargs: material)
    monkeypatch.setattr(
        learner_runtime, "require_k_run_root",
        lambda value, *, label: Path(value).resolve())
    monkeypatch.setattr(
        pack_runtime, "require_k_run_root",
        lambda value, *, label: Path(value).resolve())
    monkeypatch.setattr(
        audit, "_require_k_root", lambda value: Path(value).resolve())
    monkeypatch.setattr(
        audit, "read_normalization_recovery_v8_learner_input",
        lambda *args, **kwargs: material)

    protocol_dir = tmp_path / "protocol"
    protocol_dir.mkdir()
    fresh = tmp_path / "fresh"
    resumed = tmp_path / "resumed"
    fresh_report = learner.run_normalization_recovery_v8_learner(
        run_root=tmp_path, protocol_dir=protocol_dir,
        expected_protocol_manifest_sha256=protocol_sha,
        run_dir=fresh, run_id=_sha("fresh"), mode="fresh",
        checkpoint_interval=4)
    partial = learner.run_normalization_recovery_v8_learner(
        run_root=tmp_path, protocol_dir=protocol_dir,
        expected_protocol_manifest_sha256=protocol_sha,
        run_dir=resumed, run_id=_sha("resumed"), mode="fresh",
        checkpoint_interval=3, stop_after=5)
    assert partial["status"] == learner.NORMALIZATION_RECOVERY_V8_CHECKPOINT_OPEN
    resumed_report = learner.run_normalization_recovery_v8_learner(
        run_root=tmp_path, protocol_dir=protocol_dir,
        expected_protocol_manifest_sha256=protocol_sha,
        run_dir=resumed, run_id=_sha("resumed"), mode="resume",
        checkpoint_interval=5)
    assert fresh_report["semantic_result_sha256"] == (
        resumed_report["semantic_result_sha256"])
    fresh_manifest, fresh_outputs = learner.read_normalization_recovery_v8_learner(
        fresh, protocol_dir=protocol_dir,
        expected_protocol_manifest_sha256=protocol_sha)
    resumed_manifest, resumed_outputs = learner.read_normalization_recovery_v8_learner(
        resumed, protocol_dir=protocol_dir,
        expected_protocol_manifest_sha256=protocol_sha)
    assert normalization_recovery_v8_output_payloads(fresh_outputs) == (
        normalization_recovery_v8_output_payloads(resumed_outputs))
    assert fresh_manifest["resume_markers"]["record_count"] == 0
    assert resumed_manifest["resume_markers"]["record_count"] == 1

    pack_dir = tmp_path / "pack"
    pack_report = publish_normalization_recovery_v8_rule_pack(
        run_root=tmp_path, protocol_dir=protocol_dir,
        expected_protocol_manifest_sha256=protocol_sha,
        fresh_run_dir=fresh, resumed_run_dir=resumed, target_dir=pack_dir)
    pack_manifest, pack_outputs = read_normalization_recovery_v8_rule_pack(
        pack_dir, protocol_dir=protocol_dir,
        expected_protocol_manifest_sha256=protocol_sha,
        expected_pack_manifest_sha256=str(pack_report["manifest_sha256"]))
    assert pack_manifest["fresh_resume_output_bytes_equal"] == 1
    assert pack_manifest["production_enabled"] == 0
    assert pack_outputs == fresh_outputs

    audit_dir = tmp_path / "audit"
    audit_report = audit.publish_normalization_recovery_v8_training_audit(
        run_root=tmp_path, protocol_dir=protocol_dir,
        expected_protocol_manifest_sha256=protocol_sha,
        pack_dir=pack_dir,
        expected_pack_manifest_sha256=str(pack_report["manifest_sha256"]),
        target_dir=audit_dir)
    assert audit_report["summary"]["hard_gates_pass"] == 1
    restored, _audit_outputs = audit.read_normalization_recovery_v8_training_audit(
        audit_dir, protocol_dir=protocol_dir,
        expected_protocol_manifest_sha256=protocol_sha,
        pack_dir=pack_dir,
        expected_pack_manifest_sha256=str(pack_report["manifest_sha256"]),
        expected_audit_manifest_sha256=str(audit_report["manifest_sha256"]))
    assert restored == audit_report
    with pytest.raises(BroadQaExternalDataError, match="path 非法"):
        audit.publish_normalization_recovery_v8_training_audit(
            run_root=tmp_path, protocol_dir=protocol_dir,
            expected_protocol_manifest_sha256=protocol_sha,
            pack_dir=pack_dir,
            expected_pack_manifest_sha256=str(pack_report["manifest_sha256"]),
            target_dir=audit_dir)


def test_rule_pack_rejects_non_k_root_before_protocol_read(tmp_path: Path) -> None:
    """Formal rule-pack publication must reject non-K roots first."""
    with pytest.raises(BroadQaExternalDataError, match="K 盘"):
        publish_normalization_recovery_v8_rule_pack(
            run_root=tmp_path,
            protocol_dir=tmp_path / "protocol",
            expected_protocol_manifest_sha256="a" * 64,
            fresh_run_dir=tmp_path / "fresh",
            resumed_run_dir=tmp_path / "resumed",
            target_dir=tmp_path / "pack",
        )
