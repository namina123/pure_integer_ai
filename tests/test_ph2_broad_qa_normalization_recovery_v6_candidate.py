"""Recovery-v6 candidate runtime、preflight 与 pack 测试。"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_recovery_v6_candidate as candidate_module,
)
from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_recovery_v6_candidate_pack as pack_module,
)
from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_evaluation_commitment import (
    NORMALIZATION_RECOVERY_V5_EVALUATION_COMMITMENT_KIND,
    NORMALIZATION_RECOVERY_V5_EVALUATION_COMMITMENT_STATUS,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v6_candidate import (
    compile_normalization_recovery_v6_candidate,
    derive_normalization_recovery_v6_candidate_preflight,
    execute_normalization_recovery_v6_candidate_batch,
    reference_normalization_recovery_v6_candidate_batch,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v6_candidate_pack import (
    publish_normalization_recovery_v6_candidate_pack,
    read_normalization_recovery_v6_candidate_pack,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v6_learning_contract import (
    RECOVERY_V6_TARGET_POLICY_SCOPE,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v6_phrase_runtime import (
    compile_normalization_recovery_v6_phrase_program,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v6_training_audit import (
    NORMALIZATION_RECOVERY_V6_TRAINING_AUDIT_KIND,
    NORMALIZATION_RECOVERY_V6_TRAINING_AUDIT_STATUS,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    canonical_json_line,
)
from test_ph2_broad_qa_normalization_recovery_v6_training_audit import (
    _audit_material,
)


def _sha(value: str) -> str:
    """返回 synthetic artifact identity。"""
    return hashlib.sha256(value.encode()).hexdigest()


def _material():
    """构造 synthetic v6 pack、candidate、audit 与 commitment。"""
    values = _audit_material()
    material = values[0]
    audit_sha = values[1]
    pack_manifest = values[7]
    pack_outputs = values[8]
    phrase = compile_normalization_recovery_v6_phrase_program(
        rule_pack_manifest_sha256=pack_manifest["manifest_sha256"],
        target_whole_rules=pack_outputs["target-whole-rules.jsonl"],
        defeaters=pack_outputs["defeaters.jsonl"],
        identity_vetoes=pack_outputs["identity-vetoes.jsonl"],
        conflict_vetoes=pack_outputs["conflict-vetoes.jsonl"],
        target_index=pack_outputs["target-index.jsonl"],
    )
    commitment_sha = _sha("v6-candidate-commitment")
    candidate = compile_normalization_recovery_v6_candidate(
        phrase_program=phrase,
        v6_training_audit_manifest_sha256=audit_sha,
        evaluation_commitment_manifest_sha256=commitment_sha,
    )
    audit = {
        "artifact_kind": NORMALIZATION_RECOVERY_V6_TRAINING_AUDIT_KIND,
        "manifest_sha256": audit_sha,
        "pack_manifest_sha256": pack_manifest["manifest_sha256"],
        "predecessor_rule_pack_manifest_sha256": pack_manifest[
            "predecessor_rule_pack_manifest_sha256"],
        "protocol_manifest_sha256": material[0]["manifest_sha256"],
        "status": NORMALIZATION_RECOVERY_V6_TRAINING_AUDIT_STATUS,
        "summary": {
            "audit_outcome": "FACILITY_PASS_CAPABILITY_PASS",
            "capability_gate_pass": 1,
            "facility_failure_count": 0,
            "identity_false_change_count": 0,
            "non_identity_exact_count": 18,
            "outcome_counts": {
                "EXACT": 1510,
                "UNKNOWN": 36740,
                "WRONG": 0,
            },
            "simulation_strategy_reference_equal": 1,
        },
    }
    commitment = {
        "artifact_kind": NORMALIZATION_RECOVERY_V5_EVALUATION_COMMITMENT_KIND,
        "denominator": {"record_count": 3531},
        "dimensions": {"SYNTHETIC": {"threshold": 1}},
        "formal_contract": {
            "candidate_applicability_cannot_shrink_denominator": 1,
        },
        "manifest_sha256": commitment_sha,
        "mastery_claimed": 0,
        "production_enabled": 0,
        "source_non_manifest_file_read_count": 0,
        "status": NORMALIZATION_RECOVERY_V5_EVALUATION_COMMITMENT_STATUS,
        "training_source_read_count": 0,
    }
    return (
        material,
        pack_manifest,
        pack_outputs,
        candidate,
        audit,
        commitment,
    )


def test_v6_candidate_valid_scope_never_shrinks_denominator() -> None:
    """合法 scope 必须全 applicable，未命中只允许 identity backoff。"""
    _material_values, _pack, outputs, candidate, _audit, _commitment = (
        _material())
    rule = outputs["target-whole-rules.jsonl"][0]
    tokens = tuple(rule["structure_token_variants"][0])
    texts = (str(rule["input_text"]), "从未见过的输入")
    structures = (tokens, ())
    indexed = execute_normalization_recovery_v6_candidate_batch(
        candidate,
        texts,
        policy_scope=RECOVERY_V6_TARGET_POLICY_SCOPE,
        structure_tokens=structures,
    )
    reference = reference_normalization_recovery_v6_candidate_batch(
        candidate,
        texts,
        policy_scope=RECOVERY_V6_TARGET_POLICY_SCOPE,
        structure_tokens=structures,
    )
    assert indexed == reference
    assert [item["applicable"] for item in indexed] == [1, 1]
    assert indexed[0]["output_text"] == rule["output_text"]
    assert indexed[1]["output_text"] == texts[1]
    assert indexed[1]["scope_mismatch"] == 0

    rejected = execute_normalization_recovery_v6_candidate_batch(
        candidate,
        texts,
        policy_scope="",
        structure_tokens=structures,
    )
    assert [item["applicable"] for item in rejected] == [0, 0]
    assert [item["output_text"] for item in rejected] == list(texts)
    assert all(item["scope_mismatch"] == 1 for item in rejected)


def test_v6_candidate_preflight_covers_rules_vetoes_and_scope() -> None:
    """标签盲 preflight 必须覆盖全部 rule/veto 与 scope negative。"""
    _material_values, _pack, outputs, candidate, _audit, _commitment = (
        _material())
    preflight = derive_normalization_recovery_v6_candidate_preflight(candidate)
    expected = (
        len(outputs["target-whole-rules.jsonl"])
        + len(outputs["identity-vetoes.jsonl"])
        + len(outputs["conflict-vetoes.jsonl"])
        + 1)
    assert preflight["case_count"] == expected
    assert sum(preflight["case_kind_counts"].values()) == expected
    assert preflight["case_kind_counts"]["INVALID_SCOPE_REJECTED"] == 1
    assert preflight["failure_count"] == 0
    assert preflight["indexed_reference_mismatch_count"] == 0
    assert preflight["valid_scope_all_applicable"] == 1
    assert preflight["invalid_scope_rejected"] == 1
    assert preflight["evaluation_payload_read_count"] == 0


def test_v6_candidate_rejects_rehashed_applicability_tamper() -> None:
    """只重算 candidate 外层 SHA 不能放宽全分母 applicability。"""
    _material_values, _pack, _outputs, candidate, _audit, _commitment = (
        _material())
    tampered = copy.deepcopy(candidate)
    tampered["applicability_contract"][
        "valid_scope_applicable_for_full_denominator"] = 0
    tampered["candidate_program_sha256"] = hashlib.sha256(
        canonical_json_bytes(candidate_module._candidate_payload(tampered))
    ).hexdigest()
    with pytest.raises(BroadQaExternalDataError, match="scope 合同"):
        execute_normalization_recovery_v6_candidate_batch(
            tampered,
            ("输入",),
            policy_scope=RECOVERY_V6_TARGET_POLICY_SCOPE,
        )


def _install_readers(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ):
    """把 candidate pack 的三个严格 reader 固定到 synthetic material。"""
    values = _material()
    material, pack_manifest, pack_outputs, _candidate, audit, commitment = values
    roots = tuple(tmp_path / name for name in (
        "protocol", "predecessor-pack", "v6-pack", "v6-audit",
        "qt-source", "commitment"))
    for root in roots:
        root.mkdir()
    monkeypatch.setattr(
        pack_module,
        "read_normalization_recovery_v6_rule_pack",
        lambda *args, **kwargs: (pack_manifest, pack_outputs),
    )
    monkeypatch.setattr(
        pack_module, "_read_audit_manifest",
        lambda *args, **kwargs: audit,
    )
    monkeypatch.setattr(
        pack_module,
        "read_normalization_recovery_v5_evaluation_commitment",
        lambda *args, **kwargs: commitment,
    )
    monkeypatch.setattr(
        pack_module, "_require_k_root", lambda value: Path(value).resolve())
    return roots, material, pack_manifest, audit, commitment


def test_v6_candidate_pack_publishes_manifest_last_and_strictly_rederives(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """candidate pack 必须不可覆盖、manifest-last 且拒绝文件篡改。"""
    roots, material, pack_manifest, audit, commitment = _install_readers(
        tmp_path, monkeypatch)
    target = tmp_path / "candidate"
    write_order = []
    original_open = Path.open

    def tracked_open(path, *args, **kwargs):
        """记录 target exclusive writes 并转发真实文件操作。"""
        mode = args[0] if args else kwargs.get("mode", "r")
        if mode == "xb" and path.parent == target:
            write_order.append(path.name)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracked_open)
    arguments = {
        "run_root": tmp_path,
        "protocol_dir": roots[0],
        "expected_protocol_manifest_sha256": str(
            material[0]["manifest_sha256"]),
        "predecessor_pack_dir": roots[1],
        "expected_predecessor_pack_manifest_sha256": str(
            pack_manifest["predecessor_rule_pack_manifest_sha256"]),
        "pack_dir": roots[2],
        "expected_pack_manifest_sha256": str(
            pack_manifest["manifest_sha256"]),
        "audit_dir": roots[3],
        "expected_audit_manifest_sha256": str(audit["manifest_sha256"]),
        "qt_source_pack_dir": roots[4],
        "expected_qt_source_manifest_sha256": _sha("v6-candidate-qt"),
        "evaluation_commitment_dir": roots[5],
        "expected_evaluation_commitment_manifest_sha256": str(
            commitment["manifest_sha256"]),
        "target_dir": target,
    }
    report = publish_normalization_recovery_v6_candidate_pack(**arguments)
    assert write_order == [
        "candidate-program.json", "preflight.json", "manifest.json"]
    assert report["formal_run_count"] == 0
    assert report["evaluation_or_reserve_payload_read_count"] == 0
    assert report["summary"]["preflight_failure_count"] == 0

    read_arguments = dict(arguments)
    del read_arguments["run_root"]
    del read_arguments["target_dir"]
    restored, candidate, preflight = (
        read_normalization_recovery_v6_candidate_pack(
            target,
            expected_candidate_manifest_sha256=str(
                report["manifest_sha256"]),
            **read_arguments,
        ))
    assert restored["manifest_sha256"] == report["manifest_sha256"]
    assert candidate["candidate_program_sha256"] == report[
        "candidate_program_sha256"]
    assert preflight["failure_count"] == 0
    with pytest.raises(BroadQaExternalDataError, match="target 已存在"):
        publish_normalization_recovery_v6_candidate_pack(**arguments)

    path = target / "preflight.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(BroadQaExternalDataError, match="重派生漂移"):
        read_normalization_recovery_v6_candidate_pack(
            target,
            expected_candidate_manifest_sha256=str(
                report["manifest_sha256"]),
            **read_arguments,
        )


def test_v6_candidate_pack_rejects_non_k_root_before_write(
        tmp_path: Path,
        ) -> None:
    """正式 publisher 不得把 candidate artifact 回退到 D 盘。"""
    target = tmp_path / "candidate"
    with pytest.raises(BroadQaExternalDataError, match="K 盘"):
        publish_normalization_recovery_v6_candidate_pack(
            run_root=tmp_path,
            protocol_dir=tmp_path / "protocol",
            expected_protocol_manifest_sha256="a" * 64,
            predecessor_pack_dir=tmp_path / "predecessor-pack",
            expected_predecessor_pack_manifest_sha256="b" * 64,
            pack_dir=tmp_path / "v6-pack",
            expected_pack_manifest_sha256="c" * 64,
            audit_dir=tmp_path / "v6-audit",
            expected_audit_manifest_sha256="d" * 64,
            qt_source_pack_dir=tmp_path / "qt-source",
            expected_qt_source_manifest_sha256="e" * 64,
            evaluation_commitment_dir=tmp_path / "commitment",
            expected_evaluation_commitment_manifest_sha256="f" * 64,
            target_dir=target,
        )
    assert not target.exists()


def test_v6_candidate_audit_reader_requires_capability_pass(
        tmp_path: Path,
        ) -> None:
    """candidate 不得从 TRAIN FAIL/NE audit 晋升。"""
    root = tmp_path / "audit"
    root.mkdir()
    files = []
    for name in ("runtime-audit.jsonl", "loso-audit.jsonl"):
        payload = canonical_json_line({"name": name})
        (root / name).write_bytes(payload)
        files.append({
            "bytes": len(payload),
            "relative_path": name,
            "sha256": hashlib.sha256(payload).hexdigest(),
        })
    manifest = {
        "artifact_kind": NORMALIZATION_RECOVERY_V6_TRAINING_AUDIT_KIND,
        "candidate_pack_read_count": 0,
        "evaluation_commitment_read_count": 0,
        "evaluation_payload_read_count": 0,
        "files": files,
        "formal_run_count": 0,
        "mastery_claimed": 0,
        "pack_manifest_sha256": _sha("pack"),
        "predecessor_rule_pack_manifest_sha256": _sha("predecessor"),
        "prior_formal_item_read_count": 0,
        "production_enabled": 0,
        "protocol_manifest_sha256": _sha("protocol"),
        "reserve_identity_read_count": 0,
        "reserve_payload_read_count": 0,
        "status": NORMALIZATION_RECOVERY_V6_TRAINING_AUDIT_STATUS,
        "summary": {
            "audit_outcome": "FACILITY_PASS_CAPABILITY_FAIL",
            "capability_gate_pass": 0,
            "facility_failure_count": 0,
            "identity_false_change_count": 0,
            "non_identity_exact_count": 18,
            "outcome_counts": {
                "EXACT": 1510,
                "UNKNOWN": 36740,
                "WRONG": 0,
            },
            "simulation_strategy_reference_equal": 1,
        },
        "teacher_api_llm_call_count": 0,
    }
    encoded = canonical_json_line(manifest)
    (root / "manifest.json").write_bytes(encoded)
    with pytest.raises(BroadQaExternalDataError, match="冻结边界"):
        pack_module._read_audit_manifest(
            root,
            expected_manifest_sha256=hashlib.sha256(encoded).hexdigest(),
            expected_protocol_manifest_sha256=manifest[
                "protocol_manifest_sha256"],
            expected_predecessor_pack_manifest_sha256=manifest[
                "predecessor_rule_pack_manifest_sha256"],
            expected_pack_manifest_sha256=manifest[
                "pack_manifest_sha256"],
        )
