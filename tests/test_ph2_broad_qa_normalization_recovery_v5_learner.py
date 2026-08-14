"""Recovery-v5 learner、checkpoint、结构边界与 disabled pack 测试。"""
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
    ph2_broad_qa_normalization_recovery_v5_learner as learner,
)
from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v3_training_records import (
    GODOT_SOURCE_FAMILY,
    GODOT_SOURCE_POLICY_SCOPE,
    THUNDERBIRD_SOURCE_FAMILY,
    THUNDERBIRD_SOURCE_POLICY_SCOPE,
    normalization_recovery_v3_pair_observation,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v4_vscode_source_pack import (
    VSCODE_SOURCE_FAMILY,
    VSCODE_SOURCE_POLICY_SCOPE,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_learning_records import (
    derive_normalization_recovery_v5_learning_outputs,
    normalization_recovery_v5_output_payloads,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_libreoffice_source_pack import (
    LIBREOFFICE_SOURCE_FAMILY,
    LIBREOFFICE_SOURCE_POLICY_SCOPE,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_rule_pack import (
    publish_normalization_recovery_v5_rule_pack,
    read_normalization_recovery_v5_rule_pack,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_training_records import (
    RECOVERY_V5_TARGET_POLICY_SCOPE,
    derive_normalization_recovery_v5_fragments,
    derive_normalization_recovery_v5_groups,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes


def _sha(value: str) -> str:
    """返回测试 identity。"""
    return hashlib.sha256(value.encode()).hexdigest()


_POLICY_BY_FAMILY = {
    GODOT_SOURCE_FAMILY: GODOT_SOURCE_POLICY_SCOPE,
    LIBREOFFICE_SOURCE_FAMILY: LIBREOFFICE_SOURCE_POLICY_SCOPE,
    VSCODE_SOURCE_FAMILY: VSCODE_SOURCE_POLICY_SCOPE,
    THUNDERBIRD_SOURCE_FAMILY: THUNDERBIRD_SOURCE_POLICY_SCOPE,
}


def _observation(
        identity: str,
        *,
        source_family: str,
        input_text: str,
        output_text: str,
        structure_tokens: tuple[str, ...] = (),
        ) -> dict[str, object]:
    """构造带完整 v3/v5 schema 的 synthetic TRAIN observation。"""
    return normalization_recovery_v3_pair_observation(
        source_family=source_family,
        source_policy_scope=_POLICY_BY_FAMILY[source_family],
        license_id="MPL-2.0" if source_family in {
            LIBREOFFICE_SOURCE_FAMILY, THUNDERBIRD_SOURCE_FAMILY} else "MIT",
        source_pack_manifest_sha256=_sha(f"pack:{source_family}"),
        source_pair_id=_sha(f"pair:{identity}"),
        input_text=input_text,
        output_text=output_text,
        structure_tokens=structure_tokens,
        source_commitment={"test_identity": identity},
    )


def _work(
        observations: tuple[dict[str, object], ...],
        fragments: tuple[dict[str, object], ...],
        groups: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, object], ...]:
    """按 v5 protocol 固定三阶段顺序构造 ordered work。"""
    sources = (
        ("PAIR_OBSERVATION_INGEST", "PAIR_OBSERVATION",
         observations, "observation_id"),
        ("PHRASE_FRAGMENT_INGEST", "PHRASE_FRAGMENT",
         fragments, "fragment_id"),
        ("PHRASE_GROUP_RESOLUTION", "PHRASE_GROUP", groups, "group_id"),
    )
    values = []
    for phase, kind, records, identity_key in sources:
        for record in records:
            identity = {
                "phase": phase,
                "record_id": record[identity_key],
                "work_kind": kind,
            }
            values.append({
                **identity,
                "format_version": 1,
                "record_kind": "NORMALIZATION_RECOVERY_V5_WORK_ITEM_V1",
                "work_id": hashlib.sha256(
                    canonical_json_bytes(identity)).hexdigest(),
                "work_ordinal": len(values),
            })
    return tuple(values)


def _material(protocol_sha: str):
    """构造四来源、四 rule class、identity 与结构 token 的 TRAIN。"""
    observations = (
        _observation(
            "equal-godot", source_family=GODOT_SOURCE_FAMILY,
            input_text="開啟檔案", output_text="打开文件"),
        _observation(
            "equal-vscode", source_family=VSCODE_SOURCE_FAMILY,
            input_text="開啟檔案", output_text="打开文件"),
        _observation(
            "equal-refute", source_family=THUNDERBIRD_SOURCE_FAMILY,
            input_text="不要開啟檔案", output_text="不要開啟檔案"),
        _observation(
            "variable-godot", source_family=GODOT_SOURCE_FAMILY,
            input_text="建立新檔案", output_text="新建文件"),
        _observation(
            "variable-vscode", source_family=VSCODE_SOURCE_FAMILY,
            input_text="建立新檔案", output_text="新建文件"),
        _observation(
            "variable-libreoffice", source_family=LIBREOFFICE_SOURCE_FAMILY,
            input_text="建立新檔案", output_text="新建文件"),
        _observation(
            "variable-refute", source_family=THUNDERBIRD_SOURCE_FAMILY,
            input_text="請勿建立新檔案", output_text="請勿建立新檔案"),
        _observation(
            "source-one", source_family=VSCODE_SOURCE_FAMILY,
            input_text="偏好設定", output_text="设置"),
        _observation(
            "source-two", source_family=VSCODE_SOURCE_FAMILY,
            input_text="偏好設定", output_text="设置"),
        _observation(
            "source-refute", source_family=VSCODE_SOURCE_FAMILY,
            input_text="不要偏好設定", output_text="不要偏好設定"),
        _observation(
            "structured-godot", source_family=GODOT_SOURCE_FAMILY,
            input_text="<b>開啟</b>", output_text="<b>打开</b>",
            structure_tokens=("HTML_START:b", "HTML_END:b")),
        _observation(
            "structured-libreoffice", source_family=LIBREOFFICE_SOURCE_FAMILY,
            input_text="<b>開啟</b>", output_text="<b>打开</b>",
            structure_tokens=("HTML_START:b", "HTML_END:b")),
        _observation(
            "structured-refute", source_family=THUNDERBIRD_SOURCE_FAMILY,
            input_text="不要<b>開啟</b>", output_text="不要<b>開啟</b>",
            structure_tokens=("HTML_START:b", "HTML_END:b")),
    )
    fragments = derive_normalization_recovery_v5_fragments(observations)
    groups = derive_normalization_recovery_v5_groups(fragments)
    manifest = {
        "learner_contract": {
            "identity_preservation_hard_gate_required": 1,
            "negative_evidence_required_before_execution": 1,
            "source_scoped_candidate_target_upgrade_allowed": 0,
            "target_equal_length_min_distinct_source_family_count": 2,
            "target_variable_length_min_distinct_source_family_count": 3,
            "target_variable_length_two_family_replicated_support_allowed": 1,
            "whole_input_exact_precedes_phrase_lexicon": 1,
        },
        "manifest_sha256": protocol_sha,
        "target_policy_scope": RECOVERY_V5_TARGET_POLICY_SCOPE,
    }
    return manifest, observations, fragments, groups, _work(
        observations, fragments, groups)


def _install_material(
        monkeypatch: pytest.MonkeyPatch,
        protocol_sha: str,
        ) -> tuple[dict[str, object], ...]:
    """让共享 runtime 只消费当前 synthetic v5 TRAIN。"""
    material = _material(protocol_sha)
    monkeypatch.setattr(
        learner,
        "read_normalization_recovery_v5_learner_input",
        lambda *args, **kwargs: material,
    )
    monkeypatch.setattr(
        learner_runtime, "require_k_run_root",
        lambda value, *, label: Path(value).resolve())
    monkeypatch.setattr(
        pack_runtime, "require_k_run_root",
        lambda value, *, label: Path(value).resolve())
    return material


def test_v5_fresh_resume_authority_structure_identity_and_disabled_pack(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """fresh/resume 必须保留强 authority、结构、identity 与字节等价。"""
    protocol_sha = _sha("v5-protocol")
    material = _install_material(monkeypatch, protocol_sha)
    protocol_dir = tmp_path / "protocol"
    protocol_dir.mkdir()
    fresh = tmp_path / "fresh"
    resumed = tmp_path / "resumed"
    fresh_report = learner.run_normalization_recovery_v5_learner(
        run_root=tmp_path,
        protocol_dir=protocol_dir,
        expected_protocol_manifest_sha256=protocol_sha,
        run_dir=fresh,
        run_id=_sha("v5-fresh"),
        mode="fresh",
        checkpoint_interval=7,
    )
    partial = learner.run_normalization_recovery_v5_learner(
        run_root=tmp_path,
        protocol_dir=protocol_dir,
        expected_protocol_manifest_sha256=protocol_sha,
        run_dir=resumed,
        run_id=_sha("v5-resumed"),
        mode="fresh",
        checkpoint_interval=5,
        stop_after=11,
    )
    assert partial["status"] == learner.NORMALIZATION_RECOVERY_V5_CHECKPOINT_OPEN
    resumed_report = learner.run_normalization_recovery_v5_learner(
        run_root=tmp_path,
        protocol_dir=protocol_dir,
        expected_protocol_manifest_sha256=protocol_sha,
        run_dir=resumed,
        run_id=_sha("v5-resumed"),
        mode="resume",
        checkpoint_interval=9,
    )
    assert fresh_report["semantic_result_sha256"] == (
        resumed_report["semantic_result_sha256"])
    summary = fresh_report["summary"]
    assert summary["identity_observation_count"] == 4
    assert summary["target_rule_class_counts"][
        "WHOLE_INPUT_EQUAL_LENGTH"] >= 2
    assert summary["target_rule_class_counts"][
        "WHOLE_INPUT_VARIABLE_LENGTH"] >= 1
    assert summary["target_candidate_group_rule_class_counts"][
        "WHOLE_INPUT_VARIABLE_LENGTH"] == (
            summary["target_rule_class_counts"][
                "WHOLE_INPUT_VARIABLE_LENGTH"]
            + summary["target_deferred_candidate_rule_class_counts"][
                "WHOLE_INPUT_VARIABLE_LENGTH"])
    assert summary["source_rule_class_counts"][
        "WHOLE_INPUT_VARIABLE_LENGTH"] >= 1
    assert summary["refute_identity_preservation_count"] > 0
    assert summary["negative_evidence_closed_rule_count"] == (
        summary["target_rule_count"] + summary["source_rule_count"])
    fresh_manifest, fresh_outputs = learner.read_normalization_recovery_v5_learner(
        fresh,
        protocol_dir=protocol_dir,
        expected_protocol_manifest_sha256=protocol_sha,
    )
    resumed_manifest, resumed_outputs = learner.read_normalization_recovery_v5_learner(
        resumed,
        protocol_dir=protocol_dir,
        expected_protocol_manifest_sha256=protocol_sha,
    )
    assert normalization_recovery_v5_output_payloads(fresh_outputs) == (
        normalization_recovery_v5_output_payloads(resumed_outputs))
    assert fresh_manifest["resume_markers"]["record_count"] == 0
    assert resumed_manifest["resume_markers"]["record_count"] == 1
    variable = next(item for item in fresh_outputs[
        "target-phrase-rules.jsonl"]
        if item["rule_class"] == "WHOLE_INPUT_VARIABLE_LENGTH")
    assert variable["authority_basis"] == (
        "VARIABLE_LENGTH_WHOLE_INPUT_STRONG_CONSENSUS")
    assert len(variable["source_families"]) == 3
    structured = next(item for item in fresh_outputs[
        "target-phrase-rules.jsonl"]
        if item["structure_token_variants"])
    assert structured["application_scope"]["structure_match_required"] == 1
    assert structured["structure_token_variants"] == [[
        "HTML_START:b", "HTML_END:b"]]
    assert all(item["negative_evidence_ids"] for item in (
        fresh_outputs["target-phrase-rules.jsonl"]
        + fresh_outputs["source-phrase-rules.jsonl"]))

    pack_dir = tmp_path / "pack"
    pack_report = publish_normalization_recovery_v5_rule_pack(
        run_root=tmp_path,
        protocol_dir=protocol_dir,
        expected_protocol_manifest_sha256=protocol_sha,
        fresh_run_dir=fresh,
        resumed_run_dir=resumed,
        target_dir=pack_dir,
    )
    pack_manifest, pack_outputs = read_normalization_recovery_v5_rule_pack(
        pack_dir,
        protocol_dir=protocol_dir,
        expected_protocol_manifest_sha256=protocol_sha,
        expected_pack_manifest_sha256=str(pack_report["manifest_sha256"]),
    )
    assert pack_manifest["production_enabled"] == 0
    assert pack_manifest["fresh_resume_output_bytes_equal"] == 1
    assert pack_manifest["evaluation_or_held_out_payload_read_count"] == 0
    assert normalization_recovery_v5_output_payloads(pack_outputs) == (
        normalization_recovery_v5_output_payloads(fresh_outputs))
    outputs, _summary, emissions = derive_normalization_recovery_v5_learning_outputs(
        protocol_manifest=material[0],
        observations=material[1],
        fragments=material[2],
        groups=material[3],
        work=material[4],
    )
    assert outputs
    drifted = list(emissions)
    drifted[-1] = {
        **drifted[-1],
        "result_increment": int(drifted[-1]["result_increment"]) + 1,
    }
    with pytest.raises(BroadQaExternalDataError, match="checkpoint"):
        learner.validate_normalization_recovery_v5_checkpoint_chain(
            chain_path=fresh / "checkpoints.jsonl",
            run_id=_sha("v5-fresh"),
            protocol_manifest_sha256=protocol_sha,
            work=material[4],
            emission_counts=tuple(drifted),
            require_complete=True,
        )


def test_v5_rejects_authority_and_whole_input_structure_tamper() -> None:
    """authority basis 与 WHOLE_INPUT 结构 token 均不可在 learner 内漂移。"""
    protocol_sha = _sha("v5-tamper")
    manifest, observations, fragments, groups, work = _material(protocol_sha)
    target_index = next(index for index, item in enumerate(groups)
                        if item["authority_basis"]
                        == "VARIABLE_LENGTH_WHOLE_INPUT_STRONG_CONSENSUS")
    tampered_groups = list(groups)
    tampered_groups[target_index] = {
        **tampered_groups[target_index],
        "authority_basis": "EQUAL_LENGTH_WHOLE_INPUT_TWO_FAMILY_CONSENSUS",
    }
    with pytest.raises(BroadQaExternalDataError, match="重派生"):
        derive_normalization_recovery_v5_learning_outputs(
            protocol_manifest=manifest,
            observations=observations,
            fragments=fragments,
            groups=tuple(tampered_groups),
            work=work,
        )
    structured_index = next(index for index, item in enumerate(fragments)
                            if item["fragment_kind"] == "WHOLE_INPUT"
                            and item.get("structure_tokens"))
    tampered_fragments = list(fragments)
    tampered_fragments[structured_index] = {
        **tampered_fragments[structured_index],
        "structure_tokens": [],
    }
    with pytest.raises(BroadQaExternalDataError, match="WHOLE_INPUT"):
        derive_normalization_recovery_v5_learning_outputs(
            protocol_manifest=manifest,
            observations=observations,
            fragments=tuple(tampered_fragments),
            groups=groups,
            work=work,
        )


def test_v5_injected_finalize_failure_resumes_from_complete_checkpoint(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """终态 checkpoint 后输出写入中断必须可恢复且语义不漂移。"""
    protocol_sha = _sha("v5-injected-resume")
    _install_material(monkeypatch, protocol_sha)
    protocol_dir = tmp_path / "protocol"
    protocol_dir.mkdir()
    clean = tmp_path / "clean"
    injected = tmp_path / "injected"
    clean_report = learner.run_normalization_recovery_v5_learner(
        run_root=tmp_path,
        protocol_dir=protocol_dir,
        expected_protocol_manifest_sha256=protocol_sha,
        run_dir=clean,
        run_id=_sha("v5-clean"),
        mode="fresh",
        checkpoint_interval=13,
    )
    original = learner_runtime.write_or_verify
    calls = {"count": 0}

    def _fail_second_write(path: Path, payload: bytes, *, label: str) -> None:
        """在第二个语义输出前注入一次可恢复失败。"""
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("injected finalize failure")
        original(path, payload, label=label)

    monkeypatch.setattr(learner_runtime, "write_or_verify", _fail_second_write)
    with pytest.raises(OSError, match="injected"):
        learner.run_normalization_recovery_v5_learner(
            run_root=tmp_path,
            protocol_dir=protocol_dir,
            expected_protocol_manifest_sha256=protocol_sha,
            run_dir=injected,
            run_id=_sha("v5-injected"),
            mode="fresh",
            checkpoint_interval=11,
        )
    monkeypatch.setattr(learner_runtime, "write_or_verify", original)
    recovered = learner.run_normalization_recovery_v5_learner(
        run_root=tmp_path,
        protocol_dir=protocol_dir,
        expected_protocol_manifest_sha256=protocol_sha,
        run_dir=injected,
        run_id=_sha("v5-injected"),
        mode="resume",
        checkpoint_interval=17,
    )
    assert recovered["semantic_result_sha256"] == (
        clean_report["semantic_result_sha256"])
    _clean_manifest, clean_outputs = learner.read_normalization_recovery_v5_learner(
        clean,
        protocol_dir=protocol_dir,
        expected_protocol_manifest_sha256=protocol_sha,
    )
    _recovered_manifest, recovered_outputs = (
        learner.read_normalization_recovery_v5_learner(
            injected,
            protocol_dir=protocol_dir,
            expected_protocol_manifest_sha256=protocol_sha,
        ))
    assert normalization_recovery_v5_output_payloads(clean_outputs) == (
        normalization_recovery_v5_output_payloads(recovered_outputs))
