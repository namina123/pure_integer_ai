"""Recovery-v4 scoped learner 与 disabled pack 专项测试。"""
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
    ph2_broad_qa_normalization_recovery_v4_learner as learner,
)
from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v3_training_records import (
    GODOT_SOURCE_FAMILY,
    GODOT_SOURCE_POLICY_SCOPE,
    THUNDERBIRD_SOURCE_FAMILY,
    THUNDERBIRD_SOURCE_POLICY_SCOPE,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v4_learning_records import (
    derive_normalization_recovery_v4_learning_outputs,
    normalization_recovery_v4_output_payloads,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v4_phrase_runtime import (
    compile_normalization_recovery_v4_phrase_program,
    execute_normalization_recovery_v4_phrase_program,
    reference_normalization_recovery_v4_phrase_program,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v4_rule_pack import (
    publish_normalization_recovery_v4_rule_pack,
    read_normalization_recovery_v4_rule_pack,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v4_training_audit_records import (
    derive_normalization_recovery_v4_training_audit,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v4_training_audit import (
    publish_normalization_recovery_v4_training_audit,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v4_training_records import (
    RECOVERY_V4_TARGET_POLICY_SCOPE,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v4_vscode_source_pack import (
    VSCODE_SOURCE_FAMILY,
    VSCODE_SOURCE_POLICY_SCOPE,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes


def _sha(value: str) -> str:
    """返回测试 identity。"""
    return hashlib.sha256(value.encode()).hexdigest()


def _observation(
        identity: str,
        *,
        source_family: str,
        source_policy_scope: str,
        input_text: str,
        output_text: str,
        ) -> dict[str, object]:
    """构造 learner 所需的最小物化 observation。"""
    return {
        "format_version": 1,
        "input_text": input_text,
        "license_id": "MIT",
        "observation_id": _sha(f"observation:{identity}"),
        "output_text": output_text,
        "record_kind": "NORMALIZATION_RECOVERY_V3_PAIR_OBSERVATION_V1",
        "source_commitment": {"test_identity": identity},
        "source_family": source_family,
        "source_pack_manifest_sha256": _sha(f"pack:{source_family}"),
        "source_pair_id": _sha(f"pair:{identity}"),
        "source_policy_scope": source_policy_scope,
    }


def _fragment(
        identity: str,
        *,
        observation: dict[str, object],
        input_text: str,
        output_text: str,
        fragment_kind: str,
        ) -> dict[str, object]:
    """构造与 observation 全 span 对齐的最小 fragment。"""
    return {
        "equal_length": int(len(input_text) == len(output_text)),
        "format_version": 1,
        "fragment_id": _sha(f"fragment:{identity}"),
        "fragment_kind": fragment_kind,
        "input_end": len(input_text),
        "input_start": 0,
        "input_text": input_text,
        "license_id": observation["license_id"],
        "observation_id": observation["observation_id"],
        "output_end": len(output_text),
        "output_start": 0,
        "output_text": output_text,
        "record_kind": "NORMALIZATION_RECOVERY_V3_PHRASE_FRAGMENT_V1",
        "source_family": observation["source_family"],
        "source_policy_scope": observation["source_policy_scope"],
    }


def _group(
        identity: str,
        *,
        input_text: str,
        output_text: str,
        fragments: tuple[dict[str, object], ...],
        disposition: str,
        scope_kind: str,
        ) -> dict[str, object]:
    """构造 target 或 source-only v4 group。"""
    families = sorted({str(item["source_family"]) for item in fragments})
    policies = sorted({str(item["source_policy_scope"]) for item in fragments})
    return {
        "candidate_scope_kind": scope_kind,
        "disposition": disposition,
        "format_version": 1,
        "fragment_kind": fragments[0]["fragment_kind"],
        "group_id": _sha(f"group:{identity}"),
        "input_text": input_text,
        "negative_evidence_required_before_execution": 1,
        "output_variants": [{
            "fragment_ids": sorted(str(item["fragment_id"])
                                   for item in fragments),
            "output_text": output_text,
            "source_families": families,
            "source_policy_scopes": policies,
            "support_count": len(fragments),
        }],
        "record_kind": "NORMALIZATION_RECOVERY_V4_PHRASE_GROUP_V1",
        "source_families": families,
        "source_policy_scopes": policies,
        "target_policy_scope": (
            RECOVERY_V4_TARGET_POLICY_SCOPE
            if scope_kind == "TARGET_CROSS_FAMILY" else ""),
        "unscoped_execution_allowed": 0,
    }


def _work(
        observations: tuple[dict[str, object], ...],
        fragments: tuple[dict[str, object], ...],
        groups: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, object], ...]:
    """按 protocol 固定三阶段顺序构造 work。"""
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
                "record_kind": "NORMALIZATION_RECOVERY_V4_WORK_ITEM_V1",
                "work_id": hashlib.sha256(
                    canonical_json_bytes(identity)).hexdigest(),
                "work_ordinal": len(values),
            })
    return tuple(values)


def _material(protocol_sha: str):
    """构造含 target/source SUPPORT 与各自 REFUTE 的完整 TRAIN material。"""
    godot = _observation(
        "target-godot",
        source_family=GODOT_SOURCE_FAMILY,
        source_policy_scope=GODOT_SOURCE_POLICY_SCOPE,
        input_text="開啟檔案",
        output_text="打开文件",
    )
    vscode_target = _observation(
        "target-vscode",
        source_family=VSCODE_SOURCE_FAMILY,
        source_policy_scope=VSCODE_SOURCE_POLICY_SCOPE,
        input_text="開啟檔案",
        output_text="打开文件",
    )
    target_refute = _observation(
        "target-refute",
        source_family=THUNDERBIRD_SOURCE_FAMILY,
        source_policy_scope=THUNDERBIRD_SOURCE_POLICY_SCOPE,
        input_text="不要開啟檔案",
        output_text="不要開啟檔案",
    )
    thunderbird_target = _observation(
        "target-thunderbird",
        source_family=THUNDERBIRD_SOURCE_FAMILY,
        source_policy_scope=THUNDERBIRD_SOURCE_POLICY_SCOPE,
        input_text="開啟檔案",
        output_text="打开文件",
    )
    vscode_source_one = _observation(
        "source-one",
        source_family=VSCODE_SOURCE_FAMILY,
        source_policy_scope=VSCODE_SOURCE_POLICY_SCOPE,
        input_text="來源",
        output_text="来源",
    )
    vscode_source_two = _observation(
        "source-two",
        source_family=VSCODE_SOURCE_FAMILY,
        source_policy_scope=VSCODE_SOURCE_POLICY_SCOPE,
        input_text="來源",
        output_text="来源",
    )
    source_refute = _observation(
        "source-refute",
        source_family=VSCODE_SOURCE_FAMILY,
        source_policy_scope=VSCODE_SOURCE_POLICY_SCOPE,
        input_text="不要來源",
        output_text="不要來源",
    )
    observations = (
        godot, vscode_target, thunderbird_target, target_refute,
        vscode_source_one, vscode_source_two, source_refute,
    )
    target_fragments = (
        _fragment(
            "target-godot", observation=godot,
            input_text="開啟檔案", output_text="打开文件",
            fragment_kind="EDIT_CORE"),
        _fragment(
            "target-vscode", observation=vscode_target,
            input_text="開啟檔案", output_text="打开文件",
            fragment_kind="EDIT_CORE"),
        _fragment(
            "target-thunderbird", observation=thunderbird_target,
            input_text="開啟檔案", output_text="打开文件",
            fragment_kind="EDIT_CORE"),
    )
    source_fragments = (
        _fragment(
            "source-one", observation=vscode_source_one,
            input_text="來源", output_text="来源", fragment_kind="CONTEXT_HUNK"),
        _fragment(
            "source-two", observation=vscode_source_two,
            input_text="來源", output_text="来源", fragment_kind="CONTEXT_HUNK"),
    )
    fragments = target_fragments + source_fragments
    groups = (
        _group(
            "target", input_text="開啟檔案", output_text="打开文件",
            fragments=target_fragments,
            disposition="CROSS_FAMILY_CONSENSUS_CANDIDATE",
            scope_kind="TARGET_CROSS_FAMILY"),
        _group(
            "source", input_text="來源", output_text="来源",
            fragments=source_fragments,
            disposition="SOURCE_SCOPED_CANDIDATE",
            scope_kind="SOURCE_ONLY"),
    )
    manifest = {
        "learner_contract": {
            "source_scoped_candidate_target_upgrade_allowed": 0,
            "target_candidate_min_distinct_ui_source_family_count": 2,
        },
        "manifest_sha256": protocol_sha,
        "target_policy_scope": RECOVERY_V4_TARGET_POLICY_SCOPE,
    }
    return manifest, observations, fragments, groups, _work(
        observations, fragments, groups)


def _install_material(
        monkeypatch: pytest.MonkeyPatch,
        protocol_sha: str,
        ) -> tuple[dict[str, object], ...]:
    """让 materialized runtime 只消费当前 synthetic TRAIN。"""
    material = _material(protocol_sha)
    monkeypatch.setattr(
        learner,
        "read_normalization_recovery_v4_learner_input",
        lambda *args, **kwargs: material,
    )
    monkeypatch.setattr(
        learner_runtime, "require_k_run_root",
        lambda value, *, label: Path(value).resolve())
    monkeypatch.setattr(
        pack_runtime, "require_k_run_root",
        lambda value, *, label: Path(value).resolve())
    return material


def test_v4_fresh_resume_scope_evidence_and_disabled_pack(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """两类 rule 必须独立闭合 scope、Evidence、defeater 与字节等价。"""
    protocol_sha = _sha("v4-protocol")
    _install_material(monkeypatch, protocol_sha)
    protocol_dir = tmp_path / "protocol"
    protocol_dir.mkdir()
    fresh = tmp_path / "fresh"
    resumed = tmp_path / "resumed"
    fresh_report = learner.run_normalization_recovery_v4_learner(
        run_root=tmp_path,
        protocol_dir=protocol_dir,
        expected_protocol_manifest_sha256=protocol_sha,
        run_dir=fresh,
        run_id=_sha("v4-fresh"),
        mode="fresh",
        checkpoint_interval=3,
    )
    partial = learner.run_normalization_recovery_v4_learner(
        run_root=tmp_path,
        protocol_dir=protocol_dir,
        expected_protocol_manifest_sha256=protocol_sha,
        run_dir=resumed,
        run_id=_sha("v4-resumed"),
        mode="fresh",
        checkpoint_interval=2,
        stop_after=5,
    )
    assert partial["status"] == learner.NORMALIZATION_RECOVERY_V4_CHECKPOINT_OPEN
    resumed_report = learner.run_normalization_recovery_v4_learner(
        run_root=tmp_path,
        protocol_dir=protocol_dir,
        expected_protocol_manifest_sha256=protocol_sha,
        run_dir=resumed,
        run_id=_sha("v4-resumed"),
        mode="resume",
        checkpoint_interval=4,
    )
    assert fresh_report["semantic_result_sha256"] == (
        resumed_report["semantic_result_sha256"])
    assert fresh_report["summary"]["target_rule_count"] == 1
    assert fresh_report["summary"]["source_rule_count"] == 1
    fresh_manifest, fresh_outputs = learner.read_normalization_recovery_v4_learner(
        fresh,
        protocol_dir=protocol_dir,
        expected_protocol_manifest_sha256=protocol_sha,
    )
    resumed_manifest, resumed_outputs = learner.read_normalization_recovery_v4_learner(
        resumed,
        protocol_dir=protocol_dir,
        expected_protocol_manifest_sha256=protocol_sha,
    )
    assert normalization_recovery_v4_output_payloads(fresh_outputs) == (
        normalization_recovery_v4_output_payloads(resumed_outputs))
    assert fresh_manifest["resume_markers"]["record_count"] == 0
    assert resumed_manifest["resume_markers"]["record_count"] == 1
    target = fresh_outputs["target-phrase-rules.jsonl"][0]
    source = fresh_outputs["source-phrase-rules.jsonl"][0]
    assert target["target_policy_scope"] == RECOVERY_V4_TARGET_POLICY_SCOPE
    assert target["source_execution_family"] == ""
    assert len(target["source_families"]) == 3
    assert source["target_policy_scope"] == ""
    assert source["source_execution_family"] == VSCODE_SOURCE_FAMILY
    assert source["source_families"] == [VSCODE_SOURCE_FAMILY]
    assert target["positive_evidence_ids"] and target["negative_evidence_ids"]
    assert source["positive_evidence_ids"] and source["negative_evidence_ids"]

    pack_dir = tmp_path / "pack"
    pack_report = publish_normalization_recovery_v4_rule_pack(
        run_root=tmp_path,
        protocol_dir=protocol_dir,
        expected_protocol_manifest_sha256=protocol_sha,
        fresh_run_dir=fresh,
        resumed_run_dir=resumed,
        target_dir=pack_dir,
    )
    pack_manifest, pack_outputs = read_normalization_recovery_v4_rule_pack(
        pack_dir,
        protocol_dir=protocol_dir,
        expected_protocol_manifest_sha256=protocol_sha,
        expected_pack_manifest_sha256=str(pack_report["manifest_sha256"]),
    )
    assert pack_manifest["production_enabled"] == 0
    assert pack_manifest["fresh_resume_output_bytes_equal"] == 1
    assert normalization_recovery_v4_output_payloads(pack_outputs) == (
        normalization_recovery_v4_output_payloads(fresh_outputs))
    program = compile_normalization_recovery_v4_phrase_program(
        rule_pack_manifest_sha256=str(pack_report["manifest_sha256"]),
        target_phrase_rules=pack_outputs["target-phrase-rules.jsonl"],
        source_phrase_rules=pack_outputs["source-phrase-rules.jsonl"],
        defeaters=pack_outputs["defeaters.jsonl"],
        target_overlap_index=pack_outputs["target-overlap-index.jsonl"],
        source_overlap_index=pack_outputs["source-overlap-index.jsonl"],
    )
    assert execute_normalization_recovery_v4_phrase_program(
        program, "開啟檔案")["output_text"] == "打开文件"
    assert execute_normalization_recovery_v4_phrase_program(
        program, "來源")["output_text"] == "來源"
    scoped = execute_normalization_recovery_v4_phrase_program(
        program, "來源", source_family=VSCODE_SOURCE_FAMILY)
    assert scoped["output_text"] == "来源"
    assert scoped == reference_normalization_recovery_v4_phrase_program(
        program, "來源", source_family=VSCODE_SOURCE_FAMILY)
    assert execute_normalization_recovery_v4_phrase_program(
        program, "不要來源", source_family=VSCODE_SOURCE_FAMILY
    )["output_text"] == "不要來源"


def test_v4_records_reject_source_candidate_target_upgrade() -> None:
    """单来源 group 写入 target scope 必须在 learner 前 fail closed。"""
    protocol_sha = _sha("v4-protocol-tamper")
    manifest, observations, fragments, groups, work = _material(protocol_sha)
    tampered = list(groups)
    tampered[1] = {
        **tampered[1],
        "target_policy_scope": RECOVERY_V4_TARGET_POLICY_SCOPE,
    }
    with pytest.raises(BroadQaExternalDataError, match="phrase group"):
        derive_normalization_recovery_v4_learning_outputs(
            protocol_manifest=manifest,
            observations=observations,
            fragments=fragments,
            groups=tuple(tampered),
            work=work,
        )


def test_v4_training_audit_relearns_all_three_loso_directions() -> None:
    """三方向必须各自重学并达到非零 EXACT、零 WRONG。"""
    protocol_sha = _sha("v4-audit-protocol")
    manifest, observations, fragments, groups, work = _material(protocol_sha)
    outputs, _summary, _counts = derive_normalization_recovery_v4_learning_outputs(
        protocol_manifest=manifest,
        observations=observations,
        fragments=fragments,
        groups=groups,
        work=work,
    )
    runtime_cases, loso, summary = derive_normalization_recovery_v4_training_audit(
        protocol_manifest=manifest,
        observations=observations,
        fragments=fragments,
        groups=groups,
        pack_manifest={
            "manifest_sha256": _sha("v4-audit-pack"),
            "mastery_claimed": 0,
            "production_enabled": 0,
        },
        outputs=outputs,
    )
    assert runtime_cases
    assert len(loso) == 3
    assert summary["facility_failure_count"] == 0
    assert summary["capability_gate_pass"] == 1
    assert all(item["outcome_counts"]["EXACT"] > 0 for item in loso)
    assert all(item["outcome_counts"]["WRONG"] == 0 for item in loso)


def test_v4_training_audit_rejects_non_k_root_before_write(
        tmp_path: Path,
        ) -> None:
    """正式 audit 不得把 TRAIN artifact 回退到非 K 盘。"""
    target = tmp_path / "audit"
    with pytest.raises(BroadQaExternalDataError, match="K 盘"):
        publish_normalization_recovery_v4_training_audit(
            run_root=tmp_path,
            protocol_dir=tmp_path / "protocol",
            expected_protocol_manifest_sha256="a" * 64,
            pack_dir=tmp_path / "pack",
            expected_pack_manifest_sha256="b" * 64,
            target_dir=target,
        )
    assert not target.exists()
