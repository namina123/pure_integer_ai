"""recovery-v4 candidate、family、evaluator 与唯一 runner 专项。"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_recovery_v4_evaluation_family as family_module,
)
from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_recovery_v4_evaluation_runner as runner_module,
)
from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_candidate_compile import (
    compile_normalization_recovery_candidate,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v4_candidate import (
    NORMALIZATION_RECOVERY_V4_FIREFOX_TARGET_POLICY_SCOPE,
    compile_normalization_recovery_v4_candidate,
    execute_normalization_recovery_v4_candidate,
    execute_normalization_recovery_v4_candidate_batch,
    reference_normalization_recovery_v4_candidate,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v4_candidate_pack import (
    profile_normalization_recovery_v4_candidate,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v4_evaluation_family import (
    NORMALIZATION_RECOVERY_V4_EVALUATION_CODE_FILES,
    build_normalization_recovery_v4_evaluation_family_freeze,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v4_evaluation_runner import (
    run_normalization_recovery_v4_formal_evaluation,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v4_evaluator import (
    evaluate_normalization_recovery_v4_candidate,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v4_learning_records import (
    derive_normalization_recovery_v4_learning_outputs,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v4_vscode_source_pack import (
    VSCODE_SOURCE_POLICY_SCOPE,
)
from test_ph2_broad_qa_normalization_recovery_candidate import (
    _candidate_material,
)
from test_ph2_broad_qa_normalization_recovery_v4_learner import _material


def _sha(value: str) -> str:
    """构造 synthetic SHA-256 identity。"""
    return hashlib.sha256(value.encode()).hexdigest()


def _program(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> dict[str, object]:
    """从既有 synthetic TRAIN 真实派生 base 与 v4 composite program。"""
    evaluation, base_pack, base_outputs = _candidate_material(
        tmp_path / "base", monkeypatch)
    base = compile_normalization_recovery_candidate(
        evaluation_protocol_manifest=evaluation,
        rule_pack_manifest=base_pack,
        outputs=base_outputs,
    )
    protocol_sha = _sha("v4 protocol")
    manifest, observations, fragments, groups, work = _material(protocol_sha)
    outputs, _summary, _counts = (
        derive_normalization_recovery_v4_learning_outputs(
            protocol_manifest=manifest,
            observations=observations,
            fragments=fragments,
            groups=groups,
            work=work,
        ))
    values = dict(outputs)
    values["conflict-ledger.jsonl"] = ({
        "conflict_id": _sha("synthetic conflict"),
        "conflict_kind": "TRAIN_OUTPUT_CONFLICT",
        "input_text": "衝突詞",
        "production_enabled": 0,
        "unscoped_execution_allowed": 0,
    },)
    return compile_normalization_recovery_v4_candidate(
        base_program=base,
        base_rule_pack_manifest_sha256=base_pack["manifest_sha256"],
        v4_protocol_manifest_sha256=protocol_sha,
        v4_rule_pack_manifest_sha256=_sha("v4 pack"),
        v4_training_audit_manifest_sha256=_sha("v4 audit"),
        evaluation_commitment_manifest_sha256=_sha("commitment"),
        v4_outputs=values,
    )


def test_candidate_composes_target_source_and_unscoped_boundaries(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """target phrase、source rule、字符 backoff 与 conflict 必须严格分 scope。"""
    program = _program(tmp_path, monkeypatch)
    target = execute_normalization_recovery_v4_candidate(
        program, "開啟檔案",
        policy_scope=NORMALIZATION_RECOVERY_V4_FIREFOX_TARGET_POLICY_SCOPE,
        regional_scope="ZH_CN",
    )
    assert target["output_text"] == "打开文件"
    assert target["projection_used"] == 1
    assert target == reference_normalization_recovery_v4_candidate(
        program, "開啟檔案",
        policy_scope=NORMALIZATION_RECOVERY_V4_FIREFOX_TARGET_POLICY_SCOPE,
        regional_scope="ZH_CN",
    )

    source = execute_normalization_recovery_v4_candidate(
        program, "來源", policy_scope=VSCODE_SOURCE_POLICY_SCOPE)
    assert source["output_text"] == "来源"
    assert any(step["candidate_scope_kind"] == "SOURCE_ONLY"
               for step in source["steps"])
    isolated = execute_normalization_recovery_v4_candidate(
        program, "來源",
        policy_scope=NORMALIZATION_RECOVERY_V4_FIREFOX_TARGET_POLICY_SCOPE,
        regional_scope="ZH_CN",
    )
    assert all(step["candidate_scope_kind"] != "SOURCE_ONLY"
               for step in isolated["steps"])

    conflict = execute_normalization_recovery_v4_candidate(
        program, "衝突詞", policy_scope="")
    assert conflict["output_text"] == "衝突詞"
    assert conflict["scope_mismatch"] == 1
    assert conflict["unscoped_conflict_blocked"] == 1
    assert program["production_enabled"] == 0


def test_candidate_profile_and_six_dimensions_pass_synthetic(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """固定 TRAIN roster 双解释器一致，完整 synthetic reserve 六维通过。"""
    program = _program(tmp_path, monkeypatch)
    profile = profile_normalization_recovery_v4_candidate(program)
    assert profile["indexed"]["failure_count"] == 0
    assert profile["reference"]["mismatch_count"] == 0
    assert profile["indexed_reference_result_bytes_equal"] == 1

    character = next(item for item in program["base_character_rules"]
                     if item["input_text"] != item["output_text"])
    source_sha = _sha("firefox source")
    records = tuple(sorted(({
        "context_sensitive": 0,
        "evaluation_id": _sha("local"),
        "expected_output": character["output_text"],
        "family_keys": ["LOCAL_MAPPING_TRANSFER"],
        "format_version": 2,
        "identity_preservation": 0,
        "input_scalar_count": 1,
        "input_text": character["input_text"],
        "output_scalar_count": 1,
        "source_pack_manifest_sha256": source_sha,
        "split": "RESERVE",
    }, {
        "context_sensitive": 1,
        "evaluation_id": _sha("phrase"),
        "expected_output": "打开文件",
        "family_keys": [
            "END_TO_END_COVERAGE", "INDEPENDENT_CONTEXT_TRANSFER"],
        "format_version": 2,
        "identity_preservation": 0,
        "input_scalar_count": len("開啟檔案"),
        "input_text": "開啟檔案",
        "output_scalar_count": len("打开文件"),
        "source_pack_manifest_sha256": source_sha,
        "split": "RESERVE",
    }), key=lambda item: item["evaluation_id"]))
    commitment = {
        "denominator": {"record_count": len(records)},
        "manifest_sha256": program[
            "evaluation_commitment_manifest_sha256"],
    }
    candidate = {
        "candidate_program_sha256": program["candidate_program_sha256"],
        "manifest_sha256": _sha("candidate manifest"),
    }
    materialization = {
        "evaluation_commitment_manifest_sha256": commitment[
            "manifest_sha256"],
        "firefox_source_pack_manifest_sha256": source_sha,
        "label_materialization_count": len(records),
        "reserve_identity_sha256": _sha("reserve identity"),
        "reserve_payload_read_count": 1,
    }
    report = evaluate_normalization_recovery_v4_candidate(
        commitment=commitment,
        candidate_manifest=candidate,
        program=program,
        materialization=materialization,
        reserve_records=records,
        family_freeze_manifest_sha256=_sha("family"),
    )
    assert report["overall_outcome"] == "PASS"
    assert [item["outcome"] for item in report["dimensions"]] == [
        "PASS"] * 6
    assert report["reserve_payload_read_count"] == 1
    assert report["production_enabled"] == 0


def _family_arguments(tmp_path: Path) -> dict[str, object]:
    """构造 family builder 的完整 synthetic 参数。"""
    values: dict[str, object] = {
        "repository_root": Path(__file__).resolve().parents[1],
        "expected_candidate_manifest_sha256": _sha("candidate"),
        "expected_prior_evaluation_manifest_sha256": _sha("prior"),
        "expected_base_training_manifest_sha256": _sha("base training"),
        "expected_base_rule_pack_manifest_sha256": _sha("base pack"),
        "expected_v4_training_manifest_sha256": _sha("v4 training"),
        "expected_v4_rule_pack_manifest_sha256": _sha("v4 pack"),
        "expected_v4_training_audit_manifest_sha256": _sha("v4 audit"),
        "expected_evaluation_commitment_manifest_sha256": _sha("commitment"),
    }
    for name in (
            "candidate_dir", "prior_evaluation_protocol_dir",
            "base_training_protocol_dir", "base_rule_pack_dir",
            "v4_training_protocol_dir", "v4_rule_pack_dir",
            "v4_training_audit_dir", "evaluation_commitment_dir"):
        path = tmp_path / name
        path.mkdir(parents=True)
        values[name] = path
    return values


def test_family_freezes_code_candidate_and_label_blind_commitment(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """family 只绑定 commitment，不读取 reserve identity 或 label。"""
    arguments = _family_arguments(tmp_path)
    candidate = {
        "manifest_sha256": arguments["expected_candidate_manifest_sha256"],
    }
    program = {
        "base_character_rules": ({"value": 1},),
        "candidate_program_sha256": _sha("program"),
        "conflicts": ({"value": 1},),
        "evaluation_commitment_manifest_sha256": arguments[
            "expected_evaluation_commitment_manifest_sha256"],
        "phrase_program": {"program_sha256": _sha("phrase")},
        "transfer_profile_sha256": _sha("transfer"),
        "v4_protocol_manifest_sha256": arguments[
            "expected_v4_training_manifest_sha256"],
        "v4_rule_pack_manifest_sha256": arguments[
            "expected_v4_rule_pack_manifest_sha256"],
        "v4_training_audit_manifest_sha256": arguments[
            "expected_v4_training_audit_manifest_sha256"],
    }
    commitment = {
        "denominator": {"record_count": 2},
        "dimensions": {"RUNTIME_PRODUCTION_BEHAVIOR": {"bearing": 1}},
        "formal_contract": {"formal_run_count_max": 1},
        "manifest_sha256": arguments[
            "expected_evaluation_commitment_manifest_sha256"],
        "prior_reserve_identity": {"sha256": _sha("reserve")},
        "source_exclusion": {
            "excluded_source_pack_manifest_sha256": _sha("firefox")},
    }
    monkeypatch.setattr(
        family_module, "read_normalization_recovery_v4_candidate_pack",
        lambda *args, **kwargs: (candidate, program, {}))
    monkeypatch.setattr(
        family_module,
        "read_normalization_recovery_v3_evaluation_commitment",
        lambda *args, **kwargs: commitment)
    freeze, _candidate, _program_value, _commitment = (
        build_normalization_recovery_v4_evaluation_family_freeze(**arguments))
    assert freeze["evaluation_run_count"] == 0
    assert freeze["reserve_identity_read_count"] == 0
    assert freeze["reserve_payload_read_count"] == 0
    assert freeze["candidate_freeze"]["candidate_program_sha256"] == (
        _sha("program"))
    assert len(freeze["code_files"]) == len(
        NORMALIZATION_RECOVERY_V4_EVALUATION_CODE_FILES)


def _install_runner(
        monkeypatch: pytest.MonkeyPatch,
        *,
        fail_materialization: bool = False,
        ) -> None:
    """安装不触碰真实 reserve 的 runner synthetic readers。"""
    family = {
        "family_commitment_sha256": _sha("family commitment"),
        "manifest_sha256": _sha("family"),
    }
    candidate = {"manifest_sha256": _sha("candidate")}
    program = {
        "candidate_program_sha256": _sha("program"),
    }
    commitment = {"manifest_sha256": _sha("commitment")}
    monkeypatch.setattr(
        runner_module, "require_normalization_recovery_v4_k_root",
        lambda value: Path(value).resolve())
    monkeypatch.setattr(
        runner_module,
        "read_normalization_recovery_v4_evaluation_family_freeze",
        lambda *args, **kwargs: (family, candidate, program, commitment))
    if fail_materialization:
        def materializer(**kwargs):
            """模拟 guard 后 label materialization 失败。"""
            raise BroadQaExternalDataError("synthetic reserve drift")
    else:
        def materializer(**kwargs):
            """返回无需真实 label 的最小 synthetic material。"""
            return ({"label_materialization_count": 1}, ({} ,))
    monkeypatch.setattr(
        runner_module,
        "materialize_normalization_recovery_v4_reserve_after_guard",
        materializer)
    monkeypatch.setattr(
        runner_module, "evaluate_normalization_recovery_v4_candidate",
        lambda **kwargs: {
            "evaluation_report_sha256": _sha("report"),
            "overall_outcome": "PASS",
        })


def _runner_arguments(tmp_path: Path) -> dict[str, object]:
    """构造唯一 runner 的完整参数与物理目录。"""
    values = _family_arguments(tmp_path)
    family = tmp_path / "family"
    source = tmp_path / "firefox-source"
    family.mkdir()
    source.mkdir()
    values.update({
        "run_root": tmp_path,
        "family_freeze_dir": family,
        "publication_dir": tmp_path / "publication",
        "firefox_source_pack_dir": source,
    })
    return values


def test_formal_runner_consumes_guard_once_and_keeps_disabled(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """唯一 runner 必须 guard-first、报告禁用态并拒绝第二次运行。"""
    _install_runner(monkeypatch)
    arguments = _runner_arguments(tmp_path)
    report = run_normalization_recovery_v4_formal_evaluation(**arguments)
    publication = Path(arguments["publication_dir"])
    assert report["overall_outcome"] == "PASS"
    assert report["reserve_payload_read_count"] == 1
    assert report["production_enabled"] == 0
    assert (publication / "run-000001.guard.json").is_file()
    assert (publication / "run-000001.report.json").is_file()
    with pytest.raises(BroadQaExternalDataError, match="已消费"):
        run_normalization_recovery_v4_formal_evaluation(**arguments)


def test_formal_runner_seals_guard_after_materialization_failure(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """guard 后 label 失败必须封存 NE，且不得恢复后重跑。"""
    _install_runner(monkeypatch, fail_materialization=True)
    arguments = _runner_arguments(tmp_path)
    with pytest.raises(BroadQaExternalDataError, match="reserve drift"):
        run_normalization_recovery_v4_formal_evaluation(**arguments)
    publication = Path(arguments["publication_dir"])
    assert (publication / "run-000001.guard.json").is_file()
    assert (publication / "run-000001.failure.json").is_file()
    with pytest.raises(BroadQaExternalDataError, match="已消费"):
        run_normalization_recovery_v4_formal_evaluation(**arguments)


def test_reserve_materialization_requires_consumed_guard() -> None:
    """任何 guard 前 reserve label 物化尝试必须 fail closed。"""
    from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v4_reserve_materialization import (
        materialize_normalization_recovery_v4_reserve_after_guard,
    )
    with pytest.raises(BroadQaExternalDataError, match="guard 后"):
        materialize_normalization_recovery_v4_reserve_after_guard(
            guard_consumed=0,
            prior_evaluation_protocol_dir="unused",
            expected_prior_evaluation_manifest_sha256=_sha("unused"),
            firefox_source_pack_dir="unused",
            evaluation_commitment_dir="unused",
            expected_evaluation_commitment_manifest_sha256=_sha("unused-2"),
        )
