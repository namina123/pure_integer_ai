"""Recovery-v9 materializer、aggregate evaluator与one-shot runner测试。"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_recovery_v9_evaluation_family as family_module,
)
from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_recovery_v9_evaluation_runner as runner,
)
from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_recovery_v9_evaluator as evaluator,
)
from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_recovery_v9_label_materialization as materializer,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v9_evaluator import (
    evaluate_normalization_recovery_v9_candidate,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v9_evaluation_runner import (
    run_normalization_recovery_v9_formal_evaluation,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v9_label_materialization import (
    NORMALIZATION_RECOVERY_V9_EVALUATION_RECORD_KIND,
    materialize_normalization_recovery_v9_labels_after_guard,
)


def _sha(label: str) -> str:
    """返回测试identity SHA。"""
    return hashlib.sha256(label.encode()).hexdigest()


def test_v9_materializer_requires_guard() -> None:
    """任何source payload读取前必须已消费唯一guard。"""
    with pytest.raises(BroadQaExternalDataError, match="guard后"):
        materialize_normalization_recovery_v9_labels_after_guard(
            guard_consumed=0,
            gimp_source_pack_dir="unused",
            expected_gimp_source_manifest_sha256=_sha("source"),
            evaluation_commitment_dir="unused",
            expected_evaluation_commitment_manifest_sha256=_sha("commitment"),
            runtime_gate_dir="unused",
        )


def test_v9_materializer_rederives_full_denominator(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """guard后records必须覆盖identity、changed、eligible与冲突分账。"""
    source_sha = _sha("source")
    commitment_sha = _sha("commitment")
    pairs = (
        {
            "contains_han_both": 1,
            "equal_length": 1,
            "identity_preservation": 0,
            "official_source_text": "File",
            "pair_id": _sha("changed"),
            "single_han_difference": 1,
            "source_identity": {
                "domain": "po", "msgctxt": "", "msgid": "File",
                "msgid_plural": ""},
            "source_identity_sha256": _sha("changed source"),
            "structure_equal": 1,
            "v9_evaluation_eligible": 1,
            "within_scalar_limit": 1,
            "zh_hans": {"msgstr": "档", "source_file_id": _sha("hans 1")},
            "zh_hans_structure_tokens": [],
            "zh_hant": {"msgstr": "檔", "source_file_id": _sha("hant 1")},
            "zh_hant_structure_tokens": [],
        },
        {
            "contains_han_both": 1,
            "equal_length": 1,
            "identity_preservation": 1,
            "official_source_text": "Version",
            "pair_id": _sha("identity"),
            "single_han_difference": 0,
            "source_identity": {
                "domain": "po", "msgctxt": "", "msgid": "Version",
                "msgid_plural": ""},
            "source_identity_sha256": _sha("identity source"),
            "structure_equal": 1,
            "v9_evaluation_eligible": 1,
            "within_scalar_limit": 1,
            "zh_hans": {"msgstr": "版本", "source_file_id": _sha("hans 2")},
            "zh_hans_structure_tokens": [],
            "zh_hant": {"msgstr": "版本", "source_file_id": _sha("hant 2")},
            "zh_hant_structure_tokens": [],
        },
    )
    buckets = {
        "contains_han_both_count": 2,
        "equal_length_count": 2,
        "evaluation_eligible_count": 2,
        "identity_count": 1,
        "input_conflict_count": 0,
        "nonidentity_count": 1,
        "single_han_difference_count": 1,
        "structure_equal_count": 2,
        "structure_unequal_count": 0,
        "variable_length_count": 0,
    }
    commitment = {
        "denominator": {
            "aggregate_buckets": buckets,
            "identity_artifact": {
                "record_count": 2, "sha256": _sha("identity roster")},
            "record_count": 2,
        },
        "manifest_sha256": commitment_sha,
    }
    monkeypatch.setattr(
        materializer, "read_normalization_recovery_v9_evaluation_commitment",
        lambda *args, **kwargs: commitment)
    monkeypatch.setattr(
        materializer,
        "materialize_normalization_recovery_v9_source_pairs_after_guard",
        lambda *args, **kwargs: (
            {"manifest_sha256": source_sha}, pairs,
            {"plain_pair_count": 2}))
    result, records = materialize_normalization_recovery_v9_labels_after_guard(
        guard_consumed=1,
        gimp_source_pack_dir="source",
        expected_gimp_source_manifest_sha256=source_sha,
        evaluation_commitment_dir="commitment",
        expected_evaluation_commitment_manifest_sha256=commitment_sha,
        runtime_gate_dir="gate",
    )
    assert result["label_materialization_count"] == 2
    assert result["gimp_archive_parse_count"] == 1
    assert [item["input_text"] for item in records] == ["檔", "版本"]
    assert [item["expected_output"] for item in records] == ["档", "版本"]


def _record(
        identity: str, input_text: str, expected: str, *,
        source: str, identity_pair: int, single: int,
        tokens: list[str] | None = None,
        ) -> dict[str, object]:
    """构造完整v9 evaluator record。"""
    return {
        "contains_han_both": 1,
        "equal_length": 1,
        "evaluation_eligible": 1,
        "evaluation_id": identity,
        "expected_output": expected,
        "identity_preservation": identity_pair,
        "input_text": input_text,
        "official_source_text": source,
        "record_kind": NORMALIZATION_RECOVERY_V9_EVALUATION_RECORD_KIND,
        "single_han_difference": single,
        "source_identity": {"msgid": source},
        "source_pack_manifest_sha256": _sha("source"),
        "structure_equal": 1,
        "structure_tokens": [] if tokens is None else tokens,
        "variable_length": 0,
        "within_scalar_limit": 1,
    }


def _result(*, route: str, output: str,
            behavior: str = "EXACT") -> dict[str, object]:
    """构造candidate aggregate测试结果。"""
    return {
        "behavior": behavior,
        "output_text": output,
        "partial_commit_count": 0,
        "production_enabled": 0,
        "result_sha256": _sha(route + output + behavior),
        "route_kind": route,
        "structure_mismatch_count": 0,
    }


def _evaluation_inputs(records: tuple[dict[str, object], ...]):
    """形成与records逐项一致的commitment/materialization/candidate。"""
    outputs_by_input: dict[str, set[str]] = {}
    for item in records:
        outputs_by_input.setdefault(str(item["input_text"]), set()).add(
            str(item["expected_output"]))
    buckets = {
        "contains_han_both_count": sum(item["contains_han_both"]
                                       for item in records),
        "equal_length_count": len(records),
        "evaluation_eligible_count": len(records),
        "identity_count": sum(item["identity_preservation"]
                              for item in records),
        "input_conflict_count": sum(len(value) > 1
                                    for value in outputs_by_input.values()),
        "nonidentity_count": sum(1 - item["identity_preservation"]
                                 for item in records),
        "single_han_difference_count": sum(item["single_han_difference"]
                                           for item in records),
        "structure_equal_count": len(records),
        "structure_unequal_count": 0,
        "variable_length_count": 0,
    }
    from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v9_evaluation_commitment import (
        NORMALIZATION_RECOVERY_V9_DIMENSIONS,
        NORMALIZATION_RECOVERY_V9_DIMENSION_ORDER,
    )
    commitment = {
        "denominator": {
            "aggregate_buckets": buckets, "record_count": len(records)},
        "dimension_order": list(NORMALIZATION_RECOVERY_V9_DIMENSION_ORDER),
        "dimensions": NORMALIZATION_RECOVERY_V9_DIMENSIONS,
        "manifest_sha256": _sha("commitment"),
    }
    materialization = {
        "evaluation_commitment_manifest_sha256": commitment["manifest_sha256"],
        "evaluation_record_roster_sha256": evaluator._sha256(records),
        "gimp_source_manifest_sha256": _sha("source"),
        "gimp_source_payload_read_count": 1,
        "label_materialization_count": len(records),
    }
    candidate = {
        "candidate_program_sha256": _sha("candidate"),
        "evaluation_commitment_manifest_sha256": commitment["manifest_sha256"],
    }
    candidate_manifest = {
        "candidate_program_sha256": candidate["candidate_program_sha256"],
        "manifest_sha256": _sha("candidate manifest"),
    }
    return commitment, materialization, candidate, candidate_manifest


def test_v9_evaluator_passes_all_dimensions(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """非零两类changed、identity与结构生成应形成六维PASS。"""
    records = (
        _record(_sha("orth"), "檔", "档", source="File",
                identity_pair=0, single=1),
        _record(_sha("lexical"), "選項 %1", "选项 %1", source="Options",
                identity_pair=0, single=0, tokens=["%1"]),
        _record(_sha("identity"), "版本", "版本", source="Version",
                identity_pair=1, single=0),
    )
    outputs = (
        _result(route="ORTHOGRAPHIC_ATOM", output="档"),
        _result(route="SOURCE_CONDITIONED_LEXICAL_ATOM", output="选项 %1"),
        _result(route="IDENTITY_VETO", output="版本"),
    )
    monkeypatch.setattr(
        evaluator, "execute_normalization_recovery_v8_candidate_batch",
        lambda *args, **kwargs: outputs)
    monkeypatch.setattr(
        evaluator, "reference_normalization_recovery_v8_candidate_batch",
        lambda *args, **kwargs: outputs)
    commitment, materialization, candidate, candidate_manifest = (
        _evaluation_inputs(records))
    report = evaluate_normalization_recovery_v9_candidate(
        commitment=commitment, candidate_manifest=candidate_manifest,
        candidate=candidate, materialization=materialization,
        evaluation_records=records,
        family_freeze_manifest_sha256=_sha("family"))
    assert report["overall_outcome"] == "PASS"
    assert report["judgement_counts"] == {
        "EXACT": 3, "UNKNOWN": 0, "WRONG": 0}
    assert [item["outcome"] for item in report["dimensions"]] == ["PASS"] * 6


def test_v9_evaluator_wrong_commit_dominates_ne(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """任何错误提交必须FAIL，不能被其他缺证维度掩盖。"""
    records = (
        _record(_sha("wrong"), "檔", "档", source="File",
                identity_pair=0, single=1),
        _record(_sha("identity"), "版本", "版本", source="Version",
                identity_pair=1, single=0),
    )
    outputs = (
        _result(route="ORTHOGRAPHIC_ATOM", output="文件"),
        _result(route="UNKNOWN", output="", behavior="UNKNOWN"),
    )
    monkeypatch.setattr(
        evaluator, "execute_normalization_recovery_v8_candidate_batch",
        lambda *args, **kwargs: outputs)
    monkeypatch.setattr(
        evaluator, "reference_normalization_recovery_v8_candidate_batch",
        lambda *args, **kwargs: outputs)
    commitment, materialization, candidate, candidate_manifest = (
        _evaluation_inputs(records))
    report = evaluate_normalization_recovery_v9_candidate(
        commitment=commitment, candidate_manifest=candidate_manifest,
        candidate=candidate, materialization=materialization,
        evaluation_records=records,
        family_freeze_manifest_sha256=_sha("family"))
    assert report["overall_outcome"] == "FAIL"
    assert report["judgement_counts"]["WRONG"] == 1


def test_v9_family_freezes_candidate_code_and_zero_label_reads(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """family必须绑定live identity且只读source/gate manifest。"""
    commitment = {
        "denominator": {"record_count": 9_264},
        "dimension_order": ["D"],
        "dimensions": {"D": {}},
        "formal_contract": {"formal_run_count_max": 1},
        "manifest_sha256": _sha("commitment"),
        "runtime_gate_sha256": _sha("gate"),
    }
    candidate = {
        "candidate_program_sha256": _sha("candidate"),
        "evaluation_commitment_manifest_sha256": commitment[
            "manifest_sha256"],
    }
    candidate_manifest = {
        "candidate_program_sha256": candidate["candidate_program_sha256"],
        "manifest_sha256": _sha("candidate manifest"),
        "preflight_failure_count": 0,
    }
    source = {"manifest_sha256": _sha("source")}
    commitment["denominator"]["source_pack_manifest_sha256"] = source[
        "manifest_sha256"]
    gate = {"manifest_sha256": commitment["runtime_gate_sha256"]}
    monkeypatch.setattr(
        family_module, "read_normalization_recovery_v9_candidate_pack",
        lambda *args, **kwargs: (
            candidate_manifest, candidate, {"failure_count": 0}))
    monkeypatch.setattr(
        family_module, "read_normalization_recovery_v9_evaluation_commitment",
        lambda *args, **kwargs: commitment)
    monkeypatch.setattr(
        family_module, "_source_manifest_only", lambda *args, **kwargs: source)
    monkeypatch.setattr(
        family_module, "_runtime_gate_only", lambda *args, **kwargs: gate)
    monkeypatch.setattr(
        family_module, "_code_identity",
        lambda value: ([{"relative_path": "code.py", "sha256": _sha("code")}],
                       _sha("code identity"), _sha("head")[:40]))
    freeze, *_ = family_module.build_normalization_recovery_v9_evaluation_family_freeze(
        repository_root=tmp_path,
        base_candidate_dir=tmp_path,
        evaluation_commitment_dir=tmp_path,
        candidate_dir=tmp_path,
        gimp_source_pack_dir=tmp_path,
        runtime_gate_dir=tmp_path,
        expected_evaluation_commitment_manifest_sha256=commitment[
            "manifest_sha256"],
        expected_candidate_manifest_sha256=candidate_manifest[
            "manifest_sha256"],
        expected_gimp_source_manifest_sha256=source["manifest_sha256"],
        expected_runtime_gate_sha256=gate["manifest_sha256"],
    )
    assert freeze["individual_label_read_count"] == 0
    assert freeze["gimp_source_non_manifest_read_count"] == 0
    assert freeze["candidate_program_sha256"] == (
        candidate["candidate_program_sha256"])


def _install_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    """安装不读取真实label的runner依赖。"""
    family = {
        "family_commitment_sha256": _sha("family commitment"),
        "manifest_sha256": _sha("family"),
    }
    candidate_manifest = {"manifest_sha256": _sha("candidate manifest")}
    candidate = {"candidate_program_sha256": _sha("candidate")}
    commitment = {"manifest_sha256": _sha("commitment")}
    monkeypatch.setattr(
        runner, "require_normalization_recovery_v9_k_root",
        lambda value: Path(value).resolve())
    monkeypatch.setattr(
        runner, "read_normalization_recovery_v9_evaluation_family_freeze",
        lambda *args, **kwargs: (
            family, candidate_manifest, candidate, commitment))
    monkeypatch.setattr(
        runner, "materialize_normalization_recovery_v9_labels_after_guard",
        lambda **kwargs: ({"label_materialization_count": 1}, ({},)))
    monkeypatch.setattr(
        runner, "evaluate_normalization_recovery_v9_candidate",
        lambda **kwargs: {
            "evaluation_report_sha256": _sha("report"),
            "overall_outcome": "PASS"})


def test_v9_runner_consumes_unique_guard_and_rejects_retry(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """runner必须先写guard并拒绝任何第二次运行。"""
    _install_runner(monkeypatch)
    family = tmp_path / "family"
    family.mkdir()
    values = {
        "run_root": tmp_path,
        "repository_root": tmp_path,
        "family_freeze_dir": family,
        "publication_dir": tmp_path / "publication",
    }
    for name in runner.NORMALIZATION_RECOVERY_V9_EVALUATION_DATA_ARGUMENTS:
        path = tmp_path / name
        path.mkdir()
        values[name] = path
    for name in (
            "expected_evaluation_commitment_manifest_sha256",
            "expected_candidate_manifest_sha256",
            "expected_gimp_source_manifest_sha256",
            "expected_runtime_gate_sha256"):
        values[name] = _sha(name)
    report = run_normalization_recovery_v9_formal_evaluation(**values)
    publication = Path(values["publication_dir"])
    assert report["overall_outcome"] == "PASS"
    assert (publication / "run-000001.guard.json").is_file()
    assert (publication / "run-000001.report.json").is_file()
    with pytest.raises(BroadQaExternalDataError, match="已消费"):
        run_normalization_recovery_v9_formal_evaluation(**values)
