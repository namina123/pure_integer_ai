"""Recovery-v8 materializer、aggregate evaluator 与 one-shot runner 测试。"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_recovery_v8_evaluation_runner as runner_module,
)
from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_recovery_v8_evaluator as evaluator_module,
)
from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_recovery_v8_label_materialization as materializer_module,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_evaluation_runner import (
    run_normalization_recovery_v8_formal_evaluation,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_evaluator import (
    evaluate_normalization_recovery_v8_candidate,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_label_materialization import (
    NORMALIZATION_RECOVERY_V8_EVALUATION_RECORD_KIND,
    materialize_normalization_recovery_v8_labels_after_guard,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _commitment(count: int) -> dict[str, object]:
    from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_evaluation_commitment import (
        NORMALIZATION_RECOVERY_V8_DIMENSIONS,
    )
    return {
        "denominator": {
            "aggregate_buckets": {
                "equal_length_count": count,
                "identity_count": 1,
                "nonidentity_count": count - 1,
                "single_han_difference_count": 1,
                "structure_equal_count": count,
                "variable_length_count": 0,
            },
            "identity_artifact": {"sha256": _sha("inventory")},
            "record_count": count,
        },
        "dimensions": NORMALIZATION_RECOVERY_V8_DIMENSIONS,
        "manifest_sha256": _sha("commitment"),
    }


def _record(
        identity: str, input_text: str, expected: str, *,
        source: str, identity_pair: int, single: int,
        tokens: list[str] | None = None,
        ) -> dict[str, object]:
    return {
        "equal_length": 1,
        "evaluation_id": identity,
        "expected_output": expected,
        "identity_preservation": identity_pair,
        "input_text": input_text,
        "official_source_text": source,
        "record_kind": NORMALIZATION_RECOVERY_V8_EVALUATION_RECORD_KIND,
        "single_han_difference": single,
        "source_pack_manifest_sha256": _sha("source"),
        "structure_equal": 1,
        "structure_tokens": [] if tokens is None else tokens,
        "variable_length": 0,
    }


def _result(
        *, route: str, output: str, behavior: str = "EXACT",
        ) -> dict[str, object]:
    return {
        "behavior": behavior,
        "output_text": output,
        "partial_commit_count": 0,
        "production_enabled": 0,
        "result_sha256": _sha(route + output + behavior),
        "route_kind": route,
        "structure_mismatch_count": 0,
    }


def test_materializer_requires_guard() -> None:
    with pytest.raises(BroadQaExternalDataError, match="guard 后"):
        materialize_normalization_recovery_v8_labels_after_guard(
            guard_consumed=0, vlc_source_pack_dir="unused",
            expected_vlc_source_manifest_sha256=_sha("source"),
            v7_commitment_dir="unused",
            expected_v7_commitment_manifest_sha256=_sha("v7"),
            evaluation_commitment_dir="unused",
            expected_evaluation_commitment_manifest_sha256=_sha("v8"))


def test_materializer_rederives_complete_identity_after_guard(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    source_sha = _sha("source")
    commitment = _commitment(2)
    commitment["source_exclusion"] = {
        "excluded_source_pack_manifest_sha256": source_sha}
    pairs = (
        {
            "equal_length": 1,
            "identity_preservation": 0,
            "pair_id": _sha("changed pair"),
            "single_han_difference": 1,
            "source_identity": {
                "msgctxt": "", "msgid": "File", "msgid_plural": ""},
            "source_identity_sha256": _sha("changed source"),
            "structure_equal": 1,
            "within_scalar_limit": 1,
            "zh_hans": {"msgstr": "档", "source_file_id": _sha("hans 1")},
            "zh_hant": {"msgstr": "檔", "source_file_id": _sha("hant 1")},
            "zh_hant_structure_tokens": [],
        },
        {
            "equal_length": 1,
            "identity_preservation": 1,
            "pair_id": _sha("identity pair"),
            "single_han_difference": 0,
            "source_identity": {
                "msgctxt": "", "msgid": "Version", "msgid_plural": ""},
            "source_identity_sha256": _sha("identity source"),
            "structure_equal": 1,
            "within_scalar_limit": 1,
            "zh_hans": {"msgstr": "版本", "source_file_id": _sha("hans 2")},
            "zh_hant": {"msgstr": "版本", "source_file_id": _sha("hant 2")},
            "zh_hant_structure_tokens": [],
        },
    )
    inventory = tuple(materializer_module._identity(item) for item in pairs)
    summary = {
        "equal_length_pair_count": 2,
        "identity_pair_count": 1,
        "nonidentity_pair_count": 1,
        "single_han_difference_count": 1,
        "structure_equal_count": 2,
        "variable_length_pair_count": 0,
    }
    source = tmp_path / "source"
    source.mkdir()
    (source / materializer_module.VLC_ARCHIVE_NAME).write_bytes(b"archive")
    monkeypatch.setattr(
        materializer_module,
        "read_normalization_recovery_v8_evaluation_commitment",
        lambda *args, **kwargs: commitment)
    monkeypatch.setattr(
        materializer_module, "read_normalization_recovery_v7_vlc_source_pack",
        lambda *args, **kwargs: (
            {"manifest_sha256": source_sha}, (), inventory))
    monkeypatch.setattr(
        materializer_module, "parse_normalization_recovery_v7_vlc_archive",
        lambda *args, **kwargs: ((), pairs, summary))
    materialization, records = (
        materialize_normalization_recovery_v8_labels_after_guard(
            guard_consumed=1, vlc_source_pack_dir=source,
            expected_vlc_source_manifest_sha256=source_sha,
            v7_commitment_dir=tmp_path / "v7",
            expected_v7_commitment_manifest_sha256=_sha("v7"),
            evaluation_commitment_dir=tmp_path / "v8",
            expected_evaluation_commitment_manifest_sha256=commitment[
                "manifest_sha256"]))
    assert materialization["label_materialization_count"] == 2
    assert materialization["vlc_archive_parse_count"] == 2
    assert [item["official_source_text"] for item in records] == [
        "File", "Version"]
    assert [item["input_text"] for item in records] == ["檔", "版本"]


def test_evaluator_passes_six_dimensions_on_nonzero_routes(
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    records = (
        _record(_sha("orth"), "檔", "档", source="File", identity_pair=0,
                single=1),
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
        evaluator_module, "execute_normalization_recovery_v8_candidate_batch",
        lambda *args, **kwargs: outputs)
    monkeypatch.setattr(
        evaluator_module, "reference_normalization_recovery_v8_candidate_batch",
        lambda *args, **kwargs: outputs)
    commitment = _commitment(len(records))
    materialization = {
        "evaluation_commitment_manifest_sha256": commitment["manifest_sha256"],
        "evaluation_record_roster_sha256": evaluator_module._sha256(records),
        "label_materialization_count": len(records),
        "vlc_source_manifest_sha256": _sha("source"),
        "vlc_source_payload_read_count": 1,
    }
    candidate = {
        "candidate_program_sha256": _sha("candidate"),
        "evaluation_commitment_manifest_sha256": commitment["manifest_sha256"],
    }
    report = evaluate_normalization_recovery_v8_candidate(
        commitment=commitment,
        candidate_manifest={
            "candidate_program_sha256": candidate["candidate_program_sha256"],
            "manifest_sha256": _sha("candidate manifest")},
        candidate=candidate, materialization=materialization,
        evaluation_records=records,
        family_freeze_manifest_sha256=_sha("family"))
    assert report["overall_outcome"] == "PASS"
    assert [item["outcome"] for item in report["dimensions"]] == ["PASS"] * 6


def test_evaluator_wrong_commit_dominates_missing_dimensions(
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    records = (
        _record(_sha("wrong"), "檔", "档", source="File", identity_pair=0,
                single=1),
        _record(_sha("identity only"), "版本", "版本", source="Version",
                identity_pair=1, single=0),
    )
    outputs = (
        _result(route="ORTHOGRAPHIC_ATOM", output="文件"),
        _result(route="UNKNOWN", output="", behavior="UNKNOWN"),
    )
    monkeypatch.setattr(
        evaluator_module, "execute_normalization_recovery_v8_candidate_batch",
        lambda *args, **kwargs: outputs)
    monkeypatch.setattr(
        evaluator_module, "reference_normalization_recovery_v8_candidate_batch",
        lambda *args, **kwargs: outputs)
    commitment = _commitment(len(records))
    materialization = {
        "evaluation_commitment_manifest_sha256": commitment["manifest_sha256"],
        "evaluation_record_roster_sha256": evaluator_module._sha256(records),
        "label_materialization_count": len(records),
        "vlc_source_manifest_sha256": _sha("source"),
        "vlc_source_payload_read_count": 1,
    }
    candidate = {
        "candidate_program_sha256": _sha("candidate"),
        "evaluation_commitment_manifest_sha256": commitment["manifest_sha256"],
    }
    report = evaluate_normalization_recovery_v8_candidate(
        commitment=commitment,
        candidate_manifest={
            "candidate_program_sha256": candidate["candidate_program_sha256"],
            "manifest_sha256": _sha("candidate manifest")},
        candidate=candidate, materialization=materialization,
        evaluation_records=records,
        family_freeze_manifest_sha256=_sha("family"))
    assert report["overall_outcome"] == "FAIL"
    assert report["dimensions"][0]["outcome"] == "FAIL"
    assert report["dimensions"][-1]["metrics"]["wrong_count"] == 1


def _install_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    family = {
        "family_commitment_sha256": _sha("family commitment"),
        "manifest_sha256": _sha("family"),
    }
    candidate_manifest = {
        "manifest_sha256": _sha("candidate manifest")}
    candidate = {"candidate_program_sha256": _sha("candidate")}
    commitment = {"manifest_sha256": _sha("commitment")}
    monkeypatch.setattr(
        runner_module, "require_normalization_recovery_v8_k_root",
        lambda value: Path(value).resolve())
    monkeypatch.setattr(
        runner_module, "read_normalization_recovery_v8_evaluation_family_freeze",
        lambda *args, **kwargs: (
            family, candidate_manifest, candidate, commitment))
    monkeypatch.setattr(
        runner_module, "materialize_normalization_recovery_v8_labels_after_guard",
        lambda **kwargs: ({"label_materialization_count": 1}, ({},)))
    monkeypatch.setattr(
        runner_module, "evaluate_normalization_recovery_v8_candidate",
        lambda **kwargs: {
            "evaluation_report_sha256": _sha("report"),
            "overall_outcome": "PASS"})


def test_runner_consumes_unique_guard_and_rejects_retry(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    _install_runner(monkeypatch)
    family = tmp_path / "family"
    family.mkdir()
    values = {
        "run_root": tmp_path,
        "repository_root": tmp_path,
        "family_freeze_dir": family,
        "publication_dir": tmp_path / "publication",
    }
    for name in runner_module.NORMALIZATION_RECOVERY_V8_EVALUATION_DATA_ARGUMENTS:
        path = tmp_path / name
        path.mkdir()
        values[name] = path
    for name in (
            "expected_protocol_manifest_sha256",
            "expected_pack_manifest_sha256",
            "expected_audit_manifest_sha256",
            "expected_v7_commitment_manifest_sha256",
            "expected_evaluation_commitment_manifest_sha256",
            "expected_candidate_manifest_sha256",
            "expected_vlc_source_manifest_sha256"):
        values[name] = _sha(name)
    report = run_normalization_recovery_v8_formal_evaluation(**values)
    publication = Path(values["publication_dir"])
    assert report["overall_outcome"] == "PASS"
    assert (publication / "run-000001.guard.json").is_file()
    assert (publication / "run-000001.report.json").is_file()
    with pytest.raises(BroadQaExternalDataError, match="已消费"):
        run_normalization_recovery_v8_formal_evaluation(**values)
