"""Recovery-v6 family、materializer、八维 evaluator 与 runner 测试。"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_recovery_v6_evaluation_family as family_module,
)
from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_recovery_v6_evaluation_runner as runner_module,
)
from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_recovery_v6_label_materialization as materializer_module,
)
from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_evaluation_commitment import (
    NORMALIZATION_RECOVERY_V5_DIMENSIONS,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v6_candidate import (
    derive_normalization_recovery_v6_candidate_preflight,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v6_evaluation_family import (
    NORMALIZATION_RECOVERY_V6_CANDIDATE_V2_MANIFEST_SHA256,
    NORMALIZATION_RECOVERY_V6_EVALUATION_CODE_FILES,
    build_normalization_recovery_v6_evaluation_family_freeze,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v6_evaluation_runner import (
    run_normalization_recovery_v6_formal_evaluation,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v6_evaluator import (
    NORMALIZATION_RECOVERY_V6_DIMENSION_ORDER,
    evaluate_normalization_recovery_v6_candidate,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v6_label_materialization import (
    NORMALIZATION_RECOVERY_V6_EVALUATION_RECORD_KIND,
    materialize_normalization_recovery_v6_labels_after_guard,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from test_ph2_broad_qa_normalization_recovery_v6_candidate import _material


def _sha(value: str) -> str:
    """返回 synthetic SHA-256 identity。"""
    return hashlib.sha256(value.encode()).hexdigest()


def _commitment(
        *,
        manifest_sha: str,
        source_sha: str,
        identity_count: int,
        nonidentity_count: int,
        equal_count: int,
        variable_count: int,
        single_han_count: int,
        ) -> dict[str, object]:
    """构造保持八维合同不变的小分母 commitment。"""
    return {
        "denominator": {
            "aggregate_buckets": {
                "equal_length_count": equal_count,
                "identity_count": identity_count,
                "nonidentity_count": nonidentity_count,
                "single_han_difference_count": single_han_count,
                "variable_length_count": variable_count,
            },
            "identity_artifact": {
                "bytes": 1,
                "record_count": identity_count + nonidentity_count,
                "relative_path": "evaluation-inventory.identity.jsonl",
                "role": "QT_HELD_OUT_IDENTITY_WITHOUT_LABELS",
                "sha256": _sha("inventory"),
            },
            "label_blind": 1,
            "record_count": identity_count + nonidentity_count,
        },
        "dimensions": NORMALIZATION_RECOVERY_V5_DIMENSIONS,
        "formal_contract": {
            "candidate_applicability_cannot_shrink_denominator": 1,
            "formal_run_count_max": 1,
        },
        "manifest_sha256": manifest_sha,
        "source_exclusion": {
            "excluded_source_pack_manifest_sha256": source_sha,
        },
    }


def _family_arguments(tmp_path: Path) -> dict[str, object]:
    """构造完整 family 参数与隔离目录。"""
    values: dict[str, object] = {
        "repository_root": Path(__file__).resolve().parents[1],
        "expected_candidate_manifest_sha256": (
            NORMALIZATION_RECOVERY_V6_CANDIDATE_V2_MANIFEST_SHA256),
        "expected_protocol_manifest_sha256": _sha("protocol"),
        "expected_predecessor_pack_manifest_sha256": _sha("predecessor"),
        "expected_pack_manifest_sha256": _sha("pack"),
        "expected_audit_manifest_sha256": _sha("audit"),
        "expected_qt_source_manifest_sha256": _sha("qt source"),
        "expected_evaluation_commitment_manifest_sha256": _sha("commitment"),
    }
    for name in (
            "candidate_dir", "protocol_dir", "predecessor_pack_dir",
            "pack_dir", "audit_dir", "qt_source_pack_dir",
            "evaluation_commitment_dir"):
        path = tmp_path / name
        path.mkdir(parents=True)
        values[name] = path
    return values


def test_family_accepts_only_v2_and_freezes_zero_label_reads(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """family 必须拒绝 v1，并冻结 clone/code/commitment 而不读 labels。"""
    arguments = _family_arguments(tmp_path)
    _materials, _pack, _outputs, candidate, _audit, _old_commitment = _material()
    arguments["expected_evaluation_commitment_manifest_sha256"] = candidate[
        "evaluation_commitment_manifest_sha256"]
    preflight = derive_normalization_recovery_v6_candidate_preflight(candidate)
    manifest = {
        "candidate_program_sha256": candidate["candidate_program_sha256"],
        "manifest_sha256": NORMALIZATION_RECOVERY_V6_CANDIDATE_V2_MANIFEST_SHA256,
        "status": "FROZEN_LABEL_BLIND_PREFLIGHT_PASS_FORMAL_NOT_RUN",
    }
    commitment = _commitment(
        manifest_sha=str(arguments[
            "expected_evaluation_commitment_manifest_sha256"]),
        source_sha=str(arguments["expected_qt_source_manifest_sha256"]),
        identity_count=1, nonidentity_count=1,
        equal_count=1, variable_count=1, single_han_count=1)
    monkeypatch.setattr(
        family_module, "read_normalization_recovery_v6_candidate_pack",
        lambda *args, **kwargs: (manifest, candidate, preflight))
    monkeypatch.setattr(
        family_module, "read_normalization_recovery_v5_evaluation_commitment",
        lambda *args, **kwargs: commitment)
    freeze, _manifest, clone, _commitment_value = (
        build_normalization_recovery_v6_evaluation_family_freeze(**arguments))
    assert freeze["evaluation_or_reserve_payload_read_count"] == 0
    assert freeze["qt_source_non_manifest_read_count"] == 0
    assert freeze["evaluation_run_count"] == 0
    assert clone["candidate_program_sha256"] == candidate[
        "candidate_program_sha256"]
    assert len(freeze["code_files"]) == len(
        NORMALIZATION_RECOVERY_V6_EVALUATION_CODE_FILES)

    rejected = dict(arguments)
    rejected["expected_candidate_manifest_sha256"] = _sha("candidate v1")
    with pytest.raises(BroadQaExternalDataError, match="只接受正式 candidate v2"):
        build_normalization_recovery_v6_evaluation_family_freeze(**rejected)


def _pair(
        *,
        pair_id: str,
        input_text: str,
        expected_output: str,
        identity: int,
        single_han: int,
        ) -> dict[str, object]:
    """构造一条 synthetic Qt pair。"""
    equal = int(len(input_text) == len(expected_output))
    source_identity = {
        "comment": "",
        "context": "Synthetic",
        "message_id": pair_id,
        "module": "qtbase",
        "source": "source",
    }
    return {
        "contains_han_both": 1,
        "equal_length": equal,
        "format_version": 1,
        "identity_preservation": identity,
        "pair_id": pair_id,
        "record_kind": "QT_TRANSLATIONS_TS_PAIR_V1",
        "single_han_difference": single_han,
        "source_identity": source_identity,
        "source_identity_sha256": _sha(pair_id + " source"),
        "structure_equal": 1,
        "training_eligible": 1,
        "within_scalar_limit": 1,
        "zh_hans": {
            "source_file_id": _sha(pair_id + " hans"),
            "translation": expected_output,
        },
        "zh_hans_structure_tokens": [],
        "zh_hant": {
            "source_file_id": _sha(pair_id + " hant"),
            "translation": input_text,
        },
        "zh_hant_structure_tokens": [],
    }


def _identity(pair: dict[str, object]) -> dict[str, object]:
    """形成与 source pack 相同的无 label identity。"""
    return {
        "format_version": 1,
        "pair_id": pair["pair_id"],
        "record_kind": "QT_TRANSLATIONS_HELD_OUT_IDENTITY_V1",
        "source_identity": pair["source_identity"],
        "source_identity_sha256": pair["source_identity_sha256"],
        "zh_hans_source_file_id": pair["zh_hans"]["source_file_id"],
        "zh_hant_source_file_id": pair["zh_hant"]["source_file_id"],
    }


def test_materializer_requires_guard_and_checks_all_identity(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """materializer 只能 guard 后运行，并逐项核对完整 identity roster。"""
    with pytest.raises(BroadQaExternalDataError, match="guard 后"):
        materialize_normalization_recovery_v6_labels_after_guard(
            guard_consumed=0,
            qt_source_pack_dir="unused",
            expected_qt_source_manifest_sha256=_sha("source"),
            evaluation_commitment_dir="unused",
            expected_evaluation_commitment_manifest_sha256=_sha("commitment"),
        )
    source = tmp_path / "qt-source"
    source.mkdir()
    (source / materializer_module.QT_ARCHIVE_NAME).write_bytes(b"archive")
    source_sha = _sha("source")
    commitment_sha = _sha("commitment")
    pairs = (
        _pair(pair_id=_sha("identity pair"), input_text="文件",
              expected_output="文件", identity=1, single_han=0),
        _pair(pair_id=_sha("mapping pair"), input_text="檔",
              expected_output="档", identity=0, single_han=1),
    )
    inventory = tuple(_identity(item) for item in pairs)
    summary = {
        "equal_length_pair_count": 2,
        "identity_pair_count": 1,
        "nonidentity_pair_count": 1,
        "plain_pair_count": 2,
        "single_han_difference_count": 1,
        "variable_length_pair_count": 0,
    }
    commitment = _commitment(
        manifest_sha=commitment_sha, source_sha=source_sha,
        identity_count=1, nonidentity_count=1,
        equal_count=2, variable_count=0, single_han_count=1)
    monkeypatch.setattr(
        materializer_module,
        "read_normalization_recovery_v5_evaluation_commitment",
        lambda *args, **kwargs: commitment)
    monkeypatch.setattr(
        materializer_module, "read_normalization_recovery_v5_qt_source_pack",
        lambda *args, **kwargs: (
            {"manifest_sha256": source_sha}, (), inventory))
    monkeypatch.setattr(
        materializer_module, "parse_normalization_recovery_v5_qt_archive",
        lambda *args, **kwargs: ((), pairs, summary))
    materialization, records = (
        materialize_normalization_recovery_v6_labels_after_guard(
            guard_consumed=1,
            qt_source_pack_dir=source,
            expected_qt_source_manifest_sha256=source_sha,
            evaluation_commitment_dir=tmp_path / "commitment",
            expected_evaluation_commitment_manifest_sha256=commitment_sha,
        ))
    assert materialization["label_materialization_count"] == 2
    assert materialization["source_identity_reselection_count"] == 0
    assert [item["evaluation_id"] for item in records] == [
        item["pair_id"] for item in pairs]
    assert sum(item["local_mapping"] for item in records) == 1

    monkeypatch.setattr(
        materializer_module, "read_normalization_recovery_v5_qt_source_pack",
        lambda *args, **kwargs: (
            {"manifest_sha256": source_sha}, (), inventory[:-1]))
    with pytest.raises(BroadQaExternalDataError, match="label/identity"):
        materialize_normalization_recovery_v6_labels_after_guard(
            guard_consumed=1,
            qt_source_pack_dir=source,
            expected_qt_source_manifest_sha256=source_sha,
            evaluation_commitment_dir=tmp_path / "commitment",
            expected_evaluation_commitment_manifest_sha256=commitment_sha,
        )


def _evaluation_record(
        *,
        evaluation_id: str,
        input_text: str,
        expected_output: str,
        identity: int,
        variable: int,
        local: int,
        context: int,
        source_sha: str,
        structure_tokens: list[str],
        ) -> dict[str, object]:
    """构造 evaluator 的完整 synthetic record。"""
    single = local + context
    return {
        "contains_han_both": 1,
        "context_conditioned": context,
        "equal_length": 1 - variable,
        "evaluation_id": evaluation_id,
        "expected_output": expected_output,
        "format_version": 1,
        "identity_preservation": identity,
        "input_scalar_count": len(input_text),
        "input_text": input_text,
        "local_mapping": local,
        "output_scalar_count": len(expected_output),
        "pair_id": evaluation_id,
        "record_kind": NORMALIZATION_RECOVERY_V6_EVALUATION_RECORD_KIND,
        "single_han_difference": single,
        "source_conflict": 0,
        "source_identity": {"context": "Synthetic"},
        "source_identity_sha256": _sha(evaluation_id + " source"),
        "source_pack_manifest_sha256": source_sha,
        "structure_equal": 1,
        "structure_tokens": structure_tokens,
        "variable_length": variable,
        "within_scalar_limit": 1,
    }


def test_evaluator_keeps_full_applicability_and_eight_dimension_order() -> None:
    """candidate identity backoff 仍必须进入全分母判分，不能变 not-applicable。"""
    _materials, _pack, outputs, candidate, _audit, _old_commitment = _material()
    rule = outputs["target-whole-rules.jsonl"][0]
    tokens = list(rule["structure_token_variants"][0])
    source_sha = _sha("source")
    commitment_sha = candidate["evaluation_commitment_manifest_sha256"]
    variable = int(len(str(rule["input_text"])) != len(str(rule["output_text"])))
    records = (
        _evaluation_record(
            evaluation_id=_sha("exact"),
            input_text=str(rule["input_text"]),
            expected_output=str(rule["output_text"]),
            identity=0, variable=variable, local=0, context=1,
            source_sha=source_sha, structure_tokens=tokens),
        _evaluation_record(
            evaluation_id=_sha("identity"), input_text="从未见过的输入",
            expected_output="从未见过的输入", identity=1,
            variable=0, local=0, context=0,
            source_sha=source_sha, structure_tokens=[]),
    )
    commitment = _commitment(
        manifest_sha=commitment_sha, source_sha=source_sha,
        identity_count=1, nonidentity_count=1,
        equal_count=2 - variable, variable_count=variable,
        single_han_count=1)
    materialization = {
        "evaluation_commitment_manifest_sha256": commitment_sha,
        "evaluation_record_roster_sha256": hashlib.sha256(
            canonical_json_bytes(records)).hexdigest(),
        "label_materialization_count": len(records),
        "qt_source_manifest_sha256": source_sha,
        "qt_source_payload_read_count": 1,
    }
    report = evaluate_normalization_recovery_v6_candidate(
        commitment=commitment,
        candidate_manifest={
            "candidate_program_sha256": candidate["candidate_program_sha256"],
            "manifest_sha256": NORMALIZATION_RECOVERY_V6_CANDIDATE_V2_MANIFEST_SHA256,
        },
        candidate=candidate,
        materialization=materialization,
        evaluation_records=records,
        family_freeze_manifest_sha256=_sha("family"),
    )
    assert tuple(item["dimension_key"] for item in report["dimensions"]) == (
        NORMALIZATION_RECOVERY_V6_DIMENSION_ORDER)
    coverage = next(item for item in report["dimensions"]
                    if item["dimension_key"] == "END_TO_END_COVERAGE")
    assert coverage["metrics"]["applicable_count"] == len(records)
    assert coverage["metrics"]["exact_output_count"] == len(records)
    assert report["overall_outcome"] == "NE"


def test_evaluator_marks_identity_backoff_on_nonidentity_as_fail() -> None:
    """合法 scope 的未知非 identity 不得借 identity backoff 逃出分母。"""
    _materials, _pack, _outputs, candidate, _audit, _old_commitment = _material()
    source_sha = _sha("fail source")
    commitment_sha = candidate["evaluation_commitment_manifest_sha256"]
    records = (
        _evaluation_record(
            evaluation_id=_sha("unknown nonidentity"), input_text="未知输入",
            expected_output="未知输出", identity=0, variable=0,
            local=0, context=0, source_sha=source_sha,
            structure_tokens=[]),
        _evaluation_record(
            evaluation_id=_sha("known identity"), input_text="保持原样",
            expected_output="保持原样", identity=1, variable=0,
            local=0, context=0, source_sha=source_sha,
            structure_tokens=[]),
    )
    commitment = _commitment(
        manifest_sha=commitment_sha, source_sha=source_sha,
        identity_count=1, nonidentity_count=1,
        equal_count=2, variable_count=0, single_han_count=0)
    materialization = {
        "evaluation_commitment_manifest_sha256": commitment_sha,
        "evaluation_record_roster_sha256": hashlib.sha256(
            canonical_json_bytes(records)).hexdigest(),
        "label_materialization_count": len(records),
        "qt_source_manifest_sha256": source_sha,
        "qt_source_payload_read_count": 1,
    }
    report = evaluate_normalization_recovery_v6_candidate(
        commitment=commitment,
        candidate_manifest={
            "candidate_program_sha256": candidate["candidate_program_sha256"],
            "manifest_sha256": NORMALIZATION_RECOVERY_V6_CANDIDATE_V2_MANIFEST_SHA256,
        },
        candidate=candidate,
        materialization=materialization,
        evaluation_records=records,
        family_freeze_manifest_sha256=_sha("fail family"),
    )
    coverage = next(item for item in report["dimensions"]
                    if item["dimension_key"] == "END_TO_END_COVERAGE")
    assert coverage["outcome"] == "FAIL"
    assert coverage["metrics"]["false_reject_count"] == 1
    assert coverage["metrics"]["applicable_count"] == len(records)
    assert report["overall_outcome"] == "FAIL"


def _install_runner(
        monkeypatch: pytest.MonkeyPatch,
        *,
        fail_materialization: bool = False,
        ) -> None:
    """安装不读取真实 Qt payload 的 runner synthetic readers。"""
    family = {
        "family_commitment_sha256": _sha("family commitment"),
        "manifest_sha256": _sha("family"),
    }
    candidate_manifest = {"manifest_sha256": _sha("candidate")}
    candidate = {"candidate_program_sha256": _sha("program")}
    commitment = {"manifest_sha256": _sha("commitment")}
    monkeypatch.setattr(
        runner_module, "require_normalization_recovery_v6_k_root",
        lambda value: Path(value).resolve())
    monkeypatch.setattr(
        runner_module,
        "read_normalization_recovery_v6_evaluation_family_freeze",
        lambda *args, **kwargs: (
            family, candidate_manifest, candidate, commitment))
    if fail_materialization:
        def materializer(**kwargs):
            """模拟 guard 后 Qt identity 漂移。"""
            raise BroadQaExternalDataError("synthetic Qt drift")
    else:
        def materializer(**kwargs):
            """返回无需真实 label 的最小 synthetic material。"""
            return ({"label_materialization_count": 1}, ({},))
    monkeypatch.setattr(
        runner_module,
        "materialize_normalization_recovery_v6_labels_after_guard",
        materializer)
    monkeypatch.setattr(
        runner_module, "evaluate_normalization_recovery_v6_candidate",
        lambda **kwargs: {
            "evaluation_report_sha256": _sha("report"),
            "overall_outcome": "PASS",
        })


def _runner_arguments(tmp_path: Path) -> dict[str, object]:
    """构造 runner 的完整目录和冻结参数。"""
    values = _family_arguments(tmp_path)
    family = tmp_path / "family"
    family.mkdir()
    values.update({
        "run_root": tmp_path,
        "family_freeze_dir": family,
        "publication_dir": tmp_path / "publication",
    })
    return values


def test_runner_consumes_guard_once_and_seals_post_guard_failure(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        ) -> None:
    """唯一 runner 成功或失败都必须永久消费 publication identity。"""
    _install_runner(monkeypatch)
    arguments = _runner_arguments(tmp_path)
    report = run_normalization_recovery_v6_formal_evaluation(**arguments)
    publication = Path(arguments["publication_dir"])
    assert report["overall_outcome"] == "PASS"
    assert (publication / "run-000001.guard.json").is_file()
    assert (publication / "run-000001.report.json").is_file()
    with pytest.raises(BroadQaExternalDataError, match="已消费"):
        run_normalization_recovery_v6_formal_evaluation(**arguments)

    failed_root = tmp_path / "failed"
    failed_root.mkdir()
    failed_arguments = _runner_arguments(failed_root)
    _install_runner(monkeypatch, fail_materialization=True)
    with pytest.raises(BroadQaExternalDataError, match="Qt drift"):
        run_normalization_recovery_v6_formal_evaluation(**failed_arguments)
    failed_publication = Path(failed_arguments["publication_dir"])
    assert (failed_publication / "run-000001.guard.json").is_file()
    assert (failed_publication / "run-000001.failure.json").is_file()
    with pytest.raises(BroadQaExternalDataError, match="已消费"):
        run_normalization_recovery_v6_formal_evaluation(**failed_arguments)
