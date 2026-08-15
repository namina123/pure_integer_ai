"""执行 recovery-v8 VLC fixed-denominator 的唯一 formal run。"""
from __future__ import annotations

import hashlib
from pathlib import Path

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_evaluation_family import (
    NORMALIZATION_RECOVERY_V8_EVALUATION_DATA_ARGUMENTS,
    normalization_recovery_v8_path_within,
    read_normalization_recovery_v8_evaluation_family_freeze,
    require_normalization_recovery_v8_k_root,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_evaluator import (
    evaluate_normalization_recovery_v8_candidate,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_label_materialization import (
    materialize_normalization_recovery_v8_labels_after_guard,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


NORMALIZATION_RECOVERY_V8_FORMAL_GUARD_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V8_FORMAL_RUN_GUARD_V1")
NORMALIZATION_RECOVERY_V8_FORMAL_REPORT_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V8_FORMAL_REPORT_V1")
NORMALIZATION_RECOVERY_V8_FORMAL_FAILURE_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V8_FORMAL_FAILURE_V1")


def _sha256(payload: bytes) -> str:
    """返回 guard、report 或 failure publication SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _write_immutable(path: Path, value: dict[str, object]) -> None:
    """以 exclusive create 写入规范 publication。"""
    with path.open("xb") as handle:
        handle.write(canonical_json_line(value))


def _family_arguments(arguments: dict[str, object]) -> dict[str, object]:
    """选择 family strict reader 的全部冻结参数。"""
    excluded = {"run_root", "family_freeze_dir", "publication_dir"}
    return {key: value for key, value in arguments.items() if key not in excluded}


def _overlap(left: Path, right: Path) -> bool:
    """判断 publication 与冻结 artifact 根是否存在包含关系。"""
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def _failure_seal(
        publication: Path, error: BaseException, *, phase: str,
        family_freeze_sha256: str,
        ) -> None:
    """guard 后异常写入不可覆盖 NE failure seal。"""
    path = publication / "run-000001.failure.json"
    if path.exists() or not (publication / "run-000001.guard.json").is_file():
        return
    value = {
        "artifact_kind": NORMALIZATION_RECOVERY_V8_FORMAL_FAILURE_KIND,
        "error_type": type(error).__name__,
        "evaluation_run_count": 1,
        "failure_phase": phase,
        "family_freeze_sha256": family_freeze_sha256,
        "format_version": 1,
        "mastery_claimed": 0,
        "production_enabled": 0,
        "receipt_published": 0,
        "retry_allowed": 0,
        "run_id": 1,
        "status": "NE_NO_RECEIPT_NO_RERUN",
        "vlc_source_payload_read_count": int(
            phase in {"LABEL_MATERIALIZATION", "EVALUATION_RUNNING"}),
    }
    try:
        _write_immutable(path, value)
    except OSError:
        return


def run_normalization_recovery_v8_formal_evaluation(
        **arguments: object,
        ) -> dict[str, object]:
    """先消费唯一 guard，再物化全 VLC labels 并发布六维 aggregate。"""
    root = require_normalization_recovery_v8_k_root(arguments["run_root"])
    family_dir = normalization_recovery_v8_path_within(
        root, arguments["family_freeze_dir"], label="family_freeze_dir")
    publication = normalization_recovery_v8_path_within(
        root, arguments["publication_dir"], label="publication_dir")
    if not family_dir.is_dir():
        raise BroadQaExternalDataError("v8 formal family 缺失")
    frozen_roots = (family_dir, *tuple(
        normalization_recovery_v8_path_within(
            root, arguments[name], label=name)
        for name in NORMALIZATION_RECOVERY_V8_EVALUATION_DATA_ARGUMENTS))
    if (publication.exists()
            or any(_overlap(publication, path) for path in frozen_roots)):
        raise BroadQaExternalDataError("v8 formal publication 已消费或 artifact 混淆")
    family, candidate_manifest, candidate, commitment = (
        read_normalization_recovery_v8_evaluation_family_freeze(
            family_dir, **_family_arguments(arguments)))
    family_sha = str(family["manifest_sha256"])
    publication.mkdir()
    phase = "PRE_GUARD"
    try:
        guard = {
            "artifact_kind": NORMALIZATION_RECOVERY_V8_FORMAL_GUARD_KIND,
            "candidate_manifest_sha256": candidate_manifest["manifest_sha256"],
            "candidate_program_sha256": candidate["candidate_program_sha256"],
            "evaluation_commitment_manifest_sha256": commitment[
                "manifest_sha256"],
            "evaluation_run_count": 1,
            "family_commitment_sha256": family["family_commitment_sha256"],
            "family_freeze_sha256": family_sha,
            "format_version": 1,
            "overwrite_allowed": 0,
            "retry_allowed": 0,
            "run_id": 1,
            "status": "CONSUMED_BEFORE_VLC_LABEL_MATERIALIZATION",
            "vlc_source_payload_read_count": 0,
        }
        _write_immutable(publication / "run-000001.guard.json", guard)
        phase = "LABEL_MATERIALIZATION"
        materialization, records = (
            materialize_normalization_recovery_v8_labels_after_guard(
                guard_consumed=1,
                vlc_source_pack_dir=arguments["vlc_source_pack_dir"],
                expected_vlc_source_manifest_sha256=arguments[
                    "expected_vlc_source_manifest_sha256"],
                v7_commitment_dir=arguments["v7_commitment_dir"],
                expected_v7_commitment_manifest_sha256=arguments[
                    "expected_v7_commitment_manifest_sha256"],
                evaluation_commitment_dir=arguments[
                    "evaluation_commitment_dir"],
                expected_evaluation_commitment_manifest_sha256=arguments[
                    "expected_evaluation_commitment_manifest_sha256"],
            ))
        phase = "EVALUATION_RUNNING"
        evaluation = evaluate_normalization_recovery_v8_candidate(
            commitment=commitment, candidate_manifest=candidate_manifest,
            candidate=candidate, materialization=materialization,
            evaluation_records=records,
            family_freeze_manifest_sha256=family_sha)
        formal = {
            "artifact_kind": NORMALIZATION_RECOVERY_V8_FORMAL_REPORT_KIND,
            "evaluation_report": evaluation,
            "evaluation_report_sha256": evaluation[
                "evaluation_report_sha256"],
            "evaluation_run_count": 1,
            "family_freeze_sha256": family_sha,
            "format_version": 1,
            "individual_label_publication_count": 0,
            "label_materialization_count": materialization[
                "label_materialization_count"],
            "mastery_claimed": 0,
            "overall_outcome": evaluation["overall_outcome"],
            "production_enabled": 0,
            "receipt_published": 0,
            "retry_allowed": 0,
            "run_id": 1,
            "status": evaluation["overall_outcome"],
            "vlc_source_payload_read_count": 1,
        }
        path = publication / "run-000001.report.json"
        _write_immutable(path, formal)
        phase = "REPORT_PUBLISHED"
        return {**formal, "manifest_sha256": _sha256(path.read_bytes())}
    except BaseException as error:
        _failure_seal(
            publication, error, phase=phase,
            family_freeze_sha256=family_sha)
        raise


__all__ = [
    "NORMALIZATION_RECOVERY_V8_FORMAL_FAILURE_KIND",
    "NORMALIZATION_RECOVERY_V8_FORMAL_GUARD_KIND",
    "NORMALIZATION_RECOVERY_V8_FORMAL_REPORT_KIND",
    "run_normalization_recovery_v8_formal_evaluation",
]
