"""执行 recovery-v4 Firefox reserve 的唯一 formal run。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v4_evaluation_family import (
    normalization_recovery_v4_path_within,
    read_normalization_recovery_v4_evaluation_family_freeze,
    require_normalization_recovery_v4_k_root,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v4_evaluator import (
    evaluate_normalization_recovery_v4_candidate,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v4_reserve_materialization import (
    materialize_normalization_recovery_v4_reserve_after_guard,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


NORMALIZATION_RECOVERY_V4_FORMAL_GUARD_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V4_FORMAL_RUN_GUARD_V1")
NORMALIZATION_RECOVERY_V4_FORMAL_REPORT_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V4_FORMAL_REPORT_V1")
NORMALIZATION_RECOVERY_V4_FORMAL_FAILURE_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V4_FORMAL_FAILURE_V1")


def _sha256(payload: bytes) -> str:
    """返回 guard、report 或 failure publication SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _write_immutable(path: Path, value: dict[str, object]) -> None:
    """以 exclusive create 写入规范 publication。"""
    with path.open("xb") as handle:
        handle.write(canonical_json_line(value))


def _family_arguments(arguments: dict[str, object]) -> dict[str, object]:
    """选择 family strict reader 的冻结参数。"""
    excluded = {
        "run_root", "family_freeze_dir", "publication_dir",
        "firefox_source_pack_dir",
    }
    return {key: value for key, value in arguments.items()
            if key not in excluded}


def _failure_seal(
        publication: Path,
        error: BaseException,
        *,
        phase: str,
        family_freeze_sha256: str,
        ) -> None:
    """guard 后异常写入不可覆盖 NE failure seal。"""
    path = publication / "run-000001.failure.json"
    if path.exists() or not (publication / "run-000001.guard.json").is_file():
        return
    value = {
        "artifact_kind": NORMALIZATION_RECOVERY_V4_FORMAL_FAILURE_KIND,
        "error_type": type(error).__name__,
        "evaluation_run_count": 1,
        "failure_phase": phase,
        "family_freeze_sha256": family_freeze_sha256,
        "format_version": 1,
        "mastery_claimed": 0,
        "production_enabled": 0,
        "receipt_published": 0,
        "reserve_payload_read_count": int(
            phase in {"LABEL_MATERIALIZATION", "EVALUATION_RUNNING"}),
        "run_id": 1,
        "status": "NE_NO_RECEIPT",
    }
    try:
        _write_immutable(path, value)
    except OSError:
        return


def run_normalization_recovery_v4_formal_evaluation(
        **arguments: object,
        ) -> dict[str, object]:
    """先消费唯一 guard，再物化整 reserve label 并发布六维结果。"""
    root = require_normalization_recovery_v4_k_root(arguments["run_root"])
    family_dir = normalization_recovery_v4_path_within(
        root, arguments["family_freeze_dir"], label="family_freeze_dir")
    publication = normalization_recovery_v4_path_within(
        root, arguments["publication_dir"], label="publication_dir")
    source_pack = normalization_recovery_v4_path_within(
        root, arguments["firefox_source_pack_dir"],
        label="firefox_source_pack_dir")
    if not family_dir.is_dir() or not source_pack.is_dir():
        raise BroadQaExternalDataError(
            "recovery v4 formal family/source pack 缺失")
    if publication.exists():
        raise BroadQaExternalDataError(
            "recovery v4 formal publication 已消费")
    family, candidate, program, commitment = (
        read_normalization_recovery_v4_evaluation_family_freeze(
            family_dir, **_family_arguments(arguments)))
    family_sha = str(family["manifest_sha256"])
    publication.mkdir()
    phase = "PRE_GUARD"
    try:
        guard = {
            "artifact_kind": NORMALIZATION_RECOVERY_V4_FORMAL_GUARD_KIND,
            "candidate_manifest_sha256": candidate["manifest_sha256"],
            "candidate_program_sha256": program["candidate_program_sha256"],
            "evaluation_commitment_manifest_sha256": commitment[
                "manifest_sha256"],
            "evaluation_run_count": 1,
            "family_commitment_sha256": family[
                "family_commitment_sha256"],
            "family_freeze_sha256": family_sha,
            "format_version": 1,
            "overwrite_allowed": 0,
            "reserve_payload_read_count": 0,
            "run_id": 1,
            "status": "CONSUMED_BEFORE_RESERVE_LABEL_MATERIALIZATION",
        }
        _write_immutable(publication / "run-000001.guard.json", guard)
        phase = "LABEL_MATERIALIZATION"
        materialization, reserve = (
            materialize_normalization_recovery_v4_reserve_after_guard(
                guard_consumed=1,
                prior_evaluation_protocol_dir=arguments[
                    "prior_evaluation_protocol_dir"],
                expected_prior_evaluation_manifest_sha256=arguments[
                    "expected_prior_evaluation_manifest_sha256"],
                firefox_source_pack_dir=source_pack,
                evaluation_commitment_dir=arguments[
                    "evaluation_commitment_dir"],
                expected_evaluation_commitment_manifest_sha256=arguments[
                    "expected_evaluation_commitment_manifest_sha256"],
            ))
        phase = "EVALUATION_RUNNING"
        evaluation = evaluate_normalization_recovery_v4_candidate(
            commitment=commitment,
            candidate_manifest=candidate,
            program=program,
            materialization=materialization,
            reserve_records=reserve,
            family_freeze_manifest_sha256=family_sha,
        )
        formal = {
            "artifact_kind": NORMALIZATION_RECOVERY_V4_FORMAL_REPORT_KIND,
            "evaluation_report": evaluation,
            "evaluation_report_sha256": evaluation[
                "evaluation_report_sha256"],
            "evaluation_run_count": 1,
            "family_freeze_sha256": family_sha,
            "format_version": 1,
            "label_materialization_count": materialization[
                "label_materialization_count"],
            "mastery_claimed": 0,
            "overall_outcome": evaluation["overall_outcome"],
            "production_enabled": 0,
            "receipt_published": 0,
            "reserve_payload_read_count": 1,
            "run_id": 1,
            "status": evaluation["overall_outcome"],
        }
        report_path = publication / "run-000001.report.json"
        _write_immutable(report_path, formal)
        phase = "REPORT_PUBLISHED"
        return {**formal, "manifest_sha256": _sha256(
            report_path.read_bytes())}
    except BaseException as error:
        _failure_seal(
            publication, error, phase=phase,
            family_freeze_sha256=family_sha)
        raise


def main(argv: list[str] | None = None) -> int:
    """解析显式 K 盘路径并执行唯一 recovery-v4 formal run。"""
    parser = argparse.ArgumentParser()
    for name in (
            "run_root", "repository_root", "family_freeze_dir",
            "publication_dir", "candidate_dir",
            "prior_evaluation_protocol_dir", "base_training_protocol_dir",
            "base_rule_pack_dir", "v4_training_protocol_dir",
            "v4_rule_pack_dir", "v4_training_audit_dir",
            "evaluation_commitment_dir", "firefox_source_pack_dir"):
        parser.add_argument("--" + name.replace("_", "-"), required=True)
    for name in (
            "expected_candidate_manifest_sha256",
            "expected_prior_evaluation_manifest_sha256",
            "expected_base_training_manifest_sha256",
            "expected_base_rule_pack_manifest_sha256",
            "expected_v4_training_manifest_sha256",
            "expected_v4_rule_pack_manifest_sha256",
            "expected_v4_training_audit_manifest_sha256",
            "expected_evaluation_commitment_manifest_sha256"):
        parser.add_argument("--" + name.replace("_", "-"), required=True)
    report = run_normalization_recovery_v4_formal_evaluation(
        **vars(parser.parse_args(argv)))
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["overall_outcome"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "NORMALIZATION_RECOVERY_V4_FORMAL_FAILURE_KIND",
    "NORMALIZATION_RECOVERY_V4_FORMAL_GUARD_KIND",
    "NORMALIZATION_RECOVERY_V4_FORMAL_REPORT_KIND",
    "run_normalization_recovery_v4_formal_evaluation",
]
