"""运行唯一一次冻结的 normalization successor evaluation family。

runner 在读取 evaluation inventory 前先发布不可覆盖 guard。PASS、FAIL 或 NE
均只发布报告，不启用 pack、不声明 mastery；异常封存 failure seal 并禁止重跑。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_successor_candidate_clone import (
    compile_normalization_successor_candidate,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_successor_evaluation_family import (
    normalization_successor_path_within,
    read_normalization_successor_evaluation_family_freeze,
    require_normalization_successor_k_run_root,
    validate_normalization_successor_evaluation_paths,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_successor_evaluation_protocol import (
    read_normalization_successor_evaluation_inventory_only,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_successor_evaluator import (
    evaluate_normalization_successor_candidate,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_successor_rule_pack import (
    read_normalization_successor_rule_pack,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


NORMALIZATION_SUCCESSOR_FORMAL_RUN_GUARD_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_SUCCESSOR_FORMAL_RUN_GUARD_V1")
NORMALIZATION_SUCCESSOR_FORMAL_REPORT_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_SUCCESSOR_FORMAL_REPORT_V1")
NORMALIZATION_SUCCESSOR_FORMAL_FAILURE_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_SUCCESSOR_FORMAL_FAILURE_SEAL_V1")


def _sha256(payload: bytes) -> str:
    """返回规范文件或错误证据摘要。"""
    return hashlib.sha256(payload).hexdigest()


def _write_immutable(path: Path, value: dict[str, object]) -> None:
    """不可覆盖写入规范 JSON object。"""
    with path.open("xb") as handle:
        handle.write(canonical_json_line(value))


def _freeze_arguments(
        *,
        repository_root: str | Path,
        evaluation_protocol_dir: str | Path,
        expected_evaluation_protocol_manifest_sha256: str,
        training_protocol_dir: str | Path,
        expected_training_protocol_manifest_sha256: str,
        fresh_learner_dir: str | Path,
        expected_fresh_learner_manifest_sha256: str,
        resumed_learner_dir: str | Path,
        expected_resumed_learner_manifest_sha256: str,
        rule_pack_dir: str | Path,
        expected_rule_pack_manifest_sha256: str,
        ) -> dict[str, object]:
    """集中形成 family freeze 的完整 live identity 参数。"""
    return {
        "repository_root": repository_root,
        "evaluation_protocol_dir": evaluation_protocol_dir,
        "expected_evaluation_protocol_manifest_sha256": (
            expected_evaluation_protocol_manifest_sha256),
        "training_protocol_dir": training_protocol_dir,
        "expected_training_protocol_manifest_sha256": (
            expected_training_protocol_manifest_sha256),
        "fresh_learner_dir": fresh_learner_dir,
        "expected_fresh_learner_manifest_sha256": (
            expected_fresh_learner_manifest_sha256),
        "resumed_learner_dir": resumed_learner_dir,
        "expected_resumed_learner_manifest_sha256": (
            expected_resumed_learner_manifest_sha256),
        "rule_pack_dir": rule_pack_dir,
        "expected_rule_pack_manifest_sha256": (
            expected_rule_pack_manifest_sha256),
    }


def _failure_seal(
        publication: Path,
        error: BaseException,
        *,
        phase: str,
        family_freeze_sha256: str,
        ) -> None:
    """首次失败时封存 phase 与非明文错误证据。"""
    path = publication / "run-000001.failure.json"
    if path.exists():
        return
    error_identity = (
        type(error).__name__ + "\0" + str(error)).encode("utf-8")
    _write_immutable(path, {
        "artifact_kind": NORMALIZATION_SUCCESSOR_FORMAL_FAILURE_KIND,
        "error_evidence_sha256": _sha256(error_identity),
        "error_type": type(error).__name__,
        "evaluation_run_count": int(phase != "PRE_GUARD"),
        "failure_phase": phase,
        "family_freeze_sha256": family_freeze_sha256,
        "format_version": 1,
        "mastery_claimed": 0,
        "production_enabled": 0,
        "receipt_published": 0,
        "reserve_payload_read_count": 0,
        "status": "NE_NO_RECEIPT",
    })


def run_normalization_successor_formal_evaluation(
        *,
        run_root: str | Path,
        repository_root: str | Path,
        family_freeze_dir: str | Path,
        publication_dir: str | Path,
        evaluation_protocol_dir: str | Path,
        expected_evaluation_protocol_manifest_sha256: str,
        training_protocol_dir: str | Path,
        expected_training_protocol_manifest_sha256: str,
        fresh_learner_dir: str | Path,
        expected_fresh_learner_manifest_sha256: str,
        resumed_learner_dir: str | Path,
        expected_resumed_learner_manifest_sha256: str,
        rule_pack_dir: str | Path,
        expected_rule_pack_manifest_sha256: str,
        ) -> dict[str, object]:
    """消费冻结 family 并发布唯一报告；调用即构成正式运行。"""
    root = require_normalization_successor_k_run_root(run_root)
    arguments = _freeze_arguments(
        repository_root=repository_root,
        evaluation_protocol_dir=evaluation_protocol_dir,
        expected_evaluation_protocol_manifest_sha256=(
            expected_evaluation_protocol_manifest_sha256),
        training_protocol_dir=training_protocol_dir,
        expected_training_protocol_manifest_sha256=(
            expected_training_protocol_manifest_sha256),
        fresh_learner_dir=fresh_learner_dir,
        expected_fresh_learner_manifest_sha256=(
            expected_fresh_learner_manifest_sha256),
        resumed_learner_dir=resumed_learner_dir,
        expected_resumed_learner_manifest_sha256=(
            expected_resumed_learner_manifest_sha256),
        rule_pack_dir=rule_pack_dir,
        expected_rule_pack_manifest_sha256=(
            expected_rule_pack_manifest_sha256),
    )
    validate_normalization_successor_evaluation_paths(root, arguments)
    family_dir = normalization_successor_path_within(
        root, family_freeze_dir, label="normalization successor family freeze")
    if not family_dir.is_dir():
        raise BroadQaExternalDataError(
            "normalization successor family freeze 目录不存在")
    publication = normalization_successor_path_within(
        root, publication_dir, label="normalization successor publication")
    if publication.exists():
        raise BroadQaExternalDataError(
            "normalization successor formal family 已消费或 publication 已存在")
    freeze = read_normalization_successor_evaluation_family_freeze(
        family_dir, **arguments)
    freeze_sha = freeze["manifest_sha256"]
    publication.mkdir(parents=True)
    phase = "PRE_GUARD"
    try:
        guard = {
            "artifact_kind": NORMALIZATION_SUCCESSOR_FORMAL_RUN_GUARD_KIND,
            "evaluation_run_count": 1,
            "family_commitment_sha256": freeze[
                "family_commitment_sha256"],
            "family_freeze_sha256": freeze_sha,
            "format_version": 1,
            "overwrite_allowed": 0,
            "run_id": 1,
            "status": "CONSUMED_BEFORE_EVALUATION_PAYLOAD_READ",
        }
        _write_immutable(publication / "run-000001.guard.json", guard)
        phase = "GUARD_CONSUMED"
        protocol_manifest, evaluation = (
            read_normalization_successor_evaluation_inventory_only(
                evaluation_protocol_dir,
                expected_manifest_sha256=(
                    expected_evaluation_protocol_manifest_sha256),
            ))
        if (protocol_manifest["manifest_sha256"]
                != freeze["evaluation_protocol_manifest_sha256"]
                or protocol_manifest["evaluation_inventory"]
                != freeze["evaluation_inventory_identity"]):
            raise BroadQaExternalDataError(
                "normalization successor formal protocol/freeze 漂移")
        pack_manifest, outputs = read_normalization_successor_rule_pack(
            rule_pack_dir,
            protocol_dir=training_protocol_dir,
            expected_protocol_manifest_sha256=(
                expected_training_protocol_manifest_sha256),
            expected_pack_manifest_sha256=expected_rule_pack_manifest_sha256,
        )
        program = compile_normalization_successor_candidate(
            rule_pack_manifest=pack_manifest, outputs=outputs)
        if (program.sha256()
                != freeze["candidate_freeze"]["candidate_clone_sha256"]):
            raise BroadQaExternalDataError(
                "normalization successor formal clone/freeze 漂移")
        phase = "EVALUATION_RUNNING"
        report = evaluate_normalization_successor_candidate(
            protocol_manifest=protocol_manifest,
            evaluation_records=evaluation,
            rule_pack_manifest=pack_manifest,
            pack_outputs=outputs,
            program=program,
        )
        formal = {
            "artifact_kind": NORMALIZATION_SUCCESSOR_FORMAL_REPORT_KIND,
            "evaluation_report": report.to_dict(),
            "evaluation_report_sha256": report.sha256(),
            "evaluation_run_count": 1,
            "family_freeze_sha256": freeze_sha,
            "format_version": 1,
            "mastery_claimed": 0,
            "overall_outcome": report.overall_outcome,
            "production_enabled": 0,
            "receipt_published": 0,
            "reserve_payload_read_count": 0,
            "run_id": 1,
            "status": report.overall_outcome,
        }
        _write_immutable(publication / "run-000001.report.json", formal)
        phase = "REPORT_PUBLISHED"
        return {
            **formal,
            "manifest_sha256": _sha256(
                (publication / "run-000001.report.json").read_bytes()),
        }
    except BaseException as error:
        _failure_seal(
            publication,
            error,
            phase=phase,
            family_freeze_sha256=freeze_sha,
        )
        raise


def main(argv: list[str] | None = None) -> int:
    """解析显式 K 盘路径并运行唯一 successor formal evaluation。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--family-freeze-dir", required=True)
    parser.add_argument("--publication-dir", required=True)
    parser.add_argument("--evaluation-protocol-dir", required=True)
    parser.add_argument(
        "--expected-evaluation-protocol-manifest-sha256", required=True)
    parser.add_argument("--training-protocol-dir", required=True)
    parser.add_argument(
        "--expected-training-protocol-manifest-sha256", required=True)
    parser.add_argument("--fresh-learner-dir", required=True)
    parser.add_argument(
        "--expected-fresh-learner-manifest-sha256", required=True)
    parser.add_argument("--resumed-learner-dir", required=True)
    parser.add_argument(
        "--expected-resumed-learner-manifest-sha256", required=True)
    parser.add_argument("--rule-pack-dir", required=True)
    parser.add_argument("--expected-rule-pack-manifest-sha256", required=True)
    arguments = parser.parse_args(argv)
    report = run_normalization_successor_formal_evaluation(**vars(arguments))
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["overall_outcome"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "NORMALIZATION_SUCCESSOR_FORMAL_FAILURE_KIND",
    "NORMALIZATION_SUCCESSOR_FORMAL_REPORT_KIND",
    "NORMALIZATION_SUCCESSOR_FORMAL_RUN_GUARD_KIND",
    "run_normalization_successor_formal_evaluation",
]
