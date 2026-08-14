"""运行唯一一次冻结 normalization ICU evaluation family。

runner 在读取 evaluation inventory 前先发布不可覆盖 guard。任何 PASS、FAIL 或 NE
都只发布报告，不启用 pack、不发布 mastery；异常发布 failure seal 且禁止重跑。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_candidate_clone import (
    compile_normalization_candidate_clone,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_contrastive_protocol import (
    read_normalization_contrastive_protocol,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_icu_evaluation_family import (
    normalization_path_within,
    read_normalization_icu_evaluation_family_freeze,
    require_normalization_k_run_root,
    validate_normalization_evaluation_data_paths,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_icu_evaluation_protocol import (
    read_normalization_icu_evaluation_protocol,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_icu_evaluator import (
    evaluate_normalization_icu_candidate,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_rule_pack_v3 import (
    read_normalization_rule_pack_v3,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


NORMALIZATION_ICU_FORMAL_RUN_GUARD_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_ICU_FORMAL_RUN_GUARD_V1")
NORMALIZATION_ICU_FORMAL_REPORT_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_ICU_FORMAL_REPORT_V1")
NORMALIZATION_ICU_FORMAL_FAILURE_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_ICU_FORMAL_FAILURE_SEAL_V1")


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
        icu_source_pack_dir: str | Path,
        evaluation_protocol_dir: str | Path,
        normalization_source_pack_dir: str | Path,
        contrastive_protocol_dir: str | Path,
        rule_pack_dir: str | Path,
        fresh_learner_dir: str | Path,
        resumed_learner_dir: str | Path,
        ) -> dict[str, object]:
    """集中形成 family freeze 的 live identity 参数。"""
    return {
        "repository_root": repository_root,
        "icu_source_pack_dir": icu_source_pack_dir,
        "evaluation_protocol_dir": evaluation_protocol_dir,
        "normalization_source_pack_dir": normalization_source_pack_dir,
        "contrastive_protocol_dir": contrastive_protocol_dir,
        "rule_pack_dir": rule_pack_dir,
        "fresh_learner_dir": fresh_learner_dir,
        "resumed_learner_dir": resumed_learner_dir,
    }


def _failure_seal(
        publication: Path,
        error: BaseException,
        *,
        phase: str,
        family_freeze_sha256: str,
        ) -> None:
    """首次失败时封存 phase 和非明文错误证据。"""
    path = publication / "run-000001.failure.json"
    if path.exists():
        return
    error_identity = (
        type(error).__name__ + "\0" + str(error)).encode("utf-8")
    _write_immutable(path, {
        "artifact_kind": NORMALIZATION_ICU_FORMAL_FAILURE_KIND,
        "error_evidence_sha256": _sha256(error_identity),
        "error_type": type(error).__name__,
        "evaluation_run_count": int(phase != "PRE_GUARD"),
        "failure_phase": phase,
        "family_freeze_sha256": family_freeze_sha256,
        "format_version": 1,
        "mastery_claimed": 0,
        "production_enabled": 0,
        "receipt_published": 0,
        "status": "NE_NO_RECEIPT",
    })


def run_normalization_icu_formal_evaluation(
        *,
        run_root: str | Path,
        repository_root: str | Path,
        family_freeze_dir: str | Path,
        publication_dir: str | Path,
        icu_source_pack_dir: str | Path,
        evaluation_protocol_dir: str | Path,
        normalization_source_pack_dir: str | Path,
        contrastive_protocol_dir: str | Path,
        rule_pack_dir: str | Path,
        fresh_learner_dir: str | Path,
        resumed_learner_dir: str | Path,
        ) -> dict[str, object]:
    """消费冻结 family 并发布唯一报告；调用即构成正式运行。"""
    root = require_normalization_k_run_root(run_root)
    arguments = _freeze_arguments(
        repository_root=repository_root,
        icu_source_pack_dir=icu_source_pack_dir,
        evaluation_protocol_dir=evaluation_protocol_dir,
        normalization_source_pack_dir=normalization_source_pack_dir,
        contrastive_protocol_dir=contrastive_protocol_dir,
        rule_pack_dir=rule_pack_dir,
        fresh_learner_dir=fresh_learner_dir,
        resumed_learner_dir=resumed_learner_dir,
    )
    validate_normalization_evaluation_data_paths(root, arguments)
    family_dir = normalization_path_within(
        root, family_freeze_dir, label="normalization family freeze")
    if not family_dir.is_dir():
        raise BroadQaExternalDataError(
            "normalization family freeze 目录不存在")
    publication = normalization_path_within(
        root, publication_dir, label="normalization formal publication")
    if publication.exists():
        raise BroadQaExternalDataError(
            "normalization formal family 已消费或 publication 已存在")
    freeze = read_normalization_icu_evaluation_family_freeze(
        family_dir, **arguments)
    freeze_sha = freeze["manifest_sha256"]
    publication.mkdir(parents=True)
    phase = "PRE_GUARD"
    try:
        guard = {
            "artifact_kind": NORMALIZATION_ICU_FORMAL_RUN_GUARD_KIND,
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
        protocol_manifest, evaluation, _reserve_identity = (
            read_normalization_icu_evaluation_protocol(
                evaluation_protocol_dir,
                source_pack_dir=icu_source_pack_dir,
            ))
        if (protocol_manifest["manifest_sha256"]
                != freeze["evaluation_protocol_manifest_sha256"]
                or protocol_manifest["evaluation_inventory"]
                != freeze["evaluation_inventory_identity"]):
            raise BroadQaExternalDataError(
                "normalization formal protocol/freeze 漂移")
        pack_manifest, accepted, rejected = read_normalization_rule_pack_v3(
            rule_pack_dir,
            source_pack_dir=normalization_source_pack_dir,
            contrastive_protocol_dir=contrastive_protocol_dir,
            fresh_checkpoint_chain_path=(
                Path(fresh_learner_dir).resolve() / "checkpoints.jsonl"),
            resumed_checkpoint_chain_path=(
                Path(resumed_learner_dir).resolve() / "checkpoints.jsonl"),
        )
        _, _, trials = read_normalization_contrastive_protocol(
            contrastive_protocol_dir,
            source_pack_dir=normalization_source_pack_dir,
        )
        program = compile_normalization_candidate_clone(
            rule_pack_manifest_sha256=pack_manifest["manifest_sha256"],
            accepted_rules=accepted,
            rejected_trials=rejected,
            contrastive_trials=trials,
        )
        if (program.sha256()
                != freeze["candidate_freeze"]["candidate_clone_sha256"]):
            raise BroadQaExternalDataError(
                "normalization formal clone/freeze 漂移")
        phase = "EVALUATION_RUNNING"
        report = evaluate_normalization_icu_candidate(
            protocol_manifest=protocol_manifest,
            evaluation_records=evaluation,
            accepted_rules=accepted,
            rejected_trials=rejected,
            contrastive_trials=trials,
            program=program,
        )
        report_value = report.to_dict()
        formal = {
            "artifact_kind": NORMALIZATION_ICU_FORMAL_REPORT_KIND,
            "evaluation_report": report_value,
            "evaluation_report_sha256": report.sha256(),
            "evaluation_run_count": 1,
            "family_freeze_sha256": freeze_sha,
            "format_version": 1,
            "mastery_claimed": 0,
            "overall_outcome": report.overall_outcome,
            "production_enabled": 0,
            "receipt_published": 0,
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
    """解析显式 K 盘路径并运行唯一 formal normalization evaluation。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--family-freeze-dir", required=True)
    parser.add_argument("--publication-dir", required=True)
    parser.add_argument("--icu-source-pack-dir", required=True)
    parser.add_argument("--evaluation-protocol-dir", required=True)
    parser.add_argument("--normalization-source-pack-dir", required=True)
    parser.add_argument("--contrastive-protocol-dir", required=True)
    parser.add_argument("--rule-pack-dir", required=True)
    parser.add_argument("--fresh-learner-dir", required=True)
    parser.add_argument("--resumed-learner-dir", required=True)
    arguments = parser.parse_args(argv)
    report = run_normalization_icu_formal_evaluation(**vars(arguments))
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["overall_outcome"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "NORMALIZATION_ICU_FORMAL_FAILURE_KIND",
    "NORMALIZATION_ICU_FORMAL_REPORT_KIND",
    "NORMALIZATION_ICU_FORMAL_RUN_GUARD_KIND",
    "run_normalization_icu_formal_evaluation",
]
