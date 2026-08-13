"""normalization v3 的单族 development learner 与 K 盘恢复运行器。

learner 完整顺序扫描冻结 OpenCC TRAIN_SOURCE，只在同一 mapping 同时具有来源
SUPPORT 与 context REFUTE 时形成规则。它不读取 evaluation/validation/private，
不调用教师或 LLM，也不启用生产 consumer。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_contrastive_protocol import (
    NORMALIZATION_CONTRASTIVE_FAMILY,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_development_learning_v1 import (
    NORMALIZATION_DEVELOPMENT_DIRECTION,
    NORMALIZATION_DEVELOPMENT_OPERATOR,
    NORMALIZATION_DEVELOPMENT_OPERATOR_VERSION,
    NORMALIZATION_DEVELOPMENT_SCHEMA,
    derive_normalization_development_records_v1,
    normalization_development_output_counts_for_prefix,
    require_normalization_development_records_v1,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_evidence_v3 import (
    read_normalization_training_provenance,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_rule_pack_v3 import (
    normalization_rule_pack_v3_result_sha256,
    validate_normalization_rule_records_v3,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_rule_records_v3 import (
    BroadQaNormalizationAcceptedRuleV3,
    BroadQaNormalizationRejectedTrialV3,
    parse_normalization_accepted_rule_v3,
    parse_normalization_rejected_trial_v3,
)
from pure_integer_ai.experiments.ph2_broad_qa_source_inference_learning_checkpoint import (
    SOURCE_INFERENCE_LEARNING_CHECKPOINT_COMPLETE,
    advance_source_inference_learning_checkpoint,
    append_source_inference_learning_checkpoint,
    initial_source_inference_learning_checkpoint,
    read_source_inference_learning_chain,
    source_inference_learning_prefix_sha256,
)
from pure_integer_ai.experiments.ph2_broad_qa_source_inference_rule_pack import (
    SOURCE_INFERENCE_RULE_RUNTIME_STATE,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


NORMALIZATION_DEVELOPMENT_LEARNER_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_DEVELOPMENT_LEARNER_V1")
NORMALIZATION_DEVELOPMENT_LEARNER_STATUS = (
    "DEVELOPMENT_COMPLETE_NOT_EVALUATED_NOT_DEPLOYED")
NORMALIZATION_DEVELOPMENT_CHECKPOINT_OPEN = "DEVELOPMENT_CHECKPOINT_OPEN"


def _sha256(value: object, *, label: str) -> str:
    """要求运行身份或来源承诺为小写 SHA-256。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise BroadQaExternalDataError(f"{label} 必须是 SHA-256")
    return value


def _require_k_run_root(value: str | Path) -> Path:
    """解析唯一训练工作盘根并拒绝任何非 K 盘回退。"""
    root = Path(value).resolve()
    if not root.is_dir() or root.drive.upper() != "K:":
        raise BroadQaExternalDataError(
            "normalization development run root 必须是已存在的 K 盘目录")
    return root


def _within(root: Path, value: str | Path, *, label: str) -> Path:
    """解析运行输入输出并要求路径始终位于显式 run root。"""
    path = Path(value).resolve()
    if not path.is_relative_to(root):
        raise BroadQaExternalDataError(f"{label} 必须位于 run root 内")
    return path


def validate_normalization_development_checkpoint_chain_v1(
        *,
        chain_path: Path,
        run_id: str,
        protocol_sha: str,
        training_item_ids: tuple[str, ...],
        candidates: tuple[dict[str, object], ...],
        trials: tuple[dict[str, object], ...],
        ):
    """回读恢复链并逐 revision 重算真实 TRAIN 前缀。"""
    chain = read_source_inference_learning_chain(chain_path)
    expected_order = source_inference_learning_prefix_sha256(training_item_ids)
    for checkpoint in chain:
        expected_prefix = source_inference_learning_prefix_sha256(
            training_item_ids[:checkpoint.processed_item_count])
        expected_evidence, expected_records = (
            normalization_development_output_counts_for_prefix(
                candidates=candidates,
                trials=trials,
                processed_item_count=checkpoint.processed_item_count,
            ))
        if (checkpoint.run_id != run_id
                or checkpoint.protocol_manifest_sha256 != protocol_sha
                or checkpoint.operator_family
                != NORMALIZATION_CONTRASTIVE_FAMILY
                or checkpoint.training_item_count != len(training_item_ids)
                or checkpoint.training_item_order_sha256 != expected_order
                or checkpoint.processed_item_prefix_sha256 != expected_prefix
                or checkpoint.evidence_candidate_count != expected_evidence
                or checkpoint.rule_candidate_count != expected_records):
            raise BroadQaExternalDataError(
                "normalization development checkpoint identity/prefix/count 漂移")
    return chain


def _record_payloads(
        accepted: tuple[BroadQaNormalizationAcceptedRuleV3, ...],
        rejected: tuple[BroadQaNormalizationRejectedTrialV3, ...],
        ) -> tuple[bytes, bytes]:
    """返回已经按 record SHA 排序的两类规范 JSONL。"""
    if (tuple(item.sha256() for item in accepted)
            != tuple(sorted(item.sha256() for item in accepted))
            or tuple(item.sha256() for item in rejected)
            != tuple(sorted(item.sha256() for item in rejected))):
        raise BroadQaExternalDataError(
            "normalization development records 排序漂移")
    return (
        b"".join(item.canonical_bytes() for item in accepted),
        b"".join(item.canonical_bytes() for item in rejected),
    )


def _write_or_verify(path: Path, payload: bytes) -> None:
    """首次独占写入；恢复时只接受逐字节相同的既有产物。"""
    if path.exists():
        if path.read_bytes() != payload:
            raise BroadQaExternalDataError(
                f"normalization development 恢复产物漂移: {path.name}")
        return
    with path.open("xb") as handle:
        handle.write(payload)


def run_normalization_development_learner_v1(
        *,
        run_root: str | Path,
        source_pack_dir: str | Path,
        contrastive_protocol_dir: str | Path,
        run_dir: str | Path,
        run_id: str,
        mode: str,
        checkpoint_interval: int = 512,
        stop_after: int | None = None,
        ) -> dict[str, object]:
    """在 K 盘 fresh 或 resume 单族 learner，并以 manifest-last 封口。"""
    root = _require_k_run_root(run_root)
    source_root = _within(root, source_pack_dir, label="source_pack_dir")
    protocol_root = _within(
        root, contrastive_protocol_dir, label="contrastive_protocol_dir")
    target = _within(root, run_dir, label="run_dir")
    run_sha = _sha256(run_id, label="normalization development run_id")
    if mode not in {"fresh", "resume"}:
        raise BroadQaExternalDataError(
            "normalization development mode 必须是 fresh/resume")
    if type(checkpoint_interval) is not int or checkpoint_interval <= 0:
        raise BroadQaExternalDataError(
            "normalization development checkpoint interval 非法")
    (
        source_manifest,
        protocol_manifest,
        candidates,
        trials,
        training_item_ids,
    ) = read_normalization_training_provenance(
        source_pack_dir=source_root,
        contrastive_protocol_dir=protocol_root,
    )
    protocol_sha = str(protocol_manifest["manifest_sha256"])
    chain_path = target / "checkpoints.jsonl"
    manifest_path = target / "manifest.json"
    accepted_path = target / "accepted-rules.jsonl"
    rejected_path = target / "rejected-trials.jsonl"
    if mode == "fresh":
        if target.exists():
            raise BroadQaExternalDataError(
                "normalization development fresh target 已存在")
        target.mkdir(parents=True)
        initial = initial_source_inference_learning_checkpoint(
            run_id=run_sha,
            protocol_manifest_sha256=protocol_sha,
            operator_family=NORMALIZATION_CONTRASTIVE_FAMILY,
            training_item_ids=training_item_ids,
        )
        append_source_inference_learning_checkpoint(chain_path, initial)
        chain = (initial,)
    else:
        if (not target.is_dir() or manifest_path.exists()
                or not chain_path.is_file()):
            raise BroadQaExternalDataError(
                "normalization development resume 状态非法")
        chain = validate_normalization_development_checkpoint_chain_v1(
            chain_path=chain_path,
            run_id=run_sha,
            protocol_sha=protocol_sha,
            training_item_ids=training_item_ids,
            candidates=candidates,
            trials=trials,
        )

    total = len(training_item_ids)
    cursor = chain[-1].processed_item_count
    if stop_after is None:
        target_count = total
    else:
        if (type(stop_after) is not int or not cursor < stop_after < total):
            raise BroadQaExternalDataError(
                "normalization development stop_after 必须推进且早于完成")
        target_count = stop_after
    checkpoint = chain[-1]
    while cursor < target_count:
        next_cursor = min(cursor + checkpoint_interval, target_count)
        evidence_count, record_count = (
            normalization_development_output_counts_for_prefix(
            candidates=candidates,
            trials=trials,
            processed_item_count=next_cursor,
        ))
        checkpoint = advance_source_inference_learning_checkpoint(
            checkpoint,
            training_item_ids=training_item_ids,
            processed_item_ids=training_item_ids[:next_cursor],
            evidence_candidate_count=evidence_count,
            rule_candidate_count=record_count,
            complete=next_cursor == total,
        )
        append_source_inference_learning_checkpoint(chain_path, checkpoint)
        cursor = next_cursor

    if cursor < total:
        return {
            "checkpoint_chain_sha256": hashlib.sha256(
                chain_path.read_bytes()).hexdigest(),
            "processed_item_count": cursor,
            "run_id": run_sha,
            "status": NORMALIZATION_DEVELOPMENT_CHECKPOINT_OPEN,
            "training_item_count": total,
        }

    accepted, rejected = derive_normalization_development_records_v1(
        source_manifest=source_manifest,
        protocol_manifest=protocol_manifest,
        candidates=candidates,
        trials=trials,
    )
    validate_normalization_rule_records_v3(
        source_pack_dir=source_root,
        contrastive_protocol_dir=protocol_root,
        accepted_rules=accepted,
        rejected_trials=rejected,
    )
    accepted_payload, rejected_payload = _record_payloads(accepted, rejected)
    output_evidence_count = sum(
        len(record.evidence_commitments) for record in accepted + rejected)
    if (checkpoint.evidence_candidate_count != output_evidence_count
            or checkpoint.rule_candidate_count
            != len(accepted) + len(rejected)):
        raise BroadQaExternalDataError(
            "normalization development checkpoint/output count 漂移")
    _write_or_verify(accepted_path, accepted_payload)
    _write_or_verify(rejected_path, rejected_payload)
    chain_payload = chain_path.read_bytes()
    result_sha = normalization_rule_pack_v3_result_sha256(
        protocol_manifest_sha256=protocol_sha,
        training_item_ids=training_item_ids,
        accepted_rules=accepted,
        rejected_trials=rejected,
    )
    manifest = {
        "accepted_records_bytes": len(accepted_payload),
        "accepted_records_count": len(accepted),
        "accepted_records_sha256": hashlib.sha256(
            accepted_payload).hexdigest(),
        "artifact_kind": NORMALIZATION_DEVELOPMENT_LEARNER_KIND,
        "checkpoint_chain_bytes": len(chain_payload),
        "checkpoint_chain_sha256": hashlib.sha256(chain_payload).hexdigest(),
        "checkpoint_terminal_sha256": checkpoint.sha256(),
        "contrastive_protocol_manifest_sha256": protocol_sha,
        "evaluation_read_count": 0,
        "format_version": 1,
        "fresh_resume_byte_equivalence_required": 1,
        "operator_family": NORMALIZATION_CONTRASTIVE_FAMILY,
        "production_enabled": 0,
        "rejected_records_bytes": len(rejected_payload),
        "rejected_records_count": len(rejected),
        "rejected_records_sha256": hashlib.sha256(
            rejected_payload).hexdigest(),
        "reserve_read_count": 0,
        "result_sha256": result_sha,
        "run_id": run_sha,
        "runtime_state": SOURCE_INFERENCE_RULE_RUNTIME_STATE,
        "source_pack_manifest_sha256": source_manifest["manifest_sha256"],
        "status": NORMALIZATION_DEVELOPMENT_LEARNER_STATUS,
        "teacher_llm_call_count": 0,
        "training_item_count": total,
        "training_item_order_sha256": (
            source_inference_learning_prefix_sha256(training_item_ids)),
        "validation_read_count": 0,
    }
    with manifest_path.open("xb") as handle:
        handle.write(canonical_json_line(manifest))
    return {
        **manifest,
        "manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()).hexdigest(),
    }


def _parse_jsonl_records(payload: bytes, *, accepted: bool):
    """严格回读非空规范 accepted 或 rejected JSONL。"""
    if not payload or not payload.endswith(b"\n"):
        raise BroadQaExternalDataError(
            "normalization development records 为空或截断")
    parser = (
        parse_normalization_accepted_rule_v3
        if accepted else parse_normalization_rejected_trial_v3)
    return tuple(parser(line + b"\n") for line in payload.splitlines())


def read_normalization_development_learner_v1(
        run_dir: str | Path,
        *,
        source_pack_dir: str | Path,
        contrastive_protocol_dir: str | Path,
        ) -> tuple[
            dict[str, object],
            tuple[BroadQaNormalizationAcceptedRuleV3, ...],
            tuple[BroadQaNormalizationRejectedTrialV3, ...],
        ]:
    """重验 manifest、来源 records、结果 SHA 和完整 checkpoint 链。"""
    root = Path(run_dir).resolve()
    try:
        manifest_payload = (root / "manifest.json").read_bytes()
        manifest = json.loads(manifest_payload)
        accepted_payload = (root / "accepted-rules.jsonl").read_bytes()
        rejected_payload = (root / "rejected-trials.jsonl").read_bytes()
        chain_payload = (root / "checkpoints.jsonl").read_bytes()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "normalization development artifact 不可读") from error
    expected = {
        "accepted_records_bytes", "accepted_records_count",
        "accepted_records_sha256", "artifact_kind",
        "checkpoint_chain_bytes", "checkpoint_chain_sha256",
        "checkpoint_terminal_sha256",
        "contrastive_protocol_manifest_sha256", "evaluation_read_count",
        "format_version", "fresh_resume_byte_equivalence_required",
        "operator_family", "production_enabled", "rejected_records_bytes",
        "rejected_records_count", "rejected_records_sha256",
        "reserve_read_count", "result_sha256", "run_id", "runtime_state",
        "source_pack_manifest_sha256", "status", "teacher_llm_call_count",
        "training_item_count", "training_item_order_sha256",
        "validation_read_count",
    }
    if (not isinstance(manifest, dict) or set(manifest) != expected
            or canonical_json_line(manifest) != manifest_payload
            or manifest["artifact_kind"]
            != NORMALIZATION_DEVELOPMENT_LEARNER_KIND
            or type(manifest["format_version"]) is not int
            or manifest["format_version"] != 1
            or manifest["operator_family"] != NORMALIZATION_CONTRASTIVE_FAMILY
            or manifest["status"] != NORMALIZATION_DEVELOPMENT_LEARNER_STATUS
            or manifest["runtime_state"] != SOURCE_INFERENCE_RULE_RUNTIME_STATE
            or any(type(manifest[name]) is not int or manifest[name] != 0
                   for name in (
                       "evaluation_read_count", "production_enabled",
                       "reserve_read_count", "teacher_llm_call_count",
                       "validation_read_count"))
            or type(manifest["fresh_resume_byte_equivalence_required"])
            is not int
            or manifest["fresh_resume_byte_equivalence_required"] != 1):
        raise BroadQaExternalDataError(
            "normalization development manifest 漂移")
    for name in (
            "accepted_records_sha256", "checkpoint_chain_sha256",
            "checkpoint_terminal_sha256",
            "contrastive_protocol_manifest_sha256",
            "rejected_records_sha256", "result_sha256", "run_id",
            "source_pack_manifest_sha256", "training_item_order_sha256"):
        _sha256(manifest[name], label=f"normalization development {name}")
    for prefix, payload in (
            ("accepted", accepted_payload), ("rejected", rejected_payload)):
        if (type(manifest[f"{prefix}_records_bytes"]) is not int
                or manifest[f"{prefix}_records_bytes"] != len(payload)
                or type(manifest[f"{prefix}_records_count"]) is not int
                or manifest[f"{prefix}_records_count"] <= 0
                or manifest[f"{prefix}_records_sha256"]
                != hashlib.sha256(payload).hexdigest()):
            raise BroadQaExternalDataError(
                f"normalization development {prefix} commitment 漂移")
    if (type(manifest["checkpoint_chain_bytes"]) is not int
            or manifest["checkpoint_chain_bytes"] != len(chain_payload)
            or manifest["checkpoint_chain_sha256"]
            != hashlib.sha256(chain_payload).hexdigest()
            or type(manifest["training_item_count"]) is not int
            or manifest["training_item_count"] <= 0):
        raise BroadQaExternalDataError(
            "normalization development chain/TRAIN commitment 漂移")
    accepted = _parse_jsonl_records(accepted_payload, accepted=True)
    rejected = _parse_jsonl_records(rejected_payload, accepted=False)
    if (len(accepted) != manifest["accepted_records_count"]
            or len(rejected) != manifest["rejected_records_count"]):
        raise BroadQaExternalDataError(
            "normalization development record count 漂移")
    (
        source_manifest,
        protocol_manifest,
        candidates,
        trials,
        training_item_ids,
    ) = validate_normalization_rule_records_v3(
        source_pack_dir=source_pack_dir,
        contrastive_protocol_dir=contrastive_protocol_dir,
        accepted_rules=accepted,
        rejected_trials=rejected,
    )
    chain = validate_normalization_development_checkpoint_chain_v1(
        chain_path=root / "checkpoints.jsonl",
        run_id=manifest["run_id"],
        protocol_sha=protocol_manifest["manifest_sha256"],
        training_item_ids=training_item_ids,
        candidates=candidates,
        trials=trials,
    )
    terminal = chain[-1]
    require_normalization_development_records_v1(
        source_manifest=source_manifest,
        protocol_manifest=protocol_manifest,
        candidates=candidates,
        trials=trials,
        accepted_rules=accepted,
        rejected_trials=rejected,
    )
    result_sha = normalization_rule_pack_v3_result_sha256(
        protocol_manifest_sha256=protocol_manifest["manifest_sha256"],
        training_item_ids=training_item_ids,
        accepted_rules=accepted,
        rejected_trials=rejected,
    )
    if (terminal.status != SOURCE_INFERENCE_LEARNING_CHECKPOINT_COMPLETE
            or terminal.evidence_candidate_count
            != sum(len(record.evidence_commitments)
                   for record in accepted + rejected)
            or terminal.rule_candidate_count != len(accepted) + len(rejected)
            or manifest["checkpoint_terminal_sha256"] != terminal.sha256()
            or manifest["source_pack_manifest_sha256"]
            != source_manifest["manifest_sha256"]
            or manifest["contrastive_protocol_manifest_sha256"]
            != protocol_manifest["manifest_sha256"]
            or manifest["training_item_count"] != len(training_item_ids)
            or manifest["training_item_order_sha256"]
            != source_inference_learning_prefix_sha256(training_item_ids)
            or manifest["result_sha256"] != result_sha):
        raise BroadQaExternalDataError(
            "normalization development source/result/checkpoint 漂移")
    return ({
        **manifest,
        "manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
    }, accepted, rejected)


def verify_normalization_development_fresh_resume_v1(
        *,
        fresh_run_dir: str | Path,
        resumed_run_dir: str | Path,
        source_pack_dir: str | Path,
        contrastive_protocol_dir: str | Path,
        ) -> dict[str, object]:
    """严格回读两次独立运行并要求逻辑 records 与结果逐字节等价。"""
    fresh_root = Path(fresh_run_dir).resolve()
    resumed_root = Path(resumed_run_dir).resolve()
    if fresh_root == resumed_root:
        raise BroadQaExternalDataError(
            "normalization development fresh/resume 目录必须独立")
    fresh_manifest, fresh_accepted, fresh_rejected = (
        read_normalization_development_learner_v1(
            fresh_root,
            source_pack_dir=source_pack_dir,
            contrastive_protocol_dir=contrastive_protocol_dir,
        ))
    resumed_manifest, resumed_accepted, resumed_rejected = (
        read_normalization_development_learner_v1(
            resumed_root,
            source_pack_dir=source_pack_dir,
            contrastive_protocol_dir=contrastive_protocol_dir,
        ))
    if (fresh_manifest["run_id"] == resumed_manifest["run_id"]
            or fresh_manifest["result_sha256"]
            != resumed_manifest["result_sha256"]
            or fresh_accepted != resumed_accepted
            or fresh_rejected != resumed_rejected
            or (fresh_root / "accepted-rules.jsonl").read_bytes()
            != (resumed_root / "accepted-rules.jsonl").read_bytes()
            or (fresh_root / "rejected-trials.jsonl").read_bytes()
            != (resumed_root / "rejected-trials.jsonl").read_bytes()):
        raise BroadQaExternalDataError(
            "normalization development fresh/resume 不等价")
    return {
        "accepted_records_count": len(fresh_accepted),
        "accepted_records_sha256": fresh_manifest[
            "accepted_records_sha256"],
        "fresh_manifest_sha256": fresh_manifest["manifest_sha256"],
        "fresh_run_id": fresh_manifest["run_id"],
        "record_byte_equivalence": 1,
        "rejected_records_count": len(fresh_rejected),
        "rejected_records_sha256": fresh_manifest[
            "rejected_records_sha256"],
        "result_sha256": fresh_manifest["result_sha256"],
        "resumed_manifest_sha256": resumed_manifest["manifest_sha256"],
        "resumed_run_id": resumed_manifest["run_id"],
    }


def main(argv: list[str] | None = None) -> int:
    """运行或严格回读 normalization development learner。"""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--run-root", required=True)
    run.add_argument("--source-pack-dir", required=True)
    run.add_argument("--contrastive-protocol-dir", required=True)
    run.add_argument("--run-dir", required=True)
    run.add_argument("--run-id", required=True)
    run.add_argument("--mode", choices=("fresh", "resume"), required=True)
    run.add_argument("--checkpoint-interval", type=int, default=512)
    run.add_argument("--stop-after", type=int)
    read = subparsers.add_parser("read")
    read.add_argument("--source-pack-dir", required=True)
    read.add_argument("--contrastive-protocol-dir", required=True)
    read.add_argument("--run-dir", required=True)
    arguments = parser.parse_args(argv)
    if arguments.command == "run":
        report = run_normalization_development_learner_v1(
            run_root=arguments.run_root,
            source_pack_dir=arguments.source_pack_dir,
            contrastive_protocol_dir=arguments.contrastive_protocol_dir,
            run_dir=arguments.run_dir,
            run_id=arguments.run_id,
            mode=arguments.mode,
            checkpoint_interval=arguments.checkpoint_interval,
            stop_after=arguments.stop_after,
        )
    else:
        report, _, _ = read_normalization_development_learner_v1(
            arguments.run_dir,
            source_pack_dir=arguments.source_pack_dir,
            contrastive_protocol_dir=arguments.contrastive_protocol_dir,
        )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "NORMALIZATION_DEVELOPMENT_CHECKPOINT_OPEN",
    "NORMALIZATION_DEVELOPMENT_DIRECTION",
    "NORMALIZATION_DEVELOPMENT_LEARNER_KIND",
    "NORMALIZATION_DEVELOPMENT_LEARNER_STATUS",
    "NORMALIZATION_DEVELOPMENT_OPERATOR",
    "NORMALIZATION_DEVELOPMENT_OPERATOR_VERSION",
    "NORMALIZATION_DEVELOPMENT_SCHEMA",
    "derive_normalization_development_records_v1",
    "read_normalization_development_learner_v1",
    "run_normalization_development_learner_v1",
    "validate_normalization_development_checkpoint_chain_v1",
    "verify_normalization_development_fresh_resume_v1",
]
