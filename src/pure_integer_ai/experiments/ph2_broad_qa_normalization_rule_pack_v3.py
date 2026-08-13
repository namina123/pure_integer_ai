"""normalization v3 rule pack 的跨记录校验、checkpoint 与发布边界。

accepted/rejected 值结构由 records 模块拥有；本模块从冻结 OpenCC source pack 和
contrastive TRAIN_SOURCE 重放证据，验证双运行等价后发布默认禁用的不可覆盖 pack。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pure_integer_ai.cognition.shared.hypothesis import (
    EPISTEMIC_REFUTED,
    EPISTEMIC_SUPPORTED,
    EvidenceRecord,
    HypothesisLedger,
)
from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_contrastive_protocol import (
    NORMALIZATION_CONTRASTIVE_FAMILY,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_evidence_v3 import (
    read_normalization_training_provenance,
    validate_normalization_evidence_commitment,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_rule_identity_v3 import (
    NORMALIZATION_CONTEXT_TRIAL_HYPOTHESIS_KIND,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_rule_records_v3 import (
    NORMALIZATION_ACCEPTED_RULE_V3_KIND,
    NORMALIZATION_CONTEXT_REJECTION_KIND,
    NORMALIZATION_REJECTED_TRIAL_V3_KIND,
    BroadQaNormalizationAcceptedRuleV3,
    BroadQaNormalizationRejectedTrialV3,
    BroadQaNormalizationRuleCandidateV3,
    parse_normalization_accepted_rule_v3,
    parse_normalization_rejected_trial_v3,
)
from pure_integer_ai.experiments.ph2_broad_qa_source_inference_learning_checkpoint import (
    SOURCE_INFERENCE_LEARNING_CHECKPOINT_COMPLETE,
    read_source_inference_learning_chain,
    source_inference_learning_prefix_sha256,
    source_inference_learning_result_sha256,
)
from pure_integer_ai.experiments.ph2_broad_qa_source_inference_rule_pack import (
    SOURCE_INFERENCE_RULE_RUNTIME_STATE,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


NORMALIZATION_RULE_PACK_V3_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RULE_PACK_V3")
NORMALIZATION_RULE_PACK_V3_STATUS = "FROZEN_NOT_EVALUATED_NOT_DEPLOYED"


def _sha256(value: object, *, label: str) -> str:
    """要求 pack 身份为小写 SHA-256。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise BroadQaExternalDataError(f"{label} 必须是 SHA-256")
    return value


def _record_payloads(
        accepted_rules: tuple[BroadQaNormalizationAcceptedRuleV3, ...],
        rejected_trials: tuple[BroadQaNormalizationRejectedTrialV3, ...],
        ) -> tuple[bytes, bytes]:
    """要求两类 record 分别按 SHA 唯一排序并返回规范字节。"""
    for label, records, record_type in (
            ("accepted", accepted_rules, BroadQaNormalizationAcceptedRuleV3),
            ("rejected", rejected_trials, BroadQaNormalizationRejectedTrialV3)):
        if (not isinstance(records, tuple) or not records
                or any(not isinstance(item, record_type) for item in records)):
            raise BroadQaExternalDataError(
                f"normalization {label} records 非法")
        shas = tuple(item.sha256() for item in records)
        if shas != tuple(sorted(set(shas))):
            raise BroadQaExternalDataError(
                f"normalization {label} records 必须按 SHA 唯一排序")
    return (
        b"".join(item.canonical_bytes() for item in accepted_rules),
        b"".join(item.canonical_bytes() for item in rejected_trials),
    )


def _validate_pack_records(
        *,
        accepted_rules: tuple[BroadQaNormalizationAcceptedRuleV3, ...],
        rejected_trials: tuple[BroadQaNormalizationRejectedTrialV3, ...],
        source_manifest: dict[str, object],
        protocol_manifest: dict[str, object],
        candidates: tuple[dict[str, object], ...],
        trials: tuple[dict[str, object], ...],
        ) -> None:
    """重放全部 Evidence、rejection/defeater 引用和核心 epistemic 状态。"""
    source_sha = source_manifest["manifest_sha256"]
    protocol_sha = protocol_manifest["manifest_sha256"]
    candidate_by_id = {item["candidate_id"]: item for item in candidates}
    trial_by_id = {item["trial_id"]: item for item in trials}
    if (len(candidate_by_id) != len(candidates)
            or len(trial_by_id) != len(trials)):
        raise BroadQaExternalDataError(
            "normalization training provenance identity 重复")

    accepted_by_candidate = {}
    rejected_by_sha = {item.sha256(): item for item in rejected_trials}
    evidence_ids = []
    ledger = HypothesisLedger()
    for rule in accepted_rules:
        candidate = candidate_by_id.get(rule.candidate.mapping_candidate_id)
        if (candidate is None
                or rule.candidate.contrastive_protocol_manifest_sha256
                != protocol_sha
                or rule.candidate.source_pack_manifest_sha256 != source_sha
                or candidate["input_codepoint"] != rule.candidate.input_codepoint
                or candidate["output_codepoint"]
                != rule.candidate.output_codepoint
                or rule.candidate.mapping_candidate_id in accepted_by_candidate):
            raise BroadQaExternalDataError(
                "normalization accepted candidate 来源/identity 漂移")
        accepted_by_candidate[rule.candidate.mapping_candidate_id] = rule
        hypothesis = rule.candidate.hypothesis()
        ledger.register(hypothesis)
        for commitment in rule.evidence_commitments:
            validate_normalization_evidence_commitment(
                commitment,
                protocol_manifest_sha256=protocol_sha,
                source_pack_manifest_sha256=source_sha,
                candidate_by_id=candidate_by_id,
                trial_by_id=trial_by_id,
                expected_hypothesis=hypothesis,
                expected_qualification="SOURCE_REPLAY_SUPPORT",
            )
            evidence = EvidenceRecord.from_stable_key(commitment.evidence_key)
            evidence_ids.append(evidence.evidence_id)
            ledger.append_evidence(evidence)
        if ledger.snapshot(hypothesis).epistemic_status != EPISTEMIC_SUPPORTED:
            raise BroadQaExternalDataError(
                "normalization accepted hypothesis 未形成 SUPPORTED")

    referenced_rejections = {
        sha for rule in accepted_rules
        for sha in rule.rejection_record_sha256s
    }
    if referenced_rejections != set(rejected_by_sha):
        raise BroadQaExternalDataError(
            "normalization rejection ledger 引用未精确闭合")
    for record_sha, rejected in rejected_by_sha.items():
        rule = accepted_by_candidate.get(
            rejected.candidate.mapping_candidate_id)
        if (rule is None or rejected.candidate != rule.candidate
                or record_sha not in rule.rejection_record_sha256s):
            raise BroadQaExternalDataError(
                "normalization rejected trial 未绑定 accepted candidate")
        hypothesis = rejected.candidate.trial_hypothesis(rejected.trial_id)
        if hypothesis == rejected.candidate.hypothesis():
            raise BroadQaExternalDataError(
                "normalization accepted/rejected hypothesis 未分离")
        ledger.register(hypothesis)
        for commitment in rejected.evidence_commitments:
            validate_normalization_evidence_commitment(
                commitment,
                protocol_manifest_sha256=protocol_sha,
                source_pack_manifest_sha256=source_sha,
                candidate_by_id=candidate_by_id,
                trial_by_id=trial_by_id,
                expected_hypothesis=hypothesis,
                expected_qualification="SOURCE_REPLAY_REFUTE",
            )
            evidence = EvidenceRecord.from_stable_key(commitment.evidence_key)
            evidence_ids.append(evidence.evidence_id)
            ledger.append_evidence(evidence)
        if ledger.snapshot(hypothesis).epistemic_status != EPISTEMIC_REFUTED:
            raise BroadQaExternalDataError(
                "normalization rejected hypothesis 未形成 REFUTED")
    for candidate_id, rule in accepted_by_candidate.items():
        recorded_defeaters = tuple(sorted(
            rejected.context_defeater.stable_key()
            for rejected in rejected_trials
            if rejected.candidate.mapping_candidate_id == candidate_id
        ))
        declared_defeaters = tuple(
            item.stable_key() for item in rule.candidate.defeaters)
        if declared_defeaters != recorded_defeaters:
            raise BroadQaExternalDataError(
                "normalization candidate defeater/rejection ledger 未精确闭合")
    if len(evidence_ids) != len(set(evidence_ids)):
        raise BroadQaExternalDataError(
            "normalization Evidence id 必须全局唯一")


def normalization_rule_pack_v3_result_sha256(
        *,
        protocol_manifest_sha256: str,
        training_item_ids: tuple[str, ...],
        accepted_rules: tuple[BroadQaNormalizationAcceptedRuleV3, ...],
        rejected_trials: tuple[BroadQaNormalizationRejectedTrialV3, ...],
        ) -> str:
    """从训练身份、records 和 Evidence 重算 normalization 唯一结果。"""
    records = accepted_rules + rejected_trials
    evidence_shas = tuple(sorted({
        item.sha256()
        for record in records
        for item in record.evidence_commitments
    }))
    record_shas = tuple(sorted(record.sha256() for record in records))
    return source_inference_learning_result_sha256(
        protocol_manifest_sha256=protocol_manifest_sha256,
        operator_family=NORMALIZATION_CONTRASTIVE_FAMILY,
        processed_item_ids=training_item_ids,
        evidence_record_sha256s=evidence_shas,
        rule_record_sha256s=record_shas,
    )


def validate_normalization_rule_records_v3(
        *,
        source_pack_dir: str | Path,
        contrastive_protocol_dir: str | Path,
        accepted_rules: tuple[BroadQaNormalizationAcceptedRuleV3, ...],
        rejected_trials: tuple[BroadQaNormalizationRejectedTrialV3, ...],
        ) -> tuple[
            dict[str, object],
            dict[str, object],
            tuple[dict[str, object], ...],
            tuple[dict[str, object], ...],
            tuple[str, ...],
        ]:
    """从冻结 OpenCC 来源重放并验证一组 v3 records。"""
    provenance = read_normalization_training_provenance(
        source_pack_dir=source_pack_dir,
        contrastive_protocol_dir=contrastive_protocol_dir,
    )
    source_manifest, protocol_manifest, candidates, trials, _ = provenance
    _record_payloads(accepted_rules, rejected_trials)
    _validate_pack_records(
        accepted_rules=accepted_rules,
        rejected_trials=rejected_trials,
        source_manifest=source_manifest,
        protocol_manifest=protocol_manifest,
        candidates=candidates,
        trials=trials,
    )
    return provenance


def _validate_checkpoint_chains(
        *,
        chain_paths: tuple[Path, Path],
        run_root: Path,
        protocol_manifest_sha256: str,
        training_item_ids: tuple[str, ...],
        output_evidence_count: int,
        output_candidate_count: int,
        maximum_evidence_count: int,
        maximum_candidate_count: int,
        ) -> tuple[dict[str, str], dict[str, str]]:
    """验证两条独立 COMPLETE 链覆盖完整 normalization TRAIN_SOURCE。"""
    if (chain_paths[0] == chain_paths[1]
            or any(not path.is_relative_to(run_root) for path in chain_paths)):
        raise BroadQaExternalDataError(
            "normalization checkpoint 必须独立且位于 run root")
    lineage = []
    for path in chain_paths:
        chain = read_source_inference_learning_chain(path)
        for checkpoint in chain:
            expected_prefix = source_inference_learning_prefix_sha256(
                training_item_ids[:checkpoint.processed_item_count])
            if checkpoint.processed_item_prefix_sha256 != expected_prefix:
                raise BroadQaExternalDataError(
                    "normalization checkpoint TRAIN prefix 漂移")
        checkpoint = chain[-1]
        if (checkpoint.status != SOURCE_INFERENCE_LEARNING_CHECKPOINT_COMPLETE
                or checkpoint.protocol_manifest_sha256
                != protocol_manifest_sha256
                or checkpoint.operator_family != NORMALIZATION_CONTRASTIVE_FAMILY
                or checkpoint.training_item_count != len(training_item_ids)
                or checkpoint.training_item_order_sha256
                != source_inference_learning_prefix_sha256(training_item_ids)
                or not output_evidence_count
                <= checkpoint.evidence_candidate_count
                <= maximum_evidence_count
                or not output_candidate_count
                <= checkpoint.rule_candidate_count
                <= maximum_candidate_count):
            raise BroadQaExternalDataError(
                "normalization checkpoint 未证明完整 TRAIN_SOURCE 或预算")
        lineage.append({
            "chain_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "run_id": checkpoint.run_id,
            "terminal_sha256": checkpoint.sha256(),
        })
    if lineage[0]["run_id"] == lineage[1]["run_id"]:
        raise BroadQaExternalDataError(
            "normalization fresh/resume run identity 必须独立")
    return lineage[0], lineage[1]


def publish_normalization_rule_pack_v3(
        *,
        source_pack_dir: str | Path,
        contrastive_protocol_dir: str | Path,
        fresh_accepted_rules: tuple[
            BroadQaNormalizationAcceptedRuleV3, ...],
        fresh_rejected_trials: tuple[
            BroadQaNormalizationRejectedTrialV3, ...],
        resumed_accepted_rules: tuple[
            BroadQaNormalizationAcceptedRuleV3, ...],
        resumed_rejected_trials: tuple[
            BroadQaNormalizationRejectedTrialV3, ...],
        target_dir: str | Path,
        fresh_checkpoint_chain_path: str | Path,
        resumed_checkpoint_chain_path: str | Path,
        ) -> dict[str, object]:
    """独立重放来源和双链后，不可覆盖发布禁用态 normalization v3 pack。"""
    source_root = Path(source_pack_dir).resolve()
    protocol_root = Path(contrastive_protocol_dir).resolve()
    run_root = protocol_root.parent
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
    fresh_payloads = _record_payloads(
        fresh_accepted_rules, fresh_rejected_trials)
    resumed_payloads = _record_payloads(
        resumed_accepted_rules, resumed_rejected_trials)
    if fresh_payloads != resumed_payloads:
        raise BroadQaExternalDataError(
            "normalization fresh/resume record 字节不等价")
    accepted_rules = fresh_accepted_rules
    rejected_trials = fresh_rejected_trials
    _validate_pack_records(
        accepted_rules=accepted_rules,
        rejected_trials=rejected_trials,
        source_manifest=source_manifest,
        protocol_manifest=protocol_manifest,
        candidates=candidates,
        trials=trials,
    )
    output_evidence_count = sum(
        len(record.evidence_commitments)
        for record in accepted_rules + rejected_trials)
    output_candidate_count = len(accepted_rules) + len(rejected_trials)
    lineages = _validate_checkpoint_chains(
        chain_paths=tuple(Path(path).resolve() for path in (
            fresh_checkpoint_chain_path, resumed_checkpoint_chain_path)),
        run_root=run_root,
        protocol_manifest_sha256=protocol_manifest["manifest_sha256"],
        training_item_ids=training_item_ids,
        output_evidence_count=output_evidence_count,
        output_candidate_count=output_candidate_count,
        maximum_evidence_count=len(trials),
        maximum_candidate_count=len(candidates) + len(trials),
    )
    result_sha = normalization_rule_pack_v3_result_sha256(
        protocol_manifest_sha256=protocol_manifest["manifest_sha256"],
        training_item_ids=training_item_ids,
        accepted_rules=accepted_rules,
        rejected_trials=rejected_trials,
    )
    resumed_result_sha = normalization_rule_pack_v3_result_sha256(
        protocol_manifest_sha256=protocol_manifest["manifest_sha256"],
        training_item_ids=training_item_ids,
        accepted_rules=resumed_accepted_rules,
        rejected_trials=resumed_rejected_trials,
    )
    if result_sha != resumed_result_sha:
        raise BroadQaExternalDataError(
            "normalization fresh/resume 结果重算不等价")

    target = Path(target_dir).resolve()
    if not target.is_relative_to(run_root):
        raise BroadQaExternalDataError(
            "normalization rule pack target 必须位于 run root")
    if target.exists():
        raise BroadQaExternalDataError(
            "normalization rule pack target 已存在")
    target.mkdir(parents=True)
    accepted_path = target / "accepted-rules.jsonl"
    rejected_path = target / "rejected-trials.jsonl"
    accepted_path.write_bytes(fresh_payloads[0])
    rejected_path.write_bytes(fresh_payloads[1])
    manifest = {
        "accepted_records_bytes": len(fresh_payloads[0]),
        "accepted_records_count": len(accepted_rules),
        "accepted_records_sha256": hashlib.sha256(
            fresh_payloads[0]).hexdigest(),
        "artifact_kind": NORMALIZATION_RULE_PACK_V3_KIND,
        "contrastive_protocol_manifest_sha256": (
            protocol_manifest["manifest_sha256"]),
        "format_version": 1,
        "fresh_checkpoint_chain_sha256": lineages[0]["chain_sha256"],
        "fresh_checkpoint_terminal_sha256": lineages[0]["terminal_sha256"],
        "fresh_result_sha256": result_sha,
        "fresh_run_id": lineages[0]["run_id"],
        "operator_family": NORMALIZATION_CONTRASTIVE_FAMILY,
        "production_enabled": 0,
        "rejected_records_bytes": len(fresh_payloads[1]),
        "rejected_records_count": len(rejected_trials),
        "rejected_records_sha256": hashlib.sha256(
            fresh_payloads[1]).hexdigest(),
        "resumed_checkpoint_chain_sha256": lineages[1]["chain_sha256"],
        "resumed_checkpoint_terminal_sha256": lineages[1]["terminal_sha256"],
        "resumed_result_sha256": resumed_result_sha,
        "resumed_run_id": lineages[1]["run_id"],
        "runtime_state": SOURCE_INFERENCE_RULE_RUNTIME_STATE,
        "source_pack_manifest_sha256": source_manifest["manifest_sha256"],
        "status": NORMALIZATION_RULE_PACK_V3_STATUS,
        "training_item_count": len(training_item_ids),
        "training_item_order_sha256": source_inference_learning_prefix_sha256(
            training_item_ids),
    }
    manifest_path = target / "manifest.json"
    manifest_path.write_bytes(canonical_json_line(manifest))
    return {
        **manifest,
        "manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()).hexdigest(),
    }


def read_normalization_rule_pack_v3(
        target_dir: str | Path,
        *,
        source_pack_dir: str | Path,
        contrastive_protocol_dir: str | Path,
        fresh_checkpoint_chain_path: str | Path,
        resumed_checkpoint_chain_path: str | Path,
        ) -> tuple[
            dict[str, object],
            tuple[BroadQaNormalizationAcceptedRuleV3, ...],
            tuple[BroadQaNormalizationRejectedTrialV3, ...],
        ]:
    """重验 OpenCC 来源、双链、records 和结果摘要后回读 v3 pack。"""
    root = Path(target_dir).resolve()
    try:
        manifest_payload = (root / "manifest.json").read_bytes()
        manifest = json.loads(manifest_payload)
        accepted_payload = (root / "accepted-rules.jsonl").read_bytes()
        rejected_payload = (root / "rejected-trials.jsonl").read_bytes()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadQaExternalDataError(
            "normalization rule pack 不可读") from error
    expected = {
        "accepted_records_bytes", "accepted_records_count",
        "accepted_records_sha256", "artifact_kind",
        "contrastive_protocol_manifest_sha256", "format_version",
        "fresh_checkpoint_chain_sha256",
        "fresh_checkpoint_terminal_sha256", "fresh_result_sha256",
        "fresh_run_id", "operator_family", "production_enabled",
        "rejected_records_bytes", "rejected_records_count",
        "rejected_records_sha256", "resumed_checkpoint_chain_sha256",
        "resumed_checkpoint_terminal_sha256", "resumed_result_sha256",
        "resumed_run_id", "runtime_state", "source_pack_manifest_sha256",
        "status", "training_item_count", "training_item_order_sha256",
    }
    if (not isinstance(manifest, dict) or set(manifest) != expected
            or canonical_json_line(manifest) != manifest_payload
            or manifest["artifact_kind"] != NORMALIZATION_RULE_PACK_V3_KIND
            or type(manifest["format_version"]) is not int
            or manifest["format_version"] != 1
            or manifest["operator_family"] != NORMALIZATION_CONTRASTIVE_FAMILY
            or type(manifest["production_enabled"]) is not int
            or manifest["production_enabled"] != 0
            or manifest["runtime_state"] != SOURCE_INFERENCE_RULE_RUNTIME_STATE
            or manifest["status"] != NORMALIZATION_RULE_PACK_V3_STATUS
            or manifest["fresh_run_id"] == manifest["resumed_run_id"]
            or manifest["fresh_result_sha256"]
            != manifest["resumed_result_sha256"]):
        raise BroadQaExternalDataError(
            "normalization rule pack manifest 漂移")
    for prefix, payload in (
            ("accepted", accepted_payload), ("rejected", rejected_payload)):
        if (type(manifest[f"{prefix}_records_count"]) is not int
                or manifest[f"{prefix}_records_count"] <= 0
                or type(manifest[f"{prefix}_records_bytes"]) is not int
                or manifest[f"{prefix}_records_bytes"] != len(payload)
                or manifest[f"{prefix}_records_sha256"]
                != hashlib.sha256(payload).hexdigest()
                or not payload.endswith(b"\n")):
            raise BroadQaExternalDataError(
                f"normalization {prefix} records commitment 漂移")
    for name in (
            "accepted_records_sha256",
            "contrastive_protocol_manifest_sha256",
            "fresh_checkpoint_chain_sha256",
            "fresh_checkpoint_terminal_sha256", "fresh_result_sha256",
            "fresh_run_id", "rejected_records_sha256",
            "resumed_checkpoint_chain_sha256",
            "resumed_checkpoint_terminal_sha256", "resumed_result_sha256",
            "resumed_run_id", "source_pack_manifest_sha256",
            "training_item_order_sha256"):
        _sha256(manifest[name], label=f"normalization manifest {name}")
    if (type(manifest["training_item_count"]) is not int
            or manifest["training_item_count"] <= 0):
        raise BroadQaExternalDataError(
            "normalization training item count 非法")
    accepted = tuple(
        parse_normalization_accepted_rule_v3(line + b"\n")
        for line in accepted_payload.splitlines()
    )
    rejected = tuple(
        parse_normalization_rejected_trial_v3(line + b"\n")
        for line in rejected_payload.splitlines()
    )
    if (len(accepted) != manifest["accepted_records_count"]
            or len(rejected) != manifest["rejected_records_count"]):
        raise BroadQaExternalDataError(
            "normalization rule pack record count 漂移")
    _record_payloads(accepted, rejected)
    (
        source_manifest,
        protocol_manifest,
        candidates,
        trials,
        training_item_ids,
    ) = read_normalization_training_provenance(
        source_pack_dir=source_pack_dir,
        contrastive_protocol_dir=contrastive_protocol_dir,
    )
    if (manifest["source_pack_manifest_sha256"]
            != source_manifest["manifest_sha256"]
            or manifest["contrastive_protocol_manifest_sha256"]
            != protocol_manifest["manifest_sha256"]):
        raise BroadQaExternalDataError(
            "normalization rule pack source/protocol commitment 漂移")
    _validate_pack_records(
        accepted_rules=accepted,
        rejected_trials=rejected,
        source_manifest=source_manifest,
        protocol_manifest=protocol_manifest,
        candidates=candidates,
        trials=trials,
    )
    output_evidence_count = sum(
        len(record.evidence_commitments)
        for record in accepted + rejected)
    output_candidate_count = len(accepted) + len(rejected)
    lineages = _validate_checkpoint_chains(
        chain_paths=tuple(Path(path).resolve() for path in (
            fresh_checkpoint_chain_path, resumed_checkpoint_chain_path)),
        run_root=Path(contrastive_protocol_dir).resolve().parent,
        protocol_manifest_sha256=protocol_manifest["manifest_sha256"],
        training_item_ids=training_item_ids,
        output_evidence_count=output_evidence_count,
        output_candidate_count=output_candidate_count,
        maximum_evidence_count=len(trials),
        maximum_candidate_count=len(candidates) + len(trials),
    )
    result_sha = normalization_rule_pack_v3_result_sha256(
        protocol_manifest_sha256=protocol_manifest["manifest_sha256"],
        training_item_ids=training_item_ids,
        accepted_rules=accepted,
        rejected_trials=rejected,
    )
    if (manifest["training_item_count"] != len(training_item_ids)
            or manifest["training_item_order_sha256"]
            != source_inference_learning_prefix_sha256(training_item_ids)
            or manifest["fresh_result_sha256"] != result_sha
            or manifest["resumed_result_sha256"] != result_sha
            or manifest["fresh_checkpoint_chain_sha256"]
            != lineages[0]["chain_sha256"]
            or manifest["fresh_checkpoint_terminal_sha256"]
            != lineages[0]["terminal_sha256"]
            or manifest["fresh_run_id"] != lineages[0]["run_id"]
            or manifest["resumed_checkpoint_chain_sha256"]
            != lineages[1]["chain_sha256"]
            or manifest["resumed_checkpoint_terminal_sha256"]
            != lineages[1]["terminal_sha256"]
            or manifest["resumed_run_id"] != lineages[1]["run_id"]):
        raise BroadQaExternalDataError(
            "normalization TRAIN/result/checkpoint commitment 漂移")
    return ({
        **manifest,
        "manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
    }, accepted, rejected)


__all__ = [
    "BroadQaNormalizationAcceptedRuleV3",
    "BroadQaNormalizationRejectedTrialV3",
    "BroadQaNormalizationRuleCandidateV3",
    "NORMALIZATION_ACCEPTED_RULE_V3_KIND",
    "NORMALIZATION_CONTEXT_REJECTION_KIND",
    "NORMALIZATION_CONTEXT_TRIAL_HYPOTHESIS_KIND",
    "NORMALIZATION_REJECTED_TRIAL_V3_KIND",
    "NORMALIZATION_RULE_PACK_V3_KIND",
    "NORMALIZATION_RULE_PACK_V3_STATUS",
    "normalization_rule_pack_v3_result_sha256",
    "parse_normalization_accepted_rule_v3",
    "parse_normalization_rejected_trial_v3",
    "publish_normalization_rule_pack_v3",
    "read_normalization_rule_pack_v3",
    "validate_normalization_rule_records_v3",
]
