"""normalization development learner 的纯确定性归纳核心。"""
from __future__ import annotations

import hashlib

from pure_integer_ai.cognition.shared.identity import (
    minimal_instruction_identity,
    structure_concept_identity,
)
from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_contrastive_protocol import (
    NORMALIZATION_CONTRASTIVE_APPLICATION_DOMAIN,
    NORMALIZATION_CONTRASTIVE_QUALIFICATIONS,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_evidence_v3 import (
    normalization_evidence_commitment_from_records,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_rule_identity_v3 import (
    normalization_context_defeater,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_rule_records_v3 import (
    BroadQaNormalizationAcceptedRuleV3,
    BroadQaNormalizationRejectedTrialV3,
    BroadQaNormalizationRuleCandidateV3,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes


NORMALIZATION_DEVELOPMENT_OPERATOR = minimal_instruction_identity((817045, 1))
NORMALIZATION_DEVELOPMENT_SCHEMA = structure_concept_identity((817046, 1))
NORMALIZATION_DEVELOPMENT_OPERATOR_VERSION = 1
NORMALIZATION_DEVELOPMENT_DIRECTION = "FORWARD"


def _sha256(value: object, *, label: str) -> str:
    """要求来源承诺为小写 SHA-256。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise BroadQaExternalDataError(f"{label} 必须是 SHA-256")
    return value


def normalization_development_qualification_groups(
        trials: tuple[dict[str, object], ...],
        ) -> dict[str, dict[str, tuple[dict[str, object], ...]]]:
    """按 candidate 和来源资格规范分组全部 context trial。"""
    staged: dict[str, dict[str, list[dict[str, object]]]] = {}
    for trial in trials:
        candidate_id = str(trial["candidate_id"])
        qualification = str(trial["qualification_kind"])
        staged.setdefault(candidate_id, {}).setdefault(
            qualification, []).append(trial)
    return {
        candidate_id: {
            qualification: tuple(sorted(
                records, key=lambda item: str(item["trial_id"])))
            for qualification, records in sorted(by_qualification.items())
        }
        for candidate_id, by_qualification in sorted(staged.items())
    }


def derive_normalization_development_records_v1(
        *,
        source_manifest: dict[str, object],
        protocol_manifest: dict[str, object],
        candidates: tuple[dict[str, object], ...],
        trials: tuple[dict[str, object], ...],
        ) -> tuple[
            tuple[BroadQaNormalizationAcceptedRuleV3, ...],
            tuple[BroadQaNormalizationRejectedTrialV3, ...],
        ]:
    """从双资格 mapping 归纳 SUPPORT rule 与独立 context rejection。"""
    source_sha = _sha256(
        source_manifest.get("manifest_sha256"),
        label="normalization development source manifest",
    )
    protocol_sha = _sha256(
        protocol_manifest.get("manifest_sha256"),
        label="normalization development protocol manifest",
    )
    groups = normalization_development_qualification_groups(trials)
    accepted = []
    rejected = []
    for source_candidate in candidates:
        candidate_id = str(source_candidate["candidate_id"])
        by_qualification = groups.get(candidate_id, {})
        if set(by_qualification) != set(NORMALIZATION_CONTRASTIVE_QUALIFICATIONS):
            continue
        support_trials = by_qualification["SOURCE_REPLAY_SUPPORT"]
        refute_trials = by_qualification["SOURCE_REPLAY_REFUTE"]
        defeaters = tuple(sorted(
            (normalization_context_defeater(str(trial["trial_id"]))
             for trial in refute_trials),
            key=lambda item: item.stable_key(),
        ))
        candidate = BroadQaNormalizationRuleCandidateV3(
            protocol_sha,
            source_sha,
            candidate_id,
            source_candidate["input_codepoint"],
            source_candidate["output_codepoint"],
            NORMALIZATION_DEVELOPMENT_OPERATOR,
            NORMALIZATION_DEVELOPMENT_OPERATOR_VERSION,
            NORMALIZATION_DEVELOPMENT_SCHEMA,
            NORMALIZATION_DEVELOPMENT_DIRECTION,
            NORMALIZATION_CONTRASTIVE_APPLICATION_DOMAIN,
            defeaters,
        )
        support = tuple(sorted((
            normalization_evidence_commitment_from_records(
                contrastive_protocol_manifest_sha256=protocol_sha,
                source_pack_manifest_sha256=source_sha,
                candidate=source_candidate,
                trial=trial,
                hypothesis=candidate.hypothesis(),
            )
            for trial in support_trials
        ), key=lambda item: item.evidence_key))
        candidate_parameters_sha = hashlib.sha256(canonical_json_bytes(
            candidate.to_dict())).hexdigest()
        candidate_rejections = tuple(sorted((
            BroadQaNormalizationRejectedTrialV3(
                candidate,
                str(trial["trial_id"]),
                normalization_context_defeater(str(trial["trial_id"])),
                candidate_parameters_sha,
                (normalization_evidence_commitment_from_records(
                    contrastive_protocol_manifest_sha256=protocol_sha,
                    source_pack_manifest_sha256=source_sha,
                    candidate=source_candidate,
                    trial=trial,
                    hypothesis=candidate.trial_hypothesis(
                        str(trial["trial_id"])),
                ),),
            )
            for trial in refute_trials
        ), key=lambda item: item.sha256()))
        rejected.extend(candidate_rejections)
        accepted.append(BroadQaNormalizationAcceptedRuleV3(
            candidate,
            support,
            tuple(item.sha256() for item in candidate_rejections),
        ))
    accepted_records = tuple(sorted(accepted, key=lambda item: item.sha256()))
    rejected_records = tuple(sorted(rejected, key=lambda item: item.sha256()))
    if not accepted_records or not rejected_records:
        raise BroadQaExternalDataError(
            "normalization development 双资格来源不足")
    return accepted_records, rejected_records


def normalization_development_output_counts_for_prefix(
        *,
        candidates: tuple[dict[str, object], ...],
        trials: tuple[dict[str, object], ...],
        processed_item_count: int,
        ) -> tuple[int, int]:
    """计算当前有序 TRAIN 前缀已具备的 Evidence 与 record 候选数。"""
    processed_trial_count = max(0, processed_item_count - len(candidates))
    processed_trials = trials[:processed_trial_count]
    groups = normalization_development_qualification_groups(processed_trials)
    qualified = {
        candidate_id for candidate_id, by_qualification in groups.items()
        if set(by_qualification) == set(NORMALIZATION_CONTRASTIVE_QUALIFICATIONS)
    }
    evidence_count = sum(
        str(trial["candidate_id"]) in qualified for trial in processed_trials)
    rejected_count = sum(
        str(trial["candidate_id"]) in qualified
        and trial["qualification_kind"] == "SOURCE_REPLAY_REFUTE"
        for trial in processed_trials
    )
    return evidence_count, len(qualified) + rejected_count


def require_normalization_development_records_v1(
        *,
        source_manifest: dict[str, object],
        protocol_manifest: dict[str, object],
        candidates: tuple[dict[str, object], ...],
        trials: tuple[dict[str, object], ...],
        accepted_rules: tuple[BroadQaNormalizationAcceptedRuleV3, ...],
        rejected_trials: tuple[BroadQaNormalizationRejectedTrialV3, ...],
        ) -> None:
    """要求 records 恰好等于完整 TRAIN_SOURCE 的确定性归纳输出。"""
    expected_accepted, expected_rejected = (
        derive_normalization_development_records_v1(
            source_manifest=source_manifest,
            protocol_manifest=protocol_manifest,
            candidates=candidates,
            trials=trials,
        ))
    if (accepted_rules != expected_accepted
            or rejected_trials != expected_rejected):
        raise BroadQaExternalDataError(
            "normalization development records 不是完整确定性输出")


__all__ = [
    "NORMALIZATION_DEVELOPMENT_DIRECTION",
    "NORMALIZATION_DEVELOPMENT_OPERATOR",
    "NORMALIZATION_DEVELOPMENT_OPERATOR_VERSION",
    "NORMALIZATION_DEVELOPMENT_SCHEMA",
    "derive_normalization_development_records_v1",
    "normalization_development_output_counts_for_prefix",
    "normalization_development_qualification_groups",
    "require_normalization_development_records_v1",
]
