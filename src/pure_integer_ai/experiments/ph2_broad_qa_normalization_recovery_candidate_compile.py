"""从 recovery disabled pack 编译显式 transfer candidate program。"""
from __future__ import annotations

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_candidate_records import (
    NormalizationRecoveryCandidateProgram,
    NormalizationRecoveryConflict,
    NormalizationRecoveryPhraseOverride,
    NormalizationRecoverySourceReplay,
    NormalizationRecoveryTargetRule,
    NormalizationRecoveryTransferProfile,
    RECOVERY_SOURCE_PRECEDENCE,
    RECOVERY_TARGET_PRECEDENCE,
    RECOVERY_TRANSFER_REGION_SCOPE,
    require_sha256,
    require_text,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_evaluation_protocol import (
    NORMALIZATION_RECOVERY_EVALUATION_STATUS,
    NORMALIZATION_RECOVERY_TARGET_POLICY_SCOPE,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_learning_records import (
    NORMALIZATION_RECOVERY_COMPOSITION_RECEIPT_KIND,
    NORMALIZATION_RECOVERY_CONFLICT_KIND,
    NORMALIZATION_RECOVERY_EVIDENCE_KIND,
    NORMALIZATION_RECOVERY_GENERIC_RULE_KIND,
    NORMALIZATION_RECOVERY_GROUP_DECISION_KIND,
    NORMALIZATION_RECOVERY_OUTPUT_FILE_ROLES,
    NORMALIZATION_RECOVERY_PHRASE_RULE_KIND,
    NORMALIZATION_RECOVERY_REGIONAL_RULE_KIND,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_training_records import (
    RECOVERY_TARGET_POLICY_SCOPE,
    SOURCE_POLICY_SCOPES,
)


def _target_rule(
        record: dict[str, object],
        *,
        regional: bool,
        ) -> NormalizationRecoveryTargetRule:
    """把严格回读的 learned record 编译为 target rule。"""
    expected_kind = (
        NORMALIZATION_RECOVERY_REGIONAL_RULE_KIND
        if regional else NORMALIZATION_RECOVERY_GENERIC_RULE_KIND)
    domain = record.get("application_domain")
    if (record.get("record_kind") != expected_kind
            or record.get("runtime_state") != "LEARNED_PACK_DISABLED"
            or record.get("production_enabled") != 0
            or not isinstance(domain, dict)
            or record.get("target_policy_scope")
            != RECOVERY_TARGET_POLICY_SCOPE):
        raise BroadQaExternalDataError(
            "recovery candidate target record 边界漂移")
    input_text = require_text(record.get("input_text"), label="target input")
    if regional:
        if (domain != {
                "exact_input": input_text,
                "normalization_policy_scope": RECOVERY_TARGET_POLICY_SCOPE,
                "regional_scope": RECOVERY_TRANSFER_REGION_SCOPE,
                }
                or record.get("global_upgrade_allowed") != 0):
            raise BroadQaExternalDataError(
                "recovery candidate regional domain 漂移")
    elif domain != {
            "input_match": "EXACT_SCALAR_SEQUENCE",
            "normalization_policy_scope": RECOVERY_TARGET_POLICY_SCOPE,
            }:
        raise BroadQaExternalDataError(
            "recovery candidate generic domain 漂移")
    return NormalizationRecoveryTargetRule(
        input_text=input_text,
        output_text=require_text(
            record.get("output_text"), label="target output"),
        rule_id=require_sha256(
            record.get("rule_id"), label="target rule id"),
        mapping_kind=require_text(
            record.get("mapping_kind"), label="target mapping kind"),
        rule_scope="REGIONAL_ZH_CN" if regional else "GENERIC",
        authority_policy_scope=RECOVERY_TARGET_POLICY_SCOPE,
        regional_scope=RECOVERY_TRANSFER_REGION_SCOPE if regional else "",
    )


def _conflict_observations(record: dict[str, object]) -> tuple[str, ...]:
    """从 family outputs 提取完整 generic observation identity。"""
    values = record.get("family_outputs")
    if not isinstance(values, list) or not values:
        raise BroadQaExternalDataError(
            "recovery candidate conflict family outputs 漂移")
    identities = []
    for family in values:
        observations = (
            None if not isinstance(family, dict)
            else family.get("observation_ids"))
        if not isinstance(observations, list) or not observations:
            raise BroadQaExternalDataError(
                "recovery candidate conflict observations 漂移")
        identities.extend(
            require_sha256(value, label="conflict observation")
            for value in observations)
    if len(set(identities)) != len(identities):
        raise BroadQaExternalDataError(
            "recovery candidate conflict observation 重复")
    return tuple(sorted(identities))


def _support_replay_material(
        evidence: tuple[dict[str, object], ...],
        ) -> tuple[
            dict[str, dict[str, object]],
            dict[str, dict[str, object]],
        ]:
    """分离 source mapping SUPPORT 与 composition REFUTE Evidence。"""
    supports = {}
    refutes = {}
    for record in evidence:
        if record.get("record_kind") != NORMALIZATION_RECOVERY_EVIDENCE_KIND:
            raise BroadQaExternalDataError(
                "recovery candidate evidence record kind 漂移")
        evidence_id = require_sha256(
            record.get("evidence_id"), label="recovery evidence id")
        hypothesis = record.get("hypothesis_kind")
        stance = record.get("stance")
        if hypothesis == "SOURCE_POLICY_MAPPING" and stance == "SUPPORT":
            observation_id = require_sha256(
                record.get("observation_id"), label="support observation")
            if observation_id in supports:
                raise BroadQaExternalDataError(
                    "recovery candidate support observation 重复")
            supports[observation_id] = record
        elif (hypothesis
              == "TARGET_CHARACTER_COMPOSITION_UNDER_SOURCE_POLICY"
              and stance == "REFUTE"):
            if evidence_id in refutes:
                raise BroadQaExternalDataError(
                    "recovery candidate refute Evidence 重复")
            refutes[evidence_id] = record
        else:
            raise BroadQaExternalDataError(
                "recovery candidate Evidence hypothesis/stance 漂移")
    if not supports or not refutes:
        raise BroadQaExternalDataError(
            "recovery candidate Evidence inventory 不完整")
    return supports, refutes


def _compile_conflicts(
        records: tuple[dict[str, object], ...],
        supports: dict[str, dict[str, object]],
        ) -> tuple[
            tuple[NormalizationRecoveryConflict, ...],
            dict[str, list[str]],
        ]:
    """编译 conflict，并建立 observation 到 conflict 的反向引用。"""
    conflicts = []
    by_observation: dict[str, list[str]] = {}
    for record in records:
        if (record.get("record_kind") != NORMALIZATION_RECOVERY_CONFLICT_KIND
                or record.get("target_policy_scope")
                != RECOVERY_TARGET_POLICY_SCOPE
                or record.get("unscoped_application_allowed") != 0
                or record.get("production_enabled") != 0):
            raise BroadQaExternalDataError(
                "recovery candidate conflict record 边界漂移")
        observation_ids = _conflict_observations(record)
        conflict = NormalizationRecoveryConflict(
            input_text=require_text(
                record.get("input_text"), label="conflict input"),
            conflict_id=require_sha256(
                record.get("conflict_id"), label="conflict id"),
            conflict_kind=require_text(
                record.get("conflict_kind"), label="conflict kind"),
            observation_ids=observation_ids,
        )
        conflicts.append(conflict)
        for observation_id in observation_ids:
            if observation_id not in supports:
                raise BroadQaExternalDataError(
                    "recovery conflict/source Evidence 未闭合")
            by_observation.setdefault(
                observation_id, []).append(conflict.conflict_id)
    return (
        tuple(sorted(conflicts, key=lambda item: item.input_text)),
        by_observation,
    )


def _compile_source_replays(
        supports: dict[str, dict[str, object]],
        conflicts_by_observation: dict[str, list[str]],
        ) -> tuple[NormalizationRecoverySourceReplay, ...]:
    """把所有 source mapping SUPPORT 编译为 exact replay。"""
    values = []
    for observation_id, record in supports.items():
        policy = require_text(
            record.get("source_policy_scope"), label="source replay policy")
        if policy not in SOURCE_POLICY_SCOPES:
            raise BroadQaExternalDataError(
                "recovery source replay policy 非法")
        values.append(NormalizationRecoverySourceReplay(
            source_policy_scope=policy,
            input_text=require_text(
                record.get("input_text"), label="source replay input"),
            output_text=require_text(
                record.get("hypothesis_output"), label="source replay output"),
            evidence_id=require_sha256(
                record.get("evidence_id"), label="source replay evidence"),
            observation_id=observation_id,
            authority_role=require_text(
                record.get("authority_role"), label="source authority role"),
            conflict_ids=tuple(sorted(
                conflicts_by_observation.get(observation_id, ()))),
        ))
    result = tuple(sorted(
        values, key=lambda item: (item.source_policy_scope, item.input_text)))
    replay_by_key = {
        (item.source_policy_scope, item.input_text): item for item in result}
    if (len(replay_by_key) != len(result)
            or {item.source_policy_scope for item in result}
            != set(SOURCE_POLICY_SCOPES)):
        raise BroadQaExternalDataError(
            "recovery source replay key/policy inventory 漂移")
    return result


def _compile_phrase_overrides(
        records: tuple[dict[str, object], ...],
        source_replays: tuple[NormalizationRecoverySourceReplay, ...],
        refutes: dict[str, dict[str, object]],
        receipts: dict[str, dict[str, object]],
        ) -> tuple[NormalizationRecoveryPhraseOverride, ...]:
    """编译 source phrase override 并闭合 support/refute/replay。"""
    replay_by_key = {
        (item.source_policy_scope, item.input_text): item
        for item in source_replays}
    values = []
    for record in records:
        domain = record.get("application_domain")
        policy = require_text(
            record.get("source_policy_scope"), label="phrase policy")
        input_text = require_text(
            record.get("input_text"), label="phrase input")
        if (record.get("record_kind") != NORMALIZATION_RECOVERY_PHRASE_RULE_KIND
                or record.get("runtime_state") != "LEARNED_PACK_DISABLED"
                or record.get("production_enabled") != 0
                or record.get("target_policy_scope") != ""
                or domain != {
                    "exact_input": input_text,
                    "source_policy_scope": policy,
                }):
            raise BroadQaExternalDataError(
                "recovery candidate phrase rule 边界漂移")
        override = NormalizationRecoveryPhraseOverride(
            source_policy_scope=policy,
            input_text=input_text,
            base_output=require_text(
                receipts.get(str(record.get("rule_id")), {}).get("base_output"),
                label="phrase base output"),
            output_text=require_text(
                record.get("output_text"), label="phrase output"),
            rule_id=require_sha256(
                record.get("rule_id"), label="phrase rule id"),
            support_evidence_id=require_sha256(
                record.get("support_evidence_id"),
                label="phrase support evidence"),
            refute_evidence_id=require_sha256(
                record.get("refute_evidence_id"),
                label="phrase refute evidence"),
        )
        replay = replay_by_key.get((policy, input_text))
        refute = refutes.get(override.refute_evidence_id)
        if (replay is None or replay.output_text != override.output_text
                or replay.evidence_id != override.support_evidence_id
                or refute is None
                or refute.get("input_text") != input_text
                or refute.get("source_policy_scope") != policy
                or refute.get("observation_id") != replay.observation_id):
            raise BroadQaExternalDataError(
                "recovery phrase override/Evidence/replay 未闭合")
        values.append(override)
    return tuple(sorted(
        values, key=lambda item: (item.source_policy_scope, item.input_text)))


def _phrase_receipts(
        records: tuple[dict[str, object], ...],
        ) -> dict[str, dict[str, object]]:
    """索引显式 override receipt，并拒绝错 qualification 或重复引用。"""
    result = {}
    for record in records:
        if (record.get("record_kind")
                != NORMALIZATION_RECOVERY_COMPOSITION_RECEIPT_KIND):
            raise BroadQaExternalDataError(
                "recovery candidate composition receipt kind 漂移")
        phrase_rule_id = require_text(
            record.get("phrase_rule_id"),
            label="receipt phrase rule",
            empty=True,
        )
        if not phrase_rule_id:
            continue
        rule_id = require_sha256(
            phrase_rule_id, label="receipt phrase rule id")
        if (record.get("qualification_kind") != "EXPLICIT_OVERRIDE"
                or rule_id in result):
            raise BroadQaExternalDataError(
                "recovery candidate receipt override qualification/identity 漂移")
        result[rule_id] = record
    if not result:
        raise BroadQaExternalDataError(
            "recovery candidate phrase receipt 为空")
    return result


def _validate_decisions(
        *,
        outputs: dict[str, tuple[dict[str, object], ...]],
        target_rule_ids: set[str],
        conflict_ids: set[str],
        ) -> None:
    """要求 group decision 完整引用 target rule 与 conflict。"""
    decision_rule_ids = set()
    decision_conflict_ids = set()
    for record in outputs["group-decisions.jsonl"]:
        if (record.get("record_kind")
                != NORMALIZATION_RECOVERY_GROUP_DECISION_KIND
                or record.get("production_enabled") != 0):
            raise BroadQaExternalDataError(
                "recovery candidate group decision 边界漂移")
        rule_id = require_text(
            record.get("rule_id"), label="decision rule", empty=True)
        conflict_id = require_text(
            record.get("conflict_id"), label="decision conflict", empty=True)
        if rule_id:
            decision_rule_ids.add(require_sha256(
                rule_id, label="decision rule id"))
        if conflict_id:
            decision_conflict_ids.add(require_sha256(
                conflict_id, label="decision conflict id"))
    if (decision_rule_ids != target_rule_ids
            or decision_conflict_ids != conflict_ids):
        raise BroadQaExternalDataError(
            "recovery candidate decision/rule/conflict 未闭合")


def compile_normalization_recovery_candidate(
        *,
        evaluation_protocol_manifest: dict[str, object],
        rule_pack_manifest: dict[str, object],
        outputs: dict[str, tuple[dict[str, object], ...]],
        ) -> NormalizationRecoveryCandidateProgram:
    """从 pack 与 evaluation manifest-only identity 编译 transfer candidate。"""
    pack_sha = require_sha256(
        rule_pack_manifest.get("manifest_sha256"),
        label="recovery candidate pack manifest")
    evaluation_sha = require_sha256(
        evaluation_protocol_manifest.get("manifest_sha256"),
        label="recovery candidate evaluation manifest")
    if (rule_pack_manifest.get("runtime_state") != "LEARNED_PACK_DISABLED"
            or rule_pack_manifest.get("production_enabled") != 0
            or rule_pack_manifest.get("mastery_claimed") != 0
            or evaluation_protocol_manifest.get("status")
            != NORMALIZATION_RECOVERY_EVALUATION_STATUS
            or evaluation_protocol_manifest.get("target_policy_scope")
            != NORMALIZATION_RECOVERY_TARGET_POLICY_SCOPE):
        raise BroadQaExternalDataError(
            "recovery candidate pack/evaluation 边界漂移")
    expected_names = {
        name for name, _role, _identity
        in NORMALIZATION_RECOVERY_OUTPUT_FILE_ROLES}
    if not isinstance(outputs, dict) or set(outputs) != expected_names:
        raise BroadQaExternalDataError(
            "recovery candidate output inventory 漂移")

    generic_rules = tuple(sorted(
        (_target_rule(record, regional=False)
         for record in outputs["generic-rules.jsonl"]),
        key=lambda item: item.input_text))
    regional_rules = tuple(sorted(
        (_target_rule(record, regional=True)
         for record in outputs["regional-rules.jsonl"]),
        key=lambda item: item.input_text))
    supports, refutes = _support_replay_material(outputs["evidence.jsonl"])
    conflicts, conflicts_by_observation = _compile_conflicts(
        outputs["conflict-ledger.jsonl"], supports)
    source_replays = _compile_source_replays(
        supports, conflicts_by_observation)
    receipts = _phrase_receipts(outputs["composition-receipts.jsonl"])
    phrase_overrides = _compile_phrase_overrides(
        outputs["source-phrase-rules.jsonl"], source_replays, refutes, receipts)
    if set(receipts) != {item.rule_id for item in phrase_overrides}:
        raise BroadQaExternalDataError(
            "recovery candidate receipt/phrase rule 未闭合")
    _validate_decisions(
        outputs=outputs,
        target_rule_ids={
            item.rule_id for item in generic_rules + regional_rules},
        conflict_ids={item.conflict_id for item in conflicts},
    )
    profile = NormalizationRecoveryTransferProfile(
        rule_pack_manifest_sha256=pack_sha,
        evaluation_protocol_manifest_sha256=evaluation_sha,
        authority_policy_scope=RECOVERY_TARGET_POLICY_SCOPE,
        candidate_target_policy_scope=(
            NORMALIZATION_RECOVERY_TARGET_POLICY_SCOPE),
        regional_scope=RECOVERY_TRANSFER_REGION_SCOPE,
        target_precedence=RECOVERY_TARGET_PRECEDENCE,
        source_precedence=RECOVERY_SOURCE_PRECEDENCE,
    )
    return NormalizationRecoveryCandidateProgram(
        transfer_profile=profile,
        generic_rules=generic_rules,
        regional_rules=regional_rules,
        source_replays=source_replays,
        phrase_overrides=phrase_overrides,
        conflicts=conflicts,
        production_enabled=0,
    )


__all__ = ["compile_normalization_recovery_candidate"]
