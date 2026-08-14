"""normalization recovery 六维 evaluator 的独立 synthetic 测试。"""
from __future__ import annotations

import hashlib

from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_recovery_evaluation_protocol as protocol_module,
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
    RECOVERY_TRANSFER_PROFILE_KIND,
    RECOVERY_TRANSFER_REGION_SCOPE,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_evaluation_protocol import (
    NORMALIZATION_RECOVERY_EVALUATION_RECORD_KIND,
    NORMALIZATION_RECOVERY_TARGET_POLICY_SCOPE,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_evaluator import (
    NORMALIZATION_RECOVERY_DIMENSION_ORDER,
    evaluate_normalization_recovery_candidate,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_learning_records import (
    NORMALIZATION_RECOVERY_CONFLICT_KIND,
    NORMALIZATION_RECOVERY_OUTPUT_FILE_ROLES,
    NORMALIZATION_RECOVERY_PHRASE_RULE_KIND,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_source_pack import (
    FIREFOX_L10N_COMMIT,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_training_records import (
    GENERIC_T2S_AUTHORITY,
    RECOVERY_TARGET_POLICY_SCOPE,
    SOURCE_POLICY_SCOPES,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


def _id(*values: str) -> str:
    """从 synthetic 稳定字段构造 SHA-256 identity。"""
    return hashlib.sha256("\0".join(values).encode()).hexdigest()


def _record(
        *,
        ordinal: int,
        input_text: str,
        expected_output: str,
        family_keys: list[str],
        context_sensitive: int = 0,
        identity_preservation: int = 0,
        mapping_offset: int | None = None,
        ) -> dict[str, object]:
    """构造符合 recovery v2 schema 的 evaluation record。"""
    value = {
        "content_cluster_id": _id("cluster", str(ordinal)),
        "context_sensitive": context_sensitive,
        "evaluation_id": _id("evaluation", str(ordinal)),
        "expected_output": expected_output,
        "family_keys": family_keys,
        "format_version": 2,
        "identity_preservation": identity_preservation,
        "input_scalar_count": len(input_text),
        "input_text": input_text,
        "output_scalar_count": len(expected_output),
        "record_kind": NORMALIZATION_RECOVERY_EVALUATION_RECORD_KIND,
        "source_commit": FIREFOX_L10N_COMMIT,
        "source_occurrence_count": 1,
        "source_pack_manifest_sha256": _id("evaluation-source"),
        "source_pair_id": _id("pair", str(ordinal)),
        "source_policy_scope": "MOZILLA_FIREFOX_L10N_ZH_TW_TO_ZH_CN",
        "split": "EVALUATION",
        "split_group_sha256": _id("split", str(ordinal)),
        "target_policy_scope": NORMALIZATION_RECOVERY_TARGET_POLICY_SCOPE,
    }
    if mapping_offset is not None:
        value.update({
            "mapping_expected_character": expected_output[mapping_offset],
            "mapping_input_character": input_text[mapping_offset],
            "mapping_offset": mapping_offset,
        })
    return value


def _synthetic_family(
        *,
        wrong_local: bool = False,
        ) -> tuple[
            dict[str, object],
            tuple[dict[str, object], ...],
            dict[str, object],
            dict[str, tuple[dict[str, object], ...]],
            NormalizationRecoveryCandidateProgram,
        ]:
    """构造达到六维正式阈值的禁用态 synthetic family。"""
    evaluation_sha = _id("evaluation-protocol")
    training_sha = _id("training-protocol")
    pack_sha = _id("rule-pack")
    source_sha = _id("evaluation-source")
    input_characters = tuple(chr(0x4E00 + index) for index in range(320))
    output_characters = tuple(chr(0x6000 + index) for index in range(320))
    generic_rules = []
    records = []
    ordinal = 0
    for index, (input_text, expected) in enumerate(zip(
            input_characters, output_characters)):
        output = chr(0x7600) if wrong_local and index == 0 else expected
        generic_rules.append(NormalizationRecoveryTargetRule(
            input_text=input_text,
            output_text=output,
            rule_id=_id("character-rule", str(index), output),
            mapping_kind="CHARACTER_INPUT",
            rule_scope="GENERIC",
            authority_policy_scope=RECOVERY_TARGET_POLICY_SCOPE,
        ))
        records.append(_record(
            ordinal=ordinal,
            input_text=input_text,
            expected_output=expected,
            family_keys=["LOCAL_MAPPING_TRANSFER"],
            mapping_offset=0,
        ))
        ordinal += 1

    context_character = chr(0x8500)
    for index in range(64):
        prefix = chr(0x8600 + index)
        expected_context = prefix + chr(0x9000 + index)
        input_text = prefix + context_character
        generic_rules.append(NormalizationRecoveryTargetRule(
            input_text=input_text,
            output_text=expected_context,
            rule_id=_id("context-rule", str(index)),
            mapping_kind="PHRASE_INPUT",
            rule_scope="GENERIC",
            authority_policy_scope=RECOVERY_TARGET_POLICY_SCOPE,
        ))
        records.append(_record(
            ordinal=ordinal,
            input_text=input_text,
            expected_output=expected_context,
            family_keys=[
                "END_TO_END_COVERAGE", "INDEPENDENT_CONTEXT_TRANSFER"],
            context_sensitive=1,
            mapping_offset=1,
        ))
        ordinal += 1

    for index in range(64, 4_096):
        left = index // 320
        right = index % 320
        input_text = input_characters[left] + input_characters[right]
        expected = output_characters[left] + output_characters[right]
        records.append(_record(
            ordinal=ordinal,
            input_text=input_text,
            expected_output=expected,
            family_keys=["END_TO_END_COVERAGE"],
        ))
        ordinal += 1

    for index in range(192):
        input_text = chr(0xA000 + index) + chr(0xA100 + index)
        records.append(_record(
            ordinal=ordinal,
            input_text=input_text,
            expected_output=input_text,
            family_keys=["END_TO_END_COVERAGE"],
            identity_preservation=1,
        ))
        ordinal += 1
    records = tuple(sorted(records, key=lambda item: item["evaluation_id"]))
    payload = b"".join(canonical_json_line(item) for item in records)
    protocol = protocol_module._manifest(
        source_pack_manifest_sha256=source_sha,
        evaluation_artifact={
            "bytes": len(payload),
            "record_count": len(records),
            "relative_path": "evaluation.inventory.jsonl",
            "role": "EVALUATION_WITH_LABELS",
            "sha256": hashlib.sha256(payload).hexdigest(),
        },
        reserve_artifact={
            "bytes": 1,
            "record_count": 1,
            "relative_path": "reserve.identity.jsonl",
            "role": "RESERVE_IDENTITY_WITHOUT_LABELS",
            "sha256": _id("reserve"),
        },
        inventory_summary={
            "evaluation_count": len(records),
            "reserve_count": 1,
        },
    )
    protocol["manifest_sha256"] = evaluation_sha

    conflict_input = "乾乾"
    observation_ids = tuple(
        _id("conflict-observation", policy) for policy in SOURCE_POLICY_SCOPES[:4])
    conflict_id = _id("conflict")
    replay_outputs = (
        "乾乾", "干干", "乾干", "干乾",
    )
    source_replays = tuple(sorted((
        NormalizationRecoverySourceReplay(
            source_policy_scope=policy,
            input_text=conflict_input,
            output_text=output,
            evidence_id=_id("evidence", observation_id),
            observation_id=observation_id,
            authority_role=GENERIC_T2S_AUTHORITY,
            conflict_ids=(conflict_id,),
        )
        for policy, observation_id, output in zip(
            SOURCE_POLICY_SCOPES[:4], observation_ids, replay_outputs)
    ), key=lambda item: (item.source_policy_scope, item.input_text)))
    phrase_policy = SOURCE_POLICY_SCOPES[0]
    phrase_replay = next(
        item for item in source_replays
        if item.source_policy_scope == phrase_policy)
    phrase_rule = NormalizationRecoveryPhraseOverride(
        source_policy_scope=phrase_policy,
        input_text=conflict_input,
        base_output="干干",
        output_text=phrase_replay.output_text,
        rule_id=_id("phrase-rule"),
        support_evidence_id=phrase_replay.evidence_id,
        refute_evidence_id=_id("phrase-refute"),
    )
    conflict = NormalizationRecoveryConflict(
        input_text=conflict_input,
        conflict_id=conflict_id,
        conflict_kind="INTRA_FAMILY_CONFLICT",
        observation_ids=tuple(sorted(observation_ids)),
    )
    profile = NormalizationRecoveryTransferProfile(
        rule_pack_manifest_sha256=pack_sha,
        evaluation_protocol_manifest_sha256=evaluation_sha,
        authority_policy_scope=RECOVERY_TARGET_POLICY_SCOPE,
        candidate_target_policy_scope=NORMALIZATION_RECOVERY_TARGET_POLICY_SCOPE,
        regional_scope=RECOVERY_TRANSFER_REGION_SCOPE,
        target_precedence=RECOVERY_TARGET_PRECEDENCE,
        source_precedence=RECOVERY_SOURCE_PRECEDENCE,
        profile_kind=RECOVERY_TRANSFER_PROFILE_KIND,
    )
    program = NormalizationRecoveryCandidateProgram(
        transfer_profile=profile,
        generic_rules=tuple(sorted(
            generic_rules, key=lambda item: item.input_text)),
        regional_rules=(NormalizationRecoveryTargetRule(
            input_text="區",
            output_text="区",
            rule_id=_id("regional-rule"),
            mapping_kind="CHARACTER_INPUT",
            rule_scope="REGIONAL_ZH_CN",
            authority_policy_scope=RECOVERY_TARGET_POLICY_SCOPE,
            regional_scope=RECOVERY_TRANSFER_REGION_SCOPE,
        ),),
        source_replays=source_replays,
        phrase_overrides=(phrase_rule,),
        conflicts=(conflict,),
        production_enabled=0,
    )
    evidence = tuple({
        "hypothesis_output": replay.output_text,
        "observation_id": replay.observation_id,
        "source_policy_scope": replay.source_policy_scope,
        "stance": "SUPPORT",
    } for replay in source_replays)
    conflict_record = ({
        "conflict_id": conflict_id,
        "family_outputs": [{
            "observation_ids": list(observation_ids),
        }],
        "input_text": conflict_input,
        "record_kind": NORMALIZATION_RECOVERY_CONFLICT_KIND,
    },)
    outputs = {
        name: () for name, _role, _identity
        in NORMALIZATION_RECOVERY_OUTPUT_FILE_ROLES}
    outputs.update({
        "conflict-ledger.jsonl": conflict_record,
        "evidence.jsonl": evidence,
        "source-phrase-rules.jsonl": ({
            "record_kind": NORMALIZATION_RECOVERY_PHRASE_RULE_KIND,
            "rule_id": phrase_rule.rule_id,
        },),
    })
    pack = {
        "manifest_sha256": pack_sha,
        "mastery_claimed": 0,
        "production_enabled": 0,
        "protocol_manifest_sha256": training_sha,
        "runtime_state": "LEARNED_PACK_DISABLED",
    }
    return protocol, records, pack, outputs, program


def test_recovery_synthetic_six_dimensions_all_pass() -> None:
    """达到冻结分母、scope 与 replay 门的 synthetic family 必须全 PASS。"""
    protocol, records, pack, outputs, program = _synthetic_family()
    report = evaluate_normalization_recovery_candidate(
        protocol_manifest=protocol,
        evaluation_records=records,
        rule_pack_manifest=pack,
        pack_outputs=outputs,
        program=program,
    )
    assert report.overall_outcome == "PASS"
    assert tuple(item.dimension_key for item in report.dimensions) == (
        NORMALIZATION_RECOVERY_DIMENSION_ORDER)
    assert {item.outcome for item in report.dimensions} == {"PASS"}
    metrics = {item.dimension_key: dict(item.metrics)
               for item in report.dimensions}
    assert metrics["LOCAL_MAPPING_TRANSFER"][
        "supported_mapping_count"] == 320
    assert metrics["END_TO_END_COVERAGE"][
        "phrase_inventory_count"] == 4_096
    assert metrics["END_TO_END_COVERAGE"][
        "identity_inventory_count"] == 192
    assert metrics["INDEPENDENT_CONTEXT_TRANSFER"][
        "context_exact_support_count"] == 64
    assert metrics["SOURCE_POLICY_CONFLICT"][
        "training_source_policy_count"] == 4
    assert metrics["DEFEATER_REPRESENTATION_EXECUTABILITY"][
        "identity_only_defeater_count"] == 0
    assert report.production_enabled == 0
    assert report.reserve_payload_read_count == 0


def test_missing_candidate_is_ne_without_zero_facility_pass() -> None:
    """缺 candidate 时六个 bearing dimension 均保持 NE。"""
    protocol, records, pack, outputs, _program = _synthetic_family()
    report = evaluate_normalization_recovery_candidate(
        protocol_manifest=protocol,
        evaluation_records=records,
        rule_pack_manifest=pack,
        pack_outputs=outputs,
        program=None,
    )
    assert report.overall_outcome == "NE"
    assert {item.outcome for item in report.dimensions} == {"NE"}


def test_wrong_applicable_local_mapping_is_fail() -> None:
    """已适用 target mapping 输出错误时必须 FAIL，不能记成覆盖不足。"""
    protocol, records, pack, outputs, program = _synthetic_family(
        wrong_local=True)
    report = evaluate_normalization_recovery_candidate(
        protocol_manifest=protocol,
        evaluation_records=records,
        rule_pack_manifest=pack,
        pack_outputs=outputs,
        program=program,
    )
    dimensions = {item.dimension_key: item for item in report.dimensions}
    assert report.overall_outcome == "FAIL"
    assert dimensions["LOCAL_MAPPING_TRANSFER"].outcome == "FAIL"
    assert dict(dimensions["LOCAL_MAPPING_TRANSFER"].metrics)[
        "mapping_false_accept_count"] == 1
