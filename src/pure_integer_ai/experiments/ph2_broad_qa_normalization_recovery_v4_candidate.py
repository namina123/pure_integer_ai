"""编译并执行 recovery-v4 Firefox transfer candidate。

Candidate 只组合两项已经冻结的 authority：v4 跨 UI 来源 phrase program 与
recovery-v2 的 zh-CN 单字符 backoff。source-scoped phrase 仍只在其原 source
family/policy 内执行；把 authority 投影到 Firefox target 不会改写原 scope。
"""
from __future__ import annotations

import hashlib

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_candidate_records import (
    NormalizationRecoveryCandidateProgram,
    RECOVERY_TRANSFER_REGION_SCOPE,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v3_training_records import (
    GODOT_SOURCE_FAMILY,
    GODOT_SOURCE_POLICY_SCOPE,
    THUNDERBIRD_SOURCE_FAMILY,
    THUNDERBIRD_SOURCE_POLICY_SCOPE,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v4_learning_contract import (
    NORMALIZATION_RECOVERY_V4_OUTPUT_FILE_ROLES,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v4_phrase_runtime import (
    compile_normalization_recovery_v4_phrase_program,
    execute_normalization_recovery_v4_phrase_batch,
    reference_normalization_recovery_v4_phrase_batch,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v4_training_records import (
    RECOVERY_V4_TARGET_POLICY_SCOPE,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v4_vscode_source_pack import (
    VSCODE_SOURCE_FAMILY,
    VSCODE_SOURCE_POLICY_SCOPE,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes


NORMALIZATION_RECOVERY_V4_CANDIDATE_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V4_FIREFOX_CANDIDATE_V1")
NORMALIZATION_RECOVERY_V4_CANDIDATE_STATUS = (
    "FROZEN_CANDIDATE_PRODUCTION_DISABLED_NOT_FORMALLY_EVALUATED")
NORMALIZATION_RECOVERY_V4_FIREFOX_TARGET_POLICY_SCOPE = (
    "ZH_CN_FIREFOX_LOCALIZATION_TRANSFER_V4")
NORMALIZATION_RECOVERY_V4_TARGET_PRECEDENCE = (
    "V4_CROSS_FAMILY_WHOLE_INPUT_EXACT",
    "RECOVERY_V2_REGIONAL_CHARACTER_BACKOFF",
    "RECOVERY_V2_GENERIC_CHARACTER_BACKOFF",
    "PRESERVE_UNKNOWN",
)

_SOURCE_POLICY_TO_FAMILY = {
    GODOT_SOURCE_POLICY_SCOPE: GODOT_SOURCE_FAMILY,
    THUNDERBIRD_SOURCE_POLICY_SCOPE: THUNDERBIRD_SOURCE_FAMILY,
    VSCODE_SOURCE_POLICY_SCOPE: VSCODE_SOURCE_FAMILY,
}


def _sha256(payload: bytes) -> str:
    """返回 candidate、profile 或执行结果的 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _sha_value(value: object, *, label: str) -> str:
    """核验并返回小写 SHA-256。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(item not in "0123456789abcdef" for item in value)):
        raise BroadQaExternalDataError(f"{label} 非法")
    return value


def _program_payload(program: dict[str, object]) -> dict[str, object]:
    """移除 candidate 自摘要字段。"""
    return {key: value for key, value in program.items()
            if key != "candidate_program_sha256"}


def _character_rules(
        base_program: NormalizationRecoveryCandidateProgram,
        ) -> list[dict[str, object]]:
    """从 v2 authority program 冻结 zh-CN regional-first 字符 backoff。"""
    generic = {
        item.input_text: item for item in base_program.generic_rules
        if item.mapping_kind == "CHARACTER_INPUT"
    }
    regional = {
        item.input_text: item for item in base_program.regional_rules
        if item.mapping_kind == "CHARACTER_INPUT"
    }
    values = []
    for input_text in sorted(set(generic) | set(regional)):
        rule = regional.get(input_text, generic.get(input_text))
        if (rule is None or len(input_text) != 1
                or len(rule.output_text) != 1):
            raise BroadQaExternalDataError(
                "recovery v4 base character rule 非单 scalar 映射")
        values.append({
            "input_text": input_text,
            "output_text": rule.output_text,
            "rule_id": rule.rule_id,
            "rule_scope": rule.rule_scope,
        })
    if not values:
        raise BroadQaExternalDataError("recovery v4 base character roster 为空")
    return values


def _compact_conflicts(
        values: tuple[dict[str, object], ...],
        ) -> list[dict[str, object]]:
    """冻结不得无 scope 执行的 v4 conflict identity。"""
    compact = []
    for record in values:
        if (not isinstance(record, dict)
                or record.get("production_enabled") != 0
                or record.get("unscoped_execution_allowed") != 0
                or not isinstance(record.get("input_text"), str)
                or not record["input_text"]):
            raise BroadQaExternalDataError("recovery v4 conflict schema 漂移")
        compact.append({
            "conflict_id": _sha_value(
                record.get("conflict_id"), label="recovery v4 conflict id"),
            "conflict_kind": str(record.get("conflict_kind")),
            "input_text": record["input_text"],
        })
    compact.sort(key=lambda item: (str(item["input_text"]),
                                   str(item["conflict_id"])))
    if (not compact or len({item["conflict_id"] for item in compact})
            != len(compact)):
        raise BroadQaExternalDataError("recovery v4 conflict identity 重复")
    return compact


def compile_normalization_recovery_v4_candidate(
        *,
        base_program: NormalizationRecoveryCandidateProgram,
        base_rule_pack_manifest_sha256: str,
        v4_protocol_manifest_sha256: str,
        v4_rule_pack_manifest_sha256: str,
        v4_training_audit_manifest_sha256: str,
        evaluation_commitment_manifest_sha256: str,
        v4_outputs: dict[str, tuple[dict[str, object], ...]],
        ) -> dict[str, object]:
    """从严格回读的 base/v4 pack 编译显式 scope 的禁用态 candidate。"""
    if not isinstance(base_program, NormalizationRecoveryCandidateProgram):
        raise BroadQaExternalDataError("recovery v4 base candidate 非法")
    expected = {name for name, _role, _identity
                in NORMALIZATION_RECOVERY_V4_OUTPUT_FILE_ROLES}
    if not isinstance(v4_outputs, dict) or set(v4_outputs) != expected:
        raise BroadQaExternalDataError("recovery v4 candidate output inventory 漂移")
    identities = {
        "base_rule_pack_manifest_sha256": _sha_value(
            base_rule_pack_manifest_sha256, label="recovery v4 base pack"),
        "evaluation_commitment_manifest_sha256": _sha_value(
            evaluation_commitment_manifest_sha256,
            label="recovery v4 evaluation commitment"),
        "v4_protocol_manifest_sha256": _sha_value(
            v4_protocol_manifest_sha256, label="recovery v4 protocol"),
        "v4_rule_pack_manifest_sha256": _sha_value(
            v4_rule_pack_manifest_sha256, label="recovery v4 pack"),
        "v4_training_audit_manifest_sha256": _sha_value(
            v4_training_audit_manifest_sha256,
            label="recovery v4 training audit"),
    }
    phrase_program = compile_normalization_recovery_v4_phrase_program(
        rule_pack_manifest_sha256=identities[
            "v4_rule_pack_manifest_sha256"],
        target_phrase_rules=v4_outputs["target-phrase-rules.jsonl"],
        source_phrase_rules=v4_outputs["source-phrase-rules.jsonl"],
        defeaters=v4_outputs["defeaters.jsonl"],
        target_overlap_index=v4_outputs["target-overlap-index.jsonl"],
        source_overlap_index=v4_outputs["source-overlap-index.jsonl"],
    )
    transfer_profile = {
        "authority_policy_scope": RECOVERY_V4_TARGET_POLICY_SCOPE,
        "candidate_target_policy_scope": (
            NORMALIZATION_RECOVERY_V4_FIREFOX_TARGET_POLICY_SCOPE),
        "evaluation_commitment_manifest_sha256": identities[
            "evaluation_commitment_manifest_sha256"],
        "regional_scope": RECOVERY_TRANSFER_REGION_SCOPE,
        "source_policy_to_family": [
            {"source_family": family, "source_policy_scope": policy}
            for policy, family in sorted(_SOURCE_POLICY_TO_FAMILY.items())
        ],
        "target_precedence": list(
            NORMALIZATION_RECOVERY_V4_TARGET_PRECEDENCE),
    }
    profile_sha = _sha256(canonical_json_bytes(transfer_profile))
    program = {
        "artifact_kind": NORMALIZATION_RECOVERY_V4_CANDIDATE_KIND,
        "base_candidate_program_sha256": base_program.sha256(),
        "base_character_rules": _character_rules(base_program),
        "base_rule_pack_manifest_sha256": identities[
            "base_rule_pack_manifest_sha256"],
        "conflicts": _compact_conflicts(
            v4_outputs["conflict-ledger.jsonl"]),
        "evaluation_commitment_manifest_sha256": identities[
            "evaluation_commitment_manifest_sha256"],
        "format_version": 1,
        "mastery_claimed": 0,
        "phrase_program": phrase_program,
        "production_enabled": 0,
        "status": NORMALIZATION_RECOVERY_V4_CANDIDATE_STATUS,
        "transfer_profile": transfer_profile,
        "transfer_profile_sha256": profile_sha,
        "v4_protocol_manifest_sha256": identities[
            "v4_protocol_manifest_sha256"],
        "v4_rule_pack_manifest_sha256": identities[
            "v4_rule_pack_manifest_sha256"],
        "v4_training_audit_manifest_sha256": identities[
            "v4_training_audit_manifest_sha256"],
    }
    return {
        **program,
        "candidate_program_sha256": _sha256(canonical_json_bytes(program)),
    }


def _validate_program(
        program: dict[str, object],
        ) -> tuple[dict[str, str], dict[str, dict[str, object]]]:
    """核验 candidate 自摘要、scope、字符 roster 与 conflict index。"""
    if (not isinstance(program, dict)
            or program.get("artifact_kind")
            != NORMALIZATION_RECOVERY_V4_CANDIDATE_KIND
            or program.get("status")
            != NORMALIZATION_RECOVERY_V4_CANDIDATE_STATUS
            or program.get("format_version") != 1
            or program.get("production_enabled") != 0
            or program.get("mastery_claimed") != 0
            or program.get("candidate_program_sha256") != _sha256(
                canonical_json_bytes(_program_payload(program)))):
        raise BroadQaExternalDataError("recovery v4 candidate identity 漂移")
    for name in (
            "base_candidate_program_sha256",
            "base_rule_pack_manifest_sha256",
            "evaluation_commitment_manifest_sha256",
            "transfer_profile_sha256",
            "v4_protocol_manifest_sha256",
            "v4_rule_pack_manifest_sha256",
            "v4_training_audit_manifest_sha256"):
        _sha_value(program.get(name), label=f"recovery v4 candidate {name}")
    profile = program.get("transfer_profile")
    expected_source_map = [
        {"source_family": family, "source_policy_scope": policy}
        for policy, family in sorted(_SOURCE_POLICY_TO_FAMILY.items())
    ]
    if (not isinstance(profile, dict)
            or profile.get("authority_policy_scope")
            != RECOVERY_V4_TARGET_POLICY_SCOPE
            or profile.get("candidate_target_policy_scope")
            != NORMALIZATION_RECOVERY_V4_FIREFOX_TARGET_POLICY_SCOPE
            or profile.get("regional_scope") != RECOVERY_TRANSFER_REGION_SCOPE
            or profile.get("target_precedence")
            != list(NORMALIZATION_RECOVERY_V4_TARGET_PRECEDENCE)
            or profile.get("source_policy_to_family") != expected_source_map
            or profile.get("evaluation_commitment_manifest_sha256")
            != program["evaluation_commitment_manifest_sha256"]
            or program["transfer_profile_sha256"]
            != _sha256(canonical_json_bytes(profile))):
        raise BroadQaExternalDataError("recovery v4 transfer profile 漂移")
    rules = program.get("base_character_rules")
    if not isinstance(rules, list) or not rules:
        raise BroadQaExternalDataError("recovery v4 character roster 为空")
    character_rules = {}
    character_records = {}
    keys = []
    for record in rules:
        if (not isinstance(record, dict)
                or set(record) != {
                    "input_text", "output_text", "rule_id", "rule_scope"}
                or not isinstance(record["input_text"], str)
                or len(record["input_text"]) != 1
                or not isinstance(record["output_text"], str)
                or len(record["output_text"]) != 1
                or record["rule_scope"] not in {
                    "GENERIC", "REGIONAL_ZH_CN"}):
            raise BroadQaExternalDataError(
                "recovery v4 character record 漂移")
        _sha_value(record["rule_id"], label="recovery v4 character rule")
        key = record["input_text"]
        keys.append(key)
        character_rules[key] = record["output_text"]
        character_records[key] = record
    if keys != sorted(set(keys)):
        raise BroadQaExternalDataError("recovery v4 character index 漂移")
    conflicts = program.get("conflicts")
    if not isinstance(conflicts, list) or not conflicts:
        raise BroadQaExternalDataError("recovery v4 conflict roster 为空")
    conflict_index = {}
    conflict_keys = []
    for record in conflicts:
        if (not isinstance(record, dict)
                or set(record) != {
                    "conflict_id", "conflict_kind", "input_text"}
                or not isinstance(record["input_text"], str)
                or not record["input_text"]):
            raise BroadQaExternalDataError("recovery v4 conflict record 漂移")
        _sha_value(record["conflict_id"], label="recovery v4 conflict")
        key = (record["input_text"], record["conflict_id"])
        conflict_keys.append(key)
        conflict_index.setdefault(record["input_text"], record)
    if conflict_keys != sorted(set(conflict_keys)):
        raise BroadQaExternalDataError("recovery v4 conflict index 漂移")
    return character_rules, conflict_index


def _result(
        *,
        program: dict[str, object],
        input_text: str,
        output_text: str,
        requested_policy_scope: str,
        regional_scope: str,
        source_family: str,
        phrase_result: dict[str, object] | None,
        projection_used: int,
        scope_mismatch: int,
        conflict: dict[str, object] | None,
        character_records: dict[str, dict[str, object]],
        ) -> dict[str, object]:
    """封装一次 candidate 执行并补齐字符 rule trace。"""
    steps = []
    if phrase_result is not None:
        for source_step in phrase_result["steps"]:
            step = dict(source_step)
            rule = character_records.get(
                input_text[int(step["input_start"])])
            step["character_rule_id"] = (
                rule["rule_id"]
                if step["mode"] == "CHARACTER_BACKOFF" and rule else "")
            steps.append(step)
    conflict_ids = [] if conflict is None else [conflict["conflict_id"]]
    payload = {
        "applicable": int(scope_mismatch == 0),
        "candidate_program_sha256": program["candidate_program_sha256"],
        "conflict_ids": conflict_ids,
        "input_text": input_text,
        "mastery_claimed": 0,
        "output_text": output_text,
        "phrase_result_sha256": (
            "" if phrase_result is None else phrase_result["result_sha256"]),
        "production_enabled": 0,
        "projection_used": projection_used,
        "regional_scope": regional_scope,
        "requested_policy_scope": requested_policy_scope,
        "scope_mismatch": scope_mismatch,
        "source_family": source_family,
        "steps": steps,
        "transfer_profile_sha256": (
            program["transfer_profile_sha256"] if projection_used else ""),
        "unscoped_conflict_blocked": int(conflict is not None),
    }
    return {**payload, "result_sha256": _sha256(canonical_json_bytes(payload))}


def _execute_batch(
        program: dict[str, object],
        texts: tuple[str, ...],
        *,
        policy_scope: str,
        regional_scope: str,
        indexed: bool,
        ) -> tuple[dict[str, object], ...]:
    """一次核验 composite candidate 后批量执行同一 scope。"""
    if (not isinstance(texts, tuple) or not texts
            or any(not isinstance(item, str) or not item for item in texts)
            or not isinstance(policy_scope, str)
            or not isinstance(regional_scope, str)):
        raise BroadQaExternalDataError("recovery v4 candidate batch 输入非法")
    character_rules, conflicts = _validate_program(program)
    character_records = {
        str(item["input_text"]): item
        for item in program["base_character_rules"]
    }
    source_family = ""
    projection = 0
    valid_scope = False
    if policy_scope == NORMALIZATION_RECOVERY_V4_FIREFOX_TARGET_POLICY_SCOPE:
        valid_scope = regional_scope == RECOVERY_TRANSFER_REGION_SCOPE
        projection = int(valid_scope)
    elif policy_scope == RECOVERY_V4_TARGET_POLICY_SCOPE:
        valid_scope = regional_scope in {"", RECOVERY_TRANSFER_REGION_SCOPE}
    elif policy_scope in _SOURCE_POLICY_TO_FAMILY:
        valid_scope = regional_scope == ""
        if valid_scope:
            source_family = _SOURCE_POLICY_TO_FAMILY[policy_scope]
    if not valid_scope:
        return tuple(_result(
            program=program,
            input_text=text,
            output_text=text,
            requested_policy_scope=policy_scope,
            regional_scope=regional_scope,
            source_family="",
            phrase_result=None,
            projection_used=0,
            scope_mismatch=1,
            conflict=conflicts.get(text),
            character_records=character_records,
        ) for text in texts)
    executor = (execute_normalization_recovery_v4_phrase_batch
                if indexed else reference_normalization_recovery_v4_phrase_batch)
    phrase_results = executor(
        program["phrase_program"],
        texts,
        source_family=source_family,
        character_rules=character_rules,
    )
    return tuple(_result(
        program=program,
        input_text=text,
        output_text=str(phrase["output_text"]),
        requested_policy_scope=policy_scope,
        regional_scope=regional_scope,
        source_family=source_family,
        phrase_result=phrase,
        projection_used=projection,
        scope_mismatch=0,
        conflict=None,
        character_records=character_records,
    ) for text, phrase in zip(texts, phrase_results))


def execute_normalization_recovery_v4_candidate_batch(
        program: dict[str, object],
        texts: tuple[str, ...],
        *,
        policy_scope: str,
        regional_scope: str = "",
        ) -> tuple[dict[str, object], ...]:
    """批量执行 indexed recovery-v4 candidate。"""
    return _execute_batch(
        program, texts, policy_scope=policy_scope,
        regional_scope=regional_scope, indexed=True)


def reference_normalization_recovery_v4_candidate_batch(
        program: dict[str, object],
        texts: tuple[str, ...],
        *,
        policy_scope: str,
        regional_scope: str = "",
        ) -> tuple[dict[str, object], ...]:
    """批量执行不使用 phrase 首 scalar index 的 reference candidate。"""
    return _execute_batch(
        program, texts, policy_scope=policy_scope,
        regional_scope=regional_scope, indexed=False)


def execute_normalization_recovery_v4_candidate(
        program: dict[str, object],
        text: str,
        *,
        policy_scope: str,
        regional_scope: str = "",
        ) -> dict[str, object]:
    """执行单条 indexed recovery-v4 candidate。"""
    return execute_normalization_recovery_v4_candidate_batch(
        program, (text,), policy_scope=policy_scope,
        regional_scope=regional_scope)[0]


def reference_normalization_recovery_v4_candidate(
        program: dict[str, object],
        text: str,
        *,
        policy_scope: str,
        regional_scope: str = "",
        ) -> dict[str, object]:
    """执行单条 linear-reference recovery-v4 candidate。"""
    return reference_normalization_recovery_v4_candidate_batch(
        program, (text,), policy_scope=policy_scope,
        regional_scope=regional_scope)[0]


__all__ = [
    "NORMALIZATION_RECOVERY_V4_CANDIDATE_KIND",
    "NORMALIZATION_RECOVERY_V4_CANDIDATE_STATUS",
    "NORMALIZATION_RECOVERY_V4_FIREFOX_TARGET_POLICY_SCOPE",
    "compile_normalization_recovery_v4_candidate",
    "execute_normalization_recovery_v4_candidate",
    "execute_normalization_recovery_v4_candidate_batch",
    "reference_normalization_recovery_v4_candidate",
    "reference_normalization_recovery_v4_candidate_batch",
]
