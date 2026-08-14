"""从冻结 recovery TRAIN protocol 纯派生 Evidence 与禁用态学习记录。

本模块不读取路径、不写 artifact、不导入 evaluation。所有输出只由 protocol
manifest identity、observation、family-aware group、composition 与 ordered work 决定。
"""
from __future__ import annotations

from collections import Counter
import hashlib

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_training_records import (
    RECOVERY_TARGET_POLICY_SCOPE,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    canonical_json_line,
)


NORMALIZATION_RECOVERY_EVIDENCE_KIND = "NORMALIZATION_RECOVERY_EVIDENCE_V2"
NORMALIZATION_RECOVERY_GENERIC_RULE_KIND = (
    "NORMALIZATION_RECOVERY_GENERIC_TARGET_RULE_V2")
NORMALIZATION_RECOVERY_REGIONAL_RULE_KIND = (
    "NORMALIZATION_RECOVERY_REGIONAL_EXACT_RULE_V2")
NORMALIZATION_RECOVERY_CONFLICT_KIND = (
    "NORMALIZATION_RECOVERY_SOURCE_FAMILY_CONFLICT_V2")
NORMALIZATION_RECOVERY_GROUP_DECISION_KIND = (
    "NORMALIZATION_RECOVERY_GROUP_DECISION_V2")
NORMALIZATION_RECOVERY_COMPOSITION_RECEIPT_KIND = (
    "NORMALIZATION_RECOVERY_COMPOSITION_RECEIPT_V2")
NORMALIZATION_RECOVERY_PHRASE_RULE_KIND = (
    "NORMALIZATION_RECOVERY_SOURCE_PHRASE_RULE_V2")

NORMALIZATION_RECOVERY_OUTPUT_FILE_ROLES = (
    ("evidence.jsonl", "LEARNED_EVIDENCE", "evidence_id"),
    ("generic-rules.jsonl", "LEARNED_GENERIC_RULES", "rule_id"),
    ("regional-rules.jsonl", "LEARNED_REGIONAL_RULES", "rule_id"),
    ("conflict-ledger.jsonl", "LEARNED_CONFLICT_LEDGER", "conflict_id"),
    ("group-decisions.jsonl", "LEARNED_GROUP_DECISIONS", "decision_id"),
    ("composition-receipts.jsonl", "LEARNED_COMPOSITION_RECEIPTS", "receipt_id"),
    ("source-phrase-rules.jsonl", "LEARNED_SOURCE_PHRASE_RULES", "rule_id"),
)


def _sha256(payload: bytes) -> str:
    """返回规范记录或来源 identity 的 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _sha_value(value: object, *, label: str) -> str:
    """核验并返回小写 SHA-256。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(item not in "0123456789abcdef" for item in value)):
        raise BroadQaExternalDataError(f"{label} 非法")
    return value


def _record_id(identity: dict[str, object]) -> str:
    """对协议绑定的完整记录 identity 计算稳定摘要。"""
    return _sha256(canonical_json_bytes(identity))


def _observation_evidence(
        *,
        protocol_manifest_sha256: str,
        observation: dict[str, object],
        roster_record: dict[str, object],
        ) -> dict[str, object]:
    """把一条来源 observation 投影为 source-policy SUPPORT Evidence。"""
    identity = {
        "hypothesis_kind": "SOURCE_POLICY_MAPPING",
        "hypothesis_output": observation["expected_output"],
        "input_text": observation["input_text"],
        "observation_id": observation["observation_id"],
        "protocol_manifest_sha256": protocol_manifest_sha256,
        "source_commitment_sha256": _sha256(canonical_json_bytes(
            observation["source_commitment"])),
        "source_family": observation["source_family"],
        "source_policy_scope": observation["source_policy_scope"],
        "source_roster_id": observation["source_roster_id"],
        "stance": "SUPPORT",
        "target_policy_scope": "",
    }
    return {
        **identity,
        "authority_role": observation["authority_role"],
        "evidence_id": _record_id(identity),
        "evidence_source_scope": observation["evidence_source_scope"],
        "format_version": 2,
        "license_id": roster_record["license_id"],
        "record_kind": NORMALIZATION_RECOVERY_EVIDENCE_KIND,
        "source_commitment": observation["source_commitment"],
        "source_pack_manifest_sha256": observation[
            "source_pack_manifest_sha256"],
    }


def _group_evidence_ids(
        group: dict[str, object],
        evidence_by_observation: dict[str, dict[str, object]],
        ) -> list[str]:
    """按 group observation identity 投影完整 Evidence 集。"""
    try:
        return sorted(str(evidence_by_observation[str(value)]["evidence_id"])
                      for value in group["observation_ids"])
    except KeyError as error:
        raise BroadQaExternalDataError(
            "recovery learning group evidence 缺失") from error


def _target_rule(
        *,
        protocol_manifest_sha256: str,
        group: dict[str, object],
        evidence_by_observation: dict[str, dict[str, object]],
        observation_by_id: dict[str, dict[str, object]],
        ) -> dict[str, object] | None:
    """从 family 共识或区域 exact authority 形成非恒等 target rule。"""
    target_kind = group["target_resolution_kind"]
    if (target_kind == "NO_TARGET_AUTHORITY"
            or group["target_rule_is_identity"] == 1):
        return None
    authority_ids = [str(value) for value in
                     group["target_authority_observation_ids"]]
    try:
        observations = [observation_by_id[value] for value in authority_ids]
        evidence_ids = sorted(str(evidence_by_observation[value]["evidence_id"])
                              for value in authority_ids)
    except KeyError as error:
        raise BroadQaExternalDataError(
            "recovery target authority observation 缺失") from error
    if not observations:
        raise BroadQaExternalDataError(
            "recovery target authority evidence 为空")
    identity = {
        "authority_observation_ids": authority_ids,
        "evidence_ids": evidence_ids,
        "group_id": group["group_id"],
        "input_text": group["input_text"],
        "output_text": group["target_output"],
        "protocol_manifest_sha256": protocol_manifest_sha256,
        "target_policy_scope": RECOVERY_TARGET_POLICY_SCOPE,
        "target_resolution_kind": target_kind,
    }
    common = {
        **identity,
        "evidence_source_scopes": sorted(
            str(item["evidence_source_scope"]) for item in observations),
        "format_version": 2,
        "mapping_kind": group["mapping_kind"],
        "production_enabled": 0,
        "rule_id": _record_id(identity),
        "runtime_state": "LEARNED_PACK_DISABLED",
        "source_families": sorted({
            str(item["source_family"]) for item in observations}),
        "source_policy_scopes": sorted({
            str(item["source_policy_scope"]) for item in observations}),
    }
    if target_kind == "CROSS_FAMILY_CONSENSUS":
        return {
            **common,
            "application_domain": {
                "input_match": "EXACT_SCALAR_SEQUENCE",
                "normalization_policy_scope": RECOVERY_TARGET_POLICY_SCOPE,
            },
            "minimum_distinct_source_family_count": 2,
            "record_kind": NORMALIZATION_RECOVERY_GENERIC_RULE_KIND,
        }
    if target_kind != "REGIONAL_EXACT_AUTHORITY":
        raise BroadQaExternalDataError(
            "recovery target resolution kind 非法")
    return {
        **common,
        "application_domain": {
            "exact_input": group["input_text"],
            "normalization_policy_scope": RECOVERY_TARGET_POLICY_SCOPE,
            "regional_scope": "ZH_CN",
        },
        "global_upgrade_allowed": 0,
        "record_kind": NORMALIZATION_RECOVERY_REGIONAL_RULE_KIND,
    }


def _conflict_record(
        *,
        protocol_manifest_sha256: str,
        group: dict[str, object],
        evidence_by_observation: dict[str, dict[str, object]],
        ) -> dict[str, object] | None:
    """保留 family 内/间 generic 冲突，不覆盖可能并存的区域 exact rule。"""
    kind = group["generic_resolution_kind"]
    if kind not in {"INTRA_FAMILY_CONFLICT", "SOURCE_FAMILY_CONFLICT"}:
        return None
    generic_observation_ids = sorted({
        str(value)
        for family in group["family_outputs"]
        for value in family["observation_ids"]
    })
    try:
        evidence_ids = sorted(str(
            evidence_by_observation[value]["evidence_id"])
            for value in generic_observation_ids)
    except KeyError as error:
        raise BroadQaExternalDataError(
            "recovery conflict evidence 缺失") from error
    identity = {
        "conflict_kind": kind,
        "evidence_ids": evidence_ids,
        "family_outputs": group["family_outputs"],
        "group_id": group["group_id"],
        "input_text": group["input_text"],
        "protocol_manifest_sha256": protocol_manifest_sha256,
        "target_policy_scope": RECOVERY_TARGET_POLICY_SCOPE,
    }
    return {
        **identity,
        "conflict_id": _record_id(identity),
        "format_version": 2,
        "production_enabled": 0,
        "record_kind": NORMALIZATION_RECOVERY_CONFLICT_KIND,
        "resolution_state": "FAMILY_AUTHORITY_UNRESOLVED",
        "unscoped_application_allowed": 0,
    }


def _group_decision(
        *,
        protocol_manifest_sha256: str,
        group: dict[str, object],
        evidence_by_observation: dict[str, dict[str, object]],
        target_rule: dict[str, object] | None,
        conflict: dict[str, object] | None,
        ) -> dict[str, object]:
    """每个 exact-input group 形成一条完整 resolution receipt。"""
    if group["target_resolution_kind"] == "NO_TARGET_AUTHORITY":
        decision_kind = "DEFER_NO_TARGET_AUTHORITY"
    elif group["target_rule_is_identity"] == 1:
        decision_kind = "TARGET_AUTHORITY_IDENTITY_NOOP"
    elif group["target_resolution_kind"] == "REGIONAL_EXACT_AUTHORITY":
        decision_kind = "REGIONAL_EXACT_RULE_EMITTED"
    else:
        decision_kind = "GENERIC_CROSS_FAMILY_RULE_EMITTED"
    identity = {
        "conflict_id": "" if conflict is None else conflict["conflict_id"],
        "decision_kind": decision_kind,
        "evidence_ids": _group_evidence_ids(group, evidence_by_observation),
        "generic_resolution_kind": group["generic_resolution_kind"],
        "group_id": group["group_id"],
        "input_text": group["input_text"],
        "protocol_manifest_sha256": protocol_manifest_sha256,
        "rule_id": "" if target_rule is None else target_rule["rule_id"],
        "target_output": group["target_output"],
        "target_policy_scope": (
            RECOVERY_TARGET_POLICY_SCOPE
            if group["target_resolution_kind"] != "NO_TARGET_AUTHORITY" else ""),
        "target_resolution_kind": group["target_resolution_kind"],
    }
    return {
        **identity,
        "decision_id": _record_id(identity),
        "eligible_rule": int(target_rule is not None),
        "format_version": 2,
        "production_enabled": 0,
        "record_kind": NORMALIZATION_RECOVERY_GROUP_DECISION_KIND,
    }


def _composition_refute_evidence(
        *,
        protocol_manifest_sha256: str,
        composition: dict[str, object],
        phrase: dict[str, object],
        ) -> dict[str, object]:
    """为完整显式 phrase override 反驳逐字符 target composition。"""
    identity = {
        "composition_id": composition["composition_id"],
        "hypothesis_kind": "TARGET_CHARACTER_COMPOSITION_UNDER_SOURCE_POLICY",
        "hypothesis_output": composition["base_output"],
        "input_text": composition["input_text"],
        "observation_id": phrase["observation_id"],
        "protocol_manifest_sha256": protocol_manifest_sha256,
        "source_commitment_sha256": _sha256(canonical_json_bytes(
            phrase["source_commitment"])),
        "source_family": phrase["source_family"],
        "source_policy_scope": phrase["source_policy_scope"],
        "stance": "REFUTE",
        "target_policy_scope": RECOVERY_TARGET_POLICY_SCOPE,
    }
    return {
        **identity,
        "authority_role": phrase["authority_role"],
        "evidence_id": _record_id(identity),
        "evidence_source_scope": phrase["evidence_source_scope"],
        "format_version": 2,
        "record_kind": NORMALIZATION_RECOVERY_EVIDENCE_KIND,
        "source_commitment": phrase["source_commitment"],
        "source_pack_manifest_sha256": phrase[
            "source_pack_manifest_sha256"],
    }


def _composition_outputs(
        *,
        protocol_manifest_sha256: str,
        composition: dict[str, object],
        observation_by_id: dict[str, dict[str, object]],
        evidence_by_observation: dict[str, dict[str, object]],
        ) -> tuple[dict[str, object], dict[str, object] | None, dict[str, object] | None]:
    """构造 composition receipt，并只为完整 override 生成精确来源规则。"""
    phrase_id = str(composition["phrase_observation_id"])
    phrase = observation_by_id.get(phrase_id)
    support = evidence_by_observation.get(phrase_id)
    if (phrase is None or support is None
            or phrase["mapping_kind"] != "PHRASE_INPUT"
            or phrase["source_policy_scope"] != composition["source_policy_scope"]):
        raise BroadQaExternalDataError(
            "recovery composition phrase evidence 漂移")
    override = composition["qualification_kind"] == "EXPLICIT_OVERRIDE"
    refute = None
    phrase_rule = None
    if override:
        refute = _composition_refute_evidence(
            protocol_manifest_sha256=protocol_manifest_sha256,
            composition=composition,
            phrase=phrase,
        )
        rule_identity = {
            "composition_id": composition["composition_id"],
            "input_text": composition["input_text"],
            "output_text": composition["observed_output"],
            "protocol_manifest_sha256": protocol_manifest_sha256,
            "refute_evidence_id": refute["evidence_id"],
            "source_policy_scope": composition["source_policy_scope"],
            "support_evidence_id": support["evidence_id"],
        }
        phrase_rule = {
            **rule_identity,
            "application_domain": {
                "exact_input": composition["input_text"],
                "source_policy_scope": composition["source_policy_scope"],
            },
            "format_version": 2,
            "production_enabled": 0,
            "record_kind": NORMALIZATION_RECOVERY_PHRASE_RULE_KIND,
            "rule_id": _record_id(rule_identity),
            "runtime_state": "LEARNED_PACK_DISABLED",
            "target_policy_scope": "",
        }
    receipt_identity = {
        "base_character_group_ids": composition["base_character_group_ids"],
        "base_output": composition["base_output"],
        "composition_id": composition["composition_id"],
        "input_text": composition["input_text"],
        "observed_output": composition["observed_output"],
        "phrase_rule_id": "" if phrase_rule is None else phrase_rule["rule_id"],
        "protocol_manifest_sha256": protocol_manifest_sha256,
        "qualification_kind": composition["qualification_kind"],
        "refute_evidence_id": "" if refute is None else refute["evidence_id"],
        "source_policy_scope": composition["source_policy_scope"],
        "support_evidence_id": support["evidence_id"],
        "target_group_id": composition["target_group_id"],
    }
    receipt = {
        **receipt_identity,
        "covered_positions": composition["covered_positions"],
        "format_version": 2,
        "record_kind": NORMALIZATION_RECOVERY_COMPOSITION_RECEIPT_KIND,
        "receipt_id": _record_id(receipt_identity),
        "target_output": composition["target_output"],
        "target_resolution_kind": composition["target_resolution_kind"],
        "unknown_position_count": composition["unknown_position_count"],
    }
    return receipt, refute, phrase_rule


def _require_work_alignment(
        *,
        roster: tuple[dict[str, object], ...],
        observations: tuple[dict[str, object], ...],
        groups: tuple[dict[str, object], ...],
        compositions: tuple[dict[str, object], ...],
        work: tuple[dict[str, object], ...],
        ) -> None:
    """要求 ordered work 逐阶段精确覆盖四类 material identity。"""
    expected = (
        [("SOURCE_ROSTER_INGEST", "ROSTER", item["roster_id"])
         for item in roster]
        + [("LICENSE_PARTITIONED_OBSERVATION_INGEST", "OBSERVATION",
            item["observation_id"]) for item in observations]
        + [("SOURCE_FAMILY_GROUP_RESOLUTION", "GROUP", item["group_id"])
           for item in groups]
        + [("PHRASE_COMPOSITION_RESOLUTION", "COMPOSITION",
            item["composition_id"]) for item in compositions]
    )
    observed = [
        (item.get("phase"), item.get("work_kind"), item.get("record_id"))
        for item in work
    ]
    if (observed != expected
            or [item.get("work_ordinal") for item in work]
            != list(range(len(work)))):
        raise BroadQaExternalDataError(
            "recovery learning ordered work/material 漂移")


def normalization_recovery_prefix_output_counts(
        *,
        work: tuple[dict[str, object], ...],
        groups: tuple[dict[str, object], ...],
        compositions: tuple[dict[str, object], ...],
        processed_item_count: int,
        ) -> tuple[int, int]:
    """机械计算任意冻结 work 前缀已形成的 Evidence 与非 Evidence 记录数。"""
    if (type(processed_item_count) is not int
            or not 0 <= processed_item_count <= len(work)):
        raise BroadQaExternalDataError(
            "recovery learning processed prefix 非法")
    group_by_id = {str(item["group_id"]): item for item in groups}
    composition_by_id = {
        str(item["composition_id"]): item for item in compositions}
    if (len(group_by_id) != len(groups)
            or len(composition_by_id) != len(compositions)):
        raise BroadQaExternalDataError(
            "recovery learning group/composition identity 重复")
    evidence_count = 0
    result_count = 0
    for item in work[:processed_item_count]:
        kind = item["work_kind"]
        if kind == "ROSTER":
            continue
        if kind == "OBSERVATION":
            evidence_count += 1
            continue
        if kind == "GROUP":
            group = group_by_id.get(str(item["record_id"]))
            if group is None:
                raise BroadQaExternalDataError(
                    "recovery learning work group 缺失")
            result_count += 1
            result_count += int(
                group["target_resolution_kind"] != "NO_TARGET_AUTHORITY"
                and group["target_rule_is_identity"] == 0)
            result_count += int(group["generic_resolution_kind"] in {
                "INTRA_FAMILY_CONFLICT", "SOURCE_FAMILY_CONFLICT"})
            continue
        if kind == "COMPOSITION":
            composition = composition_by_id.get(str(item["record_id"]))
            if composition is None:
                raise BroadQaExternalDataError(
                    "recovery learning work composition 缺失")
            result_count += 1
            if composition["qualification_kind"] == "EXPLICIT_OVERRIDE":
                evidence_count += 1
                result_count += 1
            continue
        raise BroadQaExternalDataError(
            "recovery learning work kind 非法")
    return evidence_count, result_count


def derive_normalization_recovery_learning_outputs(
        *,
        protocol_manifest: dict[str, object],
        roster: tuple[dict[str, object], ...],
        observations: tuple[dict[str, object], ...],
        groups: tuple[dict[str, object], ...],
        compositions: tuple[dict[str, object], ...],
        work: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, tuple[dict[str, object], ...]], dict[str, object]]:
    """从完整 TRAIN protocol 派生唯一 Evidence、规则、账本和 receipt。"""
    protocol_sha = _sha_value(
        protocol_manifest.get("manifest_sha256"),
        label="recovery learning protocol manifest")
    if (not roster or not observations or not groups or not compositions or not work
            or protocol_manifest.get("target_policy_scope")
            != RECOVERY_TARGET_POLICY_SCOPE):
        raise BroadQaExternalDataError(
            "recovery learning protocol material 为空或 target policy 漂移")
    _require_work_alignment(
        roster=roster,
        observations=observations,
        groups=groups,
        compositions=compositions,
        work=work,
    )
    observation_by_id = {
        str(item["observation_id"]): item for item in observations}
    if len(observation_by_id) != len(observations):
        raise BroadQaExternalDataError(
            "recovery learning observation identity 重复")
    roster_by_id = {str(item["roster_id"]): item for item in roster}
    if len(roster_by_id) != len(roster):
        raise BroadQaExternalDataError(
            "recovery learning roster identity 重复")
    for observation in observations:
        roster_record = roster_by_id.get(str(observation["source_roster_id"]))
        if (roster_record is None
                or any(observation[key] != roster_record[key] for key in (
                    "authority_role", "source_family", "source_key",
                    "source_pack_manifest_sha256", "source_policy_scope"))):
            raise BroadQaExternalDataError(
                "recovery learning observation/roster 漂移")
    group_by_observation = {}
    for group in groups:
        for observation_id in group["observation_ids"]:
            key = str(observation_id)
            if key not in observation_by_id or key in group_by_observation:
                raise BroadQaExternalDataError(
                    "recovery learning group/observation 关系漂移")
            group_by_observation[key] = group
    if set(group_by_observation) != set(observation_by_id):
        raise BroadQaExternalDataError(
            "recovery learning group 未覆盖全部 observation")

    observation_evidence = tuple(_observation_evidence(
        protocol_manifest_sha256=protocol_sha,
        observation=item,
        roster_record=roster_by_id[str(item["source_roster_id"])],
    ) for item in observations)
    evidence_by_observation = {
        str(item["observation_id"]): evidence
        for item, evidence in zip(observations, observation_evidence)
    }
    generic_rules = []
    regional_rules = []
    conflicts = []
    decisions = []
    for group in groups:
        target_rule = _target_rule(
            protocol_manifest_sha256=protocol_sha,
            group=group,
            evidence_by_observation=evidence_by_observation,
            observation_by_id=observation_by_id,
        )
        conflict = _conflict_record(
            protocol_manifest_sha256=protocol_sha,
            group=group,
            evidence_by_observation=evidence_by_observation,
        )
        decisions.append(_group_decision(
            protocol_manifest_sha256=protocol_sha,
            group=group,
            evidence_by_observation=evidence_by_observation,
            target_rule=target_rule,
            conflict=conflict,
        ))
        if target_rule is not None:
            if target_rule["record_kind"] == NORMALIZATION_RECOVERY_GENERIC_RULE_KIND:
                generic_rules.append(target_rule)
            else:
                regional_rules.append(target_rule)
        if conflict is not None:
            conflicts.append(conflict)

    composition_receipts = []
    composition_refutes = []
    phrase_rules = []
    seen_phrase_ids = set()
    for composition in compositions:
        phrase_id = str(composition["phrase_observation_id"])
        if phrase_id in seen_phrase_ids:
            raise BroadQaExternalDataError(
                "recovery learning composition phrase 重复")
        seen_phrase_ids.add(phrase_id)
        receipt, refute, phrase_rule = _composition_outputs(
            protocol_manifest_sha256=protocol_sha,
            composition=composition,
            observation_by_id=observation_by_id,
            evidence_by_observation=evidence_by_observation,
        )
        composition_receipts.append(receipt)
        if refute is not None:
            composition_refutes.append(refute)
        if phrase_rule is not None:
            phrase_rules.append(phrase_rule)
    expected_phrase_ids = {
        str(item["observation_id"]) for item in observations
        if item["mapping_kind"] == "PHRASE_INPUT"}
    if seen_phrase_ids != expected_phrase_ids:
        raise BroadQaExternalDataError(
            "recovery learning composition 未覆盖全部 phrase observation")

    raw_outputs = {
        "evidence.jsonl": list(observation_evidence) + composition_refutes,
        "generic-rules.jsonl": generic_rules,
        "regional-rules.jsonl": regional_rules,
        "conflict-ledger.jsonl": conflicts,
        "group-decisions.jsonl": decisions,
        "composition-receipts.jsonl": composition_receipts,
        "source-phrase-rules.jsonl": phrase_rules,
    }
    outputs = {}
    for name, _role, identity_key in NORMALIZATION_RECOVERY_OUTPUT_FILE_ROLES:
        values = tuple(sorted(
            raw_outputs[name], key=lambda item: str(item[identity_key])))
        if len({item[identity_key] for item in values}) != len(values):
            raise BroadQaExternalDataError(
                f"recovery learning {name} identity 重复")
        outputs[name] = values
    evidence_stances = Counter(
        str(item["stance"]) for item in outputs["evidence.jsonl"])
    decision_kinds = Counter(
        str(item["decision_kind"]) for item in outputs["group-decisions.jsonl"])
    composition_kinds = Counter(
        str(item["qualification_kind"])
        for item in outputs["composition-receipts.jsonl"])
    result_count = sum(
        len(values) for name, values in outputs.items()
        if name != "evidence.jsonl")
    summary = {
        "composition_qualification_counts": dict(sorted(composition_kinds.items())),
        "composition_receipt_count": len(
            outputs["composition-receipts.jsonl"]),
        "conflict_ledger_count": len(outputs["conflict-ledger.jsonl"]),
        "evidence_count": len(outputs["evidence.jsonl"]),
        "evidence_stance_counts": {
            "REFUTE": evidence_stances["REFUTE"],
            "SUPPORT": evidence_stances["SUPPORT"],
        },
        "generic_rule_count": len(outputs["generic-rules.jsonl"]),
        "group_decision_count": len(outputs["group-decisions.jsonl"]),
        "group_decision_kind_counts": dict(sorted(decision_kinds.items())),
        "regional_rule_count": len(outputs["regional-rules.jsonl"]),
        "result_record_count": result_count,
        "source_phrase_rule_count": len(outputs["source-phrase-rules.jsonl"]),
    }
    expected_counts = normalization_recovery_prefix_output_counts(
        work=work,
        groups=groups,
        compositions=compositions,
        processed_item_count=len(work),
    )
    if expected_counts != (summary["evidence_count"], result_count):
        raise BroadQaExternalDataError(
            "recovery learning prefix/output count 漂移")
    return outputs, summary


def normalization_recovery_output_payloads(
        outputs: dict[str, tuple[dict[str, object], ...]],
        ) -> dict[str, bytes]:
    """把完整输出转为 manifest-last writer 使用的规范 JSONL 字节。"""
    if set(outputs) != {
            name for name, _role, _identity
            in NORMALIZATION_RECOVERY_OUTPUT_FILE_ROLES}:
        raise BroadQaExternalDataError(
            "recovery learning output file inventory 漂移")
    return {
        name: b"".join(canonical_json_line(item) for item in outputs[name])
        for name, _role, _identity in NORMALIZATION_RECOVERY_OUTPUT_FILE_ROLES
    }


__all__ = [
    "NORMALIZATION_RECOVERY_COMPOSITION_RECEIPT_KIND",
    "NORMALIZATION_RECOVERY_CONFLICT_KIND",
    "NORMALIZATION_RECOVERY_EVIDENCE_KIND",
    "NORMALIZATION_RECOVERY_GENERIC_RULE_KIND",
    "NORMALIZATION_RECOVERY_GROUP_DECISION_KIND",
    "NORMALIZATION_RECOVERY_OUTPUT_FILE_ROLES",
    "NORMALIZATION_RECOVERY_PHRASE_RULE_KIND",
    "NORMALIZATION_RECOVERY_REGIONAL_RULE_KIND",
    "derive_normalization_recovery_learning_outputs",
    "normalization_recovery_output_payloads",
    "normalization_recovery_prefix_output_counts",
]
