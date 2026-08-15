"""从 recovery-v5 TRAIN outputs 派生 v6 strong-whole policy records。"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_learning_contract import (
    NORMALIZATION_RECOVERY_V5_OUTPUT_FILE_ROLES,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_phrase_runtime import (
    compile_normalization_recovery_v5_phrase_program,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v6_learning_contract import (
    NORMALIZATION_RECOVERY_V6_CONFLICT_VETO_KIND,
    NORMALIZATION_RECOVERY_V6_DEFEATER_KIND,
    NORMALIZATION_RECOVERY_V6_DEFERRED_RULE_KIND,
    NORMALIZATION_RECOVERY_V6_DEFER_REASONS,
    NORMALIZATION_RECOVERY_V6_IDENTITY_VETO_KIND,
    NORMALIZATION_RECOVERY_V6_OUTPUT_FILE_ROLES,
    NORMALIZATION_RECOVERY_V6_TARGET_INDEX_KIND,
    NORMALIZATION_RECOVERY_V6_TARGET_RULE_KIND,
    RECOVERY_V6_TARGET_POLICY_SCOPE,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    canonical_json_line,
)


def _sha256(payload: bytes) -> str:
    """返回规范 projection 记录或集合摘要。"""
    return hashlib.sha256(payload).hexdigest()


def _sha_value(value: object, *, label: str) -> str:
    """核验并返回小写 SHA-256。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(item not in "0123456789abcdef" for item in value)):
        raise BroadQaExternalDataError(f"{label} 非法")
    return value


def _record_id(identity: dict[str, object]) -> str:
    """从完整语义 identity 形成稳定记录 id。"""
    return _sha256(canonical_json_bytes(identity))


def _validate_v5_outputs(
        *,
        predecessor_pack_manifest_sha256: str,
        outputs: dict[str, tuple[dict[str, object], ...]],
        ) -> str:
    """核验 v5 九文件 inventory，并借 frozen compiler 复核 rule 闭包。"""
    pack_sha = _sha_value(
        predecessor_pack_manifest_sha256,
        label="v6 predecessor pack manifest")
    expected = {name for name, _role, _identity
                in NORMALIZATION_RECOVERY_V5_OUTPUT_FILE_ROLES}
    if (not isinstance(outputs, dict) or set(outputs) != expected
            or any(not isinstance(value, tuple) for value in outputs.values())):
        raise BroadQaExternalDataError("v6 predecessor output inventory 漂移")
    compile_normalization_recovery_v5_phrase_program(
        rule_pack_manifest_sha256=pack_sha,
        target_phrase_rules=outputs["target-phrase-rules.jsonl"],
        source_phrase_rules=outputs["source-phrase-rules.jsonl"],
        defeaters=outputs["defeaters.jsonl"],
        target_overlap_index=outputs["target-overlap-index.jsonl"],
        source_overlap_index=outputs["source-overlap-index.jsonl"],
    )
    for name, identity_key in (
            ("evidence.jsonl", "evidence_id"),
            ("conflict-ledger.jsonl", "conflict_id"),
            ("deferred-groups.jsonl", "deferred_id"),
            ("identity-observations.jsonl", "identity_record_id")):
        identities = []
        for record in outputs[name]:
            if (not isinstance(record, dict)
                    or not isinstance(record.get(identity_key), str)
                    or not record[identity_key]):
                raise BroadQaExternalDataError(
                    f"v6 predecessor {name} identity 漂移")
            identities.append(str(record[identity_key]))
        if len(set(identities)) != len(identities):
            raise BroadQaExternalDataError(
                f"v6 predecessor {name} identity 重复")
    return pack_sha


def _identity_vetoes(
        *,
        protocol_sha: str,
        predecessor_pack_sha: str,
        records: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, object], ...]:
    """按 exact input 聚合 TRAIN identity observations。"""
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        input_text = record.get("input_text")
        if (not isinstance(input_text, str) or not input_text
                or record.get("output_text") != input_text
                or not isinstance(record.get("observation_id"), str)
                or not isinstance(record.get("source_family"), str)
                or not isinstance(record.get("structure_tokens"), list)):
            raise BroadQaExternalDataError(
                "v6 predecessor identity observation 漂移")
        grouped[input_text].append(record)
    values = []
    for input_text, items in sorted(grouped.items()):
        identity = {
            "input_text": input_text,
            "predecessor_pack_manifest_sha256": predecessor_pack_sha,
            "protocol_manifest_sha256": protocol_sha,
            "veto_kind": "TRAIN_IDENTITY_EXACT_INPUT",
        }
        values.append({
            **identity,
            "format_version": 1,
            "observation_ids": sorted(
                str(item["observation_id"]) for item in items),
            "predecessor_identity_record_ids": sorted(
                str(item["identity_record_id"]) for item in items),
            "production_enabled": 0,
            "record_kind": NORMALIZATION_RECOVERY_V6_IDENTITY_VETO_KIND,
            "source_families": sorted({
                str(item["source_family"]) for item in items}),
            "structure_token_variants": [list(value) for value in sorted({
                tuple(item["structure_tokens"]) for item in items})],
            "veto_id": _record_id(identity),
        })
    return tuple(values)


def _conflict_vetoes(
        *,
        protocol_sha: str,
        predecessor_pack_sha: str,
        records: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, object], ...]:
    """按 exact input 聚合 target/NONE predecessor conflicts。"""
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        scope = record.get("candidate_scope_kind")
        input_text = record.get("input_text")
        if (scope not in {"NONE", "TARGET_CROSS_FAMILY"}
                or not isinstance(input_text, str) or not input_text
                or not isinstance(record.get("conflict_kind"), str)
                or not isinstance(record.get("group_id"), str)):
            if scope in {"NONE", "TARGET_CROSS_FAMILY"}:
                raise BroadQaExternalDataError(
                    "v6 predecessor target conflict 漂移")
            continue
        grouped[input_text].append(record)
    values = []
    for input_text, items in sorted(grouped.items()):
        identity = {
            "input_text": input_text,
            "predecessor_pack_manifest_sha256": predecessor_pack_sha,
            "protocol_manifest_sha256": protocol_sha,
            "veto_kind": "TRAIN_TARGET_CONFLICT_EXACT_INPUT",
        }
        values.append({
            **identity,
            "conflict_ids": sorted(
                str(item["conflict_id"]) for item in items),
            "conflict_kinds": sorted({
                str(item["conflict_kind"]) for item in items}),
            "format_version": 1,
            "group_ids": sorted({str(item["group_id"]) for item in items}),
            "production_enabled": 0,
            "record_kind": NORMALIZATION_RECOVERY_V6_CONFLICT_VETO_KIND,
            "rule_classes": sorted({
                str(item["rule_class"]) for item in items
                if item.get("rule_class")}),
            "veto_id": _record_id(identity),
        })
    return tuple(values)


def _defer_reasons(
        rule: dict[str, object],
        *,
        identity_inputs: frozenset[str],
        conflict_inputs: frozenset[str],
        ) -> tuple[str, ...]:
    """按冻结 v6 policy 返回 predecessor rule 的全部 defer 理由。"""
    reasons = []
    if rule["candidate_scope_kind"] == "SOURCE_ONLY":
        reasons.append("SOURCE_EXECUTION_DISABLED")
    if rule["fragment_kind"] != "WHOLE_INPUT":
        reasons.append("LOCAL_EXECUTION_DISABLED")
    if (rule["candidate_scope_kind"] == "TARGET_CROSS_FAMILY"
            and rule["fragment_kind"] == "WHOLE_INPUT"
            and len(rule["source_families"]) < 3):
        reasons.append("WHOLE_THREE_FAMILY_REQUIRED")
    input_text = str(rule["input_text"])
    if (rule["candidate_scope_kind"] == "TARGET_CROSS_FAMILY"
            and rule["fragment_kind"] == "WHOLE_INPUT"):
        if input_text in identity_inputs:
            reasons.append("WHOLE_IDENTITY_VETO")
        if input_text in conflict_inputs:
            reasons.append("WHOLE_CONFLICT_VETO")
    reason_set = set(reasons)
    return tuple(reason for reason in NORMALIZATION_RECOVERY_V6_DEFER_REASONS
                 if reason in reason_set)


def _v6_defeaters(
        *,
        protocol_sha: str,
        predecessor_pack_sha: str,
        predecessor_rule: dict[str, object],
        rule_id: str,
        predecessor_by_id: dict[str, dict[str, object]],
        ) -> tuple[dict[str, object], ...]:
    """把 approved predecessor rule 的全部 negative defeater 重绑定到 v6。"""
    values = []
    for predecessor_id in predecessor_rule["defeater_ids"]:
        record = predecessor_by_id.get(str(predecessor_id))
        if record is None or record["rule_id"] != predecessor_rule["rule_id"]:
            raise BroadQaExternalDataError(
                "v6 approved rule predecessor defeater 未闭合")
        identity = {
            "context_signature_id": record["context_signature_id"],
            "predecessor_defeater_id": record["defeater_id"],
            "predecessor_pack_manifest_sha256": predecessor_pack_sha,
            "protocol_manifest_sha256": protocol_sha,
            "rule_id": rule_id,
        }
        values.append({
            **identity,
            "action": "BLOCK_WHOLE_RULE_USE_IDENTITY",
            "defeater_id": _record_id(identity),
            "format_version": 1,
            "left_boundary": record["left_boundary"],
            "left_context": record["left_context"],
            "predecessor_refute_evidence_ids": record[
                "refute_evidence_ids"],
            "production_enabled": 0,
            "record_kind": NORMALIZATION_RECOVERY_V6_DEFEATER_KIND,
            "right_boundary": record["right_boundary"],
            "right_context": record["right_context"],
        })
    return tuple(sorted(values, key=lambda item: str(item["defeater_id"])))


def _approved_rule(
        *,
        protocol_sha: str,
        predecessor_pack_sha: str,
        predecessor_rule: dict[str, object],
        predecessor_defeater_by_id: dict[str, dict[str, object]],
        ) -> tuple[dict[str, object], tuple[dict[str, object], ...],
                   dict[str, object]]:
    """把一条通过全部 v6 hard gate 的 whole target rule 物化。"""
    identity = {
        "input_text": predecessor_rule["input_text"],
        "output_text": predecessor_rule["output_text"],
        "predecessor_pack_manifest_sha256": predecessor_pack_sha,
        "predecessor_rule_id": predecessor_rule["rule_id"],
        "protocol_manifest_sha256": protocol_sha,
        "target_policy_scope": RECOVERY_V6_TARGET_POLICY_SCOPE,
    }
    rule_id = _record_id(identity)
    defeaters = _v6_defeaters(
        protocol_sha=protocol_sha,
        predecessor_pack_sha=predecessor_pack_sha,
        predecessor_rule=predecessor_rule,
        rule_id=rule_id,
        predecessor_by_id=predecessor_defeater_by_id,
    )
    rule = {
        **identity,
        "application_scope": {
            "conflict_exact_input_veto_required": 1,
            "identity_exact_input_veto_required": 1,
            "input_match": "WHOLE_INPUT_EXACT",
            "local_execution_allowed": 0,
            "source_execution_allowed": 0,
            "structure_match_required": int(any(
                predecessor_rule["structure_token_variants"])),
            "structure_token_variants": predecessor_rule[
                "structure_token_variants"],
            "unscoped_execution_allowed": 0,
        },
        "candidate_scope_kind": "TARGET_CROSS_FAMILY",
        "defeater_ids": [str(item["defeater_id"]) for item in defeaters],
        "equal_length": predecessor_rule["equal_length"],
        "format_version": 1,
        "fragment_kind": "WHOLE_INPUT",
        "license_ids": predecessor_rule["license_ids"],
        "negative_evidence_ids": predecessor_rule["negative_evidence_ids"],
        "observed_distinct_source_family_count": len(
            predecessor_rule["source_families"]),
        "positive_evidence_ids": predecessor_rule["positive_evidence_ids"],
        "predecessor_candidate_id": predecessor_rule["candidate_id"],
        "predecessor_group_id": predecessor_rule["group_id"],
        "predecessor_rule_sha256": _sha256(canonical_json_bytes(
            predecessor_rule)),
        "production_enabled": 0,
        "record_kind": NORMALIZATION_RECOVERY_V6_TARGET_RULE_KIND,
        "required_distinct_source_family_count": 3,
        "rule_class": predecessor_rule["rule_class"],
        "rule_id": rule_id,
        "runtime_state": "LEARNED_PACK_DISABLED",
        "source_families": predecessor_rule["source_families"],
        "source_execution_family": "",
        "source_execution_policy_scope": "",
        "source_fragment_ids": predecessor_rule["source_fragment_ids"],
        "source_policy_scopes": predecessor_rule["source_policy_scopes"],
        "structure_token_variants": predecessor_rule[
            "structure_token_variants"],
        "variable_length": predecessor_rule["variable_length"],
    }
    index_identity = {
        "first_scalar": ord(str(rule["input_text"])[0]),
        "input_scalar_length": len(str(rule["input_text"])),
        "rule_id": rule_id,
        "target_policy_scope": RECOVERY_V6_TARGET_POLICY_SCOPE,
    }
    index = {
        **index_identity,
        "candidate_input": rule["input_text"],
        "format_version": 1,
        "index_id": _record_id(index_identity),
        "input_match": "WHOLE_INPUT_EXACT",
        "priority_contract": "WHOLE_INPUT_EXACT_ONLY_THEN_IDENTITY",
        "record_kind": NORMALIZATION_RECOVERY_V6_TARGET_INDEX_KIND,
        "structure_token_variants": rule["structure_token_variants"],
    }
    return rule, defeaters, index


def _deferred_rule(
        *,
        protocol_sha: str,
        predecessor_pack_sha: str,
        predecessor_rule: dict[str, object],
        reasons: tuple[str, ...],
        ) -> dict[str, object]:
    """保留一条未批准 predecessor rule 的完整恢复理由。"""
    if not reasons:
        raise BroadQaExternalDataError("v6 deferred rule 缺少理由")
    identity = {
        "predecessor_pack_manifest_sha256": predecessor_pack_sha,
        "predecessor_rule_id": predecessor_rule["rule_id"],
        "protocol_manifest_sha256": protocol_sha,
    }
    return {
        **identity,
        "candidate_scope_kind": predecessor_rule["candidate_scope_kind"],
        "defer_reasons": list(reasons),
        "deferred_id": _record_id(identity),
        "format_version": 1,
        "fragment_kind": predecessor_rule["fragment_kind"],
        "input_text": predecessor_rule["input_text"],
        "output_text": predecessor_rule["output_text"],
        "predecessor_rule_sha256": _sha256(canonical_json_bytes(
            predecessor_rule)),
        "production_enabled": 0,
        "record_kind": NORMALIZATION_RECOVERY_V6_DEFERRED_RULE_KIND,
        "rule_class": predecessor_rule["rule_class"],
        "source_families": predecessor_rule["source_families"],
    }


def derive_normalization_recovery_v6_learning_outputs(
        *,
        protocol_manifest_sha256: str,
        predecessor_pack_manifest_sha256: str,
        predecessor_outputs: dict[str, tuple[dict[str, object], ...]],
        ) -> tuple[dict[str, tuple[dict[str, object], ...]], dict[str, object]]:
    """按冻结 strong-whole policy 精确分账全部 v5 executable rules。"""
    protocol_sha = _sha_value(
        protocol_manifest_sha256, label="v6 protocol manifest")
    predecessor_sha = _validate_v5_outputs(
        predecessor_pack_manifest_sha256=predecessor_pack_manifest_sha256,
        outputs=predecessor_outputs,
    )
    identity_vetoes = _identity_vetoes(
        protocol_sha=protocol_sha,
        predecessor_pack_sha=predecessor_sha,
        records=predecessor_outputs["identity-observations.jsonl"],
    )
    conflict_vetoes = _conflict_vetoes(
        protocol_sha=protocol_sha,
        predecessor_pack_sha=predecessor_sha,
        records=predecessor_outputs["conflict-ledger.jsonl"],
    )
    identity_inputs = frozenset(str(item["input_text"])
                                for item in identity_vetoes)
    conflict_inputs = frozenset(str(item["input_text"])
                                for item in conflict_vetoes)
    predecessor_defeaters = {
        str(item["defeater_id"]): item
        for item in predecessor_outputs["defeaters.jsonl"]}
    if len(predecessor_defeaters) != len(
            predecessor_outputs["defeaters.jsonl"]):
        raise BroadQaExternalDataError(
            "v6 predecessor defeater identity 重复")
    predecessor_rules = (
        predecessor_outputs["target-phrase-rules.jsonl"]
        + predecessor_outputs["source-phrase-rules.jsonl"])
    approved = []
    defeaters = []
    deferred = []
    indexes = []
    approved_predecessors = set()
    deferred_predecessors = set()
    for predecessor_rule in predecessor_rules:
        reasons = _defer_reasons(
            predecessor_rule,
            identity_inputs=identity_inputs,
            conflict_inputs=conflict_inputs,
        )
        if reasons:
            record = _deferred_rule(
                protocol_sha=protocol_sha,
                predecessor_pack_sha=predecessor_sha,
                predecessor_rule=predecessor_rule,
                reasons=reasons,
            )
            deferred.append(record)
            deferred_predecessors.add(str(predecessor_rule["rule_id"]))
            continue
        if (predecessor_rule["candidate_scope_kind"]
                != "TARGET_CROSS_FAMILY"
                or predecessor_rule["fragment_kind"] != "WHOLE_INPUT"
                or len(predecessor_rule["source_families"]) < 3):
            raise BroadQaExternalDataError(
                "v6 projection 未知 predecessor disposition")
        rule, rule_defeaters, index = _approved_rule(
            protocol_sha=protocol_sha,
            predecessor_pack_sha=predecessor_sha,
            predecessor_rule=predecessor_rule,
            predecessor_defeater_by_id=predecessor_defeaters,
        )
        approved.append(rule)
        defeaters.extend(rule_defeaters)
        indexes.append(index)
        approved_predecessors.add(str(predecessor_rule["rule_id"]))
    predecessor_ids = {str(item["rule_id"]) for item in predecessor_rules}
    if (approved_predecessors & deferred_predecessors
            or approved_predecessors | deferred_predecessors
            != predecessor_ids
            or {str(item["input_text"]) for item in approved}
            & (set(identity_inputs) | set(conflict_inputs))
            or any(item["fragment_kind"] != "WHOLE_INPUT"
                   or item["observed_distinct_source_family_count"] < 3
                   for item in approved)):
        raise BroadQaExternalDataError("v6 projection rule 分账/硬门漂移")
    raw_outputs = {
        "target-whole-rules.jsonl": approved,
        "defeaters.jsonl": defeaters,
        "identity-vetoes.jsonl": list(identity_vetoes),
        "conflict-vetoes.jsonl": list(conflict_vetoes),
        "deferred-rules.jsonl": deferred,
        "target-index.jsonl": indexes,
    }
    outputs = {}
    for name, _role, identity_key in NORMALIZATION_RECOVERY_V6_OUTPUT_FILE_ROLES:
        values = tuple(sorted(
            raw_outputs[name], key=lambda item: str(item[identity_key])))
        if len({str(item[identity_key]) for item in values}) != len(values):
            raise BroadQaExternalDataError(
                f"v6 projection {name} identity 重复")
        outputs[name] = values
    reason_counts = Counter(
        reason for item in outputs["deferred-rules.jsonl"]
        for reason in item["defer_reasons"])
    class_counts = Counter(str(item["rule_class"])
                           for item in outputs["target-whole-rules.jsonl"])
    summary = {
        "approved_predecessor_rule_ids_sha256": _sha256(
            canonical_json_bytes(sorted(approved_predecessors))),
        "approved_rule_class_counts": {
            key: class_counts[key] for key in sorted(class_counts)},
        "approved_target_rule_count": len(
            outputs["target-whole-rules.jsonl"]),
        "conflict_veto_count": len(outputs["conflict-vetoes.jsonl"]),
        "defer_reason_counts": {
            key: reason_counts[key] for key in NORMALIZATION_RECOVERY_V6_DEFER_REASONS},
        "deferred_predecessor_rule_ids_sha256": _sha256(
            canonical_json_bytes(sorted(deferred_predecessors))),
        "deferred_rule_count": len(outputs["deferred-rules.jsonl"]),
        "executable_local_rule_count": 0,
        "executable_source_rule_count": 0,
        "identity_veto_count": len(outputs["identity-vetoes.jsonl"]),
        "predecessor_conflict_ledger_sha256": _sha256(canonical_json_bytes(
            predecessor_outputs["conflict-ledger.jsonl"])),
        "predecessor_deferred_groups_sha256": _sha256(canonical_json_bytes(
            predecessor_outputs["deferred-groups.jsonl"])),
        "predecessor_pack_manifest_sha256": predecessor_sha,
        "predecessor_rule_count": len(predecessor_rules),
        "production_enabled": 0,
        "protocol_manifest_sha256": protocol_sha,
        "target_index_count": len(outputs["target-index.jsonl"]),
        "target_policy_scope": RECOVERY_V6_TARGET_POLICY_SCOPE,
    }
    if (summary["approved_target_rule_count"] + summary["deferred_rule_count"]
            != summary["predecessor_rule_count"]
            or summary["approved_target_rule_count"]
            != summary["target_index_count"]):
        raise BroadQaExternalDataError("v6 projection summary 未闭合")
    return outputs, summary


def normalization_recovery_v6_output_payloads(
        outputs: dict[str, tuple[dict[str, object], ...]],
        ) -> dict[str, bytes]:
    """把六份 projection outputs 编码为规范 JSONL。"""
    expected = {name for name, _role, _identity
                in NORMALIZATION_RECOVERY_V6_OUTPUT_FILE_ROLES}
    if set(outputs) != expected:
        raise BroadQaExternalDataError("v6 projection output inventory 漂移")
    return {
        name: b"".join(canonical_json_line(item) for item in outputs[name])
        for name, _role, _identity in NORMALIZATION_RECOVERY_V6_OUTPUT_FILE_ROLES
    }


__all__ = [
    "derive_normalization_recovery_v6_learning_outputs",
    "normalization_recovery_v6_output_payloads",
]
