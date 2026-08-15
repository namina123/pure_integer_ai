"""编译并执行 recovery-v6 strong-whole disabled program。"""
from __future__ import annotations

from collections import defaultdict
import hashlib

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v6_learning_contract import (
    NORMALIZATION_RECOVERY_V6_CONFLICT_VETO_KIND,
    NORMALIZATION_RECOVERY_V6_DEFEATER_KIND,
    NORMALIZATION_RECOVERY_V6_IDENTITY_VETO_KIND,
    NORMALIZATION_RECOVERY_V6_TARGET_INDEX_KIND,
    NORMALIZATION_RECOVERY_V6_TARGET_RULE_KIND,
    RECOVERY_V6_TARGET_POLICY_SCOPE,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes


NORMALIZATION_RECOVERY_V6_PHRASE_PROGRAM_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V6_PHRASE_PROGRAM_V1")
NORMALIZATION_RECOVERY_V6_PHRASE_PROGRAM_STATUS = (
    "CANDIDATE_ONLY_PUBLIC_PRODUCTION_DISABLED")

_PROGRAM_KEYS = {
    "artifact_kind",
    "conflict_vetoes",
    "defeaters",
    "format_version",
    "identity_vetoes",
    "local_execution_allowed",
    "production_enabled",
    "program_sha256",
    "rule_pack_manifest_sha256",
    "source_execution_allowed",
    "status",
    "target_buckets",
    "target_policy_scope",
    "whole_input_exact_only",
}


def _sha256(payload: bytes) -> str:
    """返回规范 program 或结果摘要。"""
    return hashlib.sha256(payload).hexdigest()


def _sha_value(value: object, *, label: str) -> str:
    """核验并返回小写 SHA-256。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(item not in "0123456789abcdef" for item in value)):
        raise BroadQaExternalDataError(f"{label} 非法")
    return value


def _text(value: object, *, label: str) -> str:
    """核验不含 surrogate 的非空运行文本。"""
    if (not isinstance(value, str) or not value
            or any(0xD800 <= ord(item) <= 0xDFFF for item in value)):
        raise BroadQaExternalDataError(f"{label} 非法")
    return value


def _structure_tokens(value: object, *, label: str) -> tuple[str, ...]:
    """核验并冻结调用方或记录中的结构 token 序。"""
    if (not isinstance(value, (list, tuple))
            or any(not isinstance(item, str) for item in value)):
        raise BroadQaExternalDataError(f"{label} 非法")
    return tuple(value)


def _structure_variants(value: object) -> list[list[str]]:
    """核验规范排序且互异的结构 token variants。"""
    if (not isinstance(value, list)
            or any(not isinstance(item, list)
                   or any(not isinstance(token, str) for token in item)
                   for item in value)):
        raise BroadQaExternalDataError("v6 runtime structure variants 非法")
    frozen = [tuple(item) for item in value]
    if frozen != sorted(set(frozen)):
        raise BroadQaExternalDataError(
            "v6 runtime structure variants 顺序/identity 漂移")
    return value


def _program_payload(program: dict[str, object]) -> dict[str, object]:
    """移除自摘要字段后返回 program identity payload。"""
    return {key: value for key, value in program.items()
            if key != "program_sha256"}


def _rule_order(rule: dict[str, object]) -> tuple[object, ...]:
    """固定 exact whole rule 的最长、class 与 id 完全序。"""
    return (
        -len(str(rule["input_text"])),
        str(rule["rule_class"]),
        str(rule["input_text"]),
        str(rule["rule_id"]),
    )


def _validate_rule(rule: dict[str, object]) -> tuple[str, str]:
    """核验一条 v6 approved strong-whole target rule。"""
    if not isinstance(rule, dict):
        raise BroadQaExternalDataError("v6 runtime rule 非字典")
    rule_id = _sha_value(rule.get("rule_id"), label="v6 runtime rule id")
    input_text = _text(rule.get("input_text"), label="v6 runtime input")
    output_text = rule.get("output_text")
    if not isinstance(output_text, str):
        raise BroadQaExternalDataError("v6 runtime output 非法")
    variants = _structure_variants(rule.get("structure_token_variants"))
    predecessor_pack_sha = _sha_value(
        rule.get("predecessor_pack_manifest_sha256"),
        label="v6 rule predecessor pack")
    predecessor_rule_id = _sha_value(
        rule.get("predecessor_rule_id"), label="v6 predecessor rule id")
    protocol_sha = _sha_value(
        rule.get("protocol_manifest_sha256"), label="v6 rule protocol")
    expected_rule_id = _sha256(canonical_json_bytes({
        "input_text": input_text,
        "output_text": output_text,
        "predecessor_pack_manifest_sha256": predecessor_pack_sha,
        "predecessor_rule_id": predecessor_rule_id,
        "protocol_manifest_sha256": protocol_sha,
        "target_policy_scope": RECOVERY_V6_TARGET_POLICY_SCOPE,
    }))
    application = {
        "conflict_exact_input_veto_required": 1,
        "identity_exact_input_veto_required": 1,
        "input_match": "WHOLE_INPUT_EXACT",
        "local_execution_allowed": 0,
        "source_execution_allowed": 0,
        "structure_match_required": int(any(variants)),
        "structure_token_variants": variants,
        "unscoped_execution_allowed": 0,
    }
    families = rule.get("source_families")
    expected_equal = int(len(input_text) == len(output_text))
    if (rule_id != expected_rule_id
            or rule.get("record_kind") != NORMALIZATION_RECOVERY_V6_TARGET_RULE_KIND
            or rule.get("candidate_scope_kind") != "TARGET_CROSS_FAMILY"
            or rule.get("fragment_kind") != "WHOLE_INPUT"
            or rule.get("target_policy_scope")
            != RECOVERY_V6_TARGET_POLICY_SCOPE
            or rule.get("source_execution_family") != ""
            or rule.get("source_execution_policy_scope") != ""
            or rule.get("application_scope") != application
            or not isinstance(families, list)
            or len(families) < 3
            or families != sorted(set(families))
            or rule.get("observed_distinct_source_family_count")
            != len(families)
            or rule.get("required_distinct_source_family_count") != 3
            or not isinstance(rule.get("positive_evidence_ids"), list)
            or not rule["positive_evidence_ids"]
            or not isinstance(rule.get("negative_evidence_ids"), list)
            or not rule["negative_evidence_ids"]
            or not isinstance(rule.get("defeater_ids"), list)
            or not rule["defeater_ids"]
            or rule.get("equal_length") != expected_equal
            or rule.get("variable_length") != 1 - expected_equal
            or rule.get("runtime_state") != "LEARNED_PACK_DISABLED"
            or rule.get("production_enabled") != 0):
        raise BroadQaExternalDataError(
            "v6 runtime rule scope/authority/Evidence/schema 漂移")
    _sha_value(rule.get("predecessor_rule_sha256"),
               label="v6 predecessor rule sha")
    return rule_id, input_text


def _validate_veto(
        record: dict[str, object],
        *,
        expected_kind: str,
        ) -> tuple[str, str]:
    """核验 identity/conflict exact-input veto 并返回 id/input。"""
    if not isinstance(record, dict):
        raise BroadQaExternalDataError("v6 runtime veto 非字典")
    veto_id = _sha_value(record.get("veto_id"), label="v6 runtime veto id")
    input_text = _text(record.get("input_text"), label="v6 veto input")
    predecessor_sha = _sha_value(
        record.get("predecessor_pack_manifest_sha256"),
        label="v6 veto predecessor pack")
    protocol_sha = _sha_value(
        record.get("protocol_manifest_sha256"), label="v6 veto protocol")
    veto_kind = record.get("veto_kind")
    expected_veto_kind = (
        "TRAIN_IDENTITY_EXACT_INPUT"
        if expected_kind == NORMALIZATION_RECOVERY_V6_IDENTITY_VETO_KIND
        else "TRAIN_TARGET_CONFLICT_EXACT_INPUT")
    expected_id = _sha256(canonical_json_bytes({
        "input_text": input_text,
        "predecessor_pack_manifest_sha256": predecessor_sha,
        "protocol_manifest_sha256": protocol_sha,
        "veto_kind": veto_kind,
    }))
    if (veto_id != expected_id
            or record.get("record_kind") != expected_kind
            or record.get("production_enabled") != 0
            or veto_kind != expected_veto_kind):
        raise BroadQaExternalDataError("v6 runtime veto schema 漂移")
    return veto_id, input_text


def _validate_defeater(
        record: dict[str, object],
        rule_by_id: dict[str, dict[str, object]],
        ) -> tuple[str, str]:
    """核验 v6 negative-context defeater 与 approved rule 闭包。"""
    if not isinstance(record, dict):
        raise BroadQaExternalDataError("v6 runtime defeater 非字典")
    defeater_id = _sha_value(
        record.get("defeater_id"), label="v6 runtime defeater id")
    rule_id = _sha_value(record.get("rule_id"), label="v6 defeater rule")
    predecessor_id = _sha_value(
        record.get("predecessor_defeater_id"),
        label="v6 predecessor defeater")
    predecessor_pack_sha = _sha_value(
        record.get("predecessor_pack_manifest_sha256"),
        label="v6 defeater predecessor pack")
    protocol_sha = _sha_value(
        record.get("protocol_manifest_sha256"), label="v6 defeater protocol")
    expected_id = _sha256(canonical_json_bytes({
        "context_signature_id": record.get("context_signature_id"),
        "predecessor_defeater_id": predecessor_id,
        "predecessor_pack_manifest_sha256": predecessor_pack_sha,
        "protocol_manifest_sha256": protocol_sha,
        "rule_id": rule_id,
    }))
    if (defeater_id != expected_id
            or rule_id not in rule_by_id
            or record.get("record_kind") != NORMALIZATION_RECOVERY_V6_DEFEATER_KIND
            or record.get("action") != "BLOCK_WHOLE_RULE_USE_IDENTITY"
            or record.get("production_enabled") != 0
            or record.get("left_boundary") not in {0, 1}
            or record.get("right_boundary") not in {0, 1}
            or not isinstance(record.get("left_context"), str)
            or len(record["left_context"]) > 4
            or not isinstance(record.get("right_context"), str)
            or len(record["right_context"]) > 4
            or not isinstance(record.get("predecessor_refute_evidence_ids"), list)
            or not record["predecessor_refute_evidence_ids"]):
        raise BroadQaExternalDataError("v6 runtime defeater schema 漂移")
    return defeater_id, rule_id


def _ordered_buckets(
        rules: tuple[dict[str, object], ...],
        ) -> list[dict[str, object]]:
    """把 approved whole rule 按首 scalar 分桶并冻结完全序。"""
    buckets: dict[int, list[dict[str, object]]] = defaultdict(list)
    for rule in rules:
        buckets[ord(str(rule["input_text"])[0])].append(rule)
    return [{
        "first_scalar": first_scalar,
        "rules": sorted(values, key=_rule_order),
    } for first_scalar, values in sorted(buckets.items())]


def compile_normalization_recovery_v6_phrase_program(
        *,
        rule_pack_manifest_sha256: str,
        target_whole_rules: tuple[dict[str, object], ...],
        defeaters: tuple[dict[str, object], ...],
        identity_vetoes: tuple[dict[str, object], ...],
        conflict_vetoes: tuple[dict[str, object], ...],
        target_index: tuple[dict[str, object], ...],
        ) -> dict[str, object]:
    """把严格回读的 v6 projection records 编译为禁用态 program。"""
    pack_sha = _sha_value(
        rule_pack_manifest_sha256, label="v6 runtime pack manifest")
    values = (target_whole_rules, defeaters, identity_vetoes,
              conflict_vetoes, target_index)
    if any(not isinstance(value, tuple) for value in values):
        raise BroadQaExternalDataError("v6 runtime compile 输入类型漂移")
    rule_by_id = {}
    input_outputs = set()
    for rule in target_whole_rules:
        rule_id, input_text = _validate_rule(rule)
        semantic = (input_text, str(rule["output_text"]))
        if rule_id in rule_by_id or semantic in input_outputs:
            raise BroadQaExternalDataError("v6 runtime rule identity/semantic 重复")
        rule_by_id[rule_id] = rule
        input_outputs.add(semantic)
    veto_specs = (
        (identity_vetoes, NORMALIZATION_RECOVERY_V6_IDENTITY_VETO_KIND),
        (conflict_vetoes, NORMALIZATION_RECOVERY_V6_CONFLICT_VETO_KIND),
    )
    veto_inputs = []
    for records, kind in veto_specs:
        seen_ids = set()
        seen_inputs = set()
        for record in records:
            veto_id, input_text = _validate_veto(record, expected_kind=kind)
            if veto_id in seen_ids or input_text in seen_inputs:
                raise BroadQaExternalDataError(
                    "v6 runtime veto identity/input 重复")
            seen_ids.add(veto_id)
            seen_inputs.add(input_text)
        veto_inputs.append(seen_inputs)
    if {str(item["input_text"]) for item in target_whole_rules} & (
            veto_inputs[0] | veto_inputs[1]):
        raise BroadQaExternalDataError("v6 runtime approved rule 命中 hard veto")
    defeater_by_id = {}
    defeaters_by_rule: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in defeaters:
        defeater_id, rule_id = _validate_defeater(record, rule_by_id)
        if defeater_id in defeater_by_id:
            raise BroadQaExternalDataError("v6 runtime defeater identity 重复")
        defeater_by_id[defeater_id] = record
        defeaters_by_rule[rule_id].append(record)
    for rule_id, rule in rule_by_id.items():
        declared = sorted(str(item) for item in rule["defeater_ids"])
        actual = sorted(str(item["defeater_id"])
                        for item in defeaters_by_rule.get(rule_id, ()))
        if declared != actual:
            raise BroadQaExternalDataError(
                "v6 runtime rule/defeater 未闭合")
    indexed_rules = set()
    index_ids = set()
    for record in target_index:
        if not isinstance(record, dict):
            raise BroadQaExternalDataError("v6 runtime index 非字典")
        index_id = _sha_value(record.get("index_id"), label="v6 index id")
        rule_id = _sha_value(record.get("rule_id"), label="v6 index rule")
        rule = rule_by_id.get(rule_id)
        expected_index_id = _sha256(canonical_json_bytes({
            "first_scalar": ord(str(rule["input_text"])[0]) if rule else -1,
            "input_scalar_length": len(str(rule["input_text"])) if rule else 0,
            "rule_id": rule_id,
            "target_policy_scope": RECOVERY_V6_TARGET_POLICY_SCOPE,
        }))
        if (rule is None or index_id != expected_index_id
                or index_id in index_ids or rule_id in indexed_rules
                or record.get("record_kind")
                != NORMALIZATION_RECOVERY_V6_TARGET_INDEX_KIND
                or record.get("candidate_input") != rule["input_text"]
                or record.get("first_scalar") != ord(str(rule["input_text"])[0])
                or record.get("input_scalar_length") != len(
                    str(rule["input_text"]))
                or record.get("input_match") != "WHOLE_INPUT_EXACT"
                or record.get("priority_contract")
                != "WHOLE_INPUT_EXACT_ONLY_THEN_IDENTITY"
                or record.get("target_policy_scope")
                != RECOVERY_V6_TARGET_POLICY_SCOPE
                or record.get("structure_token_variants")
                != rule["structure_token_variants"]):
            raise BroadQaExternalDataError("v6 runtime target index 漂移")
        index_ids.add(index_id)
        indexed_rules.add(rule_id)
    if indexed_rules != set(rule_by_id):
        raise BroadQaExternalDataError("v6 runtime target index 未覆盖全部规则")
    program = {
        "artifact_kind": NORMALIZATION_RECOVERY_V6_PHRASE_PROGRAM_KIND,
        "conflict_vetoes": list(conflict_vetoes),
        "defeaters": sorted(defeaters,
                            key=lambda item: str(item["defeater_id"])),
        "format_version": 1,
        "identity_vetoes": list(identity_vetoes),
        "local_execution_allowed": 0,
        "production_enabled": 0,
        "rule_pack_manifest_sha256": pack_sha,
        "source_execution_allowed": 0,
        "status": NORMALIZATION_RECOVERY_V6_PHRASE_PROGRAM_STATUS,
        "target_buckets": _ordered_buckets(target_whole_rules),
        "target_policy_scope": RECOVERY_V6_TARGET_POLICY_SCOPE,
        "whole_input_exact_only": 1,
    }
    return {
        **program,
        "program_sha256": _sha256(canonical_json_bytes(program)),
    }


def _defeater_matches(
        record: dict[str, object],
        text: str,
        start: int,
        end: int,
        ) -> bool:
    """执行 exact-adjacent negative context predicate。"""
    left = str(record["left_context"])
    right = str(record["right_context"])
    left_match = (
        start == 0 if record["left_boundary"] == 1
        else start >= len(left) and text[start - len(left):start] == left)
    right_match = (
        end == len(text) if record["right_boundary"] == 1
        else text[end:end + len(right)] == right)
    return left_match and right_match


def _validated_buckets(
        value: object,
        ) -> dict[int, tuple[dict[str, object], ...]]:
    """核验 program bucket 顺序、首 scalar 与 rule identity。"""
    if not isinstance(value, list):
        raise BroadQaExternalDataError("v6 runtime buckets 非列表")
    result = {}
    rule_ids = set()
    for bucket in value:
        if (not isinstance(bucket, dict)
                or set(bucket) != {"first_scalar", "rules"}
                or type(bucket["first_scalar"]) is not int
                or not isinstance(bucket["rules"], list)
                or bucket["first_scalar"] in result):
            raise BroadQaExternalDataError("v6 runtime bucket 漂移")
        rules = tuple(bucket["rules"])
        if (rules != tuple(sorted(rules, key=_rule_order))
                or any(ord(_validate_rule(item)[1][0]) != bucket["first_scalar"]
                       or str(item["rule_id"]) in rule_ids
                       for item in rules)):
            raise BroadQaExternalDataError("v6 runtime bucket rule 漂移")
        rule_ids.update(str(item["rule_id"]) for item in rules)
        result[bucket["first_scalar"]] = rules
    return result


def _validated_program(
        program: dict[str, object],
        ) -> tuple[dict[int, tuple[dict[str, object], ...]],
                   frozenset[str], frozenset[str],
                   dict[str, tuple[dict[str, object], ...]]]:
    """核验 program 自摘要、veto、index 语义与 defeater 闭包。"""
    if (not isinstance(program, dict) or set(program) != _PROGRAM_KEYS
            or program.get("artifact_kind")
            != NORMALIZATION_RECOVERY_V6_PHRASE_PROGRAM_KIND
            or program.get("status")
            != NORMALIZATION_RECOVERY_V6_PHRASE_PROGRAM_STATUS
            or program.get("format_version") != 1
            or program.get("production_enabled") != 0
            or program.get("local_execution_allowed") != 0
            or program.get("source_execution_allowed") != 0
            or program.get("whole_input_exact_only") != 1
            or program.get("target_policy_scope")
            != RECOVERY_V6_TARGET_POLICY_SCOPE
            or program.get("program_sha256")
            != _sha256(canonical_json_bytes(_program_payload(program)))):
        raise BroadQaExternalDataError("v6 runtime program identity 漂移")
    _sha_value(program.get("rule_pack_manifest_sha256"),
               label="v6 runtime program pack")
    buckets = _validated_buckets(program["target_buckets"])
    rule_by_id = {str(item["rule_id"]): item
                  for rules in buckets.values() for item in rules}
    veto_inputs = []
    for name, kind in (
            ("identity_vetoes", NORMALIZATION_RECOVERY_V6_IDENTITY_VETO_KIND),
            ("conflict_vetoes", NORMALIZATION_RECOVERY_V6_CONFLICT_VETO_KIND)):
        records = program.get(name)
        if not isinstance(records, list):
            raise BroadQaExternalDataError("v6 runtime program veto 漂移")
        inputs = set()
        ids = set()
        for record in records:
            veto_id, input_text = _validate_veto(record, expected_kind=kind)
            if veto_id in ids or input_text in inputs:
                raise BroadQaExternalDataError("v6 runtime program veto 重复")
            ids.add(veto_id)
            inputs.add(input_text)
        veto_inputs.append(frozenset(inputs))
    if {str(item["input_text"]) for item in rule_by_id.values()} & (
            set(veto_inputs[0]) | set(veto_inputs[1])):
        raise BroadQaExternalDataError("v6 runtime program rule/veto 冲突")
    records = program.get("defeaters")
    if not isinstance(records, list):
        raise BroadQaExternalDataError("v6 runtime program defeater 漂移")
    by_rule: dict[str, list[dict[str, object]]] = defaultdict(list)
    seen = set()
    for record in records:
        defeater_id, rule_id = _validate_defeater(record, rule_by_id)
        if defeater_id in seen:
            raise BroadQaExternalDataError("v6 runtime program defeater 重复")
        seen.add(defeater_id)
        by_rule[rule_id].append(record)
    for rule_id, rule in rule_by_id.items():
        if sorted(rule["defeater_ids"]) != sorted(
                str(item["defeater_id"]) for item in by_rule.get(rule_id, ())):
            raise BroadQaExternalDataError(
                "v6 runtime program rule/defeater 未闭合")
    return (
        buckets,
        veto_inputs[0],
        veto_inputs[1],
        {key: tuple(sorted(values,
                           key=lambda item: str(item["defeater_id"])))
         for key, values in by_rule.items()},
    )


def _execute_validated(
        *,
        text: str,
        structure_tokens: tuple[str, ...],
        program_sha256: str,
        buckets: dict[int, tuple[dict[str, object], ...]],
        identity_inputs: frozenset[str],
        conflict_inputs: frozenset[str],
        defeaters_by_rule: dict[str, tuple[dict[str, object], ...]],
        indexed: bool,
        ) -> dict[str, object]:
    """执行 exact whole rule；任一 veto/defeater/未命中均回退 identity。"""
    all_rules = tuple(rule for values in buckets.values() for rule in values)
    candidates = (
        buckets.get(ord(text[0]), ()) if indexed
        else tuple(sorted((item for item in all_rules
                           if str(item["input_text"])[0] == text[0]),
                          key=_rule_order)))
    reasons = []
    chosen = None
    blocked = []
    if text in identity_inputs:
        reasons.append("IDENTITY_VETO")
    elif text in conflict_inputs:
        reasons.append("CONFLICT_VETO")
    else:
        for rule in candidates:
            variants = tuple(tuple(item)
                             for item in rule["structure_token_variants"])
            if (rule["input_text"] != text
                    or (structure_tokens and structure_tokens not in variants)
                    or (not structure_tokens
                        and rule["application_scope"][
                            "structure_match_required"] == 1)):
                continue
            hits = tuple(
                item for item in defeaters_by_rule.get(str(rule["rule_id"]), ())
                if _defeater_matches(item, text, 0, len(text)))
            if hits:
                reasons.append("DEFEATER_BLOCK")
                blocked.extend(str(item["defeater_id"]) for item in hits)
                continue
            chosen = rule
            break
    if chosen is None:
        output = text
        reasons.append("IDENTITY_BACKOFF")
        step = {
            "blocked_defeater_ids": sorted(set(blocked)),
            "input_end": len(text),
            "input_start": 0,
            "mode": "IDENTITY",
            "output_text": text,
            "rule_class": "",
            "rule_id": "",
        }
    else:
        output = str(chosen["output_text"])
        reasons.append("WHOLE_INPUT_EXACT_COMMIT")
        step = {
            "blocked_defeater_ids": sorted(set(blocked)),
            "input_end": len(text),
            "input_start": 0,
            "mode": "WHOLE_INPUT_EXACT",
            "output_text": output,
            "rule_class": chosen["rule_class"],
            "rule_id": chosen["rule_id"],
        }
    payload = {
        "decision_reasons": sorted(set(reasons)),
        "input_text": text,
        "output_text": output,
        "program_sha256": program_sha256,
        "steps": [step],
        "structure_tokens": list(structure_tokens),
        "target_policy_scope": RECOVERY_V6_TARGET_POLICY_SCOPE,
    }
    return {**payload, "result_sha256": _sha256(canonical_json_bytes(payload))}


def _execute(
        program: dict[str, object],
        text: str,
        *,
        structure_tokens: tuple[str, ...],
        indexed: bool,
        ) -> dict[str, object]:
    """核验 program/input 后执行 indexed 或 reference 路径。"""
    source = _text(text, label="v6 runtime input")
    tokens = _structure_tokens(
        structure_tokens, label="v6 runtime structure tokens")
    buckets, identities, conflicts, defeaters = _validated_program(program)
    return _execute_validated(
        text=source,
        structure_tokens=tokens,
        program_sha256=str(program["program_sha256"]),
        buckets=buckets,
        identity_inputs=identities,
        conflict_inputs=conflicts,
        defeaters_by_rule=defeaters,
        indexed=indexed,
    )


def execute_normalization_recovery_v6_phrase_program(
        program: dict[str, object],
        text: str,
        *,
        structure_tokens: tuple[str, ...] = (),
        ) -> dict[str, object]:
    """执行按首 scalar 分桶的 v6 indexed whole-only program。"""
    return _execute(
        program, text, structure_tokens=structure_tokens, indexed=True)


def reference_normalization_recovery_v6_phrase_program(
        program: dict[str, object],
        text: str,
        *,
        structure_tokens: tuple[str, ...] = (),
        ) -> dict[str, object]:
    """执行不使用首 scalar index 的 v6 reference whole-only program。"""
    return _execute(
        program, text, structure_tokens=structure_tokens, indexed=False)


def _execute_batch(
        program: dict[str, object],
        texts: tuple[str, ...],
        *,
        structure_tokens: tuple[tuple[str, ...], ...] | None,
        indexed: bool,
        ) -> tuple[dict[str, object], ...]:
    """一次核验 program 后批量执行同一 v6 whole-only program。"""
    if not isinstance(texts, tuple):
        raise BroadQaExternalDataError("v6 runtime batch 输入类型漂移")
    token_values = (((),) * len(texts)
                    if structure_tokens is None else structure_tokens)
    if (not isinstance(token_values, tuple)
            or len(token_values) != len(texts)):
        raise BroadQaExternalDataError("v6 runtime batch structure 数量漂移")
    buckets, identities, conflicts, defeaters = _validated_program(program)
    return tuple(_execute_validated(
        text=_text(text, label="v6 runtime input"),
        structure_tokens=_structure_tokens(
            tokens, label="v6 runtime structure tokens"),
        program_sha256=str(program["program_sha256"]),
        buckets=buckets,
        identity_inputs=identities,
        conflict_inputs=conflicts,
        defeaters_by_rule=defeaters,
        indexed=indexed,
    ) for text, tokens in zip(texts, token_values))


def execute_normalization_recovery_v6_phrase_batch(
        program: dict[str, object],
        texts: tuple[str, ...],
        *,
        structure_tokens: tuple[tuple[str, ...], ...] | None = None,
        ) -> tuple[dict[str, object], ...]:
    """批量执行 v6 indexed whole-only program。"""
    return _execute_batch(
        program, texts, structure_tokens=structure_tokens, indexed=True)


def reference_normalization_recovery_v6_phrase_batch(
        program: dict[str, object],
        texts: tuple[str, ...],
        *,
        structure_tokens: tuple[tuple[str, ...], ...] | None = None,
        ) -> tuple[dict[str, object], ...]:
    """批量执行 v6 reference whole-only program。"""
    return _execute_batch(
        program, texts, structure_tokens=structure_tokens, indexed=False)


__all__ = [
    "NORMALIZATION_RECOVERY_V6_PHRASE_PROGRAM_KIND",
    "NORMALIZATION_RECOVERY_V6_PHRASE_PROGRAM_STATUS",
    "compile_normalization_recovery_v6_phrase_program",
    "execute_normalization_recovery_v6_phrase_batch",
    "execute_normalization_recovery_v6_phrase_program",
    "reference_normalization_recovery_v6_phrase_batch",
    "reference_normalization_recovery_v6_phrase_program",
]
