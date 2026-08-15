"""从 v8 LOSO 规则重建标签盲全训练候选并执行整串转换。"""
from __future__ import annotations

from collections import defaultdict
import hashlib

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_localization_structure import (
    localization_structure_tokens,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_training_records import (
    V8_TRAIN_FAMILIES,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes


NORMALIZATION_RECOVERY_V8_CANDIDATE_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V8_FULL_TRAIN_CANDIDATE_V1")
NORMALIZATION_RECOVERY_V8_CANDIDATE_STATUS = (
    "LABEL_BLIND_FULL_TRAIN_RULES_DISABLED_FORMAL_NOT_RUN")
NORMALIZATION_RECOVERY_V8_TARGET_POLICY_SCOPE = (
    "ZH_TW_TO_ZH_CN_EXTERNAL_LOCALIZATION_V8")

V8_CANDIDATE_RULE_COUNTS = {
    "identity_veto_rules": 19,
    "orthographic_rules": 62,
    "source_conditioned_rules": 49,
    "structure_obligations": 11,
}

_SEMANTIC_FIELDS = {
    "ORTHOGRAPHIC_ATOM": ("authorization_id", "candidate_id", "input_atom",
                           "output_atom"),
    "SOURCE_CONDITIONED_LEXICAL_ATOM": (
        "authorization_id", "candidate_id", "input_text",
        "official_source_text", "output_text"),
    "LAYOUT_MORPHOLOGY_OBLIGATION": (
        "authorization_id", "candidate_id", "structure_tokens"),
    "IDENTITY_VETO": ("candidate_id", "input_text", "output_text"),
}


def _sha256(value: object) -> str:
    """返回候选、规则或结果的确定性 SHA-256。"""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _families(value: object, *, label: str) -> list[str]:
    """读取三家训练来源内的有序唯一 family 集合。"""
    if (not isinstance(value, list) or value != sorted(set(value))
            or any(item not in V8_TRAIN_FAMILIES for item in value)):
        raise BroadQaExternalDataError(f"v8 candidate {label} 漂移")
    return value


def _rule_semantics(kind: str, record: dict[str, object]) -> dict[str, object]:
    """提取一条方向规则中不依赖 held-out 方向的语义字段。"""
    fields = _SEMANTIC_FIELDS[kind]
    if any(field not in record for field in fields):
        raise BroadQaExternalDataError("v8 candidate rule semantic 缺失")
    value = {field: record[field] for field in fields}
    if (kind == "ORTHOGRAPHIC_ATOM"
            and (not isinstance(value["input_atom"], str)
                 or len(str(value["input_atom"])) != 1
                 or not isinstance(value["output_atom"], str)
                 or len(str(value["output_atom"])) != 1)
            or kind == "SOURCE_CONDITIONED_LEXICAL_ATOM"
            and any(not isinstance(value[field], str) or not value[field]
                    for field in ("input_text", "official_source_text",
                                  "output_text"))
            or kind == "LAYOUT_MORPHOLOGY_OBLIGATION"
            and (not isinstance(value["structure_tokens"], list)
                 or not value["structure_tokens"]
                 or any(not isinstance(item, str) or not item
                        for item in value["structure_tokens"]))
            or kind == "IDENTITY_VETO"
            and (not isinstance(value["input_text"], str)
                 or not value["input_text"]
                 or value["output_text"] != value["input_text"])):
        raise BroadQaExternalDataError("v8 candidate rule semantic 非法")
    return value


def _deduplicate_direction_rules(
        kind: str, records: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, object], ...]:
    """按 authorization/candidate identity 合并 LOSO 方向副本。"""
    identity_field = "candidate_id" if kind == "IDENTITY_VETO" else (
        "authorization_id")
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        identity = record.get(identity_field) if isinstance(record, dict) else None
        if (not isinstance(identity, str) or len(identity) != 64
                or record.get("production_enabled") != 0):
            raise BroadQaExternalDataError("v8 candidate direction rule 漂移")
        grouped[identity].append(record)
    values = []
    for identity in sorted(grouped):
        directions = grouped[identity]
        semantic = _rule_semantics(kind, directions[0])
        if any(_rule_semantics(kind, item) != semantic for item in directions):
            raise BroadQaExternalDataError("v8 candidate direction semantic 冲突")
        support = sorted({family for item in directions
                          for family in _families(
                              item.get("train_support_families"),
                              label="train support")})
        held_out = sorted(str(item.get("held_out_family"))
                          for item in directions)
        if len(support) == 3:
            valid = (held_out == list(V8_TRAIN_FAMILIES)
                     and all(_families(
                         item.get("train_support_families"),
                         label="support-3 direction")
                         == [family for family in support
                             if family != item.get("held_out_family")]
                         for item in directions))
        elif len(support) == 2:
            valid = (len(directions) == 1
                     and held_out == [family for family in V8_TRAIN_FAMILIES
                                      if family not in support]
                     and _families(directions[0].get(
                         "train_support_families"), label="support-2 direction")
                     == support)
        else:
            valid = False
        if not valid:
            raise BroadQaExternalDataError("v8 candidate LOSO authority 漂移")
        rule_identity = {
            "rule_kind": kind,
            "source_identity": identity,
        }
        values.append({
            **semantic,
            "candidate_rule_id": _sha256(rule_identity),
            "format_version": 1,
            "production_enabled": 0,
            "rule_kind": kind,
            "training_support_families": support,
            "training_support_family_count": len(support),
        })
    return tuple(values)


def compile_normalization_recovery_v8_candidate(
        *, rule_pack_manifest: dict[str, object],
        rule_outputs: dict[str, tuple[dict[str, object], ...]],
        training_audit_manifest_sha256: str,
        evaluation_commitment_manifest_sha256: str,
        ) -> dict[str, object]:
    """把方向化 TRAIN 规则编译为去重、禁用态全训练候选。"""
    if (rule_pack_manifest.get("status")
            != "FAMILY_LOSO_FROZEN_NOT_FORMAL_NOT_DEPLOYED"
            or rule_pack_manifest.get("production_enabled") != 0
            or rule_pack_manifest.get("mastery_claimed") != 0):
        raise BroadQaExternalDataError("v8 candidate rule pack state 非法")
    specs = (
        ("orthographic_rules", "ORTHOGRAPHIC_ATOM",
         "orthographic-rules.jsonl"),
        ("source_conditioned_rules", "SOURCE_CONDITIONED_LEXICAL_ATOM",
         "source-conditioned-lexical-rules.jsonl"),
        ("structure_obligations", "LAYOUT_MORPHOLOGY_OBLIGATION",
         "layout-morphology-obligations.jsonl"),
        ("identity_veto_rules", "IDENTITY_VETO",
         "identity-veto-rules.jsonl"),
    )
    inventories = {}
    for name, kind, filename in specs:
        records = rule_outputs.get(filename)
        if not isinstance(records, tuple):
            raise BroadQaExternalDataError("v8 candidate rule inventory 缺失")
        inventories[name] = list(_deduplicate_direction_rules(kind, records))
        if len(inventories[name]) != V8_CANDIDATE_RULE_COUNTS[name]:
            raise BroadQaExternalDataError("v8 candidate deduplicated count 漂移")
    program = {
        "applicability_contract": {
            "full_frozen_denominator_judged": 1,
            "unknown_has_no_committed_output": 1,
            "wrong_is_any_incorrect_committed_output": 1,
        },
        "artifact_kind": NORMALIZATION_RECOVERY_V8_CANDIDATE_KIND,
        "evaluation_commitment_manifest_sha256": (
            evaluation_commitment_manifest_sha256),
        "format_version": 1,
        "inventories": inventories,
        "mastery_claimed": 0,
        "production_enabled": 0,
        "rule_pack_manifest_sha256": rule_pack_manifest["manifest_sha256"],
        "runtime_order": [
            "STRUCTURE_PRESERVATION_HARD_GATE",
            "SOURCE_CONDITIONED_LEXICAL_EXACT",
            "ORTHOGRAPHIC_ATOM_FULL_TEXT_COMPOSITION",
            "STABLE_COPY_IDENTITY_VETO",
            "UNKNOWN",
        ],
        "status": NORMALIZATION_RECOVERY_V8_CANDIDATE_STATUS,
        "target_policy_scope": NORMALIZATION_RECOVERY_V8_TARGET_POLICY_SCOPE,
        "teacher_api_llm_call_count": 0,
        "training_audit_manifest_sha256": training_audit_manifest_sha256,
    }
    return {**program, "candidate_program_sha256": _sha256(program)}


def _validate_candidate(candidate: dict[str, object]) -> dict[str, object]:
    """核验候选静态边界并返回四个规则 inventory。"""
    inventories = candidate.get("inventories") if isinstance(
        candidate, dict) else None
    program = {key: value for key, value in candidate.items()
               if key != "candidate_program_sha256"} if isinstance(
                   candidate, dict) else {}
    if (candidate.get("artifact_kind")
            != NORMALIZATION_RECOVERY_V8_CANDIDATE_KIND
            or candidate.get("status") != NORMALIZATION_RECOVERY_V8_CANDIDATE_STATUS
            or candidate.get("production_enabled") != 0
            or candidate.get("mastery_claimed") != 0
            or candidate.get("target_policy_scope")
            != NORMALIZATION_RECOVERY_V8_TARGET_POLICY_SCOPE
            or candidate.get("candidate_program_sha256") != _sha256(program)
            or not isinstance(inventories, dict)
            or set(inventories) != set(V8_CANDIDATE_RULE_COUNTS)
            or any(not isinstance(inventories[name], list)
                   or len(inventories[name]) != count
                   for name, count in V8_CANDIDATE_RULE_COUNTS.items())):
        raise BroadQaExternalDataError("v8 candidate program 漂移")
    return inventories


def _build_index(
        inventories: dict[str, object],
        ) -> dict[str, dict[object, dict[str, object]]]:
    """从已核验 inventory 构造三个执行索引并拒绝 key 冲突。"""
    specs = {
        "SOURCE_CONDITIONED_LEXICAL_ATOM": (
            inventories["source_conditioned_rules"],
            lambda item: (item["official_source_text"], item["input_text"])),
        "ORTHOGRAPHIC_ATOM": (
            inventories["orthographic_rules"], lambda item: item["input_atom"]),
        "IDENTITY_VETO": (
            inventories["identity_veto_rules"], lambda item: item["input_text"]),
    }
    indexes = {}
    for kind, (records, key_builder) in specs.items():
        index = {}
        for record in records:
            key = key_builder(record)
            if key in index:
                raise BroadQaExternalDataError("v8 candidate runtime key 冲突")
            index[key] = record
        indexes[kind] = index
    return indexes


def build_normalization_recovery_v8_candidate_index(
        candidate: dict[str, object],
        ) -> dict[str, dict[object, dict[str, object]]]:
    """核验候选一次并构造三个执行索引。"""
    return _build_index(_validate_candidate(candidate))


def _query(value: dict[str, object]) -> dict[str, object]:
    """核验一条整串 query 与其结构 token 自洽。"""
    if not isinstance(value, dict):
        raise BroadQaExternalDataError("v8 candidate query 非对象")
    input_text = value.get("input_text")
    source = value.get("official_source_text")
    tokens = value.get("structure_tokens")
    if (not isinstance(input_text, str) or not input_text
            or not isinstance(source, str) or not source
            or not isinstance(tokens, list)
            or any(not isinstance(item, str) or not item for item in tokens)
            or tuple(tokens) != localization_structure_tokens(input_text)):
        raise BroadQaExternalDataError("v8 candidate query 结构漂移")
    return value


def _result(
        candidate: dict[str, object], query: dict[str, object], *,
        behavior: str, route_kind: str, output_text: str,
        matched_rule_ids: list[str], reason: str,
        ) -> dict[str, object]:
    """封装一个 EXACT 或无提交 UNKNOWN 结果。"""
    if behavior not in {"EXACT", "UNKNOWN"}:
        raise BroadQaExternalDataError("v8 candidate runtime behavior 非法")
    tokens = list(localization_structure_tokens(output_text)) if output_text else []
    payload = {
        "behavior": behavior,
        "candidate_program_sha256": candidate["candidate_program_sha256"],
        "exception_count": 0,
        "input_text": query["input_text"],
        "matched_rule_ids": matched_rule_ids,
        "official_source_text": query["official_source_text"],
        "output_structure_tokens": tokens,
        "output_text": output_text,
        "partial_commit_count": int(
            behavior == "UNKNOWN" and bool(output_text)),
        "production_enabled": 0,
        "reason": reason,
        "route_kind": route_kind,
        "structure_mismatch_count": int(
            bool(output_text) and tokens != query["structure_tokens"]),
    }
    return {**payload, "result_sha256": _sha256(payload)}


def _reference_match(
        records: list[dict[str, object]], predicate: object,
        ) -> dict[str, object] | None:
    """线性查找唯一匹配，作为 indexed runtime 的独立对照。"""
    matches = [item for item in records if predicate(item)]
    if len(matches) > 1:
        raise BroadQaExternalDataError("v8 candidate reference key 冲突")
    return matches[0] if matches else None


def _execute_one(
        candidate: dict[str, object], query_value: dict[str, object], *,
        inventories: dict[str, object],
        index: dict[str, dict[object, dict[str, object]]] | None,
        ) -> dict[str, object]:
    """按冻结优先级执行一条整串 query。"""
    query = _query(query_value)
    source = str(query["official_source_text"])
    input_text = str(query["input_text"])
    if index is None:
        lexical = _reference_match(
            inventories["source_conditioned_rules"],
            lambda item: item["official_source_text"] == source
            and item["input_text"] == input_text)
        atom_by_input = None
        identity = _reference_match(
            inventories["identity_veto_rules"],
            lambda item: item["input_text"] == input_text)
    else:
        lexical = index["SOURCE_CONDITIONED_LEXICAL_ATOM"].get(
            (source, input_text))
        atom_by_input = index["ORTHOGRAPHIC_ATOM"]
        identity = index["IDENTITY_VETO"].get(input_text)
    proposed = ""
    route = "UNKNOWN"
    rule_ids: list[str] = []
    if lexical is not None:
        proposed = str(lexical["output_text"])
        route = "SOURCE_CONDITIONED_LEXICAL_ATOM"
        rule_ids = [str(lexical["candidate_rule_id"])]
    else:
        output = []
        atom_rules = []
        for character in input_text:
            rule = (_reference_match(
                inventories["orthographic_rules"],
                lambda item: item["input_atom"] == character)
                if atom_by_input is None else atom_by_input.get(character))
            output.append(character if rule is None else str(rule["output_atom"]))
            if rule is not None:
                atom_rules.append(str(rule["candidate_rule_id"]))
        composed = "".join(output)
        if composed != input_text:
            proposed = composed
            route = "ORTHOGRAPHIC_ATOM"
            rule_ids = sorted(set(atom_rules))
        elif identity is not None:
            proposed = input_text
            route = "IDENTITY_VETO"
            rule_ids = [str(identity["candidate_rule_id"])]
    if not proposed:
        return _result(
            candidate, query, behavior="UNKNOWN", route_kind="UNKNOWN",
            output_text="", matched_rule_ids=[], reason="NO_COMMITTED_RULE")
    if list(localization_structure_tokens(proposed)) != query["structure_tokens"]:
        return _result(
            candidate, query, behavior="UNKNOWN", route_kind=route,
            output_text="", matched_rule_ids=rule_ids,
            reason="STRUCTURE_PRESERVATION_HARD_GATE_REJECTED")
    return _result(
        candidate, query, behavior="EXACT", route_kind=route,
        output_text=proposed, matched_rule_ids=rule_ids,
        reason="FROZEN_RULE_COMMITTED")


def execute_normalization_recovery_v8_candidate_batch(
        candidate: dict[str, object],
        queries: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, object], ...]:
    """使用字典索引执行一批整串 query。"""
    if not isinstance(queries, tuple) or not queries:
        raise BroadQaExternalDataError("v8 candidate batch 为空")
    inventories = _validate_candidate(candidate)
    index = _build_index(inventories)
    return tuple(_execute_one(
        candidate, item, inventories=inventories, index=index)
        for item in queries)


def reference_normalization_recovery_v8_candidate_batch(
        candidate: dict[str, object],
        queries: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, object], ...]:
    """使用独立线性扫描执行同一批整串 query。"""
    if not isinstance(queries, tuple) or not queries:
        raise BroadQaExternalDataError("v8 candidate reference batch 为空")
    inventories = _validate_candidate(candidate)
    return tuple(_execute_one(
        candidate, item, inventories=inventories, index=None)
        for item in queries)


def derive_normalization_recovery_v8_candidate_preflight(
        candidate: dict[str, object],
        ) -> dict[str, object]:
    """用候选自身规则构造标签盲 exact/unknown 与双解释器预检。"""
    inventories = _validate_candidate(candidate)
    cases = []
    for item in inventories["source_conditioned_rules"]:
        cases.append({
            "expected_output": item["output_text"],
            "input_text": item["input_text"],
            "official_source_text": item["official_source_text"],
            "route_kind": "SOURCE_CONDITIONED_LEXICAL_ATOM",
        })
    for item in inventories["orthographic_rules"]:
        cases.append({
            "expected_output": item["output_atom"],
            "input_text": item["input_atom"],
            "official_source_text": "SYNTHETIC_ORTHOGRAPHIC_PREFLIGHT",
            "route_kind": "ORTHOGRAPHIC_ATOM",
        })
    for item in inventories["identity_veto_rules"]:
        cases.append({
            "expected_output": item["input_text"],
            "input_text": item["input_text"],
            "official_source_text": "SYNTHETIC_IDENTITY_PREFLIGHT",
            "route_kind": "IDENTITY_VETO",
        })
    cases.append({
        "expected_output": "",
        "input_text": "未授权候选预检文本",
        "official_source_text": "SYNTHETIC_UNKNOWN_PREFLIGHT",
        "route_kind": "UNKNOWN",
    })
    queries = tuple({
        "input_text": str(item["input_text"]),
        "official_source_text": str(item["official_source_text"]),
        "structure_tokens": list(localization_structure_tokens(
            str(item["input_text"]))),
    } for item in cases)
    indexed = execute_normalization_recovery_v8_candidate_batch(
        candidate, queries)
    reference = reference_normalization_recovery_v8_candidate_batch(
        candidate, queries)
    failures = sum(
        left != right or left["output_text"] != case["expected_output"]
        or left["route_kind"] != case["route_kind"]
        for case, left, right in zip(cases, indexed, reference))
    result = {
        "case_count": len(cases),
        "exception_count": 0,
        "failure_count": failures,
        "format_version": 1,
        "indexed_reference_mismatch_count": sum(
            left != right for left, right in zip(indexed, reference)),
        "production_enabled": 0,
        "rule_counts": V8_CANDIDATE_RULE_COUNTS,
        "structure_obligation_count": len(
            inventories["structure_obligations"]),
        "unknown_case_count": sum(
            item["behavior"] == "UNKNOWN" for item in indexed),
    }
    return {**result, "preflight_sha256": _sha256(result)}


__all__ = [
    "NORMALIZATION_RECOVERY_V8_CANDIDATE_KIND",
    "NORMALIZATION_RECOVERY_V8_CANDIDATE_STATUS",
    "NORMALIZATION_RECOVERY_V8_TARGET_POLICY_SCOPE",
    "V8_CANDIDATE_RULE_COUNTS",
    "build_normalization_recovery_v8_candidate_index",
    "compile_normalization_recovery_v8_candidate",
    "derive_normalization_recovery_v8_candidate_preflight",
    "execute_normalization_recovery_v8_candidate_batch",
    "reference_normalization_recovery_v8_candidate_batch",
]
