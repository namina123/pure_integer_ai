"""编译并执行 recovery-v5 target/source-scoped disabled phrase program。"""
from __future__ import annotations

from collections import defaultdict
import hashlib

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_learning_contract import (
    NORMALIZATION_RECOVERY_V5_DEFEATER_KIND,
    NORMALIZATION_RECOVERY_V5_RULE_CLASSES,
    NORMALIZATION_RECOVERY_V5_SOURCE_OVERLAP_KIND,
    NORMALIZATION_RECOVERY_V5_SOURCE_RULE_KIND,
    NORMALIZATION_RECOVERY_V5_TARGET_OVERLAP_KIND,
    NORMALIZATION_RECOVERY_V5_TARGET_RULE_KIND,
    normalization_recovery_v5_rule_class,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_training_records import (
    RECOVERY_V5_TARGET_POLICY_SCOPE,
    V5_SOURCE_FAMILIES,
    V5_SOURCE_POLICY_BY_FAMILY,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes


NORMALIZATION_RECOVERY_V5_PHRASE_PROGRAM_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V5_PHRASE_PROGRAM_V1")
NORMALIZATION_RECOVERY_V5_PHRASE_PROGRAM_STATUS = (
    "CANDIDATE_ONLY_PUBLIC_PRODUCTION_DISABLED")

_RULE_CLASS_ORDER = {
    "WHOLE_INPUT_EQUAL_LENGTH": 0,
    "WHOLE_INPUT_VARIABLE_LENGTH": 1,
    "CONTEXT_HUNK": 2,
    "EDIT_CORE": 3,
}

_PROGRAM_KEYS = {
    "artifact_kind",
    "defeaters",
    "format_version",
    "production_enabled",
    "program_sha256",
    "rule_pack_manifest_sha256",
    "source_programs",
    "status",
    "structured_local_execution_allowed",
    "target_buckets",
    "target_policy_scope",
    "unscoped_source_rule_execution_allowed",
    "whole_input_exact_precedes_local",
}


def _sha_value(value: object, *, label: str) -> str:
    """核验并返回小写 SHA-256。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(item not in "0123456789abcdef" for item in value)):
        raise BroadQaExternalDataError(f"{label} 非法")
    return value


def _text(value: object, *, label: str, allow_empty: bool = False) -> str:
    """核验不含 surrogate 的运行文本。"""
    if (not isinstance(value, str) or (not allow_empty and not value)
            or any(0xD800 <= ord(item) <= 0xDFFF for item in value)):
        raise BroadQaExternalDataError(f"{label} 非法")
    return value


def _structure_tokens(value: object, *, label: str) -> tuple[str, ...]:
    """核验并冻结调用方提供的结构 token 序。"""
    if (not isinstance(value, (list, tuple))
            or any(not isinstance(item, str) for item in value)):
        raise BroadQaExternalDataError(f"{label} 非法")
    return tuple(value)


def _structure_variants(value: object) -> list[list[str]]:
    """核验 rule 内规范排序且互异的结构 token variants。"""
    if (not isinstance(value, list)
            or any(not isinstance(item, list)
                   or any(not isinstance(token, str) for token in item)
                   for item in value)):
        raise BroadQaExternalDataError(
            "v5 phrase rule structure variants 非法")
    frozen = [tuple(item) for item in value]
    if frozen != sorted(set(frozen)):
        raise BroadQaExternalDataError(
            "v5 phrase rule structure variants 顺序/identity 漂移")
    return value


def _program_payload(program: dict[str, object]) -> dict[str, object]:
    """移除自摘要字段后返回 program identity payload。"""
    return {key: value for key, value in program.items()
            if key != "program_sha256"}


def _scope_identity(rule: dict[str, object]) -> dict[str, object]:
    """返回 rule/defeater 必须逐字段相等的 scope identity。"""
    return {
        "candidate_scope_kind": rule["candidate_scope_kind"],
        "source_execution_family": rule["source_execution_family"],
        "source_execution_policy_scope": rule[
            "source_execution_policy_scope"],
        "target_policy_scope": rule["target_policy_scope"],
    }


def _rule_semantic_key(rule: dict[str, object]) -> tuple[str, ...]:
    """返回跨 target/source program 唯一的 rule 语义键。"""
    return (
        str(rule["candidate_scope_kind"]),
        str(rule["source_execution_family"]),
        str(rule["rule_class"]),
        str(rule["input_text"]),
        str(rule["output_text"]),
    )


def _rule_order(rule: dict[str, object]):
    """固定 whole/local、最长匹配、rule class 与 scope 的完全序。"""
    try:
        return (
            0 if rule["fragment_kind"] == "WHOLE_INPUT" else 1,
            -len(str(rule["input_text"])),
            _RULE_CLASS_ORDER[str(rule["rule_class"])],
            str(rule["input_text"]),
            0 if rule["candidate_scope_kind"] == "TARGET_CROSS_FAMILY" else 1,
            str(rule["rule_id"]),
        )
    except (KeyError, TypeError) as error:
        raise BroadQaExternalDataError(
            "v5 phrase rule priority schema 漂移") from error


def _authority_valid(rule: dict[str, object]) -> bool:
    """复核 rule class、authority basis 与来源支持门。"""
    families = rule.get("source_families")
    required = rule.get("required_distinct_source_family_count")
    observed = rule.get("observed_distinct_source_family_count")
    if (not isinstance(families, list) or not families
            or any(item not in V5_SOURCE_FAMILIES for item in families)
            or families != sorted(set(families))
            or type(required) is not int or required <= 0
            or type(observed) is not int or observed != len(families)):
        return False
    if rule["candidate_scope_kind"] == "SOURCE_ONLY":
        return (families == [rule["source_execution_family"]]
                and required == 1
                and rule["authority_basis"] in {
                    "REPEATED_SOURCE_CONTEXT",
                    "REPEATED_SOURCE_WHOLE_INPUT",
                })
    if len(families) < required:
        return False
    expected_basis = {
        "CONTEXT_HUNK": "LOCAL_OR_CONTEXT_TWO_FAMILY_CONSENSUS",
        "EDIT_CORE": "LOCAL_OR_CONTEXT_TWO_FAMILY_CONSENSUS",
        "WHOLE_INPUT_EQUAL_LENGTH": (
            "EQUAL_LENGTH_WHOLE_INPUT_TWO_FAMILY_CONSENSUS"),
        "WHOLE_INPUT_VARIABLE_LENGTH": (
            "VARIABLE_LENGTH_WHOLE_INPUT_STRONG_CONSENSUS"),
    }
    return rule["authority_basis"] == expected_basis[rule["rule_class"]]


def _validate_rule(
        rule: dict[str, object],
        *,
        expected_scope_kind: str,
        expected_record_kind: str,
        ) -> tuple[str, str, str]:
    """核验一条禁用态 v5 rule 并返回 id/input/source family。"""
    rule_id = _sha_value(rule.get("rule_id"), label="v5 phrase rule id")
    input_text = _text(rule.get("input_text"), label="v5 phrase input")
    output_text = _text(
        rule.get("output_text"), label="v5 phrase output", allow_empty=True)
    source_family = rule.get("source_execution_family")
    source_policy = rule.get("source_execution_policy_scope")
    target_scope = rule.get("target_policy_scope")
    fragment_kind = rule.get("fragment_kind")
    rule_class = rule.get("rule_class")
    equal_length = int(len(input_text) == len(output_text))
    try:
        expected_class = normalization_recovery_v5_rule_class(
            str(fragment_kind), equal_length=equal_length)
    except ValueError as error:
        raise BroadQaExternalDataError(
            "v5 phrase rule fragment class 漂移") from error
    variants = _structure_variants(rule.get("structure_token_variants"))
    structure_required = int(any(variants))
    if expected_scope_kind == "TARGET_CROSS_FAMILY":
        scope_valid = (
            source_family == "" and source_policy == ""
            and target_scope == RECOVERY_V5_TARGET_POLICY_SCOPE)
    else:
        scope_valid = (
            source_family in V5_SOURCE_POLICY_BY_FAMILY
            and source_policy == V5_SOURCE_POLICY_BY_FAMILY[source_family]
            and target_scope == "")
    input_match = (
        "WHOLE_INPUT_EXACT"
        if fragment_kind == "WHOLE_INPUT" else "EXACT_SCALAR_SUBSEQUENCE")
    expected_application = {
        "candidate_scope_kind": expected_scope_kind,
        "defeater_required": 1,
        "input_match": input_match,
        "source_execution_family": source_family,
        "source_execution_policy_scope": source_policy,
        "structure_match_required": structure_required,
        "structure_token_variants": variants,
        "unscoped_execution_allowed": 0,
    }
    if (rule.get("record_kind") != expected_record_kind
            or rule.get("candidate_scope_kind") != expected_scope_kind
            or not scope_valid
            or rule_class not in NORMALIZATION_RECOVERY_V5_RULE_CLASSES
            or rule_class != expected_class
            or rule.get("application_scope") != expected_application
            or not _authority_valid(rule)
            or rule.get("runtime_state") != "LEARNED_PACK_DISABLED"
            or rule.get("production_enabled") != 0
            or not isinstance(rule.get("positive_evidence_ids"), list)
            or not rule["positive_evidence_ids"]
            or not isinstance(rule.get("negative_evidence_ids"), list)
            or not rule["negative_evidence_ids"]
            or not isinstance(rule.get("defeater_ids"), list)
            or not rule["defeater_ids"]
            or rule.get("equal_length") != equal_length
            or rule.get("variable_length") != 1 - equal_length):
        raise BroadQaExternalDataError(
            "v5 phrase rule scope/authority/Evidence/schema 漂移")
    return rule_id, input_text, str(source_family)


def _validate_defeater(
        record: dict[str, object],
        rule_by_id: dict[str, dict[str, object]],
        ) -> tuple[str, str]:
    """核验一个 defeater 与被阻断 rule 的逐字段闭环。"""
    if not isinstance(record, dict):
        raise BroadQaExternalDataError("v5 phrase defeater 非字典")
    defeater_id = _sha_value(
        record.get("defeater_id"), label="v5 phrase defeater id")
    rule_id = _sha_value(record.get("rule_id"), label="v5 defeater rule")
    rule = rule_by_id.get(rule_id)
    left = record.get("left_context")
    right = record.get("right_context")
    if (rule is None
            or record.get("record_kind")
            != NORMALIZATION_RECOVERY_V5_DEFEATER_KIND
            or record.get("action") != "BLOCK_PHRASE_RULE_USE_BACKOFF"
            or record.get("production_enabled") != 0
            or record.get("rule_class") != rule["rule_class"]
            or any(record.get(key) != value
                   for key, value in _scope_identity(rule).items())
            or not isinstance(left, str) or len(left) > 4
            or not isinstance(right, str) or len(right) > 4
            or record.get("left_boundary") not in {0, 1}
            or record.get("right_boundary") not in {0, 1}
            or not isinstance(record.get("refute_evidence_ids"), list)
            or not record["refute_evidence_ids"]):
        raise BroadQaExternalDataError("v5 phrase defeater schema 漂移")
    return defeater_id, rule_id


def _ordered_buckets(
        rules: list[dict[str, object]],
        ) -> list[dict[str, object]]:
    """把 rule 按首 scalar 分桶并冻结完全序。"""
    buckets: dict[int, list[dict[str, object]]] = defaultdict(list)
    for rule in rules:
        buckets[ord(str(rule["input_text"])[0])].append(rule)
    return [{
        "first_scalar": first_scalar,
        "rules": sorted(values, key=_rule_order),
    } for first_scalar, values in sorted(buckets.items())]


def compile_normalization_recovery_v5_phrase_program(
        *,
        rule_pack_manifest_sha256: str,
        target_phrase_rules: tuple[dict[str, object], ...],
        source_phrase_rules: tuple[dict[str, object], ...],
        defeaters: tuple[dict[str, object], ...],
        target_overlap_index: tuple[dict[str, object], ...],
        source_overlap_index: tuple[dict[str, object], ...],
        ) -> dict[str, object]:
    """把严格回读的 v5 pack records 编译为禁用态 scoped program。"""
    pack_sha = _sha_value(
        rule_pack_manifest_sha256, label="v5 phrase pack manifest")
    inputs = (target_phrase_rules, source_phrase_rules, defeaters,
              target_overlap_index, source_overlap_index)
    if any(not isinstance(value, tuple) for value in inputs):
        raise BroadQaExternalDataError("v5 phrase program 输入类型漂移")
    rule_by_id = {}
    semantic_keys = set()
    source_families = set()
    specifications = (
        (target_phrase_rules, "TARGET_CROSS_FAMILY",
         NORMALIZATION_RECOVERY_V5_TARGET_RULE_KIND),
        (source_phrase_rules, "SOURCE_ONLY",
         NORMALIZATION_RECOVERY_V5_SOURCE_RULE_KIND),
    )
    for rules, scope_kind, record_kind in specifications:
        for rule in rules:
            if not isinstance(rule, dict):
                raise BroadQaExternalDataError("v5 phrase rule 非字典")
            rule_id, input_text, source_family = _validate_rule(
                rule,
                expected_scope_kind=scope_kind,
                expected_record_kind=record_kind,
            )
            semantic_key = _rule_semantic_key(rule)
            if rule_id in rule_by_id or semantic_key in semantic_keys:
                raise BroadQaExternalDataError(
                    "v5 phrase rule identity/semantic key 重复")
            rule_by_id[rule_id] = rule
            semantic_keys.add(semantic_key)
            if source_family:
                source_families.add(source_family)

    defeater_by_id = {}
    defeaters_by_rule: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in defeaters:
        defeater_id, rule_id = _validate_defeater(record, rule_by_id)
        if defeater_id in defeater_by_id:
            raise BroadQaExternalDataError("v5 phrase defeater identity 重复")
        defeater_by_id[defeater_id] = record
        defeaters_by_rule[rule_id].append(record)
    for rule_id, rule in rule_by_id.items():
        declared = sorted(str(value) for value in rule["defeater_ids"])
        actual = sorted(str(item["defeater_id"])
                        for item in defeaters_by_rule.get(rule_id, ()))
        if declared != actual:
            raise BroadQaExternalDataError(
                "v5 phrase rule/defeater 未闭合")

    index_by_rule = {}
    index_specs = (
        (target_overlap_index, NORMALIZATION_RECOVERY_V5_TARGET_OVERLAP_KIND,
         "TARGET_CROSS_FAMILY"),
        (source_overlap_index, NORMALIZATION_RECOVERY_V5_SOURCE_OVERLAP_KIND,
         "SOURCE_ONLY"),
    )
    for records, expected_kind, expected_scope in index_specs:
        for record in records:
            if not isinstance(record, dict):
                raise BroadQaExternalDataError("v5 overlap index 非字典")
            index_id = _sha_value(
                record.get("index_id"), label="v5 overlap index id")
            rule_id = _sha_value(
                record.get("rule_id"), label="v5 overlap rule")
            rule = rule_by_id.get(rule_id)
            if rule is None:
                raise BroadQaExternalDataError("v5 overlap index rule 缺失")
            input_text = str(rule["input_text"])
            whole = rule["fragment_kind"] == "WHOLE_INPUT"
            expected_priority = (
                "WHOLE_INPUT_EXACT_PRECEDES_LONGEST_LOCAL_MATCH"
                if whole else "LONGEST_MATCH_THEN_RULE_CLASS_THEN_RULE_ID")
            expected_match = (
                "WHOLE_INPUT_EXACT" if whole else "EXACT_SCALAR_SUBSEQUENCE")
            if (record.get("record_kind") != expected_kind
                    or record.get("candidate_scope_kind") != expected_scope
                    or rule["candidate_scope_kind"] != expected_scope
                    or record.get("source_execution_family")
                    != rule["source_execution_family"]
                    or record.get("target_policy_scope")
                    != rule["target_policy_scope"]
                    or record.get("candidate_input") != input_text
                    or record.get("first_scalar") != ord(input_text[0])
                    or record.get("input_scalar_length") != len(input_text)
                    or record.get("rule_class") != rule["rule_class"]
                    or record.get("input_match") != expected_match
                    or record.get("priority_contract") != expected_priority
                    or rule_id in index_by_rule):
                raise BroadQaExternalDataError("v5 overlap index 漂移")
            index_by_rule[rule_id] = index_id
    if set(index_by_rule) != set(rule_by_id):
        raise BroadQaExternalDataError("v5 overlap index 未覆盖全部规则")

    source_programs = []
    for source_family in sorted(source_families):
        rules = [rule for rule in source_phrase_rules
                 if rule["source_execution_family"] == source_family]
        source_programs.append({
            "buckets": _ordered_buckets(rules),
            "source_family": source_family,
        })
    program = {
        "artifact_kind": NORMALIZATION_RECOVERY_V5_PHRASE_PROGRAM_KIND,
        "defeaters": sorted(defeaters,
                            key=lambda item: str(item["defeater_id"])),
        "format_version": 1,
        "production_enabled": 0,
        "rule_pack_manifest_sha256": pack_sha,
        "source_programs": source_programs,
        "status": NORMALIZATION_RECOVERY_V5_PHRASE_PROGRAM_STATUS,
        "structured_local_execution_allowed": 0,
        "target_buckets": _ordered_buckets(list(target_phrase_rules)),
        "target_policy_scope": RECOVERY_V5_TARGET_POLICY_SCOPE,
        "unscoped_source_rule_execution_allowed": 0,
        "whole_input_exact_precedes_local": 1,
    }
    return {
        **program,
        "program_sha256": hashlib.sha256(
            canonical_json_bytes(program)).hexdigest(),
    }


def _defeater_matches(
        record: dict[str, object],
        text: str,
        start: int,
        end: int,
        ) -> bool:
    """执行 exact-adjacent context predicate。"""
    left = str(record["left_context"])
    right = str(record["right_context"])
    left_match = (
        start == 0 if record["left_boundary"] == 1
        else start >= len(left) and text[start - len(left):start] == left)
    right_match = (
        end == len(text) if record["right_boundary"] == 1
        else text[end:end + len(right)] == right)
    return left_match and right_match


def normalization_recovery_v5_defeater_matches(
        record: dict[str, object],
        text: str,
        start: int,
        end: int,
        ) -> bool:
    """严格核验并执行一个 v5 pack-derived context defeater。"""
    source = _text(text, label="v5 defeater runtime input")
    if (not isinstance(record, dict)
            or record.get("record_kind")
            != NORMALIZATION_RECOVERY_V5_DEFEATER_KIND
            or record.get("action") != "BLOCK_PHRASE_RULE_USE_BACKOFF"
            or record.get("production_enabled") != 0
            or record.get("rule_class") not in NORMALIZATION_RECOVERY_V5_RULE_CLASSES
            or not isinstance(record.get("left_context"), str)
            or len(record["left_context"]) > 4
            or not isinstance(record.get("right_context"), str)
            or len(record["right_context"]) > 4
            or record.get("left_boundary") not in {0, 1}
            or record.get("right_boundary") not in {0, 1}
            or type(start) is not int or type(end) is not int
            or not 0 <= start < end <= len(source)):
        raise BroadQaExternalDataError("v5 defeater predicate 输入漂移")
    return _defeater_matches(record, source, start, end)


def _validated_character_rules(value: dict[str, str] | None) -> dict[str, str]:
    """核验 character backoff 只含一 scalar 到一 scalar 映射。"""
    if value is None:
        return {}
    if (not isinstance(value, dict)
            or any(not isinstance(key, str) or len(key) != 1
                   or not isinstance(output, str) or len(output) != 1
                   for key, output in value.items())):
        raise BroadQaExternalDataError("v5 character backoff 非法")
    return value


def _validated_buckets(
        values: object,
        *,
        label: str,
        ) -> dict[int, tuple[dict[str, object], ...]]:
    """核验 program bucket 顺序与首 scalar。"""
    if not isinstance(values, list):
        raise BroadQaExternalDataError(f"{label} 非法")
    buckets = {}
    rule_ids = set()
    for bucket in values:
        if (not isinstance(bucket, dict)
                or set(bucket) != {"first_scalar", "rules"}
                or type(bucket["first_scalar"]) is not int
                or not isinstance(bucket["rules"], list)
                or bucket["first_scalar"] in buckets):
            raise BroadQaExternalDataError(f"{label} 漂移")
        rules = tuple(bucket["rules"])
        if any(not isinstance(item, dict)
               or not isinstance(item.get("input_text"), str)
               or not item["input_text"] for item in rules):
            raise BroadQaExternalDataError(f"{label} rule 非法")
        if (rules != tuple(sorted(rules, key=_rule_order))
                or any(ord(item["input_text"][0]) != bucket["first_scalar"]
                       or str(item.get("rule_id")) in rule_ids
                       for item in rules)):
            raise BroadQaExternalDataError(f"{label} 顺序漂移")
        rule_ids.update(str(item.get("rule_id")) for item in rules)
        buckets[bucket["first_scalar"]] = rules
    return buckets


def _validated_program(
        program: dict[str, object],
        source_family: str,
        ) -> tuple[dict[int, tuple[dict[str, object], ...]],
                   dict[int, tuple[dict[str, object], ...]],
                   dict[str, tuple[dict[str, object], ...]]]:
    """核验 program 自摘要、source 路由和 defeater 引用。"""
    if source_family and source_family not in V5_SOURCE_FAMILIES:
        raise BroadQaExternalDataError("v5 runtime source family 非法")
    if (not isinstance(program, dict)
            or set(program) != _PROGRAM_KEYS
            or program.get("artifact_kind")
            != NORMALIZATION_RECOVERY_V5_PHRASE_PROGRAM_KIND
            or program.get("format_version") != 1
            or program.get("status")
            != NORMALIZATION_RECOVERY_V5_PHRASE_PROGRAM_STATUS
            or program.get("production_enabled") != 0
            or program.get("target_policy_scope")
            != RECOVERY_V5_TARGET_POLICY_SCOPE
            or program.get("whole_input_exact_precedes_local") != 1
            or program.get("structured_local_execution_allowed") != 0
            or program.get("unscoped_source_rule_execution_allowed") != 0
            or program.get("program_sha256") != hashlib.sha256(
                canonical_json_bytes(_program_payload(program))).hexdigest()
            or not isinstance(program.get("source_programs"), list)
            or not isinstance(program.get("defeaters"), list)):
        raise BroadQaExternalDataError("v5 phrase program identity 漂移")
    _sha_value(
        program.get("rule_pack_manifest_sha256"),
        label="v5 phrase program pack manifest")
    target_buckets = _validated_buckets(
        program.get("target_buckets"), label="v5 target bucket")
    selected_source_buckets = {}
    seen_sources = set()
    all_rule_ids = set()
    semantic_keys = set()
    rule_by_id = {}
    for rules in target_buckets.values():
        for rule in rules:
            rule_id, _input_text, source_rule_family = _validate_rule(
                rule,
                expected_scope_kind="TARGET_CROSS_FAMILY",
                expected_record_kind=NORMALIZATION_RECOVERY_V5_TARGET_RULE_KIND,
            )
            semantic_key = _rule_semantic_key(rule)
            if (source_rule_family or rule_id in all_rule_ids
                    or semantic_key in semantic_keys):
                raise BroadQaExternalDataError(
                    "v5 target program rule identity 漂移")
            all_rule_ids.add(rule_id)
            semantic_keys.add(semantic_key)
            rule_by_id[rule_id] = rule
    if any(not isinstance(item, dict)
           or set(item) != {"buckets", "source_family"}
           or item["source_family"] not in V5_SOURCE_FAMILIES
           for item in program["source_programs"]):
        raise BroadQaExternalDataError("v5 source program roster 漂移")
    source_program_families = [
        item["source_family"] for item in program["source_programs"]]
    if source_program_families != sorted(source_program_families):
        raise BroadQaExternalDataError("v5 source program roster 顺序漂移")
    for source_program in program["source_programs"]:
        if (not isinstance(source_program, dict)
                or set(source_program) != {"buckets", "source_family"}
                or source_program["source_family"] not in V5_SOURCE_FAMILIES
                or source_program["source_family"] in seen_sources):
            raise BroadQaExternalDataError("v5 source program roster 漂移")
        family = str(source_program["source_family"])
        seen_sources.add(family)
        buckets = _validated_buckets(
            source_program["buckets"], label="v5 source bucket")
        for rules in buckets.values():
            for rule in rules:
                rule_id, _input_text, source_rule_family = _validate_rule(
                    rule,
                    expected_scope_kind="SOURCE_ONLY",
                    expected_record_kind=(
                        NORMALIZATION_RECOVERY_V5_SOURCE_RULE_KIND),
                )
                semantic_key = _rule_semantic_key(rule)
                if (source_rule_family != family
                        or rule_id in all_rule_ids
                        or semantic_key in semantic_keys):
                    raise BroadQaExternalDataError(
                        "v5 source program rule identity 漂移")
                all_rule_ids.add(rule_id)
                semantic_keys.add(semantic_key)
                rule_by_id[rule_id] = rule
        if family == source_family:
            selected_source_buckets = buckets
    defeaters_by_rule: dict[str, list[dict[str, object]]] = defaultdict(list)
    defeater_ids = set()
    ordered_defeater_ids = []
    for record in program["defeaters"]:
        defeater_id, rule_id = _validate_defeater(record, rule_by_id)
        if defeater_id in defeater_ids:
            raise BroadQaExternalDataError("v5 program defeater identity 重复")
        defeater_ids.add(defeater_id)
        ordered_defeater_ids.append(defeater_id)
        defeaters_by_rule[rule_id].append(record)
    if ordered_defeater_ids != sorted(ordered_defeater_ids):
        raise BroadQaExternalDataError("v5 program defeater 顺序漂移")
    for rule_id, rule in rule_by_id.items():
        declared = sorted(str(value) for value in rule["defeater_ids"])
        actual = sorted(str(item["defeater_id"])
                        for item in defeaters_by_rule.get(rule_id, ()))
        if declared != actual:
            raise BroadQaExternalDataError(
                "v5 program rule/defeater 未闭合")
    return (
        target_buckets,
        selected_source_buckets,
        {key: tuple(sorted(values,
                           key=lambda item: str(item["defeater_id"])))
         for key, values in defeaters_by_rule.items()},
    )


def _rule_structure_matches(
        rule: dict[str, object],
        structure_tokens: tuple[str, ...],
        ) -> bool:
    """只允许 whole rule 在结构 token 精确匹配时处理结构化输入。"""
    variants = tuple(tuple(item) for item in rule["structure_token_variants"])
    if structure_tokens:
        return (rule["fragment_kind"] == "WHOLE_INPUT"
                and structure_tokens in variants)
    return not rule["application_scope"]["structure_match_required"]


def _execute_validated(
        *,
        source: str,
        source_family: str,
        structure_tokens: tuple[str, ...],
        program_sha256: str,
        target_buckets: dict[int, tuple[dict[str, object], ...]],
        source_buckets: dict[int, tuple[dict[str, object], ...]],
        defeaters_by_rule: dict[str, tuple[dict[str, object], ...]],
        backoff: dict[str, str],
    indexed: bool,
        ) -> dict[str, object]:
    """在已核验 target/source index 上执行一个输入。"""
    def bucket_candidates(position: int):
        """返回当前位置首 scalar 对应的规范候选序。"""
        scalar = ord(source[position])
        values = target_buckets.get(scalar, ()) + source_buckets.get(scalar, ())
        return tuple(sorted(values, key=_rule_order))

    all_rules = tuple(
        rule for buckets in (target_buckets, source_buckets)
        for values in buckets.values() for rule in values)

    def candidates_at(position: int):
        """按解释器模式返回当前位置的 indexed 或线性候选序。"""
        if indexed:
            return bucket_candidates(position)
        return tuple(sorted((
            item for item in all_rules
            if str(item["input_text"])[0] == source[position]),
            key=_rule_order,
        ))

    def choose(position: int, *, whole_only: bool):
        """按完全序选择首条未被结构门或 defeater 阻断的规则。"""
        blocked = []
        for rule in candidates_at(position):
            phrase = str(rule["input_text"])
            end = position + len(phrase)
            is_whole_rule = rule["fragment_kind"] == "WHOLE_INPUT"
            if (whole_only != is_whole_rule
                    or (whole_only and (position != 0 or end != len(source)))
                    or (not whole_only and structure_tokens)
                    or end > len(source) or source[position:end] != phrase
                    or not _rule_structure_matches(rule, structure_tokens)):
                continue
            hits = tuple(
                item for item in defeaters_by_rule.get(str(rule["rule_id"]), ())
                if _defeater_matches(item, source, position, end))
            if hits:
                blocked.extend(str(item["defeater_id"]) for item in hits)
                continue
            return rule, tuple(sorted(set(blocked)))
        return None, tuple(sorted(set(blocked)))

    output = []
    steps = []
    exact, exact_blocked = choose(0, whole_only=True)
    if exact is not None:
        output_text = str(exact["output_text"])
        steps.append({
            "blocked_defeater_ids": list(exact_blocked),
            "candidate_scope_kind": exact["candidate_scope_kind"],
            "fragment_kind": exact["fragment_kind"],
            "input_end": len(source),
            "input_start": 0,
            "mode": "WHOLE_INPUT_EXACT",
            "output_text": output_text,
            "rule_class": exact["rule_class"],
            "rule_id": exact["rule_id"],
            "source_execution_family": exact["source_execution_family"],
        })
        output.append(output_text)
    else:
        position = 0
        pending_blocked = list(exact_blocked)
        while position < len(source):
            rule, blocked = choose(position, whole_only=False)
            blocked_ids = sorted(set(pending_blocked + list(blocked)))
            pending_blocked = []
            if rule is not None:
                phrase = str(rule["input_text"])
                end = position + len(phrase)
                output_text = str(rule["output_text"])
                step_scope = str(rule["candidate_scope_kind"])
                rule_id = str(rule["rule_id"])
                rule_source = str(rule["source_execution_family"])
                fragment_kind = str(rule["fragment_kind"])
                rule_class = str(rule["rule_class"])
                mode = "LONGEST_LOCAL_MATCH"
            else:
                phrase = source[position]
                end = position + 1
                use_backoff = not structure_tokens and phrase in backoff
                output_text = backoff.get(phrase, phrase) if use_backoff else phrase
                step_scope = "NONE"
                rule_id = ""
                rule_source = ""
                fragment_kind = ""
                rule_class = ""
                mode = "CHARACTER_BACKOFF" if use_backoff else "IDENTITY"
            steps.append({
                "blocked_defeater_ids": blocked_ids,
                "candidate_scope_kind": step_scope,
                "fragment_kind": fragment_kind,
                "input_end": end,
                "input_start": position,
                "mode": mode,
                "output_text": output_text,
                "rule_class": rule_class,
                "rule_id": rule_id,
                "source_execution_family": rule_source,
            })
            output.append(output_text)
            position = end
    result = {
        "input_text": source,
        "output_text": "".join(output),
        "program_sha256": program_sha256,
        "source_family": source_family,
        "steps": steps,
        "structure_tokens": list(structure_tokens),
        "target_policy_scope": RECOVERY_V5_TARGET_POLICY_SCOPE,
    }
    return {
        **result,
        "result_sha256": hashlib.sha256(
            canonical_json_bytes(result)).hexdigest(),
    }


def _execute(
        program: dict[str, object],
        text: str,
        *,
        source_family: str,
        structure_tokens: tuple[str, ...],
        character_rules: dict[str, str] | None,
        indexed: bool,
        ) -> dict[str, object]:
    """核验 program/input 后执行 indexed 或 reference 路径。"""
    source = _text(text, label="v5 phrase runtime input")
    tokens = _structure_tokens(
        structure_tokens, label="v5 phrase runtime structure tokens")
    target_buckets, source_buckets, defeaters = _validated_program(
        program, source_family)
    return _execute_validated(
        source=source,
        source_family=source_family,
        structure_tokens=tokens,
        program_sha256=str(program["program_sha256"]),
        target_buckets=target_buckets,
        source_buckets=source_buckets,
        defeaters_by_rule=defeaters,
        backoff=_validated_character_rules(character_rules),
        indexed=indexed,
    )


def _execute_batch(
        program: dict[str, object],
        texts: tuple[str, ...],
        *,
        source_family: str,
        structure_tokens: tuple[tuple[str, ...], ...] | None,
        character_rules: dict[str, str] | None,
        indexed: bool,
        ) -> tuple[dict[str, object], ...]:
    """一次核验 program 后批量执行同一 source family 的输入。"""
    if not isinstance(texts, tuple):
        raise BroadQaExternalDataError("v5 phrase batch 输入类型漂移")
    token_rows = (() for _item in texts) if structure_tokens is None else (
        iter(structure_tokens))
    token_values = tuple(token_rows)
    if len(token_values) != len(texts):
        raise BroadQaExternalDataError("v5 phrase batch structure 数量漂移")
    frozen_tokens = tuple(
        _structure_tokens(item, label="v5 phrase batch structure tokens")
        for item in token_values)
    target_buckets, source_buckets, defeaters = _validated_program(
        program, source_family)
    backoff = _validated_character_rules(character_rules)
    return tuple(_execute_validated(
        source=_text(item, label="v5 phrase runtime input"),
        source_family=source_family,
        structure_tokens=tokens,
        program_sha256=str(program["program_sha256"]),
        target_buckets=target_buckets,
        source_buckets=source_buckets,
        defeaters_by_rule=defeaters,
        backoff=backoff,
        indexed=indexed,
    ) for item, tokens in zip(texts, frozen_tokens))


def execute_normalization_recovery_v5_phrase_program(
        program: dict[str, object],
        text: str,
        *,
        source_family: str = "",
        structure_tokens: tuple[str, ...] = (),
        character_rules: dict[str, str] | None = None,
        ) -> dict[str, object]:
    """执行按首 scalar 分桶的 v5 indexed program。"""
    return _execute(
        program, text, source_family=source_family,
        structure_tokens=structure_tokens,
        character_rules=character_rules, indexed=True)


def reference_normalization_recovery_v5_phrase_program(
        program: dict[str, object],
        text: str,
        *,
        source_family: str = "",
        structure_tokens: tuple[str, ...] = (),
        character_rules: dict[str, str] | None = None,
        ) -> dict[str, object]:
    """执行不使用首 scalar index 的 v5 reference program。"""
    return _execute(
        program, text, source_family=source_family,
        structure_tokens=structure_tokens,
        character_rules=character_rules, indexed=False)


def execute_normalization_recovery_v5_phrase_batch(
        program: dict[str, object],
        texts: tuple[str, ...],
        *,
        source_family: str = "",
        structure_tokens: tuple[tuple[str, ...], ...] | None = None,
        character_rules: dict[str, str] | None = None,
        ) -> tuple[dict[str, object], ...]:
    """批量执行 v5 indexed program。"""
    return _execute_batch(
        program, texts, source_family=source_family,
        structure_tokens=structure_tokens,
        character_rules=character_rules, indexed=True)


def reference_normalization_recovery_v5_phrase_batch(
        program: dict[str, object],
        texts: tuple[str, ...],
        *,
        source_family: str = "",
        structure_tokens: tuple[tuple[str, ...], ...] | None = None,
        character_rules: dict[str, str] | None = None,
        ) -> tuple[dict[str, object], ...]:
    """批量执行 v5 reference program。"""
    return _execute_batch(
        program, texts, source_family=source_family,
        structure_tokens=structure_tokens,
        character_rules=character_rules, indexed=False)


__all__ = [
    "NORMALIZATION_RECOVERY_V5_PHRASE_PROGRAM_KIND",
    "compile_normalization_recovery_v5_phrase_program",
    "execute_normalization_recovery_v5_phrase_batch",
    "execute_normalization_recovery_v5_phrase_program",
    "normalization_recovery_v5_defeater_matches",
    "reference_normalization_recovery_v5_phrase_batch",
    "reference_normalization_recovery_v5_phrase_program",
]
