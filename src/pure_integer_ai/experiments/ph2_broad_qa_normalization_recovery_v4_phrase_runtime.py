"""编译并执行 recovery-v4 target/source-scoped phrase program。

无 source family 时只能执行跨来源 target rule；显式给出 source family 时仅叠加
同一 family 的 source rule。两类规则共享 whole-input、最长匹配与 context
defeater 语义，且始终保持生产禁用。
"""
from __future__ import annotations

from collections import defaultdict
import hashlib

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v4_learning_contract import (
    NORMALIZATION_RECOVERY_V4_DEFEATER_KIND,
    NORMALIZATION_RECOVERY_V4_SOURCE_OVERLAP_KIND,
    NORMALIZATION_RECOVERY_V4_SOURCE_RULE_KIND,
    NORMALIZATION_RECOVERY_V4_TARGET_OVERLAP_KIND,
    NORMALIZATION_RECOVERY_V4_TARGET_RULE_KIND,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v4_training_records import (
    RECOVERY_V4_TARGET_POLICY_SCOPE,
    V4_SOURCE_FAMILIES,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes


NORMALIZATION_RECOVERY_V4_PHRASE_PROGRAM_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V4_PHRASE_PROGRAM_V1")
NORMALIZATION_RECOVERY_V4_PHRASE_PROGRAM_STATUS = (
    "CANDIDATE_ONLY_PUBLIC_PRODUCTION_DISABLED")


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


def _rule_order(rule: dict[str, object]):
    """固定最长匹配与 target-before-source 的完全序。"""
    return (
        -len(str(rule["input_text"])),
        str(rule["input_text"]),
        0 if rule["candidate_scope_kind"] == "TARGET_CROSS_FAMILY" else 1,
        str(rule["rule_id"]),
    )


def _validate_rule(
        rule: dict[str, object],
        *,
        expected_scope_kind: str,
        expected_record_kind: str,
        ) -> tuple[str, str, str]:
    """核验一条禁用态 scoped rule 并返回 id/input/source family。"""
    rule_id = _sha_value(rule.get("rule_id"), label="v4 phrase rule id")
    input_text = _text(rule.get("input_text"), label="v4 phrase input")
    output_text = _text(
        rule.get("output_text"), label="v4 phrase output", allow_empty=True)
    source_family = rule.get("source_execution_family")
    source_policy = rule.get("source_execution_policy_scope")
    target_scope = rule.get("target_policy_scope")
    scope = rule.get("application_scope")
    if expected_scope_kind == "TARGET_CROSS_FAMILY":
        scope_valid = (
            source_family == "" and source_policy == ""
            and target_scope == RECOVERY_V4_TARGET_POLICY_SCOPE)
    else:
        scope_valid = (
            source_family in V4_SOURCE_FAMILIES
            and isinstance(source_policy, str) and bool(source_policy)
            and target_scope == "")
    expected_application = {
        "candidate_scope_kind": expected_scope_kind,
        "defeater_required": 1,
        "input_match": (
            "WHOLE_INPUT_EXACT"
            if expected_scope_kind == "TARGET_CROSS_FAMILY"
            else "EXACT_SCALAR_SUBSEQUENCE"),
        "source_execution_family": source_family,
        "source_execution_policy_scope": source_policy,
        "unscoped_execution_allowed": 0,
    }
    if (rule.get("record_kind") != expected_record_kind
            or rule.get("candidate_scope_kind") != expected_scope_kind
            or not scope_valid or scope != expected_application
            or rule.get("runtime_state") != "LEARNED_PACK_DISABLED"
            or rule.get("production_enabled") != 0
            or not isinstance(rule.get("positive_evidence_ids"), list)
            or not rule["positive_evidence_ids"]
            or not isinstance(rule.get("negative_evidence_ids"), list)
            or not rule["negative_evidence_ids"]
            or not isinstance(rule.get("defeater_ids"), list)
            or not rule["defeater_ids"]
            or rule.get("equal_length")
            != int(len(input_text) == len(output_text))
            or rule.get("variable_length")
            != int(len(input_text) != len(output_text))):
        raise BroadQaExternalDataError(
            "v4 phrase rule scope/Evidence/schema 漂移")
    return rule_id, input_text, str(source_family)


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


def compile_normalization_recovery_v4_phrase_program(
        *,
        rule_pack_manifest_sha256: str,
        target_phrase_rules: tuple[dict[str, object], ...],
        source_phrase_rules: tuple[dict[str, object], ...],
        defeaters: tuple[dict[str, object], ...],
        target_overlap_index: tuple[dict[str, object], ...],
        source_overlap_index: tuple[dict[str, object], ...],
        ) -> dict[str, object]:
    """把严格回读的 v4 pack records 编译为禁用态 scoped program。"""
    pack_sha = _sha_value(
        rule_pack_manifest_sha256, label="v4 phrase pack manifest")
    inputs = (target_phrase_rules, source_phrase_rules, defeaters,
              target_overlap_index, source_overlap_index)
    if any(not isinstance(value, tuple) for value in inputs):
        raise BroadQaExternalDataError("v4 phrase program 输入类型漂移")
    rule_by_id = {}
    target_input_texts = set()
    source_input_texts: dict[str, set[str]] = defaultdict(set)
    for rule in target_phrase_rules:
        if not isinstance(rule, dict):
            raise BroadQaExternalDataError("v4 target phrase rule 非字典")
        rule_id, input_text, _source = _validate_rule(
            rule,
            expected_scope_kind="TARGET_CROSS_FAMILY",
            expected_record_kind=NORMALIZATION_RECOVERY_V4_TARGET_RULE_KIND,
        )
        if rule_id in rule_by_id or input_text in target_input_texts:
            raise BroadQaExternalDataError("v4 target phrase rule identity 重复")
        rule_by_id[rule_id] = rule
        target_input_texts.add(input_text)
    for rule in source_phrase_rules:
        if not isinstance(rule, dict):
            raise BroadQaExternalDataError("v4 source phrase rule 非字典")
        rule_id, input_text, source_family = _validate_rule(
            rule,
            expected_scope_kind="SOURCE_ONLY",
            expected_record_kind=NORMALIZATION_RECOVERY_V4_SOURCE_RULE_KIND,
        )
        if (rule_id in rule_by_id
                or input_text in source_input_texts[source_family]
                or input_text in target_input_texts):
            raise BroadQaExternalDataError(
                "v4 source phrase rule identity/scope 重复")
        rule_by_id[rule_id] = rule
        source_input_texts[source_family].add(input_text)

    defeater_by_id = {}
    defeaters_by_rule: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in defeaters:
        if not isinstance(record, dict):
            raise BroadQaExternalDataError("v4 phrase defeater 非字典")
        defeater_id = _sha_value(
            record.get("defeater_id"), label="v4 phrase defeater id")
        rule_id = _sha_value(
            record.get("rule_id"), label="v4 defeater rule")
        rule = rule_by_id.get(rule_id)
        left = record.get("left_context")
        right = record.get("right_context")
        if (rule is None
                or record.get("record_kind")
                != NORMALIZATION_RECOVERY_V4_DEFEATER_KIND
                or record.get("action") != "BLOCK_PHRASE_RULE_USE_BACKOFF"
                or record.get("production_enabled") != 0
                or any(record.get(key) != value
                       for key, value in _scope_identity(rule).items())
                or not isinstance(left, str) or len(left) > 4
                or not isinstance(right, str) or len(right) > 4
                or record.get("left_boundary") not in {0, 1}
                or record.get("right_boundary") not in {0, 1}
                or not isinstance(record.get("refute_evidence_ids"), list)
                or not record["refute_evidence_ids"]
                or defeater_id in defeater_by_id):
            raise BroadQaExternalDataError("v4 phrase defeater schema 漂移")
        defeater_by_id[defeater_id] = record
        defeaters_by_rule[rule_id].append(record)
    for rule_id, rule in rule_by_id.items():
        declared = sorted(str(value) for value in rule["defeater_ids"])
        actual = sorted(str(item["defeater_id"])
                        for item in defeaters_by_rule.get(rule_id, ()))
        if declared != actual:
            raise BroadQaExternalDataError(
                "v4 phrase rule/defeater 未闭合")

    index_by_rule = {}
    index_specs = (
        (target_overlap_index, NORMALIZATION_RECOVERY_V4_TARGET_OVERLAP_KIND,
         "TARGET_CROSS_FAMILY"),
        (source_overlap_index, NORMALIZATION_RECOVERY_V4_SOURCE_OVERLAP_KIND,
         "SOURCE_ONLY"),
    )
    for records, expected_kind, expected_scope in index_specs:
        for record in records:
            if not isinstance(record, dict):
                raise BroadQaExternalDataError("v4 overlap index 非字典")
            index_id = _sha_value(
                record.get("index_id"), label="v4 overlap index id")
            rule_id = _sha_value(
                record.get("rule_id"), label="v4 overlap rule")
            rule = rule_by_id.get(rule_id)
            if rule is None:
                raise BroadQaExternalDataError("v4 overlap index rule 缺失")
            input_text = str(rule["input_text"])
            expected_priority = (
                "WHOLE_INPUT_EXACT_ONLY"
                if expected_scope == "TARGET_CROSS_FAMILY"
                else "WHOLE_INPUT_EXACT_THEN_LONGEST_MATCH_THEN_CHARACTER_BACKOFF")
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
                    or record.get("priority_contract") != expected_priority
                    or rule_id in index_by_rule):
                raise BroadQaExternalDataError("v4 overlap index 漂移")
            index_by_rule[rule_id] = index_id
    if set(index_by_rule) != set(rule_by_id):
        raise BroadQaExternalDataError("v4 overlap index 未覆盖全部规则")

    source_programs = []
    for source_family in sorted(source_input_texts):
        rules = [rule for rule in source_phrase_rules
                 if rule["source_execution_family"] == source_family]
        source_programs.append({
            "buckets": _ordered_buckets(rules),
            "source_family": source_family,
        })
    program = {
        "artifact_kind": NORMALIZATION_RECOVERY_V4_PHRASE_PROGRAM_KIND,
        "defeaters": sorted(defeaters,
                            key=lambda item: str(item["defeater_id"])),
        "format_version": 1,
        "production_enabled": 0,
        "rule_pack_manifest_sha256": pack_sha,
        "source_programs": source_programs,
        "status": NORMALIZATION_RECOVERY_V4_PHRASE_PROGRAM_STATUS,
        "target_buckets": _ordered_buckets(list(target_phrase_rules)),
        "target_policy_scope": RECOVERY_V4_TARGET_POLICY_SCOPE,
        "unscoped_source_rule_execution_allowed": 0,
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


def normalization_recovery_v4_defeater_matches(
        record: dict[str, object],
        text: str,
        start: int,
        end: int,
        ) -> bool:
    """严格核验并执行一个 v4 pack-derived context defeater。"""
    source = _text(text, label="v4 defeater runtime input")
    if (not isinstance(record, dict)
            or record.get("record_kind")
            != NORMALIZATION_RECOVERY_V4_DEFEATER_KIND
            or record.get("action") != "BLOCK_PHRASE_RULE_USE_BACKOFF"
            or record.get("production_enabled") != 0
            or not isinstance(record.get("left_context"), str)
            or len(record["left_context"]) > 4
            or not isinstance(record.get("right_context"), str)
            or len(record["right_context"]) > 4
            or record.get("left_boundary") not in {0, 1}
            or record.get("right_boundary") not in {0, 1}
            or type(start) is not int or type(end) is not int
            or not 0 <= start < end <= len(source)):
        raise BroadQaExternalDataError("v4 defeater predicate 输入漂移")
    return _defeater_matches(record, source, start, end)


def _validated_character_rules(value: dict[str, str] | None) -> dict[str, str]:
    """核验 character backoff 只含一 scalar 到一 scalar 映射。"""
    if value is None:
        return {}
    if (not isinstance(value, dict)
            or any(not isinstance(key, str) or len(key) != 1
                   or not isinstance(output, str) or len(output) != 1
                   for key, output in value.items())):
        raise BroadQaExternalDataError("v4 character backoff 非法")
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
        if (rules != tuple(sorted(rules, key=_rule_order))
                or any(ord(str(item["input_text"])[0])
                       != bucket["first_scalar"]
                       or str(item["rule_id"]) in rule_ids for item in rules)):
            raise BroadQaExternalDataError(f"{label} 顺序漂移")
        rule_ids.update(str(item["rule_id"]) for item in rules)
        buckets[bucket["first_scalar"]] = rules
    return buckets


def _validated_program(
        program: dict[str, object],
        source_family: str,
        ) -> tuple[dict[int, tuple[dict[str, object], ...]],
                   dict[int, tuple[dict[str, object], ...]],
                   dict[str, tuple[dict[str, object], ...]]]:
    """核验 program 自摘要、source 路由和 defeater 引用。"""
    if source_family and source_family not in V4_SOURCE_FAMILIES:
        raise BroadQaExternalDataError("v4 runtime source family 非法")
    if (not isinstance(program, dict)
            or program.get("artifact_kind")
            != NORMALIZATION_RECOVERY_V4_PHRASE_PROGRAM_KIND
            or program.get("status")
            != NORMALIZATION_RECOVERY_V4_PHRASE_PROGRAM_STATUS
            or program.get("production_enabled") != 0
            or program.get("target_policy_scope")
            != RECOVERY_V4_TARGET_POLICY_SCOPE
            or program.get("unscoped_source_rule_execution_allowed") != 0
            or program.get("program_sha256") != hashlib.sha256(
                canonical_json_bytes(_program_payload(program))).hexdigest()
            or not isinstance(program.get("source_programs"), list)
            or not isinstance(program.get("defeaters"), list)):
        raise BroadQaExternalDataError("v4 phrase program identity 漂移")
    target_buckets = _validated_buckets(
        program.get("target_buckets"), label="v4 target bucket")
    selected_source_buckets = {}
    seen_sources = set()
    all_rule_ids = {
        str(rule["rule_id"]) for rules in target_buckets.values()
        for rule in rules}
    for source_program in program["source_programs"]:
        if (not isinstance(source_program, dict)
                or set(source_program) != {"buckets", "source_family"}
                or source_program["source_family"] not in V4_SOURCE_FAMILIES
                or source_program["source_family"] in seen_sources):
            raise BroadQaExternalDataError("v4 source program roster 漂移")
        family = str(source_program["source_family"])
        seen_sources.add(family)
        buckets = _validated_buckets(
            source_program["buckets"], label="v4 source bucket")
        family_rule_ids = {
            str(rule["rule_id"]) for rules in buckets.values()
            for rule in rules}
        if all_rule_ids.intersection(family_rule_ids):
            raise BroadQaExternalDataError("v4 program rule identity 重复")
        all_rule_ids.update(family_rule_ids)
        if family == source_family:
            selected_source_buckets = buckets
    defeaters_by_rule: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in program["defeaters"]:
        rule_id = str(record.get("rule_id"))
        if rule_id not in all_rule_ids:
            raise BroadQaExternalDataError("v4 program defeater rule 缺失")
        defeaters_by_rule[rule_id].append(record)
    return (
        target_buckets,
        selected_source_buckets,
        {key: tuple(sorted(values,
                           key=lambda item: str(item["defeater_id"])))
         for key, values in defeaters_by_rule.items()},
    )


def _execute_validated(
        *,
        source: str,
        source_family: str,
        program_sha256: str,
        target_buckets: dict[int, tuple[dict[str, object], ...]],
        source_buckets: dict[int, tuple[dict[str, object], ...]],
        defeaters_by_rule: dict[str, tuple[dict[str, object], ...]],
        backoff: dict[str, str],
        indexed: bool,
        ) -> dict[str, object]:
    """在已核验 target/source index 上执行一个输入。"""
    def bucket_candidates(position: int):
        scalar = ord(source[position])
        values = target_buckets.get(scalar, ()) + source_buckets.get(scalar, ())
        return tuple(sorted(values, key=_rule_order))

    all_rules = tuple(
        rule for buckets in (target_buckets, source_buckets)
        for values in buckets.values() for rule in values)

    def candidates_at(position: int):
        if indexed:
            return bucket_candidates(position)
        return tuple(sorted((
            item for item in all_rules
            if str(item["input_text"])[0] == source[position]),
            key=_rule_order,
        ))

    def choose(position: int, *, whole_only: bool):
        blocked = []
        for rule in candidates_at(position):
            phrase = str(rule["input_text"])
            end = position + len(phrase)
            if ((whole_only and (position != 0 or end != len(source)))
                    or (not whole_only
                        and rule["candidate_scope_kind"]
                        == "TARGET_CROSS_FAMILY")
                    or end > len(source) or source[position:end] != phrase):
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
            "input_end": len(source),
            "input_start": 0,
            "mode": "WHOLE_INPUT_EXACT",
            "output_text": output_text,
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
                mode = "LONGEST_PHRASE_MATCH"
            else:
                phrase = source[position]
                end = position + 1
                output_text = backoff.get(phrase, phrase)
                step_scope = "NONE"
                rule_id = ""
                rule_source = ""
                mode = "CHARACTER_BACKOFF" if phrase in backoff else "IDENTITY"
            steps.append({
                "blocked_defeater_ids": blocked_ids,
                "candidate_scope_kind": step_scope,
                "input_end": end,
                "input_start": position,
                "mode": mode,
                "output_text": output_text,
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
        "target_policy_scope": RECOVERY_V4_TARGET_POLICY_SCOPE,
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
        character_rules: dict[str, str] | None,
        indexed: bool,
        ) -> dict[str, object]:
    """核验 program/input 后执行 indexed 或 reference 路径。"""
    source = _text(text, label="v4 phrase runtime input")
    target_buckets, source_buckets, defeaters = _validated_program(
        program, source_family)
    return _execute_validated(
        source=source,
        source_family=source_family,
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
        character_rules: dict[str, str] | None,
        indexed: bool,
        ) -> tuple[dict[str, object], ...]:
    """一次核验 program 后批量执行同一 source family 的输入。"""
    if not isinstance(texts, tuple):
        raise BroadQaExternalDataError("v4 phrase batch 输入类型漂移")
    target_buckets, source_buckets, defeaters = _validated_program(
        program, source_family)
    backoff = _validated_character_rules(character_rules)
    return tuple(_execute_validated(
        source=_text(item, label="v4 phrase runtime input"),
        source_family=source_family,
        program_sha256=str(program["program_sha256"]),
        target_buckets=target_buckets,
        source_buckets=source_buckets,
        defeaters_by_rule=defeaters,
        backoff=backoff,
        indexed=indexed,
    ) for item in texts)


def execute_normalization_recovery_v4_phrase_program(
        program: dict[str, object],
        text: str,
        *,
        source_family: str = "",
        character_rules: dict[str, str] | None = None,
        ) -> dict[str, object]:
    """执行按首 scalar 分桶的 v4 indexed program。"""
    return _execute(
        program, text, source_family=source_family,
        character_rules=character_rules, indexed=True)


def reference_normalization_recovery_v4_phrase_program(
        program: dict[str, object],
        text: str,
        *,
        source_family: str = "",
        character_rules: dict[str, str] | None = None,
        ) -> dict[str, object]:
    """执行不使用首 scalar index 的 v4 reference program。"""
    return _execute(
        program, text, source_family=source_family,
        character_rules=character_rules, indexed=False)


def execute_normalization_recovery_v4_phrase_batch(
        program: dict[str, object],
        texts: tuple[str, ...],
        *,
        source_family: str = "",
        character_rules: dict[str, str] | None = None,
        ) -> tuple[dict[str, object], ...]:
    """批量执行 v4 indexed program。"""
    return _execute_batch(
        program, texts, source_family=source_family,
        character_rules=character_rules, indexed=True)


def reference_normalization_recovery_v4_phrase_batch(
        program: dict[str, object],
        texts: tuple[str, ...],
        *,
        source_family: str = "",
        character_rules: dict[str, str] | None = None,
        ) -> tuple[dict[str, object], ...]:
    """批量执行 v4 reference program。"""
    return _execute_batch(
        program, texts, source_family=source_family,
        character_rules=character_rules, indexed=False)


__all__ = [
    "NORMALIZATION_RECOVERY_V4_PHRASE_PROGRAM_KIND",
    "compile_normalization_recovery_v4_phrase_program",
    "execute_normalization_recovery_v4_phrase_batch",
    "execute_normalization_recovery_v4_phrase_program",
    "normalization_recovery_v4_defeater_matches",
    "reference_normalization_recovery_v4_phrase_batch",
    "reference_normalization_recovery_v4_phrase_program",
]
