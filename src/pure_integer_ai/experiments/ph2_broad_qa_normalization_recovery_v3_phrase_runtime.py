"""编译并执行 recovery-v3 禁用态多长度 phrase program。

执行顺序固定为 whole-input exact、按首 scalar 分桶的最长匹配、单 scalar
character composition backoff。context defeater 只阻断当前 phrase rule，不写回
学习状态，也不改变公开 production gate。
"""
from __future__ import annotations

from collections import defaultdict
import hashlib

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v3_learning_records import (
    NORMALIZATION_RECOVERY_V3_DEFEATER_KIND,
    NORMALIZATION_RECOVERY_V3_OVERLAP_INDEX_KIND,
    NORMALIZATION_RECOVERY_V3_PHRASE_RULE_KIND,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v3_training_records import (
    RECOVERY_V3_TARGET_POLICY_SCOPE,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes


NORMALIZATION_RECOVERY_V3_PHRASE_PROGRAM_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V3_PHRASE_PROGRAM_V1")
NORMALIZATION_RECOVERY_V3_PHRASE_PROGRAM_STATUS = (
    "CANDIDATE_ONLY_PUBLIC_PRODUCTION_DISABLED")


def _sha_value(value: object, *, label: str) -> str:
    """核验并返回小写 SHA-256。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(item not in "0123456789abcdef" for item in value)):
        raise BroadQaExternalDataError(f"{label} 非法")
    return value


def _text(value: object, *, label: str) -> str:
    """核验非空且不含 surrogate 的运行文本。"""
    if (not isinstance(value, str) or not value
            or any(0xD800 <= ord(item) <= 0xDFFF for item in value)):
        raise BroadQaExternalDataError(f"{label} 非法")
    return value


def _output_text(value: object, *, label: str) -> str:
    """核验允许删除规则空串、但不含 surrogate 的输出文本。"""
    if (not isinstance(value, str)
            or any(0xD800 <= ord(item) <= 0xDFFF for item in value)):
        raise BroadQaExternalDataError(f"{label} 非法")
    return value


def _program_payload(program: dict[str, object]) -> dict[str, object]:
    """移除自摘要字段后返回 program identity payload。"""
    return {key: value for key, value in program.items()
            if key != "program_sha256"}


def compile_normalization_recovery_v3_phrase_program(
        *,
        rule_pack_manifest_sha256: str,
        phrase_rules: tuple[dict[str, object], ...],
        defeaters: tuple[dict[str, object], ...],
        overlap_index: tuple[dict[str, object], ...],
        ) -> dict[str, object]:
    """把严格回读的 pack records 编译为无行为对象的禁用态 program。"""
    pack_sha = _sha_value(
        rule_pack_manifest_sha256, label="v3 phrase pack manifest")
    if (not isinstance(phrase_rules, tuple)
            or not isinstance(defeaters, tuple)
            or not isinstance(overlap_index, tuple)):
        raise BroadQaExternalDataError("v3 phrase program 输入类型漂移")
    rule_by_id = {}
    input_texts = set()
    for rule in phrase_rules:
        rule_id = _sha_value(
            rule.get("rule_id") if isinstance(rule, dict) else None,
            label="v3 phrase rule id")
        input_text = _text(rule.get("input_text"), label="v3 phrase input")
        output_text = _output_text(
            rule.get("output_text"), label="v3 phrase output")
        scope = rule.get("application_scope")
        if (rule.get("record_kind") != NORMALIZATION_RECOVERY_V3_PHRASE_RULE_KIND
                or rule.get("runtime_state") != "LEARNED_PACK_DISABLED"
                or rule.get("production_enabled") != 0
                or rule.get("target_policy_scope")
                != RECOVERY_V3_TARGET_POLICY_SCOPE
                or scope != {
                    "defeater_required": 1,
                    "input_match": "EXACT_SCALAR_SUBSEQUENCE",
                    "unscoped_execution_allowed": 0,
                }
                or not isinstance(rule.get("positive_evidence_ids"), list)
                or not rule["positive_evidence_ids"]
                or not isinstance(rule.get("negative_evidence_ids"), list)
                or not rule["negative_evidence_ids"]
                or not isinstance(rule.get("defeater_ids"), list)
                or not rule["defeater_ids"]
                or rule_id in rule_by_id or input_text in input_texts):
            raise BroadQaExternalDataError(
                "v3 phrase rule scope/Evidence/identity 漂移")
        if (rule.get("equal_length") != int(len(input_text) == len(output_text))
                or rule.get("variable_length")
                != int(len(input_text) != len(output_text))):
            raise BroadQaExternalDataError("v3 phrase rule 长度分账漂移")
        rule_by_id[rule_id] = rule
        input_texts.add(input_text)

    defeater_by_id = {}
    defeaters_by_rule: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in defeaters:
        defeater_id = _sha_value(
            record.get("defeater_id") if isinstance(record, dict) else None,
            label="v3 phrase defeater id")
        rule_id = _sha_value(record.get("rule_id"), label="v3 defeater rule")
        left = record.get("left_context")
        right = record.get("right_context")
        if (record.get("record_kind") != NORMALIZATION_RECOVERY_V3_DEFEATER_KIND
                or record.get("action") != "BLOCK_PHRASE_RULE_USE_BACKOFF"
                or record.get("production_enabled") != 0
                or rule_id not in rule_by_id
                or not isinstance(left, str) or len(left) > 4
                or not isinstance(right, str) or len(right) > 4
                or record.get("left_boundary") not in {0, 1}
                or record.get("right_boundary") not in {0, 1}
                or not isinstance(record.get("refute_evidence_ids"), list)
                or not record["refute_evidence_ids"]
                or defeater_id in defeater_by_id):
            raise BroadQaExternalDataError("v3 phrase defeater schema 漂移")
        defeater_by_id[defeater_id] = record
        defeaters_by_rule[rule_id].append(record)
    for rule_id, rule in rule_by_id.items():
        declared = sorted(str(value) for value in rule["defeater_ids"])
        actual = sorted(str(item["defeater_id"])
                        for item in defeaters_by_rule.get(rule_id, ()))
        if declared != actual:
            raise BroadQaExternalDataError(
                "v3 phrase rule/defeater 未闭合")

    index_by_rule = {}
    buckets: dict[int, list[dict[str, object]]] = defaultdict(list)
    for record in overlap_index:
        index_id = _sha_value(
            record.get("index_id") if isinstance(record, dict) else None,
            label="v3 overlap index id")
        rule_id = _sha_value(record.get("rule_id"), label="v3 overlap rule")
        rule = rule_by_id.get(rule_id)
        if rule is None:
            raise BroadQaExternalDataError("v3 overlap index rule 缺失")
        input_text = str(rule["input_text"])
        if (record.get("record_kind")
                != NORMALIZATION_RECOVERY_V3_OVERLAP_INDEX_KIND
                or record.get("candidate_input") != input_text
                or record.get("first_scalar") != ord(input_text[0])
                or record.get("input_scalar_length") != len(input_text)
                or record.get("priority_contract") != (
                    "WHOLE_INPUT_EXACT_THEN_LONGEST_MATCH_THEN_CHARACTER_BACKOFF")
                or rule_id in index_by_rule):
            raise BroadQaExternalDataError("v3 overlap index 漂移")
        index_by_rule[rule_id] = index_id
        buckets[ord(input_text[0])].append(rule)
    if set(index_by_rule) != set(rule_by_id):
        raise BroadQaExternalDataError("v3 overlap index 未覆盖全部规则")

    ordered_buckets = []
    for first_scalar, values in sorted(buckets.items()):
        ordered = sorted(values, key=lambda item: (
            -len(str(item["input_text"])),
            str(item["input_text"]),
            str(item["rule_id"]),
        ))
        ordered_buckets.append({
            "first_scalar": first_scalar,
            "rules": ordered,
        })
    program = {
        "artifact_kind": NORMALIZATION_RECOVERY_V3_PHRASE_PROGRAM_KIND,
        "buckets": ordered_buckets,
        "defeaters": sorted(defeaters, key=lambda item: str(item["defeater_id"])),
        "format_version": 1,
        "production_enabled": 0,
        "rule_pack_manifest_sha256": pack_sha,
        "status": NORMALIZATION_RECOVERY_V3_PHRASE_PROGRAM_STATUS,
        "target_policy_scope": RECOVERY_V3_TARGET_POLICY_SCOPE,
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
    if record["left_boundary"] == 1:
        left_match = start == 0
    else:
        left_match = start >= len(left) and text[start - len(left):start] == left
    if record["right_boundary"] == 1:
        right_match = end == len(text)
    else:
        right_match = text[end:end + len(right)] == right
    return left_match and right_match


def normalization_recovery_v3_defeater_matches(
        record: dict[str, object],
        text: str,
        start: int,
        end: int,
        ) -> bool:
    """严格核验并执行一个 pack-derived exact-context defeater。"""
    source = _text(text, label="v3 defeater runtime input")
    if (not isinstance(record, dict)
            or record.get("record_kind")
            != NORMALIZATION_RECOVERY_V3_DEFEATER_KIND
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
        raise BroadQaExternalDataError("v3 defeater predicate 输入漂移")
    return _defeater_matches(record, source, start, end)


def _validated_character_rules(
        value: dict[str, str] | None,
        ) -> dict[str, str]:
    """核验 character backoff 只含一 scalar 到一 scalar 映射。"""
    if value is None:
        return {}
    if (not isinstance(value, dict)
            or any(not isinstance(key, str) or len(key) != 1
                   or not isinstance(output, str) or len(output) != 1
                   for key, output in value.items())):
        raise BroadQaExternalDataError("v3 character backoff 非法")
    return value


def _validated_program(program: dict[str, object]) -> tuple[
        dict[int, tuple[dict[str, object], ...]],
        dict[str, tuple[dict[str, object], ...]],
        ]:
    """核验 program 自摘要、禁用态和 bucket/defeater 引用。"""
    if (not isinstance(program, dict)
            or program.get("artifact_kind")
            != NORMALIZATION_RECOVERY_V3_PHRASE_PROGRAM_KIND
            or program.get("status")
            != NORMALIZATION_RECOVERY_V3_PHRASE_PROGRAM_STATUS
            or program.get("production_enabled") != 0
            or program.get("target_policy_scope")
            != RECOVERY_V3_TARGET_POLICY_SCOPE
            or program.get("program_sha256") != hashlib.sha256(
                canonical_json_bytes(_program_payload(program))).hexdigest()
            or not isinstance(program.get("buckets"), list)
            or not isinstance(program.get("defeaters"), list)):
        raise BroadQaExternalDataError("v3 phrase program identity 漂移")
    buckets = {}
    rule_ids = set()
    for bucket in program["buckets"]:
        if (not isinstance(bucket, dict)
                or set(bucket) != {"first_scalar", "rules"}
                or type(bucket["first_scalar"]) is not int
                or not isinstance(bucket["rules"], list)
                or bucket["first_scalar"] in buckets):
            raise BroadQaExternalDataError("v3 phrase program bucket 漂移")
        rules = tuple(bucket["rules"])
        expected = tuple(sorted(rules, key=lambda item: (
            -len(str(item["input_text"])),
            str(item["input_text"]),
            str(item["rule_id"]),
        )))
        if (rules != expected or any(
                ord(str(item["input_text"])[0]) != bucket["first_scalar"]
                or str(item["rule_id"]) in rule_ids for item in rules)):
            raise BroadQaExternalDataError("v3 phrase program bucket 顺序漂移")
        rule_ids.update(str(item["rule_id"]) for item in rules)
        buckets[bucket["first_scalar"]] = rules
    defeaters_by_rule: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in program["defeaters"]:
        rule_id = str(record.get("rule_id"))
        if rule_id not in rule_ids:
            raise BroadQaExternalDataError("v3 phrase program defeater rule 缺失")
        defeaters_by_rule[rule_id].append(record)
    return (
        buckets,
        {key: tuple(sorted(values, key=lambda item: str(item["defeater_id"])))
         for key, values in defeaters_by_rule.items()},
    )


def _execute_validated(
        *,
        source: str,
        program_sha256: str,
        buckets: dict[int, tuple[dict[str, object], ...]],
        defeaters_by_rule: dict[str, tuple[dict[str, object], ...]],
        backoff: dict[str, str],
        indexed: bool,
        ) -> dict[str, object]:
    """在已核验 program/index 上执行一个输入。"""
    if indexed:
        candidates_at = lambda position: buckets.get(ord(source[position]), ())
    else:
        all_rules = tuple(
            rule for bucket in buckets.values() for rule in bucket)
        candidates_at = lambda position: tuple(sorted(
            (item for item in all_rules
             if str(item["input_text"])[0] == source[position]),
            key=lambda item: (
                -len(str(item["input_text"])),
                str(item["input_text"]),
                str(item["rule_id"]),
            )))

    def choose(position: int, *, whole_only: bool):
        blocked = []
        for rule in candidates_at(position):
            phrase = str(rule["input_text"])
            end = position + len(phrase)
            if ((whole_only and (position != 0 or end != len(source)))
                    or end > len(source)
                    or source[position:end] != phrase):
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
            "input_end": len(source),
            "input_start": 0,
            "mode": "WHOLE_INPUT_EXACT",
            "output_text": output_text,
            "rule_id": exact["rule_id"],
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
                mode = "LONGEST_PHRASE_MATCH"
                rule_id = str(rule["rule_id"])
            else:
                end = position + 1
                output_text = backoff.get(source[position], source[position])
                mode = "CHARACTER_COMPOSITION_BACKOFF"
                rule_id = ""
            output.append(output_text)
            steps.append({
                "blocked_defeater_ids": blocked_ids,
                "input_end": end,
                "input_start": position,
                "mode": mode,
                "output_text": output_text,
                "rule_id": rule_id,
            })
            position = end
    result = {
        "input_text": source,
        "output_text": "".join(output),
        "production_enabled": 0,
        "program_sha256": program_sha256,
        "steps": steps,
        "target_policy_scope": RECOVERY_V3_TARGET_POLICY_SCOPE,
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
        character_rules: dict[str, str] | None,
        indexed: bool,
        ) -> dict[str, object]:
    """执行 indexed 或独立 linear reference 路径。"""
    source = _text(text, label="v3 phrase runtime input")
    buckets, defeaters_by_rule = _validated_program(program)
    backoff = _validated_character_rules(character_rules)
    return _execute_validated(
        source=source,
        program_sha256=str(program["program_sha256"]),
        buckets=buckets,
        defeaters_by_rule=defeaters_by_rule,
        backoff=backoff,
        indexed=indexed,
    )


def _execute_batch(
        program: dict[str, object],
        texts: tuple[str, ...],
        *,
        character_rules: dict[str, str] | None,
        indexed: bool,
        ) -> tuple[dict[str, object], ...]:
    """一次核验 program 后按输入序执行完整 batch。"""
    if (not isinstance(texts, tuple) or not texts
            or any(not isinstance(item, str) for item in texts)):
        raise BroadQaExternalDataError("v3 phrase batch 输入非法")
    sources = tuple(_text(item, label="v3 phrase batch input") for item in texts)
    buckets, defeaters_by_rule = _validated_program(program)
    backoff = _validated_character_rules(character_rules)
    return tuple(_execute_validated(
        source=source,
        program_sha256=str(program["program_sha256"]),
        buckets=buckets,
        defeaters_by_rule=defeaters_by_rule,
        backoff=backoff,
        indexed=indexed,
    ) for source in sources)


def execute_normalization_recovery_v3_phrase_program(
        program: dict[str, object],
        text: str,
        *,
        character_rules: dict[str, str] | None = None,
        ) -> dict[str, object]:
    """使用首 scalar bucket 与最长匹配执行禁用态 candidate。"""
    return _execute(
        program, text, character_rules=character_rules, indexed=True)


def reference_normalization_recovery_v3_phrase_program(
        program: dict[str, object],
        text: str,
        *,
        character_rules: dict[str, str] | None = None,
        ) -> dict[str, object]:
    """用全规则线性扫描独立重放同一语义。"""
    return _execute(
        program, text, character_rules=character_rules, indexed=False)


def execute_normalization_recovery_v3_phrase_batch(
        program: dict[str, object],
        texts: tuple[str, ...],
        *,
        character_rules: dict[str, str] | None = None,
        ) -> tuple[dict[str, object], ...]:
    """一次校验 program 后以 indexed 路径执行多个输入。"""
    return _execute_batch(
        program, texts, character_rules=character_rules, indexed=True)


def reference_normalization_recovery_v3_phrase_batch(
        program: dict[str, object],
        texts: tuple[str, ...],
        *,
        character_rules: dict[str, str] | None = None,
        ) -> tuple[dict[str, object], ...]:
    """一次校验 program 后以 linear reference 执行多个输入。"""
    return _execute_batch(
        program, texts, character_rules=character_rules, indexed=False)


__all__ = [
    "NORMALIZATION_RECOVERY_V3_PHRASE_PROGRAM_KIND",
    "NORMALIZATION_RECOVERY_V3_PHRASE_PROGRAM_STATUS",
    "compile_normalization_recovery_v3_phrase_program",
    "execute_normalization_recovery_v3_phrase_batch",
    "execute_normalization_recovery_v3_phrase_program",
    "normalization_recovery_v3_defeater_matches",
    "reference_normalization_recovery_v3_phrase_batch",
    "reference_normalization_recovery_v3_phrase_program",
]
