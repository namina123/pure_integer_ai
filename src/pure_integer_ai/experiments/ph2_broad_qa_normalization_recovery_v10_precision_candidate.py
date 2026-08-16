"""从既有 TRAIN 证据编译 recovery-v10 precision-first 候选。

本模块不读取路径，也不接触任何已消费 formal 来源。v1 保留三类整串路由供
诊断；v2 仅允许三来源 exact-source 规则提交，identity 与 orthographic 只形成
非提交 trace。单字符命中、局部组合和部分提交一律禁止。
"""
from __future__ import annotations

from bisect import bisect_left
from collections import Counter
import hashlib

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_localization_structure import (
    localization_structure_tokens,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_candidate import (
    build_normalization_recovery_v8_candidate_index,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_training_records import (
    V8_TRAIN_FAMILIES,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes


NORMALIZATION_RECOVERY_V10_PRECISION_CANDIDATE_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V10_PRECISION_CANDIDATE_V1")
NORMALIZATION_RECOVERY_V10_PRECISION_CANDIDATE_STATUS = (
    "TRAIN_DERIVED_PRECISION_FIRST_DISABLED_NOT_FORMAL")
NORMALIZATION_RECOVERY_V10_PRECISION_TARGET_SCOPE = (
    "ZH_TW_TO_ZH_CN_EXTERNAL_LOCALIZATION_V10_PRECISION_FIRST")
NORMALIZATION_RECOVERY_V10_PRECISION_CANDIDATE_V2_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V10_PRECISION_CANDIDATE_V2")
NORMALIZATION_RECOVERY_V10_PRECISION_CANDIDATE_V2_STATUS = (
    "TRAIN_LOSO_ZERO_WRONG_SOURCE_ONLY_COMMIT_DISABLED_NOT_FORMAL")

_INVENTORY_NAMES = (
    "identity_veto_rules",
    "orthographic_whole_input_rules",
    "source_conditioned_rules",
)
_RUNTIME_ORDER = (
    "TRAIN_IDENTITY_VETO",
    "THREE_FAMILY_SOURCE_CONDITIONED_EXACT",
    "OPENCC_MULTI_FAMILY_WHOLE_INPUT_EXACT",
    "STRUCTURE_PRESERVATION_HARD_GATE",
    "UNKNOWN",
)
_V2_RUNTIME_ORDER = (
    "TRAIN_IDENTITY_VETO_NONCOMMITTING",
    "THREE_FAMILY_SOURCE_CONDITIONED_EXACT_COMMIT",
    "OPENCC_MULTI_FAMILY_WHOLE_INPUT_HYPOTHESIS_NONCOMMITTING",
    "STRUCTURE_PRESERVATION_HARD_GATE",
    "UNKNOWN",
)


def _sha256(value: object) -> str:
    """返回候选、规则、结果或报告的确定性 SHA-256。"""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha_value(value: object, *, label: str) -> str:
    """核验并返回小写 SHA-256。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(item not in "0123456789abcdef" for item in value)):
        raise BroadQaExternalDataError(f"v10 precision {label} 非法")
    return value


def _families(value: object, *, label: str) -> list[str]:
    """核验有序、唯一且属于三家 TRAIN 的来源集合。"""
    if (not isinstance(value, list) or value != sorted(set(value))
            or not value
            or any(item not in V8_TRAIN_FAMILIES for item in value)):
        raise BroadQaExternalDataError(f"v10 precision {label} 漂移")
    return value


def _control_outputs(
        item: dict[str, object],
        ) -> tuple[str, set[str], list[str], int]:
    """核验 exact-input control 并返回输入、输出集合和来源支持。"""
    input_text = item.get("input_text")
    outputs = item.get("outputs")
    support = _families(item.get("support_families"), label="control support")
    count = item.get("support_family_count")
    candidate_id = item.get("candidate_id")
    if (not isinstance(input_text, str) or not input_text
            or not isinstance(outputs, list) or not outputs
            or type(count) is not int or count != len(support)
            or _sha_value(candidate_id, label="control candidate id")
            != candidate_id):
        raise BroadQaExternalDataError("v10 precision control 字段漂移")
    output_values = set()
    observed_families = set()
    for output in outputs:
        if not isinstance(output, dict):
            raise BroadQaExternalDataError(
                "v10 precision control output 非对象")
        output_text = output.get("output_text")
        families = _families(
            output.get("support_families"), label="control output support")
        if (not isinstance(output_text, str) or not output_text
                or output.get("support_family_count") != len(families)):
            raise BroadQaExternalDataError(
                "v10 precision control output 漂移")
        output_values.add(output_text)
        observed_families.update(families)
    if sorted(observed_families) != support:
        raise BroadQaExternalDataError(
            "v10 precision control family 覆盖漂移")
    return input_text, output_values, support, count


def _opencc_character_routes(
        routes: dict[str, str],
        ) -> dict[str, str]:
    """只保留一对一 scalar 路由，拒绝把 phrase 当作字符映射。"""
    if (not isinstance(routes, dict) or not routes
            or any(not isinstance(left, str) or not left
                   or not isinstance(right, str) or not right
                   for left, right in routes.items())):
        raise BroadQaExternalDataError("v10 precision OpenCC routes 非法")
    values = {
        left: right for left, right in routes.items()
        if len(left) == 1 and len(right) == 1 and left != right
    }
    if not values:
        raise BroadQaExternalDataError(
            "v10 precision OpenCC character routes 为空")
    return values


def _compose_opencc_characters(
        input_text: str, routes: dict[str, str],
        ) -> tuple[str, tuple[tuple[str, str], ...]]:
    """按 scalar 顺序组成输出，并返回实际命中的唯一来源路由。"""
    output = []
    matched = []
    for scalar in input_text:
        target = routes.get(scalar)
        if target is None:
            output.append(scalar)
            continue
        output.append(target)
        matched.append((scalar, target))
    return "".join(output), tuple(sorted(set(matched)))


def _safety_indexes(
        observations: tuple[dict[str, object], ...],
        ) -> tuple[
            dict[str, set[str]],
            dict[tuple[str, str], set[str]],
            dict[str, set[str]],
        ]:
    """从全量 TRAIN observation 建立输入、source 条件与来源覆盖账。"""
    if not isinstance(observations, tuple) or not observations:
        raise BroadQaExternalDataError(
            "v10 precision safety observations 为空")
    outputs_by_input: dict[str, set[str]] = {}
    outputs_by_source_input: dict[tuple[str, str], set[str]] = {}
    families_by_input: dict[str, set[str]] = {}
    for item in observations:
        if not isinstance(item, dict):
            raise BroadQaExternalDataError(
                "v10 precision safety observation 非对象")
        input_text = item.get("input_text")
        output_text = item.get("output_text")
        source = item.get("official_source_text")
        family = item.get("source_family")
        if (not isinstance(input_text, str) or not input_text
                or not isinstance(output_text, str) or not output_text
                or not isinstance(source, str) or not source
                or family not in V8_TRAIN_FAMILIES):
            raise BroadQaExternalDataError(
                "v10 precision safety observation 字段漂移")
        outputs_by_input.setdefault(input_text, set()).add(output_text)
        outputs_by_source_input.setdefault(
            (source, input_text), set()).add(output_text)
        families_by_input.setdefault(input_text, set()).add(str(family))
    return outputs_by_input, outputs_by_source_input, families_by_input


def compile_normalization_recovery_v10_precision_candidate(
        *,
        base_candidate: dict[str, object],
        exact_input_controls: tuple[dict[str, object], ...],
        safety_observations: tuple[dict[str, object], ...],
        training_protocol_manifest_sha256: str,
        observation_pack_manifest_sha256: str,
        opencc_routes: dict[str, str],
        opencc_source_pack_manifest_sha256: str,
        ) -> dict[str, object]:
    """把 v8 TRAIN 规则和 OpenCC 来源交叉编译为保守整串候选。"""
    build_normalization_recovery_v8_candidate_index(base_candidate)
    base_sha = _sha_value(
        base_candidate.get("candidate_program_sha256"),
        label="base candidate program")
    protocol_sha = _sha_value(
        training_protocol_manifest_sha256, label="training protocol manifest")
    observation_sha = _sha_value(
        observation_pack_manifest_sha256, label="observation pack manifest")
    opencc_sha = _sha_value(
        opencc_source_pack_manifest_sha256, label="OpenCC source manifest")
    if not isinstance(exact_input_controls, tuple) or not exact_input_controls:
        raise BroadQaExternalDataError(
            "v10 precision exact-input controls 为空")
    character_routes = _opencc_character_routes(opencc_routes)
    (safety_outputs_by_input, safety_outputs_by_source_input,
     safety_families_by_input) = _safety_indexes(safety_observations)

    identity_rules = []
    orthographic_rules = []
    seen_inputs = set()
    for control in exact_input_controls:
        if not isinstance(control, dict):
            raise BroadQaExternalDataError("v10 precision control 非对象")
        input_text, outputs, support, support_count = _control_outputs(control)
        if input_text in seen_inputs:
            raise BroadQaExternalDataError(
                "v10 precision control input 重复")
        seen_inputs.add(input_text)
        control_id = str(control["candidate_id"])
        if (outputs == {input_text}
                and safety_outputs_by_input.get(input_text) == {input_text}):
            identity = {
                "control_candidate_id": control_id,
                "input_text": input_text,
                "rule_kind": "IDENTITY_VETO",
                "training_support_families": support,
                "training_support_family_count": support_count,
            }
            identity_rules.append({
                **identity,
                "candidate_rule_id": _sha256(identity),
            })
        composed, matched = _compose_opencc_characters(
            input_text, character_routes)
        if (support_count >= 2 and composed != input_text
                and outputs == {composed} and matched
                and safety_outputs_by_input.get(input_text) == {composed}):
            route_ids = [
                _sha256({
                    "input_atom": left,
                    "opencc_source_pack_manifest_sha256": opencc_sha,
                    "output_atom": right,
                    "rule_kind": "OPENCC_CHARACTER_ROUTE",
                })
                for left, right in matched
            ]
            identity = {
                "control_candidate_id": control_id,
                "input_text": input_text,
                "opencc_route_ids": route_ids,
                "output_text": composed,
                "rule_kind": "ORTHOGRAPHIC_WHOLE_INPUT",
                "safety_support_families": sorted(
                    safety_families_by_input[input_text]),
                "training_support_families": support,
                "training_support_family_count": support_count,
            }
            orthographic_rules.append({
                **identity,
                "candidate_rule_id": _sha256(identity),
            })

    source_rules = []
    base_inventories = base_candidate["inventories"]
    for rule in base_inventories["source_conditioned_rules"]:
        support = _families(
            rule.get("training_support_families"),
            label="source-conditioned support")
        if rule.get("training_support_family_count") != len(support):
            raise BroadQaExternalDataError(
                "v10 precision source-conditioned support 漂移")
        if len(support) != len(V8_TRAIN_FAMILIES):
            continue
        fields = {
            "base_candidate_rule_id": _sha_value(
                rule.get("candidate_rule_id"), label="base source rule"),
            "input_text": rule.get("input_text"),
            "official_source_text": rule.get("official_source_text"),
            "output_text": rule.get("output_text"),
            "rule_kind": "SOURCE_CONDITIONED_LEXICAL_ATOM",
            "training_support_families": support,
            "training_support_family_count": len(support),
        }
        if any(not isinstance(fields[name], str) or not fields[name]
               for name in (
                   "input_text", "official_source_text", "output_text")):
            raise BroadQaExternalDataError(
                "v10 precision source-conditioned semantic 漂移")
        if safety_outputs_by_source_input.get((
                str(fields["official_source_text"]),
                str(fields["input_text"]))) != {fields["output_text"]}:
            continue
        source_rules.append({
            **fields,
            "candidate_rule_id": _sha256(fields),
        })

    inventories = {
        "identity_veto_rules": sorted(
            identity_rules, key=lambda item: str(item["input_text"])),
        "orthographic_whole_input_rules": sorted(
            orthographic_rules, key=lambda item: str(item["input_text"])),
        "source_conditioned_rules": sorted(
            source_rules,
            key=lambda item: (
                str(item["official_source_text"]), str(item["input_text"]))),
    }
    if any(not inventories[name] for name in _INVENTORY_NAMES):
        raise BroadQaExternalDataError(
            "v10 precision 至少一个承重 inventory 为空")
    program = {
        "applicability_contract": {
            "arbitrary_atom_composition_allowed": 0,
            "identity_veto_precedes_all_rewrites": 1,
            "partial_commit_allowed": 0,
            "source_condition_requires_all_train_families": 1,
            "unknown_has_no_committed_output": 1,
            "whole_orthographic_requires_opencc_and_two_train_families": 1,
        },
        "artifact_kind": NORMALIZATION_RECOVERY_V10_PRECISION_CANDIDATE_KIND,
        "base_candidate_program_sha256": base_sha,
        "format_version": 1,
        "inventories": inventories,
        "mastery_claimed": 0,
        "observation_pack_manifest_sha256": observation_sha,
        "opencc_source_pack_manifest_sha256": opencc_sha,
        "production_enabled": 0,
        "rule_counts": {
            name: len(inventories[name]) for name in _INVENTORY_NAMES},
        "runtime_order": list(_RUNTIME_ORDER),
        "status": NORMALIZATION_RECOVERY_V10_PRECISION_CANDIDATE_STATUS,
        "safety_observation_count": len(safety_observations),
        "safety_observation_sha256": _sha256(safety_observations),
        "target_policy_scope": (
            NORMALIZATION_RECOVERY_V10_PRECISION_TARGET_SCOPE),
        "teacher_api_llm_call_count": 0,
        "training_protocol_manifest_sha256": protocol_sha,
    }
    return {**program, "candidate_program_sha256": _sha256(program)}


def _unique_index(
        records: list[dict[str, object]], key_builder,
        *, label: str,
        ) -> dict[object, dict[str, object]]:
    """构造唯一执行索引并拒绝任何 key 冲突。"""
    result = {}
    for record in records:
        key = key_builder(record)
        if key in result:
            raise BroadQaExternalDataError(
                f"v10 precision {label} key 冲突")
        result[key] = record
    return result


def _validate_candidate(
        candidate: dict[str, object],
        ) -> dict[str, list[dict[str, object]]]:
    """核验候选自绑定身份、禁用态、库存和执行合同。"""
    if not isinstance(candidate, dict):
        raise BroadQaExternalDataError("v10 precision candidate 非对象")
    program = {
        key: value for key, value in candidate.items()
        if key != "candidate_program_sha256"
    }
    inventories = candidate.get("inventories")
    counts = candidate.get("rule_counts")
    if (candidate.get("artifact_kind")
            != NORMALIZATION_RECOVERY_V10_PRECISION_CANDIDATE_KIND
            or candidate.get("status")
            != NORMALIZATION_RECOVERY_V10_PRECISION_CANDIDATE_STATUS
            or candidate.get("target_policy_scope")
            != NORMALIZATION_RECOVERY_V10_PRECISION_TARGET_SCOPE
            or candidate.get("production_enabled") != 0
            or candidate.get("mastery_claimed") != 0
            or candidate.get("candidate_program_sha256") != _sha256(program)
            or candidate.get("runtime_order") != list(_RUNTIME_ORDER)
            or not isinstance(inventories, dict)
            or set(inventories) != set(_INVENTORY_NAMES)
            or not isinstance(counts, dict)
            or set(counts) != set(_INVENTORY_NAMES)):
        raise BroadQaExternalDataError("v10 precision candidate program 漂移")
    for name in _INVENTORY_NAMES:
        records = inventories[name]
        if (not isinstance(records, list) or not records
                or counts[name] != len(records)
                or any(not isinstance(item, dict) for item in records)):
            raise BroadQaExternalDataError(
                "v10 precision candidate inventory 漂移")
    _build_index(inventories)
    return inventories


def _build_index(
        inventories: dict[str, list[dict[str, object]]],
        ) -> dict[str, dict[object, dict[str, object]]]:
    """从三个承重 inventory 构造运行时索引。"""
    return {
        "IDENTITY_VETO": _unique_index(
            inventories["identity_veto_rules"],
            lambda item: item["input_text"], label="identity"),
        "ORTHOGRAPHIC_WHOLE_INPUT": _unique_index(
            inventories["orthographic_whole_input_rules"],
            lambda item: item["input_text"], label="orthographic whole"),
        "SOURCE_CONDITIONED_LEXICAL_ATOM": _unique_index(
            inventories["source_conditioned_rules"],
            lambda item: (
                item["official_source_text"], item["input_text"]),
            label="source-conditioned"),
    }


def _query(value: dict[str, object]) -> dict[str, object]:
    """核验整串输入、official source 和结构 token 自洽。"""
    if not isinstance(value, dict):
        raise BroadQaExternalDataError("v10 precision query 非对象")
    input_text = value.get("input_text")
    source = value.get("official_source_text")
    tokens = value.get("structure_tokens")
    if (not isinstance(input_text, str) or not input_text
            or not isinstance(source, str) or not source
            or not isinstance(tokens, list)
            or any(not isinstance(item, str) or not item for item in tokens)
            or tuple(tokens) != localization_structure_tokens(input_text)):
        raise BroadQaExternalDataError("v10 precision query 结构漂移")
    return value


def _result(
        candidate: dict[str, object], query: dict[str, object], *,
        behavior: str, route_kind: str, output_text: str,
        matched_rule_ids: list[str], reason: str,
        ) -> dict[str, object]:
    """形成带结构、安全账和自绑定身份的执行结果。"""
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
        records: list[dict[str, object]], predicate,
        ) -> dict[str, object] | None:
    """线性查找唯一匹配，作为 indexed 路径的独立对照。"""
    matches = [item for item in records if predicate(item)]
    if len(matches) > 1:
        raise BroadQaExternalDataError(
            "v10 precision reference key 冲突")
    return matches[0] if matches else None


def _execute_one(
        candidate: dict[str, object], query_value: dict[str, object], *,
        inventories: dict[str, list[dict[str, object]]],
        index: dict[str, dict[object, dict[str, object]]] | None,
        ) -> dict[str, object]:
    """按 identity、source、整串正字的固定优先级执行一条 query。"""
    query = _query(query_value)
    source = str(query["official_source_text"])
    input_text = str(query["input_text"])
    if index is None:
        identity = _reference_match(
            inventories["identity_veto_rules"],
            lambda item: item["input_text"] == input_text)
        lexical = _reference_match(
            inventories["source_conditioned_rules"],
            lambda item: item["official_source_text"] == source
            and item["input_text"] == input_text)
        orthographic = _reference_match(
            inventories["orthographic_whole_input_rules"],
            lambda item: item["input_text"] == input_text)
    else:
        identity = index["IDENTITY_VETO"].get(input_text)
        lexical = index["SOURCE_CONDITIONED_LEXICAL_ATOM"].get(
            (source, input_text))
        orthographic = index["ORTHOGRAPHIC_WHOLE_INPUT"].get(input_text)

    selected = None
    route = "UNKNOWN"
    output_text = ""
    reason = "NO_PRECISION_AUTHORIZATION"
    if identity is not None:
        selected = identity
        route = "IDENTITY_VETO"
        output_text = input_text
        reason = "TRAIN_IDENTITY_VETO_COMMITTED"
    elif lexical is not None:
        selected = lexical
        route = "SOURCE_CONDITIONED_LEXICAL_ATOM"
        output_text = str(lexical["output_text"])
        reason = "THREE_FAMILY_SOURCE_CONDITION_COMMITTED"
    elif orthographic is not None:
        selected = orthographic
        route = "ORTHOGRAPHIC_WHOLE_INPUT"
        output_text = str(orthographic["output_text"])
        reason = "OPENCC_MULTI_FAMILY_WHOLE_INPUT_COMMITTED"
    if selected is None:
        return _result(
            candidate, query, behavior="UNKNOWN", route_kind=route,
            output_text="", matched_rule_ids=[], reason=reason)
    rule_ids = [str(selected["candidate_rule_id"])]
    if list(localization_structure_tokens(output_text)) != query["structure_tokens"]:
        return _result(
            candidate, query, behavior="UNKNOWN", route_kind=route,
            output_text="", matched_rule_ids=rule_ids,
            reason="STRUCTURE_PRESERVATION_HARD_GATE_REJECTED")
    return _result(
        candidate, query, behavior="EXACT", route_kind=route,
        output_text=output_text, matched_rule_ids=rule_ids, reason=reason)


def execute_normalization_recovery_v10_precision_candidate_batch(
        candidate: dict[str, object],
        queries: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, object], ...]:
    """用字典索引批量执行 precision-first 候选。"""
    if not isinstance(queries, tuple) or not queries:
        raise BroadQaExternalDataError("v10 precision batch 为空")
    inventories = _validate_candidate(candidate)
    index = _build_index(inventories)
    return tuple(_execute_one(
        candidate, item, inventories=inventories, index=index)
        for item in queries)


def reference_normalization_recovery_v10_precision_candidate_batch(
        candidate: dict[str, object],
        queries: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, object], ...]:
    """用逐 inventory 线性扫描执行独立 reference 对照。"""
    if not isinstance(queries, tuple) or not queries:
        raise BroadQaExternalDataError("v10 precision reference batch 为空")
    inventories = _validate_candidate(candidate)
    return tuple(_execute_one(
        candidate, item, inventories=inventories, index=None)
        for item in queries)


def derive_normalization_recovery_v10_precision_preflight(
        candidate: dict[str, object],
        ) -> dict[str, object]:
    """覆盖全部已授权规则和禁止局部原子组合的标签盲预检。"""
    inventories = _validate_candidate(candidate)
    cases = []
    for item in inventories["identity_veto_rules"]:
        cases.append({
            "expected_output": item["input_text"],
            "input_text": item["input_text"],
            "official_source_text": "SYNTHETIC_IDENTITY_SOURCE",
            "route_kind": "IDENTITY_VETO",
        })
    for item in inventories["source_conditioned_rules"]:
        cases.append({
            "expected_output": item["output_text"],
            "input_text": item["input_text"],
            "official_source_text": item["official_source_text"],
            "route_kind": "SOURCE_CONDITIONED_LEXICAL_ATOM",
        })
    for item in inventories["orthographic_whole_input_rules"]:
        cases.append({
            "expected_output": item["output_text"],
            "input_text": item["input_text"],
            "official_source_text": "SYNTHETIC_ORTHOGRAPHIC_SOURCE",
            "route_kind": "ORTHOGRAPHIC_WHOLE_INPUT",
        })
    forbidden = "未授权局部原子组合"
    occupied = {
        str(item["input_text"])
        for name in _INVENTORY_NAMES for item in inventories[name]
    }
    while forbidden in occupied:
        forbidden += "外"
    cases.append({
        "expected_output": "",
        "input_text": forbidden,
        "official_source_text": "SYNTHETIC_UNKNOWN_SOURCE",
        "route_kind": "UNKNOWN",
    })
    queries = tuple({
        "input_text": str(item["input_text"]),
        "official_source_text": str(item["official_source_text"]),
        "structure_tokens": list(localization_structure_tokens(
            str(item["input_text"]))),
    } for item in cases)
    indexed = execute_normalization_recovery_v10_precision_candidate_batch(
        candidate, queries)
    reference = (
        reference_normalization_recovery_v10_precision_candidate_batch(
            candidate, queries))
    failures = sum(
        left != right
        or left["output_text"] != case["expected_output"]
        or left["route_kind"] != case["route_kind"]
        for case, left, right in zip(cases, indexed, reference))
    result = {
        "arbitrary_atom_composition_case_count": 1,
        "case_count": len(cases),
        "exception_count": 0,
        "failure_count": failures,
        "format_version": 1,
        "indexed_reference_mismatch_count": sum(
            left != right for left, right in zip(indexed, reference)),
        "production_enabled": 0,
        "rule_counts": candidate["rule_counts"],
        "unknown_case_count": sum(
            item["behavior"] == "UNKNOWN" for item in indexed),
    }
    return {**result, "preflight_sha256": _sha256(result)}


def derive_normalization_recovery_v10_precision_training_audit(
        candidate: dict[str, object],
        cases: tuple[dict[str, object], ...],
        ) -> dict[str, object]:
    """在调用方提供的 TRAIN cases 上汇总双解释器和零错误门。"""
    if not isinstance(cases, tuple) or not cases:
        raise BroadQaExternalDataError("v10 precision TRAIN audit 为空")
    queries = []
    expected = []
    families = []
    for case in cases:
        if not isinstance(case, dict):
            raise BroadQaExternalDataError(
                "v10 precision TRAIN case 非对象")
        query = _query(case)
        output = case.get("expected_output")
        family = case.get("source_family")
        if (not isinstance(output, str) or not output
                or not isinstance(family, str) or not family):
            raise BroadQaExternalDataError(
                "v10 precision TRAIN case label 漂移")
        queries.append({
            "input_text": query["input_text"],
            "official_source_text": query["official_source_text"],
            "structure_tokens": query["structure_tokens"],
        })
        expected.append(output)
        families.append(family)
    query_tuple = tuple(queries)
    indexed = execute_normalization_recovery_v10_precision_candidate_batch(
        candidate, query_tuple)
    reference = (
        reference_normalization_recovery_v10_precision_candidate_batch(
            candidate, query_tuple))
    outcomes = Counter()
    route_outcomes = Counter()
    family_outcomes = Counter()
    changed_exact = 0
    for wanted, family, left, right in zip(
            expected, families, indexed, reference):
        if left != right:
            outcome = "MISMATCH"
        elif left["behavior"] == "UNKNOWN":
            outcome = "UNKNOWN"
        elif left["output_text"] == wanted:
            outcome = "EXACT"
        else:
            outcome = "WRONG"
        outcomes[outcome] += 1
        route_outcomes[(str(left["route_kind"]), outcome)] += 1
        family_outcomes[(family, outcome)] += 1
        changed_exact += int(
            outcome == "EXACT" and left["output_text"] != left["input_text"])
    wrong = outcomes["WRONG"]
    mismatch = outcomes["MISMATCH"]
    result = {
        "candidate_program_sha256": candidate["candidate_program_sha256"],
        "case_count": len(cases),
        "changed_exact_count": changed_exact,
        "facility_outcome": "PASS" if mismatch == 0 else "FAIL",
        "family_outcomes": {
            f"{family}:{outcome}": count
            for (family, outcome), count in sorted(family_outcomes.items())
        },
        "format_version": 1,
        "indexed_reference_mismatch_count": mismatch,
        "outcomes": {
            name: outcomes[name]
            for name in ("EXACT", "UNKNOWN", "WRONG", "MISMATCH")
        },
        "production_enabled": 0,
        "route_outcomes": {
            f"{route}:{outcome}": count
            for (route, outcome), count in sorted(route_outcomes.items())
        },
        "training_outcome": (
            "PASS_ZERO_WRONG_NONZERO_CHANGED_EXACT"
            if wrong == 0 and mismatch == 0 and changed_exact > 0
            else "FAIL"),
    }
    return {**result, "training_audit_sha256": _sha256(result)}


def compile_normalization_recovery_v10_precision_candidate_v2(
        **arguments: object,
        ) -> dict[str, object]:
    """从 v1 inventory 编译只允许 exact-source 提交的 v2 候选。"""
    predecessor = compile_normalization_recovery_v10_precision_candidate(
        **arguments)
    program = {
        key: value for key, value in predecessor.items()
        if key not in {
            "applicability_contract", "artifact_kind",
            "candidate_program_sha256", "runtime_order", "status",
        }
    }
    program.update({
        "applicability_contract": {
            "arbitrary_atom_composition_allowed": 0,
            "identity_veto_commit_allowed": 0,
            "identity_veto_precedes_all_rewrites": 1,
            "orthographic_hypothesis_commit_allowed": 0,
            "partial_commit_allowed": 0,
            "source_condition_requires_all_train_families": 1,
            "unknown_has_no_committed_output": 1,
        },
        "artifact_kind": (
            NORMALIZATION_RECOVERY_V10_PRECISION_CANDIDATE_V2_KIND),
        "predecessor_candidate_program_sha256": predecessor[
            "candidate_program_sha256"],
        "runtime_order": list(_V2_RUNTIME_ORDER),
        "status": NORMALIZATION_RECOVERY_V10_PRECISION_CANDIDATE_V2_STATUS,
    })
    return {**program, "candidate_program_sha256": _sha256(program)}


def _validate_candidate_v2(
        candidate: dict[str, object],
        ) -> dict[str, list[dict[str, object]]]:
    """核验 v2 自绑定身份、前驱、库存与非提交合同。"""
    if not isinstance(candidate, dict):
        raise BroadQaExternalDataError("v10 precision v2 candidate 非对象")
    program = {
        key: value for key, value in candidate.items()
        if key != "candidate_program_sha256"
    }
    inventories = candidate.get("inventories")
    counts = candidate.get("rule_counts")
    contract = candidate.get("applicability_contract")
    if (candidate.get("artifact_kind")
            != NORMALIZATION_RECOVERY_V10_PRECISION_CANDIDATE_V2_KIND
            or candidate.get("status")
            != NORMALIZATION_RECOVERY_V10_PRECISION_CANDIDATE_V2_STATUS
            or candidate.get("candidate_program_sha256") != _sha256(program)
            or candidate.get("runtime_order") != list(_V2_RUNTIME_ORDER)
            or candidate.get("production_enabled") != 0
            or candidate.get("mastery_claimed") != 0
            or _sha_value(
                candidate.get("predecessor_candidate_program_sha256"),
                label="v2 predecessor candidate")
            != candidate.get("predecessor_candidate_program_sha256")
            or not isinstance(contract, dict)
            or contract.get("identity_veto_commit_allowed") != 0
            or contract.get("orthographic_hypothesis_commit_allowed") != 0
            or not isinstance(inventories, dict)
            or set(inventories) != set(_INVENTORY_NAMES)
            or not isinstance(counts, dict)
            or set(counts) != set(_INVENTORY_NAMES)):
        raise BroadQaExternalDataError(
            "v10 precision v2 candidate program 漂移")
    for name in _INVENTORY_NAMES:
        records = inventories[name]
        if (not isinstance(records, list) or not records
                or counts[name] != len(records)
                or any(not isinstance(item, dict) for item in records)):
            raise BroadQaExternalDataError(
                "v10 precision v2 inventory 漂移")
    _build_index(inventories)
    return inventories


def _ordered_reference(
        records: list[dict[str, object]], key_builder, *, label: str,
        ) -> tuple[tuple[object, ...], tuple[dict[str, object], ...]]:
    """构造独立于 dict 索引的排序键 reference，并拒绝重复键。"""
    pairs = sorted(
        ((key_builder(record), record) for record in records),
        key=lambda item: item[0])
    keys = tuple(item[0] for item in pairs)
    if any(keys[index - 1] == keys[index] for index in range(1, len(keys))):
        raise BroadQaExternalDataError(
            f"v10 precision v2 {label} reference key 冲突")
    return keys, tuple(item[1] for item in pairs)


def _build_reference_index(
        inventories: dict[str, list[dict[str, object]]],
        ) -> dict[
            str,
            tuple[tuple[object, ...], tuple[dict[str, object], ...]],
        ]:
    """为 v2 reference 构造三个有序键表，避免全分母线性扫描。"""
    return {
        "IDENTITY_VETO": _ordered_reference(
            inventories["identity_veto_rules"],
            lambda item: item["input_text"], label="identity"),
        "ORTHOGRAPHIC_WHOLE_INPUT": _ordered_reference(
            inventories["orthographic_whole_input_rules"],
            lambda item: item["input_text"], label="orthographic"),
        "SOURCE_CONDITIONED_LEXICAL_ATOM": _ordered_reference(
            inventories["source_conditioned_rules"],
            lambda item: (
                item["official_source_text"], item["input_text"]),
            label="source-conditioned"),
    }


def _reference_lookup(
        reference: tuple[
            tuple[object, ...], tuple[dict[str, object], ...]],
        key: object,
        ) -> dict[str, object] | None:
    """在排序键 reference 中二分查找唯一规则。"""
    keys, records = reference
    index = bisect_left(keys, key)
    return records[index] if index < len(keys) and keys[index] == key else None


def _execute_one_v2(
        candidate: dict[str, object], query_value: dict[str, object], *,
        inventories: dict[str, list[dict[str, object]]],
        index: dict[str, dict[object, dict[str, object]]] | None,
        reference_index: dict[
            str,
            tuple[tuple[object, ...], tuple[dict[str, object], ...]],
        ] | None,
        ) -> dict[str, object]:
    """执行 source-only commit，并把 identity/orthographic 保留为非提交 trace。"""
    query = _query(query_value)
    source = str(query["official_source_text"])
    input_text = str(query["input_text"])
    if index is not None and reference_index is None:
        identity = index["IDENTITY_VETO"].get(input_text)
        lexical = index["SOURCE_CONDITIONED_LEXICAL_ATOM"].get(
            (source, input_text))
        orthographic = index["ORTHOGRAPHIC_WHOLE_INPUT"].get(input_text)
    elif reference_index is not None and index is None:
        identity = _reference_lookup(
            reference_index["IDENTITY_VETO"], input_text)
        lexical = _reference_lookup(
            reference_index["SOURCE_CONDITIONED_LEXICAL_ATOM"],
            (source, input_text))
        orthographic = _reference_lookup(
            reference_index["ORTHOGRAPHIC_WHOLE_INPUT"], input_text)
    else:
        raise BroadQaExternalDataError(
            "v10 precision v2 executor index 模式非法")
    if identity is not None:
        return _result(
            candidate, query, behavior="UNKNOWN",
            route_kind="IDENTITY_VETO_NONCOMMITTING", output_text="",
            matched_rule_ids=[str(identity["candidate_rule_id"])],
            reason="TRAIN_IDENTITY_VETO_BLOCKED_REWRITE")
    if lexical is not None:
        output_text = str(lexical["output_text"])
        rule_ids = [str(lexical["candidate_rule_id"])]
        if list(localization_structure_tokens(output_text)) != query[
                "structure_tokens"]:
            return _result(
                candidate, query, behavior="UNKNOWN",
                route_kind="SOURCE_CONDITIONED_LEXICAL_ATOM",
                output_text="", matched_rule_ids=rule_ids,
                reason="STRUCTURE_PRESERVATION_HARD_GATE_REJECTED")
        return _result(
            candidate, query, behavior="EXACT",
            route_kind="SOURCE_CONDITIONED_LEXICAL_ATOM",
            output_text=output_text, matched_rule_ids=rule_ids,
            reason="THREE_FAMILY_SOURCE_CONDITION_COMMITTED")
    if orthographic is not None:
        return _result(
            candidate, query, behavior="UNKNOWN",
            route_kind="ORTHOGRAPHIC_WHOLE_INPUT_HYPOTHESIS",
            output_text="",
            matched_rule_ids=[str(orthographic["candidate_rule_id"])],
            reason="ORTHOGRAPHIC_HYPOTHESIS_REQUIRES_ADDITIONAL_AUTHORIZATION")
    return _result(
        candidate, query, behavior="UNKNOWN", route_kind="UNKNOWN",
        output_text="", matched_rule_ids=[],
        reason="NO_PRECISION_AUTHORIZATION")


def execute_normalization_recovery_v10_precision_candidate_v2_batch(
        candidate: dict[str, object],
        queries: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, object], ...]:
    """用索引批量执行 v2 source-only commit 候选。"""
    if not isinstance(queries, tuple) or not queries:
        raise BroadQaExternalDataError("v10 precision v2 batch 为空")
    inventories = _validate_candidate_v2(candidate)
    index = _build_index(inventories)
    return tuple(_execute_one_v2(
        candidate, item, inventories=inventories, index=index,
        reference_index=None)
        for item in queries)


def reference_normalization_recovery_v10_precision_candidate_v2_batch(
        candidate: dict[str, object],
        queries: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, object], ...]:
    """用排序键二分查找批量执行 v2 独立 reference。"""
    if not isinstance(queries, tuple) or not queries:
        raise BroadQaExternalDataError("v10 precision v2 reference 为空")
    inventories = _validate_candidate_v2(candidate)
    reference_index = _build_reference_index(inventories)
    return tuple(_execute_one_v2(
        candidate, item, inventories=inventories, index=None,
        reference_index=reference_index)
        for item in queries)


def profile_normalization_recovery_v10_precision_candidate_v2_batch(
        candidate: dict[str, object],
        queries: tuple[dict[str, object], ...], *,
        indexed: bool,
        clock_ns,
        ) -> tuple[tuple[dict[str, object], ...], tuple[int, ...]]:
    """保持一次候选校验，并记录每条 v2 query 的整数纳秒耗时。"""
    if (not isinstance(queries, tuple) or not queries
            or type(indexed) is not bool or not callable(clock_ns)):
        raise BroadQaExternalDataError(
            "v10 precision v2 profile batch 非法")
    inventories = _validate_candidate_v2(candidate)
    index = _build_index(inventories) if indexed else None
    reference_index = (
        None if indexed else _build_reference_index(inventories))
    results = []
    durations = []
    for query in queries:
        started = clock_ns()
        results.append(_execute_one_v2(
            candidate, query, inventories=inventories, index=index,
            reference_index=reference_index))
        elapsed = clock_ns() - started
        if type(elapsed) is not int or elapsed < 0:
            raise BroadQaExternalDataError(
                "v10 precision v2 profile clock 非法")
        durations.append(elapsed)
    return tuple(results), tuple(durations)


def derive_normalization_recovery_v10_precision_v2_preflight(
        candidate: dict[str, object],
        ) -> dict[str, object]:
    """证明 source 提交与 identity/orthographic 非提交语义。"""
    inventories = _validate_candidate_v2(candidate)
    cases = []
    for item in inventories["identity_veto_rules"]:
        cases.append({
            "behavior": "UNKNOWN",
            "input_text": item["input_text"],
            "official_source_text": "SYNTHETIC_IDENTITY_SOURCE",
            "output_text": "",
            "route_kind": "IDENTITY_VETO_NONCOMMITTING",
        })
    for item in inventories["source_conditioned_rules"]:
        cases.append({
            "behavior": "EXACT",
            "input_text": item["input_text"],
            "official_source_text": item["official_source_text"],
            "output_text": item["output_text"],
            "route_kind": "SOURCE_CONDITIONED_LEXICAL_ATOM",
        })
    for item in inventories["orthographic_whole_input_rules"]:
        cases.append({
            "behavior": "UNKNOWN",
            "input_text": item["input_text"],
            "official_source_text": "SYNTHETIC_ORTHOGRAPHIC_SOURCE",
            "output_text": "",
            "route_kind": "ORTHOGRAPHIC_WHOLE_INPUT_HYPOTHESIS",
        })
    unknown = "未授权v2候选"
    occupied = {
        str(item["input_text"])
        for name in _INVENTORY_NAMES for item in inventories[name]
    }
    while unknown in occupied:
        unknown += "外"
    cases.append({
        "behavior": "UNKNOWN",
        "input_text": unknown,
        "official_source_text": "SYNTHETIC_UNKNOWN_SOURCE",
        "output_text": "",
        "route_kind": "UNKNOWN",
    })
    queries = tuple({
        "input_text": str(item["input_text"]),
        "official_source_text": str(item["official_source_text"]),
        "structure_tokens": list(localization_structure_tokens(
            str(item["input_text"]))),
    } for item in cases)
    indexed = execute_normalization_recovery_v10_precision_candidate_v2_batch(
        candidate, queries)
    reference = (
        reference_normalization_recovery_v10_precision_candidate_v2_batch(
            candidate, queries))
    failures = sum(
        left != right
        or left["behavior"] != case["behavior"]
        or left["output_text"] != case["output_text"]
        or left["route_kind"] != case["route_kind"]
        for case, left, right in zip(cases, indexed, reference))
    result = {
        "case_count": len(cases),
        "exception_count": 0,
        "failure_count": failures,
        "format_version": 2,
        "identity_noncommitting_case_count": len(
            inventories["identity_veto_rules"]),
        "indexed_reference_mismatch_count": sum(
            left != right for left, right in zip(indexed, reference)),
        "orthographic_noncommitting_case_count": len(
            inventories["orthographic_whole_input_rules"]),
        "production_enabled": 0,
        "rule_counts": candidate["rule_counts"],
        "source_commit_case_count": len(
            inventories["source_conditioned_rules"]),
    }
    return {**result, "preflight_sha256": _sha256(result)}


def derive_normalization_recovery_v10_precision_v2_training_audit(
        candidate: dict[str, object],
        cases: tuple[dict[str, object], ...],
        ) -> dict[str, object]:
    """汇总 v2 全 TRAIN 的提交准确率、非提交路由和双解释器。"""
    if not isinstance(cases, tuple) or not cases:
        raise BroadQaExternalDataError("v10 precision v2 TRAIN audit 为空")
    queries = []
    expected = []
    families = []
    for case in cases:
        query = _query(case) if isinstance(case, dict) else None
        output = case.get("expected_output") if isinstance(case, dict) else None
        family = case.get("source_family") if isinstance(case, dict) else None
        if (query is None or not isinstance(output, str) or not output
                or not isinstance(family, str) or not family):
            raise BroadQaExternalDataError(
                "v10 precision v2 TRAIN case 漂移")
        queries.append({
            "input_text": query["input_text"],
            "official_source_text": query["official_source_text"],
            "structure_tokens": query["structure_tokens"],
        })
        expected.append(output)
        families.append(family)
    query_tuple = tuple(queries)
    indexed = execute_normalization_recovery_v10_precision_candidate_v2_batch(
        candidate, query_tuple)
    reference = (
        reference_normalization_recovery_v10_precision_candidate_v2_batch(
            candidate, query_tuple))
    outcomes = Counter()
    route_outcomes = Counter()
    family_outcomes = Counter()
    changed_exact = 0
    for wanted, family, left, right in zip(
            expected, families, indexed, reference):
        if left != right:
            outcome = "MISMATCH"
        elif left["behavior"] == "UNKNOWN":
            outcome = "UNKNOWN"
        elif left["output_text"] == wanted:
            outcome = "EXACT"
        else:
            outcome = "WRONG"
        outcomes[outcome] += 1
        route_outcomes[(str(left["route_kind"]), outcome)] += 1
        family_outcomes[(family, outcome)] += 1
        changed_exact += int(
            outcome == "EXACT" and left["output_text"] != left["input_text"])
    result = {
        "candidate_program_sha256": candidate["candidate_program_sha256"],
        "case_count": len(cases),
        "changed_exact_count": changed_exact,
        "facility_outcome": (
            "PASS" if outcomes["MISMATCH"] == 0 else "FAIL"),
        "family_outcomes": {
            f"{family}:{outcome}": count
            for (family, outcome), count in sorted(family_outcomes.items())
        },
        "format_version": 2,
        "indexed_reference_mismatch_count": outcomes["MISMATCH"],
        "outcomes": {
            name: outcomes[name]
            for name in ("EXACT", "UNKNOWN", "WRONG", "MISMATCH")
        },
        "production_enabled": 0,
        "route_outcomes": {
            f"{route}:{outcome}": count
            for (route, outcome), count in sorted(route_outcomes.items())
        },
        "training_outcome": (
            "PASS_ZERO_WRONG_NONZERO_CHANGED_EXACT"
            if outcomes["WRONG"] == 0 and outcomes["MISMATCH"] == 0
            and changed_exact > 0 else "FAIL"),
    }
    return {**result, "training_audit_sha256": _sha256(result)}


def derive_normalization_recovery_v10_precision_source_loso_audit(
        *, base_candidate: dict[str, object],
        safety_observations: tuple[dict[str, object], ...],
        ) -> dict[str, object]:
    """每次删除一家来源，用其余两家重建 exact-source 规则并评分。"""
    build_normalization_recovery_v8_candidate_index(base_candidate)
    _safety_indexes(safety_observations)
    base_rules = base_candidate["inventories"]["source_conditioned_rules"]
    directions = []
    total = Counter()
    for held_out in V8_TRAIN_FAMILIES:
        train_outputs: dict[tuple[str, str], set[str]] = {}
        train_families: dict[tuple[str, str], set[str]] = {}
        held_records = []
        for item in safety_observations:
            key = (str(item["official_source_text"]), str(item["input_text"]))
            if item["source_family"] == held_out:
                held_records.append(item)
            else:
                train_outputs.setdefault(key, set()).add(
                    str(item["output_text"]))
                train_families.setdefault(key, set()).add(
                    str(item["source_family"]))
        rules = {}
        for rule in base_rules:
            support = set(_families(
                rule.get("training_support_families"),
                label="LOSO source support")) - {held_out}
            key = (
                str(rule["official_source_text"]), str(rule["input_text"]))
            if (len(support) >= 2
                    and len(train_families.get(key, set())) >= 2
                    and train_outputs.get(key) == {rule["output_text"]}):
                rules[key] = str(rule["output_text"])
        outcomes = Counter()
        for item in held_records:
            key = (str(item["official_source_text"]), str(item["input_text"]))
            output = rules.get(key)
            outcome = (
                "UNKNOWN" if output is None
                else "EXACT" if output == item["output_text"] else "WRONG")
            outcomes[outcome] += 1
            total[outcome] += 1
        directions.append({
            "held_out_family": held_out,
            "held_out_record_count": len(held_records),
            "outcomes": {
                name: outcomes[name]
                for name in ("EXACT", "UNKNOWN", "WRONG")},
            "source_rule_count": len(rules),
        })
    result = {
        "direction_count": len(directions),
        "directions": directions,
        "format_version": 1,
        "held_out_output_used_for_rule_construction": 0,
        "outcomes": {
            name: total[name] for name in ("EXACT", "UNKNOWN", "WRONG")},
        "production_enabled": 0,
        "status": (
            "PASS_ZERO_WRONG_NONZERO_EXACT"
            if total["WRONG"] == 0 and total["EXACT"] > 0 else "FAIL"),
    }
    return {**result, "loso_audit_sha256": _sha256(result)}


__all__ = [
    "NORMALIZATION_RECOVERY_V10_PRECISION_CANDIDATE_KIND",
    "NORMALIZATION_RECOVERY_V10_PRECISION_CANDIDATE_STATUS",
    "NORMALIZATION_RECOVERY_V10_PRECISION_TARGET_SCOPE",
    "NORMALIZATION_RECOVERY_V10_PRECISION_CANDIDATE_V2_KIND",
    "NORMALIZATION_RECOVERY_V10_PRECISION_CANDIDATE_V2_STATUS",
    "compile_normalization_recovery_v10_precision_candidate",
    "compile_normalization_recovery_v10_precision_candidate_v2",
    "derive_normalization_recovery_v10_precision_preflight",
    "derive_normalization_recovery_v10_precision_training_audit",
    "derive_normalization_recovery_v10_precision_source_loso_audit",
    "derive_normalization_recovery_v10_precision_v2_preflight",
    "derive_normalization_recovery_v10_precision_v2_training_audit",
    "execute_normalization_recovery_v10_precision_candidate_batch",
    "reference_normalization_recovery_v10_precision_candidate_batch",
    "execute_normalization_recovery_v10_precision_candidate_v2_batch",
    "profile_normalization_recovery_v10_precision_candidate_v2_batch",
    "reference_normalization_recovery_v10_precision_candidate_v2_batch",
]
