"""执行 recovery-v7 context-scoped local TRAIN-only segment transaction。

本模块只消费每个 LOSO 方向重新学习得到的 v5 内存材料。局部规则必须同时通过
SUPPORT 正上下文、identity/conflict veto 与既有 defeater，且只能在 v7 structure
segment 内形成 proposal。所有 TRAIN-derived obligations 闭合后才原子提交整句；
缺失、歧义、重叠或结构 token 漂移一律返回原输入。
"""
from __future__ import annotations

from collections import Counter
import hashlib

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_localization_structure import (
    localization_structure_layout_for_tokens,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_successor_simulation_records import (
    _context_key,
    _defeater_hits,
    _obligation_spans,
    _program_context,
    _rule_order,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_training_records import (
    V5_SOURCE_FAMILIES,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
)


CONTEXT_LOCAL_RULE_REPRESENTATION_KIND = (
    "NORMALIZATION_RECOVERY_V7_CONTEXT_LOCAL_RULE_REPRESENTATION_V1")
CONTEXT_LOCAL_TARGET_SCOPE = (
    "CROSS_PRODUCT_ZH_CN_VARIABLE_STRUCTURE_CONTEXT_LOCAL_V1")


def _sha256(payload: bytes) -> str:
    """返回规范记录或 TRAIN surface 的 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def _text_sha256(value: str) -> str:
    """返回 UTF-8 文本 SHA。"""
    if not isinstance(value, str):
        raise BroadQaExternalDataError("v7 context local text 非字符串")
    return _sha256(value.encode("utf-8"))


def _record_id(identity: dict[str, object]) -> str:
    """从完整语义 identity 形成稳定 id。"""
    return _sha256(canonical_json_bytes(identity))


def _target_local_rules(
        context: dict[str, object],
        ) -> tuple[dict[str, object], ...]:
    """从 indexed target buckets 恢复去重后的 local rule 完全序。"""
    buckets = context.get("target_buckets")
    if not isinstance(buckets, dict):
        raise BroadQaExternalDataError("v7 context local target buckets 漂移")
    values = {}
    for rules in buckets.values():
        if not isinstance(rules, tuple):
            raise BroadQaExternalDataError(
                "v7 context local target bucket 非 tuple")
        for rule in rules:
            rule_id = rule.get("rule_id") if isinstance(rule, dict) else None
            if not isinstance(rule_id, str) or len(rule_id) != 64:
                raise BroadQaExternalDataError(
                    "v7 context local rule identity 漂移")
            previous = values.setdefault(rule_id, rule)
            if previous != rule:
                raise BroadQaExternalDataError(
                    "v7 context local rule identity 冲突")
    return tuple(sorted(values.values(), key=_rule_order))


def derive_context_local_rule_representations(
        *,
        context: dict[str, object],
        held_out_source_family: str,
        ) -> tuple[tuple[dict[str, object], ...], dict[str, object]]:
    """冻结一个 LOSO 子集重新学得的 local applicability 表示。"""
    if held_out_source_family not in V5_SOURCE_FAMILIES:
        raise BroadQaExternalDataError("v7 context local held-out family 漂移")
    positive_contexts = context.get("positive_contexts")
    defeaters_by_rule = context.get("defeaters_by_rule")
    identity_inputs = context.get("identity_inputs")
    conflict_inputs = context.get("conflict_inputs")
    if (not isinstance(positive_contexts, dict)
            or not isinstance(defeaters_by_rule, dict)
            or not isinstance(identity_inputs, frozenset)
            or not isinstance(conflict_inputs, frozenset)):
        raise BroadQaExternalDataError(
            "v7 context local applicability context 漂移")
    values = []
    counters = Counter()
    for rule in _target_local_rules(context):
        if (rule.get("fragment_kind") == "WHOLE_INPUT"
                or rule.get("candidate_scope_kind")
                != "TARGET_CROSS_FAMILY"):
            continue
        rule_id = str(rule["rule_id"])
        input_text = rule.get("input_text")
        output_text = rule.get("output_text")
        source_families = rule.get("source_families")
        contexts = positive_contexts.get(rule_id)
        defeaters = defeaters_by_rule.get(rule_id, ())
        if (not isinstance(input_text, str) or not input_text
                or not isinstance(output_text, str)
                or output_text == input_text
                or not isinstance(source_families, list)
                or any(item not in V5_SOURCE_FAMILIES
                       for item in source_families)
                or not isinstance(contexts, frozenset)
                or any(not isinstance(item, tuple) or len(item) != 4
                       for item in contexts)
                or not isinstance(defeaters, tuple)):
            raise BroadQaExternalDataError(
                "v7 context local learned rule schema 漂移")
        nonempty_contexts = tuple(sorted(
            item for item in contexts if str(item[1]) or str(item[3])))
        reasons = []
        if len(set(source_families)) < 2:
            reasons.append("CROSS_FAMILY_SUPPORT_INSUFFICIENT")
        if not nonempty_contexts:
            reasons.append("NO_POSITIVE_SURFACE_CONTEXT")
        if not defeaters:
            reasons.append("NO_NEGATIVE_DEFEATER")
        identity = {
            "held_out_source_family": held_out_source_family,
            "predecessor_rule_id": rule_id,
            "target_policy_scope": CONTEXT_LOCAL_TARGET_SCOPE,
        }
        values.append({
            **identity,
            "atomic_whole_commit_required": 1,
            "conflict_veto_required": int(input_text in conflict_inputs),
            "defer_reasons": reasons,
            "defeater_count": len(defeaters),
            "execution_allowed": 0,
            "format_version": 1,
            "identity_veto_required": int(input_text in identity_inputs),
            "input_length": len(input_text),
            "input_sha256": _text_sha256(input_text),
            "nonempty_positive_context_count": len(nonempty_contexts),
            "output_length": len(output_text),
            "output_sha256": _text_sha256(output_text),
            "positive_context_count": len(contexts),
            "record_kind": CONTEXT_LOCAL_RULE_REPRESENTATION_KIND,
            "representation_id": _record_id(identity),
            "source_family_count": len(set(source_families)),
            "status": "REPRESENTATION_ELIGIBLE" if not reasons else "DEFERRED",
        })
        counters["rule_count"] += 1
        counters["representation_eligible_count"] += int(not reasons)
        counters["no_positive_surface_context_count"] += int(
            "NO_POSITIVE_SURFACE_CONTEXT" in reasons)
        counters["no_negative_defeater_count"] += int(
            "NO_NEGATIVE_DEFEATER" in reasons)
        counters["cross_family_support_insufficient_count"] += int(
            "CROSS_FAMILY_SUPPORT_INSUFFICIENT" in reasons)
        counters["identity_veto_rule_count"] += int(
            input_text in identity_inputs)
        counters["conflict_veto_rule_count"] += int(
            input_text in conflict_inputs)
    if not values:
        raise BroadQaExternalDataError(
            "v7 context local learned representation 为空")
    values.sort(key=lambda item: str(item["representation_id"]))
    return tuple(values), {
        "conflict_veto_rule_count": counters["conflict_veto_rule_count"],
        "cross_family_support_insufficient_count": counters[
            "cross_family_support_insufficient_count"],
        "identity_veto_rule_count": counters["identity_veto_rule_count"],
        "no_negative_defeater_count": counters[
            "no_negative_defeater_count"],
        "no_positive_surface_context_count": counters[
            "no_positive_surface_context_count"],
        "representation_eligible_count": counters[
            "representation_eligible_count"],
        "rule_count": counters["rule_count"],
    }


def prepare_context_local_execution(
        *,
        program: dict[str, object],
        outputs: dict[str, tuple[dict[str, object], ...]],
        training_groups: tuple[dict[str, object], ...],
        held_out_source_family: str,
        ) -> tuple[dict[str, object], tuple[dict[str, object], ...], dict[str, object]]:
    """建立一个 LOSO 方向的只读 indexed/reference context。"""
    context = _program_context(
        program=program,
        outputs=outputs,
        training_groups=training_groups,
        source_family=held_out_source_family,
    )
    representations, summary = derive_context_local_rule_representations(
        context=context,
        held_out_source_family=held_out_source_family,
    )
    eligible_ids = frozenset(
        str(item["predecessor_rule_id"])
        for item in representations
        if item["status"] == "REPRESENTATION_ELIGIBLE")
    reference_rules = tuple(
        rule for rule in _target_local_rules(context)
        if str(rule["rule_id"]) in eligible_ids)
    return {
        **context,
        "eligible_rule_ids": eligible_ids,
        "reference_local_rules": reference_rules,
    }, representations, summary


def _segment_spans(layout: dict[str, object]) -> tuple[tuple[int, int], ...]:
    """把 layout segments 投影回完整 input 的绝对 span。"""
    segments = layout.get("segments")
    raw_tokens = layout.get("raw_tokens")
    if (not isinstance(segments, tuple) or not isinstance(raw_tokens, tuple)
            or len(segments) != len(raw_tokens) + 1):
        raise BroadQaExternalDataError("v7 context local layout schema 漂移")
    values = []
    position = 0
    for ordinal, segment in enumerate(segments):
        if not isinstance(segment, str):
            raise BroadQaExternalDataError(
                "v7 context local segment 非字符串")
        end = position + len(segment)
        values.append((position, end))
        if ordinal < len(raw_tokens):
            token = raw_tokens[ordinal]
            if not isinstance(token, str) or not token:
                raise BroadQaExternalDataError(
                    "v7 context local raw token 漂移")
            position = end + len(token)
        else:
            position = end
    return tuple(values)


def _input_layout(
        observation: dict[str, object],
        plan: dict[str, object] | None,
        ) -> dict[str, object]:
    """只用 input/structure 字段核验并恢复 segment layout。"""
    text = observation.get("input_text")
    tokens = observation.get("structure_tokens")
    if (not isinstance(text, str) or not text
            or not isinstance(tokens, list) or not tokens
            or any(not isinstance(token, str) or not token for token in tokens)):
        raise BroadQaExternalDataError(
            "v7 context local structured observation 漂移")
    layout = localization_structure_layout_for_tokens(text, tuple(tokens))
    if plan is None:
        return layout
    segments = plan.get("segments")
    if (plan.get("observation_id") != observation.get("observation_id")
            or plan.get("source_family") != observation.get("source_family")
            or plan.get("input_sha256") != _text_sha256(text)
            or plan.get("structure_tokens") != tokens
            or not isinstance(segments, list)
            or len(segments) != len(layout["segments"])
            or plan.get("raw_structure_token_sha256")
            != [_text_sha256(token) for token in layout["raw_tokens"]]):
        raise BroadQaExternalDataError(
            "v7 context local variable plan/input 漂移")
    for ordinal, (record, segment) in enumerate(zip(
            segments, layout["segments"])):
        if (not isinstance(record, dict)
                or record.get("ordinal") != ordinal
                or record.get("input_length") != len(segment)
                or record.get("input_sha256") != _text_sha256(segment)):
            raise BroadQaExternalDataError(
                "v7 context local segment plan/input 漂移")
    return layout


def _candidate_rules(
        *,
        context: dict[str, object],
        text: str,
        position: int,
        indexed: bool,
        ) -> tuple[dict[str, object], ...]:
    """返回 indexed 或独立全扫描 reference 的当前位置候选。"""
    if indexed:
        buckets = context["target_buckets"]
        values = buckets.get(ord(text[position]), ())
        eligible = context["eligible_rule_ids"]
        return tuple(
            rule for rule in values
            if str(rule["rule_id"]) in eligible)
    return tuple(
        rule for rule in context["reference_local_rules"]
        if str(rule["input_text"])[0] == text[position])


def _proposals(
        *,
        text: str,
        segment_spans: tuple[tuple[int, int], ...],
        context: dict[str, object],
        indexed: bool,
        ) -> tuple[tuple[dict[str, object], ...], Counter]:
    """在 structure segments 内生成通过全部 applicability gate 的 proposals。"""
    values = []
    counters = Counter()
    for segment_ordinal, (segment_start, segment_end) in enumerate(
            segment_spans):
        for start in range(segment_start, segment_end):
            for rule in _candidate_rules(
                    context=context, text=text, position=start,
                    indexed=indexed):
                phrase = str(rule["input_text"])
                end = start + len(phrase)
                if (end > segment_end or text[start:end] != phrase
                        or rule["fragment_kind"] == "WHOLE_INPUT"
                        or rule["candidate_scope_kind"]
                        != "TARGET_CROSS_FAMILY"):
                    continue
                counters["proposal_examined_count"] += 1
                if phrase in context["identity_inputs"]:
                    counters["identity_veto_count"] += 1
                    continue
                if phrase in context["conflict_inputs"]:
                    counters["conflict_veto_count"] += 1
                    continue
                hits = _defeater_hits(context, rule, text, start, end)
                if hits:
                    counters["defeater_veto_count"] += 1
                    continue
                contexts = context["positive_contexts"].get(
                    str(rule["rule_id"]), frozenset())
                applicable = frozenset(
                    item for item in contexts
                    if str(item[1]) or str(item[3]))
                if _context_key(text, start, end) not in applicable:
                    counters["positive_context_miss_count"] += 1
                    continue
                values.append({
                    "input_end": end,
                    "input_start": start,
                    "output_text": str(rule["output_text"]),
                    "rule_id": str(rule["rule_id"]),
                    "segment_ordinal": segment_ordinal,
                })
                counters["applicable_proposal_count"] += 1
    values.sort(key=lambda item: (
        int(item["input_start"]), int(item["input_end"]),
        str(item["output_text"]), str(item["rule_id"])))
    return tuple(values), counters


def _overlap_count(values: tuple[dict[str, object], ...]) -> int:
    """统计 proposal span 的重叠或同 span 多解。"""
    count = 0
    for index, left in enumerate(values):
        for right in values[index + 1:]:
            if int(right["input_start"]) >= int(left["input_end"]):
                break
            if (int(left["input_start"]) < int(right["input_end"])
                    and int(right["input_start"]) < int(left["input_end"])):
                count += 1
    return count


def _rebuild_with_proposals(
        text: str,
        proposals: tuple[dict[str, object], ...],
        ) -> str:
    """按不重叠 proposal 原子重建完整 output。"""
    values = []
    position = 0
    for proposal in proposals:
        start = int(proposal["input_start"])
        end = int(proposal["input_end"])
        if start < position or not position <= start < end <= len(text):
            raise BroadQaExternalDataError(
                "v7 context local proposal span 漂移")
        values.append(text[position:start])
        values.append(str(proposal["output_text"]))
        position = end
    values.append(text[position:])
    return "".join(values)


def execute_context_scoped_local_transfer(
        *,
        observation: dict[str, object],
        plan: dict[str, object] | None,
        context: dict[str, object],
        indexed: bool,
        ) -> dict[str, object]:
    """执行一个 fail-closed segment local 原子事务。"""
    if type(indexed) is not bool:
        raise BroadQaExternalDataError("v7 context local interpreter 非法")
    text = str(observation.get("input_text", ""))
    tokens = tuple(observation.get("structure_tokens", ()))
    if plan is not None and plan.get("representation_eligible") != 1:
        payload = {
            "decision": "UNKNOWN_PLAN_INELIGIBLE",
            "input_text": text,
            "output_text": text,
            "partial_commit_count": 0,
            "proposal_count": 0,
            "structure_token_mismatch_count": 0,
        }
        return {**payload, "result_sha256": _sha256(
            canonical_json_bytes(payload))}
    layout = _input_layout(observation, plan)
    spans = _segment_spans(layout)
    proposals, counters = _proposals(
        text=text, segment_spans=spans, context=context, indexed=indexed)
    obligations = _obligation_spans(text, context["obligation_buckets"])
    overlap_count = _overlap_count(proposals)
    covered = tuple(
        obligation for obligation in obligations
        if any(int(item["input_start"]) <= obligation[0]
               and obligation[1] <= int(item["input_end"])
               for item in proposals))
    proposal_without_obligation = sum(
        not any(int(item["input_start"]) <= obligation[0]
                and obligation[1] <= int(item["input_end"])
                for obligation in obligations)
        for item in proposals)
    open_count = len(obligations) - len(covered)
    decision = "COMMIT"
    if not proposals:
        decision = "UNKNOWN_NO_PROPOSAL"
    elif overlap_count:
        decision = "UNKNOWN_OVERLAPPING_OR_AMBIGUOUS_PROPOSAL"
    elif not obligations or proposal_without_obligation:
        decision = "UNKNOWN_PROPOSAL_WITHOUT_OBLIGATION"
    elif open_count:
        decision = "UNKNOWN_OPEN_OBLIGATION"
    output = text if decision != "COMMIT" else _rebuild_with_proposals(
        text, proposals)
    structure_mismatch = 0
    if decision == "COMMIT":
        try:
            output_layout = localization_structure_layout_for_tokens(
                output, tokens)
            structure_mismatch = int(
                output_layout["raw_tokens"] != layout["raw_tokens"])
        except BroadQaExternalDataError:
            structure_mismatch = 1
        if structure_mismatch or output == text:
            output = text
            decision = (
                "UNKNOWN_STRUCTURE_TOKEN_MISMATCH" if structure_mismatch
                else "UNKNOWN_IDENTITY_PROPOSAL")
    payload = {
        "applicable_proposal_count": counters[
            "applicable_proposal_count"],
        "conflict_veto_count": counters["conflict_veto_count"],
        "decision": decision,
        "defeater_veto_count": counters["defeater_veto_count"],
        "identity_veto_count": counters["identity_veto_count"],
        "input_text": text,
        "obligation_count": len(obligations),
        "open_obligation_count": open_count,
        "output_text": output,
        "overlap_count": overlap_count,
        "partial_commit_count": 0,
        "positive_context_miss_count": counters[
            "positive_context_miss_count"],
        "proposal_count": len(proposals),
        "proposal_examined_count": counters["proposal_examined_count"],
        "proposal_without_obligation_count": proposal_without_obligation,
        "structure_token_mismatch_count": structure_mismatch,
    }
    return {**payload, "result_sha256": _sha256(
        canonical_json_bytes(payload))}


__all__ = [
    "CONTEXT_LOCAL_RULE_REPRESENTATION_KIND",
    "CONTEXT_LOCAL_TARGET_SCOPE",
    "derive_context_local_rule_representations",
    "execute_context_scoped_local_transfer",
    "prepare_context_local_execution",
]
