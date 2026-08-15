"""派生 recovery-v5 TRAIN-only successor 策略模拟记录。"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_failure_profile import (
    NORMALIZATION_RECOVERY_V5_FAILURE_PROFILE_KIND,
    NORMALIZATION_RECOVERY_V5_FAILURE_PROFILE_STATUS,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_phrase_runtime import (
    normalization_recovery_v5_defeater_matches,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_training_audit import (
    NORMALIZATION_RECOVERY_V5_TRAINING_AUDIT_KIND,
    NORMALIZATION_RECOVERY_V5_TRAINING_AUDIT_STATUS,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_training_audit_records import (
    AUDIT_BUCKETS,
    LOSO_OUTCOMES,
    derive_normalization_recovery_v5_loso_execution,
    normalization_recovery_v5_result_bucket,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_training_records import (
    V5_SOURCE_FAMILIES,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes


NORMALIZATION_RECOVERY_V5_SUCCESSOR_CASE_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V5_SUCCESSOR_CASE_V1")
NORMALIZATION_RECOVERY_V5_SUCCESSOR_FAMILY_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V5_SUCCESSOR_FAMILY_V1")
NORMALIZATION_RECOVERY_V5_SUCCESSOR_STRATEGY_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V5_SUCCESSOR_STRATEGY_V1")

SUCCESSOR_STRATEGIES = (
    "WHOLE_ONLY",
    "WHOLE_STRONG",
    "LOCAL_POSITIVE_CONTEXT",
    "LOCAL_ATOMIC_COVERAGE",
)

_RULE_CLASS_ORDER = {
    "WHOLE_INPUT_EQUAL_LENGTH": 0,
    "WHOLE_INPUT_VARIABLE_LENGTH": 1,
    "CONTEXT_HUNK": 2,
    "EDIT_CORE": 3,
}


def _sha256(payload: bytes) -> str:
    """返回规范记录或文本摘要。"""
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


def _outcome(observation: dict[str, object], actual: str) -> str:
    """在执行完成后按 sealed denominator 评分。"""
    if actual == observation["output_text"]:
        return "EXACT"
    if actual == observation["input_text"]:
        return "UNKNOWN"
    return "WRONG"


def _rule_order(rule: dict[str, object]) -> tuple[object, ...]:
    """复现 frozen v5 runtime 的 whole/local 完全序。"""
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
            "v5 successor rule priority schema 漂移") from error


def _reason_counts(values: list[str]) -> dict[str, int]:
    """把一次执行的决策原因冻结为排序整数计数。"""
    counts = Counter(values)
    return {key: counts[key] for key in sorted(counts)}


def _step(
        *,
        rule: dict[str, object] | None,
        start: int,
        end: int,
        output_text: str,
        mode: str,
        blocked_defeater_ids: tuple[str, ...] = (),
        ) -> dict[str, object]:
    """形成与 audit bucket 兼容的 committed execution step。"""
    return {
        "blocked_defeater_ids": list(blocked_defeater_ids),
        "candidate_scope_kind": (
            "NONE" if rule is None else rule["candidate_scope_kind"]),
        "fragment_kind": "" if rule is None else rule["fragment_kind"],
        "input_end": end,
        "input_start": start,
        "mode": mode,
        "output_text": output_text,
        "rule_class": "" if rule is None else rule["rule_class"],
        "rule_id": "" if rule is None else rule["rule_id"],
        "source_execution_family": (
            "" if rule is None else rule["source_execution_family"]),
    }


def _rule_structure_matches(
        rule: dict[str, object],
        structure_tokens: tuple[str, ...],
        ) -> bool:
    """复核 whole 结构 variants，并禁止结构化输入执行局部规则。"""
    variants = tuple(tuple(item) for item in rule["structure_token_variants"])
    if structure_tokens:
        return (rule["fragment_kind"] == "WHOLE_INPUT"
                and structure_tokens in variants)
    return not rule["application_scope"]["structure_match_required"]


def _program_context(
        *,
        program: dict[str, object],
        outputs: dict[str, tuple[dict[str, object], ...]],
        training_groups: tuple[dict[str, object], ...],
        source_family: str,
        ) -> dict[str, object]:
    """从已由 frozen runtime 核验的 program 建立策略只读索引。"""
    payload = {key: value for key, value in program.items()
               if key != "program_sha256"}
    if (program.get("program_sha256")
            != _sha256(canonical_json_bytes(payload))):
        raise BroadQaExternalDataError("v5 successor program identity 漂移")
    target_buckets = {
        int(item["first_scalar"]): tuple(item["rules"])
        for item in program["target_buckets"]}
    source_buckets: dict[int, tuple[dict[str, object], ...]] = {}
    for source_program in program["source_programs"]:
        if source_program["source_family"] == source_family:
            source_buckets = {
                int(item["first_scalar"]): tuple(item["rules"])
                for item in source_program["buckets"]}
            break
    all_rules = tuple(
        rule for buckets in (target_buckets, source_buckets)
        for rules in buckets.values() for rule in rules)
    rule_by_id = {str(item["rule_id"]): item for item in all_rules}
    if len(rule_by_id) != len(all_rules):
        raise BroadQaExternalDataError(
            "v5 successor program rule identity 重复")
    defeaters_by_rule: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in program["defeaters"]:
        defeaters_by_rule[str(record["rule_id"])].append(record)
    evidence_by_id = {str(item["evidence_id"]): item
                      for item in outputs["evidence.jsonl"]}
    positive_contexts = {}
    for rule_id, rule in rule_by_id.items():
        contexts = set()
        for evidence_id in rule["positive_evidence_ids"]:
            evidence = evidence_by_id.get(str(evidence_id))
            if (evidence is None or evidence.get("stance") != "SUPPORT"
                    or evidence.get("candidate_id") != rule["candidate_id"]):
                raise BroadQaExternalDataError(
                    "v5 successor rule/SUPPORT 未闭合")
            signature = evidence["context_signature"]
            contexts.add((
                int(signature["left_boundary"]),
                str(signature["left_context"]),
                int(signature["right_boundary"]),
                str(signature["right_context"]),
            ))
        if not contexts:
            raise BroadQaExternalDataError(
                "v5 successor rule 缺少正 applicability")
        positive_contexts[rule_id] = frozenset(contexts)
    obligation_phrases = set()
    for group in training_groups:
        variants = group.get("output_variants")
        if (group.get("candidate_scope_kind") == "TARGET_CROSS_FAMILY"
                and group.get("fragment_kind") in {"EDIT_CORE", "CONTEXT_HUNK"}
                and isinstance(variants, list)
                and any(isinstance(item, dict)
                        and item.get("output_text") != group.get("input_text")
                        for item in variants)):
            phrase = group.get("input_text")
            if not isinstance(phrase, str) or not phrase:
                raise BroadQaExternalDataError(
                    "v5 successor obligation phrase 漂移")
            obligation_phrases.add(phrase)
    obligation_buckets: dict[str, list[str]] = defaultdict(list)
    for phrase in obligation_phrases:
        obligation_buckets[phrase[0]].append(phrase)
    return {
        "conflict_inputs": frozenset(
            str(item["input_text"])
            for item in outputs["conflict-ledger.jsonl"]
            if item["candidate_scope_kind"] in {"NONE", "TARGET_CROSS_FAMILY"}),
        "defeaters_by_rule": {
            key: tuple(sorted(values,
                              key=lambda item: str(item["defeater_id"])))
            for key, values in defeaters_by_rule.items()},
        "identity_inputs": frozenset(
            str(item["input_text"])
            for item in outputs["identity-observations.jsonl"]),
        "obligation_buckets": {
            key: tuple(sorted(values, key=lambda item: (-len(item), item)))
            for key, values in obligation_buckets.items()},
        "positive_contexts": positive_contexts,
        "rule_by_id": rule_by_id,
        "source_buckets": source_buckets,
        "target_buckets": target_buckets,
    }


def _context_key(text: str, start: int, end: int) -> tuple[object, ...]:
    """返回与 SUPPORT context signature 等价的无摘要 predicate。"""
    return (
        int(start == 0),
        text[max(0, start - 4):start],
        int(end == len(text)),
        text[end:min(len(text), end + 4)],
    )


def _candidates_at(
        context: dict[str, object],
        text: str,
        position: int,
        ) -> tuple[dict[str, object], ...]:
    """返回当前位置首 scalar 对应的 target/source 完全序候选。"""
    scalar = ord(text[position])
    values = (context["target_buckets"].get(scalar, ())
              + context["source_buckets"].get(scalar, ()))
    return tuple(sorted(values, key=_rule_order))


def _defeater_hits(
        context: dict[str, object],
        rule: dict[str, object],
        text: str,
        start: int,
        end: int,
        ) -> tuple[str, ...]:
    """执行 frozen v5 negative-context defeater predicate。"""
    return tuple(sorted(
        str(item["defeater_id"])
        for item in context["defeaters_by_rule"].get(
            str(rule["rule_id"]), ())
        if normalization_recovery_v5_defeater_matches(
            item, text, start, end)))


def _whole_result(
        *,
        strategy: str,
        text: str,
        structure_tokens: tuple[str, ...],
        context: dict[str, object],
        ) -> dict[str, object]:
    """执行 whole-only 安全下界或 strong authority/veto 策略。"""
    reasons = []
    proposals = 0
    rejected = 0
    chosen = None
    blocked_ids: tuple[str, ...] = ()
    if text:
        for rule in _candidates_at(context, text, 0):
            if (rule["fragment_kind"] != "WHOLE_INPUT"
                    or rule["input_text"] != text
                    or not _rule_structure_matches(rule, structure_tokens)):
                continue
            proposals += 1
            if strategy == "WHOLE_STRONG":
                if rule["candidate_scope_kind"] != "TARGET_CROSS_FAMILY":
                    rejected += 1
                    reasons.append("STRONG_TARGET_SCOPE_REQUIRED")
                    continue
                if len(rule["source_families"]) < 3:
                    rejected += 1
                    reasons.append("STRONG_THREE_FAMILY_REQUIRED")
                    continue
                if text in context["identity_inputs"]:
                    rejected += 1
                    reasons.append("STRONG_IDENTITY_VETO")
                    continue
                if text in context["conflict_inputs"]:
                    rejected += 1
                    reasons.append("STRONG_CONFLICT_VETO")
                    continue
            hits = _defeater_hits(context, rule, text, 0, len(text))
            if hits:
                rejected += 1
                reasons.append("DEFEATER_BLOCK")
                blocked_ids = tuple(sorted(set(blocked_ids + hits)))
                continue
            chosen = rule
            break
    if chosen is None:
        reasons.append("WHOLE_IDENTITY_BACKOFF")
        steps = (_step(
            rule=None, start=0, end=len(text), output_text=text,
            mode="IDENTITY", blocked_defeater_ids=blocked_ids),)
        output = text
        committed = 0
    else:
        reasons.append("WHOLE_COMMIT")
        output = str(chosen["output_text"])
        steps = (_step(
            rule=chosen, start=0, end=len(text), output_text=output,
            mode="WHOLE_INPUT_EXACT", blocked_defeater_ids=blocked_ids),)
        committed = 1
    return _simulation_result(
        strategy=strategy,
        text=text,
        output=output,
        steps=steps,
        proposal_count=proposals,
        committed_proposal_count=committed,
        rejected_proposal_count=rejected,
        reasons=reasons,
        obligation_count=0,
        open_obligation_count=0,
    )


def _positive_context_result(
        *,
        text: str,
        structure_tokens: tuple[str, ...],
        context: dict[str, object],
        ) -> dict[str, object]:
    """执行 whole-first、local SUPPORT-derived applicability allowlist。"""
    whole = _whole_result(
        strategy="WHOLE_ONLY", text=text,
        structure_tokens=structure_tokens, context=context)
    if any(step["mode"] == "WHOLE_INPUT_EXACT" for step in whole["steps"]):
        return _simulation_result(
            strategy="LOCAL_POSITIVE_CONTEXT",
            text=text,
            output=str(whole["output_text"]),
            steps=tuple(whole["steps"]),
            proposal_count=int(whole["proposal_count"]),
            committed_proposal_count=int(whole["committed_proposal_count"]),
            rejected_proposal_count=int(whole["rejected_proposal_count"]),
            reasons=["WHOLE_COMMIT"],
            obligation_count=0,
            open_obligation_count=0,
        )
    output = []
    steps = []
    reasons = [
        reason
        for reason, count in whole["decision_reason_counts"].items()
        for _item in range(int(count))]
    proposals = int(whole["proposal_count"])
    committed = 0
    rejected = int(whole["rejected_proposal_count"])
    position = 0
    while position < len(text):
        chosen = None
        blocked_ids = []
        for rule in _candidates_at(context, text, position):
            phrase = str(rule["input_text"])
            end = position + len(phrase)
            if (rule["fragment_kind"] == "WHOLE_INPUT"
                    or structure_tokens
                    or end > len(text)
                    or text[position:end] != phrase
                    or not _rule_structure_matches(rule, structure_tokens)):
                continue
            proposals += 1
            hits = _defeater_hits(context, rule, text, position, end)
            if hits:
                rejected += 1
                reasons.append("DEFEATER_BLOCK")
                blocked_ids.extend(hits)
                continue
            contexts = context["positive_contexts"].get(
                str(rule["rule_id"]), frozenset())
            if _context_key(text, position, end) not in contexts:
                rejected += 1
                reasons.append("POSITIVE_CONTEXT_MISS")
                continue
            chosen = rule
            break
        if chosen is None:
            end = position + 1
            piece = text[position:end]
            reasons.append("IDENTITY_BACKOFF")
            steps.append(_step(
                rule=None, start=position, end=end, output_text=piece,
                mode="IDENTITY",
                blocked_defeater_ids=tuple(sorted(set(blocked_ids)))))
        else:
            end = position + len(str(chosen["input_text"]))
            piece = str(chosen["output_text"])
            committed += 1
            reasons.append("LOCAL_POSITIVE_CONTEXT_COMMIT")
            steps.append(_step(
                rule=chosen, start=position, end=end, output_text=piece,
                mode="LONGEST_LOCAL_MATCH",
                blocked_defeater_ids=tuple(sorted(set(blocked_ids)))))
        output.append(piece)
        position = end
    if not text:
        reasons.append("IDENTITY_BACKOFF")
    return _simulation_result(
        strategy="LOCAL_POSITIVE_CONTEXT",
        text=text,
        output="".join(output),
        steps=tuple(steps),
        proposal_count=proposals,
        committed_proposal_count=committed,
        rejected_proposal_count=rejected,
        reasons=reasons,
        obligation_count=0,
        open_obligation_count=0,
    )


def _obligation_spans(
        text: str,
        obligation_buckets: dict[str, tuple[str, ...]],
        ) -> tuple[tuple[int, int], ...]:
    """索引所有 TRAIN-derived target local normalization obligations。"""
    values = set()
    for start, scalar in enumerate(text):
        for phrase in obligation_buckets.get(scalar, ()):
            end = start + len(phrase)
            if end <= len(text) and text[start:end] == phrase:
                values.add((start, end))
    return tuple(sorted(values))


def _atomic_result(
        *,
        text: str,
        structure_tokens: tuple[str, ...],
        baseline_result: dict[str, object],
        context: dict[str, object],
        ) -> dict[str, object]:
    """把 frozen local proposals 作为整句事务做 obligation 闭包。"""
    del structure_tokens
    steps = tuple(baseline_result["steps"])
    rule_steps = tuple(step for step in steps if step["rule_id"])
    if any(step["mode"] == "WHOLE_INPUT_EXACT" for step in rule_steps):
        return _simulation_result(
            strategy="LOCAL_ATOMIC_COVERAGE",
            text=text,
            output=str(baseline_result["output_text"]),
            steps=steps,
            proposal_count=len(rule_steps),
            committed_proposal_count=len(rule_steps),
            rejected_proposal_count=sum(
                len(step["blocked_defeater_ids"]) for step in steps),
            reasons=["WHOLE_COMMIT"],
            obligation_count=0,
            open_obligation_count=0,
        )
    local_steps = tuple(
        step for step in rule_steps if step["mode"] == "LONGEST_LOCAL_MATCH")
    obligations = _obligation_spans(
        text, context["obligation_buckets"])
    selected_spans = tuple(
        (int(step["input_start"]), int(step["input_end"]))
        for step in local_steps)
    open_obligations = tuple(
        span for span in obligations
        if not any(start <= span[0] and span[1] <= end
                   for start, end in selected_spans))
    blocked_count = sum(len(step["blocked_defeater_ids"]) for step in steps)
    if local_steps and not open_obligations:
        reasons = ["LOCAL_ATOMIC_TRANSACTION_COMMIT"]
        output = str(baseline_result["output_text"])
        committed = len(local_steps)
        rejected = blocked_count
        committed_steps = steps
    else:
        reasons = [
            "ATOMIC_OBLIGATION_OPEN" if open_obligations
            else "ATOMIC_NO_LOCAL_PROPOSAL"]
        output = text
        committed = 0
        rejected = blocked_count + len(local_steps)
        committed_steps = (_step(
            rule=None, start=0, end=len(text), output_text=text,
            mode="IDENTITY"),)
    return _simulation_result(
        strategy="LOCAL_ATOMIC_COVERAGE",
        text=text,
        output=output,
        steps=committed_steps,
        proposal_count=len(local_steps) + blocked_count,
        committed_proposal_count=committed,
        rejected_proposal_count=rejected,
        reasons=reasons,
        obligation_count=len(obligations),
        open_obligation_count=len(open_obligations),
    )


def _simulation_result(
        *,
        strategy: str,
        text: str,
        output: str,
        steps: tuple[dict[str, object], ...],
        proposal_count: int,
        committed_proposal_count: int,
        rejected_proposal_count: int,
        reasons: list[str],
        obligation_count: int,
        open_obligation_count: int,
        ) -> dict[str, object]:
    """冻结一次策略执行，结果摘要不含 held-out expected。"""
    payload = {
        "committed_proposal_count": committed_proposal_count,
        "decision_reason_counts": _reason_counts(reasons),
        "input_text": text,
        "obligation_count": obligation_count,
        "open_obligation_count": open_obligation_count,
        "output_text": output,
        "proposal_count": proposal_count,
        "rejected_proposal_count": rejected_proposal_count,
        "steps": list(steps),
        "strategy": strategy,
    }
    return {**payload, "result_sha256": _sha256(canonical_json_bytes(payload))}


def simulate_normalization_recovery_v5_successor_strategy(
        *,
        strategy: str,
        observation: dict[str, object],
        baseline_result: dict[str, object],
        program: dict[str, object],
        outputs: dict[str, tuple[dict[str, object], ...]],
        training_groups: tuple[dict[str, object], ...],
        held_out_source_family: str,
        ) -> dict[str, object]:
    """只用 TRAIN-derived material 模拟一个 successor 策略。"""
    if strategy not in SUCCESSOR_STRATEGIES:
        raise BroadQaExternalDataError("v5 successor strategy 非法")
    text = observation.get("input_text")
    tokens = observation.get("structure_tokens")
    baseline_payload = {
        key: value for key, value in baseline_result.items()
        if key != "result_sha256"}
    if (not isinstance(text, str) or not text
            or not isinstance(tokens, list)
            or any(not isinstance(item, str) for item in tokens)
            or baseline_result.get("input_text") != text
            or baseline_result.get("source_family") != held_out_source_family
            or baseline_result.get("program_sha256")
            != program.get("program_sha256")
            or baseline_result.get("result_sha256")
            != _sha256(canonical_json_bytes(baseline_payload))):
        raise BroadQaExternalDataError("v5 successor observation/result 漂移")
    context = _program_context(
        program=program,
        outputs=outputs,
        training_groups=training_groups,
        source_family=held_out_source_family,
    )
    return _simulate_with_context(
        strategy=strategy,
        observation=observation,
        baseline_result=baseline_result,
        context=context,
    )


def _simulate_with_context(
        *,
        strategy: str,
        observation: dict[str, object],
        baseline_result: dict[str, object],
        context: dict[str, object],
        ) -> dict[str, object]:
    """在一个 LOSO family 的共享只读索引上执行单个策略。"""
    text = str(observation["input_text"])
    structure_tokens = tuple(observation["structure_tokens"])
    if strategy in {"WHOLE_ONLY", "WHOLE_STRONG"}:
        return _whole_result(
            strategy=strategy, text=text,
            structure_tokens=structure_tokens, context=context)
    if strategy == "LOCAL_POSITIVE_CONTEXT":
        return _positive_context_result(
            text=text, structure_tokens=structure_tokens, context=context)
    return _atomic_result(
        text=text, structure_tokens=structure_tokens,
        baseline_result=baseline_result, context=context)


def _manifest_contracts(
        *,
        protocol_sha: str,
        audit_sha: str,
        audit_manifest: dict[str, object],
        profile_sha: str,
        profile_manifest: dict[str, object],
        ) -> dict[str, object]:
    """核验 sealed audit/profile 只允许 TRAIN-only capability FAIL 输入。"""
    audit_summary = audit_manifest.get("summary")
    profile_summary = profile_manifest.get("summary")
    loso_counts = (
        audit_summary.get("loso_counts")
        if isinstance(audit_summary, dict) else None)
    if (not isinstance(loso_counts, dict)
            or any(type(loso_counts.get(f"{family}:WRONG")) is not int
                   for family in V5_SOURCE_FAMILIES)):
        raise BroadQaExternalDataError(
            "v5 successor sealed audit LOSO contract 漂移")
    audit_wrong_count = sum(
        int(loso_counts[f"{family}:WRONG"])
        for family in V5_SOURCE_FAMILIES)
    if (audit_manifest.get("artifact_kind")
            != NORMALIZATION_RECOVERY_V5_TRAINING_AUDIT_KIND
            or audit_manifest.get("status")
            != NORMALIZATION_RECOVERY_V5_TRAINING_AUDIT_STATUS
            or audit_manifest.get("protocol_manifest_sha256") != protocol_sha
            or audit_manifest.get("formal_run_count") != 0
            or audit_manifest.get("production_enabled") != 0
            or audit_manifest.get("mastery_claimed") != 0
            or not isinstance(audit_summary, dict)
            or audit_summary.get("audit_outcome")
            != "FACILITY_PASS_CAPABILITY_FAIL"
            or audit_summary.get("facility_failure_count") != 0
            or audit_summary.get("capability_gate_pass") != 0
            or profile_manifest.get("artifact_kind")
            != NORMALIZATION_RECOVERY_V5_FAILURE_PROFILE_KIND
            or profile_manifest.get("status")
            != NORMALIZATION_RECOVERY_V5_FAILURE_PROFILE_STATUS
            or profile_manifest.get("protocol_manifest_sha256") != protocol_sha
            or profile_manifest.get("training_audit_manifest_sha256")
            != audit_sha
            or profile_manifest.get("formal_run_count") != 0
            or profile_manifest.get("production_enabled") != 0
            or profile_manifest.get("mastery_claimed") != 0
            or profile_manifest.get("selection_or_threshold_changed") != 0
            or not isinstance(profile_summary, dict)
            or profile_summary.get("protocol_manifest_sha256") != protocol_sha
            or profile_summary.get("audit_manifest_sha256") != audit_sha
            or profile_summary.get("wrong_case_count") != audit_wrong_count):
        raise BroadQaExternalDataError(
            "v5 successor sealed audit/profile contract 漂移")
    _sha_value(profile_sha, label="v5 successor profile manifest")
    return audit_summary


def _case_record(
        *,
        protocol_sha: str,
        audit_sha: str,
        profile_sha: str,
        family: str,
        strategy: str,
        observation: dict[str, object],
        baseline_result: dict[str, object],
        strategy_result: dict[str, object],
        denominator_bucket: str,
        ) -> dict[str, object]:
    """记录一个固定 denominator 上的策略结果和决策原因。"""
    identity = {
        "held_out_observation_id": observation["observation_id"],
        "held_out_source_family": family,
        "profile_manifest_sha256": profile_sha,
        "strategy": strategy,
    }
    actual = str(strategy_result["output_text"])
    expected = str(observation["output_text"])
    source = str(observation["input_text"])
    outcome = _outcome(observation, actual)
    steps = strategy_result["steps"]
    return {
        **identity,
        "actual_output_sha256": _sha256(actual.encode("utf-8")),
        "audit_manifest_sha256": audit_sha,
        "baseline_result_sha256": baseline_result["result_sha256"],
        "case_id": _record_id(identity),
        "committed_proposal_count": strategy_result[
            "committed_proposal_count"],
        "decision_reason_counts": strategy_result["decision_reason_counts"],
        "denominator_bucket": denominator_bucket,
        "expected_output_sha256": _sha256(expected.encode("utf-8")),
        "format_version": 1,
        "identity_false_change": int(
            observation["identity_preservation"] == 1 and actual != source),
        "input_sha256": _sha256(source.encode("utf-8")),
        "obligation_count": strategy_result["obligation_count"],
        "open_obligation_count": strategy_result["open_obligation_count"],
        "outcome": outcome,
        "production_enabled": 0,
        "proposal_count": strategy_result["proposal_count"],
        "protocol_manifest_sha256": protocol_sha,
        "record_kind": NORMALIZATION_RECOVERY_V5_SUCCESSOR_CASE_KIND,
        "rejected_proposal_count": strategy_result[
            "rejected_proposal_count"],
        "selected_rule_ids": sorted({
            str(item["rule_id"]) for item in steps if item["rule_id"]}),
        "strategy_result_sha256": strategy_result["result_sha256"],
    }


def _family_record(
        *,
        protocol_sha: str,
        audit_sha: str,
        profile_sha: str,
        family: str,
        strategy: str,
        cases: tuple[dict[str, object], ...],
        subset_protocol_sha: str,
        subset_pack_sha: str,
        ) -> dict[str, object]:
    """聚合一个 family/strategy 的固定五桶结果。"""
    outcome_counts = Counter(str(item["outcome"]) for item in cases)
    bucket_counts = Counter((str(item["denominator_bucket"]),
                             str(item["outcome"])) for item in cases)
    reason_counts = Counter()
    for item in cases:
        reason_counts.update(item["decision_reason_counts"])
    identity = {
        "held_out_source_family": family,
        "profile_manifest_sha256": profile_sha,
        "strategy": strategy,
        "subset_protocol_manifest_sha256": subset_protocol_sha,
    }
    return {
        **identity,
        "audit_manifest_sha256": audit_sha,
        "bucket_outcome_counts": {
            f"{bucket}:{outcome}": bucket_counts[(bucket, outcome)]
            for bucket in AUDIT_BUCKETS for outcome in LOSO_OUTCOMES},
        "case_count": len(cases),
        "case_ids_sha256": _sha256(canonical_json_bytes(
            [item["case_id"] for item in cases])),
        "committed_proposal_count": sum(
            int(item["committed_proposal_count"]) for item in cases),
        "decision_reason_counts": {
            key: reason_counts[key] for key in sorted(reason_counts)},
        "family_result_id": _record_id(identity),
        "format_version": 1,
        "identity_false_change_count": sum(
            int(item["identity_false_change"]) for item in cases),
        "obligation_count": sum(int(item["obligation_count"])
                                for item in cases),
        "open_obligation_count": sum(
            int(item["open_obligation_count"]) for item in cases),
        "outcome_counts": {
            key: outcome_counts[key] for key in LOSO_OUTCOMES},
        "production_enabled": 0,
        "proposal_count": sum(int(item["proposal_count"])
                              for item in cases),
        "protocol_manifest_sha256": protocol_sha,
        "record_kind": NORMALIZATION_RECOVERY_V5_SUCCESSOR_FAMILY_KIND,
        "rejected_proposal_count": sum(
            int(item["rejected_proposal_count"]) for item in cases),
        "subset_pack_manifest_sha256": subset_pack_sha,
    }


def _strategy_record(
        *,
        protocol_sha: str,
        audit_sha: str,
        profile_sha: str,
        strategy: str,
        family_records: tuple[dict[str, object], ...],
        ) -> dict[str, object]:
    """聚合四个 LOSO family，并形成不冒充 selection 的晋升资格事实。"""
    outcomes = Counter()
    buckets = Counter()
    reasons = Counter()
    for item in family_records:
        outcomes.update(item["outcome_counts"])
        buckets.update(item["bucket_outcome_counts"])
        reasons.update(item["decision_reason_counts"])
    non_identity_exact = sum(
        buckets[f"{bucket}:EXACT"] for bucket in AUDIT_BUCKETS
        if bucket != "IDENTITY")
    identity_false_change = sum(
        int(item["identity_false_change_count"]) for item in family_records)
    promotion_eligible = int(
        outcomes["WRONG"] == 0
        and identity_false_change == 0
        and non_identity_exact > 0)
    identity = {
        "profile_manifest_sha256": profile_sha,
        "strategy": strategy,
    }
    return {
        **identity,
        "audit_manifest_sha256": audit_sha,
        "bucket_outcome_counts": {
            key: buckets[key]
            for key in sorted(buckets)},
        "committed_proposal_count": sum(
            int(item["committed_proposal_count"])
            for item in family_records),
        "decision_reason_counts": {
            key: reasons[key] for key in sorted(reasons)},
        "family_count": len(family_records),
        "family_result_ids_sha256": _sha256(canonical_json_bytes(
            [item["family_result_id"] for item in family_records])),
        "format_version": 1,
        "identity_false_change_count": identity_false_change,
        "non_identity_exact_count": non_identity_exact,
        "obligation_count": sum(int(item["obligation_count"])
                                for item in family_records),
        "open_obligation_count": sum(
            int(item["open_obligation_count"]) for item in family_records),
        "outcome_counts": {key: outcomes[key] for key in LOSO_OUTCOMES},
        "production_enabled": 0,
        "promotion_eligible": promotion_eligible,
        "proposal_count": sum(int(item["proposal_count"])
                              for item in family_records),
        "protocol_manifest_sha256": protocol_sha,
        "record_kind": NORMALIZATION_RECOVERY_V5_SUCCESSOR_STRATEGY_KIND,
        "rejected_proposal_count": sum(
            int(item["rejected_proposal_count"])
            for item in family_records),
        "strategy_result_id": _record_id(identity),
    }


def derive_normalization_recovery_v5_successor_simulation(
        *,
        protocol_manifest: dict[str, object],
        observations: tuple[dict[str, object], ...],
        fragments: tuple[dict[str, object], ...],
        audit_manifest_sha256: str,
        audit_manifest: dict[str, object],
        profile_manifest_sha256: str,
        profile_manifest: dict[str, object],
        ) -> tuple[tuple[dict[str, object], ...],
                   tuple[dict[str, object], ...],
                   tuple[dict[str, object], ...], dict[str, object]]:
    """每方向只重学一次，并在固定 denominator 上回放四策略。"""
    protocol_sha = _sha_value(
        protocol_manifest.get("manifest_sha256"),
        label="v5 successor protocol manifest")
    audit_sha = _sha_value(
        audit_manifest_sha256, label="v5 successor audit manifest")
    profile_sha = _sha_value(
        profile_manifest_sha256, label="v5 successor profile manifest")
    audit_summary = _manifest_contracts(
        protocol_sha=protocol_sha,
        audit_sha=audit_sha,
        audit_manifest=audit_manifest,
        profile_sha=profile_sha,
        profile_manifest=profile_manifest,
    )
    if not observations or not fragments:
        raise BroadQaExternalDataError("v5 successor TRAIN material 为空")
    cases = []
    families = []
    baseline_loso = Counter()
    baseline_buckets = Counter()
    for family in V5_SOURCE_FAMILIES:
        material = derive_normalization_recovery_v5_loso_execution(
            protocol_manifest_sha256=protocol_sha,
            observations=observations,
            fragments=fragments,
            held_out_source_family=family,
            include_reference=False,
        )
        context = _program_context(
            program=material["program"],
            outputs=material["outputs"],
            training_groups=material["training_groups"],
            source_family=family,
        )
        family_cases: dict[str, list[dict[str, object]]] = {
            strategy: [] for strategy in SUCCESSOR_STRATEGIES}
        for observation, baseline_result in zip(
                material["held_out_observations"],
                material["indexed_results"]):
            baseline_outcome = _outcome(
                observation, str(baseline_result["output_text"]))
            denominator_bucket = normalization_recovery_v5_result_bucket(
                observation, baseline_result)
            baseline_loso[(family, baseline_outcome)] += 1
            baseline_buckets[(denominator_bucket, baseline_outcome)] += 1
            for strategy in SUCCESSOR_STRATEGIES:
                result = _simulate_with_context(
                    strategy=strategy,
                    observation=observation,
                    baseline_result=baseline_result,
                    context=context,
                )
                case = _case_record(
                    protocol_sha=protocol_sha,
                    audit_sha=audit_sha,
                    profile_sha=profile_sha,
                    family=family,
                    strategy=strategy,
                    observation=observation,
                    baseline_result=baseline_result,
                    strategy_result=result,
                    denominator_bucket=denominator_bucket,
                )
                cases.append(case)
                family_cases[strategy].append(case)
        for strategy in SUCCESSOR_STRATEGIES:
            frozen = tuple(sorted(
                family_cases[strategy], key=lambda item: str(item["case_id"])))
            families.append(_family_record(
                protocol_sha=protocol_sha,
                audit_sha=audit_sha,
                profile_sha=profile_sha,
                family=family,
                strategy=strategy,
                cases=frozen,
                subset_protocol_sha=str(
                    material["subset_protocol_manifest_sha256"]),
                subset_pack_sha=str(material["subset_pack_manifest_sha256"]),
            ))
    actual_loso = {
        f"{family}:{outcome}": baseline_loso[(family, outcome)]
        for family in V5_SOURCE_FAMILIES for outcome in LOSO_OUTCOMES}
    actual_buckets = {
        f"{bucket}:{outcome}": baseline_buckets[(bucket, outcome)]
        for bucket in AUDIT_BUCKETS for outcome in LOSO_OUTCOMES}
    if (actual_loso != audit_summary.get("loso_counts")
            or actual_buckets != audit_summary.get("bucket_outcome_counts")):
        raise BroadQaExternalDataError(
            "v5 successor 与 sealed audit denominator 漂移")
    frozen_cases = tuple(sorted(cases, key=lambda item: str(item["case_id"])))
    frozen_families = tuple(sorted(
        families, key=lambda item: str(item["family_result_id"])))
    strategy_records = tuple(sorted((
        _strategy_record(
            protocol_sha=protocol_sha,
            audit_sha=audit_sha,
            profile_sha=profile_sha,
            strategy=strategy,
            family_records=tuple(sorted((
                item for item in frozen_families
                if item["strategy"] == strategy),
                key=lambda item: str(item["family_result_id"]))),
        ) for strategy in SUCCESSOR_STRATEGIES),
        key=lambda item: str(item["strategy_result_id"])))
    eligible = sorted(
        str(item["strategy"]) for item in strategy_records
        if item["promotion_eligible"] == 1)
    summary = {
        "audit_manifest_sha256": audit_sha,
        "baseline_bucket_outcome_counts": actual_buckets,
        "baseline_loso_counts": actual_loso,
        "case_count": len(frozen_cases),
        "family_result_count": len(frozen_families),
        "formal_run_count": 0,
        "loso_relearn_count": len(V5_SOURCE_FAMILIES),
        "mastery_claimed": 0,
        "profile_manifest_sha256": profile_sha,
        "production_enabled": 0,
        "promotion_eligible_strategies": eligible,
        "protocol_manifest_sha256": protocol_sha,
        "selection_label_read_count": 0,
        "selection_or_threshold_changed": 0,
        "strategy_replay_count": len(V5_SOURCE_FAMILIES) * len(
            SUCCESSOR_STRATEGIES),
        "strategy_result_count": len(strategy_records),
        "strategy_result_sha256": _sha256(canonical_json_bytes([
            item["strategy_result_id"] for item in strategy_records])),
    }
    if (len(frozen_cases) != len(observations) * len(SUCCESSOR_STRATEGIES)
            or len(frozen_families)
            != len(V5_SOURCE_FAMILIES) * len(SUCCESSOR_STRATEGIES)
            or len(strategy_records) != len(SUCCESSOR_STRATEGIES)):
        raise BroadQaExternalDataError(
            "v5 successor simulation inventory 未闭合")
    return frozen_cases, frozen_families, strategy_records, summary


__all__ = [
    "NORMALIZATION_RECOVERY_V5_SUCCESSOR_CASE_KIND",
    "NORMALIZATION_RECOVERY_V5_SUCCESSOR_FAMILY_KIND",
    "NORMALIZATION_RECOVERY_V5_SUCCESSOR_STRATEGY_KIND",
    "SUCCESSOR_STRATEGIES",
    "derive_normalization_recovery_v5_successor_simulation",
    "simulate_normalization_recovery_v5_successor_strategy",
]
