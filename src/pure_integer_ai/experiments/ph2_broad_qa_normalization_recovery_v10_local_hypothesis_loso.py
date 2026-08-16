"""审计 recovery-v10 局部 source-context hypothesis 的三方向 family LOSO。"""
from __future__ import annotations

from collections import Counter
import hashlib

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_phrase_learning import (
    normalization_phrase_alignment_boundary_map,
    normalization_phrase_context_signature,
    normalization_phrase_observed_output,
    normalization_phrase_occurrences,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_localization_structure import (
    localization_structure_layout,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_training_records import (
    V8_TRAIN_FAMILIES,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v10_local_hypothesis_contract import (
    LOCAL_ORTHOGRAPHIC_HYPOTHESIS_ONLY,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes


NORMALIZATION_RECOVERY_V10_LOCAL_HYPOTHESIS_LOSO_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V10_LOCAL_HYPOTHESIS_LOSO_V1")
NORMALIZATION_RECOVERY_V10_LOCAL_HYPOTHESIS_LOSO_STATUS = (
    "TRAIN_ONLY_AUTHORIZATION_IDENTIFIABILITY_AUDIT_NOT_FORMAL")


def _sha256(value: object) -> str:
    """返回 context、rule、application 或 audit 的规范 SHA-256。"""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha_value(value: object, *, label: str) -> str:
    """核验并返回小写 SHA-256。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(item not in "0123456789abcdef" for item in value)):
        raise BroadQaExternalDataError(
            f"v10 local hypothesis LOSO {label} 非法")
    return value


def _surface(observation: dict[str, object], role: str) -> str:
    """从 Qt/gettext 统一 Observation 提取唯一 locale 表面。"""
    value = observation.get(role)
    if not isinstance(value, dict):
        raise BroadQaExternalDataError(
            "v10 local hypothesis LOSO locale record 漂移")
    strings = [item for item in (
        value.get("translation"), value.get("msgstr"))
        if isinstance(item, str)]
    if len(strings) != 1 or not strings[0]:
        raise BroadQaExternalDataError(
            "v10 local hypothesis LOSO surface 漂移")
    return strings[0]


def _observations(
        observations: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, object], ...]:
    """核验全量 Observation identity、family、source 与 locale surface。"""
    if not isinstance(observations, tuple) or not observations:
        raise BroadQaExternalDataError(
            "v10 local hypothesis LOSO observations 为空")
    seen = set()
    values = []
    for item in observations:
        if not isinstance(item, dict):
            raise BroadQaExternalDataError(
                "v10 local hypothesis LOSO observation 非对象")
        identity = _sha_value(
            item.get("observation_id"), label="observation id")
        family = item.get("source_family")
        source = item.get("official_source_text")
        if (identity in seen or family not in V8_TRAIN_FAMILIES
                or not isinstance(source, str) or not source):
            raise BroadQaExternalDataError(
                "v10 local hypothesis LOSO observation source 漂移")
        seen.add(identity)
        _surface(item, "zh_hant")
        _surface(item, "zh_hans")
        values.append(item)
    return tuple(values)


def _segment_spans(value: str) -> tuple[tuple[int, int], ...]:
    """返回结构token之外的全部text segment绝对span。"""
    layout = localization_structure_layout(value)
    segments = layout["segments"]
    raw_tokens = layout["raw_tokens"]
    spans = []
    position = 0
    for ordinal, segment in enumerate(segments):
        end = position + len(segment)
        spans.append((position, end))
        if ordinal < len(raw_tokens):
            position = end + len(raw_tokens[ordinal])
        else:
            position = end
    return tuple(spans)


def _context(text: str, start: int, end: int) -> dict[str, object]:
    """形成四scalar上下文与稳定identity。"""
    return normalization_phrase_context_signature(
        text, start, end, identity_builder=_sha256)


def _projection_groups(
        projection_records: tuple[dict[str, object], ...],
        ) -> dict[
            tuple[str, str, str],
            list[dict[str, object]],
        ]:
    """按exact source、局部input和context identity聚合hypothesis occurrence。"""
    if not isinstance(projection_records, tuple) or not projection_records:
        raise BroadQaExternalDataError(
            "v10 local hypothesis LOSO projection records 为空")
    groups: dict[
        tuple[str, str, str], list[dict[str, object]]] = {}
    seen_observations = set()
    for record in projection_records:
        if not isinstance(record, dict):
            raise BroadQaExternalDataError(
                "v10 local hypothesis LOSO projection record 非对象")
        observation_id = _sha_value(
            record.get("observation_id"), label="projection observation id")
        family = record.get("source_family")
        source = record.get("official_source_text")
        input_surface = record.get("input_text")
        spans = record.get("span_hypotheses")
        if (observation_id in seen_observations
                or family not in V8_TRAIN_FAMILIES
                or not isinstance(source, str) or not source
                or not isinstance(input_surface, str) or not input_surface
                or not isinstance(spans, list)):
            raise BroadQaExternalDataError(
                "v10 local hypothesis LOSO projection record 漂移")
        seen_observations.add(observation_id)
        for span in spans:
            if (not isinstance(span, dict)
                    or span.get("authorization_kind")
                    != LOCAL_ORTHOGRAPHIC_HYPOTHESIS_ONLY
                    or span.get("source_context_authorization_id") != ""
                    or span.get("authorized_official_source_text") != ""
                    or span.get("training_support_families") != [family]):
                raise BroadQaExternalDataError(
                    "v10 local hypothesis LOSO span state 漂移")
            start = span.get("input_start")
            end = span.get("input_end")
            left = span.get("input_text")
            right = span.get("output_text")
            if (type(start) is not int or type(end) is not int
                    or not 0 <= start < end <= len(input_surface)
                    or input_surface[start:end] != left
                    or not isinstance(right, str) or not right):
                raise BroadQaExternalDataError(
                    "v10 local hypothesis LOSO span geometry 漂移")
            context = _context(input_surface, start, end)
            occurrence = {
                "context_signature": context,
                "evidence_ids": span["evidence_ids"],
                "input_text": left,
                "observation_id": observation_id,
                "official_source_text": source,
                "output_text": right,
                "source_family": family,
                "span_hypothesis_id": span["span_hypothesis_id"],
            }
            key = (source, str(left), str(context["context_signature_id"]))
            groups.setdefault(key, []).append(occurrence)
    return groups


def _direction_rules(
        groups: dict[
            tuple[str, str, str], list[dict[str, object]]], *,
        held_out_family: str,
        ) -> tuple[dict[str, object], ...]:
    """只用另外两家family形成unique-output source-context候选。"""
    expected_families = sorted(
        family for family in V8_TRAIN_FAMILIES
        if family != held_out_family)
    rules = []
    for (source, input_text, _context_id), occurrences in groups.items():
        training = [
            item for item in occurrences
            if item["source_family"] != held_out_family]
        families = sorted({str(item["source_family"]) for item in training})
        outputs = sorted({str(item["output_text"]) for item in training})
        if families != expected_families or len(outputs) != 1:
            continue
        context = training[0]["context_signature"]
        if any(item["context_signature"] != context for item in training):
            raise BroadQaExternalDataError(
                "v10 local hypothesis LOSO context group 漂移")
        semantic = {
            "context_signature": context,
            "input_text": input_text,
            "official_source_text": source,
            "output_text": outputs[0],
            "rule_kind": "LOSO_SOURCE_CONTEXT_SPAN_HYPOTHESIS",
        }
        payload = {
            **semantic,
            "evidence_ids": sorted({
                evidence
                for item in training for evidence in item["evidence_ids"]}),
            "held_out_family": held_out_family,
            "training_observation_ids": sorted({
                str(item["observation_id"]) for item in training}),
            "training_support_families": families,
        }
        rules.append({
            **payload,
            "candidate_rule_id": _sha256(payload),
            "rule_semantic_id": _sha256(semantic),
        })
    return tuple(sorted(rules, key=lambda item: str(item["candidate_rule_id"])))


def _held_out_applications(
        *, observations: tuple[dict[str, object], ...],
        rules: tuple[dict[str, object], ...],
        held_out_family: str,
        ) -> tuple[dict[str, object], ...]:
    """在held-out全部同source literal occurrence上读取实际输出，包含identity负例。"""
    rules_by_source: dict[str, list[dict[str, object]]] = {}
    for rule in rules:
        rules_by_source.setdefault(
            str(rule["official_source_text"]), []).append(rule)
    applications = []
    for observation in observations:
        if observation["source_family"] != held_out_family:
            continue
        source = str(observation["official_source_text"])
        source_rules = rules_by_source.get(source, [])
        if not source_rules:
            continue
        input_text = _surface(observation, "zh_hant")
        output_text = _surface(observation, "zh_hans")
        input_tokens = localization_structure_layout(input_text)[
            "structure_tokens"]
        output_tokens = localization_structure_layout(output_text)[
            "structure_tokens"]
        structure_equal = input_tokens == output_tokens
        boundaries = normalization_phrase_alignment_boundary_map(
            input_text, output_text)
        segment_spans = _segment_spans(input_text)
        for rule in source_rules:
            phrase = str(rule["input_text"])
            expected_context = rule["context_signature"]
            for segment_start, segment_end in segment_spans:
                segment = input_text[segment_start:segment_end]
                for local_start, local_end in normalization_phrase_occurrences(
                        segment, phrase):
                    start = segment_start + local_start
                    end = segment_start + local_end
                    context = _context(input_text, start, end)
                    if context != expected_context:
                        continue
                    actual = None if not structure_equal else (
                        normalization_phrase_observed_output(
                            {"input_text": input_text,
                             "output_text": output_text},
                            start, end, boundaries,
                            label="v10 local hypothesis LOSO"))
                    outcome = (
                        "UNKNOWN" if actual is None
                        else "EXACT" if actual == rule["output_text"]
                        else "WRONG")
                    payload = {
                        "actual_output_sha256": (
                            _sha256(actual) if actual is not None else ""),
                        "candidate_rule_id": rule["candidate_rule_id"],
                        "held_out_family": held_out_family,
                        "held_out_observation_id": observation["observation_id"],
                        "input_end": end,
                        "input_start": start,
                        "outcome": outcome,
                        "rule_semantic_id": rule["rule_semantic_id"],
                        "structure_equal": int(structure_equal),
                    }
                    applications.append({
                        **payload,
                        "application_id": _sha256(payload),
                    })
    if len({item["application_id"] for item in applications}) != len(
            applications):
        raise BroadQaExternalDataError(
            "v10 local hypothesis LOSO application identity 重复")
    return tuple(sorted(
        applications, key=lambda item: str(item["application_id"])))


def _predecessor_rules(
        rules: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, object], ...]:
    """核验现有whole-input exact-source规则，供新增性去重。"""
    if not isinstance(rules, tuple):
        raise BroadQaExternalDataError(
            "v10 local hypothesis LOSO predecessor rules 非tuple")
    values = []
    seen = set()
    for rule in rules:
        if not isinstance(rule, dict):
            raise BroadQaExternalDataError(
                "v10 local hypothesis LOSO predecessor rule 非对象")
        identity = _sha_value(
            rule.get("candidate_rule_id"), label="predecessor rule id")
        source = rule.get("official_source_text")
        input_text = rule.get("input_text")
        output_text = rule.get("output_text")
        if (identity in seen
                or not isinstance(source, str) or not source
                or not isinstance(input_text, str) or not input_text
                or not isinstance(output_text, str) or not output_text):
            raise BroadQaExternalDataError(
                "v10 local hypothesis LOSO predecessor rule 漂移")
        seen.add(identity)
        values.append(rule)
    return tuple(values)


def _predecessor_covers(
        survivor: dict[str, object],
        predecessor_rules: tuple[dict[str, object], ...],
        ) -> bool:
    """判断局部survivor是否只是既有whole-input source rule的切片。"""
    phrase = str(survivor["input_text"])
    expected_output = str(survivor["output_text"])
    expected_context = survivor["context_signature"]
    for rule in predecessor_rules:
        if rule["official_source_text"] != survivor["official_source_text"]:
            continue
        input_text = str(rule["input_text"])
        output_text = str(rule["output_text"])
        if (localization_structure_layout(input_text)["structure_tokens"]
                != localization_structure_layout(output_text)[
                    "structure_tokens"]):
            continue
        boundaries = normalization_phrase_alignment_boundary_map(
            input_text, output_text)
        for segment_start, segment_end in _segment_spans(input_text):
            segment = input_text[segment_start:segment_end]
            for local_start, local_end in normalization_phrase_occurrences(
                    segment, phrase):
                start = segment_start + local_start
                end = segment_start + local_end
                if _context(input_text, start, end) != expected_context:
                    continue
                actual = normalization_phrase_observed_output(
                    {"input_text": input_text, "output_text": output_text},
                    start, end, boundaries,
                    label="v10 local hypothesis predecessor coverage")
                if actual == expected_output:
                    return True
    return False


def derive_normalization_recovery_v10_local_hypothesis_loso(
        *, observations: tuple[dict[str, object], ...],
        projection_records: tuple[dict[str, object], ...],
        predecessor_source_rules: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
    """运行三方向LOSO，并只列出每方向均有held-out EXACT的零错survivor。"""
    checked_observations = _observations(observations)
    groups = _projection_groups(projection_records)
    predecessor_rules = _predecessor_rules(predecessor_source_rules)
    directions = []
    total = Counter()
    semantic_outcomes: dict[str, dict[str, Counter]] = {}
    semantic_rules: dict[str, dict[str, object]] = {}
    for held_out in V8_TRAIN_FAMILIES:
        rules = _direction_rules(groups, held_out_family=held_out)
        applications = _held_out_applications(
            observations=checked_observations,
            rules=rules,
            held_out_family=held_out,
        )
        outcomes = Counter(str(item["outcome"]) for item in applications)
        for item in applications:
            semantic_id = str(item["rule_semantic_id"])
            semantic_outcomes.setdefault(semantic_id, {}).setdefault(
                held_out, Counter())[str(item["outcome"])] += 1
        for rule in rules:
            semantic_rules[str(rule["rule_semantic_id"])] = {
                "context_signature": rule["context_signature"],
                "input_text": rule["input_text"],
                "official_source_text": rule["official_source_text"],
                "output_text": rule["output_text"],
                "rule_semantic_id": rule["rule_semantic_id"],
            }
        total.update(outcomes)
        directions.append({
            "application_count": len(applications),
            "candidate_rule_count": len(rules),
            "held_out_family": held_out,
            "outcomes": {
                name: outcomes[name]
                for name in ("EXACT", "UNKNOWN", "WRONG")},
        })
    survivors = []
    for semantic_id, rule in semantic_rules.items():
        by_family = semantic_outcomes.get(semantic_id, {})
        if all(
                family in by_family
                and by_family[family]["EXACT"] > 0
                and by_family[family]["WRONG"] == 0
                for family in V8_TRAIN_FAMILIES):
            covered = _predecessor_covers(rule, predecessor_rules)
            payload = {
                **rule,
                "family_outcomes": {
                    family: {
                        name: by_family[family][name]
                        for name in ("EXACT", "UNKNOWN", "WRONG")}
                    for family in V8_TRAIN_FAMILIES
                },
                "predecessor_covered": int(covered),
                "status": "THREE_DIRECTION_ZERO_WRONG_LOSO_SURVIVOR",
            }
            survivors.append({
                **payload,
                "survivor_record_sha256": _sha256(payload),
            })
    novel_survivor_count = sum(
        item["predecessor_covered"] == 0 for item in survivors)
    all_directions_exact = all(
        item["outcomes"]["EXACT"] > 0 for item in directions)
    if total["WRONG"] > 0:
        outcome = "FAIL_WRONG_NONZERO"
    elif not all_directions_exact:
        outcome = "NE_INCOMPLETE_FAMILY_COVERAGE"
    elif not survivors:
        outcome = "NE_NO_THREE_DIRECTION_SURVIVOR"
    elif novel_survivor_count == 0:
        outcome = "NE_PREDECESSOR_ONLY_SURVIVORS"
    else:
        outcome = "PASS_ZERO_WRONG_NOVEL_THREE_DIRECTION_SURVIVOR"
    payload = {
        "artifact_kind": NORMALIZATION_RECOVERY_V10_LOCAL_HYPOTHESIS_LOSO_KIND,
        "authorization_rule_count": 0,
        "direction_count": len(directions),
        "directions": directions,
        "formal_or_evaluation_payload_read_count": 0,
        "mastery_claimed": 0,
        "novel_survivor_count": novel_survivor_count,
        "outcome": outcome,
        "outcomes": {
            name: total[name] for name in ("EXACT", "UNKNOWN", "WRONG")},
        "production_enabled": 0,
        "predecessor_rule_count": len(predecessor_rules),
        "projection_group_count": len(groups),
        "status": NORMALIZATION_RECOVERY_V10_LOCAL_HYPOTHESIS_LOSO_STATUS,
        "survivor_count": len(survivors),
        "teacher_api_llm_call_count": 0,
    }
    return (
        {**payload, "loso_audit_sha256": _sha256(payload)},
        tuple(sorted(
            survivors, key=lambda item: str(item["rule_semantic_id"]))),
    )


__all__ = [
    "NORMALIZATION_RECOVERY_V10_LOCAL_HYPOTHESIS_LOSO_KIND",
    "NORMALIZATION_RECOVERY_V10_LOCAL_HYPOTHESIS_LOSO_STATUS",
    "derive_normalization_recovery_v10_local_hypothesis_loso",
]
