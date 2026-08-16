"""审计 recovery-v10 五 family 局部 hypothesis 的五方向 LOSO。"""
from __future__ import annotations

from collections import Counter
import hashlib

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
import pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v10_local_hypothesis_loso as _base
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v10_source_expansion_local_projection import (
    V10_SOURCE_EXPANSION_LOCAL_TRAIN_FAMILIES,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    canonical_json_line,
)


V10_SOURCE_EXPANSION_LOCAL_LOSO_KIND = (
    "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V10_FIVE_FAMILY_LOCAL_LOSO_V1")
V10_SOURCE_EXPANSION_LOCAL_LOSO_STATUS = (
    "FIVE_FAMILY_TRAIN_ONLY_AUTHORIZATION_IDENTIFIABILITY_AUDIT_NOT_FORMAL")


def _sha256(value: object) -> str:
    """返回rule、application、survivor或audit的规范SHA-256。"""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _source_input_sha256(source: str, input_text: str) -> str:
    """形成与五family collision ledger一致的完整query identity。"""
    return hashlib.sha256(canonical_json_line({
        "input_text": input_text,
        "official_source_text": source,
    })).hexdigest()


def _collision_ids(value: tuple[str, ...]) -> tuple[str, ...]:
    """核验固定、有序、唯一的collision source+input SHA集合。"""
    if (not isinstance(value, tuple) or value != tuple(sorted(set(value)))
            or any(_base._sha_value(item, label="collision id") != item
                   for item in value)):
        raise BroadQaExternalDataError(
            "v10 expanded local LOSO collision ledger 漂移")
    return value


def _observations(
        observations: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, object], ...]:
    """核验五family全量Observation identity、source与locale surface。"""
    if not isinstance(observations, tuple) or not observations:
        raise BroadQaExternalDataError(
            "v10 expanded local LOSO observations 为空")
    seen = set()
    values = []
    families = set()
    for item in observations:
        if not isinstance(item, dict):
            raise BroadQaExternalDataError(
                "v10 expanded local LOSO observation 非对象")
        identity = _base._sha_value(
            item.get("observation_id"), label="observation id")
        family = item.get("source_family")
        source = item.get("official_source_text")
        if (identity in seen
                or family not in V10_SOURCE_EXPANSION_LOCAL_TRAIN_FAMILIES
                or not isinstance(source, str) or not source):
            raise BroadQaExternalDataError(
                "v10 expanded local LOSO observation source 漂移")
        seen.add(identity)
        families.add(str(family))
        _base._surface(item, "zh_hant")
        _base._surface(item, "zh_hans")
        values.append(item)
    if families != set(V10_SOURCE_EXPANSION_LOCAL_TRAIN_FAMILIES):
        raise BroadQaExternalDataError(
            "v10 expanded local LOSO family denominator 漂移")
    return tuple(values)


def _projection_groups(
        projection_records: tuple[dict[str, object], ...],
        *,
        collision_ids: tuple[str, ...],
        ) -> tuple[
            dict[tuple[str, str, str], list[dict[str, object]]],
            dict[str, int],
        ]:
    """按source/input/context聚合，并在训练候选前执行冻结collision veto。"""
    if not isinstance(projection_records, tuple) or not projection_records:
        raise BroadQaExternalDataError(
            "v10 expanded local LOSO projection records 为空")
    collision_set = set(collision_ids)
    matched_collisions = set()
    vetoed_records = 0
    vetoed_hypotheses = 0
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    seen_observations = set()
    for record in projection_records:
        if not isinstance(record, dict):
            raise BroadQaExternalDataError(
                "v10 expanded local LOSO projection record 非对象")
        observation_id = _base._sha_value(
            record.get("observation_id"),
            label="projection observation id")
        family = record.get("source_family")
        source = record.get("official_source_text")
        input_surface = record.get("input_text")
        spans = record.get("span_hypotheses")
        if (observation_id in seen_observations
                or family not in V10_SOURCE_EXPANSION_LOCAL_TRAIN_FAMILIES
                or not isinstance(source, str) or not source
                or not isinstance(input_surface, str) or not input_surface
                or not isinstance(spans, list)):
            raise BroadQaExternalDataError(
                "v10 expanded local LOSO projection record 漂移")
        seen_observations.add(observation_id)
        source_input_id = _source_input_sha256(source, input_surface)
        if source_input_id in collision_set:
            matched_collisions.add(source_input_id)
            vetoed_records += 1
            vetoed_hypotheses += len(spans)
            continue
        for span in spans:
            if (not isinstance(span, dict)
                    or span.get("authorization_kind")
                    != "LOCAL_ORTHOGRAPHIC_HYPOTHESIS_ONLY"
                    or span.get("source_context_authorization_id") != ""
                    or span.get("authorized_official_source_text") != ""
                    or span.get("training_support_families") != [family]):
                raise BroadQaExternalDataError(
                    "v10 expanded local LOSO span state 漂移")
            start = span.get("input_start")
            end = span.get("input_end")
            left = span.get("input_text")
            right = span.get("output_text")
            if (type(start) is not int or type(end) is not int
                    or not 0 <= start < end <= len(input_surface)
                    or input_surface[start:end] != left
                    or not isinstance(right, str) or not right):
                raise BroadQaExternalDataError(
                    "v10 expanded local LOSO span geometry 漂移")
            context = _base._context(input_surface, start, end)
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
    if matched_collisions != collision_set:
        raise BroadQaExternalDataError(
            "v10 expanded local LOSO collision denominator 漂移")
    return groups, {
        "collision_ledger_count": len(collision_set),
        "collision_matched_count": len(matched_collisions),
        "collision_vetoed_hypothesis_count": vetoed_hypotheses,
        "collision_vetoed_projection_record_count": vetoed_records,
    }


def _direction_rules(
        groups: dict[tuple[str, str, str], list[dict[str, object]]],
        *,
        held_out_family: str,
        ) -> tuple[dict[str, object], ...]:
    """只用另外四家family形成unique-output source-context候选。"""
    expected_families = sorted(
        family for family in V10_SOURCE_EXPANSION_LOCAL_TRAIN_FAMILIES
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
                "v10 expanded local LOSO context group 漂移")
        semantic = {
            "context_signature": context,
            "input_text": input_text,
            "official_source_text": source,
            "output_text": outputs[0],
            "rule_kind": "FIVE_FAMILY_LOSO_SOURCE_CONTEXT_SPAN_HYPOTHESIS",
        }
        payload = {
            **semantic,
            "evidence_ids": sorted({
                evidence for item in training
                for evidence in item["evidence_ids"]}),
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


def derive_normalization_recovery_v10_source_expansion_local_loso(
        *,
        observations: tuple[dict[str, object], ...],
        projection_records: tuple[dict[str, object], ...],
        predecessor_source_rules: tuple[dict[str, object], ...],
        collision_source_input_sha256s: tuple[str, ...],
        ) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
    """运行五方向LOSO，并只列出各方向非零EXACT且零WRONG的survivor。"""
    checked_observations = _observations(observations)
    collisions = _collision_ids(collision_source_input_sha256s)
    groups, collision_summary = _projection_groups(
        projection_records, collision_ids=collisions)
    predecessor_rules = _base._predecessor_rules(predecessor_source_rules)
    directions = []
    total = Counter()
    semantic_outcomes: dict[str, dict[str, Counter]] = {}
    semantic_rules: dict[str, dict[str, object]] = {}
    for held_out in V10_SOURCE_EXPANSION_LOCAL_TRAIN_FAMILIES:
        rules = _direction_rules(groups, held_out_family=held_out)
        applications = _base._held_out_applications(
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
                for family in V10_SOURCE_EXPANSION_LOCAL_TRAIN_FAMILIES):
            covered = _base._predecessor_covers(rule, predecessor_rules)
            payload = {
                **rule,
                "family_outcomes": {
                    family: {
                        name: by_family[family][name]
                        for name in ("EXACT", "UNKNOWN", "WRONG")}
                    for family in V10_SOURCE_EXPANSION_LOCAL_TRAIN_FAMILIES
                },
                "predecessor_covered": int(covered),
                "status": "FIVE_DIRECTION_ZERO_WRONG_LOSO_SURVIVOR",
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
        outcome = "NE_NO_FIVE_DIRECTION_SURVIVOR"
    elif novel_survivor_count == 0:
        outcome = "NE_PREDECESSOR_ONLY_SURVIVORS"
    else:
        outcome = "PASS_ZERO_WRONG_NOVEL_FIVE_DIRECTION_SURVIVOR"
    payload = {
        "artifact_kind": V10_SOURCE_EXPANSION_LOCAL_LOSO_KIND,
        "authorization_rule_count": 0,
        **collision_summary,
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
        "status": V10_SOURCE_EXPANSION_LOCAL_LOSO_STATUS,
        "survivor_count": len(survivors),
        "teacher_api_llm_call_count": 0,
    }
    return (
        {**payload, "loso_audit_sha256": _sha256(payload)},
        tuple(sorted(
            survivors, key=lambda item: str(item["rule_semantic_id"]))),
    )


__all__ = [
    "derive_normalization_recovery_v10_source_expansion_local_loso",
]
