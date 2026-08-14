"""派生 normalization recovery 的 TRAIN-only LOSO 与库存摘要。

本模块只消费已经物化并通过 records 层 schema 校验的 TRAIN 记录。它不读路径、
不写 artifact，也不接触 Firefox evaluation、reserve、formal 或 learner 输出。
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_training_records import (
    COMPOSITION_QUALIFICATIONS,
    GENERIC_RESOLUTION_KINDS,
    NORMALIZATION_RECOVERY_LOSO_KIND,
    SOURCE_POLICY_SCOPES,
    TARGET_RESOLUTION_KINDS,
    resolve_normalization_recovery_group_authority,
    validate_normalization_recovery_observation_inventory,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes


LOSO_OUTCOMES = ("EXACT", "UNKNOWN", "WRONG")


def _sha256(payload: bytes) -> str:
    """返回 TRAIN-only audit identity 的 SHA-256。"""
    return hashlib.sha256(payload).hexdigest()


def derive_normalization_recovery_loso(
        *,
        roster: tuple[dict[str, object], ...],
        observations: tuple[dict[str, object], ...],
        ) -> tuple[dict[str, object], ...]:
    """用其余 policy 预测留出 policy，仅形成 TRAIN-only 可行性审计。"""
    validate_normalization_recovery_observation_inventory(
        roster=roster, observations=observations)
    values = []
    for held_out_policy in SOURCE_POLICY_SCOPES:
        training = tuple(item for item in observations
                         if item["source_policy_scope"] != held_out_policy)
        held_out = tuple(item for item in observations
                         if item["source_policy_scope"] == held_out_policy)
        by_input: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
        for item in training:
            by_input[str(item["input_text"])].append(item)
        target_by_input = {}
        character_map = {}
        for input_text, grouped in by_input.items():
            ordered = sorted(grouped, key=lambda item: (
                str(item["source_policy_scope"]),
                str(item["observation_id"])))
            resolution = resolve_normalization_recovery_group_authority(ordered)
            target_kind = str(resolution["target_resolution_kind"])
            target_output = str(resolution["target_output"])
            target_by_input[input_text] = (target_kind, target_output)
            if len(input_text) == 1 and target_kind != "NO_TARGET_AUTHORITY":
                character_map[input_text] = target_output
        for observation in held_out:
            input_text = str(observation["input_text"])
            expected = str(observation["expected_output"])
            target = target_by_input.get(input_text)
            if target is not None and target[0] != "NO_TARGET_AUTHORITY":
                predicted = target[1]
                prediction_route = "EXACT_INPUT_GROUP"
            elif observation["mapping_kind"] == "PHRASE_INPUT":
                output = []
                covered = []
                for offset, character in enumerate(input_text):
                    mapped = character_map.get(character)
                    if mapped is None:
                        output.append(character)
                    else:
                        output.append(mapped)
                        covered.append(offset)
                predicted = "".join(output)
                if covered:
                    prediction_route = "CHARACTER_COMPOSITION"
                else:
                    predicted = ""
                    prediction_route = "UNKNOWN"
            else:
                predicted = ""
                prediction_route = "UNKNOWN"
            if not predicted:
                outcome = "UNKNOWN"
            elif predicted == expected:
                outcome = "EXACT"
            else:
                outcome = "WRONG"
            identity = {
                "held_out_observation_id": observation["observation_id"],
                "held_out_source_policy_scope": held_out_policy,
                "prediction_route": prediction_route,
                "predicted_output": predicted,
            }
            values.append({
                **identity,
                "expected_output": expected,
                "format_version": 2,
                "input_text": input_text,
                "loso_id": _sha256(canonical_json_bytes(identity)),
                "mapping_kind": observation["mapping_kind"],
                "outcome": outcome,
                "record_kind": NORMALIZATION_RECOVERY_LOSO_KIND,
                "split": "TRAIN_SOURCE_AUDIT",
            })
    result = tuple(sorted(values, key=lambda item: str(item["loso_id"])))
    if (not result or len({item["loso_id"] for item in result}) != len(result)):
        raise BroadQaExternalDataError("recovery LOSO identity 漂移")
    return result


def normalization_recovery_training_summary(
        *,
        roster: tuple[dict[str, object], ...],
        observations: tuple[dict[str, object], ...],
        groups: tuple[dict[str, object], ...],
        compositions: tuple[dict[str, object], ...],
        loso: tuple[dict[str, object], ...],
        ) -> dict[str, object]:
    """汇总 source/family/policy、resolution、composition 与 LOSO 库存。"""
    policy_counts = Counter(str(item["source_policy_scope"])
                            for item in observations)
    source_counts = Counter(str(item["source_key"]) for item in observations)
    generic_counts = Counter(str(item["generic_resolution_kind"])
                             for item in groups)
    target_counts = Counter(str(item["target_resolution_kind"])
                            for item in groups)
    composition_counts = Counter(str(item["qualification_kind"])
                                 for item in compositions)
    loso_counts = Counter((str(item["held_out_source_policy_scope"]),
                           str(item["mapping_kind"]), str(item["outcome"]))
                          for item in loso)
    return {
        "composition_count": len(compositions),
        "composition_qualification_counts": {
            key: composition_counts[key] for key in COMPOSITION_QUALIFICATIONS},
        "generic_resolution_counts": {
            key: generic_counts[key] for key in GENERIC_RESOLUTION_KINDS},
        "group_count": len(groups),
        "loso_count": len(loso),
        "loso_counts": {
            f"{policy}:{mapping}:{outcome}": loso_counts[(policy, mapping, outcome)]
            for policy in SOURCE_POLICY_SCOPES
            for mapping in ("CHARACTER_INPUT", "PHRASE_INPUT")
            for outcome in LOSO_OUTCOMES
        },
        "observation_count": len(observations),
        "roster_policy_count": len(roster),
        "source_family_count": len({item["source_family"] for item in roster}),
        "source_observation_counts": {
            key: source_counts[key] for key in sorted(source_counts)},
        "source_policy_observation_counts": {
            key: policy_counts[key] for key in SOURCE_POLICY_SCOPES},
        "target_resolution_counts": {
            key: target_counts[key] for key in TARGET_RESOLUTION_KINDS},
        "target_rule_character_count": sum(
            item["mapping_kind"] == "CHARACTER_INPUT"
            and item["target_resolution_kind"] != "NO_TARGET_AUTHORITY"
            and item["target_rule_is_identity"] == 0 for item in groups),
        "target_rule_identity_count": sum(
            item["target_rule_is_identity"] == 1 for item in groups),
        "target_rule_phrase_count": sum(
            item["mapping_kind"] == "PHRASE_INPUT"
            and item["target_resolution_kind"] != "NO_TARGET_AUTHORITY"
            and item["target_rule_is_identity"] == 0 for item in groups),
    }


__all__ = [
    "LOSO_OUTCOMES",
    "derive_normalization_recovery_loso",
    "normalization_recovery_training_summary",
]
