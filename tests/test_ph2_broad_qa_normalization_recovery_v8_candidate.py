"""Recovery-v8 commitment、候选重建与整串解释器测试。"""
from __future__ import annotations

import hashlib

from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_recovery_v8_candidate as candidate_module,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_candidate import (
    V8_CANDIDATE_RULE_COUNTS,
    compile_normalization_recovery_v8_candidate,
    derive_normalization_recovery_v8_candidate_preflight,
    execute_normalization_recovery_v8_candidate_batch,
    reference_normalization_recovery_v8_candidate_batch,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_evaluation_commitment import (
    NORMALIZATION_RECOVERY_V8_DIMENSION_ORDER,
    build_normalization_recovery_v8_evaluation_commitment,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v8_training_records import (
    V8_TRAIN_FAMILIES,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _v7_commitment() -> dict[str, object]:
    return {
        "artifact_kind": (
            "PH2_BROAD_QA_NORMALIZATION_RECOVERY_V7_EVALUATION_COMMITMENT_V1"),
        "denominator": {
            "aggregate_buckets": {
                "equal_length_count": 1_967,
                "identity_count": 337,
                "nonidentity_count": 3_319,
                "single_han_difference_count": 350,
                "structure_equal_count": 3_652,
                "variable_length_count": 1_689,
            },
            "label_blind": 1,
            "record_count": 3_656,
        },
        "manifest_sha256": _sha("v7 commitment"),
        "mastery_claimed": 0,
        "production_enabled": 0,
        "source_exclusion": {
            "excluded_source_pack_manifest_sha256": _sha("VLC source"),
        },
        "status": (
            "LABEL_BLIND_DENOMINATOR_AND_GATES_FROZEN_BEFORE_V7_LEARNER_CHANGE"),
    }


def _directions(
        *, kind: str, identity: str, semantic: dict[str, object],
        support: tuple[str, ...],
        ) -> list[dict[str, object]]:
    values = []
    for held_out in V8_TRAIN_FAMILIES:
        after = [family for family in support if family != held_out]
        if len(after) < 2:
            continue
        values.append({
            **semantic,
            "authorization_id": identity,
            "candidate_id": _sha(identity + " candidate"),
            "held_out_family": held_out,
            "production_enabled": 0,
            "train_support_families": after,
        })
    return values


def _rule_outputs() -> dict[str, tuple[dict[str, object], ...]]:
    families = tuple(V8_TRAIN_FAMILIES)
    orthographic = []
    for index in range(V8_CANDIDATE_RULE_COUNTS["orthographic_rules"]):
        support = families if index == 0 else families[:2]
        orthographic.extend(_directions(
            kind="ORTHOGRAPHIC_ATOM", identity=_sha(f"ortho {index}"),
            semantic={"input_atom": chr(0x4E00 + index),
                      "output_atom": chr(0x5000 + index)}, support=support))
    lexical = []
    for index in range(V8_CANDIDATE_RULE_COUNTS["source_conditioned_rules"]):
        lexical.extend(_directions(
            kind="SOURCE_CONDITIONED_LEXICAL_ATOM",
            identity=_sha(f"lexical {index}"),
            semantic={"input_text": f"輸入{index}",
                      "official_source_text": f"Source {index}",
                      "output_text": f"输出{index}"}, support=families[:2]))
    structure = []
    for index in range(V8_CANDIDATE_RULE_COUNTS["structure_obligations"]):
        structure.extend(_directions(
            kind="LAYOUT_MORPHOLOGY_OBLIGATION",
            identity=_sha(f"structure {index}"),
            semantic={"structure_tokens": [f"%{index + 1}"]},
            support=families[:2]))
    identities = []
    for index in range(V8_CANDIDATE_RULE_COUNTS["identity_veto_rules"]):
        candidate_id = _sha(f"identity {index}")
        for held_out in (families[2],):
            identities.append({
                "candidate_id": candidate_id,
                "held_out_family": held_out,
                "input_text": f"稳定{index}",
                "output_text": f"稳定{index}",
                "production_enabled": 0,
                "train_support_families": list(families[:2]),
            })
    return {
        "orthographic-rules.jsonl": tuple(orthographic),
        "source-conditioned-lexical-rules.jsonl": tuple(lexical),
        "layout-morphology-obligations.jsonl": tuple(structure),
        "identity-veto-rules.jsonl": tuple(identities),
    }


def _candidate() -> dict[str, object]:
    return compile_normalization_recovery_v8_candidate(
        rule_pack_manifest={
            "manifest_sha256": _sha("pack"),
            "mastery_claimed": 0,
            "production_enabled": 0,
            "status": "FAMILY_LOSO_FROZEN_NOT_FORMAL_NOT_DEPLOYED",
        },
        rule_outputs=_rule_outputs(),
        training_audit_manifest_sha256=_sha("audit"),
        evaluation_commitment_manifest_sha256=_sha("commitment"),
    )


def test_commitment_freezes_full_denominator_and_six_dimensions() -> None:
    commitment = build_normalization_recovery_v8_evaluation_commitment(
        _v7_commitment())
    assert commitment["denominator"]["record_count"] == 3_656
    assert tuple(commitment["dimensions"]) == NORMALIZATION_RECOVERY_V8_DIMENSION_ORDER
    assert commitment["judgements"] == ["EXACT", "UNKNOWN", "WRONG"]
    assert commitment["vlc_identity_raw_or_translation_read_count"] == 0
    assert commitment["formal_contract"][
        "retry_after_any_terminal_or_post_guard_exception_allowed"] == 0


def test_candidate_deduplicates_directions_and_composes_full_text() -> None:
    candidate = _candidate()
    inventories = candidate["inventories"]
    assert {name: len(records) for name, records in inventories.items()} == (
        V8_CANDIDATE_RULE_COUNTS)
    first_atom = inventories["orthographic_rules"][0]
    second_atom = inventories["orthographic_rules"][1]
    text = first_atom["input_atom"] + second_atom["input_atom"]
    queries = ({
        "input_text": text,
        "official_source_text": "No lexical match",
        "structure_tokens": [],
    }, {
        "input_text": inventories["source_conditioned_rules"][0]["input_text"],
        "official_source_text": inventories[
            "source_conditioned_rules"][0]["official_source_text"],
        "structure_tokens": [],
    }, {
        "input_text": inventories["identity_veto_rules"][0]["input_text"],
        "official_source_text": "Identity",
        "structure_tokens": [],
    }, {
        "input_text": "未授权文本",
        "official_source_text": "Unknown",
        "structure_tokens": [],
    })
    indexed = execute_normalization_recovery_v8_candidate_batch(
        candidate, queries)
    reference = reference_normalization_recovery_v8_candidate_batch(
        candidate, queries)
    assert indexed == reference
    assert indexed[0]["output_text"] == (
        first_atom["output_atom"] + second_atom["output_atom"])
    assert [item["route_kind"] for item in indexed] == [
        "ORTHOGRAPHIC_ATOM", "SOURCE_CONDITIONED_LEXICAL_ATOM",
        "IDENTITY_VETO", "UNKNOWN"]
    assert indexed[-1]["behavior"] == "UNKNOWN"
    assert indexed[-1]["output_text"] == ""


def test_candidate_preflight_covers_all_rules_without_mismatch() -> None:
    preflight = derive_normalization_recovery_v8_candidate_preflight(_candidate())
    assert preflight["case_count"] == 131
    assert preflight["failure_count"] == 0
    assert preflight["indexed_reference_mismatch_count"] == 0
    assert preflight["unknown_case_count"] == 1


def test_each_batch_validates_candidate_once(
        monkeypatch,
        ) -> None:
    candidate = _candidate()
    queries = tuple({
        "input_text": f"未授权文本{index}",
        "official_source_text": f"Unknown {index}",
        "structure_tokens": [],
    } for index in range(16))
    original = candidate_module._validate_candidate
    calls = []

    def counted(value):
        calls.append(1)
        return original(value)

    monkeypatch.setattr(candidate_module, "_validate_candidate", counted)
    execute_normalization_recovery_v8_candidate_batch(candidate, queries)
    assert len(calls) == 1
    calls.clear()
    reference_normalization_recovery_v8_candidate_batch(candidate, queries)
    assert len(calls) == 1
