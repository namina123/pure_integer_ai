"""Recovery-v6 strong-whole policy projection 测试。"""
from __future__ import annotations

import copy
import hashlib

import pytest

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_learning_records import (
    derive_normalization_recovery_v5_learning_outputs,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v6_learning_contract import (
    NORMALIZATION_RECOVERY_V6_DEFER_REASONS,
    NORMALIZATION_RECOVERY_V6_OUTPUT_FILE_ROLES,
    RECOVERY_V6_TARGET_POLICY_SCOPE,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v6_learning_records import (
    derive_normalization_recovery_v6_learning_outputs,
    normalization_recovery_v6_output_payloads,
)
from test_ph2_broad_qa_normalization_recovery_v5_learner import _material


def _sha(value: str) -> str:
    """返回 synthetic manifest identity。"""
    return hashlib.sha256(value.encode()).hexdigest()


def _projection_material():
    """构造四来源 v5 outputs 并执行 v6 projection。"""
    protocol_sha = _sha("v6-projection-protocol")
    material = _material(protocol_sha)
    predecessor, _summary, _emissions = (
        derive_normalization_recovery_v5_learning_outputs(
            protocol_manifest=material[0],
            observations=material[1],
            fragments=material[2],
            groups=material[3],
            work=material[4],
        ))
    pack_sha = _sha("v6-projection-predecessor-pack")
    outputs, summary = derive_normalization_recovery_v6_learning_outputs(
        protocol_manifest_sha256=protocol_sha,
        predecessor_pack_manifest_sha256=pack_sha,
        predecessor_outputs=predecessor,
    )
    return material, predecessor, pack_sha, outputs, summary


def test_v6_projection_approves_only_three_family_whole_without_veto() -> None:
    """approved predecessor 集必须精确等于 frozen strong-whole predicate。"""
    _material_values, predecessor, _pack_sha, outputs, summary = (
        _projection_material())
    identity_inputs = {
        str(item["input_text"])
        for item in predecessor["identity-observations.jsonl"]}
    conflict_inputs = {
        str(item["input_text"])
        for item in predecessor["conflict-ledger.jsonl"]
        if item["candidate_scope_kind"] in {"NONE", "TARGET_CROSS_FAMILY"}}
    expected = {
        str(item["rule_id"])
        for item in predecessor["target-phrase-rules.jsonl"]
        if item["fragment_kind"] == "WHOLE_INPUT"
        and len(item["source_families"]) >= 3
        and item["input_text"] not in identity_inputs
        and item["input_text"] not in conflict_inputs}
    approved = outputs["target-whole-rules.jsonl"]
    assert {str(item["predecessor_rule_id"]) for item in approved} == expected
    assert approved
    assert all(item["target_policy_scope"] == RECOVERY_V6_TARGET_POLICY_SCOPE
               and item["observed_distinct_source_family_count"] >= 3
               and item["application_scope"]["local_execution_allowed"] == 0
               and item["application_scope"]["source_execution_allowed"] == 0
               for item in approved)
    assert not ({str(item["input_text"]) for item in approved}
                & (identity_inputs | conflict_inputs))
    assert summary["executable_local_rule_count"] == 0
    assert summary["executable_source_rule_count"] == 0


def test_v6_projection_partitions_every_predecessor_and_keeps_vetoes() -> None:
    """全部 v5 executable rule 必须 approved/deferred 二分且 veto 一等保留。"""
    _material_values, predecessor, _pack_sha, outputs, summary = (
        _projection_material())
    predecessor_ids = {
        str(item["rule_id"])
        for name in ("target-phrase-rules.jsonl", "source-phrase-rules.jsonl")
        for item in predecessor[name]}
    approved_ids = {
        str(item["predecessor_rule_id"])
        for item in outputs["target-whole-rules.jsonl"]}
    deferred_ids = {
        str(item["predecessor_rule_id"])
        for item in outputs["deferred-rules.jsonl"]}
    assert not approved_ids & deferred_ids
    assert approved_ids | deferred_ids == predecessor_ids
    assert all(item["defer_reasons"]
               and set(item["defer_reasons"]).issubset(
                   NORMALIZATION_RECOVERY_V6_DEFER_REASONS)
               for item in outputs["deferred-rules.jsonl"])
    assert summary["approved_target_rule_count"] == len(approved_ids)
    assert summary["deferred_rule_count"] == len(deferred_ids)
    assert summary["identity_veto_count"] == len(
        outputs["identity-vetoes.jsonl"])
    assert summary["conflict_veto_count"] == len(
        outputs["conflict-vetoes.jsonl"])
    rule_ids = {str(item["rule_id"])
                for item in outputs["target-whole-rules.jsonl"]}
    assert {str(item["rule_id"]) for item in outputs["target-index.jsonl"]} == (
        rule_ids)
    assert {str(item["rule_id"]) for item in outputs["defeaters.jsonl"]}.issubset(
        rule_ids)


def test_v6_projection_is_byte_stable_and_rejects_predecessor_drift() -> None:
    """相同 TRAIN outputs 必须字节相等，v5 index/rule 漂移必须拒绝。"""
    material, predecessor, pack_sha, outputs, summary = _projection_material()
    repeated, repeated_summary = derive_normalization_recovery_v6_learning_outputs(
        protocol_manifest_sha256=str(material[0]["manifest_sha256"]),
        predecessor_pack_manifest_sha256=pack_sha,
        predecessor_outputs=predecessor,
    )
    assert repeated == outputs
    assert repeated_summary == summary
    assert normalization_recovery_v6_output_payloads(repeated) == (
        normalization_recovery_v6_output_payloads(outputs))
    assert set(outputs) == {
        name for name, _role, _identity
        in NORMALIZATION_RECOVERY_V6_OUTPUT_FILE_ROLES}

    tampered = copy.deepcopy(predecessor)
    target_index = list(tampered["target-overlap-index.jsonl"])
    target_index[0] = {**target_index[0], "priority_contract": "BROKEN"}
    tampered["target-overlap-index.jsonl"] = tuple(target_index)
    with pytest.raises(BroadQaExternalDataError, match="overlap index"):
        derive_normalization_recovery_v6_learning_outputs(
            protocol_manifest_sha256=str(material[0]["manifest_sha256"]),
            predecessor_pack_manifest_sha256=pack_sha,
            predecessor_outputs=tampered,
        )
