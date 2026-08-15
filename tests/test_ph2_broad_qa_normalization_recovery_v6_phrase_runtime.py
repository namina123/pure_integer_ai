"""Recovery-v6 strong-whole disabled runtime 测试。"""
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
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_phrase_runtime import (
    compile_normalization_recovery_v5_phrase_program,
    execute_normalization_recovery_v5_phrase_program,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_successor_simulation_records import (
    simulate_normalization_recovery_v5_successor_strategy,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v6_learning_records import (
    derive_normalization_recovery_v6_learning_outputs,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v6_phrase_runtime import (
    compile_normalization_recovery_v6_phrase_program,
    execute_normalization_recovery_v6_phrase_batch,
    execute_normalization_recovery_v6_phrase_program,
    reference_normalization_recovery_v6_phrase_batch,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from test_ph2_broad_qa_normalization_recovery_v5_learner import _material


def _sha(value: str) -> str:
    """返回 synthetic manifest identity。"""
    return hashlib.sha256(value.encode()).hexdigest()


def _program_material():
    """构造 v5 predecessor、v6 projection 与两个 disabled program。"""
    protocol_sha = _sha("v6-runtime-protocol")
    material = _material(protocol_sha)
    predecessor, _summary, _emissions = (
        derive_normalization_recovery_v5_learning_outputs(
            protocol_manifest=material[0],
            observations=material[1],
            fragments=material[2],
            groups=material[3],
            work=material[4],
        ))
    predecessor_pack_sha = _sha("v6-runtime-predecessor-pack")
    v5_program = compile_normalization_recovery_v5_phrase_program(
        rule_pack_manifest_sha256=predecessor_pack_sha,
        target_phrase_rules=predecessor["target-phrase-rules.jsonl"],
        source_phrase_rules=predecessor["source-phrase-rules.jsonl"],
        defeaters=predecessor["defeaters.jsonl"],
        target_overlap_index=predecessor["target-overlap-index.jsonl"],
        source_overlap_index=predecessor["source-overlap-index.jsonl"],
    )
    outputs, projection_summary = derive_normalization_recovery_v6_learning_outputs(
        protocol_manifest_sha256=protocol_sha,
        predecessor_pack_manifest_sha256=predecessor_pack_sha,
        predecessor_outputs=predecessor,
    )
    program = compile_normalization_recovery_v6_phrase_program(
        rule_pack_manifest_sha256=_sha("v6-runtime-pack"),
        target_whole_rules=outputs["target-whole-rules.jsonl"],
        defeaters=outputs["defeaters.jsonl"],
        identity_vetoes=outputs["identity-vetoes.jsonl"],
        conflict_vetoes=outputs["conflict-vetoes.jsonl"],
        target_index=outputs["target-index.jsonl"],
    )
    return material, predecessor, v5_program, outputs, projection_summary, program


def _rehash(program: dict[str, object]) -> None:
    """在篡改测试中只重算 program 外层自摘要。"""
    payload = {key: value for key, value in program.items()
               if key != "program_sha256"}
    program["program_sha256"] = hashlib.sha256(
        canonical_json_bytes(payload)).hexdigest()


def test_v6_runtime_matches_whole_strong_simulation_for_all_train_inputs() -> None:
    """独立 v6 runtime 必须逐输入等于诊断阶段 WHOLE_STRONG 策略。"""
    material, predecessor, v5_program, _outputs, _summary, program = (
        _program_material())
    for observation in material[1]:
        baseline = execute_normalization_recovery_v5_phrase_program(
            v5_program,
            str(observation["input_text"]),
            source_family=str(observation["source_family"]),
            structure_tokens=tuple(observation["structure_tokens"]),
        )
        simulated = simulate_normalization_recovery_v5_successor_strategy(
            strategy="WHOLE_STRONG",
            observation=observation,
            baseline_result=baseline,
            program=v5_program,
            outputs=predecessor,
            training_groups=material[3],
            held_out_source_family=str(observation["source_family"]),
        )
        actual = execute_normalization_recovery_v6_phrase_program(
            program,
            str(observation["input_text"]),
            structure_tokens=tuple(observation["structure_tokens"]),
        )
        assert actual["output_text"] == simulated["output_text"]


def test_v6_runtime_is_whole_only_vetoed_and_reference_equivalent() -> None:
    """approved whole 可执行；identity/conflict/local 均回退且双解释器相等。"""
    material, predecessor, _v5, outputs, summary, program = _program_material()
    assert summary["approved_target_rule_count"] > 0
    approved = outputs["target-whole-rules.jsonl"][0]
    tokens = tuple(approved["structure_token_variants"][0])
    exact = execute_normalization_recovery_v6_phrase_program(
        program, str(approved["input_text"]), structure_tokens=tokens)
    assert exact["output_text"] == approved["output_text"]
    assert exact["steps"][0]["mode"] == "WHOLE_INPUT_EXACT"

    identity = outputs["identity-vetoes.jsonl"][0]
    identity_result = execute_normalization_recovery_v6_phrase_program(
        program, str(identity["input_text"]))
    assert identity_result["output_text"] == identity["input_text"]
    assert "IDENTITY_VETO" in identity_result["decision_reasons"]

    local = next(item for item in predecessor["target-phrase-rules.jsonl"]
                 if item["fragment_kind"] != "WHOLE_INPUT")
    local_result = execute_normalization_recovery_v6_phrase_program(
        program, str(local["input_text"]))
    assert local_result["output_text"] == local["input_text"]
    assert local_result["steps"][0]["mode"] == "IDENTITY"

    texts = tuple(str(item["input_text"]) for item in material[1])
    structures = tuple(tuple(item["structure_tokens"]) for item in material[1])
    assert execute_normalization_recovery_v6_phrase_batch(
        program, texts, structure_tokens=structures) == (
            reference_normalization_recovery_v6_phrase_batch(
                program, texts, structure_tokens=structures))


def test_v6_runtime_rejects_rehashed_rule_veto_and_index_tamper() -> None:
    """外层 program SHA 重算也不能掩盖 rule/veto/index 语义篡改。"""
    _material_values, _predecessor, _v5, outputs, _summary, program = (
        _program_material())
    rule_tampered = copy.deepcopy(program)
    rule_tampered["target_buckets"][0]["rules"][0]["output_text"] += "X"
    _rehash(rule_tampered)
    with pytest.raises(BroadQaExternalDataError, match="rule"):
        execute_normalization_recovery_v6_phrase_program(
            rule_tampered,
            str(rule_tampered["target_buckets"][0]["rules"][0]["input_text"]),
        )

    veto_tampered = copy.deepcopy(program)
    veto_tampered["identity_vetoes"][0]["input_text"] += "X"
    _rehash(veto_tampered)
    with pytest.raises(BroadQaExternalDataError, match="veto"):
        execute_normalization_recovery_v6_phrase_program(
            veto_tampered, "probe")

    bad_index = list(outputs["target-index.jsonl"])
    bad_index[0] = {**bad_index[0], "candidate_input": "BROKEN"}
    with pytest.raises(BroadQaExternalDataError, match="index"):
        compile_normalization_recovery_v6_phrase_program(
            rule_pack_manifest_sha256=_sha("v6-runtime-pack"),
            target_whole_rules=outputs["target-whole-rules.jsonl"],
            defeaters=outputs["defeaters.jsonl"],
            identity_vetoes=outputs["identity-vetoes.jsonl"],
            conflict_vetoes=outputs["conflict-vetoes.jsonl"],
            target_index=tuple(bad_index),
        )
