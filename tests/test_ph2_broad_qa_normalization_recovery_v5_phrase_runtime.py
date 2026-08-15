"""Recovery-v5 disabled phrase runtime 的 scope、结构与双解释器测试。"""
from __future__ import annotations

import copy
import hashlib

import pytest

from pure_integer_ai.experiments.ph2_broad_qa_external_data import (
    BroadQaExternalDataError,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_phrase_runtime import (
    compile_normalization_recovery_v5_phrase_program,
    execute_normalization_recovery_v5_phrase_batch,
    execute_normalization_recovery_v5_phrase_program,
    normalization_recovery_v5_defeater_matches,
    reference_normalization_recovery_v5_phrase_batch,
    reference_normalization_recovery_v5_phrase_program,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_training_records import (
    RECOVERY_V5_TARGET_POLICY_SCOPE,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_recovery_v5_learning_records import (
    derive_normalization_recovery_v5_learning_outputs,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_bytes
from test_ph2_broad_qa_normalization_recovery_v5_learner import _material


def _sha(value: str) -> str:
    """返回测试 pack identity。"""
    return hashlib.sha256(value.encode()).hexdigest()


def _rehash(program: dict[str, object]) -> None:
    """在篡改测试中重算 program 自摘要。"""
    payload = {key: value for key, value in program.items()
               if key != "program_sha256"}
    program["program_sha256"] = hashlib.sha256(
        canonical_json_bytes(payload)).hexdigest()


def _program():
    """从 synthetic v5 TRAIN outputs 编译 runtime program。"""
    protocol_sha = _sha("v5-runtime-protocol")
    manifest, observations, fragments, groups, work = _material(protocol_sha)
    outputs, _summary, _emissions = derive_normalization_recovery_v5_learning_outputs(
        protocol_manifest=manifest,
        observations=observations,
        fragments=fragments,
        groups=groups,
        work=work,
    )
    return outputs, compile_normalization_recovery_v5_phrase_program(
        rule_pack_manifest_sha256=_sha("v5-runtime-pack"),
        target_phrase_rules=outputs["target-phrase-rules.jsonl"],
        source_phrase_rules=outputs["source-phrase-rules.jsonl"],
        defeaters=outputs["defeaters.jsonl"],
        target_overlap_index=outputs["target-overlap-index.jsonl"],
        source_overlap_index=outputs["source-overlap-index.jsonl"],
    )


def test_v5_whole_input_structure_scope_and_reference_equivalence() -> None:
    """whole exact、结构 token、source scope 与 indexed/reference 必须一致。"""
    outputs, program = _program()
    equal = execute_normalization_recovery_v5_phrase_program(
        program, "開啟檔案")
    assert equal["output_text"] == "打开文件"
    variable = execute_normalization_recovery_v5_phrase_program(
        program, "建立新檔案")
    assert variable["output_text"] == "新建文件"
    structured = execute_normalization_recovery_v5_phrase_program(
        program,
        "<b>開啟</b>",
        structure_tokens=("HTML_START:b", "HTML_END:b"),
    )
    assert structured["output_text"] == "<b>打开</b>"
    assert execute_normalization_recovery_v5_phrase_program(
        program, "<b>開啟</b>")["output_text"] == "<b>開啟</b>"
    assert execute_normalization_recovery_v5_phrase_program(
        program,
        "<b>開啟</b>",
        structure_tokens=("HTML_START:i", "HTML_END:i"),
    )["output_text"] == "<b>開啟</b>"
    source = execute_normalization_recovery_v5_phrase_program(
        program, "偏好設定", source_family="MICROSOFT_VSCODE_PROJECT")
    assert source["output_text"] == "设置"
    assert execute_normalization_recovery_v5_phrase_program(
        program, "偏好設定")["output_text"] == "偏好設定"
    texts = ("開啟檔案", "建立新檔案", "<b>開啟</b>")
    tokens = ((), (), ("HTML_START:b", "HTML_END:b"))
    indexed = execute_normalization_recovery_v5_phrase_batch(
        program, texts, structure_tokens=tokens)
    reference = reference_normalization_recovery_v5_phrase_batch(
        program, texts, structure_tokens=tokens)
    assert indexed == reference
    assert all(item["target_policy_scope"] == RECOVERY_V5_TARGET_POLICY_SCOPE
               for item in indexed)
    assert outputs["identity-observations.jsonl"]


def test_v5_local_rule_and_defeater_are_fail_closed() -> None:
    """局部规则只在无结构输入执行，defeater 命中时回退而非强行改写。"""
    outputs, program = _program()
    local = next(item for item in outputs["target-phrase-rules.jsonl"]
                 if item["fragment_kind"] in {"EDIT_CORE", "CONTEXT_HUNK"})
    local_result = execute_normalization_recovery_v5_phrase_program(
        program, str(local["input_text"]))
    assert local_result["output_text"] == local["output_text"]
    structured_local = execute_normalization_recovery_v5_phrase_program(
        program,
        str(local["input_text"]),
        structure_tokens=("UNEXPECTED",),
    )
    assert structured_local["output_text"] == local["input_text"]
    refute = next(item for item in outputs["evidence.jsonl"]
                  if item["stance"] == "REFUTE"
                  and item["candidate_scope_kind"] == "TARGET_CROSS_FAMILY")
    observation = next(item for item in _material(_sha("v5-runtime-protocol"))[1]
                       if item["observation_id"] == refute["observation_id"])
    defeater = next(item for item in outputs["defeaters.jsonl"]
                    if refute["evidence_id"] in item["refute_evidence_ids"])
    assert normalization_recovery_v5_defeater_matches(
        defeater,
        str(observation["input_text"]),
        int(refute["occurrence_start"]),
        int(refute["occurrence_end"]),
    )
    result = execute_normalization_recovery_v5_phrase_program(
        program,
        str(observation["input_text"]),
        structure_tokens=tuple(observation["structure_tokens"]),
    )
    assert result["output_text"]


def test_v5_runtime_rejects_program_scope_or_index_tamper() -> None:
    """program 摘要、overlap index 与 rule scope 任一漂移都必须拒绝。"""
    outputs, program = _program()
    tampered = {**program, "whole_input_exact_precedes_local": 0}
    with pytest.raises(BroadQaExternalDataError, match="program identity"):
        execute_normalization_recovery_v5_phrase_program(
            tampered, "開啟檔案")
    scope_tampered = copy.deepcopy(program)
    scope_rule = scope_tampered["target_buckets"][0]["rules"][0]
    scope_rule["target_policy_scope"] = "BROKEN_TARGET_SCOPE"
    _rehash(scope_tampered)
    with pytest.raises(BroadQaExternalDataError, match="rule"):
        execute_normalization_recovery_v5_phrase_program(
            scope_tampered, "開啟檔案")
    defeater_tampered = copy.deepcopy(program)
    defeater_tampered["defeaters"][0]["action"] = "BROKEN_ACTION"
    _rehash(defeater_tampered)
    with pytest.raises(BroadQaExternalDataError, match="defeater"):
        execute_normalization_recovery_v5_phrase_program(
            defeater_tampered, "開啟檔案")
    bad_rules = list(outputs["target-phrase-rules.jsonl"])
    bad_rules[0] = {
        **bad_rules[0],
        "application_scope": {
            **bad_rules[0]["application_scope"],
            "input_match": "BROKEN_MATCH",
        },
    }
    with pytest.raises(BroadQaExternalDataError, match="rule"):
        compile_normalization_recovery_v5_phrase_program(
            rule_pack_manifest_sha256=_sha("v5-runtime-pack"),
            target_phrase_rules=tuple(bad_rules),
            source_phrase_rules=outputs["source-phrase-rules.jsonl"],
            defeaters=outputs["defeaters.jsonl"],
            target_overlap_index=outputs["target-overlap-index.jsonl"],
            source_overlap_index=outputs["source-overlap-index.jsonl"],
        )
