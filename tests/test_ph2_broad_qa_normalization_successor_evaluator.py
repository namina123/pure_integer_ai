"""normalization successor candidate clone 与六维 evaluator 测试。"""
from __future__ import annotations

import hashlib

from pure_integer_ai.experiments import (
    ph2_broad_qa_normalization_successor_evaluation_protocol as protocol_module,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_successor_candidate_clone import (
    compile_normalization_successor_candidate,
    execute_normalization_successor_candidate,
    reference_normalization_successor_candidate,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_successor_evaluation_protocol import (
    NORMALIZATION_SUCCESSOR_EVALUATION_RECORD_KIND,
    NORMALIZATION_SUCCESSOR_TARGET_POLICY_SCOPE,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_successor_evaluator import (
    NORMALIZATION_SUCCESSOR_DIMENSION_ORDER,
    evaluate_normalization_successor_candidate,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_successor_learning_records import (
    NORMALIZATION_SUCCESSOR_CONFLICT_KIND,
    NORMALIZATION_SUCCESSOR_CONSENSUS_RULE_KIND,
    NORMALIZATION_SUCCESSOR_CONTEXT_RULE_KIND,
)
from pure_integer_ai.experiments.ph2_broad_qa_normalization_successor_training_records import (
    ICU_SOURCE_POLICY_SCOPE,
    OPENCC_SOURCE_POLICY_SCOPE,
)
from pure_integer_ai.experiments.ph2_dataset_contract import canonical_json_line


def _id(*values: str) -> str:
    """从 synthetic 稳定字段构造 SHA-256 identity。"""
    return hashlib.sha256("\0".join(values).encode()).hexdigest()


def _synthetic_family(
        *,
        wrong_local: bool = False,
        ) -> tuple[
            dict[str, object],
            tuple[dict[str, object], ...],
            dict[str, object],
            dict[str, tuple[dict[str, object], ...]],
        ]:
    """构造超过六维全部阈值的独立 evaluation 与 learned pack。"""
    source_sha = _id("synthetic-evaluation-source")
    pack_sha = _id("synthetic-rule-pack")
    records = []
    consensus = []
    for ordinal in range(300):
        input_text = chr(0x5000 + ordinal)
        expected = chr(0x6000 + ordinal)
        output = chr(0x7000 + ordinal) if wrong_local and ordinal == 0 else expected
        rule_id = _id("local-rule", str(ordinal), output)
        consensus.append({
            "input_text": input_text,
            "mapping_kind": "CHARACTER_INPUT",
            "output_text": output,
            "record_kind": NORMALIZATION_SUCCESSOR_CONSENSUS_RULE_KIND,
            "rule_id": rule_id,
            "source_policy_scopes": [
                ICU_SOURCE_POLICY_SCOPE, OPENCC_SOURCE_POLICY_SCOPE],
            "target_policy_scope": NORMALIZATION_SUCCESSOR_TARGET_POLICY_SCOPE,
        })
        evaluation_id = _id("local-evaluation", str(ordinal))
        records.append({
            "evaluation_id": evaluation_id,
            "expected_output": expected,
            "family_keys": ["LOCAL_MAPPING_TRANSFER"],
            "format_version": 1,
            "input_text": input_text,
            "record_kind": NORMALIZATION_SUCCESSOR_EVALUATION_RECORD_KIND,
            "source_key": "UNICODE_UNIHAN",
            "source_license_id": "Unicode-3.0",
            "source_line_ordinal": ordinal + 1,
            "source_line_sha256": _id("local-line", str(ordinal)),
            "source_pack_manifest_sha256": source_sha,
            "source_property": "kSimplifiedVariant",
            "source_record_id": _id("local-source-record", str(ordinal)),
            "source_revision": "17.0.0",
            "split": "EVALUATION",
            "target_policy_scope": NORMALIZATION_SUCCESSOR_TARGET_POLICY_SCOPE,
        })
    for ordinal in range(140):
        input_text = chr(0x8000 + ordinal) + chr(0x8200 + ordinal)
        expected = chr(0x9000 + ordinal) + chr(0x9200 + ordinal)
        rule_id = _id("phrase-rule", str(ordinal))
        consensus.append({
            "input_text": input_text,
            "mapping_kind": "PHRASE_INPUT",
            "output_text": expected,
            "record_kind": NORMALIZATION_SUCCESSOR_CONSENSUS_RULE_KIND,
            "rule_id": rule_id,
            "source_policy_scopes": [
                ICU_SOURCE_POLICY_SCOPE, OPENCC_SOURCE_POLICY_SCOPE],
            "target_policy_scope": NORMALIZATION_SUCCESSOR_TARGET_POLICY_SCOPE,
        })
        context_sensitive = int(ordinal < 24)
        families = ["END_TO_END_COVERAGE"]
        if context_sensitive:
            families.append("INDEPENDENT_CONTEXT_TRANSFER")
        records.append({
            "context_sensitive": context_sensitive,
            "evaluation_id": _id("phrase-evaluation", str(ordinal)),
            "expected_output": expected,
            "family_keys": families,
            "format_version": 1,
            "input_text": input_text,
            "position_expectations": [{
                "expected_output": output_item,
                "input_text": input_item,
                "scalar_offset": offset,
            } for offset, (input_item, output_item) in enumerate(
                zip(input_text, expected))],
            "record_kind": NORMALIZATION_SUCCESSOR_EVALUATION_RECORD_KIND,
            "source_key": "MEDIAWIKI_CORE",
            "source_license_id": "GPL-2.0-or-later",
            "source_line_ordinal": ordinal + 1,
            "source_line_sha256": _id("phrase-line", str(ordinal)),
            "source_pack_manifest_sha256": source_sha,
            "source_record_id": _id("phrase-source-record", str(ordinal)),
            "source_revision": "synthetic-commit",
            "source_table": "ZH_TO_HANS",
            "split": "EVALUATION",
            "target_policy_scope": NORMALIZATION_SUCCESSOR_TARGET_POLICY_SCOPE,
        })
    records = tuple(sorted(records, key=lambda item: item["evaluation_id"]))
    evaluation_payload = b"".join(canonical_json_line(item) for item in records)
    summary = {
        "context_evaluation_count": 24,
        "context_reserve_count": 1,
        "evaluation_count": len(records),
        "evaluation_source_counts": {
            "MEDIAWIKI_CORE": 140, "UNICODE_UNIHAN": 300},
        "full_inventory_count": len(records) + 1,
        "local_mapping_evaluation_count": 300,
        "phrase_evaluation_count": 140,
        "reserve_count": 1,
        "reserve_source_counts": {"UNICODE_UNIHAN": 1},
        "split_overlap_count": 0,
    }
    protocol = protocol_module._manifest(
        source_pack_manifest_sha256=source_sha,
        evaluation_artifact={
            "bytes": len(evaluation_payload),
            "record_count": len(records),
            "relative_path": "evaluation.inventory.jsonl",
            "role": "EVALUATION_WITH_LABELS",
            "sha256": hashlib.sha256(evaluation_payload).hexdigest(),
        },
        reserve_artifact={
            "bytes": 100,
            "record_count": 1,
            "relative_path": "reserve.identity.jsonl",
            "role": "RESERVE_IDENTITY_WITHOUT_LABELS",
            "sha256": _id("synthetic-reserve"),
        },
        inventory_summary=summary,
    )
    protocol["manifest_sha256"] = hashlib.sha256(
        canonical_json_line(protocol)).hexdigest()

    conflict_input = "鍾馗"
    conflict_id = _id("synthetic-conflict")
    conflict = ({
        "conflict_id": conflict_id,
        "input_text": conflict_input,
        "record_kind": NORMALIZATION_SUCCESSOR_CONFLICT_KIND,
        "source_policy_outputs": [{
            "expected_output": "钟馗",
            "source_policy_scope": ICU_SOURCE_POLICY_SCOPE,
        }, {
            "expected_output": "锺馗",
            "source_policy_scope": OPENCC_SOURCE_POLICY_SCOPE,
        }],
    },)
    context_rule = ({
        "base_output": "钟馗",
        "context_rule_id": _id("synthetic-context-rule"),
        "input_text": conflict_input,
        "observed_output": "锺馗",
        "record_kind": NORMALIZATION_SUCCESSOR_CONTEXT_RULE_KIND,
        "source_policy_scope": OPENCC_SOURCE_POLICY_SCOPE,
    },)
    outputs = {
        "conflict-ledger.jsonl": conflict,
        "consensus-rules.jsonl": tuple(consensus),
        "context-replays.jsonl": (),
        "context-rules.jsonl": context_rule,
        "evidence.jsonl": (),
        "group-decisions.jsonl": (),
    }
    pack = {
        "manifest_sha256": pack_sha,
        "mastery_claimed": 0,
        "production_enabled": 0,
        "runtime_state": "LEARNED_PACK_DISABLED",
    }
    return protocol, records, pack, outputs


def test_successor_candidate_replays_policy_conflict_and_blocks_unscoped() -> None:
    """目标共识、两个来源输出和无 scope 阻断互不混淆。"""
    _protocol, _records, pack, outputs = _synthetic_family()
    program = compile_normalization_successor_candidate(
        rule_pack_manifest=pack, outputs=outputs)
    text = "鍾馗"
    target = execute_normalization_successor_candidate(
        program, text, policy_scope=NORMALIZATION_SUCCESSOR_TARGET_POLICY_SCOPE)
    opencc = execute_normalization_successor_candidate(
        program, text, policy_scope=OPENCC_SOURCE_POLICY_SCOPE)
    icu = execute_normalization_successor_candidate(
        program, text, policy_scope=ICU_SOURCE_POLICY_SCOPE)
    unscoped = execute_normalization_successor_candidate(
        program, text, policy_scope="")
    assert target.output_text == text
    assert opencc.output_text == "锺馗"
    assert opencc.context_rule_ids and opencc.conflict_ids
    assert icu.output_text == "钟馗" and icu.conflict_ids
    assert unscoped.output_text == text
    assert unscoped.unscoped_conflict_blocked == 1
    for policy in (
            NORMALIZATION_SUCCESSOR_TARGET_POLICY_SCOPE,
            OPENCC_SOURCE_POLICY_SCOPE, ICU_SOURCE_POLICY_SCOPE, ""):
        assert execute_normalization_successor_candidate(
            program, text, policy_scope=policy) == (
                reference_normalization_successor_candidate(
                    program, text, policy_scope=policy))


def test_successor_synthetic_six_dimensions_all_pass() -> None:
    """超过冻结门的独立 synthetic family 必须六维全 PASS。"""
    protocol, records, pack, outputs = _synthetic_family()
    program = compile_normalization_successor_candidate(
        rule_pack_manifest=pack, outputs=outputs)
    report = evaluate_normalization_successor_candidate(
        protocol_manifest=protocol,
        evaluation_records=records,
        rule_pack_manifest=pack,
        pack_outputs=outputs,
        program=program,
    )
    assert report.overall_outcome == "PASS"
    assert tuple(item.dimension_key for item in report.dimensions) == (
        NORMALIZATION_SUCCESSOR_DIMENSION_ORDER)
    assert {item.outcome for item in report.dimensions} == {"PASS"}
    metrics = {
        item.dimension_key: dict(item.metrics) for item in report.dimensions}
    assert metrics["LOCAL_MAPPING_TRANSFER"]["applicable_mapping_count"] == 300
    assert metrics["END_TO_END_COVERAGE"]["full_output_match_count"] == 140
    assert metrics["INDEPENDENT_CONTEXT_TRANSFER"][
        "context_exact_support_count"] == 24
    assert metrics["SOURCE_POLICY_CONFLICT"][
        "policy_specific_replay_mismatch_count"] == 0
    assert report.production_enabled == 0
    assert report.reserve_payload_read_count == 0


def test_missing_clone_is_ne_without_turning_zero_facility_into_pass() -> None:
    """缺 candidate 时设施维度保持 NE，总体不得 PASS。"""
    protocol, records, pack, outputs = _synthetic_family()
    report = evaluate_normalization_successor_candidate(
        protocol_manifest=protocol,
        evaluation_records=records,
        rule_pack_manifest=pack,
        pack_outputs=outputs,
        program=None,
    )
    assert report.overall_outcome == "NE"
    assert {item.outcome for item in report.dimensions} == {"NE"}


def test_wrong_applicable_mapping_is_fail_not_partial_coverage_pass() -> None:
    """已适用 target rule 输出错误时局部维度 FAIL，不用覆盖不足掩盖。"""
    protocol, records, pack, outputs = _synthetic_family(wrong_local=True)
    program = compile_normalization_successor_candidate(
        rule_pack_manifest=pack, outputs=outputs)
    report = evaluate_normalization_successor_candidate(
        protocol_manifest=protocol,
        evaluation_records=records,
        rule_pack_manifest=pack,
        pack_outputs=outputs,
        program=program,
    )
    dimensions = {item.dimension_key: item for item in report.dimensions}
    assert report.overall_outcome == "FAIL"
    assert dimensions["LOCAL_MAPPING_TRANSFER"].outcome == "FAIL"
    assert dict(dimensions["LOCAL_MAPPING_TRANSFER"].metrics)[
        "conflicting_mapping_count"] == 1
