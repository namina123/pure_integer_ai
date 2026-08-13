"""来源内可审计归纳合同的身份、负例和规范字节测试。"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import pytest

from pure_integer_ai.cognition.shared.identity import (
    GLOBAL_OWNER_SCOPE,
    SourceRef,
    VersionBundle,
    concept_identity,
    minimal_instruction_identity,
    occurrence_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.hypothesis import (
    EVIDENCE_REFUTE,
    EVIDENCE_SUPPORT,
    EvidenceRecord,
)
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.cognition.shared.semantic_object import (
    AtomicPropositionDefinition,
    AtomicRoleBinding,
    context_scope_identity,
    entity_identity,
    proposition_identity,
    role_identity,
)
from pure_integer_ai.experiments.ph2_broad_qa_source_inference_contract import (
    BroadQaSourceDerivation,
    BroadQaSourceDerivedClaim,
    BroadQaSourceInferenceError,
    BroadQaSourceInferenceRecord,
    BroadQaSourceObservation,
    BroadQaSourceOutputPart,
    BroadQaSourcePremise,
    BroadQaSourceRoleProjection,
    BroadQaSourceRoleSpan,
    parse_broad_qa_source_inference_record,
    source_inference_rule_hypothesis_key,
)


def _fixture():
    """构造不依赖自然语言同义词表的 source-bound 两前提派生。"""
    source = SourceRef(
        801, 991, 77, GLOBAL_OWNER_SCOPE, VersionBundle())
    subject_role = role_identity((810, 1))
    object_role = role_identity((810, 2))
    result_role = role_identity((810, 3))
    predicate = concept_identity((811, 1))
    derived_predicate = concept_identity((811, 2))
    subject = entity_identity(source, (812, 1))
    middle = entity_identity(source, (812, 2))
    value = entity_identity(source, (812, 3))

    def premise(
            ordinal: int,
            text: str,
            left,
            right,
            ) -> BroadQaSourcePremise:
        raw_start = ordinal * 100
        raw_end = raw_start + len(text)
        observation = BroadQaSourceObservation(
            source,
            "ZHWIKIPEDIA_20260701",
            "CC-BY-SA-4.0",
            "示例页",
            source.source_id,
            source.document_id,
            ordinal,
            raw_start,
            raw_end,
            hashlib.sha256(text.encode("utf-8")).hexdigest(),
            text,
            0,
            len(text),
            text,
        )
        definition = AtomicPropositionDefinition(
            proposition_identity(source, (813, ordinal)),
            predicate,
            occurrence_identity(
                source, start=raw_start, end=raw_end, ordinal=ordinal),
            context_scope_identity(source, (814, ordinal)),
            (
                AtomicRoleBinding(subject_role, left),
                AtomicRoleBinding(object_role, right),
            ),
        )
        separator = text.index("|")
        spans = (
            BroadQaSourceRoleSpan(
                definition.bindings[0], 0, separator, text[:separator]),
            BroadQaSourceRoleSpan(
                definition.bindings[1], separator + 1, len(text),
                text[separator + 1:]),
        )
        return BroadQaSourcePremise(observation, definition, spans)

    first = premise(1, "甲|乙", subject, middle)
    second = premise(2, "乙|丙", middle, value)
    result_definition = AtomicPropositionDefinition(
        proposition_identity(source, (815, 1)),
        derived_predicate,
        first.definition.source_anchor,
        context_scope_identity(source, (816, 1)),
        (AtomicRoleBinding(result_role, value),),
    )
    projection = BroadQaSourceRoleProjection(
        result_role, 0, 1, object_role, 0)
    operator = minimal_instruction_identity((817, 1))
    schema = structure_concept_identity((818, 1))
    applicability_scope = document_scope(source)
    rule_hypothesis = source_inference_rule_hypothesis_key(
        operator, schema, "FORWARD", 1, applicability_scope)
    rule_evidence = EvidenceRecord(
        819,
        rule_hypothesis,
        EVIDENCE_SUPPORT,
        (819, 1),
        source,
        1,
    )
    derivation = BroadQaSourceDerivation(
        operator,
        1,
        schema,
        "FORWARD",
        applicability_scope,
        (first.sha256(), second.sha256()),
        (rule_evidence.stable_key(),),
        (projection,),
        (),
        (),
    )
    claim = BroadQaSourceDerivedClaim(
        (first, second),
        derivation,
        result_definition,
        (BroadQaSourceOutputPart(result_role, 0, "丙"),),
        "丙",
    )
    return BroadQaSourceInferenceRecord(
        claim,
        "a" * 64,
        hashlib.sha256("示例问题".encode("utf-8")).hexdigest(),
        (hashlib.sha256("丙".encode("utf-8")).hexdigest(),),
        hashlib.sha256("示例终页".encode("utf-8")).hexdigest(),
    )


def test_record_round_trip_is_bit_identical_and_contract_only() -> None:
    """规范 JSONL 回读保持逐字节一致且不冒充生产能力。"""
    record = _fixture()
    restored = parse_broad_qa_source_inference_record(record.canonical_bytes())
    assert restored == record
    assert restored.sha256() == record.sha256()
    assert restored.production_enabled == 0
    assert restored.runtime_state == "CONTRACT_ONLY_DISABLED"
    assert restored.claim.epistemic_status == (
        "SOURCE_DERIVED_FROM_ASSERTIONS")
    assert restored.claim.truth_status == "NOT_ADJUDICATED"


def test_tamper_unknown_field_and_noncanonical_bytes_fail_closed() -> None:
    """字段添加、表层篡改和非规范编码均不得静默回读。"""
    record = _fixture()
    value = record.to_dict()
    value["unexpected"] = 1
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"
    with pytest.raises(BroadQaSourceInferenceError, match="字段漂移"):
        parse_broad_qa_source_inference_record(payload)

    canonical_value = record.to_dict()
    canonical_value["claim"]["rendered_text"] = "伪造"
    tampered = json.dumps(
        canonical_value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"
    with pytest.raises(BroadQaSourceInferenceError, match="来源拼接"):
        parse_broad_qa_source_inference_record(tampered)

    noncanonical = json.dumps(
        record.to_dict(), ensure_ascii=False, sort_keys=True,
    ).encode("utf-8") + b"\n"
    with pytest.raises(BroadQaSourceInferenceError, match="规范 JSON"):
        parse_broad_qa_source_inference_record(noncanonical)


def test_role_projection_cannot_replace_or_invent_argument() -> None:
    """结果 filler 必须逐身份复用映射的 premise filler。"""
    record = _fixture()
    claim = record.claim
    source = claim.premises[0].observation.source
    forged = AtomicRoleBinding(
        claim.definition.bindings[0].role,
        entity_identity(source, (999, 1)),
    )
    with pytest.raises(BroadQaSourceInferenceError, match="typed filler"):
        replace(
            claim,
            definition=replace(claim.definition, bindings=(forged,)),
        )


def test_uncleared_defeater_and_cross_source_premise_are_rejected() -> None:
    """负条件未清除或 premise 跨来源时必须失败关闭。"""
    record = _fixture()
    claim = record.claim
    blocker = concept_identity((900, 1))
    with pytest.raises(BroadQaSourceInferenceError, match="未清除"):
        replace(claim.derivation, defeaters=(blocker,))

    foreign_source = SourceRef(
        801, 992, 77, GLOBAL_OWNER_SCOPE, VersionBundle())
    foreign_observation = replace(
        claim.premises[1].observation,
        source=foreign_source,
        page_id=foreign_source.source_id,
    )
    with pytest.raises(BroadQaSourceInferenceError, match="同一来源"):
        replace(
            claim,
            premises=(
                claim.premises[0],
                replace(claim.premises[1], observation=foreign_observation),
            ),
        )


def test_output_cannot_add_text_absent_from_projected_role_span() -> None:
    """当前合同只允许逐字输出，不允许无 verifier 的改写或算术生成。"""
    claim = _fixture().claim
    part = claim.output_parts[0]
    with pytest.raises(BroadQaSourceInferenceError, match="来源外字符"):
        replace(
            claim,
            output_parts=(replace(part, surface="丙病"),),
            rendered_text="丙病",
        )


def test_rule_evidence_must_support_the_exact_bound_rule() -> None:
    """无关、反驳或不能恢复的 Evidence 不得冒充规则学习依据。"""
    derivation = _fixture().claim.derivation
    evidence = EvidenceRecord.from_stable_key(
        derivation.rule_evidence_keys[0])
    refute = replace(evidence, stance=EVIDENCE_REFUTE)
    with pytest.raises(BroadQaSourceInferenceError, match="绑定当前规则"):
        replace(derivation, rule_evidence_keys=(refute.stable_key(),))

    other_hypothesis = source_inference_rule_hypothesis_key(
        minimal_instruction_identity((817, 99)),
        derivation.schema,
        derivation.direction,
        derivation.operator_version,
        derivation.applicability_scope,
    )
    unrelated = replace(evidence, hypothesis=other_hypothesis)
    with pytest.raises(BroadQaSourceInferenceError, match="绑定当前规则"):
        replace(derivation, rule_evidence_keys=(unrelated.stable_key(),))

    with pytest.raises(BroadQaSourceInferenceError, match="无法恢复"):
        replace(derivation, rule_evidence_keys=((819, 1),))


def test_contract_contains_no_seen_failure_or_open_surface_dispatch() -> None:
    """合同源码不得含当前失败答案、标题或开放类同义词补丁。"""
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1]
        / "src/pure_integer_ai/experiments/"
        "ph2_broad_qa_source_inference_contract.py"
    ).read_text(encoding="utf-8")
    for forbidden in (
            "肾病", "继发性高血压", "顾茱莉", "去世", "学历", "硕士"):
        assert forbidden not in source
