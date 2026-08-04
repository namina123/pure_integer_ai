"""从 W08 train Observation/Evidence 编译无答案样本的 inference state。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.experiments.ph2_dataset_contract import ObservationRecord
from pure_integer_ai.experiments.ph2_w05_contract import digest_value
from pure_integer_ai.experiments.ph2_w08_inference_contract import (
    W08CandidateInferenceError,
    W08CandidateInferenceRule,
    W08CandidateInferenceState,
    make_w08_inference_rule,
    make_w08_inference_state,
    w08_inference_schema_sha256,
)
from pure_integer_ai.experiments.ph2_w08_payload import W08TrainingPayload


_COURSE_PAYLOAD_KINDS = {
    "AttributionQuotationCandidateV1",
    "DiscourseInformationCandidateV1",
    "OpenSetClarificationCandidateV1",
}
_COURSE_EVIDENCE_KINDS = {
    "AttributionQuotationCandidateV1": "ATTRIBUTION_QUOTATION_LABEL",
    "DiscourseInformationCandidateV1": "DISCOURSE_INFORMATION_LABEL",
    "OpenSetClarificationCandidateV1": "OPEN_SET_CLARIFICATION_LABEL",
}
_SAMPLE_FAMILY_STATES = {
    "AMBIGUOUS": "CONFLICT",
    "GENERATION": "TRUE",
    "NEGATIVE": "FALSE",
    "POSITIVE": "TRUE",
    "RETENTION": "TRUE",
    "REVISION": "TRUE",
    "UNKNOWN": "UNKNOWN",
}


@dataclass(frozen=True)
class _RuleDraft:
    payload_kind: str
    selector_key: str
    state_policy: str
    render_policy: str
    operation_key: str
    schema_sha256: str
    evidence_key: tuple[int, ...]


def _normalize_operation_key(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise W08CandidateInferenceError("train typed operation key 为空")
    prefix, marker, suffix = value.partition("_")
    if marker and len(prefix) == 1 and suffix:
        return suffix
    return value


def _evidence_bits_state(value: object) -> tuple[str, tuple[int, int]]:
    if not isinstance(value, dict):
        raise W08CandidateInferenceError("revision evidence bits 类型非法")
    support = value.get("support")
    refute = value.get("refute")
    if support not in {0, 1} or refute not in {0, 1}:
        raise W08CandidateInferenceError("revision evidence bits 非 binary")
    state = {
        (0, 0): "UNKNOWN",
        (0, 1): "FALSE",
        (1, 0): "TRUE",
        (1, 1): "CONFLICT",
    }[(support, refute)]
    return state, (support, refute)


def _course_draft(observation: ObservationRecord, evidence) -> _RuleDraft:
    payload = observation.typed_payload.to_value()
    typed_evidence = evidence.typed_evidence.to_value()
    if evidence.evidence_kind != _COURSE_EVIDENCE_KINDS[observation.payload_kind]:
        raise W08CandidateInferenceError("course Evidence kind 与 payload kind 漂移")
    candidate_kind = payload.get("candidate_kind")
    sample_family = payload.get("sample_family")
    visible = payload.get("observed_surface")
    result = typed_evidence.get("expected_payload")
    state = _SAMPLE_FAMILY_STATES.get(sample_family)
    if (
        not isinstance(candidate_kind, str)
        or not candidate_kind
        or state is None
        or not isinstance(visible, dict)
        or not isinstance(result, dict)
        or typed_evidence.get("expected_state") != state
        or result.get("accepted") != int(state == "TRUE")
    ):
        raise W08CandidateInferenceError("course train state/acceptance 不能由 typed rule 解释")
    hidden = visible.get("target_hidden")
    text = visible.get("text")
    accepted_text = result.get("accepted_surfaces")
    if hidden not in {0, 1} or not isinstance(text, str) or not isinstance(accepted_text, list):
        raise W08CandidateInferenceError("course train visible text receipt 非法")
    if state != "TRUE":
        if accepted_text:
            raise W08CandidateInferenceError("非 TRUE course train 错带 render target")
        render_policy = "NO_TEXT"
    elif hidden == 0:
        if accepted_text == [text]:
            render_policy = "COPY_VISIBLE_TEXT"
        elif accepted_text and all(
            isinstance(item, str) and item for item in accepted_text
        ):
            render_policy = "STRUCTURAL_GENERATOR"
        else:
            raise W08CandidateInferenceError("可见 train text 没有可执行 render Evidence")
    else:
        if not accepted_text or any(not isinstance(item, str) or not item for item in accepted_text):
            raise W08CandidateInferenceError("隐藏 train target 没有 generation Evidence")
        render_policy = "STRUCTURAL_GENERATOR"
    return _RuleDraft(
        observation.payload_kind,
        candidate_kind,
        "SAMPLE_FAMILY_STATE",
        render_policy,
        _normalize_operation_key(result.get("analysis_key")),
        w08_inference_schema_sha256(payload),
        tuple(evidence.stable_key.components),
    )


def _revision_draft(observation: ObservationRecord, evidence) -> _RuleDraft:
    payload = observation.typed_payload.to_value()
    typed_evidence = evidence.typed_evidence.to_value()
    state, bits = _evidence_bits_state(payload.get("evidence_state"))
    result = typed_evidence.get("expected_payload")
    variant = payload.get("variant_kind")
    if (
        evidence.evidence_kind != "DISCOURSE_REVISION_LABEL"
        or not isinstance(variant, str)
        or not variant
        or typed_evidence.get("expected_state") != state
        or not isinstance(result, dict)
        or result.get("result_bits") != list(bits)
        or not isinstance(result.get("decision"), str)
    ):
        raise W08CandidateInferenceError("revision train Evidence 不能由局部规则解释")
    return _RuleDraft(
        observation.payload_kind,
        variant,
        "EVIDENCE_BITS_STATE",
        "STRUCTURED_PAYLOAD",
        str(result["decision"]),
        w08_inference_schema_sha256(payload),
        tuple(evidence.stable_key.components),
    )


def _source_draft(observation: ObservationRecord, evidence) -> _RuleDraft:
    payload = observation.typed_payload.to_value()
    receipt = evidence.typed_evidence.to_value()
    if (
        evidence.evidence_kind != "SOURCE_PARSER_RECEIPT_V1"
        or payload.get("raw_observation_append_only") != 1
        or receipt.get("raw_observation_sha256") != payload.get("raw_observation_sha256")
        or receipt.get("definitive_truth_authoritative")
        != payload.get("definitive_truth_authoritative")
        or receipt.get("source_ref_key")
        != list(observation.source_ref_key.components)
        or receipt.get("parser_version") != 1
    ):
        raise W08CandidateInferenceError("source train parser receipt 不能机械重建")
    return _RuleDraft(
        observation.payload_kind,
        "*",
        "SOURCE_RECEIPT_STATE",
        "STRUCTURED_PAYLOAD",
        "SOURCE_PARSER_RECEIPT_V1",
        w08_inference_schema_sha256(payload),
        tuple(evidence.stable_key.components),
    )


def _draft(observation: ObservationRecord, evidence) -> _RuleDraft:
    if observation.payload_kind in _COURSE_PAYLOAD_KINDS:
        return _course_draft(observation, evidence)
    if observation.payload_kind == "DiscourseRevisionQuery":
        return _revision_draft(observation, evidence)
    if observation.payload_kind == "RAW_SOURCE_OBSERVATION_V1":
        return _source_draft(observation, evidence)
    raise W08CandidateInferenceError("train Observation payload kind 未注册")


def compile_w08_candidate_inference_state(
    payload: W08TrainingPayload,
) -> W08CandidateInferenceState:
    """消费 train Evidence 验证规则，但 state 不保存答案或文本值。"""
    if not isinstance(payload, W08TrainingPayload):
        raise TypeError("W08 inference compiler payload 类型非法")
    by_observation = {item.observation_key: item for item in payload.teacher_evidence}
    if (
        len(by_observation) != len(payload.teacher_evidence)
        or set(by_observation) != {item.stable_key for item in payload.observations}
    ):
        raise W08CandidateInferenceError("train Observation/Evidence 不是一对一闭合")
    drafts = [_draft(item, by_observation[item.stable_key]) for item in payload.observations]
    grouped: dict[tuple[str, str], list[_RuleDraft]] = {}
    for item in drafts:
        grouped.setdefault((item.payload_kind, item.selector_key), []).append(item)
    rules: list[W08CandidateInferenceRule] = []
    for (payload_kind, selector_key), values in sorted(grouped.items()):
        reference = values[0]
        if any(
            (
                item.state_policy,
                item.render_policy,
                item.operation_key,
                item.schema_sha256,
            )
            != (
                reference.state_policy,
                reference.render_policy,
                reference.operation_key,
                reference.schema_sha256,
            )
            for item in values[1:]
        ):
            raise W08CandidateInferenceError("同 selector 的 train rule 不确定")
        rules.append(make_w08_inference_rule(
            payload_kind=payload_kind,
            selector_key=selector_key,
            state_policy=reference.state_policy,
            render_policy=reference.render_policy,
            operation_key=reference.operation_key,
            schema_sha256=reference.schema_sha256,
            evidence_keys=tuple(sorted(item.evidence_key for item in values)),
        ))
    training_identity = digest_value({
        "evidence_keys": [
            list(item.stable_key.components) for item in payload.teacher_evidence
        ],
        "observation_keys": [
            list(item.stable_key.components) for item in payload.observations
        ],
        "rule_keys": [list(item.rule_key) for item in rules],
    })
    return make_w08_inference_state(
        tuple(rules),
        training_record_count=len(payload.observations),
        training_evidence_count=len(payload.teacher_evidence),
        training_identity_key=training_identity,
    )


__all__ = ["compile_w08_candidate_inference_state"]
