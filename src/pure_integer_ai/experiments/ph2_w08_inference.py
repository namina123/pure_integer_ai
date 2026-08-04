"""W08 Candidate 的只读、逐 Observation 可执行 inference adapter。"""
from __future__ import annotations

import hashlib
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    ObservationRecord,
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_w05_contract import digest_value
from pure_integer_ai.experiments.ph2_w08_authority import W08_DIMENSION_KEYS
from pure_integer_ai.experiments.ph2_w08_contract import W08_CONSUMER_KEYS
from pure_integer_ai.experiments.ph2_w08_inference_contract import (
    W08_CANDIDATE_INFERENCE_OUTPUT_KIND,
    W08_INFERENCE_OWNER_COUNT_KEYS,
    W08_INFERENCE_SHORTCUT_KEYS,
    W08CandidateInferenceError,
    W08CandidateInferenceOutcome,
    W08CandidateInferenceRule,
    W08CandidateInferenceState,
    w08_inference_schema_sha256,
)


_SAMPLE_FAMILY_STATES = {
    "AMBIGUOUS": "CONFLICT",
    "GENERATION": "TRUE",
    "NEGATIVE": "FALSE",
    "POSITIVE": "TRUE",
    "RETENTION": "TRUE",
    "REVISION": "TRUE",
    "UNKNOWN": "UNKNOWN",
}
_COURSE_PAYLOAD_KINDS = {
    "AttributionQuotationCandidateV1",
    "DiscourseInformationCandidateV1",
    "OpenSetClarificationCandidateV1",
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest(value: object) -> tuple[int, ...]:
    return digest_value(value)


def _projection_value(value: object, *, parent: str = "") -> object:
    if isinstance(value, dict):
        result: dict[str, object] = {}
        for key, item in sorted(value.items()):
            if key == "observed_surface" and isinstance(item, dict):
                result["visible_input_receipt"] = {
                    "append_only": item.get("append_only"),
                    "sha256": item.get("sha256"),
                    "target_hidden": item.get("target_hidden"),
                }
            elif key == "raw_observation":
                continue
            elif key == "surface" and isinstance(item, str):
                result["visible_input_sha256"] = _sha256_bytes(
                    item.encode("utf-8")
                )
            else:
                result[str(key)] = _projection_value(item, parent=str(key))
        return result
    if isinstance(value, list):
        return [_projection_value(item, parent=parent) for item in value]
    if value is None or type(value) in {bool, int} or isinstance(value, str):
        return value
    raise W08CandidateInferenceError(
        "Observation projection value 类型非法",
        reason_code="OUTPUT_CONTRACT_REJECTED",
    )


def w08_observation_projection_sha256(observation: ObservationRecord) -> str:
    if not isinstance(observation, ObservationRecord):
        raise TypeError("Candidate inference Observation 类型非法")
    value = {
        "artifact_key": list(observation.artifact_key.components),
        "content_group_key": list(observation.content_group_key.components),
        "language": observation.language,
        "payload_kind": observation.payload_kind,
        "perturbation_kind": observation.perturbation_kind,
        "representation": observation.representation,
        "shape_group_key": list(observation.shape_group_key.components),
        "source_ref_key": list(observation.source_ref_key.components),
        "substage": observation.substage,
        "template_group_key": list(observation.template_group_key.components),
        "typed_projection": _projection_value(observation.typed_payload.to_value()),
        "w_stage": observation.w_stage,
    }
    return _sha256_bytes(canonical_json_bytes(value))


def _scope_value(payload: dict[str, Any], observation: ObservationRecord) -> object:
    if isinstance(payload.get("document_scope_key"), list):
        return payload["document_scope_key"]
    surface_scope = payload.get("surface_scope")
    if isinstance(surface_scope, dict) and surface_scope.get("context_scope_key"):
        return surface_scope["context_scope_key"]
    novelty = payload.get("novelty_profile")
    if isinstance(novelty, dict) and novelty.get("context_scope_key"):
        return novelty["context_scope_key"]
    cluster = payload.get("combination_cluster_key")
    if isinstance(cluster, list) and cluster:
        return cluster
    return list(observation.content_group_key.components)


def _reference_value(payload: dict[str, Any], observation: ObservationRecord) -> object:
    for key in (
        "reference_plan",
        "discourse_relations",
        "attribution_candidates",
        "candidate_branches",
        "combination_axes",
    ):
        value = payload.get(key)
        if value not in (None, [], {}):
            return {key: _projection_value(value)}
    return {
        "prerequisite_keys": [list(item.components) for item in observation.prerequisite_keys],
        "source_ref_key": list(observation.source_ref_key.components),
    }


def w08_source_scope_reference_keys(
    observation: ObservationRecord,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    payload = observation.typed_payload.to_value()
    return (
        tuple(observation.source_ref_key.components),
        _digest({"scope": _scope_value(payload, observation)}),
        _digest({"reference": _reference_value(payload, observation)}),
    )


def _selector(observation: ObservationRecord) -> str:
    payload = observation.typed_payload.to_value()
    if observation.payload_kind in _COURSE_PAYLOAD_KINDS:
        value = payload.get("candidate_kind")
    elif observation.payload_kind == "DiscourseRevisionQuery":
        value = payload.get("variant_kind")
    elif observation.payload_kind == "RAW_SOURCE_OBSERVATION_V1":
        value = "*"
    else:
        value = None
    if not isinstance(value, str) or not value:
        raise W08CandidateInferenceError(
            "Observation selector 未注册",
            reason_code=(
                "PAYLOAD_KIND_UNSUPPORTED"
                if observation.payload_kind not in {
                    *_COURSE_PAYLOAD_KINDS,
                    "DiscourseRevisionQuery",
                    "RAW_SOURCE_OBSERVATION_V1",
                }
                else "SELECTOR_MISSING"
            ),
        )
    return value


def _state(rule: W08CandidateInferenceRule, payload: dict[str, Any]) -> str:
    if rule.state_policy == "SAMPLE_FAMILY_STATE":
        state = _SAMPLE_FAMILY_STATES.get(payload.get("sample_family"))
        if state is None:
            raise W08CandidateInferenceError(
                "sample family 没有 state rule",
                reason_code="STATE_INPUT_REJECTED",
            )
        return state
    if rule.state_policy == "EVIDENCE_BITS_STATE":
        evidence = payload.get("evidence_state")
        if not isinstance(evidence, dict):
            raise W08CandidateInferenceError(
                "revision evidence state 缺失",
                reason_code="STATE_INPUT_REJECTED",
            )
        support = evidence.get("support")
        refute = evidence.get("refute")
        if support not in {0, 1} or refute not in {0, 1}:
            raise W08CandidateInferenceError(
                "revision evidence state 非 binary",
                reason_code="STATE_INPUT_REJECTED",
            )
        return {
            (0, 0): "UNKNOWN",
            (0, 1): "FALSE",
            (1, 0): "TRUE",
            (1, 1): "CONFLICT",
        }[(support, refute)]
    if rule.state_policy == "SOURCE_RECEIPT_STATE":
        raw_sha = payload.get("raw_observation_sha256")
        raw_observation = payload.get("raw_observation")
        if (
            payload.get("raw_observation_append_only") != 1
            or not isinstance(raw_sha, str)
            or len(raw_sha) != 64
            or any(char not in "0123456789abcdef" for char in raw_sha)
            or not isinstance(raw_observation, dict)
            or _sha256_bytes(canonical_json_bytes(raw_observation)) != raw_sha
        ):
            raise W08CandidateInferenceError(
                "source receipt 输入非法",
                reason_code="STATE_INPUT_REJECTED",
            )
        return "TRUE"
    raise W08CandidateInferenceError(
        "Candidate state policy 未实现",
        reason_code="STATE_POLICY_UNSUPPORTED",
    )


def _visible_input(payload: dict[str, Any]) -> tuple[dict[str, Any] | None, int]:
    value = payload.get("observed_surface")
    if value is None:
        text = payload.get("surface")
        if isinstance(text, str) and text:
            return {
                "append_only": 1,
                "sha256": _sha256_bytes(text.encode("utf-8")),
                "target_hidden": 0,
                "text": text,
            }, 1
        return None, 0
    if not isinstance(value, dict):
        raise W08CandidateInferenceError(
            "visible input receipt 类型非法",
            reason_code="RENDER_INPUT_REJECTED",
        )
    text = value.get("text")
    sha = value.get("sha256")
    if (
        value.get("append_only") != 1
        or value.get("target_hidden") not in {0, 1}
        or not isinstance(text, str)
        or not text
        or not isinstance(sha, str)
        or _sha256_bytes(text.encode("utf-8")) != sha
    ):
        raise W08CandidateInferenceError(
            "visible input receipt SHA/state 漂移",
            reason_code="RENDER_INPUT_REJECTED",
        )
    return value, 1


def _structural_generation(payload_kind: str, payload: dict[str, Any]) -> str:
    if payload_kind == "DiscourseInformationCandidateV1":
        relations = payload.get("discourse_relations")
        relation = relations[0].get("relation_kind") if isinstance(relations, list) and relations else "RELATION"
        connective = {
            "CAUSE": "因此",
            "CONCESSION": "不过",
            "CONTRAST": "不过",
            "ELABORATION": "并且",
        }.get(str(relation), "同时")
        return f"第一项信息已经给定；{connective}第二项信息仍需核验。"
    if payload_kind == "OpenSetClarificationCandidateV1":
        branches = payload.get("candidate_branches")
        count = len(branches) if isinstance(branches, list) else 0
        if count < 2:
            return "请补充一条能够区分当前用法的证据。"
        return "你指的是第一个候选用法，还是第二个候选用法？"
    if payload_kind == "AttributionQuotationCandidateV1":
        candidates = payload.get("attribution_candidates")
        first = candidates[0] if isinstance(candidates, list) and candidates else {}
        role = {
            "PERSON": "相关人员",
            "GROUP": "相关小组",
            "SYSTEM": "监控系统",
        }.get(str(first.get("holder_role")), "来源方")
        uncertainty = str(first.get("uncertainty_state"))
        marker = "可能" if uncertainty not in {"CERTAIN", "ASSERTED"} else "已经"
        return f"据{role}判断，相关命题{marker}成立。"
    raise W08CandidateInferenceError(
        "结构生成器不支持该 payload kind",
        reason_code="RENDER_POLICY_UNSUPPORTED",
    )


def _resolution_state(payload_kind: str, selector: str, state: str) -> str:
    if state == "TRUE":
        return "RESOLVED"
    if state == "FALSE":
        return "REFUTED"
    if state == "CONFLICT":
        return "CONFLICT"
    if payload_kind == "OpenSetClarificationCandidateV1" and selector in {
        "ACCESS_BLOCKED",
        "BUDGET_BLOCKED",
        "UNKNOWN",
    }:
        return "CLARIFY"
    return "UNKNOWN"


def _result_payload(
    observation: ObservationRecord,
    rule: W08CandidateInferenceRule,
    state: str,
) -> tuple[dict[str, object], int]:
    payload = observation.typed_payload.to_value()
    visible, input_reads = _visible_input(payload)
    projection_sha = w08_observation_projection_sha256(observation)
    if rule.state_policy == "SAMPLE_FAMILY_STATE":
        outputs: list[str] = []
        if state == "TRUE" and rule.render_policy == "COPY_VISIBLE_TEXT":
            if visible is None or visible.get("target_hidden") != 0:
                raise W08CandidateInferenceError(
                    "COPY_VISIBLE_TEXT 缺少可见输入",
                    reason_code="RENDER_INPUT_REJECTED",
                )
            outputs = [str(visible["text"])]
        elif state == "TRUE" and rule.render_policy == "STRUCTURAL_GENERATOR":
            if visible is None:
                raise W08CandidateInferenceError(
                    "STRUCTURAL_GENERATOR 缺少 typed input",
                    reason_code="RENDER_INPUT_REJECTED",
                )
            outputs = [_structural_generation(observation.payload_kind, payload)]
        elif state == "TRUE" or rule.render_policy != "NO_TEXT":
            raise W08CandidateInferenceError(
                "Candidate render policy 与 state 漂移",
                reason_code="RENDER_POLICY_UNSUPPORTED",
            )
        result = {
            "accepted": int(state == "TRUE"),
            "generated_outputs": outputs,
            "operation_key": rule.operation_key,
            "render_policy": rule.render_policy,
            "render_receipt": {
                "input_hidden": int(bool(visible and visible.get("target_hidden") == 1)),
                "output_count": len(outputs),
                "output_sha256": [
                    _sha256_bytes(item.encode("utf-8")) for item in outputs
                ],
                "postcheck_state": "PASS",
            },
        }
    elif rule.state_policy == "EVIDENCE_BITS_STATE":
        evidence = payload["evidence_state"]
        result = {
            "decision": rule.operation_key,
            "result_bits": [evidence["support"], evidence["refute"]],
        }
    else:
        result = {
            "definitive_truth_authoritative": payload[
                "definitive_truth_authoritative"
            ],
            "parser_version": 1,
            "raw_observation_sha256": payload["raw_observation_sha256"],
            "source_ref_key": list(observation.source_ref_key.components),
        }
    return {
        "artifact_kind": W08_CANDIDATE_INFERENCE_OUTPUT_KIND,
        "format_version": 1,
        "operation_key": rule.operation_key,
        "payload_kind": observation.payload_kind,
        "resolution_state": _resolution_state(
            observation.payload_kind, rule.selector_key, state
        ),
        "result": result,
        "semantic_projection_sha256": projection_sha,
    }, input_reads


def _resource_available(payload: dict[str, Any]) -> bool:
    budget = payload.get("resource_budget")
    if budget is None:
        return True
    if not isinstance(budget, dict) or any(
        type(value) is not int or value < 0 for value in budget.values()
    ):
        return False
    output_units = budget.get("max_output_units", 0)
    return type(output_units) is int and output_units <= 160


def _component_active(
    dimension_key: str,
    observation: ObservationRecord,
    actual_payload: dict[str, object],
    *,
    resource_available: bool,
) -> tuple[bool, tuple[int, ...]]:
    payload = observation.typed_payload.to_value()
    source_key, scope_key, reference_key = w08_source_scope_reference_keys(observation)
    if dimension_key == W08_DIMENSION_KEYS[0]:
        visible = payload.get("observed_surface")
        revision_text = payload.get("surface")
        raw_sha = payload.get("raw_observation_sha256")
        valid = observation.language == "zh" and (
            isinstance(visible, dict)
            or isinstance(revision_text, str)
            or isinstance(raw_sha, str)
        )
        account = {"language": observation.language, "typed_input": int(valid)}
    elif dimension_key == W08_DIMENSION_KEYS[1]:
        valid = actual_payload.get("resolution_state") in {
            "RESOLVED", "REFUTED", "CONFLICT", "UNKNOWN", "CLARIFY"
        }
        account = {
            "payload_kind": observation.payload_kind,
            "resolution_state": actual_payload.get("resolution_state"),
        }
    elif dimension_key == W08_DIMENSION_KEYS[2]:
        if observation.payload_kind == "DiscourseRevisionQuery":
            valid = all(payload.get(key) == 0 for key in (
                "whole_document_recomputed",
                "unaffected_recomputed",
                "old_occurrences_rewritten",
            ))
        elif isinstance(payload.get("revision_receipt"), dict):
            valid = payload["revision_receipt"].get("raw_observation_preserved") == 1
        elif isinstance(payload.get("clarification_receipt"), dict):
            valid = payload["clarification_receipt"].get(
                "raw_observation_preserved"
            ) == 1
        else:
            valid = payload.get("raw_observation_append_only") == 1
        account = {"local_only": int(valid), "whole_document_runs": 0}
    elif dimension_key == W08_DIMENSION_KEYS[3]:
        valid = bool(source_key and scope_key and reference_key)
        account = {
            "preloaded_hot_records": 0,
            "reference_key": list(reference_key),
            "scope_key": list(scope_key),
        }
    else:
        result = actual_payload.get("result")
        valid = isinstance(result, dict) and resource_available
        account = {
            "payload_commitment": _sha256_bytes(canonical_json_bytes(result or {})),
            "postcheck": int(valid),
        }
    return valid and resource_available, _digest({
        "account": account,
        "dimension_key": dimension_key,
        "observation_key": list(observation.stable_key.components),
    })


def validate_w08_inference_outcome(
    observation: ObservationRecord,
    outcome: W08CandidateInferenceOutcome,
) -> bool:
    if not isinstance(observation, ObservationRecord) or not isinstance(
        outcome, W08CandidateInferenceOutcome
    ):
        return False
    payload = outcome.actual_payload.to_value()
    source, scope, reference = w08_source_scope_reference_keys(observation)
    return all((
        outcome.observation_key == tuple(observation.stable_key.components),
        payload.get("semantic_projection_sha256")
        == w08_observation_projection_sha256(observation),
        outcome.source_key == source,
        outcome.scope_key == scope,
        outcome.reference_key == reference,
        not any(value for _, value in outcome.shortcut_counts),
        not any(value for _, value in outcome.owner_counts),
    ))


class W08CandidateInferenceAdapter:
    """只持有 frozen state；每次调用无写入且不接收 evaluator label。"""

    def __init__(self, state: W08CandidateInferenceState) -> None:
        if not isinstance(state, W08CandidateInferenceState):
            raise TypeError("Candidate inference state 类型非法")
        self.state = state
        self.state_sha256 = state.sha256()
        self._rules = {
            (item.payload_kind, item.selector_key): item for item in state.rules
        }
        schemas_by_payload_kind: dict[str, set[str]] = {}
        for item in state.rules:
            schemas_by_payload_kind.setdefault(item.payload_kind, set()).add(
                item.schema_sha256
            )
        self._schemas_by_payload_kind = {
            key: frozenset(values)
            for key, values in schemas_by_payload_kind.items()
        }

    def infer(
        self,
        observation: ObservationRecord,
        *,
        dimension_key: str,
        disabled_components: tuple[str, ...] = (),
    ) -> W08CandidateInferenceOutcome:
        if not isinstance(observation, ObservationRecord):
            raise TypeError("Candidate inference input 必须是 ObservationRecord")
        if observation.split != "held_out":
            raise W08CandidateInferenceError(
                "Candidate inference 只接受 held-out Observation",
                reason_code="INPUT_CONTRACT_REJECTED",
            )
        if dimension_key not in W08_DIMENSION_KEYS:
            raise W08CandidateInferenceError(
                "Candidate inference bearing 未注册",
                reason_code="INPUT_CONTRACT_REJECTED",
            )
        if (
            tuple(sorted(set(disabled_components))) != disabled_components
            or any(item not in W08_DIMENSION_KEYS for item in disabled_components)
        ):
            raise W08CandidateInferenceError(
                "Candidate inference disabled component 非法",
                reason_code="INPUT_CONTRACT_REJECTED",
            )
        selector = _selector(observation)
        rule = self._rules.get((observation.payload_kind, selector))
        if rule is None:
            raise W08CandidateInferenceError(
                "Candidate inference selector 未由 train state 学得",
                reason_code="SELECTOR_UNSEEN",
            )
        typed_payload = observation.typed_payload.to_value()
        if w08_inference_schema_sha256(typed_payload) not in (
            self._schemas_by_payload_kind.get(observation.payload_kind, ())
        ):
            raise W08CandidateInferenceError(
                "Candidate inference held-out schema 漂移：未由同类 train family 学得",
                reason_code="SCHEMA_UNSEEN",
            )
        state = _state(rule, typed_payload)
        actual_payload, input_reads = _result_payload(observation, rule, state)
        resource_available = _resource_available(typed_payload)
        active, component_receipt = _component_active(
            dimension_key,
            observation,
            actual_payload,
            resource_available=resource_available,
        )
        if dimension_key in disabled_components:
            component_state = "DISABLED"
            consumer_state = "FAIL_CLOSED"
            state = "UNKNOWN"
            actual_payload = {
                "artifact_kind": W08_CANDIDATE_INFERENCE_OUTPUT_KIND,
                "format_version": 1,
                "operation_key": rule.operation_key,
                "payload_kind": observation.payload_kind,
                "publication": 0,
                "reason_code": "COMPONENT_DISABLED",
                "semantic_projection_sha256": w08_observation_projection_sha256(
                    observation
                ),
            }
        elif not resource_available:
            component_state = "FAIL_CLOSED"
            consumer_state = "BUDGET_EXHAUSTED"
            state = "UNKNOWN"
            actual_payload = {
                "artifact_kind": W08_CANDIDATE_INFERENCE_OUTPUT_KIND,
                "format_version": 1,
                "operation_key": rule.operation_key,
                "payload_kind": observation.payload_kind,
                "publication": 0,
                "reason_code": "BUDGET_EXHAUSTED",
                "semantic_projection_sha256": w08_observation_projection_sha256(
                    observation
                ),
            }
        elif not active:
            component_state = "FAIL_CLOSED"
            consumer_state = "FAIL_CLOSED"
            state = "UNKNOWN"
        else:
            component_state = "ACTIVE"
            consumer_state = "RESOLVED"
        payload_object = CanonicalJsonObject.from_value(actual_payload)
        payload_sha = _sha256_bytes(canonical_json_bytes(payload_object.to_value()))
        source_key, scope_key, reference_key = w08_source_scope_reference_keys(
            observation
        )
        consumer_states = tuple(
            (consumer, consumer_state) for consumer in W08_CONSUMER_KEYS
        )
        shortcut_counts = tuple((key, 0) for key in W08_INFERENCE_SHORTCUT_KEYS)
        owner_counts = tuple((key, 0) for key in W08_INFERENCE_OWNER_COUNT_KEYS)
        invocation = _digest({
            "actual_payload_sha256": payload_sha,
            "actual_state": state,
            "component_receipt_key": list(component_receipt),
            "component_state": component_state,
            "consumer_states": [list(item) for item in consumer_states],
            "dimension_key": dimension_key,
            "disabled_components": list(disabled_components),
            "observation_key": list(observation.stable_key.components),
            "state_commitment_sha256": self.state_sha256,
        })
        return W08CandidateInferenceOutcome(
            dimension_key,
            tuple(observation.stable_key.components),
            state,
            payload_object,
            payload_sha,
            consumer_states,
            component_state,
            component_receipt,
            source_key,
            scope_key,
            reference_key,
            shortcut_counts,
            owner_counts,
            input_reads,
            self.state_sha256,
            invocation,
        )


__all__ = [
    "W08CandidateInferenceAdapter",
    "validate_w08_inference_outcome",
    "w08_observation_projection_sha256",
    "w08_source_scope_reference_keys",
]
