"""W08 Candidate inference state 与逐 case 输出合同。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    EXPECTED_STATES,
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_w05_contract import digest_value
from pure_integer_ai.experiments.ph2_w08_authority import W08_DIMENSION_KEYS
from pure_integer_ai.experiments.ph2_w08_contract import W08_CONSUMER_KEYS


W08_CANDIDATE_INFERENCE_INTERFACE_VERSION = "PH2-W08-PRIVATE-INFERENCE-V2"
W08_CANDIDATE_INFERENCE_STATE_KIND = "PH2_W08_CANDIDATE_INFERENCE_STATE"
W08_CANDIDATE_INFERENCE_OUTPUT_KIND = "PH2_W08_CANDIDATE_INFERENCE_OUTPUT"
W08_CANDIDATE_INFERENCE_INPUT_KIND = "OBSERVATION_RECORD_V1"
W08_INFERENCE_PAYLOAD_KINDS = (
    "AttributionQuotationCandidateV1",
    "DiscourseInformationCandidateV1",
    "DiscourseRevisionQuery",
    "OpenSetClarificationCandidateV1",
    "RAW_SOURCE_OBSERVATION_V1",
)
W08_INFERENCE_SHORTCUT_KEYS = (
    "exact_surface_reads",
    "fifo_or_recency_choices",
    "full_recompute_runs",
    "preloaded_hot_records",
    "w09_future_reads",
)
W08_INFERENCE_OWNER_COUNT_KEYS = (
    "candidate_writes",
    "evaluator_label_reads",
    "future_payload_reads",
    "host_learning_writes",
    "memory_learning_writes",
    "public_writes",
)
W08_INFERENCE_CONSUMER_STATES = (
    "RESOLVED",
    "UNKNOWN",
    "CLARIFY",
    "BUDGET_EXHAUSTED",
    "FAIL_CLOSED",
)
W08_INFERENCE_COMPONENT_STATES = ("ACTIVE", "DISABLED", "FAIL_CLOSED")
W08_INFERENCE_STATE_POLICIES = (
    "SAMPLE_FAMILY_STATE",
    "EVIDENCE_BITS_STATE",
    "SOURCE_RECEIPT_STATE",
)
W08_INFERENCE_RENDER_POLICIES = (
    "COPY_VISIBLE_TEXT",
    "NO_TEXT",
    "STRUCTURAL_GENERATOR",
    "STRUCTURED_PAYLOAD",
)
W08_INFERENCE_FAILURE_KINDS = (
    "INPUT_CONTRACT_REJECTED",
    "OUTPUT_CONTRACT_REJECTED",
    "PAYLOAD_KIND_UNSUPPORTED",
    "RENDER_INPUT_REJECTED",
    "RENDER_POLICY_UNSUPPORTED",
    "SCHEMA_UNSEEN",
    "SELECTOR_MISSING",
    "SELECTOR_UNSEEN",
    "STATE_INPUT_REJECTED",
    "STATE_POLICY_UNSUPPORTED",
)


class W08CandidateInferenceError(RuntimeError):
    """Candidate state、调用或输出违反冻结 inference 合同。"""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "OUTPUT_CONTRACT_REJECTED",
    ) -> None:
        if reason_code not in W08_INFERENCE_FAILURE_KINDS:
            raise ValueError("Candidate inference failure kind 未注册")
        super().__init__(message)
        self.reason_code = reason_code


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _strict_sha256(value: object, *, where: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise W08CandidateInferenceError(f"{where} 不是规范 SHA-256")
    return value


def _strict_key(value: object, *, where: str) -> tuple[int, ...]:
    if (
        not isinstance(value, tuple)
        or not value
        or any(type(item) is not int for item in value)
    ):
        raise W08CandidateInferenceError(f"{where} 不是非空整数 key")
    return value


def _schema_shape(value: object) -> object:
    if isinstance(value, dict):
        return {
            "object": [
                [
                    str(key),
                    (
                        {"opaque_canonical_json": 1}
                        if key == "raw_observation"
                        else _schema_shape(item)
                    ),
                ]
                for key, item in sorted(value.items())
            ]
        }
    if isinstance(value, list):
        shapes = {
            _sha256(canonical_json_bytes(_schema_shape(item)))
            for item in value
        }
        return {"array": sorted(shapes), "count_class": int(bool(value))}
    if value is None:
        return "null"
    if type(value) is bool:
        return "bool"
    if type(value) is int:
        return "int"
    if isinstance(value, str):
        return "text"
    raise W08CandidateInferenceError("Candidate inference schema value 类型非法")


def w08_inference_schema_sha256(value: object) -> str:
    """承诺 inference envelope；来源专属 raw carrier 由其 SHA 承诺。"""
    return _sha256(canonical_json_bytes(_schema_shape(value)))


def _rule_identity(
    *,
    payload_kind: str,
    selector_key: str,
    state_policy: str,
    render_policy: str,
    operation_key: str,
    schema_sha256: str,
    evidence_keys: tuple[tuple[int, ...], ...],
) -> dict[str, object]:
    return {
        "evidence_keys": [list(item) for item in evidence_keys],
        "operation_key": operation_key,
        "payload_kind": payload_kind,
        "render_policy": render_policy,
        "schema_sha256": schema_sha256,
        "selector_key": selector_key,
        "state_policy": state_policy,
    }


@dataclass(frozen=True)
class W08CandidateInferenceRule:
    """一个由 train Evidence 验证的 typed 可执行规则。"""

    payload_kind: str
    selector_key: str
    state_policy: str
    render_policy: str
    operation_key: str
    schema_sha256: str
    evidence_keys: tuple[tuple[int, ...], ...]
    rule_key: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.payload_kind not in W08_INFERENCE_PAYLOAD_KINDS:
            raise W08CandidateInferenceError("inference rule payload kind 非法")
        if not self.selector_key or not self.operation_key:
            raise W08CandidateInferenceError("inference rule selector/operation 为空")
        if self.state_policy not in W08_INFERENCE_STATE_POLICIES:
            raise W08CandidateInferenceError("inference rule state policy 非法")
        if self.render_policy not in W08_INFERENCE_RENDER_POLICIES:
            raise W08CandidateInferenceError("inference rule render policy 非法")
        _strict_sha256(self.schema_sha256, where="inference rule schema")
        if (
            not self.evidence_keys
            or tuple(sorted(set(self.evidence_keys))) != self.evidence_keys
        ):
            raise W08CandidateInferenceError("inference rule Evidence identity 非法")
        for item in self.evidence_keys:
            _strict_key(item, where="inference rule Evidence")
        expected = digest_value(_rule_identity(
            payload_kind=self.payload_kind,
            selector_key=self.selector_key,
            state_policy=self.state_policy,
            render_policy=self.render_policy,
            operation_key=self.operation_key,
            schema_sha256=self.schema_sha256,
            evidence_keys=self.evidence_keys,
        ))
        if self.rule_key != expected:
            raise W08CandidateInferenceError("inference rule commitment 漂移")

    def to_dict(self) -> dict[str, object]:
        return {
            **_rule_identity(
                payload_kind=self.payload_kind,
                selector_key=self.selector_key,
                state_policy=self.state_policy,
                render_policy=self.render_policy,
                operation_key=self.operation_key,
                schema_sha256=self.schema_sha256,
                evidence_keys=self.evidence_keys,
            ),
            "rule_key": list(self.rule_key),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "W08CandidateInferenceRule":
        return cls(
            str(value["payload_kind"]),
            str(value["selector_key"]),
            str(value["state_policy"]),
            str(value["render_policy"]),
            str(value["operation_key"]),
            str(value["schema_sha256"]),
            tuple(tuple(int(part) for part in item) for item in value["evidence_keys"]),
            tuple(int(item) for item in value["rule_key"]),
        )


@dataclass(frozen=True)
class W08CandidateInferenceState:
    """Candidate dump 中不含答案或文本样本的可执行 state。"""

    interface_version: str
    rules: tuple[W08CandidateInferenceRule, ...]
    component_keys: tuple[str, ...]
    consumer_keys: tuple[str, ...]
    training_record_count: int
    training_evidence_count: int
    training_identity_key: tuple[int, ...]
    state_key: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.interface_version != W08_CANDIDATE_INFERENCE_INTERFACE_VERSION:
            raise W08CandidateInferenceError("Candidate inference interface version 漂移")
        if (
            not self.rules
            or tuple(sorted(self.rules, key=lambda item: (
                item.payload_kind, item.selector_key
            ))) != self.rules
            or len({(item.payload_kind, item.selector_key) for item in self.rules})
            != len(self.rules)
        ):
            raise W08CandidateInferenceError("Candidate inference rule inventory 漂移")
        if self.component_keys != W08_DIMENSION_KEYS:
            raise W08CandidateInferenceError("Candidate inference component inventory 漂移")
        if self.consumer_keys != W08_CONSUMER_KEYS:
            raise W08CandidateInferenceError("Candidate inference consumer inventory 漂移")
        if (
            type(self.training_record_count) is not int
            or type(self.training_evidence_count) is not int
            or self.training_record_count <= 0
            or self.training_record_count != self.training_evidence_count
        ):
            raise W08CandidateInferenceError("Candidate inference training count 非法")
        _strict_key(self.training_identity_key, where="inference training identity")
        expected = digest_value(self._identity_dict())
        if self.state_key != expected:
            raise W08CandidateInferenceError("Candidate inference state commitment 漂移")
        encoded = canonical_json_bytes(self.to_dict())
        forbidden = (
            b'"expected',
            b'"label',
            b'"surface',
            b'"observed_surface"',
            b'"accepted_surfaces"',
        )
        if any(token in encoded for token in forbidden):
            raise W08CandidateInferenceError("Candidate inference state 泄露答案或表层样本")

    def _identity_dict(self) -> dict[str, object]:
        return {
            "artifact_kind": W08_CANDIDATE_INFERENCE_STATE_KIND,
            "component_keys": list(self.component_keys),
            "consumer_keys": list(self.consumer_keys),
            "format_version": 1,
            "input_kind": W08_CANDIDATE_INFERENCE_INPUT_KIND,
            "interface_version": self.interface_version,
            "output_kind": W08_CANDIDATE_INFERENCE_OUTPUT_KIND,
            "rules": [item.to_dict() for item in self.rules],
            "training_evidence_count": self.training_evidence_count,
            "training_identity_key": list(self.training_identity_key),
            "training_record_count": self.training_record_count,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._identity_dict(), "state_key": list(self.state_key)}

    def sha256(self) -> str:
        return _sha256(canonical_json_bytes(self.to_dict()))

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "W08CandidateInferenceState":
        if (
            value.get("artifact_kind") != W08_CANDIDATE_INFERENCE_STATE_KIND
            or value.get("format_version") != 1
            or value.get("input_kind") != W08_CANDIDATE_INFERENCE_INPUT_KIND
            or value.get("output_kind") != W08_CANDIDATE_INFERENCE_OUTPUT_KIND
        ):
            raise W08CandidateInferenceError("Candidate inference state envelope 漂移")
        return cls(
            str(value["interface_version"]),
            tuple(W08CandidateInferenceRule.from_dict(item) for item in value["rules"]),
            tuple(str(item) for item in value["component_keys"]),
            tuple(str(item) for item in value["consumer_keys"]),
            int(value["training_record_count"]),
            int(value["training_evidence_count"]),
            tuple(int(item) for item in value["training_identity_key"]),
            tuple(int(item) for item in value["state_key"]),
        )


@dataclass(frozen=True)
class W08CandidateInferenceOutcome:
    """Candidate adapter 对一个 Observation、一个 bearing 的实际输出。"""

    dimension_key: str
    observation_key: tuple[int, ...]
    actual_state: str
    actual_payload: CanonicalJsonObject
    actual_payload_sha256: str
    consumer_states: tuple[tuple[str, str], ...]
    component_state: str
    component_receipt_key: tuple[int, ...]
    source_key: tuple[int, ...]
    scope_key: tuple[int, ...]
    reference_key: tuple[int, ...]
    shortcut_counts: tuple[tuple[str, int], ...]
    owner_counts: tuple[tuple[str, int], ...]
    input_text_reads: int
    state_commitment_sha256: str
    invocation_key: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.dimension_key not in W08_DIMENSION_KEYS:
            raise W08CandidateInferenceError("Candidate inference dimension 非法")
        _strict_key(self.observation_key, where="inference Observation")
        if self.actual_state not in EXPECTED_STATES:
            raise W08CandidateInferenceError("Candidate inference actual state 非法")
        if not isinstance(self.actual_payload, CanonicalJsonObject):
            raise W08CandidateInferenceError("Candidate inference actual payload 类型非法")
        if self.actual_payload_sha256 != _sha256(
            canonical_json_bytes(self.actual_payload.to_value())
        ):
            raise W08CandidateInferenceError("Candidate inference payload commitment 漂移")
        if (
            tuple(key for key, _ in self.consumer_states) != W08_CONSUMER_KEYS
            or any(state not in W08_INFERENCE_CONSUMER_STATES for _, state in self.consumer_states)
        ):
            raise W08CandidateInferenceError("Candidate inference U/R/G outcome 非法")
        if self.component_state not in W08_INFERENCE_COMPONENT_STATES:
            raise W08CandidateInferenceError("Candidate inference component state 非法")
        for value, where in (
            (self.component_receipt_key, "component receipt"),
            (self.source_key, "source"),
            (self.scope_key, "scope"),
            (self.reference_key, "reference"),
            (self.invocation_key, "invocation"),
        ):
            _strict_key(value, where=where)
        if (
            tuple(key for key, _ in self.shortcut_counts) != W08_INFERENCE_SHORTCUT_KEYS
            or any(type(value) is not int or value < 0 for _, value in self.shortcut_counts)
        ):
            raise W08CandidateInferenceError("Candidate inference shortcut account 非法")
        if (
            tuple(key for key, _ in self.owner_counts) != W08_INFERENCE_OWNER_COUNT_KEYS
            or any(type(value) is not int or value != 0 for _, value in self.owner_counts)
        ):
            raise W08CandidateInferenceError("Candidate inference owner write/read account 非零")
        if type(self.input_text_reads) is not int or self.input_text_reads not in {0, 1}:
            raise W08CandidateInferenceError("Candidate inference visible input read 计数非法")
        _strict_sha256(self.state_commitment_sha256, where="inference state")

    def safe_commitment_dict(self) -> dict[str, object]:
        """只导出不含 private payload 的 invocation commitment。"""
        return {
            "actual_payload_sha256": self.actual_payload_sha256,
            "actual_state": self.actual_state,
            "component_receipt_key": list(self.component_receipt_key),
            "component_state": self.component_state,
            "consumer_states": [list(item) for item in self.consumer_states],
            "dimension_key": self.dimension_key,
            "invocation_key": list(self.invocation_key),
            "observation_key": list(self.observation_key),
            "owner_counts": [list(item) for item in self.owner_counts],
            "reference_key": list(self.reference_key),
            "scope_key": list(self.scope_key),
            "shortcut_counts": [list(item) for item in self.shortcut_counts],
            "source_key": list(self.source_key),
            "state_commitment_sha256": self.state_commitment_sha256,
        }


def make_w08_inference_rule(
    *,
    payload_kind: str,
    selector_key: str,
    state_policy: str,
    render_policy: str,
    operation_key: str,
    schema_sha256: str,
    evidence_keys: tuple[tuple[int, ...], ...],
) -> W08CandidateInferenceRule:
    identity = _rule_identity(
        payload_kind=payload_kind,
        selector_key=selector_key,
        state_policy=state_policy,
        render_policy=render_policy,
        operation_key=operation_key,
        schema_sha256=schema_sha256,
        evidence_keys=evidence_keys,
    )
    return W08CandidateInferenceRule(
        payload_kind,
        selector_key,
        state_policy,
        render_policy,
        operation_key,
        schema_sha256,
        evidence_keys,
        digest_value(identity),
    )


def make_w08_inference_state(
    rules: tuple[W08CandidateInferenceRule, ...],
    *,
    training_record_count: int,
    training_evidence_count: int,
    training_identity_key: tuple[int, ...],
) -> W08CandidateInferenceState:
    ordered = tuple(sorted(rules, key=lambda item: (item.payload_kind, item.selector_key)))
    partial = W08CandidateInferenceState.__new__(W08CandidateInferenceState)
    object.__setattr__(partial, "interface_version", W08_CANDIDATE_INFERENCE_INTERFACE_VERSION)
    object.__setattr__(partial, "rules", ordered)
    object.__setattr__(partial, "component_keys", W08_DIMENSION_KEYS)
    object.__setattr__(partial, "consumer_keys", W08_CONSUMER_KEYS)
    object.__setattr__(partial, "training_record_count", training_record_count)
    object.__setattr__(partial, "training_evidence_count", training_evidence_count)
    object.__setattr__(partial, "training_identity_key", training_identity_key)
    object.__setattr__(partial, "state_key", (1,))
    state_key = digest_value(partial._identity_dict())
    return W08CandidateInferenceState(
        W08_CANDIDATE_INFERENCE_INTERFACE_VERSION,
        ordered,
        W08_DIMENSION_KEYS,
        W08_CONSUMER_KEYS,
        training_record_count,
        training_evidence_count,
        training_identity_key,
        state_key,
    )


__all__ = [
    "W08_CANDIDATE_INFERENCE_INPUT_KIND",
    "W08_CANDIDATE_INFERENCE_INTERFACE_VERSION",
    "W08_CANDIDATE_INFERENCE_OUTPUT_KIND",
    "W08_CANDIDATE_INFERENCE_STATE_KIND",
    "W08_INFERENCE_OWNER_COUNT_KEYS",
    "W08_INFERENCE_FAILURE_KINDS",
    "W08_INFERENCE_PAYLOAD_KINDS",
    "W08_INFERENCE_SHORTCUT_KEYS",
    "W08CandidateInferenceError",
    "W08CandidateInferenceOutcome",
    "W08CandidateInferenceRule",
    "W08CandidateInferenceState",
    "make_w08_inference_rule",
    "make_w08_inference_state",
    "w08_inference_schema_sha256",
]
