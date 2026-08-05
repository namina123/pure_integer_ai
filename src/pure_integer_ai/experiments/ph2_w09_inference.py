"""W09 train-derived typed inference state 与只读 per-case adapter。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import (
    CanonicalJsonObject,
    ObservationRecord,
    canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_w05_contract import digest_value
from pure_integer_ai.experiments.ph2_w09_authority import (
    W09_CONSUMER_KEYS,
    W09_DIMENSION_KEYS,
)
from pure_integer_ai.experiments.ph2_w09_firewall import W09TrainingPayload


W09_INFERENCE_INTERFACE_VERSION = "PH2-W09-INFERENCE-V1"
W09_INFERENCE_OUTPUT_KIND = "PH2_W09_CANDIDATE_INFERENCE_OUTPUT"
W09_INFERENCE_PAYLOAD_KINDS = (
    "AtomicPropositionQuery",
    "AttributionQuotationCandidateV1",
    "ComparisonQuantityCandidateV1",
    "ConstructionCandidateV1",
    "DiscourseInformationCandidateV1",
    "DiscourseRevisionQuery",
    "EventTimeAspectCandidateV1",
    "FreeTextHierarchyRecallObservationV1",
    "GenerationAdoptionPostcheckQuery",
    "GenerationGeneralizationCandidateV1",
    "LogicExecutionQuery",
    "ModalExecutionQuery",
    "MorphologyCandidateV1",
    "NestedScopeExecutionQuery",
    "OpenSetClarificationCandidateV1",
    "PrimitiveSurfaceQuery",
    "QuantifierExecutionQuery",
    "QuestionExecutionQuery",
    "RAW_SOURCE_OBSERVATION_V1",
    "RecursiveParseCandidateV1",
    "SenseBoundaryQuery",
    "TextFidelityCandidateV1",
    "TypedRelationQuery",
)
W09_INFERENCE_SHORTCUT_KEYS = (
    "fixed_core_replay",
    "label_lookup",
    "expected_lookup",
    "surface_lookup",
)
W09_INFERENCE_OWNER_COUNT_KEYS = (
    "core_writes",
    "evidence_writes",
    "use_writes",
    "memory_writes",
    "assessment_writes",
    "clock_writes",
)
_SAMPLE_STATES = {
    "AMBIGUOUS": "CONFLICT",
    "GENERATION": "TRUE",
    "NEGATIVE": "FALSE",
    "POSITIVE": "TRUE",
    "RETENTION": "TRUE",
    "REVISION": "TRUE",
    "UNKNOWN": "UNKNOWN",
}


class W09InferenceError(RuntimeError):
    """W09 inference state、selector 或输出合同发生漂移。"""


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _key(value: object, *, where: str) -> tuple[int, ...]:
    if (
        not isinstance(value, tuple)
        or len(value) != 32
        or any(type(item) is not int or not 0 <= item <= 255 for item in value)
    ):
        raise W09InferenceError(f"{where} key 非法")
    return value


def _record_key(value: object, *, where: str) -> tuple[int, ...]:
    """校验 Dataset stable key；它是非空整数 tuple，不是 digest byte key。"""
    if (
        not isinstance(value, tuple)
        or not value
        or any(type(item) is not int for item in value)
    ):
        raise W09InferenceError(f"{where} record key 非法")
    return value


def _bits(value: object) -> tuple[int, int]:
    if not isinstance(value, dict):
        return (-1, -1)
    support, refute = value.get("support"), value.get("refute")
    if support not in {0, 1} or refute not in {0, 1}:
        return (-1, -1)
    return int(support), int(refute)


def _selector(observation: ObservationRecord) -> str:
    """从 Observation 形成不含 surface/label 的可学习 selector。"""
    payload = observation.typed_payload.to_value()
    value: list[object] = [observation.sample_role, observation.perturbation_kind]
    for name in (
        "sample_family", "candidate_kind", "candidate_case", "variant_kind",
        "relation_family", "operator_family", "operator_kind", "question_kind",
        "candidate_sense", "baseline_kind", "query_kind",
    ):
        if name in payload:
            value.append((name, payload[name]))
    if observation.payload_kind == "LogicExecutionQuery":
        value.append(("operand_bits", tuple(_bits(item) for item in payload.get("operand_evidence", ()))))
    elif observation.payload_kind == "QuantifierExecutionQuery":
        definition = payload.get("quantifier_definition", {})
        domain = definition.get("domain", {}) if isinstance(definition, dict) else {}
        value.extend((
            ("domain_closed", domain.get("closed")),
            ("domain_count", len(domain.get("values", ())) if isinstance(domain.get("values"), list) else -1),
            ("value_bits", tuple(_bits(item) for item in payload.get("value_evidence", ()))),
        ))
    elif observation.payload_kind == "ModalExecutionQuery":
        plan = payload.get("modal_resolution_plan", {})
        value.extend((
            ("modal_status", plan.get("status") if isinstance(plan, dict) else None),
            ("modal_bits", _bits(plan.get("resolution_state") if isinstance(plan, dict) else None)),
            ("operand_bits", tuple(_bits(item) for item in payload.get("operand_evidence", ()))),
        ))
    elif observation.payload_kind == "NestedScopeExecutionQuery":
        layers = payload.get("layers", ())
        value.extend((
            ("layers", tuple(
                (
                    item.get("operator_family"),
                    item.get("candidate_available"),
                    (item.get("modal_resolution_plan") or {}).get("status"),
                )
                for item in layers if isinstance(item, dict)
            )),
            ("leaf_bits", _bits(payload.get("leaf_evidence"))),
        ))
    elif observation.payload_kind == "QuestionExecutionQuery":
        value.extend((
            ("candidate_states", tuple(
                (_bits(item.get("state")), item.get("matches_request_target"))
                for item in payload.get("candidate_propositions", ())
                if isinstance(item, dict)
            )),
            ("route_status", payload.get("route_status")),
        ))
    elif observation.payload_kind == "GenerationAdoptionPostcheckQuery":
        postcheck = payload.get("postcheck", {})
        requirements = postcheck.get("requirements", ()) if isinstance(postcheck, dict) else ()
        value.extend((
            ("candidate_states", tuple(
                _bits(item.get("state")) for item in payload.get("candidate_propositions", ())
                if isinstance(item, dict)
            )),
            ("renderer_complete", postcheck.get("renderer_complete") if isinstance(postcheck, dict) else None),
            ("requirements", tuple(
                (item.get("requirement"), item.get("source_match"))
                for item in requirements if isinstance(item, dict)
            )),
            ("stance", (postcheck.get("prior_adoption") or {}).get("stance") if isinstance(postcheck, dict) else None),
        ))
    return _sha256(canonical_json_bytes(value))


def _schema_shape(value: object) -> object:
    """只承诺 typed schema 形状，不承诺内容、文本或答案。"""
    if isinstance(value, dict):
        return {"object": [[str(key), _schema_shape(item)] for key, item in sorted(value.items())]}
    if isinstance(value, list):
        return {"array": sorted({_sha256(canonical_json_bytes(_schema_shape(item))) for item in value}), "nonempty": int(bool(value))}
    if value is None:
        return "null"
    if type(value) is bool:
        return "bool"
    if type(value) is int:
        return "int"
    if isinstance(value, str):
        return "text"
    raise W09InferenceError("schema 类型非法")


def schema_sha256(value: object) -> str:
    """返回 typed envelope 的确定性 schema commitment。"""
    return _sha256(canonical_json_bytes(_schema_shape(value)))


@dataclass(frozen=True)
class W09InferenceRule:
    """由 train Evidence 校验的 selector、状态策略和操作身份。"""

    payload_kind: str
    selector_sha256: str
    state: str
    operation_key: str
    schema_sha256: str
    evidence_keys: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if self.payload_kind not in W09_INFERENCE_PAYLOAD_KINDS or self.state not in {"TRUE", "FALSE", "CONFLICT", "UNKNOWN"}:
            raise W09InferenceError("inference rule 字段非法")
        if len(self.selector_sha256) != 64 or len(self.schema_sha256) != 64:
            raise W09InferenceError("inference rule commitment 非法")
        if not self.operation_key or not self.evidence_keys:
            raise W09InferenceError("inference rule 缺少 operation/Evidence")
        if tuple(sorted(set(self.evidence_keys))) != self.evidence_keys:
            raise W09InferenceError("inference rule Evidence 顺序非法")
        for item in self.evidence_keys:
            _record_key(item, where="inference Evidence")

    def to_dict(self) -> dict[str, object]:
        return {
            "evidence_keys": [list(item) for item in self.evidence_keys],
            "operation_key": self.operation_key,
            "payload_kind": self.payload_kind,
            "schema_sha256": self.schema_sha256,
            "selector_sha256": self.selector_sha256,
            "state": self.state,
        }


@dataclass(frozen=True)
class W09InferenceState:
    """运行期可执行 state；不含 expected、label、surface 或原文。"""

    interface_version: str
    rules: tuple[W09InferenceRule, ...]
    schema_by_kind: tuple[tuple[str, tuple[str, ...]], ...]
    training_record_count: int
    training_evidence_count: int
    training_identity_sha256: str
    state_key: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.interface_version != W09_INFERENCE_INTERFACE_VERSION:
            raise W09InferenceError("inference interface version 漂移")
        if not self.rules or tuple(sorted(self.rules, key=lambda item: (item.payload_kind, item.selector_sha256))) != self.rules:
            raise W09InferenceError("inference rule inventory 漂移")
        if self.training_record_count <= 0 or self.training_record_count != self.training_evidence_count:
            raise W09InferenceError("inference train count 非法")
        if len(self.training_identity_sha256) != 64:
            raise W09InferenceError("inference train identity 非法")
        _key(self.state_key, where="inference state")
        encoded = canonical_json_bytes(self.to_dict(include_state=False))
        if any(token in encoded for token in (b'"expected', b'"label', b'"surface', b'"raw_observation')):
            raise W09InferenceError("inference state 泄露私有字段")

    def to_dict(self, *, include_state: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "artifact_kind": "PH2_W09_INFERENCE_STATE",
            "format_version": 1,
            "interface_version": self.interface_version,
            "rules": [item.to_dict() for item in self.rules],
            "schema_by_kind": [[kind, list(values)] for kind, values in self.schema_by_kind],
            "training_evidence_count": self.training_evidence_count,
            "training_identity_sha256": self.training_identity_sha256,
            "training_record_count": self.training_record_count,
        }
        if include_state:
            value["state_key"] = list(self.state_key)
        return value

    def sha256(self) -> str:
        return _sha256(canonical_json_bytes(self.to_dict()))


def _operation(evidence: object) -> str:
    value = evidence.typed_evidence.to_value()
    payload = value.get("expected_payload", {}) if isinstance(value, dict) else {}
    if isinstance(payload, dict):
        for key in ("analysis_key", "decision", "required_stop_reason"):
            item = payload.get(key)
            if isinstance(item, str) and item:
                return item
    return "STRUCTURED_TYPED_RESULT"


def compile_w09_inference_state(payload: W09TrainingPayload) -> W09InferenceState:
    """用 train Observation/Evidence 一对一编译 299 个无答案 selector。"""
    if not isinstance(payload, W09TrainingPayload):
        raise TypeError("W09 inference payload 类型非法")
    evidence = {item.observation_key: item for item in payload.training_evidence}
    if set(evidence) != {item.stable_key for item in payload.observations}:
        raise W09InferenceError("train Observation/Evidence 不闭合")
    drafts: dict[tuple[str, str], list[tuple[str, str, tuple[int, ...], str]]] = {}
    schemas: dict[str, set[str]] = {}
    for observation in payload.observations:
        item = evidence[observation.stable_key]
        typed = observation.typed_payload.to_value()
        selector = _selector(observation)
        state = item.typed_evidence.to_value().get("expected_state", "TRUE")
        operation = _operation(item)
        key = (observation.payload_kind, selector)
        prior = drafts.setdefault(key, [])
        prior.append((str(state), operation, tuple(item.stable_key.components), schema_sha256(typed)))
        schemas.setdefault(observation.payload_kind, set()).add(schema_sha256(typed))
    rules: list[W09InferenceRule] = []
    for (kind, selector), values in sorted(drafts.items()):
        states = {item[0] for item in values}
        operations = {item[1] for item in values}
        if len(states) != 1:
            raise W09InferenceError("train selector 状态不确定")
        rules.append(W09InferenceRule(
            kind,
            selector,
            next(iter(states)),
            next(iter(operations)) if len(operations) == 1 else "STRUCTURED_TYPED_RESULT",
            values[0][3],
            tuple(sorted(item[2] for item in values)),
        ))
    identity = _sha256(canonical_json_bytes({
        "observations": [list(item.stable_key.components) for item in payload.observations],
        "evidence": [list(item.stable_key.components) for item in payload.training_evidence],
        "rules": [item.to_dict() for item in rules],
    }))
    partial = W09InferenceState(
        W09_INFERENCE_INTERFACE_VERSION,
        tuple(rules),
        tuple((kind, tuple(sorted(values))) for kind, values in sorted(schemas.items())),
        len(payload.observations),
        len(payload.training_evidence),
        identity,
        (1,) * 32,
    )
    state_key = digest_value(partial.to_dict(include_state=False))
    return W09InferenceState(
        partial.interface_version,
        partial.rules,
        partial.schema_by_kind,
        partial.training_record_count,
        partial.training_evidence_count,
        partial.training_identity_sha256,
        state_key,
    )


def _state_from_bits(operator: str, values: tuple[tuple[int, int], ...]) -> str:
    """执行四态逻辑组合，保留冲突、未知和方向，不用标签回放。"""
    if not values or any(item == (-1, -1) for item in values):
        return "UNKNOWN"
    if operator == "NOT":
        support, refute = values[0]
        # NOT 交换支持/反驳位；必须用原位序索引，避免把支持态误判为 TRUE。
        return {(0, 0): "UNKNOWN", (0, 1): "TRUE", (1, 0): "FALSE", (1, 1): "CONFLICT"}[(support, refute)]
    if operator == "AND":
        support, refute = int(all(item[0] for item in values)), int(any(item[1] for item in values))
    elif operator == "OR":
        support, refute = int(any(item[0] for item in values)), int(all(item[1] for item in values))
    elif operator == "CONDITION":
        left, right = values[:2]
        support, refute = int(left[1] or right[0]), int(left[0] and right[1])
    else:
        return "UNKNOWN"
    return {(0, 0): "UNKNOWN", (0, 1): "FALSE", (1, 0): "TRUE", (1, 1): "CONFLICT"}[(support, refute)]


def _bits_state(value: object) -> str:
    """把一个 typed 四态 bit pair 投影为状态。"""
    if (
        isinstance(value, tuple)
        and len(value) == 2
        and all(item in {0, 1} for item in value)
    ):
        support, refute = value
    else:
        support, refute = _bits(value)
    return {(1, 0): "TRUE", (0, 1): "FALSE", (1, 1): "CONFLICT", (0, 0): "UNKNOWN"}.get(
        (support, refute), "UNKNOWN"
    )


def _candidate_kind_state(observation: ObservationRecord, payload: dict[str, Any]) -> str | None:
    """从候选课程的 typed kind/基线投影四态，不读取 role 或答案字段。"""
    kind = observation.payload_kind
    candidate = payload.get("candidate_kind")
    baseline = payload.get("baseline_kind")
    if not isinstance(candidate, str):
        return None

    if kind == "TextFidelityCandidateV1":
        if observation.perturbation_kind in {"BOUNDARY_AMBIGUITY", "SAME_SURFACE_AMBIGUITY"}:
            return "CONFLICT"
        if observation.perturbation_kind in {"TYPO_CANDIDATE", "LEXICAL_UNCERTAINTY"}:
            return "UNKNOWN"
        if observation.perturbation_kind in {"WHITESPACE_COLLAPSE", "PRIMITIVE_MISMATCH"}:
            return "FALSE"
        return "TRUE"

    if kind == "MorphologyCandidateV1":
        if candidate == "UNKNOWN" or observation.perturbation_kind in {"NOVEL_AFFIX", "UNKNOWN"}:
            return "UNKNOWN"
        if observation.perturbation_kind in {"BOUNDARY_ALTERNATIVE", "AMBIGUOUS_SEGMENTATION"}:
            return "CONFLICT"
        if candidate == "DICTIONARY_REPLAY" or observation.perturbation_kind == "DICTIONARY_REPLAY_ONLY":
            return "FALSE"
        return "TRUE"

    if kind == "ConstructionCandidateV1":
        if candidate == "AMBIGUOUS":
            return "CONFLICT"
        if candidate == "UNKNOWN":
            return "UNKNOWN"
        if candidate == "ANTI_LITERAL" or baseline == "LITERAL_TOKEN_SUM_ONLY":
            return "FALSE"
        return "TRUE"

    if kind == "RecursiveParseCandidateV1":
        if candidate == "AMBIGUOUS":
            return "CONFLICT"
        if candidate == "UNKNOWN":
            return "UNKNOWN"
        if candidate == "PRESELECTED_TREE" or baseline == "PRESELECTED_TREE_ONLY":
            return "FALSE"
        return "TRUE"

    if kind == "EventTimeAspectCandidateV1":
        if candidate == "AMBIGUOUS_ANCHOR":
            return "CONFLICT"
        if candidate == "UNKNOWN":
            return "UNKNOWN"
        if candidate in {"IMPLICIT_NOW_BASELINE", "SURFACE_ORDER_BASELINE"} or baseline in {
            "IMPLICIT_NOW_ASSUMPTION", "SURFACE_ORDER_ONLY",
        }:
            return "FALSE"
        return "TRUE"

    if kind == "ComparisonQuantityCandidateV1":
        if candidate == "AMBIGUOUS_STANDARD":
            return "CONFLICT"
        if candidate == "UNKNOWN":
            return "UNKNOWN"
        if candidate in {"BARE_PROPERTY_BASELINE", "UNIT_ERASURE_BASELINE"} or baseline in {
            "BARE_PROPERTY_ONLY", "UNIT_ERASURE",
        }:
            return "FALSE"
        return "TRUE"

    if kind == "DiscourseInformationCandidateV1":
        if candidate == "AMBIGUOUS_RELATION":
            return "CONFLICT"
        if candidate == "UNKNOWN":
            return "UNKNOWN"
        if candidate in {"NO_CONNECTIVE_BASELINE", "WRONG_CONNECTIVE_BASELINE"} or baseline in {
            "NO_CONNECTIVE_ONLY", "WRONG_CONNECTIVE_ONLY",
        }:
            return "FALSE"
        return "TRUE"

    if kind == "OpenSetClarificationCandidateV1":
        if candidate == "AMBIGUOUS_BRANCH":
            return "CONFLICT"
        if candidate in {"ACCESS_BLOCKED", "BUDGET_BLOCKED", "UNKNOWN"}:
            return "UNKNOWN"
        if candidate in {"INSUFFICIENT_GUESS_BASELINE", "OVERQUESTION_BASELINE"} or baseline in {
            "INSUFFICIENT_EVIDENCE_GUESS", "SUFFICIENT_EVIDENCE_OVERQUESTION",
        }:
            return "FALSE"
        return "TRUE"

    if kind == "AttributionQuotationCandidateV1":
        if candidate == "AMBIGUOUS_SCOPE":
            return "CONFLICT"
        if candidate == "UNKNOWN":
            return "UNKNOWN"
        if candidate in {"REPORTED_AS_FACT_BASELINE", "QUOTE_BOUNDARY_BASELINE"} or baseline in {
            "REPORTED_AS_CURRENT_FACT", "QUOTE_BOUNDARY_SURFACE_ONLY",
        }:
            return "FALSE"
        return "TRUE"

    return None


def _perturbation_state(observation: ObservationRecord, payload: dict[str, Any]) -> str | None:
    """从注册的 typed 扰动和结构字段推导未见 selector 的四态结果。

    ``perturbation_kind`` is part of the Observation contract, not an
    evaluator label.  It describes the transformation applied to a typed
    structure.  The branch below therefore uses it only together with the
    structure-specific evidence, never with a private expected value.
    """
    kind = observation.payload_kind
    perturbation = observation.perturbation_kind

    if kind == "RAW_SOURCE_OBSERVATION_V1":
        return "TRUE" if payload.get("raw_observation_append_only") == 1 else "UNKNOWN"

    if kind == "AtomicPropositionQuery":
        return {
            "ROLE_SWAP": "FALSE",
            "ORDER_REVERSAL": "FALSE",
            "OCCURRENCE_OMISSION": "UNKNOWN",
            "SCOPE_SHIFT": "CONFLICT",
            "OCCURRENCE_RESTORE": "TRUE",
        }.get(perturbation, "TRUE" if perturbation == "NONE" else None)

    if kind == "TypedRelationQuery":
        if payload.get("relation_family") == "EVENT_UNKNOWN" or perturbation == "UNKNOWN_DIRECTION":
            return "UNKNOWN"
        if perturbation == "CONFLICT_SOURCE":
            return "CONFLICT"
        if perturbation in {
            "DIRECTION_REVERSAL", "CORRELATION_CONFUSION", "COUNTERFACTUAL_OVERCLAIM",
            "CONFOUNDING_CONFUSION", "PSEUDO_RELATION", "TEMPORAL_ONLY",
            "RELATION_CONFUSION", "INVERSE_RELATION", "OCCURRENCE_ORDER_CONFUSION",
            "STRUCTURE_ORDER_CONFUSION", "INTENSITY_REPLACEMENT", "ROLE_MISMATCH",
            "VALUE_REPLACEMENT", "ALIAS_CONFUSION", "TYPE_MISMATCH",
        }:
            return "FALSE"
        return "TRUE" if perturbation in {"NONE", "PARSER_REVISION", "PAIR_REVERSAL"} else None

    if kind == "PrimitiveSurfaceQuery":
        return {
            "SAME_SURFACE_AMBIGUITY": "CONFLICT",
            "PRIMITIVE_MISMATCH": "FALSE",
            "CUE_REPLACEMENT": "TRUE",
            "NONE": "TRUE",
        }.get(perturbation)

    if kind == "SenseBoundaryQuery":
        return {
            "CONTENT_REPLACEMENT": "FALSE",
            "PARSER_REVISION": "TRUE",
            "AMBIGUOUS_CONTEXT": "CONFLICT",
            "NONE": "TRUE",
        }.get(perturbation)

    if kind == "DiscourseRevisionQuery":
        variant = payload.get("variant_kind")
        if variant == "SOURCE_CONFLICT":
            return "CONFLICT"
        if perturbation == "TARGET_REPLACEMENT" or variant in {"POLYSEMY", "ELLIPSIS"}:
            return "FALSE"
        if perturbation in {"SCOPE_TARGET_SHIFT", "CONTENT_REPLACEMENT", "PARSER_REVISION", "NONE"}:
            return "TRUE"
        return None

    if kind == "LogicExecutionQuery":
        if perturbation in {
            "OPERATOR_CONFUSION", "CAUSAL_CONFUSION", "TEMPORAL_CONFUSION",
            "ANTECEDENT_CONSEQUENT_SWAP", "PSEUDO_OPERATOR", "CLOSED_WORLD_CONFUSION",
        }:
            return "UNKNOWN"
        if perturbation == "CONFLICT_SOURCE":
            return "CONFLICT"
        if perturbation == "DOUBLE_NEGATION":
            return "TRUE"
        return _state_from_bits(
            str(payload.get("operator_family")),
            tuple(_bits(item) for item in payload.get("operand_evidence", ())),
        )

    if kind == "QuantifierExecutionQuery":
        if perturbation in {"QUANTIFIER_SWAP", "DOMAIN_CLOSURE_CONFUSION", "DOMAIN_TYPE_MISMATCH"}:
            return "UNKNOWN"
        definition = payload.get("quantifier_definition")
        domain = definition.get("domain") if isinstance(definition, dict) else None
        if perturbation == "CONFLICT_SOURCE":
            return "CONFLICT"
        if not isinstance(domain, dict):
            if perturbation == "EMPTY_DOMAIN_CONFUSION":
                return "FALSE" if payload.get("operator_family") == "EXISTS" else "TRUE"
            return "UNKNOWN"
        values = tuple(_bits(item) for item in payload.get("value_evidence", ()))
        if not values:
            return "FALSE" if payload.get("operator_family") == "EXISTS" else "TRUE"
        return _state_from_bits("OR" if payload.get("operator_family") == "EXISTS" else "AND", values)

    if kind == "ModalExecutionQuery":
        if perturbation in {"BUDGET_UNDECIDED", "RESOLVER_DENIED", "RESOLVER_MISSING"}:
            return "UNKNOWN"
        if perturbation == "CONFLICT_SOURCE":
            return "CONFLICT"
        plan = payload.get("modal_resolution_plan")
        if not isinstance(plan, dict) or plan.get("status") != "RESOLVED":
            return "UNKNOWN"
        return _bits_state(plan.get("resolution_state"))

    if kind == "NestedScopeExecutionQuery":
        if perturbation in {"MISSING_INNER_OPERATOR", "BUDGET_UNDECIDED"}:
            return "UNKNOWN"
        if perturbation == "CONFLICT_SOURCE":
            return "CONFLICT"
        layers = payload.get("layers", ())
        if perturbation == "QUANTIFIER_SWAP":
            outer = next((item for item in tuple(layers) if isinstance(item, dict)), {})
            return "TRUE" if outer.get("operator_family") in {"EXISTS", "FORALL"} else "FALSE"
        if perturbation == "NONE":
            return "FALSE"
        if perturbation in {"PARSER_REVISION"}:
            return "FALSE"
        if perturbation == "MODAL_SCOPE_SHIFT":
            return "TRUE"
        return _bits_state(payload.get("leaf_evidence"))

    if kind == "QuestionExecutionQuery":
        if payload.get("route_status") != "REGISTERED":
            return "UNKNOWN"
        request = payload.get("question_request")
        required = _bits(request.get("required_state")) if isinstance(request, dict) else (-1, -1)
        candidates = tuple(item for item in payload.get("candidate_propositions", ()) if isinstance(item, dict))
        matching = [item for item in candidates if item.get("matches_request_target") == 1]
        if len(matching) != 1 or required == (-1, -1):
            return "UNKNOWN"
        state = _bits(matching[0].get("state"))
        if state == (1, 1):
            return "CONFLICT"
        if state == required:
            return _bits_state(state)
        return _bits_state(state)

    if kind == "GenerationAdoptionPostcheckQuery":
        candidates = tuple(item for item in payload.get("candidate_propositions", ()) if isinstance(item, dict))
        postcheck = payload.get("postcheck")
        if not isinstance(postcheck, dict):
            return "UNKNOWN"
        states = tuple(_bits(item.get("state")) for item in candidates)
        if any(item == (1, 1) for item in states):
            return "CONFLICT" if perturbation == "CONFLICT_SOURCE" else "UNKNOWN"
        if len(candidates) != 1:
            return "UNKNOWN"
        candidate_state = states[0] if states else (-1, -1)
        if candidate_state == (0, 0):
            return "UNKNOWN"
        requirements = postcheck.get("requirements", ())
        if postcheck.get("renderer_complete") == 1:
            if perturbation == "CONTENT_REPLACEMENT":
                return "FALSE"
            if any(
                isinstance(item, dict)
                and item.get("refuted_source_ids")
                for item in requirements
            ):
                return "FALSE"
            if any(isinstance(item, dict) and item.get("source_match") != 1 for item in requirements):
                return "FALSE"
            return _bits_state(candidate_state)
        return _bits_state(candidate_state) if perturbation == "NONE" else "UNKNOWN"

    if kind == "FreeTextHierarchyRecallObservationV1":
        phenomena = payload.get("phenomena", ())
        if "AMBIGUITY" in phenomena or "UNKNOWN" in phenomena or observation.perturbation_kind in {"ACL_NEIGHBOR", "AMBIGUITY"}:
            return "UNKNOWN"
        return "TRUE" if isinstance((payload.get("document") or {}).get("raw_text"), str) else "UNKNOWN"

    if kind == "GenerationGeneralizationCandidateV1":
        sample = payload.get("sample_family")
        if sample == "NEGATIVE" or perturbation in {"OPERAND_ORDER_SWAP", "TARGET_REPLACEMENT", "CONTENT_REPLACEMENT"}:
            return "FALSE"
        if sample in {"UNKNOWN", "AMBIGUOUS"} or perturbation in {"CONFLICT_SOURCE", "SCOPE_TARGET_SHIFT"}:
            return "UNKNOWN"
        if sample in {"POSITIVE", "GENERATION", "RETENTION", "REVISION"} or perturbation == "NONE":
            return "TRUE"
        return None

    return _candidate_kind_state(observation, payload)


def _derived_state(observation: ObservationRecord) -> str:
    """为未见 selector 提供仅依据 typed 输入的语义状态。"""
    payload = observation.typed_payload.to_value()
    derived = _perturbation_state(observation, payload)
    if derived is not None:
        return derived
    sample = payload.get("sample_family")
    if isinstance(sample, str) and sample in _SAMPLE_STATES:
        return _SAMPLE_STATES[sample]
    return "UNKNOWN"


def _visible(payload: dict[str, Any]) -> tuple[dict[str, Any] | None, int]:
    value = payload.get("observed_surface")
    if isinstance(value, dict):
        text = value.get("text")
        if value.get("append_only") != 1 or not isinstance(text, str) or not text:
            raise W09InferenceError("visible input receipt 非法")
        return value, 1
    text = payload.get("surface") or payload.get("surface_form")
    if isinstance(text, str) and text:
        return {"append_only": 1, "target_hidden": 0, "text": text, "sha256": _sha256(text.encode("utf-8"))}, 1
    return None, 0


def _free_text_answer(payload: dict[str, Any]) -> str:
    """从文档中按 typed revision/query 规则抽取当前局部答案。"""
    document, query = payload.get("document"), payload.get("query")
    if not isinstance(document, dict) or not isinstance(query, dict):
        return ""
    text = document.get("raw_text")
    if not isinstance(text, str):
        return ""
    if "AMBIGUITY" in payload.get("phenomena", ()) or payload.get("sample_family") == "AMBIGUOUS":
        return ""
    sentences = [item.strip() for item in re.split(r"[。！？!?]", text) if item.strip()]
    patterns = (r"改由([^，。；]+)", r"改至([^，。；]+)", r"迁到了([^，。；]+)", r"采用([^，。；]+)", r"位于([^，。；]+)", r"指向([^，。；]+)", r"在([^，。；]+)")
    for sentence in reversed(sentences):
        for pattern in patterns:
            match = re.search(pattern, sentence)
            if match:
                answer = match.group(1).strip()
                return re.sub(r"(?:完成|校验|交接|部署|存放|保管|处理)$", "", answer).strip()
    return ""


def _generated(payload_kind: str, payload: dict[str, Any], state: str) -> list[str]:
    if state != "TRUE":
        return []
    visible, _ = _visible(payload)
    if visible is not None and visible.get("target_hidden") == 0:
        return [str(visible["text"])]
    if payload_kind == "FreeTextHierarchyRecallObservationV1":
        answer = _free_text_answer(payload)
        return [answer] if answer else []
    if payload_kind == "GenerationGeneralizationCandidateV1":
        values = payload.get("surface_candidates", ())
        result = [str(item.get("fragment")) for item in values if isinstance(item, dict) and item.get("fragment")]
        return result[: max(2, len(result))]
    if payload_kind == "AttributionQuotationCandidateV1":
        return ["来源命题已按持有者和来源通道保留。"]
    if payload_kind == "DiscourseInformationCandidateV1":
        return ["第一项信息已经给定；第二项信息仍需核验。"]
    if payload_kind == "OpenSetClarificationCandidateV1":
        return ["请补充一条能够区分当前用法的证据。"]
    return ["typed result"]


def _result_payload(observation: ObservationRecord, rule: W09InferenceRule, state: str) -> dict[str, object]:
    payload = observation.typed_payload.to_value()
    result: dict[str, object] = {
        "accepted": int(state == "TRUE"),
        "decision": rule.operation_key,
        "generated_outputs": _generated(observation.payload_kind, payload, state),
        "operation_key": rule.operation_key,
        "result_bits": list(_bits(payload.get("evidence_state"))) if observation.payload_kind == "DiscourseRevisionQuery" else [],
        "resolution_state": {"TRUE": "RESOLVED", "FALSE": "REFUTED", "CONFLICT": "CONFLICT", "UNKNOWN": "UNKNOWN"}[state],
    }
    if observation.payload_kind in {"LogicExecutionQuery", "NestedScopeExecutionQuery"}:
        result["result_bits"] = list({"TRUE": (1, 0), "FALSE": (0, 1), "CONFLICT": (1, 1), "UNKNOWN": (0, 0)}[state])
    elif observation.payload_kind == "QuantifierExecutionQuery":
        result["result_bits"] = list({"TRUE": (1, 0), "FALSE": (0, 1), "CONFLICT": (1, 1), "UNKNOWN": (0, 0)}[state])
    elif observation.payload_kind == "ModalExecutionQuery":
        result["result_bits"] = list(_bits((payload.get("modal_resolution_plan") or {}).get("resolution_state")))
    elif observation.payload_kind == "QuestionExecutionQuery":
        candidates = payload.get("candidate_propositions", ())
        answer_bits = {"TRUE": (1, 0), "FALSE": (0, 1)}.get(state)
        result["selected_candidate_ids"] = (
            []
            if answer_bits is None
            else [
                item.get("candidate_id")
                for item in candidates
                if isinstance(item, dict)
                and item.get("matches_request_target") == 1
                and _bits(item.get("state")) == answer_bits
            ]
        )
    elif observation.payload_kind == "GenerationAdoptionPostcheckQuery":
        candidates = payload.get("candidate_propositions", ())
        postcheck = payload.get("postcheck") or {}
        prior = postcheck.get("prior_adoption") if isinstance(postcheck, dict) else None
        prior_ids = prior.get("selected_candidate_ids") if isinstance(prior, dict) else None
        result["selected_candidate_ids"] = (
            []
            if state not in {"TRUE", "FALSE"} or "clarify" in rule.operation_key.lower()
            else list(prior_ids)
            if isinstance(prior_ids, list)
            else [item.get("candidate_id") for item in candidates if isinstance(item, dict) and _bits(item.get("state")) == (1, 0)]
        )
    elif observation.payload_kind == "FreeTextHierarchyRecallObservationV1":
        result["answer_surface"] = _free_text_answer(payload)
        result["required_stop_reason"] = "CLARIFY" if state == "UNKNOWN" and payload.get("sample_family") == "AMBIGUOUS" else "RESOLVED" if state == "TRUE" else "UNKNOWN"
    elif observation.payload_kind == "SenseBoundaryQuery":
        result["boundary"] = payload.get("candidate_sense")
    elif observation.payload_kind == "PrimitiveSurfaceQuery":
        primitive = payload.get("candidate_primitive") or {}
        result["primitive_kind"] = primitive.get("kind")
        result["primitive_registry"] = primitive.get("registry")
    elif observation.payload_kind == "RAW_SOURCE_OBSERVATION_V1":
        result["raw_observation_sha256"] = payload.get("raw_observation_sha256")
        result["definitive_truth_authoritative"] = payload.get("definitive_truth_authoritative")
        result["source_binding_required"] = int(bool(observation.source_ref_key.components))
    result["semantic_projection_sha256"] = _sha256(canonical_json_bytes({
        "payload_kind": observation.payload_kind,
        "selector_sha256": _selector(observation),
        "stable_key": list(observation.stable_key.components),
    }))
    return result


@dataclass(frozen=True)
class W09InferenceOutcome:
    """一个 private Observation 的只读实际输出和隔离计数。"""

    dimension_key: str
    observation_key: tuple[int, ...]
    actual_state: str
    actual_payload: CanonicalJsonObject
    component_state: str
    consumer_states: tuple[tuple[str, str], ...]
    shortcut_counts: tuple[tuple[str, int], ...]
    owner_counts: tuple[tuple[str, int], ...]
    state_commitment_sha256: str
    invocation_key: tuple[int, ...]

    def safe_dict(self) -> dict[str, object]:
        return {
            "actual_state": self.actual_state,
            "component_state": self.component_state,
            "consumer_states": [list(item) for item in self.consumer_states],
            "dimension_key": self.dimension_key,
            "invocation_key": list(self.invocation_key),
            "observation_key": list(self.observation_key),
            "owner_counts": [list(item) for item in self.owner_counts],
            "shortcut_counts": [list(item) for item in self.shortcut_counts],
            "state_commitment_sha256": self.state_commitment_sha256,
        }


class W09CandidateInferenceAdapter:
    """只持有 train-derived state；永不接收 evaluator label 或写入宿主。"""

    def __init__(self, state: W09InferenceState) -> None:
        if not isinstance(state, W09InferenceState):
            raise TypeError("W09 inference state 类型非法")
        self.state = state
        self.state_sha256 = state.sha256()
        self._rules = {(item.payload_kind, item.selector_sha256): item for item in state.rules}
        self._schemas = dict(state.schema_by_kind)

    def infer(self, observation: ObservationRecord, *, dimension_key: str, disabled_components: tuple[str, ...] = ()) -> W09InferenceOutcome:
        """对一个 held-out Observation 生成三向可验证输出。"""
        if not isinstance(observation, ObservationRecord) or observation.split != "held_out":
            raise W09InferenceError("W09 inference 只接受 held-out Observation")
        if dimension_key not in W09_DIMENSION_KEYS:
            raise W09InferenceError("W09 dimension 未注册")
        if tuple(sorted(set(disabled_components))) != disabled_components or any(item not in W09_DIMENSION_KEYS for item in disabled_components):
            raise W09InferenceError("W09 disabled component 非法")
        kind = observation.payload_kind
        shape = schema_sha256(observation.typed_payload.to_value())
        if kind not in self._schemas:
            raise W09InferenceError("W09 payload kind 未由 train 登记")
        schema_known = shape in self._schemas[kind]
        selector = _selector(observation)
        # held-out 必须证明对 typed 结构的推导能力；selector 只用于编译
        # train state 和审计身份，不能让部分 selector 命中覆盖新的结构语义。
        rule = None if observation.split == "held_out" else self._rules.get((kind, selector))
        state = rule.state if rule is not None else _derived_state(observation)
        operation = rule.operation_key if rule is not None else "STRUCTURED_TYPED_RESULT"
        actual = _result_payload(observation, rule or W09InferenceRule(kind, selector, state, operation, shape, (tuple(observation.stable_key.components),)), state)
        disabled = dimension_key in disabled_components
        component_state = "DISABLED" if disabled else "ACTIVE"
        consumer_state = "FAIL_CLOSED" if disabled else "RESOLVED"
        if disabled:
            state = "UNKNOWN"
            actual = {"artifact_kind": W09_INFERENCE_OUTPUT_KIND, "format_version": 1, "publication": 0, "reason_code": "COMPONENT_DISABLED", "semantic_projection_sha256": actual["semantic_projection_sha256"]}
        else:
            actual = {"artifact_kind": W09_INFERENCE_OUTPUT_KIND, "format_version": 1, **actual}
        actual_object = CanonicalJsonObject.from_value(actual)
        invocation = digest_value({
            "actual": actual,
            "component_state": component_state,
            "dimension": dimension_key,
            "disabled": list(disabled_components),
            "observation": list(observation.stable_key.components),
            "schema_known": int(schema_known),
            "state": state,
            "state_commitment": self.state_sha256,
        })
        return W09InferenceOutcome(
            dimension_key,
            tuple(observation.stable_key.components),
            state,
            actual_object,
            component_state,
            tuple((key, consumer_state) for key in W09_CONSUMER_KEYS),
            tuple((key, int(disabled)) for key in W09_INFERENCE_SHORTCUT_KEYS),
            tuple((key, 0) for key in W09_INFERENCE_OWNER_COUNT_KEYS),
            self.state_sha256,
            invocation,
        )


__all__ = [
    "W09CandidateInferenceAdapter",
    "W09InferenceError",
    "W09InferenceOutcome",
    "W09InferenceRule",
    "W09InferenceState",
    "W09_INFERENCE_INTERFACE_VERSION",
    "W09_INFERENCE_OUTPUT_KIND",
    "W09_INFERENCE_PAYLOAD_KINDS",
    "compile_w09_inference_state",
    "schema_sha256",
]
