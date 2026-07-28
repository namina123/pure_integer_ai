"""GG-03 多合法表达、十轴组合 split 与分层生成课程纯合同。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import (
    EXPECTED_STATES,
    CanonicalJsonObject,
)
from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    CHOICE_KINDS,
)
from pure_integer_ai.experiments.ph2_language_course_contract import (
    LANGUAGE_OBJECTIVE_KEYS,
)
from pure_integer_ai.experiments.ph2_language_coverage_contract import (
    SAMPLE_FAMILIES,
)


COURSE_FAMILIES = (
    "CONTEXT_ADDRESSEE_CONDITION",
    "ELLIPSIS_EXPLICIT_CONTRAST",
    "MULTI_LEGAL_SURFACE",
    "MULTI_PROPOSITION_REVISION",
    "REFERENCE_RECOVERABILITY",
    "SEMANTIC_DRIFT_NEGATIVE",
    "SINGLE_PROPOSITION_RECOMBINATION",
    "SOURCE_UNCERTAINTY_QUALIFIER",
    "STANCE_LEDGER",
    "USE_OUTCOME_REPLAY",
)
CANDIDATE_CASES = (
    "ADDRESSEE_CONTEXT_NEGATIVE",
    "ELLIPSIS_CONDITION_CONTRAST",
    "EXACT_MEMORY_CONTROL",
    "MULTI_LEGAL_SET",
    "MULTI_PROPOSITION_ORDER",
    "REFERENCE_RECOVERABILITY",
    "RELATION_ROLE_SCOPE_DRIFT",
    "RETENTION_REVERIFY",
    "REVISION_SUPERSEDE",
    "SINGLE_PROPOSITION_RECOMBINATION",
    "SOURCE_UNCERTAINTY_PRESERVATION",
    "STANCE_CONTENT_WORDING",
    "STRUCTURE_SLOT_ORDER_DRIFT",
    "USE_OUTCOME_EVIDENCE_ONLY",
)
COMBINATION_KEY_AXES = (
    "direction",
    "obligation_kind",
    "discourse_strategy",
    "proposition_logic_shape",
    "structure_family",
    "lexical_realization_family",
    "context_condition_family",
    "addressee_recoverability_family",
    "source_cluster",
    "parser_course_version",
)
COMBINATION_AXES = tuple(sorted(
    axis.upper() for axis in COMBINATION_KEY_AXES))
EVALUATOR_DIMENSIONS = tuple(sorted((
    "ADDRESSEE_RECOVERABILITY",
    "COMBINATION_HELD_OUT",
    "COMMUNICATIVE_TASK",
    "EXACT_MEMORY_BASELINE_REJECT",
    "FAILURE_LAYER_LOCALIZATION",
    "LEGAL_OBJECT_COMPOSITION",
    "MULTIPLE_LEGAL_SURFACE_SET",
    "RETENTION_REVERIFY",
    "REVISION_SUPERSEDE",
    "SEMANTIC_ROLE_SCOPE_POLARITY",
    "SOURCE_UNCERTAINTY_CITATION",
    "STANCE_CONTENT_WORDING_SEPARATION",
    "STRUCTURE_SLOT_ORDER",
    "USE_OUTCOME_TEMPLATE_PROMOTION_REJECT",
)))
INDEPENDENT_VERIFIER_REQUIREMENTS = (
    "ADDRESSEE_RECOVERABILITY",
    "COMMUNICATIVE_TASK",
    "INDEPENDENT_UNDERSTANDING_READBACK",
    "LEGAL_OBJECT_COMPOSITION",
    "SOURCE_UNCERTAINTY_CITATION",
    "STRUCTURE_SLOT_ORDER",
)
ABLATION_KEYS = tuple(sorted((
    "BROADCAST_SENTENCE_OUTCOME",
    "DISABLE_OUTCOME_ASSESSMENT_CONSUMER",
    "PROMOTE_COMPLETE_ANSWER_TEMPLATE",
    "REMOVE_DISCOURSE_REFERENCE_CHOICE",
    "REMOVE_RELATION_DIRECTION_SCOPE",
    "REMOVE_SOURCE_UNCERTAINTY",
    "REMOVE_STRUCTURE_CONCEPT",
    "SURFACE_FREQUENCY_ONLY",
)))
BASELINE_KINDS = tuple(sorted((
    "COMPLETE_ANSWER_TEMPLATE",
    "EXACT_MEMORY_REPLAY",
    "SURFACE_FREQUENCY_ONLY",
    "TYPED_LAYERED_CHOICE",
)))
RETENTION_PROTOCOLS = tuple(sorted((
    "A_B_REVERIFY_A_PRE_REGISTERED",
    "EXACT_CHOICE_USE_SCOPE_REVERIFY",
    "NO_RUNTIME_RETENTION_CLAIM",
    "REVISION_DEPENDENCY_LOCALITY",
)))
VERIFIER_NE_CONDITIONS = tuple(sorted((
    "ASSESSMENT_CONSUMER_NOT_CONNECTED",
    "FORMAL_W_RUNTIME_NOT_EXECUTED",
    "INDEPENDENT_LAYER_INPUT_MISSING",
    "NO_EVALUATOR_LABEL",
    "OUT_OF_REGISTER_OR_CONTEXT_SCOPE",
    "RETENTION_EPISODE_NOT_EXECUTED",
)))
PAYLOAD_KIND = "GenerationGeneralizationCandidateV1"
PAYLOAD_KEYS = tuple(sorted((
    "candidate_case",
    "choice_candidates",
    "combination_split",
    "context_contract",
    "course_family",
    "failure_profile",
    "objective_keys",
    "observed_surface",
    "replay_evidence",
    "resource_budget",
    "retention_anchor_id",
    "sample_family",
    "selection_state",
    "surface_candidates",
    "surface_constraints",
)))
EXPECTED_PAYLOAD_KEYS = tuple(sorted((
    "accepted_surface_candidate_ids",
    "accepted_surface_variants",
    "challenge_verdict",
    "choice_layer_states",
    "expected_failure_dimension",
    "expected_stance",
    "independent_verifier_requirements",
    "rejected_surface_candidate_ids",
    "surface_set_comparison",
    "unique_expected_string_forbidden",
)))

_CONTEXT_FIELDS = {
    "addressee_context", "communicative_goal", "content_obligations",
    "discourse_state", "expression_constraints", "goal_binding",
}
_ADDRESSEE_FIELDS = {
    "addressee_id", "recoverable_reference_ids", "shared_visible_ids",
}
_OBLIGATION_FIELDS = {
    "obligation_id", "proposition_id", "requirement", "source_ids",
    "uncertainty_id",
}
_EXPRESSION_FIELDS = {
    "ellipsis_allowed", "explicit_source_required", "max_surface_units",
    "target_language",
}
_DISCOURSE_FIELDS = {
    "open_question_ids", "prior_expression_ids", "revision_dependency_ids",
    "topic_id",
}
_CHOICE_FIELDS = {
    "candidate_id", "choice_kind", "complete_answer_template",
    "condition_family", "outcome_broadcast", "selected_object_id",
    "selection_state", "source_ids", "use_ref_id",
}
_SURFACE_CANDIDATE_FIELDS = {
    "complete_answer", "fragment", "lexical_family", "source_material_only",
    "structure_family", "surface_candidate_id", "surface_family",
}
_SURFACE_CONSTRAINT_FIELDS = {
    "candidate_ids", "challenge_candidate_id", "minimum_legal_surfaces",
    "output_surface_hidden", "target_language", "unique_string_comparison",
}
_FAILURE_FIELDS = {
    "possible_failure_dimensions", "sentence_wide_penalty_forbidden",
}
_REPLAY_FIELDS = {
    "assessment_update_present", "complete_template_promotion_forbidden",
    "exact_use_ids", "replay_kind",
}
_RESOURCE_FIELDS = {
    "max_choice_candidates", "max_context_objects", "max_surface_candidates",
    "max_surface_units", "max_verifier_dimensions",
}
_EXPECTED_LAYER_FIELDS = {"choice_kind", "state"}


class GenerationGeneralizationContractError(RuntimeError):
    """GG-03 payload 泄漏 label、组合轴不完整或分层边界失真。"""


def _exact(value: Any, keys: set[str], *, where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise GenerationGeneralizationContractError(f"{where} 字段集合非法")
    return value


def _text(value: Any, *, where: str, allow_empty: bool = False) -> str:
    if (not isinstance(value, str) or value.strip() != value
            or (not allow_empty and not value)):
        raise GenerationGeneralizationContractError(f"{where} 文本非法")
    return value


def _integer(value: Any, *, where: str, positive: bool = False) -> int:
    if (type(value) is not int or (positive and value <= 0)
            or (not positive and value < 0)):
        raise GenerationGeneralizationContractError(f"{where} 整数非法")
    return value


def _flag(value: Any, *, where: str) -> int:
    if type(value) is not int or value not in (0, 1):
        raise GenerationGeneralizationContractError(f"{where} 必须是 0/1")
    return value


def _texts(
        value: Any, *, where: str, allow_empty: bool = False,
        sorted_unique: bool = False,
        ) -> tuple[str, ...]:
    if (not isinstance(value, list) or (not allow_empty and not value)
            or any(not isinstance(item, str) or not item for item in value)
            or len(set(value)) != len(value)):
        raise GenerationGeneralizationContractError(f"{where} 列表非法")
    result = tuple(value)
    if sorted_unique and result != tuple(sorted(result)):
        raise GenerationGeneralizationContractError(f"{where} 必须排序")
    return result


def _integers(
        value: Any, *, where: str, allow_empty: bool = False,
        ) -> tuple[int, ...]:
    if (not isinstance(value, list) or (not allow_empty and not value)
            or any(type(item) is not int or item <= 0 for item in value)
            or len(set(value)) != len(value)):
        raise GenerationGeneralizationContractError(f"{where} 整数列表非法")
    return tuple(value)


def surface_sha256(text: str) -> str:
    """计算 append-only observed surface 的规范 UTF-8 hash。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def combination_key(value: dict[str, Any]) -> str:
    """按文档顺序保留十轴明文组合，不以 opaque hash 代替。"""
    return "::".join(_text(value[axis], where=f"combination.{axis}")
                     for axis in COMBINATION_KEY_AXES)


@dataclass(frozen=True)
class GenerationGeneralizationPayloadAudit:
    """返回 split、候选和分层覆盖的可审计事实。"""

    combination_key: str
    combination_values: tuple[str, ...]
    surface_candidate_ids: tuple[str, ...]
    choice_kinds: tuple[str, ...]
    exact_memory_control: int
    replay_kind: str


def validate_generation_generalization_payload(
        payload: CanonicalJsonObject,
        ) -> GenerationGeneralizationPayloadAudit:
    """核学生可见 Observation，不读取 evaluator 合法输出集合。"""
    if not isinstance(payload, CanonicalJsonObject):
        raise GenerationGeneralizationContractError("GG-03 payload 类型非法")
    value = _exact(payload.to_value(), set(PAYLOAD_KEYS), where="GG-03 payload")
    if value["sample_family"] not in SAMPLE_FAMILIES:
        raise GenerationGeneralizationContractError("sample_family 未登记")
    if value["course_family"] not in COURSE_FAMILIES:
        raise GenerationGeneralizationContractError("course_family 未登记")
    if value["candidate_case"] not in CANDIDATE_CASES:
        raise GenerationGeneralizationContractError("candidate_case 未登记")
    if value["selection_state"] != "UNSELECTED":
        raise GenerationGeneralizationContractError("Observation 不得预选生成结果")
    objectives = _texts(
        value["objective_keys"], where="objective_keys", sorted_unique=True)
    if any(item not in LANGUAGE_OBJECTIVE_KEYS for item in objectives):
        raise GenerationGeneralizationContractError("objective_keys 未登记")

    observed = _exact(value["observed_surface"], {
        "append_only", "output_surface_hidden", "sha256", "text",
    }, where="observed_surface")
    text = _text(observed["text"], where="observed text")
    if (_flag(observed["append_only"], where="append_only") != 1
            or _flag(observed["output_surface_hidden"],
                     where="output_surface_hidden") != 1
            or observed["sha256"] != surface_sha256(text)):
        raise GenerationGeneralizationContractError(
            "observed surface 未保持 append-only/hash/target hidden")

    context = _exact(value["context_contract"], _CONTEXT_FIELDS,
                     where="context_contract")
    _integer(context["goal_binding"], where="goal_binding", positive=True)
    _text(context["communicative_goal"], where="communicative_goal")
    addressee = _exact(context["addressee_context"], _ADDRESSEE_FIELDS,
                       where="addressee_context")
    _integer(addressee["addressee_id"], where="addressee_id", positive=True)
    visible = _integers(addressee["shared_visible_ids"], where="visible ids",
                        allow_empty=True)
    recoverable = _integers(
        addressee["recoverable_reference_ids"], where="recoverable ids",
        allow_empty=True)
    if not set(recoverable).issubset(set(visible)):
        raise GenerationGeneralizationContractError(
            "recoverable reference 不在 shared visible 边界")
    obligations_raw = context["content_obligations"]
    if not isinstance(obligations_raw, list) or not obligations_raw:
        raise GenerationGeneralizationContractError("content obligations 为空")
    requirements = []
    for item in obligations_raw:
        obligation = _exact(item, _OBLIGATION_FIELDS,
                            where="content obligation")
        _integer(obligation["obligation_id"], where="obligation id",
                 positive=True)
        _integer(obligation["proposition_id"], where="proposition id",
                 positive=True)
        if obligation["requirement"] not in ("FORBIDDEN", "OPTIONAL", "REQUIRED"):
            raise GenerationGeneralizationContractError(
                "content obligation requirement 非法")
        requirements.append(obligation["requirement"])
        _integers(obligation["source_ids"], where="obligation source ids")
        _integer(obligation["uncertainty_id"], where="uncertainty id")
    if "REQUIRED" not in requirements:
        raise GenerationGeneralizationContractError("缺少 REQUIRED obligation")
    expression = _exact(context["expression_constraints"], _EXPRESSION_FIELDS,
                        where="expression_constraints")
    if expression["target_language"] != "zh":
        raise GenerationGeneralizationContractError("首阶段目标语言必须为 zh")
    _flag(expression["ellipsis_allowed"], where="ellipsis_allowed")
    _flag(expression["explicit_source_required"],
          where="explicit_source_required")
    _integer(expression["max_surface_units"], where="max_surface_units",
             positive=True)
    discourse = _exact(context["discourse_state"], _DISCOURSE_FIELDS,
                       where="discourse_state")
    _integer(discourse["topic_id"], where="topic_id", positive=True)
    for name in (
            "open_question_ids", "prior_expression_ids",
            "revision_dependency_ids"):
        _integers(discourse[name], where=name, allow_empty=True)

    split = value["combination_split"]
    split_fields = set(COMBINATION_KEY_AXES) | {
        "combination_key", "exact_memory_control"}
    split = _exact(split, split_fields, where="combination_split")
    values = tuple(_text(split[axis], where=f"combination.{axis}")
                   for axis in COMBINATION_KEY_AXES)
    if split["combination_key"] != combination_key(split):
        raise GenerationGeneralizationContractError("combination_key 漂移")
    exact_memory = _flag(
        split["exact_memory_control"], where="exact_memory_control")

    surface_raw = value["surface_candidates"]
    if not isinstance(surface_raw, list) or len(surface_raw) < 3:
        raise GenerationGeneralizationContractError("surface candidates 不足")
    surface_ids = []
    for item in surface_raw:
        candidate = _exact(item, _SURFACE_CANDIDATE_FIELDS,
                           where="surface candidate")
        surface_ids.append(_text(
            candidate["surface_candidate_id"], where="surface candidate id"))
        for name in (
                "fragment", "lexical_family", "structure_family",
                "surface_family"):
            _text(candidate[name], where=f"surface candidate {name}")
        if (_flag(candidate["complete_answer"], where="complete_answer") != 0
                or _flag(candidate["source_material_only"],
                         where="source_material_only") != 1):
            raise GenerationGeneralizationContractError(
                "Observation 不得携完整回答模板")
    if len(set(surface_ids)) != len(surface_ids):
        raise GenerationGeneralizationContractError("surface candidate id 重复")

    constraints = _exact(
        value["surface_constraints"], _SURFACE_CONSTRAINT_FIELDS,
        where="surface_constraints")
    constraint_ids = _texts(
        constraints["candidate_ids"], where="surface constraint candidates",
        sorted_unique=True)
    if set(constraint_ids) != set(surface_ids):
        raise GenerationGeneralizationContractError(
            "surface constraints 未精确引用候选")
    if constraints["challenge_candidate_id"] not in constraint_ids:
        raise GenerationGeneralizationContractError("challenge candidate 缺失")
    if (_integer(constraints["minimum_legal_surfaces"],
                 where="minimum_legal_surfaces", positive=True) < 2
            or _flag(constraints["output_surface_hidden"],
                     where="constraint output hidden") != 1
            or _flag(constraints["unique_string_comparison"],
                     where="unique string comparison") != 0
            or constraints["target_language"] != "zh"):
        raise GenerationGeneralizationContractError(
            "多合法 surface/hidden/set evaluator 约束非法")

    choices_raw = value["choice_candidates"]
    if (not isinstance(choices_raw, list)
            or len(choices_raw) != len(CHOICE_KINDS)):
        raise GenerationGeneralizationContractError("五层 choice 未列全")
    choice_kinds = []
    choice_ids = []
    for item in choices_raw:
        choice = _exact(item, _CHOICE_FIELDS, where="choice candidate")
        choice_kinds.append(_text(choice["choice_kind"], where="choice kind"))
        choice_ids.append(_text(choice["candidate_id"], where="choice id"))
        _text(choice["condition_family"], where="choice condition")
        _integer(choice["selected_object_id"], where="selected object",
                 positive=True)
        _integers(choice["source_ids"], where="choice source ids")
        _text(choice["use_ref_id"], where="use ref", allow_empty=True)
        if (choice["selection_state"] != "UNSELECTED"
                or _flag(choice["complete_answer_template"],
                         where="answer template") != 0
                or _flag(choice["outcome_broadcast"],
                         where="outcome broadcast") != 0):
            raise GenerationGeneralizationContractError(
                "choice 不得预选、模板化或整句广播")
    if tuple(choice_kinds) != CHOICE_KINDS or len(set(choice_ids)) != len(choice_ids):
        raise GenerationGeneralizationContractError("五层 choice 顺序或身份非法")

    failure = _exact(value["failure_profile"], _FAILURE_FIELDS,
                     where="failure_profile")
    failures = _texts(
        failure["possible_failure_dimensions"], where="failure dimensions",
        sorted_unique=True)
    if any(item not in EVALUATOR_DIMENSIONS for item in failures):
        raise GenerationGeneralizationContractError("failure dimension 未登记")
    if _flag(failure["sentence_wide_penalty_forbidden"],
             where="sentence-wide penalty") != 1:
        raise GenerationGeneralizationContractError("整句统一惩罚必须禁止")

    replay = _exact(value["replay_evidence"], _REPLAY_FIELDS,
                    where="replay_evidence")
    replay_kind = _text(replay["replay_kind"], where="replay kind")
    if replay_kind not in ("EVIDENCE_ONLY", "NONE"):
        raise GenerationGeneralizationContractError("replay kind 非法")
    use_ids = _texts(replay["exact_use_ids"], where="exact use ids",
                     allow_empty=True, sorted_unique=True)
    if bool(use_ids) != (replay_kind == "EVIDENCE_ONLY"):
        raise GenerationGeneralizationContractError("replay 与 exact Use 不一致")
    if (_flag(replay["complete_template_promotion_forbidden"],
              where="template promotion") != 1
            or _flag(replay["assessment_update_present"],
                     where="assessment update") != 0):
        raise GenerationGeneralizationContractError(
            "replay 只能作 Evidence 且不得冒充 assessment 更新")

    budget = _exact(value["resource_budget"], _RESOURCE_FIELDS,
                    where="resource_budget")
    for name, item in budget.items():
        _integer(item, where=f"resource {name}", positive=True)
    if len(surface_ids) > budget["max_surface_candidates"]:
        raise GenerationGeneralizationContractError("surface candidates 超预算")
    if len(choice_ids) > budget["max_choice_candidates"]:
        raise GenerationGeneralizationContractError("choice candidates 超预算")
    if len(failures) > budget["max_verifier_dimensions"]:
        raise GenerationGeneralizationContractError("verifier dimensions 超预算")
    _text(value["retention_anchor_id"], where="retention anchor",
          allow_empty=True)
    return GenerationGeneralizationPayloadAudit(
        split["combination_key"], values, tuple(surface_ids),
        tuple(choice_kinds), exact_memory, replay_kind)


def validate_generation_generalization_expected(
        expected: CanonicalJsonObject,
        *,
        expected_state: str,
        evaluation_dimension: str,
        observation_payload: CanonicalJsonObject,
        ) -> None:
    """核私有 label 使用集合/约束，不把唯一 expected answer 放入 Observation。"""
    if expected_state not in EXPECTED_STATES:
        raise GenerationGeneralizationContractError("expected_state 非四态")
    if evaluation_dimension not in EVALUATOR_DIMENSIONS:
        raise GenerationGeneralizationContractError("evaluation dimension 未登记")
    audit = validate_generation_generalization_payload(observation_payload)
    value = _exact(expected.to_value(), set(EXPECTED_PAYLOAD_KEYS),
                   where="GG-03 expected payload")
    accepted = _texts(
        value["accepted_surface_candidate_ids"], where="accepted candidates",
        sorted_unique=True)
    rejected = _texts(
        value["rejected_surface_candidate_ids"], where="rejected candidates",
        sorted_unique=True)
    if (len(accepted) < 2 or set(accepted) & set(rejected)
            or not set((*accepted, *rejected)).issubset(
                set(audit.surface_candidate_ids))):
        raise GenerationGeneralizationContractError(
            "多合法 accepted/rejected candidate 集非法")
    variants = _texts(
        value["accepted_surface_variants"], where="accepted variants",
        sorted_unique=True)
    if len(variants) < 2:
        raise GenerationGeneralizationContractError("合法 surface 变体不足")
    observation_text = observation_payload.payload.decode("utf-8")
    if any(item in observation_text for item in variants):
        raise GenerationGeneralizationContractError(
            "私有合法输出泄漏到 Observation")
    if value["challenge_verdict"] != expected_state:
        raise GenerationGeneralizationContractError("challenge verdict 漂移")
    if value["expected_failure_dimension"] != evaluation_dimension:
        raise GenerationGeneralizationContractError("failure dimension 漂移")
    _text(value["expected_stance"], where="expected stance")
    layers = value["choice_layer_states"]
    if not isinstance(layers, list) or len(layers) != len(CHOICE_KINDS):
        raise GenerationGeneralizationContractError("expected choice layers 未列全")
    kinds = []
    for item in layers:
        layer = _exact(item, _EXPECTED_LAYER_FIELDS,
                       where="expected choice layer")
        kinds.append(layer["choice_kind"])
        if layer["state"] not in (*EXPECTED_STATES, "NE"):
            raise GenerationGeneralizationContractError("choice layer state 非法")
    if tuple(kinds) != CHOICE_KINDS:
        raise GenerationGeneralizationContractError("expected choice layer 顺序漂移")
    requirements = value["independent_verifier_requirements"]
    if (not isinstance(requirements, dict)
            or tuple(sorted(requirements)) != INDEPENDENT_VERIFIER_REQUIREMENTS
            or any(_flag(item, where="verifier requirement") != 1
                   for item in requirements.values())):
        raise GenerationGeneralizationContractError("独立 verifier 合取未列全")
    if (value["surface_set_comparison"] != "SET_OR_CONSTRAINT"
            or _flag(value["unique_expected_string_forbidden"],
                     where="unique expected string") != 1):
        raise GenerationGeneralizationContractError("evaluator 仍依赖唯一字符串")


__all__ = [
    "ABLATION_KEYS",
    "BASELINE_KINDS",
    "CANDIDATE_CASES",
    "COMBINATION_AXES",
    "COMBINATION_KEY_AXES",
    "COURSE_FAMILIES",
    "EVALUATOR_DIMENSIONS",
    "EXPECTED_PAYLOAD_KEYS",
    "GenerationGeneralizationContractError",
    "GenerationGeneralizationPayloadAudit",
    "INDEPENDENT_VERIFIER_REQUIREMENTS",
    "PAYLOAD_KEYS",
    "PAYLOAD_KIND",
    "RETENTION_PROTOCOLS",
    "VERIFIER_NE_CONDITIONS",
    "combination_key",
    "surface_sha256",
    "validate_generation_generalization_expected",
    "validate_generation_generalization_payload",
]
