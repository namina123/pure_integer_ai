"""LC-06 比较、程度、数量、范围与度量的原创课程。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pure_integer_ai.experiments.ph2_authored_course_common import (
    AuthoredCompiledSeed,
    AuthoredCourseBuild,
    AuthoredCourseSpec,
    publish_authored_course,
)
from pure_integer_ai.experiments.ph2_capability_course_contract import (
    COURSE_EXECUTION_STATE,
    COURSE_INVARIANTS,
    COURSE_SPLIT_AXES,
    CapabilityCourseManifest,
)
from pure_integer_ai.experiments.ph2_dataset_contract import (
    EXPECTED_STATES,
    SAMPLE_ROLES,
    CanonicalJsonObject,
    canonical_json_line,
    parse_canonical_json_bytes,
)
from pure_integer_ai.experiments.ph2_language_course_contract import (
    LANGUAGE_OBJECTIVE_KEYS,
)
from pure_integer_ai.experiments.ph2_language_coverage_contract import (
    SAMPLE_FAMILIES,
)


SOURCE_KEY = "AUTHORED_CC0_V1"
LICENSE_ID = "CC0-1.0"
COURSE_VERSION = 1
ARTIFACT_VERSION = 1
ADAPTER_VERSION = 1
GENERATOR_VERSION = 1
PARSER_VERSION = 1
PACK_NAME = "AUTHORED_CC0_V1--CC0-1.0--lc06-comparison-quantity-v1"
STAGE = "W-05"
SUBSTAGE = "LC06_COMPARISON_QUANTITY_MEASURE"
PAYLOAD_KIND = "ComparisonQuantityCandidateV1"
COURSE_MANIFEST_ARTIFACT_VERSION = "LC-06-comparison-quantity-course-v1"
COURSE_MANIFEST_PATH = Path(
    "data/ph2/manifests/lc06_comparison_quantity_course_v1.json")
FORMAL_ARTIFACT_RELATIVE_ROOT = (
    "ph2_dataset_artifacts/d02_language_courses_v1")

CANDIDATE_KINDS = (
    "AMBIGUOUS_STANDARD",
    "APPROXIMATE_EXACT",
    "BARE_PROPERTY_BASELINE",
    "COMPARISON",
    "DEGREE",
    "GENERATION",
    "MEASURE",
    "QUANTITY_COUNT",
    "QUANTIFIER_SCOPE",
    "RANGE",
    "RETENTION",
    "REVISION",
    "UNIT_ERASURE_BASELINE",
    "UNKNOWN",
)
OBJECT_KINDS = ("ENTITY", "EVENT", "SET_EXPR")
SCALE_DIRECTIONS = ("DECREASING", "INCREASING", "UNKNOWN")
STANDARD_KINDS = ("CONTEXT", "EXPLICIT", "UNKNOWN")
EXACTNESS_KINDS = ("APPROXIMATE", "COUNT", "EXACT", "RANGE", "UNKNOWN")
COMPARISON_OPERATORS = ("EQ", "GE", "GT", "LE", "LT", "UNKNOWN")
QUANTIFIER_KINDS = ("COUNT_AT_LEAST", "COUNT_EXACT", "EXISTS", "FORALL")
BASELINE_KINDS = (
    "BARE_PROPERTY_ONLY",
    "TYPED_COMPARISON_OBJECTS_PRESENT",
    "UNIT_ERASURE",
)
EVALUATOR_DIMENSIONS = (
    "AMBIGUOUS_STANDARD_COMPETITION",
    "APPROXIMATE_EXACT_DISTINCTION",
    "BARE_PROPERTY_REJECT",
    "COMPARISON_DIRECTION_STANDARD",
    "DEGREE_SCALE_THRESHOLD",
    "LOCAL_QUANTITY_REVISION",
    "MEASURE_UNIT_DIMENSION",
    "QUANTIFIER_SCOPE",
    "QUANTITY_COUNT",
    "QUANTITY_UNKNOWN",
    "RANGE_BOUNDARY",
    "RETENTION_REVERIFY",
    "REVERSE_GENERATION_COMPARISON_MEASURE",
    "UNIT_ERASURE_REJECT",
)
VERIFIER_NE_CONDITIONS = (
    "CAPABILITY_LEARNED_REQUESTED",
    "GENERATION_RESULT_NOT_EXECUTED",
    "NO_EVALUATOR_LABEL",
    "OUT_OF_CONTEXT_SCOPE",
    "SCALE_STANDARD_UNIT_OR_QUANTITY_MISSING",
    "SEMANTIC_TRUTH_REQUESTED",
)
RETENTION_PROTOCOLS = (
    "A_TO_LC06_REVERIFY_A",
    "DUMP_RESUME_REQUIRED_AT_RUNTIME",
    "LOCAL_QUANTITY_SUPERSEDE_ONLY",
)
COMBINATION_AXES = (
    "OBJECT_FAMILY",
    "OBJECT_X_SCALE_X_UNIT",
    "SCALE_FAMILY",
    "UNIT_FAMILY",
)
ABLATION_KEYS = (
    "BARE_PROPERTY_ONLY",
    "DROP_QUANTITY_CANDIDATE",
    "DROP_SCALE_DEFINITION",
    "DROP_STANDARD_CANDIDATE",
    "DROP_UNIT_DEFINITION",
    "UNIT_ERASURE",
)
_OBJECT_KIND_BY_FAMILY = {
    "ENTITY": "ENTITY",
    "EVENT": "EVENT",
    "SET": "SET_EXPR",
}
_DIMENSION_BY_SCALE = {
    "COUNT": "COUNT",
    "DURATION": "TIME",
    "LENGTH": "LENGTH",
    "MASS": "MASS",
    "TEMPERATURE": "TEMPERATURE",
    "UNITLESS": "UNITLESS",
    "UNKNOWN": "UNKNOWN",
}
_UNIT_BY_SCALE = {
    "COUNT": "COUNT_UNIT",
    "DURATION": "HOUR",
    "LENGTH": "METER",
    "MASS": "KILOGRAM",
    "TEMPERATURE": "CELSIUS",
    "UNITLESS": "UNITLESS",
    "UNKNOWN": "UNKNOWN_UNIT",
}
PAYLOAD_KEYS = (
    "baseline_kind",
    "candidate_kind",
    "comparison_candidates",
    "generation_constraint",
    "object_candidates",
    "objective_keys",
    "observed_surface",
    "quantifier_scopes",
    "quantity_candidates",
    "resource_budget",
    "retention_anchor_id",
    "revision_receipt",
    "sample_family",
    "scale_definitions",
    "selection_state",
    "split_identity",
    "standard_candidates",
    "surface_scope",
    "unit_definitions",
)

_SEED_FIELDS = {
    "baseline_kind", "candidate_kind", "comparison_candidates",
    "evaluation_dimension", "expected_payload", "expected_state", "family",
    "generation_constraint", "label_owner", "license_id", "logical_order",
    "object_candidates", "object_family", "objective_keys", "observed_text",
    "perturbation_kind", "quantifier_scopes", "quantity_candidates",
    "resource_budget", "retention_anchor_id", "revision_receipt",
    "sample_family", "sample_role", "scale_definitions", "scale_family",
    "seed_id", "split", "standard_candidates", "supersedes_seed_id",
    "surface_scope", "template_family", "unit_definitions", "unit_family",
}
_SCOPE_FIELDS = {
    "context_scope_key", "genre", "language", "register", "script"}
_OBJECT_FIELDS = {
    "context_scope_key", "object_id", "object_kind", "proposition_key"}
_SCALE_FIELDS = {
    "context_scope_key", "dimension_key", "direction", "scale_id",
    "threshold_den", "threshold_num"}
_STANDARD_FIELDS = {
    "context_scope_key", "reference_object_id", "standard_id", "standard_kind"}
_UNIT_FIELDS = {
    "canonical_unit_id", "context_scope_key", "conversion_den",
    "conversion_num", "dimension_key", "unit_id"}
_QUANTITY_FIELDS = {
    "amount_den", "amount_num", "candidate_version", "exactness",
    "lower_den", "lower_inclusive", "lower_num", "object_id", "quantity_id",
    "scale_id", "unit_id", "upper_den", "upper_inclusive", "upper_num"}
_COMPARISON_FIELDS = {
    "candidate_version", "comparison_id", "context_scope_key", "left_object_id",
    "operator", "right_object_id", "scale_id", "standard_id"}
_QUANTIFIER_FIELDS = {
    "binder_id", "candidate_version", "domain_object_id", "proposition_key",
    "provisional", "quantifier_kind", "scope_order"}
_GENERATION_FIELDS = {
    "comparison_preservation_required", "direction", "input_candidate_ids",
    "output_surface_hidden", "quantity_preservation_required",
    "quantifier_scope_preservation_required", "scale_preservation_required",
    "standard_preservation_required", "unit_preservation_required"}
_REVISION_FIELDS = {
    "candidate_version", "dependency_keys", "raw_observation_preserved",
    "revision_scope", "supersedes_seed_id"}
_BUDGET_FIELDS = {
    "max_comparisons", "max_objects", "max_output_units", "max_quantifiers",
    "max_quantities", "max_scales", "max_standards", "max_units"}
_EXPECTED_FIELDS = {
    "accepted", "accepted_surfaces", "analysis_key", "reason_code"}
_SPLIT_FIELDS = {
    "combination_key", "isolation_axis", "object_family", "scale_family",
    "unit_family"}
_ROLE_BY_FAMILY = {
    "AMBIGUOUS": "conflict",
    "NEGATIVE": "refute",
    "POSITIVE": "support",
    "RETENTION": "read_only_probe",
    "REVISION": "supersede",
    "UNKNOWN": "anomaly",
}
_STATE_BY_FAMILY = {
    "AMBIGUOUS": "CONFLICT",
    "NEGATIVE": "FALSE",
    "POSITIVE": "TRUE",
    "RETENTION": "TRUE",
    "REVISION": "TRUE",
    "UNKNOWN": "UNKNOWN",
}


class AuthoredComparisonQuantityCourseError(RuntimeError):
    """LC-06 seed、typed payload 或组合 split 不满足冻结合同。"""


def _exact_keys(value: Any, keys: set[str], *, where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise AuthoredComparisonQuantityCourseError(f"{where} 字段集合非法")
    return value


def _text(value: Any, *, where: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or value.strip() != value:
        raise AuthoredComparisonQuantityCourseError(f"{where} 必须是规范文本")
    if not allow_empty and not value:
        raise AuthoredComparisonQuantityCourseError(f"{where} 不能为空")
    return value


def _integer(value: Any, *, where: str, positive: bool = False) -> int:
    if type(value) is not int or value < int(positive):
        qualifier = "正" if positive else "非负"
        raise AuthoredComparisonQuantityCourseError(
            f"{where} 必须是{qualifier}严格整数")
    return value


def _flag(value: Any, *, where: str) -> int:
    if type(value) is not int or value not in {0, 1}:
        raise AuthoredComparisonQuantityCourseError(f"{where} 必须是 0/1")
    return value


def _string_tuple(
        value: Any,
        *,
        where: str,
        sorted_unique: bool = False,
        allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise AuthoredComparisonQuantityCourseError(f"{where} 必须是文本数组")
    result = tuple(_text(item, where=where) for item in value)
    if len(result) != len(set(result)):
        raise AuthoredComparisonQuantityCourseError(f"{where} 不得重复")
    if sorted_unique and tuple(sorted(result)) != result:
        raise AuthoredComparisonQuantityCourseError(f"{where} 必须排序")
    return result


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _compare(
        left_num: int, left_den: int, right_num: int, right_den: int,
        ) -> int:
    """以整数交叉积比较两个非负 Rational。"""
    left = left_num * right_den
    right = right_num * left_den
    return (left > right) - (left < right)


def _validate_expected(
        payload: CanonicalJsonObject,
        *,
        expected_state: str) -> None:
    raw = _exact_keys(payload.to_value(), _EXPECTED_FIELDS, where="expected")
    accepted = _flag(raw["accepted"], where="expected accepted")
    surfaces = _string_tuple(
        raw["accepted_surfaces"], where="accepted surfaces", allow_empty=True)
    _text(raw["analysis_key"], where="analysis_key")
    _text(raw["reason_code"], where="reason_code")
    if expected_state == "TRUE" and (accepted != 1 or not surfaces):
        raise AuthoredComparisonQuantityCourseError("TRUE label 必须给出私有合法表层")
    if expected_state != "TRUE" and (accepted != 0 or surfaces):
        raise AuthoredComparisonQuantityCourseError("非 TRUE label 不得给出采用表层")


@dataclass(frozen=True)
class ComparisonQuantityPayloadAudit:
    """返回比较对象计数、组合键和两类负基线状态。"""

    object_count: int
    scale_count: int
    standard_count: int
    unit_count: int
    quantity_count: int
    comparison_count: int
    quantifier_count: int
    combination_key: str
    bare_property_baseline: int
    unit_erasure_baseline: int


def validate_comparison_quantity_payload(
        payload: CanonicalJsonObject | dict[str, Any],
        ) -> ComparisonQuantityPayloadAudit:
    """校验比较标准、尺度、Rational 数量、单位、范围和量词 scope。"""
    value = payload.to_value() if isinstance(payload, CanonicalJsonObject) else payload
    raw = _exact_keys(value, set(PAYLOAD_KEYS), where="comparison payload")
    candidate_kind = _text(raw["candidate_kind"], where="candidate_kind")
    if candidate_kind not in CANDIDATE_KINDS:
        raise AuthoredComparisonQuantityCourseError("candidate_kind 未登记")
    baseline = _text(raw["baseline_kind"], where="baseline_kind")
    if baseline not in BASELINE_KINDS:
        raise AuthoredComparisonQuantityCourseError("baseline_kind 未登记")
    if raw["sample_family"] not in SAMPLE_FAMILIES:
        raise AuthoredComparisonQuantityCourseError("sample_family 未登记")
    if raw["selection_state"] != "UNSELECTED":
        raise AuthoredComparisonQuantityCourseError("Observation 不得预选比较候选")
    objective_keys = _string_tuple(
        raw["objective_keys"], where="objective_keys", sorted_unique=True)
    if any(key not in LANGUAGE_OBJECTIVE_KEYS for key in objective_keys):
        raise AuthoredComparisonQuantityCourseError("objective_keys 未登记")

    scope = _exact_keys(raw["surface_scope"], _SCOPE_FIELDS, where="scope")
    if scope["language"] != "zh" or scope["script"] != "HAN":
        raise AuthoredComparisonQuantityCourseError("LC-06 首阶段只冻结 zh/HAN")
    context_scope = _text(scope["context_scope_key"], where="context scope")
    _text(scope["genre"], where="genre")
    if scope["register"] not in {"COLLOQUIAL", "FORMAL", "NEUTRAL"}:
        raise AuthoredComparisonQuantityCourseError("register 未登记")

    observed = _exact_keys(raw["observed_surface"], {
        "append_only", "sha256", "target_hidden", "text",
    }, where="observed surface")
    observed_text = _text(observed["text"], where="observed text")
    if _flag(observed["append_only"], where="append_only") != 1:
        raise AuthoredComparisonQuantityCourseError("raw Observation 必须 append-only")
    if observed["sha256"] != _sha256_text(observed_text):
        raise AuthoredComparisonQuantityCourseError("observed surface SHA-256 漂移")
    target_hidden = _flag(observed["target_hidden"], where="target_hidden")

    budget = _exact_keys(raw["resource_budget"], _BUDGET_FIELDS, where="budget")
    ceilings = {
        key: _integer(budget[key], where=key, positive=True)
        for key in sorted(_BUDGET_FIELDS)
    }
    if len(observed_text) > ceilings["max_output_units"]:
        raise AuthoredComparisonQuantityCourseError("surface 超整数输出预算")

    objects_value = raw["object_candidates"]
    if not isinstance(objects_value, list) or not objects_value:
        raise AuthoredComparisonQuantityCourseError("object_candidates 不能为空")
    if len(objects_value) > ceilings["max_objects"]:
        raise AuthoredComparisonQuantityCourseError("比较对象超预算")
    objects: dict[str, dict[str, Any]] = {}
    for item in objects_value:
        obj = _exact_keys(item, _OBJECT_FIELDS, where="comparison object")
        object_id = _text(obj["object_id"], where="object_id")
        if object_id in objects:
            raise AuthoredComparisonQuantityCourseError("object_id 重复")
        if obj["object_kind"] not in OBJECT_KINDS:
            raise AuthoredComparisonQuantityCourseError("object_kind 未登记")
        if obj["context_scope_key"] != context_scope:
            raise AuthoredComparisonQuantityCourseError("比较对象跨 ContextScope")
        _text(obj["proposition_key"], where="proposition_key")
        objects[object_id] = obj

    scales_value = raw["scale_definitions"]
    if not isinstance(scales_value, list):
        raise AuthoredComparisonQuantityCourseError("scale_definitions 必须是数组")
    if len(scales_value) > ceilings["max_scales"]:
        raise AuthoredComparisonQuantityCourseError("尺度定义超预算")
    scales: dict[str, dict[str, Any]] = {}
    for item in scales_value:
        scale = _exact_keys(item, _SCALE_FIELDS, where="scale definition")
        scale_id = _text(scale["scale_id"], where="scale_id")
        if scale_id in scales:
            raise AuthoredComparisonQuantityCourseError("scale_id 重复")
        _text(scale["dimension_key"], where="scale dimension")
        if scale["direction"] not in SCALE_DIRECTIONS:
            raise AuthoredComparisonQuantityCourseError("scale direction 未登记")
        if scale["context_scope_key"] != context_scope:
            raise AuthoredComparisonQuantityCourseError("尺度跨 ContextScope")
        _integer(scale["threshold_num"], where="threshold numerator")
        _integer(scale["threshold_den"], where="threshold denominator", positive=True)
        scales[scale_id] = scale

    standards_value = raw["standard_candidates"]
    if not isinstance(standards_value, list):
        raise AuthoredComparisonQuantityCourseError("standard_candidates 必须是数组")
    if len(standards_value) > ceilings["max_standards"]:
        raise AuthoredComparisonQuantityCourseError("比较标准超预算")
    standards: dict[str, dict[str, Any]] = {}
    for item in standards_value:
        standard = _exact_keys(item, _STANDARD_FIELDS, where="standard candidate")
        standard_id = _text(standard["standard_id"], where="standard_id")
        if standard_id in standards:
            raise AuthoredComparisonQuantityCourseError("standard_id 重复")
        if standard["standard_kind"] not in STANDARD_KINDS:
            raise AuthoredComparisonQuantityCourseError("standard_kind 未登记")
        reference = _text(
            standard["reference_object_id"], where="standard reference",
            allow_empty=True)
        if reference and reference not in objects:
            raise AuthoredComparisonQuantityCourseError("标准引用未知对象")
        if standard["standard_kind"] == "EXPLICIT" and not reference:
            raise AuthoredComparisonQuantityCourseError("显式标准必须引用对象")
        if standard["context_scope_key"] != context_scope:
            raise AuthoredComparisonQuantityCourseError("比较标准跨 ContextScope")
        standards[standard_id] = standard

    units_value = raw["unit_definitions"]
    if not isinstance(units_value, list):
        raise AuthoredComparisonQuantityCourseError("unit_definitions 必须是数组")
    if len(units_value) > ceilings["max_units"]:
        raise AuthoredComparisonQuantityCourseError("单位定义超预算")
    units: dict[str, dict[str, Any]] = {}
    for item in units_value:
        unit = _exact_keys(item, _UNIT_FIELDS, where="unit definition")
        unit_id = _text(unit["unit_id"], where="unit_id")
        if unit_id in units:
            raise AuthoredComparisonQuantityCourseError("unit_id 重复")
        _text(unit["canonical_unit_id"], where="canonical_unit_id")
        _text(unit["dimension_key"], where="unit dimension")
        numerator = _integer(
            unit["conversion_num"], where="unit conversion numerator", positive=True)
        denominator = _integer(
            unit["conversion_den"], where="unit conversion denominator", positive=True)
        if numerator <= 0 or denominator <= 0:
            raise AuthoredComparisonQuantityCourseError("单位换算必须为正 Rational")
        if unit["context_scope_key"] != context_scope:
            raise AuthoredComparisonQuantityCourseError("单位跨 ContextScope")
        units[unit_id] = unit

    quantities_value = raw["quantity_candidates"]
    if not isinstance(quantities_value, list) or not quantities_value:
        raise AuthoredComparisonQuantityCourseError("quantity_candidates 不能为空")
    if len(quantities_value) > ceilings["max_quantities"]:
        raise AuthoredComparisonQuantityCourseError("数量候选超预算")
    quantities: dict[str, dict[str, Any]] = {}
    exactnesses: set[str] = set()
    for item in quantities_value:
        quantity = _exact_keys(item, _QUANTITY_FIELDS, where="quantity candidate")
        quantity_id = _text(quantity["quantity_id"], where="quantity_id")
        if quantity_id in quantities:
            raise AuthoredComparisonQuantityCourseError("quantity_id 重复")
        object_id = _text(quantity["object_id"], where="quantity object_id")
        if object_id not in objects:
            raise AuthoredComparisonQuantityCourseError("数量引用未知对象")
        scale_id = _text(quantity["scale_id"], where="quantity scale", allow_empty=True)
        unit_id = _text(quantity["unit_id"], where="quantity unit", allow_empty=True)
        if scale_id and scale_id not in scales:
            raise AuthoredComparisonQuantityCourseError("数量引用未知尺度")
        if unit_id and unit_id not in units:
            raise AuthoredComparisonQuantityCourseError("数量引用未知单位")
        if scale_id and unit_id and (
                scales[scale_id]["dimension_key"] != units[unit_id]["dimension_key"]):
            raise AuthoredComparisonQuantityCourseError("尺度与单位维度不一致")
        exactness = _text(quantity["exactness"], where="exactness")
        if exactness not in EXACTNESS_KINDS:
            raise AuthoredComparisonQuantityCourseError("exactness 未登记")
        amount_num = _integer(quantity["amount_num"], where="amount numerator")
        amount_den = _integer(
            quantity["amount_den"], where="amount denominator", positive=True)
        lower_num = _integer(quantity["lower_num"], where="lower numerator")
        lower_den = _integer(
            quantity["lower_den"], where="lower denominator", positive=True)
        upper_num = _integer(quantity["upper_num"], where="upper numerator")
        upper_den = _integer(
            quantity["upper_den"], where="upper denominator", positive=True)
        lower_inclusive = _flag(quantity["lower_inclusive"], where="lower inclusive")
        upper_inclusive = _flag(quantity["upper_inclusive"], where="upper inclusive")
        _integer(quantity["candidate_version"], where="quantity version", positive=True)
        if exactness in {"APPROXIMATE", "RANGE"}:
            if _compare(lower_num, lower_den, upper_num, upper_den) >= 0:
                raise AuthoredComparisonQuantityCourseError("近似/范围上下界非法")
            if (_compare(amount_num, amount_den, lower_num, lower_den) < 0
                    or _compare(amount_num, amount_den, upper_num, upper_den) > 0):
                raise AuthoredComparisonQuantityCourseError("代表数量不在范围内")
        elif exactness in {"EXACT", "COUNT"}:
            if not (
                    _compare(amount_num, amount_den, lower_num, lower_den) == 0
                    and _compare(amount_num, amount_den, upper_num, upper_den) == 0
                    and lower_inclusive == upper_inclusive == 1):
                raise AuthoredComparisonQuantityCourseError("精确数量边界必须相等且闭合")
            if exactness == "COUNT" and amount_den != 1:
                raise AuthoredComparisonQuantityCourseError("计数必须是整数")
        quantities[quantity_id] = quantity
        exactnesses.add(exactness)

    comparisons_value = raw["comparison_candidates"]
    if not isinstance(comparisons_value, list):
        raise AuthoredComparisonQuantityCourseError("comparison_candidates 必须是数组")
    if len(comparisons_value) > ceilings["max_comparisons"]:
        raise AuthoredComparisonQuantityCourseError("比较候选超预算")
    comparisons: dict[str, dict[str, Any]] = {}
    for item in comparisons_value:
        comparison = _exact_keys(item, _COMPARISON_FIELDS, where="comparison candidate")
        comparison_id = _text(comparison["comparison_id"], where="comparison_id")
        if comparison_id in comparisons:
            raise AuthoredComparisonQuantityCourseError("comparison_id 重复")
        left = _text(comparison["left_object_id"], where="left object")
        right = _text(comparison["right_object_id"], where="right object")
        if left not in objects or right not in objects or left == right:
            raise AuthoredComparisonQuantityCourseError("比较对象端点非法")
        standard_id = _text(
            comparison["standard_id"], where="comparison standard",
            allow_empty=True)
        scale_id = _text(
            comparison["scale_id"], where="comparison scale", allow_empty=True)
        if standard_id and standard_id not in standards:
            raise AuthoredComparisonQuantityCourseError("比较引用未知标准")
        if scale_id and scale_id not in scales:
            raise AuthoredComparisonQuantityCourseError("比较引用未知尺度")
        if comparison["operator"] not in COMPARISON_OPERATORS:
            raise AuthoredComparisonQuantityCourseError("comparison operator 未登记")
        if comparison["context_scope_key"] != context_scope:
            raise AuthoredComparisonQuantityCourseError("比较候选跨 ContextScope")
        _integer(comparison["candidate_version"], where="comparison version", positive=True)
        comparisons[comparison_id] = comparison

    quantifiers_value = raw["quantifier_scopes"]
    if not isinstance(quantifiers_value, list):
        raise AuthoredComparisonQuantityCourseError("quantifier_scopes 必须是数组")
    if len(quantifiers_value) > ceilings["max_quantifiers"]:
        raise AuthoredComparisonQuantityCourseError("量词 scope 超预算")
    binder_ids: set[str] = set()
    scope_orders: list[int] = []
    for item in quantifiers_value:
        quantifier = _exact_keys(item, _QUANTIFIER_FIELDS, where="quantifier scope")
        binder_id = _text(quantifier["binder_id"], where="binder_id")
        if binder_id in binder_ids:
            raise AuthoredComparisonQuantityCourseError("binder_id 重复")
        binder_ids.add(binder_id)
        domain = _text(quantifier["domain_object_id"], where="quantifier domain")
        if domain not in objects or objects[domain]["object_kind"] != "SET_EXPR":
            raise AuthoredComparisonQuantityCourseError("量词域必须是 SetExpr")
        if quantifier["quantifier_kind"] not in QUANTIFIER_KINDS:
            raise AuthoredComparisonQuantityCourseError("quantifier_kind 未登记")
        scope_orders.append(_integer(
            quantifier["scope_order"], where="quantifier scope order", positive=True))
        _text(quantifier["proposition_key"], where="quantifier proposition")
        if _flag(quantifier["provisional"], where="quantifier provisional") != 1:
            raise AuthoredComparisonQuantityCourseError("量词 scope 只能 provisional")
        _integer(quantifier["candidate_version"], where="quantifier version", positive=True)
    if scope_orders != list(range(1, len(scope_orders) + 1)):
        raise AuthoredComparisonQuantityCourseError("quantifier scope order 必须连续")

    revision = _exact_keys(
        raw["revision_receipt"], _REVISION_FIELDS, where="revision receipt")
    revision_version = _integer(
        revision["candidate_version"], where="revision version", positive=True)
    dependencies = _string_tuple(
        revision["dependency_keys"], where="dependency_keys",
        sorted_unique=True, allow_empty=True)
    supersedes = _text(
        revision["supersedes_seed_id"], where="revision supersedes",
        allow_empty=True)
    if _flag(revision["raw_observation_preserved"], where="raw preserved") != 1:
        raise AuthoredComparisonQuantityCourseError("revision 不得覆盖 raw Observation")
    if candidate_kind == "REVISION":
        if (revision["revision_scope"] != "QUANTITY_ONLY" or not supersedes
                or revision_version < 2 or "QUANTITY" not in dependencies):
            raise AuthoredComparisonQuantityCourseError("数量修正 receipt 不完整")
    elif (revision["revision_scope"] != "NONE" or supersedes
          or revision_version != 1 or dependencies):
        raise AuthoredComparisonQuantityCourseError("非 revision 不得伪造修正 receipt")

    generation = _exact_keys(
        raw["generation_constraint"], _GENERATION_FIELDS,
        where="generation constraint")
    inputs = _string_tuple(
        generation["input_candidate_ids"], where="generation inputs")
    known_inputs = set(objects) | set(quantities) | set(comparisons)
    if any(item not in known_inputs for item in inputs):
        raise AuthoredComparisonQuantityCourseError("generation 引用未知 typed candidate")
    hidden = _flag(generation["output_surface_hidden"], where="output hidden")
    preservation_keys = (
        "comparison_preservation_required", "quantity_preservation_required",
        "quantifier_scope_preservation_required", "scale_preservation_required",
        "standard_preservation_required", "unit_preservation_required")
    for key in preservation_keys:
        _flag(generation[key], where=key)
    if candidate_kind == "GENERATION":
        if (generation["direction"] != "SEMANTICS_TO_SURFACE" or hidden != 1
                or target_hidden != 1
                or any(generation[key] != 1 for key in preservation_keys)):
            raise AuthoredComparisonQuantityCourseError(
                "generation 必须隐藏输出并逐层保持比较数量语义")
    elif hidden != 0 or target_hidden != 0:
        raise AuthoredComparisonQuantityCourseError("非 generation 不得隐藏目标表层")

    split = _exact_keys(raw["split_identity"], _SPLIT_FIELDS, where="split")
    object_family = _text(split["object_family"], where="split object_family")
    scale_family = _text(split["scale_family"], where="split scale_family")
    unit_family = _text(split["unit_family"], where="split unit_family")
    expected_object_kind = _OBJECT_KIND_BY_FAMILY.get(object_family)
    expected_dimension = _DIMENSION_BY_SCALE.get(scale_family)
    expected_unit = _UNIT_BY_SCALE.get(scale_family)
    if (expected_object_kind is None or expected_dimension is None
            or expected_unit is None):
        raise AuthoredComparisonQuantityCourseError("组合 split family 未登记")
    if any(
            objects[item["object_id"]]["object_kind"] != expected_object_kind
            for item in quantities_value):
        raise AuthoredComparisonQuantityCourseError(
            "组合 object_family 与数量对象不一致")
    if any(
            item["dimension_key"] != expected_dimension
            for item in scales_value):
        raise AuthoredComparisonQuantityCourseError(
            "组合 scale_family 与尺度维度不一致")
    if units_value and (
            unit_family != expected_unit
            or any(item["canonical_unit_id"] != expected_unit
                   for item in units_value)):
        raise AuthoredComparisonQuantityCourseError(
            "组合 unit_family 与 canonical unit 不一致")
    combination_key = f"{object_family}::{scale_family}::{unit_family}"
    if (split["combination_key"] != combination_key
            or split["isolation_axis"] != "OBJECT_X_SCALE_X_UNIT"):
        raise AuthoredComparisonQuantityCourseError("组合 split identity 漂移")

    retention_anchor = _text(
        raw["retention_anchor_id"], where="retention anchor", allow_empty=True)
    if candidate_kind == "RETENTION" and not retention_anchor:
        raise AuthoredComparisonQuantityCourseError("retention 必须绑定更早 anchor")
    if candidate_kind != "RETENTION" and retention_anchor:
        raise AuthoredComparisonQuantityCourseError("非 retention 不得绑定 anchor")

    bare_property = int(baseline == "BARE_PROPERTY_ONLY")
    unit_erasure = int(baseline == "UNIT_ERASURE")
    if candidate_kind == "BARE_PROPERTY_BASELINE":
        if not bare_property or scales or standards or comparisons:
            raise AuthoredComparisonQuantityCourseError(
                "裸 PROPERTY 不得冒充比较标准与尺度")
    elif bare_property:
        raise AuthoredComparisonQuantityCourseError("bare-property 基线身份错配")
    if candidate_kind == "UNIT_ERASURE_BASELINE":
        if not unit_erasure or units or any(item["unit_id"] for item in quantities_value):
            raise AuthoredComparisonQuantityCourseError("单位擦除基线必须显式缺单位")
    elif unit_erasure:
        raise AuthoredComparisonQuantityCourseError("unit-erasure 基线身份错配")

    if candidate_kind in {
            "COMPARISON", "DEGREE", "MEASURE", "RANGE", "RETENTION",
            "REVISION", "GENERATION"}:
        if not scales or not standards or not units:
            raise AuthoredComparisonQuantityCourseError("typed 比较缺 scale/standard/unit")
    if candidate_kind == "COMPARISON" and not comparisons:
        raise AuthoredComparisonQuantityCourseError("比较候选缺失")
    if candidate_kind == "DEGREE" and not any(
            item["threshold_num"] > 0 for item in scales_value):
        raise AuthoredComparisonQuantityCourseError("程度尺度阈值缺失")
    if candidate_kind == "QUANTITY_COUNT":
        if "COUNT" not in exactnesses or not quantifiers_value:
            raise AuthoredComparisonQuantityCourseError("计数或数量 scope 缺失")
    if candidate_kind == "MEASURE" and any(
            not item["unit_id"] for item in quantities_value):
        raise AuthoredComparisonQuantityCourseError("度量必须绑定单位")
    if candidate_kind == "RANGE" and "RANGE" not in exactnesses:
        raise AuthoredComparisonQuantityCourseError("范围候选缺失")
    if candidate_kind == "APPROXIMATE_EXACT" and not {
            "APPROXIMATE", "EXACT"}.issubset(exactnesses):
        raise AuthoredComparisonQuantityCourseError("近似与精确未分账")
    if candidate_kind == "QUANTIFIER_SCOPE":
        if len(quantifiers_value) < 2:
            raise AuthoredComparisonQuantityCourseError("量词 scope competition 缺失")
    if candidate_kind == "AMBIGUOUS_STANDARD":
        if len(standards) < 2 or len(comparisons) < 2:
            raise AuthoredComparisonQuantityCourseError("同表层必须保留多个标准候选")
    if candidate_kind == "UNKNOWN" and not (
            "UNKNOWN" in exactnesses
            and any(item["direction"] == "UNKNOWN" for item in scales_value)
            and any(item["standard_kind"] == "UNKNOWN" for item in standards_value)):
        raise AuthoredComparisonQuantityCourseError("数量/标准 unknown 未显式表示")

    return ComparisonQuantityPayloadAudit(
        len(objects), len(scales), len(standards), len(units), len(quantities),
        len(comparisons), len(quantifiers_value), combination_key,
        bare_property, unit_erasure)


@dataclass(frozen=True)
class AuthoredComparisonQuantitySeed:
    """一个 owner、split、比较数量候选集合和独立 label。"""

    seed_id: str
    family: str
    template_family: str
    label_owner: str
    split: str
    sample_family: str
    sample_role: str
    candidate_kind: str
    observed_text: str
    surface_scope: CanonicalJsonObject
    object_candidates: tuple[CanonicalJsonObject, ...]
    scale_definitions: tuple[CanonicalJsonObject, ...]
    standard_candidates: tuple[CanonicalJsonObject, ...]
    unit_definitions: tuple[CanonicalJsonObject, ...]
    quantity_candidates: tuple[CanonicalJsonObject, ...]
    comparison_candidates: tuple[CanonicalJsonObject, ...]
    quantifier_scopes: tuple[CanonicalJsonObject, ...]
    generation_constraint: CanonicalJsonObject
    revision_receipt: CanonicalJsonObject
    resource_budget: CanonicalJsonObject
    object_family: str
    scale_family: str
    unit_family: str
    baseline_kind: str
    objective_keys: tuple[str, ...]
    expected_state: str
    expected_payload: CanonicalJsonObject
    evaluation_dimension: str
    perturbation_kind: str
    supersedes_seed_id: str
    retention_anchor_id: str
    logical_order: int
    license_id: str

    def __post_init__(self) -> None:
        for name, value in (
                ("seed_id", self.seed_id), ("family", self.family),
                ("template_family", self.template_family),
                ("object_family", self.object_family),
                ("scale_family", self.scale_family),
                ("unit_family", self.unit_family),
                ("perturbation_kind", self.perturbation_kind)):
            _text(value, where=name)
        if self.label_owner not in {"teacher", "evaluator"}:
            raise AuthoredComparisonQuantityCourseError("label_owner 非法")
        if self.split != ("train" if self.label_owner == "teacher" else "held_out"):
            raise AuthoredComparisonQuantityCourseError("label_owner 与 split 不一致")
        if self.sample_family not in SAMPLE_FAMILIES:
            raise AuthoredComparisonQuantityCourseError("sample_family 未登记")
        if self.sample_role not in SAMPLE_ROLES:
            raise AuthoredComparisonQuantityCourseError("sample_role 未登记")
        if self.candidate_kind not in CANDIDATE_KINDS:
            raise AuthoredComparisonQuantityCourseError("candidate_kind 未登记")
        _text(self.observed_text, where="observed_text")
        for name in (
                "surface_scope", "generation_constraint", "revision_receipt",
                "resource_budget"):
            if not isinstance(getattr(self, name), CanonicalJsonObject):
                raise AuthoredComparisonQuantityCourseError(f"{name} 类型非法")
        for name in (
                "object_candidates", "scale_definitions", "standard_candidates",
                "unit_definitions", "quantity_candidates", "comparison_candidates",
                "quantifier_scopes"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or any(
                    not isinstance(item, CanonicalJsonObject) for item in values):
                raise AuthoredComparisonQuantityCourseError(f"{name} 类型非法")
        if not self.object_candidates or not self.quantity_candidates:
            raise AuthoredComparisonQuantityCourseError("对象和数量候选不能为空")
        if self.baseline_kind not in BASELINE_KINDS:
            raise AuthoredComparisonQuantityCourseError("baseline_kind 未登记")
        if self.objective_keys != tuple(sorted(set(self.objective_keys))):
            raise AuthoredComparisonQuantityCourseError("objective_keys 必须排序去重")
        if not self.objective_keys or any(
                item not in LANGUAGE_OBJECTIVE_KEYS for item in self.objective_keys):
            raise AuthoredComparisonQuantityCourseError("objective_keys 未登记")
        if self.expected_state not in EXPECTED_STATES:
            raise AuthoredComparisonQuantityCourseError("expected_state 非四态")
        if not isinstance(self.expected_payload, CanonicalJsonObject):
            raise AuthoredComparisonQuantityCourseError("expected_payload 类型非法")
        _validate_expected(self.expected_payload, expected_state=self.expected_state)
        if self.evaluation_dimension not in EVALUATOR_DIMENSIONS:
            raise AuthoredComparisonQuantityCourseError("evaluation_dimension 未登记")
        _text(self.supersedes_seed_id, where="supersedes_seed_id", allow_empty=True)
        _text(self.retention_anchor_id, where="retention_anchor_id", allow_empty=True)
        _integer(self.logical_order, where="logical_order", positive=True)
        if self.license_id != LICENSE_ID:
            raise AuthoredComparisonQuantityCourseError("原创 LC-06 课程必须为 CC0-1.0")
        validate_comparison_quantity_payload(self.observation_payload())

    def observation_payload(self) -> CanonicalJsonObject:
        generation = self.generation_constraint.to_value()
        return CanonicalJsonObject.from_value({
            "baseline_kind": self.baseline_kind,
            "candidate_kind": self.candidate_kind,
            "comparison_candidates": [
                item.to_value() for item in self.comparison_candidates],
            "generation_constraint": generation,
            "object_candidates": [item.to_value() for item in self.object_candidates],
            "objective_keys": list(self.objective_keys),
            "observed_surface": {
                "append_only": 1,
                "sha256": _sha256_text(self.observed_text),
                "target_hidden": generation["output_surface_hidden"],
                "text": self.observed_text,
            },
            "quantifier_scopes": [item.to_value() for item in self.quantifier_scopes],
            "quantity_candidates": [item.to_value() for item in self.quantity_candidates],
            "resource_budget": self.resource_budget.to_value(),
            "retention_anchor_id": self.retention_anchor_id,
            "revision_receipt": self.revision_receipt.to_value(),
            "sample_family": self.sample_family,
            "scale_definitions": [item.to_value() for item in self.scale_definitions],
            "selection_state": "UNSELECTED",
            "split_identity": {
                "combination_key": (
                    f"{self.object_family}::{self.scale_family}::{self.unit_family}"),
                "isolation_axis": "OBJECT_X_SCALE_X_UNIT",
                "object_family": self.object_family,
                "scale_family": self.scale_family,
                "unit_family": self.unit_family,
            },
            "standard_candidates": [item.to_value() for item in self.standard_candidates],
            "surface_scope": self.surface_scope.to_value(),
            "unit_definitions": [item.to_value() for item in self.unit_definitions],
        })

    def compiled_seed(self) -> AuthoredCompiledSeed:
        payload = self.observation_payload()
        audit = validate_comparison_quantity_payload(payload)
        quantity_ids = tuple(
            item.to_value()["quantity_id"] for item in self.quantity_candidates)
        return AuthoredCompiledSeed(
            self.seed_id,
            self.family,
            self.template_family,
            self.label_owner,
            self.split,
            self.sample_role,
            PAYLOAD_KIND,
            payload,
            self.expected_state,
            self.expected_payload,
            self.perturbation_kind,
            self.supersedes_seed_id,
            self.logical_order,
            (self.seed_id, quantity_ids, audit.combination_key),
            (self.observed_text, self.family),
            (
                self.candidate_kind, self.object_family, self.scale_family,
                self.unit_family, audit.object_count, audit.scale_count,
                audit.standard_count, audit.unit_count, audit.quantity_count,
                audit.comparison_count, audit.quantifier_count,
            ),
            self.evaluation_dimension,
        )


def _canonical_objects(
        values: Any, fields: set[str], *, where: str,
        ) -> tuple[CanonicalJsonObject, ...]:
    if not isinstance(values, list):
        raise AuthoredComparisonQuantityCourseError(f"{where} 必须是数组")
    return tuple(CanonicalJsonObject.from_value(
        _exact_keys(item, fields, where=where)) for item in values)


def _seed_from_value(value: dict[str, Any]) -> AuthoredComparisonQuantitySeed:
    raw = _exact_keys(value, _SEED_FIELDS, where="comparison quantity seed")
    expected = raw["expected_payload"]
    if not isinstance(expected, dict):
        raise AuthoredComparisonQuantityCourseError("expected_payload 必须是对象")
    return AuthoredComparisonQuantitySeed(
        _text(raw["seed_id"], where="seed_id"),
        _text(raw["family"], where="family"),
        _text(raw["template_family"], where="template_family"),
        _text(raw["label_owner"], where="label_owner"),
        _text(raw["split"], where="split"),
        _text(raw["sample_family"], where="sample_family"),
        _text(raw["sample_role"], where="sample_role"),
        _text(raw["candidate_kind"], where="candidate_kind"),
        _text(raw["observed_text"], where="observed_text"),
        CanonicalJsonObject.from_value(_exact_keys(
            raw["surface_scope"], _SCOPE_FIELDS, where="surface_scope")),
        _canonical_objects(raw["object_candidates"], _OBJECT_FIELDS, where="objects"),
        _canonical_objects(raw["scale_definitions"], _SCALE_FIELDS, where="scales"),
        _canonical_objects(
            raw["standard_candidates"], _STANDARD_FIELDS, where="standards"),
        _canonical_objects(raw["unit_definitions"], _UNIT_FIELDS, where="units"),
        _canonical_objects(
            raw["quantity_candidates"], _QUANTITY_FIELDS, where="quantities"),
        _canonical_objects(
            raw["comparison_candidates"], _COMPARISON_FIELDS,
            where="comparisons"),
        _canonical_objects(
            raw["quantifier_scopes"], _QUANTIFIER_FIELDS, where="quantifiers"),
        CanonicalJsonObject.from_value(_exact_keys(
            raw["generation_constraint"], _GENERATION_FIELDS,
            where="generation_constraint")),
        CanonicalJsonObject.from_value(_exact_keys(
            raw["revision_receipt"], _REVISION_FIELDS, where="revision_receipt")),
        CanonicalJsonObject.from_value(_exact_keys(
            raw["resource_budget"], _BUDGET_FIELDS, where="resource_budget")),
        _text(raw["object_family"], where="object_family"),
        _text(raw["scale_family"], where="scale_family"),
        _text(raw["unit_family"], where="unit_family"),
        _text(raw["baseline_kind"], where="baseline_kind"),
        _string_tuple(
            raw["objective_keys"], where="objective_keys", sorted_unique=True),
        _text(raw["expected_state"], where="expected_state"),
        CanonicalJsonObject.from_value(_exact_keys(
            expected, _EXPECTED_FIELDS, where="expected_payload")),
        _text(raw["evaluation_dimension"], where="evaluation_dimension"),
        _text(raw["perturbation_kind"], where="perturbation_kind"),
        _text(raw["supersedes_seed_id"], where="supersedes_seed_id", allow_empty=True),
        _text(raw["retention_anchor_id"], where="retention_anchor_id", allow_empty=True),
        _integer(raw["logical_order"], where="logical_order", positive=True),
        _text(raw["license_id"], where="license_id"),
    )


def read_authored_comparison_quantity_seeds(
        path: str | Path) -> tuple[AuthoredComparisonQuantitySeed, ...]:
    """严格读取 JSONL，并核双 owner、修正、retention 和组合 held-out。"""
    try:
        payload = Path(path).read_bytes()
    except OSError as error:
        raise AuthoredComparisonQuantityCourseError(
            "comparison quantity sample 无法读取") from error
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise AuthoredComparisonQuantityCourseError("comparison sample 换行非法")
    seeds: list[AuthoredComparisonQuantitySeed] = []
    try:
        for line in payload.splitlines(keepends=True):
            value = parse_canonical_json_bytes(line[:-1], require_object=True)
            assert isinstance(value, dict)
            if canonical_json_line(value) != line:
                raise AuthoredComparisonQuantityCourseError(
                    "comparison seed 非规范 JSON")
            seeds.append(_seed_from_value(value))
    except AuthoredComparisonQuantityCourseError:
        raise
    except Exception as error:
        raise AuthoredComparisonQuantityCourseError(
            "comparison quantity seed 损坏") from error
    identifiers = [item.seed_id for item in seeds]
    orders = [item.logical_order for item in seeds]
    if not seeds or len(identifiers) != len(set(identifiers)):
        raise AuthoredComparisonQuantityCourseError("seed_id 空或重复")
    if orders != sorted(orders) or len(orders) != len(set(orders)):
        raise AuthoredComparisonQuantityCourseError("logical_order 必须严格递增")
    index = {item.seed_id: item for item in seeds}
    for seed in seeds:
        expected_role = _ROLE_BY_FAMILY.get(seed.sample_family)
        if expected_role is not None and seed.sample_role != expected_role:
            raise AuthoredComparisonQuantityCourseError("sample family/role 不一致")
        if seed.sample_family == "GENERATION" and seed.sample_role not in {
                "support", "read_only_probe", "refute"}:
            raise AuthoredComparisonQuantityCourseError("generation sample_role 非法")
        expected_state = _STATE_BY_FAMILY.get(seed.sample_family)
        if expected_state is not None and seed.expected_state != expected_state:
            raise AuthoredComparisonQuantityCourseError("sample family/state 不一致")
        if seed.sample_family == "GENERATION" and seed.expected_state not in {
                "TRUE", "FALSE"}:
            raise AuthoredComparisonQuantityCourseError("generation state 非法")
        if seed.candidate_kind == "REVISION":
            target = index.get(seed.supersedes_seed_id)
            if (target is None or target.logical_order >= seed.logical_order
                    or target.family != seed.family or target.split != seed.split):
                raise AuthoredComparisonQuantityCourseError(
                    "revision 必须 supersede 同族更早 seed")
            old = target.observation_payload().to_value()
            new = seed.observation_payload().to_value()
            for key in (
                    "object_candidates", "scale_definitions", "standard_candidates",
                    "unit_definitions", "comparison_candidates", "quantifier_scopes"):
                if old[key] != new[key]:
                    raise AuthoredComparisonQuantityCourseError(
                        "数量 revision 不得替换标准/尺度/单位/scope 身份")
            if old["quantity_candidates"] == new["quantity_candidates"]:
                raise AuthoredComparisonQuantityCourseError("数量 revision 必须改动数量")
            if (new["revision_receipt"]["candidate_version"]
                    <= old["revision_receipt"]["candidate_version"]):
                raise AuthoredComparisonQuantityCourseError("revision version 必须前进")
        elif seed.supersedes_seed_id:
            raise AuthoredComparisonQuantityCourseError("非 revision 不得 supersede")
        if seed.candidate_kind == "RETENTION":
            anchor = index.get(seed.retention_anchor_id)
            if (anchor is None or anchor.logical_order >= seed.logical_order
                    or anchor.split != seed.split):
                raise AuthoredComparisonQuantityCourseError(
                    "retention anchor 必须是同 split 更早 seed")
        elif seed.retention_anchor_id:
            raise AuthoredComparisonQuantityCourseError("非 retention 不得绑定 anchor")

    teacher = tuple(item for item in seeds if item.label_owner == "teacher")
    evaluator = tuple(item for item in seeds if item.label_owner == "evaluator")
    for owner, owner_seeds in (("teacher", teacher), ("evaluator", evaluator)):
        if {item.sample_family for item in owner_seeds} != set(SAMPLE_FAMILIES):
            raise AuthoredComparisonQuantityCourseError(
                f"{owner} 未覆盖七类 sample family")
        if {item.candidate_kind for item in owner_seeds} != set(CANDIDATE_KINDS):
            raise AuthoredComparisonQuantityCourseError(
                f"{owner} 未覆盖完整比较数量候选族")
        bare = tuple(item for item in owner_seeds
                     if item.baseline_kind == "BARE_PROPERTY_ONLY")
        erased = tuple(item for item in owner_seeds
                       if item.baseline_kind == "UNIT_ERASURE")
        if (len(bare) != 1 or bare[0].expected_state != "FALSE"
                or len(erased) != 1 or erased[0].expected_state != "FALSE"):
            raise AuthoredComparisonQuantityCourseError(
                f"{owner} 两类比较负基线未冻结")
    if ({item.family for item in teacher} & {item.family for item in evaluator}
            or {item.template_family for item in teacher}
            & {item.template_family for item in evaluator}):
        raise AuthoredComparisonQuantityCourseError(
            "teacher/evaluator family/template 泄漏")
    teacher_objects = {item.object_family for item in teacher}
    teacher_scales = {item.scale_family for item in teacher}
    teacher_units = {item.unit_family for item in teacher}
    teacher_triples = {
        (item.object_family, item.scale_family, item.unit_family)
        for item in teacher
    }
    held_out = {
        (item.object_family, item.scale_family, item.unit_family)
        for item in evaluator
        if item.object_family in teacher_objects
        and item.scale_family in teacher_scales
        and item.unit_family in teacher_units
        and (item.object_family, item.scale_family, item.unit_family)
        not in teacher_triples
    }
    if len(held_out) < 2:
        raise AuthoredComparisonQuantityCourseError(
            "held-out object×scale×unit 组合不足")
    if {item.evaluation_dimension for item in evaluator} != set(
            EVALUATOR_DIMENSIONS):
        raise AuthoredComparisonQuantityCourseError("独立 evaluator 维度未列全")
    return tuple(seeds)


def _course_spec() -> AuthoredCourseSpec:
    return AuthoredCourseSpec(
        SOURCE_KEY,
        LICENSE_ID,
        COURSE_VERSION,
        ARTIFACT_VERSION,
        ADAPTER_VERSION,
        GENERATOR_VERSION,
        PARSER_VERSION,
        PACK_NAME,
        STAGE,
        SUBSTAGE,
        "authored-comparison-quantity-seed-v1",
        "urn:pure-integer-ai:ph2:authored-comparison-quantity-v1",
        "Pure Integer AI PH2 authored comparison quantity and measure seed",
        "COMPARISON_QUANTITY_LABEL",
        "lc06-comparison-quantity",
        384,
    )


def compile_authored_comparison_quantity_course(
        sample_path: str | Path,
        release_root: str | Path) -> AuthoredCourseBuild:
    seeds = read_authored_comparison_quantity_seeds(sample_path)
    return publish_authored_course(
        tuple(item.compiled_seed() for item in seeds),
        sample_path,
        release_root,
        _course_spec(),
    )


def build_comparison_quantity_course_manifest(
        sample_path: str | Path,
        build: AuthoredCourseBuild,
        *,
        artifact_relative_root: str = FORMAL_ARTIFACT_RELATIVE_ROOT,
        ) -> CapabilityCourseManifest:
    sample = Path(sample_path)
    seeds = read_authored_comparison_quantity_seeds(sample)
    teacher = tuple(item for item in seeds if item.label_owner == "teacher")
    evaluator = tuple(item for item in seeds if item.label_owner == "evaluator")
    return CapabilityCourseManifest(
        1,
        COURSE_MANIFEST_ARTIFACT_VERSION,
        "COURSE_FROZEN",
        "NOT_STARTED",
        ("LC-06",),
        ("COMPARISON_QUANTITY_MEASURE",),
        STAGE,
        SUBSTAGE,
        SOURCE_KEY,
        LICENSE_ID,
        f"data/ph2/{sample.name}",
        hashlib.sha256(sample.read_bytes()).hexdigest(),
        len(seeds),
        SAMPLE_FAMILIES,
        COURSE_SPLIT_AXES,
        tuple(sorted({item.family for item in teacher})),
        tuple(sorted({item.family for item in evaluator})),
        tuple(sorted({item.template_family for item in teacher})),
        tuple(sorted({item.template_family for item in evaluator})),
        PAYLOAD_KIND,
        PAYLOAD_KEYS,
        tuple(sorted({key for item in seeds for key in item.objective_keys})),
        EVALUATOR_DIMENSIONS,
        VERIFIER_NE_CONDITIONS,
        RETENTION_PROTOCOLS,
        COMBINATION_AXES,
        BASELINE_KINDS,
        ABLATION_KEYS,
        f"{artifact_relative_root}/packs/{PACK_NAME}/manifest.json",
        build.manifest.sha256(),
        build.manifest.record_count,
        build.manifest.splits,
        CanonicalJsonObject.from_value(COURSE_INVARIANTS),
        CanonicalJsonObject.from_value(COURSE_EXECUTION_STATE),
    )


def _object_kind(family: str) -> str:
    return _OBJECT_KIND_BY_FAMILY[family]


def _quantity(
        token: str,
        object_id: str,
        scale_id: str,
        unit_id: str,
        exactness: str,
        *,
        version: int,
        amount: int,
        lower: int | None = None,
        upper: int | None = None,
        ) -> dict[str, Any]:
    lower_value = amount if lower is None else lower
    upper_value = amount if upper is None else upper
    return {
        "amount_den": 1,
        "amount_num": amount,
        "candidate_version": version,
        "exactness": exactness,
        "lower_den": 1,
        "lower_inclusive": 1,
        "lower_num": lower_value,
        "object_id": object_id,
        "quantity_id": f"{token}_QUANTITY",
        "scale_id": scale_id,
        "unit_id": unit_id,
        "upper_den": 1,
        "upper_inclusive": 1,
        "upper_num": upper_value,
    }


def _scenario_components(
        owner: str,
        ordinal: int,
        candidate_kind: str,
        object_family: str,
        scale_family: str,
        unit_family: str,
        *,
        revision_version: int = 1,
        shared_revision_identity: bool = False,
        ) -> dict[str, Any]:
    token = f"{owner}_{ordinal:02d}"
    identity_token = f"{owner}_REVISION" if shared_revision_identity else token
    scope = f"{owner}_CTX_REVISION" if shared_revision_identity else f"{token}_CTX"
    first_id = f"{identity_token}_OBJECT_1"
    second_id = f"{identity_token}_OBJECT_2"
    objects = [{
        "context_scope_key": scope,
        "object_id": first_id,
        "object_kind": _object_kind(object_family),
        "proposition_key": f"{identity_token}_PROPOSITION_1",
    }]
    needs_second = candidate_kind in {
        "AMBIGUOUS_STANDARD", "COMPARISON", "GENERATION"}
    if needs_second:
        objects.append({
            "context_scope_key": scope,
            "object_id": second_id,
            "object_kind": "ENTITY",
            "proposition_key": f"{identity_token}_PROPOSITION_2",
        })
    if candidate_kind in {"QUANTITY_COUNT", "QUANTIFIER_SCOPE"}:
        if objects[0]["object_kind"] != "SET_EXPR":
            objects[0]["object_kind"] = "SET_EXPR"

    bare = candidate_kind == "BARE_PROPERTY_BASELINE"
    erased = candidate_kind == "UNIT_ERASURE_BASELINE"
    dimension = _DIMENSION_BY_SCALE[scale_family]
    scale_id = "" if bare else f"{identity_token}_SCALE"
    unit_id = "" if (bare or erased) else f"{identity_token}_UNIT"
    scales = [] if bare else [{
        "context_scope_key": scope,
        "dimension_key": dimension,
        "direction": "UNKNOWN" if scale_family == "UNKNOWN" else "INCREASING",
        "scale_id": scale_id,
        "threshold_den": 1,
        "threshold_num": 5 if candidate_kind == "DEGREE" else 0,
    }]
    units = [] if (bare or erased) else [{
        "canonical_unit_id": _UNIT_BY_SCALE[scale_family],
        "context_scope_key": scope,
        "conversion_den": 1,
        "conversion_num": 1,
        "dimension_key": dimension,
        "unit_id": unit_id,
    }]
    standards = [] if bare else [{
        "context_scope_key": scope,
        "reference_object_id": (
            second_id if needs_second else first_id),
        "standard_id": f"{identity_token}_STANDARD_1",
        "standard_kind": "UNKNOWN" if scale_family == "UNKNOWN" else "EXPLICIT",
    }]
    if candidate_kind == "AMBIGUOUS_STANDARD":
        standards.append({
            "context_scope_key": scope,
            "reference_object_id": first_id,
            "standard_id": f"{identity_token}_STANDARD_2",
            "standard_kind": "CONTEXT",
        })

    exactness = {
        "QUANTITY_COUNT": "COUNT",
        "RANGE": "RANGE",
        "UNKNOWN": "UNKNOWN",
    }.get(candidate_kind, "EXACT")
    amount = 12 if revision_version == 1 else 15
    quantities = [_quantity(
        identity_token, first_id, scale_id, unit_id, exactness,
        version=revision_version, amount=amount,
        lower=10 if exactness == "RANGE" else None,
        upper=14 if exactness == "RANGE" else None)]
    if candidate_kind == "APPROXIMATE_EXACT":
        quantities = [
            _quantity(
                f"{identity_token}_APPROX", first_id, scale_id, unit_id,
                "APPROXIMATE", version=1, amount=12, lower=10, upper=14),
            _quantity(
                f"{identity_token}_EXACT", first_id, scale_id, unit_id,
                "EXACT", version=1, amount=12),
        ]

    comparisons = []
    if candidate_kind in {"AMBIGUOUS_STANDARD", "COMPARISON", "GENERATION"}:
        comparisons.append({
            "candidate_version": revision_version,
            "comparison_id": f"{identity_token}_COMPARISON_1",
            "context_scope_key": scope,
            "left_object_id": first_id,
            "operator": "GT",
            "right_object_id": second_id,
            "scale_id": scale_id,
            "standard_id": standards[0]["standard_id"],
        })
    if candidate_kind == "AMBIGUOUS_STANDARD":
        comparisons.append({
            "candidate_version": 1,
            "comparison_id": f"{identity_token}_COMPARISON_2",
            "context_scope_key": scope,
            "left_object_id": first_id,
            "operator": "LT",
            "right_object_id": second_id,
            "scale_id": scale_id,
            "standard_id": standards[1]["standard_id"],
        })
    if candidate_kind == "UNKNOWN":
        standards[0]["reference_object_id"] = ""

    quantifiers = []
    if candidate_kind in {"QUANTITY_COUNT", "QUANTIFIER_SCOPE"}:
        quantifiers.append({
            "binder_id": f"{identity_token}_BINDER_1",
            "candidate_version": 1,
            "domain_object_id": first_id,
            "proposition_key": f"{identity_token}_QUANTIFIED_PROPOSITION_1",
            "provisional": 1,
            "quantifier_kind": "COUNT_EXACT",
            "scope_order": 1,
        })
    if candidate_kind == "QUANTIFIER_SCOPE":
        quantifiers.append({
            "binder_id": f"{identity_token}_BINDER_2",
            "candidate_version": 1,
            "domain_object_id": first_id,
            "proposition_key": f"{identity_token}_QUANTIFIED_PROPOSITION_2",
            "provisional": 1,
            "quantifier_kind": "FORALL",
            "scope_order": 2,
        })
    return {
        "comparison_candidates": comparisons,
        "object_candidates": objects,
        "quantifier_scopes": quantifiers,
        "quantity_candidates": quantities,
        "scale_definitions": scales,
        "scope": scope,
        "standard_candidates": standards,
        "unit_definitions": units,
    }


_SCENARIOS = (
    ("COMPARISON", "POSITIVE", "ENTITY", "TEMPERATURE", "CELSIUS", "COMPARISON_DIRECTION_STANDARD"),
    ("DEGREE", "POSITIVE", "ENTITY", "UNITLESS", "UNITLESS", "DEGREE_SCALE_THRESHOLD"),
    ("QUANTITY_COUNT", "POSITIVE", "SET", "COUNT", "COUNT_UNIT", "QUANTITY_COUNT"),
    ("MEASURE", "POSITIVE", "ENTITY", "LENGTH", "METER", "MEASURE_UNIT_DIMENSION"),
    ("RANGE", "POSITIVE", "ENTITY", "TEMPERATURE", "CELSIUS", "RANGE_BOUNDARY"),
    ("APPROXIMATE_EXACT", "POSITIVE", "ENTITY", "LENGTH", "METER", "APPROXIMATE_EXACT_DISTINCTION"),
    ("QUANTIFIER_SCOPE", "POSITIVE", "SET", "COUNT", "COUNT_UNIT", "QUANTIFIER_SCOPE"),
    ("AMBIGUOUS_STANDARD", "AMBIGUOUS", "ENTITY", "TEMPERATURE", "CELSIUS", "AMBIGUOUS_STANDARD_COMPETITION"),
    ("BARE_PROPERTY_BASELINE", "NEGATIVE", "ENTITY", "UNITLESS", "UNITLESS", "BARE_PROPERTY_REJECT"),
    ("UNIT_ERASURE_BASELINE", "NEGATIVE", "ENTITY", "LENGTH", "UNITLESS", "UNIT_ERASURE_REJECT"),
    ("REVISION", "REVISION", "ENTITY", "LENGTH", "METER", "LOCAL_QUANTITY_REVISION"),
    ("RETENTION", "RETENTION", "ENTITY", "TEMPERATURE", "CELSIUS", "RETENTION_REVERIFY"),
    ("UNKNOWN", "UNKNOWN", "ENTITY", "UNKNOWN", "UNKNOWN_UNIT", "QUANTITY_UNKNOWN"),
    ("GENERATION", "GENERATION", "EVENT", "DURATION", "HOUR", "REVERSE_GENERATION_COMPARISON_MEASURE"),
)
_EVALUATOR_OBJECT_OVERRIDES = {
    1: "SET",
    2: "EVENT",
    4: "SET",
    5: "EVENT",
    6: "SET",
    11: "SET",
    14: "SET",
}
_TEACHER_TEXTS = (
    "甲杯的水温高于乙杯，以乙杯为比较标准。",
    "这盏灯的亮度超过约定阈值。",
    "封闭集合中恰有十二个标记项。",
    "绳子的长度是十二米。",
    "合格温度范围是十到十四摄氏度，含两端。",
    "约十二米与精确十二米是不同数量候选。",
    "对集合先计数为十二，再在全称 scope 下核属性。",
    "这间屋子更冷，但比较标准可能是走廊或室外。",
    "墙面是冷的。",
    "绳子的长度是十二，但没有给出单位。",
    "后来修正：绳子的长度是十五米。",
    "再次核对：水温仍为十二摄氏度。",
    "物体的数量与单位目前未知。",
    "语义输入：事件持续十二小时，并且甲时段长于乙时段。",
)
_EVALUATOR_TEXTS = (
    "这组样本的温度高于参照杯，以参照杯为标准。",
    "该事件的强度超过约定阈值。",
    "实体清单中恰有十二项。",
    "集合边界的总长度是十二米。",
    "该事件的温度范围是十到十四摄氏度。",
    "该集合约十二米与精确十二米不得合并。",
    "实体域先计数，再在全称 scope 下核属性。",
    "这条路径更长，但标准可能是甲线或乙线。",
    "材料是重的。",
    "边界长度是十二，但单位缺失。",
    "补充修正：集合边界总长度是十五米。",
    "只读重验：参照杯温度仍为十二摄氏度。",
    "该对象的量级仍未知。",
    "语义输入：集合过程持续十二小时，且甲段长于乙段。",
)
_TEACHER_ACCEPTED = "该过程持续十二小时，且甲时段比乙时段更长。"
_EVALUATOR_ACCEPTED = "该集合过程持续十二小时，且甲段长于乙段。"


def build_default_comparison_quantity_seed_values() -> tuple[dict[str, Any], ...]:
    """构造冻结的双 owner 原创样本；调用方只负责规范落盘。"""
    rows: list[dict[str, Any]] = []
    logical_order = 0
    for owner, texts in (("T", _TEACHER_TEXTS), ("E", _EVALUATOR_TEXTS)):
        label_owner = "teacher" if owner == "T" else "evaluator"
        split = "train" if owner == "T" else "held_out"
        ids = {
            index: f"{owner}-lc06-{index:02d}-{scenario[0].lower().replace('_', '-')}"
            for index, scenario in enumerate(_SCENARIOS, start=1)
        }
        for index, scenario in enumerate(_SCENARIOS, start=1):
            logical_order += 1
            candidate_kind, sample_family, object_family, scale_family, unit_family, dimension = scenario
            if owner == "E" and index in _EVALUATOR_OBJECT_OVERRIDES:
                object_family = _EVALUATOR_OBJECT_OVERRIDES[index]
            shared_revision = index in {4, 11}
            revision_version = 2 if index == 11 else 1
            components = _scenario_components(
                owner, index, candidate_kind, object_family, scale_family,
                unit_family, revision_version=revision_version,
                shared_revision_identity=shared_revision)
            supersedes = ids[4] if index == 11 else ""
            retention_anchor = ids[1] if index == 12 else ""
            baseline = {
                9: "BARE_PROPERTY_ONLY",
                10: "UNIT_ERASURE",
            }.get(index, "TYPED_COMPARISON_OBJECTS_PRESENT")
            expected_state = {
                "AMBIGUOUS": "CONFLICT",
                "NEGATIVE": "FALSE",
                "UNKNOWN": "UNKNOWN",
            }.get(sample_family, "TRUE")
            accepted_surfaces: list[str] = []
            if expected_state == "TRUE":
                if candidate_kind == "GENERATION":
                    accepted_surfaces = [
                        _TEACHER_ACCEPTED if owner == "T" else _EVALUATOR_ACCEPTED]
                else:
                    accepted_surfaces = [texts[index - 1]]
            scope = components.pop("scope")
            generation_hidden = int(candidate_kind == "GENERATION")
            known_ids = [
                *[item["object_id"] for item in components["object_candidates"]],
                *[item["quantity_id"] for item in components["quantity_candidates"]],
                *[item["comparison_id"] for item in components["comparison_candidates"]],
            ]
            family_suffix = "REVISION_CHAIN" if index in {4, 11} else candidate_kind
            objective_keys = {
                "AMBIGUOUS_STANDARD": ("CROSS_CONTEXT_CONSISTENCY",),
                "GENERATION": ("GENERATION_ADOPTION", "GENERATION_FAILURE"),
                "QUANTIFIER_SCOPE": ("NEXT_DISCOURSE_UNIT", "ORDER_RECOVERY"),
                "UNKNOWN": ("CONTROLLED_PERTURBATION",),
            }.get(candidate_kind, ("INTEGER_DESCRIPTION_LENGTH",))
            rows.append({
                "baseline_kind": baseline,
                "candidate_kind": candidate_kind,
                "comparison_candidates": components["comparison_candidates"],
                "evaluation_dimension": dimension,
                "expected_payload": {
                    "accepted": int(expected_state == "TRUE"),
                    "accepted_surfaces": accepted_surfaces,
                    "analysis_key": f"{owner}_LC06_{dimension}",
                    "reason_code": "COURSE_LABEL_ONLY_NOT_RUNTIME_PASS",
                },
                "expected_state": expected_state,
                "family": f"{owner}_LC06_{family_suffix}",
                "generation_constraint": {
                    "comparison_preservation_required": generation_hidden,
                    "direction": (
                        "SEMANTICS_TO_SURFACE" if generation_hidden
                        else "SURFACE_TO_CANDIDATES"),
                    "input_candidate_ids": known_ids,
                    "output_surface_hidden": generation_hidden,
                    "quantity_preservation_required": generation_hidden,
                    "quantifier_scope_preservation_required": generation_hidden,
                    "scale_preservation_required": generation_hidden,
                    "standard_preservation_required": generation_hidden,
                    "unit_preservation_required": generation_hidden,
                },
                "label_owner": label_owner,
                "license_id": LICENSE_ID,
                "logical_order": logical_order,
                "object_candidates": components["object_candidates"],
                "object_family": object_family,
                "objective_keys": sorted(objective_keys),
                "observed_text": texts[index - 1],
                "perturbation_kind": f"LC06_{candidate_kind}",
                "quantifier_scopes": components["quantifier_scopes"],
                "quantity_candidates": components["quantity_candidates"],
                "resource_budget": {
                    "max_comparisons": 2,
                    "max_objects": 2,
                    "max_output_units": 80,
                    "max_quantifiers": 2,
                    "max_quantities": 2,
                    "max_scales": 1,
                    "max_standards": 2,
                    "max_units": 1,
                },
                "retention_anchor_id": retention_anchor,
                "revision_receipt": {
                    "candidate_version": revision_version,
                    "dependency_keys": ["QUANTITY"] if index == 11 else [],
                    "raw_observation_preserved": 1,
                    "revision_scope": "QUANTITY_ONLY" if index == 11 else "NONE",
                    "supersedes_seed_id": supersedes,
                },
                "sample_family": sample_family,
                "sample_role": {
                    "AMBIGUOUS": "conflict",
                    "GENERATION": "support",
                    "NEGATIVE": "refute",
                    "POSITIVE": "support",
                    "RETENTION": "read_only_probe",
                    "REVISION": "supersede",
                    "UNKNOWN": "anomaly",
                }[sample_family],
                "scale_definitions": components["scale_definitions"],
                "scale_family": scale_family,
                "seed_id": ids[index],
                "split": split,
                "standard_candidates": components["standard_candidates"],
                "supersedes_seed_id": supersedes,
                "surface_scope": {
                    "context_scope_key": scope,
                    "genre": "DECLARATIVE",
                    "language": "zh",
                    "register": "NEUTRAL",
                    "script": "HAN",
                },
                "template_family": f"{owner}_LC06_TEMPLATE_{candidate_kind}",
                "unit_definitions": components["unit_definitions"],
                "unit_family": unit_family,
            })
    return tuple(rows)


def default_comparison_quantity_sample_bytes() -> bytes:
    """返回规范 JSONL，供正式 sample 的可重复生成和审计。"""
    return b"".join(
        canonical_json_line(value)
        for value in build_default_comparison_quantity_seed_values())


__all__ = [
    "COURSE_MANIFEST_PATH",
    "EVALUATOR_DIMENSIONS",
    "FORMAL_ARTIFACT_RELATIVE_ROOT",
    "PACK_NAME",
    "PAYLOAD_KIND",
    "AuthoredComparisonQuantityCourseError",
    "AuthoredComparisonQuantitySeed",
    "ComparisonQuantityPayloadAudit",
    "build_comparison_quantity_course_manifest",
    "build_default_comparison_quantity_seed_values",
    "compile_authored_comparison_quantity_course",
    "default_comparison_quantity_sample_bytes",
    "read_authored_comparison_quantity_seeds",
    "validate_comparison_quantity_payload",
]
