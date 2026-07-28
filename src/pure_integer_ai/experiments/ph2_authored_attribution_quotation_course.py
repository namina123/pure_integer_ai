"""LC-14 转述、引语、holder 和认识 scope 的原创课程。"""
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
PACK_NAME = "AUTHORED_CC0_V1--CC0-1.0--lc14-attribution-quotation-v1"
STAGE = "W-08"
SUBSTAGE = "LC14_ATTRIBUTION_QUOTATION_PERSPECTIVE"
PAYLOAD_KIND = "AttributionQuotationCandidateV1"
COURSE_MANIFEST_ARTIFACT_VERSION = "LC-14-attribution-quotation-course-v1"
COURSE_MANIFEST_PATH = Path(
    "data/ph2/manifests/lc14_attribution_quotation_course_v1.json")
FORMAL_ARTIFACT_RELATIVE_ROOT = (
    "ph2_dataset_artifacts/d02_language_courses_v1")

CANDIDATE_KINDS = (
    "AMBIGUOUS_SCOPE",
    "BELIEF",
    "CLAIM",
    "DIRECT_QUOTATION",
    "GENERATION",
    "HYPOTHESIS",
    "LATER_DENIAL",
    "NESTED_HOLDER",
    "PARAPHRASE",
    "PRONOUN_TRANSFER",
    "QUOTE_BOUNDARY_BASELINE",
    "REPORTED_AS_FACT_BASELINE",
    "RETENTION",
    "REVISION",
    "SOURCE_CONFLICT",
    "TENSE_TRANSFER",
    "UNKNOWN",
)
ATTRIBUTION_KINDS = (
    "BELIEF", "CLAIM", "HYPOTHESIS", "PARAPHRASE", "QUOTATION", "UNKNOWN")
CANDIDATE_STATES = ("COMPETING", "REFUTED", "SUPPORTED", "UNKNOWN")
EPISTEMIC_STATES = (
    "ASSERTED", "CONFLICT", "DENIED", "HELD", "HYPOTHESIZED",
    "PARAPHRASED", "QUOTED", "UNKNOWN")
UNCERTAINTY_STATES = ("CONFLICT", "REPORTED", "UNCERTAIN", "UNKNOWN")
HOLDER_ROLES = ("ORGANIZATION", "PERSON", "SYSTEM", "UNKNOWN")
SOURCE_CHANNELS = ("DOCUMENT", "INFERENCE", "SPEECH", "UNKNOWN")
QUOTE_VERSION_KINDS = ("EXACT_QUOTE", "PARAPHRASE")
QUOTE_BOUNDARY_STATES = ("MATCH", "MISMATCH", "UNKNOWN")
TRANSFER_KINDS = ("NONE", "PRONOUN", "TENSE")
BASELINE_KINDS = (
    "QUOTE_BOUNDARY_SURFACE_ONLY",
    "REPORTED_AS_CURRENT_FACT",
    "TYPED_ATTRIBUTION_OBJECTS_PRESENT",
)
EVALUATOR_DIMENSIONS = (
    "AMBIGUOUS_SCOPE_COMPETITION",
    "ATTRIBUTION_LOCAL_REVISION",
    "BELIEF_HOLDER_SCOPE",
    "CLAIM_HOLDER_SCOPE",
    "DIRECT_QUOTE_BOUNDARY_FIDELITY",
    "HYPOTHESIS_UNCERTAINTY_SCOPE",
    "LATER_DENIAL_NO_CURRENT_PROJECTION",
    "NESTED_HOLDER_SCOPE",
    "PARAPHRASE_VERSION_LINK",
    "PRONOUN_TRANSFER_HOLDER_PRESERVATION",
    "QUOTE_BOUNDARY_SHORTCUT_REJECT",
    "REPORTED_AS_FACT_REJECT",
    "RETENTION_REVERIFY",
    "REVERSE_GENERATION_ATTRIBUTION_UNCERTAINTY",
    "SOURCE_CONFLICT_NO_CURRENT_PROJECTION",
    "TENSE_TRANSFER_ANCHOR_PRESERVATION",
    "UNKNOWN_SCOPE_NO_GUESS",
)
VERIFIER_NE_CONDITIONS = (
    "CAPABILITY_LEARNED_REQUESTED",
    "CURRENT_PROJECTION_TRUTH_REQUESTED",
    "GENERATION_RESULT_NOT_EXECUTED",
    "NO_EVALUATOR_LABEL",
    "OUT_OF_DOCUMENT_SCOPE",
    "RUNTIME_SCOPE_CONSUMER_NOT_EXECUTED",
)
RETENTION_PROTOCOLS = (
    "A_TO_LC14_REVERIFY_A",
    "DUMP_RESUME_REQUIRED_AT_RUNTIME",
    "LOCAL_ATTRIBUTION_SUPERSEDE_ONLY",
)
COMBINATION_AXES = (
    "ATTRIBUTION_FAMILY",
    "ATTRIBUTION_X_HOLDER_X_SOURCE",
    "HOLDER_ROLE",
    "SOURCE_CHANNEL",
)
ABLATION_KEYS = (
    "DROP_ATTRIBUTION_SCOPE",
    "DROP_HOLDER_IDENTITY",
    "DROP_QUOTATION_VERSION",
    "DROP_UNCERTAINTY",
    "FORCE_CURRENT_PROJECTION",
    "QUOTE_BOUNDARY_SURFACE_ONLY",
    "REPORTED_AS_CURRENT_FACT",
)
PAYLOAD_KEYS = (
    "attribution_candidates",
    "baseline_kind",
    "candidate_kind",
    "generation_constraint",
    "objective_keys",
    "observed_surface",
    "proposition_candidates",
    "quotation_spans",
    "resource_budget",
    "retention_anchor_id",
    "revision_receipt",
    "sample_family",
    "selection_state",
    "split_identity",
    "surface_scope",
    "transfer_receipt",
)

_SEED_FIELDS = {
    "attribution_candidates", "attribution_family", "baseline_kind",
    "candidate_kind", "evaluation_dimension", "expected_payload",
    "expected_state", "family", "generation_constraint", "holder_role",
    "label_owner", "license_id", "logical_order", "objective_keys",
    "observed_text", "perturbation_kind", "proposition_candidates",
    "quotation_spans", "resource_budget", "retention_anchor_id",
    "revision_receipt", "sample_family", "sample_role", "seed_id",
    "source_channel", "split", "supersedes_seed_id", "surface_scope",
    "template_family", "transfer_receipt",
}
_SCOPE_FIELDS = {
    "context_scope_key", "genre", "language", "register", "script"}
_PROPOSITION_FIELDS = {
    "context_scope_key", "current_projection_allowed", "proposition_id",
    "proposition_key"}
_ATTRIBUTION_FIELDS = {
    "attribution_id", "attribution_kind", "candidate_state",
    "candidate_version", "content_proposition_id",
    "current_projection_allowed", "epistemic_state", "holder_id",
    "holder_role", "nested_holder_id", "parent_attribution_id",
    "provisional", "source_channel", "source_ref_key", "uncertainty_state",
}
_QUOTE_FIELDS = {
    "attribution_id", "boundary_state", "candidate_version", "end",
    "exact_text", "paraphrase_of_quote_id", "quote_id", "start",
    "surface_sha256", "version_kind"}
_TRANSFER_FIELDS = {
    "holder_preserved", "source_form", "target_form",
    "temporal_anchor_preserved", "transfer_kind"}
_REVISION_FIELDS = {
    "affected_attribution_ids", "candidate_version",
    "raw_observation_preserved", "revision_scope", "supersedes_seed_id",
    "unaffected_quote_ids"}
_GENERATION_FIELDS = {
    "attribution_kind_preservation_required", "direction",
    "holder_preservation_required", "input_candidate_ids",
    "nesting_preservation_required", "output_surface_hidden",
    "quotation_fidelity_required", "scope_preservation_required",
    "uncertainty_preservation_required"}
_BUDGET_FIELDS = {
    "max_attributions", "max_dependency_edges", "max_nesting_depth",
    "max_output_units", "max_propositions", "max_quote_spans"}
_SPLIT_FIELDS = {
    "attribution_family", "combination_key", "holder_role",
    "isolation_axis", "source_channel"}
_EXPECTED_FIELDS = {
    "accepted", "accepted_surfaces", "analysis_key", "reason_code"}
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


class AuthoredAttributionQuotationCourseError(RuntimeError):
    """LC-14 seed、typed payload 或组合 split 不满足冻结合同。"""


def _exact_keys(value: Any, keys: set[str], *, where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise AuthoredAttributionQuotationCourseError(f"{where} 字段集合非法")
    return value


def _text(value: Any, *, where: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or value.strip() != value:
        raise AuthoredAttributionQuotationCourseError(f"{where} 必须是规范文本")
    if not allow_empty and not value:
        raise AuthoredAttributionQuotationCourseError(f"{where} 不能为空")
    return value


def _integer(value: Any, *, where: str, positive: bool = False) -> int:
    if type(value) is not int or value < int(positive):
        qualifier = "正" if positive else "非负"
        raise AuthoredAttributionQuotationCourseError(
            f"{where} 必须是{qualifier}严格整数")
    return value


def _flag(value: Any, *, where: str) -> int:
    if type(value) is not int or value not in {0, 1}:
        raise AuthoredAttributionQuotationCourseError(f"{where} 必须是 0/1")
    return value


def _strings(
        value: Any,
        *,
        where: str,
        allow_empty: bool = False,
        sorted_unique: bool = False,
        ) -> tuple[str, ...]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise AuthoredAttributionQuotationCourseError(f"{where} 必须是文本数组")
    result = tuple(_text(item, where=where) for item in value)
    if len(result) != len(set(result)):
        raise AuthoredAttributionQuotationCourseError(f"{where} 不得重复")
    if sorted_unique and result != tuple(sorted(result)):
        raise AuthoredAttributionQuotationCourseError(f"{where} 必须排序")
    return result


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_expected(payload: CanonicalJsonObject, expected_state: str) -> None:
    raw = _exact_keys(payload.to_value(), _EXPECTED_FIELDS, where="expected")
    accepted = _flag(raw["accepted"], where="expected accepted")
    surfaces = _strings(
        raw["accepted_surfaces"], where="accepted surfaces", allow_empty=True)
    _text(raw["analysis_key"], where="analysis_key")
    _text(raw["reason_code"], where="reason_code")
    if expected_state == "TRUE" and (accepted != 1 or not surfaces):
        raise AuthoredAttributionQuotationCourseError(
            "TRUE label 必须给出私有合法表层")
    if expected_state != "TRUE" and (accepted != 0 or surfaces):
        raise AuthoredAttributionQuotationCourseError(
            "非 TRUE label 不得给出采用表层")


@dataclass(frozen=True)
class AttributionQuotationPayloadAudit:
    """返回 typed 对象计数和组合隔离键。"""

    proposition_count: int
    attribution_count: int
    quotation_count: int
    dependency_edge_count: int
    combination_key: str


def validate_attribution_quotation_payload(
        payload: CanonicalJsonObject | dict[str, Any],
        ) -> AttributionQuotationPayloadAudit:
    """校验 holder、认识 scope、原话 span、转述版本和非当前投影。"""
    value = payload.to_value() if isinstance(payload, CanonicalJsonObject) else payload
    raw = _exact_keys(value, set(PAYLOAD_KEYS), where="attribution payload")
    kind = _text(raw["candidate_kind"], where="candidate_kind")
    if kind not in CANDIDATE_KINDS:
        raise AuthoredAttributionQuotationCourseError("candidate_kind 未登记")
    baseline = _text(raw["baseline_kind"], where="baseline_kind")
    if baseline not in BASELINE_KINDS:
        raise AuthoredAttributionQuotationCourseError("baseline_kind 未登记")
    if raw["sample_family"] not in SAMPLE_FAMILIES:
        raise AuthoredAttributionQuotationCourseError("sample_family 未登记")
    if raw["selection_state"] != "UNSELECTED":
        raise AuthoredAttributionQuotationCourseError(
            "Observation 不得预选 attribution 候选")
    objectives = _strings(
        raw["objective_keys"], where="objective_keys", sorted_unique=True)
    if any(item not in LANGUAGE_OBJECTIVE_KEYS for item in objectives):
        raise AuthoredAttributionQuotationCourseError("objective_keys 未登记")

    scope = _exact_keys(raw["surface_scope"], _SCOPE_FIELDS, where="scope")
    if scope["language"] != "zh" or scope["script"] != "HAN":
        raise AuthoredAttributionQuotationCourseError("LC-14 首阶段只冻结 zh/HAN")
    context_scope = _text(scope["context_scope_key"], where="context scope")
    _text(scope["genre"], where="genre")
    if scope["register"] not in {"COLLOQUIAL", "FORMAL", "NEUTRAL"}:
        raise AuthoredAttributionQuotationCourseError("register 未登记")

    observed = _exact_keys(raw["observed_surface"], {
        "append_only", "sha256", "target_hidden", "text",
    }, where="observed surface")
    observed_text = _text(observed["text"], where="observed text")
    if _flag(observed["append_only"], where="append_only") != 1:
        raise AuthoredAttributionQuotationCourseError(
            "raw Observation 必须 append-only")
    if observed["sha256"] != _sha256_text(observed_text):
        raise AuthoredAttributionQuotationCourseError(
            "observed surface SHA-256 漂移")
    target_hidden = _flag(observed["target_hidden"], where="target_hidden")

    budget = _exact_keys(raw["resource_budget"], _BUDGET_FIELDS, where="budget")
    ceilings = {
        key: _integer(budget[key], where=key, positive=True)
        for key in sorted(_BUDGET_FIELDS)
    }
    if len(observed_text) > ceilings["max_output_units"]:
        raise AuthoredAttributionQuotationCourseError("surface 超整数输出预算")

    proposition_values = raw["proposition_candidates"]
    if not isinstance(proposition_values, list) or not proposition_values:
        raise AuthoredAttributionQuotationCourseError("Proposition 不能为空")
    if len(proposition_values) > ceilings["max_propositions"]:
        raise AuthoredAttributionQuotationCourseError("Proposition 超预算")
    propositions: dict[str, dict[str, Any]] = {}
    for item in proposition_values:
        proposition = _exact_keys(item, _PROPOSITION_FIELDS, where="proposition")
        proposition_id = _text(
            proposition["proposition_id"], where="proposition_id")
        if proposition_id in propositions:
            raise AuthoredAttributionQuotationCourseError("proposition_id 重复")
        if proposition["context_scope_key"] != context_scope:
            raise AuthoredAttributionQuotationCourseError(
                "Proposition 跨 ContextScope")
        _text(proposition["proposition_key"], where="proposition_key")
        if _flag(
                proposition["current_projection_allowed"],
                where="proposition current projection") != 0:
            raise AuthoredAttributionQuotationCourseError(
                "报道 Proposition 不得进入 current projection")
        propositions[proposition_id] = proposition

    attribution_values = raw["attribution_candidates"]
    if not isinstance(attribution_values, list) or not attribution_values:
        raise AuthoredAttributionQuotationCourseError("attribution 不能为空")
    if len(attribution_values) > ceilings["max_attributions"]:
        raise AuthoredAttributionQuotationCourseError("attribution 超预算")
    attributions: dict[str, dict[str, Any]] = {}
    dependency_edges = 0
    for item in attribution_values:
        candidate = _exact_keys(item, _ATTRIBUTION_FIELDS, where="attribution")
        attribution_id = _text(
            candidate["attribution_id"], where="attribution_id")
        if attribution_id in attributions:
            raise AuthoredAttributionQuotationCourseError("attribution_id 重复")
        if candidate["attribution_kind"] not in ATTRIBUTION_KINDS:
            raise AuthoredAttributionQuotationCourseError("attribution kind 未登记")
        if candidate["candidate_state"] not in CANDIDATE_STATES:
            raise AuthoredAttributionQuotationCourseError("candidate state 未登记")
        if candidate["epistemic_state"] not in EPISTEMIC_STATES:
            raise AuthoredAttributionQuotationCourseError("epistemic state 未登记")
        if candidate["uncertainty_state"] not in UNCERTAINTY_STATES:
            raise AuthoredAttributionQuotationCourseError("uncertainty state 未登记")
        if candidate["holder_role"] not in HOLDER_ROLES:
            raise AuthoredAttributionQuotationCourseError("holder role 未登记")
        if candidate["source_channel"] not in SOURCE_CHANNELS:
            raise AuthoredAttributionQuotationCourseError("source channel 未登记")
        _text(candidate["holder_id"], where="holder_id")
        _text(candidate["source_ref_key"], where="source_ref_key")
        if candidate["content_proposition_id"] not in propositions:
            raise AuthoredAttributionQuotationCourseError(
                "attribution 引用未知 Proposition")
        if _flag(candidate["provisional"], where="provisional") != 1:
            raise AuthoredAttributionQuotationCourseError(
                "attribution 只能是 provisional candidate")
        if _flag(
                candidate["current_projection_allowed"],
                where="attribution current projection") != 0:
            raise AuthoredAttributionQuotationCourseError(
                "A 说 P 不得自动推出 current P")
        _integer(candidate["candidate_version"], where="candidate version", positive=True)
        parent = _text(
            candidate["parent_attribution_id"], where="parent attribution",
            allow_empty=True)
        nested_holder = _text(
            candidate["nested_holder_id"], where="nested holder",
            allow_empty=True)
        if parent:
            if parent not in attributions or not nested_holder:
                raise AuthoredAttributionQuotationCourseError(
                    "嵌套 holder 必须引用更早 attribution")
            dependency_edges += 1
        elif nested_holder:
            raise AuthoredAttributionQuotationCourseError(
                "非嵌套 attribution 不得声明 nested holder")
        if ((candidate["attribution_kind"] == "UNKNOWN")
                != (candidate["candidate_state"] == "UNKNOWN")):
            raise AuthoredAttributionQuotationCourseError(
                "unknown attribution 与候选状态不一致")
        attributions[attribution_id] = candidate

    quote_values = raw["quotation_spans"]
    if not isinstance(quote_values, list):
        raise AuthoredAttributionQuotationCourseError("quotation_spans 必须是数组")
    if len(quote_values) > ceilings["max_quote_spans"]:
        raise AuthoredAttributionQuotationCourseError("quotation span 超预算")
    quotes: dict[str, dict[str, Any]] = {}
    for item in quote_values:
        quote = _exact_keys(item, _QUOTE_FIELDS, where="quotation span")
        quote_id = _text(quote["quote_id"], where="quote_id")
        if quote_id in quotes:
            raise AuthoredAttributionQuotationCourseError("quote_id 重复")
        if quote["attribution_id"] not in attributions:
            raise AuthoredAttributionQuotationCourseError(
                "quotation 引用未知 attribution")
        if quote["version_kind"] not in QUOTE_VERSION_KINDS:
            raise AuthoredAttributionQuotationCourseError("quote version 未登记")
        if quote["boundary_state"] not in QUOTE_BOUNDARY_STATES:
            raise AuthoredAttributionQuotationCourseError("quote boundary 未登记")
        start = _integer(quote["start"], where="quote start")
        end = _integer(quote["end"], where="quote end", positive=True)
        exact = _text(quote["exact_text"], where="exact quote")
        if start >= end or end > len(observed_text) or observed_text[start:end] != exact:
            raise AuthoredAttributionQuotationCourseError("原话 span 与 raw 不一致")
        if quote["surface_sha256"] != _sha256_text(exact):
            raise AuthoredAttributionQuotationCourseError("原话 span SHA-256 漂移")
        _integer(quote["candidate_version"], where="quote version", positive=True)
        parent_quote = _text(
            quote["paraphrase_of_quote_id"], where="paraphrase source",
            allow_empty=True)
        if quote["version_kind"] == "PARAPHRASE":
            if parent_quote not in quotes:
                raise AuthoredAttributionQuotationCourseError(
                    "转述版本必须引用更早原话 span")
            dependency_edges += 1
        elif parent_quote:
            raise AuthoredAttributionQuotationCourseError(
                "原话 span 不得伪装转述版本")
        quotes[quote_id] = quote

    if dependency_edges > ceilings["max_dependency_edges"]:
        raise AuthoredAttributionQuotationCourseError("scope 依赖边超预算")

    transfer = _exact_keys(
        raw["transfer_receipt"], _TRANSFER_FIELDS, where="transfer receipt")
    transfer_kind = transfer["transfer_kind"]
    if transfer_kind not in TRANSFER_KINDS:
        raise AuthoredAttributionQuotationCourseError("transfer kind 未登记")
    source_form = _text(
        transfer["source_form"], where="source form", allow_empty=True)
    target_form = _text(
        transfer["target_form"], where="target form", allow_empty=True)
    holder_preserved = _flag(
        transfer["holder_preserved"], where="holder preserved")
    temporal_preserved = _flag(
        transfer["temporal_anchor_preserved"], where="temporal preserved")
    if transfer_kind == "NONE" and (
            source_form or target_form or holder_preserved or temporal_preserved):
        raise AuthoredAttributionQuotationCourseError("NONE transfer 不得伪造 receipt")
    if transfer_kind != "NONE" and (
            not source_form or not target_form or holder_preserved != 1
            or temporal_preserved != 1):
        raise AuthoredAttributionQuotationCourseError(
            "代词/时态转移必须保留 holder 和时间锚")

    revision = _exact_keys(
        raw["revision_receipt"], _REVISION_FIELDS, where="revision receipt")
    affected = _strings(
        revision["affected_attribution_ids"], where="affected attributions",
        allow_empty=True, sorted_unique=True)
    unaffected_quotes = _strings(
        revision["unaffected_quote_ids"], where="unaffected quotes",
        allow_empty=True, sorted_unique=True)
    supersedes = _text(
        revision["supersedes_seed_id"], where="revision supersedes",
        allow_empty=True)
    _integer(revision["candidate_version"], where="revision version", positive=True)
    if _flag(
            revision["raw_observation_preserved"],
            where="raw observation preserved") != 1:
        raise AuthoredAttributionQuotationCourseError(
            "revision 必须保留 append-only raw")
    if kind == "REVISION":
        if (revision["revision_scope"] != "ATTRIBUTION_STATE_ONLY"
                or not supersedes or not affected
                or any(item not in attributions for item in affected)
                or any(item not in quotes for item in unaffected_quotes)):
            raise AuthoredAttributionQuotationCourseError(
                "revision 必须局部绑定 attribution state")
    elif (revision["revision_scope"] != "NONE" or supersedes or affected
          or unaffected_quotes):
        raise AuthoredAttributionQuotationCourseError(
            "非 revision 不得声明修正 receipt")

    generation = _exact_keys(
        raw["generation_constraint"], _GENERATION_FIELDS,
        where="generation constraint")
    inputs = _strings(
        generation["input_candidate_ids"], where="generation inputs",
        allow_empty=True, sorted_unique=True)
    known_ids = set(attributions) | set(propositions) | set(quotes)
    if any(item not in known_ids for item in inputs):
        raise AuthoredAttributionQuotationCourseError(
            "generation 引用未知 typed candidate")
    flags = {
        key: _flag(generation[key], where=key)
        for key in _GENERATION_FIELDS
        if key.endswith("_required") or key == "output_surface_hidden"
    }
    if kind == "GENERATION":
        if (generation["direction"] != "ATTRIBUTION_TO_SURFACE"
                or set(flags.values()) != {1} or target_hidden != 1
                or not inputs):
            raise AuthoredAttributionQuotationCourseError(
                "生成必须隐藏表层并逐层保持 attribution/scope/uncertainty")
    elif (generation["direction"] != "SURFACE_TO_CANDIDATES"
          or any(flags.values()) or target_hidden != 0 or inputs):
        raise AuthoredAttributionQuotationCourseError(
            "非生成 Observation 不得携带生成答案")

    split = _exact_keys(raw["split_identity"], _SPLIT_FIELDS, where="split")
    if split["isolation_axis"] != "ATTRIBUTION_X_HOLDER_X_SOURCE":
        raise AuthoredAttributionQuotationCourseError("split isolation axis 非法")
    first = attribution_values[0]
    for key in ("holder_role", "source_channel"):
        if split[key] != first[key]:
            raise AuthoredAttributionQuotationCourseError(f"split {key} 伪造")
    if split["attribution_family"] != first["attribution_kind"]:
        raise AuthoredAttributionQuotationCourseError(
            "split attribution_family 伪造")
    combination = (
        f"{split['attribution_family']}::{split['holder_role']}::"
        f"{split['source_channel']}")
    if split["combination_key"] != combination:
        raise AuthoredAttributionQuotationCourseError("split combination_key 漂移")

    if kind == "DIRECT_QUOTATION" and not any(
            item["version_kind"] == "EXACT_QUOTE"
            and item["boundary_state"] == "MATCH" for item in quote_values):
        raise AuthoredAttributionQuotationCourseError("直接引语缺精确原话 span")
    if kind == "PARAPHRASE" and not any(
            item["version_kind"] == "PARAPHRASE" for item in quote_values):
        raise AuthoredAttributionQuotationCourseError("转述缺原话版本链")
    if kind == "NESTED_HOLDER" and not any(
            item["parent_attribution_id"] for item in attribution_values):
        raise AuthoredAttributionQuotationCourseError("嵌套 holder scope 缺失")
    if kind == "PRONOUN_TRANSFER" and transfer_kind != "PRONOUN":
        raise AuthoredAttributionQuotationCourseError("代词转移 receipt 缺失")
    if kind == "TENSE_TRANSFER" and transfer_kind != "TENSE":
        raise AuthoredAttributionQuotationCourseError("时态转移 receipt 缺失")
    if kind == "LATER_DENIAL" and not any(
            item["epistemic_state"] == "DENIED" for item in attribution_values):
        raise AuthoredAttributionQuotationCourseError("后文否认 scope 缺失")
    if kind == "SOURCE_CONFLICT" and (
            len({item["holder_id"] for item in attribution_values}) < 2
            or not all(item["epistemic_state"] == "CONFLICT"
                       for item in attribution_values)):
        raise AuthoredAttributionQuotationCourseError("来源冲突未分账")
    if kind == "AMBIGUOUS_SCOPE" and (
            len(attribution_values) < 2
            or not all(item["candidate_state"] == "COMPETING"
                       for item in attribution_values)):
        raise AuthoredAttributionQuotationCourseError("歧义 scope 未保留竞争")
    if kind == "QUOTE_BOUNDARY_BASELINE" and not any(
            item["boundary_state"] == "MISMATCH" for item in quote_values):
        raise AuthoredAttributionQuotationCourseError("坏引语边界基线缺失")
    if kind == "REPORTED_AS_FACT_BASELINE" and baseline != (
            "REPORTED_AS_CURRENT_FACT"):
        raise AuthoredAttributionQuotationCourseError("报道即事实负基线身份错配")
    if kind == "UNKNOWN" and not all(
            item["candidate_state"] == "UNKNOWN" for item in attribution_values):
        raise AuthoredAttributionQuotationCourseError("unknown scope 不得猜测")
    return AttributionQuotationPayloadAudit(
        len(propositions), len(attributions), len(quotes), dependency_edges,
        combination)


@dataclass(frozen=True)
class AuthoredAttributionQuotationSeed:
    """一条规范 LC-14 seed；expected 只进入 owner 私有记录。"""

    value: CanonicalJsonObject

    def __post_init__(self) -> None:
        raw = _exact_keys(self.value.to_value(), _SEED_FIELDS, where="LC-14 seed")
        for key in (
                "seed_id", "family", "template_family", "label_owner", "split",
                "sample_family", "sample_role", "candidate_kind",
                "observed_text", "attribution_family", "holder_role",
                "source_channel", "baseline_kind", "expected_state",
                "evaluation_dimension", "perturbation_kind", "license_id"):
            _text(raw[key], where=key)
        _text(raw["supersedes_seed_id"], where="supersedes", allow_empty=True)
        _text(raw["retention_anchor_id"], where="retention anchor", allow_empty=True)
        _integer(raw["logical_order"], where="logical_order", positive=True)
        if raw["label_owner"] not in {"teacher", "evaluator"}:
            raise AuthoredAttributionQuotationCourseError("label_owner 非法")
        expected_split = "train" if raw["label_owner"] == "teacher" else "held_out"
        if raw["split"] != expected_split:
            raise AuthoredAttributionQuotationCourseError("owner/split 不一致")
        if raw["sample_family"] not in SAMPLE_FAMILIES:
            raise AuthoredAttributionQuotationCourseError("sample family 未登记")
        if raw["candidate_kind"] not in CANDIDATE_KINDS:
            raise AuthoredAttributionQuotationCourseError("candidate kind 未登记")
        if raw["baseline_kind"] not in BASELINE_KINDS:
            raise AuthoredAttributionQuotationCourseError("baseline kind 未登记")
        if raw["evaluation_dimension"] not in EVALUATOR_DIMENSIONS:
            raise AuthoredAttributionQuotationCourseError("evaluator 维度未登记")
        if raw["expected_state"] not in EXPECTED_STATES:
            raise AuthoredAttributionQuotationCourseError("expected_state 非四态")
        if raw["license_id"] != LICENSE_ID:
            raise AuthoredAttributionQuotationCourseError("LC-14 必须是 CC0-1.0")
        expected = raw["expected_payload"]
        if not isinstance(expected, dict):
            raise AuthoredAttributionQuotationCourseError("expected_payload 非对象")
        _validate_expected(
            CanonicalJsonObject.from_value(expected), raw["expected_state"])
        validate_attribution_quotation_payload(self.observation_payload())

    def _raw(self) -> dict[str, Any]:
        return self.value.to_value()

    def __getattr__(self, name: str) -> Any:
        raw = self._raw()
        if name in raw:
            value = raw[name]
            if name == "objective_keys":
                return tuple(value)
            if name == "expected_payload":
                return CanonicalJsonObject.from_value(value)
            return value
        raise AttributeError(name)

    def observation_payload(self) -> CanonicalJsonObject:
        raw = self._raw()
        generation = raw["generation_constraint"]
        return CanonicalJsonObject.from_value({
            "attribution_candidates": raw["attribution_candidates"],
            "baseline_kind": raw["baseline_kind"],
            "candidate_kind": raw["candidate_kind"],
            "generation_constraint": generation,
            "objective_keys": raw["objective_keys"],
            "observed_surface": {
                "append_only": 1,
                "sha256": _sha256_text(raw["observed_text"]),
                "target_hidden": generation["output_surface_hidden"],
                "text": raw["observed_text"],
            },
            "proposition_candidates": raw["proposition_candidates"],
            "quotation_spans": raw["quotation_spans"],
            "resource_budget": raw["resource_budget"],
            "retention_anchor_id": raw["retention_anchor_id"],
            "revision_receipt": raw["revision_receipt"],
            "sample_family": raw["sample_family"],
            "selection_state": "UNSELECTED",
            "split_identity": {
                "attribution_family": raw["attribution_family"],
                "combination_key": (
                    f"{raw['attribution_family']}::{raw['holder_role']}::"
                    f"{raw['source_channel']}"),
                "holder_role": raw["holder_role"],
                "isolation_axis": "ATTRIBUTION_X_HOLDER_X_SOURCE",
                "source_channel": raw["source_channel"],
            },
            "surface_scope": raw["surface_scope"],
            "transfer_receipt": raw["transfer_receipt"],
        })

    def compiled_seed(self) -> AuthoredCompiledSeed:
        payload = self.observation_payload()
        audit = validate_attribution_quotation_payload(payload)
        raw = self._raw()
        return AuthoredCompiledSeed(
            raw["seed_id"], raw["family"], raw["template_family"],
            raw["label_owner"], raw["split"], raw["sample_role"], PAYLOAD_KIND,
            payload, raw["expected_state"],
            CanonicalJsonObject.from_value(raw["expected_payload"]),
            raw["perturbation_kind"], raw["supersedes_seed_id"],
            raw["logical_order"],
            (raw["seed_id"], audit.combination_key),
            (raw["observed_text"], raw["family"]),
            (raw["candidate_kind"], raw["attribution_family"],
             raw["holder_role"], raw["source_channel"],
             audit.proposition_count, audit.attribution_count,
             audit.quotation_count, audit.dependency_edge_count),
            raw["evaluation_dimension"],
        )


def read_authored_attribution_quotation_seeds(
        path: str | Path,
        ) -> tuple[AuthoredAttributionQuotationSeed, ...]:
    """严格读取 JSONL，并核 owner、修正、retention 和组合 held-out。"""
    try:
        payload = Path(path).read_bytes()
    except OSError as error:
        raise AuthoredAttributionQuotationCourseError("LC-14 sample 无法读取") from error
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise AuthoredAttributionQuotationCourseError("LC-14 sample 换行非法")
    seeds: list[AuthoredAttributionQuotationSeed] = []
    try:
        for line in payload.splitlines(keepends=True):
            value = parse_canonical_json_bytes(line[:-1], require_object=True)
            assert isinstance(value, dict)
            if canonical_json_line(value) != line:
                raise AuthoredAttributionQuotationCourseError(
                    "LC-14 seed 非规范 JSON")
            seeds.append(AuthoredAttributionQuotationSeed(
                CanonicalJsonObject.from_value(value)))
    except AuthoredAttributionQuotationCourseError:
        raise
    except Exception as error:
        raise AuthoredAttributionQuotationCourseError("LC-14 seed 损坏") from error
    identifiers = [item.seed_id for item in seeds]
    orders = [item.logical_order for item in seeds]
    if not seeds or len(identifiers) != len(set(identifiers)):
        raise AuthoredAttributionQuotationCourseError("seed_id 空或重复")
    if orders != sorted(orders) or len(orders) != len(set(orders)):
        raise AuthoredAttributionQuotationCourseError("logical_order 必须严格递增")
    index = {item.seed_id: item for item in seeds}
    for seed in seeds:
        expected_role = _ROLE_BY_FAMILY.get(seed.sample_family)
        if expected_role is not None and seed.sample_role != expected_role:
            raise AuthoredAttributionQuotationCourseError("sample family/role 不一致")
        if seed.sample_family == "GENERATION" and seed.sample_role not in {
                "support", "refute", "read_only_probe"}:
            raise AuthoredAttributionQuotationCourseError("generation role 非法")
        expected_state = _STATE_BY_FAMILY.get(seed.sample_family)
        if expected_state is not None and seed.expected_state != expected_state:
            raise AuthoredAttributionQuotationCourseError("sample family/state 不一致")
        if seed.sample_family == "GENERATION" and seed.expected_state not in {
                "TRUE", "FALSE"}:
            raise AuthoredAttributionQuotationCourseError("generation state 非法")
        if seed.candidate_kind == "REVISION":
            target = index.get(seed.supersedes_seed_id)
            if (target is None or target.logical_order >= seed.logical_order
                    or target.family != seed.family or target.split != seed.split):
                raise AuthoredAttributionQuotationCourseError(
                    "revision 必须 supersede 同族更早 seed")
            old = target.observation_payload().to_value()
            new = seed.observation_payload().to_value()
            for key in (
                    "proposition_candidates", "quotation_spans",
                    "transfer_receipt"):
                if old[key] != new[key]:
                    raise AuthoredAttributionQuotationCourseError(
                        "attribution revision 不得替换 Proposition/原话/transfer")
            old_items = old["attribution_candidates"]
            new_items = new["attribution_candidates"]
            if len(old_items) != len(new_items):
                raise AuthoredAttributionQuotationCourseError(
                    "attribution revision 不得增删身份")
            stable = set(_ATTRIBUTION_FIELDS) - {
                "candidate_state", "candidate_version", "epistemic_state",
                "uncertainty_state"}
            for before, after in zip(old_items, new_items, strict=True):
                if any(before[key] != after[key] for key in stable):
                    raise AuthoredAttributionQuotationCourseError(
                        "attribution revision 不得替换 holder/content/scope")
                if after["candidate_version"] <= before["candidate_version"]:
                    raise AuthoredAttributionQuotationCourseError(
                        "attribution revision version 必须前进")
        elif seed.supersedes_seed_id:
            raise AuthoredAttributionQuotationCourseError(
                "非 revision 不得 supersede")
        if seed.candidate_kind == "RETENTION":
            anchor = index.get(seed.retention_anchor_id)
            if (anchor is None or anchor.logical_order >= seed.logical_order
                    or anchor.split != seed.split):
                raise AuthoredAttributionQuotationCourseError(
                    "retention anchor 必须是同 split 更早 seed")
        elif seed.retention_anchor_id:
            raise AuthoredAttributionQuotationCourseError(
                "非 retention 不得绑定 anchor")

    teacher = tuple(item for item in seeds if item.label_owner == "teacher")
    evaluator = tuple(item for item in seeds if item.label_owner == "evaluator")
    for owner, owner_seeds in (("teacher", teacher), ("evaluator", evaluator)):
        if {item.sample_family for item in owner_seeds} != set(SAMPLE_FAMILIES):
            raise AuthoredAttributionQuotationCourseError(
                f"{owner} 未覆盖七类 sample family")
        if {item.candidate_kind for item in owner_seeds} != set(CANDIDATE_KINDS):
            raise AuthoredAttributionQuotationCourseError(
                f"{owner} 未覆盖完整 LC-14 候选族")
        negative = tuple(
            item for item in owner_seeds
            if item.baseline_kind != "TYPED_ATTRIBUTION_OBJECTS_PRESENT")
        if (len(negative) != 2
                or {item.baseline_kind for item in negative} != {
                    "QUOTE_BOUNDARY_SURFACE_ONLY", "REPORTED_AS_CURRENT_FACT"}
                or any(item.expected_state != "FALSE" for item in negative)):
            raise AuthoredAttributionQuotationCourseError(
                f"{owner} 两类 attribution 负基线未冻结")
    if ({item.family for item in teacher} & {item.family for item in evaluator}
            or {item.template_family for item in teacher}
            & {item.template_family for item in evaluator}):
        raise AuthoredAttributionQuotationCourseError(
            "teacher/evaluator family/template 泄漏")
    teacher_axes = {
        item.attribution_family for item in teacher}, {
        item.holder_role for item in teacher}, {
        item.source_channel for item in teacher}
    teacher_combinations = {
        (item.attribution_family, item.holder_role, item.source_channel)
        for item in teacher}
    held_out = {
        (item.attribution_family, item.holder_role, item.source_channel)
        for item in evaluator
        if item.attribution_family in teacher_axes[0]
        and item.holder_role in teacher_axes[1]
        and item.source_channel in teacher_axes[2]
        and (item.attribution_family, item.holder_role, item.source_channel)
        not in teacher_combinations
    }
    if len(held_out) < 4:
        raise AuthoredAttributionQuotationCourseError(
            "held-out attribution×holder×source 组合不足")
    if {item.evaluation_dimension for item in evaluator} != set(
            EVALUATOR_DIMENSIONS):
        raise AuthoredAttributionQuotationCourseError(
            "独立 evaluator 维度未列全")
    return tuple(seeds)


def _course_spec() -> AuthoredCourseSpec:
    return AuthoredCourseSpec(
        SOURCE_KEY, LICENSE_ID, COURSE_VERSION, ARTIFACT_VERSION,
        ADAPTER_VERSION, GENERATOR_VERSION, PARSER_VERSION, PACK_NAME,
        STAGE, SUBSTAGE, "authored-attribution-quotation-seed-v1",
        "urn:pure-integer-ai:ph2:authored-attribution-quotation-v1",
        "Pure Integer AI PH2 authored attribution quotation seed",
        "ATTRIBUTION_QUOTATION_LABEL", "lc14-attribution-quotation", 512)


def compile_authored_attribution_quotation_course(
        sample_path: str | Path,
        release_root: str | Path,
        ) -> AuthoredCourseBuild:
    seeds = read_authored_attribution_quotation_seeds(sample_path)
    return publish_authored_course(
        tuple(item.compiled_seed() for item in seeds), sample_path,
        release_root, _course_spec())


def build_attribution_quotation_course_manifest(
        sample_path: str | Path,
        build: AuthoredCourseBuild,
        *,
        artifact_relative_root: str = FORMAL_ARTIFACT_RELATIVE_ROOT,
        ) -> CapabilityCourseManifest:
    sample = Path(sample_path)
    seeds = read_authored_attribution_quotation_seeds(sample)
    teacher = tuple(item for item in seeds if item.label_owner == "teacher")
    evaluator = tuple(item for item in seeds if item.label_owner == "evaluator")
    return CapabilityCourseManifest(
        1, COURSE_MANIFEST_ARTIFACT_VERSION, "COURSE_FROZEN", "NOT_STARTED",
        ("LC-14",),
        ("ATTRIBUTION_QUOTATION_PERSPECTIVE", "SOURCE_UNCERTAINTY_REALITY"),
        STAGE, SUBSTAGE, SOURCE_KEY, LICENSE_ID, f"data/ph2/{sample.name}",
        hashlib.sha256(sample.read_bytes()).hexdigest(), len(seeds),
        SAMPLE_FAMILIES, COURSE_SPLIT_AXES,
        tuple(sorted({item.family for item in teacher})),
        tuple(sorted({item.family for item in evaluator})),
        tuple(sorted({item.template_family for item in teacher})),
        tuple(sorted({item.template_family for item in evaluator})),
        PAYLOAD_KIND, PAYLOAD_KEYS,
        tuple(sorted({key for item in seeds for key in item.objective_keys})),
        EVALUATOR_DIMENSIONS, VERIFIER_NE_CONDITIONS, RETENTION_PROTOCOLS,
        COMBINATION_AXES, BASELINE_KINDS, ABLATION_KEYS,
        f"{artifact_relative_root}/packs/{PACK_NAME}/manifest.json",
        build.manifest.sha256(), build.manifest.record_count,
        build.manifest.splits,
        CanonicalJsonObject.from_value(COURSE_INVARIANTS),
        CanonicalJsonObject.from_value(COURSE_EXECUTION_STATE))


def _attribution(
        token: str,
        proposition_id: str,
        attribution_kind: str,
        holder_role: str,
        source_channel: str,
        *,
        ordinal: int = 1,
        candidate_state: str = "SUPPORTED",
        epistemic_state: str = "ASSERTED",
        uncertainty_state: str = "REPORTED",
        version: int = 1,
        parent_id: str = "",
        nested_holder_id: str = "",
        ) -> dict[str, Any]:
    return {
        "attribution_id": f"{token}_ATTRIBUTION_{ordinal}",
        "attribution_kind": attribution_kind,
        "candidate_state": candidate_state,
        "candidate_version": version,
        "content_proposition_id": proposition_id,
        "current_projection_allowed": 0,
        "epistemic_state": epistemic_state,
        "holder_id": f"{holder_role}_{token}_{ordinal}",
        "holder_role": holder_role,
        "nested_holder_id": nested_holder_id,
        "parent_attribution_id": parent_id,
        "provisional": 1,
        "source_channel": source_channel,
        "source_ref_key": f"{source_channel}_{token}_{ordinal}",
        "uncertainty_state": uncertainty_state,
    }


def _quote(
        token: str,
        ordinal: int,
        attribution_id: str,
        observed_text: str,
        exact_text: str,
        *,
        version_kind: str = "EXACT_QUOTE",
        boundary_state: str = "MATCH",
        paraphrase_of: str = "",
        ) -> dict[str, Any]:
    start = observed_text.index(exact_text)
    return {
        "attribution_id": attribution_id,
        "boundary_state": boundary_state,
        "candidate_version": 1,
        "end": start + len(exact_text),
        "exact_text": exact_text,
        "paraphrase_of_quote_id": paraphrase_of,
        "quote_id": f"{token}_QUOTE_{ordinal}",
        "start": start,
        "surface_sha256": _sha256_text(exact_text),
        "version_kind": version_kind,
    }


_SCENARIOS = (
    ("CLAIM", "POSITIVE", "CLAIM", "PERSON", "DOCUMENT", "CLAIM_HOLDER_SCOPE"),
    ("BELIEF", "POSITIVE", "BELIEF", "ORGANIZATION", "SPEECH", "BELIEF_HOLDER_SCOPE"),
    ("HYPOTHESIS", "POSITIVE", "HYPOTHESIS", "PERSON", "INFERENCE", "HYPOTHESIS_UNCERTAINTY_SCOPE"),
    ("DIRECT_QUOTATION", "POSITIVE", "QUOTATION", "PERSON", "SPEECH", "DIRECT_QUOTE_BOUNDARY_FIDELITY"),
    ("PARAPHRASE", "POSITIVE", "QUOTATION", "ORGANIZATION", "DOCUMENT", "PARAPHRASE_VERSION_LINK"),
    ("NESTED_HOLDER", "POSITIVE", "CLAIM", "SYSTEM", "DOCUMENT", "NESTED_HOLDER_SCOPE"),
    ("PRONOUN_TRANSFER", "POSITIVE", "CLAIM", "PERSON", "SPEECH", "PRONOUN_TRANSFER_HOLDER_PRESERVATION"),
    ("TENSE_TRANSFER", "POSITIVE", "BELIEF", "SYSTEM", "INFERENCE", "TENSE_TRANSFER_ANCHOR_PRESERVATION"),
    ("LATER_DENIAL", "POSITIVE", "CLAIM", "ORGANIZATION", "DOCUMENT", "LATER_DENIAL_NO_CURRENT_PROJECTION"),
    ("SOURCE_CONFLICT", "POSITIVE", "CLAIM", "PERSON", "SPEECH", "SOURCE_CONFLICT_NO_CURRENT_PROJECTION"),
    ("REPORTED_AS_FACT_BASELINE", "NEGATIVE", "CLAIM", "SYSTEM", "DOCUMENT", "REPORTED_AS_FACT_REJECT"),
    ("QUOTE_BOUNDARY_BASELINE", "NEGATIVE", "QUOTATION", "ORGANIZATION", "SPEECH", "QUOTE_BOUNDARY_SHORTCUT_REJECT"),
    ("AMBIGUOUS_SCOPE", "AMBIGUOUS", "CLAIM", "PERSON", "INFERENCE", "AMBIGUOUS_SCOPE_COMPETITION"),
    ("UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN_SCOPE_NO_GUESS"),
    ("REVISION", "REVISION", "CLAIM", "PERSON", "DOCUMENT", "ATTRIBUTION_LOCAL_REVISION"),
    ("RETENTION", "RETENTION", "BELIEF", "ORGANIZATION", "SPEECH", "RETENTION_REVERIFY"),
    ("GENERATION", "GENERATION", "HYPOTHESIS", "SYSTEM", "INFERENCE", "REVERSE_GENERATION_ATTRIBUTION_UNCERTAINTY"),
)

_EVALUATOR_AXIS_OVERRIDES = {
    1: ("ORGANIZATION", "SPEECH"),
    2: ("SYSTEM", "DOCUMENT"),
    3: ("PERSON", "SPEECH"),
    4: ("SYSTEM", "SPEECH"),
    5: ("PERSON", "DOCUMENT"),
    15: ("ORGANIZATION", "SPEECH"),
}

_TEACHER_TEXTS = (
    "纪要记录：林岚声称闸门已经关闭。",
    "甲组认为备用节点仍然在线。",
    "研究员假设缓存损坏导致读取停止。",
    "林岚原话是“闸门已经关闭”。",
    "报告原话“闸门已经关闭”，纪要转述为“闸门已经合上”。",
    "调度台报告：林岚认为值班员说闸门已经关闭。",
    "林岚说她已经复核编号，其中“她”仍指林岚。",
    "系统先前认为节点正在运行，转述时保持原时间锚。",
    "甲组先声称节点在线，后文又否认这一说法。",
    "甲组称节点在线，乙组称节点离线，两项报道保持冲突。",
    "有人声称闸门关闭；课程必须拒绝把报道直接写成当前事实。",
    "记录写着“闸门已经关闭”，只凭引号猜短边界必须驳回。",
    "现有句法既可归于林岚的声称，也可归于甲组的假设。",
    "说话者、来源和所持内容均未知。",
    "后文否认旧声称；只更新原 attribution 状态，不替换 holder。",
    "只读重验：甲组仍只是认为备用节点在线。",
    "语义输入给出系统持有的假设、来源和不确定性，目标表层隐藏。",
)

_EVALUATOR_TEXTS = (
    "公告记录：乙组声称备份已经完成。",
    "监控系统认为备用链路仍然可用。",
    "分析员假设温度升高造成传感器漂移。",
    "乙组原话是“备份已经完成”。",
    "日志原话“备份已经完成”，简报转述为“备份已经做完”。",
    "控制台报告：乙组认为巡检员说备份已经完成。",
    "乙组说他们已经复核清单，其中“他们”仍指乙组。",
    "监控先前认为链路正在工作，转述时保持原时间锚。",
    "乙组先声称链路可用，后文又否认这一说法。",
    "乙组称链路可用，丙组称链路中断，两项报道保持冲突。",
    "有人声称备份完成；课程必须拒绝把报道直接写成当前事实。",
    "日志写着“备份已经完成”，只凭引号猜短边界必须驳回。",
    "现有句法既可归于乙组的声称，也可归于丙组的假设。",
    "持有者、渠道和认识状态均未知。",
    "后文否认旧声称；只更新原 attribution 状态，不替换 holder。",
    "只读重验：监控系统仍只是认为备用链路可用。",
    "语义输入给出系统持有的假设、来源和不确定性，目标表层隐藏。",
)


def _scenario_components(
        owner: str,
        index: int,
        candidate_kind: str,
        attribution_family: str,
        holder_role: str,
        source_channel: str,
        observed_text: str,
        ) -> dict[str, Any]:
    shared = index in {1, 15}
    token = f"{owner}_REVISION" if shared else f"{owner}_{index:02d}"
    scope = f"{token}_CTX"
    proposition_id = f"{token}_PROPOSITION_1"
    propositions = [{
        "context_scope_key": scope,
        "current_projection_allowed": 0,
        "proposition_id": proposition_id,
        "proposition_key": f"{token}_REPORTED_CONTENT",
    }]
    epistemic = {
        "BELIEF": "HELD",
        "HYPOTHESIS": "HYPOTHESIZED",
        "PARAPHRASE": "PARAPHRASED",
        "QUOTATION": "QUOTED",
        "UNKNOWN": "UNKNOWN",
    }.get(attribution_family, "ASSERTED")
    uncertainty = {
        "BELIEF": "UNCERTAIN",
        "HYPOTHESIS": "UNCERTAIN",
        "PARAPHRASE": "UNCERTAIN",
        "UNKNOWN": "UNKNOWN",
    }.get(attribution_family, "REPORTED")
    state = "UNKNOWN" if attribution_family == "UNKNOWN" else "SUPPORTED"
    version = 2 if candidate_kind == "REVISION" else 1
    if candidate_kind == "REVISION":
        state = "REFUTED"
        epistemic = "DENIED"
    first = _attribution(
        token, proposition_id, attribution_family, holder_role, source_channel,
        candidate_state=state, epistemic_state=epistemic,
        uncertainty_state=uncertainty, version=version)
    attributions = [first]
    quotes: list[dict[str, Any]] = []
    if candidate_kind == "DIRECT_QUOTATION":
        exact = "闸门已经关闭" if owner == "T" else "备份已经完成"
        quotes.append(_quote(token, 1, first["attribution_id"], observed_text, exact))
    elif candidate_kind == "PARAPHRASE":
        exact = "闸门已经关闭" if owner == "T" else "备份已经完成"
        paraphrase = "闸门已经合上" if owner == "T" else "备份已经做完"
        first_quote = _quote(
            token, 1, first["attribution_id"], observed_text, exact)
        second_attr = _attribution(
            token, proposition_id, "PARAPHRASE", holder_role, source_channel,
            ordinal=2, epistemic_state="PARAPHRASED",
            uncertainty_state="UNCERTAIN")
        attributions.append(second_attr)
        quotes.extend((first_quote, _quote(
            token, 2, second_attr["attribution_id"], observed_text, paraphrase,
            version_kind="PARAPHRASE", paraphrase_of=first_quote["quote_id"])))
    elif candidate_kind == "NESTED_HOLDER":
        child = _attribution(
            token, proposition_id, "BELIEF", "PERSON", "SPEECH", ordinal=2,
            epistemic_state="HELD", uncertainty_state="UNCERTAIN",
            parent_id=first["attribution_id"],
            nested_holder_id=f"PERSON_{token}_NESTED")
        attributions.append(child)
    elif candidate_kind == "LATER_DENIAL":
        attributions.append(_attribution(
            token, proposition_id, attribution_family, holder_role,
            source_channel, ordinal=2, candidate_state="REFUTED",
            epistemic_state="DENIED", uncertainty_state="REPORTED"))
    elif candidate_kind == "SOURCE_CONFLICT":
        first["epistemic_state"] = "CONFLICT"
        first["uncertainty_state"] = "CONFLICT"
        attributions.append(_attribution(
            token, proposition_id, attribution_family, "ORGANIZATION",
            "DOCUMENT", ordinal=2, epistemic_state="CONFLICT",
            uncertainty_state="CONFLICT"))
    elif candidate_kind == "AMBIGUOUS_SCOPE":
        first["candidate_state"] = "COMPETING"
        attributions.append(_attribution(
            token, proposition_id, "HYPOTHESIS", "ORGANIZATION", "INFERENCE",
            ordinal=2, candidate_state="COMPETING",
            epistemic_state="HYPOTHESIZED", uncertainty_state="UNCERTAIN"))
    elif candidate_kind == "QUOTE_BOUNDARY_BASELINE":
        first["candidate_state"] = "REFUTED"
        first["attribution_kind"] = "QUOTATION"
        first["epistemic_state"] = "QUOTED"
        exact = "闸门已经关闭" if owner == "T" else "备份已经完成"
        quotes.append(_quote(
            token, 1, first["attribution_id"], observed_text, exact,
            boundary_state="MISMATCH"))
    transfer_kind = "NONE"
    source_form = ""
    target_form = ""
    if candidate_kind == "PRONOUN_TRANSFER":
        transfer_kind, source_form, target_form = (
            "PRONOUN", "林岚" if owner == "T" else "乙组",
            "她" if owner == "T" else "他们")
    elif candidate_kind == "TENSE_TRANSFER":
        transfer_kind, source_form, target_form = ("TENSE", "正在运行", "先前正在运行")
    return {
        "attributions": attributions,
        "propositions": propositions,
        "quotes": quotes,
        "scope": scope,
        "transfer": {
            "holder_preserved": int(transfer_kind != "NONE"),
            "source_form": source_form,
            "target_form": target_form,
            "temporal_anchor_preserved": int(transfer_kind != "NONE"),
            "transfer_kind": transfer_kind,
        },
    }


def build_default_attribution_quotation_seed_values(
        ) -> tuple[dict[str, Any], ...]:
    """构造双 owner LC-14 原创课程，不执行训练或 runtime 判定。"""
    rows: list[dict[str, Any]] = []
    logical_order = 0
    for owner, texts in (("T", _TEACHER_TEXTS), ("E", _EVALUATOR_TEXTS)):
        label_owner = "teacher" if owner == "T" else "evaluator"
        split = "train" if owner == "T" else "held_out"
        ids = {
            index: f"{owner}-lc14-{index:02d}-{scenario[0].lower().replace('_', '-')}"
            for index, scenario in enumerate(_SCENARIOS, start=1)
        }
        for index, scenario in enumerate(_SCENARIOS, start=1):
            logical_order += 1
            (candidate_kind, sample_family, attribution_family, holder_role,
             source_channel, dimension) = scenario
            if owner == "E" and index in _EVALUATOR_AXIS_OVERRIDES:
                holder_role, source_channel = _EVALUATOR_AXIS_OVERRIDES[index]
            components = _scenario_components(
                owner, index, candidate_kind, attribution_family, holder_role,
                source_channel, texts[index - 1])
            actual_family = components["attributions"][0]["attribution_kind"]
            actual_holder = components["attributions"][0]["holder_role"]
            actual_channel = components["attributions"][0]["source_channel"]
            supersedes = ids[1] if index == 15 else ""
            retention_anchor = ids[2] if index == 16 else ""
            expected_state = {
                "AMBIGUOUS": "CONFLICT",
                "NEGATIVE": "FALSE",
                "UNKNOWN": "UNKNOWN",
            }.get(sample_family, "TRUE")
            accepted_surfaces: list[str] = []
            if expected_state == "TRUE":
                if candidate_kind == "GENERATION":
                    accepted_surfaces = [
                        "据监控系统推测，备用链路可能仍在工作。"
                        if owner == "T" else
                        "据控制系统推测，备份任务可能已经完成。"]
                else:
                    accepted_surfaces = [texts[index - 1]]
            hidden = int(candidate_kind == "GENERATION")
            candidate_ids = sorted([
                *[item["attribution_id"] for item in components["attributions"]],
                *[item["proposition_id"] for item in components["propositions"]],
                *[item["quote_id"] for item in components["quotes"]],
            ])
            revision_version = 2 if candidate_kind == "REVISION" else 1
            baseline = {
                "QUOTE_BOUNDARY_BASELINE": "QUOTE_BOUNDARY_SURFACE_ONLY",
                "REPORTED_AS_FACT_BASELINE": "REPORTED_AS_CURRENT_FACT",
            }.get(candidate_kind, "TYPED_ATTRIBUTION_OBJECTS_PRESENT")
            objective_keys = {
                "AMBIGUOUS_SCOPE": ("CROSS_CONTEXT_CONSISTENCY",),
                "GENERATION": ("GENERATION_ADOPTION", "GENERATION_FAILURE"),
                "REVISION": ("CONTROLLED_PERTURBATION",),
                "UNKNOWN": ("CONTROLLED_PERTURBATION",),
            }.get(candidate_kind, ("NEXT_DISCOURSE_UNIT",))
            family_suffix = (
                "REVISION_CHAIN" if index in {1, 15} else candidate_kind)
            rows.append({
                "attribution_candidates": components["attributions"],
                "attribution_family": actual_family,
                "baseline_kind": baseline,
                "candidate_kind": candidate_kind,
                "evaluation_dimension": dimension,
                "expected_payload": {
                    "accepted": int(expected_state == "TRUE"),
                    "accepted_surfaces": accepted_surfaces,
                    "analysis_key": f"{owner}_LC14_{dimension}",
                    "reason_code": "COURSE_LABEL_ONLY_NOT_RUNTIME_PASS",
                },
                "expected_state": expected_state,
                "family": f"{owner}_LC14_{family_suffix}",
                "generation_constraint": {
                    "attribution_kind_preservation_required": hidden,
                    "direction": (
                        "ATTRIBUTION_TO_SURFACE" if hidden
                        else "SURFACE_TO_CANDIDATES"),
                    "holder_preservation_required": hidden,
                    "input_candidate_ids": candidate_ids if hidden else [],
                    "nesting_preservation_required": hidden,
                    "output_surface_hidden": hidden,
                    "quotation_fidelity_required": hidden,
                    "scope_preservation_required": hidden,
                    "uncertainty_preservation_required": hidden,
                },
                "holder_role": actual_holder,
                "label_owner": label_owner,
                "license_id": LICENSE_ID,
                "logical_order": logical_order,
                "objective_keys": sorted(objective_keys),
                "observed_text": texts[index - 1],
                "perturbation_kind": f"LC14_{candidate_kind}",
                "proposition_candidates": components["propositions"],
                "quotation_spans": components["quotes"],
                "resource_budget": {
                    "max_attributions": 3,
                    "max_dependency_edges": 3,
                    "max_nesting_depth": 2,
                    "max_output_units": 160,
                    "max_propositions": 2,
                    "max_quote_spans": 2,
                },
                "retention_anchor_id": retention_anchor,
                "revision_receipt": {
                    "affected_attribution_ids": (
                        [components["attributions"][0]["attribution_id"]]
                        if candidate_kind == "REVISION" else []),
                    "candidate_version": revision_version,
                    "raw_observation_preserved": 1,
                    "revision_scope": (
                        "ATTRIBUTION_STATE_ONLY"
                        if candidate_kind == "REVISION" else "NONE"),
                    "supersedes_seed_id": supersedes,
                    "unaffected_quote_ids": (
                        sorted(item["quote_id"] for item in components["quotes"])
                        if candidate_kind == "REVISION" else []),
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
                "seed_id": ids[index],
                "source_channel": actual_channel,
                "split": split,
                "supersedes_seed_id": supersedes,
                "surface_scope": {
                    "context_scope_key": components["scope"],
                    "genre": "REPORTED_DISCOURSE",
                    "language": "zh",
                    "register": "NEUTRAL",
                    "script": "HAN",
                },
                "template_family": f"{owner}_LC14_TEMPLATE_{candidate_kind}",
                "transfer_receipt": components["transfer"],
            })
    return tuple(rows)


def default_attribution_quotation_sample_bytes() -> bytes:
    """返回规范 LC-14 JSONL，供正式 sample 重建和字节审计。"""
    return b"".join(
        canonical_json_line(value)
        for value in build_default_attribution_quotation_seed_values())


__all__ = [
    "COURSE_MANIFEST_PATH",
    "EVALUATOR_DIMENSIONS",
    "FORMAL_ARTIFACT_RELATIVE_ROOT",
    "PACK_NAME",
    "PAYLOAD_KIND",
    "AuthoredAttributionQuotationCourseError",
    "AuthoredAttributionQuotationSeed",
    "AttributionQuotationPayloadAudit",
    "build_attribution_quotation_course_manifest",
    "build_default_attribution_quotation_seed_values",
    "compile_authored_attribution_quotation_course",
    "default_attribution_quotation_sample_bytes",
    "read_authored_attribution_quotation_seeds",
    "validate_attribution_quotation_payload",
]
