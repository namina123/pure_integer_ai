"""LC-08 开放集、最小澄清与主动补证的原创课程。"""
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
PACK_NAME = "AUTHORED_CC0_V1--CC0-1.0--lc08-open-set-clarification-v1"
STAGE = "W-08"
SUBSTAGE = "LC08_OPEN_SET_CLARIFICATION"
PAYLOAD_KIND = "OpenSetClarificationCandidateV1"
COURSE_MANIFEST_ARTIFACT_VERSION = "LC-08-open-set-clarification-course-v1"
COURSE_MANIFEST_PATH = Path(
    "data/ph2/manifests/lc08_open_set_clarification_course_v1.json")
FORMAL_ARTIFACT_RELATIVE_ROOT = (
    "ph2_dataset_artifacts/d02_language_courses_v1")

CANDIDATE_KINDS = (
    "ACCESS_BLOCKED",
    "ACTIVE_EVIDENCE_REQUEST",
    "AMBIGUOUS_BRANCH",
    "ANSWER_EVIDENCE_UPDATE",
    "BUDGET_BLOCKED",
    "GENERATION",
    "INSUFFICIENT_GUESS_BASELINE",
    "KNOWN_SUFFICIENT_NO_QUESTION",
    "MINIMAL_CLARIFICATION",
    "NEW_CONSTRUCTION_DETECTION",
    "NEW_SENSE_AMBIGUITY",
    "NEW_USAGE_DETECTION",
    "NEW_WORD_DETECTION",
    "OVERQUESTION_BASELINE",
    "RETENTION",
    "REVISION",
    "UNKNOWN",
)
OPEN_SET_KINDS = (
    "KNOWN", "NEW_CONSTRUCTION", "NEW_SENSE", "NEW_USAGE", "NEW_WORD",
    "UNKNOWN")
DETECTION_STATES = ("KNOWN", "NOVEL", "UNKNOWN")
OBJECT_KINDS = ("CONSTRUCTION", "LEXEME", "SENSE", "USAGE")
LIFECYCLE_STATES = ("ACTIVE", "FORMING", "SUPERSEDED")
EVIDENCE_STATES = ("CONFLICT", "REFUTE", "SUPPORT", "UNKNOWN")
MISSING_REASONS = ("ACCESS", "AMBIGUOUS", "BUDGET", "NONE", "UNKNOWN")
OBLIGATION_STATES = ("NOT_REQUIRED", "OPEN", "SATISFIED", "UNKNOWN")
QUESTION_KINDS = (
    "EVIDENCE_REQUEST", "MINIMAL_BRANCH_ELIMINATION", "NO_QUESTION",
    "UNKNOWN")
PLAN_STATES = ("INVALID", "NE", "UNKNOWN", "VALID")
EVIDENCE_PATHS = (
    "BUDGET_STOP", "CLARIFICATION_ANSWER", "LOCAL_OBSERVATION", "NO_ACCESS",
    "NONE", "SOURCE_LOOKUP", "UNKNOWN")
REQUEST_STATES = ("NOT_REQUIRED", "OPEN", "SATISFIED", "UNKNOWN")
BASELINE_KINDS = (
    "INSUFFICIENT_EVIDENCE_GUESS",
    "SUFFICIENT_EVIDENCE_OVERQUESTION",
    "TYPED_OPEN_SET_OBJECTS_PRESENT",
)
EVALUATOR_DIMENSIONS = (
    "ACCESS_BLOCKED_DISTINCT",
    "ACTIVE_EVIDENCE_REQUEST",
    "AMBIGUOUS_BRANCH_PRESERVATION",
    "BUDGET_BLOCKED_DISTINCT",
    "CLARIFICATION_ANSWER_LOCAL_UPDATE",
    "CLARIFICATION_REVISION_SUPERSEDE",
    "INSUFFICIENT_GUESS_REJECT",
    "KNOWN_SUFFICIENT_NO_QUESTION",
    "MINIMAL_BRANCH_CLARIFICATION",
    "NEW_CONSTRUCTION_DETECTION",
    "NEW_SENSE_DETECTION",
    "NEW_USAGE_DETECTION",
    "NEW_WORD_DETECTION",
    "OPEN_SET_UNKNOWN",
    "OVERQUESTION_REJECT",
    "RETENTION_REVERIFY",
    "REVERSE_GENERATION_MINIMAL_QUESTION",
)
VERIFIER_NE_CONDITIONS = (
    "CAPABILITY_LEARNED_REQUESTED",
    "GENERATION_RESULT_NOT_EXECUTED",
    "MD03_RUNTIME_PREREQUISITE_MISSING",
    "NOVELTY_RUNTIME_NOT_EXECUTED",
    "NO_EVALUATOR_LABEL",
    "OUT_OF_DOCUMENT_SCOPE",
    "SEMANTIC_TRUTH_REQUESTED",
)
RETENTION_PROTOCOLS = (
    "A_TO_LC08_REVERIFY_A",
    "CLARIFICATION_DEPENDENCIES_ONLY",
    "DUMP_RESUME_REQUIRED_AT_RUNTIME",
)
COMBINATION_AXES = (
    "CLARIFICATION_NEED",
    "EVIDENCE_PATH",
    "OPEN_SET_KIND",
    "OPEN_SET_X_CLARIFICATION_X_EVIDENCE_PATH",
)
ABLATION_KEYS = (
    "DROP_CANDIDATE_BRANCH",
    "DROP_EVIDENCE_REQUEST",
    "DROP_MISSING_INFORMATION_OBLIGATION",
    "DROP_NOVELTY_PROFILE",
    "INSUFFICIENT_EVIDENCE_GUESS",
    "SUFFICIENT_EVIDENCE_OVERQUESTION",
)
PAYLOAD_KEYS = (
    "baseline_kind",
    "candidate_branches",
    "candidate_kind",
    "clarification_plan",
    "clarification_receipt",
    "evidence_request",
    "generation_constraint",
    "missing_information_obligations",
    "novelty_profile",
    "objective_keys",
    "observed_surface",
    "resource_budget",
    "retention_anchor_id",
    "sample_family",
    "selection_state",
    "split_identity",
    "surface_scope",
)

_SEED_FIELDS = {
    "baseline_kind", "candidate_branches", "candidate_kind",
    "clarification_need", "clarification_plan", "clarification_receipt",
    "evaluation_dimension", "evidence_path", "evidence_request",
    "expected_payload", "expected_state", "family", "generation_constraint",
    "label_owner", "license_id", "logical_order",
    "missing_information_obligations", "novelty_profile", "objective_keys",
    "observed_text", "open_set_kind", "perturbation_kind",
    "resource_budget", "retention_anchor_id", "sample_family", "sample_role",
    "seed_id", "split", "supersedes_seed_id", "surface_scope",
    "template_family",
}
_SCOPE_FIELDS = {
    "context_scope_key", "genre", "language", "register", "script"}
_NOVELTY_FIELDS = {
    "candidate_ids", "candidate_version", "context_scope_key",
    "detection_state", "observed_unit_key", "open_set_kind"}
_BRANCH_FIELDS = {
    "candidate_id", "candidate_version", "competition_key",
    "depends_on_obligation_id", "evidence_state", "lifecycle_state",
    "object_kind", "replacement_candidate_id"}
_OBLIGATION_FIELDS = {
    "blocking_reason", "candidate_ids", "candidate_version", "obligation_id",
    "required_evidence_kind", "scope_key", "status"}
_PLAN_FIELDS = {
    "access_required", "budget_cost", "expected_eliminated_branches",
    "minimality_checked", "output_surface_hidden", "plan_id", "plan_state",
    "question_kind", "target_candidate_ids"}
_REQUEST_FIELDS = {
    "access_state", "budget_state", "candidate_ids", "path_kind",
    "request_id", "required_source_kinds", "status"}
_RECEIPT_FIELDS = {
    "affected_candidate_ids", "candidate_version", "raw_observation_preserved",
    "superseded_candidate_ids", "supersedes_seed_id", "unaffected_candidate_ids",
    "update_scope"}
_GENERATION_FIELDS = {
    "branch_preservation_required", "direction", "evidence_scope_preservation_required",
    "input_candidate_ids", "novelty_preservation_required",
    "obligation_preservation_required", "output_surface_hidden",
    "question_minimality_required", "uncertainty_preservation_required"}
_BUDGET_FIELDS = {
    "max_candidate_branches", "max_dependency_edges", "max_evidence_requests",
    "max_obligations", "max_output_units", "max_question_cost"}
_EXPECTED_FIELDS = {
    "accepted", "accepted_surfaces", "analysis_key", "reason_code"}
_SPLIT_FIELDS = {
    "clarification_need", "combination_key", "evidence_path", "isolation_axis",
    "open_set_kind"}
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
_REQUIRED_EVIDENCE_BY_REASON = {
    "ACCESS": "ACCESS_GRANT",
    "AMBIGUOUS": "CONTRASTIVE_ANSWER",
    "BUDGET": "DEFERRED_BUDGET",
    "NONE": "NONE",
    "UNKNOWN": "SOURCE_EVIDENCE",
}


class AuthoredOpenSetClarificationCourseError(RuntimeError):
    """LC-08 seed、typed payload 或组合 split 不满足冻结合同。"""


def _exact_keys(value: Any, keys: set[str], *, where: str) -> dict[str, Any]:
    """要求对象字段集合精确等于合同。"""
    if not isinstance(value, dict) or set(value) != keys:
        raise AuthoredOpenSetClarificationCourseError(f"{where} 字段集合非法")
    return value


def _text(value: Any, *, where: str, allow_empty: bool = False) -> str:
    """要求文本无首尾空白，并按字段控制空值。"""
    if not isinstance(value, str) or value.strip() != value:
        raise AuthoredOpenSetClarificationCourseError(f"{where} 必须是规范文本")
    if not allow_empty and not value:
        raise AuthoredOpenSetClarificationCourseError(f"{where} 不能为空")
    return value


def _integer(value: Any, *, where: str, positive: bool = False) -> int:
    """要求资源、版本和计数使用严格整数。"""
    minimum = 1 if positive else 0
    if type(value) is not int or value < minimum:
        raise AuthoredOpenSetClarificationCourseError(
            f"{where} 必须是{'正' if positive else '非负'}严格整数")
    return value


def _flag(value: Any, *, where: str) -> int:
    """要求协议开关使用严格 0/1。"""
    if type(value) is not int or value not in {0, 1}:
        raise AuthoredOpenSetClarificationCourseError(f"{where} 必须是 0/1")
    return value


def _string_tuple(
        value: Any,
        *,
        where: str,
        sorted_unique: bool = False,
        allow_empty: bool = False,
        ) -> tuple[str, ...]:
    """恢复文本数组，并可强制排序去重。"""
    if not isinstance(value, list) or (not value and not allow_empty):
        raise AuthoredOpenSetClarificationCourseError(f"{where} 必须是文本数组")
    result = tuple(_text(item, where=where) for item in value)
    if len(result) != len(set(result)):
        raise AuthoredOpenSetClarificationCourseError(f"{where} 不得重复")
    if sorted_unique and tuple(sorted(result)) != result:
        raise AuthoredOpenSetClarificationCourseError(f"{where} 必须排序")
    return result


def _sha256_text(value: str) -> str:
    """返回 UTF-8 文本的规范 SHA-256。"""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_expected(
        payload: CanonicalJsonObject,
        *,
        expected_state: str,
        ) -> None:
    """核私有 label 与四态一致，不允许非 TRUE 携带采用表层。"""
    raw = _exact_keys(payload.to_value(), _EXPECTED_FIELDS, where="expected")
    accepted = _flag(raw["accepted"], where="expected accepted")
    surfaces = _string_tuple(
        raw["accepted_surfaces"], where="accepted surfaces", allow_empty=True)
    _text(raw["analysis_key"], where="analysis_key")
    _text(raw["reason_code"], where="reason_code")
    if expected_state == "TRUE" and (accepted != 1 or not surfaces):
        raise AuthoredOpenSetClarificationCourseError(
            "TRUE label 必须给出私有合法表层")
    if expected_state != "TRUE" and (accepted != 0 or surfaces):
        raise AuthoredOpenSetClarificationCourseError(
            "非 TRUE label 不得给出采用表层")


@dataclass(frozen=True)
class OpenSetClarificationPayloadAudit:
    """返回候选、obligation、请求和组合键计数。"""

    branch_count: int
    obligation_count: int
    request_count: int
    combination_key: str
    overquestion_baseline: int
    insufficient_guess_baseline: int


def validate_open_set_clarification_payload(
        payload: CanonicalJsonObject | dict[str, Any],
        ) -> OpenSetClarificationPayloadAudit:
    """校验开放集检测、最小澄清、补证、局部更新和停止原因。"""
    value = payload.to_value() if isinstance(payload, CanonicalJsonObject) else payload
    raw = _exact_keys(value, set(PAYLOAD_KEYS), where="open-set payload")
    candidate_kind = _text(raw["candidate_kind"], where="candidate_kind")
    if candidate_kind not in CANDIDATE_KINDS:
        raise AuthoredOpenSetClarificationCourseError("candidate_kind 未登记")
    baseline = _text(raw["baseline_kind"], where="baseline_kind")
    if baseline not in BASELINE_KINDS:
        raise AuthoredOpenSetClarificationCourseError("baseline_kind 未登记")
    if raw["sample_family"] not in SAMPLE_FAMILIES:
        raise AuthoredOpenSetClarificationCourseError("sample_family 未登记")
    if raw["selection_state"] != "UNSELECTED":
        raise AuthoredOpenSetClarificationCourseError(
            "Observation 不得预选开放集候选")
    objective_keys = _string_tuple(
        raw["objective_keys"], where="objective_keys", sorted_unique=True)
    if any(key not in LANGUAGE_OBJECTIVE_KEYS for key in objective_keys):
        raise AuthoredOpenSetClarificationCourseError("objective_keys 未登记")

    scope = _exact_keys(raw["surface_scope"], _SCOPE_FIELDS, where="scope")
    if scope["language"] != "zh" or scope["script"] != "HAN":
        raise AuthoredOpenSetClarificationCourseError("LC-08 首阶段只冻结 zh/HAN")
    context_scope = _text(scope["context_scope_key"], where="context scope")
    _text(scope["genre"], where="genre")
    if scope["register"] not in {"COLLOQUIAL", "FORMAL", "NEUTRAL"}:
        raise AuthoredOpenSetClarificationCourseError("register 未登记")

    observed = _exact_keys(raw["observed_surface"], {
        "append_only", "sha256", "target_hidden", "text",
    }, where="observed surface")
    observed_text = _text(observed["text"], where="observed text")
    if _flag(observed["append_only"], where="append_only") != 1:
        raise AuthoredOpenSetClarificationCourseError(
            "raw Observation 必须 append-only")
    if observed["sha256"] != _sha256_text(observed_text):
        raise AuthoredOpenSetClarificationCourseError(
            "observed surface SHA-256 漂移")
    target_hidden = _flag(observed["target_hidden"], where="target_hidden")

    budget = _exact_keys(raw["resource_budget"], _BUDGET_FIELDS, where="budget")
    ceilings = {
        key: _integer(budget[key], where=key, positive=True)
        for key in sorted(_BUDGET_FIELDS)
    }
    if len(observed_text) > ceilings["max_output_units"]:
        raise AuthoredOpenSetClarificationCourseError("surface 超整数输出预算")

    novelty = _exact_keys(raw["novelty_profile"], _NOVELTY_FIELDS, where="novelty")
    open_set_kind = _text(novelty["open_set_kind"], where="open_set_kind")
    if open_set_kind not in OPEN_SET_KINDS:
        raise AuthoredOpenSetClarificationCourseError("open_set_kind 未登记")
    detection_state = _text(novelty["detection_state"], where="detection_state")
    if detection_state not in DETECTION_STATES:
        raise AuthoredOpenSetClarificationCourseError("detection_state 未登记")
    if novelty["context_scope_key"] != context_scope:
        raise AuthoredOpenSetClarificationCourseError("novelty 跨 ContextScope")
    _text(novelty["observed_unit_key"], where="observed_unit_key")
    _integer(novelty["candidate_version"], where="novelty version", positive=True)
    novelty_candidate_ids = _string_tuple(
        novelty["candidate_ids"], where="novelty candidate ids",
        sorted_unique=True)
    expected_detection = (
        "KNOWN" if open_set_kind == "KNOWN"
        else "UNKNOWN" if open_set_kind == "UNKNOWN"
        else "NOVEL")
    if detection_state != expected_detection:
        raise AuthoredOpenSetClarificationCourseError(
            "open-set kind 与 novelty detection 不一致")

    branches_value = raw["candidate_branches"]
    if not isinstance(branches_value, list) or not branches_value:
        raise AuthoredOpenSetClarificationCourseError("candidate branches 不能为空")
    if len(branches_value) > ceilings["max_candidate_branches"]:
        raise AuthoredOpenSetClarificationCourseError("candidate branches 超预算")
    branches: dict[str, dict[str, Any]] = {}
    for item in branches_value:
        branch = _exact_keys(item, _BRANCH_FIELDS, where="candidate branch")
        candidate_id = _text(branch["candidate_id"], where="candidate_id")
        if candidate_id in branches:
            raise AuthoredOpenSetClarificationCourseError("candidate_id 重复")
        if branch["object_kind"] not in OBJECT_KINDS:
            raise AuthoredOpenSetClarificationCourseError("branch object_kind 未登记")
        if branch["lifecycle_state"] not in LIFECYCLE_STATES:
            raise AuthoredOpenSetClarificationCourseError("branch lifecycle 未登记")
        if branch["evidence_state"] not in EVIDENCE_STATES:
            raise AuthoredOpenSetClarificationCourseError("branch Evidence 未登记")
        _text(branch["competition_key"], where="competition_key")
        _integer(branch["candidate_version"], where="branch version", positive=True)
        dependency = _text(
            branch["depends_on_obligation_id"], where="branch dependency",
            allow_empty=True)
        replacement = _text(
            branch["replacement_candidate_id"], where="replacement candidate",
            allow_empty=True)
        if branch["lifecycle_state"] == "FORMING" and (
                branch["evidence_state"] != "UNKNOWN" or replacement):
            raise AuthoredOpenSetClarificationCourseError(
                "forming candidate 只能保持 unknown")
        if branch["lifecycle_state"] == "SUPERSEDED":
            if not replacement or branch["evidence_state"] != "REFUTE":
                raise AuthoredOpenSetClarificationCourseError(
                    "superseded candidate 必须绑定 replacement 和 refute")
        elif replacement:
            raise AuthoredOpenSetClarificationCourseError(
                "非 superseded candidate 不得绑定 replacement")
        branches[candidate_id] = branch
    if tuple(sorted(branches)) != novelty_candidate_ids:
        raise AuthoredOpenSetClarificationCourseError(
            "novelty candidate_ids 与 typed branches 不一致")
    for candidate_id, branch in branches.items():
        replacement = branch["replacement_candidate_id"]
        if replacement and (replacement not in branches or replacement == candidate_id):
            raise AuthoredOpenSetClarificationCourseError("replacement candidate 非法")

    obligations_value = raw["missing_information_obligations"]
    if not isinstance(obligations_value, list) or len(obligations_value) != 1:
        raise AuthoredOpenSetClarificationCourseError(
            "每条 LC-08 样本必须有一个 missing-information obligation")
    if len(obligations_value) > ceilings["max_obligations"]:
        raise AuthoredOpenSetClarificationCourseError("obligation 超预算")
    obligation = _exact_keys(
        obligations_value[0], _OBLIGATION_FIELDS, where="obligation")
    obligation_id = _text(obligation["obligation_id"], where="obligation_id")
    reason = _text(obligation["blocking_reason"], where="blocking reason")
    if reason not in MISSING_REASONS:
        raise AuthoredOpenSetClarificationCourseError("blocking reason 未登记")
    if obligation["status"] not in OBLIGATION_STATES:
        raise AuthoredOpenSetClarificationCourseError("obligation status 未登记")
    if obligation["scope_key"] != context_scope:
        raise AuthoredOpenSetClarificationCourseError("obligation 跨 ContextScope")
    _integer(obligation["candidate_version"], where="obligation version", positive=True)
    obligation_candidates = _string_tuple(
        obligation["candidate_ids"], where="obligation candidate ids",
        sorted_unique=True, allow_empty=True)
    if any(item not in branches for item in obligation_candidates):
        raise AuthoredOpenSetClarificationCourseError(
            "obligation 引用未知 candidate")
    if obligation["required_evidence_kind"] != _REQUIRED_EVIDENCE_BY_REASON[reason]:
        raise AuthoredOpenSetClarificationCourseError(
            "blocking reason 与 required Evidence 不一致")
    if reason == "NONE":
        if obligation["status"] != "NOT_REQUIRED" or obligation_candidates:
            raise AuthoredOpenSetClarificationCourseError(
                "Evidence 充分时不得伪造 missing obligation")
    elif not obligation_candidates or obligation["status"] == "NOT_REQUIRED":
        raise AuthoredOpenSetClarificationCourseError(
            "信息不足必须保留依赖 candidate")
    for candidate_id, branch in branches.items():
        dependency = branch["depends_on_obligation_id"]
        if (candidate_id in obligation_candidates) != (dependency == obligation_id):
            raise AuthoredOpenSetClarificationCourseError(
                "candidate 与 obligation 依赖不闭合")

    plan = _exact_keys(raw["clarification_plan"], _PLAN_FIELDS, where="plan")
    plan_id = _text(plan["plan_id"], where="plan_id")
    if plan["plan_state"] not in PLAN_STATES:
        raise AuthoredOpenSetClarificationCourseError("plan_state 未登记")
    question_kind = _text(plan["question_kind"], where="question_kind")
    if question_kind not in QUESTION_KINDS:
        raise AuthoredOpenSetClarificationCourseError("question_kind 未登记")
    plan_targets = _string_tuple(
        plan["target_candidate_ids"], where="plan targets",
        sorted_unique=True, allow_empty=True)
    if any(item not in branches for item in plan_targets):
        raise AuthoredOpenSetClarificationCourseError("plan 引用未知 candidate")
    eliminated = _integer(
        plan["expected_eliminated_branches"], where="eliminated branches")
    checked = _flag(plan["minimality_checked"], where="minimality checked")
    cost = _integer(plan["budget_cost"], where="question budget cost")
    access_required = _flag(plan["access_required"], where="access required")
    plan_hidden = _flag(plan["output_surface_hidden"], where="plan output hidden")
    if question_kind == "MINIMAL_BRANCH_ELIMINATION":
        if (len(plan_targets) < 2 or eliminated <= 0
                or eliminated >= len(plan_targets) or checked != 1):
            raise AuthoredOpenSetClarificationCourseError(
                "最小澄清问题必须排除严格子分支")
    elif question_kind == "EVIDENCE_REQUEST":
        if not plan_targets or eliminated != 0 or checked != 1:
            raise AuthoredOpenSetClarificationCourseError(
                "主动补证请求必须绑定 candidate")
    elif question_kind in {"NO_QUESTION", "UNKNOWN"}:
        if plan_targets or eliminated != 0 or cost != 0:
            raise AuthoredOpenSetClarificationCourseError(
                "无问题或 unknown plan 不得带分支/成本")

    request = _exact_keys(raw["evidence_request"], _REQUEST_FIELDS, where="request")
    request_id = _text(request["request_id"], where="request_id")
    evidence_path = _text(request["path_kind"], where="evidence path")
    if evidence_path not in EVIDENCE_PATHS:
        raise AuthoredOpenSetClarificationCourseError("evidence path 未登记")
    if request["status"] not in REQUEST_STATES:
        raise AuthoredOpenSetClarificationCourseError("request status 未登记")
    if request["access_state"] not in {
            "AVAILABLE", "NO_ACCESS", "NOT_REQUIRED", "UNKNOWN"}:
        raise AuthoredOpenSetClarificationCourseError("request access state 未登记")
    if request["budget_state"] not in {
            "AVAILABLE", "EXHAUSTED", "NOT_REQUIRED", "UNKNOWN"}:
        raise AuthoredOpenSetClarificationCourseError("request budget state 未登记")
    request_candidates = _string_tuple(
        request["candidate_ids"], where="request candidate ids",
        sorted_unique=True, allow_empty=True)
    if any(item not in branches for item in request_candidates):
        raise AuthoredOpenSetClarificationCourseError("request 引用未知 candidate")
    _string_tuple(
        request["required_source_kinds"], where="required source kinds",
        sorted_unique=True, allow_empty=True)
    if evidence_path == "NONE" and (
            request["status"] != "NOT_REQUIRED" or request_candidates):
        raise AuthoredOpenSetClarificationCourseError(
            "无补证路径不得伪造 request")
    if evidence_path != "NONE" and not request_candidates:
        raise AuthoredOpenSetClarificationCourseError(
            "补证路径必须绑定 candidate")
    if evidence_path == "NO_ACCESS" and request["access_state"] != "NO_ACCESS":
        raise AuthoredOpenSetClarificationCourseError("access blocker 未显式表示")
    if (evidence_path == "BUDGET_STOP"
            and request["budget_state"] != "EXHAUSTED"):
        raise AuthoredOpenSetClarificationCourseError("budget blocker 未显式表示")

    receipt = _exact_keys(
        raw["clarification_receipt"], _RECEIPT_FIELDS,
        where="clarification receipt")
    affected = _string_tuple(
        receipt["affected_candidate_ids"], where="affected candidates",
        sorted_unique=True, allow_empty=True)
    unaffected = _string_tuple(
        receipt["unaffected_candidate_ids"], where="unaffected candidates",
        sorted_unique=True, allow_empty=True)
    superseded_ids = _string_tuple(
        receipt["superseded_candidate_ids"], where="superseded candidates",
        sorted_unique=True, allow_empty=True)
    receipt_version = _integer(
        receipt["candidate_version"], where="receipt version", positive=True)
    receipt_supersedes = _text(
        receipt["supersedes_seed_id"], where="receipt supersedes",
        allow_empty=True)
    if _flag(receipt["raw_observation_preserved"], where="raw preserved") != 1:
        raise AuthoredOpenSetClarificationCourseError(
            "clarification answer 不得覆盖 raw Observation")
    if receipt["update_scope"] == "NONE":
        if (affected or unaffected or superseded_ids or receipt_supersedes
                or receipt_version != 1):
            raise AuthoredOpenSetClarificationCourseError(
                "无更新样本不得伪造 clarification receipt")
    elif receipt["update_scope"] == "CLARIFICATION_DEPENDENCIES_ONLY":
        if (not affected or set(affected) & set(unaffected)
                or set(affected) | set(unaffected) != set(branches)
                or not set(superseded_ids) <= set(affected)):
            raise AuthoredOpenSetClarificationCourseError(
                "clarification answer 只能更新依赖 candidate")
        if candidate_kind == "REVISION" and (
                not receipt_supersedes or receipt_version < 2):
            raise AuthoredOpenSetClarificationCourseError(
                "revision clarification receipt 不完整")
    else:
        raise AuthoredOpenSetClarificationCourseError("receipt update_scope 未登记")

    generation = _exact_keys(
        raw["generation_constraint"], _GENERATION_FIELDS,
        where="generation constraint")
    inputs = _string_tuple(
        generation["input_candidate_ids"], where="generation inputs")
    known_inputs = set(branches) | {obligation_id, plan_id, request_id}
    if any(item not in known_inputs for item in inputs):
        raise AuthoredOpenSetClarificationCourseError(
            "generation 引用未知 typed candidate")
    generation_hidden = _flag(
        generation["output_surface_hidden"], where="generation hidden")
    preservation_keys = (
        "branch_preservation_required", "evidence_scope_preservation_required",
        "novelty_preservation_required", "obligation_preservation_required",
        "question_minimality_required", "uncertainty_preservation_required",
    )
    for key in preservation_keys:
        _flag(generation[key], where=key)
    if candidate_kind == "GENERATION":
        if (generation["direction"] != "OBLIGATION_TO_QUESTION"
                or generation_hidden != 1 or plan_hidden != 1 or target_hidden != 1
                or any(generation[key] != 1 for key in preservation_keys)):
            raise AuthoredOpenSetClarificationCourseError(
                "generation 必须隐藏最小问题并逐层保持不确定性")
    elif generation_hidden != 0 or plan_hidden != 0 or target_hidden != 0:
        raise AuthoredOpenSetClarificationCourseError(
            "非 generation 不得隐藏输出表层")

    split = _exact_keys(raw["split_identity"], _SPLIT_FIELDS, where="split")
    split_open_set = _text(split["open_set_kind"], where="split open-set kind")
    clarification_need = _text(
        split["clarification_need"], where="clarification need")
    split_evidence_path = _text(split["evidence_path"], where="split evidence path")
    if split_open_set != open_set_kind:
        raise AuthoredOpenSetClarificationCourseError(
            "组合 open_set_kind 与 novelty 不一致")
    if clarification_need != question_kind:
        raise AuthoredOpenSetClarificationCourseError(
            "组合 clarification_need 与 plan 不一致")
    if split_evidence_path != evidence_path:
        raise AuthoredOpenSetClarificationCourseError(
            "组合 evidence_path 与 request 不一致")
    combination_key = f"{open_set_kind}::{clarification_need}::{evidence_path}"
    if (split["combination_key"] != combination_key
            or split["isolation_axis"] != (
                "OPEN_SET_X_CLARIFICATION_X_EVIDENCE_PATH")):
        raise AuthoredOpenSetClarificationCourseError(
            "组合 split identity 漂移")

    retention_anchor = _text(
        raw["retention_anchor_id"], where="retention anchor", allow_empty=True)
    if candidate_kind == "RETENTION" and not retention_anchor:
        raise AuthoredOpenSetClarificationCourseError(
            "retention 必须绑定更早 anchor")
    if candidate_kind != "RETENTION" and retention_anchor:
        raise AuthoredOpenSetClarificationCourseError(
            "非 retention 不得绑定 anchor")

    overquestion = int(baseline == "SUFFICIENT_EVIDENCE_OVERQUESTION")
    insufficient_guess = int(baseline == "INSUFFICIENT_EVIDENCE_GUESS")
    if candidate_kind == "OVERQUESTION_BASELINE":
        if (not overquestion or reason != "NONE"
                or question_kind != "MINIMAL_BRANCH_ELIMINATION"
                or plan["plan_state"] != "INVALID"):
            raise AuthoredOpenSetClarificationCourseError(
                "Evidence 充分时滥问必须是独立 FALSE 基线")
    elif overquestion:
        raise AuthoredOpenSetClarificationCourseError("overquestion 基线身份错配")
    if candidate_kind == "INSUFFICIENT_GUESS_BASELINE":
        if (not insufficient_guess or reason == "NONE"
                or question_kind != "NO_QUESTION"
                or plan["plan_state"] != "INVALID"):
            raise AuthoredOpenSetClarificationCourseError(
                "信息不足时猜测必须是独立 FALSE 基线")
    elif insufficient_guess:
        raise AuthoredOpenSetClarificationCourseError(
            "insufficient-guess 基线身份错配")

    expected_open_kinds = {
        "NEW_CONSTRUCTION_DETECTION": "NEW_CONSTRUCTION",
        "NEW_SENSE_AMBIGUITY": "NEW_SENSE",
        "NEW_USAGE_DETECTION": "NEW_USAGE",
        "NEW_WORD_DETECTION": "NEW_WORD",
    }
    expected_open_kind = expected_open_kinds.get(candidate_kind)
    if expected_open_kind is not None and open_set_kind != expected_open_kind:
        raise AuthoredOpenSetClarificationCourseError(
            "新现象类型与 novelty profile 不一致")
    if candidate_kind == "MINIMAL_CLARIFICATION" and not (
            reason == "AMBIGUOUS"
            and question_kind == "MINIMAL_BRANCH_ELIMINATION"
            and plan["plan_state"] == "VALID"):
        raise AuthoredOpenSetClarificationCourseError("最小澄清候选缺失")
    if candidate_kind == "ACTIVE_EVIDENCE_REQUEST" and not (
            question_kind == "EVIDENCE_REQUEST"
            and plan["plan_state"] == "VALID"
            and evidence_path == "SOURCE_LOOKUP"):
        raise AuthoredOpenSetClarificationCourseError("主动补证请求缺失")
    if candidate_kind == "ANSWER_EVIDENCE_UPDATE" and not (
            obligation["status"] == "SATISFIED"
            and receipt["update_scope"] == "CLARIFICATION_DEPENDENCIES_ONLY"):
        raise AuthoredOpenSetClarificationCourseError("澄清回答局部更新缺失")
    if candidate_kind == "KNOWN_SUFFICIENT_NO_QUESTION" and not (
            reason == "NONE" and question_kind == "NO_QUESTION"
            and plan["plan_state"] == "VALID"):
        raise AuthoredOpenSetClarificationCourseError(
            "Evidence 充分时必须停止提问")
    if candidate_kind == "BUDGET_BLOCKED" and not (
            reason == "BUDGET" and plan["plan_state"] == "NE"
            and evidence_path == "BUDGET_STOP"
            and cost > ceilings["max_question_cost"]):
        raise AuthoredOpenSetClarificationCourseError("budget blocker 未独立表示")
    if candidate_kind == "ACCESS_BLOCKED" and not (
            reason == "ACCESS" and plan["plan_state"] == "NE"
            and evidence_path == "NO_ACCESS" and access_required == 1):
        raise AuthoredOpenSetClarificationCourseError("access blocker 未独立表示")
    if candidate_kind == "AMBIGUOUS_BRANCH":
        competitions: dict[str, int] = {}
        for branch in branches_value:
            competitions[branch["competition_key"]] = (
                competitions.get(branch["competition_key"], 0) + 1)
        if (max(competitions.values()) < 2
                or question_kind != "MINIMAL_BRANCH_ELIMINATION"):
            raise AuthoredOpenSetClarificationCourseError(
                "ambiguous branch competition 缺失")
    if candidate_kind == "UNKNOWN" and not (
            open_set_kind == "UNKNOWN" and reason == "UNKNOWN"
            and question_kind == "UNKNOWN" and evidence_path == "UNKNOWN"):
        raise AuthoredOpenSetClarificationCourseError(
            "open-set unknown 未显式表示")
    if candidate_kind == "REVISION" and not (
            receipt["update_scope"] == "CLARIFICATION_DEPENDENCIES_ONLY"
            and superseded_ids
            and all(branches[item]["lifecycle_state"] == "SUPERSEDED"
                    for item in superseded_ids)):
        raise AuthoredOpenSetClarificationCourseError(
            "clarification revision supersede 缺失")

    dependency_edges = sum(
        1 for branch in branches_value if branch["depends_on_obligation_id"])
    dependency_edges += len(plan_targets) + len(request_candidates)
    if dependency_edges > ceilings["max_dependency_edges"]:
        raise AuthoredOpenSetClarificationCourseError("LC-08 依赖边超预算")
    if len(obligations_value) > ceilings["max_obligations"]:
        raise AuthoredOpenSetClarificationCourseError("obligation 超预算")
    if 1 > ceilings["max_evidence_requests"]:
        raise AuthoredOpenSetClarificationCourseError("evidence request 超预算")

    return OpenSetClarificationPayloadAudit(
        len(branches), len(obligations_value), 1, combination_key,
        overquestion, insufficient_guess)


@dataclass(frozen=True)
class AuthoredOpenSetClarificationSeed:
    """一个 owner、split、开放集候选集合和独立 label。"""

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
    novelty_profile: CanonicalJsonObject
    candidate_branches: tuple[CanonicalJsonObject, ...]
    missing_information_obligations: tuple[CanonicalJsonObject, ...]
    clarification_plan: CanonicalJsonObject
    evidence_request: CanonicalJsonObject
    clarification_receipt: CanonicalJsonObject
    generation_constraint: CanonicalJsonObject
    resource_budget: CanonicalJsonObject
    open_set_kind: str
    clarification_need: str
    evidence_path: str
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
        """核 owner/split/课程身份、私有 label 和 typed payload。"""
        for name, value in (
                ("seed_id", self.seed_id), ("family", self.family),
                ("template_family", self.template_family),
                ("open_set_kind", self.open_set_kind),
                ("clarification_need", self.clarification_need),
                ("evidence_path", self.evidence_path),
                ("perturbation_kind", self.perturbation_kind)):
            _text(value, where=name)
        if self.label_owner not in {"teacher", "evaluator"}:
            raise AuthoredOpenSetClarificationCourseError("label_owner 非法")
        if self.split != ("train" if self.label_owner == "teacher" else "held_out"):
            raise AuthoredOpenSetClarificationCourseError(
                "label_owner 与 split 不一致")
        if self.sample_family not in SAMPLE_FAMILIES:
            raise AuthoredOpenSetClarificationCourseError("sample_family 未登记")
        if self.sample_role not in SAMPLE_ROLES:
            raise AuthoredOpenSetClarificationCourseError("sample_role 未登记")
        if self.candidate_kind not in CANDIDATE_KINDS:
            raise AuthoredOpenSetClarificationCourseError("candidate_kind 未登记")
        _text(self.observed_text, where="observed_text")
        for name in (
                "surface_scope", "novelty_profile", "clarification_plan",
                "evidence_request", "clarification_receipt",
                "generation_constraint", "resource_budget"):
            if not isinstance(getattr(self, name), CanonicalJsonObject):
                raise AuthoredOpenSetClarificationCourseError(f"{name} 类型非法")
        for name in (
                "candidate_branches", "missing_information_obligations"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or any(
                    not isinstance(item, CanonicalJsonObject) for item in values):
                raise AuthoredOpenSetClarificationCourseError(f"{name} 类型非法")
        if self.baseline_kind not in BASELINE_KINDS:
            raise AuthoredOpenSetClarificationCourseError("baseline_kind 未登记")
        if self.objective_keys != tuple(sorted(set(self.objective_keys))):
            raise AuthoredOpenSetClarificationCourseError(
                "objective_keys 必须排序去重")
        if not self.objective_keys or any(
                item not in LANGUAGE_OBJECTIVE_KEYS for item in self.objective_keys):
            raise AuthoredOpenSetClarificationCourseError("objective_keys 未登记")
        if self.expected_state not in EXPECTED_STATES:
            raise AuthoredOpenSetClarificationCourseError("expected_state 非四态")
        if not isinstance(self.expected_payload, CanonicalJsonObject):
            raise AuthoredOpenSetClarificationCourseError(
                "expected_payload 类型非法")
        _validate_expected(self.expected_payload, expected_state=self.expected_state)
        if self.evaluation_dimension not in EVALUATOR_DIMENSIONS:
            raise AuthoredOpenSetClarificationCourseError(
                "evaluation_dimension 未登记")
        _text(self.supersedes_seed_id, where="supersedes_seed_id", allow_empty=True)
        _text(self.retention_anchor_id, where="retention_anchor_id", allow_empty=True)
        _integer(self.logical_order, where="logical_order", positive=True)
        if self.license_id != LICENSE_ID:
            raise AuthoredOpenSetClarificationCourseError(
                "原创 LC-08 课程必须为 CC0-1.0")
        validate_open_set_clarification_payload(self.observation_payload())

    def observation_payload(self) -> CanonicalJsonObject:
        """只投影学生可见 typed Observation，不包含 expected/label。"""
        generation = self.generation_constraint.to_value()
        return CanonicalJsonObject.from_value({
            "baseline_kind": self.baseline_kind,
            "candidate_branches": [
                item.to_value() for item in self.candidate_branches],
            "candidate_kind": self.candidate_kind,
            "clarification_plan": self.clarification_plan.to_value(),
            "clarification_receipt": self.clarification_receipt.to_value(),
            "evidence_request": self.evidence_request.to_value(),
            "generation_constraint": generation,
            "missing_information_obligations": [
                item.to_value() for item in self.missing_information_obligations],
            "novelty_profile": self.novelty_profile.to_value(),
            "objective_keys": list(self.objective_keys),
            "observed_surface": {
                "append_only": 1,
                "sha256": _sha256_text(self.observed_text),
                "target_hidden": generation["output_surface_hidden"],
                "text": self.observed_text,
            },
            "resource_budget": self.resource_budget.to_value(),
            "retention_anchor_id": self.retention_anchor_id,
            "sample_family": self.sample_family,
            "selection_state": "UNSELECTED",
            "split_identity": {
                "clarification_need": self.clarification_need,
                "combination_key": (
                    f"{self.open_set_kind}::{self.clarification_need}::"
                    f"{self.evidence_path}"),
                "evidence_path": self.evidence_path,
                "isolation_axis": (
                    "OPEN_SET_X_CLARIFICATION_X_EVIDENCE_PATH"),
                "open_set_kind": self.open_set_kind,
            },
            "surface_scope": self.surface_scope.to_value(),
        })

    def compiled_seed(self) -> AuthoredCompiledSeed:
        """复用四 owner publisher 编译 Source/Observation/Label record。"""
        payload = self.observation_payload()
        audit = validate_open_set_clarification_payload(payload)
        branch_ids = tuple(
            item.to_value()["candidate_id"] for item in self.candidate_branches)
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
            (self.seed_id, branch_ids, audit.combination_key),
            (self.observed_text, self.family),
            (
                self.candidate_kind, self.open_set_kind,
                self.clarification_need, self.evidence_path,
                audit.branch_count, audit.obligation_count, audit.request_count,
            ),
            self.evaluation_dimension,
        )


def _canonical_objects(
        values: Any,
        keys: set[str],
        *,
        where: str,
        ) -> tuple[CanonicalJsonObject, ...]:
    """把严格字段对象数组恢复为规范不可变对象。"""
    if not isinstance(values, list):
        raise AuthoredOpenSetClarificationCourseError(f"{where} 必须是数组")
    return tuple(CanonicalJsonObject.from_value(
        _exact_keys(item, keys, where=where)) for item in values)


def _seed_from_value(value: dict[str, Any]) -> AuthoredOpenSetClarificationSeed:
    """从精确字段字典恢复一条 LC-08 seed。"""
    raw = _exact_keys(value, _SEED_FIELDS, where="open-set seed")
    expected = raw["expected_payload"]
    if not isinstance(expected, dict):
        raise AuthoredOpenSetClarificationCourseError(
            "expected_payload 必须是对象")
    return AuthoredOpenSetClarificationSeed(
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
        CanonicalJsonObject.from_value(_exact_keys(
            raw["novelty_profile"], _NOVELTY_FIELDS, where="novelty")),
        _canonical_objects(
            raw["candidate_branches"], _BRANCH_FIELDS, where="branches"),
        _canonical_objects(
            raw["missing_information_obligations"], _OBLIGATION_FIELDS,
            where="obligations"),
        CanonicalJsonObject.from_value(_exact_keys(
            raw["clarification_plan"], _PLAN_FIELDS, where="plan")),
        CanonicalJsonObject.from_value(_exact_keys(
            raw["evidence_request"], _REQUEST_FIELDS, where="request")),
        CanonicalJsonObject.from_value(_exact_keys(
            raw["clarification_receipt"], _RECEIPT_FIELDS, where="receipt")),
        CanonicalJsonObject.from_value(_exact_keys(
            raw["generation_constraint"], _GENERATION_FIELDS,
            where="generation")),
        CanonicalJsonObject.from_value(_exact_keys(
            raw["resource_budget"], _BUDGET_FIELDS, where="budget")),
        _text(raw["open_set_kind"], where="open_set_kind"),
        _text(raw["clarification_need"], where="clarification_need"),
        _text(raw["evidence_path"], where="evidence_path"),
        _text(raw["baseline_kind"], where="baseline_kind"),
        _string_tuple(
            raw["objective_keys"], where="objective_keys", sorted_unique=True),
        _text(raw["expected_state"], where="expected_state"),
        CanonicalJsonObject.from_value(_exact_keys(
            expected, _EXPECTED_FIELDS, where="expected_payload")),
        _text(raw["evaluation_dimension"], where="evaluation_dimension"),
        _text(raw["perturbation_kind"], where="perturbation_kind"),
        _text(raw["supersedes_seed_id"], where="supersedes_seed_id",
              allow_empty=True),
        _text(raw["retention_anchor_id"], where="retention_anchor_id",
              allow_empty=True),
        _integer(raw["logical_order"], where="logical_order", positive=True),
        _text(raw["license_id"], where="license_id"),
    )


def read_authored_open_set_clarification_seeds(
        path: str | Path,
        ) -> tuple[AuthoredOpenSetClarificationSeed, ...]:
    """严格读取 JSONL，并核双 owner、局部修正和组合 held-out。"""
    try:
        payload = Path(path).read_bytes()
    except OSError as error:
        raise AuthoredOpenSetClarificationCourseError(
            "open-set clarification sample 无法读取") from error
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise AuthoredOpenSetClarificationCourseError(
            "open-set clarification sample 换行非法")
    seeds: list[AuthoredOpenSetClarificationSeed] = []
    try:
        for line in payload.splitlines(keepends=True):
            value = parse_canonical_json_bytes(line[:-1], require_object=True)
            assert isinstance(value, dict)
            if canonical_json_line(value) != line:
                raise AuthoredOpenSetClarificationCourseError(
                    "open-set clarification seed 非规范 JSON")
            seeds.append(_seed_from_value(value))
    except AuthoredOpenSetClarificationCourseError:
        raise
    except Exception as error:
        raise AuthoredOpenSetClarificationCourseError(
            "open-set clarification seed 损坏") from error
    identifiers = [item.seed_id for item in seeds]
    orders = [item.logical_order for item in seeds]
    if not seeds or len(identifiers) != len(set(identifiers)):
        raise AuthoredOpenSetClarificationCourseError("seed_id 空或重复")
    if orders != sorted(orders) or len(orders) != len(set(orders)):
        raise AuthoredOpenSetClarificationCourseError(
            "logical_order 必须严格递增")
    index = {item.seed_id: item for item in seeds}
    for seed in seeds:
        expected_role = _ROLE_BY_FAMILY.get(seed.sample_family)
        if expected_role is not None and seed.sample_role != expected_role:
            raise AuthoredOpenSetClarificationCourseError(
                "sample family/role 不一致")
        if seed.sample_family == "GENERATION" and seed.sample_role not in {
                "support", "read_only_probe", "refute"}:
            raise AuthoredOpenSetClarificationCourseError(
                "generation sample_role 非法")
        expected_state = _STATE_BY_FAMILY.get(seed.sample_family)
        if expected_state is not None and seed.expected_state != expected_state:
            raise AuthoredOpenSetClarificationCourseError(
                "sample family/state 不一致")
        if seed.sample_family == "GENERATION" and seed.expected_state not in {
                "TRUE", "FALSE"}:
            raise AuthoredOpenSetClarificationCourseError(
                "generation state 非法")
        if seed.candidate_kind == "REVISION":
            target = index.get(seed.supersedes_seed_id)
            if (target is None or target.logical_order >= seed.logical_order
                    or target.family != seed.family or target.split != seed.split):
                raise AuthoredOpenSetClarificationCourseError(
                    "revision 必须 supersede 同族更早 seed")
            old = target.observation_payload().to_value()
            new = seed.observation_payload().to_value()
            for key in ("novelty_profile", "clarification_plan"):
                if old[key] != new[key]:
                    raise AuthoredOpenSetClarificationCourseError(
                        "clarification answer 不得替换 novelty/plan/request 身份")
            for key in (
                    "access_state", "budget_state", "candidate_ids", "path_kind",
                    "request_id", "required_source_kinds"):
                if old["evidence_request"][key] != new["evidence_request"][key]:
                    raise AuthoredOpenSetClarificationCourseError(
                        "clarification answer 不得替换 evidence request 身份")
            if (old["evidence_request"]["status"] != "OPEN"
                    or new["evidence_request"]["status"] != "SATISFIED"):
                raise AuthoredOpenSetClarificationCourseError(
                    "clarification answer 未满足原 evidence request")
            old_branches = {
                item["candidate_id"]: item for item in old["candidate_branches"]}
            new_branches = {
                item["candidate_id"]: item for item in new["candidate_branches"]}
            if set(old_branches) != set(new_branches):
                raise AuthoredOpenSetClarificationCourseError(
                    "clarification revision 不得增删 candidate 身份")
            receipt = new["clarification_receipt"]
            affected = set(receipt["affected_candidate_ids"])
            unaffected = set(receipt["unaffected_candidate_ids"])
            stable_fields = {
                "candidate_id", "competition_key", "depends_on_obligation_id",
                "object_kind"}
            for candidate_id in sorted(old_branches):
                old_branch = old_branches[candidate_id]
                new_branch = new_branches[candidate_id]
                if any(old_branch[key] != new_branch[key] for key in stable_fields):
                    raise AuthoredOpenSetClarificationCourseError(
                        "clarification revision 替换了 candidate 身份")
                changed = old_branch != new_branch
                if changed != (candidate_id in affected):
                    raise AuthoredOpenSetClarificationCourseError(
                        "clarification answer 更新了非依赖 candidate")
                if candidate_id in affected and (
                        new_branch["candidate_version"]
                        <= old_branch["candidate_version"]):
                    raise AuthoredOpenSetClarificationCourseError(
                        "clarification candidate version 必须前进")
                if candidate_id in unaffected and changed:
                    raise AuthoredOpenSetClarificationCourseError(
                        "unaffected candidate 必须 bit-identical")
            if unaffected != (set(old_branches) - affected):
                raise AuthoredOpenSetClarificationCourseError(
                    "clarification receipt unaffected 集合不完整")
            old_obligation = old["missing_information_obligations"][0]
            new_obligation = new["missing_information_obligations"][0]
            for key in (
                    "blocking_reason", "candidate_ids", "obligation_id",
                    "required_evidence_kind", "scope_key"):
                if old_obligation[key] != new_obligation[key]:
                    raise AuthoredOpenSetClarificationCourseError(
                        "clarification answer 替换了 obligation 身份")
            if (new_obligation["status"] != "SATISFIED"
                    or new_obligation["candidate_version"]
                    <= old_obligation["candidate_version"]):
                raise AuthoredOpenSetClarificationCourseError(
                    "clarification answer 未满足原 obligation")
        elif seed.supersedes_seed_id:
            raise AuthoredOpenSetClarificationCourseError(
                "非 revision 不得 supersede")
        if seed.candidate_kind == "RETENTION":
            anchor = index.get(seed.retention_anchor_id)
            if (anchor is None or anchor.logical_order >= seed.logical_order
                    or anchor.split != seed.split):
                raise AuthoredOpenSetClarificationCourseError(
                    "retention anchor 必须是同 split 更早 seed")
        elif seed.retention_anchor_id:
            raise AuthoredOpenSetClarificationCourseError(
                "非 retention 不得绑定 anchor")

    teacher = tuple(item for item in seeds if item.label_owner == "teacher")
    evaluator = tuple(item for item in seeds if item.label_owner == "evaluator")
    for owner, owner_seeds in (("teacher", teacher), ("evaluator", evaluator)):
        if {item.sample_family for item in owner_seeds} != set(SAMPLE_FAMILIES):
            raise AuthoredOpenSetClarificationCourseError(
                f"{owner} 未覆盖七类 sample family")
        if {item.candidate_kind for item in owner_seeds} != set(CANDIDATE_KINDS):
            raise AuthoredOpenSetClarificationCourseError(
                f"{owner} 未覆盖完整 LC-08 候选族")
        overquestion = tuple(
            item for item in owner_seeds
            if item.baseline_kind == "SUFFICIENT_EVIDENCE_OVERQUESTION")
        guessing = tuple(
            item for item in owner_seeds
            if item.baseline_kind == "INSUFFICIENT_EVIDENCE_GUESS")
        if (len(overquestion) != 1 or overquestion[0].expected_state != "FALSE"
                or len(guessing) != 1 or guessing[0].expected_state != "FALSE"):
            raise AuthoredOpenSetClarificationCourseError(
                f"{owner} 两类澄清负基线未冻结")
    if ({item.family for item in teacher} & {item.family for item in evaluator}
            or {item.template_family for item in teacher}
            & {item.template_family for item in evaluator}):
        raise AuthoredOpenSetClarificationCourseError(
            "teacher/evaluator family/template 泄漏")
    teacher_open = {item.open_set_kind for item in teacher}
    teacher_clarification = {item.clarification_need for item in teacher}
    teacher_evidence = {item.evidence_path for item in teacher}
    teacher_triples = {
        (item.open_set_kind, item.clarification_need, item.evidence_path)
        for item in teacher}
    held_out = {
        (item.open_set_kind, item.clarification_need, item.evidence_path)
        for item in evaluator
        if item.open_set_kind in teacher_open
        and item.clarification_need in teacher_clarification
        and item.evidence_path in teacher_evidence
        and (item.open_set_kind, item.clarification_need, item.evidence_path)
        not in teacher_triples
    }
    if len(held_out) < 3:
        raise AuthoredOpenSetClarificationCourseError(
            "held-out open-set×clarification×evidence 组合不足")
    if {item.evaluation_dimension for item in evaluator} != set(
            EVALUATOR_DIMENSIONS):
        raise AuthoredOpenSetClarificationCourseError(
            "独立 evaluator 维度未列全")
    return tuple(seeds)


def _course_spec() -> AuthoredCourseSpec:
    """返回复用四 owner publisher 的 LC-08 课程规格。"""
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
        "authored-open-set-clarification-seed-v1",
        "urn:pure-integer-ai:ph2:authored-open-set-clarification-v1",
        "Pure Integer AI PH2 authored open-set clarification seed",
        "OPEN_SET_CLARIFICATION_LABEL",
        "lc08-open-set-clarification",
        640,
    )


def compile_authored_open_set_clarification_course(
        sample_path: str | Path,
        release_root: str | Path,
        ) -> AuthoredCourseBuild:
    """编译 LC-08 课程，不执行训练或 runtime consumer。"""
    seeds = read_authored_open_set_clarification_seeds(sample_path)
    return publish_authored_course(
        tuple(item.compiled_seed() for item in seeds),
        sample_path,
        release_root,
        _course_spec(),
    )


def build_open_set_clarification_course_manifest(
        sample_path: str | Path,
        build: AuthoredCourseBuild,
        *,
        artifact_relative_root: str = FORMAL_ARTIFACT_RELATIVE_ROOT,
        ) -> CapabilityCourseManifest:
    """构造 COURSE_FROZEN/NOT_STARTED 且 execution 全零的 manifest。"""
    sample = Path(sample_path)
    seeds = read_authored_open_set_clarification_seeds(sample)
    teacher = tuple(item for item in seeds if item.label_owner == "teacher")
    evaluator = tuple(item for item in seeds if item.label_owner == "evaluator")
    return CapabilityCourseManifest(
        1,
        COURSE_MANIFEST_ARTIFACT_VERSION,
        "COURSE_FROZEN",
        "NOT_STARTED",
        ("LC-08",),
        ("OPEN_SET_CONTINUAL_LEARNING", "PRAGMATIC_CLARIFICATION_REPAIR"),
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


def _object_kind(open_set_kind: str) -> str:
    """把开放集现象轴映射到课程候选对象族。"""
    return {
        "KNOWN": "USAGE",
        "NEW_CONSTRUCTION": "CONSTRUCTION",
        "NEW_SENSE": "SENSE",
        "NEW_USAGE": "USAGE",
        "NEW_WORD": "LEXEME",
        "UNKNOWN": "USAGE",
    }[open_set_kind]


def _request_components(
        identity_token: str,
        evidence_path: str,
        candidate_ids: list[str],
        *,
        satisfied: bool,
        ) -> dict[str, Any]:
    """按 evidence path 构造可审计的访问、预算和来源请求。"""
    if evidence_path == "NONE":
        return {
            "access_state": "NOT_REQUIRED",
            "budget_state": "NOT_REQUIRED",
            "candidate_ids": [],
            "path_kind": evidence_path,
            "request_id": f"{identity_token}_REQUEST_1",
            "required_source_kinds": [],
            "status": "NOT_REQUIRED",
        }
    source_kinds = {
        "BUDGET_STOP": ["SOURCE_REF"],
        "CLARIFICATION_ANSWER": ["CLARIFICATION_ANSWER"],
        "LOCAL_OBSERVATION": ["LOCAL_OBSERVATION"],
        "NO_ACCESS": ["SOURCE_REF"],
        "SOURCE_LOOKUP": ["SOURCE_REF"],
        "UNKNOWN": ["UNKNOWN_SOURCE"],
    }[evidence_path]
    access_state = "AVAILABLE"
    budget_state = "AVAILABLE"
    status = "SATISFIED" if satisfied else "OPEN"
    if evidence_path == "NO_ACCESS":
        access_state = "NO_ACCESS"
    elif evidence_path == "BUDGET_STOP":
        budget_state = "EXHAUSTED"
    elif evidence_path == "UNKNOWN":
        access_state = "UNKNOWN"
        budget_state = "UNKNOWN"
        status = "UNKNOWN"
    return {
        "access_state": access_state,
        "budget_state": budget_state,
        "candidate_ids": sorted(candidate_ids),
        "path_kind": evidence_path,
        "request_id": f"{identity_token}_REQUEST_1",
        "required_source_kinds": source_kinds,
        "status": status,
    }


def _scenario_components(
        owner: str,
        ordinal: int,
        candidate_kind: str,
        open_set_kind: str,
        question_kind: str,
        evidence_path: str,
        blocking_reason: str,
        *,
        revision_version: int = 1,
        shared_revision_identity: bool = False,
        ) -> dict[str, Any]:
    """构造一个 LC-08 场景，并保持澄清回答修正链身份稳定。"""
    token = f"{owner}_{ordinal:02d}"
    identity_token = f"{owner}_REVISION" if shared_revision_identity else token
    scope = (
        f"{owner}_CTX_REVISION" if shared_revision_identity
        else f"{token}_CTX")
    obligation_id = f"{identity_token}_OBLIGATION_1"
    minimum_two = question_kind == "MINIMAL_BRANCH_ELIMINATION"
    local_update = candidate_kind in {"ANSWER_EVIDENCE_UPDATE", "REVISION"}
    branch_count = 3 if local_update or shared_revision_identity else (
        2 if minimum_two else 1)
    branch_ids = [
        f"{identity_token}_CANDIDATE_{index}"
        for index in range(1, branch_count + 1)]
    dependent_ids = [] if blocking_reason == "NONE" else (
        branch_ids[:2] if branch_count >= 2 else branch_ids[:1])
    branches: list[dict[str, Any]] = []
    for index, candidate_id in enumerate(branch_ids, start=1):
        lifecycle = "FORMING"
        evidence_state = "UNKNOWN"
        version = 1
        replacement = ""
        competition = (
            f"{identity_token}_COMPETITION_SHARED"
            if minimum_two and index <= 2
            else f"{identity_token}_COMPETITION_{index}")
        if blocking_reason == "NONE":
            lifecycle = "ACTIVE"
            evidence_state = "SUPPORT"
        if shared_revision_identity and index == 3:
            lifecycle = "ACTIVE"
            evidence_state = "SUPPORT"
        if local_update:
            if index == 1:
                lifecycle = "ACTIVE"
                evidence_state = "SUPPORT"
                version = revision_version
            elif index == 2:
                lifecycle = "SUPERSEDED"
                evidence_state = "REFUTE"
                version = revision_version
                replacement = branch_ids[0]
            else:
                lifecycle = "ACTIVE"
                evidence_state = "SUPPORT"
        branches.append({
            "candidate_id": candidate_id,
            "candidate_version": version,
            "competition_key": competition,
            "depends_on_obligation_id": (
                obligation_id if candidate_id in dependent_ids else ""),
            "evidence_state": evidence_state,
            "lifecycle_state": lifecycle,
            "object_kind": _object_kind(open_set_kind),
            "replacement_candidate_id": replacement,
        })

    detection_state = (
        "KNOWN" if open_set_kind == "KNOWN"
        else "UNKNOWN" if open_set_kind == "UNKNOWN"
        else "NOVEL")
    obligation_status = "NOT_REQUIRED" if blocking_reason == "NONE" else "OPEN"
    if candidate_kind == "UNKNOWN":
        obligation_status = "UNKNOWN"
    if local_update:
        obligation_status = "SATISFIED"
    novelty = {
        "candidate_ids": sorted(branch_ids),
        "candidate_version": 1,
        "context_scope_key": scope,
        "detection_state": detection_state,
        "observed_unit_key": f"{identity_token}_OBSERVED_UNIT",
        "open_set_kind": open_set_kind,
    }
    obligation = {
        "blocking_reason": blocking_reason,
        "candidate_ids": sorted(dependent_ids),
        "candidate_version": revision_version if local_update else 1,
        "obligation_id": obligation_id,
        "required_evidence_kind": _REQUIRED_EVIDENCE_BY_REASON[blocking_reason],
        "scope_key": scope,
        "status": obligation_status,
    }

    plan_state = "VALID"
    if candidate_kind in {
            "INSUFFICIENT_GUESS_BASELINE", "OVERQUESTION_BASELINE"}:
        plan_state = "INVALID"
    elif candidate_kind in {"ACCESS_BLOCKED", "BUDGET_BLOCKED"}:
        plan_state = "NE"
    elif candidate_kind == "UNKNOWN":
        plan_state = "UNKNOWN"
    plan_targets = (
        branch_ids[:2] if question_kind == "MINIMAL_BRANCH_ELIMINATION"
        else branch_ids[:1] if question_kind == "EVIDENCE_REQUEST"
        else [])
    cost = 1 if question_kind in {
        "EVIDENCE_REQUEST", "MINIMAL_BRANCH_ELIMINATION"} else 0
    if candidate_kind == "BUDGET_BLOCKED":
        cost = 2
    generation_hidden = int(candidate_kind == "GENERATION")
    plan = {
        "access_required": int(candidate_kind == "ACCESS_BLOCKED"),
        "budget_cost": cost,
        "expected_eliminated_branches": (
            1 if question_kind == "MINIMAL_BRANCH_ELIMINATION" else 0),
        "minimality_checked": int(question_kind != "UNKNOWN"),
        "output_surface_hidden": generation_hidden,
        "plan_id": f"{identity_token}_PLAN_1",
        "plan_state": plan_state,
        "question_kind": question_kind,
        "target_candidate_ids": sorted(plan_targets),
    }
    request = _request_components(
        identity_token,
        evidence_path,
        dependent_ids,
        satisfied=local_update,
    )
    receipt = {
        "affected_candidate_ids": [],
        "candidate_version": 1,
        "raw_observation_preserved": 1,
        "superseded_candidate_ids": [],
        "supersedes_seed_id": "",
        "unaffected_candidate_ids": [],
        "update_scope": "NONE",
    }
    if local_update:
        receipt = {
            "affected_candidate_ids": sorted(branch_ids[:2]),
            "candidate_version": revision_version,
            "raw_observation_preserved": 1,
            "superseded_candidate_ids": [branch_ids[1]],
            "supersedes_seed_id": "PENDING" if candidate_kind == "REVISION" else "",
            "unaffected_candidate_ids": sorted(branch_ids[2:]),
            "update_scope": "CLARIFICATION_DEPENDENCIES_ONLY",
        }
    return {
        "candidate_branches": branches,
        "clarification_plan": plan,
        "clarification_receipt": receipt,
        "evidence_request": request,
        "missing_information_obligations": [obligation],
        "novelty_profile": novelty,
        "scope": scope,
    }


_SCENARIOS = (
    ("NEW_WORD_DETECTION", "POSITIVE", "NEW_WORD", "EVIDENCE_REQUEST", "SOURCE_LOOKUP", "UNKNOWN", "NEW_WORD_DETECTION"),
    ("NEW_SENSE_AMBIGUITY", "POSITIVE", "NEW_SENSE", "MINIMAL_BRANCH_ELIMINATION", "CLARIFICATION_ANSWER", "AMBIGUOUS", "NEW_SENSE_DETECTION"),
    ("NEW_CONSTRUCTION_DETECTION", "POSITIVE", "NEW_CONSTRUCTION", "EVIDENCE_REQUEST", "LOCAL_OBSERVATION", "UNKNOWN", "NEW_CONSTRUCTION_DETECTION"),
    ("NEW_USAGE_DETECTION", "POSITIVE", "NEW_USAGE", "EVIDENCE_REQUEST", "SOURCE_LOOKUP", "UNKNOWN", "NEW_USAGE_DETECTION"),
    ("MINIMAL_CLARIFICATION", "POSITIVE", "NEW_SENSE", "MINIMAL_BRANCH_ELIMINATION", "CLARIFICATION_ANSWER", "AMBIGUOUS", "MINIMAL_BRANCH_CLARIFICATION"),
    ("ACTIVE_EVIDENCE_REQUEST", "POSITIVE", "NEW_WORD", "EVIDENCE_REQUEST", "SOURCE_LOOKUP", "UNKNOWN", "ACTIVE_EVIDENCE_REQUEST"),
    ("ANSWER_EVIDENCE_UPDATE", "POSITIVE", "NEW_SENSE", "MINIMAL_BRANCH_ELIMINATION", "CLARIFICATION_ANSWER", "AMBIGUOUS", "CLARIFICATION_ANSWER_LOCAL_UPDATE"),
    ("KNOWN_SUFFICIENT_NO_QUESTION", "POSITIVE", "KNOWN", "NO_QUESTION", "NONE", "NONE", "KNOWN_SUFFICIENT_NO_QUESTION"),
    ("OVERQUESTION_BASELINE", "NEGATIVE", "KNOWN", "MINIMAL_BRANCH_ELIMINATION", "NONE", "NONE", "OVERQUESTION_REJECT"),
    ("INSUFFICIENT_GUESS_BASELINE", "NEGATIVE", "NEW_USAGE", "NO_QUESTION", "LOCAL_OBSERVATION", "UNKNOWN", "INSUFFICIENT_GUESS_REJECT"),
    ("BUDGET_BLOCKED", "UNKNOWN", "NEW_CONSTRUCTION", "EVIDENCE_REQUEST", "BUDGET_STOP", "BUDGET", "BUDGET_BLOCKED_DISTINCT"),
    ("ACCESS_BLOCKED", "UNKNOWN", "NEW_WORD", "EVIDENCE_REQUEST", "NO_ACCESS", "ACCESS", "ACCESS_BLOCKED_DISTINCT"),
    ("AMBIGUOUS_BRANCH", "AMBIGUOUS", "NEW_SENSE", "MINIMAL_BRANCH_ELIMINATION", "CLARIFICATION_ANSWER", "AMBIGUOUS", "AMBIGUOUS_BRANCH_PRESERVATION"),
    ("UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN", "OPEN_SET_UNKNOWN"),
    ("REVISION", "REVISION", "NEW_SENSE", "MINIMAL_BRANCH_ELIMINATION", "CLARIFICATION_ANSWER", "AMBIGUOUS", "CLARIFICATION_REVISION_SUPERSEDE"),
    ("RETENTION", "RETENTION", "NEW_CONSTRUCTION", "EVIDENCE_REQUEST", "LOCAL_OBSERVATION", "UNKNOWN", "RETENTION_REVERIFY"),
    ("GENERATION", "GENERATION", "NEW_USAGE", "MINIMAL_BRANCH_ELIMINATION", "CLARIFICATION_ANSWER", "AMBIGUOUS", "REVERSE_GENERATION_MINIMAL_QUESTION"),
)

_EVALUATOR_EVIDENCE_OVERRIDES = {
    1: "LOCAL_OBSERVATION",
    2: "SOURCE_LOOKUP",
    3: "SOURCE_LOOKUP",
    4: "LOCAL_OBSERVATION",
    5: "SOURCE_LOOKUP",
    15: "SOURCE_LOOKUP",
    17: "LOCAL_OBSERVATION",
}

_TEACHER_TEXTS = (
    "记录中首次出现词形‘澄络’，现有词表没有该项。",
    "‘澄’在这里可能指水变清，也可能指查清事实。",
    "‘把门一关’呈现了此前未登记的构式用法。",
    "熟悉词‘浮标’在本句中承担了未见的操作用法。",
    "要区分‘澄’的两个义项，只需问它指水变清还是查清事实。",
    "请为新词‘澄络’提供另一条有来源的上下文。",
    "回答确认‘澄’指查清事实，只更新依赖该回答的两个义项候选。",
    "已有两条独立证据支持该已知用法，不需要继续提问。",
    "已有充分证据，却仍提出义项澄清问题。",
    "证据不足时直接把新用法判成既有含义。",
    "当前问题预算为一，但所需补证成本为二，应停止并标记预算不足。",
    "来源不可访问，不能把缺访问权限混成语义 unknown。",
    "两个新义项仍处于同一竞争组，必须保留歧义。",
    "现象类型、可用证据和应问问题目前都未知。",
    "后续回答确认第一义项，第二义项被 supersede，第三候选保持不变。",
    "只读重验：未见构式候选和它的补证请求仍保持原状态。",
    "语义输入给出两个竞争用法和一个 missing-information obligation。",
)

_EVALUATOR_TEXTS = (
    "日志中首次出现词形‘岚序’，当前词表没有该项。",
    "‘落’在这里可能指下降，也可能指登记入册。",
    "‘给他一看’呈现了此前未登记的构式模式。",
    "熟悉词‘端口’在本句中承担了未见的管理用法。",
    "要区分‘落’的两个义项，只需问它指下降还是登记。",
    "请为新词‘岚序’提供另一条可核来源上下文。",
    "回答确认‘落’指登记，只更新依赖回答的两个义项候选。",
    "已有独立证据支持该用法，不应再提出澄清问题。",
    "证据已充分，却仍要求用户区分两个含义。",
    "信息不足时直接采用新用法的一个分支。",
    "可用预算低于补证成本，应单独标记预算停止。",
    "外部来源无访问权限，应保持 access blocker。",
    "两个候选含义仍同属一个竞争组，不得私选。",
    "新颖度、证据路径和澄清需求都尚未知。",
    "追加回答支持第一义项并替代第二义项，第三候选不变。",
    "只读重验：新构式候选及补证 obligation 没有漂移。",
    "语义输入含两个竞争用法与一个未满足的信息 obligation。",
)

_TEACHER_GENERATION_SURFACE = "你说的这个新用法，是指临时标记还是永久登记？"
_EVALUATOR_GENERATION_SURFACE = "这里的‘端口管理’，是指分配权限还是关闭连接？"


def build_default_open_set_clarification_seed_values(
        ) -> tuple[dict[str, Any], ...]:
    """构造双 owner LC-08 原创样本，不执行新颖度或澄清 runtime。"""
    rows: list[dict[str, Any]] = []
    logical_order = 0
    for owner, texts in (("T", _TEACHER_TEXTS), ("E", _EVALUATOR_TEXTS)):
        label_owner = "teacher" if owner == "T" else "evaluator"
        split = "train" if owner == "T" else "held_out"
        ids = {
            index: (
                f"{owner}-lc08-{index:02d}-"
                f"{scenario[0].lower().replace('_', '-')}")
            for index, scenario in enumerate(_SCENARIOS, start=1)
        }
        for index, scenario in enumerate(_SCENARIOS, start=1):
            logical_order += 1
            (candidate_kind, sample_family, open_set_kind, question_kind,
             evidence_path, blocking_reason, dimension) = scenario
            if owner == "E" and index in _EVALUATOR_EVIDENCE_OVERRIDES:
                evidence_path = _EVALUATOR_EVIDENCE_OVERRIDES[index]
            shared_revision = index in {2, 15}
            revision_version = 2 if index == 15 else 1
            components = _scenario_components(
                owner,
                index,
                candidate_kind,
                open_set_kind,
                question_kind,
                evidence_path,
                blocking_reason,
                revision_version=revision_version,
                shared_revision_identity=shared_revision,
            )
            supersedes = ids[2] if index == 15 else ""
            if candidate_kind == "REVISION":
                receipt = components["clarification_receipt"]
                receipt["supersedes_seed_id"] = supersedes
            retention_anchor = ids[3] if index == 16 else ""
            expected_state = {
                "AMBIGUOUS": "CONFLICT",
                "NEGATIVE": "FALSE",
                "UNKNOWN": "UNKNOWN",
            }.get(sample_family, "TRUE")
            accepted_surfaces: list[str] = []
            if expected_state == "TRUE":
                accepted = texts[index - 1]
                if candidate_kind == "MINIMAL_CLARIFICATION":
                    accepted = (
                        "你说的‘澄’是水变清还是查清事实？"
                        if owner == "T" else
                        "你说的‘落’是下降还是登记入册？")
                elif candidate_kind == "ACTIVE_EVIDENCE_REQUEST":
                    accepted = (
                        "请提供‘澄络’在另一条有来源记录中的用法。"
                        if owner == "T" else
                        "请提供‘岚序’在另一条可核日志中的用法。")
                elif candidate_kind == "GENERATION":
                    accepted = (
                        _TEACHER_GENERATION_SURFACE
                        if owner == "T" else _EVALUATOR_GENERATION_SURFACE)
                accepted_surfaces = [accepted]
            generation_hidden = int(candidate_kind == "GENERATION")
            known_ids = sorted([
                *[item["candidate_id"]
                  for item in components["candidate_branches"]],
                components["missing_information_obligations"][0]["obligation_id"],
                components["clarification_plan"]["plan_id"],
                components["evidence_request"]["request_id"],
            ])
            family_suffix = (
                "REVISION_CHAIN" if index in {2, 15} else candidate_kind)
            objective_keys = {
                "ACTIVE_EVIDENCE_REQUEST": ("CROSS_CONTEXT_CONSISTENCY",),
                "AMBIGUOUS_BRANCH": ("CROSS_CONTEXT_CONSISTENCY",),
                "GENERATION": ("GENERATION_ADOPTION", "GENERATION_FAILURE"),
                "MINIMAL_CLARIFICATION": ("NEXT_DISCOURSE_UNIT",),
                "REVISION": ("CONTROLLED_PERTURBATION",),
                "UNKNOWN": ("CONTROLLED_PERTURBATION",),
            }.get(candidate_kind, ("INTEGER_DESCRIPTION_LENGTH",))
            scope = components.pop("scope")
            baseline = {
                9: "SUFFICIENT_EVIDENCE_OVERQUESTION",
                10: "INSUFFICIENT_EVIDENCE_GUESS",
            }.get(index, "TYPED_OPEN_SET_OBJECTS_PRESENT")
            rows.append({
                "baseline_kind": baseline,
                "candidate_branches": components["candidate_branches"],
                "candidate_kind": candidate_kind,
                "clarification_need": question_kind,
                "clarification_plan": components["clarification_plan"],
                "clarification_receipt": components["clarification_receipt"],
                "evaluation_dimension": dimension,
                "evidence_path": evidence_path,
                "evidence_request": components["evidence_request"],
                "expected_payload": {
                    "accepted": int(expected_state == "TRUE"),
                    "accepted_surfaces": accepted_surfaces,
                    "analysis_key": f"{owner}_LC08_{dimension}",
                    "reason_code": "COURSE_LABEL_ONLY_NOT_RUNTIME_PASS",
                },
                "expected_state": expected_state,
                "family": f"{owner}_LC08_{family_suffix}",
                "generation_constraint": {
                    "branch_preservation_required": generation_hidden,
                    "direction": (
                        "OBLIGATION_TO_QUESTION" if generation_hidden
                        else "SURFACE_TO_OPEN_SET_CANDIDATES"),
                    "evidence_scope_preservation_required": generation_hidden,
                    "input_candidate_ids": known_ids,
                    "novelty_preservation_required": generation_hidden,
                    "obligation_preservation_required": generation_hidden,
                    "output_surface_hidden": generation_hidden,
                    "question_minimality_required": generation_hidden,
                    "uncertainty_preservation_required": generation_hidden,
                },
                "label_owner": label_owner,
                "license_id": LICENSE_ID,
                "logical_order": logical_order,
                "missing_information_obligations": (
                    components["missing_information_obligations"]),
                "novelty_profile": components["novelty_profile"],
                "objective_keys": sorted(objective_keys),
                "observed_text": texts[index - 1],
                "open_set_kind": open_set_kind,
                "perturbation_kind": f"LC08_{candidate_kind}",
                "resource_budget": {
                    "max_candidate_branches": 3,
                    "max_dependency_edges": 6,
                    "max_evidence_requests": 1,
                    "max_obligations": 1,
                    "max_output_units": 160,
                    "max_question_cost": 1,
                },
                "retention_anchor_id": retention_anchor,
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
                "split": split,
                "supersedes_seed_id": supersedes,
                "surface_scope": {
                    "context_scope_key": scope,
                    "genre": "DIALOGUE",
                    "language": "zh",
                    "register": "NEUTRAL",
                    "script": "HAN",
                },
                "template_family": f"{owner}_LC08_TEMPLATE_{candidate_kind}",
            })
    return tuple(rows)


def default_open_set_clarification_sample_bytes() -> bytes:
    """返回规范 LC-08 JSONL，供正式 sample 重建和字节审计。"""
    return b"".join(
        canonical_json_line(value)
        for value in build_default_open_set_clarification_seed_values())


__all__ = [
    "COURSE_MANIFEST_PATH",
    "EVALUATOR_DIMENSIONS",
    "FORMAL_ARTIFACT_RELATIVE_ROOT",
    "PACK_NAME",
    "PAYLOAD_KIND",
    "AuthoredOpenSetClarificationCourseError",
    "AuthoredOpenSetClarificationSeed",
    "OpenSetClarificationPayloadAudit",
    "build_default_open_set_clarification_seed_values",
    "build_open_set_clarification_course_manifest",
    "compile_authored_open_set_clarification_course",
    "default_open_set_clarification_sample_bytes",
    "read_authored_open_set_clarification_seeds",
    "validate_open_set_clarification_payload",
]
