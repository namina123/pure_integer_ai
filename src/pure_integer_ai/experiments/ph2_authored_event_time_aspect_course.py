"""LC-05 Event/State、时间锚、区间与体貌的原创课程。"""
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
PACK_NAME = "AUTHORED_CC0_V1--CC0-1.0--lc05-event-time-aspect-v1"
STAGE = "W-05"
SUBSTAGE = "LC05_EVENT_TIME_ASPECT"
PAYLOAD_KIND = "EventTimeAspectCandidateV1"
COURSE_MANIFEST_ARTIFACT_VERSION = "LC-05-event-time-aspect-course-v1"
COURSE_MANIFEST_PATH = Path(
    "data/ph2/manifests/lc05_event_time_aspect_course_v1.json")
FORMAL_ARTIFACT_RELATIVE_ROOT = (
    "ph2_dataset_artifacts/d02_language_courses_v1")

CANDIDATE_KINDS = (
    "AMBIGUOUS_ANCHOR",
    "ANCHORED_INTERVAL",
    "COMPLETED",
    "DURATIVE",
    "EVENT",
    "GENERATION",
    "HABITUAL_ITERATIVE",
    "IMPLICIT_NOW_BASELINE",
    "NARRATIVE_ORDER",
    "RETENTION",
    "REVISION",
    "STATE",
    "SURFACE_ORDER_BASELINE",
    "UNKNOWN",
)
SEMANTIC_KINDS = ("EVENT", "STATE")
ANCHOR_KINDS = ("CALENDAR", "DEICTIC", "NARRATIVE", "UNKNOWN")
ASPECT_KINDS = (
    "COMPLETED", "DURATIVE", "HABITUAL", "ITERATIVE", "NONE", "UNKNOWN")
RELATION_KINDS = ("BEFORE", "SAME", "UNKNOWN")
BASELINE_KINDS = (
    "IMPLICIT_NOW_ASSUMPTION",
    "SURFACE_ORDER_ONLY",
    "TYPED_TEMPORAL_OBJECTS_PRESENT",
)
EVALUATOR_DIMENSIONS = (
    "COMPLETED_ASPECT",
    "DURATIVE_INTERVAL",
    "EVENT_IDENTITY",
    "HABITUAL_ITERATIVE_ASPECT",
    "IMPLICIT_NOW_REJECT",
    "LOCAL_TIME_REVISION",
    "NARRATIVE_ORDER",
    "RETENTION_REVERIFY",
    "REVERSE_GENERATION_ANCHOR_ASPECT",
    "SAME_SURFACE_DIFFERENT_ANCHOR",
    "STATE_IDENTITY",
    "SURFACE_ORDER_REJECT",
    "TIME_ANCHOR_SCOPE",
    "TIME_UNKNOWN",
)
VERIFIER_NE_CONDITIONS = (
    "ANCHOR_OR_ASPECT_CANDIDATE_MISSING",
    "CAPABILITY_LEARNED_REQUESTED",
    "GENERATION_RESULT_NOT_EXECUTED",
    "NO_EVALUATOR_LABEL",
    "OUT_OF_CONTEXT_SCOPE",
    "SEMANTIC_TRUTH_REQUESTED",
)
RETENTION_PROTOCOLS = (
    "A_TO_LC05_REVERIFY_A",
    "DUMP_RESUME_REQUIRED_AT_RUNTIME",
    "LOCAL_TIME_SUPERSEDE_ONLY",
)
COMBINATION_AXES = (
    "ANCHOR_FAMILY",
    "ASPECT_FAMILY",
    "EVENT_KIND",
    "EVENT_X_ASPECT_X_ANCHOR",
)
ABLATION_KEYS = (
    "DROP_ASPECT_PROFILE",
    "DROP_EVENT_STATE_CANDIDATE",
    "DROP_TEMPORAL_ANCHOR",
    "IMPLICIT_NOW_ASSUMPTION",
    "SURFACE_ORDER_ONLY",
)
PAYLOAD_KEYS = (
    "aspect_profiles",
    "baseline_kind",
    "candidate_kind",
    "event_state_candidates",
    "generation_constraint",
    "intervals",
    "narrative_relations",
    "objective_keys",
    "observed_surface",
    "resource_budget",
    "retention_anchor_id",
    "revision_receipt",
    "sample_family",
    "selection_state",
    "split_identity",
    "surface_scope",
    "temporal_anchors",
)

_SEED_FIELDS = {
    "anchor_family", "aspect_family", "aspect_profiles", "baseline_kind",
    "candidate_kind", "evaluation_dimension", "event_kind",
    "event_state_candidates", "expected_payload", "expected_state", "family",
    "generation_constraint", "intervals", "label_owner", "license_id",
    "logical_order", "narrative_relations", "objective_keys", "observed_text",
    "perturbation_kind", "resource_budget", "retention_anchor_id",
    "revision_receipt", "sample_family", "sample_role", "seed_id", "split",
    "supersedes_seed_id", "surface_scope", "template_family",
    "temporal_anchors",
}
_SCOPE_FIELDS = {
    "context_scope_key", "genre", "language", "register", "script"}
_OBJECT_FIELDS = {
    "candidate_id", "carrier_kind", "context_scope_key", "proposition_key",
    "semantic_kind",
}
_ANCHOR_FIELDS = {
    "anchor_id", "anchor_key", "anchor_kind", "anchor_ordinal",
    "candidate_version", "context_required", "context_scope_key",
}
_INTERVAL_FIELDS = {
    "candidate_version", "closure", "duration_unit", "duration_units",
    "end_anchor_id", "interval_id", "order_known", "start_anchor_id",
}
_ASPECT_FIELDS = {
    "anchor_id", "aspect_kind", "bounded", "candidate_version", "completed",
    "habitual", "interval_id", "iteration_count", "object_id", "profile_id",
}
_RELATION_FIELDS = {
    "candidate_version", "context_scope_key", "from_object_id", "order_index",
    "relation_id", "relation_kind", "to_object_id",
}
_GENERATION_FIELDS = {
    "anchor_preservation_required", "aspect_preservation_required", "direction",
    "event_state_preservation_required", "input_candidate_ids",
    "output_surface_hidden",
}
_REVISION_FIELDS = {
    "candidate_version", "dependency_keys", "raw_observation_preserved",
    "revision_scope", "supersedes_seed_id",
}
_BUDGET_FIELDS = {
    "max_anchors", "max_intervals", "max_objects", "max_output_units",
    "max_profiles", "max_relations",
}
_EXPECTED_FIELDS = {
    "accepted", "accepted_surfaces", "analysis_key", "reason_code"}
_SPLIT_FIELDS = {
    "anchor_family", "aspect_family", "combination_key", "event_kind",
    "isolation_axis",
}
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


class AuthoredEventTimeAspectCourseError(RuntimeError):
    """LC-05 seed、typed payload 或组合 split 不满足冻结合同。"""


def _exact_keys(value: Any, keys: set[str], *, where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise AuthoredEventTimeAspectCourseError(f"{where} 字段集合非法")
    return value


def _text(value: Any, *, where: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or value.strip() != value:
        raise AuthoredEventTimeAspectCourseError(f"{where} 必须是规范文本")
    if not allow_empty and not value:
        raise AuthoredEventTimeAspectCourseError(f"{where} 不能为空")
    return value


def _integer(value: Any, *, where: str, positive: bool = False) -> int:
    if type(value) is not int or value < int(positive):
        qualifier = "正" if positive else "非负"
        raise AuthoredEventTimeAspectCourseError(
            f"{where} 必须是{qualifier}严格整数")
    return value


def _flag(value: Any, *, where: str) -> int:
    if type(value) is not int or value not in {0, 1}:
        raise AuthoredEventTimeAspectCourseError(f"{where} 必须是 0/1")
    return value


def _string_tuple(
        value: Any,
        *,
        where: str,
        sorted_unique: bool = False,
        allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise AuthoredEventTimeAspectCourseError(f"{where} 必须是文本数组")
    result = tuple(_text(item, where=where) for item in value)
    if len(result) != len(set(result)):
        raise AuthoredEventTimeAspectCourseError(f"{where} 不得重复")
    if sorted_unique and tuple(sorted(result)) != result:
        raise AuthoredEventTimeAspectCourseError(f"{where} 必须排序")
    return result


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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
        raise AuthoredEventTimeAspectCourseError("TRUE label 必须给出私有合法表层")
    if expected_state != "TRUE" and (accepted != 0 or surfaces):
        raise AuthoredEventTimeAspectCourseError("非 TRUE label 不得给出采用表层")


@dataclass(frozen=True)
class EventTimeAspectPayloadAudit:
    """返回 typed 时间对象计数、组合键和两类负基线状态。"""

    object_count: int
    anchor_count: int
    interval_count: int
    aspect_profile_count: int
    narrative_relation_count: int
    combination_key: str
    implicit_now_baseline: int
    surface_order_baseline: int


def validate_event_time_aspect_payload(
        payload: CanonicalJsonObject | dict[str, Any],
        ) -> EventTimeAspectPayloadAudit:
    """校验 Event/State 候选、锚/区间/体貌、修正和生成保持。"""
    value = payload.to_value() if isinstance(payload, CanonicalJsonObject) else payload
    raw = _exact_keys(value, set(PAYLOAD_KEYS), where="event-time payload")
    candidate_kind = _text(raw["candidate_kind"], where="candidate_kind")
    if candidate_kind not in CANDIDATE_KINDS:
        raise AuthoredEventTimeAspectCourseError("candidate_kind 未登记")
    baseline = _text(raw["baseline_kind"], where="baseline_kind")
    if baseline not in BASELINE_KINDS:
        raise AuthoredEventTimeAspectCourseError("baseline_kind 未登记")
    if raw["sample_family"] not in SAMPLE_FAMILIES:
        raise AuthoredEventTimeAspectCourseError("sample_family 未登记")
    if raw["selection_state"] != "UNSELECTED":
        raise AuthoredEventTimeAspectCourseError("Observation 不得预选时间候选")
    objective_keys = _string_tuple(
        raw["objective_keys"], where="objective_keys", sorted_unique=True)
    if any(key not in LANGUAGE_OBJECTIVE_KEYS for key in objective_keys):
        raise AuthoredEventTimeAspectCourseError("objective_keys 未登记")

    scope = _exact_keys(raw["surface_scope"], _SCOPE_FIELDS, where="scope")
    if scope["language"] != "zh" or scope["script"] != "HAN":
        raise AuthoredEventTimeAspectCourseError("LC-05 首阶段只冻结 zh/HAN")
    context_scope = _text(scope["context_scope_key"], where="context scope")
    _text(scope["genre"], where="genre")
    if scope["register"] not in {"COLLOQUIAL", "FORMAL", "NEUTRAL"}:
        raise AuthoredEventTimeAspectCourseError("register 未登记")

    observed = _exact_keys(raw["observed_surface"], {
        "append_only", "sha256", "target_hidden", "text",
    }, where="observed surface")
    observed_text = _text(observed["text"], where="observed text")
    if _flag(observed["append_only"], where="append_only") != 1:
        raise AuthoredEventTimeAspectCourseError("raw Observation 必须 append-only")
    if observed["sha256"] != _sha256_text(observed_text):
        raise AuthoredEventTimeAspectCourseError("observed surface SHA-256 漂移")
    target_hidden = _flag(observed["target_hidden"], where="target_hidden")

    budget = _exact_keys(raw["resource_budget"], _BUDGET_FIELDS, where="budget")
    ceilings = {
        key: _integer(budget[key], where=key, positive=True)
        for key in sorted(_BUDGET_FIELDS)
    }
    if len(observed_text) > ceilings["max_output_units"]:
        raise AuthoredEventTimeAspectCourseError("surface 超整数输出预算")

    objects_value = raw["event_state_candidates"]
    if not isinstance(objects_value, list) or not objects_value:
        raise AuthoredEventTimeAspectCourseError("event_state_candidates 不能为空")
    if len(objects_value) > ceilings["max_objects"]:
        raise AuthoredEventTimeAspectCourseError("Event/State 对象超预算")
    objects: dict[str, dict[str, Any]] = {}
    semantic_kinds: set[str] = set()
    for item in objects_value:
        obj = _exact_keys(item, _OBJECT_FIELDS, where="event-state candidate")
        candidate_id = _text(obj["candidate_id"], where="candidate_id")
        if candidate_id in objects:
            raise AuthoredEventTimeAspectCourseError("event-state candidate_id 重复")
        if obj["carrier_kind"] != "EVENT":
            raise AuthoredEventTimeAspectCourseError(
                "Event/State 首阶段必须复用现有 Event carrier")
        if obj["semantic_kind"] not in SEMANTIC_KINDS:
            raise AuthoredEventTimeAspectCourseError("semantic_kind 未登记")
        if obj["context_scope_key"] != context_scope:
            raise AuthoredEventTimeAspectCourseError("Event/State 跨 ContextScope")
        _text(obj["proposition_key"], where="proposition_key")
        objects[candidate_id] = obj
        semantic_kinds.add(obj["semantic_kind"])

    anchors_value = raw["temporal_anchors"]
    if not isinstance(anchors_value, list):
        raise AuthoredEventTimeAspectCourseError("temporal_anchors 必须是数组")
    if len(anchors_value) > ceilings["max_anchors"]:
        raise AuthoredEventTimeAspectCourseError("时间锚超预算")
    anchors: dict[str, dict[str, Any]] = {}
    anchor_ordinals: list[int] = []
    for item in anchors_value:
        anchor = _exact_keys(item, _ANCHOR_FIELDS, where="temporal anchor")
        anchor_id = _text(anchor["anchor_id"], where="anchor_id")
        if anchor_id in anchors:
            raise AuthoredEventTimeAspectCourseError("anchor_id 重复")
        if anchor["anchor_kind"] not in ANCHOR_KINDS:
            raise AuthoredEventTimeAspectCourseError("anchor_kind 未登记")
        if anchor["context_scope_key"] != context_scope:
            raise AuthoredEventTimeAspectCourseError("时间锚跨 ContextScope")
        _text(anchor["anchor_key"], where="anchor_key")
        anchor_ordinals.append(_integer(
            anchor["anchor_ordinal"], where="anchor ordinal", positive=True))
        _integer(anchor["candidate_version"], where="anchor version", positive=True)
        context_required = _flag(
            anchor["context_required"], where="context_required")
        if anchor["anchor_kind"] in {"DEICTIC", "NARRATIVE", "UNKNOWN"}:
            if context_required != 1:
                raise AuthoredEventTimeAspectCourseError(
                    "非日历锚必须声明上下文前置")
        anchors[anchor_id] = anchor
    if anchor_ordinals != list(range(1, len(anchor_ordinals) + 1)):
        raise AuthoredEventTimeAspectCourseError("anchor ordinal 必须连续递增")

    intervals_value = raw["intervals"]
    if not isinstance(intervals_value, list):
        raise AuthoredEventTimeAspectCourseError("intervals 必须是数组")
    if len(intervals_value) > ceilings["max_intervals"]:
        raise AuthoredEventTimeAspectCourseError("区间超预算")
    intervals: dict[str, dict[str, Any]] = {}
    for item in intervals_value:
        interval = _exact_keys(item, _INTERVAL_FIELDS, where="interval")
        interval_id = _text(interval["interval_id"], where="interval_id")
        if interval_id in intervals:
            raise AuthoredEventTimeAspectCourseError("interval_id 重复")
        start = _text(interval["start_anchor_id"], where="start_anchor_id")
        end = _text(interval["end_anchor_id"], where="end_anchor_id")
        if start not in anchors or end not in anchors or start == end:
            raise AuthoredEventTimeAspectCourseError("interval anchor 引用非法")
        if interval["closure"] not in {"CLOSED", "CLOSED_OPEN", "OPEN"}:
            raise AuthoredEventTimeAspectCourseError("interval closure 未登记")
        _text(interval["duration_unit"], where="duration_unit")
        _integer(interval["duration_units"], where="duration_units")
        _flag(interval["order_known"], where="order_known")
        _integer(interval["candidate_version"], where="interval version", positive=True)
        intervals[interval_id] = interval

    profiles_value = raw["aspect_profiles"]
    if not isinstance(profiles_value, list) or not profiles_value:
        raise AuthoredEventTimeAspectCourseError("aspect_profiles 不能为空")
    if len(profiles_value) > ceilings["max_profiles"]:
        raise AuthoredEventTimeAspectCourseError("aspect profile 超预算")
    profiles: dict[str, dict[str, Any]] = {}
    aspect_kinds: set[str] = set()
    for item in profiles_value:
        profile = _exact_keys(item, _ASPECT_FIELDS, where="aspect profile")
        profile_id = _text(profile["profile_id"], where="profile_id")
        if profile_id in profiles:
            raise AuthoredEventTimeAspectCourseError("profile_id 重复")
        object_id = _text(profile["object_id"], where="aspect object_id")
        if object_id not in objects:
            raise AuthoredEventTimeAspectCourseError("aspect 引用未知 Event/State")
        aspect_kind = _text(profile["aspect_kind"], where="aspect_kind")
        if aspect_kind not in ASPECT_KINDS:
            raise AuthoredEventTimeAspectCourseError("aspect_kind 未登记")
        anchor_id = _text(profile["anchor_id"], where="aspect anchor", allow_empty=True)
        interval_id = _text(
            profile["interval_id"], where="aspect interval", allow_empty=True)
        if anchor_id and anchor_id not in anchors:
            raise AuthoredEventTimeAspectCourseError("aspect 引用未知时间锚")
        if interval_id and interval_id not in intervals:
            raise AuthoredEventTimeAspectCourseError("aspect 引用未知区间")
        bounded = _flag(profile["bounded"], where="bounded")
        completed = _flag(profile["completed"], where="completed")
        habitual = _flag(profile["habitual"], where="habitual")
        iteration_count = _integer(
            profile["iteration_count"], where="iteration_count")
        _integer(profile["candidate_version"], where="profile version", positive=True)
        if aspect_kind == "COMPLETED" and not (bounded == completed == 1):
            raise AuthoredEventTimeAspectCourseError("完成体必须有界且完成")
        if aspect_kind == "DURATIVE" and not interval_id:
            raise AuthoredEventTimeAspectCourseError("持续体必须绑定区间")
        if aspect_kind == "HABITUAL" and habitual != 1:
            raise AuthoredEventTimeAspectCourseError("习惯体标志缺失")
        if aspect_kind == "ITERATIVE" and iteration_count < 2:
            raise AuthoredEventTimeAspectCourseError("重复体次数不足")
        profiles[profile_id] = profile
        aspect_kinds.add(aspect_kind)

    relations_value = raw["narrative_relations"]
    if not isinstance(relations_value, list):
        raise AuthoredEventTimeAspectCourseError("narrative_relations 必须是数组")
    if len(relations_value) > ceilings["max_relations"]:
        raise AuthoredEventTimeAspectCourseError("叙事关系超预算")
    relation_ids: set[str] = set()
    relation_orders: list[int] = []
    relation_kinds: set[str] = set()
    for item in relations_value:
        relation = _exact_keys(item, _RELATION_FIELDS, where="narrative relation")
        relation_id = _text(relation["relation_id"], where="relation_id")
        if relation_id in relation_ids:
            raise AuthoredEventTimeAspectCourseError("relation_id 重复")
        relation_ids.add(relation_id)
        first = _text(relation["from_object_id"], where="from_object_id")
        second = _text(relation["to_object_id"], where="to_object_id")
        if first not in objects or second not in objects or first == second:
            raise AuthoredEventTimeAspectCourseError("叙事关系端点非法")
        if relation["relation_kind"] not in RELATION_KINDS:
            raise AuthoredEventTimeAspectCourseError("relation_kind 未登记")
        if relation["context_scope_key"] != context_scope:
            raise AuthoredEventTimeAspectCourseError("叙事关系跨 ContextScope")
        relation_orders.append(_integer(
            relation["order_index"], where="relation order", positive=True))
        _integer(relation["candidate_version"], where="relation version", positive=True)
        relation_kinds.add(relation["relation_kind"])
    if relation_orders != list(range(1, len(relation_orders) + 1)):
        raise AuthoredEventTimeAspectCourseError("relation order 必须连续递增")

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
        raise AuthoredEventTimeAspectCourseError("revision 不得覆盖 raw Observation")
    if candidate_kind == "REVISION":
        if (revision["revision_scope"] != "ANCHOR_ONLY" or not supersedes
                or revision_version < 2 or "TIME_ANCHOR" not in dependencies):
            raise AuthoredEventTimeAspectCourseError("时间修正 receipt 不完整")
    elif (revision["revision_scope"] != "NONE" or supersedes
          or revision_version != 1 or dependencies):
        raise AuthoredEventTimeAspectCourseError("非 revision 不得伪造修正 receipt")

    generation = _exact_keys(
        raw["generation_constraint"], _GENERATION_FIELDS,
        where="generation constraint")
    inputs = _string_tuple(
        generation["input_candidate_ids"], where="generation inputs")
    if any(item not in objects for item in inputs):
        raise AuthoredEventTimeAspectCourseError("generation 引用未知 Event/State")
    hidden = _flag(generation["output_surface_hidden"], where="output hidden")
    for key in (
            "anchor_preservation_required", "aspect_preservation_required",
            "event_state_preservation_required"):
        _flag(generation[key], where=key)
    if candidate_kind == "GENERATION":
        if (generation["direction"] != "SEMANTICS_TO_SURFACE" or hidden != 1
                or target_hidden != 1
                or any(generation[key] != 1 for key in (
                    "anchor_preservation_required", "aspect_preservation_required",
                    "event_state_preservation_required"))):
            raise AuthoredEventTimeAspectCourseError(
                "generation 必须隐藏输出并逐层保持时间语义")
    elif hidden != 0 or target_hidden != 0:
        raise AuthoredEventTimeAspectCourseError("非 generation 不得隐藏目标表层")

    split = _exact_keys(raw["split_identity"], _SPLIT_FIELDS, where="split")
    event_kind = _text(split["event_kind"], where="split event_kind")
    aspect_family = _text(split["aspect_family"], where="split aspect_family")
    anchor_family = _text(split["anchor_family"], where="split anchor_family")
    combination_key = f"{event_kind}::{aspect_family}::{anchor_family}"
    if (split["combination_key"] != combination_key
            or split["isolation_axis"] != "EVENT_X_ASPECT_X_ANCHOR"):
        raise AuthoredEventTimeAspectCourseError("组合 split identity 漂移")

    retention_anchor = _text(
        raw["retention_anchor_id"], where="retention anchor", allow_empty=True)
    if candidate_kind == "RETENTION" and not retention_anchor:
        raise AuthoredEventTimeAspectCourseError("retention 必须绑定更早 anchor")
    if candidate_kind != "RETENTION" and retention_anchor:
        raise AuthoredEventTimeAspectCourseError("非 retention 不得绑定 anchor")
    if candidate_kind == "EVENT" and "EVENT" not in semantic_kinds:
        raise AuthoredEventTimeAspectCourseError("EVENT 候选缺 Event 身份")
    if candidate_kind == "STATE" and "STATE" not in semantic_kinds:
        raise AuthoredEventTimeAspectCourseError("STATE 候选缺 State 分账")
    if candidate_kind == "ANCHORED_INTERVAL" and (not anchors or not intervals):
        raise AuthoredEventTimeAspectCourseError("anchored interval 缺锚或区间")
    if candidate_kind == "DURATIVE" and "DURATIVE" not in aspect_kinds:
        raise AuthoredEventTimeAspectCourseError("持续体候选缺失")
    if candidate_kind == "COMPLETED" and "COMPLETED" not in aspect_kinds:
        raise AuthoredEventTimeAspectCourseError("完成体候选缺失")
    if candidate_kind == "HABITUAL_ITERATIVE" and not {
            "HABITUAL", "ITERATIVE"}.issubset(aspect_kinds):
        raise AuthoredEventTimeAspectCourseError("习惯/重复体未分账")
    if candidate_kind == "NARRATIVE_ORDER" and "BEFORE" not in relation_kinds:
        raise AuthoredEventTimeAspectCourseError("叙事顺序缺 typed BEFORE")
    if candidate_kind == "AMBIGUOUS_ANCHOR":
        selected_anchors = {
            item["anchor_id"] for item in profiles_value if item["anchor_id"]}
        if len(anchors) < 2 or len(selected_anchors) < 2:
            raise AuthoredEventTimeAspectCourseError("同表层必须保留多个时间锚候选")
    implicit_now = int(baseline == "IMPLICIT_NOW_ASSUMPTION")
    surface_order = int(baseline == "SURFACE_ORDER_ONLY")
    if candidate_kind == "IMPLICIT_NOW_BASELINE":
        if not implicit_now or anchors or intervals:
            raise AuthoredEventTimeAspectCourseError("无上下文不得伪造现实 now")
    elif implicit_now:
        raise AuthoredEventTimeAspectCourseError("implicit-now 基线身份错配")
    if candidate_kind == "SURFACE_ORDER_BASELINE":
        if not surface_order or not relations_value or "BEFORE" in relation_kinds:
            raise AuthoredEventTimeAspectCourseError("surface order 不得冒充 PRECEDES")
    elif surface_order:
        raise AuthoredEventTimeAspectCourseError("surface-order 基线身份错配")
    if candidate_kind == "UNKNOWN" and not (
            "UNKNOWN" in aspect_kinds
            or any(item["anchor_kind"] == "UNKNOWN" for item in anchors_value)):
        raise AuthoredEventTimeAspectCourseError("时间 unknown 未显式表示")

    return EventTimeAspectPayloadAudit(
        len(objects), len(anchors), len(intervals), len(profiles),
        len(relations_value), combination_key, implicit_now, surface_order)


@dataclass(frozen=True)
class AuthoredEventTimeAspectSeed:
    """一个 owner、split、时间候选集合和独立 label。"""

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
    event_state_candidates: tuple[CanonicalJsonObject, ...]
    temporal_anchors: tuple[CanonicalJsonObject, ...]
    intervals: tuple[CanonicalJsonObject, ...]
    aspect_profiles: tuple[CanonicalJsonObject, ...]
    narrative_relations: tuple[CanonicalJsonObject, ...]
    generation_constraint: CanonicalJsonObject
    revision_receipt: CanonicalJsonObject
    resource_budget: CanonicalJsonObject
    event_kind: str
    aspect_family: str
    anchor_family: str
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
                ("event_kind", self.event_kind),
                ("aspect_family", self.aspect_family),
                ("anchor_family", self.anchor_family),
                ("perturbation_kind", self.perturbation_kind)):
            _text(value, where=name)
        if self.label_owner not in {"teacher", "evaluator"}:
            raise AuthoredEventTimeAspectCourseError("label_owner 非法")
        if self.split != ("train" if self.label_owner == "teacher" else "held_out"):
            raise AuthoredEventTimeAspectCourseError("label_owner 与 split 不一致")
        if self.sample_family not in SAMPLE_FAMILIES:
            raise AuthoredEventTimeAspectCourseError("sample_family 未登记")
        if self.sample_role not in SAMPLE_ROLES:
            raise AuthoredEventTimeAspectCourseError("sample_role 未登记")
        if self.candidate_kind not in CANDIDATE_KINDS:
            raise AuthoredEventTimeAspectCourseError("candidate_kind 未登记")
        _text(self.observed_text, where="observed_text")
        for name in (
                "surface_scope", "generation_constraint", "revision_receipt",
                "resource_budget"):
            if not isinstance(getattr(self, name), CanonicalJsonObject):
                raise AuthoredEventTimeAspectCourseError(f"{name} 类型非法")
        for name in (
                "event_state_candidates", "temporal_anchors", "intervals",
                "aspect_profiles", "narrative_relations"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or any(
                    not isinstance(item, CanonicalJsonObject) for item in values):
                raise AuthoredEventTimeAspectCourseError(f"{name} 类型非法")
        if not self.event_state_candidates or not self.aspect_profiles:
            raise AuthoredEventTimeAspectCourseError("Event/State 与 aspect 候选不能为空")
        if self.baseline_kind not in BASELINE_KINDS:
            raise AuthoredEventTimeAspectCourseError("baseline_kind 未登记")
        if self.objective_keys != tuple(sorted(set(self.objective_keys))):
            raise AuthoredEventTimeAspectCourseError("objective_keys 必须排序去重")
        if not self.objective_keys or any(
                item not in LANGUAGE_OBJECTIVE_KEYS for item in self.objective_keys):
            raise AuthoredEventTimeAspectCourseError("objective_keys 未登记")
        if self.expected_state not in EXPECTED_STATES:
            raise AuthoredEventTimeAspectCourseError("expected_state 非四态")
        if not isinstance(self.expected_payload, CanonicalJsonObject):
            raise AuthoredEventTimeAspectCourseError("expected_payload 类型非法")
        _validate_expected(self.expected_payload, expected_state=self.expected_state)
        if self.evaluation_dimension not in EVALUATOR_DIMENSIONS:
            raise AuthoredEventTimeAspectCourseError("evaluation_dimension 未登记")
        _text(self.supersedes_seed_id, where="supersedes_seed_id", allow_empty=True)
        _text(self.retention_anchor_id, where="retention_anchor_id", allow_empty=True)
        _integer(self.logical_order, where="logical_order", positive=True)
        if self.license_id != LICENSE_ID:
            raise AuthoredEventTimeAspectCourseError("原创 LC-05 课程必须为 CC0-1.0")
        validate_event_time_aspect_payload(self.observation_payload())

    def observation_payload(self) -> CanonicalJsonObject:
        generation = self.generation_constraint.to_value()
        return CanonicalJsonObject.from_value({
            "aspect_profiles": [item.to_value() for item in self.aspect_profiles],
            "baseline_kind": self.baseline_kind,
            "candidate_kind": self.candidate_kind,
            "event_state_candidates": [
                item.to_value() for item in self.event_state_candidates],
            "generation_constraint": generation,
            "intervals": [item.to_value() for item in self.intervals],
            "narrative_relations": [
                item.to_value() for item in self.narrative_relations],
            "objective_keys": list(self.objective_keys),
            "observed_surface": {
                "append_only": 1,
                "sha256": _sha256_text(self.observed_text),
                "target_hidden": generation["output_surface_hidden"],
                "text": self.observed_text,
            },
            "resource_budget": self.resource_budget.to_value(),
            "retention_anchor_id": self.retention_anchor_id,
            "revision_receipt": self.revision_receipt.to_value(),
            "sample_family": self.sample_family,
            "selection_state": "UNSELECTED",
            "split_identity": {
                "anchor_family": self.anchor_family,
                "aspect_family": self.aspect_family,
                "combination_key": (
                    f"{self.event_kind}::{self.aspect_family}::{self.anchor_family}"),
                "event_kind": self.event_kind,
                "isolation_axis": "EVENT_X_ASPECT_X_ANCHOR",
            },
            "surface_scope": self.surface_scope.to_value(),
            "temporal_anchors": [item.to_value() for item in self.temporal_anchors],
        })

    def compiled_seed(self) -> AuthoredCompiledSeed:
        payload = self.observation_payload()
        audit = validate_event_time_aspect_payload(payload)
        object_ids = tuple(
            item.to_value()["candidate_id"] for item in self.event_state_candidates)
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
            (self.seed_id, object_ids, audit.combination_key),
            (self.observed_text, self.family),
            (
                self.candidate_kind, self.event_kind, self.aspect_family,
                self.anchor_family, audit.object_count, audit.anchor_count,
                audit.interval_count, audit.aspect_profile_count,
                audit.narrative_relation_count,
            ),
            self.evaluation_dimension,
        )


def _canonical_objects(
        values: Any, fields: set[str], *, where: str,
        ) -> tuple[CanonicalJsonObject, ...]:
    if not isinstance(values, list):
        raise AuthoredEventTimeAspectCourseError(f"{where} 必须是数组")
    return tuple(CanonicalJsonObject.from_value(
        _exact_keys(item, fields, where=where)) for item in values)


def _seed_from_value(value: dict[str, Any]) -> AuthoredEventTimeAspectSeed:
    raw = _exact_keys(value, _SEED_FIELDS, where="event-time seed")
    expected = raw["expected_payload"]
    if not isinstance(expected, dict):
        raise AuthoredEventTimeAspectCourseError("expected_payload 必须是对象")
    return AuthoredEventTimeAspectSeed(
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
        _canonical_objects(
            raw["event_state_candidates"], _OBJECT_FIELDS,
            where="event_state_candidates"),
        _canonical_objects(
            raw["temporal_anchors"], _ANCHOR_FIELDS,
            where="temporal_anchors"),
        _canonical_objects(raw["intervals"], _INTERVAL_FIELDS, where="intervals"),
        _canonical_objects(
            raw["aspect_profiles"], _ASPECT_FIELDS, where="aspect_profiles"),
        _canonical_objects(
            raw["narrative_relations"], _RELATION_FIELDS,
            where="narrative_relations"),
        CanonicalJsonObject.from_value(_exact_keys(
            raw["generation_constraint"], _GENERATION_FIELDS,
            where="generation_constraint")),
        CanonicalJsonObject.from_value(_exact_keys(
            raw["revision_receipt"], _REVISION_FIELDS,
            where="revision_receipt")),
        CanonicalJsonObject.from_value(_exact_keys(
            raw["resource_budget"], _BUDGET_FIELDS, where="resource_budget")),
        _text(raw["event_kind"], where="event_kind"),
        _text(raw["aspect_family"], where="aspect_family"),
        _text(raw["anchor_family"], where="anchor_family"),
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


def read_authored_event_time_aspect_seeds(
        path: str | Path) -> tuple[AuthoredEventTimeAspectSeed, ...]:
    """严格读取 JSONL，并核双 owner、修正、retention 和组合 held-out。"""
    try:
        payload = Path(path).read_bytes()
    except OSError as error:
        raise AuthoredEventTimeAspectCourseError("event-time sample 无法读取") from error
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise AuthoredEventTimeAspectCourseError("event-time sample 换行非法")
    seeds: list[AuthoredEventTimeAspectSeed] = []
    try:
        for line in payload.splitlines(keepends=True):
            value = parse_canonical_json_bytes(line[:-1], require_object=True)
            assert isinstance(value, dict)
            if canonical_json_line(value) != line:
                raise AuthoredEventTimeAspectCourseError("event-time seed 非规范 JSON")
            seeds.append(_seed_from_value(value))
    except AuthoredEventTimeAspectCourseError:
        raise
    except Exception as error:
        raise AuthoredEventTimeAspectCourseError("event-time seed 损坏") from error
    identifiers = [item.seed_id for item in seeds]
    orders = [item.logical_order for item in seeds]
    if not seeds or len(identifiers) != len(set(identifiers)):
        raise AuthoredEventTimeAspectCourseError("seed_id 空或重复")
    if orders != sorted(orders) or len(orders) != len(set(orders)):
        raise AuthoredEventTimeAspectCourseError("logical_order 必须严格递增")
    index = {item.seed_id: item for item in seeds}
    for seed in seeds:
        expected_role = _ROLE_BY_FAMILY.get(seed.sample_family)
        if expected_role is not None and seed.sample_role != expected_role:
            raise AuthoredEventTimeAspectCourseError("sample family/role 不一致")
        if seed.sample_family == "GENERATION" and seed.sample_role not in {
                "support", "read_only_probe", "refute"}:
            raise AuthoredEventTimeAspectCourseError("generation sample_role 非法")
        expected_state = _STATE_BY_FAMILY.get(seed.sample_family)
        if expected_state is not None and seed.expected_state != expected_state:
            raise AuthoredEventTimeAspectCourseError("sample family/state 不一致")
        if seed.sample_family == "GENERATION" and seed.expected_state not in {
                "TRUE", "FALSE"}:
            raise AuthoredEventTimeAspectCourseError("generation state 非法")
        if seed.candidate_kind == "REVISION":
            target = index.get(seed.supersedes_seed_id)
            if (target is None or target.logical_order >= seed.logical_order
                    or target.family != seed.family or target.split != seed.split):
                raise AuthoredEventTimeAspectCourseError(
                    "revision 必须 supersede 同族更早 seed")
            old_payload = target.observation_payload().to_value()
            new_payload = seed.observation_payload().to_value()
            old_objects = old_payload["event_state_candidates"]
            new_objects = new_payload["event_state_candidates"]
            if old_objects != new_objects:
                raise AuthoredEventTimeAspectCourseError(
                    "时间 revision 不得替换 Event/State 身份")
            if old_payload["temporal_anchors"] == new_payload["temporal_anchors"]:
                raise AuthoredEventTimeAspectCourseError("时间 revision 必须改动 anchor")
            old_version = old_payload["revision_receipt"]["candidate_version"]
            new_version = new_payload["revision_receipt"]["candidate_version"]
            if new_version <= old_version:
                raise AuthoredEventTimeAspectCourseError("revision version 必须前进")
        elif seed.supersedes_seed_id:
            raise AuthoredEventTimeAspectCourseError("非 revision 不得 supersede")
        if seed.candidate_kind == "RETENTION":
            anchor = index.get(seed.retention_anchor_id)
            if (anchor is None or anchor.logical_order >= seed.logical_order
                    or anchor.split != seed.split):
                raise AuthoredEventTimeAspectCourseError(
                    "retention anchor 必须是同 split 更早 seed")
        elif seed.retention_anchor_id:
            raise AuthoredEventTimeAspectCourseError("非 retention 不得绑定 anchor")

    teacher = tuple(item for item in seeds if item.label_owner == "teacher")
    evaluator = tuple(item for item in seeds if item.label_owner == "evaluator")
    for owner, owner_seeds in (("teacher", teacher), ("evaluator", evaluator)):
        if {item.sample_family for item in owner_seeds} != set(SAMPLE_FAMILIES):
            raise AuthoredEventTimeAspectCourseError(
                f"{owner} 未覆盖七类 sample family")
        if {item.candidate_kind for item in owner_seeds} != set(CANDIDATE_KINDS):
            raise AuthoredEventTimeAspectCourseError(
                f"{owner} 未覆盖完整 Event/State 时间候选族")
        implicit = tuple(item for item in owner_seeds
                         if item.baseline_kind == "IMPLICIT_NOW_ASSUMPTION")
        surface = tuple(item for item in owner_seeds
                        if item.baseline_kind == "SURFACE_ORDER_ONLY")
        if (len(implicit) != 1 or implicit[0].expected_state != "FALSE"
                or len(surface) != 1 or surface[0].expected_state != "FALSE"):
            raise AuthoredEventTimeAspectCourseError(
                f"{owner} 两类时间负基线未冻结")
    if ({item.family for item in teacher} & {item.family for item in evaluator}
            or {item.template_family for item in teacher}
            & {item.template_family for item in evaluator}):
        raise AuthoredEventTimeAspectCourseError("teacher/evaluator family/template 泄漏")
    teacher_events = {item.event_kind for item in teacher}
    teacher_aspects = {item.aspect_family for item in teacher}
    teacher_anchors = {item.anchor_family for item in teacher}
    teacher_triples = {
        (item.event_kind, item.aspect_family, item.anchor_family)
        for item in teacher
    }
    held_out = {
        (item.event_kind, item.aspect_family, item.anchor_family)
        for item in evaluator
        if item.event_kind in teacher_events
        and item.aspect_family in teacher_aspects
        and item.anchor_family in teacher_anchors
        and (item.event_kind, item.aspect_family, item.anchor_family)
        not in teacher_triples
    }
    if len(held_out) < 2:
        raise AuthoredEventTimeAspectCourseError(
            "held-out event×aspect×anchor 组合不足")
    if {item.evaluation_dimension for item in evaluator} != set(
            EVALUATOR_DIMENSIONS):
        raise AuthoredEventTimeAspectCourseError("独立 evaluator 维度未列全")
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
        "authored-event-time-aspect-seed-v1",
        "urn:pure-integer-ai:ph2:authored-event-time-aspect-v1",
        "Pure Integer AI PH2 authored Event/State time and aspect seed",
        "EVENT_TIME_ASPECT_LABEL",
        "lc05-event-time-aspect",
        384,
    )


def compile_authored_event_time_aspect_course(
        sample_path: str | Path,
        release_root: str | Path) -> AuthoredCourseBuild:
    seeds = read_authored_event_time_aspect_seeds(sample_path)
    return publish_authored_course(
        tuple(item.compiled_seed() for item in seeds),
        sample_path,
        release_root,
        _course_spec(),
    )


def build_event_time_aspect_course_manifest(
        sample_path: str | Path,
        build: AuthoredCourseBuild,
        *,
        artifact_relative_root: str = FORMAL_ARTIFACT_RELATIVE_ROOT,
        ) -> CapabilityCourseManifest:
    sample = Path(sample_path)
    seeds = read_authored_event_time_aspect_seeds(sample)
    teacher = tuple(item for item in seeds if item.label_owner == "teacher")
    evaluator = tuple(item for item in seeds if item.label_owner == "evaluator")
    return CapabilityCourseManifest(
        1,
        COURSE_MANIFEST_ARTIFACT_VERSION,
        "COURSE_FROZEN",
        "NOT_STARTED",
        ("LC-05",),
        ("EVENT_TIME_ASPECT",),
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


def _object(
        identifier: str, semantic_kind: str, scope: str, proposition: str,
        ) -> dict[str, Any]:
    return {
        "candidate_id": identifier,
        "carrier_kind": "EVENT",
        "context_scope_key": scope,
        "proposition_key": proposition,
        "semantic_kind": semantic_kind,
    }


def _anchors(
        prefix: str, family: str, scope: str, version: int,
        *,
        count: int) -> list[dict[str, Any]]:
    if family == "UNANCHORED":
        return []
    kinds = [family] * count
    if family == "MIXED":
        kinds = ["CALENDAR", "NARRATIVE"]
    result = []
    for ordinal, kind in enumerate(kinds, start=1):
        result.append({
            "anchor_id": f"{prefix}_ANCHOR_{ordinal}",
            "anchor_key": f"{kind}_POINT_{prefix}_{ordinal}_V{version}",
            "anchor_kind": kind,
            "anchor_ordinal": ordinal,
            "candidate_version": version,
            "context_required": int(kind != "CALENDAR"),
            "context_scope_key": scope,
        })
    return result


def _scenario_components(
        owner: str,
        ordinal: int,
        candidate_kind: str,
        event_kind: str,
        aspect_family: str,
        anchor_family: str,
        *,
        revision_version: int = 1,
        shared_revision_identity: bool = False,
        ) -> dict[str, Any]:
    token = f"{owner}_{ordinal:02d}"
    identity_token = f"{owner}_REVISION" if shared_revision_identity else token
    scope = f"{owner}_CTX_REVISION" if shared_revision_identity else f"{token}_CTX"
    first_id = f"{identity_token}_OBJECT_1"
    semantic_kind = "STATE" if event_kind == "STATE" else "EVENT"
    objects = [_object(
        first_id, semantic_kind, scope, f"{identity_token}_PROPOSITION_1")]
    if candidate_kind in {"NARRATIVE_ORDER", "SURFACE_ORDER_BASELINE"}:
        objects.append(_object(
            f"{identity_token}_OBJECT_2", "EVENT", scope,
            f"{identity_token}_PROPOSITION_2"))

    anchor_count = 0
    if anchor_family not in {"UNANCHORED", "UNKNOWN"}:
        anchor_count = 2 if candidate_kind in {
            "AMBIGUOUS_ANCHOR", "ANCHORED_INTERVAL", "COMPLETED", "DURATIVE",
            "GENERATION", "HABITUAL_ITERATIVE", "NARRATIVE_ORDER", "RETENTION",
            "REVISION", "STATE",
        } else 1
    if anchor_family == "UNKNOWN":
        anchor_count = 1
    anchors = _anchors(
        identity_token, anchor_family, scope, revision_version, count=anchor_count)
    intervals: list[dict[str, Any]] = []
    if len(anchors) >= 2 and candidate_kind != "AMBIGUOUS_ANCHOR":
        intervals.append({
            "candidate_version": revision_version,
            "closure": "CLOSED_OPEN",
            "duration_unit": "CONTEXT_UNIT",
            "duration_units": 2,
            "end_anchor_id": anchors[1]["anchor_id"],
            "interval_id": f"{identity_token}_INTERVAL_1",
            "order_known": 1,
            "start_anchor_id": anchors[0]["anchor_id"],
        })

    aspect_kinds = {
        "COMPLETED": ("COMPLETED",),
        "DURATIVE": ("DURATIVE",),
        "HABITUAL_ITERATIVE": ("HABITUAL", "ITERATIVE"),
        "UNKNOWN": ("UNKNOWN",),
    }.get(aspect_family, ("NONE",))
    profiles = []
    for index, aspect_kind in enumerate(aspect_kinds, start=1):
        profiles.append({
            "anchor_id": (
                anchors[min(index - 1, len(anchors) - 1)]["anchor_id"]
                if anchors else ""),
            "aspect_kind": aspect_kind,
            "bounded": int(aspect_kind == "COMPLETED"),
            "candidate_version": revision_version,
            "completed": int(aspect_kind == "COMPLETED"),
            "habitual": int(aspect_kind == "HABITUAL"),
            "interval_id": intervals[0]["interval_id"] if intervals else "",
            "iteration_count": 3 if aspect_kind == "ITERATIVE" else 0,
            "object_id": first_id,
            "profile_id": f"{identity_token}_ASPECT_{index}",
        })
    if candidate_kind == "AMBIGUOUS_ANCHOR":
        profiles = []
        for index, anchor in enumerate(anchors, start=1):
            profiles.append({
                "anchor_id": anchor["anchor_id"],
                "aspect_kind": "COMPLETED",
                "bounded": 1,
                "candidate_version": revision_version,
                "completed": 1,
                "habitual": 0,
                "interval_id": "",
                "iteration_count": 0,
                "object_id": first_id,
                "profile_id": f"{identity_token}_ASPECT_{index}",
            })

    relations: list[dict[str, Any]] = []
    if candidate_kind in {"NARRATIVE_ORDER", "SURFACE_ORDER_BASELINE"}:
        relations.append({
            "candidate_version": revision_version,
            "context_scope_key": scope,
            "from_object_id": objects[0]["candidate_id"],
            "order_index": 1,
            "relation_id": f"{identity_token}_RELATION_1",
            "relation_kind": (
                "BEFORE" if candidate_kind == "NARRATIVE_ORDER" else "UNKNOWN"),
            "to_object_id": objects[1]["candidate_id"],
        })
    return {
        "aspect_profiles": profiles,
        "event_state_candidates": objects,
        "intervals": intervals,
        "narrative_relations": relations,
        "scope": scope,
        "temporal_anchors": anchors,
    }


_SCENARIOS = (
    ("EVENT", "POSITIVE", "EVENT", "NONE", "CALENDAR", "EVENT_IDENTITY"),
    ("STATE", "POSITIVE", "STATE", "DURATIVE", "NARRATIVE", "STATE_IDENTITY"),
    ("ANCHORED_INTERVAL", "POSITIVE", "EVENT", "NONE", "CALENDAR", "TIME_ANCHOR_SCOPE"),
    ("DURATIVE", "POSITIVE", "EVENT", "DURATIVE", "DEICTIC", "DURATIVE_INTERVAL"),
    ("COMPLETED", "POSITIVE", "EVENT", "COMPLETED", "CALENDAR", "COMPLETED_ASPECT"),
    ("HABITUAL_ITERATIVE", "POSITIVE", "STATE", "HABITUAL_ITERATIVE", "NARRATIVE", "HABITUAL_ITERATIVE_ASPECT"),
    ("NARRATIVE_ORDER", "POSITIVE", "EVENT", "NONE", "NARRATIVE", "NARRATIVE_ORDER"),
    ("AMBIGUOUS_ANCHOR", "AMBIGUOUS", "EVENT", "COMPLETED", "MIXED", "SAME_SURFACE_DIFFERENT_ANCHOR"),
    ("IMPLICIT_NOW_BASELINE", "NEGATIVE", "EVENT", "NONE", "UNANCHORED", "IMPLICIT_NOW_REJECT"),
    ("SURFACE_ORDER_BASELINE", "NEGATIVE", "EVENT", "NONE", "UNANCHORED", "SURFACE_ORDER_REJECT"),
    ("REVISION", "REVISION", "EVENT", "NONE", "CALENDAR", "LOCAL_TIME_REVISION"),
    ("UNKNOWN", "UNKNOWN", "EVENT", "UNKNOWN", "UNKNOWN", "TIME_UNKNOWN"),
    ("RETENTION", "RETENTION", "STATE", "DURATIVE", "NARRATIVE", "RETENTION_REVERIFY"),
    ("GENERATION", "GENERATION", "EVENT", "COMPLETED", "DEICTIC", "REVERSE_GENERATION_ANCHOR_ASPECT"),
)
_EVALUATOR_OVERRIDES = {
    1: ("EVENT", "NONE", "DEICTIC"),
    2: ("STATE", "COMPLETED", "DEICTIC"),
    3: ("EVENT", "NONE", "NARRATIVE"),
    4: ("EVENT", "DURATIVE", "CALENDAR"),
    5: ("STATE", "COMPLETED", "DEICTIC"),
    6: ("EVENT", "HABITUAL_ITERATIVE", "CALENDAR"),
    7: ("EVENT", "NONE", "DEICTIC"),
    13: ("STATE", "DURATIVE", "CALENDAR"),
    14: ("STATE", "DURATIVE", "CALENDAR"),
}
_TEACHER_TEXTS = (
    "钟声在七点响起。",
    "门从清晨到中午一直开着。",
    "展览从周一持续到周三。",
    "他从出发后一直走到天黑。",
    "她在昨天傍晚已经交完表格。",
    "他每周巡检，并在本月重复三次。",
    "灯亮起，随后门打开。",
    "下周他会完成检查。",
    "设备正在运行。",
    "他进入房间。她离开大厅。",
    "后来说明：展览改为周二到周四。",
    "检查何时结束尚不清楚。",
    "门从清晨到中午仍然开着。",
    "语义输入：事件完成登记，锚为会话中的刚才。",
)
_EVALUATOR_TEXTS = (
    "汽笛在会话所指的那一刻鸣响。",
    "信号在约定时段内已经保持稳定。",
    "课程从开场后持续到休息前。",
    "水流从周二持续到周四。",
    "通道在刚才所指时段已经关闭。",
    "巡查在日历周内按惯例重复三次。",
    "警报响起，之后值班员记录事件。",
    "本周报告会提交。",
    "系统正在维护。",
    "甲查看屏幕。乙关闭窗口。",
    "补充信息：课程改为开场次日到休息次日。",
    "恢复时间仍然未知。",
    "信号在日历时段内继续保持稳定。",
    "语义输入：状态持续，锚为周一到周三。",
)
_TEACHER_ACCEPTED = "他刚刚完成了登记。"
_EVALUATOR_ACCEPTED = "设备从周一到周三保持离线。"


def build_default_event_time_aspect_seed_values() -> tuple[dict[str, Any], ...]:
    """构造冻结的双 owner 原创样本；调用方只负责规范落盘。"""
    rows: list[dict[str, Any]] = []
    logical_order = 0
    for owner, texts in (("T", _TEACHER_TEXTS), ("E", _EVALUATOR_TEXTS)):
        label_owner = "teacher" if owner == "T" else "evaluator"
        split = "train" if owner == "T" else "held_out"
        ids = {
            index: f"{owner}-lc05-{index:02d}-{scenario[0].lower().replace('_', '-')}"
            for index, scenario in enumerate(_SCENARIOS, start=1)
        }
        for index, scenario in enumerate(_SCENARIOS, start=1):
            logical_order += 1
            candidate_kind, sample_family, event_kind, aspect_family, anchor_family, dimension = scenario
            if owner == "E" and index in _EVALUATOR_OVERRIDES:
                event_kind, aspect_family, anchor_family = _EVALUATOR_OVERRIDES[index]
            shared_revision = index in {3, 11}
            revision_version = 2 if index == 11 else 1
            components = _scenario_components(
                owner, index, candidate_kind, event_kind, aspect_family,
                anchor_family, revision_version=revision_version,
                shared_revision_identity=shared_revision)
            supersedes = ids[3] if index == 11 else ""
            retention_anchor = ids[2] if index == 13 else ""
            baseline = {
                9: "IMPLICIT_NOW_ASSUMPTION",
                10: "SURFACE_ORDER_ONLY",
            }.get(index, "TYPED_TEMPORAL_OBJECTS_PRESENT")
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
            object_ids = [
                item["candidate_id"] for item in components["event_state_candidates"]]
            generation_hidden = int(candidate_kind == "GENERATION")
            family_suffix = "REVISION_CHAIN" if index in {3, 11} else candidate_kind
            objective_keys = {
                "AMBIGUOUS_ANCHOR": ("CROSS_CONTEXT_CONSISTENCY",),
                "GENERATION": ("GENERATION_ADOPTION", "GENERATION_FAILURE"),
                "NARRATIVE_ORDER": ("NEXT_DISCOURSE_UNIT", "ORDER_RECOVERY"),
                "SURFACE_ORDER_BASELINE": ("ORDER_RECOVERY",),
                "UNKNOWN": ("CONTROLLED_PERTURBATION",),
            }.get(candidate_kind, ("INTEGER_DESCRIPTION_LENGTH",))
            rows.append({
                "anchor_family": anchor_family,
                "aspect_family": aspect_family,
                "aspect_profiles": components["aspect_profiles"],
                "baseline_kind": baseline,
                "candidate_kind": candidate_kind,
                "evaluation_dimension": dimension,
                "event_kind": event_kind,
                "event_state_candidates": components["event_state_candidates"],
                "expected_payload": {
                    "accepted": int(expected_state == "TRUE"),
                    "accepted_surfaces": accepted_surfaces,
                    "analysis_key": f"{owner}_LC05_{dimension}",
                    "reason_code": (
                        "COURSE_LABEL_ONLY_NOT_RUNTIME_PASS"),
                },
                "expected_state": expected_state,
                "family": f"{owner}_LC05_{family_suffix}",
                "generation_constraint": {
                    "anchor_preservation_required": generation_hidden,
                    "aspect_preservation_required": generation_hidden,
                    "direction": (
                        "SEMANTICS_TO_SURFACE" if generation_hidden
                        else "SURFACE_TO_CANDIDATES"),
                    "event_state_preservation_required": generation_hidden,
                    "input_candidate_ids": object_ids,
                    "output_surface_hidden": generation_hidden,
                },
                "intervals": components["intervals"],
                "label_owner": label_owner,
                "license_id": LICENSE_ID,
                "logical_order": logical_order,
                "narrative_relations": components["narrative_relations"],
                "objective_keys": sorted(objective_keys),
                "observed_text": texts[index - 1],
                "perturbation_kind": f"LC05_{candidate_kind}",
                "resource_budget": {
                    "max_anchors": 2,
                    "max_intervals": 1,
                    "max_objects": 2,
                    "max_output_units": 64,
                    "max_profiles": 2,
                    "max_relations": 1,
                },
                "retention_anchor_id": retention_anchor,
                "revision_receipt": {
                    "candidate_version": revision_version,
                    "dependency_keys": ["TIME_ANCHOR"] if index == 11 else [],
                    "raw_observation_preserved": 1,
                    "revision_scope": "ANCHOR_ONLY" if index == 11 else "NONE",
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
                "seed_id": ids[index],
                "split": split,
                "supersedes_seed_id": supersedes,
                "surface_scope": {
                    "context_scope_key": scope,
                    "genre": "DECLARATIVE",
                    "language": "zh",
                    "register": "NEUTRAL",
                    "script": "HAN",
                },
                "template_family": f"{owner}_LC05_TEMPLATE_{candidate_kind}",
                "temporal_anchors": components["temporal_anchors"],
            })
    return tuple(rows)


def default_event_time_aspect_sample_bytes() -> bytes:
    """返回规范 JSONL，供正式 sample 的可重复生成和审计。"""
    return b"".join(
        canonical_json_line(value)
        for value in build_default_event_time_aspect_seed_values())


__all__ = [
    "COURSE_MANIFEST_PATH",
    "EVALUATOR_DIMENSIONS",
    "FORMAL_ARTIFACT_RELATIVE_ROOT",
    "PACK_NAME",
    "PAYLOAD_KIND",
    "AuthoredEventTimeAspectCourseError",
    "AuthoredEventTimeAspectSeed",
    "EventTimeAspectPayloadAudit",
    "build_default_event_time_aspect_seed_values",
    "build_event_time_aspect_course_manifest",
    "compile_authored_event_time_aspect_course",
    "default_event_time_aspect_sample_bytes",
    "read_authored_event_time_aspect_seeds",
    "validate_event_time_aspect_payload",
]
