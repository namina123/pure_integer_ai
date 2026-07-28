"""LC-03 多词表达与构式身份的原创 CC0 课程编译器。"""
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
PACK_NAME = "AUTHORED_CC0_V1--CC0-1.0--lc03-construction-v1"
STAGE = "W-03"
SUBSTAGE = "LC03_MULTIWORD_CONSTRUCTION"
PAYLOAD_KIND = "ConstructionCandidateV1"
COURSE_MANIFEST_ARTIFACT_VERSION = "LC-03-construction-course-v1"
COURSE_MANIFEST_PATH = Path(
    "data/ph2/manifests/lc03_construction_course_v1.json")
FORMAL_ARTIFACT_RELATIVE_ROOT = (
    "ph2_dataset_artifacts/d02_language_courses_v1")

CANDIDATE_KINDS = (
    "AMBIGUOUS",
    "ANTI_LITERAL",
    "COMPOSITIONAL",
    "DISCONTINUOUS",
    "GENERATION",
    "PARTIAL_LEXICALIZED",
    "RETENTION",
    "REVISION",
    "UNKNOWN",
    "WHOLE_LEXICALIZED",
)
LEXICALIZATION_STATES = (
    "ABSENT",
    "COMPOSITIONAL",
    "PARTIAL_LEXICALIZED",
    "WHOLE_LEXICALIZED",
)
SPAN_KINDS = (
    "CONSTRUCTION",
    "FILLER",
    "FIXED_COMPONENT",
    "VARIABLE_SLOT",
)
REGISTER_KEYS = ("COLLOQUIAL", "FORMAL", "NEUTRAL")
BASELINE_KINDS = (
    "CONSTRUCTION_OBJECT_PRESENT",
    "LITERAL_TOKEN_SUM_ONLY",
)
EVALUATOR_DIMENSIONS = (
    "ANTI_LITERAL_BASELINE",
    "CONSTRUCTION_OBJECT_ABLATION",
    "DISCONTINUOUS_SPAN",
    "EVENT_CORE_MAPPING",
    "FIXED_VARIABLE_SLOT",
    "HELD_OUT_FILLER_CONSTRUCTION",
    "LEXICALIZATION_IDENTITY",
    "REGISTER_SCOPE",
    "RETENTION_REVERIFY",
    "REVERSE_GENERATION",
    "SAME_PROPOSITION_DIFFERENT_CONSTRUCTION",
    "SAME_SURFACE_DIFFERENT_CONSTRUCTION",
)
VERIFIER_NE_CONDITIONS = (
    "CAPABILITY_LEARNED_REQUESTED",
    "CONSTRUCTION_OBJECT_MISSING",
    "GENERATION_RESULT_NOT_EXECUTED",
    "NO_EVALUATOR_LABEL",
    "OUT_OF_REGISTER_SCOPE",
    "SEMANTIC_TRUTH_REQUESTED",
)
RETENTION_PROTOCOLS = (
    "A_TO_LC03_REVERIFY_A",
    "CONSTRUCTION_SUPERSEDE_LOCAL_ONLY",
    "DUMP_RESUME_REQUIRED_AT_RUNTIME",
)
COMBINATION_AXES = (
    "CONSTRUCTION_FAMILY",
    "FILLER_FAMILY",
    "FILLER_X_CONSTRUCTION",
)
ABLATION_KEYS = (
    "LITERAL_TOKEN_SUM_ONLY",
    "REMOVE_CONSTRUCTION_OBJECT",
    "REMOVE_EVENT_CORE_MAPPING",
    "REMOVE_FIXED_COMPONENT",
    "REMOVE_VARIABLE_SLOT",
)
PAYLOAD_KEYS = (
    "baseline_kind",
    "candidate_group",
    "candidate_id",
    "candidate_kind",
    "construction_identity",
    "event_core_mapping",
    "generation_constraint",
    "objective_keys",
    "observed_surface",
    "retention_anchor_id",
    "sample_family",
    "selection_state",
    "slots",
    "spans",
    "split_identity",
    "surface_scope",
)

_SEED_FIELDS = {
    "baseline_kind", "candidate_group", "candidate_id", "candidate_kind",
    "construction_family", "construction_identity", "evaluation_dimension",
    "event_core_mapping", "expected_payload", "expected_state", "family",
    "filler_family", "generation_constraint", "label_owner", "license_id",
    "logical_order", "objective_keys", "observed_text", "perturbation_kind",
    "proposition_group", "retention_anchor_id", "sample_family",
    "sample_role", "seed_id", "slots", "spans", "split",
    "supersedes_seed_id", "surface_scope", "template_family",
}
_CONSTRUCTION_FIELDS = {
    "construction_key", "lexicalization_state", "present",
}
_SCOPE_FIELDS = {"genre", "language", "register", "script"}
_SPAN_FIELDS = {"fixed", "members", "span_id", "span_kind"}
_SLOT_FIELDS = {
    "filler_span_id", "fixed", "optional", "order_index", "role_key",
    "slot_id",
}
_EVENT_FIELDS = {"event_core_key", "mapping_state", "role_bindings"}
_EVENT_ROLE_FIELDS = {"event_role", "slot_id"}
_GENERATION_FIELDS = {
    "direction", "input_slot_ids", "output_surface_hidden",
    "requires_construction_object",
}
_EXPECTED_FIELDS = {
    "accepted", "accepted_surfaces", "analysis_key", "reason_code",
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


class AuthoredConstructionCourseError(RuntimeError):
    """LC-03 seed、构式 payload 或组合 split 不满足冻结合同。"""


def _exact_keys(value: Any, keys: set[str], *, where: str) -> dict[str, Any]:
    """要求 JSON 对象字段精确一致。"""
    if not isinstance(value, dict) or set(value) != keys:
        raise AuthoredConstructionCourseError(f"{where} 字段集合非法")
    return value


def _text(value: Any, *, where: str, allow_empty: bool = False) -> str:
    """要求文本无首尾空白；仅显式位置允许空字符串。"""
    if not isinstance(value, str) or value.strip() != value:
        raise AuthoredConstructionCourseError(f"{where} 必须是规范文本")
    if not allow_empty and not value:
        raise AuthoredConstructionCourseError(f"{where} 不能为空")
    return value


def _integer(value: Any, *, where: str) -> int:
    """要求字段为非负严格整数。"""
    if type(value) is not int or value < 0:
        raise AuthoredConstructionCourseError(f"{where} 必须是非负严格整数")
    return value


def _flag(value: Any, *, where: str) -> int:
    """要求字段为严格 0/1。"""
    if type(value) is not int or value not in {0, 1}:
        raise AuthoredConstructionCourseError(f"{where} 必须是 0/1")
    return value


def _string_tuple(
        value: Any,
        *,
        where: str,
        sorted_unique: bool = False,
        allow_empty: bool = False) -> tuple[str, ...]:
    """恢复 JSON 文本数组并校验重复和顺序。"""
    if not isinstance(value, list) or (not value and not allow_empty):
        raise AuthoredConstructionCourseError(f"{where} 必须是文本数组")
    result = tuple(_text(item, where=where) for item in value)
    if len(result) != len(set(result)):
        raise AuthoredConstructionCourseError(f"{where} 不得重复")
    if sorted_unique and tuple(sorted(result)) != result:
        raise AuthoredConstructionCourseError(f"{where} 必须排序")
    return result


def _sha256_text(value: str) -> str:
    """计算 UTF-8 文本 SHA-256。"""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_expected(
        payload: CanonicalJsonObject,
        *,
        expected_state: str) -> None:
    """校验私有 label 的采用状态与多个合法表层。"""
    raw = _exact_keys(payload.to_value(), _EXPECTED_FIELDS, where="expected payload")
    accepted = _flag(raw["accepted"], where="expected accepted")
    surfaces = _string_tuple(
        raw["accepted_surfaces"], where="accepted_surfaces", allow_empty=True)
    _text(raw["analysis_key"], where="analysis_key")
    _text(raw["reason_code"], where="reason_code")
    if expected_state == "TRUE" and (accepted != 1 or not surfaces):
        raise AuthoredConstructionCourseError("TRUE label 必须给出私有合法表层")
    if expected_state != "TRUE" and (accepted != 0 or surfaces):
        raise AuthoredConstructionCourseError("非 TRUE label 不得给出采用表层")


@dataclass(frozen=True)
class ConstructionPayloadAudit:
    """返回构式对象、连续性、组合轴和生成隐藏事实。"""

    construction_present: int
    discontinuous_span_count: int
    combination_key: str
    target_hidden: int
    literal_only: int


def validate_construction_payload(
        payload: CanonicalJsonObject | dict[str, Any]) -> ConstructionPayloadAudit:
    """校验构式身份、span、slot、event core、生成和 held-out 组合轴。"""
    value = payload.to_value() if isinstance(payload, CanonicalJsonObject) else payload
    raw = _exact_keys(value, set(PAYLOAD_KEYS), where="construction payload")
    candidate_kind = _text(raw["candidate_kind"], where="candidate_kind")
    if candidate_kind not in CANDIDATE_KINDS:
        raise AuthoredConstructionCourseError("candidate_kind 未登记")
    baseline_kind = _text(raw["baseline_kind"], where="baseline_kind")
    if baseline_kind not in BASELINE_KINDS:
        raise AuthoredConstructionCourseError("baseline_kind 未登记")
    _text(raw["candidate_group"], where="candidate_group")
    _text(raw["candidate_id"], where="candidate_id")
    if raw["selection_state"] != "UNSELECTED":
        raise AuthoredConstructionCourseError("Observation 不得私选构式候选")
    if raw["sample_family"] not in SAMPLE_FAMILIES:
        raise AuthoredConstructionCourseError("sample_family 未登记")

    scope = _exact_keys(raw["surface_scope"], _SCOPE_FIELDS, where="surface_scope")
    if scope["language"] != "zh" or scope["script"] != "HAN":
        raise AuthoredConstructionCourseError("LC-03 首阶段只冻结 zh/HAN")
    if scope["register"] not in REGISTER_KEYS:
        raise AuthoredConstructionCourseError("register 未登记")
    _text(scope["genre"], where="genre")

    observed = _exact_keys(raw["observed_surface"], {
        "append_only", "sha256", "target_hidden", "text",
    }, where="observed_surface")
    text = _text(observed["text"], where="observed text")
    if _flag(observed["append_only"], where="observed append_only") != 1:
        raise AuthoredConstructionCourseError("observed surface 必须 append-only")
    if observed["sha256"] != _sha256_text(text):
        raise AuthoredConstructionCourseError("observed surface SHA-256 漂移")
    target_hidden = _flag(observed["target_hidden"], where="target_hidden")

    identity = _exact_keys(
        raw["construction_identity"], _CONSTRUCTION_FIELDS,
        where="construction_identity")
    present = _flag(identity["present"], where="construction present")
    construction_key = _text(
        identity["construction_key"], where="construction_key",
        allow_empty=True)
    lexicalization = _text(
        identity["lexicalization_state"], where="lexicalization_state")
    if lexicalization not in LEXICALIZATION_STATES:
        raise AuthoredConstructionCourseError("lexicalization_state 未登记")
    if present and (not construction_key or lexicalization == "ABSENT"):
        raise AuthoredConstructionCourseError("构式对象缺身份或词汇化状态")
    if not present and (construction_key or lexicalization != "ABSENT"):
        raise AuthoredConstructionCourseError("缺失构式对象不得伪造身份")
    lexicalization_by_kind = {
        "COMPOSITIONAL": "COMPOSITIONAL",
        "PARTIAL_LEXICALIZED": "PARTIAL_LEXICALIZED",
        "WHOLE_LEXICALIZED": "WHOLE_LEXICALIZED",
    }
    expected_lexicalization = lexicalization_by_kind.get(candidate_kind)
    if expected_lexicalization is not None and lexicalization != expected_lexicalization:
        raise AuthoredConstructionCourseError("candidate kind 与词汇化状态不一致")

    spans_value = raw["spans"]
    if not isinstance(spans_value, list) or not spans_value:
        raise AuthoredConstructionCourseError("spans 不能为空")
    spans: dict[str, dict[str, Any]] = {}
    discontinuous = 0
    for item in spans_value:
        span = _exact_keys(item, _SPAN_FIELDS, where="construction span")
        span_id = _text(span["span_id"], where="span_id")
        if span_id in spans:
            raise AuthoredConstructionCourseError("span_id 重复")
        if span["span_kind"] not in SPAN_KINDS:
            raise AuthoredConstructionCourseError("span_kind 未登记")
        _flag(span["fixed"], where="span fixed")
        members = span["members"]
        if not isinstance(members, list) or not members:
            raise AuthoredConstructionCourseError("span members 不能为空")
        previous_end = -1
        for member in members:
            if not isinstance(member, list) or len(member) != 2:
                raise AuthoredConstructionCourseError("span member 必须是二元区间")
            start = _integer(member[0], where="span start")
            end = _integer(member[1], where="span end")
            if not (previous_end <= start < end <= len(text)):
                raise AuthoredConstructionCourseError("span member 越界或重叠")
            previous_end = end
        if len(members) > 1:
            discontinuous += 1
        spans[span_id] = span
    if candidate_kind == "DISCONTINUOUS" and discontinuous == 0:
        raise AuthoredConstructionCourseError("非连续构式缺多成员 span")

    slots_value = raw["slots"]
    if not isinstance(slots_value, list):
        raise AuthoredConstructionCourseError("slots 必须是数组")
    slots: dict[str, dict[str, Any]] = {}
    orders: list[int] = []
    for item in slots_value:
        slot = _exact_keys(item, _SLOT_FIELDS, where="construction slot")
        slot_id = _text(slot["slot_id"], where="slot_id")
        if slot_id in slots:
            raise AuthoredConstructionCourseError("slot_id 重复")
        _text(slot["role_key"], where="role_key")
        filler = _text(slot["filler_span_id"], where="filler_span_id")
        if filler not in spans:
            raise AuthoredConstructionCourseError("slot 引用未知 filler span")
        _flag(slot["fixed"], where="slot fixed")
        _flag(slot["optional"], where="slot optional")
        order = _integer(slot["order_index"], where="slot order")
        if order <= 0:
            raise AuthoredConstructionCourseError("slot order 必须为正")
        orders.append(order)
        slots[slot_id] = slot
    if orders != list(range(1, len(orders) + 1)):
        raise AuthoredConstructionCourseError("slot order 必须连续递增")

    event = _exact_keys(
        raw["event_core_mapping"], _EVENT_FIELDS, where="event_core_mapping")
    event_key = _text(
        event["event_core_key"], where="event_core_key", allow_empty=True)
    mapping_state = _text(event["mapping_state"], where="mapping_state")
    roles_value = event["role_bindings"]
    if not isinstance(roles_value, list):
        raise AuthoredConstructionCourseError("event role_bindings 必须是数组")
    role_slots: set[str] = set()
    for item in roles_value:
        role = _exact_keys(item, _EVENT_ROLE_FIELDS, where="event role")
        slot_id = _text(role["slot_id"], where="event slot_id")
        if slot_id not in slots or slot_id in role_slots:
            raise AuthoredConstructionCourseError("event role 引用未知或重复 slot")
        _text(role["event_role"], where="event_role")
        role_slots.add(slot_id)
    if present:
        if not slots or not event_key or mapping_state != "CANDIDATE" or not role_slots:
            raise AuthoredConstructionCourseError("构式对象缺 slot/event core 候选")
    elif slots or event_key or mapping_state != "ABSENT" or role_slots:
        raise AuthoredConstructionCourseError("literal baseline 不得伪造构式映射")

    generation = _exact_keys(
        raw["generation_constraint"], _GENERATION_FIELDS,
        where="generation_constraint")
    direction = _text(generation["direction"], where="generation direction")
    if direction not in {"BIDIRECTIONAL", "GENERATION", "UNDERSTANDING"}:
        raise AuthoredConstructionCourseError("generation direction 未登记")
    inputs = _string_tuple(
        generation["input_slot_ids"], where="input_slot_ids",
        allow_empty=not present)
    if any(item not in slots for item in inputs):
        raise AuthoredConstructionCourseError("generation input 引用未知 slot")
    hidden = _flag(
        generation["output_surface_hidden"], where="output_surface_hidden")
    if hidden != target_hidden:
        raise AuthoredConstructionCourseError("generation hidden 标志不一致")
    required = _flag(
        generation["requires_construction_object"],
        where="requires_construction_object")
    if required != present:
        raise AuthoredConstructionCourseError("生成约束与构式对象存在性不一致")
    if candidate_kind == "GENERATION":
        if direction != "GENERATION" or hidden != 1:
            raise AuthoredConstructionCourseError("反向生成必须隐藏目标表层")
    elif hidden:
        raise AuthoredConstructionCourseError("非生成样本不得隐藏目标")

    literal_only = int(baseline_kind == "LITERAL_TOKEN_SUM_ONLY")
    if literal_only != int(not present):
        raise AuthoredConstructionCourseError("anti-literal baseline 与构式对象不一致")
    if literal_only and candidate_kind != "ANTI_LITERAL":
        raise AuthoredConstructionCourseError("literal baseline 必须是 ANTI_LITERAL")

    split_identity = _exact_keys(raw["split_identity"], {
        "combination_key", "construction_family", "filler_family",
        "isolation_axis", "proposition_group",
    }, where="split_identity")
    filler_family = _text(split_identity["filler_family"], where="filler_family")
    construction_family = _text(
        split_identity["construction_family"], where="construction_family")
    combination_key = _text(
        split_identity["combination_key"], where="combination_key")
    if combination_key != f"{filler_family}::{construction_family}":
        raise AuthoredConstructionCourseError("filler×construction 组合键漂移")
    if split_identity["isolation_axis"] != "FILLER_X_CONSTRUCTION":
        raise AuthoredConstructionCourseError("组合 held-out 轴未冻结")
    _text(split_identity["proposition_group"], where="proposition_group")

    objectives = tuple(raw["objective_keys"])
    if objectives != tuple(sorted(set(objectives))):
        raise AuthoredConstructionCourseError("objective_keys 必须排序去重")
    if not objectives or any(item not in LANGUAGE_OBJECTIVE_KEYS for item in objectives):
        raise AuthoredConstructionCourseError("objective_keys 未登记")
    anchor = _text(
        raw["retention_anchor_id"], where="retention_anchor_id",
        allow_empty=True)
    if raw["sample_family"] == "RETENTION" and not anchor:
        raise AuthoredConstructionCourseError("retention 必须绑定旧 anchor")
    if raw["sample_family"] != "RETENTION" and anchor:
        raise AuthoredConstructionCourseError("非 retention 不得绑定 anchor")
    return ConstructionPayloadAudit(
        present, discontinuous, combination_key, target_hidden, literal_only)


@dataclass(frozen=True)
class AuthoredConstructionSeed:
    """一个保留一等构式候选、span、slot 和 event core 的 LC-03 seed。"""

    seed_id: str
    family: str
    template_family: str
    label_owner: str
    split: str
    sample_family: str
    sample_role: str
    candidate_id: str
    candidate_group: str
    candidate_kind: str
    observed_text: str
    surface_scope: CanonicalJsonObject
    construction_identity: CanonicalJsonObject
    spans: tuple[CanonicalJsonObject, ...]
    slots: tuple[CanonicalJsonObject, ...]
    event_core_mapping: CanonicalJsonObject
    generation_constraint: CanonicalJsonObject
    filler_family: str
    construction_family: str
    proposition_group: str
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
        """校验 owner、四态、许可、revision/retention 前置字段和 payload。"""
        for name, value in (
                ("seed_id", self.seed_id), ("family", self.family),
                ("template_family", self.template_family),
                ("candidate_id", self.candidate_id),
                ("candidate_group", self.candidate_group),
                ("filler_family", self.filler_family),
                ("construction_family", self.construction_family),
                ("proposition_group", self.proposition_group),
                ("perturbation_kind", self.perturbation_kind)):
            _text(value, where=name)
        if self.label_owner not in {"teacher", "evaluator"}:
            raise AuthoredConstructionCourseError("label_owner 非法")
        expected_split = "train" if self.label_owner == "teacher" else "held_out"
        if self.split != expected_split:
            raise AuthoredConstructionCourseError("label_owner 与 split 不一致")
        if self.sample_family not in SAMPLE_FAMILIES:
            raise AuthoredConstructionCourseError("sample_family 未登记")
        if self.sample_role not in SAMPLE_ROLES:
            raise AuthoredConstructionCourseError("sample_role 未登记")
        if self.candidate_kind not in CANDIDATE_KINDS:
            raise AuthoredConstructionCourseError("candidate_kind 未登记")
        _text(self.observed_text, where="observed_text")
        for name in (
                "surface_scope", "construction_identity", "event_core_mapping",
                "generation_constraint"):
            if not isinstance(getattr(self, name), CanonicalJsonObject):
                raise AuthoredConstructionCourseError(f"{name} 类型非法")
        for name in ("spans", "slots"):
            values = getattr(self, name)
            if (not isinstance(values, tuple)
                    or any(not isinstance(item, CanonicalJsonObject)
                           for item in values)):
                raise AuthoredConstructionCourseError(f"{name} 类型非法")
        if not self.spans:
            raise AuthoredConstructionCourseError("spans 不能为空")
        if self.baseline_kind not in BASELINE_KINDS:
            raise AuthoredConstructionCourseError("baseline_kind 未登记")
        if self.objective_keys != tuple(sorted(set(self.objective_keys))):
            raise AuthoredConstructionCourseError("objective_keys 必须排序去重")
        if not self.objective_keys or any(
                item not in LANGUAGE_OBJECTIVE_KEYS for item in self.objective_keys):
            raise AuthoredConstructionCourseError("objective_keys 未登记")
        if self.expected_state not in EXPECTED_STATES:
            raise AuthoredConstructionCourseError("expected_state 非四态")
        if not isinstance(self.expected_payload, CanonicalJsonObject):
            raise AuthoredConstructionCourseError("expected_payload 类型非法")
        _validate_expected(self.expected_payload, expected_state=self.expected_state)
        if self.evaluation_dimension not in EVALUATOR_DIMENSIONS:
            raise AuthoredConstructionCourseError("evaluation_dimension 未登记")
        _text(
            self.supersedes_seed_id, where="supersedes_seed_id",
            allow_empty=True)
        _text(
            self.retention_anchor_id, where="retention_anchor_id",
            allow_empty=True)
        if type(self.logical_order) is not int or self.logical_order <= 0:
            raise AuthoredConstructionCourseError("logical_order 必须为正严格整数")
        if self.license_id != LICENSE_ID:
            raise AuthoredConstructionCourseError("原创构式课程必须为 CC0-1.0")
        validate_construction_payload(self.observation_payload())

    def observation_payload(self) -> CanonicalJsonObject:
        """构造学生可见构式候选，不包含 expected 或合法生成表层。"""
        generation = self.generation_constraint.to_value()
        return CanonicalJsonObject.from_value({
            "baseline_kind": self.baseline_kind,
            "candidate_group": self.candidate_group,
            "candidate_id": self.candidate_id,
            "candidate_kind": self.candidate_kind,
            "construction_identity": self.construction_identity.to_value(),
            "event_core_mapping": self.event_core_mapping.to_value(),
            "generation_constraint": generation,
            "objective_keys": list(self.objective_keys),
            "observed_surface": {
                "append_only": 1,
                "sha256": _sha256_text(self.observed_text),
                "target_hidden": generation["output_surface_hidden"],
                "text": self.observed_text,
            },
            "retention_anchor_id": self.retention_anchor_id,
            "sample_family": self.sample_family,
            "selection_state": "UNSELECTED",
            "slots": [item.to_value() for item in self.slots],
            "spans": [item.to_value() for item in self.spans],
            "split_identity": {
                "combination_key": (
                    f"{self.filler_family}::{self.construction_family}"),
                "construction_family": self.construction_family,
                "filler_family": self.filler_family,
                "isolation_axis": "FILLER_X_CONSTRUCTION",
                "proposition_group": self.proposition_group,
            },
            "surface_scope": self.surface_scope.to_value(),
        })

    def compiled_seed(self) -> AuthoredCompiledSeed:
        """映射到既有四 owner Dataset publisher。"""
        payload = self.observation_payload()
        audit = validate_construction_payload(payload)
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
            (
                self.candidate_group, self.candidate_id, self.observed_text,
                audit.combination_key,
            ),
            (self.candidate_group, self.observed_text),
            (
                self.candidate_kind, self.sample_family,
                self.construction_family, self.proposition_group,
                audit.construction_present, audit.discontinuous_span_count,
            ),
            self.evaluation_dimension,
        )


def _seed_from_value(value: dict[str, Any]) -> AuthoredConstructionSeed:
    """从精确 JSON object 恢复一个构式课程 seed。"""
    raw = _exact_keys(value, _SEED_FIELDS, where="construction seed")
    spans = raw["spans"]
    slots = raw["slots"]
    if not isinstance(spans, list) or not spans:
        raise AuthoredConstructionCourseError("spans 不能为空")
    if not isinstance(slots, list):
        raise AuthoredConstructionCourseError("slots 必须是数组")
    expected = raw["expected_payload"]
    if not isinstance(expected, dict):
        raise AuthoredConstructionCourseError("expected_payload 必须是对象")
    return AuthoredConstructionSeed(
        _text(raw["seed_id"], where="seed_id"),
        _text(raw["family"], where="family"),
        _text(raw["template_family"], where="template_family"),
        _text(raw["label_owner"], where="label_owner"),
        _text(raw["split"], where="split"),
        _text(raw["sample_family"], where="sample_family"),
        _text(raw["sample_role"], where="sample_role"),
        _text(raw["candidate_id"], where="candidate_id"),
        _text(raw["candidate_group"], where="candidate_group"),
        _text(raw["candidate_kind"], where="candidate_kind"),
        _text(raw["observed_text"], where="observed_text"),
        CanonicalJsonObject.from_value(_exact_keys(
            raw["surface_scope"], _SCOPE_FIELDS, where="surface_scope")),
        CanonicalJsonObject.from_value(_exact_keys(
            raw["construction_identity"], _CONSTRUCTION_FIELDS,
            where="construction_identity")),
        tuple(CanonicalJsonObject.from_value(_exact_keys(
            item, _SPAN_FIELDS, where="span")) for item in spans),
        tuple(CanonicalJsonObject.from_value(_exact_keys(
            item, _SLOT_FIELDS, where="slot")) for item in slots),
        CanonicalJsonObject.from_value(_exact_keys(
            raw["event_core_mapping"], _EVENT_FIELDS,
            where="event_core_mapping")),
        CanonicalJsonObject.from_value(_exact_keys(
            raw["generation_constraint"], _GENERATION_FIELDS,
            where="generation_constraint")),
        _text(raw["filler_family"], where="filler_family"),
        _text(raw["construction_family"], where="construction_family"),
        _text(raw["proposition_group"], where="proposition_group"),
        _text(raw["baseline_kind"], where="baseline_kind"),
        _string_tuple(
            raw["objective_keys"], where="objective_keys", sorted_unique=True),
        _text(raw["expected_state"], where="expected_state"),
        CanonicalJsonObject.from_value(_exact_keys(
            expected, _EXPECTED_FIELDS, where="expected_payload")),
        _text(raw["evaluation_dimension"], where="evaluation_dimension"),
        _text(raw["perturbation_kind"], where="perturbation_kind"),
        _text(
            raw["supersedes_seed_id"], where="supersedes_seed_id",
            allow_empty=True),
        _text(
            raw["retention_anchor_id"], where="retention_anchor_id",
            allow_empty=True),
        _integer(raw["logical_order"], where="logical_order"),
        _text(raw["license_id"], where="license_id"),
    )


def read_authored_construction_seeds(
        path: str | Path) -> tuple[AuthoredConstructionSeed, ...]:
    """严格读取 JSONL，并核七类、构式族、歧义/同命题对和 held-out。"""
    try:
        payload = Path(path).read_bytes()
    except OSError as error:
        raise AuthoredConstructionCourseError("construction sample 无法读取") from error
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise AuthoredConstructionCourseError("construction sample 换行非法")
    seeds: list[AuthoredConstructionSeed] = []
    try:
        for line in payload.splitlines(keepends=True):
            value = parse_canonical_json_bytes(line[:-1], require_object=True)
            assert isinstance(value, dict)
            if canonical_json_line(value) != line:
                raise AuthoredConstructionCourseError("construction seed 非规范 JSON")
            seeds.append(_seed_from_value(value))
    except AuthoredConstructionCourseError:
        raise
    except Exception as error:
        raise AuthoredConstructionCourseError("construction seed 损坏") from error
    identifiers = [item.seed_id for item in seeds]
    candidate_ids = [item.candidate_id for item in seeds]
    orders = [item.logical_order for item in seeds]
    if not seeds or len(identifiers) != len(set(identifiers)):
        raise AuthoredConstructionCourseError("construction seed_id 空或重复")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise AuthoredConstructionCourseError("candidate_id 重复")
    if orders != sorted(orders) or len(orders) != len(set(orders)):
        raise AuthoredConstructionCourseError("logical_order 必须严格递增")
    index = {item.seed_id: item for item in seeds}
    for seed in seeds:
        expected_role = _ROLE_BY_FAMILY.get(seed.sample_family)
        if expected_role is not None and seed.sample_role != expected_role:
            raise AuthoredConstructionCourseError("sample family/role 不一致")
        if seed.sample_family == "GENERATION" and seed.sample_role not in {
                "support", "read_only_probe", "refute"}:
            raise AuthoredConstructionCourseError("generation sample_role 非法")
        expected_state = _STATE_BY_FAMILY.get(seed.sample_family)
        if expected_state is not None and seed.expected_state != expected_state:
            raise AuthoredConstructionCourseError("sample family/state 不一致")
        if seed.sample_family == "GENERATION" and seed.expected_state not in {
                "TRUE", "FALSE"}:
            raise AuthoredConstructionCourseError("generation state 非法")
        if seed.sample_family == "REVISION":
            target = index.get(seed.supersedes_seed_id)
            if (target is None or target.logical_order >= seed.logical_order
                    or target.family != seed.family or target.split != seed.split):
                raise AuthoredConstructionCourseError(
                    "revision 必须 supersede 同族更早 seed")
        elif seed.supersedes_seed_id:
            raise AuthoredConstructionCourseError("非 revision 不得 supersede")
        if seed.sample_family == "RETENTION":
            anchor = index.get(seed.retention_anchor_id)
            if (anchor is None or anchor.logical_order >= seed.logical_order
                    or anchor.split != seed.split):
                raise AuthoredConstructionCourseError(
                    "retention anchor 必须是同 split 更早 seed")
        elif seed.retention_anchor_id:
            raise AuthoredConstructionCourseError("非 retention 不得绑定 anchor")

    teacher = tuple(item for item in seeds if item.label_owner == "teacher")
    evaluator = tuple(item for item in seeds if item.label_owner == "evaluator")
    for owner, owner_seeds in (("teacher", teacher), ("evaluator", evaluator)):
        if {item.sample_family for item in owner_seeds} != set(SAMPLE_FAMILIES):
            raise AuthoredConstructionCourseError(f"{owner} 未覆盖七类 sample family")
        if {item.candidate_kind for item in owner_seeds} != set(CANDIDATE_KINDS):
            raise AuthoredConstructionCourseError(f"{owner} 未覆盖完整构式候选族")
        literal = tuple(item for item in owner_seeds
                        if item.baseline_kind == "LITERAL_TOKEN_SUM_ONLY")
        if len(literal) != 1 or literal[0].expected_state != "FALSE":
            raise AuthoredConstructionCourseError(
                f"{owner} anti-literal 负基线未冻结")
        ambiguous: dict[str, set[str]] = {}
        for item in owner_seeds:
            if item.sample_family != "AMBIGUOUS":
                continue
            identity = item.construction_identity.to_value()
            ambiguous.setdefault(item.candidate_group, set()).add(
                identity["construction_key"])
        if not ambiguous or max(len(value) for value in ambiguous.values()) < 2:
            raise AuthoredConstructionCourseError(
                f"{owner} 同表层异构式未保留竞争候选")
        proposition_groups: dict[str, set[str]] = {}
        for item in owner_seeds:
            proposition_groups.setdefault(item.proposition_group, set()).add(
                item.construction_family)
        if not any(len(value) >= 2 for value in proposition_groups.values()):
            raise AuthoredConstructionCourseError(
                f"{owner} 同命题不同构式对缺失")
    if ({item.family for item in teacher} & {item.family for item in evaluator}
            or {item.template_family for item in teacher}
            & {item.template_family for item in evaluator}):
        raise AuthoredConstructionCourseError("teacher/evaluator family/template 泄漏")

    teacher_fillers = {item.filler_family for item in teacher}
    teacher_constructions = {item.construction_family for item in teacher}
    teacher_pairs = {
        (item.filler_family, item.construction_family) for item in teacher
    }
    held_out = {
        (item.filler_family, item.construction_family) for item in evaluator
        if (item.filler_family in teacher_fillers
            and item.construction_family in teacher_constructions
            and (item.filler_family, item.construction_family) not in teacher_pairs)
    }
    if len(held_out) < 2:
        raise AuthoredConstructionCourseError(
            "held-out filler×construction 组合不足")
    if {item.evaluation_dimension for item in evaluator} != set(
            EVALUATOR_DIMENSIONS):
        raise AuthoredConstructionCourseError("独立 evaluator 维度未列全")
    return tuple(seeds)


def _course_spec() -> AuthoredCourseSpec:
    """返回复用既有 Dataset publisher 的 LC-03 课程 spec。"""
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
        "authored-construction-seed-v1",
        "urn:pure-integer-ai:ph2:authored-construction-v1",
        "Pure Integer AI PH2 authored construction seed",
        "CONSTRUCTION_LABEL",
        "lc03-construction",
        280,
    )


def compile_authored_construction_course(
        sample_path: str | Path,
        release_root: str | Path) -> AuthoredCourseBuild:
    """编译 LC-03 pack，不调用 teacher 或写学习状态。"""
    seeds = read_authored_construction_seeds(sample_path)
    return publish_authored_course(
        tuple(item.compiled_seed() for item in seeds),
        sample_path,
        release_root,
        _course_spec(),
    )


def build_construction_course_manifest(
        sample_path: str | Path,
        build: AuthoredCourseBuild,
        *,
        artifact_relative_root: str = FORMAL_ARTIFACT_RELATIVE_ROOT,
        ) -> CapabilityCourseManifest:
    """汇合 LC-03 sample、pack、组合轴、verifier、消融和零执行状态。"""
    sample = Path(sample_path)
    seeds = read_authored_construction_seeds(sample)
    teacher = tuple(item for item in seeds if item.label_owner == "teacher")
    evaluator = tuple(item for item in seeds if item.label_owner == "evaluator")
    return CapabilityCourseManifest(
        1,
        COURSE_MANIFEST_ARTIFACT_VERSION,
        "COURSE_FROZEN",
        "NOT_STARTED",
        ("LC-03",),
        ("MULTIWORD_CONSTRUCTION",),
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


__all__ = [
    "COURSE_MANIFEST_PATH",
    "EVALUATOR_DIMENSIONS",
    "FORMAL_ARTIFACT_RELATIVE_ROOT",
    "PACK_NAME",
    "PAYLOAD_KIND",
    "AuthoredConstructionCourseError",
    "AuthoredConstructionSeed",
    "ConstructionPayloadAudit",
    "build_construction_course_manifest",
    "compile_authored_construction_course",
    "read_authored_construction_seeds",
    "validate_construction_payload",
]
