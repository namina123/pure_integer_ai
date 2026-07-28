"""LC-07 篇章关系、预设、信息结构和 QUD 的原创课程。"""
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
PACK_NAME = "AUTHORED_CC0_V1--CC0-1.0--lc07-discourse-information-v1"
STAGE = "W-08"
SUBSTAGE = "LC07_DISCOURSE_INFORMATION_STRUCTURE"
PAYLOAD_KIND = "DiscourseInformationCandidateV1"
COURSE_MANIFEST_ARTIFACT_VERSION = "LC-07-discourse-information-course-v1"
COURSE_MANIFEST_PATH = Path(
    "data/ph2/manifests/lc07_discourse_information_course_v1.json")
FORMAL_ARTIFACT_RELATIVE_ROOT = (
    "ph2_dataset_artifacts/d02_language_courses_v1")

CANDIDATE_KINDS = (
    "AMBIGUOUS_RELATION",
    "CAUSE",
    "CONCESSION",
    "CONTRAST",
    "ELABORATION",
    "GENERATION",
    "GIVEN_NEW",
    "NO_CONNECTIVE_BASELINE",
    "PRESUPPOSITION_CANCELLATION",
    "PRESUPPOSITION_PROJECTION",
    "QUD",
    "RETENTION",
    "REVISION",
    "TOPIC_FOCUS",
    "UNKNOWN",
    "WRONG_CONNECTIVE_BASELINE",
)
RELATION_KINDS = (
    "CAUSE", "CONCESSION", "CONTRAST", "ELABORATION", "UNKNOWN")
RELATION_STATES = ("COMPETING", "REFUTED", "SUPPORTED", "UNKNOWN")
CONNECTIVE_STATES = ("ABSENT", "EXPLICIT", "IMPLICIT", "MISMATCH", "UNKNOWN")
PRESUPPOSITION_STATES = ("CANCELLED", "PROJECTED", "UNKNOWN")
INFORMATION_STATUS_FAMILIES = (
    "GIVEN", "MIXED", "NEW", "TOPIC_FOCUS", "UNKNOWN")
QUD_FAMILIES = ("ANSWERED", "OPEN", "UNKNOWN")
BASELINE_KINDS = (
    "NO_CONNECTIVE_ONLY",
    "TYPED_DISCOURSE_OBJECTS_PRESENT",
    "WRONG_CONNECTIVE_ONLY",
)
EVALUATOR_DIMENSIONS = (
    "AMBIGUOUS_RELATION_COMPETITION",
    "CAUSE_RELATION",
    "CONCESSION_RELATION",
    "CONTRAST_RELATION",
    "DISCOURSE_UNKNOWN",
    "ELABORATION_RELATION",
    "GIVEN_NEW_STATUS",
    "LOCAL_RELATION_REVISION",
    "NO_CONNECTIVE_REJECT",
    "PRESUPPOSITION_CANCELLATION",
    "PRESUPPOSITION_PROJECTION",
    "QUD_STATE",
    "RETENTION_REVERIFY",
    "REVERSE_GENERATION_ORDER_EXPLICITNESS",
    "TOPIC_FOCUS_MINIMAL_CONTRAST",
    "WRONG_CONNECTIVE_REJECT",
)
VERIFIER_NE_CONDITIONS = (
    "CAPABILITY_LEARNED_REQUESTED",
    "GENERATION_RESULT_NOT_EXECUTED",
    "MD02_MD03_RUNTIME_PREREQUISITE_MISSING",
    "NO_EVALUATOR_LABEL",
    "OUT_OF_DOCUMENT_SCOPE",
    "SEMANTIC_TRUTH_REQUESTED",
)
RETENTION_PROTOCOLS = (
    "A_TO_LC07_REVERIFY_A",
    "DUMP_RESUME_REQUIRED_AT_RUNTIME",
    "LOCAL_RELATION_SUPERSEDE_ONLY",
)
COMBINATION_AXES = (
    "INFORMATION_STATUS_FAMILY",
    "QUD_FAMILY",
    "RELATION_FAMILY",
    "RELATION_X_INFORMATION_X_QUD",
)
ABLATION_KEYS = (
    "DROP_DISCOURSE_RELATION",
    "DROP_INFORMATION_STRUCTURE",
    "DROP_PRESUPPOSITION_OBLIGATION",
    "DROP_QUD_CANDIDATE",
    "NO_CONNECTIVE_ONLY",
    "WRONG_CONNECTIVE_ONLY",
)
PAYLOAD_KEYS = (
    "baseline_kind",
    "candidate_kind",
    "discourse_relations",
    "generation_constraint",
    "information_structure",
    "objective_keys",
    "observed_surface",
    "presupposition_obligations",
    "proposition_candidates",
    "qud_candidates",
    "resource_budget",
    "retention_anchor_id",
    "revision_receipt",
    "sample_family",
    "selection_state",
    "split_identity",
    "surface_scope",
)

_SEED_FIELDS = {
    "baseline_kind", "candidate_kind", "discourse_relations",
    "evaluation_dimension", "expected_payload", "expected_state", "family",
    "generation_constraint", "information_status_family", "information_structure",
    "label_owner", "license_id", "logical_order", "objective_keys",
    "observed_text", "perturbation_kind", "presupposition_obligations",
    "proposition_candidates", "qud_candidates", "qud_family",
    "relation_family", "resource_budget", "retention_anchor_id",
    "revision_receipt", "sample_family", "sample_role", "seed_id", "split",
    "supersedes_seed_id", "surface_scope", "template_family",
}
_SCOPE_FIELDS = {
    "context_scope_key", "genre", "language", "register", "script"}
_PROPOSITION_FIELDS = {
    "context_scope_key", "ordinal", "proposition_id", "proposition_key"}
_RELATION_FIELDS = {
    "candidate_state", "candidate_version", "connective_state",
    "context_scope_key", "provisional", "relation_id", "relation_kind",
    "source_proposition_id", "target_proposition_id",
}
_PRESUPPOSITION_FIELDS = {
    "candidate_version", "content_proposition_id", "dependency_relation_ids",
    "obligation_id", "status", "trigger_proposition_id",
}
_INFORMATION_FIELDS = {
    "candidate_version", "contrast_set_keys", "focus_key", "information_id",
    "proposition_id", "status_family", "topic_key",
}
_QUD_FIELDS = {
    "candidate_version", "priority", "provisional", "question_key", "qud_id",
    "status", "target_proposition_id",
}
_GENERATION_FIELDS = {
    "direction", "explicitness_preservation_required",
    "given_new_preservation_required", "input_candidate_ids",
    "output_surface_hidden", "presupposition_preservation_required",
    "proposition_order_preservation_required", "qud_preservation_required",
    "relation_preservation_required", "topic_focus_preservation_required",
}
_REVISION_FIELDS = {
    "affected_relation_ids", "candidate_version", "dependency_keys",
    "raw_observation_preserved", "revision_scope", "supersedes_seed_id",
    "unaffected_proposition_ids",
}
_BUDGET_FIELDS = {
    "max_dependency_edges", "max_information_states", "max_output_units",
    "max_presuppositions", "max_propositions", "max_qud", "max_relations",
}
_EXPECTED_FIELDS = {
    "accepted", "accepted_surfaces", "analysis_key", "reason_code"}
_SPLIT_FIELDS = {
    "combination_key", "information_status_family", "isolation_axis",
    "qud_family", "relation_family",
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


class AuthoredDiscourseInformationCourseError(RuntimeError):
    """LC-07 seed、typed payload 或组合 split 不满足冻结合同。"""


def _exact_keys(value: Any, keys: set[str], *, where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise AuthoredDiscourseInformationCourseError(f"{where} 字段集合非法")
    return value


def _text(value: Any, *, where: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or value.strip() != value:
        raise AuthoredDiscourseInformationCourseError(f"{where} 必须是规范文本")
    if not allow_empty and not value:
        raise AuthoredDiscourseInformationCourseError(f"{where} 不能为空")
    return value


def _integer(value: Any, *, where: str, positive: bool = False) -> int:
    if type(value) is not int or value < int(positive):
        qualifier = "正" if positive else "非负"
        raise AuthoredDiscourseInformationCourseError(
            f"{where} 必须是{qualifier}严格整数")
    return value


def _flag(value: Any, *, where: str) -> int:
    if type(value) is not int or value not in {0, 1}:
        raise AuthoredDiscourseInformationCourseError(f"{where} 必须是 0/1")
    return value


def _string_tuple(
        value: Any,
        *,
        where: str,
        sorted_unique: bool = False,
        allow_empty: bool = False,
        ) -> tuple[str, ...]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise AuthoredDiscourseInformationCourseError(f"{where} 必须是文本数组")
    result = tuple(_text(item, where=where) for item in value)
    if len(result) != len(set(result)):
        raise AuthoredDiscourseInformationCourseError(f"{where} 不得重复")
    if sorted_unique and tuple(sorted(result)) != result:
        raise AuthoredDiscourseInformationCourseError(f"{where} 必须排序")
    return result


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_expected(
        payload: CanonicalJsonObject,
        *,
        expected_state: str,
        ) -> None:
    raw = _exact_keys(payload.to_value(), _EXPECTED_FIELDS, where="expected")
    accepted = _flag(raw["accepted"], where="expected accepted")
    surfaces = _string_tuple(
        raw["accepted_surfaces"], where="accepted surfaces", allow_empty=True)
    _text(raw["analysis_key"], where="analysis_key")
    _text(raw["reason_code"], where="reason_code")
    if expected_state == "TRUE" and (accepted != 1 or not surfaces):
        raise AuthoredDiscourseInformationCourseError(
            "TRUE label 必须给出私有合法表层")
    if expected_state != "TRUE" and (accepted != 0 or surfaces):
        raise AuthoredDiscourseInformationCourseError(
            "非 TRUE label 不得给出采用表层")


@dataclass(frozen=True)
class DiscourseInformationPayloadAudit:
    """返回 typed 对象计数、组合键和两类负基线状态。"""

    proposition_count: int
    relation_count: int
    presupposition_count: int
    information_count: int
    qud_count: int
    combination_key: str
    no_connective_baseline: int
    wrong_connective_baseline: int


def validate_discourse_information_payload(
        payload: CanonicalJsonObject | dict[str, Any],
        ) -> DiscourseInformationPayloadAudit:
    """校验篇章关系、预设、topic/focus、given/new、QUD 和局部修正。"""
    value = payload.to_value() if isinstance(payload, CanonicalJsonObject) else payload
    raw = _exact_keys(value, set(PAYLOAD_KEYS), where="discourse payload")
    candidate_kind = _text(raw["candidate_kind"], where="candidate_kind")
    if candidate_kind not in CANDIDATE_KINDS:
        raise AuthoredDiscourseInformationCourseError("candidate_kind 未登记")
    baseline = _text(raw["baseline_kind"], where="baseline_kind")
    if baseline not in BASELINE_KINDS:
        raise AuthoredDiscourseInformationCourseError("baseline_kind 未登记")
    if raw["sample_family"] not in SAMPLE_FAMILIES:
        raise AuthoredDiscourseInformationCourseError("sample_family 未登记")
    if raw["selection_state"] != "UNSELECTED":
        raise AuthoredDiscourseInformationCourseError(
            "Observation 不得预选篇章候选")
    objective_keys = _string_tuple(
        raw["objective_keys"], where="objective_keys", sorted_unique=True)
    if any(key not in LANGUAGE_OBJECTIVE_KEYS for key in objective_keys):
        raise AuthoredDiscourseInformationCourseError("objective_keys 未登记")

    scope = _exact_keys(raw["surface_scope"], _SCOPE_FIELDS, where="scope")
    if scope["language"] != "zh" or scope["script"] != "HAN":
        raise AuthoredDiscourseInformationCourseError("LC-07 首阶段只冻结 zh/HAN")
    context_scope = _text(scope["context_scope_key"], where="context scope")
    _text(scope["genre"], where="genre")
    if scope["register"] not in {"COLLOQUIAL", "FORMAL", "NEUTRAL"}:
        raise AuthoredDiscourseInformationCourseError("register 未登记")

    observed = _exact_keys(raw["observed_surface"], {
        "append_only", "sha256", "target_hidden", "text",
    }, where="observed surface")
    observed_text = _text(observed["text"], where="observed text")
    if _flag(observed["append_only"], where="append_only") != 1:
        raise AuthoredDiscourseInformationCourseError(
            "raw Observation 必须 append-only")
    if observed["sha256"] != _sha256_text(observed_text):
        raise AuthoredDiscourseInformationCourseError(
            "observed surface SHA-256 漂移")
    target_hidden = _flag(observed["target_hidden"], where="target_hidden")

    budget = _exact_keys(raw["resource_budget"], _BUDGET_FIELDS, where="budget")
    ceilings = {
        key: _integer(budget[key], where=key, positive=True)
        for key in sorted(_BUDGET_FIELDS)
    }
    if len(observed_text) > ceilings["max_output_units"]:
        raise AuthoredDiscourseInformationCourseError("surface 超整数输出预算")

    propositions_value = raw["proposition_candidates"]
    if not isinstance(propositions_value, list) or not propositions_value:
        raise AuthoredDiscourseInformationCourseError(
            "proposition_candidates 不能为空")
    if len(propositions_value) > ceilings["max_propositions"]:
        raise AuthoredDiscourseInformationCourseError("Proposition 超预算")
    propositions: dict[str, dict[str, Any]] = {}
    ordinals: list[int] = []
    for item in propositions_value:
        proposition = _exact_keys(item, _PROPOSITION_FIELDS, where="proposition")
        proposition_id = _text(proposition["proposition_id"], where="proposition_id")
        if proposition_id in propositions:
            raise AuthoredDiscourseInformationCourseError("proposition_id 重复")
        if proposition["context_scope_key"] != context_scope:
            raise AuthoredDiscourseInformationCourseError(
                "Proposition 跨 ContextScope")
        _text(proposition["proposition_key"], where="proposition_key")
        ordinals.append(_integer(
            proposition["ordinal"], where="proposition ordinal", positive=True))
        propositions[proposition_id] = proposition
    if ordinals != list(range(1, len(ordinals) + 1)):
        raise AuthoredDiscourseInformationCourseError(
            "Proposition ordinal 必须连续")

    relations_value = raw["discourse_relations"]
    if not isinstance(relations_value, list):
        raise AuthoredDiscourseInformationCourseError(
            "discourse_relations 必须是数组")
    if len(relations_value) > ceilings["max_relations"]:
        raise AuthoredDiscourseInformationCourseError("篇章关系超预算")
    relations: dict[str, dict[str, Any]] = {}
    for item in relations_value:
        relation = _exact_keys(item, _RELATION_FIELDS, where="discourse relation")
        relation_id = _text(relation["relation_id"], where="relation_id")
        if relation_id in relations:
            raise AuthoredDiscourseInformationCourseError("relation_id 重复")
        source = _text(
            relation["source_proposition_id"], where="relation source")
        target = _text(
            relation["target_proposition_id"], where="relation target")
        if source not in propositions or target not in propositions or source == target:
            raise AuthoredDiscourseInformationCourseError("篇章关系端点非法")
        if relation["relation_kind"] not in RELATION_KINDS:
            raise AuthoredDiscourseInformationCourseError("relation_kind 未登记")
        if relation["candidate_state"] not in RELATION_STATES:
            raise AuthoredDiscourseInformationCourseError("relation state 未登记")
        if relation["connective_state"] not in CONNECTIVE_STATES:
            raise AuthoredDiscourseInformationCourseError("connective state 未登记")
        if relation["context_scope_key"] != context_scope:
            raise AuthoredDiscourseInformationCourseError(
                "篇章关系跨 ContextScope")
        _integer(relation["candidate_version"], where="relation version", positive=True)
        if _flag(relation["provisional"], where="relation provisional") != 1:
            raise AuthoredDiscourseInformationCourseError(
                "篇章关系只能是 provisional candidate")
        if (relation["relation_kind"] == "UNKNOWN") != (
                relation["candidate_state"] == "UNKNOWN"):
            raise AuthoredDiscourseInformationCourseError(
                "unknown relation 与候选状态不一致")
        if (relation["connective_state"] == "MISMATCH"
                and relation["candidate_state"] != "REFUTED"):
            raise AuthoredDiscourseInformationCourseError(
                "错误 connective 必须是 refuted candidate")
        relations[relation_id] = relation

    presuppositions_value = raw["presupposition_obligations"]
    if not isinstance(presuppositions_value, list):
        raise AuthoredDiscourseInformationCourseError(
            "presupposition_obligations 必须是数组")
    if len(presuppositions_value) > ceilings["max_presuppositions"]:
        raise AuthoredDiscourseInformationCourseError("预设 obligation 超预算")
    presuppositions: dict[str, dict[str, Any]] = {}
    dependency_edges = 0
    for item in presuppositions_value:
        obligation = _exact_keys(
            item, _PRESUPPOSITION_FIELDS, where="presupposition obligation")
        obligation_id = _text(obligation["obligation_id"], where="obligation_id")
        if obligation_id in presuppositions:
            raise AuthoredDiscourseInformationCourseError("obligation_id 重复")
        trigger = _text(
            obligation["trigger_proposition_id"], where="presupposition trigger")
        content = _text(
            obligation["content_proposition_id"], where="presupposition content")
        if trigger not in propositions or content not in propositions:
            raise AuthoredDiscourseInformationCourseError(
                "预设 obligation 引用未知 Proposition")
        if obligation["status"] not in PRESUPPOSITION_STATES:
            raise AuthoredDiscourseInformationCourseError("presupposition status 未登记")
        dependencies = _string_tuple(
            obligation["dependency_relation_ids"],
            where="presupposition dependencies", sorted_unique=True,
            allow_empty=True)
        if any(value not in relations for value in dependencies):
            raise AuthoredDiscourseInformationCourseError(
                "预设 obligation 引用未知关系")
        dependency_edges += len(dependencies)
        _integer(
            obligation["candidate_version"],
            where="presupposition version", positive=True)
        presuppositions[obligation_id] = obligation
    if dependency_edges + len(relations) > ceilings["max_dependency_edges"]:
        raise AuthoredDiscourseInformationCourseError("篇章依赖边超预算")

    information_value = raw["information_structure"]
    if not isinstance(information_value, list) or len(information_value) != 1:
        raise AuthoredDiscourseInformationCourseError(
            "每条 LC-07 样本必须有一个信息结构候选")
    if len(information_value) > ceilings["max_information_states"]:
        raise AuthoredDiscourseInformationCourseError("信息结构超预算")
    information = _exact_keys(
        information_value[0], _INFORMATION_FIELDS, where="information structure")
    _text(information["information_id"], where="information_id")
    if information["proposition_id"] not in propositions:
        raise AuthoredDiscourseInformationCourseError(
            "信息结构引用未知 Proposition")
    status_family = _text(
        information["status_family"], where="information status family")
    if status_family not in INFORMATION_STATUS_FAMILIES:
        raise AuthoredDiscourseInformationCourseError(
            "information status family 未登记")
    topic = _text(information["topic_key"], where="topic_key", allow_empty=True)
    focus = _text(information["focus_key"], where="focus_key", allow_empty=True)
    contrast = _string_tuple(
        information["contrast_set_keys"], where="contrast set",
        sorted_unique=True, allow_empty=True)
    _integer(
        information["candidate_version"],
        where="information version", positive=True)
    if status_family == "GIVEN" and (not topic or focus or contrast):
        raise AuthoredDiscourseInformationCourseError("GIVEN 信息结构非法")
    if status_family == "NEW" and (topic or not focus or contrast):
        raise AuthoredDiscourseInformationCourseError("NEW 信息结构非法")
    if status_family == "MIXED" and (not topic or not focus or contrast):
        raise AuthoredDiscourseInformationCourseError("MIXED 信息结构非法")
    if status_family == "TOPIC_FOCUS" and (
            not topic or not focus or len(contrast) < 2):
        raise AuthoredDiscourseInformationCourseError(
            "topic/focus 必须保留最小对立集")
    if status_family == "UNKNOWN" and (topic or focus or contrast):
        raise AuthoredDiscourseInformationCourseError(
            "unknown 信息结构不得猜 topic/focus")

    qud_value = raw["qud_candidates"]
    if not isinstance(qud_value, list) or len(qud_value) != 1:
        raise AuthoredDiscourseInformationCourseError(
            "每条 LC-07 样本必须有一个 QUD candidate")
    if len(qud_value) > ceilings["max_qud"]:
        raise AuthoredDiscourseInformationCourseError("QUD 超预算")
    qud = _exact_keys(qud_value[0], _QUD_FIELDS, where="QUD candidate")
    _text(qud["qud_id"], where="qud_id")
    _text(qud["question_key"], where="question_key")
    if qud["target_proposition_id"] not in propositions:
        raise AuthoredDiscourseInformationCourseError(
            "QUD 引用未知 Proposition")
    if qud["status"] not in QUD_FAMILIES:
        raise AuthoredDiscourseInformationCourseError("QUD status 未登记")
    _integer(qud["priority"], where="QUD priority", positive=True)
    _integer(qud["candidate_version"], where="QUD version", positive=True)
    if _flag(qud["provisional"], where="QUD provisional") != 1:
        raise AuthoredDiscourseInformationCourseError("QUD 只能 provisional")

    revision = _exact_keys(
        raw["revision_receipt"], _REVISION_FIELDS, where="revision receipt")
    revision_version = _integer(
        revision["candidate_version"], where="revision version", positive=True)
    dependencies = _string_tuple(
        revision["dependency_keys"], where="revision dependencies",
        sorted_unique=True, allow_empty=True)
    affected = _string_tuple(
        revision["affected_relation_ids"], where="affected relations",
        sorted_unique=True, allow_empty=True)
    unaffected = _string_tuple(
        revision["unaffected_proposition_ids"], where="unaffected propositions",
        sorted_unique=True, allow_empty=True)
    supersedes = _text(
        revision["supersedes_seed_id"], where="revision supersedes",
        allow_empty=True)
    if _flag(revision["raw_observation_preserved"], where="raw preserved") != 1:
        raise AuthoredDiscourseInformationCourseError(
            "revision 不得覆盖 raw Observation")
    if candidate_kind == "REVISION":
        if (revision["revision_scope"] != "DISCOURSE_RELATION_ONLY"
                or not supersedes or revision_version < 2
                or dependencies != ("DISCOURSE_RELATION",)
                or not affected or set(affected) != set(relations)
                or set(unaffected) != set(propositions)):
            raise AuthoredDiscourseInformationCourseError(
                "篇章关系 revision receipt 不完整")
    elif (revision["revision_scope"] != "NONE" or supersedes
          or revision_version != 1 or dependencies or affected or unaffected):
        raise AuthoredDiscourseInformationCourseError(
            "非 revision 不得伪造修正 receipt")

    generation = _exact_keys(
        raw["generation_constraint"], _GENERATION_FIELDS,
        where="generation constraint")
    inputs = _string_tuple(
        generation["input_candidate_ids"], where="generation inputs")
    known_inputs = (
        set(propositions) | set(relations) | set(presuppositions)
        | {information["information_id"], qud["qud_id"]})
    if any(item not in known_inputs for item in inputs):
        raise AuthoredDiscourseInformationCourseError(
            "generation 引用未知 typed candidate")
    hidden = _flag(generation["output_surface_hidden"], where="output hidden")
    preservation_keys = (
        "explicitness_preservation_required", "given_new_preservation_required",
        "presupposition_preservation_required",
        "proposition_order_preservation_required", "qud_preservation_required",
        "relation_preservation_required", "topic_focus_preservation_required",
    )
    for key in preservation_keys:
        _flag(generation[key], where=key)
    if candidate_kind == "GENERATION":
        if (generation["direction"] != "SEMANTICS_TO_SURFACE" or hidden != 1
                or target_hidden != 1
                or any(generation[key] != 1 for key in preservation_keys)):
            raise AuthoredDiscourseInformationCourseError(
                "generation 必须隐藏输出并逐层保持篇章信息")
    elif hidden != 0 or target_hidden != 0:
        raise AuthoredDiscourseInformationCourseError(
            "非 generation 不得隐藏目标表层")

    split = _exact_keys(raw["split_identity"], _SPLIT_FIELDS, where="split")
    relation_family = _text(
        split["relation_family"], where="split relation_family")
    information_family = _text(
        split["information_status_family"], where="split information family")
    qud_family = _text(split["qud_family"], where="split QUD family")
    if candidate_kind == "AMBIGUOUS_RELATION":
        observed_relation_family = "AMBIGUOUS"
    elif not relations:
        observed_relation_family = "NONE"
    else:
        relation_kinds = {item["relation_kind"] for item in relations_value}
        if len(relation_kinds) != 1:
            raise AuthoredDiscourseInformationCourseError(
                "非歧义样本不得混合关系族")
        observed_relation_family = next(iter(relation_kinds))
    if relation_family != observed_relation_family:
        raise AuthoredDiscourseInformationCourseError(
            "组合 relation_family 与 typed 关系不一致")
    if information_family != status_family:
        raise AuthoredDiscourseInformationCourseError(
            "组合 information family 与 typed 信息状态不一致")
    if qud_family != qud["status"]:
        raise AuthoredDiscourseInformationCourseError(
            "组合 QUD family 与 typed QUD 不一致")
    combination_key = f"{relation_family}::{information_family}::{qud_family}"
    if (split["combination_key"] != combination_key
            or split["isolation_axis"] != "RELATION_X_INFORMATION_X_QUD"):
        raise AuthoredDiscourseInformationCourseError(
            "组合 split identity 漂移")

    retention_anchor = _text(
        raw["retention_anchor_id"], where="retention anchor", allow_empty=True)
    if candidate_kind == "RETENTION" and not retention_anchor:
        raise AuthoredDiscourseInformationCourseError(
            "retention 必须绑定更早 anchor")
    if candidate_kind != "RETENTION" and retention_anchor:
        raise AuthoredDiscourseInformationCourseError(
            "非 retention 不得绑定 anchor")

    no_connective = int(baseline == "NO_CONNECTIVE_ONLY")
    wrong_connective = int(baseline == "WRONG_CONNECTIVE_ONLY")
    if candidate_kind == "NO_CONNECTIVE_BASELINE":
        if not no_connective or relations:
            raise AuthoredDiscourseInformationCourseError(
                "无 connective 不得猜篇章关系")
    elif no_connective:
        raise AuthoredDiscourseInformationCourseError(
            "no-connective 基线身份错配")
    if candidate_kind == "WRONG_CONNECTIVE_BASELINE":
        if (not wrong_connective or not relations
                or any(item["connective_state"] != "MISMATCH"
                       for item in relations_value)):
            raise AuthoredDiscourseInformationCourseError(
                "错误 connective 基线必须显式 refute")
    elif wrong_connective:
        raise AuthoredDiscourseInformationCourseError(
            "wrong-connective 基线身份错配")

    direct_relations = {
        "CAUSE": "CAUSE",
        "CONCESSION": "CONCESSION",
        "CONTRAST": "CONTRAST",
        "ELABORATION": "ELABORATION",
    }
    expected_relation = direct_relations.get(candidate_kind)
    if expected_relation is not None and not (
            len(relations_value) == 1
            and relations_value[0]["relation_kind"] == expected_relation
            and relations_value[0]["candidate_state"] == "SUPPORTED"):
        raise AuthoredDiscourseInformationCourseError(
            "显式篇章关系候选缺失或状态非法")
    if candidate_kind == "PRESUPPOSITION_PROJECTION" and not (
            len(presuppositions_value) == 1
            and presuppositions_value[0]["status"] == "PROJECTED"):
        raise AuthoredDiscourseInformationCourseError("预设投射 obligation 缺失")
    if candidate_kind == "PRESUPPOSITION_CANCELLATION" and not (
            len(presuppositions_value) == 1
            and presuppositions_value[0]["status"] == "CANCELLED"):
        raise AuthoredDiscourseInformationCourseError("预设取消 obligation 缺失")
    if candidate_kind == "TOPIC_FOCUS" and status_family != "TOPIC_FOCUS":
        raise AuthoredDiscourseInformationCourseError("topic/focus 候选缺失")
    if candidate_kind == "GIVEN_NEW" and status_family != "MIXED":
        raise AuthoredDiscourseInformationCourseError("given/new 分账缺失")
    if candidate_kind == "QUD" and qud["status"] != "OPEN":
        raise AuthoredDiscourseInformationCourseError("开放 QUD 候选缺失")
    if candidate_kind == "AMBIGUOUS_RELATION":
        if (len(relations_value) < 2
                or len({item["relation_kind"] for item in relations_value}) < 2
                or any(item["candidate_state"] != "COMPETING"
                       for item in relations_value)):
            raise AuthoredDiscourseInformationCourseError(
                "篇章关系歧义竞争缺失")
    if candidate_kind == "UNKNOWN" and not (
            len(relations_value) == 1
            and relations_value[0]["relation_kind"] == "UNKNOWN"
            and status_family == "UNKNOWN" and qud["status"] == "UNKNOWN"):
        raise AuthoredDiscourseInformationCourseError(
            "relation/information/QUD unknown 未显式表示")
    if candidate_kind == "REVISION" and not (
            len(relations_value) == 1
            and relations_value[0]["candidate_state"] == "REFUTED"):
        raise AuthoredDiscourseInformationCourseError(
            "后文 relation refute 未显式表示")

    return DiscourseInformationPayloadAudit(
        len(propositions), len(relations), len(presuppositions),
        len(information_value), len(qud_value), combination_key,
        no_connective, wrong_connective,
    )


@dataclass(frozen=True)
class AuthoredDiscourseInformationSeed:
    """一个 owner、split、篇章候选集合和独立 label。"""

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
    proposition_candidates: tuple[CanonicalJsonObject, ...]
    discourse_relations: tuple[CanonicalJsonObject, ...]
    presupposition_obligations: tuple[CanonicalJsonObject, ...]
    information_structure: tuple[CanonicalJsonObject, ...]
    qud_candidates: tuple[CanonicalJsonObject, ...]
    generation_constraint: CanonicalJsonObject
    revision_receipt: CanonicalJsonObject
    resource_budget: CanonicalJsonObject
    relation_family: str
    information_status_family: str
    qud_family: str
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
                ("relation_family", self.relation_family),
                ("information_status_family", self.information_status_family),
                ("qud_family", self.qud_family),
                ("perturbation_kind", self.perturbation_kind)):
            _text(value, where=name)
        if self.label_owner not in {"teacher", "evaluator"}:
            raise AuthoredDiscourseInformationCourseError("label_owner 非法")
        if self.split != ("train" if self.label_owner == "teacher" else "held_out"):
            raise AuthoredDiscourseInformationCourseError(
                "label_owner 与 split 不一致")
        if self.sample_family not in SAMPLE_FAMILIES:
            raise AuthoredDiscourseInformationCourseError("sample_family 未登记")
        if self.sample_role not in SAMPLE_ROLES:
            raise AuthoredDiscourseInformationCourseError("sample_role 未登记")
        if self.candidate_kind not in CANDIDATE_KINDS:
            raise AuthoredDiscourseInformationCourseError("candidate_kind 未登记")
        _text(self.observed_text, where="observed_text")
        for name in (
                "surface_scope", "generation_constraint", "revision_receipt",
                "resource_budget"):
            if not isinstance(getattr(self, name), CanonicalJsonObject):
                raise AuthoredDiscourseInformationCourseError(f"{name} 类型非法")
        for name in (
                "proposition_candidates", "discourse_relations",
                "presupposition_obligations", "information_structure",
                "qud_candidates"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or any(
                    not isinstance(item, CanonicalJsonObject) for item in values):
                raise AuthoredDiscourseInformationCourseError(f"{name} 类型非法")
        if not self.proposition_candidates:
            raise AuthoredDiscourseInformationCourseError(
                "Proposition candidates 不能为空")
        if self.baseline_kind not in BASELINE_KINDS:
            raise AuthoredDiscourseInformationCourseError("baseline_kind 未登记")
        if self.objective_keys != tuple(sorted(set(self.objective_keys))):
            raise AuthoredDiscourseInformationCourseError(
                "objective_keys 必须排序去重")
        if not self.objective_keys or any(
                item not in LANGUAGE_OBJECTIVE_KEYS for item in self.objective_keys):
            raise AuthoredDiscourseInformationCourseError("objective_keys 未登记")
        if self.expected_state not in EXPECTED_STATES:
            raise AuthoredDiscourseInformationCourseError("expected_state 非四态")
        if not isinstance(self.expected_payload, CanonicalJsonObject):
            raise AuthoredDiscourseInformationCourseError(
                "expected_payload 类型非法")
        _validate_expected(self.expected_payload, expected_state=self.expected_state)
        if self.evaluation_dimension not in EVALUATOR_DIMENSIONS:
            raise AuthoredDiscourseInformationCourseError(
                "evaluation_dimension 未登记")
        _text(self.supersedes_seed_id, where="supersedes_seed_id", allow_empty=True)
        _text(self.retention_anchor_id, where="retention_anchor_id", allow_empty=True)
        _integer(self.logical_order, where="logical_order", positive=True)
        if self.license_id != LICENSE_ID:
            raise AuthoredDiscourseInformationCourseError(
                "原创 LC-07 课程必须为 CC0-1.0")
        validate_discourse_information_payload(self.observation_payload())

    def observation_payload(self) -> CanonicalJsonObject:
        generation = self.generation_constraint.to_value()
        return CanonicalJsonObject.from_value({
            "baseline_kind": self.baseline_kind,
            "candidate_kind": self.candidate_kind,
            "discourse_relations": [
                item.to_value() for item in self.discourse_relations],
            "generation_constraint": generation,
            "information_structure": [
                item.to_value() for item in self.information_structure],
            "objective_keys": list(self.objective_keys),
            "observed_surface": {
                "append_only": 1,
                "sha256": _sha256_text(self.observed_text),
                "target_hidden": generation["output_surface_hidden"],
                "text": self.observed_text,
            },
            "presupposition_obligations": [
                item.to_value() for item in self.presupposition_obligations],
            "proposition_candidates": [
                item.to_value() for item in self.proposition_candidates],
            "qud_candidates": [
                item.to_value() for item in self.qud_candidates],
            "resource_budget": self.resource_budget.to_value(),
            "retention_anchor_id": self.retention_anchor_id,
            "revision_receipt": self.revision_receipt.to_value(),
            "sample_family": self.sample_family,
            "selection_state": "UNSELECTED",
            "split_identity": {
                "combination_key": (
                    f"{self.relation_family}::"
                    f"{self.information_status_family}::{self.qud_family}"),
                "information_status_family": self.information_status_family,
                "isolation_axis": "RELATION_X_INFORMATION_X_QUD",
                "qud_family": self.qud_family,
                "relation_family": self.relation_family,
            },
            "surface_scope": self.surface_scope.to_value(),
        })

    def compiled_seed(self) -> AuthoredCompiledSeed:
        payload = self.observation_payload()
        audit = validate_discourse_information_payload(payload)
        proposition_ids = tuple(
            item.to_value()["proposition_id"]
            for item in self.proposition_candidates)
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
            (self.seed_id, proposition_ids, audit.combination_key),
            (self.observed_text, self.family),
            (
                self.candidate_kind, self.relation_family,
                self.information_status_family, self.qud_family,
                audit.proposition_count, audit.relation_count,
                audit.presupposition_count, audit.information_count,
                audit.qud_count,
            ),
            self.evaluation_dimension,
        )


def _canonical_objects(
        values: Any,
        fields: set[str],
        *,
        where: str,
        ) -> tuple[CanonicalJsonObject, ...]:
    if not isinstance(values, list):
        raise AuthoredDiscourseInformationCourseError(f"{where} 必须是数组")
    return tuple(CanonicalJsonObject.from_value(
        _exact_keys(item, fields, where=where)) for item in values)


def _seed_from_value(value: dict[str, Any]) -> AuthoredDiscourseInformationSeed:
    raw = _exact_keys(value, _SEED_FIELDS, where="discourse information seed")
    expected = raw["expected_payload"]
    if not isinstance(expected, dict):
        raise AuthoredDiscourseInformationCourseError(
            "expected_payload 必须是对象")
    return AuthoredDiscourseInformationSeed(
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
            raw["proposition_candidates"], _PROPOSITION_FIELDS,
            where="propositions"),
        _canonical_objects(
            raw["discourse_relations"], _RELATION_FIELDS,
            where="relations"),
        _canonical_objects(
            raw["presupposition_obligations"], _PRESUPPOSITION_FIELDS,
            where="presuppositions"),
        _canonical_objects(
            raw["information_structure"], _INFORMATION_FIELDS,
            where="information"),
        _canonical_objects(
            raw["qud_candidates"], _QUD_FIELDS, where="QUD"),
        CanonicalJsonObject.from_value(_exact_keys(
            raw["generation_constraint"], _GENERATION_FIELDS,
            where="generation_constraint")),
        CanonicalJsonObject.from_value(_exact_keys(
            raw["revision_receipt"], _REVISION_FIELDS,
            where="revision_receipt")),
        CanonicalJsonObject.from_value(_exact_keys(
            raw["resource_budget"], _BUDGET_FIELDS,
            where="resource_budget")),
        _text(raw["relation_family"], where="relation_family"),
        _text(
            raw["information_status_family"],
            where="information_status_family"),
        _text(raw["qud_family"], where="qud_family"),
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
        _integer(raw["logical_order"], where="logical_order", positive=True),
        _text(raw["license_id"], where="license_id"),
    )


def read_authored_discourse_information_seeds(
        path: str | Path,
        ) -> tuple[AuthoredDiscourseInformationSeed, ...]:
    """严格读取 JSONL，并核双 owner、修正、retention 和组合 held-out。"""
    try:
        payload = Path(path).read_bytes()
    except OSError as error:
        raise AuthoredDiscourseInformationCourseError(
            "discourse information sample 无法读取") from error
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise AuthoredDiscourseInformationCourseError(
            "discourse information sample 换行非法")
    seeds: list[AuthoredDiscourseInformationSeed] = []
    try:
        for line in payload.splitlines(keepends=True):
            value = parse_canonical_json_bytes(line[:-1], require_object=True)
            assert isinstance(value, dict)
            if canonical_json_line(value) != line:
                raise AuthoredDiscourseInformationCourseError(
                    "discourse information seed 非规范 JSON")
            seeds.append(_seed_from_value(value))
    except AuthoredDiscourseInformationCourseError:
        raise
    except Exception as error:
        raise AuthoredDiscourseInformationCourseError(
            "discourse information seed 损坏") from error
    identifiers = [item.seed_id for item in seeds]
    orders = [item.logical_order for item in seeds]
    if not seeds or len(identifiers) != len(set(identifiers)):
        raise AuthoredDiscourseInformationCourseError("seed_id 空或重复")
    if orders != sorted(orders) or len(orders) != len(set(orders)):
        raise AuthoredDiscourseInformationCourseError(
            "logical_order 必须严格递增")
    index = {item.seed_id: item for item in seeds}
    for seed in seeds:
        expected_role = _ROLE_BY_FAMILY.get(seed.sample_family)
        if expected_role is not None and seed.sample_role != expected_role:
            raise AuthoredDiscourseInformationCourseError(
                "sample family/role 不一致")
        if seed.sample_family == "GENERATION" and seed.sample_role not in {
                "support", "read_only_probe", "refute"}:
            raise AuthoredDiscourseInformationCourseError(
                "generation sample_role 非法")
        expected_state = _STATE_BY_FAMILY.get(seed.sample_family)
        if expected_state is not None and seed.expected_state != expected_state:
            raise AuthoredDiscourseInformationCourseError(
                "sample family/state 不一致")
        if seed.sample_family == "GENERATION" and seed.expected_state not in {
                "TRUE", "FALSE"}:
            raise AuthoredDiscourseInformationCourseError(
                "generation state 非法")
        if seed.candidate_kind == "REVISION":
            target = index.get(seed.supersedes_seed_id)
            if (target is None or target.logical_order >= seed.logical_order
                    or target.family != seed.family or target.split != seed.split):
                raise AuthoredDiscourseInformationCourseError(
                    "revision 必须 supersede 同族更早 seed")
            old = target.observation_payload().to_value()
            new = seed.observation_payload().to_value()
            for key in (
                    "proposition_candidates", "presupposition_obligations",
                    "information_structure", "qud_candidates"):
                if old[key] != new[key]:
                    raise AuthoredDiscourseInformationCourseError(
                        "relation revision 不得替换 Proposition/预设/信息结构/QUD")
            if old["discourse_relations"] == new["discourse_relations"]:
                raise AuthoredDiscourseInformationCourseError(
                    "relation revision 必须改动候选状态")
            old_relations = old["discourse_relations"]
            new_relations = new["discourse_relations"]
            if len(old_relations) != len(new_relations):
                raise AuthoredDiscourseInformationCourseError(
                    "relation revision 不得增删关系身份")
            stable_fields = {
                "connective_state", "context_scope_key", "provisional",
                "relation_id", "relation_kind", "source_proposition_id",
                "target_proposition_id",
            }
            for old_relation, new_relation in zip(
                    old_relations, new_relations, strict=True):
                if any(old_relation[key] != new_relation[key]
                       for key in stable_fields):
                    raise AuthoredDiscourseInformationCourseError(
                        "relation revision 不得替换关系端点、kind 或 connective")
                if (new_relation["candidate_version"]
                        <= old_relation["candidate_version"]):
                    raise AuthoredDiscourseInformationCourseError(
                        "relation revision version 必须前进")
        elif seed.supersedes_seed_id:
            raise AuthoredDiscourseInformationCourseError(
                "非 revision 不得 supersede")
        if seed.candidate_kind == "RETENTION":
            anchor = index.get(seed.retention_anchor_id)
            if (anchor is None or anchor.logical_order >= seed.logical_order
                    or anchor.split != seed.split):
                raise AuthoredDiscourseInformationCourseError(
                    "retention anchor 必须是同 split 更早 seed")
        elif seed.retention_anchor_id:
            raise AuthoredDiscourseInformationCourseError(
                "非 retention 不得绑定 anchor")

    teacher = tuple(item for item in seeds if item.label_owner == "teacher")
    evaluator = tuple(item for item in seeds if item.label_owner == "evaluator")
    for owner, owner_seeds in (("teacher", teacher), ("evaluator", evaluator)):
        if {item.sample_family for item in owner_seeds} != set(SAMPLE_FAMILIES):
            raise AuthoredDiscourseInformationCourseError(
                f"{owner} 未覆盖七类 sample family")
        if {item.candidate_kind for item in owner_seeds} != set(CANDIDATE_KINDS):
            raise AuthoredDiscourseInformationCourseError(
                f"{owner} 未覆盖完整 LC-07 候选族")
        no_connective = tuple(
            item for item in owner_seeds
            if item.baseline_kind == "NO_CONNECTIVE_ONLY")
        wrong_connective = tuple(
            item for item in owner_seeds
            if item.baseline_kind == "WRONG_CONNECTIVE_ONLY")
        if (len(no_connective) != 1
                or no_connective[0].expected_state != "FALSE"
                or len(wrong_connective) != 1
                or wrong_connective[0].expected_state != "FALSE"):
            raise AuthoredDiscourseInformationCourseError(
                f"{owner} 两类篇章负基线未冻结")
    if ({item.family for item in teacher} & {item.family for item in evaluator}
            or {item.template_family for item in teacher}
            & {item.template_family for item in evaluator}):
        raise AuthoredDiscourseInformationCourseError(
            "teacher/evaluator family/template 泄漏")
    teacher_relations = {item.relation_family for item in teacher}
    teacher_information = {item.information_status_family for item in teacher}
    teacher_qud = {item.qud_family for item in teacher}
    teacher_triples = {
        (item.relation_family, item.information_status_family, item.qud_family)
        for item in teacher
    }
    held_out = {
        (item.relation_family, item.information_status_family, item.qud_family)
        for item in evaluator
        if item.relation_family in teacher_relations
        and item.information_status_family in teacher_information
        and item.qud_family in teacher_qud
        and (item.relation_family, item.information_status_family,
             item.qud_family) not in teacher_triples
    }
    if len(held_out) < 3:
        raise AuthoredDiscourseInformationCourseError(
            "held-out relation×information×QUD 组合不足")
    if {item.evaluation_dimension for item in evaluator} != set(
            EVALUATOR_DIMENSIONS):
        raise AuthoredDiscourseInformationCourseError(
            "独立 evaluator 维度未列全")
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
        "authored-discourse-information-seed-v1",
        "urn:pure-integer-ai:ph2:authored-discourse-information-v1",
        "Pure Integer AI PH2 authored discourse information seed",
        "DISCOURSE_INFORMATION_LABEL",
        "lc07-discourse-information",
        512,
    )


def compile_authored_discourse_information_course(
        sample_path: str | Path,
        release_root: str | Path,
        ) -> AuthoredCourseBuild:
    seeds = read_authored_discourse_information_seeds(sample_path)
    return publish_authored_course(
        tuple(item.compiled_seed() for item in seeds),
        sample_path,
        release_root,
        _course_spec(),
    )


def build_discourse_information_course_manifest(
        sample_path: str | Path,
        build: AuthoredCourseBuild,
        *,
        artifact_relative_root: str = FORMAL_ARTIFACT_RELATIVE_ROOT,
        ) -> CapabilityCourseManifest:
    sample = Path(sample_path)
    seeds = read_authored_discourse_information_seeds(sample)
    teacher = tuple(item for item in seeds if item.label_owner == "teacher")
    evaluator = tuple(item for item in seeds if item.label_owner == "evaluator")
    return CapabilityCourseManifest(
        1,
        COURSE_MANIFEST_ARTIFACT_VERSION,
        "COURSE_FROZEN",
        "NOT_STARTED",
        ("LC-07",),
        ("DISCOURSE_INFORMATION_STRUCTURE", "REFERENCE_DISCOURSE_REVISION"),
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


def _scenario_components(
        owner: str,
        ordinal: int,
        candidate_kind: str,
        relation_family: str,
        information_status_family: str,
        qud_family: str,
        *,
        revision_version: int = 1,
        shared_revision_identity: bool = False,
        ) -> dict[str, Any]:
    """构造一条 LC-07 场景的 typed 对象，并保持修正链身份稳定。"""
    token = f"{owner}_{ordinal:02d}"
    identity_token = f"{owner}_REVISION" if shared_revision_identity else token
    scope = (
        f"{owner}_CTX_REVISION" if shared_revision_identity
        else f"{token}_CTX")
    first_id = f"{identity_token}_PROPOSITION_1"
    second_id = f"{identity_token}_PROPOSITION_2"
    propositions = [
        {
            "context_scope_key": scope,
            "ordinal": 1,
            "proposition_id": first_id,
            "proposition_key": f"{identity_token}_CLAIM_1",
        },
        {
            "context_scope_key": scope,
            "ordinal": 2,
            "proposition_id": second_id,
            "proposition_key": f"{identity_token}_CLAIM_2",
        },
    ]

    relations: list[dict[str, Any]] = []
    if candidate_kind == "AMBIGUOUS_RELATION":
        for relation_index, relation_kind in enumerate(
                ("CAUSE", "CONTRAST"), start=1):
            relations.append({
                "candidate_state": "COMPETING",
                "candidate_version": 1,
                "connective_state": "IMPLICIT",
                "context_scope_key": scope,
                "provisional": 1,
                "relation_id": (
                    f"{identity_token}_RELATION_{relation_index}"),
                "relation_kind": relation_kind,
                "source_proposition_id": first_id,
                "target_proposition_id": second_id,
            })
    elif candidate_kind != "NO_CONNECTIVE_BASELINE":
        relation_state = "SUPPORTED"
        connective_state = "EXPLICIT"
        relation_kind = relation_family
        if candidate_kind == "WRONG_CONNECTIVE_BASELINE":
            relation_state = "REFUTED"
            connective_state = "MISMATCH"
        elif candidate_kind == "UNKNOWN":
            relation_state = "UNKNOWN"
            connective_state = "UNKNOWN"
        elif candidate_kind == "REVISION":
            relation_state = "REFUTED"
        relations.append({
            "candidate_state": relation_state,
            "candidate_version": revision_version,
            "connective_state": connective_state,
            "context_scope_key": scope,
            "provisional": 1,
            "relation_id": f"{identity_token}_RELATION_1",
            "relation_kind": relation_kind,
            "source_proposition_id": first_id,
            "target_proposition_id": second_id,
        })

    presuppositions: list[dict[str, Any]] = []
    if candidate_kind in {
            "GENERATION", "PRESUPPOSITION_CANCELLATION",
            "PRESUPPOSITION_PROJECTION"}:
        status = (
            "CANCELLED" if candidate_kind == "PRESUPPOSITION_CANCELLATION"
            else "PROJECTED")
        presuppositions.append({
            "candidate_version": 1,
            "content_proposition_id": second_id,
            "dependency_relation_ids": [relations[0]["relation_id"]],
            "obligation_id": f"{identity_token}_PRESUPPOSITION_1",
            "status": status,
            "trigger_proposition_id": first_id,
        })

    topic_key = ""
    focus_key = ""
    contrast_set_keys: list[str] = []
    if information_status_family == "GIVEN":
        topic_key = f"{identity_token}_TOPIC"
    elif information_status_family == "NEW":
        focus_key = f"{identity_token}_FOCUS"
    elif information_status_family == "MIXED":
        topic_key = f"{identity_token}_TOPIC"
        focus_key = f"{identity_token}_FOCUS"
    elif information_status_family == "TOPIC_FOCUS":
        topic_key = f"{identity_token}_TOPIC"
        focus_key = f"{identity_token}_FOCUS"
        contrast_set_keys = [
            f"{identity_token}_ALTERNATIVE_A",
            f"{identity_token}_ALTERNATIVE_B",
        ]
    information = [{
        "candidate_version": 1,
        "contrast_set_keys": contrast_set_keys,
        "focus_key": focus_key,
        "information_id": f"{identity_token}_INFORMATION_1",
        "proposition_id": second_id,
        "status_family": information_status_family,
        "topic_key": topic_key,
    }]
    qud = [{
        "candidate_version": 1,
        "priority": 1,
        "provisional": 1,
        "question_key": f"{identity_token}_QUESTION",
        "qud_id": f"{identity_token}_QUD_1",
        "status": qud_family,
        "target_proposition_id": second_id,
    }]
    return {
        "discourse_relations": relations,
        "information_structure": information,
        "presupposition_obligations": presuppositions,
        "proposition_candidates": propositions,
        "qud_candidates": qud,
        "scope": scope,
    }


_SCENARIOS = (
    ("ELABORATION", "POSITIVE", "ELABORATION", "GIVEN", "ANSWERED", "ELABORATION_RELATION"),
    ("CONTRAST", "POSITIVE", "CONTRAST", "MIXED", "ANSWERED", "CONTRAST_RELATION"),
    ("CAUSE", "POSITIVE", "CAUSE", "NEW", "OPEN", "CAUSE_RELATION"),
    ("CONCESSION", "POSITIVE", "CONCESSION", "GIVEN", "OPEN", "CONCESSION_RELATION"),
    ("PRESUPPOSITION_PROJECTION", "POSITIVE", "ELABORATION", "MIXED", "ANSWERED", "PRESUPPOSITION_PROJECTION"),
    ("PRESUPPOSITION_CANCELLATION", "POSITIVE", "CONTRAST", "GIVEN", "ANSWERED", "PRESUPPOSITION_CANCELLATION"),
    ("TOPIC_FOCUS", "POSITIVE", "ELABORATION", "TOPIC_FOCUS", "OPEN", "TOPIC_FOCUS_MINIMAL_CONTRAST"),
    ("GIVEN_NEW", "POSITIVE", "CONTRAST", "MIXED", "ANSWERED", "GIVEN_NEW_STATUS"),
    ("QUD", "POSITIVE", "CAUSE", "GIVEN", "OPEN", "QUD_STATE"),
    ("NO_CONNECTIVE_BASELINE", "NEGATIVE", "NONE", "GIVEN", "ANSWERED", "NO_CONNECTIVE_REJECT"),
    ("WRONG_CONNECTIVE_BASELINE", "NEGATIVE", "CAUSE", "NEW", "ANSWERED", "WRONG_CONNECTIVE_REJECT"),
    ("AMBIGUOUS_RELATION", "AMBIGUOUS", "AMBIGUOUS", "MIXED", "OPEN", "AMBIGUOUS_RELATION_COMPETITION"),
    ("UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN", "DISCOURSE_UNKNOWN"),
    ("REVISION", "REVISION", "CAUSE", "NEW", "OPEN", "LOCAL_RELATION_REVISION"),
    ("RETENTION", "RETENTION", "CONCESSION", "GIVEN", "OPEN", "RETENTION_REVERIFY"),
    ("GENERATION", "GENERATION", "CONTRAST", "TOPIC_FOCUS", "ANSWERED", "REVERSE_GENERATION_ORDER_EXPLICITNESS"),
)

_EVALUATOR_AXIS_OVERRIDES = {
    1: ("NEW", "ANSWERED"),
    2: ("GIVEN", "OPEN"),
    3: ("MIXED", "ANSWERED"),
    4: ("NEW", "ANSWERED"),
    5: ("GIVEN", "OPEN"),
    6: ("TOPIC_FOCUS", "OPEN"),
    14: ("MIXED", "ANSWERED"),
    15: ("NEW", "ANSWERED"),
}

_TEACHER_TEXTS = (
    "档案补充了实验日期，这一信息展开了前句的记录。",
    "甲组先完成，乙组却仍在校验，两句形成对照。",
    "电源中断，因此记录器停止写入。",
    "虽然雨势加大，巡检仍按时完成。",
    "林岚又检查了闸门，这一说法暂存她此前检查过闸门的预设。",
    "林岚没有再次检查闸门，因为她此前并未检查，原预设被取消。",
    "谈到甲方案，是成本而不是速度决定了选择。",
    "该设备已经登记；它的新故障代码需要另行核验。",
    "下一步尚待回答的问题是由谁复核第二份记录。",
    "甲组完成校验。乙组提交记录。两句间没有足够证据选定篇章关系。",
    "记录器断电，然而它因此停止写入；错误连接词候选必须被驳回。",
    "灯熄灭后房间变暗，现有证据不能区分因果还是时间上的对照。",
    "两条记录的关系、信息状态和当前问题都未知。",
    "后文澄清：停机并非断电造成，原因关系候选改为驳回。",
    "只读重验：虽然雨势加大，巡检仍按时完成。",
    "语义输入包含两个有序命题、对照关系、投射预设、topic/focus 与已回答问题。",
)

_EVALUATOR_TEXTS = (
    "报告新增了采样地点，这一信息展开了上一条说明。",
    "主节点已恢复，备用节点却仍在诊断，两句形成对照。",
    "缓存损坏，因此读取任务停止。",
    "虽然道路封闭，配送仍按约完成。",
    "程澈又核对了编号，暂存他此前核对过编号的预设。",
    "程澈没有再次核对编号，因为此前并未核对，原预设被取消。",
    "谈到乙方案，是可靠性而不是价格决定了选择。",
    "该批次已经入库；它的新异常标签需要单独处理。",
    "当前开放问题是由谁确认第三份清单。",
    "主节点恢复。备用节点提交日志。不得仅凭相邻句选定篇章关系。",
    "缓存损坏，然而读取因此停止；不匹配连接词候选必须驳回。",
    "警报响起后值班员进场，证据不足以区分因果还是对照。",
    "当前关系、信息状态和待答问题均未知。",
    "后文澄清：停止读取并非缓存损坏造成，原因候选改为驳回。",
    "只读重验：虽然道路封闭，配送仍按约完成。",
    "语义输入含两个有序命题、对照关系、投射预设、topic/focus 与已回答问题。",
)

_TEACHER_GENERATION_SURFACE = (
    "谈到甲方案，系统已复核旧记录；不过新证据仍待确认。")
_EVALUATOR_GENERATION_SURFACE = (
    "谈到乙方案，节点已复核旧日志；不过新异常仍待确认。")


def build_default_discourse_information_seed_values(
        ) -> tuple[dict[str, Any], ...]:
    """构造双 owner LC-07 原创课程，不执行训练或 runtime 判定。"""
    rows: list[dict[str, Any]] = []
    logical_order = 0
    for owner, texts in (("T", _TEACHER_TEXTS), ("E", _EVALUATOR_TEXTS)):
        label_owner = "teacher" if owner == "T" else "evaluator"
        split = "train" if owner == "T" else "held_out"
        ids = {
            index: (
                f"{owner}-lc07-{index:02d}-"
                f"{scenario[0].lower().replace('_', '-')}")
            for index, scenario in enumerate(_SCENARIOS, start=1)
        }
        for index, scenario in enumerate(_SCENARIOS, start=1):
            logical_order += 1
            (candidate_kind, sample_family, relation_family,
             information_family, qud_family, dimension) = scenario
            if owner == "E" and index in _EVALUATOR_AXIS_OVERRIDES:
                information_family, qud_family = (
                    _EVALUATOR_AXIS_OVERRIDES[index])
            shared_revision = index in {3, 14}
            revision_version = 2 if index == 14 else 1
            components = _scenario_components(
                owner,
                index,
                candidate_kind,
                relation_family,
                information_family,
                qud_family,
                revision_version=revision_version,
                shared_revision_identity=shared_revision,
            )
            supersedes = ids[3] if index == 14 else ""
            retention_anchor = ids[4] if index == 15 else ""
            expected_state = {
                "AMBIGUOUS": "CONFLICT",
                "NEGATIVE": "FALSE",
                "UNKNOWN": "UNKNOWN",
            }.get(sample_family, "TRUE")
            accepted_surfaces: list[str] = []
            if expected_state == "TRUE":
                if candidate_kind == "GENERATION":
                    accepted_surfaces = [
                        _TEACHER_GENERATION_SURFACE
                        if owner == "T"
                        else _EVALUATOR_GENERATION_SURFACE]
                else:
                    accepted_surfaces = [texts[index - 1]]
            generation_hidden = int(candidate_kind == "GENERATION")
            known_ids = [
                *[item["proposition_id"]
                  for item in components["proposition_candidates"]],
                *[item["relation_id"]
                  for item in components["discourse_relations"]],
                *[item["obligation_id"]
                  for item in components["presupposition_obligations"]],
                components["information_structure"][0]["information_id"],
                components["qud_candidates"][0]["qud_id"],
            ]
            family_suffix = (
                "REVISION_CHAIN" if index in {3, 14} else candidate_kind)
            objective_keys = {
                "AMBIGUOUS_RELATION": ("CROSS_CONTEXT_CONSISTENCY",),
                "GENERATION": ("GENERATION_ADOPTION", "GENERATION_FAILURE"),
                "QUD": ("NEXT_DISCOURSE_UNIT",),
                "REVISION": ("CONTROLLED_PERTURBATION",),
                "UNKNOWN": ("CONTROLLED_PERTURBATION",),
            }.get(candidate_kind, ("ORDER_RECOVERY",))
            scope = components.pop("scope")
            baseline = {
                10: "NO_CONNECTIVE_ONLY",
                11: "WRONG_CONNECTIVE_ONLY",
            }.get(index, "TYPED_DISCOURSE_OBJECTS_PRESENT")
            relation_ids = sorted(
                item["relation_id"]
                for item in components["discourse_relations"])
            proposition_ids = sorted(
                item["proposition_id"]
                for item in components["proposition_candidates"])
            rows.append({
                "baseline_kind": baseline,
                "candidate_kind": candidate_kind,
                "discourse_relations": components["discourse_relations"],
                "evaluation_dimension": dimension,
                "expected_payload": {
                    "accepted": int(expected_state == "TRUE"),
                    "accepted_surfaces": accepted_surfaces,
                    "analysis_key": f"{owner}_LC07_{dimension}",
                    "reason_code": "COURSE_LABEL_ONLY_NOT_RUNTIME_PASS",
                },
                "expected_state": expected_state,
                "family": f"{owner}_LC07_{family_suffix}",
                "generation_constraint": {
                    "direction": (
                        "SEMANTICS_TO_SURFACE" if generation_hidden
                        else "SURFACE_TO_CANDIDATES"),
                    "explicitness_preservation_required": generation_hidden,
                    "given_new_preservation_required": generation_hidden,
                    "input_candidate_ids": known_ids,
                    "output_surface_hidden": generation_hidden,
                    "presupposition_preservation_required": generation_hidden,
                    "proposition_order_preservation_required": generation_hidden,
                    "qud_preservation_required": generation_hidden,
                    "relation_preservation_required": generation_hidden,
                    "topic_focus_preservation_required": generation_hidden,
                },
                "information_status_family": information_family,
                "information_structure": components["information_structure"],
                "label_owner": label_owner,
                "license_id": LICENSE_ID,
                "logical_order": logical_order,
                "objective_keys": sorted(objective_keys),
                "observed_text": texts[index - 1],
                "perturbation_kind": f"LC07_{candidate_kind}",
                "presupposition_obligations": (
                    components["presupposition_obligations"]),
                "proposition_candidates": components["proposition_candidates"],
                "qud_candidates": components["qud_candidates"],
                "qud_family": qud_family,
                "relation_family": relation_family,
                "resource_budget": {
                    "max_dependency_edges": 4,
                    "max_information_states": 1,
                    "max_output_units": 160,
                    "max_presuppositions": 1,
                    "max_propositions": 2,
                    "max_qud": 1,
                    "max_relations": 2,
                },
                "retention_anchor_id": retention_anchor,
                "revision_receipt": {
                    "affected_relation_ids": relation_ids if index == 14 else [],
                    "candidate_version": revision_version,
                    "dependency_keys": (
                        ["DISCOURSE_RELATION"] if index == 14 else []),
                    "raw_observation_preserved": 1,
                    "revision_scope": (
                        "DISCOURSE_RELATION_ONLY" if index == 14 else "NONE"),
                    "supersedes_seed_id": supersedes,
                    "unaffected_proposition_ids": (
                        proposition_ids if index == 14 else []),
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
                    "genre": "DISCOURSE",
                    "language": "zh",
                    "register": "NEUTRAL",
                    "script": "HAN",
                },
                "template_family": f"{owner}_LC07_TEMPLATE_{candidate_kind}",
            })
    return tuple(rows)


def default_discourse_information_sample_bytes() -> bytes:
    """返回规范 LC-07 JSONL，供正式 sample 重建和字节审计。"""
    return b"".join(
        canonical_json_line(value)
        for value in build_default_discourse_information_seed_values())


__all__ = [
    "COURSE_MANIFEST_PATH",
    "EVALUATOR_DIMENSIONS",
    "FORMAL_ARTIFACT_RELATIVE_ROOT",
    "PACK_NAME",
    "PAYLOAD_KIND",
    "AuthoredDiscourseInformationCourseError",
    "AuthoredDiscourseInformationSeed",
    "DiscourseInformationPayloadAudit",
    "build_default_discourse_information_seed_values",
    "build_discourse_information_course_manifest",
    "compile_authored_discourse_information_course",
    "default_discourse_information_sample_bytes",
    "read_authored_discourse_information_seeds",
    "validate_discourse_information_payload",
]
