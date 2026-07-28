"""LC-02 生产性形态与构词的原创 CC0 课程编译器。"""
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
PACK_NAME = "AUTHORED_CC0_V1--CC0-1.0--lc02-morphology-v1"
STAGE = "W-02"
SUBSTAGE = "LC02_MORPHOLOGY_WORD_FORM"
PAYLOAD_KIND = "MorphologyCandidateV1"
COURSE_MANIFEST_ARTIFACT_VERSION = "LC-02-morphology-course-v1"
COURSE_MANIFEST_PATH = Path(
    "data/ph2/manifests/lc02_morphology_course_v1.json")
FORMAL_ARTIFACT_RELATIVE_ROOT = (
    "ph2_dataset_artifacts/d02_language_courses_v1")

CANDIDATE_KINDS = (
    "AFFIXATION",
    "COMPOUND",
    "DICTIONARY_REPLAY",
    "EXCEPTION",
    "GENERATION",
    "REDUPLICATION",
    "RETENTION",
    "SEGMENTATION",
    "UNKNOWN",
)
UNIT_KINDS = (
    "AFFIX",
    "COMPONENT",
    "CONSTRUCTION",
    "EXCEPTION_FORM",
    "REDUPLICANT",
    "STEM",
)
RELATION_KINDS = (
    "ATTACHES_AFFIX",
    "COMPOUND_COMPONENT",
    "EXCEPTION_TO",
    "FILLS_SLOT",
    "HAS_STEM",
    "REDUPLICATES",
)
SLOT_KEYS = (
    "COMPOUND_HEAD",
    "COMPOUND_MODIFIER",
    "DERIVATIONAL_PREFIX",
    "DERIVATIONAL_SUFFIX",
    "LEXICAL_EXCEPTION",
    "REDUPLICANT",
    "SEGMENT",
    "SURFACE_FORM",
)
BASELINE_KINDS = ("DICTIONARY_REPLAY_ONLY", "STRUCTURAL_COMPOSITION")
EVALUATOR_DIMENSIONS = (
    "AMBIGUOUS_SEGMENTATION",
    "DICTIONARY_REPLAY_REJECT",
    "EXCEPTION_SCOPE",
    "HELD_OUT_STEM_CONSTRUCTION",
    "LANGUAGE_SCOPE",
    "MORPHOLOGY_RELATION_INTEGRITY",
    "RETENTION_REVERIFY",
    "REVERSE_GENERATION",
)
VERIFIER_NE_CONDITIONS = (
    "CAPABILITY_LEARNED_REQUESTED",
    "GENERATION_RESULT_NOT_EXECUTED",
    "NO_EVALUATOR_LABEL",
    "OUT_OF_LANGUAGE_SCOPE",
    "SEMANTIC_TRUTH_REQUESTED",
)
RETENTION_PROTOCOLS = (
    "A_TO_LC02_REVERIFY_A",
    "DUMP_RESUME_REQUIRED_AT_RUNTIME",
    "SOURCE_WITHDRAWAL_LOCAL_ONLY",
)
COMBINATION_AXES = (
    "CONSTRUCTION_FAMILY",
    "STEM_FAMILY",
    "STEM_X_CONSTRUCTION",
)
ABLATION_KEYS = (
    "DICTIONARY_REPLAY_ONLY",
    "REMOVE_GENERATION_CONSTRAINT",
    "REMOVE_SLOT_RELATION",
    "REMOVE_STEM_IDENTITY",
)
PAYLOAD_KEYS = (
    "analysis_units",
    "baseline_kind",
    "candidate_group",
    "candidate_id",
    "candidate_kind",
    "construction_key",
    "generation_constraint",
    "language_scope",
    "morphology_relations",
    "objective_keys",
    "observed_surface",
    "retention_anchor_id",
    "sample_family",
    "selection_state",
    "split_identity",
)

_SEED_FIELDS = {
    "analysis_units", "baseline_kind", "candidate_group", "candidate_id",
    "candidate_kind", "construction_family", "construction_key",
    "evaluation_dimension", "expected_payload", "expected_state", "family",
    "generation_constraint", "label_owner", "language_scope", "license_id",
    "logical_order", "objective_keys", "observed_text", "perturbation_kind",
    "relations", "retention_anchor_id", "sample_family", "sample_role",
    "seed_id", "split", "stem_family", "supersedes_seed_id",
    "template_family",
}
_UNIT_FIELDS = {"end", "start", "surface", "unit_id", "unit_kind"}
_RELATION_FIELDS = {
    "order_index", "relation_kind", "slot_key", "source_unit_id",
    "target_unit_id",
}
_GENERATION_FIELDS = {
    "constraint_key", "direction", "input_unit_ids", "output_surface_hidden",
}
_LANGUAGE_FIELDS = {"language", "scope_key", "script"}
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
_REQUIRED_CANDIDATE_KINDS = set(CANDIDATE_KINDS)
_REQUIRED_RELATION_KINDS = set(RELATION_KINDS)


class AuthoredMorphologyCourseError(RuntimeError):
    """LC-02 seed、typed payload 或组合 split 不满足冻结合同。"""


def _exact_keys(value: Any, keys: set[str], *, where: str) -> dict[str, Any]:
    """要求 JSON 对象字段精确一致。"""
    if not isinstance(value, dict) or set(value) != keys:
        raise AuthoredMorphologyCourseError(f"{where} 字段集合非法")
    return value


def _text(value: Any, *, where: str, allow_empty: bool = False) -> str:
    """要求文本无首尾空白；仅显式位置可为空。"""
    if not isinstance(value, str) or value.strip() != value:
        raise AuthoredMorphologyCourseError(f"{where} 必须是规范文本")
    if not allow_empty and not value:
        raise AuthoredMorphologyCourseError(f"{where} 不能为空")
    return value


def _integer(value: Any, *, where: str, nonnegative: bool = True) -> int:
    """要求严格整数并按需禁止负值。"""
    if type(value) is not int or (nonnegative and value < 0):
        raise AuthoredMorphologyCourseError(f"{where} 必须是严格整数")
    return value


def _flag(value: Any, *, where: str) -> int:
    """要求字段为严格 0/1。"""
    if type(value) is not int or value not in {0, 1}:
        raise AuthoredMorphologyCourseError(f"{where} 必须是 0/1")
    return value


def _string_tuple(
        value: Any,
        *,
        where: str,
        sorted_unique: bool = False,
        allow_empty: bool = False) -> tuple[str, ...]:
    """把 JSON 文本数组恢复为 tuple 并校验唯一性。"""
    if not isinstance(value, list) or (not value and not allow_empty):
        raise AuthoredMorphologyCourseError(f"{where} 必须是文本数组")
    result = tuple(_text(item, where=where) for item in value)
    if len(result) != len(set(result)):
        raise AuthoredMorphologyCourseError(f"{where} 不得重复")
    if sorted_unique and tuple(sorted(result)) != result:
        raise AuthoredMorphologyCourseError(f"{where} 必须排序")
    return result


def _sha256_text(value: str) -> str:
    """计算 UTF-8 文本的 SHA-256。"""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_expected_payload(
        payload: CanonicalJsonObject,
        *,
        expected_state: str) -> None:
    """校验私有 label 的采用状态与可接受表层集合。"""
    raw = _exact_keys(payload.to_value(), _EXPECTED_FIELDS, where="expected payload")
    accepted = _flag(raw["accepted"], where="expected accepted")
    surfaces = _string_tuple(
        raw["accepted_surfaces"], where="accepted_surfaces", allow_empty=True)
    _text(raw["analysis_key"], where="expected analysis_key")
    _text(raw["reason_code"], where="expected reason_code")
    if expected_state == "TRUE" and (accepted != 1 or not surfaces):
        raise AuthoredMorphologyCourseError("TRUE label 必须给出私有合法表层")
    if expected_state != "TRUE" and (accepted != 0 or surfaces):
        raise AuthoredMorphologyCourseError("非 TRUE label 不得给出采用表层")


@dataclass(frozen=True)
class MorphologyPayloadAudit:
    """返回形态 payload 的结构关系、组合轴和生成隐藏事实。"""

    relation_kinds: tuple[str, ...]
    combination_key: str
    target_hidden: int
    dictionary_replay_only: int


def validate_morphology_payload(
        payload: CanonicalJsonObject | dict[str, Any]) -> MorphologyPayloadAudit:
    """校验语言作用域、单位/span、关系、组合 split 和生成约束。"""
    value = payload.to_value() if isinstance(payload, CanonicalJsonObject) else payload
    raw = _exact_keys(value, set(PAYLOAD_KEYS), where="morphology payload")
    candidate_kind = _text(raw["candidate_kind"], where="candidate_kind")
    if candidate_kind not in CANDIDATE_KINDS:
        raise AuthoredMorphologyCourseError("candidate_kind 未登记")
    baseline_kind = _text(raw["baseline_kind"], where="baseline_kind")
    if baseline_kind not in BASELINE_KINDS:
        raise AuthoredMorphologyCourseError("baseline_kind 未登记")
    _text(raw["candidate_group"], where="candidate_group")
    _text(raw["candidate_id"], where="candidate_id")
    _text(raw["construction_key"], where="construction_key")
    if raw["selection_state"] != "UNSELECTED":
        raise AuthoredMorphologyCourseError("Observation 不得私选形态候选")
    if raw["sample_family"] not in SAMPLE_FAMILIES:
        raise AuthoredMorphologyCourseError("sample_family 未登记")

    language = _exact_keys(raw["language_scope"], _LANGUAGE_FIELDS, where="language_scope")
    if language["language"] != "zh" or language["script"] != "HAN":
        raise AuthoredMorphologyCourseError("LC-02 首阶段只冻结 zh/HAN 作用域")
    _text(language["scope_key"], where="language scope_key")

    observed = _exact_keys(raw["observed_surface"], {
        "append_only", "sha256", "target_hidden", "text",
    }, where="observed_surface")
    observed_text = _text(observed["text"], where="observed text")
    if _flag(observed["append_only"], where="observed append_only") != 1:
        raise AuthoredMorphologyCourseError("observed surface 必须 append-only")
    if observed["sha256"] != _sha256_text(observed_text):
        raise AuthoredMorphologyCourseError("observed surface SHA-256 漂移")
    target_hidden = _flag(observed["target_hidden"], where="target_hidden")

    units_value = raw["analysis_units"]
    if not isinstance(units_value, list) or not units_value:
        raise AuthoredMorphologyCourseError("analysis_units 不能为空")
    units: dict[str, dict[str, Any]] = {}
    for item in units_value:
        unit = _exact_keys(item, _UNIT_FIELDS, where="analysis unit")
        unit_id = _text(unit["unit_id"], where="unit_id")
        if unit_id in units:
            raise AuthoredMorphologyCourseError("unit_id 重复")
        kind = _text(unit["unit_kind"], where="unit_kind")
        if kind not in UNIT_KINDS:
            raise AuthoredMorphologyCourseError("unit_kind 未登记")
        start = _integer(unit["start"], where="unit start")
        end = _integer(unit["end"], where="unit end")
        surface = _text(unit["surface"], where="unit surface")
        if not (start < end <= len(observed_text)):
            raise AuthoredMorphologyCourseError("analysis unit span 越界")
        if observed_text[start:end] != surface:
            raise AuthoredMorphologyCourseError("analysis unit span/surface 漂移")
        units[unit_id] = unit

    relations_value = raw["morphology_relations"]
    if not isinstance(relations_value, list):
        raise AuthoredMorphologyCourseError("morphology_relations 必须是数组")
    relation_kinds: list[str] = []
    orders: list[int] = []
    for item in relations_value:
        relation = _exact_keys(item, _RELATION_FIELDS, where="morphology relation")
        kind = _text(relation["relation_kind"], where="relation_kind")
        if kind not in RELATION_KINDS:
            raise AuthoredMorphologyCourseError("relation_kind 未登记")
        source_id = _text(relation["source_unit_id"], where="source_unit_id")
        target_id = _text(relation["target_unit_id"], where="target_unit_id")
        if source_id not in units or target_id not in units:
            raise AuthoredMorphologyCourseError("relation 引用未知 unit")
        slot_key = _text(relation["slot_key"], where="slot_key")
        if slot_key not in SLOT_KEYS:
            raise AuthoredMorphologyCourseError("slot_key 未登记")
        order = _integer(relation["order_index"], where="relation order")
        if order <= 0:
            raise AuthoredMorphologyCourseError("relation order 必须为正")
        relation_kinds.append(kind)
        orders.append(order)
    if orders != list(range(1, len(orders) + 1)):
        raise AuthoredMorphologyCourseError("relation order 必须从 1 连续递增")

    if baseline_kind == "DICTIONARY_REPLAY_ONLY":
        if candidate_kind != "DICTIONARY_REPLAY" or relations_value:
            raise AuthoredMorphologyCourseError("词典回放基线不得伪造结构关系")
    else:
        if candidate_kind == "DICTIONARY_REPLAY":
            raise AuthoredMorphologyCourseError("词典回放不得冒充结构组合")
        required = {"FILLS_SLOT", "HAS_STEM"}
        if not required.issubset(relation_kinds):
            raise AuthoredMorphologyCourseError("形态候选缺 stem/slot 承重关系")
        required_by_kind = {
            "AFFIXATION": "ATTACHES_AFFIX",
            "COMPOUND": "COMPOUND_COMPONENT",
            "EXCEPTION": "EXCEPTION_TO",
            "REDUPLICATION": "REDUPLICATES",
            "SEGMENTATION": "COMPOUND_COMPONENT",
        }
        relation = required_by_kind.get(candidate_kind)
        if relation is not None and relation not in relation_kinds:
            raise AuthoredMorphologyCourseError("candidate kind 缺专属形态关系")

    generation = _exact_keys(
        raw["generation_constraint"], _GENERATION_FIELDS,
        where="generation_constraint")
    _text(generation["constraint_key"], where="constraint_key")
    direction = _text(generation["direction"], where="generation direction")
    if direction not in {"BIDIRECTIONAL", "GENERATION", "UNDERSTANDING"}:
        raise AuthoredMorphologyCourseError("generation direction 未登记")
    input_ids = _string_tuple(
        generation["input_unit_ids"], where="input_unit_ids")
    if any(item not in units for item in input_ids):
        raise AuthoredMorphologyCourseError("generation input 引用未知 unit")
    hidden = _flag(
        generation["output_surface_hidden"],
        where="output_surface_hidden")
    if hidden != target_hidden:
        raise AuthoredMorphologyCourseError("生成约束与 observed hidden 标志不一致")
    if candidate_kind == "GENERATION":
        if direction != "GENERATION" or hidden != 1:
            raise AuthoredMorphologyCourseError("反向生成必须隐藏目标表层")
    elif hidden:
        raise AuthoredMorphologyCourseError("非生成样本不得隐藏 observed target")

    split_identity = _exact_keys(raw["split_identity"], {
        "combination_key", "construction_family", "isolation_axis",
        "stem_family",
    }, where="split_identity")
    stem_family = _text(split_identity["stem_family"], where="stem_family")
    construction_family = _text(
        split_identity["construction_family"], where="construction_family")
    combination_key = _text(
        split_identity["combination_key"], where="combination_key")
    if combination_key != f"{stem_family}::{construction_family}":
        raise AuthoredMorphologyCourseError("stem×construction 组合键漂移")
    if split_identity["isolation_axis"] != "STEM_X_CONSTRUCTION":
        raise AuthoredMorphologyCourseError("组合 held-out 轴未冻结")

    objective_keys = tuple(raw["objective_keys"])
    if objective_keys != tuple(sorted(set(objective_keys))):
        raise AuthoredMorphologyCourseError("objective_keys 必须排序去重")
    if not objective_keys or any(
            item not in LANGUAGE_OBJECTIVE_KEYS for item in objective_keys):
        raise AuthoredMorphologyCourseError("objective_keys 未登记")
    anchor = _text(
        raw["retention_anchor_id"], where="retention_anchor_id",
        allow_empty=True)
    if raw["sample_family"] == "RETENTION" and not anchor:
        raise AuthoredMorphologyCourseError("retention 必须绑定旧 anchor")
    if raw["sample_family"] != "RETENTION" and anchor:
        raise AuthoredMorphologyCourseError("非 retention 不得绑定 anchor")
    return MorphologyPayloadAudit(
        tuple(sorted(set(relation_kinds))),
        combination_key,
        target_hidden,
        int(baseline_kind == "DICTIONARY_REPLAY_ONLY"),
    )


@dataclass(frozen=True)
class AuthoredMorphologySeed:
    """一个语言作用域明确、未私选的 LC-02 形态候选 seed。"""

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
    language_scope: CanonicalJsonObject
    analysis_units: tuple[CanonicalJsonObject, ...]
    relations: tuple[CanonicalJsonObject, ...]
    generation_constraint: CanonicalJsonObject
    stem_family: str
    construction_family: str
    construction_key: str
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
        """校验 seed owner、四态、许可和 typed payload。"""
        for name, value in (
                ("seed_id", self.seed_id), ("family", self.family),
                ("template_family", self.template_family),
                ("candidate_id", self.candidate_id),
                ("candidate_group", self.candidate_group),
                ("stem_family", self.stem_family),
                ("construction_family", self.construction_family),
                ("construction_key", self.construction_key),
                ("perturbation_kind", self.perturbation_kind)):
            _text(value, where=name)
        if self.label_owner not in {"teacher", "evaluator"}:
            raise AuthoredMorphologyCourseError("label_owner 非法")
        expected_split = "train" if self.label_owner == "teacher" else "held_out"
        if self.split != expected_split:
            raise AuthoredMorphologyCourseError("label_owner 与 split 不一致")
        if self.sample_family not in SAMPLE_FAMILIES:
            raise AuthoredMorphologyCourseError("sample_family 未登记")
        if self.sample_role not in SAMPLE_ROLES:
            raise AuthoredMorphologyCourseError("sample_role 未登记")
        if self.candidate_kind not in CANDIDATE_KINDS:
            raise AuthoredMorphologyCourseError("candidate_kind 未登记")
        _text(self.observed_text, where="observed_text")
        if (not isinstance(self.language_scope, CanonicalJsonObject)
                or not isinstance(self.generation_constraint, CanonicalJsonObject)):
            raise AuthoredMorphologyCourseError("scope/generation contract 类型非法")
        if (not isinstance(self.analysis_units, tuple) or not self.analysis_units
                or any(not isinstance(item, CanonicalJsonObject)
                       for item in self.analysis_units)):
            raise AuthoredMorphologyCourseError("analysis_units 类型非法")
        if (not isinstance(self.relations, tuple)
                or any(not isinstance(item, CanonicalJsonObject)
                       for item in self.relations)):
            raise AuthoredMorphologyCourseError("relations 类型非法")
        if self.baseline_kind not in BASELINE_KINDS:
            raise AuthoredMorphologyCourseError("baseline_kind 未登记")
        if self.objective_keys != tuple(sorted(set(self.objective_keys))):
            raise AuthoredMorphologyCourseError("objective_keys 必须排序去重")
        if not self.objective_keys or any(
                item not in LANGUAGE_OBJECTIVE_KEYS for item in self.objective_keys):
            raise AuthoredMorphologyCourseError("objective_keys 未登记")
        if self.expected_state not in EXPECTED_STATES:
            raise AuthoredMorphologyCourseError("expected_state 非四态")
        if not isinstance(self.expected_payload, CanonicalJsonObject):
            raise AuthoredMorphologyCourseError("expected_payload 类型非法")
        _validate_expected_payload(
            self.expected_payload, expected_state=self.expected_state)
        if self.evaluation_dimension not in EVALUATOR_DIMENSIONS:
            raise AuthoredMorphologyCourseError("evaluation_dimension 未登记")
        _text(
            self.supersedes_seed_id,
            where="supersedes_seed_id",
            allow_empty=True)
        _text(
            self.retention_anchor_id,
            where="retention_anchor_id",
            allow_empty=True)
        if type(self.logical_order) is not int or self.logical_order <= 0:
            raise AuthoredMorphologyCourseError("logical_order 必须为正严格整数")
        if self.license_id != LICENSE_ID:
            raise AuthoredMorphologyCourseError("原创形态课程必须为 CC0-1.0")
        validate_morphology_payload(self.observation_payload())

    def observation_payload(self) -> CanonicalJsonObject:
        """构造学生可见形态候选，不包含 expected 或合法输出集合。"""
        generation = self.generation_constraint.to_value()
        return CanonicalJsonObject.from_value({
            "analysis_units": [item.to_value() for item in self.analysis_units],
            "baseline_kind": self.baseline_kind,
            "candidate_group": self.candidate_group,
            "candidate_id": self.candidate_id,
            "candidate_kind": self.candidate_kind,
            "construction_key": self.construction_key,
            "generation_constraint": generation,
            "language_scope": self.language_scope.to_value(),
            "morphology_relations": [item.to_value() for item in self.relations],
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
            "split_identity": {
                "combination_key": (
                    f"{self.stem_family}::{self.construction_family}"),
                "construction_family": self.construction_family,
                "isolation_axis": "STEM_X_CONSTRUCTION",
                "stem_family": self.stem_family,
            },
        })

    def compiled_seed(self) -> AuthoredCompiledSeed:
        """映射到既有 Source/Observation/Teacher/Evaluator 四类记录。"""
        payload = self.observation_payload()
        audit = validate_morphology_payload(payload)
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
                self.construction_key,
            ),
            (self.candidate_group, self.observed_text),
            (
                self.candidate_kind, self.sample_family,
                audit.combination_key, *audit.relation_kinds,
            ),
            self.evaluation_dimension,
        )


def _seed_from_value(value: dict[str, Any]) -> AuthoredMorphologySeed:
    """从精确 JSON object 恢复形态课程 seed。"""
    raw = _exact_keys(value, _SEED_FIELDS, where="morphology seed")
    units_value = raw["analysis_units"]
    if not isinstance(units_value, list) or not units_value:
        raise AuthoredMorphologyCourseError("analysis_units 不能为空")
    relations_value = raw["relations"]
    if not isinstance(relations_value, list):
        raise AuthoredMorphologyCourseError("relations 必须是数组")
    expected = raw["expected_payload"]
    if not isinstance(expected, dict):
        raise AuthoredMorphologyCourseError("expected_payload 必须是对象")
    return AuthoredMorphologySeed(
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
            raw["language_scope"], _LANGUAGE_FIELDS, where="language_scope")),
        tuple(CanonicalJsonObject.from_value(
            _exact_keys(item, _UNIT_FIELDS, where="analysis unit"))
              for item in units_value),
        tuple(CanonicalJsonObject.from_value(
            _exact_keys(item, _RELATION_FIELDS, where="relation"))
              for item in relations_value),
        CanonicalJsonObject.from_value(_exact_keys(
            raw["generation_constraint"], _GENERATION_FIELDS,
            where="generation_constraint")),
        _text(raw["stem_family"], where="stem_family"),
        _text(raw["construction_family"], where="construction_family"),
        _text(raw["construction_key"], where="construction_key"),
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


def read_authored_morphology_seeds(
        path: str | Path) -> tuple[AuthoredMorphologySeed, ...]:
    """严格读取 JSONL，并核七类、关系全集、组合 held-out 和负基线。"""
    try:
        payload = Path(path).read_bytes()
    except OSError as error:
        raise AuthoredMorphologyCourseError("morphology sample 无法读取") from error
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise AuthoredMorphologyCourseError("morphology sample 换行非法")
    seeds: list[AuthoredMorphologySeed] = []
    try:
        for line in payload.splitlines(keepends=True):
            value = parse_canonical_json_bytes(line[:-1], require_object=True)
            assert isinstance(value, dict)
            if canonical_json_line(value) != line:
                raise AuthoredMorphologyCourseError("morphology seed 非规范 JSON")
            seeds.append(_seed_from_value(value))
    except AuthoredMorphologyCourseError:
        raise
    except Exception as error:
        raise AuthoredMorphologyCourseError("morphology seed 损坏") from error
    if not seeds:
        raise AuthoredMorphologyCourseError("morphology seeds 不能为空")
    identifiers = [item.seed_id for item in seeds]
    candidate_ids = [item.candidate_id for item in seeds]
    orders = [item.logical_order for item in seeds]
    if len(identifiers) != len(set(identifiers)):
        raise AuthoredMorphologyCourseError("seed_id 重复")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise AuthoredMorphologyCourseError("candidate_id 重复")
    if orders != sorted(orders) or len(orders) != len(set(orders)):
        raise AuthoredMorphologyCourseError("logical_order 必须严格递增")
    index = {item.seed_id: item for item in seeds}
    for seed in seeds:
        expected_role = _ROLE_BY_FAMILY.get(seed.sample_family)
        if expected_role is not None and seed.sample_role != expected_role:
            raise AuthoredMorphologyCourseError("sample family/role 不一致")
        if seed.sample_family == "GENERATION" and seed.sample_role not in {
                "support", "read_only_probe", "refute"}:
            raise AuthoredMorphologyCourseError("generation sample_role 非法")
        expected_state = _STATE_BY_FAMILY.get(seed.sample_family)
        if expected_state is not None and seed.expected_state != expected_state:
            raise AuthoredMorphologyCourseError("sample family/state 不一致")
        if seed.sample_family == "GENERATION" and seed.expected_state not in {
                "TRUE", "FALSE"}:
            raise AuthoredMorphologyCourseError("generation state 非法")
        if seed.sample_family == "REVISION":
            target = index.get(seed.supersedes_seed_id)
            if (target is None or target.logical_order >= seed.logical_order
                    or target.family != seed.family or target.split != seed.split):
                raise AuthoredMorphologyCourseError(
                    "revision 必须 supersede 同族更早 seed")
        elif seed.supersedes_seed_id:
            raise AuthoredMorphologyCourseError("非 revision 不得 supersede")
        if seed.sample_family == "RETENTION":
            anchor = index.get(seed.retention_anchor_id)
            if (anchor is None or anchor.logical_order >= seed.logical_order
                    or anchor.split != seed.split):
                raise AuthoredMorphologyCourseError(
                    "retention anchor 必须是同 split 更早 seed")
        elif seed.retention_anchor_id:
            raise AuthoredMorphologyCourseError("非 retention 不得绑定 anchor")

    teacher = tuple(item for item in seeds if item.label_owner == "teacher")
    evaluator = tuple(item for item in seeds if item.label_owner == "evaluator")
    for owner, owner_seeds in (("teacher", teacher), ("evaluator", evaluator)):
        if {item.sample_family for item in owner_seeds} != set(SAMPLE_FAMILIES):
            raise AuthoredMorphologyCourseError(f"{owner} 未覆盖七类 sample family")
        if {item.candidate_kind for item in owner_seeds} != _REQUIRED_CANDIDATE_KINDS:
            raise AuthoredMorphologyCourseError(f"{owner} 未覆盖完整形态候选族")
        relation_kinds = {
            kind for item in owner_seeds
            for kind in validate_morphology_payload(
                item.observation_payload()).relation_kinds
        }
        if relation_kinds != _REQUIRED_RELATION_KINDS:
            raise AuthoredMorphologyCourseError(f"{owner} 未覆盖完整形态关系族")
        ambiguous: dict[str, set[tuple[tuple[str, str], ...]]] = {}
        for item in owner_seeds:
            if item.sample_family != "AMBIGUOUS":
                continue
            signature = tuple(
                (unit.to_value()["unit_kind"], unit.to_value()["surface"])
                for unit in item.analysis_units)
            ambiguous.setdefault(item.candidate_group, set()).add(signature)
        if not ambiguous or max(len(value) for value in ambiguous.values()) < 2:
            raise AuthoredMorphologyCourseError(
                f"{owner} 歧义切分未保留多个候选")
        replay = tuple(item for item in owner_seeds
                       if item.baseline_kind == "DICTIONARY_REPLAY_ONLY")
        if len(replay) != 1 or replay[0].expected_state != "FALSE":
            raise AuthoredMorphologyCourseError(
                f"{owner} 词典回放负基线未冻结")
    if ({item.family for item in teacher} & {item.family for item in evaluator}
            or {item.template_family for item in teacher}
            & {item.template_family for item in evaluator}):
        raise AuthoredMorphologyCourseError("teacher/evaluator family/template 泄漏")

    teacher_stems = {item.stem_family for item in teacher}
    teacher_constructions = {item.construction_family for item in teacher}
    teacher_pairs = {
        (item.stem_family, item.construction_family) for item in teacher
    }
    held_out_recombinations = {
        (item.stem_family, item.construction_family) for item in evaluator
        if (item.stem_family in teacher_stems
            and item.construction_family in teacher_constructions
            and (item.stem_family, item.construction_family) not in teacher_pairs)
    }
    if len(held_out_recombinations) < 2:
        raise AuthoredMorphologyCourseError(
            "held-out stem×construction 组合不足")
    if {item.evaluation_dimension for item in evaluator} != set(
            EVALUATOR_DIMENSIONS):
        raise AuthoredMorphologyCourseError("独立 evaluator 维度未列全")
    return tuple(seeds)


def _course_spec() -> AuthoredCourseSpec:
    """返回复用既有 Dataset publisher 的 LC-02 课程 spec。"""
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
        "authored-morphology-seed-v1",
        "urn:pure-integer-ai:ph2:authored-morphology-v1",
        "Pure Integer AI PH2 authored morphology seed",
        "MORPHOLOGY_LABEL",
        "lc02-morphology",
        240,
    )


def compile_authored_morphology_course(
        sample_path: str | Path,
        release_root: str | Path) -> AuthoredCourseBuild:
    """编译 LC-02 pack，不运行训练、teacher 或学习状态写入。"""
    seeds = read_authored_morphology_seeds(sample_path)
    return publish_authored_course(
        tuple(item.compiled_seed() for item in seeds),
        sample_path,
        release_root,
        _course_spec(),
    )


def build_morphology_course_manifest(
        sample_path: str | Path,
        build: AuthoredCourseBuild,
        *,
        artifact_relative_root: str = FORMAL_ARTIFACT_RELATIVE_ROOT,
        ) -> CapabilityCourseManifest:
    """汇合 LC-02 sample、pack、组合轴、verifier、消融和零执行状态。"""
    sample = Path(sample_path)
    seeds = read_authored_morphology_seeds(sample)
    teacher = tuple(item for item in seeds if item.label_owner == "teacher")
    evaluator = tuple(item for item in seeds if item.label_owner == "evaluator")
    return CapabilityCourseManifest(
        1,
        COURSE_MANIFEST_ARTIFACT_VERSION,
        "COURSE_FROZEN",
        "NOT_STARTED",
        ("LC-02",),
        ("MORPHOLOGY_WORD_FORM",),
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
    "ABLATION_KEYS",
    "BASELINE_KINDS",
    "COMBINATION_AXES",
    "COURSE_MANIFEST_ARTIFACT_VERSION",
    "COURSE_MANIFEST_PATH",
    "EVALUATOR_DIMENSIONS",
    "FORMAL_ARTIFACT_RELATIVE_ROOT",
    "PACK_NAME",
    "PAYLOAD_KEYS",
    "PAYLOAD_KIND",
    "AuthoredMorphologyCourseError",
    "AuthoredMorphologySeed",
    "MorphologyPayloadAudit",
    "build_morphology_course_manifest",
    "compile_authored_morphology_course",
    "read_authored_morphology_seeds",
    "validate_morphology_payload",
]
