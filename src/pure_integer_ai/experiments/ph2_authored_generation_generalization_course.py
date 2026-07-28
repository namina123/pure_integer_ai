"""GG-03 D-02E.4 多合法表达与组合生成原创课程。"""
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
from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    CHOICE_KINDS,
)
from pure_integer_ai.experiments.ph2_generation_generalization_contract import (
    ABLATION_KEYS,
    BASELINE_KINDS,
    CANDIDATE_CASES,
    COMBINATION_AXES,
    COMBINATION_KEY_AXES,
    COURSE_FAMILIES,
    EVALUATOR_DIMENSIONS,
    INDEPENDENT_VERIFIER_REQUIREMENTS,
    PAYLOAD_KEYS,
    PAYLOAD_KIND,
    RETENTION_PROTOCOLS,
    VERIFIER_NE_CONDITIONS,
    GenerationGeneralizationContractError,
    combination_key,
    surface_sha256,
    validate_generation_generalization_expected,
    validate_generation_generalization_payload,
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
PACK_NAME = "AUTHORED_CC0_V1--CC0-1.0--gg03-generation-generalization-v1"
STAGE = "W-09"
SUBSTAGE = "GG03_GENERATION_GENERALIZATION"
COURSE_MANIFEST_ARTIFACT_VERSION = (
    "GG-03-D02E4-generation-generalization-course-v1")
COURSE_MANIFEST_PATH = Path(
    "data/ph2/manifests/gg03_generation_generalization_course_v1.json")
FORMAL_ARTIFACT_RELATIVE_ROOT = (
    "ph2_dataset_artifacts/d02_language_courses_v1")
TASK_KEYS = ("GG-03", "LC-13", "LC-15")
CAPABILITY_KEYS = tuple(sorted((
    "LAYERED_GENERATION",
    "PRAGMATIC_CLARIFICATION_REPAIR",
    "REFERENCE_DISCOURSE_REVISION",
    "SOURCE_UNCERTAINTY_REALITY",
    "TYPED_LEARNING_OBJECTIVES",
)))

_SEED_FIELDS = {
    "evaluation_dimension", "expected_payload", "expected_state", "family",
    "label_owner", "license_id", "logical_order", "observation_payload",
    "perturbation_kind", "sample_role", "seed_id", "split",
    "supersedes_seed_id", "template_family",
}


class AuthoredGenerationGeneralizationCourseError(RuntimeError):
    """GG-03 seed、owner、组合 split、revision 或私有 label 非法。"""


def _exact(value: Any, keys: set[str], *, where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise AuthoredGenerationGeneralizationCourseError(
            f"{where} 字段集合非法")
    return value


def _text(value: Any, *, where: str, allow_empty: bool = False) -> str:
    if (not isinstance(value, str) or value.strip() != value
            or (not allow_empty and not value)):
        raise AuthoredGenerationGeneralizationCourseError(f"{where} 文本非法")
    return value


def _positive(value: Any, *, where: str) -> int:
    if type(value) is not int or value <= 0:
        raise AuthoredGenerationGeneralizationCourseError(f"{where} 必须为正整数")
    return value


@dataclass(frozen=True)
class AuthoredGenerationGeneralizationSeed:
    """一个 owner/split 下的学生 Observation 与独立私有 label。"""

    seed_id: str
    family: str
    template_family: str
    label_owner: str
    split: str
    sample_role: str
    observation_payload: CanonicalJsonObject
    expected_state: str
    expected_payload: CanonicalJsonObject
    evaluation_dimension: str
    perturbation_kind: str
    supersedes_seed_id: str
    logical_order: int
    license_id: str

    def __post_init__(self) -> None:
        for name in (
                "seed_id", "family", "template_family", "perturbation_kind"):
            _text(getattr(self, name), where=name)
        if self.label_owner not in ("teacher", "evaluator"):
            raise AuthoredGenerationGeneralizationCourseError("label_owner 非法")
        if self.split != ("train" if self.label_owner == "teacher" else "held_out"):
            raise AuthoredGenerationGeneralizationCourseError(
                "label_owner 与 split 不一致")
        if self.sample_role not in SAMPLE_ROLES:
            raise AuthoredGenerationGeneralizationCourseError("sample_role 未登记")
        if not isinstance(self.observation_payload, CanonicalJsonObject):
            raise AuthoredGenerationGeneralizationCourseError(
                "observation_payload 类型非法")
        if not isinstance(self.expected_payload, CanonicalJsonObject):
            raise AuthoredGenerationGeneralizationCourseError(
                "expected_payload 类型非法")
        if self.expected_state not in EXPECTED_STATES:
            raise AuthoredGenerationGeneralizationCourseError("expected_state 非四态")
        if self.evaluation_dimension not in EVALUATOR_DIMENSIONS:
            raise AuthoredGenerationGeneralizationCourseError(
                "evaluation_dimension 未登记")
        _text(self.supersedes_seed_id, where="supersedes_seed_id", allow_empty=True)
        _positive(self.logical_order, where="logical_order")
        if self.license_id != LICENSE_ID:
            raise AuthoredGenerationGeneralizationCourseError(
                "GG-03 原创课程必须为 CC0-1.0")
        try:
            validate_generation_generalization_expected(
                self.expected_payload,
                expected_state=self.expected_state,
                evaluation_dimension=self.evaluation_dimension,
                observation_payload=self.observation_payload,
            )
        except GenerationGeneralizationContractError as error:
            raise AuthoredGenerationGeneralizationCourseError(
                "GG-03 typed payload 非法") from error

    @property
    def sample_family(self) -> str:
        return str(self.observation_payload.to_value()["sample_family"])

    @property
    def course_family(self) -> str:
        return str(self.observation_payload.to_value()["course_family"])

    @property
    def candidate_case(self) -> str:
        return str(self.observation_payload.to_value()["candidate_case"])

    @property
    def retention_anchor_id(self) -> str:
        return str(self.observation_payload.to_value()["retention_anchor_id"])

    def compiled_seed(self) -> AuthoredCompiledSeed:
        """只在 pack 边缘翻译为共用四-owner record。"""
        audit = validate_generation_generalization_payload(
            self.observation_payload)
        value = self.observation_payload.to_value()
        observed = value["observed_surface"]
        objectives = tuple(value["objective_keys"])
        return AuthoredCompiledSeed(
            self.seed_id,
            self.family,
            self.template_family,
            self.label_owner,
            self.split,
            self.sample_role,
            PAYLOAD_KIND,
            self.observation_payload,
            self.expected_state,
            self.expected_payload,
            self.perturbation_kind,
            self.supersedes_seed_id,
            self.logical_order,
            (
                self.seed_id,
                audit.combination_key,
                audit.surface_candidate_ids,
            ),
            (
                observed["text"],
                self.course_family,
                objectives,
            ),
            (
                self.candidate_case,
                audit.choice_kinds,
                audit.combination_values,
                audit.exact_memory_control,
                audit.replay_kind,
            ),
            self.evaluation_dimension,
        )


def _seed_from_value(value: dict[str, Any]) -> AuthoredGenerationGeneralizationSeed:
    raw = _exact(value, _SEED_FIELDS, where="GG-03 seed")
    observation = raw["observation_payload"]
    expected = raw["expected_payload"]
    if not isinstance(observation, dict) or not isinstance(expected, dict):
        raise AuthoredGenerationGeneralizationCourseError("seed payload 必须为对象")
    return AuthoredGenerationGeneralizationSeed(
        _text(raw["seed_id"], where="seed_id"),
        _text(raw["family"], where="family"),
        _text(raw["template_family"], where="template_family"),
        _text(raw["label_owner"], where="label_owner"),
        _text(raw["split"], where="split"),
        _text(raw["sample_role"], where="sample_role"),
        CanonicalJsonObject.from_value(observation),
        _text(raw["expected_state"], where="expected_state"),
        CanonicalJsonObject.from_value(expected),
        _text(raw["evaluation_dimension"], where="evaluation_dimension"),
        _text(raw["perturbation_kind"], where="perturbation_kind"),
        _text(raw["supersedes_seed_id"], where="supersedes_seed_id",
              allow_empty=True),
        _positive(raw["logical_order"], where="logical_order"),
        _text(raw["license_id"], where="license_id"),
    )


def read_authored_generation_generalization_seeds(
        path: str | Path,
        ) -> tuple[AuthoredGenerationGeneralizationSeed, ...]:
    """严格读取双 owner 课程并核十轴 held-out、revision 和 retention。"""
    try:
        payload = Path(path).read_bytes()
    except OSError as error:
        raise AuthoredGenerationGeneralizationCourseError(
            "GG-03 sample 无法读取") from error
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise AuthoredGenerationGeneralizationCourseError("GG-03 sample 换行非法")
    seeds = []
    try:
        for line in payload.splitlines(keepends=True):
            value = parse_canonical_json_bytes(line[:-1], require_object=True)
            assert isinstance(value, dict)
            if canonical_json_line(value) != line:
                raise AuthoredGenerationGeneralizationCourseError(
                    "GG-03 seed 非规范 JSON")
            seeds.append(_seed_from_value(value))
    except AuthoredGenerationGeneralizationCourseError:
        raise
    except Exception as error:
        raise AuthoredGenerationGeneralizationCourseError(
            "GG-03 seed 损坏") from error
    if not seeds:
        raise AuthoredGenerationGeneralizationCourseError("GG-03 seed 为空")
    if len({item.seed_id for item in seeds}) != len(seeds):
        raise AuthoredGenerationGeneralizationCourseError("GG-03 seed_id 重复")
    orders = [item.logical_order for item in seeds]
    if orders != sorted(orders) or len(set(orders)) != len(orders):
        raise AuthoredGenerationGeneralizationCourseError(
            "GG-03 logical_order 必须严格递增")
    index = {item.seed_id: item for item in seeds}
    for seed in seeds:
        if seed.sample_family == "REVISION":
            if not seed.supersedes_seed_id:
                raise AuthoredGenerationGeneralizationCourseError(
                    "REVISION 必须声明 supersede")
        elif seed.supersedes_seed_id:
            raise AuthoredGenerationGeneralizationCourseError(
                "非 REVISION 不得声明 supersede")
        if seed.supersedes_seed_id:
            target = index.get(seed.supersedes_seed_id)
            if (target is None or target.logical_order >= seed.logical_order
                    or target.family != seed.family or target.split != seed.split):
                raise AuthoredGenerationGeneralizationCourseError(
                    "GG-03 supersede 必须同 family/split 指向更早 seed")
        if seed.sample_family == "RETENTION":
            target = index.get(seed.retention_anchor_id)
            if (target is None or target.logical_order >= seed.logical_order
                    or target.family != seed.family or target.split != seed.split):
                raise AuthoredGenerationGeneralizationCourseError(
                    "GG-03 retention 必须同 family/split 指向更早 anchor")
        elif seed.retention_anchor_id:
            raise AuthoredGenerationGeneralizationCourseError(
                "非 RETENTION 不得声明 retention anchor")

    teacher = tuple(item for item in seeds if item.label_owner == "teacher")
    evaluator = tuple(item for item in seeds if item.label_owner == "evaluator")
    if not teacher or not evaluator:
        raise AuthoredGenerationGeneralizationCourseError("GG-03 双 owner 缺失")
    for owner in (teacher, evaluator):
        if {item.sample_family for item in owner} != set(SAMPLE_FAMILIES):
            raise AuthoredGenerationGeneralizationCourseError(
                "GG-03 七类 sample family 未覆盖")
        if {item.course_family for item in owner} != set(COURSE_FAMILIES):
            raise AuthoredGenerationGeneralizationCourseError(
                "GG-03 十类课程族未覆盖")
        if {item.candidate_case for item in owner} != set(CANDIDATE_CASES):
            raise AuthoredGenerationGeneralizationCourseError(
                "GG-03 十四 candidate case 未覆盖")
    if ({item.family for item in teacher} & {item.family for item in evaluator}
            or {item.template_family for item in teacher}
            & {item.template_family for item in evaluator}):
        raise AuthoredGenerationGeneralizationCourseError(
            "GG-03 teacher/evaluator family 或 template 泄漏")
    if {item.evaluation_dimension for item in evaluator} != set(
            EVALUATOR_DIMENSIONS):
        raise AuthoredGenerationGeneralizationCourseError(
            "GG-03 evaluator 维度未列全")

    teacher_audits = tuple(validate_generation_generalization_payload(
        item.observation_payload) for item in teacher)
    evaluator_audits = tuple(validate_generation_generalization_payload(
        item.observation_payload) for item in evaluator)
    if ({item.combination_key for item in teacher_audits}
            & {item.combination_key for item in evaluator_audits}):
        raise AuthoredGenerationGeneralizationCourseError(
            "GG-03 held-out 完整组合泄漏")
    source_index = COMBINATION_KEY_AXES.index("source_cluster")
    for axis_index, axis in enumerate(COMBINATION_KEY_AXES):
        teacher_values = {item.combination_values[axis_index]
                          for item in teacher_audits}
        evaluator_values = {item.combination_values[axis_index]
                            for item in evaluator_audits}
        if axis == "source_cluster":
            if teacher_values & evaluator_values:
                raise AuthoredGenerationGeneralizationCourseError(
                    "GG-03 source cluster 未物理隔离")
        elif not evaluator_values.issubset(teacher_values):
            raise AuthoredGenerationGeneralizationCourseError(
                f"GG-03 held-out 轴 {axis} 没有 train 分量证据")
    if (not any(item.exact_memory_control for item in teacher_audits)
            or not any(item.exact_memory_control for item in evaluator_audits)
            or not all(item.combination_values[source_index]
                       for item in (*teacher_audits, *evaluator_audits))):
        raise AuthoredGenerationGeneralizationCourseError(
            "GG-03 exact-memory/source 对照不完整")
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
        "authored-gg03-generation-generalization-seed-v1",
        "urn:ph2:authored:gg03-generation-generalization-v1",
        "PH2 authored GG-03 generation generalization course, CC0-1.0",
        "GENERATION_GENERALIZATION_LABEL",
        "gg03-generation-generalization",
        768,
    )


def compile_authored_generation_generalization_course(
        sample_path: str | Path,
        release_root: str | Path,
        ) -> AuthoredCourseBuild:
    """发布 D-02E.4 pack，不运行 generation runtime 或训练。"""
    seeds = read_authored_generation_generalization_seeds(sample_path)
    return publish_authored_course(
        tuple(item.compiled_seed() for item in seeds),
        sample_path,
        release_root,
        _course_spec(),
    )


def build_generation_generalization_course_manifest(
        sample_path: str | Path,
        build: AuthoredCourseBuild,
        *,
        artifact_relative_root: str = FORMAL_ARTIFACT_RELATIVE_ROOT,
        ) -> CapabilityCourseManifest:
    """冻结 GG-03/LC-13/15 课程事实，runtime 与 execution 保持零。"""
    sample = Path(sample_path)
    seeds = read_authored_generation_generalization_seeds(sample)
    teacher = tuple(item for item in seeds if item.label_owner == "teacher")
    evaluator = tuple(item for item in seeds if item.label_owner == "evaluator")
    objectives = tuple(sorted({
        key
        for item in seeds
        for key in item.observation_payload.to_value()["objective_keys"]
    }))
    return CapabilityCourseManifest(
        1,
        COURSE_MANIFEST_ARTIFACT_VERSION,
        "COURSE_FROZEN",
        "NOT_STARTED",
        TASK_KEYS,
        CAPABILITY_KEYS,
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
        objectives,
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


@dataclass(frozen=True)
class _Scenario:
    code: str
    course_family: str
    candidate_case: str
    sample_family: str
    sample_role: str
    expected_state: str
    evaluation_dimension: str
    stance: str
    perturbation_kind: str


_SCENARIOS = (
    _Scenario("S01", "SINGLE_PROPOSITION_RECOMBINATION",
              "SINGLE_PROPOSITION_RECOMBINATION", "POSITIVE", "support",
              "TRUE", "LEGAL_OBJECT_COMPOSITION", "ANSWER", "NONE"),
    _Scenario("S02", "MULTI_LEGAL_SURFACE", "MULTI_LEGAL_SET",
              "GENERATION", "support", "TRUE", "MULTIPLE_LEGAL_SURFACE_SET",
              "ANSWER", "NONE"),
    _Scenario("S03", "SEMANTIC_DRIFT_NEGATIVE", "RELATION_ROLE_SCOPE_DRIFT",
              "NEGATIVE", "refute", "FALSE", "SEMANTIC_ROLE_SCOPE_POLARITY",
              "REFUSE", "OPERAND_ORDER_SWAP"),
    _Scenario("S04", "SEMANTIC_DRIFT_NEGATIVE", "STRUCTURE_SLOT_ORDER_DRIFT",
              "NEGATIVE", "refute", "FALSE", "STRUCTURE_SLOT_ORDER",
              "REFUSE", "TARGET_REPLACEMENT"),
    _Scenario("S05", "SOURCE_UNCERTAINTY_QUALIFIER",
              "SOURCE_UNCERTAINTY_PRESERVATION", "AMBIGUOUS", "conflict",
              "UNKNOWN", "SOURCE_UNCERTAINTY_CITATION", "CLARIFY",
              "CONFLICT_SOURCE"),
    _Scenario("S06", "REFERENCE_RECOVERABILITY", "REFERENCE_RECOVERABILITY",
              "POSITIVE", "support", "TRUE", "ADDRESSEE_RECOVERABILITY",
              "ANSWER", "NONE"),
    _Scenario("S07", "ELLIPSIS_EXPLICIT_CONTRAST",
              "ELLIPSIS_CONDITION_CONTRAST", "AMBIGUOUS", "conflict",
              "UNKNOWN", "COMMUNICATIVE_TASK", "CLARIFY", "SCOPE_TARGET_SHIFT"),
    _Scenario("S08", "MULTI_PROPOSITION_REVISION", "MULTI_PROPOSITION_ORDER",
              "GENERATION", "support", "TRUE", "COMBINATION_HELD_OUT",
              "ANSWER", "NONE"),
    _Scenario("S09", "USE_OUTCOME_REPLAY", "EXACT_MEMORY_CONTROL", "NEGATIVE",
              "anomaly", "FALSE", "EXACT_MEMORY_BASELINE_REJECT", "REFUSE",
              "CONTENT_REPLACEMENT"),
    _Scenario("S10", "MULTI_PROPOSITION_REVISION", "REVISION_SUPERSEDE",
              "REVISION", "supersede", "TRUE", "REVISION_SUPERSEDE",
              "ANSWER", "PARSER_REVISION"),
    _Scenario("S11", "CONTEXT_ADDRESSEE_CONDITION",
              "ADDRESSEE_CONTEXT_NEGATIVE", "NEGATIVE", "refute", "FALSE",
              "FAILURE_LAYER_LOCALIZATION", "CLARIFY", "SCOPE_TARGET_SHIFT"),
    _Scenario("S12", "STANCE_LEDGER", "STANCE_CONTENT_WORDING", "UNKNOWN",
              "read_only_probe", "UNKNOWN", "STANCE_CONTENT_WORDING_SEPARATION",
              "UNKNOWN", "NONE"),
    _Scenario("S13", "USE_OUTCOME_REPLAY", "USE_OUTCOME_EVIDENCE_ONLY",
              "GENERATION", "support", "TRUE",
              "USE_OUTCOME_TEMPLATE_PROMOTION_REJECT", "ANSWER", "NONE"),
    _Scenario("S14", "USE_OUTCOME_REPLAY", "RETENTION_REVERIFY", "RETENTION",
              "read_only_probe", "TRUE", "RETENTION_REVERIFY", "ANSWER", "NONE"),
)

_AXIS_VALUES = {
    "direction": (
        "CONTEXT_TO_GENERATION", "GENERATION_CLARIFY", "GENERATION_REFUSE",
        "GENERATION_REPAIR", "UNDERSTANDING_TO_GENERATION"),
    "obligation_kind": ("ASSERT", "CITE", "CLARIFY", "REFUSE", "REVISE"),
    "discourse_strategy": (
        "CORRECTION", "DESCRIPTIVE_REFERENCE", "DIRECT", "ELLIPSIS",
        "EXPLICIT_SOURCE"),
    "proposition_logic_shape": (
        "ATOMIC", "CONJUNCTION", "MODAL", "NEGATION", "RELATION"),
    "structure_family": (
        "COORDINATION", "DECLARATIVE", "INVERSION", "REPAIR", "SUBORDINATION"),
    "lexical_realization_family": (
        "ALIAS", "DESCRIPTION", "PRONOUN", "QUALIFIED", "REPETITION"),
    "context_condition_family": (
        "CONFLICT", "NO_SHARED_VISIBLE", "OPEN_QUESTION", "PRIOR_MENTION",
        "SHARED_VISIBLE"),
    "addressee_recoverability_family": (
        "AMBIGUOUS", "CLARIFICATION_REQUIRED", "DESCRIPTION_REQUIRED",
        "NAME_RECOVERABLE", "PRONOUN_RECOVERABLE"),
}

_OUTPUT_BASES = (
    "红色方块位于蓝色圆形左侧",
    "门已经关闭",
    "甲事件先于乙事件",
    "小李把书放在桌上",
    "据记录，温度可能下降",
    "小王把钥匙交给小陈",
    "应明确重复主任这一称呼",
    "先检查阀门，再记录压力",
    "旧日志中的完整回答不能直接复用",
    "先前记录有误，阀门仍然开启",
    "当前指称不足以恢复目标杯子",
    "目前信息不足，需要继续澄清",
    "这次选择只引用了对应的使用证据",
    "复核后仍然存在两个合法表达",
)


def _axis_value(axis: str, owner_index: int, scenario_index: int) -> str:
    values = _AXIS_VALUES[axis]
    offset = scenario_index if owner_index == 0 else scenario_index * 2 + 1
    return values[(offset + COMBINATION_KEY_AXES.index(axis)) % len(values)]


def _layer_for_dimension(dimension: str) -> str:
    if dimension in {
            "SEMANTIC_ROLE_SCOPE_POLARITY", "SOURCE_UNCERTAINTY_CITATION",
            "EXACT_MEMORY_BASELINE_REJECT"}:
        return "CONTENT_CHOICE"
    if dimension in {
            "STRUCTURE_SLOT_ORDER", "LEGAL_OBJECT_COMPOSITION",
            "COMBINATION_HELD_OUT"}:
        return "PROPOSITION_STRUCTURE_CHOICE"
    if dimension in {
            "ADDRESSEE_RECOVERABILITY", "REVISION_SUPERSEDE"}:
        return "DISCOURSE_REFERENCE_CHOICE"
    if dimension in {
            "MULTIPLE_LEGAL_SURFACE_SET", "RETENTION_REVERIFY"}:
        return "LEXICAL_REALIZATION_CHOICE"
    return "COMMUNICATIVE_TASK_CHOICE"


def _objective_keys(scenario: _Scenario) -> tuple[str, ...]:
    keys = {"CROSS_CONTEXT_CONSISTENCY", "GENERATION_ADOPTION"}
    if scenario.expected_state != "TRUE":
        keys.add("GENERATION_FAILURE")
        keys.add("CONTROLLED_PERTURBATION")
    if scenario.sample_family == "REVISION":
        keys.add("NEXT_DISCOURSE_UNIT")
        keys.add("ORDER_RECOVERY")
    if scenario.sample_family == "RETENTION":
        keys.add("INTEGER_DESCRIPTION_LENGTH")
    result = tuple(sorted(keys))
    if any(item not in LANGUAGE_OBJECTIVE_KEYS for item in result):
        raise AssertionError("default objective key 未登记")
    return result


def _combination(owner_index: int, scenario_index: int,
                 family: str, exact_memory: int) -> dict[str, Any]:
    result = {
        axis: _axis_value(axis, owner_index, scenario_index)
        for axis in COMBINATION_KEY_AXES
        if axis not in {"source_cluster", "parser_course_version"}
    }
    result["source_cluster"] = (
        ("TRAIN_SOURCE_" if owner_index == 0 else "EVAL_SOURCE_") + family)
    result["parser_course_version"] = "PARSER_1_COURSE_1"
    result["combination_key"] = combination_key(result)
    result["exact_memory_control"] = exact_memory
    return result


def _context(owner_index: int, scenario_index: int,
             scenario: _Scenario, combo: dict[str, Any]) -> dict[str, Any]:
    base = 910000 + owner_index * 100000 + scenario_index * 100
    shared = [base + 2, base + 3]
    recoverable = shared[:1]
    if combo["addressee_recoverability_family"] in {
            "AMBIGUOUS", "CLARIFICATION_REQUIRED"}:
        recoverable = []
    proposition_count = 2 if scenario.course_family == (
        "MULTI_PROPOSITION_REVISION") else 1
    obligations = []
    for offset in range(proposition_count):
        obligations.append({
            "obligation_id": base + 10 + offset,
            "proposition_id": base + 20 + offset,
            "requirement": "REQUIRED",
            "source_ids": [base + 30 + offset],
            "uncertainty_id": (
                base + 40 + offset
                if scenario.course_family == "SOURCE_UNCERTAINTY_QUALIFIER"
                else 0),
        })
    return {
        "addressee_context": {
            "addressee_id": base + 1,
            "recoverable_reference_ids": recoverable,
            "shared_visible_ids": shared,
        },
        "communicative_goal": combo["obligation_kind"],
        "content_obligations": obligations,
        "discourse_state": {
            "open_question_ids": (
                [base + 50]
                if combo["context_condition_family"] == "OPEN_QUESTION" else []),
            "prior_expression_ids": [base + 51],
            "revision_dependency_ids": (
                [base + 52]
                if scenario.course_family == "MULTI_PROPOSITION_REVISION" else []),
            "topic_id": base + 53,
        },
        "expression_constraints": {
            "ellipsis_allowed": 1 if combo["discourse_strategy"] == "ELLIPSIS" else 0,
            "explicit_source_required": (
                1 if combo["discourse_strategy"] == "EXPLICIT_SOURCE" else 0),
            "max_surface_units": 96,
            "target_language": "zh",
        },
        "goal_binding": base + 60,
    }


def _surface_candidates(prefix: str, scenario_index: int,
                        combo: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for ordinal, suffix in enumerate(("A", "B", "C"), start=1):
        result.append({
            "complete_answer": 0,
            "fragment": f"{prefix}{scenario_index:02d}候选片段{suffix}",
            "lexical_family": combo["lexical_realization_family"],
            "source_material_only": 1,
            "structure_family": combo["structure_family"],
            "surface_candidate_id": f"{prefix}{scenario_index:02d}_{suffix}",
            "surface_family": f"SURFACE_FAMILY_{suffix}",
        })
    return result


def _choice_candidates(prefix: str, scenario_index: int,
                       replay: bool) -> list[dict[str, Any]]:
    result = []
    for ordinal, choice_kind in enumerate(CHOICE_KINDS, start=1):
        result.append({
            "candidate_id": f"{prefix}{scenario_index:02d}_CHOICE_{ordinal}",
            "choice_kind": choice_kind,
            "complete_answer_template": 0,
            "condition_family": f"CONDITION_{scenario_index:02d}_{ordinal}",
            "outcome_broadcast": 0,
            "selected_object_id": 930000 + scenario_index * 10 + ordinal,
            "selection_state": "UNSELECTED",
            "source_ids": [940000 + scenario_index * 10 + ordinal],
            "use_ref_id": (
                f"USE_EVIDENCE_{prefix}{scenario_index:02d}_{ordinal}"
                if replay else ""),
        })
    return result


def _expected_payload(
        scenario: _Scenario,
        candidate_ids: tuple[str, ...],
        variants: tuple[str, ...],
        ) -> dict[str, Any]:
    target_layer = _layer_for_dimension(scenario.evaluation_dimension)
    layer_states = []
    for choice_kind in CHOICE_KINDS:
        state = "TRUE"
        if choice_kind == target_layer and scenario.expected_state != "TRUE":
            state = scenario.expected_state
        layer_states.append({"choice_kind": choice_kind, "state": state})
    return {
        "accepted_surface_candidate_ids": list(candidate_ids[:2]),
        "accepted_surface_variants": list(sorted(variants)),
        "challenge_verdict": scenario.expected_state,
        "choice_layer_states": layer_states,
        "expected_failure_dimension": scenario.evaluation_dimension,
        "expected_stance": scenario.stance,
        "independent_verifier_requirements": {
            key: 1 for key in INDEPENDENT_VERIFIER_REQUIREMENTS},
        "rejected_surface_candidate_ids": [candidate_ids[2]],
        "surface_set_comparison": "SET_OR_CONSTRAINT",
        "unique_expected_string_forbidden": 1,
    }


def _seed_value(owner_index: int, scenario_index: int,
                scenario: _Scenario) -> dict[str, Any]:
    prefix = "T" if owner_index == 0 else "E"
    label_owner = "teacher" if owner_index == 0 else "evaluator"
    split = "train" if owner_index == 0 else "held_out"
    family = f"GG03_{prefix}_{scenario.course_family}"
    seed_id = f"GG03_{prefix}_{scenario.code}"
    exact_memory = 1 if scenario.candidate_case == "EXACT_MEMORY_CONTROL" else 0
    replay = scenario.candidate_case in {
        "EXACT_MEMORY_CONTROL", "RETENTION_REVERIFY",
        "USE_OUTCOME_EVIDENCE_ONLY"}
    combo = _combination(owner_index, scenario_index, family, exact_memory)
    surfaces = _surface_candidates(prefix, scenario_index, combo)
    surface_ids = tuple(item["surface_candidate_id"] for item in surfaces)
    output_base = _OUTPUT_BASES[scenario_index - 1]
    variants = tuple(sorted((
        f"{output_base}。",
        f"换一种合规说法，{output_base}。",
    )))
    retention_anchor = (
        f"GG03_{prefix}_S13" if scenario.sample_family == "RETENTION" else "")
    supersedes = (
        f"GG03_{prefix}_S08" if scenario.sample_family == "REVISION" else "")
    observed_text = (
        f"场景 {prefix}-{scenario_index:02d}：请根据编号命题、来源约束和当前受众边界组织回答。")
    objective_keys = _objective_keys(scenario)
    observation = {
        "candidate_case": scenario.candidate_case,
        "choice_candidates": _choice_candidates(prefix, scenario_index, replay),
        "combination_split": combo,
        "context_contract": _context(
            owner_index, scenario_index, scenario, combo),
        "course_family": scenario.course_family,
        "failure_profile": {
            "possible_failure_dimensions": [scenario.evaluation_dimension],
            "sentence_wide_penalty_forbidden": 1,
        },
        "objective_keys": list(objective_keys),
        "observed_surface": {
            "append_only": 1,
            "output_surface_hidden": 1,
            "sha256": surface_sha256(observed_text),
            "text": observed_text,
        },
        "replay_evidence": {
            "assessment_update_present": 0,
            "complete_template_promotion_forbidden": 1,
            "exact_use_ids": (
                [f"USE_EVIDENCE_{prefix}{scenario_index:02d}_{ordinal}"
                 for ordinal in range(1, len(CHOICE_KINDS) + 1)]
                if replay else []),
            "replay_kind": "EVIDENCE_ONLY" if replay else "NONE",
        },
        "resource_budget": {
            "max_choice_candidates": 5,
            "max_context_objects": 32,
            "max_surface_candidates": 4,
            "max_surface_units": 96,
            "max_verifier_dimensions": 8,
        },
        "retention_anchor_id": retention_anchor,
        "sample_family": scenario.sample_family,
        "selection_state": "UNSELECTED",
        "surface_candidates": surfaces,
        "surface_constraints": {
            "candidate_ids": list(sorted(surface_ids)),
            "challenge_candidate_id": (
                surface_ids[2] if scenario.expected_state == "FALSE"
                else surface_ids[0]),
            "minimum_legal_surfaces": 2,
            "output_surface_hidden": 1,
            "target_language": "zh",
            "unique_string_comparison": 0,
        },
    }
    return {
        "evaluation_dimension": scenario.evaluation_dimension,
        "expected_payload": _expected_payload(
            scenario, tuple(sorted(surface_ids)), variants),
        "expected_state": scenario.expected_state,
        "family": family,
        "label_owner": label_owner,
        "license_id": LICENSE_ID,
        "logical_order": scenario_index + owner_index * 100,
        "observation_payload": observation,
        "perturbation_kind": scenario.perturbation_kind,
        "sample_role": scenario.sample_role,
        "seed_id": seed_id,
        "split": split,
        "supersedes_seed_id": supersedes,
        "template_family": f"GG03_{prefix}_TEMPLATE_{scenario.candidate_case}",
    }


def build_default_generation_generalization_seed_values() -> tuple[dict[str, Any], ...]:
    """构造双 owner、十轴重组且完整组合不相交的 28 条原创 seed。"""
    values = []
    for owner_index in range(2):
        for scenario_index, scenario in enumerate(_SCENARIOS, start=1):
            values.append(_seed_value(owner_index, scenario_index, scenario))
    return tuple(values)


def default_generation_generalization_sample_bytes() -> bytes:
    return b"".join(canonical_json_line(item)
                    for item in build_default_generation_generalization_seed_values())


__all__ = [
    "ARTIFACT_VERSION",
    "AuthoredGenerationGeneralizationCourseError",
    "AuthoredGenerationGeneralizationSeed",
    "CAPABILITY_KEYS",
    "COURSE_MANIFEST_ARTIFACT_VERSION",
    "COURSE_MANIFEST_PATH",
    "FORMAL_ARTIFACT_RELATIVE_ROOT",
    "PACK_NAME",
    "SOURCE_KEY",
    "STAGE",
    "SUBSTAGE",
    "TASK_KEYS",
    "build_default_generation_generalization_seed_values",
    "build_generation_generalization_course_manifest",
    "compile_authored_generation_generalization_course",
    "default_generation_generalization_sample_bytes",
    "read_authored_generation_generalization_seeds",
]
