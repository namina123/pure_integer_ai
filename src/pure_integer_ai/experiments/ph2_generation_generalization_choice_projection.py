"""GG-03 Observation 到单层 GG-01 choice competition 的纯投影。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.identity import (
    OBJECT_HYPOTHESIS,
    CurriculumVersion,
    ObjectIdentity,
    OwnerScope,
    ParserVersion,
    SourceRef,
    VersionBundle,
    VISIBILITY_TENANT,
    concept_identity,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    ScopeIdentity,
    document_scope,
)
from pure_integer_ai.cognition.shared.semantic_object import (
    context_scope_identity,
)
from pure_integer_ai.crosscut.determinism.fingerprint import (
    integer_tuple_fingerprint,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.experiments.ph2_authored_generation_generalization_course import (
    AuthoredGenerationGeneralizationSeed,
)
from pure_integer_ai.experiments.ph2_dataset_contract import CanonicalJsonObject
from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    GenerationChoiceCondition,
    GenerationChoiceHypothesis,
)
from pure_integer_ai.experiments.ph2_generation_generalization_contract import (
    EVALUATOR_DIMENSIONS,
    validate_generation_generalization_payload,
)


GG03_DIMENSION_CHOICE_KIND = {
    "SEMANTIC_ROLE_SCOPE_POLARITY": "CONTENT_CHOICE",
    "SOURCE_UNCERTAINTY_CITATION": "CONTENT_CHOICE",
    "EXACT_MEMORY_BASELINE_REJECT": "CONTENT_CHOICE",
    "STRUCTURE_SLOT_ORDER": "PROPOSITION_STRUCTURE_CHOICE",
    "LEGAL_OBJECT_COMPOSITION": "PROPOSITION_STRUCTURE_CHOICE",
    "COMBINATION_HELD_OUT": "PROPOSITION_STRUCTURE_CHOICE",
    "ADDRESSEE_RECOVERABILITY": "DISCOURSE_REFERENCE_CHOICE",
    "REVISION_SUPERSEDE": "DISCOURSE_REFERENCE_CHOICE",
    "MULTIPLE_LEGAL_SURFACE_SET": "LEXICAL_REALIZATION_CHOICE",
    "RETENTION_REVERIFY": "LEXICAL_REALIZATION_CHOICE",
    "COMMUNICATIVE_TASK": "COMMUNICATIVE_TASK_CHOICE",
    "FAILURE_LAYER_LOCALIZATION": "COMMUNICATIVE_TASK_CHOICE",
    "STANCE_CONTENT_WORDING_SEPARATION": "COMMUNICATIVE_TASK_CHOICE",
    "USE_OUTCOME_TEMPLATE_PROMOTION_REJECT": "COMMUNICATIVE_TASK_CHOICE",
}

_NAMESPACE = 21120
GG03_TRAINING_OWNER = OwnerScope(21121, visibility=VISIBILITY_TENANT)
GG03_EVALUATOR_OWNER = OwnerScope(21122, visibility=VISIBILITY_TENANT)
GG03_RUNTIME_VERSIONS = VersionBundle(
    parser=ParserVersion(1), curriculum=CurriculumVersion(3))
_OWNER_BY_LABEL = {
    "teacher": GG03_TRAINING_OWNER,
    "evaluator": GG03_EVALUATOR_OWNER,
}


class GenerationGeneralizationAssessmentRuntimeError(ValueError):
    """GG-03 runtime 越过 label/split、competition 或资源边界。"""


def strict_runtime_key(
        value: tuple[int, ...], *, where: str,
        ) -> tuple[int, ...]:
    """校验 runtime 使用的非空严格整数 key。"""
    if not isinstance(value, tuple) or not value:
        raise GenerationGeneralizationAssessmentRuntimeError(
            f"{where} 必须为非空 tuple")
    assert_int(*value, _where=where)
    if any(type(item) is not int for item in value):
        raise GenerationGeneralizationAssessmentRuntimeError(
            f"{where} 必须使用严格整数")
    return value


def _pack(value: tuple[int, ...]) -> tuple[int, ...]:
    """把整数 tuple 编成带长度前缀的规范片段。"""
    return len(value), *value


def _text_values(*values: str) -> tuple[int, ...]:
    """把非空 UTF-8 文本序列编码成无歧义整数片段。"""
    encoded = []
    for value in values:
        if not isinstance(value, str) or not value:
            raise GenerationGeneralizationAssessmentRuntimeError(
                "GG-03 runtime text key 非法")
        raw = value.encode("utf-8")
        encoded.extend((len(raw), *raw))
    return tuple(encoded)


def _fingerprint(*values: str, domain: str) -> tuple[int, ...]:
    """生成带域隔离的规范文本指纹。"""
    return integer_tuple_fingerprint(_text_values(*values), domain=domain)


def _source_id(seed_id: str) -> int:
    """从 seed identity 确定性派生正整数 source id。"""
    digest = _fingerprint(seed_id, domain="gg03.runtime.source.v1")[2:10]
    value = 0
    for item in digest:
        value = value * 257 + item + 1
    return value or 1


def _context_ids(value: dict) -> tuple[int, ...]:
    """提取 Observation context 中全部学生可见对象身份。"""
    context = value["context_contract"]
    addressee = context["addressee_context"]
    discourse = context["discourse_state"]
    ids = [
        context["goal_binding"],
        addressee["addressee_id"],
        discourse["topic_id"],
        *addressee["shared_visible_ids"],
        *addressee["recoverable_reference_ids"],
        *discourse["open_question_ids"],
        *discourse["prior_expression_ids"],
        *discourse["revision_dependency_ids"],
    ]
    for obligation in context["content_obligations"]:
        ids.extend((
            obligation["obligation_id"],
            obligation["proposition_id"],
            *obligation["source_ids"],
        ))
        if obligation["uncertainty_id"]:
            ids.append(obligation["uncertainty_id"])
    if any(type(item) is not int or item <= 0 for item in ids):
        raise GenerationGeneralizationAssessmentRuntimeError(
            "GG-03 context object id 非法")
    return tuple(sorted(set(ids)))


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GenerationGeneralizationSurfaceOption:
    """一个学生可见 surface option 及其有限 GG-01 choice。"""

    surface_candidate_id: str
    choice: GenerationChoiceHypothesis

    def __post_init__(self) -> None:
        if not isinstance(self.surface_candidate_id, str) or not (
                self.surface_candidate_id):
            raise GenerationGeneralizationAssessmentRuntimeError(
                "GG-03 surface candidate id 非法")
        if not isinstance(self.choice, GenerationChoiceHypothesis):
            raise TypeError("GG-03 surface option choice 类型错误")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GenerationGeneralizationAssessmentCase:
    """只由 Observation 构造的完整三选项 competition。"""

    seed_id: str
    label_owner: str
    split: str
    evaluation_dimension: str
    choice_kind: str
    source: SourceRef
    scope: ScopeIdentity
    options: tuple[GenerationGeneralizationSurfaceOption, ...]
    baseline: GenerationGeneralizationSurfaceOption
    challenge: GenerationGeneralizationSurfaceOption
    trace: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.seed_id, str) or not self.seed_id:
            raise GenerationGeneralizationAssessmentRuntimeError(
                "GG-03 runtime seed id 非法")
        if self.label_owner not in _OWNER_BY_LABEL:
            raise GenerationGeneralizationAssessmentRuntimeError(
                "GG-03 runtime label owner 非法")
        expected_split = "train" if self.label_owner == "teacher" else (
            "held_out")
        if self.split != expected_split:
            raise GenerationGeneralizationAssessmentRuntimeError(
                "GG-03 runtime owner/split 漂移")
        if self.evaluation_dimension not in EVALUATOR_DIMENSIONS:
            raise GenerationGeneralizationAssessmentRuntimeError(
                "GG-03 runtime dimension 未注册")
        if GG03_DIMENSION_CHOICE_KIND.get(self.evaluation_dimension) != (
                self.choice_kind):
            raise GenerationGeneralizationAssessmentRuntimeError(
                "GG-03 runtime dimension/layer 漂移")
        if not isinstance(self.source, SourceRef):
            raise TypeError("GG-03 runtime source 类型错误")
        if not isinstance(self.scope, ScopeIdentity) or self.scope != (
                document_scope(self.source)):
            raise GenerationGeneralizationAssessmentRuntimeError(
                "GG-03 runtime scope 未绑定 source")
        if (not isinstance(self.options, tuple) or len(self.options) != 3
                or any(not isinstance(item, GenerationGeneralizationSurfaceOption)
                       for item in self.options)
                or len({item.surface_candidate_id for item in self.options}) != 3
                or len({item.choice for item in self.options}) != 3):
            raise GenerationGeneralizationAssessmentRuntimeError(
                "GG-03 runtime options 未精确覆盖三候选")
        if self.baseline not in self.options or self.challenge not in self.options:
            raise GenerationGeneralizationAssessmentRuntimeError(
                "GG-03 runtime baseline/challenge 不属于 options")
        if self.baseline != self.options[1]:
            raise GenerationGeneralizationAssessmentRuntimeError(
                "GG-03 runtime baseline 必须由学生可见第二候选显式提供")
        strict_runtime_key(self.trace, where="GG-03 runtime case trace")

    @property
    def choices(self) -> tuple[GenerationChoiceHypothesis, ...]:
        """按学生可见 option 顺序返回完整 choice 分母。"""
        return tuple(item.choice for item in self.options)

    def option_for_choice(
            self, choice: GenerationChoiceHypothesis,
            ) -> GenerationGeneralizationSurfaceOption:
        """返回 competition 内与 choice 精确对应的 surface option。"""
        try:
            return next(item for item in self.options if item.choice == choice)
        except StopIteration as exc:
            raise GenerationGeneralizationAssessmentRuntimeError(
                "GG-03 selected choice 不属于 case") from exc

    @property
    def visible_inputs(self) -> tuple[ObjectIdentity, ...]:
        """返回 challenge assessment 可合法观察的对象集合。"""
        choice = self.challenge.choice
        return tuple(sorted(set((
            choice.target_obligation,
            choice.condition.context,
            choice.condition.condition,
            *choice.condition.required_context_objects,
        )), key=ObjectIdentity.stable_key))


def project_generation_generalization_observation(
        seed_id: str,
        label_owner: str,
        split: str,
        evaluation_dimension: str,
        observation_payload: CanonicalJsonObject,
        ) -> GenerationGeneralizationAssessmentCase:
    """只读 Observation，构造单层三候选 competition 和显式 B baseline。"""
    if not isinstance(observation_payload, CanonicalJsonObject):
        raise TypeError("GG-03 observation payload 类型错误")
    audit = validate_generation_generalization_payload(observation_payload)
    if label_owner not in _OWNER_BY_LABEL:
        raise GenerationGeneralizationAssessmentRuntimeError(
            "GG-03 observation owner 未注册")
    if split != ("train" if label_owner == "teacher" else "held_out"):
        raise GenerationGeneralizationAssessmentRuntimeError(
            "GG-03 observation owner/split 漂移")
    choice_kind = GG03_DIMENSION_CHOICE_KIND.get(evaluation_dimension)
    if choice_kind is None:
        raise GenerationGeneralizationAssessmentRuntimeError(
            "GG-03 observation dimension 没有 layer route")
    value = observation_payload.to_value()
    owner = _OWNER_BY_LABEL[label_owner]
    source = SourceRef(
        _NAMESPACE,
        _source_id(seed_id),
        0,
        owner,
        GG03_RUNTIME_VERSIONS,
    )
    scope = document_scope(source)
    context = context_scope_identity(
        source,
        integer_tuple_fingerprint(
            tuple(observation_payload.payload),
            domain="gg03.runtime.context.v1",
        ),
    )
    layer = next(
        item for item in value["choice_candidates"]
        if item["choice_kind"] == choice_kind)
    condition = GenerationChoiceCondition(
        concept_identity(
            _fingerprint(
                layer["condition_family"],
                domain="gg03.runtime.condition.v1",
            ),
            owner=owner,
            versions=GG03_RUNTIME_VERSIONS,
        ),
        context,
        tuple(
            concept_identity(
                (_NAMESPACE, 20, item), owner=owner,
                versions=GG03_RUNTIME_VERSIONS)
            for item in _context_ids(value)
        ),
        (),
        scope,
    )
    target = concept_identity(
        (_NAMESPACE, 21, value["context_contract"]["goal_binding"]),
        owner=owner,
        versions=GG03_RUNTIME_VERSIONS,
    )
    competition_key = integer_tuple_fingerprint(
        _text_values(
            audit.combination_key,
            choice_kind,
            *audit.surface_candidate_ids,
        ),
        domain="gg03.runtime.competition.v1",
    )
    options = []
    for item in value["surface_candidates"]:
        surface_id = item["surface_candidate_id"]
        selected = concept_identity(
            _fingerprint(
                surface_id,
                item["surface_family"],
                item["structure_family"],
                item["lexical_family"],
                domain="gg03.runtime.selected.surface.v1",
            ),
            owner=owner,
            versions=GG03_RUNTIME_VERSIONS,
        )
        candidate = ObjectIdentity(
            OBJECT_HYPOTHESIS,
            integer_tuple_fingerprint(
                (*_text_values(seed_id, surface_id), *_pack(competition_key)),
                domain="gg03.runtime.choice.candidate.v1",
            ),
            owner,
            GG03_RUNTIME_VERSIONS,
        )
        choice = GenerationChoiceHypothesis(
            candidate,
            choice_kind,
            target,
            condition,
            selected,
            (source,),
            competition_key,
            scope,
        )
        options.append(GenerationGeneralizationSurfaceOption(
            surface_id, choice))
    options = tuple(options)
    challenge_id = value["surface_constraints"]["challenge_candidate_id"]
    challenge = next(
        item for item in options if item.surface_candidate_id == challenge_id)
    return GenerationGeneralizationAssessmentCase(
        seed_id,
        label_owner,
        split,
        evaluation_dimension,
        choice_kind,
        source,
        scope,
        options,
        options[1],
        challenge,
        (
            _NAMESPACE,
            *integer_tuple_fingerprint(
                _text_values(seed_id, audit.combination_key, choice_kind),
                domain="gg03.runtime.case.trace.v1",
            ),
        ),
    )


def project_generation_generalization_seed(
        seed: AuthoredGenerationGeneralizationSeed,
        ) -> GenerationGeneralizationAssessmentCase:
    """从 seed 仅转交 Observation 字段，不读取 expected payload。"""
    if not isinstance(seed, AuthoredGenerationGeneralizationSeed):
        raise TypeError("GG-03 seed 类型错误")
    return project_generation_generalization_observation(
        seed.seed_id,
        seed.label_owner,
        seed.split,
        seed.evaluation_dimension,
        seed.observation_payload,
    )


__all__ = [
    "GG03_DIMENSION_CHOICE_KIND",
    "GG03_EVALUATOR_OWNER",
    "GG03_RUNTIME_VERSIONS",
    "GG03_TRAINING_OWNER",
    "GenerationGeneralizationAssessmentCase",
    "GenerationGeneralizationAssessmentRuntimeError",
    "GenerationGeneralizationSurfaceOption",
    "project_generation_generalization_observation",
    "project_generation_generalization_seed",
]
