"""用精确 H-05 候选投影选择有限 GG-01 生成选项。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.candidate_projection import (
    CandidateGraphProjection,
)
from pure_integer_ai.cognition.shared.candidate_runtime import (
    CandidateLearningRuntime,
)
from pure_integer_ai.cognition.shared.hypothesis import HypothesisKey
from pure_integer_ai.cognition.shared.identity import ObjectIdentity
from pure_integer_ai.crosscut.determinism.fingerprint import (
    integer_tuple_fingerprint,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    CHOICE_KINDS,
    GenerationChoiceCandidateMapper,
    GenerationChoiceHypothesis,
)


ASSESSMENT_SELECTOR_REASONS = (
    "DISABLED_BASELINE",
    "UNIQUE_ACTIVE_ASSESSMENT",
    "NO_ACTIVE_BASELINE_NE",
    "MULTIPLE_ACTIVE_BASELINE_NE",
)
ASSESSMENT_CANDIDATE_STATES = (
    "NOT_READ_DISABLED",
    "UNREGISTERED",
    "REGISTERED_INACTIVE",
    "ACTIVE",
)


class GenerationChoiceAssessmentSelectorError(ValueError):
    """assessment selector 输入不完整或候选投影发生漂移。"""


def _strict_key(value: tuple[int, ...], *, where: str) -> tuple[int, ...]:
    if not isinstance(value, tuple) or not value:
        raise GenerationChoiceAssessmentSelectorError(f"{where} 必须为非空 tuple")
    assert_int(*value, _where=where)
    if any(type(item) is not int for item in value):
        raise GenerationChoiceAssessmentSelectorError(f"{where} 必须使用严格整数")
    return value


def _packed(value: tuple[int, ...]) -> tuple[int, ...]:
    return len(value), *value


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GenerationChoiceAssessmentSelectorPolicy:
    """冻结关闭层和选择 trace 的纯整数命名空间。"""

    trace_namespace: tuple[int, ...]
    disabled_choice_kinds: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _strict_key(self.trace_namespace, where="assessment selector trace namespace")
        if (not isinstance(self.disabled_choice_kinds, tuple)
                or any(item not in CHOICE_KINDS
                       for item in self.disabled_choice_kinds)
                or len(set(self.disabled_choice_kinds))
                != len(self.disabled_choice_kinds)):
            raise GenerationChoiceAssessmentSelectorError(
                "assessment selector disabled layers 非法")
        object.__setattr__(self, "disabled_choice_kinds", tuple(
            item for item in CHOICE_KINDS
            if item in self.disabled_choice_kinds))


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GenerationChoiceAssessmentCandidateProjection:
    """一个显式 option 的精确 H-05 读取结果。"""

    choice: GenerationChoiceHypothesis
    candidate: ObjectIdentity
    hypothesis: HypothesisKey | None
    projection: CandidateGraphProjection | None
    state: str

    def __post_init__(self) -> None:
        if not isinstance(self.choice, GenerationChoiceHypothesis):
            raise TypeError("assessment candidate choice 类型错误")
        if not isinstance(self.candidate, ObjectIdentity):
            raise TypeError("assessment candidate identity 类型错误")
        if self.hypothesis is not None and not isinstance(
                self.hypothesis, HypothesisKey):
            raise TypeError("assessment candidate hypothesis 类型错误")
        if self.projection is not None and not isinstance(
                self.projection, CandidateGraphProjection):
            raise TypeError("assessment candidate projection 类型错误")
        if self.state not in ASSESSMENT_CANDIDATE_STATES:
            raise GenerationChoiceAssessmentSelectorError(
                "assessment candidate state 未注册")
        if self.state in {"NOT_READ_DISABLED", "UNREGISTERED"}:
            if self.hypothesis is not None or self.projection is not None:
                raise GenerationChoiceAssessmentSelectorError(
                    "未读取/未登记 candidate 不得携带 H-05 投影")
        elif self.hypothesis is None:
            raise GenerationChoiceAssessmentSelectorError(
                "已登记 candidate 缺 Hypothesis")
        if self.state == "ACTIVE" and self.projection is None:
            raise GenerationChoiceAssessmentSelectorError(
                "active candidate 缺 lifecycle projection")

    @property
    def active(self) -> int:
        return int(self.state == "ACTIVE")

    def trace_key(self) -> tuple[int, ...]:
        values = [
            ASSESSMENT_CANDIDATE_STATES.index(self.state),
            *_packed(self.candidate.stable_key()),
        ]
        if self.hypothesis is not None:
            values.extend(_packed(self.hypothesis.stable_key()))
        if self.projection is not None:
            values.extend(_packed(self.projection.state.stable_key()))
            values.append(len(self.projection.history))
            for item in self.projection.history:
                values.extend(_packed(item.definition.event.stable_key()))
        return integer_tuple_fingerprint(
            tuple(values),
            domain="generation.choice.assessment.selector.projection.v1",
        )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GenerationChoiceAssessmentSelection:
    """完整 option 分母、候选投影、selected、reason 与 trace。"""

    options: tuple[GenerationChoiceHypothesis, ...]
    baseline: GenerationChoiceHypothesis
    candidate_projections: tuple[
        GenerationChoiceAssessmentCandidateProjection, ...]
    selected: GenerationChoiceHypothesis
    reason: str
    trace: tuple[int, ...]

    def __post_init__(self) -> None:
        if (not isinstance(self.options, tuple) or not self.options
                or any(not isinstance(item, GenerationChoiceHypothesis)
                       for item in self.options)
                or len(set(self.options)) != len(self.options)):
            raise GenerationChoiceAssessmentSelectorError(
                "assessment selector options 非法")
        if self.baseline not in self.options or self.selected not in self.options:
            raise GenerationChoiceAssessmentSelectorError(
                "assessment selector baseline/selected 不属于 options")
        if (not isinstance(self.candidate_projections, tuple)
                or len(self.candidate_projections) != len(self.options)
                or any(not isinstance(
                    item, GenerationChoiceAssessmentCandidateProjection)
                    for item in self.candidate_projections)
                or tuple(item.choice for item in self.candidate_projections)
                != self.options):
            raise GenerationChoiceAssessmentSelectorError(
                "assessment selector candidate projections 未覆盖 options")
        if self.reason not in ASSESSMENT_SELECTOR_REASONS:
            raise GenerationChoiceAssessmentSelectorError(
                "assessment selector reason 未注册")
        _strict_key(self.trace, where="assessment selector trace")
        active = tuple(
            item.choice for item in self.candidate_projections if item.active)
        if self.reason == "UNIQUE_ACTIVE_ASSESSMENT":
            if len(active) != 1 or self.selected != active[0]:
                raise GenerationChoiceAssessmentSelectorError(
                    "unique active selection 与投影不一致")
        else:
            if self.selected != self.baseline:
                raise GenerationChoiceAssessmentSelectorError(
                    "非唯一 assessment 决策不得替换 baseline")
            if (self.reason == "DISABLED_BASELINE"
                    and any(item.state != "NOT_READ_DISABLED"
                            for item in self.candidate_projections)):
                raise GenerationChoiceAssessmentSelectorError(
                    "disabled selection 不得携带 assessment 读取结果")
            if self.reason == "NO_ACTIVE_BASELINE_NE" and active:
                raise GenerationChoiceAssessmentSelectorError(
                    "no-active selection 含 active candidate")
            if (self.reason == "MULTIPLE_ACTIVE_BASELINE_NE"
                    and len(active) < 2):
                raise GenerationChoiceAssessmentSelectorError(
                    "multiple-active selection 缺多个 active candidate")

    @property
    def ne(self) -> int:
        return int(self.reason in {
            "NO_ACTIVE_BASELINE_NE", "MULTIPLE_ACTIVE_BASELINE_NE"})


def _validate_options(
        options: tuple[GenerationChoiceHypothesis, ...],
        baseline: GenerationChoiceHypothesis,
        ) -> None:
    if (not isinstance(options, tuple) or not options
            or any(not isinstance(item, GenerationChoiceHypothesis)
                   for item in options)
            or len(set(options)) != len(options)):
        raise GenerationChoiceAssessmentSelectorError(
            "assessment selector 必须接收非空唯一 option tuple")
    if baseline not in options:
        raise GenerationChoiceAssessmentSelectorError(
            "assessment selector baseline 不属于 options")
    base = options[0]
    if len({item.candidate for item in options}) != len(options):
        raise GenerationChoiceAssessmentSelectorError(
            "assessment selector option candidate 重复")
    if len({item.selected_object for item in options}) != len(options):
        raise GenerationChoiceAssessmentSelectorError(
            "assessment selector option action 重复")
    for item in options[1:]:
        if (item.choice_kind != base.choice_kind
                or item.competition_key != base.competition_key
                or item.condition != base.condition
                or item.condition.context != base.condition.context
                or item.target_obligation != base.target_obligation
                or item.authorized_scope.owner != base.authorized_scope.owner
                or item.authorized_scope.versions
                != base.authorized_scope.versions):
            raise GenerationChoiceAssessmentSelectorError(
                "assessment selector options 越过 kind/competition/context/owner/version")


def _read_candidate(
        mapper: GenerationChoiceCandidateMapper,
        learning: CandidateLearningRuntime,
        choice: GenerationChoiceHypothesis,
        ) -> GenerationChoiceAssessmentCandidateProjection:
    candidate = mapper.candidate_identity(choice)
    definition = mapper.definition(choice)
    try:
        hypothesis = learning.hypothesis_for_candidate(candidate)
    except KeyError:
        return GenerationChoiceAssessmentCandidateProjection(
            choice, candidate, None, None, "UNREGISTERED")
    if learning.engine.definition(hypothesis) != definition:
        raise GenerationChoiceAssessmentSelectorError(
            "assessment selector exact candidate 定义漂移")
    projection = learning.lifecycle_projection_if_available(candidate)
    active = learning.engine.active(hypothesis)
    if active is None:
        if (projection is not None
                and projection.state == learning.graph.protocol.active_state):
            raise GenerationChoiceAssessmentSelectorError(
                "assessment selector H-04 与 active graph 投影漂移")
        state = "REGISTERED_INACTIVE"
    else:
        if (active.definition != definition or projection is None
                or projection.state != learning.graph.protocol.active_state):
            raise GenerationChoiceAssessmentSelectorError(
                "assessment selector active candidate 投影漂移")
        state = "ACTIVE"
    return GenerationChoiceAssessmentCandidateProjection(
        choice, candidate, hypothesis, projection, state)


def select_generation_choice_by_assessment(
        mapper: GenerationChoiceCandidateMapper,
        learning: CandidateLearningRuntime,
        policy: GenerationChoiceAssessmentSelectorPolicy,
        options: tuple[GenerationChoiceHypothesis, ...],
        baseline: GenerationChoiceHypothesis,
        ) -> GenerationChoiceAssessmentSelection:
    """只读显式 options；唯一 active 可替换，其余情况保留 baseline。"""
    if not isinstance(mapper, GenerationChoiceCandidateMapper):
        raise TypeError("assessment selector mapper 类型错误")
    if not isinstance(learning, CandidateLearningRuntime):
        raise TypeError("assessment selector learning 类型错误")
    if not isinstance(policy, GenerationChoiceAssessmentSelectorPolicy):
        raise TypeError("assessment selector policy 类型错误")
    _validate_options(options, baseline)
    choice_kind = options[0].choice_kind
    if choice_kind in policy.disabled_choice_kinds:
        projections = tuple(
            GenerationChoiceAssessmentCandidateProjection(
                item, mapper.candidate_identity(item), None, None,
                "NOT_READ_DISABLED")
            for item in options)
        selected = baseline
        reason = "DISABLED_BASELINE"
    else:
        projections = tuple(
            _read_candidate(mapper, learning, item) for item in options)
        active = tuple(item.choice for item in projections if item.active)
        if len(active) == 1:
            selected = active[0]
            reason = "UNIQUE_ACTIVE_ASSESSMENT"
        elif not active:
            selected = baseline
            reason = "NO_ACTIVE_BASELINE_NE"
        else:
            selected = baseline
            reason = "MULTIPLE_ACTIVE_BASELINE_NE"
    trace_values = [
        ASSESSMENT_SELECTOR_REASONS.index(reason),
        len(options),
        options.index(baseline),
        options.index(selected),
    ]
    for item in projections:
        trace_values.extend(_packed(item.trace_key()))
    trace = (
        *policy.trace_namespace,
        *integer_tuple_fingerprint(
            tuple(trace_values),
            domain="generation.choice.assessment.selector.trace.v1",
        ),
    )
    return GenerationChoiceAssessmentSelection(
        options, baseline, projections, selected, reason, trace)


__all__ = [
    "ASSESSMENT_CANDIDATE_STATES",
    "ASSESSMENT_SELECTOR_REASONS",
    "GenerationChoiceAssessmentCandidateProjection",
    "GenerationChoiceAssessmentSelection",
    "GenerationChoiceAssessmentSelectorError",
    "GenerationChoiceAssessmentSelectorPolicy",
    "select_generation_choice_by_assessment",
]
