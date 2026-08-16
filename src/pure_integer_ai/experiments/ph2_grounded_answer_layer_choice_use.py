"""从 grounded-answer 真实 stance/content 层恢复两类 actual choice。"""
from __future__ import annotations

from dataclasses import dataclass, replace

from pure_integer_ai.cognition.shared.generation_plan import (
    GenerationCandidate,
    GenerationLayerResult,
    GenerationPlanningRequest,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_HYPOTHESIS,
    ObjectIdentity,
    concept_identity,
)
from pure_integer_ai.crosscut.determinism.fingerprint import (
    integer_tuple_fingerprint,
)
from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    CHOICE_KINDS,
    GenerationChoiceCondition,
    GenerationChoiceHypothesis,
    GenerationChoiceUseRef,
    LosslessIntegerKey,
)
from pure_integer_ai.experiments.ph2_grounded_answer_choice import (
    grounded_answer_generation_context,
)
from pure_integer_ai.experiments.ph2_grounded_answer_choice_use import (
    grounded_answer_actual_execution,
)
from pure_integer_ai.experiments.ph2_grounded_answer_runtime_factory import (
    GroundedAnswerRunLocalInstallation,
)
from pure_integer_ai.experiments.question_answer_runtime import (
    QuestionAnswerRun,
)


_NAMESPACE = 20970
_RECOVERABLE_KINDS = ("CONTENT_CHOICE", "COMMUNICATIVE_TASK_CHOICE")


# object-model: exception
class GroundedAnswerLayerChoiceUseError(ValueError):
    """stance/content actual choice 未绑定同次 generation layer。"""


def _packed(key: tuple[int, ...]) -> tuple[int, ...]:
    """给开放稳定键增加长度边界。"""
    return len(key), *key


def _owned_identity(
        values: tuple[int, ...],
        candidate: GenerationCandidate,
        ) -> ObjectIdentity:
    """在当前 query owner/version 中建立 choice Hypothesis 身份。"""
    return ObjectIdentity(
        OBJECT_HYPOTHESIS,
        integer_tuple_fingerprint(
            values, domain="grounded.answer.layer.choice.identity.v1"),
        candidate.scope.owner,
        candidate.scope.versions,
    )


def _condition(
        installation: GroundedAnswerRunLocalInstallation,
        choice_kind: str,
        ) -> GenerationChoiceCondition:
    """建立共享 generation context 下的分层条件，不编码已选动作。"""
    if choice_kind not in _RECOVERABLE_KINDS:
        raise GroundedAnswerLayerChoiceUseError(
            "grounded layer choice kind 当前不可恢复")
    variant = installation.variant
    candidate = installation.candidate
    context = grounded_answer_generation_context(variant, candidate)
    condition = concept_identity(
        integer_tuple_fingerprint(
            (*context.stable_key(), CHOICE_KINDS.index(choice_kind)),
            domain="grounded.answer.layer.choice.condition.v1",
        ),
        owner=candidate.scope.owner,
        versions=candidate.scope.versions,
    )
    required = tuple(sorted((
        variant.template.language_branch,
        variant.template.proposition_structure,
        variant.template.predicate,
    ), key=ObjectIdentity.stable_key))
    return GenerationChoiceCondition(
        condition,
        context,
        required,
        (),
        candidate.scope,
    )


def _competition_key(
        installation: GroundedAnswerRunLocalInstallation,
        run: QuestionAnswerRun,
        choice_kind: str,
        ) -> tuple[int, ...]:
    """保存该层完整竞争域，不把最终 selected object 写入竞争键。"""
    planning = installation.planning
    condition = _condition(installation, choice_kind)
    values = [
        CHOICE_KINDS.index(choice_kind),
        *_packed(planning.goal.stable_key()),
        *_packed(condition.context.stable_key()),
    ]
    if choice_kind == "CONTENT_CHOICE":
        values.append(len(planning.candidates))
        for candidate in planning.candidates:
            values.extend(_packed(candidate.stable_key()))
    else:
        if run.selection is None:
            raise GroundedAnswerLayerChoiceUseError(
                "grounded task competition 缺少 G-01 selection")
        values.extend(_packed(run.selection.protocol.stable_key()))
    return integer_tuple_fingerprint(
        tuple(values), domain="grounded.answer.layer.choice.competition.v1")


def _base_choice(
        installation: GroundedAnswerRunLocalInstallation,
        run: QuestionAnswerRun,
        choice_kind: str,
        ) -> GenerationChoiceHypothesis:
    """从已发生的 G-01 selection 恢复分层 choice，不读取 G-04。"""
    selection = run.selection
    if selection is None:
        raise GroundedAnswerLayerChoiceUseError(
            "grounded layer choice 缺少 G-01 selection")
    candidate = installation.candidate
    if choice_kind == "CONTENT_CHOICE":
        selected = candidate.proposition.template
        target = installation.planning.goal.proposition.template
    elif choice_kind == "COMMUNICATIVE_TASK_CHOICE":
        selected = selection.stance
        target = installation.planning.goal.goal_kind
    else:
        raise GroundedAnswerLayerChoiceUseError(
            "grounded layer choice kind 当前不可恢复")
    condition = _condition(installation, choice_kind)
    competition_key = _competition_key(installation, run, choice_kind)
    identity = _owned_identity(
        (*competition_key, *selected.stable_key()), candidate)
    return GenerationChoiceHypothesis(
        identity,
        choice_kind,
        target,
        condition,
        selected,
        (candidate.source,),
        competition_key,
        candidate.scope,
    )


def _layer(
        installation: GroundedAnswerRunLocalInstallation,
        run: QuestionAnswerRun,
        choice_kind: str,
        ) -> GenerationLayerResult:
    """按注入协议身份恢复实际 stance 或 content layer。"""
    if run.generation is None:
        raise GroundedAnswerLayerChoiceUseError(
            "grounded layer choice 缺少 generation")
    protocol = run.generation.plan.protocol
    expected = (
        protocol.content_layer
        if choice_kind == "CONTENT_CHOICE"
        else protocol.stance_layer
    )
    matches = tuple(
        item for item in run.generation.plan.layers if item.layer == expected)
    if len(matches) != 1:
        raise GroundedAnswerLayerChoiceUseError(
            "grounded layer choice 未恢复唯一 generation layer")
    layer = matches[0]
    selection = run.selection
    if (selection is None
            or not layer.executed
            or layer.payload != selection.stable_key()
            or layer.selected_candidate_keys
            != (installation.candidate.stable_key(),)):
        raise GroundedAnswerLayerChoiceUseError(
            "grounded layer choice 与实际 G-01/layer 漂移")
    return layer


def _selection_key(
        choice: GenerationChoiceHypothesis,
        layer: GenerationLayerResult,
        ) -> LosslessIntegerKey:
    """保存分层 choice 与实际 layer 选择的完整键。"""
    return LosslessIntegerKey((
        _NAMESPACE,
        10,
        CHOICE_KINDS.index(choice.choice_kind),
        *_packed(choice.stable_key()),
        *_packed(layer.stable_key()),
    ))


def _use_key(
        installation: GroundedAnswerRunLocalInstallation,
        run: QuestionAnswerRun,
        layer: GenerationLayerResult,
        selection_key: LosslessIntegerKey,
        ) -> LosslessIntegerKey:
    """保存 query、冻结 planning、generation 与实际 layer 的完整键。"""
    if run.generation is None:
        raise GroundedAnswerLayerChoiceUseError(
            "grounded layer use 缺少 generation")
    query_key = () if run.query is None else run.query.stable_key()
    result_key = (
        () if run.query_result is None else run.query_result.stable_key())
    return LosslessIntegerKey((
        _NAMESPACE,
        20,
        *_packed(run.request.stable_key()),
        *_packed(query_key),
        *_packed(result_key),
        *_packed(installation.planning.stable_key()),
        *_packed(installation.candidate.stable_key()),
        *_packed(selection_key.components),
        *_packed(run.generation.stable_key()),
        *_packed(layer.stable_key()),
        *_packed(installation.candidate.scope.stable_key()),
    ))


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GroundedAnswerLayerChoiceUse:
    """一个 recoverable generation layer 的 actual choice 与 exact Use。"""

    installation: GroundedAnswerRunLocalInstallation
    run: QuestionAnswerRun
    choice_before: GenerationChoiceHypothesis
    layer: GenerationLayerResult
    use: GenerationChoiceUseRef
    choice_after: GenerationChoiceHypothesis

    def __post_init__(self) -> None:
        if not isinstance(
                self.installation, GroundedAnswerRunLocalInstallation):
            raise TypeError("grounded layer use installation 类型错误")
        if not isinstance(self.run, QuestionAnswerRun):
            raise TypeError("grounded layer use run 类型错误")
        if not isinstance(self.choice_before, GenerationChoiceHypothesis):
            raise TypeError("grounded layer use choice_before 类型错误")
        if self.choice_before.choice_kind not in _RECOVERABLE_KINDS:
            raise GroundedAnswerLayerChoiceUseError(
                "grounded layer use choice kind 非法")
        if self.choice_before.exact_uses or self.choice_before.typed_outcomes:
            raise GroundedAnswerLayerChoiceUseError(
                "grounded layer base choice 不得预载 Use/outcome")
        if not isinstance(self.layer, GenerationLayerResult):
            raise TypeError("grounded layer use layer 类型错误")
        if not isinstance(self.use, GenerationChoiceUseRef):
            raise TypeError("grounded layer use ref 类型错误")
        if not isinstance(self.choice_after, GenerationChoiceHypothesis):
            raise TypeError("grounded layer use choice_after 类型错误")
        grounded_answer_actual_execution(
            self.installation.variant,
            self.installation.planning,
            self.installation.candidate,
            self.installation.lexical_choice,
            self.run,
        )
        expected_choice = _base_choice(
            self.installation, self.run, self.choice_before.choice_kind)
        expected_layer = _layer(
            self.installation, self.run, self.choice_before.choice_kind)
        expected_selection = _selection_key(expected_choice, expected_layer)
        expected_use = _use_key(
            self.installation,
            self.run,
            expected_layer,
            expected_selection,
        )
        if (self.choice_before != expected_choice
                or self.layer != expected_layer
                or self.use.use_kind != "CORE_USE"
                or self.use.selection_key != expected_selection
                or self.use.use_key != expected_use
                or self.use.scope != self.choice_before.authorized_scope
                or self.choice_after != replace(
                    self.choice_before, exact_uses=(self.use,))):
            raise GroundedAnswerLayerChoiceUseError(
                "grounded layer use 未精确回填 choice")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GroundedAnswerContentTaskUses:
    """同次 generation 中 content 与 communicative-task 两层账项。"""

    content: GroundedAnswerLayerChoiceUse
    task: GroundedAnswerLayerChoiceUse

    def __post_init__(self) -> None:
        if self.content.choice_before.choice_kind != "CONTENT_CHOICE":
            raise GroundedAnswerLayerChoiceUseError(
                "grounded content/task bundle 缺 content choice")
        if self.task.choice_before.choice_kind != "COMMUNICATIVE_TASK_CHOICE":
            raise GroundedAnswerLayerChoiceUseError(
                "grounded content/task bundle 缺 task choice")
        if (self.content.choice_before.condition.context
                != self.task.choice_before.condition.context):
            raise GroundedAnswerLayerChoiceUseError(
                "grounded content/task choice 未共享 generation context")
        if self.content.use.use_key == self.task.use.use_key:
            raise GroundedAnswerLayerChoiceUseError(
                "grounded content/task exact Use 不得跨层复用")


# object-model: runtime-ledger
class GroundedAnswerContentTaskAdoptionLedger:
    """为一个 installation 原子登记 content 与 task 两层 actual Use。"""

    def __init__(self, installation: GroundedAnswerRunLocalInstallation) -> None:
        if not isinstance(installation, GroundedAnswerRunLocalInstallation):
            raise TypeError("grounded content/task ledger installation 类型错误")
        self.installation = installation
        self._records: list[GroundedAnswerContentTaskUses] = []

    @property
    def records(self) -> tuple[GroundedAnswerContentTaskUses, ...]:
        """返回已原子登记的 content/task Use bundle。"""
        return tuple(self._records)

    def _record(
            self,
            run: QuestionAnswerRun,
            choice_kind: str,
            ) -> GroundedAnswerLayerChoiceUse:
        """从同次 run 构造一个分层 choice 与 exact Use。"""
        choice = _base_choice(self.installation, run, choice_kind)
        layer = _layer(self.installation, run, choice_kind)
        selection_key = _selection_key(choice, layer)
        use = GenerationChoiceUseRef(
            "CORE_USE",
            _use_key(
                self.installation, run, layer, selection_key),
            selection_key,
            self.installation.candidate.scope,
        )
        return GroundedAnswerLayerChoiceUse(
            self.installation,
            run,
            choice,
            layer,
            use,
            replace(choice, exact_uses=(use,)),
        )

    def adopt(self, run: QuestionAnswerRun) -> GroundedAnswerContentTaskUses:
        """只在完整同次 generation 后登记两层，拒绝重复累计。"""
        if not isinstance(run, QuestionAnswerRun):
            raise TypeError("grounded content/task ledger run 类型错误")
        grounded_answer_actual_execution(
            self.installation.variant,
            self.installation.planning,
            self.installation.candidate,
            self.installation.lexical_choice,
            run,
        )
        result = GroundedAnswerContentTaskUses(
            self._record(run, "CONTENT_CHOICE"),
            self._record(run, "COMMUNICATIVE_TASK_CHOICE"),
        )
        keys = {result.content.use.use_key, result.task.use.use_key}
        if any(
                item.content.use.use_key in keys
                or item.task.use.use_key in keys
                for item in self._records):
            raise GroundedAnswerLayerChoiceUseError(
                "同一次 grounded content/task execution 不得重复登记")
        self._records.append(result)
        return result


__all__ = [
    "GroundedAnswerContentTaskAdoptionLedger",
    "GroundedAnswerContentTaskUses",
    "GroundedAnswerLayerChoiceUse",
    "GroundedAnswerLayerChoiceUseError",
]
