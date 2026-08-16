"""记录 grounded-answer 显式 structure selection 的 actual choice 与 Use。"""
from __future__ import annotations

from dataclasses import dataclass, replace

from pure_integer_ai.cognition.shared.generation_plan import (
    GenerationLayerResult,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_HYPOTHESIS,
    ObjectIdentity,
    SourceRef,
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


_NAMESPACE = 20980
_CHOICE_KIND = "PROPOSITION_STRUCTURE_CHOICE"


# object-model: exception
class GroundedAnswerStructureChoiceUseError(ValueError):
    """结构 choice 未绑定显式 selection、syntax layer 或 S-07 execution。"""


def _packed(key: tuple[int, ...]) -> tuple[int, ...]:
    """给开放整数键增加长度边界。"""
    return len(key), *key


def _condition(
        installation: GroundedAnswerRunLocalInstallation,
        ) -> GenerationChoiceCondition:
    """建立不含 selected structure 的共享 generation 条件。"""
    variant = installation.variant
    candidate = installation.candidate
    context = grounded_answer_generation_context(variant, candidate)
    condition = concept_identity(
        integer_tuple_fingerprint(
            (*context.stable_key(), CHOICE_KINDS.index(_CHOICE_KIND)),
            domain="grounded.answer.structure.choice.condition.v1",
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
        ) -> tuple[int, ...]:
    """保存全部 structure options，不把 selected option 编入竞争键。"""
    condition = _condition(installation)
    values = [
        CHOICE_KINDS.index(_CHOICE_KIND),
        *_packed(installation.planning.goal.stable_key()),
        *_packed(condition.context.stable_key()),
        len(installation.structure_selection.options),
    ]
    for option in installation.structure_selection.options:
        values.extend(_packed(option.choice_key()))
    return integer_tuple_fingerprint(
        tuple(values), domain="grounded.answer.structure.choice.competition.v1")


def _base_choice(
        installation: GroundedAnswerRunLocalInstallation,
        ) -> GenerationChoiceHypothesis:
    """从 factory 在 syntax mapper 前保留的显式 selection 建立 GG-01 choice。"""
    candidate = installation.candidate
    selected = installation.structure_selection.selected
    condition = _condition(installation)
    competition_key = _competition_key(installation)
    identity = ObjectIdentity(
        OBJECT_HYPOTHESIS,
        integer_tuple_fingerprint(
            (*competition_key, *selected.structure.stable_key()),
            domain="grounded.answer.structure.choice.identity.v1",
        ),
        candidate.scope.owner,
        candidate.scope.versions,
    )
    source = SourceRef(
        _NAMESPACE,
        selected.structure_id,
        len(selected.pattern_ids),
        candidate.scope.owner,
        candidate.scope.versions,
    )
    return GenerationChoiceHypothesis(
        identity,
        _CHOICE_KIND,
        candidate.proposition.template,
        condition,
        selected.structure,
        (candidate.source, source),
        competition_key,
        candidate.scope,
    )


def _syntax_layer(
        installation: GroundedAnswerRunLocalInstallation,
        run: QuestionAnswerRun,
        ) -> GenerationLayerResult:
    """恢复实际 syntax layer 并核对 selected structure 的生产消费。"""
    if run.generation is None or run.generation.surface is None:
        raise GroundedAnswerStructureChoiceUseError(
            "grounded structure choice 缺少完整 generation surface")
    protocol = run.generation.plan.protocol
    matches = tuple(
        item for item in run.generation.plan.layers
        if item.layer == protocol.syntax_layer)
    if len(matches) != 1:
        raise GroundedAnswerStructureChoiceUseError(
            "grounded structure choice 未恢复唯一 syntax layer")
    layer = matches[0]
    request = run.generation.surface.preview.request
    syntax = request.structure.syntax
    selected = installation.structure_selection.selected
    if (not layer.executed
            or layer.payload != syntax.stable_key()
            or layer.selected_candidate_keys
            != (installation.candidate.stable_key(),)
            or len(syntax.sentences) != 1
            or syntax.sentences[0].structure != selected.structure
            or syntax.sentences[0].sentence != selected.sentence
            or syntax.sentences[0].slots != selected.slots
            or request.execution.request.syntax != syntax
            or not request.execution.complete):
        raise GroundedAnswerStructureChoiceUseError(
            "grounded structure choice 未被 syntax/S-07 execution 实际消费")
    return layer


def _selection_key(
        choice: GenerationChoiceHypothesis,
        installation: GroundedAnswerRunLocalInstallation,
        layer: GenerationLayerResult,
        ) -> LosslessIntegerKey:
    """保存 structure choice、完整竞争选择 artifact 与 syntax layer。"""
    return LosslessIntegerKey((
        _NAMESPACE,
        10,
        *_packed(choice.stable_key()),
        *_packed(installation.structure_selection.stable_key()),
        *_packed(layer.stable_key()),
    ))


def _use_key(
        installation: GroundedAnswerRunLocalInstallation,
        run: QuestionAnswerRun,
        layer: GenerationLayerResult,
        selection_key: LosslessIntegerKey,
        ) -> LosslessIntegerKey:
    """保存 query、generation、G-02 structure 与实际 S-07 execution。"""
    if run.generation is None or run.generation.surface is None:
        raise GroundedAnswerStructureChoiceUseError(
            "grounded structure use 缺少 generation")
    query_key = () if run.query is None else run.query.stable_key()
    result_key = (
        () if run.query_result is None else run.query_result.stable_key())
    request = run.generation.surface.preview.request
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
        *_packed(request.structure.stable_key()),
        *_packed(request.execution.stable_key()),
        *_packed(installation.candidate.scope.stable_key()),
    ))


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GroundedAnswerStructureChoiceUse:
    """一个显式 structure choice 与实际 syntax/S-07 Use 的不可变账项。"""

    installation: GroundedAnswerRunLocalInstallation
    run: QuestionAnswerRun
    choice_before: GenerationChoiceHypothesis
    layer: GenerationLayerResult
    use: GenerationChoiceUseRef
    choice_after: GenerationChoiceHypothesis

    def __post_init__(self) -> None:
        if not isinstance(
                self.installation, GroundedAnswerRunLocalInstallation):
            raise TypeError("grounded structure use installation 类型错误")
        if not isinstance(self.run, QuestionAnswerRun):
            raise TypeError("grounded structure use run 类型错误")
        if not isinstance(self.choice_before, GenerationChoiceHypothesis):
            raise TypeError("grounded structure use choice_before 类型错误")
        if not isinstance(self.layer, GenerationLayerResult):
            raise TypeError("grounded structure use layer 类型错误")
        if not isinstance(self.use, GenerationChoiceUseRef):
            raise TypeError("grounded structure use ref 类型错误")
        if not isinstance(self.choice_after, GenerationChoiceHypothesis):
            raise TypeError("grounded structure use choice_after 类型错误")
        grounded_answer_actual_execution(
            self.installation.variant,
            self.installation.planning,
            self.installation.candidate,
            self.installation.lexical_choice,
            self.run,
        )
        expected_choice = _base_choice(self.installation)
        expected_layer = _syntax_layer(self.installation, self.run)
        expected_selection = _selection_key(
            expected_choice, self.installation, expected_layer)
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
                or self.use.scope != self.installation.candidate.scope
                or self.choice_after != replace(
                    self.choice_before, exact_uses=(self.use,))):
            raise GroundedAnswerStructureChoiceUseError(
                "grounded structure use 未精确绑定 actual selection/execution")


# object-model: runtime-ledger
class GroundedAnswerStructureAdoptionLedger:
    """为一个 installation 登记不可重复的 structure actual Use。"""

    def __init__(self, installation: GroundedAnswerRunLocalInstallation) -> None:
        if not isinstance(installation, GroundedAnswerRunLocalInstallation):
            raise TypeError("grounded structure ledger installation 类型错误")
        self.installation = installation
        self._records: list[GroundedAnswerStructureChoiceUse] = []

    @property
    def records(self) -> tuple[GroundedAnswerStructureChoiceUse, ...]:
        """返回已提交的全部 structure choice Use。"""
        return tuple(self._records)

    def adopt(self, run: QuestionAnswerRun) -> GroundedAnswerStructureChoiceUse:
        """只在 structure 已被 syntax/S-07 实际消费后登记 exact Use。"""
        if not isinstance(run, QuestionAnswerRun):
            raise TypeError("grounded structure ledger run 类型错误")
        grounded_answer_actual_execution(
            self.installation.variant,
            self.installation.planning,
            self.installation.candidate,
            self.installation.lexical_choice,
            run,
        )
        choice = _base_choice(self.installation)
        layer = _syntax_layer(self.installation, run)
        selection_key = _selection_key(choice, self.installation, layer)
        use = GenerationChoiceUseRef(
            "CORE_USE",
            _use_key(
                self.installation, run, layer, selection_key),
            selection_key,
            self.installation.candidate.scope,
        )
        if any(item.use.use_key == use.use_key for item in self._records):
            raise GroundedAnswerStructureChoiceUseError(
                "同一次 grounded structure execution 不得重复登记")
        record = GroundedAnswerStructureChoiceUse(
            self.installation,
            run,
            choice,
            layer,
            use,
            replace(choice, exact_uses=(use,)),
        )
        self._records.append(record)
        return record


__all__ = [
    "GroundedAnswerStructureAdoptionLedger",
    "GroundedAnswerStructureChoiceUse",
    "GroundedAnswerStructureChoiceUseError",
]
