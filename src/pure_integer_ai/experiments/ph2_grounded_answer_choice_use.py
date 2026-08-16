"""把 grounded-answer lexical choice 绑定到一次真实生成采用。"""
from __future__ import annotations

from dataclasses import dataclass, replace

from pure_integer_ai.cognition.shared.generation_content import (
    AnswerContentSelection,
)
from pure_integer_ai.cognition.shared.generation_execution import (
    TypedGenerationExecution,
)
from pure_integer_ai.cognition.shared.generation_plan import (
    GenerationCandidate,
    GenerationPlanningRequest,
)
from pure_integer_ai.cognition.shared.generation_surface import (
    SurfaceAdoption,
)
from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    GenerationChoiceHypothesis,
    GenerationChoiceUseRef,
    LosslessIntegerKey,
)
from pure_integer_ai.experiments.ph2_grounded_answer_connector import (
    GroundedAnswerConnectorVariant,
)
from pure_integer_ai.experiments.ph2_grounded_answer_runtime_factory import (
    GroundedAnswerRunLocalInstallation,
)
from pure_integer_ai.experiments.question_answer_runtime import (
    QuestionAnswerRun,
)


_NAMESPACE = 20960


# object-model: exception
class GroundedAnswerLexicalUseError(ValueError):
    """lexical Use 未绑定同次选择、connector、生成或 R-01 采用。"""


def _packed(key: tuple[int, ...]) -> tuple[int, ...]:
    """给开放稳定键增加长度边界，保留全部整数而不做摘要。"""
    return len(key), *key


def _selection_key(
        choice: GenerationChoiceHypothesis,
        selection: AnswerContentSelection,
        variant: GroundedAnswerConnectorVariant,
        ) -> LosslessIntegerKey:
    """保存 GG-01 候选、G-01 选择与实际 connector 的完整键。"""
    return LosslessIntegerKey((
        _NAMESPACE,
        10,
        *_packed(choice.stable_key()),
        *_packed(selection.stable_key()),
        *_packed(variant.template.connector.stable_key()),
    ))


def _use_key(
        run: QuestionAnswerRun,
        planning: GenerationPlanningRequest,
        candidate: GenerationCandidate,
        selection_key: LosslessIntegerKey,
        generation: TypedGenerationExecution,
        adoptions: tuple[SurfaceAdoption, ...],
        ) -> LosslessIntegerKey:
    """保存本次 query、planning、生成和逐点 R-01 采用的完整键。"""
    query_key = () if run.query is None else run.query.stable_key()
    result_key = (
        () if run.query_result is None else run.query_result.stable_key())
    values = [
        _NAMESPACE,
        20,
        *_packed(run.request.stable_key()),
        *_packed(query_key),
        *_packed(result_key),
        *_packed(planning.stable_key()),
        *_packed(candidate.stable_key()),
        *_packed(selection_key.components),
        *_packed(generation.stable_key()),
        len(adoptions),
    ]
    for adoption in adoptions:
        values.extend(_packed(adoption.stable_key()))
    values.extend(_packed(candidate.scope.stable_key()))
    return LosslessIntegerKey(tuple(values))


def grounded_answer_actual_execution(
        variant: GroundedAnswerConnectorVariant,
        planning: GenerationPlanningRequest,
        candidate: GenerationCandidate,
        choice: GenerationChoiceHypothesis,
        run: QuestionAnswerRun,
        ) -> tuple[
            AnswerContentSelection,
            TypedGenerationExecution,
            tuple[SurfaceAdoption, ...],
        ]:
    """核验 run 确实由同一冻结 planning 和 selected variant 产生。"""
    if choice.exact_uses or choice.typed_outcomes:
        raise GroundedAnswerLexicalUseError(
            "grounded lexical base choice 必须尚未登记 Use/outcome")
    if (choice.choice_kind != "LEXICAL_REALIZATION_CHOICE"
            or choice.selected_object != variant.template.connector
            or choice.authorized_scope != candidate.scope):
        raise GroundedAnswerLexicalUseError(
            "grounded lexical choice 与 selected connector/scope 漂移")
    if planning.candidates != (candidate,):
        raise GroundedAnswerLexicalUseError(
            "grounded lexical Use 只接受 installation 的单一候选")
    if run.planning_request != planning:
        raise GroundedAnswerLexicalUseError(
            "grounded lexical Use 替换了冻结 planning")
    selection = run.selection
    if not isinstance(selection, AnswerContentSelection):
        raise GroundedAnswerLexicalUseError(
            "grounded lexical Use 缺少实际 G-01 selection")
    if (selection.request != planning
            or selection.selected_candidate_keys
            != (candidate.stable_key(),)):
        raise GroundedAnswerLexicalUseError(
            "grounded lexical Use 未精确选择冻结 candidate")
    generation = run.generation
    if (not isinstance(generation, TypedGenerationExecution)
            or not generation.complete
            or generation.plan.request != planning
            or generation.surface is None):
        raise GroundedAnswerLexicalUseError(
            "grounded lexical Use 只能登记完整同次 generation")

    preview = generation.surface.preview
    request = preview.request
    template = variant.template
    if (request.structure.selection != selection
            or request.branch != template.language_branch):
        raise GroundedAnswerLexicalUseError(
            "grounded lexical Use 的 G-03 request 与 G-01/branch 漂移")
    sentences = request.structure.syntax.sentences
    if len(sentences) != 1:
        raise GroundedAnswerLexicalUseError(
            "首轮 grounded lexical Use 只接受单句 generation")
    sentence = sentences[0]
    if (sentence.sentence != template.sentence
            or sentence.structure != template.structure
            or sentence.proposition_keys != (candidate.stable_key(),)
            or sentence.source != candidate.source
            or sentence.scope != candidate.scope
            or sentence.instance is None
            or sentence.instance.template != template.sentence
            or sentence.instance.candidate_key != candidate.stable_key()):
        raise GroundedAnswerLexicalUseError(
            "grounded lexical Use 未由 selected connector 句式产生")

    expected = tuple(
        (item.slot, item.filler, item.representation)
        for item in sorted(variant.aliases, key=lambda value: value.part_ordinal)
    )
    actual = tuple(
        (item.value.slot, item.value.filler, item.representation)
        for item in preview.slots
    )
    if actual != expected:
        raise GroundedAnswerLexicalUseError(
            "grounded lexical Use 的实际 slot/Representation 漂移")
    adoptions = generation.surface.adoptions
    if not adoptions or len(adoptions) != len(expected):
        raise GroundedAnswerLexicalUseError(
            "grounded lexical Use 缺少逐槽实际 R-01 adoption")
    return selection, generation, adoptions


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GroundedAnswerLexicalUse:
    """一次 lexical choice、真实生成与 R-01 采用的不可变账项。"""

    variant: GroundedAnswerConnectorVariant
    planning: GenerationPlanningRequest
    candidate: GenerationCandidate
    choice_before: GenerationChoiceHypothesis
    run: QuestionAnswerRun
    adoptions: tuple[SurfaceAdoption, ...]
    use: GenerationChoiceUseRef
    choice_after: GenerationChoiceHypothesis

    def __post_init__(self) -> None:
        if not isinstance(self.variant, GroundedAnswerConnectorVariant):
            raise TypeError("grounded lexical use variant 类型错误")
        if not isinstance(self.planning, GenerationPlanningRequest):
            raise TypeError("grounded lexical use planning 类型错误")
        if not isinstance(self.candidate, GenerationCandidate):
            raise TypeError("grounded lexical use candidate 类型错误")
        if not isinstance(self.choice_before, GenerationChoiceHypothesis):
            raise TypeError("grounded lexical use choice_before 类型错误")
        if not isinstance(self.run, QuestionAnswerRun):
            raise TypeError("grounded lexical use run 类型错误")
        if (not isinstance(self.adoptions, tuple)
                or any(not isinstance(item, SurfaceAdoption)
                       for item in self.adoptions)):
            raise TypeError("grounded lexical use adoptions 类型错误")
        if not isinstance(self.use, GenerationChoiceUseRef):
            raise TypeError("grounded lexical use ref 类型错误")
        if not isinstance(self.choice_after, GenerationChoiceHypothesis):
            raise TypeError("grounded lexical use choice_after 类型错误")

        selection, generation, actual_adoptions = grounded_answer_actual_execution(
            self.variant,
            self.planning,
            self.candidate,
            self.choice_before,
            self.run,
        )
        if self.adoptions != actual_adoptions:
            raise GroundedAnswerLexicalUseError(
                "grounded lexical use 未保留实际 R-01 adoption")
        expected_selection = _selection_key(
            self.choice_before, selection, self.variant)
        expected_use = _use_key(
            self.run,
            self.planning,
            self.candidate,
            expected_selection,
            generation,
            self.adoptions,
        )
        if (self.use.use_kind != "CORE_USE"
                or self.use.selection_key != expected_selection
                or self.use.use_key != expected_use
                or self.use.scope != self.candidate.scope):
            raise GroundedAnswerLexicalUseError(
                "grounded lexical use ref 未绑定完整执行与授权 scope")
        expected_choice = replace(
            self.choice_before, exact_uses=(self.use,))
        if self.choice_after != expected_choice:
            raise GroundedAnswerLexicalUseError(
                "grounded lexical use 未精确回填 GG-01 choice")


# object-model: runtime-ledger
class GroundedAnswerLexicalAdoptionLedger:
    """只为一个 run-local installation 登记不可重复的 lexical Use。"""

    def __init__(self, installation: GroundedAnswerRunLocalInstallation) -> None:
        if not isinstance(installation, GroundedAnswerRunLocalInstallation):
            raise TypeError("grounded lexical ledger installation 类型错误")
        self.installation = installation
        self._records: list[GroundedAnswerLexicalUse] = []

    @property
    def records(self) -> tuple[GroundedAnswerLexicalUse, ...]:
        """返回本 installation 已提交的全部不可变 lexical Use。"""
        return tuple(self._records)

    def adopt(self, run: QuestionAnswerRun) -> GroundedAnswerLexicalUse:
        """在完整生成后登记真实 Use，并返回带 exact Use 的新 choice。"""
        if not isinstance(run, QuestionAnswerRun):
            raise TypeError("grounded lexical ledger run 类型错误")
        installation = self.installation
        selection, generation, adoptions = grounded_answer_actual_execution(
            installation.variant,
            installation.planning,
            installation.candidate,
            installation.lexical_choice,
            run,
        )
        selection_key = _selection_key(
            installation.lexical_choice,
            selection,
            installation.variant,
        )
        use = GenerationChoiceUseRef(
            "CORE_USE",
            _use_key(
                run,
                installation.planning,
                installation.candidate,
                selection_key,
                generation,
                adoptions,
            ),
            selection_key,
            installation.candidate.scope,
        )
        if any(item.use.use_key == use.use_key for item in self._records):
            raise GroundedAnswerLexicalUseError(
                "同一次 grounded lexical execution 不得重复登记")
        choice_after = replace(
            installation.lexical_choice,
            exact_uses=(use,),
        )
        record = GroundedAnswerLexicalUse(
            installation.variant,
            installation.planning,
            installation.candidate,
            installation.lexical_choice,
            run,
            adoptions,
            use,
            choice_after,
        )
        self._records.append(record)
        return record


__all__ = [
    "GroundedAnswerLexicalAdoptionLedger",
    "GroundedAnswerLexicalUse",
    "GroundedAnswerLexicalUseError",
    "grounded_answer_actual_execution",
]
