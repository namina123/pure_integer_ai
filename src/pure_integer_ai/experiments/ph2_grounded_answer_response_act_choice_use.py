"""把 grounded response-act lexical choice 绑定到一次真实生成采用。"""
from __future__ import annotations

from dataclasses import dataclass, replace

from pure_integer_ai.cognition.shared.generation_content import (
    AnswerContentSelection,
)
from pure_integer_ai.cognition.shared.generation_execution import (
    TypedGenerationExecution,
)
from pure_integer_ai.cognition.shared.generation_plan import (
    GenerationPlanningRequest,
)
from pure_integer_ai.cognition.shared.generation_surface import SurfaceAdoption
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONTEXT_SCOPE,
    OBJECT_HYPOTHESIS,
    ObjectIdentity,
    SourceRef,
    concept_identity,
)
from pure_integer_ai.crosscut.determinism.fingerprint import (
    integer_tuple_fingerprint,
)
from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    GenerationChoiceCondition,
    GenerationChoiceHypothesis,
    GenerationChoiceUseRef,
    LosslessIntegerKey,
)
from pure_integer_ai.experiments.ph2_grounded_answer_response_act_compile import (
    GroundedResponseActVariant,
)
from pure_integer_ai.experiments.question_answer_runtime import (
    QuestionAnswerRun,
)


_NAMESPACE = 20964


# object-model: exception
class GroundedResponseActLexicalUseError(ValueError):
    """response-act choice 未绑定同次 selection、surface 或 R-01 adoption。"""


def _packed(key: tuple[int, ...]) -> tuple[int, ...]:
    """给开放稳定键增加长度边界。"""
    return len(key), *key


def _positive_fingerprint(values: tuple[int, ...], *, domain: str) -> int:
    """把内容引用压成稳定正整数，仅用于 SourceRef 路由字段。"""
    fingerprint = integer_tuple_fingerprint(values, domain=domain)
    value = int.from_bytes(bytes(fingerprint[2:10]), "big")
    value &= (1 << 63) - 1
    return value if value > 0 else 1


def _owned_identity(
        kind: int,
        values: tuple[int, ...],
        planning: GenerationPlanningRequest,
        ) -> ObjectIdentity:
    """在当前 query owner/version 内建立内容引用身份。"""
    scope = planning.goal.scope
    return ObjectIdentity(
        kind,
        integer_tuple_fingerprint(
            values, domain="grounded.response.act.lexical.identity.v1"),
        scope.owner,
        scope.versions,
    )


def grounded_response_act_generation_context(
        variant: GroundedResponseActVariant,
        planning: GenerationPlanningRequest,
        ) -> ObjectIdentity:
    """建立不含 selected lexical variant 的同次 response-act 上下文。"""
    if not isinstance(variant, GroundedResponseActVariant):
        raise TypeError("response-act generation context variant 类型错误")
    if not isinstance(planning, GenerationPlanningRequest):
        raise TypeError("response-act generation context planning 类型错误")
    goal = planning.goal
    if goal.target_branch != variant.template.branch:
        raise GroundedResponseActLexicalUseError(
            "response-act generation context branch 漂移")
    return _owned_identity(
        OBJECT_CONTEXT_SCOPE,
        (
            *goal.source.stable_key(),
            *goal.scope.stable_key(),
            *variant.template.stance.stable_key(),
            *variant.template.branch.stable_key(),
        ),
        planning,
    )


def build_grounded_response_act_lexical_choice(
        variant: GroundedResponseActVariant,
        planning: GenerationPlanningRequest,
        ) -> GenerationChoiceHypothesis:
    """把显式 learned response-act pattern 映射为 GG-01 lexical choice。"""
    context = grounded_response_act_generation_context(variant, planning)
    goal = planning.goal
    condition = concept_identity(
        integer_tuple_fingerprint(
            (
                *context.stable_key(),
                *variant.template.stance.stable_key(),
                *variant.template.branch.stable_key(),
            ),
            domain="grounded.response.act.lexical.condition.v1",
        ),
        owner=goal.scope.owner,
        versions=goal.scope.versions,
    )
    required = tuple(sorted((
        variant.template.branch,
        variant.template.stance,
        variant.structure_family,
    ), key=ObjectIdentity.stable_key))
    choice_condition = GenerationChoiceCondition(
        condition,
        context,
        required,
        (),
        goal.scope,
    )
    competition_key = integer_tuple_fingerprint(
        (
            *variant.template.stance.stable_key(),
            *variant.template.branch.stable_key(),
            *choice_condition.stable_key(),
        ),
        domain="grounded.response.act.lexical.competition.v1",
    )
    choice_identity = _owned_identity(
        OBJECT_HYPOTHESIS,
        (
            *competition_key,
            *variant.template.sentence.stable_key(),
            variant.pattern_id,
        ),
        planning,
    )
    forming_source = SourceRef(
        _NAMESPACE,
        _positive_fingerprint(
            tuple(
                value
                for key in variant.support_teacher_keys
                for value in _packed(key)
            ),
            domain="grounded.response.act.lexical.source.v1",
        ),
        variant.pattern_id,
        goal.scope.owner,
        goal.scope.versions,
    )
    return GenerationChoiceHypothesis(
        choice_identity,
        "LEXICAL_REALIZATION_CHOICE",
        variant.template.stance,
        choice_condition,
        variant.template.sentence,
        (goal.source, forming_source),
        competition_key,
        goal.scope,
    )


def _selection_key(
        choice: GenerationChoiceHypothesis,
        selection: AnswerContentSelection,
        variant: GroundedResponseActVariant,
        ) -> LosslessIntegerKey:
    """保存 GG-01 候选、G-01 selection 与实际 response template。"""
    return LosslessIntegerKey((
        _NAMESPACE,
        10,
        *_packed(choice.stable_key()),
        *_packed(selection.stable_key()),
        *_packed(variant.template.sentence.stable_key()),
    ))


def _use_key(
        run: QuestionAnswerRun,
        planning: GenerationPlanningRequest,
        selection_key: LosslessIntegerKey,
        generation: TypedGenerationExecution,
        adoptions: tuple[SurfaceAdoption, ...],
        ) -> LosslessIntegerKey:
    """保存同次 query、planning、generation 和逐点 R-01 adoption。"""
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
        *_packed(selection_key.components),
        *_packed(generation.stable_key()),
        len(adoptions),
    ]
    for adoption in adoptions:
        values.extend(_packed(adoption.stable_key()))
    values.extend(_packed(planning.goal.scope.stable_key()))
    return LosslessIntegerKey(tuple(values))


def grounded_response_act_actual_execution(
        variant: GroundedResponseActVariant,
        planning: GenerationPlanningRequest,
        choice: GenerationChoiceHypothesis,
        run: QuestionAnswerRun,
        ) -> tuple[
            AnswerContentSelection,
            TypedGenerationExecution,
            tuple[SurfaceAdoption, ...],
        ]:
    """核验 run 确由同一 planning 和 selected learned variant 产生。"""
    if choice.exact_uses or choice.typed_outcomes:
        raise GroundedResponseActLexicalUseError(
            "response-act base choice 必须尚未登记 Use/outcome")
    if (choice.choice_kind != "LEXICAL_REALIZATION_CHOICE"
            or choice.target_obligation != variant.template.stance
            or choice.selected_object != variant.template.sentence
            or choice.authorized_scope != planning.goal.scope):
        raise GroundedResponseActLexicalUseError(
            "response-act choice 与 template/scope 漂移")
    if run.planning_request != planning:
        raise GroundedResponseActLexicalUseError(
            "response-act lexical Use 替换了冻结 planning")
    selection = run.selection
    if (not isinstance(selection, AnswerContentSelection)
            or selection.request != planning
            or selection.stance != variant.template.stance):
        raise GroundedResponseActLexicalUseError(
            "response-act lexical Use 缺实际 G-01 selection")
    generation = run.generation
    if (not isinstance(generation, TypedGenerationExecution)
            or not generation.complete
            or generation.plan.request != planning
            or generation.surface is None):
        raise GroundedResponseActLexicalUseError(
            "response-act lexical Use 只接受完整同次 generation")
    request = generation.surface.preview.request
    structure = request.structure
    sentences = structure.syntax.sentences
    if (request.structure.selection != selection
            or request.branch != variant.template.branch
            or len(sentences) != 1
            or set(structure.syntax.suppressed_candidate_keys)
            != set(selection.selected_candidate_keys)):
        raise GroundedResponseActLexicalUseError(
            "response-act G-03 request 与 selection/branch/suppression 漂移")
    sentence = sentences[0]
    if (sentence.sentence != variant.template.sentence
            or sentence.structure != variant.template.slot.structure
            or sentence.proposition_keys
            or len(sentence.values) != 1
            or sentence.values[0].slot != variant.template.slot.slot
            or sentence.values[0].filler != variant.template.stance):
        raise GroundedResponseActLexicalUseError(
            "response-act sentence 未由 selected template 产生")
    preview = generation.surface.preview
    if (len(preview.slots) != 1
            or preview.slots[0].representation != variant.representation):
        raise GroundedResponseActLexicalUseError(
            "response-act actual Representation 漂移")
    adoptions = generation.surface.adoptions
    if len(adoptions) != 1:
        raise GroundedResponseActLexicalUseError(
            "response-act lexical Use 缺实际 R-01 adoption")
    return selection, generation, adoptions


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GroundedResponseActLexicalUse:
    """一次 response-act lexical choice 与 actual Use 的不可变账项。"""

    variant: GroundedResponseActVariant
    planning: GenerationPlanningRequest
    choice_before: GenerationChoiceHypothesis
    run: QuestionAnswerRun
    adoptions: tuple[SurfaceAdoption, ...]
    use: GenerationChoiceUseRef
    choice_after: GenerationChoiceHypothesis

    def __post_init__(self) -> None:
        if not isinstance(self.variant, GroundedResponseActVariant):
            raise TypeError("response-act lexical use variant 类型错误")
        if not isinstance(self.planning, GenerationPlanningRequest):
            raise TypeError("response-act lexical use planning 类型错误")
        if not isinstance(self.choice_before, GenerationChoiceHypothesis):
            raise TypeError("response-act lexical use choice 类型错误")
        if not isinstance(self.run, QuestionAnswerRun):
            raise TypeError("response-act lexical use run 类型错误")
        if (not isinstance(self.adoptions, tuple)
                or any(not isinstance(item, SurfaceAdoption)
                       for item in self.adoptions)):
            raise TypeError("response-act lexical use adoptions 类型错误")
        if not isinstance(self.use, GenerationChoiceUseRef):
            raise TypeError("response-act lexical use ref 类型错误")
        if not isinstance(self.choice_after, GenerationChoiceHypothesis):
            raise TypeError("response-act lexical choice_after 类型错误")
        selection, generation, adoptions = grounded_response_act_actual_execution(
            self.variant, self.planning, self.choice_before, self.run)
        if self.adoptions != adoptions:
            raise GroundedResponseActLexicalUseError(
                "response-act lexical use 未保留 actual adoption")
        expected_selection = _selection_key(
            self.choice_before, selection, self.variant)
        expected_use = _use_key(
            self.run,
            self.planning,
            expected_selection,
            generation,
            self.adoptions,
        )
        if (self.use.use_kind != "CORE_USE"
                or self.use.selection_key != expected_selection
                or self.use.use_key != expected_use
                or self.use.scope != self.planning.goal.scope):
            raise GroundedResponseActLexicalUseError(
                "response-act Use 未绑定完整执行与授权 scope")
        if self.choice_after != replace(
                self.choice_before, exact_uses=(self.use,)):
            raise GroundedResponseActLexicalUseError(
                "response-act Use 未精确回填 GG-01 choice")


# object-model: runtime-ledger
class GroundedResponseActLexicalAdoptionLedger:
    """为一个 response-act installation 登记不可重复的 lexical Use。"""

    def __init__(self, installation) -> None:
        required = ("variant", "planning", "lexical_choice")
        if any(not hasattr(installation, name) for name in required):
            raise TypeError("response-act lexical ledger installation 类型错误")
        self.installation = installation
        self._records: list[GroundedResponseActLexicalUse] = []

    @property
    def records(self) -> tuple[GroundedResponseActLexicalUse, ...]:
        """返回本 installation 已登记的全部 immutable Uses。"""
        return tuple(self._records)

    def adopt(self, run: QuestionAnswerRun) -> GroundedResponseActLexicalUse:
        """在完整生成后登记 actual Use，并返回带 exact Use 的新 choice。"""
        installation = self.installation
        selection, generation, adoptions = grounded_response_act_actual_execution(
            installation.variant,
            installation.planning,
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
                selection_key,
                generation,
                adoptions,
            ),
            selection_key,
            installation.planning.goal.scope,
        )
        if any(item.use.use_key == use.use_key for item in self._records):
            raise GroundedResponseActLexicalUseError(
                "同一次 response-act lexical execution 不得重复登记")
        record = GroundedResponseActLexicalUse(
            installation.variant,
            installation.planning,
            installation.lexical_choice,
            run,
            adoptions,
            use,
            replace(installation.lexical_choice, exact_uses=(use,)),
        )
        self._records.append(record)
        return record


__all__ = [
    "GroundedResponseActLexicalAdoptionLedger",
    "GroundedResponseActLexicalUse",
    "GroundedResponseActLexicalUseError",
    "build_grounded_response_act_lexical_choice",
    "grounded_response_act_actual_execution",
    "grounded_response_act_generation_context",
]
