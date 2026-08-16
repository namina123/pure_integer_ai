"""把 grounded reference choice 绑定到实际 G-02/G-03 与 R-01 Use。"""
from __future__ import annotations

from dataclasses import dataclass, replace

from pure_integer_ai.cognition.shared.generation_surface import SurfaceAdoption
from pure_integer_ai.cognition.shared.generation_verification import (
    GenerationSurfaceParseRequest,
)
from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    GenerationChoiceHypothesis,
    GenerationChoiceUseRef,
    LosslessIntegerKey,
)
from pure_integer_ai.experiments.ph2_grounded_answer_reference_parser import (
    GroundedAnswerRecoveredReference,
)
from pure_integer_ai.experiments.ph2_grounded_answer_reference_runtime_factory import (
    GroundedAnswerReferenceRunLocalInstallation,
)
from pure_integer_ai.experiments.question_answer_runtime import QuestionAnswerRun


_NAMESPACE = 21010


# object-model: exception
class GroundedAnswerReferenceUseError(ValueError):
    """reference Use 未绑定同次选择、syntax、proposal、adoption 或 generation。"""


def _packed(key: tuple[int, ...]) -> tuple[int, ...]:
    """给开放稳定键增加长度边界。"""
    return len(key), *key


def _actual_execution(
        installation: GroundedAnswerReferenceRunLocalInstallation,
        run: QuestionAnswerRun,
        ) -> tuple[
            GroundedAnswerRecoveredReference,
            SurfaceAdoption | None,
            SurfaceAdoption,
        ]:
    """恢复 selected strategy 在真实 G-02/G-03/R-01 中的逐点消费。"""
    if not isinstance(
            installation, GroundedAnswerReferenceRunLocalInstallation):
        raise TypeError("reference actual installation 类型错误")
    if not isinstance(run, QuestionAnswerRun):
        raise TypeError("reference actual run 类型错误")
    compilation = installation.compilation
    reference_selection = installation.reference_selection
    if (not run.complete
            or run.planning_request != compilation.planning
            or run.selection is None
            or run.generation is None
            or run.generation.surface is None
            or run.generation.plan.request != compilation.planning):
        raise GroundedAnswerReferenceUseError(
            "reference actual Use 缺同次完整 selection/generation")
    generation = run.generation
    surface = generation.surface
    preview = surface.preview
    expected_structure = compilation.connector.structure_planner().plan(
        run.selection)
    if (preview.request.structure != expected_structure
            or preview.request.execution.request.syntax
            != expected_structure.syntax
            or not preview.request.execution.complete):
        raise GroundedAnswerReferenceUseError(
            "reference actual Use 的 G-02/S-07 结构漂移")
    parse_request = GenerationSurfaceParseRequest.from_execution(generation)
    recovery = installation.parser.recover_reference(
        parse_request, preview)
    if (recovery.strategy_object
            != reference_selection.selected.declarative_object
            or recovery.source != reference_selection.source
            or recovery.scope != reference_selection.scope):
        raise GroundedAnswerReferenceUseError(
            "reference parser recovery 与先行 selection 漂移")
    slots = tuple(
        item for item in preview.slots
        if (item.directive.sentence == recovery.sentence
            and item.value.slot == recovery.slot)
    )
    if len(slots) != 1:
        raise GroundedAnswerReferenceUseError(
            "reference actual Use 缺唯一 reference slot")
    slot = slots[0]
    direct = tuple(
        item for item in surface.adoptions
        if (item.sentence == recovery.sentence
            and item.slot == recovery.slot
            and item.proposal == slot.surface
            and item.use_key == slot.directive.surface_use_key)
    )
    if len(direct) != 1:
        raise GroundedAnswerReferenceUseError(
            "reference strategy 缺自己的 direct surface adoption")
    reference = tuple(
        item for item in surface.adoptions
        if (item.sentence == recovery.sentence
            and item.slot == recovery.slot
            and item.proposal == slot.reference
            and item.use_key == slot.directive.reference_use_key)
    ) if slot.reference is not None else ()
    if reference_selection.selected.strategy == "ANTECEDENT_REFERENCE":
        expected_antecedent = reference_selection.selected_antecedent
        requirements = tuple(
            item for item in expected_structure.syntax.anaphora
            if (item.address == recovery.sentence and item.slot == recovery.slot)
        )
        if (len(requirements) != 1
                or requirements[0].antecedent_candidate_key
                != expected_antecedent.stable_key()
                or recovery.antecedent_candidate_key
                != expected_antecedent.stable_key()
                or recovery.antecedent
                != expected_antecedent.proposition.template
                or len(reference) != 1):
            raise GroundedAnswerReferenceUseError(
                "antecedent strategy 未实际执行唯一 reference adoption")
        reference_adoption = reference[0]
    else:
        if (expected_structure.syntax.anaphora
                or recovery.antecedent is not None
                or slot.reference is not None
                or reference):
            raise GroundedAnswerReferenceUseError(
                "explicit strategy 不得执行 reference adoption")
        reference_adoption = None
    return recovery, reference_adoption, direct[0]


def _selection_key(
        installation: GroundedAnswerReferenceRunLocalInstallation,
        run: QuestionAnswerRun,
        recovery: GroundedAnswerRecoveredReference,
        ) -> LosslessIntegerKey:
    """保存完整竞争、先行选择、actual syntax 与 parser recovery。"""
    if run.generation is None or run.generation.surface is None:
        raise GroundedAnswerReferenceUseError(
            "reference selection key 缺 actual syntax")
    syntax = run.generation.surface.preview.request.structure.syntax
    return LosslessIntegerKey((
        _NAMESPACE,
        10,
        *_packed(installation.reference_selection.choice.stable_key()),
        *_packed(installation.reference_selection.stable_key()),
        *_packed(syntax.stable_key()),
        *_packed(recovery.stable_key()),
        *_packed(installation.compilation.connector.stable_key()),
    ))


def _use_key(
        installation: GroundedAnswerReferenceRunLocalInstallation,
        run: QuestionAnswerRun,
        recovery: GroundedAnswerRecoveredReference,
        reference_adoption: SurfaceAdoption | None,
        direct_adoption: SurfaceAdoption,
        selection_key: LosslessIntegerKey,
        ) -> LosslessIntegerKey:
    """无摘要保存同次 query、generation、proposal、adoption 和 scope。"""
    if run.generation is None or run.generation.surface is None:
        raise GroundedAnswerReferenceUseError(
            "reference use key 缺完整 generation")
    query_key = () if run.query is None else run.query.stable_key()
    result_key = () if run.query_result is None else run.query_result.stable_key()
    postcheck_key = () if run.postcheck is None else run.postcheck.stable_key()
    values = [
        _NAMESPACE,
        20,
        *_packed(run.request.stable_key()),
        *_packed(query_key),
        *_packed(result_key),
        *_packed(installation.compilation.planning.stable_key()),
        *_packed(selection_key.components),
        *_packed(run.generation.stable_key()),
        *_packed(recovery.stable_key()),
        0 if reference_adoption is None else 1,
    ]
    if reference_adoption is not None:
        values.extend(_packed(reference_adoption.stable_key()))
    values.extend((
        *_packed(direct_adoption.stable_key()),
        *_packed(postcheck_key),
        *_packed(installation.reference_selection.scope.stable_key()),
    ))
    return LosslessIntegerKey(tuple(values))


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GroundedAnswerReferenceChoiceUse:
    """一次 reference selection 与实际双句执行的不可变 exact Use。"""

    installation: GroundedAnswerReferenceRunLocalInstallation
    run: QuestionAnswerRun
    choice_before: GenerationChoiceHypothesis
    recovery: GroundedAnswerRecoveredReference
    reference_adoption: SurfaceAdoption | None
    direct_adoption: SurfaceAdoption
    use: GenerationChoiceUseRef
    choice_after: GenerationChoiceHypothesis

    def __post_init__(self) -> None:
        if not isinstance(
                self.installation, GroundedAnswerReferenceRunLocalInstallation):
            raise TypeError("reference use installation 类型错误")
        if not isinstance(self.run, QuestionAnswerRun):
            raise TypeError("reference use run 类型错误")
        if not isinstance(self.choice_before, GenerationChoiceHypothesis):
            raise TypeError("reference use choice_before 类型错误")
        if not isinstance(self.recovery, GroundedAnswerRecoveredReference):
            raise TypeError("reference use recovery 类型错误")
        if (self.reference_adoption is not None
                and not isinstance(self.reference_adoption, SurfaceAdoption)):
            raise TypeError("reference use reference adoption 类型错误")
        if not isinstance(self.direct_adoption, SurfaceAdoption):
            raise TypeError("reference use direct adoption 类型错误")
        if not isinstance(self.use, GenerationChoiceUseRef):
            raise TypeError("reference use ref 类型错误")
        if not isinstance(self.choice_after, GenerationChoiceHypothesis):
            raise TypeError("reference use choice_after 类型错误")
        actual = _actual_execution(self.installation, self.run)
        if actual != (
                self.recovery,
                self.reference_adoption,
                self.direct_adoption):
            raise GroundedAnswerReferenceUseError(
                "reference use 未保存实际 parser/adoption")
        expected_choice = self.installation.reference_selection.choice
        selection_key = _selection_key(
            self.installation, self.run, self.recovery)
        use_key = _use_key(
            self.installation,
            self.run,
            self.recovery,
            self.reference_adoption,
            self.direct_adoption,
            selection_key,
        )
        if (self.choice_before != expected_choice
                or self.use.use_kind != "CORE_USE"
                or self.use.selection_key != selection_key
                or self.use.use_key != use_key
                or self.use.scope
                != self.installation.reference_selection.scope
                or self.choice_after != replace(
                    expected_choice, exact_uses=(self.use,))):
            raise GroundedAnswerReferenceUseError(
                "reference use 未精确回填 GG-01 choice")


# object-model: runtime-ledger
class GroundedAnswerReferenceAdoptionLedger:
    """为一个 run-local installation 登记不可重复的 reference exact Use。"""

    def __init__(
            self,
            installation: GroundedAnswerReferenceRunLocalInstallation,
            occupied_use_keys: tuple[LosslessIntegerKey, ...] = (),
            ) -> None:
        if not isinstance(
                installation, GroundedAnswerReferenceRunLocalInstallation):
            raise TypeError("reference ledger installation 类型错误")
        if (not isinstance(occupied_use_keys, tuple)
                or any(not isinstance(item, LosslessIntegerKey)
                       for item in occupied_use_keys)
                or len(set(occupied_use_keys)) != len(occupied_use_keys)):
            raise TypeError("reference ledger occupied use keys 类型错误")
        self.installation = installation
        self.occupied_use_keys = occupied_use_keys
        self._records: list[GroundedAnswerReferenceChoiceUse] = []

    @property
    def records(self) -> tuple[GroundedAnswerReferenceChoiceUse, ...]:
        """返回已提交的全部 reference exact Use。"""
        return tuple(self._records)

    def adopt(self, run: QuestionAnswerRun) -> GroundedAnswerReferenceChoiceUse:
        """仅在 reference/direct execution 与 parser recovery 齐全后登记。"""
        recovery, reference_adoption, direct_adoption = _actual_execution(
            self.installation, run)
        choice = self.installation.reference_selection.choice
        selection_key = _selection_key(self.installation, run, recovery)
        use = GenerationChoiceUseRef(
            "CORE_USE",
            _use_key(
                self.installation,
                run,
                recovery,
                reference_adoption,
                direct_adoption,
                selection_key,
            ),
            selection_key,
            self.installation.reference_selection.scope,
        )
        if (use.use_key in self.occupied_use_keys
                or any(item.use.use_key == use.use_key
                       for item in self._records)):
            raise GroundedAnswerReferenceUseError(
                "reference exact Use 与既有层复用或重复登记")
        record = GroundedAnswerReferenceChoiceUse(
            self.installation,
            run,
            choice,
            recovery,
            reference_adoption,
            direct_adoption,
            use,
            replace(choice, exact_uses=(use,)),
        )
        self._records.append(record)
        return record


__all__ = [
    "GroundedAnswerReferenceAdoptionLedger",
    "GroundedAnswerReferenceChoiceUse",
    "GroundedAnswerReferenceUseError",
]
