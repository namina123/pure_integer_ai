"""在同一 grounded reference episode 中登记五层 actual choice/Use。"""
from __future__ import annotations

from dataclasses import dataclass, replace

from pure_integer_ai.cognition.shared.generation_plan import GenerationLayerResult
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
from pure_integer_ai.experiments.ph2_grounded_answer_reference_choice_use import (
    GroundedAnswerReferenceAdoptionLedger,
    GroundedAnswerReferenceChoiceUse,
)
from pure_integer_ai.experiments.ph2_grounded_answer_reference_runtime_factory import (
    GroundedAnswerReferenceRunLocalInstallation,
)
from pure_integer_ai.experiments.question_answer_runtime import QuestionAnswerRun


_NAMESPACE = 21020
_LAYER_KINDS = (
    "CONTENT_CHOICE",
    "PROPOSITION_STRUCTURE_CHOICE",
    "LEXICAL_REALIZATION_CHOICE",
    "COMMUNICATIVE_TASK_CHOICE",
)


# object-model: exception
class GroundedAnswerReferenceEpisodeUseError(ValueError):
    """同次 reference episode 的四层选择或五层 Use 不完整。"""


def _packed(key: tuple[int, ...]) -> tuple[int, ...]:
    """给开放稳定键增加长度边界。"""
    return len(key), *key


def _condition(
        installation: GroundedAnswerReferenceRunLocalInstallation,
        choice_kind: str,
        ) -> GenerationChoiceCondition:
    """建立与 reference choice 共享、且不编码本层 action 的条件。"""
    if choice_kind not in _LAYER_KINDS:
        raise GroundedAnswerReferenceEpisodeUseError(
            "reference episode layer kind 非法")
    selection = installation.reference_selection
    candidates = installation.compilation.planning.candidates
    condition = concept_identity(
        integer_tuple_fingerprint(
            (*selection.context.stable_key(), CHOICE_KINDS.index(choice_kind)),
            domain="grounded.answer.reference.episode.condition.v1",
        ),
        owner=selection.scope.owner,
        versions=selection.scope.versions,
    )
    required = tuple(sorted((
        installation.parser_catalog.branch,
        *(item.proposition.template for item in candidates),
    ), key=ObjectIdentity.stable_key))
    return GenerationChoiceCondition(
        condition,
        selection.context,
        required,
        (),
        selection.scope,
    )


def _content_object(
        installation: GroundedAnswerReferenceRunLocalInstallation,
        ) -> ObjectIdentity:
    """把实际双 candidate 内容集表示为一等、顺序稳定的概念对象。"""
    selection = installation.reference_selection
    values = [len(installation.compilation.planning.candidates)]
    for candidate in installation.compilation.planning.candidates:
        values.extend(_packed(candidate.stable_key()))
    return concept_identity(
        integer_tuple_fingerprint(
            tuple(values),
            domain="grounded.answer.reference.content.object.v1",
        ),
        owner=selection.scope.owner,
        versions=selection.scope.versions,
    )


def _selected_object(
        installation: GroundedAnswerReferenceRunLocalInstallation,
        run: QuestionAnswerRun,
        choice_kind: str,
        ) -> ObjectIdentity:
    """返回由先行 selection 或实际 G-01 形成的本层动作对象。"""
    selection = run.selection
    if selection is None:
        raise GroundedAnswerReferenceEpisodeUseError(
            "reference episode layer 缺 G-01 selection")
    reference = installation.reference_selection
    if choice_kind == "CONTENT_CHOICE":
        return _content_object(installation)
    if choice_kind == "PROPOSITION_STRUCTURE_CHOICE":
        return reference.structure_selection.selected
    if choice_kind == "LEXICAL_REALIZATION_CHOICE":
        return reference.lexical_selection.selected
    if choice_kind == "COMMUNICATIVE_TASK_CHOICE":
        return selection.stance
    raise GroundedAnswerReferenceEpisodeUseError(
        "reference episode selected object kind 非法")


def _competition_key(
        installation: GroundedAnswerReferenceRunLocalInstallation,
        run: QuestionAnswerRun,
        choice_kind: str,
        ) -> tuple[int, ...]:
    """保存本层完整分母，不把 selected object 编入竞争键。"""
    reference = installation.reference_selection
    planning = installation.compilation.planning
    values = [
        CHOICE_KINDS.index(choice_kind),
        *_packed(reference.context.stable_key()),
        *_packed(planning.goal.stable_key()),
    ]
    if choice_kind == "CONTENT_CHOICE":
        values.append(len(planning.candidates))
        for candidate in planning.candidates:
            values.extend(_packed(candidate.stable_key()))
    elif choice_kind == "PROPOSITION_STRUCTURE_CHOICE":
        layer_selection = reference.structure_selection
        values.append(len(layer_selection.options))
        for option in layer_selection.options:
            values.extend(_packed(option.stable_key()))
    elif choice_kind == "LEXICAL_REALIZATION_CHOICE":
        layer_selection = reference.lexical_selection
        values.append(len(layer_selection.options))
        for option in layer_selection.options:
            values.extend(_packed(option.stable_key()))
    else:
        if run.selection is None:
            raise GroundedAnswerReferenceEpisodeUseError(
                "reference task competition 缺 G-01 selection")
        values.extend(_packed(run.selection.protocol.stable_key()))
    return integer_tuple_fingerprint(
        tuple(values),
        domain="grounded.answer.reference.episode.competition.v1",
    )


def _base_choice(
        installation: GroundedAnswerReferenceRunLocalInstallation,
        run: QuestionAnswerRun,
        choice_kind: str,
        ) -> GenerationChoiceHypothesis:
    """从先行 selection/G-01 形成四个独立 GG-01 choice。"""
    reference = installation.reference_selection
    selected = _selected_object(installation, run, choice_kind)
    competition_key = _competition_key(
        installation, run, choice_kind)
    identity = ObjectIdentity(
        OBJECT_HYPOTHESIS,
        integer_tuple_fingerprint(
            (*competition_key, *selected.stable_key()),
            domain="grounded.answer.reference.episode.choice.identity.v1",
        ),
        reference.scope.owner,
        reference.scope.versions,
    )
    target = (
        installation.compilation.planning.goal.goal_kind
        if choice_kind == "COMMUNICATIVE_TASK_CHOICE"
        else installation.compilation.planning.goal.proposition.template
    )
    return GenerationChoiceHypothesis(
        identity,
        choice_kind,
        target,
        _condition(installation, choice_kind),
        selected,
        (reference.source,),
        competition_key,
        reference.scope,
    )


def _layer(
        installation: GroundedAnswerReferenceRunLocalInstallation,
        run: QuestionAnswerRun,
        choice_kind: str,
        ) -> GenerationLayerResult:
    """恢复并核对本层实际 G-00 execution payload。"""
    if (run.selection is None or run.generation is None
            or run.generation.surface is None):
        raise GroundedAnswerReferenceEpisodeUseError(
            "reference episode layer 缺完整 generation")
    protocol = run.generation.plan.protocol
    layer_identity = {
        "CONTENT_CHOICE": protocol.content_layer,
        "PROPOSITION_STRUCTURE_CHOICE": protocol.syntax_layer,
        "LEXICAL_REALIZATION_CHOICE": protocol.surface_layer,
        "COMMUNICATIVE_TASK_CHOICE": protocol.stance_layer,
    }[choice_kind]
    matches = tuple(
        item for item in run.generation.plan.layers
        if item.layer == layer_identity)
    if len(matches) != 1:
        raise GroundedAnswerReferenceEpisodeUseError(
            "reference episode 未恢复唯一 generation layer")
    layer = matches[0]
    candidates = installation.compilation.planning.candidates
    expected_keys = tuple(item.stable_key() for item in candidates)
    preview = run.generation.surface.preview
    if not layer.executed or layer.selected_candidate_keys != expected_keys:
        raise GroundedAnswerReferenceEpisodeUseError(
            "reference episode layer 未执行完整 candidate 集")
    if choice_kind in ("CONTENT_CHOICE", "COMMUNICATIVE_TASK_CHOICE"):
        expected_payload = run.selection.stable_key()
    elif choice_kind == "PROPOSITION_STRUCTURE_CHOICE":
        expected_payload = preview.request.structure.syntax.stable_key()
    else:
        expected_payload = preview.stable_key()
    if layer.payload != expected_payload:
        raise GroundedAnswerReferenceEpisodeUseError(
            "reference episode layer payload 与实际执行漂移")
    return layer


def _non_reference_adoptions(
        installation: GroundedAnswerReferenceRunLocalInstallation,
        run: QuestionAnswerRun,
        ) -> tuple[tuple[int, ...], ...]:
    """返回 lexical choice 独占的非 reference-slot direct adoption 键。"""
    if run.generation is None or run.generation.surface is None:
        raise GroundedAnswerReferenceEpisodeUseError(
            "reference lexical Use 缺 surface")
    compilation = installation.compilation
    preview = run.generation.surface.preview
    expected = tuple(
        slot for slot in preview.slots
        if slot.value.slot != compilation.reference_slot
    )
    if not expected or any(
            item.surface is None or item.representation is None
            for item in expected):
        raise GroundedAnswerReferenceEpisodeUseError(
            "reference lexical package 缺 direct surface")
    adoptions = []
    for slot in expected:
        matches = tuple(
            item for item in run.generation.surface.adoptions
            if (item.sentence == slot.directive.sentence
                and item.slot == slot.value.slot
                and item.proposal == slot.surface
                and item.use_key == slot.directive.surface_use_key)
        )
        if len(matches) != 1:
            raise GroundedAnswerReferenceEpisodeUseError(
                "reference lexical package 未逐槽实际采用")
        adoptions.append(matches[0].stable_key())
    return tuple(adoptions)


def _selection_key(
        installation: GroundedAnswerReferenceRunLocalInstallation,
        run: QuestionAnswerRun,
        choice: GenerationChoiceHypothesis,
        layer: GenerationLayerResult,
        ) -> LosslessIntegerKey:
    """保存本层先行/实际 selection artifact 和 generation layer。"""
    reference = installation.reference_selection
    if choice.choice_kind == "CONTENT_CHOICE":
        artifact = run.selection.stable_key()
    elif choice.choice_kind == "PROPOSITION_STRUCTURE_CHOICE":
        artifact = reference.structure_selection.stable_key()
    elif choice.choice_kind == "LEXICAL_REALIZATION_CHOICE":
        artifact = reference.lexical_selection.stable_key()
    else:
        artifact = run.selection.stable_key()
    return LosslessIntegerKey((
        _NAMESPACE,
        10,
        CHOICE_KINDS.index(choice.choice_kind),
        *_packed(choice.stable_key()),
        *_packed(artifact),
        *_packed(layer.stable_key()),
    ))


def _use_key(
        installation: GroundedAnswerReferenceRunLocalInstallation,
        run: QuestionAnswerRun,
        choice: GenerationChoiceHypothesis,
        layer: GenerationLayerResult,
        selection_key: LosslessIntegerKey,
        ) -> LosslessIntegerKey:
    """无摘要保存本层 query、generation、execution 与授权 scope。"""
    if run.generation is None:
        raise GroundedAnswerReferenceEpisodeUseError(
            "reference episode use key 缺 generation")
    query_key = () if run.query is None else run.query.stable_key()
    result_key = () if run.query_result is None else run.query_result.stable_key()
    postcheck_key = () if run.postcheck is None else run.postcheck.stable_key()
    values = [
        _NAMESPACE,
        20,
        CHOICE_KINDS.index(choice.choice_kind),
        *_packed(run.request.stable_key()),
        *_packed(query_key),
        *_packed(result_key),
        *_packed(installation.compilation.planning.stable_key()),
        *_packed(selection_key.components),
        *_packed(run.generation.stable_key()),
        *_packed(layer.stable_key()),
    ]
    if choice.choice_kind == "LEXICAL_REALIZATION_CHOICE":
        adoptions = _non_reference_adoptions(installation, run)
        values.append(len(adoptions))
        for adoption in adoptions:
            values.extend(_packed(adoption))
    values.extend((
        *_packed(postcheck_key),
        *_packed(installation.reference_selection.scope.stable_key()),
    ))
    return LosslessIntegerKey(tuple(values))


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GroundedAnswerReferenceLayerUse:
    """reference episode 中一个非 reference 层的 actual choice/Use。"""

    installation: GroundedAnswerReferenceRunLocalInstallation
    run: QuestionAnswerRun
    choice_before: GenerationChoiceHypothesis
    layer: GenerationLayerResult
    use: GenerationChoiceUseRef
    choice_after: GenerationChoiceHypothesis

    def __post_init__(self) -> None:
        if not isinstance(
                self.installation, GroundedAnswerReferenceRunLocalInstallation):
            raise TypeError("reference episode use installation 类型错误")
        if not isinstance(self.run, QuestionAnswerRun):
            raise TypeError("reference episode use run 类型错误")
        if (not isinstance(self.choice_before, GenerationChoiceHypothesis)
                or self.choice_before.choice_kind not in _LAYER_KINDS):
            raise TypeError("reference episode choice_before 类型错误")
        if not isinstance(self.layer, GenerationLayerResult):
            raise TypeError("reference episode use layer 类型错误")
        if not isinstance(self.use, GenerationChoiceUseRef):
            raise TypeError("reference episode use ref 类型错误")
        if not isinstance(self.choice_after, GenerationChoiceHypothesis):
            raise TypeError("reference episode choice_after 类型错误")
        kind = self.choice_before.choice_kind
        expected_choice = _base_choice(self.installation, self.run, kind)
        expected_layer = _layer(self.installation, self.run, kind)
        selection_key = _selection_key(
            self.installation,
            self.run,
            expected_choice,
            expected_layer,
        )
        use_key = _use_key(
            self.installation,
            self.run,
            expected_choice,
            expected_layer,
            selection_key,
        )
        if (self.choice_before != expected_choice
                or self.layer != expected_layer
                or self.use.use_kind != "CORE_USE"
                or self.use.selection_key != selection_key
                or self.use.use_key != use_key
                or self.use.scope
                != self.installation.reference_selection.scope
                or self.choice_after != replace(
                    expected_choice, exact_uses=(self.use,))):
            raise GroundedAnswerReferenceEpisodeUseError(
                "reference episode layer Use 未精确回填 choice")


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GroundedAnswerReferenceFiveChoiceUses:
    """同次双命题 generation 的五层有序 choice/Use。"""

    content: GroundedAnswerReferenceLayerUse
    structure: GroundedAnswerReferenceLayerUse
    reference: GroundedAnswerReferenceChoiceUse
    lexical: GroundedAnswerReferenceLayerUse
    task: GroundedAnswerReferenceLayerUse

    def __post_init__(self) -> None:
        for label, value in (
                ("content", self.content),
                ("structure", self.structure),
                ("lexical", self.lexical),
                ("task", self.task)):
            if not isinstance(value, GroundedAnswerReferenceLayerUse):
                raise TypeError(f"reference episode {label} use 类型错误")
        if not isinstance(self.reference, GroundedAnswerReferenceChoiceUse):
            raise TypeError("reference episode reference use 类型错误")
        records = (
            self.content,
            self.structure,
            self.reference,
            self.lexical,
            self.task,
        )
        kinds = tuple(item.choice_before.choice_kind for item in records)
        if kinds != CHOICE_KINDS:
            raise GroundedAnswerReferenceEpisodeUseError(
                "reference episode 五层 choice 顺序或覆盖漂移")
        installations = {id(item.installation) for item in records}
        runs = {id(item.run) for item in records}
        contexts = {
            item.choice_before.condition.context for item in records}
        use_keys = tuple(item.use.use_key for item in records)
        if (len(installations) != 1 or len(runs) != 1
                or len(contexts) != 1 or len(set(use_keys)) != 5):
            raise GroundedAnswerReferenceEpisodeUseError(
                "reference episode 五层未共享同次执行或 Use 发生复用")

    @property
    def uses(self) -> tuple[GenerationChoiceUseRef, ...]:
        """按 GG-01 冻结层序返回五个互异 exact Use。"""
        return tuple(item.use for item in (
            self.content,
            self.structure,
            self.reference,
            self.lexical,
            self.task,
        ))


# object-model: runtime-ledger
class GroundedAnswerReferenceEpisodeAdoptionLedger:
    """原子登记同一双命题 run 的五层 actual choice/Use。"""

    def __init__(
            self,
            installation: GroundedAnswerReferenceRunLocalInstallation,
            ) -> None:
        if not isinstance(
                installation, GroundedAnswerReferenceRunLocalInstallation):
            raise TypeError("reference episode ledger installation 类型错误")
        self.installation = installation
        self._records: list[GroundedAnswerReferenceFiveChoiceUses] = []

    @property
    def records(self) -> tuple[GroundedAnswerReferenceFiveChoiceUses, ...]:
        """返回已经原子提交的五层 Use bundle。"""
        return tuple(self._records)

    def _record_layer(
            self,
            run: QuestionAnswerRun,
            choice_kind: str,
            ) -> GroundedAnswerReferenceLayerUse:
        """为一个非 reference 层构造 choice 与 exact Use。"""
        choice = _base_choice(self.installation, run, choice_kind)
        layer = _layer(self.installation, run, choice_kind)
        selection_key = _selection_key(
            self.installation, run, choice, layer)
        use = GenerationChoiceUseRef(
            "CORE_USE",
            _use_key(
                self.installation, run, choice, layer, selection_key),
            selection_key,
            self.installation.reference_selection.scope,
        )
        return GroundedAnswerReferenceLayerUse(
            self.installation,
            run,
            choice,
            layer,
            use,
            replace(choice, exact_uses=(use,)),
        )

    def adopt(
            self,
            run: QuestionAnswerRun,
            ) -> GroundedAnswerReferenceFiveChoiceUses:
        """登记五层互异 Use；任一层不完整时整次拒绝。"""
        if not isinstance(run, QuestionAnswerRun):
            raise TypeError("reference episode ledger run 类型错误")
        content = self._record_layer(run, "CONTENT_CHOICE")
        structure = self._record_layer(
            run, "PROPOSITION_STRUCTURE_CHOICE")
        lexical = self._record_layer(run, "LEXICAL_REALIZATION_CHOICE")
        task = self._record_layer(run, "COMMUNICATIVE_TASK_CHOICE")
        occupied = tuple(
            item.use.use_key for item in (
                content, structure, lexical, task))
        reference = GroundedAnswerReferenceAdoptionLedger(
            self.installation,
            occupied,
        ).adopt(run)
        record = GroundedAnswerReferenceFiveChoiceUses(
            content,
            structure,
            reference,
            lexical,
            task,
        )
        keys = set(item.use_key for item in record.uses)
        if any(
                keys & set(item.use_key for item in previous.uses)
                for previous in self._records):
            raise GroundedAnswerReferenceEpisodeUseError(
                "同一次 reference episode 不得重复登记")
        self._records.append(record)
        return record


__all__ = [
    "GroundedAnswerReferenceEpisodeAdoptionLedger",
    "GroundedAnswerReferenceEpisodeUseError",
    "GroundedAnswerReferenceFiveChoiceUses",
    "GroundedAnswerReferenceLayerUse",
]
