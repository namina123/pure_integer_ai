"""形成 grounded answer 的有限篇章引用竞争与显式选择。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.generation_plan import (
    GenerationCandidate,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_CONTEXT_SCOPE,
    OBJECT_HYPOTHESIS,
    OBJECT_STRUCTURE_CONCEPT,
    ObjectIdentity,
    SourceRef,
    concept_identity,
)
from pure_integer_ai.cognition.shared.scope_identity import ScopeIdentity
from pure_integer_ai.crosscut.determinism.fingerprint import (
    integer_tuple_fingerprint,
)
from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    GenerationChoiceCondition,
    GenerationChoiceHypothesis,
)
from pure_integer_ai.experiments.ph2_grounded_answer_course import (
    REFERENCE_STRATEGIES,
)
from pure_integer_ai.experiments.ph2_grounded_answer_learning import (
    PATTERN_CLAIM,
)
from pure_integer_ai.experiments.ph2_grounded_answer_reference_compile import (
    GroundedAnswerReferenceCompilation,
)


_NAMESPACE = 20990
_CHOICE_KIND = "DISCOURSE_REFERENCE_CHOICE"
_LAYER_SELECTION_KINDS = (
    "PROPOSITION_STRUCTURE_CHOICE",
    "LEXICAL_REALIZATION_CHOICE",
)


# object-model: exception
class GroundedAnswerReferenceChoiceError(ValueError):
    """reference 竞争集、选择或其同次边界不完整。"""


def _packed(key: tuple[int, ...]) -> tuple[int, ...]:
    """给开放稳定键增加长度边界。"""
    return len(key), *key


def _strategy_object(
        strategy: str,
        scope: ScopeIdentity,
        ) -> ObjectIdentity:
    """把冻结策略映射为可被 GG-01 选择的一等概念对象。"""
    try:
        ordinal = REFERENCE_STRATEGIES.index(strategy) + 1
    except ValueError as exc:
        raise GroundedAnswerReferenceChoiceError(
            "reference strategy 未注册") from exc
    return concept_identity(
        (_NAMESPACE, 10, ordinal),
        owner=scope.owner,
        versions=scope.versions,
    )


def _common_context(
        compilation: GroundedAnswerReferenceCompilation,
        ) -> ObjectIdentity:
    """建立不含 selected strategy 的双命题 generation context。"""
    planning = compilation.planning
    values = [
        *_packed(planning.goal.stable_key()),
        len(planning.candidates),
    ]
    for candidate in planning.candidates:
        values.extend(_packed(candidate.stable_key()))
    branch = compilation.connector.registry.templates[0].language_branch
    values.extend(_packed(branch.stable_key()))
    return ObjectIdentity(
        OBJECT_CONTEXT_SCOPE,
        integer_tuple_fingerprint(
            tuple(values),
            domain="grounded.answer.reference.context.v1",
        ),
        planning.goal.scope.owner,
        planning.goal.scope.versions,
    )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GroundedAnswerReferenceStrategy:
    """一个可执行 reference strategy 及其独立 compilation。"""

    strategy: str
    declarative_object: ObjectIdentity
    compilation: GroundedAnswerReferenceCompilation

    def __post_init__(self) -> None:
        if self.strategy not in REFERENCE_STRATEGIES:
            raise GroundedAnswerReferenceChoiceError(
                "reference strategy option 未注册")
        if not isinstance(self.declarative_object, ObjectIdentity):
            raise TypeError("reference strategy object 类型错误")
        if not isinstance(
                self.compilation, GroundedAnswerReferenceCompilation):
            raise TypeError("reference strategy compilation 类型错误")
        scope = self.compilation.planning.goal.scope
        if (self.compilation.strategy != self.strategy
                or self.declarative_object
                != _strategy_object(self.strategy, scope)):
            raise GroundedAnswerReferenceChoiceError(
                "reference strategy object/compilation 漂移")

    def stable_key(self) -> tuple[int, ...]:
        """返回策略对象和完整可执行 connector 配置。"""
        return (
            REFERENCE_STRATEGIES.index(self.strategy) + 1,
            *_packed(self.declarative_object.stable_key()),
            *_packed(self.compilation.connector.stable_key()),
        )


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GroundedAnswerReferenceLayerSelection:
    """在 syntax mapper 前形成的单层有限 option 与显式选择。"""

    choice_kind: str
    options: tuple[ObjectIdentity, ...]
    selected: ObjectIdentity
    trace: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.choice_kind not in _LAYER_SELECTION_KINDS:
            raise GroundedAnswerReferenceChoiceError(
                "reference layer selection kind 非法")
        if (not isinstance(self.options, tuple) or not self.options
                or any(not isinstance(item, ObjectIdentity)
                       for item in self.options)
                or len(set(self.options)) != len(self.options)):
            raise GroundedAnswerReferenceChoiceError(
                "reference layer selection options 非法")
        if self.selected not in self.options:
            raise GroundedAnswerReferenceChoiceError(
                "reference layer selected object 不属于 options")
        if (not isinstance(self.trace, tuple) or not self.trace
                or any(type(item) is not int for item in self.trace)):
            raise GroundedAnswerReferenceChoiceError(
                "reference layer selection trace 非法")

    def stable_key(self) -> tuple[int, ...]:
        """返回完整 option 分母、selected object 与先行 trace。"""
        values = [
            _LAYER_SELECTION_KINDS.index(self.choice_kind) + 1,
            len(self.options),
        ]
        for option in self.options:
            values.extend(_packed(option.stable_key()))
        values.extend((
            *_packed(self.selected.stable_key()),
            *_packed(self.trace),
        ))
        return tuple(values)


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class GroundedAnswerReferenceSelection:
    """在 syntax mapper 前形成的完整 reference competition/selection。"""

    options: tuple[GroundedAnswerReferenceStrategy, ...]
    selected: GroundedAnswerReferenceStrategy
    antecedent_candidates: tuple[GenerationCandidate, ...]
    selected_antecedent: GenerationCandidate
    referring_candidate: GenerationCandidate
    structure_selection: GroundedAnswerReferenceLayerSelection
    lexical_selection: GroundedAnswerReferenceLayerSelection
    source: SourceRef
    scope: ScopeIdentity
    context: ObjectIdentity
    trace: tuple[int, ...]
    choice: GenerationChoiceHypothesis

    def __post_init__(self) -> None:
        if (not isinstance(self.options, tuple)
                or any(not isinstance(item, GroundedAnswerReferenceStrategy)
                       for item in self.options)
                or tuple(item.strategy for item in self.options)
                != REFERENCE_STRATEGIES):
            raise GroundedAnswerReferenceChoiceError(
                "reference options 必须精确覆盖冻结策略顺序")
        if self.selected not in self.options:
            raise GroundedAnswerReferenceChoiceError(
                "selected reference strategy 不属于竞争集")
        if (not isinstance(self.antecedent_candidates, tuple)
                or not self.antecedent_candidates
                or any(not isinstance(item, GenerationCandidate)
                       for item in self.antecedent_candidates)
                or len(set(self.antecedent_candidates))
                != len(self.antecedent_candidates)):
            raise GroundedAnswerReferenceChoiceError(
                "reference antecedent candidates 非法")
        if self.selected_antecedent not in self.antecedent_candidates:
            raise GroundedAnswerReferenceChoiceError(
                "selected antecedent 不属于有限候选集")
        if not isinstance(self.referring_candidate, GenerationCandidate):
            raise TypeError("reference referring candidate 类型错误")
        if (not isinstance(
                self.structure_selection,
                GroundedAnswerReferenceLayerSelection)
                or self.structure_selection.choice_kind
                != "PROPOSITION_STRUCTURE_CHOICE"):
            raise TypeError("reference structure selection 类型错误")
        if (not isinstance(
                self.lexical_selection,
                GroundedAnswerReferenceLayerSelection)
                or self.lexical_selection.choice_kind
                != "LEXICAL_REALIZATION_CHOICE"):
            raise TypeError("reference lexical selection 类型错误")
        if not isinstance(self.source, SourceRef):
            raise TypeError("reference selection source 类型错误")
        if not isinstance(self.scope, ScopeIdentity):
            raise TypeError("reference selection scope 类型错误")
        if not isinstance(self.context, ObjectIdentity):
            raise TypeError("reference selection context 类型错误")
        if (not isinstance(self.trace, tuple) or not self.trace
                or any(type(item) is not int for item in self.trace)):
            raise GroundedAnswerReferenceChoiceError(
                "reference selection trace 非法")
        if not isinstance(self.choice, GenerationChoiceHypothesis):
            raise TypeError("reference selection choice 类型错误")

        base = self.options[0].compilation
        expected_candidates = tuple(
            item.candidate for item in base.claims)
        expected_antecedents = expected_candidates[:-1]
        for option in self.options:
            current = option.compilation
            if (current.episode_id != base.episode_id
                    or current.planning != base.planning
                    or current.claims != base.claims
                    or current.forming_teacher_keys
                    != base.forming_teacher_keys):
                raise GroundedAnswerReferenceChoiceError(
                    "reference strategies 未共享同一课程/planning/candidates")
        if (self.antecedent_candidates != expected_antecedents
                or self.selected_antecedent != expected_antecedents[0]
                or self.referring_candidate != expected_candidates[-1]
                or self.structure_selection != _structure_selection(
                    self.options, self.scope)
                or self.lexical_selection != _lexical_selection(
                    self.options, self.scope)
                or self.source != expected_candidates[0].source
                or self.scope != expected_candidates[0].scope
                or any(item.source != self.source or item.scope != self.scope
                       for item in expected_candidates)
                or self.context != _common_context(base)):
            raise GroundedAnswerReferenceChoiceError(
                "reference selection 候选、来源、scope 或 context 漂移")
        expected_objects = tuple(
            option.declarative_object for option in self.options)
        expected_required = tuple(sorted((
            base.connector.registry.templates[0].language_branch,
            *(item.proposition.template for item in expected_candidates),
        ), key=ObjectIdentity.stable_key))
        if (self.choice.choice_kind != _CHOICE_KIND
                or self.choice.target_obligation
                != self.referring_candidate.proposition.template
                or self.choice.selected_object
                != self.selected.declarative_object
                or self.choice.forming_sources != (self.source,)
                or self.choice.authorized_scope != self.scope
                or self.choice.condition.context != self.context
                or self.choice.condition.required_context_objects
                != expected_required
                or self.choice.condition.forbidden_context_objects
                or self.choice.exact_uses
                or self.choice.typed_outcomes):
            raise GroundedAnswerReferenceChoiceError(
                "reference GG-01 choice 未绑定 selection")
        expected_competition = _competition_key(
            self.context,
            self.antecedent_candidates,
            expected_objects,
        )
        if self.choice.competition_key != expected_competition:
            raise GroundedAnswerReferenceChoiceError(
                "reference competition key 含 selected action 或发生漂移")

    @property
    def compilation(self) -> GroundedAnswerReferenceCompilation:
        """返回显式选择对应的唯一可执行 compilation。"""
        return self.selected.compilation

    def stable_key(self) -> tuple[int, ...]:
        """返回完整候选集、selected antecedent/strategy 与选择 trace。"""
        values = [len(self.options)]
        for option in self.options:
            values.extend(_packed(option.stable_key()))
        values.extend((
            *_packed(self.selected.stable_key()),
            len(self.antecedent_candidates),
        ))
        for candidate in self.antecedent_candidates:
            values.extend(_packed(candidate.stable_key()))
        values.extend((
            *_packed(self.selected_antecedent.stable_key()),
            *_packed(self.referring_candidate.stable_key()),
            *_packed(self.structure_selection.stable_key()),
            *_packed(self.lexical_selection.stable_key()),
            *_packed(self.source.stable_key()),
            *_packed(self.scope.stable_key()),
            *_packed(self.context.stable_key()),
            *_packed(self.trace),
            *_packed(self.choice.stable_key()),
        ))
        return tuple(values)


def _competition_key(
        context: ObjectIdentity,
        antecedent_candidates: tuple[GenerationCandidate, ...],
        strategy_objects: tuple[ObjectIdentity, ...],
        ) -> tuple[int, ...]:
    """保存完整候选与策略分母，不编码 selected strategy/antecedent。"""
    values = [
        *_packed(context.stable_key()),
        len(antecedent_candidates),
    ]
    for candidate in antecedent_candidates:
        values.extend(_packed(candidate.stable_key()))
    values.append(len(strategy_objects))
    for strategy in strategy_objects:
        values.extend(_packed(strategy.stable_key()))
    return integer_tuple_fingerprint(
        tuple(values),
        domain="grounded.answer.reference.competition.v1",
    )


def _structure_shape(
        option: GroundedAnswerReferenceStrategy,
        ) -> tuple[int, ...]:
    """提取忽略词面和 selected reference action 的双句结构形状。"""
    compilation = option.compilation
    values = [len(compilation.sentences)]
    for sentence in compilation.sentences:
        aliases = tuple(sorted(
            sentence.aliases, key=lambda item: item.part_ordinal))
        values.append(len(aliases))
        for alias in aliases:
            role = 3 if alias.slot == compilation.reference_slot else (
                2 if alias.part_kind == PATTERN_CLAIM else 1)
            values.append(role)
        values.append(len(sentence.orders))
    return tuple(values)


def _structure_selection(
        options: tuple[GroundedAnswerReferenceStrategy, ...],
        scope: ScopeIdentity,
        ) -> GroundedAnswerReferenceLayerSelection:
    """建立不随 reference strategy 改写的双句 structure selection。"""
    shapes = tuple(_structure_shape(item) for item in options)
    if len(set(shapes)) != 1:
        raise GroundedAnswerReferenceChoiceError(
            "reference strategies 未共享同一双句 structure shape")
    selected = ObjectIdentity(
        OBJECT_STRUCTURE_CONCEPT,
        integer_tuple_fingerprint(
            shapes[0], domain="grounded.answer.reference.structure.v1"),
        scope.owner,
        scope.versions,
    )
    return GroundedAnswerReferenceLayerSelection(
        "PROPOSITION_STRUCTURE_CHOICE",
        (selected,),
        selected,
        (_NAMESPACE, 20, 1, len(options)),
    )


def _lexical_selection(
        options: tuple[GroundedAnswerReferenceStrategy, ...],
        scope: ScopeIdentity,
        ) -> GroundedAnswerReferenceLayerSelection:
    """只对非 reference slots 建立共享 direct lexical package selection。"""
    packages = []
    for option in options:
        compilation = option.compilation
        representations = tuple(
            alias.representation
            for sentence in compilation.sentences
            for alias in sorted(
                sentence.aliases, key=lambda item: item.part_ordinal)
            if alias.slot != compilation.reference_slot
        )
        values = [len(representations)]
        for representation in representations:
            values.extend(_packed(representation.stable_key()))
        packages.append(tuple(values))
    if len(set(packages)) != 1:
        raise GroundedAnswerReferenceChoiceError(
            "reference strategies 的非引用 lexical package 漂移")
    selected = concept_identity(
        integer_tuple_fingerprint(
            packages[0], domain="grounded.answer.reference.lexical.v1"),
        owner=scope.owner,
        versions=scope.versions,
    )
    return GroundedAnswerReferenceLayerSelection(
        "LEXICAL_REALIZATION_CHOICE",
        (selected,),
        selected,
        (_NAMESPACE, 20, 2, len(options)),
    )


def build_grounded_answer_reference_selection(
        compilations: tuple[GroundedAnswerReferenceCompilation, ...],
        selected_strategy: str,
        trace: tuple[int, ...],
        ) -> GroundedAnswerReferenceSelection:
    """从同一课程的两份 compilation 形成先于 syntax 的显式选择。"""
    if (not isinstance(compilations, tuple)
            or len(compilations) != len(REFERENCE_STRATEGIES)
            or any(not isinstance(item, GroundedAnswerReferenceCompilation)
                   for item in compilations)):
        raise GroundedAnswerReferenceChoiceError(
            "reference selection 必须接收两份 compilation")
    by_strategy = {item.strategy: item for item in compilations}
    if (len(by_strategy) != len(REFERENCE_STRATEGIES)
            or set(by_strategy) != set(REFERENCE_STRATEGIES)):
        raise GroundedAnswerReferenceChoiceError(
            "reference compilations 未精确覆盖冻结策略")
    if selected_strategy not in REFERENCE_STRATEGIES:
        raise GroundedAnswerReferenceChoiceError(
            "selected reference strategy 未注册")
    base = by_strategy[REFERENCE_STRATEGIES[0]]
    scope = base.planning.goal.scope
    options = tuple(
        GroundedAnswerReferenceStrategy(
            strategy,
            _strategy_object(strategy, scope),
            by_strategy[strategy],
        )
        for strategy in REFERENCE_STRATEGIES
    )
    selected = next(
        item for item in options if item.strategy == selected_strategy)
    candidates = tuple(item.candidate for item in base.claims)
    antecedents = candidates[:-1]
    context = _common_context(base)
    strategy_objects = tuple(item.declarative_object for item in options)
    competition_key = _competition_key(
        context, antecedents, strategy_objects)
    condition = GenerationChoiceCondition(
        concept_identity(
            integer_tuple_fingerprint(
                (*context.stable_key(), len(strategy_objects)),
                domain="grounded.answer.reference.condition.v1",
            ),
            owner=scope.owner,
            versions=scope.versions,
        ),
        context,
        tuple(sorted((
            base.connector.registry.templates[0].language_branch,
            *(item.proposition.template for item in candidates),
        ), key=ObjectIdentity.stable_key)),
        (),
        scope,
    )
    choice = GenerationChoiceHypothesis(
        ObjectIdentity(
            OBJECT_HYPOTHESIS,
            integer_tuple_fingerprint(
                (*competition_key,
                 *selected.declarative_object.stable_key(),
                 *antecedents[0].stable_key()),
                domain="grounded.answer.reference.choice.identity.v1",
            ),
            scope.owner,
            scope.versions,
        ),
        _CHOICE_KIND,
        candidates[-1].proposition.template,
        condition,
        selected.declarative_object,
        (candidates[0].source,),
        competition_key,
        scope,
    )
    return GroundedAnswerReferenceSelection(
        options,
        selected,
        antecedents,
        antecedents[0],
        candidates[-1],
        _structure_selection(options, scope),
        _lexical_selection(options, scope),
        candidates[0].source,
        scope,
        context,
        trace,
        choice,
    )


__all__ = [
    "GroundedAnswerReferenceChoiceError",
    "GroundedAnswerReferenceLayerSelection",
    "GroundedAnswerReferenceSelection",
    "GroundedAnswerReferenceStrategy",
    "build_grounded_answer_reference_selection",
]
