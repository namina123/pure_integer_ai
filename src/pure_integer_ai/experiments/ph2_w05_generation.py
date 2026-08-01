"""W05-G active Proposition construction 到 surface choice/Use/outcome。"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.identity import (
    ObjectIdentity,
    concept_identity,
)
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    GenerationChoiceOutcomeRef,
    GenerationChoiceUseRef,
    GenerationExpressionConstraints,
    LosslessIntegerKey,
)
from pure_integer_ai.experiments.ph2_w05_adapter import (
    W05AtomicPropositionCandidate,
    W05OccurrenceBinding,
    W05_IDENTITY_VERSIONS,
    W05_NAMESPACE,
)
from pure_integer_ai.experiments.ph2_w05_generation_contract import (
    W05_GENERATION_ADOPTED,
    W05_GENERATION_OPTION_REJECTED,
    W05_GENERATION_OUTCOME_NEUTRAL,
    W05_GENERATION_OUTCOME_REFUTE,
    W05_GENERATION_OUTCOME_SUPPORT,
    W05_GENERATION_READY,
    W05_GENERATION_REJECTED,
    W05_GENERATION_UNKNOWN,
    W05GenerationChoice,
    W05GenerationDecision,
    W05GenerationError,
    W05GenerationOption,
    W05GenerationOutcome,
    W05GenerationProtocol,
    W05GenerationRequest,
    W05GenerationUse,
)
from pure_integer_ai.experiments.ph2_w05_learning import (
    W05AtomicPropositionLearningRuntime,
)
from pure_integer_ai.experiments.ph2_w05_understanding import (
    W05_UNDERSTANDING_UNIQUE,
    W05UnderstandingRequest,
    W05UnderstandingRuntime,
)


_NAMESPACE = 50515
_GENERATION_DIMENSION_KEY = LosslessIntegerKey((_NAMESPACE, 301))
_INDEPENDENT_UNDERSTANDING_VERIFIER_KEY = LosslessIntegerKey((_NAMESPACE, 302))
_OUTCOME_RESULT_KEYS = {
    W05_GENERATION_OUTCOME_SUPPORT: LosslessIntegerKey((_NAMESPACE, 303)),
    W05_GENERATION_OUTCOME_REFUTE: LosslessIntegerKey((_NAMESPACE, 304)),
    W05_GENERATION_OUTCOME_NEUTRAL: LosslessIntegerKey((_NAMESPACE, 305)),
}


def _pack(value: tuple[int, ...]) -> tuple[int, ...]:
    return len(value), *value


def _role_signature(candidate: W05AtomicPropositionCandidate) -> tuple[
        tuple[ObjectIdentity, int], ...]:
    return tuple(sorted(
        ((item.role, item.ordinal)
         for item in candidate.proposition_definition.canonical_bindings()),
        key=lambda item: (item[0].stable_key(), item[1]),
    ))


def _request_role_signature(request: W05GenerationRequest) -> tuple[
        tuple[ObjectIdentity, int], ...]:
    return tuple(sorted(
        ((item.role, item.ordinal)
         for item in request.target.canonical_bindings()),
        key=lambda item: (item[0].stable_key(), item[1]),
    ))


def _construction(candidate: W05AtomicPropositionCandidate) -> ObjectIdentity:
    key = candidate.observation.template_group_key.stable_key()
    return concept_identity(
        (W05_NAMESPACE, 800, *_pack(key)),
        versions=W05_IDENTITY_VERSIONS,
    )


def _authorization_key(
        learning: W05AtomicPropositionLearningRuntime,
        candidate: W05AtomicPropositionCandidate,
        ) -> LosslessIntegerKey | None:
    hypothesis = learning.hypothesis_for(candidate.candidate)
    active = learning.learning.engine.active(hypothesis)
    if active is None:
        return None
    return LosslessIntegerKey((
        _NAMESPACE,
        1,
        *_pack(candidate.candidate.stable_key()),
        *_pack(active.hypothesis.stable_key()),
        *_pack(active.decision.stable_key()),
    ))


def _template_plan(
        candidate: W05AtomicPropositionCandidate,
        ) -> tuple[
            tuple[tuple[int, ObjectIdentity | None, int], ...],
            tuple[str, ...],
        ] | None:
    """把 active surface 无解释地拆成 predicate/Role slot 与 literal gaps。"""
    occurrences = candidate.occurrences
    if any(
            current.end > following.start
            for current, following in zip(occurrences, occurrences[1:])):
        return None
    slots = []
    bindings = candidate.proposition_definition.canonical_bindings()
    for occurrence in occurrences:
        if occurrence.identity == candidate.proposition_definition.source_anchor:
            slots.append((1, None, 0))
            continue
        matches = tuple(
            item for item in bindings
            if item.filler == occurrence.semantic_object
        )
        if len(matches) != 1:
            return None
        slots.append((2, matches[0].role, matches[0].ordinal))
    gaps = [candidate.surface[:occurrences[0].start]]
    for current, following in zip(occurrences, occurrences[1:]):
        gaps.append(candidate.surface[current.end:following.start])
    gaps.append(candidate.surface[occurrences[-1].end:])
    return tuple(slots), tuple(gaps)


def _target_occurrence(
        request: W05GenerationRequest,
        slot: tuple[int, ObjectIdentity | None, int],
        ) -> W05OccurrenceBinding | None:
    kind, role, ordinal = slot
    if kind == 1:
        return next((
            item for item in request.occurrences
            if item.identity == request.target.source_anchor
        ), None)
    binding = next((
        item for item in request.target.canonical_bindings()
        if item.role == role and item.ordinal == ordinal
    ), None)
    if binding is None:
        return None
    values = tuple(
        item for item in request.occurrences
        if item.semantic_object == binding.filler
    )
    return values[0] if len(values) == 1 else None


def _render(
        template: W05AtomicPropositionCandidate,
        request: W05GenerationRequest,
        ) -> str | None:
    planned = _template_plan(template)
    if planned is None:
        return None
    slots, gaps = planned
    target_occurrences = tuple(_target_occurrence(request, slot) for slot in slots)
    if any(item is None for item in target_occurrences):
        return None
    resolved = tuple(item for item in target_occurrences if item is not None)
    if tuple(item.identity for item in resolved) != request.occurrence_order:
        return None
    values = [gaps[0]]
    cursor = len(gaps[0])
    for index, occurrence in enumerate(resolved):
        if occurrence.start != cursor:
            return None
        values.append(occurrence.surface_fragment)
        cursor += len(occurrence.surface_fragment)
        if occurrence.end != cursor:
            return None
        values.append(gaps[index + 1])
        cursor += len(gaps[index + 1])
    return "".join(values)


def generation_request_for_candidate(
        candidate: W05AtomicPropositionCandidate,
        *,
        request_key: LosslessIntegerKey,
        uncertainty: ObjectIdentity,
        constraints: GenerationExpressionConstraints,
        ) -> W05GenerationRequest:
    """从 typed target 建立不含 evaluator expected surface/label 的请求。"""
    if not isinstance(candidate, W05AtomicPropositionCandidate):
        raise TypeError("candidate 类型非法")
    return W05GenerationRequest(
        request_key,
        candidate.proposition_definition,
        candidate.occurrences,
        candidate.occurrence_order,
        tuple(sorted(
            candidate.role_binding_identities(),
            key=ObjectIdentity.stable_key,
        )),
        candidate.source_ref,
        document_scope(candidate.source_ref),
        uncertainty,
        constraints,
    )


class W05GenerationRuntime:
    """从 active construction Evidence 形成 target Proposition 的合法表达选择。"""

    def __init__(
            self,
            learning: W05AtomicPropositionLearningRuntime,
            *,
            protocol: W05GenerationProtocol = W05GenerationProtocol(),
            ) -> None:
        if not isinstance(learning, W05AtomicPropositionLearningRuntime):
            raise TypeError("learning 必须是 W05AtomicPropositionLearningRuntime")
        if not isinstance(protocol, W05GenerationProtocol):
            raise TypeError("generation protocol 类型非法")
        self.learning = learning
        self.protocol = protocol
        self._candidate_by_id = {
            item.candidate: item for item in learning.registered_candidates()
        }
        self._request_keys: set[LosslessIntegerKey] = set()
        self._choices: list[W05GenerationChoice] = []
        self._decisions: list[W05GenerationDecision] = []
        self._uses: list[W05GenerationUse] = []
        self._outcomes: list[W05GenerationOutcome] = []

    @property
    def choices(self) -> tuple[W05GenerationChoice, ...]:
        return tuple(self._choices)

    @property
    def decisions(self) -> tuple[W05GenerationDecision, ...]:
        return tuple(self._decisions)

    @property
    def uses(self) -> tuple[W05GenerationUse, ...]:
        return tuple(self._uses)

    @property
    def outcomes(self) -> tuple[W05GenerationOutcome, ...]:
        return tuple(self._outcomes)

    @staticmethod
    def _allowed(
            request: W05GenerationRequest,
            construction: ObjectIdentity,
            surface: str,
            ) -> bool:
        constraints = request.constraints
        if len(surface) > constraints.max_output_units:
            return False
        if constraints.require_explicit_source:
            return False
        if (constraints.allowed_structure_families
                and construction not in constraints.allowed_structure_families):
            return False
        if (constraints.allowed_lexical_branches
                and constraints.target_language
                not in constraints.allowed_lexical_branches):
            return False
        return True

    def _options(
            self,
            request: W05GenerationRequest,
            ) -> tuple[W05GenerationOption, ...]:
        values = []
        target_roles = _request_role_signature(request)
        for template in self.learning.active_candidates():
            if (template.proposition_definition.predicate != request.target.predicate
                    or _role_signature(template) != target_roles):
                continue
            authorization = _authorization_key(self.learning, template)
            if authorization is None:
                continue
            construction = _construction(template)
            surface = _render(template, request)
            if surface is None or not self._allowed(request, construction, surface):
                continue
            values.append(W05GenerationOption(
                surface,
                construction,
                template.candidate,
                request.target.proposition,
                request.target.predicate,
                request.occurrence_order,
                request.role_bindings,
                request.target.context,
                request.source,
                request.uncertainty,
                request.constraints.target_language,
                authorization,
            ))
        return tuple(sorted(values, key=lambda item: item.stable_key()))

    def choose(self, request: W05GenerationRequest) -> W05GenerationChoice:
        """返回全部合法 construction；不输入或比较 evaluator expected surface。"""
        if not isinstance(request, W05GenerationRequest):
            raise TypeError("generation request 类型非法")
        if request.request_key in self._request_keys:
            raise W05GenerationError("重复 generation request key")
        if (not self.protocol.proposition_consumer_connected
                or not self.protocol.generation_bridge_connected):
            status = W05_GENERATION_UNKNOWN
            options = ()
        elif not self.protocol.structure_connected():
            status = W05_GENERATION_REJECTED
            options = ()
        else:
            options = self._options(request)
            status = W05_GENERATION_READY if options else W05_GENERATION_UNKNOWN
        choice = W05GenerationChoice(
            request,
            status,
            options,
            LosslessIntegerKey((
                _NAMESPACE,
                100,
                W05_GENERATION_READY == status and 1 or 0,
                int(self.protocol.occurrence_identity_connected),
                int(self.protocol.proposition_consumer_connected),
                int(self.protocol.role_bridge_connected),
                int(self.protocol.scope_projection_connected),
                int(self.protocol.choice_bridge_connected),
                int(self.protocol.generation_bridge_connected),
            )),
        )
        if self.protocol.choice_bridge_connected:
            self._request_keys.add(request.request_key)
            self._choices.append(choice)
        return choice

    def adopt(
            self,
            choice: W05GenerationChoice,
            selected_option_keys: tuple[tuple[int, ...], ...],
            ) -> tuple[W05GenerationUse, ...]:
        """对 choice 内每个 option 原子记录 decision 和 exact Use。"""
        if not isinstance(choice, W05GenerationChoice):
            raise TypeError("generation choice 类型非法")
        if not self.protocol.choice_bridge_connected:
            return ()
        if choice not in self._choices:
            raise W05GenerationError("choice 不属于当前 Generation runtime")
        if choice.status != W05_GENERATION_READY:
            raise W05GenerationError("非 READY choice 不得采用 option")
        if (not isinstance(selected_option_keys, tuple)
                or not selected_option_keys
                or any(not isinstance(item, tuple) or not item
                       or any(type(value) is not int for value in item)
                       for item in selected_option_keys)
                or len(set(selected_option_keys)) != len(selected_option_keys)):
            raise W05GenerationError("selected option keys 非法")
        available = {item.stable_key(): item for item in choice.options}
        if any(item not in available for item in selected_option_keys):
            raise W05GenerationError("selected option 不属于当前 choice")
        if any(item.decision.choice == choice for item in self._uses):
            raise W05GenerationError("同一 choice 不得重复形成 Use")
        selected = set(selected_option_keys)
        decisions = []
        uses = []
        for ordinal, option in enumerate(choice.options, start=1):
            action = (
                W05_GENERATION_ADOPTED
                if option.stable_key() in selected
                else W05_GENERATION_OPTION_REJECTED
            )
            decision = W05GenerationDecision(
                choice,
                option,
                action,
                LosslessIntegerKey((
                    _NAMESPACE,
                    200,
                    ordinal,
                    *_pack(choice.stable_key()),
                    *_pack(option.stable_key()),
                    1 if action == W05_GENERATION_ADOPTED else 2,
                )),
            )
            use_key = LosslessIntegerKey((
                _NAMESPACE,
                210,
                *_pack(decision.stable_key()),
            ))
            use = W05GenerationUse(
                decision,
                GenerationChoiceUseRef(
                    "CORE_USE",
                    use_key,
                    LosslessIntegerKey(option.stable_key()),
                    choice.request.scope,
                ),
            )
            decisions.append(decision)
            uses.append(use)
        self._decisions.extend(decisions)
        self._uses.extend(uses)
        return tuple(uses)

    def _current_authorization(
            self,
            option: W05GenerationOption,
            ) -> LosslessIntegerKey | None:
        candidate = self._candidate_by_id.get(option.construction_source_candidate)
        if candidate is None or _construction(candidate) != option.construction:
            return None
        return _authorization_key(self.learning, candidate)

    def verify_use(
            self,
            use: W05GenerationUse,
            *,
            understanding: W05UnderstandingRuntime,
            ) -> W05GenerationOutcome:
        """由独立 Understanding 回读 surface，并逐层归因 exact Use。"""
        if use not in self._uses:
            raise W05GenerationError("Generation Use 不属于当前 runtime")
        if not isinstance(understanding, W05UnderstandingRuntime):
            raise TypeError("independent understanding runtime 类型非法")
        option = use.decision.option
        request = use.decision.choice.request
        current = self._current_authorization(option)
        occurrence_preserved = option.occurrence_order == request.occurrence_order
        role_preserved = option.role_bindings == request.role_bindings
        scope_preserved = (
            option.context == request.target.context
            and option.source == request.source
        )
        proposition_preserved = (
            option.target_proposition == request.target.proposition
            and option.target_predicate == request.target.predicate
            and option.uncertainty == request.uncertainty
        )
        understanding_status = W05_GENERATION_UNKNOWN
        recovered = False
        if self.protocol.independent_understanding_connected:
            resolution = understanding.resolve(W05UnderstandingRequest(
                LosslessIntegerKey((
                    _NAMESPACE,
                    400,
                    len(self._outcomes) + 1,
                    *_pack(use.ref.use_key.components),
                )),
                option.surface,
                option.occurrence_order,
                option.role_bindings,
                option.context,
                False,
            ))
            understanding_status = resolution.status
            recovered = (
                resolution.status == W05_UNDERSTANDING_UNIQUE
                and resolution.selected is not None
                and resolution.selected.candidate == option.target_proposition
            )
        preserved = all((
            occurrence_preserved,
            role_preserved,
            scope_preserved,
            proposition_preserved,
            recovered,
        ))
        if use.decision.action == W05_GENERATION_ADOPTED:
            verdict = (
                W05_GENERATION_OUTCOME_SUPPORT
                if current is not None and preserved
                else W05_GENERATION_OUTCOME_REFUTE
            )
        else:
            verdict = (
                W05_GENERATION_OUTCOME_NEUTRAL
                if current is not None else W05_GENERATION_OUTCOME_SUPPORT
            )
        outcome_key = LosslessIntegerKey((
            _NAMESPACE,
            500,
            len(self._outcomes) + 1,
            *_pack(use.ref.use_key.components),
        ))
        outcome = W05GenerationOutcome(
            use,
            verdict,
            GenerationChoiceOutcomeRef(
                outcome_key,
                use.ref.use_key,
                _GENERATION_DIMENSION_KEY,
                _INDEPENDENT_UNDERSTANDING_VERIFIER_KEY,
                _OUTCOME_RESULT_KEYS[verdict],
            ),
            current,
            understanding_status,
            occurrence_preserved,
            role_preserved,
            scope_preserved,
            proposition_preserved,
        )
        self._outcomes.append(outcome)
        return outcome


def build_w05_generation_runtime(
        learning: W05AtomicPropositionLearningRuntime,
        *,
        protocol: W05GenerationProtocol = W05GenerationProtocol(),
        ) -> W05GenerationRuntime:
    """建立 W05-03 Generation consumer，不启动正式训练或开放生成。"""
    return W05GenerationRuntime(learning, protocol=protocol)


__all__ = [
    "W05GenerationRuntime",
    "build_w05_generation_runtime",
    "generation_request_for_candidate",
]
