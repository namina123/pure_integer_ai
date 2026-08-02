"""W06-R06 active event-time relation 的 Generation consumer 与 postcheck。"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.identity import ObjectIdentity
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    GenerationChoiceOutcomeRef,
    GenerationChoiceUseRef,
    GenerationExpressionConstraints,
    LosslessIntegerKey,
)
from pure_integer_ai.experiments.ph2_w06_adapter import W06RelationCandidate
from pure_integer_ai.experiments.ph2_w06_r06_contract import (
    W06R06Budget,
    W06R06ContractError,
    W06R06EventTimeQuery,
    W06R06GenerationChoice,
    W06R06GenerationOption,
    W06R06GenerationOutcome,
    W06R06GenerationRequest,
    W06R06GenerationUse,
    W06_R06_GENERATION_READY,
    W06_R06_GENERATION_REJECTED,
    W06_R06_GENERATION_UNKNOWN,
    W06_R06_OUTCOME_REFUTE,
    W06_R06_OUTCOME_SUPPORT,
    W06_R06_RUNTIME_NAMESPACE,
    W06_R06_SUBSTAGE,
    W06_R06_SUPPORTED,
    W06_R06_UNKNOWN,
    pack_key,
)
from pure_integer_ai.experiments.ph2_w06_r06_query import W06R06QueryRuntime
from pure_integer_ai.experiments.ph2_w06_r06_shared import (
    W06R06View,
    candidate_construction,
    candidate_endpoints,
    candidate_event_time_qualifier,
    candidate_role_fillers,
    w06_r06_language_branch,
)


W06_R06_POSTCHECK_BUDGET = W06R06Budget(32, 64)
_GENERATION_DIMENSION = LosslessIntegerKey((W06_R06_RUNTIME_NAMESPACE, 803))
_POSTCHECK_VERIFIER = LosslessIntegerKey((W06_R06_RUNTIME_NAMESPACE, 804))
_RESULT_KEYS = {
    W06_R06_OUTCOME_SUPPORT: LosslessIntegerKey((
        W06_R06_RUNTIME_NAMESPACE, 805)),
    W06_R06_OUTCOME_REFUTE: LosslessIntegerKey((
        W06_R06_RUNTIME_NAMESPACE, 806)),
}


def query_for_candidate(
        candidate: W06RelationCandidate,
        endpoint_resolver,
        *,
        request_key: LosslessIntegerKey,
        budget: W06R06Budget = W06_R06_POSTCHECK_BUDGET,
        ) -> W06R06EventTimeQuery:
    """从 typed Role 和 explicit qualifier 构造不含表层预期的查询。"""
    if (not isinstance(candidate, W06RelationCandidate)
            or candidate.substage_key != W06_R06_SUBSTAGE):
        raise W06R06ContractError("event-time query candidate 不属于 R06")
    if not callable(getattr(endpoint_resolver, "resolve", None)):
        raise W06R06ContractError("event-time query endpoint resolver 非法")
    subject, object_identity = tuple(
        endpoint_resolver.resolve(item)
        for item in candidate_endpoints(candidate)
    )
    return W06R06EventTimeQuery(
        request_key,
        candidate.relation_family,
        subject,
        object_identity,
        candidate_event_time_qualifier(candidate),
        budget,
        candidate.source_ref,
    )


class W06R06GenerationRuntime:
    """从 direct active event-time fact 形成 option、Use 和 postcheck。"""

    def __init__(
            self,
            view: W06R06View,
            understanding: W06R06QueryRuntime,
            ) -> None:
        if not isinstance(view, W06R06View):
            raise TypeError("R06 Generation view 类型非法")
        if (not isinstance(understanding, W06R06QueryRuntime)
                or understanding.view is not view
                or understanding.consumer != "UNDERSTANDING"):
            raise TypeError("R06 Generation 必须共享 Understanding view")
        self.view = view
        self.understanding = understanding
        self._request_keys: set[LosslessIntegerKey] = set()
        self._choices: list[W06R06GenerationChoice] = []
        self._uses: list[W06R06GenerationUse] = []
        self._outcomes: list[W06R06GenerationOutcome] = []

    @property
    def choices(self) -> tuple[W06R06GenerationChoice, ...]:
        return tuple(self._choices)

    @property
    def uses(self) -> tuple[W06R06GenerationUse, ...]:
        return tuple(self._uses)

    @property
    def outcomes(self) -> tuple[W06R06GenerationOutcome, ...]:
        return tuple(self._outcomes)

    @staticmethod
    def _allowed(
            request: W06R06GenerationRequest,
            candidate: W06RelationCandidate,
            construction: ObjectIdentity,
            branch: ObjectIdentity,
            ) -> bool:
        constraints = request.constraints
        if len(candidate.surface) > constraints.max_output_units:
            return False
        if constraints.require_explicit_source:
            return False
        if (constraints.allowed_structure_families
                and candidate.schema.schema not in constraints.allowed_structure_families
                and construction not in constraints.allowed_structure_families):
            return False
        if (constraints.allowed_lexical_branches
                and branch not in constraints.allowed_lexical_branches):
            return False
        return branch == constraints.target_language

    def _options(
            self, request: W06R06GenerationRequest,
            ) -> tuple[W06R06GenerationOption, ...]:
        if not self.view.protocol.generation_ready(request.relation_family):
            return ()
        candidate = self.view.candidate_by_id.get(request.target_proposition)
        if candidate is None or candidate.relation_family != request.relation_family:
            return ()
        qualifier = candidate_event_time_qualifier(candidate)
        if (candidate.directionality != request.directionality
                or candidate_role_fillers(candidate) != request.role_fillers
                or qualifier != request.qualifier
                or candidate.proposition.context != request.context
                or candidate.source_ref != request.source
                or candidate.uncertainty_units != request.uncertainty_units):
            return ()
        authorization = self.view.authorization_key(candidate)
        if authorization is None:
            return ()
        construction = candidate_construction(candidate)
        branch = w06_r06_language_branch(candidate)
        if not self._allowed(request, candidate, construction, branch):
            return ()
        return (W06R06GenerationOption(
            candidate.surface,
            construction,
            candidate.proposition.proposition,
            candidate.relation_family,
            candidate.directionality,
            candidate_role_fillers(candidate),
            qualifier,
            self.view.endpoints_for(candidate),
            candidate.proposition.context,
            candidate.source_ref,
            candidate.uncertainty_units,
            branch,
            authorization,
        ),)

    def choose(
            self, request: W06R06GenerationRequest,
            ) -> W06R06GenerationChoice:
        if not isinstance(request, W06R06GenerationRequest):
            raise TypeError("R06 generation request 类型非法")
        if request.request_key in self._request_keys:
            raise W06R06ContractError("重复 R06 generation request key")
        options = self._options(request)
        candidate = self.view.candidate_by_id.get(request.target_proposition)
        if options:
            status = W06_R06_GENERATION_READY
        elif (not self.view.protocol.family_ready(request.relation_family)
              or not self.view.protocol.generation_connected):
            status = W06_R06_GENERATION_UNKNOWN
        elif (not self.view.protocol.qualifier_connected
              or not self.view.protocol.source_scope_connected
              or (candidate is not None
                  and candidate.relation_family != request.relation_family)):
            status = W06_R06_GENERATION_REJECTED
        else:
            status = W06_R06_GENERATION_UNKNOWN
        choice = W06R06GenerationChoice(
            request,
            status,
            options,
            LosslessIntegerKey((
                W06_R06_RUNTIME_NAMESPACE,
                710,
                *self.view.protocol.stable_key(),
                int(status == W06_R06_GENERATION_READY),
            )),
        )
        self._request_keys.add(request.request_key)
        self._choices.append(choice)
        return choice

    def adopt(
            self,
            choice: W06R06GenerationChoice,
            option_key: tuple[int, ...],
            ) -> W06R06GenerationUse:
        if choice not in self._choices:
            raise W06R06ContractError("generation choice 不属于当前 runtime")
        if choice.status != W06_R06_GENERATION_READY:
            raise W06R06ContractError("非 READY generation choice 不得采用")
        selected = tuple(
            item for item in choice.options if item.stable_key() == option_key)
        if len(selected) != 1:
            raise W06R06ContractError("generation option key 不属于 choice")
        if any(item.choice == choice for item in self._uses):
            raise W06R06ContractError("同一 generation choice 不得重复采用")
        option = selected[0]
        candidate = self.view.candidate_by_id[option.target_proposition]
        use_key = LosslessIntegerKey((
            W06_R06_RUNTIME_NAMESPACE,
            720,
            len(self._uses) + 1,
            *pack_key(choice.stable_key()),
            *pack_key(option.stable_key()),
        ))
        committed = self.view.consume_candidate(candidate, use_key)
        use = W06R06GenerationUse(
            choice,
            option,
            committed,
            GenerationChoiceUseRef(
                "CORE_USE",
                use_key,
                LosslessIntegerKey(option.stable_key()),
                document_scope(option.source),
            ),
        )
        self._uses.append(use)
        return use

    @staticmethod
    def _surface_structure_valid(
            candidate: W06RelationCandidate,
            surface: str,
            ) -> bool:
        endpoints = tuple(sorted(
            candidate.endpoints, key=lambda item: (item.start, item.end)))
        return (
            surface == candidate.surface
            and all(
                surface[item.start:item.end] == item.surface_fragment
                for item in endpoints
            )
            and all(
                left.end <= right.start
                for left, right in zip(endpoints, endpoints[1:])
            )
        )

    def verify(
            self, use: W06R06GenerationUse,
            ) -> W06R06GenerationOutcome:
        """独立重验 authorization、raw family、qualifier、surface 和 query。"""
        if use not in self._uses:
            raise W06R06ContractError("generation Use 不属于当前 runtime")
        option = use.option
        request = use.choice.request
        candidate = self.view.candidate_by_id.get(option.target_proposition)
        authorization = (
            None if candidate is None else self.view.authorization_key(candidate))
        authorization_current = authorization == option.authorization_key
        relation_qualifier_preserved = bool(
            candidate is not None
            and candidate.relation_family == option.relation_family
            == request.relation_family
            and candidate.directionality == option.directionality
            == request.directionality
            and candidate_role_fillers(candidate) == option.role_fillers
            == request.role_fillers
            and candidate_event_time_qualifier(candidate) == option.qualifier
            == request.qualifier
            and self.view.endpoints_for(candidate) == option.canonical_endpoints
            and request.target_proposition == option.target_proposition
        )
        source_scope_preserved = bool(
            candidate is not None
            and candidate.source_ref == option.source == request.source
            and candidate.proposition.context == option.context == request.context
            and option.uncertainty_units == request.uncertainty_units
        )
        surface_valid = bool(
            candidate is not None
            and self._surface_structure_valid(candidate, option.surface))
        query_status = W06_R06_UNKNOWN
        recovered = False
        if candidate is not None and self.view.protocol.postcheck_connected:
            resolution = self.understanding.preview(query_for_candidate(
                candidate,
                self.view.endpoint_resolver,
                request_key=LosslessIntegerKey((
                    W06_R06_RUNTIME_NAMESPACE,
                    730,
                    len(self._outcomes) + 1,
                    *pack_key(use.ref.use_key.components),
                )),
            ))
            query_status = resolution.status
            recovered = (
                resolution.status == W06_R06_SUPPORTED
                and option.target_proposition in resolution.propositions
            )
        verdict = (
            W06_R06_OUTCOME_SUPPORT
            if all((
                authorization_current,
                relation_qualifier_preserved,
                source_scope_preserved,
                surface_valid,
                query_status == W06_R06_SUPPORTED,
                recovered,
            ))
            else W06_R06_OUTCOME_REFUTE
        )
        outcome_key = LosslessIntegerKey((
            W06_R06_RUNTIME_NAMESPACE,
            740,
            len(self._outcomes) + 1,
            *pack_key(use.ref.use_key.components),
        ))
        outcome = W06R06GenerationOutcome(
            use,
            verdict,
            GenerationChoiceOutcomeRef(
                outcome_key,
                use.ref.use_key,
                _GENERATION_DIMENSION,
                _POSTCHECK_VERIFIER,
                _RESULT_KEYS[verdict],
            ),
            authorization_current,
            relation_qualifier_preserved,
            source_scope_preserved,
            surface_valid,
            query_status,
            recovered,
        )
        self._outcomes.append(outcome)
        return outcome

    def state_key(self) -> tuple:
        return (
            tuple(item.stable_key() for item in self._choices),
            tuple(item.ref.stable_key() for item in self._uses),
            tuple(item.ref.stable_key() for item in self._outcomes),
        )


def generation_request_for_candidate(
        candidate: W06RelationCandidate,
        *,
        request_key: LosslessIntegerKey,
        constraints: GenerationExpressionConstraints,
        ) -> W06R06GenerationRequest:
    if (not isinstance(candidate, W06RelationCandidate)
            or candidate.substage_key != W06_R06_SUBSTAGE):
        raise W06R06ContractError("generation candidate 不属于 R06")
    return W06R06GenerationRequest(
        request_key,
        candidate.proposition.proposition,
        candidate.relation_family,
        candidate.directionality,
        candidate_role_fillers(candidate),
        candidate_event_time_qualifier(candidate),
        candidate.proposition.context,
        candidate.source_ref,
        candidate.uncertainty_units,
        constraints,
    )


__all__ = [
    "W06R06GenerationRuntime",
    "W06_R06_POSTCHECK_BUDGET",
    "generation_request_for_candidate",
    "query_for_candidate",
]
