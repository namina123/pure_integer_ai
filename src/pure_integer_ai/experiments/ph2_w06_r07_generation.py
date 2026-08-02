"""W06-R07 active direct CAUSES relation 的 Generation consumer。"""
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
from pure_integer_ai.experiments.ph2_w06_r07_contract import (
    W06R07Budget,
    W06R07CausalQuery,
    W06R07ContractError,
    W06R07GenerationChoice,
    W06R07GenerationOption,
    W06R07GenerationOutcome,
    W06R07GenerationRequest,
    W06R07GenerationUse,
    W06_R07_GENERATION_READY,
    W06_R07_GENERATION_REJECTED,
    W06_R07_GENERATION_UNKNOWN,
    W06_R07_OUTCOME_REFUTE,
    W06_R07_OUTCOME_SUPPORT,
    W06_R07_RUNTIME_NAMESPACE,
    W06_R07_SUBSTAGE,
    W06_R07_SUPPORTED,
    W06_R07_UNKNOWN,
    pack_key,
)
from pure_integer_ai.experiments.ph2_w06_r07_query import W06R07QueryRuntime
from pure_integer_ai.experiments.ph2_w06_r07_shared import (
    W06R07View,
    candidate_causal_protocol,
    candidate_construction,
    candidate_endpoints,
    candidate_role_fillers,
    w06_r07_language_branch,
)


W06_R07_POSTCHECK_BUDGET = W06R07Budget(32, 64, 512)
_GENERATION_DIMENSION = LosslessIntegerKey((W06_R07_RUNTIME_NAMESPACE, 803))
_POSTCHECK_VERIFIER = LosslessIntegerKey((W06_R07_RUNTIME_NAMESPACE, 804))
_RESULT_KEYS = {
    W06_R07_OUTCOME_SUPPORT: LosslessIntegerKey((
        W06_R07_RUNTIME_NAMESPACE, 805)),
    W06_R07_OUTCOME_REFUTE: LosslessIntegerKey((
        W06_R07_RUNTIME_NAMESPACE, 806)),
}


def query_for_candidate(
        candidate: W06RelationCandidate,
        endpoint_resolver,
        *,
        request_key: LosslessIntegerKey,
        budget: W06R07Budget = W06_R07_POSTCHECK_BUDGET,
        ) -> W06R07CausalQuery:
    if (not isinstance(candidate, W06RelationCandidate)
            or candidate.substage_key != W06_R07_SUBSTAGE):
        raise W06R07ContractError("causal query candidate 不属于 R07")
    if not callable(getattr(endpoint_resolver, "resolve", None)):
        raise W06R07ContractError("causal query endpoint resolver 非法")
    cause, effect = tuple(
        endpoint_resolver.resolve(item)
        for item in candidate_endpoints(candidate)
    )
    return W06R07CausalQuery(
        request_key,
        cause,
        effect,
        budget,
        candidate.source_ref,
    )


class W06R07GenerationRuntime:
    """从 direct active CAUSES fact 形成 statement option、Use 和 postcheck。"""

    def __init__(
            self,
            view: W06R07View,
            understanding: W06R07QueryRuntime,
            ) -> None:
        if not isinstance(view, W06R07View):
            raise TypeError("R07 Generation view 类型非法")
        if (not isinstance(understanding, W06R07QueryRuntime)
                or understanding.view is not view
                or understanding.consumer != "UNDERSTANDING"):
            raise TypeError("R07 Generation 必须共享 Understanding view")
        self.view = view
        self.understanding = understanding
        self._request_keys: set[LosslessIntegerKey] = set()
        self._choices: list[W06R07GenerationChoice] = []
        self._uses: list[W06R07GenerationUse] = []
        self._outcomes: list[W06R07GenerationOutcome] = []

    @property
    def choices(self) -> tuple[W06R07GenerationChoice, ...]:
        return tuple(self._choices)

    @property
    def uses(self) -> tuple[W06R07GenerationUse, ...]:
        return tuple(self._uses)

    @property
    def outcomes(self) -> tuple[W06R07GenerationOutcome, ...]:
        return tuple(self._outcomes)

    @staticmethod
    def _allowed(
            request: W06R07GenerationRequest,
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
            self, request: W06R07GenerationRequest,
            ) -> tuple[W06R07GenerationOption, ...]:
        if not self.view.protocol.generation_ready():
            return ()
        candidate = self.view.candidate_by_id.get(request.target_proposition)
        if candidate is None:
            return ()
        endpoint_protocol = candidate_causal_protocol(candidate)
        if (candidate.directionality != request.directionality
                or candidate_role_fillers(candidate) != request.role_fillers
                or endpoint_protocol != request.endpoints
                or candidate.proposition.context != request.context
                or candidate.source_ref != request.source
                or candidate.uncertainty_units != request.uncertainty_units):
            return ()
        authorization = self.view.authorization_key(candidate)
        witness_keys = self.view.witness_keys(candidate)
        if authorization is None or not witness_keys:
            return ()
        construction = candidate_construction(candidate)
        branch = w06_r07_language_branch(candidate)
        if not self._allowed(request, candidate, construction, branch):
            return ()
        return (W06R07GenerationOption(
            candidate.surface,
            construction,
            candidate.proposition.proposition,
            candidate.directionality,
            candidate_role_fillers(candidate),
            endpoint_protocol,
            self.view.endpoints_for(candidate),
            witness_keys,
            candidate.proposition.context,
            candidate.source_ref,
            candidate.uncertainty_units,
            branch,
            authorization,
        ),)

    def choose(
            self, request: W06R07GenerationRequest,
            ) -> W06R07GenerationChoice:
        if not isinstance(request, W06R07GenerationRequest):
            raise TypeError("R07 generation request 类型非法")
        if request.request_key in self._request_keys:
            raise W06R07ContractError("重复 R07 generation request key")
        options = self._options(request)
        candidate = self.view.candidate_by_id.get(request.target_proposition)
        if options:
            status = W06_R07_GENERATION_READY
        elif (not self.view.protocol.causes_connected
              or not self.view.protocol.witness_connected
              or not self.view.protocol.temporal_boundary_connected
              or not self.view.protocol.generation_connected):
            status = W06_R07_GENERATION_UNKNOWN
        elif (not self.view.protocol.source_scope_connected
              or (candidate is not None
                  and candidate_causal_protocol(candidate) != request.endpoints)):
            status = W06_R07_GENERATION_REJECTED
        else:
            status = W06_R07_GENERATION_UNKNOWN
        choice = W06R07GenerationChoice(
            request,
            status,
            options,
            LosslessIntegerKey((
                W06_R07_RUNTIME_NAMESPACE,
                710,
                *self.view.protocol.stable_key(),
                int(status == W06_R07_GENERATION_READY),
            )),
        )
        self._request_keys.add(request.request_key)
        self._choices.append(choice)
        return choice

    def adopt(
            self,
            choice: W06R07GenerationChoice,
            option_key: tuple[int, ...],
            ) -> W06R07GenerationUse:
        if choice not in self._choices:
            raise W06R07ContractError("generation choice 不属于当前 runtime")
        if choice.status != W06_R07_GENERATION_READY:
            raise W06R07ContractError("非 READY generation choice 不得采用")
        selected = tuple(
            item for item in choice.options if item.stable_key() == option_key)
        if len(selected) != 1:
            raise W06R07ContractError("generation option key 不属于 choice")
        if any(item.choice == choice for item in self._uses):
            raise W06R07ContractError("同一 generation choice 不得重复采用")
        option = selected[0]
        candidate = self.view.candidate_by_id[option.target_proposition]
        use_key = LosslessIntegerKey((
            W06_R07_RUNTIME_NAMESPACE,
            720,
            len(self._uses) + 1,
            *pack_key(choice.stable_key()),
            *pack_key(option.stable_key()),
        ))
        committed = self.view.consume_candidate(candidate, use_key)
        use = W06R07GenerationUse(
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
            self, use: W06R07GenerationUse,
            ) -> W06R07GenerationOutcome:
        if use not in self._uses:
            raise W06R07ContractError("generation Use 不属于当前 runtime")
        option = use.option
        request = use.choice.request
        candidate = self.view.candidate_by_id.get(option.target_proposition)
        authorization = (
            None if candidate is None else self.view.authorization_key(candidate))
        authorization_current = authorization == option.authorization_key
        witness_current = bool(
            candidate is not None
            and self.view.witness_keys(candidate) == option.witness_keys)
        causal_structure_preserved = bool(
            candidate is not None
            and candidate.directionality == option.directionality
            == request.directionality
            and candidate_role_fillers(candidate) == option.role_fillers
            == request.role_fillers
            and candidate_causal_protocol(candidate) == option.endpoints
            == request.endpoints
            and self.view.endpoints_for(candidate) == option.canonical_pair
            and candidate_endpoints(candidate) == tuple(
                binding.filler
                for binding in candidate.proposition.canonical_bindings()
                if binding.role in {
                    option.endpoints.cause_role, option.endpoints.effect_role}
            )
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
        query_status = W06_R07_UNKNOWN
        recovered = False
        if candidate is not None and self.view.protocol.postcheck_connected:
            resolution = self.understanding.preview(query_for_candidate(
                candidate,
                self.view.endpoint_resolver,
                request_key=LosslessIntegerKey((
                    W06_R07_RUNTIME_NAMESPACE,
                    730,
                    len(self._outcomes) + 1,
                    *pack_key(use.ref.use_key.components),
                )),
            ))
            query_status = resolution.status
            recovered = (
                resolution.status == W06_R07_SUPPORTED
                and option.target_proposition in resolution.propositions
            )
        verdict = (
            W06_R07_OUTCOME_SUPPORT
            if all((
                authorization_current,
                witness_current,
                causal_structure_preserved,
                source_scope_preserved,
                surface_valid,
                query_status == W06_R07_SUPPORTED,
                recovered,
            ))
            else W06_R07_OUTCOME_REFUTE
        )
        outcome_key = LosslessIntegerKey((
            W06_R07_RUNTIME_NAMESPACE,
            740,
            len(self._outcomes) + 1,
            *pack_key(use.ref.use_key.components),
        ))
        outcome = W06R07GenerationOutcome(
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
            witness_current,
            causal_structure_preserved,
            source_scope_preserved,
            surface_valid,
            query_status,
            recovered,
            False,
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
        ) -> W06R07GenerationRequest:
    if (not isinstance(candidate, W06RelationCandidate)
            or candidate.substage_key != W06_R07_SUBSTAGE):
        raise W06R07ContractError("generation candidate 不属于 R07")
    return W06R07GenerationRequest(
        request_key,
        candidate.proposition.proposition,
        candidate.directionality,
        candidate_role_fillers(candidate),
        candidate_causal_protocol(candidate),
        candidate.proposition.context,
        candidate.source_ref,
        candidate.uncertainty_units,
        constraints,
    )


__all__ = [
    "W06R07GenerationRuntime",
    "W06_R07_POSTCHECK_BUDGET",
    "generation_request_for_candidate",
    "query_for_candidate",
]
