"""W06-R05 active pair/channel 的 Generation consumer 与 postcheck。"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.identity import ObjectIdentity
from pure_integer_ai.cognition.shared.symmetric_relation import (
    SymmetricRelationBudget,
)
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    GenerationChoiceOutcomeRef,
    GenerationChoiceUseRef,
    GenerationExpressionConstraints,
    LosslessIntegerKey,
)
from pure_integer_ai.experiments.ph2_w06_adapter import W06RelationCandidate
from pure_integer_ai.experiments.ph2_w06_r05_contract import (
    W06R05ContractError,
    W06R05GenerationChoice,
    W06R05GenerationOption,
    W06R05GenerationOutcome,
    W06R05GenerationRequest,
    W06R05GenerationUse,
    W06R05PairQuery,
    W06_R05_GENERATION_READY,
    W06_R05_GENERATION_REJECTED,
    W06_R05_GENERATION_UNKNOWN,
    W06_R05_OUTCOME_REFUTE,
    W06_R05_OUTCOME_SUPPORT,
    W06_R05_SUBSTAGE,
    W06_R05_SUPPORTED,
    W06_R05_UNKNOWN,
    pack_key,
)
from pure_integer_ai.experiments.ph2_w06_r05_query import W06R05QueryRuntime
from pure_integer_ai.experiments.ph2_w06_r05_shared import (
    W06R05View,
    W06_R05_RUNTIME_NAMESPACE,
    candidate_construction,
    candidate_endpoints,
    candidate_role_fillers,
    w06_r05_language_branch,
)


W06_R05_POSTCHECK_BUDGET = SymmetricRelationBudget(32, 32)
_GENERATION_DIMENSION = LosslessIntegerKey((W06_R05_RUNTIME_NAMESPACE, 803))
_POSTCHECK_VERIFIER = LosslessIntegerKey((W06_R05_RUNTIME_NAMESPACE, 804))
_RESULT_KEYS = {
    W06_R05_OUTCOME_SUPPORT: LosslessIntegerKey((
        W06_R05_RUNTIME_NAMESPACE, 805)),
    W06_R05_OUTCOME_REFUTE: LosslessIntegerKey((
        W06_R05_RUNTIME_NAMESPACE, 806)),
}


def query_for_candidate(
        candidate: W06RelationCandidate,
        *, request_key: LosslessIntegerKey,
        budget: SymmetricRelationBudget = W06_R05_POSTCHECK_BUDGET,
        ) -> W06R05PairQuery:
    """从 typed Role 构造不含表层预期的 pair+channel 查询。"""
    if (not isinstance(candidate, W06RelationCandidate)
            or candidate.substage_key != W06_R05_SUBSTAGE):
        raise W06R05ContractError("pair query candidate 不属于 R05")
    left, right = candidate_endpoints(candidate)
    return W06R05PairQuery(
        request_key,
        candidate.relation_family,
        left,
        right,
        budget,
        candidate.source_ref,
    )


class W06R05GenerationRuntime:
    """从 direct active pair fact 形成 option、exact Use 和 postcheck。"""

    def __init__(
            self,
            view: W06R05View,
            understanding: W06R05QueryRuntime,
            ) -> None:
        """绑定共享 view 和 Understanding preview 入口。"""
        if not isinstance(view, W06R05View):
            raise TypeError("R05 Generation view 类型非法")
        if (not isinstance(understanding, W06R05QueryRuntime)
                or understanding.view is not view
                or understanding.consumer != "UNDERSTANDING"):
            raise TypeError("R05 Generation 必须共享 Understanding view")
        self.view = view
        self.understanding = understanding
        self._request_keys: set[LosslessIntegerKey] = set()
        self._choices: list[W06R05GenerationChoice] = []
        self._uses: list[W06R05GenerationUse] = []
        self._outcomes: list[W06R05GenerationOutcome] = []

    @property
    def choices(self) -> tuple[W06R05GenerationChoice, ...]:
        """返回 generation choice 历史。"""
        return tuple(self._choices)

    @property
    def uses(self) -> tuple[W06R05GenerationUse, ...]:
        """返回 generation Use 历史。"""
        return tuple(self._uses)

    @property
    def outcomes(self) -> tuple[W06R05GenerationOutcome, ...]:
        """返回 generation postcheck 历史。"""
        return tuple(self._outcomes)

    @staticmethod
    def _allowed(
            request: W06R05GenerationRequest,
            candidate: W06RelationCandidate,
            construction: ObjectIdentity,
            branch: ObjectIdentity,
            ) -> bool:
        """检查输出约束，不使用 expected surface 或 label。"""
        constraints = request.constraints
        if len(candidate.surface) > constraints.max_output_units:
            return False
        if constraints.require_explicit_source:
            return False
        if (constraints.allowed_structure_families
                and candidate.schema.schema
                not in constraints.allowed_structure_families
                and construction not in constraints.allowed_structure_families):
            return False
        if (constraints.allowed_lexical_branches
                and branch not in constraints.allowed_lexical_branches):
            return False
        return branch == constraints.target_language

    def _options(
            self, request: W06R05GenerationRequest,
            ) -> tuple[W06R05GenerationOption, ...]:
        """按 current active direct fact 形成唯一 source-grounded option。"""
        if not self.view.protocol.generation_ready(request.channel):
            return ()
        candidate = self.view.candidate_by_id.get(request.target_proposition)
        if candidate is None or candidate.relation_family != request.channel:
            return ()
        if (candidate.directionality != request.directionality
                or candidate_role_fillers(candidate) != request.role_fillers
                or candidate.proposition.context != request.context
                or candidate.source_ref != request.source
                or candidate.uncertainty_units != request.uncertainty_units):
            return ()
        authorization = self.view.authorization_key(candidate)
        if authorization is None:
            return ()
        construction = candidate_construction(candidate)
        branch = w06_r05_language_branch(candidate)
        if not self._allowed(request, candidate, construction, branch):
            return ()
        pair = tuple(sorted(
            candidate_endpoints(candidate), key=ObjectIdentity.stable_key))
        return (W06R05GenerationOption(
            candidate.surface,
            construction,
            candidate.proposition.proposition,
            candidate.relation_family,
            candidate.directionality,
            candidate_role_fillers(candidate),
            pair,
            candidate.proposition.context,
            candidate.source_ref,
            candidate.uncertainty_units,
            branch,
            authorization,
        ),)

    def choose(
            self, request: W06R05GenerationRequest,
            ) -> W06R05GenerationChoice:
        """形成 source-grounded option，不接受 expected string 或 label。"""
        if not isinstance(request, W06R05GenerationRequest):
            raise TypeError("R05 generation request 类型非法")
        if request.request_key in self._request_keys:
            raise W06R05ContractError("重复 R05 generation request key")
        options = self._options(request)
        candidate = self.view.candidate_by_id.get(request.target_proposition)
        if options:
            status = W06_R05_GENERATION_READY
        elif (not self.view.protocol.channel_ready(request.channel)
              or not self.view.protocol.generation_connected):
            status = W06_R05_GENERATION_UNKNOWN
        elif (not self.view.protocol.channel_identity_connected
              or not self.view.protocol.source_scope_connected
              or (candidate is not None
                  and candidate.relation_family != request.channel)):
            status = W06_R05_GENERATION_REJECTED
        else:
            status = W06_R05_GENERATION_UNKNOWN
        choice = W06R05GenerationChoice(
            request,
            status,
            options,
            LosslessIntegerKey((
                W06_R05_RUNTIME_NAMESPACE,
                710,
                *self.view.protocol.stable_key(),
                int(status == W06_R05_GENERATION_READY),
            )),
        )
        self._request_keys.add(request.request_key)
        self._choices.append(choice)
        return choice

    def adopt(
            self,
            choice: W06R05GenerationChoice,
            option_key: tuple[int, ...],
            ) -> W06R05GenerationUse:
        """采用 option，并提交该 target direct fact 的唯一 R-00 Use。"""
        if choice not in self._choices:
            raise W06R05ContractError("generation choice 不属于当前 runtime")
        if choice.status != W06_R05_GENERATION_READY:
            raise W06R05ContractError("非 READY generation choice 不得采用")
        selected = tuple(
            item for item in choice.options if item.stable_key() == option_key)
        if len(selected) != 1:
            raise W06R05ContractError("generation option key 不属于 choice")
        if any(item.choice == choice for item in self._uses):
            raise W06R05ContractError("同一 generation choice 不得重复采用")
        option = selected[0]
        candidate = self.view.candidate_by_id[option.target_proposition]
        use_key = LosslessIntegerKey((
            W06_R05_RUNTIME_NAMESPACE,
            720,
            len(self._uses) + 1,
            *pack_key(choice.stable_key()),
            *pack_key(option.stable_key()),
        ))
        committed = self.view.consume_candidate(candidate, use_key)
        use = W06R05GenerationUse(
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
        """只校验来源 surface 与端点 span 仍一致，不解析新语义。"""
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
            self, use: W06R05GenerationUse,
            ) -> W06R05GenerationOutcome:
        """独立重验 authorization、pair/channel、surface 和查询结论。"""
        if use not in self._uses:
            raise W06R05ContractError("generation Use 不属于当前 runtime")
        option = use.option
        request = use.choice.request
        candidate = self.view.candidate_by_id.get(option.target_proposition)
        authorization = (
            None if candidate is None else self.view.authorization_key(candidate))
        authorization_current = authorization == option.authorization_key
        pair_channel_preserved = bool(
            candidate is not None
            and candidate.relation_family == option.channel == request.channel
            and candidate.directionality == option.directionality
            and candidate_role_fillers(candidate) == option.role_fillers
            and tuple(sorted(
                candidate_endpoints(candidate), key=ObjectIdentity.stable_key
            )) == option.pair
            and request.target_proposition == option.target_proposition
            and request.directionality == option.directionality
            and request.role_fillers == option.role_fillers
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
        query_status = W06_R05_UNKNOWN
        recovered = False
        if candidate is not None and self.view.protocol.postcheck_connected:
            resolution = self.understanding.preview(query_for_candidate(
                candidate,
                request_key=LosslessIntegerKey((
                    W06_R05_RUNTIME_NAMESPACE,
                    730,
                    len(self._outcomes) + 1,
                    *pack_key(use.ref.use_key.components),
                )),
            ))
            query_status = resolution.status
            recovered = (
                resolution.status == W06_R05_SUPPORTED
                and option.target_proposition in resolution.propositions
            )
        verdict = (
            W06_R05_OUTCOME_SUPPORT
            if all((
                authorization_current,
                pair_channel_preserved,
                source_scope_preserved,
                surface_valid,
                query_status == W06_R05_SUPPORTED,
                recovered,
            ))
            else W06_R05_OUTCOME_REFUTE
        )
        outcome_key = LosslessIntegerKey((
            W06_R05_RUNTIME_NAMESPACE,
            740,
            len(self._outcomes) + 1,
            *pack_key(use.ref.use_key.components),
        ))
        outcome = W06R05GenerationOutcome(
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
            pair_channel_preserved,
            source_scope_preserved,
            surface_valid,
            query_status,
            recovered,
        )
        self._outcomes.append(outcome)
        return outcome

    def state_key(self) -> tuple:
        """返回 choices、Uses 和 outcomes 的稳定状态。"""
        return (
            tuple(item.stable_key() for item in self._choices),
            tuple(item.ref.stable_key() for item in self._uses),
            tuple(item.ref.stable_key() for item in self._outcomes),
        )


def generation_request_for_candidate(
        candidate: W06RelationCandidate,
        *, request_key: LosslessIntegerKey,
        constraints: GenerationExpressionConstraints,
        ) -> W06R05GenerationRequest:
    """从 direct typed target 建立不含 expected surface/label 的请求。"""
    if (not isinstance(candidate, W06RelationCandidate)
            or candidate.substage_key != W06_R05_SUBSTAGE):
        raise W06R05ContractError("generation candidate 不属于 R05")
    return W06R05GenerationRequest(
        request_key,
        candidate.proposition.proposition,
        candidate.relation_family,
        candidate.directionality,
        candidate_role_fillers(candidate),
        candidate.proposition.context,
        candidate.source_ref,
        candidate.uncertainty_units,
        constraints,
    )


__all__ = [
    "W06R05GenerationRuntime",
    "W06_R05_POSTCHECK_BUDGET",
    "generation_request_for_candidate",
    "query_for_candidate",
]
