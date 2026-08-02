"""W06-R04 active 部分整体结构的 Generation consumer 与 postcheck。"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.identity import ObjectIdentity
from pure_integer_ai.cognition.shared.mereology_relation import MereologyBudget
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.experiments.mereology_relation_runtime import (
    MereologyEndpointResolver,
)
from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    GenerationChoiceOutcomeRef,
    GenerationChoiceUseRef,
    GenerationExpressionConstraints,
    LosslessIntegerKey,
)
from pure_integer_ai.experiments.ph2_w06_adapter import W06RelationCandidate
from pure_integer_ai.experiments.ph2_w06_r04_contract import (
    W06R04ContractError,
    W06R04GenerationChoice,
    W06R04GenerationOption,
    W06R04GenerationOutcome,
    W06R04GenerationRequest,
    W06R04GenerationUse,
    W06R04MereologyQuery,
    W06_R04_GENERATION_READY,
    W06_R04_GENERATION_REJECTED,
    W06_R04_GENERATION_UNKNOWN,
    W06_R04_OUTCOME_REFUTE,
    W06_R04_OUTCOME_SUPPORT,
    W06_R04_SUBSTAGE,
    W06_R04_SUPPORTED,
    W06_R04_UNKNOWN,
    pack_key,
)
from pure_integer_ai.experiments.ph2_w06_r04_query import W06R04QueryRuntime
from pure_integer_ai.experiments.ph2_w06_r04_shared import (
    W06R04View,
    W06_R04_RUNTIME_NAMESPACE,
    candidate_construction,
    candidate_endpoints,
    candidate_role_fillers,
    w06_r04_language_branch,
)


W06_R04_POSTCHECK_BUDGET = MereologyBudget(32, 128, 512, 32)
_GENERATION_DIMENSION = LosslessIntegerKey((W06_R04_RUNTIME_NAMESPACE, 803))
_POSTCHECK_VERIFIER = LosslessIntegerKey((W06_R04_RUNTIME_NAMESPACE, 804))
_RESULT_KEYS = {
    W06_R04_OUTCOME_SUPPORT: LosslessIntegerKey((
        W06_R04_RUNTIME_NAMESPACE, 805)),
    W06_R04_OUTCOME_REFUTE: LosslessIntegerKey((
        W06_R04_RUNTIME_NAMESPACE, 806)),
}


def query_for_candidate(
        candidate: W06RelationCandidate,
        *, request_key: LosslessIntegerKey,
        endpoint_resolver: MereologyEndpointResolver,
        budget: MereologyBudget = W06_R04_POSTCHECK_BUDGET,
        ) -> W06R04MereologyQuery:
    """从 typed Role 构造不含表层预期的部分整体查询。"""
    if (not isinstance(candidate, W06RelationCandidate)
            or candidate.substage_key != W06_R04_SUBSTAGE):
        raise W06R04ContractError("mereology query candidate 不属于 R04")
    if not isinstance(endpoint_resolver, MereologyEndpointResolver):
        raise TypeError("mereology query endpoint resolver 类型非法")
    part, whole = tuple(
        endpoint_resolver.resolve(item)
        for item in candidate_endpoints(candidate)
    )
    return W06R04MereologyQuery(
        request_key,
        candidate.relation_family,
        part,
        whole,
        budget,
    )


class W06R04GenerationRuntime:
    """从 direct active mereology fact 形成 option、exact Use 和 postcheck。"""

    def __init__(
            self,
            view: W06R04View,
            understanding: W06R04QueryRuntime,
            ) -> None:
        """绑定共享 view 和 Understanding preview 入口。"""
        if not isinstance(view, W06R04View):
            raise TypeError("R04 Generation view 类型非法")
        if (not isinstance(understanding, W06R04QueryRuntime)
                or understanding.view is not view
                or understanding.consumer != "UNDERSTANDING"):
            raise TypeError("R04 Generation 必须共享 Understanding view")
        self.view = view
        self.understanding = understanding
        self._request_keys: set[LosslessIntegerKey] = set()
        self._choices: list[W06R04GenerationChoice] = []
        self._uses: list[W06R04GenerationUse] = []
        self._outcomes: list[W06R04GenerationOutcome] = []

    @property
    def choices(self) -> tuple[W06R04GenerationChoice, ...]:
        """返回 generation choice 历史。"""
        return tuple(self._choices)

    @property
    def uses(self) -> tuple[W06R04GenerationUse, ...]:
        """返回 generation Use 历史。"""
        return tuple(self._uses)

    @property
    def outcomes(self) -> tuple[W06R04GenerationOutcome, ...]:
        """返回 generation postcheck 历史。"""
        return tuple(self._outcomes)

    @staticmethod
    def _allowed(
            request: W06R04GenerationRequest,
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
            self, request: W06R04GenerationRequest,
            ) -> tuple[W06R04GenerationOption, ...]:
        """按 current active direct fact 形成唯一 source-grounded option。"""
        if not self.view.protocol.generation_ready():
            return ()
        candidate = self.view.candidate_by_id.get(request.target_proposition)
        if candidate is None or candidate.relation_family != request.relation_family:
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
        branch = w06_r04_language_branch(candidate)
        if not self._allowed(request, candidate, construction, branch):
            return ()
        return (W06R04GenerationOption(
            candidate.surface,
            construction,
            candidate.proposition.proposition,
            candidate.relation_family,
            candidate.directionality,
            candidate_role_fillers(candidate),
            candidate.proposition.context,
            candidate.source_ref,
            candidate.uncertainty_units,
            branch,
            authorization,
        ),)

    def choose(
            self, request: W06R04GenerationRequest,
            ) -> W06R04GenerationChoice:
        """形成 source-grounded option，不接受 expected string 或 label。"""
        if not isinstance(request, W06R04GenerationRequest):
            raise TypeError("R04 generation request 类型非法")
        if request.request_key in self._request_keys:
            raise W06R04ContractError("重复 R04 generation request key")
        options = self._options(request)
        if options:
            status = W06_R04_GENERATION_READY
        elif (not self.view.protocol.mereology_bridge_connected
              or not self.view.protocol.generation_connected):
            status = W06_R04_GENERATION_UNKNOWN
        elif (not self.view.protocol.direction_connected
              or not self.view.protocol.source_scope_connected):
            status = W06_R04_GENERATION_REJECTED
        else:
            status = W06_R04_GENERATION_UNKNOWN
        choice = W06R04GenerationChoice(
            request,
            status,
            options,
            LosslessIntegerKey((
                W06_R04_RUNTIME_NAMESPACE,
                710,
                *self.view.protocol.stable_key(),
                int(status == W06_R04_GENERATION_READY),
            )),
        )
        self._request_keys.add(request.request_key)
        self._choices.append(choice)
        return choice

    def adopt(
            self,
            choice: W06R04GenerationChoice,
            option_key: tuple[int, ...],
            ) -> W06R04GenerationUse:
        """采用 option，并经 MereologyRuntime 提交 exact direct fact Use。"""
        if choice not in self._choices:
            raise W06R04ContractError("generation choice 不属于当前 runtime")
        if choice.status != W06_R04_GENERATION_READY:
            raise W06R04ContractError("非 READY generation choice 不得采用")
        selected = tuple(
            item for item in choice.options if item.stable_key() == option_key)
        if len(selected) != 1:
            raise W06R04ContractError("generation option key 不属于 choice")
        if any(item.choice == choice for item in self._uses):
            raise W06R04ContractError("同一 generation choice 不得重复采用")
        option = selected[0]
        candidate = self.view.candidate_by_id[option.target_proposition]
        use_key = LosslessIntegerKey((
            W06_R04_RUNTIME_NAMESPACE,
            720,
            len(self._uses) + 1,
            *pack_key(choice.stable_key()),
            *pack_key(option.stable_key()),
        ))
        query = query_for_candidate(
            candidate,
            request_key=LosslessIntegerKey((
                W06_R04_RUNTIME_NAMESPACE, 721, len(self._uses) + 1)),
            endpoint_resolver=self.view.endpoint_resolver,
        )
        committed = self.view.commit(query, use_key)
        use = W06R04GenerationUse(
            choice,
            option,
            committed.uses,
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
            self, use: W06R04GenerationUse,
            ) -> W06R04GenerationOutcome:
        """独立重验 authorization、typed 结构、surface 和 R04 查询结论。"""
        if use not in self._uses:
            raise W06R04ContractError("generation Use 不属于当前 runtime")
        option = use.option
        request = use.choice.request
        candidate = self.view.candidate_by_id.get(option.target_proposition)
        authorization = (
            None if candidate is None else self.view.authorization_key(candidate))
        authorization_current = authorization == option.authorization_key
        relation_structure_preserved = bool(
            candidate is not None
            and candidate.relation_family == option.relation_family
            and candidate.directionality == option.directionality
            and candidate_role_fillers(candidate) == option.role_fillers
            and request.target_proposition == option.target_proposition
            and request.relation_family == option.relation_family
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
        query_status = W06_R04_UNKNOWN
        recovered = False
        if candidate is not None and self.view.protocol.postcheck_connected:
            resolution = self.understanding.preview(query_for_candidate(
                candidate,
                request_key=LosslessIntegerKey((
                    W06_R04_RUNTIME_NAMESPACE,
                    730,
                    len(self._outcomes) + 1,
                    *pack_key(use.ref.use_key.components),
                )),
                endpoint_resolver=self.view.endpoint_resolver,
            ))
            query_status = resolution.status
            recovered = (
                resolution.status == W06_R04_SUPPORTED
                and option.target_proposition in resolution.propositions
            )
        verdict = (
            W06_R04_OUTCOME_SUPPORT
            if all((
                authorization_current,
                relation_structure_preserved,
                source_scope_preserved,
                surface_valid,
                query_status == W06_R04_SUPPORTED,
                recovered,
            ))
            else W06_R04_OUTCOME_REFUTE
        )
        outcome_key = LosslessIntegerKey((
            W06_R04_RUNTIME_NAMESPACE,
            740,
            len(self._outcomes) + 1,
            *pack_key(use.ref.use_key.components),
        ))
        outcome = W06R04GenerationOutcome(
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
            relation_structure_preserved,
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
        ) -> W06R04GenerationRequest:
    """从 direct typed target 建立不含 expected surface/label 的请求。"""
    if (not isinstance(candidate, W06RelationCandidate)
            or candidate.substage_key != W06_R04_SUBSTAGE):
        raise W06R04ContractError("generation candidate 不属于 R04")
    return W06R04GenerationRequest(
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
    "W06R04GenerationRuntime",
    "W06_R04_POSTCHECK_BUDGET",
    "generation_request_for_candidate",
    "query_for_candidate",
]
