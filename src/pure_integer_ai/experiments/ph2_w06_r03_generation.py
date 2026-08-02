"""W06-R03 active PROPERTY 结构的 Generation consumer 与 postcheck。"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.identity import ObjectIdentity
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.cognition.shared.property_relation import PropertyQueryBudget
from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    GenerationChoiceOutcomeRef,
    GenerationChoiceUseRef,
    GenerationExpressionConstraints,
    LosslessIntegerKey,
)
from pure_integer_ai.experiments.ph2_w06_adapter import W06RelationCandidate
from pure_integer_ai.experiments.ph2_w06_r03_contract import (
    W06R03ContractError,
    W06R03GenerationChoice,
    W06R03GenerationOption,
    W06R03GenerationOutcome,
    W06R03GenerationRequest,
    W06R03GenerationUse,
    W06R03ReasoningRequest,
    W06_R03_GENERATION_READY,
    W06_R03_GENERATION_REJECTED,
    W06_R03_GENERATION_UNKNOWN,
    W06_R03_OUTCOME_REFUTE,
    W06_R03_OUTCOME_SUPPORT,
    W06_R03_REASONING_SUPPORTED,
    W06_R03_REASONING_UNRESOLVED,
    W06_R03_SUBSTAGE,
    pack_key,
)
from pure_integer_ai.experiments.ph2_w06_r03_reasoning import (
    W06R03ReasoningRuntime,
)
from pure_integer_ai.experiments.ph2_w06_r03_shared import (
    W06R03View,
    W06_R03_RUNTIME_NAMESPACE,
    candidate_claim,
    candidate_construction,
    candidate_role_fillers,
    w06_r03_language_branch,
)


W06_R03_POSTCHECK_BUDGET = PropertyQueryBudget(32, 32)
_GENERATION_DIMENSION = LosslessIntegerKey((W06_R03_RUNTIME_NAMESPACE, 803))
_POSTCHECK_VERIFIER = LosslessIntegerKey((W06_R03_RUNTIME_NAMESPACE, 804))
_RESULT_KEYS = {
    W06_R03_OUTCOME_SUPPORT: LosslessIntegerKey((
        W06_R03_RUNTIME_NAMESPACE, 805)),
    W06_R03_OUTCOME_REFUTE: LosslessIntegerKey((
        W06_R03_RUNTIME_NAMESPACE, 806)),
}


class W06R03GenerationRuntime:
    """从 direct active PROPERTY fact 形成 option、exact Use 和 postcheck。"""

    def __init__(
            self,
            view: W06R03View,
            reasoning: W06R03ReasoningRuntime,
            ) -> None:
        """绑定共享 view 和 Reasoning postcheck runtime。"""
        if not isinstance(view, W06R03View):
            raise TypeError("R03 Generation view 类型非法")
        if (not isinstance(reasoning, W06R03ReasoningRuntime)
                or reasoning.view is not view):
            raise TypeError("R03 Generation 必须共享 Reasoning view")
        self.view = view
        self.reasoning = reasoning
        self._request_keys: set[LosslessIntegerKey] = set()
        self._choices: list[W06R03GenerationChoice] = []
        self._uses: list[W06R03GenerationUse] = []
        self._outcomes: list[W06R03GenerationOutcome] = []

    @property
    def choices(self) -> tuple[W06R03GenerationChoice, ...]:
        """返回本 runtime 的 choice 历史。"""
        return tuple(self._choices)

    @property
    def uses(self) -> tuple[W06R03GenerationUse, ...]:
        """返回已采用的 generation option Use。"""
        return tuple(self._uses)

    @property
    def outcomes(self) -> tuple[W06R03GenerationOutcome, ...]:
        """返回 append-only 的独立 postcheck 结果。"""
        return tuple(self._outcomes)

    @staticmethod
    def _allowed(
            request: W06R03GenerationRequest,
            candidate: W06RelationCandidate,
            construction: ObjectIdentity,
            branch: ObjectIdentity,
            ) -> bool:
        """按预算、语言分支和结构白名单过滤 option。"""
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
            self,
            request: W06R03GenerationRequest,
            ) -> tuple[W06R03GenerationOption, ...]:
        """只从当前 active direct PROPERTY fact 构造来源化 option。"""
        if not self.view.protocol.generation_ready():
            return ()
        candidate = self.view.candidate_by_id.get(request.target_proposition)
        if candidate is None or candidate.relation_family != "PROPERTY":
            return ()
        if (candidate.directionality != request.directionality
                or candidate_role_fillers(candidate) != request.role_fillers
                or candidate.proposition.context != request.context
                or candidate.source_ref != request.source
                or candidate.uncertainty_units != request.uncertainty_units
                or self.view.claim_for(candidate) != request.claim):
            return ()
        authorization = self.view.authorization_key(candidate)
        if authorization is None:
            return ()
        construction = candidate_construction(candidate)
        branch = w06_r03_language_branch(candidate)
        if not self._allowed(request, candidate, construction, branch):
            return ()
        return (W06R03GenerationOption(
            candidate.surface,
            construction,
            candidate.proposition.proposition,
            request.claim,
            candidate.directionality,
            candidate_role_fillers(candidate),
            candidate.proposition.context,
            candidate.source_ref,
            candidate.uncertainty_units,
            branch,
            authorization,
        ),)

    def choose(
            self,
            request: W06R03GenerationRequest,
            ) -> W06R03GenerationChoice:
        """形成 source-grounded option，不接受 expected string 或 label。"""
        if not isinstance(request, W06R03GenerationRequest):
            raise TypeError("R03 generation request 类型非法")
        if request.request_key in self._request_keys:
            raise W06R03ContractError("重复 R03 generation request key")
        options = self._options(request)
        if options:
            status = W06_R03_GENERATION_READY
        elif (not self.view.protocol.property_bridge_connected
              or not self.view.protocol.generation_connected):
            status = W06_R03_GENERATION_UNKNOWN
        elif (not self.view.protocol.role_structure_connected
              or not self.view.protocol.intensity_connected
              or not self.view.protocol.source_scope_connected):
            status = W06_R03_GENERATION_REJECTED
        else:
            status = W06_R03_GENERATION_UNKNOWN
        choice = W06R03GenerationChoice(
            request,
            status,
            options,
            LosslessIntegerKey((
                W06_R03_RUNTIME_NAMESPACE,
                710,
                *self.view.protocol.stable_key(),
                int(status == W06_R03_GENERATION_READY),
            )),
        )
        self._request_keys.add(request.request_key)
        self._choices.append(choice)
        return choice

    def adopt(
            self,
            choice: W06R03GenerationChoice,
            option_key: tuple[int, ...],
            ) -> W06R03GenerationUse:
        """采用 option，并经 PropertyRelationRuntime 提交 exact direct fact Use。"""
        if choice not in self._choices:
            raise W06R03ContractError("generation choice 不属于当前 runtime")
        if choice.status != W06_R03_GENERATION_READY:
            raise W06R03ContractError("非 READY generation choice 不得采用")
        selected = tuple(
            item for item in choice.options if item.stable_key() == option_key)
        if len(selected) != 1:
            raise W06R03ContractError("generation option key 不属于 choice")
        if any(item.choice == choice for item in self._uses):
            raise W06R03ContractError("同一 generation choice 不得重复采用")
        option = selected[0]
        use_key = LosslessIntegerKey((
            W06_R03_RUNTIME_NAMESPACE,
            720,
            len(self._uses) + 1,
            *pack_key(choice.stable_key()),
            *pack_key(option.stable_key()),
        ))
        result = self.view.select(
            self.view.exact_pattern(option.claim),
            W06_R03_POSTCHECK_BUDGET,
            use_key.components,
        )
        adopted = result.selection.selected()
        if adopted is None or adopted.claim != option.claim:
            raise W06R03ContractError("generation adoption 未恢复 PROPERTY 目标")
        use = W06R03GenerationUse(
            choice,
            option,
            result.uses,
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
        """独立核对来源 span 与输出 surface，不复用 option builder。"""
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
            self,
            use: W06R03GenerationUse,
            ) -> W06R03GenerationOutcome:
        """独立重验 authorization、六维结构、surface 和 direct fact。"""
        if use not in self._uses:
            raise W06R03ContractError("generation Use 不属于当前 runtime")
        option = use.option
        request = use.choice.request
        candidate = self.view.candidate_by_id.get(option.target_proposition)
        authorization = (
            None if candidate is None else self.view.authorization_key(candidate))
        authorization_current = authorization == option.authorization_key
        relation_structure_preserved = bool(
            candidate is not None
            and candidate.directionality == option.directionality
            and candidate_role_fillers(candidate) == option.role_fillers
            and self.view.claim_for(candidate) == option.claim
            and request.target_proposition == option.target_proposition
            and request.claim == option.claim
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
        query_status = W06_R03_REASONING_UNRESOLVED
        recovered = False
        if candidate is not None and self.view.protocol.postcheck_connected:
            resolution = self.reasoning.preview(W06R03ReasoningRequest(
                LosslessIntegerKey((
                    W06_R03_RUNTIME_NAMESPACE,
                    730,
                    len(self._outcomes) + 1,
                    *pack_key(use.ref.use_key.components),
                )),
                option.claim,
                W06_R03_POSTCHECK_BUDGET,
            ))
            query_status = resolution.status
            recovered = (
                resolution.status == W06_R03_REASONING_SUPPORTED
                and option.target_proposition in resolution.propositions
            )
        verdict = (
            W06_R03_OUTCOME_SUPPORT
            if all((
                authorization_current,
                relation_structure_preserved,
                source_scope_preserved,
                surface_valid,
                query_status == W06_R03_REASONING_SUPPORTED,
                recovered,
            ))
            else W06_R03_OUTCOME_REFUTE
        )
        outcome_key = LosslessIntegerKey((
            W06_R03_RUNTIME_NAMESPACE,
            740,
            len(self._outcomes) + 1,
            *pack_key(use.ref.use_key.components),
        ))
        outcome = W06R03GenerationOutcome(
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
        """返回 choice、Use 与 postcheck outcome 的稳定状态。"""
        return (
            tuple(item.stable_key() for item in self._choices),
            tuple(item.ref.stable_key() for item in self._uses),
            tuple(item.ref.stable_key() for item in self._outcomes),
        )


def generation_request_for_candidate(
        candidate: W06RelationCandidate,
        *,
        claim,
        request_key: LosslessIntegerKey,
        constraints: GenerationExpressionConstraints,
        ) -> W06R03GenerationRequest:
    """从 typed target 建立不含 expected surface 或 evaluator label 的请求。"""
    if (not isinstance(candidate, W06RelationCandidate)
            or candidate.substage_key != W06_R03_SUBSTAGE):
        raise W06R03ContractError("generation candidate 不属于 R03")
    return W06R03GenerationRequest(
        request_key,
        candidate.proposition.proposition,
        claim,
        candidate.directionality,
        candidate_role_fillers(candidate),
        candidate.proposition.context,
        candidate.source_ref,
        candidate.uncertainty_units,
        constraints,
    )


__all__ = [
    "W06R03GenerationRuntime",
    "W06_R03_POSTCHECK_BUDGET",
    "generation_request_for_candidate",
]
