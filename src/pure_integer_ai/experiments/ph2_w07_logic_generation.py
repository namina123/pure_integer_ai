"""W-07 logic tree Generation choice、exact Use 与独立 postcheck。"""
from __future__ import annotations

from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    GenerationChoiceOutcomeRef,
    GenerationChoiceUseRef,
    GenerationExpressionConstraints,
    LosslessIntegerKey,
)
from pure_integer_ai.experiments.ph2_w07_adapter import W07LogicProposal
from pure_integer_ai.experiments.ph2_w07_logic_contract import (
    W07LogicContractError,
    W07LogicGenerationChoice,
    W07LogicGenerationOption,
    W07LogicGenerationOutcome,
    W07LogicGenerationRequest,
    W07LogicGenerationUse,
    W07LogicRequest,
    pack_key,
)
from pure_integer_ai.experiments.ph2_w07_logic_shared import (
    W07LogicView,
    W07_LOGIC_RUNTIME_NAMESPACE,
    role_tree_key,
    structure_tree_key,
    w07_logic_language_branch,
)


_DIMENSION = LosslessIntegerKey((W07_LOGIC_RUNTIME_NAMESPACE, 803))
_VERIFIER = LosslessIntegerKey((W07_LOGIC_RUNTIME_NAMESPACE, 804))
_RESULTS = {
    "SUPPORT": LosslessIntegerKey((W07_LOGIC_RUNTIME_NAMESPACE, 805)),
    "REFUTE": LosslessIntegerKey((W07_LOGIC_RUNTIME_NAMESPACE, 806)),
}


def generation_request_for_proposal(
        proposal: W07LogicProposal,
        *,
        request_key: LosslessIntegerKey,
        logic_request_key: LosslessIntegerKey,
        constraints: GenerationExpressionConstraints,
        ) -> W07LogicGenerationRequest:
    """从 target/source/scope 构造请求，不注入 expected surface 或 label。"""
    if not isinstance(proposal, W07LogicProposal):
        raise TypeError("W-07 generation proposal 类型非法")
    return W07LogicGenerationRequest(
        request_key,
        W07LogicRequest(
            logic_request_key,
            proposal.observation.substage,
            proposal.bound_root.template,
            proposal.source_binding.source_ref,
            proposal.request_scope,
        ),
        constraints,
    )


class W07LogicGenerationRuntime:
    """只从 current learned operator 和真实 S-04 结果生成结构 option。"""

    def __init__(self, view: W07LogicView) -> None:
        if not isinstance(view, W07LogicView):
            raise TypeError("W-07 generation view 类型非法")
        self.view = view
        self._request_keys: set[LosslessIntegerKey] = set()
        self._choices: list[W07LogicGenerationChoice] = []
        self._uses: list[W07LogicGenerationUse] = []
        self._outcomes: list[W07LogicGenerationOutcome] = []

    @property
    def choices(self) -> tuple[W07LogicGenerationChoice, ...]:
        return tuple(self._choices)

    @property
    def uses(self) -> tuple[W07LogicGenerationUse, ...]:
        return tuple(self._uses)

    @property
    def outcomes(self) -> tuple[W07LogicGenerationOutcome, ...]:
        return tuple(self._outcomes)

    def _option(
            self, request: W07LogicGenerationRequest,
            ) -> W07LogicGenerationOption | None:
        if not self.view.protocol.connected("GENERATION"):
            return None
        execution = self.view.execute(request.logic_request)
        proposal = self.view.proposal_for(request.logic_request)
        if execution is None or proposal is None:
            return None
        raw = proposal.observation.typed_payload.to_value()
        surface = raw.get("surface")
        if not isinstance(surface, str) or not surface:
            raise W07LogicContractError("W-07 proposal surface 非法")
        constraints = request.constraints
        branch = w07_logic_language_branch(proposal)
        if (len(surface) > constraints.max_output_units
                or constraints.require_explicit_source
                or constraints.target_language != branch
                or (constraints.allowed_lexical_branches
                    and branch not in constraints.allowed_lexical_branches)
                or (constraints.allowed_structure_families
                    and not set(execution.executed_structures).intersection(
                        constraints.allowed_structure_families))):
            return None
        return W07LogicGenerationOption(
            surface,
            proposal.bound_root.template,
            proposal.operator_families,
            structure_tree_key(proposal.bound_root),
            role_tree_key(proposal.bound_root),
            execution.evaluation.source,
            execution.evaluation.scope,
            execution.evaluation.state.stable_key(),
            execution.operator_premise_keys,
            execution.content_premise_keys,
            branch,
        )

    def choose(
            self, request: W07LogicGenerationRequest,
            ) -> W07LogicGenerationChoice:
        if not isinstance(request, W07LogicGenerationRequest):
            raise TypeError("W-07 generation request 类型非法")
        if request.request_key in self._request_keys:
            raise W07LogicContractError("重复 W-07 generation request key")
        option = self._option(request)
        if option is not None:
            status = "READY"
            options = (option,)
        elif not self.view.protocol.generation_connected:
            status = "UNKNOWN"
            options = ()
        else:
            status = "REJECTED"
            options = ()
        choice = W07LogicGenerationChoice(
            request,
            status,
            options,
            LosslessIntegerKey((
                W07_LOGIC_RUNTIME_NAMESPACE,
                710,
                1 + ("READY", "UNKNOWN", "REJECTED").index(status),
                *self.view.protocol.stable_key(),
            )),
        )
        self._request_keys.add(request.request_key)
        self._choices.append(choice)
        return choice

    def adopt(
            self,
            choice: W07LogicGenerationChoice,
            option_key: tuple[int, ...],
            ) -> W07LogicGenerationUse:
        if choice not in self._choices:
            raise W07LogicContractError("generation choice 不属于当前 runtime")
        if choice.status != "READY":
            raise W07LogicContractError("非 READY generation 不得采用")
        selected = tuple(
            item for item in choice.options if item.stable_key() == option_key)
        if len(selected) != 1:
            raise W07LogicContractError("generation option key 不属于 choice")
        if any(item.choice == choice for item in self._uses):
            raise W07LogicContractError("同一 generation choice 不得重复采用")
        option = selected[0]
        execution = self.view.execute(choice.request.logic_request)
        if (execution is None
                or execution.operator_premise_keys
                != option.operator_premise_keys
                or execution.content_premise_keys
                != option.content_premise_keys):
            raise W07LogicContractError("generation adopt 时 premise 已失效")
        use_key = LosslessIntegerKey((
            W07_LOGIC_RUNTIME_NAMESPACE,
            720,
            len(self._uses) + 1,
            *pack_key(choice.stable_key()),
            *pack_key(option.stable_key()),
        ))
        use = W07LogicGenerationUse(
            choice,
            option,
            execution,
            GenerationChoiceUseRef(
                "CORE_USE",
                use_key,
                LosslessIntegerKey(option.stable_key()),
                option.scope,
            ),
        )
        self._uses.append(use)
        return use

    def verify(
            self, use: W07LogicGenerationUse,
            ) -> W07LogicGenerationOutcome:
        if use not in self._uses:
            raise W07LogicContractError("generation Use 不属于当前 runtime")
        option = use.option
        request = use.choice.request.logic_request
        current = self.view.execute(request)
        proposal = self.view.proposal_for(request)
        adoption_current = bool(
            current is not None
            and current.operator_premise_keys == option.operator_premise_keys)
        structure_preserved = bool(
            current is not None and proposal is not None
            and structure_tree_key(proposal.bound_root)
            == option.structure_tree_key)
        # family 与 tree identity 分开核验，便于正交消融。
        if proposal is not None:
            structure_preserved = (
                structure_preserved
                and proposal.operator_families == option.operator_families)
        role_order_preserved = bool(
            proposal is not None
            and role_tree_key(proposal.bound_root) == option.role_tree_key)
        state_preserved = bool(
            current is not None
            and current.evaluation.state.stable_key() == option.state_key
            and current.content_premise_keys == option.content_premise_keys)
        source_scope_preserved = bool(
            current is not None
            and current.evaluation.source == option.source
            and current.evaluation.scope == option.scope
            and request.source == option.source)
        raw_surface = (
            None if proposal is None
            else proposal.observation.typed_payload.to_value().get("surface"))
        surface_valid = bool(
            isinstance(raw_surface, str)
            and raw_surface == option.surface
            and len(option.surface)
            <= use.choice.request.constraints.max_output_units)
        recovered_target = bool(
            self.view.protocol.postcheck_connected
            and current is not None and proposal is not None
            and proposal.bound_root.template == option.target_proposition
            == request.target_proposition)
        checks = (
            adoption_current,
            structure_preserved,
            role_order_preserved,
            state_preserved,
            source_scope_preserved,
            surface_valid,
            recovered_target,
        )
        verdict = "SUPPORT" if all(checks) else "REFUTE"
        outcome_key = LosslessIntegerKey((
            W07_LOGIC_RUNTIME_NAMESPACE,
            740,
            len(self._outcomes) + 1,
            *pack_key(use.ref.use_key.components),
        ))
        outcome = W07LogicGenerationOutcome(
            use,
            verdict,
            GenerationChoiceOutcomeRef(
                outcome_key,
                use.ref.use_key,
                _DIMENSION,
                _VERIFIER,
                _RESULTS[verdict],
            ),
            *checks,
        )
        self._outcomes.append(outcome)
        return outcome

    def state_key(self) -> tuple:
        return (
            tuple(item.stable_key() for item in self._choices),
            tuple(item.ref.stable_key() for item in self._uses),
            tuple(item.ref.stable_key() for item in self._outcomes),
        )


__all__ = [
    "W07LogicGenerationRuntime",
    "generation_request_for_proposal",
]
