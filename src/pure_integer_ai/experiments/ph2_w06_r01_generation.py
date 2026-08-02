"""W06-R01 active relation structure 的 Generation consumer 与 postcheck。"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.alias_resolution import (
    AliasRouteSearchBudget,
)
from pure_integer_ai.cognition.shared.identity import ObjectIdentity
from pure_integer_ai.cognition.shared.scope_identity import document_scope
from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    GenerationChoiceOutcomeRef,
    GenerationChoiceUseRef,
    GenerationExpressionConstraints,
    LosslessIntegerKey,
)
from pure_integer_ai.experiments.ph2_w06_adapter import W06RelationCandidate
from pure_integer_ai.experiments.ph2_w06_r01_contract import (
    W06R01ContractError,
    W06R01GenerationChoice,
    W06R01GenerationOption,
    W06R01GenerationOutcome,
    W06R01GenerationRequest,
    W06R01GenerationUse,
    W06R01UnderstandingRequest,
    W06_R01_GENERATION_READY,
    W06_R01_GENERATION_REJECTED,
    W06_R01_GENERATION_UNKNOWN,
    W06_R01_OUTCOME_REFUTE,
    W06_R01_OUTCOME_SUPPORT,
    W06_R01_SUBSTAGE,
    W06_R01_UNDERSTANDING_UNIQUE,
    W06_R01_UNDERSTANDING_UNKNOWN,
)
from pure_integer_ai.experiments.ph2_w06_r01_shared import (
    W06R01View,
    W06_R01_RUNTIME_NAMESPACE,
    candidate_construction,
    candidate_endpoints,
    candidate_role_fillers,
    pack_key,
    w06_r01_language_branch,
)
from pure_integer_ai.experiments.ph2_w06_r01_understanding import (
    W06R01UnderstandingRuntime,
)


_GENERATION_DIMENSION = LosslessIntegerKey((W06_R01_RUNTIME_NAMESPACE, 803))
_POSTCHECK_VERIFIER = LosslessIntegerKey((W06_R01_RUNTIME_NAMESPACE, 804))
_RESULT_KEYS = {
    W06_R01_OUTCOME_SUPPORT: LosslessIntegerKey((
        W06_R01_RUNTIME_NAMESPACE, 805)),
    W06_R01_OUTCOME_REFUTE: LosslessIntegerKey((
        W06_R01_RUNTIME_NAMESPACE, 806)),
}


class W06R01GenerationRuntime:
    """从 exact active relation fact 形成来源化 option、Use 和分维 outcome。"""

    def __init__(
            self,
            view: W06R01View,
            understanding: W06R01UnderstandingRuntime,
            ) -> None:
        if not isinstance(view, W06R01View):
            raise TypeError("R01 Generation view 类型非法")
        if (not isinstance(understanding, W06R01UnderstandingRuntime)
                or understanding.view is not view):
            raise TypeError("R01 Generation 必须共享 Understanding view")
        self.view = view
        self.understanding = understanding
        self._request_keys: set[LosslessIntegerKey] = set()
        self._choices: list[W06R01GenerationChoice] = []
        self._uses: list[W06R01GenerationUse] = []
        self._outcomes: list[W06R01GenerationOutcome] = []

    @property
    def choices(self) -> tuple[W06R01GenerationChoice, ...]:
        """返回全部生成 choice。"""
        return tuple(self._choices)

    @property
    def uses(self) -> tuple[W06R01GenerationUse, ...]:
        """返回已采用 option 的 exact Use。"""
        return tuple(self._uses)

    @property
    def outcomes(self) -> tuple[W06R01GenerationOutcome, ...]:
        """返回独立 postcheck 的 append-only 结果。"""
        return tuple(self._outcomes)

    def _authorization_key(
            self,
            candidate: W06RelationCandidate,
            ) -> LosslessIntegerKey | None:
        """从 current active fact 的 Hypothesis/Evidence/H-04 构造授权键。"""
        snapshot = self.view.learning.snapshot_for(
            candidate.proposition.proposition)
        fact = snapshot.active_fact
        if fact is None:
            return None
        values = [
            W06_R01_RUNTIME_NAMESPACE,
            700,
            *pack_key(fact.proposition.proposition.stable_key()),
            *pack_key(fact.hypothesis.stable_key()),
            len(fact.evidence_keys),
        ]
        for item in fact.evidence_keys:
            values.extend(pack_key(item))
        values.extend(pack_key(fact.decision_key))
        return LosslessIntegerKey(tuple(values))

    @staticmethod
    def _allowed(
            request: W06R01GenerationRequest,
            candidate: W06RelationCandidate,
            construction: ObjectIdentity,
            branch: ObjectIdentity,
            ) -> bool:
        """按预算、语言和结构白名单过滤 source-grounded option。"""
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
            request: W06R01GenerationRequest,
            ) -> tuple[W06R01GenerationOption, ...]:
        """只从 exact active relation fact 形成 option，不读 evaluator surface。"""
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
        authorization = self._authorization_key(candidate)
        if authorization is None:
            return ()
        construction = candidate_construction(candidate)
        branch = w06_r01_language_branch(candidate)
        if not self._allowed(request, candidate, construction, branch):
            return ()
        return (W06R01GenerationOption(
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
            self,
            request: W06R01GenerationRequest,
            ) -> W06R01GenerationChoice:
        """形成全部合法来源化 relation surface option，不接受 expected string。"""
        if not isinstance(request, W06R01GenerationRequest):
            raise TypeError("R01 generation request 类型非法")
        if request.request_key in self._request_keys:
            raise W06R01ContractError("重复 R01 generation request key")
        options = self._options(request)
        if options:
            status = W06_R01_GENERATION_READY
        elif not self.view.protocol.alias_refers_bridge_connected or (
                not self.view.protocol.generation_connected):
            status = W06_R01_GENERATION_UNKNOWN
        elif not self.view.protocol.direction_connected or (
                not self.view.protocol.source_scope_connected):
            status = W06_R01_GENERATION_REJECTED
        else:
            status = W06_R01_GENERATION_UNKNOWN
        choice = W06R01GenerationChoice(
            request,
            status,
            options,
            LosslessIntegerKey((
                W06_R01_RUNTIME_NAMESPACE,
                710,
                *self.view.protocol.stable_key(),
                1 if status == W06_R01_GENERATION_READY else 0,
            )),
        )
        self._request_keys.add(request.request_key)
        self._choices.append(choice)
        return choice

    def adopt(
            self,
            choice: W06R01GenerationChoice,
            option_key: tuple[int, ...],
            ) -> W06R01GenerationUse:
        """采用一个 option 并把 relation authorization 记为 exact Use。"""
        if choice not in self._choices:
            raise W06R01ContractError("generation choice 不属于当前 runtime")
        if choice.status != W06_R01_GENERATION_READY:
            raise W06R01ContractError("非 READY generation choice 不得采用")
        selected = tuple(
            item for item in choice.options if item.stable_key() == option_key)
        if len(selected) != 1:
            raise W06R01ContractError("generation option key 不属于 choice")
        if any(item.choice == choice for item in self._uses):
            raise W06R01ContractError("同一 generation choice 不得重复采用")
        option = selected[0]
        use_key = LosslessIntegerKey((
            W06_R01_RUNTIME_NAMESPACE,
            720,
            len(self._uses) + 1,
            *pack_key(choice.stable_key()),
            *pack_key(option.stable_key()),
        ))
        assert self.view.learning.closure is not None
        relation_use = self.view.learning.closure.consume(
            option.target_proposition,
            use_key=use_key.components,
        )
        use = W06R01GenerationUse(
            choice,
            option,
            relation_use,
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
            use: W06R01GenerationUse,
            ) -> W06R01GenerationOutcome:
        """独立重验 current authorization、typed 结构、surface span 和指向。"""
        if use not in self._uses:
            raise W06R01ContractError("generation Use 不属于当前 runtime")
        option = use.option
        request = use.choice.request
        candidate = self.view.candidate_by_id.get(option.target_proposition)
        authorization = (
            None if candidate is None else self._authorization_key(candidate))
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
            and candidate.proposition.context
            == option.context == request.context
            and option.uncertainty_units == request.uncertainty_units
        )
        surface_valid = bool(
            candidate is not None
            and self._surface_structure_valid(candidate, option.surface))
        understanding_status = W06_R01_UNDERSTANDING_UNKNOWN
        recovered = False
        if candidate is not None and self.view.protocol.postcheck_connected:
            origin, target = candidate_endpoints(candidate)
            resolution = self.understanding.preview(
                W06R01UnderstandingRequest(
                    LosslessIntegerKey((
                        W06_R01_RUNTIME_NAMESPACE,
                        730,
                        len(self._outcomes) + 1,
                        *pack_key(use.ref.use_key.components),
                    )),
                    origin,
                    (target.object_kind,),
                    AliasRouteSearchBudget(32, 32, 32),
                    False,
                ))
            understanding_status = resolution.status
            recovered = (
                resolution.status == W06_R01_UNDERSTANDING_UNIQUE
                and resolution.selected == target
            )
        verdict = (
            W06_R01_OUTCOME_SUPPORT
            if all((
                authorization_current,
                relation_structure_preserved,
                source_scope_preserved,
                surface_valid,
                understanding_status == W06_R01_UNDERSTANDING_UNIQUE,
                recovered,
            ))
            else W06_R01_OUTCOME_REFUTE
        )
        outcome_key = LosslessIntegerKey((
            W06_R01_RUNTIME_NAMESPACE,
            740,
            len(self._outcomes) + 1,
            *pack_key(use.ref.use_key.components),
        ))
        outcome = W06R01GenerationOutcome(
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
            understanding_status,
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
        request_key: LosslessIntegerKey,
        constraints: GenerationExpressionConstraints,
        ) -> W06R01GenerationRequest:
    """从 typed target 建立不含 expected surface 或 evaluator label 的请求。"""
    if not isinstance(candidate, W06RelationCandidate):
        raise TypeError("generation candidate 类型非法")
    if candidate.substage_key != W06_R01_SUBSTAGE:
        raise W06R01ContractError("generation candidate 不属于 R01")
    return W06R01GenerationRequest(
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
    "W06R01GenerationRuntime",
    "generation_request_for_candidate",
]
