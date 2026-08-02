"""W06-R01 稳定 alias/refers route 的 Understanding consumer。"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.hypothesis import (
    EPISTEMIC_CONFLICTED,
    LIFECYCLE_ACTIVE,
)
from pure_integer_ai.cognition.shared.identity import ObjectIdentity
from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    LosslessIntegerKey,
)
from pure_integer_ai.experiments.ph2_w06_r01_contract import (
    W06R01ContractError,
    W06R01UnderstandingOutcome,
    W06R01UnderstandingRequest,
    W06R01UnderstandingResolution,
    W06R01UnderstandingUse,
    W06_R01_OUTCOME_REFUTE,
    W06_R01_OUTCOME_SUPPORT,
    W06_R01_UNDERSTANDING_CLARIFY,
    W06_R01_UNDERSTANDING_CONFLICT,
    W06_R01_UNDERSTANDING_MULTI,
    W06_R01_UNDERSTANDING_STATUSES,
    W06_R01_UNDERSTANDING_UNIQUE,
    W06_R01_UNDERSTANDING_UNKNOWN,
)
from pure_integer_ai.experiments.ph2_w06_r01_shared import (
    W06R01View,
    W06_R01_RUNTIME_NAMESPACE,
    candidate_endpoints,
    pack_key,
)


class W06R01UnderstandingRuntime:
    """从完整 active route 形成五态结果和 exact Relation Use。"""

    def __init__(self, view: W06R01View) -> None:
        if not isinstance(view, W06R01View):
            raise TypeError("R01 Understanding view 类型非法")
        self.view = view
        self._request_keys: set[LosslessIntegerKey] = set()
        self._resolutions: list[W06R01UnderstandingResolution] = []
        self._uses: list[W06R01UnderstandingUse] = []
        self._outcomes: list[W06R01UnderstandingOutcome] = []

    @property
    def resolutions(self) -> tuple[W06R01UnderstandingResolution, ...]:
        """返回本 runtime 的查询历史。"""
        return tuple(self._resolutions)

    @property
    def uses(self) -> tuple[W06R01UnderstandingUse, ...]:
        """返回已经显式采用的 UNIQUE route。"""
        return tuple(self._uses)

    @property
    def outcomes(self) -> tuple[W06R01UnderstandingOutcome, ...]:
        """返回 append-only 的 exact Use 重验结果。"""
        return tuple(self._outcomes)

    def _matching_from_origin(
            self,
            origin: ObjectIdentity,
            target_kinds: tuple[int, ...],
            ) -> tuple[tuple[object, ObjectIdentity], ...]:
        """按 typed Role 与方向列出 origin 的 direct R01 candidates。"""
        result = []
        for candidate in self.view.candidates:
            source, target = candidate_endpoints(candidate)
            if candidate.relation_family == "PURE_ALIAS":
                if origin == source and target.object_kind in target_kinds:
                    result.append((candidate, target))
                elif origin == target and source.object_kind in target_kinds:
                    result.append((candidate, source))
            elif origin == source and target.object_kind in target_kinds:
                result.append((candidate, target))
        return tuple(sorted(
            result,
            key=lambda item: item[0].proposition.proposition.stable_key(),
        ))

    def preview(
            self,
            request: W06R01UnderstandingRequest,
            ) -> W06R01UnderstandingResolution:
        """无写入地解析完整 route，并将 current 冲突与缺失分开。"""
        proposal = None
        options: tuple[ObjectIdentity, ...] = ()
        propositions: tuple[ObjectIdentity, ...] = ()
        evidence_keys: tuple[tuple[int, ...], ...] = ()
        conflict_options: tuple[ObjectIdentity, ...] = ()
        conflict_props: tuple[ObjectIdentity, ...] = ()
        conflict_evidence: tuple[tuple[int, ...], ...] = ()
        if self.view.protocol.understanding_ready():
            proposal = self.view.alias.preview_reference(
                request.origin,
                target_kinds=request.target_object_kinds,
                budget=request.budget,
            )
            options = tuple(sorted(
                (item.value for item in proposal.result.options),
                key=ObjectIdentity.stable_key,
            ))
            propositions = tuple(sorted({
                step.fact.proposition.proposition
                for option in proposal.result.options
                for route in option.routes
                for step in route.steps
            }, key=ObjectIdentity.stable_key))
            evidence_keys = tuple(sorted({
                key
                for option in proposal.result.options
                for route in option.routes
                for step in route.steps
                for key in step.fact.evidence_keys
            }))
            conflicts = []
            for candidate, target in self._matching_from_origin(
                    request.origin, request.target_object_kinds):
                snapshot = self.view.learning.snapshot_for(
                    candidate.proposition.proposition)
                if (snapshot.snapshot.lifecycle == LIFECYCLE_ACTIVE
                        and snapshot.snapshot.epistemic_status
                        == EPISTEMIC_CONFLICTED):
                    conflicts.append((candidate, target, snapshot))
            conflict_options = tuple(sorted(
                {item[1] for item in conflicts},
                key=ObjectIdentity.stable_key,
            ))
            conflict_props = tuple(sorted(
                {item[0].proposition.proposition for item in conflicts},
                key=ObjectIdentity.stable_key,
            ))
            conflict_evidence = tuple(sorted({
                evidence.stable_key()
                for _candidate, _target, snapshot in conflicts
                for evidence in snapshot.evidence
            }))

        if conflict_props:
            status = W06_R01_UNDERSTANDING_CONFLICT
            options = conflict_options
            propositions = conflict_props
            evidence_keys = conflict_evidence
            selected = None
        elif len(options) == 1:
            status = W06_R01_UNDERSTANDING_UNIQUE
            selected = options[0]
        elif len(options) > 1 and request.allow_multiple:
            status = W06_R01_UNDERSTANDING_MULTI
            selected = None
        elif len(options) > 1:
            status = W06_R01_UNDERSTANDING_CLARIFY
            options = ()
            selected = None
        else:
            status = W06_R01_UNDERSTANDING_UNKNOWN
            selected = None
        return W06R01UnderstandingResolution(
            request,
            status,
            options,
            selected,
            proposal,
            propositions,
            evidence_keys,
            LosslessIntegerKey((
                W06_R01_RUNTIME_NAMESPACE,
                100 + W06_R01_UNDERSTANDING_STATUSES.index(status),
                *self.view.protocol.stable_key(),
            )),
        )

    def resolve(
            self,
            request: W06R01UnderstandingRequest,
            ) -> W06R01UnderstandingResolution:
        """返回 UNIQUE/MULTI/UNKNOWN/CONFLICT/CLARIFY，不私选多候选。"""
        if not isinstance(request, W06R01UnderstandingRequest):
            raise TypeError("R01 understanding request 类型非法")
        if request.request_key in self._request_keys:
            raise W06R01ContractError("重复 R01 understanding request key")
        resolution = self.preview(request)
        self._request_keys.add(request.request_key)
        self._resolutions.append(resolution)
        return resolution

    def adopt(
            self,
            resolution: W06R01UnderstandingResolution,
            ) -> W06R01UnderstandingUse:
        """采用 UNIQUE route 并提交全部 exact active fact Use。"""
        if resolution not in self._resolutions:
            raise W06R01ContractError("understanding resolution 不属于当前 runtime")
        if (resolution.status != W06_R01_UNDERSTANDING_UNIQUE
                or resolution.selected is None
                or resolution.proposal is None):
            raise W06R01ContractError("只有 UNIQUE understanding 可采用")
        use_key = LosslessIntegerKey((
            W06_R01_RUNTIME_NAMESPACE,
            200,
            len(self._uses) + 1,
            *pack_key(resolution.stable_key()),
        ))
        alias_use = self.view.alias.commit_many(((
            resolution.proposal,
            use_key.components,
        ),))[0]
        use = W06R01UnderstandingUse(
            resolution,
            resolution.selected,
            alias_use,
            use_key,
        )
        self._uses.append(use)
        return use

    def verify(
            self,
            use: W06R01UnderstandingUse,
            ) -> W06R01UnderstandingOutcome:
        """按 current route 重验历史选择，withdrawal 后不得继续 SUPPORT。"""
        if use not in self._uses:
            raise W06R01ContractError("understanding Use 不属于当前 runtime")
        current = self.preview(use.resolution.request)
        supported = (
            current.status == W06_R01_UNDERSTANDING_UNIQUE
            and current.selected == use.selection
            and current.propositions == use.resolution.propositions
            and current.evidence_keys == use.resolution.evidence_keys
        )
        outcome = W06R01UnderstandingOutcome(
            use,
            W06_R01_OUTCOME_SUPPORT if supported else W06_R01_OUTCOME_REFUTE,
            current.status,
            LosslessIntegerKey((
                W06_R01_RUNTIME_NAMESPACE,
                300,
                len(self._outcomes) + 1,
                *pack_key(use.use_key.components),
            )),
        )
        self._outcomes.append(outcome)
        return outcome

    def state_key(self) -> tuple:
        """返回 query、Use 与 outcome 的稳定状态。"""
        return (
            tuple(item.stable_key() for item in self._resolutions),
            tuple(item.use_key.components for item in self._uses),
            tuple(item.outcome_key.components for item in self._outcomes),
        )


__all__ = ["W06R01UnderstandingRuntime"]
