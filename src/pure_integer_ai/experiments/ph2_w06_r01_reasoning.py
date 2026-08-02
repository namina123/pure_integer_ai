"""W06-R01 direct typed relation 的 Reasoning consumer。"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.hypothesis import (
    EPISTEMIC_CONFLICTED,
    EPISTEMIC_REFUTED,
    LIFECYCLE_ACTIVE,
)
from pure_integer_ai.cognition.shared.identity import ObjectIdentity
from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    LosslessIntegerKey,
)
from pure_integer_ai.experiments.ph2_w06_adapter import W06RelationCandidate
from pure_integer_ai.experiments.ph2_w06_r01_contract import (
    W06R01ContractError,
    W06R01ReasoningOutcome,
    W06R01ReasoningRequest,
    W06R01ReasoningResolution,
    W06R01ReasoningUse,
    W06_R01_OUTCOME_REFUTE,
    W06_R01_OUTCOME_SUPPORT,
    W06_R01_REASONING_CONFLICT,
    W06_R01_REASONING_REFUTED,
    W06_R01_REASONING_STATUSES,
    W06_R01_REASONING_SUPPORTED,
    W06_R01_REASONING_UNRESOLVED,
)
from pure_integer_ai.experiments.ph2_w06_r01_shared import (
    W06R01View,
    W06_R01_RUNTIME_NAMESPACE,
    candidate_endpoints,
    pack_key,
)


class W06R01ReasoningRuntime:
    """按 typed family/Role/Evidence 区分支持、反驳、冲突与未决。"""

    def __init__(self, view: W06R01View) -> None:
        if not isinstance(view, W06R01View):
            raise TypeError("R01 Reasoning view 类型非法")
        self.view = view
        self._request_keys: set[LosslessIntegerKey] = set()
        self._resolutions: list[W06R01ReasoningResolution] = []
        self._uses: list[W06R01ReasoningUse] = []
        self._outcomes: list[W06R01ReasoningOutcome] = []

    @property
    def resolutions(self) -> tuple[W06R01ReasoningResolution, ...]:
        """返回本 runtime 的关系裁决历史。"""
        return tuple(self._resolutions)

    @property
    def uses(self) -> tuple[W06R01ReasoningUse, ...]:
        """返回 SUPPORTED 结论的 exact Relation Use。"""
        return tuple(self._uses)

    @property
    def outcomes(self) -> tuple[W06R01ReasoningOutcome, ...]:
        """返回 append-only 的 current lifecycle 重验结果。"""
        return tuple(self._outcomes)

    def _matching(
            self,
            request: W06R01ReasoningRequest,
            ) -> tuple[W06RelationCandidate, ...]:
        """按 relation family 与 typed Role 方向返回 direct candidates。"""
        result = []
        for candidate in self.view.candidates:
            if candidate.relation_family != request.relation_family:
                continue
            source, target = candidate_endpoints(candidate)
            matches = (
                {source, target} == {request.source, request.target}
                if request.relation_family == "PURE_ALIAS"
                else source == request.source and target == request.target
            )
            if matches:
                result.append(candidate)
        return tuple(sorted(
            result, key=lambda item: item.proposition.proposition.stable_key()))

    def preview(
            self,
            request: W06R01ReasoningRequest,
            ) -> W06R01ReasoningResolution:
        """无写入地按 current lifecycle 形成 relation 结论。"""
        matches = self._matching(request) if (
            self.view.protocol.reasoning_ready()) else ()
        snapshots = tuple(
            (item, self.view.learning.snapshot_for(
                item.proposition.proposition))
            for item in matches
        )
        conflicts = tuple(
            item for item in snapshots
            if (item[1].snapshot.lifecycle == LIFECYCLE_ACTIVE
                and item[1].snapshot.epistemic_status
                == EPISTEMIC_CONFLICTED))
        active = tuple(item for item in snapshots if item[1].active_fact is not None)
        refuted = tuple(
            item for item in snapshots
            if item[1].snapshot.epistemic_status == EPISTEMIC_REFUTED)
        if conflicts:
            status = W06_R01_REASONING_CONFLICT
            selected = conflicts
        elif active:
            status = W06_R01_REASONING_SUPPORTED
            selected = active
        elif refuted:
            status = W06_R01_REASONING_REFUTED
            selected = refuted
        else:
            status = W06_R01_REASONING_UNRESOLVED
            selected = ()
        propositions = tuple(sorted(
            (item[0].proposition.proposition for item in selected),
            key=ObjectIdentity.stable_key,
        ))
        evidence = tuple(sorted({
            record.stable_key()
            for _candidate, snapshot in selected
            for record in snapshot.evidence
        }))
        return W06R01ReasoningResolution(
            request,
            status,
            propositions,
            evidence,
            LosslessIntegerKey((
                W06_R01_RUNTIME_NAMESPACE,
                400 + W06_R01_REASONING_STATUSES.index(status),
                *self.view.protocol.stable_key(),
            )),
        )

    def resolve(
            self,
            request: W06R01ReasoningRequest,
            ) -> W06R01ReasoningResolution:
        """执行 typed adjudication，不从 cue、共现或 token order 推理。"""
        if not isinstance(request, W06R01ReasoningRequest):
            raise TypeError("R01 reasoning request 类型非法")
        if request.request_key in self._request_keys:
            raise W06R01ContractError("重复 R01 reasoning request key")
        resolution = self.preview(request)
        self._request_keys.add(request.request_key)
        self._resolutions.append(resolution)
        return resolution

    def adopt(
            self,
            resolution: W06R01ReasoningResolution,
            ) -> W06R01ReasoningUse:
        """为 SUPPORTED 结论提交全部 active proposition 的 exact Use。"""
        if resolution not in self._resolutions:
            raise W06R01ContractError("reasoning resolution 不属于当前 runtime")
        if resolution.status != W06_R01_REASONING_SUPPORTED:
            raise W06R01ContractError("只有 SUPPORTED reasoning 可采用")
        use_key = LosslessIntegerKey((
            W06_R01_RUNTIME_NAMESPACE,
            500,
            len(self._uses) + 1,
            *pack_key(resolution.stable_key()),
        ))
        assert self.view.learning.closure is not None
        relation_uses = self.view.learning.closure.consume_many(tuple(
            (
                proposition,
                (
                    *pack_key(use_key.components),
                    ordinal,
                    *pack_key(proposition.stable_key()),
                ),
            )
            for ordinal, proposition in enumerate(
                resolution.propositions, start=1)
        ))
        use = W06R01ReasoningUse(resolution, relation_uses, use_key)
        self._uses.append(use)
        return use

    def verify(
            self,
            use: W06R01ReasoningUse,
            ) -> W06R01ReasoningOutcome:
        """重算结论并核验 Use 的 Evidence/H-04 归因仍是 current。"""
        if use not in self._uses:
            raise W06R01ContractError("reasoning Use 不属于当前 runtime")
        current = self.preview(use.resolution.request)
        supported = (
            current.status == W06_R01_REASONING_SUPPORTED
            and current.propositions == use.resolution.propositions
            and current.evidence_keys == use.resolution.evidence_keys
        )
        outcome = W06R01ReasoningOutcome(
            use,
            W06_R01_OUTCOME_SUPPORT if supported else W06_R01_OUTCOME_REFUTE,
            current.status,
            LosslessIntegerKey((
                W06_R01_RUNTIME_NAMESPACE,
                600,
                len(self._outcomes) + 1,
                *pack_key(use.use_key.components),
            )),
        )
        self._outcomes.append(outcome)
        return outcome

    def state_key(self) -> tuple:
        """返回 adjudication、Use 与 outcome 的稳定状态。"""
        return (
            tuple(item.stable_key() for item in self._resolutions),
            tuple(item.use_key.components for item in self._uses),
            tuple(item.outcome_key.components for item in self._outcomes),
        )


__all__ = ["W06R01ReasoningRuntime"]
