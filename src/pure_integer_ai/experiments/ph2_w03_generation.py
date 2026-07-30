"""W03-G 从 active Sense projection 到 surface choice/Use/outcome 的闭环。"""
from __future__ import annotations

from pure_integer_ai.cognition.understanding.language_candidate import (
    ActiveSenseCandidate,
)
from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    GenerationChoiceOutcomeRef,
    GenerationChoiceUseRef,
    LosslessIntegerKey,
)
from pure_integer_ai.experiments.ph2_w03_adapter import (
    W03SenseCandidateEnvelope,
)
from pure_integer_ai.experiments.ph2_w03_generation_contract import (
    W03_GENERATION_ADOPTED,
    W03_GENERATION_CLARIFY,
    W03_GENERATION_HARD_CASES,
    W03_GENERATION_OUTCOME_NEUTRAL,
    W03_GENERATION_OUTCOME_REFUTE,
    W03_GENERATION_OUTCOME_SUPPORT,
    W03_GENERATION_READY,
    W03_GENERATION_REJECTED,
    W03_GENERATION_UNKNOWN,
    W03ExpressionConstraints,
    W03GenerationCaseResult,
    W03GenerationChoice,
    W03GenerationDecision,
    W03GenerationError,
    W03GenerationHardConjunctReport,
    W03GenerationOption,
    W03GenerationOutcome,
    W03GenerationRequest,
    W03GenerationUse,
    choice_reason_key,
)
from pure_integer_ai.experiments.ph2_w03_understanding import (
    W03UnderstandingRuntime,
)


_NAMESPACE = 30304
_GENERATION_DIMENSION_KEY = LosslessIntegerKey((_NAMESPACE, 301))
_ACTIVE_PROJECTION_VERIFIER_KEY = LosslessIntegerKey((_NAMESPACE, 302))
_OUTCOME_RESULT_KEYS = {
    W03_GENERATION_OUTCOME_SUPPORT: LosslessIntegerKey((_NAMESPACE, 303)),
    W03_GENERATION_OUTCOME_REFUTE: LosslessIntegerKey((_NAMESPACE, 304)),
    W03_GENERATION_OUTCOME_NEUTRAL: LosslessIntegerKey((_NAMESPACE, 305)),
}


def _pack(value: tuple[int, ...]) -> tuple[int, ...]:
    return len(value), *value


def _authorization_key(active: ActiveSenseCandidate) -> LosslessIntegerKey:
    """保留当前 lifecycle state、最后 Event 和 aggregate Hypothesis 全键。"""
    if not active.projection.history:
        raise W03GenerationError("active Sense projection 缺 lifecycle history")
    event = active.projection.history[-1].definition.event
    return LosslessIntegerKey((
        _NAMESPACE,
        1,
        *_pack(active.sense.stable_key()),
        *_pack(active.concept.stable_key()),
        *_pack(active.context.stable_key()),
        *_pack(active.projection.state.stable_key()),
        *_pack(event.stable_key()),
        *_pack(active.projection.candidate.hypothesis.stable_key()),
    ))


class W03GenerationRuntime:
    """只扫描 typed inventory，逐项经当前 active Sense consumer 授权。"""

    def __init__(
            self,
            understanding: W03UnderstandingRuntime,
            *,
            sense_consumer_connected: bool = True,
            choice_bridge_connected: bool = True,
            ) -> None:
        if not isinstance(understanding, W03UnderstandingRuntime):
            raise TypeError("generation understanding runtime 类型非法")
        if type(sense_consumer_connected) is not bool:
            raise TypeError("sense_consumer_connected 必须是严格 bool")
        if type(choice_bridge_connected) is not bool:
            raise TypeError("choice_bridge_connected 必须是严格 bool")
        self.understanding = understanding
        self.sense_consumer_connected = sense_consumer_connected
        self.choice_bridge_connected = choice_bridge_connected
        self._candidate_by_sense = {
            item.sense: item for item in understanding.output.candidates
        }
        if len(self._candidate_by_sense) != len(
                understanding.output.candidates):
            raise W03GenerationError("generation inventory 含重复 Sense identity")
        self._choices: list[W03GenerationChoice] = []
        self._decisions: list[W03GenerationDecision] = []
        self._uses: list[W03GenerationUse] = []
        self._outcomes: list[W03GenerationOutcome] = []
        self._request_keys: set[LosslessIntegerKey] = set()

    @property
    def choices(self) -> tuple[W03GenerationChoice, ...]:
        return tuple(self._choices)

    @property
    def decisions(self) -> tuple[W03GenerationDecision, ...]:
        return tuple(self._decisions)

    @property
    def uses(self) -> tuple[W03GenerationUse, ...]:
        return tuple(self._uses)

    @property
    def outcomes(self) -> tuple[W03GenerationOutcome, ...]:
        return tuple(self._outcomes)

    def _active_projection(
            self,
            candidate: W03SenseCandidateEnvelope,
            ) -> ActiveSenseCandidate | None:
        """用公开 consumer 入口核精确 Sense/Concept/context，不读旧表。"""
        if not self.sense_consumer_connected:
            return None
        matches = tuple(
            item for item in self.understanding.consumer.lookup(
                candidate.anchor.atom,
                context=candidate.context,
            )
            if (item.sense == candidate.sense
                and item.concept == candidate.concept
                and item.context == candidate.context
                and item.atom == candidate.anchor.atom)
        )
        if len(matches) > 1:
            raise W03GenerationError("active consumer 返回重复 Sense projection")
        return matches[0] if matches else None

    @staticmethod
    def _allowed_by_constraints(
            candidate: W03SenseCandidateEnvelope,
            request: W03GenerationRequest,
            ) -> bool:
        surface = candidate.anchor.extracted.surface
        if len(surface) > request.constraints.max_codepoints:
            return False
        if candidate.lexicalized_multiword:
            return request.constraints.allow_lexicalized_multiword
        return request.constraints.allow_single_atom

    @staticmethod
    def _option(
            candidate: W03SenseCandidateEnvelope,
            active: ActiveSenseCandidate,
            ) -> W03GenerationOption:
        return W03GenerationOption(
            candidate.anchor.extracted.surface,
            candidate.sense,
            candidate.concept,
            candidate.context,
            candidate.anchor.branch,
            candidate.anchor.atom,
            candidate.anchor.representation,
            candidate.anchor.span,
            candidate.source_ref,
            candidate.lexicalized_multiword,
            _authorization_key(active),
        )

    def choose(self, request: W03GenerationRequest) -> W03GenerationChoice:
        """验证 target 后返回全部合法 option；不自动采用 stable-order 首项。"""
        if not isinstance(request, W03GenerationRequest):
            raise TypeError("generation request 类型非法")
        if request.request_key in self._request_keys:
            raise W03GenerationError("重复 generation request key")
        target = self._candidate_by_sense.get(request.target_sense)
        target_active = (
            None if target is None else self._active_projection(target)
        )
        if target is None or target_active is None:
            status = W03_GENERATION_UNKNOWN
            options: tuple[W03GenerationOption, ...] = ()
        elif request.context is None:
            status = W03_GENERATION_CLARIFY
            options = ()
        elif (target.concept != request.target_concept
                or target.context != request.context
                or target.anchor.branch != request.branch):
            status = W03_GENERATION_UNKNOWN
            options = ()
        else:
            values = []
            for candidate in self.understanding.output.candidates:
                if (candidate.concept != request.target_concept
                        or candidate.context != request.context
                        or candidate.anchor.branch != request.branch
                        or not self._allowed_by_constraints(candidate, request)):
                    continue
                active = self._active_projection(candidate)
                if active is not None:
                    values.append(self._option(candidate, active))
            options = tuple(sorted(values, key=lambda item: item.stable_key()))
            status = (
                W03_GENERATION_READY
                if options else W03_GENERATION_UNKNOWN
            )
        choice = W03GenerationChoice(
            request,
            status,
            options,
            None,
            choice_reason_key(status),
        )
        if self.choice_bridge_connected:
            self._request_keys.add(request.request_key)
            self._choices.append(choice)
        return choice

    def adopt(
            self,
            choice: W03GenerationChoice,
            selected_option_keys: tuple[tuple[int, ...], ...],
            ) -> tuple[W03GenerationUse, ...]:
        """原子记录每个 option 的采用/拒绝及独立 exact Use。"""
        if not isinstance(choice, W03GenerationChoice):
            raise TypeError("generation choice 类型非法")
        if not self.choice_bridge_connected:
            return ()
        if choice not in self._choices:
            raise W03GenerationError("choice 不属于当前 generation bridge")
        if choice.status != W03_GENERATION_READY:
            raise W03GenerationError("非 READY choice 不得采用 option")
        if (not isinstance(selected_option_keys, tuple)
                or not selected_option_keys
                or any(not isinstance(item, tuple) or not item
                       or any(type(value) is not int for value in item)
                       for item in selected_option_keys)):
            raise W03GenerationError("selected option keys 类型非法")
        if len(set(selected_option_keys)) != len(selected_option_keys):
            raise W03GenerationError("selected option keys 重复")
        available = {item.stable_key(): item for item in choice.options}
        if any(item not in available for item in selected_option_keys):
            raise W03GenerationError("selected option 不属于当前 choice")
        if any(item.decision.choice == choice for item in self._uses):
            raise W03GenerationError("同一 choice 不得重复采用")

        selected = set(selected_option_keys)
        decisions = []
        uses = []
        for ordinal, option in enumerate(choice.options, start=1):
            action = (
                W03_GENERATION_ADOPTED
                if option.stable_key() in selected
                else W03_GENERATION_REJECTED
            )
            decision_key = LosslessIntegerKey((
                _NAMESPACE,
                10,
                ordinal,
                *_pack(choice.stable_key()),
                *_pack(option.stable_key()),
                1 if action == W03_GENERATION_ADOPTED else 2,
            ))
            decision = W03GenerationDecision(
                choice,
                option,
                action,
                decision_key,
            )
            use_key = LosslessIntegerKey((
                _NAMESPACE,
                20,
                *_pack(decision.stable_key()),
            ))
            ref = GenerationChoiceUseRef(
                "CORE_USE",
                use_key,
                LosslessIntegerKey(option.stable_key()),
                choice.request.scope,
            )
            decisions.append(decision)
            uses.append(W03GenerationUse(decision, ref))
        self._decisions.extend(decisions)
        self._uses.extend(uses)
        return tuple(uses)

    def _current_authorization(
            self,
            option: W03GenerationOption,
            ) -> LosslessIntegerKey | None:
        candidate = self._candidate_by_sense.get(option.sense)
        if candidate is None:
            return None
        if (candidate.concept != option.concept
                or candidate.context != option.context
                or candidate.anchor.branch != option.branch
                or candidate.anchor.atom != option.atom
                or candidate.anchor.representation != option.representation
                or candidate.anchor.span != option.span
                or candidate.anchor.extracted.surface != option.surface):
            raise W03GenerationError("stored option identity 与 inventory 漂移")
        active = self._active_projection(candidate)
        return None if active is None else _authorization_key(active)

    def verify_use(self, use: W03GenerationUse) -> W03GenerationOutcome:
        """按当前 active projection 重验一个 Use，并保留所有历史结果。"""
        if not isinstance(use, W03GenerationUse):
            raise TypeError("generation Use 类型非法")
        if use not in self._uses:
            raise W03GenerationError("generation Use 不属于当前 ledger")
        current = self._current_authorization(use.decision.option)
        if use.decision.action == W03_GENERATION_ADOPTED:
            verdict = (
                W03_GENERATION_OUTCOME_SUPPORT
                if current is not None
                else W03_GENERATION_OUTCOME_REFUTE
            )
        else:
            verdict = (
                W03_GENERATION_OUTCOME_NEUTRAL
                if current is not None
                else W03_GENERATION_OUTCOME_SUPPORT
            )
        outcome_ordinal = sum(
            item.use.ref.use_key == use.ref.use_key
            for item in self._outcomes
        ) + 1
        outcome_key = LosslessIntegerKey((
            _NAMESPACE,
            30,
            outcome_ordinal,
            *_pack(use.ref.use_key.components),
        ))
        ref = GenerationChoiceOutcomeRef(
            outcome_key,
            use.ref.use_key,
            _GENERATION_DIMENSION_KEY,
            _ACTIVE_PROJECTION_VERIFIER_KEY,
            _OUTCOME_RESULT_KEYS[verdict],
        )
        outcome = W03GenerationOutcome(use, verdict, ref, current)
        self._outcomes.append(outcome)
        return outcome


def build_w03_generation_runtime(
        understanding: W03UnderstandingRuntime,
        *,
        sense_consumer_connected: bool = True,
        choice_bridge_connected: bool = True,
        ) -> W03GenerationRuntime:
    """建立 W03-G test-local runtime，不启动正式 W-03 training run。"""
    return W03GenerationRuntime(
        understanding,
        sense_consumer_connected=sense_consumer_connected,
        choice_bridge_connected=choice_bridge_connected,
    )


def run_w03_generation_hard_conjunct(
        cases: tuple[W03GenerationCaseResult, ...],
        *,
        sense_consumer_connected: bool,
        choice_bridge_connected: bool,
        ) -> W03GenerationHardConjunctReport:
    """聚合五个逻辑 case；不捕获异常，基础设施错误必须向上冒泡。"""
    if (not isinstance(cases, tuple)
            or any(not isinstance(item, W03GenerationCaseResult)
                   for item in cases)):
        raise W03GenerationError("generation hard cases 类型非法")
    if tuple(item.case_name for item in cases) != W03_GENERATION_HARD_CASES:
        raise W03GenerationError("generation hard cases 缺失、重复或顺序漂移")
    if type(sense_consumer_connected) is not bool:
        raise TypeError("sense_consumer_connected 必须是严格 bool")
    if type(choice_bridge_connected) is not bool:
        raise TypeError("choice_bridge_connected 必须是严格 bool")
    passed = int(
        all(item.passed for item in cases)
        and sense_consumer_connected
        and choice_bridge_connected
    )
    return W03GenerationHardConjunctReport(
        "PASS" if passed else "FAIL",
        passed,
        1,
        1 - passed,
        0,
        cases,
        sense_consumer_connected,
        choice_bridge_connected,
    )


__all__ = [
    "W03_GENERATION_ADOPTED",
    "W03_GENERATION_CLARIFY",
    "W03_GENERATION_HARD_CASES",
    "W03_GENERATION_OUTCOME_NEUTRAL",
    "W03_GENERATION_OUTCOME_REFUTE",
    "W03_GENERATION_OUTCOME_SUPPORT",
    "W03_GENERATION_READY",
    "W03_GENERATION_REJECTED",
    "W03_GENERATION_UNKNOWN",
    "W03ExpressionConstraints",
    "W03GenerationCaseResult",
    "W03GenerationError",
    "W03GenerationRequest",
    "W03GenerationRuntime",
    "build_w03_generation_runtime",
    "run_w03_generation_hard_conjunct",
]
