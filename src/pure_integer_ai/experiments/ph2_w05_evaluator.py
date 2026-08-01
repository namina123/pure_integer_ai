"""W-05 private evaluator 的四 bearing 与 generation hard conjunct。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.cognition.shared.identity import (
    concept_identity,
    language_branch_identity,
)
from pure_integer_ai.experiments.ph2_generation_choice_contract import (
    GenerationExpressionConstraints,
    LosslessIntegerKey,
)
from pure_integer_ai.experiments.ph2_w05_adapter import W05_IDENTITY_VERSIONS
from pure_integer_ai.experiments.ph2_w05_contract import (
    W05_ABLATION_KEYS,
    W05_EVALUATION_ORDER,
)
from pure_integer_ai.experiments.ph2_w05_evaluator_contract import (
    W05_GENERATION_ABLATION_KEY,
    W05PrivateCase,
    W05PrivateDimensionResult,
    W05PrivateEvaluationError,
    evidence_commitment,
)
from pure_integer_ai.experiments.ph2_w05_generation import (
    build_w05_generation_runtime,
    generation_request_for_candidate,
)
from pure_integer_ai.experiments.ph2_w05_generation_contract import (
    W05_GENERATION_ADOPTED,
    W05_GENERATION_HARD_CASES,
    W05_GENERATION_OUTCOME_SUPPORT,
    W05_GENERATION_READY,
    W05GenerationCaseResult,
    W05GenerationProtocol,
    run_w05_generation_hard_conjunct,
)
from pure_integer_ai.experiments.ph2_w05_reasoning import (
    W05_REASONING_AUTHORIZED,
    W05_REASONING_CONFLICT,
    W05_REASONING_OUTCOME_SUPPORT,
    W05_REASONING_REJECTED,
    W05ReasoningProtocol,
    build_w05_reasoning_runtime,
    reasoning_request_for_candidate,
)
from pure_integer_ai.experiments.ph2_w05_understanding import (
    W05_UNDERSTANDING_CONFLICT,
    W05_UNDERSTANDING_OUTCOME_SUPPORT,
    W05_UNDERSTANDING_UNIQUE,
    W05_UNDERSTANDING_UNKNOWN,
    W05UnderstandingProtocol,
    build_w05_understanding_runtime,
    understanding_request_for_candidate,
)


_EVALUATOR_NAMESPACE = 50516


@dataclass(frozen=True)
class W05EvaluatorAblation:
    """一次只关闭一个预注册承重维度。"""

    key: str

    def __post_init__(self) -> None:
        if self.key not in (*W05_ABLATION_KEYS, W05_GENERATION_ABLATION_KEY):
            raise W05PrivateEvaluationError("W-05 ablation key 非法")

    @property
    def dimension_key(self) -> str:
        """返回该 ablation 唯一允许击穿的公开维度。"""
        if self.key == W05_GENERATION_ABLATION_KEY:
            return W05_EVALUATION_ORDER[-1]
        return self.key.removesuffix("-ABLATION")


def _candidate(learning, perturbation: str):
    """按课程已冻结的 perturbation 元数据取得唯一候选。"""
    values = tuple(
        item for item in learning.registered_candidates()
        if item.perturbation_kind == perturbation
    )
    if len(values) != 1:
        raise W05PrivateEvaluationError("W-05 evaluator candidate inventory 漂移")
    return values[0]


def _request_key(case: W05PrivateCase, ordinal: int) -> LosslessIntegerKey:
    """把 private challenge commitment 映射成不泄漏内容的整数请求键。"""
    return LosslessIntegerKey((
        _EVALUATOR_NAMESPACE,
        ordinal,
        len(case.challenge_key),
        *case.challenge_key,
    ))


def _generation_evidence(
        learning,
        case: W05PrivateCase,
        *,
        protocol: W05GenerationProtocol,
        ) -> tuple[bool, dict[str, object]]:
    """执行六项 W05-G construction/Use/outcome 合取并返回安全计数。"""
    supported = _candidate(learning, "NONE")
    branch = language_branch_identity(
        (_EVALUATOR_NAMESPACE, 70), versions=W05_IDENTITY_VERSIONS)
    uncertainty = concept_identity(
        (_EVALUATOR_NAMESPACE, 71), versions=W05_IDENTITY_VERSIONS)
    constraints = GenerationExpressionConstraints(
        branch, (), (), 0, 0, 0, 128)
    generation = build_w05_generation_runtime(learning, protocol=protocol)
    choice = generation.choose(generation_request_for_candidate(
        supported,
        request_key=_request_key(case, 72),
        uncertainty=uncertainty,
        constraints=constraints,
    ))
    outcomes = ()
    if choice.status == W05_GENERATION_READY and choice.options:
        selected = tuple(item.stable_key() for item in choice.options)
        uses = generation.adopt(choice, selected)
        adopted = tuple(
            item for item in uses
            if item.decision.action == W05_GENERATION_ADOPTED
        )
        independent = build_w05_understanding_runtime(learning)
        outcomes = tuple(
            generation.verify_use(item, understanding=independent)
            for item in adopted
        )
    case_values = (
        choice.status == W05_GENERATION_READY and bool(choice.options),
        bool(outcomes) and all(item.occurrence_preserved for item in outcomes),
        bool(outcomes) and all(item.role_preserved for item in outcomes),
        bool(outcomes) and all(item.scope_preserved for item in outcomes),
        bool(outcomes) and all(
            item.understanding_status == W05_UNDERSTANDING_UNIQUE
            for item in outcomes),
        bool(outcomes) and all(
            item.verdict == W05_GENERATION_OUTCOME_SUPPORT
            for item in outcomes),
    )
    hard_cases = tuple(
        W05GenerationCaseResult(
            name,
            passed,
            LosslessIntegerKey((
                _EVALUATOR_NAMESPACE, 73, ordinal,
                len(case.challenge_key), *case.challenge_key,
            )),
        )
        for ordinal, (name, passed) in enumerate(
            zip(W05_GENERATION_HARD_CASES, case_values, strict=True),
            start=1,
        )
    )
    hard = run_w05_generation_hard_conjunct(hard_cases, protocol=protocol)
    return hard.status == "PASS", {
        "adopted_use_count": len(outcomes),
        "case_count": len(hard.cases),
        "case_pass_count": sum(int(item.passed) for item in hard.cases),
        "choice_option_count": len(choice.options),
        "choice_ready": int(choice.status == W05_GENERATION_READY),
        "outcome_support_count": sum(
            int(item.verdict == W05_GENERATION_OUTCOME_SUPPORT)
            for item in outcomes),
    }


def _occurrence_evidence(
        learning,
        *,
        connected: bool,
        ) -> tuple[bool, dict[str, object]]:
    """核 occurrence 不折叠、omission supersede 与 restore active。"""
    report = learning.report()
    supported = _candidate(learning, "NONE")
    role_swap = _candidate(learning, "ROLE_SWAP")
    omission = _candidate(learning, "OCCURRENCE_OMISSION")
    restore = _candidate(learning, "OCCURRENCE_RESTORE")
    supported_ids = {item.identity for item in supported.occurrences}
    role_swap_ids = {item.identity for item in role_swap.occurrences}
    active = {item.candidate for item in learning.active_candidates()}
    superseded = {item.candidate for item in learning.superseded_candidates()}
    passed = all((
        connected,
        report.occurrence_count == 19,
        supported.surface == role_swap.surface,
        supported_ids.isdisjoint(role_swap_ids),
        omission.candidate in superseded,
        restore.candidate in active,
    ))
    return passed, {
        "active_restore": int(restore.candidate in active),
        "bridge_connected": int(connected),
        "occurrence_count": report.occurrence_count,
        "omission_superseded": int(omission.candidate in superseded),
        "same_surface_disjoint": int(supported_ids.isdisjoint(role_swap_ids)),
    }


def _proposition_evidence(
        learning,
        case: W05PrivateCase,
        *,
        connected: bool,
        ) -> tuple[bool, dict[str, object]]:
    """分别要求 Understanding、Reasoning、Generation 产生 exact Use/outcome。"""
    supported = _candidate(learning, "NONE")
    understanding = build_w05_understanding_runtime(
        learning,
        protocol=W05UnderstandingProtocol(
            proposition_consumer_connected=connected),
    )
    resolution = understanding.resolve(understanding_request_for_candidate(
        supported, request_key=_request_key(case, 20)))
    understanding_ok = False
    if resolution.status == W05_UNDERSTANDING_UNIQUE:
        use = understanding.adopt(resolution, supported)
        understanding_ok = (
            understanding.verify_use(use).verdict
            == W05_UNDERSTANDING_OUTCOME_SUPPORT
        )
    reasoning = build_w05_reasoning_runtime(
        learning,
        protocol=W05ReasoningProtocol(
            proposition_consumer_connected=connected),
    )
    reasoning_use = reasoning.authorize(reasoning_request_for_candidate(
        supported, request_key=_request_key(case, 21)))
    reasoning_ok = (
        reasoning_use.status == W05_REASONING_AUTHORIZED
        and reasoning.verify_use(reasoning_use).verdict
        == W05_REASONING_OUTCOME_SUPPORT
    )
    generation_ok, generation = _generation_evidence(
        learning,
        case,
        protocol=W05GenerationProtocol(
            proposition_consumer_connected=connected),
    )
    passed = connected and understanding_ok and reasoning_ok and generation_ok
    return passed, {
        "bridge_connected": int(connected),
        "generation": generation,
        "reasoning_authorized": int(reasoning_ok),
        "understanding_unique": int(understanding_ok),
    }


def _role_evidence(
        learning,
        case: W05PrivateCase,
        *,
        connected: bool,
        ) -> tuple[bool, dict[str, object]]:
    """比较正常 Proposition 与 ROLE_SWAP 的独立理解、推理状态。"""
    supported = _candidate(learning, "NONE")
    role_swap = _candidate(learning, "ROLE_SWAP")
    understanding = build_w05_understanding_runtime(
        learning,
        protocol=W05UnderstandingProtocol(role_bridge_connected=connected),
    )
    normal_resolution = understanding.resolve(understanding_request_for_candidate(
        supported, request_key=_request_key(case, 30)))
    swapped_resolution = understanding.resolve(understanding_request_for_candidate(
        role_swap, request_key=_request_key(case, 31)))
    reasoning = build_w05_reasoning_runtime(
        learning,
        protocol=W05ReasoningProtocol(role_bridge_connected=connected),
    )
    normal_use = reasoning.authorize(reasoning_request_for_candidate(
        supported, request_key=_request_key(case, 32)))
    swapped_use = reasoning.authorize(reasoning_request_for_candidate(
        role_swap, request_key=_request_key(case, 33)))
    passed = all((
        connected,
        normal_resolution.status == W05_UNDERSTANDING_UNIQUE,
        swapped_resolution.status == W05_UNDERSTANDING_UNKNOWN,
        normal_use.status == W05_REASONING_AUTHORIZED,
        swapped_use.status == W05_REASONING_REJECTED,
    ))
    return passed, {
        "bridge_connected": int(connected),
        "normal_authorized": int(normal_use.status == W05_REASONING_AUTHORIZED),
        "normal_unique": int(
            normal_resolution.status == W05_UNDERSTANDING_UNIQUE),
        "swap_rejected": int(swapped_use.status == W05_REASONING_REJECTED),
        "swap_unknown": int(
            swapped_resolution.status == W05_UNDERSTANDING_UNKNOWN),
    }


def _scope_evidence(
        learning,
        case: W05PrivateCase,
        *,
        connected: bool,
        ) -> tuple[bool, dict[str, object]]:
    """比较正常 Proposition 与 SCOPE_SHIFT 的独立理解、推理状态。"""
    supported = _candidate(learning, "NONE")
    scope_shift = _candidate(learning, "SCOPE_SHIFT")
    understanding = build_w05_understanding_runtime(
        learning,
        protocol=W05UnderstandingProtocol(scope_projection_connected=connected),
    )
    normal_resolution = understanding.resolve(understanding_request_for_candidate(
        supported, request_key=_request_key(case, 40)))
    shifted_resolution = understanding.resolve(understanding_request_for_candidate(
        scope_shift, request_key=_request_key(case, 41)))
    reasoning = build_w05_reasoning_runtime(
        learning,
        protocol=W05ReasoningProtocol(scope_projection_connected=connected),
    )
    normal_use = reasoning.authorize(reasoning_request_for_candidate(
        supported, request_key=_request_key(case, 42)))
    shifted_use = reasoning.authorize(reasoning_request_for_candidate(
        scope_shift, request_key=_request_key(case, 43)))
    passed = all((
        connected,
        normal_resolution.status == W05_UNDERSTANDING_UNIQUE,
        shifted_resolution.status == W05_UNDERSTANDING_CONFLICT,
        normal_use.status == W05_REASONING_AUTHORIZED,
        shifted_use.status == W05_REASONING_CONFLICT,
    ))
    return passed, {
        "bridge_connected": int(connected),
        "normal_authorized": int(normal_use.status == W05_REASONING_AUTHORIZED),
        "normal_unique": int(
            normal_resolution.status == W05_UNDERSTANDING_UNIQUE),
        "shift_conflict_reasoning": int(
            shifted_use.status == W05_REASONING_CONFLICT),
        "shift_conflict_understanding": int(
            shifted_resolution.status == W05_UNDERSTANDING_CONFLICT),
    }


def _result(
        case: W05PrivateCase,
        passed: bool,
        evidence: dict[str, object],
        ) -> W05PrivateDimensionResult:
    """把内部证据压缩为不泄漏 private 标识的 commitment。"""
    return W05PrivateDimensionResult(
        case.dimension_key,
        "PASS" if passed else "FAIL",
        int(passed),
        1,
        int(not passed),
        0,
        evidence_commitment({
            "challenge_commitment": evidence_commitment(
                list(case.challenge_key)),
            "dimension_key": case.dimension_key,
            "evidence": evidence,
            "passed": int(passed),
        }),
    )


def evaluate_w05_learning_runtime(
        learning,
        cases: tuple[W05PrivateCase, ...],
        *,
        ablation: W05EvaluatorAblation | None = None,
        ) -> tuple[W05PrivateDimensionResult, ...]:
    """在只读学习态上执行五项 1/1，并让消融只关闭目标桥。"""
    if (not isinstance(cases, tuple)
            or tuple(item.dimension_key for item in cases)
            != W05_EVALUATION_ORDER):
        raise W05PrivateEvaluationError("W-05 private case 顺序漂移")
    if ablation is not None and not isinstance(ablation, W05EvaluatorAblation):
        raise TypeError("W-05 ablation 类型非法")
    disabled = None if ablation is None else ablation.dimension_key
    evaluations = (
        _occurrence_evidence(
            learning, connected=W05_EVALUATION_ORDER[0] != disabled),
        _proposition_evidence(
            learning, cases[1], connected=W05_EVALUATION_ORDER[1] != disabled),
        _role_evidence(
            learning, cases[2], connected=W05_EVALUATION_ORDER[2] != disabled),
        _scope_evidence(
            learning, cases[3], connected=W05_EVALUATION_ORDER[3] != disabled),
        _generation_evidence(
            learning,
            cases[4],
            protocol=W05GenerationProtocol(
                generation_bridge_connected=W05_EVALUATION_ORDER[4] != disabled),
        ),
    )
    return tuple(
        _result(case, passed, evidence)
        for case, (passed, evidence) in zip(cases, evaluations, strict=True)
    )


__all__ = [
    "W05EvaluatorAblation",
    "evaluate_w05_learning_runtime",
]
