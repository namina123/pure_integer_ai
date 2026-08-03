"""W-07 独立评测的七个逻辑承重维与 W07-G。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.experiments.ph2_w07_contract import (
    W07_GENERATION_ABLATION_KEY,
    W07_GENERATION_HARD_CONJUNCT,
    W07_PUBLIC_ABLATION_KEYS,
    W07_PUBLIC_DIMENSION_KEYS,
)
from pure_integer_ai.experiments.ph2_w07_evaluator_consumers import (
    W07EvaluatorConsumerSuite,
)
from pure_integer_ai.experiments.ph2_w07_evaluator_contract import (
    W07PrivateCase,
    W07PrivateDimensionResult,
    W07PrivateEvaluationError,
    evidence_commitment,
)


_DIMENSION_TO_SUBSTAGE = {
    "W-07-AND_OR": "AND_OR",
    "W-07-CONDITION": "CONDITION",
    "W-07-EXISTS": "EXISTS",
    "W-07-FORALL": "FORALL",
    "W-07-MODAL": "MODAL",
    "W-07-NESTED_SCOPE": "NESTED_SCOPE",
    "W-07-NOT": "NOT",
}


@dataclass(frozen=True)
class W07EvaluatorAblation:
    key: str

    def __post_init__(self) -> None:
        if self.key not in W07_PUBLIC_ABLATION_KEYS:
            raise W07PrivateEvaluationError("W-07 ablation key drift")

    @property
    def dimension_key(self) -> str:
        if self.key == W07_GENERATION_ABLATION_KEY:
            return W07_GENERATION_HARD_CONJUNCT
        return self.key.removesuffix("-ABLATION")


def _result(
        case: W07PrivateCase,
        passed: bool,
        evidence: dict[str, object],
        ) -> W07PrivateDimensionResult:
    return W07PrivateDimensionResult(
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


def evaluate_w07_case(
        suite: W07EvaluatorConsumerSuite,
        case: W07PrivateCase,
        *,
        disabled_dimension: str | None = None,
        evaluation_ordinal: int = 0,
        ) -> W07PrivateDimensionResult:
    """执行一个公开维度，使 evaluator 能安全记录精确失败游标。"""
    if not isinstance(suite, W07EvaluatorConsumerSuite):
        raise TypeError("W-07 evaluator suite type drift")
    if not isinstance(case, W07PrivateCase):
        raise TypeError("W-07 evaluator case type drift")
    if (disabled_dimension is not None
            and disabled_dimension not in W07_PUBLIC_DIMENSION_KEYS):
        raise W07PrivateEvaluationError("W-07 disabled dimension drift")
    if type(evaluation_ordinal) is not int or evaluation_ordinal < 0:
        raise ValueError("W-07 evaluation ordinal is invalid")
    if case.dimension_key == W07_GENERATION_HARD_CONJUNCT:
        passed, evidence = suite.evaluate_generation_hard_conjunct(
            tuple(case.challenge_key),
            generation_connected=case.dimension_key != disabled_dimension,
            evaluation_ordinal=evaluation_ordinal,
        )
    else:
        passed, evidence = suite.evaluate_logic_dimension(
            _DIMENSION_TO_SUBSTAGE[case.dimension_key],
            tuple(case.challenge_key),
            target_connected=case.dimension_key != disabled_dimension,
            evaluation_ordinal=evaluation_ordinal,
        )
    return _result(case, passed, evidence)


def evaluate_w07_learning_runtime(
        suite: W07EvaluatorConsumerSuite,
        cases: tuple[W07PrivateCase, ...],
        *,
        ablation: W07EvaluatorAblation | None = None,
        evaluation_ordinal: int = 0,
        ) -> tuple[W07PrivateDimensionResult, ...]:
    """逐维调用 L01-L07 的 U/R/G facade 与七路生成硬合取。"""
    if not isinstance(suite, W07EvaluatorConsumerSuite):
        raise TypeError("W-07 evaluator suite type drift")
    if type(evaluation_ordinal) is not int or evaluation_ordinal < 0:
        raise ValueError("W-07 evaluation ordinal is invalid")
    if (not isinstance(cases, tuple)
            or tuple(item.dimension_key for item in cases)
            != W07_PUBLIC_DIMENSION_KEYS):
        raise W07PrivateEvaluationError("W-07 private case order drift")
    if ablation is not None and not isinstance(ablation, W07EvaluatorAblation):
        raise TypeError("W-07 evaluator ablation type drift")
    disabled = None if ablation is None else ablation.dimension_key
    return tuple(evaluate_w07_case(
        suite,
        case,
        disabled_dimension=disabled,
        evaluation_ordinal=evaluation_ordinal,
    ) for case in cases)


__all__ = [
    "W07EvaluatorAblation",
    "evaluate_w07_case",
    "evaluate_w07_learning_runtime",
]
