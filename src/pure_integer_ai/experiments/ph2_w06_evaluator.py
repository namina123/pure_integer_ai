"""W-06 private evaluator 的七 relation bearing 与 generation hard conjunct。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.experiments.ph2_w06_contract import (
    W06_ABLATION_KEYS,
    W06_EVALUATION_ORDER,
    W06_GENERATION_ABLATION_KEY,
)
from pure_integer_ai.experiments.ph2_w06_evaluator_consumers import (
    W06EvaluatorConsumerSuite,
)
from pure_integer_ai.experiments.ph2_w06_evaluator_contract import (
    W06PrivateCase,
    W06PrivateDimensionResult,
    W06PrivateEvaluationError,
    evidence_commitment,
)
from pure_integer_ai.experiments.ph2_w06_source_semantic import (
    W06_GENERATION_HARD_CONJUNCT,
)


_DIMENSION_TO_SUBSTAGE = {
    "W-06-CAUSES": "CAUSES",
    "W-06-MEREOLOGY": "MEREOLOGY",
    "W-06-PRECEDES": "PRECEDES",
    "W-06-PROPERTY": "PROPERTY",
    "W-06-PURE_ALIAS_REFERS": "PURE_ALIAS_REFERS",
    "W-06-SIMILAR_ANTONYM": "SIMILAR_ANTONYM",
    "W-06-SUBSET_MEMBER": "SUBSET_MEMBER",
}


@dataclass(frozen=True)
class W06EvaluatorAblation:
    """一次只关闭一个预注册 relation 或 generation 承重维度。"""

    key: str

    def __post_init__(self) -> None:
        if self.key not in (*W06_ABLATION_KEYS, W06_GENERATION_ABLATION_KEY):
            raise W06PrivateEvaluationError("W-06 ablation key 非法")

    @property
    def dimension_key(self) -> str:
        if self.key == W06_GENERATION_ABLATION_KEY:
            return W06_GENERATION_HARD_CONJUNCT
        return self.key.removesuffix("-ABLATION")


def _relation_evidence(
        suite: W06EvaluatorConsumerSuite,
        case: W06PrivateCase,
        *,
        substage: str,
        connected: bool,
        evaluation_ordinal: int,
        ) -> tuple[bool, dict[str, object]]:
    """通过指定 relation 的公开 facade 执行真实 U/R/G。"""
    return suite.evaluate_relation(
        substage,
        tuple(case.challenge_key),
        target_connected=connected,
        evaluation_ordinal=evaluation_ordinal,
    )


def _generation_evidence(
        suite: W06EvaluatorConsumerSuite,
        case: W06PrivateCase,
        *,
        connected: bool,
        evaluation_ordinal: int,
        ) -> tuple[bool, dict[str, object]]:
    """逐 relation 执行真实 choice/use/postcheck 硬合取。"""
    return suite.evaluate_generation_hard_conjunct(
        tuple(case.challenge_key),
        generation_connected=connected,
        evaluation_ordinal=evaluation_ordinal,
    )


def _result(
        case: W06PrivateCase,
        passed: bool,
        evidence: dict[str, object],
        ) -> W06PrivateDimensionResult:
    return W06PrivateDimensionResult(
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


def evaluate_w06_learning_runtime(
        suite: W06EvaluatorConsumerSuite,
        cases: tuple[W06PrivateCase, ...],
        *,
        ablation: W06EvaluatorAblation | None = None,
        evaluation_ordinal: int = 0,
        ) -> tuple[W06PrivateDimensionResult, ...]:
    """在七个隔离 relation owner 上执行八项真实 consumer 合取。"""
    if not isinstance(suite, W06EvaluatorConsumerSuite):
        raise TypeError("W-06 evaluator suite 类型非法")
    if (type(evaluation_ordinal) is not int or evaluation_ordinal < 0):
        raise ValueError("W-06 evaluation ordinal 非法")
    if (not isinstance(cases, tuple)
            or tuple(item.dimension_key for item in cases)
            != W06_EVALUATION_ORDER):
        raise W06PrivateEvaluationError("W-06 private case 顺序漂移")
    if ablation is not None and not isinstance(ablation, W06EvaluatorAblation):
        raise TypeError("W-06 ablation 类型非法")
    disabled = None if ablation is None else ablation.dimension_key
    evaluations = []
    for case in cases:
        if case.dimension_key == W06_GENERATION_HARD_CONJUNCT:
            evaluations.append(_generation_evidence(
                suite,
                case,
                connected=case.dimension_key != disabled,
                evaluation_ordinal=evaluation_ordinal,
            ))
        else:
            evaluations.append(_relation_evidence(
                suite,
                case,
                substage=_DIMENSION_TO_SUBSTAGE[case.dimension_key],
                connected=case.dimension_key != disabled,
                evaluation_ordinal=evaluation_ordinal,
            ))
    return tuple(
        _result(case, passed, evidence)
        for case, (passed, evidence) in zip(cases, evaluations, strict=True)
    )


__all__ = [
    "W06EvaluatorAblation",
    "evaluate_w06_learning_runtime",
]
