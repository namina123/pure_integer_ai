"""把 H-03 严格描述长度适配为 H-04 typed 成对关系。"""
from __future__ import annotations

from pure_integer_ai.cognition.shared.description_length import (
    DescriptionCandidate,
    DescriptionLengthBreakdown,
    DescriptionLengthEngine,
    DescriptionLengthProblem,
)
from pure_integer_ai.cognition.shared.hypothesis import HypothesisKey
from pure_integer_ai.cognition.shared.hypothesis_resolution import (
    PREFERENCE_EQUIVALENT,
    PREFERENCE_LEFT_BETTER,
    PREFERENCE_RIGHT_BETTER,
    ResolverPreference,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int


def _scorer_key(value) -> tuple[int, ...]:
    """校验由调用方图协议注入的描述长度 scorer 键。"""
    if not isinstance(value, tuple) or not value:
        raise ValueError("scorer_key 必须是非空整数 tuple")
    assert_int(*value, _where="DescriptionLengthResolverScorer.scorer_key")
    if any(type(item) is not int for item in value):
        raise ValueError("scorer_key 必须使用严格整数")
    return value


def _breakdown_payload(score: DescriptionLengthBreakdown) -> tuple[int, ...]:
    """完整展开 H-03 数值明细，避免 trace 只保留 total 摘要。"""
    return (
        len(score.problem_key),
        *score.problem_key,
        score.model_cost,
        score.encoded_data_cost,
        score.boundary_cost,
        score.exception_cost,
        score.total_cost,
        score.literal_baseline_cost,
        score.recursive_reuse_gain,
        score.exception_count,
        score.fragment_count,
        score.fragment_reference_count,
        score.recursive_fragment_reference_count,
        score.without_model_cost,
        score.without_exception_cost,
        score.without_boundary_cost,
        score.without_reuse_cost,
    )


class DescriptionLengthResolverScorer:
    """以总 bit 成本低者更优生成完整候选对，不用稳定键打破同分。"""

    def __init__(
            self, scorer_key: tuple[int, ...], *,
            engine: DescriptionLengthEngine,
            problem: DescriptionLengthProblem,
            candidates: tuple[DescriptionCandidate, ...],
            ) -> None:
        """绑定同一 H-03 问题及其完整候选集合。"""
        self.scorer_key = _scorer_key(scorer_key)
        if not isinstance(engine, DescriptionLengthEngine):
            raise TypeError("engine 必须是 DescriptionLengthEngine")
        if not isinstance(problem, DescriptionLengthProblem):
            raise TypeError("problem 必须是 DescriptionLengthProblem")
        if not isinstance(candidates, tuple) or any(
                not isinstance(item, DescriptionCandidate)
                for item in candidates):
            raise TypeError("candidates 只能包含 DescriptionCandidate")
        by_hypothesis = {
            item.model.hypothesis: item for item in candidates
        }
        if len(by_hypothesis) != len(candidates):
            raise ValueError("描述长度 scorer 不得重复绑定同一 Hypothesis")
        self.engine = engine
        self.problem = problem
        self._candidates = by_hypothesis

    def preferences(
            self, hypotheses: tuple[HypothesisKey, ...],
            ) -> tuple[ResolverPreference, ...]:
        """核验并比较 resolver 请求的每个候选对，完整保留双方成本明细。"""
        missing = tuple(
            item for item in hypotheses if item not in self._candidates)
        if missing:
            raise ValueError("描述长度 scorer 缺少 resolver eligible 候选")
        scores = {
            hypothesis: self.engine.score(
                self.problem, self._candidates[hypothesis])
            for hypothesis in hypotheses
        }
        preferences: list[ResolverPreference] = []
        for left_index in range(len(hypotheses)):
            for right_index in range(left_index + 1, len(hypotheses)):
                left = hypotheses[left_index]
                right = hypotheses[right_index]
                left_score = scores[left]
                right_score = scores[right]
                if left_score.total_cost < right_score.total_cost:
                    preference = PREFERENCE_LEFT_BETTER
                elif right_score.total_cost < left_score.total_cost:
                    preference = PREFERENCE_RIGHT_BETTER
                else:
                    preference = PREFERENCE_EQUIVALENT
                left_payload = _breakdown_payload(left_score)
                right_payload = _breakdown_payload(right_score)
                preferences.append(ResolverPreference(
                    self.scorer_key,
                    left,
                    right,
                    preference,
                    (
                        len(left_payload),
                        *left_payload,
                        len(right_payload),
                        *right_payload,
                    ),
                ))
        return tuple(preferences)


__all__ = ["DescriptionLengthResolverScorer"]
