"""typed language generation 的只读分维 H2 校准报告。"""
from __future__ import annotations

import copy
from dataclasses import dataclass

from pure_integer_ai.cognition.shared.types import Episode
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.experiments.evaluation_protocol import (
    EvaluationAssignment,
    EvaluationDataIdentity,
    EvaluationProtocolError,
    ProtocolKey,
)
from pure_integer_ai.experiments.language_generation_episode import (
    TypedLanguageEpisode,
    TypedLanguageRewardSignal,
)
from pure_integer_ai.experiments.round_runtime import (
    RoundRunner,
    _run_runner_episodes,
)
from pure_integer_ai.experiments.train_context import TrainContext
from pure_integer_ai.experiments.verification_orchestration import (
    APPLICABILITY_APPLICABLE,
    APPLICABILITY_NOT_APPLICABLE,
    APPLICABILITY_UNKNOWN,
    VERDICT_CONFLICTED,
    VERDICT_REFUTE,
    VERDICT_SUPPORT,
    VERDICT_UNKNOWN,
)
from pure_integer_ai.training.stages import STAGE3_REWARD

_VALID_APPLICABILITY = frozenset({
    APPLICABILITY_NOT_APPLICABLE,
    APPLICABILITY_APPLICABLE,
    APPLICABILITY_UNKNOWN,
})
_VALID_VERDICTS = frozenset({
    VERDICT_SUPPORT,
    VERDICT_REFUTE,
    VERDICT_UNKNOWN,
    VERDICT_CONFLICTED,
})


@dataclass(frozen=True)
class TypedLanguageH2Expectation:
    """声明一个 generation 维度在独立开发样本上的预期结果。"""

    dimension: ProtocolKey
    verifier: ProtocolKey
    applicability: int
    verdict: int

    def __post_init__(self) -> None:
        if not isinstance(self.dimension, ProtocolKey):
            raise TypeError("H2 expectation dimension 必须是 ProtocolKey")
        if not isinstance(self.verifier, ProtocolKey):
            raise TypeError("H2 expectation verifier 必须是 ProtocolKey")
        assert_int(
            self.applicability,
            self.verdict,
            _where="TypedLanguageH2Expectation",
        )
        if self.applicability not in _VALID_APPLICABILITY:
            raise ValueError("H2 expectation applicability 未注册")
        if self.verdict not in _VALID_VERDICTS:
            raise ValueError("H2 expectation verdict 未注册")
        if (self.applicability != APPLICABILITY_APPLICABLE
                and self.verdict != VERDICT_UNKNOWN):
            raise ValueError("非 applicable 的 H2 期望只能保持 unknown verdict")


@dataclass(frozen=True)
class TypedLanguageH2Case:
    """把一个完整 V-00 数据身份绑定到全部分维预期。"""

    identity: EvaluationDataIdentity
    expectations: tuple[TypedLanguageH2Expectation, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.identity, EvaluationDataIdentity):
            raise TypeError("H2 case identity 类型错误")
        if not isinstance(self.expectations, tuple) or not self.expectations:
            raise ValueError("H2 case 必须声明非空分维预期")
        if any(not isinstance(item, TypedLanguageH2Expectation)
               for item in self.expectations):
            raise TypeError("H2 case expectations 类型错误")
        dimensions = [item.dimension for item in self.expectations]
        if len(set(dimensions)) != len(dimensions):
            raise ValueError("同一 H2 case 不得重复 dimension")


@dataclass(frozen=True)
class TypedLanguageH2Protocol:
    """保存版本化 development split 校准集，不从实际输出生成标签。"""

    version: int
    cases: tuple[TypedLanguageH2Case, ...]

    def __post_init__(self) -> None:
        assert_int(self.version, _where="TypedLanguageH2Protocol.version")
        if type(self.version) is not int or self.version <= 0:
            raise ValueError("typed H2 protocol version 必须为正严格整数")
        if not isinstance(self.cases, tuple) or not self.cases:
            raise ValueError("typed H2 protocol cases 不能为空")
        if any(not isinstance(item, TypedLanguageH2Case)
               for item in self.cases):
            raise TypeError("typed H2 protocol cases 类型错误")
        keys = [item.identity.lookup_key() for item in self.cases]
        if len(set(keys)) != len(keys):
            raise ValueError("typed H2 protocol 不得重复数据身份")

    def case_for(self, assignment: EvaluationAssignment) -> TypedLanguageH2Case:
        """按完整来源和内容读取唯一校准 case。"""
        matches = [
            item for item in self.cases
            if item.identity == assignment.identity
        ]
        if len(matches) != 1:
            raise EvaluationProtocolError("development assignment 缺少唯一 typed H2 case")
        return matches[0]


@dataclass(frozen=True)
class TypedLanguageH2DimensionResult:
    """保存一个维度的独立期望和实际信号，不形成综合分数。"""

    dimension: ProtocolKey
    expected: TypedLanguageH2Expectation | None
    actual: TypedLanguageRewardSignal | None

    @property
    def matched(self) -> bool:
        """仅在完整协议字段相等且无运行失败时返回真。"""
        return (
            self.expected is not None
            and self.actual is not None
            and self.actual.operational_failure is None
            and self.actual.dimension == self.expected.dimension
            and self.actual.verifier == self.expected.verifier
            and self.actual.applicability == self.expected.applicability
            and self.actual.verdict == self.expected.verdict
        )


@dataclass(frozen=True)
class TypedLanguageH2CaseResult:
    """保存一个开发样本的 typed episode、逐维比较和协议失败。"""

    identity: EvaluationDataIdentity
    episode: TypedLanguageEpisode | None
    dimensions: tuple[TypedLanguageH2DimensionResult, ...]
    failure: str | None = None

    @property
    def complete(self) -> bool:
        """返回该样本是否只有精确匹配的预期维度。"""
        return (
            self.failure is None
            and bool(self.dimensions)
            and all(item.matched for item in self.dimensions)
        )


@dataclass(frozen=True)
class TypedLanguageH2Report:
    """汇总 development split 的独立样本结果，不暴露总分或权重。"""

    protocol_version: int
    split: ProtocolKey
    cases: tuple[TypedLanguageH2CaseResult, ...]

    @property
    def measured(self) -> bool:
        """返回是否实际运行了至少一个开发样本。"""
        return bool(self.cases)

    @property
    def complete(self) -> bool:
        """返回全部样本是否逐维完整匹配，仅作校准报告状态。"""
        return self.measured and all(item.complete for item in self.cases)


def _compare_h2_case(
        case: TypedLanguageH2Case,
        episode: TypedLanguageEpisode | None,
        *,
        failure: str | None = None,
        ) -> TypedLanguageH2CaseResult:
    """按期望序比较全部维度，并把额外实际维度显式保留为失败项。"""
    expected_by_dimension = {
        item.dimension: item for item in case.expectations}
    actual_signals = () if episode is None else episode.signals
    actual_by_dimension = {
        item.dimension: item for item in actual_signals}
    dimensions = [
        TypedLanguageH2DimensionResult(
            expected.dimension,
            expected,
            actual_by_dimension.get(expected.dimension),
        )
        for expected in case.expectations
    ]
    for actual in actual_signals:
        if actual.dimension not in expected_by_dimension:
            dimensions.append(TypedLanguageH2DimensionResult(
                actual.dimension,
                None,
                actual,
            ))
    if episode is not None:
        if episode.source != case.identity.source_ref:
            failure = failure or "typed H2 episode 来源与 V-00 assignment 不一致"
        elif not episode.read_only:
            failure = failure or "typed H2 episode 未标记为只读评测"
    return TypedLanguageH2CaseResult(
        case.identity,
        episode,
        tuple(dimensions),
        failure,
    )


def run_typed_language_split(
        ctx: TrainContext,
        runner: RoundRunner,
        cases: tuple[TypedLanguageH2Case, ...],
        split: ProtocolKey,
        *,
        label_prefix: str,
        version: int,
        ) -> tuple[TypedLanguageH2CaseResult, ...]:
    """逐项隔离运行指定 V-00 split，并核验唯一 typed episode。"""
    from pure_integer_ai.experiments.evaluation_isolation import (
        isolated_evaluation,
    )
    from pure_integer_ai.experiments.evaluation_runtime import (
        _evaluation_item_for,
    )

    if ctx.evaluation_plan is None or not ctx.evaluation_strictly_isolated:
        raise EvaluationProtocolError("typed language split 评测要求严格 V-00 计划")
    plan = ctx.evaluation_plan
    if split not in plan.protocol.split_keys():
        raise EvaluationProtocolError("typed language 评测 split 未注册")
    assignments = tuple(
        item for item in plan.assignments if item.split == split)
    assignment_identities = {item.identity for item in assignments}
    case_identities = {item.identity for item in cases}
    if assignment_identities != case_identities:
        raise EvaluationProtocolError("typed evaluation cases 必须精确覆盖目标 split")

    case_by_identity = {item.identity: item for item in cases}

    results = []
    for case_index, assignment in enumerate(assignments):
        case = case_by_identity[assignment.identity]
        label = f"{label_prefix}-{version}-{case_index}"
        with isolated_evaluation(ctx, label=label) as eval_ctx:
            item = copy.deepcopy(_evaluation_item_for(eval_ctx, assignment))
            episodes = _run_runner_episodes(
                eval_ctx,
                runner,
                item,
                STAGE3_REWARD,
                case_index,
            )
        typed = [
            item for item in episodes
            if isinstance(item, TypedLanguageEpisode)]
        legacy = [item for item in episodes if isinstance(item, Episode)]
        failure = None
        if legacy:
            failure = "typed H2 收到 legacy scalar episode"
        elif len(typed) != 1:
            failure = "typed H2 必须为每个开发样本产生唯一 typed episode"
        episode = typed[0] if len(typed) == 1 else None
        results.append(_compare_h2_case(
            case,
            episode,
            failure=failure,
        ))
    return tuple(results)


def run_typed_language_h2(
        ctx: TrainContext,
        runner: RoundRunner,
        protocol: TypedLanguageH2Protocol,
        ) -> TypedLanguageH2Report:
    """逐项隔离运行 development split，并返回不写正式状态的分维报告。"""
    if not isinstance(protocol, TypedLanguageH2Protocol):
        raise TypeError("typed language H2 protocol 类型错误")
    if ctx.evaluation_plan is None:
        raise EvaluationProtocolError("typed language H2 要求严格 V-00 计划")
    split = ctx.evaluation_plan.protocol.development_split
    results = run_typed_language_split(
        ctx,
        runner,
        protocol.cases,
        split,
        label_prefix="typed-language-h2",
        version=protocol.version,
    )
    return TypedLanguageH2Report(
        protocol.version,
        split,
        results,
    )


__all__ = [
    "TypedLanguageH2Case",
    "TypedLanguageH2CaseResult",
    "TypedLanguageH2DimensionResult",
    "TypedLanguageH2Expectation",
    "TypedLanguageH2Protocol",
    "TypedLanguageH2Report",
    "run_typed_language_h2",
    "run_typed_language_split",
]
