"""typed language generation 的 held-out 分维 floor。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.experiments.evaluation_protocol import ProtocolKey
from pure_integer_ai.experiments.language_generation_h2 import (
    TypedLanguageH2Case,
    TypedLanguageH2CaseResult,
    run_typed_language_split,
)
from pure_integer_ai.experiments.round_runtime import RoundRunner
from pure_integer_ai.experiments.train_context import TrainContext


@dataclass(frozen=True)
class TypedLanguageFloorRequirement:
    """声明一个维度必须使用的 verifier 和最低匹配率。"""

    dimension: ProtocolKey
    verifier: ProtocolKey
    minimum_match_permille: int

    def __post_init__(self) -> None:
        if not isinstance(self.dimension, ProtocolKey):
            raise TypeError("floor dimension 必须是 ProtocolKey")
        if not isinstance(self.verifier, ProtocolKey):
            raise TypeError("floor verifier 必须是 ProtocolKey")
        assert_int(
            self.minimum_match_permille,
            _where="TypedLanguageFloorRequirement",
        )
        if (type(self.minimum_match_permille) is not int
                or not 1 <= self.minimum_match_permille <= 1000):
            raise ValueError("floor 最低匹配率必须位于 1..1000")


@dataclass(frozen=True)
class TypedLanguageFloorProtocol:
    """绑定完整 held-out case 与逐维注入 floor。"""

    version: int
    cases: tuple[TypedLanguageH2Case, ...]
    requirements: tuple[TypedLanguageFloorRequirement, ...]

    def __post_init__(self) -> None:
        assert_int(self.version, _where="TypedLanguageFloorProtocol.version")
        if type(self.version) is not int or self.version <= 0:
            raise ValueError("typed floor version 必须为正严格整数")
        if not isinstance(self.cases, tuple) or not self.cases:
            raise ValueError("typed floor cases 不能为空")
        if not isinstance(self.requirements, tuple) or not self.requirements:
            raise ValueError("typed floor requirements 不能为空")
        if any(not isinstance(item, TypedLanguageH2Case)
               for item in self.cases):
            raise TypeError("typed floor cases 类型错误")
        if any(not isinstance(item, TypedLanguageFloorRequirement)
               for item in self.requirements):
            raise TypeError("typed floor requirements 类型错误")
        dimensions = [item.dimension for item in self.requirements]
        if len(set(dimensions)) != len(dimensions):
            raise ValueError("typed floor requirement 不得重复 dimension")
        requirement_by_dimension = {
            item.dimension: item for item in self.requirements}
        expected_dimensions = set(requirement_by_dimension)
        for case in self.cases:
            expectations = {
                item.dimension: item for item in case.expectations}
            if set(expectations) != expected_dimensions:
                raise ValueError("每个 floor case 必须完整覆盖全部 requirement")
            if any(
                    expectations[dimension].verifier
                    != requirement.verifier
                    for dimension, requirement
                    in requirement_by_dimension.items()):
                raise ValueError("floor case verifier 与 requirement 漂移")


@dataclass(frozen=True)
class TypedLanguageFloorDimensionResult:
    """保存一个维度的样本计数、匹配率和独立 floor 结论。"""

    requirement: TypedLanguageFloorRequirement
    total: int
    matched: int
    missing: int
    operational_failure: int

    @property
    def match_permille(self) -> int:
        """返回该维度的纯整数匹配率。"""
        return self.matched * 1000 // max(self.total, 1)

    @property
    def complete(self) -> bool:
        """按本维独立阈值判断，不与其他维度求均值。"""
        return (
            self.total > 0
            and self.operational_failure == 0
            and self.match_permille
            >= self.requirement.minimum_match_permille
        )


@dataclass(frozen=True)
class TypedLanguageFloorReport:
    """保存全部 held-out 样本和逐维 floor 合取结果。"""

    protocol_version: int
    split: ProtocolKey
    cases: tuple[TypedLanguageH2CaseResult, ...]
    dimensions: tuple[TypedLanguageFloorDimensionResult, ...]
    unexpected_dimensions: int

    @property
    def measured(self) -> bool:
        """返回是否实际测量了 held-out case。"""
        return bool(self.cases) and bool(self.dimensions)

    @property
    def complete(self) -> bool:
        """要求结构协议无失败且每个维度各自过 floor。"""
        return (
            self.measured
            and self.unexpected_dimensions == 0
            and all(item.failure is None for item in self.cases)
            and all(item.complete for item in self.dimensions)
        )


def run_typed_language_floor(
        ctx: TrainContext,
        runner: RoundRunner,
        protocol: TypedLanguageFloorProtocol,
        ) -> TypedLanguageFloorReport:
    """逐项隔离运行 held-out split，并按维度独立核验注入 floor。"""
    if not isinstance(protocol, TypedLanguageFloorProtocol):
        raise TypeError("typed language floor protocol 类型错误")
    if ctx.evaluation_plan is None:
        raise ValueError("typed language floor 要求 V-00 evaluation plan")
    split = ctx.evaluation_plan.protocol.held_out_split
    cases = run_typed_language_split(
        ctx,
        runner,
        protocol.cases,
        split,
        label_prefix="typed-language-floor",
        version=protocol.version,
    )
    dimension_results = []
    for requirement in protocol.requirements:
        comparisons = tuple(
            comparison
            for case in cases
            for comparison in case.dimensions
            if comparison.dimension == requirement.dimension
            and comparison.expected is not None
        )
        dimension_results.append(TypedLanguageFloorDimensionResult(
            requirement,
            len(comparisons),
            sum(int(item.matched) for item in comparisons),
            sum(int(item.actual is None) for item in comparisons),
            sum(int(
                item.actual is not None
                and item.actual.operational_failure is not None
            ) for item in comparisons),
        ))
    unexpected = sum(
        1 for case in cases for item in case.dimensions
        if item.expected is None)
    return TypedLanguageFloorReport(
        protocol.version,
        split,
        cases,
        tuple(dimension_results),
        unexpected,
    )


__all__ = [
    "TypedLanguageFloorDimensionResult",
    "TypedLanguageFloorProtocol",
    "TypedLanguageFloorReport",
    "TypedLanguageFloorRequirement",
    "run_typed_language_floor",
]
