"""把旧 dag_path/PR 运行期视图限制为 S-05 候选来源。

adapter 不解释 REACHED_SINK、PR 值、ConceptRef 或边类型，也不产生验证结果。调用方
必须注入 typed mapper，把启发式视图映射成完整 ReasoningCandidate；候选随后仍由
ReasoningPlanner 的 InferenceVerifier 裁决。
"""
from __future__ import annotations

from typing import Any, Protocol

from pure_integer_ai.cognition.shared.reasoning_planner import (
    ReasoningCandidate,
    ReasoningObligation,
)
from pure_integer_ai.cognition.shared.types import PathResult


def _validate_candidates(
        obligation: ReasoningObligation,
        candidates: tuple[ReasoningCandidate, ...],
        *,
        label: str,
        ) -> tuple[ReasoningCandidate, ...]:
    """核验 mapper 返回完整候选且结论未漂移，不做有效性或 rank 裁决。"""
    if not isinstance(candidates, tuple):
        raise TypeError(f"{label} mapper 必须返回 tuple")
    if any(not isinstance(candidate, ReasoningCandidate)
           for candidate in candidates):
        raise TypeError(f"{label} mapper 返回非法候选")
    if any(candidate.conclusion != obligation for candidate in candidates):
        raise ValueError(f"{label} candidate conclusion 与 obligation 不一致")
    return candidates


class DagPathCandidateMapper(Protocol):
    """由调用方把旧 PathResult 映射为 typed reasoning candidates。"""

    def map_candidates(
            self,
            obligation: ReasoningObligation,
            path_result: PathResult,
            ) -> tuple[ReasoningCandidate, ...]:
        """不得把 terminal/sink 自身解释成逻辑成功。"""
        ...


class DagPathCandidateProvider:
    """只读 query-scoped PathResult 的候选 provider。"""

    def __init__(
            self,
            path_result: PathResult,
            mapper: DagPathCandidateMapper,
            ) -> None:
        if not isinstance(path_result, PathResult):
            raise TypeError("path_result 必须是 PathResult")
        if not hasattr(mapper, "map_candidates"):
            raise TypeError("dag_path mapper 必须实现 map_candidates")
        self._path_result = path_result
        self._mapper = mapper

    def retrieve(
            self,
            obligation: ReasoningObligation,
            ) -> tuple[ReasoningCandidate, ...]:
        """委托 typed mapper 产候选；不读取 terminal、sink 或最短路径。"""
        if not isinstance(obligation, ReasoningObligation):
            raise TypeError("obligation 类型错误")
        return _validate_candidates(
            obligation,
            self._mapper.map_candidates(obligation, self._path_result),
            label="dag_path",
        )


class PRSnapshotSource(Protocol):
    """A3PRWrapper 和同构 query-scoped PR 设施的最小只读接口。"""

    def snapshot(self) -> dict[Any, Any]:
        """返回当前 query 的离散节点到精确/定点值快照。"""
        ...


class PRCandidateMapper(Protocol):
    """由调用方把 PR snapshot 映射为 typed reasoning candidates。"""

    def map_candidates(
            self,
            obligation: ReasoningObligation,
            snapshot: dict[Any, Any],
            ) -> tuple[ReasoningCandidate, ...]:
        """PR 值只可用于候选检索，不可作为规则有效性。"""
        ...


class PRCandidateProvider:
    """从 PR snapshot 取得候选但不把 salience 当 Evidence。"""

    def __init__(
            self,
            source: PRSnapshotSource,
            mapper: PRCandidateMapper,
            ) -> None:
        if not hasattr(source, "snapshot"):
            raise TypeError("PR source 必须实现 snapshot")
        if not hasattr(mapper, "map_candidates"):
            raise TypeError("PR mapper 必须实现 map_candidates")
        self._source = source
        self._mapper = mapper

    def retrieve(
            self,
            obligation: ReasoningObligation,
            ) -> tuple[ReasoningCandidate, ...]:
        """复制当前 snapshot 后交给 typed mapper，禁止 mapper 改写 PR owner 状态。"""
        if not isinstance(obligation, ReasoningObligation):
            raise TypeError("obligation 类型错误")
        snapshot = self._source.snapshot()
        if not isinstance(snapshot, dict):
            raise TypeError("PR snapshot 必须是 dict")
        return _validate_candidates(
            obligation,
            self._mapper.map_candidates(obligation, dict(snapshot)),
            label="PR",
        )


__all__ = [
    "DagPathCandidateMapper",
    "DagPathCandidateProvider",
    "PRCandidateMapper",
    "PRCandidateProvider",
    "PRSnapshotSource",
]
