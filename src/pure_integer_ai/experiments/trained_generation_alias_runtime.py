"""按完整协议身份组织冻结训练图的局部 alias 路由。"""
from __future__ import annotations

from typing import Type

from pure_integer_ai.cognition.shared.alias_resolution import (
    AliasResolutionProtocol,
)
from pure_integer_ai.experiments.alias_relation_runtime import (
    AliasRelationRuntime,
)


class _ProtocolFamilyView:
    """Expose one canonical protocol plus prefix steps from every profile."""

    def __init__(self, protocols: tuple[AliasResolutionProtocol, ...]) -> None:
        self._canonical = protocols[0]
        self._surface_prefix_steps = tuple(sorted({
            step for protocol in protocols
            for step in protocol.surface_prefix_steps()
        }, key=lambda item: item.stable_key()))

    def surface_prefix_steps(self):
        return self._surface_prefix_steps

    def __getattr__(self, name: str):
        return getattr(self._canonical, name)


class _SelectorView:
    def __init__(self, protocols: tuple[AliasResolutionProtocol, ...]) -> None:
        self.protocol = _ProtocolFamilyView(protocols)


class ProfiledAliasRelationRuntime(AliasRelationRuntime):
    """以完整协议键固定规范序，并核对全部适用 profile 的词形冲突。"""

    def __init__(
            self,
            runtimes: tuple[AliasRelationRuntime, ...],
            *,
            conflict_error: Type[Exception] = RuntimeError,
            ) -> None:
        """仅组织协议元数据；不为规范排序读取与当前输入无关的关系事实。"""
        if (type(runtimes) is not tuple or not runtimes
                or any(not isinstance(item, AliasRelationRuntime)
                       for item in runtimes)):
            raise TypeError("profiled alias runtimes 类型错误")
        if not isinstance(conflict_error, type) or not issubclass(
                conflict_error, Exception):
            raise TypeError("profile conflict error must be an Exception type")
        self._conflict_error = conflict_error
        by_protocol = {}
        for runtime in runtimes:
            by_protocol.setdefault(
                runtime.selector.protocol.stable_key(), runtime)
        self._profiled_runtimes = tuple(
            by_protocol[key] for key in sorted(by_protocol))
        protocols = tuple(
            item.selector.protocol for item in self._profiled_runtimes)
        canonical = self._profiled_runtimes[0]
        super().__init__(canonical.closure, canonical.selector)
        self.selector = _SelectorView(protocols)
        self._runtime_by_protocol = {
            runtime.selector.protocol.stable_key(): runtime
            for runtime in self._profiled_runtimes
        }

    @staticmethod
    def _option_signature(proposal) -> tuple:
        return tuple(item.value.stable_key()
                     for item in proposal.result.options)

    def _choose_preview(self, proposals: tuple):
        available = tuple(item for item in proposals if item.result.options)
        if not available:
            return proposals[0]
        signatures = {self._option_signature(item) for item in available}
        if len(signatures) != 1:
            raise self._conflict_error(
                "R-01 protocol profiles 产生冲突 Representation")
        return available[0]

    def preview_surface(
            self, origin, branch, *, budget,
            allowed_prefix_steps=None, expected_value=None,
            ):
        eligible = tuple(
            runtime for runtime in self._profiled_runtimes
            if (allowed_prefix_steps is None or set(allowed_prefix_steps).issubset(
                runtime.selector.protocol.surface_prefix_steps()))
        )
        if not eligible:
            raise self._conflict_error(
                "R-01 protocol profiles 不支持请求的 surface prefix")
        return self._choose_preview(tuple(
            runtime.preview_surface(
                origin, branch, budget=budget,
                allowed_prefix_steps=allowed_prefix_steps,
                expected_value=expected_value,
            )
            for runtime in eligible
        ))

    def preview_reference(self, origin, *, target_kinds, budget):
        return self._choose_preview(tuple(
            runtime.preview_reference(
                origin, target_kinds=target_kinds, budget=budget)
            for runtime in self._profiled_runtimes
        ))

    def commit_many(self, requests, *, context=None):
        if type(requests) is not tuple or not requests:
            raise ValueError("alias commit_many requests 必须是非空 tuple")
        grouped: dict[tuple[int, ...], list[tuple[int, tuple]]] = {}
        for index, request in enumerate(requests):
            if not isinstance(request, tuple) or len(request) not in (2, 3):
                raise TypeError("alias commit request 类型错误")
            protocol = getattr(request[0], "result", None)
            protocol = getattr(protocol, "protocol", None)
            key = (protocol.stable_key()
                   if isinstance(protocol, AliasResolutionProtocol) else None)
            if key not in self._runtime_by_protocol:
                raise ValueError("alias proposal 使用了未知 R-01 protocol")
            grouped.setdefault(key, []).append((index, request))

        results = [None] * len(requests)
        for key, items in grouped.items():
            committed = self._runtime_by_protocol[key].commit_many(
                tuple(request for _, request in items), context=context)
            for (index, _), use in zip(items, committed, strict=True):
                results[index] = use
        if any(item is None for item in results):
            raise RuntimeError("R-01 profile 采用结果缺失")
        return tuple(results)

    def clone_for_runtime(self, closure):
        if len(self._profiled_runtimes) != 1:
            raise ValueError(
                "multi-profile alias clone requires one closure per profile")
        return self._profiled_runtimes[0].clone_for_runtime(closure)

    def state_key(self) -> tuple:
        return tuple(
            (runtime.selector.protocol.stable_key(), runtime.state_key())
            for runtime in self._profiled_runtimes
        )


__all__ = ["ProfiledAliasRelationRuntime"]
