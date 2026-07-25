"""L-05B2A formal round 的 typed generation owner 和请求映射边界。

请求如何从真实课程、当前图和 WorkMemory 形成完全由调用方 mapper 决定。本模块只负责
fail-closed 调度：mapper 没有请求或 typed 规划失败时，不调用旧生成链，也不伪造 surface。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from pure_integer_ai.cognition.shared.generation_execution import (
    TypedGenerationExecution,
    TypedGenerationExecutor,
)
from pure_integer_ai.cognition.shared.generation_plan import (
    GenerationPlanningRequest,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_MINIMAL_INSTRUCTION,
    ObjectIdentity,
)
from pure_integer_ai.cognition.shared.generation_verification import (
    GenerationPostcheckRequest,
)
from pure_integer_ai.cognition.shared.scope_identity import (
    generation_scope,
    query_scope,
)
from pure_integer_ai.cognition.shared.types import InputPayload, ObserveResult
from pure_integer_ai.crosscut.guards.int_blocker import assert_int
from pure_integer_ai.experiments.collection import CollectedItem
from pure_integer_ai.experiments.generation_verification_runtime import (
    GenerationPostcheckRun,
    GenerationPostcheckRuntime,
)
from pure_integer_ai.experiments.train_context import TrainContext


def _packed(key: tuple[int, ...]) -> tuple[int, ...]:
    """为可变长稳定键增加长度边界。"""
    return len(key), *key


def _strict_key(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """核验 mapper trace 使用非空严格整数 tuple。"""
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{label} 必须是非空 tuple")
    assert_int(*value, _where=label)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{label} 必须使用严格整数")
    return value


def _require_instruction(value: ObjectIdentity, *, label: str) -> ObjectIdentity:
    """核验生产映射原因是一等 MinimalInstruction。"""
    if not isinstance(value, ObjectIdentity):
        raise TypeError(f"{label} 必须是 ObjectIdentity")
    if value.object_kind != OBJECT_MINIMAL_INSTRUCTION:
        raise ValueError(f"{label} 必须是 MinimalInstruction")
    return value


@dataclass(frozen=True)
class ProductionGenerationRequestDecision:
    """一次 formal item 到 typed generation request 的显式映射结果。"""

    reason: ObjectIdentity
    trace: tuple[int, ...]
    request: GenerationPlanningRequest | None = None

    def __post_init__(self) -> None:
        _require_instruction(self.reason, label="production generation reason")
        _strict_key(self.trace, label="production generation mapper trace")
        if (self.request is not None
                and not isinstance(self.request, GenerationPlanningRequest)):
            raise TypeError("production generation request 类型错误")

    def stable_key(self) -> tuple[int, ...]:
        """返回映射原因、trace 和可选请求的完整稳定键。"""
        result = [
            *_packed(self.reason.stable_key()),
            *_packed(self.trace),
            0 if self.request is None else 1,
        ]
        if self.request is not None:
            result.extend(_packed(self.request.stable_key()))
        return tuple(result)


class ProductionGenerationRequestMapper(Protocol):
    """把 formal item 的真实 typed 状态映射为 G-00 请求或显式无请求结果。"""

    def build(
            self,
            ctx: TrainContext,
            item: CollectedItem,
            input_payload: InputPayload,
            observation: ObserveResult,
            ) -> ProductionGenerationRequestDecision:
        """返回来源化请求；不得读取 role_seq/token_seq 或 legacy path winner。"""
        ...


class ProductionGenerationPostcheckMapper(Protocol):
    """为已完成的同次 typed execution 显式建立 G-04 复核请求。"""

    def build(
            self,
            ctx: TrainContext,
            item: CollectedItem,
            input_payload: InputPayload,
            observation: ObserveResult,
            execution: TypedGenerationExecution,
            ) -> GenerationPostcheckRequest:
        """返回 attachment、来源和任务要求完整的请求，不读取隐藏全局语义。"""
        ...


@dataclass(frozen=True)
class ProductionGenerationRun:
    """formal round 的请求映射、typed 执行和可选 G-04 复核结果。"""

    decision: ProductionGenerationRequestDecision
    execution: TypedGenerationExecution | None = None
    postcheck: GenerationPostcheckRun | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.decision, ProductionGenerationRequestDecision):
            raise TypeError("production generation decision 类型错误")
        has_request = self.decision.request is not None
        if has_request != (self.execution is not None):
            raise ValueError("production generation request 与 execution 状态不一致")
        if self.execution is not None:
            if not isinstance(self.execution, TypedGenerationExecution):
                raise TypeError("production generation execution 类型错误")
            if self.execution.plan.request != self.decision.request:
                raise ValueError("production generation execution 替换了 mapper 请求")
        if self.postcheck is not None:
            if not isinstance(self.postcheck, GenerationPostcheckRun):
                raise TypeError("production generation postcheck 类型错误")
            if self.execution is None:
                raise ValueError("无 typed execution 时不得携带 postcheck")
            if self.postcheck.request.execution != self.execution:
                raise ValueError("production generation postcheck 替换了同次 execution")

    @property
    def complete(self) -> bool:
        """返回是否形成完整 G-00 至 G-03 surface 和 renderer 结果。"""
        return self.execution is not None and self.execution.complete

    @property
    def representations(self) -> tuple[ObjectIdentity, ...]:
        """返回完整输出的 Representation 序列，失败或无请求时为空。"""
        if self.execution is None:
            return ()
        return self.execution.representations

    @property
    def postcheck_complete(self) -> bool | None:
        """未安装 G-04 时返回 None，否则返回六维复核是否完整通过。"""
        if self.postcheck is None:
            return None
        return self.postcheck.complete

    def stable_key(self) -> tuple[int, ...]:
        """返回 mapper 决策和可选 typed execution 的完整键。"""
        result = [
            *_packed(self.decision.stable_key()),
            0 if self.execution is None else 1,
        ]
        if self.execution is not None:
            result.extend(_packed(self.execution.stable_key()))
        result.append(0 if self.postcheck is None else 1)
        if self.postcheck is not None:
            result.extend(_packed(self.postcheck.stable_key()))
        return tuple(result)


class ProductionGenerationRuntime:
    """在 formal round 内执行注入式请求映射和同一次 typed generation。"""

    def __init__(
            self,
            mapper: ProductionGenerationRequestMapper,
            executor: TypedGenerationExecutor,
            *,
            postcheck_mapper: ProductionGenerationPostcheckMapper | None = None,
            postchecker: GenerationPostcheckRuntime | None = None,
            ) -> None:
        if not hasattr(mapper, "build"):
            raise TypeError("production generation mapper 必须实现 build")
        if not isinstance(executor, TypedGenerationExecutor):
            raise TypeError("production generation executor 类型错误")
        if (postcheck_mapper is None) != (postchecker is None):
            raise ValueError("G-04 mapper 和 runtime 必须同时安装或同时省略")
        if (postcheck_mapper is not None
                and not hasattr(postcheck_mapper, "build")):
            raise TypeError("production postcheck mapper 必须实现 build")
        if (postchecker is not None
                and not isinstance(postchecker, GenerationPostcheckRuntime)):
            raise TypeError("production postchecker 类型错误")
        self._mapper = mapper
        self._executor = executor
        self._postcheck_mapper = postcheck_mapper
        self._postchecker = postchecker

    def run(
            self,
            ctx: TrainContext,
            item: CollectedItem,
            input_payload: InputPayload,
            observation: ObserveResult,
            ) -> ProductionGenerationRun:
        """在 query/generation 生命周期内执行 typed generation，且绝不 fallback。"""
        work_memory = ctx.work_memory
        if not work_memory.episode_active:
            return self._run_unscoped(ctx, item, input_payload, observation)
        parent = work_memory.active_episode_scope
        if parent is None or input_payload.scope_identity != parent:
            raise ValueError("production generation 输入 scope 与当前 episode 不一致")
        query = query_scope(1, parent=parent)
        work_memory.begin_query(query)
        generation = generation_scope(1, parent=query)
        work_memory.begin_generation(generation)
        try:
            return self._run_unscoped(ctx, item, input_payload, observation)
        finally:
            work_memory.end_generation()
            work_memory.end_query()

    def _run_unscoped(
            self,
            ctx: TrainContext,
            item: CollectedItem,
            input_payload: InputPayload,
            observation: ObserveResult,
            ) -> ProductionGenerationRun:
        """执行纯 mapper 和 executor；生命周期由公开入口统一管理。"""
        decision = self._mapper.build(ctx, item, input_payload, observation)
        if not isinstance(decision, ProductionGenerationRequestDecision):
            raise TypeError("production generation mapper 返回类型错误")
        if decision.request is None:
            return ProductionGenerationRun(decision)
        active_query = ctx.work_memory.active_query_scope
        if (active_query is not None
                and decision.request.goal.scope != active_query):
            raise ValueError("production generation request 未绑定当前 query scope")
        execution = self._executor.execute(decision.request)
        postcheck = None
        if self._postchecker is not None:
            request = self._postcheck_mapper.build(
                ctx, item, input_payload, observation, execution)
            if not isinstance(request, GenerationPostcheckRequest):
                raise TypeError("production postcheck mapper 返回类型错误")
            if request.execution != execution:
                raise ValueError("production postcheck request 未绑定同次 execution")
            postcheck = self._postchecker.run(request)
        return ProductionGenerationRun(decision, execution, postcheck)


class ProductionGenerationRuntimeFactory(Protocol):
    """在完整 TrainContext 协议装配后建立当前 run 独占的 generation owner。"""

    def build(self, ctx: TrainContext) -> ProductionGenerationRuntime:
        """返回绑定当前图、关系 owner、planner 和 renderer 的 runtime。"""
        ...


@dataclass(frozen=True)
class ProductionGenerationInstallation:
    """一次 context 装配得到的 generation runtime 和可选 typed stage4 owner。"""

    runtime: ProductionGenerationRuntime
    stage4_runtime: Any = None

    def __post_init__(self) -> None:
        if not isinstance(self.runtime, ProductionGenerationRuntime):
            raise TypeError("production generation installation runtime 类型错误")
        if (self.stage4_runtime is not None
                and (not hasattr(self.stage4_runtime, "apply")
                     or not hasattr(self.stage4_runtime, "state_key"))):
            raise TypeError("production generation stage4 owner 协议不完整")

    def clone_for_evaluation(self) -> "ProductionGenerationRuntimeFactory":
        """返回不共享可变 mapper 或 owner 状态的评测 factory。"""
        ...

    def state_key(self) -> tuple:
        """返回 factory 配置和可变状态的完整可比较键。"""
        ...


def install_production_generation_runtime(
        ctx: TrainContext,
        factory: ProductionGenerationRuntimeFactory,
        ) -> ProductionGenerationRuntime:
    """通过调用方 factory 安装 run-local typed generation owner。"""
    if not isinstance(ctx, TrainContext):
        raise TypeError("ctx 必须是 TrainContext")
    if ctx.language_generation_runtime is not None:
        raise ValueError("TrainContext 已安装 language generation runtime")
    if ctx.language_generation_stage4_runtime is not None:
        raise ValueError("TrainContext 残留未归属的 language stage4 runtime")
    if not hasattr(factory, "build"):
        raise TypeError("language generation runtime factory 必须实现 build")
    if hasattr(factory, "build_installation"):
        installation = factory.build_installation(ctx)
        if not isinstance(installation, ProductionGenerationInstallation):
            raise TypeError("language generation factory installation 类型错误")
        runtime = installation.runtime
        ctx.language_generation_stage4_runtime = installation.stage4_runtime
    else:
        runtime = factory.build(ctx)
        if not isinstance(runtime, ProductionGenerationRuntime):
            raise TypeError("language generation runtime factory 返回类型错误")
    ctx.language_generation_runtime_factory = factory
    ctx.language_generation_runtime = runtime
    return runtime


__all__ = [
    "ProductionGenerationRequestDecision",
    "ProductionGenerationRequestMapper",
    "ProductionGenerationPostcheckMapper",
    "ProductionGenerationInstallation",
    "ProductionGenerationRun",
    "ProductionGenerationRuntime",
    "ProductionGenerationRuntimeFactory",
    "install_production_generation_runtime",
]
