"""把同一次 G-00 至 G-03 规划保留为可渲染的 typed 执行结果。

本模块不构造语义请求，也不接触旧 DAG path、PRECEDES、role_seq 或 token_seq。
它只核验最终 layer 携带的实际 G-03 artifact 与稳定 payload 一致，并在完整成功时
调用注入 renderer 产生边界输出单元。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pure_integer_ai.cognition.shared.generation_plan import (
    GenerationPlan,
    GenerationPlanner,
    GenerationPlanningRequest,
)
from pure_integer_ai.cognition.shared.generation_surface import (
    GenerationSurfacePlan,
    GenerationSurfacePreview,
)
from pure_integer_ai.cognition.shared.identity import ObjectIdentity
from pure_integer_ai.cognition.shared.representation_rendering import (
    RenderedSurface,
    RepresentationSequenceRenderer,
    render_generation_preview,
    render_generation_surface,
)


def _packed(key: tuple[int, ...]) -> tuple[int, ...]:
    """为可变长稳定键增加长度边界。"""
    return len(key), *key


class GenerationSurfaceCommitter(Protocol):
    """把已渲染成功的完整 G-03 preview 原子提交为 surface plan。"""

    def commit(
            self, preview: GenerationSurfacePreview,
            ) -> GenerationSurfacePlan:
        """提交同一 preview 的全部 R-01 proposal，并返回实际采用 trace。"""
        ...


@dataclass(frozen=True)
class TypedGenerationExecution:
    """保存完整 generation plan、实际 surface artifact 和可选渲染结果。"""

    plan: GenerationPlan
    preview: GenerationSurfacePreview | None = None
    surface: GenerationSurfacePlan | None = None
    rendered: RenderedSurface | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.plan, GenerationPlan):
            raise TypeError("typed generation execution plan 类型错误")
        final = self.plan.layers[-1]
        if self.plan.complete:
            if not isinstance(self.surface, GenerationSurfacePlan):
                raise ValueError("完整 generation plan 必须保留实际 surface plan")
            if self.preview != self.surface.preview:
                raise ValueError("typed generation preview 未绑定实际 surface plan")
            if not isinstance(self.rendered, RenderedSurface):
                raise ValueError("完整 generation plan 必须保留 renderer 结果")
            if self.rendered.representations != self.surface.representations:
                raise ValueError("renderer 结果替换或重排了 surface Representation")
            if final.artifact not in (self.preview, self.surface):
                raise ValueError("G-00 最终 layer 未保留本次 surface preview 或 plan")
            return
        if self.surface is not None or self.rendered is not None:
            raise ValueError("失败 generation plan 不得伪造完整 surface 或渲染结果")
        if self.preview is not None:
            if not isinstance(self.preview, GenerationSurfacePreview):
                raise TypeError("typed generation preview 类型错误")
            if final.artifact != self.preview:
                raise ValueError("失败 surface preview 不是最终 layer 的实际 artifact")
        elif final.artifact is not None:
            raise ValueError("非 surface 失败不得遗留未分类 generation artifact")

    @property
    def complete(self) -> bool:
        """返回六层计划、surface 和 renderer 是否完整成功。"""
        return self.plan.complete

    @property
    def representations(self) -> tuple[ObjectIdentity, ...]:
        """返回成功 surface 的 Representation 序列，失败时返回空 tuple。"""
        return () if self.surface is None else self.surface.representations

    def stable_key(self) -> tuple[int, ...]:
        """返回计划、surface preview/plan 与渲染结果的完整确定性键。"""
        result = [*_packed(self.plan.stable_key())]
        for artifact in (self.preview, self.surface, self.rendered):
            result.append(0 if artifact is None else 1)
            if artifact is not None:
                result.extend(_packed(artifact.stable_key()))
        return tuple(result)


class TypedGenerationExecutor:
    """执行 G-00 planner，并从最终 layer 的同次 artifact 产生表示渲染结果。"""

    def __init__(
            self,
            planner: GenerationPlanner,
            renderer: RepresentationSequenceRenderer,
            committer: GenerationSurfaceCommitter | None = None,
            ) -> None:
        if not isinstance(planner, GenerationPlanner):
            raise TypeError("typed generation executor planner 类型错误")
        if not hasattr(renderer, "render"):
            raise TypeError("typed generation executor renderer 必须实现 render")
        if committer is not None and not hasattr(committer, "commit"):
            raise TypeError("typed generation executor committer 必须实现 commit")
        self._planner = planner
        self._renderer = renderer
        self._committer = committer

    def execute(
            self,
            request: GenerationPlanningRequest,
            ) -> TypedGenerationExecution:
        """执行一次完整计划；失败只返回 typed 失败，不调用 renderer 或旧生成链。"""
        plan = self._planner.plan(request)
        artifact = plan.layers[-1].artifact
        if isinstance(artifact, GenerationSurfacePlan):
            rendered = render_generation_surface(artifact, self._renderer)
            return TypedGenerationExecution(
                plan,
                artifact.preview,
                artifact,
                rendered,
            )
        if isinstance(artifact, GenerationSurfacePreview):
            if artifact.complete:
                if self._committer is None:
                    raise RuntimeError("完整 surface preview 缺注入 committer")
                rendered = render_generation_preview(artifact, self._renderer)
                surface = self._committer.commit(artifact)
                if not isinstance(surface, GenerationSurfacePlan):
                    raise TypeError("surface committer 返回类型错误")
                if surface.preview != artifact:
                    raise ValueError("surface committer 替换了已渲染 preview")
                return TypedGenerationExecution(
                    plan,
                    artifact,
                    surface,
                    rendered,
                )
            return TypedGenerationExecution(plan, artifact)
        return TypedGenerationExecution(plan)


__all__ = [
    "GenerationSurfaceCommitter",
    "TypedGenerationExecution",
    "TypedGenerationExecutor",
]
