"""Representation 序列的注入式边界渲染合约。

G-03 核心只产生一等 Representation 概念序列。本模块允许宿主按表示族注入
具体 renderer；它不补空格、标点、边界词或任何语言规则。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pure_integer_ai.cognition.shared.generation_surface import (
    GenerationSurfacePlan,
    GenerationSurfacePreview,
)
from pure_integer_ai.cognition.shared.identity import (
    OBJECT_MINIMAL_INSTRUCTION,
    OBJECT_REPRESENTATION,
    ObjectIdentity,
)
from pure_integer_ai.cognition.shared.unicode_representation import (
    validate_unicode_scalars,
)
from pure_integer_ai.crosscut.guards.int_blocker import assert_int


def _packed(key: tuple[int, ...]) -> tuple[int, ...]:
    """为可变长稳定键增加长度边界。"""
    return len(key), *key


def _strict_key(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """核验表示族、输出单元和 trace 使用非空严格整数 tuple。"""
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{label} 必须是非空 tuple")
    assert_int(*value, _where=label)
    if any(type(item) is not int for item in value):
        raise ValueError(f"{label} 必须使用严格整数")
    return value


def representation_parts(
        representation: ObjectIdentity,
        ) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """按权威 identity schema 拆出 Representation 的 family 和 content 键。"""
    if not isinstance(representation, ObjectIdentity):
        raise TypeError("representation 必须是 ObjectIdentity")
    if representation.object_kind != OBJECT_REPRESENTATION:
        raise ValueError("对象不是 Representation")
    components = representation.components
    if len(components) < 4:
        raise ValueError("Representation identity components 已截断")
    family_size = components[0]
    if family_size <= 0:
        raise ValueError("Representation family 不能为空")
    content_size_index = 1 + family_size
    if content_size_index >= len(components):
        raise ValueError("Representation identity 缺 content 长度")
    family = components[1:content_size_index]
    content_size = components[content_size_index]
    content = components[content_size_index + 1:]
    if content_size <= 0 or len(content) != content_size:
        raise ValueError("Representation content 长度与 identity 不一致")
    return tuple(family), tuple(content)


@dataclass(frozen=True)
class RenderedSurface:
    """一个 renderer 对完整 Representation 序列产生的整数输出单元。"""

    renderer: ObjectIdentity
    representations: tuple[ObjectIdentity, ...]
    units: tuple[int, ...]
    trace: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.renderer, ObjectIdentity):
            raise TypeError("rendered surface renderer 类型错误")
        if self.renderer.object_kind != OBJECT_MINIMAL_INSTRUCTION:
            raise ValueError("rendered surface renderer 必须是 MinimalInstruction")
        if not isinstance(self.representations, tuple) or not self.representations:
            raise ValueError("rendered surface 必须保留非空 Representation 序列")
        if any(
                not isinstance(item, ObjectIdentity)
                or item.object_kind != OBJECT_REPRESENTATION
                for item in self.representations):
            raise ValueError("rendered surface 含非 Representation 对象")
        _strict_key(self.units, label="rendered surface units")
        _strict_key(self.trace, label="rendered surface trace")

    def stable_key(self) -> tuple[int, ...]:
        """返回 renderer、输入 Representation、输出单元和 trace 完整键。"""
        result = [
            *_packed(self.renderer.stable_key()),
            len(self.representations),
        ]
        for representation in self.representations:
            result.extend(_packed(representation.stable_key()))
        result.extend(_packed(self.units))
        result.extend(_packed(self.trace))
        return tuple(result)


class RepresentationSequenceRenderer(Protocol):
    """把一种受支持表示族的有序概念序列渲染为宿主输出单元。"""

    def render(
            self,
            representations: tuple[ObjectIdentity, ...],
            ) -> RenderedSurface:
        """按输入顺序渲染，不插入计划之外的语言内容。"""
        ...


def _render_representations(
        representations: tuple[ObjectIdentity, ...],
        renderer: RepresentationSequenceRenderer,
        ) -> RenderedSurface:
    """渲染已核验的 Representation 序列，并守住对象及顺序同一性。"""
    if not hasattr(renderer, "render"):
        raise TypeError("surface renderer 必须实现 render")
    rendered = renderer.render(representations)
    if not isinstance(rendered, RenderedSurface):
        raise TypeError("surface renderer 返回类型错误")
    if rendered.representations != representations:
        raise ValueError("surface renderer 替换或重排了 Representation")
    return rendered


def render_generation_preview(
        preview: GenerationSurfacePreview,
        renderer: RepresentationSequenceRenderer,
        ) -> RenderedSurface:
    """渲染未提交的完整 G-03 preview，不触碰任何 R-01 采用账。"""
    if not isinstance(preview, GenerationSurfacePreview):
        raise TypeError("render_generation_preview preview 类型错误")
    if not preview.complete:
        raise ValueError("失败 surface preview 不得渲染部分 Representation")
    return _render_representations(preview.representations, renderer)


def render_generation_surface(
        plan: GenerationSurfacePlan,
        renderer: RepresentationSequenceRenderer,
        ) -> RenderedSurface:
    """渲染完整 G-03 plan，并核验 renderer 没有替换或重排表示概念。"""
    if not isinstance(plan, GenerationSurfacePlan):
        raise TypeError("render_generation_surface plan 类型错误")
    return _render_representations(plan.representations, renderer)


class UnicodeRepresentationRenderer:
    """仅解码调用方注入 family 的 Unicode scalar sequence Representation。"""

    def __init__(
            self,
            family_key: tuple[int, ...],
            renderer: ObjectIdentity,
            ) -> None:
        self._family_key = _strict_key(
            family_key, label="Unicode renderer family key")
        if not isinstance(renderer, ObjectIdentity):
            raise TypeError("Unicode renderer identity 类型错误")
        if renderer.object_kind != OBJECT_MINIMAL_INSTRUCTION:
            raise ValueError("Unicode renderer identity 必须是 MinimalInstruction")
        self._renderer = renderer

    def render(
            self,
            representations: tuple[ObjectIdentity, ...],
            ) -> RenderedSurface:
        """按原序连接各 Unicode payload，不自动补任何分隔或标点。"""
        if not isinstance(representations, tuple) or not representations:
            raise ValueError("Unicode renderer 需要非空 Representation 序列")
        units: list[int] = []
        trace = [*_packed(self._family_key), len(representations)]
        for representation in representations:
            family, content = representation_parts(representation)
            if family != self._family_key:
                raise ValueError("Unicode renderer 遇到未注册表示族")
            units.extend(validate_unicode_scalars(content))
            trace.extend(_packed(representation.stable_key()))
        return RenderedSurface(
            self._renderer,
            representations,
            tuple(units),
            tuple(trace),
        )

    def text(self, rendered: RenderedSurface) -> str:
        """把本 renderer 已核验的 Unicode scalar 输出单元转换为宿主字符串。"""
        if not isinstance(rendered, RenderedSurface):
            raise TypeError("Unicode rendered surface 类型错误")
        if rendered.renderer != self._renderer:
            raise ValueError("rendered surface 来自其他 renderer")
        expected = self.render(rendered.representations)
        if rendered != expected:
            raise ValueError("rendered surface units/trace 与 Representation 不一致")
        return "".join(chr(item) for item in rendered.units)


__all__ = [
    "RenderedSurface",
    "RepresentationSequenceRenderer",
    "UnicodeRepresentationRenderer",
    "render_generation_preview",
    "render_generation_surface",
    "representation_parts",
]
