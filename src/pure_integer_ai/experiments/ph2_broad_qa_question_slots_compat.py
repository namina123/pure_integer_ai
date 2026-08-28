"""公开问式 artifact 的历史坐标适配。

这里保留现有问式资料的宿主兼容行为（固定 artifact 使用的外部转换器和
歧义上下文），不参与图内语言关系、词形学习或查询候选生成。运行时有图
surface provider 时，核心槽位算法不会调用本模块的坐标投影。
"""
from __future__ import annotations

from opencc import OpenCC


_TO_SIMPLIFIED = OpenCC("t2s")


def aligned_surface(value: str) -> str:
    """为既有公开问式表生成等长兼容视图。"""
    converted = []
    for character in value:
        simplified = _TO_SIMPLIFIED.convert(character)
        converted.append(simplified if len(simplified) == 1 else character)
    return "".join(converted)


def contextual_slot_allowed(
        question: str, start: int, kind: str, surface: str,
        ) -> bool:
    """处理既有问式 artifact 的命名/因果歧义。"""
    if kind != "CAUSE" or surface not in {"为什么", "為什麼"}:
        return True
    prefix = question[:start]
    return not (
        prefix.endswith(("称", "稱", "称之", "稱之"))
        or prefix.endswith(("称之为", "稱之為", "称为", "稱為", "称作", "稱作",
                            "叫做", "叫作"))
        or "最常见的" in prefix
        or "最常見的" in prefix
    )


__all__ = ["aligned_surface", "contextual_slot_allowed"]
