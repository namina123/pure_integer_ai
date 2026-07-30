"""W-03 六个冻结故障点与显式注入异常。"""
from __future__ import annotations

from enum import Enum


class W03FaultPoint(str, Enum):
    BEFORE_FIRST_SHARD = "BEFORE_FIRST_SHARD"
    AFTER_PARTIAL_SHARD = "AFTER_PARTIAL_SHARD"
    BEFORE_MERGE_PREVIEW = "BEFORE_MERGE_PREVIEW"
    AFTER_MERGE_BEFORE_COMMIT = "AFTER_MERGE_BEFORE_COMMIT"
    AFTER_COMMIT_BEFORE_CURSOR = "AFTER_COMMIT_BEFORE_CURSOR"
    AFTER_MANIFEST_PUBLISH = "AFTER_MANIFEST_PUBLISH"

    @classmethod
    def injectable_points(cls) -> tuple[str, ...]:
        """返回与 D-03 manifest 完全同序的六点文本。"""
        return tuple(item.value for item in cls)


class W03InjectedFault(RuntimeError):
    """仅由 test-local fault injector 产生，不作能力 FAIL/NE。"""


def hit_w03_fault(selected: str | None, point: W03FaultPoint) -> None:
    """在显式选择的唯一故障点中断。"""
    if selected is None:
        return
    if selected == point.value:
        raise W03InjectedFault(f"W-03 injected fault: {point.value}")


__all__ = [
    "W03FaultPoint",
    "W03InjectedFault",
    "hit_w03_fault",
]
