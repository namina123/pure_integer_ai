"""W-04 六故障点定义与注入。"""
from __future__ import annotations


W04_FAILURE_POINT_KEYS = (
    "BEFORE_FIRST_SHARD",
    "AFTER_PARTIAL_SHARD",
    "BEFORE_MERGE_PREVIEW",
    "AFTER_MERGE_BEFORE_COMMIT",
    "AFTER_COMMIT_BEFORE_CURSOR",
    "AFTER_MANIFEST_PUBLISH",
)


class W04InjectedFault(RuntimeError):
    """W-04 故障注入点命中，用于验证 restart/resume 幂等恢复。"""


def hit_w04_fault(current: str, configured: str | None) -> None:
    """若配置的故障点等于当前点，则抛出受控异常。"""
    if configured is None:
        return
    if configured not in W04_FAILURE_POINT_KEYS:
        raise ValueError("未知 W-04 故障点")
    if configured == current:
        raise W04InjectedFault(current)


__all__ = ["W04_FAILURE_POINT_KEYS", "W04InjectedFault", "hit_w04_fault"]
