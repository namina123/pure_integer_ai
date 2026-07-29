"""W-02 字符词形边界阶段的六个显式故障点。"""
from __future__ import annotations


class W02FaultPoint:
    """与 D-03 W-02 recovery binding 精确一致的故障点。"""

    BEFORE_FIRST_SHARD = "BEFORE_FIRST_SHARD"
    AFTER_PARTIAL_SHARD = "AFTER_PARTIAL_SHARD"
    BEFORE_MERGE_PREVIEW = "BEFORE_MERGE_PREVIEW"
    AFTER_MERGE_BEFORE_COMMIT = "AFTER_MERGE_BEFORE_COMMIT"
    AFTER_COMMIT_BEFORE_CURSOR = "AFTER_COMMIT_BEFORE_CURSOR"
    AFTER_MANIFEST_PUBLISH = "AFTER_MANIFEST_PUBLISH"
    SQLITE_PROCESS_RESTART = "SQLITE_PROCESS_RESTART"

    @classmethod
    def injectable_points(cls) -> tuple[str, ...]:
        """返回 D-03 预注册的六个进程内中断点。"""
        return (
            cls.BEFORE_FIRST_SHARD,
            cls.AFTER_PARTIAL_SHARD,
            cls.BEFORE_MERGE_PREVIEW,
            cls.AFTER_MERGE_BEFORE_COMMIT,
            cls.AFTER_COMMIT_BEFORE_CURSOR,
            cls.AFTER_MANIFEST_PUBLISH,
        )

    @classmethod
    def coverage_points(cls) -> tuple[str, ...]:
        """返回六故障点加 SQLite 新进程恢复场景。"""
        return (*cls.injectable_points(), cls.SQLITE_PROCESS_RESTART)


class W02InjectedFault(RuntimeError):
    """恢复测试在预注册边界主动终止当前阶段进程。"""


def hit_w02_fault(selected: str | None, point: str) -> None:
    """只在选定边界抛出确定性故障。"""
    if selected is None:
        return
    if selected not in W02FaultPoint.injectable_points():
        raise ValueError("未知 W-02 fault point")
    if selected == point:
        raise W02InjectedFault(f"W-02 injected fault: {point}")


__all__ = ["W02FaultPoint", "W02InjectedFault", "hit_w02_fault"]
