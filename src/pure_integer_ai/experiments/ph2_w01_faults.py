"""W-01 阶段 0 的六个显式故障点和跨进程恢复场景身份。"""
from __future__ import annotations


class W01FaultPoint:
    """与 D-03 W-01 recovery binding 一致的开放故障点集合。"""

    BEFORE_FIRST_SHARD = "BEFORE_FIRST_SHARD"
    AFTER_PARTIAL_SHARD = "AFTER_PARTIAL_SHARD"
    BEFORE_MERGE_PREVIEW = "BEFORE_MERGE_PREVIEW"
    AFTER_MERGE_BEFORE_COMMIT = "AFTER_MERGE_BEFORE_COMMIT"
    AFTER_COMMIT_BEFORE_CURSOR = "AFTER_COMMIT_BEFORE_CURSOR"
    AFTER_MANIFEST_PUBLISH = "AFTER_MANIFEST_PUBLISH"
    SQLITE_PROCESS_RESTART = "SQLITE_PROCESS_RESTART"

    @classmethod
    def injectable_points(cls) -> tuple[str, ...]:
        """返回 D-03 预注册的六个进程内故障点。"""
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
        """返回正式报告必须覆盖的六故障点加 SQLite 新进程恢复。"""
        return (*cls.injectable_points(), cls.SQLITE_PROCESS_RESTART)


class W01InjectedFault(RuntimeError):
    """测试或正式恢复演练在预注册边界主动中断。"""


def hit_w01_fault(selected: str | None, point: str) -> None:
    """仅当配置点与当前边界相同才抛出确定性故障。"""
    if selected is None:
        return
    if selected not in W01FaultPoint.injectable_points():
        raise ValueError("未知 W-01 fault point")
    if selected == point:
        raise W01InjectedFault(f"W-01 injected fault: {point}")


__all__ = ["W01FaultPoint", "W01InjectedFault", "hit_w01_fault"]
