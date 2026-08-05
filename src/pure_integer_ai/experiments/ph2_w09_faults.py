"""W-09 六个事务故障点及受控注入。"""
from __future__ import annotations

from pure_integer_ai.experiments.ph2_w09_authority import W09_FAILURE_POINT_KEYS


class W09InjectedFault(RuntimeError):
    """W-09 事务运行命中预注册故障点。"""


def hit_w09_fault(current: str, configured: str | None) -> None:
    """只在配置与当前预注册故障点相同时中断执行。"""
    if configured is None:
        return
    if configured not in W09_FAILURE_POINT_KEYS:
        raise ValueError("未知 W-09 故障点")
    if current == configured:
        raise W09InjectedFault(current)


__all__ = ["W09InjectedFault", "hit_w09_fault"]
