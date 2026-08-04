"""W-08 六个事务故障点及受控注入。"""
from __future__ import annotations

from pure_integer_ai.experiments.ph2_w08_contract import W08_FAILURE_POINT_KEYS


class W08InjectedFault(RuntimeError):
    """W-08 事务运行命中预注册故障点。"""


def hit_w08_fault(current: str, configured: str | None) -> None:
    if configured is None:
        return
    if configured not in W08_FAILURE_POINT_KEYS:
        raise ValueError("未知 W-08 故障点")
    if current == configured:
        raise W08InjectedFault(current)


__all__ = ["W08InjectedFault", "hit_w08_fault"]
