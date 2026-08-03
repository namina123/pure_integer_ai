"""W-07 六故障点定义与受控注入。"""
from __future__ import annotations

from pure_integer_ai.experiments.ph2_w07_contract import W07_FAILURE_POINT_KEYS


class W07InjectedFault(RuntimeError):
    """W-07 故障注入点命中。"""


def hit_w07_fault(current: str, configured: str | None) -> None:
    if configured is None:
        return
    if configured not in W07_FAILURE_POINT_KEYS:
        raise ValueError("未知 W-07 故障点")
    if configured == current:
        raise W07InjectedFault(current)


__all__ = ["W07InjectedFault", "hit_w07_fault"]
