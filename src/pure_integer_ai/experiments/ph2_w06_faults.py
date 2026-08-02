"""W-06 六故障点定义与受控注入。"""
from __future__ import annotations

from pure_integer_ai.experiments.ph2_w06_contract import W06_FAILURE_POINT_KEYS


class W06InjectedFault(RuntimeError):
    """W-06 故障注入点命中，用于验证 restart/resume 幂等恢复。"""


def hit_w06_fault(current: str, configured: str | None) -> None:
    """若配置故障点等于当前位置，抛出不含 payload 的受控异常。"""
    if configured is None:
        return
    if configured not in W06_FAILURE_POINT_KEYS:
        raise ValueError("未知 W-06 故障点")
    if configured == current:
        raise W06InjectedFault(current)


__all__ = ["W06InjectedFault", "hit_w06_fault"]
