"""J-F2 已发布闭包的测试专用历史重放上下文。"""
from __future__ import annotations

from pathlib import Path

from pure_integer_ai.experiments.artifact_verification_mode import (
    ARCHIVE_IDENTITY_VERIFY,
)
from pure_integer_ai.experiments.j_f2_final_joint_seal import (
    build_final_joint_seal,
)
from pure_integer_ai.experiments.ph2_j_f2_contract import build_jf2_preflight


def build_historical_jf2_preflight(repository_root: str | Path):
    """通过正式 archive 接口重建历史 preflight，不授予运行 authority。"""
    return build_jf2_preflight(
        repository_root,
        verification_mode=ARCHIVE_IDENTITY_VERIFY,
    )


def build_historical_final_joint_seal(repository_root: str | Path):
    """通过正式 archive 接口重建 seal，供逐字节兼容回归使用。"""
    return build_final_joint_seal(
        repository_root,
        verification_mode=ARCHIVE_IDENTITY_VERIFY,
    )


__all__ = [
    "build_historical_final_joint_seal",
    "build_historical_jf2_preflight",
]
