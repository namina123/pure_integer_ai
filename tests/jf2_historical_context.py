"""J-F2 已发布闭包的测试专用历史重放上下文。"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pure_integer_ai.experiments.j_f2_final_joint_seal as final_seal_module
import pure_integer_ai.experiments.ph2_j_f2_contract as preflight_module
from pure_integer_ai.experiments.j_f2_core_artifact_manifest import (
    read_core_artifact_manifest,
)


def build_historical_jf2_preflight(repository_root: str | Path):
    """只为历史 artifact 测试跳过当前源码身份，不授予生产 authority。"""
    def historical_core_reader(repository, path, *, verify_files=True):
        assert verify_files is True
        return read_core_artifact_manifest(
            repository, path, verify_files=False)

    with patch.object(
            preflight_module,
            "read_core_artifact_manifest",
            historical_core_reader):
        return preflight_module.build_jf2_preflight(repository_root)


def build_historical_final_joint_seal(repository_root: str | Path):
    """用历史 Core 身份重建 seal，供逐字节兼容回归使用。"""
    preflight = build_historical_jf2_preflight(repository_root)
    with patch.object(
            final_seal_module,
            "build_jf2_preflight",
            lambda _root: preflight):
        return final_seal_module.build_final_joint_seal(repository_root)


__all__ = [
    "build_historical_final_joint_seal",
    "build_historical_jf2_preflight",
]
