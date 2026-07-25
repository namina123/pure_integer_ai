"""项目环境变量命名和统一读取入口。"""
from __future__ import annotations

import os
from collections.abc import Mapping


PROJECT_ENV_PREFIX = "PURE_INTEGER_AI_"


def _env_suffix(name: str) -> str:
    """从完整变量名提取稳定后缀，并拒绝非项目变量。"""
    if not isinstance(name, str) or not name:
        raise TypeError("环境变量名必须是非空字符串")
    if name.startswith(PROJECT_ENV_PREFIX) and len(name) > len(PROJECT_ENV_PREFIX):
        return name[len(PROJECT_ENV_PREFIX):]
    raise ValueError("项目环境变量必须使用 PURE_INTEGER_AI_ 前缀")


def canonical_env_name(name: str) -> str:
    """返回环境变量的新规范名称。"""
    return PROJECT_ENV_PREFIX + _env_suffix(name)


def read_project_env(
        name: str,
        default: str | None = None,
        *,
        environ: Mapping[str, str] | None = None,
        ) -> str | None:
    """从注入映射或进程环境读取规范项目变量。"""
    source = os.environ if environ is None else environ
    canonical = canonical_env_name(name)
    return source.get(canonical, default)


__all__ = [
    "PROJECT_ENV_PREFIX",
    "canonical_env_name",
    "read_project_env",
]
