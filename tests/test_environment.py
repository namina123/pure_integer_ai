"""项目环境变量规范前缀和统一读取测试。"""
from __future__ import annotations

from pure_integer_ai.crosscut.environment import (
    canonical_env_name,
    read_project_env,
)


_CANONICAL = "PURE_INTEGER_AI_LOCAL_DIR"


def test_environment_name_mapping_is_stable() -> None:
    """规范变量名必须保持稳定。"""
    assert canonical_env_name(_CANONICAL) == _CANONICAL


def test_environment_reads_canonical_or_default() -> None:
    """规范变量存在时读取其值，否则返回显式默认值。"""
    assert read_project_env(_CANONICAL, environ={_CANONICAL: "new"}) == "new"
    assert read_project_env(_CANONICAL, "fallback", environ={}) == "fallback"
