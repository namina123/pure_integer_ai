"""历史身份与当前 HEAD 兼容性验证模式的闭集测试。"""
from __future__ import annotations

import pytest

from pure_integer_ai.experiments.artifact_verification_mode import (
    ARCHIVE_IDENTITY_VERIFY,
    CURRENT_HEAD_COMPATIBILITY_VERIFY,
    require_artifact_verification_mode,
)


@pytest.mark.parametrize(
    "mode",
    (ARCHIVE_IDENTITY_VERIFY, CURRENT_HEAD_COMPATIBILITY_VERIFY),
)
def test_artifact_verification_modes_are_explicit_and_closed(mode: str) -> None:
    """两个正式字符串模式必须逐值返回。"""
    assert require_artifact_verification_mode(mode) == mode


@pytest.mark.parametrize("mode", (None, True, 1, "archive", ""))
def test_artifact_verification_mode_rejects_aliases(mode: object) -> None:
    """布尔值、别名和空值不得隐式关闭 current 验证。"""
    with pytest.raises(ValueError, match="verification mode"):
        require_artifact_verification_mode(mode)
