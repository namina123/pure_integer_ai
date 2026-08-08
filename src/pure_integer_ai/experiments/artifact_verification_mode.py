"""历史封存身份与当前 HEAD 兼容性的显式验证模式。"""
from __future__ import annotations


ARCHIVE_IDENTITY_VERIFY = "ARCHIVE_IDENTITY_VERIFY"
CURRENT_HEAD_COMPATIBILITY_VERIFY = "CURRENT_HEAD_COMPATIBILITY_VERIFY"
ARTIFACT_VERIFICATION_MODES = (
    ARCHIVE_IDENTITY_VERIFY,
    CURRENT_HEAD_COMPATIBILITY_VERIFY,
)


def require_artifact_verification_mode(value: object) -> str:
    """拒绝隐式布尔值、拼写漂移和未知的验证语义。"""
    if type(value) is not str or value not in ARTIFACT_VERIFICATION_MODES:
        raise ValueError("artifact verification mode 非法")
    return value


__all__ = [
    "ARCHIVE_IDENTITY_VERIFY",
    "ARTIFACT_VERIFICATION_MODES",
    "CURRENT_HEAD_COMPATIBILITY_VERIFY",
    "require_artifact_verification_mode",
]
