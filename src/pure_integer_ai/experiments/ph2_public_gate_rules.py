"""LG-00/J-LG-D03 的公开 identity 与 secret 文件扫描口径。"""
from __future__ import annotations

import re
from pathlib import Path

from pure_integer_ai.experiments.ph2_language_baseline_manifest import (
    PublicFileIdentity,
    PublicGateBaseline,
    scan_public_patterns,
)


LEGACY_RULES = (
    (
        "LEGACY_REGISTERED_NAME_V1",
        re.compile("Zero" + r"[ \t]+" + "AI"),
    ),
    (
        "LEGACY_URN_NAMESPACE_V1",
        re.compile("urn:" + "zero" + "-ai:", re.IGNORECASE),
    ),
)

SECRET_RULES = (
    (
        "AWS_ACCESS_KEY_V1",
        re.compile(r"(?<![A-Z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])"),
    ),
    (
        "BEARER_TOKEN_V1",
        re.compile(
            r"\bBearer[ \t]+[A-Za-z0-9._~+/\-]{20,}={0,2}\b",
            re.IGNORECASE,
        ),
    ),
    (
        "GENERIC_API_KEY_ASSIGNMENT_V1",
        re.compile(
            r"\b(?:api[_-]?key|access[_-]?token|client[_-]?secret|"
            r"secret[_-]?key)[ \t]*[:=][ \t]*[\"']"
            r"[A-Za-z0-9._~+/\-]{16,}[\"']",
            re.IGNORECASE,
        ),
    ),
    (
        "GITHUB_TOKEN_V1",
        re.compile(
            r"\b(?:(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,255}|"
            r"github_pat_[A-Za-z0-9_]{40,255})\b",
        ),
    ),
    (
        "GOOGLE_API_KEY_V1",
        re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
    ),
    (
        "LLM_API_KEY_V1",
        re.compile(r"\bsk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,}\b"),
    ),
    (
        "PRIVATE_KEY_HEADER_V1",
        re.compile(
            r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    ),
)


def build_baseline_public_gate(
        repo_root: str | Path,
        inventory: tuple[PublicFileIdentity, ...],
        ) -> PublicGateBaseline:
    """按冻结规则扫描 baseline inventory，并保留最终重扫义务。"""
    legacy, legacy_binary, legacy_unreadable = scan_public_patterns(
        repo_root, inventory, LEGACY_RULES)
    secret, secret_binary, secret_unreadable = scan_public_patterns(
        repo_root, inventory, SECRET_RULES)
    if (legacy_binary != secret_binary
            or legacy_unreadable != secret_unreadable):
        raise RuntimeError("公开门两类扫描的文件范围不一致")
    return PublicGateBaseline(
        len(inventory),
        len(inventory) - len(legacy_binary) - len(legacy_unreadable),
        legacy_binary,
        legacy_unreadable,
        tuple(key for key, _ in LEGACY_RULES),
        legacy,
        "BLOCKED" if legacy else "CLEAR",
        tuple(key for key, _ in SECRET_RULES),
        secret,
        "BLOCKED" if secret or legacy_binary or legacy_unreadable else "CLEAR",
        1,
        0,
    )


__all__ = [
    "LEGACY_RULES",
    "SECRET_RULES",
    "build_baseline_public_gate",
]
