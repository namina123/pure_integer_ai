"""DLG-05 v4 source bundle 的 typed freeze 凭证。"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.experiments.conversation_heldout_v4_bundle import (
    ConversationHeldOutV4DependencyBinding,
    ConversationHeldOutV4SourceBundle,
)


# object-model: exception
class ConversationHeldOutV4FreezeError(RuntimeError):
    """v4 freeze 与当前 bundle 不一致或冻结字段不完整。"""


# object-model: value; representation=struct; interop=pending
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4Freeze:
    """只绑定完整 bundle 身份的不可覆盖 freeze 凭证。"""

    schema_version: int
    bundle_payload_size: int
    bundle_payload_sha256: tuple[int, ...]
    bundle_index: int
    turn_count: int
    source_count: int
    dependencies: ConversationHeldOutV4DependencyBinding

    def __post_init__(self) -> None:
        """核验 freeze 不含标签且所有计数/摘要为严格值。"""
        values = (
            self.schema_version,
            self.bundle_payload_size,
            self.bundle_index,
            self.turn_count,
            self.source_count,
        )
        if any(type(item) is not int or item <= 0 for item in values):
            raise ConversationHeldOutV4FreezeError(
                "freeze 版本、计数和 index 必须为正整数")
        if (not isinstance(self.bundle_payload_sha256, tuple)
                or len(self.bundle_payload_sha256) != 32
                or any(type(item) is not int or item < 0 or item > 255
                       for item in self.bundle_payload_sha256)):
            raise ConversationHeldOutV4FreezeError("freeze bundle SHA-256 非法")
        if not isinstance(self.dependencies, ConversationHeldOutV4DependencyBinding):
            raise ConversationHeldOutV4FreezeError("freeze dependencies 类型错误")

    def stable_key(self) -> tuple[int, ...]:
        """返回 freeze 凭证的完整整数键。"""
        result = [self.schema_version, self.bundle_payload_size,
                  self.bundle_index, self.turn_count, self.source_count]
        result.extend(self.bundle_payload_sha256)
        result.extend(self.dependencies.stable_key())
        return tuple(result)


def freeze_v4_bundle(
        bundle: ConversationHeldOutV4SourceBundle,
        ) -> ConversationHeldOutV4Freeze:
    """从当前完整 bundle 建立一次 typed freeze，不读取旧 v3 artifact。"""
    if not isinstance(bundle, ConversationHeldOutV4SourceBundle):
        raise TypeError("bundle 必须是 ConversationHeldOutV4SourceBundle")
    return ConversationHeldOutV4Freeze(
        1,
        bundle.payload_size,
        bundle.payload_sha256,
        bundle.index,
        len(bundle.turns),
        len(bundle.sources),
        bundle.dependencies,
    )


def verify_v4_freeze(
        bundle: ConversationHeldOutV4SourceBundle,
        freeze: ConversationHeldOutV4Freeze,
        ) -> None:
    """只读重算 bundle 身份，任何漂移都 fail closed。"""
    if not isinstance(bundle, ConversationHeldOutV4SourceBundle):
        raise TypeError("bundle 必须是 ConversationHeldOutV4SourceBundle")
    if not isinstance(freeze, ConversationHeldOutV4Freeze):
        raise TypeError("freeze 必须是 ConversationHeldOutV4Freeze")
    expected = freeze_v4_bundle(bundle)
    if expected != freeze:
        raise ConversationHeldOutV4FreezeError(
            "v4 bundle 与 freeze 凭证不一致")


__all__ = [
    "ConversationHeldOutV4Freeze",
    "ConversationHeldOutV4FreezeError",
    "freeze_v4_bundle",
    "verify_v4_freeze",
]
