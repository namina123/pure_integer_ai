"""DLG-05 v4 P3-C1b 的无载荷 annotation 构造声明。

P3-C1b 的 test transport 可以使用显式 authored annotation，但不能把一组裸
``ProtocolKey`` 当成其构造来源。本模块把来源、修订、冻结 manifest、构造/转换
代码和适用域收拢为一个不可变、可回放的声明；它不读取正文或外部文件，也不宣称
生产来源资格。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.experiments.evaluation_protocol import ProtocolKey
from pure_integer_ai.storage.integer_codec import pack_key


V4_RUNTIME_TASK_ANNOTATION_DECLARATION_SCHEMA = 1
V4_RUNTIME_TASK_ANNOTATION_ORIGIN_TEST_TRANSPORT_AUTHORED = (
    "TEST_TRANSPORT_AUTHORED"
)


class ConversationHeldOutV4RuntimeTaskAnnotationError(RuntimeError):
    """P3-C1b annotation 构造声明不完整、漂移或越出本切片边界。"""


@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4RuntimeTaskAnnotationDeclaration:
    """一份显式、无正文的 annotation 构造来源声明。

    ``declaration_identity`` 标识这份 authored declaration 本身。其余字段描述该
    declaration 所绑定的 annotation 来源、修订、manifest、代码和适用域。P3-C1b
    仅接受 test transport authored origin；生产 annotation 必须由后续独立的
    source-file/annotation adapter 物理审计后再进入该接口。
    """

    declaration_identity: ProtocolKey
    annotation_origin_status: str
    annotation_source_identity: ProtocolKey
    annotation_revision_identity: ProtocolKey
    frozen_annotation_manifest_identity: ProtocolKey
    constructing_code_identity: ProtocolKey
    annotation_transform_code_identity: ProtocolKey
    applicable_domain_identity: ProtocolKey

    def __post_init__(self) -> None:
        """拒绝裸字段缺失或当前切片不允许的 annotation origin。"""
        for label, value in (
                ("declaration", self.declaration_identity),
                ("annotation source", self.annotation_source_identity),
                ("annotation revision", self.annotation_revision_identity),
                ("annotation manifest", self.frozen_annotation_manifest_identity),
                ("constructing code", self.constructing_code_identity),
                ("annotation transform code", self.annotation_transform_code_identity),
                ("applicable domain", self.applicable_domain_identity)):
            if not isinstance(value, ProtocolKey):
                raise TypeError(f"P3-C1b annotation {label} identity type is invalid")
        if (self.annotation_origin_status
                != V4_RUNTIME_TASK_ANNOTATION_ORIGIN_TEST_TRANSPORT_AUTHORED):
            raise ValueError("P3-C1b annotation origin is not allowed in this slice")

    def stable_key(self) -> tuple[int, ...]:
        """返回完整声明身份；公开层只能发布其域分离摘要。"""
        result = [V4_RUNTIME_TASK_ANNOTATION_DECLARATION_SCHEMA]
        for value in (
                self.declaration_identity.components,
                tuple(ord(item) for item in self.annotation_origin_status),
                self.annotation_source_identity.components,
                self.annotation_revision_identity.components,
                self.frozen_annotation_manifest_identity.components,
                self.constructing_code_identity.components,
                self.annotation_transform_code_identity.components,
                self.applicable_domain_identity.components):
            pack_key(result, value)
        return tuple(result)


def require_v4_runtime_task_annotation_turn_binding(
        declaration: ConversationHeldOutV4RuntimeTaskAnnotationDeclaration,
        *,
        annotation_origin_status: str,
        annotation_source_identity: ProtocolKey,
        annotation_revision_identity: ProtocolKey,
        frozen_annotation_manifest_identity: ProtocolKey,
        constructing_code_identity: ProtocolKey,
        annotation_transform_code_identity: ProtocolKey,
        applicable_domain_identity: ProtocolKey,
        ) -> None:
    """要求一个 typed turn 逐字段绑定唯一显式 annotation declaration。"""
    if not isinstance(declaration, ConversationHeldOutV4RuntimeTaskAnnotationDeclaration):
        raise TypeError("P3-C1b annotation declaration type is invalid")
    if (annotation_origin_status != declaration.annotation_origin_status
            or annotation_source_identity != declaration.annotation_source_identity
            or annotation_revision_identity != declaration.annotation_revision_identity
            or frozen_annotation_manifest_identity
            != declaration.frozen_annotation_manifest_identity
            or constructing_code_identity != declaration.constructing_code_identity
            or annotation_transform_code_identity
            != declaration.annotation_transform_code_identity
            or applicable_domain_identity != declaration.applicable_domain_identity):
        raise ConversationHeldOutV4RuntimeTaskAnnotationError(
            "P3-C1b turn annotation declaration binding drifted")


__all__ = [
    "ConversationHeldOutV4RuntimeTaskAnnotationDeclaration",
    "ConversationHeldOutV4RuntimeTaskAnnotationError",
    "V4_RUNTIME_TASK_ANNOTATION_DECLARATION_SCHEMA",
    "V4_RUNTIME_TASK_ANNOTATION_ORIGIN_TEST_TRANSPORT_AUTHORED",
    "require_v4_runtime_task_annotation_turn_binding",
]
