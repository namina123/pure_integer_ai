"""DLG-RAW-07/10 的公开对话逻辑运行时。

本模块把已经由 host 读取的公开 payload closure 编译为一个完整、路径无关的
对话 runtime。它不读取文件、不发现安装目录，也不保存会话状态；会话 transition
与 snapshot 只能消费这里冻结的 catalog、source-bound catalog 和 binding record。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.experiments.conversation_public_answer_catalog import (
    load_public_answer_frame_catalog_from_closure,
)
from pure_integer_ai.experiments.conversation_public_frame_catalog import (
    PublicFrameCatalog,
    load_public_frame_catalog_from_closure,
)
from pure_integer_ai.experiments.conversation_public_proof_sentence_provider import (
    PublicProofSentenceProviderV1,
)
from pure_integer_ai.experiments.conversation_provider_origin_followup import (
    ProviderOriginFollowupCatalogV1,
)
from pure_integer_ai.experiments.conversation_public_provider_origin_followup_catalog import (
    PublicProviderOriginFollowupCatalogError,
    load_public_provider_origin_followup_catalog_from_closure,
)
from pure_integer_ai.experiments.conversation_public_reference_catalog import (
    load_public_reference_frame_catalog_from_closure,
)
from pure_integer_ai.experiments.conversation_public_response_act_catalog import (
    PUBLIC_RESPONSE_ACT_CATALOG_LOGICAL_KEYS_V1,
    load_public_response_act_frame_catalog_from_closure,
    merge_public_frame_catalogs,
)
from pure_integer_ai.experiments.conversation_public_source_payload_provider import (
    PublicSourcePayloadClosureV1,
    PublicSourcePayloadProviderError,
    portable_sha256_v1,
)
from pure_integer_ai.experiments.conversation_source_bound_slot_catalog import (
    SourceBoundSlotCompositionCatalog,
    SourceBoundSlotCompositionError,
    SOURCE_BOUND_SLOT_CATALOG_LOGICAL_KEY_V3,
    SOURCE_BOUND_SLOT_CATALOG_SCHEMA_V1,
    SOURCE_BOUND_SLOT_CATALOG_SCHEMA_V2,
    SOURCE_BOUND_SLOT_CATALOG_SCHEMA_V3,
    load_source_bound_slot_composition_catalog_from_closure,
)


PUBLIC_DIALOGUE_RUNTIME_PROTOCOL_V1 = 1
PUBLIC_DIALOGUE_RUNTIME_PROTOCOL_V2 = 2
PUBLIC_DIALOGUE_RUNTIME_BINDING_RECORD_V1 = 1
PUBLIC_DIALOGUE_RUNTIME_BINDING_RECORD_V2 = 2
PUBLIC_DIALOGUE_RUNTIME_CATALOG_DOMAIN_V1 = (
    b"PURE-INTEGER-AI/DLG-RAW-07/PUBLIC-DIALOGUE-CATALOG/V1")
PUBLIC_DIALOGUE_RUNTIME_SOURCE_BOUND_DOMAIN_V1 = (
    b"PURE-INTEGER-AI/DLG-RAW-07/PUBLIC-DIALOGUE-SOURCE-BOUND/V1")
PUBLIC_DIALOGUE_RUNTIME_IDENTITY_DOMAIN_V1 = (
    b"PURE-INTEGER-AI/DLG-RAW-07/PUBLIC-DIALOGUE-RUNTIME/V1")
PUBLIC_DIALOGUE_RUNTIME_IDENTITY_DOMAIN_V2 = (
    b"PURE-INTEGER-AI/DLG-RAW-10/PUBLIC-DIALOGUE-RUNTIME/V2")


# object-model: exception; interop=DLG-RAW-07
class PublicDialogueRuntimeError(ValueError):
    """公开 closure、catalog 或逻辑 runtime binding 不能形成完整对话运行时。"""


def _identity_bytes(value: bytes, *, label: str) -> tuple[int, ...]:
    """把固定 SHA-256 raw bytes 归一为跨语言 record 使用的 u8 tuple。"""
    if type(value) is not bytes or len(value) != 32:
        raise PublicDialogueRuntimeError(f"{label} 必须是 32-byte identity")
    return tuple(value)


def _catalog_identity(catalog: PublicFrameCatalog) -> tuple[int, ...]:
    """由完整 catalog record 派生 identity，不能只以 manifest digest 代替本体。"""
    if type(catalog) is not PublicFrameCatalog:
        raise PublicDialogueRuntimeError("public dialogue catalog 类型错误")
    try:
        return _identity_bytes(
            portable_sha256_v1(
                PUBLIC_DIALOGUE_RUNTIME_CATALOG_DOMAIN_V1,
                (catalog.canonical_record(),),
            ),
            label="public dialogue catalog identity",
        )
    except (PublicSourcePayloadProviderError, TypeError, ValueError) as error:
        raise PublicDialogueRuntimeError("public dialogue catalog identity 无法形成") from error


def _source_bound_identity(
        catalog: SourceBoundSlotCompositionCatalog,
        ) -> tuple[int, ...]:
    """由完整 DLG-RAW-06 record 派生 identity，锁定 alias/family 本体。"""
    if type(catalog) is not SourceBoundSlotCompositionCatalog:
        raise PublicDialogueRuntimeError("source-bound catalog 类型错误")
    try:
        return _identity_bytes(
            portable_sha256_v1(
                PUBLIC_DIALOGUE_RUNTIME_SOURCE_BOUND_DOMAIN_V1,
                (catalog.canonical_record(),),
            ),
            label="source-bound catalog identity",
        )
    except (PublicSourcePayloadProviderError, TypeError, ValueError) as error:
        raise PublicDialogueRuntimeError("source-bound catalog identity 无法形成") from error


def _active_contains_base(
        base_catalog: PublicFrameCatalog,
        active_catalog: PublicFrameCatalog,
        ) -> None:
    """确保 active exact matcher 保留每条 V1 base frame 的完整本体。"""
    for base_frame in base_catalog.frames:
        matches = tuple(
            frame for frame in active_catalog.frames
            if frame.frame_key == base_frame.frame_key)
        if len(matches) != 1 or matches[0].canonical_record() != base_frame.canonical_record():
            raise PublicDialogueRuntimeError(
                "active public catalog 未完整包含 V1 base frame")


# object-model: value; representation=struct; interop=DLG-RAW-07
@dataclass(frozen=True, slots=True)
class PublicDialogueRuntimeV1:
    """完整公开对话 core 的无路径 struct；缓存和会话均不属于该值。"""

    source_payload_closure: PublicSourcePayloadClosureV1
    base_catalog: PublicFrameCatalog
    active_catalog: PublicFrameCatalog
    source_bound_slot_catalog: SourceBoundSlotCompositionCatalog
    proof_sentence_provider: PublicProofSentenceProviderV1 | None = None
    provider_origin_followup_catalog: ProviderOriginFollowupCatalogV1 | None = None
    protocol_revision: int = PUBLIC_DIALOGUE_RUNTIME_PROTOCOL_V2

    def __post_init__(self) -> None:
        """冻结 closure/catalog 关系，并在 runtime 被使用前复核全部 source binding。"""
        if type(self.source_payload_closure) is not PublicSourcePayloadClosureV1:
            raise TypeError("public dialogue runtime source payload closure 类型错误")
        if (type(self.base_catalog) is not PublicFrameCatalog
                or type(self.active_catalog) is not PublicFrameCatalog
                or type(self.source_bound_slot_catalog) is not SourceBoundSlotCompositionCatalog):
            raise TypeError("public dialogue runtime catalog 类型错误")
        if (self.proof_sentence_provider is not None
                and type(self.proof_sentence_provider)
                is not PublicProofSentenceProviderV1):
            raise TypeError("public dialogue runtime proof provider 类型错误")
        if (self.provider_origin_followup_catalog is not None
                and type(self.provider_origin_followup_catalog)
                is not ProviderOriginFollowupCatalogV1):
            raise TypeError("public dialogue runtime follow-up catalog 类型错误")
        if (self.provider_origin_followup_catalog is not None
                and self.proof_sentence_provider is None):
            raise PublicDialogueRuntimeError(
                "provider-origin follow-up catalog 需要 proof provider")
        if (self.provider_origin_followup_catalog is not None
                and self.provider_origin_followup_catalog
                .source_payload_closure_identity_u8
                != tuple(self.source_payload_closure.closure_identity)):
            raise PublicDialogueRuntimeError(
                "provider-origin follow-up catalog closure identity 漂移")
        if (type(self.protocol_revision) is not int
                or self.protocol_revision != PUBLIC_DIALOGUE_RUNTIME_PROTOCOL_V2):
            raise PublicDialogueRuntimeError("public dialogue runtime protocol 未注册")
        if (tuple(self.source_payload_closure.closure_identity)
                != self.source_bound_slot_catalog.source_payload_closure_identity):
            raise PublicDialogueRuntimeError("source-bound catalog closure identity 漂移")
        if self.source_bound_slot_catalog.catalog_schema == (
                SOURCE_BOUND_SLOT_CATALOG_SCHEMA_V1):
            expected_binding_catalog_sha = self.base_catalog.source_sha256
        elif self.source_bound_slot_catalog.catalog_schema == (
                SOURCE_BOUND_SLOT_CATALOG_SCHEMA_V2) or self.source_bound_slot_catalog.catalog_schema == (
                SOURCE_BOUND_SLOT_CATALOG_SCHEMA_V3):
            expected_binding_catalog_sha = self.active_catalog.source_sha256
        else:
            raise PublicDialogueRuntimeError(
                "source-bound catalog schema 未注册")
        if (expected_binding_catalog_sha
                != self.source_bound_slot_catalog.base_catalog_sha256):
            raise PublicDialogueRuntimeError(
                "source-bound catalog binding identity 漂移")
        _active_contains_base(self.base_catalog, self.active_catalog)
        try:
            self.source_bound_slot_catalog.verify_sources(self.source_payload_closure)
        except SourceBoundSlotCompositionError as error:
            raise PublicDialogueRuntimeError("source-bound public source 验证失败") from error
        # 立即求值，避免首次 snapshot 时才发现 catalog record 不可运输。
        self.binding_record()

    def binding_record(self) -> tuple[int, ...]:
        """导出 snapshot 必须逐整数匹配的完整逻辑 runtime binding。"""
        closure_identity = _identity_bytes(
            self.source_payload_closure.closure_identity,
            label="source payload closure identity",
        )
        if self.proof_sentence_provider is None:
            provider_binding = (0,)
        else:
            provider_record = self.proof_sentence_provider.canonical_record()
            provider_binding = (1, len(provider_record), *provider_record)
        return (
            PUBLIC_DIALOGUE_RUNTIME_BINDING_RECORD_V2,
            self.protocol_revision,
            *closure_identity,
            *_catalog_identity(self.active_catalog),
            *_catalog_identity(self.base_catalog),
            *_source_bound_identity(self.source_bound_slot_catalog),
            *provider_binding,
        )

    def runtime_identity(self) -> tuple[int, ...]:
        """以 binding record 计算 runtime identity，供非持久化审计比较。"""
        try:
            return _identity_bytes(
                portable_sha256_v1(
                    PUBLIC_DIALOGUE_RUNTIME_IDENTITY_DOMAIN_V2,
                    (self.binding_record(),),
                ),
                label="public dialogue runtime identity",
            )
        except (PublicSourcePayloadProviderError, TypeError, ValueError) as error:
            raise PublicDialogueRuntimeError("public dialogue runtime identity 无法形成") from error


def build_public_dialogue_runtime_v1(
        source_payload_closure: PublicSourcePayloadClosureV1,
        *,
        proof_sentence_provider: PublicProofSentenceProviderV1 | None = None,
        ) -> PublicDialogueRuntimeV1:
    """构建 DLG-RAW-10 runtime；provider 只以已冻结 binding 注入核心。"""
    if type(source_payload_closure) is not PublicSourcePayloadClosureV1:
        raise TypeError("public dialogue runtime 需要完整 source payload closure")
    base_catalog = load_public_frame_catalog_from_closure(source_payload_closure)
    answer_catalog = load_public_answer_frame_catalog_from_closure(
        source_payload_closure)
    response_catalogs = tuple(
        load_public_response_act_frame_catalog_from_closure(
            source_payload_closure,
            logical_key,
        )
        for logical_key in PUBLIC_RESPONSE_ACT_CATALOG_LOGICAL_KEYS_V1
    )
    reference_catalog = load_public_reference_frame_catalog_from_closure(
        source_payload_closure)
    active_catalog = merge_public_frame_catalogs(
        base_catalog,
        answer_catalog,
        *response_catalogs,
        reference_catalog,
    )
    source_bound_slot_catalog = load_source_bound_slot_composition_catalog_from_closure(
        source_payload_closure,
        base_catalog,
        active_catalog,
        catalog_logical_key=SOURCE_BOUND_SLOT_CATALOG_LOGICAL_KEY_V3,
    )
    followup_catalog = None
    if proof_sentence_provider is not None:
        try:
            followup_catalog = load_public_provider_origin_followup_catalog_from_closure(
                source_payload_closure,
                proof_sentence_provider,
            )
        except PublicProviderOriginFollowupCatalogError as error:
            raise PublicDialogueRuntimeError(
                "provider-origin follow-up public catalog 无法加载") from error
    return PublicDialogueRuntimeV1(
        source_payload_closure,
        base_catalog,
        active_catalog,
        source_bound_slot_catalog,
        proof_sentence_provider,
        followup_catalog,
    )


__all__ = [
    "PUBLIC_DIALOGUE_RUNTIME_BINDING_RECORD_V1",
    "PUBLIC_DIALOGUE_RUNTIME_BINDING_RECORD_V2",
    "PUBLIC_DIALOGUE_RUNTIME_PROTOCOL_V1",
    "PUBLIC_DIALOGUE_RUNTIME_PROTOCOL_V2",
    "PUBLIC_DIALOGUE_RUNTIME_IDENTITY_DOMAIN_V2",
    "PublicDialogueRuntimeError",
    "PublicDialogueRuntimeV1",
    "build_public_dialogue_runtime_v1",
]
