"""DLG-RAW-11B：mixed-context 会话的 V3 逻辑 runtime binding。

本模块不调度问题、不读取路径，也不保存会话。它把已完成的 DLG-RAW-10
runtime 与 DLG-RAW-11 的 provider-origin schema 固定为一个可运输的整数
binding，供独立 V3 session 与 V2 snapshot codec 共同校验。
"""
from __future__ import annotations

from dataclasses import dataclass

from pure_integer_ai.experiments.conversation_provider_origin_anchor import (
    ProviderOriginProviderBindingV1,
    provider_origin_anchor_schema_record_v1,
    provider_origin_provider_binding_from_public_provider_v1,
    provider_origin_relation_enum_identity_v1,
    provider_origin_relation_enum_record_v1,
)
from pure_integer_ai.experiments.conversation_provider_origin_context import (
    MIXED_CONTEXT_APPEND_ACCEPTED,
    MIXED_CONTEXT_APPEND_REJECT_ANCHOR_NONE,
    MIXED_CONTEXT_APPEND_REJECT_READ_WITNESS,
    MIXED_CONTEXT_PROVIDER_ORIGIN_TURN_RECORD_V1,
    MIXED_CONTEXT_SCHEMA_V2,
    MIXED_CONTEXT_TURN_KIND_PROVIDER_ORIGIN_PROJECTION,
    MIXED_CONTEXT_WRITE_ORIGIN_NONE,
    MIXED_CONTEXT_WRITE_ORIGIN_PROVIDER_RESULT_PROJECTION,
)
from pure_integer_ai.experiments.conversation_provider_origin_context_snapshot import (
    MIXED_CONTEXT_SNAPSHOT_CODEC_REVISION_V2,
    mixed_context_snapshot_codec_identity_v2,
    mixed_context_snapshot_codec_revision_v2,
)
from pure_integer_ai.experiments.conversation_public_dialogue_runtime import (
    PublicDialogueRuntimeV1,
)
from pure_integer_ai.experiments.conversation_public_source_payload_provider import (
    PublicSourcePayloadProviderError,
    portable_sha256_v1,
)


PUBLIC_DIALOGUE_RUNTIME_PROTOCOL_V3 = 3
PUBLIC_DIALOGUE_RUNTIME_BINDING_RECORD_V3 = 3
PUBLIC_DIALOGUE_RUNTIME_V3_PROJECTION_ADMISSION_RECORD_V1 = 1
PUBLIC_DIALOGUE_RUNTIME_V3_SNAPSHOT_CODEC_REVISION_V2 = (
    MIXED_CONTEXT_SNAPSHOT_CODEC_REVISION_V2)
PUBLIC_DIALOGUE_RUNTIME_V3_IDENTITY_DOMAIN_V1 = (
    b"PURE-INTEGER-AI/DLG-RAW-11/PUBLIC-DIALOGUE-RUNTIME/V3")
PUBLIC_DIALOGUE_RUNTIME_V3_PROJECTION_ADMISSION_DOMAIN_V1 = (
    b"PURE-INTEGER-AI/DLG-RAW-11/PROJECTION-ADMISSION/V1")


# object-model: exception; interop=DLG-RAW-11B
class PublicDialogueRuntimeV3Error(ValueError):
    """V3 runtime binding 缺少 provider-origin 或 mixed-context 的冻结约束。"""


def _pack(result: list[int], value: tuple[int, ...]) -> None:
    """将有限非负整数段以显式长度写入可迁移 record。"""
    result.extend((len(value), *value))


def _record(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """拒绝依赖 Python 容器或整数子类的 runtime record。"""
    if (type(value) is not tuple or not value
            or any(type(item) is not int or item < 0 for item in value)):
        raise PublicDialogueRuntimeV3Error(
            f"{label} 必须是非空非负严格整数 tuple")
    return value


def _u8_digest(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """验证所有 V3 identity 都以 raw u8[32] 表示。"""
    checked = _record(value, label=label)
    if len(checked) != 32 or any(item > 255 for item in checked):
        raise PublicDialogueRuntimeV3Error(f"{label} 必须是 raw u8[32]")
    return checked


def _identity(
        domain: bytes,
        record: tuple[int, ...],
        *,
        label: str,
        ) -> tuple[int, ...]:
    """按冻结 portable SHA-256 framing 导出 V3 raw identity。"""
    try:
        return tuple(portable_sha256_v1(domain, (record,)))
    except (PublicSourcePayloadProviderError, TypeError, ValueError) as error:
        raise PublicDialogueRuntimeV3Error(f"{label} 无法形成") from error


def projection_admission_record_v1() -> tuple[int, ...]:
    """导出 provider-origin 投影入场机的全部固定整数约束。

    该 record 明确绑定 ``ANCHOR_NONE`` 零写、接受的 provider tagged turn，及
    V2 schema。它不是某条具体 anchor 或 provider result，因而不能被课程样本
    替代。
    """
    result = [
        PUBLIC_DIALOGUE_RUNTIME_V3_PROJECTION_ADMISSION_RECORD_V1,
        MIXED_CONTEXT_SCHEMA_V2,
        MIXED_CONTEXT_PROVIDER_ORIGIN_TURN_RECORD_V1,
        MIXED_CONTEXT_TURN_KIND_PROVIDER_ORIGIN_PROJECTION,
        MIXED_CONTEXT_WRITE_ORIGIN_NONE,
        MIXED_CONTEXT_WRITE_ORIGIN_PROVIDER_RESULT_PROJECTION,
        MIXED_CONTEXT_APPEND_ACCEPTED,
        MIXED_CONTEXT_APPEND_REJECT_ANCHOR_NONE,
        MIXED_CONTEXT_APPEND_REJECT_READ_WITNESS,
    ]
    _pack(result, provider_origin_anchor_schema_record_v1())
    _pack(result, provider_origin_relation_enum_record_v1())
    _pack(result, provider_origin_relation_enum_identity_v1())
    return tuple(result)


def projection_admission_identity_v1() -> tuple[int, ...]:
    """返回 V3 binding 可比较的 provider-origin admission raw identity。"""
    return _identity(
        PUBLIC_DIALOGUE_RUNTIME_V3_PROJECTION_ADMISSION_DOMAIN_V1,
        projection_admission_record_v1(),
        label="provider-origin projection admission identity",
    )


# object-model: value; representation=struct; interop=DLG-RAW-11B
@dataclass(frozen=True, slots=True)
class PublicDialogueRuntimeV3:
    """V2 mixed session 唯一可接受的完整 runtime binding。

    ``legacy_runtime`` 仍是 RAW-01/02 的真实生产 caller；V3 没有重写或伪造
    ``QuestionAnswerRun``。新增字段只把 provider-origin 的完整本体和 schema
    锁进新的 snapshot binding。
    """

    legacy_runtime: PublicDialogueRuntimeV1
    provider_origin_binding: ProviderOriginProviderBindingV1
    projection_admission_identity_u8: tuple[int, ...]
    snapshot_codec_revision: tuple[int, ...]
    snapshot_codec_identity_u8: tuple[int, ...]
    protocol_revision: int = PUBLIC_DIALOGUE_RUNTIME_PROTOCOL_V3

    def __post_init__(self) -> None:
        """拒绝缺 provider、跨 runtime binding 或旧 codec revision 的 V3 runtime。"""
        if type(self.legacy_runtime) is not PublicDialogueRuntimeV1:
            raise TypeError("V3 runtime legacy runtime 类型错误")
        provider = self.legacy_runtime.proof_sentence_provider
        if provider is None:
            raise PublicDialogueRuntimeV3Error(
                "V3 mixed runtime 必须绑定完整 proof sentence provider")
        if type(self.provider_origin_binding) is not ProviderOriginProviderBindingV1:
            raise TypeError("V3 runtime provider-origin binding 类型错误")
        expected_provider_binding = (
            provider_origin_provider_binding_from_public_provider_v1(provider))
        if (self.provider_origin_binding.canonical_record()
                != expected_provider_binding.canonical_record()):
            raise PublicDialogueRuntimeV3Error(
                "V3 runtime provider-origin binding 与 legacy provider 漂移")
        admission_identity = _u8_digest(
            self.projection_admission_identity_u8,
            label="V3 runtime projection admission identity",
        )
        if admission_identity != projection_admission_identity_v1():
            raise PublicDialogueRuntimeV3Error(
                "V3 runtime projection admission identity 漂移")
        codec_revision = _record(
            self.snapshot_codec_revision,
            label="V3 runtime snapshot codec revision",
        )
        if codec_revision != mixed_context_snapshot_codec_revision_v2():
            raise PublicDialogueRuntimeV3Error("V3 runtime snapshot codec revision 未注册")
        codec_identity = _u8_digest(
            self.snapshot_codec_identity_u8,
            label="V3 runtime snapshot codec identity",
        )
        if codec_identity != mixed_context_snapshot_codec_identity_v2():
            raise PublicDialogueRuntimeV3Error("V3 runtime snapshot codec identity 漂移")
        if (type(self.protocol_revision) is not int
                or self.protocol_revision != PUBLIC_DIALOGUE_RUNTIME_PROTOCOL_V3):
            raise PublicDialogueRuntimeV3Error("V3 runtime protocol revision 未注册")
        # 强制在构造时跑全 binding，避免 snapshot 时才暴露不运输的字段。
        self.binding_record()

    @property
    def provider(self):
        """返回受 V3 binding 锁定的 DLG-RAW-10 resource owner。"""
        provider = self.legacy_runtime.proof_sentence_provider
        if provider is None:
            raise AssertionError("V3 runtime 已验证 provider 不得消失")
        return provider

    def binding_record(self) -> tuple[int, ...]:
        """导出 V3 snapshot 必须逐整数匹配的完整逻辑 binding。"""
        provider_record = _record(
            self.provider.canonical_record(),
            label="V3 runtime full provider binding",
        )
        legacy_binding = _record(
            self.legacy_runtime.binding_record(),
            label="V3 runtime legacy binding",
        )
        origin_binding = _record(
            self.provider_origin_binding.canonical_record(),
            label="V3 runtime provider-origin binding",
        )
        anchor_schema = _record(
            provider_origin_anchor_schema_record_v1(),
            label="V3 runtime anchor schema",
        )
        relation_schema = _record(
            provider_origin_relation_enum_record_v1(),
            label="V3 runtime relation enum schema",
        )
        admission_record = _record(
            projection_admission_record_v1(),
            label="V3 runtime projection admission record",
        )
        codec_revision = _record(
            self.snapshot_codec_revision,
            label="V3 runtime snapshot codec revision",
        )
        result = [
            PUBLIC_DIALOGUE_RUNTIME_BINDING_RECORD_V3,
            self.protocol_revision,
        ]
        for segment in (
                legacy_binding,
                provider_record,
                origin_binding,
                anchor_schema,
                relation_schema,
                admission_record,
                self.projection_admission_identity_u8,
                codec_revision,
                self.snapshot_codec_identity_u8):
            _pack(result, segment)
        return tuple(result)

    def runtime_identity(self) -> tuple[int, ...]:
        """以 V3 binding 形成 raw identity，不借 Python 对象 identity。"""
        return _identity(
            PUBLIC_DIALOGUE_RUNTIME_V3_IDENTITY_DOMAIN_V1,
            self.binding_record(),
            label="V3 public dialogue runtime identity",
        )


def build_public_dialogue_runtime_v3(
        legacy_runtime: PublicDialogueRuntimeV1,
        ) -> PublicDialogueRuntimeV3:
    """从已验证 DLG-RAW-10 runtime 构造 V3 mixed-session binding。"""
    if type(legacy_runtime) is not PublicDialogueRuntimeV1:
        raise TypeError("构造 V3 runtime 需要 PublicDialogueRuntimeV1")
    provider = legacy_runtime.proof_sentence_provider
    if provider is None:
        raise PublicDialogueRuntimeV3Error(
            "构造 V3 runtime 需要注入 proof sentence provider")
    return PublicDialogueRuntimeV3(
        legacy_runtime,
        provider_origin_provider_binding_from_public_provider_v1(provider),
        projection_admission_identity_v1(),
        mixed_context_snapshot_codec_revision_v2(),
        mixed_context_snapshot_codec_identity_v2(),
    )


__all__ = [
    "PUBLIC_DIALOGUE_RUNTIME_BINDING_RECORD_V3",
    "PUBLIC_DIALOGUE_RUNTIME_PROTOCOL_V3",
    "PUBLIC_DIALOGUE_RUNTIME_V3_PROJECTION_ADMISSION_RECORD_V1",
    "PUBLIC_DIALOGUE_RUNTIME_V3_SNAPSHOT_CODEC_REVISION_V2",
    "PublicDialogueRuntimeV3",
    "PublicDialogueRuntimeV3Error",
    "build_public_dialogue_runtime_v3",
    "projection_admission_identity_v1",
    "projection_admission_record_v1",
]
