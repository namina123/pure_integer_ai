"""DLG-RAW-10：W03-W05 公开 proof sentence 的可迁移 provider 边界。

本模块把既有 ``PublicSentenceDemo`` 的 Python runtime 隔离在 resource-owner
adapter 内。DLG core 只能观察有限整数 record、固定结果码、SourceRef key 和
UTF-8 u8[]；``SparseQARuntime``、字符串、路径和 snapshot 解码细节不参与
provider 的规范语义。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pure_integer_ai.experiments.conversation_public_sentence_demo import (
    PUBLIC_SENTENCE_DEMO_ANSWER,
    PUBLIC_SENTENCE_DEMO_REJECT_LEXICAL_AMBIGUOUS,
    PUBLIC_SENTENCE_DEMO_REJECT_LEXICAL_MISS,
    PUBLIC_SENTENCE_DEMO_REJECT_OUTPUT_BUDGET,
    PUBLIC_SENTENCE_DEMO_REJECT_RAW,
    PUBLIC_SENTENCE_DEMO_REJECT_RUNTIME_AMBIGUOUS,
    PUBLIC_SENTENCE_DEMO_REJECT_RUNTIME_INCONSISTENT,
    PUBLIC_SENTENCE_DEMO_REJECT_RUNTIME_UNKNOWN,
    PUBLIC_SENTENCE_DEMO_ROUTE_ALIAS,
    PUBLIC_SENTENCE_DEMO_ROUTE_EXACT,
    PUBLIC_SENTENCE_DEMO_ROUTE_IMPLICIT,
    PUBLIC_SENTENCE_DEMO_ROUTE_NONE,
    PublicSentenceDemoCatalog,
    PublicSentenceDemoError,
    PublicSentenceDemoResult,
    PublicSentenceDemoSameDispatchProofProjection,
    build_public_sentence_demo_catalog,
    run_public_sentence_demo_vector,
    run_public_sentence_demo_vector_with_typed_proof,
)
from pure_integer_ai.experiments.conversation_public_source_payload_provider import (
    portable_sha256_v1,
    public_source_payload_sha256_v1,
)
from pure_integer_ai.experiments.conversation_raw_intake import (
    DLG_RAW_ACCEPT,
    DLG_RAW_REJECT_LEXICAL_AMBIGUOUS,
    DLG_RAW_REJECT_LEXICAL_MISS,
    DLG_RAW_REJECT_OUTPUT_BUDGET,
    DLG_RAW_REJECT_RUNTIME,
    ConversationRawIntake,
    encode_utf8_v1,
    intake_raw_conversation_vector,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_sparse_qa_runtime_contract import (
    SparseQARuntime,
)
from pure_integer_ai.experiments.ph2_w03_w04_w05_sparse_qa_snapshot import (
    SparseQARuntimeSnapshotError,
    load_public_sparse_qa_runtime_snapshot,
)


PUBLIC_PROOF_SENTENCE_PROVIDER_RECORD_V1 = 1
PUBLIC_PROOF_SENTENCE_PROVIDER_BINDING_RECORD_V1 = 1
PUBLIC_PROOF_SENTENCE_PROVIDER_KIND_W03_W05_V1 = 1
PUBLIC_PROOF_SENTENCE_PROVIDER_CONTEXT_NONE_NO_WRITE_V1 = 0
PUBLIC_PROOF_SENTENCE_PROVIDER_MAX_OUTPUT_BYTES = 4096
PUBLIC_PROOF_SENTENCE_PROVIDER_SNAPSHOT_RELATIVE_PATH = (
    "data/ph2/sparse_qa_runtime_snapshot_v1.json")
PUBLIC_PROOF_SENTENCE_PROVIDER_BINDING_DOMAIN_V1 = (
    b"PURE-INTEGER-AI/DLG-RAW-10/PUBLIC-PROOF-SENTENCE-PROVIDER/V1")
PUBLIC_PROOF_SENTENCE_PROVIDER_RESULT_DOMAIN_V1 = (
    b"PURE-INTEGER-AI/DLG-RAW-10/RESULT-RECORD/V1")

# 该向量是 provider adapter 的加载时自检，不参与问题解析或答案选择。
# 输入和期望摘要均以显式 u8[] 固化，使另一种整数实现可以重放同一条
# 已发布 snapshot 路径，而不依赖 Python 测试夹具或 JSON 字段顺序。
PUBLIC_PROOF_SENTENCE_PROVIDER_CONFORMANCE_INPUT_V1 = (
    229, 175, 146, 230, 189, 174, 229, 175, 188, 232, 135, 180,
    228, 187, 128, 228, 185, 136, 239, 188, 159,
)
PUBLIC_PROOF_SENTENCE_PROVIDER_CONFORMANCE_RESULT_IDENTITY_V1 = (
    164, 138, 99, 40, 72, 116, 41, 166,
    105, 39, 206, 94, 132, 181, 221, 252,
    46, 94, 180, 171, 80, 23, 6, 89,
    193, 229, 85, 107, 12, 54, 0, 26,
)

PUBLIC_PROOF_SENTENCE_PROVIDER_STATUS_ANSWER = 1
PUBLIC_PROOF_SENTENCE_PROVIDER_STATUS_RAW_REJECT = 2
PUBLIC_PROOF_SENTENCE_PROVIDER_STATUS_LEXICAL_MISS = 3
PUBLIC_PROOF_SENTENCE_PROVIDER_STATUS_UNKNOWN = 4
PUBLIC_PROOF_SENTENCE_PROVIDER_STATUS_AMBIGUOUS = 5
PUBLIC_PROOF_SENTENCE_PROVIDER_STATUS_OUTPUT_BUDGET = 6
PUBLIC_PROOF_SENTENCE_PROVIDER_STATUS_RUNTIME_REJECT = 7

_PROVIDER_STATUSES = frozenset({
    PUBLIC_PROOF_SENTENCE_PROVIDER_STATUS_ANSWER,
    PUBLIC_PROOF_SENTENCE_PROVIDER_STATUS_RAW_REJECT,
    PUBLIC_PROOF_SENTENCE_PROVIDER_STATUS_LEXICAL_MISS,
    PUBLIC_PROOF_SENTENCE_PROVIDER_STATUS_UNKNOWN,
    PUBLIC_PROOF_SENTENCE_PROVIDER_STATUS_AMBIGUOUS,
    PUBLIC_PROOF_SENTENCE_PROVIDER_STATUS_OUTPUT_BUDGET,
    PUBLIC_PROOF_SENTENCE_PROVIDER_STATUS_RUNTIME_REJECT,
})
_ROUTE_KINDS = frozenset({
    PUBLIC_SENTENCE_DEMO_ROUTE_NONE,
    PUBLIC_SENTENCE_DEMO_ROUTE_EXACT,
    PUBLIC_SENTENCE_DEMO_ROUTE_ALIAS,
    PUBLIC_SENTENCE_DEMO_ROUTE_IMPLICIT,
})
_HEX = frozenset("0123456789abcdef")


# object-model: exception; interop=DLG-RAW-10
class PublicProofSentenceProviderError(ValueError):
    """DLG-RAW-10 provider 的 binding、adapter 或结果 record 不闭合。"""


def _pack(result: list[int], value: tuple[int, ...]) -> None:
    """以明确长度前缀写入非负整数段，不借助对象序列化。"""
    result.extend((len(value), *value))


def _strict_nonnegative_vector(
        value: tuple[int, ...],
        *,
        label: str,
        allow_empty: bool,
        ) -> tuple[int, ...]:
    """验证可由任意整数语言保留的有序非负整数 sequence。"""
    if (not isinstance(value, tuple)
            or (not allow_empty and not value)
            or any(type(item) is not int or item < 0 for item in value)):
        raise PublicProofSentenceProviderError(
            f"{label} 必须是{'可空' if allow_empty else '非空'}非负严格整数 tuple")
    return value


def _u8_vector(
        value: tuple[int, ...],
        *,
        label: str,
        allow_empty: bool,
        ) -> tuple[int, ...]:
    """验证 canonical raw u8[]，不让 Python bytes 定义核心行为。"""
    result = _strict_nonnegative_vector(
        value,
        label=label,
        allow_empty=allow_empty,
    )
    if any(item > 255 for item in result):
        raise PublicProofSentenceProviderError(f"{label} 含非 u8 整数")
    return result


def _registered_status(value: int) -> int:
    """拒绝 bool、整数子类和未登记 provider status。"""
    if type(value) is not int or value not in _PROVIDER_STATUSES:
        raise PublicProofSentenceProviderError("provider status 未注册")
    return value


def _sha256_raw_u8_from_ascii(value: str, *, label: str) -> tuple[int, ...]:
    """把固定小写 SHA-256 ASCII 以 nibble 状态机转为 raw 32-byte identity。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(item not in _HEX for item in value)):
        raise PublicProofSentenceProviderError(f"{label} 不是小写 SHA-256")
    result: list[int] = []
    for cursor in range(0, 64, 2):
        high = ord(value[cursor])
        low = ord(value[cursor + 1])
        high_value = high - 0x30 if high <= 0x39 else high - 0x61 + 10
        low_value = low - 0x30 if low <= 0x39 else low - 0x61 + 10
        result.append((high_value << 4) | low_value)
    return tuple(result)


def _binding_identity(
        *,
        snapshot_sha256: tuple[int, ...],
        runtime_identity: tuple[int, ...],
        catalog_record: tuple[int, ...],
        ) -> tuple[int, ...]:
    """由 snapshot、runtime 和全量 route record 生成固定 provider identity。"""
    snapshot = _u8_vector(
        snapshot_sha256,
        label="provider snapshot SHA-256",
        allow_empty=False,
    )
    runtime = _u8_vector(
        runtime_identity,
        label="provider runtime identity",
        allow_empty=False,
    )
    catalog = _strict_nonnegative_vector(
        catalog_record,
        label="provider catalog record",
        allow_empty=False,
    )
    if len(snapshot) != 32 or len(runtime) != 32:
        raise PublicProofSentenceProviderError("provider SHA-256 identity 长度漂移")
    record = [PUBLIC_PROOF_SENTENCE_PROVIDER_BINDING_RECORD_V1]
    _pack(record, snapshot)
    _pack(record, runtime)
    _pack(record, catalog)
    try:
        return tuple(portable_sha256_v1(
            PUBLIC_PROOF_SENTENCE_PROVIDER_BINDING_DOMAIN_V1,
            (tuple(record),),
        ))
    except (TypeError, ValueError) as error:
        raise PublicProofSentenceProviderError(
            "provider binding identity 无法形成") from error


def _runtime_rejection(
        provider: "PublicProofSentenceProviderV1",
        intake: ConversationRawIntake,
        ) -> "PublicProofSentenceProviderResultV1":
    """把 adapter 或回放漂移归一为零输出的 DLG runtime 拒绝。"""
    return PublicProofSentenceProviderResultV1(
        PUBLIC_PROOF_SENTENCE_PROVIDER_STATUS_RUNTIME_REJECT,
        DLG_RAW_REJECT_RUNTIME,
        intake,
        provider.provider_identity,
        provider.runtime_identity,
        provider.catalog_record,
    )


# object-model: resource-owner; interop=DLG-RAW-10-host-adapter
@dataclass(frozen=True, slots=True)
class PublicProofSentenceProviderV1:
    """持有当前 Python W03-W05 adapter，但对 core 只导出冻结整数 binding。"""

    legacy_runtime: SparseQARuntime
    legacy_catalog: PublicSentenceDemoCatalog
    snapshot_sha256: tuple[int, ...]
    runtime_identity: tuple[int, ...]
    catalog_record: tuple[int, ...]
    provider_identity: tuple[int, ...]

    def __post_init__(self) -> None:
        """确认 legacy runtime 只是可由 binding 完全约束的物理执行器。"""
        if not isinstance(self.legacy_runtime, SparseQARuntime):
            raise TypeError("provider legacy runtime 类型错误")
        if not isinstance(self.legacy_catalog, PublicSentenceDemoCatalog):
            raise TypeError("provider legacy catalog 类型错误")
        snapshot = _u8_vector(
            self.snapshot_sha256,
            label="provider snapshot SHA-256",
            allow_empty=False,
        )
        runtime = _u8_vector(
            self.runtime_identity,
            label="provider runtime identity",
            allow_empty=False,
        )
        catalog = _strict_nonnegative_vector(
            self.catalog_record,
            label="provider catalog record",
            allow_empty=False,
        )
        identity = _u8_vector(
            self.provider_identity,
            label="provider identity",
            allow_empty=False,
        )
        if not (len(snapshot) == len(runtime) == len(identity) == 32):
            raise PublicProofSentenceProviderError(
                "provider identity 必须均为 32-byte SHA-256")
        expected_runtime = _sha256_raw_u8_from_ascii(
            self.legacy_runtime.identity_sha256,
            label="provider legacy runtime identity",
        )
        if runtime != expected_runtime:
            raise PublicProofSentenceProviderError("provider runtime identity 漂移")
        if self.legacy_catalog.runtime_identity_sha256 != self.legacy_runtime.identity_sha256:
            raise PublicProofSentenceProviderError("provider catalog/runtime binding 漂移")
        if catalog != self.legacy_catalog.canonical_record():
            raise PublicProofSentenceProviderError("provider catalog record 漂移")
        if identity != _binding_identity(
                snapshot_sha256=snapshot,
                runtime_identity=runtime,
                catalog_record=catalog,
        ):
            raise PublicProofSentenceProviderError("provider binding identity 漂移")
        object.__setattr__(self, "snapshot_sha256", snapshot)
        object.__setattr__(self, "runtime_identity", runtime)
        object.__setattr__(self, "catalog_record", catalog)
        object.__setattr__(self, "provider_identity", identity)

    def canonical_record(self) -> tuple[int, ...]:
        """导出不含 Path、临时目录、字符串或 Python 对象的 provider 本体。"""
        result = [
            PUBLIC_PROOF_SENTENCE_PROVIDER_BINDING_RECORD_V1,
            PUBLIC_PROOF_SENTENCE_PROVIDER_KIND_W03_W05_V1,
        ]
        for value in (
                self.snapshot_sha256,
                self.runtime_identity,
                self.catalog_record,
                self.provider_identity,
        ):
            _pack(result, value)
        return tuple(result)


# object-model: host-adapter; representation=legacy-carrier; interop=not-canonical
@dataclass(frozen=True, slots=True)
class PublicProofSentenceProviderSameDispatchProjection:
    """并排保留 DLG-RAW-10 result 和同次 W03-W05 legacy proof carrier。

    该对象绝不进入 provider V1 canonical record、terminal 或 snapshot。它的用途仅是
    让 DLG-RAW-11 在同一次 demo dispatch 的基础上投影来源锚点，避免按 output
    文本逆推 proof 或为锚点发起第三次查询。
    """

    provider_result: "PublicProofSentenceProviderResultV1"
    demo_proof_projection: PublicSentenceDemoSameDispatchProofProjection | None
    host_adapter_only: int = 1

    def __post_init__(self) -> None:
        """只允许 provider result 与 demo 同次 carrier 精确相等或零 proof。"""
        if type(self.provider_result) is not PublicProofSentenceProviderResultV1:
            raise TypeError("provider same-dispatch carrier 缺少 provider result")
        if type(self.host_adapter_only) is not int or self.host_adapter_only != 1:
            raise PublicProofSentenceProviderError(
                "provider same-dispatch carrier 缺少 host-only 标记")
        projection = self.demo_proof_projection
        result = self.provider_result
        if result.accepted:
            if (type(projection) is not PublicSentenceDemoSameDispatchProofProjection
                    or not projection.demo_result.accepted):
                raise PublicProofSentenceProviderError(
                    "provider ANSWER 缺少同次 demo proof carrier")
            demo = projection.demo_result
            if (result.intake != demo.intake
                    or result.demo_record != demo.canonical_record()
                    or result.route_kind != demo.selected_route_kind
                    or result.source_record_key != demo.selected_source_record_key
                    or result.output_scalars != demo.generated_proposition_scalars
                    or result.output_bytes != demo.output_bytes):
                raise PublicProofSentenceProviderError(
                    "provider ANSWER/demo same-dispatch carrier 漂移")
            return
        if projection is not None:
            demo = projection.demo_result
            if (result.intake != demo.intake
                    or result.demo_record != demo.canonical_record()
                    or result.route_kind != demo.selected_route_kind
                    or result.source_record_key != demo.selected_source_record_key
                    or result.output_scalars or result.output_bytes):
                raise PublicProofSentenceProviderError(
                    "provider rejection/demo same-dispatch carrier 漂移")


# object-model: value; representation=struct; interop=DLG-RAW-10
@dataclass(frozen=True, slots=True)
class PublicProofSentenceProviderResultV1:
    """一次 provider dispatch 的完整、无表层文本的规范整数结果。"""

    provider_status: int
    mapped_dlg_result_code: int
    intake: ConversationRawIntake
    provider_identity: tuple[int, ...]
    runtime_identity: tuple[int, ...]
    catalog_record: tuple[int, ...]
    demo_record: tuple[int, ...] = ()
    route_kind: int = PUBLIC_SENTENCE_DEMO_ROUTE_NONE
    source_record_key: tuple[int, ...] = ()
    output_scalars: tuple[int, ...] = ()
    output_bytes: tuple[int, ...] = ()
    context_policy: int = PUBLIC_PROOF_SENTENCE_PROVIDER_CONTEXT_NONE_NO_WRITE_V1

    def __post_init__(self) -> None:
        """冻结 ANSWER、拒绝、来源与零 context write 的精确组合。"""
        status = _registered_status(self.provider_status)
        if (type(self.mapped_dlg_result_code) is not int
                or self.mapped_dlg_result_code < 0):
            raise PublicProofSentenceProviderError("provider DLG result code 非法")
        if not isinstance(self.intake, ConversationRawIntake):
            raise TypeError("provider result intake 类型错误")
        provider = _u8_vector(
            self.provider_identity,
            label="provider result identity",
            allow_empty=False,
        )
        runtime = _u8_vector(
            self.runtime_identity,
            label="provider result runtime identity",
            allow_empty=False,
        )
        catalog = _strict_nonnegative_vector(
            self.catalog_record,
            label="provider result catalog record",
            allow_empty=False,
        )
        demo = _strict_nonnegative_vector(
            self.demo_record,
            label="provider nested demo record",
            allow_empty=True,
        )
        source = _strict_nonnegative_vector(
            self.source_record_key,
            label="provider SourceRef key",
            allow_empty=True,
        )
        scalars = _strict_nonnegative_vector(
            self.output_scalars,
            label="provider output scalar",
            allow_empty=True,
        )
        output = _u8_vector(
            self.output_bytes,
            label="provider output bytes",
            allow_empty=True,
        )
        if (len(provider) != 32 or len(runtime) != 32
                or type(self.context_policy) is not int
                or self.context_policy
                != PUBLIC_PROOF_SENTENCE_PROVIDER_CONTEXT_NONE_NO_WRITE_V1):
            raise PublicProofSentenceProviderError("provider result binding 或 context policy 漂移")
        if any(
                item > 0x10FFFF or 0xD800 <= item <= 0xDFFF
                for item in scalars):
            raise PublicProofSentenceProviderError("provider output scalar 非法")
        if type(self.route_kind) is not int or self.route_kind not in _ROUTE_KINDS:
            raise PublicProofSentenceProviderError("provider route kind 未注册")
        has_route = self.route_kind != PUBLIC_SENTENCE_DEMO_ROUTE_NONE
        has_output = bool(scalars or output)
        if status == PUBLIC_PROOF_SENTENCE_PROVIDER_STATUS_ANSWER:
            if (not self.intake.accepted
                    or self.mapped_dlg_result_code != DLG_RAW_ACCEPT
                    or not demo or not has_route or not source or not has_output
                    or len(output) > PUBLIC_PROOF_SENTENCE_PROVIDER_MAX_OUTPUT_BYTES
                    or encode_utf8_v1(scalars) != output):
                raise PublicProofSentenceProviderError("provider ANSWER record 不闭合")
            readback = intake_raw_conversation_vector(output)
            if (not readback.accepted
                    or readback.unicode_scalars != scalars):
                raise PublicProofSentenceProviderError("provider ANSWER UTF-8 readback 漂移")
        elif status == PUBLIC_PROOF_SENTENCE_PROVIDER_STATUS_RAW_REJECT:
            if (self.intake.accepted
                    or self.mapped_dlg_result_code != self.intake.result_code
                    or demo or has_route or source or has_output):
                raise PublicProofSentenceProviderError("provider raw rejection 不闭合")
        elif status == PUBLIC_PROOF_SENTENCE_PROVIDER_STATUS_LEXICAL_MISS:
            if (not self.intake.accepted
                    or self.mapped_dlg_result_code != DLG_RAW_REJECT_LEXICAL_MISS
                    or not demo or has_route or source or has_output):
                raise PublicProofSentenceProviderError("provider lexical miss 不闭合")
        elif status == PUBLIC_PROOF_SENTENCE_PROVIDER_STATUS_UNKNOWN:
            if (not self.intake.accepted
                    or self.mapped_dlg_result_code != DLG_RAW_REJECT_RUNTIME
                    or not demo or not has_route or not source or has_output):
                raise PublicProofSentenceProviderError("provider UNKNOWN 不闭合")
        elif status == PUBLIC_PROOF_SENTENCE_PROVIDER_STATUS_AMBIGUOUS:
            if (not self.intake.accepted
                    or self.mapped_dlg_result_code
                    != DLG_RAW_REJECT_LEXICAL_AMBIGUOUS
                    or not demo or has_output):
                raise PublicProofSentenceProviderError("provider ambiguity 不闭合")
        elif status == PUBLIC_PROOF_SENTENCE_PROVIDER_STATUS_OUTPUT_BUDGET:
            if (not self.intake.accepted
                    or self.mapped_dlg_result_code != DLG_RAW_REJECT_OUTPUT_BUDGET
                    or not demo or not has_route or not source or has_output):
                raise PublicProofSentenceProviderError("provider output budget 不闭合")
        elif (not self.intake.accepted
                or self.mapped_dlg_result_code != DLG_RAW_REJECT_RUNTIME
                or has_output):
            raise PublicProofSentenceProviderError("provider runtime rejection 不闭合")
        object.__setattr__(self, "provider_status", status)
        object.__setattr__(self, "provider_identity", provider)
        object.__setattr__(self, "runtime_identity", runtime)
        object.__setattr__(self, "catalog_record", catalog)
        object.__setattr__(self, "demo_record", demo)
        object.__setattr__(self, "source_record_key", source)
        object.__setattr__(self, "output_scalars", scalars)
        object.__setattr__(self, "output_bytes", output)

    @property
    def accepted(self) -> bool:
        """只有同次 W03-W05 proof 的完整句通过全部回读才接受。"""
        return self.provider_status == PUBLIC_PROOF_SENTENCE_PROVIDER_STATUS_ANSWER

    def canonical_record(self) -> tuple[int, ...]:
        """导出 provider 的语言中立 result record，拒绝 Python object identity。"""
        result = [
            PUBLIC_PROOF_SENTENCE_PROVIDER_RECORD_V1,
            PUBLIC_PROOF_SENTENCE_PROVIDER_KIND_W03_W05_V1,
            self.provider_status,
            self.mapped_dlg_result_code,
            self.route_kind,
            self.context_policy,
        ]
        for value in (
                self.intake.canonical_record(),
                self.provider_identity,
                self.runtime_identity,
                self.catalog_record,
                self.demo_record,
                self.source_record_key,
                self.output_scalars,
                self.output_bytes,
        ):
            _pack(result, value)
        return tuple(result)


def _result_from_demo(
        provider: PublicProofSentenceProviderV1,
        intake: ConversationRawIntake,
        demo: PublicSentenceDemoResult,
        ) -> PublicProofSentenceProviderResultV1:
    """将已验证 demo record 显式映射到 DLG code，不复制任何答案表层。"""
    if demo.intake != intake:
        raise PublicProofSentenceProviderError("provider/demo intake 漂移")
    demo_record = demo.canonical_record()
    common = {
        "intake": intake,
        "provider_identity": provider.provider_identity,
        "runtime_identity": provider.runtime_identity,
        "catalog_record": provider.catalog_record,
        "demo_record": demo_record,
        "route_kind": demo.selected_route_kind,
        "source_record_key": demo.selected_source_record_key,
    }
    if demo.result_code == PUBLIC_SENTENCE_DEMO_ANSWER:
        return PublicProofSentenceProviderResultV1(
            PUBLIC_PROOF_SENTENCE_PROVIDER_STATUS_ANSWER,
            DLG_RAW_ACCEPT,
            output_scalars=demo.generated_proposition_scalars,
            output_bytes=demo.output_bytes,
            **common,
        )
    if demo.result_code == PUBLIC_SENTENCE_DEMO_REJECT_RAW:
        return PublicProofSentenceProviderResultV1(
            PUBLIC_PROOF_SENTENCE_PROVIDER_STATUS_RAW_REJECT,
            intake.result_code,
            intake=intake,
            provider_identity=provider.provider_identity,
            runtime_identity=provider.runtime_identity,
            catalog_record=provider.catalog_record,
        )
    if demo.result_code == PUBLIC_SENTENCE_DEMO_REJECT_LEXICAL_MISS:
        return PublicProofSentenceProviderResultV1(
            PUBLIC_PROOF_SENTENCE_PROVIDER_STATUS_LEXICAL_MISS,
            DLG_RAW_REJECT_LEXICAL_MISS,
            **common,
        )
    if demo.result_code == PUBLIC_SENTENCE_DEMO_REJECT_RUNTIME_UNKNOWN:
        return PublicProofSentenceProviderResultV1(
            PUBLIC_PROOF_SENTENCE_PROVIDER_STATUS_RUNTIME_REJECT,
            DLG_RAW_REJECT_RUNTIME,
            **common,
        )
    if demo.result_code in {
            PUBLIC_SENTENCE_DEMO_REJECT_LEXICAL_AMBIGUOUS,
            PUBLIC_SENTENCE_DEMO_REJECT_RUNTIME_AMBIGUOUS,
    }:
        return PublicProofSentenceProviderResultV1(
            PUBLIC_PROOF_SENTENCE_PROVIDER_STATUS_AMBIGUOUS,
            DLG_RAW_REJECT_LEXICAL_AMBIGUOUS,
            **common,
        )
    if demo.result_code == PUBLIC_SENTENCE_DEMO_REJECT_OUTPUT_BUDGET:
        return PublicProofSentenceProviderResultV1(
            PUBLIC_PROOF_SENTENCE_PROVIDER_STATUS_OUTPUT_BUDGET,
            DLG_RAW_REJECT_OUTPUT_BUDGET,
            **common,
        )
    if demo.result_code == PUBLIC_SENTENCE_DEMO_REJECT_RUNTIME_INCONSISTENT:
        return PublicProofSentenceProviderResultV1(
            PUBLIC_PROOF_SENTENCE_PROVIDER_STATUS_RUNTIME_REJECT,
            DLG_RAW_REJECT_RUNTIME,
            **common,
        )
    raise PublicProofSentenceProviderError("provider 收到未注册 demo result code")


def run_public_proof_sentence_provider_vector(
        provider: PublicProofSentenceProviderV1,
        raw_input_bytes: tuple[int, ...],
        ) -> PublicProofSentenceProviderResultV1:
    """运行一轮 provider；core 输入和输出都只允许显式整数与有限 u8[]。"""
    return run_public_proof_sentence_provider_vector_with_typed_proof(
        provider,
        raw_input_bytes,
    ).provider_result


def run_public_proof_sentence_provider_vector_with_typed_proof(
        provider: PublicProofSentenceProviderV1,
        raw_input_bytes: tuple[int, ...],
        ) -> PublicProofSentenceProviderSameDispatchProjection:
    """host-only：一次 provider dispatch 同时保留既有 result 和同次 legacy proof。"""
    if type(provider) is not PublicProofSentenceProviderV1:
        raise TypeError("proof sentence provider 类型错误")
    raw = _u8_vector(
        raw_input_bytes,
        label="provider raw input",
        allow_empty=True,
    )
    intake = intake_raw_conversation_vector(raw)
    if not intake.accepted:
        return PublicProofSentenceProviderSameDispatchProjection(
            _result_from_demo(
                provider,
                intake,
                run_public_sentence_demo_vector(
                    provider.legacy_runtime,
                    provider.legacy_catalog,
                    raw,
                ),
            ),
            None,
        )
    try:
        demo_projection = run_public_sentence_demo_vector_with_typed_proof(
            provider.legacy_runtime,
            provider.legacy_catalog,
            raw,
        )
        return PublicProofSentenceProviderSameDispatchProjection(
            _result_from_demo(provider, intake, demo_projection.demo_result),
            demo_projection,
        )
    except (PublicProofSentenceProviderError, PublicSentenceDemoError,
            TypeError, ValueError, RuntimeError):
        return PublicProofSentenceProviderSameDispatchProjection(
            _runtime_rejection(provider, intake),
            None,
        )


def _run_public_proof_sentence_provider_from_intake(
        provider: PublicProofSentenceProviderV1,
        intake: ConversationRawIntake,
        ) -> PublicProofSentenceProviderResultV1:
    """从同次已解码 intake 运行 host adapter，并把所有内部漂移归一拒绝。"""
    if type(provider) is not PublicProofSentenceProviderV1:
        raise TypeError("proof sentence provider 类型错误")
    if not isinstance(intake, ConversationRawIntake):
        raise TypeError("proof sentence provider intake 类型错误")
    try:
        demo = run_public_sentence_demo_vector(
            provider.legacy_runtime,
            provider.legacy_catalog,
            intake.raw_input_bytes,
        )
        return _result_from_demo(provider, intake, demo)
    except (PublicProofSentenceProviderError, PublicSentenceDemoError,
            TypeError, ValueError, RuntimeError):
        return _runtime_rejection(provider, intake)


def verify_public_proof_sentence_provider_result(
        provider: PublicProofSentenceProviderV1,
        raw_input_bytes: tuple[int, ...],
        result: PublicProofSentenceProviderResultV1,
        ) -> bool:
    """重放当前 raw u8[] 并逐 record 验证 provider carrier，失败只返回假。"""
    if (type(provider) is not PublicProofSentenceProviderV1
            or type(result) is not PublicProofSentenceProviderResultV1):
        return False
    try:
        raw = _u8_vector(
            raw_input_bytes,
            label="provider verification raw input",
            allow_empty=True,
        )
        intake = intake_raw_conversation_vector(raw)
        expected = _run_public_proof_sentence_provider_from_intake(
            provider,
            intake,
        )
    except (PublicProofSentenceProviderError, TypeError, ValueError,
            RuntimeError):
        return False
    return result.canonical_record() == expected.canonical_record()


def reject_public_proof_sentence_provider_runtime(
        provider: PublicProofSentenceProviderV1,
        intake: ConversationRawIntake,
        ) -> PublicProofSentenceProviderResultV1:
    """供会话层把无法验证的 provider carrier 固定映射为零输出 runtime 拒绝。"""
    if type(provider) is not PublicProofSentenceProviderV1:
        raise TypeError("proof sentence provider 类型错误")
    if not isinstance(intake, ConversationRawIntake):
        raise TypeError("proof sentence provider intake 类型错误")
    return _runtime_rejection(provider, intake)


def _verify_frozen_provider_conformance(
        provider: PublicProofSentenceProviderV1,
        ) -> None:
    """在 host adapter 启动时重放固定向量，拒绝错误 snapshot 或实现漂移。"""
    result = run_public_proof_sentence_provider_vector(
        provider,
        PUBLIC_PROOF_SENTENCE_PROVIDER_CONFORMANCE_INPUT_V1,
    )
    if (result.provider_status
            != PUBLIC_PROOF_SENTENCE_PROVIDER_STATUS_ANSWER
            or result.mapped_dlg_result_code != DLG_RAW_ACCEPT):
        raise PublicProofSentenceProviderError(
            "provider frozen conformance vector 未产生 ANSWER")
    try:
        identity = tuple(portable_sha256_v1(
            PUBLIC_PROOF_SENTENCE_PROVIDER_RESULT_DOMAIN_V1,
            (result.canonical_record(),),
        ))
    except (PublicSourcePayloadProviderError, TypeError, ValueError) as error:
        raise PublicProofSentenceProviderError(
            "provider frozen conformance record 无法形成") from error
    if identity != PUBLIC_PROOF_SENTENCE_PROVIDER_CONFORMANCE_RESULT_IDENTITY_V1:
        raise PublicProofSentenceProviderError(
            "provider frozen conformance vector 漂移")


def _snapshot_path_under_root(root: Path) -> Path:
    """仅在 host 边缘定位固定公开 snapshot，并拒绝物理路径逃逸。"""
    try:
        base = root.resolve()
        target = (base / Path(*PUBLIC_PROOF_SENTENCE_PROVIDER_SNAPSHOT_RELATIVE_PATH.split("/"))).resolve()
        target.relative_to(base)
    except (OSError, ValueError) as error:
        raise PublicProofSentenceProviderError(
            "provider public snapshot path 非法") from error
    if not target.is_file():
        raise PublicProofSentenceProviderError("provider public snapshot 缺失")
    return target


def load_public_proof_sentence_provider_from_root(
        root: str | Path,
        ) -> PublicProofSentenceProviderV1:
    """Python host adapter：读取已发布 snapshot，绝不走重建或临时目录 fallback。"""
    try:
        base = Path(root).resolve()
        source = _snapshot_path_under_root(base)
        snapshot_payload = source.read_bytes()
        runtime = load_public_sparse_qa_runtime_snapshot(
            source,
            repository=base,
        )
        catalog = build_public_sentence_demo_catalog(runtime)
        snapshot_sha256 = tuple(public_source_payload_sha256_v1(snapshot_payload))
        runtime_identity = _sha256_raw_u8_from_ascii(
            runtime.identity_sha256,
            label="provider runtime identity",
        )
        catalog_record = catalog.canonical_record()
        provider_identity = _binding_identity(
            snapshot_sha256=snapshot_sha256,
            runtime_identity=runtime_identity,
            catalog_record=catalog_record,
        )
        provider = PublicProofSentenceProviderV1(
            runtime,
            catalog,
            snapshot_sha256,
            runtime_identity,
            catalog_record,
            provider_identity,
        )
        _verify_frozen_provider_conformance(provider)
        return provider
    except PublicProofSentenceProviderError:
        raise
    except (OSError, SparseQARuntimeSnapshotError, TypeError, ValueError,
            RuntimeError) as error:
        raise PublicProofSentenceProviderError(
            "provider public snapshot 无法形成") from error


__all__ = [
    "PUBLIC_PROOF_SENTENCE_PROVIDER_BINDING_RECORD_V1",
    "PUBLIC_PROOF_SENTENCE_PROVIDER_CONFORMANCE_INPUT_V1",
    "PUBLIC_PROOF_SENTENCE_PROVIDER_CONFORMANCE_RESULT_IDENTITY_V1",
    "PUBLIC_PROOF_SENTENCE_PROVIDER_CONTEXT_NONE_NO_WRITE_V1",
    "PUBLIC_PROOF_SENTENCE_PROVIDER_KIND_W03_W05_V1",
    "PUBLIC_PROOF_SENTENCE_PROVIDER_MAX_OUTPUT_BYTES",
    "PUBLIC_PROOF_SENTENCE_PROVIDER_RESULT_DOMAIN_V1",
    "PUBLIC_PROOF_SENTENCE_PROVIDER_RECORD_V1",
    "PUBLIC_PROOF_SENTENCE_PROVIDER_SNAPSHOT_RELATIVE_PATH",
    "PUBLIC_PROOF_SENTENCE_PROVIDER_STATUS_AMBIGUOUS",
    "PUBLIC_PROOF_SENTENCE_PROVIDER_STATUS_ANSWER",
    "PUBLIC_PROOF_SENTENCE_PROVIDER_STATUS_LEXICAL_MISS",
    "PUBLIC_PROOF_SENTENCE_PROVIDER_STATUS_OUTPUT_BUDGET",
    "PUBLIC_PROOF_SENTENCE_PROVIDER_STATUS_RAW_REJECT",
    "PUBLIC_PROOF_SENTENCE_PROVIDER_STATUS_RUNTIME_REJECT",
    "PUBLIC_PROOF_SENTENCE_PROVIDER_STATUS_UNKNOWN",
    "PublicProofSentenceProviderError",
    "PublicProofSentenceProviderSameDispatchProjection",
    "PublicProofSentenceProviderResultV1",
    "PublicProofSentenceProviderV1",
    "load_public_proof_sentence_provider_from_root",
    "reject_public_proof_sentence_provider_runtime",
    "run_public_proof_sentence_provider_vector",
    "run_public_proof_sentence_provider_vector_with_typed_proof",
    "verify_public_proof_sentence_provider_result",
]
