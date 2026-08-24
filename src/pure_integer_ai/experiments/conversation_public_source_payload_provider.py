"""DLG-RAW-07 的公开逻辑资源 payload 闭包纯核心。

本模块只接收固定 logical key、原始 ``u8[]`` 与明确整数 record。物理布局、
路径解析、安装目录发现和文件读取属于同名 host adapter，绝不能进入这里的
闭包 identity、catalog 输入或会话快照。
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib


PUBLIC_SOURCE_PAYLOAD_SCHEMA_V1 = 1
PUBLIC_SOURCE_PAYLOAD_RECORD_V1 = 1
PUBLIC_SOURCE_PAYLOAD_CLOSURE_RECORD_V1 = 1
PUBLIC_SOURCE_PAYLOAD_READ_TRACE_RECORD_V1 = 1
PUBLIC_SOURCE_PAYLOAD_CLOSURE_DOMAIN_V1 = (
    b"PURE-INTEGER-AI/DLG-RAW-07/PUBLIC-SOURCE-PAYLOAD-CLOSURE/V1")

PUBLIC_SOURCE_PAYLOAD_RESULT_OK_V1 = 0
PUBLIC_SOURCE_PAYLOAD_RESULT_INVALID_LOGICAL_KEY_V1 = 1
PUBLIC_SOURCE_PAYLOAD_RESULT_ROOT_UNAVAILABLE_V1 = 2
PUBLIC_SOURCE_PAYLOAD_RESULT_SYMLINK_REJECTED_V1 = 3
PUBLIC_SOURCE_PAYLOAD_RESULT_PATH_ESCAPE_V1 = 4
PUBLIC_SOURCE_PAYLOAD_RESULT_RESOURCE_MISSING_V1 = 5
PUBLIC_SOURCE_PAYLOAD_RESULT_READ_FAILURE_V1 = 6
PUBLIC_SOURCE_PAYLOAD_RESULT_INTEGRITY_FAILURE_V1 = 7
PUBLIC_SOURCE_PAYLOAD_WRITE_EFFECT_NONE_V1 = 0

PUBLIC_SOURCE_PAYLOAD_PROVIDER_RESULT_CODES_V1 = (
    PUBLIC_SOURCE_PAYLOAD_RESULT_OK_V1,
    PUBLIC_SOURCE_PAYLOAD_RESULT_INVALID_LOGICAL_KEY_V1,
    PUBLIC_SOURCE_PAYLOAD_RESULT_ROOT_UNAVAILABLE_V1,
    PUBLIC_SOURCE_PAYLOAD_RESULT_SYMLINK_REJECTED_V1,
    PUBLIC_SOURCE_PAYLOAD_RESULT_PATH_ESCAPE_V1,
    PUBLIC_SOURCE_PAYLOAD_RESULT_RESOURCE_MISSING_V1,
    PUBLIC_SOURCE_PAYLOAD_RESULT_READ_FAILURE_V1,
    PUBLIC_SOURCE_PAYLOAD_RESULT_INTEGRITY_FAILURE_V1,
)

# 此序列是 DLG-RAW-07 的唯一 public dialogue logical resource registry。
# 顺序为 unsigned ASCII byte sequence，不依赖目录枚举或宿主 locale。
PUBLIC_SOURCE_PAYLOAD_LOGICAL_KEYS_V1 = (
    b"data/ph2/dlg_raw_public_ambiguity_answer_course_v1.jsonl.sample",
    b"data/ph2/dlg_raw_public_ambiguity_lexical_v1_a.txt.sample",
    b"data/ph2/dlg_raw_public_ambiguity_lexical_v1_b.txt.sample",
    b"data/ph2/dlg_raw_public_answer_frame_v1.jsonl.sample",
    b"data/ph2/dlg_raw_public_contextual_ellipsis_frame_v4.jsonl.sample",
    b"data/ph2/dlg_raw_public_contextual_ellipsis_lexical_v4_a.txt.sample",
    b"data/ph2/dlg_raw_public_contextual_ellipsis_lexical_v4_b.txt.sample",
    b"data/ph2/dlg_raw_public_derived_frame_v3.jsonl.sample",
    b"data/ph2/dlg_raw_public_derived_lexical_v3_a.txt.sample",
    b"data/ph2/dlg_raw_public_derived_lexical_v3_b.txt.sample",
    b"data/ph2/dlg_raw_public_followup_lexical_evidence_v1.txt.sample",
    b"data/ph2/dlg_raw_public_followup_lexical_evidence_v1_b.txt.sample",
    b"data/ph2/dlg_raw_public_frame_v1.jsonl.sample",
    b"data/ph2/dlg_raw_public_lexical_evidence_v1.txt.sample",
    b"data/ph2/dlg_raw_public_lexical_evidence_v1_b.txt.sample",
    b"data/ph2/dlg_raw_public_provider_followup_course_v1.jsonl.sample",
    b"data/ph2/dlg_raw_public_provider_followup_lexical_v1_a.txt.sample",
    b"data/ph2/dlg_raw_public_provider_followup_lexical_v1_b.txt.sample",
    b"data/ph2/dlg_raw_public_provider_result_followup_course_v1.jsonl.sample",
    b"data/ph2/dlg_raw_public_provider_result_followup_lexical_v1_a.txt.sample",
    b"data/ph2/dlg_raw_public_provider_result_followup_lexical_v1_b.txt.sample",
    b"data/ph2/dlg_raw_public_reference_antecedent_v3_a.txt.sample",
    b"data/ph2/dlg_raw_public_reference_antecedent_v3_b.txt.sample",
    b"data/ph2/dlg_raw_public_reference_explicit_v3_a.txt.sample",
    b"data/ph2/dlg_raw_public_reference_explicit_v3_b.txt.sample",
    b"data/ph2/dlg_raw_public_reference_frame_v3.jsonl.sample",
    b"data/ph2/dlg_raw_public_reference_input_v3_a.txt.sample",
    b"data/ph2/dlg_raw_public_reference_input_v3_b.txt.sample",
    b"data/ph2/dlg_raw_public_response_act_frame_v2.jsonl.sample",
    b"data/ph2/dlg_raw_public_response_act_lexical_v2_a.txt.sample",
    b"data/ph2/dlg_raw_public_response_act_lexical_v2_b.txt.sample",
    b"data/ph2/dlg_raw_public_route_clarification_course_v1.jsonl.sample",
    b"data/ph2/dlg_raw_public_route_clarification_surface_v1_a.txt.sample",
    b"data/ph2/dlg_raw_public_route_clarification_surface_v1_b.txt.sample",
    b"data/ph2/dlg_raw_public_slot_entity_v1_a.txt.sample",
    b"data/ph2/dlg_raw_public_slot_entity_v1_b.txt.sample",
    b"data/ph2/dlg_raw_public_slot_family_v1_a.txt.sample",
    b"data/ph2/dlg_raw_public_slot_family_v1_b.txt.sample",
    b"data/ph2/dlg_raw_public_slot_family_v2_site_a.txt.sample",
    b"data/ph2/dlg_raw_public_slot_family_v2_site_b.txt.sample",
    b"data/ph2/dlg_raw_public_slot_relation_v2_passage_a.txt.sample",
    b"data/ph2/dlg_raw_public_slot_relation_v2_passage_b.txt.sample",
    b"data/ph2/dlg_raw_public_slot_relation_v2_passage_negative_a.txt.sample",
    b"data/ph2/dlg_raw_public_slot_relation_v2_site_a.txt.sample",
    b"data/ph2/dlg_raw_public_slot_relation_v2_site_b.txt.sample",
    b"data/ph2/dlg_raw_public_slot_relation_v3_east-bank-north_a.txt.sample",
    b"data/ph2/dlg_raw_public_slot_relation_v3_east-bank-north_b.txt.sample",
    b"data/ph2/dlg_raw_public_slot_relation_v3_east-bank-pier_a.txt.sample",
    b"data/ph2/dlg_raw_public_slot_relation_v3_east-bank-pier_b.txt.sample",
    b"data/ph2/dlg_raw_public_source_bound_slot_v1.jsonl.sample",
    b"data/ph2/dlg_raw_public_source_bound_slot_v2.jsonl.sample",
    b"data/ph2/dlg_raw_public_source_bound_slot_v3.jsonl.sample",
    b"data/ph2/dlg_raw_public_terminal_dialogue_act_course_v1.jsonl.sample",
    b"data/ph2/dlg_raw_public_terminal_dialogue_act_surface_v1_a.txt.sample",
    b"data/ph2/dlg_raw_public_terminal_dialogue_act_surface_v1_b.txt.sample",
    b"data/ph2/grounded_answer_train_v1.jsonl.sample",
)

_U64_EXCLUSIVE = 1 << 64


# object-model: exception; interop=DLG-RAW-07
class PublicSourcePayloadProviderError(ValueError):
    """逻辑 payload record、闭包或固定 SHA framing 不满足 DLG-RAW-07。"""


def _u64(value: int, *, label: str, minimum: int = 0) -> int:
    """验证显式 u64，不让 Python 数值范围成为协议行为。"""
    if (type(value) is not int or value < minimum
            or value >= _U64_EXCLUSIVE):
        raise PublicSourcePayloadProviderError(f"{label} 不是合法 u64")
    return value


def _u8_bytes(value: bytes, *, label: str) -> bytes:
    """只接受不可变 raw u8[]，避免可变宿主缓冲区在构造后漂移。"""
    if type(value) is not bytes:
        raise PublicSourcePayloadProviderError(f"{label} 必须是 bytes")
    if len(value) >= _U64_EXCLUSIVE:
        raise PublicSourcePayloadProviderError(f"{label} 超出 u64 长度")
    return value


def _u64_big_endian(value: int, *, label: str) -> bytes:
    """以固定八个 unsigned byte 写出一个已验证的 u64。"""
    _u64(value, label=label)
    # 仅是 Python host 的等价落地；wire 仍为显式 big-endian u8[8]，
    # 其他语言可直接对应其 unsigned integer→fixed-byte 原语。
    return value.to_bytes(8, byteorder="big", signed=False)


def _unsigned_integer_bytes(value: int, *, label: str) -> bytes:
    """用最短 unsigned big-endian bytes 编码一个非负任意精度整数。"""
    if type(value) is not int or value < 0:
        raise PublicSourcePayloadProviderError(
            f"{label} 必须是非负严格整数")
    byte_length = max(1, (value.bit_length() + 7) // 8)
    _u64(byte_length, label=f"{label} byte length")
    # 保持最短 unsigned big-endian wire 规则；避免逐 byte Python generator
    # 在启动时重复编码大型 canonical integer record。
    return value.to_bytes(byte_length, byteorder="big", signed=False)


def _framed_bytes(value: bytes, *, label: str) -> bytes:
    """将 raw bytes 置于明确 u64 大端长度前缀中。"""
    raw = _u8_bytes(value, label=label)
    return _u64_big_endian(len(raw), label=f"{label} length") + raw


def _append_record_segment(result: list[int], value: tuple[int, ...]) -> None:
    """把一个非负整数子记录以显式 u64 count 追加到外层 record。"""
    _u64(len(value), label="record segment count")
    result.extend((len(value), *value))


def portable_integer_record_bytes_v1(
        value: tuple[int, ...],
        *,
        label: str,
        ) -> bytes:
    """编码跨语言 SHA 输入的有限整数 record。

    wire 为 ``u64 count`` 后接每个 ``u64 byte_length || unsigned bytes``。
    它是 DLG-RAW-07 identity 的规范输入，不是 pickle、JSON 或对象序列化。
    """
    if (not isinstance(value, tuple)
            or any(type(item) is not int or item < 0 for item in value)):
        raise PublicSourcePayloadProviderError(
            f"{label} 必须是非负严格整数 tuple")
    result = bytearray(_u64_big_endian(len(value), label=f"{label} count"))
    for ordinal, item in enumerate(value):
        result.extend(_framed_bytes(
            _unsigned_integer_bytes(item, label=f"{label}[{ordinal}]"),
            label=f"{label}[{ordinal}]",
        ))
    return bytes(result)


def portable_sha256_v1(
        domain: bytes,
        records: tuple[tuple[int, ...], ...],
        ) -> bytes:
    """按冻结的双层 framing 计算 SHA-256 raw 32-byte identity。"""
    raw_domain = _u8_bytes(domain, label="SHA domain")
    if not raw_domain:
        raise PublicSourcePayloadProviderError("SHA domain 不得为空")
    if not isinstance(records, tuple):
        raise PublicSourcePayloadProviderError("SHA records 必须是 tuple")
    digest = hashlib.sha256()
    digest.update(_framed_bytes(raw_domain, label="SHA domain"))
    digest.update(_u64_big_endian(len(records), label="SHA record count"))
    for ordinal, record in enumerate(records):
        digest.update(_framed_bytes(
            portable_integer_record_bytes_v1(
                record,
                label=f"SHA record[{ordinal}]",
            ),
            label=f"SHA record[{ordinal}]",
        ))
    return digest.digest()


def public_source_payload_sha256_v1(payload: bytes) -> bytes:
    """将 raw u8[] 映射为标准 SHA-256 raw 32-byte digest。"""
    return hashlib.sha256(_u8_bytes(payload, label="payload")).digest()


def _logical_key_sort_record(value: bytes) -> tuple[int, ...]:
    """显式投影 ASCII key 的 unsigned byte 排序键。"""
    return tuple(value)


def _validate_logical_key(value: bytes, *, label: str) -> bytes:
    """只接受冻结 registry 中的规范 ASCII logical key。"""
    key = _u8_bytes(value, label=label)
    if (not key or any(byte < 0x20 or byte > 0x7E for byte in key)
            or key not in PUBLIC_SOURCE_PAYLOAD_LOGICAL_KEYS_V1):
        raise PublicSourcePayloadProviderError(
            f"{label} 不是已登记的 ASCII logical key")
    return key


def _validate_digest(value: bytes, *, label: str) -> bytes:
    """验证 SHA-256 以明确 raw 32-byte form 进入 record。"""
    digest = _u8_bytes(value, label=label)
    if len(digest) != 32:
        raise PublicSourcePayloadProviderError(f"{label} 必须是 32-byte SHA-256")
    return digest


def _payload_record_canonical_record(
        logical_key: bytes,
        raw_payload: bytes,
        payload_length: int,
        raw_sha256: bytes,
        result_code: int,
        ) -> tuple[int, ...]:
    """导出单条 payload 的有序整数 record，保留显式长度与 digest。"""
    result = [PUBLIC_SOURCE_PAYLOAD_RECORD_V1, result_code]
    _append_record_segment(result, tuple(logical_key))
    result.append(payload_length)
    _append_record_segment(result, tuple(raw_payload))
    _append_record_segment(result, tuple(raw_sha256))
    return tuple(result)


# object-model: value; representation=struct; interop=DLG-RAW-07
@dataclass(frozen=True, slots=True)
class PublicSourcePayloadRecordV1:
    """一条公开 logical resource 的 immutable raw payload record。"""

    logical_key: bytes
    raw_payload: bytes
    payload_length: int
    raw_sha256: bytes
    result_code: int

    def __post_init__(self) -> None:
        """拒绝 key、长度、digest 或成功码的任何内在漂移。"""
        key = _validate_logical_key(self.logical_key, label="payload logical key")
        payload = _u8_bytes(self.raw_payload, label="payload raw bytes")
        length = _u64(self.payload_length, label="payload length")
        digest = _validate_digest(self.raw_sha256, label="payload SHA-256")
        if (type(self.result_code) is not int
                or self.result_code != PUBLIC_SOURCE_PAYLOAD_RESULT_OK_V1):
            raise PublicSourcePayloadProviderError(
                "闭包 payload 只能包含成功读取 result code")
        if length != len(payload):
            raise PublicSourcePayloadProviderError("payload 显式长度漂移")
        if digest != public_source_payload_sha256_v1(payload):
            raise PublicSourcePayloadProviderError("payload SHA-256 漂移")
        if key != self.logical_key:
            raise PublicSourcePayloadProviderError("payload logical key 漂移")

    def canonical_record(self) -> tuple[int, ...]:
        """返回不含物理环境的完整有序整数 record。"""
        return _payload_record_canonical_record(
            self.logical_key,
            self.raw_payload,
            self.payload_length,
            self.raw_sha256,
            self.result_code,
        )


def public_source_payload_record_from_u8_v1(
        logical_key: bytes,
        raw_payload: bytes,
        ) -> PublicSourcePayloadRecordV1:
    """由 host 已读取的 raw u8[] 建立自校验成功 payload record。"""
    payload = _u8_bytes(raw_payload, label="payload raw bytes")
    return PublicSourcePayloadRecordV1(
        _validate_logical_key(logical_key, label="payload logical key"),
        payload,
        len(payload),
        public_source_payload_sha256_v1(payload),
        PUBLIC_SOURCE_PAYLOAD_RESULT_OK_V1,
    )


# object-model: value; representation=struct; interop=DLG-RAW-07
@dataclass(frozen=True, slots=True)
class PublicSourcePayloadReadTraceV1:
    """一次无写入物理读取归一后的纯 record，不保留物理位置。"""

    request_ordinal: int
    logical_key: bytes
    raw_payload: bytes
    payload_length: int
    raw_sha256: bytes
    result_code: int
    write_effect_code: int

    def __post_init__(self) -> None:
        """将 trace 限定为一次成功、无写入的 frozen payload read。"""
        _u64(self.request_ordinal, label="read request ordinal", minimum=1)
        PublicSourcePayloadRecordV1(
            self.logical_key,
            self.raw_payload,
            self.payload_length,
            self.raw_sha256,
            self.result_code,
        )
        if (type(self.write_effect_code) is not int
                or self.write_effect_code
                != PUBLIC_SOURCE_PAYLOAD_WRITE_EFFECT_NONE_V1):
            raise PublicSourcePayloadProviderError("payload host read 不得产生写入效果")

    def canonical_record(self) -> tuple[int, ...]:
        """导出含 request 序与无写入证据的有序 trace record。"""
        payload_record = _payload_record_canonical_record(
            self.logical_key,
            self.raw_payload,
            self.payload_length,
            self.raw_sha256,
            self.result_code,
        )
        result = [
            PUBLIC_SOURCE_PAYLOAD_READ_TRACE_RECORD_V1,
            self.request_ordinal,
            self.write_effect_code,
        ]
        _append_record_segment(result, payload_record)
        return tuple(result)


def _closure_canonical_record(
        records: tuple[PublicSourcePayloadRecordV1, ...],
        ) -> tuple[int, ...]:
    """把固定 registry 的所有 payload record 编为一个 canonical closure record。"""
    result = [
        PUBLIC_SOURCE_PAYLOAD_SCHEMA_V1,
        PUBLIC_SOURCE_PAYLOAD_CLOSURE_RECORD_V1,
        len(records),
    ]
    for record in records:
        _append_record_segment(result, record.canonical_record())
    return tuple(result)


def _validate_closure_records(
        records: tuple[PublicSourcePayloadRecordV1, ...],
        ) -> tuple[PublicSourcePayloadRecordV1, ...]:
    """验证精确登记项、无重复且按 unsigned ASCII key 规范排序。"""
    if not isinstance(records, tuple):
        raise PublicSourcePayloadProviderError("payload closure records 必须是 tuple")
    if len(records) != len(PUBLIC_SOURCE_PAYLOAD_LOGICAL_KEYS_V1):
        raise PublicSourcePayloadProviderError("payload closure resource count 漂移")
    if any(type(record) is not PublicSourcePayloadRecordV1 for record in records):
        raise PublicSourcePayloadProviderError("payload closure 含非 payload record")
    keys = tuple(record.logical_key for record in records)
    ordered = tuple(sorted(keys, key=_logical_key_sort_record))
    if keys != ordered:
        raise PublicSourcePayloadProviderError("payload closure logical key 未规范排序")
    if keys != PUBLIC_SOURCE_PAYLOAD_LOGICAL_KEYS_V1:
        raise PublicSourcePayloadProviderError(
            "payload closure 含缺失、重复或额外 logical key")
    return records


def public_source_payload_closure_identity_v1(
        records: tuple[PublicSourcePayloadRecordV1, ...],
        ) -> bytes:
    """从固定 domain 与 canonical payload record 计算 closure identity。"""
    validated = _validate_closure_records(records)
    return portable_sha256_v1(
        PUBLIC_SOURCE_PAYLOAD_CLOSURE_DOMAIN_V1,
        (_closure_canonical_record(validated),),
    )


# object-model: value; representation=struct; interop=DLG-RAW-07
@dataclass(frozen=True, slots=True)
class PublicSourcePayloadClosureV1:
    """精确登记的公开 dialogue payload 逻辑闭包。"""

    records: tuple[PublicSourcePayloadRecordV1, ...]
    closure_identity: bytes

    def __post_init__(self) -> None:
        """使构造器与 builder 都无法跳过 registry 和 identity 核验。"""
        records = _validate_closure_records(self.records)
        identity = _validate_digest(
            self.closure_identity,
            label="payload closure identity",
        )
        if identity != public_source_payload_closure_identity_v1(records):
            raise PublicSourcePayloadProviderError("payload closure identity 漂移")

    def canonical_record(self) -> tuple[int, ...]:
        """返回 identity 计算所用的完整、路径无关的 payload record。"""
        return _closure_canonical_record(self.records)

    def payload_for(self, logical_key: bytes) -> bytes:
        """按冻结 logical key 读取 raw u8[]；未登记 key 立即拒绝。"""
        key = _validate_logical_key(logical_key, label="payload lookup key")
        for record in self.records:
            if record.logical_key == key:
                return record.raw_payload
        raise PublicSourcePayloadProviderError("payload closure 缺少已登记 logical key")

    def record_for(self, logical_key: bytes) -> PublicSourcePayloadRecordV1:
        """按冻结 logical key 返回完整 record，不建立宿主字典语义。"""
        key = _validate_logical_key(logical_key, label="payload lookup key")
        for record in self.records:
            if record.logical_key == key:
                return record
        raise PublicSourcePayloadProviderError("payload closure 缺少已登记 logical key")


def build_public_source_payload_closure_v1(
        records: tuple[PublicSourcePayloadRecordV1, ...],
        ) -> PublicSourcePayloadClosureV1:
    """从任意输入顺序的完整 record 集构造规范排序 closure。"""
    if not isinstance(records, tuple):
        raise PublicSourcePayloadProviderError("payload closure 输入必须是 tuple")
    if any(type(record) is not PublicSourcePayloadRecordV1 for record in records):
        raise PublicSourcePayloadProviderError("payload closure 输入含非 payload record")
    ordered = tuple(sorted(records, key=lambda record: _logical_key_sort_record(
        record.logical_key)))
    identity = public_source_payload_closure_identity_v1(ordered)
    return PublicSourcePayloadClosureV1(ordered, identity)


def require_public_source_payload_closure_identity_v1(
        closure: PublicSourcePayloadClosureV1,
        expected_identity: bytes,
        ) -> PublicSourcePayloadClosureV1:
    """在恢复或绑定前逐 byte 核验 closure identity，漂移时不返回闭包。"""
    if type(closure) is not PublicSourcePayloadClosureV1:
        raise PublicSourcePayloadProviderError("payload closure 类型错误")
    expected = _validate_digest(
        expected_identity,
        label="expected payload closure identity",
    )
    if closure.closure_identity != expected:
        raise PublicSourcePayloadProviderError("payload closure identity 与绑定不一致")
    return closure


__all__ = [
    "PUBLIC_SOURCE_PAYLOAD_CLOSURE_DOMAIN_V1",
    "PUBLIC_SOURCE_PAYLOAD_CLOSURE_RECORD_V1",
    "PUBLIC_SOURCE_PAYLOAD_LOGICAL_KEYS_V1",
    "PUBLIC_SOURCE_PAYLOAD_PROVIDER_RESULT_CODES_V1",
    "PUBLIC_SOURCE_PAYLOAD_READ_TRACE_RECORD_V1",
    "PUBLIC_SOURCE_PAYLOAD_RECORD_V1",
    "PUBLIC_SOURCE_PAYLOAD_RESULT_INTEGRITY_FAILURE_V1",
    "PUBLIC_SOURCE_PAYLOAD_RESULT_INVALID_LOGICAL_KEY_V1",
    "PUBLIC_SOURCE_PAYLOAD_RESULT_OK_V1",
    "PUBLIC_SOURCE_PAYLOAD_RESULT_PATH_ESCAPE_V1",
    "PUBLIC_SOURCE_PAYLOAD_RESULT_READ_FAILURE_V1",
    "PUBLIC_SOURCE_PAYLOAD_RESULT_RESOURCE_MISSING_V1",
    "PUBLIC_SOURCE_PAYLOAD_RESULT_ROOT_UNAVAILABLE_V1",
    "PUBLIC_SOURCE_PAYLOAD_RESULT_SYMLINK_REJECTED_V1",
    "PUBLIC_SOURCE_PAYLOAD_SCHEMA_V1",
    "PUBLIC_SOURCE_PAYLOAD_WRITE_EFFECT_NONE_V1",
    "PublicSourcePayloadClosureV1",
    "PublicSourcePayloadProviderError",
    "PublicSourcePayloadReadTraceV1",
    "PublicSourcePayloadRecordV1",
    "build_public_source_payload_closure_v1",
    "portable_integer_record_bytes_v1",
    "portable_sha256_v1",
    "public_source_payload_closure_identity_v1",
    "public_source_payload_record_from_u8_v1",
    "public_source_payload_sha256_v1",
    "require_public_source_payload_closure_identity_v1",
]
