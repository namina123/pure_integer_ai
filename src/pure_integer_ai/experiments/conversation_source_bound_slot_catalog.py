"""DLG-RAW-06：公开来源绑定实体槽组合的 logical closure 目录与解析器。

本模块不接触 terminal、会话、SQLite、Memory 或回答表层。它只把已经通过
DLG-RAW-00 的 scalar sequence 与 DLG-RAW-07 闭包中固定、内容锁的公开 source
组合为一个完整 ``PublicFrame``。调用者必须先运行 static exact matcher；本模块
拒绝把 static surface 当作组合结果。所有可观察结果都可以导出为长度前缀整数
record，且不接触物理文件系统。
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

from pure_integer_ai.cognition.shared.identity import (
    ObjectIdentity,
    SourceRef,
    language_atom_identity,
    representation_identity,
    structure_concept_identity,
)
from pure_integer_ai.cognition.shared.semantic_object import semantic_source
from pure_integer_ai.experiments.conversation_public_frame_catalog import (
    PUBLIC_FRAME_CONTEXT_NONE,
    PublicFrame,
    PublicFrameCatalog,
    PublicFrameCatalogError,
    PublicFrameLexicalRoute,
    PublicFrameQuestionTemplate,
    PublicFrameResponseActRuntimeRecipe,
    PublicFrameRuntimeRecipe,
    PublicFrameSourceRecord,
)
from pure_integer_ai.experiments.conversation_raw_intake import (
    DLG_RAW_ACCEPT,
    DLG_RAW_REJECT_CONSTRUCTION_MISS,
    DLG_RAW_REJECT_LEXICAL_AMBIGUOUS,
    DLG_RAW_REJECT_LEXICAL_MISS,
    DLG_RAW_REJECT_SOURCE_CONFLICT,
    encode_utf8_v1,
    intake_raw_conversation_vector,
)
from pure_integer_ai.experiments.conversation_public_source_payload_provider import (
    PublicSourcePayloadClosureV1,
    PublicSourcePayloadProviderError,
    public_source_payload_sha256_v1,
)
from pure_integer_ai.experiments.ph2_dataset_core import (
    DatasetContractError,
    parse_canonical_json_bytes,
)


SOURCE_BOUND_SLOT_CATALOG_SCHEMA_V1 = 1
SOURCE_BOUND_SLOT_CATALOG_SCHEMA_V2 = 2
SOURCE_BOUND_SLOT_CATALOG_SCHEMA_V3 = 3
SOURCE_BOUND_SLOT_CATALOG_RECORD_V1 = 1
SOURCE_BOUND_SLOT_CATALOG_RECORD_V2 = 2
SOURCE_BOUND_SLOT_CATALOG_RECORD_V3 = 3
SOURCE_BOUND_SLOT_FAMILY_RECORD_V1 = 1
SOURCE_BOUND_SLOT_FAMILY_RECORD_V2 = 2
SOURCE_BOUND_SLOT_FAMILY_RECORD_V3 = 3
SOURCE_BOUND_SLOT_BINDING_RECORD_V1 = 1
SOURCE_BOUND_SLOT_BINDING_RECORD_V2 = 2
SOURCE_BOUND_SLOT_BINDING_RECORD_V3 = 3
SOURCE_BOUND_SLOT_RESOLUTION_RECORD_V1 = 1
SOURCE_BOUND_SLOT_RESOLUTION_RECORD_V2 = 2
SOURCE_BOUND_SLOT_RESOLUTION_RECORD_V3 = 3
SOURCE_BOUND_SLOT_TARGET_CANDIDATE_RECORD_V3 = 1
SOURCE_BOUND_SLOT_TARGET_CANDIDATE_KEY_RECORD_V3 = 1
SOURCE_BOUND_SLOT_PAIR_RECORD_V3 = 1
SOURCE_BOUND_SLOT_CANDIDATE_SUPPORTED_V3 = 1
SOURCE_BOUND_SLOT_CANDIDATE_CONFLICTED_V3 = 2
SOURCE_BOUND_SLOT_FRAME_DOMAIN_V1 = b"PURE-INTEGER-AI/DLG-RAW-06/FRAME/V1"
SOURCE_BOUND_SLOT_CATALOG_DOMAIN_V1 = b"PURE-INTEGER-AI/DLG-RAW-06/CATALOG/V1"
SOURCE_BOUND_SLOT_TRACE_DOMAIN_V1 = b"PURE-INTEGER-AI/DLG-RAW-06/TRACE/V1"
SOURCE_BOUND_SLOT_FRAME_DOMAIN_V2 = b"PURE-INTEGER-AI/DLG-RAW-08/FRAME/V1"
SOURCE_BOUND_SLOT_CATALOG_DOMAIN_V2 = b"PURE-INTEGER-AI/DLG-RAW-08/CATALOG/V1"
SOURCE_BOUND_SLOT_TRACE_DOMAIN_V2 = b"PURE-INTEGER-AI/DLG-RAW-08/TRACE/V1"
SOURCE_BOUND_SLOT_FRAME_DOMAIN_V3 = b"PURE-INTEGER-AI/DLG-RAW-09/FRAME/V1"
SOURCE_BOUND_SLOT_CATALOG_DOMAIN_V3 = b"PURE-INTEGER-AI/DLG-RAW-09/CATALOG/V1"
SOURCE_BOUND_SLOT_TRACE_DOMAIN_V3 = b"PURE-INTEGER-AI/DLG-RAW-09/TRACE/V1"
SOURCE_BOUND_SLOT_REPRESENTATION_FAMILY_V1 = (65001, 60, 1)
SOURCE_BOUND_SLOT_CONSTRUCTION_KEY_V1 = (65001, 60, 1)
SOURCE_BOUND_SLOT_TRACE_PREFIX_V1 = (65001, 60, 1)
SOURCE_BOUND_SLOT_REPRESENTATION_FAMILY_V2 = (65001, 61, 1)
SOURCE_BOUND_SLOT_CONSTRUCTION_KEY_V2 = (65001, 61, 1)
SOURCE_BOUND_SLOT_TRACE_PREFIX_V2 = (65001, 61, 1)
SOURCE_BOUND_SLOT_REPRESENTATION_FAMILY_V3 = (65001, 62, 1)
SOURCE_BOUND_SLOT_CONSTRUCTION_KEY_V3 = (65001, 62, 1)
SOURCE_BOUND_SLOT_TRACE_PREFIX_V3 = (65001, 62, 1)
SOURCE_BOUND_SLOT_TYPE_ENTITY_ALIAS_V1 = "ENTITY_ALIAS_V1"
SOURCE_BOUND_SLOT_TYPE_ENTITY_ALIAS_V2 = "ENTITY_ALIAS_V2"
SOURCE_BOUND_SLOT_TYPE_ENTITY_ALIAS_V3 = "ENTITY_ALIAS_V3"
SOURCE_BOUND_SLOT_CATALOG_LOGICAL_KEY_V1 = (
    b"data/ph2/dlg_raw_public_source_bound_slot_v1.jsonl.sample")
SOURCE_BOUND_SLOT_CATALOG_LOGICAL_KEY_V2 = (
    b"data/ph2/dlg_raw_public_source_bound_slot_v2.jsonl.sample")
SOURCE_BOUND_SLOT_CATALOG_LOGICAL_KEY_V3 = (
    b"data/ph2/dlg_raw_public_source_bound_slot_v3.jsonl.sample")

_MANIFEST_FIELDS = frozenset({
    "bindings",
    "catalog_schema",
    "families",
    "source_records",
})
_FAMILY_FIELDS = frozenset({
    "construction_witnesses",
    "family_key",
    "prefix",
    "slot_type",
    "suffix",
})
_FAMILY_WITNESS_FIELDS = frozenset({
    "observed_entity",
    "source_record_id",
})
_BINDING_FIELDS_V1 = frozenset({
    "base_catalog_sha256",
    "base_frame_key",
    "binding_key",
    "entity",
    "entity_witness_record_ids",
})
_BINDING_FIELDS_V2 = frozenset({
    "base_catalog_sha256",
    "base_frame_key",
    "base_frame_raw_sha256",
    "binding_key",
    "entity",
    "negative_relation_source_record_ids",
    "positive_relation_source_record_ids",
})
_SOURCE_RECORD_FIELDS = frozenset({
    "attribution",
    "license_id",
    "raw_sha256",
    "record_id",
    "relative_path",
    "source_ref_key",
    "span",
    "span_utf8_hex",
})
_SURFACE_FIELDS = frozenset({"scalars", "utf8_hex"})
_HEX = frozenset("0123456789abcdef")
_U64_EXCLUSIVE = 1 << 64


# object-model: exception; interop=DLG-RAW-06
class SourceBoundSlotCompositionError(ValueError):
    """DLG-RAW-06 的公开来源、组合闭合或可移植 record 发生漂移。"""


def _pack(result: list[int], value: tuple[int, ...]) -> None:
    """以显式长度前缀追加一段可变长整数值。"""
    result.extend((len(value), *value))


def _strict_int(value: Any, *, label: str, minimum: int | None = None) -> int:
    """拒绝 bool 和整数子类，保持 transport 的跨语言整数边界。"""
    if type(value) is not int or (minimum is not None and value < minimum):
        raise SourceBoundSlotCompositionError(f"{label} 不是合法严格整数")
    return value


def _int_vector(value: Any, *, label: str, allow_empty: bool) -> tuple[int, ...]:
    """从 JSON list 恢复有限严格整数 sequence。"""
    if not isinstance(value, list) or (not allow_empty and not value):
        raise SourceBoundSlotCompositionError(f"{label} 必须是整数 list")
    result = tuple(value)
    for item in result:
        _strict_int(item, label=label)
    return result


def _scalar_vector(value: Any, *, label: str, allow_empty: bool) -> tuple[int, ...]:
    """核验显式 Unicode scalar，不依赖宿主字符串规范化。"""
    result = _int_vector(value, label=label, allow_empty=allow_empty)
    if any(item < 0 or item > 0x10FFFF or 0xD800 <= item <= 0xDFFF
           for item in result):
        raise SourceBoundSlotCompositionError(f"{label} 含非法 Unicode scalar")
    return result


def _ascii_id(value: Any, *, label: str) -> str:
    """限制 manifest key 为有限 ASCII id，避免 locale 与 Unicode 等价影响。"""
    if (not isinstance(value, str) or not value
            or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-"
                   for character in value)):
        raise SourceBoundSlotCompositionError(f"{label} 必须是小写 ASCII record id")
    return value


def _ascii_text(value: Any, *, label: str) -> str:
    """校验公开 attribution/license 等元数据为可运输 ASCII。"""
    if (not isinstance(value, str) or not value
            or any(ord(character) < 0x20 or ord(character) > 0x7E
                   for character in value)):
        raise SourceBoundSlotCompositionError(f"{label} 必须是非空 ASCII 文本")
    return value


def _exact(value: Any, fields: frozenset[str], *, label: str) -> dict[str, Any]:
    """拒绝 JSON transport 的缺字段和尾随字段。"""
    if not isinstance(value, dict) or set(value) != fields:
        raise SourceBoundSlotCompositionError(f"{label} 字段集合漂移")
    return value


def _list(value: Any, *, label: str) -> list[Any]:
    """显式接收 JSON array，禁止 tuple/dict 静默进入 manifest。"""
    if not isinstance(value, list):
        raise SourceBoundSlotCompositionError(f"{label} 必须是 JSON list")
    return value


def _hex_bytes(value: Any, *, label: str, expected_size: int | None = None) -> tuple[int, ...]:
    """从小写 hex 手工恢复 0..255 byte vector。"""
    if (not isinstance(value, str) or len(value) % 2
            or any(character not in _HEX for character in value)):
        raise SourceBoundSlotCompositionError(f"{label} 不是小写 hex")
    result = tuple(
        (int(value[cursor], 16) << 4) | int(value[cursor + 1], 16)
        for cursor in range(0, len(value), 2))
    if expected_size is not None and len(result) != expected_size:
        raise SourceBoundSlotCompositionError(f"{label} 长度不符合合同")
    return result


def _surface(value: Any, *, label: str, allow_empty: bool) -> tuple[int, ...]:
    """交叉核对 scalar 与显式 UTF-8 bytes，得到唯一表层整数序列。"""
    raw = _exact(value, _SURFACE_FIELDS, label=label)
    scalars = _scalar_vector(raw["scalars"], label=f"{label}.scalars",
                             allow_empty=allow_empty)
    encoded = encode_utf8_v1(scalars)
    declared = _hex_bytes(raw["utf8_hex"], label=f"{label}.utf8_hex")
    if encoded != declared:
        raise SourceBoundSlotCompositionError(f"{label} scalar/UTF-8 漂移")
    return scalars


def _sha256(payload: bytes) -> tuple[int, ...]:
    """把固定 public payload SHA-256 适配为规范整数结果。"""
    return tuple(public_source_payload_sha256_v1(payload))


def _logical_payload_key(value: Any, *, label: str) -> tuple[str, bytes]:
    """把 manifest 的 POSIX logical name 映射为 closure 的 ASCII key。"""
    if not isinstance(value, str) or not value or "\\" in value:
        raise SourceBoundSlotCompositionError(f"{label} 不是规范 POSIX logical key")
    try:
        logical_key = value.encode("ascii")
    except UnicodeEncodeError as error:
        raise SourceBoundSlotCompositionError(
            f"{label} 不是 ASCII logical key") from error
    parts = logical_key.split(b"/")
    if (len(parts) != 3 or tuple(parts[:2]) != (b"data", b"ph2")
            or any(part in (b"", b".", b"..") for part in parts)):
        raise SourceBoundSlotCompositionError(
            f"{label} 不在冻结 data/ph2 logical namespace")
    return value, logical_key


def _closure_payload(
        closure: PublicSourcePayloadClosureV1,
        value: Any,
        *,
        label: str,
        ) -> tuple[str, bytes, tuple[int, ...]]:
    """从已验证 closure 取 raw bytes，并显式复核 record 内在不变量。"""
    if type(closure) is not PublicSourcePayloadClosureV1:
        raise SourceBoundSlotCompositionError("source payload closure 类型错误")
    relative_path, logical_key = _logical_payload_key(value, label=label)
    try:
        payload = closure.payload_for(logical_key)
        record = closure.record_for(logical_key)
    except PublicSourcePayloadProviderError as error:
        raise SourceBoundSlotCompositionError(
            f"{label} 未绑定到已登记 public payload") from error
    if (payload != record.raw_payload
            or record.logical_key != logical_key
            or record.payload_length != len(payload)
            or tuple(record.raw_sha256) != _sha256(payload)):
        raise SourceBoundSlotCompositionError(f"{label} closure record 漂移")
    return relative_path, payload, tuple(record.raw_sha256)


def _source_ref(value: Any, *, label: str) -> SourceRef:
    """恢复完整 SourceRef，禁止用局部 source id 偷换来源本体。"""
    key = _int_vector(value, label=label, allow_empty=False)
    try:
        return SourceRef.from_stable_key(key)
    except (TypeError, ValueError) as error:
        raise SourceBoundSlotCompositionError(f"{label} 不是完整 SourceRef") from error


def _decode_source_scalars(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """复用 DLG-RAW-00 严格 UTF-8 状态机读取原始 source span。"""
    intake = intake_raw_conversation_vector(value)
    if not intake.accepted:
        raise SourceBoundSlotCompositionError(f"{label} 不是可接受 UTF-8 source span")
    return intake.unicode_scalars


def _source_payload(
        record: PublicFrameSourceRecord,
        closure: PublicSourcePayloadClosureV1,
        ) -> bytes:
    """由调用方 closure 回读并复核一条已登记的公开 source。"""
    _, payload, payload_sha256 = _closure_payload(
        closure,
        record.relative_path,
        label=f"source {record.record_id} relative path",
    )
    if payload_sha256 != record.raw_sha256:
        raise SourceBoundSlotCompositionError(
            f"source {record.record_id} raw SHA-256 漂移")
    start, end = record.span
    if end > len(payload) or tuple(payload[start:end]) != record.span_bytes:
        raise SourceBoundSlotCompositionError(
            f"source {record.record_id} span bytes 漂移")
    if _decode_source_scalars(
            record.span_bytes, label=f"source {record.record_id} span") != record.span_scalars:
        raise SourceBoundSlotCompositionError(
            f"source {record.record_id} span UTF-8 漂移")
    return payload


def _parse_source_records(
        value: Any,
        *,
        closure: PublicSourcePayloadClosureV1,
        ) -> tuple[PublicFrameSourceRecord, ...]:
    """解析 CC0 source records，并由 closure 校验 hash、span 与 UTF-8。"""
    result = []
    for ordinal, item in enumerate(_list(value, label="source_records")):
        raw = _exact(item, _SOURCE_RECORD_FIELDS,
                     label=f"source_records[{ordinal}]")
        record_id = _ascii_id(raw["record_id"],
                              label=f"source_records[{ordinal}].record_id")
        source = _source_ref(raw["source_ref_key"],
                             label=f"source_records[{ordinal}].source_ref")
        relative_path, payload, payload_sha256 = _closure_payload(
            closure,
            raw["relative_path"],
            label=f"source_records[{ordinal}].relative_path")
        raw_sha256 = _hex_bytes(raw["raw_sha256"],
                                label=f"source_records[{ordinal}].raw_sha256",
                                expected_size=32)
        if raw["license_id"] != "CC0-1.0":
            raise SourceBoundSlotCompositionError(
                "DLG-RAW-06 source 只能使用 CC0-1.0")
        license_id = _ascii_text(raw["license_id"],
                                 label=f"source_records[{ordinal}].license")
        attribution = _ascii_text(raw["attribution"],
                                  label=f"source_records[{ordinal}].attribution")
        span_raw = raw["span"]
        if (not isinstance(span_raw, list) or len(span_raw) != 2):
            raise SourceBoundSlotCompositionError(
                f"source_records[{ordinal}].span 必须是两个整数")
        start = _strict_int(span_raw[0], label=f"source_records[{ordinal}].span[0]",
                            minimum=0)
        end = _strict_int(span_raw[1], label=f"source_records[{ordinal}].span[1]",
                          minimum=0)
        if end <= start:
            raise SourceBoundSlotCompositionError(
                f"source_records[{ordinal}].span 必须是正区间")
        span_bytes = _hex_bytes(raw["span_utf8_hex"],
                                label=f"source_records[{ordinal}].span_utf8_hex")
        if payload_sha256 != raw_sha256:
            raise SourceBoundSlotCompositionError(
                f"source_records[{ordinal}] raw SHA-256 漂移")
        if end > len(payload) or tuple(payload[start:end]) != span_bytes:
            raise SourceBoundSlotCompositionError(
                f"source_records[{ordinal}] span bytes 漂移")
        span_scalars = _decode_source_scalars(
            span_bytes, label=f"source_records[{ordinal}].span")
        result.append(PublicFrameSourceRecord(
            record_id, source, relative_path, raw_sha256, license_id,
            attribution, (start, end), span_bytes, span_scalars,
        ))
    records = tuple(sorted(result, key=lambda item: item.source.stable_key()))
    if (not records
            or len({item.record_id for item in records}) != len(records)
            or len({item.source.stable_key() for item in records}) != len(records)):
        raise SourceBoundSlotCompositionError(
            "source record id 或 SourceRef 不得重复")
    return records


def _text_record(value: str, *, label: str) -> tuple[int, ...]:
    """显式把 ASCII transport 字段投影为 UTF-8 整数 record。"""
    return tuple(_ascii_text(value, label=label).encode("utf-8"))


def _unsigned_integer_bytes(value: int, *, label: str) -> bytes:
    """编码一个非负任意精度整数，避免把 Python int 宽度写进协议。"""
    if type(value) is not int or value < 0:
        raise SourceBoundSlotCompositionError(f"{label} 必须是非负严格整数")
    size = max(1, (value.bit_length() + 7) // 8)
    return value.to_bytes(size, "big")


def _framed_bytes(value: bytes, *, label: str) -> bytes:
    """用固定 u64 big-endian 长度前缀封装原始 bytes。"""
    if len(value) >= _U64_EXCLUSIVE:
        raise SourceBoundSlotCompositionError(f"{label} 超出 u64 framing")
    return len(value).to_bytes(8, "big") + value


def _u64_count(value: int, *, label: str) -> int:
    """把 sequence count 固定为非负 u64，不能让宿主溢出成为协议行为。"""
    if type(value) is not int or value < 0 or value >= _U64_EXCLUSIVE:
        raise SourceBoundSlotCompositionError(f"{label} 超出 u64 count")
    return value


def portable_integer_record_bytes(value: tuple[int, ...], *, label: str) -> bytes:
    """把整数 tuple 变为跨语言可重放的 length-framed SHA 输入。

    布局为 ``u64 count`` 后接每个 ``u64 byte_length || unsigned-big-endian``。
    这是本模块 identity/trace SHA 的规范输入，而不是 Python pickle 或 JSON。
    """
    if not isinstance(value, tuple) or any(type(item) is not int or item < 0
                                           for item in value):
        raise SourceBoundSlotCompositionError(f"{label} 必须是非负严格整数 tuple")
    count = _u64_count(len(value), label=f"{label} count")
    result = bytearray(count.to_bytes(8, "big"))
    for ordinal, item in enumerate(value):
        result.extend(_framed_bytes(
            _unsigned_integer_bytes(item, label=f"{label}[{ordinal}]"),
            label=f"{label}[{ordinal}]"))
    return bytes(result)


def portable_sha256_v1(
        domain: bytes,
        records: tuple[tuple[int, ...], ...],
        ) -> tuple[int, ...]:
    """以公开 domain 和双层长度 framing 计算模块内 SHA-256 identity。"""
    if type(domain) is not bytes or not domain:
        raise TypeError("SHA domain 必须是非空 bytes")
    if not isinstance(records, tuple):
        raise TypeError("SHA records 必须是 tuple")
    digest = hashlib.sha256()
    digest.update(_framed_bytes(domain, label="SHA domain"))
    digest.update(_u64_count(len(records), label="SHA record count").to_bytes(
        8, "big"))
    for ordinal, record in enumerate(records):
        digest.update(_framed_bytes(
            portable_integer_record_bytes(record, label=f"SHA record[{ordinal}]"),
            label=f"SHA record[{ordinal}]"))
    return tuple(digest.digest())


# object-model: value; representation=struct; interop=DLG-RAW-06
@dataclass(frozen=True, slots=True)
class QuestionConstructionFamily:
    """一个固定 prefix/slot/suffix 构式及两个不同实体的公开观察。"""

    family_key: str
    slot_type: str
    prefix_scalars: tuple[int, ...]
    suffix_scalars: tuple[int, ...]
    witnesses: tuple[tuple[tuple[int, ...], PublicFrameSourceRecord], ...]

    def __post_init__(self) -> None:
        """冻结构式 slot、双 witness 与按 SourceRef 的确定顺序。"""
        _ascii_id(self.family_key, label="family key")
        if self.slot_type not in {
                SOURCE_BOUND_SLOT_TYPE_ENTITY_ALIAS_V1,
                SOURCE_BOUND_SLOT_TYPE_ENTITY_ALIAS_V2,
                SOURCE_BOUND_SLOT_TYPE_ENTITY_ALIAS_V3}:
            raise SourceBoundSlotCompositionError("family slot type 未注册")
        if (not isinstance(self.prefix_scalars, tuple)
                or not isinstance(self.suffix_scalars, tuple)):
            raise SourceBoundSlotCompositionError(
                "family prefix/suffix 必须是不可变 scalar tuple")
        _scalar_vector(list(self.prefix_scalars), label="family prefix",
                       allow_empty=True)
        _scalar_vector(list(self.suffix_scalars), label="family suffix",
                       allow_empty=True)
        if not self.prefix_scalars and not self.suffix_scalars:
            raise SourceBoundSlotCompositionError("family prefix/suffix 不得同时为空")
        if self.prefix_scalars:
            raise SourceBoundSlotCompositionError(
                "当前 source-bound family 只支持空 prefix")
        if (not isinstance(self.witnesses, tuple) or len(self.witnesses) < 2
                or any(not isinstance(item, tuple) or len(item) != 2
                       for item in self.witnesses)
                or any(type(item[1]) is not PublicFrameSourceRecord
                       for item in self.witnesses)
                or self.witnesses != tuple(sorted(
                    self.witnesses,
                    key=lambda item: item[1].source.stable_key()))):
            raise SourceBoundSlotCompositionError("family witness 未规范排序")
        entities = []
        source_keys = []
        for entity, source in self.witnesses:
            if not isinstance(entity, tuple):
                raise SourceBoundSlotCompositionError(
                    "family observed entity 必须是不可变 scalar tuple")
            _scalar_vector(list(entity), label="family observed entity",
                           allow_empty=False)
            if type(source) is not PublicFrameSourceRecord:
                raise TypeError("family witness source 类型错误")
            if source.span_scalars != self.suffix_scalars:
                raise SourceBoundSlotCompositionError(
                    "family witness suffix span 漂移")
            entities.append(entity)
            source_keys.append(source.source.stable_key())
        if len(set(entities)) < 2 or len(set(source_keys)) < 2:
            raise SourceBoundSlotCompositionError(
                "family 必须有两个不同实体的独立构式观察")

    def canonical_record(self) -> tuple[int, ...]:
        """导出构式与其来源观察的可移植整数 record。"""
        result = [
            (SOURCE_BOUND_SLOT_FAMILY_RECORD_V1
             if self.slot_type == SOURCE_BOUND_SLOT_TYPE_ENTITY_ALIAS_V1
             else (SOURCE_BOUND_SLOT_FAMILY_RECORD_V2
                   if self.slot_type == SOURCE_BOUND_SLOT_TYPE_ENTITY_ALIAS_V2
                   else SOURCE_BOUND_SLOT_FAMILY_RECORD_V3)),
        ]
        for value in (
                _text_record(self.family_key, label="family key"),
                _text_record(self.slot_type, label="family slot type"),
                self.prefix_scalars,
                self.suffix_scalars):
            _pack(result, value)
        result.append(len(self.witnesses))
        for entity, source in self.witnesses:
            _pack(result, entity)
            _pack(result, source.canonical_record())
        return tuple(result)


# object-model: value; representation=struct; interop=DLG-RAW-06
@dataclass(frozen=True, slots=True)
class EntityPropositionBinding:
    """一个实体 alias 到已锁定 NONE base frame 的正/反 relation 绑定。"""

    binding_key: str
    entity_scalars: tuple[int, ...]
    witnesses: tuple[PublicFrameSourceRecord, ...]
    base_catalog_sha256: tuple[int, ...]
    base_frame_key: str
    base_frame_raw_sha256: tuple[int, ...] = ()
    negative_witnesses: tuple[PublicFrameSourceRecord, ...] = ()
    catalog_schema: int = SOURCE_BOUND_SLOT_CATALOG_SCHEMA_V1

    def __post_init__(self) -> None:
        """冻结正向 relation、可选反证和 active frame identity。"""
        _ascii_id(self.binding_key, label="binding key")
        if not isinstance(self.entity_scalars, tuple):
            raise SourceBoundSlotCompositionError(
                "binding entity 必须是不可变 scalar tuple")
        _scalar_vector(list(self.entity_scalars), label="binding entity",
                       allow_empty=False)
        if (not isinstance(self.witnesses, tuple) or len(self.witnesses) < 2
                or any(type(item) is not PublicFrameSourceRecord
                       for item in self.witnesses)
                or self.witnesses != tuple(sorted(
                    self.witnesses,
                    key=lambda item: item.source.stable_key()))
                or any(item.span_scalars != self.entity_scalars
                       for item in self.witnesses)
                or len({item.source.stable_key() for item in self.witnesses}) < 2):
            raise SourceBoundSlotCompositionError(
                "binding 必须有两个独立实体 lexical witness")
        if (not isinstance(self.base_catalog_sha256, tuple)
                or len(self.base_catalog_sha256) != 32
                or any(type(item) is not int or item < 0 or item > 255
                       for item in self.base_catalog_sha256)):
                raise SourceBoundSlotCompositionError("binding base catalog SHA 非法")
        _ascii_id(self.base_frame_key, label="binding base frame key")
        if (not isinstance(self.base_frame_raw_sha256, tuple)
                or (self.base_frame_raw_sha256 and len(
                    self.base_frame_raw_sha256) != 32)
                or any(type(item) is not int or item < 0 or item > 255
                       for item in self.base_frame_raw_sha256)):
            raise SourceBoundSlotCompositionError(
                "binding base frame raw SHA 非法")
        if (not isinstance(self.negative_witnesses, tuple)
                or any(type(item) is not PublicFrameSourceRecord
                       for item in self.negative_witnesses)
                or self.negative_witnesses != tuple(sorted(
                    self.negative_witnesses,
                    key=lambda item: item.source.stable_key()))
                or any(item.span_scalars != self.entity_scalars
                       for item in self.negative_witnesses)
                or len({item.source.stable_key()
                        for item in self.negative_witnesses})
                != len(self.negative_witnesses)
                or ({item.source.stable_key() for item in self.witnesses}
                    & {item.source.stable_key()
                       for item in self.negative_witnesses})):
            raise SourceBoundSlotCompositionError(
                "binding negative relation witness 未规范闭合")
        if (type(self.catalog_schema) is not int
                or self.catalog_schema not in {
                    SOURCE_BOUND_SLOT_CATALOG_SCHEMA_V1,
                    SOURCE_BOUND_SLOT_CATALOG_SCHEMA_V2,
                    SOURCE_BOUND_SLOT_CATALOG_SCHEMA_V3}):
            raise SourceBoundSlotCompositionError("binding catalog schema 未注册")
        if self.catalog_schema == SOURCE_BOUND_SLOT_CATALOG_SCHEMA_V1:
            if self.base_frame_raw_sha256 or self.negative_witnesses:
                raise SourceBoundSlotCompositionError(
                    "V1 binding 不得携带 relation 扩展字段")
        elif not self.base_frame_raw_sha256:
            raise SourceBoundSlotCompositionError(
                "V2/V3 binding 缺 base frame raw SHA")

    def canonical_record(self) -> tuple[int, ...]:
        """导出不含回答表层、可跨语言重放的 relation binding record。"""
        result = [
            (SOURCE_BOUND_SLOT_BINDING_RECORD_V1
             if self.catalog_schema == SOURCE_BOUND_SLOT_CATALOG_SCHEMA_V1
             else (SOURCE_BOUND_SLOT_BINDING_RECORD_V2
                   if self.catalog_schema == SOURCE_BOUND_SLOT_CATALOG_SCHEMA_V2
                   else SOURCE_BOUND_SLOT_BINDING_RECORD_V3)),
        ]
        for value in (
                _text_record(self.binding_key, label="binding key"),
                self.entity_scalars,
                self.base_catalog_sha256,
                _text_record(self.base_frame_key, label="base frame key")):
            _pack(result, value)
        result.append(len(self.witnesses))
        for source in self.witnesses:
            _pack(result, source.canonical_record())
        if self.catalog_schema != SOURCE_BOUND_SLOT_CATALOG_SCHEMA_V1:
            _pack(result, self.base_frame_raw_sha256)
            result.append(len(self.negative_witnesses))
            for source in self.negative_witnesses:
                _pack(result, source.canonical_record())
        return tuple(result)


# object-model: value; representation=struct; interop=DLG-RAW-06
@dataclass(frozen=True, slots=True)
class SourceBoundSlotCompositionCatalog:
    """可删索引之外的公开 source-bound family/binding 权威 record。"""

    manifest_sha256: tuple[int, ...]
    base_catalog_sha256: tuple[int, ...]
    source_payload_closure_identity: tuple[int, ...]
    source_records: tuple[PublicFrameSourceRecord, ...]
    families: tuple[QuestionConstructionFamily, ...]
    bindings: tuple[EntityPropositionBinding, ...]
    catalog_schema: int = SOURCE_BOUND_SLOT_CATALOG_SCHEMA_V1
    manifest_logical_key: str = (
        SOURCE_BOUND_SLOT_CATALOG_LOGICAL_KEY_V1.decode("ascii"))

    def __post_init__(self) -> None:
        """核验规范排序、唯一 key 与无路径的 closure binding。"""
        for value, label in (
                (self.manifest_sha256, "manifest SHA"),
                (self.base_catalog_sha256, "base catalog SHA"),
                (self.source_payload_closure_identity, "source payload closure identity")):
            if (not isinstance(value, tuple) or len(value) != 32
                    or any(type(item) is not int or item < 0 or item > 255
                           for item in value)):
                raise SourceBoundSlotCompositionError(f"{label} 非法")
        if (type(self.catalog_schema) is not int
                or self.catalog_schema not in {
                SOURCE_BOUND_SLOT_CATALOG_SCHEMA_V1,
                SOURCE_BOUND_SLOT_CATALOG_SCHEMA_V2,
                SOURCE_BOUND_SLOT_CATALOG_SCHEMA_V3}):
            raise SourceBoundSlotCompositionError("catalog schema 未注册")
        if (not self.source_records
                or self.source_records != tuple(sorted(
                    self.source_records,
                    key=lambda item: item.source.stable_key()))
                or len({item.record_id for item in self.source_records})
                != len(self.source_records)):
            raise SourceBoundSlotCompositionError("catalog source records 未规范排序")
        if (not self.families
                or self.families != tuple(sorted(
                    self.families, key=QuestionConstructionFamily.canonical_record))
                or len({item.family_key for item in self.families})
                != len(self.families)):
            raise SourceBoundSlotCompositionError("catalog families 未规范排序")
        if (not self.bindings
                or self.bindings != tuple(sorted(
                    self.bindings, key=EntityPropositionBinding.canonical_record))
                or len({item.binding_key for item in self.bindings})
                != len(self.bindings)
                or any(item.base_catalog_sha256 != self.base_catalog_sha256
                       for item in self.bindings)
                or any(item.catalog_schema != self.catalog_schema
                       for item in self.bindings)):
            raise SourceBoundSlotCompositionError("catalog bindings 未规范排序")
        expected_logical_key = (
            SOURCE_BOUND_SLOT_CATALOG_LOGICAL_KEY_V1
            if self.catalog_schema == SOURCE_BOUND_SLOT_CATALOG_SCHEMA_V1
            else (SOURCE_BOUND_SLOT_CATALOG_LOGICAL_KEY_V2
                  if self.catalog_schema == SOURCE_BOUND_SLOT_CATALOG_SCHEMA_V2
                  else SOURCE_BOUND_SLOT_CATALOG_LOGICAL_KEY_V3)).decode("ascii")
        if self.manifest_logical_key != expected_logical_key:
            raise SourceBoundSlotCompositionError(
                "catalog manifest logical key 与 schema 不一致")
        if self.catalog_schema == SOURCE_BOUND_SLOT_CATALOG_SCHEMA_V1:
            if any(item.base_frame_raw_sha256 or item.negative_witnesses
                   for item in self.bindings):
                raise SourceBoundSlotCompositionError(
                    "V1 binding 不得携带 V2 relation 字段")
        elif any(not item.base_frame_raw_sha256 for item in self.bindings):
            raise SourceBoundSlotCompositionError(
                "V2/V3 binding 缺 base frame raw SHA")
        known = {item.record_id: item for item in self.source_records}
        for family in self.families:
            if any(known.get(source.record_id) != source
                   for _, source in family.witnesses):
                raise SourceBoundSlotCompositionError(
                    "family witness 指向外部 source record")
        for binding in self.bindings:
            if any(known.get(source.record_id) != source
                   for source in (*binding.witnesses,
                                  *binding.negative_witnesses)):
                raise SourceBoundSlotCompositionError(
                    "binding witness 指向外部 source record")

    def verify_sources(self, closure: PublicSourcePayloadClosureV1) -> None:
        """以同一 closure identity 回读 manifest 与每份公开 lexical source。"""
        if type(closure) is not PublicSourcePayloadClosureV1:
            raise SourceBoundSlotCompositionError("source payload closure 类型错误")
        if tuple(closure.closure_identity) != self.source_payload_closure_identity:
            raise SourceBoundSlotCompositionError("source payload closure identity 漂移")
        _, manifest_payload, manifest_sha256 = _closure_payload(
            closure,
            self.manifest_logical_key,
            label="composition manifest logical key",
        )
        if (manifest_sha256 != self.manifest_sha256
                or _sha256(manifest_payload) != self.manifest_sha256):
            raise SourceBoundSlotCompositionError("composition manifest SHA 漂移")
        for source in self.source_records:
            _source_payload(source, closure)

    def canonical_record(self) -> tuple[int, ...]:
        """导出不含宿主路径或缓存的完整 source-bound catalog record。"""
        result = [
            (SOURCE_BOUND_SLOT_CATALOG_RECORD_V1
             if self.catalog_schema == SOURCE_BOUND_SLOT_CATALOG_SCHEMA_V1
             else (SOURCE_BOUND_SLOT_CATALOG_RECORD_V2
                   if self.catalog_schema == SOURCE_BOUND_SLOT_CATALOG_SCHEMA_V2
                   else SOURCE_BOUND_SLOT_CATALOG_RECORD_V3)),
        ]
        if self.catalog_schema != SOURCE_BOUND_SLOT_CATALOG_SCHEMA_V1:
            _pack(result, (self.catalog_schema,))
            _pack(result, _text_record(
                self.manifest_logical_key,
                label="manifest logical key"))
        for value in (
                self.manifest_sha256,
                self.base_catalog_sha256,
                self.source_payload_closure_identity):
            _pack(result, value)
        for values in (self.source_records, self.families, self.bindings):
            result.append(len(values))
            for item in values:
                _pack(result, item.canonical_record())
        return tuple(result)


# object-model: value; representation=struct; interop=DLG-RAW-09
@dataclass(frozen=True, slots=True)
class SourceBoundSlotTargetCandidateV3:
    """一个 V3 target 候选的完整整数化语义与 relation verdict 见证。"""

    target_key: tuple[int, ...]
    query_kind_key: tuple[int, ...]
    intent_key: tuple[int, ...]
    goal_kind_key: tuple[int, ...]
    required_key: tuple[int, ...]
    evidence_scope_key: tuple[int, ...]
    response_scope_key: tuple[int, ...]
    target_branch_key: tuple[int, ...]
    authorized_target_keys: tuple[tuple[int, ...], ...]
    recipe_record: tuple[int, ...]
    context_requirement: int
    context_target_key: tuple[int, ...]
    base_frame_key: str
    base_frame_raw_sha256: tuple[int, ...]
    pair_records: tuple[tuple[int, ...], ...]
    verdict: int

    def __post_init__(self) -> None:
        """拒绝宿主可变值、未闭合 target 或非规范 witness 排序。"""
        for value, label in (
                (self.target_key, "candidate target"),
                (self.query_kind_key, "candidate query kind"),
                (self.intent_key, "candidate intent"),
                (self.goal_kind_key, "candidate goal kind"),
                (self.required_key, "candidate required state"),
                (self.evidence_scope_key, "candidate evidence scope"),
                (self.response_scope_key, "candidate response scope"),
                (self.target_branch_key, "candidate target branch"),
                (self.recipe_record, "candidate runtime recipe")):
            if (not isinstance(value, tuple) or not value
                    or any(type(item) is not int or item < 0
                           for item in value)):
                raise SourceBoundSlotCompositionError(
                    f"{label} 必须是非空严格整数 tuple")
        if (not isinstance(self.context_target_key, tuple)
                or any(type(item) is not int or item < 0
                       for item in self.context_target_key)):
            raise SourceBoundSlotCompositionError(
                "candidate context target 必须是严格整数 tuple")
        if (type(self.context_requirement) is not int
                or self.context_requirement < 0):
            raise SourceBoundSlotCompositionError(
                "candidate context requirement 非法")
        if (not isinstance(self.authorized_target_keys, tuple)
                or not self.authorized_target_keys
                or any(not isinstance(item, tuple) or not item
                       or any(type(value) is not int or value < 0
                              for value in item)
                       for item in self.authorized_target_keys)
                or self.authorized_target_keys != tuple(sorted(
                    self.authorized_target_keys))
                or len(set(self.authorized_target_keys))
                != len(self.authorized_target_keys)
                or self.target_key not in self.authorized_target_keys):
            raise SourceBoundSlotCompositionError(
                "candidate authorized target 未规范闭合")
        _ascii_id(self.base_frame_key, label="candidate base frame key")
        if (not isinstance(self.base_frame_raw_sha256, tuple)
                or len(self.base_frame_raw_sha256) != 32
                or any(type(item) is not int or item < 0 or item > 255
                       for item in self.base_frame_raw_sha256)):
            raise SourceBoundSlotCompositionError(
                "candidate base frame raw SHA 非法")
        if (not isinstance(self.pair_records, tuple)
                or not self.pair_records
                or any(not isinstance(item, tuple) or not item
                       or any(type(value) is not int or value < 0
                              for value in item)
                       for item in self.pair_records)
                or self.pair_records != tuple(sorted(self.pair_records))
                or len(set(self.pair_records)) != len(self.pair_records)):
            raise SourceBoundSlotCompositionError(
                "candidate pair witness 未规范排序")
        if (type(self.verdict) is not int or self.verdict not in {
                SOURCE_BOUND_SLOT_CANDIDATE_SUPPORTED_V3,
                SOURCE_BOUND_SLOT_CANDIDATE_CONFLICTED_V3}):
            raise SourceBoundSlotCompositionError("candidate verdict 未注册")

    def equivalence_record(self) -> tuple[int, ...]:
        """导出不含 relation identity 的 target 语义等价 record。"""
        result = [SOURCE_BOUND_SLOT_TARGET_CANDIDATE_KEY_RECORD_V3]
        for value in (
                self.target_key,
                self.query_kind_key,
                self.intent_key,
                self.goal_kind_key,
                self.required_key,
                self.evidence_scope_key,
                self.response_scope_key,
                self.target_branch_key,
                self.recipe_record,
                self.context_target_key):
            _pack(result, value)
        result.extend((self.context_requirement, len(self.authorized_target_keys)))
        for target_key in self.authorized_target_keys:
            _pack(result, target_key)
        return tuple(result)

    def canonical_record(self) -> tuple[int, ...]:
        """导出含 base 与 witness verdict 的可跨语言候选审计 record。"""
        result = [SOURCE_BOUND_SLOT_TARGET_CANDIDATE_RECORD_V3]
        _pack(result, self.equivalence_record())
        _pack(result, _text_record(
            self.base_frame_key, label="candidate base frame key"))
        _pack(result, self.base_frame_raw_sha256)
        result.extend((self.verdict, len(self.pair_records)))
        for pair_record in self.pair_records:
            _pack(result, pair_record)
        return tuple(result)


# object-model: value; representation=struct; interop=DLG-RAW-06
@dataclass(frozen=True, slots=True)
class SourceBoundSlotCompositionResolution:
    """一次 raw scalar 的零/唯一/多组合结果，唯一时给出可直接 ingress 的 catalog。"""

    result_code: int
    matched_frame_count: int
    input_scalars: tuple[int, ...]
    catalog: SourceBoundSlotCompositionCatalog
    frame: PublicFrame | None = None
    public_frame_catalog: PublicFrameCatalog | None = None
    target_candidates: tuple[SourceBoundSlotTargetCandidateV3, ...] = ()

    def __post_init__(self) -> None:
        """拒绝失败路径携带动态 frame/catalog，避免调用者忽略 result code。"""
        if (type(self.result_code) is not int
                or self.result_code not in {
                DLG_RAW_ACCEPT,
                DLG_RAW_REJECT_LEXICAL_MISS,
                DLG_RAW_REJECT_LEXICAL_AMBIGUOUS,
                DLG_RAW_REJECT_CONSTRUCTION_MISS,
                DLG_RAW_REJECT_SOURCE_CONFLICT}):
            raise SourceBoundSlotCompositionError("composition result code 未注册")
        if (type(self.matched_frame_count) is not int
                or self.matched_frame_count < 0
                or self.matched_frame_count >= _U64_EXCLUSIVE):
            raise SourceBoundSlotCompositionError("composition matched frame count 非法")
        if not isinstance(self.input_scalars, tuple):
            raise SourceBoundSlotCompositionError(
                "resolution input 必须是不可变 scalar tuple")
        _scalar_vector(list(self.input_scalars), label="resolution input",
                       allow_empty=True)
        if not isinstance(self.catalog, SourceBoundSlotCompositionCatalog):
            raise TypeError("resolution 缺 composition catalog")
        if (not isinstance(self.target_candidates, tuple)
                or any(type(item) is not SourceBoundSlotTargetCandidateV3
                       for item in self.target_candidates)):
            raise SourceBoundSlotCompositionError(
                "resolution target candidates 必须是不可变 V3 struct tuple")
        if self.result_code == DLG_RAW_ACCEPT:
            if (self.matched_frame_count != 1
                    or not isinstance(self.frame, PublicFrame)
                    or not isinstance(self.public_frame_catalog, PublicFrameCatalog)
                    or self.public_frame_catalog.frames != (self.frame,)):
                raise SourceBoundSlotCompositionError("unique composition 未形成完整 frame catalog")
        elif ((self.result_code == DLG_RAW_REJECT_LEXICAL_MISS
               and (self.matched_frame_count != 0 or self.frame is not None
                    or self.public_frame_catalog is not None))
              or (self.result_code == DLG_RAW_REJECT_LEXICAL_AMBIGUOUS
                  and (self.matched_frame_count < 2 or self.frame is not None
                       or self.public_frame_catalog is not None))
              or (self.result_code == DLG_RAW_REJECT_CONSTRUCTION_MISS
                  and (self.matched_frame_count < 1
                       or (self.matched_frame_count != 1
                           and (self.frame is not None
                                or self.public_frame_catalog is not None))
                       or (self.matched_frame_count == 1
                           and ((self.frame is None)
                                != (self.public_frame_catalog is None)))
                       or (self.frame is not None
                           and (not isinstance(self.frame, PublicFrame)
                                or not isinstance(self.public_frame_catalog,
                                                  PublicFrameCatalog)
                                or self.public_frame_catalog.frames
                                != (self.frame,)))))):
            raise SourceBoundSlotCompositionError("failed composition result 不闭合")
        if (self.result_code == DLG_RAW_REJECT_SOURCE_CONFLICT
                and (self.matched_frame_count < 1
                     or self.frame is not None
                     or self.public_frame_catalog is not None)):
            raise SourceBoundSlotCompositionError(
                "source conflict result 不得携带 frame")
        if self.catalog.catalog_schema != SOURCE_BOUND_SLOT_CATALOG_SCHEMA_V3:
            if self.target_candidates:
                raise SourceBoundSlotCompositionError(
                    "V1/V2 resolution 不得携带 V3 target candidates")
            return
        if (self.target_candidates != tuple(sorted(
                self.target_candidates,
                key=SourceBoundSlotTargetCandidateV3.equivalence_record))
                or len({item.equivalence_record()
                        for item in self.target_candidates})
                != len(self.target_candidates)
                or len({item.target_key for item in self.target_candidates})
                != len(self.target_candidates)):
            raise SourceBoundSlotCompositionError(
                "V3 target candidates 未按等价 record 规范闭合")
        if self.result_code == DLG_RAW_ACCEPT:
            if (len(self.target_candidates) != 1
                    or self.target_candidates[0].verdict
                    != SOURCE_BOUND_SLOT_CANDIDATE_SUPPORTED_V3):
                raise SourceBoundSlotCompositionError(
                    "V3 unique target candidate 未闭合")
        elif self.result_code == DLG_RAW_REJECT_LEXICAL_MISS:
            if self.target_candidates:
                raise SourceBoundSlotCompositionError(
                    "V3 lexical miss 不得携带 target candidate")
        elif self.result_code == DLG_RAW_REJECT_LEXICAL_AMBIGUOUS:
            if (len(self.target_candidates) != self.matched_frame_count
                    or any(item.verdict != SOURCE_BOUND_SLOT_CANDIDATE_SUPPORTED_V3
                           for item in self.target_candidates)):
                raise SourceBoundSlotCompositionError(
                    "V3 ambiguity target candidates 未闭合")
        elif self.result_code == DLG_RAW_REJECT_SOURCE_CONFLICT:
            if (len(self.target_candidates) != self.matched_frame_count
                    or not any(item.verdict
                               == SOURCE_BOUND_SLOT_CANDIDATE_CONFLICTED_V3
                               for item in self.target_candidates)):
                raise SourceBoundSlotCompositionError(
                    "V3 source conflict target candidates 未闭合")
        elif self.result_code == DLG_RAW_REJECT_CONSTRUCTION_MISS:
            if (self.frame is not None or self.public_frame_catalog is not None
                    or self.target_candidates):
                raise SourceBoundSlotCompositionError(
                    "V3 construction miss 不得携带 frame 或 candidate")

    @property
    def accepted(self) -> bool:
        """只有唯一组合已物化完整动态 frame 时返回真。"""
        return self.result_code == DLG_RAW_ACCEPT

    def canonical_record(self) -> tuple[int, ...]:
        """导出不依赖 Python object identity 的解析结果 record。"""
        result = [
            (SOURCE_BOUND_SLOT_RESOLUTION_RECORD_V1
             if self.catalog.catalog_schema == SOURCE_BOUND_SLOT_CATALOG_SCHEMA_V1
             else (SOURCE_BOUND_SLOT_RESOLUTION_RECORD_V2
                   if self.catalog.catalog_schema == SOURCE_BOUND_SLOT_CATALOG_SCHEMA_V2
                   else SOURCE_BOUND_SLOT_RESOLUTION_RECORD_V3)),
            self.result_code,
            self.matched_frame_count,
        ]
        for value in (
                self.input_scalars,
                self.catalog.canonical_record(),
                (() if self.frame is None else self.frame.canonical_record()),
                (() if self.public_frame_catalog is None
                 else self.public_frame_catalog.canonical_record())):
            _pack(result, value)
        if self.catalog.catalog_schema == SOURCE_BOUND_SLOT_CATALOG_SCHEMA_V3:
            result.append(len(self.target_candidates))
            for candidate in self.target_candidates:
                _pack(result, candidate.canonical_record())
        return tuple(result)


def _parse_family(
        value: Any,
        *,
        source_index: dict[str, PublicFrameSourceRecord],
        closure: PublicSourcePayloadClosureV1,
        catalog_schema: int,
        ) -> QuestionConstructionFamily:
    """解析一个构式 family，并由 closure 验证两个不同实体观察。"""
    raw = _exact(value, _FAMILY_FIELDS, label="family")
    family_key = _ascii_id(raw["family_key"], label="family key")
    prefix = _surface(raw["prefix"], label="family prefix", allow_empty=True)
    suffix = _surface(raw["suffix"], label="family suffix", allow_empty=True)
    slot_type = raw["slot_type"]
    expected_slot_type = (
        SOURCE_BOUND_SLOT_TYPE_ENTITY_ALIAS_V1
        if catalog_schema == SOURCE_BOUND_SLOT_CATALOG_SCHEMA_V1
        else (SOURCE_BOUND_SLOT_TYPE_ENTITY_ALIAS_V2
              if catalog_schema == SOURCE_BOUND_SLOT_CATALOG_SCHEMA_V2
              else SOURCE_BOUND_SLOT_TYPE_ENTITY_ALIAS_V3))
    if slot_type != expected_slot_type:
        raise SourceBoundSlotCompositionError("family slot type 未注册")
    if prefix:
        raise SourceBoundSlotCompositionError("当前 source-bound catalog 只支持空 prefix")
    witnesses = []
    for ordinal, item in enumerate(_list(
            raw["construction_witnesses"], label="family witnesses")):
        witness = _exact(item, _FAMILY_WITNESS_FIELDS,
                         label=f"family witness[{ordinal}]")
        record_id = _ascii_id(witness["source_record_id"],
                              label=f"family witness[{ordinal}].source")
        source = source_index.get(record_id)
        if source is None:
            raise SourceBoundSlotCompositionError("family witness source 不存在")
        observed_entity = _surface(
            witness["observed_entity"],
            label=f"family witness[{ordinal}].observed entity",
            allow_empty=False)
        if source.span_scalars != suffix:
            raise SourceBoundSlotCompositionError("family witness suffix span 漂移")
        payload = _source_payload(source, closure)
        entity_bytes = encode_utf8_v1(observed_entity)
        start, _ = source.span
        if start < len(entity_bytes) or tuple(payload[
                start - len(entity_bytes):start]) != entity_bytes:
            raise SourceBoundSlotCompositionError(
                "family witness 未在 source 中紧邻观察到实体与构式")
        witnesses.append((observed_entity, source))
    return QuestionConstructionFamily(
        family_key, slot_type, prefix, suffix,
        tuple(sorted(witnesses, key=lambda item: item[1].source.stable_key())),
    )


def _parse_binding(
        value: Any,
        *,
        source_index: dict[str, PublicFrameSourceRecord],
        catalog_schema: int,
        ) -> EntityPropositionBinding:
    """解析一个实体/base-frame binding，不读取答案、label 或评测表层。"""
    raw = _exact(
        value,
        (_BINDING_FIELDS_V1
         if catalog_schema == SOURCE_BOUND_SLOT_CATALOG_SCHEMA_V1
         else _BINDING_FIELDS_V2),
        label="binding")
    binding_key = _ascii_id(raw["binding_key"], label="binding key")
    entity = _surface(raw["entity"], label="binding entity", allow_empty=False)
    positive_field = (
        "entity_witness_record_ids"
        if catalog_schema == SOURCE_BOUND_SLOT_CATALOG_SCHEMA_V1
        else "positive_relation_source_record_ids")
    source_ids = tuple(_ascii_id(item, label="binding positive relation id")
                       for item in _list(raw[positive_field],
                                         label="binding positive relations"))
    if source_ids != tuple(sorted(set(source_ids))):
        raise SourceBoundSlotCompositionError(
            "binding positive relation 必须去重排序")
    witnesses: list[PublicFrameSourceRecord] = []
    for source_id in source_ids:
        source = source_index.get(source_id)
        if source is None:
            raise SourceBoundSlotCompositionError("binding positive relation source 不存在")
        witnesses.append(source)
    negative_ids = (() if catalog_schema == SOURCE_BOUND_SLOT_CATALOG_SCHEMA_V1
                    else tuple(_ascii_id(
                        item, label="binding negative relation id")
                         for item in _list(
                             raw["negative_relation_source_record_ids"],
                             label="binding negative relations")))
    if negative_ids != tuple(sorted(set(negative_ids))):
        raise SourceBoundSlotCompositionError(
            "binding negative relation 必须去重排序")
    negative_witnesses: list[PublicFrameSourceRecord] = []
    for source_id in negative_ids:
        source = source_index.get(source_id)
        if source is None:
            raise SourceBoundSlotCompositionError(
                "binding negative relation source 不存在")
        negative_witnesses.append(source)
    return EntityPropositionBinding(
        binding_key,
        entity,
        tuple(sorted(witnesses, key=lambda item: item.source.stable_key())),
        _hex_bytes(raw["base_catalog_sha256"], label="binding base catalog SHA",
                   expected_size=32),
        _ascii_id(raw["base_frame_key"], label="binding base frame key"),
        (() if catalog_schema == SOURCE_BOUND_SLOT_CATALOG_SCHEMA_V1
         else _hex_bytes(raw["base_frame_raw_sha256"],
                         label="binding base frame raw SHA", expected_size=32)),
        tuple(sorted(negative_witnesses,
                     key=lambda item: item.source.stable_key())),
        catalog_schema,
    )


def _select_base_frame(
        binding: EntityPropositionBinding,
        static_catalog: PublicFrameCatalog,
        ) -> PublicFrame:
    """只允许一个内容锁 NONE active frame 为组合提供 target 与实际 runtime recipe。"""
    if static_catalog.source_sha256 != binding.base_catalog_sha256:
        raise SourceBoundSlotCompositionError("binding base static catalog SHA 漂移")
    matches = tuple(item for item in static_catalog.frames
                    if item.frame_key == binding.base_frame_key)
    if (len(matches) != 1
            or (binding.base_frame_raw_sha256
                and matches[0].raw_line_sha256
                != binding.base_frame_raw_sha256)):
        raise SourceBoundSlotCompositionError("binding base frame 不唯一或不存在")
    frame = matches[0]
    if (frame.context_requirement != PUBLIC_FRAME_CONTEXT_NONE
            or frame.context_target_key
            or not isinstance(frame.recipe, (
                PublicFrameRuntimeRecipe,
                PublicFrameResponseActRuntimeRecipe))
            or frame.question.target_branch is None
            or frame.question.target not in frame.question.authorized_candidate_targets):
        raise SourceBoundSlotCompositionError(
            "binding base frame 不能形成 NONE source-locked runtime recipe")
    return frame


def _base_entity_for(
        base: PublicFrame,
        family: QuestionConstructionFamily,
        ) -> tuple[int, ...]:
    """由锁定 base surface 与同一 fixed suffix 机械恢复被别名的实体段。"""
    suffix = family.suffix_scalars
    if (not suffix or len(base.surface_scalars) <= len(suffix)
            or base.surface_scalars[-len(suffix):] != suffix):
        raise SourceBoundSlotCompositionError(
            "base frame surface 不能由 family suffix 恢复实体段")
    return base.surface_scalars[:-len(suffix)]


def _applicable_family_binding_pairs(
        catalog: SourceBoundSlotCompositionCatalog,
        static_catalog: PublicFrameCatalog,
        ) -> tuple[tuple[QuestionConstructionFamily, EntityPropositionBinding], ...]:
    """枚举 base surface 实际能承载的 family/binding 对。

    一个 binding 的 base frame 决定它属于哪类问句；不能因为 manifest 同时含有
    时间和地点 family，就把它们的笛卡尔积误当成语义候选。这里的 suffix 边界是
    唯一准入规则，顺序来自已验证的 canonical family/binding record，而不是文件
    行序或 Python map。base frame 自身的 SHA、recipe 和 NONE 限制仍由
    ``_select_base_frame`` 逐对复核。
    """
    if not isinstance(catalog, SourceBoundSlotCompositionCatalog):
        raise TypeError("composition catalog 类型错误")
    if not isinstance(static_catalog, PublicFrameCatalog):
        raise TypeError("composition static catalog 类型错误")
    pairs: list[tuple[QuestionConstructionFamily, EntityPropositionBinding]] = []
    family_counts = {family.family_key: 0 for family in catalog.families}
    binding_counts = {binding.binding_key: 0 for binding in catalog.bindings}
    for family in catalog.families:
        for binding in catalog.bindings:
            base = _select_base_frame(binding, static_catalog)
            suffix = family.suffix_scalars
            if (not suffix
                    or len(base.surface_scalars) <= len(suffix)
                    or base.surface_scalars[-len(suffix):] != suffix):
                continue
            # 保持 boundary derivation 与既有动态 frame 路径完全同源。
            _base_entity_for(base, family)
            pairs.append((family, binding))
            family_counts[family.family_key] += 1
            binding_counts[binding.binding_key] += 1
    if (not pairs
            or any(count == 0 for count in family_counts.values())
            or any(count == 0 for count in binding_counts.values())):
        raise SourceBoundSlotCompositionError(
            "composition family/binding 缺少可验证 base surface 边界")
    return tuple(pairs)


def _find_all_u8_subsequence_v1(
        payload: bytes,
        needle: tuple[int, ...],
        ) -> tuple[int, ...]:
    """按逐位置、允许重叠的 u8 规则返回全部 needle 起点。

    这是 DLG-RAW-06 的规范 relation scan；不得调用语言库的 ``count`` 或
    ``find``，以免不同宿主对重叠、空 needle 或 signed byte 的行为成为语义。
    """
    if type(payload) is not bytes:
        raise TypeError("relation payload 必须是 raw bytes adapter 输出")
    if (not isinstance(needle, tuple) or not needle
            or any(type(item) is not int or item < 0 or item > 255
                   for item in needle)):
        raise SourceBoundSlotCompositionError(
            "relation needle 必须是非空规范 u8 tuple")
    if len(needle) > len(payload):
        return ()
    starts = []
    last_start = len(payload) - len(needle)
    for start in range(last_start + 1):
        matched = True
        for offset, expected in enumerate(needle):
            if payload[start + offset] != expected:
                matched = False
                break
        if matched:
            starts.append(start)
    return tuple(starts)


def _verify_binding_relation(
        binding: EntityPropositionBinding,
        base_entity: tuple[int, ...],
        closure: PublicSourcePayloadClosureV1,
        witnesses: tuple[PublicFrameSourceRecord, ...],
        operator: tuple[int, ...],
        relation_label: str,
        ) -> None:
    """验证每个 witness 精确记录指定的 base/alias relation。

    base/alias/分隔符和 alias span 都必须从 raw bytes 再现。调用者以 ``=`` 验证
    positive relation，以 ``!=`` 验证 explicit counterevidence；任何其他字节、共现或
    多次命中都不构成可运输的 relation observation。
    """
    base_bytes = encode_utf8_v1(base_entity)
    alias_bytes = encode_utf8_v1(binding.entity_scalars)
    if (not isinstance(operator, tuple) or not operator
            or any(type(item) is not int or item < 0 or item > 255
                   for item in operator)):
        raise SourceBoundSlotCompositionError("relation operator 非法")
    forward = (*base_bytes, *operator, *alias_bytes)
    reverse = (*alias_bytes, *operator, *base_bytes)
    for source in witnesses:
        payload = _source_payload(source, closure)
        forward_starts = _find_all_u8_subsequence_v1(payload, forward)
        reverse_starts = _find_all_u8_subsequence_v1(payload, reverse)
        relation_starts = (forward_starts if forward == reverse else tuple(sorted(
            (*forward_starts, *reverse_starts))))
        if len(relation_starts) != 1:
            raise SourceBoundSlotCompositionError(
                f"binding witness 缺失或重复受限 base/alias {relation_label}")
        expected_starts = tuple(sorted(
            (*(start + len(base_bytes) + len(operator)
               for start in forward_starts),
             *reverse_starts)))
        expected_end = source.span[0] + len(alias_bytes)
        if (source.span[0] not in expected_starts
                or source.span[1] != expected_end
                or tuple(payload[source.span[0]:expected_end]) != alias_bytes):
            raise SourceBoundSlotCompositionError(
                f"binding witness alias span 未绑定受限 {relation_label}")


def _verify_binding_alias_relation(
        binding: EntityPropositionBinding,
        base_entity: tuple[int, ...],
        closure: PublicSourcePayloadClosureV1,
        ) -> None:
    """验证正向 relation，保留 DLG-RAW-06 的显式等价规则。"""
    _verify_binding_relation(
        binding,
        base_entity,
        closure,
        binding.witnesses,
        (0x3D,),
        "等价观察",
    )


def _verify_binding_counterevidence(
        binding: EntityPropositionBinding,
        base_entity: tuple[int, ...],
        closure: PublicSourcePayloadClosureV1,
        ) -> None:
    """验证 V2 的 explicit ``!=`` relation，不把反证解释为缺证。"""
    _verify_binding_relation(
        binding,
        base_entity,
        closure,
        binding.negative_witnesses,
        (0x21, 0x3D),
        "反证观察",
    )


def _surface_for(
        family: QuestionConstructionFamily,
        binding: EntityPropositionBinding,
        ) -> tuple[int, ...]:
    """按冻结 scalar 顺序拼接 prefix、entity slot 与 suffix。"""
    return (*family.prefix_scalars, *binding.entity_scalars,
            *family.suffix_scalars)


def _pair_record(
        catalog: SourceBoundSlotCompositionCatalog,
        family: QuestionConstructionFamily,
        binding: EntityPropositionBinding,
        surface: tuple[int, ...],
    ) -> tuple[tuple[int, ...], ...]:
    """构造所有派生 identity 共用的、没有路径/时间/缓存的 SHA 输入。"""
    return (
        catalog.manifest_sha256,
        catalog.base_catalog_sha256,
        family.canonical_record(),
        binding.canonical_record(),
        surface,
        encode_utf8_v1(surface),
    )


def _schema_protocol_values(
        catalog: SourceBoundSlotCompositionCatalog,
        ) -> tuple[bytes, bytes, bytes, tuple[int, ...], tuple[int, ...], tuple[int, ...], str]:
    """按冻结 schema 选择 identity domain 与整数 namespace，避免版本混同。"""
    if catalog.catalog_schema == SOURCE_BOUND_SLOT_CATALOG_SCHEMA_V1:
        return (
            SOURCE_BOUND_SLOT_FRAME_DOMAIN_V1,
            SOURCE_BOUND_SLOT_TRACE_DOMAIN_V1,
            SOURCE_BOUND_SLOT_CATALOG_DOMAIN_V1,
            SOURCE_BOUND_SLOT_REPRESENTATION_FAMILY_V1,
            SOURCE_BOUND_SLOT_CONSTRUCTION_KEY_V1,
            SOURCE_BOUND_SLOT_TRACE_PREFIX_V1,
            "dlg-raw-06-frame-",
        )
    if catalog.catalog_schema == SOURCE_BOUND_SLOT_CATALOG_SCHEMA_V2:
        return (
            SOURCE_BOUND_SLOT_FRAME_DOMAIN_V2,
            SOURCE_BOUND_SLOT_TRACE_DOMAIN_V2,
            SOURCE_BOUND_SLOT_CATALOG_DOMAIN_V2,
            SOURCE_BOUND_SLOT_REPRESENTATION_FAMILY_V2,
            SOURCE_BOUND_SLOT_CONSTRUCTION_KEY_V2,
            SOURCE_BOUND_SLOT_TRACE_PREFIX_V2,
            "dlg-raw-08-frame-",
        )
    if catalog.catalog_schema == SOURCE_BOUND_SLOT_CATALOG_SCHEMA_V3:
        return (
            SOURCE_BOUND_SLOT_FRAME_DOMAIN_V3,
            SOURCE_BOUND_SLOT_TRACE_DOMAIN_V3,
            SOURCE_BOUND_SLOT_CATALOG_DOMAIN_V3,
            SOURCE_BOUND_SLOT_REPRESENTATION_FAMILY_V3,
            SOURCE_BOUND_SLOT_CONSTRUCTION_KEY_V3,
            SOURCE_BOUND_SLOT_TRACE_PREFIX_V3,
            "dlg-raw-09-frame-",
        )
    raise SourceBoundSlotCompositionError("composition catalog schema 未注册")


def _derive_dynamic_frame(
        catalog: SourceBoundSlotCompositionCatalog,
        base_catalog: PublicFrameCatalog,
        family: QuestionConstructionFamily,
        binding: EntityPropositionBinding,
        ) -> PublicFrame:
    """机械重建动态 frame；不查询答案、不使用对象地址或 host hash。"""
    base = _select_base_frame(binding, base_catalog)
    surface = _surface_for(family, binding)
    records = _pair_record(catalog, family, binding, surface)
    (frame_domain, trace_domain, _catalog_domain, representation_family,
     construction_key, trace_prefix, frame_key_prefix) = _schema_protocol_values(
         catalog)
    frame_sha = portable_sha256_v1(frame_domain, records)
    trace_sha = portable_sha256_v1(trace_domain, records)
    branch = base.question.target_branch
    if branch is None:
        raise SourceBoundSlotCompositionError("base frame target branch 缺失")
    segments = (
        (family.prefix_scalars, tuple(source for _, source in family.witnesses)),
        (binding.entity_scalars, binding.witnesses),
        (family.suffix_scalars, tuple(source for _, source in family.witnesses)),
    )
    routes = []
    cursor = 0
    route_position = 0
    for scalars, evidence in segments:
        if not scalars:
            continue
        end = cursor + len(scalars)
        routes.append(PublicFrameLexicalRoute(
            route_position,
            (cursor, end),
            branch,
            representation_identity(
                representation_family,
                (route_position, len(scalars), *scalars),
                owner=branch.owner,
                versions=branch.versions),
            language_atom_identity(
                branch, (*construction_key, *frame_sha, route_position,
                         len(scalars), *scalars)),
            evidence,
            scalars,
        ))
        cursor = end
        route_position += 1
    if cursor != len(surface) or not routes:
        raise SourceBoundSlotCompositionError("动态 frame slot 边界无法形成完整 route")
    construction = structure_concept_identity(
        (*construction_key, *frame_sha),
        owner=branch.owner,
        versions=branch.versions)
    question = PublicFrameQuestionTemplate(
        base.question.query_kind,
        base.question.intent,
        base.question.goal_kind,
        base.question.target,
        base.question.required,
        base.question.evidence_scope,
        base.question.response_scope,
        (*trace_prefix, *trace_sha),
        branch,
        base.question.authorized_candidate_targets,
    )
    additions = tuple(
        source for route in routes for source in route.evidence)
    source_records = tuple(sorted(
        (*base.source_records, *additions),
        key=lambda item: item.source.stable_key(),
    ))
    if (len({item.record_id for item in source_records}) != len(source_records)
            or len({item.source.stable_key() for item in source_records})
            != len(source_records)):
        raise SourceBoundSlotCompositionError(
            "动态 frame source record 与 base source 冲突")
    return PublicFrame(
        frame_key_prefix + bytes(frame_sha).hex(),
        frame_sha,
        encode_utf8_v1(surface),
        surface,
        source_records,
        tuple(routes),
        construction,
        tuple(route.atom for route in routes),
        question,
        base.recipe,
        PUBLIC_FRAME_CONTEXT_NONE,
        (),
    )


def _dynamic_catalog_sha(
        composition_catalog: SourceBoundSlotCompositionCatalog,
        frame: PublicFrame,
        ) -> tuple[int, ...]:
    """由 source-bound manifest 和动态 frame record 派生单 frame catalog identity。"""
    _frame_domain, _trace_domain, catalog_domain, *_ = _schema_protocol_values(
        composition_catalog)
    return portable_sha256_v1(
        catalog_domain,
        (
            composition_catalog.manifest_sha256,
            composition_catalog.base_catalog_sha256,
            frame.canonical_record(),
        ),
    )


def _binding_static_catalog(
        catalog: SourceBoundSlotCompositionCatalog,
        base_catalog: PublicFrameCatalog,
        active_static_catalog: PublicFrameCatalog,
        ) -> PublicFrameCatalog:
    """按 catalog schema 选择 binding 经过 SHA 锁定的 static frame 集合。"""
    if catalog.catalog_schema == SOURCE_BOUND_SLOT_CATALOG_SCHEMA_V1:
        return base_catalog
    if catalog.catalog_schema == SOURCE_BOUND_SLOT_CATALOG_SCHEMA_V2:
        return active_static_catalog
    if catalog.catalog_schema == SOURCE_BOUND_SLOT_CATALOG_SCHEMA_V3:
        return active_static_catalog
    raise SourceBoundSlotCompositionError("composition catalog schema 未注册")


def _v3_pair_record(
        family: QuestionConstructionFamily,
        binding: EntityPropositionBinding,
        ) -> tuple[int, ...]:
    """导出一个 family/binding pair 的完整来源见证，不依赖文件顺序。"""
    result = [SOURCE_BOUND_SLOT_PAIR_RECORD_V3]
    _pack(result, family.canonical_record())
    _pack(result, binding.canonical_record())
    return tuple(result)


def _v3_target_candidate(
        family: QuestionConstructionFamily,
        binding: EntityPropositionBinding,
        base: PublicFrame,
        verdict: int,
        ) -> SourceBoundSlotTargetCandidateV3:
    """从锁定 base frame 构造 V3 target 候选，不读取答案或动态 frame。"""
    question = base.question
    branch = question.target_branch
    if branch is None:
        raise SourceBoundSlotCompositionError("V3 candidate 缺 target branch")
    source = semantic_source(question.target.template)
    if (question.evidence_scope.source != source
            or question.response_scope.source != source
            or any(semantic_source(item.template) != source
                   for item in question.authorized_candidate_targets)):
        raise SourceBoundSlotCompositionError(
            "V3 candidate target/source/scope 未闭合")
    authorized_target_keys = tuple(sorted(
        (item.stable_key() for item in question.authorized_candidate_targets)))
    return SourceBoundSlotTargetCandidateV3(
        question.target.stable_key(),
        question.query_kind.stable_key(),
        question.intent.stable_key(),
        question.goal_kind.stable_key(),
        question.required.stable_key(),
        question.evidence_scope.stable_key(),
        question.response_scope.stable_key(),
        branch.stable_key(),
        authorized_target_keys,
        base.recipe.canonical_record(),
        base.context_requirement,
        base.context_target_key,
        base.frame_key,
        base.raw_line_sha256,
        (_v3_pair_record(family, binding),),
        verdict,
    )


def _validate_global_pairs(
        catalog: SourceBoundSlotCompositionCatalog,
        base_catalog: PublicFrameCatalog,
        active_static_catalog: PublicFrameCatalog,
        closure: PublicSourcePayloadClosureV1,
        ) -> None:
    """加载时枚举有限 family x binding，拒绝 static/cross-pair surface collision。"""
    seen: dict[tuple[int, ...], tuple[str, str]] = {}
    v3_candidates: dict[
        tuple[int, ...], tuple[SourceBoundSlotTargetCandidateV3, ...]] = {}
    _validate_active_contains_base(base_catalog, active_static_catalog)
    binding_catalog = _binding_static_catalog(
        catalog, base_catalog, active_static_catalog)
    if binding_catalog.source_sha256 != catalog.base_catalog_sha256:
        if catalog.catalog_schema == SOURCE_BOUND_SLOT_CATALOG_SCHEMA_V1:
            raise SourceBoundSlotCompositionError("base catalog SHA 漂移")
        raise SourceBoundSlotCompositionError(
            "composition binding static catalog SHA 漂移")
    static_surfaces = {frame.surface_scalars
                       for frame in active_static_catalog.frames}
    for family, binding in _applicable_family_binding_pairs(
            catalog, binding_catalog):
        base = _select_base_frame(binding, binding_catalog)
        _verify_binding_alias_relation(
            binding, _base_entity_for(base, family), closure)
        if binding.negative_witnesses:
            _verify_binding_counterevidence(
                binding, _base_entity_for(base, family), closure)
        surface = _surface_for(family, binding)
        if surface in static_surfaces:
            raise SourceBoundSlotCompositionError(
                "composition 生成 surface 与 static frame 冲突")
        pair = family.family_key, binding.binding_key
        previous = seen.get(surface)
        if (previous is not None
                and catalog.catalog_schema
                == SOURCE_BOUND_SLOT_CATALOG_SCHEMA_V1):
            raise SourceBoundSlotCompositionError(
                "composition family x binding 生成重复 surface")
        if previous is None:
            seen[surface] = pair
        if catalog.catalog_schema != SOURCE_BOUND_SLOT_CATALOG_SCHEMA_V3:
            continue
        candidate = _v3_target_candidate(
            family,
            binding,
            base,
            SOURCE_BOUND_SLOT_CANDIDATE_SUPPORTED_V3,
        )
        prior_candidates = v3_candidates.get(surface, ())
        same_target = tuple(
            item for item in prior_candidates
            if item.target_key == candidate.target_key)
        if same_target:
            if same_target[0].equivalence_record() == candidate.equivalence_record():
                raise SourceBoundSlotCompositionError(
                    "V3 composition 生成重复 target candidate")
            raise SourceBoundSlotCompositionError(
                "V3 composition 同 target candidate 语义漂移")
        for prior in prior_candidates:
            if (prior.evidence_scope_key == candidate.evidence_scope_key
                    or prior.response_scope_key == candidate.response_scope_key
                    or prior.recipe_record == candidate.recipe_record):
                raise SourceBoundSlotCompositionError(
                    "V3 多 target candidate 缺独立 source/scope/recipe")
        v3_candidates[surface] = tuple(sorted(
            (*prior_candidates, candidate),
            key=SourceBoundSlotTargetCandidateV3.equivalence_record,
        ))


def _validate_active_contains_base(
        base_catalog: PublicFrameCatalog,
        active_static_catalog: PublicFrameCatalog,
        ) -> None:
    """确认 active exact catalog 保留了已锁 base frame 的完整 canonical record。"""
    if not isinstance(base_catalog, PublicFrameCatalog):
        raise TypeError("base catalog 类型错误")
    if not isinstance(active_static_catalog, PublicFrameCatalog):
        raise TypeError("active static catalog 类型错误")
    for base_frame in base_catalog.frames:
        matches = tuple(item for item in active_static_catalog.frames
                        if item.frame_key == base_frame.frame_key)
        if len(matches) != 1 or matches[0].canonical_record() != (
                base_frame.canonical_record()):
            raise SourceBoundSlotCompositionError(
                "active static catalog 未完整包含锁定 base frame")


def load_source_bound_slot_composition_catalog_from_closure(
        closure: PublicSourcePayloadClosureV1,
        base_catalog: PublicFrameCatalog,
        active_static_catalog: PublicFrameCatalog,
        *,
        catalog_logical_key: bytes = SOURCE_BOUND_SLOT_CATALOG_LOGICAL_KEY_V1,
        ) -> SourceBoundSlotCompositionCatalog:
    """由完整 public payload closure 加载 V1、V2 或 V3 composition catalog。

    V1 binding 锁定 ``base_catalog``；V2 binding 锁定实际 merged
    ``active_static_catalog`` 中的 raw-line identity。manifest、构式与 alias source
    均通过 closure logical key 取得，不接收物理根或路径。
    """
    if type(closure) is not PublicSourcePayloadClosureV1:
        raise TypeError("composition source payload closure 必须是完整 closure")
    if not isinstance(base_catalog, PublicFrameCatalog):
        raise TypeError("composition base catalog 必须是 PublicFrameCatalog")
    if not isinstance(active_static_catalog, PublicFrameCatalog):
        raise TypeError("composition active static catalog 必须是 PublicFrameCatalog")
    if catalog_logical_key not in {
            SOURCE_BOUND_SLOT_CATALOG_LOGICAL_KEY_V1,
            SOURCE_BOUND_SLOT_CATALOG_LOGICAL_KEY_V2,
            SOURCE_BOUND_SLOT_CATALOG_LOGICAL_KEY_V3}:
        raise SourceBoundSlotCompositionError(
            "composition catalog logical key 未注册")
    try:
        payload = closure.payload_for(catalog_logical_key)
        manifest_record = closure.record_for(catalog_logical_key)
    except PublicSourcePayloadProviderError as error:
        raise SourceBoundSlotCompositionError(
            "composition manifest 不在 public payload closure") from error
    if (payload != manifest_record.raw_payload
            or manifest_record.payload_length != len(payload)
            or tuple(manifest_record.raw_sha256) != _sha256(payload)):
        raise SourceBoundSlotCompositionError("composition manifest closure record 漂移")
    if not payload.endswith(b"\n") or payload.count(b"\n") != 1:
        raise SourceBoundSlotCompositionError("composition manifest 必须是一行 canonical JSONL")
    try:
        raw = parse_canonical_json_bytes(payload[:-1], require_object=True)
    except DatasetContractError as error:
        raise SourceBoundSlotCompositionError("composition manifest 不是 canonical JSON") from error
    manifest = _exact(raw, _MANIFEST_FIELDS, label="composition manifest")
    catalog_schema = _strict_int(
        manifest["catalog_schema"], label="composition catalog schema")
    expected_schema = (
        SOURCE_BOUND_SLOT_CATALOG_SCHEMA_V1
        if catalog_logical_key == SOURCE_BOUND_SLOT_CATALOG_LOGICAL_KEY_V1
        else (SOURCE_BOUND_SLOT_CATALOG_SCHEMA_V2
              if catalog_logical_key == SOURCE_BOUND_SLOT_CATALOG_LOGICAL_KEY_V2
              else SOURCE_BOUND_SLOT_CATALOG_SCHEMA_V3))
    if catalog_schema != expected_schema:
        raise SourceBoundSlotCompositionError("composition catalog schema 未注册")
    sources = _parse_source_records(
        manifest["source_records"], closure=closure)
    source_index = {source.record_id: source for source in sources}
    families = tuple(sorted(
        (_parse_family(
            item,
            source_index=source_index,
            closure=closure,
            catalog_schema=catalog_schema)
         for item in _list(manifest["families"], label="families")),
        key=QuestionConstructionFamily.canonical_record,
    ))
    bindings = tuple(sorted(
        (_parse_binding(
            item,
            source_index=source_index,
            catalog_schema=catalog_schema)
         for item in _list(manifest["bindings"], label="bindings")),
        key=EntityPropositionBinding.canonical_record,
    ))
    if not bindings:
        raise SourceBoundSlotCompositionError("composition bindings 不得为空")
    static_sha = bindings[0].base_catalog_sha256
    catalog = SourceBoundSlotCompositionCatalog(
        _sha256(payload),
        static_sha,
        tuple(closure.closure_identity),
        sources,
        families,
        bindings,
        catalog_schema,
        catalog_logical_key.decode("ascii"),
    )
    binding_catalog = _binding_static_catalog(
        catalog, base_catalog, active_static_catalog)
    if binding_catalog.source_sha256 != catalog.base_catalog_sha256:
        if catalog.catalog_schema == SOURCE_BOUND_SLOT_CATALOG_SCHEMA_V1:
            raise SourceBoundSlotCompositionError("base catalog SHA 漂移")
        raise SourceBoundSlotCompositionError(
            "composition binding catalog SHA 漂移")
    _validate_global_pairs(catalog, base_catalog, active_static_catalog, closure)
    return catalog


def load_source_bound_slot_composition_catalog(
        closure: PublicSourcePayloadClosureV1,
        base_catalog: PublicFrameCatalog,
        active_static_catalog: PublicFrameCatalog,
        *,
        catalog_logical_key: bytes = SOURCE_BOUND_SLOT_CATALOG_LOGICAL_KEY_V1,
        ) -> SourceBoundSlotCompositionCatalog:
    """旧同名入口的 closure-only 兼容别名。

    路径式加载已经废止；host 必须先构造完整 public payload closure，再调用本
    函数或显式 ``load_source_bound_slot_composition_catalog_from_closure``。
    """
    if type(closure) is not PublicSourcePayloadClosureV1:
        raise SourceBoundSlotCompositionError(
            "路径式 source-bound slot catalog 加载已废止；必须提供 payload closure")
    return load_source_bound_slot_composition_catalog_from_closure(
        closure,
        base_catalog,
        active_static_catalog,
        catalog_logical_key=catalog_logical_key,
    )


def _construction_miss_resolution(
        catalog: SourceBoundSlotCompositionCatalog,
        base_catalog: PublicFrameCatalog,
        active_static_catalog: PublicFrameCatalog,
        family: QuestionConstructionFamily,
        binding: EntityPropositionBinding,
        scalars: tuple[int, ...],
        ) -> SourceBoundSlotCompositionResolution:
    """为既有 RAW-01 的 code 9 形态尽力提供已构成 frame，而绝不执行 runtime。

    source readback 失败时，已加载的内容锁 record 仍可机械重建 route/target/recipe，供
    ingress 构造 ``REJECT_CONSTRUCTION_MISS`` 的零 request result。该 frame 不是接受
    结果，调用者不得进入 RAW-02/RAW-04；若 base 也无法构成则保留无 frame 的 code 9。
    """
    try:
        frame = _derive_dynamic_frame(
            catalog,
            _binding_static_catalog(
                catalog, base_catalog, active_static_catalog),
            family,
            binding)
        dynamic_catalog = PublicFrameCatalog(
            _dynamic_catalog_sha(catalog, frame), (frame,))
    except (PublicFrameCatalogError, SourceBoundSlotCompositionError, TypeError,
            ValueError):
        return SourceBoundSlotCompositionResolution(
            DLG_RAW_REJECT_CONSTRUCTION_MISS, 1, scalars, catalog)
    return SourceBoundSlotCompositionResolution(
        DLG_RAW_REJECT_CONSTRUCTION_MISS, 1, scalars, catalog, frame,
        dynamic_catalog)


def _v3_construction_miss_resolution(
        catalog: SourceBoundSlotCompositionCatalog,
        matched_pair_count: int,
        scalars: tuple[int, ...],
        ) -> SourceBoundSlotCompositionResolution:
    """V3 的任一闭合失败都停在 pre-frame code 9，绝不泄露可物化 frame。"""
    return SourceBoundSlotCompositionResolution(
        DLG_RAW_REJECT_CONSTRUCTION_MISS,
        max(1, matched_pair_count),
        scalars,
        catalog,
    )


def _resolve_v3_matches(
        catalog: SourceBoundSlotCompositionCatalog,
        base_catalog: PublicFrameCatalog,
        active_static_catalog: PublicFrameCatalog,
        binding_catalog: PublicFrameCatalog,
        matches: list[tuple[QuestionConstructionFamily, EntityPropositionBinding]],
        scalars: tuple[int, ...],
        closure: PublicSourcePayloadClosureV1,
        ) -> SourceBoundSlotCompositionResolution:
    """按 target candidate 而非 family/binding pair 完成 V3 的零写裁决。"""
    try:
        catalog.verify_sources(closure)
    except SourceBoundSlotCompositionError:
        return _v3_construction_miss_resolution(
            catalog, len(matches), scalars)
    resolved: list[tuple[
        SourceBoundSlotTargetCandidateV3,
        QuestionConstructionFamily,
        EntityPropositionBinding,
    ]] = []
    try:
        for family, binding in matches:
            base = _select_base_frame(binding, binding_catalog)
            base_entity = _base_entity_for(base, family)
            _verify_binding_alias_relation(binding, base_entity, closure)
            verdict = SOURCE_BOUND_SLOT_CANDIDATE_SUPPORTED_V3
            if binding.negative_witnesses:
                _verify_binding_counterevidence(binding, base_entity, closure)
                verdict = SOURCE_BOUND_SLOT_CANDIDATE_CONFLICTED_V3
            resolved.append((
                _v3_target_candidate(family, binding, base, verdict),
                family,
                binding,
            ))
    except (SourceBoundSlotCompositionError, TypeError, ValueError):
        return _v3_construction_miss_resolution(
            catalog, len(matches), scalars)
    resolved.sort(key=lambda item: item[0].equivalence_record())
    candidates = tuple(item[0] for item in resolved)
    if (len({item.target_key for item in candidates}) != len(candidates)
            or len({item.equivalence_record() for item in candidates})
            != len(candidates)):
        return _v3_construction_miss_resolution(
            catalog, len(matches), scalars)
    for left_index, left in enumerate(candidates):
        for right in candidates[left_index + 1:]:
            if (left.evidence_scope_key == right.evidence_scope_key
                    or left.response_scope_key == right.response_scope_key
                    or left.recipe_record == right.recipe_record):
                return _v3_construction_miss_resolution(
                    catalog, len(matches), scalars)
    candidate_count = len(candidates)
    if any(item.verdict == SOURCE_BOUND_SLOT_CANDIDATE_CONFLICTED_V3
           for item in candidates):
        return SourceBoundSlotCompositionResolution(
            DLG_RAW_REJECT_SOURCE_CONFLICT,
            candidate_count,
            scalars,
            catalog,
            target_candidates=candidates,
        )
    if candidate_count >= 2:
        return SourceBoundSlotCompositionResolution(
            DLG_RAW_REJECT_LEXICAL_AMBIGUOUS,
            candidate_count,
            scalars,
            catalog,
            target_candidates=candidates,
        )
    candidate, family, binding = resolved[0]
    try:
        frame = _derive_dynamic_frame(catalog, binding_catalog, family, binding)
        dynamic_catalog = PublicFrameCatalog(
            _dynamic_catalog_sha(catalog, frame), (frame,))
    except (PublicFrameCatalogError, SourceBoundSlotCompositionError, TypeError,
            ValueError):
        return _v3_construction_miss_resolution(
            catalog, candidate_count, scalars)
    return SourceBoundSlotCompositionResolution(
        DLG_RAW_ACCEPT,
        1,
        scalars,
        catalog,
        frame,
        dynamic_catalog,
        (candidate,),
    )


def resolve_source_bound_slot_composition(
        catalog: SourceBoundSlotCompositionCatalog,
        base_catalog: PublicFrameCatalog,
        active_static_catalog: PublicFrameCatalog,
        raw_scalars: tuple[int, ...],
        closure: PublicSourcePayloadClosureV1,
        ) -> SourceBoundSlotCompositionResolution:
    """在 static exact miss 后，按 canonical family x binding 顺序枚举组合。

    调用者应先调用 static catalog 的 exact matcher。本函数仍拒绝 static hit，避免
    不同入口顺序把静态问句误记为组合能力。唯一组合返回的
    ``public_frame_catalog`` 可以直接交给既有 RAW-01 ingress。
    """
    if not isinstance(catalog, SourceBoundSlotCompositionCatalog):
        raise TypeError("composition catalog 类型错误")
    if not isinstance(base_catalog, PublicFrameCatalog):
        raise TypeError("base catalog 类型错误")
    if not isinstance(active_static_catalog, PublicFrameCatalog):
        raise TypeError("active static catalog 类型错误")
    if type(closure) is not PublicSourcePayloadClosureV1:
        raise TypeError("composition source payload closure 类型错误")
    scalars = _scalar_vector(list(raw_scalars), label="composition raw scalars",
                              allow_empty=True)
    binding_catalog = _binding_static_catalog(
        catalog, base_catalog, active_static_catalog)
    if binding_catalog.source_sha256 != catalog.base_catalog_sha256:
        return SourceBoundSlotCompositionResolution(
            DLG_RAW_REJECT_CONSTRUCTION_MISS, 1, scalars, catalog)
    try:
        _validate_active_contains_base(base_catalog, active_static_catalog)
    except (SourceBoundSlotCompositionError, TypeError):
        return SourceBoundSlotCompositionResolution(
            DLG_RAW_REJECT_CONSTRUCTION_MISS, 1, scalars, catalog)
    if active_static_catalog.matching_frames(scalars):
        return SourceBoundSlotCompositionResolution(
            DLG_RAW_REJECT_LEXICAL_MISS, 0, scalars, catalog)
    matches = [
        (family, binding)
        for family, binding in _applicable_family_binding_pairs(
            catalog, binding_catalog)
        if _surface_for(family, binding) == scalars
    ]
    if not matches:
        return SourceBoundSlotCompositionResolution(
            DLG_RAW_REJECT_LEXICAL_MISS, 0, scalars, catalog)
    if catalog.catalog_schema == SOURCE_BOUND_SLOT_CATALOG_SCHEMA_V3:
        return _resolve_v3_matches(
            catalog,
            base_catalog,
            active_static_catalog,
            binding_catalog,
            matches,
            scalars,
            closure,
        )
    try:
        catalog.verify_sources(closure)
    except SourceBoundSlotCompositionError:
        if len(matches) == 1:
            family, binding = matches[0]
            return _construction_miss_resolution(
                catalog, base_catalog, active_static_catalog,
                family, binding, scalars)
        return SourceBoundSlotCompositionResolution(
            DLG_RAW_REJECT_CONSTRUCTION_MISS, len(matches), scalars, catalog)
    conflict_count = 0
    for family, binding in matches:
        try:
            base = _select_base_frame(binding, binding_catalog)
            base_entity = _base_entity_for(base, family)
            _verify_binding_alias_relation(binding, base_entity, closure)
            if binding.negative_witnesses:
                _verify_binding_counterevidence(
                    binding, base_entity, closure)
                conflict_count += 1
        except (SourceBoundSlotCompositionError, TypeError, ValueError):
            if len(matches) == 1:
                return _construction_miss_resolution(
                    catalog, base_catalog, active_static_catalog,
                    family, binding, scalars)
            return SourceBoundSlotCompositionResolution(
                DLG_RAW_REJECT_CONSTRUCTION_MISS,
                len(matches), scalars, catalog)
    if conflict_count:
        return SourceBoundSlotCompositionResolution(
            DLG_RAW_REJECT_SOURCE_CONFLICT,
            len(matches), scalars, catalog)
    if len(matches) != 1:
        return SourceBoundSlotCompositionResolution(
            DLG_RAW_REJECT_LEXICAL_AMBIGUOUS, len(matches), scalars, catalog)
    family, binding = matches[0]
    try:
        frame = _derive_dynamic_frame(
            catalog, binding_catalog, family, binding)
        dynamic_catalog = PublicFrameCatalog(
            _dynamic_catalog_sha(catalog, frame), (frame,))
    except (PublicFrameCatalogError, SourceBoundSlotCompositionError, TypeError,
            ValueError):
        return _construction_miss_resolution(
            catalog, base_catalog, active_static_catalog,
            family, binding, scalars)
    return SourceBoundSlotCompositionResolution(
        DLG_RAW_ACCEPT, 1, scalars, catalog, frame, dynamic_catalog)


__all__ = [
    "SOURCE_BOUND_SLOT_CATALOG_SCHEMA_V1",
    "SOURCE_BOUND_SLOT_CATALOG_SCHEMA_V2",
    "SOURCE_BOUND_SLOT_CATALOG_SCHEMA_V3",
    "SOURCE_BOUND_SLOT_CATALOG_RECORD_V1",
    "SOURCE_BOUND_SLOT_CATALOG_RECORD_V2",
    "SOURCE_BOUND_SLOT_CATALOG_RECORD_V3",
    "SOURCE_BOUND_SLOT_FAMILY_RECORD_V1",
    "SOURCE_BOUND_SLOT_FAMILY_RECORD_V2",
    "SOURCE_BOUND_SLOT_FAMILY_RECORD_V3",
    "SOURCE_BOUND_SLOT_BINDING_RECORD_V1",
    "SOURCE_BOUND_SLOT_BINDING_RECORD_V2",
    "SOURCE_BOUND_SLOT_BINDING_RECORD_V3",
    "SOURCE_BOUND_SLOT_CATALOG_LOGICAL_KEY_V1",
    "SOURCE_BOUND_SLOT_CATALOG_LOGICAL_KEY_V2",
    "SOURCE_BOUND_SLOT_CATALOG_LOGICAL_KEY_V3",
    "SOURCE_BOUND_SLOT_RESOLUTION_RECORD_V1",
    "SOURCE_BOUND_SLOT_RESOLUTION_RECORD_V2",
    "SOURCE_BOUND_SLOT_RESOLUTION_RECORD_V3",
    "SOURCE_BOUND_SLOT_TARGET_CANDIDATE_RECORD_V3",
    "SOURCE_BOUND_SLOT_TARGET_CANDIDATE_KEY_RECORD_V3",
    "SOURCE_BOUND_SLOT_PAIR_RECORD_V3",
    "SOURCE_BOUND_SLOT_CANDIDATE_SUPPORTED_V3",
    "SOURCE_BOUND_SLOT_CANDIDATE_CONFLICTED_V3",
    "SOURCE_BOUND_SLOT_TYPE_ENTITY_ALIAS_V1",
    "SOURCE_BOUND_SLOT_TYPE_ENTITY_ALIAS_V2",
    "SOURCE_BOUND_SLOT_TYPE_ENTITY_ALIAS_V3",
    "EntityPropositionBinding",
    "QuestionConstructionFamily",
    "SourceBoundSlotCompositionCatalog",
    "SourceBoundSlotCompositionError",
    "SourceBoundSlotCompositionResolution",
    "SourceBoundSlotTargetCandidateV3",
    "load_source_bound_slot_composition_catalog",
    "load_source_bound_slot_composition_catalog_from_closure",
    "portable_integer_record_bytes",
    "portable_sha256_v1",
    "resolve_source_bound_slot_composition",
]
