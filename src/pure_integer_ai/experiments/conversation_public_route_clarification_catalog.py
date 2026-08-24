"""DLG-RAW-14：公开来源绑定路由澄清课程的纯整数 catalog。

本模块只把固定 public payload closure 内的一条 canonical JSONL 课程及两份
独立 CC0 surface witness 编译为有限整数/bytes record。它不读取物理路径、
不重演 resolver、不运行回答链，也不保存 pending 会话状态。候选 identity 只由
V3 candidate 的 canonical record 导出，因而不会把包含本课程的 closure identity
反向纳入 candidate identity 形成循环。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pure_integer_ai.experiments.conversation_public_source_payload_provider import (
    PublicSourcePayloadClosureV1,
    PublicSourcePayloadProviderError,
    portable_sha256_v1,
    public_source_payload_sha256_v1,
)
from pure_integer_ai.experiments.conversation_raw_intake import (
    DLG_RAW_REJECT_LEXICAL_AMBIGUOUS,
    decode_utf8_v1,
    encode_utf8_v1,
)
from pure_integer_ai.experiments.conversation_source_bound_slot_catalog import (
    SOURCE_BOUND_SLOT_CANDIDATE_SUPPORTED_V3,
    SOURCE_BOUND_SLOT_CATALOG_SCHEMA_V3,
    SourceBoundSlotCompositionResolution,
    SourceBoundSlotTargetCandidateV3,
)
from pure_integer_ai.experiments.ph2_dataset_core import (
    DatasetContractError,
    parse_canonical_json_bytes,
)


PUBLIC_ROUTE_CLARIFICATION_CATALOG_SCHEMA_V1 = 1
PUBLIC_ROUTE_CLARIFICATION_SOURCE_RECORD_V1 = 1
PUBLIC_ROUTE_CLARIFICATION_OPTION_RECORD_V1 = 1
PUBLIC_ROUTE_CLARIFICATION_FORM_RECORD_V1 = 1
PUBLIC_ROUTE_CLARIFICATION_CATALOG_RECORD_V1 = 1
PUBLIC_ROUTE_CLARIFICATION_COURSE_PARSER_RECORD_V1 = 1
PUBLIC_ROUTE_CLARIFICATION_ROUTE_IDENTITY_RECORD_V1 = 1
PUBLIC_ROUTE_CLARIFICATION_OUTPUT_READBACK_RECORD_V1 = 1
PUBLIC_ROUTE_CLARIFICATION_SOURCE_KIND_COURSE_V1 = 1
PUBLIC_ROUTE_CLARIFICATION_SOURCE_KIND_SURFACE_A_V1 = 2
PUBLIC_ROUTE_CLARIFICATION_SOURCE_KIND_SURFACE_B_V1 = 3
PUBLIC_ROUTE_CLARIFICATION_OUTPUT_READBACK_ACCEPTED_V1 = 0

PUBLIC_ROUTE_CLARIFICATION_CANDIDATE_IDENTITY_DOMAIN_V1 = (
    b"PURE-INTEGER-AI/DLG-RAW-14/ROUTE-CANDIDATE/V1")
PUBLIC_ROUTE_CLARIFICATION_ROUTE_IDENTITY_DOMAIN_V1 = (
    b"PURE-INTEGER-AI/DLG-RAW-14/ROUTE-IDENTITY/V1")
PUBLIC_ROUTE_CLARIFICATION_SOURCE_IDENTITY_DOMAIN_V1 = (
    b"PURE-INTEGER-AI/DLG-RAW-14/ROUTE-SOURCE/V1")
PUBLIC_ROUTE_CLARIFICATION_OPTION_IDENTITY_DOMAIN_V1 = (
    b"PURE-INTEGER-AI/DLG-RAW-14/ROUTE-OPTION/V1")
PUBLIC_ROUTE_CLARIFICATION_FORM_IDENTITY_DOMAIN_V1 = (
    b"PURE-INTEGER-AI/DLG-RAW-14/ROUTE-FORM/V1")
PUBLIC_ROUTE_CLARIFICATION_CATALOG_IDENTITY_DOMAIN_V1 = (
    b"PURE-INTEGER-AI/DLG-RAW-14/ROUTE-CATALOG/V1")
PUBLIC_ROUTE_CLARIFICATION_COURSE_PARSER_IDENTITY_DOMAIN_V1 = (
    b"PURE-INTEGER-AI/DLG-RAW-14/ROUTE-COURSE-PARSER/V1")

PUBLIC_ROUTE_CLARIFICATION_COURSE_LOGICAL_KEY_V1 = (
    b"data/ph2/dlg_raw_public_route_clarification_course_v1.jsonl.sample")
PUBLIC_ROUTE_CLARIFICATION_SURFACE_A_LOGICAL_KEY_V1 = (
    b"data/ph2/dlg_raw_public_route_clarification_surface_v1_a.txt.sample")
PUBLIC_ROUTE_CLARIFICATION_SURFACE_B_LOGICAL_KEY_V1 = (
    b"data/ph2/dlg_raw_public_route_clarification_surface_v1_b.txt.sample")
PUBLIC_ROUTE_CLARIFICATION_LOGICAL_KEYS_V1 = (
    PUBLIC_ROUTE_CLARIFICATION_COURSE_LOGICAL_KEY_V1,
    PUBLIC_ROUTE_CLARIFICATION_SURFACE_A_LOGICAL_KEY_V1,
    PUBLIC_ROUTE_CLARIFICATION_SURFACE_B_LOGICAL_KEY_V1,
)

_U64_EXCLUSIVE = 1 << 64
_LICENSE_ID_U8 = tuple(b"CC0-1.0")
_COURSE_FIELDS = (
    "candidate_identities",
    "course_attribution",
    "form_id",
    "input_surface",
    "license_id",
    "matched_frame_count",
    "options",
    "output_max_bytes",
    "output_surface",
    "result_code",
    "route_identity",
    "schema",
    "surface_a",
    "surface_b",
)
_OPTION_FIELDS = ("candidate_identity", "option_surface")
_SURFACE_FIELDS = ("attribution", "raw_sha256", "relative_path")
_EXPECTED_SOURCE_KEY_BY_KIND = (
    (PUBLIC_ROUTE_CLARIFICATION_SOURCE_KIND_COURSE_V1,
     tuple(PUBLIC_ROUTE_CLARIFICATION_COURSE_LOGICAL_KEY_V1)),
    (PUBLIC_ROUTE_CLARIFICATION_SOURCE_KIND_SURFACE_A_V1,
     tuple(PUBLIC_ROUTE_CLARIFICATION_SURFACE_A_LOGICAL_KEY_V1)),
    (PUBLIC_ROUTE_CLARIFICATION_SOURCE_KIND_SURFACE_B_V1,
     tuple(PUBLIC_ROUTE_CLARIFICATION_SURFACE_B_LOGICAL_KEY_V1)),
)


# object-model: exception; interop=DLG-RAW-14
class PublicRouteClarificationCatalogError(ValueError):
    """公开 route clarification 课程、来源或整数 record 未能闭合。"""


def _u64(value: int, *, label: str, minimum: int = 0) -> int:
    """验证显式 u64，避免 Python 任意精度范围成为协议行为。"""
    if (type(value) is not int or value < minimum
            or value >= _U64_EXCLUSIVE):
        raise PublicRouteClarificationCatalogError(
            f"{label} 必须是范围内的严格 u64")
    return value


def _u8_tuple(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """验证不可变 raw-u8 vector，拒绝宿主缓冲对象与 bool。"""
    if (type(value) is not tuple
            or any(type(item) is not int or item < 0 or item > 255
                   for item in value)):
        raise PublicRouteClarificationCatalogError(
            f"{label} 必须是 0..255 严格整数 tuple")
    return value


def _record(value: tuple[int, ...], *, label: str, allow_empty: bool) -> tuple[int, ...]:
    """验证有限非负整数 record，不允许容器类型参与协议语义。"""
    if (type(value) is not tuple or (not allow_empty and not value)
            or any(type(item) is not int or item < 0 for item in value)):
        raise PublicRouteClarificationCatalogError(
            f"{label} 不是{'可空' if allow_empty else '非空'}非负整数 record")
    return value


def _pack(result: list[int], value: tuple[int, ...], *, label: str) -> None:
    """以明确 u64 元素计数追加一个有序子 record。"""
    record = _record(value, label=label, allow_empty=True)
    _u64(len(record), label=f"{label} count")
    result.extend((len(record), *record))


def _identity(
        domain: bytes,
        record: tuple[int, ...],
        *,
        label: str,
        ) -> tuple[int, ...]:
    """按冻结 portable SHA framing 形成一个 raw-u8[32] identity。"""
    try:
        identity = tuple(portable_sha256_v1(domain, (record,)))
    except (PublicSourcePayloadProviderError, TypeError, ValueError) as error:
        raise PublicRouteClarificationCatalogError(
            f"{label} identity 无法形成") from error
    identity = _u8_tuple(identity, label=f"{label} identity")
    if len(identity) != 32:
        raise PublicRouteClarificationCatalogError(
            f"{label} identity 长度漂移")
    return identity


def _ascii_bytes(value: Any, *, label: str) -> tuple[int, ...]:
    """把 protocol id/relative path 降解为无空白的明确 ASCII u8[]。"""
    if (type(value) is not str or not value
            or value[0] in " \t\r\n" or value[-1] in " \t\r\n"
            or any(ord(item) < 0x21 or ord(item) > 0x7E for item in value)):
        raise PublicRouteClarificationCatalogError(
            f"{label} 不是规范 ASCII transport")
    return tuple(ord(item) for item in value)


def _ascii_u8_tuple(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """验证已经降解的 ASCII u8[]，不回转宿主 str 作语义判断。"""
    result = _u8_tuple(value, label=label)
    if not result or any(item < 0x21 or item > 0x7E for item in result):
        raise PublicRouteClarificationCatalogError(
            f"{label} 不是规范 ASCII transport")
    return result


def _unicode_scalars(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """验证非空 Unicode scalar 序，拒绝 surrogate 与隐式规范化。"""
    if (type(value) is not tuple or not value
            or any(type(item) is not int or item < 0 or item > 0x10FFFF
                   or 0xD800 <= item <= 0xDFFF for item in value)):
        raise PublicRouteClarificationCatalogError(
            f"{label} 必须是非空严格 Unicode scalar tuple")
    return value


def _display_u8(
        value: Any,
        *,
        label: str,
        allow_line_feed: bool,
        ) -> tuple[int, ...]:
    """以显式 UTF-8 将 course surface 转为 raw-u8，不接受首尾空白。"""
    if (type(value) is not str or not value
            or value[0] in " \t\r\n" or value[-1] in " \t\r\n"):
        raise PublicRouteClarificationCatalogError(
            f"{label} 必须是无首尾空白的非空文本")
    scalars = tuple(ord(item) for item in value)
    if not allow_line_feed and any(item in (0x0A, 0x0D) for item in scalars):
        raise PublicRouteClarificationCatalogError(
            f"{label} 不得含换行")
    try:
        result = encode_utf8_v1(scalars)
    except (TypeError, ValueError) as error:
        raise PublicRouteClarificationCatalogError(
            f"{label} 含非法 Unicode scalar") from error
    return _u8_tuple(result, label=label)


def _surface_scalars(value: Any, *, label: str) -> tuple[int, ...]:
    """读取单行输入 surface 的 canonical scalar 序，不做 Unicode 归一化。"""
    if (type(value) is not str or not value
            or value[0] in " \t\r\n" or value[-1] in " \t\r\n"
            or "\r" in value or "\n" in value):
        raise PublicRouteClarificationCatalogError(
            f"{label} 必须是无换行的非空 input surface")
    return _unicode_scalars(tuple(ord(item) for item in value), label=label)


def _strict_int(value: Any, *, label: str, minimum: int = 0) -> int:
    """读取课程整数，拒绝 bool、float 和隐式数字转换。"""
    if type(value) is not int:
        raise PublicRouteClarificationCatalogError(
            f"{label} 必须是严格整数")
    return _u64(value, label=label, minimum=minimum)


def _exact_fields(value: Any, fields: tuple[str, ...], *, label: str) -> dict[str, Any]:
    """拒绝缺失、额外或依赖 dict 插入顺序的课程字段。"""
    if type(value) is not dict or len(value) != len(fields):
        raise PublicRouteClarificationCatalogError(f"{label} 字段数量漂移")
    for field in fields:
        if field not in value:
            raise PublicRouteClarificationCatalogError(
                f"{label} 缺少字段 {field}")
    for field in value:
        if field not in fields:
            raise PublicRouteClarificationCatalogError(
                f"{label} 含未注册字段 {field}")
    return value


def _hex_digest(value: Any, *, label: str) -> tuple[int, ...]:
    """解析固定小写 SHA-256 hex，输出显式 raw-u8[32]。"""
    if (type(value) is not str or len(value) != 64
            or any(item not in "0123456789abcdef" for item in value)):
        raise PublicRouteClarificationCatalogError(
            f"{label} 必须是固定小写 SHA-256 hex")
    try:
        result = tuple(bytes.fromhex(value))
    except ValueError as error:
        raise PublicRouteClarificationCatalogError(
            f"{label} hex 损坏") from error
    return _u8_tuple(result, label=label)


def _hex_ascii(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """将 raw-u8 digest 按固定小写十六进制编码为 raw ASCII。"""
    raw = _u8_tuple(value, label=label)
    result: list[int] = []
    for item in raw:
        high = item >> 4
        low = item & 0x0F
        result.append(0x30 + high if high < 10 else 0x61 + high - 10)
        result.append(0x30 + low if low < 10 else 0x61 + low - 10)
    return tuple(result)


def _unique_span(
        payload: bytes,
        needle: tuple[int, ...],
        *,
        label: str,
        ) -> tuple[int, int]:
    """以逐 byte 的有限扫描定位唯一 raw span，不依赖正则或文本搜索。"""
    target = _u8_tuple(needle, label=label)
    if type(payload) is not bytes or not target or len(target) > len(payload):
        raise PublicRouteClarificationCatalogError(
            f"{label} 在 source 中缺失或为空")
    found = -1
    for start in range(len(payload) - len(target) + 1):
        equal = True
        for offset, item in enumerate(target):
            if payload[start + offset] != item:
                equal = False
                break
        if equal:
            if found >= 0:
                raise PublicRouteClarificationCatalogError(
                    f"{label} 在 source 中不唯一")
            found = start
    if found < 0:
        raise PublicRouteClarificationCatalogError(f"{label} 在 source 中缺失")
    return found, found + len(target)


def _contains_u8(haystack: tuple[int, ...], needle: tuple[int, ...]) -> bool:
    """以显式 byte 比较判断一个 source span 是否含指定 raw surface。"""
    source = _u8_tuple(haystack, label="source span")
    target = _u8_tuple(needle, label="surface")
    if not target or len(target) > len(source):
        return False
    for start in range(len(source) - len(target) + 1):
        if all(source[start + offset] == item
               for offset, item in enumerate(target)):
            return True
    return False


def _has_duplicate_u8_vectors(values: tuple[tuple[int, ...], ...]) -> bool:
    """以显式两两比较检查重复向量，不让宿主 hash 容器参与协议判断。"""
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            if values[left] == values[right]:
                return True
    return False


def _json_string_literal_u8(
        surface_u8: tuple[int, ...],
        *,
        label: str,
        ) -> tuple[int, ...]:
    """重放本课程受限 JSON string raw bytes，只允许显式 LF 转义。"""
    surface = _u8_tuple(surface_u8, label=label)
    result = [0x22]
    for item in surface:
        if item == 0x0A:
            result.extend((0x5C, 0x6E))
        elif item < 0x20 or item in (0x22, 0x5C):
            raise PublicRouteClarificationCatalogError(
                f"{label} 含当前受限课程 JSON string 不允许的 byte")
        else:
            result.append(item)
    result.append(0x22)
    return tuple(result)


def _json_member_fragment_u8(
        field_ascii: tuple[int, ...],
        surface_u8: tuple[int, ...],
        *,
        label: str,
        ) -> tuple[int, ...]:
    """构造 canonical JSON 内一个 surface field 的精确 raw member span。"""
    field = _ascii_u8_tuple(field_ascii, label=f"{label} field")
    return (
        0x22,
        *field,
        0x22,
        0x3A,
        *_json_string_literal_u8(surface_u8, label=label),
    )


def _option_object_fragment_u8(
        candidate_identity_u8: tuple[int, ...],
        option_surface_u8: tuple[int, ...],
        ) -> tuple[int, ...]:
    """构造单个 canonical option object 的原始 byte span，用于 course readback。"""
    candidate = _u8_tuple(candidate_identity_u8, label="option candidate identity")
    if len(candidate) != 32:
        raise PublicRouteClarificationCatalogError("option candidate identity 长度漂移")
    return (
        0x7B,
        *_json_member_fragment_u8(
            tuple(b"candidate_identity"),
            _hex_ascii(candidate, label="option candidate identity"),
            label="option candidate identity",
        ),
        0x2C,
        *_json_member_fragment_u8(
            tuple(b"option_surface"),
            option_surface_u8,
            label="option surface",
        ),
        0x7D,
    )


def _payload(
        closure: PublicSourcePayloadClosureV1,
        logical_key: bytes,
        *,
        expected_digest: tuple[int, ...] | None,
        label: str,
        ) -> tuple[bytes, tuple[int, ...]]:
    """从已冻结 closure 读取一份 raw payload，并重核 length/SHA binding。"""
    if type(closure) is not PublicSourcePayloadClosureV1:
        raise TypeError("route clarification source payload closure 类型错误")
    try:
        record = closure.record_for(logical_key)
        payload = closure.payload_for(logical_key)
    except PublicSourcePayloadProviderError as error:
        raise PublicRouteClarificationCatalogError(
            f"{label} 不在 public payload closure") from error
    digest = _u8_tuple(tuple(public_source_payload_sha256_v1(payload)),
                       label=f"{label} SHA-256")
    if (record.logical_key != logical_key or record.raw_payload != payload
            or record.payload_length != len(payload)
            or tuple(record.raw_sha256) != digest
            or (expected_digest is not None and digest != expected_digest)):
        raise PublicRouteClarificationCatalogError(
            f"{label} payload/source digest 漂移")
    return payload, digest


def public_route_clarification_course_parser_record_v1() -> tuple[int, ...]:
    """冻结课程 parser 子集：UTF-8、LF JSONL、canonical JSON 与严格字段。"""
    return (
        PUBLIC_ROUTE_CLARIFICATION_COURSE_PARSER_RECORD_V1,
        1,  # 严格 UTF-8
        1,  # 仅 LF JSONL，且精确一个末尾 LF
        1,  # 规范 JSON 对象字节与 ASCII 键序
        1,  # 规范回读后无重复、默认或未知字段
        1,  # 仅严格整数数值字段
        1,  # 课程、输出、选项均须 raw-u8 来源回读
    )


def public_route_clarification_course_parser_identity_v1() -> tuple[int, ...]:
    """导出可被其他语言复现的 parser contract identity。"""
    return _identity(
        PUBLIC_ROUTE_CLARIFICATION_COURSE_PARSER_IDENTITY_DOMAIN_V1,
        public_route_clarification_course_parser_record_v1(),
        label="route clarification course parser",
    )


def candidate_identity_v1(
        candidate: SourceBoundSlotTargetCandidateV3,
        ) -> tuple[int, ...]:
    """仅从 V3 candidate canonical record 导出无循环的 candidate identity。"""
    if type(candidate) is not SourceBoundSlotTargetCandidateV3:
        raise TypeError("route candidate 必须是 V3 target candidate struct")
    return _identity(
        PUBLIC_ROUTE_CLARIFICATION_CANDIDATE_IDENTITY_DOMAIN_V1,
        candidate.canonical_record(),
        label="route candidate",
    )


def route_identity_record_v1(
        result_code: int,
        matched_frame_count: int,
        input_scalars: tuple[int, ...],
        candidate_identities_u8: tuple[tuple[int, ...], ...],
        ) -> tuple[int, ...]:
    """构造 route identity 的无循环 canonical integer record。

    候选顺序必须已是 resolver 的 canonical candidate 顺序；此函数既不排序也不
    以 display surface、closure identity 或 Python object identity 代替该顺序。
    """
    if result_code != DLG_RAW_REJECT_LEXICAL_AMBIGUOUS:
        raise PublicRouteClarificationCatalogError(
            "route identity 只能绑定 lexical ambiguous result")
    matched = _u64(matched_frame_count, label="route matched frame count", minimum=2)
    scalars = _unicode_scalars(input_scalars, label="route input scalars")
    if type(candidate_identities_u8) is not tuple:
        raise PublicRouteClarificationCatalogError(
            "route candidate identities 必须是 tuple")
    if len(candidate_identities_u8) != matched:
        raise PublicRouteClarificationCatalogError(
            "route candidate identity count 与 matched frame count 漂移")
    candidates: list[tuple[int, ...]] = []
    for ordinal, identity in enumerate(candidate_identities_u8, start=1):
        checked = _u8_tuple(identity, label=f"route candidate identity {ordinal}")
        if len(checked) != 32:
            raise PublicRouteClarificationCatalogError(
                "route candidate identity 必须是 raw-u8[32]")
        candidates.append(checked)
    if _has_duplicate_u8_vectors(tuple(candidates)):
        raise PublicRouteClarificationCatalogError(
            "route candidate identities 不得重复")
    result = [
        PUBLIC_ROUTE_CLARIFICATION_ROUTE_IDENTITY_RECORD_V1,
        result_code,
        matched,
    ]
    _pack(result, scalars, label="route input scalars")
    result.append(len(candidates))
    for ordinal, identity in enumerate(candidates, start=1):
        _pack(result, identity, label=f"route candidate identity {ordinal}")
    return tuple(result)


def route_identity_v1(
        result_code: int,
        matched_frame_count: int,
        input_scalars: tuple[int, ...],
        candidate_identities_u8: tuple[tuple[int, ...], ...],
        ) -> tuple[int, ...]:
    """从 code/count/input/canonical candidate identity 序列导出 route identity。"""
    return _identity(
        PUBLIC_ROUTE_CLARIFICATION_ROUTE_IDENTITY_DOMAIN_V1,
        route_identity_record_v1(
            result_code,
            matched_frame_count,
            input_scalars,
            candidate_identities_u8,
        ),
        label="route",
    )


def source_bound_route_candidate_identities_v1(
        resolution: SourceBoundSlotCompositionResolution,
        ) -> tuple[tuple[int, ...], ...]:
    """从一个可 offer 的 V3 source-bound ambiguity 导出有序 candidate identities。"""
    if type(resolution) is not SourceBoundSlotCompositionResolution:
        raise TypeError("route resolution 必须是 source-bound resolution struct")
    if (resolution.catalog.catalog_schema != SOURCE_BOUND_SLOT_CATALOG_SCHEMA_V3
            or resolution.result_code != DLG_RAW_REJECT_LEXICAL_AMBIGUOUS
            or resolution.matched_frame_count < 2
            or len(resolution.target_candidates) != resolution.matched_frame_count
            or any(candidate.verdict != SOURCE_BOUND_SLOT_CANDIDATE_SUPPORTED_V3
                   for candidate in resolution.target_candidates)):
        raise PublicRouteClarificationCatalogError(
            "source-bound resolution 不是可公开 route offer 的 V3 ambiguity")
    identities = tuple(
        candidate_identity_v1(candidate)
        for candidate in resolution.target_candidates
    )
    route_identity_record_v1(
        resolution.result_code,
        resolution.matched_frame_count,
        resolution.input_scalars,
        identities,
    )
    return identities


def route_identity_from_source_bound_resolution_v1(
        resolution: SourceBoundSlotCompositionResolution,
        ) -> tuple[int, ...]:
    """用同一 helper 从 V3 resolution 重算 route identity，供 outer runtime 复核。"""
    identities = source_bound_route_candidate_identities_v1(resolution)
    return route_identity_v1(
        resolution.result_code,
        resolution.matched_frame_count,
        resolution.input_scalars,
        identities,
    )


def _source_base_record(
        source_kind: int,
        logical_key_u8: tuple[int, ...],
        raw_sha256_u8: tuple[int, ...],
        span_start: int,
        span_end: int,
        span_u8: tuple[int, ...],
        license_id_u8: tuple[int, ...],
        attribution_u8: tuple[int, ...],
        ) -> tuple[int, ...]:
    """形成不含 identity 的来源本体 record，供独立重算与 readback 使用。"""
    result = [PUBLIC_ROUTE_CLARIFICATION_SOURCE_RECORD_V1, source_kind]
    for label, value in (
            ("source logical key", logical_key_u8),
            ("source raw SHA-256", raw_sha256_u8),
            ("source span bytes", span_u8),
            ("source license", license_id_u8),
            ("source attribution", attribution_u8)):
        _pack(result, value, label=label)
    result.extend((span_start, span_end))
    return tuple(result)


# object-model: value; representation=struct; interop=DLG-RAW-14
@dataclass(frozen=True, slots=True)
class RouteClarificationSourceV1:
    """课程或 A/B witness 内一个明确 raw span 的来源化整数 record。"""

    source_kind: int
    logical_key_u8: tuple[int, ...]
    raw_sha256_u8: tuple[int, ...]
    span_start: int
    span_end: int
    span_u8: tuple[int, ...]
    license_id_u8: tuple[int, ...]
    attribution_u8: tuple[int, ...]

    def __post_init__(self) -> None:
        """冻结 source kind、固定 logical key、raw digest、许可与精确 span。"""
        if self.source_kind not in (
                PUBLIC_ROUTE_CLARIFICATION_SOURCE_KIND_COURSE_V1,
                PUBLIC_ROUTE_CLARIFICATION_SOURCE_KIND_SURFACE_A_V1,
                PUBLIC_ROUTE_CLARIFICATION_SOURCE_KIND_SURFACE_B_V1):
            raise PublicRouteClarificationCatalogError("route source kind 未注册")
        key = _ascii_u8_tuple(self.logical_key_u8, label="route source logical key")
        expected = tuple(item[1] for item in _EXPECTED_SOURCE_KEY_BY_KIND
                         if item[0] == self.source_kind)
        if len(expected) != 1 or key != expected[0]:
            raise PublicRouteClarificationCatalogError(
                "route source kind/logical key binding 漂移")
        digest = _u8_tuple(self.raw_sha256_u8, label="route source SHA-256")
        if len(digest) != 32:
            raise PublicRouteClarificationCatalogError("route source SHA 长度漂移")
        start = _u64(self.span_start, label="route source span start")
        end = _u64(self.span_end, label="route source span end", minimum=1)
        span = _u8_tuple(self.span_u8, label="route source span bytes")
        if (not span or end < start or end - start != len(span)
                or self.license_id_u8 != _LICENSE_ID_U8):
            raise PublicRouteClarificationCatalogError(
                "route source span 或 CC0 license 漂移")
        attribution = _u8_tuple(
            self.attribution_u8,
            label="route source attribution",
        )
        if not attribution:
            raise PublicRouteClarificationCatalogError(
                "route source attribution 不得为空")

    def base_record(self) -> tuple[int, ...]:
        """返回可独立重算 source identity 的完整来源本体。"""
        return _source_base_record(
            self.source_kind,
            self.logical_key_u8,
            self.raw_sha256_u8,
            self.span_start,
            self.span_end,
            self.span_u8,
            self.license_id_u8,
            self.attribution_u8,
        )

    @property
    def source_identity_u8(self) -> tuple[int, ...]:
        """从完整来源本体导出 portable SHA-256 identity。"""
        return _identity(
            PUBLIC_ROUTE_CLARIFICATION_SOURCE_IDENTITY_DOMAIN_V1,
            self.base_record(),
            label="route source",
        )

    def canonical_record(self) -> tuple[int, ...]:
        """导出来源本体与可重算 identity 的完整有序整数 record。"""
        result = list(self.base_record())
        _pack(result, self.source_identity_u8, label="route source identity")
        return tuple(result)


# object-model: value; representation=struct; interop=DLG-RAW-14
@dataclass(frozen=True, slots=True)
class RouteClarificationOutputReadbackV1:
    """多行 selector 输出的严格 UTF-8、line-body/LF roundtrip record。"""

    result_code: int
    output_u8: tuple[int, ...]
    unicode_scalars: tuple[int, ...]
    line_bodies_u8: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        """验证 raw UTF-8、显式 scalar、非空行体和 LF framing 逐 byte 一致。"""
        if self.result_code != PUBLIC_ROUTE_CLARIFICATION_OUTPUT_READBACK_ACCEPTED_V1:
            raise PublicRouteClarificationCatalogError(
                "route output readback result code 未注册")
        output = _u8_tuple(self.output_u8, label="route output readback output")
        if not output:
            raise PublicRouteClarificationCatalogError(
                "route output readback output 不得为空")
        scalars = _unicode_scalars(
            self.unicode_scalars,
            label="route output readback scalars",
        )
        decoded = decode_utf8_v1(output)
        if (decoded is None or decoded != scalars
                or encode_utf8_v1(scalars) != output
                or any(item < 0x20 and item != 0x0A or item == 0x7F
                       for item in scalars)):
            raise PublicRouteClarificationCatalogError(
                "route output readback UTF-8/scalar roundtrip 漂移")
        if (type(self.line_bodies_u8) is not tuple
                or not self.line_bodies_u8):
            raise PublicRouteClarificationCatalogError(
                "route output readback line bodies 类型或数量漂移")
        rebuilt: list[int] = []
        for ordinal, body in enumerate(self.line_bodies_u8):
            line = _u8_tuple(
                body,
                label=f"route output readback line body {ordinal + 1}",
            )
            if (not line or any(item < 0x20 or item == 0x7F
                                for item in line)):
                raise PublicRouteClarificationCatalogError(
                    "route output readback line body 必须是非空且无控制字节")
            if ordinal:
                rebuilt.append(0x0A)
            rebuilt.extend(line)
        if tuple(rebuilt) != output:
            raise PublicRouteClarificationCatalogError(
                "route output readback line-body/LF framing 漂移")

    @property
    def accepted(self) -> bool:
        """当前仅有完整 UTF-8 roundtrip 成功这一种已注册 readback 状态。"""
        return self.result_code == PUBLIC_ROUTE_CLARIFICATION_OUTPUT_READBACK_ACCEPTED_V1

    def canonical_record(self) -> tuple[int, ...]:
        """导出不依赖宿主文本或 newline 转换的完整输出回读整数 record。"""
        result = [
            PUBLIC_ROUTE_CLARIFICATION_OUTPUT_READBACK_RECORD_V1,
            self.result_code,
        ]
        _pack(result, self.output_u8, label="route output readback output")
        _pack(result, self.unicode_scalars,
              label="route output readback scalars")
        result.append(len(self.line_bodies_u8))
        for ordinal, body in enumerate(self.line_bodies_u8, start=1):
            _pack(result, body,
                  label=f"route output readback line body {ordinal}")
        return tuple(result)



def _line_bodies_from_output_u8(
        output_u8: tuple[int, ...],
        ) -> tuple[tuple[int, ...], ...]:
    """以显式 LF separator 从 raw-u8 输出切出有序非空 line body record。"""
    output = _u8_tuple(output_u8, label="route output line-body input")
    bodies: list[tuple[int, ...]] = []
    current: list[int] = []
    for item in output:
        if item == 0x0A:
            if not current:
                raise PublicRouteClarificationCatalogError(
                    "route output 不得含空行或起始 LF")
            bodies.append(tuple(current))
            current = []
        else:
            current.append(item)
    if not current:
        raise PublicRouteClarificationCatalogError(
            "route output 不得以 LF 结束")
    bodies.append(tuple(current))
    return tuple(bodies)


def route_clarification_output_readback_v1(
        output_u8: tuple[int, ...],
        ) -> RouteClarificationOutputReadbackV1:
    """从输出 raw-u8 机械重建多行 UTF-8 readback，不进入单行用户输入 ingress。"""
    output = _u8_tuple(output_u8, label="route output readback input")
    scalars = decode_utf8_v1(output)
    if scalars is None:
        raise PublicRouteClarificationCatalogError(
            "route output readback 遇到非规范 UTF-8")
    return RouteClarificationOutputReadbackV1(
        PUBLIC_ROUTE_CLARIFICATION_OUTPUT_READBACK_ACCEPTED_V1,
        output,
        scalars,
        _line_bodies_from_output_u8(output),
    )


def _option_base_record(
        candidate_identity_u8: tuple[int, ...],
        option_surface_u8: tuple[int, ...],
        candidate_course_source: RouteClarificationSourceV1,
        surface_sources: tuple[RouteClarificationSourceV1, ...],
        ) -> tuple[int, ...]:
    """形成不含 option identity 的候选 surface/课程/witness 完整本体。"""
    result = [PUBLIC_ROUTE_CLARIFICATION_OPTION_RECORD_V1]
    _pack(result, candidate_identity_u8, label="option candidate identity")
    _pack(result, option_surface_u8, label="option surface")
    _pack(result, candidate_course_source.canonical_record(),
          label="option candidate course source")
    result.append(len(surface_sources))
    for ordinal, source in enumerate(surface_sources, start=1):
        _pack(result, source.canonical_record(),
              label=f"option surface source {ordinal}")
    return tuple(result)


# object-model: value; representation=struct; interop=DLG-RAW-14
@dataclass(frozen=True, slots=True)
class RouteClarificationOptionV1:
    """一个 candidate identity 到完整可重输问句的公开课程映射。"""

    candidate_identity_u8: tuple[int, ...]
    option_surface_u8: tuple[int, ...]
    candidate_course_source: RouteClarificationSourceV1
    surface_sources: tuple[RouteClarificationSourceV1, ...]

    def __post_init__(self) -> None:
        """验证 candidate identity、course object span 与 A/B exact surface witness。"""
        candidate = _u8_tuple(
            self.candidate_identity_u8,
            label="option candidate identity",
        )
        if len(candidate) != 32:
            raise PublicRouteClarificationCatalogError(
                "option candidate identity 必须是 raw-u8[32]")
        surface = _u8_tuple(self.option_surface_u8, label="option surface")
        if not surface or any(item in (0x0A, 0x0D) for item in surface):
            raise PublicRouteClarificationCatalogError(
                "option surface 必须是非空单行 raw-u8")
        if type(self.candidate_course_source) is not RouteClarificationSourceV1:
            raise TypeError("option 缺 candidate course source")
        if (self.candidate_course_source.source_kind
                != PUBLIC_ROUTE_CLARIFICATION_SOURCE_KIND_COURSE_V1
                or self.candidate_course_source.span_u8
                != _option_object_fragment_u8(candidate, surface)):
            raise PublicRouteClarificationCatalogError(
                "option candidate course source/readback 漂移")
        if (type(self.surface_sources) is not tuple
                or len(self.surface_sources) != 2
                or any(type(item) is not RouteClarificationSourceV1
                       for item in self.surface_sources)):
            raise PublicRouteClarificationCatalogError(
                "option 必须有两个 surface source")
        expected_kinds = (
            PUBLIC_ROUTE_CLARIFICATION_SOURCE_KIND_SURFACE_A_V1,
            PUBLIC_ROUTE_CLARIFICATION_SOURCE_KIND_SURFACE_B_V1,
        )
        if tuple(item.source_kind for item in self.surface_sources) != expected_kinds:
            raise PublicRouteClarificationCatalogError(
                "option surface source 顺序或 kind 漂移")
        if any(item.span_u8 != surface for item in self.surface_sources):
            raise PublicRouteClarificationCatalogError(
                "option surface witness 未精确见证完整重输问句")

    def base_record(self) -> tuple[int, ...]:
        """返回可重算 option identity 的全部候选课程/来源本体。"""
        return _option_base_record(
            self.candidate_identity_u8,
            self.option_surface_u8,
            self.candidate_course_source,
            self.surface_sources,
        )

    @property
    def option_identity_u8(self) -> tuple[int, ...]:
        """从候选 surface 的完整来源本体导出 portable option identity。"""
        return _identity(
            PUBLIC_ROUTE_CLARIFICATION_OPTION_IDENTITY_DOMAIN_V1,
            self.base_record(),
            label="route option",
        )

    def canonical_record(self) -> tuple[int, ...]:
        """导出 option 本体与可重算 identity 的完整整数 record。"""
        result = list(self.base_record())
        _pack(result, self.option_identity_u8, label="route option identity")
        return tuple(result)


def _form_base_record(
        form_id_u8: tuple[int, ...],
        result_code: int,
        matched_frame_count: int,
        input_scalars: tuple[int, ...],
        route_identity_u8: tuple[int, ...],
        output_max_bytes: int,
        output_u8: tuple[int, ...],
        output_readback: RouteClarificationOutputReadbackV1,
        course_source: RouteClarificationSourceV1,
        output_course_source: RouteClarificationSourceV1,
        output_surface_sources: tuple[RouteClarificationSourceV1, ...],
        options: tuple[RouteClarificationOptionV1, ...],
        ) -> tuple[int, ...]:
    """形成不含 form identity 的整条 route/course/output/options 本体 record。"""
    result = [
        PUBLIC_ROUTE_CLARIFICATION_FORM_RECORD_V1,
        result_code,
        matched_frame_count,
        output_max_bytes,
    ]
    for label, value in (
            ("route form id", form_id_u8),
            ("route form input", input_scalars),
            ("route form route identity", route_identity_u8),
            ("route form output", output_u8),
            ("route form output readback", output_readback.canonical_record()),
            ("route form course source", course_source.canonical_record()),
            ("route form output course source",
             output_course_source.canonical_record())):
        _pack(result, value, label=label)
    result.append(len(output_surface_sources))
    for ordinal, source in enumerate(output_surface_sources, start=1):
        _pack(result, source.canonical_record(),
              label=f"route form output surface source {ordinal}")
    result.append(len(options))
    for ordinal, option in enumerate(options, start=1):
        _pack(result, option.canonical_record(), label=f"route form option {ordinal}")
    return tuple(result)


# object-model: value; representation=struct; interop=DLG-RAW-14
@dataclass(frozen=True, slots=True)
class RouteClarificationFormV1:
    """一个 V3 ambiguity route 到来源绑定完整重输选项的课程 form。"""

    form_id_u8: tuple[int, ...]
    result_code: int
    matched_frame_count: int
    input_scalars: tuple[int, ...]
    route_identity_u8: tuple[int, ...]
    output_max_bytes: int
    output_u8: tuple[int, ...]
    output_readback: RouteClarificationOutputReadbackV1
    course_source: RouteClarificationSourceV1
    output_course_source: RouteClarificationSourceV1
    output_surface_sources: tuple[RouteClarificationSourceV1, ...]
    options: tuple[RouteClarificationOptionV1, ...]

    def __post_init__(self) -> None:
        """闭合 route identity、完整 output/readback 与每个候选的 A/B 证据。"""
        _ascii_u8_tuple(self.form_id_u8, label="route form id")
        matched = _u64(
            self.matched_frame_count,
            label="route form matched frame count",
            minimum=2,
        )
        if self.result_code != DLG_RAW_REJECT_LEXICAL_AMBIGUOUS:
            raise PublicRouteClarificationCatalogError(
                "route form 只能映射 lexical ambiguous result")
        scalars = _unicode_scalars(self.input_scalars, label="route form input")
        route_identity = _u8_tuple(
            self.route_identity_u8,
            label="route form route identity",
        )
        if len(route_identity) != 32:
            raise PublicRouteClarificationCatalogError(
                "route form route identity 长度漂移")
        budget = _u64(
            self.output_max_bytes,
            label="route form output budget",
            minimum=1,
        )
        output = _u8_tuple(self.output_u8, label="route form output")
        if not output or len(output) > budget:
            raise PublicRouteClarificationCatalogError(
                "route form output 超出课程预算")
        if type(self.output_readback) is not RouteClarificationOutputReadbackV1:
            raise TypeError("route form output readback 类型错误")
        if (self.output_readback.output_u8 != output
                or self.output_readback.canonical_record()
                != route_clarification_output_readback_v1(
                    output).canonical_record()):
            raise PublicRouteClarificationCatalogError(
                "route form output readback 漂移")
        if (type(self.course_source) is not RouteClarificationSourceV1
                or self.course_source.source_kind
                != PUBLIC_ROUTE_CLARIFICATION_SOURCE_KIND_COURSE_V1
                or type(self.output_course_source) is not RouteClarificationSourceV1
                or self.output_course_source.source_kind
                != PUBLIC_ROUTE_CLARIFICATION_SOURCE_KIND_COURSE_V1):
            raise PublicRouteClarificationCatalogError(
                "route form 缺 course source")
        expected_output_member = _json_member_fragment_u8(
            tuple(b"output_surface"),
            output,
            label="route form output",
        )
        if (self.output_course_source.span_u8 != expected_output_member
                or not _contains_u8(
                    self.course_source.span_u8,
                    self.output_course_source.span_u8)):
            raise PublicRouteClarificationCatalogError(
                "route form output course readback 漂移")
        if (type(self.output_surface_sources) is not tuple
                or len(self.output_surface_sources) != 2
                or any(type(item) is not RouteClarificationSourceV1
                       for item in self.output_surface_sources)):
            raise PublicRouteClarificationCatalogError(
                "route form 必须有两个 output surface source")
        expected_kinds = (
            PUBLIC_ROUTE_CLARIFICATION_SOURCE_KIND_SURFACE_A_V1,
            PUBLIC_ROUTE_CLARIFICATION_SOURCE_KIND_SURFACE_B_V1,
        )
        if (tuple(item.source_kind for item in self.output_surface_sources)
                != expected_kinds
                or any(item.span_u8 != output
                       for item in self.output_surface_sources)):
            raise PublicRouteClarificationCatalogError(
                "route form output A/B witness 漂移")
        if (type(self.options) is not tuple or len(self.options) != matched
                or any(type(item) is not RouteClarificationOptionV1
                       for item in self.options)):
            raise PublicRouteClarificationCatalogError(
                "route form option 数量或类型漂移")
        identities = tuple(item.candidate_identity_u8 for item in self.options)
        if _has_duplicate_u8_vectors(identities):
            raise PublicRouteClarificationCatalogError(
                "route form candidate identities 不得重复")
        expected_route_identity = route_identity_v1(
            self.result_code,
            matched,
            scalars,
            identities,
        )
        if route_identity != expected_route_identity:
            raise PublicRouteClarificationCatalogError(
                "route form route identity 与候选顺序/输入漂移")
        expected_lines = (
            self.output_readback.line_bodies_u8[0],
            *(option.option_surface_u8 for option in self.options),
        )
        if (len(self.output_readback.line_bodies_u8) != matched + 1
                or self.output_readback.line_bodies_u8 != expected_lines):
            raise PublicRouteClarificationCatalogError(
                "route form output line-body/option 顺序漂移")
        for option in self.options:
            if (not _contains_u8(self.course_source.span_u8,
                                 option.candidate_course_source.span_u8)
                    or _unique_span(
                        bytes(output),
                        option.option_surface_u8,
                        label="route form option output",
                    )[0] < 0):
                raise PublicRouteClarificationCatalogError(
                    "route form option 未由 course/output 精确闭合")

    def base_record(self) -> tuple[int, ...]:
        """返回 form identity 的完整 route/course/options 本体 record。"""
        return _form_base_record(
            self.form_id_u8,
            self.result_code,
            self.matched_frame_count,
            self.input_scalars,
            self.route_identity_u8,
            self.output_max_bytes,
            self.output_u8,
            self.output_readback,
            self.course_source,
            self.output_course_source,
            self.output_surface_sources,
            self.options,
        )

    @property
    def form_identity_u8(self) -> tuple[int, ...]:
        """从 form 本体导出 portable SHA-256 identity。"""
        return _identity(
            PUBLIC_ROUTE_CLARIFICATION_FORM_IDENTITY_DOMAIN_V1,
            self.base_record(),
            label="route form",
        )

    def canonical_record(self) -> tuple[int, ...]:
        """导出 form 本体与可重算 identity 的完整 canonical integer record。"""
        result = list(self.base_record())
        _pack(result, self.form_identity_u8, label="route form identity")
        return tuple(result)


# object-model: value; representation=struct; interop=DLG-RAW-14
@dataclass(frozen=True, slots=True)
class PublicRouteClarificationCatalogV1:
    """当前只含一条公开、来源绑定、两候选 route clarification form 的 catalog。"""

    source_payload_closure_identity_u8: tuple[int, ...]
    forms: tuple[RouteClarificationFormV1, ...]

    def __post_init__(self) -> None:
        """闭合 closure identity、唯一 route form 与候选顺序而不依赖 host map。"""
        closure_identity = _u8_tuple(
            self.source_payload_closure_identity_u8,
            label="route catalog closure identity",
        )
        if len(closure_identity) != 32:
            raise PublicRouteClarificationCatalogError(
                "route catalog closure identity 长度漂移")
        if (type(self.forms) is not tuple or len(self.forms) != 1
                or any(type(item) is not RouteClarificationFormV1
                       for item in self.forms)):
            raise PublicRouteClarificationCatalogError(
                "route catalog 当前必须恰有一条 form")

    def canonical_record(self) -> tuple[int, ...]:
        """导出不含 Python object identity 的完整 catalog record。"""
        result = [
            PUBLIC_ROUTE_CLARIFICATION_CATALOG_RECORD_V1,
            PUBLIC_ROUTE_CLARIFICATION_CATALOG_SCHEMA_V1,
        ]
        _pack(result, public_route_clarification_course_parser_record_v1(),
              label="route catalog course parser")
        _pack(result, public_route_clarification_course_parser_identity_v1(),
              label="route catalog course parser identity")
        _pack(result, self.source_payload_closure_identity_u8,
              label="route catalog closure identity")
        result.append(len(self.forms))
        for ordinal, form in enumerate(self.forms, start=1):
            _pack(result, form.canonical_record(), label=f"route catalog form {ordinal}")
        return tuple(result)

    @property
    def catalog_identity_u8(self) -> tuple[int, ...]:
        """从完整 catalog canonical record 导出 portable SHA identity。"""
        return _identity(
            PUBLIC_ROUTE_CLARIFICATION_CATALOG_IDENTITY_DOMAIN_V1,
            self.canonical_record(),
            label="route catalog",
        )

    def form_for_route_identity_u8(
            self,
            route_identity_u8: tuple[int, ...],
            ) -> RouteClarificationFormV1 | None:
        """按 route raw-u8 identity 线性选择唯一 form，不使用 host dict。"""
        identity = _u8_tuple(route_identity_u8, label="route lookup identity")
        if len(identity) != 32:
            raise PublicRouteClarificationCatalogError(
                "route lookup identity 必须是 raw-u8[32]")
        selected = None
        for form in self.forms:
            if form.route_identity_u8 == identity:
                if selected is not None:
                    raise PublicRouteClarificationCatalogError(
                        "route identity 在 catalog 中不唯一")
                selected = form
        return selected

    def form_for_source_bound_resolution(
            self,
            resolution: SourceBoundSlotCompositionResolution,
            ) -> RouteClarificationFormV1 | None:
        """为一个可 offer 的 V3 resolution 查唯一 form；不保留 runtime candidate 本体。"""
        if type(resolution) is not SourceBoundSlotCompositionResolution:
            raise TypeError("route lookup resolution 类型错误")
        if (resolution.catalog.catalog_schema != SOURCE_BOUND_SLOT_CATALOG_SCHEMA_V3
                or resolution.result_code != DLG_RAW_REJECT_LEXICAL_AMBIGUOUS
                or resolution.matched_frame_count < 2
                or len(resolution.target_candidates) != resolution.matched_frame_count
                or any(candidate.verdict != SOURCE_BOUND_SLOT_CANDIDATE_SUPPORTED_V3
                       for candidate in resolution.target_candidates)):
            return None
        identities = source_bound_route_candidate_identities_v1(resolution)
        route_identity = route_identity_v1(
            resolution.result_code,
            resolution.matched_frame_count,
            resolution.input_scalars,
            identities,
        )
        form = self.form_for_route_identity_u8(route_identity)
        if form is None:
            return None
        if tuple(option.candidate_identity_u8 for option in form.options) != identities:
            raise PublicRouteClarificationCatalogError(
                "route form candidate identity 序列与 resolution 漂移")
        return form


# object-model: value; representation=struct; interop=DLG-RAW-14
@dataclass(frozen=True, slots=True)
class PublicRouteClarificationCatalogValidationEntryV1:
    """一次已完成 selector catalog 验证的可丢弃派生条目。

    ``closure_record`` 与 ``catalog_record`` 都是显式 immutable integer
    records；它们只作为缓存键，不参与 dialogue binding/canonical record。
    ``validated_catalog`` 仅作“曾经完整验证”的派生见证；命中后返回当前
    catalog，不把缓存对象作为语义来源，也绝不以 Python object identity 命中。
    """

    closure_record: tuple[int, ...]
    closure_identity_u8: tuple[int, ...]
    catalog_record: tuple[int, ...]
    catalog_identity_u8: tuple[int, ...]
    validated_catalog: "PublicRouteClarificationCatalogV1"


# object-model: mutable derived cache; representation=struct; interop=DLG-RAW-14
@dataclass(slots=True)
class PublicRouteClarificationCatalogValidationCacheV1:
    """单条 selector 派生验证缓存；可随时清空，不是协议状态。"""

    entries: tuple[PublicRouteClarificationCatalogValidationEntryV1, ...] = ()

    def clear(self) -> None:
        """显式丢弃派生条目，不影响任何 runtime/state record。"""
        self.entries = ()


def _canonical_jsonl_lines(
        payload: bytes,
        ) -> tuple[tuple[int, bytes, dict[str, Any]], ...]:
    """按 raw byte 行读取严格 LF canonical JSONL，拒绝 CR、空行和尾部变体。"""
    if (type(payload) is not bytes or not payload or not payload.endswith(b"\n")
            or b"\r" in payload):
        raise PublicRouteClarificationCatalogError(
            "route clarification course 不是规范 LF JSONL")
    result: list[tuple[int, bytes, dict[str, Any]]] = []
    cursor = 0
    for line in payload[:-1].split(b"\n"):
        if not line:
            raise PublicRouteClarificationCatalogError(
                "route clarification course 含空 JSONL line")
        try:
            parsed = parse_canonical_json_bytes(line, require_object=True)
        except DatasetContractError as error:
            raise PublicRouteClarificationCatalogError(
                "route clarification course JSON line 不规范") from error
        if type(parsed) is not dict:
            raise PublicRouteClarificationCatalogError(
                "route clarification course JSON line 未产生 object")
        result.append((cursor, line, parsed))
        cursor += len(line) + 1
    return tuple(result)


def _course_source(
        course_digest: tuple[int, ...],
        span_start: int,
        span_end: int,
        span_u8: tuple[int, ...],
        attribution_u8: tuple[int, ...],
        ) -> RouteClarificationSourceV1:
    """构造课程行或其精确 canonical JSON member/object raw span 的 source record。"""
    return RouteClarificationSourceV1(
        PUBLIC_ROUTE_CLARIFICATION_SOURCE_KIND_COURSE_V1,
        tuple(PUBLIC_ROUTE_CLARIFICATION_COURSE_LOGICAL_KEY_V1),
        course_digest,
        span_start,
        span_end,
        span_u8,
        _LICENSE_ID_U8,
        attribution_u8,
    )


def _surface_descriptor(
        raw: Any,
        *,
        expected_logical_key: bytes,
        closure: PublicSourcePayloadClosureV1,
        surface_u8: tuple[int, ...],
        source_kind: int,
        license_id_u8: tuple[int, ...],
        label: str,
        ) -> RouteClarificationSourceV1:
    """由课程 descriptor 编译 A/B witness 的 SHA、唯一 span 与 attribution。"""
    descriptor = _exact_fields(raw, _SURFACE_FIELDS, label=label)
    logical_key = _ascii_bytes(
        descriptor["relative_path"],
        label=f"{label}.relative_path",
    )
    if bytes(logical_key) != expected_logical_key:
        raise PublicRouteClarificationCatalogError(
            f"{label}.relative_path 未绑定冻结 witness")
    expected_digest = _hex_digest(
        descriptor["raw_sha256"],
        label=f"{label}.raw_sha256",
    )
    payload, digest = _payload(
        closure,
        expected_logical_key,
        expected_digest=expected_digest,
        label=label,
    )
    attribution = _display_u8(
        descriptor["attribution"],
        label=f"{label}.attribution",
        allow_line_feed=False,
    )
    start, end = _unique_span(payload, surface_u8, label=label)
    return RouteClarificationSourceV1(
        source_kind,
        logical_key,
        digest,
        start,
        end,
        surface_u8,
        license_id_u8,
        attribution,
    )


def _candidate_identities_from_course(
        raw: Any,
        *,
        matched_frame_count: int,
        ) -> tuple[tuple[int, ...], ...]:
    """读取有序 candidate identity vector；顺序由课程重放 resolver 顺序锁定。"""
    if type(raw) is not list or len(raw) != matched_frame_count:
        raise PublicRouteClarificationCatalogError(
            "course candidate identities 数量与 matched frame count 漂移")
    identities = tuple(
        _hex_digest(value, label=f"course candidate identity {ordinal}")
        for ordinal, value in enumerate(raw, start=1)
    )
    if _has_duplicate_u8_vectors(identities):
        raise PublicRouteClarificationCatalogError(
            "course candidate identities 不得重复")
    return identities


def _option_from_course(
        raw: Any,
        *,
        expected_candidate_identity_u8: tuple[int, ...],
        line_offset: int,
        line: bytes,
        course_digest: tuple[int, ...],
        course_attribution_u8: tuple[int, ...],
        closure: PublicSourcePayloadClosureV1,
        license_id_u8: tuple[int, ...],
        surface_a_raw: Any,
        surface_b_raw: Any,
        ordinal: int,
        ) -> RouteClarificationOptionV1:
    """把一条课程 option 编译为 candidate identity 与双 witness 的完整 record。"""
    record = _exact_fields(raw, _OPTION_FIELDS, label=f"course option {ordinal}")
    candidate_identity = _hex_digest(
        record["candidate_identity"],
        label=f"course option {ordinal} candidate identity",
    )
    if candidate_identity != expected_candidate_identity_u8:
        raise PublicRouteClarificationCatalogError(
            "course option candidate identity 与有序 route vector 漂移")
    option_surface = _display_u8(
        record["option_surface"],
        label=f"course option {ordinal} surface",
        allow_line_feed=False,
    )
    fragment = _option_object_fragment_u8(candidate_identity, option_surface)
    start, end = _unique_span(line, fragment, label=f"course option {ordinal}")
    course_source = _course_source(
        course_digest,
        line_offset + start,
        line_offset + end,
        fragment,
        course_attribution_u8,
    )
    surface_a = _surface_descriptor(
        surface_a_raw,
        expected_logical_key=PUBLIC_ROUTE_CLARIFICATION_SURFACE_A_LOGICAL_KEY_V1,
        closure=closure,
        surface_u8=option_surface,
        source_kind=PUBLIC_ROUTE_CLARIFICATION_SOURCE_KIND_SURFACE_A_V1,
        license_id_u8=license_id_u8,
        label=f"course option {ordinal} surface_a",
    )
    surface_b = _surface_descriptor(
        surface_b_raw,
        expected_logical_key=PUBLIC_ROUTE_CLARIFICATION_SURFACE_B_LOGICAL_KEY_V1,
        closure=closure,
        surface_u8=option_surface,
        source_kind=PUBLIC_ROUTE_CLARIFICATION_SOURCE_KIND_SURFACE_B_V1,
        license_id_u8=license_id_u8,
        label=f"course option {ordinal} surface_b",
    )
    return RouteClarificationOptionV1(
        candidate_identity,
        option_surface,
        course_source,
        (surface_a, surface_b),
    )


def _form_from_course_line(
        line_offset: int,
        line: bytes,
        raw: dict[str, Any],
        *,
        closure: PublicSourcePayloadClosureV1,
        course_digest: tuple[int, ...],
        ordinal: int,
        ) -> RouteClarificationFormV1:
    """将一条 canonical course line 编译为完整 route/form/options 来源闭包。"""
    record = _exact_fields(raw, _COURSE_FIELDS, label=f"route course line {ordinal}")
    if _strict_int(record["schema"], label="route course schema") != (
            PUBLIC_ROUTE_CLARIFICATION_CATALOG_SCHEMA_V1):
        raise PublicRouteClarificationCatalogError("route course schema 未注册")
    result_code = _strict_int(record["result_code"], label="route course result code")
    if result_code != DLG_RAW_REJECT_LEXICAL_AMBIGUOUS:
        raise PublicRouteClarificationCatalogError(
            "route course 只能描述 lexical ambiguous result")
    matched_frame_count = _strict_int(
        record["matched_frame_count"],
        label="route course matched frame count",
        minimum=2,
    )
    if matched_frame_count != 2:
        raise PublicRouteClarificationCatalogError(
            "当前 route course 必须精确冻结两个候选")
    form_id = _ascii_bytes(record["form_id"], label="route course form id")
    license_id = _ascii_bytes(record["license_id"], label="route course license")
    if license_id != _LICENSE_ID_U8:
        raise PublicRouteClarificationCatalogError(
            "route course license 必须是 CC0-1.0")
    input_scalars = _surface_scalars(
        record["input_surface"],
        label="route course input surface",
    )
    candidate_identities = _candidate_identities_from_course(
        record["candidate_identities"],
        matched_frame_count=matched_frame_count,
    )
    route_identity = _hex_digest(
        record["route_identity"],
        label="route course route identity",
    )
    expected_route_identity = route_identity_v1(
        result_code,
        matched_frame_count,
        input_scalars,
        candidate_identities,
    )
    if route_identity != expected_route_identity:
        raise PublicRouteClarificationCatalogError(
            "route course route identity 与 canonical input/candidate vector 漂移")
    output = _display_u8(
        record["output_surface"],
        label="route course output surface",
        allow_line_feed=True,
    )
    output_max = _strict_int(
        record["output_max_bytes"],
        label="route course output max bytes",
        minimum=1,
    )
    if len(output) > output_max:
        raise PublicRouteClarificationCatalogError(
            "route course output 超出显式预算")
    output_readback = route_clarification_output_readback_v1(output)
    course_attribution = _display_u8(
        record["course_attribution"],
        label="route course attribution",
        allow_line_feed=False,
    )
    course_source = _course_source(
        course_digest,
        line_offset,
        line_offset + len(line),
        tuple(line),
        course_attribution,
    )
    output_member = _json_member_fragment_u8(
        tuple(b"output_surface"),
        output,
        label="route course output surface",
    )
    output_start, output_end = _unique_span(
        line,
        output_member,
        label="route course output member",
    )
    output_course_source = _course_source(
        course_digest,
        line_offset + output_start,
        line_offset + output_end,
        output_member,
        course_attribution,
    )
    output_surface_a = _surface_descriptor(
        record["surface_a"],
        expected_logical_key=PUBLIC_ROUTE_CLARIFICATION_SURFACE_A_LOGICAL_KEY_V1,
        closure=closure,
        surface_u8=output,
        source_kind=PUBLIC_ROUTE_CLARIFICATION_SOURCE_KIND_SURFACE_A_V1,
        license_id_u8=license_id,
        label="route course output surface_a",
    )
    output_surface_b = _surface_descriptor(
        record["surface_b"],
        expected_logical_key=PUBLIC_ROUTE_CLARIFICATION_SURFACE_B_LOGICAL_KEY_V1,
        closure=closure,
        surface_u8=output,
        source_kind=PUBLIC_ROUTE_CLARIFICATION_SOURCE_KIND_SURFACE_B_V1,
        license_id_u8=license_id,
        label="route course output surface_b",
    )
    options_raw = record["options"]
    if type(options_raw) is not list or len(options_raw) != matched_frame_count:
        raise PublicRouteClarificationCatalogError(
            "route course option 数量与 matched frame count 漂移")
    options = tuple(
        _option_from_course(
            option_raw,
            expected_candidate_identity_u8=candidate_identities[index],
            line_offset=line_offset,
            line=line,
            course_digest=course_digest,
            course_attribution_u8=course_attribution,
            closure=closure,
            license_id_u8=license_id,
            surface_a_raw=record["surface_a"],
            surface_b_raw=record["surface_b"],
            ordinal=index + 1,
        )
        for index, option_raw in enumerate(options_raw)
    )
    return RouteClarificationFormV1(
        form_id,
        result_code,
        matched_frame_count,
        input_scalars,
        route_identity,
        output_max,
        output,
        output_readback,
        course_source,
        output_course_source,
        (output_surface_a, output_surface_b),
        options,
    )


def load_public_route_clarification_catalog_from_closure(
        source_payload_closure: PublicSourcePayloadClosureV1,
        ) -> PublicRouteClarificationCatalogV1:
    """从固定 closure 重建当前唯一公开 route clarification form。"""
    if type(source_payload_closure) is not PublicSourcePayloadClosureV1:
        raise TypeError("route clarification catalog 需要 source payload closure")
    course_payload, course_digest = _payload(
        source_payload_closure,
        PUBLIC_ROUTE_CLARIFICATION_COURSE_LOGICAL_KEY_V1,
        expected_digest=None,
        label="route clarification course",
    )
    lines = _canonical_jsonl_lines(course_payload)
    if len(lines) != 1:
        raise PublicRouteClarificationCatalogError(
            "当前 route clarification course 必须恰有一条 form")
    offset, line, raw = lines[0]
    form = _form_from_course_line(
        offset,
        line,
        raw,
        closure=source_payload_closure,
        course_digest=course_digest,
        ordinal=1,
    )
    return PublicRouteClarificationCatalogV1(
        tuple(source_payload_closure.closure_identity),
        (form,),
    )


def validate_public_route_clarification_catalog_v1(
        catalog: PublicRouteClarificationCatalogV1,
        source_payload_closure: PublicSourcePayloadClosureV1,
        ) -> PublicRouteClarificationCatalogV1:
    """以当前 closure 重建 catalog，拒绝内存篡改和 foreign course binding。"""
    if type(catalog) is not PublicRouteClarificationCatalogV1:
        raise TypeError("route clarification catalog 类型错误")
    rebuilt = load_public_route_clarification_catalog_from_closure(
        source_payload_closure)
    if (catalog.canonical_record() != rebuilt.canonical_record()
            or catalog.catalog_identity_u8 != rebuilt.catalog_identity_u8):
        raise PublicRouteClarificationCatalogError(
            "route clarification catalog 未绑定当前 public closure")
    return rebuilt


def _catalog_validation_key_v1(
        catalog: PublicRouteClarificationCatalogV1,
        source_payload_closure: PublicSourcePayloadClosureV1,
        ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """形成缓存用显式 key，并在读 key 时观察所有可变宿主字段。

    这里故意重算 canonical records，但不重跑 JSONL parser 或 portable SHA
    identity。canonical record 已包含所有语义字段；closure identity 另行纳入
    key，以便 ``object.__setattr__`` 篡改显式 identity 时不能命中旧条目。
    """
    if type(catalog) is not PublicRouteClarificationCatalogV1:
        raise TypeError("route clarification catalog 类型错误")
    if type(source_payload_closure) is not PublicSourcePayloadClosureV1:
        raise TypeError("route clarification source closure 类型错误")
    closure_identity = source_payload_closure.closure_identity
    if type(closure_identity) is not bytes or len(closure_identity) != 32:
        raise PublicRouteClarificationCatalogError(
            "route clarification closure identity 不是 raw-u8[32]")
    closure_record = source_payload_closure.canonical_record()
    catalog_record = catalog.canonical_record()
    return closure_record, tuple(closure_identity), catalog_record


def validate_public_route_clarification_catalog_cached_v1(
        catalog: PublicRouteClarificationCatalogV1,
        source_payload_closure: PublicSourcePayloadClosureV1,
        cache: PublicRouteClarificationCatalogValidationCacheV1 | None,
        ) -> PublicRouteClarificationCatalogV1:
    """验证并复用 selector catalog 的派生编译结果。

    命中条件只由显式 closure/catalog canonical records 与 closure identity
    决定；没有 ``id``、``hash()`` 或容器顺序推断。未命中时完整调用原有
    fail-closed validator，任何异常原样向调用层传播。缓存本身不进入任何
    binding、state、turn 或 snapshot record，删除它不会改变协议语义。
    """
    if cache is None:
        return validate_public_route_clarification_catalog_v1(
            catalog,
            source_payload_closure,
        )
    if type(cache) is not PublicRouteClarificationCatalogValidationCacheV1:
        raise TypeError("route clarification catalog validation cache 类型错误")
    closure_record, closure_identity, catalog_record = _catalog_validation_key_v1(
        catalog,
        source_payload_closure,
    )
    entries = cache.entries
    if type(entries) is not tuple:
        raise PublicRouteClarificationCatalogError(
            "route clarification catalog validation cache entries 类型错误")
    for entry in entries:
        if type(entry) is not PublicRouteClarificationCatalogValidationEntryV1:
            raise PublicRouteClarificationCatalogError(
                "route clarification catalog validation cache entry 类型错误")
        if (entry.closure_record == closure_record
                and entry.closure_identity_u8 == closure_identity
                and entry.catalog_record == catalog_record):
            # The cached object is never a semantic source.  Returning the
            # current catalog after its explicit canonical key matched keeps
            # cache tampering from injecting a different object; a malformed
            # derived value is simply treated as a miss and rebuilt below.
            if type(entry.validated_catalog) is PublicRouteClarificationCatalogV1:
                return catalog

    # Miss path is intentionally the original full validator.  This preserves
    # all source/course/surface span checks and makes drift fail closed.
    rebuilt = validate_public_route_clarification_catalog_v1(
        catalog,
        source_payload_closure,
    )
    rebuilt_record = rebuilt.canonical_record()
    if rebuilt_record != catalog_record:
        raise PublicRouteClarificationCatalogError(
            "route clarification cached catalog readback 漂移")
    rebuilt_identity = tuple(rebuilt.catalog_identity_u8)
    if len(rebuilt_identity) != 32:
        raise PublicRouteClarificationCatalogError(
            "route clarification cached catalog identity 长度漂移")
    cache.entries = (
        PublicRouteClarificationCatalogValidationEntryV1(
            closure_record,
            closure_identity,
            catalog_record,
            rebuilt_identity,
            rebuilt,
        ),
    )
    return rebuilt


__all__ = [
    "PUBLIC_ROUTE_CLARIFICATION_CANDIDATE_IDENTITY_DOMAIN_V1",
    "PUBLIC_ROUTE_CLARIFICATION_CATALOG_IDENTITY_DOMAIN_V1",
    "PUBLIC_ROUTE_CLARIFICATION_CATALOG_RECORD_V1",
    "PUBLIC_ROUTE_CLARIFICATION_CATALOG_SCHEMA_V1",
    "PUBLIC_ROUTE_CLARIFICATION_COURSE_LOGICAL_KEY_V1",
    "PUBLIC_ROUTE_CLARIFICATION_COURSE_PARSER_IDENTITY_DOMAIN_V1",
    "PUBLIC_ROUTE_CLARIFICATION_COURSE_PARSER_RECORD_V1",
    "PUBLIC_ROUTE_CLARIFICATION_FORM_IDENTITY_DOMAIN_V1",
    "PUBLIC_ROUTE_CLARIFICATION_FORM_RECORD_V1",
    "PUBLIC_ROUTE_CLARIFICATION_LOGICAL_KEYS_V1",
    "PUBLIC_ROUTE_CLARIFICATION_OPTION_IDENTITY_DOMAIN_V1",
    "PUBLIC_ROUTE_CLARIFICATION_OPTION_RECORD_V1",
    "PUBLIC_ROUTE_CLARIFICATION_OUTPUT_READBACK_ACCEPTED_V1",
    "PUBLIC_ROUTE_CLARIFICATION_OUTPUT_READBACK_RECORD_V1",
    "PUBLIC_ROUTE_CLARIFICATION_ROUTE_IDENTITY_DOMAIN_V1",
    "PUBLIC_ROUTE_CLARIFICATION_ROUTE_IDENTITY_RECORD_V1",
    "PUBLIC_ROUTE_CLARIFICATION_SOURCE_IDENTITY_DOMAIN_V1",
    "PUBLIC_ROUTE_CLARIFICATION_SOURCE_KIND_COURSE_V1",
    "PUBLIC_ROUTE_CLARIFICATION_SOURCE_KIND_SURFACE_A_V1",
    "PUBLIC_ROUTE_CLARIFICATION_SOURCE_KIND_SURFACE_B_V1",
    "PUBLIC_ROUTE_CLARIFICATION_SOURCE_RECORD_V1",
    "PUBLIC_ROUTE_CLARIFICATION_SURFACE_A_LOGICAL_KEY_V1",
    "PUBLIC_ROUTE_CLARIFICATION_SURFACE_B_LOGICAL_KEY_V1",
    "PublicRouteClarificationCatalogError",
    "PublicRouteClarificationCatalogValidationCacheV1",
    "PublicRouteClarificationCatalogValidationEntryV1",
    "PublicRouteClarificationCatalogV1",
    "RouteClarificationFormV1",
    "RouteClarificationOptionV1",
    "RouteClarificationOutputReadbackV1",
    "RouteClarificationSourceV1",
    "candidate_identity_v1",
    "load_public_route_clarification_catalog_from_closure",
    "public_route_clarification_course_parser_identity_v1",
    "public_route_clarification_course_parser_record_v1",
    "route_identity_from_source_bound_resolution_v1",
    "route_identity_record_v1",
    "route_identity_v1",
    "route_clarification_output_readback_v1",
    "source_bound_route_candidate_identities_v1",
    "validate_public_route_clarification_catalog_v1",
    "validate_public_route_clarification_catalog_cached_v1",
]
