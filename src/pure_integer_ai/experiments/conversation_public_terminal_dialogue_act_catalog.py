"""DLG-RAW-13：公开终端覆盖/路由澄清话语行为的纯整数 catalog。

本模块只把固定 public payload closure 中的规范 JSONL 课程和两份独立 surface
witness 编译为有限整数/bytes record。它不读取物理路径、不检查用户文本、不决定
事实性 UNKNOWN/CLARIFY，也不写入会话状态。Python dataclass 只承载结构体；所有
可观察 identity、source、form 与输出均可由 record 重建。
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
    DLG_RAW_REJECT_LEXICAL_MISS,
    encode_utf8_v1,
)
from pure_integer_ai.experiments.ph2_dataset_core import (
    DatasetContractError,
    parse_canonical_json_bytes,
)


PUBLIC_TERMINAL_DIALOGUE_ACT_CATALOG_SCHEMA_V1 = 1
PUBLIC_TERMINAL_DIALOGUE_ACT_SOURCE_RECORD_V1 = 1
PUBLIC_TERMINAL_DIALOGUE_ACT_FORM_RECORD_V1 = 1
PUBLIC_TERMINAL_DIALOGUE_ACT_CATALOG_RECORD_V1 = 1
PUBLIC_TERMINAL_DIALOGUE_ACT_COURSE_PARSER_RECORD_V1 = 1
PUBLIC_TERMINAL_DIALOGUE_ACT_SOURCE_KIND_COURSE_V1 = 1
PUBLIC_TERMINAL_DIALOGUE_ACT_SOURCE_KIND_SURFACE_A_V1 = 2
PUBLIC_TERMINAL_DIALOGUE_ACT_SOURCE_KIND_SURFACE_B_V1 = 3
PUBLIC_TERMINAL_DIALOGUE_ACT_COVERAGE_UNSUPPORTED_V1 = 1
PUBLIC_TERMINAL_DIALOGUE_ACT_ROUTE_CLARIFICATION_V1 = 2
PUBLIC_TERMINAL_DIALOGUE_ACT_STATE_EFFECT_NONE_V1 = 0
PUBLIC_TERMINAL_DIALOGUE_ACT_CATALOG_IDENTITY_DOMAIN_V1 = (
    b"PURE-INTEGER-AI/DLG-RAW-13/TERMINAL-DIALOGUE-ACT-CATALOG/V1")
PUBLIC_TERMINAL_DIALOGUE_ACT_FORM_IDENTITY_DOMAIN_V1 = (
    b"PURE-INTEGER-AI/DLG-RAW-13/TERMINAL-DIALOGUE-ACT-FORM/V1")
PUBLIC_TERMINAL_DIALOGUE_ACT_SOURCE_IDENTITY_DOMAIN_V1 = (
    b"PURE-INTEGER-AI/DLG-RAW-13/TERMINAL-DIALOGUE-ACT-SOURCE/V1")
PUBLIC_TERMINAL_DIALOGUE_ACT_COURSE_PARSER_IDENTITY_DOMAIN_V1 = (
    b"PURE-INTEGER-AI/DLG-RAW-13/TERMINAL-DIALOGUE-ACT-COURSE-PARSER/V1")

PUBLIC_TERMINAL_DIALOGUE_ACT_COURSE_LOGICAL_KEY_V1 = (
    b"data/ph2/dlg_raw_public_terminal_dialogue_act_course_v1.jsonl.sample")
PUBLIC_TERMINAL_DIALOGUE_ACT_SURFACE_A_LOGICAL_KEY_V1 = (
    b"data/ph2/dlg_raw_public_terminal_dialogue_act_surface_v1_a.txt.sample")
PUBLIC_TERMINAL_DIALOGUE_ACT_SURFACE_B_LOGICAL_KEY_V1 = (
    b"data/ph2/dlg_raw_public_terminal_dialogue_act_surface_v1_b.txt.sample")
PUBLIC_TERMINAL_DIALOGUE_ACT_LOGICAL_KEYS_V1 = (
    PUBLIC_TERMINAL_DIALOGUE_ACT_COURSE_LOGICAL_KEY_V1,
    PUBLIC_TERMINAL_DIALOGUE_ACT_SURFACE_A_LOGICAL_KEY_V1,
    PUBLIC_TERMINAL_DIALOGUE_ACT_SURFACE_B_LOGICAL_KEY_V1,
)

_U64_EXCLUSIVE = 1 << 64
_COURSE_FIELDS = (
    "act_code",
    "base_result_code",
    "course_attribution",
    "form_id",
    "license_id",
    "output_max_bytes",
    "output_surface",
    "schema",
    "surface_a",
    "surface_b",
)
_SURFACE_FIELDS = ("attribution", "raw_sha256", "relative_path")
_LICENSE_ID = tuple(b"CC0-1.0")
_EXPECTED_FORM_MAPPING = (
    (DLG_RAW_REJECT_LEXICAL_MISS,
     PUBLIC_TERMINAL_DIALOGUE_ACT_COVERAGE_UNSUPPORTED_V1),
    (DLG_RAW_REJECT_LEXICAL_AMBIGUOUS,
     PUBLIC_TERMINAL_DIALOGUE_ACT_ROUTE_CLARIFICATION_V1),
)
_EXPECTED_SOURCE_KEY_BY_KIND = (
    (PUBLIC_TERMINAL_DIALOGUE_ACT_SOURCE_KIND_COURSE_V1,
     tuple(PUBLIC_TERMINAL_DIALOGUE_ACT_COURSE_LOGICAL_KEY_V1)),
    (PUBLIC_TERMINAL_DIALOGUE_ACT_SOURCE_KIND_SURFACE_A_V1,
     tuple(PUBLIC_TERMINAL_DIALOGUE_ACT_SURFACE_A_LOGICAL_KEY_V1)),
    (PUBLIC_TERMINAL_DIALOGUE_ACT_SOURCE_KIND_SURFACE_B_V1,
     tuple(PUBLIC_TERMINAL_DIALOGUE_ACT_SURFACE_B_LOGICAL_KEY_V1)),
)


# object-model: exception; interop=DLG-RAW-13
class PublicTerminalDialogueActCatalogError(ValueError):
    """终端话语行为课程、来源或整数 record 未能闭合。"""


def _u64(value: int, *, label: str, minimum: int = 0) -> int:
    """验证显式无符号范围，避免 Python 任意精度范围隐式成为协议。"""
    if (type(value) is not int or value < minimum
            or value >= _U64_EXCLUSIVE):
        raise PublicTerminalDialogueActCatalogError(
            f"{label} 必须是范围内的严格 u64")
    return value


def _u8_tuple(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """验证 immutable u8 vector，并拒绝 bool、list 和宿主缓冲对象。"""
    if (type(value) is not tuple
            or any(type(item) is not int or item < 0 or item > 255
                   for item in value)):
        raise PublicTerminalDialogueActCatalogError(
            f"{label} 必须是 0..255 严格整数 tuple")
    return value


def _pack(result: list[int], value: tuple[int, ...], *, label: str) -> None:
    """以显式 element count 追加非负整数子 record。"""
    if (type(value) is not tuple
            or any(type(item) is not int or item < 0 for item in value)):
        raise PublicTerminalDialogueActCatalogError(
            f"{label} 不是非负整数 record")
    _u64(len(value), label=f"{label} count")
    result.extend((len(value), *value))


def _identity(
        domain: bytes,
        record: tuple[int, ...],
        *,
        label: str,
        ) -> tuple[int, ...]:
    """以冻结 portable SHA-256 framing 形成 raw-u8 identity。"""
    try:
        result = tuple(portable_sha256_v1(domain, (record,)))
    except (PublicSourcePayloadProviderError, TypeError, ValueError) as error:
        raise PublicTerminalDialogueActCatalogError(
            f"{label} identity 无法形成") from error
    return _u8_tuple(result, label=f"{label} identity")


def terminal_dialogue_act_course_parser_record_v1() -> tuple[int, ...]:
    """冻结课程 parser 子集：UTF-8、LF JSONL、canonical JSON 和严格字段。"""
    return (
        PUBLIC_TERMINAL_DIALOGUE_ACT_COURSE_PARSER_RECORD_V1,
        1,  # strict UTF-8
        1,  # LF-only JSONL with one final LF
        1,  # canonical JSON object bytes and ASCII key order
        1,  # no duplicate/default/unknown fields after canonical readback
        1,  # integer-only numeric fields
        1,  # source/output must use explicit raw-u8 readback
    )


def terminal_dialogue_act_course_parser_identity_v1() -> tuple[int, ...]:
    """导出 parser contract 的 portable SHA-256 identity。"""
    return _identity(
        PUBLIC_TERMINAL_DIALOGUE_ACT_COURSE_PARSER_IDENTITY_DOMAIN_V1,
        terminal_dialogue_act_course_parser_record_v1(),
        label="terminal act course parser",
    )


def _ascii_bytes(value: Any, *, label: str) -> tuple[int, ...]:
    """把协议 id/路径降解为明确 ASCII bytes，不接收路径或空白变体。"""
    if (not isinstance(value, str) or not value
            or value[0] in " \t\r\n" or value[-1] in " \t\r\n"
            or any(ord(item) < 0x21 or ord(item) > 0x7E for item in value)):
        raise PublicTerminalDialogueActCatalogError(
            f"{label} 不是规范 ASCII transport")
    return tuple(ord(item) for item in value)


def _ascii_u8_tuple(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """验证已经降解的 ASCII transport，避免回转 Python 文本做校验。"""
    result = _u8_tuple(value, label=label)
    if (not result or any(item < 0x21 or item > 0x7E for item in result)):
        raise PublicTerminalDialogueActCatalogError(
            f"{label} 不是规范 ASCII transport")
    return result


def _display_u8(value: Any, *, label: str) -> tuple[int, ...]:
    """显式 UTF-8 编码课程 surface，不让宿主默认编码参与语义。"""
    if (not isinstance(value, str) or not value
            or value[0] in " \t\r\n" or value[-1] in " \t\r\n"):
        raise PublicTerminalDialogueActCatalogError(
            f"{label} 必须是无首尾空白的非空文本")
    scalars = tuple(ord(item) for item in value)
    try:
        result = encode_utf8_v1(scalars)
    except (TypeError, ValueError) as error:
        raise PublicTerminalDialogueActCatalogError(
            f"{label} 含非法 Unicode scalar") from error
    return _u8_tuple(result, label=label)


def _strict_int(value: Any, *, label: str, minimum: int = 0) -> int:
    """读取课程整数，拒绝 bool、float 和隐式数值转换。"""
    if type(value) is not int or value < minimum:
        raise PublicTerminalDialogueActCatalogError(
            f"{label} 必须是不小于 {minimum} 的严格整数")
    return value


def _exact_fields(value: Any, fields: tuple[str, ...], *, label: str) -> dict[str, Any]:
    """拒绝缺失/额外字段，不让 dict 默认项或迭代顺序决定协议。"""
    if not isinstance(value, dict) or len(value) != len(fields):
        raise PublicTerminalDialogueActCatalogError(f"{label} 字段数量漂移")
    for field in fields:
        if field not in value:
            raise PublicTerminalDialogueActCatalogError(
                f"{label} 缺少字段 {field}")
    for field in value:
        if field not in fields:
            raise PublicTerminalDialogueActCatalogError(
                f"{label} 含未注册字段 {field}")
    return value


def _hex_digest(value: Any, *, label: str) -> tuple[int, ...]:
    """解析固定小写 SHA-256 hex，输出 raw-u8 tuple。"""
    if (not isinstance(value, str) or len(value) != 64
            or any(item not in "0123456789abcdef" for item in value)):
        raise PublicTerminalDialogueActCatalogError(
            f"{label} 必须是固定小写 SHA-256 hex")
    try:
        result = tuple(bytes.fromhex(value))
    except ValueError as error:
        raise PublicTerminalDialogueActCatalogError(
            f"{label} hex 损坏") from error
    return _u8_tuple(result, label=label)


def _unique_span(
        payload: bytes,
        needle: tuple[int, ...],
        *,
        label: str,
        ) -> tuple[int, int]:
    """用显式逐 byte 比较定位唯一 surface，不依赖文本/正则搜索。"""
    target = _u8_tuple(needle, label=label)
    if type(payload) is not bytes or not target or len(target) > len(payload):
        raise PublicTerminalDialogueActCatalogError(
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
                raise PublicTerminalDialogueActCatalogError(
                    f"{label} 在 source 中不唯一")
            found = start
    if found < 0:
        raise PublicTerminalDialogueActCatalogError(
            f"{label} 在 source 中缺失")
    return found, found + len(target)


def _payload(
        closure: PublicSourcePayloadClosureV1,
        logical_key: bytes,
        *,
        expected_digest: tuple[int, ...] | None,
        label: str,
        ) -> tuple[bytes, tuple[int, ...]]:
    """从已冻结 closure 读取并重核一份 raw payload 和 SHA-256。"""
    if type(closure) is not PublicSourcePayloadClosureV1:
        raise TypeError("terminal dialogue act source payload closure 类型错误")
    try:
        record = closure.record_for(logical_key)
        raw = closure.payload_for(logical_key)
    except PublicSourcePayloadProviderError as error:
        raise PublicTerminalDialogueActCatalogError(
            f"{label} 不在 public payload closure") from error
    digest = _u8_tuple(tuple(public_source_payload_sha256_v1(raw)),
                       label=f"{label} SHA-256")
    if (record.logical_key != logical_key or record.raw_payload != raw
            or record.payload_length != len(raw)
            or tuple(record.raw_sha256) != digest
            or (expected_digest is not None and digest != expected_digest)):
        raise PublicTerminalDialogueActCatalogError(
            f"{label} payload/source digest 漂移")
    return raw, digest


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
    """形成不含身份的来源本体 record，供 identity 和 readback 共同使用。"""
    result = [PUBLIC_TERMINAL_DIALOGUE_ACT_SOURCE_RECORD_V1, source_kind]
    for label, value in (
            ("source logical key", logical_key_u8),
            ("source raw SHA-256", raw_sha256_u8),
            ("source span bytes", span_u8),
            ("source license", license_id_u8),
            ("source attribution", attribution_u8)):
        _pack(result, value, label=label)
    result.extend((span_start, span_end))
    return tuple(result)


# object-model: value; representation=struct; interop=DLG-RAW-13
@dataclass(frozen=True, slots=True)
class TerminalDialogueActSourceV1:
    """课程或 witness 内唯一 raw span 的来源化整数 record。"""

    source_kind: int
    logical_key_u8: tuple[int, ...]
    raw_sha256_u8: tuple[int, ...]
    span_start: int
    span_end: int
    span_u8: tuple[int, ...]
    license_id_u8: tuple[int, ...]
    attribution_u8: tuple[int, ...]

    def __post_init__(self) -> None:
        """冻结 source kind、raw SHA、span 与公开许可 transport。"""
        if self.source_kind not in (
                PUBLIC_TERMINAL_DIALOGUE_ACT_SOURCE_KIND_COURSE_V1,
                PUBLIC_TERMINAL_DIALOGUE_ACT_SOURCE_KIND_SURFACE_A_V1,
                PUBLIC_TERMINAL_DIALOGUE_ACT_SOURCE_KIND_SURFACE_B_V1):
            raise PublicTerminalDialogueActCatalogError("terminal act source kind 未注册")
        logical_key = _ascii_u8_tuple(
            self.logical_key_u8,
            label="terminal act source logical key",
        )
        expected_key = tuple(item[1] for item in _EXPECTED_SOURCE_KEY_BY_KIND
                             if item[0] == self.source_kind)
        if len(expected_key) != 1 or logical_key != expected_key[0]:
            raise PublicTerminalDialogueActCatalogError(
                "terminal act source kind/logical key binding 漂移")
        _u8_tuple(self.raw_sha256_u8, label="terminal act source SHA-256")
        if len(self.raw_sha256_u8) != 32:
            raise PublicTerminalDialogueActCatalogError(
                "terminal act source SHA-256 长度漂移")
        _u64(self.span_start, label="terminal act source span start")
        _u64(self.span_end, label="terminal act source span end", minimum=1)
        span = _u8_tuple(self.span_u8, label="terminal act source span bytes")
        if (not span or self.span_end - self.span_start != len(span)
                or self.license_id_u8 != _LICENSE_ID):
            raise PublicTerminalDialogueActCatalogError(
                "terminal act source span 或许可漂移")
        _u8_tuple(self.attribution_u8, label="terminal act source attribution")
        if not self.attribution_u8:
            raise PublicTerminalDialogueActCatalogError(
                "terminal act source attribution 不得为空")

    def base_record(self) -> tuple[int, ...]:
        """返回用于 source identity 的完整来源本体，不混入对象 identity。"""
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
        """从来源本体导出 portable SHA-256 identity。"""
        return _identity(
            PUBLIC_TERMINAL_DIALOGUE_ACT_SOURCE_IDENTITY_DOMAIN_V1,
            self.base_record(),
            label="terminal act source",
        )

    def canonical_record(self) -> tuple[int, ...]:
        """导出包含可重算 identity 的有序 source record。"""
        result = list(self.base_record())
        _pack(result, self.source_identity_u8, label="source identity")
        return tuple(result)


def _form_base_record(
        form_id_u8: tuple[int, ...],
        act_code: int,
        base_result_code: int,
        output_max_bytes: int,
        output_u8: tuple[int, ...],
        course_source: TerminalDialogueActSourceV1,
        surface_sources: tuple[TerminalDialogueActSourceV1, ...],
        ) -> tuple[int, ...]:
    """形成不含 form identity 的课程 form 本体 record。"""
    result = [
        PUBLIC_TERMINAL_DIALOGUE_ACT_FORM_RECORD_V1,
        act_code,
        base_result_code,
        output_max_bytes,
    ]
    _pack(result, form_id_u8, label="terminal act form id")
    _pack(result, output_u8, label="terminal act output")
    _pack(result, course_source.canonical_record(), label="terminal act course source")
    result.append(len(surface_sources))
    for ordinal, source in enumerate(surface_sources, start=1):
        _pack(result, source.canonical_record(),
              label=f"terminal act surface source {ordinal}")
    return tuple(result)


# object-model: value; representation=struct; interop=DLG-RAW-13
@dataclass(frozen=True, slots=True)
class TerminalDialogueActFormV1:
    """一个已学习的 raw reject code 到 source-bound terminal act 的映射。"""

    form_id_u8: tuple[int, ...]
    act_code: int
    base_result_code: int
    output_max_bytes: int
    output_u8: tuple[int, ...]
    course_source: TerminalDialogueActSourceV1
    surface_sources: tuple[TerminalDialogueActSourceV1, ...]

    def __post_init__(self) -> None:
        """验证闭合 code mapping、来源顺序和 exact output witness。"""
        _ascii_u8_tuple(self.form_id_u8, label="terminal act form id")
        mapping = tuple(
            pair for pair in _EXPECTED_FORM_MAPPING
            if pair[0] == self.base_result_code)
        if len(mapping) != 1 or mapping[0][1] != self.act_code:
            raise PublicTerminalDialogueActCatalogError(
                "terminal act form code mapping 未注册")
        _u64(self.output_max_bytes, label="terminal act output budget", minimum=1)
        output = _u8_tuple(self.output_u8, label="terminal act output")
        if not output or len(output) > self.output_max_bytes:
            raise PublicTerminalDialogueActCatalogError(
                "terminal act output 超出课程预算")
        if type(self.course_source) is not TerminalDialogueActSourceV1:
            raise TypeError("terminal act course source 类型错误")
        if (self.course_source.source_kind
                != PUBLIC_TERMINAL_DIALOGUE_ACT_SOURCE_KIND_COURSE_V1):
            raise PublicTerminalDialogueActCatalogError(
                "terminal act form 缺 course source")
        if (type(self.surface_sources) is not tuple
                or len(self.surface_sources) != 2
                or any(type(item) is not TerminalDialogueActSourceV1
                       for item in self.surface_sources)):
            raise PublicTerminalDialogueActCatalogError(
                "terminal act form 必须有两个 surface source")
        expected_kinds = (
            PUBLIC_TERMINAL_DIALOGUE_ACT_SOURCE_KIND_SURFACE_A_V1,
            PUBLIC_TERMINAL_DIALOGUE_ACT_SOURCE_KIND_SURFACE_B_V1,
        )
        if tuple(item.source_kind for item in self.surface_sources) != expected_kinds:
            raise PublicTerminalDialogueActCatalogError(
                "terminal act surface source 顺序或 kind 漂移")
        for source in self.surface_sources:
            if source.span_u8 != output:
                raise PublicTerminalDialogueActCatalogError(
                    "terminal act witness 未精确见证 output")

    def base_record(self) -> tuple[int, ...]:
        """返回 form identity 的完整本体 record。"""
        return _form_base_record(
            self.form_id_u8,
            self.act_code,
            self.base_result_code,
            self.output_max_bytes,
            self.output_u8,
            self.course_source,
            self.surface_sources,
        )

    @property
    def form_identity_u8(self) -> tuple[int, ...]:
        """从 form 本体导出 portable SHA-256 identity。"""
        return _identity(
            PUBLIC_TERMINAL_DIALOGUE_ACT_FORM_IDENTITY_DOMAIN_V1,
            self.base_record(),
            label="terminal act form",
        )

    def canonical_record(self) -> tuple[int, ...]:
        """导出 form 本体与其可重算 identity。"""
        result = list(self.base_record())
        _pack(result, self.form_identity_u8, label="terminal act form identity")
        return tuple(result)


# object-model: value; representation=struct; interop=DLG-RAW-13
@dataclass(frozen=True, slots=True)
class TerminalDialogueActCatalogV1:
    """恰有两条 terminal meta-act form 的来源闭合 catalog。"""

    source_payload_closure_identity_u8: tuple[int, ...]
    forms: tuple[TerminalDialogueActFormV1, ...]

    def __post_init__(self) -> None:
        """使 closure、form 数量、顺序与 code mapping 在构造期 fail closed。"""
        closure_identity = _u8_tuple(
            self.source_payload_closure_identity_u8,
            label="terminal act closure identity",
        )
        if len(closure_identity) != 32:
            raise PublicTerminalDialogueActCatalogError(
                "terminal act closure identity 长度漂移")
        if (type(self.forms) is not tuple or len(self.forms) != 2
                or any(type(item) is not TerminalDialogueActFormV1
                       for item in self.forms)):
            raise PublicTerminalDialogueActCatalogError(
                "terminal act catalog form 数量或类型漂移")
        expected_codes = tuple(item[0] for item in _EXPECTED_FORM_MAPPING)
        if tuple(item.base_result_code for item in self.forms) != expected_codes:
            raise PublicTerminalDialogueActCatalogError(
                "terminal act catalog form 顺序或 base code 漂移")
        expected_acts = tuple(item[1] for item in _EXPECTED_FORM_MAPPING)
        if tuple(item.act_code for item in self.forms) != expected_acts:
            raise PublicTerminalDialogueActCatalogError(
                "terminal act catalog act code 漂移")

    def canonical_record(self) -> tuple[int, ...]:
        """导出不含 Python object identity 的完整 catalog record。"""
        result = [PUBLIC_TERMINAL_DIALOGUE_ACT_CATALOG_RECORD_V1,
                  PUBLIC_TERMINAL_DIALOGUE_ACT_CATALOG_SCHEMA_V1]
        _pack(result, terminal_dialogue_act_course_parser_record_v1(),
              label="terminal act course parser")
        _pack(result, terminal_dialogue_act_course_parser_identity_v1(),
              label="terminal act course parser identity")
        _pack(result, self.source_payload_closure_identity_u8,
              label="terminal act closure identity")
        result.append(len(self.forms))
        for ordinal, form in enumerate(self.forms, start=1):
            _pack(result, form.canonical_record(),
                  label=f"terminal act form {ordinal}")
        return tuple(result)

    @property
    def catalog_identity_u8(self) -> tuple[int, ...]:
        """从完整 catalog record 导出 portable SHA-256 identity。"""
        return _identity(
            PUBLIC_TERMINAL_DIALOGUE_ACT_CATALOG_IDENTITY_DOMAIN_V1,
            self.canonical_record(),
            label="terminal act catalog",
        )

    def form_for_base_result_code(
            self,
            base_result_code: int,
            ) -> TerminalDialogueActFormV1 | None:
        """按冻结 result code 线性选择 form；不以 host map 或文本作路由。"""
        if type(base_result_code) is not int:
            raise TypeError("terminal act base result code 类型错误")
        selected = None
        for form in self.forms:
            if form.base_result_code == base_result_code:
                if selected is not None:
                    raise PublicTerminalDialogueActCatalogError(
                        "terminal act base result code 不唯一")
                selected = form
        return selected


def _canonical_jsonl_lines(payload: bytes) -> tuple[tuple[int, bytes, dict[str, Any]], ...]:
    """按 raw byte 行严格读取 canonical JSONL，拒绝 CR、空行和尾部变体。"""
    if (type(payload) is not bytes or not payload or not payload.endswith(b"\n")
            or b"\r" in payload):
        raise PublicTerminalDialogueActCatalogError(
            "terminal act course 不是规范 LF JSONL")
    result: list[tuple[int, bytes, dict[str, Any]]] = []
    cursor = 0
    for line in payload[:-1].split(b"\n"):
        if not line:
            raise PublicTerminalDialogueActCatalogError(
                "terminal act course 含空 JSONL line")
        try:
            parsed = parse_canonical_json_bytes(line, require_object=True)
        except DatasetContractError as error:
            raise PublicTerminalDialogueActCatalogError(
                "terminal act course JSON line 不规范") from error
        result.append((cursor, line, parsed))
        cursor += len(line) + 1
    return tuple(result)


def _surface_descriptor(
        raw: Any,
        *,
        expected_logical_key: bytes,
        closure: PublicSourcePayloadClosureV1,
        output_u8: tuple[int, ...],
        source_kind: int,
        license_id_u8: tuple[int, ...],
        label: str,
        ) -> TerminalDialogueActSourceV1:
    """从课程 descriptor 验证一个独立 witness 的 raw source/span。"""
    descriptor = _exact_fields(raw, _SURFACE_FIELDS, label=label)
    logical_key = tuple(_ascii_bytes(
        descriptor["relative_path"], label=f"{label}.relative_path"))
    if bytes(logical_key) != expected_logical_key:
        raise PublicTerminalDialogueActCatalogError(
            f"{label}.relative_path 未绑定冻结 witness")
    expected_digest = _hex_digest(descriptor["raw_sha256"],
                                  label=f"{label}.raw_sha256")
    raw_payload, digest = _payload(
        closure,
        expected_logical_key,
        expected_digest=expected_digest,
        label=label,
    )
    attribution = _display_u8(descriptor["attribution"],
                              label=f"{label}.attribution")
    start, end = _unique_span(raw_payload, output_u8, label=label)
    return TerminalDialogueActSourceV1(
        source_kind,
        logical_key,
        digest,
        start,
        end,
        output_u8,
        license_id_u8,
        attribution,
    )


def _form_from_course_line(
        line_offset: int,
        line: bytes,
        raw: dict[str, Any],
        *,
        closure: PublicSourcePayloadClosureV1,
        course_digest: tuple[int, ...],
        ordinal: int,
        ) -> TerminalDialogueActFormV1:
    """将一条规范 course line 编译为完全来源绑定的 terminal act form。"""
    record = _exact_fields(raw, _COURSE_FIELDS,
                           label=f"terminal act course line {ordinal}")
    if _strict_int(record["schema"], label="terminal act course schema") != (
            PUBLIC_TERMINAL_DIALOGUE_ACT_CATALOG_SCHEMA_V1):
        raise PublicTerminalDialogueActCatalogError(
            "terminal act course schema 未注册")
    base_code = _strict_int(record["base_result_code"],
                            label="terminal act base result code")
    act_code = _strict_int(record["act_code"], label="terminal act code")
    mapping = tuple(pair for pair in _EXPECTED_FORM_MAPPING
                    if pair[0] == base_code)
    if len(mapping) != 1 or mapping[0][1] != act_code:
        raise PublicTerminalDialogueActCatalogError(
            "terminal act course code mapping 未注册")
    form_id = _ascii_bytes(record["form_id"], label="terminal act form id")
    license_id = _ascii_bytes(record["license_id"],
                              label="terminal act license id")
    if license_id != _LICENSE_ID:
        raise PublicTerminalDialogueActCatalogError(
            "terminal act course license 必须是 CC0-1.0")
    output = _display_u8(record["output_surface"],
                         label="terminal act output surface")
    output_max = _strict_int(record["output_max_bytes"],
                             label="terminal act output max bytes", minimum=1)
    if len(output) > output_max:
        raise PublicTerminalDialogueActCatalogError(
            "terminal act course output 超出显式预算")
    course_attribution = _display_u8(record["course_attribution"],
                                     label="terminal act course attribution")
    output_start, _output_end = _unique_span(
        line, output, label="terminal act course output")
    course_source = TerminalDialogueActSourceV1(
        PUBLIC_TERMINAL_DIALOGUE_ACT_SOURCE_KIND_COURSE_V1,
        tuple(PUBLIC_TERMINAL_DIALOGUE_ACT_COURSE_LOGICAL_KEY_V1),
        course_digest,
        line_offset,
        line_offset + len(line),
        tuple(line),
        license_id,
        course_attribution,
    )
    # 输出 span 必须同时由 course line 明确携带，避免 course source 只绑定无关元数据。
    if tuple(line[output_start:output_start + len(output)]) != output:
        raise PublicTerminalDialogueActCatalogError(
            "terminal act course output span readback 漂移")
    surface_a = _surface_descriptor(
        record["surface_a"],
        expected_logical_key=PUBLIC_TERMINAL_DIALOGUE_ACT_SURFACE_A_LOGICAL_KEY_V1,
        closure=closure,
        output_u8=output,
        source_kind=PUBLIC_TERMINAL_DIALOGUE_ACT_SOURCE_KIND_SURFACE_A_V1,
        license_id_u8=license_id,
        label="terminal act surface_a",
    )
    surface_b = _surface_descriptor(
        record["surface_b"],
        expected_logical_key=PUBLIC_TERMINAL_DIALOGUE_ACT_SURFACE_B_LOGICAL_KEY_V1,
        closure=closure,
        output_u8=output,
        source_kind=PUBLIC_TERMINAL_DIALOGUE_ACT_SOURCE_KIND_SURFACE_B_V1,
        license_id_u8=license_id,
        label="terminal act surface_b",
    )
    return TerminalDialogueActFormV1(
        form_id,
        act_code,
        base_code,
        output_max,
        output,
        course_source,
        (surface_a, surface_b),
    )


def load_public_terminal_dialogue_act_catalog_from_closure(
        source_payload_closure: PublicSourcePayloadClosureV1,
        ) -> TerminalDialogueActCatalogV1:
    """从固定 closure 完整重建两条 terminal meta-act form。"""
    if type(source_payload_closure) is not PublicSourcePayloadClosureV1:
        raise TypeError("terminal act catalog source payload closure 类型错误")
    course_payload, course_digest = _payload(
        source_payload_closure,
        PUBLIC_TERMINAL_DIALOGUE_ACT_COURSE_LOGICAL_KEY_V1,
        expected_digest=None,
        label="terminal act course",
    )
    lines = _canonical_jsonl_lines(course_payload)
    if len(lines) != len(_EXPECTED_FORM_MAPPING):
        raise PublicTerminalDialogueActCatalogError(
            "terminal act course record 数量漂移")
    forms: list[TerminalDialogueActFormV1] = []
    for ordinal, (offset, line, raw) in enumerate(lines, start=1):
        forms.append(_form_from_course_line(
            offset,
            line,
            raw,
            closure=source_payload_closure,
            course_digest=course_digest,
            ordinal=ordinal,
        ))
    return TerminalDialogueActCatalogV1(
        tuple(source_payload_closure.closure_identity),
        tuple(forms),
    )


def validate_public_terminal_dialogue_act_catalog_v1(
        catalog: TerminalDialogueActCatalogV1,
        source_payload_closure: PublicSourcePayloadClosureV1,
        ) -> TerminalDialogueActCatalogV1:
    """以当前 closure 重建 catalog，拒绝内存篡改或 foreign course binding。"""
    if type(catalog) is not TerminalDialogueActCatalogV1:
        raise TypeError("terminal act catalog 类型错误")
    rebuilt = load_public_terminal_dialogue_act_catalog_from_closure(
        source_payload_closure)
    if (catalog.canonical_record() != rebuilt.canonical_record()
            or catalog.catalog_identity_u8 != rebuilt.catalog_identity_u8):
        raise PublicTerminalDialogueActCatalogError(
            "terminal act catalog 未绑定当前 public closure")
    return rebuilt


__all__ = [
    "PUBLIC_TERMINAL_DIALOGUE_ACT_CATALOG_IDENTITY_DOMAIN_V1",
    "PUBLIC_TERMINAL_DIALOGUE_ACT_CATALOG_RECORD_V1",
    "PUBLIC_TERMINAL_DIALOGUE_ACT_CATALOG_SCHEMA_V1",
    "PUBLIC_TERMINAL_DIALOGUE_ACT_COURSE_PARSER_IDENTITY_DOMAIN_V1",
    "PUBLIC_TERMINAL_DIALOGUE_ACT_COURSE_PARSER_RECORD_V1",
    "PUBLIC_TERMINAL_DIALOGUE_ACT_COURSE_LOGICAL_KEY_V1",
    "PUBLIC_TERMINAL_DIALOGUE_ACT_COVERAGE_UNSUPPORTED_V1",
    "PUBLIC_TERMINAL_DIALOGUE_ACT_FORM_IDENTITY_DOMAIN_V1",
    "PUBLIC_TERMINAL_DIALOGUE_ACT_FORM_RECORD_V1",
    "PUBLIC_TERMINAL_DIALOGUE_ACT_LOGICAL_KEYS_V1",
    "PUBLIC_TERMINAL_DIALOGUE_ACT_ROUTE_CLARIFICATION_V1",
    "PUBLIC_TERMINAL_DIALOGUE_ACT_SOURCE_IDENTITY_DOMAIN_V1",
    "PUBLIC_TERMINAL_DIALOGUE_ACT_SOURCE_KIND_COURSE_V1",
    "PUBLIC_TERMINAL_DIALOGUE_ACT_SOURCE_KIND_SURFACE_A_V1",
    "PUBLIC_TERMINAL_DIALOGUE_ACT_SOURCE_KIND_SURFACE_B_V1",
    "PUBLIC_TERMINAL_DIALOGUE_ACT_SOURCE_RECORD_V1",
    "PUBLIC_TERMINAL_DIALOGUE_ACT_STATE_EFFECT_NONE_V1",
    "PublicTerminalDialogueActCatalogError",
    "TerminalDialogueActCatalogV1",
    "TerminalDialogueActFormV1",
    "TerminalDialogueActSourceV1",
    "load_public_terminal_dialogue_act_catalog_from_closure",
    "terminal_dialogue_act_course_parser_identity_v1",
    "terminal_dialogue_act_course_parser_record_v1",
    "validate_public_terminal_dialogue_act_catalog_v1",
]
