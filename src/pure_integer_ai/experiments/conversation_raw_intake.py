"""DLG-RAW-00 的原始字节到规范整数 intake record 边界。

本模块的核心入口只接收 ``tuple[int, ...]``，不接受 Python 文本、locale、
JSON 或终端状态。``bytes`` 适配器只把宿主字节复制为 ``0..255`` 整数序列；
UTF-8 v1、行 framing、结果码和 record 布局均由这里的显式整数转移定义。
"""
from __future__ import annotations

from dataclasses import dataclass


DLG_RAW_RECORD_V1 = 1
UTF8_STRICT_V1 = 1
DLG_RAW_MAX_INPUT_BYTES = 4096

DLG_RAW_ACCEPT = 0
DLG_RAW_REJECT_INPUT_BUDGET = 1
DLG_RAW_REJECT_BOM = 2
DLG_RAW_REJECT_LINE_FRAMING = 3
DLG_RAW_REJECT_EMPTY = 4
DLG_RAW_REJECT_UTF8 = 5
DLG_RAW_REJECT_CONTROL = 6
DLG_RAW_REJECT_LEXICAL_MISS = 7
DLG_RAW_REJECT_LEXICAL_AMBIGUOUS = 8
DLG_RAW_REJECT_CONSTRUCTION_MISS = 9
DLG_RAW_REJECT_CONTEXT = 10
DLG_RAW_REJECT_RUNTIME = 11
DLG_RAW_REJECT_OUTPUT_BUDGET = 12
DLG_RAW_REJECT_SOURCE_CONFLICT = 13
DLG_RAW_REJECT_REFERENCE_AMBIGUOUS = 14

_RESULT_CODES = frozenset({
    DLG_RAW_ACCEPT,
    DLG_RAW_REJECT_INPUT_BUDGET,
    DLG_RAW_REJECT_BOM,
    DLG_RAW_REJECT_LINE_FRAMING,
    DLG_RAW_REJECT_EMPTY,
    DLG_RAW_REJECT_UTF8,
    DLG_RAW_REJECT_CONTROL,
    DLG_RAW_REJECT_LEXICAL_MISS,
    DLG_RAW_REJECT_LEXICAL_AMBIGUOUS,
    DLG_RAW_REJECT_CONSTRUCTION_MISS,
    DLG_RAW_REJECT_CONTEXT,
    DLG_RAW_REJECT_RUNTIME,
    DLG_RAW_REJECT_OUTPUT_BUDGET,
    DLG_RAW_REJECT_SOURCE_CONFLICT,
    DLG_RAW_REJECT_REFERENCE_AMBIGUOUS,
})
_UTF8_BOM = (0xEF, 0xBB, 0xBF)


# object-model: exception; interop=host-api-precondition
class ConversationRawIntakeError(ValueError):
    """宿主传入的值不是可迁移 byte vector 或 canonical record。"""


def _byte_vector(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """核验宿主已交付的有限 ``0..255`` 整数 byte sequence。"""
    if not isinstance(value, tuple):
        raise TypeError(f"{label} 必须是整数 tuple")
    if any(type(item) is not int or item < 0 or item > 255 for item in value):
        raise ConversationRawIntakeError(f"{label} 必须只含 0..255 严格整数")
    return value


def _integer_sequence(value: tuple[int, ...], *, label: str) -> tuple[int, ...]:
    """核验可携带 stable key 的有限严格整数序列。"""
    if not isinstance(value, tuple):
        raise TypeError(f"{label} 必须是整数 tuple")
    if any(type(item) is not int for item in value):
        raise ConversationRawIntakeError(f"{label} 必须只含严格整数")
    return value


def _pack(result: list[int], value: tuple[int, ...]) -> None:
    """按显式长度前缀写入一个变长整数段。"""
    result.extend((len(value), *value))


def _split_terminal_line(raw: tuple[int, ...]) -> tuple[int, ...] | None:
    """剥离唯一允许的末尾 LF/CRLF，并拒绝所有其余行控制 byte。"""
    body = raw
    if len(body) >= 2 and body[-2:] == (0x0D, 0x0A):
        body = body[:-2]
    elif body and body[-1] == 0x0A:
        body = body[:-1]
    if 0x0A in body or 0x0D in body:
        return None
    return body


def _continuation(value: int) -> bool:
    """判断一个 byte 是否为 UTF-8 continuation byte。"""
    return 0x80 <= value <= 0xBF


def decode_utf8_v1(utf8_bytes: tuple[int, ...]) -> tuple[int, ...] | None:
    """以显式整数状态机严格解码最短 UTF-8，不调用宿主文本解码器。"""
    body = _byte_vector(utf8_bytes, label="DLG-RAW UTF-8 input")
    result: list[int] = []
    cursor = 0
    length = len(body)
    while cursor < length:
        first = body[cursor]
        if first <= 0x7F:
            result.append(first)
            cursor += 1
            continue
        if 0xC2 <= first <= 0xDF:
            if cursor + 1 >= length or not _continuation(body[cursor + 1]):
                return None
            result.append(((first & 0x1F) << 6) | (body[cursor + 1] & 0x3F))
            cursor += 2
            continue
        if first == 0xE0:
            if (cursor + 2 >= length or not 0xA0 <= body[cursor + 1] <= 0xBF
                    or not _continuation(body[cursor + 2])):
                return None
        elif 0xE1 <= first <= 0xEC or 0xEE <= first <= 0xEF:
            if (cursor + 2 >= length or not _continuation(body[cursor + 1])
                    or not _continuation(body[cursor + 2])):
                return None
        elif first == 0xED:
            if (cursor + 2 >= length or not 0x80 <= body[cursor + 1] <= 0x9F
                    or not _continuation(body[cursor + 2])):
                return None
        else:
            if first == 0xF0:
                valid = (
                    cursor + 3 < length
                    and 0x90 <= body[cursor + 1] <= 0xBF
                    and _continuation(body[cursor + 2])
                    and _continuation(body[cursor + 3])
                )
            elif 0xF1 <= first <= 0xF3:
                valid = (
                    cursor + 3 < length
                    and _continuation(body[cursor + 1])
                    and _continuation(body[cursor + 2])
                    and _continuation(body[cursor + 3])
                )
            elif first == 0xF4:
                valid = (
                    cursor + 3 < length
                    and 0x80 <= body[cursor + 1] <= 0x8F
                    and _continuation(body[cursor + 2])
                    and _continuation(body[cursor + 3])
                )
            else:
                return None
            if not valid:
                return None
            result.append(
                ((first & 0x07) << 18)
                | ((body[cursor + 1] & 0x3F) << 12)
                | ((body[cursor + 2] & 0x3F) << 6)
                | (body[cursor + 3] & 0x3F)
            )
            cursor += 4
            continue
        result.append(
            ((first & 0x0F) << 12)
            | ((body[cursor + 1] & 0x3F) << 6)
            | (body[cursor + 2] & 0x3F)
        )
        cursor += 3
    return tuple(result)


def _has_control_scalars(scalars: tuple[int, ...]) -> bool:
    """识别已成功解码却不允许进入语言层的 C0/DEL scalar。"""
    return any(value <= 0x1F or value == 0x7F for value in scalars)


def encode_utf8_v1(unicode_scalars: tuple[int, ...]) -> tuple[int, ...]:
    """以严格 UTF-8 v1 把 Unicode scalar 序列编码为规范 byte vector。"""
    scalars = _integer_sequence(unicode_scalars, label="DLG-RAW unicode scalars")
    if any(value < 0 or value > 0x10FFFF or 0xD800 <= value <= 0xDFFF
           for value in scalars):
        raise ConversationRawIntakeError("DLG-RAW unicode scalar 非法")
    result: list[int] = []
    for value in scalars:
        if value <= 0x7F:
            result.append(value)
        elif value <= 0x7FF:
            result.extend((
                0xC0 | (value >> 6),
                0x80 | (value & 0x3F),
            ))
        elif value <= 0xFFFF:
            result.extend((
                0xE0 | (value >> 12),
                0x80 | ((value >> 6) & 0x3F),
                0x80 | (value & 0x3F),
            ))
        else:
            result.extend((
                0xF0 | (value >> 18),
                0x80 | ((value >> 12) & 0x3F),
                0x80 | ((value >> 6) & 0x3F),
                0x80 | (value & 0x3F),
            ))
    return tuple(result)


# object-model: value; representation=struct; interop=DLG-RAW-00
@dataclass(frozen=True, slots=True)
class ConversationRawIntake:
    """一条 raw bytes 的完整可迁移 intake record，尚未包含词义或会话状态。"""

    result_code: int
    raw_input_bytes: tuple[int, ...]
    canonical_body_bytes: tuple[int, ...] = ()
    unicode_scalars: tuple[int, ...] = ()
    typed_record: tuple[int, ...] = ()
    output_bytes: tuple[int, ...] = ()
    state_delta: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """核验 record 字段、结果码和 RAW-00 无副作用不变量。"""
        if type(self.result_code) is not int or self.result_code not in _RESULT_CODES:
            raise ConversationRawIntakeError("DLG-RAW result code 未注册")
        raw = _byte_vector(self.raw_input_bytes, label="DLG-RAW raw input")
        body = _byte_vector(
            self.canonical_body_bytes, label="DLG-RAW canonical body")
        scalars = _integer_sequence(
            self.unicode_scalars, label="DLG-RAW unicode scalars")
        if any(value < 0 or value > 0x10FFFF or 0xD800 <= value <= 0xDFFF
               for value in scalars):
            raise ConversationRawIntakeError("DLG-RAW unicode scalar 非法")
        typed = _integer_sequence(self.typed_record, label="DLG-RAW typed record")
        output = _byte_vector(self.output_bytes, label="DLG-RAW output")
        delta = _integer_sequence(self.state_delta, label="DLG-RAW state delta")
        if len(raw) > DLG_RAW_MAX_INPUT_BYTES:
            if self.result_code != DLG_RAW_REJECT_INPUT_BUDGET:
                raise ConversationRawIntakeError("超预算 raw input 必须按 budget 拒绝")
        if self.result_code <= DLG_RAW_REJECT_CONTROL:
            if typed or output or delta:
                raise ConversationRawIntakeError("RAW-00 不得携带 typed/output/state")
        object.__setattr__(self, "raw_input_bytes", raw)
        object.__setattr__(self, "canonical_body_bytes", body)
        object.__setattr__(self, "unicode_scalars", scalars)
        object.__setattr__(self, "typed_record", typed)
        object.__setattr__(self, "output_bytes", output)
        object.__setattr__(self, "state_delta", delta)

    @property
    def accepted(self) -> bool:
        """仅表示 bytes/UTF-8 边界通过，不表示语言理解成功。"""
        return self.result_code == DLG_RAW_ACCEPT

    def canonical_record(self) -> tuple[int, ...]:
        """返回 DLG-RAW-00 规范长度前缀整数 record。"""
        result = [DLG_RAW_RECORD_V1, self.result_code, UTF8_STRICT_V1]
        for value in (
                self.raw_input_bytes,
                self.canonical_body_bytes,
                self.unicode_scalars,
                self.typed_record,
                self.output_bytes,
                self.state_delta):
            _pack(result, value)
        return tuple(result)


def intake_raw_conversation_vector(
        raw_input_bytes: tuple[int, ...],
        ) -> ConversationRawIntake:
    """把宿主 byte vector 纯函数地转换为 DLG-RAW-00 record。"""
    raw = _byte_vector(raw_input_bytes, label="DLG-RAW raw input")
    if len(raw) > DLG_RAW_MAX_INPUT_BYTES:
        return ConversationRawIntake(DLG_RAW_REJECT_INPUT_BUDGET, raw)
    if raw[:len(_UTF8_BOM)] == _UTF8_BOM:
        return ConversationRawIntake(DLG_RAW_REJECT_BOM, raw)
    body = _split_terminal_line(raw)
    if body is None:
        return ConversationRawIntake(DLG_RAW_REJECT_LINE_FRAMING, raw)
    if not body:
        return ConversationRawIntake(DLG_RAW_REJECT_EMPTY, raw, body)
    scalars = decode_utf8_v1(body)
    if scalars is None:
        return ConversationRawIntake(DLG_RAW_REJECT_UTF8, raw, body)
    if _has_control_scalars(scalars):
        return ConversationRawIntake(
            DLG_RAW_REJECT_CONTROL, raw, body, scalars)
    return ConversationRawIntake(DLG_RAW_ACCEPT, raw, body, scalars)


def intake_raw_conversation_bytes(raw_input_bytes: bytes) -> ConversationRawIntake:
    """Python I/O adapter：把 ``bytes`` 复制到核心 byte vector 后立即退出边缘。"""
    if type(raw_input_bytes) is not bytes:
        raise TypeError("DLG-RAW bytes adapter 只接受 bytes")
    return intake_raw_conversation_vector(tuple(raw_input_bytes))


__all__ = [
    "ConversationRawIntake",
    "ConversationRawIntakeError",
    "DLG_RAW_ACCEPT",
    "DLG_RAW_MAX_INPUT_BYTES",
    "DLG_RAW_RECORD_V1",
    "DLG_RAW_REJECT_BOM",
    "DLG_RAW_REJECT_CONSTRUCTION_MISS",
    "DLG_RAW_REJECT_CONTEXT",
    "DLG_RAW_REJECT_CONTROL",
    "DLG_RAW_REJECT_EMPTY",
    "DLG_RAW_REJECT_INPUT_BUDGET",
    "DLG_RAW_REJECT_LEXICAL_AMBIGUOUS",
    "DLG_RAW_REJECT_LEXICAL_MISS",
    "DLG_RAW_REJECT_LINE_FRAMING",
    "DLG_RAW_REJECT_OUTPUT_BUDGET",
    "DLG_RAW_REJECT_REFERENCE_AMBIGUOUS",
    "DLG_RAW_REJECT_SOURCE_CONFLICT",
    "DLG_RAW_REJECT_RUNTIME",
    "DLG_RAW_REJECT_UTF8",
    "UTF8_STRICT_V1",
    "decode_utf8_v1",
    "encode_utf8_v1",
    "intake_raw_conversation_bytes",
    "intake_raw_conversation_vector",
]
