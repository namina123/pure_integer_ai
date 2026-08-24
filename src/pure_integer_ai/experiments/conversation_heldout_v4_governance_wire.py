"""DLG-05 v4 G0a-0 的跨语言治理 wire contract。

本模块只实现 ``GOV-CJSON-1`` 的有界 ASCII parser、encoder、detached-signature
envelope 和 domain message。它只用 SHA-256 构造已冻结 document identity，不导入或调用
签名、私钥或验签 API，不读取文件、环境或网络，也不把可解析的 document 表述为 trusted
root、source qualification 或训练资格。未来任一语言只需重现这里冻结的原始 byte 规则、
SHA-256 identity 与公开 corpus，即可替换本模块的宿主实现。
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, NoReturn


GOV_CJSON_PROFILE = "GOV-CJSON-1"
"""G0 的窄、ASCII 限定 canonical JSON profile 名称。"""

GOVERNANCE_WIRE_STATUS_REFERENCE_ONLY = "PORTABILITY_CONTRACT_REFERENCE_ONLY"
"""G0a-0 的最高状态；没有任何来源资格、密码或能力含义。"""

GOVERNANCE_WIRE_MAX_DOCUMENT_BYTES = 65_536
GOVERNANCE_WIRE_MAX_DEPTH = 16
GOVERNANCE_WIRE_MAX_OBJECT_MEMBERS = 128
GOVERNANCE_WIRE_MAX_ARRAY_ELEMENTS = 1_024
GOVERNANCE_WIRE_MAX_STRING_BYTES = 4_096
GOVERNANCE_WIRE_MAX_U63 = 9_223_372_036_854_775_807

GOVERNANCE_WIRE_ED25519_PUBLIC_KEY_BYTES = 32
GOVERNANCE_WIRE_ED25519_SIGNATURE_BYTES = 64

GOVERNANCE_WIRE_VERDICT_INVALID = 0
GOVERNANCE_WIRE_VERDICT_VALID = 1

GOVERNANCE_WIRE_REJECT_SYNTAX = 1
GOVERNANCE_WIRE_REJECT_BUDGET = 2
GOVERNANCE_WIRE_REJECT_ENVELOPE = 3
GOVERNANCE_WIRE_REJECT_COMMON_PAYLOAD = 4
GOVERNANCE_WIRE_REJECT_HEX = 5
GOVERNANCE_WIRE_REJECT_BYTE_TUPLE = 6

GOVERNANCE_WIRE_ALGORITHM = "Ed25519"
GOVERNANCE_WIRE_SCHEMA = 1
GOVERNANCE_WIRE_VERSION = 1

GOVERNANCE_WIRE_ROOT_REGISTRY = "root-registry"
GOVERNANCE_WIRE_REVOCATION_SNAPSHOT = "revocation-snapshot"
GOVERNANCE_WIRE_SOURCE_SNAPSHOT_DECLARATION = "source-snapshot-declaration"
GOVERNANCE_WIRE_ANNOTATION_SOURCE_DECLARATION = "annotation-source-declaration"

_DOMAIN_PREFIXES = {
    GOVERNANCE_WIRE_ROOT_REGISTRY: b"PIDSLCA-G0/root-registry/v1\x00",
    GOVERNANCE_WIRE_REVOCATION_SNAPSHOT: b"PIDSLCA-G0/revocation-snapshot/v1\x00",
    GOVERNANCE_WIRE_SOURCE_SNAPSHOT_DECLARATION: (
        b"PIDSLCA-G0/source-snapshot-declaration/v1\x00"),
    GOVERNANCE_WIRE_ANNOTATION_SOURCE_DECLARATION: (
        b"PIDSLCA-G0/annotation-source-declaration/v1\x00"),
}
_ENVELOPE_FIELDS = frozenset({"signature_hex", "signed_payload"})
_COMMON_PAYLOAD_FIELDS = frozenset({
    "algorithm", "key_id", "kind", "schema", "version",
})
_HEX_CHARS = frozenset("0123456789abcdef")
_TOKEN_CHARS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._:-")


# object-model: exception
class ConversationHeldOutV4GovernanceWireError(ValueError):
    """GOV-CJSON-1 的语法、预算、envelope 或固定字段不满足。"""

    def __init__(self, code: int, message: str) -> None:
        """保存跨语言 corpus 使用的稳定整数错误码。"""
        if type(code) is not int or code not in {
                GOVERNANCE_WIRE_REJECT_SYNTAX,
                GOVERNANCE_WIRE_REJECT_BUDGET,
                GOVERNANCE_WIRE_REJECT_ENVELOPE,
                GOVERNANCE_WIRE_REJECT_COMMON_PAYLOAD,
                GOVERNANCE_WIRE_REJECT_HEX,
                GOVERNANCE_WIRE_REJECT_BYTE_TUPLE,
        }:
            raise ValueError("GOV-CJSON-1 错误码未注册")
        self.code = code
        super().__init__(message)


def _fail(code: int, message: str) -> NoReturn:
    """统一以稳定码 fail closed，异常文字不属于互操作语义。"""
    raise ConversationHeldOutV4GovernanceWireError(code, message)


def _require_ascii_text(value: Any, *, label: str) -> str:
    """要求 value 是长度受限、仅含 printable ASCII 的文本。"""
    if type(value) is not str:
        _fail(GOVERNANCE_WIRE_REJECT_SYNTAX, f"{label} 必须是 ASCII string")
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError as exc:
        _fail(GOVERNANCE_WIRE_REJECT_SYNTAX, f"{label} 含非 ASCII 字符")
        raise AssertionError from exc
    if (len(encoded) > GOVERNANCE_WIRE_MAX_STRING_BYTES
            or any(byte < 0x20 or byte > 0x7e for byte in encoded)):
        _fail(GOVERNANCE_WIRE_REJECT_BUDGET, f"{label} 超出 ASCII string 边界")
    return value


def _require_field_name(value: Any, *, label: str) -> str:
    """要求 object field name 是非空、稳定的 ASCII 标识符。"""
    text = _require_ascii_text(value, label=label)
    if (not text or not text[0].isalpha()
            or any(character not in _TOKEN_CHARS for character in text)):
        _fail(GOVERNANCE_WIRE_REJECT_SYNTAX, f"{label} 不是 GOV-CJSON field name")
    return text


def _require_token(value: Any, *, label: str, maximum: int = 128) -> str:
    """要求受控协议 token 不带空白、escape 或自由文本。"""
    text = _require_ascii_text(value, label=label)
    if (not text or len(text) > maximum
            or any(character not in _TOKEN_CHARS for character in text)):
        _fail(GOVERNANCE_WIRE_REJECT_COMMON_PAYLOAD,
              f"{label} 不是受控 ASCII token")
    return text


def _require_u63(value: Any, *, label: str) -> int:
    """要求严格 JSON integer 已落在可移植的非负 signed-64 范围。"""
    if type(value) is not int or value < 0 or value > GOVERNANCE_WIRE_MAX_U63:
        _fail(GOVERNANCE_WIRE_REJECT_SYNTAX, f"{label} 不是可移植 u63 integer")
    return value


def _require_lower_hex(value: Any, *, label: str, byte_count: int) -> bytes:
    """把固定长度 lowercase hex 还原为 bytes，不接受宽松编码。"""
    if type(byte_count) is not int or byte_count <= 0:
        raise RuntimeError("GOV-CJSON-1 内部 hex 长度非法")
    # 此处的 value 已通过物理 GOV-CJSON-1 parser；非 string 属于 hex 字段语义，
    # 而不是 parser 物理语法，必须稳定归入 code 5。
    if type(value) is not str:
        _fail(GOVERNANCE_WIRE_REJECT_HEX, f"{label} 必须是 lowercase hex string")
    text = _require_ascii_text(value, label=label)
    if (len(text) != byte_count * 2
            or any(character not in _HEX_CHARS for character in text)):
        _fail(GOVERNANCE_WIRE_REJECT_HEX, f"{label} 不是固定 lowercase hex")
    return bytes.fromhex(text)


def _require_byte_tuple(
        value: Any, *, label: str, byte_count: int,
        ) -> tuple[int, ...]:
    """验证 adapter 边界使用的完整 ``0..255`` 整数 byte tuple。"""
    if (not isinstance(value, tuple) or len(value) != byte_count
            or any(type(item) is not int or item < 0 or item > 255
                   for item in value)):
        _fail(GOVERNANCE_WIRE_REJECT_BYTE_TUPLE,
              f"{label} 不是固定长度 byte tuple")
    return value


# object-model: state machine; parser cursor
class _GovCjsonParser:
    """仅接受 GOV-CJSON-1 的小型递归下降 parser，cursor 是唯一可变状态。"""

    def __init__(self, payload: bytes) -> None:
        """在解析前冻结输入字节和 cursor，拒绝 BOM、非 ASCII 与超预算。"""
        if type(payload) is not bytes or not payload:
            _fail(GOVERNANCE_WIRE_REJECT_SYNTAX,
                  "GOV-CJSON-1 payload 必须是非空 bytes")
        if len(payload) > GOVERNANCE_WIRE_MAX_DOCUMENT_BYTES:
            _fail(GOVERNANCE_WIRE_REJECT_BUDGET,
                  "GOV-CJSON-1 document 超过 byte budget")
        if payload.startswith(b"\xef\xbb\xbf") or any(byte >= 0x80 for byte in payload):
            _fail(GOVERNANCE_WIRE_REJECT_SYNTAX,
                  "GOV-CJSON-1 禁止 BOM 或非 ASCII bytes")
        self._payload = payload
        self._cursor = 0

    def parse_object_root(self) -> dict[str, Any]:
        """解析唯一允许的 object root，并拒绝尾随 byte。"""
        value = self._parse_value(depth=1)
        if not isinstance(value, dict):
            _fail(GOVERNANCE_WIRE_REJECT_SYNTAX,
                  "GOV-CJSON-1 root 必须是 object")
        if self._cursor != len(self._payload):
            _fail(GOVERNANCE_WIRE_REJECT_SYNTAX,
                  "GOV-CJSON-1 含尾随 bytes")
        return value

    def _parse_value(self, *, depth: int) -> Any:
        """按当前首 byte 解析 object、array、string 或严格 u63 integer。"""
        if depth > GOVERNANCE_WIRE_MAX_DEPTH:
            _fail(GOVERNANCE_WIRE_REJECT_BUDGET,
                  "GOV-CJSON-1 超过最大嵌套深度")
        current = self._peek()
        if current == ord("{"):
            return self._parse_object(depth=depth)
        if current == ord("["):
            return self._parse_array(depth=depth)
        if current == ord('"'):
            return self._parse_string(label="GOV-CJSON-1 string")
        if ord("0") <= current <= ord("9"):
            return self._parse_u63()
        _fail(GOVERNANCE_WIRE_REJECT_SYNTAX,
              "GOV-CJSON-1 只允许 object/array/string/u63 value")

    def _parse_object(self, *, depth: int) -> dict[str, Any]:
        """解析字段严格升序、无重复且受成员预算限制的 object。"""
        self._expect(ord("{"), label="object open")
        result: dict[str, Any] = {}
        previous: bytes | None = None
        if self._consume_if(ord("}")):
            return result
        while True:
            key = _require_field_name(
                self._parse_string(label="GOV-CJSON-1 object key"),
                label="GOV-CJSON-1 object key",
            )
            encoded_key = key.encode("ascii")
            if previous is not None and encoded_key <= previous:
                _fail(GOVERNANCE_WIRE_REJECT_SYNTAX,
                      "GOV-CJSON-1 object key 未严格按 ASCII 升序")
            if key in result:
                _fail(GOVERNANCE_WIRE_REJECT_SYNTAX,
                      "GOV-CJSON-1 object 含重复 key")
            previous = encoded_key
            if len(result) >= GOVERNANCE_WIRE_MAX_OBJECT_MEMBERS:
                _fail(GOVERNANCE_WIRE_REJECT_BUDGET,
                      "GOV-CJSON-1 object members 超过预算")
            self._expect(ord(":"), label="object colon")
            result[key] = self._parse_value(depth=depth + 1)
            if self._consume_if(ord("}")):
                return result
            self._expect(ord(","), label="object comma")

    def _parse_array(self, *, depth: int) -> list[Any]:
        """解析保序、受元素预算限制的 array，不替调用方排序。"""
        self._expect(ord("["), label="array open")
        result: list[Any] = []
        if self._consume_if(ord("]")):
            return result
        while True:
            if len(result) >= GOVERNANCE_WIRE_MAX_ARRAY_ELEMENTS:
                _fail(GOVERNANCE_WIRE_REJECT_BUDGET,
                      "GOV-CJSON-1 array elements 超过预算")
            result.append(self._parse_value(depth=depth + 1))
            if self._consume_if(ord("]")):
                return result
            self._expect(ord(","), label="array comma")

    def _parse_string(self, *, label: str) -> str:
        """解析 printable ASCII string，仅允许 ``\\\"`` 和 ``\\\\`` escape。"""
        self._expect(ord('"'), label=f"{label} open")
        content_start = self._cursor
        result = bytearray()
        while True:
            if self._cursor >= len(self._payload):
                _fail(GOVERNANCE_WIRE_REJECT_SYNTAX, f"{label} 被截断")
            current = self._payload[self._cursor]
            self._cursor += 1
            if current == ord('"'):
                if self._cursor - content_start - 1 > GOVERNANCE_WIRE_MAX_STRING_BYTES:
                    _fail(GOVERNANCE_WIRE_REJECT_BUDGET,
                          f"{label} 表示超出 string byte budget")
                return result.decode("ascii")
            if current == ord("\\"):
                if self._cursor >= len(self._payload):
                    _fail(GOVERNANCE_WIRE_REJECT_SYNTAX, f"{label} escape 被截断")
                escaped = self._payload[self._cursor]
                self._cursor += 1
                if escaped not in {ord('"'), ord("\\")}:
                    _fail(GOVERNANCE_WIRE_REJECT_SYNTAX,
                          f"{label} 含未注册 escape")
                result.append(escaped)
            elif 0x20 <= current <= 0x7e and current not in {ord('"'), ord("\\")}:
                result.append(current)
            else:
                _fail(GOVERNANCE_WIRE_REJECT_SYNTAX,
                      f"{label} 含非法 ASCII byte")
            if len(result) > GOVERNANCE_WIRE_MAX_STRING_BYTES:
                _fail(GOVERNANCE_WIRE_REJECT_BUDGET,
                      f"{label} 超过 string byte budget")

    def _parse_u63(self) -> int:
        """逐位解析无前导零 u63，绝不经 float 或宽松 JSON number。"""
        start = self._cursor
        first = self._payload[self._cursor]
        self._cursor += 1
        if first == ord("0"):
            if self._cursor < len(self._payload) and ord("0") <= self._payload[self._cursor] <= ord("9"):
                _fail(GOVERNANCE_WIRE_REJECT_SYNTAX,
                      "GOV-CJSON-1 integer 不得有前导零")
            return 0
        value = first - ord("0")
        while self._cursor < len(self._payload):
            current = self._payload[self._cursor]
            if not (ord("0") <= current <= ord("9")):
                break
            digit = current - ord("0")
            if value > (GOVERNANCE_WIRE_MAX_U63 - digit) // 10:
                _fail(GOVERNANCE_WIRE_REJECT_SYNTAX,
                      "GOV-CJSON-1 integer 超过 u63")
            value = value * 10 + digit
            self._cursor += 1
        if self._cursor == start:
            raise RuntimeError("GOV-CJSON-1 内部 integer cursor 未推进")
        return value

    def _peek(self) -> int:
        """返回当前 byte；任何物理截断都按语法错误拒绝。"""
        if self._cursor >= len(self._payload):
            _fail(GOVERNANCE_WIRE_REJECT_SYNTAX,
                  "GOV-CJSON-1 payload 被截断")
        return self._payload[self._cursor]

    def _expect(self, value: int, *, label: str) -> None:
        """要求当前位置精确等于协议 delimiter，不跳过空白。"""
        if self._peek() != value:
            _fail(GOVERNANCE_WIRE_REJECT_SYNTAX,
                  f"GOV-CJSON-1 缺少 {label}")
        self._cursor += 1

    def _consume_if(self, value: int) -> bool:
        """仅在当前位置相等时消费 delimiter，避免隐式空白容忍。"""
        if self._cursor < len(self._payload) and self._payload[self._cursor] == value:
            self._cursor += 1
            return True
        return False


def _encode_string(value: Any, *, label: str) -> bytes:
    """把受限 ASCII string 以唯一合法的 JSON escape 形式编码。"""
    text = _require_ascii_text(value, label=label)
    result = bytearray(b'"')
    for byte in text.encode("ascii"):
        if byte in {ord('"'), ord("\\")}:
            result.append(ord("\\"))
        result.append(byte)
    if len(result) - 1 > GOVERNANCE_WIRE_MAX_STRING_BYTES:
        _fail(GOVERNANCE_WIRE_REJECT_BUDGET,
              f"{label} 表示超出 string byte budget")
    result.append(ord('"'))
    return bytes(result)


def _append_encoded_bytes(result: bytearray, value: bytes, *, label: str) -> None:
    """在写入前核对 document 剩余预算，禁止先物化超限 canonical bytes。"""
    if len(result) + len(value) > GOVERNANCE_WIRE_MAX_DOCUMENT_BYTES:
        _fail(GOVERNANCE_WIRE_REJECT_BUDGET,
              f"{label} encoded document 超过 byte budget")
    result.extend(value)


def _encode_value(
        value: Any, *, depth: int, label: str, result: bytearray,
        ) -> None:
    """递归写入 GOV-CJSON-1 值，每次 append 前执行同一 document budget 门。"""
    if depth > GOVERNANCE_WIRE_MAX_DEPTH:
        _fail(GOVERNANCE_WIRE_REJECT_BUDGET,
              f"{label} 超过 GOV-CJSON-1 最大深度")
    if type(value) is str:
        _append_encoded_bytes(result, _encode_string(value, label=label), label=label)
        return
    if type(value) is int:
        _append_encoded_bytes(
            result, str(_require_u63(value, label=label)).encode("ascii"),
            label=label)
        return
    if type(value) is list:
        if len(value) > GOVERNANCE_WIRE_MAX_ARRAY_ELEMENTS:
            _fail(GOVERNANCE_WIRE_REJECT_BUDGET,
                  f"{label} array elements 超过预算")
        _append_encoded_bytes(result, b"[", label=label)
        for index, item in enumerate(value):
            if index:
                _append_encoded_bytes(result, b",", label=label)
            _encode_value(
                item, depth=depth + 1, label=f"{label}[]", result=result)
        _append_encoded_bytes(result, b"]", label=label)
        return
    if type(value) is dict:
        if len(value) > GOVERNANCE_WIRE_MAX_OBJECT_MEMBERS:
            _fail(GOVERNANCE_WIRE_REJECT_BUDGET,
                  f"{label} object members 超过预算")
        items: list[tuple[bytes, str, Any]] = []
        for key, item in value.items():
            field = _require_field_name(key, label=f"{label} field")
            items.append((field.encode("ascii"), field, item))
        items.sort(key=lambda item: item[0])
        _append_encoded_bytes(result, b"{", label=label)
        for index, (_, field, item) in enumerate(items):
            if index:
                _append_encoded_bytes(result, b",", label=label)
            _append_encoded_bytes(
                result, _encode_string(field, label=f"{label} field"), label=label)
            _append_encoded_bytes(result, b":", label=label)
            _encode_value(
                item, depth=depth + 1, label=f"{label}.{field}", result=result)
        _append_encoded_bytes(result, b"}", label=label)
        return
    _fail(GOVERNANCE_WIRE_REJECT_SYNTAX,
          f"{label} 含 GOV-CJSON-1 不支持类型")


def encode_gov_cjson(value: dict[str, Any]) -> bytes:
    """编码一个 GOV-CJSON-1 object root，输出无 BOM/空白/换行的唯一 bytes。"""
    if type(value) is not dict:
        _fail(GOVERNANCE_WIRE_REJECT_SYNTAX,
              "GOV-CJSON-1 root 必须是严格 dict")
    result = bytearray()
    _encode_value(
        value, depth=1, label="GOV-CJSON-1 root", result=result)
    return bytes(result)


def parse_gov_cjson(payload: bytes) -> dict[str, Any]:
    """解析并重编码核对一个 GOV-CJSON-1 object，拒绝任意非规范表示。"""
    value = _GovCjsonParser(payload).parse_object_root()
    if encode_gov_cjson(value) != payload:
        _fail(GOVERNANCE_WIRE_REJECT_SYNTAX,
              "GOV-CJSON-1 payload 不是唯一规范编码")
    return value


def _exact_fields(
        value: Any, fields: frozenset[str], *, label: str,
        ) -> dict[str, Any]:
    """要求一个 object 的字段集精确闭合，避免 envelope 扩展夹带数据。"""
    if type(value) is not dict or set(value) != fields:
        _fail(GOVERNANCE_WIRE_REJECT_ENVELOPE, f"{label} 字段集不闭合")
    return value


def _validate_common_signed_payload(value: Any) -> tuple[str, int, int, str]:
    """验证 G0a-0 已冻结的公共字段，document-specific schema 留给 G0b。"""
    if type(value) is not dict or not _COMMON_PAYLOAD_FIELDS <= set(value):
        _fail(GOVERNANCE_WIRE_REJECT_COMMON_PAYLOAD,
              "signed_payload 缺少公共治理字段")
    algorithm = _require_token(
        value["algorithm"], label="signed_payload algorithm")
    if algorithm != GOVERNANCE_WIRE_ALGORITHM:
        _fail(GOVERNANCE_WIRE_REJECT_COMMON_PAYLOAD,
              "signed_payload algorithm 未固定为 Ed25519")
    kind = _require_token(value["kind"], label="signed_payload kind")
    if kind not in _DOMAIN_PREFIXES:
        _fail(GOVERNANCE_WIRE_REJECT_COMMON_PAYLOAD,
              "signed_payload kind 未注册")
    schema = _require_u63(value["schema"], label="signed_payload schema")
    version = _require_u63(value["version"], label="signed_payload version")
    if schema != GOVERNANCE_WIRE_SCHEMA or version != GOVERNANCE_WIRE_VERSION:
        _fail(GOVERNANCE_WIRE_REJECT_COMMON_PAYLOAD,
              "signed_payload schema/version 未注册")
    key_id = _require_token(value["key_id"], label="signed_payload key_id")
    return kind, schema, version, key_id


# object-model: value; representation=struct; interop=GOV-CJSON-1
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4GovernanceWireEnvelope:
    """已按公共字段冻结、但尚未经任何密码 API 验证的 detached envelope。"""

    kind: str
    schema: int
    version: int
    key_id: str
    canonical_signed_payload: bytes
    detached_signature: tuple[int, ...]
    domain_prefix: bytes
    message: bytes
    document_identity: tuple[int, ...]
    status: str = GOVERNANCE_WIRE_STATUS_REFERENCE_ONLY

    def __post_init__(self) -> None:
        """复核结构字段与 SHA-256 identity，防止 host object 伪装 adapter 输入。"""
        if (type(self.kind) is not str
                or self.kind not in _DOMAIN_PREFIXES
                or _require_u63(self.schema, label="GOV-CJSON-1 envelope schema")
                != GOVERNANCE_WIRE_SCHEMA
                or _require_u63(self.version, label="GOV-CJSON-1 envelope version")
                != GOVERNANCE_WIRE_VERSION
                or type(self.status) is not str
                or self.status != GOVERNANCE_WIRE_STATUS_REFERENCE_ONLY):
            raise ValueError("GOV-CJSON-1 envelope 公共状态非法")
        _require_token(self.key_id, label="GOV-CJSON-1 envelope key_id")
        if (type(self.canonical_signed_payload) is not bytes
                or not self.canonical_signed_payload):
            raise ValueError("GOV-CJSON-1 envelope payload 必须是非空 bytes")
        signed_payload = parse_gov_cjson(self.canonical_signed_payload)
        payload_kind, payload_schema, payload_version, payload_key_id = (
            _validate_common_signed_payload(signed_payload))
        if (payload_kind != self.kind
                or payload_schema != self.schema
                or payload_version != self.version
                or payload_key_id != self.key_id):
            raise ValueError("GOV-CJSON-1 envelope 公共字段漂移")
        _require_byte_tuple(
            self.detached_signature,
            label="GOV-CJSON-1 envelope detached signature",
            byte_count=GOVERNANCE_WIRE_ED25519_SIGNATURE_BYTES,
        )
        expected_prefix = _DOMAIN_PREFIXES[self.kind]
        if type(self.domain_prefix) is not bytes or self.domain_prefix != expected_prefix:
            raise ValueError("GOV-CJSON-1 envelope domain prefix 漂移")
        if (type(self.message) is not bytes
                or self.message != expected_prefix + self.canonical_signed_payload):
            raise ValueError("GOV-CJSON-1 envelope message 漂移")
        expected_identity = tuple(hashlib.sha256(self.message).digest())
        _require_byte_tuple(
            self.document_identity,
            label="GOV-CJSON-1 envelope document identity",
            byte_count=len(expected_identity),
        )
        if self.document_identity != expected_identity:
            raise ValueError("GOV-CJSON-1 envelope identity 漂移")


def parse_governance_wire_envelope(
        payload: bytes,
        ) -> ConversationHeldOutV4GovernanceWireEnvelope:
    """解析 detached envelope 并构造固定 domain message；不执行验签。"""
    document = _exact_fields(
        parse_gov_cjson(payload), _ENVELOPE_FIELDS,
        label="GOV-CJSON-1 envelope",
    )
    signature = _require_lower_hex(
        document["signature_hex"],
        label="GOV-CJSON-1 signature_hex",
        byte_count=GOVERNANCE_WIRE_ED25519_SIGNATURE_BYTES,
    )
    signed_payload = document["signed_payload"]
    kind, schema, version, key_id = _validate_common_signed_payload(signed_payload)
    canonical_signed_payload = encode_gov_cjson(signed_payload)
    domain_prefix = _DOMAIN_PREFIXES[kind]
    message = domain_prefix + canonical_signed_payload
    return ConversationHeldOutV4GovernanceWireEnvelope(
        kind,
        schema,
        version,
        key_id,
        canonical_signed_payload,
        tuple(signature),
        domain_prefix,
        message,
        tuple(hashlib.sha256(message).digest()),
    )


def encode_governance_wire_envelope(
        signed_payload: dict[str, Any],
        detached_signature: tuple[int, ...],
        ) -> bytes:
    """以唯一 physical envelope 表示 frozen payload 与 detached signature bytes。"""
    _validate_common_signed_payload(signed_payload)
    signature = _require_byte_tuple(
        detached_signature,
        label="GOV-CJSON-1 detached signature",
        byte_count=GOVERNANCE_WIRE_ED25519_SIGNATURE_BYTES,
    )
    payload = encode_gov_cjson({
        "signature_hex": bytes(signature).hex(),
        "signed_payload": signed_payload,
    })
    parse_governance_wire_envelope(payload)
    return payload


def governance_wire_domain_prefix(kind: str) -> bytes:
    """返回一个已注册 kind 的固定 domain prefix，拒绝调用方动态注入。"""
    token = _require_token(kind, label="GOV-CJSON-1 domain kind")
    try:
        return _DOMAIN_PREFIXES[token]
    except KeyError as exc:
        _fail(GOVERNANCE_WIRE_REJECT_COMMON_PAYLOAD,
              "GOV-CJSON-1 domain kind 未注册")
        raise AssertionError from exc


__all__ = [
    "ConversationHeldOutV4GovernanceWireEnvelope",
    "ConversationHeldOutV4GovernanceWireError",
    "GOV_CJSON_PROFILE",
    "GOVERNANCE_WIRE_ALGORITHM",
    "GOVERNANCE_WIRE_ANNOTATION_SOURCE_DECLARATION",
    "GOVERNANCE_WIRE_ED25519_PUBLIC_KEY_BYTES",
    "GOVERNANCE_WIRE_ED25519_SIGNATURE_BYTES",
    "GOVERNANCE_WIRE_MAX_ARRAY_ELEMENTS",
    "GOVERNANCE_WIRE_MAX_DEPTH",
    "GOVERNANCE_WIRE_MAX_DOCUMENT_BYTES",
    "GOVERNANCE_WIRE_MAX_OBJECT_MEMBERS",
    "GOVERNANCE_WIRE_MAX_STRING_BYTES",
    "GOVERNANCE_WIRE_MAX_U63",
    "GOVERNANCE_WIRE_REJECT_BUDGET",
    "GOVERNANCE_WIRE_REJECT_BYTE_TUPLE",
    "GOVERNANCE_WIRE_REJECT_COMMON_PAYLOAD",
    "GOVERNANCE_WIRE_REJECT_ENVELOPE",
    "GOVERNANCE_WIRE_REJECT_HEX",
    "GOVERNANCE_WIRE_REJECT_SYNTAX",
    "GOVERNANCE_WIRE_REVOCATION_SNAPSHOT",
    "GOVERNANCE_WIRE_ROOT_REGISTRY",
    "GOVERNANCE_WIRE_SCHEMA",
    "GOVERNANCE_WIRE_SOURCE_SNAPSHOT_DECLARATION",
    "GOVERNANCE_WIRE_STATUS_REFERENCE_ONLY",
    "GOVERNANCE_WIRE_VERDICT_INVALID",
    "GOVERNANCE_WIRE_VERDICT_VALID",
    "GOVERNANCE_WIRE_VERSION",
    "encode_governance_wire_envelope",
    "encode_gov_cjson",
    "governance_wire_domain_prefix",
    "parse_governance_wire_envelope",
    "parse_gov_cjson",
]
