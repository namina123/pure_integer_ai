"""DLG-05 v4 G0b/G0c 的跨语言治理 schema reference。

本模块只在 ``GOV-CJSON-1`` wire 之上冻结 root-registry、revocation-snapshot
和 source-snapshot-declaration 的字段、标量、数组顺序与 genesis 规则。它不验签、
不读取路径/文件/环境/网络，不构造 capability，也不把任何输入升级为可信来源。

返回的结构体只是当前 Python host 对已冻结 byte/integer 原始值的只读视图；跨语言
语义始终由 canonical signed-payload bytes、domain message、identity 和固定整数错误码
定义，而不是由 ``dict``、dataclass 或异常文字定义。
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, NoReturn

from pure_integer_ai.experiments.conversation_heldout_v4_governance_wire import (
    ConversationHeldOutV4GovernanceWireEnvelope,
    GOVERNANCE_WIRE_ALGORITHM,
    GOVERNANCE_WIRE_ED25519_SIGNATURE_BYTES,
    GOVERNANCE_WIRE_MAX_U63,
    GOVERNANCE_WIRE_REVOCATION_SNAPSHOT,
    GOVERNANCE_WIRE_ROOT_REGISTRY,
    GOVERNANCE_WIRE_SCHEMA,
    GOVERNANCE_WIRE_SOURCE_SNAPSHOT_DECLARATION,
    GOVERNANCE_WIRE_STATUS_REFERENCE_ONLY,
    GOVERNANCE_WIRE_VERSION,
    governance_wire_domain_prefix,
    parse_gov_cjson,
    parse_governance_wire_envelope,
)


GOVERNANCE_SCHEMA_STATUS_REFERENCE_ONLY = GOVERNANCE_WIRE_STATUS_REFERENCE_ONLY
"""本切片的最高状态；它没有 signature verdict 或来源资格含义。"""

GOVERNANCE_SCHEMA_OK = 0
GOVERNANCE_SCHEMA_REJECT_EXACT_FIELDS = 101
GOVERNANCE_SCHEMA_REJECT_SCALAR = 102

GOVERNANCE_SCHEMA_SOURCE_REF_KEY_LENGTH = 11
GOVERNANCE_SCHEMA_ZERO_SHA256 = "0" * 64

_ASCII_LOWER = frozenset("abcdefghijklmnopqrstuvwxyz")
_ASCII_UPPER = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
_ASCII_DIGITS = frozenset("0123456789")
_ASCII_HEX_LOWER = frozenset("0123456789abcdef")
_ASCII_HEX_UPPER = frozenset("0123456789ABCDEF")
_TOKEN_TAIL = _ASCII_LOWER | _ASCII_DIGITS | frozenset("._:-")
_OPAQUE_TAIL = _ASCII_LOWER | _ASCII_UPPER | _ASCII_DIGITS | frozenset("._:-")
_LICENSE_CHARS = _ASCII_LOWER | _ASCII_UPPER | _ASCII_DIGITS | frozenset(".-")
_URI_TAIL_CHARS = (
    _ASCII_LOWER | _ASCII_UPPER | _ASCII_DIGITS
    | frozenset("-._~!$&'()*+,;=:@/?#%"))

_ROOT_REGISTRY_FIELDS = frozenset({
    "algorithm",
    "issuers",
    "key_id",
    "kind",
    "predecessor_registry_identity_sha256",
    "schema",
    "sequence",
    "version",
})
_ISSUER_RECORD_FIELDS = frozenset({
    "control_domain",
    "issuer_key_id",
    "not_after_registry_sequence",
    "not_before_registry_sequence",
    "public_key_hex",
    "role",
})
_REVOCATION_SNAPSHOT_FIELDS = frozenset({
    "algorithm",
    "key_id",
    "kind",
    "predecessor_revocation_identity_sha256",
    "registry_document_identity_sha256",
    "revocations",
    "schema",
    "sequence",
    "version",
})
_REVOCATION_RECORD_FIELDS = frozenset({
    "effective_declaration_sequence",
    "reason_digest_sha256",
    "revoked_key_id",
})
_SOURCE_SNAPSHOT_DECLARATION_FIELDS = frozenset({
    "algorithm",
    "control_domain",
    "ingest_code_identity_sha256",
    "key_id",
    "kind",
    "license_id",
    "license_review_artifact_identity_sha256",
    "metadata_byte_count",
    "metadata_id",
    "metadata_sha256",
    "official_uri",
    "predecessor_declaration_identity_sha256",
    "registry_document_identity_sha256",
    "revocation_document_identity_sha256",
    "schema",
    "sequence",
    "snapshot_id",
    "source_file_byte_count",
    "source_file_id",
    "source_file_sha256",
    "source_key",
    "source_ref_key",
    "transform_code_identity_sha256",
    "upstream_digest_algorithm",
    "upstream_digest_hex",
    "version",
})


# object-model: exception
class ConversationHeldOutV4GovernanceSchemaError(ValueError):
    """G0b/G0c schema 的固定整数拒绝码，异常文字不属于协议值。"""

    def __init__(self, code: int, message: str) -> None:
        """保存 corpus 可断言的 schema 拒绝码。"""
        if type(code) is not int or code not in {
                GOVERNANCE_SCHEMA_REJECT_EXACT_FIELDS,
                GOVERNANCE_SCHEMA_REJECT_SCALAR,
        }:
            raise ValueError("G0 governance schema 错误码未注册")
        self.code = code
        super().__init__(message)


def _fail_exact(message: str) -> NoReturn:
    """以 exact-field code 拒绝未知、缺失或不支持的 schema。"""
    raise ConversationHeldOutV4GovernanceSchemaError(
        GOVERNANCE_SCHEMA_REJECT_EXACT_FIELDS, message)


def _fail_scalar(message: str) -> NoReturn:
    """以 scalar/bound code 拒绝词法、数值、顺序和 genesis 漂移。"""
    raise ConversationHeldOutV4GovernanceSchemaError(
        GOVERNANCE_SCHEMA_REJECT_SCALAR, message)


def _require_exact_fields(
        value: Any, fields: frozenset[str], *, label: str,
        ) -> dict[str, Any]:
    """要求 object 无未知、缺失或宿主附带字段。"""
    if type(value) is not dict or frozenset(value) != fields:
        _fail_exact(f"{label} 字段集不精确")
    return value


def _require_text(value: Any, *, label: str) -> str:
    """要求已由 wire 限定过的 printable-ASCII string。"""
    if type(value) is not str:
        _fail_scalar(f"{label} 必须是 ASCII text")
    return value


def _require_u63(value: Any, *, label: str, positive: bool = False) -> int:
    """要求无隐式 bool/float 的固定 u63，按 schema 选择是否必须正数。"""
    if (type(value) is not int or value < 0
            or value > GOVERNANCE_WIRE_MAX_U63
            or (positive and value == 0)):
        _fail_scalar(f"{label} 不是要求的 u63")
    return value


def _require_key_id(value: Any, *, label: str) -> str:
    """验证固定 lowercase key/control-domain token，不从路径或 DNS 推导。"""
    text = _require_text(value, label=label)
    if (not 1 <= len(text) <= 64
            or text[0] not in _ASCII_LOWER
            or any(character not in _TOKEN_TAIL for character in text)):
        _fail_scalar(f"{label} 不是固定 key_id/control_domain")
    return text


def _require_opaque_id(value: Any, *, label: str) -> str:
    """验证不含路径分隔符的 opaque logical ID。"""
    text = _require_text(value, label=label)
    if (not 1 <= len(text) <= 128
            or text[0] not in (_ASCII_LOWER | _ASCII_UPPER)
            or any(character not in _OPAQUE_TAIL for character in text)):
        _fail_scalar(f"{label} 不是 opaque_id")
    return text


def _require_license_id(value: Any, *, label: str) -> str:
    """验证单一 SPDX-like license token；不判断法律事实。"""
    text = _require_text(value, label=label)
    if (not 1 <= len(text) <= 128
            or any(character not in _LICENSE_CHARS for character in text)):
        _fail_scalar(f"{label} 不是 license_id")
    return text


def _require_sha256(
        value: Any, *, label: str, allow_zero: bool = False,
        ) -> str:
    """验证完整 lowercase SHA-256 text，genesis predecessor 才可使用全零。"""
    text = _require_text(value, label=label)
    if (len(text) != 64
            or any(character not in _ASCII_HEX_LOWER for character in text)
            or (not allow_zero and text == GOVERNANCE_SCHEMA_ZERO_SHA256)):
        _fail_scalar(f"{label} 不是允许的 SHA-256")
    return text


def _require_predecessor(
        value: Any, *, sequence: int, label: str,
        ) -> str:
    """冻结 sequence=1 与全零 predecessor 的双向 genesis 规则。"""
    digest = _require_sha256(value, label=label, allow_zero=True)
    if ((sequence == 1 and digest != GOVERNANCE_SCHEMA_ZERO_SHA256)
            or (sequence != 1 and digest == GOVERNANCE_SCHEMA_ZERO_SHA256)):
        _fail_scalar(f"{label} 与 sequence genesis 规则不一致")
    return digest


def _require_public_key(value: Any, *, label: str) -> str:
    """验证未解释的固定 32-byte Ed25519 public-key hex。"""
    text = _require_text(value, label=label)
    if (len(text) != 64
            or any(character not in _ASCII_HEX_LOWER for character in text)):
        _fail_scalar(f"{label} 不是 32-byte lowercase public key")
    return text


def _require_strict_https_locator(value: Any, *, label: str) -> str:
    """验证不访问网络、不归一化 Unicode 的受限 ASCII https locator。"""
    text = _require_text(value, label=label)
    if not text.startswith("https://"):
        _fail_scalar(f"{label} 必须以 https:// 开头")
    authority_and_tail = text[len("https://"):]
    slash_index = authority_and_tail.find("/")
    if slash_index < 0:
        host = authority_and_tail
        tail = ""
    else:
        host = authority_and_tail[:slash_index]
        tail = authority_and_tail[slash_index:]
    if not host or len(host) > 253:
        _fail_scalar(f"{label} host 长度非法")
    labels = host.split(".")
    if any(
            not 1 <= len(item) <= 63
            or item[0] not in (_ASCII_LOWER | _ASCII_DIGITS)
            or item[-1] not in (_ASCII_LOWER | _ASCII_DIGITS)
            or any(character not in (_ASCII_LOWER | _ASCII_DIGITS | {"-"})
                   for character in item)
            for item in labels):
        _fail_scalar(f"{label} host 不是 lowercase ASCII DNS A-label")
    position = 0
    while position < len(tail):
        character = tail[position]
        if character not in _URI_TAIL_CHARS:
            _fail_scalar(f"{label} path/query/fragment 含非法字符")
        if character == "%":
            if (position + 2 >= len(tail)
                    or tail[position + 1] not in _ASCII_HEX_UPPER
                    or tail[position + 2] not in _ASCII_HEX_UPPER):
                _fail_scalar(f"{label} percent-encoding 非大写 hex")
            position += 3
        else:
            position += 1
    return text


def _require_common(
        payload: dict[str, Any], *, kind: str,
        ) -> tuple[str, int]:
    """复核 wire 已处理的公共字段，防止内部调用跳过固定值。"""
    if (payload["algorithm"] != GOVERNANCE_WIRE_ALGORITHM
            or payload["kind"] != kind
            or payload["schema"] != GOVERNANCE_WIRE_SCHEMA
            or payload["version"] != GOVERNANCE_WIRE_VERSION):
        _fail_scalar("signed_payload 公共常量漂移")
    return (_require_key_id(payload["key_id"], label="signed_payload key_id"),
            _require_u63(payload["sequence"],
                         label="signed_payload sequence", positive=True))


def _validate_issuers(value: Any) -> None:
    """验证 root registry 的非空、排序、唯一 issuer records。"""
    if type(value) is not list or not value:
        _fail_scalar("root-registry issuers 必须是非空 array")
    previous_key: bytes | None = None
    public_keys: set[str] = set()
    for index, item in enumerate(value):
        record = _require_exact_fields(
            item, _ISSUER_RECORD_FIELDS, label=f"root-registry issuers[{index}]")
        issuer_key = _require_key_id(
            record["issuer_key_id"], label=f"issuer[{index}] key_id")
        encoded_key = issuer_key.encode("ascii")
        if previous_key is not None and encoded_key <= previous_key:
            _fail_scalar("root-registry issuers 未严格按 issuer_key_id 排序")
        previous_key = encoded_key
        public_key = _require_public_key(
            record["public_key_hex"], label=f"issuer[{index}] public_key_hex")
        if public_key in public_keys:
            _fail_scalar("root-registry issuers public_key_hex 不得重复")
        public_keys.add(public_key)
        _require_key_id(
            record["control_domain"], label=f"issuer[{index}] control_domain")
        before = _require_u63(
            record["not_before_registry_sequence"],
            label=f"issuer[{index}] not_before_registry_sequence", positive=True)
        after = _require_u63(
            record["not_after_registry_sequence"],
            label=f"issuer[{index}] not_after_registry_sequence", positive=True)
        if before > after:
            _fail_scalar("root-registry issuer validity window 反转")
        role = _require_text(record["role"], label=f"issuer[{index}] role")
        if role not in ("SOURCE_SNAPSHOT", "ANNOTATION_SOURCE"):
            _fail_scalar("root-registry issuer role 未注册")


def _validate_root_registry(payload: dict[str, Any]) -> int:
    """验证 root-registry v1 本体，不执行 root pin 或 signature 验证。"""
    _require_exact_fields(payload, _ROOT_REGISTRY_FIELDS, label="root-registry")
    _unused_key_id, sequence = _require_common(
        payload, kind=GOVERNANCE_WIRE_ROOT_REGISTRY)
    _require_predecessor(
        payload["predecessor_registry_identity_sha256"], sequence=sequence,
        label="root-registry predecessor")
    _validate_issuers(payload["issuers"])
    return sequence


def _validate_revocations(value: Any) -> None:
    """验证可空但严格有序的累计 revocation record 集合形状。"""
    if type(value) is not list:
        _fail_scalar("revocation-snapshot revocations 必须是 array")
    previous_key: bytes | None = None
    for index, item in enumerate(value):
        record = _require_exact_fields(
            item, _REVOCATION_RECORD_FIELDS,
            label=f"revocation-snapshot revocations[{index}]")
        revoked_key = _require_key_id(
            record["revoked_key_id"], label=f"revocation[{index}] revoked_key_id")
        encoded_key = revoked_key.encode("ascii")
        if previous_key is not None and encoded_key <= previous_key:
            _fail_scalar("revocation-snapshot revocations 未严格按 revoked_key_id 排序")
        previous_key = encoded_key
        _require_u63(
            record["effective_declaration_sequence"],
            label=f"revocation[{index}] effective_declaration_sequence",
            positive=True)
        _require_sha256(
            record["reason_digest_sha256"],
            label=f"revocation[{index}] reason_digest_sha256")


def _validate_revocation_snapshot(payload: dict[str, Any]) -> int:
    """验证 registry-bound revocation snapshot 的局部 schema，不追链。"""
    _require_exact_fields(
        payload, _REVOCATION_SNAPSHOT_FIELDS, label="revocation-snapshot")
    _unused_key_id, sequence = _require_common(
        payload, kind=GOVERNANCE_WIRE_REVOCATION_SNAPSHOT)
    _require_predecessor(
        payload["predecessor_revocation_identity_sha256"], sequence=sequence,
        label="revocation-snapshot predecessor")
    _require_sha256(
        payload["registry_document_identity_sha256"],
        label="revocation-snapshot registry_document_identity_sha256")
    _validate_revocations(payload["revocations"])
    return sequence


def _validate_source_ref_key(value: Any) -> None:
    """验证固定十一项 source_ref_key，不导入 SourceRef 或任何 Python 类布局。"""
    if type(value) is not list or len(value) != GOVERNANCE_SCHEMA_SOURCE_REF_KEY_LENGTH:
        _fail_scalar("source_ref_key 必须是固定十一项 u63 array")
    fields = tuple(
        _require_u63(item, label=f"source_ref_key[{index}]")
        for index, item in enumerate(value))
    if fields[0] == 0 or fields[1] == 0:
        _fail_scalar("source_ref_key source_kind/source_id 必须为正数")
    if fields[3:7] != (0, 0, 0, 1):
        _fail_scalar("source_ref_key owner/visibility 四元组必须为 [0,0,0,1]")


def _validate_upstream_digest(payload: dict[str, Any]) -> None:
    """冻结 NONE/SHA1/SHA256 与 digest hex 长度的一一对应。"""
    algorithm = _require_text(
        payload["upstream_digest_algorithm"], label="upstream_digest_algorithm")
    digest = _require_text(payload["upstream_digest_hex"], label="upstream_digest_hex")
    expected_length = {"NONE": 0, "SHA1": 40, "SHA256": 64}.get(algorithm)
    if expected_length is None:
        _fail_scalar("upstream_digest_algorithm 未注册")
    if (len(digest) != expected_length
            or any(character not in _ASCII_HEX_LOWER for character in digest)):
        _fail_scalar("upstream_digest_hex 与 algorithm 不一致")


def _validate_source_snapshot_declaration(payload: dict[str, Any]) -> int:
    """验证 source declaration v1 的 path-free payload 字段，不读 source bytes。"""
    _require_exact_fields(
        payload, _SOURCE_SNAPSHOT_DECLARATION_FIELDS,
        label="source-snapshot-declaration")
    _unused_key_id, sequence = _require_common(
        payload, kind=GOVERNANCE_WIRE_SOURCE_SNAPSHOT_DECLARATION)
    _require_key_id(payload["control_domain"], label="declaration control_domain")
    for field in (
            "ingest_code_identity_sha256",
            "license_review_artifact_identity_sha256",
            "metadata_sha256",
            "registry_document_identity_sha256",
            "revocation_document_identity_sha256",
            "source_file_sha256",
            "transform_code_identity_sha256",
    ):
        _require_sha256(payload[field], label=f"declaration {field}")
    _require_predecessor(
        payload["predecessor_declaration_identity_sha256"], sequence=sequence,
        label="declaration predecessor")
    for field in ("metadata_byte_count", "source_file_byte_count"):
        _require_u63(payload[field], label=f"declaration {field}", positive=True)
    for field in ("metadata_id", "snapshot_id", "source_file_id", "source_key"):
        _require_opaque_id(payload[field], label=f"declaration {field}")
    _require_strict_https_locator(payload["official_uri"], label="declaration official_uri")
    _require_license_id(payload["license_id"], label="declaration license_id")
    _validate_source_ref_key(payload["source_ref_key"])
    _validate_upstream_digest(payload)
    return sequence


def _validate_payload(payload: dict[str, Any], *, kind: str) -> int:
    """按 document kind 选择唯一冻结 schema，拒绝 annotation 或未来扩展。"""
    if kind == GOVERNANCE_WIRE_ROOT_REGISTRY:
        return _validate_root_registry(payload)
    if kind == GOVERNANCE_WIRE_REVOCATION_SNAPSHOT:
        return _validate_revocation_snapshot(payload)
    if kind == GOVERNANCE_WIRE_SOURCE_SNAPSHOT_DECLARATION:
        return _validate_source_snapshot_declaration(payload)
    _fail_exact("G0b/G0c schema reference 不支持该 document kind")


# object-model: value; representation=struct; interop=GOV-CJSON-1
@dataclass(frozen=True, slots=True)
class ConversationHeldOutV4GovernanceSchemaDocument:
    """已完成 schema 解析、仍完全未验签的不可变 raw-value 视图。"""

    kind: str
    key_id: str
    sequence: int
    canonical_signed_payload: bytes
    detached_signature: tuple[int, ...]
    domain_prefix: bytes
    message: bytes
    document_identity: tuple[int, ...]
    status: str = GOVERNANCE_SCHEMA_STATUS_REFERENCE_ONLY

    def __post_init__(self) -> None:
        """从 canonical bytes 回读所有语义字段，拒绝 direct host-object 漂移。"""
        if (type(self.kind) is not str
                or type(self.key_id) is not str
                or type(self.sequence) is not int
                or type(self.canonical_signed_payload) is not bytes
                or not self.canonical_signed_payload
                or type(self.domain_prefix) is not bytes
                or type(self.message) is not bytes
                or type(self.status) is not str
                or self.status != GOVERNANCE_SCHEMA_STATUS_REFERENCE_ONLY):
            raise ValueError("G0 governance schema document 基础结构非法")
        payload = parse_gov_cjson(self.canonical_signed_payload)
        payload_kind = payload.get("kind") if type(payload) is dict else None
        if type(payload_kind) is not str:
            raise ValueError("G0 governance schema document 缺少 kind")
        parsed_sequence = _validate_payload(payload, kind=payload_kind)
        parsed_key_id = _require_key_id(
            payload["key_id"], label="schema document key_id")
        if (self.kind != payload_kind
                or self.key_id != parsed_key_id
                or self.sequence != parsed_sequence):
            raise ValueError("G0 governance schema document scalar 漂移")
        if (type(self.detached_signature) is not tuple
                or len(self.detached_signature)
                != GOVERNANCE_WIRE_ED25519_SIGNATURE_BYTES
                or any(type(item) is not int or item < 0 or item > 255
                       for item in self.detached_signature)):
            raise ValueError("G0 governance schema document signature bytes 非法")
        expected_domain = governance_wire_domain_prefix(self.kind)
        if self.domain_prefix != expected_domain:
            raise ValueError("G0 governance schema document domain 漂移")
        expected_message = expected_domain + self.canonical_signed_payload
        if self.message != expected_message:
            raise ValueError("G0 governance schema document message 漂移")
        expected_identity = tuple(hashlib.sha256(expected_message).digest())
        if (type(self.document_identity) is not tuple
                or len(self.document_identity) != len(expected_identity)
                or any(type(item) is not int or item < 0 or item > 255
                       for item in self.document_identity)
                or self.document_identity != expected_identity):
            raise ValueError("G0 governance schema document identity 漂移")


def _document_from_wire(
        envelope: ConversationHeldOutV4GovernanceWireEnvelope,
        ) -> ConversationHeldOutV4GovernanceSchemaDocument:
    """从 wire envelope 重读并验证专属 schema，再只投影冻结原始字段。"""
    payload = parse_gov_cjson(envelope.canonical_signed_payload)
    sequence = _validate_payload(payload, kind=envelope.kind)
    return ConversationHeldOutV4GovernanceSchemaDocument(
        kind=envelope.kind,
        key_id=envelope.key_id,
        sequence=sequence,
        canonical_signed_payload=envelope.canonical_signed_payload,
        detached_signature=envelope.detached_signature,
        domain_prefix=envelope.domain_prefix,
        message=envelope.message,
        document_identity=envelope.document_identity,
    )


def parse_governance_schema_document(
        payload: bytes,
        ) -> ConversationHeldOutV4GovernanceSchemaDocument:
    """解析任一已注册 G0b/G0c document；不做验签、链或授权判断。"""
    return _document_from_wire(parse_governance_wire_envelope(payload))


def parse_root_registry_schema_document(
        payload: bytes,
        ) -> ConversationHeldOutV4GovernanceSchemaDocument:
    """解析精确 root-registry schema，拒绝其他 kind 的跨域重放。"""
    document = parse_governance_schema_document(payload)
    if document.kind != GOVERNANCE_WIRE_ROOT_REGISTRY:
        _fail_exact("期待 root-registry，实际 document kind 不一致")
    return document


def parse_revocation_snapshot_schema_document(
        payload: bytes,
        ) -> ConversationHeldOutV4GovernanceSchemaDocument:
    """解析精确 revocation-snapshot schema，仍不做 registry 链验证。"""
    document = parse_governance_schema_document(payload)
    if document.kind != GOVERNANCE_WIRE_REVOCATION_SNAPSHOT:
        _fail_exact("期待 revocation-snapshot，实际 document kind 不一致")
    return document


def parse_source_snapshot_declaration_schema_document(
        payload: bytes,
        ) -> ConversationHeldOutV4GovernanceSchemaDocument:
    """解析精确 source declaration schema，绝不读取其指向的 source/metadata。"""
    document = parse_governance_schema_document(payload)
    if document.kind != GOVERNANCE_WIRE_SOURCE_SNAPSHOT_DECLARATION:
        _fail_exact("期待 source-snapshot-declaration，实际 document kind 不一致")
    return document


__all__ = [
    "ConversationHeldOutV4GovernanceSchemaDocument",
    "ConversationHeldOutV4GovernanceSchemaError",
    "GOVERNANCE_SCHEMA_OK",
    "GOVERNANCE_SCHEMA_REJECT_EXACT_FIELDS",
    "GOVERNANCE_SCHEMA_REJECT_SCALAR",
    "GOVERNANCE_SCHEMA_SOURCE_REF_KEY_LENGTH",
    "GOVERNANCE_SCHEMA_STATUS_REFERENCE_ONLY",
    "GOVERNANCE_SCHEMA_ZERO_SHA256",
    "parse_governance_schema_document",
    "parse_revocation_snapshot_schema_document",
    "parse_root_registry_schema_document",
    "parse_source_snapshot_declaration_schema_document",
]
