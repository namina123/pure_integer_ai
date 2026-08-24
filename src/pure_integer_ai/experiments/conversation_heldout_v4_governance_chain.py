"""DLG-05 v4 G0b-1 的跨语言治理链形状 reference。

本模块只接收三组完整 canonical ``GOV-CJSON-1`` envelope bytes，并验证
registry、revocation 与 source declaration 的 sequence、predecessor、scope 及
cumulative-revocation 结构。成功只返回三项 derived document identity；失败只携带
合同已冻结的整数 code。它不验签、不 root-pin、不接受 verdict、也不输出 issuer、
payload、capability 或任何可信资格。

Python 的 tuple、dict、exception 仅是当前宿主表示。可移植语义是输入 canonical bytes、
由固定 domain message 导出的 identity，以及固定整数 code。collection 的传入顺序不属于
语义，链只能由 signed sequence、predecessor 与 derived identity 重建。
"""
from __future__ import annotations

from typing import Any, NoReturn

from pure_integer_ai.experiments.conversation_heldout_v4_governance_schema import (
    ConversationHeldOutV4GovernanceSchemaDocument,
    GOVERNANCE_SCHEMA_STATUS_REFERENCE_ONLY,
    parse_revocation_snapshot_schema_document,
    parse_root_registry_schema_document,
    parse_source_snapshot_declaration_schema_document,
)
from pure_integer_ai.experiments.conversation_heldout_v4_governance_wire import (
    parse_gov_cjson,
)


GOVERNANCE_CHAIN_STATUS_REFERENCE_ONLY = GOVERNANCE_SCHEMA_STATUS_REFERENCE_ONLY
"""本模块的最高状态；不含 signature、trust 或 capability 语义。"""

GOVERNANCE_CHAIN_MAX_DOCUMENTS_TOTAL = 1_024
GOVERNANCE_CHAIN_MAX_INPUT_BYTES = 8_388_608

GOVERNANCE_CHAIN_OK = 0
GOVERNANCE_CHAIN_REJECT_REGISTRY_CHAIN = 105
GOVERNANCE_CHAIN_REJECT_REVOCATION_REGISTRY_BINDING = 108
GOVERNANCE_CHAIN_REJECT_REVOCATION_CHAIN = 109
GOVERNANCE_CHAIN_REJECT_REVOCATION_SET_OR_EFFECTIVE_SEQUENCE = 110
GOVERNANCE_CHAIN_REJECT_DECLARATION_REGISTRY_BINDING = 111
GOVERNANCE_CHAIN_REJECT_DECLARATION_REVOCATION_BINDING = 112
GOVERNANCE_CHAIN_REJECT_DECLARATION_CHAIN = 114
GOVERNANCE_CHAIN_REJECT_INPUT_COLLECTION = 117

_CHAIN_CODES = frozenset({
    GOVERNANCE_CHAIN_REJECT_REGISTRY_CHAIN,
    GOVERNANCE_CHAIN_REJECT_REVOCATION_REGISTRY_BINDING,
    GOVERNANCE_CHAIN_REJECT_REVOCATION_CHAIN,
    GOVERNANCE_CHAIN_REJECT_REVOCATION_SET_OR_EFFECTIVE_SEQUENCE,
    GOVERNANCE_CHAIN_REJECT_DECLARATION_REGISTRY_BINDING,
    GOVERNANCE_CHAIN_REJECT_DECLARATION_REVOCATION_BINDING,
    GOVERNANCE_CHAIN_REJECT_DECLARATION_CHAIN,
    GOVERNANCE_CHAIN_REJECT_INPUT_COLLECTION,
})


# object-model: exception
class ConversationHeldOutV4GovernanceChainError(ValueError):
    """G0b-1 chain-shape 的固定整数拒绝码，异常文字不属于协议值。"""

    def __init__(self, code: int, message: str) -> None:
        """保存合同冻结的 chain-shape code。"""
        if type(code) is not int or code not in _CHAIN_CODES:
            raise ValueError("G0 governance chain 错误码未注册")
        self.code = code
        super().__init__(message)


def _fail(code: int, message: str) -> NoReturn:
    """统一以冻结整数 code fail closed，绝不新增宿主专有结果。"""
    raise ConversationHeldOutV4GovernanceChainError(code, message)


def _require_input_collections(
        registry_documents: Any,
        revocation_documents: Any,
        declaration_documents: Any,
        ) -> tuple[tuple[bytes, ...], tuple[bytes, ...], tuple[bytes, ...]]:
    """冻结三组非空 canonical-envelope collection 的类型与总预算。"""
    collections = (
        registry_documents,
        revocation_documents,
        declaration_documents,
    )
    total_documents = 0
    total_bytes = 0
    for collection in collections:
        if type(collection) is not tuple or not collection:
            _fail(GOVERNANCE_CHAIN_REJECT_INPUT_COLLECTION,
                  "G0 chain collection 必须是非空 bytes tuple")
        total_documents += len(collection)
        if total_documents > GOVERNANCE_CHAIN_MAX_DOCUMENTS_TOTAL:
            _fail(GOVERNANCE_CHAIN_REJECT_INPUT_COLLECTION,
                  "G0 chain collection document 数超过固定预算")
        for document in collection:
            if type(document) is not bytes or not document:
                _fail(GOVERNANCE_CHAIN_REJECT_INPUT_COLLECTION,
                      "G0 chain collection 含非空 canonical bytes 以外的值")
            total_bytes += len(document)
            if total_bytes > GOVERNANCE_CHAIN_MAX_INPUT_BYTES:
                _fail(GOVERNANCE_CHAIN_REJECT_INPUT_COLLECTION,
                      "G0 chain collection byte 数超过固定预算")
    return registry_documents, revocation_documents, declaration_documents


def _portable_parse_order(documents: tuple[bytes, ...]) -> tuple[bytes, ...]:
    """按 unsigned raw-envelope byte 词典序固定同组 parse witness 顺序。

    collection 传入排列不属于协议语义。所有 document 尚未可解析时，不能先依赖
    sequence 或 derived identity 排序；Python ``bytes`` 的词典比较等同于 ``0..255``
    unsigned byte 比较，故可直接作为跨语言合同的 fail-closed witness 顺序。
    """
    return tuple(sorted(documents))


def _signed_payload(
        document: ConversationHeldOutV4GovernanceSchemaDocument,
        ) -> dict[str, Any]:
    """仅从 schema 已冻结的 canonical bytes 回读字段，不接收 caller object。"""
    payload = parse_gov_cjson(document.canonical_signed_payload)
    if type(payload) is not dict:
        raise RuntimeError("G0 chain 内部 schema payload 不是 object")
    return payload


def _identity_hex(
        document: ConversationHeldOutV4GovernanceSchemaDocument,
        ) -> str:
    """把已由 wire/schema 冻结的 identity 转为 payload 比较使用的 lowercase hex。"""
    return bytes(document.document_identity).hex()


def _reject_duplicate_identities(
        registry_documents: tuple[ConversationHeldOutV4GovernanceSchemaDocument, ...],
        revocation_documents: tuple[ConversationHeldOutV4GovernanceSchemaDocument, ...],
        declaration_documents: tuple[ConversationHeldOutV4GovernanceSchemaDocument, ...],
        ) -> None:
    """拒绝三集合中任意重复 derived identity，避免隐式重复输入。"""
    seen: set[tuple[int, ...]] = set()
    for collection in (
            registry_documents, revocation_documents, declaration_documents):
        for document in collection:
            identity = document.document_identity
            if identity in seen:
                _fail(GOVERNANCE_CHAIN_REJECT_INPUT_COLLECTION,
                      "G0 chain collection 含重复 derived identity")
            seen.add(identity)


def _order_complete_chain(
        documents: tuple[ConversationHeldOutV4GovernanceSchemaDocument, ...],
        *, predecessor_field: str, code: int, label: str,
        ) -> tuple[ConversationHeldOutV4GovernanceSchemaDocument, ...]:
    """按 signed sequence 重建完整连续链并核对 predecessor identity。"""
    by_sequence: dict[int, ConversationHeldOutV4GovernanceSchemaDocument] = {}
    for document in documents:
        if document.sequence in by_sequence:
            _fail(code, f"{label} chain 含重复 sequence")
        by_sequence[document.sequence] = document

    ordered: list[ConversationHeldOutV4GovernanceSchemaDocument] = []
    for sequence in range(1, len(documents) + 1):
        document = by_sequence.get(sequence)
        if document is None:
            _fail(code, f"{label} chain sequence 不连续或发生 rollback")
        if ordered:
            predecessor = _signed_payload(document)[predecessor_field]
            if predecessor != _identity_hex(ordered[-1]):
                _fail(code, f"{label} chain predecessor identity 不连续")
        ordered.append(document)
    return tuple(ordered)


def _validate_registry_chain(
        documents: tuple[ConversationHeldOutV4GovernanceSchemaDocument, ...],
        ) -> tuple[ConversationHeldOutV4GovernanceSchemaDocument, ...]:
    """验证同 structural root-key scope 内的完整 registry 链，不执行 root pin。"""
    ordered = _order_complete_chain(
        documents,
        predecessor_field="predecessor_registry_identity_sha256",
        code=GOVERNANCE_CHAIN_REJECT_REGISTRY_CHAIN,
        label="root-registry",
    )
    root_key_id = ordered[0].key_id
    if any(document.key_id != root_key_id for document in ordered[1:]):
        _fail(GOVERNANCE_CHAIN_REJECT_REGISTRY_CHAIN,
              "root-registry chain structural key_id scope 漂移")
    return ordered


def _issuer_key_ids(
        registry_head: ConversationHeldOutV4GovernanceSchemaDocument,
        ) -> set[str]:
    """从绑定 registry 的 canonical bytes 提取已 schema 验证的 issuer key 集合。"""
    payload = _signed_payload(registry_head)
    issuers = payload["issuers"]
    if type(issuers) is not list:
        raise RuntimeError("G0 chain 内部 registry issuers 不是 array")
    result: set[str] = set()
    for issuer in issuers:
        if type(issuer) is not dict:
            raise RuntimeError("G0 chain 内部 issuer record 不是 object")
        issuer_key_id = issuer["issuer_key_id"]
        if type(issuer_key_id) is not str:
            raise RuntimeError("G0 chain 内部 issuer key 不是 text")
        result.add(issuer_key_id)
    return result


def _validate_revocation_chain(
        documents: tuple[ConversationHeldOutV4GovernanceSchemaDocument, ...],
        registry_head: ConversationHeldOutV4GovernanceSchemaDocument,
        ) -> tuple[ConversationHeldOutV4GovernanceSchemaDocument, ...]:
    """验证 registry-bound revocation scope、链连续性和累计 record 不变性。"""
    expected_registry_identity = _identity_hex(registry_head)
    for document in documents:
        payload = _signed_payload(document)
        if (payload["registry_document_identity_sha256"]
                != expected_registry_identity
                or document.key_id != registry_head.key_id):
            _fail(GOVERNANCE_CHAIN_REJECT_REVOCATION_REGISTRY_BINDING,
                  "revocation-snapshot 未精确绑定 registry structural scope")

    ordered = _order_complete_chain(
        documents,
        predecessor_field="predecessor_revocation_identity_sha256",
        code=GOVERNANCE_CHAIN_REJECT_REVOCATION_CHAIN,
        label="revocation-snapshot",
    )
    allowed_issuer_keys = _issuer_key_ids(registry_head)
    previous_records: dict[str, tuple[int, str]] = {}
    for document in ordered:
        payload = _signed_payload(document)
        revocations = payload["revocations"]
        if type(revocations) is not list:
            raise RuntimeError("G0 chain 内部 revocations 不是 array")
        current_records: dict[str, tuple[int, str]] = {}
        for record in revocations:
            if type(record) is not dict:
                raise RuntimeError("G0 chain 内部 revocation record 不是 object")
            revoked_key_id = record["revoked_key_id"]
            effective_sequence = record["effective_declaration_sequence"]
            reason_digest = record["reason_digest_sha256"]
            if (type(revoked_key_id) is not str
                    or type(effective_sequence) is not int
                    or type(reason_digest) is not str):
                raise RuntimeError("G0 chain 内部 revocation scalar 非法")
            if revoked_key_id not in allowed_issuer_keys:
                _fail(GOVERNANCE_CHAIN_REJECT_REVOCATION_SET_OR_EFFECTIVE_SEQUENCE,
                      "revocation record 引用了绑定 registry 不存在的 issuer")
            current_records[revoked_key_id] = (effective_sequence, reason_digest)
        for key_id, record in previous_records.items():
            if current_records.get(key_id) != record:
                _fail(GOVERNANCE_CHAIN_REJECT_REVOCATION_SET_OR_EFFECTIVE_SEQUENCE,
                      "revocation chain 删除或改写历史 cumulative record")
        if (document.sequence > 1
                and len(current_records) <= len(previous_records)):
            _fail(GOVERNANCE_CHAIN_REJECT_REVOCATION_SET_OR_EFFECTIVE_SEQUENCE,
                  "revocation successor 必须严格增加新的 cumulative record")
        previous_records = current_records
    return ordered


def _validate_declaration_chain(
        documents: tuple[ConversationHeldOutV4GovernanceSchemaDocument, ...],
        registry_head: ConversationHeldOutV4GovernanceSchemaDocument,
        revocation_documents: tuple[ConversationHeldOutV4GovernanceSchemaDocument, ...],
        ) -> tuple[ConversationHeldOutV4GovernanceSchemaDocument, ...]:
    """验证 declaration scope/sequence 并绑定选定 revocation collection。"""
    expected_registry_identity = _identity_hex(registry_head)
    for document in documents:
        if (_signed_payload(document)["registry_document_identity_sha256"]
                != expected_registry_identity):
            _fail(GOVERNANCE_CHAIN_REJECT_DECLARATION_REGISTRY_BINDING,
                  "source declaration 未精确绑定 registry head identity")

    ordered = _order_complete_chain(
        documents,
        predecessor_field="predecessor_declaration_identity_sha256",
        code=GOVERNANCE_CHAIN_REJECT_DECLARATION_CHAIN,
        label="source-snapshot-declaration",
    )
    scope = (ordered[0].key_id, ordered[0].kind)
    if any((document.key_id, document.kind) != scope for document in ordered[1:]):
        _fail(GOVERNANCE_CHAIN_REJECT_DECLARATION_CHAIN,
              "source declaration chain key_id/kind scope 漂移")

    revocation_identities = {
        _identity_hex(document) for document in revocation_documents
    }
    for document in ordered:
        if (_signed_payload(document)["revocation_document_identity_sha256"]
                not in revocation_identities):
            _fail(GOVERNANCE_CHAIN_REJECT_DECLARATION_REVOCATION_BINDING,
                  "source declaration 绑定 collection 外的 revocation identity")
    if (_signed_payload(ordered[-1])["revocation_document_identity_sha256"]
            != _identity_hex(revocation_documents[-1])):
        _fail(GOVERNANCE_CHAIN_REJECT_DECLARATION_REVOCATION_BINDING,
              "source declaration head 未绑定 selected revocation head identity")
    return ordered


def validate_governance_chain_shape(
        registry_documents: tuple[bytes, ...],
        revocation_documents: tuple[bytes, ...],
        declaration_documents: tuple[bytes, ...],
        ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """验证三条完整治理链，成功只返回 registry/revocation/declaration head identity。"""
    registry_bytes, revocation_bytes, declaration_bytes = (
        _require_input_collections(
            registry_documents, revocation_documents, declaration_documents))
    registries = tuple(
        parse_root_registry_schema_document(document)
        for document in _portable_parse_order(registry_bytes))
    revocations = tuple(
        parse_revocation_snapshot_schema_document(document)
        for document in _portable_parse_order(revocation_bytes))
    declarations = tuple(
        parse_source_snapshot_declaration_schema_document(document)
        for document in _portable_parse_order(declaration_bytes))
    _reject_duplicate_identities(registries, revocations, declarations)

    ordered_registries = _validate_registry_chain(registries)
    ordered_revocations = _validate_revocation_chain(
        revocations, ordered_registries[-1])
    ordered_declarations = _validate_declaration_chain(
        declarations, ordered_registries[-1], ordered_revocations)
    return (
        ordered_registries[-1].document_identity,
        ordered_revocations[-1].document_identity,
        ordered_declarations[-1].document_identity,
    )


__all__ = [
    "ConversationHeldOutV4GovernanceChainError",
    "GOVERNANCE_CHAIN_MAX_DOCUMENTS_TOTAL",
    "GOVERNANCE_CHAIN_MAX_INPUT_BYTES",
    "GOVERNANCE_CHAIN_OK",
    "GOVERNANCE_CHAIN_REJECT_DECLARATION_CHAIN",
    "GOVERNANCE_CHAIN_REJECT_DECLARATION_REGISTRY_BINDING",
    "GOVERNANCE_CHAIN_REJECT_DECLARATION_REVOCATION_BINDING",
    "GOVERNANCE_CHAIN_REJECT_INPUT_COLLECTION",
    "GOVERNANCE_CHAIN_REJECT_REGISTRY_CHAIN",
    "GOVERNANCE_CHAIN_REJECT_REVOCATION_CHAIN",
    "GOVERNANCE_CHAIN_REJECT_REVOCATION_REGISTRY_BINDING",
    "GOVERNANCE_CHAIN_REJECT_REVOCATION_SET_OR_EFFECTIVE_SEQUENCE",
    "GOVERNANCE_CHAIN_STATUS_REFERENCE_ONLY",
    "validate_governance_chain_shape",
]
