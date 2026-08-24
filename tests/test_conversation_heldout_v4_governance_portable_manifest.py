"""DLG-05 v4 G0 portable conformance manifest family 专项。

消费者只经 GOV-CJSON-1 parser 读取 canonical index/page，并直接读取 index hash
钉住的 raw bytes。大 vector 不能嵌入单个 GOV-CJSON-1 string/object 是协议预算，
故它们是受 basename whitelist 约束的 ``.bin``，从不由 JSON、encoder 或 mutation
recipe 重建。本测试不是另一门语言实现的证明，也不涉及验签、root pin、capability、
private data 或网络。
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable

import pytest

from pure_integer_ai.experiments.conversation_heldout_v4_governance_chain import (
    ConversationHeldOutV4GovernanceChainError,
    validate_governance_chain_shape,
)
from pure_integer_ai.experiments.conversation_heldout_v4_governance_schema import (
    ConversationHeldOutV4GovernanceSchemaError,
    parse_governance_schema_document,
)
from pure_integer_ai.experiments.conversation_heldout_v4_governance_wire import (
    ConversationHeldOutV4GovernanceWireError,
    GOVERNANCE_WIRE_MAX_U63,
    encode_governance_wire_envelope,
    encode_gov_cjson,
    parse_governance_wire_envelope,
    parse_gov_cjson,
)


_FIXTURE_DIR = Path(__file__).parent / "fixtures"
_MANIFEST_NAME = "gov_g0_portable_conformance_manifest_v1.json"
_SIDECAR_NAME = "gov_g0_portable_conformance_manifest_v1.sha256"
_PAGE_PREFIX = "gov_g0_portable_conformance_manifest_v1_"
_RAW_PREFIX = "gov_g0_portable_conformance_manifest_v1_raw_"
_EXPECTED_PAGE_ORDER = (
    ("gov_g0_portable_conformance_manifest_v1_wire.json", "wire"),
    ("gov_g0_portable_conformance_manifest_v1_schema_positive.json", "schema-positive"),
    ("gov_g0_portable_conformance_manifest_v1_schema_negative.json", "schema-negative"),
    ("gov_g0_portable_conformance_manifest_v1_chain_chain-reference-reversed-input-v1.json", "chain"),
    ("gov_g0_portable_conformance_manifest_v1_chain_registry-successor-predecessor-break.json", "chain"),
    ("gov_g0_portable_conformance_manifest_v1_chain_revocation-foreign-registry-binding.json", "chain"),
    ("gov_g0_portable_conformance_manifest_v1_chain_revocation-successor-predecessor-break.json", "chain"),
    ("gov_g0_portable_conformance_manifest_v1_chain_revocation-successor-must-add-record.json", "chain"),
    ("gov_g0_portable_conformance_manifest_v1_chain_declaration-foreign-registry-binding.json", "chain"),
    ("gov_g0_portable_conformance_manifest_v1_chain_declaration-head-must-bind-selected-revocation.json", "chain"),
    ("gov_g0_portable_conformance_manifest_v1_chain_declaration-successor-predecessor-break.json", "chain"),
    ("gov_g0_portable_conformance_manifest_v1_chain_registry-two-invalid-order-a.json", "chain"),
    ("gov_g0_portable_conformance_manifest_v1_chain_registry-two-invalid-order-b.json", "chain"),
)
_EXPECTED_RAW_ARTIFACT_NAMES = (
    "gov_g0_portable_conformance_manifest_v1_raw_array-elements-1024.bin",
    "gov_g0_portable_conformance_manifest_v1_raw_array-elements-1025.bin",
    "gov_g0_portable_conformance_manifest_v1_raw_string-decoded-4096.bin",
    "gov_g0_portable_conformance_manifest_v1_raw_string-decoded-4097.bin",
    "gov_g0_portable_conformance_manifest_v1_raw_string-lexical-4096.bin",
    "gov_g0_portable_conformance_manifest_v1_raw_string-lexical-4097.bin",
    "gov_g0_portable_conformance_manifest_v1_raw_document-bytes-65536.bin",
    "gov_g0_portable_conformance_manifest_v1_raw_document-bytes-65537.bin",
)
_SOURCE_SHA256 = {
    "gov_cjson_v1_conformance.json": (
        "0e1a5cbf1269f675873cfacc28a4d7f56af4ed991146cdb7666cacedc517f85f"),
    "gov_g0b_g0c_schema_v1_conformance.json": (
        "375145465ef18914d95b1908a8b40f6c6f425f753580f100e2c6c2bcc0a85433"),
    "gov_g0b_chain_shape_v1_conformance.json": (
        "261b42c2ee47b0cf189b048030b6b367770e4051e83ddab6362b7b4dc8a05a51"),
}
_HEX_CHARS = frozenset("0123456789abcdef")
_BASENAME_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789._-")
_RECIPE_KEYS = frozenset({
    "base_case",
    "indexes",
    "operation",
    "path",
    "reference_collections",
    "schema_rejections",
    "signed_payload",
    "source_collection",
    "source_index",
    "source_path",
    "syntax_rejections",
    "value",
})
_PROTOCOL_ERRORS = (
    ConversationHeldOutV4GovernanceWireError,
    ConversationHeldOutV4GovernanceSchemaError,
    ConversationHeldOutV4GovernanceChainError,
)


def _sha256_hex(payload: bytes) -> str:
    """返回固定 lowercase SHA-256 text，供 artifact identity 核对。"""
    return hashlib.sha256(payload).hexdigest()


def _canonical_object(payload: bytes) -> dict[str, Any]:
    """只用 GOV-CJSON-1 parser 读取 object，并核对唯一 canonical bytes。"""
    value = parse_gov_cjson(payload)
    assert type(value) is dict
    assert encode_gov_cjson(value) == payload
    return value


def _assert_basename(
        file_name: object, *, prefix: str, suffix: str,
        ) -> str:
    """固定相对 basename grammar，拒绝 path/drive/动态发现语义。"""
    assert type(file_name) is str
    assert file_name.startswith(prefix)
    assert file_name.endswith(suffix)
    assert ".." not in file_name
    assert "/" not in file_name
    assert "\\" not in file_name
    assert ":" not in file_name
    assert all(character in _BASENAME_CHARS for character in file_name)
    return file_name


def _read_index() -> tuple[dict[str, Any], bytes]:
    """读取 index 与仅含 hash/file-name 的 sidecar，不使用 JSON library。"""
    payload = (_FIXTURE_DIR / _MANIFEST_NAME).read_bytes()
    sidecar = (_FIXTURE_DIR / _SIDECAR_NAME).read_bytes()
    expected_sidecar = (
        f"{_sha256_hex(payload)}  {_MANIFEST_NAME}").encode("ascii")
    assert sidecar == expected_sidecar
    return _canonical_object(payload), payload


def _read_pages(index: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """按 index 固定顺序读取 hash-pinned canonical pages，不枚举 consumer 输入。"""
    entries = index["page_order"]
    assert type(entries) is list
    result: list[tuple[dict[str, Any], dict[str, Any]]] = []
    declared_names: list[str] = []
    for entry in entries:
        assert type(entry) is dict
        assert set(entry) == {"byte_count", "file_name", "page_role", "sha256_hex"}
        file_name = _assert_basename(
            entry["file_name"], prefix=_PAGE_PREFIX, suffix=".json")
        assert type(entry["byte_count"]) is int and entry["byte_count"] > 0
        assert type(entry["page_role"]) is str
        assert type(entry["sha256_hex"]) is str
        payload = (_FIXTURE_DIR / file_name).read_bytes()
        assert len(payload) == entry["byte_count"]
        assert _sha256_hex(payload) == entry["sha256_hex"]
        page = _canonical_object(payload)
        assert page["page_role"] == entry["page_role"]
        assert page["profile"] == "GOV-CJSON-1"
        assert page["version"] == 1
        result.append((entry, page))
        declared_names.append(file_name)
    assert len(declared_names) == len(set(declared_names))
    actual_names = {
        path.name for path in _FIXTURE_DIR.iterdir()
        if path.name.startswith(_PAGE_PREFIX) and path.name.endswith(".json")
    }
    assert actual_names == set(declared_names)
    return result


def _raw_artifacts(index: dict[str, Any]) -> list[tuple[dict[str, Any], bytes]]:
    """只读取 index 白名单中的直接 raw parser bytes，并核对 hash/length。"""
    entries = index["raw_input_artifacts"]
    assert type(entries) is list
    result: list[tuple[dict[str, Any], bytes]] = []
    declared_names: list[str] = []
    for entry in entries:
        assert type(entry) is dict
        assert set(entry) == {
            "byte_count", "expected_code", "file_name", "input_kind", "name",
            "sha256_hex",
        }
        file_name = _assert_basename(
            entry["file_name"], prefix=_RAW_PREFIX, suffix=".bin")
        assert entry["input_kind"] == "gov-cjson-parser"
        assert type(entry["name"]) is str
        assert type(entry["expected_code"]) is int
        assert type(entry["byte_count"]) is int and entry["byte_count"] > 0
        assert type(entry["sha256_hex"]) is str
        payload = (_FIXTURE_DIR / file_name).read_bytes()
        assert len(payload) == entry["byte_count"]
        assert _sha256_hex(payload) == entry["sha256_hex"]
        result.append((entry, payload))
        declared_names.append(file_name)
    assert len(declared_names) == len(set(declared_names))
    actual_names = {
        path.name for path in _FIXTURE_DIR.iterdir()
        if path.name.startswith(_RAW_PREFIX) and path.name.endswith(".bin")
    }
    assert actual_names == set(declared_names)
    return result


def _assert_hex_value(value: object) -> None:
    """验证所有 ``*_hex`` protocol fields 是完整 lowercase direct hex。"""
    if type(value) is str:
        assert len(value) % 2 == 0
        assert all(character in _HEX_CHARS for character in value)
        return
    assert type(value) is list
    for item in value:
        assert type(item) is str
        assert len(item) % 2 == 0
        assert all(character in _HEX_CHARS for character in item)


def _assert_direct_protocol_tree(value: object) -> None:
    """拒绝 recipe/object payload，保证每个 protocol byte 输入已经直接冻结。"""
    if type(value) is int:
        assert 0 <= value <= 9_223_372_036_854_775_807
        return
    if type(value) is str:
        value.encode("ascii")
        return
    if type(value) is list:
        for item in value:
            _assert_direct_protocol_tree(item)
        return
    assert type(value) is dict
    for key, item in value.items():
        assert type(key) is str
        assert key not in _RECIPE_KEYS
        if key.endswith("_hex"):
            _assert_hex_value(item)
        _assert_direct_protocol_tree(item)


def _assert_error_code(action: Callable[[], object], expected_code: object) -> None:
    """只比较跨语言协议的稳定整数 code，不比较 Python exception text/class。"""
    assert type(expected_code) is int
    with pytest.raises(_PROTOCOL_ERRORS) as captured:
        action()
    assert captured.value.code == expected_code


def _page_by_role(
        pages: list[tuple[dict[str, Any], dict[str, Any]]], role: str,
        ) -> dict[str, Any]:
    """从已 hash-pinned pages 选择唯一非 chain role，选择逻辑不参与协议。"""
    matches = [page for entry, page in pages if entry["page_role"] == role]
    assert len(matches) == 1
    return matches[0]


def test_manifest_index_pages_and_raw_artifacts_are_canonical_hash_pinned_and_path_free():
    """index/page/raw family 必须在预算内 canonical，且只允许固定 basename 白名单。"""
    index, payload = _read_index()
    assert len(payload) <= 65_536
    assert set(index) == {
        "artifact_kind", "authoring_fixture_sha256", "page_order", "profile",
        "raw_input_artifacts", "version",
    }
    assert index["artifact_kind"] == (
        "GOVERNANCE_PORTABLE_CONFORMANCE_MANIFEST_INDEX_V1")
    assert index["profile"] == "GOV-CJSON-1"
    assert index["version"] == 1

    source_entries = index["authoring_fixture_sha256"]
    assert type(source_entries) is list
    assert [entry["file_name"] for entry in source_entries] == list(_SOURCE_SHA256)
    for entry in source_entries:
        assert type(entry) is dict
        assert set(entry) == {"file_name", "sha256_hex"}
        file_name = entry["file_name"]
        assert type(file_name) is str and file_name in _SOURCE_SHA256
        assert entry["sha256_hex"] == _SOURCE_SHA256[file_name]
        assert _sha256_hex((_FIXTURE_DIR / file_name).read_bytes()) == entry[
            "sha256_hex"]

    pages = _read_pages(index)
    assert [(entry["file_name"], entry["page_role"]) for entry, _page in pages] == list(
        _EXPECTED_PAGE_ORDER)
    assert len([entry for entry, _page in pages if entry["page_role"] == "chain"]) == 10
    raw_artifacts = _raw_artifacts(index)
    assert len(raw_artifacts) == 8
    assert [entry["file_name"] for entry, _payload in raw_artifacts] == list(
        _EXPECTED_RAW_ARTIFACT_NAMES)
    assert {entry["name"] for entry, _payload in raw_artifacts} == {
        "array-elements-1024", "array-elements-1025",
        "string-decoded-4096", "string-decoded-4097",
        "string-lexical-4096", "string-lexical-4097",
        "document-bytes-65536", "document-bytes-65537",
    }

    _assert_direct_protocol_tree(index)
    for _entry, page in pages:
        _assert_direct_protocol_tree(page)


def test_wire_page_executes_direct_parser_envelope_precedence_and_host_adapter_vectors():
    """wire page 的 bytes 只能直接执行；没有 source payload 或 mutation 重建路径。"""
    index, _payload = _read_index()
    wire = _page_by_role(_read_pages(index), "wire")
    assert set(wire) == {
        "document_precedence_cases", "page_role", "profile", "version",
        "wire_envelope_cases", "wire_host_adapter_cases", "wire_parser_cases",
        "wire_public_crypto_transport_cases",
    }

    envelope_cases = wire["wire_envelope_cases"]
    assert type(envelope_cases) is list and len(envelope_cases) == 5
    assert {case["name"] for case in envelope_cases} == {
        "root-registry-common-fields-v1",
        "envelope-extra-field-precedes-hex",
        "signature-hex-precedes-common",
        "signature-hex-non-string",
        "common-precedes-schema",
    }
    for case in envelope_cases:
        assert type(case) is dict
        physical = bytes.fromhex(case["input_envelope_hex"])
        if case["expected_code"] != 0:
            _assert_error_code(
                lambda physical=physical: parse_governance_wire_envelope(physical),
                case["expected_code"],
            )
            continue
        document = parse_governance_wire_envelope(physical)
        assert document.canonical_signed_payload.hex() == case[
            "canonical_signed_payload_hex"]
        assert document.domain_prefix.hex() == case["domain_prefix_hex"]
        assert document.message.hex() == case["message_hex"]
        assert bytes(document.document_identity).hex() == case[
            "document_identity_sha256_hex"]

    parser_cases = wire["wire_parser_cases"]
    assert type(parser_cases) is list and len(parser_cases) == 20
    assert {case["name"] for case in parser_cases} >= {
        "bom", "outer-whitespace", "key-order", "duplicate-key", "unicode-escape",
        "slash-escape", "leading-zero", "u63-overflow", "float", "non-ascii",
        "u63-max-valid",
        "field-name-empty", "field-name-nonletter", "field-name-invalid-tail",
        "field-name-escaped-key-order", "depth-root-one-16", "depth-root-one-17",
        "object-members-128", "object-members-129", "empty-object",
    }
    for case in parser_cases:
        assert type(case) is dict
        physical = bytes.fromhex(case["input_gov_cjson_hex"])
        if case["expected_code"] == 0:
            assert encode_gov_cjson(parse_gov_cjson(physical)) == physical
        else:
            _assert_error_code(
                lambda physical=physical: parse_gov_cjson(physical),
                case["expected_code"],
            )

    parser_by_name = {case["name"]: case for case in parser_cases}
    max_u63_case = parser_by_name["u63-max-valid"]
    assert max_u63_case["expected_code"] == 0
    assert bytes.fromhex(max_u63_case["input_gov_cjson_hex"]) == (
        b'{"a":9223372036854775807}')
    assert parse_gov_cjson(bytes.fromhex(max_u63_case["input_gov_cjson_hex"])) == {
        "a": GOVERNANCE_WIRE_MAX_U63,
    }
    overflow_case = parser_by_name["u63-overflow"]
    assert overflow_case["expected_code"] == 1
    assert bytes.fromhex(overflow_case["input_gov_cjson_hex"]) == (
        b'{"a":9223372036854775808}')
    _assert_error_code(
        lambda: parse_gov_cjson(bytes.fromhex(
            overflow_case["input_gov_cjson_hex"])),
        overflow_case["expected_code"],
    )

    for entry, physical in _raw_artifacts(index):
        if entry["expected_code"] == 0:
            assert encode_gov_cjson(parse_gov_cjson(physical)) == physical
        else:
            _assert_error_code(
                lambda physical=physical: parse_gov_cjson(physical),
                entry["expected_code"],
            )

    precedence = wire["document_precedence_cases"]
    assert type(precedence) is list
    assert [(case["stage"], case["expected_code"]) for case in precedence] == [
        ("physical", 1), ("envelope", 3), ("hex", 5), ("common", 4),
        ("schema", 101),
    ]
    for case in precedence:
        field = "input_gov_cjson_hex" if case["stage"] == "physical" else (
            "input_envelope_hex")
        physical = bytes.fromhex(case[field])
        _assert_error_code(
            lambda physical=physical: parse_governance_schema_document(physical),
            case["expected_code"],
        )

    host_adapter_cases = wire["wire_host_adapter_cases"]
    assert type(host_adapter_cases) is list and len(host_adapter_cases) == 2
    assert {case["name"] for case in host_adapter_cases} == {
        "host-unsigned-int-array-short-signature",
        "host-unsigned-int-array-out-of-range-256",
    }
    for host_case in host_adapter_cases:
        assert type(host_case) is dict
        assert set(host_case) == {
            "expected_code", "expected_length", "name",
            "signed_payload_canonical_gov_cjson_hex", "unsigned_values",
        }
        assert host_case["expected_code"] == 6
        assert host_case["expected_length"] == 64
        unsigned_values = host_case["unsigned_values"]
        assert type(unsigned_values) is list
        assert all(type(value) is int for value in unsigned_values)
        if host_case["name"] == "host-unsigned-int-array-short-signature":
            assert unsigned_values == [0] * 63
        else:
            assert len(unsigned_values) == host_case["expected_length"]
            assert unsigned_values[-1] == 256
        signed_payload = parse_gov_cjson(bytes.fromhex(
            host_case["signed_payload_canonical_gov_cjson_hex"]))
        assert type(signed_payload) is dict
        _assert_error_code(
            lambda signed_payload=signed_payload, unsigned_values=unsigned_values:
            encode_governance_wire_envelope(
                signed_payload, tuple(unsigned_values)),
            host_case["expected_code"],
        )

    crypto_vectors = wire["wire_public_crypto_transport_cases"]
    assert type(crypto_vectors) is list and len(crypto_vectors) == 3
    for vector in crypto_vectors:
        assert type(vector) is dict
        assert len(bytes.fromhex(vector["public_key_hex"])) == 32
        assert len(bytes.fromhex(vector["signature_hex"])) == 64
        bytes.fromhex(vector["message_hex"])
        assert vector["expected_verdict"] in {0, 1}


def test_schema_pages_execute_all_direct_positive_and_negative_envelopes():
    """schema 正反例必须均以完整 envelope bytes 执行，不依赖 authoring fixture。"""
    index, _payload = _read_index()
    positive_page = _page_by_role(_read_pages(index), "schema-positive")
    negative_page = _page_by_role(_read_pages(index), "schema-negative")
    positive_cases = positive_page["schema_cases"]
    negative_cases = negative_page["schema_cases"]
    assert type(positive_cases) is list and len(positive_cases) == 4
    assert type(negative_cases) is list and len(negative_cases) == 13
    assert {case["name"] for case in positive_cases} == {
        "root-registry-genesis-v1", "revocation-snapshot-genesis-v1",
        "source-snapshot-declaration-genesis-v1",
        "source-snapshot-declaration-max-u63-v1",
    }
    assert {case["name"] for case in negative_cases} == {
        "root-missing-issuers", "root-issuer-order", "root-public-key-duplicate",
        "root-genesis-predecessor-nonzero", "revocation-registry-identity-zero",
        "revocation-record-order", "revocation-extra-field",
        "declaration-missing-transform-identity", "declaration-path-in-opaque-id",
        "declaration-owner-not-public", "declaration-uri-lower-percent-hex",
        "declaration-upstream-digest-mismatch",
        "declaration-successor-zero-predecessor",
    }
    for case in positive_cases:
        assert type(case) is dict and case["expected_code"] == 0
        document = parse_governance_schema_document(bytes.fromhex(
            case["input_envelope_hex"]))
        assert document.canonical_signed_payload.hex() == case[
            "canonical_signed_payload_hex"]
        assert document.domain_prefix.hex() == case["domain_prefix_hex"]
        assert document.message.hex() == case["message_hex"]
        assert bytes(document.document_identity).hex() == case[
            "document_identity_sha256_hex"]
        if case["name"] == "source-snapshot-declaration-max-u63-v1":
            assert document.sequence == GOVERNANCE_WIRE_MAX_U63
            signed_payload = parse_gov_cjson(document.canonical_signed_payload)
            assert signed_payload["metadata_byte_count"] == GOVERNANCE_WIRE_MAX_U63
            assert signed_payload["source_file_byte_count"] == GOVERNANCE_WIRE_MAX_U63
            assert signed_payload["source_ref_key"] == [
                GOVERNANCE_WIRE_MAX_U63,
                GOVERNANCE_WIRE_MAX_U63,
                GOVERNANCE_WIRE_MAX_U63,
                0, 0, 0, 1,
                GOVERNANCE_WIRE_MAX_U63,
                GOVERNANCE_WIRE_MAX_U63,
                GOVERNANCE_WIRE_MAX_U63,
                GOVERNANCE_WIRE_MAX_U63,
            ]
    for case in negative_cases:
        assert type(case) is dict
        physical = bytes.fromhex(case["input_envelope_hex"])
        _assert_error_code(
            lambda physical=physical: parse_governance_schema_document(physical),
            case["expected_code"],
        )


def test_chain_pages_execute_complete_direct_collections_and_freeze_invalid_order_witness():
    """chain page 必须携带三组完整 envelope hex，双无效 registry 顺序返回同码。"""
    index, _payload = _read_index()
    pages = _read_pages(index)
    chain_pages = [page for entry, page in pages if entry["page_role"] == "chain"]
    assert len(chain_pages) == 10
    by_name: dict[str, dict[str, Any]] = {}
    for page in chain_pages:
        assert set(page) == {"chain_cases", "page_role", "profile", "version"}
        cases = page["chain_cases"]
        assert type(cases) is list and len(cases) == 1
        case = cases[0]
        assert type(case) is dict
        for field in (
                "registry_envelopes_hex", "revocation_envelopes_hex",
                "declaration_envelopes_hex"):
            assert type(case[field]) is list and case[field]
        registry = tuple(bytes.fromhex(item) for item in case[
            "registry_envelopes_hex"])
        revocation = tuple(bytes.fromhex(item) for item in case[
            "revocation_envelopes_hex"])
        declaration = tuple(bytes.fromhex(item) for item in case[
            "declaration_envelopes_hex"])
        if case["expected_code"] == 0:
            heads = validate_governance_chain_shape(
                registry, revocation, declaration)
            assert [bytes(identity).hex() for identity in heads] == case[
                "expected_head_identities_sha256_hex"]
        else:
            _assert_error_code(
                lambda registry=registry, revocation=revocation, declaration=declaration:
                validate_governance_chain_shape(
                    registry, revocation, declaration),
                case["expected_code"],
            )
        by_name[case["name"]] = case

    assert len(by_name) == 10
    order_a = by_name["registry-two-invalid-order-a"]
    order_b = by_name["registry-two-invalid-order-b"]
    assert order_a["expected_code"] == order_b["expected_code"] == 101
    assert order_a["registry_envelopes_hex"] == list(reversed(
        order_b["registry_envelopes_hex"]))
    assert order_a["revocation_envelopes_hex"] == order_b[
        "revocation_envelopes_hex"]
    assert order_a["declaration_envelopes_hex"] == order_b[
        "declaration_envelopes_hex"]
