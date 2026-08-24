"""DLG-05 v4 G0a-0 portable governance wire 专项。"""
from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path

import pytest

import pure_integer_ai.experiments.conversation_heldout_v4_governance_wire as wire_module
from pure_integer_ai.experiments.conversation_heldout_v4_governance_wire import (
    ConversationHeldOutV4GovernanceWireError,
    GOV_CJSON_PROFILE,
    GOVERNANCE_WIRE_ED25519_PUBLIC_KEY_BYTES,
    GOVERNANCE_WIRE_ED25519_SIGNATURE_BYTES,
    GOVERNANCE_WIRE_MAX_DOCUMENT_BYTES,
    GOVERNANCE_WIRE_MAX_U63,
    GOVERNANCE_WIRE_REJECT_COMMON_PAYLOAD,
    GOVERNANCE_WIRE_REJECT_ENVELOPE,
    GOVERNANCE_WIRE_REJECT_HEX,
    GOVERNANCE_WIRE_ROOT_REGISTRY,
    GOVERNANCE_WIRE_STATUS_REFERENCE_ONLY,
    encode_governance_wire_envelope,
    encode_gov_cjson,
    governance_wire_domain_prefix,
    parse_governance_wire_envelope,
    parse_gov_cjson,
)


_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "gov_cjson_v1_conformance.json"


def _corpus() -> dict[str, object]:
    """读取公开、无私钥的跨语言 conformance corpus。"""
    value = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _root_payload() -> dict[str, object]:
    """构造只含 G0a-0 公共字段的 canonical root-registry payload。"""
    return {
        "algorithm": "Ed25519",
        "key_id": "root-test-1",
        "kind": GOVERNANCE_WIRE_ROOT_REGISTRY,
        "schema": 1,
        "sequence": 1,
        "version": 1,
    }


def test_public_conformance_case_freezes_exact_wire_bytes_and_identity():
    """Python reference 必须逐字节符合公开 corpus，不调用签名或验签 API。"""
    corpus = _corpus()
    assert corpus["profile"] == GOV_CJSON_PROFILE
    cases = corpus["reference_cases"]
    assert isinstance(cases, list) and len(cases) == 1
    case = cases[0]
    assert isinstance(case, dict)
    signed_payload = case["signed_payload"]
    signature_hex = case["signature_hex"]
    assert isinstance(signed_payload, dict) and isinstance(signature_hex, str)

    encoded = encode_governance_wire_envelope(
        signed_payload, tuple(bytes.fromhex(signature_hex)))
    envelope = parse_governance_wire_envelope(encoded)

    assert encoded.hex() == case["envelope_hex"]
    assert envelope.canonical_signed_payload.hex() == case[
        "canonical_signed_payload_hex"]
    assert envelope.domain_prefix.hex() == case["domain_prefix_hex"]
    assert envelope.message.hex() == case["message_hex"]
    assert bytes(envelope.document_identity).hex() == case[
        "document_identity_sha256_hex"]
    assert envelope.status == GOVERNANCE_WIRE_STATUS_REFERENCE_ONLY
    assert tuple(envelope.detached_signature) == tuple(bytes.fromhex(signature_hex))
    assert tuple(envelope.document_identity) == tuple(hashlib.sha256(
        envelope.message).digest())


@pytest.mark.parametrize("case", _corpus()["syntax_rejections"])
def test_public_conformance_rejections_are_fail_closed_and_stably_coded(case):
    """BOM、空白、排序、escape、number 与 ASCII 漂移必须按固定码拒绝。"""
    assert isinstance(case, dict)
    payload = bytes.fromhex(case["payload_hex"])
    with pytest.raises(ConversationHeldOutV4GovernanceWireError) as captured:
        parse_gov_cjson(payload)
    assert captured.value.code == case["error_code"]


def test_encoder_and_parser_share_one_ascii_canonical_form():
    """encoder 负责排序，parser 不接受同一对象的空白、乱序或宽松表示。"""
    value = {
        "z": [0, "literal/uri?x=1", {"a": "quote\\and\"slash"}],
        "a": 9_223_372_036_854_775_807,
    }
    encoded = encode_gov_cjson(value)
    assert encoded == (
        b'{"a":9223372036854775807,"z":[0,"literal/uri?x=1",'
        b'{"a":"quote\\\\and\\\"slash"}]}')
    assert parse_gov_cjson(encoded) == value
    direct_max_u63 = b'{"a":9223372036854775807}'
    assert parse_gov_cjson(direct_max_u63) == {
        "a": GOVERNANCE_WIRE_MAX_U63,
    }

    for malformed in (
            b'{"z":1,"a":2}', b'{"a": 1}', b'{"a":+1}',
            b'{"a":true}', b'{"a":null}', b'{"a":1e0}',
            b'{"a":"\\b"}', b'{"a":"\\u0061"}',
    ):
        with pytest.raises(ConversationHeldOutV4GovernanceWireError):
            parse_gov_cjson(malformed)


def test_parser_enforces_physical_budgets_before_unbounded_host_objects():
    """超长物理输入、深度、成员数、array 与 string 不能先完整物化。"""
    with pytest.raises(ConversationHeldOutV4GovernanceWireError) as captured:
        parse_gov_cjson(b"{" + b'"a":"' + b"x" * GOVERNANCE_WIRE_MAX_DOCUMENT_BYTES + b'"}')
    assert captured.value.code == wire_module.GOVERNANCE_WIRE_REJECT_BUDGET

    escaped_string = b'{"a":"' + b"\\\\" * (
        wire_module.GOVERNANCE_WIRE_MAX_STRING_BYTES // 2 + 1) + b'"}'
    with pytest.raises(ConversationHeldOutV4GovernanceWireError) as captured:
        parse_gov_cjson(escaped_string)
    assert captured.value.code == wire_module.GOVERNANCE_WIRE_REJECT_BUDGET

    deep = b"{" + b'"a":{' * wire_module.GOVERNANCE_WIRE_MAX_DEPTH + b'"z":0' + b"}" * wire_module.GOVERNANCE_WIRE_MAX_DEPTH
    with pytest.raises(ConversationHeldOutV4GovernanceWireError) as captured:
        parse_gov_cjson(deep)
    assert captured.value.code == wire_module.GOVERNANCE_WIRE_REJECT_BUDGET


def test_encoder_enforces_document_budget_while_writing():
    """host AST 合法但组合超限时，encoder 不能先 join 出完整巨型 payload。"""
    value = {"a": ["x" * wire_module.GOVERNANCE_WIRE_MAX_STRING_BYTES] * (
        wire_module.GOVERNANCE_WIRE_MAX_ARRAY_ELEMENTS)}
    with pytest.raises(ConversationHeldOutV4GovernanceWireError) as captured:
        encode_gov_cjson(value)
    assert captured.value.code == wire_module.GOVERNANCE_WIRE_REJECT_BUDGET


def test_envelope_common_binding_domain_and_direct_constructor_all_fail_closed():
    """detached envelope 只能绑定固定 domain，直接构造也不能绕过公共字段。"""
    payload = _root_payload()
    encoded = encode_governance_wire_envelope(payload, (0,) * 64)
    envelope = parse_governance_wire_envelope(encoded)
    assert governance_wire_domain_prefix(GOVERNANCE_WIRE_ROOT_REGISTRY) == (
        b"PIDSLCA-G0/root-registry/v1\x00")

    wrong_algorithm = dict(payload)
    wrong_algorithm["algorithm"] = "Ed448"
    with pytest.raises(ConversationHeldOutV4GovernanceWireError) as captured:
        encode_governance_wire_envelope(wrong_algorithm, (0,) * 64)
    assert captured.value.code == GOVERNANCE_WIRE_REJECT_COMMON_PAYLOAD

    malformed_signature = dict(payload)
    physical = encode_gov_cjson({
        "signature_hex": "0" * 126,
        "signed_payload": malformed_signature,
    })
    with pytest.raises(ConversationHeldOutV4GovernanceWireError) as captured:
        parse_governance_wire_envelope(physical)
    assert captured.value.code == GOVERNANCE_WIRE_REJECT_HEX

    physical_with_non_string_signature = encode_gov_cjson({
        "signature_hex": 0,
        "signed_payload": payload,
    })
    assert parse_gov_cjson(physical_with_non_string_signature)["signature_hex"] == 0
    with pytest.raises(ConversationHeldOutV4GovernanceWireError) as captured:
        parse_governance_wire_envelope(physical_with_non_string_signature)
    assert captured.value.code == GOVERNANCE_WIRE_REJECT_HEX

    physical_with_extra = encode_gov_cjson({
        "extension": 1,
        "signature_hex": "0" * 128,
        "signed_payload": payload,
    })
    with pytest.raises(ConversationHeldOutV4GovernanceWireError) as captured:
        parse_governance_wire_envelope(physical_with_extra)
    assert captured.value.code == GOVERNANCE_WIRE_REJECT_ENVELOPE

    with pytest.raises(ValueError, match="公共字段漂移"):
        replace(envelope, key_id="other-root")
    with pytest.raises(ValueError, match="message 漂移"):
        replace(envelope, message=b"changed")
    with pytest.raises(ConversationHeldOutV4GovernanceWireError):
        replace(envelope, schema=True)
    with pytest.raises(ConversationHeldOutV4GovernanceWireError):
        replace(envelope, document_identity=(True,) * 32)


def test_public_crypto_vectors_are_only_frozen_transport_not_verification():
    """G0a-0 可封存 RFC public vectors，但不得借此调用或假称 Ed25519 验签。"""
    vectors = _corpus()["ed25519_public_vectors"]
    assert isinstance(vectors, list) and len(vectors) == 3
    for vector in vectors:
        assert isinstance(vector, dict)
        public_key = bytes.fromhex(vector["public_key_hex"])
        signature = bytes.fromhex(vector["signature_hex"])
        assert len(public_key) == GOVERNANCE_WIRE_ED25519_PUBLIC_KEY_BYTES
        assert len(signature) == GOVERNANCE_WIRE_ED25519_SIGNATURE_BYTES
        assert vector["expected_verdict"] in {0, 1}
    assert not hasattr(wire_module, "verify")


def test_wire_core_import_boundary_has_no_verifier_or_host_transport_dependencies():
    """portable core 只可用 SHA-256 identity primitive，不能暗带验签、I/O、网络或 runtime。"""
    source = inspect.getsource(wire_module)
    tree = ast.parse(source)
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    assert imported_roots <= {"__future__", "dataclasses", "hashlib", "typing"}
    assert not ({"cryptography", "nacl", "subprocess", "ssl", "socket", "pathlib", "os", "sqlite3"}
                & imported_roots)
    called_names = {
        node.func.id for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "open" not in called_names
