"""DLG-05 v4 G0b/G0c portable governance schema reference 专项。"""
from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import replace
import inspect
import json
from pathlib import Path

import pytest

import pure_integer_ai.experiments.conversation_heldout_v4_governance_schema as schema_module
from pure_integer_ai.experiments.conversation_heldout_v4_governance_schema import (
    ConversationHeldOutV4GovernanceSchemaError,
    GOVERNANCE_SCHEMA_REJECT_EXACT_FIELDS,
    GOVERNANCE_SCHEMA_REJECT_SCALAR,
    GOVERNANCE_SCHEMA_STATUS_REFERENCE_ONLY,
    parse_governance_schema_document,
    parse_revocation_snapshot_schema_document,
    parse_root_registry_schema_document,
    parse_source_snapshot_declaration_schema_document,
)
from pure_integer_ai.experiments.conversation_heldout_v4_governance_wire import (
    ConversationHeldOutV4GovernanceWireError,
    GOVERNANCE_WIRE_ANNOTATION_SOURCE_DECLARATION,
    GOVERNANCE_WIRE_REJECT_SYNTAX,
    encode_governance_wire_envelope,
)


_FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "gov_g0b_g0c_schema_v1_conformance.json")


def _corpus() -> dict[str, object]:
    """读取公开、零签名、无生产 root 的 schema conformance corpus。"""
    value = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _cases_by_name() -> dict[str, dict[str, object]]:
    """按公开 vector name 建立测试读取索引，索引本身不是协议语义。"""
    cases = _corpus()["reference_cases"]
    assert isinstance(cases, list)
    result: dict[str, dict[str, object]] = {}
    for case in cases:
        assert isinstance(case, dict)
        name = case["name"]
        assert isinstance(name, str)
        result[name] = case
    return result


def _encoded_case(case: dict[str, object]) -> bytes:
    """从 fixture 的公开 raw values 重建唯一 GOV-CJSON-1 envelope。"""
    signed_payload = case["signed_payload"]
    signature_hex = case["signature_hex"]
    assert isinstance(signed_payload, dict)
    assert isinstance(signature_hex, str)
    return encode_governance_wire_envelope(
        signed_payload, tuple(bytes.fromhex(signature_hex)))


def _path_parent(value: object, path: list[object]) -> tuple[object, object]:
    """定位 fixture mutation 的父节点；此 helper 不进入 production protocol。"""
    assert path
    current = value
    for component in path[:-1]:
        if isinstance(current, dict):
            current = current[component]
        else:
            assert isinstance(current, list) and isinstance(component, int)
            current = current[component]
    return current, path[-1]


def _mutated_payload(
        case: dict[str, object], rejection: dict[str, object],
        ) -> dict[str, object]:
    """应用 corpus 明示的非协议 mutation，构造 schema fail-closed 反例。"""
    signed_payload = case["signed_payload"]
    path = rejection["path"]
    operation = rejection["operation"]
    assert isinstance(signed_payload, dict)
    assert isinstance(path, list)
    assert isinstance(operation, str)
    result = deepcopy(signed_payload)
    parent, final = _path_parent(result, path)
    if operation == "set":
        assert isinstance(parent, (dict, list))
        parent[final] = deepcopy(rejection["value"])
    elif operation == "drop":
        assert isinstance(parent, dict) and isinstance(final, str)
        del parent[final]
    elif operation == "add":
        assert isinstance(parent, dict) and isinstance(final, str)
        parent[final] = deepcopy(rejection["value"])
    elif operation == "swap":
        indexes = rejection["indexes"]
        assert isinstance(parent, dict) and isinstance(final, str)
        target = parent[final]
        assert (isinstance(target, list) and isinstance(indexes, list)
                and len(indexes) == 2
                and all(isinstance(item, int) for item in indexes))
        left, right = indexes
        target[left], target[right] = target[right], target[left]
    else:
        raise AssertionError(f"未知 corpus mutation operation: {operation}")
    return result


def test_public_schema_vectors_freeze_canonical_bytes_domains_and_identities():
    """三个 document kind 必须重现公开 canonical payload/domain/identity vector。"""
    corpus = _corpus()
    assert corpus["profile"] == "GOV-CJSON-1"
    assert corpus["schema_reference_status"] == GOVERNANCE_SCHEMA_STATUS_REFERENCE_ONLY
    assert corpus["signature_semantics"] == "ZERO_BYTES_UNVERIFIED"
    cases = corpus["reference_cases"]
    assert isinstance(cases, list) and len(cases) == 3

    parsers = {
        "root-registry": parse_root_registry_schema_document,
        "revocation-snapshot": parse_revocation_snapshot_schema_document,
        "source-snapshot-declaration": (
            parse_source_snapshot_declaration_schema_document),
    }
    for case in cases:
        assert isinstance(case, dict)
        encoded = _encoded_case(case)
        document = parse_governance_schema_document(encoded)
        signed_payload = case["signed_payload"]
        assert isinstance(signed_payload, dict)
        kind = signed_payload["kind"]
        assert isinstance(kind, str)
        assert document == parsers[kind](encoded)
        assert document.canonical_signed_payload.hex() == case[
            "canonical_signed_payload_hex"]
        assert document.domain_prefix.hex() == case["domain_prefix_hex"]
        assert document.message == (
            document.domain_prefix + document.canonical_signed_payload)
        assert bytes(document.document_identity).hex() == case[
            "document_identity_sha256_hex"]
        assert document.kind == kind
        assert document.key_id == signed_payload["key_id"]
        assert document.sequence == signed_payload["sequence"]
        assert document.status == GOVERNANCE_SCHEMA_STATUS_REFERENCE_ONLY
        assert tuple(document.detached_signature) == (0,) * 64
        assert not hasattr(document, "payload")
        assert not hasattr(document, "__dict__")


@pytest.mark.parametrize("rejection", _corpus()["schema_rejections"])
def test_public_schema_rejection_vectors_are_fail_closed_and_stably_coded(
        rejection: object,
        ):
    """字段漂移、标量、排序和 genesis 反例必须仅返回冻结整数错误码。"""
    assert isinstance(rejection, dict)
    case_name = rejection["base_case"]
    assert isinstance(case_name, str)
    case = _cases_by_name()[case_name]
    mutated = _mutated_payload(case, rejection)
    encoded = encode_governance_wire_envelope(mutated, (0,) * 64)
    with pytest.raises(ConversationHeldOutV4GovernanceSchemaError) as captured:
        parse_governance_schema_document(encoded)
    assert captured.value.code == rejection["error_code"]


def test_schema_specific_parsers_reject_cross_document_domain_replay():
    """只按目标 kind 读取的 caller 不能把其他 canonical document 当成目标 schema。"""
    cases = _cases_by_name()
    root = _encoded_case(cases["root-registry-genesis-v1"])
    revocation = _encoded_case(cases["revocation-snapshot-genesis-v1"])
    declaration = _encoded_case(cases["source-snapshot-declaration-genesis-v1"])
    for parser, foreign in (
            (parse_root_registry_schema_document, revocation),
            (parse_revocation_snapshot_schema_document, declaration),
            (parse_source_snapshot_declaration_schema_document, root)):
        with pytest.raises(ConversationHeldOutV4GovernanceSchemaError) as captured:
            parser(foreign)
        assert captured.value.code == GOVERNANCE_SCHEMA_REJECT_EXACT_FIELDS


def test_schema_malformed_scalar_containers_never_leak_host_type_errors():
    """schema 字段若被 array/object 替代，必须仍归入固定 102 而非 Python TypeError。"""
    case = _cases_by_name()["root-registry-genesis-v1"]
    payload = deepcopy(case["signed_payload"])
    assert isinstance(payload, dict)
    issuers = payload["issuers"]
    assert isinstance(issuers, list)
    issuer = issuers[0]
    assert isinstance(issuer, dict)
    issuer["role"] = []
    encoded = encode_governance_wire_envelope(payload, (0,) * 64)
    with pytest.raises(ConversationHeldOutV4GovernanceSchemaError) as captured:
        parse_governance_schema_document(encoded)
    assert captured.value.code == GOVERNANCE_SCHEMA_REJECT_SCALAR


def test_schema_rejects_wire_known_but_out_of_scope_annotation_and_keeps_wire_codes():
    """annotation schema 尚未进入本切片，物理 wire 拒绝码也不得被 schema 改写。"""
    annotation = encode_governance_wire_envelope({
        "algorithm": "Ed25519",
        "key_id": "annotation-test-1",
        "kind": GOVERNANCE_WIRE_ANNOTATION_SOURCE_DECLARATION,
        "schema": 1,
        "version": 1,
    }, (0,) * 64)
    with pytest.raises(ConversationHeldOutV4GovernanceSchemaError) as captured:
        parse_governance_schema_document(annotation)
    assert captured.value.code == GOVERNANCE_SCHEMA_REJECT_EXACT_FIELDS

    with pytest.raises(ConversationHeldOutV4GovernanceWireError) as captured:
        parse_governance_schema_document(b" " + _encoded_case(
            _cases_by_name()["root-registry-genesis-v1"]))
    assert captured.value.code == GOVERNANCE_WIRE_REJECT_SYNTAX


def test_schema_document_direct_host_construction_cannot_drift_from_canonical_bytes():
    """直接替换任何投影字段都必须在结构体构造期 fail closed。"""
    document = parse_governance_schema_document(_encoded_case(
        _cases_by_name()["source-snapshot-declaration-genesis-v1"]))
    for changed in (
            {"key_id": "other-issuer"},
            {"sequence": 2},
            {"detached_signature": (0,) * 63},
            {"domain_prefix": b"wrong"},
            {"message": b"wrong"},
            {"document_identity": (0,) * 32},
            {"status": "VERIFIED"},
            {"canonical_signed_payload": b"{}"},
    ):
        with pytest.raises(ValueError):
            replace(document, **changed)


def test_schema_reference_import_boundary_has_no_crypto_or_host_operations():
    """schema core 只依赖 wire/标准库，不得引入 adapter、I/O 或 capability 生产面。"""
    source = inspect.getsource(schema_module)
    tree = ast.parse(source)
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    assert imported_roots <= {
        "__future__", "dataclasses", "hashlib", "typing", "pure_integer_ai"}
    assert not ({
        "cryptography", "nacl", "subprocess", "ssl", "socket", "pathlib", "os",
        "sqlite3", "urllib", "requests",
    } & imported_roots)
    called_names = {
        node.func.id for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "open" not in called_names
    assert not hasattr(schema_module, "verify")
    assert not any("capability" in name.lower() for name in schema_module.__all__)


def test_schema_error_codes_remain_the_only_local_protocol_rejections():
    """schema reference 只注册 contract 指定的 101/102，不能新增宿主专有结果。"""
    assert {
        GOVERNANCE_SCHEMA_REJECT_EXACT_FIELDS,
        GOVERNANCE_SCHEMA_REJECT_SCALAR,
    } == {101, 102}
