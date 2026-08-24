"""DLG-05 v4 G0b-1 portable governance chain-shape 专项。"""
from __future__ import annotations

import ast
from copy import deepcopy
import inspect
import json
from pathlib import Path

import pytest

import pure_integer_ai.experiments.conversation_heldout_v4_governance_chain as chain_module
from pure_integer_ai.experiments.conversation_heldout_v4_governance_chain import (
    ConversationHeldOutV4GovernanceChainError,
    GOVERNANCE_CHAIN_MAX_DOCUMENTS_TOTAL,
    GOVERNANCE_CHAIN_MAX_INPUT_BYTES,
    GOVERNANCE_CHAIN_OK,
    GOVERNANCE_CHAIN_REJECT_DECLARATION_CHAIN,
    GOVERNANCE_CHAIN_REJECT_DECLARATION_REGISTRY_BINDING,
    GOVERNANCE_CHAIN_REJECT_DECLARATION_REVOCATION_BINDING,
    GOVERNANCE_CHAIN_REJECT_INPUT_COLLECTION,
    GOVERNANCE_CHAIN_REJECT_REGISTRY_CHAIN,
    GOVERNANCE_CHAIN_REJECT_REVOCATION_CHAIN,
    GOVERNANCE_CHAIN_REJECT_REVOCATION_REGISTRY_BINDING,
    GOVERNANCE_CHAIN_REJECT_REVOCATION_SET_OR_EFFECTIVE_SEQUENCE,
    GOVERNANCE_CHAIN_STATUS_REFERENCE_ONLY,
    validate_governance_chain_shape,
)
from pure_integer_ai.experiments.conversation_heldout_v4_governance_schema import (
    ConversationHeldOutV4GovernanceSchemaError,
    GOVERNANCE_SCHEMA_REJECT_EXACT_FIELDS,
)
from pure_integer_ai.experiments.conversation_heldout_v4_governance_wire import (
    encode_governance_wire_envelope,
    parse_governance_wire_envelope,
)


_FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "gov_g0b_chain_shape_v1_conformance.json")
_ZERO_SIGNATURE = (0,) * 64


def _corpus() -> dict[str, object]:
    """读取公开、零签名、无 production root 的 chain-shape corpus。"""
    value = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _collections(corpus: dict[str, object]) -> dict[str, list[dict[str, object]]]:
    """复制 fixture 的 raw payload，测试 helper 本身不进入 production protocol。"""
    raw = corpus["reference_collections"]
    assert isinstance(raw, dict)
    result: dict[str, list[dict[str, object]]] = {}
    for name in ("registry", "revocation", "declaration"):
        collection = raw[name]
        assert isinstance(collection, list)
        copied: list[dict[str, object]] = []
        for payload in collection:
            assert isinstance(payload, dict)
            copied.append(deepcopy(payload))
        result[name] = copied
    return result


def _encode_collection(payloads: list[dict[str, object]]) -> tuple[bytes, ...]:
    """以零 detached signature 物化 fixture 的唯一 canonical envelope bytes。"""
    return tuple(
        encode_governance_wire_envelope(payload, _ZERO_SIGNATURE)
        for payload in payloads)


def _encoded_collections(
        collections: dict[str, list[dict[str, object]]],
        ) -> tuple[tuple[bytes, ...], tuple[bytes, ...], tuple[bytes, ...]]:
    """按 API 的三组 bytes 输入形状编码 fixture 或 mutation。"""
    return (
        _encode_collection(collections["registry"]),
        _encode_collection(collections["revocation"]),
        _encode_collection(collections["declaration"]),
    )


def _path_parent(value: object, path: list[object]) -> tuple[object, object]:
    """定位公开 mutation vector 指定的对象或 array 父节点。"""
    assert path
    current = value
    for component in path[:-1]:
        if isinstance(current, dict):
            current = current[component]
        else:
            assert isinstance(current, list) and isinstance(component, int)
            current = current[component]
    return current, path[-1]


def _apply_fixture_rejection(
        collections: dict[str, list[dict[str, object]]],
        vector: dict[str, object],
        ) -> None:
    """应用 corpus 显式 mutation，不把 mutation 规则写进 production resolver。"""
    collection_name = vector["collection"]
    index = vector["index"]
    path = vector["path"]
    operation = vector["operation"]
    assert isinstance(collection_name, str)
    assert isinstance(index, int)
    assert isinstance(path, list)
    assert isinstance(operation, str)
    target = collections[collection_name][index]
    parent, leaf = _path_parent(target, path)
    if operation == "set":
        assert isinstance(parent, dict) and isinstance(leaf, str)
        parent[leaf] = deepcopy(vector["value"])
        return
    if operation == "copy":
        source_collection = vector["source_collection"]
        source_index = vector["source_index"]
        source_path = vector["source_path"]
        assert isinstance(source_collection, str)
        assert isinstance(source_index, int)
        assert isinstance(source_path, list)
        source_parent, source_leaf = _path_parent(
            collections[source_collection][source_index], source_path)
        assert isinstance(source_parent, dict) and isinstance(source_leaf, str)
        assert isinstance(parent, dict) and isinstance(leaf, str)
        parent[leaf] = deepcopy(source_parent[source_leaf])
        return
    raise AssertionError(f"未知 chain-shape mutation operation: {operation}")


def _identity_hex(payload: dict[str, object]) -> str:
    """只供测试重建 mutation 后 predecessor，production 不消费该 helper。"""
    envelope = parse_governance_wire_envelope(
        encode_governance_wire_envelope(payload, _ZERO_SIGNATURE))
    return bytes(envelope.document_identity).hex()


def test_public_chain_shape_vector_is_order_independent_and_identity_only():
    """完整公开 vector 要冻结三条 head identity，输入排列不改变结果。"""
    corpus = _corpus()
    assert corpus["profile"] == "GOV-CJSON-1"
    assert (corpus["chain_shape_reference_status"]
            == GOVERNANCE_CHAIN_STATUS_REFERENCE_ONLY)
    assert corpus["signature_semantics"] == "ZERO_BYTES_UNVERIFIED"
    collections = _collections(corpus)
    registry, revocation, declaration = _encoded_collections(collections)

    identities = validate_governance_chain_shape(
        tuple(reversed(registry)), tuple(reversed(revocation)),
        tuple(reversed(declaration)))
    expected = corpus["expected_head_identities_sha256_hex"]
    assert isinstance(expected, list)
    assert [bytes(identity).hex() for identity in identities] == expected
    assert all(
        len(identity) == 32
        and all(type(byte) is int and 0 <= byte <= 255 for byte in identity)
        for identity in identities)
    assert type(identities) is tuple
    assert not any("capability" in name.lower() for name in chain_module.__all__)
    assert not hasattr(chain_module, "verify")
    assert not hasattr(chain_module, "VALID")


@pytest.mark.parametrize("vector", _corpus()["chain_shape_rejections"])
def test_public_chain_shape_rejections_are_fail_closed_and_stably_coded(
        vector: object,
        ):
    """公开 predecessor/scope/cumulative 反例必须严格映射到合同 code。"""
    assert isinstance(vector, dict)
    collections = _collections(_corpus())
    _apply_fixture_rejection(collections, vector)
    registry, revocation, declaration = _encoded_collections(collections)
    with pytest.raises(ConversationHeldOutV4GovernanceChainError) as captured:
        validate_governance_chain_shape(registry, revocation, declaration)
    assert captured.value.code == vector["error_code"]


def test_chain_scope_and_cumulative_rules_reject_without_authorizing_issuers():
    """root key 仅是 structural scope，issuer record 只用于 revocation-key existence。"""
    collections = _collections(_corpus())

    root_scope_drift = deepcopy(collections)
    root_scope_drift["registry"][1]["key_id"] = "root-chain-2"
    with pytest.raises(ConversationHeldOutV4GovernanceChainError) as captured:
        validate_governance_chain_shape(*_encoded_collections(root_scope_drift))
    assert captured.value.code == GOVERNANCE_CHAIN_REJECT_REGISTRY_CHAIN

    root_binding_drift = deepcopy(collections)
    root_binding_drift["revocation"][0]["key_id"] = "root-chain-2"
    with pytest.raises(ConversationHeldOutV4GovernanceChainError) as captured:
        validate_governance_chain_shape(*_encoded_collections(root_binding_drift))
    assert captured.value.code == GOVERNANCE_CHAIN_REJECT_REVOCATION_REGISTRY_BINDING

    unknown_revoked_key = deepcopy(collections)
    unknown_revoked_key["revocation"][1]["revocations"][1][
        "revoked_key_id"] = "zzz-issuer-1"
    with pytest.raises(ConversationHeldOutV4GovernanceChainError) as captured:
        validate_governance_chain_shape(*_encoded_collections(unknown_revoked_key))
    assert captured.value.code == (
        GOVERNANCE_CHAIN_REJECT_REVOCATION_SET_OR_EFFECTIVE_SEQUENCE)

    changed_history = deepcopy(collections)
    changed_history["revocation"][1]["revocations"][0][
        "reason_digest_sha256"] = "f" * 64
    with pytest.raises(ConversationHeldOutV4GovernanceChainError) as captured:
        validate_governance_chain_shape(*_encoded_collections(changed_history))
    assert captured.value.code == (
        GOVERNANCE_CHAIN_REJECT_REVOCATION_SET_OR_EFFECTIVE_SEQUENCE)

    declaration_scope_drift = deepcopy(collections)
    declaration_scope_drift["declaration"][1]["key_id"] = "other-issuer-1"
    with pytest.raises(ConversationHeldOutV4GovernanceChainError) as captured:
        validate_governance_chain_shape(
            *_encoded_collections(declaration_scope_drift))
    assert captured.value.code == GOVERNANCE_CHAIN_REJECT_DECLARATION_CHAIN


def test_historical_declaration_must_bind_an_identity_in_selected_revocation_chain():
    """旧 declaration 可绑定链内旧 snapshot，但不能指向 collection 外 identity。"""
    collections = _collections(_corpus())
    collections["declaration"][0]["revocation_document_identity_sha256"] = "f" * 64
    collections["declaration"][1][
        "predecessor_declaration_identity_sha256"] = _identity_hex(
            collections["declaration"][0])
    registry, revocation, declaration = _encoded_collections(collections)
    with pytest.raises(ConversationHeldOutV4GovernanceChainError) as captured:
        validate_governance_chain_shape(registry, revocation, declaration)
    assert captured.value.code == GOVERNANCE_CHAIN_REJECT_DECLARATION_REVOCATION_BINDING


def test_input_collection_budget_duplicate_and_host_values_fail_closed():
    """三组都必须是有界 canonical bytes collection，重复 identity 也不能被吞掉。"""
    collections = _collections(_corpus())
    registry, revocation, declaration = _encoded_collections(collections)

    invalid_inputs = (
        ((), revocation, declaration),
        (("not-bytes",), revocation, declaration),
        (registry + registry[:1], revocation, declaration),
        (registry + (registry[0],) * GOVERNANCE_CHAIN_MAX_DOCUMENTS_TOTAL,
         revocation, declaration),
        ((b"x" * (GOVERNANCE_CHAIN_MAX_INPUT_BYTES + 1),),
         revocation, declaration),
    )
    for case in invalid_inputs:
        with pytest.raises(ConversationHeldOutV4GovernanceChainError) as captured:
            validate_governance_chain_shape(*case)
        assert captured.value.code == GOVERNANCE_CHAIN_REJECT_INPUT_COLLECTION


def test_multiple_invalid_documents_have_order_independent_witness_code():
    """同组的多个 wire/schema 失败按 raw bytes 固定选择，不取决于 caller 排列。"""
    collections = _collections(_corpus())
    collections["registry"][0]["unregistered_extension"] = 1
    issuers = collections["registry"][1]["issuers"]
    assert isinstance(issuers, list)
    issuers[0], issuers[1] = issuers[1], issuers[0]
    registry, revocation, declaration = _encoded_collections(collections)

    observed_codes: list[int] = []
    for candidate_registry in (registry, tuple(reversed(registry))):
        with pytest.raises((
                ConversationHeldOutV4GovernanceChainError,
                ConversationHeldOutV4GovernanceSchemaError,
        )) as captured:
            validate_governance_chain_shape(
                candidate_registry, revocation, declaration)
        observed_codes.append(captured.value.code)

    assert observed_codes == [GOVERNANCE_SCHEMA_REJECT_EXACT_FIELDS] * 2


def test_chain_reference_boundary_has_no_verifier_or_host_transport_dependencies():
    """G0b-1 只能依赖 wire/schema reference，不能夹带密码、I/O 或 capability。"""
    source = inspect.getsource(chain_module)
    tree = ast.parse(source)
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    assert imported_roots <= {"__future__", "typing", "pure_integer_ai"}
    assert not ({
        "cryptography", "nacl", "subprocess", "ssl", "socket", "pathlib", "os",
        "sqlite3", "urllib", "requests", "hashlib",
    } & imported_roots)
    called_names = {
        node.func.id for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "open" not in called_names
    assert "verify" not in called_names
    assert "sign" not in called_names
    assert not any("capability" in name.lower() for name in chain_module.__all__)


def test_chain_codes_are_only_the_contract_codes_used_by_this_slice():
    """本 slice 只注册 0 与其实际负责的 105/108/109/110/111/112/114/117。"""
    assert GOVERNANCE_CHAIN_OK == 0
    assert {
        GOVERNANCE_CHAIN_REJECT_REGISTRY_CHAIN,
        GOVERNANCE_CHAIN_REJECT_REVOCATION_REGISTRY_BINDING,
        GOVERNANCE_CHAIN_REJECT_REVOCATION_CHAIN,
        GOVERNANCE_CHAIN_REJECT_REVOCATION_SET_OR_EFFECTIVE_SEQUENCE,
        GOVERNANCE_CHAIN_REJECT_DECLARATION_REGISTRY_BINDING,
        GOVERNANCE_CHAIN_REJECT_DECLARATION_REVOCATION_BINDING,
        GOVERNANCE_CHAIN_REJECT_DECLARATION_CHAIN,
        GOVERNANCE_CHAIN_REJECT_INPUT_COLLECTION,
    } == {105, 108, 109, 110, 111, 112, 114, 117}
