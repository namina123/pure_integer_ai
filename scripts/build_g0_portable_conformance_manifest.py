"""物化 G0 跨语言治理 conformance manifest family。

本脚本是作者侧工具：它读取既有公开 fixture，使用当前 reference 仅一次性把
mutation 结果落成完整原始 bytes。产物消费者只读取 canonical GOV-CJSON-1
index/page 或直接读取受 index SHA 约束的 ``.bin``，不需要 Python ``json``、
encoder、mutation recipe 或本脚本。

GOV-CJSON-1 自身有 4,096-byte string / 65,536-byte object 上限。因此大于可嵌
hex 字符串上限的 parser vectors 保持为逐字节 ``.bin``，而大量 chain vector
保持在独立的 canonical page 中。这是协议预算的显式适配，不是 consumer-side
展开或重建规则。
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _PROJECT_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from pure_integer_ai.experiments.conversation_heldout_v4_governance_wire import (  # noqa: E402
    GOVERNANCE_WIRE_MAX_U63,
    encode_governance_wire_envelope,
    encode_gov_cjson,
    parse_gov_cjson,
)
from pure_integer_ai.experiments.conversation_heldout_v4_governance_schema import (  # noqa: E402
    parse_source_snapshot_declaration_schema_document,
)


_FIXTURE_DIR = _PROJECT_ROOT / "tests" / "fixtures"
_ZERO_SIGNATURE = (0,) * 64
_MANIFEST_NAME = "gov_g0_portable_conformance_manifest_v1.json"
_SIDECAR_NAME = "gov_g0_portable_conformance_manifest_v1.sha256"
_PAGE_PREFIX = "gov_g0_portable_conformance_manifest_v1_"
_RAW_PREFIX = "gov_g0_portable_conformance_manifest_v1_raw_"
_PROFILE = "GOV-CJSON-1"
_SOURCE_FIXTURES = (
    (
        "gov_cjson_v1_conformance.json",
        "0e1a5cbf1269f675873cfacc28a4d7f56af4ed991146cdb7666cacedc517f85f",
    ),
    (
        "gov_g0b_g0c_schema_v1_conformance.json",
        "375145465ef18914d95b1908a8b40f6c6f425f753580f100e2c6c2bcc0a85433",
    ),
    (
        "gov_g0b_chain_shape_v1_conformance.json",
        "261b42c2ee47b0cf189b048030b6b367770e4051e83ddab6362b7b4dc8a05a51",
    ),
)


def _sha256_hex(payload: bytes) -> str:
    """返回 payload 的固定 lowercase SHA-256 text。"""
    return hashlib.sha256(payload).hexdigest()


def _read_source_fixture(file_name: str, expected_sha256: str) -> dict[str, Any]:
    """读取冻结的作者侧 fixture，并先拒绝来源字节漂移。"""
    payload = (_FIXTURE_DIR / file_name).read_bytes()
    if _sha256_hex(payload) != expected_sha256:
        raise RuntimeError(f"作者侧 fixture SHA-256 漂移: {file_name}")
    value = json.loads(payload.decode("utf-8"))
    if type(value) is not dict:
        raise RuntimeError(f"作者侧 fixture root 非 object: {file_name}")
    return value


def _canonical_page(value: dict[str, Any]) -> bytes:
    """编码并回读一个受 GOV-CJSON-1 自身预算限制的 page。"""
    payload = encode_gov_cjson(value)
    if parse_gov_cjson(payload) != value:
        raise RuntimeError("canonical page 回读值漂移")
    return payload


def _encode_envelope(payload: dict[str, Any]) -> bytes:
    """用公开零 detached signature 物化唯一完整 envelope bytes。"""
    return encode_governance_wire_envelope(payload, _ZERO_SIGNATURE)


def _path_parent(value: object, path: list[object]) -> tuple[object, object]:
    """定位作者侧 fixture mutation 的父对象；该规则绝不写入产物。"""
    if not path:
        raise RuntimeError("作者侧 mutation path 为空")
    current = value
    for component in path[:-1]:
        if type(current) is dict:
            current = current[component]
        elif type(current) is list and type(component) is int:
            current = current[component]
        else:
            raise RuntimeError("作者侧 mutation path 无法定位")
    return current, path[-1]


def _apply_schema_rejection(
        payload: dict[str, Any], vector: dict[str, Any],
        ) -> dict[str, Any]:
    """只在作者侧应用既有 schema mutation 并返回完整 payload。"""
    result = deepcopy(payload)
    path = vector["path"]
    operation = vector["operation"]
    if type(path) is not list or type(operation) is not str:
        raise RuntimeError("作者侧 schema mutation 格式非法")
    parent, final = _path_parent(result, path)
    if operation == "set":
        if type(parent) not in (dict, list):
            raise RuntimeError("作者侧 schema set 父节点非法")
        parent[final] = deepcopy(vector["value"])
    elif operation == "drop":
        if type(parent) is not dict or type(final) is not str:
            raise RuntimeError("作者侧 schema drop 父节点非法")
        del parent[final]
    elif operation == "add":
        if type(parent) is not dict or type(final) is not str:
            raise RuntimeError("作者侧 schema add 父节点非法")
        parent[final] = deepcopy(vector["value"])
    elif operation == "swap":
        indexes = vector["indexes"]
        if (type(parent) is not dict or type(final) is not str
                or type(indexes) is not list or len(indexes) != 2
                or any(type(index) is not int for index in indexes)):
            raise RuntimeError("作者侧 schema swap 格式非法")
        target = parent[final]
        if type(target) is not list:
            raise RuntimeError("作者侧 schema swap 目标不是 array")
        left, right = indexes
        target[left], target[right] = target[right], target[left]
    else:
        raise RuntimeError(f"未知作者侧 schema mutation: {operation}")
    return result


def _apply_chain_rejection(
        collections: dict[str, list[dict[str, Any]]], vector: dict[str, Any],
        ) -> None:
    """只在作者侧应用既有 chain mutation，不输出 mutation 元数据。"""
    collection_name = vector["collection"]
    index = vector["index"]
    path = vector["path"]
    operation = vector["operation"]
    if (type(collection_name) is not str or type(index) is not int
            or type(path) is not list or type(operation) is not str):
        raise RuntimeError("作者侧 chain mutation 格式非法")
    target = collections[collection_name][index]
    parent, final = _path_parent(target, path)
    if operation == "set":
        if type(parent) is not dict or type(final) is not str:
            raise RuntimeError("作者侧 chain set 父节点非法")
        parent[final] = deepcopy(vector["value"])
        return
    if operation == "copy":
        source_collection = vector["source_collection"]
        source_index = vector["source_index"]
        source_path = vector["source_path"]
        if (type(source_collection) is not str or type(source_index) is not int
                or type(source_path) is not list):
            raise RuntimeError("作者侧 chain copy 来源非法")
        source_parent, source_final = _path_parent(
            collections[source_collection][source_index], source_path)
        if (type(parent) is not dict or type(final) is not str
                or type(source_parent) is not dict
                or type(source_final) is not str):
            raise RuntimeError("作者侧 chain copy 节点非法")
        parent[final] = deepcopy(source_parent[source_final])
        return
    raise RuntimeError(f"未知作者侧 chain mutation: {operation}")


def _copy_collections(source: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """复制三组作者侧 raw payload，确保产物只含新物化的 bytes。"""
    result: dict[str, list[dict[str, Any]]] = {}
    for name in ("registry", "revocation", "declaration"):
        collection = source[name]
        if type(collection) is not list or any(type(item) is not dict for item in collection):
            raise RuntimeError(f"作者侧 chain collection 非法: {name}")
        result[name] = deepcopy(collection)
    return result


def _collection_hexes(
        collections: dict[str, list[dict[str, Any]]],
        ) -> dict[str, list[str]]:
    """把三组 raw payload 物化为完整、不可再解释的 envelope hex。"""
    return {
        f"{name}_envelopes_hex": [
            _encode_envelope(payload).hex()
            for payload in collections[name]
        ]
        for name in ("registry", "revocation", "declaration")
    }


def _parser_depth_vector(depth: int) -> bytes:
    """构造 root depth=1 计数下的 canonical nested-object vector。"""
    if type(depth) is not int or depth < 1:
        raise RuntimeError("作者侧 depth 参数非法")
    return b'{"a":' * (depth - 1) + b"{}" + b"}" * (depth - 1)


def _parser_member_vector(member_count: int) -> bytes:
    """构造按 ASCII field name 严格升序的 object-member 边界 vector。"""
    if type(member_count) is not int or not 0 <= member_count <= 1_000:
        raise RuntimeError("作者侧 member_count 参数非法")
    return b"{" + b",".join(
        f'"k{index:03d}":0'.encode("ascii")
        for index in range(member_count)) + b"}"


def _parser_array_vector(element_count: int) -> bytes:
    """构造只含 u63 零值的 canonical array-element 边界 vector。"""
    if type(element_count) is not int or not 0 <= element_count <= 2_000:
        raise RuntimeError("作者侧 element_count 参数非法")
    return b'{"a":[' + b",".join(b"0" for _ in range(element_count)) + b"]}"


def _parser_string_vector(*, lexical_content_bytes: int, decoded: bool) -> bytes:
    """构造 decoded 或 lexical string budget 边界，quote delimiter 不计入内容。"""
    if type(lexical_content_bytes) is not int or lexical_content_bytes < 0:
        raise RuntimeError("作者侧 string 长度参数非法")
    if decoded:
        content = b"x" * lexical_content_bytes
    else:
        pair_count, remainder = divmod(lexical_content_bytes, 2)
        content = b'\\"' * pair_count + b"x" * remainder
    return b'{"a":"' + content + b'"}'


def _parser_document_vector(total_bytes: int) -> bytes:
    """构造精确总 byte 长度、且不越过局部 string/array budget 的 vector。"""
    if type(total_bytes) is not int or total_bytes not in {65_536, 65_537}:
        raise RuntimeError("作者侧 document byte 边界参数非法")
    element_count = 1_024
    overhead = len(b'{"a":[') + len(b"]}") + (element_count - 1) + 2 * element_count
    content_total = total_bytes - overhead
    quotient, remainder = divmod(content_total, element_count)
    contents = [b"x" * (quotient + 1)] * remainder
    contents.extend([b"x" * quotient] * (element_count - remainder))
    payload = b'{"a":[' + b",".join(
        b'"' + content + b'"' for content in contents) + b"]}"
    if len(payload) != total_bytes:
        raise RuntimeError("作者侧 document vector byte 长度漂移")
    return payload


def _parser_direct_cases(wire_source: dict[str, Any]) -> list[dict[str, Any]]:
    """物化可嵌单页的既有和新增 parser byte vectors。"""
    result: list[dict[str, Any]] = []
    syntax_rejections = wire_source["syntax_rejections"]
    if type(syntax_rejections) is not list:
        raise RuntimeError("作者侧 wire syntax_rejections 非 array")
    for source_case in syntax_rejections:
        if type(source_case) is not dict:
            raise RuntimeError("作者侧 wire syntax case 非 object")
        result.append({
            "expected_code": source_case["error_code"],
            "input_gov_cjson_hex": source_case["payload_hex"],
            "name": source_case["name"],
        })

    extra_cases = (
        ("u63-max-valid", b'{"a":9223372036854775807}', 0),
        ("field-name-empty", b'{"":1}', 1),
        ("field-name-nonletter", b'{"1a":1}', 1),
        ("field-name-invalid-tail", b'{"a/b":1}', 1),
        ("field-name-escaped-key-order", b'{"a\\\\":1,"b":2}', 1),
        ("depth-root-one-16", _parser_depth_vector(16), 0),
        ("depth-root-one-17", _parser_depth_vector(17), 2),
        ("object-members-128", _parser_member_vector(128), 0),
        ("object-members-129", _parser_member_vector(129), 2),
        ("empty-object", b"{}", 0),
    )
    for name, payload, expected_code in extra_cases:
        if len(payload.hex()) > 4_096:
            raise RuntimeError(f"应进入 raw artifact 的 direct case: {name}")
        result.append({
            "expected_code": expected_code,
            "input_gov_cjson_hex": payload.hex(),
            "name": name,
        })
    return result


def _raw_parser_artifacts() -> list[tuple[dict[str, Any], bytes]]:
    """返回不能嵌入 GOV-CJSON string 的直接 parser bytes 与固定元数据。"""
    cases = (
        ("array-elements-1024", _parser_array_vector(1_024), 0),
        ("array-elements-1025", _parser_array_vector(1_025), 2),
        ("string-decoded-4096", _parser_string_vector(
            lexical_content_bytes=4_096, decoded=True), 0),
        ("string-decoded-4097", _parser_string_vector(
            lexical_content_bytes=4_097, decoded=True), 2),
        ("string-lexical-4096", _parser_string_vector(
            lexical_content_bytes=4_096, decoded=False), 0),
        ("string-lexical-4097", _parser_string_vector(
            lexical_content_bytes=4_097, decoded=False), 2),
        ("document-bytes-65536", _parser_document_vector(65_536), 0),
        ("document-bytes-65537", _parser_document_vector(65_537), 2),
    )
    result: list[tuple[dict[str, Any], bytes]] = []
    for name, payload, expected_code in cases:
        if len(payload.hex()) <= 4_096:
            raise RuntimeError(f"应嵌入 page 的 raw case: {name}")
        file_name = f"{_RAW_PREFIX}{name}.bin"
        result.append(({
            "byte_count": len(payload),
            "expected_code": expected_code,
            "file_name": file_name,
            "input_kind": "gov-cjson-parser",
            "name": name,
            "sha256_hex": _sha256_hex(payload),
        }, payload))
    return result


def _wire_envelope_cases(
        wire_source: dict[str, Any], schema_source: dict[str, Any],
        ) -> list[dict[str, Any]]:
    """物化 wire positive 与 3/4/5 direct rejection envelopes。"""
    wire_references = wire_source["reference_cases"]
    if type(wire_references) is not list:
        raise RuntimeError("作者侧 wire reference_cases 非 array")
    cases: list[dict[str, Any]] = []
    for source_case in wire_references:
        if type(source_case) is not dict:
            raise RuntimeError("作者侧 wire reference case 非 object")
        cases.append({
            "canonical_signed_payload_hex": source_case[
                "canonical_signed_payload_hex"],
            "document_identity_sha256_hex": source_case[
                "document_identity_sha256_hex"],
            "domain_prefix_hex": source_case["domain_prefix_hex"],
            "expected_code": 0,
            "input_envelope_hex": source_case["envelope_hex"],
            "message_hex": source_case["message_hex"],
            "name": source_case["name"],
        })
    reference_cases = schema_source["reference_cases"]
    if type(reference_cases) is not list or not reference_cases:
        raise RuntimeError("作者侧 schema reference_cases 非法")
    root_case = reference_cases[0]
    if type(root_case) is not dict:
        raise RuntimeError("作者侧 root reference case 非 object")
    root_payload = root_case["signed_payload"]
    if type(root_payload) is not dict:
        raise RuntimeError("作者侧 root payload 非 object")

    envelope_failure = encode_gov_cjson({
        "extension": 1,
        "signature_hex": "g" * 128,
        "signed_payload": root_payload,
    })
    hex_failure = encode_gov_cjson({
        "signature_hex": "g" * 128,
        "signed_payload": root_payload,
    })
    non_string_hex_failure = encode_gov_cjson({
        "signature_hex": 0,
        "signed_payload": root_payload,
    })
    common_payload = deepcopy(root_payload)
    common_payload["algorithm"] = "Ed448"
    common_failure = encode_gov_cjson({
        "signature_hex": "0" * 128,
        "signed_payload": common_payload,
    })
    for name, payload, expected_code in (
            ("envelope-extra-field-precedes-hex", envelope_failure, 3),
            ("signature-hex-precedes-common", hex_failure, 5),
            ("signature-hex-non-string", non_string_hex_failure, 5),
            ("common-precedes-schema", common_failure, 4),
    ):
        cases.append({
            "expected_code": expected_code,
            "input_envelope_hex": payload.hex(),
            "name": name,
        })
    return cases


def _precedence_cases(schema_source: dict[str, Any]) -> list[dict[str, Any]]:
    """物化 single-document physical/envelope/hex/common/schema precedence。"""
    reference_cases = schema_source["reference_cases"]
    if type(reference_cases) is not list or not reference_cases:
        raise RuntimeError("作者侧 schema reference_cases 非法")
    root_case = reference_cases[0]
    if type(root_case) is not dict or type(root_case["signed_payload"]) is not dict:
        raise RuntimeError("作者侧 root case 非法")
    root_payload = root_case["signed_payload"]
    physical = b" " + _encode_envelope(root_payload)
    envelope = encode_gov_cjson({
        "extension": 1,
        "signature_hex": "g" * 128,
        "signed_payload": root_payload,
    })
    hex_failure = encode_gov_cjson({
        "signature_hex": "g" * 128,
        "signed_payload": root_payload,
    })
    common_payload = deepcopy(root_payload)
    common_payload["algorithm"] = "Ed448"
    common_failure = encode_gov_cjson({
        "signature_hex": "0" * 128,
        "signed_payload": common_payload,
    })
    schema_payload = deepcopy(root_payload)
    del schema_payload["issuers"]
    schema_failure = encode_gov_cjson({
        "signature_hex": "0" * 128,
        "signed_payload": schema_payload,
    })
    return [
        {
            "expected_code": 1,
            "input_gov_cjson_hex": physical.hex(),
            "name": "precedence-physical",
            "stage": "physical",
        },
        {
            "expected_code": 3,
            "input_envelope_hex": envelope.hex(),
            "name": "precedence-envelope",
            "stage": "envelope",
        },
        {
            "expected_code": 5,
            "input_envelope_hex": hex_failure.hex(),
            "name": "precedence-hex",
            "stage": "hex",
        },
        {
            "expected_code": 4,
            "input_envelope_hex": common_failure.hex(),
            "name": "precedence-common",
            "stage": "common",
        },
        {
            "expected_code": 101,
            "input_envelope_hex": schema_failure.hex(),
            "name": "precedence-schema",
            "stage": "schema",
        },
    ]


def _host_adapter_cases(schema_source: dict[str, Any]) -> list[dict[str, Any]]:
    """物化不属于 wire bytes 的 code-6 host byte-tuple adapter boundary。"""
    reference_cases = schema_source["reference_cases"]
    if type(reference_cases) is not list or not reference_cases:
        raise RuntimeError("作者侧 schema reference_cases 非法")
    root_case = reference_cases[0]
    if type(root_case) is not dict:
        raise RuntimeError("作者侧 root case 非 object")
    canonical_payload = root_case["canonical_signed_payload_hex"]
    if type(canonical_payload) is not str:
        raise RuntimeError("作者侧 canonical payload 非 text")
    return [
        {
            "expected_code": 6,
            "expected_length": 64,
            "name": "host-unsigned-int-array-short-signature",
            "signed_payload_canonical_gov_cjson_hex": canonical_payload,
            "unsigned_values": [0] * 63,
        },
        {
            "expected_code": 6,
            "expected_length": 64,
            "name": "host-unsigned-int-array-out-of-range-256",
            "signed_payload_canonical_gov_cjson_hex": canonical_payload,
            "unsigned_values": [0] * 63 + [256],
        },
    ]


def _crypto_transport_cases(wire_source: dict[str, Any]) -> list[dict[str, Any]]:
    """复制 RFC public transport vectors；它们不驱动本切片的验签。"""
    vectors = wire_source["ed25519_public_vectors"]
    if type(vectors) is not list:
        raise RuntimeError("作者侧 crypto transport vectors 非 array")
    result: list[dict[str, Any]] = []
    for source_vector in vectors:
        if type(source_vector) is not dict:
            raise RuntimeError("作者侧 crypto transport vector 非 object")
        result.append({
            "expected_verdict": source_vector["expected_verdict"],
            "message_hex": source_vector["message_hex"],
            "name": source_vector["name"],
            "public_key_hex": source_vector["public_key_hex"],
            "signature_hex": source_vector["signature_hex"],
        })
    return result


def _max_u63_declaration_case(
        by_name: dict[str, dict[str, Any]],
        ) -> dict[str, Any]:
    """物化完整 max-u63 declaration，不向 consumer 暴露 mutation recipe。"""
    base_case = by_name.get("source-snapshot-declaration-genesis-v1")
    if type(base_case) is not dict:
        raise RuntimeError("作者侧 max-u63 declaration 基准缺失")
    base_payload = base_case.get("signed_payload")
    if type(base_payload) is not dict:
        raise RuntimeError("作者侧 max-u63 declaration payload 非 object")
    payload = deepcopy(base_payload)
    source_ref_key = payload.get("source_ref_key")
    if (type(source_ref_key) is not list or len(source_ref_key) != 11
            or source_ref_key[3:7] != [0, 0, 0, 1]):
        raise RuntimeError("作者侧 max-u63 source_ref_key 固定位置漂移")

    payload["sequence"] = GOVERNANCE_WIRE_MAX_U63
    payload["predecessor_declaration_identity_sha256"] = "f" * 64
    payload["metadata_byte_count"] = GOVERNANCE_WIRE_MAX_U63
    payload["source_file_byte_count"] = GOVERNANCE_WIRE_MAX_U63
    for index in (0, 1, 2, 7, 8, 9, 10):
        source_ref_key[index] = GOVERNANCE_WIRE_MAX_U63

    envelope = _encode_envelope(payload)
    document = parse_source_snapshot_declaration_schema_document(envelope)
    if document.sequence != GOVERNANCE_WIRE_MAX_U63:
        raise RuntimeError("作者侧 max-u63 declaration sequence 未穿过 schema")
    canonical_payload = parse_gov_cjson(document.canonical_signed_payload)
    if (canonical_payload["metadata_byte_count"] != GOVERNANCE_WIRE_MAX_U63
            or canonical_payload["source_file_byte_count"]
            != GOVERNANCE_WIRE_MAX_U63
            or canonical_payload["source_ref_key"] != [
                GOVERNANCE_WIRE_MAX_U63,
                GOVERNANCE_WIRE_MAX_U63,
                GOVERNANCE_WIRE_MAX_U63,
                0, 0, 0, 1,
                GOVERNANCE_WIRE_MAX_U63,
                GOVERNANCE_WIRE_MAX_U63,
                GOVERNANCE_WIRE_MAX_U63,
                GOVERNANCE_WIRE_MAX_U63,
            ]):
        raise RuntimeError("作者侧 max-u63 declaration 标量未穿过 schema")
    return {
        "canonical_signed_payload_hex": document.canonical_signed_payload.hex(),
        "document_identity_sha256_hex": bytes(document.document_identity).hex(),
        "domain_prefix_hex": document.domain_prefix.hex(),
        "expected_code": 0,
        "input_envelope_hex": envelope.hex(),
        "message_hex": document.message.hex(),
        "name": "source-snapshot-declaration-max-u63-v1",
    }


def _schema_pages(schema_source: dict[str, Any]) -> list[tuple[str, bytes]]:
    """把 schema 正反例分别物化为包含完整 envelope hex 的 canonical pages。"""
    references = schema_source["reference_cases"]
    rejections = schema_source["schema_rejections"]
    if type(references) is not list or type(rejections) is not list:
        raise RuntimeError("作者侧 schema corpus 列表非法")
    by_name: dict[str, dict[str, Any]] = {}
    positive_cases: list[dict[str, Any]] = []
    for source_case in references:
        if type(source_case) is not dict:
            raise RuntimeError("作者侧 schema reference case 非 object")
        name = source_case["name"]
        payload = source_case["signed_payload"]
        if type(name) is not str or type(payload) is not dict:
            raise RuntimeError("作者侧 schema reference case 字段非法")
        by_name[name] = source_case
        envelope = _encode_envelope(payload)
        positive_cases.append({
            "canonical_signed_payload_hex": source_case[
                "canonical_signed_payload_hex"],
            "document_identity_sha256_hex": source_case[
                "document_identity_sha256_hex"],
            "domain_prefix_hex": source_case["domain_prefix_hex"],
            "expected_code": 0,
            "input_envelope_hex": envelope.hex(),
            "message_hex": (
                bytes.fromhex(source_case["domain_prefix_hex"])
                + bytes.fromhex(source_case["canonical_signed_payload_hex"])).hex(),
            "name": name,
        })

    positive_cases.append(_max_u63_declaration_case(by_name))
    negative_cases: list[dict[str, Any]] = []
    for vector in rejections:
        if type(vector) is not dict:
            raise RuntimeError("作者侧 schema rejection 非 object")
        base_name = vector["base_case"]
        if type(base_name) is not str or base_name not in by_name:
            raise RuntimeError("作者侧 schema rejection base case 非法")
        base_payload = by_name[base_name]["signed_payload"]
        if type(base_payload) is not dict:
            raise RuntimeError("作者侧 schema base payload 非 object")
        negative_cases.append({
            "expected_code": vector["error_code"],
            "input_envelope_hex": _encode_envelope(
                _apply_schema_rejection(base_payload, vector)).hex(),
            "name": vector["name"],
        })
    return [
        (
            f"{_PAGE_PREFIX}schema_positive.json",
            _canonical_page({
                "page_role": "schema-positive",
                "profile": _PROFILE,
                "schema_cases": positive_cases,
                "version": 1,
            }),
        ),
        (
            f"{_PAGE_PREFIX}schema_negative.json",
            _canonical_page({
                "page_role": "schema-negative",
                "profile": _PROFILE,
                "schema_cases": negative_cases,
                "version": 1,
            }),
        ),
    ]


def _chain_pages(chain_source: dict[str, Any]) -> list[tuple[str, bytes]]:
    """物化每个 chain case 为单独 canonical page，避免总 manifest 超出预算。"""
    source_collections = chain_source["reference_collections"]
    rejections = chain_source["chain_shape_rejections"]
    expected_heads = chain_source["expected_head_identities_sha256_hex"]
    if (type(source_collections) is not dict or type(rejections) is not list
            or type(expected_heads) is not list):
        raise RuntimeError("作者侧 chain corpus 结构非法")

    vectors: list[dict[str, Any]] = []
    reference = _copy_collections(source_collections)
    reference_hexes = _collection_hexes(reference)
    vectors.append({
        "declaration_envelopes_hex": list(reversed(
            reference_hexes["declaration_envelopes_hex"])),
        "expected_code": 0,
        "expected_head_identities_sha256_hex": expected_heads,
        "name": "chain-reference-reversed-input-v1",
        "registry_envelopes_hex": list(reversed(
            reference_hexes["registry_envelopes_hex"])),
        "revocation_envelopes_hex": list(reversed(
            reference_hexes["revocation_envelopes_hex"])),
    })
    for source_vector in rejections:
        if type(source_vector) is not dict:
            raise RuntimeError("作者侧 chain rejection 非 object")
        collections = _copy_collections(source_collections)
        _apply_chain_rejection(collections, source_vector)
        vector = _collection_hexes(collections)
        vector["expected_code"] = source_vector["error_code"]
        vector["name"] = source_vector["name"]
        vectors.append(vector)

    multiple_invalid = _copy_collections(source_collections)
    multiple_invalid["registry"][0]["unregistered_extension"] = 1
    issuers = multiple_invalid["registry"][1]["issuers"]
    if type(issuers) is not list or len(issuers) != 2:
        raise RuntimeError("作者侧多 invalid registry issuers 非法")
    issuers[0], issuers[1] = issuers[1], issuers[0]
    invalid_hexes = _collection_hexes(multiple_invalid)
    for name, registry_hexes in (
            ("registry-two-invalid-order-a", invalid_hexes[
                "registry_envelopes_hex"]),
            ("registry-two-invalid-order-b", list(reversed(invalid_hexes[
                "registry_envelopes_hex"]))),
    ):
        vectors.append({
            "declaration_envelopes_hex": invalid_hexes[
                "declaration_envelopes_hex"],
            "expected_code": 101,
            "name": name,
            "registry_envelopes_hex": registry_hexes,
            "revocation_envelopes_hex": invalid_hexes[
                "revocation_envelopes_hex"],
        })

    pages: list[tuple[str, bytes]] = []
    for vector in vectors:
        name = vector["name"]
        if type(name) is not str:
            raise RuntimeError("chain vector name 非 text")
        pages.append((
            f"{_PAGE_PREFIX}chain_{name}.json",
            _canonical_page({
                "chain_cases": [vector],
                "page_role": "chain",
                "profile": _PROFILE,
                "version": 1,
            }),
        ))
    return pages


def _wire_page(
        wire_source: dict[str, Any], schema_source: dict[str, Any],
        ) -> tuple[str, bytes]:
    """物化 wire/parser/precedence/adapter 直接向量的 canonical page。"""
    return (
        f"{_PAGE_PREFIX}wire.json",
        _canonical_page({
            "document_precedence_cases": _precedence_cases(schema_source),
            "page_role": "wire",
            "profile": _PROFILE,
            "version": 1,
            "wire_envelope_cases": _wire_envelope_cases(
                wire_source, schema_source),
            "wire_host_adapter_cases": _host_adapter_cases(schema_source),
            "wire_parser_cases": _parser_direct_cases(wire_source),
            "wire_public_crypto_transport_cases": _crypto_transport_cases(
                wire_source),
        }),
    )


def _check_file_name(file_name: str, *, raw: bool) -> None:
    """限制 index 中的 basename，避免产物 loader 获得 host path 语义。"""
    allowed = frozenset(
        "abcdefghijklmnopqrstuvwxyz0123456789._-")
    expected_prefix = _RAW_PREFIX if raw else _PAGE_PREFIX
    expected_suffix = ".bin" if raw else ".json"
    if (type(file_name) is not str or not file_name.startswith(expected_prefix)
            or not file_name.endswith(expected_suffix) or ".." in file_name
            or any(character not in allowed for character in file_name)):
        raise RuntimeError(f"产物 basename 不满足固定 grammar: {file_name}")


def _build_outputs() -> dict[str, bytes]:
    """生成所有确定性产物；返回值按 basename 映射且不写入非 fixture 位置。"""
    source = {
        file_name: _read_source_fixture(file_name, source_sha)
        for file_name, source_sha in _SOURCE_FIXTURES
    }
    wire_source = source["gov_cjson_v1_conformance.json"]
    schema_source = source["gov_g0b_g0c_schema_v1_conformance.json"]
    chain_source = source["gov_g0b_chain_shape_v1_conformance.json"]

    pages = [_wire_page(wire_source, schema_source)]
    pages.extend(_schema_pages(schema_source))
    pages.extend(_chain_pages(chain_source))
    raw_artifacts = _raw_parser_artifacts()

    outputs: dict[str, bytes] = {}
    page_entries: list[dict[str, Any]] = []
    for file_name, payload in pages:
        _check_file_name(file_name, raw=False)
        if file_name in outputs:
            raise RuntimeError(f"重复 page basename: {file_name}")
        outputs[file_name] = payload
        page_entries.append({
            "byte_count": len(payload),
            "file_name": file_name,
            "page_role": parse_gov_cjson(payload)["page_role"],
            "sha256_hex": _sha256_hex(payload),
        })

    raw_entries: list[dict[str, Any]] = []
    for entry, payload in raw_artifacts:
        file_name = entry["file_name"]
        if type(file_name) is not str:
            raise RuntimeError("raw artifact file_name 非 text")
        _check_file_name(file_name, raw=True)
        if file_name in outputs:
            raise RuntimeError(f"重复 raw basename: {file_name}")
        outputs[file_name] = payload
        raw_entries.append(entry)

    index = _canonical_page({
        "artifact_kind": "GOVERNANCE_PORTABLE_CONFORMANCE_MANIFEST_INDEX_V1",
        "authoring_fixture_sha256": [
            {"file_name": file_name, "sha256_hex": source_sha}
            for file_name, source_sha in _SOURCE_FIXTURES
        ],
        "page_order": page_entries,
        "profile": _PROFILE,
        "raw_input_artifacts": raw_entries,
        "version": 1,
    })
    outputs[_MANIFEST_NAME] = index
    sidecar = f"{_sha256_hex(index)}  {_MANIFEST_NAME}".encode("ascii")
    outputs[_SIDECAR_NAME] = sidecar
    return outputs


def _write_outputs(outputs: dict[str, bytes]) -> None:
    """把生成结果精确写入固定 fixture 目录，写入集合没有动态发现。"""
    for file_name, payload in outputs.items():
        (_FIXTURE_DIR / file_name).write_bytes(payload)


def _check_outputs(outputs: dict[str, bytes]) -> bool:
    """逐字节核对磁盘结果，供生成后和 CI 的确定性审计使用。"""
    clean = True
    for file_name, expected in outputs.items():
        path = _FIXTURE_DIR / file_name
        if not path.is_file() or path.read_bytes() != expected:
            print(f"MISMATCH {file_name}")
            clean = False
    expected_names = set(outputs)
    actual_names = {
        path.name for path in _FIXTURE_DIR.iterdir()
        if path.name.startswith(_PAGE_PREFIX)
        or path.name in {_MANIFEST_NAME, _SIDECAR_NAME}
    }
    unexpected = sorted(actual_names - expected_names)
    if unexpected:
        for file_name in unexpected:
            print(f"UNEXPECTED {file_name}")
        clean = False
    return clean


def main() -> int:
    """运行作者侧生成或只读一致性核对，返回进程级确定性状态。"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check", action="store_true",
        help="不写入，仅逐字节核对现有 manifest family",
    )
    arguments = parser.parse_args()
    outputs = _build_outputs()
    if arguments.check:
        return 0 if _check_outputs(outputs) else 1
    _write_outputs(outputs)
    return 0 if _check_outputs(outputs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
