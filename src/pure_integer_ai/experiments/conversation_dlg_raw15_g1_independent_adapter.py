"""DLG-RAW-15 G1 resolver feasibility probe（formal status: NOT_READY）。

当前切片只验证 source-bound resolver 的候选歧义闭合。它仍是内存 authored
fixture、固定 logical key 的 adapter，两个 target 也是 UNKNOWN；因此不能被
报告为独立 held-out family、正向回答能力或 G1 PASS。正式施工需要另行提供
physical source pack、独立 source namespace、route-form 和两个 ANSWER target。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from pure_integer_ai.experiments.conversation_public_response_act_catalog import (
    load_public_response_act_frame_catalog_from_closure,
)
from pure_integer_ai.experiments.conversation_public_answer_catalog import (
    _answer_frame_from_manifest,
)
from pure_integer_ai.experiments.conversation_public_frame_catalog import (
    PublicFrameCatalog,
)
from pure_integer_ai.experiments.conversation_public_source_payload_provider import (
    PUBLIC_SOURCE_PAYLOAD_LOGICAL_KEYS_V1,
    build_public_source_payload_closure_v1,
    public_source_payload_record_from_u8_v1,
)
from pure_integer_ai.experiments.conversation_source_bound_slot_catalog import (
    SOURCE_BOUND_SLOT_CATALOG_LOGICAL_KEY_V3,
    load_source_bound_slot_composition_catalog_from_closure,
)
from pure_integer_ai.experiments.ph2_dataset_core import (
    DatasetContractError,
    canonical_json_bytes,
    canonical_json_line,
    parse_canonical_json_bytes,
)


DLG_RAW15_G1_FORMAL_STATUS = "FEASIBILITY_ONLY_NOT_READY"

_PACK_MANIFEST_FIELDS = (
    "adapter_schema", "files", "license_id", "pack_id",
    "source_namespace", "status",
)
_PACK_FILE_FIELDS = (
    "logical_key", "payload_length", "raw_sha256", "registry_key",
    "relative_path",
)
_PACK_SCHEMA = 1

_COURSE_KEY = b"data/ph2/grounded_answer_train_v1.jsonl.sample"
_LEXICAL_A_KEY = (
    b"data/ph2/dlg_raw_public_response_act_lexical_v2_a.txt.sample")
_LEXICAL_B_KEY = (
    b"data/ph2/dlg_raw_public_response_act_lexical_v2_b.txt.sample")
_RESPONSE_KEY = (
    b"data/ph2/dlg_raw_public_response_act_frame_v2.jsonl.sample")
_SLOT_KEY = SOURCE_BOUND_SLOT_CATALOG_LOGICAL_KEY_V3


def _surface(text: str) -> dict[str, list[int] | str]:
    """将 authored Unicode surface 显式写成 scalar/UTF-8 双表示。"""
    scalars = tuple(ord(character) for character in text)
    return {
        "scalars": list(scalars),
        "utf8_hex": text.encode("utf-8").hex(),
    }


def _source_record(
        record_id: str,
        source_key: list[int],
        path: str,
        payload: bytes,
        span_text: str,
        attribution: str,
        ) -> dict:
    """建立 authored source witness 的完整 raw span record。"""
    span_bytes = span_text.encode("utf-8")
    start = payload.index(span_bytes)
    return {
        "attribution": attribution,
        "license_id": "CC0-1.0",
        "raw_sha256": hashlib.sha256(payload).hexdigest(),
        "record_id": record_id,
        "relative_path": path,
        "source_ref_key": source_key,
        "span": [start, start + len(span_bytes)],
        "span_utf8_hex": span_bytes.hex(),
    }


def _answer_seed_episode() -> dict:
    """提供 response-act compiler 所需的最小 authored ANSWER seed。"""
    return {
        "artifact_kind": "PH2_GROUNDED_ANSWER_EPISODE_V1",
        "clusters": {
            "paraphrase": "g1-seed-answer-paraphrase",
            "proposition": "g1-seed-answer-proposition",
            "question_construction": "g1-seed-answer-question",
            "source": "g1-seed-answer-source",
        },
        "dialogue": {"active_scope_ids": [301, 401], "turns": [{
            "reference_ids": [], "scope_ids": [301, 401], "speaker": "USER",
            "surface": "测试台的启用年份是什么？", "turn_id": 1,
        }]},
        "episode_id": "g1-seed-answer-v1",
        "license_id": "CC0-1.0",
        "question": {
            "answer_plan": {
                "citation_source_ids": ["g1-seed-source"],
                "forbidden_claim_ids": [],
                "ordered_claim_ids": ["g1-seed-claim"],
                "required_claim_ids": ["g1-seed-claim"],
                "response_act": "ANSWER",
            },
            "context_surface": "测试台档案记载：设备于2025年启用。",
            "evidence": [{
                "claim_text": "测试台设备于2025年启用",
                "evidence_id": "g1-seed-evidence",
                "evidence_text": "设备于2025年启用",
                "proposition_id": "g1-seed-claim",
                "refute": 0,
                "scope_id": 301,
                "source_id": "g1-seed-source",
                "support": 1,
            }],
            "evidence_scope_id": 301,
            "question_surface": "测试台的启用年份是什么？",
            "response_scope_id": 401,
            "typed_intent": "ASK_EVENT_TIME",
        },
        "schema_version": 1,
        "split": "train",
        "surfaces": {
            "accepted": [{
                "carrier_kind": "PLAIN_TEXT",
                "cited_source_ids": ["g1-seed-source"],
                "claim_ids": ["g1-seed-claim"],
                "realization_id": "g1-answer-a",
                "response_act": "ANSWER",
                "scope_id": 401,
                "surface": "测试台设备于2025年启用。",
            }, {
                "carrier_kind": "PLAIN_TEXT",
                "cited_source_ids": ["g1-seed-source"],
                "claim_ids": ["g1-seed-claim"],
                "realization_id": "g1-answer-b",
                "response_act": "ANSWER",
                "scope_id": 401,
                "surface": "档案显示，测试台设备于2025年启用。",
            }],
            "minimum_legal_surfaces": 2,
            "rejected": [{
                "expected_violations": ["MISSING_REQUIRED_CLAIM"],
                "realization": {
                    "carrier_kind": "PLAIN_TEXT",
                    "cited_source_ids": ["g1-seed-source"],
                    "claim_ids": [],
                    "realization_id": "g1-answer-bad",
                    "response_act": "ANSWER",
                    "scope_id": 401,
                    "surface": "档案中有相关启用记录。",
                },
            }],
        },
    }


def _unknown_episode(ordinal: int, question: str, entity: str) -> dict:
    """建立当前 feasibility probe 使用的无证据 UNKNOWN episode。"""
    return {
        "artifact_kind": "PH2_GROUNDED_ANSWER_EPISODE_V1",
        "clusters": {
            "paraphrase": f"g1-budget-{ordinal}-paraphrase",
            "proposition": f"g1-budget-{ordinal}-proposition",
            "question_construction": f"g1-budget-{ordinal}-question",
            "source": f"g1-budget-{ordinal}-source",
        },
        "dialogue": {"active_scope_ids": [300 + ordinal, 400 + ordinal],
                      "turns": [{
            "reference_ids": [],
            "scope_ids": [300 + ordinal, 400 + ordinal],
            "speaker": "USER",
            "surface": question,
            "turn_id": 1,
        }]},
        "episode_id": f"g1-heldout-budget-{ordinal}-v1",
        "license_id": "CC0-1.0",
        "question": {
            "answer_plan": {
                "citation_source_ids": [],
                "forbidden_claim_ids": [],
                "ordered_claim_ids": [],
                "required_claim_ids": [],
                "response_act": "UNKNOWN",
            },
            "context_surface": f"公开档案只给出{entity}的名称，没有提供预算。",
            "evidence": [],
            "evidence_scope_id": 300 + ordinal,
            "question_surface": question,
            "response_scope_id": 400 + ordinal,
            "typed_intent": "ASK_QUANTITY",
        },
        "schema_version": 1,
        "split": "train",
        "surfaces": {
            "accepted": [{
                "carrier_kind": "PLAIN_TEXT",
                "cited_source_ids": [],
                "claim_ids": [],
                "realization_id": f"g1-budget-{ordinal}-a",
                "response_act": "UNKNOWN",
                "scope_id": 400 + ordinal,
                "surface": f"现有来源没有提供{entity}的建设预算。",
            }, {
                "carrier_kind": "PLAIN_TEXT",
                "cited_source_ids": [],
                "claim_ids": [],
                "realization_id": f"g1-budget-{ordinal}-b",
                "response_act": "UNKNOWN",
                "scope_id": 400 + ordinal,
                "surface": f"根据当前资料，无法确定{entity}的建设预算。",
            }],
            "minimum_legal_surfaces": 2,
            "rejected": [{
                "expected_violations": ["UNSUPPORTED_CLAIM", "NONANSWER_CLAIM"],
                "realization": {
                    "carrier_kind": "PLAIN_TEXT",
                    "cited_source_ids": [],
                    "claim_ids": [f"g1-budget-{ordinal}-invented"],
                    "realization_id": f"g1-budget-{ordinal}-invented",
                    "response_act": "UNKNOWN",
                    "scope_id": 400 + ordinal,
                    "surface": f"{entity}的建设预算是十万元。",
                },
            }, {
                "expected_violations": ["RESPONSE_ACT_DRIFT"],
                "realization": {
                    "carrier_kind": "PLAIN_TEXT",
                    "cited_source_ids": [],
                    "claim_ids": [],
                    "realization_id": f"g1-budget-{ordinal}-act",
                    "response_act": "ANSWER",
                    "scope_id": 400 + ordinal,
                    "surface": "答案如下。",
                },
            }],
        },
    }


def make_payloads() -> tuple[bytes, bytes, bytes, bytes, tuple[dict, ...]]:
    """形成当前 feasibility-only 的 authored in-memory payloads。"""
    episodes = (
        _answer_seed_episode(),
        _unknown_episode(1, "玄衡台的建设预算是多少？", "玄衡台的建设"),
        _unknown_episode(2, "玄衡台预算是多少？", "玄衡台"),
    )
    course = b"".join(canonical_json_line(item) for item in episodes)
    lexical_a = (
        "G1 source lexical A\n玄衡台的建设预算是多少？\n玄衡台预算是多少？\n"
    ).encode("utf-8")
    lexical_b = (
        "G1 source lexical B\n玄衡台的建设预算是多少？\n玄衡台预算是多少？\n"
    ).encode("utf-8")
    response_rows = []
    for ordinal, episode in enumerate(episodes[1:], 1):
        response_rows.append({
            "catalog_schema": 2,
            "course_raw_sha256": hashlib.sha256(course).hexdigest(),
            "course_relative_path": "data/ph2/grounded_answer_train_v1.jsonl.sample",
            "episode_id": episode["episode_id"],
            "frame_key": f"g1-frame-budget-{ordinal}-v1",
            "lexical_source_a": {
                "attribution": "Pure Integer AI authored G1 held-out lexical observation A",
                "license_id": "CC0-1.0",
                "raw_sha256": hashlib.sha256(lexical_a).hexdigest(),
                "relative_path": "data/ph2/dlg_raw_public_response_act_lexical_v2_a.txt.sample",
            },
            "lexical_source_b": {
                "attribution": "Pure Integer AI authored G1 held-out lexical observation B",
                "license_id": "CC0-1.0",
                "raw_sha256": hashlib.sha256(lexical_b).hexdigest(),
                "relative_path": "data/ph2/dlg_raw_public_response_act_lexical_v2_b.txt.sample",
            },
            "output_max_bytes": 4096,
        })
    response = b"".join(canonical_json_line(item) for item in response_rows)
    return course, lexical_a, lexical_b, response, episodes


def _slot_manifest(base_catalog) -> tuple[bytes, dict[bytes, bytes]]:
    """建立当前 probe 的 V3 slot manifest 与 authored source bytes。"""
    suffix = "预算是多少？"
    alias = "西岸入口"
    family_a = "甲构式"
    family_b = "乙构式"
    paths = (
        "data/ph2/dlg_raw_public_slot_family_v2_site_a.txt.sample",
        "data/ph2/dlg_raw_public_slot_family_v2_site_b.txt.sample",
        "data/ph2/dlg_raw_public_slot_relation_v3_east-bank-north_a.txt.sample",
        "data/ph2/dlg_raw_public_slot_relation_v3_east-bank-north_b.txt.sample",
        "data/ph2/dlg_raw_public_slot_relation_v3_east-bank-pier_a.txt.sample",
        "data/ph2/dlg_raw_public_slot_relation_v3_east-bank-pier_b.txt.sample",
    )
    payloads = {
        paths[0].encode(): (family_a + suffix + "\n").encode(),
        paths[1].encode(): (family_b + suffix + "\n").encode(),
        paths[2].encode(): ("玄衡台的建设=" + alias + "\n").encode(),
        paths[3].encode(): (alias + "=玄衡台的建设\n").encode(),
        paths[4].encode(): ("玄衡台=" + alias + "\n").encode(),
        paths[5].encode(): (alias + "=玄衡台\n").encode(),
    }
    source_keys = [
        [65164, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        [65165, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        [65171, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        [65172, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        [65173, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        [65174, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    ]
    spans = (suffix, suffix, alias, alias, alias, alias)
    records = tuple(_source_record(
        f"g1-source-{ordinal}", source_keys[ordinal],
        paths[ordinal], payloads[paths[ordinal].encode()], spans[ordinal],
        f"Pure Integer AI authored G1 feasibility source {ordinal}")
        for ordinal in range(6))
    frames = tuple(base_catalog.frames)
    manifest = {
        "bindings": [{
            "base_catalog_sha256": bytes(base_catalog.source_sha256).hex(),
            "base_frame_key": frames[0].frame_key,
            "base_frame_raw_sha256": bytes(frames[0].raw_line_sha256).hex(),
            "binding_key": "g1-binding-build-v1",
            "entity": _surface(alias),
            "negative_relation_source_record_ids": [],
            "positive_relation_source_record_ids": ["g1-source-2", "g1-source-3"],
        }, {
            "base_catalog_sha256": bytes(base_catalog.source_sha256).hex(),
            "base_frame_key": frames[1].frame_key,
            "base_frame_raw_sha256": bytes(frames[1].raw_line_sha256).hex(),
            "binding_key": "g1-binding-root-v1",
            "entity": _surface(alias),
            "negative_relation_source_record_ids": [],
            "positive_relation_source_record_ids": ["g1-source-4", "g1-source-5"],
        }],
        "catalog_schema": 3,
        "families": [{
            "construction_witnesses": [
                {"observed_entity": _surface(family_a), "source_record_id": "g1-source-0"},
                {"observed_entity": _surface(family_b), "source_record_id": "g1-source-1"},
            ],
            "family_key": "g1-budget-entity-slot-v1",
            "prefix": _surface(""),
            "slot_type": "ENTITY_ALIAS_V3",
            "suffix": _surface(suffix),
        }],
        "source_records": records,
    }
    return canonical_json_line(manifest), payloads


def make_closure(*, binding_indices=(0, 1)):
    """构造 feasibility-only closure；不读取或修改 production closure。"""
    course, lexical_a, lexical_b, response, episodes = make_payloads()
    preliminary = {
        _COURSE_KEY: course,
        _LEXICAL_A_KEY: lexical_a,
        _LEXICAL_B_KEY: lexical_b,
        _RESPONSE_KEY: response,
    }
    records = tuple(
        public_source_payload_record_from_u8_v1(
            key, preliminary.get(key, b""))
        for key in PUBLIC_SOURCE_PAYLOAD_LOGICAL_KEYS_V1
    )
    first_closure = build_public_source_payload_closure_v1(records)
    base_catalog = load_public_response_act_frame_catalog_from_closure(
        first_closure, _RESPONSE_KEY)
    slot_manifest, slot_payloads = _slot_manifest(base_catalog)
    replacements = dict(preliminary)
    replacements[_SLOT_KEY] = slot_manifest
    replacements.update(slot_payloads)
    records = tuple(
        public_source_payload_record_from_u8_v1(
            key, replacements.get(key, b""))
        for key in PUBLIC_SOURCE_PAYLOAD_LOGICAL_KEYS_V1
    )
    closure = build_public_source_payload_closure_v1(records)
    base_catalog = load_public_response_act_frame_catalog_from_closure(
        closure, _RESPONSE_KEY)
    if tuple(binding_indices) == (0, 1):
        return closure, episodes, base_catalog
    rows = [
        parse_canonical_json_bytes(line, require_object=True)
        for line in slot_manifest.splitlines() if line
    ]
    selected = dict(rows[0])
    selected["bindings"] = [rows[0]["bindings"][index] for index in binding_indices]
    replacements[_SLOT_KEY] = canonical_json_line(selected)
    records = tuple(
        public_source_payload_record_from_u8_v1(
            key, replacements.get(key, b""))
        for key in PUBLIC_SOURCE_PAYLOAD_LOGICAL_KEYS_V1
    )
    return build_public_source_payload_closure_v1(records), episodes, base_catalog


def load_g1_slot_catalog(closure, base_catalog):
    """加载并回读 feasibility-only V3 slot manifest。"""
    catalog = load_source_bound_slot_composition_catalog_from_closure(
        closure, base_catalog, base_catalog, catalog_logical_key=_SLOT_KEY)
    catalog.verify_sources(closure)
    return catalog


@dataclass(frozen=True, slots=True)
class G1PhysicalPackV1:
    """独立 physical pack 的只读 struct carrier。

    ``closure`` 使用现有 Python reference 的固定 logical registry 作为适配
    槽位；pack 自身的物理文件、pack identity 与 source namespace 由本 adapter
    单独锁定。未列出的 registry 槽位明确填充空 u8[]，绝不从 production root
    回读，因此该对象不能改变默认 closure。
    """

    pack_id: str
    source_namespace: str
    status: str
    manifest_sha256: bytes
    closure: object
    base_catalog: object
    slot_catalog: object


def _pack_ascii(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or any(
            ord(char) < 0x21 or ord(char) > 0x7E for char in value):
        raise ValueError(f"{label} 不是可迁移 ASCII 标识")
    return value


def _pack_hex(value: object, *, label: str) -> bytes:
    if (not isinstance(value, str) or len(value) != 64
            or value != value.lower()
            or any(char not in "0123456789abcdef" for char in value)):
        raise ValueError(f"{label} 不是小写 SHA-256 hex")
    return bytes.fromhex(value)


def _pack_manifest_object(payload: bytes) -> dict:
    """严格读取一个物理 pack manifest，identity 只由 canonical JSON 形成。"""
    raw = payload[:-1] if payload.endswith(b"\n") else payload
    if not raw or b"\r" in raw:
        raise ValueError("G1 pack manifest framing 非法")
    try:
        value = parse_canonical_json_bytes(raw, require_object=True)
    except DatasetContractError as error:
        raise ValueError("G1 pack manifest 不是 canonical JSON") from error
    if tuple(value) != _PACK_MANIFEST_FIELDS:
        raise ValueError("G1 pack manifest 字段集合漂移")
    if value["adapter_schema"] != _PACK_SCHEMA:
        raise ValueError("G1 pack manifest schema 未注册")
    if value["license_id"] != "CC0-1.0":
        raise ValueError("G1 physical pack 只能使用 CC0-1.0")
    _pack_ascii(value["pack_id"], label="pack_id")
    _pack_ascii(value["source_namespace"], label="source_namespace")
    if value["status"] != DLG_RAW15_G1_FORMAL_STATUS:
        raise ValueError("G1 physical pack 状态必须保持 NOT_READY")
    files = value["files"]
    if (not isinstance(files, list) or not files
            or tuple(item["logical_key"] for item in files)
            != tuple(sorted(item["logical_key"] for item in files))):
        raise ValueError("G1 pack files 未按 logical key 规范排序")
    seen = set()
    seen_registry = set()
    for ordinal, item in enumerate(files):
        if (not isinstance(item, dict)
                or tuple(item) != _PACK_FILE_FIELDS):
            raise ValueError(f"G1 pack file[{ordinal}] 字段集合漂移")
        key = item["logical_key"]
        try:
            logical_key = key.encode("ascii")
        except (AttributeError, UnicodeEncodeError) as error:
            raise ValueError("G1 pack logical key 必须是 ASCII") from error
        if (not logical_key.startswith(b"packs/")
                or b".." in logical_key.split(b"/")):
            raise ValueError("G1 pack logical key 未绑定独立 pack namespace")
        if logical_key in seen:
            raise ValueError("G1 pack logical key 重复")
        seen.add(logical_key)
        registry_value = item["registry_key"]
        try:
            registry_key = registry_value.encode("ascii")
        except (AttributeError, UnicodeEncodeError) as error:
            raise ValueError("G1 pack registry key 必须是 ASCII") from error
        if registry_key not in PUBLIC_SOURCE_PAYLOAD_LOGICAL_KEYS_V1:
            raise ValueError("G1 pack registry key 未登记")
        if registry_key in seen_registry:
            raise ValueError("G1 pack registry key 重复")
        seen_registry.add(registry_key)
        path = item["relative_path"]
        if (not isinstance(path, str) or not path
                or "\\" in path or "/" in path
                or path in (".", "..")):
            raise ValueError("G1 pack relative path 非法")
        if (not isinstance(item["payload_length"], int)
                or item["payload_length"] < 0):
            raise ValueError("G1 pack payload length 非法")
        _pack_hex(item["raw_sha256"], label=f"G1 pack file[{ordinal}] SHA")
    return value


def load_g1_physical_pack(
        resource_root: str | Path | None = None,
        ) -> G1PhysicalPackV1:
    """读取独立 G1 physical pack，并真实装配 V3 frame/slot catalog。

    这是 public feasibility adapter：它只读取自身 fixture root，未列 registry
    槽位使用明确空 payload，绝不调用 production host adapter 或默认 closure。
    """
    root = (Path(__file__).resolve().parents[3]
            / "tests" / "fixtures" / "dlg_raw15_g1_pack_v1"
            if resource_root is None else Path(resource_root))
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("G1 physical pack root 不可用")
    manifest_path = root / "manifest.json"
    manifest_payload = manifest_path.read_bytes()
    manifest = _pack_manifest_object(manifest_payload)
    manifest_sha256 = hashlib.sha256(
        canonical_json_bytes(manifest)).digest()
    payloads = {}
    for item in manifest["files"]:
        path = (root / item["relative_path"]).resolve(strict=True)
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError("G1 physical pack path escape") from error
        if path.is_symlink() or not path.is_file():
            raise ValueError("G1 physical pack file 类型非法")
        payload = path.read_bytes()
        if (len(payload) != item["payload_length"]
                or hashlib.sha256(payload).hexdigest() != item["raw_sha256"]):
            raise ValueError("G1 physical pack payload SHA/length 漂移")
        payloads[item["registry_key"].encode("ascii")] = payload
    records = tuple(
        public_source_payload_record_from_u8_v1(
            logical_key, payloads.get(logical_key, b""))
        for logical_key in PUBLIC_SOURCE_PAYLOAD_LOGICAL_KEYS_V1
    )
    closure = build_public_source_payload_closure_v1(records)
    response_payload = closure.payload_for(_RESPONSE_KEY)
    if not response_payload.endswith(b"\n") or response_payload.endswith(b"\n\n"):
        raise ValueError("G1 physical answer frame JSONL framing 非法")
    frames = []
    for line in response_payload[:-1].split(b"\n"):
        try:
            raw = parse_canonical_json_bytes(line, require_object=True)
            frames.append(_answer_frame_from_manifest(
                raw,
                raw_line_sha256=tuple(hashlib.sha256(line).digest()),
                closure=closure,
            ))
        except (DatasetContractError, TypeError, ValueError, RuntimeError) as error:
            raise ValueError("G1 physical ANSWER frame 无法从独立课程闭合") from error
    if len(frames) != 2:
        raise ValueError("G1 physical pack 必须含两个 ANSWER frame")
    base_catalog = PublicFrameCatalog(
        tuple(hashlib.sha256(response_payload).digest()),
        tuple(sorted(frames, key=lambda item: item.canonical_record())),
    )
    slot_catalog = load_source_bound_slot_composition_catalog_from_closure(
        closure, base_catalog, base_catalog, catalog_logical_key=_SLOT_KEY)
    return G1PhysicalPackV1(
        manifest["pack_id"], manifest["source_namespace"], manifest["status"],
        manifest_sha256, closure, base_catalog, slot_catalog,
    )


__all__ = [
    "DLG_RAW15_G1_FORMAL_STATUS",
    "load_g1_slot_catalog",
    "G1PhysicalPackV1",
    "load_g1_physical_pack",
    "make_closure",
    "make_payloads",
]
