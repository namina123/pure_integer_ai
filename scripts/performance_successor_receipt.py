"""发布规范整数流长度直算的非 readiness 性能 successor receipt。"""
from __future__ import annotations

import argparse
import hashlib
import sys
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any

from pure_integer_ai.experiments.ph2_dataset_contract import (
    canonical_json_bytes,
    parse_canonical_json_bytes,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FORMAT_VERSION = 1
ARTIFACT_KIND = "PURE_INTEGER_AI_PERFORMANCE_SUCCESSOR_RECEIPT"
ARTIFACT_VERSION = "PERFORMANCE-SUCCESSOR-INTEGER-SIZE-20260807-A"
RECEIPT_PATH = "data/ph2/manifests/performance_successor_receipt_v1.json"
PARENT_COMMIT = "ef3f38ce67d709dcce4546733e8515c71aff7750"
CHANGE_COMMIT = "0bb9a43256ed429a61001c3861c17260028675e8"
PRIOR_RECEIPT_PATH = (
    "data/ph2/manifests/struct_layout_successor_receipt_v2.json"
)
PRIOR_RECEIPT_SHA256 = (
    "1047e03de245d89ce07a8e8b35119b1a38ebc79490ddb2b2e7d2239920e6733f"
)
V2_ARTIFACT_VERSION = "PERFORMANCE-SUCCESSOR-INTEGER-SIZE-20260807-B"
V2_RECEIPT_PATH = "data/ph2/manifests/performance_successor_receipt_v2.json"
V2_PARENT_COMMIT = "3e2843cd88d0cde0a7b1e0dea129a7bf2711c12e"
V2_CHANGE_COMMIT = "b9c81704ae954ab0305a031f73424d80e73eedd2"
V2_PRIOR_RECEIPT_PATH = RECEIPT_PATH
V2_PRIOR_RECEIPT_SHA256 = (
    "01ecdb29437d3ce7ac88e126cc7a4ccff206fc458290cf7ad65f80457d9ecb17"
)
V3_ARTIFACT_VERSION = "PERFORMANCE-SUCCESSOR-CACHE-CONSTRUCT-20260807-A"
V3_RECEIPT_PATH = "data/ph2/manifests/performance_successor_receipt_v3.json"
V3_PARENT_COMMIT = "cf3b844b927a2eebb6b923fca2669d765d4718d7"
V3_CHANGE_COMMIT = "02fff5563d6886c3c600c0c5f73dd64e233b389f"
V3_PRIOR_RECEIPT_PATH = V2_RECEIPT_PATH
V3_PRIOR_RECEIPT_SHA256 = (
    "53162d1a89da5f0c3e9dfb85384bfbab285d560999ec778a57eeed8df4b7a055"
)
V4_ARTIFACT_VERSION = "PERFORMANCE-SUCCESSOR-CACHE-CLEAR-20260807-A"
V4_RECEIPT_PATH = "data/ph2/manifests/performance_successor_receipt_v4.json"
V4_PARENT_COMMIT = "0ffe4bd1b0f01c701d1fd269927d69429c34c219"
V4_CHANGE_COMMIT = "4930d57bbc77be03d86a947e61deffe3c02291fb"
V4_PRIOR_RECEIPT_PATH = V3_RECEIPT_PATH
V4_PRIOR_RECEIPT_SHA256 = (
    "13ca4b1d2256b097c7c93481ca632702b24336a161fa683250229e025ec13a8e"
)
V5_ARTIFACT_VERSION = "PERFORMANCE-SUCCESSOR-QUERY-KEY-20260807-A"
V5_RECEIPT_PATH = "data/ph2/manifests/performance_successor_receipt_v5.json"
V5_PARENT_COMMIT = "86704b541635ef67902f201e34f5b34d945c4263"
V5_CHANGE_COMMIT = "7cb4a3c080b885a96deddd1e5bf3906666fb5e51"
V5_PRIOR_RECEIPT_PATH = V4_RECEIPT_PATH
V5_PRIOR_RECEIPT_SHA256 = (
    "71cc8c02c1dd7d663cc3b551e5c46d3584cf9fb37ed6072f3ea611c34b61e97f"
)
_SOURCE_BINDINGS = [
    {
        "current_sha256": (
            "78adea2f14dd55284a4302599927bfb4ba1dd83e0562550a7f7506fe2a4a1dcb"
        ),
        "current_size_bytes": 7461,
        "parent_sha256": (
            "1b70cbb9e4e84a19d3bd30c695b1a9a7b55e0c95208098687413b94baa1a0c8f"
        ),
        "parent_size_bytes": 5677,
        "relative_path": "src/pure_integer_ai/storage/integer_codec.py",
    },
    {
        "current_sha256": (
            "9398daf7661e2546bd7286a5b625872a8b9dfd1583c7296e4dc3f75b87434b34"
        ),
        "current_size_bytes": 11321,
        "parent_sha256": (
            "77f3b3390ba48a07b2524573e67442838e0de2939b4fd0709d88157868b854d5"
        ),
        "parent_size_bytes": 11229,
        "relative_path": "src/pure_integer_ai/storage/sealed_segment.py",
    },
]
_TRANSFORMATION = {
    "algorithm": "ENCODE_THEN_LEN_TO_DIRECT_VARINT_SIZE",
    "cache_cycle": {
        "delta_per_mille": -74,
        "new_median_ns": 118303200,
        "old_median_ns": 127691900,
        "record_count": 5000,
        "repeats": 11,
    },
    "cached_segment_record_slots_decision": {
        "cache_cycle_delta_per_mille": 31,
        "decision": "REJECTED_AND_REVERTED",
        "live_bytes_after_probe": 8001608,
        "live_bytes_before": 12001608,
    },
    "canonical_decoder_changed": 0,
    "canonical_encoder_changed": 0,
    "performance_claim_scope": "LOCAL_DIRECTION_ONLY",
    "segment_record_fields_changed": 0,
}
_VERIFICATION = {
    "bounded_test_runs": [
        {"passed": 66, "scope": "CODEC_K02_R02"},
        {"passed": 83, "scope": "CODEC_K03_K04_W08_LONG_CONTEXT"},
    ],
    "codec_size_tests_passed": 17,
    "full_suite_run": 0,
    "object_model_class_count": 3140,
    "python311_direct_check_clean": 1,
    "python314_direct_check_clean": 1,
    "source_guards_clean": 1,
}
_V2_SOURCE_BINDINGS = [
    {
        "current_sha256": (
            "d9fefb91fa1425fd91a0a1654512fc475720215e423bf92323cb5fb12fce8d0d"
        ),
        "current_size_bytes": 8109,
        "parent_sha256": (
            "78adea2f14dd55284a4302599927bfb4ba1dd83e0562550a7f7506fe2a4a1dcb"
        ),
        "parent_size_bytes": 7461,
        "relative_path": "src/pure_integer_ai/storage/integer_codec.py",
    },
    {
        "current_sha256": (
            "bbc2aac66a5c92aba02508c8946ae045184c9459f1fef9a702d57337e88f787b"
        ),
        "current_size_bytes": 11440,
        "parent_sha256": (
            "9398daf7661e2546bd7286a5b625872a8b9dfd1583c7296e4dc3f75b87434b34"
        ),
        "parent_size_bytes": 11321,
        "relative_path": "src/pure_integer_ai/storage/sealed_segment.py",
    },
]
_V2_TRANSFORMATION = {
    "algorithm": "VALIDATING_SIZE_TO_FROZEN_VALIDATED_KERNEL",
    "cache_cycle": {
        "delta_per_mille": -147,
        "new_median_ns": 144359300,
        "old_median_ns": 169145800,
        "record_count": 5000,
        "repeats": 11,
    },
    "canonical_decoder_changed": 0,
    "canonical_encoder_changed": 0,
    "direct_size_call": {
        "call_count": 200000,
        "delta_per_mille": -381,
        "new_ns_per_call": 12924,
        "old_ns_per_call": 20875,
        "repeats": 11,
    },
    "performance_claim_scope": "LOCAL_DIRECTION_ONLY",
    "public_validation_changed": 0,
    "segment_record_fields_changed": 0,
    "validated_kernel_owner": "FROZEN_SLOTS_SEGMENT_RECORD",
}
_V2_VERIFICATION = {
    "bounded_test_runs": [
        {"passed": 70, "scope": "CODEC_K02_R02"},
        {"passed": 87, "scope": "CODEC_K03_K04_W08_LONG_CONTEXT"},
    ],
    "full_suite_run": 0,
    "object_model_class_count": 3140,
    "python311_direct_check_clean": 1,
    "python314_direct_check_clean": 1,
    "source_guards_clean": 1,
}
_V3_SOURCE_BINDINGS = [
    {
        "current_sha256": (
            "6cbdf509282fa3392e062d9aed0970735dad02e326427a752d3ba40281a2ee0c"
        ),
        "current_size_bytes": 14912,
        "parent_sha256": (
            "19a87dac1170889b7b6331ea3686f8c46196078308cd4a6c5d2550fb3bd44cb6"
        ),
        "parent_size_bytes": 14204,
        "relative_path": "src/pure_integer_ai/storage/segment_cache.py",
    },
]
_V3_TRANSFORMATION = {
    "algorithm": "PUBLIC_VALIDATION_TO_OWNER_VALIDATED_CONSTRUCTION",
    "cache_cycle": {
        "delta_per_mille": -308,
        "new_median_ns": 70977400,
        "old_median_ns": 102564100,
        "record_count": 5000,
        "repeats": 11,
    },
    "cached_record_fields_changed": 0,
    "cached_record_layout_changed": 0,
    "internal_constructor_sites": 6,
    "internal_post_init_calls_after": 0,
    "performance_claim_scope": "LOCAL_DIRECTION_ONLY",
    "public_validation_changed": 0,
    "validated_constructor_owner": "SEGMENT_PAGE_CACHE",
}
_V3_VERIFICATION = {
    "direct_test_files": 3,
    "direct_tests_passed": 64,
    "full_suite_run": 0,
    "object_model_class_count": 3140,
    "python311_direct_check_clean": 1,
    "python314_direct_check_clean": 1,
    "source_guards_clean": 1,
    "validated_constructor_tests_passed": 2,
}
_V4_SOURCE_BINDINGS = [
    {
        "current_sha256": (
            "13b7b0b7fd330d11717ea1bd6d2b0cb17387ad6361c91393b5b916edae90670a"
        ),
        "current_size_bytes": 15056,
        "parent_sha256": (
            "6cbdf509282fa3392e062d9aed0970735dad02e326427a752d3ba40281a2ee0c"
        ),
        "parent_size_bytes": 14912,
        "relative_path": "src/pure_integer_ai/storage/segment_cache.py",
    },
]
_V4_TRANSFORMATION = {
    "algorithm": "CLEAN_UNPINNED_PER_RECORD_EVICT_TO_DIRECT_CLEAR",
    "cache_cycle": {
        "delta_per_mille": -262,
        "new_median_ns": 87606000,
        "old_median_ns": 118593100,
        "record_count": 5000,
        "repeats": 11,
    },
    "cached_record_layout_changed": 0,
    "clean_clear_key_validation_calls_after": 0,
    "clean_clear_record_size_calls_after": 0,
    "clean_clear_sort_calls_after": 0,
    "dirty_requires_flush": 1,
    "logical_access_seq_reset": 0,
    "performance_claim_scope": "LOCAL_DIRECTION_ONLY",
    "pinned_clear_rejected": 1,
    "public_return_contract_changed": 0,
}
_V4_VERIFICATION = {
    "clear_fast_path_tests_passed": 1,
    "direct_test_files": 3,
    "direct_tests_passed": 65,
    "full_suite_run": 0,
    "object_model_class_count": 3140,
    "python311_direct_check_clean": 1,
    "python314_direct_check_clean": 1,
    "source_guards_clean": 1,
}
_V5_SOURCE_BINDINGS = [
    {
        "current_sha256": (
            "88ce77e8274c8dc5bede0f8ab786667de21872185f8c9e61ea5a0b9b6600219a"
        ),
        "current_size_bytes": 19277,
        "parent_sha256": (
            "acc48c407048082866274126745eafbd106c4e49266feec8cc46c1ad9546f1c"
        ),
        "parent_size_bytes": 19266,
        "relative_path": "src/pure_integer_ai/storage/query_hot_set.py",
    },
    {
        "current_sha256": (
            "fc68dea00d09ef246f7d49899d2d674c142867bf327ecefaa843894930ed7794"
        ),
        "current_size_bytes": 15397,
        "parent_sha256": (
            "13b7b0b7fd330d11717ea1bd6d2b0cb17387ad6361c91393b5b916edae90670a"
        ),
        "parent_size_bytes": 15056,
        "relative_path": "src/pure_integer_ai/storage/segment_cache.py",
    },
]
_V5_TRANSFORMATION = {
    "algorithm": "PUBLIC_GET_TO_QUERY_OWNER_VALIDATED_KEY_LOOKUP",
    "cache_cycle": {
        "delta_per_mille": -268,
        "new_median_ns": 62555800,
        "old_median_ns": 85376300,
        "record_count": 5000,
        "repeats": 11,
    },
    "cached_record_layout_changed": 0,
    "owner_validated_consumer": "QUERY_SEGMENT_HOT_SET_ITER_RANGE",
    "public_get_validation_changed": 0,
    "validated_lookup_sites": 1,
    "performance_claim_scope": "LOCAL_DIRECTION_ONLY",
}
_V5_VERIFICATION = {
    "direct_test_files": 3,
    "direct_tests_passed": 66,
    "full_suite_run": 0,
    "object_model_class_count": 3140,
    "owner_validated_lookup_tests_passed": 1,
    "python311_direct_check_clean": 1,
    "python314_direct_check_clean": 1,
    "source_guards_clean": 1,
}


def _exact(value: Any, keys: set[str], *, where: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{where} 字段不精确")
    return value


def _digest(value: str, *, length: int, where: str) -> str:
    if (not isinstance(value, str) or len(value) != length
            or any(char not in "0123456789abcdef" for char in value)):
        raise ValueError(f"{where} 摘要非法")
    return value


def _relative(value: str, *, where: str) -> str:
    path = PurePosixPath(value)
    if (not isinstance(value, str) or not value or path.is_absolute()
            or ".." in path.parts or "\\" in value):
        raise ValueError(f"{where} 相对路径非法")
    return value


def _identity(root: Path, relative_path: str) -> tuple[int, str]:
    _relative(relative_path, where="文件 identity")
    target = (root / Path(*relative_path.split("/"))).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise ValueError(f"文件缺失: {relative_path}")
    payload = target.read_bytes()
    return len(payload), hashlib.sha256(payload).hexdigest()


def build_performance_successor_receipt(
        repository_root: str | Path,
        ) -> dict[str, Any]:
    """构造固定性能后继证据，不写 readiness 或文件。"""
    root = Path(repository_root).resolve()
    for binding in _SOURCE_BINDINGS:
        current = _identity(root, binding["relative_path"])
        if current != (
                binding["current_size_bytes"], binding["current_sha256"]):
            raise ValueError(
                f"性能后继源码 identity 漂移: {binding['relative_path']}")
    prior_size, prior_sha = _identity(root, PRIOR_RECEIPT_PATH)
    if prior_size < 1 or prior_sha != PRIOR_RECEIPT_SHA256:
        raise ValueError("前序 struct layout receipt identity 漂移")
    return {
        "artifact_kind": ARTIFACT_KIND,
        "artifact_version": ARTIFACT_VERSION,
        "change_commit": CHANGE_COMMIT,
        "format_version": FORMAT_VERSION,
        "parent_commit": PARENT_COMMIT,
        "prior_successor_receipt": {
            "relative_path": PRIOR_RECEIPT_PATH,
            "sha256": PRIOR_RECEIPT_SHA256,
            "status": "STRUCT_LAYOUT_PREDECESSOR",
        },
        "readiness_transition": {
            "LANGUAGE_READINESS_REPUBLISHED": 0,
            "PW00A_STARTED": 0,
        },
        "receipt_relative_path": RECEIPT_PATH,
        "receipt_self_excluded": 1,
        "source_bindings": deepcopy(_SOURCE_BINDINGS),
        "status": "PERFORMANCE_SUCCESSOR_EVIDENCED",
        "transformation": deepcopy(_TRANSFORMATION),
        "verification": deepcopy(_VERIFICATION),
    }


def _validate(value: dict[str, Any]) -> None:
    raw = _exact(value, {
        "artifact_kind", "artifact_version", "change_commit",
        "format_version", "parent_commit", "prior_successor_receipt",
        "readiness_transition", "receipt_relative_path",
        "receipt_self_excluded", "source_bindings", "status",
        "transformation", "verification",
    }, where="performance successor receipt")
    if (raw["artifact_kind"] != ARTIFACT_KIND
            or raw["artifact_version"] != ARTIFACT_VERSION
            or raw["change_commit"] != CHANGE_COMMIT
            or raw["format_version"] != FORMAT_VERSION
            or raw["parent_commit"] != PARENT_COMMIT
            or raw["receipt_relative_path"] != RECEIPT_PATH
            or raw["receipt_self_excluded"] != 1
            or raw["status"] != "PERFORMANCE_SUCCESSOR_EVIDENCED"):
        raise ValueError("performance successor receipt 固定身份漂移")
    _digest(raw["change_commit"], length=40, where="change_commit")
    _digest(raw["parent_commit"], length=40, where="parent_commit")

    if raw["prior_successor_receipt"] != {
            "relative_path": PRIOR_RECEIPT_PATH,
            "sha256": PRIOR_RECEIPT_SHA256,
            "status": "STRUCT_LAYOUT_PREDECESSOR",
    }:
        raise ValueError("前序 struct layout receipt 声明漂移")
    transition = _exact(raw["readiness_transition"], {
        "LANGUAGE_READINESS_REPUBLISHED", "PW00A_STARTED",
    }, where="readiness_transition")
    if transition != {
            "LANGUAGE_READINESS_REPUBLISHED": 0,
            "PW00A_STARTED": 0,
    }:
        raise ValueError("performance successor 不得转移 readiness")
    if raw["source_bindings"] != _SOURCE_BINDINGS:
        raise ValueError("性能后继源码绑定漂移")
    previous = ""
    for binding in raw["source_bindings"]:
        path = binding["relative_path"]
        if not isinstance(path, str) or path <= previous:
            raise ValueError("性能后继源码路径未严格排序")
        _digest(binding["current_sha256"], length=64, where=path)
        _digest(binding["parent_sha256"], length=64, where=path)
        previous = path
    if raw["transformation"] != _TRANSFORMATION:
        raise ValueError("性能 transformation 证据漂移")
    if raw["verification"] != _VERIFICATION:
        raise ValueError("性能 verification 证据漂移")


def read_performance_successor_receipt(
        repository_root: str | Path,
        path: str | Path = RECEIPT_PATH,
        *, verify_current: bool = True,
        ) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    target = Path(path)
    if not target.is_absolute():
        target = root / Path(*str(target).replace("\\", "/").split("/"))
    payload = target.read_bytes()
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise ValueError("performance receipt newline 非法")
    try:
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
    except Exception as error:
        raise ValueError("performance receipt JSON 非 canonical") from error
    if canonical_json_bytes(value) + b"\n" != payload:
        raise ValueError("performance receipt canonical bytes 漂移")
    _validate(value)
    if verify_current:
        for binding in value["source_bindings"]:
            if _identity(root, binding["relative_path"]) != (
                    binding["current_size_bytes"], binding["current_sha256"]):
                raise ValueError(
                    f"性能后继源码当前 identity 漂移: {binding['relative_path']}")
        if _identity(root, PRIOR_RECEIPT_PATH)[1] != PRIOR_RECEIPT_SHA256:
            raise ValueError("前序 struct layout receipt 当前 identity 漂移")
    return value


def publish_performance_successor_receipt(
        repository_root: str | Path,
        *, target: str | Path = RECEIPT_PATH,
        ) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    destination = Path(target)
    if not destination.is_absolute():
        destination = root / Path(*str(destination).replace("\\", "/").split("/"))
    if destination.exists():
        raise ValueError("performance successor receipt 已发布，禁止覆盖")
    value = build_performance_successor_receipt(root)
    payload = canonical_json_bytes(value) + b"\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as stream:
            stream.write(payload)
    except FileExistsError as error:
        raise ValueError("performance successor receipt 已发布，禁止覆盖") from error
    restored = read_performance_successor_receipt(root, destination)
    if restored != value:
        raise ValueError("performance successor receipt 发布回读不一致")
    return restored


def build_v2_performance_successor_receipt(
        repository_root: str | Path,
        ) -> dict[str, Any]:
    """构造已验证长度核的第二版链式性能 receipt。"""
    root = Path(repository_root).resolve()
    for binding in _V2_SOURCE_BINDINGS:
        if _identity(root, binding["relative_path"]) != (
                binding["current_size_bytes"], binding["current_sha256"]):
            raise ValueError(
                f"v2 性能后继源码 identity 漂移: {binding['relative_path']}")
    prior_size, prior_sha = _identity(root, V2_PRIOR_RECEIPT_PATH)
    if prior_size < 1 or prior_sha != V2_PRIOR_RECEIPT_SHA256:
        raise ValueError("前序 performance receipt identity 漂移")
    return {
        "artifact_kind": ARTIFACT_KIND,
        "artifact_version": V2_ARTIFACT_VERSION,
        "change_commit": V2_CHANGE_COMMIT,
        "format_version": FORMAT_VERSION,
        "parent_commit": V2_PARENT_COMMIT,
        "prior_successor_receipt": {
            "relative_path": V2_PRIOR_RECEIPT_PATH,
            "sha256": V2_PRIOR_RECEIPT_SHA256,
            "status": "PERFORMANCE_PREDECESSOR",
        },
        "readiness_transition": {
            "LANGUAGE_READINESS_REPUBLISHED": 0,
            "PW00A_STARTED": 0,
        },
        "receipt_relative_path": V2_RECEIPT_PATH,
        "receipt_self_excluded": 1,
        "source_bindings": deepcopy(_V2_SOURCE_BINDINGS),
        "status": "PERFORMANCE_SUCCESSOR_EVIDENCED",
        "transformation": deepcopy(_V2_TRANSFORMATION),
        "verification": deepcopy(_V2_VERIFICATION),
    }


def _validate_v2(value: dict[str, Any]) -> None:
    raw = _exact(value, {
        "artifact_kind", "artifact_version", "change_commit",
        "format_version", "parent_commit", "prior_successor_receipt",
        "readiness_transition", "receipt_relative_path",
        "receipt_self_excluded", "source_bindings", "status",
        "transformation", "verification",
    }, where="v2 performance successor receipt")
    if (raw["artifact_kind"] != ARTIFACT_KIND
            or raw["artifact_version"] != V2_ARTIFACT_VERSION
            or raw["change_commit"] != V2_CHANGE_COMMIT
            or raw["format_version"] != FORMAT_VERSION
            or raw["parent_commit"] != V2_PARENT_COMMIT
            or raw["receipt_relative_path"] != V2_RECEIPT_PATH
            or raw["receipt_self_excluded"] != 1
            or raw["status"] != "PERFORMANCE_SUCCESSOR_EVIDENCED"):
        raise ValueError("v2 performance successor receipt 固定身份漂移")
    _digest(raw["change_commit"], length=40, where="v2 change_commit")
    _digest(raw["parent_commit"], length=40, where="v2 parent_commit")
    if raw["prior_successor_receipt"] != {
            "relative_path": V2_PRIOR_RECEIPT_PATH,
            "sha256": V2_PRIOR_RECEIPT_SHA256,
            "status": "PERFORMANCE_PREDECESSOR",
    }:
        raise ValueError("前序 performance receipt 声明漂移")
    transition = _exact(raw["readiness_transition"], {
        "LANGUAGE_READINESS_REPUBLISHED", "PW00A_STARTED",
    }, where="v2 readiness_transition")
    if transition != {
            "LANGUAGE_READINESS_REPUBLISHED": 0,
            "PW00A_STARTED": 0,
    }:
        raise ValueError("v2 performance successor 不得转移 readiness")
    if raw["source_bindings"] != _V2_SOURCE_BINDINGS:
        raise ValueError("v2 性能后继源码绑定漂移")
    if raw["transformation"] != _V2_TRANSFORMATION:
        raise ValueError("v2 性能 transformation 证据漂移")
    if raw["verification"] != _V2_VERIFICATION:
        raise ValueError("v2 性能 verification 证据漂移")


def read_v2_performance_successor_receipt(
        repository_root: str | Path,
        path: str | Path = V2_RECEIPT_PATH,
        ) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    target = Path(path)
    if not target.is_absolute():
        target = root / Path(*str(target).replace("\\", "/").split("/"))
    payload = target.read_bytes()
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise ValueError("v2 performance receipt newline 非法")
    try:
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
    except Exception as error:
        raise ValueError("v2 performance receipt JSON 非 canonical") from error
    if canonical_json_bytes(value) + b"\n" != payload:
        raise ValueError("v2 performance receipt canonical bytes 漂移")
    _validate_v2(value)
    for binding in value["source_bindings"]:
        if _identity(root, binding["relative_path"]) != (
                binding["current_size_bytes"], binding["current_sha256"]):
            raise ValueError(
                f"v2 性能后继源码当前 identity 漂移: {binding['relative_path']}")
    if _identity(root, V2_PRIOR_RECEIPT_PATH)[1] != V2_PRIOR_RECEIPT_SHA256:
        raise ValueError("前序 performance receipt 当前 identity 漂移")
    return value


def publish_v2_performance_successor_receipt(
        repository_root: str | Path,
        *, target: str | Path = V2_RECEIPT_PATH,
        ) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    destination = Path(target)
    if not destination.is_absolute():
        destination = root / Path(*str(destination).replace("\\", "/").split("/"))
    if destination.exists():
        raise ValueError("v2 performance successor receipt 已发布，禁止覆盖")
    value = build_v2_performance_successor_receipt(root)
    payload = canonical_json_bytes(value) + b"\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as stream:
            stream.write(payload)
    except FileExistsError as error:
        raise ValueError("v2 performance successor receipt 已发布，禁止覆盖") from error
    restored = read_v2_performance_successor_receipt(root, destination)
    if restored != value:
        raise ValueError("v2 performance successor receipt 发布回读不一致")
    return restored


def build_v3_performance_successor_receipt(
        repository_root: str | Path,
        ) -> dict[str, Any]:
    """构造热缓存 owner 已验证构造路径的链式性能 receipt。"""
    root = Path(repository_root).resolve()
    for binding in _V3_SOURCE_BINDINGS:
        if _identity(root, binding["relative_path"]) != (
                binding["current_size_bytes"], binding["current_sha256"]):
            raise ValueError(
                f"v3 性能后继源码 identity 漂移: {binding['relative_path']}")
    prior_size, prior_sha = _identity(root, V3_PRIOR_RECEIPT_PATH)
    if prior_size < 1 or prior_sha != V3_PRIOR_RECEIPT_SHA256:
        raise ValueError("前序 v2 performance receipt identity 漂移")
    return {
        "artifact_kind": ARTIFACT_KIND,
        "artifact_version": V3_ARTIFACT_VERSION,
        "change_commit": V3_CHANGE_COMMIT,
        "format_version": FORMAT_VERSION,
        "parent_commit": V3_PARENT_COMMIT,
        "prior_successor_receipt": {
            "relative_path": V3_PRIOR_RECEIPT_PATH,
            "sha256": V3_PRIOR_RECEIPT_SHA256,
            "status": "PERFORMANCE_PREDECESSOR",
        },
        "readiness_transition": {
            "LANGUAGE_READINESS_REPUBLISHED": 0,
            "PW00A_STARTED": 0,
        },
        "receipt_relative_path": V3_RECEIPT_PATH,
        "receipt_self_excluded": 1,
        "source_bindings": deepcopy(_V3_SOURCE_BINDINGS),
        "status": "PERFORMANCE_SUCCESSOR_EVIDENCED",
        "transformation": deepcopy(_V3_TRANSFORMATION),
        "verification": deepcopy(_V3_VERIFICATION),
    }


def _validate_v3(value: dict[str, Any]) -> None:
    raw = _exact(value, {
        "artifact_kind", "artifact_version", "change_commit",
        "format_version", "parent_commit", "prior_successor_receipt",
        "readiness_transition", "receipt_relative_path",
        "receipt_self_excluded", "source_bindings", "status",
        "transformation", "verification",
    }, where="v3 performance successor receipt")
    if (raw["artifact_kind"] != ARTIFACT_KIND
            or raw["artifact_version"] != V3_ARTIFACT_VERSION
            or raw["change_commit"] != V3_CHANGE_COMMIT
            or raw["format_version"] != FORMAT_VERSION
            or raw["parent_commit"] != V3_PARENT_COMMIT
            or raw["receipt_relative_path"] != V3_RECEIPT_PATH
            or raw["receipt_self_excluded"] != 1
            or raw["status"] != "PERFORMANCE_SUCCESSOR_EVIDENCED"):
        raise ValueError("v3 performance successor receipt 固定身份漂移")
    _digest(raw["change_commit"], length=40, where="v3 change_commit")
    _digest(raw["parent_commit"], length=40, where="v3 parent_commit")
    if raw["prior_successor_receipt"] != {
            "relative_path": V3_PRIOR_RECEIPT_PATH,
            "sha256": V3_PRIOR_RECEIPT_SHA256,
            "status": "PERFORMANCE_PREDECESSOR",
    }:
        raise ValueError("前序 v2 performance receipt 声明漂移")
    transition = _exact(raw["readiness_transition"], {
        "LANGUAGE_READINESS_REPUBLISHED", "PW00A_STARTED",
    }, where="v3 readiness_transition")
    if transition != {
            "LANGUAGE_READINESS_REPUBLISHED": 0,
            "PW00A_STARTED": 0,
    }:
        raise ValueError("v3 performance successor 不得转移 readiness")
    if raw["source_bindings"] != _V3_SOURCE_BINDINGS:
        raise ValueError("v3 性能后继源码绑定漂移")
    if raw["transformation"] != _V3_TRANSFORMATION:
        raise ValueError("v3 性能 transformation 证据漂移")
    if raw["verification"] != _V3_VERIFICATION:
        raise ValueError("v3 性能 verification 证据漂移")


def read_v3_performance_successor_receipt(
        repository_root: str | Path,
        path: str | Path = V3_RECEIPT_PATH,
        *, verify_current: bool = True,
        ) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    target = Path(path)
    if not target.is_absolute():
        target = root / Path(*str(target).replace("\\", "/").split("/"))
    payload = target.read_bytes()
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise ValueError("v3 performance receipt newline 非法")
    try:
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
    except Exception as error:
        raise ValueError("v3 performance receipt JSON 非 canonical") from error
    if canonical_json_bytes(value) + b"\n" != payload:
        raise ValueError("v3 performance receipt canonical bytes 漂移")
    _validate_v3(value)
    if verify_current:
        for binding in value["source_bindings"]:
            if _identity(root, binding["relative_path"]) != (
                    binding["current_size_bytes"], binding["current_sha256"]):
                raise ValueError(
                    "v3 性能后继源码当前 identity 漂移: "
                    f"{binding['relative_path']}")
        if (_identity(root, V3_PRIOR_RECEIPT_PATH)[1]
                != V3_PRIOR_RECEIPT_SHA256):
            raise ValueError("前序 v2 performance receipt 当前 identity 漂移")
    return value


def publish_v3_performance_successor_receipt(
        repository_root: str | Path,
        *, target: str | Path = V3_RECEIPT_PATH,
        ) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    destination = Path(target)
    if not destination.is_absolute():
        destination = root / Path(*str(destination).replace("\\", "/").split("/"))
    if destination.exists():
        raise ValueError("v3 performance successor receipt 已发布，禁止覆盖")
    value = build_v3_performance_successor_receipt(root)
    payload = canonical_json_bytes(value) + b"\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as stream:
            stream.write(payload)
    except FileExistsError as error:
        raise ValueError("v3 performance successor receipt 已发布，禁止覆盖") from error
    restored = read_v3_performance_successor_receipt(root, destination)
    if restored != value:
        raise ValueError("v3 performance successor receipt 发布回读不一致")
    return restored


def build_v4_performance_successor_receipt(
        repository_root: str | Path,
        ) -> dict[str, Any]:
    """构造 clean/unpinned 热集直接清空的链式性能 receipt。"""
    root = Path(repository_root).resolve()
    for binding in _V4_SOURCE_BINDINGS:
        if _identity(root, binding["relative_path"]) != (
                binding["current_size_bytes"], binding["current_sha256"]):
            raise ValueError(
                f"v4 性能后继源码 identity 漂移: {binding['relative_path']}")
    prior_size, prior_sha = _identity(root, V4_PRIOR_RECEIPT_PATH)
    if prior_size < 1 or prior_sha != V4_PRIOR_RECEIPT_SHA256:
        raise ValueError("前序 v3 performance receipt identity 漂移")
    return {
        "artifact_kind": ARTIFACT_KIND,
        "artifact_version": V4_ARTIFACT_VERSION,
        "change_commit": V4_CHANGE_COMMIT,
        "format_version": FORMAT_VERSION,
        "parent_commit": V4_PARENT_COMMIT,
        "prior_successor_receipt": {
            "relative_path": V4_PRIOR_RECEIPT_PATH,
            "sha256": V4_PRIOR_RECEIPT_SHA256,
            "status": "PERFORMANCE_PREDECESSOR",
        },
        "readiness_transition": {
            "LANGUAGE_READINESS_REPUBLISHED": 0,
            "PW00A_STARTED": 0,
        },
        "receipt_relative_path": V4_RECEIPT_PATH,
        "receipt_self_excluded": 1,
        "source_bindings": deepcopy(_V4_SOURCE_BINDINGS),
        "status": "PERFORMANCE_SUCCESSOR_EVIDENCED",
        "transformation": deepcopy(_V4_TRANSFORMATION),
        "verification": deepcopy(_V4_VERIFICATION),
    }


def _validate_v4(value: dict[str, Any]) -> None:
    raw = _exact(value, {
        "artifact_kind", "artifact_version", "change_commit",
        "format_version", "parent_commit", "prior_successor_receipt",
        "readiness_transition", "receipt_relative_path",
        "receipt_self_excluded", "source_bindings", "status",
        "transformation", "verification",
    }, where="v4 performance successor receipt")
    if (raw["artifact_kind"] != ARTIFACT_KIND
            or raw["artifact_version"] != V4_ARTIFACT_VERSION
            or raw["change_commit"] != V4_CHANGE_COMMIT
            or raw["format_version"] != FORMAT_VERSION
            or raw["parent_commit"] != V4_PARENT_COMMIT
            or raw["receipt_relative_path"] != V4_RECEIPT_PATH
            or raw["receipt_self_excluded"] != 1
            or raw["status"] != "PERFORMANCE_SUCCESSOR_EVIDENCED"):
        raise ValueError("v4 performance successor receipt 固定身份漂移")
    _digest(raw["change_commit"], length=40, where="v4 change_commit")
    _digest(raw["parent_commit"], length=40, where="v4 parent_commit")
    if raw["prior_successor_receipt"] != {
            "relative_path": V4_PRIOR_RECEIPT_PATH,
            "sha256": V4_PRIOR_RECEIPT_SHA256,
            "status": "PERFORMANCE_PREDECESSOR",
    }:
        raise ValueError("前序 v3 performance receipt 声明漂移")
    transition = _exact(raw["readiness_transition"], {
        "LANGUAGE_READINESS_REPUBLISHED", "PW00A_STARTED",
    }, where="v4 readiness_transition")
    if transition != {
            "LANGUAGE_READINESS_REPUBLISHED": 0,
            "PW00A_STARTED": 0,
    }:
        raise ValueError("v4 performance successor 不得转移 readiness")
    if raw["source_bindings"] != _V4_SOURCE_BINDINGS:
        raise ValueError("v4 性能后继源码绑定漂移")
    if raw["transformation"] != _V4_TRANSFORMATION:
        raise ValueError("v4 性能 transformation 证据漂移")
    if raw["verification"] != _V4_VERIFICATION:
        raise ValueError("v4 性能 verification 证据漂移")


def read_v4_performance_successor_receipt(
        repository_root: str | Path,
        path: str | Path = V4_RECEIPT_PATH,
        *, verify_current: bool = True,
        ) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    target = Path(path)
    if not target.is_absolute():
        target = root / Path(*str(target).replace("\\", "/").split("/"))
    payload = target.read_bytes()
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise ValueError("v4 performance receipt newline 非法")
    try:
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
    except Exception as error:
        raise ValueError("v4 performance receipt JSON 非 canonical") from error
    if canonical_json_bytes(value) + b"\n" != payload:
        raise ValueError("v4 performance receipt canonical bytes 漂移")
    _validate_v4(value)
    if verify_current:
        for binding in value["source_bindings"]:
            if _identity(root, binding["relative_path"]) != (
                    binding["current_size_bytes"], binding["current_sha256"]):
                raise ValueError(
                    "v4 性能后继源码当前 identity 漂移: "
                    f"{binding['relative_path']}")
        if (_identity(root, V4_PRIOR_RECEIPT_PATH)[1]
                != V4_PRIOR_RECEIPT_SHA256):
            raise ValueError("前序 v3 performance receipt 当前 identity 漂移")
    return value


def publish_v4_performance_successor_receipt(
        repository_root: str | Path,
        *, target: str | Path = V4_RECEIPT_PATH,
        ) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    destination = Path(target)
    if not destination.is_absolute():
        destination = root / Path(*str(destination).replace("\\", "/").split("/"))
    if destination.exists():
        raise ValueError("v4 performance successor receipt 已发布，禁止覆盖")
    value = build_v4_performance_successor_receipt(root)
    payload = canonical_json_bytes(value) + b"\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as stream:
            stream.write(payload)
    except FileExistsError as error:
        raise ValueError("v4 performance successor receipt 已发布，禁止覆盖") from error
    restored = read_v4_performance_successor_receipt(root, destination)
    if restored != value:
        raise ValueError("v4 performance successor receipt 发布回读不一致")
    return restored


def build_v5_performance_successor_receipt(
        repository_root: str | Path,
        ) -> dict[str, Any]:
    """构造 query owner 已验证键 lookup 的链式性能 receipt。"""
    root = Path(repository_root).resolve()
    for binding in _V5_SOURCE_BINDINGS:
        if _identity(root, binding["relative_path"]) != (
                binding["current_size_bytes"], binding["current_sha256"]):
            raise ValueError(
                f"v5 性能后继源码 identity 漂移: {binding['relative_path']}")
    prior_size, prior_sha = _identity(root, V5_PRIOR_RECEIPT_PATH)
    if prior_size < 1 or prior_sha != V5_PRIOR_RECEIPT_SHA256:
        raise ValueError("前序 v4 performance receipt identity 漂移")
    return {
        "artifact_kind": ARTIFACT_KIND,
        "artifact_version": V5_ARTIFACT_VERSION,
        "change_commit": V5_CHANGE_COMMIT,
        "format_version": FORMAT_VERSION,
        "parent_commit": V5_PARENT_COMMIT,
        "prior_successor_receipt": {
            "relative_path": V5_PRIOR_RECEIPT_PATH,
            "sha256": V5_PRIOR_RECEIPT_SHA256,
            "status": "PERFORMANCE_PREDECESSOR",
        },
        "readiness_transition": {
            "LANGUAGE_READINESS_REPUBLISHED": 0,
            "PW00A_STARTED": 0,
        },
        "receipt_relative_path": V5_RECEIPT_PATH,
        "receipt_self_excluded": 1,
        "source_bindings": deepcopy(_V5_SOURCE_BINDINGS),
        "status": "PERFORMANCE_SUCCESSOR_EVIDENCED",
        "transformation": deepcopy(_V5_TRANSFORMATION),
        "verification": deepcopy(_V5_VERIFICATION),
    }


def _validate_v5(value: dict[str, Any]) -> None:
    raw = _exact(value, {
        "artifact_kind", "artifact_version", "change_commit",
        "format_version", "parent_commit", "prior_successor_receipt",
        "readiness_transition", "receipt_relative_path",
        "receipt_self_excluded", "source_bindings", "status",
        "transformation", "verification",
    }, where="v5 performance successor receipt")
    if (raw["artifact_kind"] != ARTIFACT_KIND
            or raw["artifact_version"] != V5_ARTIFACT_VERSION
            or raw["change_commit"] != V5_CHANGE_COMMIT
            or raw["format_version"] != FORMAT_VERSION
            or raw["parent_commit"] != V5_PARENT_COMMIT
            or raw["receipt_relative_path"] != V5_RECEIPT_PATH
            or raw["receipt_self_excluded"] != 1
            or raw["status"] != "PERFORMANCE_SUCCESSOR_EVIDENCED"):
        raise ValueError("v5 performance successor receipt 固定身份漂移")
    _digest(raw["change_commit"], length=40, where="v5 change_commit")
    _digest(raw["parent_commit"], length=40, where="v5 parent_commit")
    if raw["prior_successor_receipt"] != {
            "relative_path": V5_PRIOR_RECEIPT_PATH,
            "sha256": V5_PRIOR_RECEIPT_SHA256,
            "status": "PERFORMANCE_PREDECESSOR",
    }:
        raise ValueError("前序 v4 performance receipt 声明漂移")
    transition = _exact(raw["readiness_transition"], {
        "LANGUAGE_READINESS_REPUBLISHED", "PW00A_STARTED",
    }, where="v5 readiness_transition")
    if transition != {
            "LANGUAGE_READINESS_REPUBLISHED": 0,
            "PW00A_STARTED": 0,
    }:
        raise ValueError("v5 performance successor 不得转移 readiness")
    if raw["source_bindings"] != _V5_SOURCE_BINDINGS:
        raise ValueError("v5 性能后继源码绑定漂移")
    if raw["transformation"] != _V5_TRANSFORMATION:
        raise ValueError("v5 性能 transformation 证据漂移")
    if raw["verification"] != _V5_VERIFICATION:
        raise ValueError("v5 性能 verification 证据漂移")


def read_v5_performance_successor_receipt(
        repository_root: str | Path,
        path: str | Path = V5_RECEIPT_PATH,
        ) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    target = Path(path)
    if not target.is_absolute():
        target = root / Path(*str(target).replace("\\", "/").split("/"))
    payload = target.read_bytes()
    if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise ValueError("v5 performance receipt newline 非法")
    try:
        value = parse_canonical_json_bytes(payload[:-1], require_object=True)
    except Exception as error:
        raise ValueError("v5 performance receipt JSON 非 canonical") from error
    if canonical_json_bytes(value) + b"\n" != payload:
        raise ValueError("v5 performance receipt canonical bytes 漂移")
    _validate_v5(value)
    for binding in value["source_bindings"]:
        if _identity(root, binding["relative_path"]) != (
                binding["current_size_bytes"], binding["current_sha256"]):
            raise ValueError(
                f"v5 性能后继源码当前 identity 漂移: {binding['relative_path']}")
    if _identity(root, V5_PRIOR_RECEIPT_PATH)[1] != V5_PRIOR_RECEIPT_SHA256:
        raise ValueError("前序 v4 performance receipt 当前 identity 漂移")
    return value


def publish_v5_performance_successor_receipt(
        repository_root: str | Path,
        *, target: str | Path = V5_RECEIPT_PATH,
        ) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    destination = Path(target)
    if not destination.is_absolute():
        destination = root / Path(*str(destination).replace("\\", "/").split("/"))
    if destination.exists():
        raise ValueError("v5 performance successor receipt 已发布，禁止覆盖")
    value = build_v5_performance_successor_receipt(root)
    payload = canonical_json_bytes(value) + b"\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as stream:
            stream.write(payload)
    except FileExistsError as error:
        raise ValueError("v5 performance successor receipt 已发布，禁止覆盖") from error
    restored = read_v5_performance_successor_receipt(root, destination)
    if restored != value:
        raise ValueError("v5 performance receipt 发布回读不一致")
    return restored


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="发布性能 successor receipt。")
    parser.add_argument("--revision", type=int, choices=(1, 2, 3, 4, 5), default=1)
    parser.add_argument("--publish", action="store_true", help="独占发布正式 receipt")
    parser.add_argument("--target", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")
    args = _build_parser().parse_args(argv)
    try:
        if args.revision == 5:
            target = args.target or Path(V5_RECEIPT_PATH)
            if args.publish:
                value = publish_v5_performance_successor_receipt(
                    REPOSITORY_ROOT, target=target)
            else:
                value = build_v5_performance_successor_receipt(REPOSITORY_ROOT)
        elif args.revision == 4:
            target = args.target or Path(V4_RECEIPT_PATH)
            if args.publish:
                value = publish_v4_performance_successor_receipt(
                    REPOSITORY_ROOT, target=target)
            else:
                value = build_v4_performance_successor_receipt(REPOSITORY_ROOT)
        elif args.revision == 3:
            target = args.target or Path(V3_RECEIPT_PATH)
            if args.publish:
                value = publish_v3_performance_successor_receipt(
                    REPOSITORY_ROOT, target=target)
            else:
                value = build_v3_performance_successor_receipt(REPOSITORY_ROOT)
        elif args.revision == 2:
            target = args.target or Path(V2_RECEIPT_PATH)
            if args.publish:
                value = publish_v2_performance_successor_receipt(
                    REPOSITORY_ROOT, target=target)
            else:
                value = build_v2_performance_successor_receipt(REPOSITORY_ROOT)
        else:
            target = args.target or Path(RECEIPT_PATH)
            if args.publish:
                value = publish_performance_successor_receipt(
                    REPOSITORY_ROOT, target=target)
            else:
                value = build_performance_successor_receipt(REPOSITORY_ROOT)
    except (OSError, UnicodeError, ValueError) as error:
        print(f"performance_successor_receipt: ERROR: {error}")
        return 1
    payload = canonical_json_bytes(value) + b"\n"
    print(
        "performance_successor_receipt: ready "
        f"(sha256={hashlib.sha256(payload).hexdigest()})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARTIFACT_KIND", "ARTIFACT_VERSION", "CHANGE_COMMIT", "FORMAT_VERSION",
    "PARENT_COMMIT", "RECEIPT_PATH", "build_performance_successor_receipt",
    "publish_performance_successor_receipt",
    "read_performance_successor_receipt",
    "V2_ARTIFACT_VERSION", "V2_CHANGE_COMMIT", "V2_PARENT_COMMIT",
    "V2_RECEIPT_PATH", "build_v2_performance_successor_receipt",
    "publish_v2_performance_successor_receipt",
    "read_v2_performance_successor_receipt",
    "V3_ARTIFACT_VERSION", "V3_CHANGE_COMMIT", "V3_PARENT_COMMIT",
    "V3_RECEIPT_PATH", "build_v3_performance_successor_receipt",
    "publish_v3_performance_successor_receipt",
    "read_v3_performance_successor_receipt",
    "V4_ARTIFACT_VERSION", "V4_CHANGE_COMMIT", "V4_PARENT_COMMIT",
    "V4_RECEIPT_PATH", "build_v4_performance_successor_receipt",
    "publish_v4_performance_successor_receipt",
    "read_v4_performance_successor_receipt",
    "V5_ARTIFACT_VERSION", "V5_CHANGE_COMMIT", "V5_PARENT_COMMIT",
    "V5_RECEIPT_PATH", "build_v5_performance_successor_receipt",
    "publish_v5_performance_successor_receipt",
    "read_v5_performance_successor_receipt",
]
